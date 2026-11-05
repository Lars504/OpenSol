"""Shared pieces of the EM type definitions (em_fdtd_ez / _hz, em_fdfd_ez / _hz, em_modes): pole vocabulary, settings,
feature params and texts, build callbacks, permittivity rules, physics validation (features: definition_feature_validation)."""
import math
from dataclasses import replace

import numpy as np

from ...core.constants import C0, EPS0, FACE_LABELS, FACES, MU0
from ...core.sources import GaussianPulse, RickerWavelet
from ...core.wavelengths import resolved_wavelengths, wavelength_range
from ...model.definition_helpers import eigen_factorization, study_duration
from ...model.expressions import (FIELD_COORDINATE_NAMES, TIME_NAMES, ExpressionError, evaluate_scalar,
                                  referenced_names)
from ...model.geometry import line_orientation, preview_pixel_centers
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS
from ...solvers.explicit_kernels import stepper_precision
from ...model.schema import (PERMITTIVITY_PROPERTY_NAMES, DomainRequirements, Issue, Param, array_owner,
                             coupling_writes, feature_path, output_field_names, permittivity_arrays, resolve_values,
                             state_field_names)
from . import EMPhysics
from .boundaries import BoundaryConfig
from .material_fit import MAX_LORENTZ_POLES, NK_FIT_DEFAULTS, NK_FIT_MODEL
from .materials import DrudePole, LorentzPole
from .results import (dft_result, far_field_result, flux_result, mode_monitor_result, mode_source_result,
                      probe_result)

EM_FAMILY = 'em'
_BOUNDARY_CHOICES = ('cpml', 'pec', 'pmc', 'periodic', 'symmetric', 'antisymmetric')
_PML_VISIBLE = tuple((f'bc_{face}', ('cpml',)) for face in FACES)
_MIN_CELLS_PER_WAVELENGTH = 10.0
_MIN_CELLS_PER_SKIN_DEPTH = 2.0   # FDFD: cells per lambda0 / (2 pi |Im n|) of a lossy material
_MIN_PLAUSIBLE_WAVELENGTH = 1e-9   # m: a shorter source / solved wavelength is almost surely a unit slip
_MAX_WAVELENGTH_DIAGONALS = 100.0   # likewise a wavelength above this many domain diagonals (nm typed as m)
# dispersion model -> Params of its pole dict (bounds as make_pole checks them: values >= 0, eps_inf > 0)
DISPERSION_POLE_PARAMS = {
    'drude': (
        Param('omega_p', 'float', 1.37e16, label='Plasma frequency', unit='rad/s', min=0.0,
              tip='Angular plasma frequency of the free electrons, in rad/s (not Hz); gold about 1.37e16.\n\n'
                  'In time-domain runs a large value can need smaller cells or a lower \'Time step factor\'; the '
                  'build says so.',
              help='Plasma angular frequency of the Drude current dJ/dt + gamma J = eps0 omega_p^2 E (on every E '
                   'component the polarisation carries: E_z, or E_x and E_y), in rad/s (not Hz); gold ~1.37e16, '
                   'aluminium ~2.24e16 rad/s (library aluminum: 1.94e16, a 1.55 um fit). em_fdtd_ez and em_fdtd_hz both step the current with a leap-frog (J on '
                   'the half time steps), accurate to O((omega dt)^2) but stable only where eps_inf * mu_r >= '
                   'courant_factor^2 + mu_r (omega_p dt)^2 / 4 (square cells; the build checks every cell and names '
                   'the largest stable courant_factor). That bites for eps_inf ~1 metals: omega_p 2.24e16 needs dx '
                   '<= 5.4 nm at courant_factor 0.99 (courant_factor <= 0.967 at 10 nm); otherwise the mesh only has to resolve the '
                   'skin depth of the metal (a few tens of nm for noble metals in the visible). em_fdfd_ez and em_fdfd_hz add the exact susceptibility chi(omega) of every pole at the '
                   'solve wavelength (no time stepping).'),
        Param('gamma', 'float', 1.02e14, label='Damping rate', unit='rad/s', min=0.0,
              tip='Collision rate of the electrons; it sets the loss of the metal.\n\n'
                  'Use optical values: gold about 1e14, silver about 2.7e13. Values measured for static fields give '
                  'too little loss.',
              help='Collision rate of the Drude current in rad/s; it sets the metal loss. Optical (visible / near-IR) '
                   'values: gold ~1.0e14 (the far-IR / DC value ~4e13 gives 2-3x too little near-IR loss), silver '
                   '~2.7e13 rad/s.'),
        Param('eps_inf', 'float', 1.0, label='High-frequency permittivity', unit='1', min=0.0,
              min_exclusive=True, optional=True,
              tip='Relative permittivity from everything except the free electrons; gold about 9.8, silver about '
                  '3.7. Empty means 1.\n\n'
                  'Ignored where the material sets its own relative permittivity.',
              help='Background permittivity of the Drude material, used for a permittivity array only when the '
                   'material sets no permittivity property that owns it (an explicit eps_r or component such as '
                   'eps_zz / eps_xx wins; only the first Drude pole counts); empty = 1. Gold ~9.8, silver ~3.7.'),
    ),
    'lorentz': (
        Param('delta_eps', 'float', 1.0, label='Oscillator strength', unit='1', min=0.0,
              tip='How much this resonance adds to the relative permittivity at frequencies well below it.\n\n'
                  'The static permittivity is the background permittivity plus this value.',
              help='Oscillator strength of d2P/dt2 + gamma dP/dt + omega_0^2 P = eps0 delta_eps omega_0^2 E (on '
                   'every E component the polarisation carries): the static permittivity is the background + '
                   'delta_eps, the background being eps_r or the component that owns the array (as eps_inf).'),
        Param('omega_0', 'float', 3e15, label='Resonance frequency', unit='rad/s', min=0.0,
              tip='Resonance as an angular frequency in rad/s (not Hz).\n\n'
                  '3e15 rad/s corresponds to a vacuum wavelength of 628 nm.',
              help='Resonance angular frequency in rad/s (not Hz); its vacuum wavelength is 2 pi C0 / omega_0 '
                   '(3e15 rad/s = 628 nm).'),
        Param('gamma', 'float', 1e14, label='Damping rate', unit='rad/s', min=0.0,
              tip='Width of the resonance.\n\n'
                  'Smaller values give a sharper resonance with less loss.',
              help='Damping rate of the Lorentz oscillator in rad/s, the full width of the resonance in omega; '
                   'smaller values give a sharper, less lossy resonance.'),
    ),
    # A7: expanded by materials.effective_material_data into drude / lorentz dicts before any pole is built
    NK_FIT_MODEL: (
        Param('n_lorentz', 'int', NK_FIT_DEFAULTS['n_lorentz'], label='Number of resonances', min=0,
              max=MAX_LORENTZ_POLES,
              tip='How many resonance terms the fit may use besides the free-electron term.\n\n'
                  '2 suits most visible or near-infrared bands; more terms fit wider bands but slow down '
                  'time-domain runs and use more memory.',
              help=f'Lorentz poles of the fit (0-{MAX_LORENTZ_POLES}); a Drude pole is added when Re eps < 0 '
                   'somewhere in the fit range (metals). Each pole adds 1-2 float32 arrays on its bounding box in '
                   'the FDTD; poles whose contribution stays below 1e-6 of |eps| are dropped. 2 suits a visible / '
                   'near-IR octave.'),
        Param('wavelength_start', 'float', NK_FIT_DEFAULTS['wavelength_start'], label='Fit from wavelength',
              unit='m', min=0.0, min_exclusive=True,
              tip='Shortest vacuum wavelength the fit has to match, entered in metres (4e-7 for 400 nm).\n\n'
                  'Cover every wavelength the study uses; outside the range the fitted material is extrapolated.',
              help='Shortest vacuum wavelength of the fit range, a plain number (no parameter expressions: the '
                   'fit runs once per material at validation and build). Set the range to the study band: the '
                   'frequency_domain wavelengths, the FDFD / em_modes wavelength, or the FDTD source spectra and '
                   'monitor ranges (validation warns when they leave it). It must lie inside the library table.'),
        Param('wavelength_stop', 'float', NK_FIT_DEFAULTS['wavelength_stop'], label='Fit to wavelength', unit='m',
              min=0.0, min_exclusive=True,
              tip='Longest vacuum wavelength the fit has to match, entered in metres (8e-7 for 800 nm).\n\n'
                  "It must be larger than 'Fit from wavelength'.",
              help='Longest vacuum wavelength of the fit range (> wavelength_start), a plain number. A narrower '
                   'range fits more accurately with fewer poles.'),
    ),
}
_POLE_CLASSES = {'drude': DrudePole, 'lorentz': LorentzPole}
# dispersion model -> (pole class, required keys, optional keys); nk_fit has no class (expanded before)
_POLE_MODELS = {model: (_POLE_CLASSES[model], tuple(param.name for param in params if not param.optional),
                        tuple(param.name for param in params if param.optional))
                for model, params in DISPERSION_POLE_PARAMS.items() if model in _POLE_CLASSES}


# ---- settings ----
CPML_BC_HELP = 'cpml absorbs outgoing waves in pml_cells extra cells added outside the domain.'
PERIODIC_BC_HELP = ('periodic must be set on both faces of the axis and wraps the field with the meshed window as '
                    'period (uniform mesh: the size rounded up to whole cells) and no phase step (k = 0 along that '
                    'axis, e.g. normal incidence); the em_fdtd_*, em_fdfd_* and em_modes types add a Bloch phase with '
                    'bloch_kx / bloch_ky (em_fdfd_nonlinear per input harmonic_channel; em_eme / em_eme_3d: k = 0 '
                    'only).')
MIRROR_BC_HELP = ('Mirror planes, on any face: symmetric = E mirror-even (a PMC mirror plane), antisymmetric = E '
                  'mirror-odd (a PEC mirror plane), on the domain edge itself: one ghost cell is added outside it '
                  '(the width of the edge cell, its material copied from it) to hold the mirror image, so draw the '
                  'structure about the edge (x = 0 for x_lo); heat, potential and the other physics of the model '
                  'have their faces on the same plane. The same plane and parity hold for the ez and hz types, so '
                  'one setting imposes the same symmetry on both. A source half a cell from the plane (in the '
                  'outermost domain cell) is mirrored with it. Results count the unfolded structure: a flux line '
                  'reaching a mirror face and a derived integral whose selection reaches one are doubled per such '
                  'face (result extra mirror_factor; integrands assumed mirror-even, e.g. powers and losses), far '
                  'fields add the image contour (a box touching the plane drops its face there; at most one mirror '
                  'per axis), Results.field(unfold=True) and the GUI show the mirrored field; mode-port powers, '
                  'source powers and the probes stay those of the drawn part. pec and pmc walls of the two '
                  'polarisations are half a cell apart (E_z pec and H_z pmc at the centre of the outermost domain '
                  'cell, E_z pmc and H_z pec on its inner face), so a cavity simulated in both polarisations differs '
                  'in width by one cell.')


def settings_params(bc_help, courant_help, track_absorption_help):
    """Boundary, CPML, Courant and absorption settings of an EM FDTD type (same names, defaults and visibility)."""
    return boundary_settings_params(bc_help) + (
        Param('courant_factor', 'float', 0.99, label='Time step factor', unit='1', min=0.0, min_exclusive=True,
              max=1.0, help=courant_help,
              tip='Time step as a fraction of the largest stable time step.\n\n'
                  'Keep 0.99. Lower it only if the build reports an instability, mostly with metals on a coarse '
                  'mesh; it then names a value that works.'),
        Param('track_absorption', 'bool', False, label='Track absorption', help=track_absorption_help,
              tip="Adds up the light power absorbed in lossy materials and metals, so the 'Absorbed power density' "
                  'result is filled.\n\n'
                  'Switches on by itself when a heating coupling or an expression needs it. When off, that result '
                  'is zero everywhere; tracking costs some speed.'),
    )


_BOUNDARY_TIP = ('What happens to waves at this edge of the domain.\n\n'
                 '- Absorbing layer: waves leave without reflection, as into open space.\n'
                 '- Perfect metal wall: a perfect mirror.\n'
                 '- Perfect magnetic wall: a perfect mirror of the opposite kind.\n'
                 '- Periodic: the domain repeats; set it on both opposite edges. Without a phase step (Bloch wave '
                 'vector; not in the mode solver) it is for normal incidence only.\n'
                 '- Mirror, symmetric: the electric field is mirrored unchanged at this edge, to simulate half of a '
                 'symmetric structure drawn up to the edge.\n'
                 '- Mirror, antisymmetric: the electric field is mirrored with the opposite sign at this edge.')
_BOUNDARY_CHOICE_LABELS = (('cpml', 'Absorbing layer'), ('pec', 'Perfect metal wall'),
                           ('pmc', 'Perfect magnetic wall'), ('periodic', 'Periodic'),
                           ('symmetric', 'Mirror, symmetric'), ('antisymmetric', 'Mirror, antisymmetric'))


def boundary_settings_params(bc_help):
    """bc_<face> and pml_* settings shared by every EM type (equal values merge into one BoundaryConfig)."""
    return (
        *(Param(f'bc_{face}', 'choice', 'cpml', label=FACE_LABELS[face], choices=_BOUNDARY_CHOICES,
                help=bc_help, tip=_BOUNDARY_TIP, choice_labels=_BOUNDARY_CHOICE_LABELS)
          for face in FACES),
        Param('pml_cells', 'int', 15, label='Absorbing layer thickness', unit='cells', min=1,
              visible_if=_PML_VISIBLE,
              tip='Thickness of the layer added outside each absorbing edge to soak up outgoing waves, in grid '
                  'cells. The drawn domain keeps its size.\n\n'
                  '10 to 20 is typical. Use more if waves reflect from the edges.',
              help='CPML cells added outside every cpml face, so the physical domain keeps its size and the grid '
                   'grows. Each PML cell is dx wide on a uniform grid; on a graded one they grow by max_ratio per cell '
                   'from the window edge cell up to the base size (a layer next to a refined edge is thinner). '
                   '10-20 typical; thicker layers absorb grazing and long-wavelength waves better.'),
        Param('pml_order', 'int', 3, label='Absorbing layer grading', min=1, visible_if=_PML_VISIBLE,
              tip='How steeply the absorption rises from the inner to the outer side of the absorbing layer.\n\n'
                  '3 or 4 is typical and rarely needs changing.',
              help='Polynomial order m of the CPML sigma / kappa grading with depth into the layer; 3-4 typical.'),
        Param('pml_sigma_factor', 'float', 1.0, label='Absorbing layer strength', unit='1', min=0.0,
              min_exclusive=True, visible_if=_PML_VISIBLE,
              tip='Absorption strength of the absorbing layer, relative to the best value for vacuum.\n\n'
                  'Keep 1. Where a dense cladding runs into the layer, 1 divided by its refractive index works '
                  'better: about 0.7 for silica, 0.3 for silicon.',
              help='Peak CPML conductivity as a multiple of the vacuum optimum (m + 1) / (150 pi d) S/m, with m = '
                   'pml_order and d the PML cell width; 1 is a good default, 0.5-2 when tuning reflections. A wave '
                   'of index n attenuates n times faster, so where a cladding of index n crosses the PML the '
                   'optimum is ~1/n (e.g. ~0.7 for silica, ~0.3 for silicon).'),
        Param('pml_kappa_max', 'float', 15.0, label='Absorbing layer stretch', unit='1', min=1.0,
              visible_if=_PML_VISIBLE,
              tip='How strongly the absorbing layer stretches space, which helps it absorb fast-decaying and grazing '
                  'waves.\n\n'
                  '1 means no stretch; 5 to 15 is typical.',
              help='Peak real coordinate stretching of the CPML (1 = none); larger values damp evanescent and '
                   'grazing waves better. 5-15 typical.'),
        Param('pml_alpha_max', 'float', 0.0, label='Absorbing layer low-frequency damping', unit='S/m', min=0.0,
              visible_if=_PML_VISIBLE,
              tip='Extra damping of fast-decaying and long-wavelength waves in the absorbing layer.\n\n'
                  '0 switches it off and is the default. At optical wavelengths it only has an effect from about '
                  '1e4 S/m.',
              help='Peak complex-frequency shift alpha of the CPML stretch kappa + sigma / (alpha + j omega eps0), '
                   'largest at the inner edge of the layer; 0 = plain kappa-stretched CPML. It acts only where alpha '
                   'is comparable to omega eps0 (~1.1e4 S/m at 1550 nm, ~0.06 S/m at 1 GHz), damping evanescent '
                   'and low-frequency waves; below ~100 S/m it does nothing in the optical range.'),
    )


# ---- material vocabulary (one table for the 2D EM types; only the help texts differ) ----
FACE_MEAN = 'each face takes the width-weighted mean of its two cells'   # H_z types: material values on the in-plane E faces
_LOSS_AND_MU_ROWS = (   # (name, label, unit, default = lower bound, arrays it writes)
    ('sigma_e', 'Electric conductivity', 'S/m', 0.0, ('sigma_e',)),
    ('mu_r', 'Relative permeability', '1', 1.0, ('mu_r',)),
    ('sigma_m', 'Magnetic conductivity', 'Ohm/m', 0.0, ('sigma_m',)),
)
_EM_MATERIAL_ROWS = {   # isotropic eps_r before its components (a component wins)
    'ez': (('eps_r', 'Relative permittivity', '1', 1.0, ('eps_r',)),
           ('eps_zz', 'Relative permittivity, z (out of plane)', '1', 1.0, ('eps_r',))) + _LOSS_AND_MU_ROWS,
    'hz': (('eps_r', 'Relative permittivity', '1', 1.0, ('eps_xx', 'eps_yy')),
           ('eps_xx', 'Relative permittivity, x', '1', 1.0, ('eps_xx',)),
           ('eps_yy', 'Relative permittivity, y', '1', 1.0, ('eps_yy',))) + _LOSS_AND_MU_ROWS,
    'modes': (('eps_r', 'Relative permittivity', '1', 1.0, ('eps_r', 'eps_xx', 'eps_yy')),
              ('eps_zz', 'Relative permittivity, z (along the guide)', '1', 1.0, ('eps_r',)),
              ('eps_xx', 'Relative permittivity, x', '1', 1.0, ('eps_xx',)),
              ('eps_yy', 'Relative permittivity, y', '1', 1.0, ('eps_yy',))) + _LOSS_AND_MU_ROWS,
}
_MATERIAL_TIPS = {   # GUI tips of the material properties shared by the 2D EM types (em_material_params tips= overrides)
    'eps_r': ('The square of the refractive index: about 2.1 for glass, 12.1 for silicon at 1550 nm. At least 1.\n\n'
              'Describe metals with a dispersion model; for a material with one, this value is the background '
              'permittivity.'),
    'eps_zz': ('Permittivity for the electric field along z, out of the plane: the only one this physics uses.\n\n'
               "When set it replaces 'Relative permittivity'. The library lithium niobate sets it, so change this "
               "value, not 'Relative permittivity'."),
    'eps_xx': ("Permittivity for the electric field along x. When set it replaces 'Relative permittivity' for that "
               'direction.\n\n'
               'Use it for crystals whose optic axis lies along x, for example x-cut lithium niobate: 4.571 at 1550 '
               'nm.'),
    'eps_yy': ("Permittivity for the electric field along y. When set it replaces 'Relative permittivity' for that "
               'direction.\n\n'
               'For a crystal with its optic axis along y, enter the square of its extraordinary refractive index.'),
    'sigma_e': ('Makes the material absorb light; 0 means lossless.\n\n'
                'Use it for weak absorbers. Describe metals with a dispersion model: their static conductivity gives '
                'the wrong loss at optical wavelengths.'),
    'mu_r': ('Relative magnetic permeability, at least 1.\n\n'
             'Keep 1: practically every optical material is non-magnetic.'),
    'sigma_m': ('Magnetic loss for artificial absorbers; keep 0 for real materials.\n\n'
                "Set to 'Electric conductivity' times the square of the wave impedance of the material (377 ohm in "
                'vacuum, divided by the refractive index in a non-magnetic material), it absorbs without reflection '
                'at normal incidence.'),
}
_COMPONENT_HELPS = {   # permittivity component helps shared by the FDTD and FDFD type of a polarisation
    'eps_zz': ('Out-of-plane relative permittivity, the only component E_z sees; overrides eps_r for this physics (a '
               'component wins over eps_r). Anisotropic crystals: eps_r for the in-plane components and eps_zz, e.g. '
               'z-cut lithium niobate eps_r 4.889, eps_zz 4.571 at 1.55 um; an eps_r override then does not change '
               'E_z (set eps_zz).'),
    'eps_xx': (f'x component of the relative permittivity (>= 1), the only one E_x sees without an off-diagonal '
               f'eps_xy ({FACE_MEAN}); overrides '
               'eps_r for E_x. Anisotropic crystals, e.g. x-cut lithium niobate at 1.55 um (optic axis along x): '
               'eps_xx 4.571 = n_e^2 with eps_r 4.889 = n_o^2 for the other components.'),
    'eps_yy': (f'y component of the relative permittivity (>= 1), the only one E_y sees without an off-diagonal '
               f'eps_xy ({FACE_MEAN}); overrides '
               'eps_r for E_y. For a uniaxial crystal with its optic axis along y set eps_yy = n_e^2.'),
}


def em_material_params(polarization, helps, tips=None):
    """Material Params of an EM type ('ez' / 'hz' / 'modes') from the shared table; helps: name -> help text (the
    permittivity component helps default to the shared ones), tips: name -> GUI tip overriding the shared one."""
    rows = _EM_MATERIAL_ROWS[polarization]
    names = [row[0] for row in rows]
    helps = {**{name: text for name, text in _COMPONENT_HELPS.items() if name in names}, **helps}
    if set(helps) != set(names):
        raise ValueError(f'em_material_params({polarization!r}): helps for {sorted(helps)}, need {sorted(names)}')
    tips = {**_MATERIAL_TIPS, **(tips or {})}
    return tuple(Param(name, 'float', default, label=label, unit=unit, min=default, field_dependent=True,
                       state_dependent=True, expands_to=arrays, help=helps[name], tip=tips[name])
                 for name, label, unit, default, arrays in rows)


# ---- feature parameters ----
SPAN_CELLS = 'cells [cell(start), cell(end) + 1), so an end exactly on a cell edge takes one extra cell'
BOX_CELLS = 'cells [cell(lo), cell(hi) + 1) per axis, so an edge exactly on a cell edge takes one extra cell'
POINT_SOURCE_SIGN = ('Sign: the current is added to the field update as given, i.e. it is minus the textbook '
                     'impressed current, so absolute phases (and a source power computed from probe fields) differ '
                     'by pi from textbook formulas such as E_z = -(omega mu0 I / 4) H0^(2)(k r); moduli and fluxes '
                     'do not.')
SHEET_POWER_DRIFT = ('Each recalibration (e.g. a thermo-optic eps change in the line cells) keeps the amplitude, so the '
                     'launched power ~ amplitude^2 n / (2 eta0) follows n of the line cells (~0.5% for dn = 0.01 at '
                     'n = 2.2).')
MAGNETIC_CURRENT_DRIFT = ('A fixed K_m radiates power ~ n^2 of its cell, so it drifts when eps there changes (e.g. '
                          'thermo-optic); a filament current or dipole radiates n-independent power.')
LOCAL_MEDIUM_HELP = ('Each line cell is calibrated to its own medium, so on a line crossing several media (e.g. a '
                     'high-index core and its cladding) the current steps at the interfaces, the steps diffract and '
                     'the field there is not the amplitude (validation warns); keep the line in one medium.')
_DIPOLE_VISIBLE = (('excitation', ('electric_dipole',)),)


def hz_point_source_params(time_params, time_domain):
    """Params of an H_z point_source: time_params (FDTD waveform) after excitation, or (FDFD phase) last."""
    added, cw = ('added to H_z ', 'for cw ') if time_domain else ('', '')
    excitation = Param(
        'excitation', 'choice', 'magnetic_current', label='Excitation', choices=('magnetic_current', 'electric_dipole'),
        choice_labels=(('magnetic_current', 'Magnetic line current'), ('electric_dipole', 'Electric dipole')),
        tip='Kind of point emitter.\n\n'
            '- Magnetic line current: an out-of-plane magnetic current that radiates equally in all directions.\n'
            '- Electric dipole: an in-plane electric current element that radiates most perpendicular to its axis.',
        help=f'magnetic_current: z-directed magnetic line current {added}at the cell centre (radiates the same in '
             'every direction). electric_dipole: in-plane electric current element along orientation, spread over '
             'the four E faces of the cell (radiates most broadside, nothing along its axis).')
    strengths = (
        Param('magnetic_current', 'float', 1.0, label='Magnetic line current', unit='V',
              visible_if=(('excitation', ('magnetic_current',)),),
              tip='Peak strength of the out-of-plane magnetic current. Fields scale linearly with it and do not '
                  'depend on the mesh.\n\n'
                  'Its radiated power follows the permittivity at the point: if the material there changes, for '
                  'example when heated, use an electric dipole instead.',
              help='Peak z-directed magnetic line current K_m in V (the dual of a filament current). The radiated '
                   f'field does not depend on the mesh: {cw}in a homogeneous medium of relative permittivity eps_r '
                   'and wavenumber k, |H_z| = omega eps0 eps_r K_m / 4 |H0^(2)(k r)| (1 V at wavelength 1 um in '
                   'vacuum: |H_z| ~ 1.3e3 A/m at r = 1 um). Fields are linear, so results scale with K_m. '
                   f'{MAGNETIC_CURRENT_DRIFT} {POINT_SOURCE_SIGN}'),
        Param('dipole_moment', 'float', 1e-3, label='Dipole strength', unit='A', visible_if=_DIPOLE_VISIBLE,
              tip='Peak strength of the in-plane electric current element: current times its length, per metre of '
                  'depth.\n\n'
                  'Fields scale linearly with it, so the default 1e-3 is fine.',
              help='Peak in-plane electric dipole strength M in A: the integral of the current density J over the '
                   'cross-section, i.e. A*m of current moment per m of z, along orientation. The radiated field does '
                   f'not depend on the mesh: {cw}in a homogeneous medium of wavenumber k, |H_z| = k M / 4 |H1^(2)(k '
                   'r)| |sin(phi - orientation)| at the angle phi counter-clockwise from +x (1 mA at wavelength 1 um '
                   'in vacuum: |H_z| ~ 5e2 A/m at r = 1 um broadside). Fields are linear, so results scale with M. '
                   f'{POINT_SOURCE_SIGN}'),
        Param('orientation', 'float', 0.0, label='Dipole orientation', unit='deg', visible_if=_DIPOLE_VISIBLE,
              tip='Direction of the dipole current, counter-clockwise from +x: 0 is along x, 90 along y.\n\n'
                  'It radiates most perpendicular to this direction and nothing along it.',
              help='Direction of the dipole current in degrees counter-clockwise from +x: 0 drives only the E_x '
                   'faces, 90 only the E_y faces. It radiates most broadside (perpendicular to it) and nothing along '
                   'it.'),
    )
    if time_domain:
        return (excitation,) + tuple(time_params) + strengths
    return (excitation,) + strengths + tuple(time_params)


_FIELD_SLOT_DOCS = {   # FDTD per-step field slot -> (unit, meaning)
    'ez_source': ('V/m per EM step', 'Added to E_z once per EM step (after the sources) until the next transfer: a '
                  'raw field increment, not a calibrated source. A constant buffer b acts like a current density of '
                  'about -eps0 eps_r b / dt, so its effect scales with 1/dt (mesh, courant_factor); not counted in '
                  'the DFT decimation bound.'),
    'hz_source': ('A/m per EM step', 'Added to H_z once per EM step (after the magnetic sources) until the next '
                  'transfer: a raw field increment, not a calibrated source. A constant buffer b acts like a magnetic '
                  'current density of about -mu0 mu_r b / dt, so its effect scales with 1/dt (mesh, courant_factor); '
                  'not counted in the DFT decimation bound.'),
}


def source_slot_docs(slot_arrays, field_slot=None):
    """source_slots of an EM type: the FDTD field slot (if any), then one permittivity slot per slot_arrays entry."""
    docs = {} if field_slot is None else {field_slot: _FIELD_SLOT_DOCS[field_slot]}
    for slot, array in slot_arrays.items():
        docs[slot] = ('1', f'Relative permittivity {array}: each transfer replaces the whole array, PML ring '
                           'included; index couplings (heat_em_thermo_optic) add their dn: eps = (n0 + sum dn)^2.')
    return docs


# GUI tips of feature Params that the FDTD and FDFD files define separately but word alike
EZ_CURRENT_TIP = ('Peak current through the point, flowing out of the plane. Fields scale linearly with it and do not '
                  'depend on the mesh.\n\n'
                  'The default 1e-3 is fine: ratios such as transmission do not depend on it.')
_LINE_MATERIAL_TIP = 'Keep the line inside one material'
LINE_AMPLITUDE_TIPS = {   # polarisation -> line source amplitude tip
    'ez': ('Peak out-of-plane electric field sent to each side of the line. Fields scale linearly with it; 1 is '
           f'typical.\n\n{_LINE_MATERIAL_TIP}: where it crosses an interface the launched wave is distorted.'),
    'hz': ('Peak in-plane electric field along the line, sent to each side of it. Fields scale linearly with it; 1 is '
           f'typical.\n\n{_LINE_MATERIAL_TIP}: where it crosses an interface the launched wave is distorted.'),
}
BEAM_AMPLITUDE_TIPS = {   # polarisation -> gaussian beam amplitude tip
    'ez': ('Peak out-of-plane electric field at the beam centre, sent to each side of the line. Fields scale linearly '
           f'with it; 1 is typical.\n\n{_LINE_MATERIAL_TIP}.'),
    'hz': ('Peak in-plane electric field along the line at the beam centre, sent to each side of it. Fields scale '
           f'linearly with it; 1 is typical.\n\n{_LINE_MATERIAL_TIP}.'),
}
_SHEET_FIELDS = {'ez': 'E_z', 'hz': 'tangential E (E_x for a horizontal line, E_y for a vertical one)'}
_FACE_MEDIUM_HELP = ('Its medium is the face mean of the line cell and the cell past its hi face, so on an interface '
                     'sqrt(mean eps) is not the mean index (Si / air: ~14% off).')
_SHEET_CALIBRATIONS = {   # (solver, polarisation) -> how the sheet amplitude is calibrated
    ('fdtd', 'ez'): 'the calibration uses |eta| of the line cells at the centre frequency (sigma_e and dispersion '
                    'included), so other frequencies of a pulse in a dispersive medium deviate',
    ('fdtd', 'hz'): 'the calibration uses |eta| of the line faces at the centre frequency (sigma_e and dispersion '
                    'included), so other frequencies of a pulse in a dispersive medium deviate',
    ('fdfd', 'ez'): 'the calibration uses the complex eps_r (sigma_e and dispersion included) and mu of the line cells '
                    'at the solved wavelength, so it is exact in any homogeneous medium',
    ('fdfd', 'hz'): 'the calibration uses the complex face permittivity (the mean of the line cell and the cell past '
                    'its hi face, sigma_e and dispersion included) and mu of the line cells at the solved wavelength, '
                    'so it is exact in any homogeneous medium',
}


def sheet_amplitude_param(polarization, solver, beam):
    """'amplitude' Param of a line_source (beam False) or gaussian_beam (beam True) of em_<solver>_<polarization>
    (solver 'fdtd' | 'fdfd'): the sheet field radiated to each side, its calibration and drift."""
    field = _SHEET_FIELDS[polarization]
    calibration = _SHEET_CALIBRATIONS[solver, polarization]
    medium_help = f'{LOCAL_MEDIUM_HELP} {_FACE_MEDIUM_HELP}' if polarization == 'hz' else LOCAL_MEDIUM_HELP
    if beam:
        radiated = f'Peak {field} at the beam centre on the waist line, radiated to each side in the local medium'
    else:
        plane_wave = 'a cw plane wave' if solver == 'fdtd' else 'a plane wave'
        radiated = (f'{field[0].upper()}{field[1:]} radiated to each side of the line in the local medium '
                    f'({plane_wave} of this amplitude when the line spans a periodic axis)')
    return Param('amplitude', 'float', 1.0, label='Amplitude', unit='V/m',
                 help=f'{radiated}; {calibration}. Fields are linear, so results scale with it. {medium_help} '
                      f'{SHEET_POWER_DRIFT} Typical 1 V/m.',
                 tip=(BEAM_AMPLITUDE_TIPS if beam else LINE_AMPLITUDE_TIPS)[polarization])


MODE_POWER_TIP =('Power carried by the launched mode, per metre of depth out of the plane.\n\n'
                  'Fields grow with its square root, so 1 is the usual choice.')
MODE_DIRECTION_CHOICE_LABELS = (('positive', 'Positive (+x or +y)'), ('negative', 'Negative (-x or -y)'))
PROBE_COMPONENT_CHOICE_LABELS = {   # polarisation -> field probe component drop-down texts
    'ez': (('ez', 'Electric field, out of plane (E_z)'), ('hx', 'Magnetic field along x (H_x)'),
           ('hy', 'Magnetic field along y (H_y)')),
    'hz': (('hz', 'Magnetic field, out of plane (H_z)'), ('ex', 'Electric field along x (E_x)'),
           ('ey', 'Electric field along y (E_y)')),
}
EZ_PROBE_COMPONENT_TIP = ('Field recorded at the point; electric fields in V/m, magnetic fields in A/m.\n\n'
                          '- Electric field, out of plane (E_z): the field this physics is named after.\n'
                          '- Magnetic field along x (H_x): in-plane magnetic field.\n'
                          '- Magnetic field along y (H_y): in-plane magnetic field.')
HZ_PROBE_COMPONENT_TIP = ('Field recorded at the point; magnetic fields in A/m, electric fields in V/m.\n\n'
                          '- Magnetic field, out of plane (H_z): the field this physics is named after.\n'
                          '- Electric field along x (E_x): in-plane electric field.\n'
                          '- Electric field along y (E_y): in-plane electric field.')
_GAUSSIAN_PULSE_TIP_LINE = '- Gaussian pulse: a short burst covering a band of wavelengths, for spectra.'
_CW_TIP_LINE = '- Continuous wave: one wavelength, switched on smoothly and steady after about 15 periods.'
_GAUSSIAN_WAVEFORM_HELP = ('gaussian: sin carrier of f0 = C0 / wavelength under exp(-((t - t0) / tau)^2), tau = 1 / '
                           '(pi * bandwidth * f0), centred at t0 = 4.5 tau, so it lasts about 9 / (pi * bandwidth * '
                           'f0) (550 nm, bandwidth 0.5: ~10 fs). The amplitude / current is the envelope peak: the '
                           'carrier is 0 at t0, so the source peaks lower (~0.87 at bandwidth 0.5, ~0.66 at 1).')
_CW_WAVEFORM_HELP = 'cw: sin(2 pi f0 t) ramped in by 1 - exp(-t / (3 T)), T = 1 / f0, steady after ~15 periods.'


def source_params(amplitude_help, amplitude_tip):
    """Waveform, wavelength, bandwidth and amplitude params shared by the field-amplitude sources."""
    return waveform_params() + (
        Param('amplitude', 'float', 1.0, label='Amplitude', unit='V/m', help=amplitude_help, tip=amplitude_tip),)


def waveform_params():
    """Waveform choice, centre / peak vacuum wavelength and relative bandwidth (gaussian only)."""
    return (
        Param('waveform', 'choice', 'gaussian', label='Waveform', choices=('gaussian', 'cw', 'ricker'),
              choice_labels=(('gaussian', 'Gaussian pulse'), ('cw', 'Continuous wave'), ('ricker', 'Ricker wavelet')),
              tip=f'Time shape of the source.\n\n{_GAUSSIAN_PULSE_TIP_LINE}\n{_CW_TIP_LINE}\n'
                  '- Ricker wavelet: a single, very broadband pulse.',
              help=f'{_GAUSSIAN_WAVEFORM_HELP} {_CW_WAVEFORM_HELP} ricker: Ricker wavelet whose spectrum peaks at '
                   'f0, delayed by 4 / f0 (broadband, up to ~2.5 f0: 3 % of the spectral peak there, 1 % at 2.76 '
                   'f0).'),
        Param('wavelength', 'float', 550e-9, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
              tip='Vacuum wavelength at the centre of the source spectrum, entered in metres (5.5e-7 for 550 nm).\n\n'
                  'The mesh needs at least 10, better 20, cells per wavelength inside each material.',
              help='Vacuum wavelength lambda0 = C0 / f0 of the carrier (ricker: of the spectral peak); in a medium '
                   'of index n the wave is lambda0 / n long. In m, not nm: typical 400e-9 to 1600e-9 m (validation '
                   'warns below 1e-9 m or above 100 domain diagonals); the mesh needs >= 10 (better 20) cells per '
                   'lambda0 / n at the highest source frequency.'),
        Param('bandwidth', 'float', 0.5, label='Relative bandwidth', unit='1', min=0.0, min_exclusive=True,
              visible_if=(('waveform', ('gaussian',)),),
              tip='Spectral width of the pulse as a fraction of the centre frequency: 0.5 at 550 nm covers about 440 '
                  'to 730 nm.\n\n'
                  '0.2 to 1 is typical. A wider band gives a shorter pulse but needs a finer mesh.',
              help='Spectral width / f0 of the gaussian pulse, amplitude spectrum exp(-((f - f0) / (bandwidth * '
                   'f0))^2): 0.78 of the peak at f0 * (1 +- bandwidth / 2), 1/e at f0 * (1 +- bandwidth) (power FWHM '
                   'about 1.18 * bandwidth * f0); the mesh check and auto mesh resolve f0 * (1 + bandwidth). 0.2-1 '
                   'typical; a wider band gives a shorter pulse but needs a finer mesh.'),
    )


def beam_params():
    """Waist and centre params of a gaussian beam along its line."""
    return (
        Param('waist', 'float', 1e-6, label='Waist radius', unit='m', min=0.0, min_exclusive=True,
              tip='Half-width of the beam at its narrowest point, which lies on the line; there the field falls to '
                  '37 % of its peak.\n\n'
                  '1 to 5 wavelengths is typical. Make the line several waists long so the beam is not cut off.',
              help='1/e half-width of the field profile exp(-((s - center) / waist)^2) along the line; the waist '
                   'lies on the line and the beam diverges from there. Typical 1-5 wavelengths; keep the line '
                   'several waists long so the profile is not truncated (validation warns above 1e-2 of the peak at '
                   'an end that stops inside the domain; an end on a cpml edge continues through the PML). The '
                   'amplitude at the waist is paraxial: on '
                   'axis the field is about amplitude (1 + 1/(k waist)^2) for E_z and (1 - 1/(k waist)^2) for H_z '
                   'types, k = 2 pi n / lambda0 (+-2.5% at waist = lambda0 / n).'),
        Param('center', 'float', None, label='Beam centre', unit='m', optional=True,
              tip='Position of the beam axis along the line: x for a horizontal line, y for a vertical one.\n\n'
                  'Leave empty for the middle of the line. It must lie on the line.',
              help='Position of the beam axis along the line: x for a horizontal line, y for a vertical one; empty '
                   '= the midpoint of the line. It must lie on the line (validation warns otherwise): off it only the '
                   'tail of the profile is launched.'),
    )


def spectral_params(n_wavelengths, n_wavelengths_help, n_wavelengths_tip):
    """Monitor wavelengths linspace(wavelength_start, wavelength_stop, n_wavelengths) (1 -> the centre);
    n_wavelengths_help / _tip (the feature's cost and typical count) end the n_wavelengths help / tip."""
    return (
        Param('wavelength_start', 'float', 400e-9, label='Shortest wavelength', unit='m', min=0.0,
              min_exclusive=True,
              tip='Shortest vacuum wavelength the monitor records, entered in metres (4e-7 for 400 nm).\n\n'
                  'Keep the range inside the source spectrum; outside it the results get noisy.',
              help='Shortest vacuum wavelength of the monitor spectrum. Keep the range inside the source spectrum '
                   '(gaussian: lambda0 / (1 + bandwidth / 2) to lambda0 / (1 - bandwidth / 2), e.g. 440e-9 to '
                   '733e-9 m for 550 nm and bandwidth 0.5); outside it the source spectrum falls off and normalised '
                   'results get noisy. Validation warns for wavelengths where every enabled source spectrum is below '
                   '1e-2 of its peak (gaussian beyond f0 (1 +- 2.15 bandwidth), ricker outside 0.06-2.76 f0, cw '
                   'anywhere but within 1 / duration of f0).'),
        Param('wavelength_stop', 'float', 800e-9, label='Longest wavelength', unit='m', min=0.0,
              min_exclusive=True,
              tip='Longest vacuum wavelength the monitor records, entered in metres (8e-7 for 800 nm).\n\n'
                  "It must be larger than 'Shortest wavelength' unless 'Number of wavelengths' is 1.",
              help='Longest vacuum wavelength of the monitor spectrum; it must exceed wavelength_start (equal only '
                   'with n_wavelengths = 1).'),
        Param('n_wavelengths', 'int', n_wavelengths, label='Number of wavelengths', min=1,
              tip="Number of wavelengths, evenly spaced between 'Shortest wavelength' and 'Longest wavelength'; 1 "
                  f'gives the centre of the range.\n\n{n_wavelengths_tip}',
              help='Wavelengths evenly spaced in wavelength (not frequency) from wavelength_start to '
                   f'wavelength_stop; 1 = the centre of the range. {n_wavelengths_help}'),
    )


N_ANGLES_PARAM = Param('n_angles', 'int', 360, label='Number of angles', min=1,
                       tip='Number of directions, evenly spaced over the full circle counter-clockwise from +x.\n\n'
                           '360 gives 1-degree steps.',
                       help='Far-field angles evenly spaced over 360 degrees, counter-clockwise from +x (0 = +x, pi/2 '
                            '= +y); 360 gives 1-degree steps.')


def far_field_params(n_wavelengths_help, n_wavelengths_tip):
    """Far-field spectrum params (21 wavelengths by default) plus the number of angles."""
    return spectral_params(21, n_wavelengths_help, n_wavelengths_tip) + (N_ANGLES_PARAM,)


FLUX_PARAMS = spectral_params(100, '50-200 give a smooth spectrum; the work per time step grows with the count.',
                              '50 to 200 gives a smooth spectrum; more wavelengths slow down every time step.')

MODE_NUMBER_HELP = ('1-based index of the mode in descending order of Re(n_eff) among the propagating modes (COMSOL / '
                    'Lumerical numbering; the ranking key is Re(k^2) of the transverse eigenvalue, so evanescent '
                    'modes, Re(k^2) < 0, come after every propagating one whatever their Re(n_eff)): 1 = the '
                    'fundamental, most strongly guided mode, 2 the next, ...; both parities count, so in a symmetric '
                    'slab mode 1 is even and mode 2 odd; modes localised in the PML of a line that continues '
                    'through it are discarded and not counted. The build fails when the line has fewer modes and '
                    'warns when the selected one is not guided (Re(n_eff) not above the index of a span-end cell at a '
                    'cpml or metal end: a cladding or radiation mode) or carries no forward power (an evanescent or '
                    'backward mode).')


def mode_number_param(selection_help, tip=''):
    """mode_number Param of a mode source / monitor; selection_help follows MODE_NUMBER_HELP; tip = the GUI tip."""
    return Param('mode_number', 'int', 1, label='Mode number', min=1, help=f'{MODE_NUMBER_HELP} {selection_help}',
                 tip=tip)


# first tip paragraph of every mode_number Param: .format(verb='launch' / 'measure')
MODE_NUMBER_TIP_START = ('Which guided mode to {verb}: 1 is the fundamental, most strongly guided mode, 2 the next, '
                         'and so on.')
MODE_SOURCE_NUMBER_PARAM = mode_number_param(
    'The source result table lists the first max(mode_number + 2, 4) modes.',
    tip=(MODE_NUMBER_TIP_START.format(verb='launch')
         + '\n\nExtend the line well into the cladding on both sides of the guide.'))
MODE_SOURCE_PARAMS = (
    MODE_SOURCE_NUMBER_PARAM,
    Param('direction', 'choice', 'positive', label='Direction', choices=('positive', 'negative'),
          choice_labels=MODE_DIRECTION_CHOICE_LABELS,
          tip='Direction in which the mode travels away from the line.\n\n'
              '- Positive (+x or +y): toward +x from a vertical line, toward +y from a horizontal one.\n'
              '- Negative (-x or -y): toward -x or -y.\n\n'
              'Practically nothing is launched the other way.',
          help='positive launches the mode toward +x for a vertical line (x = const) and toward +y for a horizontal '
               'one, negative toward -x / -y. The total field starts at the line cells and continues downstream; '
               'nothing is launched upstream at the source wavelength, and the backward leakage grows toward the '
               'band edges of a gaussian pulse.'),
    Param('waveform', 'choice', 'gaussian', label='Waveform', choices=('gaussian', 'cw'),
          choice_labels=(('gaussian', 'Gaussian pulse'), ('cw', 'Continuous wave')),
          tip=f'Time shape of the source.\n\n{_GAUSSIAN_PULSE_TIP_LINE}\n{_CW_TIP_LINE}',
          help=f'{_GAUSSIAN_WAVEFORM_HELP} {_CW_WAVEFORM_HELP} The complex mode profile is driven by the carrier '
               'and its quadrature (cos) partner, so ricker is not offered.'),
    Param('wavelength', 'float', 550e-9, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          tip='Vacuum wavelength of the source, entered in metres (1.55e-6 for 1550 nm).\n\n'
              'The mode shape is computed at this wavelength only, so the launch is exact there.',
          help='Vacuum wavelength lambda0 = C0 / f0 of the carrier and the only wavelength at which the mode is '
               'solved (profile, n_eff, impedance): the launch is exact and one-sided there. Typical 1.31e-6 or '
               '1.55e-6 m in silicon photonics; the mesh needs >= 10 (better 20) cells per lambda0 / n at the '
               'highest source frequency.'),
    Param('bandwidth', 'float', 0.3, label='Relative bandwidth', unit='1', min=0.0, min_exclusive=True,
          visible_if=(('waveform', ('gaussian',)),),
          tip='Spectral width of the pulse as a fraction of the centre frequency.\n\n'
              '0.1 to 0.3 is typical. A wider band leaks more power backwards and into radiation at its edges.',
          help='Spectral width / f0 of the gaussian pulse, amplitude spectrum exp(-((f - f0) / (bandwidth * f0))^2): '
               '0.78 of the peak at f0 * (1 +- bandwidth / 2), 1/e at f0 * (1 +- bandwidth); the mesh check and auto '
               'mesh resolve f0 * (1 + bandwidth). The mode is solved at f0 only, so the backward leakage and the launch into radiation grow toward '
               'the band edges as the guide disperses; 0.1-0.3 typical for waveguides (a narrower band lasts longer, '
               'about 9 / (pi * bandwidth * f0)). Above 0.6 the build warns: the cos quadrature partner (not an exact '
               'Hilbert partner) then biases fraction_* / phase_* toward the lower band edge and adds a DC part.'),
    Param('power', 'float', 1.0, label='Mode power', unit='W/m', min=0.0, min_exclusive=True, tip=MODE_POWER_TIP,
          help='Power per unit z length carried by the launched mode, 1/2 Re of the integral of (E x H*).n over the '
               'line (the 2D port input power of COMSOL): cw = the time-averaged power once the ramp has settled, '
               'gaussian = the cycle-averaged power at the envelope peak. Fields scale with sqrt(power) and are '
               'linear, so 1 W/m is the usual choice (a peak |E| of order 1e4 V/m in a 0.3 um silicon slab).'),
)
MODE_MONITOR_PARAMS = (
    mode_number_param('It picks the mode at the middle monitor wavelength; at the other wavelengths the same mode is '
                      'followed by its overlap with the neighbouring wavelength, so it keeps its identity where two '
                      'modes cross in n_eff (a build warning lists wavelengths where that overlap stays below 0.9 or '
                      'the followed mode is no longer guided, e.g. past a cutoff).',
                      tip=(MODE_NUMBER_TIP_START.format(verb='measure')
                           + '\n\nPut the line across a straight part of the guide, extending well into the '
                             'cladding.')),
) + spectral_params(21, 'Each wavelength costs one 1D mode solve at build and two complex spectra per line cell '
                        '(little GPU memory) plus work per time step; 11-41 typical.',
                    '11 to 41 is typical; each adds a mode calculation at the start of the run.')

# descriptions of the mode features (definition_fdtd adds the fields and span-end conditions)
MODE_SOURCE_TEXT = ('The mode is injected as a one-sided total-field / scattered-field line (discrete equivalent '
                    'currents): total field from the line cells downstream, while upstream only what comes back '
                    '(reflections) remains. The launch is exact and one-sided at the source wavelength; toward the '
                    'band edges of a gaussian pulse some power leaks upstream or into radiation because the profile '
                    'and impedance of the mode change with wavelength, so keep the bandwidth modest in dispersive '
                    'guides. Mode monitors normalise by it when it is the only enabled mode source of the physics. Its '
                    'interface face holds scattered field, so a flux or mode monitor on the line of a negative '
                    'source (total-field line cells, interface on their hi face) mixes both (validation warns).')
MODE_SOLVE_TEXT = ('The 1D eigenmodes along the line are FDTD-consistent (kernel cell widths, numerical frequency and '
                   'dispersion of the Yee scheme, PML stretch, complex permittivity and permeability of the line '
                   'cells at that frequency with poles, sigma_e and sigma_m) and are solved at build (finalize) on '
                   'the final materials; during the run they are re-solved, following the selected mode by overlap, '
                   'whenever the line materials drifted by more than 1e-3 relative since the last solve '
                   '(thermo-optic or Pockels coupling, state-dependent permittivity; one host transfer of the line '
                   'per material update). A mode monitor then decomposes its whole-run spectra with the latest modes '
                   '(the run log notes it), which fits only light that passed after the medium settled. Put the line '
                   'across a straight guide with the same cross-section a few cells on either side and a locally '
                   'uniform mesh (validation refuses a line whose face falls on the outer grid wall and warns when a '
                   'wall boundary pins the line cell or its face every step, when its face lies inside a CPML layer '
                   '(line in the outermost window cell) or when the cell widths along the normal change by more than '
                   '1% around the line).')
# polarisation -> (span-end conditions at wall faces, the metal condition of an end inside the domain)
_MODE_END_CONDITIONS = {
    'ez': ('an end on a pec domain edge takes E_z = 0, one on a pmc or symmetric edge dE_z/dn = 0 and one on an '
           'antisymmetric edge E_z odd about the plane', 'E_z = 0 in the next cell'),
    'hz': ('an end on a pec or antisymmetric domain edge takes tangential E = 0 on the inner face of the outermost '
           'cell (dH_z/dn = 0), one on a pmc edge H_z = 0 and one on a symmetric edge H_z odd',
           'E along the propagation direction = 0 on the face beyond the end, so dH_z/dn = 0'),
}
_MODE_FACE_PERMITTIVITY = {'fdtd': 'the permittivity the engine uses', 'fdfd': 'the permittivity of the operator'}


def mode_end_text(polarization, solver):
    """Span-end conditions of a mode_source / mode_monitor line of em_<solver>_<polarization> (mode_ports.PortLine,
    alike for FDTD and FDFD); hz adds the face-permittivity note."""
    walls, metal = _MODE_END_CONDITIONS[polarization]
    text = ('Span ends, like COMSOL port boundaries: an end on a cpml domain edge continues the solve through the PML '
            f'cells (open: radiation is absorbed); {walls}, as those walls do; an end inside the domain, or a partial '
            f'span of a periodic axis, is metal ({metal}), so extend the line well into the cladding (the build warns '
            'when the mode still exceeds 1% of its peak at a metal end); a span over a whole periodic axis is '
            'periodic.')
    if polarization == 'hz':
        text += (f' The in-plane E faces use {_MODE_FACE_PERMITTIVITY[solver]} ({FACE_MEAN}), so a guide edge on a '
                 'face line takes the mean of core and cladding there.')
    return text


MODE_TABLE_ROWS =('one row per mode for the first max(mode_number + 2, 4) modes: mode_number, n_eff (real part), '
                   'n_eff_imag (< 0 = loss in the exp(+j omega t) convention), loss_db_per_cm, n_group, edge_field '
                   '(largest |field| at the two ends of the solved line / its peak; above 1e-2 at a metal end the line '
                   'cuts the mode) and selected (True for the launched mode)')   # mode-source result table rows
MODE_SOURCE_RESULT_TEXT = (f"Result 'table' at the source wavelength, {MODE_TABLE_ROWS}; extras wavelength, power, "
                           'direction, mode_number and tracked_mode_number (the rank of the launched mode in the '
                           'latest solve: mode_number until a re-solve during the run finds it at another rank; the '
                           'selected row follows it).')
PER_CW_TEXT = ('The per_cw values exist only when every source feature of the physics shares one waveform (type, '
               'wavelength, bandwidth; coupling drives are not counted); S(omega) = the dt-weighted spectrum of that '
               'waveform over the steps run; NaN where |S|^2 is below 1e-6 of its value at the source wavelength (cw: '
               'also outside |f - f0| < 1 / (steps run x dt)). They equal the em_fdfd_* values at the same source '
               'amplitudes, exactly only at the source wavelength (sources are calibrated there).')
MODE_MONITOR_RESULT_TEXT = ("Result 'xy' versus wavelength: power_positive / power_negative = 2 |a|^2 in a.u. "
                            '(dt-weighted spectral power per unit z length in the flux_monitor convention, so '
                            'comparable with its power); power_per_cw_positive / power_per_cw_negative = |a|^2 / '
                            '|S|^2 in W/m (the cw mode power, as em_fdfd_* power_*); with exactly one enabled '
                            'mode_source in the physics also '
                            'fraction_positive / fraction_negative (unit 1) = mode power / source power at that '
                            'wavelength (|S|^2: transmission into this mode downstream of the source, reflection '
                            'upstream; NaN where |S(omega)|^2 of the source is below 1e-6 of its value at the source '
                            'wavelength, and for a cw source outside |f - f0| < 1 / (steps run x dt)) and '
                            'phase_positive / '
                            'phase_negative (rad) = the phase of that S-parameter: the mode amplitude relative to the '
                            'amplitude the source launches on its own line, not de-embedded (each mode profile is '
                            'phased with its largest field sample real and positive, the lowest-coordinate one of '
                            'mirror-equal peaks); extras frequency, wavelength, '
                            'n_eff (real part), n_eff_imag (per wavelength), mode_number, tracked_mode_number (rank of the '
                            'followed mode at the middle wavelength in the latest solve) and normalized (True when the '
                            'fractions are present). The source spectrum covers the steps run, so run until the pulse '
                            'has passed every monitor; the fractions inherit the off-centre leakage of the source, '
                            f'so they are most accurate near its wavelength. {PER_CW_TEXT}')


# ---- build: domain requirements, create ----
def boundary_config_from_settings(settings_values):
    """BoundaryConfig of the four bc_<face> settings with the pml_* CPML parameters."""
    boundary_config = BoundaryConfig(**{face: settings_values[f'bc_{face}'] for face in FACES})
    boundary_config.cpml_npml = int(settings_values['pml_cells'])
    boundary_config.cpml_order = int(settings_values['pml_order'])
    boundary_config.cpml_sigma_factor = float(settings_values['pml_sigma_factor'])
    boundary_config.cpml_kappa_max = float(settings_values['pml_kappa_max'])
    boundary_config.cpml_alpha_max = float(settings_values['pml_alpha_max'])
    return boundary_config


def domain_requirements(settings_values):
    """BoundaryConfig of the four faces + CPML parameters, and the Courant factor."""
    return DomainRequirements(boundary_config=boundary_config_from_settings(settings_values),
                              courant_factor=float(settings_values['courant_factor']))


def quasi_static_domain_requirements(settings_values):
    """The boundary configuration only (FDFD, em_modes): nothing steps in time, so no Courant factor."""
    return DomainRequirements(boundary_config=boundary_config_from_settings(settings_values))


def steps_in_time(settings_values):
    return True


# ---- solved wavelengths of the quasi-static EM types (FDFD, em_modes): the setting, or a frequency_domain list ----
FREQUENCY_DOMAIN_STUDY = 'frequency_domain'   # study whose wavelengths replace the wavelength setting
DEFAULT_WAVELENGTH = 1.55e-6                  # [m] wavelength setting default (no unused-setting info at it)


def _study_wavelengths(spec, parameters, registry):
    """Solved wavelengths [m] of the model's frequency_domain study in solve order; ValueError on bad settings."""
    settings = spec.study.settings if isinstance(spec.study.settings, dict) else {}
    return resolved_wavelengths(resolve_values(registry.study_type(FREQUENCY_DOMAIN_STUDY).params, settings,
                                               parameters))


def first_solved_wavelength(ctx, settings_values):
    """Vacuum wavelength [m] of the first solve: a frequency_domain study's first wavelength, else (also for an empty
    range) the wavelength setting."""
    if ctx.study_type != FREQUENCY_DOMAIN_STUDY:
        return settings_values['wavelength']
    wavelengths = _study_wavelengths(ctx.spec, ctx.parameters, ctx.registry)
    return wavelengths[0] if wavelengths else settings_values['wavelength']


def all_solved_wavelengths(spec, physics_spec, settings_values, parameters, registry):
    """Vacuum wavelengths [m] the physics solves (frequency_domain: the study's; otherwise the wavelength setting);
    None when the study settings do not resolve or give none."""
    if spec.study.type != FREQUENCY_DOMAIN_STUDY:
        return [settings_values['wavelength']]
    try:
        wavelengths = _study_wavelengths(spec, parameters, registry)
    except ValueError:   # reported at ('study',)
        return None
    return wavelengths or None


def shortest_solved_wavelength(spec, physics_spec, settings_values, parameters, registry):
    """PhysicsType.shortest_wavelength: min of all_solved_wavelengths, or None."""
    wavelengths = all_solved_wavelengths(spec, physics_spec, settings_values, parameters, registry)
    return min(wavelengths) if wavelengths else None


def unused_wavelength_issues(vctx, settings_values, path):
    """Info when a frequency_domain study replaces a wavelength setting changed from its default."""
    if vctx.study_type != FREQUENCY_DOMAIN_STUDY or settings_values['wavelength'] == DEFAULT_WAVELENGTH:
        return []
    return [Issue('info', path, f"the {FREQUENCY_DOMAIN_STUDY} study solves its own wavelengths (its wavelengths "
                                f"list, else wavelength_start to wavelength_stop), so the wavelength setting "
                                f"({settings_values['wavelength']:.4g} m) is not used")]


# ---- eigen-solved FDFD physics: an eigenfrequency study or the optical side of an optomechanics study ----
EIGENFREQUENCY_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
OPTOMECHANICS_STUDY = 'optomechanics'     # = model.study_optomechanics.STUDY_NAME
_OPTOMECHANICAL_FEATURE = 'optomechanical_coupling'   # solid_mechanics feature naming its optical physics
BOTH_GRIDS = ('tensor', 'triangles')   # PhysicsType / FeatureType.grids of host-solved types running on triangles too


def mesh_kind(vctx):
    """Kind of the physics' mesh estimate ('tensor' | 'triangles'), None without one."""
    return None if vctx.mesh is None else getattr(vctx.mesh, 'kind', 'tensor')


def eigen_solved(spec, physics_name):
    """True when the model's study eigen-solves this physics: an eigenfrequency study, or an optomechanics study with an
    enabled optomechanical_coupling naming it (an FDFD physics no feature names solves there as a background one)."""
    if spec.study.type == EIGENFREQUENCY_STUDY:
        return True
    return spec.study.type == OPTOMECHANICS_STUDY and any(
        feature.enabled and feature.type == _OPTOMECHANICAL_FEATURE
        and isinstance(feature.params, dict) and feature.params.get('optical_physics') == physics_name
        for physics in spec.physics for feature in physics.features)


def raw_settings(physics_spec) -> dict:
    """The physics' settings as written (a malformed non-dict, reported by validate, reads as {})."""
    return physics_spec.settings if isinstance(physics_spec.settings, dict) else {}


def auto_mesh_issues(vctx, physics_spec, path, text) -> list:
    """Warning `text` when the physics' mesh is an auto mesh (it sizes cells from the material indices, not from a
    feature value); [] otherwise or for an unknown mesh name (reported at the physics)."""
    try:
        mesh_mode = vctx.spec.mesh_of(physics_spec).mode
    except KeyError:
        return []
    return [Issue('warning', path, text)] if mesh_mode == 'auto' else []


def resolved_solved_wavelengths(vctx, physics_spec, physics_type=None) -> list:
    """Solved wavelengths [m] of a frequency-domain EM physics with its settings resolved ([] when they or the study do
    not resolve); physics_type None = the registered type of the physics."""
    if physics_type is None:
        physics_type = vctx.registry.physics_type(physics_spec.type)
    settings = vctx.values(physics_type.settings, raw_settings(physics_spec))
    if settings is None:
        return []
    return all_solved_wavelengths(vctx.spec, physics_spec, settings, vctx.parameters, vctx.registry) or []


# ---- validation blocks shared by the host-solved types (em_modes, em_fdfd_bor, em_fdfd_conical, em_fdtd_bor) ----
def eigen_only_feature_issues(vctx, physics_spec, feature_spec):
    """Warning on a source / monitor feature where the study eigen-solves the physics (it plays no part there)."""
    if not eigen_solved(vctx.spec, physics_spec.name):
        return []
    return [Issue('warning', feature_path(physics_spec, feature_spec),
                  f'the {vctx.study_type} study solves source-free resonances of this physics: this feature plays no '
                  'part (its result stays empty)')]


def resonance_table_study_issues(vctx, physics_spec, feature_spec, optomechanics=True):
    """Warning on a resonance_table where the study does not eigen-solve the physics (optomechanics: the type may be
    the optical physics of an optomechanics study)."""
    if eigen_solved(vctx.spec, physics_spec.name):
        return []
    filled_by = f'an {EIGENFREQUENCY_STUDY} study' + (f' or as the optical physics of an {OPTOMECHANICS_STUDY} study'
                                                      if optomechanics else '')
    return [Issue('warning', feature_path(physics_spec, feature_spec),
                  f'resonance_table is filled only by {filled_by} (this model runs {vctx.study_type}): its result will '
                  'be missing')]


def scipy_issues(type_name, what, path) -> list:
    """Error when scipy (find_spec only: nothing imported on the GUI thread) is missing for the host solvers `what`."""
    from ...solvers.sparse_direct import scipy_available
    if scipy_available():
        return []
    return [Issue('error', path, f'{type_name} needs scipy ({what}), which is not installed: pip install scipy')]


def host_matrix_back_end_issues(solvers, type_name, path) -> list:
    """Error for a 'linear' node with back_end cusolver_qr on a type that factorises a host matrix."""
    resolved = (solvers or {}).get('linear')
    if resolved is None or resolved.values.get('back_end') != 'cusolver_qr':
        return []
    return [Issue('error', path, f"linear solver '{resolved.node}': {type_name} factorises a host matrix (SuperLU or "
                                 "cuDSS): use back_end 'auto', 'scipy_splu', 'cudss' or 'cudss_mixed', or Automatic")]


def missing_source_issues(vctx, physics_spec, path, source_types, driven_studies) -> list:
    """Warning when a driven study (one of driven_studies) solves a physics without an enabled source feature of
    source_types."""
    if vctx.study_type not in driven_studies or any(feature.enabled and feature.type in source_types
                                                    for feature in physics_spec.features):
        return []
    return [Issue('warning', path, f'a {vctx.study_type} study solves the driven problem, but '
                                   f"'{physics_spec.name}' has no enabled {' or '.join(source_types)}: every field "
                                   f'stays zero (use an {EIGENFREQUENCY_STUDY} study for resonances)')]


def fem_size_issues(vctx, physics_spec, settings_values, path, remedy=None) -> list:
    """The host SuperLU size rule of the estimated FEM unknowns of a triangles mesh (fem_modes.size_issue_for_unknowns
    with the eigen factorization; remedy: the closing advice, None = fem_modes' default)."""
    from .fem_modes import DISCRETIZATION_ORDERS, estimated_unknowns, size_issue_for_unknowns
    mesh = vctx.mesh
    unknowns = estimated_unknowns(mesh.n_elements, mesh.n_edges, mesh.n_nodes,
                                  DISCRETIZATION_ORDERS[settings_values['discretization']])
    issue = size_issue_for_unknowns(unknowns, eigen_factorization(vctx.solvers(physics_spec)),
                                    *(() if remedy is None else (remedy,)))
    return [] if issue is None else [Issue(issue[0], path, f'estimated {issue[1]}')]


def other_em_physics_labels(vctx, own_type, allowed=()) -> list:
    """"'name' (type)" of every other family-'em' physics of the model whose type is neither own_type nor in allowed."""
    labels = []
    for physics in vctx.spec.physics:
        try:
            physics_type = vctx.registry.physics_type(physics.type)
        except KeyError:
            continue
        if physics_type.family == EM_FAMILY and physics_type.name != own_type and physics_type.name not in allowed:
            labels.append(f"'{physics.name}' ({physics_type.name})")
    return labels


# ---- run-time estimates of the host-solved types (em_modes, em_fdfd_bor, em_fdfd_conical) ----
def physics_counter(context, name) -> tuple:
    """summary.json path of the estimated physics' own work counter `name` (its work_counters())."""
    return ('solvers', context.name, 'counters', name)


def resonance_eigenpairs(context, optical=True) -> int:
    """Arnoldi eigenpairs of an eigen-solved FD type: 2 x the wanted resonances (optical: the optomechanics study's
    optical count first) + 10."""
    wanted = (context.count('optical_eigenpairs') if optical else 0) or context.count('eigenpairs', 1) or 1
    return 2 * int(wanted) + 10


def eigen_backend_kwargs(context) -> dict:
    """solver_work kwargs of the 'eigen' role: {} on a cuDSS factorization (the run backend), else the host SuperLU
    ({'backend': 'cpu'})."""
    return {} if eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS else {'backend': 'cpu'}


def role_counter(context, role, name) -> tuple:
    """summary.json path of the work counter `name` of the estimated physics' `role` runtime."""
    return ('solvers', context.name, 'roles', role, 'counters', name)


def kept_line_block_factors(context, unknowns, nnz, factors, block_size):
    """'shifted' (line_block) recipe of `factors` kept factors: factorize / factorize_operators run no sweeps of their
    own, the solves on them are the caller's (solvers.line_block.factor_solve_ops each)."""
    estimate = context.solver_work('shifted', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=factors,
                                   block_size=block_size)
    return replace(estimate, items=tuple(item for item in estimate.items if item.unit != 'sweep'))


def scaled_ops(ops, factor):
    """Ops tuple (entries an op or (multiplicity, op)) with every multiplicity times `factor`."""
    return tuple((factor * entry[0], entry[1]) if isinstance(entry, tuple) else (factor, entry) for entry in ops)


def create_em_physics(ctx, physics_spec, settings_values, polarization, background_targets):
    """EMPhysics of `polarization` at the 'stepper' solver's precision (single / double) with dispersive poles, the
    dispersive background written into the MaterialGrid arrays background_targets (e.g. ('eps_r',) or ('eps_xx',
    'eps_yy')) and absorption tracking; never finalizes."""
    em = EMPhysics(ctx.domain, polarization, boundary_config=ctx.domain.boundary_config,
                   precision=stepper_precision(ctx.solvers(physics_spec.name)))
    write_dispersive_materials(ctx, physics_spec, em, background_targets)
    if settings_values['track_absorption'] or _q_accum_referenced(ctx.spec, physics_spec.name, ctx.registry):
        em.engine.enable_q_accum = True
    return em


def write_dispersive_materials(ctx, physics_spec, solver, background_targets):
    """solver.add_dispersive per pole of every non-empty dispersive material region, its background permittivity
    written into the MaterialGrid arrays background_targets, then solver.materials_changed for those arrays. With
    solver_defaults.conformal_boundaries reaching a solver that has conformal_pole_weights (2D FDTD) the poles take
    the cut-cell shares of the region's drawn outline (BuildContext.conformal_locator_of, material_test)."""
    material_property_params = ctx.registry.physics_type(physics_spec.type).material_properties
    materials = ctx.domain.materials
    field_names = FIELD_COORDINATE_NAMES.union(ctx.parameters)   # x, y (+ z in 3D; validate rejects z in 2D)
    locator = (ctx.conformal_locator_of(physics_spec.name) if hasattr(solver, 'conformal_pole_weights')
               else None)
    written_arrays = []
    for material_index, region in enumerate(ctx.material_regions):   # spec.materials order (the locator's)
        if not region.dispersion or not bool(region.mask.any()):
            continue
        weights = (None if locator is None
                   else solver.conformal_pole_weights(region.mask, locator.material_test(material_index)))
        pole_options = {} if weights is None else {'weights': weights}
        for array in background_targets:
            background = background_permittivity(region.properties, region.dispersion, material_property_params,
                                                 array)
            # a background that depends on physics fields is written by the builder's dynamic material update
            if referenced_names(background) <= field_names:
                materials.assign(array, region.mask, ctx.eval_field(background, mask=region.mask))
                if array not in written_arrays:
                    written_arrays.append(array)
        for index, pole in enumerate(region.dispersion):
            try:
                dispersive_pole = make_pole(pole, ctx.eval_scalar)
            except ValueError as exc:
                raise ValueError(f"material '{region.name}', dispersion pole {index + 1}: {exc}") from exc
            solver.add_dispersive(dispersive_pole, region.mask, **pole_options)
    if written_arrays:
        solver.materials_changed(tuple(written_arrays))


# coupling types whose CouplingTerm reads q_accum (coupling.standard joule_heating, definitions_semiconductor): the
# runner enables loss accounting for them (coupling/runner.py _enable_loss_accounting)
_Q_ACCUM_COUPLINGS = frozenset({'em_heat_joule', 'em_semiconductor_generation'})


def _q_accum_referenced(spec, physics_name, registry):
    """True when the model reads q_accum of this physics: a material property names it, a derived value names it or
    absorbed_power, an enabled field_transfer reads it, or an enabled _Q_ACCUM_COUPLINGS type reads it (pole values
    are parameter-only)."""
    q_accum_names = frozenset(name for name, (physics, field_name) in state_field_names(spec, registry).items()
                              if physics == physics_name and field_name == 'q_accum')
    if not q_accum_names:
        return False
    absorbed_power_names = frozenset(name for name, (physics, field_name) in output_field_names(spec, registry).items()
                                     if physics == physics_name and field_name == 'absorbed_power')
    if any(_references_any(derived.expression, q_accum_names | absorbed_power_names)
           for derived in spec.derived_values):
        return True
    for coupling in spec.couplings:
        if not coupling.enabled or coupling.source != physics_name:
            continue
        if coupling.type in _Q_ACCUM_COUPLINGS:
            return True
        if (coupling.type == 'field_transfer' and isinstance(coupling.params, dict)
                and coupling.params.get('source_field') == 'q_accum'):
            return True
    return any(_references_any(value, q_accum_names) for material in spec.materials
               if isinstance(material.properties, dict) for value in material.properties.values())


def _references_any(value, names):
    """True when an expression value references any of `names` (numbers and malformed values: False)."""
    if not isinstance(value, str):
        return False
    try:
        return not referenced_names(value).isdisjoint(names)
    except ExpressionError:
        return False


def make_pole(pole, evaluate):
    """DrudePole / LorentzPole from a dispersion dict whose values are passed through evaluate(value) -> float."""
    if not isinstance(pole, dict):
        raise ValueError(f'expected a dict, got {type(pole).__name__}')
    model = pole.get('model')
    if not isinstance(model, str) or model not in _POLE_MODELS:
        raise ValueError(f"'model' must be one of {', '.join(_POLE_MODELS)}, got {model!r}")
    pole_class, required, optional = _POLE_MODELS[model]
    missing = [key for key in required if key not in pole]
    unknown = sorted(str(key) for key in pole if key != 'model' and key not in required + optional)
    if missing or unknown:
        raise ValueError(f'{model} pole needs {", ".join(required)} (optional: {", ".join(optional) or "none"}); '
                         f'missing {missing}, unknown {unknown}')
    numbers = {key: evaluate(pole[key]) for key in required + optional if key in pole}
    negative = [key for key, value in numbers.items() if value < 0]
    if negative:
        raise ValueError(f'{model} pole values must be >= 0: {", ".join(negative)}')
    if numbers.get('eps_inf', 1.0) <= 0:
        raise ValueError(f"drude eps_inf must be > 0, got {numbers['eps_inf']:g}")
    return pole_class(**numbers)


# ---- per-array permittivity (the builder's ownership rule: a component wins over the isotropic eps_r) ----
def background_permittivity(properties, dispersion, material_property_params, array):
    """Raw background value of permittivity array `array` for a material (properties merged by effective_material_data:
    no None values): the owning property's value (model.schema.array_owner), else the first Drude eps_inf, else 1."""
    owner = array_owner(array, material_property_params, properties)
    if owner is not None:
        return properties[owner]
    for pole in dispersion:
        if isinstance(pole, dict) and pole.get('model') == 'drude':
            return pole.get('eps_inf', 1.0)
    return 1.0


def _array_permittivity(properties, dispersion, material_property_params, array, parameters):
    """background_permittivity evaluated over the parameters; ExpressionError when not a plain number."""
    return evaluate_scalar(background_permittivity(properties, dispersion, material_property_params, array),
                           parameters)


def _refractive_index(material_spec, parameters, material_property_params_list):
    """Static index sqrt(eps * mu_r), eps = the largest background permittivity over the permittivity arrays of the
    given types' material_properties + Lorentz delta_eps; NaN when not a plain number."""
    try:
        properties, dispersion = _effective_material(material_spec)
        eps = max(_array_permittivity(properties, dispersion, params, array, parameters)
                  for params in material_property_params_list for array in permittivity_arrays(params))
        for pole in dispersion:
            if isinstance(pole, dict) and pole.get('model') == 'lorentz':
                eps += evaluate_scalar(pole.get('delta_eps', 0.0), parameters)
        eps_mu = eps * evaluate_scalar(properties.get('mu_r', 1.0), parameters)
    except ValueError:
        return math.nan
    return math.sqrt(eps_mu) if eps_mu > 0 else math.nan


def _complex_index(material_spec, parameters, material_property_params, wavelengths):
    """Complex index sqrt(eps mu) per vacuum wavelength [m] as the FDFD assembles it (eps = background - j sigma_e /
    (omega eps0) + every pole's chi, on the permittivity array of largest |eps|); NaN when not a plain number."""
    omega = 2.0 * math.pi * C0 / np.asarray(wavelengths, dtype=np.float64)
    try:
        properties, dispersion = _effective_material(material_spec)
        sigma_e, mu_r, sigma_m = (
            evaluate_scalar(default if properties.get(name) is None else properties[name], parameters)
            for name, default in (('sigma_e', 0.0), ('mu_r', 1.0), ('sigma_m', 0.0)))
        susceptibility = sum((make_pole(pole, lambda value: evaluate_scalar(value, parameters)).susceptibility(omega)
                              for pole in dispersion), np.zeros(omega.shape, dtype=np.complex128))
        backgrounds = np.array([_array_permittivity(properties, dispersion, material_property_params, array,
                                                    parameters)
                                for array in permittivity_arrays(material_property_params)])
        eps_by_array = backgrounds[:, None] - 1j * sigma_e / (omega * EPS0) + susceptibility
        eps = eps_by_array[np.argmax(np.abs(eps_by_array), axis=0), np.arange(omega.size)]
        mu = mu_r - 1j * sigma_m / (omega * MU0)
    except ValueError:
        return np.full(omega.shape, np.nan, dtype=np.complex128)
    return np.sqrt(eps * mu)


# ---- build: feature apply ----
def single_object(selection, feature_spec, kind):
    """Name of the one geometry object of the resolved selection; ValueError otherwise."""
    if len(selection.objects) != 1:
        raise ValueError(f"feature '{feature_spec.name}' needs exactly one {kind} object, "
                         f"got {selection.objects or 'none'}")
    return selection.objects[0]


def line_placement(ctx, selection, feature_spec):
    """(orientation, position across the line, (start, end) along it) of the feature's axis-aligned line."""
    name = single_object(selection, feature_spec, 'line')
    x0, y0, x1, y1 = ctx.line(name)
    orientation = line_orientation({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1},
                                   ctx.domain.size_x, ctx.domain.size_y)
    if orientation == 'horizontal':
        return orientation, 0.5 * (y0 + y1), (x0, x1)
    if orientation == 'vertical':
        return orientation, 0.5 * (x0 + x1), (y0, y1)
    raise ValueError(f"feature '{feature_spec.name}': line '{name}' must be horizontal or vertical")


def source_time_kwargs(values):
    """Time dependence of a source feature: phase= of an FDFD phasor (Param 'phase'), the FDTD waveform (wavelength,
    bandwidth, waveform), or both for an FDTD port with a launch phase."""
    kwargs = {'phase': values['phase']} if 'phase' in values else {}
    if 'waveform' in values or 'phase' not in values:
        kwargs.update(wavelength=values['wavelength'], bandwidth=values['bandwidth'], waveform=values['waveform'])
    return kwargs


def _monitor_wavelengths(values):
    """numpy wavelengths [m] of a spectral monitor (n_wavelengths == 1 -> the centre of the range)."""
    return wavelength_range(values['wavelength_start'], values['wavelength_stop'], values['n_wavelengths'])


def _monitor_wavelength_kwargs(values):
    """wavelengths= of an FDTD spectral monitor; {} for an FDFD one (no spectral params: the solves set them)."""
    return {'wavelengths': _monitor_wavelengths(values)} if 'wavelength_start' in values else {}


def apply_ez_point_source(ctx, solver, feature_spec, values, selection):
    x, y = ctx.point(single_object(selection, feature_spec, 'point'))
    return solver.add_point_source(x, y, current=values['current'], **source_time_kwargs(values))


def apply_component_point_source(ctx, solver, feature_spec, values, selection):
    """Point source with a current component (BOR, conical, fdtd_bor): phase or waveform by source_time_kwargs."""
    x, y = ctx.point(single_object(selection, feature_spec, 'point'))
    return solver.add_point_source(x, y, component=values['component'], current=values['current'],
                                   **source_time_kwargs(values))


def apply_hz_point_source(ctx, solver, feature_spec, values, selection):
    x, y = ctx.point(single_object(selection, feature_spec, 'point'))
    if values['excitation'] == 'magnetic_current':
        return solver.add_magnetic_line_current(x, y, magnetic_current=values['magnetic_current'],
                                                **source_time_kwargs(values))
    return solver.add_electric_dipole(x, y, dipole_moment=values['dipole_moment'], orientation=values['orientation'],
                                      **source_time_kwargs(values))


def apply_line_source(ctx, solver, feature_spec, values, selection):
    orientation, position, span = line_placement(ctx, selection, feature_spec)
    axis = 'x' if orientation == 'horizontal' else 'y'   # LineSource axis = direction along the line
    return solver.add_line_source(axis, position, amplitude=values['amplitude'], span=span,
                                  **source_time_kwargs(values))


def apply_gaussian_beam(ctx, solver, feature_spec, values, selection):
    orientation, position, span = line_placement(ctx, selection, feature_spec)
    axis = 'x' if orientation == 'horizontal' else 'y'
    center = values['center'] if values['center'] is not None else 0.5 * (span[0] + span[1])
    return solver.add_gaussian_beam(axis, position, values['waist'], center, amplitude=values['amplitude'],
                                    span=span, **source_time_kwargs(values))


def apply_tfsf(ctx, solver, feature_spec, values, selection):
    x0, y0, x1, y1 = ctx.rect(single_object(selection, feature_spec, 'rectangle'))
    return solver.add_tfsf(x0, y0, x1, y1, amplitude=values['amplitude'], **source_time_kwargs(values))


def apply_pec(ctx, solver, feature_spec, values, selection):
    """Cell mask PEC; with a conformal_locator (solver_defaults.conformal_boundaries) also its drawn outline."""
    mask = ctx.domain_mask(selection.entities)
    locator = getattr(solver, 'conformal_locator', None)
    if locator is not None:
        solver.add_conformal_pec_region(mask, locator.inside(selection.entities))
    else:
        solver.add_pec_region(mask)
    return None


def apply_field_probe(ctx, solver, feature_spec, values, selection):
    x, y = ctx.point(single_object(selection, feature_spec, 'point'))
    return solver.add_field_probe(x, y, values['component'])


def apply_flux_monitor(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    return solver.add_flux_monitor(axis, position, span, **_monitor_wavelength_kwargs(values))


def mode_line(ctx, selection, feature_spec):
    """(axis along the line, position across it, (start, end) sorted along it) of a mode feature's line."""
    orientation, position, span = line_placement(ctx, selection, feature_spec)
    return ('x' if orientation == 'horizontal' else 'y'), position, tuple(sorted(span))


def apply_mode_source(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    return solver.add_mode_source(axis, position, span, power=values['power'], mode_number=values['mode_number'],
                                  direction=values['direction'], **source_time_kwargs(values))


def apply_mode_monitor(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    return solver.add_mode_monitor(axis, position, span, mode_number=values['mode_number'],
                                   **_monitor_wavelength_kwargs(values))


def apply_dft_monitor(ctx, solver, feature_spec, values, selection):
    x0, y0, x1, y1 = ctx.rect(single_object(selection, feature_spec, 'rectangle'))
    return solver.add_dft_monitor(x0, y0, x1, y1, wavelengths=_monitor_wavelengths(values),
                                  downsample=values['downsample'])


def apply_far_field(ctx, solver, feature_spec, values, selection):
    x0, y0, x1, y1 = ctx.rect(single_object(selection, feature_spec, 'rectangle'))
    return solver.add_near_to_far(x0, y0, x1, y1, **_monitor_wavelength_kwargs(values))


# ---- results ----
def object_result(result_function, with_solver=False):
    """Feature result callback returning result_function(feature object[, solver])."""
    def result_callback(session, solver, feature_object, feature_spec, options):
        return result_function(feature_object, solver) if with_solver else result_function(feature_object)
    return result_callback


def probe_result_callback(session, solver, probe, feature_spec, options):
    return probe_result(probe, solver.renderer().field_info(probe.field_component).unit)


def dft_result_callback(session, solver, monitor, feature_spec, options):
    return dft_result(monitor, session.domain, freq_index=options.get('freq_index'), physics=solver)


flux_result_callback = object_result(flux_result, with_solver=True)
mode_source_result_callback = object_result(mode_source_result)
mode_monitor_result_callback = object_result(mode_monitor_result, with_solver=True)


def make_far_field_result_callback(far_field_feature_params, result_function=far_field_result):
    """Far-field result callback resolving n_angles with the feature type's params, then result_function(monitor,
    n_angles=...) (FDTD results.far_field_result; FDFD fdfd_monitors.harmonic_far_field_result)."""
    def far_field_result_callback(session, solver, monitor, feature_spec, options):
        n_angles = resolve_values(far_field_feature_params, feature_spec.params, session.parameters)['n_angles']
        return result_function(monitor, n_angles=n_angles)
    return far_field_result_callback


# ---- validation helpers, also of definition_feature_validation (CPU only) ----
def _window_axes(mesh):
    """Per axis (x, y): (full-grid cell widths, full-grid cell edges in the physical frame (same values as
    Domain.cell_edges, so cell lookups match the runtime exactly), first window cell, last window cell)."""
    axes = []
    for widths, edges, lo_face, hi_face in ((mesh.dx_cells, mesh.x_edges, 'x_lo', 'x_hi'),
                                            (mesh.dy_cells, mesh.y_edges, 'y_lo', 'y_hi')):
        widths = np.asarray(widths, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        if edges.size != widths.size + 1:
            raise ValueError(f'MeshEstimate edges ({edges.size}) must cover all {widths.size} cells, PML included')
        first = int(mesh.npml_sides[lo_face])
        last = widths.size - int(mesh.npml_sides[hi_face]) - 1
        axes.append((widths, edges, first, last))
    return axes


def _effective_material(material_spec):
    """materials.effective_material_data of a MaterialSpec (as the builder merges it); ValueError when malformed."""
    from ...materials import effective_material_data   # lazy: open_sol.materials imports physics.em
    return effective_material_data(material_spec.library, material_spec.properties, material_spec.dispersion)


def _uses_failed_parameter(pole, parameter_errors):
    """True when a value of the dispersion dict references a parameter that failed to evaluate."""
    if not isinstance(pole, dict):
        return False
    for key, value in pole.items():
        if key == 'model':
            continue
        try:
            if not referenced_names(value).isdisjoint(parameter_errors):
                return True
        except ExpressionError:
            continue
    return False


def _written_permittivity_slots(vctx, coupling_spec, slot_arrays):
    """Permittivity slots of slot_arrays a coupling rewrites on its target (its CouplingType.writes, any kind)."""
    return tuple(slot for slot in coupling_writes(vctx, coupling_spec) if slot in slot_arrays)


# ---- physics validation (feature validation: definition_feature_validation) ----
def _boundary_issues(settings_values, path):
    """Periodic faces paired."""
    issues = []
    for axis in ('x', 'y'):
        lo, hi = settings_values[f'bc_{axis}_lo'], settings_values[f'bc_{axis}_hi']
        if (lo == 'periodic') != (hi == 'periodic'):
            issues.append(Issue('error', path, f'periodic boundaries must be paired: bc_{axis}_lo is {lo}, '
                                               f'bc_{axis}_hi is {hi}'))
    return issues


def _em_physics_types(vctx):
    """(name of the first EM physics, distinct EM PhysicsTypes in spec order); physics of unknown type skipped."""
    first_name, physics_types = None, {}
    for physics in vctx.spec.physics:
        try:
            physics_type = vctx.registry.physics_type(physics.type)
        except KeyError:
            continue
        if physics_type.family == EM_FAMILY:
            first_name = physics.name if first_name is None else first_name
            physics_types.setdefault(physics_type.name, physics_type)
    return first_name, list(physics_types.values())


def _material_issues(vctx, em_types):
    """Model-wide checks over the permittivity arrays of every present EM type: dispersion dicts build valid poles;
    warnings for a Drude material whose permittivity is set and for a user permittivity property no array uses; the
    nk_fit report (definition_material_fit)."""
    em_params = [physics_type.material_properties for physics_type in em_types]
    # permittivity properties some present EM type knows (a registered component of an absent type is dropped)
    em_permittivity_names = tuple(name for name in PERMITTIVITY_PROPERTY_NAMES
                                  if any(param.name == name for params in em_params for param in params))
    em_type_names = ', '.join(physics_type.name for physics_type in em_types)
    issues = []
    for material in vctx.spec.materials:
        path = ('material', material.name)
        try:
            properties, dispersion = _effective_material(material)
        except ValueError:   # unknown library / malformed properties: reported by the generic material checks
            continue
        for index, pole in enumerate(dispersion):
            try:
                make_pole(pole, lambda value: evaluate_scalar(value, vctx.parameters))
            except ValueError as exc:
                if not _uses_failed_parameter(pole, vctx.parameter_errors):
                    issues.append(Issue('error', path, f'dispersion pole {index + 1}: {exc}'))
        if not material.selection:
            continue
        owners = [array_owner(array, params, properties)
                  for params in em_params for array in permittivity_arrays(params)]
        set_owners = list(dict.fromkeys(owner for owner in owners if owner is not None))
        has_drude = any(isinstance(pole, dict) and pole.get('model') == 'drude' for pole in dispersion)
        if has_drude and set_owners:
            single = len(set_owners) == 1
            issues.append(Issue('warning', path, f"{', '.join(set_owners)} {'is' if single else 'are'} set, so "
                                                 f"{'it replaces' if single else 'they replace'} the Drude pole's "
                                                 f"eps_inf as the background permittivity"))
        user_properties = material.properties if isinstance(material.properties, dict) else {}
        if (user_properties.get('eps_r') is not None and owners
                and all(owner not in (None, 'eps_r') for owner in owners)):
            components = ', '.join(set_owners)
            issues.append(Issue('warning', path, f'eps_r is not used: every permittivity array of the present EM '
                                                 f'physics is owned by a component ({components}); set '
                                                 f'{components} instead'))
        for name, value in user_properties.items():
            if name in PERMITTIVITY_PROPERTY_NAMES and value is not None and name not in em_permittivity_names:
                issues.append(Issue('warning', path, f"{name} is not used by the present EM physics "
                                                     f"({em_type_names}); set one of "
                                                     f"{', '.join(em_permittivity_names)} instead"))
    from .definition_material_fit import nk_fit_issues   # lazy: it imports this module
    return issues + nk_fit_issues(vctx)


def _permittivity_writer_issues(vctx, physics_spec, slot_arrays, material_property_params):
    """Errors when a coupling writing a permittivity slot of this physics collides with a state-dependent property
    that owns the slot's array for a selected material (the builder's ownership rule; several couplings on one slot:
    the writer rule of model.validate)."""
    writers_by_slot = {slot: [] for slot in slot_arrays}
    for coupling in vctx.spec.couplings:
        if coupling.enabled and coupling.target == physics_spec.name:
            for slot in _written_permittivity_slots(vctx, coupling, slot_arrays):
                writers_by_slot[slot].append(coupling.name)
    state_names = frozenset(vctx.state_names) | TIME_NAMES   # both make the property a DynamicMaterialUpdate
    issues, collisions = [], {}   # (material, property) -> (coupling, slot) of its first collision
    if not any(writers_by_slot.values()):
        return issues
    merged = {}   # material name -> its effective properties, merged once (not per slot)
    for material in vctx.spec.materials:
        if material.selection:
            try:
                merged[material.name] = _effective_material(material)[0]
            except ValueError:   # unknown library / malformed properties: reported by the generic material checks
                continue
    for slot, writers in writers_by_slot.items():
        if not writers:
            continue
        for material in vctx.spec.materials:
            properties = merged.get(material.name) if material.selection else None
            if properties is None:
                continue
            # only the owner writes the array (a static component beside a state-dependent eps_r is no second writer)
            owner = array_owner(slot_arrays[slot], material_property_params, properties)
            if owner is not None and _references_any(properties[owner], state_names):
                collisions.setdefault((material.name, owner), (writers[0], slot))
    for (material_name, property_name), (writer, slot) in collisions.items():
        issues.append(Issue('error', ('material', material_name),
                            f"{property_name} depends on physics fields or t while coupling '{writer}' rewrites slot "
                            f"'{slot}' of '{physics_spec.name}': two writers of one permittivity; make "
                            f"{property_name} static or disable the coupling"))
    return issues


def _upper_frequency_ratio(values):
    """Highest significant source frequency / carrier frequency: gaussian 1 + bandwidth (1/e amplitude), ricker 2
    (~20 %), cw 1."""
    if values['waveform'] == 'gaussian':
        return 1.0 + values['bandwidth']
    if values['waveform'] == 'ricker':
        return 2.0
    return 1.0


def make_source_shortest_wavelength(feature_types):
    """PhysicsType.shortest_wavelength of an FDTD type: shortest significant vacuum wavelength [m] over the enabled
    source features, or None; features whose params do not resolve are skipped (reported at their own nodes)."""
    def shortest_wavelength(spec, physics_spec, settings_values, parameters, registry):
        shortest = None
        for feature in physics_spec.features:
            if (not feature.enabled or feature.type not in feature_types
                    or feature_types[feature.type].category != 'source'):
                continue
            try:
                values = resolve_values(feature_types[feature.type].params, feature.params, parameters)
            except (ValueError, TypeError):   # e.g. params not a dict
                continue
            wavelength = values['wavelength'] / _upper_frequency_ratio(values)
            shortest = wavelength if shortest is None else min(shortest, wavelength)
        return shortest
    return shortest_wavelength


def _resolution_issues(vctx, physics_spec, wavelengths, material_property_params, time_domain):
    """Warnings (material preview x tensor mesh) for < 10 cells per wavelength in a material: FDTD at the shortest
    vacuum wavelength [m] with the static index; FDFD at every solved one with |n(omega)|, plus < 2 cells per skin
    depth. None on a triangles mesh (no cell widths; its element sizes are the user's / the auto size rule's)."""
    wavelengths = np.array([wavelength for wavelength in wavelengths if isinstance(wavelength, (int, float))
                            and math.isfinite(wavelength) and wavelength > 0], dtype=np.float64)
    if not wavelengths.size or vctx.mesh is None or getattr(vctx.mesh, 'kind', 'tensor') != 'tensor':
        return []
    material_map = vctx.material_map()
    if material_map is None:
        return []
    image, names, extent = material_map
    local_widths = []
    for (widths, edges, first, last), pixel_centers in zip(_window_axes(vctx.mesh),
                                                           preview_pixel_centers(extent, image.shape)):
        cells = np.clip(np.searchsorted(edges, pixel_centers, side='right') - 1, first, last)
        local_widths.append(widths[cells])
    cell_size = np.maximum(local_widths[0][:, None], local_widths[1][None, :])
    shortest = float(wavelengths.min())
    # per material (0 = vacuum): (shortest wavelength in it, its vacuum wavelength, index) and (skin depth, vacuum
    # wavelength); NaN lengths (unevaluable materials) are never flagged
    medium_wavelengths, skin_depths = [(shortest, shortest, 1.0)], [(math.inf, shortest)]
    for material in vctx.spec.materials:
        if time_domain:
            index = _refractive_index(material, vctx.parameters, (material_property_params,))
            medium_wavelengths.append((shortest / index, shortest, index))
            continue
        index = _complex_index(material, vctx.parameters, material_property_params, wavelengths)
        with np.errstate(divide='ignore', invalid='ignore'):
            in_medium = wavelengths / np.abs(index)
            depth = wavelengths / (2.0 * math.pi * np.abs(index.imag))
        k = int(np.argmin(np.nan_to_num(in_medium, nan=np.inf)))
        medium_wavelengths.append((float(in_medium[k]), float(wavelengths[k]), float(abs(index[k]))))
        k = int(np.argmin(np.nan_to_num(depth, nan=np.inf)))
        skin_depths.append((float(depth[k]), float(wavelengths[k])))

    def worst_pixel(length_by_material):
        """(cells per length, material index, cell size) at the pixel with the fewest cells per length."""
        cells_per_length = np.nan_to_num(np.array(length_by_material)[image] / cell_size, nan=np.inf)
        worst = np.unravel_index(int(np.argmin(cells_per_length)), image.shape)
        return float(cells_per_length[worst]), int(image[worst]), float(cell_size[worst])

    def where(material_index):
        return f"material '{names[material_index - 1]}'" if material_index else 'vacuum'

    path = ('physics', physics_spec.name)
    label, index_label = ('shortest source wavelength', 'n') if time_domain else ('solved wavelength', '|n|')
    issues = []
    cells, material_index, cell = worst_pixel([entry[0] for entry in medium_wavelengths])
    if cells < _MIN_CELLS_PER_WAVELENGTH:
        _, vacuum_wavelength, index = medium_wavelengths[material_index]
        issues.append(Issue('warning', path,
                            f'only {cells:.1f} cells per wavelength in {where(material_index)} ({label} '
                            f'{vacuum_wavelength:.4g} m, {index_label} = {index:.3g}, cell {cell:.3g} m); use at '
                            f'least {_MIN_CELLS_PER_WAVELENGTH:.0f}, about 20 for accurate results'))
    if not time_domain:
        cells, material_index, cell = worst_pixel([entry[0] for entry in skin_depths])
        if cells < _MIN_CELLS_PER_SKIN_DEPTH:
            depth, vacuum_wavelength = skin_depths[material_index]
            issues.append(Issue('warning', path,
                                f'only {cells:.1f} cells per skin depth in {where(material_index)} ({label} '
                                f'{vacuum_wavelength:.4g} m, skin depth lambda0 / (2 pi |Im n|) = {depth:.3g} m, '
                                f'cell {cell:.3g} m): the field decay, q_abs and reflection phases are under-resolved; '
                                f'use at least {_MIN_CELLS_PER_SKIN_DEPTH:.0f}, about 4 for accurate results'))
    return issues


def _enabled_features(physics_spec, physics_type):
    """(feature spec, its FeatureType) of the enabled features with dict params and a registered type."""
    for feature in physics_spec.features:
        if not feature.enabled or not isinstance(feature.params, dict):
            continue
        try:
            yield feature, physics_type.feature_type(feature.type)
        except KeyError:
            continue


def _source_values(vctx, physics_spec, physics_type):
    """[(feature spec, resolved params)] of the enabled source features with a resolving 'wavelength' param."""
    sources = []
    for feature, feature_type in _enabled_features(physics_spec, physics_type):
        if feature_type.category != 'source':
            continue
        values = vctx.values(feature_type.params, feature.params)
        if values is not None and isinstance(values.get('wavelength'), (int, float)) and values['wavelength'] > 0:
            sources.append((feature, values))
    return sources


def _wavelength_unit_issues(vctx, labelled_wavelengths):
    """Warning per (path, label, wavelength [m]) below 1e-9 m or above 100 domain diagonals: nm or um typed as m."""
    size = vctx.domain_size
    diagonal = math.hypot(*size) if size is not None else None
    issues = []
    for path, label, wavelength in labelled_wavelengths:
        if wavelength < _MIN_PLAUSIBLE_WAVELENGTH:
            problem = f'is below {_MIN_PLAUSIBLE_WAVELENGTH:g} m'
        elif diagonal and wavelength > _MAX_WAVELENGTH_DIAGONALS * diagonal:
            problem = f'is over {_MAX_WAVELENGTH_DIAGONALS:.0f} x the domain diagonal ({diagonal:.4g} m)'
        else:
            continue
        issues.append(Issue('warning', path, f'{label} {wavelength:.4g} m {problem}: lengths are in m (e.g. 1.55e-6 '
                                             f'for 1550 nm), so this is probably a unit slip'))
    return issues


def _pulse_end(values):
    """End 2 t0 [s] of a gaussian / ricker source waveform (as core.sources builds it); None for cw."""
    f0 = C0 / values['wavelength']
    if values['waveform'] == 'gaussian':
        bandwidth = values['bandwidth']   # a bandwidth <= 0 is reported by the generic param check
        return GaussianPulse(f0, bandwidth).duration if isinstance(bandwidth, (int, float)) and bandwidth > 0 else None
    if values['waveform'] == 'ricker':
        return RickerWavelet(f0).duration
    return None


def _duration_issue(vctx, physics_spec, physics_type, sources, material_property_params):
    """Warning when a time_dependent duration ends before the longest pulse (2 t0), plus the transit time
    diagonal * n_max / C0 when a spectral monitor exists; None otherwise."""
    if vctx.study_type != 'time_dependent':
        return None
    duration = study_duration(vctx)
    pulse_ends = [(end, feature.name) for feature, values in sources
                  if (end := _pulse_end(values)) is not None]
    if not isinstance(duration, (int, float)) or not pulse_ends:
        return None
    pulse_end, source_name = max(pulse_ends)
    needed, what = pulse_end, f"the pulse of source '{source_name}' ends (2 t0 = {pulse_end:.4g} s)"
    spectral_monitors = [feature.name for feature, feature_type in _enabled_features(physics_spec, physics_type)
                         if any(param.name == 'wavelength_start' for param in feature_type.params)]
    size = vctx.domain_size
    if size is not None and vctx.size_z is not None:   # em_fdtd_3d: the 3D diagonal
        size = (*size, vctx.size_z)
    if spectral_monitors and size is not None:
        indices = [_refractive_index(material, vctx.parameters, (material_property_params,))
                   for material in vctx.spec.materials]
        n_max = max([1.0] + [index for index in indices if not math.isnan(index)])
        transit = math.hypot(*size) * n_max / C0
        needed += transit
        what += (f' and its light crosses the domain (diagonal x n_max {n_max:.3g} / C0 = {transit:.4g} s), so the '
                 f"spectra of {', '.join(spectral_monitors)} are truncated")
    if duration >= needed:
        return None
    return Issue('warning', ('physics', physics_spec.name),
                 f'study duration {duration:.4g} s ends before {what}; use a duration of at least {needed:.3g} s '
                 f'(plus the ring-down of resonances)')


_IN_BAND_LEVEL = 1e-2   # source amplitude spectrum / its peak below which a monitor wavelength is out of band
_GAUSSIAN_BAND_WIDTHS = math.sqrt(math.log(1.0 / _IN_BAND_LEVEL))   # |f - f0| / (bandwidth f0) at that level
_RICKER_BAND = (0.0608, 2.76)   # f / f_peak where (f / f_peak)^2 exp(1 - (f / f_peak)^2) equals that level


def _source_band(values, duration):
    """(f_lo, f_hi) [Hz] where a source's amplitude spectrum is >= _IN_BAND_LEVEL of its peak; cw: f0 +- 1 / duration
    (the DFT resolution; f0 alone without a duration); None when the bandwidth does not resolve."""
    f0 = C0 / values['wavelength']
    if values['waveform'] == 'gaussian':
        bandwidth = values['bandwidth']   # a bandwidth <= 0 is reported by the generic param check
        if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
            return None
        half_width = _GAUSSIAN_BAND_WIDTHS * bandwidth * f0
    elif values['waveform'] == 'ricker':
        return _RICKER_BAND[0] * f0, _RICKER_BAND[1] * f0
    else:
        half_width = 1.0 / duration if duration is not None else 1e-9 * f0
    return max(0.0, f0 - half_width), f0 + half_width


def _monitor_band_issues(vctx, physics_spec, physics_type, sources):
    """Warning per enabled spectral monitor with wavelengths outside the union of the enabled sources' _source_band:
    the sources put almost no power there (cw: only switch-on leakage), so normalised results there are noise."""
    duration = study_duration(vctx) if vctx.study_type == 'time_dependent' else None
    duration = duration if isinstance(duration, (int, float)) and duration > 0 else None
    bands = [(feature, source_values, band) for feature, source_values in sources
             if (band := _source_band(source_values, duration)) is not None]
    if not bands:
        return []
    band_texts = [f"'{feature.name}' ({source_values['waveform']}) {C0 / f_hi:.4g} to "
                  f"{f'{C0 / f_lo:.4g}' if f_lo > 0 else 'inf'} m" for feature, source_values, (f_lo, f_hi) in bands]
    has_cw = any(source_values['waveform'] == 'cw' for _, source_values, _ in bands)
    issues = []
    for feature, feature_type in _enabled_features(physics_spec, physics_type):
        if not any(param.name == 'wavelength_start' for param in feature_type.params):
            continue
        values = vctx.values(feature_type.params, feature.params)
        if values is None or not all(isinstance(values.get(name), (int, float)) and values[name] > 0
                                     for name in ('wavelength_start', 'wavelength_stop', 'n_wavelengths')):
            continue   # reported by the generic param check / wavelength_range_issues
        monitor_frequencies = C0 / _monitor_wavelengths(values)
        in_band = np.zeros(monitor_frequencies.shape, dtype=bool)
        for _, _, (f_lo, f_hi) in bands:
            in_band |= (monitor_frequencies >= f_lo) & (monitor_frequencies <= f_hi)
        if in_band.all():
            continue
        outside_wavelengths = C0 / monitor_frequencies[~in_band]
        cw_note = (' (a cw source drives only its own wavelength, within 1 / duration in frequency; elsewhere its '
                   'spectrum is switch-on leakage)' if has_cw else '')
        issues.append(Issue('warning', feature_path(physics_spec, feature),
                            f'{outside_wavelengths.size} of {monitor_frequencies.size} monitor wavelengths (between '
                            f'{outside_wavelengths.min():.4g} and {outside_wavelengths.max():.4g} m) lie outside the '
                            f'spectrum of the enabled sources (amplitude >= {_IN_BAND_LEVEL:g} of its peak: '
                            f"{'; '.join(band_texts)}){cw_note}, so the sources put almost no power there and "
                            f'normalised results (fractions, phases, flux ratios) there are noise; narrow the monitor '
                            f'range or widen the source band'))
    return issues


def make_validate_physics(slot_arrays, time_domain=True, solved_wavelengths=None):
    """validate(vctx, physics_spec, settings_values) of an EM type: periodic pairing, every == 1 (time_domain),
    materials, slot writers, resolution (FDTD: at the shortest_wavelength hook, the auto mesh's; FDFD: at every
    wavelength of the solved_wavelengths hook, same signature, -> list | None), wavelength unit slips, duration vs
    pulse and monitor wavelengths vs source band (time_domain)."""
    def validate_physics(vctx, physics_spec, settings_values):
        path = ('physics', physics_spec.name)
        issues = _boundary_issues(settings_values, path)
        # a non-integer / < 1 every is reported by the generic check; without time_domain nothing steps in time
        if time_domain and isinstance(physics_spec.every, int) and physics_spec.every > 1:
            issues.append(Issue('error', path, f'EM advances one Courant step per runner tick and must use every = '
                                               f'1, got every = {physics_spec.every}'))
        first_em_name, em_types = _em_physics_types(vctx)
        if first_em_name == physics_spec.name:   # model-wide checks reported once
            issues.extend(_material_issues(vctx, em_types))
        physics_type = vctx.registry.physics_type(physics_spec.type)
        material_property_params = physics_type.material_properties
        issues.extend(_permittivity_writer_issues(vctx, physics_spec, slot_arrays, material_property_params))
        if solved_wavelengths is None:
            shortest = physics_type.shortest_wavelength(vctx.spec, physics_spec, settings_values, vctx.parameters,
                                                        vctx.registry)
            wavelengths = [] if shortest is None else [shortest]
        else:
            wavelengths = solved_wavelengths(vctx.spec, physics_spec, settings_values, vctx.parameters,
                                             vctx.registry) or []
            shortest = min(wavelengths) if wavelengths else None
        wavelength_label = 'shortest source wavelength' if time_domain else 'shortest solved wavelength'
        issues.extend(_resolution_issues(vctx, physics_spec, wavelengths, material_property_params, time_domain))
        if time_domain:
            sources = _source_values(vctx, physics_spec, physics_type)
            issues.extend(_wavelength_unit_issues(vctx, [
                (feature_path(physics_spec, feature), f"source '{feature.name}' wavelength", values['wavelength'])
                for feature, values in sources]))
            duration = _duration_issue(vctx, physics_spec, physics_type, sources, material_property_params)
            if duration is not None:
                issues.append(duration)
            issues.extend(_monitor_band_issues(vctx, physics_spec, physics_type, sources))
        elif isinstance(shortest, (int, float)) and shortest > 0:
            checked = [(path, wavelength_label, shortest)]
            longest = max(wavelengths)
            if longest != shortest and longest >= _MIN_PLAUSIBLE_WAVELENGTH:   # only its diagonal bound is news
                checked.append((path, 'longest solved wavelength', longest))
            issues.extend(_wavelength_unit_issues(vctx, checked))
        return issues
    return validate_physics

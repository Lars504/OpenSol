"""Model-layer definition of the 'elastic_waves' physics type (settings, features, material vocabulary, factory,
validation, memory estimate) plus its template and selftests. Loaded lazily by the registry (physics/acoustics/__init__
must not import it)."""
import math
from dataclasses import replace

import numpy as np

from ...core.constants import DTYPE, EPS0, FACE_LABELS, FACES
from ...core.memory_estimate import BOOL_BYTES, FLOAT32_BYTES, INT32_BYTES, MemoryEstimate
from ...core.time_estimate import explicit_step_estimate
from ...core.mirror import MIRROR_SIGNS, mirror_faces
from ...core.monitors import probe_result
from ...model.definition_helpers import (AXISYMMETRIC, AXISYMMETRIC_SPACE_SETTINGS, HALF_STORAGE_STEPPER_ROLE,
                                         axisymmetric_issues, geometry_param, is_axisymmetric, optional_material_param,
                                         parkable_material_arrays, plain_number, point_cells, reports_model_checks,
                                         resolved_stepper_precision, selected_material_data, study_duration,
                                         validate_points_inside)
from ...model.definition_resonance import ResonanceProbe, point_trace_items, resonance_monitor_feature
from ...model.expressions import Expression, ExpressionError
from ...model.geometry import line_orientation
from ...model.schema import TIME_DEPENDENT_STUDY, FeatureType, Issue, Param, PhysicsType, feature_path, known_physics
from ..mechanics.material_params import (PHOTOELASTIC_P11_PARAM, PHOTOELASTIC_P12_PARAM, POISSON_RATIO_PARAM,
                                         YOUNGS_MODULUS_PARAM)
from .axisymmetric import KERNEL_ROWS, radial_stability_factor
from ...solvers.explicit_kernels import stepper_precision
from .renderer import AcousticRenderer
from .selftests import PHOTOACOUSTIC_DESCRIPTION, PHOTOACOUSTIC_SUMMARY, photoacoustic_disc_spec
from .solver import (ACOUSTIC_PROPERTY_ARRAYS, ACOUSTIC_SCHEMA, BOUNDARY_TYPES, COEFFICIENT_INPUTS,
                     ELECTRIC_FIELD_SLOTS, HEATING_SLOT, MONOCLINIC_ARRAYS, OPTIONAL_ACOUSTIC_SCHEMA, PARKABLE_ARRAYS,
                     PHOTOELASTIC_TENSOR_ARRAYS, PIEZO_ARRAYS, STEP_GRAPHS, AcousticPhysics)
from .sources import PROBE_COMPONENTS, VELOCITY_COMPONENTS, HeatingPulse, make_waveform

PHYSICS_TYPE_NAME = 'elastic_waves'
PHYSICS_NAME = 'acoustics'   # = selftests.PHYSICS_NAME (selftests never import this)
PIEZO_COUPLING_TYPE_NAME = 'electrostatics_acoustics_piezo'   # = coupling/definitions_piezo (which imports physics)
ACOUSTO_OPTIC_COUPLING_TYPE_NAME = 'acoustics_em_acousto_optic'   # = coupling/selftests_acoustics (shear strain reader)
PHOTOACOUSTIC_COUPLING_TYPE_NAME = 'em_acoustics_photoacoustic'   # = coupling/definitions_acoustics
_OFF_DIAGONAL_EM_TYPE = 'em_fdfd_hz'   # the EM type with an eps_xy slot: the isotropic shear term reads the shear too
MIN_CELLS_PER_WAVELENGTH = 10   # resolution warning (shortest wavelength at the highest significant source frequency)


def _target_type(spec, physics_name):
    """Type name of a spec physics, None when it is unknown (reported elsewhere)."""
    try:
        return spec.physics_by_name(physics_name).type
    except KeyError:
        return None

# ---- settings ----
# 'symmetric' = the rigid slip wall that also makes the face a model mirror plane (core.mirror.own_mirror_faces reads
# the value; there is no 'antisymmetric' face: the odd mirror v_t = 0, s_nn = 0 needs samples on the lo face); the
# solver takes it as 'rigid' (_solver_faces)
_SYMMETRIC_FACE = 'symmetric'
_BOUNDARY_CHOICES = (*BOUNDARY_TYPES, _SYMMETRIC_FACE)
_BC_HELP = ('absorbing: CPML in the outermost pml_cells cells, backed by a rigid wall; these lie inside the domain unless '
            'another physics (an EM CPML) pads the grid there: keep sources, probes and objects out (validation '
            'warns). rigid: slip wall (normal velocity and shear stress 0; also the even mirror on a plane another '
            'physics sets), also where another physics pads the grid (the padding stays idle). symmetric: the same '
            'slip wall as a model mirror plane: fields unfold across it (v normal to it and sxy odd, stresses and '
            'pressure even), field derived integrals reaching it count twice, and another physics with mirror bc '
            'settings (EM) must set the face symmetric / antisymmetric; there is no antisymmetric elastic face, so '
            'parity_sectors cannot swap it (use rigid for a face only an EM plane mirrors); not on x_lo / x_hi of '
            'geometry axisymmetric. periodic: wraps to the opposite face (set both; an error where another physics '
            'pads the grid). A traction-free surface is a material boundary: cells without density and stiffness are '
            'vacuum. geometry axisymmetric: x_lo must be rigid (the axis: v_r = 0, s_rz = 0; it unfolds for display '
            'without a mirror); a rigid x_hi is a slip wall on r = r_max, an absorbing one a radial CPML.')
_BC_TIP = ('What happens at this edge of the domain.\n\n'
           '- Absorbing: waves leave without reflection through a damping layer inside the outer cells.\n'
           '- Rigid wall: the edge does not move; fluids slide along it.\n'
           '- Symmetry plane: a rigid wall that is also a mirror, so only half of a symmetric structure needs to be '
           'drawn.\n'
           '- Periodic: the domain repeats; set the opposite edge to periodic too.')
_GEOMETRY_PARAM = geometry_param(
    'plane strain per unit z length',
    '(r, z) waves without torsion (v_phi = 0); the hoop stress s_tt (field and probe component stt) is stepped with '
    'c_tt = c11, c_rt = c_zt = c12 (exact for isotropic solids and fluids, approximate for c_ij crystals: validation '
    'warns); an x-face density is the ring-volume mean of its two cells. The stiffness must be positive definite in '
    '3D (isotropic: shear_speed < 0.866 sound_speed). Point sources are per unit ring length (the axis cell is a ring '
    'of radius dx / 2), traction sheets per area (a vertical line is a cylinder, a horizontal one a disc or annulus); '
    'pressure = -(s_rr + s_zz + s_tt) / 3; the stable step is a few % shorter (see courant); an absorbing x_hi '
    'stretches only the radial derivatives',
    'forces and volume rates are per metre of depth.',
    'Forces and volume rates are per metre along the ring a point stands for. Here the Left edge setting must be '
    'Rigid wall, which is the axis condition.',
    axis_text='bc_x_lo rigid and no other physics padding the grid at x_lo, else validation errors')
_SETTINGS = (_GEOMETRY_PARAM,) + tuple(
    Param(f'bc_{face}', 'choice', 'absorbing', label=FACE_LABELS[face], choices=_BOUNDARY_CHOICES,
          choice_labels=(('absorbing', 'Absorbing'), ('rigid', 'Rigid wall'), ('periodic', 'Periodic'),
                         (_SYMMETRIC_FACE, 'Symmetry plane')),
          help=_BC_HELP, tip=_BC_TIP)
    for face in FACES) + (
    Param('pml_cells', 'int', 20, label='Absorbing layer thickness', unit='cells', min=2,
          help='CPML cells per absorbing layer (polynomial grading of order 2, kappa 1); typically 10-30, about one '
               'wavelength at the lowest frequency of interest.',
          tip='Thickness of the damping layer at absorbing edges, in cells.\n\n'
              'About 20 cells, or one wavelength, is typical. The layer lies inside the domain.'),
    Param('pml_reflection', 'float', 1e-4, label='Absorbing layer reflection', unit='1', min=0.0, min_exclusive=True,
          max=0.5,
          help='Target normal-incidence reflection R: peak damping d0 = 3 c ln(1 / R) / (2 L), c the largest '
               'longitudinal speed, L the layer thickness. 1e-3..1e-6 typical; thin layers with very small R reflect '
               'from their own grading.',
          tip='How little an absorbing layer should reflect; 1e-4 is typical.'),
    Param('courant', 'float', 0.9, label='Time step factor', unit='1', min=0.0, min_exclusive=True, max=1.0,
          help='dt = courant / (c_bound sqrt(1 / dx_min^2 + 1 / dy_min^2)) from the build materials, c_bound^2 = the '
               'largest stability stiffness S of a cell times the largest 1 / rho of its faces (face density = mean '
               'of its two cells: a solid next to vacuum has rho / 2). S = the largest eigenvalue of [[c11 wx + c66 '
               'wy, |c12 + c66| sqrt(wx wy)], [same, c66 wx + c22 wy]], wx = (1 / dx_min^2) / (1 / dx_min^2 + 1 / '
               'dy_min^2), wy = 1 - wx, at least max(c11, c22): c11 for isotropic solids and fluids, more for '
               'crystals (silicon [100]: 194 GPa on square cells vs c11 = 166 GPa); courant 1 is the exact limit of '
               'a uniform material. c16 / c26 enter by absolute value (diagonals + 2 |c16| s, + 2 |c26| s, '
               'off-diagonal + |c16| wx + |c26| wy, s = sqrt(wx wy)), a bound: dt ~2 % short of exact for YZ lithium '
               'niobate, ~8 % with c16 / c26 of 10 % of c11. Two-way piezo (back_action bound_charge): S of the '
               'stiffened c^D = c^E + e^T (eps0 eps_dc)^-1 e (a bound). geometry axisymmetric: 1 / dx_min^2 times '
               'F >= 1 (~1.047 on uniform cells), a bound: keep courant <= 0.95. Runtime material rewrites '
               '(state-dependent speeds) are not stability-checked: keep their speeds below the build values.',
          tip='Time step as a fraction of the largest stable step; 0.9 is typical.'),
    Param('heating_envelope', 'choice', 'instantaneous', label='Coupled heating envelope',
          choices=('instantaneous', 'gaussian'),
          choice_labels=(('instantaneous', 'As delivered'), ('gaussian', 'Gaussian pulse')),
          help=f"Time dependence of the coupled '{HEATING_SLOT}' slot: instantaneous = the delivered array is the "
               'heating rate (held until the next transfer); gaussian = the array (e.g. a cw absorbed power map) x '
               'exp(-((t - heating_pulse_delay) / heating_pulse_width)^2), a pulse of energy density array x sqrt(pi) '
               'width.',
          tip='How a heating rate passed from another physics varies in time.\n\n'
              '- As delivered: used as it arrives.\n'
              '- Gaussian pulse: multiplied by a short pulse, e.g. to turn a steady absorbed light power into a laser '
              'pulse.'),
    Param('heating_pulse_width', 'float', 1e-8, label='Heating pulse width', unit='s', min=0.0, min_exclusive=True,
          visible_if=(('heating_envelope', ('gaussian',)),),
          help='1/e half-width w; the pressure rise resolves only if c w spans a few cells.',
          tip='Half-width of the heating pulse. Sound travels about 1.5 um per ns in water: make the pulse span a '
              'few cells.'),
    Param('heating_pulse_delay', 'float', None, label='Heating pulse delay', unit='s', min=0.0, optional=True,
          visible_if=(('heating_envelope', ('gaussian',)),),
          help='Envelope centre time; empty = 3 x heating_pulse_width.',
          tip='Time of the pulse maximum; empty means three pulse widths.'),
)

# ---- material vocabulary ----
_STIFFNESS_TEXT = ('In-plane plane-strain stiffnesses c11, c12, c22, c66 (Voigt x = 1, y = 2, xy = 6); set, they override '
                   'the values from sound_speed / shear_speed and rho (or youngs_modulus / poisson_ratio). Isotropic: '
                   'c11 = c22 = lambda + 2 mu, c12 = lambda, c66 = mu; silicon along [100] / [010]: c11 = c22 = '
                   '165.7e9, c12 = 63.9e9, c66 = 79.6e9. The effective set with c16 / c26 must be positive definite '
                   '(c11, c22, c66 > 0, c12^2 < c11 c22, det [[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]] > 0; '
                   'a fluid: c66 = c16 = c26 = 0, c12 = c11 = c22), else validation errors.')
_STIFFNESS_SEE = 'Plane-strain stiffness; overrides the value from the speeds or E / nu. See c11.'
_MONOCLINIC_TEXT = (
    'In-plane monoclinic stiffnesses c16 / c26 (Voigt: T_xx = c11 S_xx + c12 S_yy + c16 g, T_yy = c12 S_xx + c22 S_yy '
    '+ c26 g, T_xy = c16 S_xx + c26 S_yy + c66 g, g = 2 S_xy); default 0 (none from the speeds or E / nu). Non-zero '
    "when the model axes are not crystal symmetry axes in the plane, e.g. YZ lithium niobate c26 = c14 (a {'model': "
    "'crystal'} entry sets both). Energy-conserving corner / centre 4-point means; corners without shear stiffness "
    '(fluids, vacuum, slip walls) carry none. The set must stay positive definite (see c11); dt takes |c16|, |c26| as '
    'bounds (see courant). Either one allocates 8 float32 grids (both arrays, strain accumulator, rate buffers); '
    'elastic_strain_xx / _yy then show the accumulated total strain. Planar geometry and single-precision stepper only. '
    'CPML can be unstable in some strongly anisotropic media (Becache, Fauqueux and Joly 2003): check long runs for '
    'growth near the layers.')
_MONOCLINIC_TEXTS = {MONOCLINIC_ARRAYS[0]: _MONOCLINIC_TEXT,
                     **{name: f'Monoclinic stiffness; see {MONOCLINIC_ARRAYS[0]}.' for name in MONOCLINIC_ARRAYS[1:]}}


_PHOTOELASTIC_FALLBACKS = {'photoelastic_p16': '0', 'photoelastic_p21': 'photoelastic_p12',
                           'photoelastic_p22': 'photoelastic_p11', 'photoelastic_p26': '0',
                           'photoelastic_p31': 'photoelastic_p12', 'photoelastic_p32': 'photoelastic_p12',
                           'photoelastic_p36': '0', 'photoelastic_p61': '0', 'photoelastic_p62': '0',
                           'photoelastic_p66': '(photoelastic_p11 - photoelastic_p12) / 2'}
# light polarisation of row I (6: the xy cross term)
_PHOTOELASTIC_ROWS = {'1': 'x', '2': 'y', '3': 'z (out of the plane)', '6': 'x and y together (the cross term)'}
_PHOTOELASTIC_COLUMNS = {'1': 'stretching along x', '2': 'stretching along y', '6': 'in-plane shear'}
_PHOTOELASTIC_TENSOR_TEXT = (
    'In-plane photoelastic tensor photoelastic_p16 .. _p66 in model axes, read only by acoustics_em_acousto_optic: '
    'Delta(1/n^2)_I = p_I1 S_xx + p_I2 S_yy + p_I6 g, index-ellipsoid row I = 1 xx, 2 yy, 3 zz (the diagonal '
    'permittivity slots) or 6 xy (the eps_xy slot of em_fdfd_hz: Delta eps_xy = -n0_xx^2 n0_yy^2 Delta(1/n^2)_xy), g = '
    '2 S_xy at the cell centre (4-corner mean); photoelastic_p11 / _p12 are row 1. Unset = the isotropic value (here '
    "{fallback}). Usually set by a {{'model': 'crystal'}} entry (only when the cut is not isotropic in these rows); "
    'YZ lithium niobate (model x = crystal Z, y = -Y): p11 = p33, p12 = p31, p16 = 0, p21 = p13, p22 = p11, p26 = '
    'p14, p31 = p13, p32 = p12, p36 = -p14, p61 = 0, p62 = p41, p66 = p44. Any one set allocates all ten arrays.')


def _photoelastic_text(name):
    """Full text on the first tensor entry, a pointer plus its isotropic fallback on the others."""
    first = PHOTOELASTIC_TENSOR_ARRAYS[0]
    if name == first:
        return _PHOTOELASTIC_TENSOR_TEXT.format(fallback=_PHOTOELASTIC_FALLBACKS[name])
    return f'Photoelastic tensor entry (see {first}); unset = {_PHOTOELASTIC_FALLBACKS[name]}.'


_PIEZO_TEXT = (
    'Piezoelectric stress constants e_kJ (piezo_e11 .. _e26) of the in-plane block in model axes (reduced Voigt: D_k '
    '= e_kJ S_J + eps0 eps_dc E, T_J = c_JK S_K - e_kJ E_k, k = 1 x / 2 y, J = 1 xx, 2 yy, 6 xy, S_6 = 2 S_xy); '
    'default 0. Read through electrostatics_acoustics_piezo: the stress -e^T E of the electric_field_x / _y slots '
    'and, two-way (default), the bound charge -div(e S) of the accumulated strain back into the electrostatics plus '
    "the stiffened dt. Usually set by a {'model': 'crystal'} entry of the material's dispersion list (also sets c_ij, "
    'rho and the clamped eps_dc_xx / _yy / _xy); YZ lithium niobate (euler_deg 0, 90, 90; model x = crystal Z, y = '
    '-Y): piezo_e11 = e33 = 1.3, piezo_e12 = e31 = 0.2, piezo_e22 = -e22 = -2.5, piezo_e26 = e15 = 3.7. Any one set '
    'allocates all six arrays and the strain accumulator (9 float32 grids; elastic_strain_xx / _yy then show the '
    'accumulated total strain). Planar geometry only.')
_PIEZO_TEXTS = {PIEZO_ARRAYS[0]: _PIEZO_TEXT,
                **{name: f'Piezoelectric stress constant; see {PIEZO_ARRAYS[0]}.' for name in PIEZO_ARRAYS[1:]}}


def _material_property(name, label, unit, minimum=None, help_text='', tip=''):
    return optional_material_param(name, label, unit, ACOUSTIC_PROPERTY_ARRAYS[name], minimum, help_text, tip)


_MATERIAL_PROPERTIES = (
    _material_property('rho', 'Density', 'kg/m^3', 0.0,
                       'Mass density (inertia; also read by pressure_acoustics). Unset = 0: elastic_waves treats a '
                       'cell without density and stiffness as vacuum (a traction-free surface; a face between two '
                       'such cells does not move). Typical: water 998, silica 2200, silicon 2329, aluminium 2700, '
                       'gold 19300.',
                       tip='Mass density.\n\nTypical: water 998, silica 2200, silicon 2329, gold 19300.'),
    _material_property('sound_speed', 'Sound speed (longitudinal)', 'm/s', 0.0,
                       'Longitudinal (P) wave speed: c11 = c22 = rho sound_speed^2 unless c_ij are set; a fluid sets '
                       'only this. Set (> 0), the speeds replace the youngs_modulus / poisson_ratio of the cell '
                       '(library solids carry those). pressure_acoustics: the fluid sound speed c. Typical: air 343, '
                       'water 1480, silica 5960, aluminium 6420, silicon ~8430 (isotropic average).',
                       tip='Speed of compression waves.\n\nTypical: air 343, water 1480, silica 5960, aluminium 6420.'),
    _material_property('shear_speed', 'Shear speed', 'm/s', 0.0,
                       'Transverse (S) wave speed: c66 = rho shear_speed^2, c12 = rho (sound_speed^2 - 2 '
                       'shear_speed^2) unless c_ij are set; unset = 0 = a fluid. Must be below sound_speed. Typical: '
                       'silica 3760, aluminium 3040, silicon ~5840.',
                       tip='Speed of shear waves; leave empty for liquids and gases.\n\nTypical: silica 3760, '
                           'aluminium 3040.'),
    _material_property('c11', 'Stiffness c11', 'Pa', 0.0, _STIFFNESS_TEXT,
                       tip='Stiffness along x for a crystal; replaces the value from the sound speed.'),
    _material_property('c12', 'Stiffness c12', 'Pa', None, _STIFFNESS_SEE,
                       tip='Coupling stiffness between x and y for a crystal; replaces the value from the speeds.'),
    _material_property('c22', 'Stiffness c22', 'Pa', 0.0, _STIFFNESS_SEE,
                       tip='Stiffness along y for a crystal; replaces the value from the sound speed.'),
    _material_property('c66', 'Stiffness c66', 'Pa', 0.0, _STIFFNESS_SEE,
                       tip='Shear stiffness for a crystal; replaces the value from the shear speed.'),
    *(_material_property(name, f'Stiffness {name}', 'Pa', None, _MONOCLINIC_TEXTS[name],
                         tip=f"Coupling between stretching along {'x' if name == 'c16' else 'y'} and shear, for a "
                             'crystal cut at an angle to its axes. Usually filled in by a crystal entry; empty means '
                             'none.')
      for name in MONOCLINIC_ARRAYS),
    YOUNGS_MODULUS_PARAM,
    POISSON_RATIO_PARAM,
    _material_property('acoustic_damping', 'Acoustic damping rate', '1/s', 0.0,
                       'Mass-proportional damping g of the particle velocity (rho dv/dt = div s - rho g v, '
                       'semi-implicit): a wave with omega >> g decays as exp(-g t / 2). Default 0.',
                       tip='Damping of the motion per second; empty means lossless.'),
    _material_property('gruneisen', 'Grueneisen parameter', '1', None,
                       'Thermoelastic G = beta c^2 / cp (beta the volumetric expansion): a heating rate q [W/m^3] '
                       'adds -G q to d(sxx)/dt and d(syy)/dt, so an instantly '
                       'absorbed energy density H raises the pressure by G H. Read by heat_pulse_source and the '
                       f"'{HEATING_SLOT}' slot; default 0. Typical: water 0.11 at 20 C, soft tissue ~0.2.",
                       tip='How strongly heating turns into pressure (photoacoustics).\n\nTypical: water 0.11, soft '
                           'tissue about 0.2.'),
    PHOTOELASTIC_P11_PARAM,
    PHOTOELASTIC_P12_PARAM,
    *(_material_property(name, f'Photoelastic coefficient {name[-3:]}', '1', None,
                         _photoelastic_text(name),
                         tip=f'How strongly {_PHOTOELASTIC_COLUMNS[name[-1]]} changes the refractive index seen by '
                             f'light polarised along {_PHOTOELASTIC_ROWS[name[-2]]}, for a crystal. Usually filled '
                             'in by a crystal entry; empty means the value of an isotropic material.')
      for name in PHOTOELASTIC_TENSOR_ARRAYS),
    *(_material_property(
        name, f'Piezoelectric e{name[-2:]}', 'C/m^2', None, _PIEZO_TEXTS[name],
        tip=f'Piezoelectric constant e{name[-2:]}: how strongly an electric field along '
            f"{'x' if name[-2] == '1' else 'y'} stresses the material. Usually filled in by a crystal entry.")
      for name in PIEZO_ARRAYS),
)

# ---- feature params ----
_WAVEFORM_PARAMS = (
    Param('waveform', 'choice', 'ricker', label='Waveform', choices=('ricker', 'gaussian', 'cw'),
          choice_labels=(('ricker', 'Ricker wavelet'), ('gaussian', 'Gaussian pulse'), ('cw', 'Continuous wave')),
          help='ricker: Ricker wavelet, spectrum peaking at frequency, value 1 at t0 = 4 / frequency (broadband: 3 % '
               'of the spectral peak at 2.5 f, 1 % at 2.76 f); gaussian: sin carrier under a gaussian envelope of '
               'relative bandwidth, t0 = 4.5 / (pi bandwidth frequency); cw: sin(2 pi f t) ramped in by 1 - exp(-t / '
               '(3 T)), T = 1 / f, steady after ~15 periods.',
          tip='Time shape of the source.\n\n- Ricker wavelet: one short, broadband pulse.\n- Gaussian pulse: a few '
              'oscillations at the frequency.\n- Continuous wave: a steady tone.'),
    Param('frequency', 'float', 1e6, label='Frequency', unit='Hz', min=0.0, min_exclusive=True,
          help='Centre (gaussian, cw) or peak (ricker) frequency. Mesh: >= 10 (better 20) cells per shortest '
               'wavelength = slowest speed (shear in solids) / highest significant frequency (gaussian f (1 + '
               'bandwidth), ricker 2.5 f); validation warns.',
          tip='Frequency of the source in hertz; 1e6 is 1 MHz.'),
    Param('bandwidth', 'float', 0.5, label='Relative bandwidth', unit='1', min=0.0, min_exclusive=True,
          visible_if=(('waveform', ('gaussian',)),),
          help='Spectral width / frequency: the amplitude spectrum is 1/e at f (1 +- bandwidth).',
          tip='Width of the pulse spectrum relative to the frequency; 0.5 is typical.'),
)
_FORCE_PARAMS = _WAVEFORM_PARAMS + (
    Param('force', 'float', 1.0, label='Line force', unit='N/m',
          help='Peak force per unit z length (geometry axisymmetric: per unit length of the ring through the point, '
               'total 2 pi r_c force).',
          tip='Force per metre of depth applied at the point.'),
    Param('angle', 'float', 0.0, label='Force direction', unit='deg',
          help='Counter-clockwise from +x (90 = +y).',
          tip='Direction of the force in degrees from the x axis, counter-clockwise.'),
)
_VOLUME_PARAMS = _WAVEFORM_PARAMS + (
    Param('volume_rate', 'float', 1e-3, label='Volume injection rate', unit='m^2/s',
          help='Peak volume per second and unit z length: d(sxx)/dt -= (c11 + c12) / 2 q, d(syy)/dt -= (c12 + c22) / '
               '2 q, q = rate / cell area. geometry axisymmetric: per unit length of the ring through the point '
               '(total 2 pi r_c rate), a 3D strain rate q / 3 per axis (a fluid again gets p += K q dt).',
          tip='Volume pushed out per second and per metre of depth, like a small pulsating cylinder.'),
)
_TRACTION_PARAMS = _WAVEFORM_PARAMS + (
    Param('traction', 'float', 1e3, label='Traction', unit='Pa',
          help='Peak force per area; a domain-spanning sheet sends a plane wave of stress traction / 2 and velocity '
               'traction / (2 rho c) to each side (fluid: pressure traction / 2 for a normal traction).',
          tip='Force per area of the line; each side receives half of it as a plane wave.'),
    Param('direction', 'choice', 'x', label='Force direction', choices=('x', 'y'),
          choice_labels=(('x', 'Along x'), ('y', 'Along y')),
          help='Normal to the line launches P waves, along it S waves (vertical line: x = P, y = S).',
          tip='Direction of the force; across the line it sends compression waves, along the line shear waves.'),
)
_HEAT_PULSE_PARAMS = (
    Param('power_density', 'float', 1e12, label='Peak heating power density', unit='W/m^3', field_dependent=True,
          help='Peak q0: q = q0 exp(-((t - delay) / pulse_width)^2), absorbed energy density H = q0 sqrt(pi) '
               'pulse_width; a short pulse (c pulse_width << heated size) raises the pressure by gruneisen H. E.g. '
               'Beer-Lambert exp(-mu_a (y1 - y)).',
          tip='Heat deposited per volume and second at the pulse maximum.'),
    Param('pulse_width', 'float', 1e-8, label='Pulse width', unit='s', min=0.0, min_exclusive=True,
          help='1/e half-width; keep c pulse_width >= ~5 cells (validation warns below 2), else the front is '
               'under-resolved and rings.',
          tip='Half-width of the heating pulse.'),
    Param('delay', 'float', None, label='Pulse delay', unit='s', min=0.0, optional=True,
          help='Envelope maximum; empty = 3 x pulse_width.',
          tip='Time of the pulse maximum; empty means three pulse widths.'),
)
_PROBE_UNITS = {component: ('m/s' if component in VELOCITY_COMPONENTS else 'Pa') for component in PROBE_COMPONENTS}
_PROBE_PARAMS = (
    Param('component', 'choice', 'pressure', label='Recorded quantity', choices=PROBE_COMPONENTS,
          choice_labels=(('pressure', 'Pressure'), ('vx', 'Velocity x'), ('vy', 'Velocity y'),
                         ('sxx', 'Normal stress xx'), ('syy', 'Normal stress yy'), ('sxy', 'Shear stress xy'),
                         ('stt', 'Hoop stress (axisymmetric)')),
          help='pressure = -(sxx + syy) / 2 (axisymmetric: -(s_rr + s_zz + s_tt) / 3) and the stresses [Pa] at the '
               'cell centre (sxy on its +x +y corner) at t; stt (hoop stress) axisymmetric only, else a validation '
               'error; vx / vy [m/s] on the +x / +y face at t - dt / 2.',
          tip='Quantity recorded at the point after every time step. The hoop stress exists only in axisymmetric '
              'models.'),
)


# ---- apply / result callbacks (worker thread, GPU) ----
def _waveform(values):
    return make_waveform(values['waveform'], values['frequency'], values.get('bandwidth') or 0.5)


def _apply_force_source(ctx, solver, feature_spec, values, selection):
    angle = math.radians(values['angle'])
    for ix, iy in point_cells(ctx, feature_spec, selection):
        solver.add_force_source(ix, iy, _waveform(values), values['force'] * math.cos(angle),
                                values['force'] * math.sin(angle))
    return None


def _apply_volume_source(ctx, solver, feature_spec, values, selection):
    for ix, iy in point_cells(ctx, feature_spec, selection):
        solver.add_volume_source(ix, iy, _waveform(values), values['volume_rate'])
    return None


def _line_cells(domain, x0, y0, x1, y1):
    """(line_axis, position cell, (start, stop)) of an axis-aligned line: [cell(lo), cell(hi) + 1) along it, extended
    through any padding where an end touches the domain edge; ValueError for a slanted line."""
    orientation = line_orientation({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}, domain.size_x, domain.size_y)
    if orientation is None:
        raise ValueError('the line must be horizontal or vertical')
    tolerance = 1e-9 * max(domain.size_x, domain.size_y)
    if orientation == 'vertical':
        lo, hi, size, n_cells = min(y0, y1), max(y0, y1), domain.size_y, domain.grid.ny
        start, stop = domain.pos_to_cell_y(lo), domain.pos_to_cell_y(hi) + 1
        line_axis, position = 'y', domain.pos_to_cell_x(0.5 * (x0 + x1))
    else:
        lo, hi, size, n_cells = min(x0, x1), max(x0, x1), domain.size_x, domain.grid.nx
        start, stop = domain.pos_to_cell_x(lo), domain.pos_to_cell_x(hi) + 1
        line_axis, position = 'x', domain.pos_to_cell_y(0.5 * (y0 + y1))
    if lo <= tolerance:
        start = 0
    if hi >= size - tolerance:
        stop = n_cells
    return line_axis, position, (start, stop)


def _apply_traction_line(ctx, solver, feature_spec, values, selection):
    line_axis, position, span = _line_cells(ctx.domain, *ctx.line(selection.objects[0]))
    solver.add_traction_line(line_axis, position, span, _waveform(values), values['traction'], values['direction'])
    return None


def _apply_heat_pulse(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    power_density = ctx.eval_field(values['power_density'], mask=mask)
    solver.add_heat_pulse(mask, power_density, HeatingPulse(values['pulse_width'], values['delay']))
    return None


def _apply_rigid(ctx, solver, feature_spec, values, selection):
    solver.add_rigid(ctx.domain_mask(selection.entities))
    return None


def _face_velocity(ctx, value, axis):
    """Float, or a full-grid float32 array of an x, y expression at the staggered position of the velocity along
    `axis`: the +x face centre (x_(i+1/2), y_j) of every cell for 'x', the +y face centre for 'y' (coordinates clamped
    to the physical window)."""
    if not isinstance(value, Expression) or value.names.issubset(ctx.parameters):
        return ctx.eval_scalar(value)
    domain = ctx.domain
    (x_edges, y_edges), (x_centres, y_centres) = domain.cell_edges(), domain.cell_centers()
    x_min, x_max, y_min, y_max = domain.physical_extent()
    x_points = np.clip(x_edges[1:] if axis == 'x' else x_centres, x_min, x_max)
    y_points = np.clip(y_centres if axis == 'x' else y_edges[1:], y_min, y_max)
    xp = ctx.xp
    namespace = dict(ctx.parameters)
    namespace['x'] = xp.asarray(x_points, dtype=DTYPE)[:, None]
    namespace['y'] = xp.asarray(y_points, dtype=DTYPE)[None, :]
    values = xp.empty((domain.grid.nx, domain.grid.ny), dtype=DTYPE)
    values[...] = value.evaluate(namespace, xp=xp)
    if not bool(xp.isfinite(values).all()):   # one host sync, build time only
        raise ExpressionError(f'{value.text!r} gives non-finite velocities')
    return values


def _apply_initial_velocity(ctx, solver, feature_spec, values, selection):
    """v_x / v_y at t = 0 on the +x / +y faces of the selected cells (evaluated at the face centres)."""
    solver.set_initial_velocity(ctx.domain_mask(selection.entities), _face_velocity(ctx, values['v_x'], 'x'),
                                _face_velocity(ctx, values['v_y'], 'y'))
    return None


def _apply_probe(ctx, solver, feature_spec, values, selection):
    cells = point_cells(ctx, feature_spec, selection)
    if len(cells) != 1:
        raise ValueError(f"probe '{feature_spec.name}' needs exactly one point object, got {len(cells)}")
    return solver.add_probe(*cells[0], values['component'])


def _probe_result(session, solver, probe, feature_spec, options):
    return probe_result(probe, unit=_PROBE_UNITS[probe.field_component])


# ---- resonance_monitor (model.definition_resonance) ----
def _solver_time_step(solver):
    return float(solver.dt)


def _solver_waveforms(solver):
    return solver.source_waveforms()


def _component_unit(solver, component):
    return _PROBE_UNITS[component]


def _source_waveforms(vctx, physics_spec):
    """(pulse ends [s] of the enabled pulsed sources, heating pulses and a gaussian heating_envelope, names of the
    enabled cw sources)."""
    pulse_ends, cw_names = [], []
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in _PULSED_FEATURES:
            continue
        values = vctx.values(_FEATURES_BY_NAME[feature.type].params, feature.params)
        if not values:
            continue
        if feature.type == 'heat_pulse_source':
            if values['pulse_width'] > 0.0:
                pulse_ends.append(HeatingPulse(values['pulse_width'], values['delay']).duration)
        elif values['waveform'] == 'cw':
            cw_names.append(feature.name)
        elif values['frequency'] > 0.0 and (values['waveform'] == 'ricker' or values['bandwidth'] > 0.0):
            pulse_ends.append(_waveform(values).duration)
    settings = vctx.values(_SETTINGS, physics_spec.settings)
    if settings and settings['heating_envelope'] == 'gaussian' and settings['heating_pulse_width'] > 0.0:
        pulse_ends.append(HeatingPulse(settings['heating_pulse_width'], settings['heating_pulse_delay']).duration)
    return pulse_ends, cw_names


_RESONANCE_PROBE = ResonanceProbe(
    component=replace(_PROBE_PARAMS[0], help=_PROBE_PARAMS[0].help + ' Pressure misses the S modes of a solid; '
                                                                      'stresses and velocities see both.'),
    add_probe=_apply_probe, time_step=_solver_time_step, waveforms=_solver_waveforms, unit=_component_unit,
    source_waveforms=_source_waveforms, band='frequency',
    pulse_end_text=('the end of the latest pulsed source (ricker / gaussian waveform: 2 t0; heat_pulse_source and a '
                    'gaussian heating_envelope: delay + 4.5 width)'))
_RESONANCE_SUMMARY = ('Finds the resonances of a structure that rings after a short pulse, such as a plate or a '
                      'cavity: their frequencies, quality factors and strengths, from the quantity recorded at one or '
                      'more points. Put the points where the modes are strong.')
_RESONANCE_NOTE = 'Velocities carry their own time stamps (t - dt / 2): no sample delay. dt = the acoustic step.'


# ---- validation (CPU only) ----
_MODULI = ('c11', 'c12', 'c22', 'c66')


def _material_moduli(properties, parameters):
    """(rho, (c11, c12, c22, c66)) [kg/m^3, Pa] of a material from plain numbers, None where unknown (unset, field- or
    state-dependent); precedence of the coefficient kernel: c_ij entries, then the speeds, then E / nu."""
    rho = plain_number(properties.get('rho'), parameters)
    longitudinal = plain_number(properties.get('sound_speed'), parameters)
    shear = plain_number(properties.get('shear_speed'), parameters)
    youngs = plain_number(properties.get('youngs_modulus'), parameters)
    isotropic = (None, None, None)   # (c11 = c22, c12, c66)
    if properties.get('sound_speed') is not None and longitudinal is None:
        pass                         # speed not plain: moduli unknown
    elif longitudinal and longitudinal > 0.0:
        if rho is not None and (shear is not None or properties.get('shear_speed') is None):
            shear_modulus = rho * (shear or 0.0) ** 2
            isotropic = (rho * longitudinal ** 2, rho * longitudinal ** 2 - 2.0 * shear_modulus, shear_modulus)
    elif youngs and youngs > 0.0:
        poisson = plain_number(properties.get('poisson_ratio'), parameters) or 0.0
        if -1.0 < poisson < 0.5:
            shear_modulus = youngs / (2.0 * (1.0 + poisson))
            lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
            isotropic = (lame_lambda + 2.0 * shear_modulus, lame_lambda, shear_modulus)
    defaults = dict(zip(_MODULI, (isotropic[0], isotropic[1], isotropic[0], isotropic[2])))
    return rho, tuple(plain_number(properties[name], parameters) if properties.get(name) is not None else defaults[name]
                      for name in _MODULI)


def _material_monoclinic(properties, parameters):
    """(c16, c26) [Pa] of a material: unset 0, None where set but not a plain number."""
    return tuple(plain_number(properties[name], parameters) if properties.get(name) is not None else 0.0
                 for name in MONOCLINIC_ARRAYS)


def stability_stiffness(c11, c12, c22, c66, weight_x, weight_y, c16=0.0, c26=0.0):
    """Host twin of `kernels.stability_stiffness` [Pa]: largest eigenvalue of the corner-wavenumber acoustic tensor
    (c16 / c26 as absolute-value bounds), at least max(c11, c22) and 0 (weight_x + weight_y = 1)."""
    s = math.sqrt(weight_x * weight_y)
    p = c11 * weight_x + c66 * weight_y + 2.0 * abs(c16) * s
    q = c66 * weight_x + c22 * weight_y + 2.0 * abs(c26) * s
    r = abs(c16) * weight_x + abs(c12 + c66) * s + abs(c26) * weight_y
    return max(0.5 * (p + q) + math.hypot(0.5 * (p - q), r), c11, c22, 0.0)


def piezo_stiffening(piezo, permittivity):
    """Host twin of the kernel's c^D - c^E = e^T (eps0 eps)^-1 e [Pa]: {'c11', 'c12', 'c22', 'c66', 'c16', 'c26'} from
    piezo = (e11, e12, e16, e21, e22, e26) [C/m^2] and the relative permittivity (eps_xx, eps_yy, eps_xy)."""
    eps_xx, eps_yy, eps_xy = permittivity
    determinant = eps_xx * eps_yy - eps_xy * eps_xy
    scale = 1.0 / (EPS0 * determinant) if determinant > 0.0 else 0.0
    inverse = (eps_yy * scale, -eps_xy * scale, eps_xx * scale)
    columns = dict(zip(('1', '2', '6'), zip(piezo[:3], piezo[3:])))   # Voigt J -> (e_1J, e_2J)

    def entry(first, second):
        (a1, b1), (a2, b2) = columns[first], columns[second]
        return inverse[0] * a1 * a2 + inverse[1] * (a1 * b2 + b1 * a2) + inverse[2] * b1 * b2
    return {'c11': entry('1', '1'), 'c12': entry('1', '2'), 'c22': entry('2', '2'), 'c66': entry('6', '6'),
            'c16': entry('1', '6'), 'c26': entry('2', '6')}


def _material_piezo_data(properties, parameters):
    """(e11 .. e26, (eps_xx, eps_yy, eps_xy)) of a material from plain numbers (unset or not plain = 0; eps_dc
    expands to xx / yy, default 1)."""
    piezo = tuple(plain_number(properties.get(name), parameters) or 0.0 for name in PIEZO_ARRAYS)
    isotropic = plain_number(properties.get('eps_dc'), parameters)
    permittivity = tuple(plain_number(properties.get(name), parameters) or default for name, default in
                         (('eps_dc_xx', isotropic or 1.0), ('eps_dc_yy', isotropic or 1.0), ('eps_dc_xy', 0.0)))
    return piezo, permittivity


def two_way_piezo(spec, physics_name):
    """True when an enabled electrostatics_acoustics_piezo coupling into physics_name has back_action bound_charge
    (stiffened dt, bound-charge buffers)."""
    return any(coupling.enabled and coupling.type == PIEZO_COUPLING_TYPE_NAME and coupling.target == physics_name
               and (coupling.params or {}).get('back_action', 'bound_charge') == 'bound_charge'
               for coupling in spec.couplings)


def _material_speeds(properties, parameters):
    """(rho, longitudinal speed max(c11, c22), shear speed c66, moduli) of a material; speeds None where unknown."""
    rho, moduli = _material_moduli(properties, parameters)
    c11, _, c22, c66 = moduli
    longitudinal = shear = None
    if rho and rho > 0.0:
        if c11 is not None or c22 is not None:
            longitudinal = math.sqrt(max(max(c11 or 0.0, c22 or 0.0), 0.0) / rho)
        if c66 is not None:
            shear = math.sqrt(max(c66, 0.0) / rho)
    return rho, longitudinal, shear, moduli


def _stiffness_error(name, properties, moduli, monoclinic=(0.0, 0.0)):
    """Error text when the plane-strain stiffness [[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]] is not positive
    definite (a fluid: c66 = c16 = c26 = 0 and c12^2 = c11 c22 allowed), None when it is, unknown or vacuum (all 0)."""
    c11, c12, c22, c66 = moduli
    c16, c26 = monoclinic
    if any(value is None for value in (*moduli, c16, c26)) or not any((*moduli, c16, c26)):
        return None
    if c11 > 0.0 and c22 > 0.0 and c66 >= 0.0:
        if c66 == 0.0 and c16 == 0.0 and c26 == 0.0 and c12 ** 2 <= c11 * c22 * (1.0 + 1e-6):
            return None
        determinant = c66 * (c11 * c22 - c12 ** 2) - c11 * c26 ** 2 - c22 * c16 ** 2 + 2.0 * c12 * c16 * c26
        if c66 > 0.0 and c12 ** 2 < c11 * c22 and determinant > 0.0:
            return None
    hint = (' (from the speeds: shear_speed must be below sound_speed)'
            if properties.get('shear_speed') is not None and properties.get('c12') is None else '')
    shear_coupling = f', c16 {c16:.4g}, c26 {c26:.4g}' if (c16 or c26) else ''
    return (f"material '{name}': the effective in-plane stiffness c11 {c11:.4g}, c12 {c12:.4g}, c22 {c22:.4g}, c66 "
            f'{c66:.4g}{shear_coupling} Pa is not positive definite{hint}: it needs c11 > 0, c22 > 0, c66 > 0, c12^2 < '
            'c11 c22 and det [[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]] > 0 (a fluid: c66 = c16 = c26 = 0, '
            'c12 = c11 = c22), else the waves grow without bound')


def _cell_weights(mesh, axisymmetric=False):
    """(weight_x, weight_y, sqrt(1 / dx_min^2 + 1 / dy_min^2)) of the stability stiffness over the whole grid
    (axisymmetric: 1 / dx_min^2 times the radial stability factor, as the solver)."""
    inv_dx2, inv_dy2 = 1.0 / float(mesh.dx_cells.min()) ** 2, 1.0 / float(mesh.dy_cells.min()) ** 2
    if axisymmetric:
        inv_dx2 *= radial_stability_factor(mesh.dx_cells)
    total = inv_dx2 + inv_dy2
    return inv_dx2 / total, inv_dy2 / total, math.sqrt(total)


def _selected_material_properties(spec):
    """[(name, merged properties)] of the selected materials of a ModelSpec whose library data resolves."""
    return [(material.name, properties) for material, properties, _ in selected_material_data(spec)]


def estimated_time_step(vctx, physics_spec):
    """Acoustic dt [s] of an elastic_waves physics from the plain moduli of the selected materials, None without a
    mesh, settings or a known carrier; the build's dt is this or smaller (smaller next to vacuum)."""
    mesh = vctx.mesh
    values = vctx.values(_SETTINGS, physics_spec.settings)
    if mesh is None or not values:
        return None
    return _time_step_estimate(vctx.spec, vctx.parameters, mesh, values['courant'], is_axisymmetric(values),
                               two_way_piezo(vctx.spec, physics_spec.name))


def _time_step_estimate(spec, parameters, mesh, courant, axisymmetric=False, stiffened=False):
    """estimated_time_step from a ModelSpec, its parameters, a mesh estimate, the courant and geometry settings
    (stiffened: the two-way piezo c^D)."""
    weight_x, weight_y, inverse_diagonal = _cell_weights(mesh, axisymmetric)
    fastest = 0.0
    for _, properties in _selected_material_properties(spec):
        rho, moduli = _material_moduli(properties, parameters)
        if rho and rho > 0.0 and all(value is not None for value in moduli):
            extra = dict.fromkeys(('c11', 'c12', 'c22', 'c66', 'c16', 'c26'), 0.0)
            if stiffened:
                extra = piezo_stiffening(*_material_piezo_data(properties, parameters))
            c11, c12, c22, c66 = (value + extra[name] for value, name in zip(moduli, _MODULI))
            c16, c26 = ((value or 0.0) + extra[name] for value, name in
                        zip(_material_monoclinic(properties, parameters), MONOCLINIC_ARRAYS))
            stiffness = stability_stiffness(c11, c12, c22, c66, weight_x, weight_y, c16, c26)
            fastest = max(fastest, math.sqrt(stiffness / rho))
    return courant / (fastest * inverse_diagonal) if fastest > 0.0 else None


def _highest_source_frequency(vctx, physics_spec):
    """Highest significant source frequency [Hz] over the enabled waveform features (None without one)."""
    highest = None
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in _WAVEFORM_FEATURES:
            continue
        values = vctx.values(_FEATURES_BY_NAME[feature.type].params, feature.params)
        if not values:
            continue
        factor = {'ricker': 2.5, 'gaussian': 1.0 + values['bandwidth'], 'cw': 1.0}[values['waveform']]
        frequency = factor * values['frequency']
        highest = frequency if highest is None else max(highest, frequency)
    return highest


def _material_issues(vctx, physics_spec, path):
    """Material rules (density, positive-definite stiffness), the resolution warning and the time step info."""
    parameters = vctx.parameters
    issues, carriers, slowest = [], 0, None
    for material, properties, _ in selected_material_data(vctx.spec, unresolved=True):
        if properties is None:
            carriers += 1   # reported at the material; may carry waves
            continue
        rho, longitudinal, shear, moduli = _material_speeds(properties, parameters)
        material_path = ('material', material.name)
        stiff = any(properties.get(name) is not None for name in ('sound_speed', 'c11', 'c22', 'youngs_modulus'))
        dense = properties.get('rho') is not None and (rho is None or rho > 0.0)
        if stiff and not dense:
            issues.append(Issue('warning', material_path,
                                f"material '{material.name}' sets a sound speed or stiffness (sound_speed, "
                                'youngs_modulus or c_ij) but no density rho: '
                                'elastic_waves treats it as vacuum (its faces do not move; a speed without rho gives '
                                'no stiffness)'))
        if stiff and dense:
            carriers += 1
        stiffness_error = (_stiffness_error(material.name, properties, moduli,
                                            _material_monoclinic(properties, parameters)) if dense else None)
        if stiffness_error:
            issues.append(Issue('error', material_path, stiffness_error))
        for speed in (longitudinal, shear):
            if speed is not None and speed > 0.0:
                slowest = speed if slowest is None else min(slowest, speed)
    if carriers == 0:
        issues.append(Issue('error', path, 'no selected material sets both a density (rho) and a sound speed or '
                                           'stiffness (youngs_modulus, c11 / c22): nothing carries an elastic or '
                                           'acoustic wave'))
    mesh = vctx.mesh
    if mesh is None:
        return issues
    frequency = _highest_source_frequency(vctx, physics_spec)
    if frequency and slowest:
        cells = slowest / frequency / mesh.dx_max
        if cells < MIN_CELLS_PER_WAVELENGTH:
            issues.append(Issue('warning', path,
                                f'only {cells:.3g} cells per shortest wavelength ({slowest:g} m/s at {frequency:.4g} '
                                f'Hz over cells up to {mesh.dx_max:.4g} m); use >= {MIN_CELLS_PER_WAVELENGTH} (better '
                                '20) or the waves disperse'))
    dt = estimated_time_step(vctx, physics_spec)
    if dt:
        duration = study_duration(vctx)
        steps = f', ~{duration / dt:.4g} steps for the duration' if duration else ''
        issues.append(Issue('info', path, f'acoustic time step about {dt:.4g} s from the plain material stiffness (the '
                                          f'build derives it from the cells, smaller next to vacuum){steps}'))
    return issues


def _mirror_plane_issues(vctx, values, path):
    """Warning per model mirror face (core.mirror: EM bc, mirror features, a symmetric face) whose acoustic face is
    not a slip wall (rigid / symmetric): only the slip wall on the window face is the even mirror image of the unfolded
    model (Results unfold elastic_waves fields even)."""
    if not values:
        return []
    faces = [face for face in mirror_faces(vctx.spec)
             if face in FACES and values[f'bc_{face}'] not in ('rigid', _SYMMETRIC_FACE)]
    if not faces:
        return []
    return [Issue('warning', path, f"mirror plane(s) {', '.join(faces)} of the model: this physics is not rigid there, "
                                   'but the unfolded (mirrored) model needs the slip wall (rigid) on the plane, so its '
                                   'unfolded fields and doubled totals are wrong otherwise')]


def _symmetric_face_issues(vctx, physics_spec, values, path):
    """bc 'symmetric' faces (own model mirror planes): errors on x_hi of geometry axisymmetric (x_lo: the axis rule)
    and for another physics with mirror bc settings (EM bc_<face> symmetric / antisymmetric) that does not set the
    face to one of those (its fields would unfold and its integrals double as a mirror it does not solve; the rule of
    definition_helpers.mirror_feature_issues for a face that is no feature)."""
    faces = [face for face in FACES if values.get(f'bc_{face}') == _SYMMETRIC_FACE]
    if not faces:
        return []
    issues = []
    if is_axisymmetric(values) and 'x_hi' in faces:
        issues.append(Issue('error', path, "bc_x_hi symmetric: x_hi is the cylinder r = r_max of geometry "
                                           "'axisymmetric', not a mirror plane (its image across r = r_max is no part "
                                           'of the body of revolution); use rigid'))
    for other, other_type in known_physics(vctx.spec, vctx.registry):
        if other.name == physics_spec.name:
            continue
        bc_params = {param.name for param in other_type.settings
                     if param.name.startswith('bc_') and set(MIRROR_SIGNS) <= set(param.choices or ())}
        partners = [face for face in faces if f'bc_{face}' in bc_params]
        other_values = vctx.values(other_type.settings, other.settings) if partners else None
        for face in partners if other_values else ():   # no values: reported at that physics
            value = other_values.get(f'bc_{face}')
            if value not in MIRROR_SIGNS:
                issues.append(Issue('error', path, f"bc_{face} symmetric makes {face} a model mirror plane, but "
                                                   f"'{other.name}' sets bc_{face} = {value!r}: its fields would "
                                                   'unfold (and its integrals reaching the face double) as a mirror '
                                                   f'it does not solve. Set its bc_{face} to symmetric or '
                                                   f'antisymmetric, or this bc_{face} to rigid (a slip wall that is '
                                                   'no mirror plane)'))
    return issues


def _layer_issues(vctx, physics_spec, values, path):
    """Absorbing layers that do not fit, periodic faces another physics pads, an info when a layer lies inside the
    domain, and point features in it."""
    mesh = vctx.mesh
    if mesh is None or not values:
        return []
    issues = []
    padded_periodic = [face for face in FACES if values[f'bc_{face}'] == 'periodic' and mesh.npml_sides[face] > 0]
    if padded_periodic:
        issues.append(Issue('error', path, f"periodic {' and '.join(padded_periodic)}: another physics pads the grid "
                                           'there (an EM PML), so the acoustic period would include the padding; make '
                                           'that physics periodic on this axis too, or use absorbing / rigid faces'))
    n_cells = values['pml_cells']
    edges = {'x': mesh.x_edges, 'y': mesh.y_edges}
    window = {'x': float(mesh.x_edges[mesh.nx - mesh.npml_sides['x_hi']]),
              'y': float(mesh.y_edges[mesh.ny - mesh.npml_sides['y_hi']])}
    inside = {}   # face -> (lo, hi) coordinates of the layer part inside the domain
    for axis, total in (('x', mesh.nx), ('y', mesh.ny)):
        absorbing = [face for face in (f'{axis}_lo', f'{axis}_hi') if values[f'bc_{face}'] == 'absorbing']
        if len(absorbing) * n_cells >= total:
            issues.append(Issue('error', path, f'{total} cells along {axis} cannot hold absorbing layers of {n_cells} '
                                               f'cells on {" and ".join(absorbing)}; lower pml_cells or enlarge the '
                                               'domain'))
            continue
        for face in absorbing:
            if face.endswith('_lo') and float(edges[axis][n_cells]) > 0.0:
                inside[face] = (0.0, float(edges[axis][n_cells]))
            elif face.endswith('_hi') and float(edges[axis][total - n_cells]) < window[axis]:
                inside[face] = (float(edges[axis][total - n_cells]), window[axis])
    if inside:
        issues.append(Issue('info', path, 'absorbing layers inside the domain (no other physics pads the grid): '
                                          + ', '.join(f'{face} {lo:.4g}..{hi:.4g} m' for face, (lo, hi)
                                                      in inside.items())))
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        for name in vctx.objects(feature.selection):
            shape = vctx.shapes.get(name)
            if shape is None or shape.kind != 'point':
                continue
            for face, (lo, hi) in inside.items():
                coordinate = shape.values[face[0]]
                if lo < coordinate < hi:
                    issues.append(Issue('warning', feature_path(physics_spec, feature),
                                        f"point '{name}' lies in the absorbing layer of {face}: its waves are damped "
                                        'there'))
    return issues


def _hoop_stiffness_issues(vctx):
    """Axisymmetric material rules: a warning per selected material setting c_ij (hoop stiffness assumed), an error
    when the 3D normal block [[c11, c12, c12], [c12, c22, c12], [c12, c12, c11]] is not positive semi-definite."""
    issues = []
    for name, properties in _selected_material_properties(vctx.spec):
        material_path = ('material', name)
        set_entries = [entry for entry in _MODULI if properties.get(entry) is not None]
        if set_entries:
            issues.append(Issue('warning', material_path,
                                f"material '{name}' sets {', '.join(set_entries)}: axisymmetric elastic_waves takes "
                                'the hoop stiffness c_tt = c11 and c_rt = c_zt = c12, exact only for isotropic solids '
                                'and fluids (a crystal needs transverse isotropy about the z axis)'))
        _, moduli = _material_moduli(properties, vctx.parameters)
        c11, c12, c22, _ = moduli
        if c11 is None or c12 is None or c22 is None or not any((c11, c12, c22)):
            continue
        block = np.array([[c11, c12, c12], [c12, c22, c12], [c12, c12, c11]], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(block)
        if eigenvalues[0] < -1e-6 * float(np.max(np.abs(eigenvalues))):
            issues.append(Issue('error', material_path,
                                f"material '{name}': the 3D normal stiffness block of axisymmetric elastic_waves (c11 "
                                f'{c11:.4g}, c12 {c12:.4g}, c22 {c22:.4g} Pa; c_tt = c11, c_rt = c_zt = c12) is not '
                                'positive semi-definite (isotropic: shear_speed must not exceed 0.866 sound_speed, '
                                'i.e. a non-negative bulk modulus), else the waves grow without bound'))
    return issues


ANTIPLANE_WARNING_RATIO = 1e-6    # anti-plane coupling share that warns (0 on a crystal mirror plane)


def _crystal_issues(vctx):
    """Per spec material with a 'crystal' entry that expands: an info of the sagittal reduction and a warning for an
    anti-plane coupling (model plane no mirror plane); a malformed entry or plane is reported at the material
    (validate.check_materials)."""
    from ...materials import crystal_entry
    from ...materials.crystal_tensors import sagittal_crystal
    issues = []
    for material in vctx.spec.materials:
        entry = crystal_entry(material.dispersion)
        if entry is None:
            continue
        path = ('material', material.name)
        try:
            reduced = sagittal_crystal(entry)
        except (ValueError, TypeError):   # reported at the material
            continue
        properties = reduced.properties
        issues.append(Issue('info', path, f"material '{material.name}': crystal {reduced.crystal} at euler_deg "
                                          f"{list(reduced.euler_deg)} in the sagittal plane: c11 {properties['c11']:.4g},"
                                          f" c12 {properties['c12']:.4g}, c22 {properties['c22']:.4g}, c66 "
                                          f"{properties['c66']:.4g}, c16 {reduced.c16:.4g}, c26 {reduced.c26:.4g} Pa, "
                                          'piezo_e '
                                          + ', '.join(f'{name[-2:]} {properties[name]:.3g}' for name in PIEZO_ARRAYS)
                                          + f" C/m^2, eps_dc_xx / _yy / _xy {properties['eps_dc_xx']:.4g} / "
                                            f"{properties['eps_dc_yy']:.4g} / {properties['eps_dc_xy']:.4g}"))
        ratio = max(reduced.antiplane_stiffness_ratio, reduced.antiplane_piezo_ratio)
        if ratio > ANTIPLANE_WARNING_RATIO:
            issues.append(Issue('warning', path,
                                f"material '{material.name}': the model plane is not a mirror plane of the crystal "
                                f'(anti-plane coupling: stiffness {reduced.antiplane_stiffness_ratio:.3g}, piezo '
                                f'{reduced.antiplane_piezo_ratio:.3g} of their largest entries); the anti-plane '
                                'displacement u_z is clamped to 0, so in-plane waves miss that coupling'))
    return issues


# (entries, what they are, what axisymmetric elastic_waves lacks) of the planar-only material arrays
_PLANAR_ONLY_ENTRIES = ((PIEZO_ARRAYS, 'piezo_e* entries', 'no piezoelectric coupling (planar sagittal plane only)'),
                        (MONOCLINIC_ARRAYS, 'c16 / c26', 'no monoclinic stiffness (its hoop terms assume c_tt = c11, '
                                                         'c_rt = c_zt = c12)'))


def _planar_only_issues(vctx):
    """Error per selected material setting piezo_e* or c16 / c26 entries in an axisymmetric elastic_waves."""
    return [Issue('error', ('material', name), f"material '{name}' sets {what}: axisymmetric elastic_waves has {lack}")
            for name, properties in _selected_material_properties(vctx.spec)
            for entries, what, lack in _PLANAR_ONLY_ENTRIES
            if any(properties.get(entry) is not None for entry in entries)]


def _axisymmetric_physics_issues(vctx, physics_spec, values, path):
    """The axis rules (bc_x_lo rigid, no padding at x_lo) and the shared partner warning."""
    issues = axisymmetric_issues(vctx, physics_spec, path, 'elastic_waves')
    if values.get('bc_x_lo') == _SYMMETRIC_FACE:
        issues.append(Issue('error', path, 'geometry axisymmetric: x_lo is the axis r = 0, not a mirror plane (the '
                                           'ring weights already count the whole revolution; its r < 0 side unfolds '
                                           "for display without a mirror), so bc_x_lo must be 'rigid', not "
                                           "'symmetric'"))
    elif values.get('bc_x_lo') != 'rigid':
        issues.append(Issue('error', path, f"geometry axisymmetric: x_lo is the axis r = 0, so bc_x_lo must be 'rigid' "
                                           f"(v_r = 0, s_rz = 0 there), got {values.get('bc_x_lo')!r}"))
    mesh = vctx.mesh
    if mesh is not None and mesh.npml_sides['x_lo'] > 0:
        issues.append(Issue('error', path, 'geometry axisymmetric: another physics pads the grid at x_lo (an EM PML), '
                                           'but x_lo is the axis r = 0; give that physics no padding there'))
    return issues


def _hoop_output_issues(vctx, physics_spec, path):
    """Planar geometry: errors for a probe / resonance_monitor recording stt and for a stored field '<name>.stt'."""
    issues = [Issue('error', feature_path(physics_spec, feature),
                    "component 'stt' (hoop stress) exists only with geometry axisymmetric")
              for feature in physics_spec.features
              if feature.enabled and feature.type in _POINT_TRACE_FEATURES
              and (feature.params or {}).get('component') == 'stt']
    stored = f'{physics_spec.name}.stt'
    if stored in (vctx.spec.study.stored_fields or ()):
        issues.append(Issue('error', path, f"stored field '{stored}' (hoop stress) exists only with geometry "
                                           'axisymmetric'))
    return issues


def _half_storage_issues(vctx, physics_spec, path):
    """half_storage stepper: errors for a coupling writing into this physics (its storage scale is set from the
    features) and for piezoelectric / monoclinic materials; the resonance finder refuses it by its feature
    precisions."""
    if resolved_stepper_precision(vctx.solvers(physics_spec)) != 'half_storage':
        return []
    issues = [Issue('error', path, f"coupling '{coupling.name}' writes into '{physics_spec.name}', which half_storage "
                                   '(fp16 fields) does not take: the stored-unit scale is fixed from the sources and '
                                   'heating pulses at the build, so a coupled heating rate or field could overflow; use '
                                   'a single-precision stepper solver')
              for coupling in vctx.spec.couplings if coupling.enabled and coupling.target == physics_spec.name]
    for name, properties in _selected_material_properties(vctx.spec):
        for entries, what, _ in _PLANAR_ONLY_ENTRIES:
            if any(properties.get(entry) is not None for entry in entries):
                issues.append(Issue('error', ('material', name),
                                    f"material '{name}' sets {what}, which half_storage elastic_waves does not support "
                                    '(the strain accumulator and the piezoelectric / monoclinic stress use float32 '
                                    'fields in physical units): use a single-precision stepper solver'))
    return issues


def _validate_physics(vctx, physics_spec, settings_values):
    path = ('physics', physics_spec.name)
    issues = []
    values = settings_values or {}
    for axis in ('x', 'y'):
        lo, hi = values.get(f'bc_{axis}_lo'), values.get(f'bc_{axis}_hi')
        if lo is not None and hi is not None and (lo == 'periodic') != (hi == 'periodic'):
            issues.append(Issue('error', path, f'bc_{axis}_lo ({lo}) and bc_{axis}_hi ({hi}): periodic faces must be '
                                               'paired'))
    axisymmetric = is_axisymmetric(values)
    if axisymmetric:
        issues += _axisymmetric_physics_issues(vctx, physics_spec, values, path)
    else:
        issues += _hoop_output_issues(vctx, physics_spec, path)
    issues += _half_storage_issues(vctx, physics_spec, path)
    if any(issue.level == 'error' for issue in issues):
        return issues
    issues += _layer_issues(vctx, physics_spec, values, path)
    issues += _symmetric_face_issues(vctx, physics_spec, values, path)
    issues += _mirror_plane_issues(vctx, values, path)
    if reports_model_checks(vctx, physics_spec, PHYSICS_TYPE_NAME):
        issues += _material_issues(vctx, physics_spec, path)
        issues += _crystal_issues(vctx)
        if axisymmetric:
            issues += _hoop_stiffness_issues(vctx) + _planar_only_issues(vctx)
    return issues


def _validate_line(vctx, physics_spec, feature_spec, values):
    size = vctx.domain_size
    if size is None:
        return []
    issues = []
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is not None and shape.kind == 'line' and line_orientation(shape.values, *size) is None:
            issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                f"line '{name}' must be horizontal or vertical"))
    return issues


def _validate_heat_pulse(vctx, physics_spec, feature_spec, values):
    """Warning when c pulse_width spans fewer than 2 cells of the coarsest window cell (plain speeds only)."""
    mesh = vctx.mesh
    if mesh is None or not values:
        return []
    speeds = [speed for speed in (_material_speeds(properties, vctx.parameters)[1]
                                  for _, properties in _selected_material_properties(vctx.spec)) if speed]
    if speeds and max(speeds) * values['pulse_width'] < 2.0 * mesh.dx_max:
        return [Issue('warning', feature_path(physics_spec, feature_spec),
                      f"pulse_width {values['pulse_width']:.3g} s: sound crosses only "
                      f"{max(speeds) * values['pulse_width'] / mesh.dx_max:.2g} cells in it; the pressure front is "
                      'under-resolved and rings (use c pulse_width >= ~5 cells)')]
    return []


# ---- factory, memory ----
def _solver_faces(values) -> dict:
    """AcousticPhysics faces of the bc settings: 'symmetric' is the 'rigid' slip wall (the mirror plane is model
    metadata: core.mirror reads the setting)."""
    return {face: 'rigid' if values[f'bc_{face}'] == _SYMMETRIC_FACE else values[f'bc_{face}'] for face in FACES}


# ---- parked material arrays ----
_HEAT_PULSE_FEATURE = 'heat_pulse_source'   # its HeatPulseSource reads gruneisen every step
# arrays only couplings out of this physics read: the acousto-optic transforms (photoelastic), the bound charge (piezo)
_READ_BY_COUPLINGS_OUT = ('photoelastic_p11', 'photoelastic_p12', *PHOTOELASTIC_TENSOR_ARRAYS, *PIEZO_ARRAYS)
# couplings into this physics that write source slots only (no material array, no coefficient rebuild), with the
# arrays their slots read on the device: heating_power every step (gruneisen), the electric_field slots at each
# transfer (piezo; two-way: also the bound charge and the stiffened dt)
_SLOT_COUPLINGS_IN = {PHOTOACOUSTIC_COUPLING_TYPE_NAME: ('gruneisen',), PIEZO_COUPLING_TYPE_NAME: PIEZO_ARRAYS}


def _parked_material_arrays(spec, registry, physics_spec, parameters) -> tuple:
    """MaterialGrid arrays elastic_waves moves to the host after its first coefficient build (AcousticPhysics
    park_arrays): solver.PARKABLE_ARRAYS without gruneisen when a heat pulse feature reads it every step, under the
    shared rule (definition_helpers.parkable_material_arrays over the elastic_waves material properties: no coupling
    into the physics but the slot-only photoacoustic / piezo ones, which keep gruneisen / the piezo arrays; one
    reading it keeps the photoelastic / piezo arrays). Names of unallocated optional arrays may be listed (the solver
    and _allocated_parked skip them)."""
    heat_pulse = any(feature.enabled and feature.type == _HEAT_PULSE_FEATURE for feature in physics_spec.features)
    arrays = tuple(name for name in PARKABLE_ARRAYS if not (heat_pulse and name == 'gruneisen'))
    return parkable_material_arrays(spec, registry, physics_spec, parameters,
                                    {param.name for param in _MATERIAL_PROPERTIES}, arrays,
                                    read_by_couplings_out=_READ_BY_COUPLINGS_OUT, slot_couplings_in=_SLOT_COUPLINGS_IN)


def _allocated_parked(context) -> list:
    """The _parked_material_arrays the build allocates: ACOUSTIC_SCHEMA always, an optional group (piezo, photoelastic
    tensor) when a material sets one of its arrays to anything but None (meshing.optional_material_arrays)."""
    groups = (PIEZO_ARRAYS, PHOTOELASTIC_TENSOR_ARRAYS)
    triggered = {name for group in groups if any(context.material_sets(member, default=None) for member in group)
                 for name in group}
    return [name for name in _parked_material_arrays(context.spec, context.registry, context.physics_spec,
                                                     context.parameters)
            if name in ACOUSTIC_SCHEMA or name in triggered]


def _create(ctx, physics_spec, values):
    envelope = (HeatingPulse(values['heating_pulse_width'], values['heating_pulse_delay'])
                if values['heating_envelope'] == 'gaussian' else None)
    return AcousticPhysics(ctx.domain, faces=_solver_faces(values),
                           pml_cells=values['pml_cells'], pml_reflection=values['pml_reflection'],
                           courant=values['courant'], heating_envelope=envelope,
                           axisymmetric=is_axisymmetric(values),
                           precision=stepper_precision(ctx.solvers(physics_spec.name)),
                           park_arrays=_parked_material_arrays(ctx.spec, ctx.registry, physics_spec, ctx.parameters))


_GRAPHED_TRACES = ('field_probe', 'resonance_monitor')                         # ProbeRecorder: inside the step graph
_TABLED_SOURCES = ('force_source', 'volume_source', 'traction_line_source')   # pulse tables unless waveform cw


def _time_estimate(settings_values, mesh, context):
    """Explicit velocity-stress engine: per acoustic step (the estimated dt; the build's may be smaller next to vacuum)
    its arrays streamed once, ~6 kernels (gpu: one step graph, solver.STEP_GRAPHS; with the runner hand-off one graph
    per STEP_GRAPH_TICKS steps, but the per-step host work left, the quiet check and probe stamp, still books one)
    plus two per enabled feature (gpu: probes and pulsed sources replay inside the graph)."""
    piezo = any(context.material_sets(name) for name in PIEZO_ARRAYS)
    two_way = piezo and two_way_piezo(context.spec, context.physics_spec.name)
    dt = _time_step_estimate(context.spec, context.parameters, mesh, settings_values['courant'],
                             is_axisymmetric(settings_values), two_way)
    features = [feature for feature in context.physics_spec.features if feature.enabled]
    step_graphs = context.backend == 'gpu' and STEP_GRAPHS
    if step_graphs:
        features = [feature for feature in features if not (
            feature.type in _GRAPHED_TRACES
            or (feature.type in _TABLED_SOURCES and feature.params.get('waveform', 'ricker') != 'cw'))]
    field_launches = 1 if step_graphs else 6
    # cpu: the velocity / stress kernels (CPML fused in) have numba twins (kernels._FIELD_NUMBA_TWINS), the rest numpy
    return explicit_step_estimate(context, 'velocity-stress', dt, launches=field_launches + 2 * len(features),
                                  numba_items=_numba_stepped, numba_launches=2)


def _numba_stepped(item_label):
    """Memory items (_memory_estimate labels) the numba velocity / stress twins stream."""
    return item_label.startswith(('vx, vy', 'stiffness and velocity coefficients', 'CPML ', 'radius rows',
                                  'strain accumulator', 'monoclinic strain rates'))


_STEP_TEMPORARIES = 8          # full-grid float32 temporaries of AcousticPhysics._update_time_step
# full-grid float32 temporaries of the numpy coefficients twin (kernels._coefficients_host: the moduli, the four corner
# shear moduli and the corner mean stay alive through the face coefficients' temporaries; bools counted as one)
_HOST_REBUILD_TEMPORARIES = 15


def _memory_estimate(settings_values, mesh, context):
    """5 fields (axisymmetric 6: + stt; float16 under half_storage) + 5 to 8 coefficient arrays (solver._set_storage)
    + the pressure buffer (float32 full grid), CPML psi slabs (8 per absorbing axis pair), a bool rigid keep mask, the
    probe traces over the study duration in estimated acoustic steps, piezo / monoclinic: the strain accumulator (+
    monoclinic: the rate buffers; two-way: P and bound-charge buffers); build extra: the stable-step temporaries of
    finalize, or the parked material arrays before the fields exist (_parked_material_arrays: host items then, with
    the host coefficient rebuild); run: the elastic strain pair (first acousto-optic transfer or strain render), the
    shear strain buffers (photoelastic tensor read by an acousto-optic coupling), the piezo last field / shear drive
    (first field transfer) and the trace growth."""
    cells = int(mesh.n_cells)
    axisymmetric = is_axisymmetric(settings_values)
    n_fields = 6 if axisymmetric else 5
    half = resolved_stepper_precision(context.solvers) == 'half_storage'
    field_bytes = 2 if half else FLOAT32_BYTES
    layer = settings_values['pml_cells']
    slab_x = layer * sum(settings_values[f'bc_{face}'] == 'absorbing' for face in ('x_lo', 'x_hi'))
    slab_y = layer * sum(settings_values[f'bc_{face}'] == 'absorbing' for face in ('y_lo', 'y_hi'))
    has_rigid = any(feature.enabled and feature.type == 'rigid_domain' for feature in context.physics_spec.features)
    psi_cells = max(slab_x, 1) * int(mesh.ny) + max(slab_y, 1) * int(mesh.nx)
    # solver._set_storage: c11, c12, c66c, vb_x, vb_y; + c22 when a material sets c11 / c22 (else c22 is c11); + va_x,
    # va_y when a material sets acoustic_damping (else the lossless velocity kernels)
    n_coefficients = (5 + int(context.material_sets('c11') or context.material_sets('c22'))
                      + 2 * int(context.material_sets('acoustic_damping')))
    items = ((f"vx, vy, sxx, syy, sxy{', stt' if axisymmetric else ''} ({n_fields} x "
              f"{'float16' if half else 'float32'})", n_fields * field_bytes * cells),
             (f'stiffness and velocity coefficients ({n_coefficients} x float32)',
              n_coefficients * FLOAT32_BYTES * cells),
             ('pressure buffer (float32)', FLOAT32_BYTES * cells),
             ('CPML memory variables', 4 * FLOAT32_BYTES * psi_cells),
             ('CPML profiles and slab maps', (8 * FLOAT32_BYTES + INT32_BYTES) * (int(mesh.nx) + int(mesh.ny))),
             ('radius rows (float32)', KERNEL_ROWS * FLOAT32_BYTES * int(mesh.nx) if axisymmetric else 0),
             ('rigid keep mask (bool)', BOOL_BYTES * cells if has_rigid else 0))
    piezo = any(context.material_sets(name) for name in PIEZO_ARRAYS)
    monoclinic = any(context.material_sets(name) for name in MONOCLINIC_ARRAYS)
    two_way = piezo and two_way_piezo(context.spec, context.physics_spec.name)
    tensor = any(context.material_sets(name) for name in PHOTOELASTIC_TENSOR_ARRAYS)
    reads_shear = any(coupling.enabled and coupling.type == ACOUSTO_OPTIC_COUPLING_TYPE_NAME
                      and coupling.source == context.physics_spec.name
                      and (tensor or _target_type(context.spec, coupling.target) == _OFF_DIAGONAL_EM_TYPE)
                      for coupling in context.spec.couplings)
    shear_buffers = (1 if piezo or monoclinic else 2) if reads_shear else 0   # centre (+ corner without S)
    items += (('strain accumulator (3 x float32)', 3 * FLOAT32_BYTES * cells if piezo or monoclinic else 0),
              ('monoclinic strain rates (3 x float32)', 3 * FLOAT32_BYTES * cells if monoclinic else 0),
              ('piezo polarisation and bound charge (3 x float32)', 3 * FLOAT32_BYTES * cells if two_way else 0))
    acoustic_dt = _time_step_estimate(context.spec, context.parameters, mesh, settings_values['courant'],
                                      axisymmetric, two_way)
    trace_items, trace_growth, host_items = point_trace_items(context, mesh, _POINT_TRACE_FEATURES,
                                                              dt=acoustic_dt or 0.0)   # unknown dt: initial traces
    # build peak: the stable-step temporaries (_update_time_step) of finalize beside every kept array; parked material
    # arrays (meshing books them nowhere else) stay on the device from the materials stage until the first coefficient
    # build of create (beside the storage choice's bool temporaries), and the fields, the rigid mask, the pressure
    # buffer and the two-way piezo buffers (enable_bound_charge, coupling create) exist only after that first time
    # step, which uploads a parked rho beside its temporaries
    build_extra = _STEP_TEMPORARIES * FLOAT32_BYTES * cells
    parked = _allocated_parked(context)
    if parked:
        parked_bytes = FLOAT32_BYTES * len(parked) * cells
        later = (n_fields * field_bytes + FLOAT32_BYTES + (BOOL_BYTES if has_rigid else 0)
                 + (3 * FLOAT32_BYTES if two_way else 0)) * cells
        build_extra = max(parked_bytes + 2 * BOOL_BYTES * cells - later,
                          build_extra + (FLOAT32_BYTES * cells if 'rho' in parked else 0))
        host_items = tuple(host_items) + ((f"material arrays parked on the host ({', '.join(parked)}, float32)",
                                           parked_bytes),)
        if not set(parked).isdisjoint(COEFFICIENT_INPUTS):   # AcousticPhysics._rebuild_on_host
            downloaded = sum(1 for name in COEFFICIENT_INPUTS if name not in parked)
            host_items += (('host coefficient rebuild (numpy twin: outputs, temporaries, unparked inputs; '
                            'approximate)', FLOAT32_BYTES * (n_coefficients + _HOST_REBUILD_TEMPORARIES + downloaded)
                            * cells + (BOOL_BYTES * cells if has_rigid else 0)),)
    return MemoryEstimate.from_items(items + tuple(trace_items), build_extra=build_extra,
                                     host_items=host_items,
                                     run_items=[('elastic strain buffers (2 x float32)', 2 * FLOAT32_BYTES * cells),
                                                (f'elastic shear strain buffers ({shear_buffers} x float32)',
                                                 shear_buffers * FLOAT32_BYTES * cells),
                                                ('piezo last field and shear drive (3 x float32)',
                                                 3 * FLOAT32_BYTES * cells if piezo else 0)]
                                     + trace_growth)


def _heat_pulse_memory(values, n_selected_cells, mesh):
    """Power density on the bounding box of the selection (at least the selected cells)."""
    return MemoryEstimate.from_items((('heating power on its box (float32)', FLOAT32_BYTES * int(n_selected_cells)),),
                                     build_extra=2 * FLOAT32_BYTES * int(mesh.n_cells))


# ---- type ----
_FEATURES = (
    FeatureType('force_source', 'Point force', 'source', 'point', params=_FORCE_PARAMS, object_kinds=('point',),
                max_entities=1, apply=_apply_force_source, validate=validate_points_inside,
                summary='A force at a point pushing the material back and forth; in a solid it sends both '
                        'compression and shear waves.',
                description='force * waveform(t) along angle in the cell containing the point, half on each of its '
                            'two x faces (x component) and y faces (y component) as force density over each face '
                            'dual cell, at the start of each step. A point in the outermost cell of a non-periodic '
                            'face loses the half on its outer face.'),
    FeatureType('volume_source', 'Point volume source', 'source', 'point', params=_VOLUME_PARAMS,
                object_kinds=('point',), max_entities=1, apply=_apply_volume_source, validate=validate_points_inside,
                summary='A tiny pulsating cylinder at a point, sending sound waves equally in all directions.',
                description='Monopole line source in the cell containing the point: volume_rate * waveform(t) acts '
                            'as an isotropic strain rate on the cell (sampled half a step after the velocity forces). '
                            'In a fluid it radiates |p| = rho omega Q / 4 |H0^(2)(k r)|, mesh-independent; in a solid '
                            'mainly compression waves.'),
    FeatureType('traction_line_source', 'Line source (traction sheet)', 'source', 'line', params=_TRACTION_PARAMS,
                object_kinds=('line',), max_entities=1, apply=_apply_traction_line, validate=_validate_line,
                summary='A force spread along a straight line. Across the whole domain it sends plane waves to both '
                        'sides: compression waves for a force across the line, shear waves for one along it.',
                description='traction * waveform(t) at the centres of the cells [cell(lo), cell(hi) + 1) under an '
                            'axis-aligned line, continuing through any padding where an end touches the domain edge: '
                            'the normal component split over the two faces around each line cell, the tangential one '
                            'on the faces along the line. A sheet across the whole domain (periodic or absorbing '
                            'ends) launches plane waves to each side; a shorter one diffracts.'),
    FeatureType('heat_pulse_source', 'Heating pulse (photoacoustic)', 'source', 'domain', params=_HEAT_PULSE_PARAMS,
                apply=_apply_heat_pulse, validate=_validate_heat_pulse, memory_estimate=_heat_pulse_memory,
                summary='A short heating pulse in the selected domains, such as a laser pulse absorbed in a '
                        "material. The heated material expands and sends out sound (photoacoustics); its "
                        "'Grueneisen parameter' sets how strongly.",
                description='Thermoelastic source: the heating rate q (see power_density) adds -gruneisen q to '
                            'd(sxx)/dt and d(syy)/dt of the selected cells (at the half step); cells without '
                            'gruneisen ignore it. Heating from another time-stepped physics goes into the '
                            f"'{HEATING_SLOT}' slot instead (field_transfer, optionally with heating_envelope "
                            'gaussian).'),
    FeatureType('rigid_domain', 'Rigid domain', 'domain_condition', 'domain', apply=_apply_rigid,
                summary='Makes the selected domains immovable, like an ideal rigid obstacle.',
                description='Zero velocity on every face of the selected cells (staircased): a fluid slides along '
                            'it, a solid sticks to it.'),
    FeatureType('initial_velocity', 'Initial velocity', 'domain_condition', 'domain',
                params=(Param('v_x', 'float', 0.0, label='Initial velocity x', unit='m/s', field_dependent=True,
                              help='Evaluated at the +x face centre of each selected cell (the staggered vx '
                                   'position: exact for a discrete mode). Axisymmetric: v_r.',
                              tip='Velocity along x of the material at the start.'),
                        Param('v_y', 'float', 0.0, label='Initial velocity y', unit='m/s', field_dependent=True,
                              help='Evaluated at the +y face centre of each selected cell. Axisymmetric: v_z.',
                              tip='Velocity along y of the material at the start.')),
                precisions=('single',), apply=_apply_initial_velocity,
                summary='Starts the selected domains moving with a given velocity, for example a vibration mode '
                        'released at time zero.',
                description='v_x / v_y are written into the staggered velocity fields on the +x / +y faces of the '
                            'selected cells before the first step (the leap-frog start v^(-1/2), stresses 0); a later '
                            'feature wins shared faces. Faces the scheme pins (rigid walls, rigid_domain, the axis) '
                            'drop it at the first step.'),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='timeseries', apply=_apply_probe, result=_probe_result,
                validate=validate_points_inside,
                summary='Records the pressure, a stress or a velocity at a point after every time step.',
                description='Time series in the cell containing the point (a coordinate on a cell edge goes to the '
                            'upper cell), sampled after every step.'),
    resonance_monitor_feature(_RESONANCE_PROBE, _RESONANCE_SUMMARY, description_note=_RESONANCE_NOTE),
)
_FEATURES_BY_NAME = {feature.name: feature for feature in _FEATURES}
_WAVEFORM_FEATURES = frozenset({'force_source', 'volume_source', 'traction_line_source'})
_PULSED_FEATURES = _WAVEFORM_FEATURES | {'heat_pulse_source'}
_POINT_TRACE_FEATURES = ('field_probe', 'resonance_monitor')   # growable AcousticProbe traces
_RENDER_INFOS = AcousticRenderer.field_infos

ELASTIC_WAVES = PhysicsType(
    name=PHYSICS_TYPE_NAME,
    label='Elastic and Acoustic Waves, time domain (2D, in-plane)',
    default_instance_name=PHYSICS_NAME,
    family='',
    group='structural',
    spaces_2d=('planar', AXISYMMETRIC),
    # the axis needs the slip-wall x_lo (v_r = 0, s_rz = 0): Add physics in an (r, z) model sets both
    space_settings={AXISYMMETRIC: {**AXISYMMETRIC_SPACE_SETTINGS[AXISYMMETRIC], 'bc_x_lo': 'rigid'}},
    summary='Sound in fluids and elastic waves in solids, simulated step by step in time. Gives pressure, stresses '
            'and particle velocities; supports photoacoustic heating pulses.',
    description='2D velocity-stress FDTD (Virieux staggered grid; plane strain, or axisymmetric (r, z) by setting '
                'geometry): vx / vy [m/s] on the +x / +y cell faces at half steps, sxx / syy [Pa] at the cell '
                'centres, sxy on the +x +y corners; tension positive, pressure = -(sxx + syy) / 2. Fluids '
                '(shear_speed unset), isotropic solids (sound_speed / shear_speed, or youngs_modulus / poisson_ratio '
                'as in solid_mechanics and the library solids), orthotropic (c11, c12, c22, c66) and monoclinic (+ '
                'c16, c26) solids share one grid: a face takes the width-weighted mean density of its two cells, a '
                'corner the harmonic mean of its four shear moduli (0 at fluid-solid and vacuum corners), so '
                'interfaces need no special treatment and vacuum cells (no density, no stiffness) are traction-free '
                'surfaces. dt = courant x the stability limit of the build materials; alone the physics sets '
                'dt_sync, with another time-stepping physics the shorter step wins. Tensor mesh, uniform or '
                'nonuniform (auto mode meshes by the optical index). elastic_strain_xx / _yy = the stiffness inverted '
                'on the centre stresses (fluids: half the volumetric strain each; thermoelastic stress counts as '
                'minus the thermal strain); with piezo_e* or c16 / c26 arrays the accumulated total strain instead '
                '(thermoelastic heating then shows as expansion). Couplings: acoustics_em_acousto_optic (photoelastic '
                'index of an em_fdfd_* physics, re-solved on the frozen strain every k acoustic steps; the tensor '
                'entries also drive the eps_xy of em_fdfd_hz), em_acoustics_photoacoustic (absorbed light power into '
                'heating_power), electrostatics_acoustics_piezo (stress -e^T E; two-way by default: the bound charge '
                'of the accumulated strain re-solves the electrostatics every step, so the waves are stiffened and dt '
                'uses c^D); the state fields can drive a state-dependent material property or a field_transfer. '
                'Stepper precision single, or half_storage (fields stored as fp16 in units scaled from the peak '
                'source stress and the largest impedance: 10 B per cell less, ~1e-3 relative noise floor), which '
                'refuses the resonance finder, piezoelectric and monoclinic materials and every coupling into this '
                'physics. cpu backend: numpy twins (numba for the velocity / stress updates when installed), single '
                'precision only, several times slower: small models and tests.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=ACOUSTIC_SCHEMA,
    optional_material_schema=dict(OPTIONAL_ACOUSTIC_SCHEMA),
    state_fields=tuple(_RENDER_INFOS[name] for name in ('pressure', 'sxx', 'syy', 'sxy', 'vx', 'vy')),
    output_fields=tuple(_RENDER_INFOS.values()),
    source_slots={HEATING_SLOT: ('W/m^3', 'Heating rate q (em_acoustics_photoacoustic contributions summed, or one '
                                          'field_transfer), kept by reference; adds -gruneisen q x the '
                                          'heating_envelope to d(sxx)/dt and d(syy)/dt every step.'),
                  **{slot: ('V/m', f'Electric field {slot[-1]} at the cell centres (electrostatics_acoustics_piezo): '
                                   'each transfer adds -e^T (E - E_last) to the stresses (sxy at the corners from the '
                                   '4-cell mean, none where a corner has no shear stiffness), so T = c S - e^T E; the '
                                   'first transfer counts as already present (no stress: the initial state is its '
                                   'equilibrium). Ignored without piezo_e* arrays.')
                     for slot in ELECTRIC_FIELD_SLOTS}},
    supported_studies=(TIME_DEPENDENT_STUDY,),
    default_study={'type': TIME_DEPENDENT_STUDY, 'settings': {'duration': 20e-6}},
    create=_create,
    steps_in_time=lambda settings_values: True,
    validate=_validate_physics,
    memory_estimate=_memory_estimate,
    parked_material_arrays=_parked_material_arrays,
    time_estimate=_time_estimate,
    solver_roles={'stepper': HALF_STORAGE_STEPPER_ROLE},
    backends=('gpu', 'cpu'),   # core.kernels twins (numpy) on the cpu; half_storage refused there by explicit_kernels
)


def _selftest_cases():
    """The elastic_waves SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_backend import BACKEND_SELFTEST_CASES
    from .selftests_initial_velocity import INITIAL_VELOCITY_CASES
    return SELFTEST_CASES + INITIAL_VELOCITY_CASES + BACKEND_SELFTEST_CASES


def register(registry):
    """The elastic_waves type, then its 'photoacoustic_disc' template, then its analytic and cpu-vs-gpu selftest
    loader, then pressure_acoustics (definition_harmonic.register)."""
    registry.register_physics(ELASTIC_WAVES)
    registry.register_template('photoacoustic_disc', 'Photoacoustic disc', PHOTOACOUSTIC_DESCRIPTION,
                               photoacoustic_disc_spec, summary=PHOTOACOUSTIC_SUMMARY)
    registry.register_selftests(_selftest_cases)
    from .definition_harmonic import register as register_pressure_acoustics   # after elastic_waves: shares its Params
    register_pressure_acoustics(registry)

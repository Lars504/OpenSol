"""Model definition of 'em_eme' (2D eigenmode expansion along x or y: settings, 'modes' material vocabulary, port /
section-table / length-sweep / periodic / bend features, validation, memory estimate) with its self-tests and
templates; register also registers the 3D type em_eme_3d (definition3d imports the shared feature pieces from here)."""
import math
from dataclasses import replace

from ...core import time_estimate as te
from ...core.memory_estimate import COMPLEX128_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.definition_helpers import optimization_evaluations
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path, known_physics
from ...solvers.arpack_shift_invert import APPLICATIONS_PER_BASIS_GUESS, arnoldi_basis_size
from ..em.definition_common import (CPML_BC_HELP, DEFAULT_WAVELENGTH, EM_FAMILY, FREQUENCY_DOMAIN_STUDY,
                                    PERIODIC_BC_HELP, all_solved_wavelengths, apply_pec, boundary_settings_params,
                                    em_material_params, first_solved_wavelength, make_validate_physics, object_result,
                                    quasi_static_domain_requirements, shortest_solved_wavelength, source_slot_docs,
                                    unused_wavelength_issues, write_dispersive_materials)
from ..em.definition_design import OPTIMIZATION_STUDY, design_region_feature
from ..em.memory_estimates import pec_memory_estimate
from ..em.materials import EM_MATERIAL_SCHEMAS
from ..em.slab_modes import DENSE_COMPLEX_MAX_CELLS
from .adjoint import EME_OBJECTIVE_QUANTITIES, objective_result
from .renderer import EME_FIELD_INFOS
from .selftests import EME_TEMPLATES
from .solver import (LENGTH_SWEEP_LIVE_BLOCKS, MODE_CACHE_BYTES, EME_PERMITTIVITY_ARRAYS, EME_PERMITTIVITY_SLOTS,
                     EMEPhysics, length_sweep_chunk, length_sweep_result, periodic_result, ports_result,
                     section_table_result)

TYPE_NAME = 'em_eme'
_STATIONARY_STUDY = 'stationary'
PORTS_FEATURE = 'eme_ports'
_DEFAULT_PORT_MODES = 4

# ---- settings ----
_BC_HELP = ' '.join((
    'The two faces along propagation_axis are the ports: the first and last cell layers of the domain continue to '
    'infinity, so their bc is not read (pec or pmc avoids the unused padding of cpml; periodic and mirrors are '
    'refused there). The two other (transverse) faces bound every section line as in the em_fdfd_* types:',
    CPML_BC_HELP,
    'In the mode solve it is the complex coordinate stretch of the pml_* settings and the modes localised in it are '
    'kept (unlike em_modes and the mode ports): they stand for the radiation, so a radiating structure converges in '
    'n_modes more slowly.',
    PERIODIC_BC_HELP,
    'pec, pmc and the mirrors (symmetric / antisymmetric) pin the outermost cell as the em_fdfd_* walls do.',
))
_SETTINGS = (
    Param('polarization', 'choice', 'ez', label='Polarisation', choices=('ez', 'hz'),
          choice_labels=(('ez', 'Electric field out of the plane'), ('hz', 'Magnetic field out of the plane')),
          help="Field out of the xy plane: 'ez' (E_z, H_x, H_y; reads eps_r = eps_zz) or 'hz' (H_z, E_x, E_y; reads "
               'eps_xx / eps_yy). The mode lines and interfaces are those of em_fdfd_ez / em_fdfd_hz.',
          tip='Which field points out of the drawing: the electric field (light polarised out of the plane) or the '
              'magnetic field (light polarised in the plane).'),
    Param('propagation_axis', 'choice', 'x', label='Propagation axis', choices=('x', 'y'),
          choice_labels=(('x', 'Along x'), ('y', 'Along y')),
          help='Axis along which the device is cut into sections: every cell layer normal to it is a slice, '
               'consecutive equal slices form one section (so a taper is a staircase at the mesh resolution along '
               'this axis: refine the mesh there to smooth it), and its lo / hi faces are the ports.',
          tip='Direction in which the light travels through the device; its two edges are the input and output.'),
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          follows_parameters=True,
          help='Vacuum wavelength lambda0 of a stationary solve (omega = 2 pi C0 / lambda0); a frequency_domain study '
               'solves its own wavelengths instead (one cold solve each). Materials enter at omega (sigma_e and '
               'Drude / Lorentz poles complex). The line modes need >= 10 cells per lambda0 / |n| along the line '
               '(validation checks it at the build value only). A reused sweep build (sweep reuse_build) follows a '
               'value of parameters only without rebuilding (not on an auto mesh sized by it). Typical 1.31e-6 or '
               '1.55e-6 m.',
          tip='Vacuum wavelength of the light in metres: 1.55e-6 for 1550 nm. A frequency domain study uses its own '
              'list instead.'),
    Param('n_modes', 'int', 20, label='Modes per section', min=0,
          help='Modes of each section line (slab_modes ranking: descending Re(k_t^2), evanescent last, PML modes '
               'kept); 0 = every active cell of the line, a complete basis: the mode matching is then exact for the '
               'discretised line (reciprocal S and, lossless, power conserved) but each section costs O(m^3) for m '
               'line cells. A truncated basis leaves a projection error that the section table reports as the '
               'interfaces\' reciprocity defect (a run-log warning above 1e-3); raise n_modes until the port results '
               'converge. 10-40 typical, more for radiating (cpml) lines.',
          tip='How many modes describe the light in each slice of the device. More modes are more accurate and '
              'slower; 0 uses all of them.'),
    Param('section_tolerance', 'float', 0.0, label='Section tolerance', unit='1', min=0.0,
          help='Slices whose material rows (permittivity, sigma_e, mu_r, sigma_m, pole regions, PEC) differ from the '
               'first slice of the current section by at most this fraction of each row\'s largest |value| join that '
               'section (solved once, on its first slice); 0 = identical slices only. A value > 0 merges smoothly '
               'varying (e.g. heated) regions at the cost of a staircase in the material.',
          tip='How different two neighbouring slices may be and still count as one section. Keep 0 unless the '
              'material varies smoothly, e.g. after heating.'),
) + boundary_settings_params(_BC_HELP)

# ---- material vocabulary (the em_modes rows: all three permittivity arrays exist) ----
_COMPLEX_EPS = ('The mode solves use the complex permittivity at omega: this value - j sigma_e / (omega eps0) plus '
                'every Drude / Lorentz pole chi(omega).')
_MATERIAL_PROPERTIES = em_material_params('modes', {
    'eps_r': ('Isotropic relative permittivity (>= 1): sets eps_zz (polarization ez) and eps_xx, eps_yy (hz) unless '
              f'the material sets those components; may depend on x, y and physics fields. {_COMPLEX_EPS} For a '
              'dispersive material it is the background eps_inf.'),
    'eps_zz': ('z component of the relative permittivity (>= 1), the one E_z sees (polarization ez); overrides eps_r '
               'for E_z.'),
    'sigma_e': f'Ohmic loss in S/m on every E component, default 0. {_COMPLEX_EPS}',
    'mu_r': 'Real relative permeability (>= 1), 1 for practically every optical material; mu_r - j sigma_m / (omega '
            'mu0) in the solve.',
    'sigma_m': 'Magnetic loss in Ohm/m, default 0: mu_r gains -j sigma_m / (omega mu0) (artificial absorbers only).',
}, tips={'eps_zz': ('Permittivity for the electric field along z, out of the plane, used with the electric field out '
                    "of the plane. When set it replaces 'Relative permittivity' for that direction.")})

# ---- features ----
_PORTS_PARAMS = (
    Param('input_face', 'choice', 'lo', label='Input face', choices=('lo', 'hi'),
          choice_labels=(('lo', 'Low end (x_lo or y_lo)'), ('hi', 'High end (x_hi or y_hi)')),
          help='Port face of the incident mode: lo = x_lo / y_lo of propagation_axis (travelling toward +axis), hi '
               'the opposite one.',
          tip='Edge of the device where the light comes in.'),
    Param('input_mode', 'int', 1, label='Input mode', min=1,
          help='1-based rank of the incident mode among the port modes of the input face (the mode solver ranking: '
               '1 = largest Re n_eff, the PML modes of a cpml face listed after all others, so mode k is the k-th '
               'mode the mode ports would find). It must carry power (a propagating mode), else the power fractions '
               'are NaN with a run-log warning.',
          tip='Which mode of the input waveguide comes in: 1 is the fundamental mode.'),
    Param('input_power', 'float', 1.0, label='Input power', unit='W/m', min=0.0, min_exclusive=True,
          help='Power per unit length in z of the incident mode; scales only the reconstructed fields (the S-matrix '
               'and power fractions do not depend on it).',
          tip='Power of the incoming light per metre out of the plane; it only scales the field plots.'),
    Param('n_port_modes', 'int', _DEFAULT_PORT_MODES, label='Reported port modes', min=1,
          help='Port modes per face listed in the result table and the S-matrix extras, and the number of '
               'transmission_<k> scalar outputs.',
          tip='How many modes of each end are listed in the results.'),
)
_PORTS_DESCRIPTION = (
    "Ports of the EME device and its excitation. Result 'table' (one row per reported port mode of each face, the "
    'input face first = reflection, then the output face = transmission): output_face, mode_number, n_eff, n_eff_imag '
    '(of that port mode, Im < 0 lossy), s_abs2 = |S|^2 of the bilinear-normalised amplitudes, phase_deg = arg S (the '
    'reference planes are the window faces along propagation_axis; each mode is real > 0 at its peak cell), '
    'power_fraction = |S|^2 p_out / p_in (p = the flux one unit of amplitude carries: power fractions are |S|^2 for '
    'lossless propagating modes). Port mode k = the k-th mode of the first / last section, the PML modes of cpml '
    'faces listed last. Extras: wavelength, input_face, input_mode, input_power, transmission and '
    'reflection (power fractions summed over every mode of the face), s_matrix_real / s_matrix_imag = the port '
    'S-matrix [[S11, S12], [S21, S22]] over the reported modes (lo modes first; S21 = lo -> hi) and s_matrix_modes '
    "{'lo': n, 'hi': n}. Without this feature the excitation is mode 1 at 1 W/m from lo. At most one per physics.")
SECTION_TABLE_DESCRIPTION = (
    "Sections of the last solve. Result 'table', one row per section from lo to hi: section (1-based), start and "
    'length in m along propagation_axis, n_slices (cell layers), n_modes (modes found on its line), n_eff_1 / n_eff_2 '
    '(Re n_eff of its first two modes), power_defect and reciprocity_defect of the interface to the next section '
    '(NaN for the last): power = max over the propagating incident modes of |1 - power reflected and transmitted into '
    'propagating modes| (meaningful for lossless lines without cpml only), reciprocity = max |S12 - S21^T| / max |S21| '
    '(0 for a complete basis; the convergence measure in n_modes). Extra wavelength.')


def port_params():
    """eme_ports Params of em_eme (em_eme_3d replaces the face labels and the power unit)."""
    return _PORTS_PARAMS


def apply_ports(ctx, solver, feature_spec, values, selection):
    return solver.add_ports(values['input_face'], values['input_mode'], values['input_power'],
                            values['n_port_modes'])


def apply_section_table(ctx, solver, feature_spec, values, selection):
    return solver.add_section_table()


def validate_ports(vctx, physics_spec, feature_spec, values):
    """input_mode above a non-zero n_modes: error; a second enabled eme_ports: error."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    n_modes = _settings_value(physics_spec, 'n_modes', 20)
    if isinstance(n_modes, int) and 0 < n_modes < values['input_mode']:
        issues.append(Issue('error', path, f"input_mode {values['input_mode']} exceeds n_modes {n_modes}"))
    ports = [feature.name for feature in physics_spec.features if feature.enabled and feature.type == PORTS_FEATURE]
    if len(ports) > 1 and ports[0] != feature_spec.name:
        issues.append(Issue('error', path, f"only one eme_ports feature per physics (the excitation): "
                                           f"'{ports[0]}' is already one"))
    return issues


def length_sweep_params(axis_text):
    """eme_length_sweep Params; axis_text names the propagation axis in the help texts."""
    return (
        Param('start', 'float', 0.0, label='Range start', unit='m',
              help=f'Lower end of the stretched range along {axis_text} [m, model coordinates]: every section whose '
                   'centre lies in [start, stop] gets its length times each stretch factor.',
              tip='Where the part of the device whose length is varied begins, along the propagation direction.'),
        Param('stop', 'float', 1e-6, label='Range stop', unit='m',
              help=f'Upper end of the stretched range along {axis_text} [m], > start.',
              tip='Where the part of the device whose length is varied ends.'),
        Param('stretch_factors', 'float_list', [0.5, 1.0, 1.5, 2.0], label='Stretch factors', unit='1', min=0.0,
              min_exclusive=True,
              help='Factors applied to the lengths of the sections in the range, one re-cascade each (no new mode '
                   'solve: the modes and interfaces of the solve are reused), e.g. 0.5 .. 4 for a taper length scan.',
              tip='How much the part is stretched for each point of the curve: 2 makes it twice as long.'),
    )


LENGTH_SWEEP_DESCRIPTION = (
    "Cheap length scan of the last solve: the sections whose centre lies in [start, stop] are re-propagated with their "
    'lengths times each stretch factor and the S-matrix re-cascaded from the kept interfaces (no mode solve; a taper '
    "staircase is stretched step by step, so it stays the same taper, longer). Result 'xy' versus the stretched length "
    '[m] of the range (in factor order): series transmission_<k> (k = 1 .. n_port_modes of eme_ports), transmission '
    'and reflection power fractions of the excitation, as the scalar outputs; extras stretch_factors, wavelength, '
    'n_sections_stretched. Keeps the interface S-matrices of every solve (4 N^2 complex per section).')


def apply_length_sweep(ctx, solver, feature_spec, values, selection):
    return solver.add_length_sweep(values['start'], values['stop'], values['stretch_factors'])


def validate_length_sweep(vctx, physics_spec, feature_spec, values):
    """stop <= start or an empty factor list: error."""
    path = feature_path(physics_spec, feature_spec)
    issues = _range_issues(values, path)
    if not values['stretch_factors']:
        issues.append(Issue('error', path, 'stretch_factors is empty'))
    return issues


def length_sweep_feature(axis_text, summary):
    """FeatureType eme_length_sweep (2D and 3D)."""
    return FeatureType('eme_length_sweep', 'Length sweep', 'monitor', None, params=length_sweep_params(axis_text),
                       result_kind='xy', description=LENGTH_SWEEP_DESCRIPTION, summary=summary,
                       apply=apply_length_sweep, result=object_result(length_sweep_result),
                       validate=validate_length_sweep)


_LENGTH_SWEEP_SUMMARY = ('Transmission of the device when a part of it is made longer or shorter, without solving '
                         'its modes again: for example the transmission of a taper versus its length.')


def _range_issues(values, path):
    """stop <= start: error."""
    if values['stop'] > values['start']:
        return []
    return [Issue('error', path, f"stop ({values['stop']:.6g} m) must exceed start ({values['start']:.6g} m)")]


def _range_params(axis_text, what, start_tip, stop_tip):
    """start / stop Params of a range along the propagation axis selecting the sections whose centre lies in it."""
    return (Param('start', 'float', 0.0, label='Range start', unit='m',
                  help=f'Lower end of the {what} along {axis_text} [m, model coordinates]: the sections (cell layers) '
                       'whose centre lies in [start, stop] belong to it.', tip=start_tip),
            Param('stop', 'float', 1e-6, label='Range stop', unit='m',
                  help=f'Upper end of the {what} along {axis_text} [m], > start.', tip=stop_tip))


def periodic_params(axis_text):
    """eme_periodic Params; axis_text names the propagation axis in the help texts."""
    return _range_params(axis_text, 'unit cell', 'Where the drawn period of the grating begins along the propagation '
                                                 'direction.', 'Where the drawn period ends.') + (
        Param('n_periods', 'int', 10, label='Number of periods', min=1,
              help='Periods of the device: the drawn unit cell stands for n_periods copies in a row (its S-matrix '
                   'raised to that power by repeated star products, cost log2(n_periods)); the sections after the '
                   'range follow the last period, so the hi port plane is (n_periods - 1) periods beyond the window '
                   'face. 1 = only the Bloch analysis.',
              tip='How many times the drawn period repeats. The output part of the drawing follows the last one.'),
        Param('n_bloch_modes', 'int', 4, label='Reported Bloch modes', min=1,
              help='Forward Bloch modes of the unit cell listed in the result table, least attenuated first.',
              tip='How many Bloch modes of the period are listed in the results.'))


PERIODIC_DESCRIPTION = (
    "Periodic block (gratings, Bragg mirrors, band edges): the sections whose centre lies in [start, stop] are one "
    'period, drawn once and repeated n_periods times. The unit cell S-matrix (the junction from its last to its first '
    'section, then its sections) is raised to the power n_periods - 1 by repeated Redheffer star products and '
    'inserted after the drawn period, so the port results, scalars and length sweep see the whole periodic device '
    '(at least one section must follow the range when n_periods > 1); the field plot shows the first period, the '
    'sections after it carry the field leaving the last period. Bloch modes: the eigenvectors of the unit cell, '
    'fields one period on = lam times the fields (lam = exp(-j K period)), from the generalised eigenproblem of its '
    "S-matrix. Result 'table' (forward modes: |lam| < 1, or on the unit circle with net forward power; least "
    'attenuated first): bloch_mode, n_bloch = Re K / k0 (unfolded to the Brillouin zone nearest the length-weighted '
    'mean Re n_eff of mode 1 over the cell), n_bloch_imag = Im K / k0 (< 0: decaying, a stop band when the dominant '
    'mode decays without loss), multiplier_abs = |lam| per period, phase_per_period_deg = Re K period [deg, in (-180, '
    "180]], dominant_mode = the cell mode carrying most of the Bloch mode. Extras wavelength, period [m], n_periods, "
    'first_section / last_section (1-based section table rows). At most one per physics.')
_PERIODIC_SUMMARY = ('A grating or other repeated structure drawn as one period: repeats it many times at almost no '
                     'cost and lists its Bloch modes (band gaps).')


def apply_periodic(ctx, solver, feature_spec, values, selection):
    return solver.add_periodic(values['start'], values['stop'], values['n_periods'], values['n_bloch_modes'])


def validate_periodic(vctx, physics_spec, feature_spec, values):
    """stop <= start, a second enabled eme_periodic: error."""
    path = feature_path(physics_spec, feature_spec)
    issues = _range_issues(values, path)
    blocks = [feature.name for feature in physics_spec.features if feature.enabled and feature.type == 'eme_periodic']
    if len(blocks) > 1 and blocks[0] != feature_spec.name:
        issues.append(Issue('error', path, f"only one eme_periodic feature per physics: '{blocks[0]}' is already one"))
    return issues


def periodic_feature(axis_text):
    """FeatureType eme_periodic (2D and 3D)."""
    return FeatureType('eme_periodic', 'Periodic block', 'monitor', None, params=periodic_params(axis_text),
                       result_kind='table', description=PERIODIC_DESCRIPTION, summary=_PERIODIC_SUMMARY,
                       apply=apply_periodic, result=object_result(periodic_result), validate=validate_periodic)


def bend_params(axis_text, reference_text, reference_tip):
    """eme_bend Params; reference_text describes the transverse reference coordinate."""
    return _range_params(axis_text, 'bend', 'Where the bent part begins, measured along the straightened guide.',
                         'Where the bent part ends, measured along the straightened guide.') + (
        Param('radius', 'float', 10e-6, label='Bend radius', unit='m',
              help=f'Radius R [m] of the reference line ({reference_text}); > 0 bends toward lower coordinates (the '
                   'centre of curvature at reference - R), < 0 the other way. The bend angle is the range length / '
                   '|R|; the centre of curvature must lie outside the section line or cross-section.',
              tip='Radius of the bend at the reference position. Its sign sets the direction it turns.'),
        Param('reference', 'float', 0.0, label='Reference position', unit='m',
              help=f'{reference_text} [m, model coordinates] of the line whose radius is R: the drawn range length is '
                   'its arc length and the section n_eff its beta / k0.', tip=reference_tip))


def bend_validate(vctx, physics_spec, feature_spec, values):
    """stop <= start, radius 0, overlapping enabled bends: error."""
    path = feature_path(physics_spec, feature_spec)
    issues = _range_issues(values, path)
    if values['radius'] == 0.0:
        issues.append(Issue('error', path, 'radius must be non-zero (no bend: remove the feature)'))
    for feature in physics_spec.features:
        if feature.name == feature_spec.name:
            break
        if not (feature.enabled and feature.type == 'eme_bend' and isinstance(feature.params, dict)):
            continue
        start, stop = feature.params.get('start', 0.0), feature.params.get('stop', 1e-6)
        if isinstance(start, (int, float)) and isinstance(stop, (int, float)) and (
                values['start'] <= stop and start <= values['stop']):
            issues.append(Issue('error', path, f"its range overlaps eme_bend '{feature.name}'"))
    return issues


def apply_bend(ctx, solver, feature_spec, values, selection):
    solver.add_bend(values['start'], values['stop'], values['radius'], values['reference'])
    return None


def bend_feature(axis_text, reference_text, reference_tip, description):
    """FeatureType eme_bend (2D and 3D)."""
    return FeatureType('eme_bend', 'Bend', 'domain_condition', None,
                       params=bend_params(axis_text, reference_text, reference_tip), description=description,
                       summary='Makes part of the drawn straight waveguide a circular bend of the given radius: the '
                               'drawing is the bend straightened out.',
                       apply=apply_bend, validate=bend_validate)


BEND_DESCRIPTION_2D = (
    'Circular bend: the sections whose centre lies in [start, stop] along propagation_axis are a bend of radius R '
    'about the line coordinate reference, drawn straightened (the range length = the arc length at the reference, '
    'the angle = length / |R|). Exact for the 2D line: each bent section solves the cylindrical operator (r / R^2) '
    'd/dr (r du/dr) + k0^2 q (r / R)^2 (r = R + line coordinate - reference; ez q = eps mu, hz with the eps weights '
    'of the em_fdfd_hz line), so its n_eff = beta / k0 at r = R, propagation e^{-j beta R theta}, and the straight - '
    'bent interfaces are mode-matched like any other (radiation of a tight bend goes into the PML modes of cpml line '
    'faces). The field plot shows the straightened bend (the in-plane component along the axis is H_theta / '
    'E_theta); absorbed_power is the physical density, so its integral over the straightened drawing misses the '
    'factor r / R. Ranges must not overlap.')


# ---- adjoint objective and design region (adjoint.py; optimization study) ----
_OBJECTIVE_MODE_QUANTITIES = ('transmission_mode', 'reflection_mode')
_OBJECTIVE_PARAMS = (
    Param('quantity', 'choice', 'transmission_mode', label='Quantity', choices=EME_OBJECTIVE_QUANTITIES,
          choice_labels=(('transmission_mode', 'Transmission into one mode'),
                         ('reflection_mode', 'Reflection into one mode'),
                         ('transmission', 'Total transmission'), ('reflection', 'Total reflection')),
          help='Power fraction of the excitation (eme_ports, else mode 1 from lo) this term measures, as the scalar '
               'outputs: transmission_mode = into port mode mode_number of the output face (transmission_<k>), '
               'reflection_mode = back into port mode mode_number of the input face, transmission / reflection = '
               'summed over every mode of that face. J = the sum of weight * quantity over the enabled '
               'adjoint_objective features.',
          tip='What this objective measures: the light reaching one mode or all modes of the output, or the light '
              'reflected back.'),
    Param('mode_number', 'int', 1, label='Mode number', min=1, visible_if=(('quantity', _OBJECTIVE_MODE_QUANTITIES),),
          help='1-based port mode of transmission_mode / reflection_mode (the eme_ports ranking: 1 = largest Re n_eff, '
               'PML modes last); at most n_modes when n_modes > 0.',
          tip='Which mode is measured: 1 is the fundamental mode.'),
    Param('weight', 'float', 1.0, label='Weight', unit='1',
          help='Factor of this term in J: 1 to maximise the quantity, -1 to minimise it, 0 to record it only.',
          tip='How much this measurement counts in the total objective: 1 to increase it, -1 to decrease it, 0 to only '
              'record it.'),
)
_OBJECTIVE_DESCRIPTION = (
    'Objective term of the EME cascade adjoint: weight times a port power fraction of the excitation. Every solve '
    'records the quantity; the output field eps_sensitivity is dJ/d eps per unit area [1/m^2] of J = the sum of '
    'weight * quantity (ez: eps_r; hz: the sum of the eps_xx and eps_yy terms, both differentiated: the line cells '
    'and the face means along the line), from the forward and one adjoint cascade field (Lorentz '
    'reciprocity of the section equation: no extra mode solve; computed on the slices of the design regions, on every '
    'slice without one, when the field is read). Exact for a complete basis (n_modes 0), otherwise the same formula on '
    'the truncated fields (converging with n_modes). The drawn cell of an eme_periodic block stands for all its '
    'n_periods periods: its gradient sums the field products of every period. The port modes (first and last cell '
    'layer along propagation_axis) are held fixed; sigma_e, poles and mu_r are not differentiated. An optimization '
    'study '
    "maximises J by the design_region densities. Result 'xy' versus vacuum wavelength: series value and weighted "
    '(unit 1); extras frequency, wavelength, quantity, mode_number, weight.')


def _settings_value(physics_spec, name, default):
    """Raw setting `name` of a physics spec, else default."""
    settings = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
    return settings.get(name, default)


def _validate_objective(vctx, physics_spec, feature_spec, values):
    """mode_number above a non-zero n_modes: error; optimization study without an enabled design_region of this
    physics: error."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    n_modes = _settings_value(physics_spec, 'n_modes', 20)
    if (values['quantity'] in _OBJECTIVE_MODE_QUANTITIES and isinstance(n_modes, int)
            and 0 < n_modes < values['mode_number']):
        issues.append(Issue('error', path, f"mode_number {values['mode_number']} exceeds n_modes {n_modes}"))
    if vctx.study_type == OPTIMIZATION_STUDY and not any(
            feature.enabled and feature.type == 'design_region' for feature in physics_spec.features):
        issues.append(Issue('error', path, 'no enabled design_region on this physics writes the permittivity this '
                                           'objective reads, so it adds a constant to J and no gradient; add one or '
                                           'disable this objective'))
    return issues


def _apply_objective(ctx, solver, feature_spec, values, selection):
    return solver.add_objective(values['quantity'], values['mode_number'], values['weight'])


# no memory estimate of its own: one EpsSensitivity per physics, counted once by the physics estimators
OBJECTIVE_FEATURE = FeatureType(   # definition3d replaces its description
    'adjoint_objective', 'Optimization objective', 'monitor', None, params=_OBJECTIVE_PARAMS, result_kind='xy',
    description=_OBJECTIVE_DESCRIPTION,
    summary='Marks a transmitted or reflected power fraction as a goal, so the solver also maps where a small change '
            'of the material would raise or lower it. Use it to optimise the device.',
    apply=_apply_objective, result=object_result(objective_result), validate=_validate_objective)
_FDFD_DESIGN_REGION = design_region_feature()
_DESIGN_DESCRIPTION = (
    'Density design region on the selected domains: each window cell carries a density rho in [0, 1], filtered by a '
    'conic filter of filter_radius and projected by a tanh of projection_beta / projection_eta to rho_bar, and gets '
    'eps = eps_min + rho_bar (eps_max - eps_min) (polarization ez: eps_r; hz: eps_xx and eps_yy, an isotropic '
    'change), overwriting the material permittivity there '
    "(sigma_e, poles and mu_r stay the material's). The same map and chain rule as the em_fdfd_* design_region. The "
    'optimization study varies rho to maximise the adjoint_objective sum J; any other study solves the '
    'initial_density design. Keep the region off the first and last cell layer along propagation_axis (the port '
    'sections: the build refuses it). Every distinct slice of the region becomes its own section (one mode solve '
    'each per evaluation). Design regions of one physics must not share cells. Output field design_density = rho_bar '
    "(0 outside every region). Result 'xy' versus iteration (0 = initial design): series objective (J of the whole "
    'model, unit 1: power fractions) and grayness (mean 4 rho_bar (1 - rho_bar), 0 = binary); extras n_cells, eps_min, eps_max, mean_density, '
    'final_grayness, note.')


_FEATURES = (
    FeatureType(PORTS_FEATURE, 'Ports', 'source', None, params=_PORTS_PARAMS, result_kind='table',
                description=_PORTS_DESCRIPTION,
                summary='The input and output of the device: which mode comes in, and how much of it is reflected '
                        'and transmitted into each mode.',
                apply=apply_ports, result=object_result(ports_result), validate=validate_ports),
    FeatureType('eme_section_table', 'Section table', 'monitor', None, result_kind='table',
                description=SECTION_TABLE_DESCRIPTION,
                summary='Lists the slices the device was cut into, with their length, effective indices and how '
                        'accurately neighbouring slices are joined.',
                apply=apply_section_table, result=object_result(section_table_result)),
    length_sweep_feature('propagation_axis', _LENGTH_SWEEP_SUMMARY),
    periodic_feature('propagation_axis'),
    bend_feature('propagation_axis', 'the coordinate along the line axis (the other in-plane axis)',
                 'Position across the waveguide, usually its centre, at which the bend has exactly the given radius.',
                 BEND_DESCRIPTION_2D),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor: the cells whose centre lies in the selected domains are '
                            'inactive in every section line (u = 0 for ez; hz faces next to them carry no tangential '
                            'E), staircased to the mesh.',
                summary='Turns the selected domains into a perfect metal, following the mesh cells.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0)),
    OBJECTIVE_FEATURE,
    replace(_FDFD_DESIGN_REGION, description=_DESIGN_DESCRIPTION),
)


# ---- validation ----
_SHARED_VALIDATE = make_validate_physics(EME_PERMITTIVITY_SLOTS, time_domain=False,
                                         solved_wavelengths=all_solved_wavelengths)


def other_em_issues(vctx, path, type_name=TYPE_NAME):
    """Error when another family-'em' type shares the model with type_name (one EM formulation per model)."""
    others = [f"'{physics.name}' ({physics_type.name})"
              for physics, physics_type in known_physics(vctx.spec, vctx.registry)
              if physics_type.family == EM_FAMILY and physics_type.name != type_name]
    if not others:
        return []
    return [Issue('error', path, f"{type_name} solves the model by eigenmode expansion and cannot share it with "
                                 f"{', '.join(others)}; use a separate model")]


def _face_issues(settings_values, path):
    """Propagation-axis faces: periodic / mirrors error, cpml info (unused padding); transverse cpml info."""
    axis = settings_values['propagation_axis']
    line_axis = 'y' if axis == 'x' else 'x'
    issues = []
    for side in ('lo', 'hi'):
        face = f'{axis}_{side}'
        value = settings_values[f'bc_{face}']
        if value in ('periodic', 'symmetric', 'antisymmetric'):
            issues.append(Issue('error', path, f"bc_{face} = {value}: the faces along propagation_axis are "
                                               f"semi-infinite ports (periodic sections are not supported); use pec"))
        elif value == 'cpml':
            issues.append(Issue('info', path, f'bc_{face} = cpml adds PML cells the EME never reads (its ports are '
                                              f'semi-infinite); pec avoids that memory'))
    if 'cpml' in (settings_values[f'bc_{line_axis}_lo'], settings_values[f'bc_{line_axis}_hi']):
        issues.append(Issue('info', path, f'cpml on the {line_axis} faces: the section modes localised in the PML '
                                          f'are kept and carry the radiation, so the results converge more slowly in '
                                          f'n_modes; check the section table reciprocity_defect'))
    return issues


def _line_cells(mesh, settings_values):
    """(line cells incl. padding, window slices along the propagation axis) of a MeshEstimate."""
    axis = settings_values['propagation_axis']
    sides = mesh.npml_sides
    if axis == 'x':
        return int(mesh.ny), int(mesh.nx) - sides['x_lo'] - sides['x_hi']
    return int(mesh.nx), int(mesh.ny) - sides['y_lo'] - sides['y_hi']


def _size_issues(vctx, settings_values, path):
    """Info on the slice count and the line size; n_modes above the line cells: info (capped)."""
    if vctx.mesh is None:
        return []
    line_cells, slices = _line_cells(vctx.mesh, settings_values)
    n_modes = settings_values['n_modes']
    issues = [Issue('info', path, f'{slices} slices along {settings_values["propagation_axis"]} (at most that many '
                                  f'sections, one mode solve per distinct section) on lines of {line_cells} cells')]
    if n_modes > line_cells:
        issues.append(Issue('info', path, f'n_modes {n_modes} exceeds the {line_cells} line cells: every mode of a '
                                          f'line is used (a complete basis, as n_modes = 0)'))
    return issues


def _optimization_issues(vctx, settings_values, path):
    """Optimization study: error for section_tolerance > 0 (the EME misses small design changes there); info on a
    truncated basis (approximate gradient)."""
    if vctx.study_type != OPTIMIZATION_STUDY:
        return []
    issues = []
    if settings_values['section_tolerance'] > 0.0:
        issues.append(Issue('error', path, 'section_tolerance > 0 merges slices whose permittivity differs, so the EME '
                                           'does not follow small design changes there (no gradient); use 0'))
    if settings_values['n_modes'] != 0:
        issues.append(Issue('info', path, f"n_modes {settings_values['n_modes']}: the adjoint gradient is exact for a "
                                          'complete basis (n_modes 0) and approximate for a truncated one, which can '
                                          'slow the L-BFGS-B line search'))
    return issues


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared EM checks (pairing, materials, slot writers, resolution at every solved wavelength), the unused
    wavelength setting, other EM types, the port / transverse faces, the sizes and the optimization rules."""
    path = ('physics', physics_spec.name)
    issues = _SHARED_VALIDATE(vctx, physics_spec, settings_values)
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(other_em_issues(vctx, path))
    issues.extend(_face_issues(settings_values, path))
    issues.extend(_size_issues(vctx, settings_values, path))
    issues.extend(_optimization_issues(vctx, settings_values, path))
    return issues


# ---- build, scalars, memory ----
def _create(ctx, physics_spec, settings_values):
    """EMEPhysics at the first solved wavelength with its dispersive poles and their background permittivity written
    into eps_r, eps_xx, eps_yy; never finalizes."""
    solver = EMEPhysics(ctx.domain, polarization=settings_values['polarization'],
                        propagation_axis=settings_values['propagation_axis'],
                        boundary_config=ctx.domain.boundary_config,
                        wavelength=first_solved_wavelength(ctx, settings_values), n_modes=settings_values['n_modes'],
                        section_tolerance=settings_values['section_tolerance'])
    write_dispersive_materials(ctx, physics_spec, solver, EME_PERMITTIVITY_ARRAYS)
    follow_wavelength_setting(ctx, physics_spec, settings_values, solver)
    return solver


def follow_wavelength_setting(ctx, physics_spec, settings_values, solver):
    """A reused sweep build re-sets the wavelength setting (EMEBase.set_wavelength; a frequency_domain study solves
    its own wavelengths, the setting is unused there). 2D and 3D create."""
    if ctx.study_type != FREQUENCY_DOMAIN_STUDY:
        ctx.follow_parameters(physics_spec.settings, settings_values, ('wavelength',), solver.set_wavelength, solver,
                              f"physics '{physics_spec.name}' wavelength")


def _port_mode_count(physics_spec):
    """n_port_modes of the first enabled eme_ports feature (plain int), else the default."""
    for feature in physics_spec.features:
        if feature.enabled and feature.type == PORTS_FEATURE and isinstance(feature.params, dict):
            value = feature.params.get('n_port_modes', _DEFAULT_PORT_MODES)
            return value if isinstance(value, int) and value >= 1 else _DEFAULT_PORT_MODES
    return _DEFAULT_PORT_MODES


def scalar_outputs(settings_values, physics_spec):
    """transmission_1 .. transmission_<n_port_modes>, transmission, reflection (EMEBase.scalars)."""
    return (*(f'transmission_{number}' for number in range(1, _port_mode_count(physics_spec) + 1)),
            'transmission', 'reflection')


SCALAR_OUTPUTS_HELP = (
    'Power fractions (unit 1) of the excitation (eme_ports, else mode 1 from lo) of the last solve: '
    'transmission_<k> into port mode k of the output face (k = 1 .. n_port_modes, modes found only), transmission '
    'and reflection summed over every mode of the output / input face (|S|^2 p_out / p_in; lossless: 1 = '
    'transmission + reflection, radiation into kept PML modes included). A missing value (an evanescent input mode, '
    'too few modes) gives NaN plus a run-log warning. Example derived value {"name": "T1", "expression": '
    '"eme_transmission_1"}.')


# device float64 temporaries of adjoint.sensitivity_density (cell areas, the gradient sum, the quotient) alive at once
_SENSITIVITY_TEMPORARIES = 3


def length_sweep_host_items(features, n_modes):
    """Host item of the stacked length-sweep re-cascade (one chunk of stretch factors; the monitors record one after
    another), () without an enabled eme_length_sweep (2D and 3D)."""
    counts = [len(values) if isinstance(values, (list, tuple)) else _DEFAULT_STRETCH_FACTORS
              for values in (feature.params.get('stretch_factors') for feature in features
                             if feature.type == 'eme_length_sweep' and isinstance(feature.params, dict))]
    if not counts:
        return ()
    batch = min(max(counts), length_sweep_chunk(n_modes))
    return ((f'length-sweep stacked star products ({LENGTH_SWEEP_LIVE_BLOCKS} N^2 per stretch factor of a chunk)',
             batch * LENGTH_SWEEP_LIVE_BLOCKS * COMPLEX128_BYTES * n_modes ** 2),)


def _memory_estimate(settings_values, mesh, context):
    """PhysicsType.memory_estimate of em_eme: 6 float32 device grids, the adjoint sensitivity cache (run item) and the
    upload / sensitivity transients; host items mostly (comments)."""
    line_cells, slices = _line_cells(mesh, settings_values)
    n_modes = min(settings_values['n_modes'] or line_cells, line_cells)
    n_cells = int(mesh.n_cells)
    mode_set_bytes = COMPLEX128_BYTES * (3 * n_modes * line_cells + line_cells + 2 * n_modes)
    n_rows = 7   # read arrays (hz 5), pole regions, PEC: a typical count
    # host: mode-set cache (capped), per section 4 N^2 partial cascades + 4 N^2 interfaces (every slice taken as a
    # section: an upper figure), the material stack, the reconstructed field and the q_abs inputs
    host_items = (
        ('mode-set cache (capped)', min(slices * mode_set_bytes, MODE_CACHE_BYTES)),
        ('partial cascades and interfaces (8 N^2 per section)', 8 * COMPLEX128_BYTES * n_modes ** 2 * slices),
        ('host material stack', n_rows * FLOAT64_BYTES * n_cells),
        ('reconstructed field (complex u, in-plane |field|^2)', (COMPLEX128_BYTES + FLOAT64_BYTES) * n_cells),
        ('absorbed power inputs (Gauss mean |E|^2, loss)', 2 * FLOAT64_BYTES * n_cells),
    )
    run_items = []
    # _reconstruct uploads one float32 full grid at a time (field arrays, q_abs) and frees it after the copy
    transient_items = [('field reconstruction', 'field upload (float32, one array at a time)',
                        FLOAT32_BYTES * n_cells)]
    features = [feature for feature in context.physics_spec.features if feature.enabled]
    adjoint = any(feature.type == 'adjoint_objective' for feature in features)
    if adjoint:   # one EpsSensitivity per physics (EMEBase._sensitivity), however many objectives
        n_gradients = 1 if settings_values['polarization'] == 'ez' else 2   # ez eps_r; hz eps_xx, eps_yy
        run_items.append((f'adjoint sensitivity cache ({n_gradients} x float64 + float32)',
                          (n_gradients * FLOAT64_BYTES + FLOAT32_BYTES) * n_cells))
        transient_items.append(('adjoint sensitivity', f'sensitivity density temporaries '
                                f'({_SENSITIVITY_TEMPORARIES} x float64)',
                                _SENSITIVITY_TEMPORARIES * FLOAT64_BYTES * n_cells))
        host_items += ((f'host adjoint gradients ({n_gradients} x float64)', n_gradients * FLOAT64_BYTES * n_cells),)
    periods = [feature.params.get('n_periods', 10) for feature in features
               if feature.type == 'eme_periodic' and isinstance(feature.params, dict)]
    if adjoint and periods and isinstance(periods[0], int) and periods[0] > 1:
        host_items += (('periodic adjoint: partial cascades between the periods (2 N^2 each)',
                        2 * COMPLEX128_BYTES * n_modes ** 2 * periods[0]),)
    host_items += length_sweep_host_items(features, n_modes)
    return MemoryEstimate.from_items((('EME fields and q_abs (6 x float32)', 6 * FLOAT32_BYTES * n_cells),),
                                     host_items=host_items, run_items=run_items, transient_items=transient_items)


# ---- run-time estimate (shared with definition3d) ----
# distinct sections per solve as a fraction of the window slices (calibrate from summary solvers.<physics> counters
# sections / (solves x slices)): a taper makes every slice a section, a straight MMI or coupler a handful
SECTIONS_PER_SLICE_GUESS = 0.25
# gemm-rate equivalent host flops of the 2D line eigensolvers (LAPACK runs below the gemm rate; calibrate from the
# mode-solve share of solve_seconds): dense complex eigvals per m^3, dense real eigh per m^3 (periodic lines), real
# tridiagonal bisection + inverse iteration per m x mode
COMPLEX_LINE_EIG_FLOPS = 50.0
REAL_LINE_EIG_FLOPS = 10.0
TRIDIAGONAL_EIG_FLOPS = 100.0
# complex multiply-adds / N^3 of one Redheffer star: 2 LUs (2/3 each) with 2 N right-hand columns (2 per N), 8 products
STAR_COMPLEX_PRODUCTS = 2.0 * (2.0 / 3.0 + 4.0) + 8.0
_MATERIAL_ROWS = 7             # read arrays, pole regions, PEC: as the memory estimate
_DEFAULT_STRETCH_FACTORS = 4   # eme_length_sweep stretch_factors default length


def physics_counter(context, name):
    """summary.json path of the EME physics' own work counter `name` (EMEBase.work_counters)."""
    return ('solvers', context.name, 'counters', name)


def lossy_materials(context):
    """True when a material sets sigma_e / sigma_m or carries Drude / Lorentz poles (complex section operators)."""
    if context.material_sets('sigma_e') or context.material_sets('sigma_m'):
        return True
    from ...materials import effective_material_data   # lazy: materials imports physics.em
    for material in context.spec.materials:
        try:
            _, dispersion = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, KeyError, TypeError):   # reported by validate
            continue
        if dispersion:
            return True
    return False


def eme_solves(context):
    """(EME solves, count kind, limit): stationary = the static solves, frequency_domain x its wavelengths,
    optimization its design evaluations (a guess below the evaluation limit)."""
    if context.study.type == OPTIMIZATION_STUDY:
        return optimization_evaluations(context)
    solves, kind, limit = context.static_solves()
    if context.study.type == FREQUENCY_DOMAIN_STUDY:
        points = context.count('harmonic_points', 1)
        return solves * points, kind, None if limit is None else limit * points
    return solves, kind, limit


def _star_flops(n_modes):
    return 8.0 * STAR_COMPLEX_PRODUCTS * float(n_modes) ** 3


def shared_time_estimate(context, slices, n_modes, transverse_points, solve_ops, mode_ops, mode_set_bytes):
    """(TimeEstimate, (mode solves, limit)) of the work em_eme / em_eme_3d share (timed by solve_seconds): per solve
    solve_ops, per section (SECTIONS_PER_SLICE_GUESS of the slices, at most all) mode_ops, its cascade and interface,
    field re-solves beyond the mode cache, length-sweep re-cascades, the periodic link and Bloch modes."""
    solves, kind, limit = eme_solves(context)
    bound = solves if kind != 'guess' else limit
    per_solve = max(1.0, SECTIONS_PER_SLICE_GUESS * slices)
    sections = solves * per_solve
    section_limit = None if bound is None else bound * slices
    cached = MODE_CACHE_BYTES // max(int(mode_set_bytes), 1)
    resolves = solves * max(per_solve - cached, 0.0)
    resolve_limit = None if bound is None else bound * max(slices - cached, 0)
    seconds = physics_counter(context, 'solve_seconds')
    star = te.DenseFlops(_star_flops(n_modes), host=True)
    interface = (te.DenseFlops(8.0 * (2.0 * n_modes ** 2 * transverse_points + 10.0 * n_modes ** 3), host=True),
                 te.PythonCalls(20))   # two overlaps, two reflections, transmissions, defects
    items = [te.TimeItem('EME solves (sections search, field reconstruction)', 'solve', solves, kind, solve_ops,
                         limit=limit, counter=physics_counter(context, 'solves'), seconds_counter=seconds),
             te.TimeItem('section mode solves', 'section', sections, 'guess', mode_ops, limit=section_limit,
                         counter=physics_counter(context, 'mode_solves'), seconds_counter=seconds),
             te.TimeItem('section cascade (partial cascades, amplitudes)', 'section', sections, 'guess',
                         ((1.0, star), te.PythonCalls(30)), limit=section_limit,   # two half stars per section
                         counter=physics_counter(context, 'sections'), seconds_counter=seconds),
             te.TimeItem('interface S-matrices', 'interface', solves * (per_solve - 1.0), 'guess', interface,
                         limit=None if bound is None else bound * max(slices - 1, 0),
                         counter=physics_counter(context, 'interfaces'), seconds_counter=seconds)]
    if resolves > 0:
        items.append(te.TimeItem('field re-solves (beyond the mode cache)', 'section', resolves, 'guess', mode_ops,
                                 limit=resolve_limit, counter=physics_counter(context, 'mode_resolves'),
                                 seconds_counter=seconds))
    features = [feature for feature in context.physics_spec.features
                if feature.enabled and isinstance(feature.params, dict)]
    sweeps = [feature.params.get('stretch_factors') for feature in features if feature.type == 'eme_length_sweep']
    factors = sum(len(values) if isinstance(values, (list, tuple)) else _DEFAULT_STRETCH_FACTORS for values in sweeps)
    periods = [feature.params.get('n_periods', 10) for feature in features if feature.type == 'eme_periodic']
    periodic = bool(periods) and isinstance(periods[0], int)
    # unit cell, star powers by squaring, link (assemble_total rebuilds it for every stretch factor too)
    link_stars = 2 * math.ceil(math.log2(max(periods[0], 2))) + 2 if periodic else 0
    if factors:
        items.append(te.TimeItem('length-sweep re-cascades', 'section', sections * factors, 'guess', (star,),
                                 limit=None if section_limit is None else section_limit * factors,
                                 seconds_counter=seconds))
        if link_stars:
            items.append(te.TimeItem('length-sweep periodic links', 'solve', solves * factors, kind,
                                     ((link_stars, star),), limit=None if limit is None else limit * factors,
                                     seconds_counter=seconds))
    if periodic:
        bloch = te.DenseFlops(3.0 * COMPLEX_LINE_EIG_FLOPS * (2.0 * n_modes) ** 3, host=True)   # generalised eig
        items.append(te.TimeItem('periodic link and Bloch modes', 'solve', solves, kind, ((link_stars, star), bloch),
                                 limit=limit, seconds_counter=seconds))
    mode_limit = None if section_limit is None else section_limit + resolve_limit
    return te.TimeEstimate(tuple(items)), (sections + resolves, mode_limit)


def adjoint_time_estimate(context, n_modes, n_cells):
    """Optimization: one cascade adjoint gradient per design evaluation (a guess: field products of the design slices);
    elsewhere an enabled adjoint_objective is not estimated (computed only when eps_sensitivity is read)."""
    if context.study.type == OPTIMIZATION_STUDY:
        ops = (te.DenseFlops(8.0 * 24.0 * n_modes * n_cells, host=True), te.HostStream(6.0 * FLOAT64_BYTES * n_cells),
               te.Transfer(2.0 * FLOAT64_BYTES * n_cells), te.PythonCalls(100))
        evaluations, kind, limit = optimization_evaluations(context)
        return te.TimeEstimate((te.TimeItem('cascade adjoint gradients', 'gradient', evaluations, kind, ops,
                                            limit=limit),))
    if any(feature.enabled and feature.type == 'adjoint_objective' for feature in context.physics_spec.features):
        return te.TimeEstimate(not_estimated=('adjoint sensitivity (computed when eps_sensitivity is read)',))
    return te.TimeEstimate()


def _line_mode_ops(line_cells, n_modes, complex_line, periodic):
    """Host ops of one section line's solve_slab_modes (PML modes kept) and ModeSet: real lines a tridiagonal
    (periodic: dense) eigh, complex ones dense eigvals + inverse iteration, long complex lines tridiagonal ARPACK."""
    m, n = float(line_cells), float(n_modes)
    common = (te.HostStream(COMPLEX128_BYTES * (20.0 * m + 10.0 * m * n)), te.PythonCalls(30 + 5 * n))
    if not complex_line:
        return common + (te.DenseFlops(REAL_LINE_EIG_FLOPS * m ** 3 if periodic else TRIDIAGONAL_EIG_FLOPS * m * n,
                                       host=True),)
    wanted = 2 * int(n) + 10
    if m > DENSE_COMPLEX_MAX_CELLS and wanted < m - 1:
        basis = arnoldi_basis_size(wanted, int(m))
        applications = APPLICATIONS_PER_BASIS_GUESS * basis
        return common + ((applications, te.HostStream(8.0 * COMPLEX128_BYTES * m)),
                         (applications, te.DenseFlops(16.0 * m * basis, host=True)), te.PythonCalls(applications))
    return common + (te.DenseFlops(COMPLEX_LINE_EIG_FLOPS * m ** 3, host=True),
                     te.HostStream(30.0 * COMPLEX128_BYTES * m * n))


def _time_estimate(settings_values, mesh, context):
    """Shared section work (shared_time_estimate) on the section lines, one solve_slab_modes per section (complex with
    cpml line faces, loss or poles); per solve the host stack, the 3-point Gauss field reconstruction and its uploads;
    optimization: a cascade adjoint gradient per iteration."""
    line_cells, slices = _line_cells(mesh, settings_values)
    n_modes = min(settings_values['n_modes'] or line_cells, line_cells)
    n_cells = float(mesh.n_cells)
    line_axis = 'y' if settings_values['propagation_axis'] == 'x' else 'x'
    faces = (settings_values[f'bc_{line_axis}_lo'], settings_values[f'bc_{line_axis}_hi'])
    complex_line = 'cpml' in faces or lossy_materials(context)
    # host stack (download on the gpu), slice stack / comparisons / digests; u and the in-plane field at 3 Gauss points
    # of both directions, wall cells / intensities / loss, the field and q_abs uploads
    solve_ops = (te.Transfer(_MATERIAL_ROWS * FLOAT32_BYTES * n_cells),
                 te.HostStream(3.0 * _MATERIAL_ROWS * FLOAT64_BYTES * n_cells),
                 te.DenseFlops(8.0 * 12.0 * n_modes * n_cells, host=True),
                 te.HostStream(12.0 * FLOAT64_BYTES * n_cells),
                 te.Transfer(5.0 * FLOAT32_BYTES * n_cells), te.PythonCalls(100))
    mode_set_bytes = COMPLEX128_BYTES * (3 * n_modes * line_cells + line_cells + 2 * n_modes)
    shared, _ = shared_time_estimate(context, slices, n_modes, line_cells, solve_ops,
                                     _line_mode_ops(line_cells, n_modes, complex_line, 'periodic' in faces),
                                     mode_set_bytes)
    return te.combine((shared, adjoint_time_estimate(context, n_modes, n_cells)))


# ---- type ----
_DESCRIPTION = (
    'Bidirectional eigenmode expansion (EME) of a 2D in-plane device along propagation_axis (x or y), for devices long '
    'along that axis (tapers, MMIs, couplers, gratings) where em_fdfd_* would be large. Every cell layer normal to the '
    'axis is a slice; consecutive equal slices (section_tolerance) form a section, so tapers are staircases at the '
    'mesh resolution. Each section line is solved once for n_modes slab modes (the em_fdfd_ez / _hz line stencil '
    'along the other axis, transverse bc_* faces, complex PML stretch with the PML modes kept as radiation; the same '
    'material rows repeated reuse their modes). Modes are normalised to the unconjugated product <e, h> = sum u (g u) '
    'w s = 1 (orthogonal also for lossy lines); interfaces are mode-matched by these overlaps (tangential u and its '
    'transverse partner continuous: R = (I + P P^T)^-1 (I - P P^T), T = P^T (I + R)) and the S-matrices cascaded by '
    'Redheffer star products (stable with evanescent modes). Exact for a complete basis (n_modes = 0), otherwise '
    'converging in n_modes (section table reciprocity_defect). The faces along propagation_axis are semi-infinite '
    'ports at the window planes: their bc is not read. Outputs of the excitation (eme_ports): the reconstructed '
    'phasor at the cell centres, sum over modes of a_n e^{-j k_n s} + b_n e^{-j k_n (L - s)} per section '
    '(<pol>_re / _im / _abs / _phase of the out-of-plane field; the other polarisation reads 0), e_abs (primary), '
    'e_intensity and h_abs (the in-plane component along the axis from the line differences, face mean), '
    'absorbed_power = 1/2 omega eps0 (-Im eps) times the cell mean of |E|^2 along the axis (3-point Gauss, so its '
    'integral is the exact absorbed power of the continuum propagation: 1 - T - R of a lossy guide between lossless '
    'ports), and the materials. Scalar outputs transmission_<k>, transmission, reflection. em_heat_joule reads '
    'absorbed_power (the time average of the last solve); heat_em_thermo_optic / Pockels write eps_r, eps_xx, eps_yy. '
    'eme_bend makes a range a circular bend (exact cylindrical line operator), eme_periodic repeats a drawn period '
    'n times (star powers) and lists its Bloch modes. Optimization study (ez or hz): design_region features '
    'write eps_r (ez) or eps_xx and eps_yy (hz) and adjoint_objective features weight port power fractions; dJ/d eps '
    '(output field eps_sensitivity) comes from the cascade adjoint without extra mode solves (exact for n_modes 0). '
    'Exact '
    'identities: a complete basis with the discrete '
    'propagation of the self-tests solves the em_fdfd_ez / _hz lattice of the sectioned guide (both polarisations). '
    'Studies: stationary (a solve whenever materials changed), frequency_domain (one solve per wavelength; the '
    'line solves are direct) and optimization. Bloch k is not supported; cannot share the model with another EM type. The instance '
    "is usually named 'eme'.")

EM_EME = PhysicsType(
    name=TYPE_NAME,
    label='Electromagnetic Waves, eigenmode expansion (2D)',
    family=EM_FAMILY,
    group='em',
    spaces_2d=('planar',),
    default_instance_name='eme',
    description=_DESCRIPTION,
    summary='Light through a device that is long in one direction, such as a taper or a coupler, computed by '
            'cutting it into slices with their own guided modes: transmission and reflection of each mode.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS['modes'],
    state_fields=tuple(EME_FIELD_INFOS[name] for name in ('ez_re', 'ez_im', 'hz_re', 'hz_im', 'e_intensity',
                                                           *EME_PERMITTIVITY_ARRAYS, 'sigma_e')),
    output_fields=tuple(EME_FIELD_INFOS.values()),
    source_slots=source_slot_docs(EME_PERMITTIVITY_SLOTS),
    supported_studies=(_STATIONARY_STUDY, FREQUENCY_DOMAIN_STUDY, OPTIMIZATION_STUDY),
    default_study={'type': _STATIONARY_STUDY},
    uses_dispersion=True,
    create=_create,
    domain_requirements=quasi_static_domain_requirements,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate_physics,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=scalar_outputs,
    scalar_outputs_help=SCALAR_OUTPUTS_HELP,
    backends=('gpu', 'cpu'),   # host line solves / cascade, fields on the Domain's xp
)


def _selftest_cases():
    """The em_eme / em_eme_3d SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import EME_SELFTEST_CASES
    from .selftests3d import EME_3D_SELFTEST_CASES
    from .selftests_adjoint import EME_ADJOINT_SELFTEST_CASES
    from .selftests_backend import EME_BACKEND_SELFTEST_CASES
    return (*EME_SELFTEST_CASES, *EME_ADJOINT_SELFTEST_CASES, *EME_3D_SELFTEST_CASES, *EME_BACKEND_SELFTEST_CASES)


def register(registry):
    """Register em_eme and em_eme_3d (after the physics/em types: their material Params stay first per name), their
    templates, then the selftest loader."""
    from .definition3d import EM_EME_3D   # imports this module's shared feature pieces
    from .selftests3d import EME_3D_TEMPLATES
    registry.register_physics(EM_EME)
    registry.register_physics(EM_EME_3D)
    for template in (*EME_TEMPLATES, *EME_3D_TEMPLATES):
        registry.register_template(*template)
    registry.register_selftests(_selftest_cases)

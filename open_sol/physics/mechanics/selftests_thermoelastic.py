"""Model-run self-tests of the thermoelastic_damping feature (solid_mechanics on tensor and triangle meshes,
solid_mechanics_3d on tetrahedra): Zener / Lifshitz-Roukes damping of thin-beam flexure and of a clamped circular plate
(axisymmetric, plate strength Delta (1 + nu) / (1 - nu)), the frequency shift and the combined Q with a loss factor;
the template 'silicon_beam_thermoelastic_damping'; no physics imports."""
import cmath
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import SubChecks, cpu_defaults
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                           PhysicsSpec, StudySpec)

_PHYSICS = 'solid'
_TED = 'ted'
_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
_COLUMNS = ('resonance_number', 'frequency', 'q_ted', 'ted_frequency_shift', 'q_other', 'q_total')   # TABLE_COLUMNS
# silicon-like isotropic solid; the conductivity of each run puts its mode at the chosen omega tau (Euler-Bernoulli /
# Kirchhoff estimate of omega, the checks use the computed one)
_THICKNESS = 1e-6           # m
_YOUNGS_MODULUS = 170e9     # Pa
_POISSON_RATIO = 0.28
_DENSITY = 2330.0           # kg/m^3
_HEAT_CAPACITY = 700.0      # J/(kg K)
_EXPANSION = 2.6e-6         # 1/K
_T0 = 293.15                # K
_BEAM_SLENDERNESS = 40.0    # L / h of the simply supported beam (half modelled: mirror roller at x_hi)
_BEAM_SLENDERNESS_3D = 20.0
_PLATE_SLENDERNESS = 20.0   # R / h of the clamped plate
_CLAMPED_PLATE_ROOT = 10.2158   # (lambda a)^2 of the fundamental axisymmetric mode of a clamped circular plate
_OMEGA_TAU = {'low': 0.3, 'peak': 1.0, 'high': 3.0}
_LOSS_FACTOR = 1e-4         # mechanical_loss_factor of the 'peak' / triangle beam: q_other = 1 / eta
_TOLERANCE = 1e-2           # 1 / Q_TED versus Lifshitz-Roukes (thin-beam / plate theory: shear and clamp ~ (h / L)^2)
_SHIFT_TOLERANCE = 2e-2     # relative frequency shift versus Lifshitz-Roukes
_TOLERANCE_3D = 1.5e-2      # L / h = 20: Timoshenko shear energy ~0.5 % outside thin-beam theory
_LOSS_TOLERANCE = 1e-6      # q_other = 1 / eta: exact perturbatively (tensor), to O(eta^2) for the damped pencil


def _relaxation_strength(plate=False):
    """Zener Delta = E alpha^2 T0 / (rho cp) (plate flexure: x (1 + nu) / (1 - nu))."""
    strength = _YOUNGS_MODULUS * _EXPANSION ** 2 * _T0 / (_DENSITY * _HEAT_CAPACITY)
    return strength * (1.0 + _POISSON_RATIO) / (1.0 - _POISSON_RATIO) if plate else strength


def _lifshitz_roukes(omega, diffusivity, plate=False):
    """(1 / Q, relative frequency shift) of Lifshitz-Roukes: f = 24 / (k h)^3 (k h / 2 - tan(k h / 2)), k = (1 - j)
    sqrt(omega / (2 D)); 1 / Q = Delta Im f, d omega / omega = Delta Re(1 + f) / 2."""
    k_h = (1.0 - 1j) * math.sqrt(omega / (2.0 * diffusivity)) * _THICKNESS
    f = 24.0 / k_h ** 3 * (k_h / 2.0 - cmath.tan(k_h / 2.0))
    strength = _relaxation_strength(plate)
    return strength * f.imag, 0.5 * strength * (1.0 + f).real


def _zener(omega, diffusivity, plate=False):
    """Zener Delta omega tau / (1 + (omega tau)^2), tau = h^2 / (pi^2 D)."""
    product = omega * _THICKNESS ** 2 / (math.pi ** 2 * diffusivity)
    return _relaxation_strength(plate) * product / (1.0 + product * product)


def _beam_omega(slenderness):
    """Euler-Bernoulli fundamental [rad/s] of the simply supported beam of length slenderness x h."""
    length = slenderness * _THICKNESS
    return (math.pi / length) ** 2 * math.sqrt(_YOUNGS_MODULUS * _THICKNESS ** 2 / (12.0 * _DENSITY))


def _plate_omega():
    """Kirchhoff fundamental [rad/s] of the clamped circular plate of radius _PLATE_SLENDERNESS x h."""
    radius = _PLATE_SLENDERNESS * _THICKNESS
    rigidity = _YOUNGS_MODULUS * _THICKNESS ** 3 / (12.0 * (1.0 - _POISSON_RATIO ** 2))
    return _CLAMPED_PLATE_ROOT / radius ** 2 * math.sqrt(rigidity / (_DENSITY * _THICKNESS))


def _diffusivity(omega, omega_tau):
    """D = k / (rho cp) [m^2/s] that puts omega at omega tau (tau = h^2 / (pi^2 D))."""
    return omega * _THICKNESS ** 2 / (math.pi ** 2 * omega_tau)


def _material(diffusivity, loss_factor=None):
    properties = {'youngs_modulus': _YOUNGS_MODULUS, 'poisson_ratio': _POISSON_RATIO, 'rho': _DENSITY,
                  'cp': _HEAT_CAPACITY, 'alpha': _EXPANSION, 'k': diffusivity * _DENSITY * _HEAT_CAPACITY}
    if loss_factor is not None:
        properties['mechanical_loss_factor'] = loss_factor
    return [MaterialSpec('resonator', properties=properties, selection=['all'])]


def _mesh(kind, cells_per_thickness):
    size = _THICKNESS / cells_per_thickness
    if kind == 'tensor':
        return MeshSpec(mode='uniform', dx=size)
    return MeshSpec(kind=kind, mode='uniform', dx=size, **({'mesher': 'builtin'} if kind == 'tetrahedra' else {}))


def _ted_feature():
    return FeatureSpec(_TED, 'thermoelastic_damping', params={'T0': _T0})


def _beam_spec(kind, cells_per_thickness, omega_tau, loss_factor=None):
    """Half of a simply supported plane-stress beam (L = _BEAM_SLENDERNESS h along x, thickness h along y): u_y = 0
    on x_lo (the support), roller on x_hi (the mid-span mirror), one mode."""
    diffusivity = _diffusivity(_beam_omega(_BEAM_SLENDERNESS), omega_tau)
    return ModelSpec(
        title='Self-test: thermoelastic damping of a simply supported beam (solid mechanics)',
        domain=DomainSpec(size_x=0.5 * _BEAM_SLENDERNESS * _THICKNESS, size_y=_THICKNESS),
        materials=_material(diffusivity, loss_factor),
        physics=[PhysicsSpec(_PHYSICS, 'solid_mechanics', settings={'out_of_plane': 'plane_stress'},
                             features=[FeatureSpec('support', 'displacement', ['x_lo'], params={'u_y': 0.0}),
                                       FeatureSpec('mid_span', 'roller', ['x_hi']), _ted_feature()])],
        mesh=_mesh(kind, cells_per_thickness),
        study=StudySpec(type=_STUDY, settings={'n_resonances': 1}),
        solver_defaults=cpu_defaults())   # the TED evaluation is host SuperLU either way


def _plate_spec(kind, cells_per_thickness):
    """Clamped circular plate (radius _PLATE_SLENDERNESS h, axisymmetric (r, z), clamped on x_hi) at omega tau 1."""
    return ModelSpec(
        title='Self-test: thermoelastic damping of a clamped circular plate (solid mechanics, axisymmetric)',
        domain=DomainSpec(size_x=_PLATE_SLENDERNESS * _THICKNESS, size_y=_THICKNESS),
        materials=_material(_diffusivity(_plate_omega(), 1.0)),
        physics=[PhysicsSpec(_PHYSICS, 'solid_mechanics', settings={'geometry': 'axisymmetric'},
                             features=[FeatureSpec('rim', 'fixed', ['x_hi']), _ted_feature()])],
        mesh=_mesh(kind, cells_per_thickness),
        study=StudySpec(type=_STUDY, settings={'n_resonances': 1}),
        solver_defaults=cpu_defaults())   # the TED evaluation is host SuperLU either way


def _beam_3d_spec():
    """Quarter of a simply supported beam on P2 tetrahedra (L = _BEAM_SLENDERNESS_3D h along x, width 2 h along y with
    a mirror roller on y_lo, thickness h along z): u_z = 0 on x_lo, roller on x_hi (mid-span)."""
    diffusivity = _diffusivity(_beam_omega(_BEAM_SLENDERNESS_3D), 1.0)
    return ModelSpec(
        title='Self-test: thermoelastic damping of a simply supported beam (solid mechanics 3D, tetrahedra)',
        domain=DomainSpec(size_x=0.5 * _BEAM_SLENDERNESS_3D * _THICKNESS, size_y=_THICKNESS, size_z=_THICKNESS),
        materials=_material(diffusivity),
        physics=[PhysicsSpec(_PHYSICS, 'solid_mechanics_3d', settings={'discretization': 'quadratic'},
                             features=[FeatureSpec('support', 'displacement', ['x_lo'], params={'u_z': 0.0}),
                                       FeatureSpec('mid_span', 'roller', ['x_hi']),
                                       FeatureSpec('width_mirror', 'roller', ['y_lo']), _ted_feature()])],
        meshes=[MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=0.5 * _THICKNESS,
                         mesher='builtin')],
        study=StudySpec(type=_STUDY, settings={'n_resonances': 1}),
        solver_defaults=cpu_defaults())   # the TED evaluation is host SuperLU either way


# ---- template 'silicon_beam_thermoelastic_damping' (registered by definition.py) ----
SILICON_BEAM_TED_LABEL = 'Silicon beam thermoelastic damping'
SILICON_BEAM_TED_DESCRIPTION = (
    'Solid mechanics (plane stress), eigenfrequency: half of a simply supported library-silicon beam (h = 2 um, L = 80 '
    'um, mirror roller at mid-span), its symmetric flexural modes with a resonance table and the thermoelastic_damping '
    'table (Q_TED per mode, T0 293.15 K); derived f_1 (1.19 MHz) and Q_ted_1 (1.3e5: far below the Debye peak, omega '
    'tau ~0.04; the 3rd and 5th flexural modes at 10.7 / 29 MHz reach 1.7e4 / 1.0e4, the peak Delta / 2 ~ 1e-4 sits '
    'near omega tau = 1; a lower k_beam or a thicker beam moves the fundamental towards it).')


def silicon_beam_ted_spec():
    """Template 'silicon_beam_thermoelastic_damping': Q_TED of a simply supported silicon beam (plane stress)."""
    return ModelSpec(
        title=SILICON_BEAM_TED_LABEL,
        parameters=[ParameterSpec('h_beam', '2*um', 'beam thickness along y'),
                    ParameterSpec('L_beam', '80*um', 'beam length (the half beam L / 2 is drawn)'),
                    ParameterSpec('k_beam', 130.0, 'thermal conductivity [W/(m K)] (library silicon 130)')],
        domain=DomainSpec(size_x='L_beam/2', size_y='h_beam'),
        materials=[MaterialSpec('beam', library='silicon', properties={'k': 'k_beam'}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, 'solid_mechanics', settings={'out_of_plane': 'plane_stress'},
                             features=[FeatureSpec('support', 'displacement', ['x_lo'], params={'u_y': 0.0}),
                                       FeatureSpec('mid_span', 'roller', ['x_hi']),
                                       FeatureSpec('modes', 'resonance_table'), _ted_feature()])],
        mesh=MeshSpec(mode='uniform', dx='h_beam/16'),
        study=StudySpec(type=_STUDY, settings={'n_resonances': 3}),
        derived_values=[DerivedValueSpec('f_1', 'frequency', operation='at', source=f'{_PHYSICS}.{_TED}', at=0),
                        DerivedValueSpec('Q_ted_1', 'q_ted', operation='at', source=f'{_PHYSICS}.{_TED}', at=0)])


def _first_row(results):
    """{column: value} of the first thermoelastic_damping row."""
    table = results.result(_PHYSICS, _TED)
    if not table['rows']:
        raise ValueError('the thermoelastic_damping table has no rows')
    return dict(zip(table['columns'], table['rows'][0]))


def _add_damping(checks, run, results, diffusivity, tolerance, plate=False, shift=True):
    """'<run> 1/Q' (and '<run> shift') against Lifshitz-Roukes at the computed omega; Zener and omega tau recorded."""
    row = _first_row(results)
    omega = 2.0 * math.pi * float(row['frequency'])
    inverse_q, relative_shift = _lifshitz_roukes(omega, diffusivity, plate)
    measured = 1.0 / float(row['q_ted'])
    zener = _zener(omega, diffusivity, plate)
    checks.add_relative(f'{run} 1/Q', measured, inverse_q, tolerance, zener=zener,
                        versus_zener=measured / zener - 1.0,
                        omega_tau=omega * _THICKNESS ** 2 / (math.pi ** 2 * diffusivity))
    if shift:
        checks.add_relative(f'{run} shift', float(row['ted_frequency_shift']), relative_shift, _SHIFT_TOLERANCE)
    return row


def _add_combined(checks, run, row):
    """'<run> q_other' = 1 / eta and '<run> q_total' = 1 / (1 / q_ted + eta)."""
    checks.add_relative(f'{run} q_other', float(row['q_other']), 1.0 / _LOSS_FACTOR, _LOSS_TOLERANCE)
    checks.add_relative(f'{run} q_total', float(row['q_total']),
                        1.0 / (1.0 / float(row['q_ted']) + 1.0 / float(row['q_other'])), 1e-12)


def _tensor_specs():
    specs = {f'beam_{name}': _beam_spec('tensor', 16, omega_tau, _LOSS_FACTOR if name == 'peak' else None)
             for name, omega_tau in _OMEGA_TAU.items()}
    specs['plate'] = _plate_spec('tensor', 16)
    return specs


def _check_tensor(results_by_name):
    checks = SubChecks()
    for name, omega_tau in _OMEGA_TAU.items():
        row = _add_damping(checks, f'beam_{name}', results_by_name[f'beam_{name}'],
                           _diffusivity(_beam_omega(_BEAM_SLENDERNESS), omega_tau), _TOLERANCE)
        if name == 'peak':
            _add_combined(checks, 'beam_peak', row)
        else:
            checks.add('beam_' + name + ' q_other', math.isinf(float(row['q_other'])), float(row['q_other']))
    _add_damping(checks, 'plate', results_by_name['plate'], _diffusivity(_plate_omega(), 1.0), _TOLERANCE,
                 plate=True, shift=False)
    return checks.result()


def _triangle_specs():
    return {'beam': _beam_spec('triangles', 4, 1.0, _LOSS_FACTOR), 'plate': _plate_spec('triangles', 4)}


def _check_triangles(results_by_name):
    checks = SubChecks()
    row = _add_damping(checks, 'beam', results_by_name['beam'],
                       _diffusivity(_beam_omega(_BEAM_SLENDERNESS), 1.0), _TOLERANCE)
    _add_combined(checks, 'beam', row)
    _add_damping(checks, 'plate', results_by_name['plate'], _diffusivity(_plate_omega(), 1.0), _TOLERANCE,
                 plate=True, shift=False)
    return checks.result()


def _check_3d(results_by_name):
    checks = SubChecks()
    _add_damping(checks, 'beam', results_by_name['beam'], _diffusivity(_beam_omega(_BEAM_SLENDERNESS_3D), 1.0),
                 _TOLERANCE_3D)
    return checks.result()


_THEORY = ('Lifshitz-Roukes 1 / Q = Delta Im f, f = 24 / (k h)^3 (k h / 2 - tan(k h / 2)), k = (1 - j) sqrt(omega / '
           '(2 D)), frequency shift Delta Re(1 + f) / 2, Delta = E alpha^2 T0 / (rho cp) (Zener Delta omega tau / (1 + '
           '(omega tau)^2), tau = h^2 / (pi^2 D), within 2 %, recorded per run), at the computed omega')
_MATERIAL_TEXT = (f'E {_YOUNGS_MODULUS:g} Pa, nu {_POISSON_RATIO:g}, rho {_DENSITY:g}, cp {_HEAT_CAPACITY:g}, alpha '
                  f'{_EXPANSION:g}, T0 {_T0:g} K, h {_THICKNESS * 1e6:g} um; k sets omega tau')
_TED_2D_CASES = (
    SelfTestCase(
        name='solid_thermoelastic_damping',
        label='Solid mechanics: thermoelastic damping of a beam and a plate (Zener)',
        description=('thermoelastic_damping of solid_mechanics on uniform tensor meshes (' + _MATERIAL_TEXT + '). '
                     f'Runs beam_low / _peak / _high: half of a simply supported plane-stress beam, L / h = '
                     f'{_BEAM_SLENDERNESS:g}, 16 square cells over h, conductivity for omega tau = '
                     f"{', '.join(f'{value:g}' for value in _OMEGA_TAU.values())} (Euler-Bernoulli omega); the first "
                     f'mode must match {_THEORY} within {_TOLERANCE:g} (relative; 1 / Q within 0.4 %: at 8 cells '
                     'over h the bilinear temperature misses 1 % off the peak) and the shift within '
                     f'{_SHIFT_TOLERANCE:g}. beam_peak sets a '
                     f'mechanical_loss_factor {_LOSS_FACTOR:g}: q_other = 1 / eta (perturbative on the undamped '
                     f'tensor pencil, {_LOSS_TOLERANCE:g}) and q_total = 1 / (1 / q_ted + 1 / q_other); the others '
                     f'q_other inf. Run plate: a clamped circular plate, R / h = {_PLATE_SLENDERNESS:g}, '
                     'axisymmetric, 16 cells over h, omega tau 1 (Kirchhoff omega): 1 / Q against the same '
                     'Lifshitz-Roukes form with the plate strength Delta (1 + nu) / (1 - nu) (biaxial flexure: '
                     f'hoop row), within {_TOLERANCE:g} (bilinear cells: +5 % at 4, +1.4 % at 8, +0.45 % at 16 cells '
                     'over h). measured = error = failed sub-checks.'),
        specs=_tensor_specs, check=_check_tensor),
    SelfTestCase(
        name='solid_fem_thermoelastic_damping',
        label='Solid mechanics, triangle mesh: thermoelastic damping of a beam and a plate (Zener)',
        description=('thermoelastic_damping of solid_mechanics on uniform P2 triangle meshes, 4 elements over h ('
                     + _MATERIAL_TEXT + '): run beam = half of a simply supported plane-stress beam (L / h = '
                     f'{_BEAM_SLENDERNESS:g}, omega tau 1) with a mechanical_loss_factor {_LOSS_FACTOR:g}, so the '
                     'pencil is damped and the mode complex (Re omega, complex strain source): 1 / Q against '
                     f'{_THEORY} within {_TOLERANCE:g}, shift within {_SHIFT_TOLERANCE:g}, q_other = the damped '
                     f'pencil Q = 1 / eta within {_LOSS_TOLERANCE:g}, q_total = 1 / (1 / q_ted + 1 / q_other). Run '
                     f'plate = the clamped circular plate (R / h = {_PLATE_SLENDERNESS:g}, axisymmetric, omega tau 1) '
                     f'against the plate strength Delta (1 + nu) / (1 - nu), within {_TOLERANCE:g} (~0.3 % off). '
                     'measured = error = failed sub-checks.'),
        specs=_triangle_specs, check=_check_triangles),
)


def _specs_3d():
    return {'beam': _beam_3d_spec()}


THERMOELASTIC_3D_CASES = (
    SelfTestCase(
        name='solid_3d_thermoelastic_damping',
        label='Solid mechanics 3D, tetrahedra: thermoelastic damping of a beam (Zener)',
        description=('thermoelastic_damping of solid_mechanics_3d on builtin P2 tetrahedra (h / 2): a quarter of a '
                     f'simply supported beam (L / h = {_BEAM_SLENDERNESS_3D:g}, width 2 h with a mirror roller, '
                     'mid-span roller, u_z = 0 at the support; ' + _MATERIAL_TEXT + ', omega tau 1): 1 / Q against '
                     f'{_THEORY} within {_TOLERANCE_3D:g} (~0.5 % below: the Timoshenko shear energy at L / h = 20), '
                     f'shift within {_SHIFT_TOLERANCE:g}. measured = error = failed sub-checks.'),
        specs=_specs_3d, check=_check_3d),
)


# ---- optomechanical_coupling quality_from: Q_m of the Brillouin gain per mode from the TED table ----
_FIXED_QUALITY = 1000.0
_BRILLOUIN_PHYSICS = 'solid'   # = selftests_brillouin's solid name


def _quality_from_specs():
    """The uniform-guide optomechanics model of om_backward_brillouin_chain with forward scattering (plane-strain
    modes), a thermal material, a thermoelastic_damping feature and two couplings: 'fixed' (mechanical_quality_factor)
    and 'damped' (quality_from the TED feature)."""
    from .selftests_brillouin import _chain_specs
    spec = _chain_specs()['backward']
    spec.solver_defaults = cpu_defaults()
    solid = next(physics for physics in spec.physics if physics.name == _BRILLOUIN_PHYSICS)
    solid.settings = {'target_frequency': 0.0}
    coupling = next(feature for feature in solid.features if feature.type == 'optomechanical_coupling')
    fixed = dict(coupling.params, scattering='forward', mechanical_quality_factor=_FIXED_QUALITY)
    damped = {name: value for name, value in fixed.items() if name != 'mechanical_quality_factor'}
    solid.features = [feature for feature in solid.features if feature is not coupling] + [
        FeatureSpec('fixed', 'optomechanical_coupling', params=fixed),
        FeatureSpec('damped', 'optomechanical_coupling', params={**damped, 'quality_from': _TED}), _ted_feature()]
    spec.materials[0].properties.update({'cp': _HEAT_CAPACITY, 'alpha': _EXPANSION, 'k': 1.0})
    return {'forward': spec}


def _check_quality_from(results_by_name):
    results = results_by_name['forward']
    fixed, damped = (results.result(_BRILLOUIN_PHYSICS, name) for name in ('fixed', 'damped'))
    column = fixed['columns'].index('brillouin_gain')
    ted = results.result(_BRILLOUIN_PHYSICS, _TED)
    total = ted['columns'].index('q_total')
    checks = SubChecks()
    checks.add('rows', len(fixed['rows']) == len(damped['rows']) == len(ted['rows']) > 0, len(ted['rows']))
    for number, (row_fixed, row_damped, row_ted) in enumerate(zip(fixed['rows'], damped['rows'], ted['rows']), 1):
        expected = float(row_fixed[column]) * float(row_ted[total]) / _FIXED_QUALITY
        checks.add_relative(f'mode {number} gain', float(row_damped[column]), expected, 1e-9)
    return checks.result()


THERMOELASTIC_CASES = _TED_2D_CASES + (
    SelfTestCase(
        name='om_thermoelastic_quality_factor',
        label='Optomechanics: Brillouin gain with the thermoelastic Q of each mode',
        description=('optomechanics study of the uniform guide of om_backward_brillouin_chain with forward scattering '
                     '(plane-strain modes; em_modes on the cpu) and a thermal material, two optomechanical_coupling '
                     f"features: 'fixed' with mechanical_quality_factor {_FIXED_QUALITY:g}, 'damped' with quality_from "
                     "the thermoelastic_damping feature: per mode the 'damped' brillouin_gain must be the 'fixed' one x "
                     'q_total / 1000 of the TED table (relative 1e-9; G_B is linear in Q_m). Pins the per-mode Q_m '
                     'plumbing only (the uniform guide couples weakly in forward). measured = error = failed '
                     'sub-checks.'),
        specs=_quality_from_specs, check=_check_quality_from,
        heavy=True),   # ~15 s on the cpu: the em_modes solve of the periodic guide (the TED part is milliseconds)
)

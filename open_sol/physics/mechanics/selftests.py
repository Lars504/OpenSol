"""Analytic self-tests of solid_mechanics (bilayer under tension on a graded mesh; simple shear), the strain-optic plate
of the CPU-vs-GPU parity case and the 'heated_waveguide_stress' template."""
import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, cell_edges, max_relative_error, with_exact_cg, worst_error
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec, StudySpec)

SOLID_TYPE_NAME = 'solid_mechanics'                  # = definition.SOLID_TYPE_NAME (definition imports this module)
THERMAL_STRAIN_COUPLING = 'heat_solid_thermal_strain'
STRAIN_OPTIC_COUPLING = 'solid_em_strain_optic'      # = coupling.definitions._STRAIN_OPTIC_TYPE_NAME
OPTICAL_FORCE_COUPLING = 'em_solid_optical_force'
_SOLID = 'solid'                                     # physics instance names
_HEAT = 'heat'
_EM = 'em'
_RUN = 'run'
_TOLERANCE = 1e-6                                    # the exact-CG error bound ~1e5 x 1e-12 is far below it
_CELL = 1e-6                                         # m, uniform mesh of every case
_GRADED_MIN_RATIO = 2.0                              # widest / narrowest window x cell a graded run must reach


def _lame(youngs_modulus, poisson_ratio):
    """(lambda, mu) [Pa] of an isotropic solid."""
    return (youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
            youngs_modulus / (2.0 * (1.0 + poisson_ratio)))


def _rectangle(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _elastic(youngs_modulus, poisson_ratio, **extra):
    return {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, **extra}


# ---- solid_bilayer_tension_graded: two layers stacked in y, pulled along x, on a graded mesh ----
# 100 x 20 um, base 1 um cells refined to 0.25 um around x = L / 2 (uniform 1 um cells away from it); layer a (E 100
# GPa, nu 0.3) on y < 8 um, layer b (E 50 GPa, nu 0.2) above; roller on x_lo (u_x = 0) and y_lo (u_y = 0), displacement
# u_x = d on x_hi (u_y free), y_hi free. Exact: eps_xx = d / L everywhere, sigma_yy = sigma_xy = 0, eps_yy = -nu / (1 -
# nu) eps_xx per layer (plane strain, sigma_yy = 0), so u_y is piecewise linear with its kink on the interface cell
# edge: the exact field lies in the Q1 space on any rectilinear mesh and the discrete solution equals it up to the CG
# residual. sigma_xx = E / (1 - nu^2) eps_xx, sigma_zz = nu sigma_xx, von Mises = sigma_xx sqrt(1 - nu + nu^2), the
# pull reaction F = sum of sigma_xx h per layer (N/m), the x_lo roller -F.
_BILAYER_SIZE = (100e-6, 20e-6)
_BILAYER_INTERFACE_Y = 8e-6
_BILAYER_LAYERS = ((100e9, 0.3), (50e9, 0.2))        # (E [Pa], nu) of layer a (below) and b (above)
_BILAYER_PULL = 1e-7                                 # m, strain 1e-3
_BILAYER_GRADED_CELL = 0.25e-6                       # m, refinement line x = L / 2
_BILAYER_GRADED_REACH = 30e-6                        # m


def _bilayer_closed_forms(strain_xx=_BILAYER_PULL / _BILAYER_SIZE[0]):
    """(eps_xx, eps_yy per layer, sigma_xx per layer [Pa], pull reaction F [N/m]) at the imposed strain eps_xx."""
    height = _BILAYER_SIZE[1]
    strain_yy = tuple(-nu / (1.0 - nu) * strain_xx for _, nu in _BILAYER_LAYERS)
    stress_xx = tuple(youngs / (1.0 - nu * nu) * strain_xx for youngs, nu in _BILAYER_LAYERS)
    force = stress_xx[0] * _BILAYER_INTERFACE_Y + stress_xx[1] * (height - _BILAYER_INTERFACE_Y)
    return strain_xx, strain_yy, stress_xx, force


def _bilayer_parts():
    """([rectangles 'lower', 'upper'], [MaterialSpecs layer_a, layer_b on them]) of the bilayer strip."""
    length, height = _BILAYER_SIZE
    (youngs_a, nu_a), (youngs_b, nu_b) = _BILAYER_LAYERS
    return ([_rectangle('lower', 0.0, 0.0, length, _BILAYER_INTERFACE_Y),
             _rectangle('upper', 0.0, _BILAYER_INTERFACE_Y, length, height)],
            [MaterialSpec('layer_a', properties=_elastic(youngs_a, nu_a), selection=['lower']),
             MaterialSpec('layer_b', properties=_elastic(youngs_b, nu_b), selection=['upper'])])


def _bilayer_specs():
    """'run': roller left / bottom, displacement 'pull' on x_hi, refinement line at x = L / 2."""
    length, height = _BILAYER_SIZE
    geometry, materials = _bilayer_parts()
    geometry.append(GeometrySpec('refine_line', 'line', {'x0': 0.5 * length, 'y0': 0.0, 'x1': 0.5 * length,
                                                         'y1': height}))
    mesh = MeshSpec(mode='nonuniform', dx=_CELL,
                    refinements=[RefinementSpec('refine_line', _BILAYER_GRADED_CELL, _BILAYER_GRADED_REACH)])
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: bilayer under tension (solid mechanics), graded mesh',
        domain=DomainSpec(size_x=length, size_y=height),
        geometry=geometry,
        materials=materials,
        physics=[PhysicsSpec(_SOLID, SOLID_TYPE_NAME, features=[
            FeatureSpec('left', 'roller', ['x_lo']),
            FeatureSpec('bottom', 'roller', ['y_lo']),
            FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _BILAYER_PULL})])],
        mesh=mesh,
        study=StudySpec(type='stationary')))}


# ---- the strain-optic plate (solid_em_strain_optic_plate): 40 x 20 um of 1 um cells, E 100 GPa, nu 0.3, heat holds
# all four faces at T0 + 100 K (uniform rise), rollers on x_lo and y_lo, x_hi / y_hi free ----
_PLATE_SIZE = (40e-6, 20e-6)
_PLATE_ELASTIC = (100e9, 0.3)
_PLATE_T0 = 293.15                                   # K, heat T0 = the stress-free temperature
_PLATE_RISE = 100.0                                  # K


# ---- solid_simple_shear: simple shear of a block along x (u_x = d y / H) and along y (u_y = d x / W) ----
# 20 x 10 um of 1 um cells, E 100 GPa, nu 0.3. Shear axis s, cross axis t: the base t_lo face fixed, the moved t_hi face
# displaced (u_s = d, u_t = 0), the two sliding s faces hold u_t = 0 only (u_s free, traction-free along s). Exact:
# u_s = gamma t, gamma = d / extent_t, u_t = 0, every strain 0 but strain_xy = gamma / 2, stress_xy = mu gamma (only the
# c66 term), other stresses 0; linear u, exact in Q1. Reactions (traction integrals): moved F_s = mu gamma extent_s,
# base -F_s, sliding hi face F_t = mu gamma extent_t, lo face -that, moved / base F_t = 0 (the corner shares cancel).
_SHEAR_SIZE = (20e-6, 10e-6)
_SHEAR_ELASTIC = (100e9, 0.3)
_SHEAR_DISPLACEMENT = 1e-8                           # m on the moved face (gamma 1e-3 along x, 5e-4 along y)
_SHEAR_RUNS = {'shear_x': 'x', 'shear_y': 'y'}      # run name -> shear axis s


def _shear_geometry(axis):
    """(cross axis t, moved face, base face, sliding faces (lo, hi), extent along s [m], extent along t [m])."""
    size_x, size_y = _SHEAR_SIZE
    if axis == 'x':
        return 'y', 'y_hi', 'y_lo', ('x_lo', 'x_hi'), size_x, size_y
    return 'x', 'x_hi', 'x_lo', ('y_lo', 'y_hi'), size_y, size_x


def _shear_specs():
    """'shear_x' / 'shear_y': base fixed, 'moved' displacement (u_s = d, u_t = 0), 'side_lo' / 'side_hi' u_t = 0."""
    size_x, size_y = _SHEAR_SIZE
    specs = {}
    for run, axis in _SHEAR_RUNS.items():
        across, moved_face, base_face, (side_lo, side_hi), _, _ = _shear_geometry(axis)
        specs[run] = with_exact_cg(ModelSpec(
            title=f'Self-test: simple shear along {axis} (solid mechanics)',
            domain=DomainSpec(size_x=size_x, size_y=size_y),
            materials=[MaterialSpec('block', properties=_elastic(*_SHEAR_ELASTIC), selection=['all'])],
            physics=[PhysicsSpec(_SOLID, SOLID_TYPE_NAME, features=[
                FeatureSpec('base', 'fixed', [base_face]),
                FeatureSpec('moved', 'displacement', [moved_face],
                            {f'u_{axis}': _SHEAR_DISPLACEMENT, f'u_{across}': 0.0}),
                FeatureSpec('side_lo', 'displacement', [side_lo], {f'u_{across}': 0.0}),
                FeatureSpec('side_hi', 'displacement', [side_hi], {f'u_{across}': 0.0})])],
            mesh=MeshSpec(mode='uniform', dx=_CELL),
            study=StudySpec(type='stationary')))
    return specs


# ---- checks (CPU, on Results-like objects) ----
def _window_fields(results, names, graded=False):
    """({name: float64 window values}, x edges, y edges) of solid fields; ValueError unless a graded run has x widths
    spanning at least _GRADED_MIN_RATIO."""
    values, x_edges, y_edges = {}, None, None
    for name in names:
        field_data = results.field(f'{_SOLID}.{name}')
        values[name] = np.asarray(field_data.values, dtype=np.float64)
        if x_edges is None:
            x_edges = cell_edges(np.asarray(field_data.x_centers, dtype=np.float64), float(field_data.extent[0]))
            y_edges = cell_edges(np.asarray(field_data.y_centers, dtype=np.float64), float(field_data.extent[2]))
    widths = np.diff(x_edges)
    if graded and not np.max(widths) >= _GRADED_MIN_RATIO * np.min(widths):
        raise ValueError(f'expected graded x cells (widest / narrowest >= {_GRADED_MIN_RATIO:g}), got {widths.size} '
                         f'cells of {np.min(widths):.4g} .. {np.max(widths):.4g} m')
    return values, x_edges, y_edges


def _reaction(results, feature):
    """(F_x, F_y) [N/m] of a support feature's table result."""
    table = results.result(_SOLID, feature)
    return float(table['F_x']), float(table['F_y'])


def _check_bilayer(results_by_name):
    """u_x, u_y, the strains, stresses and von Mises of both layers and the pull / roller reactions against the closed
    forms."""
    results = results_by_name[_RUN]
    names = ('u_x', 'u_y', 'strain_xx', 'strain_yy', 'stress_xx', 'stress_yy', 'stress_xy', 'stress_zz', 'von_mises')
    fields, x_edges, y_edges = _window_fields(results, names, graded=True)
    strain_xx, strain_yy, stress_xx, force = _bilayer_closed_forms()
    length, _ = _BILAYER_SIZE
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    lower = y_centres < _BILAYER_INTERFACE_Y
    (_, nu_a), (_, nu_b) = _BILAYER_LAYERS
    u_y_expected = np.where(lower, strain_yy[0] * y_centres,
                            strain_yy[0] * _BILAYER_INTERFACE_Y + strain_yy[1] * (y_centres - _BILAYER_INTERFACE_Y))
    stress_xx_expected = np.where(lower, stress_xx[0], stress_xx[1])
    stress_zz_expected = np.where(lower, nu_a * stress_xx[0], nu_b * stress_xx[1])
    von_mises_expected = np.where(lower, stress_xx[0] * np.sqrt(1.0 - nu_a + nu_a * nu_a),
                                  stress_xx[1] * np.sqrt(1.0 - nu_b + nu_b * nu_b))
    stress_scale = max(stress_xx)
    pull_x, _ = _reaction(results, 'pull')
    left_x, _ = _reaction(results, 'left')
    _, bottom_y = _reaction(results, 'bottom')
    errors = {'u_x': max_relative_error(fields['u_x'], strain_xx * x_centres[:, None], _BILAYER_PULL),
              'u_y': max_relative_error(fields['u_y'], u_y_expected[None, :], float(np.max(np.abs(u_y_expected)))),
              'strain_xx': max_relative_error(fields['strain_xx'], strain_xx, strain_xx),
              'strain_yy': max_relative_error(fields['strain_yy'], np.where(lower, *strain_yy)[None, :], strain_xx),
              'stress_xx': max_relative_error(fields['stress_xx'], stress_xx_expected[None, :], stress_scale),
              'stress_yy': max_relative_error(fields['stress_yy'], 0.0, stress_scale),
              'stress_xy': max_relative_error(fields['stress_xy'], 0.0, stress_scale),
              'stress_zz': max_relative_error(fields['stress_zz'], stress_zz_expected[None, :], stress_scale),
              'von_mises': max_relative_error(fields['von_mises'], von_mises_expected[None, :], stress_scale),
              'pull': abs(pull_x - force) / force,
              'left': abs(left_x + force) / force,
              'bottom': abs(bottom_y) / force}
    return {'measured': pull_x, 'expected': force, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N/m', 'F_pull': pull_x, 'F_left': left_x,
                        'F_bottom_y': bottom_y, 'strain_xx': strain_xx, 'strain_yy_layers': list(strain_yy),
                        'stress_xx_layers_Pa': list(stress_xx), 'n_cells': list(fields['u_x'].shape),
                        'length_m': length}}


def _check_shear(results_by_name):
    """Both shear runs: u, strains, stresses and the four support reactions against the simple-shear closed form."""
    _, mu = _lame(*_SHEAR_ELASTIC)
    names = ('u_x', 'u_y', 'strain_xx', 'strain_yy', 'strain_xy', 'stress_xx', 'stress_yy', 'stress_xy', 'stress_zz',
             'von_mises')
    component = {'x': 0, 'y': 1}
    errors, forces = {}, {}
    for run, axis in _SHEAR_RUNS.items():
        results = results_by_name[run]
        across, _, _, _, extent_s, extent_t = _shear_geometry(axis)
        gamma = _SHEAR_DISPLACEMENT / extent_t
        shear_stress = mu * gamma
        fields, x_edges, y_edges = _window_fields(results, names)
        centres = {'x': 0.5 * (x_edges[:-1] + x_edges[1:])[:, None], 'y': 0.5 * (y_edges[:-1] + y_edges[1:])[None, :]}
        expected = {name: 0.0 for name in names}
        expected.update({f'u_{axis}': gamma * centres[across], 'strain_xy': 0.5 * gamma, 'stress_xy': shear_stress,
                         'von_mises': np.sqrt(3.0) * shear_stress})
        for name in names:
            scale = _SHEAR_DISPLACEMENT if name.startswith('u_') else gamma if name.startswith('strain') else shear_stress
            errors[f'{run}.{name}'] = max_relative_error(fields[name], expected[name], scale)
        reactions = {feature: _reaction(results, feature) for feature in ('moved', 'base', 'side_lo', 'side_hi')}
        moved_force, side_force = shear_stress * extent_s, shear_stress * extent_t
        s_index, t_index = component[axis], component[across]
        expected_forces = {('moved', s_index): moved_force, ('base', s_index): -moved_force,
                           ('moved', t_index): 0.0, ('base', t_index): 0.0,
                           ('side_hi', t_index): side_force, ('side_lo', t_index): -side_force}
        for (feature, index), force in expected_forces.items():
            errors[f'{run}.F_{feature}_{"xy"[index]}'] = abs(reactions[feature][index] - force) / moved_force
        forces[run] = {feature: list(values) for feature, values in reactions.items()}
    gamma_x = _SHEAR_DISPLACEMENT / _shear_geometry('x')[5]
    measured = forces['shear_x']['moved'][0]
    return {'measured': measured, 'expected': mu * gamma_x * _shear_geometry('x')[4], 'error': worst_error(errors),
            'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N/m', 'reactions_F_x_F_y': forces, 'mu_Pa': mu}}


# ---- template ----
# heated_waveguide_stress: 6 x 4 um cross-section of a long waveguide (plane strain is exact for it: eps_zz = 0), uniform
# 20 nm mesh (300 x 200 cells), every edge on a cell edge: silica everywhere, a 0.48 x 0.22 um silicon core at x =
# 2.76..3.24 um, y = 1.5..1.72 um, a 2 x 0.2 um gold heater at x = 2..4 um, y = 3.0..3.2 um. Heat: P_heater = 20 W/m
# (Q = 5e13 W/m^3), y_lo held at 293.15 K (substrate), weak convection on y_hi; about 3 um of silica (k 1.38) to the sink
# gives a heater rise of ~10 K and ~5 K at the core. Solid: y_lo fixed (substrate), rollers on x_lo / x_hi (the
# cross-section continues periodically), y_hi free; the coupling hands T - T0 to the solid. The silicon core (alpha 2.6e-6
# against 5.5e-7 of silica) ends up in compression of the order of MPa. Validation: no errors.
HEATED_WAVEGUIDE_STRESS_LABEL = 'Heated waveguide stress'
HEATED_WAVEGUIDE_STRESS_DESCRIPTION = (
    'Heat + solid mechanics (plane strain), stationary: cross-section of a 0.48 x 0.22 um silicon waveguide in silica '
    'under a 2 x 0.2 um gold heater (P_heater = 20 W/m), substrate y_lo held at 293.15 K and clamped, rollers on the '
    'sides; heat_solid_thermal_strain hands the temperature rise (~5 K at the core) to the solid; derived T_core, '
    'stress_xx_core, stress_yy_core, von_mises_max.')
_TEMPLATE_DOMAIN = (6e-6, 4e-6)
_TEMPLATE_CELL = 20e-9
_TEMPLATE_CORE = (2.76e-6, 1.5e-6, 3.24e-6, 1.72e-6)
_TEMPLATE_HEATER = (2.0e-6, 3.0e-6, 4.0e-6, 3.2e-6)
_TEMPLATE_AMBIENT = 293.15


def heated_waveguide_stress_spec():
    """Template 'heated_waveguide_stress' (heat + solid_mechanics, stationary): thermal stress of a heated silicon
    waveguide cross-section in silica."""
    size_x, size_y = _TEMPLATE_DOMAIN
    heater_x0, heater_y0, heater_x1, heater_y1 = _TEMPLATE_HEATER
    heater_area = (heater_x1 - heater_x0) * (heater_y1 - heater_y0)
    return ModelSpec(
        title=HEATED_WAVEGUIDE_STRESS_LABEL,
        parameters=[ParameterSpec('P_heater', 20.0, 'heater power per m of waveguide length [W/m]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[_rectangle('core', *_TEMPLATE_CORE), _rectangle('heater', *_TEMPLATE_HEATER)],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('waveguide', library='silicon', selection=['core']),
                   MaterialSpec('heater_metal', library='gold', selection=['heater'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
                     FeatureSpec('heater_power', 'heat_source', ['heater'], {'Q': f'P_heater / {heater_area:.6g}'}),
                     FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': _TEMPLATE_AMBIENT}),
                     FeatureSpec('air', 'convection', ['y_hi'], {'h': 10.0, 'T_ext': _TEMPLATE_AMBIENT})]),
                 PhysicsSpec(_SOLID, SOLID_TYPE_NAME, features=[
                     FeatureSpec('substrate', 'fixed', ['y_lo']),
                     FeatureSpec('sides', 'roller', ['x_lo', 'x_hi'])])],
        couplings=[CouplingSpec('thermal_strain', THERMAL_STRAIN_COUPLING, _HEAT, _SOLID)],
        mesh=MeshSpec(mode='uniform', dx=_TEMPLATE_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_core', 'T', ['core'], 'mean'),
                        DerivedValueSpec('stress_xx_core', f'{_SOLID}_stress_xx', ['core'], 'mean'),
                        DerivedValueSpec('stress_yy_core', f'{_SOLID}_stress_yy', ['core'], 'mean'),
                        DerivedValueSpec('von_mises_max', f'{_SOLID}_von_mises', [], 'max')])


# ---- cases ----
_BUDGET_TEXT = (f'tolerance {_TOLERANCE:g}: the exact field lies in the bilinear element space, so only the CG '
                f'residual (float64, {EXACT_CG_TOLERANCE:g} times a condition number ~1e5) and the float32 '
                'storage of the fields (~1e-7 relative; reactions come from the float64 solution) remain')
_BILAYER_TEXT = (
    f'a {_BILAYER_SIZE[0] * 1e6:g} x {_BILAYER_SIZE[1] * 1e6:g} um strip, layer a (E {_BILAYER_LAYERS[0][0]:g} Pa, nu '
    f'{_BILAYER_LAYERS[0][1]:g}) on y < {_BILAYER_INTERFACE_Y * 1e6:g} um and layer b (E {_BILAYER_LAYERS[1][0]:g} Pa, '
    f'nu {_BILAYER_LAYERS[1][1]:g}) above, roller left on x_lo, roller bottom on y_lo, displacement pull u_x = '
    f'{_BILAYER_PULL:g} m on x_hi (u_y free), y_hi free; stationary, float64 CG to {EXACT_CG_TOLERANCE:g}. '
    'Exact: eps_xx = d / L, sigma_yy = sigma_xy = 0, eps_yy = -nu / (1 - nu) eps_xx per layer (u_y piecewise linear, kink '
    'on the interface cell edge), sigma_xx = E / (1 - nu^2) eps_xx, sigma_zz = nu sigma_xx, von Mises = sigma_xx sqrt(1 '
    f'- nu + nu^2), pull reaction F = sum of sigma_xx h = {_bilayer_closed_forms()[3]:.6g} N/m and the left roller -F. '
    'error = max(max |u_x - eps_xx x| / d, max |u_y - u_y,exact| / max |u_y|, max strain errors / eps_xx, max stress '
    'and von Mises errors / max sigma_xx, |F_pull - F| / F, |F_left + F| / F, |F_bottom,y| / F)')
# pcg 'mixed' (system.ElasticSystem._bind_linear_solver) on both backends: the gpu runs its CSR RawKernel products in
# captured CG blocks, the cpu scipy's in-place kernel
_MIXED_VARIANTS = (SelfTestVariant('mixed', tolerance=_TOLERANCE, precision='mixed', roles=('linear',)),
                   SelfTestVariant('mixed_cpu', tolerance=_TOLERANCE, backend='cpu', precision='mixed',
                                   roles=('linear',)))
_MIXED_TEXT = ("Variants 'mixed' / 'mixed_cpu' repeat the run with pcg precision mixed (float32 inner CG on float32 "
               'copies of the CSR and the 2 x 2 inverse blocks, float64 residual refreshes) at the same tolerance, on '
               'the gpu and the cpu backend.')
SELFTEST_CASES = (
    SelfTestCase(
        name='solid_bilayer_tension_graded', label='Solid mechanics: bilayer under tension on a graded mesh',
        description=(f'Plane-strain solid mechanics on {_BILAYER_TEXT}, on a graded mesh: mode nonuniform (base '
                     f'{_CELL * 1e6:g} um, uniform away from the band) with a vertical refinement line at x = L / 2 (dx '
                     f'{_BILAYER_GRADED_CELL * 1e6:g} um, margin {_BILAYER_GRADED_REACH * 1e6:g} um), y stays uniform; '
                     'the closed forms are evaluated at the actual cell centres (edges rebuilt from the float64 '
                     'centres) and the check raises unless the widest / narrowest x cell >= '
                     f'{_GRADED_MIN_RATIO:g}. The exact field stays in the element space on any rectilinear mesh. '
                     f'measured = F_pull; {_BUDGET_TEXT}. This pins the Q1 stiffness (both Lame terms) on equal and '
                     'unequal-width elements, the Poisson contraction under plane strain, a material interface, the '
                     'roller / displacement supports, the natural traction-free faces, the cell-centre strains, '
                     'stresses and von Mises and the reaction read-out (the uniform-mesh run of the former '
                     'solid_bilayer_tension is contained). Not yet run with the strain / von Mises items. '
                     f'{_MIXED_TEXT}'),
        specs=_bilayer_specs, check=_check_bilayer, variants=_MIXED_VARIANTS),
    SelfTestCase(
        name='solid_simple_shear', label='Solid mechanics: simple shear along x and y',
        description=(f'Plane-strain solid mechanics on a {_SHEAR_SIZE[0] * 1e6:g} x {_SHEAR_SIZE[1] * 1e6:g} um block '
                     f'of uniform {_CELL * 1e6:g} um cells (E {_SHEAR_ELASTIC[0]:g} Pa, nu {_SHEAR_ELASTIC[1]:g}), two '
                     'runs: shear_x (base fixed on y_lo, moved displacement u_x = d, u_y = 0 on y_hi, side_lo / side_hi '
                     'displacement u_y = 0 on x_lo / x_hi, u_x free) and shear_y (the same rotated: base x_lo, moved '
                     f'x_hi with u_y = d, sides y_lo / y_hi with u_x = 0); d = {_SHEAR_DISPLACEMENT:g} m; stationary. '
                     'Exact: u_s = gamma t (gamma = d / extent across), u_t = 0, strain_xy = gamma / 2, stress_xy = mu '
                     'gamma, von Mises = sqrt(3) mu gamma, every other strain / stress 0; reactions moved F_s = mu '
                     'gamma extent_s, base -F_s, side_hi F_t = mu gamma extent_t, side_lo -that, moved / base F_t = 0. '
                     'error = max(max |u - u_exact| / d, max strain errors / gamma, max stress and von Mises errors / '
                     f'(mu gamma), |F - F_exact| / (mu gamma extent_s)) over both runs; measured = F_x of moved in '
                     f'shear_x; {_BUDGET_TEXT}. This pins the shear (c66) terms of the stiffness in both cross '
                     'derivatives, stress_xy / strain_xy, the shear term of von Mises, partial (one-component) '
                     'displacement supports and their reactions, which every other case leaves at gamma = 0. Not yet '
                     'run with the von Mises item.'),
        specs=_shear_specs, check=_check_shear),
)


# ---- solid_em_strain_optic_plate: heated free plate -> thermo-optic + strain-optic index of em_fdfd_ez ----
# Only a component of the backend case solid_optics_cpu_gpu (the standalone run is the example
# multiphysics/analytic_constrained_plate_thermo_strain_optic). The _PLATE_* plate with isotropic alpha 1e-5, dndT 1e-5, eps_r 2.085, p11 0.121, p12 0.270 and an
# em_fdfd_ez physics (no source: only its eps_r matters); heat_em_thermo_optic and solid_em_strain_optic (deformation
# on) both add to eps_r_update. Uniform: total strain xx = yy = (1 + nu) alpha dT, elastic strain xx = yy = nu alpha dT,
# zz = -alpha dT, so n = n0 + dndT dT (alpha factor dropped) + (n0 + dn_pe) sqrt(1 + 2 (1 + nu) alpha dT) - n0 with
# dn_pe = -n0^3 / 2 (p11 e_zz + p12 (e_xx + e_yy)).
_OPTIC_ALPHA = 1e-5
_OPTIC_DNDT = 1e-5
_OPTIC_EPS = 2.085
_OPTIC_P11, _OPTIC_P12 = 0.121, 0.270
_OPTIC_WAVELENGTH = 20e-6                            # m: ~14 cells per wavelength in the plate
_OPTIC_TOLERANCE = 1e-3                              # relative to the index change (float32 eps: ~3e-5)


def _strain_optic_closed_form():
    """(n0, expected n, the n the alpha factor would add on top) of every plate cell."""
    _, nu = _PLATE_ELASTIC
    rise = _PLATE_RISE
    n0 = float(np.sqrt(np.float32(_OPTIC_EPS)))
    elastic_in_plane = nu * _OPTIC_ALPHA * rise
    elastic_zz = -_OPTIC_ALPHA * rise
    total_in_plane = (1.0 + nu) * _OPTIC_ALPHA * rise
    photoelastic = -0.5 * n0 ** 3 * (_OPTIC_P11 * elastic_zz + _OPTIC_P12 * 2.0 * elastic_in_plane)
    strain_shift = (n0 + photoelastic) * np.sqrt(1.0 + 2.0 * total_in_plane) - n0
    expansion_factor_shift = n0 * _OPTIC_ALPHA * rise   # what a kept (1 + alpha dT) factor would add, ~first order
    return n0, n0 + _OPTIC_DNDT * rise + strain_shift, expansion_factor_shift


def _strain_optic_specs():
    """'run': heat walls at T0 + 100 K, thermal strain, heat_em_thermo_optic and solid_em_strain_optic into 'em'."""
    size_x, size_y = _PLATE_SIZE
    youngs, nu = _PLATE_ELASTIC
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: thermo-optic + strain-optic index of a heated free plate',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        materials=[MaterialSpec('plate', properties=_elastic(
            youngs, nu, k=10.0, alpha=_OPTIC_ALPHA, dndT=_OPTIC_DNDT, eps_r=_OPTIC_EPS, photoelastic_p11=_OPTIC_P11,
            photoelastic_p12=_OPTIC_P12), selection=['all'])],
        physics=[PhysicsSpec(_EM, 'em_fdfd_ez', settings={'wavelength': _OPTIC_WAVELENGTH}),
                 PhysicsSpec(_HEAT, 'heat', settings={'T0': _PLATE_T0}, features=[
                     FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi', 'y_lo', 'y_hi'],
                                 {'temperature': _PLATE_T0 + _PLATE_RISE})]),
                 PhysicsSpec(_SOLID, SOLID_TYPE_NAME, features=[
                     FeatureSpec('left', 'roller', ['x_lo']),
                     FeatureSpec('bottom', 'roller', ['y_lo'])])],
        couplings=[CouplingSpec('expansion', THERMAL_STRAIN_COUPLING, _HEAT, _SOLID),
                   CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _EM),
                   CouplingSpec('strain_optic', STRAIN_OPTIC_COUPLING, _SOLID, _EM)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary')))}


def _check_strain_optic(results_by_name):
    """sqrt(eps_r) of every window cell against the closed form, relative to the index change."""
    results = results_by_name[_RUN]
    index = np.sqrt(np.asarray(results.field(f'{_EM}.eps_r').values, dtype=np.float64))
    n0, expected, expansion_factor_shift = _strain_optic_closed_form()
    change = expected - n0
    error = float(np.max(np.abs(index - expected))) / abs(change)
    return {'measured': float(np.mean(index)), 'expected': expected, 'error': error, 'tolerance': _OPTIC_TOLERANCE,
            'details': {'n0': n0, 'index_change_expected': change, 'index_change_mean': float(np.mean(index)) - n0,
                        'kept_alpha_factor_would_add': expansion_factor_shift, 'n_cells': list(index.shape)}}


STRAIN_OPTIC_CASES = (
    SelfTestCase(
        name='solid_em_strain_optic_plate', label='Solid -> EM: thermo-optic + strain-optic index of a heated plate',
        description=(f'A {_PLATE_SIZE[0] * 1e6:g} x {_PLATE_SIZE[1] * 1e6:g} um plate of {_CELL * 1e6:g} um cells '
                     f'(E {_PLATE_ELASTIC[0]:g} Pa, nu {_PLATE_ELASTIC[1]:g}, k 10; heat holds all four faces at T0 + '
                     f'{_PLATE_RISE:g} K, rollers on x_lo / y_lo) with isotropic alpha {_OPTIC_ALPHA:g} 1/K, dndT '
                     f'{_OPTIC_DNDT:g} 1/K, eps_r {_OPTIC_EPS:g}, photoelastic p11 {_OPTIC_P11:g} / p12 {_OPTIC_P12:g} '
                     f'and an em_fdfd_ez physics (wavelength {_OPTIC_WAVELENGTH:g} m, no source: only its eps_r is '
                     'checked); heat_em_thermo_optic and solid_em_strain_optic (deformation on) both contribute to '
                     'eps_r_update; stationary. Exact (uniform): total strain xx = yy = (1 + nu) alpha dT, elastic '
                     'strain xx = yy = nu alpha dT, zz = -alpha dT; n = n0 + dndT dT + (n0 + dn_pe) sqrt(1 + strain_xx '
                     '+ strain_yy) - n0, dn_pe = -n0^3 / 2 (p11 e_zz + p12 (e_xx + e_yy)), n0 = sqrt(float32 eps_r): '
                     'the thermo-optic term without its (1 + alpha dT) factor, which a strain-optic coupling with '
                     'deformation replaces (keeping it would add ~n0 alpha dT, about half the change). error = max '
                     f'|sqrt(eps_r) - n| / (n - n0) over the window; tolerance {_OPTIC_TOLERANCE:g} (float32 eps_r, ~3e-5). This '
                     'pins the elastic strain fields, the photoelastic and deformation terms, the E13 index sum and '
                     'the dropped alpha factor.'),
        specs=_strain_optic_specs, check=_check_strain_optic),
)

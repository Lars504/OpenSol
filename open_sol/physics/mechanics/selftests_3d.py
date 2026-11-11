"""Analytic self-tests of solid_mechanics_3d (uniaxial block on a graded mesh; extruded bilayer = the 2D plane-strain limit;
confined column along z with z-dependent feature expressions; heat -> free and confined thermal expansion;
rigid-mode rank check) and the 'heated_chip_stress_3d' template; definition3d registers them."""
import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, cell_edges, max_relative_error, with_exact_cg, worst_error
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec, StudySpec)
from .selftests import (_BILAYER_INTERFACE_Y, _BILAYER_LAYERS, _BILAYER_PULL, _BILAYER_SIZE, _bilayer_closed_forms,
                        _elastic, _lame)

SOLID_3D_TYPE_NAME = 'solid_mechanics_3d'           # = definition3d.SOLID_3D_TYPE_NAME (definition3d imports this)
THERMAL_STRAIN_COUPLING = 'heat_solid_thermal_strain'
_SOLID = 'solid'                                     # physics instance names
_HEAT = 'heat'
_RUN = 'run'
_TOLERANCE = 1e-6
_CELL = 1e-6                                         # m, uniform cubic cells of every case
_GRADED_MIN_RATIO = 2.0                              # widest / narrowest window x cell a graded run must reach
_ALL_STRESSES = ('stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'stress_xz', 'stress_yz')


def _box(name, x0, y0, z0, x1, y1, z1):
    return GeometrySpec(name, 'box', {'x0': x0, 'y0': y0, 'z0': z0, 'x1': x1, 'y1': y1, 'z1': z1})


def _solid_spec(title, size, materials, features, geometry=(), mesh=None):
    """Stationary solid_mechanics_3d-only model on a size = (x, y, z) box."""
    size_x, size_y, size_z = size
    return with_exact_cg(ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_SOLID, SOLID_3D_TYPE_NAME, features=list(features))],
        mesh=mesh or MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary')))


# ---- checks' readers (CPU, on Results-like objects) ----
def _cut_fields(results, names, graded=False):
    """({name: float64 mid-height cut}, x centres, y centres, z of the cut [m]); ValueError unless a graded run has x
    widths spanning at least _GRADED_MIN_RATIO."""
    values, x_centres, y_centres, z_cut = {}, None, None, None
    for name in names:
        field_data = results.field(f'{_SOLID}.{name}')
        values[name] = np.asarray(field_data.values, dtype=np.float64)
        if x_centres is None:
            x_centres = np.asarray(field_data.x_centers, dtype=np.float64)
            y_centres = np.asarray(field_data.y_centers, dtype=np.float64)
            z_cut = float(field_data.plane[1])   # cell-centre height of the cut
            x_edges = cell_edges(x_centres, float(field_data.extent[0]))
    widths = np.diff(x_edges)
    if graded and not np.max(widths) >= _GRADED_MIN_RATIO * np.min(widths):
        raise ValueError(f'expected graded x cells (widest / narrowest >= {_GRADED_MIN_RATIO:g}), got {widths.size} '
                         f'cells of {np.min(widths):.4g} .. {np.max(widths):.4g} m')
    return values, x_centres, y_centres, z_cut


def _reaction(results, feature):
    """(F_x, F_y, F_z) [N] of a support feature's table result."""
    table = results.result(_SOLID, feature)
    return float(table['F_x']), float(table['F_y']), float(table['F_z'])


def _probe_value(results, feature):
    """Last sample of a solid_probe."""
    return float(np.asarray(results.result(_SOLID, feature)['values'], dtype=np.float64)[-1])


def _field_errors(fields, expected, scales):
    """{name: max |field - expected| / scale} over the cut."""
    return {name: max_relative_error(fields[name], expected.get(name, 0.0), scales[name]) for name in fields}


# ---- solid_3d_uniaxial_block_graded: a block pulled along x, free lateral faces, on a graded mesh ----
# 20 x 10 x 8 um, base 1 um cells refined around x = L / 2 (uniform 1 um cells away from the band), E 100 GPa, nu 0.3;
# rollers left (x_lo), bottom (y_lo), back (z_lo), displacement pull u_x = d on x_hi (u_y, u_z free), the other faces
# free. Exact (linear, in the trilinear space on any rectilinear mesh): eps_xx = d / L, eps_yy = eps_zz = -nu eps_xx,
# u_i = eps_ii x_i, sigma_xx = E eps_xx = von Mises, every other stress 0; pull reaction F = sigma_xx H D [N], left -F,
# bottom / back 0.
_BLOCK_SIZE = (20e-6, 10e-6, 8e-6)
_BLOCK_ELASTIC = (100e9, 0.3)
_BLOCK_PULL = 2e-8                                   # m, strain 1e-3
_BLOCK_GRADED_CELL = 0.4e-6                          # m, refinement line x = L / 2 (unset z: refines every z layer)
_BLOCK_GRADED_REACH = 6e-6                           # m


def _block_specs():
    """'run': rollers left / bottom / back, displacement 'pull' on x_hi, refinement line at x = L / 2."""
    length, height, _ = _BLOCK_SIZE
    geometry = [GeometrySpec('refine_line', 'line', {'x0': 0.5 * length, 'y0': 0.0, 'x1': 0.5 * length, 'y1': height})]
    mesh = MeshSpec(mode='nonuniform', dx=_CELL,
                    refinements=[RefinementSpec('refine_line', _BLOCK_GRADED_CELL, _BLOCK_GRADED_REACH)])
    return {_RUN: _solid_spec(
        'Self-test: uniaxial block (solid mechanics 3D), graded mesh', _BLOCK_SIZE,
        [MaterialSpec('block', properties=_elastic(*_BLOCK_ELASTIC), selection=['all'])],
        [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
         FeatureSpec('back', 'roller', ['z_lo']), FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _BLOCK_PULL})],
        geometry, mesh)}


def _check_block(results_by_name):
    """Displacements, strains, stresses and von Mises of the mid-height cut and the four reactions."""
    results = results_by_name[_RUN]
    names = ('u_x', 'u_y', 'u_z', 'strain_xx', 'strain_yy', 'strain_zz') + _ALL_STRESSES + ('von_mises',)
    fields, x_centres, y_centres, z_cut = _cut_fields(results, names, graded=True)
    length, height, depth = _BLOCK_SIZE
    youngs, nu = _BLOCK_ELASTIC
    strain = _BLOCK_PULL / length
    stress = youngs * strain
    force = stress * height * depth
    expected = {'u_x': strain * x_centres[:, None], 'u_y': -nu * strain * y_centres[None, :],
                'u_z': -nu * strain * z_cut, 'strain_xx': strain, 'strain_yy': -nu * strain,
                'strain_zz': -nu * strain, 'stress_xx': stress, 'von_mises': stress}
    scales = {name: _BLOCK_PULL if name.startswith('u_') else strain if name.startswith('strain') else stress
              for name in names}
    errors = _field_errors(fields, expected, scales)
    pull, left, bottom, back = (_reaction(results, feature) for feature in ('pull', 'left', 'bottom', 'back'))
    errors.update({'F_pull': abs(pull[0] - force) / force, 'F_left': abs(left[0] + force) / force,
                   'F_bottom': abs(bottom[1]) / force, 'F_back': abs(back[2]) / force})
    return {'measured': pull[0], 'expected': force, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N', 'F_pull': list(pull), 'F_left': list(left),
                        'strain_xx': strain, 'stress_xx_Pa': stress, 'z_cut_m': z_cut,
                        'cut_cells': list(fields['u_x'].shape)}}


# ---- solid_3d_extruded_bilayer: the 2D bilayer with depth and rollers on both z faces ----
# The solid_bilayer_tension_graded layers (selftests.py) on a shorter strip of length L_e at the same strain, extruded
# to depth D with rollers on z_lo and z_hi: u_z = 0 everywhere is plane strain, so the mid-height cut reproduces the 2D
# closed forms (exact in the trilinear space at any length or depth; u_z = sigma_xz = sigma_yz = 0 added) and the pull
# reaction is the 2D one [N/m] times D [N].
_EXTRUDED_LENGTH = 20e-6                             # m (the 2D strip is 100 um: CG iterations scale with the longest edge)
_EXTRUDED_DEPTH = 2e-6                               # m, 2 z cells
_EXTRUDED_PULL = _BILAYER_PULL * _EXTRUDED_LENGTH / _BILAYER_SIZE[0]   # m: the 2D strain d / L


def _extruded_bilayer_specs():
    """'run': rollers left / bottom / sides (z_lo, z_hi), displacement 'pull' on x_hi."""
    length, height = _EXTRUDED_LENGTH, _BILAYER_SIZE[1]
    (youngs_a, nu_a), (youngs_b, nu_b) = _BILAYER_LAYERS
    return {_RUN: _solid_spec(
        'Self-test: extruded bilayer under tension (solid mechanics 3D)', (length, height, _EXTRUDED_DEPTH),
        [MaterialSpec('layer_a', properties=_elastic(youngs_a, nu_a), selection=['lower']),
         MaterialSpec('layer_b', properties=_elastic(youngs_b, nu_b), selection=['upper'])],
        [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
         FeatureSpec('sides', 'roller', ['z_lo', 'z_hi']),
         FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _EXTRUDED_PULL})],
        [_box('lower', 0.0, 0.0, 0.0, length, _BILAYER_INTERFACE_Y, _EXTRUDED_DEPTH),
         _box('upper', 0.0, _BILAYER_INTERFACE_Y, 0.0, length, height, _EXTRUDED_DEPTH)])}


def _check_extruded_bilayer(results_by_name):
    """Mid-height cut against the 2D plane-strain closed forms; reactions = the 2D ones times the depth."""
    results = results_by_name[_RUN]
    names = ('u_x', 'u_y', 'u_z') + _ALL_STRESSES
    fields, x_centres, y_centres, _ = _cut_fields(results, names)
    strain_xx, strain_yy, stress_xx, force_per_depth = _bilayer_closed_forms(_EXTRUDED_PULL / _EXTRUDED_LENGTH)
    force = force_per_depth * _EXTRUDED_DEPTH
    lower = y_centres < _BILAYER_INTERFACE_Y
    u_y_expected = np.where(lower, strain_yy[0] * y_centres,
                            strain_yy[0] * _BILAYER_INTERFACE_Y + strain_yy[1] * (y_centres - _BILAYER_INTERFACE_Y))
    expected = {'u_x': strain_xx * x_centres[:, None], 'u_y': u_y_expected[None, :],
                'stress_xx': np.where(lower, stress_xx[0], stress_xx[1])[None, :],
                'stress_zz': np.where(lower, _BILAYER_LAYERS[0][1] * stress_xx[0],
                                      _BILAYER_LAYERS[1][1] * stress_xx[1])[None, :]}
    u_y_scale = float(np.max(np.abs(u_y_expected)))
    scales = {name: max(stress_xx) for name in _ALL_STRESSES}
    scales.update({'u_x': _EXTRUDED_PULL, 'u_y': u_y_scale, 'u_z': u_y_scale})
    errors = _field_errors(fields, expected, scales)
    pull, left, bottom, sides = (_reaction(results, feature) for feature in ('pull', 'left', 'bottom', 'sides'))
    errors.update({'F_pull': abs(pull[0] - force) / force, 'F_left': abs(left[0] + force) / force,
                   'F_bottom': abs(bottom[1]) / force, 'F_sides': abs(sides[2]) / force})
    return {'measured': pull[0], 'expected': force, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N', 'F_pull': list(pull), 'F_left': list(left),
                        'F_sides': list(sides), 'force_per_depth_N_per_m': force_per_depth,
                        'length_m': _EXTRUDED_LENGTH, 'depth_m': _EXTRUDED_DEPTH}}


# ---- solid_3d_column_z_expressions: a laterally confined column along z with z-dependent feature expressions ----
# 6 x 6 x 30 um of 1 um cells, E 70 GPa, nu 0.3, isotropic alpha; rollers sides (x_lo, x_hi, y_lo, y_hi) and base
# (z_lo), traction t_z on z_hi, body_force f_z = g z / H on the upper half (box z >= H/2: a BoxValues on a partial box)
# and temperature_change delta_T = R z / H on all (a BoxValues on the grid). 1D along z (M = lambda + 2 mu); expected =
# the discrete 1D solution (cell loads f_c h / 2 to both nodes, as the solver lumps them): sigma_c = sum of the nodal
# loads above cell c, eps_c = (sigma_c + beta dT_c) / M, u_z by summing eps_c h, sigma_xx = lambda eps_c - beta dT_c
# (beta = (3 lambda + 2 mu) alpha). Exact in the trilinear space, any f(z), dT(z); base reaction -(t_z + int f dz) W^2.
_COLUMN_SIZE = (6e-6, 6e-6, 30e-6)
_COLUMN_ELASTIC = (70e9, 0.3)
_COLUMN_TRACTION = -1e6                              # Pa on z_hi (compression)
_Z_EXPRESSION_ALPHA = 1e-5                           # 1/K
_Z_EXPRESSION_FORCE_TOP = -2e10                      # N/m^3, g = f_z at z = H
_Z_EXPRESSION_RISE_TOP = 50.0                        # K, R = delta_T at z = H
_Z_EXPRESSION_PROBE_CELLS = (0, 11, 22, 29)


def _z_expression_specs():
    """'run': the confined column with the z-dependent body force 'load' on 'upper' and temperature change 'heating'."""
    size_x, size_y, size_z = _COLUMN_SIZE
    points = [GeometrySpec(f'p{index}', 'point', {'x': 0.5 * size_x + 0.5 * _CELL, 'y': 0.5 * size_y + 0.5 * _CELL,
                                                  'z': (index + 0.5) * _CELL})
              for index in _Z_EXPRESSION_PROBE_CELLS]
    probes = [FeatureSpec(f'{component}_{index}', 'solid_probe', [f'p{index}'], {'component': component})
              for index in _Z_EXPRESSION_PROBE_CELLS for component in ('u_z', 'stress_zz', 'stress_xx')]
    return {_RUN: _solid_spec(
        'Self-test: confined column with z-dependent body force and temperature change (solid mechanics 3D)',
        _COLUMN_SIZE,
        [MaterialSpec('column', properties=_elastic(*_COLUMN_ELASTIC, alpha=_Z_EXPRESSION_ALPHA), selection=['all'])],
        [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']), FeatureSpec('base', 'roller', ['z_lo']),
         FeatureSpec('top', 'boundary_load', ['z_hi'], {'t_x': 0.0, 't_y': 0.0, 't_z': _COLUMN_TRACTION}),
         FeatureSpec('load', 'body_force', ['upper'],
                     {'f_x': 0.0, 'f_y': 0.0, 'f_z': f'{_Z_EXPRESSION_FORCE_TOP!r} * z / {size_z!r}'}),
         FeatureSpec('heating', 'temperature_change', ['all'],
                     {'delta_T': f'{_Z_EXPRESSION_RISE_TOP!r} * z / {size_z!r}'})] + probes,
        [_box('upper', 0.0, 0.0, 0.5 * size_z, size_x, size_y, size_z)] + points)}


def _z_expression_discrete():
    """Discrete 1D solution per z cell: (centre u_z, sigma_zz, sigma_xx, dT, nodal u_z, sum of f_c h [Pa], eps_zz)."""
    lame_lambda, mu = _lame(*_COLUMN_ELASTIC)
    modulus = lame_lambda + 2.0 * mu
    beta = (3.0 * lame_lambda + 2.0 * mu) * _Z_EXPRESSION_ALPHA
    height = _COLUMN_SIZE[2]
    n_cells = int(round(height / _CELL))
    centres = (np.arange(n_cells) + 0.5) * _CELL
    force = np.where(centres >= 0.5 * height, _Z_EXPRESSION_FORCE_TOP * centres / height, 0.0)
    rise = _Z_EXPRESSION_RISE_TOP * centres / height
    nodal = np.zeros(n_cells + 1)
    nodal[:-1] += 0.5 * force * _CELL
    nodal[1:] += 0.5 * force * _CELL
    nodal[-1] += _COLUMN_TRACTION
    stress_zz = np.cumsum(nodal[::-1])[::-1][1:]   # cell c carries the loads of nodes c + 1 .. n
    strain_zz = (stress_zz + beta * rise) / modulus
    nodes = np.concatenate(([0.0], np.cumsum(strain_zz * _CELL)))
    return (0.5 * (nodes[:-1] + nodes[1:]), stress_zz, lame_lambda * strain_zz - beta * rise, rise, nodes,
            float(np.sum(force) * _CELL), strain_zz)


def _check_z_expressions(results_by_name):
    """Probes and the mid-height cut (u_z, strain_zz, stresses, von Mises = |sigma_zz - sigma_xx|, temperature_rise_ext)
    against the discrete 1D solution; reactions."""
    results = results_by_name[_RUN]
    size_x, size_y, _ = _COLUMN_SIZE
    u_z, stress_zz, stress_xx, rise, nodes, force_integral, strain_zz = _z_expression_discrete()
    displacement_scale = float(np.max(np.abs(nodes)))
    stress_scale = float(max(np.max(np.abs(stress_zz)), np.max(np.abs(stress_xx))))
    expected_by_name = {'u_z': u_z, 'stress_zz': stress_zz, 'stress_xx': stress_xx}
    errors, samples = {}, {}
    for index in _Z_EXPRESSION_PROBE_CELLS:
        for name, expected in expected_by_name.items():
            value = _probe_value(results, f'{name}_{index}')
            samples[f'{name}_{index}'] = value
            scale = displacement_scale if name == 'u_z' else stress_scale
            errors[f'{name}_{index}'] = abs(value - expected[index]) / scale
    names = ('u_x', 'u_y', 'u_z', 'strain_zz', 'stress_zz', 'stress_xx', 'stress_xz', 'stress_yz', 'von_mises',
             'temperature_rise_ext')
    fields, _, _, z_cut = _cut_fields(results, names)
    cut = int(round(z_cut / _CELL - 0.5))
    expected = {'u_z': u_z[cut], 'strain_zz': strain_zz[cut], 'stress_zz': stress_zz[cut], 'stress_xx': stress_xx[cut],
                'von_mises': abs(stress_zz[cut] - stress_xx[cut]),   # sigma_xx = sigma_yy, no shear
                'temperature_rise_ext': rise[cut]}
    scales = {name: displacement_scale if name.startswith('u_') else stress_scale for name in names}
    scales.update({'temperature_rise_ext': _Z_EXPRESSION_RISE_TOP, 'strain_zz': float(np.max(np.abs(strain_zz)))})
    errors.update(_field_errors(fields, expected, scales))
    base_force = -(_COLUMN_TRACTION + force_integral) * size_x * size_y
    base = _reaction(results, 'base')
    sides = _reaction(results, 'sides')
    errors.update({'F_base': abs(base[2] - base_force) / abs(base_force),
                   'F_sides_x': abs(sides[0]) / abs(base_force), 'F_sides_y': abs(sides[1]) / abs(base_force)})
    return {'measured': base[2], 'expected': base_force, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N', 'F_base': list(base), 'probes': samples,
                        'z_cut_m': z_cut, 'u_z_top_m': float(nodes[-1])}}


# ---- heat_solid_thermal_expansion_3d: a box heated uniformly, free and confined ----
# 12 x 8 x 6 um of 1 um cells, k 10, E 100 GPa, nu 0.3, alpha_xx / yy / zz = 1e-5 / 2e-5 / 0.5e-5; heat holds all six
# faces at T0 + 100 K (uniform rise), the coupling hands it to the solid. Run 'free': rollers 'low' on x_lo, y_lo, z_lo:
# u_i = alpha_ii dT x_i, zero stress, zero reactions. Run 'confined': rollers 'low' and 'high' (x_hi, y_hi, z_hi): u = 0,
# sigma_ii = -beta_i dT (beta_i = 2 mu alpha_ii + lambda sum alpha), elastic_strain_zz = -alpha_zz dT, reactions 'high' =
# (sigma_xx A_x, sigma_yy A_y, sigma_zz A_z) [N] (A_i the face area normal to i), 'low' = minus that.
_THERMAL_SIZE = (12e-6, 8e-6, 6e-6)
_THERMAL_ELASTIC = (100e9, 0.3)
_THERMAL_ALPHA = (1e-5, 2e-5, 0.5e-5)
_THERMAL_T0 = 293.15                                 # K, heat T0 = the stress-free temperature
_THERMAL_RISE = 100.0                                # K
_THERMAL_RUNS = {'free': ('low',), 'confined': ('low', 'high')}
_THERMAL_FACES = {'low': ['x_lo', 'y_lo', 'z_lo'], 'high': ['x_hi', 'y_hi', 'z_hi']}


def _thermal_specs():
    """'free' / 'confined': heat walls at T0 + 100 K on all six faces, the coupling, rollers per _THERMAL_RUNS."""
    youngs, nu = _THERMAL_ELASTIC
    alpha_xx, alpha_yy, alpha_zz = _THERMAL_ALPHA
    size_x, size_y, size_z = _THERMAL_SIZE
    faces = ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi']
    specs = {}
    for run, supports in _THERMAL_RUNS.items():
        specs[run] = with_exact_cg(ModelSpec(
            title=f'Self-test: heated box, {run} (heat -> solid mechanics 3D)',
            domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
            materials=[MaterialSpec('box', properties=_elastic(youngs, nu, k=10.0, alpha_xx=alpha_xx,
                                                               alpha_yy=alpha_yy, alpha_zz=alpha_zz),
                                    selection=['all'])],
            physics=[PhysicsSpec(_HEAT, 'heat', settings={'T0': _THERMAL_T0}, features=[
                         FeatureSpec('walls', 'temperature', faces, {'temperature': _THERMAL_T0 + _THERMAL_RISE})]),
                     PhysicsSpec(_SOLID, SOLID_3D_TYPE_NAME, features=[
                         FeatureSpec(name, 'roller', list(_THERMAL_FACES[name])) for name in supports])],
            couplings=[CouplingSpec('expansion', THERMAL_STRAIN_COUPLING, _HEAT, _SOLID)],
            mesh=MeshSpec(mode='uniform', dx=_CELL),
            study=StudySpec(type='stationary')))
    return specs


def _thermal_closed_forms():
    """(beta dT per axis [Pa], stress scale E max(alpha) dT [Pa], face areas normal to x / y / z [m^2])."""
    lame_lambda, mu = _lame(*_THERMAL_ELASTIC)
    alpha_sum = sum(_THERMAL_ALPHA)
    thermal_stress = tuple((2.0 * mu * alpha + lame_lambda * alpha_sum) * _THERMAL_RISE for alpha in _THERMAL_ALPHA)
    size_x, size_y, size_z = _THERMAL_SIZE
    return (thermal_stress, _THERMAL_ELASTIC[0] * max(_THERMAL_ALPHA) * _THERMAL_RISE,
            (size_y * size_z, size_x * size_z, size_x * size_y))


def _check_thermal(results_by_name):
    """Both runs: coupled rise, displacements, stresses, elastic_strain_zz of the mid-height cut and the reactions."""
    thermal_stress, stress_scale, areas = _thermal_closed_forms()
    alpha_xx, alpha_yy, alpha_zz = _THERMAL_ALPHA
    displacement_scale = max(_THERMAL_ALPHA) * _THERMAL_RISE * max(_THERMAL_SIZE)
    force_scale = stress_scale * max(areas)
    names = ('temperature_rise', 'u_x', 'u_y', 'u_z', 'strain_zz', 'elastic_strain_zz') + _ALL_STRESSES + ('von_mises',)
    stress_x, stress_y, stress_z = (-stress for stress in thermal_stress)
    confined_von_mises = np.sqrt(0.5 * ((stress_x - stress_y) ** 2 + (stress_y - stress_z) ** 2
                                        + (stress_z - stress_x) ** 2))
    errors, reactions = {}, {}
    for run, supports in _THERMAL_RUNS.items():
        results = results_by_name[run]
        fields, x_centres, y_centres, z_cut = _cut_fields(results, names)
        scales = {name: displacement_scale if name.startswith('u_') else stress_scale for name in names}
        scales.update({'temperature_rise': _THERMAL_RISE, 'strain_zz': alpha_zz * _THERMAL_RISE,
                       'elastic_strain_zz': alpha_zz * _THERMAL_RISE})
        expected = {'temperature_rise': _THERMAL_RISE}
        if run == 'free':
            expected.update({'u_x': alpha_xx * _THERMAL_RISE * x_centres[:, None],
                             'u_y': alpha_yy * _THERMAL_RISE * y_centres[None, :], 'u_z': alpha_zz * _THERMAL_RISE * z_cut,
                             'strain_zz': alpha_zz * _THERMAL_RISE})
            expected_forces = {'low': (0.0, 0.0, 0.0)}
        else:
            expected.update({'stress_xx': stress_x, 'stress_yy': stress_y, 'stress_zz': stress_z,
                             'von_mises': confined_von_mises, 'elastic_strain_zz': -alpha_zz * _THERMAL_RISE})
            high = tuple(-stress * area for stress, area in zip(thermal_stress, areas))
            expected_forces = {'high': high, 'low': tuple(-force for force in high)}
        errors.update({f'{run}.{name}': error for name, error in _field_errors(fields, expected, scales).items()})
        for feature in supports:
            measured = _reaction(results, feature)
            reactions[f'{run}.{feature}'] = list(measured)
            for axis, (value, target) in enumerate(zip(measured, expected_forces[feature])):
                errors[f'{run}.F_{feature}_{"xyz"[axis]}'] = abs(value - target) / force_scale
    measured_force = reactions['confined.high'][2]
    return {'measured': measured_force, 'expected': -thermal_stress[2] * areas[2], 'error': worst_error(errors),
            'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N', 'reactions_F_x_F_y_F_z': reactions,
                        'thermal_stress_Pa': list(thermal_stress), 'stress_scale_Pa': stress_scale}}


# ---- solid_3d_rigid_mode_check: the Gram rank test of ElasticSystem3D (no model run) ----
# 10 x 4 x 4 um of 1 um cells. (a) u_x held on y_lo, u_y on x_lo, u_z on z_lo: every direction held, rotation about z
# free (rank 5); (b) two blocks x < 4 um and x >= 6 um with a void gap, rollers on x_lo, y_lo, z_lo: the second block can
# slide along x; (c) the rollers of (b) on one solid block: rank 6, assembles.
_RIGID_SIZE = (10e-6, 4e-6, 4e-6)
_RIGID_GAP_CELLS = (4, 6)                            # void window x cells [4, 6) of scenario (b)


def _rigid_scenarios():
    """{name: (holds [(face, component)], gap, expected message parts or None = must assemble)}."""
    rollers = [('x_lo', 0), ('y_lo', 1), ('z_lo', 2)]
    second_block_x = (_RIGID_GAP_CELLS[1] + 0.5) * _CELL
    return {'rotation_about_z': ([('y_lo', 0), ('x_lo', 1), ('z_lo', 2)], False,
                                 ('rotation about the axis along z',)),
            'unsupported_block': (rollers, True, ('slide along x', f'x = {second_block_x:.6g} m')),
            'held_block': (rollers, False, None)}


def _check_rigid_modes(results_by_name):
    """Assemble each scenario on the cpu backend (and the gpu one when usable) and compare the raised message (or its
    absence) with the expectation."""
    from ...core.backend import gpu_available
    from ...core.domain3d import Domain3D
    from .system3d import ElasticSystem3D
    lame_lambda, mu = _lame(100e9, 0.3)
    outcomes, failures = {}, 0
    for backend in ('cpu', 'gpu') if gpu_available() else ('cpu',):
        for name, (holds, gap, message_parts) in _rigid_scenarios().items():
            domain = Domain3D.uniform(*_RIGID_SIZE, _CELL)
            domain.use_backend(backend)
            xp = domain.xp
            system = ElasticSystem3D('float64', 1e-12, 100, label=lambda: 'rigid-mode self-test', backend=backend)
            system.prepare(domain)
            solid = xp.ones(system.element_shape, dtype=xp.bool_)
            if gap:
                solid[_RIGID_GAP_CELLS[0]:_RIGID_GAP_CELLS[1]] = False
            for face, component in holds:
                system.hold_face(face, **{f'u_{"xyz"[component]}': 0.0})
            try:
                system.assemble(xp.where(solid, lame_lambda, 0.0), xp.where(solid, mu, 0.0), solid)
                message = None
            except ValueError as exc:
                message = str(exc)
            passed = (message is None if message_parts is None
                      else message is not None and all(part in message for part in message_parts))
            failures += 0 if passed else 1
            outcomes[f'{backend}.{name}'] = {'passed': passed, 'message': message, 'expected_parts': message_parts}
    return {'measured': len(outcomes) - failures, 'expected': len(outcomes), 'error': float(failures),
            'tolerance': 0.0, 'details': outcomes}


# ---- template ----
# heated_chip_stress_3d: the heat heated_chip_3d model (1 x 1 x 0.5 mm silicon block, 0.4 mm square chip dissipating
# 0.8 W in its top 50 um, z_lo held at 300 K, convection on z_hi; 25 um cells, 40 x 40 x 20) plus a solid fixed on z_lo
# (bonded to the heat sink), free elsewhere, heat_solid_thermal_strain. Library silicon (alpha 2.6e-6, E 165 GPa): a
# rise of a few K gives stresses of the order of MPa near the clamped base. No closed form; examples_validate only.
HEATED_CHIP_STRESS_LABEL = 'Heated chip stress (3D)'
_CHIP_BLOCK = (1e-3, 1e-3, 5e-4)
_CHIP_FOOTPRINT = (3e-4, 7e-4)
_CHIP_THICKNESS = 5e-5
_CHIP_POWER_DENSITY = 1e11                           # W/m^3: 0.8 W in the chip
_CHIP_SINK = 300.0                                   # K
_CHIP_CELL = 2.5e-5
HEATED_CHIP_STRESS_DESCRIPTION = (
    f'Heat + solid mechanics 3D, stationary: a {_CHIP_BLOCK[0] * 1e3:g} x {_CHIP_BLOCK[1] * 1e3:g} x '
    f'{_CHIP_BLOCK[2] * 1e3:g} mm silicon block on a {_CHIP_SINK:g} K heat sink (z_lo, also clamped) with a '
    f'{(_CHIP_FOOTPRINT[1] - _CHIP_FOOTPRINT[0]) * 1e3:g} mm square chip dissipating '
    f'{_CHIP_POWER_DENSITY * (_CHIP_FOOTPRINT[1] - _CHIP_FOOTPRINT[0]) ** 2 * _CHIP_THICKNESS:g} W in its top '
    f'{_CHIP_THICKNESS * 1e6:g} um, convection on z_hi; {THERMAL_STRAIN_COUPLING} hands the temperature rise to the '
    'solid; derived T_max, von_mises_max, u_z_max.')


def heated_chip_stress_3d_spec():
    """Template 'heated_chip_stress_3d' (heat + solid_mechanics_3d, stationary): thermal stress of a heated chip on a
    clamped silicon block."""
    low, high = _CHIP_FOOTPRINT
    size_x, size_y, size_z = _CHIP_BLOCK
    return ModelSpec(
        title=HEATED_CHIP_STRESS_LABEL,
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=[_box('chip', low, low, size_z - _CHIP_THICKNESS, high, high, size_z)],
        materials=[MaterialSpec('substrate', library='silicon', selection=['all'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
                     FeatureSpec('sink', 'temperature', ['z_lo'], {'temperature': _CHIP_SINK}),
                     FeatureSpec('air', 'convection', ['z_hi'], {'h': 10.0, 'T_ext': 293.15}),
                     FeatureSpec('chip_power', 'heat_source', ['chip'], {'Q': _CHIP_POWER_DENSITY})]),
                 PhysicsSpec(_SOLID, SOLID_3D_TYPE_NAME, features=[FeatureSpec('base', 'fixed', ['z_lo'])])],
        couplings=[CouplingSpec('thermal_strain', THERMAL_STRAIN_COUPLING, _HEAT, _SOLID)],
        mesh=MeshSpec(mode='uniform', dx=_CHIP_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max'),
                        DerivedValueSpec('von_mises_max', f'{_SOLID}_von_mises', [], 'max'),
                        DerivedValueSpec('u_z_max', f'{_SOLID}_u_z', [], 'max')])


# ---- cases ----
_BUDGET_TEXT = (f'tolerance {_TOLERANCE:g}: the exact field lies in the trilinear element space, so only the CG '
                f'residual (float64, {EXACT_CG_TOLERANCE:g}) and the float32 storage of the fields remain')
# pcg 'mixed' on the 'linear' role: float32 inner CG on the float32 brick copies, refined to the same float64 residual
_MIXED_VARIANTS = (SelfTestVariant('mixed', tolerance=_TOLERANCE, precision='mixed', roles=('linear',)),)
_MIXED_TEXT = ("Variant 'mixed' repeats the run with pcg precision mixed (float32 inner CG on float32 copies of the "
               'brick operator and block inverses, float64 residual refreshes) at the same tolerance.')
SELFTEST_CASES_3D = (
    SelfTestCase(
        name='solid_3d_uniaxial_block_graded', label='Solid mechanics 3D: uniaxial block on a graded mesh',
        description=(f'3D solid mechanics on a {_BLOCK_SIZE[0] * 1e6:g} x {_BLOCK_SIZE[1] * 1e6:g} x '
                     f'{_BLOCK_SIZE[2] * 1e6:g} um block (E {_BLOCK_ELASTIC[0]:g} Pa, nu {_BLOCK_ELASTIC[1]:g}) on a '
                     f'graded mesh: base {_CELL * 1e6:g} um cells (uniform away from the band) with a vertical '
                     f'refinement line at x = L / 2 (dx {_BLOCK_GRADED_CELL * 1e6:g} um, margin '
                     f'{_BLOCK_GRADED_REACH * 1e6:g} um; the line has no z, so every z layer is refined to that width): '
                     'unequal x, y and z cell widths per element. Rollers left / bottom / back on x_lo / y_lo / z_lo, '
                     f'displacement pull u_x = {_BLOCK_PULL:g} m on x_hi, other faces free; stationary. Exact: eps_xx '
                     '= d / L, eps_yy = eps_zz = -nu eps_xx, u_i = eps_ii x_i, sigma_xx = E eps_xx = von Mises, other '
                     'stresses 0, pull reaction sigma_xx H D [N], left -that, bottom / back 0; checked on the '
                     'mid-height cut at the actual centres, and the check raises unless the widest / narrowest x cell '
                     f'>= {_GRADED_MIN_RATIO:g}. {_BUDGET_TEXT}. This pins the full 3D law (both Lame terms, the free z '
                     'face) on equal and graded bricks (the uniform-mesh run of the former solid_3d_uniaxial_block is '
                     f'contained), the z component and the N units of the reactions. {_MIXED_TEXT}'),
        specs=_block_specs, check=_check_block, variants=_MIXED_VARIANTS),
    SelfTestCase(
        name='solid_3d_extruded_bilayer', label='Solid mechanics 3D: extruded bilayer (plane-strain limit)',
        description=(f'The solid_bilayer_tension_graded layers on a {_EXTRUDED_LENGTH * 1e6:g} um strip at the same '
                     f'strain (pull {_EXTRUDED_PULL:g} m), uniform {_CELL * 1e6:g} um cells, extruded to a depth of '
                     f'{_EXTRUDED_DEPTH * 1e6:g} um with rollers on z_lo and z_hi (u_z = 0: plane strain); the '
                     'mid-height cut must reproduce the 2D closed forms (u_z, sigma_xz, sigma_yz = 0; exact at any '
                     'length or depth) and the pull / left reactions are the 2D ones times the depth [N]. '
                     f'{_BUDGET_TEXT}. This pins the z rollers and the plane-strain limit of the 3D law. Not yet run '
                     'on the shorter, thinner strip.'),
        specs=_extruded_bilayer_specs, check=_check_extruded_bilayer),
    SelfTestCase(
        name='solid_3d_column_z_expressions',
        label='Solid mechanics 3D: confined column with z-dependent body force and temperature change',
        description=(f'3D solid mechanics on a {_COLUMN_SIZE[0] * 1e6:g} x {_COLUMN_SIZE[1] * 1e6:g} x '
                     f'{_COLUMN_SIZE[2] * 1e6:g} um column (E {_COLUMN_ELASTIC[0]:g} Pa, nu {_COLUMN_ELASTIC[1]:g}, '
                     f'alpha {_Z_EXPRESSION_ALPHA:g} 1/K), rollers on the four side faces and z_lo, boundary_load t_z = '
                     f'{_COLUMN_TRACTION:g} Pa on z_hi, and feature expressions of z: body_force f_z = '
                     f'{_Z_EXPRESSION_FORCE_TOP:g} z / H N/m^3 on the upper half only (a box) and temperature_change '
                     f'delta_T = {_Z_EXPRESSION_RISE_TOP:g} z / H K on all. Expected = the '
                     'discrete 1D solution with the lumped cell loads (sigma_c = sum of the nodal loads above cell c, '
                     'eps_c = (sigma_c + beta dT_c) / (lambda + 2 mu), beta = (3 lambda + 2 mu) alpha), exact in the '
                     'trilinear space; probes u_z / stress_zz / stress_xx at z cells '
                     f'{", ".join(str(index) for index in _Z_EXPRESSION_PROBE_CELLS)}, the mid-height cut incl. '
                     'strain_zz = eps_c, von Mises = |sigma_zz - sigma_xx|, temperature_rise_ext and the base reaction. '
                     f'{_BUDGET_TEXT}. This pins the z traction (tributary areas), the lumped body force, 3D probes, '
                     'the z reaction and the z-dependent eval_field values (BoxValues on a partial and on the full box) '
                     'of body_force and temperature_change (the former solid_3d_confined_column, a constant body force '
                     f'on all, is contained). {_MIXED_TEXT} Not yet run with the strain_zz / von Mises items or the '
                     'mixed variant.'),
        specs=_z_expression_specs, check=_check_z_expressions, variants=_MIXED_VARIANTS),
    SelfTestCase(
        name='heat_solid_thermal_expansion_3d', label='Heat -> solid mechanics 3D: heated box, free and confined',
        description=(f'Heat + 3D solid mechanics on a {_THERMAL_SIZE[0] * 1e6:g} x {_THERMAL_SIZE[1] * 1e6:g} x '
                     f'{_THERMAL_SIZE[2] * 1e6:g} um box (k 10, E {_THERMAL_ELASTIC[0]:g} Pa, nu '
                     f'{_THERMAL_ELASTIC[1]:g}, alpha_xx / yy / zz = {_THERMAL_ALPHA[0]:g} / {_THERMAL_ALPHA[1]:g} / '
                     f'{_THERMAL_ALPHA[2]:g} 1/K), heat holding all six faces at T0 + {_THERMAL_RISE:g} K, '
                     f'{THERMAL_STRAIN_COUPLING}. Run free (rollers on x_lo, y_lo, z_lo): u_i = alpha_ii dT x_i, '
                     'strain_zz = alpha_zz dT, zero stress, von Mises and reactions. Run confined (rollers on all six '
                     'faces): u = 0, strain_zz = 0, sigma_ii = -beta_i dT with beta_i = 2 mu alpha_ii + lambda '
                     '(alpha_xx + alpha_yy + alpha_zz), von Mises of these normal stresses, elastic_strain_zz = '
                     f'-alpha_zz dT, face reactions sigma_ii times the face area [N]. {_BUDGET_TEXT}. This pins beta_x '
                     '/ y / z, the 3D coupling buffer and alpha_zz as an ordinary strain. Not yet run with the '
                     'strain_zz / von Mises items.'),
        specs=_thermal_specs, check=_check_thermal),
    SelfTestCase(
        name='solid_3d_rigid_mode_check', label='Solid mechanics 3D: rigid-mode rank check',
        description=('ElasticSystem3D assembled directly (no model run) on a 10 x 4 x 4 um box: (a) u_x held on y_lo, '
                     'u_y on x_lo, u_z on z_lo holds every direction but leaves the rotation about z free, so the '
                     'assembly must raise naming that rotation; (b) two blocks split by a void gap with rollers on '
                     'x_lo, y_lo, z_lo: the far block must be named (slide along x, its first cell); (c) the same '
                     'rollers on one block assemble. error = number of scenarios whose outcome differs, tolerance 0. '
                     'This pins the 6 x 6 Gram rank test that validation\'s direction check cannot do. Runs on the cpu '
                     'backend (numpy kernel twins, scipy labels) and, with a usable GPU, on the gpu one as well.'),
        specs=dict, check=_check_rigid_modes, backends=('cpu', 'gpu')),
)

"""solid_mechanics_3d on tetrahedron meshes (SolidMechanicsFEMPhysics3D, builtin Kuhn 'tetrahedra' meshes, no gmsh):
uniaxial block (P1 and P2, exact), confined column (P2, exact quadratic), a Fixed support on an object's interface
faces, heat -> solid thermal expansion on the same mesh, an anisotropic (rotated crystal / partial c_ij) block under a
uniform strain and its free thermal expansion (exact), roller-box vibration modes against the continuum (isotropic
lowest and shifted, orthotropic), the rigid-mode rank check (CPU, no model) and the cuDSS eigen factorisations against
SuperLU (GPU with nvmath). Model runs read probes, support reactions and the resonance table. definition3d registers
them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import lithium_niobate_entry, worst_error
from ...model.spec import (CouplingSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, SolverDefaultsSpec, StudySpec)
from .selftests import _elastic, _lame

_SOLID_3D_TYPE = 'solid_mechanics_3d'   # = selftests_3d.SOLID_3D_TYPE_NAME
_THERMAL_STRAIN = 'heat_solid_thermal_strain'
_SOLID = 'solid'
_HEAT = 'heat'
UM = 1e-6
_ELASTIC = (100e9, 0.3)
# float32 probe samples (~6e-8 relative) and exact solves: the exact fields lie in the element space (first estimates)
_EXACT_TOLERANCE = 1e-6


def _box(name, x0, y0, z0, x1, y1, z1):
    return GeometrySpec(name, 'box', {'x0': x0, 'y0': y0, 'z0': z0, 'x1': x1, 'y1': y1, 'z1': z1})


def _point(name, x, y, z):
    return GeometrySpec(name, 'point', {'x': x, 'y': y, 'z': z})


def _probes(point, components):
    return [FeatureSpec(f'{component}_{point}', 'solid_probe', [point], {'component': component})
            for component in components]


def _tetrahedra(cell):
    return [MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=cell, mesher='builtin')]


def _solid_spec(title, size, materials, features, geometry=(), discretization='quadratic', cell=0.25 * UM,
                study=None, settings=None):
    """solid_mechanics_3d-only model on a size = (x, y, z) box meshed by builtin tetrahedra."""
    size_x, size_y, size_z = size
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_SOLID, _SOLID_3D_TYPE, settings={'discretization': discretization, **(settings or {})},
                             features=list(features))],
        meshes=_tetrahedra(cell),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=study or StudySpec(type='stationary'))


def _probe(results, feature):
    """Last sample of a solid_probe."""
    values = np.asarray(results.result(_SOLID, feature)['values'], dtype=np.float64)
    return float(values[-1]) if values.size else math.nan


def _reaction(results, feature):
    table = results.result(_SOLID, feature)
    return float(table['F_x']), float(table['F_y']), float(table['F_z'])


# ---- solid_fem3d_uniaxial_block: rollers x_lo / y_lo / z_lo, pull u_x = d on x_hi (exact linear field) ----
_BLOCK_SIZE = (1.0 * UM, 0.5 * UM, 0.5 * UM)
_BLOCK_PULL = 1e-9                                  # m: strain 1e-3
_BLOCK_POINT = (0.6 * UM, 0.3 * UM, 0.2 * UM)       # inside an element (not on a node)
_BLOCK_COMPONENTS = ('u_x', 'u_y', 'u_z', 'stress_xx', 'stress_yy', 'stress_xz', 'von_mises')


def _block_spec(discretization):
    return _solid_spec(
        f'Self-test: uniaxial block on tetrahedra ({discretization})', _BLOCK_SIZE,
        [MaterialSpec('block', properties=_elastic(*_ELASTIC), selection=['all'])],
        [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
         FeatureSpec('back', 'roller', ['z_lo']), FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _BLOCK_PULL}),
         *_probes('p', _BLOCK_COMPONENTS)],
        [_point('p', *_BLOCK_POINT)], discretization)


def _check_block(results_by_name):
    """Probes against u_i = eps_ii x_i, sigma_xx = E eps = von Mises, other stresses 0; reactions F = E eps H D."""
    youngs, nu = _ELASTIC
    length, height, depth = _BLOCK_SIZE
    strain = _BLOCK_PULL / length
    stress = youngs * strain
    force = stress * height * depth
    x, y, z = _BLOCK_POINT
    expected = {'u_x': strain * x, 'u_y': -nu * strain * y, 'u_z': -nu * strain * z, 'stress_xx': stress,
                'stress_yy': 0.0, 'stress_xz': 0.0, 'von_mises': stress}
    errors = {}
    for run, results in results_by_name.items():
        for name, value in expected.items():
            scale = _BLOCK_PULL if name.startswith('u_') else stress
            errors[f'{run}.{name}'] = abs(_probe(results, f'{name}_p') - value) / scale
        pull, left, bottom, back = (_reaction(results, feature) for feature in ('pull', 'left', 'bottom', 'back'))
        errors.update({f'{run}.F_pull': abs(pull[0] - force) / force, f'{run}.F_left': abs(left[0] + force) / force,
                       f'{run}.F_bottom': abs(bottom[1]) / force, f'{run}.F_back': abs(back[2]) / force})
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'force_N': force, 'strain_xx': strain}}


# ---- solid_fem3d_confined_column (P2): rollers on the sides and z_lo, top traction, gravity (quadratic u_z) ----
_COLUMN_SIZE = (0.5 * UM, 0.5 * UM, 2.0 * UM)
_COLUMN_TRACTION = -1e6                             # Pa on z_hi
_COLUMN_BODY_FORCE = -1e11                          # N/m^3 along z
_COLUMN_HEIGHTS = (0.3 * UM, 0.9 * UM, 1.7 * UM)    # probe heights on the line x = 0.2, y = 0.3 um


def _column_displacement(z):
    lame_lambda, mu = _lame(*_ELASTIC)
    height = _COLUMN_SIZE[2]
    return (_COLUMN_TRACTION * z + _COLUMN_BODY_FORCE * (height * z - 0.5 * z * z)) / (lame_lambda + 2.0 * mu)


def _column_stress_zz(z):
    return _COLUMN_TRACTION + _COLUMN_BODY_FORCE * (_COLUMN_SIZE[2] - z)


def _column_spec():
    points = [_point(f'c{index}', 0.2 * UM, 0.3 * UM, height) for index, height in enumerate(_COLUMN_HEIGHTS)]
    probes = [probe for index in range(len(_COLUMN_HEIGHTS))
              for probe in _probes(f'c{index}', ('u_z', 'stress_zz', 'stress_xx'))]
    return _solid_spec(
        'Self-test: confined column on tetrahedra (quadratic)', _COLUMN_SIZE,
        [MaterialSpec('column', properties=_elastic(*_ELASTIC), selection=['all'])],
        [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']), FeatureSpec('base', 'roller', ['z_lo']),
         FeatureSpec('top', 'boundary_load', ['z_hi'], {'t_x': 0.0, 't_y': 0.0, 't_z': _COLUMN_TRACTION}),
         FeatureSpec('gravity', 'body_force', ['all'], {'f_x': 0.0, 'f_y': 0.0, 'f_z': _COLUMN_BODY_FORCE}),
         *probes], points)


def _check_column(results_by_name):
    """Probes u_z, sigma_zz, sigma_xx = lambda / M sigma_zz at three heights; base reaction -(t + f H) W^2."""
    results = results_by_name['run']
    lame_lambda, mu = _lame(*_ELASTIC)
    lateral_ratio = lame_lambda / (lame_lambda + 2.0 * mu)
    width, _, height = _COLUMN_SIZE
    displacement_scale = abs(_column_displacement(height))
    stress_scale = abs(_column_stress_zz(0.0))
    errors = {}
    for index, z in enumerate(_COLUMN_HEIGHTS):
        errors[f'u_z_{index}'] = abs(_probe(results, f'u_z_c{index}') - _column_displacement(z)) / displacement_scale
        errors[f'stress_zz_{index}'] = abs(_probe(results, f'stress_zz_c{index}') - _column_stress_zz(z)) / stress_scale
        errors[f'stress_xx_{index}'] = abs(_probe(results, f'stress_xx_c{index}')
                                           - lateral_ratio * _column_stress_zz(z)) / stress_scale
    base_force = -(_COLUMN_TRACTION + _COLUMN_BODY_FORCE * height) * width * width
    base, sides = _reaction(results, 'base'), _reaction(results, 'sides')
    errors.update({'F_base': abs(base[2] - base_force) / abs(base_force),
                   'F_sides_x': abs(sides[0]) / abs(base_force), 'F_sides_y': abs(sides[1]) / abs(base_force)})
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'F_base': list(base), 'expected_F_base_N': base_force}}


# ---- solid_fem3d_object_support: Fixed on the interface faces of a foot box (z < h), top traction ----
_FOOT_SIZE = (0.5 * UM, 0.5 * UM, 1.0 * UM)
_FOOT_HEIGHT = 0.25 * UM                            # one Kuhn cell layer: every foot node lies on its surface
_FOOT_TRACTION = -2e6                               # Pa on z_hi
_FOOT_POINT = (0.2 * UM, 0.3 * UM, 0.7 * UM)


def _foot_spec(discretization):
    width, depth, height = _FOOT_SIZE
    return _solid_spec(
        f'Self-test: Fixed on object interface faces, tetrahedra ({discretization})', _FOOT_SIZE,
        [MaterialSpec('column', properties=_elastic(*_ELASTIC), selection=['all'])],
        [FeatureSpec('foot_fixed', 'fixed', ['foot']), FeatureSpec('sides', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
         FeatureSpec('top', 'boundary_load', ['z_hi'], {'t_x': 0.0, 't_y': 0.0, 't_z': _FOOT_TRACTION}),
         *_probes('q', ('u_z', 'stress_zz', 'stress_yy'))],
        [_box('foot', 0.0, 0.0, 0.0, width, depth, _FOOT_HEIGHT), _point('q', *_FOOT_POINT)], discretization)


def _check_foot(results_by_name):
    """Above the held foot: u_z = t (z - h) / M, sigma_zz = t, sigma_yy = lambda / M t; foot reaction F_z = -t W D."""
    lame_lambda, mu = _lame(*_ELASTIC)
    modulus = lame_lambda + 2.0 * mu
    width, depth, height = _FOOT_SIZE
    z = _FOOT_POINT[2]
    expected = {'u_z': _FOOT_TRACTION * (z - _FOOT_HEIGHT) / modulus, 'stress_zz': _FOOT_TRACTION,
                'stress_yy': lame_lambda / modulus * _FOOT_TRACTION}
    displacement_scale = abs(_FOOT_TRACTION * (height - _FOOT_HEIGHT) / modulus)
    force = -_FOOT_TRACTION * width * depth
    errors = {}
    for run, results in results_by_name.items():
        for name, value in expected.items():
            scale = displacement_scale if name == 'u_z' else abs(_FOOT_TRACTION)
            errors[f'{run}.{name}'] = abs(_probe(results, f'{name}_q') - value) / scale
        foot = _reaction(results, 'foot_fixed')
        errors[f'{run}.F_foot_z'] = abs(foot[2] - force) / abs(force)
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'expected_F_foot_z_N': force}}


# ---- heat_solid_thermal_expansion_fem3d: heat and solid on one tetrahedra mesh, uniform rise ----
_THERMAL_SIZE = (0.75 * UM, 0.5 * UM, 0.5 * UM)
_THERMAL_ALPHA = (1e-5, 2e-5, 0.5e-5)
_THERMAL_T0 = 293.15
_THERMAL_RISE = 100.0
_THERMAL_POINT = (0.4 * UM, 0.3 * UM, 0.2 * UM)
_THERMAL_RUNS = {'free': ('low',), 'confined': ('low', 'high')}
_THERMAL_FACES = {'low': ['x_lo', 'y_lo', 'z_lo'], 'high': ['x_hi', 'y_hi', 'z_hi']}
_THERMAL_COMPONENTS = ('u_x', 'u_y', 'u_z', 'stress_xx', 'stress_yy', 'stress_zz', 'elastic_strain_zz')


def _thermal_specs():
    alpha_xx, alpha_yy, alpha_zz = _THERMAL_ALPHA
    size_x, size_y, size_z = _THERMAL_SIZE
    faces = ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi']
    specs = {}
    for run, supports in _THERMAL_RUNS.items():
        specs[run] = ModelSpec(
            title=f'Self-test: heated box on tetrahedra, {run} (heat -> solid mechanics 3D)',
            domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
            geometry=[_point('p', *_THERMAL_POINT)],
            materials=[MaterialSpec('box', properties=_elastic(*_ELASTIC, k=10.0, alpha_xx=alpha_xx,
                                                               alpha_yy=alpha_yy, alpha_zz=alpha_zz),
                                    selection=['all'])],
            physics=[PhysicsSpec(_HEAT, 'heat', settings={'T0': _THERMAL_T0}, features=[
                         FeatureSpec('walls', 'temperature', faces, {'temperature': _THERMAL_T0 + _THERMAL_RISE})]),
                     PhysicsSpec(_SOLID, _SOLID_3D_TYPE, features=[
                         *(FeatureSpec(name, 'roller', list(_THERMAL_FACES[name])) for name in supports),
                         *_probes('p', _THERMAL_COMPONENTS)])],
            couplings=[CouplingSpec('expansion', _THERMAL_STRAIN, _HEAT, _SOLID)],
            meshes=_tetrahedra(0.25 * UM),
            solver_defaults=SolverDefaultsSpec(backend='cpu'),
            study=StudySpec(type='stationary'))
    return specs


def _check_thermal(results_by_name):
    """free: u_i = alpha_ii dT x_i, zero stress; confined: u = 0, sigma_ii = -beta_i dT, elastic_strain_zz = -alpha_zz
    dT, face reactions sigma_ii A_i."""
    lame_lambda, mu = _lame(*_ELASTIC)
    alpha_sum = sum(_THERMAL_ALPHA)
    beta = [(2.0 * mu * alpha + lame_lambda * alpha_sum) * _THERMAL_RISE for alpha in _THERMAL_ALPHA]
    size_x, size_y, size_z = _THERMAL_SIZE
    areas = (size_y * size_z, size_x * size_z, size_x * size_y)
    stress_scale = _ELASTIC[0] * max(_THERMAL_ALPHA) * _THERMAL_RISE
    displacement_scale = max(_THERMAL_ALPHA) * _THERMAL_RISE * max(_THERMAL_SIZE)
    strain_scale = max(_THERMAL_ALPHA) * _THERMAL_RISE
    errors = {}
    for run, results in results_by_name.items():
        if run == 'free':
            expected = {f'u_{axis}': alpha * _THERMAL_RISE * coordinate
                        for axis, alpha, coordinate in zip('xyz', _THERMAL_ALPHA, _THERMAL_POINT)}
            expected.update({'stress_xx': 0.0, 'stress_yy': 0.0, 'stress_zz': 0.0, 'elastic_strain_zz': 0.0})
        else:
            expected = {'u_x': 0.0, 'u_y': 0.0, 'u_z': 0.0, 'stress_xx': -beta[0], 'stress_yy': -beta[1],
                        'stress_zz': -beta[2], 'elastic_strain_zz': -_THERMAL_ALPHA[2] * _THERMAL_RISE}
            high = _reaction(results, 'high')
            for axis in range(3):
                errors[f'{run}.F_high_{"xyz"[axis]}'] = abs(high[axis] + beta[axis] * areas[axis]) / (
                    stress_scale * max(areas))
        for name, value in expected.items():
            scale = (displacement_scale if name.startswith('u_') else strain_scale if 'strain' in name
                     else stress_scale)
            errors[f'{run}.{name}'] = abs(_probe(results, f'{name}_p') - value) / scale
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'beta_dT_Pa': beta}}


# ---- solid_fem3d_anisotropic_uniform_strain: rollers on the low faces, tractions T n on all six faces ----
# The exact field u = (s_1 x, s_2 y, s_3 z) (diagonal strain: the low-face rollers allow no shear strain) has the uniform
# stress T = c S, whose shear rows are non-zero for a rotated crystal; with T n applied on every face (the held normal
# parts too) K u = f holds on every dof, so the reactions are 0. Thermal run: free expansion u_i = alpha_ii dT x_i with
# zero stress needs the shear rows of beta = c alpha in the load.
_ANISO_SIZE = (0.75 * UM, 0.5 * UM, 0.5 * UM)
_ANISO_STRAIN = (1e-3, -0.4e-3, 0.7e-3)            # S_xx, S_yy, S_zz
_ANISO_EULER = [31.0, 47.0, 73.0]                  # deg: no symmetry left, all 21 c_IJ non-zero
_ANISO_ALPHA = (1.5e-5, 0.4e-5, 0.8e-5)            # alpha_xx, alpha_yy, alpha_zz [1/K]
_ANISO_RISE = 100.0                                # K
_ANISO_PARTIAL = {'c11': 180e9, 'c14': 10e9, 'c44': 30e9}   # over E / nu = _ELASTIC: the other entries fall back
_ANISO_POINT = (0.4 * UM, 0.3 * UM, 0.2 * UM)
_ANISO_STRESSES = ('stress_xx', 'stress_yy', 'stress_zz', 'stress_yz', 'stress_xz', 'stress_xy')
_ANISO_COMPONENTS = ('u_x', 'u_y', 'u_z', *_ANISO_STRESSES)
_ANISO_ROLLERS = (('x_roller', 'x_lo'), ('y_roller', 'y_lo'), ('z_roller', 'z_lo'))
_ANISO_RUNS = {'crystal_quadratic': ('crystal', 'quadratic'), 'crystal_linear': ('crystal', 'linear'),
               'partial': ('partial', 'quadratic'), 'thermal': ('thermal', 'quadratic')}


def _aniso_stiffness(material):
    """(6, 6) Voigt c [Pa] of run material 'crystal' / 'thermal' (the rotated crystal) or 'partial' (Lame + overrides)."""
    if material != 'partial':
        from ...materials.crystal_tensors import full_crystal
        return full_crystal(lithium_niobate_entry(_ANISO_EULER)).stiffness
    lame_lambda, mu = _lame(*_ELASTIC)
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = lame_lambda
    stiffness[np.arange(3), np.arange(3)] += 2.0 * mu
    stiffness[np.arange(3, 6), np.arange(3, 6)] = mu
    for name, value in _ANISO_PARTIAL.items():
        row, column = int(name[1]) - 1, int(name[2]) - 1
        stiffness[row, column] = stiffness[column, row] = value
    return stiffness


def _aniso_stress(material):
    """Voigt (6,) T = c S of the uniform strain."""
    return _aniso_stiffness(material) @ np.array([*_ANISO_STRAIN, 0.0, 0.0, 0.0])


def _aniso_spec(run):
    material, discretization = _ANISO_RUNS[run]
    features = [FeatureSpec(name, 'roller', [face]) for name, face in _ANISO_ROLLERS]
    if material == 'thermal':
        properties = dict(zip(('alpha_xx', 'alpha_yy', 'alpha_zz'), _ANISO_ALPHA))
        features.append(FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _ANISO_RISE}))
    else:
        properties = _elastic(*_ELASTIC, **_ANISO_PARTIAL) if material == 'partial' else {}
        stress = _aniso_stress(material)
        tensor = np.array([[stress[0], stress[5], stress[4]], [stress[5], stress[1], stress[3]],
                           [stress[4], stress[3], stress[2]]])
        for axis, letter in enumerate('xyz'):
            for side, sign in (('lo', -1.0), ('hi', 1.0)):
                traction = sign * tensor[:, axis]
                features.append(FeatureSpec(f'load_{letter}_{side}', 'boundary_load', [f'{letter}_{side}'],
                                            {'t_x': float(traction[0]), 't_y': float(traction[1]),
                                             't_z': float(traction[2])}))
    dispersion = None if material == 'partial' else [lithium_niobate_entry(_ANISO_EULER)]
    return _solid_spec(
        f'Self-test: anisotropic block on tetrahedra ({run})', _ANISO_SIZE,
        [MaterialSpec('block', properties=properties, dispersion=dispersion, selection=['all'])],
        [*features, *_probes('p', _ANISO_COMPONENTS)], [_point('p', *_ANISO_POINT)], discretization)


def _check_aniso(results_by_name):
    """Probes u and the six stresses against the uniform field, the three roller reactions against 0."""
    x, y, z = _ANISO_POINT
    width, depth, height = _ANISO_SIZE
    errors, stress_scales = {}, {}
    for run, results in results_by_name.items():
        material, _ = _ANISO_RUNS[run]
        if material == 'thermal':
            strain = [alpha * _ANISO_RISE for alpha in _ANISO_ALPHA]
            stress = np.zeros(6)
            scale = float(np.max(np.abs(_aniso_stiffness(material)[:, :3] @ np.array(strain))))   # |beta dT|
        else:
            strain = list(_ANISO_STRAIN)
            stress = _aniso_stress(material)
            scale = float(np.max(np.abs(stress)))
        stress_scales[run] = scale
        displacement_scale = max(abs(value) for value in strain) * max(_ANISO_SIZE)
        expected = {'u_x': strain[0] * x, 'u_y': strain[1] * y, 'u_z': strain[2] * z,
                    **dict(zip(_ANISO_STRESSES, stress))}
        for name, value in expected.items():
            errors[f'{run}.{name}'] = abs(_probe(results, f'{name}_p') - value) / (
                displacement_scale if name.startswith('u_') else scale)
        force_scale = scale * max(width * depth, width * height, depth * height)
        for name, _ in _ANISO_ROLLERS:
            errors[f'{run}.F_{name}'] = max(abs(value) for value in _reaction(results, name)) / force_scale
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _EXACT_TOLERANCE,
            'details': {'errors': errors, 'stress_scales_Pa': stress_scales}}


# ---- solid_fem3d_roller_box_modes: P2 modes of a roller box against the continuum ----
# Rollers on all six faces: u_a = A_a sin(k_a x_a) prod_{d != a} cos(k_d x_d), k = (p pi / L_x, q pi / L_y, r pi / L_z),
# component a present for p_a > 0; per triple the 3 x 3 orthotropic symbol (axes along x, y, z) Gamma_xx = c11 k_x^2 +
# c66 k_y^2 + c55 k_z^2, Gamma_yy = c66 k_x^2 + c22 k_y^2 + c44 k_z^2, Gamma_zz = c55 k_x^2 + c44 k_y^2 + c33 k_z^2,
# Gamma_xy = (c12 + c66) k_x k_y, Gamma_xz = (c13 + c55) k_x k_z, Gamma_yz = (c23 + c44) k_y k_z (isotropic: longitudinal
# c_L |k| and transverse c_T |k|). Sizes 1 x 0.625 x 0.375 um keep the 6 lowest isotropic modes apart (L100, T110,
# T210, L010, T101, T011). P2 on h = 62.5 nm: eigenvalue error ~(k h)^4 / 1e2 (estimate).
_MODES_SIZE = (1.0 * UM, 0.625 * UM, 0.375 * UM)
_MODES_CELL = 0.0625 * UM
_MODES_ELASTIC = (70e9, 0.25)
_MODES_DENSITY = 2200.0
_MODES_LOWEST = 6
_MODES_SHIFTED = 3
_MODES_SHIFT = (3, 1.03)                            # target 3 % above the 4th lowest continuum mode
_MODES_TOLERANCE = 1e-3                             # relative frequency (first estimate)
_TABLE = 'modes'
# orthotropic run: c_ij only (no youngs_modulus: solid by c11 > 0), the other twelve entries 0 by default
_ORTHOTROPIC = {'c11': 200e9, 'c22': 160e9, 'c33': 120e9, 'c12': 60e9, 'c13': 50e9, 'c23': 40e9, 'c44': 45e9,
                'c55': 55e9, 'c66': 65e9}
_ORTHOTROPIC_DENSITY = 2330.0


def _isotropic_constants():
    """The nine orthotropic-pattern c_ij [Pa] of the isotropic _MODES_ELASTIC."""
    lame_lambda, mu = _lame(*_MODES_ELASTIC)
    return {**{name: lame_lambda + 2.0 * mu for name in ('c11', 'c22', 'c33')},
            **{name: lame_lambda for name in ('c12', 'c13', 'c23')}, **{name: mu for name in ('c44', 'c55', 'c66')}}


def _continuum_modes(constants=None, density=_MODES_DENSITY, orders=4):
    """Sorted continuum frequencies [Hz] of the roller box of orthotropic constants (default isotropic _MODES_ELASTIC),
    every order triple below `orders`."""
    c = constants or _isotropic_constants()
    shear = np.array([[0.0, c['c66'], c['c55']], [c['c66'], 0.0, c['c44']], [c['c55'], c['c44'], 0.0]])
    normal = np.array([c['c11'], c['c22'], c['c33']])
    coupling = np.array([[0.0, c['c12'] + c['c66'], c['c13'] + c['c55']],
                         [c['c12'] + c['c66'], 0.0, c['c23'] + c['c44']],
                         [c['c13'] + c['c55'], c['c23'] + c['c44'], 0.0]])
    frequencies = []
    for triple in np.ndindex(orders, orders, orders):
        k = np.array([order * math.pi / side for order, side in zip(triple, _MODES_SIZE)])
        present = [axis for axis in range(3) if triple[axis] > 0]
        if not present:
            continue
        symbol = coupling * np.outer(k, k) + np.diag(normal * k ** 2 + shear @ k ** 2)
        values = np.linalg.eigvalsh(symbol[np.ix_(present, present)]) / density
        frequencies.extend(math.sqrt(value) / (2.0 * math.pi) for value in values)
    return sorted(frequencies)


def _modes_target():
    index, factor = _MODES_SHIFT
    return factor * _continuum_modes()[index]


def _modes_spec(target, count, properties=None):
    return _solid_spec(
        'Self-test: roller-box vibration modes on tetrahedra (quadratic)', _MODES_SIZE,
        [MaterialSpec('block', properties=properties or _elastic(*_MODES_ELASTIC, rho=_MODES_DENSITY),
                      selection=['all'])],
        [FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi']),
         FeatureSpec(_TABLE, 'resonance_table')], cell=_MODES_CELL,
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': count}),
        settings={'target_frequency': target})


def _modes_specs():
    return {'lowest': _modes_spec(0.0, _MODES_LOWEST), 'shifted': _modes_spec(_modes_target(), _MODES_SHIFTED),
            'orthotropic': _modes_spec(0.0, _MODES_LOWEST, {**_ORTHOTROPIC, 'rho': _ORTHOTROPIC_DENSITY})}


def _check_modes(results_by_name):
    """Table frequencies in rank order against the continuum (shifted: ranked by |f - f_target|; orthotropic: the c_ij
    block's own continuum)."""
    continuum = _continuum_modes()
    target = _modes_target()
    expected = {'lowest': continuum[:_MODES_LOWEST],
                'shifted': sorted(continuum, key=lambda value: abs(value - target))[:_MODES_SHIFTED],
                'orthotropic': _continuum_modes(_ORTHOTROPIC, _ORTHOTROPIC_DENSITY)[:_MODES_LOWEST]}
    errors, measured = {}, {}
    for run, references in expected.items():
        if run not in results_by_name:
            continue
        rows = results_by_name[run].result(_SOLID, _TABLE)['rows']
        measured[run] = [float(row[1]) for row in rows]
        if len(rows) < len(references):
            errors[run] = math.inf
            continue
        errors[run] = max(abs(value / reference - 1.0) for value, reference in zip(measured[run], references))
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _MODES_TOLERANCE,
            'details': {'errors': errors, 'measured_Hz': measured, 'continuum_Hz': expected,
                        'target_frequency_shifted': target}}


# ---- solid_fem3d_rigid_mode_check: the Gram rank test of ElasticFEMSystem3D (no model run) ----
def _check_rigid_modes(_):
    """Scenarios on a 4 x 2 x 2 Kuhn box: (a) u_x on y_lo, u_y on x_lo, u_z on z_lo: rotation about z free; (b) two
    blocks split by a void gap, rollers on the low faces: the far block slides along x; (c) rollers on one block:
    assembles."""
    from ...fem.elasticity3d import isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from .system_fem3d import ElasticFEMSystem3D
    edges = np.linspace(0.0, 4.0 * UM, 5)
    mesh = structured_box_mesh(edges, edges[:3], edges[:3])
    lame_lambda, mu = _lame(*_ELASTIC)
    centroids = mesh.centroids()
    gap = (centroids[:, 0] > 2.0 * UM) & (centroids[:, 0] < 3.0 * UM)
    rollers = [('x_lo', 'u_x'), ('y_lo', 'u_y'), ('z_lo', 'u_z')]
    scenarios = {'rotation_about_z': ([('y_lo', 'u_x'), ('x_lo', 'u_y'), ('z_lo', 'u_z')], False,
                                      ('rotation about the axis along z',)),
                 'unsupported_block': (rollers, True, ('slide along x',)),
                 'held_block': (rollers, False, None)}
    outcomes, failures = {}, 0
    for name, (holds, split, parts) in scenarios.items():
        for order in (1, 2):
            system = ElasticFEMSystem3D(mesh, order, lambda: 'rigid-mode self-test')
            for face, component in holds:
                system.hold(system.full.face_dofs(system.face_faces(face)), **{component: 0.0})
            solid = ~gap if split else np.ones(mesh.n_elements, dtype=bool)
            try:
                system.assemble(isotropic_matrix_3d(np.where(solid, lame_lambda, 0.0), np.where(solid, mu, 0.0)), solid)
                message = None
            except ValueError as exc:
                message = str(exc)
            passed = message is None if parts is None else message is not None and all(p in message for p in parts)
            failures += 0 if passed else 1
            outcomes[f'{name}_p{order}'] = {'passed': passed, 'message': message, 'expected_parts': parts}
    return {'measured': len(outcomes) - failures, 'expected': len(outcomes), 'error': float(failures),
            'tolerance': 0.0, 'details': outcomes}


# ---- solid_fem3d_modes_cudss: cuDSS LDL^T (double, mixed) against host SuperLU on the same pencil (GPU) ----
_CUDSS_EIGEN_TOLERANCE = 1e-9   # relative eigenvalue difference (both refined to full accuracy; first estimate)


def _check_cudss(_):
    """The roller-box pencil (P2, 62.5 nm Kuhn cells) by eigsh_shifted at 0 with factorization superlu, cudss and
    cudss_mixed; error = largest relative eigenvalue difference to SuperLU. Passes noted without nvmath-python."""
    from ...fem.elasticity3d import isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from ...solvers.cudss_factor import cudss_available
    from .system_fem3d import ElasticFEMSystem3D
    if not cudss_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _CUDSS_EIGEN_TOLERANCE,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    axes = [np.linspace(0.0, side, int(round(side / _MODES_CELL)) + 1) for side in _MODES_SIZE]
    mesh = structured_box_mesh(*axes)
    system = ElasticFEMSystem3D(mesh, 2, lambda: 'cuDSS self-test')
    for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi'):
        system.hold(system.full.face_dofs(system.face_faces(face)), **{f'u_{face[0]}': 0.0})
    lame_lambda, mu = _lame(*_MODES_ELASTIC)
    solid = np.ones(mesh.n_elements, dtype=bool)
    system.assemble(isotropic_matrix_3d(np.full(mesh.n_elements, lame_lambda), np.full(mesh.n_elements, mu)), solid)
    free = np.flatnonzero(system.free)
    stiffness = system.operator[free][:, free].tocsc()
    mass = system.mass(np.full(mesh.n_elements, _MODES_DENSITY))[free][:, free].tocsc()
    start = np.random.default_rng(6).standard_normal(free.size)
    eigenvalues, details = {}, {'free_dofs': int(free.size)}
    for factorization in ('superlu', 'cudss', 'cudss_mixed'):
        solver = ArpackShiftInvertSolver(factorization=factorization)
        values, _, applications, converged, factor = solver.eigsh_shifted(stiffness, mass, 0.0, _MODES_LOWEST,
                                                                          start_vector=start)
        eigenvalues[factorization] = np.sort(values)
        details[factorization] = {
            'converged': bool(converged), 'applications': int(applications),
            'factor': getattr(factor, 'description', 'SuperLU'), 'host_bytes': int(factor.host_bytes or 0),
            **{name: getattr(factor, name) for name in ('device_bytes', 'plan_seconds', 'factorize_seconds',
                                                        'factor_seconds', 'single_residual', 'probe_residual',
                                                        'refinement_steps', 'symmetric', 'solve_seconds', 'n_solves')
               if hasattr(factor, name)}}
        if hasattr(factor, 'release'):
            factor.release()
    reference = eigenvalues['superlu']
    errors = {name: float(np.max(np.abs(values / reference - 1.0))) for name, values in eigenvalues.items()
              if name != 'superlu'}
    details['errors'] = errors
    details['frequencies_Hz'] = (np.sqrt(reference) / (2.0 * math.pi)).tolist()
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _CUDSS_EIGEN_TOLERANCE,
            'details': details}


FEM_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_uniaxial_block', label='Solid mechanics 3D on tetrahedra: uniaxial block',
        description=(f'solid_mechanics_3d on a builtin tetrahedra mesh (Kuhn cubes of 0.25 um, no gmsh) of a '
                     f'{_BLOCK_SIZE[0] / UM:g} x {_BLOCK_SIZE[1] / UM:g} x {_BLOCK_SIZE[2] / UM:g} um block (E '
                     f'{_ELASTIC[0]:g} Pa, nu {_ELASTIC[1]:g}), rollers on x_lo / y_lo / z_lo, displacement u_x = '
                     f'{_BLOCK_PULL:g} m on x_hi; quadratic and linear. Exact linear field: probes at an interior '
                     'point (u_i = eps_ii x_i, sigma_xx = E eps = von Mises, sigma_yy = sigma_xz = 0) and the four '
                     f'reactions (pull E eps H D [N], left -that, bottom / back 0). Tolerance {_EXACT_TOLERANCE:g} '
                     '(float32 probes; first estimate, not yet run).'),
        specs=lambda: {order: _block_spec(order) for order in ('quadratic', 'linear')}, check=_check_block),
    SelfTestCase(
        name='solid_fem3d_confined_column', label='Solid mechanics 3D on tetrahedra: confined column',
        description=(f'Quadratic tetrahedra (0.25 um Kuhn cubes) on a {_COLUMN_SIZE[0] / UM:g} x '
                     f'{_COLUMN_SIZE[1] / UM:g} x {_COLUMN_SIZE[2] / UM:g} um column: rollers on the four sides and '
                     f'z_lo, traction t_z = {_COLUMN_TRACTION:g} Pa on z_hi, body force f_z = {_COLUMN_BODY_FORCE:g} '
                     'N/m^3. P2 holds the quadratic u_z = (t z + f (H z - z^2 / 2)) / (lambda + 2 mu) exactly: probes '
                     'u_z, sigma_zz = t + f (H - z) and sigma_xx = lambda / (lambda + 2 mu) sigma_zz at three heights '
                     '(node-averaged stresses of an exact linear stress field), base reaction -(t + f H) W^2. Pins the '
                     f'consistent face and body loads and the nodal recovery. Tolerance {_EXACT_TOLERANCE:g} (first '
                     'estimate, not yet run).'),
        specs=lambda: {'run': _column_spec()}, check=_check_column),
    SelfTestCase(
        name='solid_fem3d_object_support', label='Solid mechanics 3D on tetrahedra: Fixed on object faces',
        description=(f'A {_FOOT_SIZE[0] / UM:g} x {_FOOT_SIZE[1] / UM:g} x {_FOOT_SIZE[2] / UM:g} um column with a '
                     f'box object foot (z < {_FOOT_HEIGHT / UM:g} um, one cell layer) held by a Fixed feature on the '
                     'object (its interface faces, mesh.object_faces, outer faces of the foot included), rollers on '
                     f'the four sides, traction t_z = {_FOOT_TRACTION:g} Pa on top; quadratic and linear. Above the '
                     'foot u_z = t (z - h) / (lambda + 2 mu), sigma_zz = t, sigma_yy = lambda / (lambda + 2 mu) t '
                     '(probes) and the foot reaction F_z = -t W D. Tolerance '
                     f'{_EXACT_TOLERANCE:g} (first estimate, not yet run).'),
        specs=lambda: {order: _foot_spec(order) for order in ('quadratic', 'linear')}, check=_check_foot),
    SelfTestCase(
        name='heat_solid_thermal_expansion_fem3d',
        label='Heat -> solid mechanics 3D on one tetrahedra mesh: heated box, free and confined',
        description=(f'Heat and solid_mechanics_3d on the same builtin tetrahedra mesh (0.25 um Kuhn cubes, both '
                     f'quadratic) of a {_THERMAL_SIZE[0] / UM:g} x {_THERMAL_SIZE[1] / UM:g} x '
                     f'{_THERMAL_SIZE[2] / UM:g} um box, heat holding all six faces at T0 + {_THERMAL_RISE:g} K (an '
                     f'exact uniform rise), {_THERMAL_STRAIN} per element. Run free (rollers on x_lo, y_lo, z_lo): u_i = '
                     'alpha_ii dT x_i and zero stress at a probe point; run confined (rollers on all six faces): u = 0, '
                     'sigma_ii = -beta_i dT, elastic_strain_zz = -alpha_zz dT and the high-face reactions -beta_i dT '
                     f'A_i. Tolerance {_EXACT_TOLERANCE:g} (first estimate, not yet run). Pins the same-mesh coupling '
                     '(the transform copies the per-element rise of a heat without a window).'),
        specs=_thermal_specs, check=_check_thermal),
    SelfTestCase(
        name='solid_fem3d_roller_box_modes', label='Solid mechanics 3D on tetrahedra: roller-box vibration modes',
        description=(f'eigenfrequency study on quadratic tetrahedra ({_MODES_CELL / UM:g} um Kuhn cubes) of a '
                     f'{_MODES_SIZE[0] / UM:g} x {_MODES_SIZE[1] / UM:g} x {_MODES_SIZE[2] / UM:g} um block (E '
                     f'{_MODES_ELASTIC[0]:g} Pa, nu {_MODES_ELASTIC[1]:g}, rho {_MODES_DENSITY:g}) with rollers on all '
                     f"six faces, host SuperLU eigen factor: run 'lowest' ({_MODES_LOWEST} modes at target 0) and run "
                     f"'shifted' ({_MODES_SHIFTED} modes nearest {_MODES_SHIFT[1]:g} x the "
                     f'{_MODES_SHIFT[0] + 1}th lowest) against the continuum roller-box modes (longitudinal c_L |k|, '
                     'transverse c_T |k|, k = (p pi / L_x, q pi / L_y, r pi / L_z)) in rank order; run '
                     f"'orthotropic' ({_MODES_LOWEST} lowest) with c_ij entries only (c11 / c22 / c33 "
                     f"{_ORTHOTROPIC['c11']:g} / {_ORTHOTROPIC['c22']:g} / {_ORTHOTROPIC['c33']:g}, c12 / c13 / c23, c44 "
                     f'/ c55 / c66 all different, rho {_ORTHOTROPIC_DENSITY:g}, no youngs_modulus) against the '
                     'orthotropic roller-box continuum (the 3 x 3 symbol per k with c11 k_x^2 + c66 k_y^2 + c55 k_z^2 '
                     '.., (c12 + c66) k_x k_y ..). Relative tolerance '
                     f'{_MODES_TOLERANCE:g} (discretisation error estimate, orthotropic run not yet run). Pins the '
                     'consistent mass, the target shift, the ranking and the 3D c_ij stiffness (c44 / c55 / c66 '
                     'separately) on tetrahedra.'),
        specs=_modes_specs, check=_check_modes, max_wall_time=450.0),
    SelfTestCase(
        name='solid_fem3d_anisotropic_uniform_strain',
        label='Solid mechanics 3D on tetrahedra: anisotropic block, uniform strain and free expansion',
        description=(f'solid_mechanics_3d on builtin tetrahedra (0.25 um Kuhn cubes) of a {_ANISO_SIZE[0] / UM:g} x '
                     f'{_ANISO_SIZE[1] / UM:g} x {_ANISO_SIZE[2] / UM:g} um block with rollers on x_lo / y_lo / z_lo. '
                     f"Runs crystal_quadratic / crystal_linear: lithium niobate as a {{'model': 'crystal', 'plane': "
                     f"'full'}} entry at euler_deg {_ANISO_EULER} (all 21 c_IJ non-zero), tractions T n on all six "
                     f'faces with T = c S of the diagonal strain S = {list(_ANISO_STRAIN)}; run partial (quadratic): '
                     f'E / nu {_ELASTIC[0]:g} / {_ELASTIC[1]:g} with only c11, c14, c44 set (the other entries fall '
                     'back to the Lame values), the same load; run thermal (quadratic): the crystal with alpha '
                     f'{list(_ANISO_ALPHA)} 1/K and a temperature_change of {_ANISO_RISE:g} K, no tractions. Exact in '
                     'P1 and P2: probes u_i = S_ii x_i and the six stresses T (shear rows non-zero from the rotated '
                     'c / c14), thermal u_i = alpha_ii dT x_i and zero stress (needs the shear rows of beta = c alpha '
                     'in the load), and every roller reaction 0 (the full traction is applied on the held faces too). '
                     f'Tolerance {_EXACT_TOLERANCE:g} (float32 probes; first estimate, not yet run).'),
        specs=lambda: {run: _aniso_spec(run) for run in _ANISO_RUNS}, check=_check_aniso),
    SelfTestCase(
        name='solid_fem3d_rigid_mode_check', label='Solid mechanics 3D on tetrahedra: rigid-mode rank check',
        description=('ElasticFEMSystem3D assembled directly (no model run, P1 and P2) on a 4 x 2 x 2 um Kuhn box: (a) '
                     'u_x held on y_lo, u_y on x_lo, u_z on z_lo leaves the rotation about z free (the message must '
                     'name it); (b) two blocks split by a void gap with rollers on the low faces: the far block slides '
                     'along x; (c) the rollers on one block assemble. error = number of differing outcomes, tolerance '
                     '0.'),
        specs=dict, check=_check_rigid_modes),
    SelfTestCase(
        name='solid_fem3d_modes_cudss', label='Solid mechanics 3D on tetrahedra: cuDSS eigen factor vs SuperLU',
        description=('The solid_fem3d_roller_box_modes pencil (P2, built directly, no model run) by eigsh_shifted at 0 '
                     "with eigen factorization superlu, cudss (real LDL^T, double) and cudss_mixed (float32 LDL^T "
                     'refined to full accuracy): the six lowest eigenvalues of the cuDSS runs within relative '
                     f'{_CUDSS_EIGEN_TOLERANCE:g} of SuperLU (first estimate, not yet run). details record the cuDSS '
                     'plan / factorise seconds, device bytes, single_residual and refinement_steps (the measurements of '
                     'agent_help section 6). GPU only; passes with a note without nvmath-python.'),
        specs=dict, check=_check_cudss, backends=('gpu',)),
)

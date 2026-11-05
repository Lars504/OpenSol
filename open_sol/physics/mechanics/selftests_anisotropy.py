"""Self-tests of the anisotropic c_ij stiffness of solid_mechanics (piezoelectric plan stage 5): the host Q1 operator
with c16 / c26 and the stiffness reductions (CPU) and uniform-strain block runs (plane strain, plane stress, thermal, a
YZ lithium niobate crystal entry; exact in Q1); physics imports inside the checks."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre, with_exact_cg
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_VOIGT = ('1', '2', '3', '6')
# monoclinic test material [Pa] (diagonally dominant, so the 4 x 4 (1, 2, 3, 6) block is positive definite)
_MONOCLINIC = {'c11': 200e9, 'c12': 60e9, 'c13': 70e9, 'c22': 160e9, 'c23': 50e9, 'c33': 220e9, 'c16': 15e9,
               'c26': -10e9, 'c36': 8e9, 'c66': 55e9}
_OPERATOR_TOLERANCE = 1e-10


def _matrix(coefficients):
    """(4, 4) symmetric Voigt block (1, 2, 3, 6) of a {c_ij: value} dict."""
    block = np.zeros((4, 4))
    for a, row in enumerate(_VOIGT):
        for b, column in enumerate(_VOIGT):
            low, high = sorted((row, column), key=_VOIGT.index)
            block[a, b] = coefficients[f'c{low}{high}']
    return block


# ---- 1. solid_anisotropic_operator (CPU) ----
def _check_operator(results_by_name):
    """Host Q1 stiffness with c16 / c26 on a graded 3 x 2 mesh of per-element random monoclinic sets: symmetry, the
    energy of a uniform strain = sum of area S^T C S, zero interior forces of a linear field (one material), plane
    stress condensation = the inverse of the compliance block, thermal coefficients, element_stiffness fallbacks."""
    import scipy.sparse
    from .stiffness import (ElementStiffness, element_stiffness, elastic_stress, operator_coefficients,
                            thermal_coefficients)
    from .system import ElasticPattern, stiffness_stencil
    rng = np.random.default_rng(3)
    dx, dy = np.array([0.7e-3, 1.0e-3, 1.3e-3]), np.array([0.9e-3, 0.6e-3])
    shape = (dx.size, dy.size)
    errors = {}

    def operator(values):
        stencil = stiffness_stencil(values['c11'], values['c12'], values['c22'], values['c66'], dx, dy, np.float64,
                                    xp=np, c16=values['c16'], c26=values['c26'])
        return ElasticPattern(shape[0] + 1, shape[1] + 1, xp=np).csr(stencil, scipy.sparse)

    varied = {name: value * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, shape)) for name, value in _MONOCLINIC.items()}
    varied_operator = operator(varied)
    errors['symmetry'] = (abs(varied_operator - varied_operator.T).max() / abs(varied_operator).max())
    x_nodes, y_nodes = np.concatenate(([0.0], np.cumsum(dx))), np.concatenate(([0.0], np.cumsum(dy)))
    node_x, node_y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    gradient = np.array([[1.3e-3, -0.4e-3], [0.7e-3, -0.9e-3]])   # du_a / dx_b
    displacement = np.stack((gradient[0, 0] * node_x + gradient[0, 1] * node_y,
                             gradient[1, 0] * node_x + gradient[1, 1] * node_y), axis=-1).reshape(-1)
    strain = np.array([gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]])   # S_xx, S_yy, g
    in_plane = [0, 1, 3]
    area = dx[:, None] * dy[None, :]
    expected = sum(area[i, j] * strain @ _matrix({name: values[i, j] for name, values in varied.items()})[
        np.ix_(in_plane, in_plane)] @ strain for i in range(shape[0]) for j in range(shape[1]))
    errors['uniform strain energy'] = abs(displacement @ (varied_operator @ displacement) / expected - 1.0)
    uniform = {name: np.full(shape, value) for name, value in _MONOCLINIC.items()}
    forces = (operator(uniform) @ displacement).reshape(shape[0] + 1, shape[1] + 1, 2)
    errors['interior forces'] = float(np.max(np.abs(forces[1:-1, 1:-1]))) / float(np.max(np.abs(forces)))
    # reductions of one material through ElementStiffness
    stiffness = ElementStiffness({name: np.full((1, 1), value) for name, value in _MONOCLINIC.items()},
                                 np.ones((1, 1), dtype=bool), True, True)
    compliance = np.linalg.inv(_matrix(_MONOCLINIC))
    reduced = np.linalg.inv(compliance[np.ix_(in_plane, in_plane)])
    condensed = [float(value[0, 0]) for value in operator_coefficients(stiffness, True)]   # 11, 12, 22, 66, 16, 26
    expected_condensed = [reduced[0, 0], reduced[0, 1], reduced[1, 1], reduced[2, 2], reduced[0, 2], reduced[1, 2]]
    errors['plane stress condensation'] = max(abs(a - b) for a, b in zip(condensed, expected_condensed)) / reduced[0, 0]
    alpha = np.array([2e-5, 1e-5, 3e-5])
    betas = [float(value[0, 0]) for value in thermal_coefficients(stiffness, *alpha)]
    expected_betas = _matrix(_MONOCLINIC)[:, :3] @ alpha
    errors['thermal coefficients'] = float(np.max(np.abs(np.array(betas) - expected_betas))) / float(
        np.max(np.abs(expected_betas)))
    strain_4 = np.array([1e-3, -2e-4, 3e-4, 5e-4])
    stress = [float(value[0, 0]) for value in elastic_stress(stiffness, *strain_4)]
    expected_stress = _matrix(_MONOCLINIC) @ strain_4
    errors['stress'] = float(np.max(np.abs(np.array(stress) - expected_stress))) / float(np.max(np.abs(expected_stress)))
    # fallbacks: c11 set, the rest NaN or unallocated -> the isotropic Lame values; void cells (no stiffness) 0
    lame_lambda, mu = np.array([[50e9, 0.0]]), np.array([[30e9, 0.0]])
    tensor = {'c11': np.array([[250e9, np.nan]]), 'c12': np.array([[np.nan, np.nan]]), 'c16': np.array([[0.0, 0.0]])}
    mixed = element_stiffness(lame_lambda, mu, tensor, np.float64)
    expected_values = {'c11': 250e9, 'c12': 50e9, 'c22': 110e9, 'c33': 110e9, 'c13': 50e9, 'c66': 30e9, 'c16': 0.0}
    errors['fallbacks'] = max(abs(float(mixed[name][0, 0]) - value) for name, value in expected_values.items()) / 250e9
    if not (bool(mixed.solid[0, 0]) and not bool(mixed.solid[0, 1]) and mixed.anisotropic and not mixed.monoclinic):
        errors['fallback flags'] = 1.0
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _OPERATOR_TOLERANCE,
            'details': {'errors': errors, 'worst': worst}}


# ---- 2. solid_anisotropic_uniform_strain (model runs, exact in Q1) ----
# 4 x 3 mm block of 0.5 mm cells (8 x 6), rollers on x_lo (u_x) and y_lo (u_y). Target u = (s_x x, s_y y): uniform
# strain (S_xx, S_yy, g = 0), so with the uniform stress T = C S - beta dT the tractions x_hi (T_xx, T_xy), y_hi (T_xy,
# T_yy), x_lo (0, -T_xy) and y_lo (-T_xy, 0) make it the exact discrete solution (linear u lies in Q1); the rollers take
# the normal parts (anchor_x reaction F_x = -T_xx H). Runs: plane_strain (S_zz = 0, stress_zz = T_zz), plane_stress
# (S_zz = -(c31 S_xx + c32 S_yy) / c33, stress_zz 0), thermal (plane strain, u = (alpha_xx dT x, alpha_yy dT y): T_I =
# -c_I3 alpha_zz dT), crystal (YZ lithium niobate entry, plane strain).
_BLOCK = (4e-3, 3e-3)
_BLOCK_CELL = 0.5e-3
_BLOCK_CELLS = (8, 6)
_TARGET = (1e-3, -4e-4)             # (s_x, s_y) of the mechanical runs
_ALPHA = (2e-5, 1e-5, 3e-5)         # alpha_xx, alpha_yy, alpha_zz [1/K]
_TEMPERATURE_CHANGE = 50.0          # K
_YZ_EULER = [0.0, 90.0, 90.0]
_FIELD_TOLERANCE = 1e-5             # float32 outputs, relative to each field's scale
_REACTION_TOLERANCE = 1e-6


def _crystal_coefficients():
    """{c_ij} [Pa] of YZ lithium niobate in the sagittal model axes (materials.crystal_tensors)."""
    from ...materials.crystal_tensors import model_tensors
    stiffness, _, _, _ = model_tensors('lithium_niobate', tuple(_YZ_EULER))
    index = {'1': 0, '2': 1, '3': 2, '6': 5}
    return {f'c{row}{column}': float(stiffness[index[row], index[column]])
            for row, column in ('11', '12', '13', '22', '23', '33', '16', '26', '36', '66')}


def _state(run):
    """(c_ij, strain (S_xx, S_yy, S_zz), stress (T_xx, T_yy, T_zz, T_xy), elastic S_zz) expected in a run."""
    coefficients = _crystal_coefficients() if run == 'crystal' else _MONOCLINIC
    block = _matrix(coefficients)
    if run == 'thermal':
        s_x, s_y = _ALPHA[0] * _TEMPERATURE_CHANGE, _ALPHA[1] * _TEMPERATURE_CHANGE
        stress = -block[:, 2] * _ALPHA[2] * _TEMPERATURE_CHANGE
        return coefficients, (s_x, s_y, 0.0), stress, -_ALPHA[2] * _TEMPERATURE_CHANGE
    s_x, s_y = _TARGET
    s_z = -(block[2, 0] * s_x + block[2, 1] * s_y) / block[2, 2] if run == 'plane_stress' else 0.0
    stress = block @ np.array([s_x, s_y, s_z, 0.0])
    if run == 'plane_stress':
        stress[2] = 0.0
    return coefficients, (s_x, s_y, s_z), stress, s_z


def _block_spec(run):
    coefficients, _, stress, _ = _state(run)
    t_xx, t_yy, _, t_xy = (float(value) for value in stress)
    properties = {'youngs_modulus': 1e9, 'poisson_ratio': 0.25}   # every c_ij below overrides these
    dispersion = None
    if run == 'crystal':
        dispersion = [{'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': list(_YZ_EULER)}]
    else:
        properties.update(coefficients)
    features = [FeatureSpec('anchor_x', 'roller', ['x_lo']), FeatureSpec('anchor_y', 'roller', ['y_lo']),
                FeatureSpec('load_x_hi', 'boundary_load', ['x_hi'], {'t_x': t_xx, 't_y': t_xy}),
                FeatureSpec('load_y_hi', 'boundary_load', ['y_hi'], {'t_x': t_xy, 't_y': t_yy}),
                FeatureSpec('load_x_lo', 'boundary_load', ['x_lo'], {'t_x': 0.0, 't_y': -t_xy}),
                FeatureSpec('load_y_lo', 'boundary_load', ['y_lo'], {'t_x': -t_xy, 't_y': 0.0})]
    if run == 'thermal':
        properties.update(alpha_xx=_ALPHA[0], alpha_yy=_ALPHA[1], alpha_zz=_ALPHA[2])
        features.append(FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _TEMPERATURE_CHANGE}))
    settings = {'out_of_plane': 'plane_stress'} if run == 'plane_stress' else {}
    return with_exact_cg(ModelSpec(
        title=f'Self-test: anisotropic block under uniform strain, {run} (solid mechanics)',
        domain=DomainSpec(size_x=_BLOCK[0], size_y=_BLOCK[1]),
        materials=[MaterialSpec('crystal', properties=properties, dispersion=dispersion, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_BLOCK_CELL),
        study=StudySpec(type='stationary')))


_BLOCK_RUNS = ('plane_strain', 'plane_stress', 'thermal', 'crystal')


def _block_specs():
    return {run: _block_spec(run) for run in _BLOCK_RUNS}


def _field(results, name):
    values = np.asarray(results.field(f'{_PHYSICS}.{name}').values, dtype=np.float64)
    if values.shape != _BLOCK_CELLS:
        raise ValueError(f'{name} window shape {values.shape} differs from the {_BLOCK_CELLS} block cells')
    return values


def _check_block(results_by_name):
    """Every run's displacement, strain, stress and elastic_strain_zz fields against the uniform state (float32,
    relative to each field's scale), the plane-strain anchor_x reaction F_x = -T_xx H (float64)."""
    errors, tolerances, details = {}, {}, {}
    x_centres = cell_centre(np.arange(_BLOCK_CELLS[0]), _BLOCK_CELL)[:, None]
    y_centres = cell_centre(np.arange(_BLOCK_CELLS[1]), _BLOCK_CELL)[None, :]
    for run in _BLOCK_RUNS:
        results = results_by_name[run]
        _, (s_x, s_y, _), stress, elastic_zz = _state(run)
        stress_scale = float(np.max(np.abs(stress)))
        expected = {'u_x': (s_x * x_centres, abs(s_x) * _BLOCK[0]), 'u_y': (s_y * y_centres, abs(s_y) * _BLOCK[1]),
                    'strain_xx': (s_x, abs(s_x)), 'strain_yy': (s_y, abs(s_x)), 'strain_xy': (0.0, abs(s_x)),
                    'stress_xx': (stress[0], stress_scale), 'stress_yy': (stress[1], stress_scale),
                    'stress_zz': (stress[2], stress_scale), 'stress_xy': (stress[3], stress_scale),
                    'elastic_strain_zz': (elastic_zz, abs(s_x))}
        for name, (value, scale) in expected.items():
            errors[f'{run} {name}'] = float(np.max(np.abs(_field(results, name) - value))) / scale
            tolerances[f'{run} {name}'] = _FIELD_TOLERANCE
        details[run] = {'strain': [s_x, s_y], 'stress': [float(value) for value in stress]}
    _, _, stress, _ = _state('plane_strain')
    reaction = results_by_name['plane_strain'].result(_PHYSICS, 'anchor_x')['F_x']
    errors['plane_strain anchor_x F_x'] = abs(reaction / (-stress[0] * _BLOCK[1]) - 1.0)
    tolerances['plane_strain anchor_x F_x'] = _REACTION_TOLERANCE
    normalized = {item: error / tolerances[item] for item, error in errors.items()}
    worst = max(normalized, key=lambda item: math.inf if math.isnan(normalized[item]) else normalized[item])
    details.update(errors=errors, worst_item=worst, crystal_coefficients=_crystal_coefficients())
    return {'measured': normalized[worst], 'expected': 0.0, 'error': normalized[worst], 'tolerance': 1.0,
            'details': details}


# The orthotropic roller-box modes (c_ij-only cells, c22 != c11) are pinned by solid_eigenfrequency_roller_box
# (selftests_eigenfrequency.py), which runs that material.
ANISOTROPY_CASES = (
    SelfTestCase(
        name='solid_anisotropic_operator',
        label='Solid mechanics: anisotropic Q1 operator and stiffness reductions (CPU)',
        description='Host checks (specs {}) of the c_ij path of solid_mechanics: the Q1 stencil with c16 / c26 on a '
                    'graded 3 x 2 element mesh with per-element monoclinic sets is symmetric, gives u^T K u = sum of '
                    'area S^T C S for a uniform strain and zero interior nodal forces of a linear field in one '
                    'material; stiffness.operator_coefficients(plane stress) equals the inverse of the in-plane '
                    'compliance block, thermal_coefficients C alpha, elastic_stress C S; element_stiffness falls back '
                    'per component to the Lame values (NaN / unallocated) and keeps void cells at 0. measured = the '
                    f'worst relative error, tolerance {_OPERATOR_TOLERANCE:g}.',
        specs=lambda: {}, check=_check_operator),
    SelfTestCase(
        name='solid_anisotropic_uniform_strain',
        label='Solid mechanics: anisotropic block under uniform strain (plane strain / stress, thermal, crystal)',
        description=f'stationary runs of a {_BLOCK[0] * 1e3:g} x {_BLOCK[1] * 1e3:g} mm block of {_BLOCK_CELL * 1e3:g} '
                    'mm cells with rollers on x_lo and y_lo and boundary loads on all four faces chosen so that u = '
                    '(s_x x, s_y y) with the uniform stress T = C S - beta dT is the exact Q1 solution (exact CG): '
                    'plane_strain and plane_stress (a monoclinic c_ij set with c16, c26, c36 != 0), thermal (plane '
                    'strain, alpha_xx / _yy / _zz and a temperature change: T_I = -c_I3 alpha_zz dT) and crystal (a YZ '
                    "lithium niobate {'model': 'crystal'} entry, euler 0, 90, 90). Fields u_x, u_y, strain_*, "
                    f'stress_xx / _yy / _xy / _zz and elastic_strain_zz within {_FIELD_TOLERANCE:g} of their scale, '
                    f'the plane-strain anchor reaction F_x = -T_xx H within {_REACTION_TOLERANCE:g}. measured = error '
                    '= the worst deviation in tolerance units, tolerance 1: this pins the monoclinic operator and '
                    'thermal shear load, the plane-stress condensation, stress_zz and the crystal expansion of the c_ij.',
        specs=_block_specs, check=_check_block),
)

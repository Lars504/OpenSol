"""Self-tests of the frequency-dependent viscoelastic moduli and the mixed u-p formulation of the FEM solids
(frequency_response): the per-frequency term machinery of frequency_response.HarmonicSystem against dense solves, the
validation rules and the enclosed-volume check, a viscoelastic P-wave strip (triangles, axisymmetric, tetrahedra;
displacement and mixed), Stokes' second problem of a viscous liquid and of two touching liquid layers (mixed, GPU) and
the drag of a rigid sphere oscillating in water (axisymmetric m = 0 and m = 1). definition registers them; no physics
imports at module level."""
import math

import numpy as np

from ...model.schema import FREQUENCY_RESPONSE_STUDY, SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import (SubChecks, available_host_direct, cpu_defaults, float32_value,
                                       scaled_worst_result)
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, RefinementSpec, SolverDefaultsSpec, StudySpec)
from .selftests import _rectangle
from .selftests_fem import _PATCH_SIZE, _patch_mesh
from .selftests_fem3d import _point, _solid_spec

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'          # = selftests.SOLID_TYPE_NAME
_TRIANGLE_MESH = 'triangles'
_MIXED = {'formulation': 'mixed'}
_QUADRATIC = {'discretization': 'quadratic'}


def _reaction(results, feature, component):
    """(frequencies [Hz], complex reaction [N or N/m]) of a support's frequency-response table, ascending f."""
    table = results.result(_PHYSICS, feature)
    columns = list(table['columns'])
    rows = np.asarray(table['rows'], dtype=np.float64)
    order = np.argsort(rows[:, 0], kind='stable')
    rows = rows[order]
    return rows[:, 0], rows[:, columns.index(f'{component}_re')] + 1j * rows[:, columns.index(f'{component}_im')]


def _probe(results, feature, component):
    """(frequencies [Hz], complex displacement [m]) of a frequency_probe, ascending f."""
    result = results.result(_PHYSICS, feature)
    frequencies = np.asarray(result['x'], dtype=np.float64)
    series = result['series']
    values = np.asarray(series[f'{component}_re'], dtype=np.float64) + 1j * np.asarray(series[f'{component}_im'],
                                                                                       dtype=np.float64)
    order = np.argsort(frequencies, kind='stable')
    return frequencies[order], values[order]


def _derived(results, name):
    return float(results.summary()['derived'][name])


def _pair(value):
    """[real, imag] floats of a complex value (JSON-able details)."""
    value = complex(value)
    return [value.real, value.imag]


# ---- solid_harmonic_terms (check({}), cpu + gpu half): HarmonicSystem with per-frequency terms vs dense solves ----
# A 1D P1 chain of _TERMS_NODES nodes: K = 1e4 x the Laplacian (+ a small diagonal), M the consistent P1 mass x 1e-4, loss
# 0.02 K, node 0 held at 1e-3 (the held-value product), a fixed pseudo-random load; terms D1 (the Laplacian on nodes
# 0-24) with c1(w) = 1e3 (1 + 0.3 j w tau) / (1 + j w tau) and D2 (nodes 15-39) with c2(w) = j w mu. 'saddle' adds
# _TERMS_PRESSURE pressure rows: [[A, -B^T], [-B, 0]] with B on 10-wide node bands, the zero pressure block kept as a
# zero-coefficient FrequencyTerm (stored diagonal) and a SchurScaleTerm of B (equilibrated on every route, SuperLU
# included: the saddle runs check that every solution is the unscaled one). Per runtime (private SuperLU,
# scipy_splu, MUMPS / PARDISO when installed, cuDSS on a GPU) and frequency: x and A x against the dense solve, the
# product_at_rows rows (0 elsewhere), the rows-only solve (fields False) and its last_solution; plan_reduced refuses the
# terms; terms=() is bit-identical to no keyword.
_TERMS_NODES = 40
_TERMS_PRESSURE = 4
_TERMS_TAU = 1e-4                       # s
_TERMS_MU = 0.5                         # coefficient of j w on D2
_TERMS_OMEGAS = (2.0e3, 1.5e4, 6.0e4)   # rad/s (K - w^2 M indefinite at the last two)
_TERMS_HELD = 1e-3
_TERMS_ROWS = (0, 7, 20, 39)            # product_at_rows check rows (+ the first pressure row in 'saddle')
_TERMS_TOLERANCE = 1e-10                # relative max |dx| / max |x| of host SuperLU (first estimate)
_TERMS_DIRECT_TOLERANCE = 1e-9          # MUMPS / PARDISO / cuDSS (refined factors; first estimate)


def _shear_coefficient(omega):
    return 1e3 * (1.0 + 0.3j * omega * _TERMS_TAU) / (1.0 + 1j * omega * _TERMS_TAU)


def _viscous_coefficient(omega):
    return 1j * omega * _TERMS_MU


def _zero_coefficient(omega):
    return 0j


def _chain(first, last, scale, n_total, mass=False):
    """CSR (n_total, n_total) of the 1D P1 Laplacian (mass: the consistent mass) on nodes first .. last - 1."""
    import scipy.sparse
    block = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0 if mass else np.array([[1.0, -1.0], [-1.0, 1.0]])
    rows, columns, values = [], [], []
    for node in range(first, last - 1):
        for a in range(2):
            for b in range(2):
                rows.append(node + a)
                columns.append(node + b)
                values.append(scale * block[a, b])
    return scipy.sparse.csr_matrix((values, (rows, columns)), shape=(n_total, n_total))


def _terms_problem(saddle):
    """(operator, loss, mass, terms, scale_terms, free, fixed, load) of the synthetic pencil over n (+ the pressure)."""
    import scipy.sparse
    from ...fem.assembly import pattern_sum
    from ...fem.mixed_elasticity import embedded, saddle_coupling
    from .frequency_response import FrequencyTerm, SchurScaleTerm
    n = _TERMS_NODES
    n_total = n + (_TERMS_PRESSURE if saddle else 0)
    stiffness = pattern_sum([_chain(0, n, 1e4, n_total),
                             scipy.sparse.diags(np.r_[np.full(n, 10.0), np.zeros(n_total - n)], format='csr')])
    terms = [FrequencyTerm(_chain(0, 25, 1.0, n_total), _shear_coefficient, 'shear group'),
             FrequencyTerm(_chain(15, n, 1.0, n_total), _viscous_coefficient, 'viscous group')]
    scale_terms = []
    if saddle:
        rng = np.random.default_rng(11)
        rows = np.repeat(np.arange(_TERMS_PRESSURE), 10)
        columns = (10 * rows + np.tile(np.arange(10), _TERMS_PRESSURE)) % n
        coupling = scipy.sparse.csr_matrix((rng.standard_normal(rows.size), (rows, columns)),
                                           shape=(_TERMS_PRESSURE, n))
        stiffness = pattern_sum([stiffness, saddle_coupling(coupling, n)])
        pressure_mass = scipy.sparse.csr_matrix(np.diag(np.full(_TERMS_PRESSURE, 0.5))
                                                + np.diag(np.full(_TERMS_PRESSURE - 1, 0.1), 1)
                                                + np.diag(np.full(_TERMS_PRESSURE - 1, 0.1), -1))
        block = embedded(pressure_mass, n_total, row_offset=n, column_offset=n)
        terms.append(FrequencyTerm(block, _zero_coefficient, 'incompressible block'))
        scale_terms.append(SchurScaleTerm(embedded(coupling, n_total, row_offset=n)))
    free = np.ones(n_total, dtype=bool)
    free[0] = False
    fixed = np.zeros(n_total)
    fixed[0] = _TERMS_HELD
    load = np.where(free, np.random.default_rng(5).standard_normal(n_total), 0.0)
    return (stiffness, 0.02 * stiffness if not saddle else 0.02 * _chain(0, n, 1e4, n_total),
            _chain(0, n, 1e-4, n_total, mass=True), tuple(terms), tuple(scale_terms), free, fixed, load)


def _dense_solution(problem, omega):
    """(x, A x) of the dense pencil of _terms_problem at omega."""
    operator, loss, mass, terms, _, free, fixed, load = problem
    matrix = (operator.toarray() + 1j * loss.toarray() - omega * omega * mass.toarray()).astype(np.complex128)
    for term in terms:
        matrix = matrix + complex(term.coefficient(omega)) * term.matrix.toarray()
    solution = fixed.astype(np.complex128)
    rhs = load - matrix[:, ~free] @ fixed[~free]
    solution[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs[free])
    return solution, matrix @ solution


def _terms_runtimes():
    """[(name, SparseDirectSolver or None, tolerance)]: private SuperLU, scipy_splu, installed host direct libraries,
    cuDSS when a GPU and nvmath-python are there."""
    from ...core.backend import gpu_available
    from ...solvers.cudss_factor import cudss_available
    from ...solvers.sparse_direct import SparseDirectSolver
    runtimes = [('private SuperLU', None, _TERMS_TOLERANCE),
                ('scipy_splu', SparseDirectSolver('scipy_splu', backend='cpu'), _TERMS_TOLERANCE)]
    runtimes += [(name, SparseDirectSolver(name, backend='cpu'), _TERMS_DIRECT_TOLERANCE)
                 for name in available_host_direct()]
    if gpu_available() and cudss_available():
        runtimes.append(('cudss', SparseDirectSolver('cudss', backend='gpu'), _TERMS_DIRECT_TOLERANCE))
    return runtimes


def _release(harmonic):
    factor = harmonic.factor
    if hasattr(factor, 'release'):
        factor.release()


def _check_terms(_):
    from .frequency_response import HarmonicSystem
    checks, details = SubChecks(), {}
    for variant in ('plain', 'saddle'):
        problem = _terms_problem(variant == 'saddle')
        operator, loss, mass, terms, scale_terms, free, fixed, load = problem
        references = {omega: _dense_solution(problem, omega) for omega in _TERMS_OMEGAS}
        rows = np.array(_TERMS_ROWS + ((_TERMS_NODES,) if variant == 'saddle' else ()), dtype=np.int64)
        for name, runtime, tolerance in _terms_runtimes():
            harmonic = HarmonicSystem(operator, loss, mass, free, runtime=runtime, terms=terms,
                                      scale_terms=scale_terms)
            on_rows = HarmonicSystem(operator, loss, mass, free, runtime=runtime, terms=terms,
                                     scale_terms=scale_terms)
            on_rows.set_output_rows(rows)
            others = np.setdiff1d(np.arange(operator.shape[0]), rows)
            for omega in _TERMS_OMEGAS:
                expected, expected_product = references[omega]
                solution, product = harmonic.solve(omega, fixed, load)
                label = f'{variant} {name} w={omega:g}'
                checks.add_error(f'{label} x', np.max(np.abs(solution - expected)) / np.max(np.abs(expected)),
                                 tolerance)
                checks.add_error(f'{label} A x', np.max(np.abs(product - expected_product))
                                 / np.max(np.abs(expected_product)), tolerance)
                _, row_product = on_rows.solve(omega, fixed, load, product_at_rows=True)
                checks.add_error(f'{label} A x at the output rows', np.max(np.abs(row_product[rows]
                                                                                   - expected_product[rows]))
                                 / np.max(np.abs(expected_product)), tolerance)
                checks.add(f'{label} A x 0 off the output rows', bool(np.all(row_product[others] == 0.0)))
                # rows-only direct solve (_rows_solve) and its kept free solution (last_solution)
                row_solution, row_product = on_rows.solve(omega, fixed, load, fields=False)
                scale = np.max(np.abs(expected))
                checks.add_error(f'{label} rows-only x at the output rows',
                                 np.max(np.abs(row_solution[rows] - expected[rows])) / scale, tolerance)
                checks.add_error(f'{label} rows-only A x at the output rows', np.max(np.abs(
                    row_product[rows] - expected_product[rows])) / np.max(np.abs(expected_product)), tolerance)
                checks.add_error(f'{label} last_solution', np.max(np.abs(on_rows.last_solution() - expected)) / scale,
                                 tolerance)
            checks.add_raises(f'{variant} {name} plan_reduced refuses the terms',
                              lambda: harmonic.plan_reduced(np.array(_TERMS_OMEGAS), 'adaptive', fixed, load))
            details[f'{variant} {name}'] = {'factorizations': int(harmonic.factorizations)}
            _release(harmonic)
            _release(on_rows)
    # no terms: the keyword-free system and terms=() / scale_terms=() give the same bits (plain pencil)
    operator, loss, mass, _, _, free, fixed, load = _terms_problem(False)
    for name, runtime, _ in _terms_runtimes()[:2]:
        plain = HarmonicSystem(operator, loss, mass, free, runtime=runtime)
        empty = HarmonicSystem(operator, loss, mass, free, runtime=runtime, terms=(), scale_terms=())
        same = all(all(np.array_equal(a, b) for a, b in zip(plain.solve(omega, fixed, load),
                                                            empty.solve(omega, fixed, load)))
                   for omega in _TERMS_OMEGAS)
        checks.add(f'terms=() bit-identical ({name})', same)
        _release(plain)
        _release(empty)
    return checks.result(**details)


# ---- solid_viscoelastic_validation (check({}), no run): one spec per rule of definition_viscoelastic, and the
# enclosed-volume check of an incompressible region (solver_common.HarmonicOperators.check_held_volumes) on the 2 x 2 um
# patch mesh of solid_fem_patch (P2, every boundary dof held: the whole liquid is enclosed, one gauge): a rigid
# translation of the boundary keeps the volume (passes), a uniform expansion changes it (refused); then two touching
# liquids (an oil drop of 4x viscosity in the patch disc, by element centroid, water around it): neither group is
# enclosed alone, their union is (one gauge; the per-group detection found none and left the saddle point singular).
_RULE_SIZE = 1e-3
_RULE_FREQUENCY = 1e3
_GEL = {'rho': 1000.0, 'shear_modulus': 1e6, 'bulk_modulus': 1e7}
_GEL_VISCOUS = {**_GEL, 'viscosity': 1.0}
_LIQUID = {'rho': 1000.0, 'viscosity': 1e-3}
_INCOMPRESSIBLE_GEL = {'rho': 1000.0, 'shear_modulus': 1e6}   # mixed: no bulk modulus, not frequency dependent
_GAUGE_STRAIN = 1e-3
_GAUGE_SHIFT = 1e-9                     # m
_PATCH_CENTRE = 0.5 * _PATCH_SIZE      # the solid_fem_patch disc (centre, radius): the oil drop of the gauge check
_PATCH_DROP_RADIUS = 0.25 * _PATCH_SIZE


def _rule_spec(properties, settings=None, study=FREQUENCY_RESPONSE_STUDY, kind=_TRIANGLE_MESH, materials=None,
               geometry=(), axisymmetric=False, study_settings=None, physics=()):
    """Validation spec: a 1 mm square of one material ('gel', or `materials`), fixed x_lo (axisymmetric: y_lo) and a
    displacement drive on x_hi (y_hi), quadratic, one frequency (or `study`)."""
    settings = {**_QUADRATIC, **({'geometry': 'axisymmetric'} if axisymmetric else {}), **(settings or {})}
    low, high = ('y_lo', 'y_hi') if axisymmetric else ('x_lo', 'x_hi')
    drive = {'u_y': 1e-9} if axisymmetric else {'u_x': 1e-9}
    if study == FREQUENCY_RESPONSE_STUDY:
        study_spec = StudySpec(type=study, settings={'frequencies': [_RULE_FREQUENCY], **(study_settings or {})})
    else:
        study_spec = StudySpec(type=study, settings=dict(study_settings or {}))
    return ModelSpec(
        title='Self-test: viscoelastic validation rule', domain=DomainSpec(size_x=_RULE_SIZE, size_y=_RULE_SIZE),
        geometry=list(geometry),
        materials=list(materials) if materials is not None else [MaterialSpec('gel', properties=dict(properties),
                                                                              selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings=settings, features=[
            FeatureSpec('anchor', 'fixed', [low]), FeatureSpec('drive', 'displacement', [high], drive)]),
                 *physics],
        meshes=[MeshSpec(name='mesh', kind=kind, mode='uniform', dx=_RULE_SIZE / 4.0)],
        solver_defaults=cpu_defaults(), study=study_spec)


def _rule_cases():
    """{name: (spec, level or None (no error), phrase)} of the rules."""
    strips = [_rectangle(f'strip_{index}', index * _RULE_SIZE / 33.0, 0.0, (index + 1) * _RULE_SIZE / 33.0,
                         _RULE_SIZE) for index in range(33)]
    many = [MaterialSpec(f'gel_{index}', properties={**_GEL, 'viscosity': 1e-3 * (index + 1)},
                         selection=[f'strip_{index}']) for index in range(33)]
    return {
        'valid displacement, viscous gel': (_rule_spec(_GEL_VISCOUS), None, ''),
        'valid mixed liquid': (_rule_spec(_LIQUID, _MIXED), None, ''),
        'V1 mixed on a tensor mesh': (_rule_spec(_GEL, _MIXED, kind='tensor'), 'error',
                                      'needs a triangle or tetrahedron mesh'),
        'V2 mixed outside frequency_response': (_rule_spec(_GEL, _MIXED, study='eigenfrequency'), 'error',
                                                'runs in frequency_response only'),
        'V3 mixed linear elements': (_rule_spec(_GEL, {**_MIXED, 'discretization': 'linear'}), 'error',
                                     'needs discretization quadratic'),
        'V4 frequency-dependent and piezoelectric': (_rule_spec(_GEL_VISCOUS, {'piezoelectric': True}), 'error',
                                                     'not available with piezoelectric on'),
        'V5 frequency-dependent and plane_stress': (_rule_spec(_GEL_VISCOUS, {'out_of_plane': 'plane_stress'}),
                                                    'error', 'not available with plane_stress'),
        'V6 mixed and stress_stiffening': (_rule_spec(_GEL, {**_MIXED, 'stress_stiffening': True}), 'error',
                                           'stress_stiffening or geometric_nonlinearity'),
        'V7 frequency-dependent adaptive sweep': (_rule_spec(_GEL_VISCOUS, study_settings={'sweep_method': 'adaptive'}),
                                                  'error', 'needs a pencil polynomial'),
        'V7b mixed incompressible adaptive at m = 1': (   # not viscous: V7 (frequency dependent) would fire first
            _rule_spec(_INCOMPRESSIBLE_GEL, {**_MIXED, 'azimuthal_order': 1}, axisymmetric=True,
                       study_settings={'sweep_method': 'adaptive'}), 'error', 'cannot take an incompressible material'),
        'V8 two routes': (_rule_spec({'rho': 1000.0, 'youngs_modulus': 1e9, 'poisson_ratio': 0.3,
                                      'shear_modulus': 1e6}), 'error', 'one route per material'),
        'V9 c_ij and viscosity': (_rule_spec({'rho': 1000.0, 'c11': 2e9, 'c12': 1e9, 'c22': 2e9, 'c66': 5e8,
                                              'c13': 1e9, 'c23': 1e9, 'c33': 2e9, 'viscosity': 1.0}),
                                  'error', 'the viscoelastic law is isotropic'),
        'V10 branch without time': (_rule_spec({**_GEL, 'shear_relaxation_modulus_1': 1e5}), 'error',
                                    'without shear_relaxation_time_1 > 0'),
        'V11 bulk viscosity without bulk modulus': (_rule_spec({'rho': 1000.0, 'shear_modulus': 1e6,
                                                                'bulk_viscosity': 1.0}, _MIXED),
                                                    'error', 'needs a finite bulk modulus'),
        'V12 shear modulus without bulk modulus': (_rule_spec({'rho': 1000.0, 'shear_modulus': 1e6}), 'error',
                                                   'needs bulk_modulus'),
        'V13 liquid vocabulary, displacement': (_rule_spec({'rho': 1000.0, 'shear_modulus': 0.0,
                                                            'bulk_modulus': 1e9}),
                                                'error', 'a liquid cell, use formulation mixed'),
        'V14 bulk modulus only, mixed': (_rule_spec({'rho': 1000.0, 'bulk_modulus': 1e9}, _MIXED), 'error',
                                         'nothing for the mixed formulation'),
        'V15 library water, displacement': (
            _rule_spec({}, materials=[MaterialSpec('water', library='water', selection=['all'])]), 'warning',
            'void for'),
        'V16 viscosity outside frequency_response': (_rule_spec(_GEL_VISCOUS, study='stationary'), 'warning',
                                                     'ignored by the'),
        'V17 liquid in a static study': (_rule_spec({'rho': 1000.0, 'shear_modulus': 0.0, 'bulk_modulus': 1e9},
                                                    study='stationary'), 'error', 'a liquid has no static stiffness'),
        'V18 shear modulus on a tensor mesh': (_rule_spec(_GEL, kind='tensor'), 'error',
                                               'triangle / tetrahedron meshes only'),
        'V19 laminar_flow shares the viscosity': (_rule_spec(_GEL_VISCOUS, physics=[PhysicsSpec('flow',
                                                                                                 'laminar_flow')]),
                                                  'warning', 'treats these cells as fluid'),
        'V20 too many parameter groups': (_rule_spec({}, materials=many, geometry=strips), 'error', 'exceed 32'),
        'V21 nearly incompressible, displacement': (_rule_spec({'rho': 1000.0, 'youngs_modulus': 1e6,
                                                                'poisson_ratio': 0.49}),
                                                    'info', 'consider formulation mixed'),
        'V22 mixed with a PML': (_rule_spec(_GEL, {**_MIXED, 'pml_x_hi': True}), 'error',
                                 'formulation mixed is not available with'),
    }


def _gauge_checks(checks):
    """The enclosed-volume check on the patch mesh (case comment)."""
    import scipy.sparse
    from ...fem.dofs import lagrange_dofs
    from ...fem.elasticity import assemble_elastic_stiffness_groups
    from ...fem.mixed_elasticity import assemble_pressure_terms, pressure_space
    from .solver_common import mixed_harmonic_operators
    from .stiffness import unit_operator_coefficients
    from .viscoelastic import ModulusLaw, ViscoelasticGroup
    mesh = _patch_mesh()
    dofs = lagrange_dofs(mesh, 2)
    n_elements = mesh.n_elements
    indices = np.zeros(n_elements, dtype=np.int64)
    liquid = ViscoelasticGroup(0, np.ones(n_elements, dtype=bool), ModulusLaw(0.0, 0.0, 1e-3, ()),
                               ModulusLaw(math.inf, 0.0, 0.0, ()))
    deviatoric = [assemble_elastic_stiffness_groups(dofs, [unit_operator_coefficients('shear', n_elements, False)],
                                                    indices, 1)[0][0]]
    space = pressure_space(dofs, indices, 1)
    coupling, masses = assemble_pressure_terms(dofs, space)
    n_u = 2 * dofs.n_dofs
    boundary = dofs.edge_dofs(dofs.boundary_edges())
    free = np.ones(n_u, dtype=bool)
    free[2 * boundary] = free[2 * boundary + 1] = False
    operators = mixed_harmonic_operators((liquid,), deviatoric, coupling, masses, space,
                                         scipy.sparse.identity(n_u, format='csr'), free,
                                         np.ones(space.n_dofs, dtype=bool), None, lambda: 'enclosed-volume self-test')
    checks.add('gauge: one enclosed region pinned', len(operators.pressure_gauges) == 1
               and len(operators.volume_checks) == 1, value=len(operators.pressure_gauges))
    ids = dofs.global_ids[boundary]
    on_node = ids < mesh.n_points
    points = np.empty((boundary.size, 2))
    points[on_node] = mesh.points[ids[on_node]]
    ends = mesh.edges[ids[~on_node] - mesh.n_points].astype(np.int64)
    points[~on_node] = 0.5 * (mesh.points[ends[:, 0]] + mesh.points[ends[:, 1]])   # straight outer edges
    translation = np.zeros(operators.n_unknowns)
    translation[2 * boundary] = _GAUGE_SHIFT
    translation[2 * boundary + 1] = -0.5 * _GAUGE_SHIFT
    try:
        operators.check_held_volumes(translation, lambda: 'enclosed-volume self-test')
        checks.add('gauge: a rigid translation keeps the volume', True)
    except ValueError as error:
        checks.add('gauge: a rigid translation keeps the volume', False, message=str(error))
    expansion = np.zeros(operators.n_unknowns)
    expansion[2 * boundary] = _GAUGE_STRAIN * points[:, 0]
    expansion[2 * boundary + 1] = _GAUGE_STRAIN * points[:, 1]
    checks.add_raises('gauge: a uniform expansion is refused',
                      lambda: operators.check_held_volumes(expansion, lambda: 'enclosed-volume self-test'))
    # two touching incompressible liquids (an oil drop of 4x viscosity in the disc, water around it): neither group is
    # enclosed alone (the interface displacements are free), their union is: one gauge for both
    centroids = mesh.points[mesh.triangles].mean(axis=1)
    drop = np.hypot(centroids[:, 0] - _PATCH_CENTRE, centroids[:, 1] - _PATCH_CENTRE) < _PATCH_DROP_RADIUS
    indices = drop.astype(np.int64)
    oil = ViscoelasticGroup(1, drop, ModulusLaw(0.0, 0.0, 4e-3, ()), ModulusLaw(math.inf, 0.0, 0.0, ()))
    water = ViscoelasticGroup(0, ~drop, ModulusLaw(0.0, 0.0, 1e-3, ()), ModulusLaw(math.inf, 0.0, 0.0, ()))
    deviatoric = [matrices[0] for matrices in assemble_elastic_stiffness_groups(
        dofs, [unit_operator_coefficients('shear', n_elements, False)], indices, 2)]
    space = pressure_space(dofs, indices, 2)
    coupling, masses = assemble_pressure_terms(dofs, space)
    operators = mixed_harmonic_operators((water, oil), deviatoric, coupling, masses, space,
                                         scipy.sparse.identity(n_u, format='csr'), free,
                                         np.ones(space.n_dofs, dtype=bool), None, lambda: 'enclosed-volume self-test')
    checks.add('gauge: two touching liquids pinned once (their union)', len(operators.pressure_gauges) == 1
               and len(operators.volume_checks) == 1, value=len(operators.pressure_gauges),
               drop_elements=int(drop.sum()))
    try:
        operators.check_held_volumes(translation, lambda: 'enclosed-volume self-test')
        checks.add('gauge: two touching liquids, a rigid translation keeps the volume', True)
    except ValueError as error:
        checks.add('gauge: two touching liquids, a rigid translation keeps the volume', False, message=str(error))
    checks.add_raises('gauge: two touching liquids, a uniform expansion is refused',
                      lambda: operators.check_held_volumes(expansion, lambda: 'enclosed-volume self-test'))


def _check_rules(_):
    from ...model.validate import validate
    checks = SubChecks()
    for name, (spec, level, phrase) in _rule_cases().items():
        issues = validate(spec)
        if level is None:
            found = [issue.message for issue in issues if issue.level == 'error']
            checks.add(name, not found, found=found[:2])
        else:
            found = [issue.message for issue in issues if issue.level == level and phrase in issue.message]
            checks.add(name, bool(found), found=found[:1],
                       other=[issue.message for issue in issues if issue.level == level][:2] if not found else [])
    _gauge_checks(checks)
    return checks.result()


# ---- solid_viscoelastic_strip: 1D P-wave in a viscoelastic strip (rollers on x_lo, traction t on x_hi) ----
# M*(w) = K*(w) + 4/3 G*(w), G* = G0 (1 + j eta) + j w mu + G1 j w tau1 / (1 + j w tau1), K* = K0 (1 + j eta) + j w mu_b
# + K1 j w tauK / (1 + j w tauK), k = w sqrt(rho / M*) (exp(+j w t); even in k: the branch is irrelevant): u(x) = t
# sin(k x) / (M* k cos kL), sigma_xx = M* u' = t cos(k x) / cos(kL), the x_lo roller reaction -t A / cos(kL) (A = W per
# m of depth, pi W^2 whole-ring axisymmetric, W^2 in 3D). Two frequencies per run (0.5 and 1.3 f1, f1 = sqrt(M0 / rho)
# / (4 L), M0 = K0 + 4/3 G0): the frequency probes and the reaction at both, the stress at the last (the live one).
_STRIP = (10e-3, 1e-3)                  # L, W [m]
_STRIP_H = 0.5e-3
_STRIP_TRACTION = 1.0                   # Pa
_STRIP_FACTORS = (0.5, 1.3)             # x f1
_STRIP_MATERIAL = {name: float32_value(value) for name, value in (
    ('rho', 1000.0), ('shear_modulus', 1e6), ('bulk_modulus', 1e7), ('shear_relaxation_modulus_1', 2e6),
    ('shear_relaxation_time_1', 5e-5), ('viscosity', 20.0), ('bulk_relaxation_modulus_1', 5e6),
    ('bulk_relaxation_time_1', 2e-4), ('bulk_viscosity', 10.0), ('mechanical_loss_factor', 0.02))}
_STRIP_RUNS = ('triangles', 'triangles_mixed', 'axisymmetric', 'tetrahedra', 'tetrahedra_mixed')
_STRIP_TOLERANCE = {'u': 2e-3, 'stress': 1e-2, 'reaction': 2e-3}   # of |u(L)|, |t|, relative (first estimates)


def _strip_moduli(omega):
    """(G*, K*) [Pa] of the strip material at omega [rad/s]."""
    m = _STRIP_MATERIAL
    jw = 1j * omega
    shear = (m['shear_modulus'] * (1.0 + 1j * m['mechanical_loss_factor']) + jw * m['viscosity']
             + m['shear_relaxation_modulus_1'] * jw * m['shear_relaxation_time_1']
             / (1.0 + jw * m['shear_relaxation_time_1']))
    bulk = (m['bulk_modulus'] * (1.0 + 1j * m['mechanical_loss_factor']) + jw * m['bulk_viscosity']
            + m['bulk_relaxation_modulus_1'] * jw * m['bulk_relaxation_time_1']
            / (1.0 + jw * m['bulk_relaxation_time_1']))
    return shear, bulk


def _strip_frequencies():
    m = _STRIP_MATERIAL
    first = math.sqrt((m['bulk_modulus'] + 4.0 * m['shear_modulus'] / 3.0) / m['rho']) / (4.0 * _STRIP[0])
    return [factor * first for factor in _STRIP_FACTORS]


def _strip_closed_form(frequency, x):
    """(u(x), sigma(x), cos kL) of the strip at frequency [Hz]."""
    omega = 2.0 * math.pi * frequency
    shear, bulk = _strip_moduli(omega)
    modulus = bulk + 4.0 * shear / 3.0
    wavenumber = omega * np.sqrt(_STRIP_MATERIAL['rho'] / modulus)
    end = np.cos(wavenumber * _STRIP[0])
    displacement = _STRIP_TRACTION * np.sin(wavenumber * x) / (modulus * wavenumber * end)
    stress = _STRIP_TRACTION * np.cos(wavenumber * x) / end
    return displacement, stress, end


def _strip_spec(run):
    length, width = _STRIP
    material = [MaterialSpec('gel', properties=dict(_STRIP_MATERIAL), selection=['all'])]
    study = StudySpec(type=FREQUENCY_RESPONSE_STUDY, settings={'frequencies': _strip_frequencies()})
    probes = [FeatureSpec('probe_mid', 'frequency_probe', ['mid']), FeatureSpec('probe_end', 'frequency_probe', ['end'])]
    settings = {**_QUADRATIC, **(_MIXED if run.endswith('_mixed') else {})}
    if run.startswith('tetrahedra'):
        spec = _solid_spec(
            f'Self-test: viscoelastic strip on tetrahedra ({run})', (length, width, width), material,
            [FeatureSpec('anchor', 'roller', ['x_lo']), FeatureSpec('rails', 'roller', ['y_lo', 'y_hi', 'z_lo', 'z_hi']),
             FeatureSpec('push', 'boundary_load', ['x_hi'], {'t_x': _STRIP_TRACTION, 't_y': 0.0, 't_z': 0.0}), *probes],
            [_point('mid', 0.5 * length, 0.5 * width, 0.5 * width), _point('end', length, 0.5 * width, 0.5 * width)],
            cell=_STRIP_H, study=study, settings=settings)
        spec.derived_values = [DerivedValueSpec('stress_mid', 'stress_xx', [], 'at', point='mid')]
        return spec
    axisymmetric = run == 'axisymmetric'
    if axisymmetric:   # r in [0, W] (axis x_lo), z in [0, L]: rollers on r = W and z = 0, traction on z = L
        domain = DomainSpec(size_x=width, size_y=length)
        points = [GeometrySpec('mid', 'point', {'x': 0.5 * width, 'y': 0.5 * length}),
                  GeometrySpec('end', 'point', {'x': 0.5 * width, 'y': length})]
        features = [FeatureSpec('anchor', 'roller', ['y_lo']), FeatureSpec('rails', 'roller', ['x_hi']),
                    FeatureSpec('push', 'boundary_load', ['y_hi'], {'t_x': 0.0, 't_y': _STRIP_TRACTION})]
        settings = {**settings, 'geometry': 'axisymmetric'}
    else:
        domain = DomainSpec(size_x=length, size_y=width)
        points = [GeometrySpec('mid', 'point', {'x': 0.5 * length, 'y': 0.5 * width}),
                  GeometrySpec('end', 'point', {'x': length, 'y': 0.5 * width})]
        features = [FeatureSpec('anchor', 'roller', ['x_lo']), FeatureSpec('rails', 'roller', ['y_lo', 'y_hi']),
                    FeatureSpec('push', 'boundary_load', ['x_hi'], {'t_x': _STRIP_TRACTION, 't_y': 0.0})]
    return ModelSpec(
        title=f'Self-test: viscoelastic strip on triangles ({run})', domain=domain, geometry=points,
        materials=material, physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings=settings,
                                                 features=features + probes)],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_STRIP_H)],
        solver_defaults=cpu_defaults(), study=study,
        derived_values=[DerivedValueSpec('stress_mid', 'stress_yy' if axisymmetric else 'stress_xx', [], 'at',
                                         point='mid')])


def _check_strip(results_by_name):
    length, width = _STRIP
    frequencies = _strip_frequencies()
    errors, tolerances, details = {}, {}, {}

    def add(item, error, kind):
        errors[item], tolerances[item] = float(error), _STRIP_TOLERANCE[kind]

    for run, results in results_by_name.items():
        axial = 'u_y' if run == 'axisymmetric' else 'u_x'
        force = 'F_y' if run == 'axisymmetric' else 'F_x'
        area = math.pi * width ** 2 if run == 'axisymmetric' else (width ** 2 if run.startswith('tetrahedra')
                                                                   else width)
        _, reaction = _reaction(results, 'anchor', force)
        record = details[run] = {}
        for index, frequency in enumerate(sorted(frequencies)):
            scale = abs(_strip_closed_form(frequency, length)[0])
            for feature, fraction in (('probe_mid', 0.5), ('probe_end', 1.0)):
                expected = _strip_closed_form(frequency, fraction * length)[0]
                measured = _probe(results, feature, axial)[1][index]
                add(f'{run} {feature} {frequency:.6g} Hz', abs(measured - expected) / scale, 'u')
                record[f'{feature} {frequency:.6g} Hz'] = {'measured': _pair(measured),
                                                           'expected': _pair(expected)}
            expected_force = -_STRIP_TRACTION * area / _strip_closed_form(frequency, length)[2]
            add(f'{run} reaction {frequency:.6g} Hz', abs(reaction[index] - expected_force) / abs(expected_force),
                'reaction')
            record[f'reaction {frequency:.6g} Hz'] = {'measured': _pair(reaction[index]),
                                                      'expected': _pair(expected_force)}
        last = frequencies[-1]
        stress = _derived(results, 'stress_mid')
        expected_stress = float(_strip_closed_form(last, 0.5 * length)[1].real)
        add(f'{run} stress at L / 2 ({last:.6g} Hz)', abs(stress - expected_stress) / abs(_STRIP_TRACTION), 'stress')
        record['stress_mid'] = {'measured': stress, 'expected': expected_stress}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_stokes_second_problem: a viscous liquid layer (G0 = 0, incompressible) over a plate oscillating in-plane --
# Layer 0 < y < H, the bottom moved by u_x = X, the top fixed, the side faces hold u_y only (the exact shear flow has p =
# 0 and sigma_xx = 0 there). G* = j w mu: u_x(y) = X sinh(lam (H - y)) / sinh(lam H), lam = sqrt(j w rho / mu) (the
# principal root, Re lam > 0), sigma_xy = j w mu u_x', the bottom reaction F_x = j w mu lam coth(lam H) X A (A = the
# bottom length per m of depth, its area in 3D). Two frequencies per run with delta = sqrt(2 mu / (rho w)) = H / 3 and
# H / 8: the probes and the reaction at both, the stress and the pressure at the last. 3D: the layer normal along z.
# GPU: Automatic 'linear' -> cuDSS LDL^T of the complex symmetric saddle point with its zero pressure block; the cpu
# variant: host SuperLU of the equilibrated saddle point R A R (unscaled, its diagonal pivot test across the u / p units
# loses the pressure field while the reactions hold: max |p| is the item that catches it).
# Two-layer runs (triangles): the liquid below y = h1 = _STOKES_INTERFACE H, a second incompressible liquid (viscosity
# 4x, rho 800: its own pressure group, the groups touch along y = h1) above. 'two_layers': the same holds; [u_x,
# sigma_xy] crosses a layer of depth d by T = [[cosh(lam d), sinh(lam d) / Z], [Z sinh(lam d), cosh(lam d)]], Z = j w
# mu lam (u and sigma_xy continuous at h1, p = 0 exact), u(H) = 0 of T2 T1 [X, sigma_xy(0)] fixes sigma_xy(0), the
# bottom reaction is -sigma_xy(0) A. 'enclosed_layers': rollers on the sides and both walls moved by u_y = X, so every
# normal displacement is held: q = 1 over BOTH groups is a null vector (the interface terms of the groups cancel),
# neither group alone is (solver_common._pressure_gauges pins the union region; without it the saddle point is
# singular); the held translation keeps the volume (check_held_volumes passes). Exact: u = (0, X), dp/dy = rho w^2 X
# per layer (p continuous at h1), the summed wall reactions R_y = -w^2 X W (rho1 h1 + rho2 (H - h1)); P2 / P1 hold it,
# so only round-off and float32 field storage remain.
_STOKES_H = 100e-6
_STOKES_WIDTH = _STOKES_H / 4.0
_STOKES_X = 1e-9                        # m
_STOKES_VISCOSITY = float32_value(1e-3)
_STOKES_DENSITY = float32_value(1000.0)
_STOKES_DEPTHS = (1.0 / 3.0, 1.0 / 8.0)   # delta / H
_STOKES_FRACTIONS = (0.25, 0.5, 0.75)   # y / H of the probes
_STOKES_H_TRIANGLES = _STOKES_H / 40.0
_STOKES_H_TETRAHEDRA = _STOKES_H / 32.0
# first estimates; pressure: max |p| of the exact-zero field is P2 / P1 discretisation of the exponential profile on an
# unstructured mesh (cpu SuperLU runs: 3.1e-3 of the wall shear at H / 40 on triangles and two_layers, 3.9e-3 on
# tetrahedra at H / 32, 8.0e-4 at H / 80: ~4x per halving), not solver error
_STOKES_TOLERANCE = {'u': 2e-3, 'reaction': 2e-3, 'stress': 1e-2, 'pressure': 1e-2}
_STOKES_INTERFACE = 0.4                 # h1 / H of the two-layer runs (between the probes, on a mesh line)
_STOKES_UPPER_VISCOSITY = float32_value(4e-3)
_STOKES_UPPER_DENSITY = float32_value(800.0)
_STOKES_LAYER_RUNS = ('two_layers', 'enclosed_layers')
_STOKES_RUNS = ('triangles', 'tetrahedra', *_STOKES_LAYER_RUNS)
# the cpu variant leaves out tetrahedra: a 3D P2 saddle point on host SuperLU (12288 tets: 67 s, 2.0 GB factor)
_STOKES_CPU_RUNS = tuple(run for run in _STOKES_RUNS if run != 'tetrahedra')
# enclosed_layers (exact up to round-off): of X, relative, of the pressure rise (float32 fields); first estimates
_STOKES_ENCLOSED_TOLERANCE = {'u': 1e-6, 'reaction': 1e-6, 'pressure': 1e-5}


def _stokes_frequencies():
    return [_STOKES_VISCOSITY / (math.pi * _STOKES_DENSITY * (depth * _STOKES_H) ** 2) for depth in _STOKES_DEPTHS]


def _stokes_closed_form(frequency, y):
    """(u_x(y), sigma_xy(y), lam) of the layer at frequency [Hz]."""
    omega = 2.0 * math.pi * frequency
    lam = np.sqrt(1j * omega * _STOKES_DENSITY / _STOKES_VISCOSITY)
    displacement = _STOKES_X * np.sinh(lam * (_STOKES_H - y)) / np.sinh(lam * _STOKES_H)
    slope = -_STOKES_X * lam * np.cosh(lam * (_STOKES_H - y)) / np.sinh(lam * _STOKES_H)
    return displacement, 1j * omega * _STOKES_VISCOSITY * slope, lam


def _layer_transfer(omega, viscosity, density, depth):
    """2 x 2 complex transfer matrix of [u_x, sigma_xy] across a liquid layer of `depth` [m] (G* = j w mu): u = A
    cosh(lam s) + B sinh(lam s), sigma_xy = Z (A sinh + B cosh), Z = j w mu lam, lam = sqrt(j w rho / mu)."""
    lam = np.sqrt(1j * omega * density / viscosity)
    impedance = 1j * omega * viscosity * lam
    return np.array([[np.cosh(lam * depth), np.sinh(lam * depth) / impedance],
                     [impedance * np.sinh(lam * depth), np.cosh(lam * depth)]])


def _layers_closed_form(frequency, y):
    """(u_x(y), sigma_xy(y), sigma_xy(0)) of the two-layer shear wave at frequency [Hz] (case comment)."""
    omega = 2.0 * math.pi * frequency
    interface = _STOKES_INTERFACE * _STOKES_H
    lower = _layer_transfer(omega, _STOKES_VISCOSITY, _STOKES_DENSITY, interface)
    whole = _layer_transfer(omega, _STOKES_UPPER_VISCOSITY, _STOKES_UPPER_DENSITY, _STOKES_H - interface) @ lower
    bottom = np.array([_STOKES_X, -whole[0, 0] * _STOKES_X / whole[0, 1]])   # u(H) = 0 fixes sigma_xy(0)
    if y <= interface:
        state = _layer_transfer(omega, _STOKES_VISCOSITY, _STOKES_DENSITY, y) @ bottom
    else:
        state = _layer_transfer(omega, _STOKES_UPPER_VISCOSITY, _STOKES_UPPER_DENSITY, y - interface) @ lower @ bottom
    return state[0], state[1], bottom[1]


def _enclosed_pressure_rise(frequency, low, high):
    """p(high) - p(low) [Pa] of the enclosed two-layer translation (dp/dy = rho w^2 X per layer, case comment)."""
    omega = 2.0 * math.pi * frequency
    interface = _STOKES_INTERFACE * _STOKES_H

    def level(y):
        return omega ** 2 * _STOKES_X * (_STOKES_DENSITY * min(y, interface)
                                         + _STOKES_UPPER_DENSITY * max(y - interface, 0.0))
    return level(high) - level(low)


def _layered_spec(run, study, probes, gpu):
    """The two-layer runs on triangles: 'two_layers' (the shear wave, the side and wall holds of the triangles run)
    and 'enclosed_layers' (rollers on the sides, both walls moved by u_y = X: every normal displacement held)."""
    middle = 0.5 * _STOKES_WIDTH
    interface = _STOKES_INTERFACE * _STOKES_H
    materials = [MaterialSpec('liquid', properties={'viscosity': _STOKES_VISCOSITY, 'rho': _STOKES_DENSITY},
                              selection=['lower']),
                 MaterialSpec('upper_liquid', properties={'viscosity': _STOKES_UPPER_VISCOSITY,
                                                          'rho': _STOKES_UPPER_DENSITY}, selection=['upper'])]
    if run == 'two_layers':
        features = [FeatureSpec('sides', 'displacement', ['x_lo', 'x_hi'], {'u_y': 0.0}),
                    FeatureSpec('bottom', 'displacement', ['y_lo'], {'u_x': _STOKES_X, 'u_y': 0.0}),
                    FeatureSpec('top', 'fixed', ['y_hi'])]
        derived = [DerivedValueSpec('stress_mid', 'stress_xy', [], 'at', point='p1'),
                   DerivedValueSpec('pressure_max', f'sqrt({_PHYSICS}_pressure**2 + {_PHYSICS}_pressure_im**2)',
                                    [], 'max')]
    else:
        features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi']),
                    FeatureSpec('bottom', 'displacement', ['y_lo'], {'u_y': _STOKES_X}),
                    FeatureSpec('top', 'displacement', ['y_hi'], {'u_y': _STOKES_X})]
        derived = [DerivedValueSpec(f'pressure_p{index}', f'{_PHYSICS}_pressure', [], 'at', point=f'p{index}')
                   for index in (0, len(_STOKES_FRACTIONS) - 1)]
    return ModelSpec(
        title=f'Self-test: Stokes second problem, two liquid layers ({run})',
        domain=DomainSpec(size_x=_STOKES_WIDTH, size_y=_STOKES_H),
        geometry=[_rectangle('lower', 0.0, 0.0, _STOKES_WIDTH, interface),
                  _rectangle('upper', 0.0, interface, _STOKES_WIDTH, _STOKES_H),
                  *(GeometrySpec(f'p{index}', 'point', {'x': middle, 'y': fraction * _STOKES_H})
                    for index, fraction in enumerate(_STOKES_FRACTIONS))],
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={**_QUADRATIC, **_MIXED}, features=features + probes)],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_STOKES_H_TRIANGLES)],
        solver_defaults=gpu, study=study, derived_values=derived)


def _stokes_spec(run):
    material = [MaterialSpec('liquid', properties={'viscosity': _STOKES_VISCOSITY, 'rho': _STOKES_DENSITY},
                             selection=['all'])]
    study = StudySpec(type=FREQUENCY_RESPONSE_STUDY, settings={'frequencies': _stokes_frequencies()})
    probes = [FeatureSpec(f'probe_{index}', 'frequency_probe', [f'p{index}'])
              for index in range(len(_STOKES_FRACTIONS))]
    pressure = f'sqrt({_PHYSICS}_pressure**2 + {_PHYSICS}_pressure_im**2)'
    gpu = SolverDefaultsSpec(backend='gpu')
    if run in _STOKES_LAYER_RUNS:
        return _layered_spec(run, study, probes, gpu)
    if run == 'tetrahedra':
        middle = 0.5 * _STOKES_WIDTH
        spec = _solid_spec(
            'Self-test: Stokes second problem on tetrahedra', (_STOKES_WIDTH, _STOKES_WIDTH, _STOKES_H), material,
            [FeatureSpec('xsides', 'displacement', ['x_lo', 'x_hi'], {'u_z': 0.0}),
             FeatureSpec('ysides', 'roller', ['y_lo', 'y_hi']),
             FeatureSpec('bottom', 'displacement', ['z_lo'], {'u_x': _STOKES_X, 'u_y': 0.0, 'u_z': 0.0}),
             FeatureSpec('top', 'fixed', ['z_hi']), *probes],
            [_point(f'p{index}', middle, middle, fraction * _STOKES_H)
             for index, fraction in enumerate(_STOKES_FRACTIONS)], cell=_STOKES_H_TETRAHEDRA, study=study,
            settings={**_QUADRATIC, **_MIXED})
        spec.solver_defaults = gpu
        spec.derived_values = [DerivedValueSpec('stress_mid', 'stress_xz', [], 'at', point='p1'),
                               DerivedValueSpec('pressure_max', pressure, [], 'max')]
        return spec
    middle = 0.5 * _STOKES_WIDTH
    return ModelSpec(
        title='Self-test: Stokes second problem on triangles', domain=DomainSpec(size_x=_STOKES_WIDTH, size_y=_STOKES_H),
        geometry=[GeometrySpec(f'p{index}', 'point', {'x': middle, 'y': fraction * _STOKES_H})
                  for index, fraction in enumerate(_STOKES_FRACTIONS)],
        materials=material,
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={**_QUADRATIC, **_MIXED}, features=[
            FeatureSpec('sides', 'displacement', ['x_lo', 'x_hi'], {'u_y': 0.0}),
            FeatureSpec('bottom', 'displacement', ['y_lo'], {'u_x': _STOKES_X, 'u_y': 0.0}),
            FeatureSpec('top', 'fixed', ['y_hi']), *probes])],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_STOKES_H_TRIANGLES)],
        solver_defaults=gpu, study=study,
        derived_values=[DerivedValueSpec('stress_mid', 'stress_xy', [], 'at', point='p1'),
                        DerivedValueSpec('pressure_max', pressure, [], 'max')])


def _check_stokes(results_by_name):
    frequencies = sorted(_stokes_frequencies())
    errors, tolerances, details = {}, {}, {}

    def add(item, error, kind, table=_STOKES_TOLERANCE):
        errors[item], tolerances[item] = float(error), table[kind]

    for run, results in results_by_name.items():
        if run == 'enclosed_layers':
            details[run] = _check_enclosed_layers(results, frequencies, add)
            continue
        if run == 'two_layers':
            details[run] = _check_two_layers(results, frequencies, add)
            continue
        area = _STOKES_WIDTH ** 2 if run == 'tetrahedra' else _STOKES_WIDTH
        _, reaction = _reaction(results, 'bottom', 'F_x')
        record = details[run] = {}
        for index, frequency in enumerate(frequencies):
            omega = 2.0 * math.pi * frequency
            for probe, fraction in enumerate(_STOKES_FRACTIONS):
                expected = _stokes_closed_form(frequency, fraction * _STOKES_H)[0]
                measured = _probe(results, f'probe_{probe}', 'u_x')[1][index]
                add(f'{run} u_x at y = {fraction:g} H, {frequency:.6g} Hz', abs(measured - expected) / _STOKES_X, 'u')
            lam = _stokes_closed_form(frequency, 0.0)[2]
            expected_force = 1j * omega * _STOKES_VISCOSITY * lam / np.tanh(lam * _STOKES_H) * _STOKES_X * area
            add(f'{run} bottom reaction {frequency:.6g} Hz',
                abs(reaction[index] - expected_force) / abs(expected_force), 'reaction')
            record[f'reaction {frequency:.6g} Hz'] = {'measured': _pair(reaction[index]),
                                                      'expected': _pair(expected_force)}
        last = frequencies[-1]
        omega = 2.0 * math.pi * last
        lam = _stokes_closed_form(last, 0.0)[2]
        scale = omega * _STOKES_VISCOSITY * abs(lam) * _STOKES_X   # the wall shear stress amplitude
        expected_stress = float(_stokes_closed_form(last, 0.5 * _STOKES_H)[1].real)
        stress = _derived(results, 'stress_mid')
        add(f'{run} stress at H / 2', abs(stress - expected_stress) / scale, 'stress')
        add(f'{run} pressure (exact 0)', _derived(results, 'pressure_max') / scale, 'pressure')
        record['stress_mid'] = {'measured': stress, 'expected': expected_stress, 'scale': scale}
    return scaled_worst_result(errors, tolerances, **details)


def _check_two_layers(results, frequencies, add) -> dict:
    """The two-layer shear wave against the transfer-matrix solution (_layers_closed_form): probes, the bottom
    reaction at both frequencies, Re sigma_xy at H / 2 (the upper layer) and max |p| at the last; -> details."""
    run, record = 'two_layers', {}
    _, reaction = _reaction(results, 'bottom', 'F_x')
    for index, frequency in enumerate(frequencies):
        for probe, fraction in enumerate(_STOKES_FRACTIONS):
            expected = _layers_closed_form(frequency, fraction * _STOKES_H)[0]
            measured = _probe(results, f'probe_{probe}', 'u_x')[1][index]
            add(f'{run} u_x at y = {fraction:g} H, {frequency:.6g} Hz', abs(measured - expected) / _STOKES_X, 'u')
        expected_force = -_layers_closed_form(frequency, 0.0)[2] * _STOKES_WIDTH
        add(f'{run} bottom reaction {frequency:.6g} Hz', abs(reaction[index] - expected_force) / abs(expected_force),
            'reaction')
        record[f'reaction {frequency:.6g} Hz'] = {'measured': _pair(reaction[index]), 'expected': _pair(expected_force)}
    last = frequencies[-1]
    _, middle, wall = _layers_closed_form(last, 0.5 * _STOKES_H)
    scale = abs(wall)   # the wall shear stress amplitude
    stress = _derived(results, 'stress_mid')
    add(f'{run} stress at H / 2', abs(stress - float(middle.real)) / scale, 'stress')
    add(f'{run} pressure (exact 0)', _derived(results, 'pressure_max') / scale, 'pressure')
    record['stress_mid'] = {'measured': stress, 'expected': float(middle.real), 'scale': scale}
    return record


def _check_enclosed_layers(results, frequencies, add) -> dict:
    """The enclosed two-layer translation (exact): u_y = X at the probes, the summed wall reactions -w^2 X W (rho1 h1 +
    rho2 (H - h1)) at both frequencies, the pressure rise between the outer probes at the last; -> details."""
    run, record = 'enclosed_layers', {}
    interface = _STOKES_INTERFACE * _STOKES_H
    mass = _STOKES_WIDTH * (_STOKES_DENSITY * interface + _STOKES_UPPER_DENSITY * (_STOKES_H - interface))   # kg/m
    bottom, top = _reaction(results, 'bottom', 'F_y')[1], _reaction(results, 'top', 'F_y')[1]
    for index, frequency in enumerate(frequencies):
        omega = 2.0 * math.pi * frequency
        for probe, fraction in enumerate(_STOKES_FRACTIONS):
            measured = _probe(results, f'probe_{probe}', 'u_y')[1][index]
            add(f'{run} u_y at y = {fraction:g} H, {frequency:.6g} Hz', abs(measured - _STOKES_X) / _STOKES_X, 'u',
                _STOKES_ENCLOSED_TOLERANCE)
        expected_force = -omega ** 2 * _STOKES_X * mass
        total = bottom[index] + top[index]
        add(f'{run} wall reactions {frequency:.6g} Hz', abs(total - expected_force) / abs(expected_force), 'reaction',
            _STOKES_ENCLOSED_TOLERANCE)
        record[f'wall reactions {frequency:.6g} Hz'] = {'measured': _pair(total), 'expected': expected_force}
    low, high = _STOKES_FRACTIONS[0] * _STOKES_H, _STOKES_FRACTIONS[-1] * _STOKES_H
    expected_rise = _enclosed_pressure_rise(frequencies[-1], low, high)
    rise = _derived(results, f'pressure_p{len(_STOKES_FRACTIONS) - 1}') - _derived(results, 'pressure_p0')
    add(f'{run} pressure rise p(3 H / 4) - p(H / 4)', abs(rise - expected_rise) / abs(expected_rise), 'pressure',
        _STOKES_ENCLOSED_TOLERANCE)
    record['pressure rise'] = {'measured': rise, 'expected': expected_rise}
    return record


# ---- solid_oscillating_sphere_drag: a rigid bead oscillating in water (axisymmetric, mixed) ----
# Bead radius a at (0, R) of an r < R, 0 < z < 2 R container of library water (formulation mixed: incompressible,
# viscous), walls fixed (x_hi, y_lo, y_hi; x_lo is the axis); the bead (E 3.5e9, nu 0.34, rho_b: a finite bulk modulus,
# so its pressure block is regular) held by prescribed_displacement_domain: m0 (azimuthal_order 0) moves it by X along
# z, m1 (azimuthal_order 1) by X along x (u_x = X, u_z = -X: U_r = X, U_phi = -X). Reference (Stokes 1851, Landau-
# Lifshitz section 24; exp(+j w t), U = j w X): the force of the liquid on the bead F = -6 pi mu a U (1 + lam a + lam^2
# a^2 / 9), lam = sqrt(j w rho / mu) = (1 + j) / delta with Re lam > 0. Sign check: lam^2 a^2 / 9 gives -(2/3) pi a^3
# rho j w U = -(1/2) m_fluid dU/dt (the added mass opposes the acceleration) and the real part -6 pi mu a (1 + a /
# delta) U the drag opposing the velocity (Landau-Lifshitz's exp(-i w t) form with i -> -j). The support reaction R
# (frequency_response.reaction_table: sum of A x - f over the held dofs) is the force the support exerts on the bead, and
# the bead moves rigidly: -w^2 m_b X = R + F, so F_fem = -R - w^2 m_b X (m_b = rho_b 4/3 pi a^3, whole ring; at m = 1
# R = F_x = sum R_r - sum R_phi, the net force along x). Frequencies a / delta = 0.5 and 2. Wall corrections: the
# vortical part decays as exp(-(R - a) / delta), the potential dipole as (a / R)^3.
# Pressure (same solution, p > 0 in compression): the linearised momentum rho j w u = -grad p + mu lap u with div u = 0
# makes p harmonic; u = curl curl (f(r) U) gives p = mu U . grad(lap f - lam^2 f), and f = -c1 / (lam^2 r) + b
# exp(-lam r) / r (lap f - lam^2 f = c1 / r). u = U on r = a (u_r = -2 U cos(theta) f' / r, u_theta = U sin(theta) (f''
# + f' / r): f'(a) = -a / 2, f''(a) = -1 / 2) fixes c1 = -(3 a / 2) (1 + lam a + lam^2 a^2 / 3), so p(r, theta) =
# (3 / 2) mu a U (1 + lam a + lam^2 a^2 / 3) cos(theta) / r^2, theta from the direction of motion. Checks: w -> 0 gives
# Stokes' (3 / 2) mu a U cos(theta) / r^2; lam -> inf the potential dipole rho j w U a^3 cos(theta) / (2 r^2) (p = -rho
# d phi / dt); - int p cos(theta) dS on r = a = -2 pi mu a U (1 + lam a + lam^2 a^2 / 3): one third of the Stokes drag
# and the whole added-mass term of F above. Read at r = 1.5 a in the water (the nodal display averages the bead's own
# pressure group at the surface nodes) at the last frequency (the live one): m0 the axis points z = R +- 1.5 a as the
# difference p(front) - p(back) (the enclosed water's pressure gauge, a wall dof, shifts p by a constant; the box is
# symmetric about z = R, so the difference is free of it), m1 the cos(phi) amplitude at r = 1.5 a, z = R (no gauge at
# m >= 1; u_r = X cos(phi) moves the bead towards phi = 0, the front). Wall reflection of the dipole ~ 2 (1.5 a / R)^3.
_SPHERE_RADIUS = 1e-6
_SPHERE_CONTAINER = 12.0                # R / a
_SPHERE_X = 1e-9                        # m
_SPHERE_RATIOS = (0.5, 2.0)             # a / delta
_SPHERE_BEAD = {'youngs_modulus': float32_value(3.5e9), 'poisson_ratio': float32_value(0.34),
                'rho': float32_value(1050.0)}
_SPHERE_RUNS = ('m0', 'm1')
_SPHERE_TOLERANCE = {'drag': 2e-2, 'm0 vs m1': 5e-3, 'axis pressure': 1e-9, 'pressure': 2e-2}   # first estimates
_SPHERE_PRESSURE_DISTANCE = 1.5         # r / a of the pressure points (front, back on the axis, side at z = R)


def _water_constants():
    """(mu [Pa s], rho [kg/m^3]) of library water as the MaterialGrid stores them (float32)."""
    from ...materials import effective_material_data
    properties, _ = effective_material_data('water', {}, None)
    return float32_value(properties['viscosity']), float32_value(properties['rho'])


def _sphere_frequencies():
    viscosity, density = _water_constants()
    return [viscosity * ratio ** 2 / (math.pi * density * _SPHERE_RADIUS ** 2) for ratio in _SPHERE_RATIOS]


def _sphere_spec(run):
    radius = _SPHERE_RADIUS
    container = _SPHERE_CONTAINER * radius
    held = ({'u_x': 0.0, 'u_y': _SPHERE_X} if run == 'm0'
            else {'u_x': _SPHERE_X, 'u_y': 0.0, 'u_z': -_SPHERE_X})
    pressure = f'sqrt({_PHYSICS}_pressure**2 + {_PHYSICS}_pressure_im**2)'
    distance = _SPHERE_PRESSURE_DISTANCE * radius
    points = {'front': (0.0, container + distance), 'back': (0.0, container - distance), 'side': (distance, container)}
    parts = {'re': f'{_PHYSICS}_pressure', 'im': f'{_PHYSICS}_pressure_im'}
    return ModelSpec(
        title=f'Self-test: sphere oscillating in water ({run})',
        domain=DomainSpec(size_x=container, size_y=2.0 * container),
        geometry=[GeometrySpec('bead', 'circle', {'cx': 0.0, 'cy': container, 'r': radius}),
                  GeometrySpec('axis_point', 'point', {'x': 0.0, 'y': container + 3.0 * radius}),
                  *(GeometrySpec(f'{name}_point', 'point', {'x': x, 'y': y}) for name, (x, y) in points.items())],
        materials=[MaterialSpec('water', library='water', selection=['all']),
                   MaterialSpec('bead', properties=dict(_SPHERE_BEAD), selection=['bead'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={
            **_QUADRATIC, **_MIXED, 'geometry': 'axisymmetric', 'azimuthal_order': 0 if run == 'm0' else 1},
            features=[FeatureSpec('walls', 'fixed', ['x_hi', 'y_lo', 'y_hi']),
                      FeatureSpec('bead_drive', 'prescribed_displacement_domain', ['bead'], held)])],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='nonuniform', dx=container / 8.0, max_ratio=1.25,
                         refinements=[RefinementSpec('bead', radius / (4.0 * max(_SPHERE_RATIOS)))])],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type=FREQUENCY_RESPONSE_STUDY, settings={'frequencies': _sphere_frequencies()}),
        derived_values=[DerivedValueSpec('pressure_axis', pressure, [], 'at', point='axis_point'),
                        DerivedValueSpec('pressure_max', pressure, [], 'max'),
                        *(DerivedValueSpec(f'pressure_{name}_{part}', field, [], 'at', point=f'{name}_point')
                          for name in points for part, field in parts.items())])


def _sphere_pressure(frequency, distance):
    """Complex pressure amplitude [Pa] in front of the bead (theta = 0) at distance [m] from its centre (case
    comment: (3 / 2) mu a U (1 + lam a + lam^2 a^2 / 3) / r^2, U = j w X)."""
    viscosity, density = _water_constants()
    omega = 2.0 * math.pi * frequency
    lam = np.sqrt(1j * omega * density / viscosity)
    if lam.real <= 0.0:
        raise ValueError('the principal root of j omega rho / mu has Re > 0')
    radius = _SPHERE_RADIUS
    return (1.5 * viscosity * radius * 1j * omega * _SPHERE_X * (1.0 + lam * radius + (lam * radius) ** 2 / 3.0)
            / distance ** 2)


def _measured_pressure(results, name) -> complex:
    return complex(_derived(results, f'pressure_{name}_re'), _derived(results, f'pressure_{name}_im'))


def _sphere_drag(frequency):
    """Complex force [N] of the liquid on the bead at frequency [Hz] (case comment)."""
    viscosity, density = _water_constants()
    omega = 2.0 * math.pi * frequency
    lam = np.sqrt(1j * omega * density / viscosity)
    if lam.real <= 0.0:
        raise ValueError('the principal root of j omega rho / mu has Re > 0')
    velocity = 1j * omega * _SPHERE_X
    return -6.0 * math.pi * viscosity * _SPHERE_RADIUS * velocity * (
        1.0 + lam * _SPHERE_RADIUS + (lam * _SPHERE_RADIUS) ** 2 / 9.0)


def _check_sphere(results_by_name):
    frequencies = sorted(_sphere_frequencies())
    bead_mass = _SPHERE_BEAD['rho'] * 4.0 / 3.0 * math.pi * _SPHERE_RADIUS ** 3
    errors, tolerances, details, forces = {}, {}, {}, {}
    for run, results in results_by_name.items():
        _, reaction = _reaction(results, 'bead_drive', 'F_y' if run == 'm0' else 'F_x')
        omega = 2.0 * math.pi * np.asarray(frequencies)
        forces[run] = -reaction - omega ** 2 * bead_mass * _SPHERE_X
        record = details[run] = {}
        for index, frequency in enumerate(frequencies):
            expected = _sphere_drag(frequency)
            item = f'{run} drag a / delta = {_SPHERE_RATIOS[index]:g}'
            errors[item] = abs(forces[run][index] - expected) / abs(expected)
            tolerances[item] = _SPHERE_TOLERANCE['drag']
            record[f'{frequency:.6g} Hz'] = {'F_fem': _pair(forces[run][index]),
                                             'F_stokes': _pair(expected)}
        if run == 'm1':   # a cos(phi) pressure is 0 on the axis (held there)
            maximum = _derived(results, 'pressure_max')
            errors['m1 axis pressure'] = _derived(results, 'pressure_axis') / maximum if maximum > 0.0 else math.inf
            tolerances['m1 axis pressure'] = _SPHERE_TOLERANCE['axis pressure']
        # the pressure at r = 1.5 a at the last frequency (case comment): m0 front - back, m1 the side amplitude
        expected = _sphere_pressure(frequencies[-1], _SPHERE_PRESSURE_DISTANCE * _SPHERE_RADIUS)
        if run == 'm0':   # p(back) = -p(front) + the gauge constant
            measured = _measured_pressure(results, 'front') - _measured_pressure(results, 'back')
            expected = 2.0 * expected
        else:
            measured = _measured_pressure(results, 'side')
        item = f'{run} pressure at r = {_SPHERE_PRESSURE_DISTANCE:g} a'
        errors[item] = abs(measured - expected) / abs(expected)
        tolerances[item] = _SPHERE_TOLERANCE['pressure']
        record['pressure'] = {'measured': _pair(measured), 'expected': _pair(expected)}
    if len(forces) == 2:
        for index, ratio in enumerate(_SPHERE_RATIOS):
            item = f'm0 vs m1 a / delta = {ratio:g}'
            errors[item] = abs(forces['m0'][index] - forces['m1'][index]) / abs(_sphere_drag(frequencies[index]))
            tolerances[item] = _SPHERE_TOLERANCE['m0 vs m1']
    return scaled_worst_result(errors, tolerances, bead_mass_kg=bead_mass, **details)


VISCOELASTIC_CASES = (
    SelfTestCase(
        name='solid_harmonic_terms',
        label='Solid mechanics: per-frequency terms of the harmonic system against dense solves',
        description=('No model run: a synthetic 1D P1 chain pencil (K, 0.02 K loss, consistent M, node 0 held) with two '
                     'FrequencyTerms (a Maxwell-like c1(w) = 1e3 (1 + 0.3 j w tau) / (1 + j w tau) and a viscous c2 = '
                     f'j w mu on overlapping sub-blocks) and its saddle variant ({_TERMS_PRESSURE} pressure rows, '
                     '[[A, -B^T], [-B, 0]], the zero block a zero-coefficient term, a SchurScaleTerm) solved by '
                     'frequency_response.HarmonicSystem at '
                     f"{', '.join(f'{omega:g}' for omega in _TERMS_OMEGAS)} rad/s with the private SuperLU, "
                     'scipy_splu, MUMPS / PARDISO when installed (the HostPencil extra slots) and, with a GPU and '
                     'nvmath-python, cuDSS (the DevicePencil extra slots): x and A x against the dense solve '
                     f'({_TERMS_TOLERANCE:g} SuperLU, {_TERMS_DIRECT_TOLERANCE:g} refined factors; first estimates), '
                     'product_at_rows exact at the output rows and 0 elsewhere, the rows-only solve (fields False) '
                     'and its last_solution against the dense solve (the saddle variant equilibrates SuperLU too: the '
                     'scaled factor must hand out the unscaled solution), plan_reduced refusing the terms, '
                     'terms=() bit-identical to no keyword. error = number of failed sub-checks.'),
        specs=dict, check=_check_terms, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='solid_viscoelastic_validation',
        label='Solid mechanics: viscoelastic moduli and mixed formulation validation rules',
        description=('No run: one model per validation rule of definition_viscoelastic (formulation mixed on a tensor '
                     'mesh, outside frequency_response, with linear elements, piezoelectric, plane_stress, '
                     'stress_stiffening, a PML; reduced sweeps with frequency-dependent moduli and with an '
                     'incompressible material at azimuthal_order 1; one route per material, c_ij with a viscosity, '
                     'branches without times, bulk parts without a bulk modulus, a liquid in the displacement '
                     'formulation and in a static study, library water void, ignored viscosities, tensor meshes, '
                     'laminar_flow sharing the viscosity, more than 32 parameter groups, the nearly incompressible '
                     'info) must report its severity and key phrase, two valid models none; and the enclosed-volume '
                     'check on the solid_fem_patch mesh (P2, every boundary dof held, one liquid group): one gauge, a '
                     'rigid translation passes, a uniform expansion is refused (relative tolerance 1e-9, first '
                     'estimate); the same with two touching incompressible liquids (an oil drop of 4x viscosity in '
                     'water): one gauge for their union, neither being enclosed alone. error = number of failed '
                     'sub-checks.'),
        specs=dict, check=_check_rules),
    SelfTestCase(
        name='solid_viscoelastic_strip',
        label='Solid mechanics: viscoelastic P-wave strip (frequency response, displacement and mixed)',
        description=(f'A {_STRIP[0] * 1e3:g} x {_STRIP[1] * 1e3:g} mm strip (rollers on x_lo and the sides, traction '
                     f'{_STRIP_TRACTION:g} Pa on x_hi; axisymmetric: r in [0, W], rollers on r = W and z = 0, traction '
                     'on z = L) of a generalized Maxwell material (rho 1000, G0 1e6, K0 1e7, G1 2e6 tau1 5e-5 s, '
                     'viscosity 20, K1 5e6 tauK 2e-4 s, bulk_viscosity 10, mechanical_loss_factor 0.02 on G0, K0), '
                     f'P2, h = {_STRIP_H * 1e3:g} mm, cpu, at {_STRIP_FACTORS[0]:g} and {_STRIP_FACTORS[1]:g} x f1 = '
                     'sqrt(M0 / rho) / (4 L): runs triangles and triangles_mixed (plane strain), axisymmetric, '
                     'tetrahedra and tetrahedra_mixed (builtin). Reference u = t sin(k x) / (M* k cos kL), M* = K* + '
                     '4/3 G*, k = w sqrt(rho / M*): frequency probes at L / 2 and L within '
                     f"{_STRIP_TOLERANCE['u']:g} of |u(L)|, the x_lo roller reaction -t A / cos kL within "
                     f"{_STRIP_TOLERANCE['reaction']:g} relative (A = W per m, pi W^2, W^2), Re sigma at L / 2 of the "
                     f"last frequency within {_STRIP_TOLERANCE['stress']:g} of |t| (node-averaged; pins the "
                     'complex-correct stress). Tolerances are first estimates (P2 error ~ (k h)^3). measured = error = '
                     'the worst deviation in tolerance units.'),
        specs=lambda: {run: _strip_spec(run) for run in _STRIP_RUNS}, check=_check_strip),
    SelfTestCase(
        name='solid_stokes_second_problem',
        label="Solid mechanics: Stokes' second problem, a viscous liquid on an oscillating plate (mixed)",
        description=(f'A {_STOKES_H * 1e6:g} um liquid layer (viscosity {_STOKES_VISCOSITY:g} Pa s, rho '
                     f'{_STOKES_DENSITY:g}, no modulus: G* = j w mu, incompressible, formulation mixed) driven by its '
                     f'bottom moving u_x = {_STOKES_X:g} m, the top fixed, the sides holding the normal-free u_y only; '
                     'runs triangles (planar, h = H / 40) and tetrahedra (the layer along z, h = H / 32), P2, gpu '
                     "backend (Automatic 'linear': cuDSS LDL^T of the complex symmetric saddle point with its zero "
                     'pressure block; skipped without a GPU) and variant cpu (host SuperLU of the equilibrated saddle '
                     'point; the 2D runs only), at delta = sqrt(2 mu / (rho w)) = H / 3 and H / 8. '
                     'Reference u_x = X sinh(lam (H - y)) / sinh(lam H), lam = sqrt(j w rho / mu): the probes at H / '
                     f"4, H / 2, 3 H / 4 within {_STOKES_TOLERANCE['u']:g} of X, the bottom reaction j w mu lam "
                     f"coth(lam H) X A within {_STOKES_TOLERANCE['reaction']:g} relative, Re sigma_xy at H / 2 "
                     f"within {_STOKES_TOLERANCE['stress']:g} and max |p| (exact 0; P2 / P1 discretisation ~3e-3 here) "
                     f"below {_STOKES_TOLERANCE['pressure']:g} of the wall shear stress w mu |lam| X (last frequency). "
                     'Two-layer runs on triangles (the liquid '
                     f'below {_STOKES_INTERFACE:g} H, above it a second incompressible liquid of viscosity '
                     f'{_STOKES_UPPER_VISCOSITY:g} Pa s, rho {_STOKES_UPPER_DENSITY:g}): two_layers (the same holds) '
                     'against the transfer-matrix solution (per layer [[cosh(lam d), sinh(lam d) / Z], [Z sinh(lam '
                     'd), cosh(lam d)]] on [u_x, sigma_xy], Z = j w mu lam) with the tolerances above (scale: the '
                     'bottom wall shear stress); enclosed_layers (rollers on the sides, both walls moved by u_y = X: '
                     'the two touching liquids are one enclosed incompressible region, singular unless its pressure '
                     'level is pinned, and the held translation keeps its volume): u_y = X at the probes within '
                     f"{_STOKES_ENCLOSED_TOLERANCE['u']:g} of X, the summed wall reactions -w^2 X W (rho1 h1 + rho2 "
                     f"(H - h1)) within {_STOKES_ENCLOSED_TOLERANCE['reaction']:g} relative, the pressure rise "
                     'p(3 H / 4) - p(H / 4) (dp/dy = rho w^2 X per layer) within '
                     f"{_STOKES_ENCLOSED_TOLERANCE['pressure']:g} relative (exact in P2 / P1: round-off and float32 "
                     'fields). Tolerances are first estimates. measured = error = the worst deviation in tolerance '
                     'units.'),
        specs=lambda: {run: _stokes_spec(run) for run in _STOKES_RUNS}, check=_check_stokes,
        backends=('gpu',), variants=(SelfTestVariant('cpu', tolerance=1.0, backend='cpu', runs=_STOKES_CPU_RUNS),)),
    SelfTestCase(
        name='solid_oscillating_sphere_drag',
        label='Solid mechanics: drag of a sphere oscillating in water (axisymmetric m = 0 and m = 1, mixed)',
        description=(f'A rigid bead of radius {_SPHERE_RADIUS * 1e6:g} um (E 3.5e9, nu 0.34, rho 1050, held by '
                     'prescribed_displacement_domain) in library water (formulation mixed: incompressible, viscous) '
                     f'in a fixed container r < {_SPHERE_CONTAINER:g} a, |z| < {_SPHERE_CONTAINER:g} a, P2 triangles '
                     f'(h = delta_min / 4 on the bead, growth 1.25), cpu, at a / delta = '
                     f"{', '.join(f'{ratio:g}' for ratio in _SPHERE_RATIOS)}: run m0 (axial motion X, azimuthal_order "
                     '0, reaction F_y) and m1 (sideways motion, azimuthal_order 1, u_x = X, u_z = -X, reaction F_x = '
                     'sum R_r - sum R_phi). F_fem = -R - w^2 m_b X (the held bead\'s own inertia) against F = -6 pi '
                     'mu a U (1 + lam a + lam^2 a^2 / 9), U = j w X, lam = sqrt(j w rho / mu) (Re > 0; sign check in '
                     f"the module): within {_SPHERE_TOLERANCE['drag']:g} relative (wall corrections exp(-(R - a) / "
                     f"delta), (a / R)^3), m0 and m1 agree within {_SPHERE_TOLERANCE['m0 vs m1']:g}, the m1 pressure "
                     'on the axis is 0, and the pressure p = (3 / 2) mu a U (1 + lam a + lam^2 a^2 / 3) cos(theta) / '
                     f'r^2 (derivation in the module) at r = {_SPHERE_PRESSURE_DISTANCE:g} a, last frequency (m0: '
                     'front minus back on the axis, free of the enclosed water\'s pressure gauge; m1: the cos(phi) '
                     f"amplitude beside the bead) within {_SPHERE_TOLERANCE['pressure']:g} relative. Tolerances are "
                     'first estimates. measured = error = the worst deviation in tolerance units.'),
        specs=lambda: {run: _sphere_spec(run) for run in _SPHERE_RUNS}, check=_check_sphere),
)

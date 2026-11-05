"""Self-tests of the conical eigen paths added with triangles / cuDSS: the Yee symmetrizer D (CPU), em_fdfd_conical on
triangle meshes (fem_conical: CPU closed forms, the em_modes equivalence, a model run) and the cuDSS LDL^T factors of
both operators (GPU); physics imports inside the checks only. Written, not run: tolerances are first estimates."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, skip_without_cudss
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_bor import _match
from .selftests_modes_fem import _report, _sizing

_TYPE = 'em_fdfd_conical'
_PHYSICS = 'em'
_TABLE = 'resonances'
_EIGEN_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
_CUDSS_FACTORIZATIONS = ('cudss', 'cudss_mixed')
_CUDSS_TOLERANCE = 1e-9           # relative omega of a cuDSS solve to the SuperLU one
# the lossy triangles box: double cudss without the Rayleigh polish reached 5.0e-9 on the 5th nearest resonance once
# eigs ran the device Krylov-Schur (cudss_mixed polished 5.3e-10)
_FEM_CUDSS_TOLERANCE = 1e-8

# ---- Yee symmetrizer: graded 14 x 10 cells, anisotropic lossy eps, lossy mu, a PEC block, complex k_z, four face sets
_YEE_CELLS = (14, 10)
_YEE_SPACING = 20e-9
_YEE_WAVELENGTH = 1.55e-6
_YEE_KZ = 6.0e6 - 2.0e5j
_YEE_ASYMMETRY_TOLERANCE = 1e-12   # relative |D A - (D A)^T| (the cuDSS LDL^T acceptance)
_YEE_UNSIGNED_MINIMUM = 1e-3       # |D| without the E_z sign: must stay far from symmetric
_YEE_FACES = {'cpml x, pec / pmc y': ('cpml', 'cpml', 'pec', 'pmc'),
              'mirrors and cpml': ('antisymmetric', 'cpml', 'symmetric', 'cpml'),
              'periodic x (k = 0), pec / cpml y': ('periodic', 'periodic', 'pec', 'cpml')}

# ---- uniaxial PEC box on triangles: 1.0 x 0.6 um, eps_xx = eps_yy = 2, eps_zz = 3, mu 1, k_z fixed; TE^z k0^2 = (kc^2 +
# k_z^2) / eps_t, TM^z k0^2 = kc^2 / eps_zz + k_z^2 / eps_t, kc^2 = (m pi / a)^2 + (n pi / b)^2
_BOX_SIZE = (1.0e-6, 0.6e-6)
_BOX_EPS = (2.0, 3.0)              # eps_t, eps_zz
_BOX_KZ = 5.0e6                    # rad/m
_BOX_TARGET = 1.2e-6               # m
_BOX_H = 50e-9
_BOX_RESONANCES = 8
_BOX_CHECKED = 5
_BOX_QUADRATIC_TOLERANCE = 1e-4    # relative k0
_BOX_LINEAR_TOLERANCE = 1e-2
_BOX_MAX_ORDER = 12
_BOX_LOSS = 0.02                   # -Im eps_t of the GPU case (complex symmetric pencil)

# ---- em_modes equivalence: anisotropic magnetic core in a pec box (the same discrete problem read the other way)
_EQUIVALENCE_SIZE = (1.2e-6, 0.9e-6)
_EQUIVALENCE_CORE = (0.4e-6, 0.3e-6, 0.8e-6, 0.55e-6)   # x0, y0, x1, y1
_EQUIVALENCE_CORE_EPS = (3.0, 4.0, 5.0)                  # eps_xx, eps_yy, eps_zz
_EQUIVALENCE_CORE_MU = 1.2
_EQUIVALENCE_WAVELENGTH = 1.3e-6
_EQUIVALENCE_H = 60e-9
_EQUIVALENCE_TOLERANCE = 1e-8      # relative omega of the conical resonance at k_z = beta to the em_modes omega

# ---- Yee cuDSS: closed lossy dielectric box (pec / pmc / antisymmetric / pec faces, graded cells, a PEC block)
_YEE_CUDSS_CELLS = (40, 30)
_YEE_CUDSS_EPS = 2.25 - 0.01j
_YEE_CUDSS_KZ = 4.0e6
_YEE_CUDSS_TARGET = 1.0e-6
_YEE_CUDSS_RESONANCES = 6

# ---- model run: vacuum circular PEC pipe (radius 0.5 um, pec_domain outside the circle) at k_z, triangles h = a / 16
_PIPE_RADIUS = 0.5e-6
_PIPE_WINDOW = 1.1e-6
_PIPE_KZ = 4.0e6
_PIPE_TARGET = 1.0e-6
_PIPE_RESONANCES = 4
_PIPE_TOLERANCE = 2e-4             # relative k0
_PIPE_MAX_ORDER = 8


# ---- Yee symmetrizer ----
def _graded_widths(n_cells, phase):
    """Deterministic graded cell widths [m] around _YEE_SPACING."""
    return _YEE_SPACING * (1.0 + 0.4 * np.sin(np.arange(n_cells) * 1.3 + phase) ** 2)


def _yee_materials(shape, uniform_eps=None):
    """CrossSectionMaterials of the Yee cases: random anisotropic lossy eps / lossy mu (or uniform_eps, mu 1) and a
    2 x 2 PEC block."""
    from .cross_section_modes import CrossSectionMaterials
    random = np.random.default_rng(3)
    if uniform_eps is None:
        eps = [2.0 + random.random(shape) - 0.05j * random.random(shape) for _ in range(3)]
        mu = 1.0 + 0.5 * random.random(shape) - 0.02j * random.random(shape)
    else:
        eps = [np.full(shape, uniform_eps, dtype=np.complex128) for _ in range(3)]
        mu = np.ones(shape, dtype=np.complex128)
    pec = np.zeros(shape, dtype=bool)
    pec[shape[0] // 2:shape[0] // 2 + 2, shape[1] // 3:shape[1] // 3 + 2] = True
    return CrossSectionMaterials(eps_xx=eps[0], eps_yy=eps[1], eps_zz=eps[2], mu=mu, pec=pec)


def _relative_asymmetry(matrix, row_scale):
    """max |S - S^T| / max |S| of S = diag(row_scale) matrix."""
    import scipy.sparse
    scaled = (scipy.sparse.diags(row_scale) @ matrix).tocsr()
    return float(abs(scaled - scaled.T).max() / abs(scaled).max())


def _check_yee_symmetrizer(results_by_name):
    """D A complex symmetric to round-off for every face set; |D| (no E_z sign) far from it; None with a Bloch phase."""
    from . import conical
    from .boundaries import BoundaryConfig
    nx, ny = _YEE_CELLS
    widths = (_graded_widths(nx, 0.0), _graded_widths(ny, 0.7))
    materials = _yee_materials((nx, ny))
    omega = 2.0 * math.pi * C0 / _YEE_WAVELENGTH
    normalized, details = {}, {}
    for label, faces in _YEE_FACES.items():
        config = BoundaryConfig(*faces)
        config.cpml_npml = 4
        system = conical.assemble_conical(widths, config, _YEE_SPACING, materials, omega, _YEE_KZ)
        if system.symmetrizer is None:
            normalized[f'{label}: symmetrizer'] = math.inf
            continue
        signed = _relative_asymmetry(system.matrix, system.symmetrizer)
        unsigned = _relative_asymmetry(system.matrix, np.abs(system.symmetrizer))
        normalized[f'{label}: D A symmetric'] = signed / _YEE_ASYMMETRY_TOLERANCE
        normalized[f'{label}: E_z sign needed'] = 0.0 if unsigned > _YEE_UNSIGNED_MINIMUM else math.inf
        details[label] = {'asymmetry': signed, 'asymmetry_unsigned_d': unsigned, 'unknowns': system.n_unknowns}
    bloch = BoundaryConfig('bloch', 'bloch', 'pec', 'pmc')
    bloch.bloch_kx = 0.3 * math.pi / float(np.sum(widths[0]))
    system = conical.assemble_conical(widths, bloch, _YEE_SPACING, materials, omega, _YEE_KZ)
    normalized['Bloch phase: no symmetrizer'] = 0.0 if system.symmetrizer is None else math.inf
    return _report(normalized, details)


def _check_yee_cudss(results_by_name):
    """Closed lossy box (Yee): every SuperLU resonance matched by cudss / cudss_mixed; both LDL^T, cudss_mixed polished."""
    skipped = skip_without_cudss()
    if skipped is not None:
        return skipped
    from . import conical
    from .boundaries import BoundaryConfig
    nx, ny = _YEE_CUDSS_CELLS
    widths = (_graded_widths(nx, 0.2), _graded_widths(ny, 1.1))
    materials = _yee_materials((nx, ny), _YEE_CUDSS_EPS)
    config = BoundaryConfig('pec', 'pmc', 'antisymmetric', 'pec')
    omega = 2.0 * math.pi * C0 / _YEE_CUDSS_TARGET

    def solve(eigen_solver=None):
        return conical.solve_conical_resonances(widths, config, _YEE_SPACING, materials, _YEE_CUDSS_KZ,
                                                _YEE_CUDSS_RESONANCES, omega, eigen_solver=eigen_solver)
    return _cudss_comparison(solve)


def _cudss_comparison(solve, n_checked=None, tolerance=_CUDSS_TOLERANCE):
    """Normalised errors (relative omega / tolerance) of cudss / cudss_mixed solves against the SuperLU one
    (solve(eigen_solver) -> EigenSolve): its n_checked resonances nearest the target (None = all), errors recorded."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    omegas = [complex(resonance.omega) for resonance in solve().resonances]
    if not omegas:
        raise ValueError('the SuperLU solve found no resonance')
    checked = omegas[:n_checked]   # nearest the target first
    normalized, details = {}, {'omega_superlu': [[value.real, value.imag] for value in omegas],
                               'n_checked': len(checked)}
    for factorization in _CUDSS_FACTORIZATIONS:
        eigen_solver = ArpackShiftInvertSolver(factorization=factorization)
        result = solve(eigen_solver)
        found = [complex(resonance.omega) for resonance in result.resonances]
        errors = [min(abs(value - expected) for value in found) / abs(expected) if found else math.inf
                  for expected in omegas]
        error = max(errors[:len(checked)])
        normalized[f'{factorization} omega'] = error / tolerance
        normalized[f'{factorization} LDL^T'] = 0.0 if eigen_solver.symmetric_factor else math.inf
        details[factorization] = {'relative_error': error, 'relative_errors_all': errors,
                                  'symmetric_factor': bool(eigen_solver.symmetric_factor),
                                  'refinement_steps': int(eigen_solver.refinement_steps),
                                  'device_bytes': int(eigen_solver.device_bytes),
                                  'rayleigh_change': result.info.get('rayleigh_change'),
                                  'factor_text': result.info.get('factor_text'),
                                  'factor_seconds': result.info['factor_seconds'],
                                  'eigen_seconds': result.info['eigen_seconds']}
    normalized['cudss_mixed polished'] = 0.0 if details['cudss_mixed']['rayleigh_change'] is not None else math.inf
    return _report(normalized, details)


# ---- triangles: uniaxial PEC box ----
def _box_mesh():
    """Uniform triangles mesh (builtin mesher) of the empty box window."""
    from ...model.triangle_meshing import triangulate
    spec = ModelSpec(title='Self-test: FEM conical box', domain=DomainSpec(size_x=_BOX_SIZE[0], size_y=_BOX_SIZE[1]))
    return triangulate(spec, {}, sizing=_sizing(_BOX_H))


def _box_analytic():
    """Sorted exact k0 [rad/m] of the uniaxial PEC box at _BOX_KZ (TE^z and TM^z, orders up to _BOX_MAX_ORDER)."""
    eps_t, eps_z = _BOX_EPS
    (a, b), values = _BOX_SIZE, []
    for m in range(_BOX_MAX_ORDER + 1):
        for n in range(_BOX_MAX_ORDER + 1):
            cutoff = (m * math.pi / a) ** 2 + (n * math.pi / b) ** 2
            if m or n:
                values.append(math.sqrt((cutoff + _BOX_KZ ** 2) / eps_t))
            if m and n:
                values.append(math.sqrt(cutoff / eps_z + _BOX_KZ ** 2 / eps_t))
    return sorted(values)


def _box_solve(mesh, degree, loss=0.0, eigen_solver=None):
    """EigenSolve of the uniaxial PEC box (fem_conical directly) at the box target."""
    from .boundaries import BoundaryConfig
    from .fem_conical import assemble_fem_conical, fem_conical_layout, solve_fem_conical_resonances
    from .fem_modes import FEMModeMaterials
    n = mesh.n_elements
    eps_t, eps_z = _BOX_EPS
    transverse = np.full(n, eps_t - 1j * loss, dtype=np.complex128)
    materials = FEMModeMaterials(eps_xx=transverse, eps_yy=transverse.copy(),
                                 eps_zz=np.full(n, eps_z, dtype=np.complex128), mu=np.ones(n, dtype=np.complex128),
                                 pec=np.zeros(n, dtype=bool))
    layout = fem_conical_layout(mesh, degree, BoundaryConfig('pec', 'pec', 'pec', 'pec'))
    system = assemble_fem_conical(layout, materials, 2.0 * math.pi * C0 / _BOX_TARGET, _BOX_KZ)
    return solve_fem_conical_resonances(system, _BOX_RESONANCES, eigen_solver=eigen_solver)


def _check_fem_box(results_by_name):
    """Quadratic and linear resonances of the uniaxial PEC box vs the TE^z / TM^z closed forms, no spurious ones."""
    mesh = _box_mesh()
    analytic = _box_analytic()
    k_target = 2.0 * math.pi / _BOX_TARGET
    normalized, details = {}, {'elements': int(mesh.n_elements)}
    for degree, tolerance in ((2, _BOX_QUADRATIC_TOLERANCE), (1, _BOX_LINEAR_TOLERANCE)):
        solve = _box_solve(mesh, degree)
        computed = [resonance.omega.real / C0 for resonance in solve.resonances]
        worst, spurious = _match(computed, analytic, k_target, _BOX_CHECKED)
        key = 'quadratic' if degree == 2 else 'linear'
        normalized[key] = worst / tolerance if not spurious else math.inf
        details[key] = {'computed_k0': float_list(sorted(computed)), 'worst_relative_error': worst,
                        'spurious': spurious, 'unknowns': solve.info['n_unknowns'],
                        'max_abs_imag_omega': float(max((abs(resonance.omega.imag) for resonance in solve.resonances),
                                                        default=0.0))}
    return _report(normalized, details)


def _check_fem_cudss(results_by_name):
    """Lossy uniaxial PEC box (triangles): the _BOX_CHECKED SuperLU resonances nearest the target matched by cudss /
    cudss_mixed; LDL^T, polish."""
    skipped = skip_without_cudss()
    if skipped is not None:
        return skipped
    mesh = _box_mesh()
    return _cudss_comparison(lambda eigen_solver=None: _box_solve(mesh, 2, _BOX_LOSS, eigen_solver), _BOX_CHECKED,
                             _FEM_CUDSS_TOLERANCE)


# ---- triangles: em_modes equivalence ----
def _check_fem_modes_equivalence(results_by_name):
    """em_modes (fem_modes) beta of mode 1 at omega0, then fem_conical at k_z = beta: a resonance at omega0."""
    from ...model.triangle_meshing import triangulate
    from .boundaries import BoundaryConfig
    from .fem_conical import assemble_fem_conical, fem_conical_layout, solve_fem_conical_resonances
    from .fem_modes import FEMModeMaterials, solve_fem_modes
    x0, y0, x1, y1 = _EQUIVALENCE_CORE
    spec = ModelSpec(title='Self-test: FEM conical vs em_modes',
                     domain=DomainSpec(size_x=_EQUIVALENCE_SIZE[0], size_y=_EQUIVALENCE_SIZE[1]),
                     geometry=[GeometrySpec('core', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})])
    mesh = triangulate(spec, {}, sizing=_sizing(_EQUIVALENCE_H))
    core = mesh.region == mesh.region_names.index('core')
    eps = [np.where(core, value, 1.0).astype(np.complex128) for value in _EQUIVALENCE_CORE_EPS]
    materials = FEMModeMaterials(eps_xx=eps[0], eps_yy=eps[1], eps_zz=eps[2],
                                 mu=np.where(core, _EQUIVALENCE_CORE_MU, 1.0).astype(np.complex128),
                                 pec=np.zeros(mesh.n_elements, dtype=bool))
    config = BoundaryConfig('pec', 'pec', 'pec', 'pec')
    layout = fem_conical_layout(mesh, 2, config)
    omega0 = 2.0 * math.pi * C0 / _EQUIVALENCE_WAVELENGTH
    modes = solve_fem_modes(layout, materials, omega0, 2).modes
    if not modes:
        raise ValueError('fem_modes found no mode')
    beta = complex(modes[0].beta)
    system = assemble_fem_conical(layout, materials, omega0, beta.real)
    solve = solve_fem_conical_resonances(system, 4)
    if not solve.resonances:
        raise ValueError('fem_conical found no resonance at k_z = beta')
    nearest = min(solve.resonances, key=lambda resonance: abs(resonance.omega - omega0))
    error = abs(nearest.omega / omega0 - 1.0)
    return _report({'omega at k_z = beta': error / _EQUIVALENCE_TOLERANCE,
                    'beta real': 0.0 if abs(beta.imag) <= 1e-9 * abs(beta.real) else math.inf},
                   {'beta': [beta.real, beta.imag], 'omega_relative_error': error, 'unknowns': layout.n_unknowns,
                    'omegas': [[complex(resonance.omega).real, complex(resonance.omega).imag]
                               for resonance in solve.resonances]})


# ---- model run: circular PEC pipe on triangles ----
def _pipe_specs():
    """{'pipe': em_fdfd_conical eigenfrequency run of the vacuum PEC pipe at k_z on a triangles mesh, cpu backend}."""
    side, centre = _PIPE_WINDOW, 0.5 * _PIPE_WINDOW
    settings = {'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec', 'k_z': _PIPE_KZ,
                'wavelength': _PIPE_TARGET, 'discretization': 'quadratic'}
    spec = ModelSpec(
        title='em_fdfd_conical on triangles: PEC pipe at k_z',
        domain=DomainSpec(size_x=side, size_y=side),
        geometry=[GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': side, 'y1': side}),
                  GeometrySpec('pipe', 'circle', {'cx': centre, 'cy': centre, 'r': _PIPE_RADIUS}),
                  GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['pipe']})],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec(_TABLE, 'resonance_table'), FeatureSpec('wall', 'pec_domain', ['metal'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_PIPE_RADIUS / 16.0)],
        study=StudySpec(type=_EIGEN_STUDY, settings={'n_resonances': _PIPE_RESONANCES}))
    spec.solver_defaults.backend = 'cpu'
    return {'pipe': spec}


def _pipe_analytic():
    """Exact k0 [rad/m] of the vacuum PEC pipe at _PIPE_KZ: (zero / a)^2 + k_z^2 of the J_n (TM) and J_n' (TE) zeros."""
    import scipy.special
    values = []
    for order in range(_PIPE_MAX_ORDER + 1):
        for zero in (*scipy.special.jn_zeros(order, 4), *scipy.special.jnp_zeros(order, 4)):
            values.append(math.sqrt((zero / _PIPE_RADIUS) ** 2 + _PIPE_KZ ** 2))
    return values


def _check_pipe(results_by_name):
    """Every table resonance matches a pipe mode at k_z (relative k0), >= n - 1 rows, one snapshot per row."""
    results = results_by_name['pipe']
    rows = results.result(_PHYSICS, _TABLE)['rows']
    computed = [2.0 * math.pi * float(row[2]) / C0 for row in rows]
    analytic = _pipe_analytic()
    errors = [min(abs(value / reference - 1.0) for reference in analytic) for value in computed]
    labels = results.summary().get('snapshot_labels') or []
    normalized = {f'resonance {number}': error / _PIPE_TOLERANCE for number, error in enumerate(errors, start=1)}
    normalized['row count'] = 0.0 if len(rows) >= _PIPE_RESONANCES - 1 else math.inf
    normalized['snapshots'] = 0.0 if len(labels) == len(rows) else math.inf
    return _report(normalized, {'computed_k0': float_list(computed), 'relative_errors': float_list(errors),
                                'snapshot_labels': list(labels)})


CONICAL_FEM_SELFTESTS = (
    SelfTestCase(
        name='em_conical_symmetrizer',
        label='EM conical k_z: Yee symmetrizer',
        description=(
            'CPU check of physics/em/conical.assemble_conical (no solve): on a graded 14 x 10 grid with random '
            'anisotropic lossy eps, lossy mu, a 2 x 2 PEC block and complex k_z, the operator A scaled by the '
            'symmetrizer D (layout dual lengths x CPML stretch per axis, E_z negated) must be complex symmetric, '
            f'max |D A - (D A)^T| / max |D A| <= {_YEE_ASYMMETRY_TOLERANCE:g} (the cuDSS LDL^T acceptance), for cpml x '
            'with pec / pmc y, antisymmetric / symmetric mirrors with cpml, and periodic x (k = 0); with |D| (no E_z '
            f'sign) it must stay above {_YEE_UNSIGNED_MINIMUM:g} (the k_z blocks are Hermitian pairs), and a Bloch '
            'phase gives no symmetrizer. measured = error = the worst item in tolerance units, tolerance 1. Written, '
            'not run.'),
        specs=lambda: {},
        check=_check_yee_symmetrizer,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_conical_fem_pec_box',
        label='EM conical k_z on triangles: uniaxial PEC box',
        description=(
            'CPU check of physics/em/fem_conical (no model run): a 1.0 x 0.6 um PEC box filled with eps_xx = eps_yy = '
            f'{_BOX_EPS[0]:g}, eps_zz = {_BOX_EPS[1]:g} on a uniform {_BOX_H * 1e9:g} nm triangles mesh (builtin '
            f'mesher) at k_z = {_BOX_KZ:g} rad/m, squared-Cayley shift-invert near {_BOX_TARGET * 1e6:g} um. Exact: '
            'TE^z k0^2 = (kc^2 + k_z^2) / eps_t, TM^z k0^2 = kc^2 / eps_zz + k_z^2 / eps_t, kc^2 = (m pi / a)^2 + (n '
            f'pi / b)^2. The {_BOX_CHECKED} exact k0 nearest the target each matched within '
            f'{_BOX_QUADRATIC_TOLERANCE:g} (quadratic) / {_BOX_LINEAR_TOLERANCE:g} (linear), no computed one in that '
            'band matching none (spurious). Pins the k_z coupling blocks, eps_t on E_t and eps_zz on E_z. '
            'Tolerances are first estimates; written, not run.'),
        specs=lambda: {},
        check=_check_fem_box,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_conical_fem_modes_equivalence',
        label='EM conical k_z on triangles: em_modes equivalence',
        description=(
            'CPU check (fem_modes and fem_conical directly): an anisotropic magnetic core (eps 3 / 4 / 5, mu 1.2) in a '
            f'vacuum pec box on one quadratic triangles mesh; mode 1 of fem_modes at lambda = '
            f'{_EQUIVALENCE_WAVELENGTH * 1e6:g} um gives beta, then fem_conical at k_z = beta must have a resonance at '
            f'that omega within {_EQUIVALENCE_TOLERANCE:g} relative: both pencils are the same discrete problem (E_t '
            '= e_t / beta, E_z = j w) read at fixed omega or fixed k_z. Pins the k_z blocks, nu_t and the walls '
            'against em_modes. Written, not run.'),
        specs=lambda: {},
        check=_check_fem_modes_equivalence,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_conical_yee_cudss',
        label='EM conical k_z: cuDSS LDL^T of the symmetrized Yee operator',
        description=(
            'GPU (conical directly, no model run): a closed lossy dielectric box (eps 2.25 - 0.01j, graded 40 x 30 '
            f'cells, a PEC block, pec / pmc / antisymmetric / pec faces) at k_z = {_YEE_CUDSS_KZ:g} rad/m solved with '
            'the eigen factorization superlu, cudss and cudss_mixed. Expected: every SuperLU resonance matched by the '
            f'nearest one of each cuDSS solve to {_CUDSS_TOLERANCE:g} relative (first estimate), both cuDSS factors '
            'the LDL^T of D A (the symmetrizer passed the 1e-12 symmetry check) and cudss_mixed Rayleigh-polished. '
            'Recorded: refinement steps, device bytes, the factor text, times. Skipped without nvmath-python. '
            'Written, not run.'),
        specs=lambda: {},
        check=_check_yee_cudss,
        backends=('gpu',)),
    SelfTestCase(
        name='em_conical_fem_cudss',
        label='EM conical k_z on triangles: cuDSS LDL^T factor',
        description=(
            'GPU (fem_conical directly): the em_conical_fem_pec_box pencil with a lossy transverse eps (-Im = '
            f'{_BOX_LOSS:g}, complex symmetric) solved with superlu, cudss and cudss_mixed: the {_BOX_CHECKED} '
            f'SuperLU resonances nearest the target matched to {_FEM_CUDSS_TOLERANCE:g} relative (the farther ones '
            f'of the {_BOX_RESONANCES} only recorded: first run, all eight, cudss 1.4e-8 and cudss_mixed 9.2e-5 '
            'worst, Rayleigh change up to 1e-2; device Krylov-Schur run: checked worst cudss 5.0e-9, cudss_mixed '
            '5.3e-10), both cuDSS factors LDL^T (D = I), cudss_mixed polished. Skipped without nvmath-python.'),
        specs=lambda: {},
        check=_check_fem_cudss,
        backends=('gpu',)),
    SelfTestCase(
        name='em_fdfd_conical_fem_pipe',
        label='EM conical k_z on triangles: PEC pipe model run',
        description=(
            'Model run of em_fdfd_conical on a triangles mesh (cpu backend, no gmsh): a vacuum PEC pipe of radius '
            f'{_PIPE_RADIUS * 1e6:g} um as the circle inside a pec_domain (window box minus the circle), h = a / 16, '
            f'quadratic, k_z = {_PIPE_KZ:g} rad/m, eigenfrequency study ({_PIPE_RESONANCES} resonances near '
            f'{_PIPE_TARGET * 1e6:g} um). Expected: every resonance_table row matches a pipe mode k0^2 = (x / a)^2 + '
            f'k_z^2 (x a zero of J_n or J_n\', n <= {_PIPE_MAX_ORDER}) within {_PIPE_TOLERANCE:g} relative k0 (first '
            f'estimate), at least {_PIPE_RESONANCES - 1} rows, one snapshot per row. Pins the builder path (create on '
            'triangles, element PEC masks, the curved isoparametric wall, resonance table, fem_field snapshots). '
            'Written, not run.'),
        specs=_pipe_specs,
        check=_check_pipe),
)

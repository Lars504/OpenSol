"""Self-tests of em_fdfd_bor on triangle meshes (fem_bor, solvers/fdfd_bor_fem): CPU cases on fem_bor directly (PEC
and uniaxial cylinder cavities vs Bessel zeros, a dielectric sphere with PML vs its Mie poles), a GPU cuDSS case and a
cpu-backend model run (PEC sphere cavity, curved pec_domain wall); physics imports inside the checks only."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, skip_without_cudss
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_bor import _analytic_cavity, _match
from .selftests_modes_fem import _report, _sizing

_TYPE = 'em_fdfd_bor'
_PHYSICS = 'em'
_TABLE = 'resonances'
_EIGEN_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME

# ---- PEC cylinder cavity on triangles (radius = height = 1 um, pec r_hi / z_lo / z_hi, the em_bor_pec_cylinder closed
# forms): the three axis classes m = 0, 1, 2; quadratic elements at h = 50 nm (k h ~ 0.2), linear at the same mesh for
# m = 2 (no axis layer: the clean order-2 check). Tolerances are first estimates (not run).
_CAVITY_SIZE = 1.0e-6
_CAVITY_H = 50e-9
_CAVITY_TARGET = 1.5e-6
_CAVITY_ORDERS = (0, 1, 2)
_UNIAXIAL_ORDERS = (0, 2)
_UNIAXIAL_EPS = (2.0, 3.0)            # eps_rr = eps_phiphi, eps_zz
_CAVITY_RESONANCES = 8
_CAVITY_CHECKED = 4
_QUADRATIC_TOLERANCE = 1e-4           # relative k
_LINEAR_ORDER = 2
_LINEAR_TOLERANCE = 1e-2

# ---- dielectric sphere (n 2, radius 1 um) in vacuum, quarter (r, z) disk with a pmc z_lo mirror, 0.8 um cylindrical PML
# rings (12 layers, pml_kappa_max 1: radiating fields) on r_hi and z_hi, m = 4: every computed resonance with Q >=
# _MIE_MIN_Q among those nearest the target is matched to the Mie pole (TE / TM, l >= 4) its secant refinement
# reaches; first-estimate tolerances on Re omega and Q (PML reflection, mesh)
_MIE_RADIUS = 1.0e-6
_MIE_INDEX = 2.0
_MIE_WINDOW = 1.6e-6
_MIE_RING = (0.8e-6, 12)
_MIE_CPML = {'cpml_npml': 8, 'cpml_kappa_max': 1.0, 'cpml_sigma_factor': 1.0}
_MIE_H = 60e-9
_MIE_ORDER = 4
_MIE_TARGET = 1.75e-6
_MIE_RESONANCES = 6
_MIE_MIN_Q = 10.0
_MIE_MAX_DEGREE = 12
_MIE_MATCHED_MIN = 2
_MIE_TOLERANCES = (2e-3, 5e-2)        # relative Re omega, relative Q
_CUDSS_FACTORIZATIONS = ('cudss', 'cudss_mixed')
_CUDSS_TOLERANCE = 1e-9               # relative omega to the SuperLU solve (first estimate)

# ---- model run: PEC sphere cavity (radius 1 um, pec_domain outside the circle, pmc z_lo mirror) at m = 1 on a
# triangles mesh (h = a / 16, quadratic, isoparametric wall): every table resonance vs the PEC sphere roots
_SPHERE_RADIUS = 1.0e-6
_SPHERE_WINDOW = 1.05e-6
_SPHERE_ORDER = 1
_SPHERE_TARGET = 2.0 * math.pi * _SPHERE_RADIUS / 4.0   # k a = 4
_SPHERE_RESONANCES = 4
_SPHERE_TOLERANCE = 2e-4              # relative k (first estimate)
_SPHERE_MAX_DEGREE = 8


def _cavity_mesh():
    """Uniform triangles mesh (builtin mesher) of the empty 1 x 1 um cylinder window."""
    from ...model.triangle_meshing import triangulate
    spec = ModelSpec(title='Self-test: FEM BOR cylinder', domain=DomainSpec(size_x=_CAVITY_SIZE, size_y=_CAVITY_SIZE))
    return triangulate(spec, {}, sizing=_sizing(_CAVITY_H))


def _isotropic_materials(n_elements, eps):
    """BORMaterials of complex128 per-element eps (all three components), mu 1, no PEC."""
    from .body_of_revolution import BORMaterials
    return BORMaterials(eps=np.asarray(eps, dtype=np.complex128), mu=np.ones(n_elements, dtype=np.complex128),
                        pec=np.zeros(n_elements, dtype=bool))


def _cavity_solves(mesh, degree, eps_transverse, eps_axial, orders):
    """{order: EigenSolve} of the PEC cylinder (fem_bor directly, one assembly) at the cavity target."""
    from .body_of_revolution import BORMaterials
    from .boundaries import BoundaryConfig
    from .fem_bor import FEMBORAssembly, fem_bor_layout, solve_fem_bor_resonances
    n = mesh.n_elements
    transverse = np.full(n, eps_transverse, dtype=np.complex128)
    materials = BORMaterials(eps=transverse, mu=np.ones(n, dtype=np.complex128), pec=np.zeros(n, dtype=bool),
                             eps_rr=transverse, eps_zz=np.full(n, eps_axial, dtype=np.complex128))
    layout = fem_bor_layout(mesh, degree, BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='pec', y_hi='pec'))
    assembly = FEMBORAssembly(layout, materials, 2.0 * math.pi * C0 / _CAVITY_TARGET)
    return {order: solve_fem_bor_resonances(assembly.system(order), _CAVITY_RESONANCES) for order in orders}


def _cavity_case(orders, eps_transverse, eps_axial, linear):
    """Normalised errors of the quadratic cavity per order (and the linear m = _LINEAR_ORDER check)."""
    mesh = _cavity_mesh()
    k_target = 2.0 * math.pi / _CAVITY_TARGET
    normalized, details = {}, {'elements': int(mesh.n_elements)}
    runs = [(2, orders, _QUADRATIC_TOLERANCE)] + ([(1, (_LINEAR_ORDER,), _LINEAR_TOLERANCE)] if linear else [])
    for degree, run_orders, tolerance in runs:
        for order, solve in _cavity_solves(mesh, degree, eps_transverse, eps_axial, run_orders).items():
            computed = [resonance.omega.real / C0 for resonance in solve.resonances]
            analytic = _analytic_cavity(order, _CAVITY_SIZE, _CAVITY_SIZE, eps_transverse, eps_axial)
            worst, spurious = _match(computed, analytic, k_target, _CAVITY_CHECKED)
            key = f"{'quadratic' if degree == 2 else 'linear'} m = {order}"
            normalized[key] = worst / tolerance if not spurious else math.inf
            details[key] = {'computed_k': float_list(sorted(computed)), 'worst_relative_error': worst,
                            'spurious': spurious, 'unknowns': solve.info['n_unknowns'],
                            'max_abs_imag_omega': float(max((abs(resonance.omega.imag) for resonance
                                                             in solve.resonances), default=0.0))}
    return _report(normalized, details)


def _check_cavity(results_by_name):
    """Vacuum PEC cylinder on triangles vs the Bessel closed form (quadratic m = 0..2, linear m = 2)."""
    return _cavity_case(_CAVITY_ORDERS, 1.0, 1.0, linear=True)


def _check_uniaxial(results_by_name):
    """z-uniaxial filling (eps_rr = eps_phiphi != eps_zz) of the cylinder on triangles vs the closed form."""
    return _cavity_case(_UNIAXIAL_ORDERS, *_UNIAXIAL_EPS, linear=False)


# ---- Mie poles (exp(-i omega t), outgoing h1; the em3d selftests_fdfd3d form, which physics/em may not import) ----
def _mie_denominator(kind, order, size, relative_index):
    """Pole condition of the Mie coefficient at the complex outer size parameter x: 'te' psi(m x) xi'(x) - m xi(x)
    psi'(m x), 'tm' m psi(m x) xi'(x) - xi(x) psi'(m x)."""
    from scipy.special import spherical_jn, spherical_yn
    inner = relative_index * size
    j_inner = spherical_jn(order, inner)
    psi, psi_prime = inner * j_inner, j_inner + inner * spherical_jn(order, inner, derivative=True)
    h = spherical_jn(order, size) + 1j * spherical_yn(order, size)
    h_prime = spherical_jn(order, size, derivative=True) + 1j * spherical_yn(order, size, derivative=True)
    xi, xi_prime = size * h, h + size * h_prime
    if kind == 'te':
        return psi * xi_prime - relative_index * xi * psi_prime
    return relative_index * psi * xi_prime - xi * psi_prime


def _secant_root(function, start, steps=60):
    """Complex secant iteration from start (second point 1e-4 off); None without convergence."""
    previous, current = complex(start), complex(start) * (1.0 + 1e-4)
    f_previous, f_current = complex(function(previous)), complex(function(current))
    for _ in range(steps):
        if f_current == f_previous:
            return None
        step = f_current * (current - previous) / (f_current - f_previous)
        previous, f_previous = current, f_current
        current -= step
        f_current = complex(function(current))
        if abs(step) <= 1e-13 * abs(current):
            return current
    return None


def _nearest_mie_pole(omega):
    """(open_sol omega of the Mie pole nearest a computed resonance omega (secant from it, TE / TM, l = _MIE_ORDER ..
    _MIE_MAX_DEGREE), its label), (None, None) when no refinement stays within 5 % of the start."""
    start = complex(omega).conjugate() * _MIE_RADIUS / C0   # exp(+j omega t), Im > 0 -> exp(-i omega t), Im x < 0
    best, label = None, None
    for kind in ('te', 'tm'):
        for degree in range(max(_MIE_ORDER, 1), _MIE_MAX_DEGREE + 1):
            root = _secant_root(lambda size: _mie_denominator(kind, degree, size, _MIE_INDEX), start)
            if root is None or abs(root - start) > 0.05 * abs(start):
                continue
            if best is None or abs(root - start) < abs(best - start):
                best, label = root, f'{kind.upper()}_{degree}'
    return (None, None) if best is None else (C0 * best.conjugate() / _MIE_RADIUS, label)


def _mie_spec():
    """Quarter (r, z) disk of the sphere (centre on the window corner) in a 1.6 x 1.6 um window."""
    return ModelSpec(title='Self-test: FEM BOR dielectric sphere',
                     domain=DomainSpec(size_x=_MIE_WINDOW, size_y=_MIE_WINDOW),
                     geometry=[GeometrySpec('sphere', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': _MIE_RADIUS})])


def _mie_solve(eigen_solver=None, mesh=None):
    """(EigenSolve at m = _MIE_ORDER, mesh) of the dielectric sphere (fem_bor directly, quadratic, PML rings)."""
    from ...model.triangle_meshing import triangulate
    from .boundaries import BoundaryConfig
    from .fem_bor import FEMBORAssembly, fem_bor_layout, solve_fem_bor_resonances
    if mesh is None:
        width, layers = _MIE_RING
        mesh = triangulate(_mie_spec(), {}, sizing=_sizing(_MIE_H, {'x_hi': width, 'y_hi': width}, layers))
    eps = np.ones(mesh.n_elements, dtype=np.complex128)
    eps[mesh.region == mesh.region_names.index('sphere')] = _MIE_INDEX ** 2
    config = BoundaryConfig(x_lo='pmc', x_hi='cpml', y_lo='pmc', y_hi='cpml')
    for name, value in _MIE_CPML.items():
        setattr(config, name, value)
    config._validate()
    layout = fem_bor_layout(mesh, 2, config)
    assembly = FEMBORAssembly(layout, _isotropic_materials(mesh.n_elements, eps), 2.0 * math.pi * C0 / _MIE_TARGET)
    return solve_fem_bor_resonances(assembly.system(_MIE_ORDER), _MIE_RESONANCES, eigen_solver=eigen_solver), mesh


def _check_mie(results_by_name):
    """Computed resonances (Q >= _MIE_MIN_Q) vs the Mie poles their secant refinement reaches: Re omega and Q."""
    solve, mesh = _mie_solve()
    normalized, details = {}, {'elements': int(mesh.n_elements), 'unknowns': solve.info['n_unknowns'], 'matched': []}
    for number, resonance in enumerate(solve.resonances, start=1):
        if not resonance.q_factor >= _MIE_MIN_Q:
            continue
        pole, label = _nearest_mie_pole(resonance.omega)
        record = {'number': number, 'wavelength': resonance.wavelength, 'q': resonance.q_factor, 'pole': label}
        if pole is not None:
            q_exact = pole.real / (2.0 * pole.imag)
            frequency_error = abs(resonance.omega.real / pole.real - 1.0)
            q_error = abs(resonance.q_factor / q_exact - 1.0)
            record.update(q_exact=q_exact, wavelength_exact=2.0 * math.pi * C0 / pole.real,
                          frequency_error=frequency_error, q_error=q_error)
            normalized[f'{number} {label} Re omega'] = frequency_error / _MIE_TOLERANCES[0]
            normalized[f'{number} {label} Q'] = q_error / _MIE_TOLERANCES[1]
        details['matched'].append(record)
    matched = sum(1 for record in details['matched'] if record['pole'] is not None)
    normalized['matched count'] = 0.0 if matched >= _MIE_MATCHED_MIN else math.inf
    return _report(normalized, details)


def _check_fem_cudss(results_by_name):
    """Every SuperLU resonance of the sphere-with-PML pencil matched by the nearest cudss / cudss_mixed one; both
    factors LDL^T, cudss_mixed polished."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    skipped = skip_without_cudss()
    if skipped is not None:
        return skipped
    reference, mesh = _mie_solve()
    omegas = [complex(resonance.omega) for resonance in reference.resonances]
    if not omegas:
        raise ValueError('the SuperLU solve found no resonance')
    normalized, details = {}, {'omega_superlu': [[value.real, value.imag] for value in omegas]}
    for factorization in _CUDSS_FACTORIZATIONS:
        eigen_solver = ArpackShiftInvertSolver(factorization=factorization)
        solve, _ = _mie_solve(eigen_solver, mesh)
        found = [complex(resonance.omega) for resonance in solve.resonances]
        error = max((min(abs(value - expected) for value in found) / abs(expected) if found else math.inf)
                    for expected in omegas)
        normalized[f'{factorization} omega'] = error / _CUDSS_TOLERANCE
        normalized[f'{factorization} LDL^T'] = 0.0 if eigen_solver.symmetric_factor else math.inf
        details[factorization] = {'relative_error': error, 'symmetric_factor': bool(eigen_solver.symmetric_factor),
                                  'refinement_steps': int(eigen_solver.refinement_steps),
                                  'device_bytes': int(eigen_solver.device_bytes),
                                  'rayleigh_change': solve.info.get('rayleigh_change'),
                                  'factor_text': solve.info.get('factor_text'),
                                  'factor_seconds': solve.info['factor_seconds'],
                                  'eigen_seconds': solve.info['eigen_seconds']}
    normalized['cudss_mixed polished'] = 0.0 if details['cudss_mixed']['rayleigh_change'] is not None else math.inf
    return _report(normalized, details)


# ---- model run: PEC sphere ----
def _sphere_roots(kind, degree, top):
    """Sorted roots 0.5 < x < top of the PEC sphere: 'te' j_l(x) = 0, 'tm' (x j_l(x))' = 0."""
    import scipy.optimize
    import scipy.special

    def function(x):
        value = scipy.special.spherical_jn(degree, x)
        return value if kind == 'te' else value + x * scipy.special.spherical_jn(degree, x, derivative=True)
    grid = np.linspace(0.5, top, 4000)
    values = function(grid)
    return [scipy.optimize.brentq(function, grid[index], grid[index + 1]) for index in range(grid.size - 1)
            if values[index] * values[index + 1] < 0.0]


def _sphere_model_specs():
    """{'sphere': em_fdfd_bor PEC sphere cavity on triangles, m = 1, eigenfrequency, cpu backend}."""
    side, radius = _SPHERE_WINDOW, _SPHERE_RADIUS
    settings = {'bc_x_hi': 'pec', 'bc_y_lo': 'pmc', 'bc_y_hi': 'pec', 'azimuthal_order': _SPHERE_ORDER,
                'wavelength': _SPHERE_TARGET, 'discretization': 'quadratic'}
    spec = ModelSpec(
        title='em_fdfd_bor on triangles: PEC sphere cavity (m = 1)',
        domain=DomainSpec(size_x=side, size_y=side),
        geometry=[GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': side, 'y1': side}),
                  GeometrySpec('cavity', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': radius}),
                  GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['cavity']})],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec(_TABLE, 'resonance_table'), FeatureSpec('wall', 'pec_domain', ['metal'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=radius / 16.0)],
        study=StudySpec(type=_EIGEN_STUDY, settings={'n_resonances': _SPHERE_RESONANCES}))
    spec.solver_defaults.backend = 'cpu'
    return {'sphere': spec}


def _check_sphere_model(results_by_name):
    """Every table resonance matches a PEC sphere root (TE / TM, l >= 1), >= 3 rows, one snapshot per row."""
    results = results_by_name['sphere']
    rows = results.result(_PHYSICS, _TABLE)['rows']
    computed = [2.0 * math.pi * float(row[2]) * _SPHERE_RADIUS / C0 for row in rows]   # k a
    top = 2.0 * max(computed, default=4.0)
    analytic = sorted(root for kind in ('te', 'tm') for degree in range(1, _SPHERE_MAX_DEGREE + 1)
                      for root in _sphere_roots(kind, degree, top))
    errors = [min(abs(value / root - 1.0) for root in analytic) for value in computed]
    labels = results.summary().get('snapshot_labels') or []
    normalized = {f'resonance {number}': error / _SPHERE_TOLERANCE for number, error in enumerate(errors, start=1)}
    normalized['row count'] = 0.0 if len(rows) >= _SPHERE_RESONANCES - 1 else math.inf
    normalized['snapshots'] = 0.0 if len(labels) == len(rows) else math.inf
    return _report(normalized, {'computed_ka': float_list(computed), 'relative_errors': float_list(errors),
                                'snapshot_labels': list(labels)})


BOR_FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='em_bor_fem_pec_cylinder',
        label='EM body of revolution on triangles: PEC cylinder cavity',
        description=(
            'CPU check of physics/em/fem_bor (no model run): the em_bor_pec_cylinder cavity (vacuum, radius = height '
            f'= 1 um, pec r_hi / z_lo / z_hi) on a uniform {_CAVITY_H * 1e9:g} nm triangles mesh (builtin mesher), '
            f'squared-Cayley shift-invert near {_CAVITY_TARGET * 1e6:g} um, {_CAVITY_RESONANCES} resonances per order. '
            f'Quadratic elements at m = {", ".join(map(str, _CAVITY_ORDERS))} (the three axis classes: w = r E_phi '
            'fixed on the axis, E_z there too for |m| >= 1, the 1 / r penalty of the axis layer): the '
            f'{_CAVITY_CHECKED} analytic resonances nearest the target each matched within {_QUADRATIC_TOLERANCE:g} '
            f'relative k, no computed one in that band matching none (spurious modes); linear elements at m = '
            f'{_LINEAR_ORDER} within {_LINEAR_TOLERANCE:g}. One assembly serves every order (K0 + m K1 + m^2 K2). '
            'measured = error = the worst item in tolerance units, tolerance 1; tolerances are first estimates. '
            'Written, not run.'),
        specs=lambda: {},
        check=_check_cavity,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_fem_uniaxial_cavity',
        label='EM body of revolution on triangles: uniaxial cylinder cavity',
        description=(
            'CPU check of the diagonal anisotropy of physics/em/fem_bor: the em_bor_fem_pec_cylinder cavity filled with '
            f'eps_rr = eps_phiphi = {_UNIAXIAL_EPS[0]:g}, eps_zz = {_UNIAXIAL_EPS[1]:g}, quadratic elements, orders '
            f'{", ".join(map(str, _UNIAXIAL_ORDERS))}, matched to the TE / TM closed forms of em_bor_uniaxial_cavity '
            f'within {_QUADRATIC_TOLERANCE:g} (no spurious resonance). Pins eps_rr on E_r, eps_phiphi on w and eps_zz '
            'on E_z. Tolerance a first estimate; written, not run.'),
        specs=lambda: {},
        check=_check_uniaxial,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_fem_mie_sphere',
        label='EM body of revolution on triangles: dielectric sphere Mie poles',
        description=(
            'CPU check of the curved isoparametric interface and the cylindrical PML ring of physics/em/fem_bor: a '
            f'sphere of index {_MIE_INDEX:g}, radius {_MIE_RADIUS * 1e6:g} um, in vacuum as a quarter disk (centre on '
            f'the window corner, pmc z_lo mirror) in a {_MIE_WINDOW * 1e6:g} um square window, {_MIE_RING[0] * 1e6:g} '
            f'um PML rings of {_MIE_RING[1]} layers on r_hi and z_hi (pml_kappa_max 1), quadratic elements at '
            f'{_MIE_H * 1e9:g} nm, m = {_MIE_ORDER}, {_MIE_RESONANCES} resonances near {_MIE_TARGET * 1e6:g} um. Each '
            f'resonance with Q >= {_MIE_MIN_Q:g} is matched to the Mie pole (TE / TM, l = {_MIE_ORDER} .. '
            f'{_MIE_MAX_DEGREE}) that a complex secant started at it reaches (within 5 %); at least '
            f'{_MIE_MATCHED_MIN} matched, each within {_MIE_TOLERANCES[0]:g} relative Re omega and '
            f'{_MIE_TOLERANCES[1]:g} relative Q. measured = error = the worst item in tolerance units, tolerance 1; '
            'tolerances are first estimates; written, not run.'),
        specs=lambda: {},
        check=_check_mie,
        max_wall_time=180.0),
    SelfTestCase(
        name='em_bor_fem_cudss',
        label='EM body of revolution on triangles: cuDSS LDL^T factor',
        description=(
            'GPU (fem_bor directly, no model run): the em_bor_fem_mie_sphere pencil (complex symmetric with the PML '
            'ring) solved with the eigen factorization superlu, cudss and cudss_mixed. Expected: every SuperLU '
            f'resonance omega matched by the nearest one of each cuDSS solve to {_CUDSS_TOLERANCE:g} relative (first '
            'estimate), both cuDSS factors the LDL^T (D = I, equilibrated by (|diag K| + k0^2 |diag M|)^-1/2) and the '
            'cudss_mixed solve Rayleigh-polished. Recorded: refinement steps, device bytes, the factor text (unrefined '
            'complex64 residual), times. Skipped without nvmath-python. Written, not run.'),
        specs=lambda: {},
        check=_check_fem_cudss,
        backends=('gpu',)),
    SelfTestCase(
        name='em_bor_fem_sphere_model',
        label='EM body of revolution on triangles: PEC sphere model run',
        description=(
            'Model run of em_fdfd_bor on a triangles mesh (cpu backend, no gmsh): a PEC sphere cavity of radius '
            f'{_SPHERE_RADIUS * 1e6:g} um as the vacuum quarter disk inside a pec_domain (window box minus the circle, '
            f'{_SPHERE_WINDOW * 1e6:g} um square, pmc z_lo mirror, pec r_hi / z_hi), h = a / 16, discretization '
            f'quadratic, m = {_SPHERE_ORDER}, eigenfrequency study ({_SPHERE_RESONANCES} resonances near k a = 4). '
            'Expected: every resonance_table row matches a PEC sphere root (TE: j_l(k a) = 0, TM: (x j_l(x))\' = 0, '
            f'l = 1 .. {_SPHERE_MAX_DEGREE}) within {_SPHERE_TOLERANCE:g} relative k (first estimate), at least '
            f'{_SPHERE_RESONANCES - 1} rows, one snapshot per row. Pins the builder path (create on triangles, element '
            'PEC masks, the curved isoparametric wall, resonance table, fem_field snapshots). Written, not run.'),
        specs=_sphere_model_specs,
        check=_check_sphere_model),
)

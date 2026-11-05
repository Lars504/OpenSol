"""Self-tests of em_fdfd_ez / em_fdfd_hz on triangle meshes (fem_fdfd: CPU closed forms of the eigen pencil and of the
driven point sources in a PML ring, model runs on the builtin mesher, the cuDSS LDL^T on the GPU); physics imports
inside the checks only. Written, not run: tolerances are first estimates."""
import math

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, skip_without_cudss
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_bor import _match
from .selftests_conical_fem import _cudss_comparison
from .selftests_modes_fem import _report, _sizing

_PHYSICS = 'em'
_TABLE = 'resonances'
_EIGEN_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME

# ---- PEC box on triangles: 1.0 x 0.6 um; ez eps 2: k0^2 eps = kc^2 (m, n >= 1); hz eps_xx 2, eps_yy 3, mu 1: k0^2 =
# (m pi / a)^2 / eps_yy + (n pi / b)^2 / eps_xx (m, n >= 0, not both 0: pec walls are natural for H_z)
_BOX_SIZE = (1.0e-6, 0.6e-6)
_BOX_EZ_EPS = 2.0
_BOX_HZ_EPS = (2.0, 3.0)          # eps_xx, eps_yy
_BOX_TARGET = 1.0e-6              # m
_BOX_H = 50e-9
_BOX_RESONANCES = 8
_BOX_CHECKED = 5
_BOX_QUADRATIC_TOLERANCE = 1e-4   # relative k0
_BOX_LINEAR_TOLERANCE = 1e-2
_BOX_MAX_ORDER = 12
_BOX_LOSS = 0.02                  # -Im eps of the GPU case (complex symmetric pencil)

# ---- driven point sources in vacuum with a PML ring on every face: 2 x 2 um window, lambda 1 um, h = lambda / 20
_FREE_WINDOW = 2.0e-6
_FREE_WAVELENGTH = 1.0e-6
_FREE_H = 50e-9
_FREE_RING = (0.5e-6, 10)         # ring thickness [m], element layers
_FREE_RADII = (0.3e-6, 0.5e-6, 0.7e-6)
_FREE_ANGLES = (0.0, 0.9, 2.3, 4.0)   # rad
_FREE_TOLERANCE = 2e-2            # relative complex error against the Hankel closed form
_CURRENT = 1e-3                   # A (ez), V (hz magnetic current), A (hz dipole moment)

# ---- model runs: vacuum circular PEC disk (pec_domain = window box minus the circle), h = a / 16, quadratic
_DISK_RADIUS = 0.5e-6
_DISK_WINDOW = 1.1e-6
_DISK_TARGET = 0.6e-6
_DISK_RESONANCES = 4
_DISK_TOLERANCE = 2e-4            # relative k0
_DISK_MAX_ORDER = 8


def _box_mesh():
    """Uniform triangles mesh (builtin mesher) of the empty box window."""
    from ...model.triangle_meshing import triangulate
    spec = ModelSpec(title='Self-test: FEM FDFD box', domain=DomainSpec(size_x=_BOX_SIZE[0], size_y=_BOX_SIZE[1]))
    return triangulate(spec, {}, sizing=_sizing(_BOX_H))


def _box_materials(n, polarization, loss=0.0):
    """FEMFDFDMaterials of the filled box (no PEC element)."""
    from .fem_fdfd import FEMFDFDMaterials
    if polarization == 'ez':
        eps = np.full(n, _BOX_EZ_EPS - 1j * loss, dtype=np.complex128)
    else:
        eps = np.zeros((n, 2, 2), dtype=np.complex128)
        eps[:, 0, 0], eps[:, 1, 1] = _BOX_HZ_EPS
    return FEMFDFDMaterials(eps=eps, mu=np.ones(n, dtype=np.complex128), pec=np.zeros(n, dtype=bool))


def _box_analytic(polarization):
    """Sorted exact k0 [rad/m] of the filled PEC box (module constants)."""
    (a, b), values = _BOX_SIZE, []
    for m in range(_BOX_MAX_ORDER + 1):
        for n in range(_BOX_MAX_ORDER + 1):
            if polarization == 'ez' and m and n:
                values.append(math.sqrt(((m * math.pi / a) ** 2 + (n * math.pi / b) ** 2) / _BOX_EZ_EPS))
            elif polarization == 'hz' and (m or n):
                values.append(math.sqrt((m * math.pi / a) ** 2 / _BOX_HZ_EPS[1] + (n * math.pi / b) ** 2
                                        / _BOX_HZ_EPS[0]))
    return sorted(values)


def _box_solve(mesh, polarization, degree, loss=0.0, eigen_solver=None):
    """EigenSolve of the filled PEC box (fem_fdfd directly) at the box target."""
    from .boundaries import BoundaryConfig
    from .fem_fdfd import assemble_fem_fdfd, fem_fdfd_layout, solve_fem_fdfd_resonances
    layout = fem_fdfd_layout(mesh, degree, BoundaryConfig('pec', 'pec', 'pec', 'pec'), polarization)
    system = assemble_fem_fdfd(layout, _box_materials(mesh.n_elements, polarization, loss),
                               2.0 * math.pi * C0 / _BOX_TARGET)
    return solve_fem_fdfd_resonances(system, _BOX_RESONANCES, eigen_solver=eigen_solver)


def _check_box(results_by_name):
    """ez / hz, quadratic and linear resonances of the PEC box vs the closed forms; no spurious one (quadratic)."""
    mesh = _box_mesh()
    k_target = 2.0 * math.pi / _BOX_TARGET
    normalized, details = {}, {'elements': int(mesh.n_elements)}
    for polarization in ('ez', 'hz'):
        analytic = _box_analytic(polarization)
        for degree, tolerance in ((2, _BOX_QUADRATIC_TOLERANCE), (1, _BOX_LINEAR_TOLERANCE)):
            solve = _box_solve(mesh, polarization, degree)
            computed = [resonance.omega.real / C0 for resonance in solve.resonances]
            worst, spurious = _match(computed, analytic, k_target, _BOX_CHECKED)
            key = f"{polarization} {'quadratic' if degree == 2 else 'linear'}"
            normalized[key] = worst / tolerance if not (spurious and degree == 2) else math.inf
            details[key] = {'computed_k0': float_list(sorted(computed)), 'worst_relative_error': worst,
                            'spurious': spurious, 'unknowns': solve.info['n_unknowns']}
    return _report(normalized, details)


def _check_box_cudss(results_by_name):
    """Lossy ez box: every SuperLU resonance matched by cudss / cudss_mixed; both LDL^T, cudss_mixed polished."""
    skipped = skip_without_cudss()
    if skipped is not None:
        return skipped
    mesh = _box_mesh()
    return _cudss_comparison(lambda eigen_solver=None: _box_solve(mesh, 'ez', 2, _BOX_LOSS, eigen_solver))


# ---- driven point sources ----
def _free_layout(polarization):
    """(layout, materials, omega) of the vacuum window with a PML ring on every face (pml_kappa_max 1: radiation)."""
    from ...model.triangle_meshing import triangulate
    from .boundaries import BoundaryConfig
    from .fem_fdfd import FEMFDFDMaterials, fem_fdfd_layout
    spec = ModelSpec(title='Self-test: FEM FDFD point sources',
                     domain=DomainSpec(size_x=_FREE_WINDOW, size_y=_FREE_WINDOW))
    width, layers = _FREE_RING
    mesh = triangulate(spec, {}, sizing=_sizing(_FREE_H, {face: width for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')},
                                                layers))
    config = BoundaryConfig('cpml', 'cpml', 'cpml', 'cpml')
    config.cpml_npml = layers
    config.cpml_kappa_max = 1.0
    config._validate()
    n = mesh.n_elements
    eps = np.ones(n, dtype=np.complex128) if polarization == 'ez' else np.broadcast_to(
        np.eye(2, dtype=np.complex128), (n, 2, 2)).copy()
    materials = FEMFDFDMaterials(eps=eps, mu=np.ones(n, dtype=np.complex128), pec=np.zeros(n, dtype=bool))
    return fem_fdfd_layout(mesh, 2, config, polarization), materials, 2.0 * math.pi * C0 / _FREE_WAVELENGTH


def _free_solution(layout, materials, omega, load):
    """(system, full coefficients) of one driven solve with load (n_free,) by host SuperLU."""
    import scipy.sparse.linalg
    from .fem_fdfd import assemble_fem_fdfd
    system = assemble_fem_fdfd(layout, materials, omega)
    return system, layout.coefficients(scipy.sparse.linalg.spsolve(system.operator.tocsc(), load))


def _check_point_sources(results_by_name):
    """ez line current E_z and H_y, hz magnetic line current H_z and x electric dipole H_z vs the Hankel closed forms
    (complex, relative) at several radii / angles around the window centre."""
    import scipy.special
    from .fem_fdfd import dipole_load, point_fields, point_load
    centre = 0.5 * _FREE_WINDOW
    k = 2.0 * math.pi / _FREE_WAVELENGTH
    normalized, details = {}, {}
    cases = {}
    layout, materials, omega = _free_layout('ez')
    cases['ez line current'] = (layout, materials, omega,
                                point_load(layout, (centre, centre), 1j * omega * MU0 * _CURRENT))
    layout, materials, omega = _free_layout('hz')
    cases['hz magnetic current'] = (layout, materials, omega,
                                    point_load(layout, (centre, centre), 1j * omega * EPS0 * _CURRENT))
    cases['hz x dipole'] = (layout, materials, omega,
                            dipole_load(layout, materials, (centre, centre), (_CURRENT, 0.0)))
    for name, (layout, materials, omega, load) in cases.items():
        system, coefficients = _free_solution(layout, materials, omega, load)
        errors = []
        for radius in _FREE_RADII:
            for angle in _FREE_ANGLES:
                point = (centre + radius * math.cos(angle), centre + radius * math.sin(angle))
                fields = point_fields(system, coefficients, point)
                h0, h1 = scipy.special.hankel2(0, k * radius), scipy.special.hankel2(1, k * radius)
                if name == 'ez line current':
                    pairs = [(fields['ez'], omega * MU0 * _CURRENT / 4.0 * h0)]
                    if angle == 0.0:   # H_y = j k I H1^(2)(k r) / 4 on the +x ray
                        pairs.append((fields['hy'], 1j * k * _CURRENT * h1 / 4.0))
                elif name == 'hz magnetic current':
                    pairs = [(fields['hz'], omega * EPS0 * _CURRENT / 4.0 * h0)]
                elif abs(math.sin(angle)) > 0.2:   # (R p) . grad' G: H_z = j k M H1^(2)(k r) sin(phi) / 4
                    pairs = [(fields['hz'], 1j * k * _CURRENT * h1 * math.sin(angle) / 4.0)]
                else:                              # near the dipole axis (|H_z| small): not compared
                    pairs = []
                errors.extend(abs(measured - expected) / abs(expected) for measured, expected in pairs)
        worst = max(errors)
        normalized[name] = worst / _FREE_TOLERANCE
        details[name] = {'worst_relative_error': worst, 'unknowns': layout.n_unknowns}
    return _report(normalized, details)


# ---- model runs ----
def _disk_specs():
    """{'ez', 'hz': em_fdfd_<pol> eigenfrequency runs of the vacuum PEC disk on a triangles mesh, cpu backend}."""
    side, centre = _DISK_WINDOW, 0.5 * _DISK_WINDOW
    specs = {}
    for polarization in ('ez', 'hz'):
        settings = {'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec',
                    'wavelength': _DISK_TARGET, 'discretization': 'quadratic'}
        spec = ModelSpec(
            title=f'em_fdfd_{polarization} on triangles: PEC disk cavity',
            domain=DomainSpec(size_x=side, size_y=side),
            geometry=[GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': side, 'y1': side}),
                      GeometrySpec('disk', 'circle', {'cx': centre, 'cy': centre, 'r': _DISK_RADIUS}),
                      GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['disk']})],
            physics=[PhysicsSpec(_PHYSICS, f'em_fdfd_{polarization}', settings=settings, features=[
                FeatureSpec(_TABLE, 'resonance_table'), FeatureSpec('wall', 'pec_domain', ['metal'])])],
            meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_DISK_RADIUS / 16.0)],
            study=StudySpec(type=_EIGEN_STUDY, settings={'n_resonances': _DISK_RESONANCES}))
        spec.solver_defaults.backend = 'cpu'
        specs[polarization] = spec
    return specs


def _disk_analytic(polarization):
    """Exact k0 [rad/m] of the vacuum PEC disk: zeros of J_n (E_z, TM) or J_n' (H_z, TE) over the radius."""
    import scipy.special
    zeros = scipy.special.jn_zeros if polarization == 'ez' else scipy.special.jnp_zeros
    return [zero / _DISK_RADIUS for order in range(_DISK_MAX_ORDER + 1) for zero in zeros(order, 4) if zero > 0.0]


def _check_disk(results_by_name):
    """Every table resonance of both runs matches a disk mode (relative k0), >= n - 1 rows, one snapshot per row."""
    normalized, details = {}, {}
    for polarization in ('ez', 'hz'):
        results = results_by_name[polarization]
        rows = results.result(_PHYSICS, _TABLE)['rows']
        computed = [2.0 * math.pi * float(row[2]) / C0 for row in rows]
        analytic = _disk_analytic(polarization)
        errors = [min(abs(value / reference - 1.0) for reference in analytic) for value in computed]
        labels = results.summary().get('snapshot_labels') or []
        for number, error in enumerate(errors, start=1):
            normalized[f'{polarization} resonance {number}'] = error / _DISK_TOLERANCE
        normalized[f'{polarization} row count'] = 0.0 if len(rows) >= _DISK_RESONANCES - 1 else math.inf
        normalized[f'{polarization} snapshots'] = 0.0 if len(labels) == len(rows) else math.inf
        details[polarization] = {'computed_k0': float_list(computed), 'relative_errors': float_list(errors),
                                 'snapshot_labels': list(labels)}
    return _report(normalized, details)


_DRIVEN_PROBE_OFFSET = 0.6e-6   # probe at (centre + offset, centre): the +x ray


def _driven_specs():
    """{'driven': em_fdfd_ez stationary run on triangles: a point current at the centre of a cpml-ringed vacuum window
    (pml_kappa_max 1, 12 element layers) with ez and hy probes on the +x ray, cpu backend}."""
    centre = 0.5 * _FREE_WINDOW
    probe = {'x': centre + _DRIVEN_PROBE_OFFSET, 'y': centre}
    settings = {'wavelength': _FREE_WAVELENGTH, 'pml_kappa_max': 1.0, 'pml_cells': 12, 'discretization': 'quadratic'}
    spec = ModelSpec(
        title='em_fdfd_ez on triangles: point current in vacuum',
        domain=DomainSpec(size_x=_FREE_WINDOW, size_y=_FREE_WINDOW),
        geometry=[GeometrySpec('source_point', 'point', {'x': centre, 'y': centre}),
                  GeometrySpec('probe_point', 'point', probe)],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_ez', settings=settings, features=[
            FeatureSpec('source', 'point_source', ['source_point'], params={'current': _CURRENT}),
            FeatureSpec('probe_ez', 'field_probe', ['probe_point'], params={'component': 'ez'}),
            FeatureSpec('probe_hy', 'field_probe', ['probe_point'], params={'component': 'hy'})])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_FREE_H)],
        study=StudySpec(type='stationary'))
    spec.solver_defaults.backend = 'cpu'
    return {'driven': spec}


def _check_driven(results_by_name):
    """Probe phasors (abs, phase) of E_z and H_y at r on the +x ray vs (omega mu0 I / 4) H0^(2)(k r) and j k I
    H1^(2)(k r) / 4 (complex, relative)."""
    import scipy.special
    results = results_by_name['driven']
    k = 2.0 * math.pi / _FREE_WAVELENGTH
    omega = k * C0
    expected = {'probe_ez': omega * MU0 * _CURRENT / 4.0 * scipy.special.hankel2(0, k * _DRIVEN_PROBE_OFFSET),
                'probe_hy': 1j * k * _CURRENT / 4.0 * scipy.special.hankel2(1, k * _DRIVEN_PROBE_OFFSET)}
    normalized, details = {}, {}
    for name, reference in expected.items():
        result = results.result(_PHYSICS, name)
        measured = complex(result['series']['abs'][0] * np.exp(1j * result['series']['phase'][0]))
        error = abs(measured - reference) / abs(reference)
        normalized[name] = error / _FREE_TOLERANCE
        details[name] = {'measured': [measured.real, measured.imag], 'expected': [reference.real, reference.imag],
                         'relative_error': error}
    return _report(normalized, details)


FDFD_FEM_SELFTESTS = (
    SelfTestCase(
        name='em_fdfd_fem_pec_box',
        label='EM FDFD on triangles: PEC box resonances',
        description=(
            'CPU check of physics/em/fem_fdfd (no model run): a 1.0 x 0.6 um PEC box on a uniform '
            f'{_BOX_H * 1e9:g} nm triangles mesh (builtin mesher), shift-invert near {_BOX_TARGET * 1e6:g} um. ez: '
            f'eps {_BOX_EZ_EPS:g}, exact k0^2 eps = (m pi / a)^2 + (n pi / b)^2 (m, n >= 1, E_z = 0 on the walls); hz: '
            f'eps_xx {_BOX_HZ_EPS[0]:g}, eps_yy {_BOX_HZ_EPS[1]:g}, exact k0^2 = (m pi / a)^2 / eps_yy + (n pi / b)^2 / '
            'eps_xx (m, n >= 0 not both 0: pec walls natural for H_z). The '
            f'{_BOX_CHECKED} exact k0 nearest the target each matched within {_BOX_QUADRATIC_TOLERANCE:g} (quadratic) '
            f'/ {_BOX_LINEAR_TOLERANCE:g} (linear), no unmatched computed one in that band (quadratic). Pins the '
            'essential / natural walls per polarisation and the hz stiffness tensor eps_t / det eps_t. Tolerances '
            'are first estimates; written, not run.'),
        specs=lambda: {},
        check=_check_box,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_fdfd_fem_point_sources',
        label='EM FDFD on triangles: point sources vs Hankel',
        description=(
            f'CPU check of physics/em/fem_fdfd driven loads (no model run): a {_FREE_WINDOW * 1e6:g} um vacuum window '
            f'with a {_FREE_RING[0] * 1e6:g} um PML ring of {_FREE_RING[1]} element layers on every face (kappa_max '
            f'1), quadratic, h = {_FREE_H * 1e9:g} nm, lambda = {_FREE_WAVELENGTH * 1e6:g} um, sources at the centre. '
            'Expected (complex, relative, at r = 0.3 / 0.5 / 0.7 um and four angles): ez line current E_z = (omega mu0 '
            'I / 4) H0^(2)(k r) and on the +x ray H_y = j k I H1^(2)(k r) / 4; hz magnetic current H_z = (omega eps0 '
            'K_m / 4) H0^(2)(k r); hz x-directed electric dipole H_z = j k M H1^(2)(k r) sin(phi) / 4 (angles with '
            f'|sin phi| > 0.2), each within {_FREE_TOLERANCE:g} (PML reflection, first estimate). Pins the source '
            'signs and amplitudes (the Yee FDFD meanings), the in-plane recovery and the PML ring stretch. Written, '
            'not run.'),
        specs=lambda: {},
        check=_check_point_sources,
        max_wall_time=180.0),
    SelfTestCase(
        name='em_fdfd_fem_disk_model',
        label='EM FDFD on triangles: PEC disk model runs',
        description=(
            'Model runs of em_fdfd_ez and em_fdfd_hz on a triangles mesh (cpu backend, no gmsh): a vacuum PEC disk of '
            f'radius {_DISK_RADIUS * 1e6:g} um as the circle inside a pec_domain (window box minus the circle), h = a / '
            f'16, quadratic, eigenfrequency study ({_DISK_RESONANCES} resonances near {_DISK_TARGET * 1e6:g} um). '
            'Expected: every resonance_table row matches k0 a = a zero of J_n (ez) or J_n\' (hz), n <= '
            f'{_DISK_MAX_ORDER}, within {_DISK_TOLERANCE:g} relative (first estimate), at least '
            f'{_DISK_RESONANCES - 1} rows, one snapshot per row. Pins create on triangles, element PEC masks (E_z = 0 '
            'on the closure, natural H_z on the curved isoparametric outline), resonance table and fem_field '
            'snapshots. Written, not run.'),
        specs=_disk_specs,
        check=_check_disk),
    SelfTestCase(
        name='em_fdfd_fem_driven_model',
        label='EM FDFD on triangles: point current model run',
        description=(
            'Model run of em_fdfd_ez on a triangles mesh (cpu backend, stationary study): a 1 mA point current at the '
            f'centre of a {_FREE_WINDOW * 1e6:g} um vacuum window, cpml faces with pml_kappa_max 1 and 12 ring '
            f'layers, h = {_FREE_H * 1e9:g} nm, lambda = {_FREE_WAVELENGTH * 1e6:g} um; field probes ez and hy at '
            f'{_DRIVEN_PROBE_OFFSET * 1e6:g} um on the +x ray. Expected (abs and phase as one complex number): E_z = '
            '(omega mu0 I / 4) H0^(2)(k r), H_y = j k I H1^(2)(k r) / 4 within '
            f'{_FREE_TOLERANCE:g} relative (first estimate). Pins create, the sparse_direct linear role on triangles, '
            'point_source / field_probe applies and the probe result. Written, not run.'),
        specs=_driven_specs,
        check=_check_driven),
    SelfTestCase(
        name='em_fdfd_fem_cudss',
        label='EM FDFD on triangles: cuDSS LDL^T factor',
        description=(
            'GPU (fem_fdfd directly): the em_fdfd_fem_pec_box ez pencil with a lossy eps (-Im = '
            f'{_BOX_LOSS:g}, complex symmetric) solved with the eigen factorization superlu, cudss and cudss_mixed: '
            'every SuperLU resonance matched to 1e-9 relative (first estimate), both cuDSS factors LDL^T (D = I), '
            'cudss_mixed Rayleigh-polished; refinement steps, device bytes and the factor text recorded. Skipped '
            'without nvmath-python. Written, not run.'),
        specs=lambda: {},
        check=_check_box_cudss,
        backends=('gpu',)),
)

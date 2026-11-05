"""Self-tests of the tetrahedral Nedelec eigen path (igor3d_plan section 7, T0 / T3 / T4 / T4b): the sparse-mass eigen
layer, PEC cube spectra on structured Kuhn meshes (static count, convergence orders, mirror sectors), FEMPhysics3D on
a lossy dielectric cube with PML symmetry checks (CPU, specs() empty, check({})), Bloch faces on periodic Kuhn cubes,
em_fdfd_3d on a builtin 'tetrahedra' mesh through the model layer (CPU runs: PEC cube; Bloch cube with field derived
values); the gmsh Mie sphere is examples/em3d/fem_mie_sphere_resonance_3d."""
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from .selftests_conformal3d import _failed_count_result

_LENGTH = 1e-6                    # cube side [m]
_UNIT = (math.pi / _LENGTH) ** 2  # (pi / L)^2: PEC cube eigenvalues are integer multiples
_STATIC_RELATIVE = 1e-6           # |lambda| below this x _UNIT counts as a discrete gradient
_N_COMPARED = 5                   # nonzero cube eigenvalues compared: the (1,1,0) triple and the (1,1,1) pair
# observed eigenvalue convergence orders (asymptotic 2 / 4) at the cells below; degree 2 at 2 cells is pre-asymptotic
# for the (1,1,1) pair (k h ~ 2.7): measured per-mode orders 4.1 / 3.8 / 3.4, max-error order 3.37 (degree 1: 1.85)
_ORDER_MIN = {1: 1.6, 2: 3.2}
_ORDER_CELLS = {1: (3, 6), 2: (2, 4)}
_COARSE_ERROR = {1: 0.25, 2: 3e-2}   # relative error ceilings on the finer mesh (estimates)
_MIRROR_RTOL = 5e-2               # half-cube spectra (degree 2, L / 4 cells) vs the parity-filtered cube modes
_MIRROR_CELLS = (2, 4, 4)
_SOLVER_EPS = 2.25
_SOLVER_Q = 100.0                 # loss of the lossy cube: sigma_e = eps omega_t eps0 / Q
_SOLVER_RTOL = 1e-8               # solve_resonances omegas vs the frozen closed form of the dense spectrum
_SYMMETRY_TOLERANCE = 1e-12
_SPARSE_MASS_RTOL = 1e-10
_SPARSE_MASS_CHAIN = 200
# Bloch / periodic faces on Kuhn meshes (all six faces periodic, k along x)
_PERIODIC_FACES = {face: 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')}
_BLOCH_K_FRACTION = 0.25          # bloch_kx in units of 2 pi / L
_BLOCH_CELLS = 4                  # dense spectra
_BLOCH_COMPARED = 12              # G = 0, G = -x and the four transverse G of the empty lattice, two polarisations each
_BLOCH_RTOL = 5e-2                # degree 2, L / 4 cells vs the empty lattice (estimate: k h up to 1.6)
_BLOCH_SOLVER_CELLS = 3
_BLOCH_PHASE_TOLERANCE = 1e-10    # tangential E on x = L vs e^{-j k L} x on x = 0


# ---- closed forms and meshes ----
def _pec_box_wavenumbers(lengths, count, x_parity=None):
    """count smallest k^2 [1/m^2] with multiplicity of a PEC box; x_parity 'odd' / 'even' keeps the modes whose x
    index has that parity (a half box with a symmetric / antisymmetric mirror at its x_hi face)."""
    values = []
    limit = 6
    for m in range(limit):
        if x_parity is not None and (m % 2 == 1) != (x_parity == 'odd'):
            continue
        for n in range(limit):
            for p in range(limit):
                zeros = (m == 0) + (n == 0) + (p == 0)
                if zeros >= 2:
                    continue
                k_squared = (m * math.pi / lengths[0]) ** 2 + (n * math.pi / lengths[1]) ** 2 + (
                    p * math.pi / lengths[2]) ** 2
                values += [k_squared] * (2 if zeros == 0 else 1)
    return np.sort(values)[:count]


def _cube_physics(cells, degree, faces=None, eps_r=1.0, sigma_e=0.0, wavelength=1.55e-6, eigen_solver=None,
                  lengths=(_LENGTH, _LENGTH, _LENGTH), shell=None, bloch_k=(0.0, 0.0, 0.0)):
    """FEMPhysics3D of a uniform Kuhn box (cells per axis) filled with one material; faces {face: type} (default pec);
    shell = (cells, width) of a PML shell on every cpml face; bloch_k [rad/m] of periodic axes."""
    from ...fem.mesh3d import structured_box_mesh
    from .boundaries3d import BoundaryConfig3D
    from .fem3d import FEMMaterial3D, FEMPhysics3D

    faces = faces or {face: 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')}
    axes = []
    for axis, (count, length) in enumerate(zip(cells, lengths)):
        edges = list(np.linspace(0.0, length, count + 1))
        name = 'xyz'[axis]
        if shell is not None:
            shell_cells, width = shell
            if faces.get(f'{name}_lo') == 'cpml':
                edges = list(np.linspace(-width, 0.0, shell_cells + 1)[:-1]) + edges
            if faces.get(f'{name}_hi') == 'cpml':
                edges += list(np.linspace(length, length + width, shell_cells + 1)[1:])
        axes.append(np.array(edges))
    mesh = structured_box_mesh(*axes, window=(0.0, 0.0, 0.0, *lengths))
    config = BoundaryConfig3D(faces, npml=shell[0] if shell is not None else 15, bloch_k=bloch_k)
    return FEMPhysics3D(mesh, config, background=FEMMaterial3D(eps_r=eps_r, sigma_e=sigma_e), element_order=degree,
                        wavelength=wavelength, eigen_solver=eigen_solver)


def _dense_spectrum(physics):
    """Sorted real generalised eigenvalues of (K, M) on the free dofs (a lossless, PML-free system: real symmetric, or
    Hermitian with Bloch phases)."""
    import scipy.linalg
    _ = physics.n_unknowns   # assembles
    curl, mass = physics.system.curl.toarray(), physics.system.mass.toarray()
    if not (np.abs(curl.imag).max(initial=0.0) > 0.0 or np.abs(mass.imag).max(initial=0.0) > 0.0):
        curl, mass = curl.real, mass.real
    return np.sort(scipy.linalg.eigh(curl, mass, eigvals_only=True))


def _interior_lagrange_count(mesh, degree):
    """Lagrange dofs off the outer boundary: interior nodes (+ interior edges for degree 2): the discrete gradient null
    space of a PEC box."""
    from ...fem.dofs3d import nedelec_dofs_3d
    boundary_faces = np.concatenate(list(mesh.face_tags.values()))
    interior_nodes = mesh.n_points - len(np.unique(mesh.faces[boundary_faces]))
    if degree == 1:
        return interior_nodes
    return interior_nodes + mesh.n_edges - len(nedelec_dofs_3d(mesh, 1).face_dofs(boundary_faces))


def _split_static(values):
    """(static count, the nonzero eigenvalues)."""
    static = np.abs(values) < _STATIC_RELATIVE * _UNIT
    return int(static.sum()), values[~static]


# ---- em_eigen_sparse_mass ----
def _check_sparse_mass(_):
    """A 1D chain pencil through solve_resonances with a diagonal mass and with the same mass as a sparse matrix."""
    import scipy.linalg
    import scipy.sparse

    from ..em.eigenfrequency import solve_resonances

    n = _SPARSE_MASS_CHAIN
    spacing = _LENGTH / n
    stiffness = scipy.sparse.diags([2.0 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1]) / spacing ** 2
    mass = 1.0 + 0.5 * np.sin(np.linspace(0.0, 3.0, n)) ** 2
    exact = np.sort(scipy.linalg.eigh(stiffness.toarray(), np.diag(mass), eigvals_only=True))
    target_squared = exact[5] * 1.02
    omega_target = C0 * math.sqrt(target_squared)
    operator = (target_squared * scipy.sparse.diags(mass) - stiffness).tocsr()
    results = {}
    for form in ('diagonal', 'sparse'):
        mass_form = mass if form == 'diagonal' else scipy.sparse.diags(mass).tocsr()
        weights = np.ones(n) if form == 'diagonal' else scipy.sparse.identity(n, format='csr')
        solve = solve_resonances(operator, mass_form, omega_target, weights, np.ones(n, dtype=bool), 4)
        results[form] = np.sort_complex(np.array([resonance.omega for resonance in solve.resonances]))
    exact_omegas = C0 * np.sqrt(exact)
    nearest = [float(np.min(np.abs(exact_omegas - omega.real)) / omega.real) for omega in results['sparse']]
    agreement = (float(np.max(np.abs(results['sparse'] - results['diagonal']) / np.abs(results['diagonal'])))
                 if len(results['sparse']) == len(results['diagonal']) else math.inf)
    return _failed_count_result(
        {'found': (abs(len(results['sparse']) - 4), 0.0), 'sparse_equals_diagonal': (agreement, _SPARSE_MASS_RTOL),
         'sparse_vs_dense_eig': (max(nearest, default=math.inf), _SPARSE_MASS_RTOL)},
        {'omegas_sparse': [[omega.real, omega.imag] for omega in results['sparse']]})


# ---- em3d_fem_pec_cube ----
def _check_pec_cube(_):
    """Static count, convergence orders of the first nonzero eigenvalues (degree 1 / 2) and mirror half cubes."""
    checks, details = {}, {}
    expected = _pec_box_wavenumbers((_LENGTH,) * 3, _N_COMPARED)
    for degree, cell_counts in _ORDER_CELLS.items():
        errors = []
        for cells in cell_counts:
            physics = _cube_physics((cells,) * 3, degree)
            static, nonzero = _split_static(_dense_spectrum(physics))
            expected_static = _interior_lagrange_count(physics.mesh, degree)
            checks[f'static_count_degree_{degree}_n{cells}'] = (abs(static - expected_static), 0.0)
            error = float(np.max(np.abs(nonzero[:_N_COMPARED] / expected - 1.0)))
            errors.append(error)
            details[f'degree_{degree}_n{cells}'] = {'unknowns': physics.n_unknowns, 'static': static,
                                                    'expected_static': expected_static, 'relative_error': error,
                                                    'eigenvalues_over_unit': (nonzero[:_N_COMPARED] / _UNIT).tolist()}
        order = math.log(errors[0] / errors[1]) / math.log(cell_counts[1] / cell_counts[0])
        details[f'degree_{degree}_order'] = order
        checks[f'order_degree_{degree}'] = (max(_ORDER_MIN[degree] - order, 0.0), 0.0)
        checks[f'error_degree_{degree}'] = (errors[1], _COARSE_ERROR[degree])
    half = (_LENGTH / 2.0, _LENGTH, _LENGTH)
    for mirror, parity in (('symmetric', 'odd'), ('antisymmetric', 'even')):
        faces = {face: 'pec' for face in ('x_lo', 'y_lo', 'y_hi', 'z_lo', 'z_hi')}
        faces['x_hi'] = mirror
        physics = _cube_physics(_MIRROR_CELLS, 2, faces=faces, lengths=half)
        _, nonzero = _split_static(_dense_spectrum(physics))
        reference = _pec_box_wavenumbers((_LENGTH,) * 3, 6, x_parity=parity)
        error = float(np.max(np.abs(nonzero[:6] / reference - 1.0)))
        checks[f'mirror_{mirror}'] = (error, _MIRROR_RTOL)
        details[f'mirror_{mirror}'] = {'eigenvalues_over_unit': (nonzero[:6] / _UNIT).tolist(),
                                       'expected_over_unit': (reference / _UNIT).tolist()}
    return _failed_count_result(checks, details)


# ---- em3d_fem_eigen_solver ----
def _check_eigen_solver(_):
    """FEMPhysics3D.solve_eigenfrequencies on a lossy dielectric cube vs k^2 = lambda_h / (eps - j sigma / (omega_t
    eps0)) of the dense vacuum spectrum (frozen at the target: exact there); PML shell operator symmetry."""
    from ...solvers.cudss_factor import relative_asymmetry

    cells, degree = 3, 2
    vacuum = _dense_spectrum(_cube_physics((cells,) * 3, degree))
    _, nonzero = _split_static(vacuum)
    k_target = math.sqrt(nonzero[0] / _SOLVER_EPS)
    wavelength = 2.0 * math.pi / k_target
    omega_target = C0 * k_target
    sigma_e = _SOLVER_EPS * omega_target * EPS0 / _SOLVER_Q
    physics = _cube_physics((cells,) * 3, degree, eps_r=_SOLVER_EPS, sigma_e=sigma_e, wavelength=wavelength)
    solve = physics.solve_eigenfrequencies(3)
    predicted = C0 * np.sqrt(nonzero[:8] / (_SOLVER_EPS - 1j * sigma_e / (omega_target * EPS0)))
    found = [complex(resonance.omega) for resonance in solve.resonances]
    errors = [float(np.min(np.abs(predicted - omega)) / abs(omega)) for omega in found]
    physics.select_resonance(1)
    centre = np.array([[0.5 * _LENGTH, 0.4 * _LENGTH, 0.3 * _LENGTH]])
    sampled = physics.fields_at(centre)
    shell = _cube_physics((cells,) * 3, degree, faces={'x_lo': 'pec', 'y_lo': 'pec', 'z_lo': 'pec', 'x_hi': 'cpml',
                                                       'y_hi': 'cpml', 'z_hi': 'cpml'},
                          shell=(2, 0.5 * _LENGTH), wavelength=wavelength)
    _ = shell.n_unknowns
    operator = shell.system.operator
    checks = {'found': (abs(len(found) - 3), 0.0), 'omega': (max(errors, default=math.inf), _SOLVER_RTOL),
              'field_finite': (float(not np.isfinite(sampled['e']).all() or not np.isfinite(sampled['h']).all()), 0.0),
              'pml_symmetric': (relative_asymmetry(operator), _SYMMETRY_TOLERANCE),
              'pml_complex': (float(not np.abs(operator.data.imag).max() > 0.0), 0.0)}
    details = {'unknowns': physics.n_unknowns, 'omega_errors': errors, 'q_factors': [r.q_factor for r in solve.resonances],
               'solve_summary': solve.summary(), 'pml_unknowns': shell.n_unknowns}
    return _failed_count_result(checks, details)


# ---- em3d_fem_bloch_lattice ----
def _bloch_kx():
    return _BLOCH_K_FRACTION * 2.0 * math.pi / _LENGTH


def _empty_lattice(k_vector, count):
    """count smallest |k + G|^2 [1/m^2] of the empty cubic lattice of period _LENGTH, two polarisations each."""
    unit = 2.0 * math.pi / _LENGTH
    orders = range(-3, 4)
    values = [float(np.sum((np.asarray(k_vector) + unit * np.array((l, m, n))) ** 2))
              for l in orders for m in orders for n in orders]
    return np.sort(np.repeat(values, 2))[:count]


def _check_bloch_lattice(_):
    """Periodic Kuhn cubes: static counts (k != 0: the Bloch Lagrange masters; k = 0: + 2 harmonic fields), Hermitian
    complex operators, the empty-lattice bands (degree 2); FEMPhysics3D resonances and the Bloch phase of its field."""
    from ...fem.dofs3d import lagrange_dofs_3d
    from .fem3d import bloch_identification

    k_x = _bloch_kx()
    checks, details = {}, {}
    for degree, k_value in ((1, k_x), (2, k_x), (1, 0.0)):
        label = f'degree_{degree}' + ('_k0' if k_value == 0.0 else '')
        physics = _cube_physics((_BLOCH_CELLS,) * 3, degree, faces=_PERIODIC_FACES, bloch_k=(k_value, 0.0, 0.0))
        static, nonzero = _split_static(_dense_spectrum(physics))
        masters = len(np.unique(bloch_identification(lagrange_dofs_3d(physics.mesh, degree),
                                                     physics.boundary_config).master))
        expected_static = masters if k_value != 0.0 else masters + 2   # k = 0: constant fields minus the constant
        operator = physics.system.operator
        asymmetry = float(abs(operator - operator.conj().T).max() / abs(operator).max())
        checks[f'static_count_{label}'] = (abs(static - expected_static), 0.0)
        checks[f'hermitian_{label}'] = (asymmetry, _SYMMETRY_TOLERANCE)
        details[label] = {'unknowns': physics.n_unknowns, 'static': static, 'expected_static': expected_static,
                          'symmetric_flag': physics.system.symmetric}
        if k_value != 0.0:
            checks[f'complex_{label}'] = (float(not np.abs(operator.data.imag).max() > 0.0), 0.0)
        if degree == 2:
            expected = _empty_lattice((k_x, 0.0, 0.0), _BLOCH_COMPARED)
            error = float(np.max(np.abs(nonzero[:_BLOCH_COMPARED] / expected - 1.0)))
            checks['empty_lattice_degree_2'] = (error, _BLOCH_RTOL)
            details[label].update(relative_error=error, bands_over_unit=(nonzero[:_BLOCH_COMPARED] / _UNIT).tolist())
    cells = (_BLOCH_SOLVER_CELLS,) * 3
    _, vacuum = _split_static(_dense_spectrum(_cube_physics(cells, 2, faces=_PERIODIC_FACES, bloch_k=(k_x, 0.0, 0.0))))
    wavelength = _MODEL_OFFSET * 2.0 * math.pi * math.sqrt(_SOLVER_EPS / vacuum[0])
    physics = _cube_physics(cells, 2, faces=_PERIODIC_FACES, eps_r=_SOLVER_EPS, wavelength=wavelength,
                            bloch_k=(k_x, 0.0, 0.0))
    solve = physics.solve_eigenfrequencies(2)
    predicted = C0 * np.sqrt(vacuum[:8] / _SOLVER_EPS)
    found = [complex(resonance.omega) for resonance in solve.resonances]
    errors = [float(np.min(np.abs(predicted - omega)) / abs(omega)) for omega in found]
    physics.select_resonance(1)
    field = physics.fields_at(np.array([[0.0, 0.37 * _LENGTH, 0.61 * _LENGTH],
                                        [_LENGTH, 0.37 * _LENGTH, 0.61 * _LENGTH]]))['e']
    tangential = field[:, 1:]
    mismatch = float(np.max(np.abs(tangential[1] - np.exp(-1j * k_x * _LENGTH) * tangential[0]))
                     / max(float(np.max(np.abs(tangential))), 1e-300))
    checks.update({'solver_found': (abs(len(found) - 2), 0.0), 'solver_omega': (max(errors, default=math.inf),
                                                                                 _SOLVER_RTOL),
                   'solver_lu': (float(physics.system.symmetric), 0.0),
                   'bloch_phase': (mismatch, _BLOCH_PHASE_TOLERANCE)})
    details['solver'] = {'unknowns': physics.n_unknowns, 'omega_errors': errors, 'solve_summary': solve.summary()}
    return _failed_count_result(checks, details)


# ---- em3d_fem_model_cube (model layer, T4b) ----
_MODEL_CELLS = 3                  # Kuhn cubes per side of the builtin 'tetrahedra' mesh
_MODEL_RESONANCES = 3
_MODEL_OFFSET = 0.97              # target wavelength = this x the lowest mode's (off the mode: a regular factor)


def _model_reference():
    """(sorted nonzero vacuum k^2 of the degree-2 Kuhn cube, target wavelength [m]) of the model run."""
    _, nonzero = _split_static(_dense_spectrum(_cube_physics((_MODEL_CELLS,) * 3, 2)))
    return nonzero, _MODEL_OFFSET * 2.0 * math.pi * math.sqrt(_SOLVER_EPS / nonzero[0])


def _model_specs():
    """em_fdfd_3d PEC cube of eps 2.25 on a builtin 'tetrahedra' mesh (dx = L / 3), quadratic Nedelec, cpu backend."""
    from ...model.spec import (DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                               SolverDefaultsSpec, StudySpec)

    _, wavelength = _model_reference()
    settings = {f'bc_{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')}
    settings.update(wavelength=wavelength, discretization='quadratic')
    return {'run': ModelSpec(
        title='Self-test: em_fdfd_3d PEC cube on tetrahedra (model layer)',
        domain=DomainSpec(size_x=_LENGTH, size_y=_LENGTH, size_z=_LENGTH),
        materials=[MaterialSpec('fill', properties={'eps_r': _SOLVER_EPS}, selection=['all'])],
        physics=[PhysicsSpec('em', 'em_fdfd_3d', settings=settings,
                             features=[FeatureSpec('resonances', 'resonance_table')])],
        mesh=MeshSpec(kind='tetrahedra', mode='uniform', dx=_LENGTH / _MODEL_CELLS, mesher='builtin'),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _MODEL_RESONANCES}))}


def _check_model_cube(results_by_name):
    """Resonance-table omegas vs C0 sqrt(k^2 / eps) of the direct FEM spectrum on the same Kuhn mesh nearest the
    target (lossless: 1 / Q ~ 0)."""
    from .selftests_fdfd3d import _nearest, _pairwise_error, _table_omegas

    nonzero, wavelength = _model_reference()
    omega_target = 2.0 * math.pi * C0 / wavelength
    found = _table_omegas(results_by_name['run'].result('em', 'resonances'))
    expected = _nearest([C0 * math.sqrt(value / _SOLVER_EPS) for value in nonzero], omega_target, len(found))
    error = _pairwise_error([omega.real for omega in found], expected) if len(found) == _MODEL_RESONANCES else math.inf
    inverse_q = max((2.0 * abs(omega.imag) / omega.real for omega in found), default=math.inf)
    return _failed_count_result(
        {'omega': (error, _SOLVER_RTOL), 'inverse_q': (inverse_q, _SOLVER_RTOL)},
        {'found_wavelengths': [2.0 * math.pi * C0 / omega.real for omega in found],
         'expected_wavelengths': [2.0 * math.pi * C0 / omega for omega in expected]})


# ---- em3d_fem_model_bloch (model layer: Bloch faces + field derived values on tetrahedra) ----
_DERIVED_EXACT_RTOL = 1e-10       # integrals of the per-element eps_r over the window / a cell-aligned box
_DERIVED_FIELD_RTOL = 5e-2        # |E|^2 of the discrete plane-wave pair: mean / max / point vs uniform (estimate)
_INNER_FRACTION = 2.0 / 3.0       # x extent of the 'inner' box (cell aligned at dx = L / 3)


def _model_bloch_reference():
    """(sorted nonzero vacuum k^2 of the periodic degree-2 Kuhn cube at bloch_kx, target wavelength [m])."""
    _, nonzero = _split_static(_dense_spectrum(_cube_physics((_MODEL_CELLS,) * 3, 2, faces=_PERIODIC_FACES,
                                                             bloch_k=(_bloch_kx(), 0.0, 0.0))))
    return nonzero, _MODEL_OFFSET * 2.0 * math.pi * math.sqrt(_SOLVER_EPS / nonzero[0])


def _model_bloch_specs():
    """em_fdfd_3d periodic cube of eps 2.25 (all faces periodic, bloch_kx) on a builtin 'tetrahedra' mesh (dx = L / 3),
    quadratic Nedelec, cpu backend; field derived values of resonance 1 (eps_r integrals, e_intensity reductions)."""
    from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec,
                               ModelSpec, PhysicsSpec, SolverDefaultsSpec, StudySpec)

    _, wavelength = _model_bloch_reference()
    settings = dict({f'bc_{axis}_{side}': 'periodic' for axis in 'xyz' for side in ('lo', 'hi')},
                    wavelength=wavelength, discretization='quadratic', bloch_kx=_bloch_kx())
    inner = {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'x1': _INNER_FRACTION * _LENGTH, 'y1': _LENGTH, 'z1': _LENGTH}
    probe = {'x': 0.41 * _LENGTH, 'y': 0.53 * _LENGTH, 'z': 0.37 * _LENGTH}
    derived = [DerivedValueSpec('eps_volume', 'eps_r', [], 'integral'),
               DerivedValueSpec('inner_eps_volume', 'eps_r', ['inner'], 'integral'),
               DerivedValueSpec('e_mean', 'e_intensity', [], 'mean'),
               DerivedValueSpec('e_max', 'e_intensity', [], 'max'),
               DerivedValueSpec('e_at', 'e_intensity', [], 'at', point='probe'),
               DerivedValueSpec('mode_volume', 'e_intensity', [], 'mode_volume')]
    return {'run': ModelSpec(
        title='Self-test: em_fdfd_3d Bloch cube on tetrahedra (model layer, derived values)',
        domain=DomainSpec(size_x=_LENGTH, size_y=_LENGTH, size_z=_LENGTH),
        geometry=[GeometrySpec('inner', 'box', inner), GeometrySpec('probe', 'point', probe)],
        materials=[MaterialSpec('fill', properties={'eps_r': _SOLVER_EPS}, selection=['all']),
                   MaterialSpec('inner_fill', properties={'eps_r': _SOLVER_EPS}, selection=['inner'])],
        physics=[PhysicsSpec('em', 'em_fdfd_3d', settings=settings,
                             features=[FeatureSpec('resonances', 'resonance_table')])],
        mesh=MeshSpec(kind='tetrahedra', mode='uniform', dx=_LENGTH / _MODEL_CELLS, mesher='builtin'),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': 2}),
        derived_values=derived)}


def _check_model_bloch(results_by_name):
    """Resonance-table omegas vs the dense Bloch spectrum of the same mesh; eps_r integrals exact; the plane-wave
    |E|^2 nearly uniform (mean / max, point / mean, mode volume / L^3)."""
    from .selftests_fdfd3d import _nearest, _pairwise_error, _table_omegas

    nonzero, wavelength = _model_bloch_reference()
    run = results_by_name['run']
    found = _table_omegas(run.result('em', 'resonances'))
    expected = _nearest([C0 * math.sqrt(value / _SOLVER_EPS) for value in nonzero], 2.0 * math.pi * C0 / wavelength,
                        len(found))
    error = _pairwise_error([omega.real for omega in found], expected) if len(found) == 2 else math.inf
    derived = {name: float(value) for name, value in run.summary()['derived'].items()}   # float() also reads 'nan'
    volume = _LENGTH ** 3

    def relative(value, reference):
        return abs(value / reference - 1.0) if math.isfinite(value) else math.inf

    return _failed_count_result(
        {'omega': (error, _SOLVER_RTOL),
         'eps_volume': (relative(derived['eps_volume'], _SOLVER_EPS * volume), _DERIVED_EXACT_RTOL),
         'inner_eps_volume': (relative(derived['inner_eps_volume'], _SOLVER_EPS * _INNER_FRACTION * volume),
                              _DERIVED_EXACT_RTOL),
         'mean_over_max': (relative(derived['e_mean'], derived['e_max']), _DERIVED_FIELD_RTOL),
         'point_over_mean': (relative(derived['e_at'], derived['e_mean']), _DERIVED_FIELD_RTOL),
         'mode_volume': (relative(derived['mode_volume'], volume), _DERIVED_FIELD_RTOL)},
        {'found_wavelengths': [2.0 * math.pi * C0 / omega.real for omega in found],
         'expected_wavelengths': [2.0 * math.pi * C0 / omega for omega in expected], 'derived': derived})


FEM_3D_CASES = (
    SelfTestCase(
        name='em_eigen_sparse_mass',
        label='Eigen layer: sparse mass matrix = diagonal mass',
        description='CPU (igor3d_plan T0): a 200-unknown chain pencil with a varying diagonal mass solved by '
                    'eigenfrequency.solve_resonances (shift-invert, host SuperLU) with the mass and energy weights as '
                    'arrays and as scipy sparse matrices: 4 resonances each, equal to 1e-10, and equal to the dense '
                    'generalised eigenvalues (1e-10). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_sparse_mass,
        max_wall_time=120.0),
    SelfTestCase(
        name='em3d_fem_pec_cube',
        label='em3d FEM: PEC cube spectra on tetrahedra',
        description='CPU (igor3d_plan T3): dense generalised eigenvalues of the Nedelec curl-curl / mass pair of a '
                    'vacuum PEC cube (L = 1 um) on uniform Kuhn meshes: the static count equals the interior Lagrange '
                    'dofs (nodes, + edges for degree 2: no spurious modes); the (1,1,0) triple and the (1,1,1) pair '
                    '((pi / L)^2 x 2, 3) converge with order >= 1.6 (degree 1, 3 / 6 cells per side) and >= 3.2 '
                    '(degree 2, 2 / 4 cells: pre-asymptotic at 2 cells, measured 3.37), finer-mesh errors under '
                    '0.25 / 3e-2; half cubes (x in [0, L / 2], degree 2, L / 4 cells) with a symmetric / '
                    'antisymmetric x_hi mirror give the first 6 cube modes of odd / even x index (5e-2). error = '
                    'number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_pec_cube,
        max_wall_time=600.0),
    SelfTestCase(
        name='em3d_fem_eigen_solver',
        label='em3d FEM: FEMPhysics3D resonances of a lossy cube',
        description='CPU (igor3d_plan T4): FEMPhysics3D (degree 2, 3 cells per side, host SuperLU, squared Cayley, '
                    'sparse mass) on a PEC cube of eps 2.25 with sigma_e for Q = 100, target on the (1,1,0) triple: '
                    '3 resonances equal to k^2 = lambda_h / (eps - j sigma / (omega_t eps0)) of the dense vacuum '
                    'spectrum (1e-8: exact at the frozen target); sampled E / H finite; with a PML shell on the hi '
                    'faces the operator is complex symmetric (1e-12) with complex entries. error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_eigen_solver,
        max_wall_time=300.0),
    SelfTestCase(
        name='em3d_fem_model_cube',
        label="em3d FEM: em_fdfd_3d on a 'tetrahedra' mesh (model layer)",
        description='CPU (igor3d_plan T4b): the eigenfrequency study of em_fdfd_3d on a builtin tetrahedra mesh (Kuhn '
                    'split, dx = L / 3) of a PEC cube (L = 1 um, eps 2.25 on all, quadratic Nedelec) through the model '
                    'layer (mesh kind, TetrahedronDomain, per-element materials, FEMResonancePhysics3D, resonance '
                    'table): 3 resonances equal to C0 sqrt(k^2 / eps) of the dense spectrum of the same mesh nearest '
                    'the target (1e-8), 1 / Q below 1e-8. error = number of failed sub-checks, tolerance 0.',
        specs=_model_specs,
        check=_check_model_cube,
        max_wall_time=300.0),
    SelfTestCase(
        name='em3d_fem_bloch_lattice',
        label='em3d FEM: Bloch faces on tetrahedra',
        description='CPU (igor3d_plan T6 Bloch): vacuum cubes (L = 1 um, all faces periodic, uniform Kuhn meshes of L / 4) '
                    'at bloch_kx = 0.25 x 2 pi / L: the dense spectrum of P^H (K, M) P has the Bloch Lagrange masters '
                    'as static count (degree 1 and 2; at k = 0 two more: the harmonic constant fields), a Hermitian '
                    '(1e-12) complex operator, and degree 2 gives the 12 lowest empty-lattice bands |k + G|^2 within '
                    '5e-2; FEMPhysics3D (eps 2.25, L / 3, LU) finds the G = 0 pair at the dense-spectrum omegas (1e-8) '
                    'and its tangential E on x = L equals e^{-j k L} x that on x = 0 (1e-10). Tolerances are '
                    'estimates, not yet run. error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_bloch_lattice,
        max_wall_time=600.0),
    SelfTestCase(
        name='em3d_fem_model_bloch',
        label="em3d FEM: Bloch cube and field derived values on a 'tetrahedra' mesh",
        description='CPU (model layer): em_fdfd_3d eigenfrequency study on a builtin tetrahedra mesh (dx = L / 3) of an '
                    'all-periodic cube of eps 2.25 at bloch_kx = 0.25 x 2 pi / L (quadratic Nedelec): the 2 resonances '
                    'equal the dense Bloch spectrum of the same mesh (1e-8); field derived values at quadrature points '
                    '(TetrahedraAccess.evaluate): integral of eps_r over the window and over a cell-aligned box '
                    'exact (1e-10), and the plane-wave pair nearly uniform: e_intensity mean / max, value at a point / '
                    'mean and mode_volume / L^3 within 5e-2 (estimates). error = number of failed sub-checks, '
                    'tolerance 0.',
        specs=_model_bloch_specs,
        check=_check_model_bloch,
        max_wall_time=300.0),
)

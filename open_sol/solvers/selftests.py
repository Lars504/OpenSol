"""CPU vs GPU parity self-tests of the solver runtimes (solver_plan S6 / stage B): pcg, sparse_direct and line_block on
small synthetic 2D grid operators, each backend against a scipy reference, the cuDSS eigen factorization against
SuperLU and the cuDSS sparse_direct back ends against scipy. No model run; the gpu half is skipped without a usable GPU.
Also the CPU checks of the solver_defaults.precision resolution (S9) with the Automatic diffusion multigrid (mesh M0) and
of eigsh_shifted with its shift on an eigenvalue. The MUMPS / PARDISO host direct back ends and eigen factorizations
against scipy (skipped without their wrappers)."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import norm_relative_error

_GRID = (30, 20)                  # nx, ny of the synthetic operators (i * ny + j unknown order)
_SEED = 20261017
_PARITY_TOLERANCE = 1e-6          # relative solution error vs the scipy reference, every backend / precision
_STRUCTURAL_FAILURE = 1.0         # error reported when a sub-check (convergence, refusal, acceptance) fails


def _laplacian(nx, ny):
    """5-point Dirichlet Laplacian of unit spacing on the i * ny + j order (scipy CSR, SPD)."""
    import scipy.sparse as sparse

    def second_difference(n):
        ones = np.ones(n)
        return sparse.diags([-ones[:-1], 2.0 * ones, -ones[:-1]], [-1, 0, 1])

    return (sparse.kron(second_difference(nx), sparse.identity(ny))
            + sparse.kron(sparse.identity(nx), second_difference(ny))).tocsr()


def _outcome(errors, failed, details):
    """SelfTestCase record: worst relative error, or _STRUCTURAL_FAILURE when a sub-check failed."""
    measured = max(errors.values())
    error = max(measured, _STRUCTURAL_FAILURE) if failed else measured
    return {'measured': measured, 'expected': 0.0, 'error': error, 'tolerance': _PARITY_TOLERANCE,
            'details': {'failed': failed, 'relative_errors': errors, **details}}


def _backends():
    """('cpu', 'gpu') with a usable GPU, else ('cpu',): the gpu half is skipped without cupy / a CUDA device."""
    from ..core.backend import gpu_available
    return ('cpu', 'gpu') if gpu_available() else ('cpu',)


def _device(backend, host_array):
    """host_array on the backend (numpy as is, cupy copy)."""
    from ..core.backend import array_module
    return array_module(backend).asarray(host_array)


def _device_matrix(backend, matrix):
    from ..core.backend import sparse_module
    return matrix if backend == 'cpu' else sparse_module(backend).csr_matrix(matrix)


# --- pcg ----------------------------------------------------------------------------------------------------------

def _pcg_run(backend, precision, matrix, rhs, matrix_free):
    """(host solution, iterations, converged) of PCGSolver on the backend: CSR + Jacobi, or matrix-free + callback
    (mixed: plus the float32 callbacks of the inner CG)."""
    from ..core.backend import array_module, to_host
    from .pcg import PCGSolver
    xp = array_module(backend)
    solver = PCGSolver(precision=precision, tolerance=1e-10, max_iterations=20_000, backend=backend)
    operator = _device_matrix(backend, matrix.astype(solver.dtype))
    if matrix_free:
        def callbacks(dtype):
            """(apply, precondition) on a dtype copy of the operator and of its inverse diagonal."""
            typed_operator = _device_matrix(backend, matrix.astype(dtype))
            inverse_diagonal = xp.asarray(1.0 / matrix.diagonal(), dtype=dtype)

            def apply(vector, out):
                out[...] = typed_operator @ vector

            def precondition(residual, out):
                xp.multiply(inverse_diagonal, residual, out=out)
            return apply, precondition

        apply, precondition = callbacks(solver.dtype)
        single = {}
        if precision == 'mixed':
            single['single_operator'], single['single_preconditioner'] = callbacks(np.float32)
        solver.factorize(apply, precondition, n_unknowns=matrix.shape[0], **single)
    else:
        solver.factorize(operator)
    solution = xp.zeros(matrix.shape[0], dtype=solver.dtype)
    converged = solver.solve(xp.asarray(rhs, dtype=solver.dtype), solution)
    return to_host(solution), solver.last_iters, converged


def _check_pcg(results_by_name):
    """pcg double / mixed (CSR + Jacobi) and double / mixed matrix-free (callback preconditioner) on cpu and gpu."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .pcg import CG_CHECK_INTERVAL, PCGSolver
    rng = np.random.default_rng(_SEED)
    nx, ny = _GRID
    matrix = (_laplacian(nx, ny) + scipy.sparse.diags(rng.uniform(0.01, 1.0, nx * ny))).tocsr()
    rhs = rng.standard_normal(nx * ny)
    reference = scipy.sparse.linalg.spsolve(matrix.tocsc(), rhs)
    errors, iterations, failed = {}, {}, []
    backends = _backends()
    for backend in backends:
        for precision, matrix_free in (('double', False), ('mixed', False), ('double', True), ('mixed', True)):
            key = f"{backend}_{precision}{'_matrix_free' if matrix_free else ''}"
            solution, iterations[key], converged = _pcg_run(backend, precision, matrix, rhs, matrix_free)
            errors[key] = norm_relative_error(solution, reference)
            if not converged:
                failed.append(f'{key}_converged')
    if 'gpu' in backends:
        for key in [name for name in iterations if name.startswith('cpu_double')]:   # mixed: outer passes may differ
            twin = 'gpu_' + key[len('cpu_'):]
            if abs(iterations[key] - iterations[twin]) > CG_CHECK_INTERVAL:   # one residual-check block of rounding
                failed.append(f'{key}_iterations_match_gpu')
        try:
            PCGSolver(backend='cpu').factorize(_device_matrix('gpu', matrix))
            failed.append('cpu_refuses_device_operator')
        except TypeError:
            pass
    return _outcome(errors, failed, {'iterations': iterations, 'backends': list(backends)})


# --- sparse_direct ------------------------------------------------------------------------------------------------

def _general_operator(rng, nx, ny):
    """Complex non-symmetric grid operator: Laplacian + complex diagonal + a one-sided x difference (scipy CSR)."""
    import scipy.sparse as sparse
    forward = sparse.kron(sparse.diags([-np.ones(nx), np.ones(nx - 1)], [0, 1]), sparse.identity(ny))
    diagonal = (0.5 + 0.2j) * rng.uniform(1.0, 2.0, nx * ny)
    return (_laplacian(nx, ny) + sparse.diags(diagonal) + 0.3 * forward).tocsr().astype(np.complex128)


def _check_sparse_direct(results_by_name):
    """sparse_direct factorize / solve / transposed solve / recycled GMRES: cpu (scipy operator) vs gpu (scipy_splu,
    cusolver_qr), and the cpu refusal of cusolver_qr."""
    import scipy.sparse
    import scipy.sparse.linalg
    from ..core.backend import to_host
    from .sparse_direct import SparseDirectSolver
    rng = np.random.default_rng(_SEED + 1)
    nx, ny = _GRID
    operator = _general_operator(rng, nx, ny)
    perturbed = (operator + scipy.sparse.diags(1e-3 * rng.uniform(0.0, 1.0, nx * ny))).tocsr()
    rhs = rng.standard_normal(nx * ny) + 1j * rng.standard_normal(nx * ny)
    references = {'solve': scipy.sparse.linalg.spsolve(operator.tocsc(), rhs),
                  'transpose': scipy.sparse.linalg.spsolve(operator.T.tocsc(), rhs),
                  'recycled': scipy.sparse.linalg.spsolve(perturbed.tocsc(), rhs)}
    errors, failed, back_ends = {}, [], {}
    backends = _backends()
    for backend, back_end in (('cpu', 'auto'), ('gpu', 'scipy_splu'), ('gpu', 'cusolver_qr')):
        if backend not in backends:
            continue
        key = f'{backend}_{back_end}'
        solver = SparseDirectSolver(back_end=back_end, backend=backend)
        factor = solver.factorize(_device_matrix(backend, operator), nx, ny, False, False)
        back_ends[key] = solver.back_end
        device_rhs = _device(backend, rhs)
        errors[f'{key}_solve'] = norm_relative_error(to_host(factor.solve(device_rhs)), references['solve'])
        errors[f'{key}_transpose'] = norm_relative_error(to_host(factor.solve(device_rhs, transpose=True)),
                                                     references['transpose'])
        if factor.recyclable:
            solution, _iterations = solver.solve_recycled(factor, _device_matrix(backend, perturbed), device_rhs)
            if solution is None:
                failed.append(f'{key}_recycled_converged')
            else:
                errors[f'{key}_recycled'] = norm_relative_error(to_host(solution), references['recycled'])
        if (backend == 'cpu') != isinstance(factor.solve(device_rhs), np.ndarray):
            failed.append(f'{key}_solution_on_backend')
    if back_ends['cpu_auto'] != 'scipy_splu':
        failed.append('cpu_auto_resolves_scipy_splu')
    try:
        _ = SparseDirectSolver(back_end='cusolver_qr', backend='cpu').back_end
        failed.append('cpu_refuses_cusolver_qr')
    except ValueError:
        pass
    return _outcome(errors, failed, {'back_ends': back_ends, 'backends': list(backends)})


# --- line_block ---------------------------------------------------------------------------------------------------

_LINE_BLOCK_OMEGAS = (0.1, 10.0)

def _check_line_block(results_by_name):
    """line_block mixed / double on cpu and gpu: every shift accepted and equal to the per-shift scipy solve."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .line_block import LineBlockLayout, LineBlockSolver
    rng = np.random.default_rng(_SEED + 2)
    nx, ny = 24, 10
    n_unknowns = nx * ny
    stiffness = _laplacian(nx, ny)
    terms = (np.ones(n_unknowns), rng.uniform(0.5, 2.0, n_unknowns))   # capacity-like and q^2-like diagonals
    # the omega extremes: near the real Laplacian (refinement-limited) and diagonal-dominant
    coefficients = np.array([(1j * omega, q_squared) for omega in _LINE_BLOCK_OMEGAS for q_squared in (0.0, 0.5)])
    rhs = rng.standard_normal((2, n_unknowns)) + 1j * rng.standard_normal((2, n_unknowns))
    references = []
    for shift in coefficients:
        matrix = (stiffness + scipy.sparse.diags(shift[0] * terms[0] + shift[1] * terms[1])).tocsc()
        references.append(np.stack([scipy.sparse.linalg.spsolve(matrix, row) for row in rhs]))
    layout = LineBlockLayout.from_grid_cells(np.arange(n_unknowns), (nx, ny))
    # factorize_operators (Part II stage 3): whole complex non-symmetric operators, one per shift, no shift family
    batch_operators = [_general_operator(rng, nx, ny) for _ in range(3)]
    batch_references = [scipy.sparse.linalg.spsolve(operator.tocsc(), rhs[0]) for operator in batch_operators]
    errors, failed = {}, []
    for backend in _backends():
        for precision in ('mixed', 'double'):
            key = f'{backend}_{precision}'
            solver = LineBlockSolver(precision=precision, backend=backend)
            solver.set_operator(stiffness, layout, terms)
            worst = 0.0
            for start, solutions, _relative, accepted in solver.solve_shifted(coefficients, rhs):
                if not accepted.all():
                    failed.append(f'{key}_accepted')
                for offset, solution in enumerate(solutions):
                    worst = max(worst, norm_relative_error(solution, references[start + offset]))
            errors[key] = worst
            batch = solver.factorize_operators(batch_operators, layout)
            worst = 0.0
            for index in (2, 0, 1):   # any order: each shift sweeps its own operator
                solution, _relative, accepted = batch.solve(rhs[0], index)
                if not accepted:
                    failed.append(f'{key}_operator_batch_accepted')
                worst = max(worst, norm_relative_error(solution, batch_references[index]))
            errors[f'{key}_operator_batch'] = worst
            solver.set_operator(stiffness, layout, terms)
            if batch.valid:
                failed.append(f'{key}_operator_batch_invalid_after_set_operator')
    return _outcome(errors, failed, {'block_axis': layout.block_axis, 'n_blocks': layout.n_blocks,
                                     'block_size': layout.block_size})


# --- arpack_shift_invert eigsh_shifted with the shift exactly on an eigenvalue -------------------------------------
# K = the _GRID Dirichlet Laplacian, M = 2 I: lambda_pq = (d_p(nx) + d_q(ny)) / 2, d_p(n) = 2 - 2 cos(p pi / (n + 1)).
# The shift is lambda_21 itself, so K - shift M is singular to rounding; eigsh_shifted offsets sigma.
_ON_EIGENVALUE_MODE = (2, 1)
_ON_EIGENVALUE_PAIRS = 6
_ON_EIGENVALUE_CHECKED = 4          # nearest references matched (the 6th / 7th may swap with the offset)
_ON_EIGENVALUE_TOLERANCE = 1e-10    # relative eigenvalue error


def _check_eigsh_on_eigenvalue(results_by_name):
    """eigsh_shifted of (Laplacian, 2 I) with the shift exactly on lambda_21: converged, the nearest exact eigenvalues
    to 1e-10 relative."""
    import scipy.sparse
    from .arpack_shift_invert import EIGSH_SHIFT_RELATIVE_OFFSET, ArpackShiftInvertSolver
    nx, ny = _GRID
    line_x = 2.0 - 2.0 * np.cos(np.arange(1, nx + 1) * np.pi / (nx + 1))
    line_y = 2.0 - 2.0 * np.cos(np.arange(1, ny + 1) * np.pi / (ny + 1))
    exact = 0.5 * (line_x[:, None] + line_y[None, :]).ravel()
    shift = 0.5 * float(line_x[_ON_EIGENVALUE_MODE[0] - 1] + line_y[_ON_EIGENVALUE_MODE[1] - 1])
    references = exact[np.argsort(np.abs(exact - shift))][:_ON_EIGENVALUE_CHECKED]
    solver = ArpackShiftInvertSolver()
    start = np.random.default_rng(_SEED + 4).standard_normal(nx * ny)
    values, _vectors, applications, converged, _factor = solver.eigsh_shifted(
        _laplacian(nx, ny), 2.0 * scipy.sparse.identity(nx * ny, format='csc'), shift, _ON_EIGENVALUE_PAIRS,
        start_vector=start)
    error = _matched_error(values, references) if np.size(values) else _STRUCTURAL_FAILURE
    if not converged:
        error = max(error, _STRUCTURAL_FAILURE)
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _ON_EIGENVALUE_TOLERANCE,
            'details': {'shift': shift, 'sigma': shift * (1.0 + EIGSH_SHIFT_RELATIVE_OFFSET), 'converged': converged,
                        'applications': applications, 'eigenvalues': sorted(np.asarray(values).tolist()),
                        'references': references.tolist()}}


# --- arpack_shift_invert factorization cudss ----------------------------------------------------------------------

_CUDSS_SHIFT = 2.0 + 0.1j         # inside the spectrum of the sparse_direct case operator
_CUDSS_EIGENPAIRS = 6
_RESONANT_MODE = (5, 4)           # K (1 + i eta) - lambda_54 I: a drive on a lightly damped Laplacian mode
_RESONANT_LOSS_FACTOR = 1e-6      # eta = 1 / Q: eps32 kappa(E) ~ 0.7, so plain float64 refinement nearly stalls


def _resonant_operator(nx, ny):
    """Complex symmetric K (1 + i eta) - lambda I of the Dirichlet Laplacian K driven on its exact eigenvalue
    lambda of _RESONANT_MODE (smallest eigenvalue eta lambda), and a random complex rhs."""
    import scipy.sparse
    line_x = 2.0 - 2.0 * np.cos(np.arange(1, nx + 1) * np.pi / (nx + 1))
    line_y = 2.0 - 2.0 * np.cos(np.arange(1, ny + 1) * np.pi / (ny + 1))
    resonance = float(line_x[_RESONANT_MODE[0] - 1] + line_y[_RESONANT_MODE[1] - 1])
    operator = ((1.0 + 1j * _RESONANT_LOSS_FACTOR) * _laplacian(nx, ny)
                - resonance * scipy.sparse.identity(nx * ny)).tocsr().astype(np.complex128)
    rng = np.random.default_rng(_SEED + 6)
    return operator, rng.standard_normal(nx * ny) + 1j * rng.standard_normal(nx * ny)


def _matched_error(values, references) -> float:
    """Worst relative distance of each reference eigenvalue to its nearest value."""
    values = np.asarray(values)
    return float(max(np.min(np.abs(values - reference)) / abs(reference) for reference in references))


def _check_arpack_cudss(results_by_name):
    """arpack_shift_invert factorization cudss vs superlu (GPU): solve, eigs shift-invert of a complex non-symmetric
    operator, eigsh_shifted of a real symmetric pencil, a mixed LDL^T driven on a Q = 1e6 resonance (GMRES refinement,
    no double fallback), release and the cpu-only backend misfit."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .arpack_shift_invert import (ArpackShiftInvertSolver, backend_misfit, device_arnoldi_fits,
                                      device_factor_matvec)
    from .cudss_factor import (ACCEPT_RESIDUAL, BACKWARD_ERROR_TOLERANCE, MIXED_MAX_REFINEMENTS, CudssFactorization,
                               cudss_available)
    if not cudss_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _PARITY_TOLERANCE,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    rng = np.random.default_rng(_SEED + 3)
    nx, ny = _GRID
    n_unknowns = nx * ny
    shifted = (_general_operator(rng, nx, ny) - _CUDSS_SHIFT * scipy.sparse.identity(n_unknowns)).tocsr()
    rhs = rng.standard_normal(n_unknowns) + 1j * rng.standard_normal(n_unknowns)
    start = rng.standard_normal(n_unknowns) + 1j * rng.standard_normal(n_unknowns)
    stiffness = _laplacian(nx, ny)
    mass = scipy.sparse.diags(rng.uniform(0.5, 2.0, n_unknowns)).tocsr()
    errors, failed, details, eigenvalues = {}, [], {}, {}
    for factorization in ('superlu', 'cudss'):
        solver = ArpackShiftInvertSolver(factorization=factorization)
        factor = solver.factorize_host(shifted)
        theta, _vectors, _applications, converged = solver.eigs(factor.solve_host, n_unknowns, _CUDSS_EIGENPAIRS, start)
        if not converged:
            failed.append(f'{factorization}_eigs_converged')
        eigenvalues[f'{factorization}_eigs'] = _CUDSS_SHIFT + 1.0 / np.asarray(theta)
        symmetric, _vectors, _applications, converged, _factor = solver.eigsh_shifted(stiffness, mass, 1.0,
                                                                                     _CUDSS_EIGENPAIRS)
        if not converged:
            failed.append(f'{factorization}_eigsh_converged')
        eigenvalues[f'{factorization}_eigsh'] = symmetric
        if factorization == 'superlu':   # the device Krylov-Schur around the host SuperLU solve (packed next-row read)
            routed_theta, _vectors, _applications, converged, _kept = solver.eigs_device(
                factor.solve_host, n_unknowns, _CUDSS_EIGENPAIRS, start, matvec_on_host=True)
            if not converged or solver.last_arnoldi != 'device Krylov-Schur':
                failed.append('superlu_eigs_device_matvec')
            eigenvalues['superlu_eigs_routed'] = _CUDSS_SHIFT + 1.0 / np.asarray(routed_theta)
            # a device_arnoldi runtime keeps host ARPACK below HOST_MATVEC_DEVICE_MIN_ENTRIES (600 x 21 entries)
            small = ArpackShiftInvertSolver(device_arnoldi=True)
            small.eigs(factor.solve_host, n_unknowns, _CUDSS_EIGENPAIRS, start)
            if (small.last_arnoldi != 'ARPACK' or device_arnoldi_fits(n_unknowns, 20, host_matvec=True)
                    or not device_arnoldi_fits(n_unknowns, 20)):
                failed.append('device_arnoldi_size_rule')
        if factorization != 'cudss':
            continue
        # S5b / S5a: the device Krylov-Schur on the same factor (cupy solves), then a widening resumed from its
        # decomposition (2 x the eigenpairs), both against host ARPACK on SuperLU
        device_theta, _vectors, first_applications, converged, kept = solver.eigs_device(
            factor.solve, n_unknowns, _CUDSS_EIGENPAIRS, start)
        if not converged or kept is None:
            failed.append('cudss_eigs_device_converged')
        eigenvalues['cudss_eigs_device'] = _CUDSS_SHIFT + 1.0 / np.asarray(device_theta)
        wider_theta, _vectors, resumed_applications, converged, _kept = solver.eigs_device(
            factor.solve, n_unknowns, 2 * _CUDSS_EIGENPAIRS, start, resume=kept)
        if not converged or np.size(wider_theta) != 2 * _CUDSS_EIGENPAIRS:
            failed.append('cudss_eigs_device_resumed')
        eigenvalues['cudss_eigs_device_resumed'] = _CUDSS_SHIFT + 1.0 / np.asarray(wider_theta)
        details['eigs_device_applications'] = {'first': first_applications, 'resumed': resumed_applications}
        # the same widening on storage reserved by the first pass (reserve_basis): expanded in place, no second basis
        _theta, _vectors, _applications, converged, reserved = solver.eigs_device(
            factor.solve, n_unknowns, _CUDSS_EIGENPAIRS, start, reserve_basis=2 * 20)
        in_place_theta, _vectors, _applications, converged_wider, widened = solver.eigs_device(
            factor.solve, n_unknowns, 2 * _CUDSS_EIGENPAIRS, start, resume=reserved)
        if (not (converged and converged_wider) or reserved is None or widened is None
                or widened[0] is not reserved[0] or np.size(in_place_theta) != 2 * _CUDSS_EIGENPAIRS):
            failed.append('cudss_eigs_device_in_place')
        eigenvalues['cudss_eigs_device_in_place'] = _CUDSS_SHIFT + 1.0 / np.asarray(in_place_theta)
        # eigs with device_factor_matvec (B = I) of the cuDSS factor: the device Krylov-Schur, no host solves
        routed_theta, _vectors, _applications, converged = solver.eigs(
            factor.solve_host, n_unknowns, _CUDSS_EIGENPAIRS, start,
            device_matvec=device_factor_matvec(factor, scipy.sparse.identity(n_unknowns, format='csr')))
        if not converged or solver.last_arnoldi != 'device Krylov-Schur':
            failed.append('cudss_eigs_device_matvec')
        eigenvalues['cudss_eigs_routed'] = _CUDSS_SHIFT + 1.0 / np.asarray(routed_theta)
        # S5c: a model runtime factors cudss_mixed below mixed_min_unknowns in double; a script runtime keeps mixed
        if (ArpackShiftInvertSolver(factorization='cudss_mixed', mixed_min_unknowns=n_unknowns + 1)
                .factorization_for(n_unknowns) != 'cudss'
                or ArpackShiftInvertSolver(factorization='cudss_mixed').factorization_for(n_unknowns) != 'cudss_mixed'):
            failed.append('cudss_mixed_small_rule')
        if not solver.symmetric_factor:
            failed.append('cudss_eigsh_ldlt')
        complex_symmetric = (stiffness + scipy.sparse.diags((0.5 + 0.2j) * rng.uniform(1.0, 2.0, n_unknowns))
                             - _CUDSS_SHIFT * scipy.sparse.identity(n_unknowns)).tocsr().astype(np.complex128)
        symmetric_reference = scipy.sparse.linalg.spsolve(complex_symmetric.tocsc(), rhs)
        ldlt = CudssFactorization(complex_symmetric, symmetric=True)
        if not ldlt.symmetric:
            failed.append('cudss_complex_symmetric_ldlt')
        errors['cudss_ldlt_solve'] = norm_relative_error(ldlt.solve_host(rhs), symmetric_reference)
        details['ldlt_device_bytes'] = ldlt.device_bytes
        if CudssFactorization(shifted, symmetric=True).symmetric:
            failed.append('cudss_asymmetric_falls_back_to_lu')
        mixed = CudssFactorization(complex_symmetric, symmetric=True, precision='mixed')
        errors['cudss_mixed_ldlt_solve'] = norm_relative_error(mixed.solve_host(rhs), symmetric_reference)
        details.update(mixed_unrefined_residual=mixed.single_residual, mixed_refinement_steps=mixed.refinement_steps,
                       mixed_device_bytes=mixed.device_bytes)
        if mixed.factor_precision != 'mixed':   # the double fallback must not hide a mixed factor that stopped working
            failed.append('cudss_mixed_ldlt_kept_mixed')
        resonant, resonant_rhs = _resonant_operator(nx, ny)   # GMRES refinement removes the near-null mode
        resonant_mixed = CudssFactorization(resonant, symmetric=True, precision='mixed')
        if resonant_mixed.factor_precision != 'mixed':
            failed.append('cudss_mixed_resonant_kept_mixed')
        details.update(resonant_unrefined_residual=resonant_mixed.single_residual,
                       resonant_gmres_iterations=resonant_mixed.refinement_steps)
        resonant_solution = resonant_mixed.solve_host(resonant_rhs)
        errors['cudss_mixed_resonant_residual'] = float(np.linalg.norm(resonant_rhs - resonant @ resonant_solution)
                                                        / np.linalg.norm(resonant_rhs))
        plain = resonant_mixed.solve_host(resonant_rhs, refinement_steps=MIXED_MAX_REFINEMENTS)   # diagnostic only
        details['resonant_plain_refinement_residual'] = float(np.linalg.norm(resonant_rhs - resonant @ plain)
                                                              / np.linalg.norm(resonant_rhs))
        resonant_mixed.release()
        mixed_solver = ArpackShiftInvertSolver(factorization='cudss_mixed')
        mixed_lu = mixed_solver.factorize_host(shifted)   # non-symmetric: complex64 LU, solves refined to 1e-12
        if mixed_lu.symmetric or mixed_lu.factor_precision != 'mixed':
            failed.append('cudss_mixed_lu')
        theta, _vectors, _applications, converged = mixed_solver.eigs(mixed_lu.solve_host, n_unknowns,
                                                                      _CUDSS_EIGENPAIRS, start)
        if not converged:
            failed.append('cudss_mixed_eigs_converged')
        eigenvalues['cudss_mixed_eigs'] = _CUDSS_SHIFT + 1.0 / np.asarray(theta)
        errors['cudss_solve'] = norm_relative_error(factor.solve_host(rhs),
                                                scipy.sparse.linalg.spsolve(shifted.tocsc(), rhs))
        details.update(probe_residual=factor.probe_residual, probe_backward_error=factor.probe_backward_error,
                       refinement_steps=factor.refinement_steps, device_bytes=factor.device_bytes,
                       lu_entries=factor.lu_entries)
        if not (factor.probe_residual <= ACCEPT_RESIDUAL or factor.probe_backward_error <= BACKWARD_ERROR_TOLERANCE):
            failed.append('cudss_probe_residual')
        if not solver.work_counters().get('eigen_factor_device_bytes', 0) > 0:
            failed.append('cudss_device_bytes_counted')
        factor.release()
        if factor.valid:
            failed.append('cudss_invalid_after_release')
        try:
            factor.solve_host(rhs)
            failed.append('cudss_solve_refused_after_release')
        except RuntimeError:
            pass
    errors['cudss_eigs'] = _matched_error(eigenvalues['cudss_eigs'], eigenvalues['superlu_eigs'])
    errors['cudss_eigsh'] = _matched_error(eigenvalues['cudss_eigsh'], eigenvalues['superlu_eigsh'])
    errors['cudss_mixed_eigs'] = _matched_error(eigenvalues['cudss_mixed_eigs'], eigenvalues['superlu_eigs'])
    errors['cudss_eigs_device'] = _matched_error(eigenvalues['cudss_eigs_device'], eigenvalues['superlu_eigs'])
    for widened in ('cudss_eigs_device_resumed', 'cudss_eigs_device_in_place'):
        errors[widened] = _matched_error(eigenvalues[widened], eigenvalues['superlu_eigs'])
    for routed in ('superlu_eigs_routed', 'cudss_eigs_routed'):
        errors[routed] = _matched_error(eigenvalues[routed], eigenvalues['superlu_eigs'])
    for factorization in ('cudss', 'cudss_mixed'):
        if not backend_misfit({'factorization': factorization}, 'cpu') or backend_misfit(
                {'factorization': factorization}, 'gpu'):
            failed.append(f'{factorization}_misfit_cpu_only')
    return _outcome(errors, failed, details)


# --- sparse_direct back ends cudss / cudss_mixed ------------------------------------------------------------------

def _cudss_role_checks(failed):
    """CPU: the cuDSS back ends fit a role that lists them (heat, em_fdfd_ez, drift_diffusion, em_fdfd_bor,
    em_fdfd_conical linear; magnetic_fields_ac cudss only) and misfit one that does not (magnetic_fields_ac
    cudss_mixed: opt_in_settings), misfit the cpu backend only and have no host size rule."""
    from ..model.schema import get_registry
    from .roles import solver_misfit
    from .sparse_direct import CUDSS_BACK_ENDS, backend_misfit, size_issue
    registry = get_registry()
    sparse_type = registry.solver_type('sparse_direct')
    every = tuple(CUDSS_BACK_ENDS)
    for physics_name, offered_back_ends in (('heat', every), ('em_fdfd_ez', every), ('drift_diffusion', every),
                                            ('em_fdfd_bor', every), ('em_fdfd_conical', every),
                                            ('magnetic_fields_ac', ('cudss',))):
        physics_type = registry.physics_type(physics_name)
        for back_end in CUDSS_BACK_ENDS:
            offered = back_end in offered_back_ends
            misfit = solver_misfit(sparse_type, {'back_end': back_end}, physics_type, 'linear', 'stationary', 'gpu')
            if bool(misfit) == offered:
                failed.append(f'{physics_name}_{back_end}_offered_{offered}')
    for back_end in CUDSS_BACK_ENDS:
        if not backend_misfit({'back_end': back_end}, 'cpu') or backend_misfit({'back_end': back_end}, 'gpu'):
            failed.append(f'{back_end}_misfit_cpu_only')
        if size_issue(back_end, 5000, 5000, False, False) is not None:
            failed.append(f'{back_end}_no_host_size_rule')
    _auto_back_end_checks(failed, registry)
    _cudss_recipe_checks(failed)


def _cudss_recipe_checks(failed):
    """CPU, no profile: the cuDSS time recipe prices a plan on a kept ordering (SolverWork.ordering_kept, R1c) at
    CUDSS_KEPT_ORDERING_PLAN_FRACTION of a fresh one, adds R3's deferred check per refreshed double factor from device
    values (factorisations - 1; none mixed or from the host route) and, in 3D, uploads no values from device values."""
    from dataclasses import dataclass
    from ..core import time_estimate as te
    from .sparse_direct import CUDSS_KEPT_ORDERING_PLAN_FRACTION, _cudss_time_items

    @dataclass(frozen=True)
    class _KeptOrderingWork(te.SolverWork):
        ordering_kept: bool = True

    base = dict(physics='p', role='linear', unknowns=1e5, nnz=1.2e6, value_bytes=16, factorizations=5.0)

    def items(back_end, work_type=te.SolverWork, **extra):
        return {item.label: item for item in _cudss_time_items(back_end, work_type(**base, **extra))}

    fresh, kept = items('cudss'), items('cudss', _KeptOrderingWork)
    if kept['cuDSS plans'].ops[0].scale != CUDSS_KEPT_ORDERING_PLAN_FRACTION * fresh['cuDSS plans'].ops[0].scale:
        failed.append('recipe_kept_ordering_plan')
    device, device_mixed = items('cudss', device_values=True), items('cudss_mixed', device_values=True)
    check = device.get('cuDSS deferred accuracy checks')
    if (check is None or check.count != base['factorizations'] - 1.0 or 'cuDSS deferred accuracy checks' in fresh
            or 'cuDSS deferred accuracy checks' in device_mixed):
        failed.append('recipe_deferred_checks')
    three_d, three_d_device = items('cudss', dimensions=3), items('cudss', dimensions=3, device_values=True)
    if (not any(isinstance(op, te.Transfer) for op in three_d['cuDSS factorisations'].ops)
            or any(isinstance(op, te.Transfer) for op in three_d_device['cuDSS factorisations'].ops)):
        failed.append('recipe_3d_value_uploads')


def _auto_back_end_checks(failed, registry):
    """CPU: back_end 'auto' of the pressure_acoustics linear role becomes cudss_mixed on gpu triangles only when
    nvmath-python is installed (never on the cpu, a tensor grid or for an explicit back end); a role naming a back end
    it does not offer raises."""
    from ..model.schema import auto_back_end_values
    from .cudss_factor import cudss_available
    from .roles import SolverRole
    from .sparse_direct import AUTO_BACK_END_KEY
    pressure_role = registry.physics_type('pressure_acoustics').solver_roles['linear']
    expected = 'cudss_mixed' if cudss_available() else 'auto'
    for backend, grid, given, wanted in (('gpu', 'triangles', 'auto', expected), ('gpu', 'tetrahedra', 'auto', expected),
                                         ('cpu', 'triangles', 'auto', 'auto'), ('gpu', 'tensor', 'auto', 'auto'),
                                         ('gpu', 'triangles', 'scipy_splu', 'scipy_splu')):
        values = auto_back_end_values({'back_end': given}, pressure_role, backend, grid)
        resolved = values['back_end']
        if resolved != wanted:
            failed.append(f'auto_back_end_{backend}_{grid}_{given}_{resolved}')
        if bool(values.get(AUTO_BACK_END_KEY)) != (resolved != given):   # B1b: marked exactly when replaced
            failed.append(f'auto_back_end_marked_{backend}_{grid}_{given}')
    try:
        SolverRole('general', 'sparse_direct', auto_back_ends={'triangles': 'cudss'})   # no setting_choices back_end
        failed.append('auto_back_end_unoffered_accepted')
    except ValueError:
        pass


def _check_sparse_direct_cudss(results_by_name):
    """sparse_direct back ends cudss / cudss_mixed (GPU) against scipy: DiffusionDirectSolver Cholesky (mixed: float32
    LDL^T, its double fallback a Cholesky) with plan reuse and re-plan, a complex LU (transpose refused), a symmetrizer
    LDL^T with transposed solves, a real host LU."""
    import scipy.sparse
    import scipy.sparse.linalg
    from ..core.backend import to_host
    from .cudss_factor import cudss_available
    from .sparse_direct import CUDSS_BACK_ENDS, DiffusionDirectSolver, SparseDirectSolver
    failed, errors, details = [], {}, {}
    _cudss_role_checks(failed)
    if not cudss_available():
        return _outcome({'role_checks': 0.0}, failed, {'skipped': 'nvmath-python (cuDSS) is not installed'})
    rng = np.random.default_rng(_SEED + 5)
    nx, ny = _GRID
    n_unknowns = nx * ny
    spd = (_laplacian(nx, ny) + scipy.sparse.diags(rng.uniform(0.01, 1.0, n_unknowns))).tocsr()
    changed = (spd + scipy.sparse.diags(rng.uniform(0.0, 0.5, n_unknowns))).tocsr()   # same pattern, new values
    coupling = scipy.sparse.csr_matrix(([0.1, 0.1], ([0, 2], [2, 0])), shape=spd.shape)
    repatterned = (changed + coupling).tocsr()   # one symmetric entry outside the 5-point pattern, still SPD
    real_rhs = rng.standard_normal(n_unknowns)
    for back_end in CUDSS_BACK_ENDS:
        runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
        direct = DiffusionDirectSolver(runtime)
        plans = []
        for name, matrix in (('spd', spd), ('values', changed), ('pattern', repatterned)):
            direct.factorize(_device_matrix('gpu', matrix))
            solution = _device('gpu', np.zeros(n_unknowns))
            direct.solve(_device('gpu', real_rhs), solution)
            errors[f'{back_end}_{name}'] = norm_relative_error(to_host(solution),
                                                           scipy.sparse.linalg.spsolve(matrix.tocsc(), real_rhs))
            plans.append(runtime.work_counters()['cudss_plans'])
        mixed = back_end == 'cudss_mixed'   # never a float32 Cholesky (H0): mixed SPD plans the float32 LDL^T
        if (direct._factor.matrix_kind != ('symmetric' if mixed else 'spd')
                or direct._factor.factor_precision != ('mixed' if mixed else 'double')):
            failed.append(f'{back_end}_spd_kind_{direct._factor.matrix_kind}_{direct._factor.factor_precision}')
        details[f'{back_end}_spd_description'] = direct._factor.description
        if plans != [1, 1, 2]:
            failed.append(f'{back_end}_plan_reused_then_replanned')
        details[f'{back_end}_counters'] = runtime.work_counters()
    _mixed_spd_fallback_checks(spd, real_rhs, failed, errors)
    _mixed_rejection_memory_checks(spd, real_rhs, failed, errors, details)
    _host_route_device_checks(spd, changed, real_rhs, failed, errors, details)
    _mixed_target_checks(spd, real_rhs, failed, errors, details)
    _auto_fallback_checks(spd, real_rhs, failed, errors, details)
    complex_rhs = rng.standard_normal(n_unknowns) + 1j * rng.standard_normal(n_unknowns)
    general = _general_operator(rng, nx, ny)
    runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
    factor = runtime.factorize(_device_matrix('gpu', general), nx, ny, False, False)
    errors['cudss_general'] = norm_relative_error(to_host(factor.solve(_device('gpu', complex_rhs))),
                                              scipy.sparse.linalg.spsolve(general.tocsc(), complex_rhs))
    try:
        factor.solve(_device('gpu', complex_rhs), transpose=True)
        failed.append('cudss_lu_transpose_refused')
    except NotImplementedError:
        pass
    factor.release()
    symmetric = (_laplacian(nx, ny) + scipy.sparse.diags((0.5 + 0.2j) * rng.uniform(1.0, 2.0, n_unknowns))).tocsr()
    symmetrizer = (1.0 + 0.5j) * rng.uniform(0.5, 2.0, n_unknowns)
    unsymmetric = (scipy.sparse.diags(1.0 / symmetrizer) @ symmetric).tocsr()   # D A symmetric, A not
    for back_end in CUDSS_BACK_ENDS:
        runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
        factor = runtime.factorize(_device_matrix('gpu', unsymmetric), nx, ny, False, False, symmetrizer=symmetrizer)
        if not factor.symmetric:
            failed.append(f'{back_end}_symmetrizer_ldlt')
        device_rhs = _device('gpu', complex_rhs)
        errors[f'{back_end}_symmetrizer'] = norm_relative_error(
            to_host(factor.solve(device_rhs)), scipy.sparse.linalg.spsolve(unsymmetric.tocsc(), complex_rhs))
        errors[f'{back_end}_symmetrizer_transpose'] = norm_relative_error(
            to_host(factor.solve(device_rhs, transpose=True)),
            scipy.sparse.linalg.spsolve(unsymmetric.T.tocsc(), complex_rhs))
        factor.release()
    forward = scipy.sparse.kron(scipy.sparse.diags([-np.ones(nx), np.ones(nx - 1)], [0, 1]),
                                scipy.sparse.identity(ny))
    real_general = (spd + 0.3 * forward).tocsr()
    runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
    factor = runtime.factorize_host(real_general, real=True)
    host_solution = factor.solve_host(real_rhs)
    if host_solution.dtype != np.float64 or factor.symmetric:
        failed.append('cudss_real_lu')
    errors['cudss_real_lu'] = norm_relative_error(host_solution,
                                              scipy.sparse.linalg.spsolve(real_general.tocsc(), real_rhs))
    factor.release()
    return _outcome(errors, failed, details)


def _mixed_spd_fallback_checks(spd, rhs, failed, errors):
    """A mixed SPD factor (float32 LDL^T, H0) whose deferred check misses (stopped by the stall test, H4) re-plans as
    the double Cholesky in place and solves right."""
    import scipy.sparse.linalg
    from .cudss_factor import MIXED_STALL_ITERATIONS, CudssFactorization
    canonical = spd.tocsr(copy=True)
    canonical.sum_duplicates()
    factor = CudssFactorization(canonical, real=True, positive_definite=True, precision='mixed')
    if factor.matrix_kind != 'symmetric' or not factor.mixed_factor:
        failed.append(f'mixed_spd_plans_ldlt_{factor.matrix_kind}')
    factor.refactorize_device_values(_device('gpu', canonical.data), canonical.indptr, canonical.indices,
                                     defer_check=True)
    if not factor._check_pending:
        failed.append('mixed_spd_check_deferred')
    factor._single_solve = lambda vector: vector.copy()   # a low-precision solve that lost its accuracy
    errors['cudss_mixed_spd_fallback'] = norm_relative_error(factor.solve_host(rhs),
                                                         scipy.sparse.linalg.spsolve(canonical.tocsc(), rhs))
    if factor.matrix_kind != 'spd' or factor.factor_precision != 'double' or not factor.last_double_fallback:
        failed.append(f'mixed_spd_fallback_double_cholesky_{factor.matrix_kind}_{factor.factor_precision}')
    if factor.last_refinements != MIXED_STALL_ITERATIONS:   # the stalled check's GMRES (the double solve sets none)
        failed.append(f'mixed_spd_deferred_check_stalled_after_{factor.last_refinements}')
    if not factor.last_ordering_reused:   # the Cholesky re-plan on its own LDL^T ordering (one ordering family)
        failed.append('mixed_spd_fallback_ordering_reused')
    factor.release()


def _host_route_device_checks(spd, changed, rhs, failed, errors, details):
    """H1 / R7: factorize_host refactorisations on the planned pattern run on the device (plan reused, asymmetry
    checked there); an unsymmetric operator of the same pattern asked for symmetry is found unsymmetric on the device
    and re-planned as a general LU there; a factor planned from device values (from_device_values, also a vouched
    symmetric DevicePencil) solves right."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .cudss_factor import SYMMETRY_TOLERANCE, CudssFactorization
    from .sparse_direct import SparseDirectSolver
    nx, ny = _GRID
    forward = scipy.sparse.kron(scipy.sparse.diags([-np.ones(nx), np.ones(nx - 1)], [0, 1]),
                                scipy.sparse.identity(ny))
    unsymmetric = (changed + 0.3 * forward).tocsr()   # inside the 5-point pattern
    runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
    factor = runtime.factorize_host(spd, real=True, symmetric=True)   # host plan: LDL^T
    factor = runtime.factorize_host(changed, real=True, symmetric=True, previous=factor)   # device refresh
    errors['host_route_device_refresh'] = norm_relative_error(factor.solve_host(rhs),
                                                          scipy.sparse.linalg.spsolve(changed.tocsc(), rhs))
    refreshed = (factor.last_plan_reused, factor.matrix_kind, factor.asymmetry)
    if not (factor.last_plan_reused and factor.matrix_kind == 'symmetric' and factor.asymmetry <= SYMMETRY_TOLERANCE):
        failed.append('host_route_device_refresh')
    factor = runtime.factorize_host(unsymmetric, real=True, symmetric=True, previous=factor)
    errors['host_route_device_unsymmetric'] = norm_relative_error(factor.solve_host(rhs),
                                                              scipy.sparse.linalg.spsolve(unsymmetric.tocsc(), rhs))
    counters = runtime.work_counters()
    details['host_route_device'] = {'refreshed': refreshed, 'unsymmetric': (factor.last_plan_reused,
                                    factor.matrix_kind, factor.asymmetry), 'cudss_plans': counters['cudss_plans']}
    if (factor.last_plan_reused or factor.matrix_kind != 'general' or not factor.asymmetry > SYMMETRY_TOLERANCE
            or counters['cudss_plans'] != 2):
        failed.append('host_route_device_unsymmetric_lu')
    factor.release()
    canonical = spd.tocsr(copy=True)
    canonical.sum_duplicates()
    planned = CudssFactorization.from_device_values(_device('gpu', canonical.data), canonical.indptr,
                                                    canonical.indices, real=True, positive_definite=True)
    errors['from_device_values'] = norm_relative_error(planned.solve_host(rhs),
                                                   scipy.sparse.linalg.spsolve(spd.tocsc(), rhs))
    if planned.matrix_kind != 'spd' or planned.n_plans != 1:
        failed.append('from_device_values_cholesky')
    planned.release()
    pencil = runtime.device_pencil(canonical.indptr, canonical.indices, [canonical.data], symmetric=True,
                                   symmetric_known=True)
    factor = runtime.factorize_pencil(pencil, [1.0])
    errors['vouched_pencil'] = norm_relative_error(factor.solve_host(rhs), scipy.sparse.linalg.spsolve(spd.tocsc(), rhs))
    if not (pencil.symmetric and factor.matrix_kind == 'symmetric'):
        failed.append('vouched_pencil_ldlt')
    factor.release()


def _auto_fallback_checks(spd, rhs, failed, errors, details):
    """B1b: with every cuDSS probe made to fail, a runtime whose back end came from 'auto' (fallback_back_end) warns,
    switches to host SuperLU for good and solves right (factorize_host, then factorize_pencil of a DevicePencil, also
    after the switch); one without the fallback raises the CudssAccuracyError."""
    import warnings
    import scipy.sparse.linalg
    from .cudss_factor import CudssAccuracyError, CudssFactorization
    from .sparse_direct import DevicePencil, DiffusionDirectSolver, Factorization, SparseDirectSolver
    reference = scipy.sparse.linalg.spsolve(spd.tocsc(), rhs)
    canonical = spd.tocsr(copy=True)
    canonical.sum_duplicates()
    probe_accepted = CudssFactorization._probe_accepted
    CudssFactorization._probe_accepted = lambda self, tolerance: False   # cuDSS refuses every factor
    try:
        strict = SparseDirectSolver('cudss', backend='gpu')
        try:
            strict.factorize_host(spd, real=True, positive_definite=True)
            failed.append('auto_fallback_strict_raises')
        except CudssAccuracyError:
            pass
        runtime = SparseDirectSolver('cudss', backend='gpu', fallback_back_end='scipy_splu')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            factor = runtime.factorize_host(spd, real=True, positive_definite=True)
        errors['auto_fallback_host'] = norm_relative_error(factor.solve_host(rhs), reference)
        counters = runtime.work_counters()
        details['auto_fallback'] = {'counters': counters, 'text': runtime.auto_fallback}
        if (runtime.back_end != 'scipy_splu' or counters.get('auto_fallbacks') != 1 or not caught
                or not isinstance(factor, Factorization)):
            failed.append('auto_fallback_superlu')
        pencil_runtime = SparseDirectSolver('cudss', backend='gpu', fallback_back_end='scipy_splu')
        pencil = DevicePencil(canonical.indptr, canonical.indices, [canonical.data], symmetric=True,
                              symmetric_known=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            first = pencil_runtime.factorize_pencil(pencil, [1.0])
        second = pencil_runtime.factorize_pencil(pencil, [2.0], previous=first)   # SuperLU from the switch on
        errors['auto_fallback_pencil'] = norm_relative_error(first.solve_host(rhs), reference)
        errors['auto_fallback_pencil_after'] = norm_relative_error(2.0 * second.solve_host(rhs), reference)
        if pencil_runtime.back_end != 'scipy_splu' or isinstance(second, CudssFactorization):
            failed.append('auto_fallback_pencil')
        from ..core.backend import sparse_module   # DiffusionDirectSolver: factorize_device_csr's fallback, then SuperLU
        diffusion = DiffusionDirectSolver(SparseDirectSolver('cudss', backend='gpu', fallback_back_end='scipy_splu'))
        device_rhs, solution = _device('gpu', rhs), _device('gpu', np.zeros_like(rhs))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            diffusion.factorize(sparse_module('gpu').csr_matrix(spd))
        diffusion.solve(device_rhs, solution)
        errors['auto_fallback_diffusion'] = norm_relative_error(solution.get(), reference)
        diffusion.factorize(sparse_module('gpu').csr_matrix(spd))   # SuperLU from the switch on
        if diffusion.runtime.back_end != 'scipy_splu' or isinstance(diffusion._factor, CudssFactorization):
            failed.append('auto_fallback_diffusion')
    finally:
        CudssFactorization._probe_accepted = probe_accepted


_LOOSE_BACKWARD_ERROR = 1e-8   # a caller's looser mixed solve target (decision D2 knob)


def _mixed_target_checks(spd, rhs, failed, errors, details):
    """B8: a mixed solve (CGS2 GMRES-IR) meets the default backward error; a per-call looser target takes no more
    GMRES iterations and still meets it; the same target on the factor (solve_backward_error) acts alike."""
    import scipy.sparse.linalg
    from .cudss_factor import MIXED_SOLVE_BACKWARD_ERROR, CudssFactorization
    reference = scipy.sparse.linalg.spsolve(spd.tocsc(), rhs)
    scale = float(abs(spd).sum(axis=1).max())   # ||A||_inf, the backward error's operator norm (A symmetric)

    def backward(solution):
        return float(np.linalg.norm(rhs - spd @ solution) / (scale * np.linalg.norm(solution) + np.linalg.norm(rhs)))

    factor = CudssFactorization(spd, real=True, positive_definite=True, precision='mixed')
    tight = factor.solve_host(rhs)
    tight_steps = factor.last_refinements
    loose = factor.solve_host(rhs, backward_error=_LOOSE_BACKWARD_ERROR)
    loose_steps = factor.last_refinements
    factor.solve_backward_error = _LOOSE_BACKWARD_ERROR
    default_loose = factor.solve_host(rhs)
    details['mixed_target_gmres_iterations'] = {'default': tight_steps, 'loose': loose_steps,
                                                'loose_on_factor': factor.last_refinements}
    errors['mixed_default_target'] = norm_relative_error(tight, reference)
    # the target holds for E = R A R; 100: the equilibration and norm bounds between E's and A's backward errors
    errors['mixed_loose_target'] = max(0.0, backward(loose) - 100.0 * _LOOSE_BACKWARD_ERROR)
    errors['mixed_loose_on_factor'] = max(0.0, backward(default_loose) - 100.0 * _LOOSE_BACKWARD_ERROR)
    details['mixed_target_backward_errors'] = {'default': backward(tight), 'loose': backward(loose),
                                               'default_target': MIXED_SOLVE_BACKWARD_ERROR}
    if loose_steps > tight_steps or factor.last_refinements != loose_steps:
        failed.append('mixed_loose_target_steps')
    factor.release()


def _mixed_rejection_memory_checks(spd, rhs, failed, errors, details):
    """H4 / R1b: a mixed probe that stalls is rejected after MIXED_STALL_ITERATIONS GMRES iterations and both plans of
    the call are counted; inside cudss_orderings_kept a second factor of that pattern plans double at once (no mixed
    attempt, no fallback counted), outside the block mixed is tried again."""
    import scipy.sparse.linalg
    from .cudss_factor import MIXED_STALL_ITERATIONS, CudssFactorization, cudss_orderings_kept
    from .sparse_direct import SparseDirectSolver
    reference = scipy.sparse.linalg.spsolve(spd.tocsc(), rhs)
    single_solve = CudssFactorization._single_solve
    with cudss_orderings_kept():
        runtime = SparseDirectSolver(back_end='cudss_mixed', backend='gpu')
        CudssFactorization._single_solve = lambda self, vector: vector.copy()   # every float32 solve useless
        try:
            first = runtime.factorize_host(spd, real=True, positive_definite=True)
        finally:
            CudssFactorization._single_solve = single_solve
        counters = runtime.work_counters()
        details['mixed_rejection_stalled'] = {'rejection': first.mixed_rejection, 'counters': counters}
        if (first.factor_precision != 'double' or first.matrix_kind != 'spd'
                or f'after {MIXED_STALL_ITERATIONS} iterations' not in (first.mixed_rejection or '')):
            failed.append('mixed_probe_stall_rejected')
        if (counters['cudss_plans'] != 2 or counters['cudss_double_fallbacks'] != 1   # float32 LDL^T + Cholesky,
                or counters['cudss_reused_orderings'] != 1):                          # the latter on its ordering
            failed.append('mixed_rejection_plans_counted')
        errors['cudss_mixed_stalled_fallback'] = norm_relative_error(first.solve_host(rhs), reference)
        second = runtime.factorize_host(spd, real=True, positive_definite=True)   # a new object, same pattern
        counters = runtime.work_counters()
        if (second is first or second.n_plans != 1 or second.factor_precision != 'double'
                or second.matrix_kind != 'spd' or second.last_double_fallback
                or 'remembered' not in (second.mixed_rejection or '')):
            failed.append('mixed_rejection_remembered')
        if (counters['cudss_plans'] != 3 or counters['cudss_double_fallbacks'] != 1
                or counters['cudss_reused_orderings'] != 2):   # the kept ordering of the pattern
            failed.append('mixed_rejection_remembered_counted')
        errors['cudss_mixed_remembered'] = norm_relative_error(second.solve_host(rhs), reference)
        first.release()
        second.release()
    third = CudssFactorization(spd, real=True, positive_definite=True, precision='mixed')
    if not third.mixed_factor:   # the store ended with the block
        failed.append('mixed_rejection_forgotten_after_block')
    third.release()


_PENCIL_OMEGAS = (0.3, 0.9, 1.4)   # rad/s of K + j omega C - omega^2 M: inside the synthetic pencil's spectrum
_PENCIL_LOSS = 0.02                # K = Laplacian (1 + j loss)


def _aligned_terms(matrices):
    """(union CSR, data of each matrix on the union pattern) by fem.assembly.pattern_sum (explicit zeros kept)."""
    import scipy.sparse
    from ..fem.assembly import pattern_sum
    union = pattern_sum(matrices)
    zero = scipy.sparse.csr_matrix((np.zeros(union.nnz), union.indices, union.indptr), shape=union.shape)
    return union, [pattern_sum([matrix, zero]).data for matrix in matrices]


def _double_deferred_checks(runtime, pencil, factor, rhs, operator, failed, errors):
    """R3 / E7-7a on a double pencil factor: the pencil's own pattern arrays are matched by identity; a deferred check
    that misses once turns on cuDSS refinement and passes on the re-solve (kept by the next factor); one that misses
    again raises CudssAccuracyError from the solve and releases the factor."""
    import scipy.sparse.linalg
    from .cudss_factor import REFINEMENT_STEPS, CudssAccuracyError
    omega = _PENCIL_OMEGAS[1]
    coefficients = [1.0, 1j * omega, -omega * omega]
    factor = runtime.factorize_pencil(pencil, coefficients, previous=factor)
    source = factor._pattern_source
    if source is None or source[0] is not pencil.indptr or source[1] is not pencil.indices:
        failed.append('cudss_pattern_identity_kept')
    calls = []

    def miss_once(rhs_vector, solution, judge=factor._double_solve_accepted):
        calls.append(len(calls))
        return False if len(calls) == 1 else judge(rhs_vector, solution)

    factor._double_solve_accepted = miss_once
    errors['cudss_deferred_refined'] = norm_relative_error(factor.solve_host(rhs),
                                                       scipy.sparse.linalg.spsolve(operator(omega), rhs))
    if len(calls) != 2 or factor.refinement_steps != REFINEMENT_STEPS or factor._check_pending:
        failed.append('cudss_deferred_check_refined')
    factor = runtime.factorize_pencil(pencil, coefficients, previous=factor)
    if not factor._check_pending or factor.refinement_steps != REFINEMENT_STEPS:   # ir_num_steps of the plan kept
        failed.append('cudss_deferred_refinement_kept')
    factor._double_solve_accepted = lambda rhs_vector, solution: False
    try:
        factor.solve_host(rhs)
        failed.append('cudss_deferred_check_raises')
    except CudssAccuracyError:
        if factor.valid:
            failed.append('cudss_deferred_check_released')


def _lower_pencil_checks(runtime, pencil, full_terms, operator, failed, errors):
    """R4: a symmetric DevicePencil stores each term's lower triangle only (less device memory; full_term mirrors it
    back exactly) and a factor planned from it leaves the operator norm lazy, equal to the host route's eager one;
    E7-8: slots of a symmetric block's lower entries (upper / absent entries refused) and factorize_pencil(extra=)."""
    import cupy as cp
    from .cudss_factor import CudssFactorization
    if not pencil.lower_stored or pencil.device_bytes >= sum(int(np.asarray(term).nbytes) for term in full_terms):
        failed.append('lower_stored_pencil')
    errors['lower_pencil_full_term'] = max(
        float(np.max(np.abs(cp.asnumpy(pencil.full_term(index)) - term))) / float(np.max(np.abs(term)))
        for index, term in enumerate(full_terms))
    omega = _PENCIL_OMEGAS[0]
    coefficients = [1.0, 1j * omega, -omega * omega]
    factor = runtime.factorize_pencil(pencil, coefficients)   # planned from the lower values, probed
    factor = runtime.factorize_pencil(pencil, coefficients, previous=factor)   # refreshed by the kernel, deferred
    if factor._operator_norm is not None:   # nothing has read it yet
        failed.append('lower_pencil_norm_lazy')
    host = CudssFactorization(operator(omega).tocsr(), symmetric=True)   # the host route: eager norm of E
    errors['lower_pencil_lazy_norm'] = abs(factor.operator_norm - host.operator_norm) / host.operator_norm
    host.release()
    # E7-8: a small symmetric block (a port DtN stand-in) added on the device at its lower-triangle slots
    import scipy.sparse
    import scipy.sparse.linalg
    n_unknowns = pencil.n_rows
    rows, columns, values = np.array([5, 6, 6]), np.array([5, 5, 6]), np.array([0.3 + 0.2j, -0.1j, 0.4])
    block = scipy.sparse.csr_matrix((np.concatenate([values, values[1:2]]), (np.concatenate([rows, [5]]),
                                     np.concatenate([columns, [6]]))), shape=(n_unknowns, n_unknowns))
    slots = pencil.slots(rows, columns)
    if slots is None or pencil.slots([5], [6]) is not None or pencil.slots([0], [n_unknowns - 1]) is not None:
        failed.append('pencil_slots')   # lower entries found; an upper or an absent entry refused
    else:
        factor = runtime.factorize_pencil(pencil, coefficients, previous=factor, extra=(slots, values))
        rhs = np.random.default_rng(_SEED + 11).standard_normal(n_unknowns) + 0j
        errors['pencil_extra_block'] = norm_relative_error(factor.solve_host(rhs), scipy.sparse.linalg.spsolve(
            (operator(omega) + block).tocsc(), rhs))
    factor.release()


def _check_cudss_device_pencil(results_by_name):
    """DevicePencil + SparseDirectSolver.factorize_pencil (cudss / cudss_mixed) against scipy spsolve: a complex
    symmetric pencil (LDL^T, one plan, a caller row_scale), a real pencil, an unsymmetric term (general LU) and
    mechanics frequency_response.HarmonicSystem on a cuDSS runtime against its private SuperLU."""
    import scipy.sparse
    import scipy.sparse.linalg
    from ..core.backend import gpu_available
    from ..physics.mechanics.frequency_response import HarmonicSystem
    from .cudss_factor import ACCEPT_RESIDUAL, BACKWARD_ERROR_TOLERANCE, cudss_available
    from .sparse_direct import CUDSS_BACK_ENDS, SparseDirectSolver
    failed, errors, details = [], {}, {}
    if not (gpu_available() and cudss_available()):
        return _outcome({'skipped': 0.0}, failed, {'skipped': 'no usable GPU or nvmath-python (cuDSS) not installed'})
    rng = np.random.default_rng(_SEED + 9)
    nx, ny = _GRID
    n_unknowns = nx * ny
    laplacian, mass = _laplacian(nx, ny), _consistent_mass(nx, ny)
    stiffness = (laplacian * (1.0 + 1j * _PENCIL_LOSS)).tocsr()
    damping = scipy.sparse.diags(rng.uniform(0.01, 0.1, n_unknowns)).tocsr()
    union, (stiffness_data, damping_data, mass_data) = _aligned_terms([stiffness, damping, mass])
    rhs = rng.standard_normal(n_unknowns) + 1j * rng.standard_normal(n_unknowns)

    def operator(omega, matrix=stiffness):
        return (matrix + 1j * omega * damping - omega * omega * mass).tocsc()

    for back_end in CUDSS_BACK_ENDS:
        runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
        pencil = runtime.device_pencil(union.indptr, union.indices, [stiffness_data, damping_data, mass_data],
                                       symmetric=True)
        if pencil is None or not pencil.symmetric:
            failed.append(f'{back_end}_symmetric_pencil')
            continue
        factor = None
        for index, omega in enumerate(_PENCIL_OMEGAS):
            factor = runtime.factorize_pencil(pencil, [1.0, 1j * omega, -omega * omega], previous=factor)
            deferred = index > 0   # reused plan: the solve is the check (double too, R3)
            if factor._check_pending != deferred:
                failed.append(f'{back_end}_check_pending_{omega:g}')
            errors[f'{back_end}_omega_{omega:g}'] = norm_relative_error(
                factor.solve_host(rhs), scipy.sparse.linalg.spsolve(operator(omega), rhs))
            backward = factor.probe_backward_error   # None: the relative residual decided alone (R4, double)
            checked = (backward <= BACKWARD_ERROR_TOLERANCE if backward is not None
                       else factor.probe_residual <= ACCEPT_RESIDUAL)
            if factor._check_pending or not checked:
                failed.append(f'{back_end}_checked_{omega:g}')
        omega = _PENCIL_OMEGAS[-1]
        scale = 1.0 / np.sqrt(np.abs(laplacian.diagonal()) + omega * omega * np.abs(mass.diagonal()))
        factor = runtime.factorize_pencil(pencil, [1.0, 1j * omega, -omega * omega], row_scale=scale, previous=factor)
        errors[f'{back_end}_row_scale'] = norm_relative_error(factor.solve_host(rhs),
                                                          scipy.sparse.linalg.spsolve(operator(omega), rhs))
        if not factor.symmetric:
            failed.append(f'{back_end}_ldlt')
        counters = runtime.work_counters()
        if counters['cudss_plans'] != 1 or counters['factorizations'] != len(_PENCIL_OMEGAS) + 1:
            failed.append(f'{back_end}_one_plan')
        details[f'{back_end}_counters'] = counters
        if back_end == 'cudss_mixed':   # a deferred check that misses: double re-planned in place, the rhs solved again
            omega = _PENCIL_OMEGAS[0]
            factor = runtime.factorize_pencil(pencil, [1.0, 1j * omega, -omega * omega], previous=factor)
            factor._single_solve = lambda vector: vector.copy()   # a low-precision solve that lost its accuracy
            errors['cudss_mixed_deferred_fallback'] = norm_relative_error(
                factor.solve_host(rhs), scipy.sparse.linalg.spsolve(operator(omega), rhs))
            counters = runtime.work_counters()
            if (factor.factor_precision != 'double' or not factor.last_double_fallback
                    or counters['cudss_double_fallbacks'] != 1 or counters['cudss_plans'] != 2):
                failed.append('cudss_mixed_deferred_fallback')
            details['cudss_mixed_fallback_ordering'] = {'kept': factor._ordering is not None,
                                                        'reused': factor.last_ordering_reused}
            if factor._ordering is not None and not factor.last_ordering_reused:   # re-plan of its own pattern
                failed.append('cudss_mixed_fallback_ordering_reused')
            if counters['cudss_reused_orderings'] != int(factor.last_ordering_reused):   # R1b: the listener counts it
                failed.append('cudss_mixed_fallback_reuse_counted')
            details['cudss_mixed_fallback_counters'] = counters
            details['cudss_mixed_rejection'] = factor.mixed_rejection
        else:
            _double_deferred_checks(runtime, pencil, factor, rhs, operator, failed, errors)
            _lower_pencil_checks(runtime, pencil, [stiffness_data, damping_data, mass_data], operator, failed, errors)
        factor.release()
    runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
    real_union, real_terms = _aligned_terms([laplacian, mass])
    real_pencil = runtime.device_pencil(real_union.indptr, real_union.indices, real_terms, symmetric=True)
    factor = runtime.factorize_pencil(real_pencil, [1.0, -0.01])
    real_solution = factor.solve_host(rhs.real)
    if real_solution.dtype != np.float64 or not factor.symmetric:
        failed.append('cudss_real_pencil')
    errors['cudss_real_pencil'] = norm_relative_error(real_solution, scipy.sparse.linalg.spsolve(
        (laplacian - 0.01 * mass).tocsc(), rhs.real))
    factor.release()
    forward = scipy.sparse.kron(scipy.sparse.diags([-np.ones(nx), np.ones(nx - 1)], [0, 1]),
                                scipy.sparse.identity(ny))
    unsymmetric = (stiffness + 0.3 * forward).tocsr()
    unsymmetric_union, unsymmetric_terms = _aligned_terms([unsymmetric, damping, mass])
    unsymmetric_pencil = runtime.device_pencil(unsymmetric_union.indptr, unsymmetric_union.indices,
                                               unsymmetric_terms, symmetric=True)
    if unsymmetric_pencil.symmetric:
        failed.append('cudss_unsymmetric_pencil_detected')
    omega = _PENCIL_OMEGAS[1]
    factor = runtime.factorize_pencil(unsymmetric_pencil, [1.0, 1j * omega, -omega * omega])
    if factor.symmetric:
        failed.append('cudss_unsymmetric_general_lu')
    errors['cudss_unsymmetric'] = norm_relative_error(factor.solve_host(rhs), scipy.sparse.linalg.spsolve(
        operator(omega, unsymmetric), rhs))
    factor.release()
    # mechanics frequency_response.HarmonicSystem: (K + j eta K - omega^2 M) with two held dofs and the row_scale
    real_stiffness = (laplacian + scipy.sparse.diags(rng.uniform(0.01, 0.1, n_unknowns))).tocsr()
    free = np.ones(n_unknowns, dtype=bool)
    free[[0, n_unknowns - 1]] = False
    fixed = np.zeros(n_unknowns, dtype=np.complex128)
    fixed[[0, n_unknowns - 1]] = (1.0, 0.5j)
    private = HarmonicSystem(real_stiffness, _PENCIL_LOSS * real_stiffness, mass, free)
    for back_end in CUDSS_BACK_ENDS:
        runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
        harmonic = HarmonicSystem(real_stiffness, _PENCIL_LOSS * real_stiffness, mass, free, runtime=runtime)
        for omega in _PENCIL_OMEGAS:
            solution, _ = harmonic.solve(omega, fixed, rhs)
            expected, _ = private.solve(omega, fixed, rhs)
            errors[f'{back_end}_frequency_response_{omega:g}'] = norm_relative_error(solution, expected)
        if not harmonic._pencil or runtime.work_counters()['cudss_plans'] != 1:
            failed.append(f'{back_end}_frequency_response_device_pencil')
        harmonic.factor.release()
    _ordering_reuse_checks(failed, errors, details)
    return _outcome(errors, failed, details)


_ORDERING_AXIS = 160   # cells per axis of the ordering-reuse operator (25,600 unknowns: a nested-dissection plan)


def _ordering_reuse_checks(failed, errors, details):
    """cuDSS ordering reuse (cudss_factor "Ordering reuse"): inside cudss_orderings_kept a second factor object of the
    same pattern plans on the first one's permutation + partition tree (same lu_nnz, solution right, counted); the store
    is off outside the block, where a fresh factor plans from scratch; a real Cholesky's ordering serves a complex
    LDL^T of the same pattern (same fill as fresh), an LU between them plans fresh (its own family)."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .cudss_factor import cudss_orderings_kept, remembers_cudss_orderings
    from .sparse_direct import SparseDirectSolver
    n_unknowns = _ORDERING_AXIS * _ORDERING_AXIS
    operator = (_laplacian(_ORDERING_AXIS, _ORDERING_AXIS) * (1.0 + 1j * _PENCIL_LOSS)
                - 0.05 * scipy.sparse.identity(n_unknowns)).tocsr()   # complex symmetric: LDL^T
    rng = np.random.default_rng(_SEED + 10)
    rhs = rng.standard_normal(n_unknowns) + 1j * rng.standard_normal(n_unknowns)
    reference = scipy.sparse.linalg.spsolve(operator.tocsc(), rhs)
    if remembers_cudss_orderings():
        failed.append('ordering_store_on_outside_block')
    with cudss_orderings_kept():
        runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
        first = runtime.factorize_host(operator, symmetric=True)
        second = runtime.factorize_host(operator, symmetric=True)   # a new object (no previous=): the kept ordering
        errors['ordering_reuse_first'] = norm_relative_error(first.solve_host(rhs), reference)
        errors['ordering_reuse_second'] = norm_relative_error(second.solve_host(rhs), reference)
        counters = runtime.work_counters()
        details['ordering_reuse'] = {'plan_seconds': (first.plan_seconds, second.plan_seconds),
                                     'lu_entries': (first.lu_entries, second.lu_entries),
                                     'kinds': (first.matrix_kind, second.matrix_kind),
                                     'tree_entries': None if first._ordering is None else int(first._ordering[2].size),
                                     'cudss_plans': counters['cudss_plans'],
                                     'cudss_reused_orderings': counters['cudss_reused_orderings']}
        if first._ordering is None:
            failed.append('ordering_not_kept')
        if first.last_ordering_reused or not second.last_ordering_reused:
            failed.append('ordering_reused_by_second_factor')
        if second.lu_entries != first.lu_entries:
            failed.append('ordering_reuse_same_fill')
        if counters['cudss_plans'] != 2 or counters['cudss_reused_orderings'] != 1:
            failed.append('ordering_reuse_counted')
        first.release()
        second.release()
    if remembers_cudss_orderings():
        failed.append('ordering_store_on_after_block')
    fresh = SparseDirectSolver(back_end='cudss', backend='gpu').factorize_host(operator, symmetric=True)
    if fresh.last_ordering_reused:
        failed.append('ordering_reused_outside_block')
    fresh_entries = fresh.lu_entries
    fresh.release()
    real_spd = _laplacian(_ORDERING_AXIS, _ORDERING_AXIS)   # the same pattern, real SPD
    with cudss_orderings_kept():   # one ordering family: the real Cholesky's ordering serves the complex LDL^T
        runtime = SparseDirectSolver(back_end='cudss', backend='gpu')
        cholesky = runtime.factorize_host(real_spd, real=True, positive_definite=True)
        general = runtime.factorize_host(real_spd, real=True)   # LU: its own family, plans fresh
        ldlt = runtime.factorize_host(operator, symmetric=True)
        errors['ordering_reuse_cholesky_to_ldlt'] = norm_relative_error(ldlt.solve_host(rhs), reference)
        details['ordering_reuse_cross_kind'] = {
            'kinds': (cholesky.matrix_kind, general.matrix_kind, ldlt.matrix_kind),
            'reused': (cholesky.last_ordering_reused, general.last_ordering_reused, ldlt.last_ordering_reused),
            'lu_entries': (fresh_entries, ldlt.lu_entries)}
        if (cholesky.matrix_kind != 'spd' or ldlt.matrix_kind != 'symmetric' or cholesky.last_ordering_reused
                or general.last_ordering_reused or not ldlt.last_ordering_reused):
            failed.append('ordering_reuse_cholesky_to_ldlt')
        if ldlt.lu_entries != fresh_entries:
            failed.append('ordering_reuse_cross_kind_same_fill')
        for factor in (cholesky, general, ldlt):
            factor.release()


# --- solver_defaults.precision (S9) -------------------------------------------------------------------------------

# (check name, physics type, role, study, default precision, node (type, settings) or None, backend,
#  expected (resolved type, resolved precision, listed by default_precision_ignored))
_PRECISION_CASES = (
    ('heat_auto', 'heat', 'linear', 'stationary', 'auto', None, 'gpu', ('pcg', 'double', False)),
    ('heat_single', 'heat', 'linear', 'stationary', 'single', None, 'gpu', ('pcg', 'single', False)),
    ('heat_mixed', 'heat', 'linear', 'stationary', 'mixed', None, 'gpu', ('pcg', 'mixed', False)),
    ('heat_node_filled', 'heat', 'linear', 'stationary', 'single', ('pcg', {'tolerance': 1e-6}), 'gpu',
     ('pcg', 'single', False)),
    ('heat_node_own_precision', 'heat', 'linear', 'stationary', 'single', ('pcg', {'precision': 'double'}), 'gpu',
     ('pcg', 'double', True)),
    ('heat_node_same_precision', 'heat', 'linear', 'stationary', 'single', ('pcg', {'precision': 'single'}), 'gpu',
     ('pcg', 'single', False)),
    ('heat_cpu_direct', 'heat', 'linear', 'stationary', 'single', None, 'cpu', ('sparse_direct', 'double', True)),
    ('solid_single', 'solid_mechanics', 'linear', 'stationary', 'single', None, 'gpu', ('pcg', 'single', False)),
    # solid_mechanics (tensor): its role accepts mixed, Automatic on the gpu backend is pcg mixed (as the 3D type)
    ('solid_mixed', 'solid_mechanics', 'linear', 'stationary', 'mixed', None, 'gpu', ('pcg', 'mixed', False)),
    ('solid_auto_gpu_mixed', 'solid_mechanics', 'linear', 'stationary', 'auto', None, 'gpu', ('pcg', 'mixed', False)),
    ('solid_auto_cpu_double', 'solid_mechanics', 'linear', 'stationary', 'auto', None, 'cpu',
     ('pcg', 'double', False)),
    ('solid_default_double_wins', 'solid_mechanics', 'linear', 'stationary', 'double', None, 'gpu',
     ('pcg', 'double', False)),
    ('solid_3d_mixed', 'solid_mechanics_3d', 'linear', 'stationary', 'mixed', None, 'gpu', ('pcg', 'mixed', False)),
    ('solid_3d_auto_gpu_mixed', 'solid_mechanics_3d', 'linear', 'stationary', 'auto', None, 'gpu',
     ('pcg', 'mixed', False)),
    ('solid_3d_auto_cpu_double', 'solid_mechanics_3d', 'linear', 'stationary', 'auto', None, 'cpu',
     ('pcg', 'double', False)),
    ('solid_3d_default_double_wins', 'solid_mechanics_3d', 'linear', 'stationary', 'double', None, 'gpu',
     ('pcg', 'double', False)),
    # a node of the Automatic type that sets no precision takes the Automatic one (schema.with_node_precision)
    ('solid_3d_node_takes_automatic_mixed', 'solid_mechanics_3d', 'linear', 'stationary', 'auto',
     ('pcg', {'tolerance': 1e-10}), 'gpu', ('pcg', 'mixed', False)),
    ('solid_3d_node_cpu_double', 'solid_mechanics_3d', 'linear', 'stationary', 'auto',
     ('pcg', {'tolerance': 1e-10}), 'cpu', ('pcg', 'double', False)),
    ('solid_3d_node_default_double_wins', 'solid_mechanics_3d', 'linear', 'stationary', 'double',
     ('pcg', {'tolerance': 1e-10}), 'gpu', ('pcg', 'double', False)),
    ('solid_3d_node_own_double', 'solid_mechanics_3d', 'linear', 'stationary', 'auto',
     ('pcg', {'precision': 'double'}), 'gpu', ('pcg', 'double', False)),
    ('fdtd_double', 'em_fdtd_ez', 'stepper', 'time_dependent', 'double', None, 'gpu',
     ('explicit_kernels', 'double', False)),
    ('fdtd_half_storage_refused_by_role', 'em_fdtd_ez', 'stepper', 'time_dependent', 'half_storage', None, 'gpu',
     ('explicit_kernels', 'single', True)),
)


def _precision_spec(physics_type, role, study, precision, node, backend, size_z=None):
    """Minimal ModelSpec of one physics 'p' (node: its `role` solver 'node', else Automatic; size_z: a 3D model)."""
    from ..model.spec import DomainSpec, ModelSpec, PhysicsSpec, SolverDefaultsSpec, SolverSpec, StudySpec
    spec = ModelSpec(domain=DomainSpec(size_z=size_z), physics=[PhysicsSpec('p', physics_type)],
                     study=StudySpec(type=study), solver_defaults=SolverDefaultsSpec(backend=backend, precision=precision))
    if node is not None:
        spec.solvers = [SolverSpec('node', node[0], dict(node[1]))]
        spec.physics[0].solvers = {role: 'node'}
    return spec


def _check_default_precision(results_by_name):
    """schema.resolved_solvers / default_precision_ignored / with_default_precision against the S9 rules (CPU)."""
    from ..model.schema import (SolverRole, default_precision_ignored, get_registry, resolved_precision,
                                resolved_solvers, with_default_precision)
    registry = get_registry()
    mismatches, outcomes = [], {}
    for name, physics_type, role, study, precision, node, backend, expected in _PRECISION_CASES:
        spec = _precision_spec(physics_type, role, study, precision, node, backend)
        resolved = resolved_solvers(spec, registry, spec.physics[0], study, {})[role]
        ignored = any(entry[:2] == ('p', role) for entry in default_precision_ignored(spec, registry, study))
        got = (resolved.solver_type.name, resolved_precision(resolved.solver_type, resolved.values), ignored)
        outcomes[name] = {'got': list(got), 'expected': list(expected)}
        if got != expected:
            mismatches.append(name)
        if node is not None and spec.solvers[0].settings != node[1]:
            mismatches.append(f'{name}_node_settings_unchanged')
    pcg_type = registry.solver_type('pcg')
    # a pcg role of the default precisions (single / double): no registered pcg role refuses mixed any more
    refusing_role = SolverRole('spd', 'pcg')
    mixed_spec = _precision_spec('solid_mechanics', 'linear', 'stationary', 'mixed', None, 'gpu')
    direct = {'no_role_fills': (with_default_precision(mixed_spec, pcg_type, {}), {'precision': 'mixed'}),
              'role_refuses': (with_default_precision(mixed_spec, pcg_type, {}, refusing_role), {}),
              'explicit_wins': (with_default_precision(mixed_spec, pcg_type, {'precision': 'double'}),
                                {'precision': 'double'})}
    for name, (got, expected) in direct.items():
        outcomes[f'with_default_precision_{name}'] = {'got': got, 'expected': expected}
        if got != expected:
            mismatches.append(f'with_default_precision_{name}')
    _multigrid_resolution_checks(registry, mismatches, outcomes)
    n_checks = len(_PRECISION_CASES) + len(direct) + len(_MULTIGRID_CASES) + len(_MULTIGRID_3D_CASES)
    return {'measured': len(mismatches), 'expected': 0, 'error': float(len(mismatches)), 'tolerance': 0.0,
            'details': {'n_checks': n_checks, 'mismatches': mismatches, 'outcomes': outcomes}}


# --- Automatic multigrid of the diffusion types (mesh plan M0 / D16) ------------------------------------------------

# (check name, physics type, default precision, node (type, settings) or None, backend,
#  expected (resolved type, resolved precision, resolved preconditioner or None))
_MULTIGRID_CASES = (
    ('heat_auto', 'heat', 'auto', None, 'gpu', ('pcg', 'double', 'multigrid')),
    ('currents_auto', 'electric_currents', 'auto', None, 'gpu', ('pcg', 'double', 'multigrid')),
    ('electrostatics_auto', 'electrostatics', 'auto', None, 'gpu', ('pcg', 'double', 'multigrid')),
    ('magnetostatics_auto', 'magnetostatics', 'auto', None, 'gpu', ('pcg', 'double', 'multigrid')),
    ('heat_single', 'heat', 'single', None, 'gpu', ('pcg', 'single', 'multigrid')),
    ('heat_mixed_automatic', 'heat', 'mixed', None, 'gpu', ('pcg', 'mixed', 'jacobi')),
    ('heat_mixed_node', 'heat', 'auto', ('pcg', {'precision': 'mixed'}), 'gpu', ('pcg', 'mixed', 'jacobi')),
    ('heat_mixed_default_node', 'heat', 'mixed', ('pcg', {'tolerance': 1e-6}), 'gpu', ('pcg', 'mixed', 'jacobi')),
    ('heat_node_default_jacobi', 'heat', 'auto', ('pcg', {}), 'gpu', ('pcg', 'double', 'jacobi')),
    ('heat_node_multigrid_single', 'heat', 'auto', ('pcg', {'precision': 'single', 'preconditioner': 'multigrid'}),
     'gpu', ('pcg', 'single', 'multigrid')),
    ('heat_cpu_direct', 'heat', 'auto', None, 'cpu', ('sparse_direct', 'double', None)),
    ('solid_auto_jacobi', 'solid_mechanics', 'auto', None, 'gpu', ('pcg', 'mixed', 'jacobi')),
)
# the same rows in a 3D model (grid 'tensor_3d': Automatic pcg multigrid on the cpu too)
_MULTIGRID_3D_CASES = (
    ('heat_3d_auto', 'heat', 'auto', None, 'gpu', ('pcg', 'double', 'multigrid')),
    ('heat_3d_cpu_auto', 'heat', 'auto', None, 'cpu', ('pcg', 'double', 'multigrid')),
    ('heat_3d_cpu_mixed_automatic', 'heat', 'mixed', None, 'cpu', ('pcg', 'mixed', 'jacobi')),
    ('heat_3d_cpu_direct_node', 'heat', 'auto', ('sparse_direct', {}), 'cpu', ('sparse_direct', 'double', None)),
)
_MULTIGRID_3D_SIZE_Z = 4e-6


def _multigrid_resolution_checks(registry, mismatches, outcomes):
    """'linear' role resolution against M0 (Automatic diffusion pcg = multigrid, mixed = jacobi; 3D on both backends),
    appended as 'multigrid_<name>': the pcg node validate passes and linear_solver_multigrid (the estimators) follows
    it."""
    from types import SimpleNamespace

    from ..model.definition_helpers import linear_solver_multigrid
    from ..model.schema import resolved_precision, resolved_solvers
    rows = ([row + (None,) for row in _MULTIGRID_CASES]
            + [row + (_MULTIGRID_3D_SIZE_Z,) for row in _MULTIGRID_3D_CASES])
    for row_name, physics_type, precision, node, backend, expected, size_z in rows:
        name = f'multigrid_{row_name}'
        spec = _precision_spec(physics_type, 'linear', 'stationary', precision, node, backend, size_z)
        resolved = resolved_solvers(spec, registry, spec.physics[0], 'stationary', {})['linear']
        solver_type, values = resolved.solver_type, resolved.values
        got = (solver_type.name, resolved_precision(solver_type, values), values.get('preconditioner'))
        issues = ([] if solver_type.validate is None
                  else solver_type.validate(SimpleNamespace(spec=spec), values, ('solver', 'linear')))
        errors = [issue.message for issue in issues if issue.level == 'error']
        estimator_multigrid = linear_solver_multigrid(SimpleNamespace(solvers={'linear': resolved}))
        outcomes[name] = {'got': list(got), 'expected': list(expected), 'validate_errors': errors,
                          'estimator_multigrid': estimator_multigrid}
        if got != expected or errors or estimator_multigrid != (expected[2] == 'multigrid'):
            mismatches.append(name)
        if node is not None and spec.solvers[0].settings != node[1]:
            mismatches.append(f'{name}_node_settings_unchanged')


# --- shift-invert core (B1): inertia counts, plan reuse across shifts, band / multi-shift merge ----------------------
# Inertia: K = the _GRID Laplacian, M = the bilinear consistent mass (9-point pattern, so K - 0 M keeps M's entries);
# sigma halfway between neighbouring exact eigenvalues (-1: below the lowest).
_INERTIA_INDICES = (-1, 37, 150, 400)
_INERTIA_EIGENPAIRS = 4
# Band: square 12 x 12 Laplacian, M = 2 I (lambda_pq = lambda_qp: degenerate pairs), edges halfway around these indices
_BAND_GRID = 12
_BAND_INDICES = (5, 33)
_BAND_PER_SLICE = 4
_BAND_TARGET_INDICES = (10, 12)    # two overlapping 'nearest' runs of _BAND_PER_SLICE modes
_BAND_TOLERANCE = 1e-8             # relative eigenvalue error, merge clusters
_BAND_ORTHOGONALITY = 1e-6         # largest |cos| between two returned modes in the M inner product


def _consistent_mass(nx, ny):
    """Bilinear consistent mass of unit cells on the i * ny + j order (scipy CSR, SPD, 9-point pattern)."""
    import scipy.sparse as sparse

    def line_mass(n):
        ones = np.ones(n)
        return sparse.diags([ones[:-1] / 6.0, 2.0 * ones / 3.0, ones[:-1] / 6.0], [-1, 0, 1])

    return sparse.kron(line_mass(nx), line_mass(ny)).tocsr()


def _check_eigen_inertia(results_by_name):
    """count_below and eigsh_shifted(inertia=True) against the dense eigh count per sigma: SuperLU, and on the GPU cuDSS
    double (exact) and mixed (recorded only); one cuDSS plan for every shift; the counting SuperLU directly."""
    import scipy.linalg
    from .arpack_shift_invert import EIGSH_SHIFT_RELATIVE_OFFSET, ArpackShiftInvertSolver, pattern_combination
    from .cudss_factor import cudss_available
    from .sparse_direct import superlu_counting_inertia
    nx, ny = _GRID
    stiffness, mass = _laplacian(nx, ny), _consistent_mass(nx, ny)
    exact = np.sort(scipy.linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True))
    sigmas = [0.5 * exact[0] if index < 0 else 0.5 * (exact[index] + exact[index + 1]) for index in _INERTIA_INDICES]
    factorizations = ['superlu']
    if 'gpu' in _backends() and cudss_available():
        factorizations += ['cudss', 'cudss_mixed']
    outcomes, mismatches = {}, []
    for factorization in factorizations:
        solver = ArpackShiftInvertSolver(factorization=factorization)
        factor = None
        for index, sigma in zip(_INERTIA_INDICES, sigmas):
            expected = int(np.count_nonzero(exact < sigma))
            counted, factor = solver.count_below(stiffness, mass, sigma, previous=factor)
            result = solver.eigsh_shifted(stiffness, mass, sigma / (1.0 + EIGSH_SHIFT_RELATIVE_OFFSET),
                                          _INERTIA_EIGENPAIRS, previous=factor, inertia=True)
            factor, shifted = result[4], result[5]
            got = [None if item is None else item.n_below for item in (counted, shifted)]
            key = f'{factorization}_{index}'
            outcomes[key] = {'expected': expected, 'count_below': got[0], 'eigsh_shifted': got[1],
                             'exact': None if shifted is None else shifted.exact}
            if factorization != 'cudss_mixed' and got != [expected, expected]:
                mismatches.append(key)
        if factorization != 'superlu':
            outcomes[f'{factorization}_plans'] = factor.n_plans
            if factor.n_plans != 1:
                mismatches.append(f'{factorization}_one_plan')
            factor.release()
    direct = superlu_counting_inertia(pattern_combination((stiffness, mass), (1.0, -sigmas[2])))
    outcomes['superlu_counting'] = direct
    if direct is None or direct[1] != int(np.count_nonzero(exact < sigmas[2])):
        mismatches.append('superlu_counting')
    error = float(len(mismatches))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 0.0,
            'details': {'mismatches': mismatches, 'factorizations': factorizations, 'outcomes': outcomes}}


def _band_edge(exact, index):
    """lambda halfway between exact[i - 1] and exact[i] for the first i >= index with a relative gap > 1e-6."""
    while exact[index] - exact[index - 1] <= 1e-6 * exact[index]:
        index += 1
    return 0.5 * (exact[index - 1] + exact[index])


def _clustered_union(runs, tolerance):
    """Sorted multiset union of eigenvalue runs: per cluster (relative tolerance) the largest count any run has."""
    values = sorted(value for run in runs for value in run)
    union, cluster = [], []
    for value in values + [None]:
        if cluster and (value is None or value - cluster[-1] > tolerance * value):
            union += [cluster[0]] * max(sum(abs(item - cluster[0]) <= tolerance * cluster[0] for item in run)
                                        for run in runs)
            cluster = []
        if value is not None:
            cluster.append(value)
    return np.array(union)


def _distinct(values, tolerance):
    """First value of each cluster (relative tolerance) of sorted values."""
    kept = []
    for value in values:
        if not kept or value - kept[-1] > tolerance * value:
            kept.append(value)
    return np.array(kept)


def _check_eigen_band(results_by_name):
    """solve_band_modes vs dense eigh (every eigenvalue in the band once, degenerate pairs twice, expected_count, M
    orthogonal modes) and solve_modes_at_shifts vs the clustered union of two solve_modes runs."""
    import math
    import scipy.linalg
    import scipy.sparse
    from ..physics.mechanics.modal import ModalPencil, solve_band_modes, solve_modes, solve_modes_at_shifts
    n_side = _BAND_GRID
    n_unknowns = n_side * n_side
    stiffness = _laplacian(n_side, n_side)
    mass = 2.0 * scipy.sparse.identity(n_unknowns, format='csr')
    exact = np.sort(scipy.linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True))
    lambda_lo, lambda_hi = _band_edge(exact, _BAND_INDICES[0]), _band_edge(exact, _BAND_INDICES[1] + 1)
    reference = exact[(exact > lambda_lo) & (exact < lambda_hi)]
    free = np.ones(n_unknowns, dtype=bool)

    def hertz(value):
        return math.sqrt(value) / (2.0 * math.pi)

    band = solve_band_modes(ModalPencil(stiffness, mass, free, 1), hertz(lambda_lo), hertz(lambda_hi),
                            _BAND_PER_SLICE, 200)
    found = np.sort([resonance.omega ** 2 for resonance in band.resonances])
    failed, errors = [], {}
    if found.size != reference.size:
        failed.append('band_count')
    else:
        errors['band'] = float(np.max(np.abs(found - reference) / reference))
    if band.info.get('expected_count') != reference.size:
        failed.append('band_expected_count')
    modes = np.column_stack([resonance.displacement for resonance in band.resonances])
    gram = modes.T @ (mass @ modes)
    scale = np.sqrt(np.diag(gram))
    cosines = np.abs(gram) / np.outer(scale, scale) - np.identity(len(scale))
    orthogonality = float(np.max(cosines)) if cosines.size else 0.0
    if not orthogonality <= _BAND_ORTHOGONALITY:
        failed.append('band_orthogonality')
    degenerate = int(np.count_nonzero(np.diff(reference) <= _BAND_TOLERANCE * reference[1:]))
    if degenerate == 0:
        failed.append('band_has_degenerate_pair')
    targets = [math.sqrt(exact[index]) * (1.0 + 1e-3) for index in _BAND_TARGET_INDICES]
    runs = []
    for target in targets:
        nearest = solve_modes(stiffness, mass, free, target, _BAND_PER_SLICE, 1)
        runs.append([resonance.omega ** 2 for resonance in nearest.resonances])
    union = _clustered_union(runs, _BAND_TOLERANCE)
    merged = np.sort([resonance.omega ** 2 for resonance in solve_modes_at_shifts(
        ModalPencil(stiffness, mass, free, 1), targets, _BAND_PER_SLICE).resonances])
    # a degenerate pair cut by two runs may come back as two independent vectors: compare distinct values, bound counts
    distinct_union, distinct_merged = _distinct(union, _BAND_TOLERANCE), _distinct(merged, _BAND_TOLERANCE)
    if not (union.size <= merged.size <= sum(len(run) for run in runs)) or distinct_merged.size != distinct_union.size:
        failed.append('targets_union')
    else:
        errors['targets'] = float(np.max(np.abs(distinct_merged - distinct_union) / distinct_union))
    measured = max(errors.values(), default=0.0)
    error = max(measured, _STRUCTURAL_FAILURE) if failed else measured
    return {'measured': measured, 'expected': 0.0, 'error': error, 'tolerance': _BAND_TOLERANCE,
            'details': {'failed': failed, 'relative_errors': errors, 'reference_count': int(reference.size),
                        'found_count': int(found.size), 'degenerate_pairs': degenerate,
                        'orthogonality': orthogonality, 'shifts': band.info.get('shifts'),
                        'band_warnings': band.info.get('warnings'), 'union_count': int(union.size),
                        'merged_count': int(merged.size)}}


# --- host direct back ends (MUMPS / PARDISO) ------------------------------------------------------------------------

_HOST_INERTIA_SHIFT_INDEX = 150   # real symmetric indefinite K - sigma I: sigma halfway after the 151st eigenvalue


def _host_direct_role_checks(failed):
    """CPU, no wrapper needed: the host direct back ends fit the roles listing them (heat, em_fdfd_ez,
    em_fdfd_nonlinear linear; magnetic_fields_ac double only; em_fdfd_ez eigen all four, membrane eigen double only)
    and misfit em_modes 'shifted' (opt-in setting); no cpu misfit; the SuperLU size rule; resolve_back_end gives the
    name when the wrapper is installed, else raises missing_text; factorization_for maps a host *_mixed request
    below mixed_min_unknowns to its double name."""
    from ..model.schema import get_registry
    from .arpack_shift_invert import FACTORIZATIONS, ArpackShiftInvertSolver
    from .arpack_shift_invert import backend_misfit as eigen_backend_misfit
    from .host_direct import (HOST_DIRECT_BACK_ENDS, HOST_DIRECT_DOUBLE_BACK_ENDS, host_direct_available,
                              missing_text)
    from .roles import solver_misfit
    from .sparse_direct import LINEAR_SOLVER_CHOICES, backend_misfit, resolve_back_end, size_issue
    registry = get_registry()
    sparse_type, eigen_type = registry.solver_type('sparse_direct'), registry.solver_type('arpack_shift_invert')
    every, double = tuple(HOST_DIRECT_BACK_ENDS), tuple(HOST_DIRECT_DOUBLE_BACK_ENDS)
    for solver_type, setting, physics_name, role, study, grid, offered in (
            (sparse_type, 'back_end', 'heat', 'linear', 'stationary', 'tensor', every),
            (sparse_type, 'back_end', 'em_fdfd_ez', 'linear', 'stationary', 'tensor', every),
            (sparse_type, 'back_end', 'em_fdfd_nonlinear', 'linear', 'stationary', 'tensor', every),
            (sparse_type, 'back_end', 'magnetic_fields_ac', 'linear', 'stationary', 'tensor', double),
            (sparse_type, 'back_end', 'em_modes', 'shifted', 'mode_analysis', 'tensor', ()),
            (eigen_type, 'factorization', 'em_fdfd_ez', 'eigen', 'eigenfrequency', 'tensor', every),
            (eigen_type, 'factorization', 'membrane', 'eigen', 'eigenfrequency', 'triangles', double)):
        physics_type = registry.physics_type(physics_name)
        for name in HOST_DIRECT_BACK_ENDS:   # gpu: every listed physics runs there (the cpu: backend_misfit below)
            misfit = solver_misfit(solver_type, {setting: name}, physics_type, role, study, 'gpu', grid)
            if bool(misfit) == (name in offered):
                failed.append(f'{physics_name}_{role}_{name}_offered_{name in offered}')
    for name in HOST_DIRECT_BACK_ENDS:
        if name not in LINEAR_SOLVER_CHOICES or name not in FACTORIZATIONS:
            failed.append(f'{name}_listed')
        if backend_misfit({'back_end': name}, 'cpu') or eigen_backend_misfit({'factorization': name}, 'cpu'):
            failed.append(f'{name}_runs_on_cpu')
        if (size_issue(name, 5000, 5000, False, False) or ('',))[0] != 'error' or size_issue(name, 30, 20, False,
                                                                                               False) is not None:
            failed.append(f'{name}_superlu_size_rule')
        if host_direct_available(name):
            if resolve_back_end(name, 'cpu') != name:
                failed.append(f'{name}_resolves')
        else:
            try:
                resolve_back_end(name, 'cpu')
                failed.append(f'{name}_missing_raises')
            except ValueError as error:
                if str(error) != missing_text(name):
                    failed.append(f'{name}_missing_text')
    for name in HOST_DIRECT_BACK_ENDS:
        if name.endswith('_mixed'):
            solver = ArpackShiftInvertSolver(factorization=name, mixed_min_unknowns=1000)
            if solver.factorization_for(600) != name[:-len('_mixed')] or solver.factorization_for(2000) != name:
                failed.append(f'{name}_double_below_mixed_min_unknowns')


def _quiet_openblas_checks(failed, details):
    """host_direct.quiet_openblas (no wrapper needed): holds nothing for names that are not PARDISO; for a PARDISO
    name every loaded OpenBLAS pool reads one thread inside (nested too) and its earlier count after; without
    threadpoolctl or an OpenBLAS it holds nothing."""
    from .host_direct import _openblas_pools, quiet_openblas
    for name in ('mumps', 'mumps_mixed', 'scipy_splu', 'superlu', 'cudss', None):
        with quiet_openblas(name) as held:
            if held:
                failed.append(f'quiet_openblas_holds_for_{name}')
    pools = _openblas_pools(refresh=True)
    controllers = () if pools is None else tuple(pools.lib_controllers)

    def counts():
        return [controller.num_threads for controller in controllers]

    before = counts()
    with quiet_openblas('pardiso_mixed') as held:
        inside = counts()
        with quiet_openblas('pardiso') as nested:
            innermost = counts()
        after_nested = counts()
    if held != len(controllers) or nested != held:
        failed.append('quiet_openblas_pool_count')
    if any(count != 1 for count in inside + innermost + after_nested):
        failed.append('quiet_openblas_one_thread_inside')
    if counts() != before:
        failed.append('quiet_openblas_restores')
    details['openblas_pools'] = {'held': held, 'threads': before}


def _host_direct_spd_checks(back_end, spd, changed, rhs, failed, errors, details):
    """DiffusionDirectSolver (cpu) on a host direct back end: real SPD factor (PARDISO double: Cholesky, else LDL^T),
    a value-only change on the kept analysis (one analysis), inertia (n, 0), solves against spsolve, release."""
    import scipy.sparse.linalg
    from .sparse_direct import DiffusionDirectSolver, SparseDirectSolver
    runtime = SparseDirectSolver(back_end=back_end, backend='cpu')
    direct = DiffusionDirectSolver(runtime)
    for name, matrix in (('spd', spd), ('values', changed)):
        direct.factorize(matrix)
        solution = np.zeros(matrix.shape[0])
        direct.solve(rhs, solution)
        errors[f'{back_end}_{name}'] = norm_relative_error(solution,
                                                           scipy.sparse.linalg.spsolve(matrix.tocsc(), rhs))
    factor = direct._factor
    mixed = back_end.endswith('_mixed')
    expected_kind = 'spd' if back_end == 'pardiso' else 'symmetric'
    if factor.matrix_kind != expected_kind or factor.factor_precision != ('mixed' if mixed else 'double'):
        failed.append(f'{back_end}_spd_kind_{factor.matrix_kind}_{factor.factor_precision}')
    if not factor.last_plan_reused:
        failed.append(f'{back_end}_values_on_kept_analysis')
    counters = runtime.work_counters()
    if counters.get('host_direct_analyses') != 1 or counters.get('host_direct_reused_analyses') != 1:
        failed.append(f'{back_end}_one_analysis')
    inertia = factor.inertia()
    if inertia is not None and tuple(inertia) != (spd.shape[0], 0):
        failed.append(f'{back_end}_spd_inertia')
    details[f'{back_end}_spd'] = {'description': factor.description, 'inertia': inertia, 'counters': counters,
                                  'probe_residual': factor.probe_residual, 'single_residual': factor.single_residual}
    direct.release_factor()
    if factor.valid:
        failed.append(f'{back_end}_released')


def _host_direct_general_checks(back_end, operators, rhs, failed, errors, details):
    """sparse_direct factorize on the cpu (scipy operators, host_arrays): complex LU (solve, transposed solve, GMRES on
    the factor of a perturbed operator), the symmetrizer LDL^T of D A (solve and transposed solve), a real LU by
    factorize_host(real=True) (float64 solve, a complex rhs as two real solves), each against spsolve."""
    import scipy.sparse.linalg
    from .sparse_direct import SparseDirectSolver
    general, perturbed, unsymmetric, symmetrizer, real_general = operators
    nx, ny = _GRID
    runtime = SparseDirectSolver(back_end=back_end, backend='cpu')
    factor = runtime.factorize(general, nx, ny, False, False)
    if not factor.host_arrays or factor.symmetric or not factor.recyclable:
        failed.append(f'{back_end}_general_lu')
    errors[f'{back_end}_general'] = norm_relative_error(factor.solve(rhs),
                                                        scipy.sparse.linalg.spsolve(general.tocsc(), rhs))
    errors[f'{back_end}_general_transpose'] = norm_relative_error(
        factor.solve(rhs, transpose=True), scipy.sparse.linalg.spsolve(general.T.tocsc(), rhs))
    solution, iterations = runtime.solve_recycled(factor, perturbed, rhs)
    details[f'{back_end}_recycled_iterations'] = iterations
    if solution is None:
        failed.append(f'{back_end}_recycled_converged')
    else:
        errors[f'{back_end}_recycled'] = norm_relative_error(solution,
                                                             scipy.sparse.linalg.spsolve(perturbed.tocsc(), rhs))
    factor.release()
    factor = runtime.factorize(unsymmetric, nx, ny, False, False, symmetrizer=symmetrizer)
    if not factor.symmetric:
        failed.append(f'{back_end}_symmetrizer_ldlt')
    errors[f'{back_end}_symmetrizer'] = norm_relative_error(factor.solve(rhs),
                                                            scipy.sparse.linalg.spsolve(unsymmetric.tocsc(), rhs))
    errors[f'{back_end}_symmetrizer_transpose'] = norm_relative_error(
        factor.solve(rhs, transpose=True), scipy.sparse.linalg.spsolve(unsymmetric.T.tocsc(), rhs))
    factor.release()
    factor = runtime.factorize_host(real_general, real=True)
    real_rhs = np.ascontiguousarray(rhs.real)
    real_solution = factor.solve_host(real_rhs)
    if real_solution.dtype != np.float64 or factor.symmetric:
        failed.append(f'{back_end}_real_lu')
    errors[f'{back_end}_real_lu'] = norm_relative_error(real_solution,
                                                        scipy.sparse.linalg.spsolve(real_general.tocsc(), real_rhs))
    errors[f'{back_end}_real_lu_complex_rhs'] = norm_relative_error(
        factor.solve_host(rhs), scipy.sparse.linalg.spsolve(real_general.tocsc(), rhs))
    factor.release()
    details[f'{back_end}_counters'] = runtime.work_counters()


def _host_direct_pencil_checks(back_end, spd, rhs, failed, errors, details):
    """HostPencil + factorize_pencil on the cpu: a complex symmetric two-term pencil (+ a diagonal extra block) at two
    shifts against spsolve, symmetric once (LDL^T), the second factor on the kept analysis with its probe deferred to
    the solve (double back ends; mixed ones probe); an unsymmetric term gives an LU pencil."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .cudss_factor import csr_rows
    from .sparse_direct import HostPencil, SparseDirectSolver
    pattern = spd.tocsr(copy=True)
    pattern.sum_duplicates()
    pattern.sort_indices()
    shape, rows = pattern.shape, csr_rows(pattern.indptr)
    mass = np.where(pattern.indices == rows, 1.0 + rows % 3, 0.0)   # a diagonal mass term on the same pattern
    runtime = SparseDirectSolver(back_end=back_end, backend='cpu')
    pencil = runtime.pencil(pattern.indptr, pattern.indices, [pattern.data, mass], symmetric=True)
    if not isinstance(pencil, HostPencil) or not pencil.symmetric:
        failed.append(f'{back_end}_host_pencil_symmetric')
    extra_rows = np.arange(0, shape[0], 7)
    slots = pencil.slots(extra_rows, extra_rows)
    factor, outcomes = None, []
    for index, shift in enumerate((-0.5 + 0.05j, -0.7 + 0.05j)):
        added = (index + 0.3j) * np.ones(extra_rows.size)
        factor = runtime.factorize_pencil(pencil, [1.0, shift], previous=factor, extra=(slots, added))
        pending = factor._check_pending
        reference = (scipy.sparse.csr_matrix((pattern.data + shift * mass, pattern.indices, pattern.indptr), shape=shape)
                     + scipy.sparse.csr_matrix((added, (extra_rows, extra_rows)), shape=shape))
        errors[f'{back_end}_host_pencil_{index}'] = norm_relative_error(
            factor.solve_host(rhs), scipy.sparse.linalg.spsolve(reference.tocsc(), rhs))
        outcomes.append({'kind': factor.matrix_kind, 'reused': factor.last_plan_reused, 'deferred': pending,
                         'pending_after_solve': factor._check_pending, 'probe_residual': factor.probe_residual})
        if factor.matrix_kind != 'symmetric':
            failed.append(f'{back_end}_host_pencil_ldlt_{index}')
    last = outcomes[-1]
    if not last['reused']:
        failed.append(f'{back_end}_host_pencil_kept_analysis')
    if last['deferred'] != (not back_end.endswith('_mixed')) or last['pending_after_solve']:
        failed.append(f'{back_end}_host_pencil_deferred_check')
    factor.release()
    skewed = pattern.data * (1.0 + 0.1 * (pattern.indices > rows))   # unsymmetric values on the symmetric pattern
    unsymmetric = runtime.pencil(pattern.indptr, pattern.indices, [skewed], symmetric=True)
    if unsymmetric.symmetric:
        failed.append(f'{back_end}_host_pencil_asymmetry_found')
    factor = runtime.factorize_pencil(unsymmetric, [1.0])
    if factor.symmetric:
        failed.append(f'{back_end}_host_pencil_lu')
    errors[f'{back_end}_host_pencil_lu'] = norm_relative_error(
        factor.solve_host(rhs), scipy.sparse.linalg.spsolve(
            scipy.sparse.csr_matrix((skewed, pattern.indices, pattern.indptr), shape=shape).tocsc(), rhs))
    factor.release()
    details[f'{back_end}_host_pencil'] = outcomes


def _host_direct_inertia_checks(back_end, failed, details):
    """Real symmetric indefinite K - sigma I by factorize_host(real=True, symmetric=True): LDL^T, inertia() = the
    counting SuperLU's (None allowed: perturbed pivots, recorded); refactorize on the same pattern keeps the analysis."""
    import scipy.sparse
    from .sparse_direct import SparseDirectSolver, superlu_counting_inertia
    nx, ny = _GRID
    stiffness = _laplacian(nx, ny)
    exact = np.sort(np.linalg.eigvalsh(stiffness.toarray()))
    shifted = []
    for index in (_HOST_INERTIA_SHIFT_INDEX, _HOST_INERTIA_SHIFT_INDEX + 100):
        sigma = 0.5 * (exact[index] + exact[index + 1])
        shifted.append((stiffness - sigma * scipy.sparse.identity(stiffness.shape[0], format='csr')).tocsr())
    runtime = SparseDirectSolver(back_end=back_end, backend='cpu')
    factor, outcomes = None, []
    for operator in shifted:
        factor = runtime.factorize_host(operator, real=True, symmetric=True, previous=factor)
        got, expected = factor.inertia(), superlu_counting_inertia(operator)
        outcomes.append({'inertia': got, 'superlu': expected, 'perturbed_pivots': factor.perturbed_pivots,
                         'reused': factor.last_plan_reused})
        if not factor.symmetric:
            failed.append(f'{back_end}_indefinite_ldlt')
        if (got is not None and expected is not None and not back_end.endswith('_mixed')
                and tuple(got) != tuple(expected)):
            failed.append(f'{back_end}_inertia')
    if not factor.last_plan_reused:
        failed.append(f'{back_end}_inertia_kept_analysis')
    factor.release()
    details[f'{back_end}_inertia'] = outcomes


def _host_direct_fallback_checks(back_end, spd, rhs, failed, errors):
    """A *_mixed factor whose low-precision solves are made useless (identity) misses its probe: refactored in double
    (last_double_fallback, factor_precision double, mixed_rejection set), the solve right."""
    import scipy.sparse.linalg
    from .host_direct import HostDirectFactorization, host_direct_factorization
    unit_solve = HostDirectFactorization._unit_solve
    HostDirectFactorization._unit_solve = lambda self, vector, transpose=False: np.array(vector, copy=True)
    try:
        factor = host_direct_factorization(spd, back_end, real=True, positive_definite=True)
    finally:
        HostDirectFactorization._unit_solve = unit_solve
    if not (factor.last_double_fallback and factor.factor_precision == 'double' and factor.mixed_rejection):
        failed.append(f'{back_end}_double_fallback')
    errors[f'{back_end}_double_fallback'] = norm_relative_error(factor.solve_host(rhs),
                                                                scipy.sparse.linalg.spsolve(spd.tocsc(), rhs))
    factor.release()


def _host_direct_gpu_checks(back_end, general, spd, complex_rhs, real_rhs, failed, errors):
    """gpu backend: factorize of a cupyx operator (one host copy; cupy solve in / out, transposed too) and
    DiffusionDirectSolver on a cupyx SPD operator, against spsolve."""
    import scipy.sparse.linalg
    from ..core.backend import to_host
    from .sparse_direct import DiffusionDirectSolver, SparseDirectSolver
    nx, ny = _GRID
    runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
    factor = runtime.factorize(_device_matrix('gpu', general), nx, ny, False, False)
    device_rhs = _device('gpu', complex_rhs)
    solution = factor.solve(device_rhs)
    if isinstance(solution, np.ndarray) or factor.host_arrays:
        failed.append(f'gpu_{back_end}_solution_on_device')
    errors[f'gpu_{back_end}_general'] = norm_relative_error(to_host(solution),
                                                            scipy.sparse.linalg.spsolve(general.tocsc(), complex_rhs))
    errors[f'gpu_{back_end}_general_transpose'] = norm_relative_error(
        to_host(factor.solve(device_rhs, transpose=True)), scipy.sparse.linalg.spsolve(general.T.tocsc(), complex_rhs))
    factor.release()
    direct = DiffusionDirectSolver(SparseDirectSolver(back_end=back_end, backend='gpu'))
    direct.factorize(_device_matrix('gpu', spd))
    solution = _device('gpu', np.zeros(spd.shape[0]))
    direct.solve(_device('gpu', real_rhs), solution)
    errors[f'gpu_{back_end}_spd'] = norm_relative_error(to_host(solution),
                                                        scipy.sparse.linalg.spsolve(spd.tocsc(), real_rhs))
    direct.release_factor()


def _check_host_direct(results_by_name):
    """sparse_direct back ends mumps / mumps_mixed / pardiso / pardiso_mixed (each installed library, both precisions)
    against scipy on 30 x 20 grid operators, CPU and (with a GPU) cupyx operators; the role / misfit / resolution part
    always (passes as skipped without python-mumps / pydiso)."""
    import scipy.sparse
    from ..model.selftest_helpers import available_host_direct
    from .host_direct import HOST_DIRECT_BACK_ENDS, library_of
    failed, errors, details = [], {}, {}
    _host_direct_role_checks(failed)
    _quiet_openblas_checks(failed, details)
    libraries = tuple(available_host_direct())
    details['libraries'] = list(libraries)
    if not libraries:
        return _outcome({'role_checks': 0.0}, failed, dict(details, skipped='python-mumps / pydiso not installed'))
    rng = np.random.default_rng(_SEED + 7)
    nx, ny = _GRID
    n_unknowns = nx * ny
    spd = (_laplacian(nx, ny) + scipy.sparse.diags(rng.uniform(0.01, 1.0, n_unknowns))).tocsr()
    changed = (spd + scipy.sparse.diags(rng.uniform(0.0, 0.5, n_unknowns))).tocsr()   # same pattern, new values
    real_rhs = rng.standard_normal(n_unknowns)
    complex_rhs = rng.standard_normal(n_unknowns) + 1j * rng.standard_normal(n_unknowns)
    general = _general_operator(rng, nx, ny)
    perturbed = (general + scipy.sparse.diags(1e-3 * rng.uniform(0.0, 1.0, n_unknowns))).tocsr()
    symmetric = (_laplacian(nx, ny) + scipy.sparse.diags((0.5 + 0.2j) * rng.uniform(1.0, 2.0, n_unknowns))).tocsr()
    symmetrizer = (1.0 + 0.5j) * rng.uniform(0.5, 2.0, n_unknowns)
    unsymmetric = (scipy.sparse.diags(1.0 / symmetrizer) @ symmetric).tocsr()   # D A symmetric, A not
    forward = scipy.sparse.kron(scipy.sparse.diags([-np.ones(nx), np.ones(nx - 1)], [0, 1]),
                                scipy.sparse.identity(ny))
    real_general = (spd + 0.3 * forward).tocsr()
    operators = (general, perturbed, unsymmetric, symmetrizer, real_general)
    gpu = 'gpu' in _backends()
    for back_end in HOST_DIRECT_BACK_ENDS:
        if library_of(back_end) not in libraries:
            continue
        _host_direct_spd_checks(back_end, spd, changed, real_rhs, failed, errors, details)
        _host_direct_general_checks(back_end, operators, complex_rhs, failed, errors, details)
        _host_direct_pencil_checks(back_end, spd, complex_rhs, failed, errors, details)
        _host_direct_inertia_checks(back_end, failed, details)
        if back_end.endswith('_mixed'):
            _host_direct_fallback_checks(back_end, spd, real_rhs, failed, errors)
        if gpu:
            _host_direct_gpu_checks(back_end, general, spd, complex_rhs, real_rhs, failed, errors)
    details['backends'] = list(_backends())
    return _outcome(errors, failed, details)


_SCALE_ROWS = 40_000        # rows of the random CSR of solver_host_scale_numba
_SCALE_PER_ROW = 60         # stored entries per row (2.4e6: the nonlinear profiler's reduced tangent)
_SCALE_TOLERANCE = 1e-14    # same products, other association
_SCALE_CALLS = (3, 10)      # timed (numpy, numba) calls


def _check_host_scale_numba(_):
    """numba vs numpy host_direct.scaled_values: real, real with a symmetrizer, complex with a complex one."""
    from ..core.kernels import numba_available
    from ..model.selftest_helpers import NUMBA_MISSING_DETAILS, numba_twin_result
    from .host_direct import scaled_values
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _SCALE_TOLERANCE,
                'details': dict(NUMBA_MISSING_DETAILS)}
    import numba
    rng = np.random.default_rng(_SEED)
    indptr = (np.arange(_SCALE_ROWS + 1) * _SCALE_PER_ROW).astype(np.int32)
    indices = np.sort(rng.integers(0, _SCALE_ROWS, (_SCALE_ROWS, _SCALE_PER_ROW)), axis=1).astype(np.int32).ravel()
    row_scale = rng.random(_SCALE_ROWS) + 0.5
    real_values = rng.standard_normal(indices.size)
    complex_values = real_values + 1j * rng.standard_normal(indices.size)

    def twins(symmetrizer, dtype):   # the scaled values (complex as float64 pairs) are the compared output
        def run(numba_flag):
            def call(values, out):
                out[...] = scaled_values(values, indptr, indices, row_scale, symmetrizer, dtype,
                                         numba=numba_flag).view(np.float64)
            return call
        return run(False), run(True)
    variants = {'real': (None, np.float64, real_values),
                'real symmetrizer': (rng.random(_SCALE_ROWS) + 0.5, np.float64, real_values),
                'complex symmetrizer': (rng.standard_normal(_SCALE_ROWS) + 1j * rng.standard_normal(_SCALE_ROWS),
                                        np.complex128, complex_values)}
    comparisons, timings = {}, {}
    for label, (symmetrizer, dtype, values) in variants.items():
        out = np.zeros(values.size * (2 if np.dtype(dtype).kind == 'c' else 1))
        comparisons[label] = (*twins(symmetrizer, dtype), (values, out))
    timings['real'] = (*twins(None, np.float64), (real_values, np.zeros(real_values.size)), *_SCALE_CALLS)
    result = numba_twin_result(comparisons, timings, _SCALE_TOLERANCE, numba)
    result['details']['stored_entries'] = int(indices.size)
    return result


def _check_arpack_host_direct(results_by_name):
    """arpack_shift_invert factorizations mumps / pardiso (+ _mixed): count_below and eigsh_shifted(inertia=True) of
    the Laplacian with the consistent mass against the dense eigh counts (mixed recorded only), eigsh_shifted
    eigenvalues against eigh, one analysis for every shift (double), eigs_shifted of a complex symmetric pencil against
    superlu, a Hermitian count_below by the counting SuperLU; passes as skipped without python-mumps / pydiso."""
    import scipy.linalg
    import scipy.sparse
    from ..model.selftest_helpers import available_host_direct
    from .arpack_shift_invert import EIGSH_SHIFT_RELATIVE_OFFSET, ArpackShiftInvertSolver
    from .host_direct import HOST_DIRECT_FACTORIZATIONS, library_of
    libraries = tuple(available_host_direct())
    if not libraries:
        return _outcome({'none': 0.0}, [], {'skipped': 'python-mumps / pydiso not installed'})
    nx, ny = _GRID
    stiffness, mass = _laplacian(nx, ny), _consistent_mass(nx, ny)
    exact = np.sort(scipy.linalg.eigh(stiffness.toarray(), mass.toarray(), eigvals_only=True))
    sigmas = [0.5 * exact[0] if index < 0 else 0.5 * (exact[index] + exact[index + 1]) for index in _INERTIA_INDICES]
    complex_stiffness = ((1.0 + 0.01j) * stiffness).tocsr()
    reference_solver = ArpackShiftInvertSolver(factorization='superlu')
    start = np.random.default_rng(_SEED + 8).standard_normal(nx * ny)
    reference = reference_solver.eigs_shifted(complex_stiffness, mass, sigmas[1], _INERTIA_EIGENPAIRS,
                                              start_vector=start)[0]
    errors, failed, outcomes = {}, [], {}
    for factorization in HOST_DIRECT_FACTORIZATIONS:
        if library_of(factorization) not in libraries:
            continue
        mixed = factorization.endswith('_mixed')
        solver = ArpackShiftInvertSolver(factorization=factorization)
        factor, worst = None, 0.0
        for index, sigma in zip(_INERTIA_INDICES, sigmas):
            expected = int(np.count_nonzero(exact < sigma))
            counted, factor = solver.count_below(stiffness, mass, sigma, previous=factor)
            result = solver.eigsh_shifted(stiffness, mass, sigma / (1.0 + EIGSH_SHIFT_RELATIVE_OFFSET),
                                          _INERTIA_EIGENPAIRS, previous=factor, inertia=True)
            factor, shifted = result[4], result[5]
            got = [None if item is None else item.n_below for item in (counted, shifted)]
            key = f'{factorization}_{index}'
            outcomes[key] = {'expected': expected, 'count_below': got[0], 'eigsh_shifted': got[1],
                             'exact': None if shifted is None else shifted.exact}
            if not mixed and got != [expected, expected]:
                failed.append(f'{key}_count')
            if not result[3]:
                failed.append(f'{key}_converged')
            # the nearest but one: the last returned pair may swap with its neighbour outside the set
            nearest = exact[np.argsort(np.abs(exact - sigma))][:_INERTIA_EIGENPAIRS - 1]
            worst = max(worst, _matched_error(result[0], nearest))
        errors[f'{factorization}_eigsh'] = worst
        outcomes[f'{factorization}_analyses'] = factor.n_analyses
        if not mixed and factor.n_analyses != 1:
            failed.append(f'{factorization}_one_analysis')
        factor.release()
        values, _vectors, _applications, converged, factor = solver.eigs_shifted(
            complex_stiffness, mass, sigmas[1], _INERTIA_EIGENPAIRS, start_vector=start)
        errors[f'{factorization}_eigs_shifted'] = _matched_error(values, reference)
        if not converged:
            failed.append(f'{factorization}_eigs_shifted_converged')
        factor.release()
        hermitian, _previous = solver.count_below(stiffness.astype(np.complex128), mass, sigmas[2])   # Hermitian
        if hermitian is None or hermitian.n_below != int(np.count_nonzero(exact < sigmas[2])):
            failed.append(f'{factorization}_hermitian_count')
        outcomes[f'{factorization}_counters'] = solver.work_counters()
    return _outcome(errors, failed, {'libraries': list(libraries), 'outcomes': outcomes})


SOLVER_SELFTEST_CASES = (
    SelfTestCase(
        name='solver_pcg_cpu_gpu',
        label='Iterative solver: processor and graphics card agree',
        description='pcg on the CPU backend (numpy / scipy CSR, the numpy twins of the fused CG kernels) and the GPU '
                    'backend (cupy) on a 30 x 20 5-point Laplacian + a random diagonal in [0.01, 1] (SPD, condition '
                    '~400), tolerance 1e-10: double and mixed with Jacobi on the CSR, double and mixed matrix-free '
                    'with a callback preconditioner (mixed: float32 apply / precondition callbacks on float32 copies, '
                    'the path of solid_mechanics_3d). Each solution against scipy spsolve (relative error <= '
                    f'{_PARITY_TOLERANCE:g}: condition x tolerance); every solve converged, CPU and GPU double '
                    'iteration counts within one residual-check block, a CPU solver refuses a cupyx operator.',
        specs=lambda: {}, check=_check_pcg, max_wall_time=120.0, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='solver_sparse_direct_cpu_gpu',
        label='Sparse direct solver: processor and graphics card agree',
        description='sparse_direct on a complex non-symmetric 30 x 20 grid operator (Laplacian + (0.5 + 0.2 j) x [1, 2] '
                    'diagonal + 0.3 x a one-sided x difference): CPU backend (scipy operator, auto -> scipy_splu, '
                    'numpy solutions), GPU scipy_splu and GPU cusolver_qr; solve, transposed solve and (SuperLU) '
                    'GMRES on the factor of a 1e-3-perturbed operator, each against scipy spsolve (relative error <= '
                    f'{_PARITY_TOLERANCE:g}; the recycled solve stops at a 1e-10 residual); the CPU backend refuses '
                    'cusolver_qr.',
        specs=lambda: {}, check=_check_sparse_direct, max_wall_time=120.0, backends=('cpu', 'gpu'), heavy=True),
    SelfTestCase(
        name='solver_line_block_cpu_gpu',
        label='Line-block solver: processor and graphics card agree',
        description='line_block mixed and double on the CPU (numpy) and GPU (cupy) backends: K = 24 x 10 Laplacian, '
                    'shifts K + j omega M1 + q^2 M2 (M1 = 1, M2 random in [0.5, 2]) for omega '
                    f"{', '.join(f'{omega:g}' for omega in _LINE_BLOCK_OMEGAS)} (the extremes: near the real "
                    'Laplacian and diagonal-dominant) and q^2 0, 0.5, two complex right-hand sides; every shift '
                    'accepted and each solution against the per-shift scipy spsolve (relative error <= '
                    f'{_PARITY_TOLERANCE:g}; refinement to a 1e-9 residual). '
                    'factorize_operators (Part II stage 3): a batch of three complex non-symmetric grid operators '
                    '(the sparse_direct case operator with fresh random diagonals), each shift solved against its '
                    'own scipy spsolve with the same tolerance, accepted, and invalid after the next set_operator. '
                    'Not yet run with two omega values.',
        specs=lambda: {}, check=_check_line_block, max_wall_time=120.0, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='solver_eigsh_shift_on_eigenvalue',
        label='Symmetric eigen solver: target exactly on an eigenvalue',
        description='CPU: arpack_shift_invert eigsh_shifted (the solid_mechanics eigenfrequency path, SuperLU, machine '
                    'precision) of the 30 x 20 Dirichlet Laplacian with M = 2 I, the shift exactly on the exact '
                    f'eigenvalue lambda_{_ON_EIGENVALUE_MODE[0]}{_ON_EIGENVALUE_MODE[1]} = (d_2(30) + d_1(20)) / 2 '
                    '(d_p(n) = 2 - 2 cos(p pi / (n + 1))), so K - shift M is singular to rounding; eigsh_shifted '
                    'factors and runs eigsh at sigma = shift (1 + 1e-4). Expected: converged, the '
                    f'{_ON_EIGENVALUE_CHECKED} exact eigenvalues nearest the shift each matched by a returned one to '
                    f'{_ON_EIGENVALUE_TOLERANCE:g} relative (error 1 when not converged). Written, not run.',
        specs=lambda: {}, check=_check_eigsh_on_eigenvalue, max_wall_time=60.0),
    SelfTestCase(
        name='solver_arpack_cudss',
        label='Eigen solver on a GPU factorisation (cuDSS) agrees with SuperLU',
        description='arpack_shift_invert factorization cudss (nvmath-python DirectSolver) against superlu: the '
                    f'sparse_direct case operator shifted by {_CUDSS_SHIFT} (complex non-symmetric, 30 x 20), a cuDSS '
                    'solve against scipy spsolve, eigs shift-invert of its '
                    f'{_CUDSS_EIGENPAIRS} eigenvalues nearest the shift and eigsh_shifted of the 30 x 20 Laplacian '
                    'with a random mass diagonal in [0.5, 2] near 1 (real float64 cuDSS LDL^T), each set matched to '
                    f'the SuperLU one (relative error <= {_PARITY_TOLERANCE:g}); a complex symmetric operator (the '
                    'Laplacian + a complex diagonal - the shift) factorised as LDL^T and solved against spsolve, the '
                    'non-symmetric one asked for LDL^T falling back to LU; precision mixed (complex64 factor of the '
                    'equilibrated operator, float64 refinement to 1e-12): the complex symmetric solve against spsolve '
                    'and factorization cudss_mixed eigs (complex64 LU) matched to SuperLU; the device Krylov-Schur '
                    '(eigs_device on the cudss LU) and its widening resumed to twice the eigenpairs (copied, and in '
                    'place on rows reserved by the first pass) matched to SuperLU, as eigs_device around the host SuperLU solve and eigs on device_factor_matvec of the '
                    'cudss LU, a device_arnoldi runtime below the host-matvec size rule on host ARPACK; cudss_mixed below mixed_min_unknowns resolved to cudss; every solve converged, the probe '
                    'residual within the cuDSS accept tolerance, the device bytes counted, no solve after release, '
                    'cudss and cudss_mixed misfits on the cpu backend only. Passes as skipped without nvmath-python.',
        specs=lambda: {}, check=_check_arpack_cudss, max_wall_time=120.0, backends=('gpu',)),
    SelfTestCase(
        name='solver_sparse_direct_cudss',
        label='Sparse direct solver on the graphics card (cuDSS) agrees with SciPy',
        description='sparse_direct back ends cudss and cudss_mixed (nvmath-python DirectSolver) against scipy spsolve '
                    f'(relative error <= {_PARITY_TOLERANCE:g}) on 30 x 20 grid operators: DiffusionDirectSolver (the '
                    'diffusion / elastic linear path) on the 5-point Laplacian + a random diagonal (real SPD: '
                    'Cholesky in double, a float32 LDL^T in mixed, never a float32 Cholesky), then a value-only change '
                    '(one cuDSS plan for both factorisations) and one extra symmetric entry (a second plan); a mixed '
                    'SPD factor whose deferred check misses (stall test after 4 GMRES iterations) re-plans as the '
                    'double Cholesky; with useless float32 solves the mixed probe stalls, both plans of the call are '
                    'counted, and inside cudss_orderings_kept a second factor of that pattern plans double at once '
                    '(remembered rejection, no fallback counted), outside it mixed again; factorize_host with '
                    'previous= on the planned pattern refreshes on the device (plan reused, asymmetry checked there) '
                    'and an unsymmetric operator of that pattern asked for symmetry is re-planned there as LU; a '
                    'factor planned from device values (Cholesky) and a vouched symmetric DevicePencil (LDL^T) solve '
                    'right; '
                    f'a mixed Cholesky-request factor solved at the default target and at backward error '
                    f'{_LOOSE_BACKWARD_ERROR:g} (per call, then on the factor): no more GMRES iterations, target met; '
                    'a complex non-symmetric '
                    'operator by factorize (LU; transposed solve refused); D^-1 S with S complex symmetric and the '
                    'symmetrizer D passed (LDL^T of D A, solve and transposed solve); a real non-symmetric operator by '
                    'factorize_host(real=True) (float64 LU, solve_host). CPU part: the back ends fit heat linear '
                    '(setting_choices lists them) and misfit '
                    'drift_diffusion linear (opt-in), misfit the cpu backend only, no host size rule; back_end auto of '
                    'the pressure_acoustics linear role resolves to cudss_mixed on gpu triangles / tetrahedra with '
                    'nvmath-python only (SolverRole.auto_back_ends), marked as resolved from auto; with every cuDSS '
                    'probe made to fail an auto runtime falls back to host SuperLU for good (factorize_host, '
                    'factorize_pencil, warned, counted) while an explicit cudss one raises; the cuDSS time recipe '
                    'prices a kept-ordering plan '
                    'at CUDSS_KEPT_ORDERING_PLAN_FRACTION, a deferred check per refreshed double device-values factor '
                    'and no 3D value upload from device values. Passes as skipped without nvmath-python (the CPU '
                    'part still judged). Written, not run.',
        specs=lambda: {}, check=_check_sparse_direct_cudss, max_wall_time=120.0, backends=('gpu',)),
    SelfTestCase(
        name='solver_cudss_device_pencil',
        label='Sparse direct solver: frequency sweeps from pencil values on the graphics card (cuDSS)',
        description='sparse_direct DevicePencil + factorize_pencil (term values kept on the device, each frequency '
                    'combined, equilibrated and refactorised there) on back ends cudss and cudss_mixed against scipy '
                    f'spsolve (relative error <= {_PARITY_TOLERANCE:g}): K = 30 x 20 Laplacian (1 + '
                    f'{_PENCIL_LOSS:g} j), C a random diagonal, M the bilinear consistent mass (9-point union pattern) '
                    f"at omega {', '.join(f'{omega:g}' for omega in _PENCIL_OMEGAS)} rad/s and once with a caller "
                    'row_scale (diag K + omega^2 diag M): found symmetric, LDL^T, one cuDSS plan for all four; '
                    'refactorisations on the reused plan (both back ends) leave their accuracy check to the first '
                    'solve (pending after factorise, judged by it); cudss: the pencil pattern matched by identity, a '
                    'check that misses once refined and passed (refinement kept by the next factor), one that misses '
                    'again raising CudssAccuracyError with the factor released, the symmetric pencil stored as its '
                    'lower triangle (fewer device bytes, full_term mirrors it back exactly) with the operator norm '
                    'left lazy after a refresh and equal to the host route\'s, and a small symmetric block added on '
                    'the device at its lower-triangle slots (factorize_pencil extra=, E7-8); cudss_mixed: a '
                    'low-precision solve '
                    'replaced by the identity makes that check miss: the double factor re-planned in place, the solve '
                    'right, one more plan, its reused ordering and one double fallback counted; a real '
                    'pencil K - 0.01 M (float64 LDL^T, real solution); K + 0.3 x a one-sided x difference found '
                    'unsymmetric and factored by the general LU; physics/mechanics frequency_response.HarmonicSystem '
                    'on a cuDSS runtime (K + j 0.02 K - omega^2 M, two held dofs) against its private SuperLU, the '
                    'device pencil used with one plan. Ordering reuse: the in-place double re-plan of the fallback '
                    'takes its own kept ordering; inside cudss_orderings_kept a second factor object of a 160 x 160 '
                    'complex symmetric Laplacian plans on the first one\'s permutation + partition tree (same lu_nnz, '
                    'solution right, cudss_reused_orderings 1 of 2 plans), off before and after the block, where a '
                    'fresh factor plans from scratch; in a second block a real Cholesky of the real Laplacian (same '
                    'pattern) hands its ordering to the complex LDL^T (same fill as fresh, solution right) while a '
                    'real LU between them plans fresh (its own ordering family). Passes as skipped without a GPU or '
                    'nvmath-python.',
        specs=lambda: {}, check=_check_cudss_device_pencil, max_wall_time=120.0, backends=('gpu',)),
    SelfTestCase(
        name='solver_default_precision',
        label='Solver precision default: which solvers follow it',
        description='solver_defaults.precision (S9) resolution on the CPU, no run: schema.resolved_solvers and '
                    'default_precision_ignored on one-physics models (heat linear: auto / single / mixed Automatic, a '
                    'pcg node without, with another and with the same precision, the cpu backend Automatic '
                    'sparse_direct (double only); solid_mechanics and solid_mechanics_3d linear single and mixed (their '
                    'roles accept mixed), their Automatic (gpu: mixed, cpu: double; '
                    'solver_defaults.precision double wins over the gpu Automatic mixed); '
                    'em_fdtd_ez stepper double and half_storage (its role refuses half_storage)): resolved solver type, '
                    'precision and whether the default is reported ignored; node settings are never written; '
                    'with_default_precision directly (fills without a role, a role refusing mixed (a pcg role of the '
                    'default precisions) and an explicit precision leave the settings). Also the Automatic linear solver multigrid resolution of the diffusion types '
                    '(mesh plan M0, rows multigrid_*): heat, '
                    'electric_currents, electrostatics and magnetostatics Automatic on the gpu backend resolve pcg '
                    'double with preconditioner multigrid (single with solver_defaults.precision single); '
                    'solver_defaults.precision mixed, a pcg node with precision mixed and a node filled with mixed '
                    'resolve jacobi; a pcg node without a preconditioner keeps jacobi, one setting multigrid single '
                    'keeps it; the cpu backend Automatic stays sparse_direct, solid_mechanics Automatic stays jacobi '
                    '(pcg mixed on the gpu backend). '
                    'Each resolution passes the solver type validate without errors, linear_solver_multigrid (the '
                    'memory estimators) agrees and node settings are never written. error = number of mismatches, '
                    'tolerance 0.',
        specs=lambda: {}, check=_check_default_precision, max_wall_time=60.0),
    SelfTestCase(
        name='eigen_inertia_count',
        label='Eigen solver: counting eigenvalues below a shift',
        description='arpack_shift_invert count_below and eigsh_shifted(inertia=True) (Sylvester inertia of K - sigma M) '
                    'of the 30 x 20 Laplacian with the bilinear consistent mass, sigma halfway between neighbouring '
                    'eigenvalues (below the lowest, after the 38th, 151st and 401st): the number of eigenvalues below '
                    'sigma equals the dense scipy eigh count for SuperLU (pivots on the diagonal, else a counting '
                    'SuperLU) and, with a usable GPU and nvmath-python, the cuDSS double LDL^T (cudss_mixed recorded '
                    'only); the cuDSS factor is refactorised across all eight shifts with one plan (the shifted '
                    'operator keeps the mass pattern at sigma = 0 too); the counting SuperLU directly. error = number '
                    'of mismatches, tolerance 0. Written, not run.',
        specs=lambda: {}, check=_check_eigen_inertia, max_wall_time=120.0, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='eigen_band_merge',
        label='Eigen solver: every mode of a frequency band, degenerate pairs kept',
        description='CPU: physics.mechanics.modal solve_band_modes of the 12 x 12 Dirichlet Laplacian with M = 2 I '
                    '(lambda_pq = lambda_qp: exactly degenerate pairs) between the 6th and 34th eigenvalues, 4 modes '
                    'per shift (SuperLU): every dense eigh eigenvalue of the band found once (degenerate pairs twice, '
                    f'relative error <= {_BAND_TOLERANCE:g}), info expected_count = the dense count, the returned '
                    f'modes M-orthogonal (|cos| <= {_BAND_ORTHOGONALITY:g}); solve_modes_at_shifts at two targets '
                    '(just above the 11th and 13th eigenvalues) gives the distinct eigenvalues of the two solve_modes '
                    'runs, with no fewer modes than their clustered union and no more than both runs together. '
                    'Written, not run.',
        specs=lambda: {}, check=_check_eigen_band, max_wall_time=120.0),
    SelfTestCase(
        name='solver_host_direct_cpu_gpu',
        label='Sparse direct solver on the processor (MUMPS / PARDISO) agrees with SciPy',
        description='sparse_direct back ends mumps / mumps_mixed / pardiso / pardiso_mixed (python-mumps, pydiso; each '
                    'installed library, both precisions) against scipy spsolve (relative error <= '
                    f'{_PARITY_TOLERANCE:g}) on 30 x 20 grid operators: DiffusionDirectSolver on the 5-point Laplacian '
                    '+ a random diagonal (real SPD: PARDISO double Cholesky, MUMPS and mixed LDL^T), then a value-only '
                    'change on the kept analysis (one analysis), inertia (n, 0), release; the complex non-symmetric '
                    'operator of solver_sparse_direct_cpu_gpu by factorize on the cpu (LU, solve, transposed solve, '
                    'GMRES on the factor of a 1e-3-perturbed operator); D^-1 S with S complex symmetric and the '
                    'symmetrizer D (LDL^T of D A, solve and transposed solve); a real non-symmetric operator by '
                    'factorize_host(real=True) (float64 LU, a complex rhs as two real solves); a HostPencil of the '
                    'SPD operator and a diagonal mass at two complex shifts with a diagonal extra block by '
                    'factorize_pencil (LDL^T, kept analysis, double: the probe deferred to the solve) and an '
                    'unsymmetric-valued pencil (LU); the real symmetric '
                    'indefinite Laplacian - sigma I at two shifts (LDL^T, inertia equal to the counting SuperLU unless '
                    'pivots were perturbed, the second shift on the kept analysis); mixed: low-precision solves made '
                    'useless force the probe miss and the double fallback (solve right); with a GPU the same LU '
                    '(cupy solves, transposed too) and DiffusionDirectSolver on cupyx operators. Always (no wrapper '
                    'needed): the back ends fit heat, em_fdfd_ez and em_fdfd_nonlinear linear (all four), '
                    'magnetic_fields_ac linear (double only), em_fdfd_ez eigen (all four), membrane eigen (double '
                    'only), misfit em_modes shifted, run on the cpu, keep the SuperLU size rule; resolve_back_end '
                    'raises missing_text without the wrapper; a host mixed eigen request below mixed_min_unknowns '
                    'factors in double; quiet_openblas holds the OpenBLAS pools at one thread for PARDISO names only '
                    '(nested, restored after; nothing without threadpoolctl). Passes as skipped without python-mumps / '
                    'pydiso (the role part still judged). Run with pydiso; the MUMPS part written, not run.',
        specs=lambda: {}, check=_check_host_direct, max_wall_time=120.0, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='solver_arpack_host_direct',
        label='Eigen solver on a processor factorisation (MUMPS / PARDISO) agrees with dense counts',
        description='arpack_shift_invert factorizations mumps / mumps_mixed / pardiso / pardiso_mixed (each installed '
                    'library): count_below and eigsh_shifted(inertia=True) of the 30 x 20 Laplacian with the bilinear '
                    'consistent mass at the sigmas of eigen_inertia_count against the dense eigh count (mixed recorded '
                    'only), the eigsh_shifted eigenvalues nearest each sigma against eigh (relative error <= '
                    f'{_PARITY_TOLERANCE:g}), one analysis for all eight shifts (double), eigs_shifted of the complex '
                    'symmetric pencil (1 + 0.01 j) K, M against superlu, a Hermitian count_below by the counting '
                    'SuperLU. Passes as skipped without python-mumps / pydiso. Written, not run.',
        specs=lambda: {}, check=_check_arpack_host_direct, max_wall_time=120.0),
    SelfTestCase(
        name='solver_host_scale_numba',
        label='Host direct equilibration scaling: numba row loop vs numpy (CPU)',
        description='host_direct.scaled_values (values x symmetrizer[row] x r[row] x r[column], every MUMPS / PARDISO '
                    'refactorisation) with numba=True against numba=False on a random CSR of '
                    f'{_SCALE_ROWS} rows and {_SCALE_PER_ROW} entries per row: real, real with a symmetrizer, complex '
                    'values with a complex symmetrizer; measured = error = the worst max |numba - numpy| relative to '
                    f'the peak, tolerance {_SCALE_TOLERANCE:g}. Details: ms per call of both. Without numba nothing '
                    'is compared. Not yet run.',
        specs=lambda: {}, check=_check_host_scale_numba, max_wall_time=120.0, backends=('cpu',)),
)

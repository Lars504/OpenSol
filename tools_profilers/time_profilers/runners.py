"""One timed repeat of a profiler Problem per SolverType, in the child process, through the calls open_sol's own
consumers make (SparseDirectSolver.factorize like fdfd_2d, DiffusionDirectSolver like DiffusionSystem, PCGSolver,
BandedSolver, LineBlockSolver.factorize_operators, eigenfrequency.solve_resonances) on the cpu or gpu backend: phase
wall times, the relative residual against its tolerance and solver details. open_sol is imported inside the runners."""
# Phases (seconds, None where the solver does not expose them): analysis = MUMPS / PARDISO ordering + symbolic
# analysis (inside factor), factor = the first (cold) factorisation / preconditioner set-up incl. analysis and the
# accuracy probe, solve = one solve (pcg: the CG iterations), refactor = a value-only refactorisation of the same
# pattern (host direct: kept analysis, SuperLU: kept ordering, cuDSS: kept plan, multigrid: kept aggregates; the
# scaled copy is made before its timer), total = factor + solve. Eigen: factor = the shifted factorisation, solve =
# the ARPACK iterations, total = the whole solve_resonances.
# Backends: the operator / rhs move to the device once per case (backend_arrays, before the memory baseline), as a
# gpu model assembles them there; the consumers take the backend's arrays (gpu: cupyx CSR, cupy vectors), except
# line_block and the eigen solve, which take host operators on both backends (as fdfd_2d hands them). Every clock
# reading synchronises the device first, so queued kernels count in the phase that launched them.
# Tolerance of a repeat: pcg its (floored) stop tolerance, line_block its accept tolerance, direct factors and eigen
# pairs cudss_factor.ACCEPT_RESIDUAL (the probe bound of open_sol's own factors; eigen: at least the ARPACK
# tolerance). A residual above ACCURACY_MARGIN x tolerance makes an 'ok' repeat 'inaccurate'.
import math
import time
from dataclasses import dataclass

import numpy as np

REFACTOR_SCALE = 1.0 + 1e-3   # the refactor phase factorises A x this: a value-only change, the same pattern
ACCURACY_MARGIN = 10.0        # residual / tolerance above which a repeat is 'inaccurate'


@dataclass(frozen=True, eq=False)
class BackendArrays:
    """The problem's operator and rhs on the run backend ('cpu': the host scipy / numpy objects themselves, 'gpu':
    cupyx CSR / cupy copies) and that backend's array module."""
    backend: str
    operator: object
    rhs: object
    xp: object


def backend_arrays(problem, backend) -> BackendArrays:
    """BackendArrays of a problem; on 'gpu' the device copies are made here (and the GPU budget set as api.run does)."""
    if backend == 'cpu':
        return BackendArrays('cpu', problem.operator, problem.rhs, np)
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    from open_sol.core import gpu_memory
    gpu_memory.set_budget()
    if problem.kind == 'eigen':   # solve_resonances takes the host operator on both backends
        return BackendArrays('gpu', problem.operator, None, cp)
    arrays = BackendArrays('gpu', cpsp.csr_matrix(problem.operator), cp.asarray(problem.rhs), cp)
    cp.cuda.Device().synchronize()
    return arrays


def _clock(backend) -> float:
    """perf_counter after the device finished its queued work (gpu)."""
    if backend == 'gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()
    return time.perf_counter()


def _host(array):
    """numpy view / copy of a numpy or cupy array."""
    return array if isinstance(array, np.ndarray) else array.get()


def _label():
    return 'profiler solver'


def relative_residual(operator, rhs, solution) -> float:
    """||rhs - A x|| / ||rhs|| in float64 / complex128 on the host (x cast up first)."""
    rhs = np.asarray(rhs)
    solution = _host(solution).astype(np.result_type(rhs.dtype, operator.dtype, np.float64), copy=False)
    norm = float(np.linalg.norm(rhs))
    residual = float(np.linalg.norm(rhs - operator @ solution))
    return residual / norm if norm > 0.0 else residual


def _direct_tolerance() -> float:
    from open_sol.solvers.cudss_factor import ACCEPT_RESIDUAL
    return float(ACCEPT_RESIDUAL)


def _scaled(operator):
    """A copy of the CSR operator (scipy or cupyx) with every value times REFACTOR_SCALE (same indptr / indices)."""
    scaled = operator.copy()
    scaled.data *= REFACTOR_SCALE
    return scaled


def _release(factor) -> None:
    release = getattr(factor, 'release', None)   # HostDirectFactorization / CudssFactorization / LineBlockFactor
    if release is not None:
        release()


def _number(value):
    """float / int / None of a numpy or Python number (JSON)."""
    if value is None:
        return None
    return value.item() if hasattr(value, 'item') else value


def _factor_details(factor) -> dict:
    """Details of a sparse_direct Factorization / HostDirectFactorization / CudssFactorization after its first solve."""
    return {'factor_kind': getattr(factor, 'description', None) or getattr(factor, 'back_end', None) or 'SuperLU',
            'factor_precision': getattr(factor, 'factor_precision', 'double'),
            'analysis_seconds': _number(getattr(factor, 'analysis_seconds', None)),
            'factor_bytes': _number(getattr(factor, 'host_bytes', None)),
            'device_bytes': _number(getattr(factor, 'device_bytes', None)),
            'library_peak_bytes': _number(getattr(factor, 'peak_bytes', None)),
            'factor_entries': _number(getattr(factor, 'lu_entries', None)),
            'library_threads': _number(getattr(factor, 'threads', None)),
            'openblas_pools_held': _number(getattr(factor, 'openblas_pools', None)),
            'probe_refinements': _number(getattr(factor, 'refinement_steps', None)),
            'solve_refinements': _number(getattr(factor, 'last_refinements', None)),
            'perturbed_pivots': _number(getattr(factor, 'perturbed_pivots', None))}


def _repeat(phases, residual, details, tolerance, status='ok') -> dict:
    """Repeat record; an 'ok' repeat whose residual is non-finite or above ACCURACY_MARGIN x tolerance becomes
    'inaccurate'."""
    phases = {name: _number(seconds) for name, seconds in phases.items()}
    if status == 'ok' and residual is not None and tolerance is not None and (
            not math.isfinite(residual) or residual > ACCURACY_MARGIN * tolerance):
        status = 'inaccurate'
    return {'status': status, 'phases': phases, 'residual': residual, 'tolerance': tolerance, 'details': details}


def run_sparse_direct(problem, values, arrays) -> dict:
    """sparse_direct: a real SPD problem through DiffusionDirectSolver (the diffusion types' consumer), the FDFD drive
    through SparseDirectSolver.factorize (the LDL^T of the symmetrised D A on cuDSS / MUMPS / PARDISO, as fdfd_2d)."""
    if problem.kind == 'spd':
        return _run_diffusion_direct(problem, values, arrays)
    return _run_grid_direct(problem, values, arrays)


def _run_grid_direct(problem, values, arrays) -> dict:
    from open_sol.solvers.host_direct import HOST_DIRECT_BACK_ENDS
    from open_sol.solvers.sparse_direct import CUDSS_BACK_ENDS, SparseDirectSolver
    back_end, backend = values['back_end'], arrays.backend
    nx, ny = problem.grid_shape
    symmetric_back_ends = HOST_DIRECT_BACK_ENDS + CUDSS_BACK_ENDS
    options = {'symmetric': True, 'symmetrizer': problem.symmetrizer} if back_end in symmetric_back_ends else {}
    runtime = SparseDirectSolver(back_end=back_end, label=_label, backend=backend)   # fresh: no kept ordering
    started = _clock(backend)
    factor = runtime.factorize(arrays.operator, nx, ny, False, False, **options)
    factored = _clock(backend)
    solution = factor.solve(arrays.rhs)
    solved = _clock(backend)
    details = _factor_details(factor)
    analysis = details.pop('analysis_seconds')
    scaled = _scaled(arrays.operator)
    refactor_started = _clock(backend)
    factor = runtime.factorize(scaled, nx, ny, False, False, previous=factor, **options)
    refactored = _clock(backend)
    details['refactor_reused_analysis'] = bool(getattr(factor, 'last_plan_reused', False))
    details['reused_orderings'] = runtime.orderings.n_reused
    _release(factor)
    phases = {'analysis': analysis, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details, _direct_tolerance())


def _run_diffusion_direct(problem, values, arrays) -> dict:
    from open_sol.solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
    backend = arrays.backend
    runtime = SparseDirectSolver(back_end=values['back_end'], label=_label, backend=backend)
    direct = DiffusionDirectSolver(runtime)
    direct.grid_shape = problem.grid_shape   # a 3D window: SuperLU starts from grid nested dissection
    solution = arrays.xp.zeros(problem.n_unknowns)
    started = _clock(backend)
    direct.factorize(arrays.operator)
    factored = _clock(backend)
    direct.solve(arrays.rhs, solution)
    solved = _clock(backend)
    details = _factor_details(direct._factor)   # the solver keeps its factor private; read-only here
    analysis = details.pop('analysis_seconds')
    scaled = _scaled(arrays.operator)
    refactor_started = _clock(backend)
    direct.factorize(scaled)   # host direct / cuDSS: previous factor refactorised; SuperLU: kept order
    refactored = _clock(backend)
    details['refactor_reused_analysis'] = bool(getattr(direct._factor, 'last_plan_reused', False))
    details['reused_orderings'] = runtime.orderings.n_reused
    direct.release_factor()
    phases = {'analysis': analysis, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details, _direct_tolerance())


def run_pcg(problem, values, arrays) -> dict:
    """pcg: factorize (Jacobi / multigrid set-up) on the operator in the solver dtype, one cold CG solve; single
    precision runs at a tolerance of at least pcg.FLOAT32_TOLERANCE_FLOOR (the validate warning's bound). The residual
    is against the float64 operator; a float32 solve also records it against its own float32 operator."""
    from open_sol.solvers import pcg
    backend, xp = arrays.backend, arrays.xp
    precision = values['precision']
    dtype = pcg.PRECISION_DTYPES[precision]
    tolerance = float(values['tolerance'])
    if precision == 'single':
        tolerance = max(tolerance, pcg.FLOAT32_TOLERANCE_FLOOR)
    max_iterations = int(values['max_iterations']) or pcg.auto_max_iterations(problem.longest_axis)
    solver = pcg.PCGSolver(precision=precision, tolerance=tolerance, max_iterations=max_iterations, label=_label,
                           backend=backend, preconditioner=values['preconditioner'])
    operator = arrays.operator.astype(dtype)
    rhs = xp.ascontiguousarray(arrays.rhs, dtype=dtype)
    solution = xp.zeros(problem.n_unknowns, dtype=dtype)
    started = _clock(backend)
    solver.factorize(operator)
    factored = _clock(backend)
    converged = solver.solve(rhs, solution, reference='rhs')
    solved = _clock(backend)
    iterations = solver.last_iters
    scaled = _scaled(operator)
    refactor_started = _clock(backend)
    solver.factorize(scaled)
    refactored = _clock(backend)
    counters = solver.work_counters()
    details = {'tolerance': tolerance, 'max_iterations': max_iterations, 'iterations': _number(iterations),
               'converged': bool(converged), 'last_warning': solver.last_warning,
               'counters': {name: _number(value) for name, value in counters.items()}}
    if np.dtype(dtype) != np.dtype(problem.operator.dtype):
        details['residual_own_operator'] = relative_residual(problem.operator.astype(dtype),
                                                             np.asarray(problem.rhs, dtype=dtype), solution)
    phases = {'analysis': None, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details, tolerance,
                   'ok' if converged else 'unconverged')


def run_banded(problem, values, arrays) -> dict:
    """banded: host banded Cholesky of a 1D window (BandedSolver refuses wider bands: 'refused'; gpu: the operator,
    rhs and solution move per call)."""
    from open_sol.solvers.banded import BandedSolver
    backend = arrays.backend
    solver = BandedSolver(label=_label, backend=backend)
    solution = arrays.xp.zeros(problem.n_unknowns)
    started = _clock(backend)
    solver.factorize(arrays.operator)
    factored = _clock(backend)
    solver.solve(arrays.rhs, solution)
    solved = _clock(backend)
    scaled = _scaled(arrays.operator)
    refactor_started = _clock(backend)
    solver.factorize(scaled)
    refactored = _clock(backend)
    phases = {'analysis': None, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution),
                   {'bandwidth': solver.bandwidth}, _direct_tolerance())


def run_line_block(problem, values, arrays) -> dict:
    """line_block ('linear' of em_fdfd_ez / _hz): one kept factor of the host operator (factorize_operators on the
    FDFD layout, as fdfd_2d; block inverses on the backend) and one solve of the host rhs; a solve above the accept
    tolerance is 'rejected' (the physics redoes it on host SuperLU)."""
    from open_sol.physics.em.fdfd_line_block import fdfd_layout
    from open_sol.solvers.line_block import LineBlockSolver
    backend = arrays.backend
    nx, ny = problem.grid_shape
    layout = fdfd_layout(nx, ny, False, False)
    solver = LineBlockSolver(precision=values['precision'], label=_label, backend=backend)
    started = _clock(backend)
    factor = solver.factorize_operators([problem.operator], layout)
    factored = _clock(backend)
    solution, relative, accepted = factor.solve(problem.rhs, 0)
    solved = _clock(backend)
    factor.release()
    scaled = _scaled(problem.operator)
    refactor_started = _clock(backend)
    factor = solver.factorize_operators([scaled], layout)
    refactored = _clock(backend)
    factor.release()
    solver.release()
    details = {'blocks': layout.n_blocks, 'block_size': layout.block_size, 'line_block_residual': float(relative),
               'accepted': bool(accepted), 'refinement_sweeps': solver.n_refinement_sweeps}
    phases = {'analysis': None, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details,
                   solver.accept_tolerance, 'ok' if accepted else 'rejected')


def _eigen_residual(problem, resonance) -> float:
    """||A u + (k^2 - k_t^2) M u|| / (||A u|| + |k^2 - k_t^2| ||M u||) of a resonance (A = L + k_t^2 M assembled at
    the target, eigenfrequency module header: L u = -k^2 M u)."""
    from open_sol.core.constants import C0
    vector = np.asarray(_host(resonance.field), dtype=np.complex128).ravel()
    shift = (complex(resonance.omega) / C0) ** 2 - (problem.eigen['omega'] / C0) ** 2
    operator_part = problem.operator @ vector
    mass_part = problem.eigen['mass'] * vector
    scale = float(np.linalg.norm(operator_part)) + abs(shift) * float(np.linalg.norm(mass_part))
    residual = float(np.linalg.norm(operator_part + shift * mass_part))
    return residual / scale if scale > 0.0 else residual


def run_arpack(problem, values, arrays) -> dict:
    """arpack_shift_invert: eigenfrequency.solve_resonances on the host operator with the factorization, the FDFD
    symmetrizer (used by the LDL^T factorizations only, as fdfd_2d) and the model runtime's mixed_min_unknowns (S5c: a
    *_mixed request below it factors in double); on the gpu backend device_arnoldi as a gpu model runtime; eigenvalues
    returned for the comparison against SuperLU."""
    from open_sol.physics.em.eigenfrequency import solve_resonances
    from open_sol.solvers.arpack_shift_invert import MIXED_EIGEN_MIN_UNKNOWNS, ArpackShiftInvertSolver
    backend = arrays.backend
    eigen = problem.eigen
    solver = ArpackShiftInvertSolver(tolerance=values['tolerance'], subspace_size=values['subspace_size'],
                                     label=_label, factorization=values['factorization'],
                                     mixed_min_unknowns=MIXED_EIGEN_MIN_UNKNOWNS, device_arnoldi=backend == 'gpu')
    started = _clock(backend)
    solve = solve_resonances(problem.operator, eigen['mass'], eigen['omega'], eigen['energy_weights'],
                             eigen['in_window'], eigen['n_resonances'], eigen_solver=solver,
                             symmetrizer=problem.symmetrizer)
    finished = _clock(backend)
    info = solve.info
    residuals = [_eigen_residual(problem, resonance) for resonance in solve.resonances]
    details = {'factorization_used': solver.factorization_for(problem.n_unknowns),
               'factor_kind': info.get('factor_text') or 'SuperLU', 'library_threads': solver.host_direct_threads,
               'factor_entries': _number(solver.lu_entries), 'operator_applications': info['operator_applications'],
               'widenings': info['widenings'], 'eigenvalues_converged': info['eigenvalues_converged'],
               'resonances_found': len(solve.resonances), 'warnings': list(info['warnings']),
               'eigenvalues': [[float(complex(resonance.omega).real), float(complex(resonance.omega).imag)]
                               for resonance in solve.resonances]}
    phases = {'analysis': None, 'factor': info['factor_seconds'], 'solve': info['eigen_seconds'], 'refactor': None,
              'total': finished - started}
    tolerance = max(float(values['tolerance']), _direct_tolerance())
    return _repeat(phases, max(residuals) if residuals else None, details, tolerance,
                   'ok' if solve.resonances else 'no_resonances')


RUNNERS = {'sparse_direct': run_sparse_direct, 'pcg': run_pcg, 'banded': run_banded, 'line_block': run_line_block,
           'arpack_shift_invert': run_arpack}

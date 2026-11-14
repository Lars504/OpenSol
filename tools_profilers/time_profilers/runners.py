"""One timed repeat of a profiler Problem per SolverType, in the child process, through the calls open_sol's own
consumers make (SparseDirectSolver.factorize like fdfd_2d, DiffusionDirectSolver like DiffusionSystem, PCGSolver,
BandedSolver, LineBlockSolver.factorize_operators, eigenfrequency.solve_resonances, time_filter_eigen (an FDTD run on an
FDFDPhysics domain + Rayleigh-Ritz), ElasticSystem3D.lowest_modes,
fem.harmonic frequency sweeps: solve per frequency, batched_direct through plan_batched) on
the cpu or gpu backend: phase wall times, the relative residual against its tolerance and solver details. open_sol is
imported inside the runners."""
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
# Matrix-free vibration modes (problem kind 'modes'): the system (ElasticSystem3D with the Automatic 'linear' PCG) is
# assembled on the backend by backend_arrays and kept for every repeat; arpack_shift_invert and block_eigen both run
# the physics' own lowest_modes (eigsh_inverse with inner PCG solves / the block path). Phases: solve = the eigen
# iterations (ModalSolve info eigen_seconds), factor = block_eigen's set-up around them (float32 copies, a private
# multigrid hierarchy; arpack: none), total = the lowest_modes call. Residual = max ||K u - omega^2 M u|| / (omega^2 ||M
# u||) of the returned modes (float64 brick products; block_eigen's own measure), tolerance at least the solver's
# (arpack: the Lanczos tolerance it used); reference_error = max relative omega error against the closed form (rank
# order); gpu_peak_bytes = the cupy pool's high water of the call over its live bytes at the start (cached blocks
# freed before the timer).
# Frequency sweeps (problem kind 'sweep', acoustic_sweep_2d): a fresh fem.harmonic system per repeat on the runtime
# (assembled before the timer), every frequency solved in order; phases factor = the first frequency, solve = the mean
# of each later one, total = the sweep (refactor None: no sweep column); the residual is the worst frequency's.
import math
import time
from dataclasses import dataclass

import numpy as np

REFACTOR_SCALE = 1.0 + 1e-3   # the refactor phase factorises A x this: a value-only change, the same pattern
ACCURACY_MARGIN = 10.0        # residual / tolerance above which a repeat is 'inaccurate'


@dataclass(frozen=True, eq=False)
class BackendArrays:
    """The problem's operator and rhs on the run backend ('cpu': the host scipy / numpy objects themselves, 'gpu':
    cupyx CSR / cupy copies) and that backend's array module; a matrix-free problem: operator / rhs None and system =
    its problems.ModesSystem assembled on the backend."""
    backend: str
    operator: object
    rhs: object
    xp: object
    system: object = None


def backend_arrays(problem, backend) -> BackendArrays:
    """BackendArrays of a problem; on 'gpu' the device copies are made here (and the GPU budget set as api.run does);
    a 'modes' problem's system is assembled here on either backend."""
    if problem.kind == 'modes':
        from tools_profilers.time_profilers.problems import elastic_modes_system
        if backend == 'gpu':
            from open_sol.core import gpu_memory
            gpu_memory.set_budget()
        modes = elastic_modes_system(problem, backend)
        _clock(backend)   # the assembly's queued kernels finish before the memory baseline
        return BackendArrays(backend, None, None, modes.system.xp, system=modes)
    if backend == 'cpu':
        return BackendArrays('cpu', problem.operator, problem.rhs, np)
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    from open_sol.core import gpu_memory
    gpu_memory.set_budget()
    if problem.kind in ('eigen', 'sweep'):   # solve_resonances / fem.harmonic take host operators on both backends
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
    through SparseDirectSolver.factorize (the LDL^T of the symmetrised D A on cuDSS / MUMPS / PARDISO, as fdfd_2d),
    an acoustic sweep through fem.harmonic (_run_sweep: one factor per frequency)."""
    if problem.kind == 'spd':
        return _run_diffusion_direct(problem, values, arrays)
    if problem.kind == 'sweep':
        from open_sol.solvers.sparse_direct import SparseDirectSolver
        return _run_sweep(problem, SparseDirectSolver(back_end=values['back_end'], label=_label,
                                                      backend=arrays.backend), arrays, batched=False)
    return _run_grid_direct(problem, values, arrays)


def run_batched_direct(problem, values, arrays) -> dict:
    """batched_direct ('linear' of pressure_acoustics / membrane, opt-in): the acoustic sweep's frequencies through
    fem.harmonic plan_batched (as the physics' plan_frequency_response hook) on the runtime the SolverType create
    makes; the other problems have no batch of same-pattern systems ('refused')."""
    if problem.kind != 'sweep':
        raise ValueError(f"batched_direct runs on the frequency sweep problems only, not on a {problem.kind!r} one")
    from open_sol.solvers.batched_direct import BatchedDirectSolver
    runtime = BatchedDirectSolver(precision=values['precision'], batch_size=values['batch_size'], label=_label,
                                  backend=arrays.backend)
    return _run_sweep(problem, runtime, arrays, batched=True)


def _run_sweep(problem, runtime, arrays, batched) -> dict:
    """Every frequency of a 'sweep' problem through a fresh problems.sweep_system on `runtime` (assembled before the
    timer): solve(omega) in order, after plan_batched(omegas) when batched (the physics hook's call). Phases: factor =
    the first frequency (plan, first factor / first batch), solve = the mean of each later one (its refactorisation
    and solve; batched: mostly taking a solved system, plus the later batches), total = the whole sweep. Residual: the
    largest ||b - A(omega) x|| / ||b|| over the frequencies (host, after the timer); details: the runtime's work
    counters."""
    from tools_profilers.time_profilers.problems import sweep_system
    backend = arrays.backend
    system = sweep_system(problem, runtime)
    pencil = system.pencil
    omegas = [float(omega) for omega in problem.eigen['omegas']]
    solutions = []
    started = _clock(backend)
    if batched:
        system.plan_batched(omegas)
    first = None
    for omega in omegas:
        system.solve(omega)
        solutions.append(system.solution[pencil.free].copy())   # a host copy of n values per frequency
        if first is None:
            first = _clock(backend)
    finished = _clock(backend)
    load = system.load_vector()[pencil.free]
    residuals = [relative_residual(pencil.reduced(omega), load, solution) for omega, solution in zip(omegas, solutions)]
    counters = {name: _number(value) if not isinstance(value, str) else value
                for name, value in runtime.work_counters().items()}
    details = {'frequencies': len(omegas), 'counters': counters,
               'factor_kind': getattr(system.factor, 'description', None) or counters.get('back_end'),
               'device_bytes': _number(counters.get('cudss_factor_device_bytes') or counters.get('batch_permanent_bytes'))}
    system.clear_reduced()   # the batched plan's outside-pool memory
    _release(system.factor)
    later = len(omegas) - 1
    phases = {'analysis': None, 'factor': first - started,
              'solve': (finished - first) / later if later else None, 'refactor': None, 'total': finished - started}
    return _repeat(phases, max(residuals), details, _direct_tolerance())


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
    returned for the comparison against SuperLU. A 'modes' problem: _run_arpack_modes."""
    from open_sol.physics.em.eigenfrequency import solve_resonances
    from open_sol.solvers.arpack_shift_invert import MIXED_EIGEN_MIN_UNKNOWNS, ArpackShiftInvertSolver
    backend = arrays.backend
    eigen = problem.eigen
    solver = ArpackShiftInvertSolver(tolerance=values['tolerance'], subspace_size=values['subspace_size'],
                                     label=_label, factorization=values['factorization'],
                                     mixed_min_unknowns=MIXED_EIGEN_MIN_UNKNOWNS, device_arnoldi=backend == 'gpu')
    if problem.kind == 'modes':
        return _run_arpack_modes(problem, arrays, solver)
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


def _solve_modes(problem, arrays, eigen_solver):
    """(ModalSolve, wall seconds, gpu pool high water [bytes] or None) of one timed ElasticSystem3D.lowest_modes call
    on the problem's system with its density, as SolidMechanics3DPhysics.solve_eigenfrequencies makes it; on the gpu
    the pool's cached blocks are freed first (outside the timer), so the reserved bytes after the call over the live
    bytes before are the call's device high water."""
    modes = arrays.system
    backend = arrays.backend
    live_before = None
    if backend == 'gpu':
        import cupy as cp
        from open_sol.core import gpu_memory
        cp.get_default_memory_pool().free_all_blocks()
        live_before = gpu_memory.pool_used_bytes()
    started = _clock(backend)
    solve = modes.system.lowest_modes(modes.density, problem.eigen['n_resonances'], eigen_solver)
    finished = _clock(backend)
    high_water = None if live_before is None else max(gpu_memory.pool_reserved_bytes() - live_before, 0)
    return solve, finished - started, high_water


def _mode_residuals(modes, resonances) -> list:
    """||K u - omega^2 M u|| / (omega^2 ||M u||) of each resonance (u: every dof, 0 on the held ones) by the system's
    float64 brick product (masked) and consistent mass on its backend."""
    system = modes.system
    xp, kernels = system.xp, system.kernels
    product = xp.empty(system.n_dofs, dtype=np.float64)
    massed = xp.empty(system.n_dofs, dtype=np.float64)
    residuals = []
    for resonance in resonances:
        vector = xp.asarray(np.ascontiguousarray(resonance.displacement, dtype=np.float64))
        kernels.product(vector, product, system.lame_lambda, system.mu, system.widths_device, system.free,
                        masked=True)
        kernels.mass_product(vector, massed, modes.density, system.widths_device, system.free)
        eigenvalue = float(resonance.omega) ** 2
        scale = eigenvalue * float(xp.linalg.norm(massed))
        residual = float(xp.linalg.norm(product - eigenvalue * massed))
        residuals.append(residual / scale if scale > 0.0 else residual)
    return residuals


def _reference_error(problem, omegas):
    """max |omega - omega_ref| / omega_ref of the found omegas against the closed-form ones, both in ascending order
    (the found ones only; None when none)."""
    pairs = list(zip(sorted(omegas), problem.eigen['reference_omegas']))
    return max(abs(omega - reference) / reference for omega, reference in pairs) if pairs else None


def _modes_details(problem, arrays, solve, high_water) -> dict:
    """Details shared by the mode runners: method, counts, warnings, eigenvalues ([omega, 0] pairs for the report's
    difference against the baseline), the closed-form ones and the error against them, the linear solver settings."""
    info = solve.info
    omegas = [float(resonance.omega) for resonance in solve.resonances]
    details = {'method': info.get('eigen_method') or info.get('method'), 'n_free': info.get('n_free'),
               'operator_applications': info.get('operator_applications'),
               'eigenvalues_requested': info.get('eigenvalues_requested'),
               'eigenvalues_converged': info.get('eigenvalues_converged'), 'resonances_found': len(omegas),
               'warnings': list(info.get('warnings') or ()), 'eigenvalues': [[omega, 0.0] for omega in omegas],
               'reference_eigenvalues': list(problem.eigen['reference_omegas']),
               'reference_error': _reference_error(problem, omegas), 'linear_solver': arrays.system.linear_values}
    if high_water is not None:
        details['gpu_peak_bytes'] = int(high_water)
    return details


def _modes_status(problem, solve, unconverged) -> str:
    """'no_resonances' (none), 'unconverged' (fewer than asked, or the solver stopped short), else 'ok'."""
    if not solve.resonances:
        return 'no_resonances'
    if unconverged or len(solve.resonances) < problem.eigen['n_resonances']:
        return 'unconverged'
    return 'ok'


def _run_arpack_modes(problem, arrays, solver) -> dict:
    """arpack_shift_invert on a 'modes' problem: lowest_modes runs eigsh_inverse (shift-invert Lanczos at 0, one inner
    PCG solve of the system's Automatic linear solver per application; the factorization is unused: no factor).
    Tolerance: the Lanczos tolerance it used (max(node, LANCZOS_TOLERANCE_PER_CG x the CG tolerance)), at least
    ACCEPT_RESIDUAL, as run_arpack."""
    linear_solver = arrays.system.system.linear_solver
    unconverged_before = linear_solver.n_unconverged
    solve, total, high_water = _solve_modes(problem, arrays, solver)
    info = solve.info
    details = _modes_details(problem, arrays, solve, high_water)
    details.update(iterations=_number(info.get('cg_iterations')), inner_solves=info.get('operator_applications'),
                   lanczos_tolerance=info.get('lanczos_tolerance'),
                   counters={name: _number(value) for name, value in solver.work_counters().items()})
    residuals = _mode_residuals(arrays.system, solve.resonances)
    unconverged = (linear_solver.n_unconverged > unconverged_before
                   or info.get('eigenvalues_converged', 0) < info.get('eigenvalues_requested', 0))
    phases = {'analysis': None, 'factor': None, 'solve': info.get('eigen_seconds'), 'refactor': None, 'total': total}
    tolerance = max(float(info.get('lanczos_tolerance') or 0.0), _direct_tolerance())
    return _repeat(phases, max(residuals) if residuals else None, details, tolerance,
                   _modes_status(problem, solve, unconverged))


def run_block_eigen(problem, values, arrays) -> dict:
    """block_eigen ('eigen' of solid_mechanics_3d on brick meshes): lowest_modes with the BlockEigenSolver made from
    the values as the SolverType create makes it (max_iterations 0: lowest_modes passes the linear solver's limit;
    preconditioner multigrid: the linear solver's hierarchy, else one built inside the call, so every repeat pays it
    as every model eigen solve does). Tolerance: the solver's, at least ACCEPT_RESIDUAL."""
    if problem.kind != 'modes':
        raise ValueError(f"block_eigen runs on the matrix-free 'modes' problems only, not on a {problem.kind!r} one")
    from open_sol.solvers.block_eigen import BlockEigenSolver
    solver = BlockEigenSolver(tolerance=values['tolerance'], guard_vectors=values['guard_vectors'],
                              max_iterations=values['max_iterations'], precision=values['precision'],
                              preconditioner=values['preconditioner'], label=_label, backend=arrays.backend)
    solve, total, high_water = _solve_modes(problem, arrays, solver)
    info = solve.info
    counters = {name: _number(value) for name, value in solver.work_counters().items()}
    details = _modes_details(problem, arrays, solve, high_water)
    details.update(iterations=counters['eigen_iterations'], single_iterations=counters['eigen_single_iterations'],
                   double_iterations=counters['eigen_double_iterations'], preconditioner=info.get('preconditioner'),
                   preconditioner_applications=counters['preconditioner_applications'],
                   mass_applications=counters['mass_applications'], block_size=counters['eigen_block_size'],
                   block_residual=info.get('block_residual'), counters=counters)
    residuals = _mode_residuals(arrays.system, solve.resonances)
    eigen_seconds = info.get('eigen_seconds')
    phases = {'analysis': None, 'factor': None if eigen_seconds is None else max(total - eigen_seconds, 0.0),
              'solve': eigen_seconds, 'refactor': None, 'total': total}
    tolerance = max(solver.tolerance, _direct_tolerance())
    return _repeat(phases, max(residuals) if residuals else None, details, tolerance,
                   _modes_status(problem, solve, counters['eigen_unconverged'] > 0))


def _time_filter_physics(problem, backend, runtime):
    """FDFDPhysics (em_fdfd_ez, all CPML) on a Domain of the problem's grid with its disc written into eps_r: what
    time_filter_eigen time-steps (problems.FDFD_* constants; the domain's float32 cell width is the FDTD's own, the
    projection uses the problem's float64 operator)."""
    from open_sol.physics.em import FDFDPhysics
    from open_sol.physics.em.boundaries import BoundaryConfig
    from tools_profilers.time_profilers.problems import FDFD_DISC_EPS, FDFD_PML_CELLS, FDFD_WAVELENGTH, FDFD_WINDOW
    from open_sol.core.domain import Domain
    n = problem.grid_shape[0]
    cell = FDFD_WINDOW / n
    config = BoundaryConfig.all_pml(FDFD_PML_CELLS)
    window = (n - 2 * FDFD_PML_CELLS) * cell
    domain = Domain.uniform(window, window, cell, boundary_config=config, verbose=False, backend=backend)
    if (domain.grid.nx, domain.grid.ny) != tuple(problem.grid_shape):
        raise ValueError(f'time_filter profiler domain {domain.grid.nx} x {domain.grid.ny} differs from the problem '
                         f'grid {problem.grid_shape}')
    physics = FDFDPhysics(domain, 'ez', boundary_config=config, wavelength=FDFD_WAVELENGTH, eigen_solver=runtime)
    disc = domain.xp.asarray(np.asarray(problem.eigen['mass']).reshape(problem.grid_shape).real > 1.0)
    domain.materials.set_mask(disc, eps_r=FDFD_DISC_EPS)
    physics.materials_changed(('eps_r',))
    return physics


def run_time_filter(problem, values, arrays) -> dict:
    """time_filter ('eigen' of em_fdfd_ez / _hz, opt-in): physics.em.time_filter_eigen.solve_time_filter_resonances
    with the arguments fdfd_2d.solve_eigenfrequencies hands it (pinned cells none: all CPML, cell areas, window rows,
    the symmetrizer), the runtime the SolverType create makes, an FDFDPhysics on a Domain of the problem's grid for
    its FDTD run and the problem's own operator / mass as the system (gpu: a cupyx copy made before the timer), so the
    residual and eigenvalues are measured as the arpack rows'. Phases: factor = the FDTD run (no factorisation), solve
    = the projection, polish and filter, total = the call. Tolerance: the runtime's RESIDUAL_WARNING."""
    if problem.kind != 'eigen':
        raise ValueError(f"time_filter runs on the FDFD eigen problems only, not on a {problem.kind!r} one")
    from types import SimpleNamespace
    from open_sol.physics.em.time_filter_eigen import solve_time_filter_resonances
    from open_sol.solvers import time_filter
    backend = arrays.backend
    eigen = problem.eigen
    runtime = time_filter.TimeFilterEigenSolver(
        bandwidth=values['bandwidth'], sampled_frequencies=values['sampled_frequencies'], periods=values['periods'],
        sources=values['sources'], polish_steps=values['polish_steps'], precision=values['precision'], label=_label,
        backend=backend)
    physics = _time_filter_physics(problem, backend, runtime)
    xp = physics.xp
    operator = problem.operator
    if backend == 'gpu':
        import cupyx.scipy.sparse as cpsp
        operator = cpsp.csr_matrix(problem.operator)
    system = SimpleNamespace(operator=operator, cell_eps=eigen['mass'], cell_mu=None, polarization='ez',
                             omega=eigen['omega'])
    shape = tuple(problem.grid_shape)
    pinned = xp.zeros(shape, dtype=bool)
    started = _clock(backend)
    solve = solve_time_filter_resonances(physics, system, runtime, eigen['n_resonances'], pinned,
                                         eigen['energy_weights'], eigen['in_window'],
                                         lambda vector: vector.reshape(shape), symmetrizer=problem.symmetrizer)
    finished = _clock(backend)
    info = solve.info
    residuals = [_eigen_residual(problem, resonance) for resonance in solve.resonances]
    counters = {name: _number(value) for name, value in runtime.work_counters().items()}
    details = {'factor_kind': info.get('factor_text'), 'time_steps': info.get('time_steps'),
               'sampled_frequencies': info.get('sampled_frequencies'), 'basis_columns': counters['basis_columns'],
               'operator_applications': info['operator_applications'], 'resonances_found': len(solve.resonances),
               'warnings': list(info['warnings']), 'own_residuals': list(info.get('residuals') or ()),
               'counters': counters,
               'eigenvalues': [[float(complex(resonance.omega).real), float(complex(resonance.omega).imag)]
                               for resonance in solve.resonances]}
    step_seconds = info.get('step_seconds')
    phases = {'analysis': None, 'factor': step_seconds,
              'solve': None if step_seconds is None else max(info['eigen_seconds'] - step_seconds, 0.0),
              'refactor': None, 'total': finished - started}
    return _repeat(phases, max(residuals) if residuals else None, details, float(time_filter.RESIDUAL_WARNING),
                   'ok' if solve.resonances else 'no_resonances')


RUNNERS = {'sparse_direct': run_sparse_direct, 'batched_direct': run_batched_direct, 'pcg': run_pcg,
           'banded': run_banded, 'line_block': run_line_block, 'arpack_shift_invert': run_arpack,
           'block_eigen': run_block_eigen, 'time_filter': run_time_filter}

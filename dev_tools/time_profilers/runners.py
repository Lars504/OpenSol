"""One timed repeat of a profiler Problem per SolverType, in the child process, through the calls open_sol's own
consumers make (SparseDirectSolver.factorize like fdfd_2d, DiffusionDirectSolver like DiffusionSystem, PCGSolver,
BandedSolver, LineBlockSolver.factorize_operators, eigenfrequency.solve_resonances): phase wall times, the relative
residual and solver details. open_sol is imported inside the runners."""
# Phases (seconds, None where the solver does not expose them): analysis = MUMPS / PARDISO ordering + symbolic
# analysis (inside factor), factor = the first (cold) factorisation / preconditioner set-up incl. analysis and the
# accuracy probe, solve = one solve (pcg: the CG iterations), refactor = a value-only refactorisation of the same
# pattern (host direct: kept analysis, SuperLU: kept ordering, multigrid: kept aggregates), total = factor + solve.
# Eigen: factor = the shifted factorisation, solve = the ARPACK iterations, total = the whole solve_resonances.
import time

import numpy as np

REFACTOR_SCALE = 1.0 + 1e-3   # the refactor phase factorises A x this: a value-only change, the same pattern


def _label():
    return 'profiler solver'


def relative_residual(operator, rhs, solution) -> float:
    """||rhs - A x|| / ||rhs|| in float64 / complex128 (x cast up first)."""
    rhs = np.asarray(rhs)
    solution = np.asarray(solution).astype(np.result_type(rhs.dtype, operator.dtype, np.float64), copy=False)
    norm = float(np.linalg.norm(rhs))
    residual = float(np.linalg.norm(rhs - operator @ solution))
    return residual / norm if norm > 0.0 else residual


def _scaled(operator):
    """A copy of the CSR operator with every value times REFACTOR_SCALE (same indptr / indices)."""
    scaled = operator.copy()
    scaled.data *= REFACTOR_SCALE
    return scaled


def _release(factor) -> None:
    release = getattr(factor, 'release', None)   # HostDirectFactorization / LineBlockFactor (SuperLU: none)
    if release is not None:
        release()


def _number(value):
    """float / int / None of a numpy or Python number (JSON)."""
    if value is None:
        return None
    return value.item() if hasattr(value, 'item') else value


def _factor_details(factor) -> dict:
    """Details of a sparse_direct Factorization / HostDirectFactorization after its first solve."""
    return {'factor_kind': getattr(factor, 'description', None) or 'SuperLU',
            'factor_precision': getattr(factor, 'factor_precision', 'double'),
            'analysis_seconds': _number(getattr(factor, 'analysis_seconds', None)),
            'factor_bytes': _number(getattr(factor, 'host_bytes', None)),
            'library_peak_bytes': _number(getattr(factor, 'peak_bytes', None)),
            'factor_entries': _number(getattr(factor, 'lu_entries', None)),
            'library_threads': _number(getattr(factor, 'threads', None)),
            'openblas_pools_held': _number(getattr(factor, 'openblas_pools', None)),
            'probe_refinements': _number(getattr(factor, 'refinement_steps', None)),
            'solve_refinements': _number(getattr(factor, 'last_refinements', None)),
            'perturbed_pivots': _number(getattr(factor, 'perturbed_pivots', None))}


def _repeat(phases, residual, details, status='ok') -> dict:
    phases = {name: _number(seconds) for name, seconds in phases.items()}
    return {'status': status, 'phases': phases, 'residual': residual, 'details': details}


def run_sparse_direct(problem, values) -> dict:
    """sparse_direct: a real SPD problem through DiffusionDirectSolver (the diffusion types' consumer), the FDFD drive
    through SparseDirectSolver.factorize (the LDL^T of the symmetrised D A on MUMPS / PARDISO, as fdfd_2d)."""
    if problem.kind == 'spd':
        return _run_diffusion_direct(problem, values)
    return _run_grid_direct(problem, values)


def _run_grid_direct(problem, values) -> dict:
    from open_sol.solvers.host_direct import HOST_DIRECT_BACK_ENDS
    from open_sol.solvers.sparse_direct import SparseDirectSolver
    back_end = values['back_end']
    nx, ny = problem.grid_shape
    options = {'symmetric': True, 'symmetrizer': problem.symmetrizer} if back_end in HOST_DIRECT_BACK_ENDS else {}
    runtime = SparseDirectSolver(back_end=back_end, label=_label, backend='cpu')   # fresh: no kept ordering
    started = time.perf_counter()
    factor = runtime.factorize(problem.operator, nx, ny, False, False, **options)
    factored = time.perf_counter()
    solution = factor.solve(problem.rhs)
    solved = time.perf_counter()
    details = _factor_details(factor)
    analysis = details.pop('analysis_seconds')
    refactor_started = time.perf_counter()
    factor = runtime.factorize(_scaled(problem.operator), nx, ny, False, False, previous=factor, **options)
    refactored = time.perf_counter()
    details['refactor_reused_analysis'] = bool(getattr(factor, 'last_plan_reused', False))
    details['reused_orderings'] = runtime.orderings.n_reused
    _release(factor)
    phases = {'analysis': analysis, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details)


def _run_diffusion_direct(problem, values) -> dict:
    from open_sol.solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
    runtime = SparseDirectSolver(back_end=values['back_end'], label=_label, backend='cpu')
    direct = DiffusionDirectSolver(runtime)
    direct.grid_shape = problem.grid_shape   # a 3D window: SuperLU starts from grid nested dissection
    solution = np.zeros(problem.n_unknowns)
    started = time.perf_counter()
    direct.factorize(problem.operator)
    factored = time.perf_counter()
    direct.solve(problem.rhs, solution)
    solved = time.perf_counter()
    details = _factor_details(direct._factor)   # the solver keeps its factor private; read-only here
    analysis = details.pop('analysis_seconds')
    refactor_started = time.perf_counter()
    direct.factorize(_scaled(problem.operator))   # host direct: previous factor refactorised; SuperLU: kept order
    refactored = time.perf_counter()
    details['refactor_reused_analysis'] = bool(getattr(direct._factor, 'last_plan_reused', False))
    details['reused_orderings'] = runtime.orderings.n_reused
    direct.release_factor()
    phases = {'analysis': analysis, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details)


def run_pcg(problem, values) -> dict:
    """pcg: factorize (Jacobi / multigrid set-up) on the operator in the solver dtype, one cold CG solve; single
    precision runs at a tolerance of at least pcg.FLOAT32_TOLERANCE_FLOOR (the validate warning's bound)."""
    from open_sol.solvers import pcg
    precision = values['precision']
    dtype = pcg.PRECISION_DTYPES[precision]
    tolerance = float(values['tolerance'])
    if precision == 'single':
        tolerance = max(tolerance, pcg.FLOAT32_TOLERANCE_FLOOR)
    max_iterations = int(values['max_iterations']) or pcg.auto_max_iterations(problem.longest_axis)
    solver = pcg.PCGSolver(precision=precision, tolerance=tolerance, max_iterations=max_iterations, label=_label,
                           backend='cpu', preconditioner=values['preconditioner'])
    operator = problem.operator.astype(dtype)
    rhs = np.ascontiguousarray(problem.rhs, dtype=dtype)
    solution = np.zeros(problem.n_unknowns, dtype=dtype)
    started = time.perf_counter()
    solver.factorize(operator)
    factored = time.perf_counter()
    converged = solver.solve(rhs, solution, reference='rhs')
    solved = time.perf_counter()
    iterations = solver.last_iters
    refactor_started = time.perf_counter()
    solver.factorize(_scaled(operator))
    refactored = time.perf_counter()
    counters = solver.work_counters()
    details = {'tolerance': tolerance, 'max_iterations': max_iterations, 'iterations': iterations,
               'converged': bool(converged), 'last_warning': solver.last_warning,
               'counters': {name: _number(value) for name, value in counters.items()}}
    phases = {'analysis': None, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details,
                   'ok' if converged else 'unconverged')


def run_banded(problem, values) -> dict:
    """banded: host banded Cholesky of a 1D window (BandedSolver refuses wider bands: 'refused')."""
    from open_sol.solvers.banded import BandedSolver
    solver = BandedSolver(label=_label, backend='cpu')
    solution = np.zeros(problem.n_unknowns)
    started = time.perf_counter()
    solver.factorize(problem.operator)
    factored = time.perf_counter()
    solver.solve(problem.rhs, solution)
    solved = time.perf_counter()
    refactor_started = time.perf_counter()
    solver.factorize(_scaled(problem.operator))
    refactored = time.perf_counter()
    phases = {'analysis': None, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution),
                   {'bandwidth': solver.bandwidth})


def run_line_block(problem, values) -> dict:
    """line_block ('linear' of em_fdfd_ez / _hz): one kept factor of the operator (factorize_operators on the FDFD
    layout, as fdfd_2d) and one solve; a solve above the accept tolerance is 'rejected' (the physics redoes it on host
    SuperLU)."""
    from open_sol.physics.em.fdfd_line_block import fdfd_layout
    from open_sol.solvers.line_block import LineBlockSolver
    nx, ny = problem.grid_shape
    layout = fdfd_layout(nx, ny, False, False)
    solver = LineBlockSolver(precision=values['precision'], label=_label, backend='cpu')
    started = time.perf_counter()
    factor = solver.factorize_operators([problem.operator], layout)
    factored = time.perf_counter()
    solution, relative, accepted = factor.solve(problem.rhs, 0)
    solved = time.perf_counter()
    factor.release()
    refactor_started = time.perf_counter()
    factor = solver.factorize_operators([_scaled(problem.operator)], layout)
    refactored = time.perf_counter()
    factor.release()
    solver.release()
    details = {'blocks': layout.n_blocks, 'block_size': layout.block_size, 'line_block_residual': float(relative),
               'accepted': bool(accepted), 'refinement_sweeps': solver.n_refinement_sweeps}
    phases = {'analysis': None, 'factor': factored - started, 'solve': solved - factored,
              'refactor': refactored - refactor_started, 'total': solved - started}
    return _repeat(phases, relative_residual(problem.operator, problem.rhs, solution), details,
                   'ok' if accepted else 'rejected')


def _eigen_residual(problem, resonance) -> float:
    """||A u + (k^2 - k_t^2) M u|| / (||A u|| + |k^2 - k_t^2| ||M u||) of a resonance (A = L + k_t^2 M assembled at
    the target, eigenfrequency module header: L u = -k^2 M u)."""
    from open_sol.core.constants import C0
    vector = np.asarray(resonance.field, dtype=np.complex128).ravel()
    shift = (complex(resonance.omega) / C0) ** 2 - (problem.eigen['omega'] / C0) ** 2
    operator_part = problem.operator @ vector
    mass_part = problem.eigen['mass'] * vector
    scale = float(np.linalg.norm(operator_part)) + abs(shift) * float(np.linalg.norm(mass_part))
    residual = float(np.linalg.norm(operator_part + shift * mass_part))
    return residual / scale if scale > 0.0 else residual


def run_arpack(problem, values) -> dict:
    """arpack_shift_invert: eigenfrequency.solve_resonances with the factorization, the FDFD symmetrizer (used by
    the LDL^T factorizations only, as fdfd_2d) and the model runtime's mixed_min_unknowns (S5c: a *_mixed request
    below it factors in double); eigenvalues returned for the comparison against SuperLU."""
    from open_sol.physics.em.eigenfrequency import solve_resonances
    from open_sol.solvers.arpack_shift_invert import MIXED_EIGEN_MIN_UNKNOWNS, ArpackShiftInvertSolver
    eigen = problem.eigen
    solver = ArpackShiftInvertSolver(tolerance=values['tolerance'], subspace_size=values['subspace_size'],
                                     label=_label, factorization=values['factorization'],
                                     mixed_min_unknowns=MIXED_EIGEN_MIN_UNKNOWNS)
    started = time.perf_counter()
    solve = solve_resonances(problem.operator, eigen['mass'], eigen['omega'], eigen['energy_weights'],
                             eigen['in_window'], eigen['n_resonances'], eigen_solver=solver,
                             symmetrizer=problem.symmetrizer)
    finished = time.perf_counter()
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
    return _repeat(phases, max(residuals) if residuals else None, details,
                   'ok' if solve.resonances else 'no_resonances')


RUNNERS = {'sparse_direct': run_sparse_direct, 'pcg': run_pcg, 'banded': run_banded, 'line_block': run_line_block,
           'arpack_shift_invert': run_arpack}

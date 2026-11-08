"""The 'optimization' StudyType (C7): density topology optimisation of the design_region features of the em_fdfd_* /
em_eme physics, maximising J = the sum of their adjoint_objective terms by scipy L-BFGS-B on the adjoint gradient."""
import time

import numpy as np

from .schema import Issue, Param, StudyType
from .solver_hooks import AdjointSolver, DesignAdjointSolver, HarmonicAdjointSolver
from .study import (_LIVE_UPDATE_PARAM, Cancelled, LogEvent, _count_progress, _LiveFeed, _log_solve_summaries,
                    _scipy_issues, _SnapshotRecorder, _SolverWarnings, _StageProgress)

STUDY_NAME = 'optimization'
_DESIGN_FEATURE = 'design_region'          # = physics.em.definition_design feature type (also em_eme's)
_OBJECTIVE_FEATURE = 'adjoint_objective'   # = physics.em.definition_adjoint / physics.eme.definition feature type
_EVALUATIONS_PER_ITERATION = 5             # L-BFGS-B maxfun = this x max_iterations (line-search trials included)
_OBJECTIVE_FLOOR = 1e-30                   # |J| (W/m, em_eme 1) of the initial design below it leaves J unscaled


def _objective_unit_text(solvers):
    """' W/m' (FDFD objectives), '' when every physics declares objective_unit '1' (em_eme power fractions)."""
    units = {getattr(solver, 'objective_unit', 'W/m') for solver in solvers.values()}
    return '' if units == {'1'} else ' W/m'


class _DesignProblem:
    """J and dJ/d densities of every design region: writes the densities, solves every physics at its wavelength
    setting and chains dJ/d eps of every physics reading a region's arrays back to that region's densities."""

    def __init__(self, session, solvers, max_iterations, recorder, live, emit, control, service):
        self.runner = session.runner
        self.solvers = solvers
        # each FDFD physics' wavelength setting; a DesignAdjointSolver (em_fdfd_nonlinear) solves its own channels
        self.omegas = {name: solver.omega for name, solver in solvers.items()
                       if not isinstance(solver, DesignAdjointSolver)}
        self.regions = [region for solver in solvers.values() for region in solver.design_regions]
        for index, region in enumerate(self.regions):   # same-physics overlaps are refused when the region is added
            for other in self.regions[index + 1:]:
                if region.overlaps(other):
                    raise ValueError('two design regions of different physics write the same permittivity array on a '
                                     'common cell; give them disjoint domains')
        self.offsets = np.concatenate(([0], np.cumsum([region.n_cells for region in self.regions]))).astype(int)
        self.written_arrays = tuple(sorted({name for region in self.regions for name in region.array_names}))
        self.max_iterations = max_iterations
        self.recorder, self.live, self.emit = recorder, live, emit
        self.control, self.service = control, service
        self.solver_warnings = _SolverWarnings(solvers, emit)
        self.unit_text = _objective_unit_text(solvers)   # ' W/m' or '' after J in labels and log lines
        self.scale = 1.0              # 1 / |J| of the initial design (L-BFGS-B sees an O(1) objective)
        self.iteration = 0            # accepted L-BFGS-B iterations
        self.n_evaluations = 0
        self.last = None              # (densities, J, dJ/d densities) of the state the solvers hold
        self.snapshot_current = False   # the last snapshot shows the state the solvers hold
        self.wall_start = time.perf_counter()

    @property
    def n_cells(self):
        return int(self.offsets[-1])

    def initial_densities(self):
        """Concatenated densities of every region (the initial_density designs at the start)."""
        return np.concatenate([region.density for region in self.regions])

    def evaluate(self, densities):
        """(J [W/m, em_eme 1], dJ/d densities) at `densities` (no solve when the solvers already hold that state)."""
        if self.last is not None and np.array_equal(self.last[0], densities):
            return self.last[1], self.last[2]
        self.control.checkpoint(self.service)
        self._progress(f'Iteration {self.iteration + 1}: solving evaluation {self.n_evaluations + 1}')
        for index, region in enumerate(self.regions):
            region.set_density(densities[self.offsets[index]:self.offsets[index + 1]])
        for solver in self.solvers.values():
            solver.materials_changed(self.written_arrays)
        objective = 0.0
        eps_gradients = [np.zeros(region.n_cells) for region in self.regions]
        for name, solver in self.solvers.items():
            if name in self.omegas:
                solver.solve_harmonic(self.omegas[name])
            else:
                solver.solve_design()
            sensitivity = solver.eps_sensitivity()   # one transposed solve (per channel) when it has objectives
            objective += sensitivity.objective
            for array_name, cell_gradient in sensitivity.cell_gradients.items():
                for index, region in enumerate(self.regions):
                    if array_name in region.array_names:
                        eps_gradients[index] += region.gather(cell_gradient)
        gradient = np.concatenate([region.density_gradient(eps_gradient)
                                   for region, eps_gradient in zip(self.regions, eps_gradients)])
        self.n_evaluations += 1
        self.last = (np.array(densities, dtype=np.float64), objective, gradient)
        self.snapshot_current = False
        self.solver_warnings.check()
        self._progress(f'Iteration {self.iteration + 1}, evaluation {self.n_evaluations}: J = '
                       f'{objective:.6g}{self.unit_text}')
        self.live.update()
        return objective, gradient

    def scaled(self, densities):
        """scipy objective: (-J scale, -dJ/d densities scale) (L-BFGS-B minimises)."""
        objective, gradient = self.evaluate(np.asarray(densities, dtype=np.float64))
        return -self.scale * objective, -self.scale * gradient

    def accept(self, densities):
        """L-BFGS-B iteration callback: the accepted iterate (solved again only if the last trial was elsewhere)."""
        objective, _ = self.evaluate(np.asarray(densities, dtype=np.float64))
        self.iteration += 1
        self.record(objective)

    def record(self, objective):
        """History point of every region and one snapshot labelled 'iteration k: J = ...' (+ ' W/m' unless em_eme)."""
        for region in self.regions:
            region.record(self.iteration, objective)
        self.recorder.record(label=f'iteration {self.iteration}: J = {objective:.6g}{self.unit_text}')
        self.snapshot_current = not self.recorder.full
        self.emit(LogEvent('info', f'optimization iteration {self.iteration}: J = {objective:.6g}{self.unit_text} '
                                   f'({self.n_evaluations} evaluations)'))

    def _progress(self, message):
        self.emit(_count_progress(self.runner, min(self.iteration / self.max_iterations, 1.0), self.iteration,
                                  self.wall_start, message, 'iteration'))


def _design_solvers(runner):
    """{name: solver} of every physics; ValueError unless each is an FDFD or 2D EME physics (eps_sensitivity), nothing
    couples them and at least one has a design region."""
    for name, solver in runner.physics.items():
        if not isinstance(solver, AdjointSolver):
            raise ValueError(f"physics '{name}' has no adjoint sensitivity: an optimization study runs em_fdfd_ez / "
                             'em_fdfd_hz / em_fdfd_nonlinear / em_eme physics only')
    if runner.couplings:
        raise ValueError(f"'{runner.couplings[0].name}' couples physics or rewrites materials, which an optimization "
                         'study does not support (it would overwrite the design permittivity or need its own '
                         'gradient): remove couplings and state-dependent material properties')
    if not any(solver.design_regions for solver in runner.physics.values()):
        raise ValueError(f'no {_DESIGN_FEATURE} feature: add one to an em_fdfd_ez / em_fdfd_hz / em_eme physics')
    return dict(runner.physics)


def _run_optimization(session, values, emit, control, service):
    """Solve the initial design, then per continuation stage (projection beta raised toward the design_region values)
    L-BFGS-B over every design density (bounds [0, 1]) maximising J with one snapshot and history point per accepted
    iteration; the solvers end on the returned design."""
    import scipy.optimize
    runner = session.runner
    solvers = _design_solvers(runner)
    runner.prepare()
    recorder = _SnapshotRecorder(session, emit)
    live = _LiveFeed(session, control, emit, values['live_update_hz'])
    max_iterations = values['max_iterations']
    stages = values['continuation_stages']
    stage_emit = _StageProgress(emit, ['Optimization']).stage_emit('Optimization')
    problem = _DesignProblem(session, solvers, max_iterations * stages, recorder, live, stage_emit, control, service)
    final_betas = [region.map.projection_beta for region in problem.regions]
    stage_text = f' in each of {stages} beta-continuation stages' if stages > 1 else ''
    emit(LogEvent('info', f'Optimization study: {problem.n_cells} design cells in {len(problem.regions)} design '
                          f'region(s), L-BFGS-B up to {max_iterations} iterations{stage_text}; physics '
                          f"{', '.join(solvers)}"))
    status, stage_messages = 'completed', []
    initial_objective = final_objective = float('nan')
    try:
        densities = problem.initial_densities()
        for stage in range(stages):
            if stages > 1:
                beta_factor = values['beta_growth'] ** (stage + 1 - stages)   # the last stage = the feature beta
                for region, final_beta in zip(problem.regions, final_betas):
                    region.set_projection_beta(final_beta * beta_factor)
                problem.last = None   # equal densities now map to another eps: re-solve
                emit(LogEvent('info', f'continuation stage {stage + 1}/{stages}: projection beta = {beta_factor:.4g} '
                                      'x the design_region projection_beta'))
            if stage == 0:
                initial_objective, _ = problem.evaluate(densities)
                _log_solve_summaries(solvers, emit, ' (initial design)')
                if abs(initial_objective) > _OBJECTIVE_FLOOR:
                    problem.scale = 1.0 / abs(initial_objective)
                else:
                    emit(LogEvent('warning', f'J of the initial design is {initial_objective:.3g}{problem.unit_text}: '
                                             'the objective is not scaled, and a zero gradient leaves the design '
                                             'unchanged'))
                problem.record(initial_objective)
            else:   # the stage start at its beta (L-BFGS-B's first call reuses it: no extra solve)
                problem.evaluate(densities)
            result = scipy.optimize.minimize(
                problem.scaled, densities, jac=True, method='L-BFGS-B',
                bounds=scipy.optimize.Bounds(np.zeros(problem.n_cells), np.ones(problem.n_cells)),
                callback=problem.accept,
                options={'maxiter': max_iterations, 'maxfun': _EVALUATIONS_PER_ITERATION * max_iterations,
                         'ftol': values['tolerance'], 'gtol': 0.0})
            densities = np.asarray(result.x, dtype=np.float64)
            stage_messages.append(f'stage {stage + 1}: {result.message}' if stages > 1 else str(result.message))
        final_objective, _ = problem.evaluate(densities)   # no solve when current
        emit(LogEvent('info', f'L-BFGS-B stopped after {problem.iteration} iteration(s), {problem.n_evaluations} '
                              f"evaluations: {'; '.join(stage_messages)}; J {initial_objective:.6g} -> "
                              f'{final_objective:.6g}{problem.unit_text}'))
    except Cancelled:
        status = 'cancelled'
        if problem.last is not None:
            final_objective = problem.last[1]
    message = '; '.join(stage_messages)
    if not problem.snapshot_current and problem.last is not None:   # the state differs from the last snapshot
        recorder.record(final=True, label=f'final: J = {problem.last[1]:.6g}{problem.unit_text}')
    live.update(final=True)
    _log_solve_summaries(solvers, emit, ' (last evaluation)')
    return {'status': status, 'ticks': problem.iteration, 'tick_unit': 'iteration', 't_end': runner.t,
            'optimization': {'iterations': problem.iteration, 'evaluations': problem.n_evaluations,
                             'design_cells': problem.n_cells, 'objective_initial': initial_objective,
                             'objective_final': final_objective, 'continuation_stages': stages,
                             'message': message}}


def _validate_optimization(vctx, values):
    """Errors without scipy, without an enabled design_region or adjoint_objective feature, and for enabled
    couplings."""
    issues = _scipy_issues(STUDY_NAME, 'L-BFGS-B')
    enabled_types = {feature.type for physics in vctx.spec.physics for feature in physics.features if feature.enabled}
    for feature_type, role in ((_DESIGN_FEATURE, 'the cells it may change'), (_OBJECTIVE_FEATURE, 'what it maximises')):
        if feature_type not in enabled_types:
            issues.append(Issue('error', ('study',), f'the optimization study needs an enabled {feature_type} feature '
                                                     f'on an em_fdfd_ez / em_fdfd_hz / em_fdfd_nonlinear / em_eme '
                                                     f'physics ({role})'))
    for coupling in vctx.spec.couplings:
        if coupling.enabled:
            issues.append(Issue('error', ('coupling', coupling.name), 'an optimization study runs the FDFD physics '
                                                                      'uncoupled: disable this coupling'))
    if values['continuation_stages'] > 1 and not any(
            feature.enabled and feature.type == _DESIGN_FEATURE and feature.params.get('projection_beta', 0.0) != 0.0
            for physics in vctx.spec.physics for feature in physics.features):
        issues.append(Issue('info', ('study',), 'continuation_stages > 1 has no effect: no design_region sets a '
                                                'projection_beta > 0 (the stages only scale it)'))
    return issues


def _default_stored_fields(spec):
    """'<physics>.design_density' of every physics with an enabled design_region (snapshots keep the design)."""
    return [f'{physics.name}.design_density' for physics in spec.physics
            if any(feature.enabled and feature.type == _DESIGN_FEATURE for feature in physics.features)]


_OPTIMIZATION_PARAMS = (
    Param('max_iterations', 'int', 20, label='Maximum iterations', min=1, max=10000,
          help=f'L-BFGS-B iterations at most; each needs one or more evaluations (line-search trials, at most '
               f'{_EVALUATIONS_PER_ITERATION} x max_iterations in all), and one evaluation = per FDFD physics one '
               'assembly, one solve (recycled factor or a new factorisation) and one transposed (adjoint) solve. '
               'Typical 20-200. With continuation_stages > 1 it holds per stage.',
          tip='Largest number of design updates (per sharpening stage). Each costs one or a few field solves; 20 to '
              '200 is typical.'),
    Param('continuation_stages', 'int', 1, label='Sharpening stages', unit='1', min=1, max=20,
          help='Beta continuation of the design_region tanh projection: L-BFGS-B runs this many times in a row, stage '
               's (1 .. n) with projection_beta x beta_growth^(s - n), so the last stage uses the design_region value '
               'and the first ones optimise a smoother (greyer) problem. 1 = no continuation. Regions with '
               'projection_beta 0 stay unprojected. Typical 3-5 with a final beta 8-32.',
          tip='Number of rounds in which the design is pushed step by step toward empty or filled cells. 1 uses the '
              'projection sharpness of the design regions from the start.'),
    Param('beta_growth', 'float', 2.0, label='Sharpening growth', unit='1', min=1.0, min_exclusive=True,
          help='Factor between the projection betas of consecutive continuation stages (unused at continuation_stages '
               '1); typical 1.5-2.',
          tip='How much sharper each round is than the one before; 2 doubles it. Unused with one stage.'),
    Param('tolerance', 'float', 1e-5, label='Relative tolerance', unit='1', min=0.0,
          help='L-BFGS-B ftol on the scaled objective f = -J / |J0| (J0 = J of the initial design): stop once an '
               'iteration lowers f by at most tolerance * max(|f|, 1); 0 runs all max_iterations. Typical 1e-4 to '
               '1e-6.',
          tip='The optimisation stops when an update improves the objective by less than this fraction. 0 always '
              'runs the maximum number of iterations.'),
    _LIVE_UPDATE_PARAM,
)
_DESCRIPTION = (
    'Density-based topology optimisation over the design_region features of em_fdfd_ez / em_fdfd_hz physics: '
    'maximises J = the sum of weight * quantity of every adjoint_objective of every physics (use weight -1 to '
    'minimise a quantity), each physics solved at its own wavelength setting (several physics of one polarisation '
    'at different wavelengths share the permittivity arrays, so one design region serves a broadband objective). '
    'Each design cell has a density rho in [0, 1], filtered and projected to rho_bar and mapped to eps = eps_min + '
    'rho_bar (eps_max - eps_min) (see design_region). scipy L-BFGS-B with bounds [0, 1] minimises -J / |J0| (J0 = J '
    'of the initial design); every evaluation writes the densities, solves every physics (scipy_splu re-uses the '
    'previous factor as a GMRES preconditioner while the design changes little) and takes dJ/d eps of the '
    'objectives from one transposed solve (eps_sensitivity), then dJ/d rho by the transposed projection and filter. '
    'Stops at max_iterations, at the tolerance or after 5 x max_iterations evaluations, per continuation stage '
    '(continuation_stages > 1: the projection beta grows by beta_growth per stage up to the design_region value; '
    'iterations count on across stages, objective_initial is J at the first stage beta). Snapshots: the initial '
    'design (iteration 0) and every accepted iteration, labelled "iteration k: J = <value> W/m" (em_eme: no unit, '
    'J is a power fraction); with stored_fields '
    '[] they hold every output field, among them "<physics>.design_density" of every physics with a design_region '
    '(the design per iteration), plus "final: J = ..." when the last evaluated state differs; afterwards the physics hold '
    'the returned design (Results.field, design_density). Each design_region result is J and its grayness versus '
    'iteration; summary key optimization {iterations, evaluations, design_cells, objective_initial, objective_final, '
    'continuation_stages, message}; ticks = iterations. Every '
    'physics must be em_fdfd_ez / em_fdfd_hz / em_fdfd_nonlinear / em_eme; couplings and state-dependent material '
    'properties are not supported. '
    'The gradient is exact for the discrete model except on source, port and objective lines (keep design regions '
    'off them); sigma_e, poles and mu_r of the design cells are not optimised. An em_fdfd_nonlinear physics '
    'takes part with its design_region (writing eps_r, eps_xx and eps_yy) and the adjoint_objective features of its '
    'channels: each evaluation solves the channels and their adjoint (undepleted: one transposed solve per channel in '
    'reverse solve order; pump_depleted: the transposed fixed point by GMRES, each product one transposed solve per '
    'channel; no free carriers, and iteration_method newton with a chi3 medium) (inverse-designed harmonic '
    'generation; the chi2 / chi3 tensors and poling stay fixed). An em_eme physics (polarization ez or hz, '
    'section_tolerance 0) takes part with its design_region (ez: eps_r; hz: eps_xx '
    'and eps_yy) and adjoint_objective features (port power fractions, unit 1: the labels and log lines show J '
    'without a unit, the design_region objective series has unit 1): each evaluation re-solves the changed sections '
    'and takes dJ/d eps from the cascade adjoint (one adjoint cascade field, no extra mode solve; exact for n_modes '
    '0).')


def _time_counts(values):
    """Run-time estimate counts (StudyType.time_counts): the iteration limit over every stage and the evaluation limit
    (the initial design + maxfun per stage; each a forward and an adjoint solve, line-search trials included)."""
    iterations, stages = int(values['max_iterations']), int(values['continuation_stages'])
    return {'iterations': iterations * stages, 'evaluations': 1 + stages * _EVALUATIONS_PER_ITERATION * iterations}


def register(registry):
    """Register the 'optimization' StudyType (no background study)."""
    registry.register_study(StudyType(
        STUDY_NAME, 'Optimization',
        summary='Changes the material of the design regions cell by cell to maximise the optimisation objectives of a '
                'frequency-domain wave model, for example the power reaching an output waveguide.',
        description=_DESCRIPTION, params=_OPTIMIZATION_PARAMS, run=_run_optimization,
        validate=_validate_optimization, default_stored_fields=_default_stored_fields, time_counts=_time_counts,
        solver_protocols=(HarmonicAdjointSolver, DesignAdjointSolver)))

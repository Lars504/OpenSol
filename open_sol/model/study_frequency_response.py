"""The 'frequency_response' StudyType (piezoelectric plan stage 5): background physics to a stationary fixed point, then
the driven time-harmonic response of every solver with solve_frequency_response (solid_mechanics, membrane, pressure_acoustics) at each
frequency of a list in Hz, one labelled snapshot per frequency (setting snapshots: all, last or none)."""
# A frequency-domain study (a frequency list in Hz) for the mechanics; the EM frequency_domain study stays
# wavelength-based. Each frequency is one complex factorisation per solved physics (a cuDSS factor reuses its plan).
import time

from .schema import Issue, Param, StudyType
from .solver_hooks import FrequencyResponseSolver, ReducedSweepSolver
from .study import (_PARITY_SECTORS_PARAM, LogEvent, _background_fixed_point_params, _background_stage,
                    _count_progress, _frequency_list, _frequency_range_issues, _scipy_issues, _selected_physics,
                    _solve_static_states, _solver_roles, _study_solved_types, _with_parity_sectors)

STUDY_NAME = 'frequency_response'
_FREQUENCY_STAGE = 'Frequency solves'
_SOLVED_TYPES_TEXT = 'solid_mechanics, solid_mechanics_3d, membrane or pressure_acoustics'   # the frequency-response types (missing-physics remedy)
SNAPSHOT_CHOICES = ('all', 'last', 'none')   # setting snapshots: frequencies that keep a snapshot
SWEEP_METHOD_CHOICES = ('direct', 'adaptive', 'modal')   # setting sweep_method (fem.reduced_harmonic for the others)
_REDUCED_METHODS = (('sweep_method', ('adaptive', 'modal')),)   # visible_if of the reduced-model settings


def frequency_label(frequency):
    """Snapshot label of one frequency: 'f = 1.8e+07 Hz' style (6 significant digits)."""
    return f'f = {frequency:.6g} Hz'


def _run_frequency_response(session, values, emit, control, service):
    """Background stationary solve, background -> solved couplings once, then per frequency solve_frequency_response
    of every solved physics and a snapshot labelled 'f = <Hz> Hz' as the snapshots setting selects."""
    runner = session.runner
    frequencies = _frequency_list(values)
    snapshots = values['snapshots']
    solved, background = _solver_roles(runner.physics, FrequencyResponseSolver, 'frequency response',
                                       f'add a {_SOLVED_TYPES_TEXT} physics', selected=_selected_physics(session))
    stage = _background_stage(
        session, background, solved, values, emit, control, service,
        ('frequency response', 'a frequency response study feeds no harmonic solution back into other physics'),
        f"Frequency response study: {len(frequencies)} frequencies, {min(frequencies):.6g}-{max(frequencies):.6g} Hz; "
        f"solved physics {', '.join(solved)}; background physics {', '.join(background) or 'none'}",
        [_FREQUENCY_STAGE])
    status, n_solved, frequency_sweep = stage.status, 0, None
    if status == 'completed':
        _solve_static_states(runner, solved, False, emit)   # loads are the drive: the initial stress alone
        solvers = {name: runner.physics[name] for name in solved}
        for solver in solvers.values():
            solver.reset_frequency_response()
        emit_stage = stage.stages.stage_emit(_FREQUENCY_STAGE)
        frequency_sweep = _plan_reduced_sweeps(solvers, frequencies, values, emit, emit_stage, runner)
        wall_start = time.perf_counter()
        for index, frequency in enumerate(frequencies):
            if control.checkpoint_cancelled(service):
                status = 'cancelled'
                break
            label = frequency_label(frequency)
            emit_stage(_count_progress(runner, index / len(frequencies), index, wall_start,
                                       f"Solving {', '.join(solved)} at {label} ({index + 1}/{len(frequencies)})",
                                       'frequency'))
            for solver in solvers.values():
                solver.solve_frequency_response(frequency)
            if snapshots == 'all':
                stage.recorder.record(final=index == len(frequencies) - 1, label=label)
            stage.live.update()
            n_solved = index + 1
            if index == 0:
                first_seconds = time.perf_counter() - wall_start
            elif index == 1:   # E7: the first frequency also pays the assembly / factor plan, so extrapolate the second
                second_seconds = time.perf_counter() - wall_start - first_seconds
                emit(LogEvent('info', f'first frequency {first_seconds:.3g} s, second {second_seconds:.3g} s; all '
                                      f'{len(frequencies)} about '
                                      f'{first_seconds + second_seconds * (len(frequencies) - 1):.3g} s'))
        emit_stage(_count_progress(runner, 1.0, n_solved, wall_start, f'{n_solved} frequencies solved', 'frequency'))
        if n_solved:   # A4 / R6: a fields-free last solve (snapshots last / none, or a cancel) is shown now
            for solver in solvers.values():
                if isinstance(solver, ReducedSweepSolver):
                    solver.publish_frequency_response()
        if snapshots == 'last' and n_solved:   # the last solved frequency (also after a cancel) = the live state
            stage.recorder.record(final=True, label=frequency_label(frequencies[n_solved - 1]))
    stage.live.update(final=True)
    return {'status': status, 'ticks': n_solved, 'tick_unit': 'frequency', 't_end': runner.t,
            'fixed_point': stage.fixed_point, 'response_frequencies': frequencies[:n_solved],
            'frequency_sweep': frequency_sweep}


def _plan_reduced_sweeps(solvers, frequencies, values, emit, emit_stage, runner):
    """plan_frequency_response of every solved physics: 'direct' only passes the field policy (R6) -> None; 'adaptive' /
    'modal' build the reduced models (logged) -> summary record {method, physics: {name: info}}."""
    method = values['sweep_method']
    fields = 'all' if values['snapshots'] == 'all' else 'last'   # A4 / R6: fields of stored snapshots only
    if method == 'direct':
        for solver in solvers.values():   # a solver without the hook publishes every frequency
            if isinstance(solver, ReducedSweepSolver):
                solver.plan_frequency_response(frequencies, {'method': method, 'fields': fields})
        return None
    from ..fem.reduced_harmonic import sweep_text
    options = {'method': method, 'tolerance': float(values['rom_tolerance']), 'moments': int(values['rom_moments']),
               'max_points': int(values['rom_max_points']), 'modal_count': int(values['modal_count']),
               'fields': fields}
    records = {}
    wall_start = time.perf_counter()
    for index, (name, solver) in enumerate(solvers.items()):
        if not isinstance(solver, ReducedSweepSolver):
            raise ValueError(f"sweep_method '{method}': physics '{name}' ({type(solver).__name__}) has no reduced "
                             "frequency sweep; use sweep_method 'direct'")
        emit_stage(_count_progress(runner, 0.0, 0, wall_start, f'Building the {method} reduced model of {name} '
                                                               f'({index + 1}/{len(solvers)})', 'frequency'))
        info = solver.plan_frequency_response(frequencies, options)
        records[name] = info
        emit(LogEvent('info', f'{name}: {sweep_text(info)}'))
        for warning in info.get('warnings') or ():
            emit(LogEvent('warning', f'{name} reduced sweep: {warning}'))
    return {'method': method, 'physics': records}


def _validate_frequency_response(vctx, values):
    """Errors without scipy (host SuperLU) and for a frequency range that does not increase (unless a list is given)."""
    # scipy: the reductions need it even with a cuDSS factor
    issues = _scipy_issues(STUDY_NAME, 'host sparse factorisation') + _frequency_range_issues(values)
    issues.extend(_sweep_method_issues(vctx, values['sweep_method']))
    return issues


def _sweep_method_issues(vctx, method):
    """Errors for a sweep_method a solved physics' type does not list in frequency_sweep_methods."""
    if method == 'direct':
        return []
    issues = []
    for physics, physics_type in _study_solved_types(vctx, STUDY_NAME):
        if method not in physics_type.frequency_sweep_methods:
            issues.append(Issue('error', ('study',), f"sweep_method '{method}': physics '{physics.name}' "
                                                     f"({physics_type.label}) supports "
                                                     f"{', '.join(physics_type.frequency_sweep_methods)} only; use "
                                                     "sweep_method 'direct'"))
    return issues


_PARAMS = (
    Param('frequency_start', 'float', 1e6, label='Start frequency', unit='Hz', min=0.0, min_exclusive=True,
          help='Lowest frequency in Hz of the range (used when frequencies is empty).',
          tip='Lowest frequency of the sweep, in hertz.'),
    Param('frequency_stop', 'float', 2e6, label='Stop frequency', unit='Hz', min=0.0, min_exclusive=True,
          help='Highest frequency in Hz of the range; must exceed frequency_start (equal is allowed with '
               'n_frequencies = 1).',
          tip="Highest frequency of the sweep, in hertz. Must be larger than 'Start frequency'."),
    Param('n_frequencies', 'int', 101, label='Number of frequencies', min=1,
          help='Frequencies of the range, both ends included, spaced as log_spacing selects (1 = the arithmetic or '
               'geometric mean). Each costs one complex sparse factorisation per solved physics; resolve a resonance '
               'of quality factor Q at f0 with a step well below f0 / Q.',
          tip='Number of frequencies between the start and stop frequency.\n\nEach costs one sparse solve. A sharp '
              'resonance needs a fine step.'),
    Param('log_spacing', 'bool', False, label='Logarithmic spacing',
          help='false (default): evenly spaced in f (linspace, a resonance sweep); true: evenly in log f (geomspace, a '
               'spectrum over decades). Ignored when frequencies lists any frequency.',
          tip='On: frequencies evenly spaced on a logarithmic scale.\n\nOff: equal steps in hertz.'),
    Param('frequencies', 'float_list', None, label='Frequency list', unit='Hz', min=0.0, min_exclusive=True,
          optional=True,
          help='Explicit frequencies in Hz (each > 0), solved in this order; a non-empty list overrides frequency_start, '
               'frequency_stop and n_frequencies. Feature results (admittance) are sorted by frequency.',
          tip='Frequencies in hertz, solved in this order; when filled, it replaces the range above.'),
    Param('snapshots', 'choice', 'all', label='Stored snapshots', choices=SNAPSHOT_CHOICES,
          choice_labels=(('all', 'Every frequency'), ('last', 'Last frequency only'), ('none', 'None')),
          help="Which frequencies keep a snapshot of the stored fields (the industry-standard store-fields-in-output "
               "choice): 'all' (default) one per frequency, 'last' only the last solved one (also after a cancel), "
               "'none' no snapshot. Each snapshot holds the stored fields on up to 250,000 cells (~1 MB per field for "
               'a large window), so a sweep over hundreds of frequencies may keep only the last. Feature records per '
               'frequency (probes, port / support / mode_projection records, admittance) and the live last-frequency '
               "state are the same for every choice. With 'last' / 'none' the fields are formed for the last solved "
               'frequency only (every sweep_method; the live field view is not refreshed during the sweep).',
          tip='Which frequencies keep a copy of the field plots.\n\n'
              '- Every frequency: one plot per frequency (memory grows with the sweep).\n'
              '- Last frequency only: only the final one.\n'
              '- None: no stored copies.\n\n'
              'Probe curves, tables and the final shown fields are kept in every case. With the last frequency only or '
              'none, the field plot is updated at the end of the sweep only (faster).'),
    Param('sweep_method', 'choice', 'direct', label='Sweep method', choices=SWEEP_METHOD_CHOICES,
          choice_labels=(('direct', 'Direct (one solve per frequency)'), ('adaptive', 'Adaptive reduced model'),
                         ('modal', 'Modal superposition')),
          help="How the frequencies are solved. 'direct' (default): one complex sparse factorisation per frequency. "
               "'adaptive' (asymptotic waveform evaluation / an adaptive fast sweep): a reduced model trained "
               'on the frequency list: factorisations at the ends and the middle, each adding the solution and '
               'rom_moments frequency derivatives (back-solves on the same factor) to a Galerkin basis, then one more '
               'at the frequency of the largest estimated relative residual until it is <= rom_tolerance or '
               'rom_max_points factorisations; every listed frequency is then a small dense solve (cheap: use a fine '
               'list). Pays off in a window with a few to tens of resonances; a band holding hundreds of modes needs '
               "as many basis vectors (keep 'direct'). 'modal': the modal_count eigenmodes nearest the band centre "
               "(the membrane's 'eigen' solver; the solids and pressure_acoustics a host shift-invert of K, M) plus the "
               'direct solution at the band centre (static correction of '
               'the truncated modes); exact for a uniform loss_factor once every mode is in the basis, approximate '
               'otherwise. Both end with one verification factorisation at the worst frequency; summary '
               'frequency_sweep {method, physics: {name: {basis_size, factorizations, expansion_frequencies [Hz], '
               'max_residual, worst_frequency [Hz], verified_error (true relative error there), converged, warnings, '
               '...}}}. With a reduced sweep, probe / projection / reaction / electrode records come from the reduced '
               "coordinates and the fields are built only for the stored snapshots (snapshots 'all': every frequency, "
               'else the last). Types: membrane (not with ports), solid_mechanics and solid_mechanics_3d (modal: not '
               'piezoelectric), pressure_acoustics (a background_pressure_field or a damping boundary: adaptive only, '
               'one snapshot per expansion, no derivatives).',
          tip='Direct: one full solve per frequency.\n\n'
              '- Adaptive reduced model: a few full solves build a small model that gives every frequency of the list '
              'quickly; best for a window holding a few to tens of resonances.\n'
              '- Modal superposition: built from the resonances nearest the middle of the window.\n\n'
              'Not with ports. With a reduced model, field plots are made only for the stored snapshots; probe '
              'curves and tables are kept for every frequency.'),
    Param('rom_tolerance', 'float', 1e-6, label='Reduced model tolerance', min=0.0, min_exclusive=True,
          visible_if=_REDUCED_METHODS,
          help="Largest estimated relative residual ||b - A x|| / ||b|| over the frequency list ('adaptive': training "
               "stops below it; 'modal': a warning above it). Near a resonance of quality factor Q the relative error "
               'can be up to about Q times the residual, so for Q = 1e4 peaks use 1e-8 or below; the summary reports '
               'the true error at the worst frequency.',
          tip='Accuracy target of the reduced model; smaller is more accurate but takes more full solves.'),
    Param('rom_moments', 'int', 4, label='Derivatives per expansion', min=0, max=20,
          visible_if=(('sweep_method', ('adaptive',)),),
          help='Frequency derivatives of the solution added per expansion frequency (adaptive): back-solves on the '
               'factor already made, so each factorisation adds rom_moments + 1 basis vectors.',
          tip='How many extra directions each full solve adds; 4 is typical.'),
    Param('rom_max_points', 'int', 40, label='Most expansion frequencies', min=1,
          visible_if=(('sweep_method', ('adaptive',)),),
          help='Largest number of expansion frequencies (factorisations, not counting the verification) of the '
               'adaptive model; reaching it without rom_tolerance logs a warning.',
          tip='Upper limit on the full solves the adaptive model may use.'),
    Param('modal_count', 'int', 20, label='Number of modes', min=1,
          visible_if=(('sweep_method', ('modal',)),),
          help="Eigenmodes nearest the band centre (frequency list's (min + max) / 2) in the modal basis; take every "
               'resonance of the band plus a margin on either side.',
          tip='How many resonances near the middle of the window build the modal model.'),
    *_background_fixed_point_params('the frequency solves', 'the frequency solves'),
    _PARITY_SECTORS_PARAM,
)
_DESCRIPTION = (
    'Driven time-harmonic response (exp(+j omega t), omega = 2 pi f) at a list of frequencies in Hz, the '
    'frequency-domain study of the mechanics (the EM frequency_domain study takes wavelengths). Solved physics: every '
    f'type listing frequency_response among its supported studies ({_SOLVED_TYPES_TEXT}; solid_mechanics, 2D: planar '
    'plane strain / plane '
    'stress or axisymmetric; not generalized plane strain or guided modes; solid_mechanics_3d on a tetrahedron mesh; '
    'pressure_acoustics: -div(grad p / rho) - omega^2 p / (rho c^2) = j omega q on triangles / tetrahedra, its '
    'frequency_probe / surface_pressure records): '
    '(K (1 + j eta) - omega^2 M) u = f with '
    'the consistent mass of rho (every solid cell needs rho > 0), the loss factor mechanical_loss_factor eta and, '
    'piezoelectric, the coupled potential with eps (1 - j dielectric_loss_tangent); every load, prescribed '
    'displacement, electrode voltage and floating charge is the complex amplitude (phase 0) of the drive; one complex '
    'symmetric factor per frequency of the pencil reduced once to the free dofs (2D: host SuperLU outside the solver '
    "roles, counters in the summary; solid_mechanics_3d: its 'linear' sparse_direct solver, host SuperLU or cuDSS "
    'LDL^T with the plan reused across frequencies). Results: the admittance feature '
    '(Y = I / V of an electrode versus f with a Butterworth-Van Dyke fit), one snapshot per frequency of the stored '
    'fields (setting snapshots: all, last = the last solved frequency only, none; feature records per frequency are '
    'kept either way) labelled "f = <Hz> Hz" holding the real parts (phase 0) of the response, u_abs = the amplitude |u| of the '
    'complex displacement, u_x_im / u_y_im (/ u_z_im) the imaginary parts (triangle and tetrahedron meshes; store them '
    'through stored_fields); the last frequency stays live. On triangle and tetrahedron meshes the frequency_probe '
    'feature records the complex displacement at a point per frequency (series u_x_re, u_x_im, u_x_abs, u_x_phase, '
    '...) and every support result becomes a table of its complex reaction per frequency (F_x_re, F_x_im, ...); '
    'solid_probe samples and electrode tables are not filled. With the stress_stiffening setting the initial_stress '
    'is first relaxed by a static solve (loads stay the drive, not a preload) and its geometric stiffness K_sigma '
    'joins K (real, no loss). A free body (no supports) is allowed: K - omega^2 M is regular at f > 0. Every other '
    'physics must support the stationary study and runs as a background physics first (its couplings into the solved '
    'physics applied once); couplings reading a solved physics are a validation error; StudySpec.solved_physics '
    'restricts the solved physics to those listed (the others run as background physics). membrane (triangles, 1D): '
    "the pressure_load / line_force / point_force phasors, one complex factor per frequency on its 'linear' "
    'sparse_direct solver, field_probe, port and mode_projection records per frequency. sweep_method adaptive / modal '
    '(membrane, the solids, pressure_acoustics) replaces the factorisation per frequency by a reduced model built '
    'first (see the setting). ticks = '
    'frequencies; summary response_frequencies (+ frequency_sweep with a reduced sweep). parity_sectors runs a half '
    '(quarter) model once per mirror parity and combines the drawn drive (see the setting; membrane symmetry / '
    'antisymmetry faces are mirror planes).')


def _time_counts(values):
    """Run-time estimate counts (StudyType.time_counts): frequencies, driven factorisations (a reduced sweep: at most
    rom_max_points + 1 / two), the background iteration limit."""
    frequencies = values.get('frequencies') or []
    points = len(frequencies) if frequencies else int(values['n_frequencies'])
    method = values.get('sweep_method', 'direct')
    factorizations = {'direct': points, 'adaptive': min(points, int(values.get('rom_max_points', 40))) + 1,
                      'modal': 2}.get(method, points)
    return {'harmonic_points': points, 'harmonic_factorizations': factorizations,
            'fixed_point_limit': int(values['max_iterations'])}


def register(registry):
    """Register the 'frequency_response' StudyType (background study 'stationary')."""
    registry.register_study(StudyType(
        STUDY_NAME, 'Frequency response',
        summary='Solves the steady vibration of a structure or membrane driven at a list of frequencies, for example the '
                'electrical admittance of a piezoelectric resonator. Any heat or current physics is first solved to a '
                'steady state.',
        description=_DESCRIPTION, params=_PARAMS, background_study='stationary', run=_run_frequency_response,
        validate=_with_parity_sectors(_validate_frequency_response),
        feedback_alternative='the frequency response study feeds no harmonic solution back into other physics',
        time_counts=_time_counts, solver_protocols=(FrequencyResponseSolver,)))

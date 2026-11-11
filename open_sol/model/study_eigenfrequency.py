"""The 'eigenfrequency' StudyType (A6): background physics to a stationary fixed point, then the source-free resonances
of every solver with solve_eigenfrequencies (em_fdfd_ez / _hz / _bor, solid_mechanics(_3d), membrane), one labelled
snapshot per
resonance number."""
import math

from .schema import Issue, Param, StudyType
from .solver_hooks import EigenfrequencySolver, MultiShiftEigenSolver
from .study import (_PARITY_SECTORS_PARAM, _background_fixed_point_params, _background_stage, _log_eigen_solve,
                    _numbered_snapshots, _reselect_first_resonance, _scipy_issues, _selected_physics,
                    _solve_static_states, _solver_roles, _study_solved_types, _with_parity_sectors)

STUDY_NAME = 'eigenfrequency'
_RESONANCE_STAGE = 'Resonance solve'
# the eigen-solved types (missing-physics remedy and texts)
_SOLVED_TYPES_TEXT = ('em_fdfd_ez, em_fdfd_hz, em_fdfd_bor, em_fdfd_conical, em_fdfd_3d, solid_mechanics, '
                      'solid_mechanics_3d or membrane')
SEARCH_CHOICES = ('nearest', 'targets', 'band')   # setting search (PhysicsType.eigen_searches lists a type's)
_TARGETS_SEARCH = (('search', ('targets',)),)     # visible_if
_BAND_SEARCH = (('search', ('band',)),)
# run-time estimate of a band: shifts ~ resonances per slice of half-overlapping trusted windows (B1 march), guess
_BAND_SHIFTS_PER_SLICE = 2.0
_BAND_COUNT_FACTORIZATIONS = 2      # inertia counts at the band edges (symmetric pencils)


def _search_text(values) -> str:
    """Log text of the search: nearest the target setting, the listed targets, or the band."""
    search = values.get('search', 'nearest')
    if search == 'targets':
        targets = ', '.join(f'{target:.6g}' for target in values['target_frequencies'] or ())
        return f"up to {values['n_resonances']} resonance(s) nearest each of f = {targets} Hz"
    if search == 'band':
        return (f"every resonance in f = {values['frequency_min']:.6g} - {values['frequency_max']:.6g} Hz (at most "
                f"{values['max_resonances']}, {values['n_resonances']} per shift)")
    return f"up to {values['n_resonances']} resonance(s) nearest the target (wavelength or target_frequency setting)"


def _run_eigenfrequency(session, values, emit, control, service):
    """Background physics to a stationary fixed point, background -> eigen-solved couplings once, the static states of
    the prestressed solvers, the resonances of every eigen-solved physics, then one labelled snapshot per resonance
    number; resonance 1 stays shown."""
    runner = session.runner
    solved, background = _solver_roles(runner.physics, EigenfrequencySolver, 'eigenfrequency',
                                       f'add an {_SOLVED_TYPES_TEXT} physics', selected=_selected_physics(session))
    stage = _background_stage(
        session, background, solved, values, emit, control, service,
        ('eigenfrequency', 'an eigenfrequency study feeds no resonance field back into other physics (use a '
                           'stationary study with a source for that)'),
        f"Eigenfrequency study: {_search_text(values)} of {', '.join(solved)}; background physics "
        f"{', '.join(background) or 'none'}",
        [_RESONANCE_STAGE])
    status, n_found = stage.status, 0
    if status == 'completed':
        _solve_static_states(runner, solved, True, emit)
        status, n_found = _solve_resonance_snapshots(session, solved, values, stage.recorder, stage.live,
                                                     stage.stages.stage_emit(_RESONANCE_STAGE), control, service)
    stage.live.update(final=True)
    return {'status': status, 'ticks': n_found, 'tick_unit': 'resonance', 't_end': runner.t,
            'fixed_point': stage.fixed_point, 'n_resonances': n_found}


def _search_solve(name, solver, values):
    """The eigen solve of one physics as the search setting asks: solve_eigenfrequencies (nearest), or the
    MultiShiftEigenSolver hooks (targets / band; ValueError for a solver without them)."""
    search = values.get('search', 'nearest')
    n_resonances = values['n_resonances']
    if search == 'nearest':
        return solver.solve_eigenfrequencies(n_resonances)
    if not isinstance(solver, MultiShiftEigenSolver):
        raise ValueError(f"eigenfrequency search '{search}': physics '{name}' ({type(solver).__name__}) solves the "
                         "resonances nearest its target only; use search 'nearest'")
    if search == 'targets':
        return solver.solve_eigenfrequencies_targets(n_resonances, [float(target) for target in
                                                                    values['target_frequencies']])
    return solver.solve_eigenfrequencies_band(float(values['frequency_min']), float(values['frequency_max']),
                                              n_resonances, int(values['max_resonances']))


def _solve_resonance_snapshots(session, solved, values, recorder, live, emit, control, service):
    """study._numbered_snapshots of the resonances: the search's eigen solve of every eigen-solved physics (nearest:
    nearest the target first; targets / band: frequency order), one snapshot per resonance number labelled 'resonance
    k: lambda = ... nm, Q = ...', resonance 1 selected again -> (status, largest count)."""
    def solve(name, solver):
        eigen_solve = _search_solve(name, solver, values)
        _log_eigen_solve(name, eigen_solve, emit)
        return [resonance.text() for resonance in eigen_solve.resonances]

    return _numbered_snapshots(session, solved, solve, lambda solver, number: solver.select_resonance(number),
                               _reselect_first_resonance, 'resonance', 'eigen solve', recorder, live, emit, control, service)


def _validate_eigenfrequency(vctx, values):
    """Error when scipy (the host shift-invert eigensolver) is missing (find_spec only, nothing imported); search
    checks (_search_issues)."""
    return _scipy_issues(STUDY_NAME, 'host shift-invert eigensolver') + _search_issues(vctx, values)


def _search_issues(vctx, values):
    """Errors for an empty target list, a band whose end does not exceed its start, and a search a solved physics'
    type does not list in eigen_searches (at that physics)."""
    search = values.get('search', 'nearest')
    if search == 'nearest':
        return []
    issues = []
    if search == 'targets' and not values.get('target_frequencies'):
        issues.append(Issue('error', ('study',), "search 'targets' needs at least one frequency in target_frequencies"))
    if search == 'band' and not values['frequency_max'] > values['frequency_min']:
        issues.append(Issue('error', ('study',), f"search 'band': frequency_max {values['frequency_max']:.6g} Hz must "
                                                 f"exceed frequency_min {values['frequency_min']:.6g} Hz"))
    for physics, physics_type in _study_solved_types(vctx, STUDY_NAME):
        if search not in physics_type.eigen_searches:
            issues.append(Issue('error', ('physics', physics.name),
                                f"eigenfrequency search '{search}': physics '{physics.name}' ({physics_type.label}) "
                                f"supports {', '.join(physics_type.eigen_searches)} only (one shift-invert about its "
                                "own target); use search 'nearest' or leave it out of solved_physics"))
    return issues


_EIGENFREQUENCY_PARAMS = (
    Param('n_resonances', 'int', 6, label='Number of resonances', min=1, max=100,
          help='Resonances kept per eigen-solved physics: the shift-invert eigen-solve finds the 2 n_resonances + 10 '
               'eigenvalues k^2 nearest the target (k0 of the wavelength setting)^2, drops those with more than half '
               'of their |field|^2 dA in PML cells (and static / non-propagating ones), ranks the rest by |omega - '
               'omega_target| and keeps up to n_resonances (fewer may come back, with a run-log warning). With PML '
               'many candidates are PML resonances, so ask for a few more than needed. solid_mechanics (plane strain or '
               'guided) and solid_mechanics_3d on tetrahedra: the n_resonances + 4 eigenvalues omega^2 nearest (2 pi '
               'target_frequency)^2 (a free body: + its rigid motions, whose 0 Hz modes are dropped; with a loss factor '
               'or an elastic PML 2 n_resonances + 4 complex ones), ranked by |omega - omega_target|; solid_mechanics_3d '
               'on a tensor mesh: the n_resonances + 4 lowest (each Lanczos step one PCG solve). membrane: the '
               'n_resonances + 4 (real) or 2 n_resonances + 4 (loss_factor or PML: complex) eigenvalues nearest (2 pi '
               'target_frequency)^2, PML modes dropped. 1-10 typical. search targets: kept per target; search band: '
               'the resonances per shift of the march through the band (more per shift = fewer, larger shifts).',
          tip='How many resonances to find, starting with the one closest to the wavelength set in the wave '
              'physics.\n\nFewer may be found. 1 to 10 is typical. With several targets: per target; with a band: '
              'per step through the band.'),
    Param('search', 'choice', 'nearest', label='Search', choices=SEARCH_CHOICES,
          choice_labels=(('nearest', 'Nearest the target of each physics'), ('targets', 'Nearest several targets'),
                         ('band', 'Every resonance in a band')),
          help="Which resonances are solved. 'nearest' (default): n_resonances nearest each solved physics' own target "
               "(wavelength or target_frequency setting). 'targets': the union of the n_resonances nearest each "
               'frequency of target_frequencies (overriding target_frequency), from one build and assembly: one '
               'shift-invert factorisation per target (a cuDSS factor keeps its plan across targets), duplicates of '
               "two targets merged. 'band': every resonance with frequency_min <= Re f <= frequency_max (at most "
               'max_resonances, the lowest): shifts march upward, each trusting the eigenvalues out to its farthest '
               'one; a real symmetric pencil (an undamped solid without Bloch phase, a lossless membrane without PML) '
               'counts the modes per interval by Sylvester inertia (the summary gives the count, a short interval '
               'gets a repair shift, a mismatch a warning); damped, PML, loaded or complex Bloch pencils march with '
               'half-overlapping shifts and no count (warning: modes may be missing). targets and band list the '
               'resonances in frequency order (resonance 1 = the lowest). Types: solid_mechanics (piezoelectric: '
               'targets only), solid_mechanics_3d on tetrahedra (piezoelectric: targets only), membrane (the PML '
               'strength at pml_frequency, else the mean target / the band centre); the EM types and '
               'solid_mechanics_3d on a tensor mesh take nearest only.',
          tip='Which resonances to find.\n\n'
              '- Nearest the target of each physics: the resonances closest to the target set in the physics.\n'
              '- Nearest several targets: the resonances closest to each frequency of a list, from one build.\n'
              '- Every resonance in a band: all resonances between two frequencies, checked by a count where '
              'possible.\n\n'
              'Solid mechanics and membranes only for the last two.'),
    Param('target_frequencies', 'float_list', None, label='Target frequencies', unit='Hz', min=0.0, optional=True,
          visible_if=_TARGETS_SEARCH,
          help="search 'targets': the target frequencies in Hz (>= 0; 0 = the lowest modes of a supported solid; a "
               'damped, PML or loaded pencil needs > 0); n_resonances are kept nearest each, the union in frequency '
               'order. Overrides the target_frequency setting of every solved physics.',
          tip='Frequencies in hertz to search near; the resonances closest to each are found.'),
    Param('frequency_min', 'float', 0.0, label='Band start', unit='Hz', min=0.0, visible_if=_BAND_SEARCH,
          help="search 'band': lower band edge in Hz (>= 0; resonances with Re f >= it are kept). A free body's 0 Hz "
               'rigid-body modes are dropped.',
          tip='Lowest frequency of the band, in hertz.'),
    Param('frequency_max', 'float', 1e6, label='Band end', unit='Hz', min=0.0, min_exclusive=True,
          visible_if=_BAND_SEARCH,
          help="search 'band': upper band edge in Hz; must exceed frequency_min.",
          tip="Highest frequency of the band, in hertz. Must be larger than 'Band start'."),
    Param('max_resonances', 'int', 200, label='Most resonances in the band', min=1, visible_if=_BAND_SEARCH,
          help="search 'band': at most this many resonances (the lowest of the band); the march stops once reached "
               '(warning). Bounds the run time and the stored mode shapes (one host vector each).',
          tip='Upper limit on the resonances found in the band; the lowest ones are kept.'),
    *_background_fixed_point_params('the eigen-solve', 'the resonances are computed'),
    _PARITY_SECTORS_PARAM,
)
_DESCRIPTION = (
    'Source-free resonances (eigenfrequencies) of every physics listing eigenfrequency among its supported studies '
    f'({_SOLVED_TYPES_TEXT}); the EM types are searched near the target = that physics\' wavelength setting lambda_t '
    '(omega_t = 2 pi C0 '
    '/ lambda_t). The FDFD operator is assembled at omega_t and the generalised eigenproblem L u = -(omega / C0)^2 M u '
    '(M = eps cells for E_z, mu cells for H_z) is solved on the host by shift-invert Arnoldi about omega_t (scipy eigs '
    'on a SuperLU factor: the eigen solver role, arpack_shift_invert, whatever the linear back end, so its size limit '
    'of ~1e6 unknowns applies). Accuracy limit: the '
    'material dispersion (sigma_e, Drude / Lorentz poles) and the PML stretch depend on omega but are evaluated at '
    'omega_t only, so the problem is linear in omega^2 by freezing them: a resonance far from the target carries an '
    'error of first order in |omega / omega_t - 1| (exact only for non-dispersive, PML-free models); for an accurate '
    'value set the wavelength setting near the resonance and re-run (a dispersive or open resonator converges in one '
    'or two such steps). The shift-invert shift sits 1e-4 above the target, so a target on a resonance still '
    'factors. Results: complex omega with '
    'Im > 0 for a decaying resonance (exp(+j omega t)), Q = Re(omega) / (2 Im(omega)) (radiation into the PML plus '
    'material loss), the resonance_table feature, and one snapshot per resonance number k of the stored fields '
    '(default every output field) labelled "resonance k: lambda = <nm> nm, Q = <value>" (several '
    'physics: their names and values joined; mirror faces add the parity, "resonance k [x_lo symmetric]: ..."), '
    'nearest the target first; each field is scaled to 1 (V/m for E_z, A/m '
    'for H_z) real at its peak, and its in-plane fields come from the curl at omega_t. Afterwards each physics shows '
    'resonance 1 (Results.field); ticks = the number of resonance snapshots. Sources and monitors play no part (probe, '
    'flux and mode results stay empty). Every other physics must support the stationary study and runs as a '
    'background physics first (heat is always steady here), iterated to a stationary fixed point, then every coupling '
    'from background to eigen-solved physics (heat_em_thermo_optic, field_transfer, state-dependent material '
    'properties) is applied once; couplings or material properties reading an eigen-solved physics are a validation '
    'error. StudySpec.solved_physics (a list of physics names) restricts the eigen-solved physics to those listed: '
    'the others, also of an eigen-capable type, then run as stationary background physics and may feed the listed '
    'ones through couplings (e.g. a solid relaxed to its static state before the modes of another physics). For Q or '
    'the resonance wavelength versus a parameter use a parameter sweep with a derived value of the '
    'resonance_table result. solid_mechanics (2D plane strain): the undamped vibration modes K u = omega^2 M u '
    '(consistent mass of rho) nearest its target_frequency setting (0 = the lowest), host shift-invert Lanczos (scipy '
    'eigsh on a real SuperLU factor: the eigen solver role, arpack_shift_invert), exact for the discretisation (no '
    'frozen terms), Q infinite; with its axial_wavenumber setting k_z the guided (2.5D) modes u ~ exp(i (k_z z - omega '
    't)) of the cross-section instead (u_z coupled; sweep a parameter for the dispersion f(k_z)); labels "resonance k: '
    'f = <Hz> Hz", mode shapes scaled to a peak node |u| of 1 m; loads and thermal strain play no part unless the '
    'stress_stiffening setting is on (triangle and tetrahedron meshes: the static state of every load, initial_stress '
    'and thermal strain is solved first and its geometric stiffness K_sigma joins K; a free body, Bloch cell or elastic '
    'PML takes the initial_stress alone). On triangle and tetrahedron meshes a mechanical_loss_factor eta makes the '
    'modes complex: K_el (1 + j eta) (K_sigma stays real), Q = Re omega / (2 Im omega) in the resonance table. '
    'solid_mechanics_3d: on a tensor mesh the n_resonances lowest modes only (no target), by matrix-free '
    "shift-invert Lanczos at 0 whose steps are PCG solves of its double 'linear' solver (no host factor; accuracy set "
    'by the CG tolerance); on tetrahedra the modes nearest target_frequency, with target_frequency > 0 also of a free '
    'body without supports (its rigid-body modes at 0 Hz are dropped). '
    'em_fdfd_bor (body of revolution): the same shift-invert search near its wavelength setting at its azimuthal '
    'order m, or n_resonances per order of its azimuthal_orders list from one assembly (resonance numbers run order '
    'by order, labels "resonance k: m = <m>, lambda = ..."), on the full-vector (r, z) curl-curl operator (E_r, E_phi, '
    'E_z; M = eps ring volumes); static gradient solutions (|k^2| < 1e-6 k_t^2) are dropped; each resonance is scaled '
    'to 1 V/m real at its largest component. em_fdfd_conical: the same search at its out-of-plane wave number k_z on '
    'the three-component curl-curl operator (E_x, E_y, E_z; M = eps of the E points), static solutions dropped as for '
    'em_fdfd_bor, each resonance scaled to 1 V/m real at its largest E sample; at k_z = beta the resonances are the '
    'guided modes of propagation constant beta (a k_z sweep gives the dispersion). em_fdfd_3d: the same search on the '
    '3D curl-curl operator of the live Yee E faces (M = the face eps; squared Cayley map as em_fdfd_bor, static '
    'gradient solutions dropped), each resonance scaled to 1 V/m real at its largest E face; snapshots are its z cut '
    'planes; its host factor grows ~ n^(4/3) (size rule 1e6 / 4e6 E unknowns). membrane (triangles, 1D): resonances '
    'of -div(T (1 + j eta) grad w) = rho_s omega^2 w nearest its target_frequency (fem.harmonic_resonances on the '
    'eigen role; complex with loss_factor / PML: Q = Re omega / (2 Im omega)), labels "resonance k: f = <Hz> Hz[, Q = '
    '...]", scaled to 1 m real at the peak; clamps held at 0, loads ignored. Setting search: nearest (the above), '
    'targets (several targets from one build: one shift each, the union in frequency order) or band (every resonance '
    'in a frequency band, inertia-counted on real symmetric pencils) for solid_mechanics, solid_mechanics_3d on '
    'tetrahedra and membrane; the resonance_table then has target_frequency NaN and carries targets / band / '
    'expected_count.')


def _time_counts(values):
    """Run-time estimate counts (StudyType.time_counts): eigenpairs per shift, shift solves (eigen_shifts) and inertia
    count factorisations (band), resonances kept (a band: max_resonances, an upper bound), the background iteration
    limit (definition_helpers.eigen_search_counts reads them)."""
    n_resonances = int(values['n_resonances'])
    counts = {'eigenpairs': n_resonances, 'eigen_shifts': 1, 'resonances': n_resonances,
              'fixed_point_limit': int(values['max_iterations'])}
    search = values.get('search', 'nearest')
    if search == 'targets':
        shifts = max(len(values.get('target_frequencies') or ()), 1)
        counts.update(eigen_shifts=shifts, resonances=n_resonances * shifts)
    elif search == 'band':
        bound = int(values['max_resonances'])
        counts.update(eigen_shifts=int(math.ceil(_BAND_SHIFTS_PER_SLICE * bound / n_resonances)),
                      eigen_count_factorizations=_BAND_COUNT_FACTORIZATIONS, eigen_band=1, resonances=bound)
    return counts


def register(registry):
    """Register the 'eigenfrequency' StudyType (background study 'stationary')."""
    registry.register_study(StudyType(
        STUDY_NAME, 'Eigenfrequency',
        summary='Finds the resonances of a frequency-domain wave model near the wavelength set in the wave physics: '
                'resonance wavelength, quality factor and field shape. Heat or current physics are first solved to '
                'a steady state. Accurate only near that wavelength. For solid mechanics and membranes it finds the vibration '
                'frequencies and mode shapes.',
        description=_DESCRIPTION,
        params=_EIGENFREQUENCY_PARAMS, background_study='stationary', run=_run_eigenfrequency,
        validate=_with_parity_sectors(_validate_eigenfrequency), supports_adaptation=True,
        feedback_alternative='use a stationary study with a source to couple the FDFD field into other physics',
        time_counts=_time_counts, solver_protocols=(EigenfrequencySolver,)))

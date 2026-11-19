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
          help='Resonances kept per eigen-solved physics (fewer may come back, with a warning; 1-10 typical; '
               'targets: per target; band: per shift of the march, more = fewer, larger shifts). EM: the 2 n + 10 '
               'eigenvalues k^2 nearest k_t^2 are found, those with over half their |field|^2 in PML cells (and '
               'static ones) dropped, the rest ranked by |omega - omega_t|: with PML ask for a few more. '
               'solid_mechanics, solid_mechanics_3d on tetrahedra and membrane: the n + 4 eigenvalues omega^2 nearest '
               '(2 pi target_frequency)^2 (2 n + 4 complex ones with a loss factor or PML; a free solid adds its '
               'rigid motions, dropped); solid_mechanics_3d on a tensor mesh: the n + 4 lowest.',
          tip='How many resonances to find, starting with the one closest to the wavelength set in the wave '
              'physics.\n\nFewer may be found. 1 to 10 is typical. With several targets: per target; with a band: '
              'per step through the band.'),
    Param('search', 'choice', 'nearest', label='Search', choices=SEARCH_CHOICES,
          choice_labels=(('nearest', 'Nearest the target of each physics'), ('targets', 'Nearest several targets'),
                         ('band', 'Every resonance in a band')),
          help="'nearest': n_resonances nearest each solved physics' own target (wavelength or target_frequency "
               "setting). 'targets': the union of the n_resonances nearest each of target_frequencies, from one "
               'build (one shift-invert factorisation per target; duplicates merged). '
               "'band': every resonance with frequency_min <= Re f <= frequency_max (at most max_resonances, the "
               'lowest): shifts march upward; a real symmetric pencil (an undamped solid without Bloch phase, a '
               'lossless membrane without PML) is counted per interval by Sylvester inertia (the summary gives the '
               'count, a short interval gets a repair shift, a mismatch warns); damped, PML, loaded or complex Bloch '
               'pencils march with half-overlapping shifts and no count (warning: modes may be missing). targets and '
               "band list in frequency order and leave the resonance_table's target_frequency NaN (it carries "
               'targets / band / expected_count). They serve solid_mechanics and solid_mechanics_3d on tetrahedra '
               '(piezoelectric: targets only) and membrane (PML strength at pml_frequency, else the mean target / '
               'the band centre); the other types take nearest only.',
          tip='Which resonances to find.\n\n'
              '- Nearest the target of each physics: the resonances closest to the target set in the physics.\n'
              '- Nearest several targets: the resonances closest to each frequency of a list, from one build.\n'
              '- Every resonance in a band: all resonances between two frequencies, checked by a count where '
              'possible.\n\n'
              'Solid mechanics and membranes only for the last two.'),
    Param('target_frequencies', 'float_list', None, label='Target frequencies', unit='Hz', min=0.0, optional=True,
          visible_if=_TARGETS_SEARCH,
          help="0 = the lowest modes of a supported solid; a damped, PML or loaded pencil needs > 0. Overrides the "
               'target_frequency setting of every solved physics.',
          tip='Frequencies in hertz to search near; the resonances closest to each are found.'),
    Param('frequency_min', 'float', 0.0, label='Band start', unit='Hz', min=0.0, visible_if=_BAND_SEARCH,
          help="Resonances with Re f >= it are kept; a free body's 0 Hz rigid-body modes are dropped.",
          tip='Lowest frequency of the band, in hertz.'),
    Param('frequency_max', 'float', 1e6, label='Band end', unit='Hz', min=0.0, min_exclusive=True,
          visible_if=_BAND_SEARCH,
          help='Must exceed frequency_min.',
          tip="Highest frequency of the band, in hertz. Must be larger than 'Band start'."),
    Param('max_resonances', 'int', 200, label='Most resonances in the band', min=1, visible_if=_BAND_SEARCH,
          help='The march stops once reached (warning). Bounds the run time and the stored mode shapes (one host '
               'vector each).',
          tip='Upper limit on the resonances found in the band; the lowest ones are kept.'),
    *_background_fixed_point_params('the resonances are computed'),
    _PARITY_SECTORS_PARAM,
)
_DESCRIPTION = (
    f'Source-free resonances of every eigen-capable physics ({_SOLVED_TYPES_TEXT}; StudySpec.solved_physics '
    'restricts them, the others then run as stationary background physics). Background physics iterate to a '
    'stationary fixed point, then their couplings into the eigen-solved physics apply once; couplings or material '
    'properties reading an eigen-solved physics: validation error. Results: complex omega, Im > 0 decaying (exp(+j '
    'omega t)), Q = Re(omega) / (2 Im(omega)) (radiation into the PML plus material loss), the resonance_table '
    'feature, one snapshot per resonance k (several physics: names and values joined; mirror faces add the parity), '
    'nearest the target first; afterwards each physics shows resonance 1; ticks = the resonance snapshots. Sources '
    'and monitors play no part (their results stay empty). Q or the wavelength versus a parameter: a sweep with a '
    'derived value of resonance_table. em_fdfd_ez / _hz: near omega_t = 2 pi C0 / lambda_t of the wavelength '
    'setting, L u = -(omega / C0)^2 M u (M = eps cells for E_z, mu cells for H_z) by host shift-invert Arnoldi '
    '(scipy eigs on SuperLU, the eigen role arpack_shift_invert whatever the linear back end: ~1e6 unknowns at most; '
    'the shift sits 1e-4 above the target). Accuracy: dispersion (sigma_e, poles) and the PML stretch are frozen at '
    'omega_t, so a resonance far from the target errs to first order in |omega / omega_t - 1| (exact only without '
    'dispersion and PML): set the wavelength near the resonance and re-run (one or two steps converge). Each field '
    'is scaled to 1 (V/m E_z, A/m H_z) real at its peak, in-plane fields from the curl at omega_t. em_fdfd_bor: at '
    'its azimuthal order m, or n_resonances per order of azimuthal_orders from one assembly (labels "resonance k: m = '
    '<m>, lambda = ..."), on the full-vector (r, z) curl-curl operator (M = eps ring volumes). em_fdfd_conical: at '
    'its k_z on the three-component operator; at k_z = beta the resonances are the guided modes of propagation '
    'constant beta. em_fdfd_3d: the 3D Yee curl-curl operator (M = face eps), snapshots its z cut planes, host '
    'factor ~ n^(4/3) (size rule 1e6 / 4e6 E unknowns). These three drop static gradient solutions (|k^2| < 1e-6 '
    'k_t^2) and scale each resonance to 1 V/m real at its largest component. solid_mechanics (plane strain): '
    'undamped K u = omega^2 M u (consistent mass) nearest target_frequency (0 = the lowest) by host shift-invert '
    'Lanczos, exact for the discretisation, Q infinite; with axial_wavenumber k_z the guided (2.5D) modes u ~ exp(i '
    '(k_z z - omega t)); labels "resonance k: f = <Hz> Hz", shapes scaled to a peak |u| of 1 m; loads and thermal '
    'strain play no part unless stress_stiffening (triangles, tetrahedra: the geometric stiffness K_sigma of the '
    'static state joins K; a free body, Bloch cell or elastic PML takes initial_stress alone). Triangles / '
    'tetrahedra: a mechanical_loss_factor eta makes K_el (1 + j eta) complex (K_sigma stays real). '
    'solid_mechanics_3d: tensor mesh the n_resonances lowest only (matrix-free shift-invert Lanczos at 0 on PCG solves '
    "of its double 'linear' solver, accuracy by the CG tolerance); tetrahedra nearest target_frequency (> 0 also for a "
    'free body, its 0 Hz modes dropped). membrane: -div(T (1 + j eta) grad w) = rho_s omega^2 w nearest '
    'target_frequency (complex with loss_factor / PML), labels "resonance k: f = <Hz> Hz[, Q = ...]", scaled to 1 m '
    'real at the peak; clamps held at 0, loads ignored.')


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

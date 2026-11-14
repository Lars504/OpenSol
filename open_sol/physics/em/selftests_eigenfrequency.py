"""Model-run self-test of the eigenfrequency study on em_fdfd_ez / em_fdfd_hz (PEC rectangular cavity: discrete
closed-form resonances, lossless and with a uniform sigma_e, resonance field shape; also on a line_block linear node)
and the 'disk_resonator_eigenmodes' template; no physics imports."""
import cmath
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, solver_role
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, SolverSpec, StudySpec)
from .selftests import tolerance_units_result
from .selftests_fdfd import cpu_specs

_PHYSICS = 'em'
_TABLE = 'resonances'
_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
_OMEGA_COLUMNS = (2, 3)     # frequency, frequency_imag [Hz] in the resonance_table rows
_Q_COLUMN = 4

# ---- PEC rectangular cavity (vacuum), uniform 40 nm cells (the closed form is exact for any h; 20 nm before, and the
# line-block twin passed the same 1e-8 omega tolerance on 40 nm), no PML. em_fdfd_ez pec: E_z = 0 in the outermost cells, so
# the Dirichlet planes are the centres of cells 0 and n-1, L = (n - 1) h, orders m >= 1; em_fdfd_hz pec: no flux through
# the inner face of each outermost cell, so the Neumann planes are faces 0 and n-2, L = (n - 2) h, orders m >= 0 (not
# both 0). Either way the 5-point operator has the exact eigenvalues k^2 = K_x(m)^2 + K_y(n)^2, K(m) = (2 / h) sin(m pi h
# / (2 L)) (sin / cos eigenvectors), here L = 2.0 x 1.2 um. Target 1.55 um: the nearest resonances are (2, 1) at
# ~1.5366 um, then ez (1, 1) and (3, 1), hz (3, 0) and (1, 1). With a uniform sigma_e the frozen permittivity eps_t =
# 1 - j sigma_e / (omega_t eps0) gives omega = C0 K / sqrt(eps_t) exactly (Q ~ omega_t eps0 / sigma_e ~ 1080), which pins
# the sign of Im omega and the Q formula.
_CELL = 20e-9                                  # m, the disk_resonator_eigenmodes template
_CAVITY_CELL = 40e-9                           # m
_TARGET_WAVELENGTH = 1.55e-6                   # m, the wavelength setting
_N_RESONANCES = 3
_SIGMA_E = 10.0                                # S/m, lossy run
_RUNS = {'ez': ('ez', (51, 31), 0.0), 'hz': ('hz', (52, 32), 0.0), 'ez_lossy': ('ez', (51, 31), _SIGMA_E)}
_MAX_ORDER = 8                                 # orders searched for the expected nearest resonances
_OMEGA_TOLERANCE = 1e-8                        # relative complex omega
_Q_TOLERANCE = 1e-6                            # relative Q (lossy run)
_FIELD_TOLERANCE = 1e-5                        # max |E_z - closed form| of resonance 1 (peak 1, float32 output)
# em_fdfd_eigenfrequency_line_block (solver_plan Part II stage 3): the same runs with a line_block 'linear' node, whose
# factor of the operator at the target is the shift-invert factor; (omega, Q) tolerances per precision (mixed: complex64
# block factors refined against the complex128 operator to 1e-9 per Arnoldi step)
_LINE_BLOCK_TOLERANCES = {'double': (_OMEGA_TOLERANCE, _Q_TOLERANCE), 'mixed': (1e-7, 1e-4)}
# the same 2.0 x 1.2 um cavities on 80 nm cells (the closed form is exact for any h; the block sweep is launch-bound,
# so its time goes with the block count), mixed only on the lossy, non-Hermitian run and no hz run (the line-block
# factor and its refinement are operator-agnostic; the hz walls are pinned by em_fdfd_eigenfrequency_pec_cavity):
# ~59 s with 20 nm cells and every run in both precisions, ~11 s on 40 nm cells with hz double
_LINE_BLOCK_CELL = 80e-9
_LINE_BLOCK_RUN_CELLS = {'ez': (26, 16), 'ez_lossy': (26, 16)}
_LINE_BLOCK_RUNS = {'double': ('ez', 'ez_lossy'), 'mixed': ('ez_lossy',)}
# em_fdfd_eigenfrequency_time_filter: the same three cavities with a time_filter 'eigen' node (FDTD run of the cavity +
# Rayleigh-Ritz of the FDFD pencil). Bandwidth: the references lie at -25 % (ez / hz (1, 1)), +16 % (hz (3, 0)) and
# +33 % (ez (3, 1)) of the target, and only resonances inside +-bandwidth are reported. Closed lossless (or uniformly
# lossy) cavities: the FDTD and FDFD share the Yee Laplacian, so the filtered fields are exact combinations of the
# discrete modes and only the out-of-band leakage (smooth across the samples: a few span directions) and the float32
# stepping noise (~1e-6) remain; the bilinear Rayleigh quotient squares them: omega to 1e-8 (as ARPACK's case), Q to
# 1e-5 (Q ~ 1080: a Q error ~ 2 Q x the omega error), the field (first order in the span error) to 1e-4.
_TIME_FILTER_NODE = 'time_filter'
_TIME_FILTER_BANDWIDTH = 0.4
_TIME_FILTER_TOLERANCES = (1e-8, 1e-5)          # relative complex omega, relative Q
_TIME_FILTER_FIELD_TOLERANCE = 1e-4
# em_fdfd_time_filter_open_disc: a small silicon disc (eps_r of the template) in a CPML box, time_filter against
# arpack_shift_invert (Automatic) on the same model: every high-Q arpack resonance within the inner band against the
# nearest time-filter one (the FDTD's CPML differs from the frozen-stretch FDFD PML: the polish steps remove it)
_DISC_RADIUS = 0.6e-6                           # m: whispering-gallery order ~7, FSR ~10 % of the target
_DISC_DOMAIN = 2.0e-6                           # m, square (CPML outside)
_DISC_CELL = 25e-9                              # m: 80 x 80 window cells + the CPML ring
_DISC_RESONANCES = 4
_DISC_BANDWIDTH = 0.15
_DISC_COMPARED_BAND = 0.8                       # compared: |f / f_target - 1| <= this x the bandwidth
_DISC_MIN_Q = 100.0                             # compared: arpack Q at least this (leaky modes are not the point)
_DISC_TOLERANCES = (1e-6, 1e-3)                 # relative wavelength, relative Q

# 'disk_resonator_eigenmodes' template: the disk of the FDTD 'disk_resonator' template (silicon eps_r 12.11, r = 1 um,
# 4 x 4 um CPML box, 20 nm mesh: (200 + 30)^2 = 52,900 unknowns), solved for its whispering-gallery modes near 1.55 um
# (azimuthal order ~12, free spectral range ~0.1 um); each mode comes as a cos / sin pair split by the staircase.
DISK_EIGENMODES_LABEL = 'Disk resonator eigenmodes'
DISK_EIGENMODES_DESCRIPTION = (
    'EM FDFD (E_z), eigenfrequency study: the whispering-gallery resonances of a silicon disk (eps_r 12.11, radius '
    'r_disk = 1 um) in air nearest 1.55 um (6 resonances; each mode is a near-degenerate pair on the staircase disk), '
    'with wavelength, Q and field shape per resonance; derived lambda_1 and Q_1 of the nearest one. The same modes as '
    'the FDTD disk_resonator template (ring-down), without a source or a long run.')
DISK_EIGENMODES_SUMMARY = ('Finds the resonances of a silicon disk in air near 1550 nm directly, without a light pulse: '
                           'wavelength, quality factor and field pattern of each whispering-gallery mode.')
_DISK_RADIUS = 1e-6                            # m
_DISK_DOMAIN = 4e-6                            # m, square
_DISK_EPS_R = 12.11


def disk_resonator_eigenmodes_spec():
    """Template 'disk_resonator_eigenmodes': silicon disk in a CPML box, em_fdfd_ez eigenfrequency study near 1.55 um
    with a resonance table and the nearest resonance's wavelength / Q as derived values."""
    centre = 0.5 * _DISK_DOMAIN
    return ModelSpec(
        title=DISK_EIGENMODES_LABEL,
        parameters=[ParameterSpec('r_disk', _DISK_RADIUS, 'disk radius [m]')],
        domain=DomainSpec(size_x=_DISK_DOMAIN, size_y=_DISK_DOMAIN),
        geometry=[GeometrySpec('disk', 'circle', {'cx': centre, 'cy': centre, 'r': 'r_disk'})],
        materials=[MaterialSpec('silicon_disk', properties={'eps_r': _DISK_EPS_R}, selection=['disk'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_ez', settings={'wavelength': _TARGET_WAVELENGTH},
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type=_STUDY, settings={'n_resonances': 6}),
        derived_values=[DerivedValueSpec('lambda_1', 'wavelength', operation='at', source=f'{_PHYSICS}.{_TABLE}', at=0),
                        DerivedValueSpec('Q_1', 'q_factor', operation='at', source=f'{_PHYSICS}.{_TABLE}', at=0)])


def _cavity_spec(polarization, cells, sigma_e, cell=_CAVITY_CELL):
    """em_fdfd_<polarization> in a PEC box of `cells` cells of size `cell` (vacuum, or uniform sigma_e) with a
    resonance table."""
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings['wavelength'] = _TARGET_WAVELENGTH
    materials = ([MaterialSpec('lossy_vacuum', properties={'eps_r': 1.0, 'sigma_e': sigma_e}, selection=['all'])]
                 if sigma_e else [])
    return ModelSpec(
        title=f'em_fdfd_{polarization} PEC cavity eigenfrequencies' + (' (lossy)' if sigma_e else ''),
        domain=DomainSpec(size_x=cells[0] * cell, size_y=cells[1] * cell),
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS, f'em_fdfd_{polarization}', settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _N_RESONANCES}))


def _cavity_specs():
    return {run: _cavity_spec(*arguments) for run, arguments in _RUNS.items()}


def _wall_layout(polarization, cells, cell):
    """((L_x, L_y) between the wall planes [m], lowest order) of the documented ez / hz pec planes."""
    offset, lowest = (1, 1) if polarization == 'ez' else (2, 0)
    return tuple((count - offset) * cell for count in cells), lowest


def _expected_resonances(polarization, cells, sigma_e, cell=_CAVITY_CELL):
    """[(complex omega, (m, n))] of the _N_RESONANCES discrete resonances nearest the target omega."""
    omega_target = 2.0 * math.pi * C0 / _TARGET_WAVELENGTH
    lengths, lowest = _wall_layout(polarization, cells, cell)
    frozen_eps = 1.0 - 1j * sigma_e / (omega_target * EPS0)
    candidates = []
    for m in range(lowest, _MAX_ORDER + 1):
        for n in range(lowest, _MAX_ORDER + 1):
            if m == 0 and n == 0:
                continue
            wavenumber = math.sqrt(sum((2.0 / cell * math.sin(order * math.pi * cell / (2.0 * length))) ** 2
                                       for order, length in zip((m, n), lengths)))
            candidates.append((C0 * wavenumber / cmath.sqrt(frozen_eps), (m, n)))
    candidates.sort(key=lambda candidate: abs(candidate[0] - omega_target))
    return candidates[:_N_RESONANCES]


def _field_error(results, cells, cell):
    """max |E_z - closed form| of resonance 1 (the live field: (2, 1) of the ez cavity) up to a global sign, plus the
    largest |Im E_z|; both relative to the peak 1. The solver scales a resonance to 1 at its largest cell, so the
    closed form is scaled to 1 at its own largest cell too (on 51 x 31 cells the (2, 1) peak i = 12.5 lies between two
    cells, where sin is 0.998: a peak-1 closed form would be 2e-3 off there)."""
    real_part = np.asarray(results.field(f'{_PHYSICS}.ez_re').values, dtype=np.float64)
    imaginary_part = np.asarray(results.field(f'{_PHYSICS}.ez_im').values, dtype=np.float64)
    m, n = _expected_resonances('ez', cells, 0.0, cell)[0][1]
    i_index, j_index = np.arange(cells[0])[:, None], np.arange(cells[1])[None, :]
    closed_form = np.sin(m * math.pi * i_index / (cells[0] - 1)) * np.sin(n * math.pi * j_index / (cells[1] - 1))
    closed_form /= np.max(np.abs(closed_form))
    if real_part.shape != closed_form.shape:
        raise ValueError(f'ez_re window shape {real_part.shape} differs from the {cells} cavity cells')
    shape_error = min(float(np.max(np.abs(real_part - closed_form))), float(np.max(np.abs(real_part + closed_form))))
    return max(shape_error, float(np.max(np.abs(imaginary_part))))


def _cavity_errors(results_by_run, omega_tolerance, q_tolerance, cell=_CAVITY_CELL, run_cells=None,
                   field_tolerance=_FIELD_TOLERANCE):
    """(errors, tolerances, details) of the _RUNS in results_by_run (_RUNS key -> Results; run_cells: cells per run
    when not the _RUNS ones): complex omega of resonances 1..3 vs the closed form, Q of the lossy run, snapshot count
    and the resonance-1 field of the lossless ez run (when present)."""
    errors, tolerances, details = {}, {}, {}
    run_cells = run_cells or {run: cells for run, (_, cells, _) in _RUNS.items()}
    for run, results in results_by_run.items():
        polarization, _, sigma_e = _RUNS[run]
        cells = run_cells[run]
        rows = results.result(_PHYSICS, _TABLE)['rows']
        if len(rows) < _N_RESONANCES:
            raise ValueError(f'{run}: expected {_N_RESONANCES} resonances in the table, got {len(rows)}')
        expected = _expected_resonances(polarization, cells, sigma_e, cell)
        measured = [2.0 * math.pi * complex(float(row[_OMEGA_COLUMNS[0]]), float(row[_OMEGA_COLUMNS[1]]))
                    for row in rows[:_N_RESONANCES]]
        errors[f'{run} omega'] = max(abs(value / reference - 1.0) for value, (reference, _) in zip(measured, expected))
        tolerances[f'{run} omega'] = omega_tolerance
        if sigma_e:
            q_expected = [reference.real / (2.0 * reference.imag) for reference, _ in expected]
            q_measured = [float(row[_Q_COLUMN]) for row in rows[:_N_RESONANCES]]
            errors[f'{run} Q'] = max(abs(value / reference - 1.0) for value, reference in zip(q_measured, q_expected))
            tolerances[f'{run} Q'] = q_tolerance
            details[f'{run} Q'] = {'measured': q_measured, 'expected': q_expected}
        labels = results.summary().get('snapshot_labels') or []
        errors[f'{run} snapshots'] = abs(len(labels) - _N_RESONANCES)
        tolerances[f'{run} snapshots'] = 0.5
        details[run] = {'orders': [list(orders) for _, orders in expected],
                        'wavelength_expected': [2.0 * math.pi * C0 / reference.real for reference, _ in expected],
                        'wavelength_measured': float_list([float(row[1]) for row in rows[:_N_RESONANCES]]),
                        'snapshot_labels': list(labels)}
    if 'ez' in results_by_run:
        errors['ez field'] = _field_error(results_by_run['ez'], run_cells['ez'], cell)
        tolerances['ez field'] = field_tolerance
    return errors, tolerances, details


def _check_cavity(results_by_name):
    """_cavity_errors of the Automatic runs; measured = error = the worst deviation in tolerance units."""
    return tolerance_units_result(*_cavity_errors(results_by_name, _OMEGA_TOLERANCE, _Q_TOLERANCE))


def _line_block_cavity_specs():
    """'<run>_<precision>': the _LINE_BLOCK_RUNS cavities on _LINE_BLOCK_CELL cells with a line_block node of that
    precision as the 'linear' solver (so also the shift-invert factor)."""
    specs = {}
    for precision, runs in _LINE_BLOCK_RUNS.items():
        for run in runs:
            polarization, _, sigma_e = _RUNS[run]
            spec = _cavity_spec(polarization, _LINE_BLOCK_RUN_CELLS[run], sigma_e, _LINE_BLOCK_CELL)
            node = f'line_block_{precision}'
            spec.solvers.append(SolverSpec(node, 'line_block', {'precision': precision}))
            spec.physics_by_name(_PHYSICS).solvers['linear'] = node
            specs[f'{run}_{precision}'] = spec
    return specs


def _check_line_block_cavity(results_by_name):
    """Per precision _cavity_errors with its tolerances, plus the summary record of the linear role: type line_block,
    every ARPACK step through the kept factor (factor_solves > 0), rejected_factor_solves 0."""
    errors, tolerances, details = {}, {}, {}
    for precision, (omega_tolerance, q_tolerance) in _LINE_BLOCK_TOLERANCES.items():
        runs = {run: results_by_name[f'{run}_{precision}'] for run in _LINE_BLOCK_RUNS[precision]}
        part_errors, part_tolerances, part_details = _cavity_errors(runs, omega_tolerance, q_tolerance,
                                                                    _LINE_BLOCK_CELL, _LINE_BLOCK_RUN_CELLS)
        errors.update({f'{precision} {item}': value for item, value in part_errors.items()})
        tolerances.update({f'{precision} {item}': value for item, value in part_tolerances.items()})
        details[precision] = part_details
        for run, results in runs.items():
            linear = solver_role(results, _PHYSICS)
            counters = linear.get('counters') or {}
            used = (linear.get('type') == 'line_block' and float(counters.get('factor_solves', 0)) > 0
                    and float(counters.get('rejected_factor_solves', math.inf)) == 0.0)
            errors[f'{precision} {run} line_block record'] = 0.0 if used else math.inf
            tolerances[f'{precision} {run} line_block record'] = 1.0
            details[f'{precision} {run} counters'] = dict(counters)
    return tolerance_units_result(errors, tolerances, details)


def _with_time_filter(spec, bandwidth):
    """spec with a time_filter node of `bandwidth` (else its defaults) as the physics' 'eigen' solver."""
    spec.solvers.append(SolverSpec(_TIME_FILTER_NODE, 'time_filter', {'bandwidth': bandwidth}))
    spec.physics_by_name(_PHYSICS).solvers['eigen'] = _TIME_FILTER_NODE
    return spec


def _time_filter_cavity_specs():
    """The _RUNS cavities (the run backend: gpu, cpu variant) with a time_filter eigen node."""
    return {run: _with_time_filter(_cavity_spec(*arguments), _TIME_FILTER_BANDWIDTH) for run, arguments in _RUNS.items()}


def _time_filter_record(results, errors, tolerances, details, key):
    """errors[key] 0 when the run summary records the eigen role as time_filter with one solve and time steps > 0."""
    eigen = solver_role(results, _PHYSICS, 'eigen')
    counters = eigen.get('counters') or {}
    used = (eigen.get('type') == _TIME_FILTER_NODE and float(counters.get('eigen_solves', 0)) == 1.0
            and float(counters.get('time_steps', 0)) > 0)
    errors[key] = 0.0 if used else math.inf
    tolerances[key] = 1.0
    details[f'{key} counters'] = dict(counters)


def _check_time_filter_cavity(results_by_name):
    """_cavity_errors with the time-filter tolerances plus each run's eigen role record (_time_filter_record)."""
    errors, tolerances, details = _cavity_errors(results_by_name, *_TIME_FILTER_TOLERANCES,
                                                 field_tolerance=_TIME_FILTER_FIELD_TOLERANCE)
    for run, results in results_by_name.items():
        _time_filter_record(results, errors, tolerances, details, f'{run} time_filter record')
    return tolerance_units_result(errors, tolerances, details)


def _disc_spec(time_filter):
    """Silicon disc of radius _DISC_RADIUS centred in a _DISC_DOMAIN CPML box, em_fdfd_ez eigenfrequency study near
    the target (resonance table); time_filter: a time_filter eigen node, else Automatic (arpack_shift_invert)."""
    centre = 0.5 * _DISC_DOMAIN
    spec = ModelSpec(
        title='Silicon disc resonances (' + ('time_filter' if time_filter else 'arpack_shift_invert') + ')',
        domain=DomainSpec(size_x=_DISC_DOMAIN, size_y=_DISC_DOMAIN),
        geometry=[GeometrySpec('disc', 'circle', {'cx': centre, 'cy': centre, 'r': _DISC_RADIUS})],
        materials=[MaterialSpec('silicon_disc', properties={'eps_r': _DISK_EPS_R}, selection=['disc'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_ez', settings={'wavelength': _TARGET_WAVELENGTH},
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_DISC_CELL),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _DISC_RESONANCES}))
    return _with_time_filter(spec, _DISC_BANDWIDTH) if time_filter else spec


def _disc_specs():
    return {'arpack': _disc_spec(False), 'time_filter': _disc_spec(True)}


def _check_disc(results_by_name):
    """Every arpack resonance with Q >= _DISC_MIN_Q within _DISC_COMPARED_BAND x the bandwidth of the target against the
    time-filter resonance of nearest complex frequency: relative wavelength and Q (at least one compared), plus the
    time_filter record."""
    omega_target = 2.0 * math.pi * C0 / _TARGET_WAVELENGTH
    reference_rows = results_by_name['arpack'].result(_PHYSICS, _TABLE)['rows']
    filter_rows = results_by_name['time_filter'].result(_PHYSICS, _TABLE)['rows']

    def frequency(row):
        return complex(float(row[_OMEGA_COLUMNS[0]]), float(row[_OMEGA_COLUMNS[1]]))
    errors, tolerances, details = {}, {}, {'arpack': [float_list(row) for row in reference_rows],
                                          'time_filter': [float_list(row) for row in filter_rows]}
    compared = [row for row in reference_rows if float(row[_Q_COLUMN]) >= _DISC_MIN_Q
                and abs(2.0 * math.pi * frequency(row).real / omega_target - 1.0) <= _DISC_COMPARED_BAND
                * _DISC_BANDWIDTH]
    errors['compared resonances'] = 0.0 if compared and filter_rows else math.inf
    tolerances['compared resonances'] = 1.0
    for row in compared if filter_rows else ():
        match = min(filter_rows, key=lambda candidate: abs(frequency(candidate) - frequency(row)))
        key = f'resonance {int(row[0])}'
        errors[f'{key} wavelength'] = abs(float(match[1]) / float(row[1]) - 1.0)
        tolerances[f'{key} wavelength'] = _DISC_TOLERANCES[0]
        errors[f'{key} Q'] = abs(float(match[_Q_COLUMN]) / float(row[_Q_COLUMN]) - 1.0)
        tolerances[f'{key} Q'] = _DISC_TOLERANCES[1]
    _time_filter_record(results_by_name['time_filter'], errors, tolerances, details, 'time_filter record')
    return tolerance_units_result(errors, tolerances, details)


EIGENFREQUENCY_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_eigenfrequency_pec_cavity',
        label='EM FDFD: eigenfrequencies of a PEC cavity',
        description=f'eigenfrequency study (target {_TARGET_WAVELENGTH * 1e6:g} um, n_resonances {_N_RESONANCES}) of a '
                    f'vacuum rectangle with pec on all four faces, uniform {_CAVITY_CELL * 1e9:g} nm cells, no PML. '
                    'Runs ez (em_fdfd_ez, 51 x 31 cells: E_z = 0 in the outermost cells, so the walls are their '
                    'centres, 2.0 x 1.2 um, orders >= 1), hz (em_fdfd_hz, 52 x 32 cells: no flux through the inner face of '
                    'the outermost cells, 2.0 x 1.2 um, orders >= 0) and ez_lossy (the ez cavity filled with sigma_e '
                    f'{_SIGMA_E:g} S/m). The three resonances nearest the target must equal the discrete closed form '
                    'omega = C0 K / sqrt(eps_t), K^2 = K_x(m)^2 + K_y(n)^2, K(m) = (2 / h) sin(m pi h / (2 L)), '
                    'eps_t = 1 - j sigma_e / (omega_t eps0) frozen at the target (exact for this discretisation), '
                    'in that order (ranked by |omega - omega_target|): relative complex omega error '
                    f'{_OMEGA_TOLERANCE:g}, the lossy Q = Re omega / (2 Im omega) (~1080, decaying: Im omega > 0) to '
                    f'{_Q_TOLERANCE:g}; each run stores {_N_RESONANCES} snapshots; the live field of the ez run '
                    '(resonance 1 = the (2, 1) mode) equals sin(2 pi i / 50) sin(pi j / 30) (scaled to 1 at its own '
                    'largest cell) up to the sign within '
                    f'{_FIELD_TOLERANCE:g} (peak 1, Im 0). measured = error = the worst deviation in tolerance units, '
                    'tolerance 1: this pins both wall planes, the shift-invert eigenvalue map, the ranking, the Q '
                    'sign convention and the peak normalisation. cpu backend (host ARPACK on both) and 40 nm cells: '
                    'not yet confirmed by a run with these settings (the omega item used 0.61 of its tolerance at 20 '
                    'nm, an ARPACK floor).',
        specs=lambda: cpu_specs(_cavity_specs()),
        check=_check_cavity),
    SelfTestCase(
        name='em_fdfd_eigenfrequency_line_block',
        label='EM FDFD: line-block shift-invert eigenfrequencies',
        description='solver_plan Part II stage 3: the em_fdfd_eigenfrequency_pec_cavity cavities ez and ez_lossy on '
                    f'{_LINE_BLOCK_CELL * 1e9:g} nm cells (26 x 16; the hz walls are pinned by the SuperLU case, the '
                    'line-block factor is operator-agnostic) with a line_block solver node assigned '
                    "as the 'linear' solver, precision double (both) and mixed (ez_lossy; 3 runs; not yet confirmed "
                    'by a run with these settings): the '
                    'shift-invert factor of the eigen-solve is then the line-block factor of the operator at the '
                    'target (every ARPACK step one block sweep plus refinement) instead of host SuperLU. The same '
                    'closed-form checks as em_fdfd_eigenfrequency_pec_cavity with relative omega / Q tolerances '
                    f"double {_LINE_BLOCK_TOLERANCES['double'][0]:g} / {_LINE_BLOCK_TOLERANCES['double'][1]:g}, "
                    f"mixed {_LINE_BLOCK_TOLERANCES['mixed'][0]:g} / {_LINE_BLOCK_TOLERANCES['mixed'][1]:g}, the field "
                    f'shape to {_FIELD_TOLERANCE:g}; and the run summary must record the linear role as line_block with '
                    'factor_solves > 0 and rejected_factor_solves 0 (a rejection redoes the eigen-solve on SuperLU and '
                    'would hide the line-block result). measured = error = the worst item in tolerance units, '
                    'tolerance 1.',
        specs=_line_block_cavity_specs,
        check=_check_line_block_cavity),
)

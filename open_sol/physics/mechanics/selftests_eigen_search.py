"""Self-tests of the eigenfrequency study's search 'band' / 'targets' on solid_mechanics (roller boxes with discrete
closed-form modes) and solid_mechanics_3d on tetrahedra (a cantilever: modal band / targets against dense eigh, and a
band study against a 'nearest' study of the same mesh). solid_eigenfrequency_band_host_direct: the rectangle band search
with a MUMPS / PARDISO eigen factorization (inertia counts from the host LDL^T; skipped without python-mumps / pydiso)."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (available_host_direct, float32_value, skip_without_host_direct, solver_role,
                                       with_direct_node)
from ...model.spec import FeatureSpec, MaterialSpec, StudySpec
from .selftests import _elastic, _lame
from .selftests_eigenfrequency import _CELL, _PHYSICS, _TABLE, _box_spec, _discrete_modes
from .selftests_fem3d import UM, _solid_spec

_FREQUENCY_COLUMN = 1             # frequency [Hz] in the resonance_table rows
_FREQUENCY_TOLERANCE = 1e-8       # relative (float64 pencils, eigsh to round-off)
_GAP = 1e-6                       # relative: frequencies closer than this are one (degenerate) group, never split by a band edge

# ---- solid_eigenfrequency_band_roller_box: the orthotropic roller box of selftests_eigenfrequency ('rectangle', 20 x 8
# cells) and a square one ('square', 8 x 8 cells, c11 = c22: the (p, q) and (q, p) modes are exactly degenerate pairs),
# each with a band around its sorted discrete modes _BAND_INDICES (widened past degenerate groups), _PER_SHIFT per shift
_SQUARE_CELLS = (8, 8)
_SQUARE_SIZE = (_SQUARE_CELLS[0] * _CELL, _SQUARE_CELLS[1] * _CELL)
_SQUARE_STIFFNESS = {name: float32_value(value)
                     for name, value in (('c11', 165.7e9), ('c12', 63.9e9), ('c22', 165.7e9), ('c66', 79.6e9))}
_BAND_INDICES = (2, 13)           # sorted mode indices inside the band (12 modes before widening)
_PER_SHIFT = 4
_MAX_RESONANCES = 100


def _band_edges(frequencies, first, last):
    """(f_lo, f_hi) [Hz] at the gap middles enclosing sorted frequencies[first .. last], widened past groups closer than
    _GAP so no degenerate pair is split."""
    def apart(index):
        return frequencies[index] - frequencies[index - 1] > _GAP * frequencies[index]
    while first > 0 and not apart(first):
        first -= 1
    while last + 1 < len(frequencies) and not apart(last + 1):
        last += 1
    low = 0.5 * (frequencies[first - 1] + frequencies[first]) if first > 0 else 0.5 * frequencies[0]
    return low, 0.5 * (frequencies[last] + frequencies[last + 1])


def _band_runs():
    """{run: (cells, stiffness or None, size or None)} of the band case."""
    return {'rectangle': ((20, 8), None, None), 'square': (_SQUARE_CELLS, _SQUARE_STIFFNESS, _SQUARE_SIZE)}


def _run_band(cells, stiffness):
    """(f_lo, f_hi, expected sorted Hz in the band) of a roller box."""
    frequencies = [frequency for frequency, _ in _discrete_modes(cells, stiffness)]
    low, high = _band_edges(frequencies, *_BAND_INDICES)
    return low, high, [frequency for frequency in frequencies if low <= frequency <= high]


def _band_specs():
    specs = {}
    for run, (cells, stiffness, size) in _band_runs().items():
        low, high, _ = _run_band(cells, stiffness)
        specs[run] = _box_spec(0.0, _PER_SHIFT, **({} if size is None else {'size': size}), stiffness=stiffness,
                               study_settings={'search': 'band', 'frequency_min': low, 'frequency_max': high,
                                               'max_resonances': _MAX_RESONANCES})
    return specs


def _sorted_error(measured, expected):
    """Largest relative deviation of sorted measured from sorted expected (inf on a count mismatch or none)."""
    measured, expected = np.sort(np.asarray(measured, float)), np.sort(np.asarray(expected, float))
    if measured.size != expected.size or not measured.size:
        return math.inf
    return float(np.max(np.abs(measured / expected - 1.0)))


def _normalized_report(errors, tolerances, details):
    """measured = error = the worst deviation in tolerance units, tolerance 1."""
    normalized = {item: error / tolerances[item] for item, error in errors.items()}
    worst_item = max(normalized, key=lambda item: math.inf if math.isnan(normalized[item]) else normalized[item])
    details.update(errors=errors, worst_item=worst_item)
    return {'measured': normalized[worst_item], 'expected': 0.0, 'error': normalized[worst_item], 'tolerance': 1.0,
            'details': details}


def _check_band(results_by_name):
    """Per run: the table equals the discrete modes of the band (count, values, frequency order), the inertia count
    equals their number, one snapshot per mode."""
    errors, tolerances, details = {}, {}, {}
    for run, (cells, stiffness, _) in _band_runs().items():
        low, high, expected = _run_band(cells, stiffness)
        results = results_by_name[run]
        table = results.result(_PHYSICS, _TABLE)
        measured = [float(row[_FREQUENCY_COLUMN]) for row in table['rows']]
        labels = results.summary().get('snapshot_labels') or []
        errors[f'{run} frequencies'] = _sorted_error(measured, expected)
        tolerances[f'{run} frequencies'] = _FREQUENCY_TOLERANCE
        errors[f'{run} order'] = float(any(np.diff(measured) < 0.0))
        errors[f'{run} expected_count'] = abs((table.get('expected_count') or 0) - len(expected))
        errors[f'{run} snapshots'] = abs(len(labels) - len(expected))
        for item in ('order', 'expected_count', 'snapshots'):
            tolerances[f'{run} {item}'] = 0.5
        details[run] = {'band_Hz': [low, high], 'frequency_expected': expected, 'frequency_measured': measured,
                        'expected_count': table.get('expected_count'), 'snapshot_labels': list(labels)}
    return _normalized_report(errors, tolerances, details)


# ---- solid_eigenfrequency_band_host_direct: the 'rectangle' band run with an eigen node per installed host direct
# factorization (literals: the selftest rule), cpu backend: inertia counts from MUMPS INFOG(12) / PARDISO iparm ----
_HOST_DIRECT_FACTORIZATIONS = ('mumps', 'pardiso')
_HOST_DIRECT_RUN = 'rectangle'


def _host_direct_band_specs():
    """One band run of the _HOST_DIRECT_RUN box per installed factorization of _HOST_DIRECT_FACTORIZATIONS; {} without
    python-mumps / pydiso."""
    factorizations = available_host_direct(_HOST_DIRECT_FACTORIZATIONS)
    if not factorizations:
        return {}
    cells, stiffness, _ = _band_runs()[_HOST_DIRECT_RUN]
    low, high, _ = _run_band(cells, stiffness)
    specs = {}
    for factorization in factorizations:
        spec = _box_spec(0.0, _PER_SHIFT, stiffness=stiffness,
                         study_settings={'search': 'band', 'frequency_min': low, 'frequency_max': high,
                                         'max_resonances': _MAX_RESONANCES})
        spec.solver_defaults.backend = 'cpu'
        specs[factorization] = with_direct_node(spec, factorization, role='eigen', physics_names=(_PHYSICS,))
    return specs


def _check_band_host_direct(results_by_name):
    """Per factorization as _check_band (discrete modes of the band, frequency order, the inertia count, snapshots) and
    an 'eigen' record of a host direct factor (eigen_host_direct_analyses > 0)."""
    if not results_by_name:
        return skip_without_host_direct(_HOST_DIRECT_FACTORIZATIONS)
    cells, stiffness, _ = _band_runs()[_HOST_DIRECT_RUN]
    low, high, expected = _run_band(cells, stiffness)
    errors, tolerances, details = {}, {}, {}
    for factorization, results in results_by_name.items():
        table = results.result(_PHYSICS, _TABLE)
        measured = [float(row[_FREQUENCY_COLUMN]) for row in table['rows']]
        labels = results.summary().get('snapshot_labels') or []
        counters = solver_role(results, _PHYSICS, 'eigen').get('counters') or {}
        errors[f'{factorization} frequencies'] = _sorted_error(measured, expected)
        tolerances[f'{factorization} frequencies'] = _FREQUENCY_TOLERANCE
        errors[f'{factorization} order'] = float(any(np.diff(measured) < 0.0))
        errors[f'{factorization} expected_count'] = abs((table.get('expected_count') or 0) - len(expected))
        errors[f'{factorization} snapshots'] = abs(len(labels) - len(expected))
        errors[f'{factorization} record'] = 0.0 if int(counters.get('eigen_host_direct_analyses') or 0) > 0 else 1.0
        for item in ('order', 'expected_count', 'snapshots', 'record'):
            tolerances[f'{factorization} {item}'] = 0.5
        details[factorization] = {'band_Hz': [low, high], 'frequency_expected': expected,
                                  'frequency_measured': measured, 'expected_count': table.get('expected_count'),
                                  'counters': dict(counters)}
    return _normalized_report(errors, tolerances, details)


# ---- solid_eigenfrequency_targets: the orthotropic box at two targets vs two 'nearest' runs and the closed form ----
_TARGETS = ((3, 1.03), (5, 0.99))   # (sorted mode index, factor): targets overlap in their nearest modes
_PER_TARGET = 3


def _target_frequencies():
    frequencies = [frequency for frequency, _ in _discrete_modes()]
    return [factor * frequencies[index] for index, factor in _TARGETS]


def _targets_specs():
    targets = _target_frequencies()
    specs = {'targets': _box_spec(0.0, _PER_TARGET, study_settings={'search': 'targets',
                                                                    'target_frequencies': targets})}
    for number, target in enumerate(targets, start=1):
        specs[f'nearest_{number}'] = _box_spec(target, _PER_TARGET)
    return specs


def _distinct(frequencies):
    """Sorted frequencies with copies closer than _GAP (relative) merged."""
    kept = []
    for frequency in sorted(frequencies):
        if not kept or frequency - kept[-1] > _GAP * frequency:
            kept.append(frequency)
    return kept


def _check_targets(results_by_name):
    """The targets table equals the union of the nearest runs' tables and of the closed-form nearest modes (frequency
    order, shared modes once), target_frequency NaN and the targets listed, one snapshot per mode."""
    frequencies = [frequency for frequency, _ in _discrete_modes()]
    targets = _target_frequencies()
    expected = _distinct([frequency for target in targets
                          for frequency in sorted(frequencies, key=lambda value: abs(value - target))[:_PER_TARGET]])
    results = results_by_name['targets']
    table = results.result(_PHYSICS, _TABLE)
    measured = [float(row[_FREQUENCY_COLUMN]) for row in table['rows']]
    nearest = _distinct([float(row[_FREQUENCY_COLUMN]) for number in range(1, len(targets) + 1)
                         for row in results_by_name[f'nearest_{number}'].result(_PHYSICS, _TABLE)['rows']])
    labels = results.summary().get('snapshot_labels') or []
    listed = table.get('targets') or []
    errors = {'closed form': _sorted_error(measured, expected), 'nearest runs': _sorted_error(measured, nearest),
              'order': float(any(np.diff(measured) < 0.0)), 'snapshots': abs(len(labels) - len(expected)),
              'table extras': float(not math.isnan(table['target_frequency'])
                                    or _sorted_error(listed, targets) > _FREQUENCY_TOLERANCE)}
    tolerances = {'closed form': _FREQUENCY_TOLERANCE, 'nearest runs': _FREQUENCY_TOLERANCE, 'order': 0.5,
                  'snapshots': 0.5, 'table extras': 0.5}
    return _normalized_report(errors, tolerances, {'targets_Hz': targets, 'frequency_expected': expected,
                                                   'frequency_nearest_runs': nearest, 'frequency_measured': measured,
                                                   'snapshot_labels': list(labels)})


# ---- solid_fem3d_eigen_band_dense: a cantilever 1 x 0.25 x 0.25 um (Kuhn cubes of 0.25 um, P2, x_lo clamped), E 100
# GPa, nu 0.3, rho 2330. Operator: modal.ModalPencil + solve_band_modes / solve_modes_at_shifts (host SuperLU) against
# scipy eigh of the dense free-dof pencil. Study: search 'band' against a 'nearest' run at 0 of the same mesh (the
# _LOWEST_3D lowest modes cover the band) between _BAND_3D_FACTORS x the Euler-Bernoulli first bending frequency.
_BAR_3D = (1.0 * UM, 0.25 * UM, 0.25 * UM)
_BAR_3D_CELL = 0.25 * UM
_BAR_3D_MATERIAL = (100e9, 0.3, 2330.0)   # E [Pa], nu, rho [kg/m^3]
_DENSE_BAND_INDICES = (2, 11)
_DENSE_TARGETS = ((2, 1.02), (6, 0.98))   # (sorted dense mode index, factor)
_BAND_3D_FACTORS = (0.5, 4.0)
_LOWEST_3D = 30
_PER_SHIFT_3D = 4


def _cantilever_system():
    """(K, M, free bool) of the clamped P2 bar on structured Kuhn cubes (host scipy)."""
    from ...fem.elasticity3d import isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from .system_fem3d import ElasticFEMSystem3D
    edges = [np.linspace(0.0, side, int(round(side / _BAR_3D_CELL)) + 1) for side in _BAR_3D]
    mesh = structured_box_mesh(*edges)
    system = ElasticFEMSystem3D(mesh, 2, lambda: 'eigen band self-test')
    system.hold(system.full.face_dofs(system.face_faces('x_lo')), u_x=0.0, u_y=0.0, u_z=0.0)
    lame_lambda, mu = _lame(*_BAR_3D_MATERIAL[:2])
    count = mesh.n_elements
    system.assemble(isotropic_matrix_3d(np.full(count, lame_lambda), np.full(count, mu)), np.ones(count, dtype=bool))
    mass = system.mass(np.full(count, _BAR_3D_MATERIAL[2]))
    return system.operator, mass, np.asarray(system.free, dtype=bool)


def _dense_frequencies(stiffness, mass, free):
    """Sorted Hz of the dense free-dof pencil (scipy eigh)."""
    import scipy.linalg
    dofs = np.flatnonzero(free.ravel())
    reduced_stiffness = stiffness.tocsr()[dofs][:, dofs].toarray()
    reduced_mass = mass.tocsr()[dofs][:, dofs].toarray()
    eigenvalues = scipy.linalg.eigh(reduced_stiffness, reduced_mass, eigvals_only=True)
    return np.sqrt(np.clip(eigenvalues, 0.0, None)) / (2.0 * math.pi)


def _operator_items():
    """(errors, tolerances, details) of the modal band / targets against dense eigh."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .modal import ModalPencil, solve_band_modes, solve_modes_at_shifts
    stiffness, mass, free = _cantilever_system()
    dense = list(_dense_frequencies(stiffness, mass, free))
    low, high = _band_edges(dense, *_DENSE_BAND_INDICES)
    expected_band = [frequency for frequency in dense if low <= frequency <= high]
    band = solve_band_modes(ModalPencil(stiffness, mass, free, 3, ArpackShiftInvertSolver()), low, high,
                            _PER_SHIFT_3D, _MAX_RESONANCES)
    band_measured = [resonance.frequency for resonance in band.resonances]
    targets = [factor * dense[index] for index, factor in _DENSE_TARGETS]
    expected_targets = _distinct([frequency for target in targets
                                  for frequency in sorted(dense, key=lambda value: abs(value - target))[:_PER_SHIFT_3D]])
    union = solve_modes_at_shifts(ModalPencil(stiffness, mass, free, 3, ArpackShiftInvertSolver()),
                                  [2.0 * math.pi * target for target in targets], _PER_SHIFT_3D)
    union_measured = [resonance.frequency for resonance in union.resonances]
    errors = {'band': _sorted_error(band_measured, expected_band),
              'band count': abs((band.info['expected_count'] or 0) - len(expected_band)),
              'targets': _sorted_error(union_measured, expected_targets)}
    tolerances = {'band': _FREQUENCY_TOLERANCE, 'band count': 0.5, 'targets': _FREQUENCY_TOLERANCE}
    details = {'band_Hz': [low, high], 'band_expected': expected_band, 'band_measured': band_measured,
               'band_warnings': band.info['warnings'], 'targets_Hz': targets, 'targets_expected': expected_targets,
               'targets_measured': union_measured}
    return errors, tolerances, details


# ---- solid_fem3d_eigen_band_cudss: the cantilever pencil through ModalPencil on cuDSS (ShiftPencil -> DevicePencil,
# factorize_pencil per shift, GPU): band (cudss, inertia counts) and targets (cudss_mixed) against dense eigh with one
# cuDSS plan per pencil (R1e; a mixed double fallback adds its own); a damped copy K (1 + j eta) (eigs_shifted, complex
# LDL^T) against the same pencil on SuperLU
_DAMPED_LOSS = 1e-3
_DAMPED_TARGET_INDEX = 4          # sorted dense mode the damped solve targets
_DAMPED_TOLERANCE = 1e-9          # relative complex frequency difference cuDSS vs SuperLU (first estimate)


def _complex_frequencies(solve):
    """Sorted (by real part) complex frequencies f + j Im f [Hz] of a ModalSolve."""
    return np.sort_complex(np.array([resonance.frequency + 1j * resonance.frequency_imag
                                     for resonance in solve.resonances]))


def _device_pencil_solver(factorization):
    """ArpackShiftInvertSolver whose ShiftPencils take a DevicePencil at any size (the cantilever is far below
    DEVICE_PENCIL_MIN_ENTRIES)."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    solver = ArpackShiftInvertSolver(factorization=factorization)
    solver.device_pencil_min_entries = 0
    return solver


def _check_band_cudss(_):
    """Band / targets of the cantilever pencil on cuDSS against dense eigh and one plan per pencil; the damped pencil
    on cuDSS against SuperLU; its ShiftPencil holds a DevicePencil. Passes noted without nvmath-python."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from ...solvers.cudss_factor import cudss_available
    from .modal import ModalPencil, solve_band_modes, solve_modes_at_shifts
    if not cudss_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': 1.0,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    stiffness, mass, free = _cantilever_system()
    dense = list(_dense_frequencies(stiffness, mass, free))
    low, high = _band_edges(dense, *_DENSE_BAND_INDICES)
    expected_band = [frequency for frequency in dense if low <= frequency <= high]
    band_solver = _device_pencil_solver('cudss')
    band = solve_band_modes(ModalPencil(stiffness, mass, free, 3, band_solver), low, high, _PER_SHIFT_3D,
                            _MAX_RESONANCES)
    band_measured = [resonance.frequency for resonance in band.resonances]
    band_counters = band_solver.work_counters()
    targets = [factor * dense[index] for index, factor in _DENSE_TARGETS]
    expected_targets = _distinct([frequency for target in targets
                                  for frequency in sorted(dense, key=lambda value: abs(value - target))[:_PER_SHIFT_3D]])
    targets_solver = _device_pencil_solver('cudss_mixed')
    union = solve_modes_at_shifts(ModalPencil(stiffness, mass, free, 3, targets_solver),
                                  [2.0 * math.pi * target for target in targets], _PER_SHIFT_3D)
    union_measured = [resonance.frequency for resonance in union.resonances]
    targets_counters = targets_solver.work_counters()
    damped_stiffness = (stiffness * (1.0 + 1j * _DAMPED_LOSS)).tocsr()
    omega_target = 2.0 * math.pi * dense[_DAMPED_TARGET_INDEX]
    damped, device_pencil = {}, False
    for factorization in ('superlu', 'cudss'):
        solver = (ArpackShiftInvertSolver(factorization=factorization) if factorization == 'superlu'
                  else _device_pencil_solver(factorization))
        pencil = ModalPencil(damped_stiffness, mass, free, 3, solver, damped=True)
        try:
            damped[factorization] = _complex_frequencies(pencil.solve_at(omega_target, _PER_SHIFT_3D))
            if factorization == 'cudss':
                device_pencil = pencil.shift_pencil is not None and pencil.shift_pencil.device is not None
        finally:
            pencil.release()
    reference = damped['superlu']
    damped_error = (float(np.max(np.abs(damped['cudss'] - reference) / np.abs(reference)))
                    if damped['cudss'].size == reference.size and reference.size else math.inf)
    errors = {'band': _sorted_error(band_measured, expected_band),
              'band count': abs((band.info['expected_count'] or 0) - len(expected_band)),
              'band plans': abs(band_counters['eigen_cudss_plans'] - 1),
              'targets': _sorted_error(union_measured, expected_targets),
              'targets plans': abs(targets_counters['eigen_cudss_plans'] - 1
                                   - targets_counters['eigen_double_fallbacks']),
              'damped': damped_error, 'damped device pencil': float(not device_pencil)}
    tolerances = {'band': _FREQUENCY_TOLERANCE, 'band count': 0.5, 'band plans': 0.5, 'targets': _FREQUENCY_TOLERANCE,
                  'targets plans': 0.5, 'damped': _DAMPED_TOLERANCE, 'damped device pencil': 0.5}
    details = {'band_Hz': [low, high], 'band_expected': expected_band, 'band_measured': band_measured,
               'band_warnings': band.info['warnings'], 'band_counters': band_counters, 'targets_Hz': targets,
               'targets_expected': expected_targets, 'targets_measured': union_measured,
               'targets_counters': targets_counters,
               'damped_Hz': {name: [[value.real, value.imag] for value in values] for name, values in damped.items()}}
    return _normalized_report(errors, tolerances, details)


def _bending_frequency():
    """Euler-Bernoulli first bending frequency [Hz] of the cantilever (square a x a: 1.875^2 / (2 pi) sqrt(E a^2 / (12
    rho L^4)))."""
    youngs_modulus, _, density = _BAR_3D_MATERIAL
    length, side = _BAR_3D[0], _BAR_3D[1]
    return 1.875 ** 2 / (2.0 * math.pi) * math.sqrt(youngs_modulus * side ** 2 / (12.0 * density * length ** 4))


def _cantilever_spec(target_frequency, n_resonances, study_settings=None):
    youngs_modulus, poisson_ratio, density = _BAR_3D_MATERIAL
    return _solid_spec('Self-test: eigenfrequency search of a clamped bar on tetrahedra', _BAR_3D,
                       [MaterialSpec('bar', properties=_elastic(youngs_modulus, poisson_ratio, rho=density),
                                     selection=['all'])],
                       [FeatureSpec('clamp', 'fixed', ['x_lo']), FeatureSpec(_TABLE, 'resonance_table')],
                       cell=_BAR_3D_CELL,
                       study=StudySpec(type='eigenfrequency', settings={'n_resonances': n_resonances,
                                                                         **(study_settings or {})}),
                       settings={'target_frequency': target_frequency})


def _band_3d_edges():
    reference = _bending_frequency()
    return _BAND_3D_FACTORS[0] * reference, _BAND_3D_FACTORS[1] * reference


def _dense_specs():
    low, high = _band_3d_edges()
    return {'lowest': _cantilever_spec(0.0, _LOWEST_3D),
            'band': _cantilever_spec(0.0, _PER_SHIFT_3D, {'search': 'band', 'frequency_min': low,
                                                          'frequency_max': high,
                                                          'max_resonances': _MAX_RESONANCES})}


def _check_dense(results_by_name):
    """Study band table = the 'nearest' run's modes inside the band (and its inertia count), plus the operator items."""
    low, high = _band_3d_edges()
    lowest = [float(row[_FREQUENCY_COLUMN]) for row in results_by_name['lowest'].result(_PHYSICS, _TABLE)['rows']]
    expected = [frequency for frequency in lowest if low <= frequency <= high]
    table = results_by_name['band'].result(_PHYSICS, _TABLE)
    measured = [float(row[_FREQUENCY_COLUMN]) for row in table['rows']]
    errors = {'study band': _sorted_error(measured, expected),
              'study band count': abs((table.get('expected_count') or 0) - len(expected)),
              'lowest covers band': float(not lowest or max(lowest) <= high)}
    tolerances = {'study band': _FREQUENCY_TOLERANCE, 'study band count': 0.5, 'lowest covers band': 0.5}
    operator_errors, operator_tolerances, operator_details = _operator_items()
    errors.update({f'operator {item}': error for item, error in operator_errors.items()})
    tolerances.update({f'operator {item}': tolerance for item, tolerance in operator_tolerances.items()})
    return _normalized_report(errors, tolerances, {'band_Hz': [low, high], 'study_expected': expected,
                                                   'study_measured': measured, 'operator': operator_details})


EIGEN_SEARCH_CASES = (
    SelfTestCase(
        name='solid_eigenfrequency_band_roller_box',
        label="Solid mechanics: eigenfrequency search 'band' on roller boxes",
        description="eigenfrequency study, search 'band', of the orthotropic roller box of "
                    "solid_eigenfrequency_roller_box ('rectangle', 20 x 8 cells) and of a square 8 x 8 cell box with "
                    "c11 = c22 ('square': the (p, q) and (q, p) discrete modes are exact degenerate pairs), "
                    f'{_PER_SHIFT} modes per shift, the band edges at gap middles around the sorted discrete modes '
                    f'{_BAND_INDICES[0]} .. {_BAND_INDICES[1]} (widened past degenerate groups). Per run the table holds '
                    f'every discrete mode of the band once (relative {_FREQUENCY_TOLERANCE:g}, frequency order: '
                    'degenerate pairs both kept, duplicates of two shifts dropped), its expected_count (Sylvester '
                    'inertia) equals their number and one snapshot is stored per mode. measured = error = the worst '
                    'deviation in tolerance units, tolerance 1 (first estimates, not yet run).',
        specs=_band_specs, check=_check_band),
    SelfTestCase(
        name='solid_eigenfrequency_band_host_direct',
        label="Solid mechanics: eigenfrequency search 'band' on a MUMPS / PARDISO eigen factor",
        description="The 'rectangle' run of solid_eigenfrequency_band_roller_box (orthotropic roller box, 20 x 8 cells, "
                    f'{_PER_SHIFT} modes per shift) on the cpu backend with an arpack_shift_invert node per installed '
                    'factorization of mumps, pardiso: the shift factors are the library LDL^T of K - sigma M and the '
                    "band's Sylvester counts come from the library inertia (MUMPS INFOG(12), PARDISO iparm 22 / 23; "
                    'perturbed pivots: a counting SuperLU). Per run as that case: every discrete mode of the band '
                    f'once (relative {_FREQUENCY_TOLERANCE:g}, frequency order), expected_count equal to their number, '
                    'one snapshot per mode, and an eigen record with host direct analyses. measured = error = the '
                    'worst deviation in tolerance units, tolerance 1; passes as skipped without python-mumps / pydiso. '
                    'Not run yet.',
        specs=_host_direct_band_specs, check=_check_band_host_direct),
    SelfTestCase(
        name='solid_eigenfrequency_targets',
        label="Solid mechanics: eigenfrequency search 'targets'",
        description="eigenfrequency study, search 'targets', of the orthotropic roller box at two targets "
                    f'(1.03 x the 4th and 0.99 x the 6th lowest discrete mode, {_PER_TARGET} per target, overlapping) '
                    "against two 'nearest' runs at the same targets: the table equals the union of their tables and "
                    f'of the closed-form nearest modes (relative {_FREQUENCY_TOLERANCE:g}, shared modes once, '
                    'frequency order), target_frequency is NaN and targets lists both, one snapshot per mode. '
                    'measured = error = the worst deviation in tolerance units, tolerance 1 (first estimates, not yet '
                    'run).',
        specs=_targets_specs, check=_check_targets),
)

EIGEN_SEARCH_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_eigen_band_dense',
        label="Solid mechanics 3D on tetrahedra: eigenfrequency search 'band' against dense eigh",
        description='A cantilever 1 x 0.25 x 0.25 um (0.25 um Kuhn cubes, P2, x_lo clamped; E 100 GPa, nu 0.3, rho '
                    '2330; cpu backend, host SuperLU). Operator: modal.solve_band_modes (4 modes per shift, band '
                    'around the dense modes 2 .. 11) and solve_modes_at_shifts (two targets) on a ModalPencil of the '
                    'directly assembled ElasticFEMSystem3D equal scipy eigh of the dense free-dof pencil (relative '
                    f"{_FREQUENCY_TOLERANCE:g}; the band's inertia count equals the dense count). Study: search 'band' "
                    f'between {_BAND_3D_FACTORS[0]:g} and {_BAND_3D_FACTORS[1]:g} x the Euler-Bernoulli first bending '
                    f"frequency equals the modes inside it of a 'nearest' run at 0 ({_LOWEST_3D} lowest) of the same "
                    'model, with that count. measured = error = the worst deviation in tolerance units, tolerance 1 '
                    '(first estimates, not yet run).',
        specs=_dense_specs, check=_check_dense),
    SelfTestCase(
        name='solid_fem3d_eigen_band_cudss',
        label="Solid mechanics 3D on tetrahedra: eigenfrequency band and targets on cuDSS device pencils",
        description='The solid_fem3d_eigen_band_dense cantilever pencil (built directly, no model run) through '
                    'modal.ModalPencil on cuDSS (ShiftPencil -> DevicePencil uploaded once, factorize_pencil per '
                    'shift; device_pencil_min_entries 0, the pencil is far below the size threshold): solve_band_modes with factorization cudss (double LDL^T, inertia counts) and '
                    'solve_modes_at_shifts with cudss_mixed equal scipy eigh of the dense free-dof pencil (relative '
                    f"{_FREQUENCY_TOLERANCE:g}; the band's inertia count equals the dense count), each pencil on one "
                    'cuDSS plan (eigen_cudss_plans 1; a mixed double fallback adds its own); a damped copy K (1 + j '
                    f'{_DAMPED_LOSS:g}) (eigs_shifted, complex LDL^T) at the dense mode {_DAMPED_TARGET_INDEX + 1} '
                    f'gives the complex frequencies of the same pencil on SuperLU (relative {_DAMPED_TOLERANCE:g}) '
                    'and its ShiftPencil held a DevicePencil. measured = error = the worst deviation in tolerance '
                    'units, tolerance 1 (first estimates, not yet run). GPU only; passes with a note without '
                    'nvmath-python.',
        specs=dict, check=_check_band_cudss, backends=('gpu',)),
)

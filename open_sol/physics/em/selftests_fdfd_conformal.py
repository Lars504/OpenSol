"""Model-run self-tests of the FDFD conformal PEC (solver_defaults.conformal_boundaries, solver_plan S7.6): PEC disk
cavity eigenfrequencies vs the Bessel zeros at two resolutions (cpu backend), and the reciprocity of a point pair around
a curved PEC disk (em_fdfd_hz and em_fdfd_ez); no physics imports."""
import math

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import probe_phasor
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, SolverDefaultsSpec,
                           StudySpec)

_PHYSICS = 'em'
_TABLE = 'resonances'
_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME

# ---- disk cavity: vacuum disk cut out of a PEC block (as model/selftests_conformal, repeated: selftest import rule),
# radius 15.37 coarse cells, centre (0.31, 0.17) coarse cells off a cell centre; the fine mesh halves the cells ----
_COARSE_CELL = 50e-9
_CELLS = 36                                    # coarse cells per side
_SIZE = _CELLS * _COARSE_CELL
_RADIUS = 15.37 * _COARSE_CELL
_CENTRE = ((18.0 + 0.31) * _COARSE_CELL, (18.0 + 0.17) * _COARSE_CELL)
_BESSEL_ZERO = {'ez': 2.404825557695773,       # TM01: first zero of J0
                'hz': 1.841183781340659}       # TE11: first zero of J1'
_MODE_NAMES = {'ez': 'TM01', 'hz': 'TE11'}
_N_RESONANCES = 4                              # TE11 is a split pair on the grid
_REFINEMENTS = {'coarse': 1, 'fine': 2}
# conformal omega vs c j / R: the 5-point dispersion (~1e-3 at 15 cells) included; estimated, not measured
_COARSE_TOLERANCE = {'ez': 3e-3, 'hz': 5e-3}
_FINE_FACTOR = 3.0                             # fine error <= coarse tolerance / this (second order: ~4)

# ---- reciprocity: all-CPML 2 x 2 um box, 25 nm cells, PEC disk between the points A and B (both in uncut cells) ----
_PAIR_CELL = 25e-9
_PAIR_SIZE = 2.0e-6
_PAIR_WAVELENGTH = 1e-6
_PAIR_DISK = (1.0123e-6, 0.9871e-6, 0.3137e-6)   # cx, cy, r: off the cell centres
_PAIR_POINTS = {'a': (0.4125e-6, 0.5125e-6), 'b': (1.5625e-6, 1.4625e-6)}
_PAIR_TOLERANCE = 1e-8                         # relative |F(B; A) - F(A; B)| (direct solve round-off)
_PAIR_MIN_CHANGE = 1e-3                        # conformal on must change F(B; A) against the staircase by more
_PAIR_SOURCES = {'ez': {'current': 1e-3, 'phase': 0.0},
                 'hz': {'excitation': 'magnetic_current', 'magnetic_current': 1.0, 'phase': 0.0}}


def _resonance_wavelength(polarization):
    """Vacuum wavelength [m] of the Bessel resonance of the disk: 2 pi R / j."""
    return 2.0 * math.pi * _RADIUS / _BESSEL_ZERO[polarization]


def _disk_spec(polarization, refinement, conformal):
    """em_fdfd_<polarization> eigenfrequency study (cpu) of the PEC disk cavity at the Bessel wavelength."""
    cx, cy = _CENTRE
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings['wavelength'] = _resonance_wavelength(polarization)
    return ModelSpec(
        title=f'Self-test: FDFD conformal PEC disk cavity ({polarization}, {refinement}, conformal {conformal})',
        domain=DomainSpec(size_x=_SIZE, size_y=_SIZE),
        geometry=[GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _SIZE, 'y1': _SIZE}),
                  GeometrySpec('disk', 'circle', {'cx': cx, 'cy': cy, 'r': _RADIUS}),
                  GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['disk']})],
        physics=[PhysicsSpec(_PHYSICS, f'em_fdfd_{polarization}', settings=settings,
                             features=[FeatureSpec('metal', 'pec_domain', ['metal']),
                                       FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_COARSE_CELL / _REFINEMENTS[refinement]),
        solver_defaults=SolverDefaultsSpec(backend='cpu', conformal_boundaries=conformal),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _N_RESONANCES}))


def _disk_runs():
    """{run name: (polarization, refinement, conformal)}: both refinements conformal, the coarse staircase too."""
    runs = {}
    for polarization in ('ez', 'hz'):
        for refinement in _REFINEMENTS:
            runs[f'{polarization}_{refinement}_on'] = (polarization, refinement, True)
        runs[f'{polarization}_coarse_off'] = (polarization, 'coarse', False)
    return runs


def _disk_specs():
    return {run: _disk_spec(*arguments) for run, arguments in _disk_runs().items()}


def _nearest_error(results, polarization):
    """|omega / omega_exact - 1| of the table resonance nearest c j / R, and the measured wavelengths."""
    table = results.result(_PHYSICS, _TABLE)
    columns = list(table['columns'])
    frequencies = [float(row[columns.index('frequency')]) for row in table['rows']]
    exact = C0 * _BESSEL_ZERO[polarization] / (2.0 * math.pi * _RADIUS)
    error = min((abs(frequency / exact - 1.0) for frequency in frequencies), default=math.inf)
    return error, [C0 / frequency for frequency in frequencies if frequency > 0.0]


def _check_disk(results_by_name):
    """Coarse conformal TM01 / TE11 within the coarse tolerance, fine within coarse tolerance / _FINE_FACTOR; observed
    orders and the staircase errors in the details. error = number of failed sub-checks, tolerance 0."""
    errors, wavelengths = {}, {}
    for run, (polarization, _, _) in _disk_runs().items():
        errors[run], wavelengths[run] = _nearest_error(results_by_name[run], polarization)
    checks, details = {}, {'errors': errors, 'wavelengths_m': wavelengths}
    for polarization in ('ez', 'hz'):
        mode = _MODE_NAMES[polarization]
        coarse, fine = errors[f'{polarization}_coarse_on'], errors[f'{polarization}_fine_on']
        checks[f'{mode}_coarse'] = (coarse, _COARSE_TOLERANCE[polarization])
        checks[f'{mode}_fine'] = (fine, _COARSE_TOLERANCE[polarization] / _FINE_FACTOR)
        details[f'{mode}_observed_order'] = (math.log2(coarse / fine) if coarse > 0.0 and fine > 0.0 else math.nan)
        details[f'{mode}_staircase_coarse'] = errors[f'{polarization}_coarse_off']
    entries = {name: {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}
               for name, (value, tolerance) in checks.items()}
    failed = [name for name, entry in entries.items() if not entry['passed']]
    details.update(failed=failed, checks=entries)
    return {'measured': float(len(failed)), 'expected': 0.0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': details}


def _pair_spec(polarization, conformal):
    """em_fdfd_<polarization> physics around the PEC disk (cpu): from_a drives A and probes B; the conformal run adds
    from_b, the reverse (the staircase run only feeds the change test)."""
    cx, cy, radius = _PAIR_DISK
    geometry = [GeometrySpec('disk', 'circle', {'cx': cx, 'cy': cy, 'r': radius})]
    geometry += [GeometrySpec(f'point_{name}', 'point', {'x': x, 'y': y}) for name, (x, y) in _PAIR_POINTS.items()]
    directions = (('a', 'b'), ('b', 'a')) if conformal else (('a', 'b'),)
    physics = [PhysicsSpec(f'from_{source}', f'em_fdfd_{polarization}', settings={'wavelength': _PAIR_WAVELENGTH},
                           features=[FeatureSpec('metal', 'pec_domain', ['disk']),
                                     FeatureSpec('source', 'point_source', [f'point_{source}'],
                                                 dict(_PAIR_SOURCES[polarization])),
                                     FeatureSpec('probe', 'field_probe', [f'point_{target}'],
                                                 {'component': polarization})])
               for source, target in directions]
    return ModelSpec(
        title=f'Self-test: em_fdfd_{polarization} reciprocity around a curved PEC disk (conformal {conformal})',
        domain=DomainSpec(size_x=_PAIR_SIZE, size_y=_PAIR_SIZE),
        geometry=geometry,
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_PAIR_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu', conformal_boundaries=conformal),
        study=StudySpec(type='stationary'))


def _pair_specs():
    return {f'{polarization}_{"on" if conformal else "off"}': _pair_spec(polarization, conformal)
            for polarization in ('hz', 'ez') for conformal in (True, False)}


def _check_pair(results_by_name):
    """Per polarisation |F(B; A) - F(A; B)| / |F(B; A)| of the conformal run (inf unless it differs from the staircase
    by more than _PAIR_MIN_CHANGE); error = the worst, tolerance _PAIR_TOLERANCE."""
    errors, details = {}, {}
    for polarization in ('hz', 'ez'):
        conformal, staircase = results_by_name[f'{polarization}_on'], results_by_name[f'{polarization}_off']
        forward, backward = probe_phasor(conformal, 'from_a', 'probe'), probe_phasor(conformal, 'from_b', 'probe')
        staircase_forward = probe_phasor(staircase, 'from_a', 'probe')
        change = abs(forward - staircase_forward) / abs(forward)
        error = abs(forward - backward) / abs(forward)
        errors[polarization] = error if change > _PAIR_MIN_CHANGE else math.inf
        details[polarization] = {'b_from_a': [forward.real, forward.imag], 'a_from_b': [backward.real, backward.imag],
                                 'staircase_b_from_a': [staircase_forward.real, staircase_forward.imag],
                                 'relative_change_by_conformal': change}
    worst = max(errors, key=lambda item: errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _PAIR_TOLERANCE,
            'details': {'worst': worst, 'errors': errors, 'min_change': _PAIR_MIN_CHANGE, **details}}


_DISK_TEXT = (f'vacuum disk of radius {_RADIUS * 1e9:g} nm (15.37 cells of {_COARSE_CELL * 1e9:g} nm, centre (0.31, '
              '0.17) cells off a cell centre) cut out of a PEC block (pec_domain, pec faces)')
FDFD_CONFORMAL_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_conformal_disk_eigen',
        label='EM FDFD: conformal PEC disk cavity eigenfrequencies',
        description=f'eigenfrequency study (cpu backend) of em_fdfd_ez / em_fdfd_hz in a {_DISK_TEXT}, '
                    'solver_defaults.conformal_boundaries on, at the coarse mesh and at half the cell size: the '
                    'resonance nearest c j / R (TM01 j = 2.4048, TE11 j = 1.8412) within '
                    f'{_COARSE_TOLERANCE["ez"]:g} / {_COARSE_TOLERANCE["hz"]:g} relative on the coarse mesh and '
                    f'within that / {_FINE_FACTOR:g} on the fine one (convergence; observed orders and the coarse '
                    'staircase errors in the details). Tolerances estimated, not measured. error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=_disk_specs,
        check=_check_disk),
    SelfTestCase(
        name='em_fdfd_conformal_reciprocity',
        label='EM FDFD: reciprocity around a conformal PEC disk',
        description='em_fdfd_hz and em_fdfd_ez (cpu backend), all-CPML 2 x 2 um box, 25 nm cells, a PEC disk of '
                    'radius 313.7 nm (centre off the cell centres) between two points A and B, '
                    'solver_defaults.conformal_boundaries on: physics from_a drives a magnetic line current (hz) / '
                    'electric line current (ez) at A and probes H_z / E_z at B, from_b the reverse; the staircase '
                    'run solves from_a only. Not yet confirmed by a run with these settings. Lorentz reciprocity F(B; A) = F(A; B) holds '
                    'discretely because the area-scaled conformal operator stays complex symmetric (one free fraction '
                    'per face, the free area as the k0^2 mu mass; E_z wall links change only the free diagonal). '
                    f'error = the worse |F(B; A) - F(A; B)| / |F(B; A)|, tolerance {_PAIR_TOLERANCE:g}; inf unless '
                    f'conformal changes F(B; A) by more than {_PAIR_MIN_CHANGE:g} against the staircase run.',
        specs=_pair_specs,
        check=_check_pair),
)

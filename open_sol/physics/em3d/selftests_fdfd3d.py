"""Self-tests of em_fdfd_3d (igor3d_plan stages 0-3): the 3D Yee operators of operators3d (curl identities, symmetry,
the discrete PEC box spectrum and its mirror sectors, CPU), the eigenfrequency model runs (dielectric and lossy PEC
boxes, the Bloch empty lattice, parity sectors, a k_z-swept Bragg stack cell), the FDTD ring-down of the same grid and
the host factor ordering on an eighth Mie sphere grid; templates pec_box_eigenmodes_3d and
nanobeam_mirror_cell_bands_3d. Model runs use the gpu backend with the cuDSS mixed-precision factor (_eigen_solvers);
only em3d_fdfd_eigen_pec_box also runs on the cpu (host SuperLU)."""
import cmath
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, SolverDefaultsSpec, SolverSpec, StudySpec, SweepSpec)
from .selftests_conformal3d import _failed_count_result, _leapfrog_frequency

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdfd_3d'          # literal (selftests never import the definition)
_FDTD_TYPE = 'em_fdtd_3d'
_TABLE = 'resonances'
_STUDY = 'eigenfrequency'
_CELL = 50e-9                         # m
_IDENTITY_TOLERANCE = 1e-12           # curl identities and symmetry, relative to the factor entries
_SPECTRUM_TOLERANCE = 1e-10           # dense eigenvalues vs the discrete closed form
_RUN_TOLERANCE = 1e-6                 # model runs (float32 cell widths ~3e-8, ARPACK at machine precision)
_ZERO_FRACTION = 1e-9                 # |eigenvalue| below this times the largest: a static (gradient) solution
_EIGEN_NODE = 'eigen'                 # model runs: gpu with the cuDSS mixed-precision factor, one small case also cpu


def _eigen_solvers(backend):
    """(ModelSpec solvers, PhysicsSpec solvers): an arpack_shift_invert node with factorization cudss_mixed on the gpu
    backend with nvmath installed, else none (Automatic: host SuperLU)."""
    from ...solvers.cudss_factor import cudss_available
    if backend != 'gpu' or not cudss_available():
        return [], {}
    return ([SolverSpec(_EIGEN_NODE, 'arpack_shift_invert', {'factorization': 'cudss_mixed'})],
            {'eigen': _EIGEN_NODE})


# ---- discrete closed forms ----
def _box_wavenumbers(cells, cell):
    """Sorted k^2 [1/m^2] with multiplicity of the Yee modes of a vacuum PEC box whose walls lie at the centres of cells
    0 and n - 1: indices m_a = 0 .. n_a - 2, k^2 = sum (2 / d sin(m_a pi / (2 (n_a - 1))))^2, two polarisations with
    every index >= 1, one with exactly one 0."""
    parts = [(2.0 / cell * np.sin(np.arange(count - 1) * math.pi / (2.0 * (count - 1)))) ** 2 for count in cells]
    squares = parts[0][:, None, None] + parts[1][None, :, None] + parts[2][None, None, :]
    zeros = sum((np.arange(count - 1) == 0).reshape(shape)
                for count, shape in zip(cells, ((-1, 1, 1), (1, -1, 1), (1, 1, -1))))
    multiplicity = np.where(zeros == 0, 2, np.where(zeros == 1, 1, 0))
    return np.sort(np.repeat(squares.ravel(), multiplicity.ravel()))


def _gradient_count(cells):
    """Static (gradient) solutions of the PEC box: one per interior centre potential."""
    return math.prod(count - 2 for count in cells)


def _nearest(values, target, count):
    """The `count` values nearest target (complex or real), sorted by distance."""
    values = sorted(values, key=lambda value: abs(value - target))
    return values[:count]


def _pairwise_error(found, expected):
    """max relative difference of two equally long sorted lists (inf when the lengths differ)."""
    if len(found) != len(expected) or not found:
        return math.inf
    found = sorted(found, key=lambda value: (value.real, value.imag) if isinstance(value, complex) else value)
    expected = sorted(expected, key=lambda value: (value.real, value.imag) if isinstance(value, complex) else value)
    return max(abs(a - b) / abs(b) for a, b in zip(found, expected))


# ---- stage 0: operators (CPU, check({})) ----
def _pec_faces(axes=(0, 1, 2)):
    return {f'{"xyz"[axis]}_{side}': 'pec' for axis in axes for side in ('lo', 'hi')}


def _relative_max(product, left, right):
    """max |product| over max |left| max |right| (a cancelling sparse identity: ~1e-16)."""
    scale = float(abs(left).max()) * float(abs(right).max())
    return float(abs(product).max()) / scale if scale > 0.0 else math.inf


def _check_identities(_results):
    """div curl_h = 0 and curl_e grad = 0 on graded grids (CPML at omega, Bloch, mirrors), the symmetry of diag(V) K on
    a real graded grid with PEC cells and both mirror kinds, and the complex symmetry of diag(V s) A with CPML at omega
    (the cuDSS LDL^T symmetrizer of fdfd3d)."""
    import scipy.sparse
    from ...solvers.cudss_factor import relative_asymmetry
    from .boundaries3d import BoundaryConfig3D
    from .operators3d import (axis_operator, curl_curl_operator, face_dual_volumes, face_stretches, live_unknowns_3d,
                              yee_operators_3d)
    rng = np.random.default_rng(3)
    widths = tuple(_CELL * (0.6 + 0.8 * rng.random(count)) for count in (7, 6, 5))
    omega = 2.0 * math.pi * C0 / 1.0e-6
    periodic_faces = {f'{axis}_{side}': 'periodic' for axis in 'xyz' for side in ('lo', 'hi')}
    configs = {
        'graded_cpml': (BoundaryConfig3D({'x_lo': 'cpml', 'x_hi': 'pec', 'y_lo': 'cpml', 'y_hi': 'cpml',
                                          'z_lo': 'pmc', 'z_hi': 'cpml'}, npml=2), omega),
        'bloch': (BoundaryConfig3D(periodic_faces, bloch_k=(0.7e6, -1.3e6, 2.1e6)), omega),
        'mirrors': (BoundaryConfig3D({'x_lo': 'antisymmetric', 'x_hi': 'pec', 'y_lo': 'symmetric', 'y_hi': 'pmc',
                                      'z_lo': 'periodic', 'z_hi': 'periodic'}), None),
    }
    checks, details = {}, {}
    for name, (config, config_omega) in configs.items():
        ops = yee_operators_3d(widths, config, (_CELL,) * 3, config_omega)
        gradient = scipy.sparse.vstack([axis_operator(pair[1], axis, ops.shape)
                                        for axis, pair in enumerate(ops.differences)]).tocsr()
        divergence = scipy.sparse.hstack([axis_operator(pair[0], axis, ops.shape)
                                          for axis, pair in enumerate(ops.differences)]).tocsr()
        checks[f'{name}_curl_grad'] = (_relative_max(ops.curl_e @ gradient, ops.curl_e, gradient),
                                       _IDENTITY_TOLERANCE)
        checks[f'{name}_div_curl'] = (_relative_max(divergence @ ops.curl_h, divergence, ops.curl_h),
                                      _IDENTITY_TOLERANCE)
    config = configs['mirrors'][0]
    shape = tuple(axis_widths.size for axis_widths in widths)
    pec = np.zeros(shape, dtype=bool)
    pec[3:5, 2:4, 1:3] = True
    window = (slice(1, shape[0]), slice(1, shape[1]), slice(0, shape[2]))   # x_lo / y_lo ghosts at index 0
    ops = yee_operators_3d(widths, config, (_CELL,) * 3)
    live = live_unknowns_3d(shape, config, window, pec)
    inverse_mu = np.tile(1.0 / (1.0 + rng.random(math.prod(shape))), 3)   # graded real mu
    volumes = face_dual_volumes(widths, (False, False, True))[live.full_index]
    volumes[live.plane_faces] *= 0.5
    scaled = scipy.sparse.diags(volumes) @ curl_curl_operator(ops, live, inverse_mu)
    checks['mirrors_symmetry'] = (float(abs(scaled - scaled.T).max()) / float(abs(scaled).max()),
                                  _IDENTITY_TOLERANCE)
    details['mirrors_unknowns'] = live.n_unknowns
    details['mirrors_plane_faces'] = int(np.count_nonzero(live.plane_faces))
    # the fdfd3d symmetrizer: stretched dual volumes (half on the antisymmetric plane) with CPML at omega, lossy eps
    cpml_config = BoundaryConfig3D({'x_lo': 'antisymmetric', 'x_hi': 'cpml', 'y_lo': 'symmetric', 'y_hi': 'cpml',
                                    'z_lo': 'pec', 'z_hi': 'cpml'}, npml=2)
    cpml_window = (slice(1, shape[0] - 2), slice(1, shape[1] - 2), slice(0, shape[2] - 2))
    cpml_ops = yee_operators_3d(widths, cpml_config, (_CELL,) * 3, omega)
    cpml_live = live_unknowns_3d(shape, cpml_config, cpml_window, pec)
    face_eps = np.tile(2.0 + rng.random(math.prod(shape)) - 0.3j * rng.random(math.prod(shape)), 3)
    operator = (scipy.sparse.diags((omega / C0) ** 2 * face_eps[cpml_live.full_index])
                - curl_curl_operator(cpml_ops, cpml_live, inverse_mu)).tocsr()
    volumes = face_dual_volumes(widths, (False,) * 3)[cpml_live.full_index]
    volumes[cpml_live.plane_faces] *= 0.5
    symmetrizer = volumes * face_stretches(shape, cpml_config, (_CELL,) * 3, omega)[cpml_live.full_index]
    checks['cpml_complex_symmetry'] = (relative_asymmetry(scipy.sparse.diags(symmetrizer) @ operator),
                                       _IDENTITY_TOLERANCE)
    details['cpml_asymmetry_without_stretches'] = relative_asymmetry(scipy.sparse.diags(volumes) @ operator)
    details['cpml_unknowns'] = cpml_live.n_unknowns
    return _failed_count_result(checks, details)


def _dense_spectrum(cells, faces, window):
    """(sorted nonzero real eigenvalues, number of static ones) of the vacuum K of a uniform grid (operators3d)."""
    from .boundaries3d import BoundaryConfig3D
    from .operators3d import curl_curl_operator, live_unknowns_3d, yee_operators_3d
    config = BoundaryConfig3D(faces)
    ops = yee_operators_3d(tuple(np.full(count, _CELL) for count in cells), config, (_CELL,) * 3)
    live = live_unknowns_3d(cells, config, window)
    values = np.linalg.eigvals(curl_curl_operator(ops, live).toarray())
    static = np.abs(values) <= _ZERO_FRACTION * float(np.max(np.abs(values)))
    return np.sort(values[~static].real), int(np.count_nonzero(static)), float(np.max(np.abs(values[~static].imag)))


_BOX_CELLS = (5, 6, 7)
_SECTOR_FULL_CELLS = (6, 6, 5)        # even x / y: the mirror planes x = 3 dx, y = 3 dy are cell faces


def _check_pec_box_discrete(_results):
    """Dense spectrum of the PEC box vs the discrete closed form (values, static count); the four mirror sectors of a
    quarter box (x_lo / y_lo symmetric or antisymmetric) together vs the full box."""
    checks, details = {}, {}
    full_window = tuple(slice(0, count) for count in _BOX_CELLS)
    values, n_static, imaginary = _dense_spectrum(_BOX_CELLS, _pec_faces(), full_window)
    expected = _box_wavenumbers(_BOX_CELLS, _CELL)
    error = (float(np.max(np.abs(values / expected - 1.0))) if values.size == expected.size else math.inf)
    checks['box_spectrum'] = (error, _SPECTRUM_TOLERANCE)
    checks['box_static_count'] = (abs(n_static - _gradient_count(_BOX_CELLS)), 0)
    details['box'] = {'n_modes': int(values.size), 'expected_modes': int(expected.size), 'n_static': n_static,
                      'expected_static': _gradient_count(_BOX_CELLS), 'largest_imaginary': imaginary}
    sector_full = tuple(slice(0, count) for count in _SECTOR_FULL_CELLS)
    full_values, full_static, _ = _dense_spectrum(_SECTOR_FULL_CELLS, _pec_faces(), sector_full)
    quarter = (_SECTOR_FULL_CELLS[0] // 2 + 1, _SECTOR_FULL_CELLS[1] // 2 + 1, _SECTOR_FULL_CELLS[2])
    quarter_window = (slice(1, quarter[0]), slice(1, quarter[1]), slice(0, quarter[2]))
    union, union_static, sectors = [], 0, {}
    for x_kind in ('symmetric', 'antisymmetric'):
        for y_kind in ('symmetric', 'antisymmetric'):
            faces = {**_pec_faces(), 'x_lo': x_kind, 'y_lo': y_kind}
            sector_values, sector_static, _ = _dense_spectrum(quarter, faces, quarter_window)
            union.extend(sector_values.tolist())
            union_static += sector_static
            sectors[f'x {x_kind}, y {y_kind}'] = {'n_modes': int(sector_values.size), 'n_static': sector_static}
    union = np.sort(np.asarray(union))
    checks['sector_union'] = ((float(np.max(np.abs(union / full_values - 1.0))) if union.size == full_values.size
                               else math.inf), _SPECTRUM_TOLERANCE)
    checks['sector_static_count'] = (abs(union_static - full_static), 0)
    details['sectors'] = {'full_modes': int(full_values.size), 'union_modes': int(union.size),
                          'full_static': full_static, 'union_static': union_static, 'per_sector': sectors}
    return _failed_count_result(checks, details)


# ---- stage 1: model runs ----
_RUN_CELLS = (10, 8, 6)               # PEC box, walls at the centres of cells 0 and n - 1
_RUN_EPS = 2.25
_RUN_SIGMA = 1.0e3                    # S/m: eps_c = 2.25 - 0.05 j at the target
_RUN_RESONANCES = 4
_LOWEST_INDICES = (1, 1, 0)           # the target of the box runs: this mode's lossless wavelength (~0.835 um)


def _mode_wavelength(cells, indices, eps, cell=_CELL):
    """Vacuum wavelength [m] of the lossless discrete box mode `indices` in a medium eps."""
    squared = sum((2.0 / cell * math.sin(index * math.pi / (2.0 * (count - 1)))) ** 2
                  for index, count in zip(indices, cells))
    return 2.0 * math.pi * math.sqrt(eps) / math.sqrt(squared)


_BOX_TARGET = 0.97 * _mode_wavelength(_RUN_CELLS, _LOWEST_INDICES, _RUN_EPS)   # off the mode: a regular factor


def _box_spec(title, backend='gpu', sigma_e=0.0, wavelength=_BOX_TARGET, cells=_RUN_CELLS, faces=None,
              parity_sectors=False, n_resonances=_RUN_RESONANCES):
    """em_fdfd_3d PEC box (faces override bc entries) filled with eps 2.25 (+ sigma_e), a resonance table."""
    settings = {f'bc_{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')}
    settings.update(faces or {})
    settings['wavelength'] = wavelength
    properties = {'eps_r': _RUN_EPS, **({'sigma_e': sigma_e} if sigma_e else {})}
    solvers, assigned = _eigen_solvers(backend)
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=cells[0] * _CELL, size_y=cells[1] * _CELL, size_z=cells[2] * _CELL),
        materials=[MaterialSpec('fill', properties=properties, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')], solvers=assigned)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solvers=solvers,
        solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type=_STUDY, settings={'n_resonances': n_resonances, 'parity_sectors': parity_sectors}))


def _table_omegas(table):
    """Complex omega [rad/s] of every row of a resonance table."""
    columns = list(table['columns'])
    return [2.0 * math.pi * complex(row[columns.index('frequency')], row[columns.index('frequency_imag')])
            for row in table['rows']]


def _add_band_checks(checks, run, found, expected, count):
    """checks[run]: Re of the found omegas vs the real `expected` ones (inf unless `count` were found);
    checks[run_inverse_q]: their largest 1/Q = 2 |Im| / Re (lossless: ~0)."""
    checks[run] = (_pairwise_error([omega.real for omega in found], expected) if len(found) == count else math.inf,
                   _RUN_TOLERANCE)
    checks[f'{run}_inverse_q'] = (max((2.0 * abs(omega.imag) / omega.real for omega in found), default=math.inf),
                                  _RUN_TOLERANCE)


def _box_omegas(cells, frozen_eps):
    """Complex omega of every box mode in a medium of frozen complex eps."""
    return [C0 * math.sqrt(value) / cmath.sqrt(frozen_eps) for value in _box_wavenumbers(cells, float(np.float32(_CELL)))]


def _box_specs():
    return {backend: _box_spec(f'Self-test: em_fdfd_3d dielectric PEC box ({backend})', backend)
            for backend in ('cpu', 'gpu')}


def _check_box(results_by_name):
    """Per run present: the found omegas vs the closed-form ones nearest the target (lossless: 1/Q ~ 0)."""
    omega_target = 2.0 * math.pi * C0 / _BOX_TARGET
    expected_all = _box_omegas(_RUN_CELLS, _RUN_EPS)
    checks, details = {}, {}
    for run, results in results_by_name.items():
        found = _table_omegas(results.result(_PHYSICS_NAME, _TABLE))
        expected = _nearest(expected_all, omega_target, len(found))
        _add_band_checks(checks, run, found, [omega.real for omega in expected], _RUN_RESONANCES)
        details[run] = {'found_wavelengths': [2.0 * math.pi * C0 / omega.real for omega in found],
                        'expected_wavelengths': [2.0 * math.pi * C0 / omega.real for omega in expected]}
    if not results_by_name:
        checks['runs'] = (math.inf, 0.0)
    return _failed_count_result(checks, details)


def _lossy_target():
    return _mode_wavelength(_RUN_CELLS, _LOWEST_INDICES, _RUN_EPS)   # on the lossless mode: the frozen eps is exact


def _check_lossy_box(results_by_name):
    """Complex omegas vs C0 sqrt(k^2 / eps_c), eps_c = 2.25 - j sigma_e / (omega_t eps0) frozen at the target."""
    omega_target = 2.0 * math.pi * C0 / _lossy_target()
    frozen_eps = _RUN_EPS - 1j * _RUN_SIGMA / (omega_target * EPS0)
    found = _table_omegas(results_by_name['run'].result(_PHYSICS_NAME, _TABLE))
    expected = _nearest(_box_omegas(_RUN_CELLS, frozen_eps), omega_target, len(found))
    error = _pairwise_error(found, expected) if len(found) == _RUN_RESONANCES else math.inf
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _RUN_TOLERANCE,
            'details': {'found': [[omega.real, omega.imag] for omega in found],
                        'expected': [[omega.real, omega.imag] for omega in expected],
                        'expected_q': [omega.real / (2.0 * omega.imag) for omega in expected]}}


_LATTICE_CELLS = 6                    # per axis, periodic on all six faces
_LATTICE_PERIOD = _LATTICE_CELLS * _CELL
_LATTICE_RUNS = {'k_quarter': 0.25, 'k_edge': 0.5}   # bloch_kx in units of 2 pi / L
_LATTICE_RESONANCES = 6


def _lattice_omegas(k_fraction, cell):
    """Complex omegas (2 per plane wave) of the vacuum Yee lattice at Bloch k_x = k_fraction 2 pi / L."""
    orders = np.arange(_LATTICE_CELLS) - _LATTICE_CELLS // 2 + 1
    g = 2.0 * math.pi / (_LATTICE_CELLS * cell)
    parts = [(2.0 / cell * np.sin(0.5 * (shift + orders * g) * cell)) ** 2 for shift in (k_fraction * g, 0.0, 0.0)]
    squares = (parts[0][:, None, None] + parts[1][None, :, None] + parts[2][None, None, :]).ravel()
    return [complex(C0 * math.sqrt(value)) for value in np.repeat(squares[squares > 0.0], 2)]


def _lattice_target(k_fraction):
    """Between the lowest band (the k_x plane wave) and the next: 1.2 x the lowest frequency."""
    lowest = min(omega.real for omega in _lattice_omegas(k_fraction, _CELL))
    return 2.0 * math.pi * C0 / (1.2 * lowest)


def _lattice_spec(k_fraction):
    """Vacuum unit cell periodic on all six faces at bloch_kx, a resonance table."""
    settings = {f'bc_{axis}_{side}': 'periodic' for axis in 'xyz' for side in ('lo', 'hi')}
    settings.update(bloch_kx=k_fraction * 2.0 * math.pi / _LATTICE_PERIOD, wavelength=_lattice_target(k_fraction))
    solvers, assigned = _eigen_solvers('gpu')
    return ModelSpec(
        title=f'Self-test: em_fdfd_3d Bloch empty lattice (k = {k_fraction:g} 2 pi / L)',
        domain=DomainSpec(size_x=_LATTICE_PERIOD, size_y=_LATTICE_PERIOD, size_z=_LATTICE_PERIOD),
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')], solvers=assigned)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solvers=solvers,
        solver_defaults=SolverDefaultsSpec(backend='gpu'),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _LATTICE_RESONANCES}))


def _check_lattice(results_by_name):
    """Per k: the found omegas vs the discrete plane-wave bands nearest the target (Q = inf)."""
    checks, details = {}, {}
    for run, k_fraction in _LATTICE_RUNS.items():
        found = _table_omegas(results_by_name[run].result(_PHYSICS_NAME, _TABLE))
        omega_target = 2.0 * math.pi * C0 / _lattice_target(k_fraction)
        expected = _nearest(_lattice_omegas(k_fraction, float(np.float32(_CELL))), omega_target, len(found))
        _add_band_checks(checks, run, found, [omega.real for omega in expected], _LATTICE_RESONANCES)
        details[run] = {'found': [omega.real for omega in found], 'expected': [omega.real for omega in expected]}
    return _failed_count_result(checks, details)


_SECTOR_RESONANCES = 4                # per sector; the full box's 4 nearest must be among them


def _sector_specs():
    """'quarter': the run box cut by x_lo / y_lo mirrors (parity_sectors: four sectors)."""
    quarter = (_RUN_CELLS[0] // 2, _RUN_CELLS[1] // 2, _RUN_CELLS[2])   # mirrors at the centre faces of the box
    return {'quarter': _box_spec('Self-test: em_fdfd_3d PEC box parity sectors', cells=quarter,
                                 faces={'bc_x_lo': 'symmetric', 'bc_y_lo': 'antisymmetric'}, parity_sectors=True,
                                 n_resonances=_SECTOR_RESONANCES)}


def _check_sectors(results_by_name):
    """Every sector's omegas are full-box modes; the union's nearest four are the full box's nearest four."""
    omega_target = 2.0 * math.pi * C0 / _BOX_TARGET
    expected_all = [omega.real for omega in _box_omegas(_RUN_CELLS, _RUN_EPS)]
    tables = results_by_name['quarter'].sector_result(_PHYSICS_NAME, _TABLE)
    labels = results_by_name['quarter'].summary().get('parity_sectors', {}).get('labels', [])
    checks, details, union = {}, {}, []
    for index, table in enumerate(tables):
        label = labels[index] if index < len(labels) else f'sector {index + 1}'
        found = [] if table is None else [omega.real for omega in _table_omegas(table)]
        union.extend(found)
        mismatch = max((min(abs(value / reference - 1.0) for reference in expected_all) for value in found),
                       default=math.inf)
        checks[f'sector {index + 1}'] = (mismatch, _RUN_TOLERANCE)
        details[label] = [2.0 * math.pi * C0 / value for value in found]
    nearest_expected = _nearest(expected_all, omega_target, _SECTOR_RESONANCES)
    checks['union'] = (_pairwise_error(_nearest(union, omega_target, _SECTOR_RESONANCES), nearest_expected),
                       _RUN_TOLERANCE)
    details['expected_nearest_wavelengths'] = [2.0 * math.pi * C0 / value for value in nearest_expected]
    return _failed_count_result(checks, details)


# ---- FDTD ring-down vs the FDFD eigenvalues on one grid (heavy) ----
_RING_CELLS = (22, 18, 14)            # the em3d_resonance_pec_box grid with a dielectric block
_RING_BLOCK = ((6, 12), (5, 10), (3, 8))   # cells of the eps 4 block (cell edges)
_RING_EPS = 4.0
_RING_BAND = (1.0e-6, 2.2e-6)         # m
_RING_TARGET = 1.5e-6
_RING_RESONANCES = 16
_RING_TOLERANCE = 1e-4                # leap-frog frequency of the FDFD eigenvalue vs the FDTD fit
_RING_DURATION = 0.4e-12


def _ring_geometry():
    block = {f'{axis}{end}': cells[end] * _CELL for axis, cells in zip('xyz', _RING_BLOCK) for end in (0, 1)}
    return [GeometrySpec('block', 'box', block),
            GeometrySpec('source_point', 'point', {'x': 5.5 * _CELL, 'y': 4.5 * _CELL, 'z': 3.8 * _CELL}),
            GeometrySpec('probe_point', 'point', {'x': 15.5 * _CELL, 'y': 12.5 * _CELL, 'z': 9.5 * _CELL})]


def _ring_specs():
    """'fdtd': em_fdtd_3d ring-down (z dipole pulse, resonance_monitor E_z) and 'fdfd': em_fdfd_3d eigenfrequencies of
    the same PEC box with the block, both on the gpu backend."""
    domain = DomainSpec(size_x=_RING_CELLS[0] * _CELL, size_y=_RING_CELLS[1] * _CELL, size_z=_RING_CELLS[2] * _CELL)
    materials = [MaterialSpec('block', properties={'eps_r': _RING_EPS}, selection=['block'])]
    pec = {f'bc_{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')}
    fdtd = ModelSpec(
        title='Self-test: em_fdfd_3d vs em_fdtd_3d (FDTD ring-down)', domain=domain, geometry=_ring_geometry(),
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, _FDTD_TYPE, settings=pec, features=[
            FeatureSpec('dipole', 'dipole_source', ['source_point'],
                        {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': 1.5e-6, 'bandwidth': 0.6,
                         'current_moment': 1e-12}),
            FeatureSpec(_TABLE, 'resonance_monitor', ['probe_point'],
                        {'component': 'ez', 'wavelength_min': _RING_BAND[0], 'wavelength_max': _RING_BAND[1]})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL), solver_defaults=SolverDefaultsSpec(backend='gpu'),
        study=StudySpec(type='time_dependent', settings={'duration': _RING_DURATION}))
    solvers, assigned = _eigen_solvers('gpu')
    fdfd = ModelSpec(
        title='Self-test: em_fdfd_3d vs em_fdtd_3d (FDFD eigenfrequencies)', domain=domain, geometry=_ring_geometry(),
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings={**pec, 'wavelength': _RING_TARGET},
                             features=[FeatureSpec(_TABLE, 'resonance_table')], solvers=assigned)],
        mesh=MeshSpec(mode='uniform', dx=_CELL), solvers=solvers, solver_defaults=SolverDefaultsSpec(backend='gpu'),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _RING_RESONANCES}))
    return {'fdtd': fdtd, 'fdfd': fdfd}


def _check_ring(results_by_name):
    """Each FDTD frequency vs the nearest leap-frog frequency asin(dt omega / 2) / (pi dt) of the FDFD omegas."""
    ring = results_by_name['fdtd'].result(_PHYSICS_NAME, _TABLE)
    columns = list(ring['columns'])
    found = [float(row[columns.index('frequency')]) for row in ring['rows']]
    dt = float(ring['dt'])
    eigen = _table_omegas(results_by_name['fdfd'].result(_PHYSICS_NAME, _TABLE))
    predicted = [_leapfrog_frequency(omega.real ** 2, dt) for omega in eigen]
    errors = [min(abs(value / reference - 1.0) for reference in predicted) for value in found] if predicted else []
    error = max(errors) if len(errors) >= 2 else math.inf
    return {'measured': float(len(found)), 'expected': float(len(found)), 'error': error, 'tolerance': _RING_TOLERANCE,
            'details': {'fdtd_frequencies': found, 'fdfd_leapfrog_frequencies': predicted, 'errors': errors,
                        'note': 'at least two FDTD modes are needed'}}


# ---- eighth Mie sphere grid of the factor-ordering case (model run: example fdfd_mie_sphere_resonance_3d) ----
_MIE_RADIUS = 200e-9                  # m
_MIE_N_SPHERE = 3.47                  # silicon in D2O (igor_experiment 4.2)
_MIE_N_BACKGROUND = 1.32
_MIE_TARGET = 1.49e-6                 # m: the TE_1 (magnetic dipole) pole lies at ~1.486 um, Q ~6.1
_MIE_CELL = 25e-9                     # 8 cells per radius, ~17 per wavelength in silicon
# eighth model: the sphere centre on the domain corner. The m_z magnetic dipole (E azimuthal about z) is E-odd about
# x = 0 and y = 0 (tangential E 0: antisymmetric) and E-even about z = 0 (symmetric); no electric dipole shares it.
_MIE_FACES = {'bc_x_lo': 'antisymmetric', 'bc_y_lo': 'antisymmetric', 'bc_z_lo': 'symmetric',
              'bc_x_hi': 'cpml', 'bc_y_hi': 'cpml', 'bc_z_hi': 'cpml'}


# ---- stage 3: Bloch unit cell of a Bragg stack, k_z swept (igor_experiment 4.3) ----
# One period of a Si / D2O stack along z (Si 110 nm, D2O 280 nm: near quarter-wave at 1.5 um, gap ~1.16-2.13 um at the
# zone edge) in a 2 x 2 cell x / y periodic column at k_t = 0: the normal-incidence E_x / E_y bands (two polarisations
# per band) of the Yee chain E_{k+1} = (2 - (k0 n_k d)^2) E_k - E_{k-1} on the z cell centres, whose one-period
# transfer matrix T has trace T = 2 cos(k_z a) (det = 1). Transverse orders lie at k ~ 2 / dx, far away.
_BRAGG_CELL = 10e-9                   # m, uniform
_BRAGG_LAYERS = ((3.47, 11), (1.32, 28))   # (refractive index, cells) per period, the Si layer at z = 0
_BRAGG_TRANSVERSE_CELLS = 2
_BRAGG_K_FRACTIONS = (0.25, 0.5)      # bloch_kz in units of 2 pi / a: mid-zone and zone edge
_BRAGG_TARGET = 1.5e-6                # m: between the first two bands at both k (the gap centre at the edge)
_BRAGG_RESONANCES = 4                 # the two bands nearest the target, two polarisations each
_BRAGG_CONTINUUM_TOLERANCE = 1e-2     # discrete vs continuum transfer matrix (Yee dispersion ~2e-3 in Si)
_BRAGG_GRID = 6000                    # k0 samples of the root bracketing up to 3 x the target k0


def _bragg_period():
    """Stack period [m] of the parameter a (the meshed period differs at float32 round-off)."""
    return sum(cells for _, cells in _BRAGG_LAYERS) * _BRAGG_CELL


def _discrete_trace(k0, cell):
    """trace of the one-period transfer matrix of the Yee chain at vacuum wavenumber k0."""
    transfer = np.eye(2)
    for index, cells in _BRAGG_LAYERS:
        step = np.array([[2.0 - (k0 * index * cell) ** 2, -1.0], [1.0, 0.0]])
        transfer = np.linalg.matrix_power(step, cells) @ transfer
    return float(np.trace(transfer))


def _continuum_trace(k0, cell):
    """trace of the continuum two-layer transfer matrix (normal incidence) at vacuum wavenumber k0."""
    (index_a, cells_a), (index_b, cells_b) = _BRAGG_LAYERS
    phase_a, phase_b = index_a * k0 * cells_a * cell, index_b * k0 * cells_b * cell
    return 2.0 * (math.cos(phase_a) * math.cos(phase_b)
                  - 0.5 * (index_a / index_b + index_b / index_a) * math.sin(phase_a) * math.sin(phase_b))


def _bragg_omegas(k_fraction, trace):
    """omegas [rad/s] (each twice: E_x and E_y) of the bands at bloch_kz = k_fraction 2 pi / a: roots of trace(k0) =
    2 cos(k_z L) (L the meshed period), bracketed on a k0 grid and refined by brentq."""
    from scipy.optimize import brentq
    cell = float(np.float32(_BRAGG_CELL))
    meshed_period = sum(cells for _, cells in _BRAGG_LAYERS) * cell
    bloch_trace = 2.0 * math.cos(k_fraction * 2.0 * math.pi * meshed_period / _bragg_period())
    k0_max = 3.0 * 2.0 * math.pi / _BRAGG_TARGET
    grid = np.linspace(k0_max / _BRAGG_GRID, k0_max, _BRAGG_GRID)
    values = np.array([trace(k0, cell) - bloch_trace for k0 in grid])
    roots = [brentq(lambda k0: trace(k0, cell) - bloch_trace, grid[index], grid[index + 1], xtol=1e-13 * k0_max)
             for index in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))]
    return [C0 * root for root in roots for _ in range(2)]


def _bragg_spec():
    """2 x 2 cell column, x / y periodic at k_t = 0, z periodic at bloch_kz = k_fraction 2 pi / a swept over
    _BRAGG_K_FRACTIONS; derived frequency / frequency_imag of the first _BRAGG_RESONANCES table rows (gpu)."""
    side = _BRAGG_TRANSVERSE_CELLS * _BRAGG_CELL
    settings = {f'bc_{axis}_{end}': 'periodic' for axis in 'xyz' for end in ('lo', 'hi')}
    settings.update(bloch_kz='k_fraction * 2 * pi / a', wavelength=_BRAGG_TARGET)
    (index_a, cells_a), (index_b, _) = _BRAGG_LAYERS
    layer = {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'x1': side, 'y1': side, 'z1': cells_a * _BRAGG_CELL}
    derived = [DerivedValueSpec(f'{quantity}_{row + 1}', quantity, [], 'at', source=f'{_PHYSICS_NAME}.{_TABLE}', at=row)
               for row in range(_BRAGG_RESONANCES) for quantity in ('frequency', 'frequency_imag')]
    solvers, assigned = _eigen_solvers('gpu')
    return ModelSpec(
        title='Self-test: em_fdfd_3d Bragg stack unit cell (k_z sweep)',
        parameters=[ParameterSpec('a', _bragg_period(), 'stack period [m]'),
                    ParameterSpec('k_fraction', _BRAGG_K_FRACTIONS[-1], 'Bloch kz in units of 2 pi / a')],
        domain=DomainSpec(size_x=side, size_y=side, size_z=_bragg_period()),
        geometry=[GeometrySpec('silicon_layer', 'box', layer)],
        materials=[MaterialSpec('d2o', properties={'eps_r': index_b ** 2}, selection=['all']),
                   MaterialSpec('silicon', properties={'eps_r': index_a ** 2}, selection=['silicon_layer'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')], solvers=assigned)],
        mesh=MeshSpec(mode='uniform', dx=_BRAGG_CELL),
        solvers=solvers,
        solver_defaults=SolverDefaultsSpec(backend='gpu', subpixel_smoothing=False),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _BRAGG_RESONANCES},
                        sweep=SweepSpec('k_fraction', list(_BRAGG_K_FRACTIONS))),
        derived_values=derived)


def _swept_omegas(derived, value_index):
    """Finite complex omegas of the derived table rows at one sweep value."""
    omegas = []
    for row in range(1, _BRAGG_RESONANCES + 1):
        real, imaginary = derived.get(f'frequency_{row}') or [], derived.get(f'frequency_imag_{row}') or []
        if value_index < min(len(real), len(imaginary)):
            omega = 2.0 * math.pi * complex(float(real[value_index]), float(imaginary[value_index]))
            if math.isfinite(omega.real) and math.isfinite(omega.imag):
                omegas.append(omega)
    return omegas


def _check_bragg(results_by_name):
    """Sweep layout; per k: the found omegas vs the discrete transfer-matrix bands nearest the target, 1/Q ~ 0, and
    those discrete bands vs the continuum ones (the oracle's own sanity)."""
    sweep = results_by_name['sweep'].summary().get('sweep') or {}
    derived = sweep.get('derived') or {}
    values = [float(value) for value in sweep.get('values') or []]
    omega_target = 2.0 * math.pi * C0 / _BRAGG_TARGET
    checks = {'sweep_values': (0.0 if values == list(_BRAGG_K_FRACTIONS) else math.inf, 0.0)}
    details = {'sweep_values': values}
    for value_index, k_fraction in enumerate(_BRAGG_K_FRACTIONS):
        run = f'k_{k_fraction:g}'
        found = _swept_omegas(derived, value_index)
        expected = _nearest(_bragg_omegas(k_fraction, _discrete_trace), omega_target, _BRAGG_RESONANCES)
        continuum = _bragg_omegas(k_fraction, _continuum_trace)
        _add_band_checks(checks, run, found, expected, _BRAGG_RESONANCES)
        checks[f'{run}_continuum'] = (max((min(abs(value / reference - 1.0) for reference in continuum)
                                           for value in expected), default=math.inf) if continuum else math.inf,
                                      _BRAGG_CONTINUUM_TOLERANCE)
        details[run] = {'found_wavelengths': [2.0 * math.pi * C0 / omega.real for omega in found],
                        'discrete_wavelengths': [2.0 * math.pi * C0 / omega for omega in expected],
                        'continuum_wavelengths': [2.0 * math.pi * C0 / omega
                                                  for omega in _nearest(continuum, omega_target, _BRAGG_RESONANCES)]}
    return _failed_count_result(checks, details)


# ---- host factor ordering: minimum degree vs nested dissection (heavy, cpu) ----
_ORDERING_CELLS = 20                  # per axis incl. the mirror ghost and the CPML: ~2e4 unknowns (24: 3.6e4
                                      # unknowns, MMD 76 s / ND 37 s, fill ratio 0.84; factor time ~ cells^6)
_ORDERING_PML_CELLS = 6
_ORDERING_RESIDUAL = 1e-8             # ||A x - b|| / ||b|| of either factor
_ORDERING_FILL_RATIO = 1.0            # nested-dissection L + U entries / minimum-degree ones


def _eighth_sphere_system(count, n_pml, cell):
    """(A, live) of the eighth Mie sphere (_MIE_FACES, staircase eps with each face its own cell's, uniform cells;
    count per axis incl. the mirror ghost and n_pml CPML cells) at _MIE_TARGET."""
    import scipy.sparse
    from .boundaries3d import BoundaryConfig3D
    from .operators3d import curl_curl_operator, live_unknowns_3d, yee_operators_3d
    config = BoundaryConfig3D({face[3:]: kind for face, kind in _MIE_FACES.items()}, npml=n_pml)
    omega = 2.0 * math.pi * C0 / _MIE_TARGET
    shape = (count,) * 3
    ops = yee_operators_3d(tuple(np.full(count, cell) for _ in shape), config, (cell,) * 3, omega)
    live = live_unknowns_3d(shape, config, (slice(1, count - n_pml),) * 3)
    centres = np.arange(count) - 0.5      # cell centres in cells from the mirror planes (ghost at -0.5)
    radius_squared = centres[:, None, None] ** 2 + centres[None, :, None] ** 2 + centres[None, None, :] ** 2
    cell_eps = np.where(radius_squared < (_MIE_RADIUS / cell) ** 2, _MIE_N_SPHERE ** 2, _MIE_N_BACKGROUND ** 2)
    mass = np.tile(cell_eps.ravel(), 3)[live.full_index]
    operator = (scipy.sparse.diags((omega / C0) ** 2 * mass) - curl_curl_operator(ops, live)).tocsr()
    return operator, live


def _check_ordering(_results):
    """SuperLU of the A of an eighth Mie sphere grid (_MIE_FACES, staircase eps) with its own minimum degree and with
    the nested-dissection pre-ordering: residuals and the fill ratio checked, seconds in the details."""
    import time
    from ...solvers.sparse_direct import factorize_host
    from .operators3d import nested_dissection_ordering
    operator, live = _eighth_sphere_system(_ORDERING_CELLS, _ORDERING_PML_CELLS, _MIE_CELL)
    rhs = np.random.default_rng(5).standard_normal(live.n_unknowns).astype(np.complex128)
    checks, details = {}, {'unknowns': live.n_unknowns}
    for name, ordering in (('minimum_degree', None), ('nested_dissection', nested_dissection_ordering(live))):
        started = time.perf_counter()
        factor = factorize_host(operator, ordering=ordering)
        factored = time.perf_counter()
        solution = factor.solve_host(rhs)
        residual = float(np.linalg.norm(operator @ solution - rhs) / np.linalg.norm(rhs))
        details[name] = {'factor_seconds': factored - started, 'solve_seconds': time.perf_counter() - factored,
                         'lu_bytes': factor.host_bytes, 'residual': residual}
        checks[f'{name}_residual'] = (residual, _ORDERING_RESIDUAL)
        del factor                        # one L + U on the host at a time
    minimum_degree, nested = details['minimum_degree'], details['nested_dissection']
    details['speedup'] = minimum_degree['factor_seconds'] / max(nested['factor_seconds'], 1e-9)
    checks['fill_ratio'] = (nested['lu_bytes'] / minimum_degree['lu_bytes'], _ORDERING_FILL_RATIO)
    return _failed_count_result(checks, details)


# ---- template ----
PEC_BOX_EIGENMODES_KEY = 'pec_box_eigenmodes_3d'
PEC_BOX_EIGENMODES_LABEL = 'PEC box cavity with a silicon cube: eigenmodes (3D)'
PEC_BOX_EIGENMODES_DESCRIPTION = ('em_fdfd_3d: a 1.0 x 0.8 x 0.6 um vacuum box with pec walls on all six faces and a '
                                  '0.3 um eps 12 cube at its centre; the eigenfrequency study finds the 4 resonances '
                                  'nearest 1.4 um (resonance table: wavelength, Q = inf for this lossless closed '
                                  'cavity) and their mode shapes on 40 nm cells (10 per wavelength in silicon).')
_TEMPLATE_SIZE = (1.0e-6, 0.8e-6, 0.6e-6)
_TEMPLATE_CUBE = 0.3e-6
_TEMPLATE_CELL = 40e-9


def pec_box_eigenmodes_spec():
    """Template 'pec_box_eigenmodes_3d': PEC box with a silicon cube, em_fdfd_3d eigenfrequencies near 1.4 um."""
    cube = {f'{axis}{end}': 0.5 * size + (0.5 if end else -0.5) * _TEMPLATE_CUBE
            for axis, size in zip('xyz', _TEMPLATE_SIZE) for end in (0, 1)}
    return ModelSpec(
        title=PEC_BOX_EIGENMODES_LABEL,
        description=PEC_BOX_EIGENMODES_DESCRIPTION,
        domain=DomainSpec(size_x=_TEMPLATE_SIZE[0], size_y=_TEMPLATE_SIZE[1], size_z=_TEMPLATE_SIZE[2]),
        geometry=[GeometrySpec('cube', 'box', cube)],
        materials=[MaterialSpec('silicon_cube', properties={'eps_r': 12.0}, selection=['cube'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE,
                             settings={**{f'bc_{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')},
                                       'wavelength': 1.4e-6},
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_TEMPLATE_CELL),
        study=StudySpec(type=_STUDY, settings={'n_resonances': 4}))


PEC_BOX_EIGENMODES_TEMPLATE = (PEC_BOX_EIGENMODES_KEY, PEC_BOX_EIGENMODES_LABEL, PEC_BOX_EIGENMODES_DESCRIPTION,
                               pec_box_eigenmodes_spec,
                               'Resonances of a closed metal box with a small silicon cube inside, found directly: '
                               'wavelength, quality factor and the 3D field pattern of each mode.')


# 'nanobeam_mirror_cell_bands_3d' template (igor_experiment 4.3, the Sung et al. mirror cell): one period of the silicon
# nanobeam in D2O, holes along y through the beam, quarter cross-section. The TE-like fundamental (E_x dominant, even in
# x and y) is E-odd about x = 0 (tangential E_y / E_z odd: antisymmetric, a PEC plane) and E-even about y = 0
# (symmetric, PMC). At bloch_kz >= 0.4 x 2 pi / a the bands near 1.5 um lie below the D2O light line (guided).
# dx = a / 16 so the period is whole cells (a uniform mesh has ceil(size / dx) cells).
NANOBEAM_CELL_KEY = 'nanobeam_mirror_cell_bands_3d'
NANOBEAM_CELL_LABEL = 'Nanobeam mirror cell band diagram (3D, frequency domain)'
NANOBEAM_CELL_DESCRIPTION = (
    'em_fdfd_3d: one mirror cell of a silicon nanobeam photonic crystal in D2O (igor_experiment 4.3: period a = 394.3 '
    'nm, hole d = 279.3 nm along y through a 595.2 x 220 nm beam, n 3.47 / 1.32), quarter cross-section (x_lo '
    'antisymmetric = PEC plane, y_lo symmetric = PMC plane: the TE-like sector), 300 nm D2O and 8 CPML cells on x_hi / '
    'y_hi, z periodic with bloch_kz = k_fraction 2 pi / a swept over 0.4-0.5 (guided, below the D2O light line); an '
    'eigenfrequency study per k lists the 4 bands nearest 1.496 um (resonance_table). At k_fraction 0.5 the dielectric '
    '(longer wavelength) and air band edges bound the TE-like gap that should contain 1.496 um; set a and d to a taper '
    'cell to follow the band-edge grading. a / 16 cells (24.6 nm) with subpixel smoothing, ~4e4 unknowns per k (one '
    'host factor each, minutes); more padding mainly raises the guided-band Q (the evanescent tail reaches the CPML).')
_NANOBEAM_K_FRACTIONS = (0.4, 0.45, 0.5)


def nanobeam_mirror_cell_spec():
    """Template 'nanobeam_mirror_cell_bands_3d': quarter mirror cell, k_z sweep to the zone edge, eigenfrequencies."""
    settings = {'bc_x_lo': 'antisymmetric', 'bc_y_lo': 'symmetric', 'bc_x_hi': 'cpml', 'bc_y_hi': 'cpml',
                'bc_z_lo': 'periodic', 'bc_z_hi': 'periodic', 'pml_cells': 8,
                'bloch_kz': 'k_fraction * 2 * pi / a', 'wavelength': 'lambda_t'}
    return ModelSpec(
        title=NANOBEAM_CELL_LABEL,
        description=NANOBEAM_CELL_DESCRIPTION,
        parameters=[ParameterSpec('a', '394.3*nm', 'period (mirror cell; a taper cell: its a_i)'),
                    ParameterSpec('d', '279.3*nm', 'hole diameter (a taper cell: its d_i)'),
                    ParameterSpec('w', '595.2*nm', 'beam width (x)'),
                    ParameterSpec('thickness', '220*nm', 'beam thickness (y)'),
                    ParameterSpec('pad', '300*nm', 'D2O between the beam and the CPML (x and y)'),
                    ParameterSpec('n_si', 3.47, 'silicon refractive index'),
                    ParameterSpec('n_d2o', 1.32, 'D2O refractive index'),
                    ParameterSpec('lambda_t', '1496*nm', 'target wavelength (the measured cavity resonance)'),
                    ParameterSpec('k_fraction', _NANOBEAM_K_FRACTIONS[-1],
                                  'Bloch kz in units of 2 pi / a (0.5 = zone edge)')],
        domain=DomainSpec(size_x='w/2 + pad', size_y='thickness/2 + pad', size_z='a'),
        geometry=[GeometrySpec('beam', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'x1': 'w/2', 'y1': 'thickness/2',
                                               'z1': 'a'}),
                  GeometrySpec('hole', 'circle', {'cx': 0.0, 'cy': 'a/2', 'r': 'd/2', 'z0': 0.0, 'z1': 'thickness/2',
                                                  'axis': 'y'})],
        materials=[MaterialSpec('d2o', properties={'eps_r': 'n_d2o**2'}, selection=['all']),
                   MaterialSpec('silicon', properties={'eps_r': 'n_si**2'}, selection=['beam']),
                   MaterialSpec('d2o_hole', properties={'eps_r': 'n_d2o**2'}, selection=['hole'])],   # later wins
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx='a/16'),
        solver_defaults=SolverDefaultsSpec(subpixel_smoothing=True),
        study=StudySpec(type=_STUDY, settings={'n_resonances': 4},
                        sweep=SweepSpec('k_fraction', list(_NANOBEAM_K_FRACTIONS))))


NANOBEAM_CELL_TEMPLATE = (NANOBEAM_CELL_KEY, NANOBEAM_CELL_LABEL, NANOBEAM_CELL_DESCRIPTION, nanobeam_mirror_cell_spec,
                          'Band diagram of one mirror cell of a silicon nanobeam cavity in heavy water: the phase step '
                          'between cells swept to the zone edge, where the two band edges bound the mirror gap.')


FDFD_3D_CASES = (
    SelfTestCase(
        name='em3d_fdfd_operator_identities',
        label='EM 3D FDFD operators: curl identities and symmetry',
        description='CPU (physics/em3d/operators3d, scipy sparse): on a 7 x 6 x 5 grid of random widths (0.6-1.4 x 50 '
                    'nm) div curl_h = 0 and curl_e grad = 0 to round-off (relative to the factor entries, '
                    f'{_IDENTITY_TOLERANCE:g}) with CPML on five faces stretched at 1 um, with Bloch phases on three '
                    'periodic axes, and with x_lo antisymmetric / y_lo symmetric mirrors and a periodic z axis; on '
                    'the mirror grid with a PEC block and graded real mu, diag(V) K symmetric to round-off (V = the E '
                    'dual volumes, halved for the normal E on the antisymmetric plane); with x_lo antisymmetric, y_lo '
                    'symmetric, z_lo pec and 2-cell CPML on the hi faces at 1 um, a random lossy eps and the PEC block, '
                    'diag(V s) (k^2 eps - K) complex symmetric to round-off (s = operators3d.face_stretches: the '
                    'symmetrizer of the cuDSS LDL^T in fdfd3d; the asymmetry without s in the details). error = '
                    'number of failed sub-checks, tolerance 0.',
        specs=dict,
        check=_check_identities),
    SelfTestCase(
        name='em3d_fdfd_pec_box_discrete',
        label='EM 3D FDFD operators: PEC box spectrum and mirror sectors',
        description='CPU (operators3d, dense eigenvalues): the vacuum curl-curl K of a 5 x 6 x 7 cell PEC box (walls '
                    'at the centres of the outermost cells) vs the discrete Yee closed form k^2 = sum (2 / d sin(m '
                    f'pi / (2 (n - 1))))^2 (two polarisations with every index >= 1, one with one 0) within '
                    f'{_SPECTRUM_TOLERANCE:g}, plus exactly the (n_x - 2)(n_y - 2)(n_z - 2) static (gradient) zero '
                    'eigenvalues and no spurious mode; the four parity sectors of a quarter of a 6 x 6 x 5 box '
                    '(x_lo / y_lo symmetric or antisymmetric ghosts, corner images included) together equal the '
                    'full spectrum and its static count. error = number of failed sub-checks, tolerance 0.',
        specs=dict,
        check=_check_pec_box_discrete),
    SelfTestCase(
        name='em3d_fdfd_eigen_pec_box',
        label='EM 3D FDFD eigenfrequency: dielectric PEC box',
        description=f'em_fdfd_3d eigenfrequency study (n_resonances {_RUN_RESONANCES}) of a 10 x 8 x 6 cell PEC box '
                    f'(50 nm cells) filled with eps {_RUN_EPS:g}, target 0.97 x the (1, 1, 0) wavelength, on the cpu '
                    '(host SuperLU: the one cpu eigen run of em_fdfd_3d) and gpu backends (gpu skipped without a GPU; '
                    'the cuDSS mixed-precision factor, SuperLU without nvmath, as every other em_fdfd_3d run): the '
                    'found omegas vs the discrete closed form C0 '
                    f'sqrt(k^2 / eps) nearest the target within {_RUN_TOLERANCE:g}, 1/Q below it. error = number of '
                    'failed sub-checks, tolerance 0.',
        specs=_box_specs,
        check=_check_box),
    SelfTestCase(
        name='em3d_fdfd_eigen_lossy_box',
        label='EM 3D FDFD eigenfrequency: lossy PEC box (complex omega)',
        description=f'The em3d_fdfd_eigen_pec_box box with sigma_e = {_RUN_SIGMA:g} S/m, target = the lossless (1, 1, '
                    '0) wavelength: complex omegas vs C0 sqrt(k^2 / eps_c), eps_c = 2.25 - j sigma_e / (omega_t eps0) '
                    'frozen at the target (Im omega > 0 decays, Q in the details) within '
                    f'{_RUN_TOLERANCE:g} (complex relative difference).',
        specs=lambda: {'run': _box_spec('Self-test: em_fdfd_3d lossy PEC box', sigma_e=_RUN_SIGMA,
                                        wavelength=_lossy_target())},
        check=_check_lossy_box),
    SelfTestCase(
        name='em3d_fdfd_bloch_empty_lattice',
        label='EM 3D FDFD eigenfrequency: Bloch empty lattice',
        description=f'em_fdfd_3d vacuum unit cell of {_LATTICE_CELLS}^3 cells of 50 nm, periodic on all six faces, at '
                    'bloch_kx = 0.25 and 0.5 x 2 pi / L (target 1 / 1.2 of the lowest band frequency): the '
                    f'{_LATTICE_RESONANCES} found omegas vs the discrete plane-wave bands C0 sqrt(sum (2 / d sin(K_a '
                    'd / 2))^2) (two polarisations each, K = k + G) nearest the target within '
                    f'{_RUN_TOLERANCE:g}, 1/Q below it (the static solutions of every K are not returned). error = '
                    'number of failed sub-checks, tolerance 0.',
        specs=lambda: {run: _lattice_spec(k_fraction) for run, k_fraction in _LATTICE_RUNS.items()},
        check=_check_lattice),
    SelfTestCase(
        name='em3d_fdfd_mirror_sectors',
        label='EM 3D FDFD eigenfrequency: parity sectors of a mirrored box',
        description='The em3d_fdfd_eigen_pec_box box cut to a quarter by x_lo / y_lo mirror planes (5 x 4 x 6 cells + '
                    'ghosts), eigenfrequency study with parity_sectors (four sectors, '
                    f'{_SECTOR_RESONANCES} resonances each): every sector resonance is a full-box mode of the '
                    f'closed form within {_RUN_TOLERANCE:g}, and the union\'s {_SECTOR_RESONANCES} nearest the target '
                    'are the full box\'s. error = number of failed sub-checks, tolerance 0.',
        specs=_sector_specs,
        check=_check_sectors),
    SelfTestCase(
        name='em3d_fdfd_bragg_cell',
        label='EM 3D FDFD eigenfrequency: Bragg stack unit cell (k_z sweep)',
        description='em_fdfd_3d unit cell of a 1D Si / D2O stack (n 3.47 / 1.32, 110 / 280 nm, a = 390 nm, 10 nm '
                    'cells) along z in a 2 x 2 cell x / y periodic column (k_t = 0), z periodic with bloch_kz = '
                    'k_fraction 2 pi / a swept over 0.25 and 0.5 (the zone edge) by a parameter sweep, subpixel '
                    f'smoothing off, target {_BRAGG_TARGET * 1e6:g} um (gpu). The {_BRAGG_RESONANCES} resonances per k '
                    '(derived values over the table rows) vs the bands of the discrete transfer matrix of the Yee '
                    'chain E_{k+1} = (2 - (k0 n_k d)^2) E_k - E_{k-1} (roots of trace T = 2 cos(k_z L), each band '
                    f'twice: E_x and E_y) nearest the target within {_RUN_TOLERANCE:g}, 1/Q below it; those discrete '
                    'bands vs the continuum transfer-matrix bands (cos(k a) = cos(phi_a) cos(phi_b) - (n_a / n_b + n_b '
                    f'/ n_a) / 2 sin(phi_a) sin(phi_b)) within {_BRAGG_CONTINUUM_TOLERANCE:g} (Yee dispersion). error = '
                    'number of failed sub-checks, tolerance 0. Not yet run.',
        specs=lambda: {'sweep': _bragg_spec()},
        check=_check_bragg),
    SelfTestCase(
        name='em3d_fdfd_vs_fdtd',
        label='EM 3D FDFD vs FDTD: one grid, leap-frog map',
        description='The em3d_resonance_pec_box grid (22 x 18 x 14 cells of 50 nm, pec walls) with an eps 4 block '
                    f'(cells 6-11, 5-9, 3-7): em_fdtd_3d ring-down ({_RING_DURATION * 1e12:g} ps, z dipole pulse, '
                    'resonance_monitor E_z, band 1.0-2.2 um) and em_fdfd_3d eigenfrequencies '
                    f'({_RING_RESONANCES} nearest 1.5 um), both on the gpu backend. Every FDTD frequency equals the '
                    'leap-frog frequency asin(dt omega / 2) / (pi dt) of the nearest FDFD omega within '
                    f'{_RING_TOLERANCE:g} (the same E faces, face means and walls; at least two FDTD modes). Not '
                    'yet run.',
        specs=_ring_specs,
        check=_check_ring,
        heavy=True),
    SelfTestCase(
        name='em3d_fdfd_factor_ordering',
        label='EM 3D FDFD host factor: minimum degree vs nested dissection',
        description=f'CPU (no model run): the shift-invert operator A of an eighth Mie sphere grid ({_ORDERING_CELLS}^3 '
                    f'cells of {_MIE_CELL * 1e9:g} nm incl. the mirror ghost and {_ORDERING_PML_CELLS} CPML cells, the '
                    'fdfd_mie_sphere_resonance_3d example faces, staircase eps, ~2e4 unknowns) factorised by host '
                    'SuperLU with its MMD_AT_PLUS_A ordering and with the nested-dissection pre-ordering em_fdfd_3d uses '
                    '(operators3d.nested_dissection_ordering, permc_spec NATURAL): both residuals ||A x - b|| / ||b|| '
                    f'within {_ORDERING_RESIDUAL:g} and the nested-dissection L + U at most {_ORDERING_FILL_RATIO:g} x '
                    'the minimum-degree one; factor / solve seconds and the speed-up in the details. error = number '
                    'of failed sub-checks, tolerance 0. Measured at 24^3 cells: fill ratio 0.84, speed-up 2.0; not yet '
                    'run at this size.',
        specs=dict,
        check=_check_ordering,
        heavy=True),
)

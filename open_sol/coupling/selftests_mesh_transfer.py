"""Per-physics mesh self-tests (S7.2, mesh plan stage P2): the GridTransfer weights on the CPU, and the strips of
coupling/selftests.py with heat on its own mesh on the cpu backend (coarse heat vs the shared mesh; in the currents
case also an identical second mesh reproducing the one-mesh run bit for bit). Registered by definitions.register; no physics imports."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.spec import MeshSpec
from .selftests import (_CELL, _CURRENTS, _DNDT, _EM, _GLASS_INDEX, _HEAT, _MODE_DOWN, _MODE_UP, _N_CELLS, _T0,
                        _currents_strip_spec, _mode_phase, _monitor_phase_shift, _strip_spec, _unconverged_failures,
                        _wrap)

_HEAT_MESH = 'heat_mesh'
_COARSE_CELL = 2 * _CELL                  # m, heat mesh of the two-mesh strips: two EM cells per heat cell
_COARSE_CELLS = (_N_CELLS[0] // 2, _N_CELLS[1] // 2)
_SHARED, _SPLIT, _TWO = 'shared', 'split', 'two'
_RISE_TOLERANCE = 1e-3                    # relative to the peak rise: coarse vs fine heat discretisation, ~1e-4
_PHASE_TOLERANCE = 1e-3                   # relative to the predicted thermo-optic shift: T interpolated onto EM cells
_POWER_TOLERANCE = 1e-6                   # relative: the conservative transfer keeps the window total (float32 sums)
_EXACT_TOLERANCE = 1e-12                  # float64 host weights
_FLOAT32_TOLERANCE = 1e-6                 # the float32 device application


# ---- grid_transfer_cpu (CPU, no run) ----
def _padded(window_edges, n_pad, pad_width):
    """(full edges with n_pad cells of pad_width on both sides, window slice) of float64 window edges."""
    window_edges = np.asarray(window_edges, dtype=np.float64)
    low = window_edges[0] - pad_width * np.arange(n_pad, 0, -1)
    high = window_edges[-1] + pad_width * np.arange(1, n_pad + 1)
    return np.concatenate((low, window_edges, high)), slice(n_pad, n_pad + window_edges.size - 1)


def _graded(length, n_cells, ratio):
    """Window edges [0, length] of n_cells geometric cells (ratio between neighbours), exact ends."""
    widths = ratio ** np.arange(n_cells)
    edges = np.concatenate(([0.0], np.cumsum(widths))) * (length / widths.sum())
    edges[-1] = length
    return edges


def _uniform(length, n_cells):
    edges = np.linspace(0.0, length, n_cells + 1)
    edges[-1] = length
    return edges


def _cell_measures(axes, measures, interval_measure):
    """Outer product of the per-axis window cell measures (0 in padding) over full-grid axes [(edges, window)]."""
    total = np.ones(())
    for (edges, window), measure in zip(axes, measures):
        along = np.zeros(edges.size - 1)
        along[window] = interval_measure(edges[window.start:window.stop], edges[window.start + 1:window.stop + 1],
                                         measure)
        total = np.multiply.outer(total, along)
    return total


def _host_apply(matrices, values):
    """Separable float64 application of the host CSR matrices."""
    for axis, matrix in enumerate(matrices):
        moved = np.moveaxis(values, axis, 0)
        product = matrix @ moved.reshape(moved.shape[0], -1)
        values = np.moveaxis(product.reshape((matrix.shape[0],) + moved.shape[1:]), 0, axis)
    return values


def _conservative_errors(grid_transfer_class, interval_measure, source_axes, target_axes, measures, rng):
    """(host integral error, float32 integral error, padding max |value|, constant-density error) of a conservative
    transfer of a random density between full-grid axes [(edges, window)]."""
    transfer = grid_transfer_class('conservative', [edges for edges, _ in source_axes],
                                   [window for _, window in source_axes], [edges for edges, _ in target_axes],
                                   [window for _, window in target_axes], measures, backend='cpu')
    density = rng.uniform(0.5, 2.0, transfer.source_shape)
    source_volume = _cell_measures(source_axes, measures, interval_measure)
    target_volume = _cell_measures(target_axes, measures, interval_measure)
    integral = float(np.sum(density * source_volume))
    exact = _host_apply(transfer.host_matrices, density)
    applied = transfer.apply(density.astype(np.float32)).astype(np.float64)
    in_target = target_volume > 0.0
    constant = _host_apply(transfer.host_matrices, np.full(transfer.source_shape, 3.0))
    return (abs(float(np.sum(exact * target_volume)) - integral) / integral,
            abs(float(np.sum(applied * target_volume)) - integral) / integral,
            float(np.max(np.abs(exact[~in_target]), initial=0.0)),
            float(np.max(np.abs(constant[in_target] - 3.0))) / 3.0)


def _interpolation_errors(grid_transfer_class, source_axes, target_axes):
    """(host, float32) max errors of interpolating f = 1 + 2 x - 3 y (+ 0.5 z) from the source window centres onto
    every target cell, against f at the target centres clamped into the source window centre range."""
    transfer = grid_transfer_class('interpolate', [edges for edges, _ in source_axes],
                                   [window for _, window in source_axes], [edges for edges, _ in target_axes],
                                   [window for _, window in target_axes], backend='cpu')
    slopes = (2.0, -3.0, 0.5)

    def linear(centres):
        value = np.ones(())
        for axis, axis_centres in enumerate(centres):
            value = np.add.outer(value, slopes[axis] * axis_centres)
        return value

    source_centres = [0.5 * (edges[:-1] + edges[1:]) for edges, _ in source_axes]
    target_centres = [np.clip(0.5 * (edges[:-1] + edges[1:]), centres[window][0], centres[window][-1])
                      for (edges, _), centres, (_, window) in zip(target_axes, source_centres, source_axes)]
    field = linear(source_centres)
    expected = linear(target_centres)
    scale = float(np.max(np.abs(expected)))
    exact = _host_apply(transfer.host_matrices, field)
    applied = transfer.apply(field.astype(np.float32)).astype(np.float64)
    return (float(np.max(np.abs(exact - expected))) / scale, float(np.max(np.abs(applied - expected))) / scale)


def _check_grid_transfer(results_by_name):
    """Conservative totals (graded -> coarse, coarse -> fine, polar, axisymmetric, spherical, 3D), padding zeros and
    constants kept; interpolation exact for linear fields (edge-extended)."""
    from ..core.grid_transfer import GridTransfer, interval_measure

    rng = np.random.default_rng(20261019)
    errors, details = {}, {}

    def add(name, error, tolerance):
        errors[name] = error / tolerance
        details[name] = error

    length = ('length', 0.0)
    graded_x = _padded(_graded(3.0, 24, 1.12), 2, 0.2)
    coarse_x = _padded(_uniform(3.0, 7), 1, 0.4)
    fine_x = _padded(_graded(3.0, 40, 0.97), 0, 0.0)
    y_ten, y_four = _padded(_uniform(1.0, 10), 0, 0.0), _padded(_uniform(1.0, 4), 1, 0.25)
    angle_six, angle_nine = _padded(_uniform(0.5 * math.pi, 6), 0, 0.0), _padded(_uniform(0.5 * math.pi, 9), 0, 0.0)
    theta_five, theta_eight = _padded(_uniform(math.pi, 5), 0, 0.0), _padded(_graded(math.pi, 8, 1.2), 0, 0.0)
    cases = {
        'graded_to_coarse': ([graded_x, y_ten], [coarse_x, y_four], (length, length)),
        'coarse_to_fine': ([coarse_x, y_four], [fine_x, y_ten], (length, length)),
        'polar': ([graded_x, angle_six], [coarse_x, angle_nine], (('radial', 0.5), length)),
        'axisymmetric': ([fine_x, y_four], [graded_x, y_ten], (('radial', 0.0), length)),
        'spherical': ([graded_x, theta_five], [fine_x, theta_eight], (('radial_squared', 0.3), ('sine', 0.0))),
        '3d': ([graded_x, y_ten, y_four], [coarse_x, y_four, y_ten], (length, length, length)),
    }
    for name, (source_axes, target_axes, measures) in cases.items():
        host, applied, padding, constant = _conservative_errors(GridTransfer, interval_measure, source_axes,
                                                                target_axes, measures, rng)
        add(f'{name}.total_host', host, _EXACT_TOLERANCE)
        add(f'{name}.total_float32', applied, _FLOAT32_TOLERANCE)
        add(f'{name}.padding_zero', padding, _EXACT_TOLERANCE)
        add(f'{name}.constant_kept', constant, _EXACT_TOLERANCE)
    for name, (source_axes, target_axes) in {'interpolate_2d': ([graded_x, y_ten], [coarse_x, y_four]),
                                             'interpolate_fine': ([coarse_x, y_four], [fine_x, y_ten]),
                                             'interpolate_3d': ([graded_x, y_ten, y_four],
                                                                [coarse_x, y_four, y_ten])}.items():
        host, applied = _interpolation_errors(GridTransfer, source_axes, target_axes)
        add(f'{name}.linear_host', host, _EXACT_TOLERANCE)
        add(f'{name}.linear_float32', applied, _FLOAT32_TOLERANCE)
    worst = max(errors, key=errors.get)
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'worst': worst, 'errors': details, 'normalised_errors': errors}}


# ---- two-mesh strips (cpu runs) ----
def _on_cpu(spec):
    spec.solver_defaults.backend = 'cpu'
    return spec


def _heat_on_own_mesh(spec, cell):
    """The spec with heat on a second uniform mesh of `cell` (derived values of heat fields dropped on a coarser mesh:
    Session results are not mesh-aware before stage P3)."""
    spec.meshes = [spec.meshes[0], MeshSpec(_HEAT_MESH, mode='uniform', dx=cell)]
    spec.physics_by_name(_HEAT).mesh = _HEAT_MESH
    if cell != _CELL:
        spec.derived_values = [derived for derived in spec.derived_values if derived.name != 'T_max']
    return spec


def _two_mesh_specs(make_spec, with_identical=False):
    """shared (heated) on one mesh, split = heated with heat on the coarse mesh, with_identical: two = heated with heat
    on a second mesh equal to the first (no PML there); cpu."""
    specs = {_SHARED: _on_cpu(make_spec('heated', _DNDT)),
             _SPLIT: _on_cpu(_heat_on_own_mesh(make_spec('heated', _DNDT), _COARSE_CELL))}
    if with_identical:
        specs[_TWO] = _on_cpu(_heat_on_own_mesh(make_spec('heated', _DNDT), _CELL))
    return specs


def _values(results, name, shape):
    """float64 window values of a field; ValueError unless it has `shape`."""
    values = np.asarray(results.field(name).values, dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f'{name}: expected the window shape {shape}, got {values.shape}')
    return values


def _monitor_phase_difference(results):
    """phi_down - phi_up [rad] of the mode monitors of one run."""
    return _mode_phase(results, _MODE_DOWN)[0] - _mode_phase(results, _MODE_UP)[0]


def _bit_identity(one, two):
    """(differing items, largest difference per item) of the currents strip run on one mesh vs an identical second."""
    differing, largest = [], {}
    for name in (f'{_HEAT}.T', f'{_HEAT}.Q', f'{_CURRENTS}.q_joule', f'{_EM}.eps_r'):
        first, second = _values(one, name, _N_CELLS), _values(two, name, _N_CELLS)
        largest[name] = float(np.max(np.abs(first - second)))
        if not np.array_equal(first, second):
            differing.append(name)
    for feature in (_MODE_UP, _MODE_DOWN):
        first, second = _mode_phase(one, feature)[0], _mode_phase(two, feature)[0]
        largest[f'{feature}.phase_positive'] = abs(first - second)
        if first != second:
            differing.append(f'{feature}.phase_positive')
    return differing, largest


def _make_two_mesh_check(power_field):
    """check of a two-mesh strip whose heat source is power_field [W/m^3] on the first (20 nm) mesh; a 'two' run, when
    present, must equal the shared one bit for bit (each differing item a failed check)."""
    def check(results_by_name):
        shared, split = results_by_name[_SHARED], results_by_name[_SPLIT]
        # 1. heat on the coarse mesh against the pair means of the shared-mesh rise at its centres
        rise_split = _values(split, f'{_HEAT}.T', _COARSE_CELLS).mean(axis=1) - _T0
        rise_shared = _values(shared, f'{_HEAT}.T', _N_CELLS).mean(axis=1).reshape(-1, 2).mean(axis=1) - _T0
        peak = float(np.max(rise_shared))
        err_rise = float(np.max(np.abs(rise_split - rise_shared))) / peak
        # 2. thermo-optic phase of the EM physics fed by the interpolated coarse T: split vs shared monitor phase
        #    difference (the dndT = 0 reference cancels), scaled by the discrete-k shift of the shared theta (~2 %)
        phase_difference = _wrap(_monitor_phase_difference(split) - _monitor_phase_difference(shared))
        theta_shared = _values(shared, f'{_HEAT}.T', _N_CELLS).mean(axis=1) - _T0
        dphi_predicted = _monitor_phase_shift(_GLASS_INDEX + _DNDT * theta_shared)
        err_phase = abs(phase_difference) / abs(dphi_predicted)
        # 3. the conservative transfer keeps the total: integral of Q (coarse) = integral of the source density (fine)
        power_source = float(np.sum(_values(split, power_field, _N_CELLS))) * _CELL ** 2
        power_heat = float(np.sum(_values(split, f'{_HEAT}.Q', _COARSE_CELLS))) * _COARSE_CELL ** 2
        err_power = abs(power_heat - power_source) / abs(power_source)
        failed = _unconverged_failures({name: results.summary() for name, results in results_by_name.items()})
        details = {}
        if _TWO in results_by_name:
            # 4. an identical second mesh reproduces the one-mesh (shared) run bit for bit
            differing, largest = _bit_identity(shared, results_by_name[_TWO])
            failed += [f'{_TWO}: {item} differs from {_SHARED}' for item in differing]
            details.update(bit_identity_differing=differing, bit_identity_largest_difference=largest)
        normalised = {'rise': err_rise / _RISE_TOLERANCE, 'phase': err_phase / _PHASE_TOLERANCE,
                      'power': err_power / _POWER_TOLERANCE}
        error = max(normalised.values()) + len(failed)
        details.update(err_rise=err_rise, err_phase=err_phase, err_power=err_power, normalised_errors=normalised,
                       failed_checks=failed, peak_rise_K=peak, phase_difference_rad=phase_difference,
                       dphi_predicted_shared_rad=dphi_predicted, power_source_W_per_m=power_source,
                       power_heat_W_per_m=power_heat)
        return {'measured': phase_difference, 'expected': 0.0, 'error': error, 'tolerance': 1.0, 'details': details}
    return check


GRID_TRANSFER_CASE = SelfTestCase(
    name='grid_transfer_cpu',
    label='Per-physics meshes: grid transfer weights (CPU)',
    description='core.grid_transfer.GridTransfer between two tensor grids of one window (CPU, no run). Conservative '
                '(densities): a random density on graded / uniform x (padding cells included) times uniform y, '
                'transferred graded -> coarse, coarse -> fine, on a polar (r dr, radius_min 0.5) and an axisymmetric '
                '(r dr from the axis) window, a spherical one (r^2 dr, sin theta dtheta) and in 3D: the window '
                'integral sum q m of the target equals the source one to 1e-12 with the float64 host weights (each '
                'merged-edge piece maps its measure exactly, the window ends snapped) and to 1e-6 with the float32 '
                'backend matrices, target padding cells get exactly 0 and a constant density stays constant (1e-12). '
                'Interpolate (fields): f = 1 + 2x - 3y (+ 0.5z) on the source window centres reproduced at every target '
                'cell, padding included, as f at the target centres clamped into the source centre range (linear '
                'interpolation is exact for linear fields; edge extension beyond the outer centres), 1e-12 host, 1e-6 '
                'float32 (relative to max |f|). error = the largest error / its tolerance, tolerance 1.',
    specs=lambda: {},
    check=_check_grid_transfer)

JOULE_CASE = SelfTestCase(
    name='two_mesh_joule_thermo_optic_strip',
    label='Per-physics meshes: EM (20 nm) + heat (40 nm) strip vs one mesh (cpu)',
    description='The heated em_heat_thermo_optic_strip (em_fdfd_ez + heat, dndT 1e-3 /K, absorbing slab) on the cpu '
                'backend, once on one 20 nm mesh (shared) and once with heat on its own uniform 40 nm mesh (split: '
                'em_heat_joule moves the absorbed power conservatively onto the heat cells, heat_em_thermo_optic '
                'interpolates T onto the EM cells, the heat dndT painted on the EM mesh). Checks: the split rise '
                '(y-mean per heat cell) against the mean of the two shared cells it covers, max difference / peak '
                'rise, tolerance 1e-3 (the 3-point scheme on 40 nm cells with the cell totals of q differs from the '
                '20 nm one by ~q h^2 / (8 k) ~ 1e-4 of the 33 K peak); the thermo-optic phase, |wrap([phi_down - '
                'phi_up]_split - [phi_down - phi_up]_shared)| (the phase of a dndT = 0 reference cancels, so none is '
                'run) / |dphi_predicted| with dphi_predicted = -sum [k_num(n0 + dndT theta_shared) - k_num(n0)] dx '
                'between the monitor line cells (the discrete-k shift of em_heat_thermo_optic_strip, within ~2 % of '
                'the measured shift), tolerance 1e-3 (linear interpolation of the coarse T: theta\'\' h^2 / 8 ~ '
                '1e-4); the integral of heat Q over the coarse window equals the integral of em absorbed_power over '
                'the EM window (conservative transfer, weights 1/2 exact in float32), tolerance 1e-6; no \'did not '
                'converge\' warning. error = the largest error / tolerance + failed checks, tolerance 1; measured = '
                'the split - shared phase difference in rad, expected 0. Field shapes are checked (heat on 200 x 5 '
                'cells). Not yet run without the reference run.',
    specs=lambda: _two_mesh_specs(_strip_spec),
    check=_make_two_mesh_check(f'{_EM}.absorbed_power'))

CURRENTS_CASE = SelfTestCase(
    name='two_mesh_currents_heat',
    label='Per-physics meshes: currents + EM (20 nm) and heat (40 nm or an identical 20 nm) strip vs one mesh (cpu)',
    description='The heated currents_heat_thermo_optic_strip (electric_currents resistor + heat + em_fdfd_ez) on the '
                'cpu backend, on one 20 nm mesh (shared), with heat on its own uniform 40 nm mesh (split: '
                'currents_heat_joule moves q_joule conservatively, heat_em_thermo_optic interpolates T back) and with '
                'heat on a second mesh of the same 20 nm cells (two; padding-free: the CPML stays on the EM / currents '
                'mesh). Checks as two_mesh_joule_thermo_optic_strip (no reference run): the coarse rise against the '
                'pair means of the shared one (1e-3 of the peak), the split - shared monitor phase difference over '
                'the discrete-k shift of the shared theta (1e-3), the integral of heat Q = the integral of currents '
                'q_joule (1e-6), no \'did not converge\' warning. Bit identity: on equal window cells the '
                'conservative weights are exactly 1 and the interpolation weights (1, 0), and the EM padding gets the '
                'edge value the one-mesh heat writes into its ring, so heat T and Q, currents q_joule, em eps_r and '
                'both mode-monitor phases of two must equal those of shared bit for bit (each differing item a '
                'failed check): this pins the per-mesh build path (contexts, material arrays incl. the heat dndT '
                'painted on the EM mesh, transfers) against the one-mesh build. error = the largest error / tolerance '
                '+ failed checks, tolerance 1. Not yet run as these three runs.',
    specs=lambda: _two_mesh_specs(_currents_strip_spec, with_identical=True),
    check=_make_two_mesh_check(f'{_CURRENTS}.q_joule'))

MESH_TRANSFER_CASES = (GRID_TRANSFER_CASE, JOULE_CASE, CURRENTS_CASE)

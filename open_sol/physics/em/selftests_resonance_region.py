"""Self-tests of the resonance region energy fraction (new_modules_plan P2): eigenfrequency.solve_resonances and
fem.harmonic_resonances on two decoupled Dirichlet chains (CPU, no model run), and the em_fdfd_ez resonance_table
region on a PEC box split by a pec_domain wall into a cavity and a side room (tensor and triangles meshes)."""
import math
from types import SimpleNamespace

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec)
from .selftests_fdfd import cpu_specs

_N_RESONANCES = 3
_MIN_REGION_FRACTION = 0.5
_FREQUENCY_COLUMNS = (2, 3)        # frequency, frequency_imag [Hz] in the resonance_table rows

# ---- two decoupled Dirichlet chains (-d^2/dx^2 on n nodes, ends held: k_m = (2 / dx) sin(m pi / (2 (n + 1)))):
# chain 0 = the region. Target 1 um (k_t 6.28e6 1/m): nearest chain 0 m = 3, chain 1 m = 2, chain 0 m = 4 (then chain 0
# m = 2), so the unfiltered three hold a chain 1 resonance and the filtered three are chain 0 m = 3, 4, 2.
_CHAIN_NODES = (150, 97)
_CHAIN_STEP = 10e-9                # m
_CHAIN_TARGET_WAVELENGTH = 1.0e-6  # m
_CHAIN_OMEGA_TOLERANCE = 1e-9      # relative omega (exact discrete pencil, ARPACK to machine precision)
_FRACTION_TOLERANCE = 1e-8         # region fraction of a resonance living in one decoupled block (1 or 0)

# ---- PEC box (em_fdfd_ez, pec on all faces, 40 nm cells, 67 x 31) split by a one-cell pec_domain column (cell 40,
# x 1.60-1.64 um) into the selected 'cavity' (x < 1.6 um) and a side room. Tensor: E_z = 0 in the outermost and the PEC
# cells, so the Dirichlet planes are cell centres: cavity L_x = 40 h, side room 26 h, L_y = 30 h, the 5-point closed form
# K(m) = (2 / h) sin(m pi h / (2 L)) is exact. Triangles (quadratic): the drawn walls, cavity 1.6 um, side room 1.04 um,
# L_y 1.24 um, the continuum K = m pi / L. Target 1.55 um: nearest side room (1, 1), then cavity (2, 1), (1, 1), (1, 2).
_BOX_CELL = 40e-9                  # m
_BOX_CELLS = (67, 31)
_WALL_COLUMN = 40
_BOX_TARGET_WAVELENGTH = 1.55e-6   # m
_MAX_ORDER = 8
_BOX_OMEGA_TOLERANCES = {'tensor': 1e-8, 'triangles': 1e-4}   # relative omega: exact discrete form / quadratic FEM
_PHYSICS = 'em'
_TABLE = 'resonances'
_CAVITY = 'cavity'
_ROOMS = (_CAVITY, 'side_room')


# ---- CPU: solve_resonances and solve_harmonic_resonances on the chains ----
def _chain_laplacian():
    """(scipy CSR block-diagonal second difference of the chains [1/m^2], bool rows of chain 0)."""
    import scipy.sparse
    blocks = [scipy.sparse.diags([np.ones(count - 1), -2.0 * np.ones(count), np.ones(count - 1)], [-1, 0, 1])
              / _CHAIN_STEP ** 2 for count in _CHAIN_NODES]
    in_first_chain = np.zeros(sum(_CHAIN_NODES), dtype=bool)
    in_first_chain[:_CHAIN_NODES[0]] = True
    return scipy.sparse.block_diag(blocks, format='csr'), in_first_chain


def _chain_expected(omega_target):
    """[(omega [rad/s], chain)] of every chain resonance, nearest omega_target first."""
    entries = [(C0 * 2.0 / _CHAIN_STEP * math.sin(order * math.pi / (2.0 * (count + 1))), chain)
               for chain, count in enumerate(_CHAIN_NODES) for order in range(1, count + 1)]
    return sorted(entries, key=lambda entry: abs(entry[0] - omega_target))


def _compare(label, resonances, expected, errors, tolerances, details, omega_tolerance, with_region=True):
    """errors[label omega / fraction] of resonances (omega, region_fraction) vs expected [(omega, in-region bool)]."""
    measured = [complex(resonance.omega) for resonance in resonances]
    if len(measured) < len(expected):
        errors[f'{label} count'] = math.inf
        tolerances[f'{label} count'] = 0.5
        details[label] = {'omega_measured': [abs(value) for value in measured]}
        return
    errors[f'{label} omega'] = max(abs(value - reference) / reference
                                   for value, (reference, _) in zip(measured, expected))
    tolerances[f'{label} omega'] = omega_tolerance
    fractions = [float(resonance.region_fraction) for resonance in resonances[:len(expected)]]
    if with_region:
        errors[f'{label} fraction'] = max(abs(value - (1.0 if inside else 0.0))
                                          for value, (_, inside) in zip(fractions, expected))
    else:   # no region: NaN on every resonance
        errors[f'{label} fraction'] = 0.0 if all(math.isnan(value) for value in fractions) else math.inf
    tolerances[f'{label} fraction'] = _FRACTION_TOLERANCE
    details[label] = {'omega_measured': [value.real for value in measured],
                      'omega_expected': [reference for reference, _ in expected],
                      'region_fraction': [None if math.isnan(value) else value for value in float_list(fractions)]}


def _raises(call) -> float:
    """0 when call() raises ValueError, inf otherwise."""
    try:
        call()
    except ValueError:
        return 0.0
    return math.inf


def _check_chains(results_by_name):
    """EM and neutral resonance solves of the chain pencil: filtered / listed / plain vs the closed form, fractions,
    table columns, the dropped count, the region-argument errors and element_region_rows."""
    import scipy.sparse

    from ...fem.harmonic_resonances import resonance_table, solve_harmonic_resonances
    from .eigenfrequency import REGION_COLUMN, element_region_rows, resonance_table_result, solve_resonances
    laplacian, in_first_chain = _chain_laplacian()
    n_unknowns = laplacian.shape[0]
    omega_target = 2.0 * math.pi * C0 / _CHAIN_TARGET_WAVELENGTH
    ordered = [(omega, chain == 0) for omega, chain in _chain_expected(omega_target)]
    listed_expected = ordered[:_N_RESONANCES]
    filtered_expected = [entry for entry in ordered if entry[1]][:_N_RESONANCES]
    if all(inside for _, inside in listed_expected):
        raise ValueError('test design: the unfiltered nearest resonances hold no chain 1 resonance')
    # EM pencil A = dx L + k_t^2 dx I, mass dx (a resonance solves L u = -k^2 u), weights dx, no PML
    k_target_squared = (omega_target / C0) ** 2
    operator = (_CHAIN_STEP * laplacian + k_target_squared * _CHAIN_STEP * scipy.sparse.identity(n_unknowns)).tocsr()
    mass = np.full(n_unknowns, _CHAIN_STEP, dtype=np.complex128)
    weights = np.full(n_unknowns, _CHAIN_STEP)
    window = np.ones(n_unknowns, dtype=bool)

    def em_solve(**region):
        return solve_resonances(operator, mass, omega_target, weights, window, _N_RESONANCES, **region)

    # neutral pencil K u = omega^2 M u, K = -C0^2 dx L, M = dx I (real symmetric: eigsh_shifted)
    stiffness = (-(C0 ** 2) * _CHAIN_STEP * laplacian).tocsr()
    mass_matrix = (_CHAIN_STEP * scipy.sparse.identity(n_unknowns)).tocsr()
    target_frequency = omega_target / (2.0 * math.pi)

    def neutral_solve(**region):
        return solve_harmonic_resonances(stiffness, mass_matrix, target_frequency, _N_RESONANCES, **region)

    errors, tolerances, details = {}, {}, {}
    solves = {'em filtered': em_solve(in_region=in_first_chain, min_region_fraction=_MIN_REGION_FRACTION),
              'em listed': em_solve(in_region=in_first_chain), 'em plain': em_solve(),
              'neutral filtered': neutral_solve(in_region=in_first_chain, min_region_fraction=_MIN_REGION_FRACTION),
              'neutral listed': neutral_solve(in_region=in_first_chain), 'neutral plain': neutral_solve()}
    for label, solve in solves.items():
        expected = filtered_expected if label.endswith('filtered') else listed_expected
        _compare(label, solve.resonances, expected, errors, tolerances, details, _CHAIN_OMEGA_TOLERANCE,
                 with_region=not label.endswith('plain'))
    for family in ('em', 'neutral'):   # the filter dropped at least the chain 1 resonance near the target
        dropped = solves[f'{family} filtered'].info.get('region_resonances_dropped', 0)
        errors[f'{family} region dropped'] = 0.0 if dropped >= 1 else math.inf
        tolerances[f'{family} region dropped'] = 0.5
    tables = {'em filtered': resonance_table_result(None, SimpleNamespace(eigen_solve=solves['em filtered'],
                                                                          selected_resonance=None)),
              'em plain': resonance_table_result(None, SimpleNamespace(eigen_solve=solves['em plain'],
                                                                       selected_resonance=None)),
              'neutral filtered': resonance_table(solves['neutral filtered']),
              'neutral plain': resonance_table(solves['neutral plain'])}
    for label, table in tables.items():
        has_column = table['columns'][-1] == REGION_COLUMN[0] and table['units'][-1] == REGION_COLUMN[1]
        rows_match = all(len(row) == len(table['columns']) for row in table['rows'])
        wanted = label.endswith('filtered')
        errors[f'{label} table columns'] = 0.0 if has_column == wanted and rows_match else math.inf
        tolerances[f'{label} table columns'] = 0.5
        details[f'{label} columns'] = list(table['columns'])
    for label, call in (('em fraction without region', lambda: em_solve(min_region_fraction=0.5)),
                        ('em fraction above 1', lambda: em_solve(in_region=in_first_chain, min_region_fraction=1.5)),
                        ('neutral fraction without region', lambda: neutral_solve(min_region_fraction=0.5)),
                        ('neutral fraction above 1',
                         lambda: neutral_solve(in_region=in_first_chain, min_region_fraction=1.5))):
        errors[label] = _raises(call)
        tolerances[label] = 0.5
    # element_region_rows: elements 0 / 2 in the map (element 1 outside), dofs (0, 1) / (1, 2), free dofs 0, 2, 3
    dofs = SimpleNamespace(element_position=np.array([0, -1, 1]), element_dofs=np.array([[0, 1], [1, 2]]), n_dofs=4)
    rows = element_region_rows(dofs, np.array([0, 2, 3]), np.array([True, True, False]))
    errors['element_region_rows'] = 0.0 if rows.tolist() == [True, False, False] else math.inf
    tolerances['element_region_rows'] = 0.5
    return _outcome(errors, tolerances, details)


def _outcome(errors, tolerances, details):
    """measured = error = the worst errors[item] / tolerances[item] (NaN ranks worst), tolerance 1."""
    normalized = {item: error / tolerances[item] for item, error in errors.items()}
    worst_item = max(normalized, key=lambda item: math.inf if math.isnan(normalized[item]) else normalized[item])
    details.update(errors=errors, worst_item=worst_item)
    return {'measured': normalized[worst_item], 'expected': 0.0, 'error': normalized[worst_item], 'tolerance': 1.0,
            'details': details}


# ---- model runs: em_fdfd_ez resonance_table with a region ----
def _box_spec(kind, min_region_fraction):
    """em_fdfd_ez in the split PEC box on a `kind` ('tensor' | 'triangles') mesh with a resonance_table on the cavity
    region (min_region_fraction as given)."""
    size_x, size_y = (count * _BOX_CELL for count in _BOX_CELLS)
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings['wavelength'] = _BOX_TARGET_WAVELENGTH
    mesh = MeshSpec(mode='uniform', dx=_BOX_CELL)
    if kind == 'triangles':
        mesh = MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_BOX_CELL)
    return ModelSpec(
        title=f'em_fdfd_ez resonance region ({kind}, min_region_fraction {min_region_fraction:g})',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('wall', 'rectangle', {'x0': _WALL_COLUMN * _BOX_CELL, 'y0': 0.0,
                                                     'x1': (_WALL_COLUMN + 1) * _BOX_CELL, 'y1': size_y}),
                  GeometrySpec(_CAVITY, 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _WALL_COLUMN * _BOX_CELL,
                                                      'y1': size_y})],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_ez', settings=settings, features=[
            FeatureSpec('wall', 'pec_domain', ['wall']),
            FeatureSpec(_TABLE, 'resonance_table', [_CAVITY], params={'min_region_fraction': min_region_fraction})])],
        meshes=[mesh],
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _N_RESONANCES}))


def _box_specs():
    return cpu_specs({f'{kind}_{run}': _box_spec(kind, fraction) for kind in _BOX_OMEGA_TOLERANCES
                      for run, fraction in (('filtered', _MIN_REGION_FRACTION), ('listed', 0.0))})


def _room_lengths(kind):
    """{room: L_x} and L_y [m] between the Dirichlet planes of the mesh kind (module notes: the x lengths agree, the
    tensor y walls are the outermost cell centres)."""
    lengths_x = {_CAVITY: _WALL_COLUMN * _BOX_CELL, 'side_room': (_BOX_CELLS[0] - 1 - _WALL_COLUMN) * _BOX_CELL}
    return lengths_x, (_BOX_CELLS[1] - (1 if kind == 'tensor' else 0)) * _BOX_CELL


def _box_expected(kind):
    """[(omega [rad/s], in the cavity)] of both rooms (discrete on tensor, continuum on triangles), nearest first."""
    omega_target = 2.0 * math.pi * C0 / _BOX_TARGET_WAVELENGTH
    lengths_x, length_y = _room_lengths(kind)

    def wavenumber(order, length):
        if kind == 'tensor':
            return 2.0 / _BOX_CELL * math.sin(order * math.pi * _BOX_CELL / (2.0 * length))
        return order * math.pi / length

    entries = [(C0 * math.hypot(wavenumber(m, lengths_x[room]), wavenumber(n, length_y)), room == _CAVITY)
               for room in _ROOMS for m in range(1, _MAX_ORDER + 1) for n in range(1, _MAX_ORDER + 1)]
    return sorted(entries, key=lambda entry: abs(entry[0] - omega_target))


def _check_box(results_by_name):
    """Per mesh kind: the filtered table = the cavity resonances nearest the target, the listed one = the nearest of
    both rooms, region_energy_fraction 1 / 0, the last column, n snapshots each."""
    errors, tolerances, details = {}, {}, {}
    for kind, omega_tolerance in _BOX_OMEGA_TOLERANCES.items():
        ordered = _box_expected(kind)
        expected_by_run = {'filtered': [entry for entry in ordered if entry[1]][:_N_RESONANCES],
                           'listed': ordered[:_N_RESONANCES]}
        if all(inside for _, inside in expected_by_run['listed']):
            raise ValueError(f'test design ({kind}): the unfiltered nearest resonances hold no side-room resonance')
        for run, expected in expected_by_run.items():
            name = f'{kind}_{run}'
            results = results_by_name[name]
            table = results.result(_PHYSICS, _TABLE)
            columns = list(table['columns'])
            errors[f'{name} region column'] = 0.0 if columns[-1] == 'region_energy_fraction' else math.inf
            tolerances[f'{name} region column'] = 0.5
            resonances = [SimpleNamespace(omega=2.0 * math.pi * complex(float(row[_FREQUENCY_COLUMNS[0]]),
                                                                        float(row[_FREQUENCY_COLUMNS[1]])),
                                          region_fraction=float(row[-1])) for row in table['rows']]
            _compare(name, resonances, expected, errors, tolerances, details, omega_tolerance)
            labels = results.summary().get('snapshot_labels') or []
            errors[f'{name} snapshots'] = abs(len(labels) - _N_RESONANCES)
            tolerances[f'{name} snapshots'] = 0.5
            details[f'{name} columns'] = columns
    return _outcome(errors, tolerances, details)


RESONANCE_REGION_SELFTEST_CASES = (
    SelfTestCase(
        name='em_resonance_region_fraction',
        label='Resonance solves: region energy fraction and filter',
        description=(
            'CPU check of eigenfrequency.solve_resonances (EM pencil A = dx L + k_t^2 dx I, mass dx) and '
            'fem.harmonic_resonances.solve_harmonic_resonances (K = -C0^2 dx L, M = dx I, real symmetric) on two '
            f'decoupled Dirichlet chains ({_CHAIN_NODES[0]} and {_CHAIN_NODES[1]} nodes, dx {_CHAIN_STEP * 1e9:g} nm; '
            'k_m = (2 / dx) sin(m pi / (2 (n + 1))) exact), region = chain 0, target '
            f'{_CHAIN_TARGET_WAVELENGTH * 1e6:g} um, {_N_RESONANCES} resonances. Filtered (min_region_fraction '
            f'{_MIN_REGION_FRACTION:g}): the three chain-0 resonances nearest the target, region fraction 1, at least '
            'one dropped; listed (region, no filter): the three nearest of both chains with fractions 1 / 0; plain (no '
            'region): the same omegas, fraction NaN, no region_energy_fraction table column (present and last on the '
            'filtered tables). Relative omega to '
            f'{_CHAIN_OMEGA_TOLERANCE:g}, fractions to {_FRACTION_TOLERANCE:g}; min_region_fraction > 0 without a '
            'region or > 1 raises ValueError in both; element_region_rows on a hand-made DofMap. measured = error = '
            'the worst item in tolerance units, tolerance 1. Written, not run.'),
        specs=lambda: {},
        check=_check_chains,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_fdfd_resonance_region',
        label='EM FDFD: resonance table region (cavity passes, side room filtered)',
        description=(
            f'em_fdfd_ez eigenfrequency study (target {_BOX_TARGET_WAVELENGTH * 1e6:g} um, {_N_RESONANCES} '
            f'resonances) in a vacuum box of {_BOX_CELLS[0]} x {_BOX_CELLS[1]} cells of {_BOX_CELL * 1e9:g} nm, pec '
            f'on all faces, split by a pec_domain column (x {_WALL_COLUMN * _BOX_CELL * 1e6:g} to '
            f'{(_WALL_COLUMN + 1) * _BOX_CELL * 1e6:g} um) into the cavity (the resonance_table selection) and a side '
            'room; tensor (exact 5-point closed form of both rooms, the walls at cell centres: cavity 40 h, side room '
            '26 h, L_y 30 h) and triangles (quadratic elements, the continuum modes of the drawn rooms 1.6 / 1.04 x '
            '1.24 um) meshes, cpu backend. min_region_fraction '
            f'{_MIN_REGION_FRACTION:g}: the table lists the cavity resonances nearest the target ((2, 1), (1, 1), '
            '(1, 2); the side-room (1, 1), nearest of all, is dropped); min_region_fraction 0: the nearest of both '
            'rooms (side room (1, 1), cavity (2, 1), (1, 1)). region_energy_fraction is the last column, 1 for cavity '
            f'and 0 for side-room resonances (to {_FRACTION_TOLERANCE:g}: the PEC column decouples the rooms), '
            f"relative omega tensor {_BOX_OMEGA_TOLERANCES['tensor']:g}, triangles "
            f"{_BOX_OMEGA_TOLERANCES['triangles']:g}, {_N_RESONANCES} snapshots per run. measured = error = the worst "
            'item in tolerance units, tolerance 1. Written, not run.'),
        specs=_box_specs,
        check=_check_box,
        max_wall_time=120.0),
)

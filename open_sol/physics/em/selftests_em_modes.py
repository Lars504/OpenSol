"""Model-run self-test of em_modes (PEC rectangular waveguide: discrete TE_mn closed form, TE fraction, group index,
dispersion tracking) and the 'strip_waveguide_modes' / 'thermo_optic_phase_shifter' templates; no physics imports."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, solver_role
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec, SolverSpec, StudySpec,
                           SweepSpec)
from .selftests import normalized_result

_TYPE = 'em_modes'      # = definition_modes.EM_MODES.name (definition_modes imports this module)
_PHYSICS = 'modes'
_TABLE = 'table'
_DISPERSION = 'dispersion'
_TABLE_COLUMNS = {'n_eff': 1, 'n_group': 4, 'te_fraction': 5}   # column index in the mode_table rows

# ---- PEC rectangular waveguide (vacuum): every wall plane sits on the inner face of the outermost cell, so the guide
# spans the cells 1 .. n-2 of each axis: a = (42 - 2) * 50 nm = 2.0 um, b = (20 - 2) * 50 nm = 0.9 um. The discrete TE_mn
# modes of a homogeneous Yee cross-section obey beta^2 = k0^2 - K_x(m)^2 - K_y(n)^2 with K(m) = (2 / h) sin(m pi h /
# (2 a)) exactly (the 3-point eigenvectors sin(m pi (i - 1/2) / (n - 2)) meet the odd ghost rows), so the eigen-solve
# and round-off are the whole budget at any h (50 nm: the closed form holds exactly, the cell only sets how far the
# discrete values sit from the continuum ones). At 1.55 um the three largest n_eff are TE10 (0.922, E_y only:
# te_fraction 0), TE20 (0.632, E_y only) and TE01 (0.509, E_x only: te_fraction 1); TE11 / TM11 (0.329) follow. The
# group index of a hollow guide is 1 / n_eff (K does not depend on omega).
_PEC_CELL = 50e-9                       # m
_PEC_CELLS = (42, 20)                   # window cells (no PML: pec on every face)
_PEC_WIDTHS = tuple((cells - 2) * _PEC_CELL for cells in _PEC_CELLS)   # (a, b) between the wall planes [m]
_PEC_WAVELENGTH = 1.55e-6               # m, mode_analysis run
_PEC_MODES = ((1, 0), (2, 0), (0, 1))   # (m, n) of modes 1..3
_PEC_TE_FRACTIONS = (0.0, 0.0, 1.0)     # |E_x|^2 share of modes 1..3
_PEC_DISPERSION_WAVELENGTHS = (1.45e-6, 1.55e-6, 1.65e-6)   # m, frequency_domain run (TE10 followed)
_PEC_TOLERANCE = 1e-5                   # relative n_eff / n_group, absolute te_fraction


def _pec_waveguide_spec(study):
    """em_modes in a vacuum PEC rectangle (2.1 x 1.0 um, 50 nm cells) with a mode table and a TE10 dispersion."""
    size_x, size_y = (cells * _PEC_CELL for cells in _PEC_CELLS)
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings.update(wavelength=_PEC_WAVELENGTH, n_modes=len(_PEC_MODES), group_index=study.type == 'mode_analysis')
    return ModelSpec(
        title=f'em_modes PEC waveguide ({study.type})',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec(_TABLE, 'mode_table'),
            FeatureSpec(_DISPERSION, 'mode_dispersion', params={'mode_number': 1})])],
        mesh=MeshSpec(mode='uniform', dx=_PEC_CELL),
        study=study,
        derived_values=[DerivedValueSpec(f'scalar_n_eff_{number}', f'{_PHYSICS}_n_eff_{number}')   # E5 scalars
                        for number in range(1, len(_PEC_MODES) + 1)] if study.type == 'mode_analysis' else [])


def _pec_waveguide_specs():
    return {'modes': _pec_waveguide_spec(StudySpec(type='mode_analysis')),
            'dispersion': _pec_waveguide_spec(StudySpec(type='frequency_domain', settings={
                'wavelengths': list(_PEC_DISPERSION_WAVELENGTHS)}))}


def _pec_discrete_index(wavelength, m, n):
    """Discrete n_eff of TE_mn between the wall planes (see the layout comment); NaN below cutoff."""
    k0 = 2.0 * math.pi / wavelength
    transverse = sum((2.0 / _PEC_CELL * math.sin(order * math.pi * _PEC_CELL / (2.0 * width))) ** 2
                     for order, width in zip((m, n), _PEC_WIDTHS))
    index_squared = 1.0 - transverse / k0 ** 2
    return math.sqrt(index_squared) if index_squared > 0.0 else math.nan


def _check_pec_waveguide(results_by_name):
    """n_eff of modes 1..3 and the TE10 dispersion (when a 'dispersion' run is given) vs the discrete closed form,
    te_fraction vs 0 / 1, n_group of TE10 vs 1 / n_eff; measured = error = the worst deviation in tolerance units."""
    rows = results_by_name['modes'].result(_PHYSICS, _TABLE)['rows']
    if len(rows) < len(_PEC_MODES):
        raise ValueError(f'expected {len(_PEC_MODES)} modes in the table, got {len(rows)}')
    expected = [_pec_discrete_index(_PEC_WAVELENGTH, m, n) for m, n in _PEC_MODES]
    n_eff = [float(row[_TABLE_COLUMNS['n_eff']]) for row in rows[:len(_PEC_MODES)]]
    te_fraction = [float(row[_TABLE_COLUMNS['te_fraction']]) for row in rows[:len(_PEC_MODES)]]
    n_group = float(rows[0][_TABLE_COLUMNS['n_group']])
    errors = {f'mode {number} n_eff': abs(measured / reference - 1.0)
              for number, (measured, reference) in enumerate(zip(n_eff, expected), start=1)}
    errors.update({f'mode {number} te_fraction': abs(measured - reference)
                   for number, (measured, reference) in enumerate(zip(te_fraction, _PEC_TE_FRACTIONS), start=1)})
    errors['mode 1 n_group'] = abs(n_group * expected[0] - 1.0)
    derived = results_by_name['modes'].summary()['derived']
    scalar_n_eff = [float(derived[f'scalar_n_eff_{number}']) for number in range(1, len(_PEC_MODES) + 1)]
    errors.update({f'mode {number} scalar n_eff_{number}': abs(scalar / table_value - 1.0)
                   for number, (scalar, table_value) in enumerate(zip(scalar_n_eff, n_eff), start=1)})
    continuum = [math.sqrt(max(0.0, 1.0 - sum((order * _PEC_WAVELENGTH / (2.0 * width)) ** 2
                                              for order, width in zip((m, n), _PEC_WIDTHS)))) for m, n in _PEC_MODES]
    details = {'n_eff': n_eff, 'scalar_n_eff': scalar_n_eff, 'n_eff_discrete': expected, 'n_eff_continuum': continuum,
               'te_fraction': te_fraction, 'n_group': n_group, 'n_group_expected': 1.0 / expected[0]}
    if 'dispersion' in results_by_name:
        dispersion = results_by_name['dispersion'].result(_PHYSICS, _DISPERSION)
        dispersion_expected = [_pec_discrete_index(wavelength, 1, 0) for wavelength in _PEC_DISPERSION_WAVELENGTHS]
        wavelengths = np.asarray(dispersion['wavelength'], dtype=np.float64)
        if wavelengths.shape != (len(_PEC_DISPERSION_WAVELENGTHS),) or not np.allclose(
                wavelengths, _PEC_DISPERSION_WAVELENGTHS, rtol=1e-9, atol=0.0):
            raise ValueError(f'dispersion: expected the study wavelengths {list(_PEC_DISPERSION_WAVELENGTHS)} m, got '
                             f'{float_list(wavelengths)}')
        dispersion_n_eff = np.asarray(dispersion['series']['n_eff'], dtype=np.float64)
        errors['dispersion n_eff'] = float(np.max(np.abs(dispersion_n_eff / np.asarray(dispersion_expected) - 1.0)))
        details.update(dispersion_n_eff=float_list(dispersion_n_eff), dispersion_n_eff_discrete=dispersion_expected,
                       dispersion_overlap=float_list(dispersion['series']['overlap']))
    return normalized_result({item: error / _PEC_TOLERANCE for item, error in errors.items()}, details, errors=errors)


# ---- Bloch parallel plate (symmetry 2.2): vacuum, x periodic with bloch_kx over L = 100 cells of 20 nm, pec y walls
# b = (22 - 2) h = 0.4 um apart (every K_y(n >= 1) mode is far below cutoff at 1.55 um). The discrete modes of the
# homogeneous Yee cross-section are beta^2 = k0^2 - K_x(m)^2 - K_y(n)^2 with K_x(m) = (2 / h) sin((k_x + 2 pi m / L) h
# / 2) on the phased wrap (the y-uniform n = 0 family has E_y only, one mode per m). At k_x = 0.3 k0 the three largest
# Re beta^2 are m = 0 and m = -1 (propagating) and m = +1 (evanescent, beta^2 < 0), all non-degenerate.
_BLOCH_CELL = 20e-9                     # m
_BLOCH_CELLS = (100, 22)                # window cells (no PML)
_BLOCH_WAVELENGTH = 1.55e-6             # m
_BLOCH_K_FRACTION = 0.3                 # bloch_kx = 0.3 k0
_BLOCH_ORDERS = (0, -1, 1)              # m of modes 1..3 (n = 0)
_BLOCH_TOLERANCE = 1e-8                 # |beta^2 - closed form| / k0^2 (eigen-solve round-off)


def _bloch_plate_spec():
    """em_modes in the x-periodic vacuum parallel plate at bloch_kx = 0.3 k0 on the cpu backend, a mode table."""
    size_x, size_y = (cells * _BLOCH_CELL for cells in _BLOCH_CELLS)
    k0 = 2.0 * math.pi / _BLOCH_WAVELENGTH
    settings = {'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec',
                'bloch_kx': _BLOCH_K_FRACTION * k0, 'wavelength': _BLOCH_WAVELENGTH, 'n_modes': len(_BLOCH_ORDERS)}
    spec = ModelSpec(
        title='em_modes Bloch parallel plate',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[FeatureSpec(_TABLE, 'mode_table')])],
        mesh=MeshSpec(mode='uniform', dx=_BLOCH_CELL),
        study=StudySpec(type='mode_analysis'),
        derived_values=[DerivedValueSpec('scalar_n_eff_1', f'{_PHYSICS}_n_eff_1')])
    spec.solver_defaults.backend = 'cpu'
    return spec


def _check_bloch_plate(results_by_name):
    """beta^2 of modes 1..3 (complex, from n_eff and n_eff_imag) vs the Bloch closed form, scalar n_eff_1 = the table
    value; measured = error = the worst |beta^2 - expected| / k0^2 over the tolerance."""
    results = results_by_name['run']
    rows = results.result(_PHYSICS, _TABLE)['rows']
    if len(rows) < len(_BLOCH_ORDERS):
        raise ValueError(f'expected {len(_BLOCH_ORDERS)} modes in the table, got {len(rows)}')
    k0 = 2.0 * math.pi / _BLOCH_WAVELENGTH
    bloch_kx = _BLOCH_K_FRACTION * k0
    cell = float(np.float32(_BLOCH_CELL))   # the float32 grid widths the operator uses (L off by -6e-9 otherwise)
    period = _BLOCH_CELLS[0] * cell
    expected = [1.0 - (2.0 / cell * math.sin(0.5 * (bloch_kx + 2.0 * math.pi * order / period) * cell)) ** 2 / k0 ** 2
                for order in _BLOCH_ORDERS]
    measured = [complex(float(row[1]), float(row[2])) ** 2 for row in rows[:len(_BLOCH_ORDERS)]]   # (beta / k0)^2
    errors = {f'mode {number} beta^2': abs(value - reference)
              for number, (value, reference) in enumerate(zip(measured, expected), start=1)}
    scalar = float(results.summary()['derived']['scalar_n_eff_1'])
    errors['scalar n_eff_1'] = abs(scalar - float(rows[0][1]))
    return normalized_result({item: error / _BLOCH_TOLERANCE for item, error in errors.items()},
                             {'n_eff_squared': [[value.real, value.imag] for value in measured],
                              'n_eff_squared_expected': expected}, errors=errors)


# ---- em_modes_line_block (solver_plan Part II stage 3): the PEC waveguide runs (mode_analysis with the group index:
# omega and omega (1 +- 1e-3) in one batch; frequency_domain at three wavelengths: one planned batch) and the Bloch
# plate (periodic x inside the blocks, cpu backend) with a line_block node as the 'shifted' solver. The dispersion run
# is double only: the planned-wavelength batch takes the same factorize_operators call as the modes batch, whose mixed
# precision modes_mixed covers.
_LINE_BLOCK_PRECISIONS = ('double', 'mixed')
_LINE_BLOCK_DISPERSION_PRECISION = 'double'
_LINE_BLOCK_BATCHES = {'modes': 3, 'dispersion': len(_PEC_DISPERSION_WAVELENGTHS), 'bloch': 1}   # batched_operators


def _with_line_block(spec, precision):
    """spec with a line_block node of `precision` assigned as the em_modes 'shifted' solver."""
    node = f'line_block_{precision}'
    spec.solvers.append(SolverSpec(node, 'line_block', {'precision': precision}))
    spec.physics_by_name(_PHYSICS).solvers['shifted'] = node
    return spec


def _line_block_specs():
    """'modes_<precision>', 'dispersion_double' (the PEC waveguide) and 'bloch_double' (the Bloch plate)."""
    specs = {}
    for precision in _LINE_BLOCK_PRECISIONS:
        for run, spec in _pec_waveguide_specs().items():
            if run == 'modes' or precision == _LINE_BLOCK_DISPERSION_PRECISION:
                specs[f'{run}_{precision}'] = _with_line_block(spec, precision)
    specs['bloch_double'] = _with_line_block(_bloch_plate_spec(), 'double')
    return specs


def _shifted_record(results):
    """(type, counters) of the summary 'solvers' shifted role record of the em_modes physics."""
    shifted = solver_role(results, _PHYSICS, 'shifted')
    return shifted.get('type', ''), shifted.get('counters') or {}


def _check_line_block(results_by_name):
    """The PEC waveguide checks per precision and the Bloch plate check (double), each in its own tolerance units, plus
    the shifted role record: type line_block, batched_operators as _LINE_BLOCK_BATCHES, rejected_factor_solves 0. The
    PEC (gpu) runs are judged only when they ran (skipped without a usable GPU)."""
    normalized, details = {}, {}
    outcomes = {f'pec {precision}': _check_pec_waveguide({
                    run: results_by_name[f'{run}_{precision}'] for run in ('modes', 'dispersion')
                    if f'{run}_{precision}' in results_by_name})
                for precision in _LINE_BLOCK_PRECISIONS if f'modes_{precision}' in results_by_name}
    outcomes['bloch double'] = _check_bloch_plate({'run': results_by_name['bloch_double']})
    for item, outcome in outcomes.items():
        normalized[item] = outcome['error'] / outcome['tolerance']
        details[item] = outcome['details']
    for name, results in results_by_name.items():
        solver_type, counters = _shifted_record(results)
        expected = _LINE_BLOCK_BATCHES[name.rsplit('_', 1)[0]]
        good = (solver_type == 'line_block' and float(counters.get('batched_operators', -1)) == expected
                and float(counters.get('rejected_factor_solves', math.inf)) == 0.0)
        normalized[f'{name} record'] = 0.0 if good else math.inf
        details[f'{name} record'] = {'type': solver_type, 'counters': dict(counters),
                                     'expected_batched_operators': expected}
    return normalized_result(normalized, details)


EM_MODES_SELFTEST_CASES = (
    SelfTestCase(
        name='em_modes_bloch_plate',
        label='EM modes: Bloch k on a periodic cross-section',
        description='em_modes on the cpu backend in a vacuum parallel plate, x periodic over L = 100 cells of 20 nm '
                    f'with bloch_kx = {_BLOCH_K_FRACTION:g} k0 at 1.55 um, pec y faces 0.4 um apart (22 cells). Each '
                    'mode is a Bloch wave of the phased wrap: the y-uniform family has beta^2 = k0^2 - K_x(m)^2 with '
                    'K_x(m) = (2 / h) sin((k_x + 2 pi m / L) h / 2), h = the float32 grid width, L = 100 h, exact '
                    'for the homogeneous Yee cross-section. '
                    'Expected: modes 1..3 are m = 0, -1 (propagating) and +1 (evanescent: n_eff 0, n_eff_imag < 0), '
                    'beta^2 = (n_eff + j n_eff_imag)^2 k0^2 to the closed form, and the derived value modes_n_eff_1 = '
                    'the table n_eff. A missing wrap phase gives the k = 0 values (|beta^2| change ~0.1 k0^2). '
                    f'measured = error = the worst |beta^2 - expected| / k0^2 over {_BLOCH_TOLERANCE:g}, tolerance 1.',
        specs=lambda: {'run': _bloch_plate_spec()},
        check=_check_bloch_plate),
    SelfTestCase(
        name='em_modes_line_block',
        label='EM modes: line-block shift-invert batches',
        description="solver_plan Part II stage 3: em_modes with a line_block solver node as its 'shifted' solver, "
                    'whose block factors replace the host SuperLU of the shift-invert operators A - sigma B (unknowns '
                    'ordered by grid rows or columns, a periodic axis inside the blocks). Runs: a vacuum PEC '
                    f'rectangle ({_PEC_CELLS[0]} x {_PEC_CELLS[1]} cells of {_PEC_CELL * 1e9:g} nm, wall planes on the '
                    'inner faces of the outermost cells) in mode_analysis at 1.55 um (n_modes 3, group index on: the '
                    'operators at omega and omega (1 +- '
                    '1e-3) in one factorize_operators batch) and frequency_domain runs (1.45, 1.55, 1.65 um: the '
                    'planned wavelengths in one batch), mode_analysis at precision double and mixed, frequency_domain '
                    'at double (the same factorize_operators call as the modes batch), and the em_modes_bloch_plate run '
                    '(periodic x with the Bloch phase, cpu backend, double). Each must pass its own check '
                    '(PEC: n_eff of TE10, TE20, TE01 vs the discrete closed form n_eff^2 = 1 - (K_x(m)^2 + K_y(n)^2) / '
                    'k0^2, K(m) = (2 / h) sin(m pi h / (2 a)), te_fraction 0, 0, 1, n_group of TE10 = 1 / n_eff, the '
                    'scalar outputs modes_n_eff_1..3 = the table n_eff and the mode-1 dispersion where run, all to '
                    f'{_PEC_TOLERANCE:g}; Bloch: beta^2 '
                    f'to {_BLOCH_TOLERANCE:g} k0^2), and the run summary must record the shifted role as line_block '
                    'with batched_operators 3 (modes), 3 (dispersion), 1 (Bloch) and rejected_factor_solves 0 (a '
                    'rejection redoes that eigen-solve on SuperLU). measured = error = the worst item in tolerance '
                    'units, tolerance 1. Not yet confirmed by a run.',
        specs=_line_block_specs,
        check=_check_line_block),
)


# ---- templates ----
# strip_waveguide_modes: a 500 x 220 nm silicon strip in silica, uniform 20 nm cells (core 25 x 11 cells, every edge
# on a cell edge; 22 cells per wavelength in silicon), 3.1 x 2.0 um window + 15 PML cells per side = 180 x 130 cells
# (46,800 unknowns). Expected n_eff of the quasi-TE0 mode ~2.4 (finer meshes converge to ~2.44), quasi-TM0 ~1.8.
STRIP_WAVEGUIDE_MODES_LABEL = 'Strip waveguide modes'
STRIP_WAVEGUIDE_MODES_DESCRIPTION = (
    'EM cross-section modes, mode_analysis: a 500 x 220 nm silicon strip in silica at 1.55 um (20 nm mesh, CPML on '
    'every side); mode table of 4 modes (n_eff, loss, TE fraction, effective area), derived n_eff_1 and te_fraction_1 '
    '(quasi-TE0: n_eff ~2.4, te_fraction ~1).')
_STRIP_CELL = 20e-9                         # m
_STRIP_DOMAIN = (3.1e-6, 2.0e-6)            # m
_STRIP_CORE = (1.30e-6, 0.90e-6, 1.80e-6, 1.12e-6)   # m, (x0, y0, x1, y1)
_STRIP_SETTINGS = {'wavelength': 1.55e-6, 'n_modes': 4, 'pml_sigma_factor': 0.7}   # 0.7 ~ 1 / n of silica


def strip_waveguide_modes_spec():
    """Template 'strip_waveguide_modes' (em_modes, mode_analysis): silicon strip in silica, mode table + n_eff_1."""
    x0, y0, x1, y1 = _STRIP_CORE
    return ModelSpec(
        title=STRIP_WAVEGUIDE_MODES_LABEL,
        domain=DomainSpec(size_x=_STRIP_DOMAIN[0], size_y=_STRIP_DOMAIN[1]),
        geometry=[GeometrySpec('core', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['core'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=dict(_STRIP_SETTINGS),
                             features=[FeatureSpec('mode_table', 'mode_table')])],
        mesh=MeshSpec(mode='uniform', dx=_STRIP_CELL),
        study=StudySpec(type='mode_analysis'),
        derived_values=[DerivedValueSpec('n_eff_1', 'n_eff', operation='at', source=f'{_PHYSICS}.mode_table', at=0),
                        DerivedValueSpec('te_fraction_1', 'te_fraction', operation='at',
                                         source=f'{_PHYSICS}.mode_table', at=0)])


# thermo_optic_phase_shifter: the strip of strip_waveguide_modes (core x 2.85..3.35, y 1.5..1.72 um) in a 6.2 x 4.0 um
# silica cross-section under a 2 x 0.15 um gold heater (y 2.72..2.87 um, 1 um above the core) heated by a heat_source
# Q = P_heater / (2 um x 0.15 um); the bottom face (the silicon substrate under the oxide) is held at 293.15 K, the
# other faces are insulated. Graded mesh: base 40 nm, 10 nm in the heater (2.3 cells per gold skin depth of ~23 nm, above
# the validation minimum 2), 20 nm in the core. Image-line estimate: dT_core ~ P / (2 pi k_silica) ln(4.22 / 1.0) ~
# 0.17 K per W/m, so 20 W/m (20 mW per mm of heater) gives dT ~3.3 K and dn_eff ~5e-4 (silicon dndT 1.8e-4 /K): ~2 rad
# per mm at 1.55 um. The sweep over P_heater (0, 10, 20 W/m) gives n_eff_1 and T_core per value.
THERMO_OPTIC_PHASE_SHIFTER_LABEL = 'Thermo-optic phase shifter'
THERMO_OPTIC_PHASE_SHIFTER_DESCRIPTION = (
    'Heat + EM cross-section modes, mode_analysis with a sweep over P_heater (0, 10, 20 W/m): a gold heater 1 um '
    'above a 500 x 220 nm silicon strip in silica, substrate face at 293.15 K; steady heat, heat_em_thermo_optic, then '
    'the modes on the heated permittivity; derived n_eff_1 (dn_eff ~5e-4 at 20 W/m: ~2 rad per mm) and T_core.')
_SHIFTER_DOMAIN = (6.2e-6, 4.0e-6)              # m
_SHIFTER_CORE = (2.85e-6, 1.50e-6, 3.35e-6, 1.72e-6)     # m
_SHIFTER_HEATER = (2.10e-6, 2.72e-6, 4.10e-6, 2.87e-6)   # m
_SHIFTER_MESH = {'dx': 40e-9, 'heater': (10e-9, 20e-9), 'core': (20e-9, 40e-9)}   # m: base, (dx, margin) per object
_SHIFTER_POWERS = (0.0, 10.0, 20.0)             # W/m


def thermo_optic_phase_shifter_spec():
    """Template 'thermo_optic_phase_shifter' (heat + em_modes, mode_analysis, sweep P_heater): n_eff of a heated
    silicon strip."""
    def rectangle(name, box):
        return GeometrySpec(name, 'rectangle', dict(zip(('x0', 'y0', 'x1', 'y1'), box)))

    heater_width, heater_height = _SHIFTER_HEATER[2] - _SHIFTER_HEATER[0], _SHIFTER_HEATER[3] - _SHIFTER_HEATER[1]
    return ModelSpec(
        title=THERMO_OPTIC_PHASE_SHIFTER_LABEL,
        parameters=[ParameterSpec('P_heater', _SHIFTER_POWERS[-1], 'heater power per metre of length [W/m]')],
        domain=DomainSpec(size_x=_SHIFTER_DOMAIN[0], size_y=_SHIFTER_DOMAIN[1]),
        geometry=[rectangle('core', _SHIFTER_CORE), rectangle('heater', _SHIFTER_HEATER)],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['core']),
                   MaterialSpec('gold_heater', library='gold', selection=['heater'])],
        physics=[PhysicsSpec('heat', 'heat', features=[
                     FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': 293.15}),
                     FeatureSpec('heater_power', 'heat_source', ['heater'], {'Q': f'P_heater / ({heater_width:g} * {heater_height:g})'})]),
                 PhysicsSpec(_PHYSICS, _TYPE, settings=dict(_STRIP_SETTINGS),
                             features=[FeatureSpec('mode_table', 'mode_table')])],
        couplings=[CouplingSpec('thermo_optic', 'heat_em_thermo_optic', 'heat', _PHYSICS)],
        mesh=MeshSpec(mode='nonuniform', dx=_SHIFTER_MESH['dx'],
                      refinements=[RefinementSpec(name, *_SHIFTER_MESH[name]) for name in ('heater', 'core')]),
        study=StudySpec(type='mode_analysis', sweep=SweepSpec('P_heater', list(_SHIFTER_POWERS))),
        derived_values=[DerivedValueSpec('n_eff_1', f'{_PHYSICS}_n_eff_1'),   # scalar output (E5)
                        DerivedValueSpec('T_core', 'T', ['core'], 'mean')])

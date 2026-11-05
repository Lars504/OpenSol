"""Self-tests of the conical / out-of-plane k_z operator (physics/em/conical, symmetry 2.9): CPU checks (Bloch empty
lattice, current-sheet closed forms) and cpu-backend model runs of em_fdfd_conical (driven current sheet, Bloch
eigenfrequencies); the walled-guide eigenfrequencies are the example em/analytic_conical_walled_guide."""
import math

import numpy as np

from ...core.constants import C0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, probe_phasor, single_value, vertical_line
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, SolverDefaultsSpec, SolverSpec, StudySpec)
from .selftests_cross_section import _isotropic

_TYPE = 'em_fdfd_conical'
_PHYSICS = 'em'

_WAVELENGTH = 1.55e-6                             # m, vacuum

# Empty lattice: all-Bloch uniform cell, (kx Lx / pi, ky Ly / pi, k_z [rad/m]) per point
_LATTICE_CELLS = (8, 6)
_LATTICE_SPACING = (50e-9, 40e-9)                 # m
_LATTICE_EPS, _LATTICE_MU = 2.25, 1.2
_LATTICE_POINTS = ((0.3, 0.2, 1.0e7), (-0.45, 0.7, 2.5e7), (0.0, 0.0, 4.0e6))
_LATTICE_SHIFT_INDEX = 3                          # shift between distinct exact values 3 and 4 (each doubly degenerate)
_LATTICE_COMPARED = 6
_LATTICE_TOLERANCE = 1e-9                         # |k0^2 - exact| / shift

# Current sheets: J_y and J_z sheets at one cell column, y periodic (2 cells), uniform eps
_SHEET_DX, _SHEET_DY = 20e-9, 30e-9
_SHEET_EPS = 2.25
_SHEET_KZ_FRACTION = 0.6                          # k_z / (k0 n)
_SHEET_PERIODIC_CELLS = 100                       # periodic x: exact standing wave
_SHEET_PERIODIC_TOLERANCE = 1e-9                  # max field error / max field
_SHEET_PML_CELLS, _SHEET_PML_INTERIOR = 20, 120   # cpml x: outgoing wave
_SHEET_PML_MARGIN = 10                            # interior cells next to the PML left out of the comparison
_SHEET_PML_TOLERANCE = 2e-3                       # max field error / |A| (CPML reflection)

# Model runs (cpu backend): the cpml sheet case through em_fdfd_conical (square 20 nm cells, the J_z sheet as a
# line_source of K = 1 A/m on window column 60, flux lines on columns 39 and 80, a probe 25 cells off the sheet), and
# the empty lattice (8 x 6 cells of 50 nm, eps 2.25, mu_r 1.25: binary-exact in float32) at its second point
_MODEL_SHEET_SOURCE, _MODEL_SHEET_PROBE = 60, 85  # window columns
_MODEL_FLUX_COLUMNS = (39, 80)                    # left / right flux lines (hi faces at 39.5, 80.5)
_MODEL_SHEET_AMPLITUDE = 1.0                      # A/m
_MODEL_IDENTITY_TOLERANCE = 1e-8                  # (flux_right - flux_left) / source power - 1: exact discrete theorem
_MODEL_LATTICE_SPACING = (50e-9, 50e-9)
_MODEL_LATTICE_MU = 1.25
_MODEL_LATTICE_POINT = _LATTICE_POINTS[1]
_MODEL_LATTICE_TOLERANCE = 1e-6                   # |k0^2 - exact| / shift (float32 cell widths of the mesh)

# cuDSS 'linear' back ends: the model sheet in a frequency_domain study at these wavelengths (k_z fixed, still below
# k0 n at the longest), Automatic SuperLU vs a sparse_direct node per back end; one plan for all wavelengths
_CUDSS_WAVELENGTHS = (1.5e-6, 1.55e-6, 1.6e-6)
_CUDSS_BACK_ENDS = ('cudss', 'cudss_mixed')
_CUDSS_NODE = 'sparse_direct_cudss'
_CUDSS_TOLERANCE = 1e-8                           # fluxes / probes relative to the largest SuperLU value (first estimate)


def _lattice_exact(kx, ky, kz, cells=_LATTICE_CELLS, spacing=_LATTICE_SPACING, eps_mu=_LATTICE_EPS * _LATTICE_MU):
    """Sorted exact discrete k0^2 of the uniform Bloch cell: (Kx^2 + Ky^2 + kz^2) / (eps mu) twice per harmonic."""
    (nx, ny), (dx, dy) = cells, spacing
    harmonics_x = kx + 2.0 * math.pi * np.arange(nx) / (nx * dx)
    harmonics_y = ky + 2.0 * math.pi * np.arange(ny) / (ny * dy)
    big_kx = (2.0 / dx) * np.sin(harmonics_x * dx / 2.0)
    big_ky = (2.0 / dy) * np.sin(harmonics_y * dy / 2.0)
    values = ((big_kx[:, None] ** 2 + big_ky[None, :] ** 2 + kz ** 2) / eps_mu).ravel()
    return np.sort(np.concatenate((values, values)))


def _lattice_expected(exact):
    """(target k0^2 between the 3rd and 4th distinct exact values, shift, the _LATTICE_COMPARED exact values the solve
    returns: nearest by |sqrt(k0^2) - sqrt(target)|, as it ranks by |omega - omega_guess|)."""
    distinct = np.unique(np.round(exact / exact[-1], 12)) * exact[-1]
    target = 0.5 * (distinct[_LATTICE_SHIFT_INDEX - 1] + distinct[_LATTICE_SHIFT_INDEX])
    expected = exact[np.argsort(np.abs(np.sqrt(exact) - math.sqrt(target)), kind='stable')][:_LATTICE_COMPARED]
    return target, target * (1.0 + 1e-6), expected


def _check_empty_lattice(results_by_name):
    """Uniform all-Bloch cell: eigen k0^2 at fixed (kx, ky, k_z) vs the exact Yee plane-wave dispersion."""
    from . import conical, cross_section_modes
    from .boundaries import BoundaryConfig
    (nx, ny), (dx, dy) = _LATTICE_CELLS, _LATTICE_SPACING
    widths = (np.full(nx, dx), np.full(ny, dy))
    materials = _isotropic(cross_section_modes, np.full((nx, ny), _LATTICE_EPS), _LATTICE_MU)
    errors, violations, details = {}, [], {}
    for number, (fraction_x, fraction_y, kz) in enumerate(_LATTICE_POINTS):
        config = BoundaryConfig('bloch', 'bloch', 'bloch', 'bloch')
        config.bloch_kx, config.bloch_ky = fraction_x * math.pi / (nx * dx), fraction_y * math.pi / (ny * dy)
        target, shift, expected = _lattice_expected(_lattice_exact(config.bloch_kx, config.bloch_ky, kz))
        if np.max(np.abs(expected - shift)) >= 0.9 * shift:
            violations.append(f'point {number}: compared values are not nearer the shift than the static solutions')
        solve = conical.solve_conical_resonances(widths, config, dx, materials, kz, _LATTICE_COMPARED,
                                                 C0 * math.sqrt(target))
        measured = np.sort((solve.omegas / C0) ** 2)
        if measured.size != _LATTICE_COMPARED:
            violations.append(f'point {number}: {measured.size} resonances returned, expected {_LATTICE_COMPARED}')
            continue
        errors[f'point_{number}'] = float(np.max(np.abs(measured - np.sort(expected)))) / shift
        details[f'point_{number}'] = {'k0_squared': measured.real.tolist(), 'exact': np.sort(expected).tolist(),
                                      'operator_applications': solve.info['operator_applications']}
    measured_error = max(errors.values(), default=float('nan'))
    return {'measured': measured_error, 'expected': 0.0, 'error': measured_error + len(violations),
            'tolerance': _LATTICE_TOLERANCE, 'details': {'errors': errors, 'points': details,
                                                         'violations': violations}}


def _sheet_solve(conical, cross_section_modes, config, nx, source_column):
    """(fields, k0, k_z, k^2, theta) of unit J_y and J_z sheets [A/m^2] at source_column (all y cells)."""
    ny = 2
    k0 = 2.0 * math.pi / _WAVELENGTH
    kz = _SHEET_KZ_FRACTION * k0 * math.sqrt(_SHEET_EPS)
    k_squared = k0 ** 2 * _SHEET_EPS - kz ** 2
    theta = math.acos(1.0 - 0.5 * k_squared * _SHEET_DX ** 2)
    materials = _isotropic(cross_section_modes, np.full((nx, ny), _SHEET_EPS))
    system = conical.assemble_conical((np.full(nx, _SHEET_DX), np.full(ny, _SHEET_DY)), config, _SHEET_DX, materials,
                                      C0 * k0, kz)
    sheet = np.zeros((nx, ny), dtype=np.complex128)
    sheet[source_column, :] = 1.0
    fields, _info = conical.solve_conical_driven(system, {'y': sheet, 'z': sheet})
    return fields, k0, kz, k_squared, theta


def _sheet_errors(fields, profile, columns, k0, kz, k_squared, scale):
    """{component: max |field - closed form| / scale} over the cell columns; profile(columns) = the 1D Green's function
    of -D^2 - k^2 per unit s (times dx^2), s = j omega mu0 J; E_x on the hi-x faces = -j k_z D^h E_z / k^2."""
    source = 1j * C0 * k0 * MU0
    columns = np.asarray(columns)
    green, green_next = profile(columns), profile(columns + 1)
    z_factor = k_squared / (k0 ** 2 * _SHEET_EPS)
    expected = {'ey': source * green, 'ez': z_factor * source * green,
                'ex': -1j * kz * z_factor * source * (green_next - green) / (k_squared * _SHEET_DX)}
    return {name: float(np.max(np.abs(fields[name][columns, :] - value[:, None]))) / scale
            for name, value in expected.items()}


def _check_current_sheets(results_by_name):
    """J_y / J_z current sheets at k_z: periodic x standing wave (exact) and cpml x outgoing wave vs closed forms."""
    from . import conical, cross_section_modes
    from .boundaries import BoundaryConfig
    violations = []
    # periodic x: E_n = -s dx^2 cos(theta (n - N/2)) / (2 sin theta sin(theta N / 2)), source at column 0
    n_cells = _SHEET_PERIODIC_CELLS
    config = BoundaryConfig('periodic', 'periodic', 'periodic', 'periodic')
    fields, k0, kz, k_squared, theta = _sheet_solve(conical, cross_section_modes, config, n_cells, 0)
    resonance_margin = abs(math.sin(theta * n_cells / 2.0))
    if resonance_margin < 0.1:
        violations.append(f'periodic sheet: |sin(theta N / 2)| = {resonance_margin:.3g} is near a cavity resonance')

    def standing(distance):
        wrapped = np.mod(distance, n_cells)
        return (-_SHEET_DX ** 2 * np.cos(theta * (wrapped - n_cells / 2.0))
                / (2.0 * math.sin(theta) * math.sin(theta * n_cells / 2.0)))

    columns = np.arange(n_cells)
    peak = max(float(np.max(np.abs(fields[name]))) for name in ('ex', 'ey', 'ez'))
    periodic_errors = _sheet_errors(fields, standing, columns, k0, kz, k_squared, peak)
    # cpml x: E_n = s dx^2 exp(-j theta |n - n0|) / (2 j sin theta)
    config = BoundaryConfig('cpml', 'cpml', 'periodic', 'periodic')
    config.cpml_npml = _SHEET_PML_CELLS
    n_cells = 2 * _SHEET_PML_CELLS + _SHEET_PML_INTERIOR
    source_column = n_cells // 2
    fields, k0, kz, k_squared, theta = _sheet_solve(conical, cross_section_modes, config, n_cells, source_column)

    def outgoing(column):
        return _SHEET_DX ** 2 * np.exp(-1j * theta * np.abs(column - source_column)) / (2j * math.sin(theta))

    first, last = _SHEET_PML_CELLS + _SHEET_PML_MARGIN, n_cells - _SHEET_PML_CELLS - _SHEET_PML_MARGIN - 1
    columns = np.arange(first, last)
    amplitude = abs(C0 * k0 * MU0 * _SHEET_DX ** 2 / (2.0 * math.sin(theta)))
    pml_errors = _sheet_errors(fields, outgoing, columns, k0, kz, k_squared, amplitude)
    normalised = {**{f'periodic_{name}': value / _SHEET_PERIODIC_TOLERANCE for name, value in periodic_errors.items()},
                  **{f'pml_{name}': value / _SHEET_PML_TOLERANCE for name, value in pml_errors.items()}}
    measured = max(normalised.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': 1.0,
            'details': {'periodic_errors': periodic_errors, 'pml_errors': pml_errors, 'normalised': normalised,
                        'theta': theta, 'k_z': kz, 'violations': violations}}


CONICAL_CPU_SELFTESTS = (
    SelfTestCase(
        name='em_conical_empty_lattice',
        label='EM conical k_z: Bloch empty lattice',
        description='CPU check of physics/em/conical (no model run): a uniform all-Bloch cell (8 x 6 cells of 50 x 40 '
                    'nm, eps 2.25, mu 1.2) at three (kx, ky, k_z) points, both in-plane Bloch phases and k_z != 0 '
                    'together. Every discrete plane wave of harmonic (p, q) is an exact Yee solution with k0^2 eps mu '
                    '= Kx^2 + Ky^2 + k_z^2, K = (2/d) sin((k + 2 pi p / L) d / 2), twice (two polarisations); the '
                    'curl-free fields solve it at k0 = 0 and are dropped. The 6 eigenvalues nearest a shift between '
                    'the 3rd and 4th distinct exact values must match. error = max |k0^2 - exact| / shift + failed '
                    'checks; tolerance 1e-9 (round-off of the shift-invert solve).',
        specs=lambda: {},
        check=_check_empty_lattice,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_conical_current_sheets',
        label='EM conical k_z: current sheets',
        description='CPU check of the conical driven solve (no model run): unit J_y and J_z sheets on one cell column '
                    '(y periodic, 2 cells, dx 20 nm, eps 2.25, 1.55 um, k_z = 0.6 k0 n). Uniform in y they decouple '
                    'into exact 1D problems -D^2 E - k^2 E = c s delta, k^2 = k0^2 eps - k_z^2, s = j omega mu0 J: E_y '
                    'with c = 1, E_z with c = k^2 / (k0^2 eps) and E_x = -j k_z D^h E_z / k^2 (the k_z coupling of '
                    'the x row). Periodic x (100 cells): the standing wave -s dx^2 cos(theta (n - N/2)) / (2 sin '
                    'theta sin(theta N / 2)), cos theta = 1 - k^2 dx^2 / 2, to round-off (1e-9 of the peak field); '
                    '20 cpml cells on both x faces: the outgoing s dx^2 exp(-j theta |n - n0|) / (2 j sin theta) over '
                    'the interior 10 cells away from the layers, 2e-3 of |A| (the CPML reflection). error = max of '
                    'the errors over their tolerances + failed checks; tolerance 1.',
        specs=lambda: {},
        check=_check_current_sheets,
        max_wall_time=60.0),
)


def _sheet_wavenumbers():
    """(k0, k_z, k^2 = k0^2 eps - k_z^2, theta with cos theta = 1 - k^2 dx^2 / 2) of the sheet cases."""
    k0 = 2.0 * math.pi / _WAVELENGTH
    kz = _SHEET_KZ_FRACTION * k0 * math.sqrt(_SHEET_EPS)
    k_squared = k0 ** 2 * _SHEET_EPS - kz ** 2
    return k0, kz, k_squared, math.acos(1.0 - 0.5 * k_squared * _SHEET_DX ** 2)


def _column_x(column):
    """x [m] of the centre of window column `column` of the 20 nm sheet mesh."""
    return (column + 0.5) * _SHEET_DX


def _model_sheet_specs():
    """{'sheet': em_fdfd_conical stationary run of the cpml J_z sheet (line_source K = 1 A/m), two ez probes and two
    full-period flux lines; cpu backend}."""
    _, kz, _, _ = _sheet_wavenumbers()
    size_x, size_y = _SHEET_PML_INTERIOR * _SHEET_DX, 2 * _SHEET_DX
    left, right = _MODEL_FLUX_COLUMNS
    geometry = [GeometrySpec('medium', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': size_x, 'y1': size_y}),
                vertical_line('sheet', _column_x(_MODEL_SHEET_SOURCE), 0.0, size_y),
                vertical_line('left', _column_x(left), 0.0, size_y),
                vertical_line('right', _column_x(right), 0.0, size_y),
                GeometrySpec('at_source', 'point', {'x': _column_x(_MODEL_SHEET_SOURCE), 'y': 0.5 * _SHEET_DX}),
                GeometrySpec('off_source', 'point', {'x': _column_x(_MODEL_SHEET_PROBE), 'y': 0.5 * _SHEET_DX})]
    features = [FeatureSpec('sheet', 'line_source', ['sheet'], {'component': 'z', 'amplitude': _MODEL_SHEET_AMPLITUDE}),
                FeatureSpec('probe_source', 'field_probe', ['at_source'], {'component': 'ez'}),
                FeatureSpec('probe_off', 'field_probe', ['off_source'], {'component': 'ez'}),
                FeatureSpec('flux_left', 'flux_monitor', ['left']),
                FeatureSpec('flux_right', 'flux_monitor', ['right'])]
    settings = {'bc_x_lo': 'cpml', 'bc_x_hi': 'cpml', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic',
                'pml_cells': _SHEET_PML_CELLS, 'wavelength': _WAVELENGTH, 'k_z': kz}
    return {'sheet': ModelSpec(
        title='em_fdfd_conical current sheet at k_z',
        domain=DomainSpec(size_x=size_x, size_y=size_y), geometry=geometry,
        materials=[MaterialSpec('medium', properties={'eps_r': _SHEET_EPS}, selection=['medium'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_SHEET_DX), study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('P_source', f'{_PHYSICS}_source_power')],
        solver_defaults=SolverDefaultsSpec(backend='cpu'))}


def _check_model_sheet(results_by_name):
    """Flux identity (right - left line = source power), ez probes vs the outgoing closed form, half the power per
    side."""
    results = results_by_name['sheet']
    k0, _, k_squared, theta = _sheet_wavenumbers()
    flux_left = single_value(results.result(_PHYSICS, 'flux_left')['power'], 'flux_left')
    flux_right = single_value(results.result(_PHYSICS, 'flux_right')['power'], 'flux_right')
    source = float(results.summary()['derived']['P_source'])
    density = _MODEL_SHEET_AMPLITUDE / _SHEET_DX   # J_z of the sheet cells [A/m^2]
    z_factor = k_squared / (k0 ** 2 * _SHEET_EPS)
    # z_factor (j omega mu0 J) dx^2 / (2 j sin theta): the (real) E_z on the sheet
    scale = z_factor * C0 * k0 * MU0 * density * _SHEET_DX ** 2 / (2.0 * math.sin(theta))

    def outgoing(distance):
        return scale * np.exp(-1j * theta * abs(distance))

    measured = {'probe_source': probe_phasor(results, _PHYSICS, 'probe_source'),
                'probe_off': probe_phasor(results, _PHYSICS, 'probe_off')}
    expected = {'probe_source': outgoing(0), 'probe_off': outgoing(_MODEL_SHEET_PROBE - _MODEL_SHEET_SOURCE)}
    normalised = {name: abs(measured[name] - expected[name]) / abs(scale) / _SHEET_PML_TOLERANCE for name in measured}
    normalised['flux_identity'] = abs((flux_right - flux_left) / source - 1.0) / _MODEL_IDENTITY_TOLERANCE
    normalised['half_power'] = abs(flux_right / (0.5 * source) - 1.0) / (5.0 * _SHEET_PML_TOLERANCE)
    error = max(normalised.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'normalised': normalised, 'flux_left': flux_left, 'flux_right': flux_right,
                        'source_power': source, 'ez_measured': {name: [value.real, value.imag]
                                                                 for name, value in measured.items()},
                        'ez_expected': {name: [complex(value).real, complex(value).imag]
                                        for name, value in expected.items()}}}


def _model_lattice_target():
    """(bloch_kx, bloch_ky, k_z, target, shift, expected k0^2) of the model lattice point."""
    (nx, ny), (dx, dy) = _LATTICE_CELLS, _MODEL_LATTICE_SPACING
    fraction_x, fraction_y, kz = _MODEL_LATTICE_POINT
    kx, ky = fraction_x * math.pi / (nx * dx), fraction_y * math.pi / (ny * dy)
    exact = _lattice_exact(kx, ky, kz, _LATTICE_CELLS, _MODEL_LATTICE_SPACING, _LATTICE_EPS * _MODEL_LATTICE_MU)
    return (kx, ky, kz) + _lattice_expected(exact)


def _model_lattice_specs():
    """{'lattice': em_fdfd_conical eigenfrequency run of the uniform all-Bloch cell at (kx, ky, k_z); cpu backend}."""
    (nx, ny), (dx, dy) = _LATTICE_CELLS, _MODEL_LATTICE_SPACING
    kx, ky, kz, target, _, _ = _model_lattice_target()
    settings = {'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic',
                'bloch_kx': kx, 'bloch_ky': ky, 'k_z': kz, 'wavelength': 2.0 * math.pi / math.sqrt(target)}
    return {'lattice': ModelSpec(
        title='em_fdfd_conical Bloch empty lattice at k_z',
        domain=DomainSpec(size_x=nx * dx, size_y=ny * dy),
        geometry=[GeometrySpec('medium', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': nx * dx, 'y1': ny * dy})],
        materials=[MaterialSpec('medium', properties={'eps_r': _LATTICE_EPS, 'mu_r': _MODEL_LATTICE_MU},
                                selection=['medium'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings,
                             features=[FeatureSpec('resonances', 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=dx),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _LATTICE_COMPARED}),
        solver_defaults=SolverDefaultsSpec(backend='cpu'))}


def _check_model_lattice(results_by_name):
    """Resonance table k0^2 vs the exact Yee plane-wave dispersion, one snapshot per resonance."""
    results = results_by_name['lattice']
    _, _, _, _, shift, expected = _model_lattice_target()
    rows = results.result(_PHYSICS, 'resonances')['rows']
    measured = np.sort([(2.0 * math.pi * float(row[2]) / C0) ** 2 for row in rows])
    labels = results.summary().get('snapshot_labels') or []
    violations = []
    if measured.size != _LATTICE_COMPARED:
        violations.append(f'{measured.size} resonances in the table, expected {_LATTICE_COMPARED}')
    if len(labels) != measured.size:
        violations.append(f'{len(labels)} snapshots for {measured.size} resonances')
    error = (float(np.max(np.abs(measured - np.sort(expected)))) / shift if measured.size == _LATTICE_COMPARED
             else math.inf)
    return {'measured': error, 'expected': 0.0, 'error': error + len(violations), 'tolerance': _MODEL_LATTICE_TOLERANCE,
            'details': {'k0_squared': float_list(measured), 'exact': float_list(np.sort(expected)),
                        'snapshot_labels': list(labels), 'violations': violations}}


def _cudss_specs():
    """'sheet_sparse_direct' (a scipy_splu node: Automatic may resolve to cuDSS on the gpu) and 'sheet_<back end>'
    frequency_domain runs of the model sheet (gpu); {} without nvmath-python."""
    from ...solvers.cudss_factor import cudss_available
    if not cudss_available():
        return {}
    specs = {}
    for back_end in ('sparse_direct',) + _CUDSS_BACK_ENDS:
        spec = _model_sheet_specs()['sheet']
        spec.solver_defaults.backend = 'gpu'
        physics = spec.physics_by_name(_PHYSICS)
        physics.settings.pop('wavelength', None)
        spec.study = StudySpec(type='frequency_domain', settings={'wavelengths': list(_CUDSS_WAVELENGTHS)})
        node_back_end = 'scipy_splu' if back_end == 'sparse_direct' else back_end   # the reference: host SuperLU
        spec.solvers.append(SolverSpec(_CUDSS_NODE, 'sparse_direct', {'back_end': node_back_end}))
        physics.solvers['linear'] = _CUDSS_NODE
        specs[f'sheet_{back_end}'] = spec
    return specs


def _cudss_compared_values(results):
    """complex128 per-wavelength flux_left, flux_right and the two ez probe phasors of a sheet run."""
    values = [np.asarray(results.result(_PHYSICS, name)['power'], dtype=np.complex128)
              for name in ('flux_left', 'flux_right')]
    for name in ('probe_source', 'probe_off'):
        series = results.result(_PHYSICS, name)['series']
        values.append(np.asarray(series['abs'], dtype=np.float64) * np.exp(1j * np.asarray(series['phase'],
                                                                                           dtype=np.float64)))
    return np.concatenate(values)


def _check_cudss(results_by_name):
    """Per back end: max |value - SuperLU| / max |SuperLU| over fluxes and probes in tolerance units, and the linear
    record: that back end, one LDL^T per wavelength on one plan (+ one per cudss_mixed double fallback)."""
    if not results_by_name:
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': 1.0,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    reference = _cudss_compared_values(results_by_name['sheet_sparse_direct'])
    scale = float(np.max(np.abs(reference)))
    n_wavelengths = len(_CUDSS_WAVELENGTHS)
    errors, details = {}, {}
    for back_end in _CUDSS_BACK_ENDS:
        results = results_by_name[f'sheet_{back_end}']
        values = _cudss_compared_values(results)
        deviation = float(np.max(np.abs(values - reference))) / scale if values.shape == reference.shape else math.inf
        roles = ((results.summary().get('solvers') or {}).get(_PHYSICS) or {}).get('roles') or {}
        counters = (roles.get('linear') or {}).get('counters') or {}
        expected = {'back_end': back_end, 'factorizations': n_wavelengths, 'cudss_symmetric_factors': n_wavelengths,
                    'cudss_plans': 1 + int(counters.get('cudss_double_fallbacks') or 0)}
        record = {name: counters.get(name) for name in expected}
        errors[back_end] = deviation / _CUDSS_TOLERANCE
        errors[f'{back_end} record'] = 0.0 if record == expected else math.inf
        details[back_end] = {'deviation': deviation, 'counters': dict(counters)}
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    details.update(normalised_errors=errors, worst_item=worst)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


CONICAL_MODEL_SELFTESTS = (
    SelfTestCase(
        name='em_fdfd_conical_sheet',
        label='EM conical FDFD: current sheet model run',
        description='Model run of em_fdfd_conical (stationary, cpu backend) of the cpml case of em_conical_current_sheets '
                    f'with square {_SHEET_DX * 1e9:g} nm cells: a z line_source of K = 1 A/m across the 2-cell y period '
                    '(J_z = K / dx), eps 2.25, 1.55 um, k_z = 0.6 k0 n, 20 cpml cells on both x faces. Normalised '
                    'errors (passes at <= 1): flux_right - flux_left of two full-period vertical flux lines enclosing '
                    'the sheet vs the scalar output source_power (the exact discrete Poynting theorem, '
                    f'{_MODEL_IDENTITY_TOLERANCE:g}), the ez probes on and 25 cells off the sheet vs the outgoing '
                    'closed form (k^2 / (k0^2 eps)) j omega mu0 J dx^2 exp(-j theta |n - n0|) / (2 j sin theta) '
                    f'({_SHEET_PML_TOLERANCE:g} of its modulus: the CPML reflection) and flux_right vs half the source '
                    f'power ({5 * _SHEET_PML_TOLERANCE:g}). Pins the model layer (k_z setting, line source, probes, flux '
                    'pairing and sign, source power).',
        specs=_model_sheet_specs,
        check=_check_model_sheet),
    SelfTestCase(
        name='em_fdfd_conical_lattice',
        label='EM conical FDFD: Bloch eigenfrequencies model run',
        description='Model run of em_fdfd_conical (eigenfrequency, cpu backend): the uniform all-periodic cell of '
                    'em_conical_empty_lattice on a square 50 nm mesh (8 x 6 cells, eps 2.25, mu_r 1.25) with bloch_kx '
                    f'= {_MODEL_LATTICE_POINT[0]:g} pi / Lx, bloch_ky = {_MODEL_LATTICE_POINT[1]:g} pi / Ly and k_z = '
                    f'{_MODEL_LATTICE_POINT[2]:g} rad/m, the wavelength setting between the 3rd and 4th distinct exact '
                    'values. The resonance_table k0^2 = (2 pi f / C0)^2 of its 6 resonances vs the exact discrete '
                    f'(Kx^2 + Ky^2 + k_z^2) / (eps mu) nearest the target, within {_MODEL_LATTICE_TOLERANCE:g} of the '
                    'shift (float32 mesh widths), one snapshot per resonance. Pins the k_z and Bloch settings through '
                    'the model, the eigen hook and the resonance table.',
        specs=_model_lattice_specs,
        check=_check_model_lattice),
    SelfTestCase(
        name='em_fdfd_conical_cudss',
        label="EM conical FDFD: cuDSS 'linear' back ends vs SuperLU",
        description='The em_fdfd_conical_sheet model (cpml x, periodic y without a Bloch phase, k_z fixed) in a '
                    f"frequency_domain study at {', '.join(f'{value * 1e6:g}' for value in _CUDSS_WAVELENGTHS)} um "
                    '(gpu), solved by the Automatic sparse_direct (host SuperLU) and by a sparse_direct node with '
                    "back_end cudss and cudss_mixed as the 'linear' solver (3 runs). cuDSS factors D A (D the dual "
                    'areas times the CPML stretches, the E_z block negated: complex symmetric) as an LDL^T and '
                    'refactorises the kept factor for the next wavelength (one plan). Per back end max |value - '
                    'SuperLU| / max |SuperLU| over both flux lines and both ez probe phasors at every wavelength below '
                    f'{_CUDSS_TOLERANCE:g}, and the linear record: that back end, factorizations = '
                    f'cudss_symmetric_factors = {len(_CUDSS_WAVELENGTHS)} on 1 plan (+ 1 per cudss_mixed double '
                    'fallback). measured = error = the worst item in tolerance units, tolerance 1; passes as skipped '
                    'without nvmath-python. Run on an RTX 3050 (0.4 s): deviations 8e-14 (cudss) / 4e-14 '
                    '(cudss_mixed, unrefined residual 1e-5), 3 LDL^T on 1 plan, no fallback.',
        specs=_cudss_specs,
        check=_check_cudss),
)

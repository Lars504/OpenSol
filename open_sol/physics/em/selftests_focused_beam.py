"""Self-tests of the focused_beam source: em_fdfd_ez / em_fdfd_hz power balance and one-sided launch, the focal field
against the continuum angular-spectrum integral, and solve_moved_beam against a rebuilt focus together with the
discrete residual of the incident wave (model runs on the cpu; the moved case scripted); no physics imports."""
# Layout of the model runs: water (eps_r 1.78223, n = 1.335) everywhere, vacuum wavelength 532 nm (398 nm in water),
# uniform 15 nm mesh (26.6 cells per wavelength in the medium), window 290 x 160 cells (4.35 x 2.4 um), 20-cell CPML
# on every face (330 x 200 = 66,000 unknowns per run). The beam line spans the whole window width in row 26 (y =
# 0.3975 um, interface = the lo face of row 26), direction positive (+y), waist 0.5 um, focus on the centre of cell
# (145, 93) (2.1825, 1.4025) um, 67 cells (1.005 um) downstream; the beam is ~0.56 um wide at the line, so its
# continuum level at the line ends (2.18 um from the focus) is ~3e-7 (< the 1e-6 validation level). Flux monitors span
# the window width: downstream in row 133 (face at 2.01 um, 26 cells from the PML), upstream in row 12 (scattered
# field).
import math

from ...core.constants import C0, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (cell_centre, max_relative_error, no_runs, normalized_result, probe_phasor,
                                       single_value, worst_result)
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_fdfd import _PHYSICS_NAME, cpu_specs

_POLARIZATIONS = ('ez', 'hz')
_BEAM = 'beam'
_DOWNSTREAM, _UPSTREAM = 'downstream', 'upstream'
_BEAM_WAVELENGTH = 532e-9          # m, vacuum
_WATER_EPS_R = 1.78223             # n = 1.3350 at 532 nm (materials water_532nm)
_BEAM_CELL = 15e-9                 # m
_BEAM_CELLS = (290, 160)           # window cells (x, y)
_BEAM_PML = 20                     # cells per CPML face
_LINE_ROW = 26                     # window row of the beam line
_FOCUS_CELL = (145, 93)            # window cell whose centre is the focus
_DOWNSTREAM_ROW, _UPSTREAM_ROW = 133, 12
_BEAM_WAIST = 0.5e-6               # m
_BEAM_POWER = 1.0                  # W/m
_FOCUS_APERTURES = {'ez': None, 'hz': 1.2}   # hz also exercises the numerical-aperture cut (Gaussian 3.5e-6 there)
_PROBE_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))   # cells from the focus cell; the first is the focus
_POWER_TOLERANCE = 1e-6            # relative, downstream - upstream flux vs power (discrete-exact flux)
_UPSTREAM_TOLERANCE = 1e-6         # of power: the upstream flux = PML reflection only (first estimate)
_FOCUS_TOLERANCE = 1e-2            # relative |u| at the focus vs the continuum (discrete flux ~ (k h)^2 / 12 in |u|)

# em_fdfd_focused_beam_moved (scripted, cpu): water, 20 nm cells, window 160 x 110 cells, 15-cell CPML (190 x 140 =
# 26,600 unknowns), the beam line over the window width in row 15, focus on the centre of cell (80, 60), moved by a
# non-integer number of cells.
_MOVED_CELL = 20e-9
_MOVED_CELLS = (160, 110)
_MOVED_PML = 15
_MOVED_LINE_ROW = 15
_MOVED_FOCUS_CELL = (80, 60)
_MOVED_OFFSET = (0.113e-6, -0.071e-6)   # m (5.65 and -3.55 cells)
_MOVED_TOLERANCE = 1e-10           # relative: identical operators and right-hand sides on SuperLU (round-off)


# ---- specs ----
def _probe_name(index):
    return f'probe_{index}'


def _beam_spec(polarization, aperture=None, probes=False):
    """The model-run layout (module comment) in em_fdfd_<polarization>: the focused beam, both flux monitors and (probes)
    field probes of the out-of-plane field at the focus cell and its four neighbours."""
    size_x, size_y = (count * _BEAM_CELL for count in _BEAM_CELLS)
    line_y = cell_centre(_LINE_ROW, _BEAM_CELL)
    focus = tuple(cell_centre(index, _BEAM_CELL) for index in _FOCUS_CELL)
    geometry = [GeometrySpec('beam_line', 'line', {'x0': 0.0, 'y0': line_y, 'x1': size_x, 'y1': line_y})]
    features = [FeatureSpec(_BEAM, 'focused_beam', ['beam_line'], {
        'power': _BEAM_POWER, 'phase': 0.0, 'waist': _BEAM_WAIST, 'numerical_aperture': aperture,
        'focus_x': focus[0], 'focus_y': focus[1], 'direction': 'positive'})]
    for name, row in ((_DOWNSTREAM, _DOWNSTREAM_ROW), (_UPSTREAM, _UPSTREAM_ROW)):
        y = cell_centre(row, _BEAM_CELL)
        geometry.append(GeometrySpec(f'{name}_line', 'line', {'x0': 0.0, 'y0': y, 'x1': size_x, 'y1': y}))
        features.append(FeatureSpec(name, 'flux_monitor', [f'{name}_line']))
    if probes:
        for index, (offset_i, offset_j) in enumerate(_PROBE_OFFSETS):
            point = f'probe_point_{index}'
            geometry.append(GeometrySpec(point, 'point', {
                'x': cell_centre(_FOCUS_CELL[0] + offset_i, _BEAM_CELL),
                'y': cell_centre(_FOCUS_CELL[1] + offset_j, _BEAM_CELL)}))
            features.append(FeatureSpec(_probe_name(index), 'field_probe', [point], {'component': polarization}))
    physics_type = f'em_fdfd_{polarization}'
    return ModelSpec(
        title=f'Self-test: {physics_type} focused beam in water',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('immersion_water', properties={'eps_r': _WATER_EPS_R}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type,
                             settings={'wavelength': _BEAM_WAVELENGTH, 'pml_cells': _BEAM_PML}, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_BEAM_CELL),
        study=StudySpec(type='stationary'))


def focused_beam_power_specs():
    """{polarization: spec} of em_fdfd_focused_beam_power (no backend set: em_fdfd_cpu_gpu reuses them per backend)."""
    return {polarization: _beam_spec(polarization) for polarization in _POLARIZATIONS}


def _focus_specs():
    """{polarization: spec} of em_fdfd_focused_beam_focus: the probes, hz with its numerical aperture."""
    return {polarization: _beam_spec(polarization, _FOCUS_APERTURES[polarization], probes=True)
            for polarization in _POLARIZATIONS}


# ---- checks ----
def _flux(results, feature):
    return single_value(results.result(_PHYSICS_NAME, feature)['power'], f'{feature} flux')


def _check_power(results_by_name):
    """Per polarisation: (downstream - upstream flux) / power vs 1 (whatever the PML reflects passes both lines, so the
    difference is the injected power exactly) and |upstream flux| / power vs 0, in tolerance units."""
    normalized, details = {}, {'power': _BEAM_POWER, 'unit': 'W/m'}
    for polarization in _POLARIZATIONS:
        results = results_by_name[polarization]
        downstream, upstream = _flux(results, _DOWNSTREAM), _flux(results, _UPSTREAM)
        normalized[f'{polarization} power balance'] = (abs((downstream - upstream) / _BEAM_POWER - 1.0)
                                                       / _POWER_TOLERANCE)
        normalized[f'{polarization} upstream'] = abs(upstream) / _BEAM_POWER / _UPSTREAM_TOLERANCE
        details[polarization] = {'downstream': downstream, 'upstream': upstream}
    return normalized_result(normalized, details)


def _check_focus(results_by_name):
    """Per polarisation: |u| at the focus vs the continuum beam of the same power, waist and aperture
    (focused_beam.continuum_focus_amplitude), and the focus cell above its four neighbours (peak within dx / 2)."""
    from .focused_beam import continuum_focus_amplitude
    omega = 2.0 * math.pi * float(C0) / _BEAM_WAVELENGTH
    normalized, details = {}, {}
    for polarization in _POLARIZATIONS:
        results = results_by_name[polarization]
        values = [abs(probe_phasor(results, _PHYSICS_NAME, _probe_name(index)))
                  for index in range(len(_PROBE_OFFSETS))]
        flux_material = float(MU0) if polarization == 'ez' else float(EPS0) * _WATER_EPS_R
        expected = continuum_focus_amplitude(omega, math.sqrt(_WATER_EPS_R), _BEAM_WAIST,
                                             _FOCUS_APERTURES[polarization], flux_material, _BEAM_POWER)
        normalized[f'{polarization} focal amplitude'] = abs(values[0] / expected - 1.0) / _FOCUS_TOLERANCE
        normalized[f'{polarization} peak at the focus'] = 0.0 if values[0] > max(values[1:]) else math.inf
        details[polarization] = {'focus': values[0], 'neighbours': values[1:], 'continuum': expected,
                                 'unit': 'V/m' if polarization == 'ez' else 'A/m'}
    return normalized_result(normalized, details)


def _scripted_beam(polarization, focus):
    """(FDFDPhysics, FocusedBeamPhasor) of the moved-case layout (module constants) on a cpu Domain, solved once."""
    from ...core.domain import Domain
    from .boundaries import BoundaryConfig
    from .solvers.fdfd_2d import FDFDPhysics
    config = BoundaryConfig.all_pml(_MOVED_PML)
    size_x, size_y = (count * _MOVED_CELL for count in _MOVED_CELLS)
    domain = Domain.uniform(size_x, size_y, _MOVED_CELL, boundary_config=config, verbose=False, backend='cpu')
    physics = FDFDPhysics(domain, polarization, boundary_config=config, wavelength=_BEAM_WAVELENGTH)
    for name in physics.permittivity_arrays:
        getattr(domain.materials, name)[...] = _WATER_EPS_R
    physics.materials_changed(physics.permittivity_arrays)
    beam = physics.add_focused_beam('x', cell_centre(_MOVED_LINE_ROW, _MOVED_CELL), (0.0, size_x), _BEAM_POWER,
                                    _BEAM_WAIST, focus)
    physics.finalize()
    physics.step()
    return physics, beam


def _incident_residual(physics, beam):
    """max |A u_inc| over the window rows from the upstream row to the second-to-last (every neighbour filled, none in
    the PML), the span cells but its two ends, of the incident wave continued over all those rows; relative to max
    |c u_inc| (c = the cell term k0^2 eps or k0^2 mu): the discrete waves solve the uniform rows exactly."""
    import numpy as np
    system = physics.system
    line = beam.line
    normal_window = physics.domain.physical_window()[1]   # the line runs along x
    rows = np.arange(line.line_cell - 2, normal_window.stop)
    values = beam.incident_rows(physics, system, rows)   # (rows, span cells)
    full = np.zeros(line.shape, dtype=np.complex128)
    full[line.span[0]:line.span[1], rows[0]:rows[-1] + 1] = values.T
    residual = (system.operator @ full.ravel()).reshape(line.shape)
    inner = residual[line.span[0] + 1:line.span[1] - 1, rows[1]:rows[-1]]
    k0_squared = (system.omega / float(C0)) ** 2
    cell_material = _WATER_EPS_R if system.polarization == 'ez' else 1.0
    return float(np.max(np.abs(inner))) / (k0_squared * cell_material * float(np.max(np.abs(values))))


def _check_moved(_results_by_name):
    """Per polarisation: solve_moved_beam(beam, offset) of a solved physics vs the solution of a physics built with the
    focus moved (relative max), its drive_fields vs the rebuilt one's, the incident residual, and the bookkeeping (the
    shown solution, n_solves untouched, extra_solves 1, offset 0 = the solution itself: 0 or inf)."""
    import numpy as np
    errors, details = {}, {}
    focus = tuple(cell_centre(index, _MOVED_CELL) for index in _MOVED_FOCUS_CELL)
    moved_focus = (focus[0] + _MOVED_OFFSET[0], focus[1] + _MOVED_OFFSET[1])
    for polarization in _POLARIZATIONS:
        reference, reference_beam = _scripted_beam(polarization, moved_focus)
        physics, beam = _scripted_beam(polarization, focus)
        shown, solves = physics.solution.copy(), physics.n_solves
        moved = physics.solve_moved_beam(beam, _MOVED_OFFSET)
        errors[f'{polarization} moved vs rebuilt'] = max_relative_error(moved, reference.solution)
        moved_fields = physics.drive_fields(moved)
        reference_fields = reference.drive_fields(reference.solution)
        for name, values in reference_fields.fields.items():
            errors[f'{polarization} drive field {name}'] = max_relative_error(moved_fields.fields[name], values)
        errors[f'{polarization} incident residual'] = _incident_residual(reference, reference_beam)
        kept = (np.array_equal(physics.solution, shown) and physics.n_solves == solves
                and physics.work_counters().get('extra_solves') == 1
                and physics.solve_moved_beam(beam, (0.0, 0.0)) is physics.solution)
        errors[f'{polarization} bookkeeping'] = 0.0 if kept else math.inf
        details[polarization] = {'incident_power': beam.incident_power, 'wavelength': beam.wavelength,
                                 'extra_solves': physics.work_counters().get('extra_solves')}
    return worst_result(errors, _MOVED_TOLERANCE, **details)


_LAYOUT_TEXT = (f'water (eps_r {_WATER_EPS_R:g}), vacuum wavelength {_BEAM_WAVELENGTH * 1e9:g} nm, uniform '
                f'{_BEAM_CELL * 1e9:g} nm mesh ({_BEAM_WAVELENGTH / math.sqrt(_WATER_EPS_R) / _BEAM_CELL:.3g} cells '
                f'per wavelength in water), a {_BEAM_CELLS[0]} x {_BEAM_CELLS[1]} cell window with {_BEAM_PML}-cell '
                f'CPML; a focused_beam over the window width (+y, waist {_BEAM_WAIST * 1e6:g} um) focused '
                f'{_FOCUS_CELL[1] - _LINE_ROW} cells downstream of its line on a cell centre; cpu backend, stationary')

FOCUSED_BEAM_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_focused_beam_power',
        label='EM FDFD focused beam: injected power and one-sided launch',
        description=f'em_fdfd_ez and em_fdfd_hz, {_LAYOUT_TEXT}; flux monitors over the window width downstream (beyond '
                    'the focus) and upstream. The beam power is the discrete Yee flux of the injected incident '
                    'field through the interface face, and whatever the far PML reflects crosses both lines, so '
                    f'(downstream - upstream) / power must be 1 ({_POWER_TOLERANCE:g}; discrete flux conservation of '
                    'the uniform lossless medium, the line ends at ~3e-7 of the beam); the upstream flux is only that '
                    f'reflection (the launch is discrete-exact): |upstream| / power below {_UPSTREAM_TOLERANCE:g} '
                    '(first estimate: the CPML reflection of the beam spectrum). measured = error = the worst in '
                    'tolerance units, tolerance 1. Not yet run.',
        specs=lambda: cpu_specs(focused_beam_power_specs()), check=_check_power),
    SelfTestCase(
        name='em_fdfd_focused_beam_focus',
        label='EM FDFD focused beam: focal field vs the continuum angular spectrum',
        description=f'The runs of em_fdfd_focused_beam_power ({_LAYOUT_TEXT}) with field probes of the out-of-plane '
                    'field at the focus cell and its four neighbours; em_fdfd_hz with numerical_aperture '
                    f'{_FOCUS_APERTURES["hz"]:g} (the cut at 3.5e-6 of the Gaussian). The focus must be the largest '
                    'of the five (peak within dx / 2) and |u| there must equal the continuum beam of the same power, '
                    'waist and aperture, sqrt(P / P_unit) (1 / 2 pi) int A dk with P_unit = (1 / 2 pi) int A^2 '
                    'kappa / (2 omega m) dk (m = mu0 for E_z, eps0 eps for H_z), to '
                    f'{_FOCUS_TOLERANCE:g} relative: the waves meet in phase at the focus by construction, so only the '
                    'discrete flux (E at the centre, H on the face: sin(kappa h) / h, ~(k h)^2 / 12 = 5e-3 in |u|) '
                    'and the discrete propagating cut remain. Not yet run.',
        specs=lambda: cpu_specs(_focus_specs()), check=_check_focus),
    SelfTestCase(
        name='em_fdfd_focused_beam_moved',
        label='EM FDFD focused beam: moved focus vs a rebuilt one',
        description=f'Scripted FDFDPhysics on cpu Domains (water, {_MOVED_CELL * 1e9:g} nm cells, window '
                    f'{_MOVED_CELLS[0]} x {_MOVED_CELLS[1]} cells, {_MOVED_PML}-cell CPML, Automatic = SuperLU), ez '
                    'and hz: solve_moved_beam(beam, offset) of a solved physics with offset '
                    f'({_MOVED_OFFSET[0] * 1e6:g}, {_MOVED_OFFSET[1] * 1e6:g}) um (non-integer cells) must equal '
                    'the solution of a physics built with the focus moved, and so must its drive_fields (hz: the face '
                    f'means of a given solution); relative max {_MOVED_TOLERANCE:g} (the same operator and right-hand '
                    'side: round-off only). Also the discrete residual max |A u_inc| / max |c u_inc| of the incident '
                    'wave continued over the window rows (interior cells, c the cell term) must stay below it: the '
                    'waves solve the uniform rows exactly, so the launch is one-sided to round-off. Bookkeeping: the '
                    'shown solution and n_solves untouched, extra_solves 1, offset 0 returns the solution itself. Not '
                    'yet run.',
        specs=no_runs, check=_check_moved),
)

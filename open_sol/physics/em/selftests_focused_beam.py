"""Self-tests of the focused_beam source: em_fdfd_ez / em_fdfd_hz power balance and one-sided launch, the focal field
against the continuum angular-spectrum integral, and solve_moved_beam against a rebuilt focus together with the
discrete residual of the incident wave; em_fdfd_bor's Debye beam: its power with the standing-wave ring weight and its
leakage, the focal field against the direct Debye integral, and solve_moved_beam / drive_fields against a rebuilt focus
(model runs on the cpu; the moved cases scripted); physics and materials imports inside the functions. Exports the
focused_beam FeatureSpec builder of the Brillouin self-tests and the power specs of the cpu-gpu parity cases."""
import math

from ...core.constants import C0, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (cell_centre, horizontal_line, max_relative_error, no_runs, normalized_result,
                                       probe_phasor, raises, scaled_worst_result, single_value, worst_result)
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_fdfd import _PHYSICS_NAME, cpu_specs

_POLARIZATIONS = ('ez', 'hz')
_BEAM = 'beam'
BEAM_LINE = 'beam_line'            # geometry name of the focused_beam line
_DOWNSTREAM, _UPSTREAM = 'downstream', 'upstream'
_BEAM_WAVELENGTH = 532e-9          # m, vacuum
_BEAM_CELL = 15e-9                 # m
_BEAM_CELLS = (290, 160)           # window cells (x, y); 330 x 200 = 66,000 unknowns with the CPML
_BEAM_PML = 20                     # cells per CPML face
_LINE_ROW = 26                     # window row of the beam line (y = 0.3975 um; the TFSF interface is its lo face)
_FOCUS_CELL = (145, 93)            # window cell whose centre is the focus: (2.1825, 1.4025) um, 1.005 um downstream
_DOWNSTREAM_ROW, _UPSTREAM_ROW = 133, 12   # flux rows: 2.01 um (26 cells from the PML); upstream in the scattered field
_BEAM_WAIST = 0.5e-6               # m: ~0.56 um wide at the line, ~3e-7 of the peak at its ends (2.18 um away)
_BEAM_POWER = 1.0                  # W/m
_FOCUS_APERTURES = {'ez': None, 'hz': 1.2}   # hz also exercises the numerical-aperture cut (Gaussian 3.5e-6 there)
_PROBE_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))   # cells from the focus cell; the first is the focus
_POWER_TOLERANCE = 1e-6            # relative, downstream - upstream flux vs power (discrete-exact flux)
_UPSTREAM_TOLERANCE = 1e-6         # of power: the upstream flux = PML reflection only (first estimate)
_FOCUS_TOLERANCE = 1e-2            # relative |u| at the focus vs the continuum (discrete flux ~ (k h)^2 / 12 in |u|)

# ---- em_fdfd_focused_beam_moved (scripted, cpu) ----
_MOVED_CELL = 20e-9
_MOVED_CELLS = (160, 110)          # 190 x 140 = 26,600 unknowns with the CPML
_MOVED_PML = 15
_MOVED_LINE_ROW = 15
_MOVED_FOCUS_CELL = (80, 60)
_MOVED_OFFSET = (0.113e-6, -0.071e-6)   # m (5.65 and -3.55 cells)
_MOVED_TOLERANCE = 1e-10           # relative: identical operators and right-hand sides on SuperLU (round-off)

# ---- em_fdfd_bor (bor_focused_beam) model runs ----
_BOR_CELL = 16e-9
_BOR_CELLS = (120, 120)            # window cells (r, z); 140 x 160 = 22,400 cells, 67,200 unknowns with the CPML
_BOR_PML = 20
_BOR_LINE_ROW = 20                 # z = 0.328 um (the TFSF interface is its lo face)
_BOR_FOCUS_ROW = 60                # z = 0.968 um, 0.64 um downstream
_BOR_DOWNSTREAM_ROW, _BOR_UPSTREAM_ROW = 100, 8   # flux rows: z = 1.616 um (19 rows below the PML); scattered field
_BOR_POWER = 1.0                   # W of the linearly polarised beam (the m = 1 standing wave)
_BOR_POWER_BEAM = {'numerical_aperture': 0.8, 'fill_factor': 0.5}   # pupil edge at exp(-4) of its centre
# high NA: a strong E_z; pupil edge exp(-4): the hard edge's side-lobe tail cut at the line end biases the focal E_z
# linearly in its end level (fill 0.6, edge exp(-2.8): E_z / E_r 1.2 % low at 16 and 8 nm alike, 0.7 % with 1.5x the
# radius)
_BOR_DEBYE_BEAM = {'numerical_aperture': 0.9, 'fill_factor': 0.5}
_BOR_EZ_PROBE_CELL = 10            # r cell of the focal-row E_z probe (k rho sin(theta_max) ~ 1.8: near the J1 peak)
_BOR_POWER_TOLERANCE = 2e-3        # relative standing-wave power vs power (first estimate: the radial O((k NA h)^2))
_BOR_LEAKAGE_TOLERANCE = 2e-3      # upstream flux / launched flux (first estimate, the same origin)
_BOR_DEBYE_TOLERANCE = 1e-2        # relative focal amplitudes and component ratios vs the direct Debye integral
_BOR_PROBES = (('er_axis', 'er', 0), ('ephi_axis', 'ephi', 0), ('ez_ring', 'ez', _BOR_EZ_PROBE_CELL))   # focal row

# ---- em_bor_focused_beam_moved (scripted, cpu; the beam of _BOR_POWER_BEAM) ----
_BOR_MOVED_CELL = 20e-9
_BOR_MOVED_CELLS = (60, 80)        # 72 x 104 = 7,488 cells, 22,464 unknowns with the CPML
_BOR_MOVED_PML = 12
_BOR_MOVED_LINE_ROW = 12
_BOR_MOVED_FOCUS_ROW = 45
_BOR_MOVED_OFFSET = (0.0, 0.0731e-6)   # m (dr, dz): 3.655 cells along z
_BOR_AXIS_TOLERANCE = 1e-12        # drive_fields A_phi (axis cell) / A_r (axis edge) + 1: the recovered axis E_r


# ---- specs ----
def _water_eps_r():
    """eps_r of library water_532nm (the specs hold an inline material of this eps_r only)."""
    from ...materials import WATER_532NM
    return WATER_532NM.optical['eps_r']


def focused_beam_feature(power, direction='positive', **params):
    """FeatureSpec 'beam' of type focused_beam on the geometry BEAM_LINE: params power, phase 0, then `params` in the
    given order (waist / numerical_aperture / fill_factor, focus_x / focus_y or focus_z) and direction."""
    return FeatureSpec(_BEAM, 'focused_beam', [BEAM_LINE], {'power': power, 'phase': 0.0, **params,
                                                            'direction': direction})


def _probe_name(index):
    return f'probe_{index}'


def _water_beam_spec(title, physics_type, settings, cell, cells, line_row, beam, monitor_rows, probes):
    """A stationary model on a uniform mesh of `cells` x `cell` [m] filled with water: `beam` (focused_beam_feature) on
    a line over the window width / radius at the centre of row `line_row`, a flux monitor per (name, row) over the same
    span at the row centre and a field probe per (feature, point, component, x, y)."""
    size_x, size_y = (count * cell for count in cells)
    geometry = [horizontal_line(BEAM_LINE, cell_centre(line_row, cell), 0.0, size_x)]
    features = [beam]
    for name, row in monitor_rows:
        geometry.append(horizontal_line(f'{name}_line', cell_centre(row, cell), 0.0, size_x))
        features.append(FeatureSpec(name, 'flux_monitor', [f'{name}_line']))
    for name, point, component, x, y in probes:
        geometry.append(GeometrySpec(point, 'point', {'x': x, 'y': y}))
        features.append(FeatureSpec(name, 'field_probe', [point], {'component': component}))
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('immersion_water', properties={'eps_r': _water_eps_r()}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='stationary'))


def _beam_spec(polarization, aperture=None, probes=False):
    """The planar model-run layout (module constants) in em_fdfd_<polarization>: the focused beam, both flux monitors
    and (probes) field probes of the out-of-plane field at the focus cell and its four neighbours."""
    focus = tuple(cell_centre(index, _BEAM_CELL) for index in _FOCUS_CELL)
    beam = focused_beam_feature(_BEAM_POWER, waist=_BEAM_WAIST, numerical_aperture=aperture, focus_x=focus[0],
                                focus_y=focus[1])
    points = [(_probe_name(index), f'probe_point_{index}', polarization,
               cell_centre(_FOCUS_CELL[0] + offset_i, _BEAM_CELL), cell_centre(_FOCUS_CELL[1] + offset_j, _BEAM_CELL))
              for index, (offset_i, offset_j) in enumerate(_PROBE_OFFSETS)] if probes else []
    physics_type = f'em_fdfd_{polarization}'
    return _water_beam_spec(f'Self-test: {physics_type} focused beam in water', physics_type,
                            {'wavelength': _BEAM_WAVELENGTH, 'pml_cells': _BEAM_PML}, _BEAM_CELL, _BEAM_CELLS,
                            _LINE_ROW, beam, ((_DOWNSTREAM, _DOWNSTREAM_ROW), (_UPSTREAM, _UPSTREAM_ROW)), points)


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
        flux_material = float(MU0) if polarization == 'ez' else float(EPS0) * _water_eps_r()
        expected = continuum_focus_amplitude(omega, math.sqrt(_water_eps_r()), _BEAM_WAIST,
                                             _FOCUS_APERTURES[polarization], flux_material, _BEAM_POWER)
        normalized[f'{polarization} focal amplitude'] = abs(values[0] / expected - 1.0) / _FOCUS_TOLERANCE
        normalized[f'{polarization} peak at the focus'] = 0.0 if values[0] > max(values[1:]) else math.inf
        details[polarization] = {'focus': values[0], 'neighbours': values[1:], 'continuum': expected,
                                 'unit': 'V/m' if polarization == 'ez' else 'A/m'}
    return normalized_result(normalized, details)


def _solved_beam(make_physics, config, cells, cell, add_beam):
    """(physics, beam) on a cpu Domain of `cells` x `cell` [m] filled with water: make_physics(domain) builds the
    physics, add_beam(physics, size) adds the focused beam (size: the window extents [m]); finalized, solved once."""
    from ...core.domain import Domain
    size = tuple(count * cell for count in cells)
    domain = Domain.uniform(size[0], size[1], cell, boundary_config=config, verbose=False, backend='cpu')
    physics = make_physics(domain)
    eps_r = _water_eps_r()
    for name in physics.permittivity_arrays:
        getattr(domain.materials, name)[...] = eps_r
    physics.materials_changed(physics.permittivity_arrays)
    beam = add_beam(physics, size)
    physics.finalize()
    physics.step()
    return physics, beam


def _scripted_beam(polarization, focus):
    """(FDFDPhysics, FocusedBeamPhasor) of the moved-case layout (module constants), solved once."""
    from .boundaries import BoundaryConfig
    from .solvers.fdfd_2d import FDFDPhysics
    config = BoundaryConfig.all_pml(_MOVED_PML)
    return _solved_beam(
        lambda domain: FDFDPhysics(domain, polarization, boundary_config=config, wavelength=_BEAM_WAVELENGTH),
        config, _MOVED_CELLS, _MOVED_CELL,
        lambda physics, size: physics.add_focused_beam('x', cell_centre(_MOVED_LINE_ROW, _MOVED_CELL), (0.0, size[0]),
                                                       _BEAM_POWER, _BEAM_WAIST, focus))


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
    cell_material = _water_eps_r() if system.polarization == 'ez' else 1.0
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


# ---- em_fdfd_bor ----
def _bor_beam_spec(beam, monitors=True, probes=()):
    """The BOR model-run layout (module constants): em_fdfd_bor at azimuthal_order 1 with a focused_beam of `beam`
    (numerical_aperture, fill_factor), (monitors) both flux monitors and the field probes (name, component, r cell) in
    the focal row."""
    focal_z = cell_centre(_BOR_FOCUS_ROW, _BOR_CELL)
    feature = focused_beam_feature(_BOR_POWER, **beam, focus_z=focal_z)
    points = [(name, f'{name}_point', component, cell_centre(cell, _BOR_CELL), focal_z)
              for name, component, cell in probes]
    rows = ((_DOWNSTREAM, _BOR_DOWNSTREAM_ROW), (_UPSTREAM, _BOR_UPSTREAM_ROW)) if monitors else ()
    return _water_beam_spec('Self-test: em_fdfd_bor focused beam in water', 'em_fdfd_bor', {
        'wavelength': _BEAM_WAVELENGTH, 'azimuthal_order': 1, 'pml_cells': _BOR_PML, 'bc_x_hi': 'cpml',
        'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml'}, _BOR_CELL, _BOR_CELLS, _BOR_LINE_ROW, feature, rows, points)


def bor_focused_beam_power_specs():
    """{'bor': spec} of em_bor_focused_beam_power (no backend set: em_bor_cpu_gpu reuses it per backend)."""
    return {'bor': _bor_beam_spec(_BOR_POWER_BEAM)}


def _check_bor_power(results_by_name):
    """Standing-wave power = (downstream - upstream flux) x ring_angle(1) / (2 pi) vs power (flux_monitor counts the
    travelling exp(-j phi) wave with 2 pi, the linearly polarised beam rings carry pi) and
    the upstream leakage |upstream| / (downstream - upstream), in tolerance units."""
    from ...core.metric import ring_angle
    results = results_by_name['bor']
    downstream, upstream = _flux(results, _DOWNSTREAM), _flux(results, _UPSTREAM)
    travelling = downstream - upstream
    standing = travelling * ring_angle(1) / (2.0 * math.pi)
    normalized = {'standing-wave power (ring weight pi of m = 1)': abs(standing / _BOR_POWER - 1.0)
                  / _BOR_POWER_TOLERANCE,
                  'upstream leakage': abs(upstream) / abs(travelling) / _BOR_LEAKAGE_TOLERANCE}
    return normalized_result(normalized, {'power': _BOR_POWER, 'unit': 'W', 'downstream': downstream,
                                          'upstream': upstream, 'standing_wave_power': standing,
                                          'travelling_over_power': travelling / _BOR_POWER})


def _debye_reference(rho, z_offset, numerical_aperture, fill_factor):
    """(A_r, A_phi, A_z) at (rho, z - z_f) [m] of the continuum Debye beam toward +z of _BOR_POWER W, phase 0, in water
    (Novotny-Hecht I00 / I01 / I02 in exp(+j omega t) by scipy quad, the power from the angular-spectrum flux; written
    out here, independent of bor_focused_beam)."""
    import scipy.integrate
    import scipy.special
    from ...core.constants import ETA0
    index = math.sqrt(_water_eps_r())
    wavenumber = 2.0 * math.pi * index / _BEAM_WAVELENGTH
    sin_max = numerical_aperture / index
    theta_max = math.asin(sin_max)

    def pupil(theta):
        return math.exp(-(math.sin(theta) / (fill_factor * sin_max)) ** 2)

    def integral(order, factor):
        def part(theta, real):
            value = (pupil(theta) * math.sqrt(math.cos(theta)) * math.sin(theta) * factor(theta)
                     * scipy.special.jv(order, wavenumber * rho * math.sin(theta))
                     * complex(math.cos(wavenumber * z_offset * math.cos(theta)),
                               -math.sin(wavenumber * z_offset * math.cos(theta))))
            return value.real if real else value.imag
        return complex(*(scipy.integrate.quad(part, 0.0, theta_max, args=(real,), limit=200, epsabs=0.0,
                                              epsrel=1e-12)[0] for real in (True, False)))
    i00 = integral(0, lambda theta: 1.0 + math.cos(theta))
    i01 = integral(1, math.sin)
    i02 = integral(2, lambda theta: 1.0 - math.cos(theta))
    flux = scipy.integrate.quad(lambda theta: pupil(theta) ** 2 * math.sin(theta) * math.cos(theta), 0.0, theta_max,
                                epsabs=0.0, epsrel=1e-12)[0]
    unit_power = 4.0 * math.pi * flux / ((float(ETA0) / index) * wavenumber ** 2)
    scale = math.sqrt(_BOR_POWER / unit_power)
    return scale * (i00 + i02), scale * (i02 - i00), 2j * scale * i01


def _check_bor_debye(results_by_name):
    """Focal-row probes vs the direct Debye integral sampled as the centre phasors average the Yee points: |E_r| of the
    axis cell (the normalisation), E_phi / E_r of the axis cell (the sin(phi) component: E_phi = j u = j A_phi) and E_z
    / E_r (E_z of cell _BOR_EZ_PROBE_CELL: the +2 j I01 sign), relative, in tolerance units."""
    results = results_by_name['bor']
    na, fill = _BOR_DEBYE_BEAM['numerical_aperture'], _BOR_DEBYE_BEAM['fill_factor']
    h = _BOR_CELL
    probes = {name: probe_phasor(results, _PHYSICS_NAME, name) for name, _, _ in _BOR_PROBES}
    axis_r = 0.5 * (_debye_reference(0.0, 0.0, na, fill)[0] + _debye_reference(h, 0.0, na, fill)[0])
    axis_phi = 1j * _debye_reference(0.5 * h, 0.0, na, fill)[1]
    ring_rho = cell_centre(_BOR_EZ_PROBE_CELL, h)
    ring_z = 0.5 * sum(_debye_reference(ring_rho, offset, na, fill)[2] for offset in (-0.5 * h, 0.5 * h))
    normalized = {
        '|E_r| on the axis': abs(abs(probes['er_axis']) / abs(axis_r) - 1.0) / _BOR_DEBYE_TOLERANCE,
        'E_phi / E_r on the axis': abs((probes['ephi_axis'] / probes['er_axis']) / (axis_phi / axis_r) - 1.0)
        / _BOR_DEBYE_TOLERANCE,
        'E_z / E_r (off the axis / on it)': abs((probes['ez_ring'] / probes['er_axis']) / (ring_z / axis_r) - 1.0)
        / _BOR_DEBYE_TOLERANCE}
    return normalized_result(normalized, {'probes': {name: [value.real, value.imag] for name, value in probes.items()},
                                          'reference': {'er_axis': [axis_r.real, axis_r.imag],
                                                        'ephi_axis': [axis_phi.real, axis_phi.imag],
                                                        'ez_ring': [ring_z.real, ring_z.imag]}, 'unit': 'V/m'})


def _scripted_bor_beam(focus_z):
    """(BORPhysics, BORFocusedBeam) of the BOR moved-case layout (module constants), solved once."""
    from .boundaries import BoundaryConfig
    from .solvers.fdfd_bor import BORPhysics
    config = BoundaryConfig(x_lo='pmc', x_hi='cpml', y_lo='cpml', y_hi='cpml')
    config.cpml_npml = _BOR_MOVED_PML
    return _solved_beam(
        lambda domain: BORPhysics(domain, boundary_config=config, wavelength=_BEAM_WAVELENGTH, azimuthal_order=1),
        config, _BOR_MOVED_CELLS, _BOR_MOVED_CELL,
        lambda physics, size: physics.add_focused_beam(
            'x', cell_centre(_BOR_MOVED_LINE_ROW, _BOR_MOVED_CELL), (0.0, size[0]), _BOR_POWER,
            _BOR_POWER_BEAM['numerical_aperture'], (focus_z,), fill_factor=_BOR_POWER_BEAM['fill_factor']))


def _check_bor_moved(_results_by_name):
    """solve_moved_beam(beam, (0, dz)) of a solved BORPhysics vs the solution of one built with the focus moved, its
    drive_fields vs the rebuilt one's (relative max, round-off), drive_fields' A_phi of the focal axis cell / A_r on
    the axis edge = -1 (the A_phi = -j E_phi convention brillouin.azimuthal_parts reads: the axis E_r is recovered as j
    E_phi(r_0), so A_r(0) = -A_phi(r_0) to round-off), and the bookkeeping
    (shown solution and n_solves untouched, extra_solves 1, offset 0 = the solution itself, a radial offset and an
    order-sum term refused: 0 or inf)."""
    import numpy as np
    focus = cell_centre(_BOR_MOVED_FOCUS_ROW, _BOR_MOVED_CELL)
    reference, _ = _scripted_bor_beam(focus + _BOR_MOVED_OFFSET[1])
    physics, beam = _scripted_bor_beam(focus)
    shown, solves = physics.solution.copy(), physics.n_solves
    moved = physics.solve_moved_beam(beam, _BOR_MOVED_OFFSET)
    errors = {'moved vs rebuilt': max_relative_error(moved, reference.solution)}
    tolerances = {'moved vs rebuilt': _MOVED_TOLERANCE}
    moved_fields = physics.drive_fields(moved)
    reference_fields = reference.drive_fields(reference.solution)
    for name, values in reference_fields.fields.items():
        errors[f'drive field {name}'] = max_relative_error(moved_fields.fields[name], values)
        tolerances[f'drive field {name}'] = _MOVED_TOLERANCE
    shown_fields = physics.drive_fields(physics.solution)
    focal_row = _BOR_MOVED_FOCUS_ROW   # window row = drive-field row (the window starts at z = 0)
    ratio = complex(shown_fields.fields['phi'][0, focal_row] / shown_fields.fields['r'][0, focal_row])
    errors['axis A_phi / A_r + 1'] = abs(ratio + 1.0)
    tolerances['axis A_phi / A_r + 1'] = _BOR_AXIS_TOLERANCE
    system = physics.system
    kept = (np.array_equal(physics.solution, shown) and physics.n_solves == solves
            and physics.work_counters().get('extra_solves') == 1
            and physics.solve_moved_beam(beam, (0.0, 0.0)) is physics.solution
            and raises(lambda: physics.solve_moved_beam(beam, (1e-9, 0.0)))
            and raises(lambda: beam.term_rhs(system, order=1)))
    errors['bookkeeping'] = 0.0 if kept else math.inf
    tolerances['bookkeeping'] = 1.0
    return scaled_worst_result(errors, tolerances, incident_power=beam.incident_power, wavelength=beam.wavelength,
                               focus=list(beam.focus), axis_ratio=[ratio.real, ratio.imag],
                               extra_solves=physics.work_counters().get('extra_solves'))


_LAYOUT_TEXT = (f'water (eps_r of library water_532nm, n 1.335), vacuum wavelength {_BEAM_WAVELENGTH * 1e9:g} nm, '
                f'uniform {_BEAM_CELL * 1e9:g} nm mesh (26.6 cells per wavelength in water), a {_BEAM_CELLS[0]} x '
                f'{_BEAM_CELLS[1]} cell window with {_BEAM_PML}-cell '
                f'CPML; a focused_beam over the window width (+y, waist {_BEAM_WAIST * 1e6:g} um) focused '
                f'{_FOCUS_CELL[1] - _LINE_ROW} cells downstream of its line on a cell centre; cpu backend, stationary')
_BOR_LAYOUT_TEXT = (f'water (eps_r of library water_532nm, n 1.335), vacuum wavelength {_BEAM_WAVELENGTH * 1e9:g} '
                    f'nm, uniform {_BOR_CELL * 1e9:g} nm mesh (24.9 cells per wavelength in water), a {_BOR_CELLS[0]} '
                    f'x {_BOR_CELLS[1]} cell (r, z) window with '
                    f'{_BOR_PML}-cell CPML on r_hi and both z faces; a focused_beam (+z) from a line over the window '
                    f'radius in row {_BOR_LINE_ROW}, focused on the axis {_BOR_FOCUS_ROW - _BOR_LINE_ROW} rows '
                    'downstream at a row centre; cpu backend, stationary')

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
                    f'values of a given solution); relative max {_MOVED_TOLERANCE:g} (the same operator and right-hand '
                    'side: round-off only). Also the discrete residual max |A u_inc| / max |c u_inc| of the incident '
                    'wave continued over the window rows (interior cells, c the cell term) must stay below it: the '
                    'waves solve the uniform rows exactly, so the launch is one-sided to round-off. Bookkeeping: the '
                    'shown solution and n_solves untouched, extra_solves 1, offset 0 returns the solution itself. Not '
                    'yet run.',
        specs=no_runs, check=_check_moved),
    SelfTestCase(
        name='em_bor_focused_beam_power',
        label='EM BOR focused beam: standing-wave power (ring weight pi) and one-sided launch',
        description=f'em_fdfd_bor at azimuthal_order 1, {_BOR_LAYOUT_TEXT}, numerical_aperture '
                    f'{_BOR_POWER_BEAM["numerical_aperture"]:g}, fill_factor {_BOR_POWER_BEAM["fill_factor"]:g}; flux '
                    'monitors from the axis over the window radius downstream (beyond the focus) and upstream. The '
                    'beam is the linearly polarised m = 1 standing wave (E_r, E_z ~ cos phi, E_phi ~ sin phi), whose '
                    'rings carry core.metric.ring_angle(1) = pi where flux_monitor weights the travelling exp(-j phi) '
                    'wave with 2 pi, so (downstream - upstream) x ring_angle(1) / (2 pi) must equal power '
                    f'({_BOR_POWER_TOLERANCE:g}): this pins the ring weight of the incident power that the '
                    'brillouin_scattering normalisation divides by. The upstream flux is the '
                    'leakage of the launch plus the CPML reflection: |upstream| / (downstream - upstream) below '
                    f'{_BOR_LEAKAGE_TOLERANCE:g} (the z dependence is Yee-exact, the radial Bessel profiles are '
                    'continuum: O((k NA h)^2)). Tolerances first estimates; measured = error = the worst in tolerance '
                    'units, tolerance 1. Not yet run.',
        specs=lambda: cpu_specs(bor_focused_beam_power_specs()), check=_check_bor_power),
    SelfTestCase(
        name='em_bor_focused_beam_debye',
        label='EM BOR focused beam: focal field vs the direct Debye integral',
        description=f'em_fdfd_bor at azimuthal_order 1, {_BOR_LAYOUT_TEXT}, numerical_aperture '
                    f'{_BOR_DEBYE_BEAM["numerical_aperture"]:g}, fill_factor {_BOR_DEBYE_BEAM["fill_factor"]:g} (a '
                    'strong axial field; the pupil edge at exp(-4) keeps the side-lobe tail the line end cuts small: '
                    'it biases the focal E_z linearly in the end level, 1.2 % at fill_factor 0.6 on any mesh); field '
                    'probes in the focal row: E_r and E_phi of the axis cell, E_z of r '
                    f'cell {_BOR_EZ_PROBE_CELL}. Reference: the Richards-Wolf / Debye integrals I00, I01, I02 '
                    '(Novotny-Hecht, exp(+j omega t)) by scipy quad, normalised by the angular-spectrum power and '
                    'averaged as the centre phasors average the Yee points. Checks (relative, '
                    f'{_BOR_DEBYE_TOLERANCE:g}): |E_r| on the axis (the normalisation; the discrete interface flux '
                    'defines power, ~(k h)^2 / 8 off the continuum one), E_phi / E_r on the axis (E_phi = j A_phi, '
                    'A_phi -> -A_r) and E_z / E_r (the +2 j I01 sign of the axial field); the ratios cancel the '
                    'propagation phase. Not yet run.',
        specs=lambda: cpu_specs({'bor': _bor_beam_spec(_BOR_DEBYE_BEAM, monitors=False, probes=_BOR_PROBES)}),
        check=_check_bor_debye),
    SelfTestCase(
        name='em_bor_focused_beam_moved',
        label='EM BOR focused beam: moved focus and drive fields vs a rebuilt one',
        description=f'Scripted BORPhysics on cpu Domains (water, {_BOR_MOVED_CELL * 1e9:g} nm cells, window '
                    f'{_BOR_MOVED_CELLS[0]} x {_BOR_MOVED_CELLS[1]} cells, {_BOR_MOVED_PML}-cell CPML on r_hi and both '
                    'z faces, azimuthal_order 1, private SuperLU): solve_moved_beam(beam, (0, dz)) of a solved '
                    f'physics with dz = {_BOR_MOVED_OFFSET[1] * 1e6:g} um (non-integer cells) must equal the solution '
                    'of a physics built with the focus moved, and so must its drive_fields (relative max '
                    f'{_MOVED_TOLERANCE:g}: the same operator and right-hand side, round-off only). drive_fields '
                    'hands brillouin_scattering the standing-wave amplitudes {r: E_r on the r edges, phi: -j E_phi, z: '
                    'E_z on the z edges}; the axis E_r is recovered as j E_phi(r_0), so A_phi of the focal axis cell / '
                    f'A_r on the axis edge + 1 must stay below {_BOR_AXIS_TOLERANCE:g} (round-off: it pins the '
                    'convention and the edge layout). Bookkeeping: the shown solution '
                    'and n_solves untouched, extra_solves 1, offset 0 returns the solution itself, a radial offset and '
                    'a driven-order-sum term raise. Not yet run.',
        specs=no_runs, check=_check_bor_moved),
)

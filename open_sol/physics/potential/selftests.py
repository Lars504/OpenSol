"""Analytic self-tests of electric_currents / electrostatics (divider with terminals and two-dielectric parallel plates
as one twin model on a graded mesh; the uniform divider fixture reused by heat's diffusion_cpu_gpu) and the
'thermo_optic_heater' template; specs built fresh."""
import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (EXACT_CG_TOLERANCE, cell_centre, cell_edges, max_relative_error, relative_error,
                                       with_exact_cg, worst_error)
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec, SolverDefaultsSpec,
                           StudySpec)

CURRENTS_TYPE_NAME = 'electric_currents'       # = definition.CURRENTS_TYPE_NAME (definition imports this module)
ELECTROSTATICS_TYPE_NAME = 'electrostatics'
_CURRENTS = 'currents'                         # physics instance names = the types' default instances
_ELECTROSTATICS = 'electrostatics'
_RUN = 'run'

# Layout shared by every case: a 1 x 0.1 mm strip of 10 um cells (100 x 10 window cells; no EM physics, so no
# PML), insulated y faces (1D in x); window cell k spans [10 k, 10 (k + 1)] um. The held blocks are cells 0..9 (inner
# face x = 0.1 mm) and 90..99 (inner face 0.9 mm), ideal conductors (sigma_dc 0 / eps_dc 1 there: the held value
# sits on their faces toward the free cells whatever their own coefficient); the material interface x = 0.5 mm is a
# cell edge, so L_a = L_b = 0.4 mm from the inner held faces to it. The 3-point finite-volume scheme (harmonic-mean face
# conductances, half-cell Dirichlet faces, also toward held cells) is exact for piecewise-linear profiles: every closed
# form below is a discrete identity up to the CG residual (1e-12 times a condition number ~1e4) and the float32 storage
# of V, J / D and q_joule (~1e-7 relative); the held fluxes come from the float64 solution.
_STRIP_LENGTH = 1e-3               # m, x extent
_STRIP_WIDTH = 1e-4                # m, y extent (w of the closed forms)
_CELL = 1e-5                       # m, uniform mesh
_N_CELLS = (round(_STRIP_LENGTH / _CELL), round(_STRIP_WIDTH / _CELL))   # (100, 10)
_HELD_CELLS = 10                   # cells along x of each terminal block
_HELD_LENGTH = _HELD_CELLS * _CELL
_INTERFACE_X = 0.5e-3              # m, a cell edge
_INNER_FACE_B = _STRIP_LENGTH - _HELD_LENGTH                   # 0.9 mm, lo face of cell 90
_VOLTAGE = 1.0                     # V across the strip
_TOLERANCE = 1e-6

# Graded variant (potential_divider_plates_graded): mesh 'nonuniform' with a vertical refinement line on the interface,
# x cells 2.5 um there blending linearly back to the 10 um base 0.45 mm away (past both inner held faces, so held-free
# and interior faces join unequal widths; y stays 10 um). The rescaled cells put no edge on the nominal positions: the
# closed forms are then evaluated on the actual cells (edges rebuilt from the float64 centres, rectangles rasterised on
# float32 centres like the build), which keeps them discrete identities on any widths.
_GRADED_INTERFACE_CELL = 2.5e-6    # m
_GRADED_REACH = 0.45e-3            # m
_GRADED_MIN_RATIO = 2.0            # widest / narrowest window x cell a graded run must reach (no silent uniform fallback)

# Divider (currents) + plates (electrostatics) in one model: coefficient c_a on x < 0.5 mm, c_b beyond, terminals in
# the held blocks. Series half-cell resistances: flux per m of depth F = w V / sum over the free cells of dx_i / c_i
# (uniform: L_a / c_a + L_b / c_b; current with c = sigma_dc, charge with c = eps0 eps_dc), flux density F / w in every
# free cell (0 in the held blocks), V dropping by (F / w) dx_i / (2 c_i) per half cell from the inner held face.
_SIGMA_A, _SIGMA_B = 10.0, 30.0    # S/m
_EPS_A, _EPS_B = 2.0, 6.0          # eps_dc
_PROBE_CELL = (49, 5)              # window cell of the potential probe on the uniform mesh (the last cell of half_a)


def _divider_solution(edges, coefficient_a, coefficient_b):
    """Exact discrete divider on the cells `edges`: (flux per m of depth F, V at the centres, held mask, coefficient per
    cell); held / material cells by float32 centre in [x0, x1) as the build rasterises the rectangles."""
    centres = (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)
    held = (centres < np.float32(_HELD_LENGTH)) | (centres >= np.float32(_STRIP_LENGTH - _HELD_LENGTH))
    coefficient = np.where(centres < np.float32(_INTERFACE_X), coefficient_a, coefficient_b)
    free_resistance = np.where(held, 0.0, np.diff(edges) / coefficient)   # dx_i / c_i, 0 in the held blocks
    flux = _STRIP_WIDTH * _VOLTAGE / float(np.sum(free_resistance))
    drop_to_centre = np.cumsum(free_resistance) - 0.5 * free_resistance
    return flux, _VOLTAGE - flux / _STRIP_WIDTH * drop_to_centre, held, coefficient


def _containing_cell(edges, position):
    """Index of the cell of `edges` containing `position` (an edge coordinate goes to the upper cell)."""
    return int(np.clip(np.searchsorted(edges, position, side='right') - 1, 0, edges.size - 2))


# ---- specs ----
def _strip_model(title, physics, materials, geometry=(), derived_values=(), graded=False):
    """Stationary strip along x of the (name, type, features) physics, cpu backend, float64 CG to 1e-12, on the
    uniform _CELL mesh or graded toward the interface by the refinement line 'refine_line'."""
    geometry, mesh = list(geometry), MeshSpec(mode='uniform', dx=_CELL)
    if graded:
        title = f'{title}, graded mesh'
        geometry.append(GeometrySpec('refine_line', 'line',
                                     {'x0': _INTERFACE_X, 'y0': 0.0, 'x1': _INTERFACE_X, 'y1': _STRIP_WIDTH}))
        mesh = MeshSpec(mode='nonuniform', dx=_CELL,
                        refinements=[RefinementSpec('refine_line', _GRADED_INTERFACE_CELL, _GRADED_REACH)])
    return with_exact_cg(ModelSpec(
        title=title,
        domain=DomainSpec(size_x=_STRIP_LENGTH, size_y=_STRIP_WIDTH),
        geometry=geometry,
        materials=list(materials),
        physics=[PhysicsSpec(name, physics_type, features=list(features)) for name, physics_type, features in physics],
        mesh=mesh,
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=list(derived_values)))


def _full_height_rectangle(name, x0, x1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': 0.0, 'x1': x1, 'y1': _STRIP_WIDTH})


def _divider_geometry():
    """Terminal blocks 'left' / 'right' (cells 0..9 / 90..99) and the material halves 'half_a' / 'half_b'."""
    return [_full_height_rectangle('left', 0.0, _HELD_LENGTH),
            _full_height_rectangle('right', _STRIP_LENGTH - _HELD_LENGTH, _STRIP_LENGTH),
            _full_height_rectangle('half_a', 0.0, _INTERFACE_X),
            _full_height_rectangle('half_b', _INTERFACE_X, _STRIP_LENGTH)]


def _terminal_divider_specs(graded=False, with_plates=False):
    """'run': sigma 10 / 30 S/m halves between the non-conducting terminals left (1 V) and right (0 V), a probe in uniform
    cell (49, 5); with_plates adds the uncoupled electrostatics plates (eps_dc 2 / 6, eps_dc 1 terminal blocks)."""
    probe_x, probe_y = (cell_centre(index, _CELL) for index in _PROBE_CELL)
    title = 'Self-test: two-conductor divider with terminals (electric currents)'
    physics = [(_CURRENTS, CURRENTS_TYPE_NAME, [FeatureSpec('left', 'terminal', ['left'], {'voltage': _VOLTAGE}),
                                                FeatureSpec('right', 'terminal', ['right'], {'voltage': 0.0}),
                                                FeatureSpec('divider', 'potential_probe', ['probe_point'])])]
    derived = [DerivedValueSpec('P_joule', f'{_CURRENTS}_q_joule', ['all'], 'integral'),
               DerivedValueSpec('I_left', f'{_CURRENTS}_current_left', [], 'mean')]
    permittivity = {'a': {}, 'b': {}, 'held': {}}   # eps_dc per material (with_plates only)
    if with_plates:
        title = 'Self-test: divider and two-dielectric plates (electric currents + electrostatics)'
        physics.append((_ELECTROSTATICS, ELECTROSTATICS_TYPE_NAME,
                        [FeatureSpec('plate_hi', 'terminal', ['left'], {'voltage': _VOLTAGE}),
                         FeatureSpec('plate_lo', 'terminal', ['right'], {'voltage': 0.0})]))
        derived.append(DerivedValueSpec('Q_hi', f'{_ELECTROSTATICS}_charge_plate_hi', [], 'mean'))
        permittivity = {'a': {'eps_dc': _EPS_A}, 'b': {'eps_dc': _EPS_B}, 'held': {'eps_dc': 1.0}}
    return {_RUN: _strip_model(
        title, physics,
        [MaterialSpec('conductor_a', properties={'sigma_dc': _SIGMA_A, **permittivity['a']}, selection=['half_a']),
         MaterialSpec('conductor_b', properties={'sigma_dc': _SIGMA_B, **permittivity['b']}, selection=['half_b']),
         MaterialSpec('contact', properties={'sigma_dc': 0.0, **permittivity['held']}, selection=['left', 'right'])],
        geometry=[*_divider_geometry(), GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y})],
        derived_values=derived, graded=graded)}


# ---- checks (CPU, on Results-like objects) ----
def _window_field(results, field_name, graded=False):
    """(float64 window values indexed (x, y), float64 x cell edges) of a field; ValueError unless a uniform run has the
    100 x 10 cells and a graded one widths spanning at least _GRADED_MIN_RATIO."""
    field_data = results.field(field_name)
    values = np.asarray(field_data.values, dtype=np.float64)
    edges = cell_edges(np.asarray(field_data.x_centers, dtype=np.float64), float(field_data.extent[0]))
    widths = np.diff(edges)
    if not graded and values.shape != _N_CELLS:
        raise ValueError(f'{field_name}: expected the window shape {_N_CELLS}, got {values.shape}')
    if graded and not (values.shape[0] == widths.size and np.max(widths) >= _GRADED_MIN_RATIO * np.min(widths)):
        raise ValueError(f'{field_name}: expected graded x cells (widest / narrowest >= {_GRADED_MIN_RATIO:g}), got '
                         f'{widths.size} cells of {np.min(widths):.4g} .. {np.max(widths):.4g} m')
    return values, edges


def _held_flux(results, physics_name, feature, quantity):
    """The 'current' / 'charge' extra [A/m or C/m] of a voltage / terminal table result."""
    return float(results.result(physics_name, feature)[quantity])


def _derived_value(results, name):
    return float(results.summary()['derived'][name])   # float() also reads 'nan'


def merged_kind_check(results_by_name, checks):
    """Every kind's check {kind: check} on one twin run; the result of the kind with the largest error / tolerance (a
    NaN wins), with every kind's full result in the details."""
    outcomes = {kind: check(results_by_name) for kind, check in checks.items()}

    def scaled_error(kind):
        value = outcomes[kind]['error'] / outcomes[kind]['tolerance']
        return float('inf') if np.isnan(value) else value
    worst = max(outcomes, key=scaled_error)
    return {**outcomes[worst], 'details': {'worst_kind': worst, **outcomes}}


def _check_terminal_divider(results_by_name, graded=False):
    """Free-cell V profile, J_x (0 in the held blocks), q_joule on every cell, both terminal currents, the probe sample
    and P_joule of the two-conductor divider against the closed forms on the run's cells."""
    results = results_by_name[_RUN]
    potential, edges = _window_field(results, f'{_CURRENTS}.V', graded)
    current_density, _ = _window_field(results, f'{_CURRENTS}.J_x', graded)
    joule, _ = _window_field(results, f'{_CURRENTS}.q_joule', graded)
    current, expected_profile, held, sigma = _divider_solution(edges, _SIGMA_A, _SIGMA_B)
    free = ~held
    density_closed = current / _STRIP_WIDTH
    joule_closed = np.where(held, 0.0, density_closed ** 2 / sigma)   # a held-free face dissipates in its free half
    joule_scale = float(np.max(joule_closed))
    probe_cell = _containing_cell(edges, cell_centre(_PROBE_CELL[0], _CELL))
    probe_closed = float(expected_profile[probe_cell])
    current_left = _held_flux(results, _CURRENTS, 'left', 'current')
    current_right = _held_flux(results, _CURRENTS, 'right', 'current')
    probe_samples = np.asarray(results.result(_CURRENTS, 'divider')['values'], dtype=np.float64)
    probe_voltage = float(probe_samples[-1])
    power = _derived_value(results, 'P_joule')
    errors = {'profile': max_relative_error(potential[free], expected_profile[free][:, None], _VOLTAGE),
              'current_density': max_relative_error(current_density[free], density_closed, density_closed),
              'held_current_density': max_relative_error(current_density[held], 0.0, density_closed),
              'joule': max_relative_error(joule, joule_closed[:, None], joule_scale),
              'left': relative_error(current_left, current),
              'left_scalar_output': relative_error(_derived_value(results, 'I_left'), current),
              'right': relative_error(current_right, -current),
              'probe': relative_error(probe_voltage, probe_closed),
              'power': relative_error(power, _VOLTAGE * current)}
    return {'measured': probe_voltage, 'expected': probe_closed, 'error': worst_error(errors),
            'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_current': 'A/m', 'I_left': current_left, 'I_right': current_right,
                        'I_closed': current, 'J_closed_A_per_m2': density_closed,
                        'P_joule_W_per_m': power, 'P_closed_W_per_m': _VOLTAGE * current,
                        'q_joule_probe_cell_and_next_W_per_m3': [float(joule[probe_cell, 0]),
                                                                 float(joule[probe_cell + 1, 0])],
                        'probe_cell': probe_cell, 'probe_samples': int(probe_samples.size), 'V_probe': probe_voltage,
                        'n_cells': list(potential.shape)}}


def _check_parallel_plates(results_by_name, graded=False):
    """Free-cell V profile and D_x, D_x = 0 in the plates and both plate charges of the two-dielectric capacitor
    against the closed forms on the run's cells."""
    results = results_by_name[_RUN]
    potential, edges = _window_field(results, f'{_ELECTROSTATICS}.V', graded)
    displacement, _ = _window_field(results, f'{_ELECTROSTATICS}.D_x', graded)
    charge, expected_profile, held, _ = _divider_solution(edges, EPS0 * _EPS_A, EPS0 * _EPS_B)
    free = ~held
    charge_hi = _held_flux(results, _ELECTROSTATICS, 'plate_hi', 'charge')
    charge_lo = _held_flux(results, _ELECTROSTATICS, 'plate_lo', 'charge')
    displacement_closed = charge / _STRIP_WIDTH
    errors = {'profile': max_relative_error(potential[free], expected_profile[free][:, None], _VOLTAGE),
              'displacement': max_relative_error(displacement[free], displacement_closed, displacement_closed),
              'held_displacement': max_relative_error(displacement[held], 0.0, displacement_closed),
              'plate_hi': relative_error(charge_hi, charge),
              'plate_hi_scalar_output': relative_error(_derived_value(results, 'Q_hi'), charge),
              'plate_lo': relative_error(charge_lo, -charge)}
    return {'measured': charge_hi, 'expected': charge, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_charge': 'C/m', 'Q_hi': charge_hi, 'Q_lo': charge_lo,
                        'Q_closed': charge, 'capacitance_F_per_m': charge_hi / _VOLTAGE,
                        'D_closed_C_per_m2': displacement_closed, 'n_cells': list(potential.shape)}}


def _check_divider_plates(results_by_name, graded=False):
    """The divider and the parallel-plates checks on their one twin run."""
    return merged_kind_check(results_by_name, {
        'currents': lambda results: _check_terminal_divider(results, graded),
        'electrostatics': lambda results: _check_parallel_plates(results, graded)})


# ---- template ----
# thermo_optic_heater: uniform 40 nm mesh, 12 x 4.8 um domain (300 x 120 window cells; 330 x 150 = 49,500 em_fdfd_ez
# unknowns with the PML: scipy_splu far below its 1e6 warning, and a 453 MiB cusolver_qr workspace below its 512 MiB
# warning should scipy be missing), every geometry edge on a cell edge, the mode lines at cell centres (x = 0.62 /
# 11.38 um: cells 15 / 284). Silicon slab y = 1.0..1.48 um (12 cells; 11.1 cells per wavelength in silicon at 1.55 um,
# above the 10-cell warning), gold heater x = 1.6..10.4 um, y = 3.0..4.0 um (1.52 um above the slab) with its ends held
# by the terminals x = 1.6..2.0 um (V_heater) and 10.0..10.4 um (0 V): free channel cells 50..249, L = 8.0 um between
# the inner held faces, cold conductance sigma w / L = 4.1e7 * 1e-6 / 8.0e-6 = 5.1e6 S per m of depth, so 0.02 V
# drives I = 1.0e5 A/m and P = 2.0e3 W/m (q = 2.6e14 W/m^3 in the channel). Sink: out-of-plane convection h = 7e5
# W/(m^2 K) through one face of a 1 um slab = 7e11 W/(m^3 K) (silica through ~2 um of oxide), lateral decay sqrt(k /
# h_vol) = 1.4 um in silica: heater rise ~60-80 K (the temperature coefficient 0.0034 /K of gold cuts P by ~20 %),
# waveguide ~20 K, dn = dndT dT ~ 3.6e-3 in silicon, phase ~0.1 rad over the 8.8 um heater. The mode lines run from
# y = 0 (cpml window edge: the port solve continues through the PML) to 2.5 um (a metal end at 2.52 um, where the TE0
# tail is ~5e-6 of its peak; the heater starts at 3.0 um). Validation: no errors, infos only.
THERMO_OPTIC_HEATER_LABEL = 'Thermo-optic heater'
THERMO_OPTIC_HEATER_DESCRIPTION = (
    'Electric currents + heat + EM FDFD (E_z), stationary: a gold strip heater (1 x 8.8 um, V_heater = 0.02 V, '
    'sigma_dc(T)) 1.52 um beside a 0.48 um silicon slab waveguide in silica; Joule heating (~2 kW/m), out-of-plane '
    'sink (7e5 W/(m^2 K) through 1 um: heater rise ~60-80 K, waveguide ~20 K), thermo-optic phase of the guided mode '
    '(~0.1 rad, mode_monitor phase_positive); derived P_heater, T_max, T_waveguide.')
_HEATER_CELL = 40e-9                          # m
_HEATER_DOMAIN = (12e-6, 4.8e-6)              # m
_HEATER_WAVELENGTH = 1.55e-6                  # m
_HEATER_VOLTAGE = 0.02                        # V
_WAVEGUIDE_Y_RANGE = (1.0e-6, 1.48e-6)        # m
_HEATER_X_RANGE = (1.6e-6, 10.4e-6)           # m
_HEATER_Y_RANGE = (3.0e-6, 4.0e-6)            # m
_HEATER_TERMINAL_LENGTH = 0.4e-6              # m, held length at each heater end
_HEATER_LINE_X = {'source': 0.62e-6, 'through': 11.38e-6}   # m, mode lines at cell centres
_HEATER_LINE_TOP = 2.5e-6                     # m, y end of the mode lines (metal end at 2.52 um)
_HEATER_PROBE = (6.0e-6, 3.5e-6)              # m, the heater centre
_HEATER_SINK = {'h': 7e5, 'T_ext': 293.15, 'thickness': 1e-6, 'sides': 1}
_GOLD_SIGMA_DC_OF_T = '4.1e7 / (1 + 0.0034 * (heat_T - 293.15))'   # S/m: the library value at T0, 0.0034 /K


def thermo_optic_heater_spec():
    """Template 'thermo_optic_heater' (em_fdfd_ez + heat + electric_currents, stationary): a gold strip heater with
    sigma_dc(T) beside a silicon slab waveguide in silica; Joule, thermo-optic and resistive feedback iterate."""
    size_x, size_y = _HEATER_DOMAIN
    heater_x0, heater_x1 = _HEATER_X_RANGE
    heater_y0, heater_y1 = _HEATER_Y_RANGE
    waveguide_y0, waveguide_y1 = _WAVEGUIDE_Y_RANGE
    probe_x, probe_y = _HEATER_PROBE

    def rectangle(name, x0, y0, x1, y1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    def mode_line(name, x):
        return GeometrySpec(name, 'line', {'x0': x, 'y0': 0.0, 'x1': x, 'y1': _HEATER_LINE_TOP})

    return ModelSpec(
        title=THERMO_OPTIC_HEATER_LABEL,
        parameters=[ParameterSpec('V_heater', _HEATER_VOLTAGE, 'heater voltage [V]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[rectangle('waveguide', 0.0, waveguide_y0, size_x, waveguide_y1),
                  rectangle('heater', heater_x0, heater_y0, heater_x1, heater_y1),
                  rectangle('terminal_a', heater_x0, heater_y0, heater_x0 + _HEATER_TERMINAL_LENGTH, heater_y1),
                  rectangle('terminal_b', heater_x1 - _HEATER_TERMINAL_LENGTH, heater_y0, heater_x1, heater_y1),
                  mode_line('source_line', _HEATER_LINE_X['source']),
                  mode_line('monitor_line', _HEATER_LINE_X['through']),
                  GeometrySpec('heater_centre', 'point', {'x': probe_x, 'y': probe_y})],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('core', library='silicon', selection=['waveguide']),
                   MaterialSpec('gold_heater', library='gold', properties={'sigma_dc': _GOLD_SIGMA_DC_OF_T},
                                selection=['heater'])],
        physics=[PhysicsSpec('em', 'em_fdfd_ez', settings={'wavelength': _HEATER_WAVELENGTH}, features=[
                     FeatureSpec('source', 'mode_source', ['source_line'],
                                 {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}),
                     FeatureSpec('through', 'mode_monitor', ['monitor_line'], {'mode_number': 1})]),
                 PhysicsSpec('heat', 'heat', features=[
                     FeatureSpec('substrate', 'out_of_plane_convection', ['all'], dict(_HEATER_SINK)),
                     FeatureSpec('T_heater', 'temperature_probe', ['heater_centre'])]),
                 PhysicsSpec(_CURRENTS, CURRENTS_TYPE_NAME, features=[
                     FeatureSpec('terminal_a', 'terminal', ['terminal_a'], {'voltage': 'V_heater'}),
                     FeatureSpec('terminal_b', 'terminal', ['terminal_b'], {'voltage': 0.0})])],
        couplings=[CouplingSpec('joule', 'currents_heat_joule', _CURRENTS, 'heat'),
                   CouplingSpec('thermo_optic', 'heat_em_thermo_optic', 'heat', 'em')],
        mesh=MeshSpec(mode='uniform', dx=_HEATER_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('P_heater', f'{_CURRENTS}_q_joule', ['heater'], 'integral'),
                        DerivedValueSpec('T_max', 'T', [], 'max'),
                        DerivedValueSpec('T_waveguide', 'T', ['waveguide'], 'mean')])


# ---- cases ----
_LAYOUT_TEXT = (f'a {_STRIP_LENGTH * 1e3:g} x {_STRIP_WIDTH * 1e3:g} mm strip (base cell {_CELL * 1e6:g} um; no EM '
                'physics, so no PML) with insulated y faces (1D in x), solved in float64 to a CG tolerance of '
                f'{EXACT_CG_TOLERANCE:g} in a stationary study on the cpu backend (numpy, the exact pcg node)')
TWIN_TEXT = ('Two uncoupled physics, electric_currents and electrostatics, in one model on the same cells (the '
             'materials carry both sigma_dc and eps_dc, so each solve equals a separate run); each kind keeps its own '
             'check below and the case reports the kind with the larger error / tolerance (both results in the '
             'details).')
_BUDGET_TEXT = (f'tolerance {_TOLERANCE:g}: the 3-point finite-volume scheme (harmonic-mean face conductances, '
                'half-cell Dirichlet faces) is exact for piecewise-linear profiles on any widths, so only the CG '
                f'residual ({EXACT_CG_TOLERANCE:g} times a condition number ~1e4) and the float32 storage of the '
                'fields (~1e-7 relative; the held fluxes come from the float64 solution) remain')
_DIVIDER_TEXT = (f"rectangles 'left' (x < {_HELD_LENGTH * 1e3:g} mm) and 'right' (x > {_INNER_FACE_B * 1e3:g} mm) "
                 f"hold the terminals, 'half_a' (x < {_INTERFACE_X * 1e3:g} mm) and 'half_b' the two materials, a "
                 'later material on the terminal blocks makes them ideal conductors without a coefficient of their '
                 'own (the held voltage sits on their inner faces whatever it is)')
_GRADED_TEXT = (f"graded mesh: mode 'nonuniform' (base {_CELL * 1e6:g} um) with a vertical refinement line on the "
                f'interface x = {_INTERFACE_X * 1e3:g} mm (dx {_GRADED_INTERFACE_CELL * 1e6:g} um, margin '
                f'{_GRADED_REACH * 1e3:g} mm), so the x cells blend linearly from {_GRADED_INTERFACE_CELL * 1e6:g} um '
                f'there to {_CELL * 1e6:g} um beyond both inner held faces (y stays {_CELL * 1e6:g} um); the rescaled '
                'cells put no edge on the nominal interface or held faces, so the held blocks and halves are the '
                'cells whose float32 centres fall in the rectangles (as the build rasterises them), the edges come '
                'from the float64 centres, and the check raises unless the widest / narrowest x cell >= '
                f'{_GRADED_MIN_RATIO:g} (no silent uniform fallback)')

SELFTEST_CASES = (
    SelfTestCase(
        name='potential_divider_plates_graded',
        label='Electric currents + electrostatics: divider and two-dielectric plates on a graded mesh',
        description=(f'{TWIN_TEXT} Both on {_LAYOUT_TEXT}, on a {_GRADED_TEXT}; {_DIVIDER_TEXT}. The closed forms '
                     'are evaluated on the actual cells. (1) Electric currents: sigma_dc = '
                     f'{_SIGMA_A:g} S/m on half_a and {_SIGMA_B:g} S/m on half_b, material contact = sigma_dc 0 on '
                     f'the terminal blocks (they still anchor and carry the current), terminals left = {_VOLTAGE:g} '
                     'V and right = 0 V, a potential_probe divider at the centre of uniform cell '
                     f'{list(_PROBE_CELL)}, derived P_joule = the integral of q_joule over all and I_left = the scalar '
                     'output currents_current_left. Exact (series resistors): I = w V / sum over the free cells of '
                     'dx_i / sigma_i (left +I, right -I), V_i = 1 - (I / w) (sum of dx_j / sigma_j over the free '
                     'cells before i + dx_i / (2 sigma_i)), J_x = I / w and q_joule = J^2 / sigma in every free cell '
                     '(the interface face splits by half-cell resistance), 0 in the held blocks, the probe expected '
                     'at the centre of the cell containing the probe point, P_joule = V I. error = max(max |V - '
                     'V_exact| / V and max |J_x - J| / J over the free cells, max |J_x| / J over the held blocks, '
                     'max |q_joule - q| / max q, |I_left - I| / I, |I_right + I| / I, the probe, P_joule, the '
                     f'scalar output); {_BUDGET_TEXT}. This pins the non-uniform face conductances (held-free half '
                     'cells, unequal-width harmonic means), non-conducting terminals (sum of A V - b over the held '
                     'cells), the resistance-weighted q_joule split, the cell-area-weighted integral and the probe '
                     f'lookup on graded cells. (2) Electrostatics: eps_dc = {_EPS_A:g} on half_a and {_EPS_B:g} on '
                     'half_b, eps_dc 1 on the terminal blocks, terminals plate_hi = 1 V and plate_lo = 0 V, derived '
                     'Q_hi = the scalar output electrostatics_charge_plate_hi. Q = eps0 w V / sum over the free '
                     'cells of dx_i / eps_i, D_x = Q / w in every free cell and 0 in the plates, V as in (1) with '
                     'eps0 eps in place of sigma; error = max(|Q_hi - Q| / Q, |Q_lo + Q| / Q, D_x, V, the scalar '
                     'output), same tolerance. This pins the eps0-scaled operator, the terminal charge read-out and '
                     'that an eps_dc 1 electrode adds no vacuum layer on graded cells. Not yet run as one twin model '
                     'on the cpu backend.'),
        specs=lambda: _terminal_divider_specs(graded=True, with_plates=True),
        check=lambda results_by_name: _check_divider_plates(results_by_name, True)),
)

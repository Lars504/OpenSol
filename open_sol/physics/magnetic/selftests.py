"""Analytic self-tests of magnetostatics (planar current sheet on a graded mesh; axisymmetric solenoid; permeable slab and
empty axisymmetric box in an external field; remanent slab; coil pair inductance; saturating sheet (Picard); the sheet
in a tabulated B-H curve is run by selftests_backend) and the 'flux_concentrator_needle' template; specs built fresh.
Every expectation is the exact finite-volume solution on the run's cells."""
import math

import numpy as np

from ...core.constants import MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (EXACT_CG_TOLERANCE, cell_edges, max_relative_error, relative_error, solver_role,
                                       with_direct_node, with_exact_cg, worst_error)
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, RefinementSpec, SolverSpec, StudySpec, SweepSpec)

MAGNETOSTATICS_TYPE_NAME = 'magnetostatics'   # = the definition's type name (the definition imports this module)
_MAGNETIC = 'magnetic'                        # physics instance name = the type's default instance
_RUN = 'run'
_TOLERANCE = 1e-6   # the exact-CG error ~1e4 x 1e-12 is far below it

# Layout shared by the 1D cases: a 1 x 0.1 mm window of 10 um cells (100 x 10, no EM physics, so no PML), symmetry
# (Neumann) y faces, so every quantity depends on x (axisymmetric: r) only. The band x in [0.4, 0.6] mm (cells
# 40..59, edges on cell edges) holds the current sheet, the solenoid winding, the iron slab or the magnet. The
# 3-point scheme makes the face fluxes exact (conservation) and A the exact sum of face flux times half-cell
# reluctance: the closed forms below are discrete identities up to the CG residual and the float32 field storage.
_LENGTH = 1e-3                     # m, x (r) extent
_WIDTH = 1e-4                      # m, y (z) extent
_CELL = 1e-5                       # m
_N_CELLS = (round(_LENGTH / _CELL), round(_WIDTH / _CELL))   # (100, 10)
_BAND = (0.4e-3, 0.6e-3)           # m
_CURRENT_DENSITY = 1e6             # A/m^2: sheet / winding, B ~ mu0 J 0.2 mm = 2.5e-4 T
_PROBE_CELL = (24, 5)              # window cell of the vector potential probe (uniform mesh)

# Graded variant: refinement line on x = 0.5 mm (dx 2.5 um, margin 0.45 mm), the band edges then off the cell edges:
# the band is the cells whose float32 centres fall in it (as the build rasterises), edges rebuilt from the centres.
_GRADED_LINE_X = 0.5e-3
_GRADED_CELL = 2.5e-6
_GRADED_REACH = 0.45e-3
_GRADED_MIN_RATIO = 2.0

# External-field cases: the outer window ring is held, so only the free cells (ring excluded) are compared.
_APPLIED_FIELD = 1e-3              # T
_SLAB_PERMEABILITY = 1e5           # mu_dc of the slab (pins the contrast)
_UNIFORM_DOMAIN = (0.5e-3, 0.2e-3)   # m, axisymmetric empty box (50 x 20 cells)

_REMANENCE = 1.0                   # T along y in the band: H = -B_r t / (mu0 L) = -1.59e5 A/m, B 0.8 / -0.2 T

# Saturation (stage 4): iron mu_dc 1000 with J_s = 1 T (Frohlich-Kennelly curve, a = mu0 (mu - 1) / J_s = 1.255e-3 m/A).
# _SATURATING_FIELD: the applied B_x of the flux-driven slabs of selftests_fem (B uniform whatever nu). Current-
# driven: J = 1e7 A/m^2 in the band, insulation x_lo, symmetry x_hi and y faces, iron everywhere: face H = -(current to
# the right of the face) exactly, so H runs 0 .. -2000 A/m and every cell solves B = B(H_centre) of the curve (0.72 T at
# 2000 A/m, nu 3.5 x its initial value; d ln nu / d ln B up to 2.5 needs relaxation < 0.57; the damped Picard factor
# |1 - r (1 + k)| over k in [0, 2.5] is smallest, ~0.56 at both ends, at r = 2 / (2 + 2.5) = 0.44, the sheet's value).
_IRON_PERMEABILITY = 1000.0
_IRON_SATURATION = 1.0             # T
_SATURATING_FIELD = 0.8            # T, applied B_x of the flux-driven slab (selftests_fem)
_SATURATING_CURRENT_DENSITY = 1e7  # A/m^2
# Loop stop 1e-8 of ln nu: well above the solve noise of the proposals (the sheet solves with host SuperLU, round-off
# only; the tabulated sheet CG ~1e-11), well below the 1e-6 tolerance in B. The sheet runs the damped Picard loop, the
# tabulated sheet (selftests_backend) Anderson at relaxation 0.5 on the exact pcg node; Anderson on a J_s curve deep in
# saturation is the example magnetic/bh_curve_table_core. The sheet's sparse_direct node replaces ~170 warm-started CG
# iterations per loop step by one host LU of the 1000 unknowns. _NONLINEAR_SETTINGS also serves selftests_3d / _fem3d.
_NONLINEAR_SETTINGS = {'nonlinear_tolerance': 1e-8, 'nonlinear_max_iterations': 300, 'nonlinear_relaxation': 0.3}
_PICARD_SETTINGS = {**_NONLINEAR_SETTINGS, 'nonlinear_method': 'picard', 'nonlinear_relaxation': 0.44}
_DIRECT_NAME = 'direct'            # sparse_direct node of the saturating sheet
_ANDERSON_SETTINGS = {**_NONLINEAR_SETTINGS, 'nonlinear_method': 'anderson', 'nonlinear_relaxation': 0.5}
# Tabulated B-H curve: the test iron's curve sampled at _TABLE_FIELDS as a bh_curve without the origin row (bh_curve
# prepends it); the 1e7 A/m^2 sheet drives H to 2000 A/m, past the last row (1500 A/m), so the cells left of the band
# sit on the mu0 extension B = B_n + mu0 (H - H_n) and the band cells cross every row.
_TABLE_FIELDS = (50.0, 100.0, 200.0, 400.0, 700.0, 1000.0, 1500.0)   # A/m

# Coil pair: 'go' x in [0.2, 0.3] mm carries +I, 'back' x in [0.7, 0.8] mm carries -I, 10 turns each.
_COIL_RANGES = {'go': (0.2e-3, 0.3e-3), 'back': (0.7e-3, 0.8e-3)}
_COIL_CURRENT = 0.1                # A per turn
_COIL_TURNS = 10


# ---- exact discrete solutions (numpy, float64) ----
def _in_range(edges, lo, hi):
    """Cells whose float32 centre lies in [lo, hi), as the build rasterises rectangles."""
    centres = (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)
    return (centres >= np.float32(lo)) & (centres < np.float32(hi))


def _planar_line_solution(edges, current_density, permeability, remanence=None):
    """Exact discrete planar solution in x between A = 0 faces: H_y at the faces and centres [A/m], A [T m] and B_y [T]
    at the centres, energy per m of depth per m of y width [J/m^2]."""
    widths = np.diff(edges)
    reluctivity = 1.0 / (MU0 * permeability)
    half_resistance = 0.5 * widths / reluctivity            # dA / H over a half cell
    resistance = np.zeros(widths.size + 1)
    resistance[:-1] += half_resistance
    resistance[1:] += half_resistance
    remanence = np.zeros_like(widths) if remanence is None else remanence
    drive = np.zeros(widths.size + 1)                        # A_L - A_R from B_r over the two half cells
    drive[:-1] += 0.5 * remanence * widths
    drive[1:] += 0.5 * remanence * widths
    enclosed = np.concatenate(([0.0], np.cumsum(current_density * widths)))   # Ampere: H_f - H_0
    face_field = enclosed - (np.sum(enclosed * resistance) + np.sum(drive)) / np.sum(resistance)
    drop = face_field * resistance + drive                   # A_L - A_R per face (sum 0: A = 0 on both faces)
    centre_field = 0.5 * (face_field[:-1] + face_field[1:])
    return {'H_y': centre_field, 'A': -np.cumsum(drop[:-1]), 'B_y': centre_field / reluctivity + remanence,
            'energy_per_width': 0.5 * float(np.sum(drop ** 2 / resistance))}


def _axisymmetric_line_solution(edges, current_density):
    """Exact discrete (mu = 1) psi(r) between the axis (psi = 0, doubled half-cell conductance) and a symmetry x_hi:
    psi [T m^2], A_phi, B_z [T] at the centres and energy per m of height [J/m]."""
    widths = np.diff(edges)
    radii = 0.5 * (edges[:-1] + edges[1:])
    reluctivity = 1.0 / MU0
    coefficient = reluctivity / radii                        # c = nu / r_c
    conductance = np.empty(widths.size)                      # faces 0..n-1 per m of height
    conductance[0] = 2.0 * (2.0 * coefficient[0] / widths[0])
    conductance[1:] = 1.0 / (0.5 * widths[:-1] / coefficient[:-1] + 0.5 * widths[1:] / coefficient[1:])
    sources = current_density * widths
    flux = np.zeros(widths.size + 1)                         # toward +r per m of height (= -H_z), 0 on x_hi
    flux[:-1] = -np.cumsum(sources[::-1])[::-1]
    psi = -np.cumsum(flux[:-1] / conductance)
    return {'psi': psi, 'A': psi / radii, 'B_y': -0.5 * (flux[:-1] + flux[1:]) / reluctivity,
            'energy_per_height': math.pi * float(np.sum(flux[:-1] ** 2 / conductance))}


def _curve_flux_density(field):
    """B [T] of the Frohlich-Kennelly curve of the test iron at H [A/m] (odd in H)."""
    slope = MU0 * (_IRON_PERMEABILITY - 1.0) / _IRON_SATURATION
    magnitude = np.abs(field)
    return np.sign(field) * (MU0 * magnitude + _IRON_SATURATION * slope * magnitude / (1.0 + slope * magnitude))


def _curve_field(flux_density):
    """H [A/m] of the curve at B >= 0 [T]: bisection on the monotone B(H) (independent of the solver's closed form)."""
    low, high = 0.0, flux_density / MU0
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if _curve_flux_density(middle) < flux_density else (low, middle)
    return 0.5 * (low + high)


def _table_rows():
    """bh_curve rows [[H, B], ...] of the test iron's curve at _TABLE_FIELDS (plain floats, no origin row)."""
    return [[field, float(_curve_flux_density(field))] for field in _TABLE_FIELDS]


def _table_flux_density(field):
    """B [T] of the piecewise-linear test table at H [A/m] (odd in H): linear between the origin and the rows, slope
    mu0 beyond the last row (forward B(H), independent of the solver's inverse H(B))."""
    nodes = np.array([[0.0, 0.0]] + _table_rows())
    magnitude = np.abs(field)
    flux = np.where(magnitude > nodes[-1, 0], nodes[-1, 1] + MU0 * (magnitude - nodes[-1, 0]),
                    np.interp(magnitude, nodes[:, 0], nodes[:, 1]))
    return np.sign(field) * flux


def _nonlinear_iterations(results):
    """Picard iterations of the run (study summary solver records), None when not reported."""
    record = (results.summary().get('solvers') or {}).get(_MAGNETIC, {})
    return (record.get('counters') or {}).get('nonlinear_iterations')


# ---- specs ----
def _band_rectangle(name, x0, x1, height=_WIDTH):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': 0.0, 'x1': x1, 'y1': height})


def _model(title, features, materials=(), geometry=(), derived_values=(), axisymmetric=False, graded=False,
           size=(_LENGTH, _WIDTH), extra_settings=None, direct=False):
    """Stationary magnetostatics-only model, float64 CG to 1e-12 (direct: host SuperLU), uniform _CELL mesh or graded
    toward x = 0.5 mm."""
    settings = dict(extra_settings or {})
    if axisymmetric:
        settings['geometry'] = 'axisymmetric'
    geometry, mesh = list(geometry), MeshSpec(mode='uniform', dx=_CELL)
    if graded:
        title = f'{title}, graded mesh'
        geometry.append(GeometrySpec('refine_line', 'line',
                                     {'x0': _GRADED_LINE_X, 'y0': 0.0, 'x1': _GRADED_LINE_X, 'y1': size[1]}))
        mesh = MeshSpec(mode='nonuniform', dx=_CELL,
                        refinements=[RefinementSpec('refine_line', _GRADED_CELL, _GRADED_REACH)])
    spec = ModelSpec(
        title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]), geometry=geometry, materials=list(materials),
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_TYPE_NAME, settings=settings, features=list(features))],
        mesh=mesh, study=StudySpec(type='stationary'), derived_values=list(derived_values))
    if direct:   # the host SuperLU node _DIRECT_NAME as the magnetic 'linear' solver
        return with_direct_node(spec, 'scipy_splu', physics_names=(_MAGNETIC,), node_name=_DIRECT_NAME)
    return with_exact_cg(spec)


def _mirror_y():
    """Fresh symmetry feature on both y faces (1D in x)."""
    return FeatureSpec('mirror', 'symmetry', ['y_lo', 'y_hi'])


def _energy_value():
    """Fresh derived value W = the energy scalar (a constant field: 'max' reads it)."""
    return DerivedValueSpec('W', f'{_MAGNETIC}_energy', [], 'max')


def _current_sheet_specs(graded=False):
    """'run': J = 1e6 A/m^2 in the band between insulation x faces, a probe in uniform cell (24, 5)."""
    probe_x, probe_y = ((index + 0.5) * _CELL for index in _PROBE_CELL)
    return {_RUN: _model(
        'Self-test: planar current sheet (magnetostatics)',
        [FeatureSpec('walls', 'magnetic_insulation', ['x_lo', 'x_hi']), _mirror_y(),
         FeatureSpec('sheet', 'current_source', ['band'], {'density': _CURRENT_DENSITY}),
         FeatureSpec('potential', 'probe', ['probe_point'])],
        geometry=[_band_rectangle('band', *_BAND), GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y})],
        derived_values=[_energy_value()], graded=graded)}


def _solenoid_specs():
    """'run': axisymmetric winding J_phi = 1e6 A/m^2 in a < r < b, symmetry on x_hi and both z faces."""
    return {_RUN: _model(
        'Self-test: long solenoid (axisymmetric magnetostatics)',
        [FeatureSpec('mirror', 'symmetry', ['x_hi', 'y_lo', 'y_hi']),
         FeatureSpec('winding', 'current_source', ['band'], {'density': _CURRENT_DENSITY})],
        geometry=[_band_rectangle('band', *_BAND)], derived_values=[_energy_value()], axisymmetric=True)}


def _permeable_slab_specs():
    """'run': mu_dc 1e5 band across the full y span in an applied B_x = 1e-3 T (normal to the slab faces)."""
    return {_RUN: _model(
        'Self-test: permeable slab in an applied field (magnetostatics)',
        [FeatureSpec('applied', 'external_field', ['all'], {'B_x': _APPLIED_FIELD, 'B_y': 0.0})],
        materials=[MaterialSpec('iron', properties={'mu_dc': _SLAB_PERMEABILITY}, selection=['band'])],
        geometry=[_band_rectangle('band', *_BAND)], derived_values=[_energy_value()])}


def _uniform_field_axisymmetric_specs():
    """'run': empty axisymmetric box, applied axial B = 1e-3 T."""
    return {_RUN: _model(
        'Self-test: uniform axial field (axisymmetric magnetostatics)',
        [FeatureSpec('applied', 'external_field', ['all'], {'B_x': 0.0, 'B_y': _APPLIED_FIELD})],
        axisymmetric=True, size=_UNIFORM_DOMAIN)}


def _remanent_slab_specs():
    """'run': magnet with remanence_y = 1 T in the band between insulation x faces."""
    return {_RUN: _model(
        'Self-test: remanent slab between insulating faces (magnetostatics)',
        [FeatureSpec('walls', 'magnetic_insulation', ['x_lo', 'x_hi']), _mirror_y()],
        materials=[MaterialSpec('magnet', properties={'remanence_y': _REMANENCE}, selection=['band'])],
        geometry=[_band_rectangle('band', *_BAND)])}


def _iron(selection):
    return MaterialSpec('iron', properties={'mu_dc': _IRON_PERMEABILITY, 'saturation_polarization': _IRON_SATURATION},
                        selection=list(selection))


def _saturating_sheet_specs(density=_SATURATING_CURRENT_DENSITY, settings=None, material=None, direct=True):
    """'run': J in the band, saturating iron everywhere (J_s curve unless material is given), insulation x_lo,
    symmetry x_hi and y faces (damped Picard unless settings are given; sparse_direct node, else exact pcg)."""
    return {_RUN: _model(
        'Self-test: current sheet in saturating iron (magnetostatics)',
        [FeatureSpec('wall', 'magnetic_insulation', ['x_lo']),
         FeatureSpec('mirror', 'symmetry', ['x_hi', 'y_lo', 'y_hi']),
         FeatureSpec('sheet', 'current_source', ['band'], {'density': density})],
        materials=[_iron(['all']) if material is None else material], geometry=[_band_rectangle('band', *_BAND)],
        extra_settings=_PICARD_SETTINGS if settings is None else settings, direct=direct)}


def _tabulated_sheet_specs():
    """'run': the 1e7 A/m^2 sheet in iron given by the bh_curve table (Anderson at relaxation 0.5, exact pcg node;
    selftests_backend magnet_cpu_gpu)."""
    return _saturating_sheet_specs(settings=_ANDERSON_SETTINGS, direct=False, material=MaterialSpec(
        'iron_table', properties={'bh_curve': _table_rows()}, selection=['all']))


def _coil_pair_specs():
    """'run': coils go (+I) and back (-I), 10 turns each, between insulation x faces; derived W, L_go, L_back."""
    return {_RUN: _model(
        'Self-test: go / return coil pair (magnetostatics)',
        [_mirror_y(),
         FeatureSpec('go', 'coil', ['go'], {'current': _COIL_CURRENT, 'turns': _COIL_TURNS}),
         FeatureSpec('back', 'coil', ['back'], {'current': -_COIL_CURRENT, 'turns': _COIL_TURNS})],
        geometry=[_band_rectangle(name, *span) for name, span in _COIL_RANGES.items()],
        derived_values=[_energy_value(), DerivedValueSpec('L_go', f'{_MAGNETIC}_go_inductance', [], 'max'),
                        DerivedValueSpec('L_back', f'{_MAGNETIC}_back_inductance', [], 'max')])}


# ---- checks (CPU, on Results-like objects) ----
def _window_field(results, component, shape=_N_CELLS, graded=False):
    """(float64 window values, float64 x edges, float64 y centres) of a magnetic field; ValueError unless a uniform
    run has `shape` and a graded one x widths spanning at least _GRADED_MIN_RATIO."""
    field_data = results.field(f'{_MAGNETIC}.{component}')
    values = np.asarray(field_data.values, dtype=np.float64)
    edges = cell_edges(np.asarray(field_data.x_centers, dtype=np.float64), float(field_data.extent[0]))
    widths = np.diff(edges)
    if not graded and values.shape != shape:
        raise ValueError(f'{component}: expected the window shape {shape}, got {values.shape}')
    if graded and not (values.shape[0] == widths.size and np.max(widths) >= _GRADED_MIN_RATIO * np.min(widths)):
        raise ValueError(f'{component}: expected graded x cells (widest / narrowest >= {_GRADED_MIN_RATIO:g}), got '
                         f'{widths.size} cells of {np.min(widths):.4g} .. {np.max(widths):.4g} m')
    return values, edges, np.asarray(field_data.y_centers, dtype=np.float64)


def _derived_value(results, name):
    return float(results.summary()['derived'][name])   # float() also reads 'nan'


def _cg_iterations(results):
    """CG iterations of the run's solves (study summary solvers: the linear role's counters), None when not reported."""
    return solver_role(results, _MAGNETIC).get('counters', {}).get('cg_iterations')


def _report(measured, expected, errors, **details):
    return {'measured': measured, 'expected': expected, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, **details}}


def _check_current_sheet(results_by_name, graded=False):
    """A, B_y, B_x = 0, H_y, the energy scalar and the probe against the exact discrete sheet solution."""
    results = results_by_name[_RUN]
    potential, edges, _ = _window_field(results, 'A', graded=graded)
    flux_density_y, _, _ = _window_field(results, 'B_y', graded=graded)
    flux_density_x, _, _ = _window_field(results, 'B_x', graded=graded)
    field_y, _, _ = _window_field(results, 'H_y', graded=graded)
    density = np.where(_in_range(edges, *_BAND), _CURRENT_DENSITY, 0.0)
    exact = _planar_line_solution(edges, density, np.ones_like(density))
    energy_closed = _WIDTH * exact['energy_per_width']
    energy = _derived_value(results, 'W')
    probe_cell = int(np.clip(np.searchsorted(edges, (_PROBE_CELL[0] + 0.5) * _CELL, side='right') - 1,
                             0, edges.size - 2))
    probe = float(np.asarray(results.result(_MAGNETIC, 'potential')['values'], dtype=np.float64)[-1])
    potential_scale = float(np.max(np.abs(exact['A'])))
    field_scale = float(np.max(np.abs(exact['B_y'])))
    errors = {'A': max_relative_error(potential, exact['A'][:, None], potential_scale),
              'B_y': max_relative_error(flux_density_y, exact['B_y'][:, None], field_scale),
              'B_x': max_relative_error(flux_density_x, 0.0, field_scale),
              'H_y': max_relative_error(field_y, exact['H_y'][:, None], float(np.max(np.abs(exact['H_y'])))),
              'energy': relative_error(energy, energy_closed),
              'probe': max_relative_error(probe, exact['A'][probe_cell], potential_scale)}
    return _report(energy, energy_closed, errors, energy_J_per_m=energy, B_max_T=field_scale,
                   A_probe=probe, probe_cell=probe_cell, n_cells=list(potential.shape))


def _check_solenoid(results_by_name):
    """A_phi = psi / r_c, B_z (also against mu0 J (b - a) on the axis column), B_r = 0 and the energy."""
    results = results_by_name[_RUN]
    potential, edges, _ = _window_field(results, 'A')
    axial, _, _ = _window_field(results, 'B_y')
    radial, _, _ = _window_field(results, 'B_x')
    density = np.where(_in_range(edges, *_BAND), _CURRENT_DENSITY, 0.0)
    exact = _axisymmetric_line_solution(edges, density)
    inside = MU0 * _CURRENT_DENSITY * (_BAND[1] - _BAND[0])
    energy_closed = _WIDTH * exact['energy_per_height']
    energy = _derived_value(results, 'W')
    errors = {'A': max_relative_error(potential, exact['A'][:, None], float(np.max(np.abs(exact['A'])))),
              'B_z': max_relative_error(axial, exact['B_y'][:, None], inside),
              'B_z_axis': max_relative_error(axial[0], inside, inside),
              'B_r': max_relative_error(radial, 0.0, inside),
              'energy': relative_error(energy, energy_closed)}
    return _report(float(np.mean(axial[0])), inside, errors, B_inside_closed_T=inside, energy_J=energy,
                   energy_closed_J=energy_closed, n_cells=list(potential.shape))


def _check_permeable_slab(results_by_name):
    """B_x = B, B_y = 0, A = B y and H_x = B / (mu0 mu) in the free cells and the energy; reports the CG iterations."""
    results = results_by_name[_RUN]
    potential, edges, y_centres = _window_field(results, 'A')
    flux_density_x, _, _ = _window_field(results, 'B_x')
    flux_density_y, _, _ = _window_field(results, 'B_y')
    field_x, _, _ = _window_field(results, 'H_x')
    free = (slice(1, -1), slice(1, -1))
    permeability = np.where(_in_range(edges, *_BAND), _SLAB_PERMEABILITY, 1.0)
    widths = np.diff(edges)
    free_height = float(np.sum(np.diff(cell_edges(y_centres, 0.0))[1:-1]))
    energy_closed = 0.5 * _APPLIED_FIELD ** 2 * free_height * float(np.sum(widths[1:-1] / (MU0 * permeability[1:-1])))
    energy = _derived_value(results, 'W')
    field_closed = _APPLIED_FIELD / (MU0 * permeability)
    iterations = _cg_iterations(results)
    errors = {'B_x': max_relative_error(flux_density_x[free], _APPLIED_FIELD, _APPLIED_FIELD),
              'B_y': max_relative_error(flux_density_y[free], 0.0, _APPLIED_FIELD),
              'A': max_relative_error(potential[free], _APPLIED_FIELD * y_centres[None, 1:-1], _APPLIED_FIELD * _WIDTH),
              'H_x': max_relative_error(field_x[free], field_closed[1:-1, None], float(np.max(field_closed))),
              'energy': relative_error(energy, energy_closed)}
    return _report(float(np.mean(flux_density_x[free])), _APPLIED_FIELD, errors, cg_iterations=iterations,
                   cg_iterations_over_20x_axis=None if iterations is None else iterations > 20 * max(_N_CELLS),
                   energy_J_per_m=energy, energy_closed_J_per_m=energy_closed, n_cells=list(potential.shape))


def _check_uniform_field_axisymmetric(results_by_name):
    """B_z = B, B_r = 0 and A_phi = B r_c / 2 in every free cell (the axis column included)."""
    results = results_by_name[_RUN]
    shape = tuple(round(size / _CELL) for size in _UNIFORM_DOMAIN)
    potential, edges, _ = _window_field(results, 'A', shape=shape)
    axial, _, _ = _window_field(results, 'B_y', shape=shape)
    radial, _, _ = _window_field(results, 'B_x', shape=shape)
    free = (slice(0, -1), slice(1, -1))   # the ring: x_hi column and both z rows (x_lo is the axis)
    radii = 0.5 * (edges[:-1] + edges[1:])
    errors = {'B_z': max_relative_error(axial[free], _APPLIED_FIELD, _APPLIED_FIELD),
              'B_z_axis': max_relative_error(axial[0, 1:-1], _APPLIED_FIELD, _APPLIED_FIELD),
              'B_r': max_relative_error(radial[free], 0.0, _APPLIED_FIELD),
              'A': max_relative_error(potential[free], 0.5 * _APPLIED_FIELD * radii[:-1, None],
                                      0.5 * _APPLIED_FIELD * _UNIFORM_DOMAIN[0])}
    return _report(float(np.mean(axial[0, 1:-1])), _APPLIED_FIELD, errors, n_cells=list(potential.shape))


def _check_remanent_slab(results_by_name):
    """H_y uniform = -B_r t / (mu0 L), B_y = mu0 H + B_r and A against the exact discrete magnet solution."""
    results = results_by_name[_RUN]
    potential, edges, _ = _window_field(results, 'A')
    flux_density, _, _ = _window_field(results, 'B_y')
    field, _, _ = _window_field(results, 'H_y')
    remanence = np.where(_in_range(edges, *_BAND), _REMANENCE, 0.0)
    exact = _planar_line_solution(edges, np.zeros_like(remanence), np.ones_like(remanence), remanence)
    field_closed = -_REMANENCE * (_BAND[1] - _BAND[0]) / (MU0 * _LENGTH)
    errors = {'H_y': max_relative_error(field, exact['H_y'][:, None], abs(field_closed)),
              'H_closed': max_relative_error(exact['H_y'], field_closed, abs(field_closed)),
              'B_y': max_relative_error(flux_density, exact['B_y'][:, None], _REMANENCE),
              'A': max_relative_error(potential, exact['A'][:, None], float(np.max(np.abs(exact['A']))))}
    return _report(float(np.mean(field)), field_closed, errors, B_inside_T=float(np.max(exact['B_y'])),
                   B_outside_T=float(np.min(exact['B_y'])), n_cells=list(potential.shape))


def _check_saturating_sheet(results_by_name, sheet_density=_SATURATING_CURRENT_DENSITY,
                            curve_flux_density=_curve_flux_density):
    """H_y = -(current to the right) at the faces, centre mean exact; B_y = B(H_y) of the curve in every cell."""
    results = results_by_name[_RUN]
    flux_density, edges, _ = _window_field(results, 'B_y')
    field, _, _ = _window_field(results, 'H_y')
    flux_density_x, _, _ = _window_field(results, 'B_x')
    widths = np.diff(edges)
    density = np.where(_in_range(edges, *_BAND), sheet_density, 0.0)
    right_current = np.concatenate((np.cumsum((density * widths)[::-1])[::-1], [0.0]))
    field_closed = -0.5 * (right_current[:-1] + right_current[1:])
    flux_closed = curve_flux_density(field_closed)
    field_scale = float(np.max(np.abs(field_closed)))
    flux_scale = float(np.max(np.abs(flux_closed)))
    errors = {'H_y': max_relative_error(field, field_closed[:, None], field_scale),
              'B_y': max_relative_error(flux_density, flux_closed[:, None], flux_scale),
              'B_x': max_relative_error(flux_density_x, 0.0, flux_scale)}
    return _report(float(np.min(flux_density)), float(np.min(flux_closed)), errors, unit='T',
                   linear_B_min_T=float(np.min(MU0 * _IRON_PERMEABILITY * field_closed)),
                   nonlinear_iterations=_nonlinear_iterations(results), n_cells=list(field.shape))


def _check_coil_pair(results_by_name):
    """A, the energy, the go coil's flux linkage and L_go + L_back = 2 W / I^2 against the exact discrete pair."""
    results = results_by_name[_RUN]
    potential, edges, _ = _window_field(results, 'A')
    widths = np.diff(edges)
    masks = {name: _in_range(edges, *span) for name, span in _COIL_RANGES.items()}
    sign = {'go': 1.0, 'back': -1.0}
    density = sum(sign[name] * _COIL_TURNS * _COIL_CURRENT / (_WIDTH * float(np.sum(widths[mask])))
                  * mask.astype(np.float64) for name, mask in masks.items())
    exact = _planar_line_solution(edges, density, np.ones_like(widths))
    energy_closed = _WIDTH * exact['energy_per_width']
    loop_inductance = 2.0 * energy_closed / _COIL_CURRENT ** 2
    linkage_closed = _COIL_TURNS * float(np.sum(exact['A'][masks['go']] * widths[masks['go']])
                                         / np.sum(widths[masks['go']]))
    energy = _derived_value(results, 'W')
    inductance_sum = _derived_value(results, 'L_go') + _derived_value(results, 'L_back')
    linkage = float(results.result(_MAGNETIC, 'go')['flux_linkage'])
    errors = {'A': max_relative_error(potential, exact['A'][:, None], float(np.max(np.abs(exact['A'])))),
              'energy': relative_error(energy, energy_closed),
              'flux_linkage_go': relative_error(linkage, linkage_closed),
              'loop_inductance': relative_error(inductance_sum, loop_inductance)}
    return _report(inductance_sum, loop_inductance, errors, unit='H/m', energy_J_per_m=energy,
                   flux_linkage_go_Wb_per_m=linkage, n_cells=list(potential.shape))


# ---- template ----
# flux_concentrator_needle: Metglas needle r = 31 um, l = 15 mm (mu_dc 1e5) on the axis of a 10 x 40 mm (r, z) box in
# an axial applied field; sensor disc (r < 31 um, 10 um thick) 'gap' below the lower tip. Mesh: base 0.5 mm, the
# needle's side refined in r only (vertical line, 5 um), the lower tip zone in r and z (5 um, holds every sensor
# position of the sweep), the upper tip at 20 um. The external-field ring sits 10 mm (r) and 12.5 mm (z) from the
# needle, where its own dipole field is small but not zero. Not run yet: mesh size, CG count and gains unverified.
FLUX_CONCENTRATOR_LABEL = 'Flux-concentrator needle'
FLUX_CONCENTRATOR_DESCRIPTION = (
    'Magnetostatics, axisymmetric, stationary sweep: a Metglas needle (r = 31 um, l = 15 mm, mu_dc 1e5) in a uniform '
    'axial field B_ext = 1e-4 T concentrates the flux at its tip; derived B_sensor (mean |B| over a 31 um disc a gap '
    'below the tip) and gain = B_sensor / B_ext (EF = 20 log10(gain) dB), swept over gap = 25 .. 400 um.')
_NEEDLE_RADIUS = 31e-6             # m
_NEEDLE_LENGTH = 15e-3             # m
_NEEDLE_BOTTOM = 12.5e-3           # m, z of the lower tip
_NEEDLE_DOMAIN = (10e-3, 40e-3)    # m (r, z)
_SENSOR_THICKNESS = 10e-6          # m
_GAPS = [25e-6, 50e-6, 100e-6, 200e-6, 400e-6]   # m


_NEEDLE_PARAMETERS = (('B_ext', 1e-4, 'applied axial flux density [T] (AC: amplitude)'),
                      ('mu_needle', 1e5, 'relative permeability of the needle'))
_GAP_PARAMETER = ('gap', 100e-6, 'distance from the lower needle tip to the sensor centre [m]')


def needle_model(title, description, physics_type, settings, parameters, needle_properties, sweep):
    """Flux-concentrator needle model shared by the DC (gap sweep) and AC (frequency sweep) templates: geometry, mesh,
    external field, metglas needle, derived B_sensor / gain; parameters (name, value, description) after B_ext and
    mu_needle, needle_properties added to mu_dc = mu_needle."""
    size_r, size_z = _NEEDLE_DOMAIN
    top = _NEEDLE_BOTTOM + _NEEDLE_LENGTH
    half_sensor = 0.5 * _SENSOR_THICKNESS

    def rectangle(name, x0, y0, x1, y1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    return ModelSpec(
        title=title,
        description=description,
        parameters=[ParameterSpec(*entry) for entry in (*_NEEDLE_PARAMETERS, *parameters)],
        domain=DomainSpec(size_x=size_r, size_y=size_z),
        geometry=[rectangle('needle', 0.0, _NEEDLE_BOTTOM, _NEEDLE_RADIUS, top),
                  rectangle('sensor', 0.0, f'{_NEEDLE_BOTTOM!r} - gap - {half_sensor!r}', _NEEDLE_RADIUS,
                            f'{_NEEDLE_BOTTOM!r} - gap + {half_sensor!r}'),
                  GeometrySpec('needle_side', 'line', {'x0': _NEEDLE_RADIUS, 'y0': _NEEDLE_BOTTOM,
                                                       'x1': _NEEDLE_RADIUS, 'y1': top}),
                  rectangle('tip_zone', 0.0, _NEEDLE_BOTTOM - 0.6e-3, 0.2e-3, _NEEDLE_BOTTOM + 0.2e-3),
                  rectangle('top_zone', 0.0, top - 0.2e-3, 0.2e-3, top + 0.2e-3)],
        materials=[MaterialSpec('metglas', properties={'mu_dc': 'mu_needle', **needle_properties},
                                selection=['needle'])],
        physics=[PhysicsSpec(_MAGNETIC, physics_type, settings={'geometry': 'axisymmetric', **settings},
                             features=[FeatureSpec('applied', 'external_field', ['all'],
                                                   {'B_x': 0.0, 'B_y': 'B_ext'})])],
        mesh=MeshSpec(mode='nonuniform', dx=0.5e-3,
                      refinements=[RefinementSpec('needle_side', 5e-6, 2e-3),
                                   RefinementSpec('tip_zone', 5e-6, 1e-3),
                                   RefinementSpec('top_zone', 20e-6, 1e-3)]),
        study=StudySpec(type='stationary', sweep=sweep),
        derived_values=[DerivedValueSpec('B_sensor', 'B_abs', ['sensor'], 'mean'),
                        DerivedValueSpec('gain', 'B_abs / B_ext', ['sensor'], 'mean')])


def flux_concentrator_needle_spec():
    """Template 'flux_concentrator_needle' (axisymmetric magnetostatics, stationary sweep over gap) with a pcg node
    'cg' at tolerance 1e-10 (the 1e5 permeability contrast)."""
    spec = needle_model(FLUX_CONCENTRATOR_LABEL, FLUX_CONCENTRATOR_DESCRIPTION, MAGNETOSTATICS_TYPE_NAME, {},
                        [_GAP_PARAMETER], {}, SweepSpec('gap', list(_GAPS)))
    spec.solvers.append(SolverSpec('cg', 'pcg', {'tolerance': 1e-10, 'max_iterations': 100000}))
    spec.physics[0].solvers['linear'] = 'cg'
    return spec


# ---- mirror parity (symmetry 2.5; CPU, no run) ----
_MIRROR_EM = 'em'   # a wave physics whose bc_x_lo symmetric makes x_lo a model mirror plane (mirror_faces reads specs)


def _mirror_parity_spec(face_features):
    """Spec (never built) of the magnetic physics with face_features and an em_fdfd_ez setting the x_lo mirror."""
    return ModelSpec(title='magnet mirror parity', domain=DomainSpec(size_x=_LENGTH / 2, size_y=_WIDTH),
                     physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_TYPE_NAME, features=list(face_features)),
                              PhysicsSpec(_MIRROR_EM, 'em_fdfd_ez', settings={'bc_x_lo': 'symmetric'})])


def _check_mirror_parity(_results_by_name):
    """Plane signs from the magnetic face feature (symmetry +1, default / explicit insulation -1, last wins) and the
    unfolded half of the exact symmetric sheet / antisymmetric sheet pair vs the full solution (A, B_y, H_y)."""
    from ...core.mirror import mirror_faces, unfold_field
    from ...model.session import FieldData
    from .renderer import MAGNETIC_FIELD_INFOS
    symmetry = FeatureSpec('mirror', 'symmetry', ['x_lo'])
    insulation = FeatureSpec('wall', 'magnetic_insulation', ['x_lo'])
    signs = {'symmetry': mirror_faces(_mirror_parity_spec([symmetry]), _MAGNETIC).get('x_lo'),
             'default': mirror_faces(_mirror_parity_spec([]), _MAGNETIC).get('x_lo'),
             'last_wins': mirror_faces(_mirror_parity_spec([symmetry, insulation]), _MAGNETIC).get('x_lo')}
    errors = {f'sign_{name}': 0.0 if sign == expected else math.inf
              for (name, sign), expected in zip(signs.items(), (1, -1, -1))}
    edges = np.linspace(0.0, _LENGTH, _N_CELLS[0] + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    plane = _N_CELLS[0] // 2   # the plane x = 0.5 mm, the half's x_lo
    in_band = (centres >= _BAND[0]) & (centres < _BAND[1])
    layouts = {'symmetry': np.where(in_band, _CURRENT_DENSITY, 0.0),   # A even about the plane
               'default': np.where(in_band, _CURRENT_DENSITY * np.sign(centres - _LENGTH / 2), 0.0)}   # A odd
    for name, density in layouts.items():
        exact = _planar_line_solution(edges, density, np.ones_like(density))
        faces = {'x_lo': signs[name]}
        for component in ('A', 'B_y', 'H_y'):
            half = FieldData(_MAGNETIC, component, exact[component][plane:, None].astype(np.float32),
                             (_LENGTH / 2, _LENGTH, 0.0, _WIDTH), MAGNETIC_FIELD_INFOS[component], 0.0, None)
            unfolded = np.asarray(unfold_field(half, faces).values, dtype=np.float64)[:, 0]
            errors[f'{name}_{component}'] = (math.inf if unfolded.shape != exact[component].shape else
                                             max_relative_error(unfolded, exact[component],
                                                                float(np.max(np.abs(exact[component])))))
    return _report(signs['symmetry'], 1, errors, signs=signs)


# ---- cases ----
_WINDOW_TEXT = (f'a {_LENGTH * 1e3:g} x {_WIDTH * 1e3:g} mm window of uniform {_CELL * 1e6:g} um cells '
                f'({_N_CELLS[0]} x {_N_CELLS[1]}, no PML) with symmetry y faces (1D in x)')
_LAYOUT_TEXT = f'{_WINDOW_TEXT}, solved in float64 to a CG tolerance of {EXACT_CG_TOLERANCE:g} in a stationary study'
_DIRECT_LAYOUT_TEXT = (f"{_WINDOW_TEXT}, each loop step solved by host SuperLU (sparse_direct node, back_end "
                       "'scipy_splu', float64) in a stationary study")
_BUDGET_TEXT = (f'tolerance {_TOLERANCE:g}: the expectation is the exact finite-volume solution on the cells (face '
                'fluxes from Ampere / conservation, A summed over the half-cell reluctances), so only the CG residual '
                'and the float32 field storage (~1e-7) remain; the energy scalar comes from the float64 solution')

SELFTEST_CASES = (
    SelfTestCase(
        name='magnet_slab_current_sheet_graded', label='Magnetostatics: planar current sheet on a graded mesh',
        description=(f'Magnetostatics on a {_LENGTH * 1e3:g} x {_WIDTH * 1e3:g} mm window (symmetry y faces, 1D in '
                     f"x), float64 CG to {EXACT_CG_TOLERANCE:g}, stationary, mode 'nonuniform': base "
                     f'{_CELL * 1e6:g} um and a refinement line on x = {_GRADED_LINE_X * 1e3:g} mm (dx '
                     f'{_GRADED_CELL * 1e6:g} um, margin {_GRADED_REACH * 1e3:g} mm). J_z = {_CURRENT_DENSITY:g} '
                     f'A/m^2 (current_source) in the band x in [{_BAND[0] * 1e3:g}, {_BAND[1] * 1e3:g}] mm, magnetic '
                     'insulation A = 0 on x_lo / x_hi, a vector potential probe at the centre of base cell (24, 5). '
                     'The band edges fall off the cell edges, so the band is the cells whose float32 centres lie in '
                     'it and the closed forms run on the actual cells (edges rebuilt from the float64 centres); the '
                     f'check raises unless the widest / narrowest x cell >= {_GRADED_MIN_RATIO:g}. Exact: H_y at the '
                     'faces from Ampere (H_0 plus the enclosed current, H_0 fixed by A = 0 on both faces), A the sum '
                     'of H times the half-cell reluctance dx / (2 nu), B_y = mu0 H (centre average of two exact face '
                     'values), B_x = 0, the energy (1/2) sum H^2 r_f w. error = max of the relative A, B_y, B_x, H_y, '
                     f'energy and probe errors; measured = the energy scalar; {_BUDGET_TEXT}. Pins the c_xx = nu_yy '
                     'assembly, B = curl A and its sign (B_y > 0 right of a +z current), H, the energy scalar, the '
                     'unequal-width face reluctances and the graded probe lookup.'),
        specs=lambda: _current_sheet_specs(True),
        check=lambda results_by_name: _check_current_sheet(results_by_name, True)),
    SelfTestCase(
        name='magnet_solenoid_axisymmetric', label='Magnetostatics: long solenoid (axisymmetric)',
        description=(f"Axisymmetric magnetostatics (geometry 'axisymmetric') through {_LAYOUT_TEXT} (x = r, y = z): "
                     f'J_phi = {_CURRENT_DENSITY:g} A/m^2 in the winding r in [{_BAND[0] * 1e3:g}, '
                     f'{_BAND[1] * 1e3:g}] mm, symmetry on x_hi and both z faces (a long solenoid: B = 0 outside). '
                     'Exact: the r-face flux (= -H_z per m of height) is minus the current outside the face, psi = r '
                     'A_phi the sum of flux / conductance from the axis (psi = 0 with the doubled half-cell '
                     'conductance 8 nu / dr^2 of the axis cell) over the planar operator with c = nu / r_c, whose '
                     'harmonic face mean is nu / r_face: psi = B r^2 / 2 exactly inside, B_z = mu0 J (b - a) = '
                     f'{MU0 * _CURRENT_DENSITY * (_BAND[1] - _BAND[0]):.6g} T inside (also on the axis column), mu0 J '
                     '(b - r) in the winding, 0 outside, B_r = 0, the energy pi h sum F^2 / G. error = max of the '
                     f'relative A_phi, B_z, axis B_z, B_r and energy errors; {_BUDGET_TEXT}. Pins the psi '
                     'formulation on the planar operator (never the ring-weighted one), the axis rule (x_lo '
                     'conductance_factor 2: assembly, face flux in B and the energy face term) and the axisymmetric '
                     'signs of B.'),
        specs=_solenoid_specs, check=_check_solenoid),
    SelfTestCase(
        name='magnet_permeable_slab', label='Magnetostatics: permeable slab in an applied field',
        description=(f'Magnetostatics on a {_LENGTH * 1e3:g} x {_WIDTH * 1e3:g} mm window of uniform '
                     f'{_CELL * 1e6:g} um cells ({_N_CELLS[0]} x {_N_CELLS[1]}, float64 CG to '
                     f'{EXACT_CG_TOLERANCE:g}, stationary): an '
                     f'external_field B_x = {_APPLIED_FIELD:g} T (outer ring held) and a slab of mu_dc = '
                     f'{_SLAB_PERMEABILITY:g} across the full y span on the band (B normal to its faces). Exact in '
                     'every free cell: B_x = B (normal B continuous), B_y = 0, A = B y, H_x = B / (mu0 mu), energy '
                     '(1/2) B^2 sum over the free columns of dx / (mu0 mu) times the free height (the held-free faces '
                     'count their free half cell). error = max of the relative B_x, B_y, A, H_x and energy errors; '
                     f'{_BUDGET_TEXT}; the CG iterations are reported (details, flagged above 20 x the longest axis). '
                     'Pins the held-ring values at the inner faces (centre values would give B (1 + dx / L)), the '
                     '1e5 contrast and H = B / (mu0 mu).'),
        specs=_permeable_slab_specs, check=_check_permeable_slab),
    SelfTestCase(
        name='magnet_uniform_field_axisymmetric', label='Magnetostatics: uniform axial field (axisymmetric)',
        description=(f'Axisymmetric magnetostatics in an empty {_UNIFORM_DOMAIN[0] * 1e3:g} x '
                     f'{_UNIFORM_DOMAIN[1] * 1e3:g} mm (r, z) box of {_CELL * 1e6:g} um cells: external_field B_y = '
                     f'{_APPLIED_FIELD:g} T (axial). The ring holds psi = B r^2 / 2 on both z rows and the exact '
                     'half-cell value (B / 2) r_c (2 r_face - r_c) on x_hi; x_lo is the axis. Exact in every free '
                     'cell including the axis column: B_z = B, B_r = 0, A_phi = B r_c / 2. error = max of the '
                     f'relative B_z, axis B_z, B_r and A_phi errors; {_BUDGET_TEXT}. Pins the axisymmetric ring '
                     'values and the axis conductance.'),
        specs=_uniform_field_axisymmetric_specs, check=_check_uniform_field_axisymmetric),
    SelfTestCase(
        name='magnet_remanent_slab', label='Magnetostatics: remanent slab between insulating faces',
        description=(f'Magnetostatics through {_LAYOUT_TEXT}: remanence_y = {_REMANENCE:g} T (mu_dc 1) in the band, '
                     'magnetic insulation on x_lo / x_hi. No current, so H_y is uniform; A = 0 on both faces makes '
                     f'the integral of B_y vanish: H = -B_r t / (mu0 L) = '
                     f'{-_REMANENCE * (_BAND[1] - _BAND[0]) / (MU0 * _LENGTH):.6g} A/m, B_y = mu0 H + B_r (0.8 T in '
                     'the magnet, -0.2 T outside), exactly also in the cells at the magnet edges (H from the physical '
                     'face flux, B = mu0 mu H + B_r), A the sum of (H r_f + the B_r half-cell drive). error = max of '
                     f'the relative H_y, closed-form H, B_y and A errors; {_BUDGET_TEXT}. Pins the remanence face '
                     'sources (sign and size) and the H / B reconstruction.'),
        specs=_remanent_slab_specs, check=_check_remanent_slab),
    SelfTestCase(
        name='magnet_coil_inductance', label='Magnetostatics: go / return coil pair inductance',
        description=(f'Magnetostatics through {_LAYOUT_TEXT}: stranded coils go (x in [0.2, 0.3] mm, '
                     f'+{_COIL_CURRENT:g} A) and back (x in [0.7, 0.8] mm, -{_COIL_CURRENT:g} A), {_COIL_TURNS} turns each, J = N I / '
                     'area, insulation x faces (default). Exact discrete A as magnet_slab_current_sheet_graded, energy W, '
                     'the go flux linkage N * mean A over the coil, and the loop inductance 2 W / I^2 = L_go + L_back '
                     '(each coil feature reports flux linkage / its own current; for series coils with +I and -I they '
                     'add up to the loop value). error = max of the relative A, energy, go flux linkage and loop '
                     f'inductance errors; measured = L_go + L_back [H/m]; {_BUDGET_TEXT}. Pins the coil current '
                     'density, the flux linkage / inductance results and scalars, and the energy identity.'),
        specs=_coil_pair_specs, check=_check_coil_pair),
    SelfTestCase(
        name='magnet_saturating_sheet', label='Magnetostatics: current sheet in saturating iron',
        description=(f'Magnetostatics through {_DIRECT_LAYOUT_TEXT}: J_z = {_SATURATING_CURRENT_DENSITY:g} A/m^2 in the '
                     f'band, iron (mu_dc {_IRON_PERMEABILITY:g}, saturation_polarization {_IRON_SATURATION:g} T) '
                     'everywhere, insulation on x_lo, symmetry on x_hi and the y faces. Face H_y = -(current to the '
                     'right of the face) exactly whatever nu (Ampere), so each cell must satisfy B_y = B(H_y) of the '
                     'curve at its centre mean H: 0 .. -0.72 T (linear iron would give -2.5 T). The current imposes H, '
                     "so undamped Picard on nu oscillates (d ln nu / d ln B up to 2.5): nonlinear_method 'picard', "
                     f"relaxation {_PICARD_SETTINGS['nonlinear_relaxation']:g} (balances the factor |1 - r (1 + k)| "
                     'over k = d ln nu / d ln B in [0, 2.5], below the 0.57 limit), tolerance '
                     f"{_PICARD_SETTINGS['nonlinear_tolerance']:g}. error = max of the relative H_y, B_y and B_x "
                     f'errors; measured = min B_y [T]; tolerance {_TOLERANCE:g} (the direct solves leave round-off '
                     'and the float32 field storage, ~1e-7, plus the loop stop); the Picard iterations are reported. '
                     'Pins the damped Picard loop on a current-driven part. Not run yet with the sparse_direct node '
                     'and relaxation 0.44 (was pcg, 0.3: 42 iterations).'),
        specs=_saturating_sheet_specs, check=_check_saturating_sheet),
    SelfTestCase(
        name='magnet_mirror_parity', label='Magnetostatics: mirror parity of A, B and H',
        description=('Symmetry 2.5, CPU, no run: on a model mirror plane (an em_fdfd_ez bc_x_lo symmetric in the '
                     'spec) the magnetic fields unfold with the sign of their own face feature there '
                     '(core.mirror.mirror_faces: symmetry +1, magnetic insulation -1, also the default of an unset '
                     'face; the last feature wins). The exact discrete 1D solutions (the uniform 10 um sheet '
                     'layout, insulation x faces) of a sheet symmetric about x = 0.5 mm (A even) and of an '
                     'antisymmetric +J / -J sheet pair (A odd, A = 0 on the plane) are cut to x >= 0.5 mm and '
                     'unfolded over x_lo (core.mirror.unfold_field with the declared parities: A polar_z, B_y / H_y '
                     'axial_y). Expected: signs +1 / -1 / -1 and the unfolded A, B_y, H_y equal the full solutions. '
                     f'error = the largest relative deviation (signs: 0 or inf); tolerance {_TOLERANCE:g} (float32 '
                     'storage).'),
        specs=dict, check=_check_mirror_parity),
)

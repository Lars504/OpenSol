"""Self-tests of the magnetic couplings (induction heating of the skin-depth slab; Lorentz force on a current sheet) and
the 'induction_billet' template; registered by coupling/definitions_magnetic. Expectations are exact finite-volume
identities on the run's cells (host float64)."""
import numpy as np

from ...core.constants import MU0
from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, max_relative_error, relative_error, with_exact_cg
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec, StudySpec)
from .selftests import (_BAND, _CELL, _CURRENT_DENSITY, _LENGTH, _MAGNETIC, _RUN, _TOLERANCE, _WIDTH,
                        MAGNETOSTATICS_TYPE_NAME, _band_rectangle, _derived_value, _in_range, _report, _window_field)
from .selftests_ac import (_COPPER, _DRIVE_POTENTIAL, _SKIN_FREQUENCY, MAGNETIC_AC_TYPE_NAME, _skin_solution,
                           _skin_specs)

INDUCTION_COUPLING_NAME = 'magnetic_heat_induction'        # = the coupling type names (definitions_magnetic)
LORENTZ_COUPLING_NAME = 'magnetic_solid_lorentz_force'
HEAT_TYPE_NAME = 'heat'                                    # = coupling.definitions.HEAT_TYPE_NAME
SOLID_TYPE_NAME = 'solid_mechanics'                        # = physics.mechanics SOLID_TYPE_NAME (not imported)
_HEAT = 'heat'
_SOLID = 'solid'
_T0 = 293.15                       # K
_SLAB_CONDUCTIVITY = 1.0           # W/(m K), unphysical for copper: a ~1 K rise from the ~1e4 W/m^2 skin loss
_YOUNGS_MODULUS = 1.1e11           # Pa
_POISSON_RATIO = 0.34
# the same runs on the cpu backend (numpy fields, host SuperLU AC solve, numpy pcg): the identities hold there too
_CPU_VARIANTS = (SelfTestVariant('cpu', tolerance=_TOLERANCE, backend='cpu'),)


# ---- exact discrete solutions (numpy, float64) ----
def _dirichlet_line_rise(widths, power_density, conductivity):
    """theta of the 3-point heat operator between theta = 0 x faces (half cells 2 k / w), per unit y length."""
    faces = np.empty(widths.size + 1)
    faces[0] = 2.0 * conductivity / widths[0]
    faces[1:-1] = 2.0 * conductivity / (widths[:-1] + widths[1:])
    faces[-1] = 2.0 * conductivity / widths[-1]
    matrix = np.diag(faces[:-1] + faces[1:]) - np.diag(faces[1:-1], 1) - np.diag(faces[1:-1], -1)
    return np.linalg.solve(matrix, power_density * widths)


def _sheet_force_density(edges):
    """f_x = -J B_y at the centres of the sheet between A = 0 on x_lo and a symmetry x_hi: face H_y = -(current to the
    right of the face) exactly, B_y = mu0 times its centre mean; total F_x = mu0 J^2 w^2 L_y / 2 per m of depth."""
    widths = np.diff(edges)
    density = np.where(_in_range(edges, *_BAND), _CURRENT_DENSITY, 0.0)
    right_current = np.concatenate((np.cumsum((density * widths)[::-1])[::-1], [0.0]))   # per unit y, at the faces
    flux_density = -MU0 * 0.5 * (right_current[:-1] + right_current[1:])
    band_width = float(np.sum(widths[density > 0]))
    return -density * flux_density, 0.5 * MU0 * (_CURRENT_DENSITY * band_width) ** 2 * _WIDTH


# ---- specs ----
def _induction_specs():
    """'run': the selftests_ac skin-slab model + heat (k = 1, T0 walls on x_lo / x_hi) and magnetic_heat_induction."""
    spec = _skin_specs()[_RUN]
    spec.title = 'Self-test: induction heating of the skin-depth slab (magnetic_fields_ac -> heat)'
    spec.materials = [MaterialSpec('copper', properties={'sigma_dc': _COPPER, 'k': _SLAB_CONDUCTIVITY},
                                   selection=['slab'])]
    spec.physics.append(PhysicsSpec(_HEAT, HEAT_TYPE_NAME, settings={'T0': _T0}, features=[
        FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': _T0})]))
    spec.couplings = [CouplingSpec('induction', INDUCTION_COUPLING_NAME, _MAGNETIC, _HEAT)]
    spec.derived_values.append(DerivedValueSpec('Q_total', 'Q', ['all'], 'integral'))
    return {_RUN: with_exact_cg(spec)}


def _lorentz_specs():
    """'run': J = 1e6 A/m^2 in the band, insulation x_lo, symmetry x_hi and y faces; a solid fixed on both x faces and
    magnetic_solid_lorentz_force."""
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: Lorentz force on a current sheet (magnetostatics -> solid_mechanics)',
        domain=DomainSpec(size_x=_LENGTH, size_y=_WIDTH), geometry=[_band_rectangle('band', *_BAND)],
        materials=[MaterialSpec('copper', properties={'youngs_modulus': _YOUNGS_MODULUS,
                                                      'poisson_ratio': _POISSON_RATIO}, selection=['all'])],
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_TYPE_NAME, features=[
                     FeatureSpec('wall', 'magnetic_insulation', ['x_lo']),
                     FeatureSpec('mirror', 'symmetry', ['x_hi', 'y_lo', 'y_hi']),
                     FeatureSpec('sheet', 'current_source', ['band'], {'density': _CURRENT_DENSITY})]),
                 PhysicsSpec(_SOLID, SOLID_TYPE_NAME, features=[
                     FeatureSpec('left', 'fixed', ['x_lo']), FeatureSpec('right', 'fixed', ['x_hi'])])],
        couplings=[CouplingSpec('lorentz', LORENTZ_COUPLING_NAME, _MAGNETIC, _SOLID)],
        mesh=MeshSpec(mode='uniform', dx=_CELL), study=StudySpec(type='stationary')))}


# ---- checks ----
def _plain_field(results, name):
    """float64 window values of a field of any physics (the 1D layout's shape)."""
    values = np.asarray(results.field(name).values, dtype=np.float64)
    if values.shape != (round(_LENGTH / _CELL), round(_WIDTH / _CELL)):
        raise ValueError(f'{name}: expected the 1D layout window, got {values.shape}')
    return values


def _check_induction(results_by_name):
    """Q = q_eddy + q_magnetic cellwise, Q against the exact skin loss, integral Q = eddy_loss, and T - T0 against
    the exact discrete Dirichlet line of the run's own Q."""
    results = results_by_name[_RUN]
    loss_density, edges, _ = _window_field(results, 'q_eddy')
    magnetic_loss_density, _, _ = _window_field(results, 'q_magnetic')
    heat_source = _plain_field(results, f'{_HEAT}.Q')
    rise = _plain_field(results, f'{_HEAT}.T_rise')
    widths = np.diff(edges)
    exact_loss = _skin_solution(edges)
    rise_closed = _dirichlet_line_rise(widths, heat_source.mean(axis=1), _SLAB_CONDUCTIVITY)
    rise_peak = float(np.max(rise_closed))
    eddy_loss = _derived_value(results, 'P')
    total_source = _derived_value(results, 'Q_total')
    errors = {'Q_transfer': max_relative_error(heat_source, loss_density + magnetic_loss_density,
                                               float(np.max(loss_density))),
              'Q_skin': max_relative_error(heat_source[1:], exact_loss['q_eddy'][:, None], float(np.max(exact_loss['q_eddy']))),
              'Q_integral': relative_error(total_source, eddy_loss),
              'T_rise': max_relative_error(rise, rise_closed[:, None], rise_peak)}
    return _report(float(np.max(rise)), rise_peak, errors, eddy_loss_W_per_m=eddy_loss,
                   Q_integral_W_per_m=total_source, T_rise_max_K=float(np.max(rise)),
                   drive_potential_T_m=_DRIVE_POTENTIAL, frequency_Hz=_SKIN_FREQUENCY, n_cells=list(rise.shape))


def _check_lorentz(results_by_name):
    """body_force_x = -J B_y (exact discrete B) and body_force_y = 0 in the solid; the support reactions balance the
    closed-form total force mu0 J^2 w^2 L_y / 2."""
    results = results_by_name[_RUN]
    flux_density, edges, _ = _window_field(results, 'B_y')
    force_x = _plain_field(results, f'{_SOLID}.body_force_x')
    force_y = _plain_field(results, f'{_SOLID}.body_force_y')
    force_closed, total_closed = _sheet_force_density(edges)
    widths = np.diff(edges)
    cell_height = _WIDTH / force_x.shape[1]
    total_load = float(np.sum(force_x * widths[:, None])) * cell_height
    reaction = sum(float(results.result(_SOLID, name)['F_x']) for name in ('left', 'right'))
    scale = float(np.max(np.abs(force_closed)))
    errors = {'body_force_x': max_relative_error(force_x, force_closed[:, None], scale),
              'body_force_y': max_relative_error(force_y, 0.0, scale),
              'total_force': relative_error(total_load, total_closed),
              'reaction_balance': relative_error(-reaction, total_closed)}
    return _report(total_load, total_closed, errors, unit='N/m', reaction_N_per_m=reaction,
                   B_y_min_T=float(np.min(flux_density)), n_cells=list(force_x.shape))


# ---- template ----
# induction_billet: aluminium billet r = 20 mm, 60 mm tall on the axis of a 20-turn stranded copper coil (r 28 .. 34 mm,
# 70 mm tall) at 1 kHz (skin depth in the billet ~2.7 mm), in an 80 x 120 mm (r, z) air box of magnetic insulation
# (the return flux is squeezed by the box: a template, not a calibrated heater). Transient axisymmetric heat over 20 s,
# T0 on the outer faces; the coil's own Joule loss is not coupled. Mesh: base 2 mm, billet 0.5 mm. Not run yet: the
# loss (~1 kW estimate) and the rise (~100 K) are unverified.
INDUCTION_BILLET_LABEL = 'Induction heating of a billet'
INDUCTION_BILLET_DESCRIPTION = (
    'Magnetic fields AC + heat, axisymmetric, time-dependent: a 20-turn coil carrying I = 300 A peak at f = 1 kHz '
    'heats an aluminium billet (r = 20 mm, 60 mm tall) by eddy currents in its skin (delta ~2.7 mm); the eddy loss '
    'feeds transient heat for 20 s through magnetic_heat_induction. Probes record the surface and core temperature; '
    'derived the eddy loss P [W], the coil resistance and inductance and the hottest billet temperature.')
_BILLET = (20e-3, 30e-3, 90e-3)            # r_max, z_min, z_max [m]
_COIL = (28e-3, 34e-3, 25e-3, 95e-3)        # r_min, r_max, z_min, z_max [m]
_BILLET_DOMAIN = (80e-3, 120e-3)            # (r, z) [m]
_ALUMINIUM = {'sigma_dc': 3.5e7, 'k': 237.0, 'rho': 2700.0, 'cp': 900.0}
_AIR = {'k': 0.026, 'rho': 1.2, 'cp': 1005.0}
_COIL_WIRE = {'sigma_dc': 5.8e7, 'k': 400.0, 'rho': 8960.0, 'cp': 385.0}


def induction_billet_spec():
    """Template 'induction_billet' (axisymmetric magnetic_fields_ac + transient heat, time_dependent)."""
    r_billet, z_bottom, z_top = _BILLET
    r_in, r_out, z_coil_bottom, z_coil_top = _COIL
    z_middle = 0.5 * (z_bottom + z_top)

    def rectangle(name, x0, y0, x1, y1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    def point(name, x, y):
        return GeometrySpec(name, 'point', {'x': x, 'y': y})

    return ModelSpec(
        title=INDUCTION_BILLET_LABEL, description=INDUCTION_BILLET_DESCRIPTION,
        parameters=[ParameterSpec('f', 1000.0, 'coil frequency [Hz]'),
                    ParameterSpec('I', 300.0, 'coil current per turn, peak [A]')],
        domain=DomainSpec(size_x=_BILLET_DOMAIN[0], size_y=_BILLET_DOMAIN[1]),
        geometry=[rectangle('billet', 0.0, z_bottom, r_billet, z_top),
                  rectangle('coil', r_in, z_coil_bottom, r_out, z_coil_top),
                  point('surface_point', r_billet - 0.25e-3, z_middle), point('core_point', 0.25e-3, z_middle)],
        materials=[MaterialSpec('air', properties=dict(_AIR), selection=['all']),
                   MaterialSpec('aluminium', properties=dict(_ALUMINIUM), selection=['billet']),
                   MaterialSpec('coil_copper', properties=dict(_COIL_WIRE), selection=['coil'])],
        physics=[PhysicsSpec(_MAGNETIC, MAGNETIC_AC_TYPE_NAME, settings={'geometry': 'axisymmetric', 'frequency': 'f'},
                             features=[FeatureSpec('coil', 'coil', ['coil'],
                                                   {'conductor': 'stranded', 'current': 'I', 'turns': 20,
                                                    'fill_factor': 0.6})]),
                 PhysicsSpec(_HEAT, HEAT_TYPE_NAME,
                             settings={'geometry': 'axisymmetric', 'time_mode': 'transient', 'time_step': 0.5,
                                       'T0': _T0},
                             features=[FeatureSpec('ambient', 'temperature', ['x_hi', 'y_lo', 'y_hi'],
                                                   {'temperature': _T0}),
                                       FeatureSpec('surface', 'temperature_probe', ['surface_point']),
                                       FeatureSpec('core', 'temperature_probe', ['core_point'])])],
        couplings=[CouplingSpec('induction', INDUCTION_COUPLING_NAME, _MAGNETIC, _HEAT)],
        mesh=MeshSpec(mode='nonuniform', dx=2e-3, refinements=[RefinementSpec('billet', 0.5e-3, 5e-3)]),
        study=StudySpec(type='time_dependent', settings={'duration': 20.0}),
        derived_values=[DerivedValueSpec('P', f'{_MAGNETIC}_eddy_loss', [], 'max'),
                        DerivedValueSpec('R_coil', f'{_MAGNETIC}_coil_resistance', [], 'max'),
                        DerivedValueSpec('L_coil', f'{_MAGNETIC}_coil_inductance', [], 'max'),
                        DerivedValueSpec('T_billet_max', 'T', ['billet'], 'max')])


# ---- cases ----
INDUCTION_CASES = (
    SelfTestCase(
        name='magnet_induction_heating_strip', label='Magnetic AC -> heat: induction heating of a copper slab',
        description=('The skin-slab model of magnet_ac_cpu_gpu (copper window of 10 um cells, A = '
                     f'{_DRIVE_POTENTIAL:g} T m held on the first column, f = {_SKIN_FREQUENCY:g} Hz, symmetry y '
                     f'faces) plus heat with k = {_SLAB_CONDUCTIVITY:g} W/(m K) (a ~1 K rise), T0 walls on x_lo / x_hi '
                     f'(float64 CG to {EXACT_CG_TOLERANCE:g}) and {INDUCTION_COUPLING_NAME}, stationary. '
                     'Exact: heat Q = q_eddy + q_magnetic cellwise (a float32 copy), Q against the exact discrete skin '
                     'loss sigma omega^2 |A|^2 / 2, the integral of Q = the eddy_loss scalar, and T - T0 against the '
                     '3-point Dirichlet line solved densely for the run\'s own y-mean Q. error = max of the four; '
                     f'measured = max T - T0 [K]; tolerance {_TOLERANCE:g} (CG residual and float32 storage). Pins the '
                     "transfer, its 'sum' slot and the units of the loss density. Variant cpu: the same run on the "
                     'cpu backend, same tolerance.'),
        specs=_induction_specs, check=_check_induction, variants=_CPU_VARIANTS),
)

LORENTZ_CASES = (
    SelfTestCase(
        name='magnet_lorentz_force_sheet', label='Magnetostatics -> solid: Lorentz force on a current sheet',
        description=(f'J_z = {_CURRENT_DENSITY:g} A/m^2 in the band x in [{_BAND[0] * 1e3:g}, {_BAND[1] * 1e3:g}] mm '
                     f'of the 1 x 0.1 mm window ({_CELL * 1e6:g} um cells), magnetic insulation on x_lo, symmetry on '
                     'x_hi and the y faces: face H_y = -(current right of the face) exactly, B_y = mu0 times its centre '
                     'mean (<= 0 in the band). A solid (E = 1.1e11 Pa, nu = 0.34) everywhere, fixed on both x faces, '
                     f'float64 CG to {EXACT_CG_TOLERANCE:g}, receives {LORENTZ_COUPLING_NAME}. Exact: '
                     'body_force_x = -J B_y, body_force_y = 0, the total force mu0 J^2 w^2 L_y / 2 per m of depth (the '
                     'trapezoid rule is exact on the linear H), and the two support reactions balance it. error = max '
                     f'of the four; measured = the total load [N/m]; tolerance {_TOLERANCE:g}. Pins the sign of J x B, '
                     'the transfer of the force pair and its reach into the solid load. Variant cpu: the same run on '
                     'the cpu backend, same tolerance.'),
        specs=_lorentz_specs, check=_check_lorentz, variants=_CPU_VARIANTS),
)

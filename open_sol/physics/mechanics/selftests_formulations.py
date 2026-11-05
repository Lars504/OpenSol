"""Self-tests of the solid_mechanics out-of-plane conditions and geometry (symmetry plan 2.4): plane stress (uniform
plate, bilayer), generalized plane strain (two layers side by side, uniform and with the bending terms), axisymmetric
(r, z) operator identities (CPU), pressurised heated cylinder (exact), spinning shaft (convergence) and radial breathing
modes (Bessel roots)."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, max_relative_error, with_exact_cg, worst_error
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, StudySpec)
from .selftests import SOLID_TYPE_NAME, _elastic, _lame, _reaction, _rectangle, _window_fields

_SOLID = 'solid'                                   # = selftests._SOLID (the helpers read this instance)
_RUN = 'run'
_TOLERANCE = 1e-6                                  # exact-in-Q1 cases (float64 CG, float32 fields)
_CELL = 1e-6                                       # m
_PLANE_STRESS = {'out_of_plane': 'plane_stress'}
_AXISYMMETRIC = {'geometry': 'axisymmetric'}
_BUDGET_TEXT = (f'tolerance {_TOLERANCE:g}: the exact field lies in the bilinear element space, so only the CG residual '
                f'(float64, {EXACT_CG_TOLERANCE:g}) and the float32 field storage remain')


def _model(title, size, geometry, materials, settings, features, mesh=None, study=None):
    """Exact-CG ModelSpec of one solid physics."""
    return with_exact_cg(ModelSpec(
        title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]), geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_SOLID, SOLID_TYPE_NAME, settings=dict(settings), features=features)],
        mesh=mesh or MeshSpec(mode='uniform', dx=_CELL), study=study or StudySpec(type='stationary')))


def _centres(edges):
    return 0.5 * (edges[:-1] + edges[1:])


# ---- solid_plane_stress: 'plate' (uniaxial traction + free anisotropic heating) and 'bilayer' (two layers pulled) ----
# plate: 40 x 20 um, E 100 GPa, nu 0.3, alpha_xx / _yy / _zz 1e-5 / 2e-5 / 0.5e-5, temperature_change 100 K, rollers on
# x_lo / y_lo, traction t_x = T on x_hi. Plane stress: sigma_xx = T, sigma_yy = sigma_zz = sigma_xy = 0, eps_xx = T / E +
# alpha_xx dT, eps_yy = -nu T / E + alpha_yy dT, eps_zz = -nu T / E + alpha_zz dT (elastic_strain_zz = -nu T / E); linear
# u, exact in Q1 (the reduced stiffness and thermal stress are the exact static condensation of sigma_zz = 0). bilayer:
# selftests' strip (layers in y, u_x = d on x_hi): eps_yy = -nu eps_xx, sigma_xx = E eps_xx per layer, pull F = sum E
# eps_xx h.
_PLATE_SIZE = (40e-6, 20e-6)
_PLATE_ELASTIC = (100e9, 0.3)
_PLATE_ALPHA = (1e-5, 2e-5, 0.5e-5)
_PLATE_RISE = 100.0                                # K
_PLATE_TRACTION = 1e8                              # Pa on x_hi
_PLATE_THICKNESS = 2e-6                            # m: the plate reactions are in N for this thickness
_BILAYER_SIZE = (100e-6, 20e-6)
_BILAYER_INTERFACE_Y = 8e-6
_BILAYER_LAYERS = ((100e9, 0.3), (50e9, 0.2))
_BILAYER_PULL = 1e-7                               # m


def _plane_stress_specs():
    alpha_xx, alpha_yy, alpha_zz = _PLATE_ALPHA
    plate = _model('Self-test: plane-stress plate under traction and heating (solid mechanics)', _PLATE_SIZE, [],
                   [MaterialSpec('plate', properties=_elastic(*_PLATE_ELASTIC, alpha_xx=alpha_xx, alpha_yy=alpha_yy,
                                                              alpha_zz=alpha_zz), selection=['all'])],
                   {**_PLANE_STRESS, 'thickness': _PLATE_THICKNESS},
                   [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
                    FeatureSpec('pull', 'boundary_load', ['x_hi'], {'t_x': _PLATE_TRACTION, 't_y': 0.0}),
                    FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _PLATE_RISE})])
    length, height = _BILAYER_SIZE
    (youngs_a, nu_a), (youngs_b, nu_b) = _BILAYER_LAYERS
    bilayer = _model('Self-test: plane-stress bilayer under tension (solid mechanics)', _BILAYER_SIZE,
                     [_rectangle('lower', 0.0, 0.0, length, _BILAYER_INTERFACE_Y),
                      _rectangle('upper', 0.0, _BILAYER_INTERFACE_Y, length, height)],
                     [MaterialSpec('layer_a', properties=_elastic(youngs_a, nu_a), selection=['lower']),
                      MaterialSpec('layer_b', properties=_elastic(youngs_b, nu_b), selection=['upper'])],
                     _PLANE_STRESS,
                     [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
                      FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _BILAYER_PULL})])
    return {'plate': plate, 'bilayer': bilayer}


def _check_plane_stress(results_by_name):
    """Both runs against the plane-stress closed forms (fields and reactions)."""
    errors = {}
    youngs, nu = _PLATE_ELASTIC
    alpha_xx, alpha_yy, alpha_zz = _PLATE_ALPHA
    names = ('u_x', 'u_y', 'stress_xx', 'stress_yy', 'stress_xy', 'stress_zz', 'elastic_strain_zz')
    plate = results_by_name['plate']
    fields, x_edges, y_edges = _window_fields(plate, names)
    strain_xx = _PLATE_TRACTION / youngs + alpha_xx * _PLATE_RISE
    strain_yy = -nu * _PLATE_TRACTION / youngs + alpha_yy * _PLATE_RISE
    elastic_zz = -nu * _PLATE_TRACTION / youngs
    height = _PLATE_SIZE[1]
    expected = {'u_x': strain_xx * _centres(x_edges)[:, None], 'u_y': strain_yy * _centres(y_edges)[None, :],
                'stress_xx': _PLATE_TRACTION, 'stress_yy': 0.0, 'stress_xy': 0.0, 'stress_zz': 0.0,
                'elastic_strain_zz': elastic_zz}
    scales = {'u_x': strain_xx * _PLATE_SIZE[0], 'u_y': abs(strain_yy) * height, 'elastic_strain_zz': abs(elastic_zz)}
    for name in names:
        errors[f'plate.{name}'] = max_relative_error(fields[name], expected[name], scales.get(name, _PLATE_TRACTION))
    left_x, _ = _reaction(plate, 'left')
    _, bottom_y = _reaction(plate, 'bottom')
    force = _PLATE_TRACTION * height * _PLATE_THICKNESS
    errors['plate.F_left'] = abs(left_x + force) / force
    errors['plate.F_bottom'] = abs(bottom_y) / force

    bilayer = results_by_name['bilayer']
    fields, x_edges, y_edges = _window_fields(bilayer, ('u_x', 'u_y', 'stress_xx', 'stress_yy', 'stress_zz'))
    length, total_height = _BILAYER_SIZE
    pull_strain = _BILAYER_PULL / length
    y_centres = _centres(y_edges)
    lower = y_centres < _BILAYER_INTERFACE_Y
    (youngs_a, nu_a), (youngs_b, nu_b) = _BILAYER_LAYERS
    u_y = np.where(lower, -nu_a * pull_strain * y_centres,
                   -nu_a * pull_strain * _BILAYER_INTERFACE_Y - nu_b * pull_strain * (y_centres - _BILAYER_INTERFACE_Y))
    stress_xx = np.where(lower, youngs_a, youngs_b) * pull_strain
    stress_scale = youngs_a * pull_strain
    pull_force = pull_strain * (youngs_a * _BILAYER_INTERFACE_Y + youngs_b * (total_height - _BILAYER_INTERFACE_Y))
    pull_x, _ = _reaction(bilayer, 'pull')
    errors.update({'bilayer.u_x': max_relative_error(fields['u_x'], pull_strain * _centres(x_edges)[:, None],
                                                     _BILAYER_PULL),
                   'bilayer.u_y': max_relative_error(fields['u_y'], u_y[None, :], float(np.max(np.abs(u_y)))),
                   'bilayer.stress_xx': max_relative_error(fields['stress_xx'], stress_xx[None, :], stress_scale),
                   'bilayer.stress_yy': max_relative_error(fields['stress_yy'], 0.0, stress_scale),
                   'bilayer.stress_zz': max_relative_error(fields['stress_zz'], 0.0, stress_scale),
                   'bilayer.F_pull': abs(pull_x - pull_force) / pull_force})
    return {'measured': pull_x, 'expected': pull_force, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'plate_strain_xx': strain_xx, 'plate_strain_yy': strain_yy,
                        'plate_elastic_strain_zz': elastic_zz, 'bilayer_F_pull_N_per_m': pull_x}}


# ---- solid_generalized_plane_strain: two layers side by side along x, free ends along z ----
# 40 x 20 um, layer a (x < 16 um: E 100 GPa, nu 0.3, alpha_xx / _yy / _zz 1e-5 / 2e-5 / 0.5e-5), layer b (E 50 GPa, nu
# 0.2, alpha 3e-5); rollers on x_lo and on y_lo / y_hi (eps_yy = 0), x_hi free, temperature_change 100 K, axial_force F,
# bending 'none' (uniform eps_zz = e; the bending terms: solid_generalized_plane_strain_bending).
# 1D in x: sigma_xx = 0 gives eps_xx = (beta_x dT - lambda e) / L per layer, sigma_zz = lambda eps_xx + L e - beta_z dT,
# sigma_yy = lambda (eps_xx + e) - beta_y dT (beta = the 3D C : alpha); sum w_i H sigma_zz,i = F fixes e. u_x piecewise
# linear (kink on the interface cell edge), u_y = 0: exact in Q1, and the axial-force integral is exact too.
_GPS_SIZE = (40e-6, 20e-6)
_GPS_INTERFACE_X = 16e-6
_GPS_LAYERS = (((100e9, 0.3), (1e-5, 2e-5, 0.5e-5)), ((50e9, 0.2), (3e-5, 3e-5, 3e-5)))   # (E, nu), alpha_xx/_yy/_zz
_GPS_RISE = 100.0                                  # K
_GPS_AXIAL_FORCE = 2e-2                            # N on the drawn 8e-10 m^2 cross-section (~2.5e7 Pa)


def _gps_specs():
    length, height = _GPS_SIZE
    materials = []
    for name, region, ((youngs, nu), (alpha_xx, alpha_yy, alpha_zz)) in zip(('layer_a', 'layer_b'),
                                                                             ('left_part', 'right_part'), _GPS_LAYERS):
        materials.append(MaterialSpec(name, properties=_elastic(youngs, nu, alpha_xx=alpha_xx, alpha_yy=alpha_yy,
                                                                alpha_zz=alpha_zz), selection=[region]))
    return {_RUN: _model('Self-test: generalized plane strain of two heated layers (solid mechanics)', _GPS_SIZE,
                         [_rectangle('left_part', 0.0, 0.0, _GPS_INTERFACE_X, height),
                          _rectangle('right_part', _GPS_INTERFACE_X, 0.0, length, height)],
                         materials, {'out_of_plane': 'generalized_plane_strain', 'axial_force': _GPS_AXIAL_FORCE,
                                     'bending': 'none'},
                         [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
                          FeatureSpec('top', 'roller', ['y_hi']),
                          FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _GPS_RISE})])}


def _gps_closed_forms():
    """(e, per layer (eps_xx, sigma_yy, sigma_zz, alpha_zz), layer widths)."""
    length, height = _GPS_SIZE
    widths = (_GPS_INTERFACE_X, length - _GPS_INTERFACE_X)
    numerator, denominator, layers = _GPS_AXIAL_FORCE / height, 0.0, []
    for width, ((youngs, nu), (alpha_xx, alpha_yy, alpha_zz)) in zip(widths, _GPS_LAYERS):
        lame_lambda, mu = _lame(youngs, nu)
        longitudinal = lame_lambda + 2.0 * mu
        beta_x = longitudinal * alpha_xx + lame_lambda * (alpha_yy + alpha_zz)
        beta_z = longitudinal * alpha_zz + lame_lambda * (alpha_xx + alpha_yy)
        numerator += width * _GPS_RISE * (beta_z - lame_lambda * beta_x / longitudinal)
        denominator += width * (longitudinal - lame_lambda * lame_lambda / longitudinal)
    axial_strain = numerator / denominator
    for (youngs, nu), (alpha_xx, alpha_yy, alpha_zz) in _GPS_LAYERS:
        lame_lambda, mu = _lame(youngs, nu)
        longitudinal = lame_lambda + 2.0 * mu
        beta_x = longitudinal * alpha_xx + lame_lambda * (alpha_yy + alpha_zz)
        beta_y = longitudinal * alpha_yy + lame_lambda * (alpha_xx + alpha_zz)
        beta_z = longitudinal * alpha_zz + lame_lambda * (alpha_xx + alpha_yy)
        strain_xx = (beta_x * _GPS_RISE - lame_lambda * axial_strain) / longitudinal
        layers.append((strain_xx, lame_lambda * (strain_xx + axial_strain) - beta_y * _GPS_RISE,
                       lame_lambda * strain_xx + longitudinal * axial_strain - beta_z * _GPS_RISE, alpha_zz))
    return axial_strain, layers, widths


def _check_gps(results_by_name):
    """u_x, u_y, stresses, elastic_strain_zz = e - alpha_zz dT and the roller reactions against the closed form."""
    results = results_by_name[_RUN]
    names = ('u_x', 'u_y', 'stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'elastic_strain_zz')
    fields, x_edges, _ = _window_fields(results, names)
    axial_strain, layers, widths = _gps_closed_forms()
    x_centres = _centres(x_edges)
    left = x_centres < _GPS_INTERFACE_X
    (strain_a, stress_yy_a, stress_zz_a, alpha_a), (strain_b, stress_yy_b, stress_zz_b, alpha_b) = layers
    u_x = np.where(left, strain_a * x_centres, strain_a * _GPS_INTERFACE_X + strain_b * (x_centres - _GPS_INTERFACE_X))
    stress_scale = max(abs(value) for value in (stress_yy_a, stress_yy_b, stress_zz_a, stress_zz_b))
    expected = {'u_x': u_x[:, None], 'u_y': 0.0, 'stress_xx': 0.0, 'stress_xy': 0.0,
                'stress_yy': np.where(left, stress_yy_a, stress_yy_b)[:, None],
                'stress_zz': np.where(left, stress_zz_a, stress_zz_b)[:, None],
                'elastic_strain_zz': (axial_strain - np.where(left, alpha_a, alpha_b) * _GPS_RISE)[:, None]}
    displacement_scale = float(np.max(np.abs(u_x)))
    scales = {'u_x': displacement_scale, 'u_y': displacement_scale, 'elastic_strain_zz': abs(axial_strain)}
    errors = {name: max_relative_error(fields[name], expected[name], scales.get(name, stress_scale)) for name in names}
    top_force = stress_yy_a * widths[0] + stress_yy_b * widths[1]
    _, top_y = _reaction(results, 'top')
    _, bottom_y = _reaction(results, 'bottom')
    left_x, _ = _reaction(results, 'left')
    force_scale = stress_scale * _GPS_SIZE[0]
    errors.update({'F_top': abs(top_y - top_force) / force_scale, 'F_bottom': abs(bottom_y + top_force) / force_scale,
                   'F_left': abs(left_x) / (stress_scale * _GPS_SIZE[1])})
    measured = float(np.mean(fields['elastic_strain_zz'][~left] + alpha_b * _GPS_RISE))
    return {'measured': measured, 'expected': axial_strain, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'axial_strain': axial_strain, 'F_top_N_per_m': top_y,
                        'F_top_expected': top_force, 'layers_eps_xx_sigma_yy_sigma_zz': [list(layer[:3])
                                                                                         for layer in layers]}}


# ---- solid_generalized_plane_strain_bending: the GPS layers with eps_zz = e0 + e1 x + e2 y ('along_x', 'along_y') ----
# The two layers of solid_generalized_plane_strain stacked along s = x (run along_x) or s = y (along_y; the section
# transposed), bending 'xy', axial_force F, first moment int s sigma_zz dA = Q (M_y = -Q, resp. M_x = Q) and the other
# moment = F times the transverse mid-line (so that term is 0). Rollers on the s_lo face and both transverse faces, s_hi
# free: u is 1D in s, u_t = 0. Discrete-exact per element: mean sigma_ss = 0 gives eps_ss = (beta_s dT - lambda eps_zz,c)
# / L with eps_zz,c = e0 + e1 s_c; sigma_zz = lambda eps_ss + L (e0 + e1 s) - beta_z dT is linear in s, so int sigma_zz
# ds = w sigma_zz,c and int s sigma_zz ds = s_c w sigma_zz,c + L e1 w^3 / 12; these two conditions fix (e0, e1).
_BEND_LENGTH = 40e-6                               # m along the layered axis s
_BEND_WIDTH = 20e-6                                # m across it
_BEND_INTERFACE = 16e-6                            # layer a below
_BEND_AXIAL_FORCE = 2e-2                           # N
_BEND_FIRST_MOMENT = 1e-7                          # N m: int s sigma_zz dA
_BEND_RUNS = {'along_x': 0, 'along_y': 1}


def _bending_spec(along):
    length, width = _BEND_LENGTH, _BEND_WIDTH
    if along == 0:
        size, parts = (length, width), [_rectangle('part_a', 0.0, 0.0, _BEND_INTERFACE, width),
                                        _rectangle('part_b', _BEND_INTERFACE, 0.0, length, width)]
        supports = [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
                    FeatureSpec('top', 'roller', ['y_hi'])]
        moments = {'bending_moment_x': 0.5 * width * _BEND_AXIAL_FORCE, 'bending_moment_y': -_BEND_FIRST_MOMENT}
    else:
        size, parts = (width, length), [_rectangle('part_a', 0.0, 0.0, width, _BEND_INTERFACE),
                                        _rectangle('part_b', 0.0, _BEND_INTERFACE, width, length)]
        supports = [FeatureSpec('bottom', 'roller', ['y_lo']), FeatureSpec('left', 'roller', ['x_lo']),
                    FeatureSpec('right', 'roller', ['x_hi'])]
        moments = {'bending_moment_x': _BEND_FIRST_MOMENT, 'bending_moment_y': -0.5 * width * _BEND_AXIAL_FORCE}
    materials = [MaterialSpec(name, properties=_elastic(youngs, nu, alpha_xx=alpha_xx, alpha_yy=alpha_yy,
                                                        alpha_zz=alpha_zz), selection=[region])
                 for name, region, ((youngs, nu), (alpha_xx, alpha_yy, alpha_zz))
                 in zip(('layer_a', 'layer_b'), ('part_a', 'part_b'), _GPS_LAYERS)]
    return _model(f"Self-test: generalized plane strain with bending, layers along {'xy'[along]} (solid mechanics)",
                  size, parts, materials,
                  {'out_of_plane': 'generalized_plane_strain', 'axial_force': _BEND_AXIAL_FORCE, 'bending': 'xy',
                   **moments},
                  supports + [FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _GPS_RISE})])


def _bending_closed_form(edges, along):
    """Discrete-exact 1D solution along the layered axis s of edges: per element eps_ss, transverse stress, sigma_zz,
    eps_zz and alpha_zz, the centre displacement u_s and (e0, e1) of eps_zz = e0 + e1 s."""
    widths, centres = np.diff(edges), _centres(edges)
    lower = centres < _BEND_INTERFACE
    columns = []   # per layer (lambda, L, beta_s, beta_t, beta_z, alpha_zz)
    for (youngs, nu), (alpha_xx, alpha_yy, alpha_zz) in _GPS_LAYERS:
        lame_lambda, mu = _lame(youngs, nu)
        longitudinal = lame_lambda + 2.0 * mu
        beta_x = longitudinal * alpha_xx + lame_lambda * (alpha_yy + alpha_zz)
        beta_y = longitudinal * alpha_yy + lame_lambda * (alpha_xx + alpha_zz)
        beta_z = longitudinal * alpha_zz + lame_lambda * (alpha_xx + alpha_yy)
        beta_s, beta_t = (beta_x, beta_y) if along == 0 else (beta_y, beta_x)
        columns.append((lame_lambda, longitudinal, beta_s, beta_t, beta_z, alpha_zz))
    lame_lambda, longitudinal, beta_s, beta_t, beta_z, alpha_zz = (np.where(lower, a, b) for a, b in zip(*columns))
    reduced = longitudinal - lame_lambda ** 2 / longitudinal
    offset = (beta_z - lame_lambda * beta_s / longitudinal) * _GPS_RISE
    matrix = np.array([[np.sum(widths * reduced), np.sum(widths * centres * reduced)],
                       [np.sum(widths * centres * reduced),
                        np.sum(widths * centres ** 2 * reduced + longitudinal * widths ** 3 / 12.0)]])
    right_hand = np.array([_BEND_AXIAL_FORCE / _BEND_WIDTH + np.sum(widths * offset),
                           _BEND_FIRST_MOMENT / _BEND_WIDTH + np.sum(widths * centres * offset)])
    e0, e1 = np.linalg.solve(matrix, right_hand)
    strain_zz = e0 + e1 * centres
    strain_ss = (beta_s * _GPS_RISE - lame_lambda * strain_zz) / longitudinal
    nodes = np.concatenate(([0.0], np.cumsum(strain_ss * widths)))
    return {'strain_ss': strain_ss, 'stress_t': lame_lambda * (strain_ss + strain_zz) - beta_t * _GPS_RISE,
            'stress_zz': lame_lambda * strain_ss + longitudinal * strain_zz - beta_z * _GPS_RISE,
            'strain_zz': strain_zz, 'alpha_zz': alpha_zz, 'u_s': 0.5 * (nodes[:-1] + nodes[1:]), 'terms': (e0, e1)}


def _check_bending(results_by_name):
    """Both runs: u, stresses, elastic_strain_zz = eps_zz - alpha_zz dT and the transverse roller reactions against
    the discrete-exact 1D solution."""
    names = ('u_x', 'u_y', 'stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'elastic_strain_zz')
    errors, terms = {}, {}
    for run, along in _BEND_RUNS.items():
        results = results_by_name[run]
        fields, x_edges, y_edges = _window_fields(results, names)
        edges = (x_edges, y_edges)[along]
        exact = _bending_closed_form(edges, along)
        terms[run] = exact['terms']
        profile_shape = (-1, 1) if along == 0 else (1, -1)   # 1D profile along s onto the (nx, ny) fields
        shaped = {name: exact[name].reshape(profile_shape) for name in ('u_s', 'stress_t', 'stress_zz')}
        stress_scale = float(np.max(np.abs(np.concatenate((exact['stress_t'], exact['stress_zz'])))))
        displacement_scale = float(np.max(np.abs(exact['u_s'])))
        along_name, across_name = ('x', 'y') if along == 0 else ('y', 'x')
        elastic_zz = exact['strain_zz'] - exact['alpha_zz'] * _GPS_RISE
        expected = {f'u_{along_name}': shaped['u_s'], f'u_{across_name}': 0.0,
                    f'stress_{along_name * 2}': 0.0, f'stress_{across_name * 2}': shaped['stress_t'],
                    'stress_zz': shaped['stress_zz'], 'stress_xy': 0.0,
                    'elastic_strain_zz': elastic_zz.reshape(profile_shape)}
        scales = {'u_x': displacement_scale, 'u_y': displacement_scale,
                  'elastic_strain_zz': float(np.max(np.abs(elastic_zz)))}
        for name in names:
            errors[f'{run}.{name}'] = max_relative_error(fields[name], expected[name], scales.get(name, stress_scale))
        transverse_force = float(np.sum(np.diff(edges) * exact['stress_t']))
        force_scale = stress_scale * _BEND_LENGTH
        held_face, opposite_face, end_face = ('top', 'bottom', 'left') if along == 0 else ('right', 'left', 'bottom')
        held = _reaction(results, held_face)[1 - along]
        opposite = _reaction(results, opposite_face)[1 - along]
        end = _reaction(results, end_face)[along]
        errors.update({f'{run}.F_{held_face}': abs(held - transverse_force) / force_scale,
                       f'{run}.F_{opposite_face}': abs(opposite + transverse_force) / force_scale,
                       f'{run}.F_{end_face}': abs(end) / (stress_scale * _BEND_WIDTH)})
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'e0_e1_expected': {run: list(values) for run, values in terms.items()}}}


# ---- solid_axisymmetric_operator (CPU): identities of the Gauss (r, z) stiffness and the ring mass ----
# Graded radial / axial widths, random element Lame pairs: the planar Gauss stencil equals the closed-form one, the (r,
# z) operator is symmetric with a z translation in its null space; for a uniform material the uniform-strain field u =
# (c r, d z) gives K u = the consistent tractions of sigma_rr = sigma_tt, sigma_zz on the outer faces (0 on the axis and
# inside: the integrands are polynomials the 2-point rule integrates exactly); the ring mass sums to the body's mass and
# u_r = r gives u^T M u = sum rho 2 pi b (r_hi^4 - r_lo^4) / 4 (cubic in r: exact).
_OPERATOR_DX = (0.3, 0.5, 0.8, 1.0, 1.4, 2.0)      # radial widths (arbitrary unit)
_OPERATOR_DY = (0.7, 1.0, 1.3, 0.9)
_OPERATOR_TOLERANCE = 1e-11                         # float64 host round-off relative to the operator scale
_PATCH_STRAINS = (2e-3, -1e-3)                      # c = eps_rr = eps_tt, d = eps_zz


def _operator_check(_results_by_name):
    from scipy import sparse as host_sparse
    from .system import ElasticPattern, axisymmetric_mass_stencil, gauss_stiffness_stencil, stiffness_stencil
    dx, dy = np.asarray(_OPERATOR_DX), np.asarray(_OPERATOR_DY)
    n_x, n_y = dx.size, dy.size
    r_nodes = np.concatenate(([0.0], np.cumsum(dx)))
    z_nodes = np.concatenate(([0.0], np.cumsum(dy)))
    rng = np.random.default_rng(24)
    lame_lambda = rng.uniform(0.5, 2.0, (n_x, n_y))
    mu = rng.uniform(0.5, 2.0, (n_x, n_y))
    longitudinal = lame_lambda + 2.0 * mu
    pattern = ElasticPattern(n_x + 1, n_y + 1)
    errors = {}

    closed = stiffness_stencil(longitudinal, lame_lambda, longitudinal, mu, dx, dy, np.float64)
    gauss = gauss_stiffness_stencil(longitudinal, lame_lambda, longitudinal, mu, dx, dy, np.float64)
    errors['planar Gauss = closed form'] = float(np.max(np.abs(gauss - closed)) / np.max(np.abs(closed)))

    def ring_operator(lambda_values, mu_values):
        longitudinal_values = lambda_values + 2.0 * mu_values
        stencil = gauss_stiffness_stencil(longitudinal_values, lambda_values, longitudinal_values, mu_values, dx, dy,
                                          np.float64, element_r_lo=r_nodes[:-1],
                                          hoop=(lambda_values, lambda_values, longitudinal_values))
        return pattern.csr(stencil, host_sparse)

    operator = ring_operator(lame_lambda, mu)
    scale = float(np.max(np.abs(operator.data)))
    errors['symmetry'] = float(np.max(np.abs((operator - operator.T).data), initial=0.0)) / scale
    translation = np.zeros((n_x + 1, n_y + 1, 2))
    translation[..., 1] = 1.0
    errors['z translation'] = float(np.max(np.abs(operator @ translation.ravel()))) / scale

    lambda_0, mu_0 = 1.3, 0.8
    uniform = ring_operator(np.full((n_x, n_y), lambda_0), np.full((n_x, n_y), mu_0))
    strain_r, strain_z = _PATCH_STRAINS
    patch = np.zeros((n_x + 1, n_y + 1, 2))
    patch[..., 0] = strain_r * r_nodes[:, None]
    patch[..., 1] = strain_z * z_nodes[None, :]
    stress_rr = (lambda_0 + 2.0 * mu_0) * strain_r + lambda_0 * (strain_r + strain_z)   # = sigma_tt
    stress_zz = 2.0 * lambda_0 * strain_r + (lambda_0 + 2.0 * mu_0) * strain_z
    expected = np.zeros((n_x + 1, n_y + 1, 2))
    tributary_z = np.zeros(n_y + 1)
    tributary_z[:-1] += 0.5 * dy
    tributary_z[1:] += 0.5 * dy
    expected[-1, :, 0] += stress_rr * 2.0 * math.pi * r_nodes[-1] * tributary_z
    ring = np.zeros(n_x + 1)   # integral of N 2 pi r dr per node
    ring[:-1] += 2.0 * math.pi * dx * (2.0 * r_nodes[:-1] + r_nodes[1:]) / 6.0
    ring[1:] += 2.0 * math.pi * dx * (r_nodes[:-1] + 2.0 * r_nodes[1:]) / 6.0
    expected[:, 0, 1] -= stress_zz * ring
    expected[:, -1, 1] += stress_zz * ring
    force_scale = max(abs(stress_rr), abs(stress_zz)) * float(np.max(ring))
    errors['uniform strain = consistent tractions'] = float(
        np.max(np.abs((uniform @ patch.ravel()).reshape(expected.shape) - expected))) / force_scale

    density = rng.uniform(1.0, 3.0, (n_x, n_y))
    mass = pattern.csr(axisymmetric_mass_stencil(density, dx, dy, r_nodes[:-1], np.float64), host_sparse)
    ring_volumes = math.pi * (r_nodes[1:] ** 2 - r_nodes[:-1] ** 2)[:, None] * dy[None, :]
    total = float(np.sum(density * ring_volumes))
    unit_r = np.zeros((n_x + 1, n_y + 1, 2))
    unit_r[..., 0] = 1.0
    errors['mass total'] = abs(float(unit_r.ravel() @ (mass @ unit_r.ravel())) - total) / total
    radial = np.zeros((n_x + 1, n_y + 1, 2))
    radial[..., 0] = r_nodes[:, None]
    moment = float(np.sum(density * 2.0 * math.pi * dy[None, :] * (r_nodes[1:] ** 4 - r_nodes[:-1] ** 4)[:, None] / 4.0))
    errors['mass r^2 moment'] = abs(float(radial.ravel() @ (mass @ radial.ravel())) - moment) / moment
    errors['mass symmetry'] = float(np.max(np.abs((mass - mass.T).data), initial=0.0)) / float(np.max(mass.data))
    worst = worst_error(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _OPERATOR_TOLERANCE,
            'details': {'errors': errors, 'n_cells': [n_x, n_y]}}


# ---- solid_axisymmetric_cylinder: pressurised, heated solid cylinder between z rollers ('graded') ----
# R 20 um x H 10 um, E 100 GPa, nu 0.3, alpha 1e-5, temperature_change 100 K, rollers on y_lo / y_hi (eps_zz = 0),
# traction t_x = -p on x_hi (r = R). Exact: u_r = c r with c = (beta dT - p) / (2 (lambda + mu)), beta = (3 lambda + 2
# mu) alpha; sigma_rr = sigma_tt = -p, sigma_zz = 2 lambda c - beta dT, u_z = 0; the roller reactions are +-sigma_zz pi
# R^2 (N, whole ring). u_r = c r lies in the element space and its stress is uniform, so the discrete solution is exact on
# any rectilinear mesh: one graded run (refinement line at r = R / 2, uniform 1 um cells outside the band, so the axis
# column and the outer face keep uniform cells).
_CYLINDER_SIZE = (20e-6, 10e-6)
_CYLINDER_ELASTIC = (100e9, 0.3)
_CYLINDER_ALPHA = 1e-5
_CYLINDER_RISE = 100.0
_CYLINDER_PRESSURE = 5e7                           # Pa
_CYLINDER_GRADED_CELL = 0.25e-6
_CYLINDER_GRADED_REACH = 6e-6


def _cylinder_spec():
    radius, height = _CYLINDER_SIZE
    geometry = [GeometrySpec('refine_line', 'line', {'x0': 0.5 * radius, 'y0': 0.0, 'x1': 0.5 * radius, 'y1': height})]
    mesh = MeshSpec(mode='nonuniform', dx=_CELL,
                    refinements=[RefinementSpec('refine_line', _CYLINDER_GRADED_CELL, _CYLINDER_GRADED_REACH)])
    return _model('Self-test: pressurised heated cylinder (axisymmetric solid mechanics, graded)',
                  _CYLINDER_SIZE, geometry,
                  [MaterialSpec('core', properties=_elastic(*_CYLINDER_ELASTIC, alpha=_CYLINDER_ALPHA),
                                selection=['all'])],
                  _AXISYMMETRIC,
                  [FeatureSpec('bottom', 'roller', ['y_lo']), FeatureSpec('top', 'roller', ['y_hi']),
                   FeatureSpec('pressure', 'boundary_load', ['x_hi'], {'t_x': -_CYLINDER_PRESSURE, 't_y': 0.0}),
                   FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _CYLINDER_RISE})],
                  mesh=mesh)


def _cylinder_closed_form():
    """(c, sigma_zz [Pa])."""
    lame_lambda, mu = _lame(*_CYLINDER_ELASTIC)
    beta = (3.0 * lame_lambda + 2.0 * mu) * _CYLINDER_ALPHA
    strain = (beta * _CYLINDER_RISE - _CYLINDER_PRESSURE) / (2.0 * (lame_lambda + mu))
    return strain, 2.0 * lame_lambda * strain - beta * _CYLINDER_RISE


def _check_cylinder(results_by_name):
    """u, strain, stresses (hoop = stress_zz, axial = stress_yy), hoop elastic strain and reactions of the graded
    run."""
    strain, stress_axial = _cylinder_closed_form()
    stress_scale = max(_CYLINDER_PRESSURE, abs(stress_axial))
    names = ('u_x', 'u_y', 'strain_xx', 'stress_xx', 'stress_yy', 'stress_xy', 'stress_zz', 'elastic_strain_zz')
    results = results_by_name['graded']
    fields, x_edges, _ = _window_fields(results, names, graded=True)
    radius = float(x_edges[-1])
    expected = {'u_x': strain * _centres(x_edges)[:, None], 'u_y': 0.0, 'strain_xx': strain,
                'stress_xx': -_CYLINDER_PRESSURE, 'stress_yy': stress_axial, 'stress_xy': 0.0,
                'stress_zz': -_CYLINDER_PRESSURE, 'elastic_strain_zz': strain - _CYLINDER_ALPHA * _CYLINDER_RISE}
    scales = {'u_x': abs(strain) * radius, 'u_y': abs(strain) * radius, 'strain_xx': abs(strain),
              'elastic_strain_zz': abs(strain - _CYLINDER_ALPHA * _CYLINDER_RISE)}
    errors = {name: max_relative_error(fields[name], expected[name], scales.get(name, stress_scale)) for name in names}
    ring_force = stress_axial * math.pi * radius ** 2
    top_x, top_y = _reaction(results, 'top')
    bottom_x, bottom_y = _reaction(results, 'bottom')
    force_scale = stress_scale * math.pi * radius ** 2
    errors['F_top'] = abs(top_y - ring_force) / force_scale
    errors['F_bottom'] = abs(bottom_y + ring_force) / force_scale
    errors['F_x'] = (abs(top_x) + abs(bottom_x)) / force_scale
    return {'measured': top_y, 'expected': ring_force, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N', 'eps_rr': strain, 'sigma_zz_Pa': stress_axial,
                        'reactions': {'top': top_y, 'bottom': bottom_y, 'expected_top': ring_force}}}


# ---- solid_axisymmetric_spinning_shaft: radial body force k r (a spinning shaft, k = rho omega^2), z rollers ----
# R 1 mm x H 0.1 mm, E 100 GPa, nu 0.3, f_r = k r, rollers on y_lo / y_hi, r = R free; 'coarse' 50 um and 'fine' 25 um
# cells. Exact (plane strain shaft): u_r = -k r^3 / (8 L) + C r, C = (3 L + lambda) k R^2 / (16 L (lambda + mu)), L =
# lambda + 2 mu; sigma_tt = A r^2 (3 lambda + L) + C (lambda + L), A = -k / (8 L); sigma_rr = A r^2 (3 L + lambda) + C (L +
# lambda). Not in the element space: the fine errors must lie below the tolerance and fall at order >= 1.5.
_SHAFT_SIZE = (1e-3, 0.1e-3)
_SHAFT_ELASTIC = (100e9, 0.3)
_SHAFT_LOAD = 1e13                                 # k [N/m^4]: f_r = k r
_SHAFT_CELLS = {'coarse': 50e-6, 'fine': 25e-6}
_SHAFT_TOLERANCE = 5e-3                            # estimate: Q1 O(h^2) error with 40 cells over R
_SHAFT_MIN_ORDER = 1.5


def _shaft_specs():
    return {run: _model(f'Self-test: spinning shaft, {run} mesh (axisymmetric solid mechanics)', _SHAFT_SIZE, [],
                        [MaterialSpec('shaft', properties=_elastic(*_SHAFT_ELASTIC), selection=['all'])],
                        _AXISYMMETRIC,
                        [FeatureSpec('ends', 'roller', ['y_lo', 'y_hi']),
                         FeatureSpec('spin', 'body_force', ['all'], {'f_x': f'{_SHAFT_LOAD:g} * x', 'f_y': 0.0})],
                        mesh=MeshSpec(mode='uniform', dx=cell))
            for run, cell in _SHAFT_CELLS.items()}


def _shaft_closed_form(radii):
    """(u_r [m], sigma_rr, sigma_tt [Pa]) at the radii."""
    lame_lambda, mu = _lame(*_SHAFT_ELASTIC)
    longitudinal = lame_lambda + 2.0 * mu
    outer = _SHAFT_SIZE[0]
    cubic = -_SHAFT_LOAD / (8.0 * longitudinal)
    linear = (3.0 * longitudinal + lame_lambda) * _SHAFT_LOAD * outer ** 2 / (16.0 * longitudinal * (lame_lambda + mu))
    return (cubic * radii ** 3 + linear * radii, cubic * radii ** 2 * (3.0 * longitudinal + lame_lambda)
            + linear * (longitudinal + lame_lambda), cubic * radii ** 2 * (3.0 * lame_lambda + longitudinal)
            + linear * (lame_lambda + longitudinal))


def _check_shaft(results_by_name):
    """Per run the u_r, sigma_rr and hoop errors at the cell centres; fine errors below the tolerance, observed order
    of the worst error >= _SHAFT_MIN_ORDER."""
    worst_by_run, errors = {}, {}
    for run in _SHAFT_CELLS:
        fields, x_edges, _ = _window_fields(results_by_name[run], ('u_x', 'stress_xx', 'stress_zz'))
        u_r, stress_rr, stress_tt = (values[:, None] for values in _shaft_closed_form(_centres(x_edges)))
        stress_scale = float(np.max(np.abs(stress_tt)))
        run_errors = {'u_r': max_relative_error(fields['u_x'], u_r, float(np.max(np.abs(u_r)))),
                      'sigma_rr': max_relative_error(fields['stress_xx'], stress_rr, stress_scale),
                      'sigma_tt': max_relative_error(fields['stress_zz'], stress_tt, stress_scale)}
        errors[run] = run_errors
        worst_by_run[run] = worst_error(run_errors)
    ratio = _SHAFT_CELLS['coarse'] / _SHAFT_CELLS['fine']
    order = math.log(worst_by_run['coarse'] / worst_by_run['fine']) / math.log(ratio) \
        if worst_by_run['fine'] > 0.0 else math.inf
    normalized = {'fine error': worst_by_run['fine'] / _SHAFT_TOLERANCE,
                  'order': _SHAFT_MIN_ORDER / order if order > 0.0 else math.inf}
    worst = max(normalized.values())
    return {'measured': worst_by_run['fine'], 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'errors': errors, 'observed_order': order, 'normalized': normalized}}


# ---- solid_axisymmetric_breathing_modes: radial modes of a thin slice of a long cylinder (eigenfrequency) ----
# R 4 mm x H 62.5 um, uniform 62.5 um cells (64 x 1: one cell over the height), E 70 GPa, nu 0.25, rho 2200, rollers on
# y_lo / y_hi: with a single cell both node rows are rollers, so u_z is held everywhere and no z-varying mode exists; the
# lowest modes are the z-independent radial breathing modes u_r = J1(k r) of the free r = R surface, (lambda + 2 mu) x
# J0(x) = 2 mu J1(x), x = k R, omega = x c_L / R (the radial Q1 dispersion error depends on dr only).
_MODES_CELL = 62.5e-6
_MODES_SIZE = (4e-3, _MODES_CELL)
_MODES_MATERIAL = (70e9, 0.25, 2200.0)
_MODES_COUNT = 3
_MODES_TOLERANCE = 3e-3                            # estimate: (k h)^2 / 24 ~ 8e-4 for the third mode
_MODES_TABLE = 'modes'
_FREQUENCY_COLUMN = 1


def _modes_specs():
    youngs, nu, density = _MODES_MATERIAL
    return {'lowest': _model('Self-test: radial breathing modes of a cylinder (axisymmetric solid mechanics)',
                             _MODES_SIZE, [],
                             [MaterialSpec('rod', properties=_elastic(youngs, nu, rho=density), selection=['all'])],
                             _AXISYMMETRIC,
                             [FeatureSpec('ends', 'roller', ['y_lo', 'y_hi']),
                              FeatureSpec(_MODES_TABLE, 'resonance_table')],
                             mesh=MeshSpec(mode='uniform', dx=_MODES_CELL),
                             study=StudySpec(type='eigenfrequency', settings={'n_resonances': _MODES_COUNT}))}


def _breathing_frequencies(count):
    """Lowest `count` breathing-mode frequencies [Hz] from the roots of (lambda + 2 mu) x J0(x) - 2 mu J1(x)."""
    from scipy.optimize import brentq
    from scipy.special import j0, j1
    youngs, nu, density = _MODES_MATERIAL
    lame_lambda, mu = _lame(youngs, nu)
    longitudinal = lame_lambda + 2.0 * mu

    def residual(x):
        return longitudinal * x * j0(x) - 2.0 * mu * j1(x)

    grid = np.arange(0.05, 40.0, 0.01)
    values = residual(grid)
    roots = [brentq(residual, grid[index], grid[index + 1])
             for index in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))[:count]]
    speed = math.sqrt(longitudinal / density)
    return [root * speed / (_MODES_SIZE[0] * 2.0 * math.pi) for root in roots]


def _check_modes(results_by_name):
    """Table frequencies (rank order) against the breathing-mode roots."""
    expected = _breathing_frequencies(_MODES_COUNT)
    rows = results_by_name['lowest'].result(_SOLID, _MODES_TABLE)['rows']
    if len(rows) < _MODES_COUNT:
        raise ValueError(f'expected {_MODES_COUNT} modes in the table, got {len(rows)}')
    measured = [float(row[_FREQUENCY_COLUMN]) for row in rows[:_MODES_COUNT]]
    relative = [abs(value / reference - 1.0) for value, reference in zip(measured, expected)]
    return {'measured': measured[0], 'expected': expected[0], 'error': max(relative), 'tolerance': _MODES_TOLERANCE,
            'details': {'frequency_expected': expected, 'frequency_measured': measured, 'relative_errors': relative}}


# ---- cases ----
FORMULATION_CASES = (
    SelfTestCase(
        name='solid_plane_stress', label='Solid mechanics: plane stress (plate and bilayer)',
        description=('out_of_plane plane_stress, two runs on 1 um cells. plate: a 40 x 20 um plate (E 100 GPa, nu 0.3, '
                     'alpha_xx / _yy / _zz 1e-5 / 2e-5 / 0.5e-5 1/K) with rollers on x_lo / y_lo, traction t_x = '
                     f'{_PLATE_TRACTION:g} Pa on x_hi and temperature_change {_PLATE_RISE:g} K: sigma_xx = T, every other '
                     'stress 0, eps_xx = T / E + alpha_xx dT, eps_yy = -nu T / E + alpha_yy dT, elastic_strain_zz = -nu T '
                     f'/ E, roller reactions -T H d and 0 in N (thickness d = {_PLATE_THICKNESS:g} m). bilayer: the solid_bilayer_tension_graded strip on equal cells in plane stress: eps_yy '
                     '= -nu eps_xx, sigma_xx = E eps_xx per layer, sigma_yy = sigma_zz = 0, pull F = sum E eps_xx h. '
                     f'error = the worst field / reaction error over both runs; {_BUDGET_TEXT}. This pins the reduced '
                     'plane-stress stiffness, the reduced thermal stress beta - lambda beta_z / (lambda + 2 mu), the '
                     'eps_zz output and stress_zz = 0.'),
        specs=_plane_stress_specs, check=_check_plane_stress),
    SelfTestCase(
        name='solid_generalized_plane_strain', label='Solid mechanics: generalized plane strain of two heated layers',
        description=('out_of_plane generalized_plane_strain with axial_force '
                     f'{_GPS_AXIAL_FORCE:g} N on a 40 x 20 um section of 1 um cells: layer a (x < 16 um, E 100 GPa, nu '
                     '0.3, alpha_xx / _yy / _zz 1e-5 / 2e-5 / 0.5e-5) beside layer b (E 50 GPa, nu 0.2, alpha 3e-5), '
                     'rollers on x_lo, y_lo and y_hi, x_hi free, temperature_change '
                     f'{_GPS_RISE:g} K. Exact (1D in x): eps_xx = (beta_x dT - lambda e) / L per layer, sigma_zz = lambda '
                     'eps_xx + L e - beta_z dT, sigma_yy = lambda (eps_xx + e) - beta_y dT and the axial balance sum w H '
                     'sigma_zz = F fixes the uniform e; u_x piecewise linear, u_y = 0. error = the worst of the u, '
                     'stress, elastic_strain_zz = e - alpha_zz dT and y-roller reaction (+-sum sigma_yy w) errors; '
                     f'{_BUDGET_TEXT}. This pins the unit-strain superposition, the axial-force integral and the '
                     "reaction bookkeeping with the axial-strain load (bending 'none')."),
        specs=_gps_specs, check=_check_gps),
    SelfTestCase(
        name='solid_generalized_plane_strain_bending',
        label='Solid mechanics: generalized plane strain with bending terms',
        description=("out_of_plane generalized_plane_strain, bending 'xy' (eps_zz = e0 + e1 x + e2 y), the two "
                     'heated layers of solid_generalized_plane_strain stacked along s = x (run along_x) and, '
                     'transposed, along s = y (along_y), 1 um cells: rollers on the s_lo face and both transverse '
                     f'faces, s_hi free, axial_force {_BEND_AXIAL_FORCE:g} N, first moment int s sigma_zz dA = '
                     f'{_BEND_FIRST_MOMENT:g} N m (bending_moment_y = -Q, resp. bending_moment_x = Q) and the other '
                     'moment = F times the transverse mid-line (that bending term 0). Discrete-exact 1D solution: mean '
                     'sigma_ss = 0 per element gives eps_ss = (beta_s dT - lambda eps_zz,c) / L, and the force / first '
                     'moment of the linear-in-s sigma_zz (int s sigma_zz ds = s_c w sigma_zz,c + L e1 w^3 / 12) fix '
                     '(e0, e1). error = the worst u, stress, elastic_strain_zz and roller reaction error over both runs; '
                     f'{_BUDGET_TEXT}. This pins the linear-term loads (x and y weights), the moment integrals about the '
                     'drawing origin and the three-term superposition.'),
        specs=lambda: {run: _bending_spec(along) for run, along in _BEND_RUNS.items()}, check=_check_bending),
    SelfTestCase(
        name='solid_axisymmetric_operator', label='Solid mechanics: axisymmetric operator identities (CPU)',
        description=(f'Host numpy on a {len(_OPERATOR_DX)} x {len(_OPERATOR_DY)} graded grid (no model run): the '
                     'planar 2 x 2 Gauss stencil equals the closed-form Q1 stencil; the (r, z) stiffness with random '
                     'element Lame pairs is symmetric and has the z translation in its null space; for a uniform '
                     'material the uniform-strain field u = (c r, d z) gives K u = the consistent tractions of its '
                     'uniform stress on the outer faces (0 on the axis and inside); the ring mass of random densities '
                     'sums to the body mass, u_r = r gives the exact r^2 moment and M is symmetric. error = the worst '
                     f'relative deviation; tolerance {_OPERATOR_TOLERANCE:g} (float64 round-off).'),
        specs=lambda: {}, check=_operator_check),
    SelfTestCase(
        name='solid_axisymmetric_cylinder', label='Solid mechanics: pressurised heated cylinder (axisymmetric)',
        description=('geometry axisymmetric, a solid cylinder R 20 x H 10 um (E 100 GPa, nu 0.3, alpha 1e-5) between '
                     f'z rollers (y_lo, y_hi), traction t_x = -{_CYLINDER_PRESSURE:g} Pa on x_hi and temperature_change '
                     f'{_CYLINDER_RISE:g} K; one run graded (base {_CELL * 1e6:g} um cells, uniform outside the band, a '
                     f'refinement line at r = R / 2, dx {_CYLINDER_GRADED_CELL * 1e6:g} um; it contains the former '
                     'uniform run). Exact: u_r = c r, c = (beta dT - p) / (2 (lambda + mu)), sigma_rr = sigma_tt '
                     '(stress_zz, the hoop) = -p, axial stress_yy = 2 lambda c - beta dT, elastic_strain_zz = c - alpha '
                     'dT, roller reactions +-sigma_zz pi R^2 in N (whole ring). error = the worst field / reaction '
                     f'error; {_BUDGET_TEXT}. This pins the ring weights, the hoop terms of the stiffness and thermal '
                     'load on equal and unequal widths, the cylinder traction, the axis rule u_r = 0 and the '
                     'whole-ring reactions.'),
        specs=lambda: {'graded': _cylinder_spec()}, check=_check_cylinder),
    SelfTestCase(
        name='solid_axisymmetric_spinning_shaft', label='Solid mechanics: spinning shaft (axisymmetric, convergence)',
        description=('geometry axisymmetric, a slice R 1 mm x H 0.1 mm of a long shaft (E 100 GPa, nu 0.3) between z '
                     f'rollers with the radial body force f_x = {_SHAFT_LOAD:g} x (a spinning shaft, rho omega^2 r), r = '
                     'R free; runs coarse (50 um) and fine (25 um cells). Closed form u_r = -k r^3 / (8 L) + C r, C = (3 '
                     'L + lambda) k R^2 / (16 L (lambda + mu)), sigma_rr / sigma_tt from it; not in the element space. '
                     f'Passes when the worst fine error (u_r, sigma_rr, hoop stress at the cell centres) is below '
                     f'{_SHAFT_TOLERANCE:g} and the observed order of the worst error is >= {_SHAFT_MIN_ORDER:g}; '
                     'tolerances are estimates. This pins the ring-weighted body force and the near-axis hoop terms.'),
        specs=_shaft_specs, check=_check_shaft),
    SelfTestCase(
        name='solid_axisymmetric_breathing_modes', label='Solid mechanics: radial breathing modes (axisymmetric)',
        description=('eigenfrequency study of geometry axisymmetric on a slice R '
                     f'{_MODES_SIZE[0] * 1e3:g} mm x H {_MODES_SIZE[1] * 1e6:g} um (one row of '
                     f'{round(_MODES_SIZE[0] / _MODES_CELL)} cells of {_MODES_CELL * 1e6:g} um; E 70 GPa, nu 0.25, rho '
                     '2200) between z rollers (u_z held on every node) with a resonance table: the lowest '
                     f'{_MODES_COUNT} modes are the radial breathing modes u_r = J1(k r) of the free surface, roots of '
                     '(lambda + 2 mu) x J0(x) = 2 mu J1(x), f = x c_L / (2 pi R). error = the worst relative frequency '
                     f'error; tolerance {_MODES_TOLERANCE:g} (an estimate of the Q1 dispersion error, set by dr). This '
                     'pins the whole-ring mass and the axisymmetric stiffness in the modal solve. Not yet run with one '
                     'cell over the height (was 8).'),
        specs=_modes_specs, check=_check_modes),
)

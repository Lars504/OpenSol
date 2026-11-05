"""Self-test solid_segment_load of the segment_load feature (tractions on parts of a face and on interior lines):
uniaxial blocks loaded over a whole face (planar plane strain and axisymmetric, exact in Q1) and partial / interior
segments whose consistent loads the support reactions balance exactly."""
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, max_relative_error, with_exact_cg, worst_error
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests import SOLID_TYPE_NAME, _elastic, _reaction, _window_fields

_SOLID = 'solid'                                   # = selftests._SOLID (the helpers read this instance)
_TOLERANCE = 1e-6                                  # float64 CG to EXACT_CG_TOLERANCE, float32 fields
_CELL = 1e-6                                       # m, uniform mesh
_BLOCK = (20e-6, 10e-6)                            # m: planar W x H; axisymmetric R x H
_ELASTIC = (100e9, 0.3)                            # E [Pa], nu
_PRESSURE = 1e6                                    # Pa pressing down on the top: t_y = -p
_TOP_PART = (4.5e-6, 13.25e-6)                     # m: x (r) span of the partial top segment, both ends inside cells
_INNER_LINE = (10e-6, 2.5e-6, 7.5e-6)              # m: vertical interior segment on the node line x (r), y from, to
_INNER_TRACTION = 2e5                              # Pa: t_x planar, t_y axisymmetric (the axis reaction is not read)
_PLANAR, _AXISYMMETRIC = {}, {'geometry': 'axisymmetric'}
# run -> (settings, full top or partial top + inner line)
_RUNS = {'full': (_PLANAR, False), 'partial': (_PLANAR, True), 'ring_full': (_AXISYMMETRIC, False),
         'ring_partial': (_AXISYMMETRIC, True)}


def _line(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'line', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _segment_load_specs():
    """Per run a block on rollers (planar: x_lo and y_lo; axisymmetric: y_lo, the axis holds u_r) with segment loads:
    the whole top line, or the partial top segment plus the vertical interior segment."""
    width, height = _BLOCK
    specs = {}
    for run, (settings, partial) in _RUNS.items():
        if partial:
            inner_x, inner_y0, inner_y1 = _INNER_LINE
            geometry = [_line('top_part', _TOP_PART[0], height, _TOP_PART[1], height),
                        _line('inner', inner_x, inner_y0, inner_x, inner_y1)]
            inner_traction = ({'t_x': 0.0, 't_y': _INNER_TRACTION} if settings
                              else {'t_x': _INNER_TRACTION, 't_y': 0.0})
            loads = [FeatureSpec('top_load', 'segment_load', ['top_part'], {'t_x': 0.0, 't_y': -_PRESSURE}),
                     FeatureSpec('inner_load', 'segment_load', ['inner'], inner_traction)]
        else:
            geometry = [_line('top', 0.0, height, width, height)]
            loads = [FeatureSpec('top_load', 'segment_load', ['top'], {'t_x': 0.0, 't_y': -_PRESSURE})]
        supports = [FeatureSpec('bottom', 'roller', ['y_lo'])]
        if not settings:
            supports.insert(0, FeatureSpec('left', 'roller', ['x_lo']))
        specs[run] = with_exact_cg(ModelSpec(
            title=f'Self-test: segment loads, run {run} (solid mechanics)',
            domain=DomainSpec(size_x=width, size_y=height), geometry=geometry,
            materials=[MaterialSpec('block', properties=_elastic(*_ELASTIC), selection=['all'])],
            physics=[PhysicsSpec(_SOLID, SOLID_TYPE_NAME, settings=dict(settings), features=supports + loads)],
            mesh=MeshSpec(mode='uniform', dx=_CELL), study=StudySpec(type='stationary')))
    return specs


def _uniaxial_errors(results, axisymmetric):
    """Field errors of a block under the uniform top stress sigma_yy = -p (sigma_xx = 0): planar plane strain eps_yy =
    (1 - nu^2) sigma / E, eps_xx = -nu (1 + nu) sigma / E, sigma_zz = nu sigma; axisymmetric eps_zz = sigma / E, eps_rr =
    -nu sigma / E, hoop stress 0. Linear u: exact in Q1 at the centres."""
    youngs, nu = _ELASTIC
    stress = -_PRESSURE
    strain_y = stress / youngs if axisymmetric else (1.0 - nu * nu) * stress / youngs
    strain_x = -nu * stress / youngs if axisymmetric else -nu * (1.0 + nu) * stress / youngs
    names = ('u_x', 'u_y', 'stress_xx', 'stress_yy', 'stress_xy', 'stress_zz')
    fields, x_edges, y_edges = _window_fields(results, names)
    x_centres, y_centres = 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:])
    width, height = _BLOCK
    expected = {'u_x': strain_x * x_centres[:, None], 'u_y': strain_y * y_centres[None, :], 'stress_xx': 0.0,
                'stress_yy': stress, 'stress_xy': 0.0, 'stress_zz': 0.0 if axisymmetric else nu * stress}
    scales = {'u_x': abs(strain_x) * width, 'u_y': abs(strain_y) * height}
    return {name: max_relative_error(fields[name], expected[name], scales.get(name, _PRESSURE)) for name in names}


def _check_segment_loads(results_by_name):
    """Whole-top runs: the uniaxial fields and the base reaction; partial runs: the support reactions against the
    segment totals t L (planar) / t pi (r_1^2 - r_0^2) and t 2 pi r L (axisymmetric)."""
    width, height = _BLOCK
    start, stop = _TOP_PART
    inner_x, inner_y0, inner_y1 = _INNER_LINE
    inner_length = inner_y1 - inner_y0
    errors, reactions = {}, {}
    for run, (settings, partial) in _RUNS.items():
        results = results_by_name[run]
        axisymmetric = bool(settings)
        force_scale = _PRESSURE * (math.pi * width ** 2 if axisymmetric else width)   # the whole top face
        if partial:
            top_force = _PRESSURE * (math.pi * (stop ** 2 - start ** 2) if axisymmetric else stop - start)
        else:
            top_force = force_scale
        bottom_x, bottom_y = _reaction(results, 'bottom')
        expected_bottom = top_force   # the roller pushes up against the load pressing down
        if partial and axisymmetric:   # the inner line pulls the ring up: the base holds it down
            expected_bottom -= _INNER_TRACTION * 2.0 * math.pi * inner_x * inner_length
        errors[f'{run}.F_bottom'] = abs(bottom_y - expected_bottom) / force_scale
        reactions[run] = {'bottom_y': bottom_y, 'expected_bottom_y': expected_bottom}
        if not axisymmetric:
            left_x, _ = _reaction(results, 'left')
            expected_left = -_INNER_TRACTION * inner_length if partial else 0.0
            errors[f'{run}.F_left'] = abs(left_x - expected_left) / force_scale
            reactions[run].update(left_x=left_x, expected_left_x=expected_left)
        if not partial:
            errors.update({f'{run}.{name}': error for name, error in _uniaxial_errors(results, axisymmetric).items()})
    return {'measured': reactions['partial']['bottom_y'], 'expected': reactions['partial']['expected_bottom_y'],
            'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'reactions': reactions, 'unit_force': {'planar': 'N/m', 'axisymmetric': 'N'}}}


SEGMENT_LOAD_CASES = (
    SelfTestCase(
        name='solid_segment_load', label='Solid mechanics: tractions on face segments and interior lines',
        description=(f'segment_load on a {_BLOCK[0] * 1e6:g} x {_BLOCK[1] * 1e6:g} um block (E {_ELASTIC[0]:g} Pa, nu '
                     f'{_ELASTIC[1]:g}, {_CELL * 1e6:g} um cells). Runs full (planar plane strain, rollers x_lo / y_lo) '
                     'and ring_full (geometry axisymmetric, roller y_lo): a line over the whole top face, t_y = '
                     f'-{_PRESSURE:g} Pa, gives the uniform uniaxial stress (exact in Q1: u, stresses and the base '
                     'reaction p W per m / p pi R^2 N). Runs partial / ring_partial: the top segment x = '
                     f'{_TOP_PART[0] * 1e6:g} .. {_TOP_PART[1] * 1e6:g} um (both ends inside cells) plus a vertical '
                     f'interior segment at x = {_INNER_LINE[0] * 1e6:g} um, y = {_INNER_LINE[1] * 1e6:g} .. '
                     f'{_INNER_LINE[2] * 1e6:g} um with {_INNER_TRACTION:g} Pa along x (planar) / y (axisymmetric): '
                     'the consistent nodal loads integrate the shape functions over the clipped segment, so the '
                     'support reactions equal the totals t L, p pi (r_1^2 - r_0^2) and t 2 pi r L exactly (a '
                     'half-cell rounding of the ends would miss by ~5 %, a planar weight on the annulus by more). '
                     f'Tolerance {_TOLERANCE:g} (float64 CG to {EXACT_CG_TOLERANCE:g}, float32 fields).'),
        specs=_segment_load_specs, check=_check_segment_loads),
)

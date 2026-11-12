"""Self-tests of interface-aware sampling (igor 1.8: outline samples, one-sided least-squares extrapolation on
Cartesian and polar windows) and of max / min over mirror images (igor 1.9); registered by model.study.register."""
import math
from types import SimpleNamespace

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import scaled_worst_result
from .spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec,
                   SelectionSpec, SolverDefaultsSpec, StudySpec)

_PHYSICS = 'em'

# ---- model_interface_sampling (CPU): a disk (r = 1 um) and a slab with a round hole (difference) in a 4 x 4 um
# window of 20 nm cells; one-sided values of fields linear on each side must be exact (least-squares planes through
# side cells only), outline lengths match the perimeters, normals point out of the objects (into the hole).
_GRID_CELLS = 200
_GRID_CELL = 20e-9                 # m
_DISK = {'cx': 2e-6, 'cy': 1.4e-6, 'r': 1e-6}
_SLAB = {'x0': 0.5e-6, 'y0': 2.6e-6, 'x1': 3.5e-6, 'y1': 3.4e-6}
_HOLE = {'cx': 2e-6, 'cy': 3e-6, 'r': 0.25e-6}
_LINEAR_FIELDS = {'inside': (1.0, 2e6, -1e6), 'outside': (-3.0, 0.5e6, 4e6)}   # a + b x + c y (x, y in m)
_EXACT_TOLERANCE = 1e-9            # largest |one-sided - linear field| / field scale (float64 round-off)
_LENGTH_TOLERANCE = 1e-4           # relative outline length (circle polygon ~ (pi / n)^2 / 6)
_NORMAL_TOLERANCE = 1e-6           # |normal - exact radial normal| of the circle samples
_MIN_USABLE_FRACTION = 0.99        # outline samples with a well-posed fit on both sides
_LINEAR_FIELDS_Y = {'inside': (-2.0, -1e6, 3e6), 'outside': (0.5, 1.5e6, -2e6)}   # y component of a vector field
_PARTNER_NAMES = {'ex_re': 'ey_re', 'E_x': 'E_y', 'hx': 'hy', 'u_x': 'u_y', 'stress_xx': None, 'eps_xy': None,
                  'Q_ext': None, 'T': None}   # interface_sampling.partner_component

# ---- model_interface_polar (CPU): a polar sector r = 1 .. 4 um, phi = 0 .. 90 deg of 150 x 315 cells (dr = 20 nm,
# arc <= 20 nm) with a disk and a rectangle (oblique to the cells) in the drawing plane. One-sided values of fields
# linear in the drawing plane are exact (planes through the cells' drawing-plane centres); a projected line profile
# of a vector field given by its grid-frame (radial, angular) components reads its drawing-plane normal / tangential
# part (not linear on the cells: curvature error ~ (2.5 h / r)^2).
_POLAR_RADIUS_MIN = 1e-6           # m
_POLAR_DEPTH = 3e-6                # m
_POLAR_SPAN = 0.5 * math.pi        # rad
_POLAR_CELLS = (150, 315)
_POLAR_DISK = {'cx': 2.5e-6 * math.cos(0.25 * math.pi), 'cy': 2.5e-6 * math.sin(0.25 * math.pi), 'r': 0.6e-6}
_POLAR_BLOCK = {'x0': 2.6e-6, 'y0': 0.3e-6, 'x1': 3.4e-6, 'y1': 1.1e-6}
_POLAR_MIN_USABLE_FRACTION = 0.98
_POLAR_PROJECTION_TOLERANCE = 1e-3   # |profile - exact part| / field scale (10)


# ---- model_interface_sampling ----
def _sampling_spec():
    return ModelSpec(
        title='Self-test: interface sampling geometry',
        domain=DomainSpec(size_x=_GRID_CELLS * _GRID_CELL, size_y=_GRID_CELLS * _GRID_CELL),
        geometry=[GeometrySpec('disk', 'circle', dict(_DISK)), GeometrySpec('slab', 'rectangle', dict(_SLAB)),
                  GeometrySpec('hole', 'circle', dict(_HOLE)),
                  GeometrySpec('holey', 'difference', {'base': 'slab', 'subtract': ['hole']})],
        selections=[SelectionSpec(f'{name}_{side}', 'interface', [name], side)
                    for name in ('disk', 'holey') for side in ('inside', 'outside')])


def _linear_field(side, x, y, coefficients=_LINEAR_FIELDS):
    a, b, c = coefficients[side]
    return a + b * x + c * y


def _add_stencil_errors(spec, resolved, frame, name, values, min_usable, errors, tolerances, details):
    """Per side of object `name`: the one-sided interface_stencil values of the linear field `values` against the exact
    side field ('exact', / 10) and the usable share below min_usable ('usable'), into errors / tolerances / details."""
    from .interface_sampling import interface_stencil
    for side in ('inside', 'outside'):
        stencil = interface_stencil(spec, spec.selection(f'{name}_{side}'), resolved, frame)
        one_sided = stencil.one_sided(values[stencil.cells_x, stencil.cells_y])
        usable = stencil.usable
        expected = _linear_field(side, stencil.points[usable, 0], stencil.points[usable, 1])
        errors[f'{name} {side} exact'] = float(np.max(np.abs(one_sided[usable] - expected))) / 10.0
        tolerances[f'{name} {side} exact'] = _EXACT_TOLERANCE
        errors[f'{name} {side} usable'] = max(0.0, min_usable - float(np.mean(usable)))
        tolerances[f'{name} {side} usable'] = 1e-12
        details[f'{name} {side}'] = {'samples': int(usable.size), 'usable': int(np.count_nonzero(usable))}


def _check_sampling(_results_by_name):
    """Outline lengths / normals and one-sided values of side-wise linear fields (exact) on the disk and the holey
    slab, an interface line profile along the slab top, normal / tangential projections of a side-wise linear vector
    field along the slab top and left side, and partner_component names; measured = error = the worst item in
    tolerance units."""
    from .geometry import rasterize_window, resolve_geometry
    from .interface_sampling import (SAMPLE_SPACING, interface_line_profile, object_outline, partner_component,
                                     window_frame)
    spec = _sampling_spec()
    resolved = resolve_geometry(spec, {})
    centres = (np.arange(_GRID_CELLS) + 0.5) * _GRID_CELL
    box = (0.0, _GRID_CELLS * _GRID_CELL, 0.0, _GRID_CELLS * _GRID_CELL)
    frame = window_frame(centres, centres, box=box)
    x_cells, y_cells = np.meshgrid(centres, centres, indexing='ij')
    errors, tolerances, details = {}, {}, {}
    for name in ('disk', 'holey'):
        window, cropped = rasterize_window(name, resolved, centres.astype(np.float32), centres.astype(np.float32), np)
        inside = np.zeros((_GRID_CELLS, _GRID_CELLS), dtype=bool)
        inside[window] = cropped
        values = np.where(inside, _linear_field('inside', x_cells, y_cells), _linear_field('outside', x_cells, y_cells))
        _add_stencil_errors(spec, resolved, frame, name, values, _MIN_USABLE_FRACTION, errors, tolerances, details)
    spacing = SAMPLE_SPACING * _GRID_CELL
    perimeters = {'disk': 2.0 * math.pi * _DISK['r'],
                  'holey': 2.0 * ((_SLAB['x1'] - _SLAB['x0']) + (_SLAB['y1'] - _SLAB['y0'])) + 2.0 * math.pi * _HOLE['r']}
    for name, perimeter in perimeters.items():
        points, normals, lengths = object_outline(name, resolved, spacing, frame.inside)
        errors[f'{name} length'] = abs(float(np.sum(lengths)) / perimeter - 1.0)
        tolerances[f'{name} length'] = _LENGTH_TOLERANCE
        details[f'{name} length'] = float(np.sum(lengths))
        circle = _DISK if name == 'disk' else _HOLE
        radial = points - np.array([circle['cx'], circle['cy']])
        distance = np.hypot(radial[:, 0], radial[:, 1])
        on_circle = np.abs(distance - circle['r']) < 0.01 * circle['r']
        exact = radial[on_circle] / distance[on_circle, None] * (1.0 if name == 'disk' else -1.0)   # hole: inward
        errors[f'{name} normals'] = float(np.max(np.abs(normals[on_circle] - exact)))
        tolerances[f'{name} normals'] = _NORMAL_TOLERANCE
    window, cropped = rasterize_window('holey', resolved, centres.astype(np.float32), centres.astype(np.float32), np)
    inside = np.zeros((_GRID_CELLS, _GRID_CELLS), dtype=bool)
    inside[window] = cropped
    field_data = SimpleNamespace(values=np.where(inside, _linear_field('inside', x_cells, y_cells), 0.0),
                                 extent=box, x_centers=centres, y_centers=centres)
    start, end = (0.8e-6, _SLAB['y1']), (3.2e-6, _SLAB['y1'])   # along the slab top
    _, samples, (line_x, line_y) = interface_line_profile(spec, {}, 'holey_inside', field_data, start, end)
    errors['line profile'] = float(np.nanmax(np.abs(samples - _linear_field('inside', line_x, line_y)))) / 10.0
    tolerances['line profile'] = _EXACT_TOLERANCE
    errors['line profile finite'] = float(np.mean(~np.isfinite(samples)))
    tolerances['line profile finite'] = 1e-12
    partner_data = SimpleNamespace(values=np.where(inside, _linear_field('inside', x_cells, y_cells, _LINEAR_FIELDS_Y),
                                                   0.0), extent=box, x_centers=centres, y_centers=centres)
    lines = {'top': ((0.8e-6, _SLAB['y1']), (3.2e-6, _SLAB['y1']), (0.0, 1.0)),   # (start, end, outward normal)
             'left': ((_SLAB['x0'], 2.7e-6), (_SLAB['x0'], 3.3e-6), (-1.0, 0.0))}
    for line_name, (start, end, (normal_x, normal_y)) in lines.items():
        for projection in ('normal', 'tangential'):
            _, samples, (line_x, line_y) = interface_line_profile(spec, {}, 'holey_inside', field_data, start, end,
                                                                  partner_data=partner_data, projection=projection)
            along_x, along_y = (normal_x, normal_y) if projection == 'normal' else (-normal_y, normal_x)
            expected = (along_x * _linear_field('inside', line_x, line_y)
                        + along_y * _linear_field('inside', line_x, line_y, _LINEAR_FIELDS_Y))
            item = f'{projection} projection along the slab {line_name}'
            errors[item] = float(np.max(np.abs(samples - expected))) / 10.0   # NaN: a sample without a fit
            tolerances[item] = _EXACT_TOLERANCE
    errors['partner names'] = float(sum(partner_component(name) != partner for name, partner in _PARTNER_NAMES.items()))
    tolerances['partner names'] = 0.5
    return scaled_worst_result(errors, tolerances, **details)


# ---- model_interface_polar ----
def _polar_spec():
    return ModelSpec(
        title='Self-test: interface sampling on a polar window',
        domain=DomainSpec(size_x=_POLAR_DEPTH, size_y=_POLAR_SPAN, coordinates='polar', radius_min=_POLAR_RADIUS_MIN),
        geometry=[GeometrySpec('disk', 'circle', dict(_POLAR_DISK)),
                  GeometrySpec('block', 'rectangle', dict(_POLAR_BLOCK))],
        selections=[SelectionSpec(f'{name}_{side}', 'interface', [name], side)
                    for name in ('disk', 'block') for side in ('inside', 'outside')])


def _check_polar(_results_by_name):
    """Exact one-sided values of drawing-plane linear fields on the polar cells, and normal / tangential line profiles
    of a vector field in grid-frame components along the block top (native FieldData); worst item in tolerance
    units."""
    from .geometry import domain_map, rasterize, resolve_geometry
    from .interface_sampling import interface_line_profile, window_frame
    spec = _polar_spec()
    resolved = resolve_geometry(spec, {})
    window_map = domain_map(spec, {})
    n_radial, n_angular = _POLAR_CELLS
    x_centres = (np.arange(n_radial) + 0.5) * (_POLAR_DEPTH / n_radial)
    y_centres = (np.arange(n_angular) + 0.5) * (_POLAR_SPAN / n_angular)
    frame = window_frame(x_centres, y_centres, window_map=window_map)
    mapped_x, mapped_y = frame.mapped_x, frame.mapped_y
    errors, tolerances, details = {}, {}, {}
    masks = {}
    for name in ('disk', 'block'):
        masks[name] = np.asarray(rasterize(name, resolved, mapped_x.astype(np.float32), mapped_y.astype(np.float32),
                                           np), dtype=bool)
        values = np.where(masks[name], _linear_field('inside', mapped_x, mapped_y),
                          _linear_field('outside', mapped_x, mapped_y))
        _add_stencil_errors(spec, resolved, frame, name, values, _POLAR_MIN_USABLE_FRACTION, errors, tolerances,
                            details)
    angle = np.arctan2(mapped_y, mapped_x)   # planar polar grid frame: radial (cos, sin), angular (-sin, cos)
    vector_x = _linear_field('inside', mapped_x, mapped_y)
    vector_y = _linear_field('inside', mapped_x, mapped_y, _LINEAR_FIELDS_Y)
    components = {'radial': vector_x * np.cos(angle) + vector_y * np.sin(angle),
                  'angular': -vector_x * np.sin(angle) + vector_y * np.cos(angle)}

    def native(values):
        return SimpleNamespace(values=np.where(masks['block'], values, np.nan), x_centers=_POLAR_RADIUS_MIN + x_centres,
                               y_centers=y_centres, mapped_centres=(mapped_x, mapped_y),
                               extent=(_POLAR_RADIUS_MIN, _POLAR_RADIUS_MIN + _POLAR_DEPTH, 0.0, _POLAR_SPAN))
    start, end = (2.7e-6, _POLAR_BLOCK['y1']), (3.3e-6, _POLAR_BLOCK['y1'])   # along the block top, normal +y
    for projection, (along_x, along_y) in (('normal', (0.0, 1.0)), ('tangential', (-1.0, 0.0))):
        _, samples, (line_x, line_y) = interface_line_profile(
            spec, {}, 'block_inside', native(components['radial']), start, end,
            partner_data=native(components['angular']), projection=projection)
        expected = (along_x * _linear_field('inside', line_x, line_y)
                    + along_y * _linear_field('inside', line_x, line_y, _LINEAR_FIELDS_Y))
        item = f'{projection} projection along the block top'
        errors[item] = float(np.max(np.abs(samples - expected))) / 10.0
        tolerances[item] = _POLAR_PROJECTION_TOLERANCE
    return scaled_worst_result(errors, tolerances, **details)


# ---- em_fdfd_mirror_extremes: max / min over the mirror images (igor 1.9). em_fdfd_ez PEC box halved by an
# antisymmetric x_lo mirror (window 51 x 61 cells of 20 nm, the em_fdfd_parity_sectors_eigen layout): its resonance
# near 1.55 um is the (2, 1) mode, odd about the plane, so the drawn half holds one sign of E_z only ----
_CAVITY_CELL = 20e-9                             # m
_MIRROR_CELLS = (51, 61)
_MIRROR_SIDE = (0.4e-6, 0.2e-6, 0.8e-6, 1.0e-6)   # m, x0, y0, x1, y1 of 'side' (cells 20..39 x 10..49, off the plane)
_MIRROR_EXTREME_TOLERANCE = 1e-6                 # relative to max |E_z| (the same float32 values)


def _mirror_extremes_spec():
    """Antisymmetric-x_lo half PEC box (eigenfrequency, cpu) with max / min of Re E_z and of x over the window and
    the max of Re E_z over 'side' (off the plane: no images)."""
    settings = {'bc_x_lo': 'antisymmetric', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec',
                'wavelength': 1.55e-6}
    x0, y0, x1, y1 = _MIRROR_SIDE
    field_name = f'{_PHYSICS}_ez_re'
    return ModelSpec(
        title='Self-test: max / min over the mirror image of an odd mode',
        domain=DomainSpec(size_x=_MIRROR_CELLS[0] * _CAVITY_CELL, size_y=_MIRROR_CELLS[1] * _CAVITY_CELL),
        geometry=[GeometrySpec('side', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_ez', settings=settings,
                             features=[FeatureSpec('resonances', 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_CAVITY_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}),
        derived_values=[DerivedValueSpec('Ez_max', field_name, operation='max'),
                        DerivedValueSpec('Ez_min', field_name, operation='min'),
                        DerivedValueSpec('x_max', 'x', operation='max'),
                        DerivedValueSpec('x_min', 'x', operation='min'),
                        DerivedValueSpec('Ez_max_side', field_name, selection=['side'], operation='max')])


def _check_mirror_extremes(results_by_name):
    """Unfolded extremes vs the drawn-half field: max = max(f, -f), min = -max, x_min = -x_max (the plane at x = 0),
    'side' = the plain maximum of its cells; error = worst item in tolerance units."""
    results = results_by_name['half']
    derived = {name: float(value) for name, value in results.summary()['derived'].items()}
    values = np.asarray(results.field(f'{_PHYSICS}.ez_re').values, dtype=np.float64)
    scale = float(np.max(np.abs(values)))
    x0, y0, x1, y1 = (round(bound / _CAVITY_CELL) for bound in _MIRROR_SIDE)
    unfolded_max = max(float(values.max()), -float(values.min()))
    x_last = (_MIRROR_CELLS[0] - 0.5) * _CAVITY_CELL
    measured = {'Ez_max': derived['Ez_max'], 'Ez_min': derived['Ez_min'], 'x_max': derived['x_max'],
                'x_min': derived['x_min'], 'Ez_max_side': derived['Ez_max_side']}
    expected = {'Ez_max': unfolded_max, 'Ez_min': -unfolded_max, 'x_max': x_last, 'x_min': -x_last,
                'Ez_max_side': float(values[x0:x1, y0:y1].max())}
    scales = {'Ez_max': scale, 'Ez_min': scale, 'Ez_max_side': scale, 'x_max': x_last, 'x_min': x_last}
    errors = {item: (abs(measured[item] - expected[item]) / scales[item] if scale > 0.0 else math.inf)
              for item in expected}
    errors['one_signed_half'] = 0.0 if min(float(values.max()), -float(values.min())) < 0.05 * scale else math.inf
    tolerances = dict.fromkeys(errors, _MIRROR_EXTREME_TOLERANCE)
    return scaled_worst_result(errors, tolerances, measured=measured, expected=expected, field_scale=scale,
                               folded_max=float(values.max()), folded_min=float(values.min()))


INTERFACE_SELFTEST_CASES = (
    SelfTestCase(
        name='model_interface_sampling',
        label='Interface sampling: outlines and exact one-sided values',
        description=f'CPU, no run: a disk (r = 1 um) and a 3 x 0.8 um slab minus a round hole (r = 0.25 um, a '
                    f'difference composite) on a {_GRID_CELLS} x {_GRID_CELLS} window of {_GRID_CELL * 1e9:g} nm '
                    'cells, interface selections inside / outside of each. Cell values a + b x + c y with different '
                    'coefficients on the two sides: every usable one-sided outline value (least-squares planes '
                    'through side cells 1.5 and 2.5 cells deep, extrapolated linearly) must equal the linear field '
                    f'of its own side ({_EXACT_TOLERANCE:g} of the field scale), at least {_MIN_USABLE_FRACTION:g} of '
                    'the samples usable; outline lengths = 2 pi r and the slab perimeter + hole circumference '
                    f'({_LENGTH_TOLERANCE:g}); circle normals = the exact radial ones, pointing into the hole for the '
                    f'composite ({_NORMAL_TOLERANCE:g}); an interface line profile along the slab top reads the '
                    'inside field exactly, and projected profiles of a side-wise linear vector field (x / y '
                    'components) along the slab top and left side give its normal / tangential part exactly; '
                    'partner_component maps ex_re -> ey_re, E_x -> E_y, hx -> hy, u_x -> u_y and finds none in '
                    'stress_xx, eps_xy, Q_ext, T. measured = error = the worst item in tolerance units.',
        specs=dict,
        check=_check_sampling),
    SelfTestCase(
        name='model_interface_polar',
        label='Interface sampling on a polar window',
        description=f'CPU, no run: a polar sector r = {_POLAR_RADIUS_MIN * 1e6:g} .. '
                    f'{(_POLAR_RADIUS_MIN + _POLAR_DEPTH) * 1e6:g} um, phi = 0 .. 90 deg of {_POLAR_CELLS[0]} x '
                    f'{_POLAR_CELLS[1]} cells, a disk (r = 0.6 um at r = 2.5 um, 45 deg) and a 0.8 x 0.8 um '
                    'rectangle in the drawing plane (oblique to the cells), interface selections inside / outside '
                    'of each. Cell values a + b x + c y (drawing-plane x, y at the mapped centres) per side: every '
                    f'usable one-sided value exact ({_EXACT_TOLERANCE:g} of the field scale; planes fitted through '
                    f'the drawing-plane centres), at least {_POLAR_MIN_USABLE_FRACTION:g} of the samples usable. A '
                    'vector field linear in the drawing plane, given on native (r, phi) FieldData by its radial / '
                    'angular components (NaN outside the rectangle): normal and tangential line profiles along the '
                    'rectangle top match its drawing-plane y / -x part within '
                    f'{_POLAR_PROJECTION_TOLERANCE:g} of the field scale (grid components are not linear: '
                    'curvature error ~ (2.5 h / r)^2). measured = error = the worst item in tolerance units.',
        specs=dict,
        check=_check_polar),
    SelfTestCase(
        name='em_fdfd_mirror_extremes',
        label='EM FDFD: max / min over mirror images',
        description=f'em_fdfd_ez (cpu) PEC box halved by an antisymmetric x_lo mirror ({_MIRROR_CELLS[0]} x '
                    f'{_MIRROR_CELLS[1]} cells of {_CAVITY_CELL * 1e9:g} nm), eigenfrequency study near 1.55 um: the '
                    '(2, 1) mode, odd about the plane, so the drawn half holds one sign of E_z (checked: the smaller '
                    'side extreme below 5 % of max |E_z|). Derived max / min of Re E_z over the window must include '
                    'the mirror image -E_z: max = max(max f, -min f), min = -max; max / min of x include the reflected '
                    'x (plane at x = 0): x_min = -x_max = -(N - 1/2) h; the max over a rectangle off the plane is '
                    'the plain maximum of its cells (no image). error = the worst deviation / scale in tolerance '
                    f'units ({_MIRROR_EXTREME_TOLERANCE:g}: the same float32 values).',
        specs=lambda: {'half': _mirror_extremes_spec()},
        check=_check_mirror_extremes),
)

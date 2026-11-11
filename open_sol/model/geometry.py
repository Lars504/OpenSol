"""Geometry objects: kinds, entity listings, evaluation, rasterisation (numpy or cupy via ``xp``) and outlines.
Coordinates are physical [m] with the origin at the physical-domain corner (polar / spherical domains: at the polar
origin, DomainMap); masks test cell centres (mapped ones on such domains: rasterize_mapped). 3D models
(spec v2 size_z) extrude the 2D kinds along an axis over z0..z1, optionally tapered (Z_PARAM_NAMES), and add box /
cylinder_z / sphere; every 3D object is rasterised as xy cross-sections per z cell (_section). An array resolves into
moved copies of its base (extra resolved shapes named '<array>[i,j]', is_array_copy), OR-ed like a union."""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..core.metric import ANGLE_WRAPS, AXIS_ANGLE_SNAP, CARTESIAN, Coordinates, from_cartesian, to_cartesian
from .expressions import error_message, evaluate_scalar
from .schema import Param, resolve_values, short_repr
from .spec import outer_faces

AXIS_TOLERANCE = 1e-6   # relative to max(size_x, size_y): tolerance for "line is horizontal / vertical"
_RADIUS_SLACK = 1e-9    # relative: a rounded_rectangle radius this far above half its shorter side is clamped to it
# 3D-only params of the 2D kinds (and of arrays): must stay unset (z_param_set False) in 2D models
Z_PARAM_NAMES = ('z0', 'z1', 'z', 'axis', 'sidewall_angle', 'nz', 'pitch_z')
_UNSET_3D_VALUES = {'axis': 'z', 'sidewall_angle': 0.0, 'nz': 1}   # values of Z_PARAM_NAMES that mean unset besides None
UNION_KINDS = ('union', 'array')   # composites whose mask is the OR of their (resolved) inputs
ARRAY_MAX_COPIES = 200_000         # copies of one array (each is a resolved shape)
_BUCKETED_COPIES = 8               # arrays of at least this many copies rasterize each copy only near its points
# plain kinds whose rasterize test broadcasts (n, 1) x-centres against (1, m) y-centres (rasterize_window): only the
# bool result and one float array are window-sized (full broadcast_to operands made 3-4 float64 window temporaries)
_SEPARABLE_KINDS = ('rectangle', 'box', 'rounded_rectangle', 'circle', 'cylinder_z', 'sphere', 'ellipse')
_PLANE_AXES = {'z': ('x', 'y'), 'x': ('y', 'z'), 'y': ('x', 'z')}   # extrusion axis -> model axes of the shape's (x, y)


def plane_axes(axis) -> tuple:
    """Model axes ('x' | 'y' | 'z', 'x' | 'y' | 'z') of a prism's (x, y) for extrusion axis 'z' | 'x' | 'y'."""
    return _PLANE_AXES[axis]


def z_param_set(name, value) -> bool:
    """True when a Z_PARAM_NAMES param holds a value only 3D models may have (None, axis 'z' and sidewall_angle 0
    count as unset)."""
    return value is not None and not (name in _UNSET_3D_VALUES and value == _UNSET_3D_VALUES[name])


@dataclass(frozen=True)
class GeometryKind:
    """A geometry object kind: entity kind, params and the description describe('geometry') shows."""
    name: str
    label: str
    entity_kind: str          # 'domain' | 'point' | 'line'
    params: tuple = ()        # Param, in display order
    composite: bool = False   # boolean combination of other domain objects
    description: str = ''
    summary: str = ''         # GUI Add geometry menu tooltip; '' -> none
    dimensions: tuple = (2, 3)   # model dimensions the kind is allowed in (box / cylinder_z / sphere: (3,))


def _length(name, default, label, help_text, tip, positive=False):
    return Param(name=name, kind='float', default=default, label=label, unit='m', help=help_text, tip=tip,
                 min=0.0 if positive else None, min_exclusive=positive)


# description sentences shared by several kinds
_ORIGIN = ('Coordinates are physical, in m, with the origin at the lower-left corner of the physical domain '
           '(the PML lies outside).')
_FROM_ORIGIN = 'measured from the lower-left corner of the physical domain'
_ON_CELL_EDGE = 'A coordinate exactly on a cell edge goes to the upper cell.'
_CONSUMES_INPUTS = ('Composites (union / intersection / difference / array) consume their inputs, which can no longer '
                    'be selected on their own and may feed only one composite. Inputs are domain objects, composites '
                    'included.')
_CONSUMED_INPUT = 'it can no longer be selected on its own and may feed only one composite'
_TIP_FROM_ORIGIN = 'measured from the lower-left corner of the simulated region'   # GUI tips
_TIP_CENTRE_X = f'x of the centre, {_TIP_FROM_ORIGIN}.'
_TIP_CENTRE_Y = f'y of the centre, {_TIP_FROM_ORIGIN}.'

_COMPOSITE_INPUTS = Param(name='inputs', kind='names', default=(), label='Input objects', ref_kind='geometry',
                          help='Domain objects combined by this operation (at least one); each input is consumed: '
                               f'{_CONSUMED_INPUT}.',
                          tip='Objects combined by this operation.\n\n'
                              'Each then belongs to this object only: it can no longer be selected on its own or '
                              'used in another combination.')

_ONLY_3D = 'Only for 3D models (domain size_z set); it must stay null in a 2D model.'
_TIP_ONLY_3D = 'Used only when the model is 3D.'
# extrusion of the 2D domain kinds (prisms in 3D): interval along the axis (None = the full height, PML included,
# axis z only), the axis and the taper
_PRISM_3D_PARAMS = (
    Param(name='z0', kind='float', default=None, label='Extrusion start', unit='m', optional=True,
          help='Lower end of the extrusion along its axis (z, or x / y with axis x / y) in m; null = from the bottom '
               'of the domain (full height, PML included; axis z only). Cell centres with min(z0, z1) <= coordinate '
               '< max(z0, z1) are inside (half-open). ' + _ONLY_3D,
          tip="Where the object starts along its extrusion axis: its lowest height for the usual z axis.\n\n"
              "Leave empty to start at the bottom of the simulated region (z axis only). " + _TIP_ONLY_3D),
    Param(name='z1', kind='float', default=None, label='Extrusion end', unit='m', optional=True,
          help='Upper end of the extrusion along its axis in m; null = to the top of the domain (axis z only). '
               + _ONLY_3D,
          tip="Where the object ends along its extrusion axis: its highest height for the usual z axis.\n\n"
              "Leave empty to reach the top of the simulated region (z axis only). " + _TIP_ONLY_3D),
    Param(name='axis', kind='choice', default='z', label='Extrusion axis', choices=('z', 'x', 'y'),
          choice_labels=(('z', 'z (shape in the x-y plane)'), ('x', 'x (shape in the y-z plane)'),
                         ('y', 'y (shape in the x-z plane)')),
          help="Axis the 2D shape is extruded along. 'z' (default): the shape lies in the x-y plane. 'x' / 'y': its "
               "coordinates (x, y) are the model's (y, z) / (x, z), the ellipse angle turns from the first toward "
               "the second, and z0 / z1 are its ends along x / y, which must then both be set (beyond the domain to "
               "run into the PML): a circle with axis 'x' is a cylinder along x (holes through a slab, rods, "
               "wires). Faces parallel to the mesh axes become snap nodes on refined meshes; curved faces are "
               "staircased. 3D models only; it must stay 'z' in a 2D model.",
          tip="Direction the shape is extruded along.\n\n"
              "- z: the shape is drawn in the x-y plane (the usual case)\n"
              "- x or y: the shape's x and y are read as y, z (axis x) or x, z (axis y), and Extrusion start / "
              "end give its ends along that axis; both must be set\n\n"
              "A circle along x or y is a cylinder lying on its side. " + _TIP_ONLY_3D),
    Param(name='sidewall_angle', kind='float', default=0.0, label='Sidewall angle', unit='deg',
          help='Tilt of the side walls from the extrusion axis in degrees, -90 < angle < 90 (0 = straight walls). '
               'The shape as given sits at min(z0, z1) and is offset inward by (s - min(z0, z1)) tan(angle) up to '
               'the other end, s the coordinate along the axis: positive angles narrow it toward z1 (etched ridges, '
               'tapered holes, waveguide tapers along x / y), negative ones widen it (undercuts; V-grooves as a '
               'widening hole subtracted from the substrate). Rectangles shrink on every side, circles become '
               'cones, ellipses elliptic frusta (both semi-axes shrink by the same offset), polygons are offset per '
               'vertex (mitred; simple polygons only). Needs both z0 and z1; an offset that collapses or '
               'self-intersects the shape is an error. The slanted walls are staircased by the cells (one 2D raster '
               'per z cell); along x / y each cross-section is a polygon sampled in 256 steps along the axis '
               '(tapered rectangles exact, curved and polygon walls within a fraction of a step). 3D models '
               'only; it must stay 0 in a 2D model.',
          tip="Tilt of the side walls away from the extrusion axis.\n\n"
              "0 gives straight walls. A positive angle makes the object narrower toward Extrusion end (an etched "
              "ridge, a taper), a negative one wider. The shape as entered sits at Extrusion start.\n\n"
              "Needs Extrusion start and end set. " + _TIP_ONLY_3D))
_PRISM_3D = ("In 3D models it is extruded along 'axis' (z by default) between z0 and z1, optionally with tilted side "
             "walls (sidewall_angle).")
_PLANE_Z = Param(name='z', kind='float', default=None, label='z', unit='m', optional=True,
                 help='Height (z) of the point / the plane holding the line, in m; null = mid-height size_z / 2; on '
                      'a cell edge it goes to the upper cell. ' + _ONLY_3D,
                 tip='Height above the bottom of the simulated region.\n\nLeave empty for mid-height. '
                     + _TIP_ONLY_3D)
_TIP_CENTRE_Z = 'Height of the centre above the bottom of the simulated region.'

GEOMETRY_KINDS = {kind.name: kind for kind in (
    GeometryKind('rectangle', 'Rectangle', 'domain', (
        _length('x0', 1.5e-6, 'Corner 1, x',
                f'x of one corner, {_FROM_ORIGIN}; the corners may be given in any order.',
                f'x of one corner, {_TIP_FROM_ORIGIN}.\n\nAny two opposite corners work, in any order.'),
        _length('y0', 1.5e-6, 'Corner 1, y', f'y of the same corner as x0, {_FROM_ORIGIN}.',
                "y of the same corner as 'Corner 1, x'."),
        _length('x1', 2.5e-6, 'Corner 2, x', 'x of the opposite corner; cell centres with min(x0, x1) <= x < '
                                             'max(x0, x1) are inside (half-open).',
                'x of the opposite corner.'),
        _length('y1', 2.5e-6, 'Corner 2, y', 'y of the opposite corner; cell centres with min(y0, y1) <= y < '
                                             'max(y0, y1) are inside (half-open).',
                'y of the opposite corner.'), *_PRISM_3D_PARAMS),
        summary='Rectangle aligned with the x and y axes, given by two opposite corners.',
        description='Axis-aligned rectangle (domain object) between two opposite corners. Rectangle corners may be '
                    'given in any order. Shape inclusion tests cell centres: rectangles are half-open [x0, x1) x '
                    '[y0, y1) (corners sorted), so a centre exactly on x0 / y0 is inside and one on x1 / y1 is '
                    'outside. ' + _PRISM_3D + ' ' + _ORIGIN),
    GeometryKind('rounded_rectangle', 'Rounded rectangle', 'domain', (
        _length('x0', 1.5e-6, 'Corner 1, x',
                f'x of one corner of the enclosing rectangle, {_FROM_ORIGIN}; the corners may be given in any order.',
                f'x of one corner of the enclosing rectangle, {_TIP_FROM_ORIGIN}.\n\nAny two opposite corners work, '
                'in any order.'),
        _length('y0', 1.5e-6, 'Corner 1, y', f'y of the same corner as x0, {_FROM_ORIGIN}.',
                "y of the same corner as 'Corner 1, x'."),
        _length('x1', 2.5e-6, 'Corner 2, x', 'x of the opposite corner of the enclosing rectangle.',
                'x of the opposite corner.'),
        _length('y1', 2.5e-6, 'Corner 2, y', 'y of the opposite corner of the enclosing rectangle.',
                'y of the opposite corner.'),
        Param(name='r', kind='float', default=0.1e-6, label='Corner radius', unit='m', min=0.0,
              help='Radius of the four quarter-circle corners in m, 0 <= r <= half the shorter side (0 = a plain '
                   'rectangle; half the shorter side = a stadium; half of both sides of a square = a circle).',
              tip='Radius of the rounded corners.\n\n0 gives sharp corners; at most half the shorter side, which '
                  'makes the short ends fully round.'),
        *_PRISM_3D_PARAMS),
        summary='Rectangle aligned with the x and y axes with quarter-circle rounded corners.',
        description='Axis-aligned rectangle (domain object) between two opposite corners with its four corners rounded '
                    'to quarter circles of radius r (0 <= r <= half the shorter side; the same as a union of two '
                    'rectangles and four circles). Shape inclusion tests cell centres: half-open like rectangles '
                    'along the straight sides ([x0, x1) x [y0, y1), corners sorted), closed like circles on the '
                    'rounded corners. Triangle meshes follow the corner arcs exactly (P2 curved edges); tetrahedron '
                    'meshes (gmsh) export an untapered z prism as two boxes and four z cylinders, other axes and '
                    'tapers are tagged per element. ' + _PRISM_3D + ' ' + _ORIGIN),
    GeometryKind('circle', 'Circle', 'domain', (
        _length('cx', 2e-6, 'Centre x', f'x of the centre, {_FROM_ORIGIN}.', _TIP_CENTRE_X),
        _length('cy', 2e-6, 'Centre y', f'y of the centre, {_FROM_ORIGIN}.', _TIP_CENTRE_Y),
        _length('r', 0.5e-6, 'Radius', 'Radius (> 0); a cell centre exactly on the circle is inside (closed test).',
                'Radius of the circle, greater than 0.', positive=True), *_PRISM_3D_PARAMS),
        summary='Circle given by its centre and radius.',
        description='Circle (domain object) of radius r around (cx, cy). Shape inclusion tests cell centres: '
                    'circles are closed, so a centre exactly on the circle is inside. ' + _PRISM_3D + ' ' + _ORIGIN),
    GeometryKind('ellipse', 'Ellipse', 'domain', (
        _length('cx', 2e-6, 'Centre x', f'x of the centre, {_FROM_ORIGIN}.', _TIP_CENTRE_X),
        _length('cy', 2e-6, 'Centre y', f'y of the centre, {_FROM_ORIGIN}.', _TIP_CENTRE_Y),
        _length('a', 0.8e-6, 'First semi-axis', 'Semi-axis along the local x axis, which the rotation turns away '
                                                'from +x (> 0).',
                "Half the length of the first axis, which lies along +x when 'Rotation' is 0.", positive=True),
        _length('b', 0.4e-6, 'Second semi-axis', 'Semi-axis along the local y axis (> 0).',
                'Half the length of the second axis, at right angles to the first.', positive=True),
        Param(name='angle', kind='float', default=0.0, label='Rotation', unit='deg',
              help='Rotation of semi-axis a about the centre, in degrees counter-clockwise from +x; 0 puts a '
                   'along x and b along y.',
              tip='Counter-clockwise rotation of the first semi-axis away from the +x direction.'),
        *_PRISM_3D_PARAMS),
        summary='Ellipse given by its centre, two semi-axes and a rotation.',
        description='Ellipse (domain object) with semi-axes a and b about (cx, cy). The ellipse angle is in '
                    'degrees, counter-clockwise from +x. Shape inclusion tests cell centres: ellipses are closed, '
                    'so a centre exactly on the ellipse is inside. ' + _PRISM_3D + ' ' + _ORIGIN),
    GeometryKind('polygon', 'Polygon', 'domain', (
        Param(name='points', kind='points', default=((1.5e-6, 1.5e-6), (2.5e-6, 1.5e-6), (2e-6, 2.5e-6)),
              label='Vertices', unit='m',
              help=f'Vertices [(x, y), ...] {_FROM_ORIGIN}; at least 3, not all on one line. The last vertex '
                   'connects back to the first, and with the even-odd rule overlapping parts of a self-intersecting '
                   'polygon are holes.',
              tip='Corner points as x, y pairs separated by semicolons: at least 3, not all on one line.\n\n'
                  'The last point connects back to the first. Overlapping parts of a self-crossing polygon become '
                  'holes.'), *_PRISM_3D_PARAMS),
        summary='Closed shape through a list of corner points.',
        description='Polygon (domain object) through its vertices. Polygons are implicitly closed, use the even-odd '
                    'fill rule and need at least 3 points, not all on one line. Shape inclusion tests cell centres: '
                    'polygons are half-open like rectangles, so a centre exactly on a left / bottom edge is inside '
                    'and one on a right / top edge is outside. ' + _PRISM_3D + ' ' + _ORIGIN),
    GeometryKind('point', 'Point', 'point', (
        _length('x', 2e-6, 'x', f'x of the point, {_FROM_ORIGIN}; on a cell edge it goes to the upper cell.',
                f'x of the point, {_TIP_FROM_ORIGIN}.'),
        _length('y', 2e-6, 'y', f'y of the point, {_FROM_ORIGIN}; on a cell edge it goes to the upper cell.',
                f'y of the point, {_TIP_FROM_ORIGIN}.'), _PLANE_Z),
        summary='A point, used to place point sources and probes.',
        description='Zero-area point entity for sources, probes and monitors (point source, field probe, '
                    'temperature probe). ' + _ON_CELL_EDGE + ' ' + _ORIGIN),
    GeometryKind('line', 'Line segment', 'line', (
        _length('x0', 1e-6, 'Start x', f'x of the start point, {_FROM_ORIGIN}.',
                f'x of the start point, {_TIP_FROM_ORIGIN}.'),
        _length('y0', 2e-6, 'Start y', f'y of the start point, {_FROM_ORIGIN}.',
                f'y of the start point, {_TIP_FROM_ORIGIN}.'),
        _length('x1', 3e-6, 'End x', 'x of the end point; x1 = x0 makes a vertical line (EM line features '
                                     'need a horizontal or vertical line).',
                "x of the end point. Equal to 'Start x' gives a vertical line.\n\n"
                'Lines used by electromagnetic sources and monitors must be horizontal or vertical.'),
        _length('y1', 2e-6, 'End y', 'y of the end point; y1 = y0 makes a horizontal line. An end exactly on a '
                                     'cell edge goes to the upper cell.',
                "y of the end point. Equal to 'Start y' gives a horizontal line."), _PLANE_Z),
        summary='A straight segment, used to place line sources, beams and monitors.',
        description='Zero-area line segment for sources, probes and monitors (line source, gaussian beam, flux '
                    'monitor); lines used by EM features must be horizontal or vertical. ' + _ON_CELL_EDGE + ' '
                    + _ORIGIN),
    GeometryKind('union', 'Union', 'domain', (_COMPOSITE_INPUTS,), composite=True,
                 summary='Combines objects: a cell is inside if it is inside any of them.',
                 description='Union of domain objects: a cell centre inside any input is inside. ' + _CONSUMES_INPUTS),
    GeometryKind('intersection', 'Intersection', 'domain', (_COMPOSITE_INPUTS,), composite=True,
                 summary='Overlap of objects: a cell is inside only where all of them overlap.',
                 description='Intersection of domain objects: a cell centre must be inside every input. '
                             + _CONSUMES_INPUTS),
    GeometryKind('difference', 'Difference', 'domain', (
        Param(name='base', kind='ref', default=None, label='Base object', ref_kind='geometry',
              help=f'Domain object the others are subtracted from; it is consumed: {_CONSUMED_INPUT}.',
              tip='Object the others are cut out of.\n\n'
                  'It then belongs to this object only and can no longer be selected on its own.'),
        Param(name='subtract', kind='names', default=(), label='Objects to subtract', ref_kind='geometry',
              help=f'Domain objects removed from the base (may be empty); each is consumed: {_CONSUMED_INPUT}.',
              tip="Objects cut out of 'Base object'; the list can be empty.\n\n"
                  'They then belong to this object only and can no longer be selected on their own.')),
        composite=True,
        summary='One object with other objects cut out of it.',
        description='The base object minus the subtracted objects: a cell centre inside the base and outside every '
                    'subtracted object is inside. ' + _CONSUMES_INPUTS),
    GeometryKind('array', 'Array', 'domain', (
        Param(name='base', kind='ref', default=None, label='Base object', ref_kind='geometry',
              help='Domain object repeated by the array (a primitive or a composite); it is the first copy, at '
                   f'its own position, and is consumed: {_CONSUMED_INPUT}.',
              tip='Object repeated on the grid; it stays where it is as the first copy.\n\n'
                  'It then belongs to this array only and can no longer be selected on its own.'),
        Param(name='nx', kind='int', default=2, label='Copies along x', unit='1', min=1.0,
              help='Number of copies along x (>= 1), the base included.',
              tip='How many copies along x, counting the base object.'),
        Param(name='ny', kind='int', default=2, label='Copies along y', unit='1', min=1.0,
              help='Number of copies along y (>= 1), the base included.',
              tip='How many copies along y, counting the base object.'),
        Param(name='pitch_x', kind='float', default=1e-6, label='Pitch x', unit='m', min=0.0, min_exclusive=True,
              help='Displacement between neighbouring copies along x in m (> 0): copy (i, j) is the base moved by '
                   '(i pitch_x, j pitch_y). Copies whose bounding boxes overlap get a validation warning (overlapping '
                   'copies merge into one region).',
              tip='Distance from one copy to the next along x.'),
        Param(name='pitch_y', kind='float', default=1e-6, label='Pitch y', unit='m', min=0.0, min_exclusive=True,
              help='Displacement between neighbouring copies along y in m (> 0).',
              tip='Distance from one copy to the next along y.'),
        Param(name='nz', kind='int', default=1, label='Copies along z', unit='1', min=1.0,
              help='Number of copies along z (>= 1, the base included); above 1 it needs pitch_z. ' + _ONLY_3D
              + ' In 2D models it must stay 1.',
              tip='How many copies along z, counting the base object. ' + _TIP_ONLY_3D),
        Param(name='pitch_z', kind='float', default=None, label='Pitch z', unit='m', optional=True, min=0.0,
              min_exclusive=True,
              help='Displacement between neighbouring copies along z in m (> 0); needed when nz > 1. A base with an '
                   'open z end (z0 / z1 null) reaches the domain top / bottom in every copy, so such copies overlap. '
                   + _ONLY_3D,
              tip='Distance from one copy to the next along z; needed when there is more than one copy along z. '
                  + _TIP_ONLY_3D)),
        composite=True,
        summary='Repeats one object on a rectangular grid of copies (holes of a photonic crystal, pillar arrays).',
        description='Rectangular array of nx x ny (x nz in 3D) copies of the base object: copy (i, j, '
                    'k) is the base moved by (i pitch_x, j pitch_y, k pitch_z), copy (0, 0, 0) is the base itself; a '
                    'cell centre inside any copy is inside (a union of the copies). The array is one domain object: '
                    'materials, features, selections and mesh refinements naming it act on every copy; meshes follow '
                    "every copy's outline (triangles: curved edges on circles, ellipses and rounded-rectangle "
                    'arcs; tetrahedra (gmsh): the solids of each copy). Composite bases are copied whole (a hole '
                    'with a ring cut out, nested arrays). At most ' + f'{ARRAY_MAX_COPIES}' + ' copies. '
                    + _CONSUMES_INPUTS + ' ' + _ORIGIN),
    GeometryKind('box', 'Box', 'domain', (
        _length('x0', 1.5e-6, 'Corner 1, x', f'x of one corner, {_FROM_ORIGIN}; corners in any order.',
                f'x of one corner, {_TIP_FROM_ORIGIN}.'),
        _length('y0', 1.5e-6, 'Corner 1, y', 'y of the same corner as x0.', "y of the same corner as 'Corner 1, x'."),
        _length('z0', 1.5e-6, 'Corner 1, z', 'z of the same corner as x0.', "z of the same corner as 'Corner 1, x'."),
        _length('x1', 2.5e-6, 'Corner 2, x', 'x of the opposite corner.', 'x of the opposite corner.'),
        _length('y1', 2.5e-6, 'Corner 2, y', 'y of the opposite corner.', 'y of the opposite corner.'),
        _length('z1', 2.5e-6, 'Corner 2, z', 'z of the opposite corner; cell centres with min <= coordinate < max '
                                             'on each axis are inside (half-open).', 'z of the opposite corner.')),
        dimensions=(3,),
        summary='Box aligned with the x, y and z axes, given by two opposite corners (3D models).',
        description='Axis-aligned box (3D domain object) between two opposite corners, half-open on every axis '
                    'like rectangles; the same as a rectangle with z0 / z1. 3D models only. ' + _ORIGIN),
    GeometryKind('cylinder_z', 'Cylinder (z axis)', 'domain', (
        _length('cx', 2e-6, 'Centre x', f'x of the axis, {_FROM_ORIGIN}.', _TIP_CENTRE_X),
        _length('cy', 2e-6, 'Centre y', f'y of the axis, {_FROM_ORIGIN}.', _TIP_CENTRE_Y),
        _length('r', 0.5e-6, 'Radius', 'Radius (> 0); closed test like circles.', 'Radius, greater than 0.',
                positive=True),
        _length('z0', 1.5e-6, 'Bottom z', 'One end of the cylinder along z (ends in any order; half-open in z).',
                'Height of one end.'),
        _length('z1', 2.5e-6, 'Top z', 'The other end along z.', 'Height of the other end.')),
        dimensions=(3,),
        summary='Cylinder standing along z, given by its axis position, radius and end heights (3D models).',
        description='Cylinder with its axis along z (3D domain object): a circle extruded from z0 to z1 (a circle '
                    "with axis 'x' / 'y' gives a cylinder along x / y, sidewall_angle a cone). 3D models only. "
                    + _ORIGIN),
    GeometryKind('sphere', 'Sphere', 'domain', (
        _length('cx', 2e-6, 'Centre x', f'x of the centre, {_FROM_ORIGIN}.', _TIP_CENTRE_X),
        _length('cy', 2e-6, 'Centre y', f'y of the centre, {_FROM_ORIGIN}.', _TIP_CENTRE_Y),
        _length('cz', 2e-6, 'Centre z', 'z of the centre, measured from the bottom of the physical domain.',
                _TIP_CENTRE_Z),
        _length('r', 0.5e-6, 'Radius', 'Radius (> 0); a cell centre exactly on the sphere is inside (closed test).',
                'Radius of the sphere, greater than 0.', positive=True)),
        dimensions=(3,),
        summary='Sphere given by its centre and radius (3D models).',
        description='Sphere (3D domain object) of radius r around (cx, cy, cz); closed test like circles. The 2D '
                    'canvas draws its cross-section at the shown layer. 3D models only. ' + _ORIGIN),
)}
PRISM_KINDS = ('rectangle', 'rounded_rectangle', 'circle', 'ellipse', 'polygon')   # 2D kinds extruded in 3D
_QUARTER_TURN_STARTS = (-0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi)   # corner arcs, counter-clockwise from (x1, y0)


def rounded_corners(values) -> tuple:
    """((cx, cy, t_start), ...) of a rounded_rectangle's four quarter-circle corners (sorted corners, radius r, arc
    t_start .. t_start + pi / 2), counter-clockwise from the (x1, y0) corner."""
    x0, y0, x1, y1, radius = values['x0'], values['y0'], values['x1'], values['y1'], values['r']
    centres = ((x1 - radius, y0 + radius), (x1 - radius, y1 - radius), (x0 + radius, y1 - radius),
               (x0 + radius, y0 + radius))
    return tuple((cx, cy, start) for (cx, cy), start in zip(centres, _QUARTER_TURN_STARTS))


def _rounded_outline_points(values, n_per_arc) -> np.ndarray:
    """Float64 (4 (n_per_arc + 1), 2) open outline of a rounded_rectangle, counter-clockwise: every corner arc sampled
    at n_per_arc + 1 points (r = 0: repeated corner points), so the count never depends on r."""
    angles = np.linspace(0.0, 0.5 * math.pi, int(n_per_arc) + 1)
    radius = values['r']
    return np.concatenate([np.column_stack([cx + radius * np.cos(start + angles), cy + radius * np.sin(start + angles)])
                           for cx, cy, start in rounded_corners(values)])


@dataclass(frozen=True)
class ResolvedShape:
    name: str
    kind: str
    values: dict              # evaluated floats; rectangle corners sorted (x0 <= x1, y0 <= y1); prism 'axis' text


def _axis(values) -> str:
    """Extrusion axis of a prism's values ('z' when absent: internal plain shapes carry none)."""
    return values.get('axis') or 'z'


def is_plain(shape) -> bool:
    """True unless shape is a prism extruded along x / y or with tilted side walls: a plain object's xy cross-section
    is its 2D shape wherever its z interval holds it."""
    values = shape.values
    return shape.kind not in PRISM_KINDS or (_axis(values) == 'z' and not values.get('sidewall_angle'))


# ---- entity listings ----
def composite_inputs(kind, values) -> list:
    """Object names a composite reads ('names' and 'ref' params, defaults when absent), in order; a resolved array
    (values from resolve_geometry): its copies, the base first."""
    if kind.name == 'array' and isinstance(values.get('inputs'), list):
        return list(values['inputs'])
    inputs = []
    for param in kind.params:
        value = values.get(param.name, param.default)
        if param.kind == 'names' and isinstance(value, (list, tuple)):
            inputs.extend(item for item in value if isinstance(item, str) and item)
        elif param.kind == 'ref' and isinstance(value, str) and value:
            inputs.append(value)
    return inputs


def primitive_names(name, shapes) -> list:
    """Names of the plain shapes under resolved object name: itself, or its composite inputs recursively (inputs
    missing from shapes skipped)."""
    kind = GEOMETRY_KINDS[shapes[name].kind]
    if not kind.composite:
        return [name]
    return [primitive for item in composite_inputs(kind, shapes[name].values) if item in shapes
            for primitive in primitive_names(item, shapes)]


def consumed_objects(spec) -> set:
    """Names used as inputs of composite objects (no longer selectable on their own)."""
    consumed = set()
    for obj in spec.geometry:
        kind = GEOMETRY_KINDS.get(obj.kind)
        if kind is not None and kind.composite and isinstance(obj.params, dict):
            consumed.update(composite_inputs(kind, obj.params))
    return consumed


def _objects_of_kind(spec, wanted_entity_kind, exclude=frozenset()) -> list:
    return [obj.name for obj in spec.geometry
            if obj.kind in GEOMETRY_KINDS and GEOMETRY_KINDS[obj.kind].entity_kind == wanted_entity_kind
            and obj.name not in exclude]


def domain_entities(spec) -> list:
    return ['all'] + _objects_of_kind(spec, 'domain', consumed_objects(spec))


def boundary_entities(spec) -> list:
    return list(outer_faces(spec)) + _objects_of_kind(spec, 'domain', consumed_objects(spec))


def interface_entities(spec) -> list:
    """Domain objects whose outline an 'interface' selection may hold (no 'all', no outer faces)."""
    return _objects_of_kind(spec, 'domain', consumed_objects(spec))


def point_entities(spec) -> list:
    return _objects_of_kind(spec, 'point')


def line_entities(spec) -> list:
    return _objects_of_kind(spec, 'line')


def expand_entities(spec, entities) -> list:
    """Replace named selections by their entities (recursively, dedup, order kept); ValueError on a cycle."""
    selections = {selection.name: selection for selection in spec.selections}
    expanded, seen = [], set()

    def visit(name, path):
        if not isinstance(name, str):
            raise ValueError(f'entity names must be text, got {name!r}')
        selection = selections.get(name)
        if selection is None:
            if name not in seen:
                seen.add(name)
                expanded.append(name)
            return
        if name in path:
            raise ValueError('selection cycle: ' + ' -> '.join(path[path.index(name):] + [name]))
        for child in selection.entities:
            visit(child, path + [name])

    for name in entities:
        visit(name, [])
    return expanded


def material_selections(spec, failed=None) -> list:
    """expand_entities of each spec material's selection, in order; `failed` for one that fails to expand (validate
    reports it)."""
    selections = []
    for material in spec.materials:
        try:
            selections.append(expand_entities(spec, material.selection))
        except (ValueError, TypeError):
            selections.append(failed)
    return selections


def line_orientation(values, size_x, size_y):
    """'horizontal' | 'vertical' when the segment is axis-aligned within AXIS_TOLERANCE, else None."""
    tolerance = AXIS_TOLERANCE * max(float(size_x), float(size_y))
    span_x = abs(values['x1'] - values['x0'])
    span_y = abs(values['y1'] - values['y0'])
    if span_y <= tolerance < span_x:
        return 'horizontal'
    if span_x <= tolerance < span_y:
        return 'vertical'
    return None


# ---- evaluation ----
def domain_size(spec, parameters) -> tuple:
    """Evaluated (size_x, size_y) [m] of the physical domain (a 1D model: size_y = LINE_SPANS of its coordinates, the
    logical span of its one cell across); ValueError (ExpressionError included) unless both > 0."""
    size_x = evaluate_scalar(spec.domain.size_x, parameters)
    if spec.domain.size_y is None:
        if spec.domain.coordinates not in LINE_SPANS:
            raise ValueError(f"domain coordinates must be one of {', '.join(LINE_SPANS)}, got "
                             f'{short_repr(spec.domain.coordinates)}')
        size_y = LINE_SPANS[spec.domain.coordinates]
    else:
        size_y = evaluate_scalar(spec.domain.size_y, parameters)
    if not (size_x > 0 and size_y > 0):
        raise ValueError(f'domain size must be > 0, got ({size_x:g}, {size_y:g}) m')
    return size_x, size_y


def domain_size_z(spec, parameters):
    """Evaluated size_z [m] of a 3D model (> 0, else ValueError), None for a 2D model (DomainSpec.size_z None)."""
    if spec.domain.size_z is None:
        return None
    size_z = evaluate_scalar(spec.domain.size_z, parameters)
    if not size_z > 0:
        raise ValueError(f'domain size_z must be > 0 (or null for a 2D model), got {size_z:g} m')
    return size_z


# ---- domain coordinates (symmetry plan 2.16) ----
DOMAIN_COORDINATES = ('cartesian', 'polar', 'spherical')
# 1D models (DomainSpec.size_y null, symmetry 2.20): logical y span of the one cell across ([m] Cartesian: totals per
# m^2; [rad] the full circle of a cylinder per m of length, theta 0..pi of a whole sphere)
LINE_SPANS = {'cartesian': 1.0, 'polar': 2.0 * math.pi, 'spherical': math.pi}
LINE_LABELS = {'cartesian': '1D slab (x)', 'polar': '1D cylindrical (r)', 'spherical': '1D spherical (r)'}
_QUARTER_TURN = 0.5 * math.pi


@dataclass(frozen=True)
class DomainMap:
    """In-plane grid of a 2D model or of a 3D model's xy plane (DomainSpec.coordinates; 3D polar = a cylindrical
    (r, phi, z) sector, z unmapped): 'cartesian', or a polar window drawn at its mapped
    position ('polar': x = r cos phi, y = r sin phi; 'spherical': the meridional half plane, x = r sin theta the
    distance to the rotation axis y, y = r cos theta). size_x / size_y = logical window sizes ([m]; [rad] on an angle);
    angle_wrap (polar): core.metric.ANGLE_WRAPS, y_hi joined to y_lo (no outer faces there). line (1D model): one cell
    across (size_y = LINE_SPANS), drawn on the x axis of the drawing plane: x = radius_min + the logical x (r, or the
    slab coordinate), y = 0; mapped on every kind."""
    kind: str
    size_x: float
    size_y: float
    radius_min: float = 0.0
    angle_min: float = 0.0
    angle_wrap: str = 'none'
    line: bool = False

    @property
    def mapped(self) -> bool:
        return self.kind != 'cartesian' or self.line

    @property
    def grid_text(self) -> str:
        """'1D slab (x)' ... of a line, else the coordinates kind (messages)."""
        return LINE_LABELS[self.kind] if self.line else self.kind

    @property
    def axisymmetric(self) -> bool:
        """True for 'spherical': every physics solves the window as rings about the y axis."""
        return self.kind == 'spherical'

    @property
    def radius_max(self) -> float:
        return self.radius_min + self.size_x

    @property
    def angle_max(self) -> float:
        return self.angle_min + self.size_y

    def coordinates(self) -> Coordinates:
        """core.metric.Coordinates of the Domain ('polar' for both polar kinds; line set on a 1D model)."""
        if self.kind == 'cartesian':
            return Coordinates(line=True) if self.line else CARTESIAN
        return Coordinates('polar', self.radius_min, self.angle_min, self.angle_wrap, line=self.line)

    def wrapped_faces(self) -> tuple:
        """('y_lo', 'y_hi') when a phi wrap joins them (no boundary there: features may not select them), else ()."""
        return ('y_lo', 'y_hi') if self.angle_wrap != 'none' else ()

    def to_physical(self, x_logical, y_logical, xp=np):
        """Drawing-plane (x, y) [m] of logical window positions (identity on a Cartesian grid; a line: (radius_min +
        x, 0), broadcast over both inputs)."""
        if self.line:
            return self.radius_min + x_logical + 0.0 * y_logical, 0.0 * (x_logical + y_logical)
        if not self.mapped:
            return x_logical, y_logical
        return to_cartesian(self.coordinates(), self.axisymmetric, x_logical, y_logical, xp)

    def to_logical(self, x, y):
        """Logical window (x, y) (r - radius_min, angle - angle_min on a mapped grid; a line: x - radius_min and the
        centre of its one cell) of drawing points (numpy)."""
        if self.line:
            x, _ = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
            return x - self.radius_min, np.full(x.shape, 0.5 * self.size_y)
        if not self.mapped:
            return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return from_cartesian(self.coordinates(), self.axisymmetric, self.size_y, x, y)

    def contains(self, x, y, tolerance=0.0):
        """Bool (numpy, shape of x) of drawing points inside the closed window, within tolerance [m] (a line: on the
        x axis, |y| <= tolerance)."""
        if self.line:
            x_logical = np.asarray(x, dtype=np.float64) - self.radius_min
            return ((x_logical >= -tolerance) & (x_logical <= self.size_x + tolerance)
                    & (np.abs(np.asarray(y, dtype=np.float64)) <= tolerance))
        x_logical, y_logical = self.to_logical(x, y)
        angle_tolerance = (tolerance / np.maximum(x_logical + self.radius_min, 1e-300)) if self.mapped else tolerance
        return ((x_logical >= -tolerance) & (x_logical <= self.size_x + tolerance)
                & (y_logical >= -angle_tolerance) & (y_logical <= self.size_y + angle_tolerance))

    def face_polylines(self, n_segments=64) -> dict:
        """{outer face: (N, 2) drawing-plane polyline [m]}: rays (y_lo / y_hi) and arcs (x_lo / x_hi) when mapped; a
        line: its two end points (x_lo / x_hi) as ticks across the strip of extent()."""
        if self.line:
            half = 0.5 * self.size_x
            return {face: np.array([[r, -half], [r, half]])
                    for face, r in (('x_lo', self.radius_min), ('x_hi', self.radius_max))}
        if not self.mapped:
            sx, sy = self.size_x, self.size_y
            return {'x_lo': np.array([[0.0, 0.0], [0.0, sy]]), 'x_hi': np.array([[sx, 0.0], [sx, sy]]),
                    'y_lo': np.array([[0.0, 0.0], [sx, 0.0]]), 'y_hi': np.array([[0.0, sy], [sx, sy]])}
        angles = np.linspace(0.0, self.size_y, max(2, int(n_segments * self.size_y / _QUARTER_TURN) + 2))
        quarter_turns = np.arange(math.ceil(self.angle_min / _QUARTER_TURN),
                                  math.floor(self.angle_max / _QUARTER_TURN) + 1) * _QUARTER_TURN - self.angle_min
        inner_turns = quarter_turns[(quarter_turns > 0.0) & (quarter_turns < self.size_y)]   # exact arc extremes
        angles = np.unique(np.concatenate((angles, inner_turns)))
        radii = np.array([0.0, self.size_x])

        def polyline(x_logical, y_logical):
            return np.column_stack(self.to_physical(x_logical, y_logical))
        return {'x_lo': polyline(np.zeros_like(angles), angles),
                'x_hi': polyline(np.full_like(angles, self.size_x), angles),
                'y_lo': polyline(radii, np.zeros(2)), 'y_hi': polyline(radii, np.full(2, self.size_y))}

    def outline(self, n_segments=64) -> np.ndarray:
        """Closed (N, 2) drawing-plane polyline [m] of the window boundary (a line: the strip of extent())."""
        if self.line:
            x0, x1, y0, y1 = self.extent()
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        faces = self.face_polylines(n_segments)
        return np.concatenate((faces['y_lo'], faces['x_hi'], faces['y_hi'][::-1], faces['x_lo'][::-1],
                               faces['y_lo'][:1]))

    def extent(self) -> tuple:
        """(x_min, x_max, y_min, y_max) [m] of the window in the drawing plane (the sector's bounding box; a line: the
        strip |y| <= size_x / 2 about its x axis, so previews and line profiles get a finite height)."""
        if self.line:
            return float(self.radius_min), float(self.radius_max), -0.5 * float(self.size_x), 0.5 * float(self.size_x)
        if not self.mapped:
            return 0.0, float(self.size_x), 0.0, float(self.size_y)
        points = self.outline()
        return (float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()),
                float(points[:, 1].max()))

    def logical_bounds(self, box) -> tuple:
        """Logical (x_lo, x_hi, y_lo, y_hi) window range, clipped to the window, covering the drawing-plane box (x0, y0,
        x1, y1) [m]: r from its nearest point to its farthest corner, the angle over its corners (all angles when it
        holds the origin; a box across the angle wrap opposite the sector gets the whole range: an upper bound)."""
        if not self.mapped:
            raise ValueError('logical_bounds maps polar / spherical windows; Cartesian boxes are already logical')
        x0, y0, x1, y1 = (float(value) for value in box)
        if self.line:   # the box's x range; the one cell across whole
            return (min(max(x0 - self.radius_min, 0.0), self.size_x), min(max(x1 - self.radius_min, 0.0), self.size_x),
                    0.0, self.size_y)
        corners_x, corners_y = np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1])
        nearest = math.hypot(min(max(0.0, x0), x1), min(max(0.0, y0), y1))
        farthest = float(np.max(np.hypot(corners_x, corners_y)))
        if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
            angle_lo, angle_hi = 0.0, self.size_y
        else:
            _, angles = self.to_logical(corners_x, corners_y)
            angle_lo, angle_hi = float(angles.min()), float(angles.max())

        def clip(value, size):
            return min(max(value, 0.0), size)
        return (clip(nearest - self.radius_min, self.size_x), clip(farthest - self.radius_min, self.size_x),
                clip(angle_lo, self.size_y), clip(angle_hi, self.size_y))

    def zero_area_faces(self) -> tuple:
        """Outer faces without area (their BCs do nothing): the origin x_lo (radius_min 0), spherical axis faces (a
        line: the centre x_lo of a cylinder / sphere with radius_min 0; y_lo / y_hi are no faces of a 1D model)."""
        if self.line:
            return ('x_lo',) if self.kind != 'cartesian' and self.radius_min == 0.0 else ()
        if not self.mapped:
            return ()
        faces = ('x_lo',) if self.radius_min == 0.0 else ()
        if self.axisymmetric:
            faces += (('y_lo',) if abs(self.angle_min) <= AXIS_ANGLE_SNAP else ()) + (
                ('y_hi',) if abs(self.angle_max - math.pi) <= AXIS_ANGLE_SNAP else ())
        return faces


def domain_map(spec, parameters) -> DomainMap:
    """Evaluated DomainMap of the spec; ValueError (ExpressionError included) for a bad size or coordinates value, a
    spherical 3D model, radius_min < 0, an angle range beyond 2 pi (polar) / 0..pi (spherical) or an angle_wrap off a
    polar domain; size_y null: the 1D line map (_line_domain_map). Polar + size_z = a cylindrical (r, phi, z) sector
    (symmetry 2.21): the map is the (r, phi) plane, z passes through."""
    size_x, size_y = domain_size(spec, parameters)
    kind = spec.domain.coordinates
    if kind not in DOMAIN_COORDINATES:
        raise ValueError(f"domain coordinates must be one of {', '.join(DOMAIN_COORDINATES)}, got {short_repr(kind)}")
    angle_wrap = spec.domain.angle_wrap
    if angle_wrap not in ANGLE_WRAPS:
        raise ValueError(f"domain angle_wrap must be one of {', '.join(ANGLE_WRAPS)}, got {short_repr(angle_wrap)}")
    if angle_wrap != 'none' and kind != 'polar':
        raise ValueError(f"domain angle_wrap {angle_wrap!r} joins the angle faces of a polar domain, not of a {kind} "
                         "one: set it to 'none'")
    if spec.domain.size_y is None:
        return _line_domain_map(spec, parameters, kind, size_x, size_y)
    if kind == 'cartesian':
        return DomainMap(kind, size_x, size_y)
    if spec.domain.size_z is not None and kind != 'polar':
        raise ValueError(f"domain coordinates '{kind}' need a 2D model: clear size_z (a 3D model takes cartesian, or "
                         'polar = a cylindrical (r, phi, z) sector)')
    radius_min = evaluate_scalar(spec.domain.radius_min, parameters)
    angle_min = evaluate_scalar(spec.domain.angle_min, parameters)
    if not radius_min >= 0.0:
        raise ValueError(f'domain radius_min must be >= 0 m, got {radius_min:g}')
    if kind == 'polar' and size_y > 2.0 * math.pi + AXIS_ANGLE_SNAP:
        raise ValueError(f'a polar domain spans at most 2 pi in phi: size_y = {size_y:.6g} rad (> 6.283185)')
    if kind == 'spherical' and (angle_min < -AXIS_ANGLE_SNAP or angle_min + size_y > math.pi + AXIS_ANGLE_SNAP):
        raise ValueError(f'a spherical domain needs 0 <= theta <= pi: angle_min = {angle_min:.6g} rad, angle_min + '
                         f'size_y = {angle_min + size_y:.6g} rad')
    return DomainMap(kind, size_x, size_y, radius_min, angle_min, angle_wrap)


def _line_domain_map(spec, parameters, kind, size_x, size_y) -> DomainMap:
    """DomainMap of a 1D model (size_y null); ValueError with size_z set, an angle_wrap, angle_min != 0, radius_min < 0
    or radius_min != 0 on a slab."""
    domain = spec.domain
    if domain.size_z is not None:
        raise ValueError('a 1D model (size_y null) has no size_z: clear it, or set size_y for a 3D model')
    if domain.angle_wrap != 'none':
        raise ValueError(f"a 1D model spans the whole angle: set angle_wrap 'none', got {domain.angle_wrap!r}")
    angle_min = evaluate_scalar(domain.angle_min, parameters)
    if angle_min != 0.0:
        raise ValueError(f'a 1D model spans the whole angle: set angle_min 0, got {angle_min:g} rad')
    radius_min = evaluate_scalar(domain.radius_min, parameters)
    if not radius_min >= 0.0:
        raise ValueError(f'domain radius_min must be >= 0 m, got {radius_min:g}')
    if kind == 'cartesian' and radius_min != 0.0:
        raise ValueError(f'a 1D slab runs from x = 0 to size_x: radius_min applies to polar / spherical 1D models, got '
                         f'{radius_min:g} m')
    return DomainMap(kind, size_x, size_y, radius_min, line=True)


def domain_extent(spec, parameters) -> tuple:
    """(x_min, x_max, y_min, y_max) [m] of the physical window in the drawing plane: (0, size_x, 0, size_y) on a
    Cartesian grid, the sector's bounding box on a polar / spherical one; ValueError as domain_map."""
    return domain_map(spec, parameters).extent()


def _resolve_object_values(obj, parameters) -> dict:
    """Evaluate one object's params (resolve_values checks types and r, a, b > 0) and check its shape; z0 / z1 come
    back sorted when both are set."""
    kind = GEOMETRY_KINDS.get(obj.kind)
    if kind is None:
        raise ValueError(f"unknown geometry kind {obj.kind!r}; valid: {', '.join(GEOMETRY_KINDS)}")
    if not isinstance(obj.params, dict):
        raise ValueError(f'params must be a dict, got {short_repr(obj.params)}')
    values = dict(resolve_values(kind.params, obj.params, parameters))
    if values.get('z0') is not None and values.get('z1') is not None:
        z0, z1 = sorted((values['z0'], values['z1']))
        if z0 == z1:
            raise ValueError(f"{kind.name} has zero thickness along {values.get('axis') or 'z'} "
                             f"(z0 = z1 = {z0:g} m)")
        values.update(z0=z0, z1=z1)
    if kind.name in ('rectangle', 'box', 'rounded_rectangle'):
        x0, x1 = sorted((values['x0'], values['x1']))
        y0, y1 = sorted((values['y0'], values['y1']))
        if x0 == x1 or y0 == y1:
            raise ValueError(f'{kind.name} has zero width or height (x from {x0:g} to {x1:g} m, '
                             f'y from {y0:g} to {y1:g} m)')
        values.update(x0=x0, y0=y0, x1=x1, y1=y1)
        half_side = 0.5 * min(x1 - x0, y1 - y0)
        if kind.name == 'rounded_rectangle' and values['r'] > half_side * (1.0 + _RADIUS_SLACK):
            raise ValueError(f"rounded_rectangle corner radius r = {values['r']:g} m exceeds half the shorter side "
                             f"({half_side:g} m)")
        if kind.name == 'rounded_rectangle':   # a stadium / circle typed with rounded corners stays one exactly
            values['r'] = min(values['r'], half_side)
    elif kind.name == 'polygon' and len(values['points']) < 3:
        raise ValueError(f"polygon needs at least 3 points, got {len(values['points'])}")
    elif kind.name == 'polygon' and _points_on_one_line(values['points']):
        raise ValueError('polygon has zero area (all vertices lie on one line)')
    elif kind.name == 'line' and values['x0'] == values['x1'] and values['y0'] == values['y1']:
        raise ValueError(f"line has zero length (both ends at ({values['x0']:g}, {values['y0']:g}) m)")
    elif kind.composite:
        empty_param = next((param.name for param in kind.params
                            if param.kind == 'names' and '' in values.get(param.name, ())), None)
        if empty_param is not None:
            raise ValueError(f'{empty_param}: empty object name')
        if not composite_inputs(kind, values):
            raise ValueError('array needs a base object (base)' if kind.name == 'array'
                             else f'{kind.name} needs at least one input object')
        if kind.name == 'array':
            _check_array(values)
    if kind.name in PRISM_KINDS:
        _check_prism_3d(kind.name, values)
    return values


def _check_array(values):
    """ValueError unless an array's counts fit: pitch_z set when nz > 1, at most ARRAY_MAX_COPIES copies."""
    if values['nz'] > 1 and values.get('pitch_z') is None:
        raise ValueError(f"nz = {values['nz']} copies along z need pitch_z (the displacement between them)")
    n_copies = values['nx'] * values['ny'] * values['nz']
    if n_copies > ARRAY_MAX_COPIES:
        raise ValueError(f'array of {n_copies} copies exceeds the limit of {ARRAY_MAX_COPIES}; split it or use a '
                         f'coarser pitch')


def _array_copies(name, values, shapes) -> dict:
    """{name: ResolvedShape} of every new shape of array `name` (resolved base in shapes): copy (i, j, k) of the base
    moved by (i pitch_x, j pitch_y, k pitch_z) named '<array>[i,j]' ('<array>[i,j,k]' when nz > 1; composite bases
    also copy their inputs, '<input>@<copy name>'), post-order; sets values['inputs'] = the copies, base first, i
    fastest, then j, then k."""
    n_x, n_y, n_z = values['nx'], values['ny'], values['nz']
    pitch_x, pitch_y, pitch_z = values['pitch_x'], values['pitch_y'], values.get('pitch_z') or 0.0
    added, inputs = {}, []
    for k in range(n_z):
        for j in range(n_y):
            for i in range(n_x):
                if i == j == k == 0:
                    inputs.append(values['base'])
                    continue
                copy_name = f'{name}[{i},{j}]' if n_z == 1 else f'{name}[{i},{j},{k}]'
                offset = (i * pitch_x, j * pitch_y, k * pitch_z)
                inputs.append(_moved_copy(values['base'], shapes, offset, copy_name, copy_name, added))
    values['inputs'] = inputs
    return added


def is_array_copy(name) -> bool:
    """True for the names resolve_geometry gives array copies (never a valid object name: they hold '[')."""
    return '[' in name


def _moved_copy(source, shapes, offset, copy_name, suffix, added) -> str:
    """copy_name, after adding to `added` the shape `source` moved by offset (dx, dy, dz) [m]; a composite's inputs
    are moved too, as '<input>@<suffix>' (once each)."""
    if copy_name in added:
        return copy_name
    shape = shapes[source] if source in shapes else added[source]
    values = dict(shape.values)
    if GEOMETRY_KINDS[shape.kind].composite:
        for key in ('inputs', 'subtract'):
            if isinstance(values.get(key), list):
                values[key] = [_moved_copy(item, shapes, offset, f'{item}@{suffix}', suffix, added)
                               for item in values[key]]
        if values.get('base'):
            values['base'] = _moved_copy(values['base'], shapes, offset, f"{values['base']}@{suffix}", suffix, added)
    else:
        values = _moved_values(shape, offset)
    added[copy_name] = ResolvedShape(copy_name, shape.kind, values)
    return copy_name


def _moved_values(shape, offset) -> dict:
    """Values of a primitive moved by offset (dx, dy, dz) [m], each coordinate value + its axis offset (prisms along x
    / y: the plane coordinates by geometry.plane_axes, z0 / z1 along the axis)."""
    values = dict(shape.values)
    shift = dict(zip('xyz', offset))
    if shape.kind in PRISM_KINDS:
        along = _axis(values)
        first, second = _PLANE_AXES[along]
    else:   # box, cylinder_z, sphere: model coordinates
        along, first, second = 'z', 'x', 'y'
    if shape.kind in ('rectangle', 'rounded_rectangle', 'box'):
        values.update(x0=values['x0'] + shift[first], x1=values['x1'] + shift[first],
                      y0=values['y0'] + shift[second], y1=values['y1'] + shift[second])
    elif shape.kind == 'polygon':
        values['points'] = [(point_x + shift[first], point_y + shift[second]) for point_x, point_y in values['points']]
    else:   # circle, ellipse, cylinder_z, sphere
        values.update(cx=values['cx'] + shift[first], cy=values['cy'] + shift[second])
        if shape.kind == 'sphere':
            values['cz'] = values['cz'] + shift['z']
    for end in ('z0', 'z1'):
        if values.get(end) is not None:
            values[end] = values[end] + shift[along]
    return values


def array_overlap(name, resolved):
    """Warning text when neighbouring copies of resolved array `name` have overlapping bounding boxes (they may merge
    into one region), else None."""
    values = _shape(name, resolved).values
    base = values['inputs'][0]
    x0, y0, x1, y1 = bounding_box(base, resolved)
    z_lo, z_hi = z_extent(base, resolved)
    spans = (('x', x1 - x0, values['pitch_x'], values['nx']), ('y', y1 - y0, values['pitch_y'], values['ny']),
             ('z', z_hi - z_lo, values.get('pitch_z'), values['nz']))
    overlaps = [f'{axis} (bounding box {width:.4g} m, pitch_{axis} {pitch:.4g} m)'
                for axis, width, pitch, count in spans if count > 1 and width > pitch]
    if not overlaps:
        return None
    return (f"copies of '{base}' overlap along {' and '.join(overlaps)}: overlapping copies merge into one region; "
            f"raise the pitch if they should stay apart")


def _check_prism_3d(kind_name, values):
    """ValueError unless the prism's 3D params fit: axis x / y with both ends set; a sidewall angle in (-90, 90) deg
    with both ends and an end section that neither collapses nor (polygons) self-intersects."""
    axis, angle = values['axis'], values['sidewall_angle']
    ends_set = values.get('z0') is not None and values.get('z1') is not None
    if axis != 'z' and not ends_set:
        raise ValueError(f"axis {axis!r} needs z0 and z1, the ends along {axis} (set them beyond the domain to run "
                         f"into the PML)")
    if not angle:
        return
    if not (math.isfinite(angle) and abs(angle) < 90.0):
        raise ValueError(f'sidewall_angle must lie strictly between -90 and 90 deg, got {angle:g}')
    if not ends_set:
        raise ValueError('sidewall_angle needs z0 and z1: the taper runs from the shape as given at the lower end '
                         'to the upper one')
    top_offset = (values['z1'] - values['z0']) * math.tan(math.radians(angle))
    if kind_name in ('rectangle', 'rounded_rectangle'):
        limit = 0.5 * min(values['x1'] - values['x0'], values['y1'] - values['y0'])
    else:
        limit = {'circle': lambda: values['r'], 'ellipse': lambda: min(values['a'], values['b']),
                 'polygon': lambda: math.inf}[kind_name]()
    if top_offset >= limit:
        raise ValueError(f'sidewall_angle {angle:g} deg collapses the {kind_name} before its top: the inward offset '
                         f'at z1 would be {top_offset:g} m and must stay below {limit:g} m; lower the angle or the '
                         f'height')
    if kind_name == 'polygon':
        _check_polygon_taper(values['points'], top_offset)


def _check_polygon_taper(points, top_offset):
    """ValueError when a polygon cannot be offset by up to top_offset [m] (mitred): it crosses itself, an edge
    vanishes or reverses, or the top section crosses itself (offsets are linear, so the top decides)."""
    vertices = np.asarray(points, dtype=np.float64)
    if _polygon_crosses_itself(vertices):
        raise ValueError('sidewall_angle needs a simple polygon (no crossing edges)')
    mitres = _inward_mitres(vertices)
    edges = np.roll(vertices, -1, axis=0) - vertices
    top_edges = edges + top_offset * (np.roll(mitres, -1, axis=0) - mitres)
    shrunk = np.flatnonzero(np.sum(top_edges * edges, axis=1) <= 0.0)
    if shrunk.size:
        raise ValueError(f'sidewall_angle collapses polygon edge {int(shrunk[0]) + 1} (vertex {int(shrunk[0]) + 1} '
                         f'to the next) before z1; lower the angle or the height')
    if _polygon_crosses_itself(vertices + top_offset * mitres):
        raise ValueError('sidewall_angle makes the polygon cross itself before z1; lower the angle or the height')


def _inward_mitres(vertices) -> np.ndarray:
    """(n, 2) float64 mitre vectors of a closed polygon's (n, 2) vertices: vertex i + d * mitre i offsets both adjacent
    edges inward by d [m] (orientation from the signed area); ValueError on a zero-length edge or a vertex turning
    back on itself."""
    edges = np.roll(vertices, -1, axis=0) - vertices    # edge i runs from vertex i to vertex i + 1
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if np.any(lengths <= 1e-9 * lengths.max()):
        raise ValueError(f'polygon vertex {int(np.argmin(lengths)) + 1} repeats the next one; remove it')
    signed_area = 0.5 * float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                                     - np.roll(vertices[:, 0], -1) * vertices[:, 1]))
    normals = np.sign(signed_area) * np.column_stack([-edges[:, 1], edges[:, 0]]) / lengths[:, None]
    previous = np.roll(normals, 1, axis=0)                # edge i - 1 ends at vertex i
    denominators = 1.0 + np.sum(previous * normals, axis=1)
    if np.any(denominators < 1e-9):
        raise ValueError(f'polygon turns back on itself at vertex {int(np.argmin(denominators)) + 1}')
    return (previous + normals) / denominators[:, None]


def _polygon_crosses_itself(vertices) -> bool:
    """True when two non-adjacent edges of the closed polygon (n, 2) cross (strictly; vectorised over edge pairs)."""
    n_vertices = len(vertices)
    if n_vertices < 4:
        return False
    starts, ends = vertices, np.roll(vertices, -1, axis=0)

    def side(origin, tip, point):   # sign of cross(tip - origin, point - origin), broadcast
        return ((tip[..., 0] - origin[..., 0]) * (point[..., 1] - origin[..., 1])
                - (tip[..., 1] - origin[..., 1]) * (point[..., 0] - origin[..., 0]))

    start_a, end_a, start_b, end_b = starts[:, None], ends[:, None], starts[None, :], ends[None, :]
    crossing = ((side(start_a, end_a, start_b) * side(start_a, end_a, end_b) < 0.0)
                & (side(start_b, end_b, start_a) * side(start_b, end_b, end_a) < 0.0))
    index = np.arange(n_vertices)
    gap = (index[None, :] - index[:, None]) % n_vertices
    return bool(np.any(crossing & (gap > 1) & (gap < n_vertices - 1)))


def _points_on_one_line(points) -> bool:
    """True when every vertex lies on the line from the first to the farthest vertex (relative 1e-9), so the polygon
    has zero area; a shoelace test would also reject even-odd bow ties, whose signed area cancels."""
    origin_x, origin_y = points[0]
    offsets = [(point_x - origin_x, point_y - origin_y) for point_x, point_y in points[1:]]
    far_x, far_y = max(offsets, key=lambda offset: offset[0] ** 2 + offset[1] ** 2)
    far_squared = far_x ** 2 + far_y ** 2
    return all(abs(far_x * offset_y - far_y * offset_x) <= 1e-9 * far_squared for offset_x, offset_y in offsets)


def resolve_geometry(spec, parameters, errors=None) -> dict:
    """Evaluate every geometry object -> {name: ResolvedShape} in spec order; ValueError naming the first failure.
    errors=dict -> partial mode: record {name: message}, skip the object and its dependents, never raise."""
    failures = {}
    evaluated = {}            # name -> (kind name, values) for objects whose own params resolved
    seen = set()
    for obj in spec.geometry:
        if obj.name in seen:
            failures[obj.name] = 'duplicate geometry name'
            evaluated.pop(obj.name, None)
            continue
        seen.add(obj.name)
        try:
            evaluated[obj.name] = (obj.kind, _resolve_object_values(obj, parameters))
        except (ValueError, KeyError, TypeError) as exc:
            failures[obj.name] = error_message(exc)

    shapes, pending, copies = {}, {}, {}   # copies: array name -> {copy name: ResolvedShape}
    for name, (kind_name, values) in evaluated.items():
        kind = GEOMETRY_KINDS[kind_name]
        if not kind.composite:
            shapes[name] = ResolvedShape(name, kind_name, values)
            continue
        inputs = composite_inputs(kind, values)
        unknown = next((item for item in inputs if item not in seen), None)
        if unknown is not None:
            candidates = [other for other in _objects_of_kind(spec, 'domain') if other != name]
            failures[name] = (f"unknown input object {unknown!r}; domain objects: "
                              f"{', '.join(map(str, candidates)) or 'none'}")
        else:
            pending[name] = inputs

    progress = True
    while pending and progress:
        progress = False
        for name, inputs in list(pending.items()):
            failed_input = next((item for item in inputs if item in failures), None)
            if failed_input is not None:
                failures[name] = f'input object {failed_input!r} failed'
            elif all(item in shapes for item in inputs):
                wrong = next((item for item in inputs
                              if GEOMETRY_KINDS[shapes[item].kind].entity_kind != 'domain'), None)
                if wrong is not None:
                    candidates = [other for other in _objects_of_kind(spec, 'domain') if other != name]
                    failures[name] = (f"input {wrong!r} is a {shapes[wrong].kind}, not a domain object; domain "
                                      f"objects: {', '.join(map(str, candidates)) or 'none'}")
                else:
                    kind_name, values = evaluated[name]
                    if kind_name == 'array':   # copies join shapes for lookups (an outer array copies them)
                        copies[name] = _array_copies(name, values, shapes)
                        shapes.update(copies[name])
                    shapes[name] = ResolvedShape(name, kind_name, values)
            else:
                continue
            del pending[name]
            progress = True
    for name in pending:
        failures[name] = f'composite inputs form a cycle among {sorted(pending)}'

    if failures:
        if errors is None:
            first = next(obj.name for obj in spec.geometry if obj.name in failures)
            raise ValueError(f'geometry {first!r}: {failures[first]}')
        errors.update(failures)
    ordered = {}   # spec order, each array's copies right after it (the meshers read every primitive in this order)
    for obj in spec.geometry:
        if obj.name in shapes:
            ordered[obj.name] = shapes[obj.name]
            ordered.update(copies.get(obj.name, {}))
    return ordered


# ---- rasterisation ----
def _shape(name, resolved) -> ResolvedShape:
    """resolved[name]; KeyError naming the object when it is unknown or failed to resolve."""
    shape = resolved.get(name)
    if shape is None:
        raise KeyError(f'geometry object {name!r} is not resolved (unknown or failed); resolved: {list(resolved)}')
    return shape


def _domain_shape(name, resolved) -> ResolvedShape:
    """_shape of a domain object to rasterize; ValueError for a point / line object."""
    shape = _shape(name, resolved)
    if GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
        raise ValueError(f'rasterize: {name!r} is a {shape.kind} object; only domain objects rasterize')
    return shape


def _polygon_mask(points, X, Y, xp):
    """Even-odd point-in-polygon test of the coordinates (X, Y); the polygon is implicitly closed."""
    inside = xp.zeros(X.shape, dtype=bool)
    n_points = len(points)
    for index in range(n_points):
        xa, ya = points[index]
        xb, yb = points[(index + 1) % n_points]
        if ya == yb:
            continue              # horizontal edges never toggle a crossing
        straddles = (Y < ya) != (Y < yb)
        x_crossing = xa + (Y - ya) * ((xb - xa) / (yb - ya))
        inside ^= straddles & (X < x_crossing)
    return inside


def _polygon_mask_grid(points, x_column, y_row, xp):
    """_polygon_mask on the grid x_column (1D xp) x y_row (1D non-decreasing numpy): each edge XORs only the y
    columns it straddles, found by searchsorted in the dtype of y_row, so the result is bit-identical."""
    inside = xp.zeros((len(x_column), len(y_row)), dtype=bool)
    y_device = xp.asarray(y_row)
    x_device = x_column[:, None]
    n_points = len(points)
    for index in range(n_points):
        xa, ya = points[index]
        xb, yb = points[(index + 1) % n_points]
        if ya == yb:
            continue              # horizontal edges never toggle a crossing
        # y_row < v holds exactly for indices below searchsorted(y_row, v, 'left'), so the straddled range is:
        row_a, row_b = np.searchsorted(y_row, np.array((ya, yb), dtype=y_row.dtype), side='left')
        row_start, row_stop = int(min(row_a, row_b)), int(max(row_a, row_b))
        if row_start == row_stop:
            continue
        x_crossing = xa + (y_device[row_start:row_stop] - ya) * ((xb - xa) / (yb - ya))
        inside[:, row_start:row_stop] ^= x_device < x_crossing[None, :]
    return inside


def rasterize(name, resolved, X, Y, xp):
    """Bool mask (shape of X) of cell centres (X, Y) inside domain object name (3D: its xy footprint); composites
    recurse. Rectangles (and polygons) are half-open [x0, x1) x [y0, y1) like Domain.rect_mask; circles/ellipses
    closed."""
    shape = _domain_shape(name, resolved)
    values = shape.values
    kind = shape.kind
    if not is_plain(shape):
        footprint = _footprint(shape)
        return rasterize(footprint.name, {footprint.name: footprint}, X, Y, xp)
    if kind in ('rectangle', 'box'):
        return (X >= values['x0']) & (X < values['x1']) & (Y >= values['y0']) & (Y < values['y1'])
    if kind == 'rounded_rectangle':
        return _rounded_inside(values, 0.0, X, Y, xp)
    if kind in ('circle', 'cylinder_z', 'sphere'):   # a sphere's 2D footprint: its equator disc
        return (X - values['cx']) ** 2 + (Y - values['cy']) ** 2 <= values['r'] ** 2
    if kind == 'ellipse':
        angle = math.radians(values['angle'])
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        offset_x = X - values['cx']
        offset_y = Y - values['cy']
        local_u = (offset_x * cos_angle + offset_y * sin_angle) / values['a']
        local_v = (offset_y * cos_angle - offset_x * sin_angle) / values['b']
        return local_u ** 2 + local_v ** 2 <= 1.0
    if kind == 'polygon':
        return _polygon_mask(values['points'], X, Y, xp)
    if kind == 'array' and len(values['inputs']) >= _BUCKETED_COPIES:
        mask = _array_mask(shape, resolved, X, Y, xp)
        if mask is not None:
            return mask
    if kind in UNION_KINDS + ('intersection',):
        inputs = values['inputs']
        mask = rasterize(inputs[0], resolved, X, Y, xp)
        for other in inputs[1:]:
            if kind in UNION_KINDS:
                mask |= rasterize(other, resolved, X, Y, xp)
            else:
                mask &= rasterize(other, resolved, X, Y, xp)
        return mask
    if kind == 'difference':
        mask = rasterize(values['base'], resolved, X, Y, xp)
        for other in values['subtract']:
            mask &= ~rasterize(other, resolved, X, Y, xp)
        return mask
    raise ValueError(f'rasterize: unsupported geometry kind {kind!r}')


def _rounded_inside(values, offset, u, v, xp):
    """Points (u, v) inside a rounded_rectangle offset inward by `offset` [m] (a number or per point; corner radius
    max(r - offset, 0)): half-open along the straight sides like rectangles, closed on the corner arcs like circles."""
    x0, y0 = values['x0'] + offset, values['y0'] + offset
    x1, y1 = values['x1'] - offset, values['y1'] - offset
    if isinstance(offset, (int, float)):   # Python floats keep float32 centres float32
        radius = max(values['r'] - offset, 0.0)
    else:
        radius = xp.maximum(values['r'] - offset, 0.0)
    beyond_x = xp.maximum(xp.maximum(x0 + radius - u, u - (x1 - radius)), 0.0)
    beyond_y = xp.maximum(xp.maximum(y0 + radius - v, v - (y1 - radius)), 0.0)
    return (u >= x0) & (u < x1) & (v >= y0) & (v < y1) & (beyond_x ** 2 + beyond_y ** 2 <= radius ** 2)


def _array_mask(shape, resolved, X, Y, xp):
    """rasterize of an array by _array_points (None when the plain union over its copies is cheaper)."""
    X, Y = xp.broadcast_arrays(X, Y)
    inside = _array_points(shape, resolved, (X.reshape(-1), Y.reshape(-1)),
                           lambda copy, at: rasterize(copy, resolved, at[0], at[1], xp), xp)
    return None if inside is None else inside.reshape(X.shape)


def _array_points(shape, resolved, coordinates, test, xp):
    """Bool per point of an array without a full test per copy (None when its copies overlap so far that the plain
    union is cheaper): coordinates = flat (x, y) or (x, y, z) arrays; per point the copies whose bounding box can hold
    it, by the pitch arithmetic on the base box (z only for 3D points of a z-bounded base, else every layer), each
    copy tested only at its points by test(copy name, coordinates there), so it equals the union of the copies."""
    values = shape.values
    copies = values['inputs']
    counts = (values['nx'], values['ny'], values['nz'])
    pitches = (values['pitch_x'], values['pitch_y'], values.get('pitch_z') or 0.0)
    x0, y0, x1, y1 = bounding_box(copies[0], resolved)
    spans = [(x0, x1), (y0, y1)]
    if len(coordinates) == 3 and counts[2] > 1:
        z_lo, z_hi = z_extent(copies[0], resolved)
        if math.isfinite(z_lo) and math.isfinite(z_hi):
            spans.append((z_lo, z_hi))
    axes, passes = [], 1
    for (lo, hi), pitch, count in zip(spans, pitches, counts):
        margin = 1e-6 * (abs(lo) + abs(hi) + count * pitch)   # >> float32 rounding of the centres
        reach = int(math.floor((hi - lo + 2.0 * margin) / pitch)) + 2
        axes.append((lo, hi, pitch, count, margin, reach))
        passes *= reach
    if passes >= len(copies):
        return None
    candidates = [_copy_candidates(coordinate, *axis, xp) for coordinate, axis in zip(coordinates, axes)]
    strides = (1, counts[0], counts[0] * counts[1])
    layers = range(1, 1 if len(axes) == 3 else counts[2])   # z not bucketed: the copies above share the candidates
    inside = xp.zeros(coordinates[0].shape, dtype=bool)
    for combination in itertools.product(*candidates):
        valid = ~inside
        for _, axis_valid in combination:
            valid &= axis_valid
        hit = xp.flatnonzero(valid)
        if hit.size == 0:
            continue
        cell = sum(index[hit] * stride for (index, _), stride in zip(combination, strides))
        order = xp.argsort(cell)
        hit, cell = hit[order], cell[order]
        starts = xp.flatnonzero(xp.concatenate((xp.ones(1, dtype=bool), cell[1:] != cell[:-1])))
        bounds = starts.tolist() + [int(hit.size)]
        for start, stop, first_copy in zip(bounds[:-1], bounds[1:], cell[starts].tolist()):
            points = hit[start:stop]
            at = [coordinate[points] for coordinate in coordinates]
            found = test(copies[first_copy], at)
            for layer_index in layers:
                found = found | test(copies[first_copy + layer_index * strides[2]], at)
            inside[points] = found
    return inside


def _copy_candidates(coordinate, lo, hi, pitch, count, margin, reach, xp) -> list:
    """[(int64 copy index, bool valid)] per candidate step along one axis: the copies c (0 <= c < count) whose interval
    [lo, hi] + c pitch, widened by margin, holds each point's coordinate (float64 arithmetic)."""
    coordinate = coordinate.astype(xp.float64)
    first = xp.ceil((coordinate - (hi + margin)) / pitch)
    candidates = []
    for step in range(reach):
        index = first + step
        valid = ((index >= 0) & (index < count) & (coordinate >= (lo - margin) + index * pitch)
                 & (coordinate <= (hi + margin) + index * pitch))
        candidates.append((xp.clip(xp.nan_to_num(index), 0, count - 1).astype(xp.int64), valid))
    return candidates


def _box_reaches(name, resolved, x_centres, y_centres) -> bool:
    """False when the bounding box of `name` holds no centre of the 1D centres (rasterize_window gives it empty)."""
    x0, y0, x1, y1 = bounding_box(name, resolved)
    x_window, y_window = _centre_window(x_centres, x0, x1), _centre_window(y_centres, y0, y1)
    return x_window.stop > x_window.start and y_window.stop > y_window.start


def _array_inputs_near(values, resolved, x_centres, y_centres) -> list:
    """The copies of an array whose box (the base box moved by the pitches, copies i fastest) can reach the centres;
    an upper bound of the copies rasterize_window finds non-empty there (every input for a non-positive pitch)."""
    copies = values['inputs']
    counts = (values['nx'], values['ny'], values['nz'])
    if len(copies) < _BUCKETED_COPIES or x_centres.size == 0 or y_centres.size == 0:
        return copies
    x0, y0, x1, y1 = bounding_box(copies[0], resolved)
    ranges = []
    for lo, hi, centres, pitch, count in ((x0, x1, x_centres, values['pitch_x'], counts[0]),
                                          (y0, y1, y_centres, values['pitch_y'], counts[1])):
        if count == 1:
            ranges.append(range(1))
            continue
        if not pitch > 0.0:
            return copies
        first_centre, last_centre = float(centres[0]), float(centres[-1])
        # >= the _centre_window margin of every copy box, plus the rounding of index * pitch
        margin = 1e-6 * (abs(lo) + abs(hi) + abs(first_centre) + abs(last_centre) + 2.0 * count * pitch)
        first = max(0, math.ceil((first_centre - margin - hi) / pitch))
        last = min(count - 1, math.floor((last_centre + margin - lo) / pitch))
        ranges.append(range(first, last + 1))
    return [copies[i + counts[0] * (j + counts[1] * k)] for k in range(counts[2]) for j in ranges[1]
            for i in ranges[0]]


def _centre_window(centres, lo, hi) -> slice:
    """Slice of the non-decreasing 1D centres within [lo, hi], widened by ~16 float32 ulps of the bounds so every
    centre a rounded inside test can accept stays in."""
    margin = 1e-6 * (abs(lo) + abs(hi))
    start = int(np.searchsorted(centres, lo - margin, side='left'))
    stop = int(np.searchsorted(centres, hi + margin, side='right'))
    return slice(start, max(start, stop))


def _intersect_windows(window_a, window_b) -> tuple:
    """Overlap of two (slice x, slice y) windows (possibly empty)."""
    overlap = []
    for slice_a, slice_b in zip(window_a, window_b):
        start = max(slice_a.start, slice_b.start)
        overlap.append(slice(start, max(start, min(slice_a.stop, slice_b.stop))))
    return tuple(overlap)


def _relative_window(window, outer) -> tuple:
    """window (inside outer) as slices into an array cropped to outer."""
    return tuple(slice(inner.start - box.start, inner.stop - box.start) for inner, box in zip(window, outer))


def _window_shape(window) -> tuple:
    return tuple(axis.stop - axis.start for axis in window)


def rasterize_window(name, resolved, x_centres, y_centres, xp, z=None) -> tuple:
    """((slice x, slice y), fresh bool mask cropped to it) of domain object name on the outer grid of the
    non-decreasing 1D numpy centres; each primitive is tested only on its bounding-box sub-window (as rasterize).
    z (3D: a float32 cell-centre height) gives the cross-section at that height (None: the xy footprint); empty
    window when it misses."""
    shape = _domain_shape(name, resolved)
    values = shape.values
    kind = shape.kind
    if not GEOMETRY_KINDS[kind].composite:
        parts = _section(shape, z)[1] if z is not None else [shape if is_plain(shape) else _footprint(shape)]
        if not (len(parts) == 1 and parts[0] is shape):   # sphere slab circle, taper offset, x / y extrusion boxes
            return _rasterize_parts(name, parts, x_centres, y_centres, xp)
    if kind in UNION_KINDS + ('intersection',):
        inputs = values['inputs']
        if kind == 'array':
            inputs = _array_inputs_near(values, resolved, x_centres, y_centres)
        elif kind == 'union':   # an input whose box misses the centres rasterizes empty: skip its recursion
            inputs = [item for item in inputs if _box_reaches(item, resolved, x_centres, y_centres)]
        parts = [rasterize_window(item, resolved, x_centres, y_centres, xp, z) for item in inputs]
        if kind in UNION_KINDS:
            parts = [part for part in parts if 0 not in part[1].shape]   # an empty part must not widen the window
            if not parts:
                return _EMPTY_WINDOW, xp.zeros((0, 0), dtype=bool)
            window = tuple(slice(min(part_window[axis].start for part_window, _ in parts),
                                 max(part_window[axis].stop for part_window, _ in parts)) for axis in (0, 1))
            mask = xp.zeros(_window_shape(window), dtype=bool)
            for part_window, part_mask in parts:
                mask[_relative_window(part_window, window)] |= part_mask
            return window, mask
        window = parts[0][0]
        for part_window, _ in parts[1:]:
            window = _intersect_windows(window, part_window)
        mask = xp.ones(_window_shape(window), dtype=bool)
        if 0 not in mask.shape:
            for part_window, part_mask in parts:
                mask &= part_mask[_relative_window(window, part_window)]
        return window, mask
    if kind == 'difference':
        window, mask = rasterize_window(values['base'], resolved, x_centres, y_centres, xp, z)
        for other in values['subtract']:
            if 0 in mask.shape or not _box_reaches(other, resolved, x_centres[window[0]], y_centres[window[1]]):
                continue
            other_window, other_mask = rasterize_window(other, resolved, x_centres, y_centres, xp, z)
            overlap = _intersect_windows(window, other_window)
            if 0 not in _window_shape(overlap):
                mask[_relative_window(overlap, window)] &= ~other_mask[_relative_window(overlap, other_window)]
        return window, mask
    x0, y0, x1, y1 = bounding_box(name, resolved)
    window = (_centre_window(x_centres, x0, x1), _centre_window(y_centres, y0, y1))
    window_shape = _window_shape(window)
    x_window = xp.asarray(x_centres[window[0]])
    if kind == 'polygon':
        return window, _polygon_mask_grid(values['points'], x_window, y_centres[window[1]], xp)
    y_window = xp.asarray(y_centres[window[1]])
    if kind in _SEPARABLE_KINDS:   # (n, 1) x (1, m) operands: the float temporaries of one axis stay 1D
        return window, rasterize(name, resolved, x_window[:, None], y_window[None, :], xp)
    X = xp.broadcast_to(x_window[:, None], window_shape)
    Y = xp.broadcast_to(y_window[None, :], window_shape)
    return window, rasterize(name, resolved, X, Y, xp)


def rasterize_mapped(name, resolved, x_mapped, y_mapped, xp, z=None) -> tuple:
    """rasterize_window on a polar / spherical window: ((slice x, slice y) of the selected cells' index box, fresh
    bool mask cropped to it) from the 2D drawing-plane cell centres x_mapped, y_mapped (DomainMap.to_physical); z (a
    cylindrical sector, float32 cell-centre height): the cross-section there, None the xy footprint."""
    mask = _mapped_section_mask(name, resolved, x_mapped, y_mapped, xp, z)
    window = _occupied_window(mask, xp)
    if window is None:
        return _EMPTY_WINDOW, xp.zeros((0, 0), dtype=bool)
    return window, xp.ascontiguousarray(mask[window])


def rasterize_mapped_box(name, resolved, x_mapped, y_mapped, z_centres, xp) -> tuple:
    """rasterize_box on a cylindrical (r, phi, z) sector: ((slice x, slice y, slice z), fresh bool mask cropped to it)
    from the 2D drawing-plane centres x_mapped, y_mapped and the 1D numpy float32 z centres, one full-plane raster
    per distinct cross-section."""
    z_lo, z_hi = z_extent(name, resolved)
    window_z = _centre_window(z_centres, max(z_lo, float(z_centres[0])), min(z_hi, float(z_centres[-1])))
    sections, layer_keys = {}, []
    for k in range(window_z.start, window_z.stop):
        key = cross_section_key(name, resolved, z_centres[k])
        if key not in sections:
            sections[key] = _mapped_section_mask(name, resolved, x_mapped, y_mapped, xp, z_centres[k])
        layer_keys.append(key)
    footprint = None
    for section in sections.values():
        footprint = section.copy() if footprint is None else footprint | section
    window_xy = None if footprint is None else _occupied_window(footprint, xp)
    if window_xy is None:
        return _EMPTY_WINDOW + (slice(0, 0),), xp.zeros((0, 0, 0), dtype=bool)
    mask = xp.empty(_window_shape(window_xy) + (len(layer_keys),), dtype=bool)
    for offset, key in enumerate(layer_keys):
        mask[:, :, offset] = sections[key][window_xy]
    return window_xy + (window_z,), mask


def _mapped_section_mask(name, resolved, x_mapped, y_mapped, xp, z=None):
    """Bool mask (shape of x_mapped) of the drawing-plane centres inside domain object name: its cross-section at
    height z (_section parts, composites recursive), None = the xy footprint (rasterize)."""
    if z is None:
        return rasterize(name, resolved, x_mapped, y_mapped, xp)
    shape = _domain_shape(name, resolved)
    if shape.kind in UNION_KINDS + ('intersection',):
        masks = [_mapped_section_mask(item, resolved, x_mapped, y_mapped, xp, z)
                 for item in shape.values['inputs']]
        mask = masks[0]
        for other in masks[1:]:
            mask = (mask | other) if shape.kind in UNION_KINDS else (mask & other)
        return mask
    if shape.kind == 'difference':
        mask = _mapped_section_mask(shape.values['base'], resolved, x_mapped, y_mapped, xp, z)
        for other in shape.values['subtract']:
            mask = mask & ~_mapped_section_mask(other, resolved, x_mapped, y_mapped, xp, z)
        return mask
    mask = xp.zeros(x_mapped.shape, dtype=bool)
    for index, part in enumerate(_section(shape, z)[1]):
        part_name = f'{name}#{index}'
        mask |= rasterize(part_name, {part_name: ResolvedShape(part_name, part.kind, part.values)}, x_mapped, y_mapped,
                          xp)
    return mask


def _occupied_window(mask, xp):
    """(slice x, slice y) index box of the True cells of a 2D bool mask, None when it has none."""
    rows, columns = (xp.flatnonzero(mask.any(axis=axis)) for axis in (1, 0))
    if not rows.size:
        return None
    return (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1))


def _rasterize_parts(name, parts, x_centres, y_centres, xp) -> tuple:
    """rasterize_window of the union of the plain shapes `parts` standing in for object name (empty for none)."""
    if not parts:
        return _EMPTY_WINDOW, xp.zeros((0, 0), dtype=bool)
    local = {f'{name}#{index}': ResolvedShape(f'{name}#{index}', part.kind, part.values)
             for index, part in enumerate(parts)}
    if len(parts) == 1:
        return rasterize_window(next(iter(local)), local, x_centres, y_centres, xp)
    union_name = f'{name}#union'
    local[union_name] = ResolvedShape(union_name, 'union', {'inputs': list(local)})
    return rasterize_window(union_name, local, x_centres, y_centres, xp)


def domain_object_boxes(spec, shapes, entity_lists, skip_all=False):
    """Yield (name, (x0, y0, x1, y1)) of the domain objects of each expanded entity list with a finite bounding box
    (a list that fails to expand is skipped: validate reports it; skip_all: lists holding 'all' too); the CPU
    estimates' rough interface counts."""
    for entities in entity_lists:
        try:
            names = expand_entities(spec, entities)
        except (ValueError, KeyError, TypeError):
            continue
        if skip_all and 'all' in names:
            continue
        for name in names:
            shape = shapes.get(name)
            if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
                continue
            box = bounding_box(name, shapes)
            if all(math.isfinite(value) for value in box):
                yield name, box


def bounding_box(name, resolved) -> tuple:
    """(x0, y0, x1, y1) enclosing the object (3D kinds: their xy footprint, z_extent gives z); an empty intersection
    collapses to a degenerate box."""
    shape = _shape(name, resolved)
    values = shape.values
    kind = shape.kind
    if not is_plain(shape):
        footprint = _footprint(shape)
        return bounding_box(footprint.name, {footprint.name: footprint})
    if kind in ('rectangle', 'box', 'rounded_rectangle'):
        return (values['x0'], values['y0'], values['x1'], values['y1'])
    if kind in ('circle', 'cylinder_z', 'sphere'):
        cx, cy, r = values['cx'], values['cy'], values['r']
        return (cx - r, cy - r, cx + r, cy + r)
    if kind == 'ellipse':
        angle = math.radians(values['angle'])
        a, b = values['a'], values['b']
        half_width = math.hypot(a * math.cos(angle), b * math.sin(angle))
        half_height = math.hypot(a * math.sin(angle), b * math.cos(angle))
        return (values['cx'] - half_width, values['cy'] - half_height,
                values['cx'] + half_width, values['cy'] + half_height)
    if kind == 'polygon':
        xs = [px for px, _ in values['points']]
        ys = [py for _, py in values['points']]
        return (min(xs), min(ys), max(xs), max(ys))
    if kind == 'point':
        return (values['x'], values['y'], values['x'], values['y'])
    if kind == 'line':
        return (min(values['x0'], values['x1']), min(values['y0'], values['y1']),
                max(values['x0'], values['x1']), max(values['y0'], values['y1']))
    if kind in UNION_KINDS + ('intersection',):
        inputs = values['inputs']
        if kind == 'array':   # coordinates only grow with the copy index: the first and last copy bound them all
            inputs = [inputs[0], inputs[-1]]
        boxes = [bounding_box(item, resolved) for item in inputs]
        if kind in UNION_KINDS:
            return (min(box[0] for box in boxes), min(box[1] for box in boxes),
                    max(box[2] for box in boxes), max(box[3] for box in boxes))
        x0, y0 = max(box[0] for box in boxes), max(box[1] for box in boxes)
        x1, y1 = min(box[2] for box in boxes), min(box[3] for box in boxes)
        return (x0, y0, max(x0, x1), max(y0, y1))
    if kind == 'difference':
        return bounding_box(values['base'], resolved)
    raise ValueError(f'bounding_box: unsupported geometry kind {kind!r}')


def outline(name, resolved, n_segments=128, z=None) -> list:
    """Float64 (N, 2) polylines for drawing; closed shapes repeat the first vertex, composites give []. z (3D layer
    height): the cross-section there ([] when the object misses it; points / lines ignore z); None: the footprint."""
    shape = _shape(name, resolved)
    kind_info = GEOMETRY_KINDS[shape.kind]
    if kind_info.entity_kind == 'domain' and not kind_info.composite:
        parts = _section(shape, z)[1] if z is not None else [shape if is_plain(shape) else _footprint(shape)]
        if not (len(parts) == 1 and parts[0] is shape):
            return [polyline for part in parts for polyline in outline(part.name, {part.name: part}, n_segments)]
    values = shape.values
    kind = shape.kind
    if kind in ('rectangle', 'box'):
        x0, y0, x1, y1 = values['x0'], values['y0'], values['x1'], values['y1']
        return [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)]
    if kind == 'rounded_rectangle':
        if not values['r'] > 0.0:
            return outline(name, {name: ResolvedShape(name, 'rectangle', values)}, n_segments)
        points = _rounded_outline_points(values, max(1, math.ceil(int(n_segments) / 4)))
        return [np.vstack([points, points[:1]])]
    if kind in ('circle', 'ellipse', 'cylinder_z', 'sphere'):
        theta = np.linspace(0.0, 2.0 * np.pi, int(n_segments) + 1)
        if kind != 'ellipse':
            semi_a = semi_b = values['r']
            angle = 0.0
        else:
            semi_a, semi_b, angle = values['a'], values['b'], math.radians(values['angle'])
        local_u = semi_a * np.cos(theta)
        local_v = semi_b * np.sin(theta)
        polyline = np.column_stack([values['cx'] + local_u * math.cos(angle) - local_v * math.sin(angle),
                                    values['cy'] + local_u * math.sin(angle) + local_v * math.cos(angle)])
        polyline[-1] = polyline[0]
        return [polyline]
    if kind == 'polygon':
        vertices = np.array(values['points'], dtype=np.float64)
        return [np.vstack([vertices, vertices[:1]])]
    if kind == 'point':
        return [np.array([[values['x'], values['y']]], dtype=np.float64)]
    if kind == 'line':
        return [np.array([[values['x0'], values['y0']], [values['x1'], values['y1']]], dtype=np.float64)]
    if GEOMETRY_KINDS[kind].composite:
        return []
    raise ValueError(f'outline: unsupported geometry kind {kind!r}')


# ---- 3D (spec v2): z intervals, cross-sections, box masks ----
_EMPTY_WINDOW = (slice(0, 0), slice(0, 0))


def _section(shape, z) -> tuple:
    """(key, parts) of a primitive at height z (float32 compare): parts = plain shapes whose union is its xy
    cross-section, key equal for heights with an identical one; (None, []) when z misses it. Plain prisms (half-open,
    None ends open): (True, [shape]); spheres (closed): (radius, [circle]); tapers: (offset, [offset shape]); x / y
    extrusions: (intervals, [one rectangle per interval where their plane shape crosses the height]), tapered:
    (height, [sampled polygons])."""
    values = shape.values
    if shape.kind == 'sphere':
        offset_z = float(np.float32(z)) - values['cz']
        if offset_z * offset_z > values['r'] ** 2:
            return None, []
        radius = math.sqrt(values['r'] ** 2 - offset_z * offset_z)
        return radius, [ResolvedShape(shape.name, 'circle', {'cx': values['cx'], 'cy': values['cy'], 'r': radius})]
    height = np.float32(z)
    if shape.kind in PRISM_KINDS and _axis(values) != 'z':
        return _extrusion_section(shape, float(height))
    z0, z1 = values.get('z0'), values.get('z1')
    if (z0 is not None and height < np.float32(z0)) or (z1 is not None and not height < np.float32(z1)):
        return None, []
    if is_plain(shape):
        return True, [shape]
    offset = (float(height) - z0) * _taper_tan(values)
    return offset, [_plane_shape(shape, offset)]


def _extrusion_section(shape, height) -> tuple:
    """_section of a prism extruded along x / y: its plane shape cut where its second coordinate (z) = height gives
    intervals of the first (y / x), each a rectangle between the extrusion's ends along x / y; a taper sweeps the
    offset shape's intervals along the axis into polygons (_swept_polygons)."""
    values = shape.values
    start, end = values['z0'], values['z1']
    tan_angle = _taper_tan(values)
    if tan_angle:
        offset_shape = _plane_offsetter(shape)
        polygons = _swept_polygons(np.linspace(start, end, _TAPER_LEVELS + 1), lambda level: _crossing_intervals(
            offset_shape((level - start) * tan_angle), 'y', height))
        return (height, [_model_polygon(shape.name, polygon, _axis(values)) for polygon in polygons]) if polygons \
            else (None, [])
    intervals = tuple(_crossing_intervals(_plane_shape(shape), 'y', height))
    if not intervals:
        return None, []
    boxes = [(start, lo, end, hi) if _axis(values) == 'x' else (lo, start, hi, end) for lo, hi in intervals]
    return intervals, [_rectangle(shape.name, box) for box in boxes]


_TAPER_LEVELS = 256   # steps along the axis of a sampled taper section (x / y extrusions, side views of z tapers)
_BISECTIONS = 48      # halvings locating where a swept section changes its number of intervals


def _taper_tan(values) -> float:
    """tan(sidewall_angle) of a prism's values (0 when absent)."""
    return math.tan(math.radians(values.get('sidewall_angle') or 0.0))


def _widest_offset(values) -> float:
    """Inward offset [m] of a prism's widest section: 0 (the shape as given) or, widening tapers, the far end's."""
    tan_angle = _taper_tan(values)
    return min(0.0, (values['z1'] - values['z0']) * tan_angle) if tan_angle else 0.0


def _plane_offsetter(shape):
    """offset -> _plane_shape(shape, offset), a tapered polygon's mitres computed once (untapered: offsets are 0)."""
    if shape.kind != 'polygon' or not _taper_tan(shape.values):
        return lambda offset: _plane_shape(shape, offset)
    vertices = np.asarray(shape.values['points'], dtype=np.float64)
    mitres = _inward_mitres(vertices)
    plane_values = _plane_shape(shape).values
    return lambda offset: ResolvedShape(shape.name, 'polygon',
                                        dict(plane_values, points=(vertices + offset * mitres).tolist()))


def _crossing_intervals(shape, axis, position) -> list:
    """_line_intervals of non-zero length."""
    return [(lo, hi) for lo, hi in _line_intervals(shape, axis, position) if hi > lo]


def _swept_polygons(levels, intervals_at) -> list:
    """Closed float64 (N, 2) polylines in (value, level) of a region whose cut at each sorted level is the interval list
    intervals_at(level): one per interval of each run of levels with equally many, run ends bisected onto the change."""
    samples = []
    for level in (float(level) for level in levels):
        cut = intervals_at(level)
        while samples and len(cut) != len(samples[-1][1]):
            count, below, above = len(samples[-1][1]), samples[-1][0], level
            for _ in range(_BISECTIONS):
                middle = 0.5 * (below + above)
                if len(intervals_at(middle)) == count:
                    below = middle
                else:
                    above = middle
            if below > samples[-1][0]:
                samples.append((below, intervals_at(below)))
            if not above < level:
                break
            samples.append((above, intervals_at(above)))   # may differ from both neighbours: bisect again
        samples.append((level, cut))
    polylines, run = [], []
    for sample in samples + [None]:
        if sample is not None and run and len(sample[1]) == len(run[-1][1]):
            run.append(sample)
            continue
        if len(run) > 1:
            for index in range(len(run[0][1])):
                low_side = [(cut[index][0], level) for level, cut in run]               # upward
                high_side = [(cut[index][1], level) for level, cut in reversed(run)]    # downward
                polylines.append(np.array(low_side + high_side + low_side[:1], dtype=np.float64))
        run = [sample]
    return polylines


def _model_polygon(name, polyline, axis) -> ResolvedShape:
    """Plain xy polygon of a closed (value, level) polyline of a prism extruded along axis 'x' (level = x, value = y)
    or 'y' (value = x, level = y)."""
    points = polyline[:-1, ::-1] if axis == 'x' else polyline[:-1]
    return ResolvedShape(name, 'polygon', {'points': points.tolist()})


def _rectangle(name, box) -> ResolvedShape:
    """Plain rectangle ResolvedShape over box = (x0, y0, x1, y1), sorted."""
    return ResolvedShape(name, 'rectangle', dict(zip(('x0', 'y0', 'x1', 'y1'), box)))


def _plane_shape(shape, offset=0.0) -> ResolvedShape:
    """A prism's 2D shape as a plain prism with open ends in its own plane coordinates, offset inward by `offset` [m]
    (negative: outward): rectangles on every side, the circle radius, both ellipse semi-axes, polygons mitred; rounded
    rectangles on every side with corner radius max(r - offset, 0) (the exact offset curve)."""
    values = dict(shape.values, axis='z', sidewall_angle=0.0, z0=None, z1=None)
    if offset and shape.kind in ('rectangle', 'rounded_rectangle'):
        values.update(x0=values['x0'] + offset, y0=values['y0'] + offset, x1=values['x1'] - offset,
                      y1=values['y1'] - offset)
        if shape.kind == 'rounded_rectangle':
            values['r'] = max(values['r'] - offset, 0.0)
    elif offset and shape.kind == 'circle':
        values['r'] -= offset
    elif offset and shape.kind == 'ellipse':
        values.update(a=values['a'] - offset, b=values['b'] - offset)
    elif offset:
        vertices = np.asarray(values['points'], dtype=np.float64)
        values['points'] = [tuple(map(float, vertex)) for vertex in vertices + offset * _inward_mitres(vertices)]
    return ResolvedShape(shape.name, shape.kind, values)


def _footprint(shape) -> ResolvedShape:
    """Plain xy shape of a non-plain prism's footprint: a taper's wider end, the box of an x / y extrusion (its ends
    across the first-coordinate range of its plane shape; tapered: that range swept along the axis, a polygon)."""
    values = shape.values
    axis = _axis(values)
    if axis == 'z':
        return _plane_shape(shape, _widest_offset(values))
    start, end = values['z0'], values['z1']
    tan_angle = _taper_tan(values)
    if tan_angle:
        offset_shape = _plane_offsetter(shape)

        def first_range(level):
            plane = offset_shape((level - start) * tan_angle)
            return [bounding_box(shape.name, {shape.name: plane})[0::2]]

        polyline = _swept_polygons(np.linspace(start, end, _TAPER_LEVELS + 1), first_range)[0]
        return _model_polygon(shape.name, polyline, axis)
    u_lo, _, u_hi, _ = bounding_box(shape.name, {shape.name: _plane_shape(shape)})
    return _rectangle(shape.name, (start, u_lo, end, u_hi) if axis == 'x' else (u_lo, start, u_hi, end))


def axis_parallel_faces(name, resolved) -> dict:
    """{axis: face positions [m]} of the object's flat faces normal to x / y / z (mesh snap nodes): rectangle / box
    sides and set prism / cylinder ends along their axis; tilted walls, curves, spheres, points, lines and composites
    give none."""
    shape = _shape(name, resolved)
    values = shape.values
    if shape.kind == 'box':
        return {axis: (values[f'{axis}0'], values[f'{axis}1']) for axis in ('x', 'y', 'z')}
    if shape.kind not in PRISM_KINDS + ('cylinder_z',):
        return {}
    axis = _axis(values)
    faces = {axis: tuple(values[end] for end in ('z0', 'z1') if values.get(end) is not None)}
    if shape.kind == 'rectangle' and not values.get('sidewall_angle'):
        first, second = _PLANE_AXES[axis]
        faces.update({first: (values['x0'], values['x1']), second: (values['y0'], values['y1'])})
    elif shape.kind == 'rounded_rectangle' and not values.get('sidewall_angle'):   # flat sides of non-zero length
        first, second = _PLANE_AXES[axis]
        diameter = 2.0 * values['r'] * (1.0 + _RADIUS_SLACK)
        if values['y1'] - values['y0'] > diameter:
            faces[first] = (values['x0'], values['x1'])
        if values['x1'] - values['x0'] > diameter:
            faces[second] = (values['y0'], values['y1'])
    return faces


def cross_section_key(name, resolved, z):
    """Hashable key equal for heights where the object's cross-section is identical (plain prisms: inside or not;
    spheres: the slab radius; tapers: the offset; x / y extrusions: the crossing intervals, tapered the height;
    composites: their inputs' keys): consecutive z cells with equal keys share one 2D mask."""
    shape = _shape(name, resolved)
    kind = GEOMETRY_KINDS[shape.kind]
    if shape.kind == 'array':   # x / y moves keep where a copy's key changes: one column of copies along z decides
        layer = shape.values['nx'] * shape.values['ny']
        return tuple(cross_section_key(item, resolved, z) for item in shape.values['inputs'][::layer])
    if kind.composite:
        return tuple(cross_section_key(item, resolved, z) for item in composite_inputs(kind, shape.values))
    return _section(shape, z)[0]


def cross_section_runs(names, resolved, z_centres) -> list:
    """[(k_start, k_stop)] runs of consecutive z cells (1D float32 centres) over which every named domain object keeps
    its cross-section (cross_section_key): one 2D rasterisation per run (3D build, per z-slab)."""
    runs, previous = [], object()
    for k, z in enumerate(z_centres):
        key = tuple(cross_section_key(name, resolved, z) for name in names)
        if runs and key == previous:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1])
        previous = key
    return [tuple(run) for run in runs]


def z_extent(name, resolved) -> tuple:
    """(z_lo, z_hi) [m] of the object, -inf / inf for open prism ends (x / y extrusions: the second-coordinate range of
    their plane shape); points / lines (z, z), (-inf, inf) when unset."""
    shape = _shape(name, resolved)
    values = shape.values
    kind = shape.kind
    if kind == 'sphere':
        return (values['cz'] - values['r'], values['cz'] + values['r'])
    if kind in PRISM_KINDS and _axis(values) != 'z':
        _, v_lo, _, v_hi = bounding_box(name, {name: _plane_shape(shape, _widest_offset(values))})
        return (v_lo, v_hi)
    if kind in ('point', 'line'):
        return (-math.inf, math.inf) if values.get('z') is None else (values['z'], values['z'])
    if kind in UNION_KINDS + ('intersection',):
        inputs = values['inputs'] if kind != 'array' else [values['inputs'][0], values['inputs'][-1]]
        extents = [z_extent(item, resolved) for item in inputs]
        if kind in UNION_KINDS:
            return (min(lo for lo, _ in extents), max(hi for _, hi in extents))
        z_lo = max(lo for lo, _ in extents)
        return (z_lo, max(z_lo, min(hi for _, hi in extents)))
    if kind == 'difference':
        return z_extent(values['base'], resolved)
    z0, z1 = values.get('z0'), values.get('z1')
    return (-math.inf if z0 is None else z0, math.inf if z1 is None else z1)


def point_z(values, size_z) -> float:
    """Height [m] of a 3D point / line: its z param, mid-height size_z / 2 when None."""
    return 0.5 * float(size_z) if values.get('z') is None else float(values['z'])


def rasterize_box(name, resolved, x_centres, y_centres, z_centres, xp) -> tuple:
    """((slice x, slice y, slice z), fresh bool mask cropped to it) of a 3D domain object on the 1D numpy float32
    centres: one rasterize_window per distinct cross-section (no full-grid temporaries)."""
    window = box_window(name, resolved, x_centres, y_centres, z_centres)
    window_xy, window_z = window[:2], window[2]
    mask = xp.zeros(_window_shape(window), dtype=bool)
    sections = {}
    for offset, k in enumerate(range(window_z.start, window_z.stop)):
        key = cross_section_key(name, resolved, z_centres[k])
        if key not in sections:
            sections[key] = rasterize_window(name, resolved, x_centres, y_centres, xp, z=z_centres[k])
        section_window, section_mask = sections[key]
        overlap = _intersect_windows(section_window, window_xy)
        if 0 not in _window_shape(overlap):
            mask[_relative_window(overlap, window_xy) + (offset,)] = section_mask[_relative_window(overlap,
                                                                                                  section_window)]
    return window, mask


def _offset_plane_inside(plane, offset, u, v):
    """Points (u, v) inside a plain 2D shape offset inward by the per-point `offset` [m] (_plane_shape rules:
    rectangles on every side, rounded rectangles also r - offset, circle radius, both ellipse semi-axes, polygons
    mitred per vertex; even-odd polygons)."""
    values = plane.values
    if plane.kind == 'rectangle':
        return ((u >= values['x0'] + offset) & (u < values['x1'] - offset) & (v >= values['y0'] + offset)
                & (v < values['y1'] - offset))
    if plane.kind == 'rounded_rectangle':
        return _rounded_inside(values, offset, u, v, np)
    if plane.kind == 'circle':
        radius = values['r'] - offset
        return (radius >= 0.0) & ((u - values['cx']) ** 2 + (v - values['cy']) ** 2 <= radius ** 2)
    if plane.kind == 'ellipse':
        angle = math.radians(values['angle'])
        offset_u, offset_v = u - values['cx'], v - values['cy']
        semi_a, semi_b = values['a'] - offset, values['b'] - offset
        local_u = (offset_u * math.cos(angle) + offset_v * math.sin(angle)) / np.where(semi_a > 0.0, semi_a, 1.0)
        local_v = (offset_v * math.cos(angle) - offset_u * math.sin(angle)) / np.where(semi_b > 0.0, semi_b, 1.0)
        return (semi_a > 0.0) & (semi_b > 0.0) & (local_u ** 2 + local_v ** 2 <= 1.0)
    vertices = np.asarray(values['points'], dtype=np.float64)
    mitres = _inward_mitres(vertices)
    inside = np.zeros(np.shape(u), dtype=bool)
    for index in range(len(vertices)):
        following = (index + 1) % len(vertices)
        start_u = vertices[index, 0] + offset * mitres[index, 0]
        start_v = vertices[index, 1] + offset * mitres[index, 1]
        end_u = vertices[following, 0] + offset * mitres[following, 0]
        end_v = vertices[following, 1] + offset * mitres[following, 1]
        straddles = (v < start_v) != (v < end_v)
        rise = np.where(straddles, end_v - start_v, 1.0)
        inside ^= straddles & (u < start_u + (v - start_v) * (end_u - start_u) / rise)
    return inside


def contains_points_3d(name, resolved, x, y, z) -> np.ndarray:
    """Host bool per 3D point (1D float64 x, y, z [m]) inside a domain object by the build's inclusion rules (spheres
    and circles closed, prisms half-open along their axis); tapered prisms offset per point (_offset_plane_inside);
    composites recurse. Any equal-shaped arrays. Tetrahedron element tags (model.tetrahedral_meshing) and the 3D
    conformal locator (boundary_locator)."""
    shape = _shape(name, resolved)
    values = shape.values
    kind = GEOMETRY_KINDS[shape.kind]
    if kind.entity_kind != 'domain':
        raise ValueError(f'contains_points_3d: {name!r} is a {shape.kind} object; only domain objects hold points')
    if shape.kind == 'difference':
        inside = contains_points_3d(values['base'], resolved, x, y, z)
        for other in values['subtract']:
            inside &= ~contains_points_3d(other, resolved, x, y, z)
        return inside
    if shape.kind == 'array' and len(values['inputs']) >= _BUCKETED_COPIES:
        coordinates = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in (x, y, z))
        inside = _array_points(shape, resolved, coordinates,
                               lambda copy, at: contains_points_3d(copy, resolved, *at), np)
        if inside is not None:
            return inside.reshape(np.shape(x))
    if kind.composite:
        masks = [contains_points_3d(item, resolved, x, y, z) for item in values['inputs']]
        return np.logical_or.reduce(masks) if shape.kind in UNION_KINDS else np.logical_and.reduce(masks)
    if shape.kind == 'sphere':
        return (x - values['cx']) ** 2 + (y - values['cy']) ** 2 + (z - values['cz']) ** 2 <= values['r'] ** 2
    axis = _axis(values) if shape.kind in PRISM_KINDS else 'z'
    along = {'x': x, 'y': y, 'z': z}[axis]
    first, second = ({'x': x, 'y': y, 'z': z}[model_axis] for model_axis in _PLANE_AXES[axis])
    tan_angle = _taper_tan(values)
    if tan_angle:   # tapers have both ends set; the shape as given sits at min(z0, z1)
        bottom, top = min(values['z0'], values['z1']), max(values['z0'], values['z1'])
        offset = (along - bottom) * tan_angle
        return (along >= bottom) & (along < top) & _offset_plane_inside(_plane_shape(shape), offset, first, second)
    inside = np.asarray(rasterize(name, {name: _plane_shape(shape) if axis != 'z' else shape}, first, second, np),
                        dtype=bool)
    if values.get('z0') is not None:
        inside &= along >= values['z0']
    if values.get('z1') is not None:
        inside &= along < values['z1']
    return inside


def box_window(name, resolved, x_centres, y_centres, z_centres) -> tuple:
    """(slice x, slice y, slice z) of the cells (1D numpy centres) a 3D object's bounding box can cover."""
    x0, y0, x1, y1 = bounding_box(name, resolved)
    z_lo, z_hi = z_extent(name, resolved)
    return (_centre_window(x_centres, x0, x1), _centre_window(y_centres, y0, y1),
            _centre_window(z_centres, max(z_lo, float(z_centres[0])), min(z_hi, float(z_centres[-1]))))


# ---- vertical sections (3D side views: z-stack objects cut by x = const or y = const) ----
def _check_cut(cut) -> tuple:
    """(axis, position) of a vertical cut: plane x = position (axis 'x', u = y) or y = position (axis 'y', u = x)."""
    axis, position = cut
    if axis not in ('x', 'y'):
        raise ValueError(f"cut axis must be 'x' or 'y', got {axis!r}")
    position = float(position)
    if not math.isfinite(position):
        raise ValueError(f'cut position must be finite, got {position}')
    return axis, position


def _along_across(axis, point_x, point_y) -> tuple:
    """(u, across) of a drawing-plane point on the vertical cut along axis (u as in _check_cut)."""
    return (point_x, point_y) if axis == 'y' else (point_y, point_x)


def _line_intervals(shape, axis, position) -> list:
    """[(u_lo, u_hi)] where the cut line crosses the xy shape of a box / prism / cylinder_z (u as in _check_cut;
    rectangles half-open across the line, circles / ellipses closed, polygons even-odd)."""
    values = shape.values
    kind = shape.kind
    along_x = axis == 'y'
    if kind in ('rectangle', 'box'):
        u0, across_0 = _along_across(axis, values['x0'], values['y0'])
        u1, across_1 = _along_across(axis, values['x1'], values['y1'])
        return [(u0, u1)] if across_0 <= position < across_1 else []
    if kind == 'rounded_rectangle':   # the corner arcs pull the ends in by r - sqrt(r^2 - depth^2)
        u0, across_0 = _along_across(axis, values['x0'], values['y0'])
        u1, across_1 = _along_across(axis, values['x1'], values['y1'])
        if not across_0 <= position < across_1:
            return []
        radius = values['r']
        depth = max(across_0 + radius - position, position - (across_1 - radius), 0.0)
        inset = radius - math.sqrt(max(radius * radius - depth * depth, 0.0))
        return [(u0 + inset, u1 - inset)]
    if kind in ('circle', 'cylinder_z'):
        centre_u, centre_across = _along_across(axis, values['cx'], values['cy'])
        offset = position - centre_across
        if offset * offset > values['r'] ** 2:
            return []
        half_chord = math.sqrt(values['r'] ** 2 - offset * offset)
        return [(centre_u - half_chord, centre_u + half_chord)]
    if kind == 'ellipse':   # swapping x and y turns the semi-axis a from angle to 90 deg - angle
        centre_u, centre_across = _along_across(axis, values['cx'], values['cy'])
        angle = math.radians(values['angle'] if along_x else 90.0 - values['angle'])
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        inverse_a2, inverse_b2 = values['a'] ** -2, values['b'] ** -2
        offset = position - centre_across
        quadratic = cos_angle ** 2 * inverse_a2 + sin_angle ** 2 * inverse_b2
        linear = 2.0 * offset * cos_angle * sin_angle * (inverse_a2 - inverse_b2)
        constant = offset * offset * (sin_angle ** 2 * inverse_a2 + cos_angle ** 2 * inverse_b2) - 1.0
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return []
        root = math.sqrt(discriminant)
        return [(centre_u + (-linear - root) / (2.0 * quadratic), centre_u + (-linear + root) / (2.0 * quadratic))]
    if kind == 'polygon':
        points = [_along_across(axis, point_x, point_y) for point_x, point_y in values['points']]
        crossings = []
        for index, (u_a, across_a) in enumerate(points):
            u_b, across_b = points[(index + 1) % len(points)]
            if across_a != across_b and (position < across_a) != (position < across_b):
                crossings.append(u_a + (position - across_a) * ((u_b - u_a) / (across_b - across_a)))
        crossings.sort()
        return list(zip(crossings[0::2], crossings[1::2]))
    raise ValueError(f'vertical section: unsupported geometry kind {kind!r}')


def vertical_section(name, resolved, cut, size_z, tolerance=0.0, n_segments=128) -> list:
    """Float64 (N, 2) polylines in (u, z) [m] of the object on the vertical cut (axis, position) (u as in _check_cut):
    boxes / prisms / cylinders closed rectangles (open ends clamped to [0, size_z]), tapers sampled trapezoids, x / y
    extrusions their plane shape or rectangles (tapered: offset shape or sampled sweeps), spheres circles; points one vertex within tolerance of the cut, lines a
    segment lying in it or their crossing vertex; [] for composites and misses."""
    axis, position = _check_cut(cut)
    shape = _shape(name, resolved)
    values = shape.values
    kind = shape.kind
    if GEOMETRY_KINDS[kind].composite:
        return []
    if kind == 'point':
        u, across = _along_across(axis, values['x'], values['y'])
        return [np.array([[u, point_z(values, size_z)]])] if abs(across - position) <= tolerance else []
    if kind == 'line':
        height = point_z(values, size_z)
        u0, across_0 = _along_across(axis, values['x0'], values['y0'])
        u1, across_1 = _along_across(axis, values['x1'], values['y1'])
        if abs(across_0 - position) <= tolerance and abs(across_1 - position) <= tolerance:
            return [np.array([[u0, height], [u1, height]])]
        if across_0 != across_1 and min(across_0, across_1) <= position <= max(across_0, across_1):
            return [np.array([[u0 + (position - across_0) * ((u1 - u0) / (across_1 - across_0)), height]])]
        return []
    if kind == 'sphere':
        centre_u, centre_across = _along_across(axis, values['cx'], values['cy'])
        offset = position - centre_across
        if offset * offset > values['r'] ** 2:
            return []
        radius = math.sqrt(values['r'] ** 2 - offset * offset)
        theta = np.linspace(0.0, 2.0 * np.pi, int(n_segments) + 1)
        polyline = np.column_stack([centre_u + radius * np.cos(theta), values['cz'] + radius * np.sin(theta)])
        polyline[-1] = polyline[0]
        return [polyline]
    if kind in PRISM_KINDS and _axis(values) != 'z':
        return _extrusion_vertical_section(shape, axis, position, n_segments)
    if not is_plain(shape):
        return _taper_vertical_section(shape, axis, position, size_z, n_segments)
    z_lo, z_hi = z_extent(name, resolved)
    z_lo, z_hi = max(z_lo, 0.0), min(z_hi, float(size_z))
    if not z_lo < z_hi:
        return []
    return [np.array([[u0, z_lo], [u1, z_lo], [u1, z_hi], [u0, z_hi], [u0, z_lo]], dtype=np.float64)
            for u0, u1 in _crossing_intervals(shape, axis, position)]


def _extrusion_vertical_section(shape, cut_axis, position, n_segments) -> list:
    """vertical_section of a prism extruded along x / y: a cut normal to its axis shows the plane shape (between the
    ends, half-open; tapered: offset at the cut), a cut along it one rectangle per interval of the plane shape across
    the cut position (tapered: those intervals swept along the axis in n_segments steps)."""
    values = shape.values
    start, end = values['z0'], values['z1']
    tan_angle = _taper_tan(values)
    offset_shape = _plane_offsetter(shape)
    if cut_axis == _axis(values):
        if not start <= position < end:
            return []
        return outline(shape.name, {shape.name: offset_shape((position - start) * tan_angle)}, n_segments)
    levels = np.linspace(start, end, int(n_segments) + 1 if tan_angle else 2)
    polygons = _swept_polygons(levels, lambda level: _crossing_intervals(offset_shape((level - start) * tan_angle),
                                                                         'x', position))
    return [np.ascontiguousarray(polygon[:, ::-1]) for polygon in polygons]   # (v, s) -> (u = s, z = v)


def _taper_vertical_section(shape, cut_axis, position, size_z, n_levels) -> list:
    """vertical_section of a tapered z prism sampled at n_levels + 1 heights over its ends (clamped to [0, size_z]):
    per run of heights crossing the cut in equally many intervals, one closed polyline per interval."""
    values = shape.values
    z_lo, z_hi = max(values['z0'], 0.0), min(values['z1'], float(size_z))
    if not z_lo < z_hi:
        return []
    tan_angle = _taper_tan(values)
    offset_shape = _plane_offsetter(shape)
    return _swept_polygons(np.linspace(z_lo, z_hi, int(n_levels) + 1), lambda height: _crossing_intervals(
        offset_shape((height - values['z0']) * tan_angle), cut_axis, position))


def _cut_line_centres(axis, position, u_centres) -> tuple:
    """1D (x centres, y centres) of the cut line: u_centres along the line, the one position across it."""
    line = np.array([position], dtype=np.float64)
    return (u_centres, line) if axis == 'y' else (line, u_centres)


def section_mask(name, resolved, cut, u_centres, z_centres) -> np.ndarray:
    """Bool (len(u_centres), len(z_centres)) mask of a domain object on the vertical cut (axis, position) at 1D numpy
    centres; one rasterize_window per distinct cross-section (cross_section_runs)."""
    axis, position = _check_cut(cut)
    x_centres, y_centres = _cut_line_centres(axis, position, np.asarray(u_centres, dtype=np.float64))
    mask = np.zeros((len(u_centres), len(z_centres)), dtype=bool)
    for k_start, k_stop in cross_section_runs([name], resolved, z_centres):
        window, cropped = rasterize_window(name, resolved, x_centres, y_centres, np, z=z_centres[k_start])
        row = np.zeros((x_centres.size, y_centres.size), dtype=bool)
        row[window] = cropped
        mask[:, k_start:k_stop] = row.reshape(-1, 1)
    return mask


# ---- surface faces (3D preview, view_3d C: the continuous shape, so it reads prism values beside _section) ----
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def surface_faces(name, resolved, size, n_segments=48) -> tuple:
    """(faces, edges) of a primitive 3D domain object clipped to the physical box size = (size_x, size_y, size_z) [m]:
    faces = float64 (N, 3) planar polygons [m], edges = float64 (M, 3) polylines of cap rims and polygon corners."""
    shape = _shape(name, resolved)
    size = tuple(float(length) for length in size)
    if shape.kind == 'sphere':
        faces, edges = _sphere_faces(shape.values, n_segments), []
    elif shape.kind in PRISM_KINDS + ('box', 'cylinder_z'):
        faces, edges = _prism_faces(shape, size, n_segments)
    else:
        raise ValueError(f'surface_faces: {shape.kind!r} is not a primitive domain object')
    clipped_faces = [face for face in (_clip_polygon(face, size) for face in faces) if face is not None]
    return clipped_faces, [piece for polyline in edges for piece in _clip_polyline(polyline, size)]


def _prism_faces(shape, size, n_segments) -> tuple:
    """Unclipped (faces, edge polylines) of a prism / box / cylinder_z: caps at its ends along the axis (open ends and
    ends beyond the domain at the domain faces, with the taper offset of that height), side quads between matching cap
    vertices (offsets are linear in the height, so they are exact), closed cap rims and rectangle / polygon corners."""
    values = shape.values
    axis = _axis(values) if shape.kind in PRISM_KINDS else 'z'
    first, second = (_AXIS_INDEX[plane_axis] for plane_axis in _PLANE_AXES[axis])
    along = _AXIS_INDEX[axis]
    start = 0.0 if values.get('z0') is None else max(values['z0'], 0.0)
    end = size[along] if values.get('z1') is None else min(values['z1'], size[along])
    if not start < end:
        return [], []
    plane_kind = {'box': 'rectangle', 'cylinder_z': 'circle'}.get(shape.kind, shape.kind)
    plane = ResolvedShape(shape.name, plane_kind, values)
    tan_angle = _taper_tan(values)
    if plane_kind == 'rectangle' and not tan_angle:   # clamped to the domain: closed caps where it leaves the box
        bounds = dict(x0=max(values['x0'], 0.0), y0=max(values['y0'], 0.0), x1=min(values['x1'], size[first]),
                      y1=min(values['y1'], size[second]))
        if not (bounds['x0'] < bounds['x1'] and bounds['y0'] < bounds['y1']):
            return [], []
        plane = ResolvedShape(shape.name, 'rectangle', dict(values, **bounds))
    rings = []
    for height in (start, end):
        offset = (height - values['z0']) * tan_angle if tan_angle else 0.0
        if plane_kind == 'rounded_rectangle':   # equal vertex counts at both caps, also where a taper ends the arcs
            polyline = _rounded_outline_points(_plane_shape(plane, offset).values, max(1, math.ceil(n_segments / 4)))
        else:
            polyline = outline(shape.name, {shape.name: _plane_shape(plane, offset)}, n_segments)[0][:-1]
        ring = np.empty((len(polyline), 3), dtype=np.float64)
        ring[:, first], ring[:, second], ring[:, along] = polyline[:, 0], polyline[:, 1], height
        rings.append(ring)
    bottom, top = rings
    following = np.roll(np.arange(len(bottom)), -1)
    sides = np.stack([bottom, bottom[following], top[following], top], axis=1)    # (n, 4, 3) quads
    edges = [np.vstack([ring, ring[:1]]) for ring in rings]
    if plane_kind in ('rectangle', 'polygon'):
        edges.extend(np.stack([bottom, top], axis=1))
    return [bottom, top] + list(sides), edges


def _sphere_faces(values, n_segments) -> list:
    """Lat-long quads of a sphere (n_segments around, the pole rows degenerate to triangles)."""
    n_around = max(8, int(n_segments))
    n_rows = max(4, n_around // 3)
    polar = np.linspace(0.0, np.pi, n_rows + 1)[:, None]
    azimuth = np.linspace(0.0, 2.0 * np.pi, n_around + 1)[None, :]
    radius = values['r']
    grid = np.stack([values['cx'] + radius * np.sin(polar) * np.cos(azimuth),
                     values['cy'] + radius * np.sin(polar) * np.sin(azimuth),
                     values['cz'] + radius * np.cos(polar) * np.ones_like(azimuth)], axis=-1)   # (rows+1, around+1, 3)
    corners = np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2)
    return list(corners.reshape(-1, 4, 3))


def _clip_polygon(polygon, size):
    """(N, 3) planar polygon clipped to the box [0, size] per axis (Sutherland-Hodgman); None when < 3 vertices stay."""
    points = np.asarray(polygon, dtype=np.float64)
    for axis in range(3):
        for sign, bound in ((1.0, 0.0), (-1.0, size[axis])):
            inside_distance = sign * (points[:, axis] - bound)
            outside = inside_distance < 0.0
            if not outside.any():
                continue
            if outside.all():
                return None
            following = np.roll(points, -1, axis=0)
            following_distance = np.roll(inside_distance, -1)
            kept = []
            for point, next_point, distance, next_distance in zip(points, following, inside_distance,
                                                                  following_distance):
                if distance >= 0.0:
                    kept.append(point)
                if (distance >= 0.0) != (next_distance >= 0.0):
                    kept.append(point + (distance / (distance - next_distance)) * (next_point - point))
            points = np.array(kept)
    return points if len(points) >= 3 else None


def _clip_segment(start, end, size):
    """(2, 3) part of the segment start -> end inside the box [0, size] per axis (Liang-Barsky), None when it misses."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    delta = end - start
    t_enter, t_leave = 0.0, 1.0
    for axis in range(3):
        for direction, room in ((-delta[axis], start[axis]), (delta[axis], size[axis] - start[axis])):
            if direction == 0.0:
                if room < 0.0:
                    return None
            elif direction < 0.0:
                t_enter = max(t_enter, room / direction)
            else:
                t_leave = min(t_leave, room / direction)
    if t_enter > t_leave:
        return None
    return np.array([start if t_enter == 0.0 else start + t_enter * delta,     # unclipped ends stay exact, so
                     end if t_leave == 1.0 else start + t_leave * delta])       # _clip_polyline can join them


def _clip_polyline(polyline, size) -> list:
    """(M, 3) polyline clipped to the box [0, size] per axis: the pieces inside, consecutive clipped segments joined."""
    pieces, current = [], None
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = _clip_segment(start, end, size)
        if segment is None:
            current = None
            continue
        if current is not None and np.array_equal(current[-1], segment[0]):
            current.append(segment[1])
        else:
            current = [segment[0], segment[1]]
            pieces.append(current)
    return [np.array(piece, dtype=np.float64) for piece in pieces]


# ---- perimeters ----
def _neighbour_covered(mask, xp):
    """Four bool arrays: is the -x / +x / -y / +y neighbour inside mask (beyond the array edge counts as inside)."""
    covered_x_lo = xp.ones_like(mask)
    covered_x_lo[1:, :] = mask[:-1, :]
    covered_x_hi = xp.ones_like(mask)
    covered_x_hi[:-1, :] = mask[1:, :]
    covered_y_lo = xp.ones_like(mask)
    covered_y_lo[:, 1:] = mask[:, :-1]
    covered_y_hi = xp.ones_like(mask)
    covered_y_hi[:, :-1] = mask[:, 1:]
    return covered_x_lo, covered_x_hi, covered_y_lo, covered_y_hi


def perimeter_mask(mask, xp):
    """Cells of mask with at least one 4-neighbour outside it (the array edge counts as inside)."""
    mask = xp.asarray(mask, dtype=bool)
    covered_x_lo, covered_x_hi, covered_y_lo, covered_y_hi = _neighbour_covered(mask, xp)
    return mask & ~(covered_x_lo & covered_x_hi & covered_y_lo & covered_y_hi)


def _inward_mask_gradient(mask, dx_cells, dy_cells, xp, y_width_scales=None):
    """float64 (g_x, g_y) Sobel gradient of the mask per cell over centre spacings [1/m], pointing into the mask;
    the array edge replicates (a shape crossing it continues straight). y_width_scales (nx,): physical / logical y
    widths per row (polar: r_c), dividing g_y."""
    padded = xp.pad(mask.astype(xp.float64), 1, mode='edge')
    smoothed_along_y = padded[:, :-2] + 2.0 * padded[:, 1:-1] + padded[:, 2:]
    smoothed_along_x = padded[:-2, :] + 2.0 * padded[1:-1, :] + padded[2:, :]
    widths_x = xp.pad(dx_cells, 1, mode='edge')
    widths_y = xp.pad(dy_cells, 1, mode='edge')
    centre_span_x = 0.5 * widths_x[:-2] + widths_x[1:-1] + 0.5 * widths_x[2:]
    centre_span_y = 0.5 * widths_y[:-2] + widths_y[1:-1] + 0.5 * widths_y[2:]
    gradient_x = (smoothed_along_y[2:, :] - smoothed_along_y[:-2, :]) / centre_span_x[:, None]
    gradient_y = (smoothed_along_x[:, 2:] - smoothed_along_x[:, :-2]) / centre_span_y[None, :]
    if y_width_scales is not None:
        gradient_y = gradient_y / xp.asarray(y_width_scales, dtype=xp.float64)[:, None]
    return gradient_x, gradient_y


def _boundary_normal_components(mask, exposed_sides, dx_cells, dy_cells, xp, y_width_scales=None):
    """float64 (|n_x|, |n_y|) per cell: the Sobel gradient summed over the 3x3 perimeter cells sharing an exposed
    side; 1, 1 (staircase) on thin cells (both sides of an axis exposed) and right-angle axis-aligned corners."""
    nx, ny = mask.shape

    def neighbour(padded, offset_x, offset_y):
        return padded[1 + offset_x:1 + offset_x + nx, 1 + offset_y:1 + offset_y + ny]

    exposed_x_lo, exposed_x_hi, exposed_y_lo, exposed_y_hi = exposed_sides
    keeps_staircase = (exposed_x_lo & exposed_x_hi) | (exposed_y_lo & exposed_y_hi)
    padded_mask = xp.pad(mask, 1, mode='edge')
    for side_x, exposed_x in ((-1, exposed_x_lo), (1, exposed_x_hi)):
        for side_y, exposed_y in ((-1, exposed_y_lo), (1, exposed_y_hi)):
            # both exposed faces continue straight along their edge: a rectangle corner, whose staircase is exact
            x_face_continues = neighbour(padded_mask, 0, -side_y) & ~neighbour(padded_mask, side_x, -side_y)
            y_face_continues = neighbour(padded_mask, -side_x, 0) & ~neighbour(padded_mask, -side_x, side_y)
            keeps_staircase |= exposed_x & exposed_y & x_face_continues & y_face_continues
    sources = ~keeps_staircase
    gradient_x, gradient_y = _inward_mask_gradient(mask, dx_cells, dy_cells, xp, y_width_scales)
    padded_gradient_x = xp.pad(gradient_x, 1)
    padded_gradient_y = xp.pad(gradient_y, 1)
    padded_source_sides = [xp.pad(side & sources, 1) for side in exposed_sides]
    summed_x = xp.zeros_like(gradient_x)
    summed_y = xp.zeros_like(gradient_y)
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            shares_side = xp.zeros_like(mask)
            for side, padded_side in zip(exposed_sides, padded_source_sides):
                shares_side |= side & neighbour(padded_side, offset_x, offset_y)
            summed_x += xp.where(shares_side, neighbour(padded_gradient_x, offset_x, offset_y), 0.0)
            summed_y += xp.where(shares_side, neighbour(padded_gradient_y, offset_x, offset_y), 0.0)
    del gradient_x, gradient_y, padded_gradient_x, padded_gradient_y, padded_source_sides
    magnitude = xp.hypot(summed_x, summed_y)
    keeps_staircase |= magnitude == 0.0
    safe_magnitude = xp.where(keeps_staircase, 1.0, magnitude)
    normal_x = xp.where(keeps_staircase, 1.0, xp.abs(summed_x) / safe_magnitude)
    normal_y = xp.where(keeps_staircase, 1.0, xp.abs(summed_y) / safe_magnitude)
    return normal_x, normal_y


def perimeter_weights(mask, dx_cells, dy_cells, xp, x_face_scales=None, y_face_scales=None, y_width_scales=None):
    """float32 perimeter length / cell area [1/m] of mask cells: each exposed face times its |normal component|, so
    slanted and curved edges get their true length to ~2 % (any dx); dx_cells (nx,), dy_cells (ny,) match mask.
    x_face_scales: optional (lo, hi) float64 (nx,) factors of exposed x_lo / x_hi faces (face area x dx / volume:
    axisymmetric r_face / r_c); y_face_scales: (lo, hi) (nx, ny) factors of the y faces (WindowMetric
    y_face_volume_ratios: polar 1 / r_c); y_width_scales (nx,): physical / logical y widths of the normal estimate
    (polar r_c, dy_cells then in rad)."""
    mask = xp.asarray(mask, dtype=bool)
    dx_cells = xp.asarray(dx_cells, dtype=xp.float64)
    dy_cells = xp.asarray(dy_cells, dtype=xp.float64)
    if mask.shape != (dx_cells.size, dy_cells.size):
        raise ValueError(f'perimeter_weights: mask shape {mask.shape} does not match '
                         f'cell widths ({dx_cells.size}, {dy_cells.size})')
    exposed_sides = [mask & ~covered for covered in _neighbour_covered(mask, xp)]
    normal_x, normal_y = _boundary_normal_components(mask, exposed_sides, dx_cells, dy_cells, xp, y_width_scales)
    exposed_x_lo, exposed_x_hi, exposed_y_lo, exposed_y_hi = exposed_sides
    # an exposed x-face has length dy_j over area dx_i dy_j -> |n_x| / dx_i (y-faces -> |n_y| / dy_j); int8 face
    # counts and in-place float32 products keep the full-grid temporaries at 4 B/cell
    weights = normal_x.astype(xp.float32)
    del normal_x
    if x_face_scales is None:
        weights *= exposed_x_lo.astype(xp.int8) + exposed_x_hi
    else:
        lo_scale, hi_scale = (xp.asarray(scale, dtype=xp.float32)[:, None] for scale in x_face_scales)
        weights *= exposed_x_lo * lo_scale + exposed_x_hi * hi_scale
    weights *= (1.0 / dx_cells).astype(xp.float32)[:, None]
    y_weights = normal_y.astype(xp.float32)
    del normal_y
    if y_face_scales is None:
        y_weights *= exposed_y_lo.astype(xp.int8) + exposed_y_hi
    else:
        lo_scale, hi_scale = (xp.asarray(scale, dtype=xp.float32) for scale in y_face_scales)
        y_weights *= exposed_y_lo * lo_scale + exposed_y_hi * hi_scale
    y_weights *= (1.0 / dy_cells).astype(xp.float32)[None, :]
    weights += y_weights
    return weights


# ---- material preview ----
def preview_pixel_centers(extent, shape) -> tuple:
    """1D float64 (x centres (mx,), y centres (my,)) [m] of the material_preview pixels over extent, image shape."""
    x_min, x_max, y_min, y_max = extent
    n_x, n_y = shape
    return (x_min + (np.arange(n_x) + 0.5) * ((x_max - x_min) / n_x),
            y_min + (np.arange(n_y) + 0.5) * ((y_max - y_min) / n_y))


def material_paint_order(expanded_selections) -> list:
    """Material indices, lowest priority first: background materials (selection 'all'), then the others, each in list
    order, so painting in this order lets the last one win; None (failed) selections are skipped."""
    return sorted((index for index, entities in enumerate(expanded_selections) if entities is not None),
                  key=lambda index: 'all' not in expanded_selections[index])


def paint_material_index(expanded_selections, object_mask, shape, xp):
    """int16 image of 1 + winning material index (0 = none) painted in material_paint_order; object_mask(name) ->
    (window slices, cropped bool mask), or None for an entity that paints nothing (outer face, failed object)."""
    index_image = xp.zeros(shape, dtype=xp.int16)
    for material_index in material_paint_order(expanded_selections):
        entities = expanded_selections[material_index]
        if 'all' in entities:
            index_image[...] = material_index + 1
            continue
        for entity in entities:
            window_mask = object_mask(entity)
            if window_mask is not None:
                window, cropped = window_mask
                xp.copyto(index_image[window], np.int16(material_index + 1), where=cropped)
    return index_image


def domain_object_rasterizer(shapes, x_centres, y_centres, xp, z=None, reuse=None, mapped=None):
    """Cached object_mask(name) for paint_material_index on 1D host centres: rasterize_window of resolved domain
    objects (3D: their cross-section at height z; None = the xy footprint), None for anything else. reuse = a dict the
    caller keeps across heights: an object whose cross_section_key equals its last one there gets that mask again.
    mapped = (x, y) 2D drawing-plane centres of a polar / spherical window or cylindrical sector (rasterize_mapped at z;
    the 1D centres unused)."""
    cache = {}

    def object_mask(name):
        resolved = shapes.get(name)
        if resolved is None or GEOMETRY_KINDS[resolved.kind].entity_kind != 'domain':
            return None
        if name not in cache:
            key = cross_section_key(name, shapes, z) if reuse is not None and z is not None else None
            previous = reuse.get(name) if key is not None else None
            if previous is not None and previous[0] == key:
                cache[name] = previous[1]
            else:
                cache[name] = (rasterize_mapped(name, shapes, *mapped, xp, z) if mapped is not None
                               else rasterize_window(name, shapes, x_centres, y_centres, xp, z))
                if key is not None:
                    reuse[name] = (key, cache[name])
        return cache[name]
    return object_mask


def material_preview(spec, parameters, max_cells=400, shapes=None, z=None, cut=None):
    """CPU material map of the physical window: (int16 (mx, my) image of 1 + winning material index (0 = none),
    material names, extent domain_extent: (0, size_x, 0, size_y), or the sector's box with 0 outside a polar /
    spherical window); shapes = partial resolve_geometry (resolved here when None); z = the layer height of a 3D model
    (None: every object's xy footprint). cut = (axis, position) of a 3D model: the vertical section instead,
    (max_cells, max_cells) pixels over extent (0, size_u, 0, size_z) (u as in _check_cut)."""
    size_x, size_y = domain_size(spec, parameters)
    max_cells = int(max_cells)
    if max_cells < 1:
        raise ValueError('max_cells must be >= 1')
    if shapes is None:
        shapes = resolve_geometry(spec, parameters, errors={})
    expanded_selections = material_selections(spec)
    names = [material.name for material in spec.materials]
    if cut is not None:
        if z is not None:
            raise ValueError('material_preview: give a layer height z or a vertical cut, not both')
        extent = _section_extent(spec, parameters, cut, size_x, size_y)
        window_map = domain_map(spec, parameters)
        if not window_map.mapped:
            return _section_preview(shapes, expanded_selections, cut, extent, max_cells), names, extent
        # a cylindrical sector: u over the sector's drawing-plane box, 0 where the cut line leaves the sector
        box = window_map.extent()
        axis, position = _check_cut(cut)
        extent = ((box[0], box[1]) if axis == 'y' else (box[2], box[3])) + extent[2:]
        index_image = _section_preview(shapes, expanded_selections, cut, extent, max_cells)
        u_centres, _ = preview_pixel_centers(extent, index_image.shape)
        x_line, y_line = _cut_line_centres(axis, position, u_centres)
        index_image[~window_map.contains(*np.broadcast_arrays(x_line, y_line)), :] = 0
        return index_image, names, extent
    window_map = domain_map(spec, parameters)
    extent = window_map.extent()
    width, height = extent[1] - extent[0], extent[3] - extent[2]
    if window_map.line:   # one pixel row on the x axis (y = 0, the centre of the strip)
        mx, my = max_cells, 1
    elif width >= height:
        mx = max_cells
        my = min(max_cells, max(1, int(round(max_cells * height / width))))
    else:
        my = max_cells
        mx = min(max_cells, max(1, int(round(max_cells * width / height))))
    x_centres, y_centres = preview_pixel_centers(extent, (mx, my))
    index_image = paint_material_index(expanded_selections,
                                       domain_object_rasterizer(shapes, x_centres, y_centres, np, z), (mx, my), np)
    if window_map.mapped:
        index_image[~window_map.contains(x_centres[:, None], y_centres[None, :])] = 0
    return index_image, names, extent


def _section_extent(spec, parameters, cut, size_x, size_y) -> tuple:
    """(0, size_u, 0, size_z) of a vertical cut; ValueError in a 2D model."""
    size_z = domain_size_z(spec, parameters)
    if size_z is None:
        raise ValueError('a vertical cut needs a 3D model (domain size_z set)')
    axis, _ = _check_cut(cut)
    return (0.0, float(size_x if axis == 'y' else size_y), 0.0, float(size_z))


def _section_preview(shapes, expanded_selections, cut, extent, max_cells) -> np.ndarray:
    """int16 (max_cells, max_cells) material index image of a vertical cut over its extent: one painted line per run
    of z rows with equal cross-sections of every domain object."""
    axis, position = _check_cut(cut)
    u_centres, z_centres = preview_pixel_centers(extent, (max_cells, max_cells))
    x_centres, y_centres = _cut_line_centres(axis, position, u_centres)
    domain_names = [name for name, shape in shapes.items()   # array copies: in their array's key
                    if GEOMETRY_KINDS[shape.kind].entity_kind == 'domain' and not is_array_copy(name)]
    index_image = np.zeros((max_cells, max_cells), dtype=np.int16)
    for k_start, k_stop in cross_section_runs(domain_names, shapes, z_centres):
        line = paint_material_index(expanded_selections,
                                    domain_object_rasterizer(shapes, x_centres, y_centres, np, z_centres[k_start]),
                                    (x_centres.size, y_centres.size), np)
        index_image[:, k_start:k_stop] = line.reshape(-1, 1)
    return index_image

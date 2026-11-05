"""Coordinate metric of the 2D window (symmetry plan 2.16; numpy only, GUI-thread safe): the Domain's in-plane grid
(Coordinates: Cartesian or polar) times the solve's out-of-plane reduction (planar or axisymmetric) gives exact,
separable face areas, physical cell widths and cell volumes (WindowMetric), the one geometry an FV operator reads;
WindowMetric3D likewise for a 3D window, Cartesian or a cylindrical (r, phi, z) sector (symmetry plan 2.21)."""
import math
from dataclasses import dataclass

import numpy as np

GRIDS = ('cartesian', 'polar')
# phi wrap of a planar polar window (y_lo joined to y_hi): none, periodic (u continues), antiperiodic (u changes sign:
# the real angular Bloch phase pi of a C_N sector); complex Bloch phases need a complex operator
ANGLE_WRAPS = ('none', 'periodic', 'antiperiodic')
_WRAP_SIGNS = {'none': 0, 'periodic': 1, 'antiperiodic': -1}
# rad: a spherical (polar + axisymmetric) angle face this close to 0 or pi lies on the axis (float32 cell widths)
AXIS_ANGLE_SNAP = 1e-6


def ring_angle(azimuthal_order: int = 0) -> float:
    """Angle weight of an axisymmetric ring (symmetry 2.14): 2 pi at order m = 0, pi (the integral of cos^2(m phi)) for
    an order-m amplitude u_m(r, z) cos(m phi)."""
    return 2.0 * math.pi if int(azimuthal_order) == 0 else math.pi


@dataclass(frozen=True)
class Coordinates:
    """In-plane grid of a 2D Domain. 'cartesian': x, y lengths [m]. 'polar': x = r [m] (radius_min at the window's x_lo
    edge), y = an angle [rad] (angle_min at its y_lo edge): phi about the out-of-plane axis in a planar solve, the polar
    angle theta from the +z axis in an axisymmetric one (spherical (r, theta), 0 <= theta <= pi). angle_wrap (polar,
    planar solves only): ANGLE_WRAPS, the y_hi face joined to the y_lo face instead of two outer faces. line (symmetry
    2.20): a 1D window of one cell across y (1 m Cartesian: totals per m^2; polar 2 pi: per m of cylinder; spherical
    0..pi: whole spheres), the metric unchanged. On a Domain3D 'polar' makes (x, y) the (r, phi) plane of a
    cylindrical sector with z the axis (window_metric_3d)."""
    grid: str = 'cartesian'
    radius_min: float = 0.0
    angle_min: float = 0.0
    angle_wrap: str = 'none'
    line: bool = False

    def __post_init__(self):
        if self.grid not in GRIDS:
            raise ValueError(f"coordinate grid must be one of {', '.join(GRIDS)}, got {self.grid!r}")
        for name in ('radius_min', 'angle_min'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"coordinates {name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if not self.polar and (self.radius_min != 0.0 or self.angle_min != 0.0):
            raise ValueError("radius_min / angle_min belong to a polar grid; a Cartesian grid has its origin at the "
                             "window's lower corner")
        if self.radius_min < 0.0:
            raise ValueError(f"polar radius_min must be >= 0, got {self.radius_min!r}")
        if self.angle_wrap not in ANGLE_WRAPS:
            raise ValueError(f"angle_wrap must be one of {', '.join(ANGLE_WRAPS)}, got {self.angle_wrap!r}")
        if self.angle_wrap != 'none' and not self.polar:
            raise ValueError(f"angle_wrap {self.angle_wrap!r} joins the angle faces of a polar grid; a Cartesian grid "
                             "has none")
        if self.line and (self.angle_min != 0.0 or self.angle_wrap != 'none'):
            raise ValueError('a 1D line window spans the whole angle range: angle_min 0, angle_wrap none')

    @property
    def polar(self) -> bool:
        return self.grid == 'polar'

    @property
    def wrap_sign(self) -> int:
        """0 without a phi wrap, +1 periodic, -1 antiperiodic (u beyond y_hi = wrap_sign x u at y_lo)."""
        return _WRAP_SIGNS[self.angle_wrap]

    def label(self, axisymmetric: bool) -> str:
        """Coordinate names of the window axes: '(x, y)', '(r, z)', '(r, phi)' or '(r, theta)'; 1D lines '(x)', '(r)
        cylindrical' or '(r) spherical'."""
        if self.line:
            return ('(r) spherical' if axisymmetric else '(r) cylindrical') if self.polar else '(x)'
        if self.polar:
            return '(r, theta)' if axisymmetric else '(r, phi)'
        return '(r, z)' if axisymmetric else '(x, y)'


CARTESIAN = Coordinates()


def to_cartesian(coordinates: Coordinates, axisymmetric: bool, x_logical, y_logical, xp=np):
    """Drawing-plane (x, y) [m] of logical polar window positions (r = radius_min + x_logical, angle = angle_min +
    y_logical): (r cos phi, r sin phi) planar, (r sin theta, r cos theta) = (distance to the axis, height) spherical."""
    radius = coordinates.radius_min + x_logical
    angle = coordinates.angle_min + y_logical
    if axisymmetric:
        return radius * xp.sin(angle), radius * xp.cos(angle)
    return radius * xp.cos(angle), radius * xp.sin(angle)


def from_cartesian(coordinates: Coordinates, axisymmetric: bool, angle_span, x, y):
    """Logical (r - radius_min, angle - angle_min) of drawing-plane points (numpy); a planar phi is wrapped into the 2 pi
    window centred on the sector [angle_min, angle_min + angle_span], so points just outside it stay just outside."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    radius = np.hypot(x, y)
    if axisymmetric:
        angle = np.arctan2(x, y)   # theta from the +y (rotation) axis; x < 0 lies outside (theta < 0)
    else:
        centre = coordinates.angle_min + 0.5 * float(angle_span)
        angle = centre + np.mod(np.arctan2(y, x) - centre + np.pi, 2.0 * np.pi) - np.pi
    return radius - coordinates.radius_min, angle - coordinates.angle_min


@dataclass(frozen=True, eq=False)
class Separable:
    """Host float64 array a[i, j] = along_x[i] * along_y[j]."""
    along_x: np.ndarray
    along_y: np.ndarray

    def factors(self):
        """The two 1D factors reshaped to broadcast along x and y of a 2D array (as Separable3.factors)."""
        return self.along_x.reshape(-1, 1), self.along_y.reshape(1, -1)

    def full(self) -> np.ndarray:
        return np.outer(self.along_x, self.along_y)


@dataclass(frozen=True, eq=False)
class WindowMetric:
    """Exact FV geometry of the window cells: x_face_area (nx+1, ny) / y_face_area (nx, ny+1) [m^2; planar: m per metre
    of depth], x_length / y_length (nx, ny) physical cell widths [m], volume (nx, ny) [m^3; planar: m^2 per metre];
    axisymmetric = whole rings (2 pi). Faces on an axis (r = 0, theta = 0 or pi) have zero area."""
    coordinates: Coordinates
    axisymmetric: bool
    x_face_area: Separable
    y_face_area: Separable
    x_length: Separable
    y_length: Separable
    volume: Separable

    @property
    def cartesian_planar(self) -> bool:
        """True for the plain (x, y) metric (unit weights)."""
        return not (self.axisymmetric or self.coordinates.polar)

    @property
    def wrap_sign(self) -> int:
        """Coordinates.wrap_sign of the window: 0, or +-1 when its y_hi face is joined to its y_lo face."""
        return self.coordinates.wrap_sign

    def label(self) -> str:
        return self.coordinates.label(self.axisymmetric)

    @property
    def face_area(self) -> tuple:
        """(x_face_area, y_face_area): per axis, as WindowMetric3D.face_area."""
        return self.x_face_area, self.y_face_area

    @property
    def length(self) -> tuple:
        """(x_length, y_length): per axis, as WindowMetric3D.length."""
        return self.x_length, self.y_length

    def x_face_volume_ratios(self):
        """(lo, hi) (nx,) host arrays: area of each cell's x_lo / x_hi face times its width dx over its volume (1 on
        the (x, y) grid, r_face / r_c on (r, z) and (r, phi)); the y parts of area and volume cancel."""
        area, volume = self.x_face_area.along_x, self.volume.along_x
        widths = self.x_length.along_x
        return area[:-1] * widths / volume, area[1:] * widths / volume

    def y_face_volume_ratios(self, dy_window):
        """(lo, hi) (nx, ny) host arrays: area of each cell's y_lo / y_hi face times its logical width dy_window [m or
        rad] over its volume (1 on (x, y) and (r, z), 1 / r_c on (r, phi), ~1 / r_c on (r, theta))."""
        dy = np.asarray(dy_window, dtype=np.float64)
        along_x = self.y_face_area.along_x / self.volume.along_x
        area_y, volume_y = self.y_face_area.along_y, self.volume.along_y
        return (np.outer(along_x, area_y[:-1] * dy / volume_y), np.outer(along_x, area_y[1:] * dy / volume_y))


def face_positions(widths, origin) -> np.ndarray:
    """float64 face coordinates (n + 1,) of consecutive cell widths starting at origin."""
    return origin + np.concatenate(([0.0], np.cumsum(np.asarray(widths, dtype=np.float64))))


@dataclass(frozen=True, eq=False)
class YeeRadii:
    """Radii [m] of the Yee samples along r of a polar grid (symmetry 2.21; for the polar EM operators): centre r_c
    (n,), face r_f (n + 1,), link (n,) = mean radius between centres i and i + 1 (the last cell: its hi face)."""
    centre: np.ndarray
    face: np.ndarray
    link: np.ndarray


def yee_radii(dr_widths, radius_first_face) -> YeeRadii:
    """YeeRadii of consecutive radial widths [m] whose first face lies at radius_first_face [m] (host float64)."""
    face = face_positions(dr_widths, float(radius_first_face))
    centre = 0.5 * (face[:-1] + face[1:])
    return YeeRadii(centre, face, np.append(0.5 * (centre[:-1] + centre[1:]), face[-1]))


def window_metric(dx_window, dy_window, coordinates: Coordinates = CARTESIAN,
                  axisymmetric: bool = False, azimuthal_order: int = 0) -> WindowMetric:
    """WindowMetric of the window cell widths (x [m]; y [m], or [rad] on a polar grid), planar (per metre of depth) or
    axisymmetric (rings about the x_lo window edge on a Cartesian grid, about the theta = 0 line on a polar one); an
    (r, z) azimuthal_order m != 0 weighs its rings by ring_angle(m) = pi instead of 2 pi."""
    dx = np.asarray(dx_window, dtype=np.float64)
    dy = np.asarray(dy_window, dtype=np.float64)
    if dx.ndim != 1 or dy.ndim != 1 or not dx.size or not dy.size:
        raise ValueError(f"window_metric needs non-empty 1D cell widths, got shapes {dx.shape}, {dy.shape}")
    if not (np.all(dx > 0.0) and np.all(dy > 0.0)):
        raise ValueError("window_metric needs positive cell widths")
    ones_x, ones_y = np.ones(dx.size), np.ones(dy.size)
    ones_x_faces, ones_y_faces = np.ones(dx.size + 1), np.ones(dy.size + 1)
    x_length = Separable(dx, ones_y)
    if azimuthal_order and (coordinates.polar or not axisymmetric):
        raise ValueError(f"azimuthal_order {azimuthal_order} needs an axisymmetric (r, z) window on a Cartesian grid")
    if not coordinates.polar:
        if not axisymmetric:   # (x, y)
            return WindowMetric(coordinates, False, Separable(ones_x_faces, dy), Separable(dx, ones_y_faces),
                                x_length, Separable(ones_x, dy), Separable(dx, dy))
        # (r, z): x = r from the x_lo edge; angle r_c dr is the exact annulus area (ring volume per dz; angle 2 pi)
        angle = ring_angle(azimuthal_order)
        r_faces = face_positions(dx, 0.0)
        annulus = 0.5 * angle * (r_faces[:-1] + r_faces[1:]) * dx
        return WindowMetric(coordinates, True, Separable(angle * r_faces, dy), Separable(annulus, ones_y_faces),
                            x_length, Separable(ones_x, dy), Separable(annulus, dy))

    r_faces = face_positions(dx, coordinates.radius_min)
    r_centres = 0.5 * (r_faces[:-1] + r_faces[1:])
    angle_faces = face_positions(dy, coordinates.angle_min)
    y_length = Separable(r_centres, dy)   # r_c d_angle
    if not axisymmetric:   # (r, phi): r_c dr dphi is the exact sector area
        span = angle_faces[-1] - angle_faces[0]
        if span > 2.0 * np.pi + AXIS_ANGLE_SNAP:
            raise ValueError(f"a polar window spans at most 2 pi in phi, got {span!r} rad")
        return WindowMetric(coordinates, False, Separable(r_faces, dy), Separable(dx, ones_y_faces),
                            x_length, y_length, Separable(r_centres * dx, dy))

    # (r, theta), rotation about the theta = 0 axis
    if coordinates.wrap_sign:
        raise ValueError(f"angle_wrap {coordinates.angle_wrap!r} needs a planar (r, phi) solve: the theta range of a "
                         "spherical (r, theta) grid ends on the axis or on cones, never wraps")
    if angle_faces[0] < -AXIS_ANGLE_SNAP or angle_faces[-1] > np.pi + AXIS_ANGLE_SNAP:
        raise ValueError(f"a spherical (r, theta) window needs 0 <= theta <= pi, got {angle_faces[0]!r} .. "
                         f"{angle_faces[-1]!r} rad")
    sin_faces = np.sin(angle_faces)
    sin_faces[(np.abs(angle_faces) <= AXIS_ANGLE_SNAP) | (np.abs(angle_faces - np.pi) <= AXIS_ANGLE_SNAP)] = 0.0
    angle_centres = 0.5 * (angle_faces[:-1] + angle_faces[1:])
    band = 2.0 * np.sin(angle_centres) * np.sin(0.5 * dy)   # cos theta_lo - cos theta_hi
    shell = dx * (r_faces[:-1] ** 2 + r_faces[:-1] * r_faces[1:] + r_faces[1:] ** 2)   # r_hi^3 - r_lo^3
    two_pi = 2.0 * np.pi
    return WindowMetric(coordinates, True, Separable(two_pi * r_faces ** 2, band),
                        Separable(two_pi * r_centres * dx, sin_faces), x_length, y_length,
                        Separable(two_pi / 3.0 * shell, band))


@dataclass(frozen=True, eq=False)
class Separable3:
    """Host float64 array a[i, j, k] = along_x[i] * along_y[j] * along_z[k]."""
    along_x: np.ndarray
    along_y: np.ndarray
    along_z: np.ndarray

    def factors(self):
        """The three 1D factors reshaped to broadcast along x, y and z of a 3D array."""
        return (self.along_x.reshape(-1, 1, 1), self.along_y.reshape(1, -1, 1), self.along_z.reshape(1, 1, -1))

    def full(self) -> np.ndarray:
        x_factor, y_factor, z_factor = self.factors()
        return x_factor * y_factor * z_factor


@dataclass(frozen=True, eq=False)
class WindowMetric3D:
    """Exact FV geometry of a 3D window (symmetry 2.21): face_area[a] of the faces normal to axis a (n_a + 1 along a)
    [m^2], length[a] the physical cell lengths along a [m], volume [m^3]; (x, y, z) or cylindrical (r, phi, z)."""
    coordinates: Coordinates
    face_area: tuple
    length: tuple
    volume: Separable3

    @property
    def wrap_sign(self) -> int:
        """Coordinates.wrap_sign: 0, or +-1 when the y_hi (phi) face is joined to the y_lo face."""
        return self.coordinates.wrap_sign

    def label(self) -> str:
        return '(r, phi, z)' if self.coordinates.polar else '(x, y, z)'


def window_metric_3d(dx_window, dy_window, dz_window, coordinates: Coordinates = CARTESIAN) -> WindowMetric3D:
    """WindowMetric3D of the window cell widths: x, y, z [m], or on a polar grid x = r - radius_min [m], y = phi [rad]
    (at most 2 pi, angle_wrap allowed), z [m] (a cylindrical sector; r faces r_f dphi dz, volumes r_c dr dphi dz)."""
    widths = [np.asarray(axis_widths, dtype=np.float64) for axis_widths in (dx_window, dy_window, dz_window)]
    if any(axis_widths.ndim != 1 or not axis_widths.size for axis_widths in widths):
        raise ValueError(f"window_metric_3d needs non-empty 1D cell widths, got shapes "
                         f"{[axis_widths.shape for axis_widths in widths]}")
    if not all(np.all(axis_widths > 0.0) for axis_widths in widths):
        raise ValueError("window_metric_3d needs positive cell widths")
    if coordinates.line:
        raise ValueError('a 1D line window is a 2D Domain one cell across; a 3D window has no line form')
    dx, dy, dz = widths
    ones = [np.ones(axis_widths.size) for axis_widths in widths]
    face_ones = [np.ones(axis_widths.size + 1) for axis_widths in widths]
    if not coordinates.polar:
        face_area = (Separable3(face_ones[0], dy, dz), Separable3(dx, face_ones[1], dz),
                     Separable3(dx, dy, face_ones[2]))
        length = (Separable3(dx, ones[1], ones[2]), Separable3(ones[0], dy, ones[2]), Separable3(ones[0], ones[1], dz))
        return WindowMetric3D(coordinates, face_area, length, Separable3(dx, dy, dz))
    span = float(np.sum(dy))
    if span > 2.0 * np.pi + AXIS_ANGLE_SNAP:
        raise ValueError(f"a cylindrical sector spans at most 2 pi in phi, got {span!r} rad")
    r_faces = face_positions(dx, coordinates.radius_min)
    r_centres = 0.5 * (r_faces[:-1] + r_faces[1:])
    annulus = r_centres * dx   # exact sector area per radian (r_hi^2 - r_lo^2) / 2
    face_area = (Separable3(r_faces, dy, dz), Separable3(dx, face_ones[1], dz), Separable3(annulus, dy, face_ones[2]))
    length = (Separable3(dx, ones[1], ones[2]), Separable3(r_centres, dy, ones[2]), Separable3(ones[0], ones[1], dz))
    return WindowMetric3D(coordinates, face_area, length, Separable3(annulus, dy, dz))

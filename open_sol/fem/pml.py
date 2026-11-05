"""Physics-neutral PML coordinate stretches (new_modules_plan S1): PMLStretch of the layer ring of a TriangleMesh (planar
or the axisymmetric stretched radius) or the shell of a TetrahedronMesh, with the coefficient transforms of a scalar
operator -div(a grad u) + b u. Host numpy."""
# Stretch (exp(+j omega t)): s = 1 - j beta (d / w)^n at depth d into the layer of face f (thickness w, n =
# PML_GRADING_ORDER) along that face's axis, 1 elsewhere: the map of physics/mechanics/pml3d. An outgoing wave
# exp(-j k x) decays there as exp(-k beta w (d / w)^(n + 1) / (n + 1)). beta is fixed at build from a reference angular
# frequency omega_r, the wave speed c and a target round-trip reflection R: beta = (n + 1) ln(1 / R) c / (2 omega_r w)
# (pml_strength). It does not depend on omega, so K and M stay frequency independent and an eigen pencil stays linear
# in omega^2; higher frequencies and slower waves are absorbed more, lower ones less (eigen studies: omega_r = the
# target; frequency response: the lowest sweep frequency). Scalar operator: d/dx_i -> d/dx_i / s_i and dV -> J dV (J =
# prod s_i), so a_ij -> a_ij J / (s_i s_j) and b -> b J (tensor_factor, volume_factor), unconjugated: the forms stay
# complex symmetric. Axisymmetric (r, z) = (x, y): the ring weight r -> r~ = r - j beta w (d / w)^(n + 1) / (n + 1)
# beyond x_hi (stretched_coordinates), applied as the factor r~ / r on top of the assembler's own r weight; a layer on
# x_lo (r < 0) is refused there.
import math
from dataclasses import dataclass

import numpy as np

from .mesh import FACE_BITS

PML_GRADING_ORDER = 2
DEFAULT_PML_REFLECTION = 1e-6
AXES = 'xyz'


def pml_strength(width, wave_speed, omega_reference, reflection=DEFAULT_PML_REFLECTION) -> float:
    """beta of a layer of thickness width [m] > 0: round-trip reflection `reflection` in (0, 1) of a normally incident
    wave of speed wave_speed [m/s] at omega_reference [rad/s] (module header)."""
    if not (math.isfinite(omega_reference) and omega_reference > 0.0):
        raise ValueError(f'a PML needs a reference angular frequency > 0 (its strength is set there), got '
                         f'{omega_reference!r} rad/s')
    if not (0.0 < reflection < 1.0):
        raise ValueError(f'PML reflection must lie in (0, 1), got {reflection!r}')
    if not (math.isfinite(wave_speed) and wave_speed > 0.0):
        raise ValueError(f'PML wave speed must be a finite number > 0 [m/s], got {wave_speed!r}')
    if not width > 0.0:
        raise ValueError(f'PML layer thickness must be > 0, got {width!r} m')
    return (PML_GRADING_ORDER + 1) * math.log(1.0 / reflection) * wave_speed / (2.0 * omega_reference * width)


@dataclass(frozen=True, eq=False)
class PMLStretch:
    """Fixed complex stretch of the layers beyond a window: window (x0, y0[, z0], x1, y1[, z1]) [m], thickness [m] and
    strength beta per outer face (0 = no layer); 2D or 3D by the window length."""
    window: tuple
    widths: dict
    strength: dict

    @property
    def dimensions(self) -> int:
        return len(self.window) // 2

    @property
    def faces(self) -> tuple:
        """Faces with a layer, in AXES order (lo before hi)."""
        return tuple(face for face in _faces(self.dimensions) if self.widths.get(face, 0.0) > 0.0)

    def _layers(self, coordinate, axis):
        """Yield (face, inside (bool like coordinate), fraction d / w clipped to [0, 1], sign: -1 lo, +1 hi)."""
        name = AXES[axis]
        low, high = self.window[axis], self.window[self.dimensions + axis]
        for face, depth, sign in ((f'{name}_lo', low - coordinate, -1.0), (f'{name}_hi', coordinate - high, 1.0)):
            width = self.widths.get(face, 0.0)
            if width <= 0.0:
                continue
            inside = depth > 0.0
            if inside.any():
                yield face, inside, np.clip(depth[inside] / width, 0.0, 1.0), sign

    def stretches(self, points, elements=None) -> np.ndarray:
        """Complex128 (..., d) (s_x, s_y[, s_z]) at points (..., d) [m] (1 inside the window); `elements` unused (the
        callable form of fem.assembly coefficients and the fem.elasticity3d stretch)."""
        points = self._points(points)
        result = np.ones(points.shape, dtype=np.complex128)
        for axis in range(self.dimensions):
            column = result[..., axis]
            for face, inside, fraction, _ in self._layers(points[..., axis], axis):
                column[inside] = 1.0 - 1j * self.strength[face] * fraction ** PML_GRADING_ORDER
        return result

    def stretch_derivatives(self, points) -> np.ndarray:
        """Complex128 (..., d) ds_i / dx_i at points (..., d) [m]: -/+ j beta n f^(n - 1) / w on lo / hi layers (f = d /
        w; n = 2, so s is C1 and s' = 0 at the window edge), 0 elsewhere (plate bending's stretched Hessian)."""
        points = self._points(points)
        result = np.zeros(points.shape, dtype=np.complex128)
        for axis in range(self.dimensions):
            column = result[..., axis]
            for face, inside, fraction, sign in self._layers(points[..., axis], axis):
                column[inside] = (-1j * sign * self.strength[face] * PML_GRADING_ORDER
                                  * fraction ** (PML_GRADING_ORDER - 1) / self.widths[face])
        return result

    def stretched_coordinates(self, points) -> np.ndarray:
        """Complex128 (..., d) x~ = window edge + the integral of s from it (= x inside the window): lo layers add
        + j beta w f^(n + 1) / (n + 1), hi layers - j beta w f^(n + 1) / (n + 1) (f = d / w)."""
        points = self._points(points)
        result = points.astype(np.complex128)
        for axis in range(self.dimensions):
            column = result[..., axis]
            for face, inside, fraction, sign in self._layers(points[..., axis], axis):
                shift = self.strength[face] * self.widths[face] * fraction ** (PML_GRADING_ORDER + 1) / (
                    PML_GRADING_ORDER + 1)
                column[inside] = column[inside] - sign * 1j * shift
        return result

    def jacobian(self, points, elements=None) -> np.ndarray:
        """Complex128 (...,) J = s_x s_y [s_z] at points (..., d) (the stretched volume element)."""
        return np.prod(self.stretches(points), axis=-1)

    def radial_ratio(self, points) -> np.ndarray:
        """Complex128 (...,) r~ / r of axisymmetric points (x = r >= 0; 1 where r is unstretched, also on the axis);
        ValueError for a layer on x_lo (r < 0)."""
        if self.widths.get('x_lo', 0.0) > 0.0:
            raise ValueError('an axisymmetric PML cannot have a layer on x_lo (r < 0): the axis lies there')
        points = self._points(points)
        radius = points[..., 0]
        stretched = self.stretched_coordinates(points)[..., 0]
        ratio = np.ones(radius.shape, dtype=np.complex128)
        moved = stretched != radius
        ratio[moved] = stretched[moved] / radius[moved]
        return ratio

    def volume_factor(self, points, axisymmetric=False) -> np.ndarray:
        """Complex128 (...,) factor of a scalar (mass-type) coefficient b -> b J (x r~ / r when axisymmetric)."""
        factor = self.jacobian(points)
        return factor * self.radial_ratio(points) if axisymmetric else factor

    def tensor_factor(self, points, axisymmetric=False) -> np.ndarray:
        """Complex128 (..., d, d) factor of a stiffness tensor a_ij -> a_ij J / (s_i s_j) (x r~ / r when
        axisymmetric); a diagonal a gets diag(s_y s_z / s_x, ...)."""
        stretches = self.stretches(points)   # once: J = prod(s) from them, not a second stretches() call
        inverse = 1.0 / stretches
        factor = np.prod(stretches, axis=-1)
        if axisymmetric:
            factor = factor * self.radial_ratio(points)
        return factor[..., None, None] * inverse[..., :, None] * inverse[..., None, :]

    def in_layer(self, points) -> np.ndarray:
        """Bool (...,) points beyond the window on a face with a layer."""
        points = self._points(points)
        result = np.zeros(points.shape[:-1], dtype=bool)
        for axis in range(self.dimensions):
            for _, inside, _, _ in self._layers(points[..., axis], axis):
                result |= inside
        return result

    def text(self) -> str:
        """'x_hi 0.34 m, beta 3.3' per layer (study logs)."""
        return ', '.join(f'{face} {self.widths[face]:.4g} m, beta {self.strength[face]:.3g}' for face in self.faces)

    def _points(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != self.dimensions:
            raise ValueError(f'PMLStretch of dimension {self.dimensions}: points must have {self.dimensions} '
                             f'coordinates, got shape {points.shape}')
        return points


def _faces(dimensions) -> tuple:
    return tuple(f'{AXES[axis]}_{side}' for axis in range(dimensions) for side in ('lo', 'hi'))


def pml_stretch(window, widths, wave_speed, omega_reference, reflection=DEFAULT_PML_REFLECTION) -> PMLStretch:
    """PMLStretch of layers `widths` {face: thickness [m], 0 = none} beyond window; wave_speed [m/s] a number or
    {face: speed} (the fastest wave in that layer); strength per face by pml_strength."""
    window = tuple(float(value) for value in window)
    if len(window) not in (4, 6):
        raise ValueError(f'pml_stretch: window must be (x0, y0, x1, y1) or (x0, y0, z0, x1, y1, z1), got {window!r}')
    dimensions = len(window) // 2
    unknown = [face for face in widths if face not in _faces(dimensions)]
    if unknown:
        raise ValueError(f"pml_stretch: unknown face(s) {', '.join(unknown)} for a {dimensions}D window")
    full_widths, strength = {}, {}
    for face in _faces(dimensions):
        width = float(widths.get(face, 0.0))
        if width < 0.0:
            raise ValueError(f'pml_stretch: layer thickness of {face} must be >= 0, got {width!r} m')
        speed = wave_speed.get(face, 0.0) if isinstance(wave_speed, dict) else wave_speed
        full_widths[face] = width
        strength[face] = pml_strength(width, float(speed), float(omega_reference), reflection) if width > 0.0 else 0.0
    return PMLStretch(window=window, widths=full_widths, strength=strength)


def ring_widths(mesh) -> dict:
    """{face: layer thickness [m], 0 = none} of a TriangleMesh's PML layer ring (pml_faces bits)."""
    x0, y0, x1, y1 = (float(value) for value in mesh.window)
    window_edges = {'x_lo': x0, 'x_hi': x1, 'y_lo': y0, 'y_hi': y1}
    widths = {}
    for face, bit in FACE_BITS.items():
        ring = (mesh.pml_faces & bit) != 0
        width = 0.0
        if ring.any():
            coordinates = mesh.points[mesh.triangles[ring]][..., 0 if face[0] == 'x' else 1]
            width = float(window_edges[face] - coordinates.min() if face.endswith('lo')
                          else coordinates.max() - window_edges[face])
        widths[face] = max(width, 0.0)
    return widths


def shell_widths(mesh) -> dict:
    """{face: shell thickness [m], 0 = none} of a TetrahedronMesh (node bounds beyond the window, 1e-9 x the longest
    bound side of slack)."""
    bounds, window = mesh.bounds, tuple(float(value) for value in mesh.window)
    slack = 1e-9 * max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    widths = {}
    for axis, name in enumerate(AXES):
        for face, width in ((f'{name}_lo', window[axis] - bounds[axis]),
                            (f'{name}_hi', bounds[3 + axis] - window[3 + axis])):
            widths[face] = width if width > slack else 0.0
    return widths


def _checked_widths(widths, faces, what) -> dict:
    """widths, ValueError when the layers present differ from `faces` (None = take the mesh's)."""
    if faces is None:
        return widths
    for face, width in widths.items():
        if (width > 0.0) != (face in faces):
            raise ValueError(f"{what} PML layer on {face}: {'present' if width > 0.0 else 'missing'}, but the PML "
                             f"is {'off' if width > 0.0 else 'on'} there; mesh and PML settings disagree")
    return widths


def pml_ring(mesh, wave_speed, omega_reference, reflection=DEFAULT_PML_REFLECTION, faces=None) -> PMLStretch:
    """PMLStretch of a TriangleMesh's layer ring (pml_stretch arguments); faces (None = the ring's) must equal the
    ringed faces, else ValueError."""
    widths = _checked_widths(ring_widths(mesh), faces, 'triangle mesh')
    return pml_stretch(mesh.window, widths, wave_speed, omega_reference, reflection)


def pml_shell(mesh, wave_speed, omega_reference, reflection=DEFAULT_PML_REFLECTION, faces=None) -> PMLStretch:
    """PMLStretch of a TetrahedronMesh's shell (pml_stretch arguments); faces (None = the shell's) must equal the
    shelled faces, else ValueError."""
    widths = _checked_widths(shell_widths(mesh), faces, 'tetrahedron mesh')
    return pml_stretch(mesh.window, widths, wave_speed, omega_reference, reflection)

"""Host field containers of the Session reads (FieldData, LineProfile, VolumeData) and the cut-plane axis names;
model.session re-exports them."""
from dataclasses import dataclass

import numpy as np

from ..core.backend import to_host
from ..core.renderer import FieldInfo

COORDINATE_AXES = ('x', 'y', 'z')
PLANE_AXES = {'z': ('x', 'y'), 'y': ('x', 'z'), 'x': ('y', 'z')}   # in-plane (u, v) axes of a cut normal to axis


@dataclass
class FieldData:
    """Host copy of one field over the physical window; values[i, j] = (u, v) along `axes` ((x, y) unless a 3D side
    cut), extent (u0, u1, v0, v1) [m]; x_centers / y_centers are the u / v centres. A polar / spherical window comes
    as a containing-cell image over its drawing-plane box (NaN outside the sector, native_shape set), or with
    field(native=True) on its cells: axes ('r', 'phi' | 'theta'), v in rad, mapped_centres set."""
    physics: str
    component: str
    values: np.ndarray          # float32 2D, fresh copy
    extent: tuple
    info: FieldInfo
    time: float
    time_label: str | None
    x_centers: np.ndarray | None = None   # cell centres when values sit on native (possibly non-uniform) cells;
    y_centers: np.ndarray | None = None   # None = uniform cells over extent
    native_shape: tuple | None = None     # (nu, nv) window cells when values are a nearest-cell display resampling
    plane: tuple | None = None            # 3D models: (normal axis 'x' | 'y' | 'z', cell-centre position [m]); None 2D
    axes: tuple = ('x', 'y')              # names of the values' first / second axis (PLANE_AXES of the plane)
    unfolded: bool = False                # already mirrored over the mirror faces (core.mirror.unfold_field)
    mapped_centres: tuple | None = None   # native polar / spherical cells: drawing-plane (x, y) centres [m], float64
                                          # (nu, nv) each; None otherwise
    mesh: str = ''                        # model mesh the field lives on (S7.2); '' outside a Session / older saves
    triangulation: tuple | None = None    # triangle meshes: (nodes (n, 2) float64 [m], triangles (k, 3) int32, node
                                          # values (n,) float32) of the display triangulation; values = its raster


@dataclass
class LineProfile:
    """Host field profile along a 3D segment (Session.segment_profile): containing-cell (or trilinear) values, NaN
    outside the window; centres = (x, y, z) [m] of the sampled cells (trilinear: the sample points)."""
    physics: str
    component: str
    distance: np.ndarray        # float64 [m] from start
    values: np.ndarray          # float64
    centres: tuple              # (x, y, z) float64 arrays [m], NaN outside the window
    start: tuple                # (x, y, z) [m]
    end: tuple
    info: FieldInfo
    time: float
    time_label: str | None


@dataclass
class VolumeData:
    """Host copy of a 3D field on the window cells (Session.volume): values[i, j, k] at centres (x, y, z) [m]."""
    physics: str
    component: str
    values: np.ndarray          # float32 (wx, wy, wz)
    centres: tuple              # (x, y, z) float64 cell-centre arrays [m]
    info: FieldInfo
    time: float
    time_label: str | None


def host_float32(device_values):
    """Fresh C-contiguous float32 numpy copy of a device (or numpy) array."""
    return np.array(to_host(device_values), dtype=np.float32, order='C')

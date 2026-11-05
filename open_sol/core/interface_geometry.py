"""Interface geometry of subpixel smoothing (igor 1.5): per averaging location whose volume two materials share, the
fill fraction, normal and two representative cells (2D: InterfaceGeometry, 3D: InterfaceGeometry3D); built once by
model.subpixel_geometry, averaged by the physics."""
from dataclasses import dataclass

# Device bytes per location of an InterfaceSet (int64 location, 3 x int32 cells, 4 x float32).
BYTES_PER_LOCATION = 8 + 3 * 4 + 4 * 4


@dataclass(frozen=True, eq=False)
class InterfaceSet:
    """Interface locations of one kind (cupy 1D arrays, one entry per location, sorted by location): int64 flat
    full-grid index i * ny + j of the location (a face is stored at its lower cell), int32 flat cells holding the
    dominant material a / the second material b, float32 fill (volume fraction of a among a and b), float32
    normal_square (n_d^2 of the field component d; 0 for cells), int32 upper cell of the face mean (cells: itself)
    and float32 lower / upper shares of that mean (yee_grid.face_shares; cells 1 / 0)."""
    location: object
    cell_a: object
    cell_b: object
    fill: object
    normal_square: object
    upper_cell: object
    lower_share: object
    upper_share: object

    @property
    def size(self) -> int:
        return int(self.location.size)


@dataclass(frozen=True, eq=False)
class InterfaceGeometry:
    """InterfaceSets of a 2D build per location kind: 'cells' (cell centres, E_z), 'faces_y' (hi-y faces: H_z
    polarisation E_x, normal component n_x), 'faces_x' (hi-x faces: E_y, n_y); supersampling = (fine, coarse) samples
    per cell of the rasters. Static for the session (couplings change values, not shapes)."""
    cells: InterfaceSet
    faces_y: InterfaceSet
    faces_x: InterfaceSet
    supersampling: tuple

    def kind(self, name) -> InterfaceSet:
        """InterfaceSet of 'cells' | 'faces_y' | 'faces_x'."""
        return getattr(self, name)


# Device bytes per location of an InterfaceSet3D (int64 location, 2 x int64 cells, 2 x float32).
BYTES_PER_LOCATION_3D = 3 * 8 + 2 * 4


@dataclass(frozen=True, eq=False)
class InterfaceSet3D:
    """Interface faces of one E component of a 3D build (cupy 1D arrays sorted by location): int64 flat lower cell
    (i ny + j) nz + k of the face, int64 flat cells holding material a / b, float32 fill (volume fraction of a among a
    and b in the face's dual volume) and normal_square (n_c^2 of the component c the face carries)."""
    location: object
    cell_a: object
    cell_b: object
    fill: object
    normal_square: object

    @property
    def size(self) -> int:
        return int(self.location.size)


@dataclass(frozen=True, eq=False)
class InterfaceGeometry3D:
    """InterfaceSet3D of the E_x hi-x ('faces_x'), E_y hi-y ('faces_y') and E_z hi-z ('faces_z') faces of a 3D build
    (em_fdtd_3d layout); supersampling = (fine, coarse) in-plane samples per cell, z_samples per half cell. Static."""
    faces_x: InterfaceSet3D
    faces_y: InterfaceSet3D
    faces_z: InterfaceSet3D
    supersampling: tuple
    z_samples: int

    def kind(self, name) -> InterfaceSet3D:
        """InterfaceSet3D of 'faces_x' | 'faces_y' | 'faces_z'."""
        return getattr(self, name)

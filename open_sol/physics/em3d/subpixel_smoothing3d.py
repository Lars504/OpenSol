"""Subpixel smoothing of em_fdtd_3d (igor 1.5): the averaged eps of the build's interface faces
(core.interface_geometry.InterfaceGeometry3D) from the engine's current table values, re-quantised (material_index3d)
into a uint16 face index per lower cell (0 = none) and a per-id (eps_x, eps_y, eps_z) table the E coefficients read."""
import numpy as np
from ...core.backend import array_module_of, cp

from ...core.constants import DTYPE
from ...core.kernels import twin
from .material_index3d import build_material_index

FACE_KINDS = ('faces_x', 'faces_y', 'faces_z')   # E_x / E_y / E_z faces (component axis 0 / 1 / 2)
MAX_FACE_IDS = 65535                            # uint16 face index; id 0 = no override


def interface_eps(eps_a, eps_b, fill, normal_square):
    """Kottke average 1 / (n^2 <1/eps> + (1 - n^2) / <eps>) of two materials' eps (numpy, the arrays' dtype); fill =
    volume fraction of a, normal_square = n_c^2 of the face's component (em_fdfd_3d and the host twin below)."""
    mean = fill * eps_a + (1.0 - fill) * eps_b
    inverse_mean = fill / eps_a + (1.0 - fill) / eps_b
    return 1.0 / (normal_square * inverse_mean + (1.0 - normal_square) / mean)


# interface_eps of the two representative cells' table eps; 0 (no override) when either is PEC.
def _smoothed_3d_host(material_id, table, cell_a, cell_b, fill, normal_square):
    eps_a, eps_b = table[material_id[cell_a]], table[material_id[cell_b]]
    one = np.float32(1.0)
    conductor = (eps_a <= 0.0) | (eps_b <= 0.0)
    eps_a, eps_b = np.where(conductor, one, eps_a), np.where(conductor, one, eps_b)   # no division by a PEC entry
    smoothed = interface_eps(eps_a, eps_b, fill, normal_square)
    return np.where(conductor, np.float32(0.0), smoothed).astype(DTYPE, copy=False)


_SMOOTHED_3D = twin('em3d.subpixel_smoothed', cp.ElementwiseKernel(
    'raw uint16 material_id, raw float32 table, int64 cell_a, int64 cell_b, float32 fill, float32 normal_square',
    'float32 smoothed',
    '''
    float eps_a = table[material_id[cell_a]];
    float eps_b = table[material_id[cell_b]];
    if (eps_a <= 0.0f || eps_b <= 0.0f) {
        smoothed = 0.0f;
    } else {
        float mean = fill * eps_a + (1.0f - fill) * eps_b;
        float inverse_mean = fill / eps_a + (1.0f - fill) / eps_b;
        smoothed = 1.0f / (normal_square * inverse_mean + (1.0f - normal_square) / mean);
    }
    ''',
    'em3d_subpixel_smoothed'), _smoothed_3d_host)


class SubpixelSmoothing3D:
    """Face overrides of one FDTDEngine3D: `face_id` (uint16 full grid, None without interface faces) and `table`
    (float32 (n_ids + 1) x 3 flat, row 0 = no override; a 0 entry keeps that component's face mean), rebuilt by
    refresh(engine) after every material index rebuild; quantisation_error = max deviation of the last refresh."""

    def __init__(self, geometry, shape):
        self.geometry = geometry
        locations = [geometry.kind(kind).location for kind in FACE_KINDS]
        self.xp = xp = array_module_of(locations[0])   # the geometry's backend (the build's)
        self.cells = xp.unique(xp.concatenate(locations))   # sorted lower cells holding any override
        self._positions = [xp.searchsorted(self.cells, location) for location in locations]
        n_cells = int(self.cells.size)
        self.face_id = xp.zeros(shape, dtype=np.uint16) if n_cells else None
        self.table = xp.zeros(3, dtype=DTYPE)
        self._values = [xp.zeros(n_cells, dtype=DTYPE) for _ in FACE_KINDS]
        self._ids = xp.zeros(n_cells, dtype=np.uint16)
        self._delta_hints = {}
        self.quantisation_error = None

    @property
    def size(self) -> int:
        """Number of cells holding an override."""
        return int(self.cells.size)

    def refresh(self, engine):
        """Averaged face eps of the engine's current material index / tables, re-quantised into face_id / table."""
        if self.face_id is None:
            return
        tables = (engine.table_eps_x, engine.table_eps_y, engine.table_eps_z)
        material_id = engine.material_id.ravel()
        for axis, kind in enumerate(FACE_KINDS):
            values = self._values[axis]
            values.fill(0)
            locations = self.geometry.kind(kind)
            if locations.size:
                values[self._positions[axis]] = _SMOOTHED_3D(material_id, tables[axis], locations.cell_a,
                                                             locations.cell_b, locations.fill, locations.normal_square)
        arrays = dict(zip('xyz', self._values))
        id_tables, errors, deltas = build_material_index(arrays, [(slice(None),)], self._ids, MAX_FACE_IDS - 1,
                                                         self._delta_hints)
        self._delta_hints.update(deltas)
        n_ids = int(id_tables['x'].size)
        table = self.xp.zeros((n_ids + 1, 3), dtype=DTYPE)
        for axis, name in enumerate('xyz'):
            table[1:, axis] = id_tables[name]
        self.table = table.ravel()
        self.face_id.fill(0)
        self.face_id.ravel()[self.cells] = self._ids + np.uint16(1)
        self.quantisation_error = max(errors.values()) if errors else None

    def plane_values(self, plane_ids, axis):
        """float32 override eps of component `axis` for a plane of face ids (integer array); 0 = none."""
        return self.table.reshape(-1, 3)[plane_ids.astype(np.int64), axis]

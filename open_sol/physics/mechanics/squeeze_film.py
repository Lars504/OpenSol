"""Runtime of the solid_mechanics_3d feature squeeze_film_damping (new_modules_plan S6w, tetrahedra): SqueezeFilmDamping
finds the flat footprint faces on the gap side, builds a fem.surface_mesh footprint with its fem.squeeze_film.SqueezeFilm
(vents from the rim) and evaluates the gas damping rows per resonance; vent choices and the face filter."""
# The physics-neutral film (equation, slip, SqueezeFilm, mode_gas_damping, gas_damping_totals, the table columns) lives
# in fem/squeeze_film.py (shared with the membrane feature). Candidate tet faces (window faces of an outer face, object
# interface faces) are filtered to the flat faces on the gap side (`film_faces`: normal along the side's axis, the solid
# element behind them, the outermost such plane), then one SurfaceMesh / SqueezeFilm (P2) is built on first evaluation
# from the solid of the last assembly (rebuilt when it changes). Vents: 'all' rim edges, 'none', or the rim minus the
# edges lying on window planes (every in-plane axis: 'window_sealed', one axis: '<axis>_sealed'): sealed rims are mirror
# planes of a half / quarter model. Per resonance: w = -u . n_out of the mode (peak |u| 1 m), m_eff = Re(psi^H M psi)
# over every dof of the pencil (piezoelectric: the massless potential adds nothing), omega = Re omega. Coordinates are the
# mesh's (fem.surface_mesh keeps the tet in-plane coordinates); planar integrals (no r weight).
import numpy as np

from ...core.constants import FACES_3D
from ...fem.mesh import side_members
from ...fem.mesh3d import PLANE_TOLERANCE
from ...fem.squeeze_film import SqueezeFilm, film_table, mode_gas_damping
from ...fem.surface_mesh import normal_displacement, surface_triangle_mesh

GAP_SIDES = ('auto',) + tuple(FACES_3D)   # feature gap_side: auto = the selected outer face
VENT_CHOICES = ('all', 'window_sealed', 'x_sealed', 'y_sealed', 'z_sealed', 'none')
FILM_ORDER = 2                            # Lagrange order of the film pressure (any solid order)
_FLAT_TOLERANCE = 1e-9   # 1 - |n . e_axis| of a face counted as flat on an axis plane


def film_faces(tet_mesh, faces, solid, side) -> np.ndarray:
    """Int64 faces of a candidate set flat on a plane normal to the axis of side ('x_lo' ... 'z_hi'), with exactly one
    solid element (host bool per element) beside them, outward normal towards side, on the outermost such plane."""
    faces = np.unique(np.asarray(faces, dtype=np.int64).ravel())
    if not faces.size:
        return faces
    axis, direction = 'xyz'.index(side[0]), (1.0 if side.endswith('hi') else -1.0)
    solid = np.asarray(solid, dtype=bool)
    pairs = tet_mesh.face_elements[faces].astype(np.int64)
    first, second = side_members(pairs, solid)
    owners = np.where(first, pairs[:, 0], pairs[:, 1])
    corners = tet_mesh.points[tet_mesh.faces[faces].astype(np.int64)]   # (k, 3, 3)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    flat = np.abs(normals[:, axis]) >= (1.0 - _FLAT_TOLERANCE) * lengths
    planes = corners[:, :, axis].mean(axis=1)
    keep = (first != second) & flat
    if keep.any():
        outward = np.sign(planes[keep] - tet_mesh.centroids(owners[keep])[:, axis]) == direction
        keep[np.flatnonzero(keep)[~outward]] = False
    if not keep.any():
        return faces[keep]
    bounds = np.array(tet_mesh.bounds)
    slack = PLANE_TOLERANCE * float((bounds[3:] - bounds[:3]).max())
    outermost = planes[keep].max() if direction > 0.0 else planes[keep].min()
    return faces[keep & (np.abs(planes - outermost) <= slack)]


def sealed_rim_axes(vents, normal_axis) -> tuple:
    """Tet axes whose window-plane rim edges a vents choice seals (() for 'all' / 'none'); ValueError for the normal."""
    if vents not in VENT_CHOICES:
        raise ValueError(f"vents must be one of {', '.join(VENT_CHOICES)}, got {vents!r}")
    in_plane = tuple(axis for axis in range(3) if axis != normal_axis)
    if vents == 'window_sealed':
        return in_plane
    if vents.endswith('_sealed'):
        axis = 'xyz'.index(vents[0])
        if axis == normal_axis:
            raise ValueError(f"vents {vents!r} names the film's normal axis {vents[0]}: its rim lies in no "
                             f'{vents[0]} = const window plane; seal an in-plane axis')
        return (axis,)
    return ()


class SqueezeFilmDamping:
    """Runtime of one squeeze_film_damping feature: candidate tet faces, gap side, the SqueezeFilm inputs (ambient None
    = incompressible) and vents; the SurfaceMesh / SqueezeFilm built on first evaluation; rows of the last solve."""

    def __init__(self, name, faces, side, gap, viscosity, ambient_pressure=None, *, mean_free_path=0.0,
                 slip='veijola', vents='all'):
        if side not in FACES_3D:
            raise ValueError(f"squeeze film '{name}': gap side must be one of {', '.join(FACES_3D)}, got {side!r}")
        self.name = str(name)
        self.candidate_faces = np.unique(np.asarray(faces, dtype=np.int64).ravel())
        self.side = side
        self.inputs = dict(gap=float(gap), viscosity=float(viscosity),
                           ambient_pressure=None if ambient_pressure is None else float(ambient_pressure),
                           mean_free_path=float(mean_free_path), slip=slip)
        sealed_rim_axes(vents, 'xyz'.index(side[0]))   # fails early on a bad choice
        self.vents = vents
        self.surface = None   # fem.surface_mesh.SurfaceMesh of the footprint
        self.film = None      # SqueezeFilm on it
        self.rows = None      # mode_gas_damping rows of the last evaluated solve
        self._solid = None    # the solid mask the footprint was built from

    def label(self):
        return f"squeeze film '{self.name}'"

    def build(self, tet_mesh, solid) -> SqueezeFilm:
        """The SqueezeFilm of the footprint for this solid mask (kept until the mask changes)."""
        solid = np.asarray(solid, dtype=bool)
        if self.film is not None and np.array_equal(solid, self._solid):
            return self.film
        faces = film_faces(tet_mesh, self.candidate_faces, solid, self.side)
        if not faces.size:
            raise ValueError(f'{self.label()}: none of the {self.candidate_faces.size} selected faces is a flat face of '
                             f'the solid whose outward normal points to {self.side} (gap side); select the face beside '
                             'the gap or set gap_side')
        axis = 'xyz'.index(self.side[0])
        try:
            surface = surface_triangle_mesh(tet_mesh, faces, solid, axis=axis)
        except ValueError as exc:
            raise ValueError(f'{self.label()}: {exc}') from None
        self.film = SqueezeFilm(surface.mesh, self.inputs['gap'], self.inputs['viscosity'],
                                self.inputs['ambient_pressure'], mean_free_path=self.inputs['mean_free_path'],
                                slip=self.inputs['slip'], vent_edges=self._vent_edges(surface, tet_mesh),
                                order=FILM_ORDER, label=self.label)
        self.surface, self._solid = surface, solid.copy()
        return self.film

    def _vent_edges(self, surface, tet_mesh):
        """Vented rim edges of the footprint (None = every rim edge)."""
        if self.vents == 'all':
            return None
        mesh = surface.mesh
        rim = mesh.boundary_edges().astype(np.int64)
        if self.vents == 'none':
            return rim[:0]
        window = np.asarray(tet_mesh.window, dtype=np.float64)
        slack = PLANE_TOLERANCE * float((window[3:] - window[:3]).max())
        ends = tet_mesh.points[surface.nodes[mesh.edges[rim].astype(np.int64)]]   # (k, 2, 3)
        sealed = np.zeros(len(rim), dtype=bool)
        for axis in sealed_rim_axes(self.vents, surface.axis):
            for plane in (window[axis], window[axis + 3]):
                sealed |= (np.abs(ends[:, :, axis] - plane) <= slack).all(axis=1)
        return rim[~sealed]

    @property
    def area(self) -> float:
        """Footprint area [m^2] (0 before the first build)."""
        return 0.0 if self.surface is None else float(np.abs(self.surface.mesh.areas()).sum())

    def evaluate(self, tet_mesh, solid, resonances, mass, n_vector) -> list:
        """Rows (mode_gas_damping, effective_mass [kg] included) per resonance (MechanicalResonance: displacement over
        every pencil dof, the first n_vector the interleaved u) with the pencil's host mass CSR; kept as rows."""
        film = self.build(tet_mesh, solid)
        shapes, omegas, masses = [], [], []
        for resonance in resonances:
            mode = np.asarray(resonance.displacement)
            shapes.append(normal_displacement(self.surface, mode[:n_vector]))
            omegas.append(float(resonance.omega))
            masses.append(float(np.real(np.vdot(mode, mass @ mode))))
        self.rows = mode_gas_damping(film, shapes, omegas, masses)
        return self.rows

    def table(self) -> dict:
        """'table' result of the rows (FILM_TABLE_COLUMNS) with extras area [m^2], gap [m], surface_elements and
        vent_edges; ValueError before an evaluation."""
        if self.rows is None:
            raise ValueError(f'{self.label()}: not evaluated yet: the gas damping is computed per resonance by an '
                             'eigenfrequency study')
        return film_table(self.rows, area=self.area, gap=self.inputs['gap'],
                          surface_elements=int(self.surface.mesh.n_elements), vent_edges=int(len(self.film.vent_edges)))

"""Surface mode shapes of the tetrahedra solid (feature mode_shapes, result kind 'mode_shapes'): the boundary
triangles of the solid window elements at the displacement's Lagrange resolution and the complex nodal displacement of
every resonance of the last eigenfrequency solve on them, with the Bloch wavenumbers and periods of the cell. Host
numpy."""
# Surface = the faces with exactly one solid window element (fem.mesh3d.interface_faces of solid_elements & window:
# solid / void interfaces, the outer box faces of solid elements, the window side of a PML shell). Nodes = the scalar
# full-numbering Lagrange dofs of those faces (P2: the three vertices and the three edge dofs n_points + edge; P1:
# vertices only), renumbered compactly in ascending dof order; positions = vertices / edge midpoints (affine elements:
# straight-sided). P2 splits each face 1 -> 4 at its edge midpoints, so the quadratic displacement is drawn at its own
# resolution; every triangle is counter-clockwise seen from outside the solid (normal out of the solid element).
# Displacement = rows of the resonance's (n_dofs, 3) block (a piezoelectric potential dropped), the solver's
# normalisation kept (peak node |u| = 1 m, its largest component real positive): no interpolation, no copy of the
# volume vectors. The surface is cached on the feature's ModeShapes per (dof map, solid set).
from dataclasses import dataclass

import numpy as np

from ...fem.mesh3d import interface_faces

MODE_SHAPES_KIND = 'mode_shapes'
# fem.dofs.bloch_master_map: each hi-side dof is its lo-side partner times exp(-j k L) per crossed period
PHASE_CONVENTION = 'u(x + L) = u(x) exp(-j k L), time exp(+j omega t)'
_COMPONENTS = 3   # interleaved displacement dofs 3 d + component


@dataclass(frozen=True, eq=False)
class SolidSurface:
    """Display surface of the solid: scalar dof per node, node positions and the (split) triangles."""
    dofs: np.ndarray        # int64 (n,) full-numbering scalar Lagrange dof of each node (ascending)
    nodes: np.ndarray       # float64 (n, 3) [m]
    triangles: np.ndarray   # int32 (m, 3) indices into nodes, counter-clockwise seen from outside the solid


def solid_surface(dofs, solid) -> SolidSurface:
    """SolidSurface of a Lagrange P1 / P2 DofMap3D over the elements where solid (bool per element) is set;
    ValueError when no face bounds a solid element."""
    mesh = dofs.mesh
    solid = np.asarray(solid, dtype=bool)
    faces = interface_faces(mesh, solid)
    if faces.size == 0:
        raise ValueError('mode shapes: the solid has no surface (no element with stiffness in the window)')
    local = dofs.face_local_dofs(faces)   # (k, 3 | 6): nodes a < b < c, then (P2) edges ab, ac, bc
    elements = mesh.face_elements[faces].astype(np.int64)
    owner = np.where(solid[elements[:, 0]], elements[:, 0], elements[:, 1])   # the face's one solid element
    corners = mesh.points[local[:, :3]]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = corners.mean(axis=1) - mesh.centroids(owner)
    flip = np.einsum('ij,ij->i', normal, outward) < 0.0   # (a, c, b) is counter-clockwise from outside
    first = local[:, 0]
    second, third = np.where(flip, local[:, 2], local[:, 1]), np.where(flip, local[:, 1], local[:, 2])
    if dofs.order == 1:
        triangles = np.stack([first, second, third], axis=1)
    else:
        ab, ac, bc = local[:, 3], local[:, 4], local[:, 5]
        mid_12, mid_23, mid_31 = np.where(flip, ac, ab), bc, np.where(flip, ab, ac)
        triangles = np.concatenate([np.stack(corner, axis=1) for corner in (
            (first, mid_12, mid_31), (mid_12, second, mid_23), (mid_31, mid_23, third), (mid_12, mid_23, mid_31))])
    used, inverse = np.unique(triangles.ravel(), return_inverse=True)
    vertex = used < mesh.n_points
    nodes = np.empty((used.size, 3))
    nodes[vertex] = mesh.points[used[vertex]]
    edge_nodes = mesh.edges[used[~vertex] - mesh.n_points]
    nodes[~vertex] = 0.5 * (mesh.points[edge_nodes[:, 0]] + mesh.points[edge_nodes[:, 1]])
    return SolidSurface(dofs=used.astype(np.int64), nodes=nodes,
                        triangles=np.asarray(inverse).reshape(-1, 3).astype(np.int32))


class ModeShapes:
    """Runtime object of a mode_shapes feature: the SolidSurface of the last (dof map, solid set), rebuilt when either
    changes."""

    def __init__(self):
        self._key = None
        self._surface = None

    def surface(self, dofs, solid) -> SolidSurface:
        solid = np.asarray(solid, dtype=bool)
        if self._key is None or self._key[0] is not dofs or not np.array_equal(self._key[1], solid):
            self._surface = solid_surface(dofs, solid)
            self._key = (dofs, solid.copy())
        return self._surface


def mode_shapes_result(solver, shapes) -> dict:
    """'mode_shapes' result (schema.RESULT_KIND_DOCS) of the last eigenfrequency solve of a SolidMechanicsFEMPhysics3D
    on the surface of its solid window elements; ValueError before such a solve."""
    solve = solver.eigen_solve
    if solve is None:
        raise ValueError('not solved yet: mode shapes need an eigenfrequency (or optomechanics) study')
    system = solver.system
    mesh = solver.domain.mesh
    surface = shapes.surface(system.full, solver.solid_elements & np.asarray(mesh.window_elements(), dtype=bool))
    resonances = list(solve.resonances)
    n_vector = _COMPONENTS * system.n_dofs
    real = np.zeros((len(resonances), surface.dofs.size, _COMPONENTS), dtype=np.float32)
    imaginary = np.zeros_like(real)
    for index, resonance in enumerate(resonances):
        block = np.asarray(resonance.displacement)[:n_vector].reshape(-1, _COMPONENTS)[surface.dofs]
        real[index] = block.real
        if np.iscomplexobj(block):
            imaginary[index] = block.imag
    bloch = system.bloch
    periodic = (False,) * 3 if bloch is None else bloch.periodic
    wavenumbers = (0.0,) * 3 if bloch is None else bloch.wavenumbers
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {'kind': MODE_SHAPES_KIND, 'nodes': surface.nodes.copy(), 'triangles': surface.triangles.copy(),
            'resonance_number': np.arange(1, len(resonances) + 1, dtype=np.int32),
            'frequency': np.array([resonance.frequency for resonance in resonances], dtype=np.float64),
            'q_factor': np.array([resonance.q_factor if resonance.frequency_imag else np.nan
                                  for resonance in resonances], dtype=np.float64),
            'displacement_re': real, 'displacement_im': imaginary,
            'bloch_k': np.array(wavenumbers, dtype=np.float64), 'periodic': np.array(periodic, dtype=np.int8),
            'period': bounds[3:] - bounds[:3], 'phase_convention': PHASE_CONVENTION, 'unit': 'm'}

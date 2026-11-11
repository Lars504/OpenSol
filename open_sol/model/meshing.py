"""Mesh planning from a ModelSpec, per mesh (S7.2: ModelSpec.meshes, PhysicsSpec.mesh): merged domain requirements of
the physics on a mesh, CPU mesh estimates (no GPU work) and the Domain. Uniform grids use Domain.uniform; graded grids
MeshConfig + Domain.from_mesh; no boundary_config -> no PML. Triangle meshes (S7.5): sizes (triangle_sizing) and a
size-field element-count estimate here, the mesher in model.triangle_meshing / fem."""
import cmath
import copy
import hashlib
import heapq
import math
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np

from ..core.constants import C0, FACES, FACES_3D, Z_FACES
from ..core.domain import Domain, containing_cell
from ..core.domain3d import Domain3D
from ..core.grid import DEFAULT_COURANT_FACTOR, courant_time_step, edges_from_widths, uniform_cell_count
from ..core.grid3d import courant_time_step_3d
from ..core.memory_estimate import BOOL_BYTES, FLOAT32_BYTES, MemoryEstimate, combine, grid_transfer_estimate
from ..core.mesh import SEGMENT_PROFILES, MeshConfig, complete_axis_edges, has_uniform_spacing, segment_widths
from ..materials import effective_material_data
from ..physics.em.dispersion import complex_permittivity
from .expressions import TIME_NAMES, ExpressionError, evaluate_scalar, referenced_names
from .geometry import (GEOMETRY_KINDS, axis_parallel_faces, bounding_box, contains_points_3d, domain_map, domain_size,
                       domain_size_z, expand_entities, is_plain, line_orientation, material_selections,
                       paint_material_index, point_z, rasterize_window, resolve_geometry, z_extent)
from .schema import (MESH_KIND_DEFAULTS, MESH_PARAMS, DomainRequirements, array_owner, get_registry,
                     permittivity_arrays, resolve_values, resolved_solvers, solver_backend, state_field_names)
from .spec import REFINEMENT_PLACES
from .boundary_locator import conformal_memory_estimate
from .subpixel_geometry import subpixel_memory_estimate

MAX_MESH_CELLS = 400_000_000
MAX_AXIS_CELLS = 50_000
MESH_ESTIMATE_ERRORS = (ValueError, KeyError, ArithmeticError)   # estimate_mesh failures that are model errors
_REFINEMENT_RTOL = 1e-9         # a refinement applies only when finer than base_dx * (1 - rtol)
_MODE_LABELS = dict(next(param.choice_labels for param in MESH_PARAMS if param.name == 'mode'))   # for notes
# auto mode without an EM physics on the mesh (S7.8 physics-controlled rule): base = longest side / 50, objects of the
# mesh's physics refined to their smallest side / 4; lattice Boltzmann types uniform at the smallest side / 16
PHYSICS_BASE_DIVISOR = 50.0
PHYSICS_OBJECT_DIVISOR = 4.0
UNIFORM_AUTO_DIVISORS = {'laminar_flow': 16.0}
# triangle element-count estimate (S7.5 2.10): N ~ fill x integral 4 / (sqrt(3) h^2) dA on a raster of the size field.
# The mesher refines until every edge <= h(midpoint), so its elements average an equilateral size ~h / 1.5, not h:
# fill 2.3 (built / estimate 2.0 at the former 1.15 on uniform and size-point graded L-shape / rib meshes, h / 8 .. h)
TRIANGLE_FILL = 2.3
# thin / point sources (_thin_source_density): the mesher builds the steep band beside a refined film finer still; their
# density above the h_max floor x this (built / estimate of selftest mesh_triangle_estimate thin_film: 49676 vs 3400 +
# 38666 band -> 1.2; the hirsch E5 film was counted ~0.87-0.89 x built without it)
TRIANGLE_BAND_FILL = 1.2
# PML ring: window-edge segments per integral ds / h of the edge (sampled near h, encroached ones split; rib ring)
TRIANGLE_EDGE_FILL = 1.72
_TRIANGLE_RASTER = 256          # raster pixels along the longer side
_TRIANGLE_MIN_RASTER = 8        # ... at least this many along the shorter one
_TRIANGLE_TRANSFER_ENTRIES = 6  # grid-transfer entries per target cell / element when a triangle mesh is involved
# tetrahedron element-count estimate (igor3d_plan T4b): N ~ fill x integral 6 sqrt(2) / h^3 dV (a regular tetrahedron
# of edge h has volume h^3 / (6 sqrt 2)) on a voxel raster of the size field over the box, PML shell included. Fill
# uncalibrated (gmsh's Delaunay elements are less regular than the edge-h one): recalibrate on built meshes (T5)
TETRAHEDRON_FILL = 1.5
_TETRAHEDRON_RASTER = 48        # voxels along the longest box side at least (the raster of an unrefined mesh)
# gmsh raster voxel: the finest object / edge size, but no finer than this many voxels over the box (one distance
# transform per distinct size: the cost is voxels x sizes) and no coarser than _TETRAHEDRON_RASTER
_TETRAHEDRON_MAX_VOXELS = 500_000
_TETRAHEDRON_SUBVOXEL_POINTS = 8   # midpoint samples of the signed distance across a voxel (its mean 1 / h^3)
TETRAHEDRA_PER_NODE = 5.5       # large tetrahedron meshes: nodes ~ T / 5.5, edges ~ nodes + T, faces ~ 2 T (Euler)
SIZE_POINT_STEPS = 8            # MeshSpec.size_points sizes are rounded down to 2^(k / SIZE_POINT_STEPS)
_SUBPIXEL_VARIATION = 1.25      # estimate: raster pixels where size points vary h by more than this are sub-sampled,
_SUBPIXEL_SPACING = 0.5         # ... at this x the finest size there,
_MAX_SUBSAMPLES = 32            # ... at most this many samples per axis
_MAX_TOTAL_SUBSAMPLES = 8_000_000
_SUBSAMPLE_CHUNK = 1_000_000


@dataclass
class MeshEstimate:
    """CPU description of the tensor grid make_domain builds for one mesh: full-grid widths (PML included),
    physical-frame edges."""
    nx: int
    ny: int
    n_cells: int
    dt: float | None            # EM Courant step of the grid; None when no time-stepping physics uses one
    dx_min: float               # smallest / largest cell width over both axes (full grid)
    dx_max: float
    npml_sides: dict
    dx_cells: np.ndarray        # float64 (nx,)
    dy_cells: np.ndarray        # float64 (ny,)
    x_edges: np.ndarray         # float64 (nx+1,), origin at the physical corner (PML edges < 0)
    y_edges: np.ndarray         # float64 (ny+1,)
    uniform: bool               # window widths constant per axis (core.mesh.has_uniform_spacing; 3D: z too)
    note: str = ''              # why the grid differs from the requested mode ('' when it does not)
    wavelength_min: float | None = None   # auto mode: the vacuum wavelength [m] it resolves; None in other modes
    wavelength_source: str = ''           # auto mode: 'mesh' (the wavelength_min setting) or the physics name
    # 3D models (spec v2 size_z): z cells, graded like x / y on graded meshes (n_cells, dx_min / dx_max, dt and
    # npml_sides z_lo / z_hi include them)
    nz: int = 1
    dz_cells: np.ndarray | None = None    # float64 (nz,), None in 2D
    z_edges: np.ndarray | None = None     # float64 (nz+1,), origin at the physical bottom
    custom_axes: tuple = ()               # axes whose cells come from MeshAxisSpec edges / segments
    # 2D line_block layouts of the window cells: block axis ('x': a block = one y row) -> (n_blocks, block_size)
    line_blocks: dict = field(default_factory=dict)
    # polar / spherical domains: the geometry.DomainMap (x / y edges and widths are then logical: r - radius_min [m],
    # angle - angle_min [rad]; dx_min / dx_max physical lengths, dt None); None on a Cartesian grid
    domain_map: object = None
    name: str = 'mesh'                    # the MeshSpec it describes
    kind: str = 'tensor'
    auto_rule: str = ''                   # auto mode: 'wavelength', 'physics' (physics-controlled) or 'uniform_flow'
    auto_dx: float | None = None          # auto mode: the base cell size its rule chose [m]
    auto_refinements: tuple = ()          # auto mode: (object, size [m]) its rule refines, finer than auto_dx

    @property
    def window_axis_cells(self) -> tuple:
        """Physical-window cells per axis (nx, ny[, nz]): the counts without the npml_sides padding (z in 3D only)."""
        sides = self.npml_sides
        counts = [self.nx - sides['x_lo'] - sides['x_hi'], self.ny - sides['y_lo'] - sides['y_hi']]
        if self.dz_cells is not None:
            counts.append(self.nz - sides.get('z_lo', 0) - sides.get('z_hi', 0))
        return tuple(int(count) for count in counts)


@dataclass
class TriangleSizing:
    """Resolved size parameters of a 'triangles' mesh (COMSOL Free Triangular): the one size rule of the element-count
    estimate and the mesher (model.triangle_meshing, S7.5 2.2)."""
    h_max: float                  # maximum element size [m]
    h_min: float                  # minimum element size [m]
    growth: float                 # maximum element growth rate (> 1)
    curvature_factor: float       # element size along circle / ellipse edges <= this x the radius of curvature
    object_sizes: dict            # geometry object -> maximum element size in / along / at it [m] (< h_max only)
    pml_widths: dict              # FACES face -> absorbing ring thickness [m]: cpml_npml x h_max on cpml faces, else 0
    pml_layers: int               # structured element layers across each ring side
    mesher: str                   # 'builtin' | 'gmsh'
    wavelength_min: float | None = None   # auto mode: as MeshEstimate
    wavelength_source: str = ''
    auto_rule: str = ''
    auto_dx: float | None = None
    auto_refinements: tuple = ()
    # MeshSpec.size_points as float64 (n, 3) [x, y, h] (tetrahedra (n, 4) [x, y, z, h]), h rounded down to
    # SIZE_POINT_STEPS per octave (few size levels); repr shows their sha1 instead (numpy elides long arrays: the GUI
    # mesh keys are reprs)
    size_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)
    size_points_digest: str = ''
    # axes a physics on the mesh makes periodic: triangles matching window sides (no PML ring there); tetrahedra gmsh
    # periodic surfaces
    periodic_axes: tuple = ()
    # tetrahedra: geometry object -> element size along its edges [m] (< h_max only), from the physics' mesh_edge_sizes
    # and the RefinementSpec.where 'edges' refinements (gmsh)
    edge_sizes: dict = field(default_factory=dict)
    # triangles: 'x' / 'y' mirror lines at the window centre (MeshSpec.mirror_x / mirror_y; the half window is meshed)
    mirror_axes: tuple = ()
    # tetrahedra: fem.gmsh3d.SizedFace per refined box face (MeshSpec.face_sizes, sizes < h_max only)
    face_sizes: tuple = ()
    # builtin tetrahedra: float64 node coordinates [m] per axis (x, y, z) of builtin_axis_edges, shell included (the one
    # rule of mesher and estimate); repr shows their sha1 (the GUI mesh keys are reprs); () elsewhere
    axis_edges: tuple = field(default=(), repr=False)
    axis_edges_digest: str = ''
    # gmsh tetrahedra: geometry object -> element size at its surface [m] (RefinementSpec.where 'surface', < h_max
    # only): size + (growth - 1) x |distance to the surface| (tetrahedral_meshing.sized_surfaces)
    surface_sizes: dict = field(default_factory=dict)
    # tetrahedra with MeshSpec.sweep_axis 'z' (plan D5, model.swept_meshing): 'z', and the float64 z node coordinates
    # [m] of the prism layers (graded_window_edges z, PML shell included); repr shows their sha1; '' / () elsewhere
    sweep_axis: str = ''
    layer_edges: tuple | np.ndarray = field(default=(), repr=False)
    layer_edges_digest: str = ''


@dataclass
class TriangleMeshEstimate:
    """CPU estimate of a 'triangles' mesh (S7.5 2.10): element / node / edge counts from the size field on a coarse
    raster; the MeshEstimate names consumers share (n_cells = elements, dx_min / dx_max = smallest / largest size);
    material_elements = estimated window elements each spec material paints (the builder's paint order)."""
    name: str
    sizing: TriangleSizing
    n_elements: int
    n_nodes: int                  # P1 nodes ~ elements / 2
    n_edges: int                  # ~ 3 elements / 2
    dx_min: float
    dx_max: float
    extent: tuple                 # (x0, x1, y0, y1) [m] of the meshed region, absorbing ring included
    note: str = ''
    kind: str = 'triangles'
    dt: float | None = None
    nz: int = 1
    dz_cells: None = None
    z_edges: None = None
    domain_map: None = None
    uniform: bool = False
    custom_axes: tuple = ()
    line_blocks: dict = field(default_factory=dict)
    npml_sides: dict = field(default_factory=lambda: {face: 0 for face in FACES})
    material_elements: dict = field(default_factory=dict)
    counts_built: bool = False    # True: the counts are those of the built mesh (with_built_counts), not size-based
    # fem.dissection.MeshDissection of the built mesh (built_mesh_estimates; None on size-based counts): the factor
    # entries a sparse direct solver needs on this mesh, counted lazily per element order by the estimators that ask
    dissection: object = field(default=None, compare=False, repr=False)

    @property
    def n_cells(self):
        return self.n_elements

    @property
    def wavelength_min(self):
        return self.sizing.wavelength_min

    @property
    def wavelength_source(self):
        return self.sizing.wavelength_source

    @property
    def auto_rule(self):
        return self.sizing.auto_rule

    @property
    def auto_dx(self):
        return self.sizing.auto_dx

    @property
    def auto_refinements(self):
        return self.sizing.auto_refinements


class TetrahedronSizing(TriangleSizing):
    """Resolved sizes of a 'tetrahedra' mesh (COMSOL Free Tetrahedral; igor3d_plan T4b): the TriangleSizing fields,
    pml_widths over the six 3D faces (the shell thickness), size_points [x, y, z, h], pml_layers and mirror_axes
    unused."""


@dataclass
class TetrahedronMeshEstimate(TriangleMeshEstimate):
    """CPU estimate of a 'tetrahedra' mesh (igor3d_plan T4b): element / node / edge / face counts (exact for the
    builtin Kuhn split, from the size field on a voxel raster with gmsh); extent (x0, x1, y0, y1, z0, z1) [m] of the
    meshed box, PML shell included; material_elements = estimated elements each spec material paints (the builder's
    paint order on centres clamped into the window), e.g. a solid's share of the dofs."""
    n_faces: int = 0
    kind: str = 'tetrahedra'
    npml_sides: dict = field(default_factory=lambda: {face: 0 for face in FACES_3D})


@dataclass
class _Refinement:
    object_name: str
    dx: float                   # isotropic size (auto mode, and the default of every axis)
    margin: float | None        # None -> 3 * the axis dx
    axis_dx: dict = field(default_factory=dict)   # per-axis sizes (RefinementSpec dx_x / dx_y / dx_z)
    where: str = 'inside'       # RefinementSpec.where: 'edges' / 'surface' size gmsh tetrahedra only

    def along(self, axis):
        return self.axis_dx.get(axis, self.dx)

    def refines(self, axis, base_dx):
        return self.along(axis) < base_dx * (1.0 - _REFINEMENT_RTOL)

    def margin_along(self, axis):
        return 3.0 * self.along(axis) if self.margin is None else self.margin


@dataclass
class _AxisPlan:
    """Resolved MeshAxisSpec of one axis: max_ratio override, coordinate bands [(lo, hi, dx)], custom edges."""
    max_ratio: float | None = None
    bands: list = field(default_factory=list)
    edges: np.ndarray | None = None


@dataclass
class _MeshPlan:
    size_x: float
    size_y: float
    npml_sides: dict
    boundary_config: object
    courant_factor: float       # fraction of the 2D CFL limit
    courant_dt: bool            # a time-stepping physics runs on the grid's Courant step
    uniform_dx: float | None    # set -> uniform grid
    mesh_config: MeshConfig | None
    note: str = ''
    wavelength_min: float | None = None
    wavelength_source: str = ''
    size_z: float | None = None   # 3D models; graded z lives in mesh_config (its size_z)
    domain_map: object = None     # polar / spherical: geometry.DomainMap, with mapped_widths (dr [m], d_angle [rad])
    mapped_widths: tuple | None = None
    custom_axes: tuple = ()       # polar / spherical: axes given by MeshAxisSpec edges / segments
    name: str = 'mesh'
    auto_rule: str = ''
    auto_dx: float | None = None
    auto_refinements: tuple = ()


# ---- public API ----
def physics_settings(physics_type, physics_spec, parameters):
    """resolve_values of a physics' settings; ValueError naming the physics on bad input."""
    if not isinstance(physics_spec.settings, dict):
        raise ValueError(f"physics '{physics_spec.name}': settings must be a dict, "
                         f"got {type(physics_spec.settings).__name__}")
    try:
        return resolve_values(physics_type.settings, physics_spec.settings, parameters)
    except ValueError as exc:
        raise ValueError(f"physics '{physics_spec.name}' settings: {exc}") from exc


def default_mesh_name(spec) -> str:
    """The mesh one-mesh callers mean (estimate_mesh / make_domain without a name): the first used mesh."""
    used = spec.used_meshes()
    if not used:
        raise ValueError('the model has no mesh: add one (ModelSpec.meshes)')
    return used[0].name


def estimate_mesh(spec, parameters, registry=None, mesh=None, shapes=None):
    """MeshEstimate (tensor), TriangleMeshEstimate or TetrahedronMeshEstimate of one mesh (its name; None =
    default_mesh_name) without any GPU allocation (partial geometry; shapes: the caller's resolve_geometry of these
    parameters, None = resolved here); ValueError (incl. ExpressionError) / KeyError on model errors."""
    registry = registry if registry is not None else get_registry()
    mesh_spec = spec.mesh_by_name(default_mesh_name(spec) if mesh is None else mesh)
    shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
    kind = mesh_values(mesh_spec, parameters)['kind']
    if kind == 'triangles':
        return _triangle_estimate(spec, mesh_spec, parameters, registry, shapes)
    if kind == 'tetrahedra':
        return _tetrahedron_estimate(spec, mesh_spec, parameters, registry, shapes)
    plan = _plan(spec, parameters, registry, shapes, mesh_spec)
    if plan.mapped_widths is not None:
        return _mapped_estimate(plan)
    sides = plan.npml_sides
    if plan.mesh_config is None:
        dx = plan.uniform_dx
        nx, ny = _uniform_cell_counts(plan)
        dx_cells = np.full(nx, dx, dtype=np.float64)
        dy_cells = np.full(ny, dx, dtype=np.float64)
        dt = float(courant_time_step(dx, dx, plan.courant_factor)) if plan.courant_dt else None
    else:
        dx_arr, dy_arr = plan.mesh_config.cell_spacings(sides, pml_dx=plan.mesh_config.dx_base,
                                                        ghost_faces=_ghost_faces(plan))
        nx, ny = len(dx_arr), len(dy_arr)
        dx_cells = dx_arr.astype(np.float64)
        dy_cells = dy_arr.astype(np.float64)
        dt = (float(courant_time_step(float(np.min(dx_arr)), float(np.min(dy_arr)), plan.courant_factor))
              if plan.courant_dt else None)
    uniform_grid = plan.mesh_config is None
    x_edges = edges_from_widths(dx_cells, sides['x_lo'], uniform_grid)
    y_edges = edges_from_widths(dy_cells, sides['y_lo'], uniform_grid)
    window_dx = dx_cells[sides['x_lo']:nx - sides['x_hi']]
    window_dy = dy_cells[sides['y_lo']:ny - sides['y_hi']]
    estimate = MeshEstimate(
        nx=nx, ny=ny, n_cells=nx * ny, dt=dt,
        dx_min=float(min(dx_cells.min(), dy_cells.min())), dx_max=float(max(dx_cells.max(), dy_cells.max())),
        npml_sides=dict(sides), dx_cells=dx_cells, dy_cells=dy_cells, x_edges=x_edges, y_edges=y_edges,
        uniform=bool(has_uniform_spacing(window_dx) and has_uniform_spacing(window_dy)), note=plan.note,
        wavelength_min=plan.wavelength_min, wavelength_source=plan.wavelength_source,
        custom_axes=plan.mesh_config.custom_axes() if plan.mesh_config is not None else (), name=plan.name,
        auto_rule=plan.auto_rule, auto_dx=plan.auto_dx, auto_refinements=plan.auto_refinements)
    if plan.size_z is not None:
        _add_z_axis(estimate, plan)
    else:   # a block of one y row holds the window's x cells (solvers.line_block block_axis 0), and vice versa
        estimate.line_blocks = {'x': (len(window_dy), len(window_dx)), 'y': (len(window_dx), len(window_dy))}
    return estimate


def estimate_meshes(spec, parameters, registry=None, shapes=None) -> dict:
    """{name: estimate_mesh} of every used mesh (spec.used_meshes order; geometry resolved once, or the caller's
    shapes); the first failure raises."""
    registry = registry if registry is not None else get_registry()
    shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
    return {mesh_spec.name: estimate_mesh(spec, parameters, registry, mesh_spec.name, shapes)
            for mesh_spec in spec.used_meshes()}


def _mapped_estimate(plan):
    """MeshEstimate of a polar / spherical plan: logical edges, no PML, physical cell lengths in dx_min / dx_max (a
    cylindrical sector: its z cells too, no line blocks)."""
    dx_cells, dy_cells = plan.mapped_widths[:2]
    nx, ny = len(dx_cells), len(dy_cells)
    x_edges, y_edges = edges_from_widths(dx_cells, 0, False), edges_from_widths(dy_cells, 0, False)
    radius_centres = plan.domain_map.radius_min + 0.5 * (x_edges[:-1] + x_edges[1:])
    arc_lengths = (radius_centres.min() * dy_cells.min(), radius_centres.max() * dy_cells.max())   # r_c d_angle
    if plan.domain_map.line:   # one cell across: only the cells along x are lengths of the mesh
        arc_lengths = (float(dx_cells.min()), float(dx_cells.max()))
    estimate = MeshEstimate(
        nx=nx, ny=ny, n_cells=nx * ny, dt=None, dx_min=float(min(dx_cells.min(), arc_lengths[0])),
        dx_max=float(max(dx_cells.max(), arc_lengths[1])), npml_sides={face: 0 for face in FACES},
        dx_cells=dx_cells, dy_cells=dy_cells, x_edges=x_edges, y_edges=y_edges,
        uniform=bool(has_uniform_spacing(dx_cells) and has_uniform_spacing(dy_cells)), note=plan.note,
        custom_axes=plan.custom_axes, line_blocks={'x': (ny, nx), 'y': (nx, ny)}, domain_map=plan.domain_map,
        name=plan.name)
    if len(plan.mapped_widths) == 3:   # a cylindrical (r, phi, z) sector
        dz_cells = plan.mapped_widths[2]
        estimate.nz = len(dz_cells)
        estimate.dz_cells = dz_cells
        estimate.z_edges = edges_from_widths(dz_cells, 0, False)
        estimate.n_cells = nx * ny * estimate.nz
        estimate.dx_min = min(estimate.dx_min, float(dz_cells.min()))
        estimate.dx_max = max(estimate.dx_max, float(dz_cells.max()))
        estimate.npml_sides.update(z_lo=0, z_hi=0)
        estimate.uniform = bool(estimate.uniform and has_uniform_spacing(dz_cells))
        estimate.line_blocks = {}
    return estimate


def largest_adjacent_ratio(estimate, axis) -> float:
    """Largest ratio of neighbouring window cell widths along axis ('x', 'y', 'z'; 1 for fewer than two cells)."""
    widths = {'x': estimate.dx_cells, 'y': estimate.dy_cells, 'z': estimate.dz_cells}[axis]
    lo, hi = int(estimate.npml_sides.get(f'{axis}_lo', 0)), int(estimate.npml_sides.get(f'{axis}_hi', 0))
    window = widths[lo:len(widths) - hi]
    if window.size < 2:
        return 1.0
    ratios = window[1:] / window[:-1]
    return float(np.max(np.maximum(ratios, 1.0 / ratios)))


def line_block_hint(estimate) -> str:
    """'' in 3D, else the line_block cost of both block axes (n_blocks b^3) and the cheaper one (solver_plan S7.0)."""
    if not estimate.line_blocks:
        return ''
    costs = {axis: n_blocks * float(block_size) ** 3 for axis, (n_blocks, block_size) in estimate.line_blocks.items()}
    parts = [f'blocks along {axis}: {n_blocks} blocks of b = {block_size} cells (cost ~ n b^3 = {costs[axis]:.3g})'
             for axis, (n_blocks, block_size) in estimate.line_blocks.items()]
    cheaper = min(costs, key=costs.get)
    return (f"line_block: {'; '.join(parts)}; blocks along {cheaper} are cheaper (refine the line axis, not the "
            'block axis)')


def _ghost_faces(plan):
    """Mirror faces of the plan's boundary config (EM ghost_faces; () without one), padded at the edge cell width."""
    return getattr(plan.boundary_config, 'ghost_faces', ())


def _z_cells(plan):
    """(dz_arr (nz,) incl. PML, n_lo, n_hi) of a 3D model: float64 cells at the uniform dx (as Domain3D.uniform), else
    float32 cells graded by the plan's MeshConfig z bands (as its xy widths); PML cells z_lo / z_hi from the boundary
    config's npml_sides (0 when it has none), a mirror ghost at the edge cell width."""
    n_lo, n_hi = (int(plan.npml_sides.get(face, 0)) for face in Z_FACES)
    if plan.mesh_config is None:
        dz = plan.uniform_dx
        return np.full(uniform_cell_count(plan.size_z + (n_lo + n_hi) * dz, dz), dz, dtype=np.float64), n_lo, n_hi
    return (plan.mesh_config.z_cell_spacings(n_lo, n_hi, pml_dx=plan.mesh_config.dx_base,
                                             ghost_faces=_ghost_faces(plan)), n_lo, n_hi)


def _add_z_axis(estimate, plan):
    """Fill the 3D fields of a MeshEstimate (z cells, cell count, uniform, dt from the 3D CFL limit); cell-count
    guard."""
    dz_arr, n_lo, n_hi = _z_cells(plan)
    nz = len(dz_arr)
    _check_cell_count(estimate.nx, estimate.ny, nz)
    estimate.nz = nz
    estimate.dz_cells = dz_arr.astype(np.float64)
    estimate.z_edges = edges_from_widths(estimate.dz_cells, n_lo, plan.mesh_config is None)
    estimate.n_cells = estimate.nx * estimate.ny * nz
    estimate.dx_min = min(estimate.dx_min, float(estimate.dz_cells.min()))
    estimate.dx_max = max(estimate.dx_max, float(estimate.dz_cells.max()))
    estimate.npml_sides.update(z_lo=n_lo, z_hi=n_hi)
    estimate.uniform = bool(estimate.uniform and has_uniform_spacing(estimate.dz_cells[n_lo:nz - n_hi]))
    if estimate.dt is not None:
        estimate.dt = float(courant_time_step_3d(float(estimate.dx_cells.min()), float(estimate.dy_cells.min()),
                                                 float(estimate.dz_cells.min()), plan.courant_factor))


def boundary_config_difference_text(first, other) -> str:
    """'' when two boundary configurations are equal (they merge), else '<setting>: <first> vs <other>; ...' with the
    MODEL_SETTING_NAMES of the config class; duck-typed on the config's differences(other) (EM BoundaryConfig)."""
    differences = tuple(first.differences(other))
    if not all(hasattr(first, name) and hasattr(other, name) for name in differences):   # e.g. ('type',)
        return ', '.join(differences)
    setting_names = getattr(type(first), 'MODEL_SETTING_NAMES', {})

    def shown(value):   # enum members (BCType) by their value
        return getattr(value, 'value', value)

    return '; '.join(f'{setting_names.get(name, name)}: {shown(getattr(first, name))} vs '
                     f'{shown(getattr(other, name))}' for name in differences)


_kept_element_meshes = None   # mesh name -> (element_mesh_key, mesh) while remember_element_meshes is on, else None


def remember_element_meshes(enabled=True):
    """Keep the last triangle / tetrahedron mesh made per mesh name and reuse it while its element_mesh_key matches
    (the GUI worker: a Build mesh preview or a build meshes once for the next build); off forgets them."""
    global _kept_element_meshes
    _kept_element_meshes = {} if enabled else None


def remembers_element_meshes() -> bool:
    """True while remember_element_meshes is on."""
    return _kept_element_meshes is not None


def kept_element_mesh_keys() -> dict:
    """{mesh name: element_mesh_key} of the meshes kept now ({} while remember_element_meshes is off)."""
    return {} if _kept_element_meshes is None else {name: kept[0] for name, kept in _kept_element_meshes.items()}


def kept_element_mesh_counts() -> dict:
    """{mesh name: (element_mesh_key, built_mesh_counts)} of the meshes kept now ({} while remember_element_meshes
    is off): the GUI reads them after a build that meshed and then failed."""
    if _kept_element_meshes is None:
        return {}
    return {name: (key, built_mesh_counts(mesh)) for name, (key, mesh) in _kept_element_meshes.items()}


def element_meshes_kept():
    """Context manager: remember_element_meshes inside the block, the previous state restored after (run_study: sweep
    values / parity sectors with the same mesh settings and geometry mesh once)."""
    return _KeptElementMeshes()


class _KeptElementMeshes:
    def __enter__(self):
        self._was_on = remembers_element_meshes()
        if not self._was_on:
            remember_element_meshes()
        return self

    def __exit__(self, *exc_info):
        if not self._was_on:   # forgets the kept meshes (host memory)
            remember_element_meshes(False)
        return False


def element_mesh_key(sizing, size, shapes) -> str:
    """Text naming what an element mesh is made from (resolved sizes, window size, resolved geometry); equal text =
    the same mesh (the built-in meshers are deterministic; gmsh is taken as such)."""
    geometry = [(name, shape.kind, sorted(shape.values.items(), key=lambda item: item[0]))
                for name, shape in shapes.items()]
    return repr((sizing, tuple(float(length) for length in size), geometry))


def element_window_size(spec, parameters, kind) -> tuple:
    """(size_x, size_y(, size_z)) [m] of the window an element mesh of `kind` covers."""
    size = tuple(domain_size(spec, parameters))
    return size + (domain_size_z(spec, parameters),) if kind == 'tetrahedra' else size


def element_mesh(spec, parameters, registry=None, mesh=None, *, shapes=None, log=None):
    """(fem TriangleMesh | TetrahedronMesh, reused) of a 'triangles' / 'tetrahedra' mesh (its name; None =
    default_mesh_name); reused True = the kept mesh of the same element_mesh_key (remember_element_meshes); log(level,
    text) gets the mesher stage times."""
    registry = registry if registry is not None else get_registry()
    mesh_spec = spec.mesh_by_name(default_mesh_name(spec) if mesh is None else mesh)
    shapes = resolve_geometry(spec, parameters) if shapes is None else shapes
    kind = mesh_values(mesh_spec, parameters)['kind']
    if kind == 'tetrahedra':
        from .tetrahedral_meshing import tetrahedralize
        mesher = partial(tetrahedralize, log=log)
        sizing = tetrahedron_sizing(spec, mesh_spec, parameters, registry, shapes)
    else:
        from .triangle_meshing import triangulate
        mesher = partial(triangulate, log=log)   # logs its stage times (time_estimate mesh items)
        sizing = triangle_sizing(spec, mesh_spec, parameters, registry, shapes)
    if _kept_element_meshes is None:
        return mesher(spec, parameters, registry, mesh_spec.name, shapes=shapes, sizing=sizing), False
    key = element_mesh_key(sizing, element_window_size(spec, parameters, kind), shapes)
    kept = _kept_element_meshes.pop(mesh_spec.name, None)   # a stale mesh is dropped before meshing (host memory)
    if kept is not None and kept[0] == key:
        _kept_element_meshes[mesh_spec.name] = kept
        return kept[1], True
    built = mesher(spec, parameters, registry, mesh_spec.name, shapes=shapes, sizing=sizing)
    _kept_element_meshes[mesh_spec.name] = (key, built)
    return built, False


def built_mesh_counts(mesh) -> dict:
    """{'n_elements', 'n_nodes', 'n_edges'(, 'n_faces': tetrahedra)} of a built fem TriangleMesh / TetrahedronMesh
    (plain ints) + 'dissection': its fem.dissection.MeshDissection (factor entries counted lazily on first ask)."""
    from ..fem.dissection import MeshDissection
    counts = {'n_elements': int(mesh.n_elements), 'n_nodes': int(mesh.n_points), 'n_edges': int(mesh.n_edges)}
    n_faces = getattr(mesh, 'n_faces', None)
    if n_faces is not None:
        counts['n_faces'] = int(n_faces)
    counts['dissection'] = MeshDissection(mesh)
    return counts


def with_built_counts(estimate, counts):
    """The element mesh estimate with its built mesh's counts (built_mesh_counts) in place of the size-based ones:
    counts_built True, material_elements scaled by the element ratio, the dissection attached."""
    element_ratio = counts['n_elements'] / max(estimate.n_elements, 1)
    material_elements = {name: count * element_ratio for name, count in estimate.material_elements.items()}
    built = {name: int(count) for name, count in counts.items() if name != 'dissection' and hasattr(estimate, name)}
    return replace(estimate, **built, material_elements=material_elements, counts_built=True,
                   dissection=counts.get('dissection'))


def has_estimated_counts(estimate) -> bool:
    """True for an element mesh estimate whose counts are still size-based; tensor estimates and the builtin
    tetrahedra (grid boxes split into 6) are exact."""
    kind = getattr(estimate, 'kind', 'tensor')
    if kind not in ('triangles', 'tetrahedra') or estimate.counts_built:
        return False
    return not (kind == 'tetrahedra' and estimate.sizing.mesher == 'builtin' and not estimate.sizing.sweep_axis)


def awaits_built_mesh(estimate) -> bool:
    """True for an element mesh estimate not yet counted on its built mesh: size-based counts, or exact builtin
    tetrahedra counts still without the built mesh's factor count (MeshEstimate.dissection)."""
    return getattr(estimate, 'kind', 'tensor') in ('triangles', 'tetrahedra') and not estimate.counts_built


def built_mesh_estimates(spec, parameters, registry, mesh_estimates, shapes=None, log=None) -> dict:
    """mesh_estimates {mesh name: estimate} with every element mesh built (element_mesh: a kept one is reused) and
    its counts in place (with_built_counts: its MeshDissection attached): the numbers a memory judgement is final
    on."""
    built = {}
    for name, estimate in mesh_estimates.items():
        if awaits_built_mesh(estimate):
            mesh, _reused = element_mesh(spec, parameters, registry, name, shapes=shapes, log=log)
            estimate = with_built_counts(estimate, built_mesh_counts(mesh))
        built[name] = estimate
    return built


def make_domain(spec, parameters, registry=None, *, resolved_geometry=None, mesh=None, log=None):
    """Domain (GPU MaterialGrid) of one mesh (its name; None = default_mesh_name); resolved_geometry defaults to a
    strict resolve_geometry. A 'triangles' mesh gives a fem.domain.TriangleDomain (model.triangle_meshing), a
    'tetrahedra' mesh a fem.domain3d.TetrahedronDomain (model.tetrahedral_meshing), through element_mesh (a reuse
    is logged as info to log(level, text))."""
    registry = registry if registry is not None else get_registry()
    mesh_spec = spec.mesh_by_name(default_mesh_name(spec) if mesh is None else mesh)
    shapes = resolve_geometry(spec, parameters) if resolved_geometry is None else resolved_geometry
    kind = mesh_values(mesh_spec, parameters)['kind']
    if kind in ('triangles', 'tetrahedra'):
        element_grid, reused = element_mesh(spec, parameters, registry, mesh_spec.name, shapes=shapes, log=log)
        if reused and log is not None:
            log('info', f"mesh '{mesh_spec.name}': reused the {element_grid.n_elements:,} {kind} made earlier from "
                        'the same mesh settings and geometry (not meshed again)')
        if kind == 'triangles':
            from ..fem.domain import TriangleDomain
            return TriangleDomain(element_grid)
        from ..fem.domain3d import TetrahedronDomain
        return TetrahedronDomain(element_grid)
    plan = _plan(spec, parameters, registry, shapes, mesh_spec)
    if plan.mapped_widths is not None and plan.size_z is not None:   # a cylindrical (r, phi, z) sector, no padding
        return Domain3D.from_widths(*plan.mapped_widths, {}, plan.size_x, plan.size_y, plan.size_z,
                                    courant=plan.courant_factor, coordinates=plan.domain_map.coordinates())
    if plan.size_z is not None:
        return _make_domain_3d(plan)
    if plan.mapped_widths is not None:
        return Domain.from_widths(*plan.mapped_widths, coordinates=plan.domain_map.coordinates(),
                                  courant=plan.courant_factor)
    if plan.mesh_config is None:
        return Domain.uniform(plan.size_x, plan.size_y, plan.uniform_dx, npml=0,
                              courant=plan.courant_factor, boundary_config=plan.boundary_config, verbose=False)
    return Domain.from_mesh(plan.mesh_config, npml=0, boundary_config=plan.boundary_config,
                            courant=plan.courant_factor, verbose=False)


def _make_domain_3d(plan):
    """Domain3D of a 3D model: uniform cubic cells, or the plan's graded xy grid times its graded z cells (_z_cells)."""
    sides = dict(plan.npml_sides)
    dz_arr, n_lo, n_hi = _z_cells(plan)
    sides.update(z_lo=n_lo, z_hi=n_hi)
    if plan.mesh_config is None:
        _check_cell_count(*_uniform_cell_counts(plan), len(dz_arr))
        return Domain3D.uniform(plan.size_x, plan.size_y, plan.size_z, plan.uniform_dx, npml_sides=sides,
                                courant=plan.courant_factor, boundary_config=plan.boundary_config)
    dx_arr, dy_arr = plan.mesh_config.cell_spacings(sides, pml_dx=plan.mesh_config.dx_base,
                                                    ghost_faces=_ghost_faces(plan))
    _check_cell_count(len(dx_arr), len(dy_arr), len(dz_arr))
    return Domain3D.from_widths(dx_arr, dy_arr, dz_arr, sides, plan.size_x, plan.size_y, plan.size_z,
                                courant=plan.courant_factor, boundary_config=plan.boundary_config)


# ---- planning ----
def _plan(spec, parameters, registry, shapes, mesh_spec):
    """Grid decision of one tensor mesh, shared by estimate_mesh and make_domain (guards the cell count before any 1D
    meshing); only the physics on the mesh set its requirements and auto sizes."""
    size_x, size_y = domain_size(spec, parameters)
    on_mesh = spec.physics_on(mesh_spec.name)
    requirements, courant_dt = _requirements(on_mesh, parameters, registry)
    boundary_config = requirements.boundary_config
    sides = (dict(boundary_config.npml_sides) if boundary_config is not None
             else {face: 0 for face in FACES})
    courant_factor = (DEFAULT_COURANT_FACTOR if requirements.courant_factor is None
                      else float(requirements.courant_factor))
    plan = _MeshPlan(size_x, size_y, sides, boundary_config, courant_factor, courant_dt, None, None,
                     size_z=domain_size_z(spec, parameters), name=mesh_spec.name)

    mesh = mesh_values(mesh_spec, parameters)
    window_map = domain_map(spec, parameters)
    if window_map.mapped:
        return _mapped_plan(spec, parameters, plan, mesh, window_map, mesh_spec)
    mode = mesh['mode']
    if mode == 'uniform':
        plan.uniform_dx = mesh['dx']
        _check_cell_count(*_uniform_cell_counts(plan))
        return plan

    if mode == 'nonuniform':
        base_dx = mesh['dx']
        refinements = _explicit_refinements(spec, mesh_spec, parameters, shapes)
    else:
        auto = _auto_sizes(spec, on_mesh, parameters, registry, shapes, mesh)
        plan.wavelength_min, plan.wavelength_source, plan.auto_rule = (auto.wavelength_min, auto.wavelength_source,
                                                                       auto.rule)
        plan.auto_dx, plan.auto_refinements = auto.base_dx, auto.finer_refinements()
        if auto.rule == 'uniform_flow':
            plan.uniform_dx, plan.note = auto.base_dx, auto.note
            _check_cell_count(*_uniform_cell_counts(plan), reason=plan.note)
            return plan
        base_dx, refinements = auto.base_dx, auto.refinements + _explicit_refinements(spec, mesh_spec, parameters,
                                                                                      shapes)
    lengths = {'x': size_x, 'y': size_y} if plan.size_z is None else {'x': size_x, 'y': size_y, 'z': plan.size_z}
    axis_plans = _axis_plans(mesh_spec, parameters, lengths)
    refinements = [refinement for refinement in refinements
                   if any(refinement.refines(axis, base_dx) for axis in lengths)]
    for axis_plan in axis_plans.values():
        axis_plan.bands = [band for band in axis_plan.bands if band[2] < base_dx * (1.0 - _REFINEMENT_RTOL)]
    custom_widths = [np.diff(axis_plan.edges) for axis_plan in axis_plans.values() if axis_plan.edges is not None]
    finest_dx = min([base_dx] + [refinement.along(axis) for refinement in refinements for axis in lengths
                                 if refinement.refines(axis, base_dx)]
                    + [band[2] for axis_plan in axis_plans.values() for band in axis_plan.bands]
                    + [float(widths.min()) for widths in custom_widths])
    uniform_features = _features_requiring_uniform_mesh(on_mesh, registry)

    if not (refinements or custom_widths or any(axis_plan.bands for axis_plan in axis_plans.values())):
        plan.uniform_dx = base_dx
        plan.note = (f'{_MODE_LABELS[mode]} mesh: no refinement is finer than the main cell size, so the grid is '
                     f'uniform at {base_dx:.4g} m')
    elif uniform_features:
        plan.uniform_dx = finest_dx
        plan.note = (f"Uniform grid at the finest refinement cell size {finest_dx:.4g} m: "
                     f"{', '.join(uniform_features)} need{'s' if len(uniform_features) == 1 else ''} a uniform mesh")
    if plan.uniform_dx is not None:
        _check_cell_count(*_uniform_cell_counts(plan), reason=plan.note)
        return plan

    mesh_config, intervals = _graded_config(lengths, base_dx, mesh['max_ratio'], mesh, axis_plans, refinements, shapes)
    plan.mesh_config = mesh_config
    bounds = []
    for axis, length in lengths.items():   # core.mesh caps each snap segment's target at its length
        pml_cells = int(sides.get(f'{axis}_lo', 0)) + int(sides.get(f'{axis}_hi', 0))   # z: 0 without z PML
        if axis_plans[axis].edges is not None:
            bounds.append(len(axis_plans[axis].edges) - 1 + pml_cells)
            continue
        bounds.append(_snapped_axis_cell_bound(mesh_config, axis, length, base_dx, intervals[axis], mesh) + pml_cells)
    _check_cell_count(*bounds)
    return plan


def _graded_config(lengths, base_dx, max_ratio, mesh, axis_plans, refinements, shapes):
    """(MeshConfig at base_dx over lengths {'x', 'y'(, 'z')} [m], {axis: [(lo, hi, dx)] refinement intervals}) with the
    object refinements and the MeshAxisSpec plans (bands, custom edges) added: the graded tensor planner shared by
    tensor meshes and builtin tetrahedra."""
    mesh_config = MeshConfig(lengths['x'], lengths['y'], base_dx, max_ratio=max_ratio, size_z=lengths.get('z'),
                             axis_max_ratios={axis: axis_plan.max_ratio for axis, axis_plan in axis_plans.items()
                                              if axis_plan.max_ratio is not None},
                             outer_dx=mesh['outer_dx'], outer_ratio=mesh['outer_ratio'])
    intervals = {axis: [] for axis in lengths}
    for refinement in refinements:
        for axis, axis_intervals in _add_refinement(mesh_config, refinement, shapes, lengths['x'], lengths['y'],
                                                    base_dx).items():
            intervals[axis] += axis_intervals
        if 'z' in lengths and refinement.refines('z', base_dx):
            intervals['z'] += _add_z_refinement(mesh_config, refinement, shapes, lengths['z'])
    for axis, axis_plan in axis_plans.items():
        for lo, hi, dx in axis_plan.bands:   # band ends snap: else the whole-axis rescale moves the band
            mesh_config.add_refinement_band(axis, lo, hi, dx, (lo, hi))
            intervals[axis].append((lo, hi, dx))
        if axis_plan.edges is not None:
            mesh_config.set_axis_edges(axis, axis_plan.edges)
    return mesh_config, intervals


def _snapped_axis_cell_bound(mesh_config, axis, length, base_dx, intervals, mesh):
    """Cell-count bound of a graded axis: refinement intervals (lo, hi, dx), each snap segment capping its target at
    its length, the outer grading, + one rounded remainder per extra segment."""
    snaps = mesh_config.snap_positions(axis)
    edges = [0.0] + snaps + [float(length)]
    segments = [(lo, hi, hi - lo) for lo, hi in zip(edges[:-1], edges[1:])]
    core = mesh_config.outer_core(axis)
    outer = None if core is None else (*core, float(mesh['outer_dx']), mesh_config.axis_outer_ratio(axis))
    return _graded_axis_cell_bound(length, base_dx, mesh_config.ratio(axis), list(intervals) + segments,
                                   outer) + len(snaps)


def _mapped_plan(spec, parameters, plan, mesh, window_map, mesh_spec):
    """Plan of a polar / spherical window: radial cells at dx (nonuniform: graded by the x MeshAxisSpec bands / edges
    / segments and outer grading), angle cells uniform at dx along the outer arc unless a y MeshAxisSpec gives edges
    or segments [rad]; a cylindrical sector (size_z) adds _mapped_z_axis; no PML, auto mode or (outside uniform mode)
    object refinements (ValueError)."""
    kind = window_map.grid_text
    if any(plan.npml_sides.values()):
        raise ValueError(f"a {kind} domain has no absorbing (PML) padding: remove the physics that declares one")
    if mesh['mode'] == 'auto':
        raise ValueError(f"mesh mode 'auto' needs a 2D Cartesian domain (this one: {kind}); use uniform or nonuniform")
    if mesh_spec.refinements and mesh['mode'] != 'uniform':   # uniform mode ignores them (validate's info)
        raise ValueError(f"object mesh refinements need a 2D Cartesian domain (this one: {kind}): grade the radius by "
                         "a mesh axis 'x' entry (bands, edges or segments) instead")
    base_dx = mesh['dx']
    size_x, size_y = plan.size_x, plan.size_y
    lengths = {'x': size_x, 'y': size_y} if plan.size_z is None else {'x': size_x, 'y': size_y, 'z': plan.size_z}
    axis_plans = (_axis_plans(mesh_spec, parameters, lengths) if mesh['mode'] != 'uniform'
                  else {axis: _AxisPlan() for axis in lengths})
    radial_plan, angle_plan = axis_plans['x'], axis_plans['y']
    z_cells = () if plan.size_z is None else (_mapped_z_axis(plan.size_z, base_dx, mesh, axis_plans['z']),)
    z_bound = tuple(cell_bound for cell_bound, _ in z_cells)
    if window_map.line and (angle_plan.bands or angle_plan.edges is not None or angle_plan.max_ratio is not None):
        raise ValueError(f"a {kind} model has one cell across y: remove the mesh axis 'y' entry")
    if angle_plan.bands or angle_plan.max_ratio is not None:
        raise ValueError(f"mesh axis 'y' of a {kind} domain is the angle: give it edges or segments in rad (no bands "
                         'or max_ratio)')
    if window_map.line:
        n_angle = 1
    else:
        n_angle = (len(angle_plan.edges) - 1 if angle_plan.edges is not None
                   else max(1, math.ceil(size_y * window_map.radius_max / base_dx - 1e-9)))
    bands = [band for band in radial_plan.bands if band[2] < base_dx * (1.0 - _REFINEMENT_RTOL)]
    if mesh['mode'] == 'uniform' or not (bands or radial_plan.edges is not None):
        n_radial = uniform_cell_count(size_x, base_dx)
        _check_cell_count(n_radial, n_angle, *z_bound)
        radial_widths = np.full(n_radial, size_x / n_radial)
    else:
        mesh_config = MeshConfig(size_x, size_y, base_dx, max_ratio=mesh['max_ratio'],
                                 axis_max_ratios={} if radial_plan.max_ratio is None else {'x': radial_plan.max_ratio},
                                 outer_dx=mesh['outer_dx'], outer_ratio=mesh['outer_ratio'])
        for lo, hi, dx in bands:
            mesh_config.add_refinement_band('x', lo, hi, dx, (lo, hi))
        if radial_plan.edges is not None:
            mesh_config.set_axis_edges('x', radial_plan.edges)
            n_radial = len(radial_plan.edges) - 1
        else:
            n_radial = _snapped_axis_cell_bound(mesh_config, 'x', size_x, base_dx, bands, mesh)
        _check_cell_count(n_radial, n_angle, *z_bound)
        mesh_config.set_axis_edges('y', np.linspace(0.0, size_y, 2))   # the angle cells follow below
        radial_widths = mesh_config.cell_spacings()[0]
    angle_widths = (np.diff(angle_plan.edges) if angle_plan.edges is not None else np.full(n_angle, size_y / n_angle))
    plan.domain_map = window_map
    plan.mapped_widths = tuple(np.asarray(widths, dtype=np.float32).astype(np.float64)   # the Domain's float32 cells
                               for widths in (radial_widths, angle_widths) + tuple(make() for _, make in z_cells))
    plan.custom_axes = tuple(axis for axis, axis_plan in axis_plans.items() if axis_plan.edges is not None)
    return plan


def _mapped_z_axis(size_z, base_dx, mesh, z_plan):
    """(cell-count bound, make() -> float64 z widths [m]) of a cylindrical sector's axis: uniform at dx (size_z / n),
    or graded by the z MeshAxisSpec bands / edges / segments, max_ratio and outer grading (nonuniform mode)."""
    bands = [band for band in z_plan.bands if band[2] < base_dx * (1.0 - _REFINEMENT_RTOL)]
    if mesh['mode'] == 'uniform' or not (bands or z_plan.edges is not None):
        n_axial = uniform_cell_count(size_z, base_dx)
        return n_axial, lambda: np.full(n_axial, size_z / n_axial)
    mesh_config = MeshConfig(size_z, size_z, base_dx, max_ratio=mesh['max_ratio'], size_z=size_z,
                             axis_max_ratios={} if z_plan.max_ratio is None else {'z': z_plan.max_ratio},
                             outer_dx=mesh['outer_dx'], outer_ratio=mesh['outer_ratio'])
    for lo, hi, dx in bands:
        mesh_config.add_refinement_band('z', lo, hi, dx, (lo, hi))
    if z_plan.edges is not None:
        mesh_config.set_axis_edges('z', z_plan.edges)
        n_axial = len(z_plan.edges) - 1
    else:
        n_axial = _snapped_axis_cell_bound(mesh_config, 'z', size_z, base_dx, bands, mesh)
    return n_axial, lambda: np.asarray(mesh_config.z_cell_spacings(0, 0, pml_dx=base_dx), dtype=np.float64)


def _requirements(physics_specs, parameters, registry):
    """(merged DomainRequirements of the physics_specs (those on one mesh), True when a time-stepping physics declares
    a Courant factor); equal boundary configs merge, ValueError for different ones; periodic axes join, edge sizes
    take the smallest per object. Physics of unregistered types are skipped (validate reports them)."""
    boundary_config, provider, courant_factor, courant_dt = None, None, None, False
    periodic_axes, edge_sizes = set(), {}
    for physics_spec in physics_specs:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is None:
            continue
        settings = physics_settings(physics_type, physics_spec, parameters)
        requirements = (physics_type.domain_requirements(settings)
                        if physics_type.domain_requirements is not None else DomainRequirements())
        if requirements.boundary_config is not None:
            if provider is None:
                boundary_config, provider = requirements.boundary_config, physics_spec.name
            else:
                difference_text = boundary_config_difference_text(boundary_config, requirements.boundary_config)
                if difference_text:
                    raise ValueError(f"physics '{provider}' and '{physics_spec.name}' define different boundary "
                                     f"configurations ({difference_text}); physics sharing a mesh must use "
                                     f"equal bc_* and pml_* settings")
        if requirements.courant_factor is not None:
            factor = float(requirements.courant_factor)
            if not 0.0 < factor <= 1.0:
                raise ValueError(f"physics '{physics_spec.name}': Courant factor must be in (0, 1], got {factor}")
            courant_factor = factor if courant_factor is None else min(courant_factor, factor)
            if physics_type.steps_in_time is not None and physics_type.steps_in_time(settings):
                courant_dt = True
        periodic_axes.update(getattr(requirements, 'periodic_axes', ()))
        if physics_type.mesh_edge_sizes is not None:
            for name, size in physics_type.mesh_edge_sizes(settings, physics_spec):
                edge_sizes[name] = min(edge_sizes.get(name, float('inf')), float(size))
    return (DomainRequirements(boundary_config=boundary_config, courant_factor=courant_factor,
                               periodic_axes=tuple(axis for axis in 'xyz' if axis in periodic_axes),
                               edge_sizes=tuple(sorted(edge_sizes.items()))), courant_dt)


def _registered_physics_type(registry, physics_spec):
    try:
        return registry.physics_type(physics_spec.type)
    except KeyError:
        return None


def _features_requiring_uniform_mesh(physics_specs, registry):
    """Texts "<type label> 'physics.feature'" of the physics_specs' enabled features whose type requires a uniform
    mesh."""
    labels = []
    for physics_spec in physics_specs:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is None:
            continue
        for feature_spec in physics_spec.features:
            if not feature_spec.enabled:
                continue
            try:
                feature_type = physics_type.feature_type(feature_spec.type)
            except KeyError:
                continue
            if feature_type.requires_uniform_mesh:
                labels.append(f"{feature_type.label} '{physics_spec.name}.{feature_spec.name}'")
    return labels


def mesh_values(mesh_spec, parameters) -> dict:
    """resolve_values of a MeshSpec's MESH_PARAMS with the nulls of MESH_KIND_DEFAULTS filled for its kind;
    ValueError naming the mesh."""
    raw = {param.name: getattr(mesh_spec, param.name) for param in MESH_PARAMS}
    try:
        values = resolve_values(MESH_PARAMS, raw, parameters)
    except ValueError as exc:
        raise ValueError(f"mesh '{mesh_spec.name}': {exc}") from exc
    if values['dx'] is None and values['mode'] != 'auto':
        raise ValueError(f"mesh '{mesh_spec.name}': dx (cell size) is required in mode {values['mode']} (empty only "
                         'in auto mode)')
    for name, default in MESH_KIND_DEFAULTS[values['kind']].items():
        if values[name] is None:
            values[name] = default
    return values


def _positive_length(value, parameters, label, allow_zero=False):
    number = evaluate_scalar(value, parameters)
    if not (number >= 0.0 if allow_zero else number > 0.0):
        raise ValueError(f"{label} must be {'>= 0' if allow_zero else '> 0'}, got {number:g}")
    return number


def _explicit_refinements(spec, mesh_spec, parameters, shapes, places=('inside',)):
    """_Refinements of MeshSpec.refinements with numeric sizes; ValueError naming the entry on bad input or a `where`
    outside places (default 'inside': tensor meshes, triangles, builtin tetrahedra)."""
    refinements = []
    for index, refinement in enumerate(mesh_spec.refinements):
        if refinement.automatic and mesh_spec.mode == 'auto':   # filled-in auto sizes: auto mode computes its own
            continue
        label = f"mesh '{mesh_spec.name}' refinement {index + 1} ({refinement.object})"
        if refinement.object not in shapes:
            raise ValueError(f'{label}: geometry object {refinement.object!r} is unknown or failed to evaluate')
        where = getattr(refinement, 'where', 'inside')
        if where not in REFINEMENT_PLACES:
            raise ValueError(f"{label}: where must be one of {', '.join(REFINEMENT_PLACES)}, got {where!r}")
        if where not in places:
            raise ValueError(f"{label}: where '{where}' applies to tetrahedron meshes with mesher 'gmsh' only (size "
                             "fields along the edges / at the surface); use 'inside' or mesher gmsh")
        dx = _positive_length(refinement.dx, parameters, f'{label} dx')
        margin = (None if refinement.margin is None
                  else _positive_length(refinement.margin, parameters, f'{label} margin', allow_zero=True))
        axis_dx = {}
        for axis in ('x', 'y', 'z'):
            value = getattr(refinement, f'dx_{axis}')
            if value is None:
                continue
            if axis == 'z' and spec.domain.size_z is None:
                raise ValueError(f'{label}: dx_z needs a 3D model (domain size_z)')
            axis_dx[axis] = _positive_length(value, parameters, f'{label} dx_{axis}')
        refinements.append(_Refinement(refinement.object, dx, margin, axis_dx, where))
    return refinements


def _axis_plans(mesh_spec, parameters, lengths):
    """{axis: _AxisPlan} of every axis in lengths from MeshSpec.axes (ValueError on bad input, naming the entry)."""
    plans = {axis: _AxisPlan() for axis in lengths}
    seen = set()
    for index, axis_spec in enumerate(mesh_spec.axes):
        label = f'mesh axis entry {index + 1} ({axis_spec.axis!r})'
        if axis_spec.axis not in ('x', 'y', 'z'):
            raise ValueError(f"{label}: axis must be 'x', 'y' or 'z'")
        if axis_spec.axis not in lengths:
            raise ValueError(f'{label}: axis z needs a 3D model (domain size_z)')
        if axis_spec.axis in seen:
            raise ValueError(f'{label}: axis {axis_spec.axis} has another entry; use one entry per axis')
        seen.add(axis_spec.axis)
        try:
            _fill_axis_plan(plans[axis_spec.axis], axis_spec, parameters, float(lengths[axis_spec.axis]), label)
        except ExpressionError as exc:   # the other errors name their entry already
            raise ValueError(f'{label}: {exc}') from exc
    return plans


def _fill_axis_plan(axis_plan, axis_spec, parameters, length, label):
    """Resolve one MeshAxisSpec into axis_plan (max_ratio, bands, custom edges)."""
    if axis_spec.max_ratio is not None:
        axis_plan.max_ratio = evaluate_scalar(axis_spec.max_ratio, parameters)
        if not axis_plan.max_ratio > 1.0:
            raise ValueError(f'{label}: max_ratio must be > 1, got {axis_plan.max_ratio:g}')
    for band_number, band in enumerate(axis_spec.bands, start=1):
        where = f'{label} band {band_number}'
        lo, hi = (evaluate_scalar(value, parameters) for value in (band.lo, band.hi))
        if not hi > lo:
            raise ValueError(f'{where}: hi must be > lo, got lo = {lo:g}, hi = {hi:g}')
        axis_plan.bands.append((lo, hi, _positive_length(band.dx, parameters, f'{where} dx')))
    if axis_spec.edges and axis_spec.segments:
        raise ValueError(f'{label}: give edges or segments, not both')
    if axis_spec.edges:
        axis_plan.edges = complete_axis_edges([evaluate_scalar(value, parameters) for value in axis_spec.edges],
                                              length, axis_spec.axis)
    elif axis_spec.segments:
        axis_plan.edges = _segment_edges(axis_spec.segments, parameters, length, label)


def _segment_edges(segments, parameters, length, label):
    """Float64 edges 0 .. length of a custom axis built from MeshSegmentSpec pieces (core.mesh.segment_widths)."""
    edges, start = [0.0], 0.0
    for number, segment in enumerate(segments, start=1):
        where = f'{label} segment {number}'
        last = number == len(segments)
        if segment.end is None and not last:
            raise ValueError(f'{where}: only the last segment may leave end empty (= the axis end)')
        end = length if segment.end is None else evaluate_scalar(segment.end, parameters)
        if last and abs(end - length) > 1e-9 * length:
            raise ValueError(f'{where}: the last segment must end at the axis end {length:g} m (or leave end '
                             f'empty), got {end:g}')
        if not end > start:
            raise ValueError(f'{where}: end {end:g} must be beyond the previous end {start:g}')
        cells = evaluate_scalar(segment.cells, parameters)
        if not (cells >= 1 and abs(cells - round(cells)) < 1e-6):
            raise ValueError(f'{where}: cells must be a whole number >= 1, got {cells:g}')
        if segment.profile not in SEGMENT_PROFILES:
            raise ValueError(f"{where}: profile must be one of {', '.join(SEGMENT_PROFILES)}, got "
                             f'{segment.profile!r}')
        try:
            widths = segment_widths(end - start, round(cells), segment.profile,
                                    evaluate_scalar(segment.ratio, parameters))
        except ValueError as exc:
            raise ValueError(f'{where}: {exc}') from exc
        edges.extend(start + np.cumsum(widths)[:-1])
        edges.append(length if last else end)
        start = end
    return np.array(edges, dtype=np.float64)


@dataclass
class _AutoSizes:
    """Auto mode of one mesh: base size, object refinements and the rule that chose them."""
    base_dx: float
    refinements: list
    rule: str                     # 'wavelength' | 'physics' | 'uniform_flow' (MeshEstimate.auto_rule)
    wavelength_min: float | None = None
    wavelength_source: str = ''
    note: str = ''

    def finer_refinements(self) -> tuple:
        """(object, size) of the refinements finer than the base size (the ones the mesh applies)."""
        return tuple((refinement.object_name, float(refinement.dx)) for refinement in self.refinements
                     if refinement.dx < self.base_dx * (1.0 - _REFINEMENT_RTOL))

    def scaled_to_smallest(self, smallest_dx):
        """These sizes scaled by one factor so the finest applied size (base or finer refinement) is smallest_dx."""
        finest_dx = min([self.base_dx] + [size for _name, size in self.finer_refinements()])
        factor = float(smallest_dx) / finest_dx
        refinements = [replace(refinement, dx=refinement.dx * factor) for refinement in self.refinements]
        note = self.note
        if self.rule == 'uniform_flow':
            note = f'Automatic mesh: uniform cells at the cell size {float(smallest_dx):.4g} m'
        return replace(self, base_dx=self.base_dx * factor, refinements=refinements, note=note)


def _auto_sizes(spec, on_mesh, parameters, registry, shapes, mesh):
    """_AutoSizes of auto mode on a mesh holding the physics on_mesh (_auto_rule_sizes), scaled so the finest cells
    take the mesh dx when it is set (the smallest cell size)."""
    sizes = _auto_rule_sizes(spec, on_mesh, parameters, registry, shapes, mesh)
    return sizes if mesh['dx'] is None else sizes.scaled_to_smallest(mesh['dx'])


def _auto_rule_sizes(spec, on_mesh, parameters, registry, shapes, mesh):
    """_AutoSizes of the auto rule alone: the wavelength rule with an EM physics (family 'em') on the mesh or
    wavelength_min set, else the S7.8 physics-controlled rule (lattice Boltzmann types: uniform)."""
    present_types = _present_physics_types(on_mesh, registry)
    if mesh['wavelength_min'] is not None or any(physics_type.family == 'em' for physics_type in present_types):
        wavelength_min, source = _auto_wavelength(spec, on_mesh, parameters, registry, mesh['wavelength_min'])
        base_dx, refinements = _auto_refinements(spec, on_mesh, parameters, shapes, mesh, registry, wavelength_min)
        return _AutoSizes(base_dx, refinements, 'wavelength', wavelength_min, source)
    lengths = _window_lengths(spec, parameters)
    sides = {}   # domain object -> smallest bounding-box side, over the objects the mesh's physics use
    for physics_spec in on_mesh:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is not None:
            for name in physics_objects(spec, physics_spec, physics_type, shapes):
                side = smallest_side(name, shapes, lengths)
                if side is not None:
                    sides[name] = side
    flow_types = [physics_type.name for physics_type in present_types if physics_type.name in UNIFORM_AUTO_DIVISORS]
    if flow_types:
        divisor = max(UNIFORM_AUTO_DIVISORS[name] for name in flow_types)
        dx = min(list(sides.values()) or [min(lengths)]) / divisor
        return _AutoSizes(dx, [], 'uniform_flow', note=f"Automatic mesh: uniform cells at {dx:.4g} m (smallest object "
                                                       f"side / {divisor:g}) for {', '.join(flow_types)}")
    base_dx = max(lengths) / PHYSICS_BASE_DIVISOR
    refinements = [_Refinement(name, side / PHYSICS_OBJECT_DIVISOR, None) for name, side in sides.items()
                   if side / PHYSICS_OBJECT_DIVISOR < base_dx * (1.0 - _REFINEMENT_RTOL)]
    return _AutoSizes(base_dx, refinements, 'physics')


def _window_lengths(spec, parameters):
    """Physical window lengths [m]: (size_x,) of a 1D model, (size_x, size_y) in 2D, + size_z in 3D."""
    size_x, size_y = domain_size(spec, parameters)
    if spec.dimensions == 1:
        return (size_x,)
    size_z = domain_size_z(spec, parameters)
    return (size_x, size_y) if size_z is None else (size_x, size_y, size_z)


def physics_objects(spec, physics_spec, physics_type, shapes) -> list:
    """Resolved domain objects a physics uses: material selections setting one of its type's properties (library
    merged) and its enabled features' selections (spec order, once each)."""
    property_names = {param.name for param in physics_type.material_properties}
    names = []
    for material in spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
            if property_names & {name for name, value in properties.items() if value is not None}:
                names += expand_entities(spec, material.selection)
        except (ValueError, TypeError, KeyError):   # reported by validate
            continue
    for feature in physics_spec.features:
        if feature.enabled:
            try:
                names += expand_entities(spec, feature.selection)
            except (ValueError, TypeError, KeyError):
                continue
    return [name for name in dict.fromkeys(names)
            if name in shapes and GEOMETRY_KINDS[shapes[name].kind].entity_kind == 'domain']


def smallest_side(name, shapes, lengths) -> float | None:
    """Smallest positive side of a domain object's bounding box clipped to the window (3D: its z extent too), None
    when it has none."""
    x0, y0, x1, y1 = bounding_box(name, shapes)
    spans = [(x0, x1, lengths[0])] + ([(y0, y1, lengths[1])] if len(lengths) > 1 else [])
    if len(lengths) == 3:
        spans.append((*z_extent(name, shapes), lengths[2]))
    sides = [min(hi, length) - max(lo, 0.0) for lo, hi, length in spans]
    sides = [side for side in sides if side > 0.0 and math.isfinite(side)]
    return min(sides) if sides else None


def _auto_wavelength(spec, on_mesh, parameters, registry, wavelength_setting):
    """(wavelength_min [m], 'mesh' or the physics name) of auto mode: the setting, else the shortest over the
    shortest_wavelength hooks of the physics on the mesh; ValueError when neither gives one."""
    if wavelength_setting is not None:
        return wavelength_setting, 'mesh'
    shortest, source = None, ''
    for physics_spec in on_mesh:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is None or physics_type.shortest_wavelength is None:
            continue
        settings = physics_settings(physics_type, physics_spec, parameters)
        wavelength = physics_type.shortest_wavelength(spec, physics_spec, settings, parameters, registry)
        if wavelength is not None and (shortest is None or wavelength < shortest):
            shortest, source = float(wavelength), physics_spec.name
    if shortest is None:
        raise ValueError('auto mesh: no physics gives a wavelength to resolve (EM physics take their shortest solved '
                         'or source wavelength: check their sources and the study settings); set the mesh '
                         'wavelength_min')
    return shortest, source


def _auto_refinements(spec, on_mesh, parameters, shapes, mesh, registry, wavelength_min):
    """(base dx, refinements) of the wavelength rule: background index from 'all' materials, objects refined by
    index, both over the permittivity arrays of the types on the mesh."""
    points_per_wavelength = mesh['points_per_wavelength']
    present_types = _present_physics_types(on_mesh, registry)
    background_index = 1.0
    object_targets = {}
    for material in spec.materials:
        index = _material_index(material, parameters, present_types, wavelength_min)
        try:
            entities = expand_entities(spec, material.selection)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"auto mesh: material '{material.name}' selection: {exc}") from exc
        if 'all' in entities:
            background_index = max(background_index, index)
            continue
        if index <= 1.0:
            continue
        target_dx = wavelength_min / (points_per_wavelength * index)
        for entity in entities:
            shape = shapes.get(entity)
            if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
                continue   # faces, failed objects (reported by validate)
            object_targets[entity] = min(object_targets.get(entity, math.inf), target_dx)
    base_dx = wavelength_min / (points_per_wavelength * background_index)
    return base_dx, [_Refinement(name, dx, None) for name, dx in object_targets.items()]


def _present_physics_types(physics_specs, registry):
    """Registered physics types of the physics_specs (all of a spec, or those on one mesh), each once, in order."""
    present_types = {}
    for physics_spec in physics_specs:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is not None:
            present_types.setdefault(physics_type.name, physics_type)
    return list(present_types.values())


def mesh_material_types(spec, registry, mesh) -> list:
    """PhysicsTypes whose material arrays mesh `mesh` (name) holds (S7.2): those of its physics (spec order), then the
    partner types of enabled cross-mesh couplings whose transform runs on it and reads the partner's materials
    (CouplingType.partner_materials: a 'field' coupling's source type on the target mesh, a 'value' one's target type
    on the source mesh). One mesh: exactly the model's types."""
    types = _present_physics_types(spec.physics_on(mesh), registry)
    for coupling in spec.couplings:
        try:
            coupling_type = registry.coupling_type(coupling.type)
            source_mesh, target_mesh = spec.mesh_of(coupling.source).name, spec.mesh_of(coupling.target).name
        except KeyError:   # reported by validate
            continue
        if (not coupling.enabled or not coupling_type.partner_materials or source_mesh == target_mesh
                or coupling_type.mesh_transfer is None):
            continue
        field = coupling_type.mesh_transfer == 'field'
        if (target_mesh if field else source_mesh) != mesh:
            continue
        partner = _registered_physics_type(registry, spec.physics_by_name(coupling.source if field else coupling.target))
        if partner is not None and all(partner.name != physics_type.name for physics_type in types):
            types.append(partner)
    return types


def _material_index(material, parameters, present_types, wavelength):
    """Largest |sqrt(eps(omega) mu_r)| (>= 1) over the present types' permittivity arrays at the vacuum wavelength:
    background (owning property, else first Drude eps_inf, else 1) - j sigma_e/(omega eps0) + used poles' chi(omega)."""
    from ..physics.em.definition_common import background_permittivity, make_pole
    try:
        properties, dispersion = effective_material_data(material.library, material.properties, material.dispersion)
    except ValueError as exc:
        raise ValueError(f"auto mesh: material '{material.name}': {exc}") from exc
    set_names = frozenset(properties)
    used_properties = {param.name for physics_type in present_types for param in physics_type.material_properties}
    mu_r = _index_value(material, 'mu_r', properties['mu_r'], parameters) if (
        'mu_r' in used_properties and 'mu_r' in set_names) else 1.0
    sigma_e = _index_value(material, 'sigma_e', properties['sigma_e'], parameters) if (
        'sigma_e' in used_properties and 'sigma_e' in set_names) else 0.0
    poles_with_masks = []
    if any(physics_type.uses_dispersion for physics_type in present_types):
        for pole_number, pole in enumerate(dispersion, start=1):
            try:
                poles_with_masks.append((make_pole(pole, lambda value: evaluate_scalar(value, parameters)), None))
            except ValueError as exc:
                raise ValueError(f"auto mesh: material '{material.name}' dispersion pole {pole_number}: {exc}") from exc
    omega = 2.0 * math.pi * C0 / wavelength
    index = 1.0
    for physics_type in present_types:
        for array in permittivity_arrays(physics_type.material_properties):
            label = array_owner(array, physics_type.material_properties, set_names) or 'drude eps_inf'
            background = _index_value(material, label, background_permittivity(
                properties, dispersion, physics_type.material_properties, array), parameters)
            permittivity = complex(complex_permittivity(background, sigma_e, poles_with_masks, omega))
            index = max(index, abs(cmath.sqrt(permittivity * mu_r)))
    return index


def _index_value(material, name, value, parameters):
    """Value of a material property entering the auto-mesh index; it must evaluate over the parameters."""
    try:
        return evaluate_scalar(value, parameters)
    except ExpressionError as exc:
        raise ValueError(f"auto mesh: material '{material.name}' {name} must be a number or an expression "
                         f"of parameters ({exc}); use mesh mode 'nonuniform'") from exc


def _add_refinement(mesh_config, refinement, shapes, size_x, size_y, base_dx):
    """Add the x / y bands of one object refinement to mesh_config, each axis at its own dx (only the axes it refines);
    returns {axis: [(lo, hi, dx)]}."""
    shape = shapes[refinement.object_name]
    values = shape.values
    if shape.kind == 'line':
        orientation = line_orientation(values, size_x, size_y)
        if orientation in ('horizontal', 'vertical'):   # a horizontal line refines the y spacing around its y
            axis = 'y' if orientation == 'horizontal' else 'x'
            if not refinement.refines(axis, base_dx):
                return {}
            dx = refinement.along(axis)
            width = max(refinement.margin_along(axis), dx)
            position = 0.5 * (values[f'{axis}0'] + values[f'{axis}1'])
            mesh_config.add_refinement_line(axis, position, dx, width)
            return {axis: [(position - width, position + width, dx)]}
    added = {}
    faces = axis_parallel_faces(refinement.object_name, shapes)   # snapped to cell edges (core.mesh snap_positions)
    for axis in ('x', 'y'):
        if not refinement.refines(axis, base_dx):
            continue
        dx, margin = refinement.along(axis), refinement.margin_along(axis)
        if shape.kind in ('circle', 'cylinder_z') and is_plain(shape):   # 3D: z by _add_z_refinement
            centre, reach = values[f'c{axis}'], values['r'] + margin
            lo, hi = centre - reach, centre + reach
        else:   # the rest, x / y extrusions and tapers included: bounding box (rectangle / box: its sides)
            reach = max(margin, dx) if shape.kind == 'point' else margin
            box = bounding_box(refinement.object_name, shapes)
            lo, hi = (box[0], box[2]) if axis == 'x' else (box[1], box[3])
            lo, hi = lo - reach, hi + reach
        mesh_config.add_refinement_band(axis, lo, hi, dx, faces.get(axis, ()))
        added[axis] = [(lo, hi, dx)]
    return added


def _add_z_refinement(mesh_config, refinement, shapes, size_z):
    """Add the z band of one refinement (3D): the object's z_extent (points / lines at z, unset z = the whole height;
    open prism ends clamped) plus its margin (points / lines max(margin, dx)), at the refinement's z size; its flat z
    faces (geometry.axis_parallel_faces) become z snap nodes. Returns [(lo, hi, dx)] along z ([] when the band misses
    the domain)."""
    dx, margin = refinement.along('z'), refinement.margin_along('z')
    shape = shapes[refinement.object_name]
    reach = max(margin, dx) if shape.kind in ('point', 'line') else margin
    z_lo, z_hi = z_extent(refinement.object_name, shapes)
    lo, hi = max(z_lo - reach, 0.0), min(z_hi + reach, float(size_z))
    if not hi > lo:
        return []
    mesh_config.add_refinement_band('z', lo, hi, dx, axis_parallel_faces(refinement.object_name, shapes).get('z', ()))
    return [(lo, hi, dx)]


def _graded_axis_cell_bound(length, base_dx, max_ratio, intervals, outer=None):
    """Upper bound on the physical cells of one graded axis: per piece between interval edges, its finest covering dx
    (else base_dx; outer = (core_lo, core_hi, outer_dx, outer_ratio): outer_dx and outer_ratio ramps outside the core)
    plus ramps from the graded profile at its two edges, so overlapping intervals count once."""
    clipped = sorted((max(lo, 0.0), min(hi, length), dx) for lo, hi, dx in intervals
                     if min(hi, length) > max(lo, 0.0))
    edges = sorted({0.0, float(length)} | {edge for lo, hi, _ in clipped for edge in (lo, hi)}
                   | (set(outer[:2]) if outer is not None else set()))
    piece_widths = [hi - lo for lo, hi in zip(edges[:-1], edges[1:])]
    piece_targets, piece_slopes, active, next_interval = [], [], [], 0   # active: heap of (dx, hi) started so far
    for piece_lo, piece_hi in zip(edges[:-1], edges[1:]):
        while next_interval < len(clipped) and clipped[next_interval][0] <= piece_lo:
            heapq.heappush(active, (clipped[next_interval][2], clipped[next_interval][1]))
            next_interval += 1
        while active and active[0][1] <= piece_lo:
            heapq.heappop(active)
        outside = outer is not None and (piece_hi <= outer[0] or piece_lo >= outer[1])
        background, ratio = (outer[2], outer[3]) if outside else (base_dx, max_ratio)
        piece_targets.append(min(background, active[0][0]) if active else background)
        piece_slopes.append((ratio - 1.0) / ratio)   # core.mesh `_graded_profile` slope (growth 2 - 1/ratio per cell)
    n_pieces = len(piece_widths)
    from_left, from_right = [math.inf] * (n_pieces + 1), [math.inf] * (n_pieces + 1)   # graded profile at the edges
    for k in range(n_pieces):
        from_left[k + 1] = min(from_left[k] + piece_slopes[k] * piece_widths[k], piece_targets[k])
    for k in range(n_pieces - 1, -1, -1):
        from_right[k] = min(from_right[k + 1] + piece_slopes[k] * piece_widths[k], piece_targets[k])
    count = 2.0
    for k in range(n_pieces):
        width, target, log_growth = piece_widths[k], piece_targets[k], math.log(1.0 + piece_slopes[k])
        count += width / target
        for edge_dx in (from_left[k], from_right[k + 1]):
            if edge_dx < target:
                count += min(math.log(target / edge_dx) / log_growth + 1.0, width / edge_dx)
    return math.ceil(count)


def _uniform_cell_counts(plan):
    """(nx, ny) of Domain.uniform for the plan (core.grid.uniform_cell_count, PML included)."""
    dx = plan.uniform_dx
    sides = plan.npml_sides
    total_x = plan.size_x + (sides['x_lo'] + sides['x_hi']) * dx
    total_y = plan.size_y + (sides['y_lo'] + sides['y_hi']) * dx
    return uniform_cell_count(total_x, dx), uniform_cell_count(total_y, dx)


def _check_cell_count(*axis_counts, reason=''):
    """ValueError('mesh too large: ~N cells ...') above the cell limits for (nx, ny) or (nx, ny, nz); reason explains
    a uniform fallback."""
    n_cells = math.prod(int(count) for count in axis_counts)
    if n_cells > MAX_MESH_CELLS or max(axis_counts) > MAX_AXIS_CELLS:
        shape_text = ' x '.join(f'{int(count):,}' for count in axis_counts)
        raise ValueError(f'mesh too large: ~{n_cells:,} cells ({shape_text}); the limits are '
                         f'{MAX_MESH_CELLS:,} cells and {MAX_AXIS_CELLS:,} cells per axis'
                         + (f' ({reason})' if reason else '')
                         + '; lengths are in m (e.g. 4e-6 for 4 um), so check the domain size and cell sizes')


# ---- triangle meshes (S7.5, CPU only) ----
def triangle_sizing(spec, mesh_spec, parameters, registry=None, shapes=None) -> TriangleSizing:
    """Resolved sizes of a 'triangles' mesh: h_max = dx (auto: the rules of its physics), object sizes from its (and
    the auto) refinements, h_min, growth, curvature factor and the PML ring of its physics' boundary configuration;
    ValueError when triangles do not fit (not 2D Cartesian, axes options, per-axis refinement sizes, h_min > h_max)."""
    window_map = domain_map(spec, parameters)
    if spec.dimensions != 2 or window_map.mapped:
        model_text = f'{spec.dimensions}D' if spec.dimensions != 2 else window_map.grid_text
        raise ValueError(f"mesh '{mesh_spec.name}': triangle meshes need a 2D Cartesian domain (this model: "
                         f"{model_text}); use kind 'tensor'{' or tetrahedra' if spec.dimensions == 3 else ''}")
    return _element_sizing(TriangleSizing, FACES, 'triangles', spec, mesh_spec, parameters, registry, shapes)


def tetrahedron_sizing(spec, mesh_spec, parameters, registry=None, shapes=None) -> TetrahedronSizing:
    """Resolved sizes of a 'tetrahedra' mesh: the triangle_sizing rules (h_max, object sizes, h_min, growth,
    curvature factor, size points [x, y, z, h]) with the PML shell thickness cpml_npml x h_max per cpml face and the
    face sizes, mesher 'builtin' also its axis_edges (builtin_axis_edges); ValueError when tetrahedra do not fit (not
    3D Cartesian, gmsh with axes options / per-axis refinement sizes, builtin with size_points, mirror options, h_min >
    h_max)."""
    if spec.dimensions != 3 or domain_map(spec, parameters).mapped:
        raise ValueError(f"mesh '{mesh_spec.name}': tetrahedron meshes need a 3D Cartesian domain (this model: "
                         f"{spec.dimensions}D, coordinates '{spec.domain.coordinates}'); use kind 'tensor'"
                         f"{' or triangles' if spec.dimensions == 2 else ''}")
    return _element_sizing(TetrahedronSizing, FACES_3D, 'tetrahedra', spec, mesh_spec, parameters, registry, shapes)


def _element_sizing(sizing_class, faces, kind_text, spec, mesh_spec, parameters, registry, shapes):
    """The sizes of a triangle / tetrahedron mesh (triangle_sizing); PML widths over `faces`."""
    registry = registry if registry is not None else get_registry()
    shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
    label = f"mesh '{mesh_spec.name}'"
    three_d = faces == FACES_3D
    if mesh_spec.face_sizes and not three_d:
        raise ValueError(f'{label}: face_sizes apply to tetrahedron meshes only; use refinements for {kind_text}')
    mesh = mesh_values(mesh_spec, parameters)
    swept = mesh['sweep_axis'] == 'z'   # prism layers (plan D5): the 2D mesher in-plane, z graded like builtin
    if swept:
        from .swept_meshing import layer_edges, swept_mesh_problem, swept_refinement_problem
        problem = swept_mesh_problem(mesh_spec, three_d)
        if problem:
            raise ValueError(f'{label}: {problem}')
    graded_builtin = three_d and mesh['mesher'] == 'builtin' and not swept   # the tensor planner: axes, per-axis sizes
    if mesh_spec.axes and not (graded_builtin or swept):
        raise ValueError(f'{label}: mesh axes options apply to tensor meshes and builtin tetrahedra only; remove them '
                         f"for {kind_text}{' or use mesher builtin' if three_d else ''}")
    if graded_builtin and mesh_spec.size_points:
        raise ValueError(f"{label}: size_points need mesher 'gmsh' (builtin tetrahedra are graded along whole axes by "
                         'refinements, face_sizes and axes options); remove them or use mesher gmsh')
    mirror_axes = tuple(axis for axis in ('x', 'y') if mesh[f'mirror_{axis}'] is True)
    if mirror_axes and three_d:
        raise ValueError(f'{label}: mirror_x / mirror_y apply to triangle meshes only')
    on_mesh = spec.physics_on(mesh_spec.name)
    requirements, _ = _requirements(on_mesh, parameters, registry)
    sides = _absorbing_cells(requirements.boundary_config, faces)
    auto = _auto_sizes(spec, on_mesh, parameters, registry, shapes, mesh) if mesh['mode'] == 'auto' else None
    h_max = mesh['dx'] if auto is None else auto.base_dx
    refinements = []
    places = REFINEMENT_PLACES if three_d and not (graded_builtin or swept) else ('inside',)   # gmsh fields (D3)
    if mesh['mode'] == 'nonuniform' or (auto is not None and auto.rule != 'uniform_flow'):
        refinements = ([] if auto is None else auto.refinements) + _explicit_refinements(spec, mesh_spec, parameters,
                                                                                        shapes, places)
    object_sizes, surface_sizes, refined_edges = {}, {}, []
    for refinement in refinements:
        problem = swept_refinement_problem(refinement) if swept else ''
        if problem:
            raise ValueError(f'{label} refinement ({refinement.object_name}): {problem}')
        if refinement.axis_dx and not (graded_builtin or swept):
            raise ValueError(f'{label} refinement ({refinement.object_name}): dx_x / dx_y / dx_z apply to tensor '
                             'meshes and builtin tetrahedra only; give dx')
        if not refinement.dx < h_max * (1.0 - _REFINEMENT_RTOL):
            continue
        if refinement.where == 'edges':   # joins the physics' edge sizes (smallest per object)
            refined_edges.append((refinement.object_name, refinement.dx))
            continue
        sizes = surface_sizes if refinement.where == 'surface' else object_sizes
        sizes[refinement.object_name] = min(sizes.get(refinement.object_name, math.inf), refinement.dx)
    h_min = h_max / 100.0 if mesh['h_min'] is None else float(mesh['h_min'])
    if h_min > h_max:
        raise ValueError(f'{label}: h_min {h_min:g} m exceeds the maximum element size {h_max:g} m')
    dimensions = 3 if three_d else 2
    problem = size_points_problem(mesh_spec.size_points, dimensions)
    if problem:
        raise ValueError(f'{label}: {problem}')
    size_points = quantised_size_points(mesh_spec.size_points, h_min, dimensions)
    pml_widths = {face: int(sides.get(face, 0)) * float(h_max) for face in faces}
    for axis in mirror_axes:
        if pml_widths[f'{axis}_lo'] != pml_widths[f'{axis}_hi']:
            raise ValueError(f'{label}: mirror_{axis} needs equal absorbing rings on {axis}_lo and {axis}_hi (now '
                             f"{pml_widths[f'{axis}_lo']:g} m and {pml_widths[f'{axis}_hi']:g} m): give both faces "
                             'the same boundary condition')
    periodic_axes = tuple(axis for axis in requirements.periodic_axes if three_d or axis != 'z')
    for axis in periodic_axes if not three_d else ():
        if pml_widths.get(f'{axis}_lo') or pml_widths.get(f'{axis}_hi'):
            raise ValueError(f'{label}: periodic {axis} needs no absorbing ring on {axis}_lo / {axis}_hi; '
                             'give those faces a non-absorbing condition')
    sizing = sizing_class(
        h_max=float(h_max), h_min=h_min, growth=float(mesh['max_ratio']),
        curvature_factor=float(mesh['curvature_factor']), object_sizes=object_sizes,
        pml_widths=pml_widths, pml_layers=int(mesh['pml_layers']), mesher=mesh['mesher'],
        wavelength_min=None if auto is None else auto.wavelength_min,
        wavelength_source='' if auto is None else auto.wavelength_source, auto_rule='' if auto is None else auto.rule,
        auto_dx=None if auto is None else float(auto.base_dx),
        auto_refinements=() if auto is None else auto.finer_refinements(), size_points=size_points,
        size_points_digest=hashlib.sha1(size_points.tobytes()).hexdigest() if len(size_points) else '',
        periodic_axes=periodic_axes,
        edge_sizes=(_edge_sizes(spec, (*requirements.edge_sizes, *refined_edges), shapes, h_max) if three_d
                    else {}),
        mirror_axes=mirror_axes,
        face_sizes=_sized_faces(spec, mesh_spec, parameters, shapes, h_max) if three_d else (),
        surface_sizes=surface_sizes)
    if swept:   # the z layers: graded_window_edges z (object faces snapped) + the PML shell
        edges = layer_edges(spec, mesh_spec, parameters, shapes, sizing)
        return replace(sizing, sweep_axis='z', layer_edges=edges,
                       layer_edges_digest=hashlib.sha1(edges.tobytes()).hexdigest())
    if not graded_builtin:
        return sizing
    axis_edges = tuple(builtin_axis_edges(spec, mesh_spec, parameters, shapes, sizing))
    return replace(sizing, axis_edges=axis_edges,
                   axis_edges_digest=hashlib.sha1(b''.join(edges.tobytes() for edges in axis_edges)).hexdigest())


def _sized_faces(spec, mesh_spec, parameters, shapes, h_max) -> tuple:
    """fem.gmsh3d.SizedFace per refined face of MeshSpec.face_sizes (outer faces: the window face; domain objects:
    their bounding-box faces clipped to the window, all six or the named one), sizes below h_max only; ValueError
    naming the entry on bad input."""
    if not mesh_spec.face_sizes:
        return ()
    from ..fem.gmsh3d import SizedFace
    window_high = np.array([*domain_size(spec, parameters), domain_size_z(spec, parameters)], dtype=np.float64)
    sized = []
    for index, entry in enumerate(mesh_spec.face_sizes):
        label = f"mesh '{mesh_spec.name}' face size {index + 1} ({entry.entity!r})"
        size = _positive_length(entry.size, parameters, f'{label} size')
        distance = _positive_length(entry.distance, parameters, f'{label} distance', allow_zero=True)
        if entry.entity in FACES_3D:
            if entry.face:
                raise ValueError(f"{label}: face applies to geometry objects only; leave it '' for an outer face")
            low, high, chosen = np.zeros(3), window_high, (entry.entity,)
        else:
            shape = shapes.get(entry.entity)
            if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
                raise ValueError(f'{label}: the entity must be an outer face ({", ".join(FACES_3D)}) or a domain '
                                 'object')
            if entry.face not in ('', *FACES_3D):
                raise ValueError(f"{label}: face must be '' (all six) or one of {', '.join(FACES_3D)}, got "
                                 f'{entry.face!r}')
            x0, y0, x1, y1 = bounding_box(entry.entity, shapes)
            z0, z1 = z_extent(entry.entity, shapes)
            low = np.clip(np.array([x0, y0, z0], dtype=np.float64), 0.0, window_high)
            high = np.clip(np.array([x1, y1, z1], dtype=np.float64), 0.0, window_high)
            chosen = (entry.face,) if entry.face else FACES_3D
        if not size < h_max * (1.0 - _REFINEMENT_RTOL):
            continue
        for face in chosen:
            axis = 'xyz'.index(face[0])
            others = [k for k in range(3) if k != axis]
            sized.append(SizedFace(axis, float((high if face.endswith('hi') else low)[axis]),
                                   tuple(float(low[k]) for k in others), tuple(float(high[k]) for k in others),
                                   float(size), float(distance)))
    return tuple(sized)


def _edge_sizes(spec, entity_sizes, shapes, h_max) -> dict:
    """Geometry object -> smallest edge size [m] below h_max of the (entity or named selection, size) pairs."""
    sizes = {}
    for entity, size in entity_sizes:
        try:
            names = expand_entities(spec, [entity])
        except (ValueError, TypeError):   # reported by validate
            continue
        for name in names:
            if name in shapes and 0.0 < size < h_max * (1.0 - _REFINEMENT_RTOL):
                sizes[name] = min(sizes.get(name, math.inf), float(size))
    return sizes


def size_points_problem(entries, dimensions=2) -> str:
    """'' when MeshSpec.size_points is a list of [x, y, h] (dimensions 3: [x, y, z, h]) finite numbers with h > 0, else
    the first problem."""
    text = '[x, y, h]' if dimensions == 2 else '[x, y, z, h]'
    if not isinstance(entries, list):
        return f'size_points must be a list of {text} entries, got {type(entries).__name__}'
    for index, entry in enumerate(entries):
        numbers = (isinstance(entry, (list, tuple)) and len(entry) == dimensions + 1
                   and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in entry))
        if not numbers or not all(math.isfinite(value) for value in entry) or not entry[-1] > 0:
            return f'size_points[{index}] must be {text} numbers in m with h > 0, got {entry!r}'
    return ''


def quantised_size_points(entries, h_min, dimensions=2) -> np.ndarray:
    """Float64 (n, dimensions + 1) [x, y(, z), h] of valid size_points entries, h rounded down to SIZE_POINT_STEPS per
    octave and floored at h_min."""
    points = np.asarray(entries, dtype=np.float64).reshape(-1, dimensions + 1).copy()
    if len(points):
        steps = np.floor(SIZE_POINT_STEPS * np.log2(points[:, -1]) + 1e-9) / SIZE_POINT_STEPS
        points[:, -1] = np.maximum(2.0 ** steps, h_min)
    return points


def size_point_function(size_points, growth, h_max):
    """Callable points (n, d) -> min(h_max, min over size points of h + (growth - 1) x distance) [m] of (n, d + 1)
    [x, y(, z), h] size points (d = 2 or 3): one KD-tree per distinct size (quantised_size_points keeps them few); the
    fem SizeField of the mesher and gmsh3d's Ball fields add the same sources."""
    from scipy.spatial import cKDTree
    size_points = np.asarray(size_points, dtype=np.float64)
    size_points = size_points.reshape(-1, size_points.shape[-1] if size_points.ndim == 2 else 3)
    dimensions = size_points.shape[1] - 1
    slope = float(growth) - 1.0
    trees = [(float(size), cKDTree(size_points[size_points[:, -1] == size, :-1]))
             for size in np.unique(size_points[:, -1]) if size < h_max]

    def sizes(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, dimensions)
        result = np.full(len(points), float(h_max))
        for size, tree in trees:
            distance, _ = tree.query(points, distance_upper_bound=(h_max - size) / slope)
            np.minimum(result, size + slope * distance, out=result)
        return result
    return sizes


def _absorbing_cells(boundary_config, faces=FACES):
    """{face: PML cells} of a boundary config over `faces`: cpml_npml on its cpml_sides (EM BoundaryConfig(3D);
    mirror ghost cells are no absorber: element meshes put mirrors on the window edge), npml_sides for a config
    without them."""
    if boundary_config is None:
        return {face: 0 for face in faces}
    cpml_sides = getattr(boundary_config, 'cpml_sides', None)
    if cpml_sides is None:
        return {face: int(boundary_config.npml_sides.get(face, 0)) for face in faces}
    return {face: int(boundary_config.cpml_npml) if cpml_sides.get(face) else 0 for face in faces}


def triangle_size_field(sizing, shapes, x_centres, y_centres) -> np.ndarray:
    """Target element size h [m] at the (nx, ny) points of 1D uniform centre arrays: min(h_max, object sizes inside /
    along / at their objects growing by (growth - 1) x distance outside, curvature_factor x radius near circle and
    ellipse outlines), floored at h_min (S7.5 2.2). The raster form of the estimate: an ellipse takes its smallest
    radius of curvature all round, where fem's mesher_size_field uses the local one (so the estimate counts more
    elements near flat ellipse parts: an upper bound). Region and outline sources of one size share one distance
    transform of their union (sized_unions)."""
    from scipy.ndimage import binary_erosion
    x_centres = np.asarray(x_centres, dtype=np.float64)
    y_centres = np.asarray(y_centres, dtype=np.float64)
    x, y = np.meshgrid(x_centres, y_centres, indexing='ij')
    sampling = (float(x_centres[1] - x_centres[0]) if len(x_centres) > 1 else 1.0,
                float(y_centres[1] - y_centres[0]) if len(y_centres) > 1 else 1.0)
    slope = sizing.growth - 1.0
    size = np.full(x.shape, sizing.h_max)

    def add_source(distance, source_size):
        np.minimum(size, source_size + slope * distance, out=size)

    unions = SizedUnions(x.shape)
    for name, object_size in sizing.object_sizes.items():
        shape = shapes.get(name)
        if shape is None:
            continue
        values = shape.values
        if shape.kind == 'point':
            add_source(np.hypot(x - values['x'], y - values['y']), object_size)
        elif shape.kind == 'line':
            add_source(_segment_distance(x, y, values), object_size)
        else:
            window, mask = rasterize_window(name, shapes, x_centres, y_centres, np)
            if mask.any():
                unions.add(object_size, window, mask)
            else:   # no pixel centre inside (thinner than a pixel): a source at the distance to its bounding box
                x0, y0, x1, y1 = bounding_box(name, shapes)
                add_source(np.hypot(np.maximum(np.maximum(x0 - x, x - x1), 0.0),
                                    np.maximum(np.maximum(y0 - y, y - y1), 0.0)), object_size)
    for name, shape in shapes.items():
        radius = _smallest_curvature_radius(shape)
        if radius is None or sizing.curvature_factor * radius >= sizing.h_max:
            continue
        window, mask = rasterize_window(name, shapes, x_centres, y_centres, np)
        if mask.any():
            unions.add(sizing.curvature_factor * radius, window, mask & ~binary_erosion(mask))
    unions.apply(add_source, sampling)
    if len(sizing.size_points):
        point_sizes = size_point_function(sizing.size_points, sizing.growth, sizing.h_max)
        np.minimum(size, point_sizes(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape), out=size)
    return np.maximum(size, sizing.h_min)


class SizedUnions:
    """Distance sources grouped by size (S2): min over the sources of one size of s + g d_i = s + g d(their union), so
    each size needs one distance transform of the OR of its rasters (region masks or outlines), not one per source."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.unions = {}

    def add(self, size, window, mask):
        """OR a cropped bool raster (window = one slice per axis) into its size's union; empty masks are skipped."""
        if not mask.any():
            return
        union = self.unions.get(float(size))
        if union is None:
            union = self.unions[float(size)] = np.zeros(self.shape, dtype=bool)
        union[tuple(window)] |= mask

    def apply(self, add_source, sampling):
        """add_source(distance to the union [m], size) once per size."""
        from scipy.ndimage import distance_transform_edt
        for size, union in self.unions.items():
            add_source(distance_transform_edt(~union, sampling=sampling), size)


def _segment_distance(x, y, values):
    """Distance of the points (x, y) to the segment (x0, y0)-(x1, y1) of a line object's values."""
    start_x, start_y, end_x, end_y = (values[key] for key in ('x0', 'y0', 'x1', 'y1'))
    along_x, along_y = end_x - start_x, end_y - start_y
    length_squared = along_x ** 2 + along_y ** 2
    fraction = (np.clip(((x - start_x) * along_x + (y - start_y) * along_y) / length_squared, 0.0, 1.0)
                if length_squared > 0.0 else np.zeros_like(x))
    return np.hypot(x - (start_x + fraction * along_x), y - (start_y + fraction * along_y))


def _smallest_curvature_radius(shape):
    """Smallest radius of curvature [m] of a circle / ellipse / rounded-rectangle outline (ellipse: b^2 / a, a >= b),
    a sphere or z cylinder surface; None otherwise."""
    if shape.kind in ('circle', 'sphere', 'cylinder_z'):
        return float(shape.values['r'])
    if shape.kind == 'rounded_rectangle' and shape.values['r'] > 0.0:
        return float(shape.values['r'])
    if shape.kind == 'ellipse':
        major, minor = sorted((float(shape.values['a']), float(shape.values['b'])), reverse=True)
        return minor ** 2 / major
    return None


def triangle_raster(size_x, size_y) -> tuple:
    """([x centres, y centres] float64 [m], (pixel width x, pixel width y) [m]) of the coarse window raster the
    triangle-count estimates integrate over (_TRIANGLE_RASTER pixels along the longer side)."""
    pixel = max(size_x, size_y) / _TRIANGLE_RASTER
    counts = [max(math.ceil(length / pixel - 1e-9), _TRIANGLE_MIN_RASTER) for length in (size_x, size_y)]
    centres = [(np.arange(count) + 0.5) * (length / count) for count, length in zip(counts, (size_x, size_y))]
    return centres, (size_x / counts[0], size_y / counts[1])


def triangle_counts(density, pixel_sides, pml_widths, pml_layers) -> dict:
    """2D mesher triangle counts from density 4 / (sqrt(3) h^2) per raster pixel: 'per_pixel' / 'window' = TRIANGLE_FILL
    x density x pixel area (array / sum), 'ring' = the structured absorbing ring (pml_layers across; along each ringed
    face 'along' = TRIANGLE_EDGE_FILL x integral ds / h of the window edge, + 'corner_columns' = pml_layers per ringed
    y face in the x strips, as fem.mesher.add_layer_ring)."""
    area = pixel_sides[0] * pixel_sides[1]
    inverse_size = np.sqrt(math.sqrt(3.0) * density / 4.0)   # 1 / h of the edge pixels (sub-sampled ones included)
    edge_integrals = {'x_lo': pixel_sides[1] * float(inverse_size[0].sum()),
                      'x_hi': pixel_sides[1] * float(inverse_size[-1].sum()),
                      'y_lo': pixel_sides[0] * float(inverse_size[:, 0].sum()),
                      'y_hi': pixel_sides[0] * float(inverse_size[:, -1].sum())}
    along = {face: math.ceil(TRIANGLE_EDGE_FILL * integral - 1e-9) for face, integral in edge_integrals.items()}
    ringed = [face for face in FACES if pml_widths[face] > 0.0]
    corner_columns = pml_layers * sum(face.startswith('y') for face in ringed)
    n_ring = sum(2 * pml_layers * (along[face] + (corner_columns if face.startswith('x') else 0)) for face in ringed)
    return {'per_pixel': TRIANGLE_FILL * area * density, 'window': TRIANGLE_FILL * float(np.sum(density)) * area,
            'ring': n_ring, 'along': along, 'corner_columns': corner_columns}


def _triangle_estimate(spec, mesh_spec, parameters, registry, shapes):
    """TriangleMeshEstimate: triangle_counts of triangle_size_field over the triangle_raster of the window (thin and
    point objects by their pixel means instead: _thin_source_density; mirror axes: the denser of a pixel and its image),
    window + structured absorbing ring."""
    sizing = triangle_sizing(spec, mesh_spec, parameters, registry, shapes)
    size_x, size_y = domain_size(spec, parameters)
    centres, pixel_sides = triangle_raster(size_x, size_y)
    thin = _thin_sources(sizing, shapes, pixel_sides)
    raster_sizing = replace(sizing, object_sizes={name: size for name, size in sizing.object_sizes.items()
                                                  if name not in thin})
    if len(sizing.size_points):
        density, smallest = _size_point_density(raster_sizing, shapes, centres, pixel_sides)
    else:
        size = triangle_size_field(raster_sizing, shapes, *centres)
        density, smallest = 4.0 / (math.sqrt(3.0) * size ** 2), float(size.min())
    if thin:
        density = np.maximum(density, _thin_source_density(sizing, thin, centres, pixel_sides))
        smallest = min(smallest, min(size for _, size in thin.values()))
    for axis in sizing.mirror_axes:   # the mirrored mesh: the smaller size of a point and its image (centred raster)
        density = np.maximum(density, np.flip(density, axis='xy'.index(axis)))
    widths = sizing.pml_widths
    counted = triangle_counts(density, pixel_sides, widths, sizing.pml_layers)
    n_elements = max(int(math.ceil(counted['window'])) + counted['ring'], 2)
    if n_elements > MAX_MESH_CELLS:
        raise ValueError(f"mesh '{mesh_spec.name}' too large: ~{n_elements:,} triangles; the limit is "
                         f'{MAX_MESH_CELLS:,}; lengths are in m, so check the domain size and element sizes')
    return TriangleMeshEstimate(
        name=mesh_spec.name, sizing=sizing, n_elements=n_elements, n_nodes=n_elements // 2 + 1,
        n_edges=3 * n_elements // 2 + 1, dx_min=smallest, dx_max=sizing.h_max,
        extent=(-widths['x_lo'], size_x + widths['x_hi'], -widths['y_lo'], size_y + widths['y_hi']),
        material_elements=_material_elements_2d(spec, shapes, centres, counted['per_pixel']))


def _material_elements_2d(spec, shapes, centres, elements) -> dict:
    """{material name: window triangles of the raster pixels it paints}: geometry.paint_material_index (the builder's
    order) of rasterize_window masks on the pixel centres; elements per pixel (the absorbing ring left out)."""
    return _painted_elements(spec, shapes, elements,
                             lambda name: rasterize_window(name, shapes, centres[0], centres[1], np))


def _painted_elements(spec, shapes, elements, window_mask) -> dict:
    """{material name: sum of `elements` over the raster cells it wins}: geometry.paint_material_index (the builder's
    order) of window_mask(name) -> (window, cropped bool mask) or None; names missing from shapes (outer faces, failed
    objects) and masks that fail paint nothing (validate reports them)."""
    selections = material_selections(spec, failed=())

    def object_mask(name):
        if name not in shapes:
            return None
        try:
            return window_mask(name)
        except (ValueError, KeyError, TypeError):
            return None

    index = paint_material_index(selections, object_mask, elements.shape, np)
    return {material.name: float(elements[index == number + 1].sum())
            for number, material in enumerate(spec.materials)}


def _thin_sources(sizing, shapes, pixel_sides) -> dict:
    """{name: ((x0, y0, x1, y1) bounding box [m], size [m])} of the refined objects a centre raster under-samples:
    points, axis-parallel lines and objects whose box is thinner than a pixel along x or y (size floored at h_min,
    capped at h_max)."""
    thin = {}
    for name, object_size in sizing.object_sizes.items():
        shape = shapes.get(name)
        if shape is None:
            continue
        box = bounding_box(name, shapes)
        if shape.kind == 'line':
            if box[2] > box[0] and box[3] > box[1]:   # a slanted line keeps its exact segment distance
                continue
        elif shape.kind != 'point' and box[2] - box[0] >= pixel_sides[0] and box[3] - box[1] >= pixel_sides[1]:
            continue
        thin[name] = (tuple(float(value) for value in box),
                      min(max(float(object_size), sizing.h_min), sizing.h_max))
    return thin


def _thin_source_density(sizing, thin, centres, pixel_sides) -> np.ndarray:
    """(nx, ny) density 4 / (sqrt(3) h^2) of the _thin_sources, each alone (h = min(h_max, size + (growth - 1) d), d the
    distance to its box), the max over them: sampled at the pixel centres, then a box thin along one axis takes the
    exact mean across each pixel along that axis (x its overlap with the long extent), and a point (or a box thin both
    ways) the mean over the disk of the pixel's area in the pixel that holds it; the part above 1 / h_max^2 x
    TRIANGLE_BAND_FILL. The band within a pixel of a film holds most of its elements (~ 1 / ((growth - 1) size) per side
    and unit length), which a centre sample misses."""
    x_centres, y_centres = (np.asarray(values, dtype=np.float64) for values in centres)
    pixel_x, pixel_y = pixel_sides
    slope, h_max = sizing.growth - 1.0, sizing.h_max
    density = np.zeros((x_centres.size, y_centres.size))
    for (x0, y0, x1, y1), size in thin.values():
        distance_x = np.maximum(np.maximum(x0 - x_centres, x_centres - x1), 0.0)
        distance_y = np.maximum(np.maximum(y0 - y_centres, y_centres - y1), 0.0)
        source = 1.0 / np.minimum(size + slope * np.hypot(distance_x[:, None], distance_y[None, :]), h_max) ** 2
        thin_x, thin_y = x1 - x0 < pixel_x, y1 - y0 < pixel_y
        if thin_x and thin_y:
            i = min(max(int(0.5 * (x0 + x1) / pixel_x), 0), x_centres.size - 1)
            j = min(max(int(0.5 * (y0 + y1) / pixel_y), 0), y_centres.size - 1)
            radius = math.sqrt(pixel_x * pixel_y / math.pi)
            source[i, j] = max(source[i, j], _disk_mean(size, slope, h_max, radius))
        elif thin_y:
            across = _band_mean(y0, y1, y_centres - 0.5 * pixel_y, y_centres + 0.5 * pixel_y, size, slope, h_max)
            along = np.clip(np.minimum(x_centres + 0.5 * pixel_x, x1) - np.maximum(x_centres - 0.5 * pixel_x, x0),
                            0.0, None) / pixel_x
            source = np.maximum(source, along[:, None] * across[None, :])
        else:
            across = _band_mean(x0, x1, x_centres - 0.5 * pixel_x, x_centres + 0.5 * pixel_x, size, slope, h_max)
            along = np.clip(np.minimum(y_centres + 0.5 * pixel_y, y1) - np.maximum(y_centres - 0.5 * pixel_y, y0),
                            0.0, None) / pixel_y
            source = np.maximum(source, across[:, None] * along[None, :])
        np.maximum(density, source, out=density)
    floor = 1.0 / h_max ** 2
    return 4.0 / math.sqrt(3.0) * (floor + TRIANGLE_BAND_FILL * np.maximum(density - floor, 0.0))


def _growth_integral(near, far, size, slope, h_max):
    """int_near^far dd / min(h_max, size + slope d)^2 for 0 <= near <= far (arrays) [1/m]."""
    cap = max(h_max - size, 0.0) / slope
    low, high = np.minimum(near, cap), np.minimum(far, cap)
    graded = (1.0 / (size + slope * low) - 1.0 / (size + slope * high)) / slope
    return graded + (np.maximum(far, cap) - np.maximum(near, cap)) / h_max ** 2


def _band_mean(low_edge, high_edge, pixel_low, pixel_high, size, slope, h_max):
    """Mean of 1 / h^2 over each pixel interval [pixel_low, pixel_high] (arrays) across a band source [low_edge,
    high_edge]: size inside, size + slope d outside (capped at h_max)."""
    inside = np.clip(np.minimum(pixel_high, high_edge) - np.maximum(pixel_low, low_edge), 0.0, None) / size ** 2
    below = _growth_integral(np.clip(low_edge - pixel_high, 0.0, None), np.clip(low_edge - pixel_low, 0.0, None),
                             size, slope, h_max)
    above = _growth_integral(np.clip(pixel_low - high_edge, 0.0, None), np.clip(pixel_high - high_edge, 0.0, None),
                             size, slope, h_max)
    return (inside + below + above) / (pixel_high - pixel_low)


def _disk_mean(size, slope, h_max, radius) -> float:
    """Mean of 1 / h^2 over the disk of `radius` around a point source (h = min(h_max, size + slope r))."""
    graded_radius = min(radius, max(h_max - size, 0.0) / slope)
    ratio = (size + slope * graded_radius) / size
    graded = (math.log(ratio) + 1.0 / ratio - 1.0) / slope ** 2
    flat = (radius ** 2 - graded_radius ** 2) / (2.0 * h_max ** 2)
    return 2.0 * (graded + flat) / radius ** 2


def _size_point_density(sizing, shapes, centres, pixel_sides) -> tuple:
    """(4 / (sqrt(3) h^2) per raster pixel, smallest h) with size points: the pixel-centre value, averaged over s x s
    sub-samples in pixels where the size points set h and may vary it by more than _SUBPIXEL_VARIATION inside the
    pixel (the object part of the field taken at the pixel centre), so adapted sizes below the raster are counted."""
    base = triangle_size_field(replace(sizing, size_points=np.zeros((0, 3))), shapes, *centres).ravel()
    x, y = np.meshgrid(*centres, indexing='ij')
    flat = np.column_stack([x.ravel(), y.ravel()])
    point_sizes = size_point_function(sizing.size_points, sizing.growth, sizing.h_max)
    at_centres = point_sizes(flat)
    size = np.maximum(np.minimum(base, at_centres), sizing.h_min)
    density = 4.0 / (math.sqrt(3.0) * size ** 2)
    smallest = float(size.min())
    finest = np.maximum(at_centres - (sizing.growth - 1.0) * 0.5 * math.hypot(*pixel_sides), sizing.h_min)
    flagged = np.flatnonzero((at_centres < base) & (at_centres > _SUBPIXEL_VARIATION * finest))
    if flagged.size:
        samples = np.clip(np.ceil(max(pixel_sides) / (_SUBPIXEL_SPACING * finest[flagged])), 2, _MAX_SUBSAMPLES)
        total = float(np.sum(samples ** 2))
        if total > _MAX_TOTAL_SUBSAMPLES:
            samples = np.maximum(np.floor(samples * math.sqrt(_MAX_TOTAL_SUBSAMPLES / total)), 2)
        samples = samples.astype(np.int64)
        for count in np.unique(samples):
            group = flagged[samples == count]
            fractions = (np.arange(count) + 0.5) / count - 0.5
            offset_x, offset_y = np.meshgrid(fractions * pixel_sides[0], fractions * pixel_sides[1], indexing='ij')
            offsets = np.column_stack([offset_x.ravel(), offset_y.ravel()])
            per_chunk = max(1, _SUBSAMPLE_CHUNK // len(offsets))
            for start in range(0, group.size, per_chunk):
                rows = group[start:start + per_chunk]
                sub = point_sizes((flat[rows][:, None, :] + offsets[None]).reshape(-1, 2)).reshape(len(rows), -1)
                sizes = np.maximum(np.minimum(sub, base[rows, None]), sizing.h_min)
                density[rows] = np.mean(4.0 / (math.sqrt(3.0) * sizes ** 2), axis=1)
                smallest = min(smallest, float(sizes.min()))
    return density.reshape(x.shape), smallest


# ---- tetrahedron meshes (igor3d_plan T4b, CPU only) ----
def tetrahedron_box(spec, parameters, sizing) -> tuple:
    """(low (3,), high (3,)) float64 corners [m] of the meshed box of a 'tetrahedra' mesh: the window plus the PML
    shell of sizing.pml_widths."""
    sizes = (*domain_size(spec, parameters), domain_size_z(spec, parameters))
    widths = sizing.pml_widths
    low = np.array([-widths[f'{axis}_lo'] for axis in 'xyz'], dtype=np.float64)
    high = np.array([size + widths[f'{axis}_hi'] for axis, size in zip('xyz', sizes)], dtype=np.float64)
    return low, high


def kuhn_axis_edges(length, h_max, width_lo, width_hi) -> np.ndarray:
    """Float64 node coordinates [m] along one axis of the uniform builtin tetrahedron mesh: the window [0, length] in
    equal cells <= h_max, each PML shell side (width = cells x h_max) in cells of h_max."""
    return shell_axis_edges(_equal_window_edges(length, h_max), h_max, width_lo, width_hi)


def _equal_window_edges(length, h_max) -> np.ndarray:
    """Float64 edges 0 .. length [m] in equal cells <= h_max."""
    return np.linspace(0.0, float(length), max(1, math.ceil(length / h_max - 1e-9)) + 1)


def shell_axis_edges(window_edges, h_max, width_lo, width_hi) -> np.ndarray:
    """Window edges (0 .. length) [m] extended by the PML shell sides (width = cells x h_max) in cells of h_max."""
    window_edges = np.asarray(window_edges, dtype=np.float64)
    below = window_edges[0] - h_max * np.arange(round(width_lo / h_max), 0, -1)
    above = window_edges[-1] + h_max * np.arange(1, round(width_hi / h_max) + 1)
    return np.concatenate([below, window_edges, above])


def builtin_axis_edges(spec, mesh_spec, parameters, shapes, sizing) -> list:
    """[x, y, z] float64 node coordinates [m] of the builtin tetrahedron mesh (its Kuhn split): graded_window_edges plus
    the PML shell cells of h_max (sizing.pml_widths)."""
    window = graded_window_edges(spec, mesh_spec, parameters, shapes, sizing)
    return [shell_axis_edges(window[axis], sizing.h_max, sizing.pml_widths[f'{axis}_lo'],
                             sizing.pml_widths[f'{axis}_hi']) for axis in 'xyz']


def sizing_axis_edges(sizing, window_sizes) -> list:
    """[x, y, z] node coordinates of a builtin tetrahedron sizing: its axis_edges, else (a sizing made by hand) the
    uniform Kuhn edges of the window sizes (size_x, size_y, size_z)."""
    if sizing.axis_edges:
        return [np.asarray(edges, dtype=np.float64) for edges in sizing.axis_edges]
    widths = sizing.pml_widths
    return [kuhn_axis_edges(float(size), sizing.h_max, widths[f'{axis}_lo'], widths[f'{axis}_hi'])
            for axis, size in zip('xyz', window_sizes)]


def graded_window_edges(spec, mesh_spec, parameters, shapes, sizing, axes='xyz', keep_uniform=True) -> dict:
    """{axis: float64 window edges 0 .. length [m]} of builtin tetrahedra at base h_max: the tensor nonuniform planner
    (refinements + auto ones with dx_x / dx_y / dx_z and margins, MeshAxisSpec plans, max_ratio, outer grading; not in
    uniform mode), sizing.face_sizes as bands across their face (every mode) and every domain object's flat faces
    snapped; keep_uniform and nothing finer than h_max anywhere: equal cells <= h_max, no snaps. ValueError above the
    cell-count guard."""
    mesh = mesh_values(mesh_spec, parameters)
    h_max = float(sizing.h_max)
    size_x, size_y = domain_size(spec, parameters)
    lengths = {'x': float(size_x), 'y': float(size_y), 'z': float(domain_size_z(spec, parameters))}
    refinements, axis_plans = [], {axis: _AxisPlan() for axis in lengths}
    if mesh['mode'] != 'uniform' and sizing.auto_rule != 'uniform_flow':
        candidates = ([_Refinement(name, float(size), None) for name, size in sizing.auto_refinements]
                      + _explicit_refinements(spec, mesh_spec, parameters, shapes))
        refinements = [refinement for refinement in candidates
                       if any(refinement.refines(axis, h_max) for axis in lengths)]
        axis_plans = _axis_plans(mesh_spec, parameters, lengths)
        for axis_plan in axis_plans.values():
            axis_plan.bands = [band for band in axis_plan.bands if band[2] < h_max * (1.0 - _REFINEMENT_RTOL)]
    if keep_uniform and not (refinements or sizing.face_sizes or any(axis_plan.bands or axis_plan.edges is not None
                                                                     for axis_plan in axis_plans.values())):
        _check_cell_count(*(max(1, math.ceil(lengths[axis] / h_max - 1e-9)) for axis in axes))
        return {axis: _equal_window_edges(lengths[axis], h_max) for axis in axes}
    mesh_config, intervals = _graded_config(lengths, h_max, float(sizing.growth), mesh, axis_plans, refinements,
                                            shapes)
    for face in sizing.face_sizes:   # the band [c - distance, c + distance] across the face plane, the plane snapped
        axis = 'xyz'[face.axis]
        lo, hi = face.coordinate - face.distance, face.coordinate + face.distance
        mesh_config.add_refinement_band(axis, lo, hi, float(face.size), (face.coordinate,))
        intervals[axis].append((lo, hi, float(face.size)))
    for name, shape in shapes.items():   # composite inputs and array copies included: every flat face conforms
        if GEOMETRY_KINDS[shape.kind].entity_kind == 'domain':
            for axis, positions in axis_parallel_faces(name, shapes).items():
                mesh_config.add_snap_edges(axis, positions, h_max)
    _check_cell_count(*(len(axis_plans[axis].edges) - 1 if axis_plans[axis].edges is not None
                        else _snapped_axis_cell_bound(mesh_config, axis, lengths[axis], h_max, intervals[axis], mesh)
                        for axis in axes))
    return {axis: mesh_config.window_edges(axis) for axis in axes}


def tetrahedron_element_density(sizing, shapes, centres, size_z, solids=(), surfaces=()) -> tuple:
    """(mean 1 / h^3 per voxel (nx, ny, nz), smallest h [m]) of the target element size on the raster of three 1D
    uniform centre arrays: h = min(h_max, object sizes in / along / at their objects growing by (growth - 1) x distance,
    curvature_factor x radius near sphere, cylinder, circle and ellipse surfaces, the edge_size of gmsh3d solids at
    their edges, sizing.face_sizes within their band, sizing.size_points [x, y, z, h], the gmsh3d SizedRegion surfaces
    (SizedRegion.surface_distance, the analytic twin of the gmsh sized_surfaces): the twins of gmsh3d's fields),
    floored at h_min. Objects of one size share one distance transform of their union (min over them of
    s + g d_i = s + g d(union)); each voxel averages h^-3 of its winning source along that source's signed distance over
    +- half a voxel (the centre value overcounts objects thinner than a voxel)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    centres = [np.asarray(axis_centres, dtype=np.float64) for axis_centres in centres]
    shape = tuple(len(axis_centres) for axis_centres in centres)
    sampling = tuple(float(axis_centres[1] - axis_centres[0]) if len(axis_centres) > 1 else 1.0
                     for axis_centres in centres)
    half_voxel = 0.5 * min(step for step, count in zip(sampling, shape) if count > 1) if max(shape) > 1 else 0.0
    slope = sizing.growth - 1.0
    centre_size = np.full(shape, sizing.h_max)   # h at the voxel centre (of the winning source)
    source_size = np.full(shape, sizing.h_max)   # the winning source's own size s
    signed = np.zeros(shape)                     # its signed distance [m] (inside < 0)
    grids = []

    def coordinates():
        if not grids:
            grids.extend(np.meshgrid(*centres, indexing='ij'))
        return grids

    def add_source(distance, size):
        at_centre = size + slope * np.maximum(distance, 0.0)
        wins = at_centre < centre_size
        centre_size[wins] = at_centre[wins]
        source_size[wins] = size
        signed[wins] = distance[wins]

    def union_mask(names):
        """(bool raster of the objects' union, names covering no voxel centre)."""
        union, tiny = np.zeros(shape, dtype=bool), []
        for name in names:
            block = _object_block(name, shapes, centres)
            mask = None if block is None else _block_mask(name, shapes, centres, block)
            if mask is None or not mask.any():
                tiny.append(name)
            else:
                union[block] |= mask
        return union, tiny

    groups = {}
    for name, object_size in sizing.object_sizes.items():
        shape_of = shapes.get(name)
        if shape_of is None:
            continue
        values = shape_of.values
        x, y, z = coordinates() if shape_of.kind in ('point', 'line') else (None, None, None)
        if shape_of.kind == 'point':
            add_source(np.sqrt((x - values['x']) ** 2 + (y - values['y']) ** 2 + (z - point_z(values, size_z)) ** 2),
                       object_size)
        elif shape_of.kind == 'line':
            add_source(np.hypot(_segment_distance(x, y, values), z - point_z(values, size_z)), object_size)
        else:
            groups.setdefault(float(object_size), []).append(name)
    for object_size, names in groups.items():
        union, tiny = union_mask(names)
        if union.all():   # no voxel outside: no distance transform
            add_source(np.full(shape, -np.inf), object_size)
        elif union.any():   # voxel-centre distances shifted by half a voxel to the surface between them
            outside = distance_transform_edt(~union, sampling=sampling) - half_voxel
            depth = distance_transform_edt(union, sampling=sampling) - half_voxel
            add_source(np.where(union, -depth, outside), object_size)
            del outside, depth
        for name in tiny:   # smaller than a voxel: a point source at its box centre
            x0, y0, x1, y1 = bounding_box(name, shapes)
            z_lo, z_hi = (min(max(value, 0.0), size_z) for value in z_extent(name, shapes))
            x, y, z = coordinates()
            add_source(np.sqrt((x - 0.5 * (x0 + x1)) ** 2 + (y - 0.5 * (y0 + y1)) ** 2 + (z - 0.5 * (z_lo + z_hi)) ** 2),
                       object_size)
    curved_surfaces = SizedUnions(shape)
    for name, shape_of in shapes.items():
        radius = _smallest_curvature_radius(shape_of)
        if radius is None or sizing.curvature_factor * radius >= sizing.h_max:
            continue
        block = _object_block(name, shapes, centres)
        mask = None if block is None else _block_mask(name, shapes, centres, block)
        if mask is not None and mask.any():   # its surface voxels, one distance transform per curvature size
            curved_surfaces.add(sizing.curvature_factor * radius, block, mask & ~binary_erosion(mask))
    curved_surfaces.apply(add_source, sampling)
    for solid in solids:
        if 0.0 < solid.edge_size < sizing.h_max:
            add_source(solid.edge_distance(np.stack(coordinates(), axis=-1)), solid.edge_size)
    for surface in surfaces:   # unsigned: the size holds on the surface, growing to both sides
        if 0.0 < surface.size < sizing.h_max:
            add_source(surface.surface_distance(np.stack(coordinates(), axis=-1)), surface.size)
    if sizing.face_sizes or len(sizing.size_points):   # the gmsh MathEval / Ball fields of gmsh3d (G1 twin)
        points = np.stack(coordinates(), axis=-1)
        for face in sizing.face_sizes:   # signed against the band: inside it the face size holds
            add_source(face.face_distance(points) - face.distance, face.size)
        if len(sizing.size_points):
            from scipy.spatial import cKDTree
            size_points = np.asarray(sizing.size_points, dtype=np.float64).reshape(-1, 4)
            flat = points.reshape(-1, 3)
            for size in np.unique(size_points[:, 3]):
                if size >= sizing.h_max:
                    continue
                distance, _ = cKDTree(size_points[size_points[:, 3] == size, :3]).query(
                    flat, distance_upper_bound=(sizing.h_max - size) / slope)
                add_source(distance.reshape(shape), float(size))
        del points
    density = np.zeros(shape)
    smallest = sizing.h_max
    for fraction in (np.arange(_TETRAHEDRON_SUBVOXEL_POINTS) + 0.5) / _TETRAHEDRON_SUBVOXEL_POINTS:
        size = np.clip(source_size + slope * np.maximum(signed + (2.0 * fraction - 1.0) * half_voxel, 0.0),
                       sizing.h_min, sizing.h_max)
        density += size ** -3.0
        smallest = min(smallest, float(size.min()))
    return density / _TETRAHEDRON_SUBVOXEL_POINTS, smallest


def _object_block(name, shapes, centres):
    """Raster index slices (x, y, z) of a geometry object's bounding box on 1D sorted centre arrays (open ends
    unbounded), None when it holds no centre."""
    x0, y0, x1, y1 = bounding_box(name, shapes)
    z0, z1 = z_extent(name, shapes)
    block = []
    for axis_centres, low, high in zip(centres, (x0, y0, z0), (x1, y1, z1)):
        start = int(np.searchsorted(axis_centres, low, side='left'))
        stop = int(np.searchsorted(axis_centres, high, side='right'))
        if stop <= start:
            return None
        block.append(slice(start, stop))
    return tuple(block)


def _block_mask(name, shapes, centres, block):
    """Bool raster block of geometry.contains_points_3d at the centres of an _object_block."""
    grids = np.meshgrid(*(axis_centres[part] for axis_centres, part in zip(centres, block)), indexing='ij')
    return contains_points_3d(name, shapes, *(grid.ravel() for grid in grids)).reshape(grids[0].shape)


def _material_elements(spec, shapes, centres, elements, window_low, window_high) -> dict:
    """{material name: elements of the raster voxels it paints}: geometry.paint_material_index (the builder's order)
    on the voxel centres clamped into the window (the inset element centroids of the build); elements per voxel."""
    clamped = [np.clip(axis_centres, low, high) for axis_centres, low, high in zip(centres, window_low, window_high)]

    def window_mask(name):
        block = _object_block(name, shapes, clamped)
        return None if block is None else (block, _block_mask(name, shapes, clamped, block))
    return _painted_elements(spec, shapes, elements, window_mask)


def _tetrahedron_estimate(spec, mesh_spec, parameters, registry, shapes):
    """TetrahedronMeshEstimate: the builtin Kuhn split counted exactly (6 per cell of the sizing's axis_edges); gmsh:
    TETRAHEDRON_FILL x integral 6 sqrt(2) / h^3 of tetrahedron_element_density over a voxel raster of the box (voxel =
    the finest object / edge size within the _TETRAHEDRON_MAX_VOXELS and _TETRAHEDRON_RASTER bounds). Both: the
    elements each material paints on the cell / voxel centres (material_elements)."""
    sizing = tetrahedron_sizing(spec, mesh_spec, parameters, registry, shapes)
    low, high = tetrahedron_box(spec, parameters, sizing)
    sides = high - low
    size_z = domain_size_z(spec, parameters)
    window_low = np.zeros(3)
    window_high = np.array([*domain_size(spec, parameters), size_z], dtype=np.float64)
    if sizing.sweep_axis:   # swept prism layers (plan D5): footprint triangles x 3 x layers
        from .swept_meshing import swept_estimate
        n_elements, n_nodes, n_edges, smallest, centres, elements = swept_estimate(spec, parameters, shapes, sizing)
    elif sizing.mesher == 'builtin':
        edges = sizing_axis_edges(sizing, window_high)
        cells =[len(axis_edges) - 1 for axis_edges in edges]
        nodes = [count + 1 for count in cells]
        n_elements = 6 * math.prod(cells)
        n_nodes = math.prod(nodes)
        n_axis_edges = sum(cells[k] * math.prod(nodes) // nodes[k] for k in range(3))
        n_face_diagonals = sum(nodes[k] * math.prod(cells) // cells[k] for k in range(3))
        n_edges = n_axis_edges + n_face_diagonals + math.prod(cells)   # + one body diagonal per cell
        smallest = float(min(np.diff(coordinates).min() for coordinates in edges))
        centres = [0.5 * (axis_edges[1:] + axis_edges[:-1]) for axis_edges in edges]
        elements = np.full(tuple(cells), 6.0)
    else:
        coarsest = float(sides.max()) / _TETRAHEDRON_RASTER
        finest = min([*sizing.object_sizes.values(), *sizing.edge_sizes.values(), *sizing.surface_sizes.values(),
                      *(face.size for face in sizing.face_sizes),
                      *(np.asarray(sizing.size_points, dtype=np.float64).reshape(-1, 4)[:, 3].tolist()
                        if len(sizing.size_points) else ()), sizing.h_max])
        budget = float(np.prod(sides) / _TETRAHEDRON_MAX_VOXELS) ** (1.0 / 3.0)
        voxel = min(coarsest, max(finest, budget))
        counts = [max(math.ceil(float(side) / voxel - 1e-9), _TRIANGLE_MIN_RASTER) for side in sides]
        centres = [low[k] + (np.arange(counts[k]) + 0.5) * (sides[k] / counts[k]) for k in range(3)]
        solids, surfaces = (), ()
        if sizing.edge_sizes:
            from .tetrahedral_meshing import model_solids
            solids = model_solids(shapes, sizing, low, high)
        if sizing.surface_sizes:
            from .tetrahedral_meshing import sized_surfaces
            surfaces = sized_surfaces(shapes, sizing, low, high)
        density, smallest = tetrahedron_element_density(sizing, shapes, centres, size_z, solids, surfaces)
        elements = TETRAHEDRON_FILL * 6.0 * math.sqrt(2.0) * float(np.prod(sides / np.array(counts))) * density
        n_elements = max(int(math.ceil(float(elements.sum()))), 6)
        n_nodes = int(n_elements / TETRAHEDRA_PER_NODE) + 4
        n_edges = n_nodes + n_elements
    if n_elements > MAX_MESH_CELLS:
        raise ValueError(f"mesh '{mesh_spec.name}' too large: ~{n_elements:,} tetrahedra; the limit is "
                         f'{MAX_MESH_CELLS:,}; lengths are in m, so check the domain size and element sizes')
    return TetrahedronMeshEstimate(
        name=mesh_spec.name, sizing=sizing, n_elements=n_elements, n_nodes=n_nodes, n_edges=n_edges,
        n_faces=1 - n_nodes + n_edges + n_elements, dx_min=smallest, dx_max=sizing.h_max,
        extent=(float(low[0]), float(high[0]), float(low[1]), float(high[1]), float(low[2]), float(high[2])),
        material_elements=_material_elements(spec, shapes, centres, elements, window_low, window_high))


# ---- memory estimate (E8, CPU only) ----
@dataclass(frozen=True)
class MaterialRegionInfo:
    """One spec material for the memory estimators: full-grid cells of its selection's bounding box (3D: x, y and z;
    'all': every cell, PML included; an upper bound of the effective region) and its dispersion poles per model."""
    name: str
    box_cells: int
    n_drude: int
    n_lorentz: int
    box_bounds: tuple = ()      # per axis (first, last) full-grid cell of that box; () when nothing resolves


@dataclass(frozen=True)
class MemoryContext:
    """Third argument of PhysicsType.memory_estimate: the model around the physics."""
    spec: object
    physics_spec: object
    parameters: dict
    registry: object
    material_regions: tuple     # MaterialRegionInfo per spec material on the physics' mesh (poles only when a type on
                                # that mesh uses dispersion)
    solvers: dict = field(default_factory=dict)   # schema.resolved_solvers of the physics: {role: ResolvedSolver}
    mesh: object = None         # the estimate of the physics' own mesh (S7.2; the estimator's mesh argument)

    def material_sets(self, property_name) -> bool:
        """True when a material (library data merged) sets property_name to anything but None or a literal 0."""
        for material in self.spec.materials:
            try:
                properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
            except (ValueError, TypeError):   # reported by validate
                continue
            value = properties.get(property_name)
            if value is not None and not (isinstance(value, (int, float)) and value == 0):
                return True
        return False


def estimate_memory(spec, parameters, registry=None, mesh_estimate=None, by_mesh=None,
                    by_physics=None, shapes=None) -> MemoryEstimate:
    """CPU MemoryEstimate of building and running the spec, summed over its used meshes (each: MaterialGrid arrays of
    the types on it, build caches, state-dependent material updates, coupling arrays into its physics, every physics'
    and enabled feature's memory_estimate on it (none: 0 bytes); labels prefixed "mesh '<name>'" when several) plus
    the grid transfers of cross-mesh couplings. mesh_estimate: {name: estimate}, or one estimate of the default mesh;
    missing ones are estimated. by_mesh: a dict that receives {name: MemoryEstimate} per mesh (transfers excluded);
    by_physics: {physics name: MemoryEstimate} of the physics' own and its features' estimates (the time recipes'
    array traffic). shapes: the caller's resolve_geometry (None = resolved here). MESH_ESTIMATE_ERRORS on model errors
    (validate reports them)."""
    registry = registry if registry is not None else get_registry()
    shapes = resolve_geometry(spec, parameters, errors={}) if shapes is None else shapes
    meshes = _mesh_estimate_map(spec, parameters, registry, mesh_estimate, shapes)
    several = len(meshes) > 1
    parts, build_cache_bytes = [], 0
    for name, mesh in meshes.items():
        on_mesh = spec.physics_on(name) if spec.physics else []
        part, cache_bytes = _mesh_memory(spec, parameters, registry, shapes, mesh, name, on_mesh, several, by_physics)
        if by_mesh is not None:
            by_mesh[name] = part
        parts.append(part.labelled(f"mesh '{name}'") if several else part)
        build_cache_bytes += cache_bytes   # the build keeps every mesh's caches until release_caches
    transfers = mesh_transfers(spec, registry, meshes)
    parts.append(MemoryEstimate.from_items(
        (f"grid transfer '{record['coupling']}' ({record['source_mesh']} -> {record['target_mesh']}, "
         f"{record['kind']})", record['bytes']) for record in transfers))
    return combine(parts, build_cache_bytes=build_cache_bytes)


def backend_memory(spec, estimate) -> MemoryEstimate:
    """The estimate as the run backend holds it: on the cpu backend the device arrays live in RAM (host += gpu_peak,
    no device part, so no GPU budget applies; cuDSS is gpu-only, its external items do not occur); unchanged on the
    gpu backend."""
    if solver_backend(spec) != 'cpu':
        return estimate
    return MemoryEstimate(host=estimate.host + estimate.gpu_peak,
                          host_items=estimate.host_items + estimate.items + estimate.run_items
                          + tuple((label, size) for _, label, size in estimate.transient_items))


def _mesh_estimate_map(spec, parameters, registry, mesh_estimate, shapes=None):
    """{name: estimate} of every used mesh from a dict / one estimate (its name) / None, estimating the missing."""
    given = (dict(mesh_estimate) if isinstance(mesh_estimate, dict)
             else {} if mesh_estimate is None else {getattr(mesh_estimate, 'name', default_mesh_name(spec)):
                                                    mesh_estimate})
    return {mesh_spec.name: given[mesh_spec.name] if given.get(mesh_spec.name) is not None
            else estimate_mesh(spec, parameters, registry, mesh_spec.name, shapes)
            for mesh_spec in spec.used_meshes()}


def _mesh_memory(spec, parameters, registry, shapes, mesh, mesh_name, on_mesh, several, by_physics=None):
    """(MemoryEstimate, build-cache bytes) of one mesh and the physics on it (on_mesh; its material arrays those of
    mesh_material_types); physics and features whose type does not run on the mesh's kind are left out (validate
    refuses them); one mesh: identical to the whole model. by_physics receives each physics' own parts combined."""
    names = {physics_spec.name for physics_spec in on_mesh}
    present_types = mesh_material_types(spec, registry, mesh_name)
    view = _spec_with_physics(spec, on_mesh) if several else spec
    regions = _material_region_infos(view, shapes, mesh, present_types)
    parts = [_model_arrays_estimate(spec, registry, mesh, present_types, None if not several else names)]
    if mesh.kind == 'tensor':   # subpixel smoothing and conformal boundaries are moot on triangles
        parts += [subpixel_memory_estimate(view, registry, shapes, mesh),
                  conformal_memory_estimate(view, registry, shapes, mesh)]
    for physics_spec in on_mesh:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is None or mesh.kind not in physics_type.grids:
            continue
        first_part = len(parts)
        if physics_type.memory_estimate is not None:
            settings = physics_settings(physics_type, physics_spec, parameters)
            solvers = resolved_solvers(spec, registry, physics_spec, spec.study.type, parameters)
            context = MemoryContext(spec, physics_spec, parameters, registry, regions, solvers, mesh=mesh)
            parts.append(physics_type.memory_estimate(settings, mesh, context).labelled(
                f"physics '{physics_spec.name}'"))
        for feature_spec in physics_spec.features:
            if not feature_spec.enabled:
                continue
            feature_type = physics_type.feature_type(feature_spec.type)   # KeyError: unknown feature type
            if feature_type.memory_estimate is None or mesh.kind not in feature_type.grids:
                continue
            if not isinstance(feature_spec.params, dict):
                raise ValueError(f"feature '{feature_spec.name}': params must be a dict")
            values = resolve_values(feature_type.params, feature_spec.params, parameters)
            n_cells = (_selection_cells(spec, feature_spec.selection, shapes, mesh)
                       if feature_type.selection_kind is not None else 0)
            parts.append(feature_type.memory_estimate(values, n_cells, mesh).labelled(
                f"feature '{physics_spec.name}.{feature_spec.name}'"))
        if by_physics is not None:
            by_physics[physics_spec.name] = combine(parts[first_part:])
    return combine(parts), _build_cache_bytes(mesh, regions)


def _spec_with_physics(spec, physics_specs):
    """Shallow copy of spec holding only physics_specs and the couplings among them (per-mesh estimator views)."""
    view = copy.copy(spec)
    names = {physics_spec.name for physics_spec in physics_specs}
    view.physics = list(physics_specs)
    view.couplings = [coupling for coupling in spec.couplings if coupling.source in names and coupling.target in names]
    return view


def mesh_transfers(spec, registry, meshes) -> list:
    """[{coupling, source_mesh, target_mesh, kind, nnz, bytes}] of the enabled couplings whose source and target run
    on different meshes with a CouplingType.mesh_transfer ('field': interpolate; 'value': its value_kind); meshes =
    {name: estimate} (couplings touching a missing one are skipped). Tensor pairs: core grid_transfer_estimate; a
    triangle mesh: _TRIANGLE_TRANSFER_ENTRIES per target cell / element (calibrated by T2)."""
    records = []
    for coupling in spec.couplings:
        if not coupling.enabled:
            continue
        try:
            coupling_type = registry.coupling_type(coupling.type)
            source_mesh, target_mesh = (spec.mesh_of(coupling.source).name, spec.mesh_of(coupling.target).name)
        except KeyError:   # reported by validate
            continue
        if source_mesh == target_mesh or coupling_type.mesh_transfer is None:
            continue
        source, target = meshes.get(source_mesh), meshes.get(target_mesh)
        if source is None or target is None:
            continue
        kind = 'interpolate' if coupling_type.mesh_transfer == 'field' else coupling_type.value_kind
        if source.kind == 'tensor' and target.kind == 'tensor':
            axes = ('x_edges', 'y_edges') if source.z_edges is None else ('x_edges', 'y_edges', 'z_edges')
            nnz, size = grid_transfer_estimate([getattr(source, axis) for axis in axes],
                                               [getattr(target, axis) for axis in axes], kind)
        else:
            nnz = _TRIANGLE_TRANSFER_ENTRIES * int(target.n_cells)
            size = nnz * (2 * FLOAT32_BYTES) + (int(target.n_cells) + 1) * FLOAT32_BYTES
        records.append({'coupling': coupling.name, 'source_mesh': source_mesh, 'target_mesh': target_mesh,
                        'kind': kind, 'nnz': int(nnz), 'bytes': int(size)})
    return records


def _material_region_infos(spec, shapes, mesh, present_types):
    """MaterialRegionInfo per spec material (selection bounding box, pole counts)."""
    uses_dispersion = any(physics_type.uses_dispersion for physics_type in present_types)
    regions = []
    for material in spec.materials:
        models = []
        if uses_dispersion:
            try:
                _, dispersion = effective_material_data(material.library, material.properties, material.dispersion)
            except (ValueError, TypeError):   # reported by validate
                dispersion = []
            models = [pole.get('model') for pole in dispersion if isinstance(pole, dict)]
        bounds = _selection_bounds(spec, material.selection, shapes, mesh) if mesh.kind == 'tensor' else ()
        regions.append(MaterialRegionInfo(material.name, _selection_cells(spec, material.selection, shapes, mesh),
                                          models.count('drude'), models.count('lorentz'), bounds))
    return tuple(regions)


def _selection_cells(spec, entities, shapes, mesh):
    """Full-grid cells of the union bounding box of the selected objects (_selection_bounds; 0 when no object
    resolves); a triangle mesh: its elements times the box's share of the meshed area."""
    if mesh.kind == 'tensor':
        return _bounds_cells(_selection_bounds(spec, entities, shapes, mesh))
    try:
        expanded = expand_entities(spec, entities)
    except (ValueError, KeyError, TypeError):   # reported by validate
        return 0
    if 'all' in expanded:
        return int(mesh.n_elements)
    names = [name for name in expanded if name in shapes]
    if not names:
        return 0
    boxes = [bounding_box(name, shapes) for name in names]
    extent = mesh.extent   # (x0, x1, y0, y1(, z0, z1))
    ranges = [(min(box[0] for box in boxes), max(box[2] for box in boxes)),
              (min(box[1] for box in boxes), max(box[3] for box in boxes))]
    if len(extent) == 6:   # tetrahedra: the union z extent too
        z_extents = [z_extent(name, shapes) for name in names]
        ranges.append((min(lo for lo, _ in z_extents), max(hi for _, hi in z_extents)))
    share = 1.0
    for axis, (lo, hi) in enumerate(ranges):
        mesh_lo, mesh_hi = extent[2 * axis], extent[2 * axis + 1]
        share *= max(min(hi, mesh_hi) - max(lo, mesh_lo), 0.0) / (mesh_hi - mesh_lo)
    return int(math.ceil(mesh.n_elements * min(share, 1.0)))


def _bounds_cells(bounds):
    cells = 1
    for first, last in bounds:
        cells *= last - first + 1
    return cells if bounds else 0


def _selection_bounds(spec, entities, shapes, mesh):
    """Per axis (first, last) full-grid cells of the union bounding box of the selected objects, each edge in its
    containing cell; 3D: also the z cells of the union z extent (open prism ends clamped to the grid); polar /
    spherical: the logical cells of DomainMap.logical_bounds of each object's box ('all': every cell, PML included;
    () when no object resolves)."""
    try:
        expanded = expand_entities(spec, entities)
    except (ValueError, KeyError, TypeError):   # reported by validate
        return ()
    counts = (mesh.nx, mesh.ny) if mesh.z_edges is None else (mesh.nx, mesh.ny, mesh.nz)
    if 'all' in expanded:
        return tuple((0, int(count) - 1) for count in counts)
    names = [name for name in expanded if name in shapes]
    boxes = [bounding_box(name, shapes) for name in names]
    if not boxes:
        return ()
    if mesh.domain_map is not None:   # (x_lo, x_hi, y_lo, y_hi) logical ranges; edges logical too
        ranges = [mesh.domain_map.logical_bounds(box) for box in boxes]
        boxes = [(x_lo, y_lo, x_hi, y_hi) for x_lo, x_hi, y_lo, y_hi in ranges]
    spans = [(mesh.x_edges, min(box[0] for box in boxes), max(box[2] for box in boxes)),
             (mesh.y_edges, min(box[1] for box in boxes), max(box[3] for box in boxes))]
    if mesh.z_edges is not None:
        z_extents = [z_extent(name, shapes) for name in names]
        z_bounds = (float(mesh.z_edges[0]), float(mesh.z_edges[-1]))
        spans.append((mesh.z_edges, min(max(min(lo for lo, _ in z_extents), z_bounds[0]), z_bounds[1]),
                      max(min(max(hi for _, hi in z_extents), z_bounds[1]), z_bounds[0])))
    return tuple((containing_cell(lo, edges, 0, len(edges) - 2), containing_cell(hi, edges, 0, len(edges) - 2))
                 for edges, lo, hi in spans)


def optional_material_arrays(spec, physics_types) -> dict:
    """{array: default} of the physics types' optional_material_schema arrays a spec material (library merged) triggers
    by setting one of their trigger properties (the builder allocates exactly these)."""
    candidates = {array: entry for physics_type in physics_types
                  for array, entry in physics_type.optional_material_schema.items()}
    if not candidates:
        return {}
    set_names = set()
    for material in spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):   # reported by validate
            continue
        set_names.update(name for name, value in properties.items() if value is not None)
    return {array: default for array, (default, triggers) in candidates.items() if not set_names.isdisjoint(triggers)}


def _model_arrays_estimate(spec, registry, mesh, present_types, names=None):
    """Model-level device arrays of one mesh: the present types' MaterialGrid arrays (material_schema union + the
    triggered optional arrays), value + mask per state-dependent material property (DynamicMaterialUpdate), one
    transfer array per enabled coupling into its physics; run items: fixed-point and solution-feedback copies, the
    stored-cut plane fetches of 3D snapshots. names: the physics on the mesh (None: every physics, one mesh)."""
    n_cells = int(mesh.n_cells)
    arrays = list(dict.fromkeys(name for physics_type in present_types for name in physics_type.material_schema))
    arrays += [name for name in optional_material_arrays(spec, present_types) if name not in arrays]
    n_all_updates = n_updates = _state_dependent_property_count(spec, registry)
    if names is not None:   # several meshes: the updates of properties a type on this mesh reads
        n_updates = _state_dependent_property_count(
            spec, registry, {param.name for physics_type in present_types for param in physics_type.material_properties})
    n_couplings = sum(1 for coupling in spec.couplings
                      if coupling.enabled and (names is None or coupling.target in names))
    n_monitored = _fixed_point_physics_count(spec, registry, n_all_updates, names)
    n_feedback = _feedback_copy_count(spec, registry, n_all_updates, names)
    cut_cells = 0   # 3D stored cuts: the largest full-grid plane of their axes (snapshot fetch temporaries)
    if mesh.z_edges is not None:
        counts = {'x': mesh.nx, 'y': mesh.ny, 'z': mesh.nz}
        cut_axes = {cut[0] for cut in spec.study.stored_cuts
                    if isinstance(cut, (list, tuple)) and cut and cut[0] in counts}
        cut_cells = max((n_cells // int(counts[axis]) for axis in cut_axes), default=0)
    return MemoryEstimate.from_items((
        (f'material arrays ({len(arrays)} x float32)', FLOAT32_BYTES * len(arrays) * n_cells),
        (f'state-dependent material updates ({n_updates})', (FLOAT32_BYTES + BOOL_BYTES) * n_updates * n_cells),
        (f'coupling transfer arrays ({n_couplings}, approximate)', FLOAT32_BYTES * n_couplings * n_cells),
    ), run_items=(
        (f'fixed-point field copies ({n_monitored} x float32, upper bound)', FLOAT32_BYTES * n_monitored * n_cells),
        (f'solution-feedback field copies ({n_feedback} x float32, upper bound)',
         FLOAT32_BYTES * n_feedback * n_cells),
    ), transient_items=(   # freed after each snapshot's plane fetch
        ('snapshot cut fetch', 'stored-cut snapshot planes (4 x float32 plane, approximate)',
         FLOAT32_BYTES * 4 * cut_cells),
    )).labelled('model')


def _fixed_point_physics_count(spec, registry, n_updates, names=None):
    """Upper bound of the physics (of names; None: all) whose primary field the stationary fixed point keeps a copy of
    (study._FixedPointMonitor): in a study with a coupling loop solve (a 'tolerance' setting), every physics a
    coupling touches, every physics with a state-dependent material property; 0 otherwise."""
    try:
        study_type = registry.study_type(spec.study.type)
    except KeyError:   # reported by validate
        return 0
    if not any(param.name == 'tolerance' for param in study_type.params):
        return 0
    if n_updates:
        return len(spec.physics) if names is None else len(names)
    touched = {name for coupling in spec.couplings if coupling.enabled for name in (coupling.source, coupling.target)}
    if len(touched) <= 1:
        return 0
    return len(touched) if names is None else len(touched & set(names))


def _feedback_copy_count(spec, registry, n_updates, names=None):
    """Upper bound of the primary-field copies (of the physics in names; None: all) of study._feedback_rounds' monitor
    (alive beside the background fixed point's): every physics when a coupling feeds a study-solved physics into the
    background and something (a coupling or a state-dependent material property) loops back into a solved one; 0
    otherwise."""
    try:
        study_type = registry.study_type(spec.study.type)
    except KeyError:   # reported by validate
        return 0
    if not study_type.solution_feedback:
        return 0
    solved = set()
    for physics_spec in spec.physics:
        physics_type = _registered_physics_type(registry, physics_spec)
        if physics_type is not None and study_type.name in physics_type.supported_studies:
            solved.add(physics_spec.name)
    enabled = [coupling for coupling in spec.couplings if coupling.enabled]
    if not any(coupling.source in solved and coupling.target not in solved for coupling in enabled):
        return 0
    loop_back = n_updates > 0 or any(coupling.source not in solved and coupling.target in solved for coupling in enabled)
    if not loop_back:
        return 0
    return len(spec.physics) if names is None else len(names)


def _state_dependent_property_count(spec, registry, property_names=None):
    """Material property values (of property_names; None: any) that reference a physics state field or time t (one
    DynamicMaterialUpdate each)."""
    state_names = frozenset(state_field_names(spec, registry)) | TIME_NAMES
    count = 0
    for material in spec.materials:
        if not isinstance(material.properties, dict):
            continue
        for name, value in material.properties.items():
            if not isinstance(value, str) or (property_names is not None and name not in property_names):
                continue
            try:
                if not referenced_names(value).isdisjoint(state_names):
                    count += 1
            except ExpressionError:   # reported by validate
                continue
    return count


def _build_cache_bytes(mesh, regions):
    """Build caches alive until BuildContext.release_caches: a bool mask per material region, the int16 paint image,
    one float32 field-dependent value and the cropped object masks (bool, bounding boxes). 3D: the int16 material
    index, one region's box mask and z-dependent value at a time, an int16 slab image and a float32 xy value layer."""
    n_cells = int(mesh.n_cells)
    object_cells = sum(region.box_cells for region in regions if region.box_cells < n_cells)
    if mesh.dz_cells is not None:   # 3D
        largest_box = max((region.box_cells for region in regions), default=0)
        return (2 * n_cells + largest_box * (BOOL_BYTES + FLOAT32_BYTES)
                + mesh.nx * mesh.ny * (2 + FLOAT32_BYTES) + object_cells * BOOL_BYTES)
    return (len(regions) * BOOL_BYTES + 2 + FLOAT32_BYTES) * n_cells + object_cells * BOOL_BYTES

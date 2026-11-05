"""Finite-volume scalar diffusion on the physical window: -div(c grad u) = s with Dirichlet / Neumann / Robin outer
faces, fixed cells eliminated, Jacobi-PCG (solvers.pcg). Shared by heat (c = k, u = T - T0) and the potential physics
(c = sigma or eps0 eps_dc, u = V). Units: conductances [c], rhs and fluxes [c u], sources [c u / m^2] times the cell
area (3D: the 7-point operator of fv_diffusion_3d, sources [c u / m^3] times the cell volume). 2D face areas, widths and
cell areas come from the window's core.metric.WindowMetric ((x, y), (r, z), (r, phi) or (r, theta)); a phi wrap
(WindowMetric.wrap_sign) joins the y_hi face of the last angle cells to the y_lo face of the first ones; an (r, z)
azimuthal order m != 0 (symmetry 2.14) solves the amplitude u_m of u_m(r, z) cos(m phi): + c_xx m^2 / r_c^2 x volume on
the diagonal, rings weighted by pi.
Array-module agnostic: the Domain's backend (cupy / cupyx on the gpu, numpy / scipy on the cpu)."""
import numpy as np

from ..solvers.pcg import DTYPE_PRECISIONS, SOLVER_DTYPE_NAMES, PCGSolver
from .backend import array_module_of, cp, ndimage_module, sparse_module, to_host
from .box_values import BoxValues
from .constants import DTYPE, FACES, FACES_3D
from .error_indicators import compliance, face_jump_indicator
from .fv_diffusion_3d import (FaceGeometry, anchor_link_graph, build_laplacian, edge_extension_views_3d,
                              face_lower_shares, flux_at_centres)
from .metric import GRIDS, face_positions, ring_angle, window_metric, window_metric_3d
from .monitors import FieldProbe
from .solver_base import PhysicsSolver


class BCSide:
    """Outer-face condition: dirichlet (value required = the unknown at the face), neumann (value = inward flux
    [c u / m], default 0 = insulated) or robin (h [c / m] >= 0 and external both required). Other arguments raise.
    conductance_factor (> 0, dirichlet / robin) scales the face's half-cell conductance 2 c / w (magnetic axis: 2)."""

    __slots__ = ("kind", "value", "h", "external", "conductance_factor")

    # Arguments each kind accepts and the ones it requires.
    _KIND_ARGUMENTS = {'dirichlet': (('value', 'conductance_factor'), ('value',)),
                       'neumann': (('value',), ()),
                       'robin': (('h', 'external', 'conductance_factor'), ('h', 'external'))}

    def __init__(self, kind: str = "neumann", value: float = None,
                 h: float = None, external: float = None, conductance_factor: float = None):
        if kind not in self._KIND_ARGUMENTS:
            raise ValueError(f"unknown BC kind '{kind}'")
        given = {name: number for name, number in (('value', value), ('h', h), ('external', external),
                                                   ('conductance_factor', conductance_factor))
                 if number is not None}
        accepted, required = self._KIND_ARGUMENTS[kind]
        irrelevant = sorted(set(given) - set(accepted))
        if irrelevant:
            raise ValueError(f"BCSide('{kind}') does not take {', '.join(irrelevant)} "
                             f"(accepts {', '.join(accepted)})")
        missing = [name for name in required if name not in given]
        if missing:
            raise ValueError(f"BCSide('{kind}') requires {', '.join(missing)}")
        for name, number in given.items():
            if not np.isfinite(float(number)):
                raise ValueError(f"BCSide {name} must be finite, got {number}")
        if given.get('h', 0.0) < 0.0:
            raise ValueError(f"transfer coefficient h must be >= 0, got {h}")
        if given.get('conductance_factor', 1.0) <= 0.0:
            raise ValueError(f"conductance_factor must be > 0, got {conductance_factor}")
        self.kind = kind
        self.value = float(given.get('value', 0.0))   # unused slots stay 0.0
        self.h = float(given.get('h', 0.0))
        self.external = float(given.get('external', 0.0))
        self.conductance_factor = float(given.get('conductance_factor', 1.0))


class ScalarBoundaryConfig:
    """One BCSide per outer window face; unset faces are insulated (zero-flux neumann); z_lo / z_hi act only on a 3D
    domain."""

    def __init__(self,
                 x_lo: BCSide = None, x_hi: BCSide = None,
                 y_lo: BCSide = None, y_hi: BCSide = None,
                 z_lo: BCSide = None, z_hi: BCSide = None):
        given = {'x_lo': x_lo, 'x_hi': x_hi, 'y_lo': y_lo, 'y_hi': y_hi, 'z_lo': z_lo, 'z_hi': z_hi}
        for face in FACES_3D:
            setattr(self, face, given[face] if given[face] is not None else BCSide("neumann"))

    def sides(self):
        return {face: getattr(self, face) for face in FACES}


class CrossQuads:
    """c_xy cross term of one 2D Cartesian assembly per interior vertex (the quad of its four cells, (nx-1, ny-1)):
    area-weighted mean c_xy (0 off `conducting`), lower cells' width shares and centre spacings [m] along x / y."""

    __slots__ = ('coefficient', 'conducting', 'lower_x', 'lower_y', 'spacing_x', 'spacing_y')

    def __init__(self, coefficient, conducting, lower_x, lower_y, spacing_x, spacing_y):
        self.coefficient = coefficient   # (nx-1, ny-1) [c]
        self.conducting = conducting     # (nx-1, ny-1) bool: all four cells conduct (c_xx, c_yy > 0) or are held
        self.lower_x = lower_x           # (nx-1, 1) lower-x cell's share of the centre spacing
        self.lower_y = lower_y           # (1, ny-1)
        self.spacing_x = spacing_x       # (nx-1, 1) centre spacing [m]
        self.spacing_y = spacing_y       # (1, ny-1)

    def mean_differences(self, unknown):
        """(mean x difference, mean y difference) of a window-shaped unknown per quad: its exact linear-u steps."""
        mean_dx = 0.5 * ((unknown[1:, :-1] - unknown[:-1, :-1]) + (unknown[1:, 1:] - unknown[:-1, 1:]))
        mean_dy = 0.5 * ((unknown[:-1, 1:] - unknown[:-1, :-1]) + (unknown[1:, 1:] - unknown[1:, :-1]))
        return mean_dx, mean_dy


# Quad corners as window slices (SW, SE, NW, NE) and each corner cell's share of the quad's area.
_QUAD_CORNERS = ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(None, -1)),
                 (slice(None, -1), slice(1, None)), (slice(1, None), slice(1, None)))


def _quad_corner_shares(cross):
    return (cross.lower_x * cross.lower_y, (1.0 - cross.lower_x) * cross.lower_y,
            cross.lower_x * (1.0 - cross.lower_y), (1.0 - cross.lower_x) * (1.0 - cross.lower_y))


def cross_quads(c_xx, c_yy, c_xy, dx_cells, dy_cells, dtype, held=None):
    """CrossQuads of window coefficient arrays (host widths dx_cells / dy_cells, held a bool window mask or None)."""
    xp = array_module_of(c_xx)
    dx_host = np.asarray(dx_cells, dtype=np.float64)
    dy_host = np.asarray(dy_cells, dtype=np.float64)
    c_xy = xp.asarray(c_xy, dtype=dtype)
    conducting = (c_xx > 0) & (c_yy > 0)
    if held is not None:
        conducting = conducting | held
    quad_conducting = conducting[:-1, :-1] & conducting[1:, 1:] & conducting[:-1, 1:] & conducting[1:, :-1]
    lower_x = xp.asarray(dx_host[:-1] / (dx_host[:-1] + dx_host[1:]), dtype=dtype)[:, None]   # lower cell's share
    lower_y = xp.asarray(dy_host[:-1] / (dy_host[:-1] + dy_host[1:]), dtype=dtype)[None, :]
    quad_mean = (lower_x * (lower_y * c_xy[:-1, :-1] + (1.0 - lower_y) * c_xy[:-1, 1:])
                 + (1.0 - lower_x) * (lower_y * c_xy[1:, :-1] + (1.0 - lower_y) * c_xy[1:, 1:]))
    spacing_x = xp.asarray(0.5 * (dx_host[:-1] + dx_host[1:]), dtype=dtype)[:, None]
    spacing_y = xp.asarray(0.5 * (dy_host[:-1] + dy_host[1:]), dtype=dtype)[None, :]
    return CrossQuads(xp.where(quad_conducting, quad_mean, 0.0), quad_conducting, lower_x, lower_y,
                      spacing_x, spacing_y)


def build_anisotropic_laplacian(c_xx, c_yy, c_xy, dx_cells, dy_cells, bcs,
                                pattern=None, dtype=np.float32, reference=0.0, held=None, metric=None, held_wall=None,
                                apertures=None, kxy_active=None, cross=None, geometry=None):
    """2D FV diffusion operator (fv_diffusion_3d.build_laplacian): A [c], boundary rhs [c u], pattern, interior
    (conductance_x, conductance_y), {face: boundary_face_terms} and the phi-wrap conductance (nx,) (None without a
    wrap) for u - reference; c window-shaped (nx, ny) backend arrays (c_xx's module sets the backend: cupyx or scipy
    CSR), host widths, held / held_wall (cut cells: (t_x, t_y) face fractions) as fv_diffusion_3d.face_conductances;
    apertures = (a_x (nx-1, ny), a_y (nx, ny-1)) backend arrays scale the interior face areas (cut cells at insulated
    walls, core.cut_cells).
    metric: the window's core.metric.WindowMetric (None = (x, y) of the widths): its face areas and physical widths
    make A and rhs totals per metre of depth (planar) or per whole ring (axisymmetric, [c u m]); with its wrap_sign s
    the wrap pair couples the last and first angle cells by -s g (y_lo / y_hi then take no condition: insulated only).
    geometry: the caller's FaceGeometry of that metric in dtype (None = built here).
    kxy_active: the caller's decision whether c_xy is nonzero (None = test it here, one host sync); cross: the caller's
    cross_quads of these coefficients (None = built here when kxy_active). Cross term 2 c_xy d2u/dxdy per interior
    vertex: quad energy c_q/2 [(u_NE - u_SW)^2 - (u_NW - u_SE)^2] = 2 c_q (mean x difference) (mean y difference),
    the exact cross energy of a linear u over the rectangle between the four centres on any widths; c_q = the
    area-weighted mean c_xy of that rectangle (0 unless all four conduct or are held); no cross flux through outer
    faces; for uniform c_xy the interior stencil is NE/SW pairs -c_xy/2, NW/SE pairs +c_xy/2 with no diagonal."""
    xp = array_module_of(c_xx)
    nx, ny = c_xx.shape
    dx_host = np.asarray(dx_cells, dtype=np.float64)
    dy_host = np.asarray(dy_cells, dtype=np.float64)
    if dx_host.shape != (nx,) or dy_host.shape != (ny,):
        raise ValueError(
            f"cell widths {dx_host.shape}, {dy_host.shape} do not match coefficient arrays {c_xx.shape}")
    if metric is None:
        metric = window_metric(dx_host, dy_host)
    elif metric.volume.along_x.shape != (nx,) or metric.volume.along_y.shape != (ny,):
        raise ValueError(f"metric of {metric.volume.along_x.size} x {metric.volume.along_y.size} cells does not "
                         f"match coefficient arrays {c_xx.shape}")

    c_xx = xp.asarray(c_xx, dtype=dtype)
    c_yy = xp.asarray(c_yy, dtype=dtype)

    if kxy_active is None:
        kxy_active = bool(array_module_of(c_xy).any(c_xy))
    if kxy_active and not metric.cartesian_planar:
        raise NotImplementedError(f"c_xy cross conduction is not available on the {metric.label()} grid: set the "
                                  "xy component to 0.")
    if metric.wrap_sign:
        if ny < 3:
            raise ValueError(f"a phi wrap needs at least 3 angle cells, got {ny}")
        for face in ('y_lo', 'y_hi'):
            bc = getattr(bcs, face)
            if bc.kind != 'neumann' or bc.value != 0.0:
                raise ValueError(f"{face} of a polar window with angle_wrap {metric.coordinates.angle_wrap!r} is joined "
                                 f"to the opposite face (phi wrap) and takes no boundary condition, got {bc.kind} "
                                 f"(value {bc.value:g})")
    if geometry is None:
        geometry = FaceGeometry(metric, dtype, xp)
    if kxy_active and cross is None:
        cross = cross_quads(c_xx, c_yy, c_xy, dx_host, dy_host, dtype, held)
    return build_laplacian((c_xx, c_yy), geometry, bcs, pattern=pattern, dtype=dtype, reference=reference, held=held,
                           held_wall=held_wall, apertures=apertures, cross=cross if kxy_active else None)


def edge_extension_views(array, x_range, y_range):
    """(destination, source) view pairs copying the outermost window row/column of a full-grid 2D array into the PML
    ring (fv_diffusion_3d.edge_extension_views_3d on the window slices)."""
    return edge_extension_views_3d(array, (slice(*x_range), slice(*y_range)))


# Eliminated rhs (solved) and un-eliminated rhs (flux accounting) of a steady solve, one fused pass.
_source_rhs = cp.ElementwiseKernel(
    'float32 coupled_source, float32 applied_source, T cell_area, T free_mask, T rhs_offset, T rhs_drive',
    'T rhs, T rhs_full',
    'rhs = free_mask * ((T)coupled_source + (T)applied_source) * cell_area + rhs_offset;'
    'rhs_full = ((T)coupled_source + (T)applied_source) * cell_area + rhs_drive;',
    'fv_source_rhs')

# Same with the implicit-Euler capacity term C/dt * u_old.
_source_rhs_capacity = cp.ElementwiseKernel(
    'float32 coupled_source, float32 applied_source, T cell_area, T free_mask, T rhs_offset, T rhs_drive, '
    'T capacity_over_dt, T unknown_old',
    'T rhs, T rhs_full',
    'rhs = free_mask * (((T)coupled_source + (T)applied_source) * cell_area'
    ' + capacity_over_dt * unknown_old) + rhs_offset;'
    'rhs_full = ((T)coupled_source + (T)applied_source) * cell_area + rhs_drive'
    ' + capacity_over_dt * unknown_old;',
    'fv_source_rhs_capacity')


def _source_rhs_host(coupled_source, applied_source, cell_area, free_mask, rhs_offset, rhs_drive, rhs, rhs_full):
    """numpy twin of _source_rhs (same arguments, writes rhs / rhs_full; a change there changes this)."""
    source = (np.asarray(coupled_source, dtype=rhs.dtype) + np.asarray(applied_source, dtype=rhs.dtype)) * cell_area
    np.add(free_mask * source, rhs_offset, out=rhs)
    np.add(source, rhs_drive, out=rhs_full)


def _source_rhs_capacity_host(coupled_source, applied_source, cell_area, free_mask, rhs_offset, rhs_drive,
                              capacity_over_dt, unknown_old, rhs, rhs_full):
    """numpy twin of _source_rhs_capacity."""
    source = (np.asarray(coupled_source, dtype=rhs.dtype) + np.asarray(applied_source, dtype=rhs.dtype)) * cell_area
    capacity = capacity_over_dt * unknown_old
    np.add(free_mask * (source + capacity), rhs_offset, out=rhs)
    np.add(source + rhs_drive, capacity, out=rhs_full)


# A_solve values of the fixed-cell elimination: entry x free(row) x free(column), one pass over the CSR entries.
_eliminate_fixed_entries = cp.ElementwiseKernel(
    'T data, I row, I column, raw T free', 'T solve_data',
    'solve_data = data * free[row] * free[column]',
    'fv_eliminate_fixed_entries')


def _eliminate_fixed_entries_host(data, row, column, free, solve_data):
    """numpy twin of _eliminate_fixed_entries."""
    np.multiply(data * free[row], free[column], out=solve_data)


# Per-cell code of what the anchor check reads (bits 0-6: c_xx, c_yy, c_zz > 0, held, fixed, loss anchor, capacity >
# 0; bit 7, an outer face with conductance > 0, is added by the caller): an unchanged code skips the labelling.
_anchor_code = cp.ElementwiseKernel(
    'T c_xx, T c_yy, T c_zz, bool held, bool fixed, bool loss, T capacity', 'uint8 code',
    'code = (unsigned char)((c_xx > 0) | ((c_yy > 0) << 1) | ((c_zz > 0) << 2) | (held << 3) | (fixed << 4)'
    ' | (loss << 5) | ((capacity > 0) << 6))',
    'fv_anchor_code')


def _anchor_code_host(c_xx, c_yy, c_zz, held, fixed, loss, capacity, code):
    """numpy twin of _anchor_code (c_zz / capacity may be scalars)."""
    code[...] = 0
    for bit, flag in enumerate((c_xx > 0, c_yy > 0, np.asarray(c_zz) > 0, held, fixed, loss,
                                np.asarray(capacity) > 0)):
        code |= np.asarray(flag, dtype=np.uint8) << np.uint8(bit)


def _join_wrapped_regions(labels_last, labels_first, wrap_links, anchored, wrap_sign):
    """(regions, floating regions) of the link-graph labels (host; anchored indexed by label, 0 = none) joined across
    the conducting phi-wrap links: a union-find with sign parity, where a wrap closing a loop with u = -u (antiperiodic)
    anchors its region."""
    parent = np.arange(anchored.size)
    parity = np.zeros(anchored.size, dtype=np.int8)   # sign of a label relative to its parent (1 = flipped)
    size = np.ones(anchored.size, dtype=np.int64)
    anchored = np.array(anchored, dtype=bool)
    flip = 1 if wrap_sign < 0 else 0   # the wrap link asks u_last = wrap_sign u_first

    def find(label):
        relative = 0
        while parent[label] != label:
            relative ^= int(parity[label])
            label = parent[label]
        return label, relative

    for last, first in zip(np.asarray(labels_last)[wrap_links], np.asarray(labels_first)[wrap_links]):
        root_last, parity_last = find(int(last))
        root_first, parity_first = find(int(first))
        if root_last == root_first:
            if parity_last ^ parity_first != flip:
                anchored[root_last] = True   # an odd loop: only u = 0 satisfies it
            continue
        if size[root_last] < size[root_first]:
            root_last, root_first = root_first, root_last
            parity_last, parity_first = parity_first, parity_last
        parent[root_first] = root_last
        parity[root_first] = parity_last ^ parity_first ^ flip
        size[root_last] += size[root_first]
        anchored[root_last] |= anchored[root_first]
    roots = [label for label in range(1, anchored.size) if parent[label] == label]
    return len(roots), sum(1 for root in roots if not anchored[root])


def _forwarded(owner_attribute, name, cast=None, check=None, doc=None):
    """Property reading `name` of self.<owner_attribute>; with cast also a setter (check(value) first, then the cast
    value written there)."""
    def getter(self):
        return getattr(getattr(self, owner_attribute), name)
    setter = None
    if cast is not None:
        def setter(self, value):
            if check is not None:
                check(value)
            setattr(getattr(self, owner_attribute), name, cast(value))
    return property(getter, setter, doc=doc)


def _require_positive_tolerance(value):
    if not value > 0.0:
        raise ValueError(f"need tol > 0, got {value}")


def _require_iteration_limit(value):
    if int(value) < 1:
        raise ValueError(f"need max_iter >= 1, got {value}")


class DiffusionSystem:
    """Window system of one FV diffusion unknown: buffers, fixed-cell elimination, anchor check, the Jacobi-PCG solve
    (`linear_solver`, a solvers.pcg.PCGSolver holding the convergence record) and the flux / dissipation
    post-processing; the owning physics calls prepare(domain) first and supplies coefficients, sources and BCs."""

    def __init__(self, domain, precision, tol, max_iter, label, unknown_name, anchor_text, remedy_text,
                 compute_residual=False, axisymmetric=False, linear_solver=None, azimuthal_order=0):
        from ..solvers.sparse_direct import DiffusionDirectSolver, SparseDirectSolver
        if isinstance(linear_solver, SparseDirectSolver):   # a sparse_direct node / the 2D cpu Automatic: direct solves
            linear_solver = DiffusionDirectSolver(linear_solver)
        if linear_solver is not None:   # the build's 'linear' solver (ctx.solvers): its precision, tol and limit win
            precision = SOLVER_DTYPE_NAMES[linear_solver.precision]
        if precision not in DTYPE_PRECISIONS:
            raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")
        if not callable(label):
            raise TypeError(f"label must be a zero-argument callable returning the message prefix, "
                            f"got {type(label).__name__}")
        if isinstance(azimuthal_order, bool) or int(azimuthal_order) != azimuthal_order or azimuthal_order < 0:
            raise ValueError(f"{label()}: azimuthal_order must be an integer >= 0, got {azimuthal_order!r}")
        if azimuthal_order and not axisymmetric:
            raise ValueError(f"{label()}: azimuthal_order {azimuthal_order} needs the axisymmetric (r, z) geometry")
        self.domain = domain
        self.backend = domain.backend.name   # 'gpu' (cupy / cupyx) or 'cpu' (numpy / scipy)
        self.xp = domain.backend.xp
        self.precision = precision
        self.dtype = np.float64 if precision == 'float64' else np.float32
        # whole rings: (r, z) about the window's x_lo edge, or (r, theta) about theta = 0 on a polar domain
        self.axisymmetric = bool(axisymmetric)
        # symmetry 2.14: (r, z) order m of u = u_m(r, z) cos(m phi); m != 0 adds c_xx m^2 / r_c^2 per volume, rings pi
        self.azimuthal_order = int(azimuthal_order)
        self.metric = None   # core.metric.WindowMetric of the window (2D), set by prepare
        self.metric_3d = None   # core.metric.WindowMetric3D of a 3D window ((x, y, z) or (r, phi, z)), set by prepare
        self.label = label                # () -> e.g. "heat 'heat'" (the runner names the physics after construction)
        self.unknown_name = unknown_name
        self.anchor_text = anchor_text
        self.remedy_text = remedy_text
        # raises for tol <= 0 / max_iter < 1; factorized at every assemble
        if linear_solver is None:
            linear_solver = PCGSolver(DTYPE_PRECISIONS[precision], float(tol), int(max_iter), label=label,
                                      backend=self.backend)
        if getattr(linear_solver, 'backend', self.backend) != self.backend:
            raise ValueError(f"{label()}: the linear solver runs on the {linear_solver.backend} backend, the domain on "
                             f"the {self.backend} one")
        linear_solver.label = label         # messages name the physics
        linear_solver.compute_residual = bool(compute_residual)
        self.linear_solver = linear_solver
        self.require_anchors = True         # False: no singular-system check (the heat noise solve adds j omega C)
        self.n_diagonal_refreshes = 0       # refresh_diagonal calls that replaced a full assembly
        self.n_held_refreshes = 0           # refresh_held_values calls that replaced a full assembly
        self._assembled_held_count = None   # count of fixed_user cells at the last assembly (0-d device / int)
        # owner hook (DiffusionPhysicsBase): called with new / old volume fractions (window device array) whenever the
        # insulated-wall cut cells change cell_area, so perimeter flux densities keep their totals
        self.volume_fraction_listener = None
        self.operator_dirty = True

    @property
    def operator_dirty(self) -> bool:
        """True when the last assembly is stale; setting it (either way) also clears the diagonal-only mark."""
        return self._operator_dirty

    @operator_dirty.setter
    def operator_dirty(self, value: bool) -> None:
        self._operator_dirty = bool(value)
        self._diagonal_only_change = False   # True only while the losses alone changed since the last assembly
        self._held_values_only_change = False   # True only while set_fixed alone ran since the last assembly

    def _mark_diagonal_changed(self) -> None:
        """operator_dirty after a loss change; keeps refresh_diagonal possible when nothing else changed."""
        diagonal_only = (not self._operator_dirty or self._diagonal_only_change) and self.operator_full is not None
        self.operator_dirty = True
        self._diagonal_only_change = diagonal_only

    @property
    def ring_angle(self) -> float:
        """Angle weight [rad] of the axisymmetric rings (core.metric.ring_angle: 2 pi, pi at an order m != 0)."""
        return ring_angle(self.azimuthal_order)

    def _azimuthal_diagonal(self, c_xx):
        """c_xx m^2 / r_c^2 x cell volume [c] (window, solve dtype) of an order m != 0, 0 on held cells (ideal: their
        own azimuthal exchange is internal); None at m = 0."""
        if self._azimuthal_factor is None:
            return None
        xp = self.xp
        diagonal = xp.asarray(c_xx, dtype=self.dtype) * self._azimuthal_factor * self.cell_area
        return xp.where(self.fixed_user, 0.0, diagonal).astype(self.dtype, copy=False)

    # --- solver settings and convergence record (the PCGSolver holds them) ------

    tol = _forwarded('linear_solver', 'tolerance', float)
    max_iter = _forwarded('linear_solver', 'max_iterations', int)
    compute_residual = _forwarded('linear_solver', 'compute_residual', bool)
    n_solves = _forwarded('linear_solver', 'n_solves')
    last_iters = _forwarded('linear_solver', 'last_iters')
    last_converged = _forwarded('linear_solver', 'last_converged')
    last_residual = _forwarded('linear_solver', 'last_residual', doc=(
        "||b - A x|| relative to the last solve's stop reference (see solve); nan unless compute_residual or "
        "unconverged."))
    last_warning = _forwarded('linear_solver', 'last_warning')
    n_unconverged = _forwarded('linear_solver', 'n_unconverged')

    # --- allocation -----------------------------------------------------------

    def prepare(self, domain) -> None:
        """Allocate every window buffer on the domain's physical window (re-allocates on a second call: never after
        fixed cells or losses were added)."""
        if domain.backend.name != self.backend:
            raise ValueError(f"{self.label()}: the system was built for the {self.backend} backend, the domain runs on "
                             f"{domain.backend.name}")
        self.domain = domain
        self.dimensions = domain.dimensions
        xp = self.xp
        if self.dimensions == 3:
            self._prepare_window_3d(domain)
        else:
            nx, ny = domain.grid.nx, domain.grid.ny
            self.grid_shape = (nx, ny)

            window_x, window_y = domain.physical_window()
            x_start, x_stop, _ = window_x.indices(nx)
            y_start, y_stop, _ = window_y.indices(ny)
            self.window = (slice(x_start, x_stop), slice(y_start, y_stop))
            window_shape = (x_stop - x_start, y_stop - y_start)
            self.window_shape = window_shape
            self.n_window = window_shape[0] * window_shape[1]

            dx_cells, dy_cells = domain.cell_widths()
            self.dx_cells = np.asarray(dx_cells, dtype=np.float64)
            self.dy_cells = np.asarray(dy_cells, dtype=np.float64)
            self.dx_window = self.dx_cells[x_start:x_stop]
            self.dy_window = self.dy_cells[y_start:y_stop]

            dtype = self.dtype
            self.dx_window_device = xp.asarray(self.dx_window, dtype=dtype)
            self.dy_window_device = xp.asarray(self.dy_window, dtype=dtype)
            # exact face areas / widths / volumes of the domain's grid ((x, y), (r, z), (r, phi), (r, theta))
            # (an order m != 0 on (r, z): rings of angle pi, raises on any other grid)
            self.metric = window_metric(self.dx_window, self.dy_window, domain.coordinates, self.axisymmetric,
                                        self.azimuthal_order)
            self.metric_3d = None
            # cell area [m^2] (planar) or volume [m^3] (axisymmetric): every source, loss and capacity term scales by it
            self.cell_area = xp.asarray(self.metric.volume.full(), dtype=dtype)
            # m^2 / r_c^2 (wx, 1) of an order m != 0 (x = r from the window's x_lo edge), None at m = 0
            self._azimuthal_factor = None
            if self.azimuthal_order:
                r_faces = face_positions(self.dx_window, 0.0)
                r_centres = 0.5 * (r_faces[:-1] + r_faces[1:])
                self._azimuthal_factor = xp.asarray(
                    (self.azimuthal_order ** 2 / r_centres ** 2)[:, None], dtype=dtype)
            # device face-area factors and physical lengths (1D factors; the polar r_c dphi lengths one window array)
            self._geometry = FaceGeometry(self.metric, dtype, xp)
        dtype = self.dtype
        window_shape = self.window_shape
        from ..solvers.sparse_direct import DiffusionDirectSolver
        if isinstance(self.linear_solver, DiffusionDirectSolver):   # unknowns = window cells in C order
            self.linear_solver.grid_shape = tuple(window_shape)
        self.rhs = xp.zeros(self.n_window, dtype=dtype)          # eliminated rhs (solved)
        self.rhs2d = self.rhs.reshape(window_shape)
        self.unknown = xp.zeros(self.n_window, dtype=dtype)      # u - reference
        self.unknown2d = self.unknown.reshape(window_shape)
        self.extra_diag = xp.zeros(window_shape, dtype=dtype)    # diagonal loss conductance [c]
        self.extra_rhs = xp.zeros(window_shape, dtype=dtype)     # diagonal loss drive [c u]
        self.fixed_user = xp.zeros(window_shape, dtype=xp.bool_)
        self.fixed_user_values = xp.zeros(window_shape, dtype=dtype)
        # host flag: False while no set_fixed ran (fixed_user all False), so assemblies skip the held-face branches
        self._maybe_held = False
        self.held_wall = None   # (t_x, t_y) interior-face held-edge fractions of set_held_wall (2D cut cells)
        # insulated-wall cut cells (2D): material_index test, the conducting cells the walls were located on (device
        # bool, re-located when an assembly's pattern differs: k(T)), (a_x, a_y) face apertures, host volume fractions
        self._insulated_wall_test = None
        self._insulated_wall_conducting = None
        self.n_insulated_wall_locations = 0
        # cross term: kxy_active decided (one host sync) at the first assembly and after the owner sets
        # cross_term_dirty (its materials_changed of the xy array), not at every re-assembly
        self.cross_term_dirty = True
        self._kxy_active = False
        self.apertures = None
        self.volume_fraction = None
        self.geometric_cell_area = self.cell_area   # cell_area before volume fractions (dissipation_density divisor)
        # perimeter losses of add_surface_loss [c] / [c u] (totals inside extra_diag / extra_rhs that volume fractions
        # never rescale; None = none) and the owner's hook volume_fraction_listener(ratio) (outline flux densities)
        self.surface_diag = None
        self.surface_rhs = None
        self._rhs_full_buffer = xp.zeros(window_shape, dtype=dtype)   # un-eliminated rhs, written by build_rhs
        # CG solution w = u - u_fix (the residual / direction workspace is the PCGSolver's, sized at assemble)
        self._solution = xp.zeros(self.n_window, dtype=dtype)

        self.pattern = None            # CSR index structure, reused across rebuilds
        self.operator_full = None      # A_eff CSR (conduction + losses + capacity), rows of fixed cells included
        self._conduction_diagonal = None   # diag(A) of the last assembly without losses / capacity, flat
        self.azimuthal_diagonal = None     # _azimuthal_diagonal of the last assembly (inside the conduction diagonal)
        self._loss_anchor_cells = None     # extra_diag > 0 at the last assembly (anchors of its check)
        self.rhs_full = None          # un-eliminated rhs of the last build_rhs (2D); None before it
        self.rhs_drive = None          # boundary + loss drive [c u] (2D)
        self.face_terms = None         # {face: (cells, conductance, rhs)} of the last assembly
        self.face_conductances = None  # interior (conductance_x, conductance_y) [c] of the last assembly
        # phi wrap (polar 2D / cylindrical 3D, metric wrap_sign s = +-1): conductance (nx,) (3D: (nx, nz)) of the last
        # assembly's face j = ny-1 | j = 0
        metric = self.metric if self.metric is not None else self.metric_3d
        self.wrap_sign = 0 if metric is None else metric.wrap_sign
        self.wrap_conductance = None
        self.assembled_c_xx = None     # owned solve-dtype copies of the last assembly's c_xx / c_yy (/ c_zz in 3D)
        self.assembled_c_yy = None
        self.assembled_c_zz = None
        self.cross = None                # CrossQuads of the last assembly (2D c_xy != 0), None without a cross term
        self._lower_cell_shares = None   # fv_diffusion_3d.face_lower_shares of the last assembly, filled by dissipation_density
        self.A_solve = None
        self.inverse_diagonal = None   # Jacobi preconditioner 1 / diag(A_solve), flat
        self.free = None
        self.rhs_offset = None
        self.fixed_values = None       # window values of the fixed cells (0 on free cells), added back after CG
        self.capacity_over_dt = None
        self.anchor_signature = None       # uint8 anchor code (_anchor_code) of the last checked assembly
        # gpu re-assembly with a PCG solver: (device bool unchanged, code, _label_anchors arguments), read by solve()
        self._pending_anchor_check = None
        self.operator_dirty = True

    def _prepare_window_3d(self, domain):
        """3D part of prepare: window slices, per-axis widths, the metric ((x, y, z) or a cylindrical (r, phi, z)
        sector of a polar Domain3D, symmetry 2.21) and the cell volumes (kept as cell_area)."""
        if self.axisymmetric:
            raise ValueError(f"{self.label()}: axisymmetric (r, z) geometry needs a 2D domain, got a 3D one")
        self.metric = None   # 3D: metric_3d
        self._azimuthal_factor = None   # axisymmetric (and so an azimuthal order) raised above
        self.grid_shape = tuple(domain.materials.shape)
        self.window = tuple(domain.physical_window())
        self.window_shape = tuple(window.stop - window.start for window in self.window)
        self.n_window = int(np.prod(self.window_shape))
        self.dx_cells, self.dy_cells, self.dz_cells = (np.asarray(widths, dtype=np.float64)
                                                       for widths in domain.cell_widths())
        self.dx_window, self.dy_window, self.dz_window = (
            widths[window] for widths, window in zip((self.dx_cells, self.dy_cells, self.dz_cells), self.window))
        self.metric_3d = window_metric_3d(self.dx_window, self.dy_window, self.dz_window, domain.coordinates)
        # device face-area factors and physical lengths (small: 1D factors, lengths at most (wx, wy, 1))
        self._geometry = FaceGeometry(self.metric_3d, self.dtype, self.xp)
        # cell volume [m^3]: every source, loss and capacity term scales with it (product of the device factors)
        x_factor, y_factor, z_factor = (self.xp.asarray(factor, dtype=self.dtype)
                                        for factor in self.metric_3d.volume.factors())
        self.cell_area = x_factor * y_factor * z_factor

    # --- window helpers ---------------------------------------------------------

    @property
    def faces(self) -> tuple:
        """Outer face names of the window: FACES, FACES_3D on a 3D domain."""
        return FACES_3D if self.dimensions == 3 else FACES

    @property
    def geometry_3d(self):
        """FaceGeometry of a 3D window (device face-area factors and lengths; fv_diffusion_3d helpers), else None."""
        return self._geometry if self.dimensions == 3 else None

    def window_mask(self, mask, label):
        """Window view (bool) of a full-grid mask; ValueError naming `label` for another shape."""
        mask = self.xp.asarray(mask)
        if mask.shape != self.grid_shape:
            raise ValueError(
                f"{label}: mask must be a full-grid {self.grid_shape} boolean array, "
                f"got shape {mask.shape}")
        return mask[self.window].astype(np.bool_, copy=False)

    def window_values(self, value, label, dtype):
        """0-d device array for a scalar; window view cast to dtype for a full-grid array (3D also an (nx, ny, 1)
        layer, broadcast along z: BuildContext.eval_field's x, y values; a BoxValues of z-dependent ones)."""
        if isinstance(value, BoxValues):
            return value.window_array(self.window, dtype)
        values = self.xp.asarray(value, dtype=dtype)
        if values.ndim == 0:
            return values
        if len(self.grid_shape) == 3 and values.shape == self.grid_shape[:2] + (1,):
            return values[self.window[0], self.window[1]]
        if values.shape != self.grid_shape:
            raise ValueError(
                f"{label} must be a scalar or a full-grid {self.grid_shape} array, "
                f"got shape {values.shape}")
        return values[self.window]

    def add_masked(self, target, mask_window, value, label) -> None:
        """target += value under mask_window, in place (target window-shaped; value as window_values); a BoxValues
        adds on its box only (box-sized temporary)."""
        if isinstance(value, BoxValues):
            value.add_masked_to(target, mask_window, self.window)
            return
        target += self.xp.where(mask_window, self.window_values(value, label, target.dtype), target.dtype.type(0))

    def copy_masked(self, target, mask_window, value, label) -> None:
        """target = value under mask_window, in place (as add_masked)."""
        if isinstance(value, BoxValues):
            value.copy_masked_to(target, mask_window, self.window)
            return
        self.xp.copyto(target, self.window_values(value, label, target.dtype), where=mask_window)

    def cell_volume(self, ix, iy, iz=None) -> float:
        """Host float64 area [m^2] (planar) or volume [m^3] (axisymmetric; 3D: (ix, iy, iz)) of a full-grid window
        cell, as cell_area (insulated-wall volume fractions only once located: the first assembly)."""
        if iz is not None:
            volume = self.metric_3d.volume
            i, j, k = (index - window.start for index, window in zip((ix, iy, iz), self.window))
            return float(volume.along_x[i] * volume.along_y[j] * volume.along_z[k])
        volume = self.metric.volume
        i, j = ix - self.window[0].start, iy - self.window[1].start
        fraction = 1.0 if self.volume_fraction is None else float(self.volume_fraction[i, j])
        return float(volume.along_x[i] * volume.along_y[j]) * fraction

    def require_window_cell(self, ix, iy, label, iz=None):
        """ValueError naming `label` when full-grid cell (ix, iy) (3D: (ix, iy, iz)) lies outside the solve window."""
        cell = (ix, iy) if iz is None else (ix, iy, iz)
        cell_text = '(' + ', '.join(str(index) for index in cell) + ')'
        if len(cell) != len(self.window):
            raise ValueError(f"{label} needs a {len(self.window)}D cell index on this domain, got {cell_text}")
        if not all(window.start <= index < window.stop for index, window in zip(cell, self.window)):
            ranges = ', '.join(f'{axis} [{window.start}, {window.stop})' for axis, window in zip('xyz', self.window))
            raise ValueError(f"{label} cell {cell_text} lies outside the solve window {ranges}")

    # --- problem setup ----------------------------------------------------------

    def set_insulated_wall_test(self, material_index) -> None:
        """Cut cells (2D): locate the curved insulated walls at the next assembly from material_index(x, y)
        (host, 1 + material, model.boundary_locator) and that assembly's conducting cells (core.cut_cells), again at
        every assembly whose conducting cells differ (k(T), sigma(T))."""
        self._insulated_wall_test = material_index
        self._insulated_wall_conducting = None
        self.operator_dirty = True

    def set_insulated_walls(self, geometry):
        """Apply core.cut_cells insulated_wall_geometry (aperture_x (nx-1, ny), aperture_y (nx, ny-1), volume_fraction
        (nx, ny)) or None (no walls): cell_area = geometric_cell_area x the fractions, the volumetric losses added so far
        follow (add_surface_loss totals do not), volume_fraction_listener gets the ratio new / old; returns that ratio
        (window device array, solve dtype), None when neither the old nor the new state has walls."""
        xp = self.xp
        if geometry is None and self.volume_fraction is None:
            return None
        old = np.ones(self.window_shape) if self.volume_fraction is None else self.volume_fraction
        if geometry is None:
            self.apertures, self.volume_fraction = None, None
            new = np.ones(self.window_shape)
        else:
            aperture_x, aperture_y, volume_fraction = geometry
            self.apertures = (xp.asarray(aperture_x, dtype=self.dtype), xp.asarray(aperture_y, dtype=self.dtype))
            self.volume_fraction = np.asarray(volume_fraction, dtype=np.float64)
            new = self.volume_fraction
        ratio = xp.asarray(new / old, dtype=self.dtype)
        self.cell_area = self.geometric_cell_area * xp.asarray(new, dtype=self.dtype)
        for total, surface in ((self.extra_diag, self.surface_diag), (self.extra_rhs, self.surface_rhs)):
            if surface is not None:
                total -= surface
            total *= ratio
            if surface is not None:
                total += surface
        if self.volume_fraction_listener is not None:
            self.volume_fraction_listener(ratio)
        self.operator_dirty = True
        return ratio

    def set_held_wall(self, fraction_x, fraction_y, located_x, located_y) -> None:
        """Cut cells (2D): on the located interior faces (bool (nx-1, ny) / (nx, ny-1) windows) the held cell's edge
        lies at fraction t of the centre spacing from the free cell (face_conductances held_wall); later calls win."""
        xp = self.xp
        if self.held_wall is None:
            self.held_wall = (xp.full(located_x.shape, np.nan, dtype=self.dtype),
                              xp.full(located_y.shape, np.nan, dtype=self.dtype))
        for stored, fraction, located in zip(self.held_wall, (fraction_x, fraction_y), (located_x, located_y)):
            xp.copyto(stored, xp.asarray(fraction, dtype=self.dtype), where=xp.asarray(located))
        self.operator_dirty = True

    @property
    def _held_cells(self):
        """fixed_user for the held-face formulas, None while no cell was ever held (no per-assembly where pairs)."""
        return self.fixed_user if self._maybe_held else None

    def set_fixed(self, mask_window, values_window) -> None:
        """Hold the unknown at values_window (0-d or window-shaped, solve dtype) on the masked window cells."""
        self.xp.copyto(self.fixed_user_values, values_window, where=mask_window)
        self.fixed_user |= mask_window
        self._maybe_held = True
        self.xp.copyto(self.unknown2d, values_window, where=mask_window)
        held_only = (not self._operator_dirty or self._held_values_only_change) and self.operator_full is not None
        self.operator_dirty = True
        self._held_values_only_change = held_only

    def refresh_held_values(self) -> bool:
        """Re-eliminate the last assembly with the current held values (set_fixed alone ran since, on cells already
        held): fixed_values, coupling and rhs_offset, no Laplacian, anchor check or factorisation; False (nothing done:
        assemble) when anything else changed or a new cell was held (one sync)."""
        if not self._held_values_only_change:
            return False
        xp = self.xp
        if bool(xp.count_nonzero(self.fixed_user) != self._assembled_held_count):   # set_fixed only adds cells
            return False
        window_shape = self.window_shape
        fixed = self.free == 0
        # held cells take the new values, pinned (unconnected) cells keep theirs: assemble's fixed_values
        fixed_values = xp.where(self.fixed_user, self.fixed_user_values, self.fixed_values).astype(self.dtype,
                                                                                                    copy=False)
        coupling_to_fixed = self.operator_full.dot(fixed_values.ravel()).reshape(window_shape)
        self.rhs_offset = self.free * (self.rhs_drive - coupling_to_fixed)
        xp.copyto(self.unknown2d, fixed_values, where=fixed)
        self.fixed_values = fixed_values
        self.n_held_refreshes += 1
        self.operator_dirty = False
        return True

    def add_diagonal_loss(self, mask_window, coefficient_window, external_window, reference) -> None:
        """Add a volumetric loss coefficient [c / m^2] * (u - external) on the masked window cells."""
        conductance = self.xp.where(mask_window, coefficient_window * self.cell_area, 0.0)
        self.extra_diag += conductance
        self.extra_rhs += conductance * (external_window - reference)
        self._mark_diagonal_changed()

    def add_surface_loss(self, mask_window, conductance_window, external_window, reference) -> None:
        """Add a perimeter loss of total conductance [c] per cell (not per volume) * (u - external) on the masked window
        cells (cut-cell convection); insulated-wall volume fractions never rescale it."""
        xp = self.xp
        conductance = xp.where(mask_window, xp.asarray(conductance_window, dtype=self.dtype), 0.0).astype(
            self.dtype, copy=False)
        drive = (conductance * (external_window - reference)).astype(self.dtype, copy=False)
        if self.surface_diag is None:
            self.surface_diag = xp.zeros(self.window_shape, dtype=self.dtype)
            self.surface_rhs = xp.zeros(self.window_shape, dtype=self.dtype)
        self.surface_diag += conductance
        self.surface_rhs += drive
        self.extra_diag += conductance
        self.extra_rhs += drive
        self._mark_diagonal_changed()

    def set_diagonal_loss(self, mask_window, coefficient_window, external_window, reference) -> None:
        """Replace (set, not add) the volumetric loss on the masked window cells by coefficient [c / m^2] * (u -
        external): a term re-evaluated per iteration or cleared on some cells (coefficient 0; it also drops an
        add_surface_loss term there, which no caller combines with it today)."""
        conductance = coefficient_window * self.cell_area
        self.xp.copyto(self.extra_diag, conductance, where=mask_window)
        self.xp.copyto(self.extra_rhs, conductance * (external_window - reference), where=mask_window)
        self._mark_diagonal_changed()

    def add_cell_loss(self, cells, conductance_change, drive_change) -> None:
        """Add conductance [c] to the diagonal and drive [c u] to the rhs of flat window cells (unique indices; the
        changes may be negative: a loss re-linearised per solve, e.g. heat radiation)."""
        self.extra_diag.reshape(-1)[cells] += conductance_change
        self.extra_rhs.reshape(-1)[cells] += drive_change
        self._mark_diagonal_changed()

    def assemble(self, c_xx, c_yy, c_xy, bcs, reference, capacity_over_dt=None, c_zz=None,
                 check_anchors: bool = True) -> None:
        """Assemble A_eff = A + diag(losses) (+ diag(C/dt)), eliminate fixed cells, check anchors, rebuild Jacobi;
        c arrays window-shaped device arrays (3D: c_zz required, c_xy must be 0), capacity_over_dt a window array [c]
        or None. Held cells are ideal (c -> inf): their value sits on their faces toward free cells, whatever their
        own coefficient. check_anchors=False skips the anchor check (caller guarantees an unchanged c > 0 pattern)."""
        dtype = self.dtype
        xp = self.xp
        c_xx = xp.array(c_xx, dtype=dtype)   # owned copies: the flux / dissipation post-processing reads them later
        c_yy = xp.array(c_yy, dtype=dtype)
        if self.cross_term_dirty:
            self._kxy_active = c_xy is not None and bool(array_module_of(c_xy).any(c_xy))
            self.cross_term_dirty = False
        if self.dimensions == 3:
            if c_zz is None:
                raise ValueError(f"{self.label()}: a 3D assembly needs the c_zz coefficients")
            if self._kxy_active:
                raise NotImplementedError(f"{self.label()}: xy cross conduction is not available in 3D: set the xy "
                                          "component to 0.")
            c_zz = xp.array(c_zz, dtype=dtype)
            operator, rhs_bc, self.pattern, face_conductances_xy, face_terms, wrap_conductance = build_laplacian(
                (c_xx, c_yy, c_zz), self._geometry, bcs,
                pattern=self.pattern, dtype=dtype, reference=reference, held=self._held_cells)
            cross = None
        else:
            if self._insulated_wall_test is not None:   # cut cells: the walls of this assembly's conducting cells
                conducting = (c_xx > 0) & (c_yy > 0)
                located = self._insulated_wall_conducting
                # re-located when the pattern changed (one sync; check_anchors=False promises an unchanged pattern)
                if located is None or (check_anchors and not bool(xp.array_equal(located, conducting))):
                    from .cut_cells import locate_insulated_walls
                    ratio = self.set_insulated_walls(
                        locate_insulated_walls(self, conducting, self._insulated_wall_test))
                    self._insulated_wall_conducting = conducting
                    self.n_insulated_wall_locations += 1
                    if capacity_over_dt is not None and ratio is not None:   # computed on the previous areas
                        capacity_over_dt = xp.asarray(capacity_over_dt, dtype=dtype) * ratio
            # kept for the cross flux / dissipation post-processing (the operator raises off a Cartesian plane)
            cross = (cross_quads(c_xx, c_yy, c_xy, self.dx_window, self.dy_window, dtype, self._held_cells)
                     if self._kxy_active and (self.metric is None or self.metric.cartesian_planar) else None)
            operator, rhs_bc, self.pattern, face_conductances_xy, face_terms, wrap_conductance = (
                build_anisotropic_laplacian(
                    c_xx, c_yy, c_xy, self.dx_window, self.dy_window, bcs,
                    pattern=self.pattern, dtype=dtype, reference=reference, held=self._held_cells, metric=self.metric,
                    held_wall=self.held_wall, apertures=self.apertures, kxy_active=self._kxy_active, cross=cross,
                    geometry=self._geometry))
        pattern = self.pattern
        diagonal_positions = pattern.diagonal_positions
        window_shape = self.window_shape
        n_window = self.n_window

        diagonal_extra = self.extra_diag.copy()
        if capacity_over_dt is not None:
            capacity_over_dt = xp.asarray(capacity_over_dt, dtype=dtype)
            diagonal_extra += capacity_over_dt
        effective_data = operator.data   # modified in place: `operator` becomes A_eff
        azimuthal_diagonal = self._azimuthal_diagonal(c_xx)   # order m != 0: part of the conduction diagonal
        if azimuthal_diagonal is not None:
            effective_data[diagonal_positions] += azimuthal_diagonal.ravel()
        conduction_diagonal = effective_data[diagonal_positions]   # a copy: refresh_diagonal re-adds the losses to it
        effective_data[diagonal_positions] += diagonal_extra.ravel()

        # Cells with an empty row (no conduction, loss or capacity) are pinned at their current value.
        effective_diagonal = effective_data[diagonal_positions].reshape(window_shape)
        unconnected = (effective_diagonal <= 0) & ~self.fixed_user
        fixed = self.fixed_user | unconnected
        fixed_values = xp.where(self.fixed_user, self.fixed_user_values,
                                xp.where(unconnected, self.unknown2d, 0.0)).astype(dtype, copy=False)
        loss_anchor_cells = self.extra_diag > 0
        if self.require_anchors and check_anchors:
            self._check_anchors(c_xx, c_yy, fixed, capacity_over_dt, face_terms, c_zz, loss_anchor_cells)

        # CG unknown w = u - u_fix (0 on fixed cells): A_solve = D_free A_eff D_free + D_fixed, rhs_offset = D_free (rhs
        # - A_eff u_fix). Keeping u_fix [u] out of b keeps every entry of b in [c u], so the tolerance relative to ||b||
        # does not depend on the coefficient scale (eps0-scaled rows used to stop after a few iterations).
        free = (~fixed).astype(dtype)
        free_flat = free.ravel()
        solve_data = xp.empty_like(effective_data)
        (_eliminate_fixed_entries_host if self.backend == 'cpu' else _eliminate_fixed_entries)(
            effective_data, pattern.entry_rows, pattern.indices, free_flat, solve_data)
        solve_data[diagonal_positions] += fixed.ravel().astype(dtype)
        self.A_solve = sparse_module(self.backend).csr_matrix((solve_data, pattern.indices, pattern.indptr),
                                                              shape=(n_window, n_window))
        coupling_to_fixed = operator.dot(fixed_values.ravel()).reshape(window_shape)
        rhs_drive = rhs_bc + self.extra_rhs
        self.rhs_offset = free * (rhs_drive - coupling_to_fixed)

        self.inverse_diagonal = 1.0 / solve_data[diagonal_positions]
        self.linear_solver.factorize(self.A_solve, self.inverse_diagonal)

        xp.copyto(self.unknown2d, fixed_values, where=fixed)
        self.fixed_values = fixed_values
        self.free = free
        self.capacity_over_dt = capacity_over_dt
        self.operator_full = operator
        self._conduction_diagonal = conduction_diagonal
        self.azimuthal_diagonal = azimuthal_diagonal
        self._loss_anchor_cells = loss_anchor_cells
        self.rhs_drive = rhs_drive
        self.face_terms = face_terms
        self.face_conductances = face_conductances_xy
        self.wrap_conductance = wrap_conductance
        self.assembled_c_xx = c_xx
        self.assembled_c_yy = c_yy
        self.assembled_c_zz = c_zz
        self.cross = cross
        self.assembled_reference = reference   # the unknown's reference of the face rhs terms (update_face_value)
        self._lower_cell_shares = None
        self._assembled_held_count = xp.count_nonzero(self.fixed_user)   # refresh_held_values: no sync here
        self.operator_dirty = False

    def update_face_value(self, face, old_bc, new_bc) -> bool:
        """Replace one outer face's rhs term after a value-only change of its BCSide (same kind, h and conductance
        factor: a dirichlet value, neumann flux or robin external) in the last 2D assembly: face_terms, rhs_drive and
        rhs_offset in place, no re-assembly; False (nothing done: assemble) when stale, 3D or the conductance changes."""
        if (self.operator_dirty or self.face_terms is None or self.dimensions == 3 or face not in self.face_terms
                or (old_bc.kind, old_bc.h, old_bc.conductance_factor)
                != (new_bc.kind, new_bc.h, new_bc.conductance_factor)):
            return False
        cells, conductance, old_rhs = self.face_terms[face]
        # boundary_face_terms' rhs on the assembly's conductance (unchanged: same kind, h and factor), bit-identical
        if new_bc.kind == 'dirichlet':
            new_rhs = conductance * (new_bc.value - self.assembled_reference)
        elif new_bc.kind == 'robin':
            new_rhs = conductance * (new_bc.external - self.assembled_reference)
        else:   # neumann: value x face area (boundary_face_terms' own formula)
            new_rhs = self._geometry.outer_face_rhs(face, new_bc.value, old_rhs)
        change = new_rhs - old_rhs
        self.face_terms[face] = (cells, conductance, new_rhs)
        self.rhs_drive[cells] += change
        self.rhs_offset[cells] += self.free[cells] * change
        return True

    def refresh_diagonal(self) -> bool:
        """Re-add the current losses to the last assembly (elimination, rhs_offset, Jacobi) without the Laplacian or
        anchor check; False (nothing done: assemble) unless only losses changed and pinned / anchor cells stay."""
        if not self._diagonal_only_change:
            return False
        return self._refresh_diagonal_terms(self.capacity_over_dt)

    def refresh_capacity(self, capacity_over_dt) -> bool:
        """Replace the last assembly's capacity C/dt (window array [c]) as refresh_diagonal re-adds losses (they too);
        False (nothing done: assemble) when the operator is otherwise stale, the last assembly had no C/dt, or the
        pinned / loss-anchored / C/dt > 0 cells change."""
        if (self.operator_full is None or self.capacity_over_dt is None
                or (self._operator_dirty and not self._diagonal_only_change)):
            return False
        return self._refresh_diagonal_terms(self.xp.asarray(capacity_over_dt, dtype=self.dtype))

    def _refresh_diagonal_terms(self, capacity_over_dt) -> bool:
        """Body of refresh_diagonal / refresh_capacity with this C/dt (None, the last assembly's or a new array)."""
        dtype = self.dtype
        xp = self.xp
        window_shape = self.window_shape
        diagonal_positions = self.pattern.diagonal_positions
        diagonal_extra = self.extra_diag.copy()   # the sum order of assemble: bit-identical to a full re-assembly
        if capacity_over_dt is not None:
            diagonal_extra += capacity_over_dt
        effective_diagonal = self._conduction_diagonal + diagonal_extra.ravel()
        fixed = self.fixed_user | (effective_diagonal.reshape(window_shape) <= 0)   # held | pinned (empty rows)
        loss_anchor_cells = self.extra_diag > 0
        changed = xp.any(fixed != (self.free == 0)) | xp.any(loss_anchor_cells != self._loss_anchor_cells)
        if capacity_over_dt is not self.capacity_over_dt:   # a new C/dt: its > 0 cells are anchors too
            changed = changed | xp.any((capacity_over_dt > 0) != (self.capacity_over_dt > 0))
        if bool(changed):
            return False   # the elimination or the anchor check would differ

        self.operator_full.data[diagonal_positions] = effective_diagonal
        free_flat = self.free.ravel()
        solve_diagonal = effective_diagonal * free_flat * free_flat + fixed.ravel().astype(dtype)
        self.A_solve.data[diagonal_positions] = solve_diagonal
        coupling_to_fixed = self.operator_full.dot(self.fixed_values.ravel()).reshape(window_shape)
        rhs_bc = xp.zeros(window_shape, dtype=dtype)   # boundary rhs rebuilt from the face terms (assembly order)
        for face in self.faces:
            cells, _, rhs = self.face_terms[face]
            rhs_bc[cells] += rhs
        self.rhs_drive = rhs_bc + self.extra_rhs
        self.rhs_offset = self.free * (self.rhs_drive - coupling_to_fixed)
        self.inverse_diagonal = 1.0 / solve_diagonal
        self.linear_solver.factorize(self.A_solve, self.inverse_diagonal)
        xp.copyto(self.unknown2d, self.fixed_values, where=fixed)
        self.capacity_over_dt = capacity_over_dt
        self.n_diagonal_refreshes += 1
        self.operator_dirty = False
        return True

    def _check_anchors(self, c_xx, c_yy, fixed, capacity_over_dt, face_terms, c_zz, loss_anchor_cells):
        """Anchor check of an assembly: _label_anchors only when the per-cell anchor code differs from the last checked
        one; on the gpu with a PCG solver the comparison waits for the next solve (whose CG syncs right after)."""
        xp = self.xp
        zero = self.dtype(0.0)
        code = xp.empty(fixed.shape, dtype=np.uint8)
        (_anchor_code_host if self.backend == 'cpu' else _anchor_code)(
            c_xx, c_yy, zero if c_zz is None else c_zz, self.fixed_user, fixed, loss_anchor_cells,
            zero if capacity_over_dt is None else capacity_over_dt, code)
        for cells, conductance, _ in face_terms.values():
            code[cells] |= (conductance > 0).astype(np.uint8) << np.uint8(7)
        arguments = (c_xx, c_yy, fixed, capacity_over_dt, face_terms, c_zz, loss_anchor_cells)
        self._pending_anchor_check = None
        cached = self.anchor_signature
        if cached is None or cached.shape != code.shape:   # the first assembly fails early
            self._finish_anchor_check(False, code, arguments)
            return
        unchanged = xp.array_equal(cached, code)
        if self.backend == 'gpu' and isinstance(self.linear_solver, PCGSolver):
            self._pending_anchor_check = (unchanged, code, arguments)
        else:
            self._finish_anchor_check(unchanged, code, arguments)

    def _finish_anchor_check(self, unchanged, code, arguments):
        """Label (raising for a floating region) unless `unchanged` (a device or host bool); then cache `code`."""
        if not bool(unchanged):
            self._label_anchors(*arguments)
        self.anchor_signature = code

    def _label_anchors(self, c_xx, c_yy, fixed, capacity_over_dt, face_terms, c_zz, loss_anchor_cells):
        """Raise ValueError when a conducting region (cells joined by non-zero face links) has no anchor (singular
        system); an outer face anchors where its conductance is > 0 (dirichlet, h > 0 robin; never the r = 0 axis), an
        antiperiodic phi wrap closing a loop of links anchors its region (no constant null mode), an azimuthal order
        m != 0 every cell with c_xx > 0."""
        held = self.fixed_user   # a held cell anchors its conducting neighbours, whatever its own coefficient
        xp = self.xp
        # link graph at 2x resolution (cells on even pixels / voxels, face links between them, so a zero face is a cut;
        # a phi wrap adds a last y layer of wrap links, labelled with the last cells, joined to the first below)
        coefficients = (c_xx, c_yy, c_zz) if self.dimensions == 3 else (c_xx, c_yy)
        link_image, conducting, anchors, wrap_links = anchor_link_graph(
            coefficients, fixed, held, face_terms, wrap=bool(self.wrap_sign))
        anchors |= loss_anchor_cells
        if self._azimuthal_factor is not None:   # order m != 0: c_xx m^2 / r^2 anchors (no constant mode)
            anchors |= c_xx > 0
        if capacity_over_dt is not None:
            anchors |= capacity_over_dt > 0
        anchors &= conducting

        cell_pixels = (slice(None, None, 2),) * fixed.ndim
        # default structure: 4- (3D: 6-) connected
        link_labels, n_regions = ndimage_module(self.backend).label(link_image)
        labels = link_labels[cell_pixels]
        n_regions = int(n_regions)
        if n_regions:
            anchored = xp.zeros(n_regions + 1, dtype=xp.bool_)
            anchored[labels[anchors]] = True
            if wrap_links is not None:
                labels_last, labels_first = to_host(xp.stack((labels[:, -1], labels[:, 0])))   # one transfer
                n_regions, n_floating = _join_wrapped_regions(
                    labels_last, labels_first, to_host(wrap_links), to_host(anchored), self.wrap_sign)
            else:
                n_floating = n_regions - int(xp.count_nonzero(anchored[1:]))
            if n_floating:
                raise ValueError(
                    f"{self.label()}: {n_floating} of {n_regions} conducting region(s) touch no "
                    f"{self.anchor_text}, so their {self.unknown_name} is undetermined (singular system). "
                    f"{self.remedy_text}")

    # --- solve ----------------------------------------------------------------

    def build_rhs(self, coupled_source, applied_source):
        """Write the eliminated rhs (returned, flat) and `rhs_full` from two float32 window source densities;
        with a capacity term it reads the previous unknown, so call before solve()."""
        if self.A_solve is None:
            raise RuntimeError('assemble before build_rhs')
        on_cpu = self.backend == 'cpu'
        if self.capacity_over_dt is None:
            (_source_rhs_host if on_cpu else _source_rhs)(
                coupled_source, applied_source, self.cell_area, self.free, self.rhs_offset,
                self.rhs_drive, self.rhs2d, self._rhs_full_buffer)
        else:
            (_source_rhs_capacity_host if on_cpu else _source_rhs_capacity)(
                coupled_source, applied_source, self.cell_area, self.free, self.rhs_offset,
                self.rhs_drive, self.capacity_over_dt, self.unknown2d, self.rhs2d, self._rhs_full_buffer)
        self.rhs_full = self._rhs_full_buffer
        return self.rhs

    def seed_unknown(self, values_window) -> None:
        """Warm start: write a window-shaped unknown (u - reference) on the free cells of the last assembly; fixed and
        pinned cells keep their values."""
        if self.free is None:
            raise RuntimeError('assemble before seed_unknown')
        values = self.xp.asarray(values_window, dtype=self.dtype)
        if values.shape != self.window_shape:
            raise ValueError(f'{self.label()}: seed must have the window shape {self.window_shape}, got {values.shape}')
        self.xp.copyto(self.unknown2d, values, where=self.free.astype(np.bool_))

    def solve(self) -> None:
        """Jacobi-PCG (linear_solver) on the assembled system from the current unknown; stop reference ||b|| in a
        steady solve, the step's initial residual with a capacity term (implicit Euler: PHY-1)."""
        if self.A_solve is None:
            raise RuntimeError('assemble before solve')
        solution = self._solution
        fixed_values = self.fixed_values.ravel()
        self.xp.subtract(self.unknown, fixed_values, out=solution)
        reference = 'rhs' if self.capacity_over_dt is None else 'initial_residual'
        if self._pending_anchor_check is not None:   # its read joins the CG's first sync (rhs norm) right after
            pending, self._pending_anchor_check = self._pending_anchor_check, None
            self._finish_anchor_check(*pending)
        self.linear_solver.solve(self.rhs, solution, reference)
        self.xp.add(solution, fixed_values, out=self.unknown)

    # --- post-processing --------------------------------------------------------

    def _require_assembled(self, caller):
        if self.operator_dirty or self.face_terms is None:
            raise RuntimeError(f"{self.label()}: {caller} reads the last assembly, which is missing or stale "
                               f"(operator_dirty): assemble first")

    def flux_at_centres(self):
        """(flux_x, flux_y) [c u / m] = -c grad u toward +x / +y (polar grids: +r / +angle) at the cell centres:
        axis_flux_at_centres plus the c_xy cross_flux_at_centres; 3D: (flux_x, flux_y, flux_z) [c u / m^2]."""
        axis_flux = self.axis_flux_at_centres()
        cross_flux = self.cross_flux_at_centres()
        if cross_flux is None:
            return axis_flux
        return tuple(axis + cross for axis, cross in zip(axis_flux, cross_flux))

    def cross_flux_at_centres(self):
        """(-c_xy du/dy, -c_xy du/dx) [c u / m] at the cell centres: each conducting quad's coefficient times its
        linear-u gradient, averaged over the cell's quads (equal quarters); 0 in held cells; None without c_xy."""
        self._require_assembled('cross_flux_at_centres')
        cross = self.cross
        if cross is None:
            return None
        xp = self.xp
        dtype = self.dtype
        mean_dx, mean_dy = cross.mean_differences(self.unknown2d)

        def quads_to_cells(quad_values):
            cells = xp.zeros(self.window_shape, dtype=dtype)
            for corner in _QUAD_CORNERS:
                cells[corner] += quad_values
            return cells
        quad_count = quads_to_cells(cross.conducting.astype(dtype))
        counted = (quad_count > 0) & ~self.fixed_user
        divisor = xp.where(counted, quad_count, 1.0)
        flux_x = quads_to_cells(-cross.coefficient * mean_dy / cross.spacing_y)
        flux_y = quads_to_cells(-cross.coefficient * mean_dx / cross.spacing_x)
        return xp.where(counted, flux_x / divisor, 0.0), xp.where(counted, flux_y / divisor, 0.0)

    def axis_flux_at_centres(self):
        """Axis-face part (-c_xx du/dx, -c_yy du/dy) [c u / m]: face flows over face areas averaged to the centres, 0 in
        held cells (their flux: fixed_region_outflow); 3D: (flux_x, flux_y, flux_z) [c u / m^2]."""
        self._require_assembled('axis_flux_at_centres')
        return flux_at_centres(self.unknown2d, self.face_conductances, self.face_terms, self._geometry,
                               ~self.fixed_user, self.wrap_conductance, self.wrap_sign)

    def dissipation_density(self):
        """Per-cell dissipation [c u^2 / m^2] of the last assembly: interior face g (du)^2 split by half-cell resistance
        (held-free face: all to the free cell; the phi-wrap face g (u_last - s u_first)^2 alike), an outer Dirichlet /
        Robin face's g (u_face - u_cell)^2 to its cell, an order m != 0 also azimuthal_diagonal u^2 (twice the phi mean
        of the dissipation of u_m cos(m phi), so its pi-ring integral is the total), c_xy quads by area share."""
        self._require_assembled('dissipation_density')
        if self.dimensions == 3:
            raise NotImplementedError(f"{self.label()}: dissipation_density is 2D only")
        dtype = self.dtype
        xp = self.xp
        unknown = self.unknown2d

        conductance_x, conductance_y = self.face_conductances
        if self._lower_cell_shares is None:
            self._lower_cell_shares = face_lower_shares((self.assembled_c_xx, self.assembled_c_yy), self._geometry,
                                                        self._held_cells)
        lower_share_x, lower_share_y, wrap_last_share = self._lower_cell_shares
        dissipation = xp.zeros(self.window_shape, dtype=dtype)
        face_power_x = conductance_x * (unknown[:-1, :] - unknown[1:, :]) ** 2
        dissipation[:-1, :] += lower_share_x * face_power_x
        dissipation[1:, :] += (1.0 - lower_share_x) * face_power_x
        face_power_y = conductance_y * (unknown[:, :-1] - unknown[:, 1:]) ** 2
        dissipation[:, :-1] += lower_share_y * face_power_y
        dissipation[:, 1:] += (1.0 - lower_share_y) * face_power_y
        if self.wrap_conductance is not None:
            wrap_power = self.wrap_conductance * (unknown[:, -1] - self.wrap_sign * unknown[:, 0]) ** 2
            dissipation[:, -1] += wrap_last_share * wrap_power
            dissipation[:, 0] += (1.0 - wrap_last_share) * wrap_power
        if self.azimuthal_diagonal is not None:   # order m != 0: the azimuthal gradient's c m^2 / r^2 u^2 per cell
            dissipation += self.azimuthal_diagonal * unknown ** 2
        if self.cross is not None:   # c_xy: each quad's 2 c_q (mean x step)(mean y step), the operator's cross energy
            cross = self.cross
            mean_dx, mean_dy = cross.mean_differences(unknown)
            quad_power = 2.0 * cross.coefficient * mean_dx * mean_dy
            free = ~self.fixed_user
            shares = _quad_corner_shares(cross)
            free_shares = [share * free[corner] for share, corner in zip(shares, _QUAD_CORNERS)]
            free_total = sum(free_shares)
            has_free = free_total > 0
            free_divisor = xp.where(has_free, free_total, 1.0)
            for corner, share, free_share in zip(_QUAD_CORNERS, shares, free_shares):
                # area shares of the quad; a held corner's share goes to the free corners (as a held-free face)
                dissipation[corner] += quad_power * xp.where(has_free, free_share / free_divisor, share)

        for cells, conductance, rhs in self.face_terms.values():
            conducting = conductance > 0
            dissipation[cells] += xp.where(
                conducting, (conductance * unknown[cells] - rhs) ** 2 / xp.where(conducting, conductance, 1.0), 0.0)
        return dissipation / self.geometric_cell_area   # whole-cell densities: their area integral is the total

    def fixed_region_outflow(self, mask_window) -> float:
        """Flux [c u] leaving the masked cells into the rest of the domain (positive out), minus the sources inside
        them: sum over the mask of (A_eff u - rhs_full), in float64 from the faces crossing the mask edge plus the exact
        row sums (outer faces, losses, capacity), so interior rows' rounding (~eps g_held u each) never enters."""
        if self.rhs_full is None:
            raise RuntimeError(f"{self.label()}: fixed_region_outflow needs a solve first (no rhs built yet)")
        pattern = self.pattern
        mask_flat = mask_window.ravel()
        unknown = self.unknown.astype(np.float64)
        # Off-diagonal entries A_rc = -g_rc from a masked row to an unmasked column: sum A_rc (u_c - u_r) = g (u_r - u_c).
        crossing = mask_flat[pattern.entry_rows] & ~mask_flat[pattern.indices]
        crossing_rows = pattern.entry_rows[crossing]
        crossing_columns = pattern.indices[crossing]
        crossing_coupling = self.operator_full.data[crossing].astype(np.float64)
        edge_outflow = self.xp.sum(crossing_coupling * (unknown[crossing_columns] - unknown[crossing_rows]))

        # Exact row sum of A_eff (interior, cross-term and periodic-wrap parts sum to 0): outer-face conductance + loss
        # + capacity + azimuthal diagonal, + 2 g on both ends of an antiperiodic wrap pair (diagonal g, entry +g).
        row_sum = self.extra_diag.astype(np.float64)
        if self.capacity_over_dt is not None:
            row_sum += self.capacity_over_dt
        if self.azimuthal_diagonal is not None:   # order m != 0 (0 on held cells)
            row_sum += self.azimuthal_diagonal
        for cells, conductance, _ in self.face_terms.values():
            row_sum[cells] += conductance
        if self.wrap_conductance is not None and self.wrap_sign < 0:
            wrap_row_sum = 2.0 * self.wrap_conductance.astype(np.float64)
            row_sum[:, -1] += wrap_row_sum
            row_sum[:, 0] += wrap_row_sum
        own_outflow = row_sum * unknown.reshape(self.window_shape) - self.rhs_full.astype(np.float64)
        return float(edge_outflow + own_outflow[mask_window].sum())

    def face_inward_flux(self, face) -> float:
        """Flux [c u] entering the window through one outer face (positive in), from the last assembly's terms."""
        faces = self.faces
        if face not in faces:
            raise ValueError(f"unknown face {face!r}; expected one of {faces}")
        if self.face_terms is None:
            raise RuntimeError('assemble before face_inward_flux')
        cells, conductance, rhs = self.face_terms[face]
        return float((rhs - conductance * self.unknown2d[cells]).sum())

    def flux_jump_indicator(self, flux=None):
        """S7.7 Kelly eta^2 parts (2, *window) host float64 of the last 2D assembly: normal-flux jumps of flux (default
        flux_at_centres()) over the faces it couples (g > 0, both cells free), 1 / c per cell; outer / wrap faces none."""
        self._require_assembled('flux_jump_indicator')
        if self.dimensions == 3:
            raise NotImplementedError(f"{self.label()}: the flux-jump error indicator is 2D only")
        flux_x, flux_y = self.flux_at_centres() if flux is None else flux
        metric = self.metric   # host float64 face areas and cell widths
        free = ~np.asarray(to_host(self.fixed_user), dtype=bool)
        conductance_x, conductance_y = (np.asarray(to_host(conductance)) for conductance in self.face_conductances)
        coupled_x = (conductance_x > 0) & free[:-1, :] & free[1:, :]
        coupled_y = (conductance_y > 0) & free[:, :-1] & free[:, 1:]
        return face_jump_indicator((to_host(flux_x),), (to_host(flux_y),), coupled_x, coupled_y,
                                   metric.x_length.full(), metric.y_length.full(),
                                   metric.x_face_area.full()[1:-1, :], metric.y_face_area.full()[:, 1:-1],
                                   compliance(to_host(self.assembled_c_xx)), compliance(to_host(self.assembled_c_yy)))


class DiffusionPhysicsBase(PhysicsSolver):
    """Shared wrapper of a DiffusionSystem physics (heat, potential): delegated settings and convergence record, face
    BCs, perimeter flux, probes and the PML edge extension of the unknown."""

    _probe_field = 'u'            # fields attribute a probe records
    _probe_label = 'probe'        # lower-case probe name for messages
    coordinate_grids = GRIDS      # the DiffusionSystem reads the domain's metric (subclasses may narrow it)

    def _init_diffusion(self, domain, bcs, system) -> None:
        """Common constructor state; subclasses call it before their prepare."""
        self.domain = domain
        self.bcs = bcs
        self.system = system
        self.fields = None
        self.probes = []
        # the cut-cell outline part of the applied source density (add_boundary_flux(outline=True)); rescaled by
        # 1 / ratio when insulated-wall volume fractions change cell_area, so its totals stay
        self._outline_source_window = None
        system.volume_fraction_listener = self._volume_fractions_changed

    # --- solver settings and convergence record (the DiffusionSystem holds them) ---

    @property
    def window(self):
        """(slice_x, slice_y) (3D: + slice_z) of the physical window the unknown is solved on."""
        return self.system.window

    @property
    def axisymmetric(self) -> bool:
        """True for whole-ring solves ((r, z), x = r from the window's x_lo edge; (r, theta) on a polar domain); Session
        weights field derived values by ring_angle r (model domains are Cartesian)."""
        return self.system.axisymmetric

    @property
    def azimuthal_order(self) -> int:
        """(r, z) order m (symmetry 2.14): fields, sources and boundary values are amplitudes of cos(m phi)."""
        return self.system.azimuthal_order

    @property
    def ring_angle(self) -> float:
        """Ring angle weight [rad] of axisymmetric totals and Session field derived values: 2 pi, pi at m != 0."""
        return self.system.ring_angle

    precision = _forwarded('system', 'precision')
    tol = _forwarded('system', 'tol', float, _require_positive_tolerance)
    max_iter = _forwarded('system', 'max_iter', int, _require_iteration_limit)
    compute_residual = _forwarded('system', 'compute_residual', bool)
    n_solves = _forwarded('system', 'n_solves')
    last_iters = _forwarded('system', 'last_iters')
    last_converged = _forwarded('system', 'last_converged')
    last_residual = _forwarded('system', 'last_residual', doc=(
        "||b - A x|| of the last solve relative to its stop reference (||b|| steady, the initial residual of an "
        "implicit time step); nan unless compute_residual or unconverged."))
    last_warning = _forwarded('system', 'last_warning',
                              doc="Message of the latest non-converged solve (never cleared), None before one.")
    n_unconverged = _forwarded('system', 'n_unconverged')

    # --- shared API -------------------------------------------------------------

    def set_face_bc(self, face: str, bc: BCSide) -> None:
        """Set the boundary condition of one outer window face ('x_lo', 'x_hi', 'y_lo', 'y_hi'; 3D also 'z_lo',
        'z_hi')."""
        faces = self.system.faces
        if face not in faces:
            raise ValueError(f"unknown face {face!r}; expected one of {faces}")
        if not isinstance(bc, BCSide):
            raise TypeError(f"bc must be a BCSide, got {type(bc).__name__}")
        setattr(self.bcs, face, bc)
        self.system.operator_dirty = True

    def set_face_value(self, face: str, bc: BCSide) -> None:
        """set_face_bc for a value-only change (a time-dependent feature value): the assembled rhs is updated in place
        (DiffusionSystem.update_face_value), else (no current assembly, other kind / h, 3D) the operator is dirtied."""
        faces = self.system.faces
        if face not in faces:
            raise ValueError(f"unknown face {face!r}; expected one of {faces}")
        if not isinstance(bc, BCSide):
            raise TypeError(f"bc must be a BCSide, got {type(bc).__name__}")
        previous = getattr(self.bcs, face)
        setattr(self.bcs, face, bc)
        if self.system.update_face_value(face, previous, bc):
            self._applied_source_changed()   # the rhs changed: a solve-when-dirty physics (potential) re-solves
        else:
            self.system.operator_dirty = True

    def add_boundary_flux(self, weights, value, outline=False) -> None:
        """Add an inward flux `value` through perimeter faces: applied source += value * weights [1/m] (window cells).
        outline=True: weights = drawn outline length / geometric cell area (cut cells, core.cut_cells outline_pieces),
        kept as totals when insulated-wall volume fractions change the cell areas."""
        system = self.system
        weights_window = system.window_values(weights, 'boundary flux weights', DTYPE)
        if weights_window.ndim == 0:
            raise ValueError("boundary flux weights must be a full-grid array [1/m]")
        flux = system.window_values(value, 'boundary flux value', DTYPE)
        density = flux * weights_window
        if outline:
            if system.volume_fraction is not None:
                density = density / system.xp.asarray(system.volume_fraction, dtype=DTYPE)
            if self._outline_source_window is None:
                self._outline_source_window = system.xp.zeros(system.window_shape, dtype=DTYPE)
            self._outline_source_window += density
        self._applied_source_window += density
        self._applied_source_changed()

    def _volume_fractions_changed(self, ratio) -> None:
        """DiffusionSystem hook: outline flux densities / ratio (new / old volume fractions), so their totals stay."""
        if self._outline_source_window is None:
            return
        rescaled = (self._outline_source_window / ratio).astype(DTYPE)
        self._applied_source_window += rescaled - self._outline_source_window
        self._outline_source_window[...] = rescaled
        self._applied_source_changed()

    def add_probe(self, ix: int, iy: int, iz: int = None) -> FieldProbe:
        """Record the unknown at full-grid cell (ix, iy) (3D: (ix, iy, iz)) whenever step records probes."""
        self.system.require_window_cell(ix, iy, self._probe_label, iz)
        probe = FieldProbe(ix, iy, self._probe_field, iz=iz)
        self.probes.append(probe)
        return probe

    def add_probe_at(self, x: float, y: float, z: float = None) -> FieldProbe:
        """add_probe in the cell containing physical (x, y) [m] (3D: (x, y, z))."""
        position = self._physical_position(x, y, z)
        self.domain.validate_physical_pos(*position, self._probe_label.capitalize())
        return self.add_probe(*self.domain.pos_to_cell(*position))

    def error_indicator(self):
        """S7.7 mesh refinement hook (mesh plan A2): Kelly flux-jump eta^2 parts (2, *window), host float64, of the
        shown solution (DiffusionSystem.flux_jump_indicator of _indicator_flux())."""
        return self.system.flux_jump_indicator(self._indicator_flux())

    # --- internals ------------------------------------------------------------

    def _indicator_flux(self):
        """(flux_x, flux_y) window arrays error_indicator jumps; None = the system's flux_at_centres()."""
        return None

    def _physical_position(self, x, y, z):
        """(x, y), or (x, y, z) on a 3D domain; ValueError when z is missing in 3D or given in 2D."""
        if (z is None) != (self.system.dimensions == 2):
            raise ValueError(f"a {self.system.dimensions}D domain takes "
                             f"{'x and y only' if self.system.dimensions == 2 else 'x, y and z'}")
        return (x, y) if z is None else (x, y, z)

    def _set_insulated_wall_locator(self, locator) -> None:
        """Cut cells at curved insulated walls on 2D windows (Cartesian, (r, z), polar, spherical): the system locates
        them at its next assembly from locator.material_index (model.boundary_locator, duck-typed, Domain coordinates);
        None or 3D keep the staircase."""
        system = self.system
        if locator is not None and system.dimensions == 2:
            system.set_insulated_wall_test(locator.material_index)

    def _bind_window(self, unknown_full):
        """Edge-extension copies of the full-grid unknown into the PML ring; returns the window (x_range, y_range)
        (3D: the copies of edge_extension_views_3d, returns None)."""
        if self.system.dimensions == 3:
            self._pml_copies = edge_extension_views_3d(unknown_full, self.system.window)
            return None
        window_x, window_y = self.system.window
        x_range, y_range = (window_x.start, window_x.stop), (window_y.start, window_y.stop)
        self._pml_copies = edge_extension_views(unknown_full, x_range, y_range)
        return x_range, y_range

    def _applied_source_changed(self) -> None:
        """Hook after the applied source window changed (potential marks its rhs dirty)."""

    def _extend_into_pml(self):
        xp = self.system.xp
        for destination, source in self._pml_copies:
            xp.copyto(destination, source)

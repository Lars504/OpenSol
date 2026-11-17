"""2D linear elasticity on the physical window: bilinear (Q1) elements on the window cells, nodal displacements (u_x,
u_y) on the cell corners, constrained dofs eliminated, block-Jacobi PCG. Planar (forces per unit z length; the physics
picks the plane strain / stress coefficients) or axisymmetric (x = r from the x_lo window edge, y = z; ring weights 2 pi
r, hoop strain u_r / r, whole-ring forces). Arrays on the run backend (cupy / cupyx or numpy / scipy)."""
import math

import numpy as np
from ...core.backend import array_module_of, cp, to_host
from ...core.kernels import twin

from .solver_common import ElasticSystemBase

# Local nodes of an element (offset along x, offset along y): SW, SE, NE, NW.
LOCAL_NODES = ((0, 0), (1, 0), (1, 1), (0, 1))
_SLOPE_SIGN = (-1.0, 1.0)   # sign of the derivative of the lower / upper 1D linear shape function

# Node index of each outer face on a (nx_nodes, ny_nodes[, 2]) node array.
FACE_NODES = {'x_lo': (0, slice(None)), 'x_hi': (-1, slice(None)),
              'y_lo': (slice(None), 0), 'y_hi': (slice(None), -1)}

_NEIGHBOURS = 9   # 3 x 3 neighbour nodes per dof row (x components: 9 c stencil entries, 18 in plane strain)
_CENTRE_NEIGHBOUR = 4   # (di, dj) = (0, 0) inside a row: the diagonal block starts at 4 c
# 2-point Gauss abscissae on [0, 1] (weight 1/2 each): exact for the cubic r N_k N_l of the axisymmetric mass
_GAUSS_POINTS = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))
_TWO_PI = 2.0 * math.pi


def _linear_shape(local, position):
    """Lower (local 0) / upper (local 1) 1D linear shape function at the fraction position of the element."""
    return 1.0 - position if local == 0 else position


def _block_precondition_host(residual_x, residual_y, inverse_xx, inverse_xy, inverse_yy, preconditioned_x,
                             preconditioned_y):
    """numpy twin of the kernel below (outputs written in place)."""
    preconditioned_x[...] = inverse_xx * residual_x + inverse_xy * residual_y
    preconditioned_y[...] = inverse_xy * residual_x + inverse_yy * residual_y
    return preconditioned_x, preconditioned_y


# z = B^-1 r per node with the symmetric 2x2 inverse diagonal block (inverse_xx, inverse_xy, inverse_yy): the
# precondition callback of the PCGSolver (a core.kernels twin: a change here changes the host twin above).
_block_precondition = twin('mechanics.elastic_block_jacobi', cp.ElementwiseKernel(
    'T residual_x, T residual_y, T inverse_xx, T inverse_xy, T inverse_yy',
    'T preconditioned_x, T preconditioned_y',
    'preconditioned_x = inverse_xx * residual_x + inverse_xy * residual_y;'
    'preconditioned_y = inverse_xy * residual_x + inverse_yy * residual_y;',
    'elastic_block_jacobi'), _block_precondition_host)


def _apply_inverse_blocks(inverse_blocks, residual, preconditioned):
    """preconditioned = B^-1 residual per node (flat interleaved vectors, blocks of their dtype); no allocation, so a
    captured CG block may hold it."""
    inverse_xx, inverse_xy, inverse_yy = inverse_blocks
    residual_nodes = residual.reshape(-1, 2)
    preconditioned_nodes = preconditioned.reshape(-1, 2)
    _block_precondition(residual_nodes[:, 0], residual_nodes[:, 1], inverse_xx, inverse_xy, inverse_yy,
                        preconditioned_nodes[:, 0], preconditioned_nodes[:, 1])


class _CsrApply:
    """apply(vector, out) = A vector of a backend CSR for a matrix-free PCGSolver ('mixed'): gpu the pcg CSR RawKernel
    (pcg.DeviceCsrProduct.apply, capturable in a CUDA graph, unlike a cupyx SpMV) into out; cpu scipy's in-place
    kernel."""

    def __init__(self, matrix, backend):
        from ...solvers.pcg import DeviceCsrProduct, host_csr_matvec
        self.matrix = matrix
        self._device_product = DeviceCsrProduct(matrix) if backend == 'gpu' else None
        self._host_matvec = None if backend == 'gpu' else host_csr_matvec()

    def __call__(self, vector, out):
        if self._device_product is not None:
            self._device_product.apply(vector, out)
            return
        matrix = self.matrix
        if self._host_matvec is None:
            out[...] = matrix.dot(vector)
            return
        out.fill(0)   # the kernel accumulates y += A x
        self._host_matvec(matrix.shape[0], matrix.shape[1], matrix.indptr, matrix.indices, matrix.data, vector, out)


class ElasticPattern:
    """CSR index structure of the c n_nodes square operator for one node grid (c = components, 2 in plane strain, 3 for
    the 2.5D guided modes; xp = numpy or cupy): dof = c node + component, node = i ny_nodes + j; each row holds the 3 x 3
    neighbour nodes x c components that exist (stencil layout (i, j, a, di, dj, b) row-major, so a row's entries are
    contiguous and their columns ascending)."""

    __slots__ = ('node_shape', 'components', 'stencil_positions', 'indices', 'indptr', 'entry_rows', 'block_positions')

    def __init__(self, nx_nodes, ny_nodes, components=2, xp=np):
        self.node_shape = (nx_nodes, ny_nodes)
        self.components = components
        layout = (nx_nodes, ny_nodes, components, 3, 3, components)
        row_length = _NEIGHBOURS * components
        node_i = xp.arange(nx_nodes, dtype=xp.int64)[:, None, None, None, None, None]
        node_j = xp.arange(ny_nodes, dtype=xp.int64)[None, :, None, None, None, None]
        offset_i = xp.arange(-1, 2, dtype=xp.int64)[None, None, None, :, None, None]
        offset_j = xp.arange(-1, 2, dtype=xp.int64)[None, None, None, None, :, None]
        component = xp.arange(components, dtype=xp.int64)[None, None, None, None, None, :]
        neighbour_i, neighbour_j = node_i + offset_i, node_j + offset_j
        inside = ((neighbour_i >= 0) & (neighbour_i < nx_nodes) & (neighbour_j >= 0) & (neighbour_j < ny_nodes))
        inside = xp.broadcast_to(inside, layout).ravel()   # copies the broadcast view
        columns = xp.broadcast_to(components * (neighbour_i * ny_nodes + neighbour_j) + component, layout).ravel()
        self.stencil_positions = xp.flatnonzero(inside)    # stencil slot of every CSR entry, in CSR order
        self.indices = columns[self.stencil_positions].astype(xp.int32)
        del columns
        row_counts = inside.reshape(-1, row_length).sum(axis=1)
        self.indptr = xp.zeros(row_counts.size + 1, dtype=xp.int32)
        self.indptr[1:] = xp.cumsum(row_counts)
        self.entry_rows = (self.stencil_positions // row_length).astype(xp.int32)
        # CSR slot of the diagonal c x c block entries (node, a, b): stencil slot (c node + a) 9 c + 4 c + b
        n_nodes = nx_nodes * ny_nodes
        block_slots = ((components * xp.arange(n_nodes, dtype=xp.int64)[:, None, None]
                        + xp.arange(components, dtype=xp.int64)[None, :, None]) * row_length
                       + _CENTRE_NEIGHBOUR * components + xp.arange(components, dtype=xp.int64)[None, None, :])
        self.block_positions = xp.searchsorted(self.stencil_positions, block_slots)

    def matches(self, nx_nodes, ny_nodes):
        return self.node_shape == (nx_nodes, ny_nodes)

    def csr(self, stencil, sparse_module):
        """sparse_module (cupyx.scipy.sparse / scipy.sparse) CSR of a (nx_nodes, ny_nodes, c, 3, 3, c) stencil."""
        n_dofs = self.components * self.node_shape[0] * self.node_shape[1]
        return sparse_module.csr_matrix((stencil.ravel()[self.stencil_positions], self.indices, self.indptr),
                                        shape=(n_dofs, n_dofs))


def stiffness_stencil(c11, c12, c22, c66, dx_elements, dy_elements, dtype, xp=None, c16=None, c26=None):
    """(nx_nodes, ny_nodes, 2, 3, 3, 2) stencil [N/m per m of depth] of the Q1 plane stiffness: element integrals of
    S^T C S, S = (u_x,x, u_y,y, u_x,y + u_y,x), C = [[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]] (exact, = 2x2
    Gauss) on a x b rectangles; c16 / c26 None = 0; xp None = the module of c11."""
    xp = array_module_of(c11) if xp is None else xp
    n_x, n_y = c11.shape
    stencil = xp.zeros((n_x + 1, n_y + 1, 2, 3, 3, 2), dtype=dtype)
    ratio_x = dy_elements[None, :] / (6.0 * dx_elements[:, None])   # b / (6 a)
    ratio_y = dx_elements[:, None] / (6.0 * dy_elements[None, :])   # a / (6 b)
    xx_along_x, xx_along_y = c11 * ratio_x, c66 * ratio_y   # u_x-u_x parts from u_x,x and u_x,y
    yy_along_y, yy_along_x = c22 * ratio_y, c66 * ratio_x   # u_y-u_y parts from u_y,y and u_y,x
    monoclinic = c16 is not None or c26 is not None
    if monoclinic:   # integrals of N_k,x N_l,x and N_k,y N_l,y carry the 1D mass factors as above
        c16 = 0.0 if c16 is None else c16
        c26 = 0.0 if c26 is None else c26
        xy_along_x, xy_along_y = c16 * ratio_x, c26 * ratio_y
    for k_x, k_y in LOCAL_NODES:
        target = stencil[k_x:k_x + n_x, k_y:k_y + n_y]
        for l_x, l_y in LOCAL_NODES:
            sign_x = _SLOPE_SIGN[k_x] * _SLOPE_SIGN[l_x]
            sign_y = _SLOPE_SIGN[k_y] * _SLOPE_SIGN[l_y]
            mass_x = 2.0 if k_x == l_x else 1.0   # 1D mass factor (1 + delta) of the x / y shape functions
            mass_y = 2.0 if k_y == l_y else 1.0
            di, dj = l_x - k_x + 1, l_y - k_y + 1
            # integrals of dN_k/dx dN_l/dy = s_kx s_ly / 4 and dN_k/dy dN_l/dx = s_ky s_lx / 4 (independent of a, b)
            cross_xy = 0.25 * _SLOPE_SIGN[k_x] * _SLOPE_SIGN[l_y]
            cross_yx = 0.25 * _SLOPE_SIGN[k_y] * _SLOPE_SIGN[l_x]
            target[:, :, 0, di, dj, 0] += sign_x * mass_y * xx_along_x + sign_y * mass_x * xx_along_y
            target[:, :, 1, di, dj, 1] += sign_y * mass_x * yy_along_y + sign_x * mass_y * yy_along_x
            target[:, :, 0, di, dj, 1] += cross_xy * c12 + cross_yx * c66
            target[:, :, 1, di, dj, 0] += cross_yx * c12 + cross_xy * c66
            if monoclinic:   # 2 c16 S_xx g + 2 c26 S_yy g
                target[:, :, 0, di, dj, 0] += (cross_xy + cross_yx) * c16
                target[:, :, 1, di, dj, 1] += (cross_xy + cross_yx) * c26
                target[:, :, 0, di, dj, 1] += sign_x * mass_y * xy_along_x + sign_y * mass_x * xy_along_y
                target[:, :, 1, di, dj, 0] += sign_x * mass_y * xy_along_x + sign_y * mass_x * xy_along_y
    return stencil


def gauss_stiffness_stencil(c11, c12, c22, c66, dx_elements, dy_elements, dtype, element_r_lo=None, hoop=None,
                            xp=None):
    """stiffness_stencil by 2 x 2 Gauss; element_r_lo (n_x,) inner radii [m]: the axisymmetric (r, z) stiffness [N/m,
    whole ring] with weight 2 pi r and, hoop = (c_rt, c_zt, c_tt) [Pa], the hoop strain u_r / r terms."""
    xp = array_module_of(c11) if xp is None else xp
    n_x, n_y = c11.shape
    stencil = xp.zeros((n_x + 1, n_y + 1, 2, 3, 3, 2), dtype=dtype)
    width_x, width_y = dx_elements[:, None], dy_elements[None, :]
    quarter_area = 0.25 * width_x * width_y   # Gauss weight of each of the 4 points
    for s in _GAUSS_POINTS:
        if element_r_lo is None:
            weight, inverse_r = quarter_area, None
        else:
            radius = element_r_lo[:, None] + s * width_x
            weight, inverse_r = _TWO_PI * radius * quarter_area, 1.0 / radius
        for t in _GAUSS_POINTS:
            nodes = []   # (k_x, k_y, d/dx, d/dy, N) of each local node at (s, t)
            for k_x, k_y in LOCAL_NODES:
                shape_x, shape_y = _linear_shape(k_x, s), _linear_shape(k_y, t)
                nodes.append((k_x, k_y, _SLOPE_SIGN[k_x] * shape_y / width_x, _SLOPE_SIGN[k_y] * shape_x / width_y,
                              shape_x * shape_y))
            for k_x, k_y, dk_x, dk_y, shape_k in nodes:
                target = stencil[k_x:k_x + n_x, k_y:k_y + n_y]
                for l_x, l_y, dl_x, dl_y, shape_l in nodes:
                    di, dj = l_x - k_x + 1, l_y - k_y + 1
                    xx = c11 * dk_x * dl_x + c66 * dk_y * dl_y
                    xy = c12 * dk_x * dl_y + c66 * dk_y * dl_x
                    yx = c12 * dk_y * dl_x + c66 * dk_x * dl_y
                    yy = c22 * dk_y * dl_y + c66 * dk_x * dl_x
                    if hoop is not None:   # strain u_r / r: N / r per u_r dof
                        c_rt, c_zt, c_tt = hoop
                        hoop_k, hoop_l = shape_k * inverse_r, shape_l * inverse_r
                        xx = xx + c_rt * (dk_x * hoop_l + hoop_k * dl_x) + c_tt * hoop_k * hoop_l
                        xy = xy + c_zt * hoop_k * dl_y
                        yx = yx + c_zt * dk_y * hoop_l
                    target[:, :, 0, di, dj, 0] += weight * xx
                    target[:, :, 0, di, dj, 1] += weight * xy
                    target[:, :, 1, di, dj, 0] += weight * yx
                    target[:, :, 1, di, dj, 1] += weight * yy
    return stencil


def axisymmetric_mass_stencil(density, dx_elements, dy_elements, element_r_lo, dtype, xp=None):
    """(nx_nodes, ny_nodes, 2, 3, 3, 2) consistent Q1 ring mass [kg, whole ring]: integrals of rho N_k N_l 2 pi r by 2 x
    2 Gauss (exact: cubic in r), the same on both components."""
    xp = array_module_of(density) if xp is None else xp
    n_x, n_y = density.shape
    stencil = xp.zeros((n_x + 1, n_y + 1, 2, 3, 3, 2), dtype=dtype)
    width_x = dx_elements[:, None]
    quarter_mass = 0.25 * density * width_x * dy_elements[None, :]
    for s in _GAUSS_POINTS:
        ring_mass = _TWO_PI * (element_r_lo[:, None] + s * width_x) * quarter_mass
        for t in _GAUSS_POINTS:
            for k_x, k_y in LOCAL_NODES:
                target = stencil[k_x:k_x + n_x, k_y:k_y + n_y]
                shape_k = _linear_shape(k_x, s) * _linear_shape(k_y, t)
                for l_x, l_y in LOCAL_NODES:
                    di, dj = l_x - k_x + 1, l_y - k_y + 1
                    product = ring_mass * (shape_k * _linear_shape(l_x, s) * _linear_shape(l_y, t))
                    target[:, :, 0, di, dj, 0] += product
                    target[:, :, 1, di, dj, 1] += product
    return stencil


def mass_stencil(density, dx_elements, dy_elements, dtype, xp=None, components=2):
    """(nx_nodes, ny_nodes, c, 3, 3, c) stencil [kg/m per m of depth] of the consistent Q1 mass (c = components): element
    integrals of rho N_k N_l = rho a b (1 + delta_x) (1 + delta_y) / 36, the same on every component, none between; xp
    None = the module of density."""
    xp = array_module_of(density) if xp is None else xp
    n_x, n_y = density.shape
    stencil = xp.zeros((n_x + 1, n_y + 1, components, 3, 3, components), dtype=dtype)
    element_mass = density * dx_elements[:, None] * dy_elements[None, :] / 36.0
    for k_x, k_y in LOCAL_NODES:
        target = stencil[k_x:k_x + n_x, k_y:k_y + n_y]
        for l_x, l_y in LOCAL_NODES:
            factor = (2.0 if k_x == l_x else 1.0) * (2.0 if k_y == l_y else 1.0)
            di, dj = l_x - k_x + 1, l_y - k_y + 1
            for component in range(components):
                target[:, :, component, di, dj, component] += factor * element_mass
    return stencil


def segment_node_integrals(positions, low, high, radial=False):
    """Host float64 integral of every node's 1D linear shape function over [low, high] clipped to each element of the
    node coordinates `positions`; radial: of N r dr times 2 pi (positions are radii; the ring weights)."""
    positions = np.asarray(positions, dtype=np.float64)
    left, right = positions[:-1], positions[1:]
    width = right - left
    start, stop = np.clip(float(low), left, right), np.clip(float(high), left, right)   # per-element overlap
    if radial:
        first, second = 0.5 * (stop ** 2 - start ** 2), (stop ** 3 - start ** 3) / 3.0   # integrals of r, r^2
        lower, upper = (_TWO_PI * (right * first - second) / width, _TWO_PI * (second - left * first) / width)
    else:
        lower = ((right - start) ** 2 - (right - stop) ** 2) / (2.0 * width)
        upper = ((stop - left) ** 2 - (start - left) ** 2) / (2.0 * width)
    weights = np.zeros(positions.size)
    weights[:-1] += lower
    weights[1:] += upper
    return weights


def spread_to_corners(node_component, element_values):
    """Add element_values (n_x, n_y) to each of the 4 corner nodes of every element of a (n_x+1, n_y+1) node array."""
    node_component[:-1, :-1] += element_values
    node_component[1:, :-1] += element_values
    node_component[:-1, 1:] += element_values
    node_component[1:, 1:] += element_values


class ElasticSystem(ElasticSystemBase):
    """Window system of the 2D displacement: node grid, held dofs (eliminated), rigid-mode check, block-Jacobi PCG
    (PCGSolver on the CSR) or a direct factor of A_solve (sparse_direct, made by the first solve after an assembly) and
    the reaction / gradient post-processing; the physics supplies coefficients and loads. axisymmetric (set before
    prepare): (r, z) ring weights and the hoop terms."""

    axisymmetric = False
    accepts_direct = True

    # --- allocation -----------------------------------------------------------

    def prepare(self, domain) -> None:
        """Allocate every window / node buffer on the domain's physical window (never after supports or loads were
        added: they are cleared)."""
        self.require_backend(domain)
        xp = self.xp
        nx, ny = domain.grid.nx, domain.grid.ny
        self.grid_shape = (nx, ny)
        window_x, window_y = domain.physical_window()
        x_start, x_stop, _ = window_x.indices(nx)
        y_start, y_stop, _ = window_y.indices(ny)
        self.window = (slice(x_start, x_stop), slice(y_start, y_stop))
        self.element_shape = (x_stop - x_start, y_stop - y_start)
        self.node_shape = (self.element_shape[0] + 1, self.element_shape[1] + 1)
        self.n_nodes = self.node_shape[0] * self.node_shape[1]
        self.n_dofs = 2 * self.n_nodes

        dx_cells, dy_cells = domain.cell_widths()
        self.dx_window = np.asarray(dx_cells[x_start:x_stop], dtype=np.float64)
        self.dy_window = np.asarray(dy_cells[y_start:y_stop], dtype=np.float64)
        dtype = self.dtype
        self.dx_device = xp.asarray(self.dx_window, dtype=dtype)
        self.dy_device = xp.asarray(self.dy_window, dtype=dtype)
        self.element_area = self.dx_device[:, None] * self.dy_device[None, :]
        # axisymmetric: node radii from the x_lo window edge (the axis); per element the load weights of
        # add_stress_load / add_volume_load (planar: b / 2, a / 2 and a b / 4)
        self.r_nodes = np.concatenate(([0.0], np.cumsum(self.dx_window)))
        half_x, half_y = 0.5 * self.dx_device[:, None], 0.5 * self.dy_device[None, :]
        if self.axisymmetric:
            r_lo = xp.asarray(self.r_nodes[:-1], dtype=dtype)[:, None]
            r_hi = xp.asarray(self.r_nodes[1:], dtype=dtype)[:, None]
            self.r_lo_device = r_lo[:, 0]
            self.r_centres = 0.5 * (r_lo + r_hi)                                      # (n_x, 1)
            self._half_width_y = _TWO_PI * self.r_centres * half_y                   # integral r dN/dr dz per sign
            ring_lo = _TWO_PI * self.dx_device[:, None] * (2.0 * r_lo + r_hi) / 6.0   # integral of N_lo r dr
            ring_hi = _TWO_PI * self.dx_device[:, None] * (r_lo + 2.0 * r_hi) / 6.0
            self._ring_x = (ring_lo, ring_hi)
            self._corner_volumes = (ring_lo * half_y, ring_hi * half_y)              # integral of N_k r dA, x 2 pi
        else:
            self.r_lo_device = self.r_centres = None
            self._half_width_y = half_y
            self._ring_x = (half_x, half_x)
            quarter = 0.25 * self.element_area
            self._corner_volumes = (quarter, quarter)

        node_shape = self.node_shape
        self.unknown = xp.zeros(self.n_dofs, dtype=dtype)                 # nodal (u_x, u_y) [m], interleaved
        self.unknown_nodes = self.unknown.reshape(*node_shape, 2)
        self.held = xp.zeros((*node_shape, 2), dtype=xp.bool_)           # dofs held by supports
        self.held_values = xp.zeros((*node_shape, 2), dtype=dtype)       # their prescribed displacement [m]
        self.applied_load = xp.zeros((*node_shape, 2), dtype=dtype)      # nodal forces of the load features [N/m]
        self.rhs = xp.zeros(self.n_dofs, dtype=dtype)
        self._solution = xp.zeros(self.n_dofs, dtype=dtype)              # CG unknown w = u - u_fix

        self.pattern = None
        self.operator_full = None   # assembled stiffness, rows of held dofs included
        self.load_full = None       # nodal load of the last build_rhs (flat)
        self.A_solve = None
        self.free = None
        self.fixed_values = None    # prescribed values on held / pinned dofs, 0 on free ones (flat)
        self.fixed_coupling = None  # operator_full . fixed_values
        self.inverse_blocks = None  # (inverse_xx, inverse_xy, inverse_yy) per node
        self._single_products = None   # pcg 'mixed': (float64 apply, float32 apply, float32 inverse blocks)
        self._factor_stale = False     # sparse_direct: A_solve changed since its factor (the next solve factorises)
        self._rigid_signature = None
        self.operator_dirty = True

    # --- node geometry ----------------------------------------------------------

    def face_node_lengths(self, face):
        """Tributary face length [m] of every node of an outer face (half of each adjacent segment); axisymmetric: the
        tributary ring area [m^2] (integral of N 2 pi r along the face, 0 on the axis)."""
        xp = self.xp
        x_face = face in ('x_lo', 'x_hi')
        widths = self.dy_device if x_face else self.dx_device
        lengths = xp.zeros(widths.size + 1, dtype=self.dtype)
        if self.axisymmetric and not x_face:   # a z face: annulus segments, consistent integral of N r dr
            ring_lo, ring_hi = self._ring_x
            lengths[:-1] += ring_lo[:, 0]
            lengths[1:] += ring_hi[:, 0]
            return lengths
        lengths[:-1] += 0.5 * widths
        lengths[1:] += 0.5 * widths
        if self.axisymmetric:   # an r = const face: a cylinder of radius r
            lengths *= _TWO_PI * float(self.r_nodes[0 if face == 'x_lo' else -1])
        return lengths

    def add_stress_load(self, load_nodes, stress_x, stress_y, stress_hoop=None, stress_xy=None) -> None:
        """Add the consistent nodal forces (integral of B^T sigma dV) of element-constant stresses [Pa] to a (nx_nodes,
        ny_nodes, 2) load: planar per m of depth (stress_xy: a shear stress, planar only), axisymmetric whole ring with
        the hoop stress term."""
        if stress_xy is not None:
            if self.axisymmetric:
                raise ValueError(f'{self.label()}: an axisymmetric load has no shear stress term')
            shear_x = stress_xy * (0.5 * self.dx_device[:, None])   # integral of dN/dy sigma_xy per sign (a / 2)
            shear_y = stress_xy * (0.5 * self.dy_device[None, :])   # integral of dN/dx sigma_xy per sign (b / 2)
            load_nodes[:-1, :-1, 0] -= shear_x   # u_x rows by dN/dy: south nodes -, north nodes +
            load_nodes[1:, :-1, 0] -= shear_x
            load_nodes[:-1, 1:, 0] += shear_x
            load_nodes[1:, 1:, 0] += shear_x
            load_nodes[:-1, :-1, 1] -= shear_y   # u_y rows by dN/dx: west nodes -, east nodes +
            load_nodes[:-1, 1:, 1] -= shear_y
            load_nodes[1:, :-1, 1] += shear_y
            load_nodes[1:, 1:, 1] += shear_y
        load_x, load_y = load_nodes[..., 0], load_nodes[..., 1]
        force_x = stress_x * self._half_width_y
        ring_lo, ring_hi = self._ring_x
        force_y_lo, force_y_hi = stress_y * ring_lo, stress_y * ring_hi
        load_x[:-1, :-1] -= force_x   # dN/dx: west nodes -, east nodes +
        load_x[:-1, 1:] -= force_x
        load_x[1:, :-1] += force_x
        load_x[1:, 1:] += force_x
        load_y[:-1, :-1] -= force_y_lo   # dN/dy: south nodes -, north nodes + (lower / upper x node weights)
        load_y[1:, :-1] -= force_y_hi
        load_y[:-1, 1:] += force_y_lo
        load_y[1:, 1:] += force_y_hi
        if stress_hoop is not None:   # integral of (N / r) sigma_tt 2 pi r dA = 2 pi sigma_tt a b / 4 per corner
            spread_to_corners(load_x, stress_hoop * (0.25 * _TWO_PI * self.element_area))

    def node_positions(self, along):
        """Host float64 node coordinates [m] along 'x' (= r axisymmetric) or 'y' from the window's lower-left corner
        (the physical origin)."""
        return self.r_nodes if along == 'x' else np.concatenate(([0.0], np.cumsum(self.dy_window)))

    def segment_node_lengths(self, along, low, high, radius=0.0):
        """Consistent traction weights of the span [low, high] [m] of one node line: per node the integral of its
        linear shape function over the overlap, in m; axisymmetric whole rings in m^2 (a row along 'x': of N 2 pi r
        dr; a column along 'y': times 2 pi radius)."""
        radial = self.axisymmetric and along == 'x'
        weights = segment_node_integrals(self.node_positions(along), low, high, radial=radial)
        if self.axisymmetric and along == 'y':
            weights = weights * (_TWO_PI * float(radius))
        return self.xp.asarray(weights, dtype=self.dtype)

    def add_volume_load(self, load_component, density) -> None:
        """Add the consistent nodal forces of an element-constant force density [N/m^3] to one (nx_nodes, ny_nodes)
        load component (planar a b / 4 per corner, axisymmetric the integral of N 2 pi r)."""
        volume_lo, volume_hi = self._corner_volumes
        load_component[:-1, :-1] += density * volume_lo
        load_component[:-1, 1:] += density * volume_lo
        load_component[1:, :-1] += density * volume_hi
        load_component[1:, 1:] += density * volume_hi

    def window_values(self, value, label, dtype):
        """0-d backend array for a scalar; window view cast to dtype for a full-grid array."""
        values = self.xp.asarray(value, dtype=dtype)
        if values.ndim == 0:
            return values
        if values.shape != self.grid_shape:
            raise ValueError(f"{label} must be a scalar or a full-grid {self.grid_shape} array, got shape "
                             f"{values.shape}")
        return values[self.window]

    # --- problem setup ----------------------------------------------------------

    def hold(self, dof_mask, values) -> None:
        """Hold the masked dofs ((nx_nodes, ny_nodes, 2) bool) at values [m] (0-d or the same shape); later holds win."""
        self.xp.copyto(self.held_values, self.xp.asarray(values, dtype=self.dtype), where=dof_mask)
        self.held |= dof_mask
        self.operator_dirty = True

    def assemble(self, c11, c12, c22, c66, solid_elements, hoop=None, c16=None, c26=None) -> None:
        """Assemble the stiffness from element coefficients [Pa] (window-shaped, solve dtype; axisymmetric: hoop = (c_rt,
        c_zt, c_tt) too; planar: the monoclinic c16 / c26, None = 0), eliminate held and unconnected dofs, check
        rigid-body modes per solid region, rebuild the block-Jacobi preconditioner."""
        xp, sparse = self.xp, self.sparse
        dtype = self.dtype
        nx_nodes, ny_nodes = self.node_shape
        if self.pattern is None or not self.pattern.matches(nx_nodes, ny_nodes):
            self.pattern = ElasticPattern(nx_nodes, ny_nodes, xp=xp)
        pattern = self.pattern
        if self.axisymmetric:
            if hoop is None:
                raise ValueError(f'{self.label()}: an axisymmetric assembly needs the hoop coefficients')
            stencil = gauss_stiffness_stencil(c11, c12, c22, c66, self.dx_device, self.dy_device, dtype,
                                              element_r_lo=self.r_lo_device, hoop=hoop, xp=xp)
        else:
            stencil = stiffness_stencil(c11, c12, c22, c66, self.dx_device, self.dy_device, dtype, xp=xp, c16=c16,
                                        c26=c26)
        data = stencil.ravel()[pattern.stencil_positions]
        del stencil
        n_dofs = self.n_dofs
        operator = sparse.csr_matrix((data, pattern.indices, pattern.indptr), shape=(n_dofs, n_dofs))

        diagonal_positions = xp.stack((pattern.block_positions[:, 0, 0], pattern.block_positions[:, 1, 1]),
                                      axis=1).ravel()
        held = self.held.ravel()
        unconnected = (data[diagonal_positions] <= 0) & ~held   # dofs of nodes touching no solid element
        fixed = held | unconnected
        self._check_rigid_modes(solid_elements)
        fixed_values = xp.where(held, self.held_values.ravel(), 0.0).astype(dtype, copy=False)

        # CG unknown w = u - u_fix (0 on fixed dofs): A_solve = D_free A D_free + D_fixed, rhs = D_free (f - A u_fix).
        free = (~fixed).astype(dtype)
        solve_data = data * free[pattern.entry_rows] * free[pattern.indices]
        solve_data[diagonal_positions] += fixed.astype(dtype)
        self.A_solve = sparse.csr_matrix((solve_data, pattern.indices, pattern.indptr), shape=(n_dofs, n_dofs))
        self.fixed_coupling = operator.dot(fixed_values)

        block_xx = solve_data[pattern.block_positions[:, 0, 0]]
        block_xy = solve_data[pattern.block_positions[:, 0, 1]]
        block_yy = solve_data[pattern.block_positions[:, 1, 1]]
        determinant = block_xx * block_yy - block_xy * block_xy
        invertible = determinant > 0
        safe_determinant = xp.where(invertible, determinant, 1.0)
        self.inverse_blocks = (xp.where(invertible, block_yy / safe_determinant, 1.0 / block_xx),
                               xp.where(invertible, -block_xy / safe_determinant, 0.0),
                               xp.where(invertible, block_xx / safe_determinant, 1.0 / block_yy))

        xp.copyto(self.unknown, fixed_values, where=fixed)
        self.fixed_values = fixed_values
        self.free = free
        self.operator_full = operator
        self._bind_linear_solver()
        self.operator_dirty = False

    def _bind_linear_solver(self) -> None:
        """Factorize the PCGSolver on the new A_solve with gpu check blocks captured as CUDA graphs (a factorize starts
        new graphs); 'mixed': float32 copies of A_solve and the inverse blocks run the inner CG, the float64 A_solve
        refreshes the outer residual. sparse_direct: the factor waits for the next solve (piezoelectric and modal
        solves never ask), which refreshes the kept one on its plan."""
        solver = self.linear_solver
        self._single_products = None   # free the previous float32 copies first
        if self.direct:
            self._factor_stale = True
            return
        if solver.precision != 'mixed':
            solver.factorize(self.A_solve, self._precondition, capture_blocks=True)
            return
        matrix = self.A_solve
        single = self.sparse.csr_matrix((matrix.data.astype(np.float32), matrix.indices, matrix.indptr),
                                        shape=matrix.shape)
        double_product, single_product = _CsrApply(matrix, self.backend), _CsrApply(single, self.backend)
        single_blocks = tuple(block.astype(np.float32) for block in self.inverse_blocks)
        self._single_products = (double_product, single_product, single_blocks)
        solver.factorize(double_product, n_unknowns=self.n_dofs, single_operator=single_product,
                         single_preconditioner=self._precondition_single, capture_blocks=True)

    def _check_rigid_modes(self, solid_elements):
        """Raise ValueError when a solid region (elements with stiffness, joined through shared corners) admits a
        rigid motion: no held u_x, no held u_y, or held u_x nodes on one horizontal line and held u_y nodes on one
        vertical line (rotation free); axisymmetric: no held u_z (= u_y) only (u_r strains the hoop). Skipped while the
        solid and held sets are unchanged."""
        xp = self.xp
        signature = (solid_elements.copy(), self.held.copy())
        cached = self._rigid_signature
        if (cached is not None and bool(xp.array_equal(cached[0], signature[0]))
                and bool(xp.array_equal(cached[1], signature[1]))):
            return
        labels, n_regions = self.ndimage.label(solid_elements, structure=np.ones((3, 3), dtype=bool))   # 8-connected
        n_regions = int(n_regions)
        if n_regions:
            padded = xp.zeros((labels.shape[0] + 2, labels.shape[1] + 2), dtype=labels.dtype)
            padded[1:-1, 1:-1] = labels
            node_labels = xp.maximum(xp.maximum(padded[:-1, :-1], padded[1:, :-1]),
                                     xp.maximum(padded[:-1, 1:], padded[1:, 1:]))
            node_i, node_j = xp.meshgrid(xp.arange(self.node_shape[0]), xp.arange(self.node_shape[1]),
                                         indexing='ij')
            held_x = self.held[..., 0] & (node_labels > 0)
            held_y = self.held[..., 1] & (node_labels > 0)
            # rotation is blocked by held u_x nodes at different y (rows j) or held u_y nodes at different x (columns i)
            ok_x, spread_x = _region_spread(to_host(node_labels[held_x]), to_host(node_j[held_x]), n_regions)
            ok_y, spread_y = _region_spread(to_host(node_labels[held_y]), to_host(node_i[held_y]), n_regions)
            stable = ok_y if self.axisymmetric else ok_x & ok_y & (spread_x | spread_y)
            if not stable.all():
                first = int(np.flatnonzero(~stable)[0])
                if self.axisymmetric:
                    reason = 'nothing holds u_y (the axial u_z), so it can slide along the axis'
                elif not (ok_x[first] or ok_y[first]):
                    reason = 'it has no support at all'
                elif not ok_x[first]:
                    reason = 'nothing holds u_x, so it can slide along x'
                elif not ok_y[first]:
                    reason = 'nothing holds u_y, so it can slide along y'
                else:
                    reason = ('its held u_x nodes lie on one horizontal line and its held u_y nodes on one vertical '
                              'line, so it can rotate about their crossing')
                cell_i, cell_j = (int(index) for index in to_host(xp.argwhere(labels == first + 1)[0]))
                x_cell = float(np.sum(self.dx_window[:cell_i]) + 0.5 * self.dx_window[cell_i])
                y_cell = float(np.sum(self.dy_window[:cell_j]) + 0.5 * self.dy_window[cell_j])
                raise ValueError(
                    f"{self.label()}: {int(np.count_nonzero(~stable))} of {n_regions} solid region(s) (cells with "
                    'youngs_modulus or c11 > 0, joined through shared edges or corners) can move as a rigid body, so '
                    f'the solve is singular; e.g. the region containing the cell at x = {x_cell:.6g} m, y = {y_cell:.6g} '
                    f'm (from the window corner): {reason}. Hold every solid region with Fixed, Roller or '
                    'Displacement supports on outer faces or a Fixed (domain) region, e.g. rollers on x_lo and y_lo, '
                    'or give the material around it stiffness so that it joins a held region.')
        self._rigid_signature = signature

    # --- solve ----------------------------------------------------------------

    def build_rhs(self, load_nodes) -> None:
        """Eliminated rhs from the total nodal load [N/m] ((nx_nodes, ny_nodes, 2), kept as load_full)."""
        if self.A_solve is None:
            raise RuntimeError('assemble before build_rhs')
        self.load_full = load_nodes.reshape(-1)
        self.xp.multiply(self.free, self.load_full - self.fixed_coupling, out=self.rhs)

    def _assembled(self) -> bool:
        return self.free is not None

    def _solvable(self) -> bool:
        return self.A_solve is not None

    def _precondition(self, residual, preconditioned):
        _apply_inverse_blocks(self.inverse_blocks, residual, preconditioned)

    def _precondition_single(self, residual, preconditioned):
        _apply_inverse_blocks(self._single_products[2], residual, preconditioned)

    def _factorize_if_stale(self) -> None:
        """sparse_direct: factorise A_solve of the last assembly (a kept cuDSS factor is refreshed on its plan)."""
        if self._factor_stale:
            self.linear_solver.factorize(self.A_solve)
            self._factor_stale = False

    def solve(self) -> None:
        """The base solve (PCG, or the direct factor made first when stale)."""
        if self._solvable():
            self._factorize_if_stale()
        super().solve()

    def solve_homogeneous(self, load_nodes, solution) -> None:
        """PCG of K w = f on the free dofs of the last assembly with every held dof at 0 (flat solution: the warm start
        in, w out); the generalized plane-strain unit axial-strain solve."""
        if self.A_solve is None:
            raise RuntimeError('assemble before solve_homogeneous')
        self._factorize_if_stale()
        self.xp.multiply(solution, self.free, out=solution)
        self.linear_solver.solve(self.free * load_nodes.reshape(-1), solution)

    # --- post-processing --------------------------------------------------------

    def reaction(self, dof_mask) -> tuple:
        """(F_x, F_y) [N/m; axisymmetric N, whole ring] the supports exert on the solid at the masked dofs: sum of K u - f
        of the last solve."""
        if self.load_full is None or self.n_solves == 0:
            raise RuntimeError(f'{self.label()}: reactions need a solve first')
        xp = self.xp
        operator = self.operator_full   # K u - f cancels: a single-precision product would lose digits before the sum
        if operator.dtype != np.float64:
            operator = operator.astype(np.float64)
        nodal_reaction = (operator.dot(self.unknown.astype(np.float64))
                          - self.load_full.astype(np.float64)).reshape(*self.node_shape, 2)
        masked = xp.where(xp.asarray(dof_mask), nodal_reaction, 0.0)
        totals = to_host(masked.sum(axis=(0, 1)))
        return float(totals[0]), float(totals[1])

    def element_kinematics(self, nodes=None):
        """(centre displacement (n_x, n_y, 2) [m], du/dx (n_x, n_y, 2), du/dy (n_x, n_y, 2)) at the element centres of
        the nodal displacement (default the unknown; (nx_nodes, ny_nodes, 2))."""
        nodes = self.unknown_nodes if nodes is None else nodes
        south_west, south_east = nodes[:-1, :-1], nodes[1:, :-1]
        north_east, north_west = nodes[1:, 1:], nodes[:-1, 1:]
        centre = 0.25 * (south_west + south_east + north_east + north_west)
        d_dx = ((south_east - south_west) + (north_east - north_west)) / (2.0 * self.dx_device[:, None, None])
        d_dy = ((north_west - south_west) + (north_east - south_east)) / (2.0 * self.dy_device[None, :, None])
        return centre, d_dx, d_dy


def _region_spread(region_labels, coordinates, n_regions):
    """(has_held, spread) bool arrays over regions 1..n_regions: any held dof, held dofs at more than one coordinate."""
    region_labels = np.asarray(region_labels, dtype=np.int64)
    counts = np.bincount(region_labels, minlength=n_regions + 1)[1:]
    lowest = np.full(n_regions + 1, np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.full(n_regions + 1, -1, dtype=np.int64)
    if region_labels.size:   # per label group of one stable sort (no unbuffered ufunc.at)
        order = np.argsort(region_labels, kind='stable')
        labels, ordered = region_labels[order], np.asarray(coordinates, dtype=np.int64)[order]
        starts = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))
        lowest[labels[starts]] = np.minimum.reduceat(ordered, starts)
        highest[labels[starts]] = np.maximum.reduceat(ordered, starts)
    return counts > 0, highest[1:] > lowest[1:]

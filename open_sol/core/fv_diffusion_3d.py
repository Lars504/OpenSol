"""Finite-volume diffusion operator on a 2D or 3D window from separable face factors (B3): slot-layout CSR pattern,
series face conductances and lower-cell shares, outer-face terms, the anchor link graph, face flows / centre fluxes, and
(3D) cell-drive face terms and the PML edge extension; DiffusionSystem calls these on a Domain or Domain3D.
Window arrays are (wx, wy[, wz]) in C order; units as in core.fv_diffusion (2D areas per metre of depth or whole rings,
3D [m^2]). The geometry is a core.metric.WindowMetric / WindowMetric3D (FaceGeometry on the device, built once per
window); a phi wrap (wrap_sign) joins the y_hi face of the last angle cells to the y_lo face of the first (symmetry
2.21). Array-module agnostic: cupy / cupyx on the gpu backend, numpy / scipy on the cpu (the module of the
coefficients)."""
import numpy as np

from .backend import array_module_of, backend_of, sparse_module
from .constants import FACES, FACES_3D

# Boundary-cell index of each outer face on a window-shaped array, and the face's normal axis.
FACE_CELLS = {
    'x_lo': (0, slice(None)), 'x_hi': (-1, slice(None)),
    'y_lo': (slice(None), 0), 'y_hi': (slice(None), -1),
}
FACE_CELLS_3D = {
    'x_lo': (0, slice(None), slice(None)), 'x_hi': (-1, slice(None), slice(None)),
    'y_lo': (slice(None), 0, slice(None)), 'y_hi': (slice(None), -1, slice(None)),
    'z_lo': (slice(None), slice(None), 0), 'z_hi': (slice(None), slice(None), -1),
}
FACE_AXIS_3D = {'x_lo': 0, 'x_hi': 0, 'y_lo': 1, 'y_hi': 1, 'z_lo': 2, 'z_hi': 2}
_AXIS_NAMES = 'xyz'
_LOWER, _UPPER = slice(None, -1), slice(1, None)

# Cut cells (solver_plan S7.6): a located held edge closer than this fraction of the centre spacing counts at it (the
# free cell's conductance stays <= 1 / floor x that of a whole link; Jacobi scaling absorbs the large diagonal).
HELD_WALL_FLOOR = 0.02

# Stands for the phi-wrap displacement +-(wy - 1) along y when the slot displacements are sorted.
_WRAP_STEP = 1 << 30


def face_cells(face, ndim):
    """Boundary-cell index of an outer face on a window array of ndim (2 or 3) axes."""
    return (FACE_CELLS if ndim == 2 else FACE_CELLS_3D)[face]


class _SlotLayout:
    """CSR slots of one row in column order (the neighbour displacements sorted lexicographically, as C-order flat
    columns sort): -x, [-y wrap], -y, [-z, ] diagonal, ..., +x; a 2D c_xy cross term adds the 4 corners, a phi wrap
    (wy >= 3) the wrap neighbours j = wy-1 -> j = 0 after -x and j = 0 -> j = wy-1 before +x."""

    __slots__ = ('displacements', 'n_slots', 'lower', 'upper', 'diagonal', 'corners', 'wrap_to_first', 'wrap_to_last')

    def __init__(self, ndim, wrap=False, cross=False):
        units = [tuple(int(other == axis) for other in range(ndim)) for axis in range(ndim)]
        displacements = [tuple(-step for step in unit) for unit in units] + [(0,) * ndim] + units
        corner_steps = ((-1, -1), (-1, 1), (1, -1), (1, 1)) if cross else ()
        displacements += list(corner_steps)
        wrap_steps = ()
        if wrap:
            wrap_steps = tuple((0, sign * _WRAP_STEP) + (0,) * (ndim - 2) for sign in (-1, 1))
            displacements += list(wrap_steps)
        displacements.sort()
        self.displacements = tuple(displacements)
        self.n_slots = len(displacements)
        self.lower = tuple(displacements.index(tuple(-step for step in unit)) for unit in units)
        self.upper = tuple(displacements.index(unit) for unit in units)
        self.diagonal = displacements.index((0,) * ndim)
        self.corners = {step: displacements.index(step) for step in corner_steps}
        self.wrap_to_first, self.wrap_to_last = ((displacements.index(step) for step in wrap_steps) if wrap
                                                 else (None, None))

    def column_offsets(self, shape):
        """Column offset of every slot (flat C-order index) on a window of this shape."""
        strides = [int(np.prod(shape[axis + 1:])) for axis in range(len(shape))]
        offsets = []
        for displacement in self.displacements:
            steps = [(np.sign(step) * (shape[1] - 1) if abs(step) == _WRAP_STEP else step) for step in displacement]
            offsets.append(int(sum(step * stride for step, stride in zip(steps, strides))))
        return offsets


def _cut(axis, part):
    """Index tuple taking `part` (slice or int) along axis of an array and everything along the other axes."""
    return (slice(None),) * axis + (part,)


def _scale_by(value, factors):
    """value times every broadcastable factor (a new array)."""
    for factor in factors:
        value = value * factor
    return value


def _interior(factors, axis):
    """Face-area factors of the interior faces normal to axis (the factor along axis without its two outer faces)."""
    return tuple(factor[_cut(axis, slice(1, -1))] if factor.shape[axis] > 1 else factor for factor in factors)


def _at_face(factors, axis, edge):
    """Face-area factors of the outer faces at index edge (0 or -1) along axis, that axis dropped."""
    return tuple(factor[_cut(axis, edge)] for factor in factors)


class FaceGeometry:
    """Device copies of a core.metric.WindowMetric (2D) or WindowMetric3D in one dtype: per axis the face-area factors
    (n + 1 along that axis; unit factors dropped) and the physical cell lengths (full along that axis, broadcastable
    over the window); built once per window, never per assembly."""

    __slots__ = ('area_factors', 'length', 'wrap_sign', 'shape')

    def __init__(self, metric, dtype, xp):
        def device_factors(separable, keep_axis=None):
            return tuple(xp.asarray(factor, dtype=dtype) for axis, factor in enumerate(separable.factors())
                         if axis == keep_axis or not np.all(factor == 1.0))
        self.area_factors = tuple(device_factors(area) for area in metric.face_area)
        lengths = []
        for axis, length in enumerate(metric.length):
            factors = device_factors(length, axis)
            lengths.append(_scale_by(factors[0], factors[1:]))
        self.length = tuple(lengths)
        self.wrap_sign = metric.wrap_sign
        self.shape = tuple(factor.size for factor in metric.volume.factors())

    def outer_face_rhs(self, face, value, like):
        """Neumann face rhs [c u] = value x the outer face's areas, shaped like the face's cells (`like`)."""
        axis = FACE_AXIS_3D[face]
        edge = face_cells(face, len(self.shape))[axis]
        return _scale_by(value * array_module_of(like).ones_like(like), _at_face(self.area_factors[axis], axis, edge))


class LaplacianPattern:
    """CSR index structure of the window operator (5 / 7 slots, + 4 corners of a 2D c_xy term, + 2 with a phi wrap;
    columns sorted per row); slot_positions picks the existing entries out of a dense (n_cells, n_slots) slot array,
    so assembly needs no sort. xp: numpy or cupy."""

    __slots__ = ("shape", "wrap", "cross", "layout", "indptr", "indices", "entry_rows", "slot_positions",
                 "diagonal_positions")

    def __init__(self, shape, xp=None, wrap=False, cross=False):
        from .backend import cp
        xp = cp if xp is None else xp
        shape = tuple(int(count) for count in shape)
        n_cells = int(np.prod(shape))
        layout = _SlotLayout(len(shape), wrap, cross)
        n_slots = layout.n_slots
        if n_slots * n_cells >= 2 ** 31:
            raise ValueError(f'diffusion window of {n_cells} cells exceeds the int32 CSR index range')
        if wrap and shape[1] < 3:
            raise ValueError(f"a phi wrap needs at least 3 angle cells, got {shape[1]}")
        self.shape = shape
        self.wrap = bool(wrap)
        self.cross = bool(cross)
        self.layout = layout
        ndim = len(shape)
        axis_index = [xp.arange(count, dtype=xp.int32).reshape([-1 if other == axis else 1 for other in range(ndim)])
                      for axis, count in enumerate(shape)]
        present = xp.ones(shape + (n_slots,), dtype=xp.bool_)
        for slot, displacement in enumerate(layout.displacements):
            if slot == layout.wrap_to_first:
                present[..., slot] = axis_index[1] == shape[1] - 1
                continue
            if slot == layout.wrap_to_last:
                present[..., slot] = axis_index[1] == 0
                continue
            for axis, step in enumerate(displacement):
                if step < 0:
                    present[..., slot] &= axis_index[axis] > 0
                elif step > 0:
                    present[..., slot] &= axis_index[axis] < shape[axis] - 1
        present = present.reshape(n_cells, n_slots)
        self.indptr = xp.zeros(n_cells + 1, dtype=xp.int32)
        self.indptr[1:] = xp.cumsum(present.sum(axis=1, dtype=xp.int32))
        self.diagonal_positions = self.indptr[:-1] + present[:, :layout.diagonal].sum(axis=1, dtype=xp.int32)
        self.slot_positions = xp.flatnonzero(present).astype(xp.int32)   # int32 (nnz,) into the (n_cells * n) slots
        del present
        self.entry_rows = self.slot_positions // n_slots
        column_offsets = xp.asarray(layout.column_offsets(shape), dtype=xp.int32)
        self.indices = self.entry_rows + column_offsets[self.slot_positions % n_slots]

    def matches(self, shape, wrap=False, cross=False):
        return (self.shape == tuple(int(count) for count in shape) and self.wrap == bool(wrap)
                and self.cross == bool(cross))


def _series_link(c_lower, c_upper, length_lower, length_upper, area, held_lower=None, held_upper=None, wall=None):
    """Series conductance [c] 2 c_l c_u A / (L_l c_u + L_u c_l) of a face (0 when either coefficient is 0); a
    held-free pair (bool masks) gets the free half cell 2 c_f A / L_f, or with a located wall fraction t (cut cells,
    NaN = none) c_f A / (max(t, HELD_WALL_FLOOR) d), d the centre spacing; area a tuple of factors."""
    xp = array_module_of(c_lower)
    denominator = length_lower * c_upper + length_upper * c_lower
    conductance = _scale_by(2.0 * c_lower * c_upper / xp.where(denominator > 0, denominator, 1.0), area)
    if held_lower is None:
        return conductance
    free_upper = _scale_by(2.0 * c_upper / length_upper, area)   # lower cell held
    free_lower = _scale_by(2.0 * c_lower / length_lower, area)   # upper cell held
    if wall is not None:   # the free cell's conductance to the drawn held edge (Shortley-Weller distance)
        located = ~xp.isnan(wall)
        distance = xp.maximum(xp.where(located, wall, 1.0), HELD_WALL_FLOOR) * (0.5 * (length_lower + length_upper))
        free_upper = xp.where(located, _scale_by(c_upper / distance, area), free_upper)
        free_lower = xp.where(located, _scale_by(c_lower / distance, area), free_lower)
    return xp.where(held_lower & ~held_upper, free_upper,
                    xp.where(held_upper & ~held_lower, free_lower, conductance))


def _lower_share(c_lower, c_upper, length_lower, length_upper, held_lower=None, held_upper=None):
    """Fraction of a face's series resistance in its lower half cell, L_l c_u / (L_l c_u + L_u c_l); 0 on
    non-conducting faces; a held half cell (bool masks) has none."""
    xp = array_module_of(c_lower)
    lower_resistance = length_lower * c_upper
    denominator = lower_resistance + length_upper * c_lower
    share = xp.where(denominator > 0, lower_resistance / xp.where(denominator > 0, denominator, 1.0), 0.0)
    if held_lower is not None:
        share = xp.where(held_lower & ~held_upper, 0.0, xp.where(held_upper & ~held_lower, 1.0, share))
    return share


def _axis_pairs(values, axis):
    """(lower, upper) neighbour views along axis of a window array or a cell length (full along axis); None stays
    None."""
    if values is None:
        return None, None
    return values[_cut(axis, _LOWER)], values[_cut(axis, _UPPER)]


def _wrap_pairs(values):
    """(last, first) angle-cell views (j = wy-1, j = 0) of a window array or a length factor; None stays None."""
    if values is None:
        return None, None
    return values[:, -1], values[:, 0]


def face_conductances(coefficients, geometry, held=None, held_wall=None, apertures=None):
    """Series conductances [c] of the interior faces normal to each axis (n - 1 along it; geometry a FaceGeometry);
    held a bool window mask (held-free faces: the free half cell), held_wall per-axis wall fractions (2D cut cells,
    NaN = none), apertures per-axis interior face area factors (2D cut cells at insulated walls)."""
    conductances = []
    for axis, coefficient in enumerate(coefficients):
        area = _interior(geometry.area_factors[axis], axis)
        if apertures is not None:
            area = area + (apertures[axis],)
        c_lower, c_upper = _axis_pairs(coefficient, axis)
        length_lower, length_upper = _axis_pairs(geometry.length[axis], axis)
        held_lower, held_upper = _axis_pairs(held, axis)
        conductances.append(_series_link(c_lower, c_upper, length_lower, length_upper, area, held_lower, held_upper,
                                         None if held_wall is None else held_wall[axis]))
    return tuple(conductances)


def wrap_face_conductance(c_yy, geometry, held=None):
    """Series conductance [c] ((wx,) 2D, (wx, wz) 3D) of the phi-wrap face between the last (j = wy-1) and first
    (j = 0) angle cells (the y_hi face areas); held as face_conductances."""
    area = _at_face(geometry.area_factors[1], 1, -1)
    return _series_link(*_wrap_pairs(c_yy), *_wrap_pairs(geometry.length[1]), area, *_wrap_pairs(held))


def face_lower_shares(coefficients, geometry, held=None):
    """Per axis the lower half cell's share of each interior face's series resistance (_lower_share), plus that of
    the last angle cell on the phi-wrap face (None without a wrap)."""
    shares = tuple(_lower_share(*_axis_pairs(coefficient, axis), *_axis_pairs(geometry.length[axis], axis),
                                *_axis_pairs(held, axis))
                   for axis, coefficient in enumerate(coefficients))
    wrap_share = None
    if geometry.wrap_sign:
        wrap_share = _lower_share(*_wrap_pairs(coefficients[1]), *_wrap_pairs(geometry.length[1]), *_wrap_pairs(held))
    return shares + (wrap_share,)


def boundary_face_terms(face, bc, coefficients, geometry, reference):
    """Boundary-cell index, conductance [c] and rhs [c u] (both face-shaped) of one outer face for u - reference
    (zero on faces of zero area: the r = 0 axis)."""
    xp = array_module_of(coefficients[0])
    cells = face_cells(face, len(geometry.shape))
    axis = FACE_AXIS_3D[face]
    edge = cells[axis]
    c_boundary = coefficients[axis][cells]
    area = _at_face(geometry.area_factors[axis], axis, edge)
    normal_width = geometry.length[axis][_cut(axis, edge)]
    if bc.kind == 'dirichlet':
        conductance = _scale_by(bc.conductance_factor * 2.0 * c_boundary / normal_width, area)
        rhs = conductance * (bc.value - reference)
    elif bc.kind == 'robin':
        two_c = bc.conductance_factor * 2.0 * c_boundary
        denominator = two_c + bc.h * normal_width
        conductance = _scale_by(bc.h * two_c / xp.where(denominator > 0, denominator, 1.0), area)
        rhs = conductance * (bc.external - reference)
    else:  # neumann: value is an inward flux [c u / m]
        conductance = xp.zeros_like(c_boundary)
        rhs = geometry.outer_face_rhs(face, bc.value, c_boundary)
    return cells, conductance, rhs


def build_laplacian(coefficients, geometry, bcs, pattern=None, dtype=np.float64, reference=0.0, held=None,
                    held_wall=None, apertures=None, cross=None):
    """FV operator A [c], boundary rhs [c u], pattern, interior face conductances per axis, {face:
    boundary_face_terms} over the window's outer faces and the phi-wrap conductance (None without a wrap) for u -
    reference; coefficients (c_xx, c_yy[, c_zz]) window-shaped backend arrays (their module sets the backend),
    geometry the window's FaceGeometry (in dtype), held / held_wall / apertures as face_conductances, cross a 2D
    CrossQuads (core.fv_diffusion: c_xy corners) or None. A wrap (geometry.wrap_sign s) couples the last and first
    angle cells by -s g; y_lo / y_hi then take no condition (insulated only)."""
    xp = array_module_of(coefficients[0])
    shape = tuple(coefficients[0].shape)
    ndim = len(shape)
    if tuple(geometry.shape) != shape:
        raise ValueError(f'metric of {geometry.shape} cells does not match coefficient arrays {shape}')
    coefficients = tuple(xp.asarray(coefficient, dtype=dtype) for coefficient in coefficients)
    wrap_sign = geometry.wrap_sign
    if wrap_sign:
        for face in ('y_lo', 'y_hi'):
            bc = getattr(bcs, face)
            if bc.kind != 'neumann' or bc.value != 0.0:
                raise ValueError(f"{face} of a window with a phi wrap is joined to the opposite face and takes no "
                                 f"boundary condition, got {bc.kind} (value {bc.value:g})")
    if pattern is None or not pattern.matches(shape, bool(wrap_sign), cross is not None):
        pattern = LaplacianPattern(shape, xp, wrap=bool(wrap_sign), cross=cross is not None)
    layout = pattern.layout

    conductances = face_conductances(coefficients, geometry, held, held_wall, apertures)
    slots = xp.zeros(shape + (layout.n_slots,), dtype=dtype)
    diagonal = slots[..., layout.diagonal]   # view: the in-place updates below write into slots
    for axis, conductance in enumerate(conductances):
        lower, upper = _cut(axis, _LOWER), _cut(axis, _UPPER)
        diagonal[lower] += conductance
        diagonal[upper] += conductance
        slots[upper + (Ellipsis, layout.lower[axis])] = -conductance   # upper cell -> its lower neighbour
        slots[lower + (Ellipsis, layout.upper[axis])] = -conductance   # lower cell -> its upper neighbour
    wrap_conductance = None
    if wrap_sign:   # g (u_last - s u_first)^2: +g on both diagonals, -s g on the pair
        wrap_conductance = wrap_face_conductance(coefficients[1], geometry, held)
        diagonal[:, -1] += wrap_conductance
        diagonal[:, 0] += wrap_conductance
        slots[:, -1, ..., layout.wrap_to_first] = -float(wrap_sign) * wrap_conductance
        slots[:, 0, ..., layout.wrap_to_last] = -float(wrap_sign) * wrap_conductance

    rhs_bc = xp.zeros(shape, dtype=dtype)
    face_terms = {}
    for face in (FACES if ndim == 2 else FACES_3D):
        cells, conductance, rhs = boundary_face_terms(face, getattr(bcs, face), coefficients, geometry, reference)
        face_terms[face] = (cells, conductance, rhs)
        diagonal[cells] += conductance
        rhs_bc[cells] += rhs

    if cross is not None:
        # Cross term 2 c_xy d2u/dxdy per interior vertex (core.fv_diffusion.CrossQuads): quad energy c_q/2 [(u_NE -
        # u_SW)^2 - (u_NW - u_SE)^2]; NE / SW pairs -c_q/2, NW / SE pairs +c_q/2, every row sums to 0.
        half = 0.5 * cross.coefficient
        diagonal[:-1, :-1] += half   # SW
        diagonal[1:, 1:] += half     # NE
        diagonal[:-1, 1:] -= half    # NW
        diagonal[1:, :-1] -= half    # SE
        corners = layout.corners
        slots[:-1, :-1, corners[(1, 1)]] = -half   # (i, j) -> (i+1, j+1)
        slots[1:, 1:, corners[(-1, -1)]] = -half
        slots[:-1, 1:, corners[(1, -1)]] = half    # (i, j+1) -> (i+1, j)
        slots[1:, :-1, corners[(-1, 1)]] = half

    data = slots.reshape(-1)[pattern.slot_positions]
    del slots, diagonal
    n_cells = int(np.prod(shape))
    operator = sparse_module(backend_of(data)).csr_matrix((data, pattern.indices, pattern.indptr),
                                                          shape=(n_cells, n_cells))
    return operator, rhs_bc, pattern, conductances, face_terms, wrap_conductance


def anchor_link_graph(coefficients, fixed, held, face_terms, wrap=False):
    """(link image, conducting, anchors, wrap links) of the anchor check: link image at 2x resolution (cells on even
    pixels / voxels, non-zero face links between them; a phi wrap adds a last y layer of wrap links, labelled with the
    last angle cells), cells with a link, and the anchors from fixed cells, held neighbours and outer faces with
    conductance > 0 (the caller adds losses / capacity and masks with conducting); wrap links or None."""
    xp = array_module_of(fixed)
    shape = fixed.shape
    conducting = xp.zeros(shape, dtype=xp.bool_)
    image_shape = [2 * count - 1 for count in shape]
    if wrap:
        image_shape[1] += 1
    link_image = xp.zeros(tuple(image_shape), dtype=xp.bool_)
    cell_voxels = tuple(slice(0, 2 * count - 1, 2) for count in shape)
    anchors = fixed.copy()
    for axis, coefficient in enumerate(coefficients):
        lower, upper = _cut(axis, _LOWER), _cut(axis, _UPPER)
        conducts = coefficient > 0
        links = conducts[lower] & conducts[upper]
        conducting[lower] |= links
        conducting[upper] |= links
        link_voxels = list(cell_voxels)
        link_voxels[axis] = slice(1, 2 * shape[axis] - 1, 2)
        link_image[tuple(link_voxels)] = links
        anchors[lower] |= held[upper] & conducts[lower]
        anchors[upper] |= held[lower] & conducts[upper]
    wrap_links = None
    if wrap:
        conducts_y = coefficients[1] > 0
        wrap_links = conducts_y[:, -1] & conducts_y[:, 0]
        conducting[:, -1] |= wrap_links
        conducting[:, 0] |= wrap_links
        link_image[(cell_voxels[0], -1) + cell_voxels[2:]] = wrap_links
        anchors[:, -1] |= held[:, 0] & conducts_y[:, -1]
        anchors[:, 0] |= held[:, -1] & conducts_y[:, 0]
    link_image[cell_voxels] = conducting
    for cells, conductance, _ in face_terms.values():
        anchors[cells] |= conductance > 0
    return link_image, conducting, anchors, wrap_links


def face_flows_3d(unknown, interior_conductances, face_terms, wrap_conductance=None, wrap_sign=0):
    """Per axis the face flows [c u] toward +x / +y (/ +z) (n + 1 along that axis; 2D or 3D windows) of the last
    assembly's operator: interior conductance * (u_lower - u_upper), outer faces from their face terms (a phi wrap:
    out of the last angle cells, into the first as its sign image)."""
    xp = array_module_of(unknown)
    flows = []
    for axis, conductance in enumerate(interior_conductances):
        face_shape = list(unknown.shape)
        face_shape[axis] += 1
        face_flow = xp.zeros(face_shape, dtype=unknown.dtype)
        face_flow[_cut(axis, slice(1, -1))] = conductance * (unknown[_cut(axis, _LOWER)] - unknown[_cut(axis, _UPPER)])
        for side, edge, sign in (('lo', 0, 1.0), ('hi', -1, -1.0)):
            cells, boundary_conductance, rhs = face_terms[f'{_AXIS_NAMES[axis]}_{side}']
            face_flow[_cut(axis, edge)] = sign * (rhs - boundary_conductance * unknown[cells])   # inward -> +axis
        if axis == 1 and wrap_conductance is not None:
            wrap_flow = wrap_conductance * (unknown[:, -1] - wrap_sign * unknown[:, 0])
            face_flow[:, -1] = wrap_flow
            face_flow[:, 0] = wrap_sign * wrap_flow
        flows.append(face_flow)
    return tuple(flows)


def flux_at_centres(unknown, interior_conductances, face_terms, geometry, free, wrap_conductance=None, wrap_sign=0):
    """Per axis the flux [c u / m^2] (2D planar: [c u / m]) toward +x / +y (/ +z) (polar: +r / +phi) at the cell
    centres: face_flows_3d of the last assembly over the face areas (geometry without apertures), averaged; 0 where
    free is False (held cells)."""
    xp = array_module_of(unknown)
    fluxes = []
    flows = face_flows_3d(unknown, interior_conductances, face_terms, wrap_conductance, wrap_sign)
    for axis, face_flow in enumerate(flows):
        for factor in geometry.area_factors[axis]:   # zero-area faces (the r = 0 axis) carry no flow
            face_flow /= xp.where(factor > 0, factor, 1.0)
        fluxes.append(xp.where(free, 0.5 * (face_flow[_cut(axis, _LOWER)] + face_flow[_cut(axis, _UPPER)]), 0.0))
    return tuple(fluxes)


def face_flow_energy_3d(flows, interior_conductances, face_terms, free):
    """Sum of flow^2 / conductance [c u^2] over the faces touching a free cell with conductance > 0 (flows as
    face_flows_3d, a drive's terms included): twice the discrete energy of those flows. Float64 host scalar."""
    xp = array_module_of(free)
    total = 0.0
    for axis, flow in enumerate(flows):
        conductance = xp.zeros(flow.shape, dtype=interior_conductances[axis].dtype)
        touches_free = xp.zeros(flow.shape, dtype=bool)
        interior = _cut(axis, slice(1, -1))
        conductance[interior] = interior_conductances[axis]
        touches_free[interior] = free[_cut(axis, _LOWER)] | free[_cut(axis, _UPPER)]
        for side, edge in (('lo', 0), ('hi', -1)):
            cells, boundary_conductance, _ = face_terms[f'{_AXIS_NAMES[axis]}_{side}']
            conductance[_cut(axis, edge)] = boundary_conductance
            touches_free[_cut(axis, edge)] = free[cells]
        counted = touches_free & (conductance > 0)
        total += float(xp.sum(xp.where(counted, flow * flow / xp.where(counted, conductance, 1.0), 0.0),
                              dtype=xp.float64))
    return total


def face_drive_integrals_3d(drive, axis, geometry, free):
    """Integral [u] of a cell-centred drive component w (0-d or window array, along axis) across each face normal to
    axis (n + 1 along it): w L / 2 of both neighbour cells, of the boundary cell at an outer face; held cells (free
    False) add none."""
    xp = array_module_of(free)
    half = xp.where(free, 0.5 * drive * geometry.length[axis], 0.0)
    face_shape = list(half.shape)
    face_shape[axis] += 1
    integral = xp.zeros(face_shape, dtype=half.dtype)
    integral[_cut(axis, slice(1, -1))] = half[_cut(axis, _LOWER)] + half[_cut(axis, _UPPER)]
    integral[_cut(axis, 0)] = half[_cut(axis, 0)]
    integral[_cut(axis, -1)] = half[_cut(axis, -1)]
    return integral


def face_drive_terms_3d(drives, interior_conductances, face_terms, geometry, free):
    """Face terms [c u] toward +x / +y / +z ((wx+1, wy, wz), (wx, wy+1, wz), (wx, wy, wz+1)) of a cell-centred drive
    (w_x, w_y, w_z) [u / m] (0-d or window arrays) whose physical flux is -c (grad u - w): flux * face area =
    conductance * (u_lower - u_upper) + term, term = conductance * face_drive_integrals_3d. Held cells (free False)
    carry none; outer faces only through their face term conductance (Neumann: none). Conductances and face terms of
    the last assembly; geometry its FaceGeometry."""
    xp = array_module_of(interior_conductances[0])
    dtype = interior_conductances[0].dtype
    terms = []
    for axis in range(3):
        integral = face_drive_integrals_3d(drives[axis], axis, geometry, free).astype(dtype, copy=False)
        term = xp.empty_like(integral)
        interior = _cut(axis, slice(1, -1))
        term[interior] = interior_conductances[axis] * integral[interior]
        for side, edge in (('lo', 0), ('hi', -1)):
            _cells, conductance, _ = face_terms[f'{_AXIS_NAMES[axis]}_{side}']
            term[_cut(axis, edge)] = conductance * integral[_cut(axis, edge)]
        terms.append(term)
    return tuple(terms)


def centre_gradient_3d(values, axis, geometry):
    """d values / d axis [values / m] at the cell centres of a window array (>= 2 cells along axis): differences over
    the centre spacing on the interior faces normal to axis, an outer face copies its interior neighbour (one-sided in
    the boundary cells), the two faces of each cell averaged (Cartesian lengths)."""
    xp = array_module_of(values)
    length = geometry.length[axis]
    face_shape = list(values.shape)
    face_shape[axis] += 1
    gradient = xp.zeros(face_shape, dtype=values.dtype)
    spacing = 0.5 * (length[_cut(axis, _LOWER)] + length[_cut(axis, _UPPER)])
    gradient[_cut(axis, slice(1, -1))] = (values[_cut(axis, _UPPER)] - values[_cut(axis, _LOWER)]) / spacing
    gradient[_cut(axis, slice(0, 1))] = gradient[_cut(axis, slice(1, 2))]
    gradient[_cut(axis, slice(-1, None))] = gradient[_cut(axis, slice(-2, -1))]
    return 0.5 * (gradient[_cut(axis, _LOWER)] + gradient[_cut(axis, _UPPER)])


def face_drive_divergence_3d(terms):
    """Window rhs [c u] of face_drive_terms_3d: minus the net outflow of the terms per cell."""
    return -sum(term[_cut(axis, _UPPER)] - term[_cut(axis, _LOWER)] for axis, term in enumerate(terms))


def face_terms_at_centres_3d(terms, geometry, free):
    """(x, y, z) centre averages [c u / m^2] of per-face terms [c u] (face_drive_terms_3d) over their face areas, 0
    where free is False: the drive part of the physical flux, as flux_at_centres is the operator part."""
    xp = array_module_of(terms[0])
    centres = []
    for axis, term in enumerate(terms):
        for factor in geometry.area_factors[axis]:
            term = term / xp.where(factor > 0, factor, 1.0)
        centres.append(xp.where(free, 0.5 * (term[_cut(axis, _LOWER)] + term[_cut(axis, _UPPER)]), 0.0))
    return tuple(centres)


def edge_extension_views_3d(array, window):
    """(destination, source) view pairs copying the outermost window planes (2D: rows / columns) of a full-grid 2D or
    3D array into the PML ring / shell, axis by axis (applied in order, later axes span the already extended earlier
    ones: edges and corners); window = one slice per axis."""
    copies = []
    for axis in range(array.ndim):
        index = [slice(None)] * axis + [None] + [window[later] for later in range(axis + 1, array.ndim)]
        start, stop = window[axis].start, window[axis].stop

        def view(part, index=index, axis=axis):
            index = list(index)
            index[axis] = part
            return array[tuple(index)]

        if start > 0:
            copies.append((view(slice(None, start)), view(slice(start, start + 1))))
        if stop < array.shape[axis]:
            copies.append((view(slice(stop, None)), view(slice(stop - 1, stop))))
    return copies

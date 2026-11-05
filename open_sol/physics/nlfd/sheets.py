"""Nonlinear sheets of the NLFD-chi solver (nonlinear_optics_plan 3.3 / 3.4, stage 4): surface chi(2) on the interface
faces of a domain selection and 2D-material sheets (chi_s(2), sigma(3)) on an axis-aligned line, as sites that sample
the channel grid fields and deposit their surface polarisation back by the exact adjoint (numpy at module level)."""
# Site s samples a model-frame vector E_s = (E_x, E_y, E_z) from a channel's grid fields (ez: E_z at the cell centres;
# hz: E_x on hi-y faces, E_y on hi-x faces) through two taps per component, E_s,d = sum_t a_t E_d[k_t]. Its surface
# polarisation P_s [C/m] (site length L_s = the interface / line length it stands for) is deposited as the grid density
# P_d[k_t] += L_s a_t P_s,d / A_k (A_k = cell area at a centre; Yee width x dual spacing at a face), so sum_k A_k
# conj(E_k) P_k = sum_s L_s conj(E_s) . P_s: the site work W = (omega / 2) sum_s L_s Im(conj(E_s) . P_s) is the grid
# work and sum_k W_k = 0 for fully permutation-symmetric real site tensors (as the bulk C / C*). ez densities enter the
# rhs as -omega^2 mu0 P at the centres, hz densities as face currents -j omega P directly (they already sit on faces).
# Surface sites = faces between a selected cell and an unselected one (or one of interface_with): an x-normal face
# samples its tangential E_y on the face itself (exact), E_z as the mean of its two cells and the normal E_x as the
# face mean C at the centre of the selvedge cell (inside or outside); y-normal faces alike with x <-> y. Staircase:
# n = +-e_axis from the inside cell outward, L = the face length. Smoothed: n = the drawn boundary's normal where it
# crosses the segment between the face's two cell centres (bisection of a point test; the mean angle of the two
# crossings of a small circle around that point: exact for lines and circles), L = face length x |n_across| (the face
# projected on the boundary: x faces per unit length |n_x| / h, so sum L = the curve length up to O(h^1.5) end terms;
# sites stay on the faces). Not from cell fill fractions: their gradient errs by a mesh-independent few degrees and
# percent of length, which keeps a curved surface's SH power from converging.
# Line sites (2D materials) = the line cells: E_z at their centres and the in-plane tangent E on their hi faces (the
# conducting_sheet / H_z line-source sites), L = the cell width along the line; the normal E does not enter.
# sheet_site_absorption books the site work per tap in the tap's cell (the em_heat_joule density of sheet SPM / XPM);
# sheet_cotangents is the transpose of deposit o site polynomial o sampling (the adjoint of nlfd/adjoint.py).
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import EPS0
from .mixing import MediumTensor
from .tensors import chi3_isotropic, rotate_tensor, symmetrize_sources

SURFACE_COEFFICIENTS = ('chi_s_nnn', 'chi_s_ntt', 'chi_s_tnt')
SHEET_CHI2_COEFFICIENTS = {'D3h': ('chi_s',),
                           'general': ('chi_s_ttt', 'chi_s_tzz', 'chi_s_ttz', 'chi_s_ztt', 'chi_s_zzz', 'chi_s_zzt')}
_GENERAL_ENTRIES = {'chi_s_ttt': ((0, 0, 0),), 'chi_s_tzz': ((0, 1, 1),), 'chi_s_ttz': ((0, 0, 1), (0, 1, 0)),
                    'chi_s_ztt': ((1, 0, 0),), 'chi_s_zzz': ((1, 1, 1),), 'chi_s_zzt': ((1, 1, 0), (1, 0, 1))}
SELVEDGES = ('inside', 'outside')
SURFACE_INTERFACES = ('staircase', 'smoothed')
RESPONSES = ('polarization', 'current')
_REPRESENTATIVE_ANGLES = tuple(math.radians(15.0 * index) for index in range(24))
_BISECTION_STEPS = 40          # smoothed normals: crossings to 2^-40 of their bracket (round-off level)
_CHORD_RADIUS = 0.25           # circle around the crossing, as a fraction of the two cell centres' distance


# ---- tensors (host numpy) ----
def surface_local_tensor(chi_nnn, chi_ntt, chi_tnt):
    """(3, 3, 3) isotropic-surface (C_inf_v) chi_s [m^2/V] in the local (n, t, z) frame: nnn; ntt = nzz; tnt = ttn
    = znz = zzn (n outward normal, t = z x n)."""
    local = np.zeros((3, 3, 3))
    local[0, 0, 0] = chi_nnn
    local[0, 1, 1] = local[0, 2, 2] = chi_ntt
    local[1, 0, 1] = local[1, 1, 0] = local[2, 0, 2] = local[2, 2, 0] = chi_tnt
    return local


def surface_frames(normal_x, normal_y):
    """(n, 3, 3) rotations local (n, t, z) -> model axes: columns e_n = (n_x, n_y, 0), e_t = (-n_y, n_x, 0), e_z."""
    normal_x = np.atleast_1d(np.asarray(normal_x, dtype=np.float64))
    normal_y = np.atleast_1d(np.asarray(normal_y, dtype=np.float64))
    frames = np.zeros((normal_x.size, 3, 3))
    frames[:, 0, 0], frames[:, 1, 0] = normal_x, normal_y
    frames[:, 0, 1], frames[:, 1, 1] = -normal_y, normal_x
    frames[:, 2, 2] = 1.0
    return frames


def rotate_sites(local, frames):
    """(n,) + local.shape model-axis tensors T_ij.. = R_ia R_jb .. local_ab.. per frame R (n, 3, 3)."""
    local = np.asarray(local)
    rotated = np.array(np.broadcast_to(local, (frames.shape[0],) + local.shape))
    for axis in range(1, rotated.ndim):
        moved = np.moveaxis(rotated, axis, -1)
        rotated = np.moveaxis(np.einsum('s...a,sia->s...i', moved, frames), -1, axis)
    return rotated


def surface_mixing_tensor(local):
    """(3, 3, 3) max |T(n)| over in-plane normals every 15 deg: the entry pattern a surface sheet gives the mixing
    table (validation and build alike)."""
    angles = np.asarray(_REPRESENTATIVE_ANGLES)
    return np.abs(rotate_sites(local, surface_frames(np.cos(angles), np.sin(angles)))).max(axis=0)


def line_frame(axis):
    """(3, 3) sheet frame (t, z, normal) -> model axes of a line along 'x' (t = x, normal y) or 'y' (t = y, normal x)."""
    tangent, normal = (0, 1) if axis == 'x' else (1, 0)
    frame = np.zeros((3, 3))
    frame[tangent, 0] = frame[2, 1] = frame[normal, 2] = 1.0
    return frame


def sheet_chi2_local(crystal_class, coefficients, angle=0.0):
    """(3, 3, 3) sheet chi_s(2) [m^2/V] in the sheet frame (t, z, normal; normal entries 0). D3h: chi_aaa = chi_s,
    chi_abb = chi_bab = chi_bba = -chi_s with the armchair axis a at `angle` [deg] from t toward z; general: the
    chi_s_<ijk> of (t, z) directly (ttz = tzt, zzt = ztz)."""
    local = np.zeros((3, 3, 3))
    if crystal_class == 'D3h':
        chi = float(coefficients['chi_s'])
        local[0, 0, 0] = chi
        local[0, 1, 1] = local[1, 0, 1] = local[1, 1, 0] = -chi
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return rotate_tensor(local, np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]))
    if crystal_class != 'general':
        raise ValueError(f"sheet crystal_class must be one of {', '.join(SHEET_CHI2_COEFFICIENTS)}, got "
                         f"{crystal_class!r}")
    for name, entries in _GENERAL_ENTRIES.items():
        for entry in entries:
            local[entry] = float(coefficients[name])
    return local


def sheet_sigma3_local(sigma):
    """(3, 3, 3, 3) in-plane isotropic sheet sigma(3) [S m^2/V^2] in the sheet frame: sigma_tttt = sigma_zzzz = sigma,
    sigma_ttzz = sigma_tztz = sigma_tzzt = sigma / 3 (t <-> z alike), normal entries 0."""
    local = np.array(chi3_isotropic(complex(sigma)), dtype=np.complex128)
    for axis in range(4):
        index = [slice(None)] * 4
        index[axis] = 2
        local[tuple(index)] = 0.0
    return local


def line_mixing_tensor(local):
    """max |T| of a sheet-frame tensor on a line along x and along y: the mixing-table entry pattern of a line sheet."""
    return np.maximum(np.abs(rotate_tensor(local, line_frame('x'))), np.abs(rotate_tensor(local, line_frame('y'))))


# ---- sites ----
@dataclass(frozen=True, eq=False)
class SheetMedium:
    """Nonlinear sheet: per-site model-axis tensors (source indices symmetrised) on sites sampling the grid fields
    (module header). response 'polarization': P_s = eps0 T : E..E [C/m]; 'current': K = T : E..E [A/m], P_s = K / (j
    omega_target)."""
    name: str
    order: int
    tensor: np.ndarray       # host representative (max |T| over orientations): the mixing-table tensor
    site_tensor: object       # xp complex128 (n_sites,) + (3,) * (order + 1)
    site_length: object       # xp float64 (n_sites,) L_s [m]
    taps: tuple               # per model component 0, 1, 2: None or xp (index (n, 2) int64 flat, sample (n, 2), deposit
                              # (n, 2) = L_s a / A_k) float64
    normals: np.ndarray       # host (n_sites, 2) in-plane outward normal (surface) or line normal (line)
    positions: np.ndarray     # host (n_sites, 2) face midpoint / line-cell centre [m] from the first grid edge
    response: str = 'polarization'
    process_scales: tuple = ()

    @property
    def n_sites(self) -> int:
        return int(self.normals.shape[0])

    @property
    def total_length(self) -> float:
        return float(np.sum(np.asarray(self.site_length.get() if hasattr(self.site_length, 'get')
                                       else self.site_length)))

    def mixing_tensor(self):
        """The MediumTensor the mixing table enumerates (CPU)."""
        return MediumTensor(name=self.name, order=self.order, tensor=self.tensor, process_scales=self.process_scales)


def _dual_widths(widths):
    """Face spacings (w_k + w_(k+1) mod n) / 2 of yee_inverse_spacings (the last entry the periodic wrap)."""
    return 0.5 * (widths + np.roll(widths, -1))


def _neighbour(index, offset, count, periodic):
    shifted = index + offset
    return np.mod(shifted, count) if periodic else np.clip(shifted, 0, count - 1)


def _bisect(is_inside, inside_end, outside_end):
    """Per-site crossing parameter of is_inside(parameters) -> bool between brackets True at inside_end, False at
    outside_end."""
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (inside_end + outside_end)
        hit = is_inside(middle)
        inside_end, outside_end = np.where(hit, middle, inside_end), np.where(hit, outside_end, middle)
    return 0.5 * (inside_end + outside_end)


def _smoothed_normal(boundary_test, axis, inner, outer, along):
    """(n_across, n_along) outward normal of the drawn boundary at its crossing of the segments inner -> outer (across
    coordinates of the inside / outside cell centres on rows at along coordinates); the staircase normal where a
    bracket fails (the test disagrees with the mask, or a feature thinner than the circle)."""
    if inner.size == 0:
        return np.zeros(0), np.zeros(0)

    def test(across, row):
        return np.asarray(boundary_test(*((across, row) if axis == 0 else (row, across))), dtype=bool)
    count = inner.size
    step = outer - inner
    sign = np.sign(step)
    crossing = inner + step * _bisect(lambda fraction: test(inner + fraction * step, along), np.zeros(count),
                                      np.ones(count))
    radius = _CHORD_RADIUS * np.abs(step)

    def on_circle(angle):   # angle from the outward axis toward +along; angle holds whole copies of the site set
        copies = angle.size // count
        return test(np.tile(crossing, copies) + np.tile(sign * radius, copies) * np.cos(angle),
                    np.tile(along, copies) + np.tile(radius, copies) * np.sin(angle))
    valid = (test(inner, along) & ~test(outer, along) & ~on_circle(np.zeros(count))
             & on_circle(np.full(count, math.pi)))
    # one crossing in (0, pi), one in (-pi, 0); their mean angle is the normal (a line: crossings at beta +- pi / 2;
    # a circle: the two are mirror images about its radius)
    angles = _bisect(on_circle, np.concatenate((np.full(count, math.pi), np.full(count, -math.pi))),
                     np.zeros(2 * count))
    normal_angle = 0.5 * (angles[:count] + angles[count:])
    return np.where(valid, sign * np.cos(normal_angle), sign), np.where(valid, np.sin(normal_angle), 0.0)


def _surface_axis_sites(axis, inside, outside, widths, periodic, selvedge, boundary_test, origin):
    """Host sites of the faces normal to model axis `axis` (0: x-normal faces): dict with normals (n, 2), positions
    (n, 2), length (n,) and taps {component: (index (n, 2), sample (n, 2), grid-site area (n, 2))}."""
    shape = inside.shape
    transpose = (lambda array: array) if axis == 0 else (lambda array: array.T)
    inside_t, outside_t = transpose(inside), transpose(outside)
    across_widths, along_widths = widths[axis], widths[1 - axis]
    across_periodic, along_periodic = periodic[axis], periodic[1 - axis]
    count_across, count_along = inside_t.shape
    lower_faces = np.arange(count_across if across_periodic else count_across - 1)
    upper_faces = (lower_faces + 1) % count_across
    inside_lower = inside_t[lower_faces] & outside_t[upper_faces]
    inside_upper = inside_t[upper_faces] & outside_t[lower_faces]
    face, along = np.nonzero(inside_lower | inside_upper)
    lower, upper = lower_faces[face], upper_faces[face]
    sign = np.where(inside_lower[face, along], 1.0, -1.0)   # outward: +axis when the lower cell is inside
    inner, outer = np.where(sign > 0.0, lower, upper), np.where(sign > 0.0, upper, lower)
    selvedge_cell = inner if selvedge == 'inside' else outer
    along_lower = _neighbour(along, -1, count_along, along_periodic)
    if not along_periodic:
        along_lower = np.where(along == 0, along, along_lower)   # cell 0 pairs with its own hi face (C)

    def flat(across_index, along_index):
        return (across_index * shape[1] + along_index if axis == 0 else along_index * shape[1] + across_index)

    across_dual, along_dual = _dual_widths(across_widths), _dual_widths(along_widths)
    tangent, normal = 1 - axis, axis   # model components: the face's own E (tangent in-plane) and the normal one
    half = np.full((face.size, 2), 0.5)
    taps = {
        tangent: (np.stack((flat(lower, along), flat(lower, along)), axis=1),
                  np.stack((np.ones(face.size), np.zeros(face.size)), axis=1),
                  np.repeat((across_dual[lower] * along_widths[along])[:, None], 2, axis=1)),
        2: (np.stack((flat(lower, along), flat(upper, along)), axis=1), half,
            np.stack((across_widths[lower], across_widths[upper]), axis=1) * along_widths[along][:, None]),
        normal: (np.stack((flat(selvedge_cell, along), flat(selvedge_cell, along_lower)), axis=1), half,
                 across_widths[selvedge_cell][:, None] * np.stack((along_dual[along], along_dual[along_lower]),
                                                                  axis=1)),
    }
    across_position = np.cumsum(across_widths)[lower]                     # the hi edge of the lower cell
    along_position = np.cumsum(along_widths)[along] - 0.5 * along_widths[along]
    if boundary_test is None:
        normal_across, normal_along = sign, np.zeros(face.size)
    else:   # Domain coordinates; a periodic wrap face's upper centre continues past the last edge
        face_coordinate = origin[axis] + across_position
        lower_centre = face_coordinate - 0.5 * across_widths[lower]
        upper_centre = face_coordinate + 0.5 * across_widths[upper]
        normal_across, normal_along = _smoothed_normal(
            boundary_test, axis, np.where(sign > 0.0, lower_centre, upper_centre),
            np.where(sign > 0.0, upper_centre, lower_centre), origin[1 - axis] + along_position)
    length = along_widths[along] * np.abs(normal_across)
    normals = np.stack((normal_across, normal_along) if axis == 0 else (normal_along, normal_across), axis=1)
    positions = np.stack((across_position, along_position) if axis == 0 else (along_position, across_position), axis=1)
    return {'normals': normals, 'length': length, 'taps': taps, 'positions': positions}


def _device_medium(name, order, local, representative, frames, normals, positions, length, taps, response,
                   process_scales, xp):
    """SheetMedium with per-site tensors rotate_sites(local, frames) and deposit weights L a / A on the device."""
    if length.size == 0:
        raise ValueError(f"nonlinear sheet {name!r}: no site (the selection has no interface face, or the line no "
                         'cell)')
    site_tensor = rotate_sites(symmetrize_sources(np.asarray(local, dtype=np.complex128)), frames)
    device_taps = []
    for component in range(3):
        if component not in taps:
            device_taps.append(None)
            continue
        index, sample, area = taps[component]
        deposit = length[:, None] * sample / area
        device_taps.append((xp.asarray(index, dtype=xp.int64), xp.asarray(sample, dtype=xp.float64),
                            xp.asarray(deposit, dtype=xp.float64)))
    tensor = MediumTensor.from_tensor(name, order, representative, process_scales)
    return SheetMedium(name=name, order=tensor.order, tensor=tensor.tensor,
                       site_tensor=xp.asarray(site_tensor), site_length=xp.asarray(length, dtype=xp.float64),
                       taps=tuple(device_taps), normals=np.asarray(normals, dtype=np.float64),
                       positions=np.asarray(positions, dtype=np.float64), response=response,
                       process_scales=tensor.process_scales)


def surface_sheet(name, local, inside, outside, cell_widths, periodic, selvedge='inside', boundary_test=None,
                  origin=(0.0, 0.0), xp=np):
    """Surface chi(2) SheetMedium of a host bool (nx, ny) selection on the faces to `outside` cells (host bool; None
    = every unselected cell); local = surface_local_tensor; cell_widths = Yee (dx, dy); periodic = (x, y) flags;
    boundary_test = host (x, y) -> bool of the selection's drawn shape (smoothed normals) or None (staircase), in
    coordinates whose first grid edge is `origin`."""
    if selvedge not in SELVEDGES:
        raise ValueError(f"selvedge must be one of {', '.join(SELVEDGES)}, got {selvedge!r}")
    inside = np.asarray(inside, dtype=bool)
    outside = ~inside if outside is None else np.asarray(outside, dtype=bool) & ~inside
    widths = tuple(np.asarray(width, dtype=np.float64) for width in cell_widths)
    origin = tuple(float(value) for value in origin)
    parts = [_surface_axis_sites(axis, inside, outside, widths, tuple(periodic), selvedge, boundary_test, origin)
             for axis in (0, 1)]
    normals, positions, length = (np.concatenate([part[key] for part in parts])
                                  for key in ('normals', 'positions', 'length'))
    taps = {component: tuple(np.concatenate([part['taps'][component][item] for part in parts])
                             for item in range(3)) for component in range(3)}
    return _device_medium(name, 2, local, surface_mixing_tensor(local),
                          surface_frames(normals[:, 0], normals[:, 1]), normals, positions, length, taps,
                          'polarization', (), xp)


def line_sheet(name, order, local, axis, line_cell, along, shape, cell_widths, response='polarization',
               process_scales=(), xp=np):
    """2D-material SheetMedium on the line cells `along` (slice) of row / column line_cell of a line along 'x' / 'y'
    (placement.sheet_line); local = sheet-frame tensor (sheet_chi2_local / sheet_sigma3_local)."""
    if response not in RESPONSES:
        raise ValueError(f"response must be one of {', '.join(RESPONSES)}, got {response!r}")
    dx_cells, dy_cells = (np.asarray(width, dtype=np.float64) for width in cell_widths)
    along_axis = 0 if axis == 'x' else 1
    along_widths, across_widths = (dx_cells, dy_cells) if axis == 'x' else (dy_cells, dx_cells)
    across_dual = _dual_widths(across_widths)
    cells = np.arange(along.start, along.stop)
    flat = cells * shape[1] + line_cell if axis == 'x' else line_cell * shape[1] + cells
    pair = np.stack((flat, flat), axis=1)
    sample = np.stack((np.ones(cells.size), np.zeros(cells.size)), axis=1)
    taps = {along_axis: (pair, sample, np.repeat((along_widths[cells] * across_dual[line_cell])[:, None], 2, axis=1)),
            2: (pair, sample, np.repeat((along_widths[cells] * across_widths[line_cell])[:, None], 2, axis=1))}
    normals, positions = np.zeros((cells.size, 2)), np.zeros((cells.size, 2))
    normals[:, 1 - along_axis] = 1.0
    positions[:, along_axis] = np.cumsum(along_widths)[cells] - 0.5 * along_widths[cells]
    positions[:, 1 - along_axis] = np.cumsum(across_widths)[line_cell] - 0.5 * across_widths[line_cell]
    frames = np.broadcast_to(line_frame(axis), (cells.size, 3, 3))
    return _device_medium(name, order, local, line_mixing_tensor(local), frames, normals, positions,
                          along_widths[cells].copy(), taps, response, tuple(process_scales), xp)


# ---- evaluation (xp = numpy or cupy) ----
def site_fields(sheet, grid_fields, xp):
    """(n_sites, 3) complex128 site vectors E_s of one channel's grid fields (E_x hi-y faces | None, E_y hi-x faces
    | None, E_z centres | None)."""
    values = xp.zeros((sheet.n_sites, 3), dtype=xp.complex128)
    for component, taps in enumerate(sheet.taps):
        field = grid_fields[component]
        if taps is None or field is None:
            continue
        index, sample, _ = taps
        values[:, component] = (field.ravel()[index] * sample).sum(axis=1)
    return values


def _contract(tensor, slots):
    """sum_jk.. tensor[s, j, k, ..] slot_0[s, j] slot_1[s, k] .. of per-site tensors (n,) + (3,) * len(slots)."""
    result = tensor
    for slot in slots:
        result = (result * slot.reshape(slot.shape + (1,) * (result.ndim - 2))).sum(axis=1)
    return result


def site_polarization(terms, fields, sheet, components, omega, xp, tangents=None):
    """(n_sites, 3) complex128 site P_s [C/m] of a sheet's MixingTerms on one target (its model `components`, target
    angular frequency omega) from fields {channel index: (n, 3) site E | None} (terms with an unsolved source give
    0); tangents alike: the product-rule derivative along them (a conjugated slot takes conj V)."""
    density = xp.zeros((sheet.n_sites, 3), dtype=xp.complex128)
    scale = EPS0 if sheet.response == 'polarization' else 1.0 / (1j * float(omega))
    for term in terms:
        values = [fields.get(index) for index, _ in term.sources]
        if any(value is None for value in values):
            continue
        values = [xp.conj(value) if conjugate else value for value, (_, conjugate) in zip(values, term.sources)]
        if tangents is None:
            slot_sets = [values]
        else:
            directions = [tangents.get(index) for index, _ in term.sources]
            if any(direction is None for direction in directions):
                continue
            directions = [xp.conj(direction) if conjugate else direction
                          for direction, (_, conjugate) in zip(directions, term.sources)]
            slot_sets = [values[:slot] + [directions[slot]] + values[slot + 1:] for slot in range(len(values))]
        coefficient = scale * term.factor
        for component in components:
            for slots in slot_sets:
                density[:, component] += coefficient * _contract(sheet.site_tensor[:, component], slots)
    return density


def sheet_site_polarizations(terms, grid_fields, sheets, components, omega, xp, grid_tangents=None):
    """{sheet name: (n_sites, 3) site P_s} of the sheet terms of one target; grid_fields {channel index: (E_x faces,
    E_y faces, E_z) | None} (grid_tangents alike: the derivative along them, Newton-Krylov)."""
    result = {}
    for name, sheet in sheets.items():
        own = [term for term in terms if term.medium == name]
        if not own:
            continue
        indices = sorted({index for term in own for index, _ in term.sources})

        def sampled(grids):
            return {index: None if grids.get(index) is None else site_fields(sheet, grids[index], xp)
                    for index in indices}
        result[name] = site_polarization(own, sampled(grid_fields), sheet, components, omega, xp,
                                         None if grid_tangents is None else sampled(grid_tangents))
    return result


def _scatter_add(target, index, values, xp):
    """target[index] += values with repeated indices summed (flat complex128 target)."""
    index, values = index.ravel(), values.ravel()
    if xp is np:
        from ...fem.fixed_pattern import scatter_add   # contract nlfd_tables_numpy_only: inside the function
        scatter_add(target, index, values)
        return
    import cupyx
    for part, unit in ((values.real, 1.0), (values.imag, 1j)):
        accumulated = xp.zeros(target.shape, dtype=xp.float64)
        cupyx.scatter_add(accumulated, index, xp.ascontiguousarray(part))
        target += unit * accumulated


def sheet_grid_density(site_polarizations, sheets, components, shape, xp):
    """{component: complex128 full-grid density [C/m^2]} of deposited site P_s (ez {2: centres}; hz {0: E_x hi-y
    faces, 1: E_y hi-x faces})."""
    density = {component: xp.zeros(shape[0] * shape[1], dtype=xp.complex128) for component in components}
    for name, values in site_polarizations.items():
        sheet = sheets[name]
        for component in components:
            taps = sheet.taps[component]
            if taps is not None:
                index, _, deposit = taps
                _scatter_add(density[component], index, deposit * values[:, component:component + 1], xp)
    return {component: grid.reshape(shape) for component, grid in density.items()}


def sheet_site_work(site_polarizations, sheets, target_grid_fields, omega, xp):
    """W [W/m] = (omega / 2) sum_s L_s Im(conj(E_s) . P_s) of site densities on the target's grid fields (host sync)."""
    total = 0.0
    for name, values in site_polarizations.items():
        sheet = sheets[name]
        field = site_fields(sheet, target_grid_fields, xp)
        total += float(xp.sum(sheet.site_length * (xp.conj(field) * values).sum(axis=1).imag))
    return 0.5 * float(omega) * total


def sheet_site_absorption(site_polarizations, sheets, target_grid_fields, omega, cell_area, xp):
    """float64 full-grid [W/m^3] -(omega / 2) L_s a_t Im(conj(E[k_t]) P_s) per tap, in the cell owning tap k_t (a face:
    the cell whose hi face it is) over its area cell_area (nx, ny): sum A q = -sheet_site_work exactly."""
    shape = cell_area.shape
    flat = xp.zeros(shape[0] * shape[1], dtype=xp.complex128)
    for name, values in site_polarizations.items():
        sheet = sheets[name]
        for component, taps in enumerate(sheet.taps):
            field = target_grid_fields[component]
            if taps is None or field is None:
                continue
            index, sample, _ = taps
            weights = sheet.site_length[:, None] * sample
            _scatter_add(flat, index, weights * xp.conj(field.ravel()[index]) * values[:, component:component + 1], xp)
    return (-0.5 * float(omega)) * flat.imag.reshape(shape) / cell_area


def sheet_cotangents(terms, grid_fields, sheets, components, omega, grid_cotangent, shape, xp):
    """{source channel index: [E_x hi-y faces, E_y hi-x faces, E_z centres] complex128 cotangents (None = 0)} of the
    deposited sheet density of one target's sheet terms for dJ = 2 Re(c . dD), grid_cotangent {component: c of D}
    (ez {2: centres}, hz {0, 1: faces}): deposit gathered onto the sites, each slot transposed (a conjugated slot
    conjugated), sampling taps scattered back; the exact adjoint of sheet_grid_density o sheet_site_polarizations."""
    result = {}
    n_cells = shape[0] * shape[1]
    for name, sheet in sheets.items():
        own = [term for term in terms if term.medium == name]
        if not own:
            continue
        site_cotangent = xp.zeros((sheet.n_sites, 3), dtype=xp.complex128)   # c of P_s per component
        for component in components:
            taps, cotangent = sheet.taps[component], grid_cotangent.get(component)
            if taps is not None and cotangent is not None:
                index, _, deposit = taps
                site_cotangent[:, component] = (cotangent.ravel()[index] * deposit).sum(axis=1)
        indices = sorted({index for term in own for index, _ in term.sources})
        fields = {index: None if grid_fields.get(index) is None else site_fields(sheet, grid_fields[index], xp)
                  for index in indices}
        scale = EPS0 if sheet.response == 'polarization' else 1.0 / (1j * float(omega))
        site_result = {}   # source index -> (n_sites, 3) c of E_s
        for term in own:
            values = [fields.get(index) for index, _ in term.sources]
            if any(value is None for value in values):
                continue
            values = [xp.conj(value) if conjugate else value for value, (_, conjugate) in zip(values, term.sources)]
            coefficient = scale * term.factor
            for slot, (index, conjugate) in enumerate(term.sources):
                others = values[:slot] + values[slot + 1:]
                gradient = xp.zeros((sheet.n_sites, 3), dtype=xp.complex128)
                for component in components:
                    tensor = xp.moveaxis(sheet.site_tensor[:, component], 1 + slot, -1)   # free slot index last
                    weight = coefficient * site_cotangent[:, component]
                    gradient += weight[:, None] * _contract(tensor, others)
                gradient = xp.conj(gradient) if conjugate else gradient
                site_result[index] = gradient if index not in site_result else site_result[index] + gradient
        for index, gradient in site_result.items():
            parts = result.setdefault(index, [None, None, None])
            for component, taps in enumerate(sheet.taps):
                if taps is None or grid_fields[index][component] is None:
                    continue
                tap_index, sample, _ = taps
                flat = xp.zeros(n_cells, dtype=xp.complex128)
                _scatter_add(flat, tap_index, sample * gradient[:, component:component + 1], xp)
                grid = flat.reshape(shape)
                parts[component] = grid if parts[component] is None else parts[component] + grid
    return result

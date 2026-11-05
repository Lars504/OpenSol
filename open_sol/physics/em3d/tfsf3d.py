"""TFSF plane wave of em_fdtd_3d: an axis-aligned plane wave (E along the polarisation axis p, travelling along +-d) on
a total-field box, its incident field from a 1D auxiliary Yee line in the lossless background medium."""
# Box = total-field cells [lo_a, hi_a) per axis (engine3d layout); samples on the box surface (cell edges lo - 1/2,
# hi - 1/2) are scattered field. A side is open (None, no correction): both sides of a periodic axis (the box spans the
# period) and the downstream side when both transverse axes are periodic (one-sided plane source: the total field runs
# into the CPML). Corrections follow the Yee stencils: an H_a on a closed face whose curl term s d_m E_p reads a
# total-field E across it gets +-db s inv_h E_inc (lo / hi side); a total-field E_a whose term s d_m H_q reads a
# scattered H across it gets -+cb s inv_e H_inc (s = Levi-Civita sign (a, m, p) / (a, m, q)).
# Incident (q = the third axis): E_p = e, H_q = sigma eps_dpq h on the aux line along xi = sigma x_d (e at cell centres,
# h at hi faces, index _AUX_MARGIN = the first total-field cell upstream); the line solves eps de/dt = -dh/dxi,
# mu dh/dt = -de/dxi with the background eps / mu on the box cells' own d widths (graded too; uniform margins at the
# widths of the cells before / after the box), so it is the 3D update restricted to a transversely uniform field.
import math

import numpy as np
from ...core.backend import cp, to_host

from ...core.constants import DTYPE, EPS0, MU0
from ...core.kernels import raw_twin
from ..em.sources import _aux_line_terms, _carrier_angular_frequency, _tfsf_aux_e_graded, _tfsf_aux_h
from .boundaries3d import AXES
from .kernels3d import launch

PLANE_WAVE_DIRECTIONS = ('+x', '-x', '+y', '-y', '+z', '-z')
_AUX_MARGIN = 20          # aux cells upstream of the box (and downstream of a closed one)
_AUX_SOURCE_INDEX = 10    # aux soft-source node, upstream of the box
_AUX_THREADS = 256

# field[cell] += coef * aux[offset + step * coordinate along inc_axis] (inc_axis < 0: aux[offset]) over a patch box.
_tfsf_patch_3d_gpu = cp.RawKernel(r'''
typedef long long idx_t;
extern "C" __global__
void tfsf_patch_3d(float* __restrict__ field, const float* __restrict__ aux,
                   int i0, int j0, int k0, int n0, int n1, int n2, int ny, int nz,
                   int inc_axis, int inc_offset, int inc_step, float coef)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= n0 || j >= n1 || k >= n2) return;
    int gi = i0 + i, gj = j0 + j, gk = k0 + k;
    int coordinate = (inc_axis == 0) ? gi : (inc_axis == 1) ? gj : gk;
    int a = (inc_axis < 0) ? inc_offset : inc_offset + inc_step * coordinate;
    field[((idx_t)gi * ny + gj) * nz + gk] += coef * aux[a];
}
''', 'tfsf_patch_3d')


def _tfsf_patch_3d_host(field, aux, i0, j0, k0, n0, n1, n2, ny, nz, inc_axis, inc_offset, inc_step, coef):
    origin, extents = (int(i0), int(j0), int(k0)), (int(n0), int(n1), int(n2))
    patch = field[tuple(slice(start, start + count) for start, count in zip(origin, extents))]
    inc_axis = int(inc_axis)
    if inc_axis < 0:
        patch += coef * aux[int(inc_offset)]
        return
    coordinates = np.arange(origin[inc_axis], origin[inc_axis] + extents[inc_axis])
    shape = [1, 1, 1]
    shape[inc_axis] = -1
    patch += (coef * aux[int(inc_offset) + int(inc_step) * coordinates]).reshape(shape)


_tfsf_patch_3d = raw_twin('em3d.tfsf_patch', _tfsf_patch_3d_gpu, _tfsf_patch_3d_host)


def levi_civita(first, second, third):
    """Sign of the permutation (first, second, third) of the axes (0, 1, 2); 0 when two repeat."""
    if len({first, second, third}) < 3:
        return 0
    return 1 if (second - first) % 3 == 1 else -1


class _Patch:
    """One correction plane: component, launch, integer kernel arguments, medium coefficient kind ('db' / 'cb') and
    geometric factor (sign x inverse spacing); coef set by calibrate."""

    def __init__(self, component, origin, extents, inc_axis, inc_offset, inc_step, medium, factor, grid_shape):
        self.component = component
        self.medium = medium
        self.factor = float(factor)
        self.launch = launch(extents)
        self.args = tuple(np.int32(value) for value in (*origin, *extents, grid_shape[1], grid_shape[2], inc_axis,
                                                        inc_offset, inc_step))
        self.coef = DTYPE(0.0)


class PlaneWaveSource3D:
    """Plane wave of incident E_p peak `amplitude` [V/m] (envelope peak of `temporal`) along direction_sign x axis d on
    the total-field box `bounds` = per axis (lo | None, hi | None) cell indices; uniform or graded cells."""

    def __init__(self, engine, bounds, direction_axis, direction_sign, polarization_axis, temporal, amplitude):
        d, p = int(direction_axis), int(polarization_axis)
        if p == d:
            raise ValueError(f'plane wave: the polarisation axis {AXES[p]} must differ from the propagation axis')
        q = 3 - d - p
        sigma = 1 if direction_sign > 0 else -1
        self.direction_axis, self.polarization_axis, self.direction_sign = d, p, sigma
        self.bounds = tuple((None if lo is None else int(lo), None if hi is None else int(hi)) for lo, hi in bounds)
        self.temporal = temporal
        self.amplitude = float(amplitude)
        self.warnings = []
        lo_d, hi_d = self.bounds[d]
        upstream = lo_d if sigma > 0 else hi_d
        if upstream is None:
            raise ValueError('plane wave: the upstream box face must lie inside the domain (one scattered-field cell '
                             'before it)')
        for axis, (lo, hi) in enumerate(self.bounds):
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError(f'plane wave box spans no cell along {AXES[axis]}')
        self._upstream_cell = upstream - 1 if sigma > 0 else upstream   # scattered-field cell before the box
        widths = engine.widths[d]
        if lo_d is not None and hi_d is not None:
            box_widths = widths[lo_d:hi_d]
            downstream_width = float(widths[hi_d] if sigma > 0 else widths[lo_d - 1])
        else:   # one-sided: only the upstream face corrects; the line continues at its first total-field cell's width
            box_widths = widths[upstream:upstream + 1] if sigma > 0 else widths[upstream - 1:upstream]
            downstream_width = float(box_widths[0])
        # aux line in travel order: upstream margin, the box cells' d widths, downstream margin; e index of cell i =
        # sigma i + e_offset, h index of its hi face = sigma i + h_offset (aux face a lies between aux cells a, a + 1)
        self._aux_widths = np.concatenate((np.full(_AUX_MARGIN, float(widths[self._upstream_cell])),
                                           box_widths if sigma > 0 else box_widths[::-1],
                                           np.full(_AUX_MARGIN, downstream_width)))
        self._aux_face_spacings = 0.5 * (self._aux_widths[:-1] + self._aux_widths[1:])
        self._aux_cells = self._aux_widths.size
        self._e_offset = _AUX_MARGIN - lo_d if sigma > 0 else _AUX_MARGIN + hi_d - 1
        self._h_offset = self._e_offset if sigma > 0 else self._e_offset - 1
        xp = engine.xp
        self.aux_e = xp.zeros(self._aux_cells, dtype=DTYPE)
        self.aux_h = xp.zeros(self._aux_cells, dtype=DTYPE)
        self._aux_coef_e = xp.zeros(self._aux_cells, dtype=DTYPE)        # per centre, set by calibrate
        self._aux_coef_h = xp.zeros(self._aux_cells - 1, dtype=DTYPE)    # per inner face
        h_sign = sigma * levi_civita(d, p, q)
        self._h_patches, self._e_patches = self._patches(engine, d, p, q, sigma, h_sign)
        self._calibrated = False
        self._aux_args = None

    # ---- geometry ----
    def _ranges(self, shape):
        """Per axis (centre range, face range) of the total-field samples (cells lo..hi-1, hi faces lo..hi-2)."""
        ranges = []
        for (lo, hi), n in zip(self.bounds, shape):
            first = 0 if lo is None else lo
            ranges.append(((first, n if hi is None else hi), (first, n if hi is None else hi - 1)))
        return ranges

    def _patches(self, engine, d, p, q, sigma, h_sign):
        """(H patches reading the incident E_p, E patches reading the incident H_q)."""
        shape = engine.shape
        ranges = self._ranges(shape)
        widths = engine.widths
        h_patches, e_patches = [], []
        # H_a (a != p), curl term s d_m E_p: corrected on the closed faces normal to m
        for a in (d, q):
            m = 3 - a - p
            s = levi_civita(a, m, p)
            for side, bound in enumerate(self.bounds[m]):
                if bound is None:
                    continue
                layer = bound - 1
                incident_cell = bound if side == 0 else bound - 1
                extents_by_axis = {a: ranges[a][0], p: ranges[p][1], m: (layer, layer + 1)}
                inverse_h = 2.0 / (widths[m][layer] + widths[m][(layer + 1) % shape[m]])
                factor = (1.0 if side == 0 else -1.0) * s * inverse_h
                h_patches.append(self._patch('h' + AXES[a], extents_by_axis, m, d, sigma, incident_cell,
                                             self._e_offset, 'db', factor, shape))
        # E_a (a != q), curl term s d_m H_q: corrected on the closed faces normal to m
        for a in (d, p):
            m = 3 - a - q
            s = levi_civita(a, m, q)
            for side, bound in enumerate(self.bounds[m]):
                if bound is None:
                    continue
                layer = bound if side == 0 else bound - 1
                incident_face = bound - 1
                extents_by_axis = {a: ranges[a][1], q: ranges[q][0], m: (layer, layer + 1)}
                factor = (-1.0 if side == 0 else 1.0) * s * h_sign / widths[m][layer]
                e_patches.append(self._patch('e' + AXES[a], extents_by_axis, m, d, sigma, incident_face,
                                             self._h_offset, 'cb', factor, shape))
        return [patch for patch in h_patches if patch is not None], [patch for patch in e_patches if patch is not None]

    @staticmethod
    def _patch(component, extents_by_axis, m, d, sigma, incident_index, offset, medium, factor, shape):
        """_Patch of one face (None when empty); the incident index is fixed (m = d) or follows the d coordinate."""
        origin = tuple(int(extents_by_axis[axis][0]) for axis in range(3))
        extents = tuple(int(extents_by_axis[axis][1] - extents_by_axis[axis][0]) for axis in range(3))
        if min(extents) <= 0:
            return None
        if m == d:
            inc_axis, inc_offset, inc_step = -1, sigma * incident_index + offset, 0
        else:
            inc_axis, inc_offset, inc_step = d, offset, sigma
        return _Patch(component, origin, extents, inc_axis, inc_offset, inc_step, medium, factor, shape)

    # ---- calibration ----
    def calibrate(self, engine):
        """Background eps / mu / losses from the material tables at the cell before the upstream face (one host sync),
        the aux line coefficients, the soft-source scale (exact at the carrier) and the correction coefficients."""
        d = self.direction_axis
        cell = [0, 0, 0]
        for axis, ((first, stop), _) in enumerate(self._ranges(engine.shape)):
            cell[axis] = (first + stop - 1) // 2
        cell[d] = self._upstream_cell
        material = engine.material_id[tuple(cell)]   # the table values the engine steps (re-quantised), one transfer
        eps_x, eps_y, eps_z, mu, *losses = (float(value) for value in to_host(engine.xp.stack([
            table[material] for table in (engine.table_eps_x, engine.table_eps_y, engine.table_eps_z, engine.table_mu,
                                          engine.table_sigma, engine.table_sigma_m)])))
        axis_eps = (eps_x, eps_y, eps_z)   # each E patch steps with its own component's eps (uniaxial backgrounds)
        eps = axis_eps[self.polarization_axis]   # the incident wave's medium
        if eps <= 0.0:
            raise ValueError('plane wave: the cell before the upstream box face is a perfect conductor; the incident '
                             'wave needs a dielectric background there')
        self.warnings = []
        if any(loss != 0.0 for loss in losses):
            self.warnings.append(f'plane wave: the background before the box is lossy (sigma_e / sigma_m '
                                 f'{", ".join(f"{loss:.4g}" for loss in losses)}), but the incident wave is lossless '
                                 f'(eps_r = {eps:.4g}): the box leaks')
        outline = self._outline_warning(engine)
        if outline is not None:
            self.warnings.append(outline)
        dt = float(engine.dt)
        index = math.sqrt(eps * mu)
        half_omega_dt = 0.5 * _carrier_angular_frequency(self.temporal) * dt
        # soft source (exact at the carrier) and carrier-matched Mur ends on the uniform margins
        courant, cos_half_kappa, mur_lo = _aux_line_terms(self._aux_widths[0], dt, half_omega_dt, index)
        _, _, mur_hi = _aux_line_terms(self._aux_widths[-1], dt, half_omega_dt, index)
        self._source_scale = 2.0 * courant * cos_half_kappa * self.amplitude
        self._aux_coef_e[...] = engine.xp.asarray(-dt / (EPS0 * eps * self._aux_widths), dtype=DTYPE)
        self._aux_coef_h[...] = engine.xp.asarray(-dt / (MU0 * mu * self._aux_face_spacings), dtype=DTYPE)
        self._aux_args = (np.int32(self._aux_cells), self._aux_coef_e, np.int32(_AUX_SOURCE_INDEX))
        self._mur = (DTYPE(mur_lo), DTYPE(mur_hi))
        for patch in self._h_patches + self._e_patches:
            medium = (dt / (MU0 * mu) if patch.medium == 'db'
                      else dt / (EPS0 * axis_eps[AXES.index(patch.component[1])]))
            patch.coef = DTYPE(medium * patch.factor)
        self._calibrated = True

    def _outline_warning(self, engine):
        """Warning when the cell layers on both sides of the closed box faces hold more than one material (one transfer)."""
        extended = []
        for (lo, hi), n in zip(self.bounds, engine.shape):
            extended.append(slice(0 if lo is None else lo - 1, n if hi is None else hi + 1))
        extremes = []
        for axis, (lo, hi) in enumerate(self.bounds):
            for bound in (lo, hi):
                if bound is None:
                    continue
                index = list(extended)
                index[axis] = slice(bound - 1, bound + 1)
                layers = engine.material_id[tuple(index)]
                extremes.append(engine.xp.stack((layers.min(), layers.max())))
        if not extremes:
            return None
        values = to_host(engine.xp.stack(extremes))
        if int(values[:, 0].min()) == int(values[:, 1].max()):
            return None
        return ('plane wave: the cells on both sides of the box faces hold different materials; the incident wave is '
                'that of the background before the upstream face, so the box leaks where it crosses another material '
                '(keep the box faces in one uniform medium)')

    # ---- stepping ----
    def inject_magnetic(self, fields, t):
        """Scattered-field H corrections from the incident E at t (E^n), then the aux H update to t + dt/2."""
        if not self._calibrated:
            return
        for patch in self._h_patches:
            self._apply(fields, patch, self.aux_e)
        _tfsf_aux_h(self.aux_e[1:], self.aux_e[:-1], self._aux_coef_h, self.aux_h[:-1])

    def inject(self, fields, t):
        """Total-field E corrections from the incident H at t (the E update centre), then the aux E update."""
        if not self._calibrated:
            return
        for patch in self._e_patches:
            self._apply(fields, patch, self.aux_h)
        n_cells, e_coef, source_index = self._aux_args
        _tfsf_aux_e_graded((1,), (_AUX_THREADS,), (self.aux_e, self.aux_h, n_cells, e_coef, source_index,
                                                   DTYPE(self._source_scale * self.temporal(t)), *self._mur))

    @staticmethod
    def _apply(fields, patch, aux):
        grid, block = patch.launch
        _tfsf_patch_3d(grid, block, (getattr(fields, patch.component), aux, *patch.args, patch.coef))

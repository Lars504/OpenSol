"""Electromagnetic source injection for 2D FDTD (E_z and H_z polarisations): TFSF plane waves and current-calibrated
soft sources. H_z sources implement inject_magnetic(fields, t) and / or inject_electric(fields, t)."""

import math

import numpy as np
from ...core.backend import array_module_of, array_module_or_default, cp
from ...core.kernels import real_raw_twin, twin

from ...core.constants import DTYPE, C0, EPS0, MU0
from ...core.sources import (
    GaussianPulse, ContinuousWave, RickerWavelet,
    PointSource, LineSource, _add_scaled_profile,
)
from .dispersion import complex_permittivity
from .yee_grid import grid_cell_widths, sheet_launch_factor

__all__ = [
    'GaussianPulse', 'ContinuousWave', 'RickerWavelet',
    'PointSource', 'LineSource',
    'TFSFSource', 'TFSFSourceHz',
    'MagneticLineCurrentSource', 'ElectricDipoleSource', 'TangentialSheetCurrentSource',
]


# ---- TFSF kernels -----------------------------------------------------------
# The 1D incident-field grid lives on the GPU: every per-step operation is a tiny
# launch with scalar params (no host<->device transfers or syncs in the loop).
# Correction coefficients per box side = the engine's vacuum update factor there (yee_grid spacings: dt / (mu0 or eps0)
# over the cell width for a centre update, over the mean of the two cell widths for a hi-face update), so graded
# x / y widths are exact; coef_x_lo / _hi act on the x0 / x1 - 1 sides, coef_y_lo / _hi on the y0 / y1 - 1 sides.
# Every kernel is a core.kernels twin: the numpy twins (2D field arrays, same arguments) follow the cupy sources. real_t
# / T = the field dtype (stepper single / double: the aux line, coefficients and source value follow it).

def _tfsf_correct_h_host(hx, hy, aux_ez, x0, x1, y0, y1, aux_x0, ny, coef_x_lo, coef_x_hi, coef_y_lo, coef_y_hi):
    x0, x1, y0, y1, aux_x0 = int(x0), int(x1), int(y0), int(y1), int(aux_x0)
    hy[x0 - 1, y0:y1] -= coef_x_lo * aux_ez[aux_x0]
    hy[x1 - 1, y0:y1] += coef_x_hi * aux_ez[aux_x0 + (x1 - 1 - x0)]
    incident = aux_ez[aux_x0:aux_x0 + (x1 - x0)]
    hx[x0:x1, y0 - 1] += coef_y_lo * incident
    hx[x0:x1, y1 - 1] -= coef_y_hi * incident


def _tfsf_correct_e_host(ez, aux_hy, x0, x1, y0, y1, aux_x0, ny, coef_x_lo, coef_x_hi):
    x0, x1, y0, y1, aux_x0 = int(x0), int(x1), int(y0), int(y1), int(aux_x0)
    ez[x0, y0:y1] -= coef_x_lo * aux_hy[aux_x0 - 1]
    ez[x1 - 1, y0:y1] += coef_x_hi * aux_hy[aux_x0 + (x1 - 1 - x0)]


def _tfsf_hz_correct_h_host(hz, aux_ey, x0, x1, y0, y1, aux_x0, ny, coef_x_lo, coef_x_hi):
    x0, x1, y0, y1, aux_x0 = int(x0), int(x1), int(y0), int(y1), int(aux_x0)
    hz[x0, y0:y1] += coef_x_lo * aux_ey[aux_x0 - 1]
    hz[x1 - 1, y0:y1] -= coef_x_hi * aux_ey[aux_x0 + (x1 - 1 - x0)]


def _tfsf_hz_correct_e_host(ex, ey, aux_hz, x0, x1, y0, y1, aux_x0, ny, coef_x_lo, coef_x_hi, coef_y_lo, coef_y_hi):
    x0, x1, y0, y1, aux_x0 = int(x0), int(x1), int(y0), int(y1), int(aux_x0)
    ey[x0 - 1, y0:y1] += coef_x_lo * aux_hz[aux_x0]
    ey[x1 - 1, y0:y1] -= coef_x_hi * aux_hz[aux_x0 + (x1 - 1 - x0)]
    incident = aux_hz[aux_x0:aux_x0 + (x1 - x0)]
    ex[x0:x1, y0 - 1] -= coef_y_lo * incident
    ex[x0:x1, y1 - 1] += coef_y_hi * incident


def _tfsf_aux_h_host(e_hi, e_lo, coef, h):
    h += coef * (e_hi - e_lo)


def _tfsf_aux_e_graded_host(ez, hy, n, ce, source_index, source_value, mur_lo, mur_hi):
    n = int(n)
    edge_old = (ez[0].copy(), ez[1].copy(), ez[n - 2].copy(), ez[n - 1].copy())
    ez[1:n - 1] += ce[1:n - 1] * (hy[1:n - 1] - hy[0:n - 2])
    ez[int(source_index)] += source_value
    ez[0] = edge_old[1] + mur_lo * (ez[1] - edge_old[0])
    ez[n - 1] = edge_old[2] + mur_hi * (ez[n - 2] - edge_old[3])


_tfsf_correct_h = real_raw_twin('em2d.tfsf_correct_h', r"""
extern "C" __global__
void tfsf_correct_h(real_t* __restrict__ hx, real_t* __restrict__ hy,
                    const real_t* __restrict__ aux_ez,
                    int x0, int x1, int y0, int y1, int aux_x0, int ny,
                    real_t coef_x_lo, real_t coef_x_hi, real_t coef_y_lo, real_t coef_y_hi)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < y1 - y0) {
        int j = y0 + k;
        hy[(x0 - 1) * ny + j] -= coef_x_lo * aux_ez[aux_x0];
        hy[(x1 - 1) * ny + j] += coef_x_hi * aux_ez[aux_x0 + (x1 - 1 - x0)];
    }
    if (k < x1 - x0) {
        int i = x0 + k;
        real_t incident = aux_ez[aux_x0 + k];
        hx[i * ny + (y0 - 1)] += coef_y_lo * incident;
        hx[i * ny + (y1 - 1)] -= coef_y_hi * incident;
    }
}
""", "tfsf_correct_h", _tfsf_correct_h_host)

_tfsf_correct_e = real_raw_twin('em2d.tfsf_correct_e', r"""
extern "C" __global__
void tfsf_correct_e(real_t* __restrict__ ez, const real_t* __restrict__ aux_hy,
                    int x0, int x1, int y0, int y1, int aux_x0, int ny, real_t coef_x_lo, real_t coef_x_hi)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= y1 - y0) return;
    int j = y0 + k;
    ez[x0 * ny + j] -= coef_x_lo * aux_hy[aux_x0 - 1];
    ez[(x1 - 1) * ny + j] += coef_x_hi * aux_hy[aux_x0 + (x1 - 1 - x0)];
}
""", "tfsf_correct_e", _tfsf_correct_e_host)

# H_z box (TFSFSourceHz): total-field Hz cells [x0, x1) x [y0, y1); the box-edge faces ey[x0-1], ey[x1-1], ex[:, y0-1],
# ex[:, y1-1] are scattered-field (sc = tot - inc). coef = dt/(MU0 w) (H side, cell widths) or dt/(EPS0 s) (E side, face
# spacings). The incident wave has Ex = 0, so only the Ey faces correct Hz.
_tfsf_hz_correct_h = real_raw_twin('em2d.tfsf_hz_correct_h', r"""
extern "C" __global__
void tfsf_hz_correct_h(real_t* __restrict__ hz, const real_t* __restrict__ aux_ey,
                       int x0, int x1, int y0, int y1, int aux_x0, int ny, real_t coef_x_lo, real_t coef_x_hi)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= y1 - y0) return;
    int j = y0 + k;
    // hz[x0]: -(ey[x0] - ey[x0-1])/dx with scattered ey[x0-1] is too small by ey_inc/dx -> add.
    hz[x0 * ny + j] += coef_x_lo * aux_ey[aux_x0 - 1];
    // hz[x1-1]: -(ey[x1-1] - ey[x1-2])/dx with scattered ey[x1-1] is too large by ey_inc/dx -> subtract.
    hz[(x1 - 1) * ny + j] -= coef_x_hi * aux_ey[aux_x0 + (x1 - 1 - x0)];
}
""", "tfsf_hz_correct_h", _tfsf_hz_correct_h_host)

_tfsf_hz_correct_e = real_raw_twin('em2d.tfsf_hz_correct_e', r"""
extern "C" __global__
void tfsf_hz_correct_e(real_t* __restrict__ ex, real_t* __restrict__ ey, const real_t* __restrict__ aux_hz,
                       int x0, int x1, int y0, int y1, int aux_x0, int ny,
                       real_t coef_x_lo, real_t coef_x_hi, real_t coef_y_lo, real_t coef_y_hi)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < y1 - y0) {
        int j = y0 + k;
        // ey[x0-1]: -(hz[x0] - hz[x0-1])/dx with total hz[x0] is too small by hz_inc/dx -> add.
        ey[(x0 - 1) * ny + j] += coef_x_lo * aux_hz[aux_x0];
        // ey[x1-1]: -(hz[x1] - hz[x1-1])/dx with total hz[x1-1] is too large by hz_inc/dx -> subtract.
        ey[(x1 - 1) * ny + j] -= coef_x_hi * aux_hz[aux_x0 + (x1 - 1 - x0)];
    }
    if (k < x1 - x0) {
        int i = x0 + k;
        real_t incident = aux_hz[aux_x0 + k];
        // ex[y0-1]: +(hz[y0] - hz[y0-1])/dy with total hz[y0] is too large by hz_inc/dy -> subtract.
        ex[i * ny + (y0 - 1)] -= coef_y_lo * incident;
        // ex[y1-1]: +(hz[y1] - hz[y1-1])/dy with total hz[y1-1] is too small by hz_inc/dy -> add.
        ex[i * ny + (y1 - 1)] += coef_y_hi * incident;
    }
}
""", "tfsf_hz_correct_e", _tfsf_hz_correct_e_host)

_tfsf_aux_h = twin('em2d.tfsf_aux_h', cp.ElementwiseKernel(
    'T e_hi, T e_lo, T coef',
    'T h',
    'h += coef * (e_hi - e_lo)',
    'tfsf_aux_h_update'), _tfsf_aux_h_host)

# Single block: interior E update with a coefficient per aux node (graded widths), soft source, then first-order Mur
# ABC at both ends, one coefficient per end (the uniform margins of the two ends may differ in width); also em3d's line.
_tfsf_aux_e_graded = real_raw_twin('em2d.tfsf_aux_e_graded', r"""
extern "C" __global__
void tfsf_aux_e_graded(real_t* ez, const real_t* hy, int n, const real_t* __restrict__ ce,
                       int source_index, real_t source_value, real_t mur_lo, real_t mur_hi)
{
    __shared__ real_t edge_old[4];
    if (threadIdx.x == 0) {
        edge_old[0] = ez[0];
        edge_old[1] = ez[1];
        edge_old[2] = ez[n - 2];
        edge_old[3] = ez[n - 1];
    }
    __syncthreads();
    for (int k = 1 + threadIdx.x; k < n - 1; k += blockDim.x)
        ez[k] += ce[k] * (hy[k] - hy[k - 1]);
    __syncthreads();
    if (threadIdx.x == 0) {
        ez[source_index] += source_value;
        ez[0] = edge_old[1] + mur_lo * (ez[1] - edge_old[0]);
        ez[n - 1] = edge_old[2] + mur_hi * (ez[n - 2] - edge_old[3]);
    }
}
""", "tfsf_aux_e_graded", _tfsf_aux_e_graded_host)

_THREADS = 256



def _aux_line_terms(width, dt, half_omega_dt, index=1.0):
    """(Courant number c dt / (index width), cos(kappa0 / 2), Mur coefficient) of a uniform 1D line of refractive
    index `index` at the carrier; ValueError when the carrier does not propagate on it (> pi radians per cell)."""
    courant = C0 * float(dt) / (float(index) * float(width))
    sin_half_kappa = math.sin(half_omega_dt) / courant
    if sin_half_kappa >= 1.0:
        raise ValueError(f"TFSF: the centre frequency has no propagating wave on the 1D grid (sin(k dx/2) = "
                         f"{sin_half_kappa:.3g} >= 1, at most pi cells per wavelength in the medium); refine the mesh")
    cos_half_kappa = math.sqrt(1.0 - sin_half_kappa ** 2)
    # Mur ends matched to the numerical phase velocity at the carrier (S' = tan(omega0 dt/2) / tan(kappa0/2)):
    # with S' = S the upstream end echoes ~3e-3 of the source back into the incident wave (+-0.5 % power).
    mur_courant = courant * cos_half_kappa / math.cos(half_omega_dt)
    return courant, cos_half_kappa, (mur_courant - 1.0) / (mur_courant + 1.0)


class _PlaneWaveBox:
    """+x vacuum plane wave on a total-field box [x0, x1) x [y0, y1), shared by both polarisations: box checks, vacuum
    side coefficients and the 1D aux grid (aux index = _AUX_MARGIN + i - x0, cells at centres) on the grid's x widths.

    cell_widths None: uniform cells of width dx; else (dx_cells, dy_cells) of the full grid (yee_grid.grid_cell_widths):
    the aux line takes the box columns' widths and uniform margins at the widths of cells x0 - 1 / x1, so its update is
    the 2D one restricted to a y-invariant field (graded x and y exact: no leakage beyond float32 rounding).

    open_faces: 'y_lo' / 'y_hi' sides left uncorrected (the box opens onto a mirror plane there: the wall rows on the
    plane face own it, placement.tfsf_open_faces).

    dtype: the field dtype (float32 default, float64 for the double stepper) of the aux line, coefficients and source
    value."""

    _AUX_MARGIN = 20        # aux cells before / after the box
    _AUX_SOURCE_INDEX = 10  # aux source node, upstream of the box (the box samples carry only the +x wave)

    def __init__(self, x0, x1, y0, y1, temporal, dx, dt, amplitude, cell_widths=None, open_faces=(), dtype=None):
        self.dtype = np.dtype(DTYPE if dtype is None else dtype)
        self._real = self.dtype.type
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        if not (1 <= x0 < x1 and 1 <= y0 < y1):
            raise ValueError(
                f"TFSF box [{x0}, {x1}) x [{y0}, {y1}) must be non-empty and keep one scattered-field "
                f"cell on the lo sides (x0 >= 1, y0 >= 1)")
        unknown = set(open_faces) - {'y_lo', 'y_hi'}
        if unknown:
            raise ValueError(f"TFSF box: only the y sides may open onto a mirror plane, got {sorted(unknown)}")
        self.open_faces = frozenset(open_faces)
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.temporal = temporal
        self.amplitude = float(amplitude)
        self._grid_shape = None
        if cell_widths is None:
            dx_cells, dy_cells = np.full(x1 + 1, float(dx)), np.full(y1 + 1, float(dx))
        else:
            dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
            if x1 >= dx_cells.size or y1 >= dy_cells.size:
                raise ValueError(f"TFSF box [{x0}, {x1}) x [{y0}, {y1}) needs a scattered-field cell on the hi sides "
                                 f"of a {dx_cells.size} x {dy_cells.size} grid")
        dt = float(dt)

        # aux widths: upstream margin at cell x0 - 1's width, the box columns, downstream margin at cell x1's width
        margin = self._AUX_MARGIN
        upstream_width, downstream_width = float(dx_cells[x0 - 1]), float(dx_cells[x1])
        self._aux_widths = np.concatenate((np.full(margin, upstream_width), dx_cells[x0:x1],
                                           np.full(margin, downstream_width)))
        self._aux_face_spacings = 0.5 * (self._aux_widths[:-1] + self._aux_widths[1:])   # hi face a of aux cell a
        # A soft E source adding s per step in 1D FDTD radiates E = s / (2 S cos(kappa/2)) each way (S = c dt / dx,
        # sin(kappa/2) = sin(omega dt/2) / S, kappa = k_num dx), so s = 2 S cos(kappa0/2) amplitude temporal(t): exact
        # at the centre frequency omega0 (CW exact, pulses at f0), as the FDFD PlaneWavePhasor; the source sits in the
        # uniform upstream margin.
        half_omega_dt = 0.5 * _carrier_angular_frequency(temporal) * dt
        courant, cos_half_kappa, mur_lo = _aux_line_terms(upstream_width, dt, half_omega_dt)
        _, _, mur_hi = _aux_line_terms(downstream_width, dt, half_omega_dt)
        self._source_scale = 2.0 * courant * cos_half_kappa * self.amplitude
        self._mur_lo, self._mur_hi = self._real(mur_lo), self._real(mur_hi)
        # main-grid spacings at the box sides: widths of the first / last box column (centre updates) and spacings of
        # the faces x0 - 1 / x1 - 1 and y0 - 1 / y1 - 1 (hi-face updates: the mean of the two cells' widths)
        self._side_widths_x = (float(dx_cells[x0]), float(dx_cells[x1 - 1]))
        self._side_spacings = tuple(0.5 * float(widths[index] + widths[index + 1])
                                    for widths, index in ((dx_cells, x0 - 1), (dx_cells, x1 - 1),
                                                          (dy_cells, y0 - 1), (dy_cells, y1 - 1)))
        self._dt = dt

        self._aux_x0 = margin
        self._aux_cells = self._aux_widths.size
        self._aux_size = np.int32(self._aux_cells)
        self._aux_source_index = np.int32(self._AUX_SOURCE_INDEX)

        self._args = (np.int32(x0), np.int32(x1), np.int32(y0), np.int32(y1), np.int32(self._aux_x0))
        self._blocks_all_sides = ((max(x1 - x0, y1 - y0) + _THREADS - 1) // _THREADS,)
        self._blocks_x_sides = ((y1 - y0 + _THREADS - 1) // _THREADS,)   # the x0 / x1 sides span y0..y1

    def _side_coefficients(self, constant, spacings):
        """dt / (constant * spacing) in the field dtype per side spacing (x_lo, x_hi[, y_lo, y_hi]); 0 on an open y
        side."""
        sides = ('x_lo', 'x_hi', 'y_lo', 'y_hi')[:len(spacings)]
        return tuple(self._real(0.0 if side in self.open_faces else self._dt / (constant * spacing))
                     for side, spacing in zip(sides, spacings))

    def _bind_grid(self, field):
        """Check once per grid shape that the box plus its scattered-field ring fits and fields are C-ordered."""
        nx, ny = field.shape
        if self.x1 >= nx or self.y1 >= ny:
            raise ValueError(f"TFSF box [{self.x0}, {self.x1}) x [{self.y0}, {self.y1}) needs a scattered-field "
                             f"cell on the hi sides of a {nx} x {ny} grid")
        if field.dtype != self.dtype or not field.flags.c_contiguous:
            raise TypeError(f"TFSF fields must be C-contiguous {self.dtype.name} arrays (the source's dtype), got "
                            f"{field.dtype}")
        self._grid_shape = field.shape
        self._ny = np.int32(ny)


class TFSFSource(_PlaneWaveBox):
    """Plane wave along +x in vacuum: total-field Ez cells [x0, x1) x [y0, y1), incident Ez peak = amplitude [V/m];
    graded grids with cell_widths (see _PlaneWaveBox). Call inject_h after the H update and inject_e after the E update."""

    def __init__(self, x0, x1, y0, y1, temporal, dx, dt, amplitude=1.0, cell_widths=None, open_faces=(), xp=None,
                 dtype=None):
        super().__init__(x0, x1, y0, y1, temporal, dx, dt, amplitude, cell_widths, open_faces, dtype)
        xp = array_module_or_default(xp)   # the run backend's (aux line and the twins follow the field arrays)
        dtype = self.dtype
        self.aux_ez = xp.zeros(self._aux_cells, dtype=dtype)   # cell centres
        self.aux_hy = xp.zeros(self._aux_cells, dtype=dtype)   # hi faces
        # hy x0-1 / x1-1 and hx y0-1 / y1-1 (face spacings); ez x0 / x1-1 (cell widths)
        self._coef_h_sides = self._side_coefficients(MU0, self._side_spacings)
        self._coef_e_sides = self._side_coefficients(EPS0, self._side_widths_x)
        self._aux_coef_h = xp.asarray(self._dt / (MU0 * self._aux_face_spacings), dtype=dtype)   # (n - 1,) hi faces
        self._aux_coef_e = xp.asarray(self._dt / (EPS0 * self._aux_widths), dtype=dtype)         # (n,) centres

    def inject_h(self, hx, hy):
        """Scattered-field H corrections from the incident Ez (time n), then the auxiliary H update."""
        if hy.shape != self._grid_shape:
            self._bind_grid(hy)
        _tfsf_correct_h(self._blocks_all_sides, (_THREADS,),
                        (hx, hy, self.aux_ez, *self._args, self._ny, *self._coef_h_sides))
        _tfsf_aux_h(self.aux_ez[1:], self.aux_ez[:-1], self._aux_coef_h, self.aux_hy[:-1])

    def inject_e(self, ez, t: float):
        """Total-field Ez corrections from the incident H (time n+1/2), then the auxiliary E update at time t."""
        if ez.shape != self._grid_shape:
            self._bind_grid(ez)
        _tfsf_correct_e(self._blocks_x_sides, (_THREADS,),
                        (ez, self.aux_hy, *self._args, self._ny, *self._coef_e_sides))
        source_value = self._real(self._source_scale * self.temporal(t))
        _tfsf_aux_e_graded((1,), (_THREADS,),
                           (self.aux_ez, self.aux_hy, self._aux_size, self._aux_coef_e,
                            self._aux_source_index, source_value, self._mur_lo, self._mur_hi))


class TFSFSourceHz(_PlaneWaveBox):
    """Plane wave along +x in vacuum for an H_z engine: total-field Hz cells [x0, x1) x [y0, y1) (box-edge E faces are
    scattered-field), incident Ey peak = amplitude [V/m] with Hz = Ey / eta0; graded grids with cell_widths."""

    def __init__(self, x0, x1, y0, y1, temporal, dx, dt, amplitude=1.0, cell_widths=None, open_faces=(), xp=None,
                 dtype=None):
        super().__init__(x0, x1, y0, y1, temporal, dx, dt, amplitude, cell_widths, open_faces, dtype)
        xp = array_module_or_default(xp)
        dtype = self.dtype
        # 1D vacuum pair mu dHz/dt = -dEy/dx, eps0 dEy/dt = -dHz/dx: the E_z aux kernels with negated coefficients.
        self.aux_hz = xp.zeros(self._aux_cells, dtype=dtype)   # cell centres (aux_hz[0] is never used)
        self.aux_ey = xp.zeros(self._aux_cells, dtype=dtype)   # hi faces; the soft source and the Mur ends sit here
        # hz x0 / x1-1 (cell widths); ey x0-1 / x1-1 and ex y0-1 / y1-1 (face spacings)
        self._coef_h_sides = self._side_coefficients(MU0, self._side_widths_x)
        self._coef_e_sides = self._side_coefficients(EPS0, self._side_spacings)
        self._aux_coef_h = xp.asarray(-self._dt / (MU0 * self._aux_widths[1:]), dtype=dtype)   # hz centres 1 .. n-1
        # ey faces 0 .. n-1 (the last one is never updated: its spacing repeats the last width)
        face_spacings = np.append(self._aux_face_spacings, self._aux_widths[-1])
        self._aux_coef_e = xp.asarray(-self._dt / (EPS0 * face_spacings), dtype=dtype)

    def inject_magnetic(self, fields, t: float):
        """Total-field Hz corrections from the incident Ey (time n), then the auxiliary Hz update to n+1/2."""
        hz = fields.hz
        if hz.shape != self._grid_shape:
            self._bind_grid(hz)
        _tfsf_hz_correct_h(self._blocks_x_sides, (_THREADS,),
                           (hz, self.aux_ey, *self._args, self._ny, *self._coef_h_sides))
        # aux_hz[k] += -dt/(MU0 w_k) (aux_ey[k] - aux_ey[k-1]) for k >= 1.
        _tfsf_aux_h(self.aux_ey[1:], self.aux_ey[:-1], self._aux_coef_h, self.aux_hz[1:])

    def inject_electric(self, fields, t: float):
        """Scattered-field Ex / Ey corrections from the incident Hz (time n+1/2), then the auxiliary Ey update at t."""
        if fields.hz.shape != self._grid_shape:
            self._bind_grid(fields.hz)
        _tfsf_hz_correct_e(self._blocks_all_sides, (_THREADS,),
                           (fields.ex, fields.ey, self.aux_hz, *self._args, self._ny, *self._coef_e_sides))
        source_value = self._real(self._source_scale * self.temporal(t))
        # aux_ey[k] += -dt/(EPS0 s_k) (aux_hz[k+1] - aux_hz[k]) for 1 <= k <= n-2 (the kernel's hy[k] - hy[k-1]).
        _tfsf_aux_e_graded((1,), (_THREADS,),
                           (self.aux_ey, self.aux_hz[1:], self._aux_size, self._aux_coef_e,
                            self._aux_source_index, source_value, self._mur_lo, self._mur_hi))


# ---- Current-calibrated soft sources -------------------------------------------

def _line_injection_profile_host(cb, admittance, weight, profile):
    profile[...] = cb * admittance * weight


# cb of the field dtype (T), the calibrated float32 profile.
_line_injection_profile = twin('em2d.line_injection_profile', cp.ElementwiseKernel(
    'T cb, float32 admittance, float32 weight',
    'float32 profile',
    'profile = (float)(cb * admittance * weight)',
    'em_line_injection_profile'), _line_injection_profile_host)


def _carrier_angular_frequency(temporal) -> float:
    """2 pi times the centre frequency [rad/s]: f0 of a GaussianPulse / ContinuousWave, f_peak of a RickerWavelet."""
    frequency = temporal.f_peak if isinstance(temporal, RickerWavelet) else temporal.f0
    return 2.0 * np.pi * float(frequency)


def _line_cell_permittivity(axis, position, span, background, sigma_e, dispersive_regions, omega):
    """complex128 eps(omega) of the cells [span) of a grid line (axis 'x': along x at y-cell `position`; span None =
    whole line): background - j sigma_e/(omega eps0) + chi of each region [(pole, box | None, cropped mask | None)]."""
    along_axis = 0 if axis == 'x' else 1
    lo, hi = span if span is not None else (0, background.shape[along_axis])

    def line(array):
        return array[lo:hi, position] if axis == 'x' else array[position, lo:hi]

    permittivity = complex_permittivity(line(background), line(sigma_e), (), omega, xp=array_module_of(background))
    for pole, box, cropped_mask in dispersive_regions:
        susceptibility = complex(pole.susceptibility(omega))
        if box is None:
            permittivity += susceptibility
            continue
        along_box, normal_box = (box[0], box[1]) if axis == 'x' else (box[1], box[0])
        start, stop = max(lo, along_box.start), min(hi, along_box.stop)
        if not (normal_box.start <= position < normal_box.stop and start < stop):
            continue
        if cropped_mask is None:
            permittivity[start - lo:stop - lo] += susceptibility
            continue
        along_in_box, normal_in_box = slice(start - along_box.start, stop - along_box.start), position - normal_box.start
        mask_line = (cropped_mask[along_in_box, normal_in_box] if axis == 'x'
                     else cropped_mask[normal_in_box, along_in_box])
        permittivity[start - lo:stop - lo] += susceptibility * mask_line
    return permittivity


def _sheet_admittance(line_eps, mu_r, sigma_m, omega, normal_spacing, dt):
    """float32 |sqrt(eps / mu) cos(kappa/2)| of line cells at omega (mu = mu_r - j sigma_m/(omega mu0)): a Yee sheet
    current K radiates |eta(omega)| K / (2 |cos(kappa/2)|) per side (yee_grid.sheet_launch_factor)."""
    xp = array_module_of(line_eps)
    permeability = mu_r.astype(np.complex128) - 1j * sigma_m.astype(np.float64) / (omega * MU0)
    launch_factor = sheet_launch_factor(line_eps * permeability, omega, normal_spacing, dt, xp=xp)
    return xp.abs(xp.sqrt(line_eps / permeability) * launch_factor).astype(DTYPE)


class MagneticLineCurrentSource(PointSource):
    """z-directed magnetic line current of an H_z engine at cell (ix, iy): inject_magnetic adds weight * amplitude *
    temporal(t) to hz after the H update (weight = db K_m / (dx_i dy_j), set by PointCurrentCalibration('db'))."""

    def inject_magnetic(self, fields, t: float):
        self.inject(fields.hz, t)


class ElectricDipoleSource(PointSource):
    """In-plane electric dipole of an H_z engine in cell (ix, iy): `faces` = (('ex' | 'ey', i, j, J [A/m^2]), ...);
    inject_electric adds face_weights[k] * amplitude * temporal(t) per face (weights set by DipoleCalibration)."""

    def __init__(self, ix, iy, temporal, faces, xp=None):
        super().__init__(ix, iy, temporal, amplitude=1.0, weight=None)
        faces = tuple((component, int(i), int(j), float(current_density))
                      for component, i, j, current_density in faces)
        if not faces or any(face[0] not in ('ex', 'ey') for face in faces):
            raise ValueError(f"ElectricDipoleSource needs at least one ('ex' | 'ey', i, j, J) face, got {faces}")
        self.faces = faces
        self.face_weights = array_module_or_default(xp).zeros(len(faces), dtype=DTYPE)
        self._face_weight_views = tuple(self.face_weights[k:k + 1] for k in range(len(faces)))

    def inject_electric(self, fields, t: float):
        scale = DTYPE(self.amplitude * self.temporal(t))
        if scale == 0:   # exact no-op (a pulse past its envelope)
            return
        for (component, i, j, _), weight in zip(self.faces, self._face_weight_views):
            _add_scaled_profile(weight, scale, getattr(fields, component)[i:i + 1, j])


class TangentialSheetCurrentSource(LineSource):
    """In-plane sheet current of an H_z engine on the hi faces of its line cells: E_x faces for axis 'x' (horizontal
    line), E_y faces for axis 'y'; inject_electric adds amplitude * temporal(t) * profile after the E update."""

    hi_face = True

    def __init__(self, axis, position, temporal, amplitude=1.0, span=None, profile=None):
        super().__init__(axis, position, temporal, amplitude, span=span, profile=profile)
        self.component = 'ex' if axis == 'x' else 'ey'

    def inject_electric(self, fields, t: float):
        self.inject(getattr(fields, self.component), t)


# Bloch-phased line injection: target += scale * phase_factor * profile (phase_factor = cos(k.r) or -sin(k.r)).
def _add_phased_profile_host(profile, phase_factor, scale, target):
    target += scale * phase_factor * profile


_add_phased_profile = twin('em2d.add_phased_profile', cp.ElementwiseKernel(
    'float32 profile, float32 phase_factor, float32 scale',
    'T target',
    'target += (T)(scale * phase_factor * profile)',
    'em_add_phased_profile'), _add_phased_profile_host)


def bloch_line_phase(grid, boundary_config, axis, position, span):
    """float64 phase k.r [rad] of the cells [span) of a grid line (axis 'x': along x at y-cell `position`; span None =
    whole line), r = grid cell centres from the grid's lo corner, k = (bloch_kx, bloch_ky) on bloch axes (0 elsewhere)."""
    cell_dx, cell_dy = grid_cell_widths(grid)
    kx = float(boundary_config.bloch_kx) if boundary_config.bloch_x else 0.0
    ky = float(boundary_config.bloch_ky) if boundary_config.bloch_y else 0.0
    centres_x = np.cumsum(cell_dx) - 0.5 * cell_dx
    centres_y = np.cumsum(cell_dy) - 0.5 * cell_dy
    along_centres, k_along, k_normal, normal_centre = (
        (centres_x, kx, ky, centres_y[position]) if axis == 'x' else (centres_y, ky, kx, centres_x[position]))
    lo, hi = span if span is not None else (0, along_centres.size)
    return k_along * along_centres[lo:hi] + k_normal * normal_centre


class _BlochPhasedLine:
    """Mixin of a calibrated line source under Bloch k != 0: the sheet carries e^{-j k.r} (an oblique plane wave when it
    spans a bloch axis), so the real field set gets cos(k.r) and the imaginary set -sin(k.r) times the profile."""

    component = 'ez'

    def set_bloch_phase(self, phase):
        """float32 cos(phase) / -sin(phase) per line cell on the profile's backend (phase [rad] from
        bloch_line_phase)."""
        phase = np.asarray(phase, dtype=np.float64)
        if phase.shape != self.profile.shape:
            raise ValueError(f"Bloch phase has {phase.size} cells, the line source {self.profile.size}")
        xp = array_module_of(self.profile)
        self._phase_cos = xp.asarray(np.cos(phase), dtype=DTYPE)
        self._phase_minus_sin = xp.asarray(-np.sin(phase), dtype=DTYPE)

    def inject(self, target, t: float):
        value = DTYPE(self.amplitude * self.temporal(t))
        _add_phased_profile(self.profile, self._phase_cos, value, self.target_line(target))

    def inject_electric_imaginary(self, imaginary_fields, t: float):
        """Imaginary-set injection at the electric source time t (the engine calls it after the real one)."""
        value = DTYPE(self.amplitude * self.temporal(t))
        _add_phased_profile(self.profile, self._phase_minus_sin, value,
                            self.target_line(getattr(imaginary_fields, self.component)))


class BlochLineSource(_BlochPhasedLine, LineSource):
    """E_z sheet current with the Bloch phase e^{-j k.r} along the line (LineSource otherwise)."""


class BlochTangentialSheetCurrentSource(_BlochPhasedLine, TangentialSheetCurrentSource):
    """H_z in-plane sheet current with the Bloch phase e^{-j k.r} along the line (component set by the axis)."""


class PointCurrentCalibration:
    """Line current at a PointSource cell: device weight = engine.<coefficient> * current / (dx_i dy_j), with 'cb' for
    an E_z filament current I [A] and 'db' for an H_z magnetic line current K_m [V]."""

    def __init__(self, source: PointSource, current: float, cell_area: float, coefficient: str = 'cb'):
        if source.weight is None:
            raise ValueError("PointCurrentCalibration needs a PointSource with a device weight")
        if coefficient not in ('cb', 'db'):
            raise ValueError(f"PointCurrentCalibration coefficient must be 'cb' or 'db', got {coefficient!r}")
        self.source = source
        self.coefficient = coefficient
        self.current_density = DTYPE(float(current) / float(cell_area))   # J [A/m^2] or M [V/m^2]

    def apply(self, engine, materials):
        engine.xp.multiply(self.source.target_cell(getattr(engine, self.coefficient)), self.current_density,
                           out=self.source.weight)


class SheetCurrentCalibration:
    """Sheet current radiating `amplitude` [V/m] per side at omega0: profile = cb * |sqrt(eps(omega0) / mu(omega0))
    cos(kappa0/2)| * weight, `weight` = 2 * shape / (ETA0 * w_perp). E_z LineSource: cb, eps_r of the line cells; H_z
    TangentialSheetCurrentSource: cb_x|y on the hi E faces, eps = width-weighted face mean of the eps_xx|yy cells
    (complex eps: sigma_e, poles of `physics.dispersive_regions`; mu_r, sigma_m of the line cells)."""

    def __init__(self, source: LineSource, weight, physics, normal_spacing):
        if source.profile is None or source.profile.shape != weight.shape:
            raise ValueError("SheetCurrentCalibration needs a LineSource profile shaped like its weight")
        self.source = source
        self.weight = weight
        self.physics = physics
        self.normal_spacing = float(normal_spacing)
        self.omega = _carrier_angular_frequency(source.temporal)

    def _row_permittivity(self, row, cell_eps, materials):
        source = self.source
        return _line_cell_permittivity(source.axis, row, source.span, cell_eps, materials.sigma_e,
                                       self.physics.dispersive_regions, self.omega)

    def apply(self, engine, materials):
        source = self.source
        if not isinstance(source, TangentialSheetCurrentSource):
            cb_line, line_eps = engine.cb, self._row_permittivity(source.position, materials.eps_r, materials)
        else:
            if source.component == 'ex':
                cb_line, cell_eps = engine.cb_x, materials.eps_xx
            else:
                cb_line, cell_eps = engine.cb_y, materials.eps_yy
            # Hi face of the line cells: width-weighted mean of the line cell and the next one across (wrapped on a
            # periodic axis; a non-periodic last-cell line was refused at placement), as yee_grid.face_average.
            normal_widths = grid_cell_widths(self.physics.domain.grid)[1 if source.axis == 'x' else 0]
            lower_row, upper_row = source.position, (source.position + 1) % normal_widths.size
            lower_width, upper_width = float(normal_widths[lower_row]), float(normal_widths[upper_row])
            line_eps = (self._row_permittivity(lower_row, cell_eps, materials) * lower_width
                        + self._row_permittivity(upper_row, cell_eps, materials) * upper_width
                        ) / (lower_width + upper_width)
        admittance = _sheet_admittance(line_eps, source.target_line(materials.mu_r),
                                       source.target_line(materials.sigma_m), self.omega, self.normal_spacing,
                                       float(self.physics.domain.grid.dt))
        _line_injection_profile(source.target_line(cb_line), admittance, self.weight, source.profile)


class DipoleCalibration:
    """ElectricDipoleSource weights: face_weights[k] = cb_face[i, j] * J_k (cb_x on E_x faces, cb_y on E_y faces)."""

    def __init__(self, source: ElectricDipoleSource):
        self.source = source
        self._current_densities = tuple(DTYPE(current_density) for *_, current_density in source.faces)

    def apply(self, engine, materials):
        face_weights = self.source.face_weights
        for k, ((component, i, j, _), current_density) in enumerate(zip(self.source.faces, self._current_densities)):
            cb_face = engine.cb_x if component == 'ex' else engine.cb_y
            engine.xp.multiply(cb_face[i:i + 1, j], current_density, out=face_weights[k:k + 1])

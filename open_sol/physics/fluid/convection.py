"""Convective heat transport fluid -> heat: the advective source q = -rho cp u . grad T, the transform of the E13 'sum'
contributor fluid_heat_convection (coupling/definitions_fluid.py) to the heat slot volumetric_power."""
from ...core.backend import cp
import numpy as np

from ...core.constants import DTYPE, FACES
from ...core.kernels import raw_twin
from .lattice import BLOCK_SIZE

TARGET_SLOT = 'volumetric_power'

# Advective form via face fluxes: q_i = -(rho cp)_i / dx * sum over faces of u_n,f (T_f - T_i), T_f the mean of the two
# cells (interior faces between fluid cells, u_f their mean normal velocity; faces to a solid carry nothing) or the
# Dirichlet face temperature of heat (outer faces; other outer faces carry nothing). Zero for uniform T whatever div u.
_SOURCE = r'''
extern "C" __global__
void advective_heat_source(const REAL* __restrict__ rise, const float* __restrict__ velocity_x,
                           const float* __restrict__ velocity_y, const float* __restrict__ mass_density,
                           const float* __restrict__ heat_capacity, const bool* __restrict__ fluid,
                           const int* __restrict__ face_dirichlet, const REAL* __restrict__ face_rise,
                           float* __restrict__ power, const int wx, const int wy, const int ox, const int oy,
                           const int ny, const REAL inverse_dx)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= wx * wy) return;
    const int i = n / wy;
    const int j = n - i * wy;
    const int g = (i + ox) * ny + (j + oy);
    const REAL capacity = (REAL)mass_density[g] * (REAL)heat_capacity[g];
    if (!fluid[n] || !(capacity > (REAL)0)) { power[g] = 0.f; return; }
    const REAL own = rise[n];
    REAL total = 0;
    if (i + 1 < wx) {
        if (fluid[n + wy]) total += (REAL)0.25 * ((REAL)velocity_x[g] + velocity_x[g + ny]) * (rise[n + wy] - own);
    } else if (face_dirichlet[1]) total += (REAL)velocity_x[g] * (face_rise[1] - own);
    if (i > 0) {
        if (fluid[n - wy]) total -= (REAL)0.25 * ((REAL)velocity_x[g] + velocity_x[g - ny]) * (rise[n - wy] - own);
    } else if (face_dirichlet[0]) total -= (REAL)velocity_x[g] * (face_rise[0] - own);
    if (j + 1 < wy) {
        if (fluid[n + 1]) total += (REAL)0.25 * ((REAL)velocity_y[g] + velocity_y[g + 1]) * (rise[n + 1] - own);
    } else if (face_dirichlet[3]) total += (REAL)velocity_y[g] * (face_rise[3] - own);
    if (j > 0) {
        if (fluid[n - 1]) total -= (REAL)0.25 * ((REAL)velocity_y[g] + velocity_y[g - 1]) * (rise[n - 1] - own);
    } else if (face_dirichlet[2]) total -= (REAL)velocity_y[g] * (face_rise[2] - own);
    power[g] = (float)(-capacity * total * inverse_dx);
}
'''

_kernels = {}


def _advective_heat_source_host(rise, velocity_x, velocity_y, mass_density, heat_capacity, fluid, face_dirichlet,
                                face_rise, power, wx, wy, ox, oy, ny, inverse_dx):
    """numpy twin of advective_heat_source (the RawKernel args; REAL = rise.dtype): writes the window of power."""
    wx, wy, ox, oy = int(wx), int(wy), int(ox), int(oy)
    real = rise.dtype
    window = (slice(ox, ox + wx), slice(oy, oy + wy))
    rise = rise.reshape(wx, wy)
    fluid = fluid.reshape(wx, wy)
    velocity_x = velocity_x[window].astype(real)
    velocity_y = velocity_y[window].astype(real)
    capacity = mass_density[window].astype(real) * heat_capacity[window].astype(real)
    total = np.zeros((wx, wy), dtype=real)
    # interior faces between fluid cells: cell lo gains u_f (T_hi - T_lo), cell hi loses u_f (T_lo - T_hi): the same term
    face_x = np.where(fluid[:-1, :] & fluid[1:, :],
                      real.type(0.25) * (velocity_x[:-1, :] + velocity_x[1:, :]) * (rise[1:, :] - rise[:-1, :]), 0)
    total[:-1, :] += face_x
    total[1:, :] += face_x
    face_y = np.where(fluid[:, :-1] & fluid[:, 1:],
                      real.type(0.25) * (velocity_y[:, :-1] + velocity_y[:, 1:]) * (rise[:, 1:] - rise[:, :-1]), 0)
    total[:, :-1] += face_y
    total[:, 1:] += face_y
    # Dirichlet outer faces (FACES order x_lo, x_hi, y_lo, y_hi)
    if int(face_dirichlet[1]):
        total[-1, :] += velocity_x[-1, :] * (face_rise[1] - rise[-1, :])
    if int(face_dirichlet[0]):
        total[0, :] -= velocity_x[0, :] * (face_rise[0] - rise[0, :])
    if int(face_dirichlet[3]):
        total[:, -1] += velocity_y[:, -1] * (face_rise[3] - rise[:, -1])
    if int(face_dirichlet[2]):
        total[:, 0] -= velocity_y[:, 0] * (face_rise[2] - rise[:, 0])
    power[window] = np.where(fluid & (capacity > 0), -capacity * total * inverse_dx, 0).astype(power.dtype)


def _kernel(dtype):
    """advective_heat_source compiled for the heat solve precision (float64 / float32 rise); a core.kernels twin."""
    real = 'double' if dtype == cp.float64 else 'float'
    if real not in _kernels:
        _kernels[real] = raw_twin(f'fluid.advective_heat_source_{real}',
                                  cp.RawKernel(_SOURCE.replace('REAL', real), 'advective_heat_source'),
                                  _advective_heat_source_host)
    return _kernels[real]


class AdvectiveHeatSource:
    """CouplingTerm transform (velocity_x, fluid, heat) -> -rho cp u . grad T [W/m^3], full grid float32, from the
    fluid u_x, u_y and fluid cells and the heat solve-precision rise T - T0 and Dirichlet faces (a persistent buffer)."""

    def __init__(self, name: str = ''):
        self.name = name
        self.buffer = None
        self._face_dirichlet = None
        self._face_rise = None
        self._face_key = None

    def __call__(self, velocity_x, fluid, heat):
        """Advective power into the buffer (one kernel; the heat face table is uploaded only when it changed)."""
        domain = fluid.domain
        xp = domain.xp   # the run backend (the kernel is a core.kernels twin)
        nx, ny = domain.grid.nx, domain.grid.ny
        if self.buffer is None:
            self.buffer = xp.zeros((nx, ny), dtype=DTYPE)
        system = heat.system
        window_x, window_y = fluid.window
        if (window_x, window_y) != tuple(system.window):
            raise ValueError(f"coupling '{self.name}': the fluid and heat physics solve on different windows")
        key = tuple((heat.bcs.sides()[face].kind == 'dirichlet', heat.bcs.sides()[face].value) for face in FACES)
        if key != self._face_key or self._face_rise is None or self._face_rise.dtype != system.dtype:
            self._face_dirichlet = xp.asarray([int(dirichlet) for dirichlet, _ in key], dtype=np.int32)
            self._face_rise = xp.asarray([value - heat.T0 if dirichlet else 0.0 for dirichlet, value in key],
                                         dtype=system.dtype)
            self._face_key = key
        wx, wy = fluid.window_shape
        real = np.float64 if system.dtype == cp.float64 else np.float32
        materials = domain.materials
        _kernel(system.dtype)(
            ((wx * wy + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,),
            (system.unknown2d, velocity_x, fluid.fields.u_y, materials.rho, materials.cp, fluid.fluid_mask,
             self._face_dirichlet, self._face_rise, self.buffer, np.int32(wx), np.int32(wy),
             np.int32(window_x.start), np.int32(window_y.start), np.int32(ny), real(1.0 / fluid.dx)))
        return self.buffer

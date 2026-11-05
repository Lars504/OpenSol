"""
Open_sol example: thermo-optic shark-fin resonance in a 2D ring resonator.

Physics
-------
A Si ring sits in SiO2 cladding (optically and thermally). A CW source shaped
to the bus mode drives a straight Si bus side-coupled to the ring. Ohmic loss
in the Si ring (sigma_e tuned for ~50% round-trip power loss) heats the ring. The thermo-optic coefficient
(dn/dT ~ 1.8e-4 K^-1 in Si) red-shifts the resonance. Sweeping the source
wavelength up and back across one resonance produces the classic asymmetric
"shark fin" / bistable lineshape.

Script structure
----------------
- BusModeSource is a plain object with `inject(ez, t)`; EMPhysics.step injects
  every entry of `em.sources` itself, so swapping the source per wavelength is
  all the sweep needs.
- The steady-state heat solve only needs to run every N EM steps, not every
  femtosecond. The runner supports that natively via the `every=` cadence on
  add_physics, so heat is registered with every=COUPLING_EVERY and the runner
  skips redundant CG solves during the optical buildup.

Tuning notes
------------
- The 50% per-round-trip target sets sigma_e via the low-loss power attenuation
  alpha_p = sigma_e * eta0 / n_Si,  alpha_p * L_RT = -ln(0.5),  L_RT = 2*pi*R.
- The E_z thermo-optic update reads dndT_zz (the out-of-plane index), so the
  thermal stamp must write it; dndT_xx / dndT_yy alone give no shift.
- Heat BCs are Dirichlet 300 K on all four faces (perfect heatsink) so the
  steady solver has a well-posed ground.
- Bistability requires absorbed power large enough that the thermal shift
  exceeds the cold linewidth. Crank `source_amplitude` if the up- and down-
  sweep curves overlap.
"""

import os
import sys

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_sol import Simulation
from open_sol.core.domain import Domain
from open_sol.core.constants import C0, ETA0, DTYPE
from open_sol.physics.em import EMPhysics, BoundaryConfig
from open_sol.physics.heat import HeatPhysics, HeatBoundaryConfig, BCSide
from open_sol.coupling import joule_heating, thermo_optic


# --- geometry / material parameters --------------------------------------------

SIZE_X = 8.0e-6           # bus runs along x, so x is the long axis
SIZE_Y = 7.0e-6
DX = 30e-9                # ~15 cells per lambda_Si at 1550 nm
NPML = 15

R_OUTER = 2.00e-6
R_INNER = 1.80e-6
R_MEAN = 0.5 * (R_OUTER + R_INNER)
RING_WIDTH = R_OUTER - R_INNER
L_RT = 2.0 * np.pi * R_MEAN

# Bus: straight Si waveguide along the bottom, side-coupled to the ring.
W_BUS = 0.20e-6
GAP = 0.10e-6                       # ring-to-bus edge-to-edge gap
RING_CX = SIZE_X / 2
RING_CY = SIZE_Y - R_OUTER - GAP - W_BUS - 0.6e-6   # leave PML margin on top
BUS_Y = RING_CY - R_OUTER - GAP - W_BUS / 2          # bus centerline
SRC_X = 0.6e-6                      # source inside bus, past left PML
PROBE_X = SIZE_X - 0.6e-6           # through-port probe past the ring

N_SI = 3.48
EPS_SI = N_SI ** 2
N_SIO2 = 1.444
EPS_SIO2 = N_SIO2 ** 2

# 50% per round-trip power loss: alpha_p*L = -ln(0.5), alpha_p = sigma eta0 / n
ALPHA = -np.log(0.5) / L_RT
SIGMA_RING = N_SI * ALPHA / ETA0           # ~540 S/m for these numbers

# thermal: Si has k=130, dn/dT=1.8e-4; SiO2 cladding k=1.38, dn/dT=1.0e-5
K_SI, DNDT_SI = 130.0, 1.8e-4
K_SIO2, DNDT_SIO2 = 1.38, 1.0e-5

T_AMBIENT = 300.0

# --- sweep parameters ----------------------------------------------------------

# Pick a wavelength band around an expected WGM. Ring FSR ~ lambda^2/(n*L_RT)
# ~ 100 nm at 1550, so this window straddles ~1.5 resonances; the sweep will
# walk through at least one.
LAMBDA_MIN, LAMBDA_MAX = 1500e-9, 1620e-9
N_WAVELENGTHS = 50
STEPS_PER_POINT = 4000           # ~280 fs at dx=30nm -> ~55 optical periods
COUPLING_EVERY = 200             # EM steps between heat solves / TO updates

SOURCE_AMPLITUDE = 1.0e6         # crank until you see shark-fin asymmetry


# --- helpers -------------------------------------------------------------------

def _ring_mask(domain):
    """Ring annulus cells, the same circle tests as the painted Si / SiO2 circles."""
    return (domain.circle_mask(RING_CX, RING_CY, R_OUTER)
            & ~domain.circle_mask(RING_CX, RING_CY, R_INNER))


def _bus_mask(domain):
    nx, ny = domain.grid.nx, domain.grid.ny
    yy, xx = cp.meshgrid(cp.arange(ny), cp.arange(nx))
    iy_lo = domain.pos_to_cell_y(BUS_Y - W_BUS / 2)
    iy_hi = domain.pos_to_cell_y(BUS_Y + W_BUS / 2)
    return (yy >= iy_lo) & (yy <= iy_hi)


def _stamp_thermal(materials, ring_mask):
    """Silica cladding everywhere, silicon on the ring annulus (isotropic dn/dT: xx, yy and the E_z zz)."""
    materials.k_xx[:] = K_SIO2;     materials.k_yy[:] = K_SIO2
    materials.k_xx[ring_mask] = K_SI; materials.k_yy[ring_mask] = K_SI
    for dndT in (materials.dndT_xx, materials.dndT_yy, materials.dndT_zz):
        dndT[:] = DNDT_SIO2
        dndT[ring_mask] = DNDT_SI


class BusModeSource:
    """Soft CW source on a vertical column of cells, transverse profile shaped
    to the fundamental TE slab mode of the bus. Couples ~all power into the
    bus instead of radiating cylindrically like a single-cell point source."""

    def __init__(self, ix, iy_lo, iy_hi, frequency, amplitude=1.0,
                 ramp_periods=3.0, t_start=0.0):
        self.t_start = float(t_start)   # the ramp restarts here (each sweep point resets the EM fields)
        self.ix = int(ix)
        self.iy_lo = int(iy_lo)
        self.iy_hi = int(iy_hi)
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.omega = 2.0 * np.pi * self.frequency
        self.tau_ramp = ramp_periods / self.frequency
        n = self.iy_hi - self.iy_lo + 1
        # cosine bell across the bus cross-section, zero at the cladding edges
        s = np.arange(n) - (n - 1) / 2.0
        prof = np.cos(np.pi * s / n).astype(np.float32)
        self.profile = cp.asarray(prof, dtype=DTYPE)

    def inject(self, target, t):
        ramp = 1.0 - np.exp(-max(t - self.t_start, 0.0) / self.tau_ramp)
        val = self.amplitude * ramp * np.sin(self.omega * t)
        target[self.ix, self.iy_lo:self.iy_hi + 1] += DTYPE(val) * self.profile


def _reset_em_fields(em):
    em.engine.fields.ez.fill(0.0)
    em.engine.fields.hx.fill(0.0)
    em.engine.fields.hy.fill(0.0)
    em.engine.reset_q_accum()


# --- build the simulation -----------------------------------------------------

def build_sim():
    bc_em = BoundaryConfig.all_pml(NPML)
    domain = Domain.uniform(SIZE_X, SIZE_Y, DX, npml=NPML, boundary_config=bc_em)
    sim = Simulation.multiphysics(domain)

    em = EMPhysics(domain, boundary_config=bc_em)
    sim.add_physics('em', em)

    # SiO2 cladding on the whole grid (the oversized rectangle also covers the
    # PML ring), then the bus (lossless Si, also oversized in x so it runs
    # through the PML instead of ending abruptly at the window edge), then the
    # ring on top (lossy Si annulus) with a SiO2 core. Order matters because
    # rectangles/circles overwrite; the geometry keeps a finite bus-ring gap.
    em.add_dielectric_rectangle(-SIZE_X, -SIZE_Y, 2.0 * SIZE_X, 2.0 * SIZE_Y,
                                eps_r=EPS_SIO2, sigma_e=0.0)
    em.add_dielectric_rectangle(
        -SIZE_X, BUS_Y - W_BUS / 2, 2.0 * SIZE_X, BUS_Y + W_BUS / 2,
        eps_r=EPS_SI, sigma_e=0.0,
    )
    em.add_dielectric_circle(RING_CX, RING_CY, R_OUTER,
                             eps_r=EPS_SI, sigma_e=float(SIGMA_RING))
    em.add_dielectric_circle(RING_CX, RING_CY, R_INNER,
                             eps_r=EPS_SIO2, sigma_e=0.0)

    # Heat: Dirichlet 300 K on every face = perfect heatsink at the box edge.
    bc_heat = HeatBoundaryConfig(
        x_lo=BCSide('dirichlet', value=T_AMBIENT),
        x_hi=BCSide('dirichlet', value=T_AMBIENT),
        y_lo=BCSide('dirichlet', value=T_AMBIENT),
        y_hi=BCSide('dirichlet', value=T_AMBIENT),
    )
    heat = HeatPhysics(domain, bcs=bc_heat, T0=T_AMBIENT,
                       tol=1e-5, max_iter=400)
    sim.add_physics('heat', heat, every=COUPLING_EVERY)

    # Stamp Si thermal props on the ring annulus (the bus is lossless, so it
    # generates no heat and its k doesn't matter for steady state), SiO2
    # elsewhere; rebuild A.
    ring_mask = _ring_mask(domain)
    _stamp_thermal(domain.materials, ring_mask)
    heat.rebuild_operator()

    # The model's em_heat_joule / heat_em_thermo_optic couplings (every None = the heat stride); thermo_optic
    # rewrites eps right after each heat solve.
    sim.add_coupling(joule_heating(em='em', heat='heat'))
    sim.add_coupling(thermo_optic(heat='heat', em='em', permittivity_arrays=em.permittivity_arrays))

    return sim, em, heat, ring_mask


# --- the sweep ----------------------------------------------------------------

def sweep(sim, em, heat, ring_mask, through_mask, wavelengths, label,
          snapshot_best=None):
    print(f"\n[{label}] sweep over {len(wavelengths)} wavelengths")
    avg_intensity = np.zeros_like(wavelengths)
    avg_through = np.zeros_like(wavelengths)
    avg_dT = np.zeros_like(wavelengths)
    best = -1.0

    for i, lam in enumerate(wavelengths):
        # Replace the source with a BusModeSource at the new wavelength.
        # Driving Ez on a cosine-bell column across the bus cross-section
        # launches the guided slab mode directly; symmetric, so half goes
        # left into the PML and half goes right past the ring.
        ix_src = sim.domain.pos_to_cell_x(SRC_X)
        iy_lo = sim.domain.pos_to_cell_y(BUS_Y - W_BUS / 2)
        iy_hi = sim.domain.pos_to_cell_y(BUS_Y + W_BUS / 2)
        em.sources.clear()
        em.sources.append(BusModeSource(
            ix=ix_src,
            iy_lo=iy_lo,
            iy_hi=iy_hi,
            frequency=C0 / float(lam),
            amplitude=SOURCE_AMPLITUDE,
            t_start=sim.runner.t,
        ))
        # Fresh EM transient at each point; carry the thermal state to expose
        # the path-dependent (bistable) lineshape.
        _reset_em_fields(em)

        duration = STEPS_PER_POINT * float(em.engine.dt)
        sim.runner.advance(sys.maxsize, sim.runner.t + duration)

        ez = em.engine.fields.ez
        T = heat.fields.T
        I_ring = float(cp.mean((ez[ring_mask]) ** 2).item())
        I_through = float(cp.mean((ez[through_mask]) ** 2).item())
        dT_ring = float(cp.mean(T[ring_mask] - T_AMBIENT).item())
        avg_intensity[i] = I_ring
        avg_through[i] = I_through
        avg_dT[i] = dT_ring
        print(f"  {lam*1e9:6.1f} nm   ring={I_ring:.3e}   "
              f"through={I_through:.3e}   "
              f"<dT>_ring={dT_ring:6.2f} K   "
              f"heat iters={heat.last_iters}")

        if snapshot_best is not None and I_ring > best:
            best = I_ring
            try:
                sim.plot_field('ez',
                               title=f'Ez at peak so far ({label}, {lam*1e9:.1f} nm)',
                               filename=snapshot_best)
            except Exception as e:
                print(f"    snapshot failed: {e}")

    return avg_intensity, avg_through, avg_dT


def run():
    sim, em, heat, ring_mask = build_sim()
    domain = sim.domain
    bus_mask = _bus_mask(domain)
    # Through-port window: a few cells of bus at the right end, away from PML.
    ix_lo = domain.pos_to_cell_x(PROBE_X - 0.3e-6)
    ix_hi = domain.pos_to_cell_x(PROBE_X + 0.3e-6)
    through_mask = bus_mask.copy()
    through_mask[:ix_lo, :] = False
    through_mask[ix_hi:, :] = False

    wavelengths_up = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_WAVELENGTHS)
    wavelengths_dn = wavelengths_up[::-1]

    I_up, T_up, dT_up = sweep(sim, em, heat, ring_mask, through_mask,
                              wavelengths_up, "UP",
                              snapshot_best='ring_ez_peak_up.png')
    I_dn, T_dn, dT_dn = sweep(sim, em, heat, ring_mask, through_mask,
                              wavelengths_dn, "DOWN",
                              snapshot_best='ring_ez_peak_dn.png')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        ax1.plot(wavelengths_up * 1e9, I_up, 'r.-', label='up sweep')
        ax1.plot(wavelengths_dn * 1e9, I_dn, 'b.-', label='down sweep')
        ax1.set_ylabel(r'$\langle |E_z|^2 \rangle_{ring}$ (peak on res)')
        ax1.legend(); ax1.set_title('Thermo-optic shark-fin resonance')
        ax2.plot(wavelengths_up * 1e9, T_up, 'r.-')
        ax2.plot(wavelengths_dn * 1e9, T_dn, 'b.-')
        ax2.set_ylabel(r'$\langle |E_z|^2 \rangle_{through}$ (dip on res)')
        ax3.plot(wavelengths_up * 1e9, dT_up, 'r.-')
        ax3.plot(wavelengths_dn * 1e9, dT_dn, 'b.-')
        ax3.set_xlabel('Source wavelength (nm)')
        ax3.set_ylabel(r'$\langle \Delta T \rangle_{ring}$ (K)')
        plt.tight_layout()
        plt.savefig('ring_sharkfin.png', dpi=150)
        plt.close()
        print("\nSaved ring_sharkfin.png")

        sim.plot_field('ez', title='Ez at last wavelength',
                       filename='ring_ez.png')
        print("Saved ring_ez.png")
    except ImportError:
        print("matplotlib not available; skipping plots")

    return (sim,
            (wavelengths_up, I_up, T_up, dT_up),
            (wavelengths_dn, I_dn, T_dn, dT_dn))


if __name__ == '__main__':
    run()

"""
Open_sol Example: TFLN Waveguide Coupled to a Ring Resonator
============================================================

Top view of a thin-film lithium niobate (TFLN) bus waveguide side-coupled to a
ring resonator in SiO2 (all-pass filter). A broadband pulse launched into the
bus gives the through-port transmission (normalised to a bus-only reference
run) and, from a probe in the ring, the ring-down and intracavity spectrum.
A last pass records |Ez|^2 maps on and off resonance.

The core is the library material lithium_niobate_z_cut and the cladding is
silica. tfln_ring_resonator.json is the GUI version of the spectrum pass.

Model notes
-----------
- 2D, E_z polarisation: E_z is normal to the film. Only eps_zz enters, so LN's
  birefringence reduces to one index: n_e for the z-cut film (E_z along the
  optic axis), n_o for x-cut. The film is infinitely thick here; use slab
  effective indices for quantitative work.
- LN is transparent: a real TFLN ring has Q ~ 1e6 and would need nanoseconds of
  simulated time. The ring gets an artificial absorption (RING_ROUND_TRIP_LOSS)
  comparable to the bus coupling, so it sits near critical coupling with a
  loaded Q ~ 1e3 and rings down within a few ps. With 0 the ring is a lossless
  all-pass filter and the transmission dips almost vanish.
- Three passes (reference, spectrum, DFT maps), ~280k steps in total.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_sol import Simulation, C0, ETA0, LITHIUM_NIOBATE_Z_CUT, SILICA


# --- library materials (optical data at 1550 nm) ------------------------------

EPS_CORE = LITHIUM_NIOBATE_Z_CUT.optical['eps_zz']  # n_e^2 (E_z), n_e = 2.138; eps_r is in-plane n_o^2
EPS_CLAD = SILICA.optical['eps_r']                  # n = 1.444
N_CORE = EPS_CORE ** 0.5
N_CLAD = EPS_CLAD ** 0.5

# --- geometry -----------------------------------------------------------------

DX = 25e-9                       # ~29 cells per wavelength in LN
NPML = 15
WG_WIDTH = 0.45e-6               # bus and ring; single-mode for Ez below 0.47 um
RING_RADIUS = 5.0e-6             # ring centre line
GAP = 0.25e-6                    # bus-to-ring edge-to-edge gap
MARGIN = 1.5e-6                  # cladding between the structure and the PML

RING_OUTER_RADIUS = RING_RADIUS + WG_WIDTH / 2
RING_INNER_RADIUS = RING_RADIUS - WG_WIDTH / 2
RING_LENGTH = 2.0 * np.pi * RING_RADIUS
SIZE_X = 2.0 * (RING_OUTER_RADIUS + MARGIN)
BUS_Y = MARGIN                   # bus centre line; the bus runs along x
RING_CX = SIZE_X / 2
RING_CY = BUS_Y + WG_WIDTH / 2 + GAP + RING_OUTER_RADIUS
SIZE_Y = RING_CY + RING_OUTER_RADIUS + MARGIN

SOURCE_X = 1.0e-6
THROUGH_X = SIZE_X - 1.0e-6
LINE_HALF_SPAN = 1.0e-6          # half length of the source and flux lines across the bus

RING_ROUND_TRIP_LOSS = 0.12      # artificial power loss fraction per round trip (see docstring)

# --- spectrum / run -----------------------------------------------------------

WAVELENGTH_CENTER = 1550e-9
BANDWIDTH = 0.1                  # pulse 1/e field half-width, fraction of the centre frequency
WAVELENGTH_MIN, WAVELENGTH_MAX = 1480e-9, 1620e-9
N_WAVELENGTHS = 1401             # 0.1 nm spacing
RING_DOWN_TIME = 8.0e-12
DIP_MAX_TRANSMISSION = 0.7       # a resonance counts as a dip below this


def slab_mode(core_width, n_core, n_clad, wavelength):
    """Fundamental Ez mode of a symmetric slab: (effective index, group index, core power fraction,
    1/e field half-width [m])."""
    half_width = 0.5 * core_width
    k0 = 2.0 * np.pi / wavelength
    v_half = k0 * half_width * np.sqrt(n_core ** 2 - n_clad ** 2)
    # core_phase * tan(core_phase) = clad_decay has its first root in (0, min(v_half, pi/2)): bisect.
    phase_lo, phase_hi = 0.0, min(v_half, 0.5 * np.pi)
    for _ in range(60):
        core_phase = 0.5 * (phase_lo + phase_hi)
        if core_phase * np.tan(core_phase) < np.sqrt(v_half ** 2 - core_phase ** 2):
            phase_lo = core_phase
        else:
            phase_hi = core_phase
    core_phase = 0.5 * (phase_lo + phase_hi)                # k_y * half_width
    clad_decay = np.sqrt(v_half ** 2 - core_phase ** 2)     # decay constant * half_width

    n_eff = np.sqrt(n_core ** 2 - (core_phase / (k0 * half_width)) ** 2)
    core_energy = 1.0 + np.sin(2.0 * core_phase) / (2.0 * core_phase)
    clad_energy = np.cos(core_phase) ** 2 / clad_decay
    core_fraction = core_energy / (core_energy + clad_energy)
    # Non-dispersive media: n_group * n_eff = <n^2> weighted by |Ez|^2.
    n_group = (n_core ** 2 * core_fraction + n_clad ** 2 * (1.0 - core_fraction)) / n_eff

    edge_field = np.cos(core_phase)
    if edge_field > np.exp(-1.0):
        mode_half_width = half_width * (1.0 + (1.0 + np.log(edge_field)) / clad_decay)
    else:
        mode_half_width = half_width * np.arccos(np.exp(-1.0)) / core_phase
    return n_eff, n_group, core_fraction, mode_half_width


def ring_loss_conductivity(round_trip_loss, n_eff, core_fraction):
    """sigma_e [S/m] of the ring core for a power loss fraction `round_trip_loss` per round trip."""
    attenuation = -np.log(1.0 - round_trip_loss) / RING_LENGTH   # modal power attenuation [1/m]
    return attenuation * n_eff / (ETA0 * core_fraction)


def build_simulation(with_ring, mode_half_width, ring_sigma):
    """Bus (plus ring) in SiO2 with a pulsed source shaped like the bus mode."""
    sim = Simulation(size_x=SIZE_X, size_y=SIZE_Y, dx=DX, npml=NPML)

    # Cladding and bus continue through the PML ring, so the guided mode leaves without reflection.
    pml_overlap = 2 * NPML * DX
    sim.add_dielectric_rectangle(-pml_overlap, -pml_overlap, SIZE_X + pml_overlap, SIZE_Y + pml_overlap,
                                 eps_r=EPS_CLAD)
    sim.add_dielectric_rectangle(-pml_overlap, BUS_Y - WG_WIDTH / 2, SIZE_X + pml_overlap, BUS_Y + WG_WIDTH / 2,
                                 eps_r=EPS_CORE)
    if with_ring:
        sim.add_dielectric_circle(RING_CX, RING_CY, RING_OUTER_RADIUS, eps_r=EPS_CORE, sigma_e=ring_sigma)
        sim.add_dielectric_circle(RING_CX, RING_CY, RING_INNER_RADIUS, eps_r=EPS_CLAD)

    # Line across the bus (axis 'y' = along y at x = SOURCE_X); half the power goes left into the PML.
    sim.add_gaussian_beam(
        axis='y', position=SOURCE_X, waist=mode_half_width, center=BUS_Y,
        wavelength=WAVELENGTH_CENTER, bandwidth=BANDWIDTH, waveform='gaussian',
        span=(BUS_Y - LINE_HALF_SPAN, BUS_Y + LINE_HALF_SPAN),
    )
    return sim


def add_through_monitor(sim, wavelengths):
    return sim.add_flux_monitor(
        axis='y', position=THROUGH_X,
        span=(BUS_Y - LINE_HALF_SPAN, BUS_Y + LINE_HALF_SPAN),
        wavelengths=wavelengths,
    )


def probe_power_spectrum(probe, wavelengths):
    """|FFT|^2 of a probe series at `wavelengths` (zero-padded, so finer than 1 / run time)."""
    times, values = probe.get_data()
    n_fft = 8 * values.size
    frequencies = np.fft.rfftfreq(n_fft, times[1] - times[0])
    power = np.abs(np.fft.rfft(values, n=n_fft)) ** 2
    return np.interp(C0 / wavelengths, frequencies, power)


def find_dips(wavelengths, transmission, min_separation):
    """Indices of the local transmission minima below DIP_MAX_TRANSMISSION, at least `min_separation` [m] apart."""
    dips = []
    for index in np.argsort(transmission):
        if transmission[index] > DIP_MAX_TRANSMISSION:
            break
        if index in (0, transmission.size - 1):
            continue
        if transmission[index] > min(transmission[index - 1], transmission[index + 1]):
            continue
        if all(abs(wavelengths[index] - wavelengths[other]) >= min_separation for other in dips):
            dips.append(int(index))
    return sorted(dips)


def dip_fwhm(wavelengths, transmission, dip):
    """Full width [m] of a dip at half its depth below T = 1; nan when the dip runs off the band."""
    half_level = 0.5 * (1.0 + transmission[dip])

    def half_level_crossing(direction):
        inside = dip
        while transmission[inside + direction] < half_level:
            inside += direction
            if inside in (0, transmission.size - 1):
                return np.nan
        outside = inside + direction
        fraction = (half_level - transmission[inside]) / (transmission[outside] - transmission[inside])
        return wavelengths[inside] + fraction * (wavelengths[outside] - wavelengths[inside])

    return half_level_crossing(+1) - half_level_crossing(-1)


def run_tfln_ring():
    n_eff, n_group, core_fraction, mode_half_width = slab_mode(WG_WIDTH, N_CORE, N_CLAD, WAVELENGTH_CENTER)
    fsr_estimate = WAVELENGTH_CENTER ** 2 / (n_group * RING_LENGTH)
    ring_sigma = ring_loss_conductivity(RING_ROUND_TRIP_LOSS, n_eff, core_fraction)
    wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, N_WAVELENGTHS)

    print("--- TFLN ring setup ---")
    print(f"Materials: core {LITHIUM_NIOBATE_Z_CUT.name} (n = {N_CORE:.3f}), cladding {SILICA.name} "
          f"(n = {N_CLAD:.3f})")
    # n_eff, the core fraction and the half-width are the n_eff / core_fraction / w_mode parameters of the .json.
    print(f"Bus mode: n_eff = {n_eff:.3f}, n_group = {n_group:.3f}, core power fraction = {core_fraction:.3f}, "
          f"1/e half-width = {mode_half_width*1e9:.0f} nm")
    print(f"Ring: radius = {RING_RADIUS*1e6:.1f} um, gap = {GAP*1e9:.0f} nm, "
          f"sigma_e = {ring_sigma:.1f} S/m ({RING_ROUND_TRIP_LOSS:.0%} loss per round trip)")
    print(f"Estimated FSR = {fsr_estimate*1e9:.1f} nm")

    # Pass 1: bus only, the incident power spectrum at the through port.
    reference = build_simulation(False, mode_half_width, ring_sigma)
    flux_reference = add_through_monitor(reference, wavelengths)
    transit_time = (THROUGH_X - SOURCE_X) * n_group / C0
    reference.run(duration=reference.sources[0].temporal.duration + 3.0 * transit_time, print_every=5000)
    incident_power = flux_reference.flux()

    # Pass 2: bus + ring until the ring has emptied.
    sim = build_simulation(True, mode_half_width, ring_sigma)
    flux_through = add_through_monitor(sim, wavelengths)
    ring_probe = sim.add_field_probe(RING_CX, RING_CY + RING_RADIUS)
    sim.run(duration=RING_DOWN_TIME, print_every=20000)

    transmission = flux_through.flux() / incident_power
    ring_buildup = probe_power_spectrum(ring_probe, wavelengths) / incident_power
    dips = find_dips(wavelengths, transmission, min_separation=0.5 * fsr_estimate)

    print("\n--- TFLN ring results ---")
    if not dips:
        print(f"No resonance dip below T = {DIP_MAX_TRANSMISSION}: change GAP or RING_ROUND_TRIP_LOSS.")
    for dip in dips:
        fwhm = dip_fwhm(wavelengths, transmission, dip)
        extinction_db = -10.0 * np.log10(max(transmission[dip], 1e-12))
        print(f"  {wavelengths[dip]*1e9:7.2f} nm: T_min = {transmission[dip]:.3f} ({extinction_db:.1f} dB), "
              f"FWHM = {fwhm*1e9:.2f} nm, loaded Q = {wavelengths[dip]/fwhm:.0f}")
    if len(dips) > 1:
        fsr_measured = np.diff(wavelengths[dips]).mean()
        print(f"  FSR: measured {fsr_measured*1e9:.2f} nm, slab-mode estimate {fsr_estimate*1e9:.2f} nm")

    # Pass 3: |Ez|^2 maps at the resonance nearest the band centre and half an FSR away from it.
    dft_sim = dft_monitor = None
    if dips:
        on_resonance = wavelengths[min(dips, key=lambda dip: abs(wavelengths[dip] - WAVELENGTH_CENTER))]
        off_resonance = on_resonance + 0.5 * fsr_estimate
        dft_sim = build_simulation(True, mode_half_width, ring_sigma)
        dft_monitor = dft_sim.add_dft_monitor(0.0, 0.0, SIZE_X, SIZE_Y,
                                              wavelengths=np.array([on_resonance, off_resonance]))
        dft_sim.run(duration=RING_DOWN_TIME, print_every=20000)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sim.plot_setup(filename='tfln_ring_setup.png')

        wavelengths_nm = wavelengths * 1e9
        fig, (ax_through, ax_ring, ax_time) = plt.subplots(3, 1, figsize=(10, 10))

        ax_through.plot(wavelengths_nm, transmission, 'b-', linewidth=1.2)
        for dip in dips:
            ax_through.axvline(wavelengths_nm[dip], color='gray', linestyle=':', linewidth=0.8)
        ax_through.set_xlabel('Wavelength (nm)')
        ax_through.set_ylabel('Through-port transmission')
        ax_through.set_title(f'TFLN ring: R = {RING_RADIUS*1e6:.0f} µm, gap = {GAP*1e9:.0f} nm')
        ax_through.grid(True, alpha=0.3)

        ax_ring.plot(wavelengths_nm, ring_buildup / ring_buildup.max(), 'r-', linewidth=1.2)
        ax_ring.set_xlabel('Wavelength (nm)')
        ax_ring.set_ylabel('Ring |Ez|² / incident power (norm.)')
        ax_ring.grid(True, alpha=0.3)

        probe_times, probe_ez = ring_probe.get_data()
        ax_time.plot(probe_times * 1e12, probe_ez, 'k-', linewidth=0.4)
        ax_time.set_xlabel('Time (ps)')
        ax_time.set_ylabel('Ez in the ring (V/m)')
        ax_time.set_title('Ring-down')
        ax_time.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('tfln_ring_spectrum.png', dpi=150)
        plt.close()
        print("Saved tfln_ring_spectrum.png")

        if dft_monitor is not None:
            dft_sim.viz.plot_dft_fields(dft_monitor, filename='tfln_ring_dft.png')

    except ImportError:
        print("matplotlib not available, skipping plots")

    return sim


if __name__ == '__main__':
    run_tfln_ring()

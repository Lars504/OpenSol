"""
Open_sol Example: Plasmonic Gold Nanoparticle
=============================================

Localized surface plasmon (LSPR) of a Drude gold cylinder in H_z polarisation: only in-plane E puts charge on the
surface (with E_z along the axis a cylinder has no LSPR). A TFSF plane-wave pulse lights the particle; closed flux
boxes give the scattered and absorbed power, divided by the incident intensity of an empty-domain reference run:
the efficiencies Q_sca, Q_abs and Q_ext. The quasi-static dipole sits where Re eps = -eps_host, i.e. omega_p /
sqrt(eps_inf + eps_host) (~453 nm for the library gold, which has no interband terms); retardation red-shifts it.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_sol import Simulation, C0, GOLD, dft_result


# Layout of the validated em_hz_mie_drude_cylinder selftest: 2.5 nm cells, window cell k spans [2.5 k, 2.5 (k + 1)] nm.
DOMAIN_SIZE = 1.2e-6
CELL = 2.5e-9
NPML = 20
CENTER = 0.6e-6                   # particle axis at (CENTER, CENTER)
RADIUS = 50e-9                    # 20 cells
EPS_HOST = 1.0                    # vacuum around the particle (the TFSF box edges must be in vacuum)
TFSF_BOX = (0.35e-6, 0.35e-6, 0.85e-6, 0.85e-6)
PULSE_WAVELENGTH = 480e-9
PULSE_BANDWIDTH = 0.6
WAVELENGTHS = np.linspace(380e-9, 650e-9, 55)
OFF_RESONANCE_WAVELENGTH = 620e-9

# Flux lines at cell centres measure through the cell's hi face: the incident line through x = 0.6 um over 0.4 um,
# the scattered box through the faces at 0.3 / 0.9 um (scattered-field region), the absorbed box at 0.5 / 0.7 um
# (total-field region, around the particle).
INCIDENT_LINE_X = 0.59875e-6
INCIDENT_SPAN = (0.40125e-6, 0.79875e-6)
INCIDENT_LENGTH = 0.4e-6
SCATTERED_BOX = ((0.29875e-6, 0.89875e-6), (0.30125e-6, 0.89875e-6))   # (line positions, span)
ABSORBED_BOX = ((0.49875e-6, 0.69875e-6), (0.50125e-6, 0.69875e-6))

REFERENCE_DURATION = 50e-15       # the vacuum pulse has crossed the domain
DEVICE_DURATION = 150e-15         # Drude ring-down exp(-gamma t / 2) down to ~5e-4


def quasi_static_resonance(drude, eps_host):
    """Vacuum wavelength [m] where Re eps of the Drude pole = -eps_host: the 2D quasi-static dipole plasmon."""
    omega_sp = math.sqrt(drude.omega_p ** 2 / (drude.eps_inf + eps_host) - drude.gamma ** 2)
    return 2.0 * math.pi * C0 / omega_sp


def build_simulation(with_particle):
    """All-PML H_z domain lit by the +x TFSF pulse, with or without the gold cylinder."""
    sim = Simulation(size_x=DOMAIN_SIZE, size_y=DOMAIN_SIZE, dx=CELL, npml=NPML, polarization='hz')
    if with_particle:
        sim.add_drude_circle(CENTER, CENTER, RADIUS, GOLD.em)
    sim.add_tfsf(*TFSF_BOX, wavelength=PULSE_WAVELENGTH, bandwidth=PULSE_BANDWIDTH)
    return sim


def add_flux_box(sim, line_positions, span):
    """Closed box of four flux lines: vertical ones at x = line_positions, horizontal ones at y = line_positions."""
    lo, hi = line_positions
    return {'left': sim.add_flux_monitor('y', lo, span, wavelengths=WAVELENGTHS),
            'right': sim.add_flux_monitor('y', hi, span, wavelengths=WAVELENGTHS),
            'bottom': sim.add_flux_monitor('x', lo, span, wavelengths=WAVELENGTHS),
            'top': sim.add_flux_monitor('x', hi, span, wavelengths=WAVELENGTHS)}


def net_outflow(flux_box):
    """Outward spectral flux (right - left + top - bottom) of a closed flux box."""
    power = {side: monitor.flux() for side, monitor in flux_box.items()}
    return power['right'] - power['left'] + power['top'] - power['bottom']


def run_plasmonic():
    gold = GOLD.em
    lspr_quasi_static = quasi_static_resonance(gold, EPS_HOST)
    print("--- Plasmonic gold setup ---")
    print(f"Gold cylinder: radius = {RADIUS*1e9:.0f} nm, H_z polarisation (in-plane E)")
    print(f"Drude: wp = {gold.omega_p:.2e} rad/s, gamma = {gold.gamma:.2e} rad/s, eps_inf = {gold.eps_inf}")
    print(f"Quasi-static LSPR (Re eps = -{EPS_HOST:g}): {lspr_quasi_static*1e9:.0f} nm")

    # Pass 1: empty domain, the incident intensity (power per unit length) of the TFSF pulse.
    reference = build_simulation(False)
    incident = reference.add_flux_monitor('y', INCIDENT_LINE_X, INCIDENT_SPAN, wavelengths=WAVELENGTHS)
    reference.run(duration=REFERENCE_DURATION, print_every=5000)
    incident_intensity = incident.flux() / INCIDENT_LENGTH

    # Pass 2: the particle, scattered (outer box) and absorbed (inner box, net inflow) power, near-field maps.
    sim = build_simulation(True)
    scattered_box = add_flux_box(sim, *SCATTERED_BOX)
    absorbed_box = add_flux_box(sim, *ABSORBED_BOX)
    near_field = sim.add_dft_monitor(
        CENTER - 2 * RADIUS, CENTER - 2 * RADIUS, CENTER + 2 * RADIUS, CENTER + 2 * RADIUS,
        wavelengths=np.array([lspr_quasi_static, OFF_RESONANCE_WAVELENGTH]),
    )
    sim.run(duration=DEVICE_DURATION, print_every=5000)

    geometric_power = incident_intensity * 2.0 * RADIUS
    q_sca = net_outflow(scattered_box) / geometric_power
    q_abs = -net_outflow(absorbed_box) / geometric_power
    q_ext = q_sca + q_abs

    print("\n--- Plasmonic results ---")
    for label, efficiency in (('Q_sca', q_sca), ('Q_abs', q_abs), ('Q_ext', q_ext)):
        peak = int(np.argmax(efficiency))
        print(f"{label} peak: {efficiency[peak]:.2f} at {WAVELENGTHS[peak]*1e9:.0f} nm")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return sim

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for label, efficiency in (('Q_sca', q_sca), ('Q_abs', q_abs), ('Q_ext', q_ext)):
        ax.plot(WAVELENGTHS * 1e9, efficiency, label=label)
    ax.axvline(lspr_quasi_static * 1e9, color='g', linestyle='--', alpha=0.5, label='quasi-static LSPR')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Efficiency (per diameter)')
    ax.set_title('Cylinder efficiencies')
    ax.legend()

    # |E|^2 (E_x, E_y) over the DFT box at the quasi-static resonance and off resonance.
    for i, ax in enumerate(axes[1:]):
        result = dft_result(near_field, sim.domain, freq_index=i)
        extent_nm = [edge * 1e9 for edge in result['extent']]
        im = ax.imshow(result['intensity'].T, cmap='hot', origin='lower', aspect='equal', extent=extent_nm)
        ax.set_title(f"|E|² at {result['wavelength'][i]*1e9:.0f} nm")
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Gold Nanoparticle LSPR (H_z polarisation)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plasmonic_gold.png', dpi=150)
    plt.close()
    print("Saved plasmonic_gold.png")

    return sim


if __name__ == '__main__':
    run_plasmonic()

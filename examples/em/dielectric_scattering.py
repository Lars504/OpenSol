"""
Open_sol Example: Dielectric Cylinder Scattering
================================================

A Gaussian pulse point source scatters off a high-index dielectric cylinder.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_sol import Simulation, C0, dft_result


def run_dielectric_scattering():
    wavelength = 550e-9
    domain_size = 8e-6
    npml = 15

    cylinder_radius = 1e-6
    cylinder_eps = 4.0
    # 20 cells per centre wavelength inside the cylinder (~15 at the pulse's shortest significant wavelength)
    dx = wavelength / (20 * np.sqrt(cylinder_eps))

    sim = Simulation(size_x=domain_size, size_y=domain_size, dx=dx, npml=npml)

    cx, cy = domain_size / 2, domain_size / 2
    sim.add_dielectric_circle(cx, cy, cylinder_radius, eps_r=cylinder_eps)

    # z-directed filament current of 1 mA (the default `current`).
    sim.add_point_source(
        domain_size * 0.25, domain_size / 2,
        wavelength=wavelength, bandwidth=0.3, waveform='gaussian',
    )

    probe_trans = sim.add_field_probe(domain_size * 0.75, domain_size / 2)
    probe_refl = sim.add_field_probe(domain_size * 0.15, domain_size / 2)

    wavelengths_monitor = np.array([500e-9, 550e-9, 600e-9])
    dft_mon = sim.add_dft_monitor(
        0, 0, domain_size, domain_size, wavelengths=wavelengths_monitor,
    )

    duration = 3 * domain_size / C0
    sim.run(duration=duration, print_every=1000)

    print("\n--- Results ---")
    _, v_trans = probe_trans.get_data()
    _, v_refl = probe_refl.get_data()
    print(f"Transmission probe max: {np.max(np.abs(v_trans)):.4e}")
    print(f"Reflection probe max: {np.max(np.abs(v_refl)):.4e}")

    for i, wl in enumerate(wavelengths_monitor):
        intensity = dft_result(dft_mon, sim.domain, freq_index=i)['intensity']
        print(f"DFT intensity at {wl*1e9:.0f} nm: max={intensity.max():.4e}")

    try:
        import matplotlib  # noqa: F401  (availability check only: sim.viz builds pyplot-free figures)
    except ImportError:
        print("matplotlib not available, skipping plots")
        return sim

    sim.plot_field('ez', title='Ez field - Cylinder Scattering',
                   filename='field_snapshot.png')
    sim.viz.plot_probes([probe_refl, probe_trans],
                        labels=['Reflection probe', 'Transmission probe'],
                        filename='time_signals.png')
    sim.viz.plot_dft_fields(dft_mon, filename='dft_intensity.png')
    sim.viz.plot_spectrum(probe_trans, wavelength_range=(300e-9, 900e-9),
                          label='Transmission probe spectrum', filename='spectrum.png')

    return sim


if __name__ == '__main__':
    run_dielectric_scattering()

"""
Open_sol Example: Thin-Film Interference (Fabry-Pérot)
======================================================

Validates transmission/reflection spectra against analytic Fresnel for a
dielectric slab. A normally incident plane wave (periodic y) is normalised
by an empty-domain reference run.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_sol import Simulation, C0


def analytic_transmission(wavelengths, n_film, thickness):
    k = 2 * np.pi * n_film / wavelengths
    delta = k * thickness
    r12 = (1 - n_film) / (1 + n_film)
    R = r12 ** 2
    T = ((1 - R) ** 2) / (1 - 2 * R * np.cos(2 * delta) + R ** 2)
    return T


def run_thin_film():
    wavelength_center = 600e-9
    # 25 nm puts both film faces on cell edges (24 nm would stair-step the film to 480 nm).
    dx = 25e-9
    domain_x = 6e-6
    domain_y = 3e-6
    npml = 15

    n_film = 1.5
    film_thickness = 500e-9

    wavelengths = np.linspace(400e-9, 800e-9, 50)

    film_center_x = domain_x / 2
    film_x0 = film_center_x - film_thickness / 2
    film_x1 = film_center_x + film_thickness / 2
    source_x = domain_x * 0.2
    duration = 4 * domain_x / C0

    def run_case(with_film):
        """Pulse run with or without the film: (sim, transmitted flux, net flux at the reflection plane, Ez snapshot)."""
        # PML on x, periodic y: the full-height line source launches a plane wave toward +x and -x.
        sim = Simulation(size_x=domain_x, size_y=domain_y, dx=dx, npml=npml,
                         boundaries='periodic_y')
        if with_film:
            sim.add_dielectric_rectangle(film_x0, 0, film_x1, domain_y,
                                          eps_r=n_film ** 2)
            # The analytic curve assumes the meshed film is exactly film_thickness thick.
            x_centers, _ = sim.domain.cell_centers()
            x_widths, _ = sim.domain.cell_widths()
            in_film = (x_centers >= film_x0) & (x_centers < film_x1)
            meshed_thickness = float(x_widths[in_film].sum())
            assert abs(meshed_thickness - film_thickness) < 1e-3 * dx, \
                f"meshed film is {meshed_thickness * 1e9:.1f} nm, not {film_thickness * 1e9:.1f} nm"

        # axis 'y' = line along y at x = position.
        source = sim.add_line_source(
            axis='y', position=source_x,
            wavelength=wavelength_center, bandwidth=0.8,
            waveform='gaussian',
        )

        # Same convention for monitors; a line along y measures flux > 0 toward +x.
        flux_trans = sim.add_flux_monitor(
            axis='y', position=domain_x * 0.75, span=(0, domain_y),
            wavelengths=wavelengths,
        )
        flux_refl = sim.add_flux_monitor(
            axis='y', position=domain_x * 0.3, span=(0, domain_y),
            wavelengths=wavelengths,
        )

        # Pause for the Ez snapshot when the pulse peak reaches the film centre.
        t_snapshot = source.temporal.t0 + (film_center_x - source_x) / C0
        sim.run(duration=t_snapshot, print_every=1000)
        ez_snapshot = sim.get_field_snapshot('ez')
        sim.run(duration=duration - t_snapshot, print_every=1000)
        return sim, flux_trans.flux(), flux_refl.flux(), ez_snapshot

    _, incident_trans, incident_refl, _ = run_case(with_film=False)
    sim, film_trans, film_refl, ez_snapshot = run_case(with_film=True)

    T_sim = film_trans / incident_trans
    # The reflection-plane monitor sees incident minus reflected power.
    R_sim = 1.0 - film_refl / incident_refl
    T_analytic = analytic_transmission(wavelengths, n_film, film_thickness)
    R_analytic = 1.0 - T_analytic   # lossless slab

    print("\n--- Thin-Film Validation ---")
    print(f"Film: n={n_film}, thickness={film_thickness*1e9:.0f} nm")
    print(f"Wavelength range: {wavelengths[0]*1e9:.0f}-{wavelengths[-1]*1e9:.0f} nm")
    print(f"max |T_fdtd - T_analytic| = {np.max(np.abs(T_sim - T_analytic)):.3e}")
    print(f"max |R_fdtd - R_analytic| = {np.max(np.abs(R_sim - R_analytic)):.3e}")
    print(f"max |R_fdtd + T_fdtd - 1| = {np.max(np.abs(R_sim + T_sim - 1.0)):.3e}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return sim

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    ax.plot(wavelengths * 1e9, T_analytic, 'b-', label='T analytic (Fresnel)', linewidth=2)
    ax.plot(wavelengths * 1e9, T_sim, 'b--', label='T FDTD', linewidth=2)
    ax.plot(wavelengths * 1e9, R_analytic, 'r-', label='R analytic (Fresnel)', linewidth=2)
    ax.plot(wavelengths * 1e9, R_sim, 'r--', label='R FDTD', linewidth=2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmission / Reflection')
    ax.set_title(f'Thin-Film Spectra (n={n_film}, d={film_thickness*1e9:.0f} nm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    vmax = max(abs(ez_snapshot.min()), abs(ez_snapshot.max())) or 1
    ax2.imshow(ez_snapshot.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               origin='lower', aspect='equal',
               extent=np.array(sim.domain.physical_extent()) * 1e6)
    for x_face in (film_x0, film_x1):
        ax2.axvline(x_face * 1e6, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('x (µm)')
    ax2.set_ylabel('y (µm)')
    ax2.set_title('Ez as the pulse reaches the film')

    plt.tight_layout()
    plt.savefig('thin_film_validation.png', dpi=150)
    plt.close()
    print("Saved thin_film_validation.png")

    return sim


if __name__ == '__main__':
    run_thin_film()

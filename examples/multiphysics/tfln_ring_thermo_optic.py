"""
Open_sol example: thermo-optic self-heating of the TFLN ring resonator (FDFD + heat), headless
=============================================================================================

tfln_ring_thermo_optic.json is em/tfln_ring_resonator_fdfd.json (same ring, gap, artificial ring loss and 25 nm
mesh) plus a heat physics: the power the ring absorbs heats it (em_heat_joule), the temperature raises its index
(heat_em_thermo_optic), and the resonance red-shifts with the input power P_in.

The model's study does the work: a coupled frequency_domain scan from blue to red (each wavelength an FDFD + heat
fixed point continued from the previous wavelength, relaxed by the study setting relaxation = w_relax), swept over
P_in = 1 and 4000 W/m. At 4000 W/m the ring stays hot while the scan drags the resonance along: the dip becomes the
thermal triangle, its minimum moves to the red, and past it the ring drops back to the cold branch.

The script
1. locates the cold resonance: the same model without the sweep at P_in = 1 W/m over one free spectral range in
   1 nm steps (parabola through the smallest through-port transmission);
2. runs the model's P_in sweep with lambda_cold set to it (scan lambda_cold - 3 nm .. + 12 nm in 0.25 nm steps);
3. prints the dip shift and ring temperature per power and saves the overlays (api Results.sweep_figure) plus the
   T and |E_z| maps of the hottest wavelength of the last power.

Model notes
-----------
- 2D top view: powers are per m of z (W/m). Heat spreads in plane and leaves through the buried oxide to the
  substrate (out_of_plane_convection, h = k_silica / t_box on a t_film thick film); P_in * t_film is roughly the
  device power (4 kW/m * 0.6 um = 2.4 mW).
- The ring loss is artificial (round_trip_loss, as in the FDFD example): loaded Q ~ 1e3, cold linewidth ~1.5 nm, so
  kW/m powers are needed.
- heat_em_thermo_optic also applies the in-plane thermal expansion through alpha as an equivalent index change
  n alpha dT (library LN alpha 1.5e-5 /K); override "alpha" on the LN materials (0.0 = thermo-optic effect alone).
- The relaxation is one fixed factor per power (w_relax = 1 / (1 + P_in / P_relax)): it damps the oscillating hot
  branch but slows the off-resonance wavelengths (about 1 - w per iteration). The log has one line per wavelength
  with its iteration count; summary sweep 'converged' says whether every wavelength of a power converged.
- Cost: one sparse factorisation per wavelength (~3.3e5 unknowns, needs scipy); the further fixed-point solves at
  that wavelength run GMRES on that factor.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_sol import api, plot_field_array

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tfln_ring_thermo_optic.json')
OUT_DIR = 'tfln_ring_thermo_optic_out'

COLD_POWER = 1.0                                  # W/m, heats the ring by ~0.01 K
COLD_SCAN = {'wavelength_start': 1532e-9, 'wavelength_stop': 1568e-9, 'n_wavelengths': 37}   # one FSR, 1 nm steps


def through_transmission(result):
    """(wavelengths [m], through-port transmission) of a through_port mode_monitor result."""
    return np.asarray(result['wavelength']), np.asarray(result['series']['fraction_positive'])


def transmission_dip(wavelengths, transmission):
    """Wavelength [m] of the smallest transmission, refined by a parabola through it and its neighbours."""
    index = int(np.argmin(transmission))
    if index in (0, len(transmission) - 1):
        return float(wavelengths[index])
    before, at, after = transmission[index - 1:index + 2]
    step = wavelengths[index + 1] - wavelengths[index]
    return float(wavelengths[index] + 0.5 * step * (before - after) / (before - 2.0 * at + after))


def find_cold_resonance(spec):
    """Cold through-port dip [m]: the model without its sweep at COLD_POWER over COLD_SCAN."""
    cold_spec = spec.copy()
    cold_spec.study.sweep = None
    cold_spec.study.settings.update(COLD_SCAN)
    cold_spec.parameter('P_in').expression = repr(COLD_POWER)
    with api.run(cold_spec, progress=True) as results:
        wavelengths, transmission = through_transmission(results.result('em.through_port'))
    dip = transmission_dip(wavelengths, transmission)
    print(f"\nCold resonance near {dip*1e9:.2f} nm (T_min = {transmission.min():.3f} on the 1 nm grid)")
    return dip


def report_sweep(results):
    """Dip wavelength, shift and ring temperature rise per swept power; overlay figures; maps of the hottest point."""
    sweep = results.sweep()
    through = results.sweep_result('em', 'through_port')
    print("\n--- thermo-optic resonance shift ---")
    cold_dip = None
    for power, result, series, converged in zip(sweep['values'], through, sweep['derived_series'],
                                                sweep['converged']):
        if result is None or series is None:
            print(f"  P_in = {power:7g} W/m: failed")
            continue
        wavelengths, transmission = through_transmission(result)
        dip_index = int(np.argmin(transmission))
        dip = wavelengths[dip_index]
        cold_dip = dip if cold_dip is None else cold_dip
        series_wavelengths = np.asarray(series['x'])
        ring_rise = np.asarray(series['series']['dT_ring'])
        dip_rise = ring_rise[int(np.argmin(np.abs(series_wavelengths - dip)))]
        print(f"  P_in = {power:7g} W/m: dip at {dip*1e9:8.2f} nm (T = {transmission[dip_index]:.3f}), "
              f"shift {(dip - cold_dip)*1e9:5.2f} nm, ring dT there {dip_rise:6.1f} K "
              f"(max {ring_rise.max():6.1f} K), every wavelength converged: {converged}")

    for quantity, series_name, filename in (('em.through_port', 'fraction_positive', 'tfln_ring_thermo_optic.png'),
                                            ('derived_series:dT_ring', None, 'tfln_ring_thermo_optic_dT.png'),
                                            ('derived_series:A_ring', None, 'tfln_ring_thermo_optic_A.png')):
        print(f"Saved {results.sweep_figure(quantity, filename, series=series_name)}")

    # the live snapshots are the last power's, one per wavelength in scan order (= its derived series points)
    last_series = sweep['derived_series'][-1]
    if last_series is not None:
        hottest = int(np.argmax(last_series['series']['A_ring']))
        wavelength = last_series['x'][hottest]
        for name, filename in (('heat.T', 'tfln_ring_thermo_optic_T.png'),
                               ('em.ez_abs', 'tfln_ring_thermo_optic_ez.png')):
            field_data = results.snapshot_field(hottest, name)
            plot_field_array(field_data, title=f"{field_data.info.label} at {wavelength*1e9:.2f} nm, "
                                               f"P_in = {sweep['values'][-1]:g} W/m (largest absorption)",
                             filename=filename)
            print(f"Saved {filename}")


def run_thermo_optic_ring():
    spec = api.load(MODEL_FILE)
    cold_resonance = find_cold_resonance(spec)
    with api.run(spec, out_dir=OUT_DIR, progress=True, parameters={'lambda_cold': cold_resonance}) as results:
        print(f"run: {results.status} (summary, log and results in {OUT_DIR})")
        report_sweep(results)


if __name__ == '__main__':
    run_thermo_optic_ring()

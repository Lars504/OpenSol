"""Time-filter eigenfrequency solve of the 2D FDFD types (solvers.time_filter): an FDTD run of the same structure
(EMPhysics on the FDFD Domain, its PEC cells and dispersive poles) from seeded broadband point sources, a running DFT
of the whole grid at the sampled frequencies, then the Rayleigh-Ritz projection of the assembled FDFD pencil onto those
fields and the resonance filter of eigenfrequency.solve_resonances -> the same EigenSolve, without a factorisation."""
# Pencil as solve_resonances (eigenfrequency header): A = L + k_t^2 M, A u = mu M u with mu = k_t^2 - k^2, so theta =
# 1 / (k^2 - k_t^2) = -1 / mu feeds _filtered_candidates (shift_invert map) and its polish by D. Field layout: the FDFD
# unknown vector is the full grid (PML and wall cells included), flat i * ny + j, exactly the DFT monitor's (nx, ny)
# box over every cell; pinned rows (PEC, outermost wall / mirror cells: mass 0) are zeroed in the basis. Conventions:
# both use exp(+j omega t) phasors (the DFT phase is exp(-j omega t)); the Hz DFT phase offset (t - dt / 2) is one
# scalar per sampled frequency: irrelevant for a span. Not supported (validation refuses them, this module raises):
# a Bloch phase, polar coordinates, subpixel smoothing, conformal PEC, conducting sheets (the FDTD would differ).
import math
import time

import numpy as np

from ...core.backend import to_host
from .eigenfrequency import (EigenSolve, _energy_weights, _filtered_candidates, _PencilMass, _resonance,
                             _ResonanceFilter, region_filter)


def unsupported_reason(physics) -> str:
    """'' when the FDFD physics can run a time-filter solve, else the feature that stops it (validation reuses it)."""
    if physics.boundary_config.has_bloch_phase:
        return 'a Bloch phase'
    if physics.metric is not None:
        return 'polar coordinates'
    if physics.subpixel_smoothing is not None:
        return 'subpixel smoothing'
    if physics.conformal_pec is not None or physics.conformal_locator is not None:
        return 'conformal boundaries'
    if physics.conducting_sheets:
        return 'conducting sheets'
    return ''


def _source_cells(physics, pinned, count, generator):
    """(x, y) physical positions of `count` distinct window cells that are not pinned (PEC / wall), seeded."""
    window = physics.domain.physical_window()
    centres = physics.domain.cell_centers()
    open_cells = ~np.asarray(to_host(pinned), dtype=bool)
    rows = np.arange(open_cells.shape[0])[window[0]]
    columns = np.arange(open_cells.shape[1])[window[1]]
    candidates = [(i, j) for i in rows for j in columns if open_cells[i, j]] if open_cells.size <= 4_000_000 else []
    if not candidates:
        raise ValueError('time-filter eigen solve: no free window cell for its sources')
    picks = generator.choice(len(candidates), size=min(int(count), len(candidates)), replace=False)
    return [(float(centres[0][candidates[k][0]]), float(centres[1][candidates[k][1]])) for k in picks]


def filtered_fields(physics, solver, pinned, n_resonances, target_omega):
    """(basis (n, m) complex128 of the run backend, details) of one FDTD run: seeded point sources (ez: filament
    currents, hz: magnetic line currents) with a Gaussian pulse centred on the target, the DFT of every cell at the
    sampled frequencies accumulated from step 0 over run_periods target periods (+ the pulse); pinned = xp bool (nx,
    ny) cells of the FDFD identity rows; target_omega [rad/s] = the system's."""
    from ...core.sources import GaussianPulse
    from ...solvers import time_filter as tf
    from . import EMPhysics
    from .monitors import DFTMonitor
    reason = unsupported_reason(physics)
    if reason:
        raise ValueError(f'time-filter eigen solve: {reason} is not supported (use arpack_shift_invert)')
    domain, polarization = physics.domain, physics.polarization
    started = time.perf_counter()
    fdtd = EMPhysics(domain, polarization, boundary_config=physics.boundary_config, precision=solver.precision)
    if physics.pec_cells is not None:
        fdtd.add_pec_region(physics.pec_cells)
    xp = domain.xp
    for pole, box, mask in physics.dispersive_regions:   # (pole, box, cropped mask); box None = everywhere
        if box is None:
            full = xp.ones(tuple(pinned.shape), dtype=bool)
        else:
            full = xp.zeros(tuple(pinned.shape), dtype=bool)
            full[box] = xp.asarray(mask, dtype=bool)
        fdtd.add_dispersive(pole, full)
    target = target_omega / (2.0 * math.pi)
    count = tf.sampled_count(n_resonances, solver.sampled_frequencies)
    frequencies = tf.sampled_frequencies(target, solver.bandwidth, count)
    pulse_bandwidth = min(1.0, tf.PULSE_BANDWIDTH_FACTOR * solver.bandwidth)
    generator = np.random.default_rng(tf.SOURCE_SEED)
    for x, y in _source_cells(physics, pinned, solver.sources, generator):
        amplitude = float(generator.choice((-1.0, 1.0)) * (0.5 + generator.random()))
        if polarization == 'ez':
            fdtd.add_point_source(x, y, frequency=target, bandwidth=pulse_bandwidth, current=1e-3 * amplitude)
        else:
            fdtd.add_magnetic_line_current(x, y, frequency=target, bandwidth=pulse_bandwidth,
                                           magnetic_current=amplitude)
    grid = domain.grid
    bc = physics.boundary_config
    monitor = DFTMonitor(slice(0, grid.nx), slice(0, grid.ny), frequencies, float(grid.dt), polarization=polarization,
                         periodic_x=bc.periodic_x, periodic_y=bc.periodic_y, xp=xp)
    fdtd.monitors.append(monitor)
    fdtd.finalize()
    pulse_steps = int(math.ceil(GaussianPulse(target, pulse_bandwidth).duration / float(grid.dt)))
    steps = pulse_steps + tf.periods_steps(tf.run_periods(solver.bandwidth, solver.periods), target, float(grid.dt))
    stepping = time.perf_counter()
    for _ in range(steps):
        fdtd.step()
    spectra = monitor.ez_dft if polarization == 'ez' else monitor.hz_dft
    basis = xp.ascontiguousarray(spectra.reshape(count, -1).T).astype(np.complex128)
    to_host(basis[:1, :1])   # wait for the device: the stepping time is complete
    step_seconds = time.perf_counter() - stepping
    solver.count_run(steps, count, step_seconds)
    del fdtd, monitor, spectra
    return basis, {'steps': steps, 'sampled': count, 'frequencies': frequencies, 'dt': float(grid.dt),
                   'step_seconds': step_seconds, 'setup_seconds': stepping - started,
                   'sources': solver.sources, 'pulse_steps': pulse_steps}


def solve_time_filter_resonances(physics, system, solver, n_resonances, pinned, energy_weights, in_window, field_of,
                                 symmetrizer=None, in_region=None, min_region_fraction=0.0):
    """EigenSolve of up to n_resonances resonances nearest the target of the FDFD system (its backend operator, mass
    from the cell permittivity / permeability at the target, 0 on pinned rows) from one filtered FDTD run (module
    header); the same filter, polish and info keys as solve_resonances (factor_text describes the run)."""
    started = time.perf_counter()
    from ...core.constants import C0
    xp = physics.xp
    cell_mass = system.cell_eps if system.polarization == 'ez' else system.cell_mu
    mass_host = np.asarray(to_host(cell_mass), dtype=np.complex128).ravel().copy()
    mass_host[np.asarray(to_host(pinned), dtype=bool).ravel()] = 0.0
    mass = xp.asarray(mass_host)
    operator = system.operator
    n_unknowns = mass_host.size
    k_target_squared = (system.omega / C0) ** 2
    basis, run = filtered_fields(physics, solver, pinned, n_resonances, system.omega)
    device_symmetrizer = None if symmetrizer is None else xp.asarray(np.asarray(symmetrizer, dtype=np.complex128))
    wanted = int(n_resonances) + 2
    pairs = solver.ritz_pairs(lambda block: operator @ block, mass, basis, wanted, 0.0,
                              symmetrizer=device_symmetrizer, xp=xp)
    del basis
    mu = pairs['mu']
    vectors = np.asarray(to_host(pairs['vectors']))
    nonzero = np.abs(mu) > 0.0
    theta = np.zeros(mu.shape, dtype=np.complex128)
    theta[nonzero] = -1.0 / mu[nonzero]
    host_operator = operator.get() if hasattr(operator, 'get') else operator
    b_mass = _PencilMass.of_mass(mass_host)
    region, min_region_fraction = region_filter(in_region, min_region_fraction, n_unknowns)
    resonance_filter = _ResonanceFilter(host_operator, b_mass, _energy_weights(energy_weights, n_unknowns),
                                        np.asarray(in_window, dtype=bool).ravel(), region,
                                        None if symmetrizer is None else np.asarray(symmetrizer).ravel(),
                                        physics.xp is not np, quotients=symmetrizer is not None)
    candidates, dropped, rayleigh_change = _filtered_candidates(
        theta, vectors, 'shift_invert', k_target_squared, resonance_filter, system.omega, symmetrizer is not None,
        min_region_fraction)
    chosen = candidates[:int(n_resonances)]
    resonances = [_resonance(omega, pml_fraction, field_of(vector) if field_of is not None else vector,
                             region_fraction)
                  for _, omega, pml_fraction, region_fraction, vector in chosen]
    residuals = [_relative_residual(host_operator, mass_host, k_target_squared - (omega / C0) ** 2, vector)
                 for _, omega, _, _, vector in chosen]
    solver.last_residual = float(max(residuals, default=0.0))
    solver.n_solves += 1
    target_wavelength = 2.0 * math.pi * C0 / system.omega
    warnings = []
    if len(resonances) < n_resonances:
        warnings.append(f'{len(resonances)} of {n_resonances} resonances found within +-{solver.bandwidth:.3g} of the '
                        f'target frequency ({dropped["pml"]} PML resonances dropped): raise the time filter bandwidth '
                        'or the sampled frequencies, or set the wavelength nearer the resonance')
    poor = [index + 1 for index, value in enumerate(residuals) if value > _residual_warning()]
    if poor:
        warnings.append(f'resonance(s) {", ".join(map(str, poor))} reached relative residual(s) up to '
                        f'{max(residuals):.2g} only (above {_residual_warning():g}): raise polish_steps, periods or '
                        'sampled_frequencies, or use arpack_shift_invert for machine-precision eigenvalues')
    info = {'n_unknowns': n_unknowns, 'n_requested': int(n_resonances), 'target_wavelength': target_wavelength,
            'factor_host_bytes': 0, 'factor_seconds': 0.0,
            'factor_text': (f"time filter (no factorisation): {run['steps']} FDTD steps ({solver.precision}) in "
                            f"{run['step_seconds']:.3g} s, {run['sampled']} sampled frequencies over +-"
                            f"{solver.bandwidth:.3g} of the target, {pairs['columns']} basis columns after "
                            f"{solver.polish_steps} polish steps, worst relative residual {solver.last_residual:.2g}"),
            'eigen_seconds': time.perf_counter() - started, 'seconds': time.perf_counter() - started,
            'operator_applications': int(pairs['applications']), 'eigenvalues_requested': wanted, 'widenings': 0,
            'eigenvalues_converged': int(mu.size), 'pml_resonances_dropped': dropped['pml'], 'warnings': warnings,
            'nested_dissection': False, 'region': region is not None,
            'region_resonances_dropped': dropped['region'], 'min_region_fraction': min_region_fraction,
            'basis_kept': False, 'time_steps': run['steps'], 'sampled_frequencies': run['sampled'],
            'step_seconds': run['step_seconds'], 'residuals': residuals}
    if rayleigh_change is not None:
        info['rayleigh_change'] = rayleigh_change
    return EigenSolve(resonances=resonances, info=info)


def _residual_warning() -> float:
    from ...solvers.time_filter import RESIDUAL_WARNING
    return RESIDUAL_WARNING


def _relative_residual(host_operator, mass, mu, vector) -> float:
    """||A u - mu M u|| / (||A u|| + |mu| ||M u||) of one resonance (host float64 products)."""
    vector = np.asarray(vector, dtype=np.complex128).ravel()
    applied = host_operator @ vector
    massed = mass * vector
    scale = float(np.linalg.norm(applied) + abs(mu) * np.linalg.norm(massed))
    return float(np.linalg.norm(applied - mu * massed) / scale) if scale > 0.0 else 0.0

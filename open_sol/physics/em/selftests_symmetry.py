"""Model-run self-tests of the symmetry tools of the 2D EM types: mirror unfolding and totals (half vs full domain,
symmetry 2.5), TFSF boxes opening onto a mirror plane (FDFD and FDTD), parity sectors (scripted, inside one study,
with a parameter sweep, time-dependent FDTD, driven and eigenfrequency, per-angle far fields, 2.6) and the
reciprocity / unitarity of a two-port S matrix (symmetry 2.10); no physics imports."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, max_relative_error, worst_result
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, StudySpec, SweepSpec)
from .selftests_fdfd import cpu_specs

_PHYSICS_NAME = 'em'
_PHYSICS_TYPES = {'ez': 'em_fdfd_ez', 'hz': 'em_fdfd_hz'}

# em_fdfd_mirror_unfold: uniform 25 nm mesh, lambda 1 um, all-CPML full domain 2.0 x 2.5 um (window cell k spanning
# [25 k, 25 (k + 1)] nm) symmetric about the cell edge x = 1.0 um (edge 40), against its half 1.0 x 2.5 um with an x_lo
# mirror on that plane (one ghost cell padded outside x = 0). A lossy block (eps_r 4, sigma_e 5e3 S/m) x in [0.7, 1.3]
# um (half: [0, 0.3]), y in [1.0, 1.6] um; sources in row 12 (y = 0.3125 um): even runs a line source along the whole
# width (through the x PML), odd runs two point sources at x = 1 -+ 0.2125 um (cells 31 / 48) with phases 0 / 180 deg
# (half: one at x = 0.2125 um, cell 8). Flux lines in row 88 (y = 2.2125 um): 'transmitted' across the whole width
# (reaches the mirror face: x 2), 'partial' over cells [60, 76) (half [20, 36): x 1). Far-field box columns 23 / 56
# (centres 1 -+ 0.4125 um; half: columns 0 / 16, touching the plane) and rows 20 / 80. The half model is the exact
# even / odd restriction of the full discrete problem, so every compared value agrees to the solver round-off and the
# float32 field storage. Unknowns 110 x 130 (full) and 56 x 130 (half) per run.
_MIRROR_CELL = 25e-9                  # m
_MIRROR_WAVELENGTH = 1e-6             # m
_MIRROR_PLANE = 1.0e-6                # m, x of the full model's symmetry plane
_MIRROR_HEIGHT = 2.5e-6               # m
_MIRROR_BLOCK = (0.3e-6, 1.0e-6, 1.6e-6)     # m, half width, y0, y1
_MIRROR_INNER = (0.1e-6, 0.3e-6)             # m, x range of 'inner' (half model; full: plane + it)
_MIRROR_SOURCE_Y = 0.3125e-6          # m
_MIRROR_POINT_OFFSET = 0.2125e-6      # m, point sources at plane -+ it
_MIRROR_FLUX_Y = 2.2125e-6            # m
_MIRROR_PARTIAL = (0.5e-6, 0.9e-6)    # m, half-model x span of 'partial' (full: plane + it)
_MIRROR_BOX = (0.4125e-6, 0.3875e-6, 0.5125e-6, 1.9875e-6)   # m: full box x in plane - [0], plane + [1]; y0, y1
_MIRROR_N_ANGLES = 180
_MIRROR_TOLERANCE = 1e-5              # relative to each quantity's full-domain magnitude
# run pair -> (polarisation, mirror face type, source: 'line' (even field) or 'pair' (odd field))
_MIRROR_RUNS = {'ez_even': ('ez', 'symmetric', 'line'), 'ez_odd': ('ez', 'antisymmetric', 'pair'),
                'hz_even': ('hz', 'antisymmetric', 'line'), 'hz_odd': ('hz', 'symmetric', 'pair')}
_MIRROR_FIELD = {'ez': 'em.ez_re', 'hz': 'em.hz_re'}
_MIRROR_DERIVED = ('P_block', 'P_inner', 'I_mean')
# em_fdfd_parity_sectors, study part: the same layout, the half run with the study setting parity_sectors (both sectors
# in one run, the model's own x_lo symmetric last): combined snapshot 'combined', then the two unfolded sector snapshots
_SECTOR_LABELS = ['x_lo antisymmetric', 'x_lo symmetric']
_SECTOR_SNAPSHOT_LABELS = ['combined', 'sector [x_lo antisymmetric]', 'sector [x_lo symmetric]']
# em_fdfd_parity_sectors_sweep: the same pair with the block permittivity a parameter swept over _SWEEP_EPS (full: a
# plain sweep; half: parity sectors per sweep value)
_SWEEP_PARAMETER = 'block_eps'
_SWEEP_EPS = [4.0, 6.0]
# em_fdtd_parity_sectors_time: the em_fdfd_parity_sectors layout in em_fdtd_ez / _hz (time_dependent, a gaussian
# pulse at 1 um for 20 fs on the 25 nm mesh, the same dt in both runs): the half is the exact restriction step by step,
# so only float32 rounding of the two sector runs against the full one remains (at any duration: 20 fs lets the pulse
# reach the flux line, ~1.9 um or ~6 fs plus the pulse length). The runs are not bit-identical (fused multiply-adds
# round mirror images differently: spectra agree to ~1e-7), and a late field is a residual orders of magnitude below
# the pulse, so the rounding left from the pulse is ~1e-3 of it: fields are compared in the 15 fs snapshot (pulse in
# the window; output_interval snapshots at 15 fs and the final 20 fs). em_tfsf_mirror's FDTD runs share the duration.
_FDTD_TYPES = {'ez': 'em_fdtd_ez', 'hz': 'em_fdtd_hz'}
_FDTD_PULSE = {'waveform': 'gaussian', 'wavelength': 1e-6, 'bandwidth': 0.5}
_FDTD_SPECTRUM = {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 5}
_FDTD_DURATION = 20e-15               # s
_FDTD_OUTPUT_INTERVAL = 15e-15        # s
_FDTD_FIELD_TIME = 15e-15             # s, compared snapshot
_FDTD_FIELD = {'ez': 'em.ez', 'hz': 'em.hz'}
_FDTD_TOLERANCE = 1e-4                # relative to each quantity's full-domain magnitude (float32 stepping)

# em_tfsf_mirror: +x plane wave (TFSF box) on a lossy block, a 2.5 x 2.0 um all-CPML full model symmetric about the
# cell edge y = 1.0 um (uniform 25 nm mesh, lambda 1 um) against its half 2.5 x 1.0 um with a y_lo mirror of the
# plane wave's parity (ez symmetric, hz antisymmetric) whose TFSF box starts on the plane (opens onto it). Box x cells
# [16, 84), full y rows [12, 68) (half [0, 28)), block x [1.0, 1.6) um, |y - 1 um| < 0.2 um (eps_r 4, sigma_e 5e3 S/m);
# a vertical flux line at x = 2.3125 um (column 92, scattered field) across the whole height (half: mirror_factor 2).
_TFSF_SIZE_X = 2.5e-6                 # m
_TFSF_PLANE = 1.0e-6                  # m, y of the full model's symmetry plane
_TFSF_BOX = (0.4125e-6, 2.0875e-6, 0.6875e-6)    # m: x0, x1, half-height about the plane
_TFSF_BLOCK = (1.0e-6, 1.6e-6, 0.2e-6)           # m: x0, x1, half-height about the plane
_TFSF_LINE_X = 2.3125e-6              # m
_TFSF_MIRRORS = {'ez': 'symmetric', 'hz': 'antisymmetric'}
# run -> (physics type, polarisation, compared fields)
_TFSF_RUNS = {'fdfd_ez': ('em_fdfd_ez', 'ez', ('em.ez_re', 'em.ez_im')),
              'fdfd_hz': ('em_fdfd_hz', 'hz', ('em.hz_re', 'em.hz_im')),
              'fdtd_ez': ('em_fdtd_ez', 'ez', ('em.ez',)),
              'fdtd_hz': ('em_fdtd_hz', 'hz', ('em.hz',))}

# em_fdfd_parity_sectors_eigen: em_fdfd_ez PEC cavity (vacuum, uniform 20 nm cells, no PML) halved by an x_lo mirror:
# window 51 x 61 cells, pec on x_hi / y_lo / y_hi, so the full cavity is 102 x 61 cells with its walls at the centres of
# the outermost cells, L = 101 h x 60 h. The full discrete spectrum k^2 = K_x(m)^2 + K_y(n)^2, K(m) = (2 / h)
# sin(m pi h / (2 L)) (m, n >= 1) splits by the parity of sin(m pi s / L) about the plane: x_lo symmetric (E_z even) =
# odd m, antisymmetric = even m. Target 1.55 um: symmetric (1, 1), (3, 1), (1, 2); antisymmetric (2, 1), (2, 2), (4, 1).
_EIGEN_CELL = 20e-9                   # m
_EIGEN_CELLS = (51, 61)               # half window
_EIGEN_TARGET = 1.55e-6               # m
_EIGEN_N = 3                          # resonances per sector
_EIGEN_MAX_ORDER = 8
_EIGEN_OMEGA_COLUMNS = (2, 3)         # frequency, frequency_imag [Hz] in the resonance_table rows
_EIGEN_OMEGA_TOLERANCE = 1e-8         # relative complex omega (exact discrete spectrum)
_EIGEN_FIELD_TOLERANCE = 1e-5         # max |E_z - closed form| of each sector's resonance 1 (peak 1, float32)

# em_fdfd_reciprocity: single-mode parallel-plate guide 5 x 0.7 um (ez: pec, hz: pmc on y; CPML along x), uniform 25 nm
# mesh: the out-of-plane field vanishes at the centres of rows 0 and 27, width 0.675 um, so TE1 / TM1 propagates and
# the second mode (cutoff 0.675 um) decays by >= exp(-7) between the scatterer and the ports at 0.95-1.05 um. An
# asymmetric scatterer (eps_r 4 at x [2.2, 2.7] um, y [0, 0.3] um plus eps_r 2.5 at x [2.7, 2.9], y [0.3, 0.55] um;
# the 'lossy' runs add sigma_e 2e3 S/m to the first block). Two FDFD physics share the materials: 'forward' launches
# mode 1 toward +x from x = 0.5125 um, 'backward' toward -x from x = 4.5125 um; both decompose the field at the ports
# A (x = 1.0125 um) and B (x = 4.0125 um). Per wavelength out = S in with in = (a_A+, a_B-), out = (a_A-, a_B+) of the
# two runs gives S = OUT IN^-1 (CPML reflections and source leakage enter IN, so they cancel). Unknowns 230 x 28 per
# physics.
_RECIPROCITY_SIZE = (5.0e-6, 0.7e-6)  # m
_RECIPROCITY_CELL = 25e-9             # m
_RECIPROCITY_LINES = (('source_forward', 0.5125e-6), ('port_a', 1.0125e-6), ('port_b', 4.0125e-6),
                      ('source_backward', 4.5125e-6))   # (line, x [m])
_RECIPROCITY_WALLS = {'ez': 'pec', 'hz': 'pmc'}
# reciprocity and unitarity hold per wavelength: two keep the multi-wavelength monitor path
_RECIPROCITY_STUDY = {'wavelength_start': 0.95e-6, 'wavelength_stop': 1.05e-6, 'n_wavelengths': 2}
_RECIPROCITY_LOSS = 2e3               # S/m, 'lossy' runs
_RECIPROCITY_TOLERANCE = 1e-3         # relative |S_AB - S_BA| and |S^H S - I| (lossless)
_RECIPROCITY_MIN_LOSS = 1e-2          # the lossy runs must absorb more than this fraction


# ---- mirror specs ----
def _rectangle(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _horizontal_line(name, x0, x1, y):
    return GeometrySpec(name, 'line', {'x0': x0, 'y0': y, 'x1': x1, 'y1': y})


def _point_source_values(polarization, phase):
    if polarization == 'ez':
        return {'current': 1e-3, 'phase': phase}
    return {'excitation': 'magnetic_current', 'magnetic_current': 1.0, 'phase': phase}


def _mirror_spec(run, half):
    """The full-domain model of a _MIRROR_RUNS entry, or (half) its x_lo-mirrored half."""
    polarization, mirror, source = _MIRROR_RUNS[run]
    return _mirror_model(polarization, mirror, source, half, run)


def _mirror_model(polarization, mirror, source, half, run):
    """Full or (half) x_lo-mirrored model of the shared layout; source 'line', 'pair' (odd image pair in full) or
    'single' (the point at plane + offset only, no image: no definite parity)."""
    offset = 0.0 if half else _MIRROR_PLANE           # x of the plane in this model
    size_x = _MIRROR_PLANE if half else 2.0 * _MIRROR_PLANE
    half_width, block_y0, block_y1 = _MIRROR_BLOCK
    box_lo, box_hi, box_y0, box_y1 = _MIRROR_BOX
    geometry = [
        _rectangle('block', offset - (0.0 if half else half_width), block_y0, offset + half_width, block_y1),
        _rectangle('inner', offset + _MIRROR_INNER[0], block_y0, offset + _MIRROR_INNER[1], block_y1),
        _rectangle('far_field_box', 0.0 if half else offset - box_lo, box_y0, offset + box_hi, box_y1),
        _horizontal_line('transmitted_line', 0.0, size_x, _MIRROR_FLUX_Y),
        _horizontal_line('partial_line', offset + _MIRROR_PARTIAL[0], offset + _MIRROR_PARTIAL[1], _MIRROR_FLUX_Y)]
    features = [FeatureSpec('far_field', 'far_field', ['far_field_box'], {'n_angles': _MIRROR_N_ANGLES}),
                FeatureSpec('transmitted', 'flux_monitor', ['transmitted_line']),
                FeatureSpec('partial', 'flux_monitor', ['partial_line'])]
    if source == 'line':
        geometry.append(_horizontal_line('source_line', 0.0, size_x, _MIRROR_SOURCE_Y))
        features.append(FeatureSpec('source', 'line_source', ['source_line'], {'amplitude': 1.0, 'phase': 0.0}))
    else:
        points = [('source_plus', offset + _MIRROR_POINT_OFFSET, 0.0)]
        if not half and source == 'pair':   # the odd field's image source: opposite phase (E_z / H_z magnetic current)
            points.append(('source_minus', offset - _MIRROR_POINT_OFFSET, 180.0))
        for name, x, phase in points:
            geometry.append(GeometrySpec(f'{name}_point', 'point', {'x': x, 'y': _MIRROR_SOURCE_Y}))
            features.append(FeatureSpec(name, 'point_source', [f'{name}_point'],
                                        _point_source_values(polarization, phase)))
    settings = {'wavelength': _MIRROR_WAVELENGTH}
    if half:
        settings['bc_x_lo'] = mirror
    variant = f'half, {mirror} x_lo' if half else 'full'
    return ModelSpec(
        title=f'{_PHYSICS_TYPES[polarization]} mirror unfold: {run} ({variant})',
        domain=DomainSpec(size_x=size_x, size_y=_MIRROR_HEIGHT),
        geometry=geometry,
        materials=[MaterialSpec('lossy_glass', properties={'eps_r': 4.0, 'sigma_e': 5e3}, selection=['block'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPES[polarization], settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_MIRROR_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('P_block', f'{_PHYSICS_NAME}_absorbed_power', ['block'], 'integral'),
                        DerivedValueSpec('P_inner', f'{_PHYSICS_NAME}_absorbed_power', ['inner'], 'integral'),
                        DerivedValueSpec('I_mean', f'{_PHYSICS_NAME}_e_intensity', [], 'mean'),
                        DerivedValueSpec('P_per_I', 'P_block/I_mean')])   # a ratio value (parity sectors recompute it)


def _mirror_specs():
    """'<run>_full' and '<run>_half' of every _MIRROR_RUNS entry (cpu)."""
    return cpu_specs({f'{run}_{variant}': _mirror_spec(run, variant == 'half')
                      for run in _MIRROR_RUNS for variant in ('full', 'half')})


# ---- mirror check ----
def _check_mirror(results_by_name):
    """Per run pair: unfolded half field vs the full one, flux (x 2 reaching the plane, x 1 off it), unfolded far
    field, derived integrals (x 2 touching, x 1 not) and the mean; error = the largest relative deviation."""
    errors, details = {}, {}
    for run, (polarization, _, _) in _MIRROR_RUNS.items():
        full, half = results_by_name[f'{run}_full'], results_by_name[f'{run}_half']
        name = _MIRROR_FIELD[polarization]
        errors[f'{run}_field'] = max_relative_error(half.field(name, unfold=True).values, full.field(name).values)
        for feature, factor in (('transmitted', 2.0), ('partial', 1.0)):
            half_result, full_result = half.result(_PHYSICS_NAME, feature), full.result(_PHYSICS_NAME, feature)
            errors[f'{run}_{feature}'] = max_relative_error(half_result['power'], full_result['power'])
            errors[f'{run}_{feature}_mirror_factor'] = abs(float(half_result['mirror_factor']) - factor)
            details[f'{run}_{feature}_power'] = float_list(np.atleast_1d(full_result['power']))
        errors[f'{run}_far_field'] = max_relative_error(half.result(_PHYSICS_NAME, 'far_field')['power'],
                                               full.result(_PHYSICS_NAME, 'far_field')['power'])
        full_derived, half_derived = full.summary()['derived'], half.summary()['derived']
        for value_name in _MIRROR_DERIVED:
            full_value, half_value = float(full_derived[value_name]), float(half_derived[value_name])
            errors[f'{run}_{value_name}'] = max_relative_error(half_value, full_value)
            details[f'{run}_{value_name}'] = [full_value, half_value]
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _MIRROR_TOLERANCE,
            'details': {'worst_item': worst, 'errors': errors, **details}}


# ---- parity sectors (symmetry 2.6) ----
def _parity_specs():
    """Per polarisation '<pol>_full' (one point source off the plane, no image) and '<pol>_<mirror>' for both sectors
    of its half (core.mirror.sector_specs of the x_lo-mirrored half)."""
    from ...core.mirror import sector_specs
    specs = {}
    for polarization in _PHYSICS_TYPES:
        specs[f'{polarization}_full'] = _mirror_model(polarization, None, 'single', False, 'parity sectors')
        half = _mirror_model(polarization, 'symmetric', 'single', True, 'parity sectors')
        for label, sector in sector_specs(half).items():
            specs[f"{polarization}_{label.split()[1]}"] = sector
    return specs


def _check_parity(results_by_name):
    """Per polarisation: combined sector Re / Im phasors vs the full one, combined flux (whole width) and derived
    totals over symmetric regions (block absorption, mean intensity) vs the full ones; error = largest deviation."""
    from ...core.mirror import combine_sector_fields, combine_sector_totals
    errors, details = {}, {}
    for polarization in _PHYSICS_TYPES:
        full = results_by_name[f'{polarization}_full']
        sectors = [results_by_name[f'{polarization}_{mirror}'] for mirror in ('symmetric', 'antisymmetric')]
        for part in ('re', 'im'):
            name = f'{_PHYSICS_NAME}.{polarization}_{part}'
            combined = combine_sector_fields([sector.field(name, unfold=True) for sector in sectors])
            errors[f'{polarization}_{part}'] = max_relative_error(combined.values, full.field(name).values)
        flux = combine_sector_totals([sector.result(_PHYSICS_NAME, 'transmitted')['power'] for sector in sectors])
        errors[f'{polarization}_transmitted'] = max_relative_error(flux, full.result(_PHYSICS_NAME, 'transmitted')['power'])
        for value_name in ('P_block', 'I_mean'):
            combined_value = float(combine_sector_totals([float(sector.summary()['derived'][value_name])
                                                          for sector in sectors]))
            full_value = float(full.summary()['derived'][value_name])
            errors[f'{polarization}_{value_name}'] = max_relative_error(combined_value, full_value)
            details[f'{polarization}_{value_name}'] = [full_value, combined_value]
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _MIRROR_TOLERANCE,
            'details': {'worst_item': worst, 'errors': errors, **details}}


# ---- parity sectors inside one study (symmetry 2.6, study setting parity_sectors) ----
def _parity_merged_specs():
    """_parity_specs (scripted sectors against '<pol>_full') plus per polarisation '<pol>_sectors' (the x_lo-mirrored
    half with parity_sectors on: both sectors in one run, compared with the same full run); cpu."""
    specs = _parity_specs()
    for polarization in _PHYSICS_TYPES:
        half = _mirror_model(polarization, 'symmetric', 'single', True, 'parity sectors study')
        half.study.settings['parity_sectors'] = True
        specs[f'{polarization}_sectors'] = half
    return cpu_specs(specs)


def _check_parity_merged(results_by_name):
    """_check_parity (scripted sectors) and _check_parity_study (study setting) on the shared full runs; error = the
    largest relative deviation of both."""
    errors, details = {}, {}
    for part, check in (('scripted', _check_parity), ('study', _check_parity_study)):
        outcome = check(results_by_name)
        errors.update({f'{part} {item}': error for item, error in outcome['details']['errors'].items()})
        details[part] = {key: value for key, value in outcome['details'].items() if key != 'errors'}
    return worst_result(errors, _MIRROR_TOLERANCE, **details)


def _check_parity_study(results_by_name):
    """Per polarisation: sector order and snapshot labels, the combined snapshot's Re / Im phasors and the combined
    derived totals vs the full run, the per-sector fluxes combined vs the full flux; error = largest deviation."""
    from ...core.mirror import combine_sector_totals
    errors, details = {}, {}
    for polarization in _PHYSICS_TYPES:
        full, sectors = results_by_name[f'{polarization}_full'], results_by_name[f'{polarization}_sectors']
        summary = sectors.summary()
        table = summary.get('parity_sectors') or {}
        layout = (table.get('labels'), summary.get('snapshot_labels'), table.get('combined'))
        details[f'{polarization}_layout'] = list(layout)
        expected_layout = (_SECTOR_LABELS, _SECTOR_SNAPSHOT_LABELS, True)
        errors[f'{polarization}_layout'] = 0.0 if layout == expected_layout else math.inf
        for part in ('re', 'im'):
            name = f'{_PHYSICS_NAME}.{polarization}_{part}'
            errors[f'{polarization}_{part}'] = max_relative_error(sectors.snapshot_field(0, name).values, full.field(name).values)
        sector_fluxes = sectors.sector_result(_PHYSICS_NAME, 'transmitted')
        flux = combine_sector_totals([result['power'] for result in sector_fluxes])
        full_flux = full.result(_PHYSICS_NAME, 'transmitted')['power']
        errors[f'{polarization}_transmitted'] = max_relative_error(flux, full_flux)
        stored = sectors.sector_result(_PHYSICS_NAME, 'transmitted', combined=True)   # the run's combined result
        errors[f'{polarization}_transmitted_stored'] = max_relative_error(stored['power'], full_flux)
        try:   # the 'partial' line misses the plane (mirror_factor 1): no combination rule, nothing stored
            sectors.sector_result(_PHYSICS_NAME, 'partial', combined=True)
            errors[f'{polarization}_partial_not_combined'] = math.inf
        except KeyError:
            errors[f'{polarization}_partial_not_combined'] = 0.0
        for value_name in ('P_block', 'I_mean', 'P_per_I'):   # P_per_I: recomputed from the combined values
            combined_value = float(summary['derived'][value_name])
            full_value = float(full.summary()['derived'][value_name])
            errors[f'{polarization}_{value_name}'] = max_relative_error(combined_value, full_value)
            details[f'{polarization}_{value_name}'] = [full_value, combined_value]
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _MIRROR_TOLERANCE,
            'details': {'worst_item': worst, 'errors': errors, **details}}


def _eigen_sector_spec():
    """The x_lo-mirrored half PEC cavity, eigenfrequency study with parity_sectors and a resonance table."""
    settings = {'bc_x_lo': 'symmetric', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec',
                'wavelength': _EIGEN_TARGET}
    return ModelSpec(
        title='em_fdfd_ez half PEC cavity: parity-sector eigenfrequencies',
        domain=DomainSpec(size_x=_EIGEN_CELLS[0] * _EIGEN_CELL, size_y=_EIGEN_CELLS[1] * _EIGEN_CELL),
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdfd_ez', settings=settings,
                             features=[FeatureSpec('resonances', 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_EIGEN_CELL),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _EIGEN_N, 'parity_sectors': True}))


def _full_cavity_lengths():
    """(L_x, L_y) [m] between the wall planes of the unfolded cavity (2 n_x - 1 and n_y - 1 cells)."""
    return (2 * _EIGEN_CELLS[0] - 1) * _EIGEN_CELL, (_EIGEN_CELLS[1] - 1) * _EIGEN_CELL


def _sector_resonances(sector):
    """[(omega, (m, n))] of the _EIGEN_N discrete cavity resonances of a sector nearest the target (m odd for
    'symmetric', even for 'antisymmetric')."""
    omega_target = 2.0 * math.pi * C0 / _EIGEN_TARGET
    lengths = _full_cavity_lengths()
    first_m = 1 if sector == 'symmetric' else 2
    candidates = []
    for m in range(first_m, _EIGEN_MAX_ORDER + 1, 2):
        for n in range(1, _EIGEN_MAX_ORDER + 1):
            wavenumber = math.sqrt(sum((2.0 / _EIGEN_CELL * math.sin(order * math.pi * _EIGEN_CELL / (2.0 * length)))
                                       ** 2 for order, length in zip((m, n), lengths)))
            candidates.append((C0 * wavenumber, (m, n)))
    candidates.sort(key=lambda candidate: abs(candidate[0] - omega_target))
    return candidates[:_EIGEN_N]


def _check_parity_eigen(results_by_name):
    """Sector order, parity-tagged labels, every sector's resonances vs the exact discrete spectrum of its parity and
    each sector's unfolded resonance-1 field vs the full-cavity closed form; error in tolerance units."""
    results = results_by_name['sectors']
    summary = results.summary()
    labels = list(summary.get('snapshot_labels') or [])
    table = summary.get('parity_sectors') or {}
    errors = {'layout': 0.0 if table.get('labels') == _SECTOR_LABELS and len(labels) == 2 * _EIGEN_N else math.inf}
    details = {'snapshot_labels': labels}
    length_x, length_y = _full_cavity_lengths()
    for index, (sector, sector_table) in enumerate(zip(('antisymmetric', 'symmetric'),
                                                       results.sector_result(_PHYSICS_NAME, 'resonances'))):
        expected = _sector_resonances(sector)
        rows = sector_table['rows'][:_EIGEN_N]
        measured = [2.0 * math.pi * complex(float(row[_EIGEN_OMEGA_COLUMNS[0]]), float(row[_EIGEN_OMEGA_COLUMNS[1]]))
                    for row in rows]
        errors[f'{sector}_omega'] = (math.inf if len(measured) < _EIGEN_N else max(
            abs(value / reference - 1.0) for value, (reference, _) in zip(measured, expected))) / _EIGEN_OMEGA_TOLERANCE
        heads = [label.split(':')[0].strip() for label in labels[index * _EIGEN_N:(index + 1) * _EIGEN_N]]
        wanted = [f'resonance {number} [x_lo {sector}]' for number in range(1, _EIGEN_N + 1)]
        errors[f'{sector}_labels'] = 0.0 if heads == wanted else math.inf
        field = np.asarray(results.snapshot_field(index * _EIGEN_N, f'{_PHYSICS_NAME}.ez_re').values, dtype=np.float64)
        m, n = expected[0][1]
        cells_x, cells_y = round(length_x / _EIGEN_CELL) + 1, _EIGEN_CELLS[1]
        closed_form = (np.sin(m * math.pi * np.arange(cells_x)[:, None] * _EIGEN_CELL / length_x)
                       * np.sin(n * math.pi * np.arange(cells_y)[None, :] * _EIGEN_CELL / length_y))
        closed_form /= np.max(np.abs(closed_form))
        errors[f'{sector}_field'] = (math.inf if field.shape != closed_form.shape else min(
            float(np.max(np.abs(field - closed_form))), float(np.max(np.abs(field + closed_form))))
            / _EIGEN_FIELD_TOLERANCE)
        details[sector] = {'orders': [list(orders) for _, orders in expected],
                           'wavelength_expected': [2.0 * math.pi * C0 / reference for reference, _ in expected],
                           'wavelength_measured': float_list([2.0 * math.pi * C0 / value.real for value in measured])}
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0,
            'details': {'worst_item': worst, 'errors': errors, **details}}


# ---- parity sectors with a parameter sweep ----
def _with_eps_sweep(spec):
    """spec with the block eps_r the parameter _SWEEP_PARAMETER, swept over _SWEEP_EPS."""
    spec.parameters = [ParameterSpec(_SWEEP_PARAMETER, _SWEEP_EPS[0])]
    spec.materials[0].properties['eps_r'] = _SWEEP_PARAMETER
    spec.study.sweep = SweepSpec(_SWEEP_PARAMETER, list(_SWEEP_EPS))
    return spec


def _parity_sweep_specs():
    """Per polarisation '<pol>_full' (the single off-plane source, swept) and '<pol>_sectors' (its x_lo-mirrored half
    with parity_sectors on, swept: both sectors per sweep value)."""
    specs = {}
    for polarization in _PHYSICS_TYPES:
        specs[f'{polarization}_full'] = _with_eps_sweep(
            _mirror_model(polarization, None, 'single', False, 'parity sectors sweep'))
        half = _with_eps_sweep(_mirror_model(polarization, 'symmetric', 'single', True, 'parity sectors sweep'))
        half.study.settings['parity_sectors'] = True
        specs[f'{polarization}_sectors'] = half
    return cpu_specs(specs)


def _check_parity_sweep(results_by_name):
    """Per polarisation: sweep layout (values, per-sector derived of both sectors per value, sector labels), the
    combined derived values of every sweep value vs the full sweep, the last value's combined snapshot (Re / Im) and its
    per-angle combined far field (combine_sector_far_fields) vs the full live ones; error = largest deviation."""
    from ...core.mirror import combine_sector_far_fields
    errors, details = {}, {}
    for polarization in _PHYSICS_TYPES:
        full, sectors = results_by_name[f'{polarization}_full'], results_by_name[f'{polarization}_sectors']
        full_sweep, sector_sweep = full.summary()['sweep'] or {}, sectors.summary()['sweep'] or {}
        layout = ([float(value) for value in sector_sweep.get('values') or []],
                  [len(per_value.get('P_block', [])) for per_value in sector_sweep.get('sector_derived') or []],
                  (sectors.summary().get('parity_sectors') or {}).get('labels'))
        details[f'{polarization}_layout'] = list(layout)
        expected_layout = ([float(value) for value in _SWEEP_EPS], [2] * len(_SWEEP_EPS), _SECTOR_LABELS)
        errors[f'{polarization}_layout'] = 0.0 if layout == expected_layout else math.inf
        for value_name in ('P_block', 'I_mean'):
            errors[f'{polarization}_{value_name}'] = max_relative_error(sector_sweep['derived'][value_name],
                                                               full_sweep['derived'][value_name])
            details[f'{polarization}_{value_name}'] = [float_list(full_sweep['derived'][value_name]),
                                                       float_list(sector_sweep['derived'][value_name])]
        for part in ('re', 'im'):
            name = f'{_PHYSICS_NAME}.{polarization}_{part}'
            errors[f'{polarization}_{part}'] = max_relative_error(sectors.snapshot_field(0, name).values, full.field(name).values)
        combined = combine_sector_far_fields(sectors.sector_result(_PHYSICS_NAME, 'far_field'))
        errors[f'{polarization}_far_field'] = max_relative_error(combined['power'],
                                                        full.result(_PHYSICS_NAME, 'far_field')['power'])
    return worst_result(errors, _MIRROR_TOLERANCE, **details)


# ---- parity sectors of a time-dependent FDTD run ----
def _fdtd_sector_spec(polarization, half):
    """The em_fdfd_parity_sectors layout as em_fdtd_<polarization>: full (one off-plane point source) or its x_lo
    mirrored half with parity_sectors on, time_dependent, flux / far-field spectra at 0.9-1.1 um, no derived values."""
    spec = _mirror_model(polarization, 'symmetric' if half else None, 'single', half, 'parity sectors time')
    physics = spec.physics[0]
    physics.type = _FDTD_TYPES[polarization]
    physics.settings.pop('wavelength', None)
    for feature in physics.features:
        if feature.type == 'point_source':
            feature.params = ({'current': 1e-3} if polarization == 'ez'
                              else {'excitation': 'magnetic_current', 'magnetic_current': 1.0})
            feature.params.update(_FDTD_PULSE)
        else:   # flux monitors, far field
            feature.params = {**feature.params, **_FDTD_SPECTRUM}
    spec.title = spec.title.replace(_PHYSICS_TYPES[polarization], _FDTD_TYPES[polarization])
    spec.derived_values = []
    spec.study = StudySpec(type='time_dependent', settings={'duration': _FDTD_DURATION, 'parity_sectors': half,
                                                            'output_interval': _FDTD_OUTPUT_INTERVAL})
    return spec


def _snapshot_near(results, time, label=None):
    """Index of the snapshot nearest `time` [s] (among those labelled `label` when given), from summary()."""
    summary = results.summary()
    return min((abs(float(snapshot_time) - time), index) for index, (snapshot_time, snapshot_label)
               in enumerate(zip(summary['snapshot_times'], summary['snapshot_labels']))
               if label is None or snapshot_label == label)[1]


def _fdtd_parity_specs():
    """Per polarisation '<pol>_full' and '<pol>_sectors' (both sectors of the half in one time_dependent run)."""
    return {f'{polarization}_{variant}': _fdtd_sector_spec(polarization, variant == 'sectors')
            for polarization in _FDTD_TYPES for variant in ('full', 'sectors')}


def _check_fdtd_parity(results_by_name):
    """Per polarisation: sector layout (combined snapshots first, then both sectors'), the combined 15 fs snapshot of
    the out-of-plane field, the combined flux spectrum (combine_sector_totals) and the per-angle combined far field vs
    the full run; error = largest deviation."""
    from ...core.mirror import combine_sector_far_fields, combine_sector_totals
    errors, details = {}, {}
    for polarization in _FDTD_TYPES:
        full, sectors = results_by_name[f'{polarization}_full'], results_by_name[f'{polarization}_sectors']
        summary = sectors.summary()
        table = summary.get('parity_sectors') or {}
        labels = list(summary.get('snapshot_labels') or [])
        n_times = len(full.summary()['snapshot_times'])
        layout = (table.get('labels'), table.get('combined'), len(labels), labels[:n_times])
        details[f'{polarization}_layout'] = [*layout[:3], labels]
        errors[f'{polarization}_layout'] = (0.0 if layout == (_SECTOR_LABELS, True, 3 * n_times,
                                                              ['combined'] * n_times) else math.inf)
        name = _FDTD_FIELD[polarization]
        full_index, combined_index = _snapshot_near(full, _FDTD_FIELD_TIME), _snapshot_near(sectors, _FDTD_FIELD_TIME,
                                                                                            'combined')
        details[f'{polarization}_field_times'] = [float(full.summary()['snapshot_times'][full_index]),
                                                  float(summary['snapshot_times'][combined_index])]
        errors[f'{polarization}_field'] = max_relative_error(sectors.snapshot_field(combined_index, name).values,
                                                    full.snapshot_field(full_index, name).values)
        flux = combine_sector_totals([result['power'] for result in sectors.sector_result(_PHYSICS_NAME,
                                                                                           'transmitted')])
        full_flux = full.result(_PHYSICS_NAME, 'transmitted')['power']
        errors[f'{polarization}_transmitted'] = max_relative_error(flux, full_flux)
        combined = combine_sector_far_fields(sectors.sector_result(_PHYSICS_NAME, 'far_field'))
        full_far_field = full.result(_PHYSICS_NAME, 'far_field')['power']
        errors[f'{polarization}_far_field'] = max_relative_error(combined['power'], full_far_field)
        for feature, reference in (('transmitted', full_flux), ('far_field', full_far_field)):   # stored with the run
            stored = sectors.sector_result(_PHYSICS_NAME, feature, combined=True)
            errors[f'{polarization}_{feature}_stored'] = max_relative_error(stored['power'], reference)
    return worst_result(errors, _FDTD_TOLERANCE, **details)


# ---- TFSF box opening onto a mirror plane ----
def _tfsf_mirror_spec(run, half):
    """The full em_tfsf_mirror model of a _TFSF_RUNS entry, or (half) its y_lo-mirrored half."""
    physics_type, polarization, _ = _TFSF_RUNS[run]
    plane = 0.0 if half else _TFSF_PLANE
    size_y = _TFSF_PLANE if half else 2.0 * _TFSF_PLANE
    box_x0, box_x1, box_half = _TFSF_BOX
    block_x0, block_x1, block_half = _TFSF_BLOCK
    geometry = [_rectangle('tfsf_box', box_x0, 0.0 if half else plane - box_half, box_x1, plane + box_half),
                _rectangle('block', block_x0, 0.0 if half else plane - block_half, block_x1, plane + block_half),
                GeometrySpec('forward_line', 'line', {'x0': _TFSF_LINE_X, 'y0': 0.0, 'x1': _TFSF_LINE_X,
                                                      'y1': size_y})]
    time_domain = physics_type.startswith('em_fdtd')
    plane_wave = {'amplitude': 1.0, **(_FDTD_PULSE if time_domain else {'phase': 0.0})}
    features = [FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'], plane_wave),
                FeatureSpec('forward', 'flux_monitor', ['forward_line'], dict(_FDTD_SPECTRUM) if time_domain else {})]
    settings = {} if time_domain else {'wavelength': _MIRROR_WAVELENGTH}
    if half:
        settings['bc_y_lo'] = _TFSF_MIRRORS[polarization]
    study = (StudySpec(type='time_dependent', settings={'duration': _FDTD_DURATION}) if time_domain
             else StudySpec(type='stationary'))
    return ModelSpec(
        title=f"{physics_type} TFSF box onto a mirror ({'half, y_lo ' + settings['bc_y_lo'] if half else 'full'})",
        domain=DomainSpec(size_x=_TFSF_SIZE_X, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('lossy_glass', properties={'eps_r': 4.0, 'sigma_e': 5e3}, selection=['block'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_MIRROR_CELL),
        study=study)


def _tfsf_mirror_specs():
    """'<run>_full' and '<run>_half' of every _TFSF_RUNS entry (FDFD runs on the cpu, FDTD on the default backend)."""
    specs = {f'{run}_{variant}': _tfsf_mirror_spec(run, variant == 'half')
             for run in _TFSF_RUNS for variant in ('full', 'half')}
    cpu_specs({name: spec for name, spec in specs.items() if name.startswith('fdfd')})
    return specs


def _check_tfsf_mirror(results_by_name):
    """Per run pair: the unfolded half fields vs the full ones and the forward flux (half: mirror_factor 2); error in
    tolerance units (FDFD _MIRROR_TOLERANCE, FDTD _FDTD_TOLERANCE)."""
    errors, details = {}, {}
    for run, (physics_type, _, fields) in _TFSF_RUNS.items():
        full, half = results_by_name[f'{run}_full'], results_by_name[f'{run}_half']
        tolerance = _FDTD_TOLERANCE if physics_type.startswith('em_fdtd') else _MIRROR_TOLERANCE
        for name in fields:
            errors[f'{run}_{name}'] = max_relative_error(half.field(name, unfold=True).values,
                                                full.field(name).values) / tolerance
        half_flux, full_flux = half.result(_PHYSICS_NAME, 'forward'), full.result(_PHYSICS_NAME, 'forward')
        errors[f'{run}_forward'] = max_relative_error(half_flux['power'], full_flux['power']) / tolerance
        errors[f'{run}_mirror_factor'] = 0.0 if float(half_flux['mirror_factor']) == 2.0 else math.inf
        details[f'{run}_forward_power'] = float_list(np.atleast_1d(full_flux['power']))
    return worst_result(errors, 1.0, **details)


def _reciprocity_spec(polarization, lossy):
    """The two-physics parallel-plate model: 'forward' and 'backward' mode sources, mode monitors at ports A and B."""
    size_x, size_y = _RECIPROCITY_SIZE
    geometry = [GeometrySpec(f'{name}_line', 'line', {'x0': x, 'y0': 0.0, 'x1': x, 'y1': size_y})
                for name, x in _RECIPROCITY_LINES]
    geometry += [_rectangle('block_1', 2.2e-6, 0.0, 2.7e-6, 0.3e-6), _rectangle('block_2', 2.7e-6, 0.3e-6, 2.9e-6,
                                                                              0.55e-6)]
    first_block = {'eps_r': 4.0, 'sigma_e': _RECIPROCITY_LOSS} if lossy else {'eps_r': 4.0}
    wall = _RECIPROCITY_WALLS[polarization]
    settings = {'bc_y_lo': wall, 'bc_y_hi': wall, 'wavelength': 1e-6}
    monitors = [FeatureSpec(port, 'mode_monitor', [f'{port}_line'], {'mode_number': 1})
                for port in ('port_a', 'port_b')]
    physics = [PhysicsSpec(physics_name, _PHYSICS_TYPES[polarization], settings=dict(settings), features=[
        FeatureSpec('source', 'mode_source', [f'source_{physics_name}_line'],
                    {'mode_number': 1, 'direction': direction, 'power': 1.0, 'phase': 0.0}), *monitors])
        for physics_name, direction in (('forward', 'positive'), ('backward', 'negative'))]
    return ModelSpec(
        title=f"{_PHYSICS_TYPES[polarization]} two-port reciprocity ({'lossy' if lossy else 'lossless'})",
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('block_1', properties=first_block, selection=['block_1']),
                   MaterialSpec('block_2', properties={'eps_r': 2.5}, selection=['block_2'])],
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_RECIPROCITY_CELL),
        study=StudySpec(type='frequency_domain', settings=dict(_RECIPROCITY_STUDY)))


def _reciprocity_specs():
    """'<polarization>_lossless' and '<polarization>_lossy' for ez (pec walls) and hz (pmc walls); cpu."""
    return cpu_specs({f'{polarization}_{kind}': _reciprocity_spec(polarization, kind == 'lossy')
                      for polarization in _PHYSICS_TYPES for kind in ('lossless', 'lossy')})


# ---- reciprocity check ----
def _port_amplitudes(results, physics_name, port):
    """(a_positive, a_negative) complex128 per ascending wavelength of a mode monitor, relative to the physics' source
    amplitude (sqrt(fraction) e^{j phase})."""
    result = results.result(physics_name, port)
    if not result.get('normalized'):
        raise ValueError(f"mode monitor '{physics_name}.{port}' has no source-normalised amplitudes")
    series = result['series']
    return tuple(np.sqrt(np.asarray(series[f'fraction_{direction}'], dtype=np.float64))
                 * np.exp(1j * np.asarray(series[f'phase_{direction}'], dtype=np.float64))
                 for direction in ('positive', 'negative'))


def _scattering_matrices(results):
    """(n_wavelengths, 2, 2) S = OUT IN^-1 over the two runs (in = (a_A+, a_B-), out = (a_A-, a_B+); ports A, B)."""
    inputs, outputs = [], []
    for physics_name in ('forward', 'backward'):
        a_plus, a_minus = _port_amplitudes(results, physics_name, 'port_a')
        b_plus, b_minus = _port_amplitudes(results, physics_name, 'port_b')
        inputs.append(np.stack((a_plus, b_minus), axis=-1))
        outputs.append(np.stack((a_minus, b_plus), axis=-1))
    incoming = np.stack(inputs, axis=-1)    # (n_wl, port, run)
    outgoing = np.stack(outputs, axis=-1)
    return outgoing @ np.linalg.inv(incoming)


def _check_reciprocity(results_by_name):
    """Every run: |S_AB - S_BA| / max |S_AB|, |S_BA| (reciprocity); lossless runs also max |S^H S - I| (unitarity),
    lossy runs an absorbed fraction above _RECIPROCITY_MIN_LOSS for both incidences; error in tolerance units."""
    errors, details = {}, {}
    for run, results in results_by_name.items():
        matrices = _scattering_matrices(results)
        s_ab, s_ba = matrices[:, 0, 1], matrices[:, 1, 0]
        errors[f'{run}_reciprocity'] = float(np.max(np.abs(s_ab - s_ba) / np.maximum(np.abs(s_ab), np.abs(s_ba))))
        gram = np.conj(np.swapaxes(matrices, 1, 2)) @ matrices
        deviation = float(np.max(np.abs(gram - np.eye(2))))
        details[f'{run}_S'] = [[float_list(np.abs(matrices[:, row, column])) for column in range(2)]
                               for row in range(2)]
        details[f'{run}_unitarity_deviation'] = deviation
        if run.endswith('lossless'):
            errors[f'{run}_unitarity'] = deviation
        else:
            absorbed = 1.0 - np.sum(np.abs(matrices) ** 2, axis=1)   # per incidence column: 1 - |S_.j|^2
            details[f'{run}_absorbed'] = float_list(absorbed.ravel())
            errors[f'{run}_absorption'] = 0.0 if float(np.min(absorbed)) > _RECIPROCITY_MIN_LOSS else math.inf
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst] / _RECIPROCITY_TOLERANCE,
            'tolerance': 1.0, 'details': {'worst_item': worst, 'errors': errors, **details}}


MIRROR_UNFOLD_CASE = SelfTestCase(
    name='em_fdfd_mirror_unfold',
    label='EM frequency domain: mirror unfolding and totals',
    description='Half vs full domain (symmetry 2.5): a 2.0 x 2.5 um all-CPML em_fdfd model symmetric about x = 1 um '
                '(uniform 25 nm mesh, lambda 1 um, a lossy eps_r 4 block across the plane) against its half with an '
                'x_lo mirror on that plane, in four pairs: ez symmetric and hz antisymmetric with a line source (even '
                'field), ez antisymmetric and hz symmetric with an opposite-phase point-source pair (odd field). The '
                'half model is the exact even / odd restriction of the full discrete problem, so: the out-of-plane '
                'phasor (Re) unfolded by Results.field(unfold=True) equals the full one; a flux line across the whole '
                'width (mirror_factor 2) and one off the plane (mirror_factor 1) equal the full fluxes; the far field '
                'of a box touching the plane (its plane face dropped, the image contour added) equals the full box '
                "pattern; derived integrals of absorbed_power over the block (touching: x 2) and over a part off the "
                'plane (x 1), and the mean of e_intensity, equal the full ones. error = the largest relative '
                f'deviation, tolerance {_MIRROR_TOLERANCE:g} (solver round-off and float32 field storage). cpu '
                'backend (not yet confirmed by a run there).',
    specs=_mirror_specs,
    check=_check_mirror)

RECIPROCITY_CASE = SelfTestCase(
    name='em_fdfd_reciprocity',
    label='EM frequency domain: two-port reciprocity and unitarity',
    description='Symmetry 2.10: a single-mode parallel-plate guide (0.675 um between the walls: ez pec, hz pmc; CPML '
                'along x, 25 nm mesh, cpu backend) with an asymmetric dielectric scatterer, solved at 0.95 and 1.05 '
                'um (frequency_domain; the identities hold per wavelength; not yet confirmed by a run) by two '
                "em_fdfd physics sharing the materials: 'forward' launches mode 1 toward +x, 'backward' toward -x; mode "
                'monitors at the ports A and B on both sides give (a_A+, a_B-) in and (a_A-, a_B+) out per run, so S = '
                'OUT IN^-1 (CPML reflections and source leakage cancel). Lorentz reciprocity: S_AB = S_BA (complex), '
                'in the lossless runs also unitarity S^H S = I, in the lossy runs (sigma_e 2e3 S/m in one block) an '
                f'absorbed fraction above {_RECIPROCITY_MIN_LOSS:g} for both incidences. error = the largest relative '
                f'|S_AB - S_BA| / max |S| or |S^H S - I| entry / {_RECIPROCITY_TOLERANCE:g}, tolerance 1 (budget: the '
                'evanescent second mode at the ports, < 1e-3 in amplitude, and the discrete mode decomposition).',
    specs=_reciprocity_specs,
    check=_check_reciprocity)

PARITY_SECTORS_CASE = SelfTestCase(
    name='em_fdfd_parity_sectors',
    label='EM frequency domain: parity-sector decomposition, scripted and in one study',
    description='Symmetry 2.6 (cpu backend; merged with the former em_fdfd_parity_sectors_study, whose full runs were '
                'the same: not yet confirmed by a run as one case). Scripted part: the em_fdfd_mirror_unfold layout '
                '(lossy eps_r 4 block symmetric about x = 1 um) driven '
                'by one point source 0.2125 um off the plane (no image, so the field has no definite parity), solved '
                'on the full domain and as the two parity sectors of the x_lo-mirrored half (core.mirror.sector_specs: '
                'symmetric and antisymmetric), for ez and hz. The sectors solve the source plus its even / odd image, '
                'so combine_sector_fields (mean of the unfolded fields) gives the full Re and Im phasors, and '
                'combine_sector_totals (mean / 2) the full flux across the whole width, block absorption and mean '
                'e_intensity (regions symmetric about the plane: the cross terms are odd). Study part (study setting '
                'parity_sectors): the same half with parity_sectors on (one stationary run building both sectors, '
                "the model's own x_lo symmetric last), against the same full run, for ez and hz. Expected: sector "
                'order [antisymmetric, symmetric], '
                "snapshots ['combined', 'sector [x_lo antisymmetric]', 'sector [x_lo symmetric]']; the combined "
                'snapshot (mean of the unfolded sector fields) equals the full Re and Im phasors; the combined derived '
                'values (sector mean / 2) equal the full block absorption and mean e_intensity, and the ratio value '
                'P_block / I_mean (recomputed from the combined values, not combined) the full ratio; '
                'combine_sector_totals of Results.sector_result fluxes and the stored combined result '
                '(sector_result(..., combined=True)) equal the full flux across the whole width; the partial line '
                '(off the plane) has no stored combination. '
                f'error = the largest relative deviation of both parts, tolerance {_MIRROR_TOLERANCE:g} (exact '
                'restriction: round-off and float32 fields).',
    specs=_parity_merged_specs,
    check=_check_parity_merged)

PARITY_SECTORS_EIGEN_CASE = SelfTestCase(
    name='em_fdfd_parity_sectors_eigen',
    label='EM eigenfrequency: parity-labelled sector resonances',
    description='Symmetry 2.6: an em_fdfd_ez vacuum PEC cavity halved by an x_lo mirror (window 51 x 61 cells of 20 nm, '
                'pec x_hi / y_lo / y_hi; unfolded 102 x 61 cells, walls at the outermost cell centres, L = 101 h x '
                '60 h), eigenfrequency study near 1.55 um with n_resonances 3 and parity_sectors on. The full discrete '
                'spectrum k^2 = K_x(m)^2 + K_y(n)^2, K(m) = (2 / h) sin(m pi h / (2 L)), splits by the parity about '
                'the plane: x_lo symmetric = odd m, antisymmetric = even m. Expected: sectors [antisymmetric, '
                "symmetric], six snapshots labelled 'resonance k [x_lo <sector>]: ...' sector by sector, each "
                "sector's resonance_table (Results.sector_result) = the three resonances of its parity nearest the "
                f'target (relative complex omega {_EIGEN_OMEGA_TOLERANCE:g}), and each sector\'s first snapshot, '
                'unfolded over the mirror, = sin(m pi s / L_x) sin(n pi j / 60) of the full cavity up to the sign '
                f'(peak 1, {_EIGEN_FIELD_TOLERANCE:g}). error = the worst deviation in tolerance units, tolerance 1. '
                'cpu backend (host ARPACK on both; not yet confirmed by a run there).',
    specs=lambda: cpu_specs({'sectors': _eigen_sector_spec()}),
    check=_check_parity_eigen)

PARITY_SECTORS_SWEEP_CASE = SelfTestCase(
    name='em_fdfd_parity_sectors_sweep',
    label='EM frequency domain: parity sectors with a parameter sweep',
    description='Symmetry 2.6 (cpu backend; not yet confirmed by a run there): the study pair of '
                'em_fdfd_parity_sectors (one point source 0.2125 um off the plane of the lossy block symmetric about '
                'x = 1 um) with the block eps_r a parameter swept over '
                f'{_SWEEP_EPS} (lossy, sigma_e 5e3 S/m): the full model as a plain sweep, the x_lo-mirrored half with '
                'parity_sectors on (both sectors per sweep value), for ez and hz. Expected: sweep values and two '
                'per-sector derived values per value (summary sweep sector_derived), sector labels [antisymmetric, '
                'symmetric]; per sweep value the combined block absorption and mean e_intensity equal the full ones; '
                "the last value's combined snapshot (Re / Im phasors) and its far field combined per angle by "
                'core.mirror.combine_sector_far_fields of the sector results (mean of the complex amplitudes) equal '
                f'the full live ones. error = the largest relative deviation, tolerance {_MIRROR_TOLERANCE:g} (exact '
                'restriction: round-off and float32 fields).',
    specs=_parity_sweep_specs,
    check=_check_parity_sweep)

FDTD_PARITY_SECTORS_CASE = SelfTestCase(
    name='em_fdtd_parity_sectors_time',
    label='EM time domain: parity sectors of a time-dependent run',
    description='Symmetry 2.6, study setting parity_sectors in a time_dependent study: the em_fdfd_parity_sectors '
                'layout (lossy block symmetric about x = 1 um, uniform 25 nm mesh, all CPML) in em_fdtd_ez / _hz with '
                'one point source 0.2125 um off the plane (gaussian pulse at 1 um, bandwidth 0.5, 20 fs: the '
                'identity holds at any duration, the pulse reaches the flux line by then; not yet confirmed by a run '
                'at 20 fs), the full '
                'model against the x_lo-mirrored half with parity_sectors on (both sectors in one run, the same time '
                "step, snapshots every 15 fs). Expected: sectors [antisymmetric, symmetric], combined, the combined "
                'snapshots first then both sectors\'; the combined 15 fs snapshot of E_z / H_z equals the full one '
                '(a late field is a residual far below the pulse, where the runs\' float32 round-off is not '
                'negligible), the combined flux spectrum across the whole width '
                '(combine_sector_totals of the sector results) and the far field combined per angle '
                '(combine_sector_far_fields) equal the full ones, likewise both results stored combined with the run '
                '(sector_result(..., combined=True)). The half is the exact restriction step by step, so '
                f'error = the largest relative deviation, tolerance {_FDTD_TOLERANCE:g} (float32 stepping of three '
                'runs).',
    specs=_fdtd_parity_specs,
    check=_check_fdtd_parity)

TFSF_MIRROR_CASE = SelfTestCase(
    name='em_tfsf_mirror',
    label='EM: plane-wave box opening onto a mirror plane',
    description='Symmetry 2.5: a +x plane wave (TFSF box) on a lossy block (eps_r 4, sigma_e 5e3 S/m) in a 2.5 x 2.0 '
                'um all-CPML model symmetric about y = 1 um (uniform 25 nm mesh, lambda 1 um), against its half with a '
                'y_lo mirror of the wave\'s parity (ez symmetric, hz antisymmetric) whose TFSF box starts on the plane, '
                'so the box opens there and its image completes it; em_fdfd_ez / _hz (stationary, cpu backend) and '
                'em_fdtd_ez / _hz (time_dependent, gaussian pulse at 1 um for 20 fs: the scattered wave reaches the '
                'flux line after ~5-8 fs; not yet confirmed by a run with these settings). Expected: the unfolded half fields (FDFD Re / Im '
                'phasors, FDTD final E_z / H_z) equal the full ones and the forward flux of a line across the whole '
                'height (mirror_factor 2) equals the full one. error = the largest relative deviation in tolerance '
                f'units (FDFD {_MIRROR_TOLERANCE:g}: exact restriction; FDTD {_FDTD_TOLERANCE:g}: float32 stepping), '
                'tolerance 1.',
    specs=_tfsf_mirror_specs,
    check=_check_tfsf_mirror)

SYMMETRY_SELFTEST_CASES = (MIRROR_UNFOLD_CASE, RECIPROCITY_CASE, PARITY_SECTORS_CASE, PARITY_SECTORS_EIGEN_CASE,
                           PARITY_SECTORS_SWEEP_CASE, FDTD_PARITY_SECTORS_CASE, TFSF_MIRROR_CASE)

"""Self-tests of the em_fdtd_3d Bloch phase (symmetry 2.8): the empty-lattice bands of a fully periodic vacuum cell at a
Bloch k against the exact discrete Yee dispersion, the monitor path of the imaginary field set with the k sign (a
phased dipole array radiating through CPML z faces at +-k_x), the port modes at the Bloch k (CPU), and the
'photonic_crystal_slab_bands_3d' template."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                           PhysicsSpec, StudySpec, SweepSpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_3d'          # literal (selftests never import the definition)
_RESONANCE_FEATURE = 'resonances'
_CELL = 50e-9                         # m
_CELLS = (20, 10, 10)                 # periods 1.0 x 0.5 x 0.5 um, every face periodic
_PERIODS = tuple(cells * _CELL for cells in _CELLS)
_G = tuple(2.0 * math.pi / period for period in _PERIODS)   # reciprocal lattice vectors [rad/m]
# run -> Bloch k [rad/m]: 0.25 g_x, 0.1 g_y and 0.1 g_z, so all three wrap axes carry sin(kL) != 0 (also the edges /
# corners where two phased wraps meet)
_RUNS = {'kxyz': (0.25 * _G[0], 0.1 * _G[1], 0.1 * _G[2])}
# Bands in the 1.0-5.0 um window: K = k + G with G = 0 (~2.6 um) and G = -g_x (~1.25 um); the next (+g_x, ~0.78 um;
# -g_y / -g_z, ~0.55 um) lie outside
_ORDERS = ((0, 0, 0), (-1, 0, 0))
_BAND = (1.0e-6, 5.0e-6)              # m
_SOURCE = (0.27e-6, 0.13e-6, 0.21e-6)   # m, z dipole off every symmetry plane
_PROBE = (0.71e-6, 0.36e-6, 0.33e-6)
_PULSE = {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': 2.0e-6, 'bandwidth': 1.0,
          'current_moment': 1e-12}
# s: ~34 periods of the longest band, ~300 decimated samples: a lossless noise-free ring-down resolves to ~1e-6
_DURATION = 0.3e-12
_TOLERANCE = 1e-3

# em3d_bloch_array_monitors: period 11 x 10 cells of 50 nm (0.55 x 0.5 um, x / y periodic), height 3 um with CPML z
# faces (15 cells each). A z dipole at the centre of window cell (0, 3), z = 0.8 um, phased by the Bloch k into an
# infinite array; at 1.55 um (k0 = 4.05 / um) only the zero order K_t = (k_x, k_y) propagates (the first diffracted
# orders, |K_t| >= 9.7 / um, decay by >= exp(-12) up to the field layer at z = 2.2 um). Runs 'plus' (k_x, k_y) and
# 'minus' (-k_x, k_y): the mirror x -> 2 x_dipole - x maps the Yee lattice, the CPML and the dipole onto themselves and
# k_x to -k_x, so S_x(minus) = -S_x(plus), S_y and S_z equal; the x flux plane sits at the edge opposite the dipole
# (0.30 um, mapped onto itself modulo the period).
_ARRAY_CELLS = (11, 10)
_ARRAY_PERIODS = tuple(cells * _CELL for cells in _ARRAY_CELLS)
_ARRAY_HEIGHT = 3.0e-6                # m
_ARRAY_K = (0.15 * 2.0 * math.pi / _ARRAY_PERIODS[0], 0.1 * 2.0 * math.pi / _ARRAY_PERIODS[1])   # rad/m
_ARRAY_RUNS = {'plus': 1.0, 'minus': -1.0}   # sign of k_x
_ARRAY_SOURCE = (0.025e-6, 0.175e-6, 0.8e-6)   # m: window cell (0, 3) centre, z on a face
_ARRAY_WAVELENGTH = 1.55e-6           # m
_ARRAY_PULSE = {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': _ARRAY_WAVELENGTH, 'bandwidth': 0.5,
                'current_moment': 1e-12}
_ARRAY_PLANES = {'flux_x': ('x', 0.30e-6), 'flux_y': ('y', 0.25e-6), 'flux_z': ('z', 2.0e-6)}   # (normal, position)
_ARRAY_LAYER_Z = 2.2e-6               # m
# s: pulse 2 t0 ~30 fs + the oblique transit to the layer; the mirror relations hold for any truncation, the layer
# uniformity for any truncated DFT at fixed k (one zero order), the signs are set by f0
_ARRAY_DURATION = 90e-15
_ARRAY_MIN_TRANSVERSE = 1e-2          # |S_x|, S_y / S_z above this: the imaginary set's flux is present


def _lattice_spec(bloch_k):
    """Vacuum unit cell periodic on all six faces with Bloch k, a z dipole pulse and a resonance monitor (E_z)."""
    settings = {f'bc_{axis}_{side}': 'periodic' for axis in 'xyz' for side in ('lo', 'hi')}
    settings.update({f'bloch_k{axis}': float(k) for axis, k in zip('xyz', bloch_k)})
    return ModelSpec(
        title='Self-test: 3D Bloch empty lattice',
        domain=DomainSpec(size_x=_PERIODS[0], size_y=_PERIODS[1], size_z=_PERIODS[2]),
        geometry=[GeometrySpec('source_point', 'point', dict(zip('xyz', _SOURCE))),
                  GeometrySpec('probe_point', 'point', dict(zip('xyz', _PROBE)))],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings, features=[
            FeatureSpec('dipole', 'dipole_source', ['source_point'], dict(_PULSE)),
            FeatureSpec(_RESONANCE_FEATURE, 'resonance_monitor', ['probe_point'],
                        {'component': 'ez', 'wavelength_min': _BAND[0], 'wavelength_max': _BAND[1]})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))


def _discrete_band_frequency(bloch_k, order, dt):
    """Frequency [Hz] of the plane wave K = k + G(order) from the lossless Yee dispersion: sin(omega dt / 2) = (C0 dt /
    2) kappa, kappa^2 = sum (2 / d)^2 sin^2(K_i d / 2)."""
    kappa_squared = sum((2.0 / _CELL) ** 2 * math.sin(0.5 * (k + m * g) * _CELL) ** 2
                        for k, m, g in zip(bloch_k, order, _G))
    return 2.0 * math.asin(0.5 * C0 * dt * math.sqrt(kappa_squared)) / (2.0 * math.pi * dt)


def _check_lattice(results_by_name):
    """Per run and band: frequency vs the discrete dispersion and 1/Q (~0, lossless); worst of both."""
    errors, per_band = {}, {}
    for run, bloch_k in _RUNS.items():
        table = results_by_name[run].result(_PHYSICS_NAME, _RESONANCE_FEATURE)
        columns = list(table['columns'])
        rows = [list(row) for row in table['rows']]
        frequencies = np.asarray([row[columns.index('frequency')] for row in rows], dtype=np.float64)
        for order in _ORDERS:
            label = f'{run}_G{order[0]}{order[1]}{order[2]}'
            expected = _discrete_band_frequency(bloch_k, order, float(table['dt']))
            if frequencies.size == 0:
                errors[label], per_band[label] = math.inf, {'expected_frequency': expected, 'note': table.get('note')}
                continue
            row = rows[int(np.argmin(np.abs(frequencies - expected)))]
            measured = row[columns.index('frequency')]
            inverse_q = 2.0 * abs(row[columns.index('frequency_imag')]) / measured
            errors[label] = max(abs(measured / expected - 1.0), inverse_q)
            per_band[label] = {'frequency': measured, 'expected_frequency': expected, 'inverse_Q': inverse_q,
                               'n_rows': len(rows)}
    worst = max(errors, key=lambda item: errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _TOLERANCE,
            'details': {'worst_band': worst, 'bands': per_band}}


def _array_spec(k_sign):
    """x / y periodic column with Bloch (k_sign k_x, k_y), CPML z, a z dipole pulse, three flux planes and a field
    layer at the dipole wavelength."""
    settings = {f'bc_{axis}_{side}': 'periodic' for axis in 'xy' for side in ('lo', 'hi')}
    settings.update({'bc_z_lo': 'cpml', 'bc_z_hi': 'cpml', 'bloch_kx': k_sign * _ARRAY_K[0],
                     'bloch_ky': _ARRAY_K[1], 'bloch_kz': 0.0})
    spectrum = {'wavelength_start': _ARRAY_WAVELENGTH, 'wavelength_stop': _ARRAY_WAVELENGTH, 'n_wavelengths': 1}
    features = [FeatureSpec('dipole', 'dipole_source', ['source_point'], dict(_ARRAY_PULSE)),
                FeatureSpec('layer', 'dft_plane', [], {'position': _ARRAY_LAYER_Z, **spectrum})]
    features += [FeatureSpec(name, 'flux_plane', [], {'normal': normal, 'position': position, **spectrum})
                 for name, (normal, position) in _ARRAY_PLANES.items()]
    return ModelSpec(
        title=f"Self-test: 3D Bloch dipole array ({'+' if k_sign > 0 else '-'}k_x)",
        domain=DomainSpec(size_x=_ARRAY_PERIODS[0], size_y=_ARRAY_PERIODS[1], size_z=_ARRAY_HEIGHT),
        geometry=[GeometrySpec('source_point', 'point', dict(zip('xyz', _ARRAY_SOURCE)))],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _ARRAY_DURATION}))


def _check_array(results_by_name):
    """Mirror relations of the three fluxes between the runs (exact), the field-layer intensity uniformity per run
    (one propagating Bloch order: |E| uniform) and the flow direction (S_x, S_y along +k, S_z > 0)."""
    flux = {run: {name: float(np.atleast_1d(results_by_name[run].result(_PHYSICS_NAME, name)['power'])[0])
                  for name in _ARRAY_PLANES} for run in _ARRAY_RUNS}
    plus, minus = flux['plus'], flux['minus']
    errors = {'flux_x_antisymmetry': abs(plus['flux_x'] + minus['flux_x']) / abs(plus['flux_x']),
              'flux_y_symmetry': abs(plus['flux_y'] - minus['flux_y']) / abs(plus['flux_y']),
              'flux_z_symmetry': abs(plus['flux_z'] - minus['flux_z']) / abs(plus['flux_z'])}
    details = {'flux': flux}
    for run in _ARRAY_RUNS:
        intensity = np.asarray(results_by_name[run].result(_PHYSICS_NAME, 'layer', freq_index=0)['intensity'],
                               dtype=np.float64)
        errors[f'{run}_layer_uniformity'] = float((intensity.max() - intensity.min()) / intensity.mean())
        details[f'{run}_layer_mean_intensity'] = float(intensity.mean())
    along_k = (plus['flux_z'] > 0.0 and plus['flux_x'] > _ARRAY_MIN_TRANSVERSE * plus['flux_z']
               and plus['flux_y'] > _ARRAY_MIN_TRANSVERSE * plus['flux_z']
               and minus['flux_x'] < -_ARRAY_MIN_TRANSVERSE * minus['flux_z'])
    errors['direction'] = 0.0 if along_k else math.inf
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _TOLERANCE,
            'details': {'worst_item': worst, 'errors': errors, **details}}


# em3d_bloch_port_modes (CPU, no run): port plane normal to z of a column x periodic (4 cells of 50 nm) at Bloch k_x,
# y pec (port natural ends), a slab n = 2.0 (0.8 um) in n = 1.444 along y on 20 nm cells, 1.55 um. Uniform in x, the
# semi-discrete operator (x Yee-discrete, z continuous) is invariant under rotation in the x-z plane once d/dx ->
# -j K_x, K_x = (2/dx) sin(k_x dx / 2): every k = 0 mode beta_0 has a Bloch partner beta^2 = beta_0^2 - K_x^2, with
# fields F(x + dx) = F(x) e^{-j k_x dx}.
_PORT_CELLS_X, _PORT_CELL_X = 4, 50e-9
_PORT_CELL_Y = 20e-9
_PORT_CLADDING, _PORT_CORE = 1.5e-6, 0.8e-6   # m: cladding per side, core thickness
_PORT_INDICES = (2.0, 1.444)                  # core, cladding
_PORT_WAVELENGTH = 1.55e-6
_PORT_K_FRACTION = 0.4                        # k_x = 0.4 k0
_PORT_MODES = 4                               # TE0, TM0, TE1, TM1 (V = 2.24)
_PORT_TOLERANCE = 1e-8                        # |beta^2 - (beta_0^2 - K_x^2)| / k0^2 and the quasi-periodicity


def _port_solve(bloch_kx):
    """CrossSectionSolve of the slab port (normal z) at Bloch k_x through mode_ports3d.port_boundary, and that boundary."""
    from ..em.cross_section_modes import CrossSectionMaterials, solve_cross_section_modes
    from .boundaries3d import BoundaryConfig3D
    from .mode_ports3d import port_boundary
    config = BoundaryConfig3D({'x_lo': 'periodic', 'x_hi': 'periodic', 'y_lo': 'pec', 'y_hi': 'pec'},
                              bloch_k=(bloch_kx, 0.0, 0.0))
    boundary = port_boundary(config, 2)
    n_cladding, n_core = (int(round(length / _PORT_CELL_Y)) for length in (_PORT_CLADDING, _PORT_CORE))
    dy = np.full(2 * n_cladding + n_core, _PORT_CELL_Y)
    dx = np.full(_PORT_CELLS_X, _PORT_CELL_X)
    eps = np.full((dx.size, dy.size), _PORT_INDICES[1] ** 2, dtype=np.complex128)
    eps[:, n_cladding:n_cladding + n_core] = _PORT_INDICES[0] ** 2
    materials = CrossSectionMaterials(eps_xx=eps, eps_yy=eps.copy(), eps_zz=eps.copy(),
                                      mu=np.ones(eps.shape, dtype=np.complex128), pec=np.zeros(eps.shape, dtype=bool))
    omega = 2.0 * math.pi * C0 / _PORT_WAVELENGTH
    return solve_cross_section_modes((dx, dy), boundary, _PORT_CELL_Y, materials, omega, _PORT_MODES), boundary


def _check_port_modes(_results_by_name):
    """port_boundary carries k_x; beta^2 shift by K_x^2 per mode and the e^{-j k_x dx} step of every field."""
    k0 = 2.0 * math.pi / _PORT_WAVELENGTH
    bloch_kx = _PORT_K_FRACTION * k0
    straight, _ = _port_solve(0.0)
    bloch, boundary = _port_solve(bloch_kx)
    errors = {'boundary': 0.0 if (boundary.bloch_x and boundary.bloch_kx == bloch_kx and not boundary.bloch_y
                                  and boundary.has_bloch_phase) else math.inf}
    transverse_squared = (2.0 / _PORT_CELL_X * math.sin(0.5 * bloch_kx * _PORT_CELL_X)) ** 2
    if len(straight.modes) < _PORT_MODES or len(bloch.modes) < _PORT_MODES:
        errors['mode_count'] = math.inf
    step = np.exp(-1j * bloch_kx * _PORT_CELL_X)
    for number, (mode_0, mode_k) in enumerate(zip(straight.modes, bloch.modes), start=1):
        expected = mode_0.beta ** 2 - transverse_squared
        errors[f'mode{number}_beta'] = abs(mode_k.beta ** 2 - expected) / k0 ** 2
        for field in 'eh':
            # scale per field (E or H), not per component: TE has E_y = 0 and TM H_y = 0 exactly, so those components
            # hold only round-off whose own max would make the step error O(1)
            components = {component: np.asarray(getattr(mode_k, field + component)) for component in 'xyz'}
            scale = max(float(np.max(np.abs(values))) for values in components.values())
            for component, values in components.items():
                errors[f'mode{number}_{field}{component}_step'] = (
                    float(np.max(np.abs(values[1:] - step * values[:-1]))) / scale if scale > 0.0 else math.inf)
    worst = max(errors, key=lambda item: math.inf if math.isnan(errors[item]) else errors[item])
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _PORT_TOLERANCE,
            'details': {'worst_item': worst, 'errors': errors,
                        'n_eff_k0': [[mode.n_eff.real, mode.n_eff.imag] for mode in straight.modes],
                        'n_eff_bloch': [[mode.n_eff.real, mode.n_eff.imag] for mode in bloch.modes]}}


# 'photonic_crystal_slab_bands_3d' template: square lattice (a = 0.5 um) of air holes (r = 0.3 a) in an eps 12 slab
# 0.275 um thick (11 cells of 25 nm, the mid-plane z = 1.1375 um on a cell centre), 1 um of vacuum and CPML above and
# below; one unit cell x / y periodic with bloch_kx = k_fraction 2 pi / a swept over Gamma-X. An x dipole on the
# mid-plane drives only the TE-like (mid-plane even) bands; the resonance finder lists them per k (guided below the
# light line lambda = a / k_fraction, leaky with finite Q above it).
SLAB_BANDS_TEMPLATE_KEY = 'photonic_crystal_slab_bands_3d'
SLAB_BANDS_LABEL = 'Photonic crystal slab band diagram (3D)'
SLAB_BANDS_DESCRIPTION = ('em_fdtd_3d: one unit cell of a square lattice of air holes (a = 0.5 um, r = 0.3 a) in a '
                          '0.275 um eps 12 slab with vacuum and CPML above and below, x / y periodic with bloch_kx = '
                          'k_fraction 2 pi / a swept over 0-0.5 (Gamma-X); an in-plane dipole on the mid-plane and a '
                          'resonance finder per k give the TE-like band frequencies and Q (guided below the light '
                          'line, leaky above it).')
_SLAB_LATTICE = 0.5e-6
_SLAB_HOLE_RADIUS = 0.3 * _SLAB_LATTICE
_SLAB_EPS = 12.0
_SLAB_Z = (1.0e-6, 1.275e-6)          # m: 11 cells, mid-plane 1.1375 um = a cell centre
_SLAB_HEIGHT = 2.275e-6               # m window height (91 cells)
_SLAB_CELL = 25e-9
_SLAB_K_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def photonic_crystal_slab_bands_spec():
    """Template 'photonic_crystal_slab_bands_3d': hole-slab unit cell, k sweep along Gamma-X, resonance finder."""
    mid_plane = 0.5 * (_SLAB_Z[0] + _SLAB_Z[1])
    centre = 0.5 * _SLAB_LATTICE
    periodic = {f'bc_{axis}_{side}': 'periodic' for axis in 'xy' for side in ('lo', 'hi')}
    return ModelSpec(
        title=SLAB_BANDS_LABEL,
        description=SLAB_BANDS_DESCRIPTION,
        parameters=[ParameterSpec('a', _SLAB_LATTICE, 'lattice constant [m]'),
                    ParameterSpec('k_fraction', _SLAB_K_FRACTIONS[-1],
                                  'Bloch kx in units of 2 pi / a (Gamma-X: 0-0.5)')],
        domain=DomainSpec(size_x=_SLAB_LATTICE, size_y=_SLAB_LATTICE, size_z=_SLAB_HEIGHT),
        geometry=[GeometrySpec('slab', 'box', {'x0': 0.0, 'y0': 0.0, 'z0': _SLAB_Z[0], 'x1': _SLAB_LATTICE,
                                               'y1': _SLAB_LATTICE, 'z1': _SLAB_Z[1]}),
                  GeometrySpec('hole', 'circle', {'cx': centre, 'cy': centre, 'r': _SLAB_HOLE_RADIUS,
                                                  'z0': _SLAB_Z[0], 'z1': _SLAB_Z[1]}),
                  GeometrySpec('source_point', 'point', {'x': 0.11e-6, 'y': 0.06e-6, 'z': mid_plane}),
                  GeometrySpec('monitor_point', 'point', {'x': 0.38e-6, 'y': 0.09e-6, 'z': mid_plane})],
        materials=[MaterialSpec('slab', properties={'eps_r': _SLAB_EPS}, selection=['slab']),
                   MaterialSpec('air_hole', properties={'eps_r': 1.0}, selection=['hole'])],   # later wins
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE,
                             settings={**periodic, 'pml_cells': 12, 'bloch_kx': 'k_fraction * 2 * pi / a'},
                             features=[
            FeatureSpec('dipole', 'dipole_source', ['source_point'],
                        {'dipole_axis': 'x', 'waveform': 'gaussian', 'wavelength': 1.8e-6, 'bandwidth': 0.8,
                         'current_moment': 1e-12}),
            FeatureSpec('bands', 'resonance_monitor', ['monitor_point'],
                        {'component': 'ex', 'wavelength_min': 1.1e-6, 'wavelength_max': 3.0e-6})])],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': 1e-12},
                        sweep=SweepSpec('k_fraction', list(_SLAB_K_FRACTIONS))))


SLAB_BANDS_TEMPLATE = (SLAB_BANDS_TEMPLATE_KEY, SLAB_BANDS_LABEL, SLAB_BANDS_DESCRIPTION,
                       photonic_crystal_slab_bands_spec,
                       'Band diagram of a photonic crystal slab with air holes: one unit cell, the phase step between '
                       'cells swept from the centre to the edge of the first zone, and the resonance finder listing '
                       'the band frequencies and their quality factors at each step.')


BLOCH_3D_CASES = (
    SelfTestCase(
        name='em3d_bloch_empty_lattice',
        label='EM 3D Bloch phase: empty-lattice bands (FDTD)',
        description='em_fdtd_3d vacuum unit cell 1.0 x 0.5 x 0.5 um (50 nm cells), periodic on all six faces, at Bloch '
                    'k = (0.25 g_x, 0.1 g_y, 0.1 g_z) (g = 2 pi / period; all three wrap axes carry a phase, so the '
                    'imaginary field set, every wrap sign and the edges where two phased wraps meet enter; a y-only '
                    'or z-only wrap fault is not told apart in the details, but shifts the bands). A z dipole pulse '
                    f'(gaussian 2 um, bandwidth 1) drives the real set for {_DURATION * 1e12:g} ps; a resonance_monitor '
                    '(E_z, band 1-5 um) finds the folded plane waves K = k (~2.6 um) and K = k - g_x (~1.25 um). '
                    'Expected: the exact '
                    'lossless Yee dispersion sin(omega dt / 2) = (C0 dt / 2) kappa, kappa^2 = sum (2/d)^2 sin^2(K_i d '
                    '/ 2), and Q = inf. error = the worst of |f / f_exact - 1| and 1/Q, tolerance 1e-3 (a missing wrap '
                    'phase moves the bands to the k = 0 lattice, tens of percent; H and E wraps of inconsistent sign '
                    'break the dispersion). The +-k sign itself is not seen here (time reversal: equal bands). Not yet '
                    'run with the single three-axis k and the shorter ring-down.',
        specs=lambda: {run: _lattice_spec(bloch_k) for run, bloch_k in _RUNS.items()},
        check=_check_lattice),
    SelfTestCase(
        name='em3d_bloch_array_monitors',
        label='EM 3D Bloch phase: imaginary-set monitors and k sign (FDTD)',
        description='em_fdtd_3d column 0.55 x 0.5 um (50 nm cells, x / y periodic), 3 um high with CPML z faces, Bloch '
                    'k = (+-0.15 g_x, 0.1 g_y, 0): one z dipole pulse (gaussian 1.55 um, bandwidth 0.5) per period '
                    'radiates a phased array whose only propagating order at 1.55 um is the plane wave K_t = (k_x, '
                    'k_y). Flux planes normal to x (opposite the dipole), y and z (2 um) and a field layer (2.2 um), '
                    "all at 1.55 um: the monitors add j F_im of the imaginary set. Expected: the mirror x -> 2 x_dipole "
                    '- x maps the +k_x run onto the -k_x run, so S_x(-k_x) = -S_x(+k_x) and S_y, S_z equal (exact up '
                    'to float32 round-off); the layer intensity is uniform (a single Bloch order; without the j F_im '
                    'term the layer would show the standing wave of +-K_t); the power flows along +k (F(x + L) = F(x) '
                    'e^{-j k L} with exp(+j omega t): S_z > 0, S_x and S_y > 1e-2 S_z for +k_x, S_x < 0 for -k_x). '
                    f'error = the largest relative deviation (a wrong direction: inf), tolerance 1e-3. '
                    f'{_ARRAY_DURATION * 1e15:g} fs runs, not yet run at that duration.',
        specs=lambda: {run: _array_spec(sign) for run, sign in _ARRAY_RUNS.items()},
        check=_check_array),
    SelfTestCase(
        name='em3d_bloch_port_modes',
        label='EM 3D Bloch phase: port modes at the Bloch k (CPU)',
        description='CPU, no run: the em_fdtd_3d port mode solve (mode_ports3d.port_boundary -> cross_section_modes) '
                    'of a z-normal port on a column x periodic (4 cells of 50 nm) at Bloch k_x = 0.4 k0, y pec '
                    '(natural port ends), a slab n = 2.0, 0.8 um thick, in n = 1.444 along y (20 nm cells), 1.55 um. '
                    'Uniform in x, the semi-discrete operator is invariant under rotation in the x-z plane once d/dx '
                    '-> -j K_x, K_x = (2/dx) sin(k_x dx / 2). Expected: port_boundary carries k_x on the periodic '
                    'axis; the first four modes (TE0, TM0, TE1, TM1) satisfy beta^2 = beta_0^2 - K_x^2 of the k = 0 '
                    'solve, and every field component steps by e^{-j k_x dx} from cell to cell (the sign convention '
                    'F(x + L) = F(x) e^{-j k L} of the engine). error = the largest relative deviation, tolerance '
                    f'{_PORT_TOLERANCE:g} (ARPACK round-off).',
        specs=dict, check=_check_port_modes),
)

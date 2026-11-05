"""Self-test of the em_fdtd_3d resonance_monitor (igor 1.2): the ring-down of a vacuum box with pec walls on all six
faces against its exact discrete Yee resonances (lossless: Q = inf)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_3d'          # literal (selftests never import the definition)
_RESONANCE_FEATURE = 'resonances'
_CELL = 50e-9                         # m
_CELLS = (22, 18, 14)                 # the outermost layer of each pec face is conductor: walls at centres 0 and n - 1
# TM_z (p, q, r) in the band: E_z ~ sin(p pi x' / Lx) sin(q pi y' / Ly) cos(r pi z' / Lz), L = (n - 1) dx;
# ~1.32 um, ~0.93 um, ~0.90 um; (1, 2, 0) ~0.79 um lies outside, TE_z modes (E_z = 0) are not excited
_MODES = ((1, 1, 0), (1, 1, 1), (2, 1, 0))
_BAND = (0.85e-6, 1.45e-6)            # m
_SOURCE_CELL = (5, 4, 3)              # z dipole, off the nodal planes of the three modes
_PROBE_CELL = (15, 12, 9)
_PULSE = {'dipole_axis': 'z', 'waveform': 'gaussian', 'wavelength': 1.05e-6, 'bandwidth': 0.5,
          'current_moment': 1e-12}
# s: ~57 periods of the longest band wavelength, ~140 decimated samples: the matrix pencil resolves the three lossless
# modes (~3 % apart) of the clean ring-down to well under 1e-5
_DURATION = 0.25e-12
_TOLERANCE = 1e-3


def _cavity_spec():
    """'run': vacuum box of pec walls, a z dipole pulse, a resonance_monitor (E_z) at another cell."""
    size_x, size_y, size_z = (cells * _CELL for cells in _CELLS)
    source = [(index + 0.5) * _CELL for index in _SOURCE_CELL]
    source[2] = (_SOURCE_CELL[2] + 0.8) * _CELL   # not halfway between two E_z faces
    probe = [(index + 0.5) * _CELL for index in _PROBE_CELL]
    settings = {f'bc_{axis}_{side}': 'pec' for axis in 'xyz' for side in ('lo', 'hi')}
    return ModelSpec(
        title='Self-test: pec box resonances (3D)',
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=[GeometrySpec('source_point', 'point', dict(zip('xyz', source))),
                  GeometrySpec('probe_point', 'point', dict(zip('xyz', probe)))],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, settings=settings, features=[
            FeatureSpec('dipole', 'dipole_source', ['source_point'], dict(_PULSE)),
            FeatureSpec(_RESONANCE_FEATURE, 'resonance_monitor', ['probe_point'],
                        {'component': 'ez', 'wavelength_min': _BAND[0], 'wavelength_max': _BAND[1]})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))


def _discrete_box_frequency(mode, dt):
    """Frequency [Hz] of box mode (p, q, r) from the lossless Yee recursion: sin(omega dt / 2) = (C0 dt / 2) kappa,
    kappa^2 = sum (2 / dx)^2 sin^2(p pi / (2 (n - 1)))."""
    kappa_squared = sum((2.0 / _CELL) ** 2 * math.sin(index * math.pi / (2 * (cells - 1))) ** 2
                        for index, cells in zip(mode, _CELLS))
    return 2.0 * math.asin(0.5 * C0 * dt * math.sqrt(kappa_squared)) / (2.0 * math.pi * dt)


def _check_box(results_by_name):
    """Frequency vs the discrete recursion and 1/Q (~0 for a lossless box) of each mode; worst of both."""
    table = results_by_name['run'].result(_PHYSICS_NAME, _RESONANCE_FEATURE)
    columns = list(table['columns'])
    rows = [list(row) for row in table['rows']]
    frequencies = [row[columns.index('frequency')] for row in rows]
    errors, per_mode = [], {}
    for mode in _MODES:
        label = 'TM{}{}{}'.format(*mode)
        expected = _discrete_box_frequency(mode, float(table['dt']))
        if not frequencies:
            errors.append(math.inf)
            per_mode[label] = None
            continue
        row = rows[int(np.argmin(np.abs(np.asarray(frequencies) - expected)))]
        measured = row[columns.index('frequency')]
        inverse_q = 2.0 * abs(row[columns.index('frequency_imag')]) / measured
        mode_error = max(abs(measured / expected - 1.0), inverse_q)
        errors.append(mode_error)
        per_mode[label] = {'frequency': measured, 'expected_frequency': expected, 'inverse_Q': inverse_q,
                           'error': mode_error}
    worst = int(np.argmax(errors))
    return {'measured': float(len(rows)), 'expected': float(len(_MODES)), 'error': float(errors[worst]),
            'tolerance': _TOLERANCE,
            'details': {'worst_mode': 'TM{}{}{}'.format(*_MODES[worst]), 'modes': per_mode, 'n_rows': len(rows),
                        'sample_interval': table.get('sample_interval'), 'n_samples': table.get('n_samples'),
                        'note': table.get('note')}}


RESONANCE_3D_CASES = (
    SelfTestCase(
        name='em3d_resonance_pec_box',
        label='EM 3D resonance finder: pec box (FDTD)',
        description='em_fdtd_3d vacuum box of pec walls on all six faces (22 x 18 x 14 cells of 50 nm; the outermost '
                    'cell layer of each face is conductor, so tangential E vanishes at the centres of cells 0 and n - '
                    f'1), a z dipole pulse (gaussian 1.05 um, bandwidth 0.5) rings it for {_DURATION * 1e12:g} ps and a '
                    'resonance_monitor (E_z, band 0.85-1.45 um) finds TM_z (1,1,0) ~1.32 um, (1,1,1) ~0.93 um and '
                    '(2,1,0) ~0.90 um. Expected: the lossless Yee recursion sin(omega dt / 2) = (C0 dt / 2) kappa, '
                    'kappa^2 = sum over x, y, z of (2/dx)^2 sin^2(p pi / (2 (n - 1))), and Q = inf. error = the worst '
                    'of |f / f_exact - 1| and 1/Q = 2 |frequency_imag| / f, tolerance 1e-3: float32 coefficients and '
                    'fit noise leave ~1e-6; a wall one cell off moves f by ~5 %, a lossy or growing wall shows in '
                    f'1/Q. Not yet run at {_DURATION * 1e12:g} ps.',
        specs=lambda: {'run': _cavity_spec()},
        check=_check_box),
)

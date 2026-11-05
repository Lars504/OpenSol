"""CPU vs GPU backend cases of em_fdtd_3d (solver_plan S6, core.kernels twins): existing analytic 3D FDTD cases run on
solver_defaults.backend 'cpu' (numpy twins of every engine, CPML, source, TFSF, monitor, mode-port and pole kernel;
numba twins of the single-precision field kernels when installed) and 'gpu' through physics/em
selftests_backend.backend_parity_case, plus the CPU case em3d_kernels_numba (numba vs numpy field kernel twins);
registered by definition.register."""
import math
from dataclasses import replace

import numpy as np

from ...core.constants import DTYPE
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import NUMBA_MISSING_DETAILS, numba_twin_result
from ..em.selftests_backend import backend_parity_case
from .selftests import SELFTEST_CASES, cubic_time_step
from .selftests_bloch3d import BLOCH_3D_CASES
from .selftests_materials import MATERIAL_CASES
from .selftests_mirror3d import MIRROR_3D_CASES
from .selftests_resonance3d import RESONANCE_3D_CASES
from .selftests_sources3d import SOURCE_CASES

_ALL_CASES = (*SELFTEST_CASES, *SOURCE_CASES, *RESONANCE_3D_CASES, *BLOCH_3D_CASES, *MATERIAL_CASES, *MIRROR_3D_CASES)


def _case(name, parity_value=None):
    """The registered case `name`; parity_value(details) -> float replaces its 'measured' (the parity compares it) for
    cases whose measured is an error near 0 or a count, so float32 rounding differences cannot dominate it."""
    case = next(case for case in _ALL_CASES if case.name == name)
    if parity_value is None:
        return case

    def check(results_by_name):
        outcome = dict(case.check(results_by_name))
        try:
            outcome['measured'] = float(parity_value(outcome['details']))
        except (KeyError, IndexError, TypeError, ValueError):   # a missing band / mode: NaN fails the parity loudly
            outcome['measured'] = math.nan
        return outcome
    return replace(case, check=check)


def _first_flux_power(details):
    """Full-domain flux box power at the first wavelength of the first mirror type (em3d_mirror_dipole_pair)."""
    key = sorted(name for name in details if name.endswith('_flux_power'))[0]
    return details[key][0]


# numba vs numpy field kernel twins (kernels3d, single): random fields / material ids (the last id PEC) / tables on a
# small grid, per periodic set (x, y, z), with Bloch wrap scalars, subpixel face overrides off and on, CPML slabs at
# both ends of every axis and an ohmic box; ms per call of both on the speed grid (details only)
_NUMBA_SHAPE = (9, 8, 7)
_NUMBA_PERIODIC_SETS = ((0, 0, 0), (1, 1, 1), (1, 0, 1))
_NUMBA_SPEED_SHAPE = (32, 32, 32)
_NUMBA_TIMED_CALLS = (3, 5)           # numpy, numba calls per timing (details only, not graded)
_NUMBA_TOLERANCE = 1e-5    # relative to the peak output: float32 round-off of a different operation order
_NUMBA_IDS = 6
_NUMBA_CELL = 1e-8
_NUMBA_SLAB_ROWS = 3
_ETA0 = 376.73


class _RandomGrid:
    """Random kernel inputs of one grid shape (float32 fields and tables, uint16 ids, int32 / float32 scalars)."""

    def __init__(self, rng, shape):
        self.rng = rng
        self.shape = shape
        self.ints = tuple(np.int32(n) for n in shape)
        self.dt = DTYPE(cubic_time_step(_NUMBA_CELL, 0.5))
        self.mat = rng.integers(0, _NUMBA_IDS, shape).astype(np.uint16)
        self.t_eps = [self.table(1.0, 12.0) for _ in range(3)]
        for table in self.t_eps:
            table[-1] = -1.0   # PEC id
        self.t_sig, self.t_mu, self.t_sm = self.table(0.0, 1e5), self.table(1.0, 3.0), self.table(0.0, 1e9)
        self.t_sig[0] = self.t_sm[0] = 0.0
        self.inverse = [(rng.uniform(0.8, 1.2, n) / _NUMBA_CELL).astype(np.float32) for n in shape]
        lower = [rng.uniform(0.3, 0.7, n).astype(np.float32) for n in shape]
        self.shares = [value for share in lower for value in (share, (1.0 - share).astype(np.float32))]
        # cylindrical curl factors (kernels3d header): random radii-like rows along x, and CPML row weights per axis
        radii = [rng.uniform(0.5, 2.0, shape[0]).astype(np.float32) for _ in range(3)]
        self.radial_h = (radii[0], (1.0 / radii[0]).astype(np.float32), (1.0 / radii[1]).astype(np.float32))
        self.radial_e = (radii[2], (1.0 / radii[2]).astype(np.float32), (1.0 / radii[0]).astype(np.float32))
        self.row_weights = [(rng.uniform(0.5, 2.0, n).astype(np.float32), rng.uniform(0.5, 2.0, n).astype(np.float32))
                            for n in shape]

    def table(self, lo, hi):
        return self.rng.uniform(lo, hi, _NUMBA_IDS).astype(np.float32)

    def fields(self, scale=1.0):
        return [(self.rng.standard_normal(self.shape) * scale).astype(np.float32) for _ in range(3)]

    def face_args(self, use_face):
        if not use_face:
            return np.zeros((1, 1, 1), dtype=np.uint16), np.zeros(3, dtype=np.float32), np.int32(0)
        ids = np.where(self.rng.random(self.shape) < 0.3, self.rng.integers(1, 4, self.shape), 0).astype(np.uint16)
        table = self.rng.uniform(1.0, 12.0, 12).astype(np.float32)
        table[self.rng.random(12) < 0.25] = 0.0   # 0 keeps the mean
        return ids, table, np.int32(1)

    def wrap_args(self):
        angles = self.rng.uniform(0.0, 2.0 * math.pi, 3)
        return tuple(DTYPE(value) for angle in angles for value in (math.cos(angle), math.sin(angle)))

    def slab(self, axis, row0, n_rows):
        """(psi_first, psi_second, b, a, inv) of a CPML slab."""
        shape = list(self.shape)
        shape[axis] = n_rows
        psi = [(self.rng.standard_normal(shape) * 1e-2).astype(np.float32) for _ in range(2)]
        decay = self.rng.uniform(0.5, 1.0, n_rows).astype(np.float32)
        return (*psi, decay, (decay - 1.0).astype(np.float32), self.inverse[axis][row0:row0 + n_rows].copy())

    def update_h(self):
        """H reads the wraps on every axis (the engine zeroes ih[n-1] on non-periodic ones)."""
        return (*self.fields(1.0 / _ETA0), *self.fields(), *self.fields(), self.mat, self.t_mu, self.t_sm,
                *self.inverse, *self.radial_h, *self.ints, self.dt, DTYPE(1.0), *self.wrap_args())

    def update_e(self, periodic, use_face):
        return (*self.fields(), *self.fields(1.0 / _ETA0), *self.fields(1.0 / _ETA0), self.mat, *self.t_eps,
                self.t_sig, *self.face_args(use_face), *self.inverse, *self.radial_e, *self.shares, *self.ints,
                *(np.int32(flag) for flag in periodic), self.dt, DTYPE(1.0), *self.wrap_args())

    def cpml_h(self, axis, row0):
        return (*self.fields(1.0 / _ETA0)[:2], *self.fields()[:2], *self.slab(axis, row0, _NUMBA_SLAB_ROWS),
                *self.row_weights[axis], self.mat, self.t_mu, self.t_sm, np.int32(axis), np.int32(row0),
                np.int32(_NUMBA_SLAB_ROWS), *self.ints, DTYPE(1.0), DTYPE(-1.0), self.dt, DTYPE(1.0))

    def cpml_e(self, axis, row0, periodic, use_face):
        first, second = (comp for comp in range(3) if comp != axis)
        return (*self.fields()[:2], *self.fields(1.0 / _ETA0)[:2], *self.slab(axis, row0, _NUMBA_SLAB_ROWS),
                *self.row_weights[axis], self.mat,
                *self.t_eps, self.t_sig, *self.face_args(use_face), *self.shares, np.int32(first), np.int32(second),
                np.int32(axis), np.int32(row0), np.int32(_NUMBA_SLAB_ROWS), *self.ints,
                *(np.int32(flag) for flag in periodic), DTYPE(-1.0), DTYPE(1.0), self.dt, DTYPE(1.0))

    def ohmic(self, periodic, origin, extents):
        q = (self.rng.random(self.shape) * 1e-12).astype(np.float32)
        return (q, *self.fields(), self.mat, self.t_sig, *(np.int32(n) for n in (*origin, *extents)), *self.ints,
                *(np.int32(flag) for flag in periodic), DTYPE(0.5 * self.dt))


def _check_kernels_numba(results_by_name):
    """numba vs numpy kernels3d field twins (single) on random inputs; ms per call of both on the speed grid."""
    from ...core.kernels import numba_available
    from . import kernels3d
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _NUMBA_TOLERANCE,
                'details': dict(NUMBA_MISSING_DETAILS)}
    import numba
    twins = {name: (numpy_twin, factory(numba)) for name, (numpy_twin, factory) in kernels3d._FIELD_HOST_TWINS.items()
             if factory is not None}   # conformal_h_3d has no numba twin
    grid = _RandomGrid(np.random.default_rng(20261018), _NUMBA_SHAPE)
    comparisons = {'update_h_3d': (*twins['update_h_3d'], grid.update_h())}   # no periodic flag: reads every wrap
    for periodic in _NUMBA_PERIODIC_SETS:
        tag = ''.join(str(flag) for flag in periodic)
        comparisons[f'ohmic_loss_3d:{tag}'] = (*twins['ohmic_loss_3d'], grid.ohmic(periodic, (0, 1, 2), (9, 6, 4)))
        for use_face in (0, 1):
            comparisons[f'update_e_3d:{tag}:face{use_face}'] = (*twins['update_e_3d'],
                                                                grid.update_e(periodic, use_face))
            for axis in range(3):
                n = _NUMBA_SHAPE[axis]
                for row0 in (1, n - _NUMBA_SLAB_ROWS):   # E slabs read row0 - 1, H slabs row0 + n_rows
                    comparisons[f'cpml_e_3d:{tag}:face{use_face}:axis{axis}:row{row0}'] = (
                        *twins['cpml_e_3d'], grid.cpml_e(axis, row0, periodic, use_face))
    for axis in range(3):
        for row0 in (0, _NUMBA_SHAPE[axis] - 1 - _NUMBA_SLAB_ROWS):
            comparisons[f'cpml_h_3d:axis{axis}:row{row0}'] = (*twins['cpml_h_3d'], grid.cpml_h(axis, row0))
    speed = _RandomGrid(np.random.default_rng(1), _NUMBA_SPEED_SHAPE)
    periodic = (0, 0, 0)
    timed_arguments = {'update_h_3d': speed.update_h(), 'update_e_3d': speed.update_e(periodic, 0),
                       'cpml_h_3d': speed.cpml_h(2, 0), 'cpml_e_3d': speed.cpml_e(2, 1, periodic, 0),
                       'ohmic_loss_3d': speed.ohmic(periodic, (0, 0, 0), _NUMBA_SPEED_SHAPE)}
    timings = {name: (*twins[name], arguments, *_NUMBA_TIMED_CALLS) for name, arguments in timed_arguments.items()}
    result = numba_twin_result(comparisons, timings, _NUMBA_TOLERANCE, numba)
    result['details']['speed_shape'] = list(_NUMBA_SPEED_SHAPE)
    result['details']['registered_host_kinds'] = {
        name: kernel.host_kind for name, kernel in kernels3d.field_kernels('single').items()}
    return result


_KERNELS_NUMBA_CASE = SelfTestCase(
    name='em3d_kernels_numba', label='EM 3D FDTD: numba vs numpy field kernels (CPU)',
    description=('The numba twins of the em_fdtd_3d single-precision field kernels (update_h_3d, update_e_3d, '
                 'cpml_h_3d, cpml_e_3d, ohmic_loss_3d) against their numpy twins on random '
                 f'{_NUMBA_SHAPE[0]} x {_NUMBA_SHAPE[1]} x {_NUMBA_SHAPE[2]} inputs with PEC ids, for periodic sets '
                 'none / all / x and z (update_h_3d, which takes no periodic flag, once), Bloch wrap scalars, random cylindrical curl factors and CPML row weights, '
                 'subpixel face overrides off and on, CPML slabs at both '
                 'ends of every axis and an ohmic box; measured = error = the worst max |numba - numpy| over every '
                 f'written array relative to its peak, tolerance {_NUMBA_TOLERANCE:g}. Details: ms per call of both '
                 f'on a {_NUMBA_SPEED_SHAPE[0]}^3 grid. Without numba nothing is compared (error 0).'),
    specs=lambda: {}, check=_check_kernels_numba, max_wall_time=600.0, heavy=True, backends=('cpu',))


EM3D_BACKEND_SELFTEST_CASES = (
    backend_parity_case(
        'em3d_fdtd_cpu_gpu', 'EM 3D FDTD: CPU backend vs GPU',
        (_case('em3d_far_field_dipole',           # CPML on six faces, dipole calibration, closed flux box, far field
               lambda details: details['flux']['power_per_cw'][1]),
         _case('em3d_plane_wave_fresnel'),        # one-sided TFSF, periodic x / y, flux planes
         _case('em3d_drude_film_fresnel'),        # Drude pole region (wrap faces), pole corrections
         _case('em3d_resonance_pec_box',          # pec wall layers, point probe, resonance finder
               lambda details: details['modes']['TM110']['frequency']),
         _case('em3d_bloch_empty_lattice',        # Bloch k != 0: imaginary field set, wrap mixing
               lambda details: details['bands']['kxyz_G000']['frequency'])),
        heavy=True, max_wall_time=1800.0),
    backend_parity_case(
        'em3d_fdtd_cpu_gpu_media', 'EM 3D FDTD: CPU backend vs GPU (mode ports, losses, mirrors, Bloch monitors)',
        (_case('em3d_mode_source_transmission'),  # mode source / monitors (host cross-section solves, cb planes)
         _case('em3d_heat_chain',                 # ohmic loss / q_accum into 3D heat, thermo-optic index
               lambda details: details['P_heat_W']),
         _case('em3d_mirror_dipole_pair', _first_flux_power),   # mirror rows, mirrored flux box and far field
         _case('em3d_bloch_array_monitors',       # imaginary-set flux planes and DFT layer
               lambda details: details['flux']['plus']['flux_z'])),
        heavy=True, max_wall_time=3600.0),
    _KERNELS_NUMBA_CASE,
)

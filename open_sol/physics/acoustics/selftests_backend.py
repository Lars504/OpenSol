"""CPU vs GPU backend cases of elastic_waves (solver_plan S6, core.kernels twins): the analytic acoustics cases run on
solver_defaults.backend 'cpu' (numpy twins of every kernel, numba twins of the velocity / stress updates when
installed) and 'gpu' through physics/em's `backend_parity_case`, plus the CPU case elastic_waves_kernels_numba (numba
vs numpy update twins); registered by definition.register."""
from dataclasses import replace

import numpy as np

from ...core.constants import DTYPE
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import NUMBA_MISSING_DETAILS, cases_named, numba_twin_result
from ..em.selftests_backend import PARITY_TOLERANCE, backend_parity_case
from .selftests import ELASTIC_CASES


def _acoustic_parity_case(name, label, case_names, covers, max_wall_time):
    """backend_parity_case over the named acoustics cases, its description naming the kernels they cover."""
    # ELASTIC_CASES incl. the cases run only inside these parities
    case = backend_parity_case(name, label, cases_named(ELASTIC_CASES, *case_names), heavy=True,
                               max_wall_time=max_wall_time)
    return replace(case, description=(
        f"The analytic elastic_waves cases {', '.join(case_names)} run on solver_defaults.backend cpu (core.kernels "
        f'twins, numba for the velocity / stress updates when installed, else numpy: {covers}) and on gpu. Each case\'s own check grades the cpu runs, and its measured value must '
        f'match the gpu runs\' within a relative {PARITY_TOLERANCE:g} (float32 arithmetic on both; the host twins '
        'round products and sums separately where CUDA may fuse them). error = the worst of cpu error / case '
        'tolerance and difference / parity tolerance, tolerance 1. Without a usable GPU only the cpu checks count.'))


# numba vs numpy velocity / stress update twins (single): random fields, coefficients (corners without shear) and CPML
# profiles on a small grid with x / y absorbing slabs, per periodic set (x, y) and kernel variant; ms per step (velocity
# + stress) of both on the speed grid (details only)
_NUMBA_SHAPE = (13, 11)
_NUMBA_SPEED_SHAPE = (512, 512)
_NUMBA_TOLERANCE = 1e-5   # relative to the peak output: float32 round-off of a different operation order
_NUMBA_PERIODIC_SETS = ((0, 0), (1, 0), (0, 1), (1, 1))
_NUMBA_LAYERS = ((3, 2), (2, 3))   # (lo, hi) absorbing cells along x, along y
_NUMBA_VELOCITY_VARIANTS = {'velocity': False, 'velocity_axisymmetric': True}
# key -> (axisymmetric, strain_accumulator, monoclinic)
_NUMBA_STRESS_VARIANTS = {'stress': (False, False, False), 'stress_axisymmetric': (True, False, False),
                          'stress_strain': (False, True, False), 'stress_strain_monoclinic': (False, True, True)}


class _RandomStrip:
    """Random update-kernel inputs of one (nx, ny) grid (float32 arrays, int32 / float32 scalars)."""

    def __init__(self, rng, shape):
        from .cpml import slab_map
        self.rng = rng
        self.shape = shape
        nx, ny = shape
        self.slab_x, self.slab_y = slab_map(nx, *_NUMBA_LAYERS[0]), slab_map(ny, *_NUMBA_LAYERS[1])
        self.n_slab = (sum(_NUMBA_LAYERS[0]), sum(_NUMBA_LAYERS[1]))
        self.profiles = []
        for n in shape:   # rows b, a (centres), b, a (hi faces), a = b - 1
            rows = np.empty((4, n), dtype=np.float32)
            rows[0::2] = rng.uniform(0.5, 1.0, (2, n))
            rows[1::2] = rows[0::2] - 1.0
            self.profiles.append(rows)
        self.inverse = tuple(rng.uniform(0.8, 1.2, n).astype(np.float32) for n in (nx, nx, ny, ny))
        self.radial = rng.uniform(0.2, 2.0, (7, nx)).astype(np.float32)
        self.ints = (np.int32(nx), np.int32(ny), np.int32(self.n_slab[1]))

    def array(self, lo=None, hi=None, shape=None):
        shape = self.shape if shape is None else shape
        values = self.rng.standard_normal(shape) if lo is None else self.rng.uniform(lo, hi, shape)
        return values.astype(np.float32)

    def common(self):
        return (*self.inverse, self.slab_x, self.slab_y, *self.profiles)

    def psi(self):
        nx, ny = self.shape
        return tuple(self.array(shape=shape) * np.float32(1e-2)
                     for shape in ((self.n_slab[0], ny), (self.n_slab[0], ny), (nx, self.n_slab[1]),
                                   (nx, self.n_slab[1])))

    def velocity(self, axisymmetric, periodic):
        fields = [self.array() for _ in range(6 if axisymmetric else 5)]
        coefficients = (self.array(0.9, 1.0), self.array(0.05, 0.15), self.array(0.9, 1.0), self.array(0.05, 0.15))
        return (*fields, *coefficients, *self.common(), *((self.radial,) if axisymmetric else ()), *self.psi(),
                DTYPE(1.0), *self.ints, *(np.int32(flag) for flag in periodic))

    def stress(self, axisymmetric, strain_accumulator, monoclinic, periodic):
        stresses = [self.array() for _ in range(4 if axisymmetric else 3)]
        shear = self.array(0.3, 1.0) * (self.rng.random(self.shape) > 0.3)   # corners without shear: c66c = 0
        moduli = (self.array(1.0, 3.0), self.array(0.2, 0.8), self.array(1.0, 3.0), shear.astype(np.float32))
        extra = [self.array() for _ in range(3 * (int(strain_accumulator) + int(monoclinic)))]
        return (*stresses, self.array(), self.array(), *moduli, *extra, *self.common(),
                *((self.radial,) if axisymmetric else ()), *self.psi(), DTYPE(0.1), DTYPE(1.0),
                self.array(0.5, 1.5), self.array(), DTYPE(0.05),   # the fused heating slot (gruneisen, q, scale)
                *self.ints, *(np.int32(flag) for flag in periodic))


def _check_updates_numba(results_by_name):
    """numba vs numpy velocity / stress update twins (single) on random inputs; ms per call of both at 512 x 512."""
    from ...core.kernels import numba_available
    from . import kernels
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _NUMBA_TOLERANCE,
                'details': dict(NUMBA_MISSING_DETAILS)}
    import numba
    twins = {key: (kernels._FIELD_HOST_TWINS[key], factory(numba))
             for key, factory in kernels._FIELD_NUMBA_TWINS.items()}
    grid = _RandomStrip(np.random.default_rng(20261018), _NUMBA_SHAPE)
    comparisons = {}
    for periodic in _NUMBA_PERIODIC_SETS:
        tag = ''.join(str(flag) for flag in periodic)
        for key, axisymmetric in _NUMBA_VELOCITY_VARIANTS.items():
            comparisons[f'{key}:{tag}'] = (*twins[key], grid.velocity(axisymmetric, periodic))
        for key, flags in _NUMBA_STRESS_VARIANTS.items():
            comparisons[f'{key}:{tag}'] = (*twins[key], grid.stress(*flags, periodic))
    speed = _RandomStrip(np.random.default_rng(1), _NUMBA_SPEED_SHAPE)
    timings = {'velocity': (*twins['velocity'], speed.velocity(False, (0, 0)), 5, 10),
               'stress': (*twins['stress'], speed.stress(False, False, False, (0, 0)), 5, 10)}
    result = numba_twin_result(comparisons, timings, _NUMBA_TOLERANCE, numba)
    result['details']['speed_shape'] = list(_NUMBA_SPEED_SHAPE)
    result['details']['registered_host_kinds'] = {key: kernels.field_kernels('single')[key].host_kind
                                                  for key in kernels._FIELD_NUMBA_TWINS}
    return result


_UPDATES_NUMBA_CASE = SelfTestCase(
    name='elastic_waves_kernels_numba', label='Elastic waves: numba vs numpy velocity / stress updates (CPU)',
    description=('The numba twins of the elastic_waves single-precision velocity and stress updates (planar, '
                 'axisymmetric, strain-accumulating and monoclinic variants) against their numpy twins on random '
                 f'{_NUMBA_SHAPE[0]} x {_NUMBA_SHAPE[1]} inputs with x / y absorbing slabs and corners without shear, '
                 'for every periodic set; measured = error = the worst max |numba - numpy| over every written array '
                 f'relative to its peak, tolerance {_NUMBA_TOLERANCE:g}. Details: ms per call of both on a '
                 f'{_NUMBA_SPEED_SHAPE[0]} x {_NUMBA_SPEED_SHAPE[1]} grid. Without numba nothing is compared '
                 '(error 0).'),
    specs=lambda: {}, check=_check_updates_numba, max_wall_time=600.0, heavy=True, backends=('cpu',))


BACKEND_SELFTEST_CASES = (
    _acoustic_parity_case(
        'elastic_waves_cpu_gpu', 'Elastic waves: CPU backend vs GPU',
        ('elastic_wave_speeds', 'acoustic_normal_reflection', 'acoustic_thermoelastic_slab',
         'acoustic_plate_resonances'),
        'planar velocity / stress updates with the x and y CPML, coefficients and stability stiffness of isotropic, '
        'orthotropic and fluid media, traction / force scatters, thermoelastic stress, pressure probes, slip walls and '
        'the resonance finder', 1800.0),
    _acoustic_parity_case(
        'elastic_waves_cpu_gpu_variants', 'Elastic waves: CPU backend vs GPU (axisymmetric, monoclinic)',
        ('elastic_axisymmetric_cylinder', 'elastic_axisymmetric_hoop_stress', 'elastic_monoclinic_plane_waves'),
        'the axisymmetric velocity / stress updates with the hoop stress and axisymmetric pressure, the hoop '
        'thermoelastic stress, the strain-accumulating monoclinic stress update and the c16 / c26 stress', 1800.0),
    _UPDATES_NUMBA_CASE,
)

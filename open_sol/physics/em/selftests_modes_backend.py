"""CPU vs GPU backend case of em_modes (solver_plan S6): a heated, partly metal-walled silicon strip (heat +
heat_em_thermo_optic + em_modes) on both backends; no physics imports."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role, with_exact_cg
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, StudySpec)

_TYPE = 'em_modes'
_PHYSICS = 'modes'
_HEAT = 'heat'
_TABLE = 'mode_table'
_N_EFF_COLUMN = 1                       # mode_table row index of n_eff
_HEATED_RUNS = {'cpu': 'heated_cpu', 'gpu': 'heated_gpu'}
# heated strip: 500 x 220 nm silicon in silica, 1.6 x 1.2 um window, uniform 20 nm cells (+15 cpml cells per side),
# Q in the core with the y_lo face held (a few K of rise), a small pec block in a corner (the cpu PEC mask path). The
# case compares cpu with gpu on one model, so the window only has to hold the core and the block, not converge.
_HEATED_CELL = 20e-9                    # m
_HEATED_DOMAIN = (1.6e-6, 1.2e-6)       # m
_HEATED_CORE = (0.56e-6, 0.50e-6, 1.06e-6, 0.72e-6)   # m, (x0, y0, x1, y1)
_HEATED_METAL = (0.10e-6, 0.90e-6, 0.30e-6, 1.10e-6)  # m
_HEATED_POWER_DENSITY = 1e14            # W/m^3 in the core (~11 W/m)
_HEATED_MODES = 2
_TOLERANCE = 1e-5   # relative n_eff, max-normalised e_intensity / T rise: one host eigen code, float32 inputs ~1e-7


def _on_backend(spec, backend):
    spec.solver_defaults.backend = backend
    return spec


def _heated_strip_spec(backend):
    """heat (exact pcg node) + em_modes with heat_em_thermo_optic, mode_analysis, on `backend`."""
    def rectangle(name, box):
        return GeometrySpec(name, 'rectangle', dict(zip(('x0', 'y0', 'x1', 'y1'), box)))

    spec = ModelSpec(
        title=f'em_modes heated strip ({backend})',
        domain=DomainSpec(size_x=_HEATED_DOMAIN[0], size_y=_HEATED_DOMAIN[1]),
        geometry=[rectangle('core', _HEATED_CORE), rectangle('metal', _HEATED_METAL)],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['core'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
                     FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': 293.15}),
                     FeatureSpec('core_heating', 'heat_source', ['core'], {'Q': _HEATED_POWER_DENSITY})]),
                 PhysicsSpec(_PHYSICS, _TYPE, settings={'wavelength': 1.55e-6, 'n_modes': _HEATED_MODES,
                                                        'pml_sigma_factor': 0.7},
                             features=[FeatureSpec(_TABLE, 'mode_table'),
                                       FeatureSpec('metal_block', 'pec_domain', ['metal'])])],
        couplings=[CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _PHYSICS)],
        mesh=MeshSpec(mode='uniform', dx=_HEATED_CELL),
        study=StudySpec(type='mode_analysis'),
        derived_values=[DerivedValueSpec('n_eff_1', f'{_PHYSICS}_n_eff_1'),
                        DerivedValueSpec('T_core', 'T', ['core'], 'mean')])
    return _on_backend(with_exact_cg(spec, [_HEAT]), backend)


def _backend_runs():
    """The heated strip on both backends (api skips the gpu one without a GPU)."""
    return {run: _heated_strip_spec(backend) for backend, run in _HEATED_RUNS.items()}


def _field(results, name):
    return np.asarray(results.field(name).values, dtype=np.float64)


def _check_backends(results_by_name):
    """Heated strip: relative n_eff of every mode, max-normalised e_intensity and the T rise cpu vs gpu; raises unless
    the cpu run recorded the eigen role on the cpu."""
    role = solver_role(results_by_name[_HEATED_RUNS['cpu']], _PHYSICS, 'eigen')
    if role.get('backend') != 'cpu':
        raise ValueError(f"run '{_HEATED_RUNS['cpu']}': expected the eigen role on the cpu backend, summary 'solvers' "
                         f'records {role!r}')
    errors = {}
    cpu = results_by_name[_HEATED_RUNS['cpu']]
    cpu_rows = cpu.result(_PHYSICS, _TABLE)['rows']
    cpu_derived = cpu.summary()['derived']
    details = {'cpu_n_eff': [float(row[_N_EFF_COLUMN]) for row in cpu_rows], 'cpu_T_core': float(cpu_derived['T_core'])}
    if len(cpu_rows) < _HEATED_MODES:
        errors['heated_cpu_modes_found'] = math.inf
    if _HEATED_RUNS['gpu'] not in results_by_name:   # no usable GPU: api.selftest skipped the gpu run
        details['gpu_runs'] = 'skipped (no usable GPU)'
        errors['heated_cpu_n_eff_finite'] = 0.0 if math.isfinite(float(cpu_derived['n_eff_1'])) else math.inf
    else:
        gpu = results_by_name[_HEATED_RUNS['gpu']]
        gpu_rows = gpu.result(_PHYSICS, _TABLE)['rows']
        n_common = min(len(cpu_rows), len(gpu_rows))
        errors['heated_mode_count'] = 0.0 if len(cpu_rows) == len(gpu_rows) else math.inf
        errors['heated_n_eff'] = max((abs(float(a[_N_EFF_COLUMN]) / float(b[_N_EFF_COLUMN]) - 1.0)
                                      for a, b in zip(cpu_rows[:n_common], gpu_rows[:n_common])), default=math.inf)
        intensity = {backend: _field(results_by_name[run], f'{_PHYSICS}.e_intensity')
                     for backend, run in _HEATED_RUNS.items()}
        errors['heated_e_intensity'] = (float(np.max(np.abs(intensity['cpu'] - intensity['gpu'])))
                                        / float(np.max(intensity['gpu'])))
        temperature = {backend: _field(results_by_name[run], f'{_HEAT}.T') for backend, run in _HEATED_RUNS.items()}
        rise = float(np.max(temperature['gpu']) - np.min(temperature['gpu']))
        errors['heated_T'] = float(np.max(np.abs(temperature['cpu'] - temperature['gpu']))) / rise
        details.update(gpu_n_eff=[float(row[_N_EFF_COLUMN]) for row in gpu_rows],
                       gpu_T_core=float(gpu.summary()['derived']['T_core']), T_rise_K=rise)
    finite = all(np.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
    error = max(errors.values()) if finite else float('nan')
    details['errors'] = errors
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE, 'details': details}


MODES_BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='em_modes_cpu_gpu',
        label='EM modes: CPU backend vs GPU',
        description=('em_modes on solver_defaults.backend cpu (numpy mode fields and MaterialGrid, host SuperLU + '
                     'ARPACK as on the gpu): a heated strip (a 500 x 220 nm silicon core in silica, '
                     f'{_HEATED_DOMAIN[0] * 1e6:g} x {_HEATED_DOMAIN[1] * 1e6:g} um window, uniform '
                     f'{_HEATED_CELL * 1e9:g} nm cells, Q = {_HEATED_POWER_DENSITY:g} W/m^3 in the core, y_lo held, '
                     'exact pcg heat, heat_em_thermo_optic into em_modes, a small pec_domain block) on both backends: '
                     'relative n_eff of every mode, max-normalised e_intensity of mode 1 and the temperature '
                     f'difference over the rise. Tolerance {_TOLERANCE:g}: the eigen solve is the same host code, only '
                     'the float32 thermo-optic transform and the device arrays differ (~1e-7). The cpu run must record '
                     'the eigen role on the cpu. em_modes_bloch_plate checks the cpu backend against an exact '
                     'reference (the closed-form PEC guide is the example analytic_modes_pec_rectangular_waveguide). '
                     'Without a '
                     'usable GPU the gpu run is skipped and only the cpu parts are checked. Not yet confirmed by a '
                     'run in the 1.6 x 1.2 um window.'),
        specs=_backend_runs, check=_check_backends, heavy=True),
)

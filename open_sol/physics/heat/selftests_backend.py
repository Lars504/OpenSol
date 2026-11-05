"""CPU vs GPU backend case of the DiffusionSystem physics (solver_plan S6 stage 2): heat and electric_currents models
solved on solver_defaults.backend 'cpu' (numpy / scipy: Automatic = host SuperLU, or a pcg node on numpy) against the
same models on 'gpu' and the analytic slab peak; registered by definition.register. api.selftest skips the gpu runs without a usable GPU."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role
from ..potential import selftests as potential_selftests
from . import selftests as heat_selftests

_SLAB_CPU = 'heated_slab_cpu_direct'
_POINT_RUNS = {'cpu': 'point_source_cpu_cg', 'gpu': 'point_source_gpu_cg'}
_DIVIDER_RUNS = {'cpu': 'divider_cpu_direct', 'gpu': 'divider_gpu_cg'}
_CURRENTS_POTENTIAL_FIELD = 'currents.V'
_TOLERANCE = 1e-6   # float32 fields of solves converged to 1e-12 (pcg) or exact (SuperLU): agreement ~1e-7


def _on_backend(spec, backend):
    spec.solver_defaults.backend = backend
    return spec


def _automatic(spec):
    """spec without solver nodes or assignments (every role Automatic)."""
    spec.solvers = []
    for physics in spec.physics:
        physics.solvers = {}
    return spec


def _backend_runs():
    """The heated slab template on the cpu (Automatic linear solver), the exact-CG point source on both backends and
    the terminal divider (cpu: Automatic SuperLU, gpu: its exact pcg node)."""
    point_source = heat_selftests._point_source_convection_runs
    divider = potential_selftests._terminal_divider_specs
    run = heat_selftests.RUN_NAME
    return {
        _SLAB_CPU: _on_backend(heat_selftests.heated_slab_spec(), 'cpu'),
        _POINT_RUNS['cpu']: _on_backend(point_source()[run], 'cpu'),
        _POINT_RUNS['gpu']: _on_backend(point_source()[run], 'gpu'),
        _DIVIDER_RUNS['cpu']: _on_backend(_automatic(divider()[potential_selftests._RUN]), 'cpu'),
        _DIVIDER_RUNS['gpu']: _on_backend(divider()[potential_selftests._RUN], 'gpu'),   # its exact pcg node
    }


def _field(results, name):
    return np.asarray(results.field(name).values, dtype=np.float64)


def _check_backends(results_by_name):
    """Largest of: the cpu slab peak rise vs Q L^2 / (8 k), max |T_cpu - T_gpu| / max rise and the energy balance of
    both point-source runs (scaled onto _TOLERANCE), max |V_cpu - V_gpu| / 1 V and the terminal current difference of
    the divider; raises when the cpu runs did not record backend 'cpu' (and SuperLU for the Automatic ones)."""
    for run, physics, solver_type in ((_SLAB_CPU, heat_selftests.PHYSICS_NAME, 'sparse_direct'),
                                      (_POINT_RUNS['cpu'], heat_selftests.PHYSICS_NAME, 'pcg'),
                                      (_DIVIDER_RUNS['cpu'], 'currents', 'sparse_direct')):
        role = solver_role(results_by_name[run], physics)
        if role.get('backend') != 'cpu' or role.get('type') != solver_type:
            raise ValueError(f"run '{run}': expected a cpu {solver_type} linear solver, summary 'solvers' records "
                             f'{role!r}')
    slab = heat_selftests._check_heated_slab({heat_selftests.RUN_NAME: results_by_name[_SLAB_CPU]})
    if _POINT_RUNS['gpu'] not in results_by_name:   # no usable GPU: api.selftest skipped the gpu runs
        return _check_cpu_only(results_by_name, slab)
    point_balance = {backend: _point_balance_error(results_by_name[run]) for backend, run in _POINT_RUNS.items()}
    temperature = {backend: _field(results_by_name[run], heat_selftests.TEMPERATURE_FIELD)
                   for backend, run in _POINT_RUNS.items()}
    rise = float(np.max(temperature['gpu']) - heat_selftests.DEFAULT_T0)
    point_error = float(np.max(np.abs(temperature['cpu'] - temperature['gpu']))) / rise
    potential = {backend: _field(results_by_name[run], _CURRENTS_POTENTIAL_FIELD)
                 for backend, run in _DIVIDER_RUNS.items()}
    potential_error = float(np.max(np.abs(potential['cpu'] - potential['gpu']))) / potential_selftests._VOLTAGE
    current = {backend: float(results_by_name[run].summary()['derived']['I_left'])
               for backend, run in _DIVIDER_RUNS.items()}
    current_error = abs(current['cpu'] - current['gpu']) / abs(current['gpu'])
    errors = {'slab_peak': slab['error'], 'point_source_T': point_error, 'divider_V': potential_error,
              'divider_current': current_error, 'point_source_balance_cpu': point_balance['cpu'],
              'point_source_balance_gpu': point_balance['gpu']}
    finite = all(np.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
    error = max(errors.values()) if finite else float('nan')
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'slab_peak_rise_K': slab['measured'], 'divider_current_A_per_m': current,
                        'cpu_solvers': {run: solver_role(results_by_name[run], physics) for run, physics in (
                            (_SLAB_CPU, heat_selftests.PHYSICS_NAME), (_DIVIDER_RUNS['cpu'], 'currents'))}}}


def _point_balance_error(results):
    """Energy-balance error of one point-source run (heat _check_point_source_convection) scaled onto _TOLERANCE."""
    point = heat_selftests._check_point_source_convection({heat_selftests.RUN_NAME: results})
    return point['error'] / point['tolerance'] * _TOLERANCE


def _check_cpu_only(results_by_name, slab):
    """Without the gpu runs: the slab peak plus the point-source and divider analytic checks of the cpu runs, each
    error scaled by its own tolerance onto _TOLERANCE."""
    divider = potential_selftests._check_terminal_divider(
        {potential_selftests._RUN: results_by_name[_DIVIDER_RUNS['cpu']]})
    errors = {'slab_peak': slab['error'],
              'point_source_balance_cpu': _point_balance_error(results_by_name[_POINT_RUNS['cpu']]),
              'divider_analytic': divider['error'] / divider['tolerance'] * _TOLERANCE}
    finite = all(np.isfinite(value) for value in errors.values())
    error = max(errors.values()) if finite else float('nan')
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'gpu_runs': 'skipped (no usable GPU)', 'slab_peak_rise_K': slab['measured']}}


BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='diffusion_cpu_gpu',
        label='Heat and electric currents: CPU backend vs GPU',
        description=('The DiffusionSystem physics on solver_defaults.backend cpu (numpy arrays, scipy CSR): the '
                     'heated_slab template with the Automatic cpu linear solver (host SuperLU) against the peak rise '
                     'Q L^2 / (8 k); the point heat source under convection (exact pcg node on numpy) and the '
                     'electric_currents terminal divider (cpu: Automatic SuperLU, gpu: exact pcg) against the same '
                     f'models on the gpu backend (max field difference, terminal current). Tolerance {_TOLERANCE:g}: '
                     'float32 fields of solves converged to 1e-12 or exact. Both point-source runs are also checked '
                     f'against their exact discrete energy balance ({heat_selftests.BOX_SIZE * 1e3:g} mm square, k_xx '
                     f'= {heat_selftests.THERMAL_CONDUCTIVITY:g}, k_yy = {heat_selftests.POINT_K_YY:g} W/(m*K), point '
                     f'P = {heat_selftests.POINT_POWER:g} W/m, inward heat_flux on x_lo, convection h = '
                     f'{heat_selftests.POINT_H:g} W/(m^2*K) on x_hi, y_lo, y_hi: the Robin outflow, sum (T_cell - '
                     'T_ext) w / (1 / h + w / (2 k_normal)), equals P + flux * L; tolerance '
                     f'{heat_selftests.EXACT_IDENTITY_TOLERANCE:g}, scaled onto {_TOLERANCE:g}; h w / (2 k) = 0.25 '
                     'on x_hi, so a wrong half-cell series term or k_normal fails; this pins the point source P / '
                     'cell area, the Neumann flux and the Robin faces in 2D). Without a usable GPU the gpu runs are '
                     'skipped and the cpu runs are checked against their closed forms instead (each error scaled '
                     'by its own tolerance).'),
        specs=_backend_runs, check=_check_backends),
)

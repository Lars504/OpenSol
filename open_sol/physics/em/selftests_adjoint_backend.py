"""CPU vs GPU backend case of the FDFD adjoint (adjoint.py, design_density.py on the run backend): the
em_fdfd_adjoint_gradient gradient runs, the eps_xy gradient runs and a short em_fdfd_topology_gap optimisation on
solver_defaults.backend 'cpu' against 'gpu' (the finite-difference check runs on the cpu in em_fdfd_adjoint_gradient).
No physics imports."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import worst_item
from .selftests_adjoint import (_GRADIENT_VALUE, _OBJECTIVES, _PHYSICS_NAME, _XY_GRADIENT_VALUE, _adjoint_specs,
                                _gradient_run, _objective_value, _xy_adjoint_specs)
from .selftests_fdfd_backend import _difference, _role_backends, _run_name
from .selftests_optimization import _DESIGN_FEATURE, _gap_spec

_BACKENDS = ('cpu', 'gpu')
_GRADIENT_KEYS = tuple(_gradient_run(polarization, objective) for polarization in ('ez', 'hz')
                       for objective in _OBJECTIVES)
_XY_KEYS = tuple(f'{objective}_gradient' for objective in _OBJECTIVES)
_OPTIMIZATION_RUN = 'gap'
_OPTIMIZATION_ITERATIONS = 3
_OPTIMIZATION_MIN_GAIN = 1.1     # J_final / J_initial of the cpu run at least (em_fdfd_topology_gap's 8 reach >= 1.5)
_TOLERANCE = 1e-5                # relative: complex128 assembly; host SuperLU on the cpu, Automatic on the gpu (cuDSS
                                 # double with nvmath-python: ~1e-10, well inside)
_OPTIMIZATION_TOLERANCE = 1e-4   # relative per history point: L-BFGS-B follows round-off differences of the gradient


def _optimization_spec():
    """em_fdfd_topology_gap with _OPTIMIZATION_ITERATIONS L-BFGS-B iterations."""
    spec = _gap_spec()
    spec.study.settings = {**spec.study.settings, 'max_iterations': _OPTIMIZATION_ITERATIONS}
    return spec


def _backend_runs():
    """Per backend the em_fdfd_adjoint_gradient gradient runs, the eps_xy gradient runs and the short optimisation (the
    finite differences run in em_fdfd_adjoint_gradient itself, on the cpu)."""
    runs = {}
    for backend in _BACKENDS:
        gradient_specs = {key: spec for key, spec in _adjoint_specs().items() if key in _GRADIENT_KEYS}
        groups = {'adjoint': gradient_specs,
                  'xy':{key: spec for key, spec in _xy_adjoint_specs().items() if key in _XY_KEYS},
                  'optimization': {_OPTIMIZATION_RUN: _optimization_spec()}}
        for group, specs in groups.items():
            for key, spec in specs.items():
                spec.solver_defaults.backend = backend
                runs[_run_name(group, key, backend)] = spec
    return runs


def _history(results):
    """(float64 objective history, its unit) of the design_region result."""
    history = results.result(_PHYSICS_NAME, _DESIGN_FEATURE)
    return np.asarray(history['series']['objective'], dtype=np.float64), history['series_units']['objective']


def _gradient_errors(cpu_results, gpu_results, value_names):
    """{item: relative cpu - gpu difference} of the eps_sensitivity field, the derived block integrals and both
    objective values of one gradient run."""
    errors = {}
    for field in ('eps_sensitivity',) + (('eps_xy_sensitivity',) if _XY_GRADIENT_VALUE in value_names else ()):
        name = f'{_PHYSICS_NAME}.{field}'
        errors[field] = _difference(np.asarray(cpu_results.field(name).values, dtype=np.float64),
                                    np.asarray(gpu_results.field(name).values, dtype=np.float64), phase=False)
    cpu_derived, gpu_derived = cpu_results.summary()['derived'], gpu_results.summary()['derived']
    for value_name in value_names:
        errors[value_name] = _difference(np.asarray(float(cpu_derived[value_name])),
                                         np.asarray(float(gpu_derived[value_name])), phase=False)
    for objective in _OBJECTIVES:
        errors[objective] = _difference(np.asarray(_objective_value(cpu_results, objective)),
                                        np.asarray(_objective_value(gpu_results, objective)), phase=False)
    return errors


def _check_backends(results_by_name):
    """cpu: J gain of the short optimisation and the W/m unit of its history; with the gpu runs: cpu - gpu differences
    in tolerance units. Raises unless every cpu role solver ran on the cpu."""
    cpu_names = [name for name in results_by_name if name.endswith('__cpu')]
    for name in cpu_names:
        backends = _role_backends(results_by_name[name])
        if not backends or any(backend != 'cpu' for backend in backends.values()):
            raise ValueError(f"run '{name}': expected every role solver of '{_PHYSICS_NAME}' on the cpu backend, "
                             f"summary 'solvers' records {backends!r}")
    cpu_history, unit = _history(results_by_name[_run_name('optimization', _OPTIMIZATION_RUN, 'cpu')])
    gain = float(cpu_history[-1] / cpu_history[0]) if cpu_history[0] > 0.0 else math.nan
    normalized = {'cpu optimisation gain': 0.0 if gain >= _OPTIMIZATION_MIN_GAIN else math.nan,
                  'objective unit W/m': 0.0 if unit == 'W/m' else math.nan}
    details = {'cpu_objective_history': cpu_history.tolist(),
               'cpu_gain': gain, 'objective_unit': unit, 'tolerance': _TOLERANCE,
               'optimization_tolerance': _OPTIMIZATION_TOLERANCE}
    if any(name.endswith('__gpu') for name in results_by_name):
        worst_items = {}
        for group, keys, value_names in (('adjoint', _GRADIENT_KEYS, (_GRADIENT_VALUE,)),
                                         ('xy', _XY_KEYS, (_GRADIENT_VALUE, _XY_GRADIENT_VALUE))):
            for key in keys:
                errors = _gradient_errors(results_by_name[_run_name(group, key, 'cpu')],
                                          results_by_name[_run_name(group, key, 'gpu')], value_names)
                worst_items[f'{group} {key}'], error = worst_item(errors)
                normalized[f'{group} {key}'] = error / _TOLERANCE
        gpu_history, _ = _history(results_by_name[_run_name('optimization', _OPTIMIZATION_RUN, 'gpu')])
        history_error = _difference(cpu_history, gpu_history, phase=False)
        normalized['optimisation history'] = history_error / _OPTIMIZATION_TOLERANCE
        details.update(gpu_objective_history=gpu_history.tolist(), worst_item_per_run=worst_items)
    else:
        details['gpu_runs'] = 'skipped (no usable GPU)'
    worst_name, worst = worst_item(normalized)
    details.update(normalized_errors=normalized, worst_item=worst_name)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


ADJOINT_BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_adjoint_cpu_gpu',
        label='EM FDFD adjoint and optimisation: CPU backend vs GPU',
        description=('The FDFD adjoint (adjoint.py, design_density.py) on solver_defaults.backend cpu against gpu. '
                     'Runs per backend: the em_fdfd_adjoint_gradient gradient runs (ez and hz, one objective weighted '
                     'per run, its 40 nm layout), the two em_fdfd_hz_eps_xy_adjoint_gradient gradient runs and '
                     f'em_fdfd_topology_gap cut to {_OPTIMIZATION_ITERATIONS} L-BFGS-B iterations (the finite '
                     'differences run on the cpu in em_fdfd_adjoint_gradient itself, so they are not repeated here). '
                     'Items, each in tolerance units: the cpu optimisation raising J by at least '
                     f'the factor {_OPTIMIZATION_MIN_GAIN:g} (NaN below), the design_region history unit W/m; with the '
                     'gpu runs per gradient run the largest relative cpu - gpu difference of eps_sensitivity (and '
                     'eps_xy_sensitivity), the derived block integrals and both objective values (tolerance '
                     f'{_TOLERANCE:g}: complex128 assembly; the gpu Automatic linear solver may be cuDSS) and the '
                     'objective history of the '
                     f'optimisation ({_OPTIMIZATION_TOLERANCE:g}: L-BFGS-B follows the round-off of the gradient). '
                     'measured = error = the worst item, tolerance 1. The check raises unless every cpu role solver '
                     'records backend cpu. Not yet confirmed by a run.'),
        specs=_backend_runs, check=_check_backends, max_wall_time=900.0, heavy=True),
)

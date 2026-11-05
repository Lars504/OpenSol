"""CPU vs GPU backend case of drift_diffusion (solver_plan S6 stage 2): the pn junction on solver_defaults.backend
'cpu' against its closed form and against the same model on 'gpu' (the system is host float64 on both and the linear
solver pinned to SuperLU, so only the field arrays differ). Registered by definition.register; api.selftest skips the
gpu run without a usable GPU."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role
from ...model.spec import SolverSpec
from . import selftests as semiconductor_selftests

_MODEL = 'pn_junction'
_COMPARED_FIELDS = ('V', 'N', 'P')
_TOLERANCE = 1e-6   # the same host float64 solve on both backends: float32 field rounding only
# pinned: Automatic resolves to cuDSS on the gpu (B1a), compared with SuperLU by semiconductor_cudss
_SUPERLU_NODE = 'sparse_direct_scipy_splu'


def _run_name(backend):
    return f'{_MODEL}_{backend}'


def _backend_runs():
    runs = {}
    for backend in ('cpu', 'gpu'):
        spec = semiconductor_selftests._pn_junction_specs()[semiconductor_selftests._RUN]
        spec.solver_defaults.backend = backend
        spec.solvers.append(SolverSpec(_SUPERLU_NODE, 'sparse_direct', {'back_end': 'scipy_splu'}))
        spec.physics_by_name(semiconductor_selftests._PHYSICS).solvers['linear'] = _SUPERLU_NODE
        runs[_run_name(backend)] = spec
    return runs


def _check_backends(results_by_name):
    """Largest of: the cpu run's closed-form error scaled by its own tolerance onto _TOLERANCE and (with the gpu run)
    max |f_cpu - f_gpu| / max |f_gpu| of V, N and P; raises when the cpu run did not record backend 'cpu'."""
    physics = semiconductor_selftests._PHYSICS
    cpu_results = results_by_name[_run_name('cpu')]
    role = solver_role(cpu_results, physics)
    if role.get('backend') != 'cpu':
        raise ValueError(f"run '{_run_name('cpu')}': expected a cpu linear solver, summary 'solvers' records {role!r}")
    outcome = semiconductor_selftests._check_pn_junction({semiconductor_selftests._RUN: cpu_results})
    errors = {f'{_MODEL}_closed_form': outcome['error'] / outcome['tolerance'] * _TOLERANCE}
    gpu_results = results_by_name.get(_run_name('gpu'))   # absent without a usable GPU
    if gpu_results is not None:
        for field in _COMPARED_FIELDS:
            cpu_values, _ = semiconductor_selftests._window(cpu_results, field)
            gpu_values, _ = semiconductor_selftests._window(gpu_results, field)
            scale = float(np.max(np.abs(gpu_values)))
            errors[f'{_MODEL}_{field}_cpu_gpu'] = float(np.max(np.abs(cpu_values - gpu_values))) / scale
    finite = all(np.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
    error = max(errors.values()) if finite else float('nan')
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'gpu_runs': gpu_results is not None}}


BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='semiconductor_cpu_gpu',
        label='Drift-diffusion: CPU backend vs GPU',
        description=('drift_diffusion on solver_defaults.backend cpu (numpy output fields; the host float64 '
                     'Scharfetter-Gummel system and its SuperLU solve as on the gpu; both runs pin a sparse_direct '
                     "node to scipy_splu, since Automatic is cuDSS on the gpu): an abrupt pn junction in "
                     'equilibrium (N_A 1e23 | N_D 2e23 1/m^3 at 0.5 um, 1 nm cells, both ohmic contacts at 0 V) '
                     'against its closed form (built-in potential, junction potential and face field from the exact '
                     'Poisson-Boltzmann first integral, flat quasi-Fermi levels; tolerance 2e-3, the error scaled '
                     f'by it onto {_TOLERANCE:g}) and against the same model on the gpu backend (max V, N and P '
                     'difference relative to the field maximum); the run must record a cpu linear solver. Without a '
                     'usable GPU the gpu run is skipped. Not yet run with the junction alone.'),
        specs=_backend_runs, check=_check_backends),
)

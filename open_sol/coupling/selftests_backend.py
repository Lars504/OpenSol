"""CPU vs GPU backend cases of the coupled chains (solver_plan S8 E, the coupling transforms as core.kernels twins): the
existing coupling cases whose partners all run on the cpu, on solver_defaults.backend 'cpu' and 'gpu'; each case's
own check grades the cpu runs and its measured value must match the gpu runs'. Registered by definitions.register
(chains) and definitions_pockels.register (Pockels); no physics imports."""
import math

from ..model.schema import SelfTestCase
from .selftests import CASE, CURRENTS_CASE, FDTD_CASE, HZ_CASE
from .selftests_index_sum import POCKELS_INDEX_SUM_CASE
from .selftests_slot_sum import SUPERPOSITION_CASE

BACKENDS = ('cpu', 'gpu')
PARITY_TOLERANCE = 1e-3    # relative |measured_cpu - measured_gpu|: float32 fields, 2e-6 fixed points, host rounding
_SEPARATOR = '__'


def _run_name(backend, case, run):
    return f'{backend}{_SEPARATOR}{case.name}{_SEPARATOR}{run}'


def _role_backends(results):
    """{(physics, role): backend} of every solver role in the run summary's 'solvers' record."""
    solvers = results.summary().get('solvers') or {}
    return {(physics, role): (record or {}).get('backend')
            for physics, physics_record in solvers.items()
            for role, record in ((physics_record or {}).get('roles') or {}).items()}


def _relative_difference(value, reference):
    value, reference = float(value), float(reference)
    if not (math.isfinite(value) and math.isfinite(reference)):
        return 0.0 if value == reference else math.inf
    return abs(value - reference) / max(abs(reference), 1e-300)


def _parity_values(outcome):
    """{name: value} compared cpu vs gpu: the check's details 'parity_measured' when it gives one, else its measured."""
    return (outcome.get('details') or {}).get('parity_measured') or {'measured': outcome['measured']}


def coupling_backend_case(name, label, wrapped_cases, heavy=False, max_wall_time=600.0,
                          parity_tolerance=PARITY_TOLERANCE):
    """SelfTestCase running every wrapped case's specs on both backends: error = the worst of (cpu check error / its
    tolerance) and (cpu vs gpu measured difference / parity_tolerance), tolerance 1; raises when a cpu run recorded a
    solver role off the cpu. Without a usable GPU the gpu runs are skipped (api.selftest), only the cpu checks count."""
    wrapped_cases = tuple(wrapped_cases)

    def specs():
        runs = {}
        for backend in BACKENDS:
            for case in wrapped_cases:
                for run, spec in case.specs().items():
                    spec.solver_defaults.backend = backend
                    runs[_run_name(backend, case, run)] = spec
        return runs

    def check(results_by_name):
        errors, details = {}, {}
        for case in wrapped_cases:
            runs = {backend: {} for backend in BACKENDS}
            for run_name, results in results_by_name.items():
                backend, case_name, run = run_name.split(_SEPARATOR, 2)
                if case_name == case.name:
                    runs[backend][run] = results
            for run, results in runs['cpu'].items():
                roles = _role_backends(results)
                if not roles or any(value != 'cpu' for value in roles.values()):
                    raise ValueError(f"case '{case.name}' run '{run}': expected every solver role on the cpu, summary "
                                     f"'solvers' records {roles!r}")
            cpu = case.check(runs['cpu'])
            errors[f'{case.name}.cpu_check'] = float(cpu['error']) / float(cpu['tolerance'])
            entry = {'cpu_measured': cpu['measured'], 'expected': cpu['expected'], 'cpu_error': cpu['error'],
                     'tolerance': cpu['tolerance']}
            if runs['gpu']:
                gpu = case.check(runs['gpu'])
                cpu_values, gpu_values = _parity_values(cpu), _parity_values(gpu)
                differences = {key: _relative_difference(value, gpu_values[key]) for key, value in cpu_values.items()}
                parity = max(differences.values())   # _relative_difference never returns NaN
                errors[f'{case.name}.parity'] = parity / parity_tolerance
                if len(differences) > 1:
                    entry['relative_differences'] = differences
                entry.update(gpu_measured=gpu['measured'], gpu_error=gpu['error'], relative_difference=parity)
            else:
                entry['gpu_runs'] = 'skipped (no usable GPU)'
            details[case.name] = entry
        finite = all(math.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
        error = max(errors.values()) if finite else math.nan
        worst = max(errors, key=lambda key: math.inf if not math.isfinite(errors[key]) else errors[key])
        return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
                'details': {'worst': worst, 'normalised_errors': errors, 'cases': details,
                            'parity_tolerance': parity_tolerance}}

    case_names = ', '.join(case.name for case in wrapped_cases)
    return SelfTestCase(
        name=name, label=label,
        description=(f'The coupling cases {case_names} run on solver_defaults.backend cpu (numpy arrays, the coupling '
                     'transforms and SlotSums through their core.kernels numpy twins, host solvers) and on gpu. Each '
                     "case's own check grades the cpu runs, every solver role of a cpu run must record backend cpu, "
                     "and the case's measured value (each of its details 'parity_measured' when given) must match the "
                     "gpu runs' within a relative "
                     f'{parity_tolerance:g} (float32 fields, fixed points to 2e-6, host rounding). error = the worst '
                     'of cpu error / case tolerance and difference / parity tolerance, tolerance 1. Without a usable '
                     'GPU only the cpu checks count.'),
        specs=specs, check=check, heavy=heavy, max_wall_time=max_wall_time)


# registered by definitions.register (em_fdfd_ez / _hz, em_fdtd_hz, heat, electric_currents)
CHAIN_BACKEND_CASES = (
    coupling_backend_case(
        'coupling_cpu_gpu', 'Couplings: EM / currents + heat chains, CPU backend vs GPU',
        (CASE,                  # em_fdfd_ez + heat: em_heat_joule, heat_em_thermo_optic
         CURRENTS_CASE,         # electric_currents + heat + em_fdfd_ez: currents_heat_joule
         HZ_CASE,               # em_fdfd_hz: eps_xx / eps_yy index slots with expansion, field_transfer
         SUPERPOSITION_CASE),   # two 'sum' contributors on volumetric_power
        max_wall_time=1800.0),
    coupling_backend_case(
        'coupling_fdtd_cpu_gpu', 'Couplings: FDTD + transient heat, CPU backend vs GPU',
        (FDTD_CASE,),           # em_fdtd_hz windowed Joule power, FDTD thermo-optic slots
        max_wall_time=3600.0),
)

# registered by definitions_pockels.register (electrostatics + em_fdfd_hz)
POCKELS_BACKEND_CASES = (
    coupling_backend_case(
        'pockels_cpu_gpu', 'Couplings: Pockels and the thermo-optic + Pockels index sum, CPU backend vs GPU',
        (POCKELS_INDEX_SUM_CASE,), max_wall_time=1800.0),   # 3 runs per backend (shared reference)
)

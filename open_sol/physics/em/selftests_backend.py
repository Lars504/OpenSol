"""CPU vs GPU backend cases of em_fdtd_ez / _hz (solver_plan S6, core.kernels twins): existing analytic FDTD cases run
on solver_defaults.backend 'cpu' (numpy / numba twins of every kernel) and 'gpu'; each wrapped case's own check grades
the cpu runs and their measured value must match the gpu runs'. `backend_parity_case` is the reusable wrapper (em3d /
acoustics / fluid cases can wrap theirs the same way); registered by definition.register."""
import math

from ...model.schema import SelfTestCase
from .selftests import LINE_CURRENT_HANKEL_CASE
from .selftests_bloch import oblique_fresnel_case
from .selftests_hz import HZ_MIE_CYLINDER_CASE, fresnel_case, mie_drude_cylinder_case
from .selftests_mode_ports import mode_port_fresnel_case, mode_port_slab_case
from .selftests_nonlinear import nonlinear_case

BACKENDS = ('cpu', 'gpu')
PARITY_TOLERANCE = 1e-3    # relative |measured_cpu - measured_gpu|: float32 rounding (no FMA on the host) only
_SEPARATOR = '__'
# em_fdtd_cpu_gpu_media: the Drude cylinder on 5 nm cells (4x fewer cells, 2x fewer steps than its 2.5 nm base case),
# graded by parity only: its Mie / energy budget is the 2.5 nm one, which em_mie_drude_cylinder grades on the gpu
_COARSE_DRUDE_CELL = 5e-9


def _run_name(backend, case, run):
    return f'{backend}{_SEPARATOR}{case.name}{_SEPARATOR}{run}'


def _stepper_backends(results):
    """{physics: backend} of every 'stepper' role in the run summary's 'solvers' record."""
    solvers = results.summary().get('solvers') or {}
    return {physics: (record.get('roles') or {}).get('stepper', {}).get('backend')
            for physics, record in solvers.items() if 'stepper' in (record.get('roles') or {})}


def _relative_difference(value, reference):
    value, reference = float(value), float(reference)
    if not (math.isfinite(value) and math.isfinite(reference)):
        return 0.0 if value == reference else math.inf
    return abs(value - reference) / max(abs(reference), 1e-300)


def backend_parity_case(name, label, wrapped_cases, heavy=False, max_wall_time=600.0,
                        parity_tolerance=PARITY_TOLERANCE, parity_only=()):
    """SelfTestCase running every wrapped case's specs on both backends: error = the worst of (cpu check error / its
    tolerance; not for the case names in parity_only, whose cpu check is a detail) and (cpu vs gpu measured difference
    / parity_tolerance), tolerance 1. Without a usable GPU the gpu runs are skipped (api.selftest) and only the graded
    cpu checks count; raises when a cpu run's stepper did not record backend cpu."""
    wrapped_cases = tuple(wrapped_cases)
    parity_only = frozenset(parity_only)
    unknown = parity_only - {case.name for case in wrapped_cases}
    if unknown:
        raise ValueError(f'parity_only names no wrapped case: {sorted(unknown)}')

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
                steppers = _stepper_backends(results)
                if not steppers or any(value != 'cpu' for value in steppers.values()):
                    raise ValueError(f"case '{case.name}' run '{run}': expected cpu stepper roles, summary 'solvers' "
                                     f'records {steppers!r}')
            cpu = case.check(runs['cpu'])
            entry = {'cpu_measured': cpu['measured'], 'expected': cpu['expected'], 'cpu_error': cpu['error'],
                     'tolerance': cpu['tolerance']}
            if case.name in parity_only:
                entry['cpu_check'] = 'detail only (parity graded)'
            else:
                errors[f'{case.name}.cpu_check'] = float(cpu['error']) / float(cpu['tolerance'])
            if runs['gpu']:
                gpu = case.check(runs['gpu'])
                parity = _relative_difference(cpu['measured'], gpu['measured'])
                errors[f'{case.name}.parity'] = parity / parity_tolerance
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
    parity_text = (f" Graded by parity only (cpu check error in details): {', '.join(sorted(parity_only))}."
                   if parity_only else '')
    return SelfTestCase(
        name=name, label=label,
        description=(f'The analytic FDTD cases {case_names} run on solver_defaults.backend cpu (core.kernels numpy '
                     'twins of every engine, CPML, source, TFSF, monitor, mode-port and medium kernel; numba twins of '
                     'the Yee updates when numba is installed) and on gpu. Each case\'s own check grades the cpu runs, '
                     'and its measured value must match the gpu runs\' within a relative '
                     f'{parity_tolerance:g} (float32 arithmetic on both; the host twins round products and sums '
                     'separately where CUDA may fuse them, ~1e-7 per step). error = the worst of cpu error / case '
                     'tolerance and difference / parity tolerance, tolerance 1. Without a usable GPU only the cpu '
                     f'checks count.{parity_text}'),
        specs=specs, check=check, heavy=heavy, max_wall_time=max_wall_time)


_COARSE_DRUDE_CASE = mie_drude_cylinder_case('em_fdtd_ez', 'ez', cell=_COARSE_DRUDE_CELL)

FDTD_BACKEND_SELFTEST_CASES = (
    backend_parity_case(
        'em_fdtd_cpu_gpu', 'EM FDTD: CPU backend vs GPU',
        (LINE_CURRENT_HANKEL_CASE,                                  # ez: CPML, point source, probe, far field
         fresnel_case('em_fdtd_hz', 'hz', graded=False),            # hz: line source, periodic y, flux monitor
         mode_port_fresnel_case('hz'),                              # hz: mode source / monitors
         nonlinear_case('tpa', 'hz'),                               # hz: TPA correction, face pole power, q_accum_hz
         nonlinear_case('gain', 'ez')),                             # ez: four-level gain
        heavy=True, max_wall_time=1800.0),
    backend_parity_case(
        'em_fdtd_cpu_gpu_media', 'EM FDTD: CPU backend vs GPU (TFSF, Drude, Bloch)',
        (_COARSE_DRUDE_CASE,                                         # ez: TFSF, Drude pole, q_accum (parity only)
         HZ_MIE_CYLINDER_CASE,                                       # hz: TFSF, closed flux box
         oblique_fresnel_case('em_fdtd_hz', 'hz'),                   # hz: Bloch k != 0 (imaginary field set)
         mode_port_slab_case('ez')),                                 # ez: guided mode ports through the y PML
        heavy=True, max_wall_time=3600.0, parity_only=(_COARSE_DRUDE_CASE.name,)),
)

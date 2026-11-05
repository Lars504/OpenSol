"""CPU vs GPU backend case of em_fdfd_nonlinear (solver_plan S8 E): undepleted SHG strips (E_z, and the H_z pump with
the C collocation), pump-depleted Kerr SPM (picard: lagged operator term + Anderson; newton: tangents + real GMRES), a
surface chi2 dipole layer, the three 2D-material line sheets, the cascade / pump-depleted adjoint gradients and the SHG
optimisation run on solver_defaults.backend 'cpu' against 'gpu'; no physics imports. Without a usable GPU api.selftest
skips the gpu runs and the cpu runs meet their own references."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import probe_phasor, relative_error, single_value, worst_item
from .selftests import (_PHYSICS, _check_mixed_polarization, _check_shg_uniform, _check_spm,
                        _mixed_polarization_specs, _shg_uniform_specs, _solve_record, _spm_specs)
from .selftests_adjoint import (_check_depleted_gradient, _check_gradient, _check_optimization, _depleted_specs,
                                _gradient_specs, _optimization_specs)
from .selftests_sheets import _check_sheets, _check_surface, _sheet_specs, _surface_specs

_BACKENDS = ('cpu', 'gpu')
_TOLERANCE = 1e-5          # relative: complex128 assembly and the same host SuperLU on both; depleted fixed points 1e-8
_CHANNEL_FLOOR = 1e-12     # a channel's probe scale is at least this x the run's largest probe (all-zero channels)
_LIVE_FIELDS = ('e_abs', 'ez_re', 'ez_im', 'hz_re', 'hz_im')   # first input channel, re-selected after the study
# gradient groups: the finite-difference neighbours (_plus / _minus) run on the cpu only (its fallback reference);
# only the *_gradient runs are compared
_GRADIENT_GROUPS = ('adjoint', 'depleted_adjoint')
_COMPARED_SUFFIX = '_gradient'


def _groups():
    """{group: {key: spec}} of the models (the nlfd / sheet / adjoint self-test builders)."""
    return {
        'shg': _shg_uniform_specs(),
        'mixed': _mixed_polarization_specs(),
        'kerr': _spm_specs(),
        'surface': _surface_specs(),
        'sheet': _sheet_specs(),
        'adjoint': _gradient_specs(),
        'depleted_adjoint': _depleted_specs(),
        'optimization': _optimization_specs(),
    }


def _compared(group, key):
    """True for a model run on both backends (else cpu only)."""
    return group not in _GRADIENT_GROUPS or key.endswith(_COMPARED_SUFFIX)


def _run_name(group, key, backend):
    return f'{group}__{key}__{backend}'


def _backend_runs():
    """Every model of _groups() on the cpu and the compared ones on the gpu (solver_defaults.backend set)."""
    runs = {}
    for backend in _BACKENDS:
        for group, specs in _groups().items():
            for key, spec in specs.items():
                if backend == 'cpu' or _compared(group, key):
                    spec.solver_defaults.backend = backend
                    runs[_run_name(group, key, backend)] = spec
    return runs


def _features(spec, feature_type):
    """{feature name: channel name} of the enabled `feature_type` features of the model's nonlinear physics."""
    return {feature.name: feature.params.get('channel') for feature in spec.physics[0].features
            if feature.enabled and feature.type == feature_type}


def _scalar_errors(spec, cpu_results, gpu_results):
    """{item: relative difference} of the adjoint objective values, derived values and the optimization record."""
    errors = {}
    for name in _features(spec, 'adjoint_objective'):
        values = [single_value(results.result(_PHYSICS, name)['series']['value'], f'{name} objective')
                  for results in (cpu_results, gpu_results)]
        errors[f'objective {name}'] = relative_error(*values)
    cpu_summary, gpu_summary = cpu_results.summary(), gpu_results.summary()
    cpu_derived = cpu_summary.get('derived') or {}
    for name, gpu_value in (gpu_summary.get('derived') or {}).items():
        errors[f'derived {name}'] = relative_error(float(cpu_derived.get(name, math.nan)), float(gpu_value))
    cpu_record, gpu_record = cpu_summary.get('optimization'), gpu_summary.get('optimization')
    if gpu_record:
        for key in ('objective_initial', 'objective_final'):
            errors[f'optimization {key}'] = relative_error(float((cpu_record or {}).get(key, math.nan)),
                                                      float(gpu_record[key]))
        same = (cpu_record or {}).get('iterations') == gpu_record.get('iterations')
        errors['optimization iterations'] = 0.0 if same else math.inf
    return errors


def _pair_errors(spec, cpu_results, gpu_results):
    """{item: relative difference} of one model on both backends: every probe phasor relative to the largest probe of
    its channel (floored by _CHANNEL_FLOOR x the run's largest), the live window fields relative to their largest
    value, the objective / derived / optimization values, and convergence of the cpu run."""
    channels = _features(spec, 'field_probe')
    cpu = {name: probe_phasor(cpu_results, _PHYSICS, name) for name in channels}
    gpu = {name: probe_phasor(gpu_results, _PHYSICS, name) for name in channels}
    run_scale = max((abs(value) for value in gpu.values()), default=0.0)
    scales = {}
    for name, channel in channels.items():
        scales[channel] = max(scales.get(channel, 0.0), abs(gpu[name]))
    errors = {}
    for name, channel in channels.items():
        scale = max(scales[channel], _CHANNEL_FLOOR * run_scale)
        difference = abs(cpu[name] - gpu[name])
        errors[f'probe {name}'] = difference / scale if scale > 0.0 else difference
    fields = {name: (np.asarray(cpu_results.field(f'{_PHYSICS}.{name}').values, dtype=np.float64),
                     np.asarray(gpu_results.field(f'{_PHYSICS}.{name}').values, dtype=np.float64))
              for name in _LIVE_FIELDS}
    field_scale = max(float(np.max(np.abs(gpu_values))) for _, gpu_values in fields.values())
    for name, (cpu_values, gpu_values) in fields.items():
        if cpu_values.shape != gpu_values.shape:
            errors[f'field {name}'] = math.nan
            continue
        difference = float(np.max(np.abs(cpu_values - gpu_values))) if cpu_values.size else 0.0
        errors[f'field {name}'] = difference / field_scale if field_scale > 0.0 else difference
    errors.update(_scalar_errors(spec, cpu_results, gpu_results))
    record = _solve_record(cpu_results)
    errors['cpu not converged'] = 0.0 if record.get('converged', True) else math.inf
    return errors


def _role_backends(results):
    """{role: backend} of the nonlinear physics' role solvers in the summary 'solvers' record."""
    roles = ((results.summary().get('solvers') or {}).get(_PHYSICS) or {}).get('roles') or {}
    return {role: record.get('backend') for role, record in roles.items()}


def _check_backends(results_by_name):
    """Per compared model: max relative cpu - gpu difference (_pair_errors) in units of _TOLERANCE; raises unless every
    cpu role solver ran on the cpu; cpu-only references without the gpu runs."""
    groups = _groups()
    for group, specs in groups.items():
        for key in specs:
            backends = _role_backends(results_by_name[_run_name(group, key, 'cpu')])
            if not backends or any(backend != 'cpu' for backend in backends.values()):
                raise ValueError(f"run '{_run_name(group, key, 'cpu')}': expected every role solver of '{_PHYSICS}' "
                                 f"on the cpu backend, summary 'solvers' records {backends!r}")
    if not any(name.endswith('__gpu') for name in results_by_name):   # no usable GPU: the gpu runs were skipped
        return _check_cpu_only(results_by_name, groups)
    normalized, worst_items = {}, {}
    for group, specs in groups.items():
        for key, spec in specs.items():
            if not _compared(group, key):
                continue
            errors = _pair_errors(spec, results_by_name[_run_name(group, key, 'cpu')],
                                  results_by_name[_run_name(group, key, 'gpu')])
            item, error = worst_item(errors)
            normalized[f'{group} {key}'] = error / _TOLERANCE
            worst_items[f'{group} {key}'] = item
    worst_run, worst = worst_item(normalized)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalized_errors': normalized, 'worst_item_per_run': worst_items, 'worst_run': worst_run,
                        'tolerance': _TOLERANCE}}


def _check_cpu_only(results_by_name, groups):
    """Without the gpu runs: the references of the source self-tests (nlfd_shg_uniform_undepleted, _mixed_polarization,
    _kerr_spm, _surface_sheet, _sheet_nonlinearity, _cascade_adjoint_gradient, _depleted_adjoint_gradient,
    _shg_optimization_run) on the cpu runs, each in its own tolerance units (optimization: failed checks)."""
    def cpu_group(group):
        return {key: results_by_name[_run_name(group, key, 'cpu')] for key in groups[group]}
    checks = {'shg': _check_shg_uniform(cpu_group('shg')), 'mixed': _check_mixed_polarization(cpu_group('mixed')),
              'kerr': _check_spm(cpu_group('kerr')), 'surface': _check_surface(cpu_group('surface')),
              'sheet': _check_sheets(cpu_group('sheet')), 'adjoint': _check_gradient(cpu_group('adjoint')),
              'depleted_adjoint': _check_depleted_gradient(cpu_group('depleted_adjoint'))}
    normalized = {name: check['error'] / check['tolerance'] for name, check in checks.items()}
    normalized['optimization'] = _check_optimization(cpu_group('optimization'))['error']   # tolerance 0: failures
    worst_name, worst = worst_item(normalized)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalized_errors': normalized, 'worst_item': worst_name,
                        'gpu_runs': 'skipped (no usable GPU)'}}


NLFD_BACKEND_SELFTESTS = (
    SelfTestCase(
        name='nlfd_cpu_gpu',
        label='Nonlinear FDFD: CPU backend vs GPU',
        description=('em_fdfd_nonlinear on solver_defaults.backend cpu (numpy channel store, sources, Anderson / GMRES '
                     'vectors, scipy assembly and host SuperLU per channel, sheet deposits by np.add.at) against the '
                     'same models on gpu: nlfd_shg_uniform_undepleted (E_z cascade), nlfd_mixed_polarization (H_z '
                     'pump, C / C* collocation, polarization both), both nlfd_kerr_spm runs (pump_depleted picard with '
                     'the lagged Kerr operator and Anderson mixing; newton with the exact tangents and real GMRES), '
                     'nlfd_surface_sheet (one run, selvedges inside and outside: surface sites, hz face deposit), the '
                     'nlfd_sheet_nonlinearity runs (D3h line sheets ez / hz, graphene sigma3), the gradient runs of '
                     'nlfd_cascade_adjoint_gradient (ez, hz, mixed) and nlfd_depleted_adjoint_gradient (picard SHG, '
                     'newton Kerr: the fixed-point adjoint GMRES) and the nlfd_shg_optimization_run (design region, '
                     '2 L-BFGS-B iterations). Per model the largest difference of every field_probe phasor relative '
                     f'to the largest probe of its channel (floored at {_CHANNEL_FLOOR:g} of the largest probe of the '
                     'run), of the live window fields (e_abs, ez_re / _im, hz_re / _im of the shown channel) relative '
                     'to their largest value, and of the adjoint objective values, derived values (the block '
                     'eps_sensitivity integral) and initial / final optimization objective (equal iteration counts), '
                     f'in units of {_TOLERANCE:g}; an unconverged cpu run counts as infinite; measured = error = the '
                     'worst model, tolerance 1. The check raises unless every cpu role solver records backend cpu. '
                     'The finite-difference neighbour runs of the gradient cases run on the cpu only: without a '
                     'usable GPU the gpu runs are skipped and the cpu runs are checked against the references of '
                     'those eight self-tests instead (each in its own tolerance units, the optimization by its '
                     'failed checks). Not yet confirmed by a run.'),
        specs=_backend_runs, check=_check_backends, heavy=True),
)

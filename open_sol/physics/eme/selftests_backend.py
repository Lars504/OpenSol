"""CPU vs GPU backend case of em_eme / em_eme_3d (solver_plan S6): the lossy-film closed forms and the 3D PEC CPU
cascade on the cpu backend, a radiating 2D step and the 3D PEC step on both backends; no physics imports."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from .selftests import _PHYSICS, _check_lossy_film, _eme_step_spec, _film_model_spec, _on_backend
from .selftests3d import _check_lossy_film_3d, _check_pec_model_3d, _pec_step_spec
from .selftests3d import _film_model_spec as _film_model_spec_3d

_LOSSY_2D_RUNS = {'ez': 'lossy_ez_cpu', 'hz': 'lossy_hz_cpu'}
_LOSSY_3D_RUN = 'lossy_3d_cpu'
_STEP_RUNS = {'cpu': 'step_cpml_cpu', 'gpu': 'step_cpml_gpu'}
_PEC_RUNS = {'cpu': 'pec_3d_cpu', 'gpu': 'pec_3d_gpu'}
_BACKEND_TOLERANCE = 1e-7   # cpu vs gpu: the same host solves on equal float32 materials (ARPACK start ~1e-10)
_FIELD_TOLERANCE = 1e-6     # float32 output fields: an ulp flip is ~6e-8 of the peak


def _backend_runs():
    runs = {run: _on_backend(_film_model_spec(polarization, lossy=True), 'cpu')
            for polarization, run in _LOSSY_2D_RUNS.items()}
    runs[_LOSSY_3D_RUN] = _on_backend(_film_model_spec_3d(), 'cpu')
    runs.update({run: _on_backend(_eme_step_spec('cpml'), backend) for backend, run in _STEP_RUNS.items()})
    runs.update({run: _on_backend(_pec_step_spec(), backend) for backend, run in _PEC_RUNS.items()})
    return runs


def _scaled(check):
    """A check result's error in its own tolerance units."""
    return float(check['error']) / float(check['tolerance'])


def _s_abs(results):
    ports = results.result(_PHYSICS, 'ports')
    return np.abs(np.asarray(ports['s_matrix_real']) + 1j * np.asarray(ports['s_matrix_imag']))


def _field_error(results_by_name, runs, name):
    """max |cpu - gpu| / max |gpu| of a window field (3D: the mid-height z cut)."""
    values = {backend: np.asarray(results_by_name[run].field(f'{_PHYSICS}.{name}').values, dtype=np.float64)
              for backend, run in runs.items()}
    if values['cpu'].shape != values['gpu'].shape:
        return math.inf
    return float(np.max(np.abs(values['cpu'] - values['gpu']))) / float(np.max(np.abs(values['gpu'])))


def _check_backends(results_by_name):
    """Closed forms of the cpu runs (2D / 3D lossy film, 3D PEC CPU cascade) in their tolerance units; with the gpu
    runs |S| of the ports and T1 / T3 cpu vs gpu over _BACKEND_TOLERANCE, e_abs over _FIELD_TOLERANCE; error = the
    worst."""
    lossy_2d = _check_lossy_film({polarization: results_by_name[run] for polarization, run in _LOSSY_2D_RUNS.items()})
    lossy_3d = _check_lossy_film_3d({'run': results_by_name[_LOSSY_3D_RUN]})
    pec_3d = _check_pec_model_3d({'run': results_by_name[_PEC_RUNS['cpu']]})
    errors = {'lossy_2d_closed_form': _scaled(lossy_2d), 'lossy_3d_closed_form': _scaled(lossy_3d),
              'pec_3d_cpu_cascade': _scaled(pec_3d)}
    details = {'gpu_runs': 'compared'}
    if _STEP_RUNS['gpu'] in results_by_name and _PEC_RUNS['gpu'] in results_by_name:
        comparisons = {}
        for label, runs in (('step_2d', _STEP_RUNS), ('pec_3d', _PEC_RUNS)):
            s_abs = {backend: _s_abs(results_by_name[run]) for backend, run in runs.items()}
            comparisons[f'{label}_s_abs'] = (float(np.max(np.abs(s_abs['cpu'] - s_abs['gpu'])))
                                             if s_abs['cpu'].shape == s_abs['gpu'].shape else math.inf)
            comparisons[f'{label}_e_abs'] = _field_error(results_by_name, runs, 'e_abs')
        derived = {backend: results_by_name[run].summary()['derived'] for backend, run in _STEP_RUNS.items()}
        for name in ('T1', 'T3'):
            comparisons[f'step_2d_{name}'] = abs(float(derived['cpu'][name]) - float(derived['gpu'][name]))
        details['cpu_vs_gpu'] = comparisons
        errors.update({name: value / (_FIELD_TOLERANCE if name.endswith('_e_abs') else _BACKEND_TOLERANCE)
                       for name, value in comparisons.items()})
    else:   # no usable GPU: api.selftest skipped the gpu runs
        details['gpu_runs'] = 'skipped (no usable GPU)'
    finite = all(np.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
    error = max(errors.values()) if finite else float('nan')
    details['errors_in_tolerance_units'] = errors
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0, 'details': details}


EME_BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='eme_cpu_gpu',
        label='EME 2D / 3D: CPU backend vs GPU',
        description=('em_eme and em_eme_3d on solver_defaults.backend cpu (numpy fields, q_abs and masks; the line / '
                     'cross-section solves and cascades are host code on both backends). On the cpu: the lossy-film '
                     'runs of eme_lossy_film_absorption (ez, hz) and eme_3d_lossy_film_model against their checks '
                     '(transfer matrices, section lengths, scalars, 1 - R - T), and the em_eme_3d PEC step (the '
                     'pec_cells variant of eme_3d_field_reduces_to_slab, lossless, 2 PEC cell columns at each x end '
                     'drawn with pec_domain, 40 modes) against the CPU 3D cascade of the same cross-sections (|S| of '
                     'the port block and input-face n_eff within 1e-7). On both '
                     'backends: the radiating cpml slab step of eme_radiating_step_pml_modes (em_eme, n_modes 0) and '
                     'the 3D PEC step: max | |S|_cpu - |S|_gpu | of the reported port block and T1 / T3 over '
                     f'{_BACKEND_TOLERANCE:g}, the max-normalised e_abs difference (float32 fields) over '
                     f'{_FIELD_TOLERANCE:g}. error = the worst item in '
                     'its own tolerance units, tolerance 1. Without a usable GPU the gpu runs are skipped and only the '
                     'cpu closed forms are checked.'),
        specs=_backend_runs, check=_check_backends),
)

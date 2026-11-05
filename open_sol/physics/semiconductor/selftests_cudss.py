"""cuDSS 'linear' back ends of drift_diffusion against host SuperLU: a forward-biased abrupt pn junction strip and the
silicon_pn_photodiode template (reverse bias + generation), each solved by a sparse_direct node pinned to scipy_splu (the
reference), one per cuDSS back end and the Automatic solver (on the gpu with nvmath-python: the role's cuDSS back end,
B1a). Registered by definition.register; passes as skipped without nvmath-python."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role
from ...model.spec import SolverSpec
from . import selftests as semiconductor_selftests

_BACK_ENDS = ('cudss', 'cudss_mixed')
_REFERENCE = 'scipy_splu'           # pinned: the Automatic solver resolves to cuDSS on the gpu
_AUTOMATIC = 'automatic'            # run label of the Automatic sparse_direct (no node)
_NODE_PREFIX = 'sparse_direct_'
_FORWARD_BIAS = 0.4                 # V on the anode (p side): 4 continuation steps of the default max_bias_step 0.1
_COMPARED_FIELDS = ('V', 'N', 'P')
_CONTACTS = ('anode', 'cathode')
# relative to max |SuperLU field| / |SuperLU current| (first estimate: Newton converges to 1e-8 in max |du|, |dn| / n)
_TOLERANCE = 1e-6
_EQUILIBRIUM_AND_COUPLED_PLANS = 2  # one cuDSS analysis for the Poisson cold start, one for every coupled Newton step


def _forward_junction_spec():
    """The semiconductor_cpu_gpu pn junction with the anode at _FORWARD_BIAS."""
    features = semiconductor_selftests._junction_features()
    next(feature for feature in features if feature.name == 'anode').params['voltage'] = _FORWARD_BIAS
    return semiconductor_selftests._strip_model('Self-test: forward-biased abrupt pn junction (drift-diffusion)',
                                                semiconductor_selftests._JUNCTION_SIZE,
                                                semiconductor_selftests._JUNCTION_CELL,
                                                semiconductor_selftests._junction_geometry(), features)


_MODELS = {'forward_junction': _forward_junction_spec,
           'photodiode': semiconductor_selftests.silicon_pn_photodiode_spec}


def _cudss_specs():
    """'<model>_scipy_splu' (reference), '<model>_<back end>' and '<model>_automatic' runs; {} without nvmath-python."""
    from ...solvers.cudss_factor import cudss_available
    if not cudss_available():
        return {}
    specs = {}
    for model, factory in _MODELS.items():
        for back_end in (_REFERENCE, *_BACK_ENDS):
            spec = factory()
            node = _NODE_PREFIX + back_end
            spec.solvers.append(SolverSpec(node, 'sparse_direct', {'back_end': back_end}))
            spec.physics_by_name(semiconductor_selftests._PHYSICS).solvers['linear'] = node
            specs[f'{model}_{back_end}'] = spec
        specs[f'{model}_{_AUTOMATIC}'] = factory()
    return specs


def _linear_counters(results):
    """Counters of the summary 'solvers' linear role record of the semiconductor physics ({} when missing)."""
    return solver_role(results, semiconductor_selftests._PHYSICS).get('counters') or {}


def _deviations(results, reference):
    """{name: relative deviation} of the V, N, P windows and the contact currents of results from reference."""
    deviations = {}
    for field in _COMPARED_FIELDS:
        values, _ = semiconductor_selftests._window(results, field)
        expected, _ = semiconductor_selftests._window(reference, field)
        deviations[field] = (float(np.max(np.abs(values - expected))) / float(np.max(np.abs(expected)))
                             if values.shape == expected.shape else math.inf)
    for contact in _CONTACTS:
        current = semiconductor_selftests._table(results, contact)['current']
        expected = semiconductor_selftests._table(reference, contact)['current']
        deviations[f'I_{contact}'] = abs(current - expected) / abs(expected)
    return deviations


def _check_cudss(results_by_name):
    """Per model and cuDSS back end (and the Automatic run): field / current deviations from SuperLU in tolerance units;
    the linear record must show that back end (Automatic: a cuDSS one of an 'auto' runtime, auto_fallbacks 0) with more
    factorisations than plans and 2 plans (+ one per double fallback of cudss_mixed)."""
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS
    if not results_by_name:
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': 1.0,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    normalized_errors, details = {}, {}
    for model in _MODELS:
        reference = results_by_name[f'{model}_{_REFERENCE}']
        for back_end in (*_BACK_ENDS, _AUTOMATIC):
            results = results_by_name[f'{model}_{back_end}']
            deviations = _deviations(results, reference)
            for name, deviation in deviations.items():
                normalized_errors[f'{model} {back_end} {name}'] = deviation / _TOLERANCE
            counters = _linear_counters(results)
            plans = _EQUILIBRIUM_AND_COUPLED_PLANS + int(counters.get('cudss_double_fallbacks') or 0)
            if back_end == _AUTOMATIC:
                back_end_ok = counters.get('back_end') in CUDSS_BACK_ENDS and counters.get('auto_fallbacks') == 0
            else:
                back_end_ok = counters.get('back_end') == back_end
            record_ok = (back_end_ok and counters.get('cudss_plans') == plans
                         and int(counters.get('factorizations') or 0) > plans)
            normalized_errors[f'{model} {back_end} record'] = 0.0 if record_ok else math.inf
            details[f'{model}_{back_end}'] = {'deviations': deviations, 'counters': dict(counters),
                                              'superlu_factor_seconds': _linear_counters(reference).get(
                                                  'factor_seconds')}
    worst_item = max(normalized_errors, key=lambda item: math.inf if math.isnan(normalized_errors[item])
                     else normalized_errors[item])
    worst = normalized_errors[worst_item]
    details.update(normalized_errors=normalized_errors, worst_item=worst_item)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


CUDSS_SELFTEST_CASES = (
    SelfTestCase(
        name='semiconductor_cudss',
        label="Drift-diffusion cuDSS 'linear' back ends vs SuperLU",
        description=('drift_diffusion solved by a sparse_direct node pinned to scipy_splu (host SuperLU, complex128: '
                     "the reference), by one with back_end cudss and one with cudss_mixed as the 'linear' solver, and by "
                     "the Automatic sparse_direct (on the gpu the role's cuDSS back end; its record must show a cuDSS "
                     "back end of an 'auto' runtime with auto_fallbacks 0) (8 runs, gpu): "
                     f'the semiconductor_cpu_gpu pn junction strip forward-biased to {_FORWARD_BIAS:g} V (1000 x 2 '
                     'cells) and the silicon_pn_photodiode template (400 x 20 cells, 1 V reverse, generation). '
                     'cuDSS factorises the L1-equilibrated Newton Jacobian as a real LU and refactorises the last '
                     'factor of the same size in place (plan reused). Per model and back end max |cuDSS - SuperLU| '
                     '/ max |SuperLU| of V, N, P and the relative contact-current deviations must stay below '
                     f'{_TOLERANCE:g} (first estimate), and the linear record must show that back end with 2 cuDSS '
                     'plans (equilibrium Poisson + coupled; + 1 per cudss_mixed double fallback) and more '
                     'factorisations than plans. measured = error = the worst item in tolerance units, tolerance 1; '
                     'passes as skipped without nvmath-python. Passed on an RTX 3050 (deviations not yet '
                     'recorded here).'),
        specs=_cudss_specs, check=_check_cudss),
)

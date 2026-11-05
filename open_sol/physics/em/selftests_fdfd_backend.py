"""CPU vs GPU backend case of em_fdfd_ez / em_fdfd_hz (solver_plan S6 stage 2): the FDFD self-test models (sources,
TFSF, poles, mode ports, flux / far-field monitors, walls, PEC, eigenfrequency, the Joule / thermo-optic heat chain) on
solver_defaults.backend 'cpu' (numpy / scipy assembly, host SuperLU) against the same models on 'gpu'; no physics
imports. Without a usable GPU api.selftest skips the gpu runs and the cpu runs are checked against their closed forms."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import worst_item
from ...model.spec import MeshSpec
from .selftests import wall_run_name
from .selftests_anisotropy import _slab_spec as _rotated_slab_spec
from .selftests_conducting_sheet import _sheet_spec
from .selftests_eigenfrequency import _check_cavity, _cavity_specs
from .selftests_fdfd import (_PHYSICS_NAME, _check_far_field_power, _check_mode_port_slab, _check_wall_reflection,
                             _drude_spec, _far_field_specs, _fresnel_specs, _graded_fresnel_specs, _slab_specs,
                             _wall_specs, fd_joule_heating_spec)
from .selftests_symmetry import _mirror_spec

_BACKENDS = ('cpu', 'gpu')
_DRUDE_WAVELENGTH = 0.9e-6   # m: one wavelength of em_fdfd_mie_drude_cylinder per polarisation (TFSF + Drude pole)
_JOULE_RUN = 'joule'
# a coarsened copy of the fd_joule_heating template (8 nm instead of 4 nm: ~24k instead of 78k FDFD unknowns; parity
# needs no resolution, the cpu-only fallback checks finite fields only)
_JOULE_CELL = 8e-9
# Fresnel: the glass strips only (vacuum and all-glass run the same code path; the media are data)
_FRESNEL_KEYS = ('ez_glass', 'hz_glass')
# walls: one pin rule per kind (ez pec / antisymmetric, hz pmc / symmetric, the ez pec_domain block)
_WALL_RUNS = (('ez', 'pec'), ('ez', 'antisymmetric'), ('hz', 'pmc'), ('hz', 'symmetric'), ('ez', 'pec_domain'))
# features the analytic cases run on the cpu only (selftests_fdfd cpu_specs), one small model each: a graded Fresnel
# strip, the rotated eps_xy slab, the conducting sheets (ez constant, hz graphene) and one mirror half
_FINITE_ONLY_GROUPS = ('drude', _JOULE_RUN, 'fresnel', 'graded', 'eps_xy', 'sheet', 'mirror')
# result-bearing feature types compared between the backends (mode_source tables hold text columns)
_RESULT_FEATURE_TYPES = frozenset(('field_probe', 'flux_monitor', 'mode_monitor', 'far_field', 'resonance_table'))
_TOLERANCE = 1e-5         # relative: both backends assemble in complex128; host SuperLU on the cpu, Automatic on the
                          # gpu (cuDSS double with nvmath-python: ~1e-10, well inside)
_JOULE_TOLERANCE = 1e-3   # the coupled fixed point stops at its own tolerance (cpu heat: SuperLU, gpu heat: pcg)


def _groups():
    """{group: {key: spec}} of the compared models (spec builders of the FDFD self-test modules)."""
    joule = fd_joule_heating_spec()
    joule.mesh = MeshSpec(mode='uniform', dx=_JOULE_CELL)
    wall_names = {wall_run_name(*run) for run in _WALL_RUNS}
    return {
        'far_field': _far_field_specs(),
        'fresnel': {key: spec for key, spec in _fresnel_specs().items() if key in _FRESNEL_KEYS},
        'slab': _slab_specs(),
        'drude': {polarization: _drude_spec(polarization, _DRUDE_WAVELENGTH) for polarization in ('ez', 'hz')},
        'wall': {name: spec for name, spec in _wall_specs().items() if name in wall_names},
        'cavity': _cavity_specs(),
        _JOULE_RUN: {_JOULE_RUN: joule},
        'graded': {'hz_glass': _graded_fresnel_specs()['hz_glass']},
        'eps_xy': {'x_rotated': _rotated_slab_spec('x', True)},
        'sheet': {f'{polarization}_device': _sheet_spec(polarization, 'device') for polarization in ('ez', 'hz')},
        'mirror': {'ez_odd_half': _mirror_spec('ez_odd', True)},
    }


def _run_name(group, key, backend):
    return f'{group}__{key}__{backend}'


def _backend_runs():
    """Every model of _groups() once per backend (solver_defaults.backend set explicitly)."""
    runs = {}
    for backend in _BACKENDS:
        for group, specs in _groups().items():
            for key, spec in specs.items():
                spec.solver_defaults.backend = backend
                runs[_run_name(group, key, backend)] = spec
    return runs


def _numeric_leaves(value, path=''):
    """{path: float64 array} of the numeric entries of a result dict (nested dicts / lists; text entries skipped)."""
    if isinstance(value, dict):
        leaves = {}
        for key, item in value.items():
            leaves.update(_numeric_leaves(item, f'{path}.{key}' if path else str(key)))
        return leaves
    if isinstance(value, (str, bytes)) or value is None:
        return {}
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return {}
    return {path: array}


def _finite_peak(values):
    finite = np.abs(values[np.isfinite(values)])
    return float(finite.max()) if finite.size else 0.0


def _scaled_difference(cpu_values, gpu_values, scale):
    """max |cpu - gpu| / scale over the entries that differ (equal infinities, e.g. a lossless Q, and both NaN count as
    equal); NaN when a differing entry is not finite."""
    equal = (cpu_values == gpu_values) | (np.isnan(cpu_values) & np.isnan(gpu_values))
    if np.all(equal):
        return 0.0
    cpu_values, gpu_values = cpu_values[~equal], gpu_values[~equal]
    if not (np.all(np.isfinite(cpu_values)) and np.all(np.isfinite(gpu_values))):
        return math.nan
    difference = float(np.max(np.abs(cpu_values - gpu_values)))
    return difference / scale if scale > 0.0 else difference


def _difference(cpu_values, gpu_values, phase):
    """max |cpu - gpu| / max finite |gpu| (_scaled_difference; phase entries: the wrapped angle difference / pi); NaN on
    a shape mismatch."""
    if cpu_values.shape != gpu_values.shape:
        return math.nan
    if not cpu_values.size:
        return 0.0
    if phase:
        return float(np.max(np.abs(np.angle(np.exp(1j * (cpu_values - gpu_values)))))) / math.pi
    return _scaled_difference(cpu_values, gpu_values, _finite_peak(gpu_values))


def _role_backends(results):
    """{role: backend} of the physics 'em' role solvers in the summary 'solvers' record."""
    roles = ((results.summary().get('solvers') or {}).get(_PHYSICS_NAME) or {}).get('roles') or {}
    return {role: record.get('backend') for role, record in roles.items()}


def _pair_errors(spec, cpu_results, gpu_results):
    """{item: relative difference} of one model on both backends: the out-of-plane phasor fields, every compared
    feature result and every derived value."""
    physics = spec.physics[0]
    polarization = physics.type.rsplit('_', 1)[1]
    errors = {}
    for part in ('re', 'im'):
        name = f'{_PHYSICS_NAME}.{polarization}_{part}'
        errors[name] = _difference(np.asarray(cpu_results.field(name).values, dtype=np.float64),
                                   np.asarray(gpu_results.field(name).values, dtype=np.float64), phase=False)
    for feature in physics.features:
        if feature.enabled and feature.type in _RESULT_FEATURE_TYPES:
            cpu_leaves = _numeric_leaves(cpu_results.result(_PHYSICS_NAME, feature.name))
            gpu_leaves = _numeric_leaves(gpu_results.result(_PHYSICS_NAME, feature.name))
            for path, gpu_values in gpu_leaves.items():
                cpu_values = cpu_leaves.get(path)
                errors[f'{feature.name}.{path}'] = (math.nan if cpu_values is None else
                                                    _difference(cpu_values, gpu_values, 'phase' in path))
    cpu_derived, gpu_derived = cpu_results.summary().get('derived') or {}, gpu_results.summary().get('derived') or {}
    for name, gpu_value in gpu_derived.items():
        errors[f'derived.{name}'] = _difference(np.asarray(float(cpu_derived.get(name, math.nan))),
                                                np.asarray(float(gpu_value)), phase=False)
    return errors


def _check_backends(results_by_name):
    """Per model: max relative cpu - gpu difference of the fields, results and derived values (tolerance _TOLERANCE,
    the Joule chain _JOULE_TOLERANCE), in tolerance units; raises unless every cpu role solver ran on the cpu."""
    groups = _groups()
    for group, specs in groups.items():
        for key in specs:
            backends = _role_backends(results_by_name[_run_name(group, key, 'cpu')])
            if not backends or any(backend != 'cpu' for backend in backends.values()):
                raise ValueError(f"run '{_run_name(group, key, 'cpu')}': expected every role solver of "
                                 f"'{_PHYSICS_NAME}' on the cpu backend, summary 'solvers' records {backends!r}")
    if not any(name.endswith('__gpu') for name in results_by_name):   # no usable GPU: the gpu runs were skipped
        return _check_cpu_only(results_by_name, groups)
    normalized, worst_items = {}, {}
    for group, specs in groups.items():
        tolerance = _JOULE_TOLERANCE if group == _JOULE_RUN else _TOLERANCE
        for key, spec in specs.items():
            errors = _pair_errors(spec, results_by_name[_run_name(group, key, 'cpu')],
                                  results_by_name[_run_name(group, key, 'gpu')])
            item, error = worst_item(errors)
            normalized[f'{group} {key}'] = error / tolerance
            worst_items[f'{group} {key}'] = item
    worst_run, worst = worst_item(normalized)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalized_errors': normalized, 'worst_item_per_run': worst_items, 'worst_run': worst_run,
                        'tolerance': _TOLERANCE, 'joule_tolerance': _JOULE_TOLERANCE}}


def _cpu_group(results_by_name, group):
    """{key: cpu Results} of one group."""
    return {key: results_by_name[_run_name(group, key, 'cpu')] for key in _groups()[group]}


def _check_cpu_only(results_by_name, groups):
    """Without the gpu runs: the closed-form checks of the far-field, mode-port, wall and cavity models on the cpu (each
    in its own tolerance units) and finite fields of the other runs (their cases run on the cpu anyway)."""
    checks = {'far_field': _check_far_field_power(_cpu_group(results_by_name, 'far_field')),
              'slab': _check_mode_port_slab(_cpu_group(results_by_name, 'slab')),
              'wall': _check_wall_reflection(_cpu_group(results_by_name, 'wall'), _WALL_RUNS),
              'cavity': _check_cavity(_cpu_group(results_by_name, 'cavity'))}
    normalized = {name: check['error'] / check['tolerance'] for name, check in checks.items()}
    for group in _FINITE_ONLY_GROUPS:
        for key, spec in groups[group].items():
            polarization = spec.physics[0].type.rsplit('_', 1)[1]
            field = np.asarray(results_by_name[_run_name(group, key, 'cpu')].field(
                f'{_PHYSICS_NAME}.{polarization}_re').values, dtype=np.float64)
            normalized[f'{group} {key} finite'] = 0.0 if np.all(np.isfinite(field)) else math.nan
    worst_name, worst = worst_item(normalized)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalized_errors': normalized, 'worst_item': worst_name,
                        'gpu_runs': 'skipped (no usable GPU)'}}


FDFD_BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_cpu_gpu',
        label='EM FDFD: CPU backend vs GPU',
        description=('em_fdfd_ez / em_fdfd_hz on solver_defaults.backend cpu (numpy arrays, scipy CSR assembly, host '
                     'SuperLU) against the same models on gpu: the far-field boxes of em_fdfd_line_current_hankel '
                     '(filament current, hz electric dipole, far field, flux box), the two glass strips of '
                     'em_fdfd_fresnel_transmission (periodic line sources), the em_fdfd_mode_port_slab guides '
                     '(frequency_domain at its two wavelengths, mode source and monitors), '
                     f'the em_fdfd_mie_drude_cylinder runs at {_DRUDE_WAVELENGTH * 1e9:g} nm (TFSF, Drude pole, '
                     'q_abs integral), five em_fdfd_pec_reflection_phase strips (ez pec / antisymmetric, hz pmc / '
                     'symmetric, ez pec_domain), the em_fdfd_eigenfrequency_pec_cavity runs (eigenfrequency study), the '
                     f'fd_joule_heating template coarsened to {_JOULE_CELL * 1e9:g} nm (em_heat_joule + '
                     'heat_em_thermo_optic stationary fixed point) and, since those analytic cases run on the cpu '
                     'only, one small model per feature they alone cover: the graded hz glass strip of '
                     'em_fdfd_fresnel_transmission_graded, the rotated eps_xy slab of em_fdfd_hz_rotated_slab, the two '
                     'em_fdfd_conducting_sheet device strips and the ez odd half of em_fdfd_mirror_unfold (x_lo '
                     'antisymmetric mirror). Not yet confirmed by a run with this set. Per model the largest '
                     'relative difference of the out-of-plane phasor fields (Re, Im), of every probe / flux / mode '
                     'monitor / far-field / resonance-table result entry (phases: wrapped difference / pi) and of '
                     f'the derived values, in units of {_TOLERANCE:g} ({_JOULE_TOLERANCE:g} for the Joule chain, '
                     'whose fixed point and heat solver differ between the backends); measured = error = the worst '
                     'model, tolerance 1. The check raises unless every cpu role solver records backend cpu. Without '
                     'a usable GPU the gpu runs are skipped and the far-field, slab, wall and cavity cpu runs are checked '
                     'against the closed forms of their own self-tests instead (each in its own tolerance units; the '
                     'other runs: finite fields, their own cases run on the cpu anyway).'),
        specs=_backend_runs, check=_check_backends, heavy=True),
)

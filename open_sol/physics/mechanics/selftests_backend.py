"""Backend cases of solid_mechanics and solid_mechanics_3d (solver_plan S6 / S8 E): existing exact cases run on
solver_defaults.backend 'cpu' (numpy / scipy: CSR + block-Jacobi twin in 2D, kernels3d numpy twins in 3D; the modal,
piezo and optomechanics solves are host either way), graded by each wrapped case's own closed-form check. The 2D / 3D
mechanics cases are cpu-only (their standalone runs are the gpu half, and two backends that both pass a 1e-6 .. 1e-5
check agree within the parity tolerance); the optics chains also run on 'gpu', whose named output fields must match
(the 1e-5 parity is stricter than their own checks). Registered by definition.register. Without a usable GPU
api.selftest skips the gpu runs and only the cpu checks count."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role
from .selftests import SELFTEST_CASES, STRAIN_OPTIC_CASES
from .selftests_3d import SELFTEST_CASES_3D
from .selftests_brillouin import BRILLOUIN_CHAIN_CASES
from .selftests_eigenfrequency import EIGENFREQUENCY_3D_CASES, EIGENFREQUENCY_CASES
from .selftests_optical_force import OPTICAL_FORCE_CASES
from .selftests_optomechanics import OPTOMECHANICS_CHAIN_CASES
from .selftests_piezo import PIEZO_CASES

BACKENDS = ('cpu', 'gpu')
FIELD_TOLERANCE = 1e-5   # max |f_cpu - f_gpu| / max |gpu fields of its kind|: float32 outputs of 1e-12 float64 solves
_SOLID = 'solid'         # physics instance of every wrapped case
_SEPARATOR = '__'
_FIELDS_2D = ('u_x', 'u_y', 'strain_xx', 'stress_xx', 'von_mises')
_GUIDED_FIELDS = ('u_x', 'u_y', 'u_z', 'von_mises')
_FDFD_EZ_TYPE = 'em_fdfd_ez'
_MODES_TYPE = 'em_modes'


def _case_by_name(cases, name):
    return next(case for case in cases if case.name == name)


def _run_name(backend, case, run):
    return f'{backend}{_SEPARATOR}{case.name}{_SEPARATOR}{run}'


def _linear_backend(results):
    """Backend of the solid's 'linear' role in the run summary's 'solvers' record (None when missing)."""
    return solver_role(results, _SOLID).get('backend')


def _field_family(name):
    """'u', 'strain' or 'stress' (von_mises included): the fields of one kind share a parity scale."""
    return 'stress' if name == 'von_mises' else name.split('_', 1)[0]


def _field_pair(cpu_results, gpu_results, name):
    """(cpu, gpu) float64 values of solid field `name`; None when the run has no such field."""
    try:
        return (np.asarray(cpu_results.field(f'{_SOLID}.{name}').values, dtype=np.float64),
                np.asarray(gpu_results.field(f'{_SOLID}.{name}').values, dtype=np.float64))
    except (KeyError, ValueError):   # a run without the solid (e.g. an empty reference run)
        return None


def _field_differences(cpu_results, gpu_results, field_names):
    """{name: max |cpu - gpu| / the max |gpu| of its family (u / strain / stress)}: an exactly zero field (u_x under
    rollers, stress_xx of a free plate) is graded against its family, not against its own round-off."""
    pairs = {name: pair for name in field_names if (pair := _field_pair(cpu_results, gpu_results, name)) is not None}
    scales = {}
    for name, (cpu, gpu) in pairs.items():
        if cpu.shape == gpu.shape and gpu.size:
            family = _field_family(name)
            scales[family] = max(scales.get(family, 0.0), float(np.max(np.abs(gpu))))
    differences = {}
    for name, (cpu, gpu) in pairs.items():
        if cpu.shape != gpu.shape:
            differences[name] = math.inf
            continue
        scale = scales.get(_field_family(name), 0.0)
        difference = float(np.max(np.abs(cpu - gpu))) if gpu.size else 0.0
        if not (math.isfinite(scale) and math.isfinite(difference)):
            differences[name] = math.nan
        else:
            differences[name] = difference / scale if scale > 0.0 else (0.0 if difference == 0.0 else math.inf)
    return differences


def solid_parity_case(name, label, wrapped, heavy=True, max_wall_time=1800.0, field_tolerance=FIELD_TOLERANCE,
                      backends=BACKENDS):
    """SelfTestCase running every wrapped (case, solid field names) on `backends` (cpu, optionally gpu): error = the
    worst of (cpu check error / its tolerance) and (cpu vs gpu field difference / field_tolerance), tolerance 1; raises
    when a cpu run's solid 'linear' role did not record backend cpu. Field names are unused without 'gpu'."""
    wrapped = tuple(wrapped)
    with_gpu = 'gpu' in backends

    def specs():
        runs = {}
        for backend in backends:
            for case, _ in wrapped:
                for run, spec in case.specs().items():
                    spec.solver_defaults.backend = backend
                    runs[_run_name(backend, case, run)] = spec
        return runs

    def check(results_by_name):
        errors, details = {}, {}
        for case, field_names in wrapped:
            runs = {backend: {} for backend in BACKENDS}
            for run_name, results in results_by_name.items():
                backend, case_name, run = run_name.split(_SEPARATOR, 2)
                if case_name == case.name:
                    runs[backend][run] = results
            for run, results in runs['cpu'].items():
                solid_backend = _linear_backend(results)
                if solid_backend not in (None, 'cpu') or (solid_backend is None and _has_solid(results)):
                    raise ValueError(f"case '{case.name}' run '{run}': expected the solid's linear solver on the cpu, "
                                     f"summary 'solvers' records {solid_backend!r}")
            cpu = case.check(runs['cpu'])
            errors[f'{case.name}.cpu_check'] = float(cpu['error']) / float(cpu['tolerance'])
            entry = {'cpu_measured': cpu['measured'], 'expected': cpu['expected'], 'cpu_error': cpu['error'],
                     'tolerance': cpu['tolerance']}
            if runs['gpu']:
                gpu = case.check(runs['gpu'])
                differences = {}
                for run, cpu_results in runs['cpu'].items():
                    for field_name, difference in _field_differences(cpu_results, runs['gpu'][run],
                                                                     field_names).items():
                        differences[f'{run}.{field_name}'] = difference
                if not differences:
                    raise ValueError(f"case '{case.name}': none of the solid fields {field_names} exists in its runs")
                worst = max(differences.values()) if all(math.isfinite(v) for v in differences.values()) else math.nan
                errors[f'{case.name}.fields'] = worst / field_tolerance
                entry.update(gpu_measured=gpu['measured'], gpu_error=gpu['error'], field_differences=differences)
            elif with_gpu:
                entry['gpu_runs'] = 'skipped (no usable GPU)'
            details[case.name] = entry
        finite = all(math.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
        error = max(errors.values()) if finite else math.nan
        worst_key = max(errors, key=lambda key: math.inf if not math.isfinite(errors[key]) else errors[key])
        return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
                'details': {'worst': worst_key, 'normalised_errors': errors, 'cases': details,
                            'field_tolerance': field_tolerance}}

    case_names = ', '.join(case.name for case, _ in wrapped)
    backend_text = (
        'and on gpu. Each case\'s own check grades the cpu runs, and its named solid fields must match the gpu runs\' '
        f'within a relative {field_tolerance:g} (max difference over the max of the gpu fields of its kind: '
        'displacement, strain or stress incl. von_mises, so an exactly zero field such as u_x under rollers is not '
        'graded by its own round-off; float32 outputs of float64 solves). error = the worst of cpu error / case tolerance and field difference / field tolerance, '
        'tolerance 1. Without a usable GPU only the cpu checks count.' if with_gpu else
        'only: each case\'s own closed-form check grades the cpu runs (its standalone runs are the gpu half: two '
        'backends that both pass it agree within twice its tolerance). error = the worst cpu error / case tolerance, '
        'tolerance 1.')
    return SelfTestCase(
        name=name, label=label,
        description=(f'The exact solid mechanics cases {case_names} run on solver_defaults.backend cpu (numpy arrays, '
                     'scipy CSR / labels, numpy twins of the block-Jacobi and kernels3d kernels; the pcg linear solver '
                     f'on numpy) {backend_text}'),
        specs=specs, check=check, heavy=heavy, max_wall_time=max_wall_time)


def _has_solid(results):
    """True when the run's summary lists a solid physics (its linear role must then be recorded)."""
    return _SOLID in (results.summary().get('solvers') or {})


# numba vs numpy 3D stiffness product twins (kernels3d elastic_product): random Lame coefficients with void elements,
# graded widths, free-dof masks on and off, float32 and float64 solves; ms per call of both on the speed window
_NUMBA_ELEMENTS = (5, 4, 3)
_NUMBA_SPEED_ELEMENTS = (32, 32, 32)
_NUMBA_SPEED_CALLS = (1, 5)   # (numpy, numba) timed calls: the timings are details only
_NUMBA_TOLERANCE = 1e-5   # relative to the peak of K u: float32 round-off of a different summation (Gauss vs closed form)


def _random_product_arguments(rng, element_shape, dtype, masked):
    """Args of one elastic_product call: void elements ~20 %, widths 0.5 .. 1.5 um, fixed dofs ~10 %."""
    ex, ey, ez = element_shape
    n_dofs = 3 * (ex + 1) * (ey + 1) * (ez + 1)
    solid = rng.random(element_shape) > 0.2
    lame_lambda = (rng.uniform(5e10, 1.5e11, element_shape) * solid).astype(dtype)
    mu = (rng.uniform(3e10, 8e10, element_shape) * solid).astype(dtype)
    widths = tuple((rng.uniform(0.5, 1.5, n) * 1e-6).astype(dtype) for n in element_shape)
    u = (rng.standard_normal(n_dofs) * 1e-9).astype(dtype)
    free_dofs = rng.random(n_dofs) > 0.1
    return (u, np.zeros(n_dofs, dtype=dtype), lame_lambda, mu, *widths, free_dofs, np.int32(1 if masked else 0),
            *(np.int32(n) for n in element_shape))


def _check_product_numba(results_by_name):
    """numba vs numpy elastic_product on random windows (float32 / float64, masked or not); ms per call on the speed
    window."""
    from ...core.kernels import numba_available
    from ...model.selftest_helpers import NUMBA_MISSING_DETAILS, numba_twin_result
    from . import kernels3d
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _NUMBA_TOLERANCE,
                'details': dict(NUMBA_MISSING_DETAILS)}
    import numba
    numpy_twin, numba_twin = kernels3d._elastic_product_host, kernels3d._elastic_product_numba(numba)
    rng = np.random.default_rng(20261018)
    comparisons = {f'{np.dtype(dtype).name}:{"masked" if masked else "full"}': (
        numpy_twin, numba_twin, _random_product_arguments(rng, _NUMBA_ELEMENTS, dtype, masked))
        for dtype in (np.float32, np.float64) for masked in (False, True)}
    timings = {np.dtype(dtype).name: (numpy_twin, numba_twin,
                                      _random_product_arguments(rng, _NUMBA_SPEED_ELEMENTS, dtype, True),
                                      *_NUMBA_SPEED_CALLS)
               for dtype in (np.float32, np.float64)}
    result = numba_twin_result(comparisons, timings, _NUMBA_TOLERANCE, numba)
    result['details']['speed_elements'] = list(_NUMBA_SPEED_ELEMENTS)
    result['details']['registered_host_kind'] = kernels3d._TWINS['elastic_product'].host_kind
    return result


_PRODUCT_NUMBA_CASE = SelfTestCase(
    name='solid_3d_product_numba', label='Solid mechanics 3D: numba vs numpy stiffness product (CPU)',
    description=('The numba twin of the solid_mechanics_3d matrix-free stiffness product (mechanics3d.elastic_product: '
                 'the per-node gather of the closed-form brick blocks) against its numpy twin (2-point Gauss element '
                 f'integrals) on random {_NUMBA_ELEMENTS[0]} x {_NUMBA_ELEMENTS[1]} x {_NUMBA_ELEMENTS[2]}-element '
                 'windows with void elements and graded widths, fixed-dof masking off and on, float32 and float64; '
                 'measured = error = the worst max |numba - numpy| relative to the peak of K u, tolerance '
                 f'{_NUMBA_TOLERANCE:g}. Details: ms per call of both on {_NUMBA_SPEED_ELEMENTS[0]}^3 elements '
                 f'({_NUMBA_SPEED_CALLS[0]} numpy / {_NUMBA_SPEED_CALLS[1]} numba calls after one numba warm-up; '
                 'was 48^3 with 3 / 20). Without numba nothing is compared (error 0). Not yet run with the smaller '
                 'timing (heavy flag to re-time).'),
    specs=lambda: {}, check=_check_product_numba, max_wall_time=600.0, heavy=True, backends=('cpu',))


def backend_cases(em_types):
    """The 2D and 3D cpu cases (heavy: re-time them), the numba product case, plus the optics cpu vs gpu chains (strain
    optic, optical force with em_fdfd_ez: wrapped here only, their gpu half is the examples; the em_modes
    optomechanics chains with em_modes)."""
    cases = [   # cpu only; guided modes / piezo modes are host solves on either backend, so they are not wrapped
        solid_parity_case(
            'solid_mechanics_cpu', 'Solid mechanics: CPU backend',
            ((_case_by_name(SELFTEST_CASES, 'solid_bilayer_tension_graded'), ()),   # graded Q1, reactions
             (_case_by_name(SELFTEST_CASES, 'solid_simple_shear'), ()),   # heat chain on the cpu: solid_optics_cpu_gpu
             (_case_by_name(EIGENFREQUENCY_CASES, 'solid_eigenfrequency_roller_box'), ()),   # modal K / M from host stencils
             (_case_by_name(PIEZO_CASES, 'solid_piezo_bar_static'), ())),   # device electric outputs
            backends=('cpu',)),
        solid_parity_case(
            'solid_mechanics_3d_cpu', 'Solid mechanics 3D: CPU backend',
            ((_case_by_name(SELFTEST_CASES_3D, 'solid_3d_uniaxial_block_graded'), ()),
             (_case_by_name(SELFTEST_CASES_3D, 'solid_3d_column_z_expressions'), ()),   # BoxValues loads
             (_case_by_name(SELFTEST_CASES_3D, 'heat_solid_thermal_expansion_3d'), ()),
             (_case_by_name(EIGENFREQUENCY_3D_CASES, 'solid_3d_eigenfrequency_roller_box'), ())),
            backends=('cpu',)),
        _PRODUCT_NUMBA_CASE,
    ]
    optics = []
    if _FDFD_EZ_TYPE in em_types:
        optics += [(_case_by_name(STRAIN_OPTIC_CASES, 'solid_em_strain_optic_plate'), _FIELDS_2D),
                   (_case_by_name(OPTICAL_FORCE_CASES, 'em_solid_optical_force_slab'), ('u_x', 'u_y'))]
    if _MODES_TYPE in em_types:
        optics += [(_case_by_name(OPTOMECHANICS_CHAIN_CASES, 'om_strain_optic_chain'),
                    ('u_x', 'u_y', 'strain_xx', 'strain_yy')),
                   (_case_by_name(BRILLOUIN_CHAIN_CASES, 'om_backward_brillouin_chain'), _GUIDED_FIELDS)]
    if optics:
        cases.append(solid_parity_case('solid_optics_cpu_gpu', 'Solid mechanics optics couplings: CPU backend vs GPU',
                                       optics, max_wall_time=3600.0))
    return tuple(cases)

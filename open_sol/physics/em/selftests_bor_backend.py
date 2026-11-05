"""CPU vs GPU backend case of em_fdfd_bor (solver_plan S8 E): the BOR model runs (cavity eigenfrequency m = 1, the order
list with and without a line_block node, the off-axis dipole order sum) and a pipe-port run (mode source / monitors,
also on a line_block node) on solver_defaults.backend 'cpu' against 'gpu'; no physics imports. Without a usable GPU
the gpu runs are skipped and the cpu runs are checked against their closed forms."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import worst_item
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverSpec, StudySpec)
from .selftests_bor import _PHYSICS, _TYPE, _check_model_cavity, _model_cavity_specs
from .selftests_bor_orders import (_LINE_BLOCK_NODE, _check_model_offaxis, _check_model_order_list,
                                   _model_offaxis_specs, _model_order_list_specs)
from .selftests_fdfd_backend import _finite_peak, _numeric_leaves, _role_backends, _scaled_difference

_BACKENDS = ('cpu', 'gpu')
_TOLERANCE = 1e-6               # relative: both backends assemble and factorise on the host (SuperLU, ARPACK)
_LINE_BLOCK_TOLERANCE = 1e-5    # line_block runs: numpy vs cupy batched block inverses, refined against the operator
_PHASE_MIN_FRACTION = 1e-6      # mode-monitor phases compared only where their power fraction reaches this
_RESULT_FEATURE_TYPES = frozenset(('field_probe', 'flux_monitor', 'mode_monitor', 'far_field', 'resonance_table'))
# resonance_table columns compared (Im omega, Q and the PML share of a lossless cavity are roundoff)
_TABLE_COLUMNS = ('wavelength', 'frequency', 'azimuthal_order')
_DRIVEN_FIELDS = ('e_intensity', 'er', 'ez', 'hphi')   # shown phasor fields of the driven runs (eigen: degeneracies)
# er / ez are compared relative to the peak |E| (sqrt of the e_intensity peak), not to their own peak: a component that
# vanishes by symmetry (E_z of the TE11 pipe launch) is roundoff on both backends and differs by itself between
# numpy and cupy line-block solves while every monitor agrees to 1e-15.
_E_SCALED_FIELDS = ('er', 'ez')

# ---- pipe ports (examples/em/analytic_bor_pipe_te11_section.json, here 2 of its wavelengths): PEC pipe a = 1 um along the axis,
# m = 1 (TE11), cpml z ends, a filling dielectric section eps 2.25 of 1 um; mode source and monitors on cell-centre rows.
# Exact: T = 1 / (1 + F sin^2(2 pi n2 L / lambda)), F = (1 - eps)^2 / (4 n1^2 n2^2), n1,2 = sqrt(1 | eps - (lambda /
# lambda_c)^2), lambda_c = 2 pi a / x'11.
_PORT_CELL = 25e-9
_PORT_RADIUS = 1.0e-6
_PORT_LENGTH = 3.0e-6
_PORT_EPS = 2.25
_PORT_SECTION = (1.0e-6, 2.0e-6)
_PORT_LINES = {'source': 0.4e-6, 'incident': 0.8e-6, 'transmitted': 2.4e-6}   # + half a cell
_PORT_BAND = (1.9e-6, 2.6e-6, 2)   # wavelength start, stop [m], count (parity holds per wavelength)
_PORT_X11 = 1.8411837813           # first zero of J1'
_PORT_T_TOLERANCE = 0.01           # T and R vs the closed form (second-order mesh error, z PML reflection)
_PORT_LAUNCH_TOLERANCE = 5e-3      # launched power vs 1 W


def _ports_specs():
    """{'superlu', 'line_block'}: the pipe-port model, frequency_domain; the second with a line_block (double) node in
    the shifted role (the driven single-order line-block path)."""
    half = _PORT_CELL / 2.0
    settings = {'azimuthal_order': 1, 'bc_x_hi': 'pec', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml', 'pml_cells': 20}
    specs = {}
    for run in ('superlu', 'line_block'):
        line_block = run == 'line_block'
        geometry = [GeometrySpec('section', 'rectangle', {'x0': 0.0, 'y0': _PORT_SECTION[0], 'x1': _PORT_RADIUS,
                                                          'y1': _PORT_SECTION[1]})]
        geometry += [GeometrySpec(f'{name}_line', 'line', {'x0': 0.0, 'y0': z + half, 'x1': _PORT_RADIUS,
                                                           'y1': z + half}) for name, z in _PORT_LINES.items()]
        features = [FeatureSpec('launch', 'mode_source', selection=['source_line'],
                                params={'power': 1.0, 'mode_number': 1, 'direction': 'positive', 'phase': 0.0}),
                    FeatureSpec('incident', 'mode_monitor', selection=['incident_line'], params={'mode_number': 1}),
                    FeatureSpec('transmitted', 'mode_monitor', selection=['transmitted_line'],
                                params={'mode_number': 1})]
        specs[run] = ModelSpec(
            title=f'em_fdfd_bor pipe TE11 ports through a dielectric section ({run})',
            domain=DomainSpec(size_x=_PORT_RADIUS, size_y=_PORT_LENGTH),
            geometry=geometry,
            materials=[MaterialSpec('section_material', properties={'eps_r': _PORT_EPS}, selection=['section'])],
            physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=dict(settings), features=features,
                                 solvers={'shifted': _LINE_BLOCK_NODE} if line_block else {})],
            solvers=[SolverSpec(_LINE_BLOCK_NODE, 'line_block', {'precision': 'double'})] if line_block else [],
            mesh=MeshSpec(mode='uniform', dx=_PORT_CELL),
            study=StudySpec(type='frequency_domain', settings={
                'wavelength_start': _PORT_BAND[0], 'wavelength_stop': _PORT_BAND[1], 'n_wavelengths': _PORT_BAND[2]}))
    return specs


def _port_transmission(wavelength):
    """Closed-form TE11 power transmission through the filling section at `wavelength` [m] (array)."""
    cutoff = 2.0 * math.pi * _PORT_RADIUS / _PORT_X11
    ratio = (np.asarray(wavelength, dtype=np.float64) / cutoff) ** 2
    n_empty, n_section = np.sqrt(1.0 - ratio), np.sqrt(_PORT_EPS - ratio)
    contrast = (1.0 - _PORT_EPS) ** 2 / (4.0 * n_empty ** 2 * n_section ** 2)
    length = _PORT_SECTION[1] - _PORT_SECTION[0]
    return 1.0 / (1.0 + contrast * np.sin(2.0 * math.pi * n_section * length / wavelength) ** 2)


def _check_ports(results_by_name):
    """Per run: transmitted fraction vs T, incident backward fraction vs 1 - T, launched power vs 1 W (normalised)."""
    errors = {}
    for run, results in results_by_name.items():
        transmitted = results.result(_PHYSICS, 'transmitted')
        incident = results.result(_PHYSICS, 'incident')
        expected = _port_transmission(np.asarray(transmitted['x'], dtype=np.float64))
        errors[f'{run} T'] = float(np.max(np.abs(np.asarray(transmitted['series']['fraction_positive'])
                                                 - expected))) / _PORT_T_TOLERANCE
        errors[f'{run} R'] = float(np.max(np.abs(np.asarray(incident['series']['fraction_negative'])
                                                 - (1.0 - expected)))) / (2.0 * _PORT_T_TOLERANCE)
        errors[f'{run} launch'] = float(np.max(np.abs(np.asarray(incident['series']['power_positive'])
                                                      - 1.0))) / _PORT_LAUNCH_TOLERANCE
    item, worst = worst_item(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalised_errors': errors, 'worst_item': item}}


def _offaxis_line_block_spec():
    """The em_bor_offaxis_dipole_model order sum with a line_block (double) node in the shifted role: one batched
    factor of the order family instead of 5 host SuperLU factorisations of ~19,000 unknowns (7 of 42,505 on the
    former 2 x 3 um grid took ~85 s per backend)."""
    spec = _model_offaxis_specs()['offaxis']
    spec.solvers.append(SolverSpec(_LINE_BLOCK_NODE, 'line_block', {'precision': 'double'}))
    spec.physics[0].solvers['shifted'] = _LINE_BLOCK_NODE
    return spec


def _groups():
    """{key: (spec, driven, tolerance)} of the compared models (fresh specs per call)."""
    order_list, ports = _model_order_list_specs(), _ports_specs()
    return {'cavity': (_model_cavity_specs()['cavity'], False, _TOLERANCE),
            'order_list_superlu': (order_list['superlu'], False, _TOLERANCE),
            'order_list_line_block': (order_list['line_block'], False, _LINE_BLOCK_TOLERANCE),
            'offaxis': (_offaxis_line_block_spec(), True, _LINE_BLOCK_TOLERANCE),
            'ports_superlu': (ports['superlu'], True, _TOLERANCE),
            'ports_line_block': (ports['line_block'], True, _LINE_BLOCK_TOLERANCE)}


def _run_name(key, backend):
    return f'{key}__{backend}'


def _backend_runs():
    """Every model of _groups() once per backend (solver_defaults.backend set explicitly)."""
    runs = {}
    for backend in _BACKENDS:
        for key, (spec, _, _) in _groups().items():
            spec.solver_defaults.backend = backend
            runs[_run_name(key, backend)] = spec
    return runs


def _phase_difference(cpu_values, gpu_values, fraction):
    """max wrapped phase difference / pi where the matching power fraction reaches _PHASE_MIN_FRACTION (all entries
    without one)."""
    keep = np.ones(gpu_values.shape, dtype=bool)
    if fraction is not None and fraction.shape == gpu_values.shape:
        keep = np.nan_to_num(fraction) >= _PHASE_MIN_FRACTION
    if not keep.any():
        return 0.0
    return float(np.max(np.abs(np.angle(np.exp(1j * (cpu_values[keep] - gpu_values[keep])))))) / math.pi


def _group(path):
    """Scale group of a leaf: the first word of its last path part ('power_orders' -> 'power', 'n_eff_imag' -> 'n')."""
    return path.rsplit('.', 1)[-1].split('_', 1)[0]


def _compared_leaves(result):
    """{path: float64 array} of a feature result; a table's rows as the _TABLE_COLUMNS columns."""
    leaves = _numeric_leaves(result)
    if result.get('kind') == 'table' and 'rows' in result:
        leaves.pop('rows', None)
        rows = np.asarray(result['rows'], dtype=np.float64)
        for index, column in enumerate(result['columns']):
            if column in _TABLE_COLUMNS:
                leaves[f'rows.{column}'] = rows[:, index] if rows.ndim == 2 else np.zeros(0)
    return leaves


def _pair_errors(spec, cpu_results, gpu_results, driven):
    """{item: difference} of one model on both backends: the shown fields of a driven run (e_intensity / hphi relative
    to their peak, er / ez to the peak |E|), every compared feature result entry (relative to the peak of its scale
    group; phases: wrapped / pi)."""
    errors = {}
    if driven:
        pairs = {}
        for field in _DRIVEN_FIELDS:
            name = f'{_PHYSICS}.{field}'
            pairs[field] = (np.asarray(cpu_results.field(name).values, dtype=np.float64),
                            np.asarray(gpu_results.field(name).values, dtype=np.float64))
        e_peak = math.sqrt(_finite_peak(pairs['e_intensity'][1]))
        for field, (cpu_values, gpu_values) in pairs.items():
            scale = e_peak if field in _E_SCALED_FIELDS else _finite_peak(gpu_values)
            errors[f'{_PHYSICS}.{field}'] = (math.nan if cpu_values.shape != gpu_values.shape
                                             else _scaled_difference(cpu_values, gpu_values, scale))
    for feature in spec.physics[0].features:
        if not (feature.enabled and feature.type in _RESULT_FEATURE_TYPES):
            continue
        cpu_leaves = _compared_leaves(cpu_results.result(_PHYSICS, feature.name))
        gpu_leaves = _compared_leaves(gpu_results.result(_PHYSICS, feature.name))
        scales = {}
        for path, values in gpu_leaves.items():
            scales[_group(path)] = max(scales.get(_group(path), 0.0), _finite_peak(values))
        for path, gpu_values in gpu_leaves.items():
            cpu_values = cpu_leaves.get(path)
            item = f'{feature.name}.{path}'
            if cpu_values is None or cpu_values.shape != gpu_values.shape:
                errors[item] = math.nan
            elif 'phase' in path:
                fraction = gpu_leaves.get(path.replace('phase_', 'fraction_')) if 'phase_' in path else None
                errors[item] = _phase_difference(cpu_values, gpu_values, fraction)
            else:
                errors[item] = _scaled_difference(cpu_values, gpu_values, scales[_group(path)])
    return errors


def _check_backends(results_by_name):
    """Per model: the largest cpu - gpu difference in units of its tolerance; raises unless every cpu role solver ran
    on the cpu. Without the gpu runs: the closed-form checks of the cpu runs."""
    groups = _groups()
    for key in groups:
        backends = _role_backends(results_by_name[_run_name(key, 'cpu')])
        if not backends or any(backend != 'cpu' for backend in backends.values()):
            raise ValueError(f"run '{_run_name(key, 'cpu')}': expected every role solver of '{_PHYSICS}' on the cpu "
                             f"backend, summary 'solvers' records {backends!r}")
    if not any(name.endswith('__gpu') for name in results_by_name):   # no usable GPU: the gpu runs were skipped
        return _check_cpu_only(results_by_name)
    normalized, worst_items = {}, {}
    for key, (spec, driven, tolerance) in groups.items():
        errors = _pair_errors(spec, results_by_name[_run_name(key, 'cpu')], results_by_name[_run_name(key, 'gpu')],
                              driven)
        item, error = worst_item(errors)
        normalized[key] = error / tolerance
        worst_items[key] = item
    worst_run, worst = worst_item(normalized)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalized_errors': normalized, 'worst_item_per_run': worst_items, 'worst_run': worst_run,
                        'tolerance': _TOLERANCE, 'line_block_tolerance': _LINE_BLOCK_TOLERANCE}}


def _check_cpu_only(results_by_name):
    """The closed-form checks of the cpu runs (cavity, order list, off-axis order sum, ports), each normalised by its
    own tolerance."""
    def cpu(key):
        return results_by_name[_run_name(key, 'cpu')]

    checks = {'cavity': _check_model_cavity({'cavity': cpu('cavity')}),
              'order_list': _check_model_order_list({'superlu': cpu('order_list_superlu'),
                                                     'line_block': cpu('order_list_line_block')}),
              'offaxis': _check_model_offaxis({'offaxis': cpu('offaxis')}),
              'ports': _check_ports({'superlu': cpu('ports_superlu'), 'line_block': cpu('ports_line_block')})}
    normalized = {name: check['error'] / check['tolerance'] for name, check in checks.items()}
    worst_name, worst = worst_item(normalized)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'normalized_errors': normalized, 'worst_item': worst_name,
                        'gpu_runs': 'skipped (no usable GPU)'}}


BOR_BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='em_bor_cpu_gpu',
        label='EM body of revolution: CPU backend vs GPU',
        description=('em_fdfd_bor on solver_defaults.backend cpu (numpy fields and masks, host assembly, SuperLU / '
                     'ARPACK, cpu line_block) against the same models on gpu: the em_bor_pec_cylinder cavity as an '
                     'eigenfrequency model run (m = 1, 25 nm), the em_bor_order_list_model runs (azimuthal_orders [0, 1, 2], 40 nm, Automatic and a '
                     'line_block node), the em_bor_offaxis_dipole_model driven order sum (orders 0..4, 1.2 x 1.8 um; '
                     'flux lines, far field) on a line_block node (one batched factor of the order family; its host '
                     'SuperLU factorisations are host code identical on both) and a '
                     'pipe TE11 port model (analytic_bor_pipe_te11_section at 2 wavelengths, Automatic and a line_block '
                     'node: the driven single-order line-block path). Per model the largest difference of the shown '
                     'fields (driven runs: e_intensity and hphi relative to their peak, er and ez relative to the peak '
                     '|E|: E_z of the TE11 launch vanishes by symmetry and is roundoff) and of every probe / '
                     'flux / mode-monitor / far-field / resonance-table entry (relative to the peak of its scale '
                     'group; tables: wavelength, frequency, azimuthal_order; phases: wrapped difference / pi where the '
                     f'power fraction reaches {_PHASE_MIN_FRACTION:g}), in units of {_TOLERANCE:g} '
                     f'({_LINE_BLOCK_TOLERANCE:g} for the line_block runs); measured = error = the worst model, '
                     'tolerance 1. The check raises unless every cpu role solver records backend cpu. Without a usable '
                     'GPU the gpu runs are skipped and the cpu runs are checked against the closed forms of their own '
                     f'self-tests instead (ports: T within {_PORT_T_TOLERANCE:g}, R within '
                     f'{2.0 * _PORT_T_TOLERANCE:g}, launched power within {_PORT_LAUNCH_TOLERANCE:g}). Not yet run '
                     '(no record).'),
        specs=_backend_runs, check=_check_backends, heavy=True),
)

"""Per-physics mesh Session / results self-test (S7.2, mesh plan stage P3): the currents strip of coupling/selftests
with heat on its own coarse mesh (cpu run): fields and snapshots on their own meshes, derived values per mesh (D3)
and across meshes, the D3 validation info, summary records and the results_io round trip of mesh names and
meshes.npz; registered by study.register."""
import math
import tempfile

import numpy as np

from .schema import SelfTestCase
from .spec import DerivedValueSpec, MeshSpec

_HEAT_MESH = 'heat_mesh'
_SPLIT = 'split'
_POWER_TOLERANCE = 1e-6        # relative: the conservative transfer keeps the window total (float32 sums)
_INTERPOLATION_TOLERANCE = 1e-5   # relative to the mean rise: float32 T (~3e-5 K ulp at 320 K) and weights


def _split_spec():
    """The heated currents strip (cpu) with heat on a uniform mesh of twice the cell; derived values P_heat (heat Q
    integral, heat mesh), rise_coarse (mean rise, heat mesh) and rise_fine (the same rise + 0 * em eps_r: evaluated
    on the first mesh, D3)."""
    from ..coupling.selftests import _CELL, _DNDT, _HEAT, _T0, _currents_strip_spec
    spec = _currents_strip_spec('heated', _DNDT)
    spec.solver_defaults.backend = 'cpu'
    spec.meshes = [spec.meshes[0], MeshSpec(_HEAT_MESH, mode='uniform', dx=2 * _CELL)]
    spec.physics_by_name(_HEAT).mesh = _HEAT_MESH
    rise = f'(heat_T - {_T0!r})'
    spec.derived_values += [DerivedValueSpec('P_heat', 'heat_Q', ['all'], 'integral'),
                            DerivedValueSpec('rise_coarse', rise, [], 'mean'),
                            DerivedValueSpec('rise_fine', f'{rise} + 0.0 * em_eps_r', [], 'mean')]
    return spec


def _expected_fine_rise(coarse_rise, coarse_extent, fine_shape):
    """Mean over the fine window cells of the coarse rise, linear between the coarse centres and clamped beyond the
    outer ones (separable, np.interp), as the Session's window GridTransfer."""
    x_min, x_max, y_min, y_max = coarse_extent

    def centres(low, high, count):
        return low + (np.arange(count) + 0.5) * ((high - low) / count)
    coarse_x, coarse_y = (centres(x_min, x_max, coarse_rise.shape[0]), centres(y_min, y_max, coarse_rise.shape[1]))
    fine_x, fine_y = centres(x_min, x_max, fine_shape[0]), centres(y_min, y_max, fine_shape[1])
    along_x = np.stack([np.interp(fine_x, coarse_x, column) for column in coarse_rise.T], axis=1)   # (fine x, coarse y)
    both = np.stack([np.interp(fine_y, coarse_y, row) for row in along_x], axis=0)              # (fine x, fine y)
    return float(both.mean())


def _uniform_edges(extent, shape):
    x_min, x_max, y_min, y_max = extent
    return np.linspace(x_min, x_max, shape[0] + 1), np.linspace(y_min, y_max, shape[1] + 1)


def _round_trip_problems(heat, currents, field_meshes):
    """Failures of save_results / load_results of a FieldRecord, a snapshot field and two tensor meshes (mesh names,
    meshes.npz edges, summary field_meshes)."""
    from ..io.results_io import FieldRecord, ResultsData, load_results, save_results
    from .session import Snapshot
    problems = []
    edges = {field.mesh: _uniform_edges(field.extent, field.values.shape) for field in (heat, currents)}
    record = FieldRecord(heat.physics, heat.component, heat.values, *edges[heat.mesh], heat.info, heat.time,
                         heat.time_label, mesh=heat.mesh)
    meshes = {name: {'kind': 'tensor', 'x_edges': x_edges, 'y_edges': y_edges}
              for name, (x_edges, y_edges) in edges.items()}
    data = ResultsData(summary={'status': 'completed', 'field_meshes': dict(field_meshes)},
                       fields={record.key: record}, snapshots=[Snapshot(time=0.0, fields={'heat.T': heat})],
                       meshes=meshes)
    with tempfile.TemporaryDirectory() as directory:
        save_results(directory, data)
        loaded = load_results(directory)
    if loaded.summary.get('files', {}).get('meshes') != 'meshes.npz':
        problems.append(f"summary files.meshes is {loaded.summary.get('files', {}).get('meshes')!r}")
    if loaded.fields[record.key].mesh != _HEAT_MESH:
        problems.append(f'saved field mesh came back as {loaded.fields[record.key].mesh!r}')
    if loaded.snapshots[0].fields['heat.T'].mesh != _HEAT_MESH:
        problems.append(f"snapshot field mesh came back as {loaded.snapshots[0].fields['heat.T'].mesh!r}")
    if loaded.summary.get('field_meshes') != dict(field_meshes):
        problems.append('summary field_meshes did not round-trip')
    if sorted(loaded.meshes) != sorted(meshes) or not all(
            loaded.meshes[name]['kind'] == 'tensor'
            and all(np.array_equal(loaded.meshes[name][axis], meshes[name][axis]) for axis in ('x_edges', 'y_edges'))
            for name in meshes):
        problems.append('meshes.npz did not round-trip')
    return problems


def _check_session_per_mesh(results_by_name):
    """Field shapes / meshes / extents, the transferred power on both meshes, the cross-mesh rise, T_max on the heat
    mesh, summary records, the D3 info and the results round trip; error = largest error / tolerance + failures."""
    from .schema import get_registry
    from .validate import validate
    results = results_by_name[_SPLIT]
    summary = results.summary()
    derived = summary['derived']
    failed = []
    heat, currents = results.field('heat.T'), results.field('currents.q_joule')
    fine_shape, coarse_shape = currents.values.shape, heat.values.shape
    if fine_shape != (2 * coarse_shape[0], 2 * coarse_shape[1]):
        failed.append(f'window shapes {fine_shape} (currents) and {coarse_shape} (heat) are not 2:1')
    if (heat.mesh, currents.mesh) != (_HEAT_MESH, 'mesh'):
        failed.append(f'field meshes {heat.mesh!r} / {currents.mesh!r}, expected {_HEAT_MESH!r} / \'mesh\'')
    if not np.allclose(heat.extent, currents.extent, rtol=1e-9, atol=0.0):
        failed.append(f'extents differ: {heat.extent} vs {currents.extent}')
    snapshot_heat = results.snapshot_field(-1, 'heat.T')
    if snapshot_heat.mesh != _HEAT_MESH or snapshot_heat.values.shape != coarse_shape:
        failed.append(f'snapshot heat.T on {snapshot_heat.mesh!r} with shape {snapshot_heat.values.shape}')
    # transferred density: the heat Q total (heat mesh) equals the currents Joule total (first mesh)
    err_power = abs(derived['P_heat'] - derived['P_joule']) / abs(derived['P_joule'])
    # D3: rise_fine reads heat T interpolated onto the first mesh (em eps_r named first in model order)
    from ..coupling.selftests import _T0
    coarse_rise = np.asarray(heat.values, dtype=np.float64) - _T0
    expected_fine = _expected_fine_rise(coarse_rise, heat.extent, fine_shape)
    err_interpolation = abs(derived['rise_fine'] - expected_fine) / abs(expected_fine)
    if abs(derived['rise_coarse'] - float(coarse_rise.mean())) > _INTERPOLATION_TOLERANCE * abs(coarse_rise.mean()):
        failed.append(f"rise_coarse {derived['rise_coarse']} != the heat mesh mean {float(coarse_rise.mean())}")
    t_max = float(np.max(heat.values))
    if not (math.isfinite(derived['T_max']) and abs(derived['T_max'] - t_max) <= 1e-7 * t_max):
        failed.append(f"T_max {derived['T_max']} != max of heat.T {t_max}")
    # summary records
    field_meshes = summary.get('field_meshes') or {}
    if field_meshes.get('heat.T') != _HEAT_MESH:
        failed.append(f'summary field_meshes {field_meshes}')
    records = {record['name']: record for record in summary.get('meshes') or []}
    if sorted(records) != sorted(['mesh', _HEAT_MESH]) or records[_HEAT_MESH]['window_cells'] != math.prod(
            coarse_shape) or records[_HEAT_MESH]['physics'] != ['heat']:
        failed.append(f'summary meshes {sorted(records)}')
    # validation: the D3 info of rise_fine names the evaluation mesh
    issues = validate(_split_spec(), get_registry())
    if not any(issue.level == 'info' and tuple(issue.path) == ('derived', 'rise_fine')
               and "evaluated on mesh 'mesh'" in issue.message for issue in issues):
        failed.append('no D3 info at derived rise_fine')
    failed += _round_trip_problems(heat, currents, field_meshes)
    normalised = {'power': err_power / _POWER_TOLERANCE, 'interpolation': err_interpolation / _INTERPOLATION_TOLERANCE}
    error = max(normalised.values()) + len(failed)
    return {'measured': derived['rise_fine'], 'expected': expected_fine, 'error': error, 'tolerance': 1.0,
            'details': {'err_power': err_power, 'err_interpolation': err_interpolation,
                        'normalised_errors': normalised, 'failed_checks': failed, 'P_joule': derived['P_joule'],
                        'P_heat': derived['P_heat'], 'rise_coarse': derived['rise_coarse'],
                        'rise_fine': derived['rise_fine']}}


PER_MESH_SELFTEST_CASES = (
    SelfTestCase(
        name='session_per_mesh',
        label='Per-physics meshes: Session fields, derived values and results per mesh (cpu)',
        description='The heated currents_heat_thermo_optic_strip on the cpu backend with heat on its own uniform 40 nm '
                    'mesh (currents and em_fdfd_ez on the 20 nm first mesh). Checks: heat.T comes on 200 x 5 cells of '
                    "mesh 'heat_mesh' and currents.q_joule on 400 x 10 of 'mesh' with equal extents (live field and "
                    'final snapshot); P_heat (integral of heat Q, evaluated on the heat mesh) equals P_joule (integral '
                    'of currents q_joule on the first mesh) to 1e-6 (conservative transfer); rise_fine = mean of (T - '
                    'T0) + 0 * em eps_r is evaluated on the first mesh (D3: em comes first in model order) with T '
                    'interpolated linearly between the heat centres (clamped), equal to that interpolation done in '
                    'numpy to 1e-5; rise_coarse and T_max equal the heat-mesh mean / max; summary field_meshes and '
                    'meshes; validate gives the D3 info at derived rise_fine; save_results / load_results keep the '
                    'field and snapshot mesh names, summary field_meshes and the meshes.npz edges. error = the largest '
                    'error / tolerance + failed checks, tolerance 1.',
        specs=lambda: {_SPLIT: _split_spec()},
        check=_check_session_per_mesh),
)

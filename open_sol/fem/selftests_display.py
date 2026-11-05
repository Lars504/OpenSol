"""Self-test of triangle-mesh display and saving (mesh plan stage T6, CPU; specs() empty, check({}) works directly):
raster and display triangulation of a P2 field, the tripcolor PNG, mesh line sets and the results.npz node tables.
error = number of failed sub-checks (tolerance 0) with the measured values in details."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from .selftests_mesher import _DISC, disc_mesh

_UM = 1e-6
_DISPLAY_H = 0.2 * _UM
_RASTER_PIXELS = (40, 40)
_VALUE_RTOL = 1e-9          # P2 isoparametric reproduces the linear field: point values to evaluation accuracy


def _linear_field(x, y):
    return 1.0 + 3.0 * x / _UM - 2.0 * y / _UM


def _check_display(results_by_name):
    import tempfile
    from pathlib import Path

    from ..core.renderer import FieldInfo
    from ..io.results_io import FieldRecord, ResultsData, load_results, save_results
    from ..model.session import FieldData, Snapshot
    from ..viz.common import triangle_mesh_lines
    from ..viz.result_plots import save_field_figure
    from .assembly import interpolate
    from .dofs import lagrange_dofs
    from .sampling import display_triangulation, evaluate, raster_sampler
    from .sampling import node_values as node_values_of

    checks = SubChecks()
    add = checks.add
    mesh = disc_mesh(_DISPLAY_H)   # the cached patch mesh
    dofs = lagrange_dofs(mesh, 2)
    coefficients = interpolate(dofs, _linear_field)
    x0, y0, x1, y1 = mesh.window
    x_centres = x0 + (np.arange(_RASTER_PIXELS[0]) + 0.5) * ((x1 - x0) / _RASTER_PIXELS[0])
    y_centres = y0 + (np.arange(_RASTER_PIXELS[1]) + 0.5) * ((y1 - y0) / _RASTER_PIXELS[1])
    raster = raster_sampler(mesh, x_centres, y_centres, dofs.geometry_order).sample(dofs, coefficients)
    exact_raster = _linear_field(x_centres[:, None], y_centres[None, :])
    scale = float(np.abs(exact_raster).max())
    add('raster_covers_window', bool(np.isfinite(raster).all()), int(np.count_nonzero(~np.isfinite(raster))))
    raster_error = float(np.nanmax(np.abs(raster - exact_raster)) / scale)
    add('raster_equals_field', raster_error <= _VALUE_RTOL, raster_error)
    nodes, triangles = display_triangulation(mesh, 2)
    node_values = evaluate(dofs, coefficients, nodes)
    node_error = float(np.max(np.abs(node_values - _linear_field(nodes[:, 0], nodes[:, 1]))) / scale)
    add('node_values_equal_field', node_error <= _VALUE_RTOL, node_error)
    direct_error = float(np.max(np.abs(node_values_of(dofs, coefficients) - node_values)) / scale)   # no location
    add('direct_node_values_equal_located', direct_error <= _VALUE_RTOL, direct_error)
    add('four_subtriangles_per_element', len(triangles) == 4 * mesh.n_elements, int(len(triangles)))

    info = FieldInfo('u', 'u', unit='V', primary=True)
    triangulation = (np.asarray(nodes, dtype=np.float64), np.asarray(triangles, dtype=np.int32),
                     np.asarray(node_values, dtype=np.float32))
    field_data = FieldData('test', 'u', raster.astype(np.float32), (x0, x1, y0, y1), info, 0.0, None, mesh='triangles',
                           triangulation=triangulation)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(save_field_figure(field_data, Path(folder) / 'triangles.png', contours=6))
        size = path.stat().st_size if path.is_file() else 0
        add('tripcolor_png_written', size > 0, size)
        x_edges, y_edges = np.linspace(x0, x1, _RASTER_PIXELS[0] + 1), np.linspace(y0, y1, _RASTER_PIXELS[1] + 1)
        record = FieldRecord('test', 'u', field_data.values, x_edges, y_edges, info, 0.0, None, mesh='triangles',
                             triangulation=triangulation)
        snapshots = [Snapshot(time=float(index), fields={'test.u': field_data}) for index in range(2)]
        save_results(Path(folder) / 'run', ResultsData(summary={'status': 'completed'}, fields={'test.u': record},
                                                       snapshots=snapshots))
        with np.load(Path(folder) / 'run' / 'results.npz', allow_pickle=False) as npz:
            tables = [name for name in npz.files if name.startswith('triangulation/') and name.endswith('/nodes')]
        add('one_shared_node_table', len(tables) == 1, len(tables))
        loaded = load_results(Path(folder) / 'run')
        round_trip = [loaded.fields['test.u'].triangulation, *(snapshot.fields['test.u'].triangulation
                                                               for snapshot in loaded.snapshots)]
        same = all(item is not None and all(np.array_equal(saved, original)
                                            for saved, original in zip(item, triangulation)) for item in round_trip)
        add('node_tables_round_trip', same, len(round_trip))

    lines = triangle_mesh_lines(mesh)
    n_curved = int(np.count_nonzero(mesh.edge_curve >= 0))
    n_segments = (len(lines.segments) + len(lines.ring_segments)) // 2
    add('mesh_lines_every_edge', n_segments == mesh.n_edges + n_curved and not lines.thinned,
        {'segments': n_segments, 'edges': int(mesh.n_edges), 'curved': n_curved})
    on_circle = np.abs(np.hypot(lines.segments[:, 0] - _DISC[0], lines.segments[:, 1] - _DISC[1]) - _DISC[2])
    add('curved_edges_through_midside_nodes', n_curved > 0 and int(np.count_nonzero(on_circle < 1e-12 * _UM))
        >= 3 * n_curved, n_curved)
    thinned = triangle_mesh_lines(mesh, max_edges=1)
    add('thinned_keeps_outlines', thinned.thinned and 0 < len(thinned.segments) < len(lines.segments),
        int(len(thinned.segments)))
    add('min_angle_reported', math.isfinite(float(lines.quality.get('min_angle', math.nan))),
        lines.quality.get('min_angle'))
    return checks_result(checks)


DISPLAY_SELFTEST_CASES = (
    SelfTestCase(
        name='viz_triangle_plot',
        label='Triangle mesh display: raster, tripcolor, node tables',
        description='CPU (mesh plan T6) on the 2 x 2 um disc mesh (h 0.2 um): a linear field interpolated in P2 '
                    '(isoparametric) gives a raster covering the window and node values of the 1 -> 4 display '
                    'triangulation equal to the field (1e-9 relative); save_field_figure draws it by tripcolor with '
                    'contours (PNG written); results.npz stores one shared node table for a saved field and two '
                    'snapshots and reads every triangulation back unchanged; the mesh line sets hold every edge '
                    '(curved ones as two chords through the midside node on the circle), thinning keeps the outlines. '
                    'error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_display,
        max_wall_time=120.0),
)

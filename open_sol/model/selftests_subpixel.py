"""Self-tests of subpixel smoothing (igor 1.5): interface geometry, the averaged eps of the FDTD / FDFD / em_modes /
em_fdtd_3d engines and of the mode ports (builds only, on the cpu backend and the gpu, cpu vs gpu), and the
translation invariance of a dielectric-disk resonance (model runs; em_modes slab modes: example
analytic_modes_subpixel_slab_thickness)."""
import math

import numpy as np

from ..core.constants import C0, EPS0
from .schema import SelfTestCase
from .selftest_helpers import SubChecks, float_list
from .spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                   SolverDefaultsSpec, StudySpec)

_PHYSICS = 'em'

# ---- consistency case: a glass disk and a silicon slab (full width, edges off the cell edges), 20 nm cells ----
_SIZE = 1.6e-6
_CELL = 20e-9
_DISK = {'cx': 0.8037e-6, 'cy': 1.0123e-6, 'r': 0.3891e-6}
_GUIDE = {'x0': 0.0, 'y0': 0.2507e-6, 'x1': _SIZE, 'y1': 0.4413e-6}
_DISK_EPS, _GUIDE_EPS = 4.0, 12.0
_SPLIT_Y = 0.5e-6                      # locations below: the slab, above: the disk
_PORT_X = 0.2e-6                       # port line along y through the slab, clear of the disk
_PORT_OMEGA = 2.0 * math.pi * C0 / 1.55e-6
_EPS_UPDATE = (0.5, 0.25)              # eps_xx / eps_yy added through the coupling slots (live re-averaging)
_RELATIVE_TOLERANCE = 1e-5             # float32 averaging vs the host formula / FDTD vs FDFD / port lines
_AREA_TOLERANCE = 1e-3                 # disk area and slab thickness from the averaged E_z cells (64-sample axis)
_THICKNESS_TOLERANCE = 2e-3
_STRAIGHT_NORMAL_TOLERANCE = 1e-3      # n_d^2 on the straight slab edges (radial weight: exact direction)
_DISK_NORMAL_MEAN, _DISK_NORMAL_MAX = 0.02, 0.1
_GEOMETRY_TOLERANCE = 1e-5             # cpu vs gpu fill / n_d^2 / face shares (float32 sums in another order)
_SET_NAMES = ('location', 'cell_a', 'cell_b', 'fill', 'normal_square')
_SET_NAMES_2D = _SET_NAMES + ('upper_cell', 'lower_share', 'upper_share')
_KINDS_2D = ('cells', 'faces_y', 'faces_x')
_KINDS_3D = ('faces_x', 'faces_y', 'faces_z')


def _consistency_spec(type_name, backend='auto'):
    """Glass disk + silicon slab in vacuum, uniform 20 nm cells, CPML, subpixel smoothing on, one physics, on
    `backend` ('auto' | 'gpu' | 'cpu')."""
    if type_name.startswith('em_fdtd'):
        study = StudySpec(type='time_dependent', settings={'duration': 1e-15})
    else:
        study = StudySpec(type='mode_analysis' if type_name == 'em_modes' else 'stationary')
    return ModelSpec(
        title=f'subpixel smoothing consistency ({type_name})',
        domain=DomainSpec(size_x=_SIZE, size_y=_SIZE),
        geometry=[GeometrySpec('disk', 'circle', dict(_DISK)), GeometrySpec('guide', 'rectangle', dict(_GUIDE))],
        materials=[MaterialSpec('glass', properties={'eps_r': _DISK_EPS}, selection=['disk']),
                   MaterialSpec('silicon', properties={'eps_r': _GUIDE_EPS}, selection=['guide'])],
        physics=[PhysicsSpec(_PHYSICS, type_name)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(backend=backend, subpixel_smoothing=True),
        study=study)


def _host(array):
    """numpy copy-or-view of a numpy / cupy array."""
    from ..core.backend import to_host
    return to_host(array)


def _host_set(locations, names=_SET_NAMES):
    """numpy copies of an InterfaceSet's arrays."""
    return {name: np.array(_host(getattr(locations, name))) for name in names}


def _expected_smoothed(locations, eps_cells, complement=False):
    """Host float64 1 / (n^2 <1/eps> + (1 - n^2) / <eps>) of the two representative cells (complement: n^2 = 1 -
    the stored one, the other in-plane component, as em_modes' E_x / E_y faces)."""
    eps = eps_cells.ravel().astype(np.float64)
    eps_a, eps_b = eps[locations['cell_a']], eps[locations['cell_b']]
    fill, normal_square = locations['fill'].astype(np.float64), locations['normal_square'].astype(np.float64)
    if complement:
        normal_square = 1.0 - normal_square
    mean = fill * eps_a + (1.0 - fill) * eps_b
    inverse_mean = fill / eps_a + (1.0 - fill) / eps_b
    return 1.0 / (normal_square * inverse_mean + (1.0 - normal_square) / mean)


def _relative_difference(measured, expected):
    """max |measured / expected - 1| (0 for empty arrays)."""
    measured, expected = np.asarray(measured, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(measured / expected - 1.0))) if measured.size else 0.0


def _location_points(domain, kind, location):
    """(x, y) [m] of locations: cell centres, the hi-y face (faces_y) or the hi-x face (faces_x)."""
    ny = domain.grid.ny
    i_index, j_index = np.divmod(location, ny)
    x_edges, y_edges = domain.cell_edges()
    x_centres, y_centres = domain.cell_centers()
    x = x_edges[i_index + 1] if kind == 'faces_x' else x_centres[i_index]
    y = y_edges[j_index + 1] if kind == 'faces_y' else y_centres[j_index]
    return x, y


def _build(spec, registry):
    from .builder import build
    session = build(spec, registry, log=lambda level, message: None)
    return session, session.sim.physics[_PHYSICS]


def _ez_fdtd_checks(registry, add, window_values, backend='auto', geometry_sets=None):
    """E_z FDTD: averaged cells vs the host formula, disk area and slab thickness, the ez port line; geometry_sets
    (a dict) receives the host copies of the three InterfaceSets."""
    from ..physics.em.mode_ports import PortLine
    session, physics = _build(_consistency_spec('em_fdtd_ez', backend), registry)
    try:
        domain, engine = session.domain, physics.engine
        geometry = physics.subpixel_smoothing.geometry
        if geometry_sets is not None:
            geometry_sets.update({kind: _host_set(geometry.kind(kind), _SET_NAMES_2D) for kind in _KINDS_2D})
        cells = _host_set(geometry.cells)
        eps_effective = float(engine.dt) / (EPS0 * _host(engine.cb).astype(np.float64))
        eps_cells = _host(domain.materials.eps_r)
        add('ez_cells_formula', _relative_difference(eps_effective.ravel()[cells['location']],
                                                     _expected_smoothed(cells, eps_cells)), _RELATIVE_TOLERANCE)
        window = domain.physical_window()
        window_eps = eps_effective[window]
        window_values['ez'] = window_eps
        dx_cells, dy_cells = domain.cell_widths()
        cell_area = dx_cells[window[0]][:, None] * dy_cells[window[1]][None, :]
        y_centres = domain.cell_centers()[1][window[1]]
        upper = (y_centres > _SPLIT_Y)[None, :]
        disk_area = float(np.sum(np.where(upper, (window_eps - 1.0) / (_DISK_EPS - 1.0), 0.0) * cell_area))
        add('ez_disk_area', abs(disk_area / (math.pi * _DISK['r'] ** 2) - 1.0), _AREA_TOLERANCE)
        slab_area = float(np.sum(np.where(upper, 0.0, (window_eps - 1.0) / (_GUIDE_EPS - 1.0)) * cell_area))
        slab_thickness = slab_area / float(np.sum(dx_cells[window[0]]))
        add('ez_slab_thickness', abs(slab_thickness / (_GUIDE['y1'] - _GUIDE['y0']) - 1.0), _THICKNESS_TOLERANCE)

        line_cell = domain.pos_to_cell_x(_PORT_X)
        port = PortLine(physics, 'y', line_cell, (window[1].start, window[1].stop), 'self-test port')
        port.read_materials(physics)
        line = port.slab_line(_PORT_OMEGA)
        a, b = port.solve_range
        add('ez_port_cells', _relative_difference(np.real(line.index_term / line.cell_weight),
                                                  eps_effective[line_cell, a:b]), _RELATIVE_TOLERANCE)
    finally:
        session.close()


def _hz_fdtd_checks(registry, add, window_values, backend='auto'):
    """H_z FDTD: averaged faces vs the host formula, normals (slab exact, disk radial), port line, live update."""
    from ..physics.em.mode_ports import PortLine
    session, physics = _build(_consistency_spec('em_fdtd_hz', backend), registry)
    try:
        domain, engine, materials = session.domain, physics.engine, session.domain.materials
        geometry = physics.subpixel_smoothing.geometry
        window = domain.physical_window()
        faces = {'faces_y': (_host_set(geometry.faces_y), engine.eps_x, 'eps_xx'),
                 'faces_x': (_host_set(geometry.faces_x), engine.eps_y, 'eps_yy')}
        for kind, (locations, face_eps, array) in faces.items():
            measured = _host(face_eps).astype(np.float64) / EPS0
            window_values[kind] = measured[window]
            add(f'hz_{kind}_formula', _relative_difference(
                measured.ravel()[locations['location']],
                _expected_smoothed(locations, _host(getattr(materials, array)))), _RELATIVE_TOLERANCE)
            x, y = _location_points(domain, kind, locations['location'])
            i_index, j_index = np.divmod(locations['location'], domain.grid.ny)
            inside = ((window[0].start <= i_index) & (i_index < window[0].stop) & (window[1].start <= j_index)
                      & (j_index < window[1].stop))   # the grid-edge PML cells repeat their raster column
            slab = (y < _SPLIT_Y) & inside
            straight_expected = 1.0 if kind == 'faces_x' else 0.0
            add(f'hz_{kind}_slab_normal', float(np.max(np.abs(locations['normal_square'][slab] - straight_expected),
                                                       initial=0.0)), _STRAIGHT_NORMAL_TOLERANCE)
            disk = y >= _SPLIT_Y
            offset_x, offset_y = x[disk] - _DISK['cx'], y[disk] - _DISK['cy']
            component = offset_x if kind == 'faces_y' else offset_y
            disk_error = np.abs(locations['normal_square'][disk] - component ** 2 / (offset_x ** 2 + offset_y ** 2))
            add(f'hz_{kind}_disk_normal_mean', float(np.mean(disk_error)) if disk_error.size else math.inf,
                _DISK_NORMAL_MEAN)
            add(f'hz_{kind}_disk_normal_max', float(np.max(disk_error, initial=0.0)), _DISK_NORMAL_MAX)

        line_cell = domain.pos_to_cell_x(_PORT_X)
        port = PortLine(physics, 'y', line_cell, (window[1].start, window[1].stop), 'self-test port')
        port.read_materials(physics)
        line = port.slab_line(_PORT_OMEGA)
        a, b = port.solve_range
        face_index = np.clip(np.arange(a - 1, b), 0, domain.grid.ny - 1)
        eps_x, eps_y = (_host(array).astype(np.float64) / EPS0 for array in (engine.eps_x, engine.eps_y))
        add('hz_port_transverse', _relative_difference(np.real(line.cell_weight), eps_y[line_cell, a:b]),
            _RELATIVE_TOLERANCE)
        add('hz_port_longitudinal', _relative_difference(np.real(1.0 / line.face_weight),
                                                         eps_x[line_cell, face_index]), _RELATIVE_TOLERANCE)

        for slot, array, change in (('eps_xx_update', 'eps_xx', _EPS_UPDATE[0]),
                                    ('eps_yy_update', 'eps_yy', _EPS_UPDATE[1])):
            physics.inject_source(slot, getattr(materials, array) + change)
        engine.refresh_coefficients()
        for kind, (locations, face_eps, array) in faces.items():
            add(f'hz_{kind}_after_update', _relative_difference(
                _host(face_eps).astype(np.float64).ravel()[locations['location']] / EPS0,
                _expected_smoothed(locations, _host(getattr(materials, array)))), _RELATIVE_TOLERANCE)
    finally:
        session.close()


def _fdfd_checks(registry, add, window_values, backend='auto'):
    """FDFD assembly (no source, one solve) = the FDTD averaged values on the window: ez cell eps, hz face eps."""
    for polarization in ('ez', 'hz'):
        session, physics = _build(_consistency_spec(f'em_fdfd_{polarization}', backend), registry)
        try:
            physics.solve_harmonic(physics.omega)
            system, window = physics.system, session.domain.physical_window()
            if polarization == 'ez':
                pairs = (('ez', system.cell_eps),)
            else:
                pairs = (('faces_y', system.eps_hi_y_faces), ('faces_x', system.eps_hi_x_faces))
            for key, values in pairs:
                assembled = np.real(_host(values))[window]
                add(f'fdfd_{polarization}_{key}_equals_fdtd', _relative_difference(assembled, window_values[key]),
                    _RELATIVE_TOLERANCE)
        finally:
            session.close()


def _modes_checks(registry, add, window_values, backend='auto'):
    """em_modes (no solve): point-material E_x hi-x / E_y hi-y faces vs the host formula with the other in-plane
    normal component, E_z cells = the E_z FDTD averaged cells on the window, also after an eps_r slot write."""
    session, physics = _build(_consistency_spec('em_modes', backend), registry)
    try:
        domain, materials = session.domain, session.domain.materials
        geometry = physics.subpixel_smoothing.geometry
        window = domain.physical_window()

        def compare(label):
            points = physics.point_materials(physics.omega)
            for kind, values, array in (('faces_x', points.eps_x_faces, 'eps_xx'),
                                        ('faces_y', points.eps_y_faces, 'eps_yy')):
                locations = _host_set(geometry.kind(kind))
                add(f'modes_{kind}_formula{label}', _relative_difference(
                    np.real(values).ravel()[locations['location']],
                    _expected_smoothed(locations, _host(getattr(materials, array)), complement=True)),
                    _RELATIVE_TOLERANCE)
            return np.real(points.eps_z_centres)

        add('modes_cells_equal_ez_fdtd', _relative_difference(compare('')[window], window_values['ez']),
            _RELATIVE_TOLERANCE)
        physics.inject_source('eps_r_update', materials.eps_r + _EPS_UPDATE[0])
        cells = _host_set(geometry.cells)
        add('modes_cells_after_update', _relative_difference(
            compare('_after_update').ravel()[cells['location']],
            _expected_smoothed(cells, _host(materials.eps_r))), _RELATIVE_TOLERANCE)
    finally:
        session.close()


# ---- 3D consistency (em_fdtd_3d builds): a silicon slab (full width, z faces off the cell edges) under a glass pillar ----
_SIZE_3D = 1.0e-6                       # 40 interior cells per axis (was 1.2 um; not yet confirmed by a run)
_CELL_3D = 25e-9
_PML_3D = 10
_SLAB_Z = (0.1123e-6, 0.3047e-6)
_PILLAR = {'cx': 0.5037e-6, 'cy': 0.4873e-6, 'r': 0.2891e-6, 'z0': 0.3812e-6, 'z1': 0.7987e-6}
_PILLAR_CUT_Z = 0.59e-6                 # a layer inside the pillar, far from its ends (em_modes / port comparison)
_SLAB_COLUMN = (0.1e-6, 0.1e-6)         # an (x, y) column clear of the pillar (slab thickness from the faces)
_Z_THICKNESS_TOLERANCE = 5e-3           # 16 heights per half cell: <= dz / 64 per interface


def _spec_3d(backend='auto'):
    """em_fdtd_3d build (no run): slab eps 12 + pillar eps 4 in vacuum, uniform 25 nm cells, CPML, smoothing on."""
    slab = {'x0': 0.0, 'y0': 0.0, 'z0': _SLAB_Z[0], 'x1': _SIZE_3D, 'y1': _SIZE_3D, 'z1': _SLAB_Z[1]}
    return ModelSpec(
        title='subpixel smoothing consistency (em_fdtd_3d)',
        domain=DomainSpec(size_x=_SIZE_3D, size_y=_SIZE_3D, size_z=_SIZE_3D),
        geometry=[GeometrySpec('slab', 'box', slab), GeometrySpec('pillar', 'circle', dict(_PILLAR))],
        materials=[MaterialSpec('silicon', properties={'eps_r': _GUIDE_EPS}, selection=['slab']),
                   MaterialSpec('glass', properties={'eps_r': _DISK_EPS}, selection=['pillar'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdtd_3d', settings={'pml_cells': _PML_3D})],
        mesh=MeshSpec(mode='uniform', dx=_CELL_3D),
        solver_defaults=SolverDefaultsSpec(backend=backend, subpixel_smoothing=True),
        study=StudySpec(type='time_dependent', settings={'duration': 1e-15}))


def _pillar_modes_spec(backend='auto'):
    """The pillar's cross-section as an em_modes model on the same (x, y) grid."""
    disk = {name: _PILLAR[name] for name in ('cx', 'cy', 'r')}
    return ModelSpec(
        title='subpixel smoothing consistency (em_modes cross-section of the 3D pillar)',
        domain=DomainSpec(size_x=_SIZE_3D, size_y=_SIZE_3D),
        geometry=[GeometrySpec('disk', 'circle', disk)],
        materials=[MaterialSpec('glass', properties={'eps_r': _DISK_EPS}, selection=['disk'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_modes', settings={'pml_cells': _PML_3D})],
        mesh=MeshSpec(mode='uniform', dx=_CELL_3D),
        solver_defaults=SolverDefaultsSpec(backend=backend, subpixel_smoothing=True),
        study=StudySpec(type='mode_analysis'))


def _engine_face_eps(engine):
    """Host float64 (3, nx, ny, nz) relative eps the E updates use (dt / (eps0 cb); no conductivity here)."""
    cb = engine.xp.empty(engine.shape, dtype=np.float32)
    values = []
    for component in range(3):
        engine.face_cb_into(cb, component, (0, 0, 0), engine.shape)
        values.append(float(engine.dt) / (EPS0 * _host(cb).astype(np.float64)))
    return np.stack(values)


def _face_formula_errors(engine):
    """Per kind: max |engine - formula| / (1e-5 formula + face quantisation error) over the interface faces, formula =
    the averaging of the representative cells' current table eps."""
    smoothing = engine.subpixel_smoothing
    measured = _engine_face_eps(engine)
    material_id = np.array(_host(engine.material_id)).ravel()
    tables = [_host(table).astype(np.float64) for table in (engine.table_eps_x, engine.table_eps_y,
                                                            engine.table_eps_z)]
    allowance = smoothing.quantisation_error or 0.0
    errors = {}
    for axis, kind in enumerate(('faces_x', 'faces_y', 'faces_z')):
        locations = _host_set(smoothing.geometry.kind(kind))
        cell_values = tables[axis][material_id]
        expected = _expected_smoothed(locations, cell_values)
        difference = np.abs(measured[axis].ravel()[locations['location']] - expected)
        errors[kind] = float(np.max(difference / (1e-5 * expected + allowance), initial=0.0))
    return errors, measured


def _em3d_checks(registry, add, backend='auto', geometry_sets=None):
    """em_fdtd_3d: face eps vs the formula (also after an eps_r slot write), pillar normals, slab thickness from the E_x
    / E_z faces of one column, the pillar layer vs em_modes' smoothed cross-section and vs a port's materials;
    geometry_sets (a dict) receives the host copies of the three InterfaceSet3D."""
    from ..physics.em3d.mode_ports3d import port_boundary, port_materials
    from ..physics.em.cross_section_modes import mode_point_materials
    session, physics = _build(_spec_3d(backend), registry)
    try:
        domain, engine = session.domain, physics.engine
        if geometry_sets is not None:
            geometry_sets.update({kind: _host_set(engine.subpixel_smoothing.geometry.kind(kind))
                                  for kind in _KINDS_3D})
        errors, measured = _face_formula_errors(engine)
        for kind, value in errors.items():
            add(f'em3d_{kind}_formula', value, 1.0)
        _, ny, nz = engine.shape
        window = domain.physical_window()
        x_edges, y_edges, z_edges = domain.cell_edges()
        x_centres, y_centres, z_centres = domain.cell_centers()
        pillar_layers = (z_centres > _PILLAR['z0'] + _CELL_3D) & (z_centres < _PILLAR['z1'] - _CELL_3D)
        for axis, kind in enumerate(('faces_x', 'faces_y')):
            locations = _host_set(engine.subpixel_smoothing.geometry.kind(kind))
            i_index, rest = np.divmod(locations['location'], ny * nz)
            j_index, k_index = np.divmod(rest, nz)
            inside = ((window[0].start <= i_index) & (i_index < window[0].stop) & (window[1].start <= j_index)
                      & (j_index < window[1].stop) & pillar_layers[k_index])
            x = (x_edges[i_index + 1] if axis == 0 else x_centres[i_index])[inside] - _PILLAR['cx']
            y = (y_edges[j_index + 1] if axis == 1 else y_centres[j_index])[inside] - _PILLAR['cy']
            error = np.abs(locations['normal_square'][inside] - (x if axis == 0 else y) ** 2 / (x ** 2 + y ** 2))
            add(f'em3d_{kind}_pillar_normal_mean', float(np.mean(error)) if error.size else math.inf,
                _DISK_NORMAL_MEAN)
            add(f'em3d_{kind}_pillar_normal_max', float(np.max(error, initial=0.0)), _DISK_NORMAL_MAX)

        i_column, j_column = domain.pos_to_cell_x(_SLAB_COLUMN[0]), domain.pos_to_cell_y(_SLAB_COLUMN[1])
        layers = slice(window[2].start, window[2].stop)
        dz = np.diff(z_edges)[layers]
        tangential = measured[0, i_column, j_column, layers]
        thickness_x = float(np.sum((tangential - 1.0) / (_GUIDE_EPS - 1.0) * dz))
        dual = np.diff(z_centres)[window[2].start:window[2].stop - 1]
        normal = measured[2, i_column, j_column, window[2].start:window[2].stop - 1]
        thickness_z = float(np.sum((1.0 / normal - 1.0) / (1.0 / _GUIDE_EPS - 1.0) * dual))
        expected_thickness = _SLAB_Z[1] - _SLAB_Z[0]
        add('em3d_slab_thickness_ex', abs(thickness_x / expected_thickness - 1.0), _Z_THICKNESS_TOLERANCE)
        add('em3d_slab_thickness_ez', abs(thickness_z / expected_thickness - 1.0), _Z_THICKNESS_TOLERANCE)

        cut = domain.pos_to_cell_z(_PILLAR_CUT_Z)
        plane = (window[0], window[1])
        layer_eps = [measured[axis][:, :, cut][plane] for axis in range(3)]
        modes_session, modes = _build(_pillar_modes_spec(backend), registry)
        try:
            points = modes.point_materials(modes.omega)
            for name, values, axis in (('x_faces', points.eps_x_faces, 0), ('y_faces', points.eps_y_faces, 1),
                                       ('z_centres', points.eps_z_centres, 2)):
                add(f'em3d_layer_{name}_equal_em_modes', _relative_difference(layer_eps[axis], np.real(values)[plane]),
                    _RELATIVE_TOLERANCE)
        finally:
            modes_session.close()

        pml_dx = max(float(max(widths[0], widths[-1])) for widths in engine.widths[:2])
        port_points = mode_point_materials((engine.widths[0], engine.widths[1]), port_boundary(engine.config, 2),
                                           pml_dx, port_materials(physics, 2, cut, _PORT_OMEGA))
        for name, values, axis in (('x_faces', port_points.eps_x_faces, 0), ('y_faces', port_points.eps_y_faces, 1),
                                   ('z_centres', port_points.eps_z_centres, 2)):
            add(f'em3d_port_{name}_equal_engine', _relative_difference(np.real(values)[plane], layer_eps[axis]),
                _RELATIVE_TOLERANCE)

        physics.inject_source('eps_r_update', domain.materials.eps_r + _EPS_UPDATE[0])
        physics.step()   # rebuilds the index and the face overrides
        errors, _ = _face_formula_errors(engine)
        for kind, value in errors.items():
            add(f'em3d_{kind}_formula_after_update', value, 1.0)
    finally:
        session.close()


def _consistency_2d(registry, add, backend='auto', geometry_sets=None):
    """The 2D consistency builds (E_z / H_z FDTD, FDFD, em_modes) on `backend`; returns their window values."""
    window_values = {}
    _ez_fdtd_checks(registry, add, window_values, backend, geometry_sets)
    _hz_fdtd_checks(registry, add, window_values, backend)
    _fdfd_checks(registry, add, window_values, backend)
    _modes_checks(registry, add, window_values, backend)
    return window_values


def _check_adder(checks, prefix=''):
    """add(name, value, tolerance) of SubChecks `checks` under prefix + name (passed: finite value <= tolerance)."""
    def add(name, value, tolerance):
        checks.add_error(prefix + name, value, tolerance)
    return add


def _geometry_differences(add, label, cpu_sets, gpu_sets, names):
    """cpu vs gpu interface sets per kind: locations only one backend has, mismatched integer entries and the max
    |difference| of the float entries (`names` after 'location') on the common locations."""
    for kind, cpu_set in cpu_sets.items():
        gpu_set = gpu_sets[kind]
        add(f'cpu_gpu_{label}{kind}_unmatched_locations',
            float(np.setxor1d(cpu_set['location'], gpu_set['location']).size), 0.0)
        _, cpu_rows, gpu_rows = np.intersect1d(cpu_set['location'], gpu_set['location'], return_indices=True)
        for name in names[1:]:
            cpu_values, gpu_values = cpu_set[name][cpu_rows], gpu_set[name][gpu_rows]
            if np.issubdtype(cpu_values.dtype, np.integer):
                add(f'cpu_gpu_{label}{kind}_{name}_mismatches', float(np.count_nonzero(cpu_values != gpu_values)),
                    0.0)
            else:
                difference = np.abs(cpu_values.astype(np.float64) - gpu_values.astype(np.float64))
                add(f'cpu_gpu_{label}{kind}_{name}', float(np.max(difference, initial=0.0)), _GEOMETRY_TOLERANCE)


def _check_cpu_gpu(results_by_name):
    """The consistency sub-checks (2D and em_fdtd_3d) on the cpu backend, and with a usable GPU on the gpu too, plus
    the cpu vs gpu interface geometry and the 2D averaged window eps; error = number of failed sub-checks, tolerance
    0."""
    from ..core.backend import gpu_available
    from .schema import get_registry
    registry = get_registry()
    checks = SubChecks()
    backends = ('cpu', 'gpu') if gpu_available() else ('cpu',)
    window_values, sets_2d, sets_3d = {}, {}, {}
    for backend in backends:
        add = _check_adder(checks, f'{backend}_')
        sets_2d[backend], sets_3d[backend] = {}, {}
        window_values[backend] = _consistency_2d(registry, add, backend, sets_2d[backend])
        _em3d_checks(registry, add, backend, sets_3d[backend])
    if 'gpu' not in backends:
        return checks.result(gpu_builds='skipped (no usable GPU)')
    add = _check_adder(checks)
    _geometry_differences(add, '', sets_2d['cpu'], sets_2d['gpu'], _SET_NAMES_2D)
    _geometry_differences(add, 'em3d_', sets_3d['cpu'], sets_3d['gpu'], _SET_NAMES)
    for key in ('ez', 'faces_y', 'faces_x'):
        add(f'cpu_gpu_{key}_window_eps', _relative_difference(window_values['cpu'][key], window_values['gpu'][key]),
            _RELATIVE_TOLERANCE)
    n_locations = {**{kind: int(entry['location'].size) for kind, entry in sets_2d['gpu'].items()},
                   **{f'em3d_{kind}': int(entry['location'].size) for kind, entry in sets_3d['gpu'].items()}}
    return checks.result(n_locations=n_locations)


# ---- disk resonance case: translation invariance of a whispering-gallery pair ----
_DISK_RADIUS = 1e-6
_DISK_EPS_R = 12.11
# 3.0 um box (0.5 um disk-to-CPML gap, the m ~ 10 WGM tail decays in ~80 nm) and 4 resonances (FSR ~0.1 um: the 4
# nearest the target hold a cos / sin pair); 2026-10-20 runs used 3.6 um and 6: not yet confirmed by a run
_DISK_DOMAIN = 3.0e-6
_DISK_CELL = 25e-9
_DISK_TARGET = 1.55e-6
_DISK_RESONANCES = 4
_OFFSETS = ((0.0, 0.0), (0.31, 0.17), (0.57, 0.43))   # disk centre shifts in cells (x, y)
_WAVELENGTH_COLUMN, _Q_COLUMN = 1, 4
# (max - min) / mean of the pair wavelength over the offsets: smoothed 9.1e-5 (E_z) / 1.3e-4 (H_z), staircase 7.2e-4 /
# 6.9e-4 (2026-10-20 runs; the staircase runs are not repeated: the tolerances sit between the two)
_SPREAD_TOLERANCE = {'ez': 3e-4, 'hz': 3e-4}
_CLOSED_FORM_TOLERANCE = 1.5e-2                         # FDFD numerical dispersion at ~18 cells per wavelength in Si


def _disk_spec(polarization, offset):
    """em_fdfd_<polarization> eigenfrequency study of a silicon disk shifted by `offset` cells, subpixel smoothing."""
    centre = 0.5 * _DISK_DOMAIN
    return ModelSpec(
        title=f'subpixel disk resonance ({polarization}, offset {offset})',
        domain=DomainSpec(size_x=_DISK_DOMAIN, size_y=_DISK_DOMAIN),
        geometry=[GeometrySpec('disk', 'circle', {'cx': centre + offset[0] * _DISK_CELL,
                                                  'cy': centre + offset[1] * _DISK_CELL, 'r': _DISK_RADIUS})],
        materials=[MaterialSpec('silicon_disk', properties={'eps_r': _DISK_EPS_R}, selection=['disk'])],
        physics=[PhysicsSpec(_PHYSICS, f'em_fdfd_{polarization}', settings={'wavelength': _DISK_TARGET},
                             features=[FeatureSpec('resonances', 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_DISK_CELL),
        solver_defaults=SolverDefaultsSpec(subpixel_smoothing=True),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _DISK_RESONANCES}))


def _disk_specs():
    return {f'{polarization}_{index}': _disk_spec(polarization, offset)
            for polarization in ('ez', 'hz') for index, offset in enumerate(_OFFSETS)}


def _pair_wavelengths(results):
    """Wavelengths [m] of the two highest-Q resonances (the whispering-gallery cos / sin pair)."""
    rows = results.result(_PHYSICS, 'resonances')['rows']
    ranked = sorted(rows, key=lambda row: -float(row[_Q_COLUMN]))[:2]
    if len(ranked) < 2:
        raise ValueError(f'expected at least 2 resonances, got {len(rows)}')
    return [float(row[_WAVELENGTH_COLUMN]) for row in ranked]


def _closed_form_wavelengths(polarization):
    """Vacuum wavelengths [m] of the high-Q disk resonances in 1.4..1.7 um: roots k of p J_m'(n k R) / J_m(n k R) =
    H2_m'(k R) / H2_m(k R) (p = n for E_z, 1/n for H_z; outgoing exp(+j omega t) waves), 2 pi / Re k."""
    from scipy.optimize import newton
    from scipy.special import h2vp, hankel2, jv, jvp
    index = math.sqrt(_DISK_EPS_R)
    weight = index if polarization == 'ez' else 1.0 / index
    radius = _DISK_RADIUS
    roots = []
    for order in range(4, 30):
        def mismatch(k, order=order):
            inner, outer = index * k * radius, k * radius
            return weight * jvp(order, inner) / jv(order, inner) - h2vp(order, outer) / hankel2(order, outer)
        wavenumbers = np.linspace(2.0 * math.pi / 1.7e-6, 2.0 * math.pi / 1.4e-6, 800)
        magnitude = np.abs(mismatch(wavenumbers))   # scipy.special ufuncs: one vectorised scan
        minima = np.flatnonzero((magnitude[1:-1] < magnitude[:-2]) & (magnitude[1:-1] < magnitude[2:])) + 1
        for position in minima:
            try:
                root = complex(newton(mismatch, complex(wavenumbers[position]), tol=1e-12 * wavenumbers[position],
                                      maxiter=200))
            except (RuntimeError, OverflowError, ZeroDivisionError):
                continue
            if abs(mismatch(root)) < 1e-6 and abs(root.imag) < 1e-3 * root.real:   # high-Q roots
                roots.append(2.0 * math.pi / root.real)
    return sorted(set(round(value, 15) for value in roots))


def _check_disk(results_by_name):
    """Per polarisation: spread of the smoothed pair wavelength over the offsets and its distance to the nearest
    closed-form resonance; measured = error = the worst in tolerance units."""
    errors, tolerances, details = {}, {}, {}
    for polarization in ('ez', 'hz'):
        closed_form = _closed_form_wavelengths(polarization)
        pairs = [_pair_wavelengths(results_by_name[f'{polarization}_{index}']) for index in range(len(_OFFSETS))]
        means = np.array([np.mean(pair) for pair in pairs])
        nearest = min(closed_form, key=lambda value: abs(value - means.mean())) if closed_form else math.nan
        summary = {'pair_wavelengths': [float_list(pair) for pair in pairs],
                   'spread': float((means.max() - means.min()) / means.mean()),
                   'pair_split': float(max(abs(pair[0] - pair[1]) / np.mean(pair) for pair in pairs)),
                   'closed_form': nearest,
                   'closed_form_error': float(abs(means.mean() / nearest - 1.0))}
        details[polarization] = summary
        errors[f'{polarization} spread'] = summary['spread']
        tolerances[f'{polarization} spread'] = _SPREAD_TOLERANCE[polarization]
        errors[f'{polarization} closed form'] = summary['closed_form_error']
        tolerances[f'{polarization} closed form'] = _CLOSED_FORM_TOLERANCE
    normalized = {item: error / tolerances[item] for item, error in errors.items()}
    worst_item = max(normalized, key=lambda item: math.inf if math.isnan(normalized[item]) else normalized[item])
    details.update(errors=errors, worst_item=worst_item)
    return {'measured': normalized[worst_item], 'expected': 0.0, 'error': normalized[worst_item], 'tolerance': 1.0,
            'details': details}


SUBPIXEL_SELFTEST_CASES = (
    SelfTestCase(
        name='subpixel_smoothing_cpu_gpu',
        label='Subpixel smoothing: interface geometry and averaged permittivity, CPU backend vs GPU',
        description='Builds (no run) on solver_defaults.backend cpu (numpy interface-geometry build, the numpy twins of '
                    'the averaging kernels) and, with a usable GPU, on the gpu, each with all sub-checks (prefixed '
                    'cpu_ / gpu_): a glass disk (eps 4, r 0.389 um) and a silicon slab (eps 12, full width, '
                    'edges off the cell edges) in vacuum, 20 nm cells, solver_defaults.subpixel_smoothing on, for '
                    'em_fdtd_ez / _hz and em_fdfd_ez / _hz. Checks: the engines\' averaged eps at every interface '
                    'location equals 1 / (n^2 <1/eps> + (1 - n^2) / <eps>) of its two representative cells '
                    f'({_RELATIVE_TOLERANCE:g} relative), also after eps_xx / eps_yy slot writes (re-averaged '
                    'live); the averaged E_z cells give the disk area and the slab thickness (1e-3 / 2e-3); n_d^2 is '
                    f'0 / 1 on the straight slab edges ({_STRAIGHT_NORMAL_TOLERANCE:g}) and the radial value at the '
                    f'disk faces (mean {_DISK_NORMAL_MEAN:g}, max {_DISK_NORMAL_MAX:g}); a mode-port line through '
                    'the slab sees the engine\'s averaged cells / faces; the FDFD assembly equals the FDTD values on '
                    'the window; em_modes (no solve) E_x / E_y faces follow the formula with the other in-plane normal '
                    'component and its E_z cells equal the E_z FDTD ones (also after an eps_r slot write). em_fdtd_3d '
                    f'(a silicon slab with z faces off the cell edges under a glass pillar, {_SIZE_3D * 1e6:g} um box '
                    'of 25 nm cells): every '
                    'interface face eps the E update uses equals the formula within 1e-5 plus the face re-quantisation '
                    'error (also after an eps_r slot write and rebuild), pillar sidewall n^2 radial, the slab '
                    'thickness from the E_x (arithmetic) and E_z (harmonic) faces of one column within '
                    f'{_Z_THICKNESS_TOLERANCE:g}, a pillar layer equals the em_modes smoothed cross-section and a z '
                    'mode port\'s point materials. With both backends: the interface geometry (2D and 3D) has the '
                    'same locations and representative / face-mean cells, fill, n_d^2 and '
                    f'face shares within {_GEOMETRY_TOLERANCE:g}, and the 2D averaged window eps (E_z cells, H_z '
                    f'faces) agree within {_RELATIVE_TOLERANCE:g} relative. Without a usable GPU only the cpu '
                    'sub-checks run. error = number of failed sub-checks, tolerance 0. The 1.0 um 3D box not yet '
                    'confirmed by a run.',
        specs=lambda: {},
        check=_check_cpu_gpu,
        max_wall_time=480.0, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='subpixel_smoothing_disk_resonance',
        label='Subpixel smoothing: dielectric-disk resonance on a coarse mesh',
        description=f'eigenfrequency study (em_fdfd_ez and em_fdfd_hz, target {_DISK_TARGET * 1e6:g} um, '
                    f'{_DISK_RESONANCES} resonances) of a silicon disk (eps {_DISK_EPS_R:g}, r 1 um) in a '
                    f'{_DISK_DOMAIN * 1e6:g} um CPML box of {_DISK_CELL * 1e9:g} nm cells, the centre shifted by '
                    '(0, 0), (0.31, 0.17) and (0.57, 0.43) cells, subpixel smoothing. The mean wavelength of the '
                    'highest-Q pair (whispering-gallery cos / sin modes) must not move with the shift: '
                    f'(max - min) / mean <= {_SPREAD_TOLERANCE["ez"]:g} (E_z) / {_SPREAD_TOLERANCE["hz"]:g} (H_z, '
                    'diagonal averaging; the staircase gives ~7e-4), and lie within '
                    f'{_CLOSED_FORM_TOLERANCE:g} of the nearest closed-form resonance (roots of p J_m\'/J_m = '
                    'H2_m\'/H2_m, p = n or 1/n; the margin is the FDFD numerical dispersion). The pair splitting and '
                    'closed-form errors are in the details. measured = error = the worst item in tolerance units, '
                    'tolerance 1. Needs scipy. Box size and resonance count not yet confirmed by a run.',
        specs=_disk_specs,
        check=_check_disk,
        heavy=True),
)

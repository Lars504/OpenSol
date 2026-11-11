"""Model features of em_fdtd_3d for plane waves, mode ports and far fields (B2 part 2): params, apply, results,
validation and memory estimates, plus the mesh placement checks definition.py shares; it appends SOURCE_FEATURES."""
import numpy as np

from ...core.constants import FACES_3D
from ...core.domain import containing_cell
from ...core.memory_estimate import COMPLEX128_BYTES, FLOAT32_BYTES, MemoryEstimate, host_lu_bytes
from ...core.mirror import MIRROR_SIGNS, plane_wave_mirror_issue
from ...model.definition_helpers import wavelength_range_issues
from ...model.schema import FeatureType, Issue, Param, feature_path, resolve_values
from ..em.cross_section_modes import size_issue_for
from ..em.definition_common import (_monitor_wavelengths, object_result, single_object, source_time_kwargs,
                                    spectral_params, waveform_params)
from ...solvers.sparse_direct import scipy_available
from .boundaries3d import AXES, MIRROR_TYPES_3D
from .mode_ports3d import MODE_DIRECTIONS, mode_monitor_result_3d, mode_source_result_3d
from .near_to_far3d import CUT_PLANES, FAR_FIELD_QUANTITY, far_field_result_3d
from .tfsf3d import PLANE_WAVE_DIRECTIONS

_EDGE_TOLERANCE = 1e-9   # relative to the domain size: a coordinate this close to an edge lies on it
# stepper precisions of the TFSF / mode-port features: their kernels add float32 corrections in physical units
_FLOAT32_FIELDS = ('single',)

# ---- params ----
_PLANE_WAVE_PARAMS = (
    Param('direction', 'choice', '+z', label='Direction', choices=PLANE_WAVE_DIRECTIONS,
          choice_labels=tuple((name, f'Toward {name}') for name in PLANE_WAVE_DIRECTIONS),
          tip='Direction in which the plane wave travels.',
          help='Propagation direction of the plane wave along one axis (normal incidence on the box faces); it may '
               'not run along a periodic axis.'),
    Param('polarization', 'choice', 'x', label='Electric field direction', choices=AXES,
          choice_labels=(('x', 'Along x'), ('y', 'Along y'), ('z', 'Along z')),
          tip='Direction of the electric field of the wave. It must be across the direction of travel.',
          help='Axis of the incident E (linear polarisation), perpendicular to the propagation axis; H = k x E / '
               '(eta) lies along the third axis.'),
    *waveform_params(),
    Param('amplitude', 'float', 1.0, label='Amplitude', unit='V/m',
          tip='Peak electric field of the incident wave. Fields scale linearly with it; 1 is typical.',
          help='Peak incident E [V/m] of the waveform envelope in the background medium (exact at the centre '
               'wavelength: the 1D incident line is calibrated there); the cw intensity is amplitude^2 / (2 eta), '
               'eta = eta0 sqrt(mu_r / eps_r) of the background.'),
)

_NORMAL_PARAM = Param('normal', 'choice', 'z', label='Propagation axis', choices=AXES,
                      choice_labels=(('x', 'Along x'), ('y', 'Along y'), ('z', 'Along z')),
                      tip='Axis along which the guide runs; the port plane is perpendicular to it.',
                      help='Normal of the port plane (the waveguide axis); the plane spans the whole grid '
                           'cross-section (PML included) and must cut a straight guide.')
_POSITION_PARAM = Param('position', 'float', 0.0, label='Plane position', unit='m',
                        tip='Coordinate of the port plane along its axis, from the lower domain edge.',
                        help='Coordinate [m] of the plane along the normal: a source takes the cell containing it '
                             '(its first total-field layer), a monitor the nearest cell edge.')
_MODE_NUMBER_HELP = ('1-based mode index in descending Re(beta^2) of the cross-section solve (1 = the fundamental, most '
                     'strongly guided mode); a symmetric guide has near-degenerate pairs (square cores, fibres) that '
                     'come back as arbitrary mixtures: make the cross-section asymmetric to select one.')
_MODE_SOURCE_PARAMS = (
    _NORMAL_PARAM,
    _POSITION_PARAM,
    Param('mode_number', 'int', 1, label='Mode number', min=1,
          tip='Which guided mode to launch: 1 is the fundamental, most strongly guided mode, 2 the next, and so on.',
          help=f'{_MODE_NUMBER_HELP} The source result table lists the first max(mode_number + 2, 4) modes.'),
    Param('direction', 'choice', 'positive', label='Direction', choices=MODE_DIRECTIONS,
          choice_labels=(('positive', 'Toward + along the axis'), ('negative', 'Toward - along the axis')),
          tip='Direction in which the mode travels away from the plane. Practically nothing goes the other way.',
          help='positive launches toward +normal (total field from the plane cell downstream), negative toward '
               '-normal; one-sided at the source wavelength.'),
    Param('waveform', 'choice', 'gaussian', label='Waveform', choices=('gaussian', 'cw'),
          choice_labels=(('gaussian', 'Gaussian pulse'), ('cw', 'Continuous wave')),
          tip='Time shape of the source: a short pulse for spectra, or one steady wavelength.',
          help='gaussian (sin carrier under a gaussian envelope) or cw; the complex mode profile is driven by the '
               'carrier and its quadrature (cos) partner, so ricker is not offered.'),
    Param('wavelength', 'float', 1.55e-6, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          tip='Vacuum wavelength of the source, entered in metres (1.55e-6 for 1550 nm). The mode is computed at '
              'this wavelength only.',
          help='Vacuum wavelength lambda0 = C0 / f0 of the carrier, the only wavelength at which the mode is solved '
               '(profile, n_eff): the launch is exact and one-sided there.'),
    Param('bandwidth', 'float', 0.3, label='Relative bandwidth', unit='1', min=0.0, min_exclusive=True,
          visible_if=(('waveform', ('gaussian',)),),
          tip='Spectral width of the pulse as a fraction of the centre frequency; 0.1 to 0.3 is typical.',
          help='Spectral width / f0 of the gaussian pulse; away from f0 the dispersing guide makes the launch leak '
               'backward and into radiation, and above 0.6 the build warns (quadrature partner, DC part).'),
    Param('power', 'float', 1.0, label='Mode power', unit='W', min=0.0, min_exclusive=True,
          tip='Power carried by the launched mode. Fields grow with its square root, so 1 is the usual choice.',
          help='Mode power [W] as a flux plane on the guide reads it at the source wavelength (the discrete Yee '
               'flux; cw: time-averaged after the ramp, gaussian: at the envelope peak).'),
)
_MODE_MONITOR_PARAMS = (
    _NORMAL_PARAM,
    _POSITION_PARAM,
    Param('mode_number', 'int', 1, label='Mode number', min=1,
          tip='Which guided mode to measure: 1 is the fundamental, most strongly guided mode.',
          help=f'{_MODE_NUMBER_HELP} It picks the mode at the middle wavelength; the others follow it by overlap '
               f'(a build warning lists wavelengths where that overlap stays below 0.9).'),
) + spectral_params(5, 'Each wavelength costs one cross-section mode solve at build (host scipy) and four complex128 '
                       'window planes; 3-11 typical, so the default is 5 (21 for the 2D mode_monitor).',
                    'Each wavelength adds a mode calculation at the start; 3 to 11 is typical.')
_FAR_FIELD_PARAMS = spectral_params(5, 'Each wavelength stores four complex128 spectra on the six box faces (64 B '
                                       'per face cell), so the default is 5 (21 for the 2D far_field).',
                                    'A few wavelengths are typical; each one adds memory on the box faces.') + (
    Param('cut_plane', 'choice', 'xz', label='Cut plane', choices=tuple(CUT_PLANES),
          choice_labels=(('xz', 'x-z plane'), ('yz', 'y-z plane'), ('xy', 'x-y plane')),
          tip='Plane through the box centre in which the radiation pattern is shown.',
          help='Plane of the observation directions: angles run counter-clockwise from its first axis (+x for xz '
               'and xy, +y for yz) toward its second (+z, +z, +y).'),
    Param('n_angles', 'int', 360, label='Number of angles', min=1,
          tip='Number of directions, evenly spaced over the full circle of the cut plane; 360 gives 1-degree steps.',
          help='Directions evenly spaced over 360 degrees in the cut plane.'),
)

# ---- descriptions ----
_PLANE_WAVE_TEXT = ('Total-field / scattered-field plane wave along one axis on a box object: total field inside the '
                    'box (incident + scattered), only the scattered field outside, fed by a 1D incident line in the '
                    'background medium (read from the cell before the upstream face), on uniform or graded meshes (the '
                    'line runs on the grid\'s own cell widths along the propagation axis, so grading leaks nothing). '
                    'Every box face needs one scattered-field cell inside the domain, '
                    'except: on a periodic axis the box spans the whole period (its extent there is ignored), and with '
                    'both transverse axes periodic a downstream face on the domain edge opens the box into the CPML '
                    '(a one-sided plane-wave source: everything downstream is total field, upstream only what comes '
                    'back), and a transverse face may reach a mirror face about which the incident wave is even '
                    '(symmetric when the plane is normal to the third axis, antisymmetric when normal to the '
                    'polarization; the other parity is an error): the box opens there and its image completes it. '
                    'Keep the box faces in one uniform lossless medium (the build warns otherwise).')
_MODE_SOURCE_TEXT = ('Guided mode of the cross-section (full-vector A5 solve with scipy at build, on the whole grid '
                     'layer containing the position, PML included, at the FDTD numerical frequency) injected as a '
                     'one-sided total-field / scattered-field plane: total field from that layer downstream. The '
                     'mode is solved once before the first step; later material changes keep it. pec / pmc domain '
                     'faces are approximate for modes that reach them. Result \'table\' at the source wavelength: '
                     'mode_number, n_eff, n_eff_imag, loss_db_per_cm, te_fraction, effective_area [m^2], '
                     'pml_fraction, selected; extras wavelength, power, direction, mode_number.')
_MODE_MONITOR_TEXT = ("Mode decomposition on the cell-edge plane nearest the position across the physical window: "
                      "running DFTs of the tangential E (averaged across the plane) and H, projected on the "
                      "cross-section mode solved per wavelength at build. Result 'xy' versus wavelength: "
                      'power_positive / power_negative (a.u., the flux-plane convention, so comparable with a flux '
                      'plane), power_per_cw_* [W] when every source shares one waveform, fraction_* = mode power / '
                      'source power with exactly one mode source (transmission downstream, reflection upstream; NaN '
                      'where its spectrum is below 1e-6 of the peak); extras n_eff (real part), n_eff_imag, mode_number, '
                      'normalized.')
_FAR_FIELD_TEXT = ('Near-to-far-field transform of the tangential E and H on the six faces of a closed box object '
                   '(each face snapped to the nearest cell edge; the box must enclose every source and scatterer, '
                   'with vacuum outside). Result \'far_field\' in one cut plane: angles [rad] from its first axis, '
                   'power (n_f, n_angles) = radiated power per steradian of the dt-weighted spectra (a.u., no 1/2), '
                   'power_per_cw [W/sr] when every source shares one waveform (its integral over the sphere is the '
                   'flux box power_per_cw / cos(omega dt / 2): the flux spectra carry that re-timing factor, the far '
                   'field does not; ~0.3 % at 25 cells per wavelength), quantity \'' + FAR_FIELD_QUANTITY + '\', extras cut_plane, angle_axes. '
                   'Mirror faces: the pattern of the unfolded structure (the images of the box currents are added '
                   'and a box face on a mirror plane is dropped, so the box and its images must enclose the '
                   'unfolded sources; at most one mirror per axis).')


# ---- apply ----
def box_corners(ctx, selection, feature_spec):
    """(lo, hi) corners [m] of the selected box object (definition.py flux boxes too)."""
    shape = ctx.resolved_geometry[single_object(selection, feature_spec, 'box')].values
    return (shape['x0'], shape['y0'], shape['z0']), (shape['x1'], shape['y1'], shape['z1'])


def _apply_plane_wave(ctx, solver, feature_spec, values, selection):
    lo, hi = box_corners(ctx, selection, feature_spec)
    return solver.add_plane_wave(lo, hi, direction=values['direction'], polarization=values['polarization'],
                                 amplitude=values['amplitude'], **source_time_kwargs(values))


def _apply_mode_source(ctx, solver, feature_spec, values, selection):
    return solver.add_mode_source(values['normal'], values['position'], direction=values['direction'],
                                  mode_number=values['mode_number'], power=values['power'],
                                  **source_time_kwargs(values))


def _apply_mode_monitor(ctx, solver, feature_spec, values, selection):
    return solver.add_mode_monitor(values['normal'], values['position'], mode_number=values['mode_number'],
                                   wavelengths=_monitor_wavelengths(values))


def _apply_far_field(ctx, solver, feature_spec, values, selection):
    lo, hi = box_corners(ctx, selection, feature_spec)
    return solver.add_near_to_far(lo, hi, wavelengths=_monitor_wavelengths(values))


def _far_field_result(session, solver, monitor, feature_spec, options):
    values = resolve_values(_FAR_FIELD_PARAMS, feature_spec.params, session.parameters)
    return far_field_result_3d(monitor, solver, values['cut_plane'], values['n_angles'])


# ---- validation (CPU) ----
def domain_sizes(vctx):
    """(size_x, size_y, size_z) [m] or None."""
    if vctx.domain_size is None or vctx.size_z is None:
        return None
    return (*vctx.domain_size, vctx.size_z)


def face_types(physics_spec):
    """{face: bc type} of the six faces from the raw settings ('cpml' when unset or the settings are malformed)."""
    settings = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
    return {face: settings.get(f'bc_{face}', 'cpml') for face in FACES_3D}


def _periodic_axes(physics_spec):
    faces = face_types(physics_spec)
    return {axis: faces[f'{axis}_lo'] == 'periodic' for axis in AXES}


def window_axes(mesh):
    """Per axis x, y, z: (full-grid cell edges [m] in the physical frame as Domain3D.cell_edges, first and last window
    cell); None without a 3D mesh estimate."""
    if mesh is None or getattr(mesh, 'z_edges', None) is None:
        return None
    axes = []
    for axis, edges in zip(AXES, (mesh.x_edges, mesh.y_edges, mesh.z_edges)):
        edges = np.asarray(edges, dtype=np.float64)
        axes.append((edges, int(mesh.npml_sides.get(f'{axis}_lo', 0)),
                     edges.size - 2 - int(mesh.npml_sides.get(f'{axis}_hi', 0))))
    return axes


def nearest_window_edge(axis_data, coordinate):
    """Grid edge index nearest `coordinate` [m] among the window's cell edges (as EMPhysics3D._nearest_edge)."""
    edges, first, last = axis_data
    return first + int(np.argmin(np.abs(edges[first:last + 2] - float(coordinate))))


def plane_edge_message(axis_data, edge, axis):
    """Problem text when a PlaneFlux3D at grid edge `edge` has no cell on one side (a face without CPML), else None."""
    edges, first, last = axis_data
    if 1 <= edge <= edges.size - 2:
        return None
    inside = edges[first + 1] if edge == 0 else edges[last]
    return (f'snaps to the domain edge {axis} = {edges[edge]:.4g} m of a face without CPML, where it has no cell on the '
            f'outer side; move it at least one cell inside (to {axis} = {inside:.4g} m)')


def box_edge_issues(vctx, path, feature_spec, label):
    """Errors of the selected boxes whose faces snap to one cell edge or onto a face without CPML (FluxBox3D)."""
    axes = window_axes(vctx.mesh)
    if axes is None:
        return []
    issues = []
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is None:
            continue
        for axis, axis_data in zip(AXES, axes):
            low, high = sorted((shape.values[f'{axis}0'], shape.values[f'{axis}1']))
            first, last = nearest_window_edge(axis_data, low), nearest_window_edge(axis_data, high)
            if last <= first:
                issues.append(Issue('error', path, f"{label} box '{name}' spans no cell along {axis}: both faces snap "
                                                   f'to the same cell edge'))
                continue
            for edge in (first, last):
                message = plane_edge_message(axis_data, edge, axis)
                if message is not None:
                    issues.append(Issue('error', path, f"{label} box '{name}': a face {message}"))
    return issues


def _plane_wave_edge_issues(vctx, path, name, shape, axis_d, sign, periodic, faces, polarization):
    """EMPhysics3D.add_plane_wave's box rules on the mesh: every closed face one scattered-field cell inside the
    window, a downstream face on the window edge only when one-sided (both transverse axes periodic), a face on a
    mirror plane only of the wave's parity (core.mirror.plane_wave_mirror_issue; the box opens there)."""
    axes = window_axes(vctx.mesh)
    one_sided = all(periodic[axis] for axis in AXES if axis != axis_d)
    issues = []
    for axis, axis_data in zip(AXES, axes):
        if periodic[axis]:
            continue
        _, start, last_cell = axis_data
        stop = last_cell + 1
        low, high = sorted((shape.values[f'{axis}0'], shape.values[f'{axis}1']))
        first, last = nearest_window_edge(axis_data, low), nearest_window_edge(axis_data, high)
        if last <= first:
            issues.append(Issue('error', path, f"box '{name}' spans no cell along {axis}: both faces snap to the same "
                                               f'cell edge'))
            continue
        open_first = axis == axis_d and one_sided and sign < 0 and first == start
        open_last = axis == axis_d and one_sided and sign > 0 and last == stop
        for side, edge, window_edge in (('lo', first, start), ('hi', last, stop)):
            face = f'{axis}_{side}'
            mirror_sign = MIRROR_SIGNS.get(faces[face])
            if edge != window_edge or mirror_sign is None or (open_first if side == 'lo' else open_last):
                continue
            problem = plane_wave_mirror_issue(face, mirror_sign, axis_d, polarization)
            if problem is not None:
                issues.append(Issue('error', path, f"box '{name}' {problem}"))
                return issues
            issues.append(Issue('info', path, f"box '{name}' opens onto the {face} mirror plane: its image completes "
                                              'the box there'))
            open_first, open_last = (True, open_last) if side == 'lo' else (open_first, True)
        if (not open_first and first < start + 1) or (not open_last and last > stop - 1):
            issues.append(Issue('error', path, f"box '{name}' has a face along {axis} on the outermost window cell "
                                               f'edge: every box face needs one scattered-field cell inside the domain '
                                               f'(a downstream face may reach the edge only with both transverse axes '
                                               f"periodic, a transverse face a mirror plane of the wave's parity)"))
    return issues


def _validate_plane_wave(vctx, physics_spec, feature_spec, values):
    path = feature_path(physics_spec, feature_spec)
    axis_d = values['direction'][1]
    sign = 1 if values['direction'][0] == '+' else -1
    issues = []
    if values['polarization'] == axis_d:
        issues.append(Issue('error', path, f"polarization {axis_d} lies along the propagation direction: the field of "
                                           f"a plane wave is transverse, pick another axis"))
    periodic = _periodic_axes(physics_spec)
    if periodic[axis_d]:
        issues.append(Issue('error', path, f'the plane wave travels along the periodic axis {axis_d}; it needs '
                                           f'absorbing (or wall) faces along its direction'))
        return issues
    sizes = domain_sizes(vctx)
    one_sided = all(periodic[axis] for axis in AXES if axis != axis_d)
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is None or sizes is None:
            continue
        if window_axes(vctx.mesh) is not None:   # the build's cell-edge snapping
            issues.extend(_plane_wave_edge_issues(vctx, path, name, shape, axis_d, sign, periodic,
                                                  face_types(physics_spec), values['polarization']))
            continue
        faces = face_types(physics_spec)
        for axis, size in zip(AXES, sizes):
            if periodic[axis]:
                continue
            low, high = sorted((shape.values[f'{axis}0'], shape.values[f'{axis}1']))
            tolerance = _EDGE_TOLERANCE * size
            downstream_face = (f'{axis}_hi' if sign > 0 else f'{axis}_lo') if axis == axis_d and one_sided else None
            for face, on_edge in ((f'{axis}_lo', low <= tolerance), (f'{axis}_hi', high >= size - tolerance)):
                mirror_sign = MIRROR_SIGNS.get(faces[face])
                if not on_edge or face == downstream_face or (
                        mirror_sign is not None
                        and plane_wave_mirror_issue(face, mirror_sign, axis_d, values['polarization']) is None):
                    continue
                issues.append(Issue('error', path, f"box '{name}' reaches the domain edge {face}: every box face "
                                                   f'needs one scattered-field cell inside the domain (a downstream '
                                                   f'face may reach it only with both transverse axes periodic, a '
                                                   f"transverse face a mirror plane of the wave's parity)"))
    return issues


def _validate_mode_feature(vctx, physics_spec, feature_spec, values):
    path = feature_path(physics_spec, feature_spec)
    issues = wavelength_range_issues(values, path) if 'wavelength_start' in values else []
    if not scipy_available():
        issues.append(Issue('error', path, 'mode ports solve the cross-section modes with scipy on the host: pip '
                                           'install scipy'))
    axis = AXES.index(values['normal'])
    if _periodic_axes(physics_spec)[values['normal']]:
        issues.append(Issue('error', path, f"the port normal {values['normal']} is a periodic axis: a mode port needs "
                                           f'a guide ending in absorbing faces'))
    sizes = domain_sizes(vctx)
    if sizes is not None and not 0.0 <= values['position'] <= sizes[axis]:
        issues.append(Issue('error', path, f"position {values['position']:.6g} m lies outside the domain along "
                                           f"{values['normal']} [0, {sizes[axis]:.6g}] m"))
    elif window_axes(vctx.mesh) is not None:   # the build's placement rules
        axis_data = window_axes(vctx.mesh)[axis]
        if 'direction' in values:   # mode source: its interface face needs a scattered-field cell upstream
            edges, first, last = axis_data
            cell = containing_cell(values['position'], edges, first, last)
            if (values['direction'] == 'positive' and cell <= first) or (values['direction'] == 'negative'
                                                                         and cell >= last):
                issues.append(Issue('error', path, f"position {values['position']:.6g} m lies in the outermost window "
                                                   f"cell along {values['normal']}: the interface face of a "
                                                   f"{values['direction']} mode source would sit on the window edge "
                                                   f'(in the CPML or on the wall); move it at least one cell inside'))
        else:
            message = plane_edge_message(axis_data, nearest_window_edge(axis_data, values['position']),
                                         values['normal'])
            if message is not None:
                issues.append(Issue('error', path, f'the monitor plane {message}'))
    if vctx.mesh is not None:
        counts = (int(vctx.mesh.nx), int(vctx.mesh.ny), int(vctx.mesh.nz))
        issue = size_issue_for(counts[(axis + 1) % 3], counts[(axis + 2) % 3])
        if issue is not None:
            issues.append(Issue(issue[0], path, issue[1]))
    return issues


def _validate_far_field(vctx, physics_spec, feature_spec, values):
    """Wavelength range, box faces (FluxBox3D snapping) and mirrors on both faces of one axis (the image unfolding of
    NearToFar3D needs at most one per axis)."""
    path = feature_path(physics_spec, feature_spec)
    issues = wavelength_range_issues(values, path) + box_edge_issues(vctx, path, feature_spec, 'far-field')
    mirrors = [face for face, kind in face_types(physics_spec).items() if kind in MIRROR_TYPES_3D]
    for axis in AXES:
        if f'{axis}_lo' in mirrors and f'{axis}_hi' in mirrors:
            issues.append(Issue('error', path, f'mirror faces {axis}_lo and {axis}_hi: the far field unfolds a mirror '
                                               'by its image, and two parallel mirrors have infinitely many; make one '
                                               'of them cpml'))
    periodic = [axis for axis, flag in _periodic_axes(physics_spec).items() if flag]
    if periodic:
        issues.append(Issue('error', path, f"axis {', '.join(periodic)} is periodic: the far field transforms the box "
                                           'currents into free space, but on a periodic axis they include the '
                                           'neighbour cells (no array factor, no Floquet orders); make the faces cpml'))
    return issues


# ---- memory (E8) ----
def plane_cells(mesh, normal_axis, window):
    """Cells of a plane normal to normal_axis: the physical window's (window=True) or the whole grid's."""
    counts = [int(mesh.nx), int(mesh.ny), int(mesh.nz)]
    if window:
        sides = mesh.npml_sides
        for index, axis in enumerate(AXES):
            counts[index] -= sides.get(f'{axis}_lo', 0) + sides.get(f'{axis}_hi', 0)
    return counts[(normal_axis + 1) % 3] * counts[(normal_axis + 2) % 3]


def _plane_wave_memory(values, n_selected_cells, mesh):
    cells = max(int(mesh.nx), int(mesh.ny), int(mesh.nz)) + 40
    return MemoryEstimate.from_items((('incident line (fields + per-node coefficients, 4 x float32)',
                                       4 * FLOAT32_BYTES * cells),))


def _mode_source_memory(values, n_selected_cells, mesh):
    """8 incident planes (re / im of 4 corrections) + 3 live coefficient planes; the host factor of the solve."""
    cells = plane_cells(mesh, AXES.index(values['normal']), window=False)
    return MemoryEstimate.from_items((('mode source planes (11 x float32)', 11 * FLOAT32_BYTES * cells),),
                                     build_extra=4 * FLOAT32_BYTES * cells,
                                     host_items=(('cross-section mode solve (SuperLU factor)', host_lu_bytes(2 * cells)),))


def _mode_monitor_memory(values, n_selected_cells, mesh):
    axis = AXES.index(values['normal'])
    n_wavelengths = int(values['n_wavelengths'])
    full = plane_cells(mesh, axis, window=False)
    return MemoryEstimate.from_items(
        ((f'plane spectra (4 x {n_wavelengths} wavelengths x complex128)',
          4 * n_wavelengths * COMPLEX128_BYTES * plane_cells(mesh, axis, window=True)),),
        host_items=(('cross-section mode solves (SuperLU factor, one at a time)', host_lu_bytes(2 * full)),
                    (f'followed modes ({n_wavelengths} x 8 complex128 planes)', n_wavelengths * 8 * COMPLEX128_BYTES
                     * full)))


def box_face_spectra_memory(values, n_selected_cells, mesh):
    """Six faces of 4 complex128 spectra (flux box, far field); the face cells of a cube with the selection's 3D box
    cell count."""
    n_wavelengths = int(values['n_wavelengths'])
    side = max(int(n_selected_cells), 1) ** (1.0 / 3.0)
    cells = 6 * side * side
    return MemoryEstimate.from_items(((f'box face spectra (4 x {n_wavelengths} wavelengths x complex128, approximate)',
                                       int(4 * n_wavelengths * COMPLEX128_BYTES * cells)),))


SOURCE_FEATURES = (
    FeatureType('plane_wave_source', 'Plane wave', 'source', 'domain', params=_PLANE_WAVE_PARAMS, object_kinds=('box',),
                max_entities=1, description=_PLANE_WAVE_TEXT,
                summary='A plane wave filling a box, for scattering by objects inside it or reflection from a '
                        'layer.',
                apply=_apply_plane_wave, validate=_validate_plane_wave, memory_estimate=_plane_wave_memory,
                precisions=_FLOAT32_FIELDS),
    FeatureType('mode_source', 'Mode source', 'source', None, params=_MODE_SOURCE_PARAMS, result_kind='table',
                description=_MODE_SOURCE_TEXT,
                summary='Launches a guided mode of a waveguide from a plane across it.',
                apply=_apply_mode_source, result=object_result(mode_source_result_3d),
                validate=_validate_mode_feature, memory_estimate=_mode_source_memory, precisions=_FLOAT32_FIELDS),
    FeatureType('mode_monitor', 'Mode monitor', 'monitor', None, params=_MODE_MONITOR_PARAMS, result_kind='xy',
                description=_MODE_MONITOR_TEXT,
                summary='Measures how much power travels in one guided mode, in each direction, per wavelength.',
                apply=_apply_mode_monitor, result=object_result(mode_monitor_result_3d, with_solver=True),
                validate=_validate_mode_feature, memory_estimate=_mode_monitor_memory, precisions=_FLOAT32_FIELDS),
    FeatureType('far_field', 'Far-field pattern', 'monitor', 'domain', params=_FAR_FIELD_PARAMS, object_kinds=('box',),
                max_entities=1, result_kind='far_field', description=_FAR_FIELD_TEXT,
                summary='Computes the power radiated in each direction of a plane from the fields on a box around the '
                        'source.',
                apply=_apply_far_field, result=_far_field_result, validate=_validate_far_field,
                memory_estimate=box_face_spectra_memory),
)

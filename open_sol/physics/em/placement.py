"""Cell geometry of EM sources and monitors on a Domain (host numpy, no cupy): boxes, port, flux and sheet lines, probe
cells and dipole faces, plus the per-polarisation vocabulary, shared by EMPhysics and the FDFD physics."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.mirror import MIRROR_SIGNS, plane_wave_mirror_issue, touch_factor
from .yee_grid import grid_cell_widths, yee_inverse_spacings

# Per polarisation: the MaterialGrid permittivity arrays solvers read, the probe components (out-of-plane first).
PERMITTIVITY_ARRAYS = {'ez': ('eps_r',), 'hz': ('eps_xx', 'eps_yy')}
# Per polarisation: inject_source slot -> the permittivity array it overwrites ('<array>_update').
PERMITTIVITY_SLOTS = {polarization: {f'{array}_update': array for array in arrays}
                      for polarization, arrays in PERMITTIVITY_ARRAYS.items()}
PROBE_COMPONENTS = {'ez': ('ez', 'hx', 'hy'), 'hz': ('hz', 'ex', 'ey')}


def require_polarization(polarization, required, method):
    """ValueError when source `method`, which exists only for `required` ('ez' | 'hz'), is called on the other one."""
    if polarization == required:
        return
    if required == 'hz':
        raise ValueError(f"{method} is an H_z source; on an ez physics use add_point_source (E_z filament current)")
    raise ValueError(f"{method} is the E_z filament current; on an hz physics use add_magnetic_line_current or "
                     f"add_electric_dipole")


def probe_cell(domain, polarization, x, y, field_component):
    """(ix, iy, component) of a field probe at (x, y); component None = the out-of-plane field."""
    components = PROBE_COMPONENTS[polarization]
    if field_component is None:
        field_component = components[0]
    if field_component not in components:
        raise ValueError(f"field probe component of an {polarization!r} physics must be one of {components}, "
                         f"got {field_component!r}")
    domain.validate_physical_pos(x, y, "Field probe")
    ix, iy = domain.pos_to_cell(x, y)
    return ix, iy, field_component


def point_source_cell(domain, x, y, label):
    """(ix, iy, Yee cell area dx_i dy_j [m^2] from grid_cell_widths) of a point-like source at (x, y)."""
    domain.validate_physical_pos(x, y, label)
    ix, iy = domain.pos_to_cell(x, y)
    dx_cells, dy_cells = grid_cell_widths(domain.grid)
    return ix, iy, float(dx_cells[ix] * dy_cells[iy])


def flux_line_cells(domain, axis, position, span):
    """(normal axis = flux direction, line cell, half-open span cells, full-grid float64 widths along the line [m]) of a
    flux monitor; axis 'x' = line along x at y = position (as sources and ports), span = (start, end) along it."""
    if span is None or len(span) != 2:
        raise ValueError(f"Flux monitor needs span = (start, end) along the line [m], got {span!r}")
    lo, hi = sorted((float(span[0]), float(span[1])))
    dx_cells, dy_cells = domain.cell_widths()
    if axis == 'x':
        domain.validate_physical_pos(lo, position, "Flux monitor start")
        domain.validate_physical_pos(hi, position, "Flux monitor end")
        return 'y', domain.pos_to_cell_y(position), (domain.pos_to_cell_x(lo), domain.pos_to_cell_x(hi) + 1), dx_cells
    if axis == 'y':
        domain.validate_physical_pos(position, lo, "Flux monitor start")
        domain.validate_physical_pos(position, hi, "Flux monitor end")
        return 'x', domain.pos_to_cell_x(position), (domain.pos_to_cell_y(lo), domain.pos_to_cell_y(hi) + 1), dy_cells
    raise ValueError(f"Flux monitor axis (along the line) must be 'x' or 'y', got {axis!r}")


def mirror_planes(domain, boundary_config):
    """{face: (plane coordinate [m], E parity sign +1 symmetric / -1 antisymmetric)} of the mirror faces of a 2D
    boundary config (BoundaryConfig.ghost_faces): each plane is the window edge."""
    planes = {}
    window = domain.physical_window()
    edges = domain.cell_edges()
    for face in getattr(boundary_config, 'ghost_faces', ()):
        axis = 'xy'.index(face[0])
        index = window[axis].start if face.endswith('lo') else window[axis].stop
        planes[face] = (float(edges[axis][index]), MIRROR_SIGNS[getattr(boundary_config, face).value])
    return planes


def line_mirror_factor(domain, boundary_config, normal_axis, span):
    """Unfolded-total factor of a flux line (normal_axis = flux direction, half-open span cells along the line): 2 per
    mirror face of the along axis whose window edge cell the span covers (core.mirror.touch_factor)."""
    along = 'y' if normal_axis == 'x' else 'x'
    window = domain.physical_window()['xy'.index(along)]
    touched = {f'{along}_lo'} if span[0] <= window.start else set()
    if span[1] >= window.stop:
        touched.add(f'{along}_hi')
    return touch_factor(getattr(boundary_config, 'ghost_faces', ()), touched)


def box_cells(domain, x0, y0, x1, y1, label):
    """(ix0, ix1, iy0, iy1): half-open cells [cell(x_lo), cell(x_hi)+1) x [cell(y_lo), cell(y_hi)+1) of a physical
    box."""
    domain.validate_physical_pos(x0, y0, f"{label} corner 0")
    domain.validate_physical_pos(x1, y1, f"{label} corner 1")
    x_lo, x_hi = sorted((x0, x1))
    y_lo, y_hi = sorted((y0, y1))
    return (domain.pos_to_cell_x(x_lo), domain.pos_to_cell_x(x_hi) + 1,
            domain.pos_to_cell_y(y_lo), domain.pos_to_cell_y(y_hi) + 1)


def tfsf_box_cells(domain, x0, y0, x1, y1, full_period_y=False, boundary_config=None, polarization='ez'):
    """box_cells of a TFSF total-field box; ValueError unless one scattered-field cell stays inside the physical window
    on every side, except a y mirror face of boundary_config of the plane wave's parity (the box opens onto it: its
    image completes it; core.mirror.plane_wave_mirror_issue). full_period_y (FDFD on a periodic y axis) also accepts
    every y cell [0, ny) (no y edges)."""
    ix0, ix1, iy0, iy1 = box_cells(domain, x0, y0, x1, y1, "TFSF box")
    window_x, window_y = domain.physical_window()
    reached = box_window_faces(domain, (ix0, ix1, iy0, iy1))
    open_faces = tfsf_open_faces(boundary_config, reached, polarization)
    spans_period_y = full_period_y and iy0 == 0 and iy1 == domain.grid.ny
    y_inside = spans_period_y or not ({'y_lo', 'y_hi'} & (reached - open_faces))
    if 'x_lo' in reached or 'x_hi' in reached or not y_inside:
        period_text = " (or, along the periodic y axis, every y cell)" if full_period_y else ""
        raise ValueError(f"TFSF box cells [{ix0}, {ix1}) x [{iy0}, {iy1}) need at least one scattered-field "
                         f"cell inside the physical window [{window_x.start}, {window_x.stop}) x "
                         f"[{window_y.start}, {window_y.stop}) on every side{period_text}, except a y mirror face "
                         f"(ez symmetric, hz antisymmetric) the box opens onto")
    return ix0, ix1, iy0, iy1


TFSF_POLARIZATION_AXES = {'ez': 'z', 'hz': 'y'}   # incident E of the 2D +x plane wave per polarisation


def box_window_faces(domain, cells) -> set:
    """Window faces whose edge cell the half-open box cells (ix0, ix1, iy0, iy1) cover."""
    ix0, ix1, iy0, iy1 = cells
    window_x, window_y = domain.physical_window()
    return {face for face, reaches in (('x_lo', ix0 <= window_x.start), ('x_hi', ix1 >= window_x.stop),
                                       ('y_lo', iy0 <= window_y.start), ('y_hi', iy1 >= window_y.stop)) if reaches}


def tfsf_open_faces(boundary_config, reached, polarization) -> frozenset:
    """Mirror faces (BoundaryConfig.ghost_faces) among `reached` (the window edges a TFSF box reaches) onto which the
    +x plane wave box opens; ValueError for a reached mirror face of the wrong parity or normal to x."""
    mirrors = getattr(boundary_config, 'ghost_faces', ()) if boundary_config is not None else ()
    open_faces = set()
    for face in sorted(set(reached) & set(mirrors)):
        sign = MIRROR_SIGNS[getattr(boundary_config, face).value]
        problem = plane_wave_mirror_issue(face, sign, 'x', TFSF_POLARIZATION_AXES[polarization])
        if problem is not None:
            raise ValueError(f'TFSF box {problem}')
        open_faces.add(face)
    return frozenset(open_faces)


def near_to_far_box_cells(domain, x0, y0, x1, y1):
    """box_cells of a near-to-far-field box; ValueError when its samples in column x1 / row y1 (one cell beyond the
    half-open box) leave the physical window."""
    ix0, ix1, iy0, iy1 = box_cells(domain, x0, y0, x1, y1, "NF2FF box")
    window_x, window_y = domain.physical_window()
    if ix1 >= window_x.stop or iy1 >= window_y.stop:
        raise ValueError(f"NF2FF box cells [{ix0}, {ix1}) x [{iy0}, {iy1}) sample column {ix1} / row {iy1} "
                         f"outside the physical window [{window_x.start}, {window_x.stop}) x "
                         f"[{window_y.start}, {window_y.stop}); keep the box one cell off the x_hi / y_hi edges")
    return ix0, ix1, iy0, iy1


def _validate_on_line(domain, axis, position, coord, label):
    """validate_physical_pos of the point at `coord` along a line (axis 'x' = line along x at y = position)."""
    if axis == 'x':
        domain.validate_physical_pos(coord, position, label)
    else:
        domain.validate_physical_pos(position, coord, label)


def _line_cell(domain, axis, position, label):
    """(cell across the line, pos_to_cell of the along axis) of a line (axis 'x' = along x at y = position); ValueError
    for another axis or a position outside the physical window."""
    if axis not in ('x', 'y'):
        raise ValueError(f"{label} axis must be 'x' or 'y', got {axis!r}")
    _validate_on_line(domain, axis, position, 0.0, f"{label} ({'y' if axis == 'x' else 'x'})")
    if axis == 'x':
        return domain.pos_to_cell_y(position), domain.pos_to_cell_x
    return domain.pos_to_cell_x(position), domain.pos_to_cell_y


def port_cells(domain, axis, position, span, label):
    """(line cell across the line, span cells [cell(start), cell(end)+1) along it, warning label) of a port line."""
    if span is None or len(span) != 2:
        raise ValueError(f"{label} needs span = (start, end) along the line [m], got {span!r}")
    start, end = sorted((float(span[0]), float(span[1])))
    line_cell, along_cell = _line_cell(domain, axis, position, label)
    for coord in (start, end):
        _validate_on_line(domain, axis, position, coord, f"{label} span")
    span_cells = (along_cell(start), along_cell(end) + 1)
    orientation, across = ('horizontal', 'y') if axis == 'x' else ('vertical', 'x')
    warning_label = f"{label.lower()} on the {orientation} line at {across} = {float(position):.4g} m"
    return line_cell, span_cells, warning_label


@dataclass(frozen=True, eq=False)
class SheetLine:
    """Cells of a sheet current: line cell across the line, span cells along it (None = whole grid line) as `along`,
    width w_perp [m] across the sheet, float64 transverse shape on the covered cells."""
    line_cell: int
    span_cells: tuple | None
    along: slice
    normal_width: float
    transverse_shape: np.ndarray


def sheet_line(domain, boundary_config, polarization, axis, position, span, label, waist=None, center=None):
    """SheetLine of a line source / beam (axis 'x' = line along x at y = position): ez through the line-cell centres
    (w_perp = Domain cell width), hz on their hi faces (w_perp = dual spacing; ValueError on the outer grid wall); a
    span end on a cpml window edge continues through the PML."""
    if polarization not in ('ez', 'hz'):
        raise ValueError(f"{label} polarization must be 'ez' or 'hz', got {polarization!r}")
    line_cell, along_cell = _line_cell(domain, axis, position, label)
    along_axis = 0 if axis == 'x' else 1
    along_centers = domain.cell_centers()[along_axis]
    normal_width = float(domain.cell_widths()[1 - along_axis][line_cell])
    if polarization == 'hz':
        normal_width = _sheet_face_spacing(domain, boundary_config, axis, line_cell, label)

    if span is None:
        span_cells = None
        along = slice(0, along_centers.size)
    else:
        start, end = sorted((float(span[0]), float(span[1])))
        for coord in (start, end):
            _validate_on_line(domain, axis, position, coord, f"{label} span")
        span_lo, span_hi = along_cell(start), along_cell(end) + 1
        # As PortLine: an end on a cpml window edge continues through the PML (a uniform profile is untouched by the
        # stretch), so a window-wide line does not diffract like a slit.
        window = domain.physical_window()[along_axis]
        if span_lo == window.start and getattr(boundary_config, f'{axis}_lo').value == 'cpml':
            span_lo = 0
        if span_hi == window.stop and getattr(boundary_config, f'{axis}_hi').value == 'cpml':
            span_hi = along_centers.size
        span_cells = (span_lo, span_hi)
        along = slice(*span_cells)

    if waist is None:
        transverse_shape = np.ones(along.stop - along.start)
    else:
        transverse_shape = np.exp(-((along_centers[along] - float(center)) / float(waist)) ** 2)
    return SheetLine(line_cell=line_cell, span_cells=span_cells, along=along, normal_width=normal_width,
                     transverse_shape=transverse_shape)


def _sheet_face_spacing(domain, boundary_config, axis, line_cell, label):
    """Dual spacing [m] across the hi face of `line_cell` that carries an H_z sheet (y spacing for axis 'x')."""
    grid = domain.grid
    spacings = yee_inverse_spacings(*grid_cell_widths(grid))
    if axis == 'x':
        n_cells, periodic, inv_face_spacing = grid.ny, boundary_config.periodic_y, spacings.inv_dy_h
    else:
        n_cells, periodic, inv_face_spacing = grid.nx, boundary_config.periodic_x, spacings.inv_dx_h
    if line_cell == n_cells - 1 and not periodic:
        raise ValueError(f"{label}: the H_z sheet sits on the hi face of cell {line_cell}, the outer grid wall; "
                         f"move the line at least one cell inside")
    return 1.0 / float(inv_face_spacing[line_cell])


def dipole_faces(cell, n_cells, periodic):
    """(face index, moment fraction) of the hi face `cell` and lo face `cell - 1` along one axis, half each; on a
    non-periodic axis a missing face (-1, or the never-updated outer face n_cells - 1) passes its half to the other."""
    if periodic:
        return [(cell, 0.5), ((cell - 1) % n_cells, 0.5)]
    faces = [face for face in (cell, cell - 1) if 0 <= face <= n_cells - 2]
    if not faces:
        raise ValueError(f"an electric dipole needs at least 2 cells along a non-periodic axis, got {n_cells}")
    return [(face, 1.0 / len(faces)) for face in faces]


def electric_dipole_faces(domain, boundary_config, ix, iy, dipole_moment, orientation):
    """[('ex' | 'ey', i, j, J [A/m^2]), ...] of an in-plane dipole M = `dipole_moment` [A] along `orientation` [deg, ccw
    from +x] in cell (ix, iy), shared over the faces around it; ValueError when no face carries current."""
    grid = domain.grid
    dx_cells, dy_cells = grid_cell_widths(grid)
    spacings = yee_inverse_spacings(dx_cells, dy_cells)
    theta = math.radians(float(orientation))
    moment_x = float(dipole_moment) * math.cos(theta)
    moment_y = float(dipole_moment) * math.sin(theta)
    # A face carrying the moment fraction f has J = f M_axis / (its dual area): dx_i * dual dy on E_x faces,
    # dual dx * dy_j on E_y faces (dual spacing = 1 / inv_d*_h).
    faces = [('ex', ix, j, fraction * moment_x * spacings.inv_dy_h[j] / dx_cells[ix])
             for j, fraction in dipole_faces(iy, grid.ny, boundary_config.periodic_y)]
    faces += [('ey', i, iy, fraction * moment_y * spacings.inv_dx_h[i] / dy_cells[iy])
              for i, fraction in dipole_faces(ix, grid.nx, boundary_config.periodic_x)]
    largest = max(abs(face[3]) for face in faces)
    faces = [face for face in faces if abs(face[3]) > 1e-12 * largest]   # also drops the cos / sin round-off
    if not faces:
        raise ValueError(f"Electric dipole needs a finite non-zero dipole_moment and a finite orientation, got "
                         f"dipole_moment={dipole_moment}, orientation={orientation}")
    return faces

"""Runtime of the brillouin_scattering feature (frequency_response): the bound optical fields of every scan position
sampled at the solid's quadrature points, the complex drive columns, the per-frequency FDT records and the result."""
# The constructor is the FROZEN contract of the feature's apply (definition_brillouin; it checks every value, a
# ValueError naming it); the math is brillouin.py, the FEM assembly fem.elastic_drives. What its values mean:
# - optical_physics: an em_fdfd_ez / _hz / _bor physics; collection_physics '' = the optical physics; incident_source /
#   collection_source: focused_beam feature names ('' = the only enabled one);
# - axial_length: planar solids only (scale mirror_factor / L; axisymmetric: mirror_factor);
# - scan_offsets: focus offsets [m] in position order, brillouin.scan_positions (x outer, y inner; BOR: (0, dz));
#   position 1 = row 0 = the shown solution and the reduced-sweep load;
# - element_mask (None = every solid element) and the crystal_regions masks: per solid mesh element (a rectangular-cell
#   solid: per window cell, flat C order (n_x, n_y)); crystal p: (6, 6) Voigt in model axes, planar only;
# - mirror_factor: 2 ** (model mirror planes) (core.mirror.mirror_faces(spec)).
# Flow (SolidMechanicsFEMPhysics / SolidMechanicsPhysics.bind_physics_drives, called by the frequency_response study):
# - bind (docstring): per scan position one optical solve on the optics' kept factor (solve_moved_beam: only the
#   illumination moves; the collection field is the collection physics' solution, fixed: transmission detection does
#   not co-move) and its DriveFields, then the column kappa (int B^T sigma~ dV + int N^T n f ds). Columns per position:
#   planar one; axisymmetric one per part of brillouin.AZIMUTHAL_PARTS[detection] at the solid's azimuthal order (0:
#   (0, even); 2: (2, even) co- / (2, odd) cross-polarised), summed per position in record. The solid's form is a
#   geometry object (_ElementGeometry: triangles; _CellGeometry: rectangular cells), the only place the two mesh kinds
#   differ (points, samplers, interfaces, assembly, raster cells).
# - Optical fields: DriveFields components on their own grids (DriveFields.grids, em sub file "Focused beams":
#   em_fdfd_hz E_x / E_y on their Yee faces, em_fdfd_bor E_r / E_z on their edges, the rest at the cell centres), so
#   nothing is averaged along the beam (a face mean along a plane wave is low by cos(k h / 2)); their cell-centre values
#   (_Optical.centred) come from the edge grids by the phase-corrected half step brillouin.edges_to_centres (exact for
#   a plane wave). Every field interpolation removes the local phase first (brillouin.phase_corrected_sample: plain
#   bilinear interpolation lowers a 2 n k0 drive by sinc^4(q h / 4), 1 - (q h)^2 / 24 in amplitude).
# - Sampling (triangles): each component phase-corrected bilinear on its own grid at the DrivePoints (points of the
#   half cell between the outermost sample coordinates and the window edge, i.e. within one spacing of their hull,
#   take the hull value; farther ones raise: brillouin.bilinear_weights). eps per point: the staircase eps of the
#   nearest optical cell (largest bilinear weight among the 4 of the cell-centre stencil) whose centre lies in a solid
#   element of the point's own material (equal solid members flag and material values: the conforming solid mesh knows
#   the material, the centre-tested staircase may not); a point with no such cell among the 4 (a feature thinner than
#   an optical cell, or material values varying in space) takes the containing cell's eps and is counted in a bind
#   warning. A drive point whose cell is PEC raises. The clamped edge half cell is the even mirror rule on a
#   mirror-plane edge (bilinear between a sample and its even image); em_fdfd_bor adds the axis rule on x_lo (r = 0):
#   the order-1 E_z ~ r (on the r centres) is scaled by r / r_0 below the first centre r_0 (E_r starts on the axis
#   edge, A_phi stays finite there). An odd field component on a mirror plane keeps its edge value in that half cell
#   (no mirror signs here: an O(half cell) error of the strip).
#   Rectangular cells (the optics on the solid's own mesh: same window cells, checked): one point per selected cell at
#   its centre with weight dA, the cell's own (centre) fields and eps (no interpolation).
# - Moving boundary (triangles): interior edges between two window elements, touching the drive selection, whose sides
#   differ in a solid material value (the conforming solid mesh marks material interfaces exactly; an optical interface
#   inside one solid material, or between two materials of equal mechanics, is not found) AND in the optical eps of
#   their materials; each side's eps per Gauss point is the eps of the nearest cell of that side's material (as the
#   drive points'; none among the 4: the staircase cell one local centre diagonal off the point into that side, clamped
#   to the window, counted in the warning), isotropic (the mean of the component eps). The fields are read at the Gauss
#   point itself: E (cell-centre values) and D = eps_cell E of the in-plane components, interpolated with one
#   phase-corrected stencil, so E_par and D_n (continuous across the interface) are interpolated as such and no sample
#   reaches into the other material (brillouin.interface_traction). Rectangular cells: the interior Yee faces between
#   two window cells, touching the selection, one of them a solid member, differing in a solid material value and in
#   the optical eps; the sides are the two cells (normal +x / +y out of the lo cell), f l_face per face (the two-sided
#   brillouin.moving_boundary_traction, brillouin.face_drive).
# - record: psd per position (brillouin.fdt_psd x scale, parts summed). result(): the references describe the
#   photoelastic drive: local = per position and material group the DHO lines of the points, compressed in relative |q|
#   bins of LINE_BIN with the weighted-mean q per bin (the line shape is smooth in q: a 1e-4 bin shifts no line by more
#   than ~1e-8 of its width); homogeneous = the kappa sigma~ raster (cell-centre fields) of the optical cells inside the
#   drive selection (planar, no mirror planes: the FFT needs the unfolded drive, one uniform isotropic material,
#   uniform cells over its box; zero padding per axis from the phonon attenuation: brillouin.homogeneous_padding). The
#   result is kept until the records or the bound drive change (_changed).
import copy
import math
from dataclasses import dataclass

import numpy as np

from ...fem.elastic_drives import assemble_edge_drive, assemble_stress_drive, drive_points, edge_drive_points
from ...model.solver_hooks import DriveOpticsSolver
from .brillouin import (AZIMUTHAL_PARTS, REFERENCES, BrillouinSpectra, _checked_detection, _temperature,
                        amplitude_prefactor, azimuthal_interface_traction, azimuthal_parts, bilinear_sample,
                        bilinear_weights, brillouin_result, cell_stress_drive, crystal_regions_checked,
                        edges_to_centres, face_drive, fdt_psd, field_products, homogeneous_reference,
                        interface_traction, local_reference, local_wavenumbers, longitudinal_function,
                        moving_boundary_traction, optical_stress, phase_corrected_sample, solid_rows)
from .viscoelastic import parameter_groups

TORSION_WARNING = 1e-3   # bind warns when the dropped m = 0 torsion share of a cross-polarised drive exceeds it
LINE_BIN = 1e-4          # relative |q| bin of the compressed local-reference lines
_EPS_RTOL = 1e-12        # two eps samples differ (an optical interface) above this share of the larger
_OMEGA_RTOL = 1e-9       # incident and collection optics at one wavelength
_SPACING_RTOL = 1e-9     # uniform optical cells over the homogeneous raster
# DriveFields components -> brillouin field names: planar optics by polarization; em_fdfd_bor gives the m = 1
# standing-wave amplitudes (E_r, A_phi = u = -j E_phi, E_z) as 'r', 'phi', 'z' already (fdfd_bor.drive_fields)
_PLANAR_FIELDS = {'ez': (('ez', 'z'),), 'hz': (('ex', 'x'), ('ey', 'y'))}
_BOR_FIELDS = (('r', 'r'), ('phi', 'phi'), ('z', 'z'))
# in-plane components (the moving boundary's D = eps E is read of these): planar (x, y), axisymmetric (r, z)
_IN_PLANE = {True: ('x', 'y'), False: ('r', 'z')}
# (aa, bb, ab) rows of the solid stress the longitudinal projection reads: planar (xx, yy, xy), axisymmetric
# (rr, zz, rz)
_PLANE_ROWS = {False: (0, 1, 2), True: (0, 1, 3)}


@dataclass(frozen=True, eq=False)
class SolidDriveInputs:
    """What a drive reads of the solid at bind time (SolidMechanicsFEMPhysics.bind_physics_drives): its label, the dof
    map of the last assembly, the form (axisymmetric, azimuthal order, ring angle), n_unknowns of the harmonic system
    (mixed: + the pressure dofs), host bool per mesh element `solid` (the members the frequency response solves) and
    `window` (window elements), every material property as host float64 per element, the viscoelastic.ElementModuli
    of the members (required form: E / nu elements included) and locate (points (n, 2) [m] -> containing element,
    -1 outside). Read-only."""
    label: str
    dofs: object
    axisymmetric: bool
    azimuthal_order: int
    ring_angle: float
    n_unknowns: int
    solid: np.ndarray
    window: np.ndarray
    materials: dict
    moduli: object
    locate: object


@dataclass(frozen=True, eq=False)
class CellDriveInputs:
    """What a drive reads of a rectangular-cell (Q1) solid at bind time (SolidMechanicsPhysics.bind_physics_drives):
    its label, the window cell centres x (n_x,), y (n_y,) [m] in the drawing frame (the optics' DriveFields frame: the
    optics must run on the solid's mesh, bind compares the centres), the window widths dx, dy [m], n_unknowns (2 per
    window node, node-major (u_x, u_y) over (n_x + 1, n_y + 1) nodes), host bool per window cell `solid` (flat C order
    (n_x, n_y): the members the frequency response solves), every material property the solid reads as host float64
    per window cell (flat) and the viscoelastic.ElementModuli of the cells (the E / nu route: a constant loss factor
    only). Planar plane strain only (axisymmetric False). Read-only."""
    label: str
    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    n_unknowns: int
    solid: np.ndarray
    materials: dict
    moduli: object
    axisymmetric = False   # class constants: the form the planar plane-strain drive needs
    azimuthal_order = 0
    ring_angle = 2.0 * math.pi


class _GridSamples:
    """Bilinear stencils (brillouin.bilinear_weights) of fixed points on one grid of sample coordinates x (n_x,), y
    (n_y,), points within one spacing outside the coordinates' hull clamped onto it (module header): plain samples
    (smooth values: the local wavenumbers) and phase-corrected ones (optical fields:
    brillouin.phase_corrected_sample)."""

    def __init__(self, x, y, points, what):
        try:
            self.index, self.weight = bilinear_weights(x, y, _clamped(x, y, points))
        except ValueError as exc:
            raise ValueError(f'{what}: {exc}') from exc

    def sample(self, values) -> np.ndarray:
        return bilinear_sample(values, self.index, self.weight)

    def phased(self, arrays, phase_count=None) -> list:
        return phase_corrected_sample(arrays, self.index, self.weight, phase_count)


def _axis_factor(x, points, axis):
    """(n,) r / r_0 below the first r centre r_0 (1 elsewhere) of axisymmetric points (the order-1 E_z ~ r on the
    axis), or None (planar, or no point below r_0)."""
    if not axis or not (points[:, 0] < x[0]).any():
        return None
    return np.clip(points[:, 0] / x[0], 0.0, 1.0)


def _axis_rule(names, samples, factor) -> list:
    """[samples] with every 'z' entry (E_z, D_z) times the _axis_factor (None: unchanged)."""
    if factor is None:
        return list(samples)
    return [values * factor if name == 'z' else values for name, values in zip(names, samples)]


class _FieldSamples:
    """The optical fields at fixed points (triangles; module header): one _GridSamples per distinct component grid
    (DriveFields.grids), each component phase-corrected on its own grid, the cell-centre grid's (`centre`) also for
    plain samples of cell values (the local wavenumbers) and the own-material cells; axis=True (em_fdfd_bor): the
    axis rule on 'z'."""

    def __init__(self, x, y, grids, points, what, axis=False):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.centre = _GridSamples(x, y, points, what)
        made = [(x, y, self.centre)]
        self.components = {}
        for name, (grid_x, grid_y) in grids.items():
            sampler = next((kept for kept_x, kept_y, kept in made
                            if np.array_equal(kept_x, grid_x) and np.array_equal(kept_y, grid_y)), None)
            if sampler is None:
                sampler = _GridSamples(grid_x, grid_y, points, what)
                made.append((grid_x, grid_y, sampler))
            self.components[name] = sampler
        self.axis_factor = _axis_factor(x, points, axis)

    def sample(self, values) -> np.ndarray:
        return self.centre.sample(values)

    def fields(self, optical) -> dict:
        """{name: (n,) samples} of an _Optical's components, each on its own grid (the axis rule on 'z')."""
        names = list(optical.values)
        samples = [self.components[name].phased([values])[0] for name, values in optical.values.items()]
        return dict(zip(names, _axis_rule(names, samples, self.axis_factor)))


class _CellSamples:
    """_FieldSamples' interface on the window cells themselves (a rectangular-cell solid on the optics' mesh): the
    cell-centre value of each listed flat cell, no interpolation."""

    def __init__(self, cells):
        self.cell = np.asarray(cells, dtype=np.int64)

    def sample(self, values) -> np.ndarray:
        return np.asarray(values).reshape(-1)[self.cell]

    def fields(self, optical) -> dict:
        return {name: self.sample(values) for name, values in optical.centred.items()}

    cells = sample


class _InterfaceSamples:
    """Moving-boundary fields at the interface Gauss points themselves (triangles; module header): one cell-centre
    stencil; E (the cell-centre values of every component) and D = eps_cell E of the in-plane ones phase-corrected
    with E's shared step, so E_par and D_n are interpolated as the continuous quantities they are (projections
    commute with the stencil); the axis rule on 'z' (E_z and D_z)."""

    def __init__(self, x, y, points, what, planar):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.grid = _GridSamples(x, y, points, what)
        self.in_plane = _IN_PLANE[planar]
        self.axis_factor = _axis_factor(x, points, not planar)

    def fields(self, optical, eps, shape) -> tuple:
        """({name: E samples}, {in-plane name: D samples}) of an _Optical, eps the in-plane cell eps {'xx': ...},
        reshaped to shape."""
        names = list(optical.centred)
        d_names = [name for name in names if name in self.in_plane]
        arrays = [optical.centred[name] for name in names] + [eps[name + name] * optical.centred[name]
                                                              for name in d_names]
        samples = _axis_rule(names + d_names, self.grid.phased(arrays, phase_count=len(names)), self.axis_factor)
        values = [array.reshape(shape) for array in samples]
        return dict(zip(names, values[:len(names)])), dict(zip(d_names, values[len(names):]))


def _clamped(x, y, points) -> np.ndarray:
    """(n, 2) copy of points with coordinates within one spacing outside the hull of x, y moved onto it."""
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    for axis, centres in enumerate((x, y)):
        low, high = centres[0], centres[-1]
        values = points[:, axis]
        inside = (values >= low - (centres[1] - centres[0])) & (values <= high + (centres[-1] - centres[-2]))
        points[:, axis] = np.where(inside, np.clip(values, low, high), values)
    return points


def _centre_spacings(x, y, points) -> list:
    """[(n,) x spacing, (n,) y spacing] of the cell centres around points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    widths = []
    for axis, centres in enumerate((x, y)):
        index = np.clip(np.searchsorted(centres, points[:, axis]) - 1, 0, len(centres) - 2)
        widths.append(centres[index + 1] - centres[index])
    return widths


def _nearest_cells(x, y, points) -> np.ndarray:
    """(n,) flat index of the cell whose centre is nearest along each axis (coordinates beyond the centres clamped:
    never raises)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    indices = []
    for axis, centres in enumerate((x, y)):
        values = points[:, axis]
        index = np.clip(np.searchsorted(centres, values, side='right') - 1, 0, len(centres) - 2)
        index += values - centres[index] > centres[index + 1] - values
        indices.append(index)
    return indices[0] * len(y) + indices[1]


def _uniform_spacing(centres, what) -> float:
    """The cell-centre spacing of a run of centres; ValueError when it is not uniform or shorter than 2 cells."""
    steps = np.diff(np.asarray(centres, dtype=np.float64))
    if not steps.size or float(np.ptp(steps)) > _SPACING_RTOL * float(steps.mean()):
        raise ValueError(f'{what}: the homogeneous reference needs uniform optical cells (>= 2 per axis) over the '
                         "drive region's box; use reference 'local'")
    return float(steps.mean())


@dataclass(frozen=True, eq=False)
class _Optical:
    """One solved optical field of a position in brillouin component names: values on their own grids (DriveFields
    .grids, (x, y) coordinates per component) and the cell-centre values (centred: the edge grids' phase-corrected half
    steps, brillouin.edges_to_centres; a centre grid's own array)."""
    values: dict
    grids: dict
    centred: dict


def _optical_arrays(fields, planar, label) -> tuple:
    """(_Optical, {'cc': complex eps (n_x, n_y)}) of DriveFields; ValueError when its components do not fit the
    solid's form, or a component grid is neither the cell centres nor the cell edges along an axis (a bug)."""
    names = _PLANAR_FIELDS.get(fields.polarization) if planar else _BOR_FIELDS
    keys = None if names is None else {key for key, _ in names}
    if keys is None or set(fields.fields) != keys or set(fields.eps) != keys or set(fields.grids) != keys:
        wanted = ('a planar solid: em_fdfd_ez {ez} or em_fdfd_hz {ex, ey}' if planar
                  else 'an axisymmetric solid: em_fdfd_bor {r, phi, z}')
        raise ValueError(f"{label}: the optical drive fields ({fields.polarization}: "
                         f"{', '.join(sorted(fields.fields))}) do not fit {wanted}")
    x, y = np.asarray(fields.x, dtype=np.float64), np.asarray(fields.y, dtype=np.float64)
    values, grids, centred = {}, {}, {}
    for key, name in names:
        array = np.asarray(fields.fields[key], dtype=np.complex128)
        grid = tuple(np.asarray(coordinates, dtype=np.float64) for coordinates in fields.grids[key])
        if array.shape != tuple(len(coordinates) for coordinates in grid):
            raise ValueError(f'{label}: drive field {key} has shape {array.shape}, its grid '
                             f'{tuple(len(coordinates) for coordinates in grid)} (a bug)')
        centre = array
        for axis, (coordinates, centres) in enumerate(zip(grid, (x, y))):
            if np.array_equal(coordinates, centres):
                continue
            if not (len(coordinates) == len(centres) + 1 and (coordinates[:-1] < centres).all()
                    and (centres < coordinates[1:]).all()):
                raise ValueError(f'{label}: drive field {key} lies on neither the cell centres nor the cell edges '
                                 f'along axis {axis} (a bug)')
            centre = edges_to_centres(centre, axis)
        values[name], grids[name], centred[name] = array, grid, centre
    eps = {name + name: np.asarray(fields.eps[key], dtype=np.complex128) for key, name in names}
    if any(values.shape != (len(x), len(y)) for values in eps.values()):
        raise ValueError(f'{label}: the drive eps arrays are not on the {len(x)} x {len(y)} window cells (a bug)')
    return _Optical(values=values, grids=grids, centred=centred), eps


def _same_grids(first, second) -> bool:
    """True when two {component: (x, y)} grid dicts hold the same components on equal coordinates."""
    return set(first) == set(second) and all(np.array_equal(a, b) for name in first
                                             for a, b in zip(first[name], second[name]))


@dataclass(frozen=True, eq=False)
class _Interfaces:
    """Moving-boundary samples: the EdgeDrivePoints (rectangular cells: _FacePoints), the _InterfaceSamples of their
    flattened Gauss points (rectangular cells: None) or the two side samplers (rectangular cells: the lo / hi cell of
    each face; triangles: None), the isotropic eps of each side's material, the sample shape ((k, q); rectangular cells
    (k,)) and the points whose side eps fell back to the staircase cell (module header)."""
    points: object
    sampler: object
    sides: tuple
    eps_sides: tuple
    shape: tuple
    fallbacks: int


@dataclass(frozen=True, eq=False)
class _Raster:
    """Optical cells of the drive region for the homogeneous reference: flat cell indices, their box indices i / j,
    the box shape, spacing dx / dy [m], the material law index, photoelastic p per cell and eps per cell."""
    cells: np.ndarray
    i: np.ndarray
    j: np.ndarray
    shape: tuple
    dx: float
    dy: float
    law: int
    p11: np.ndarray
    p12: np.ndarray
    p44: object
    eps: dict


@dataclass(frozen=True, eq=False)
class _Context:
    """Per-bind sampling state of one optical grid (made at the first scan position)."""
    x: np.ndarray
    y: np.ndarray
    omega: float
    grids: dict               # brillouin component -> (x, y) sample coordinates (the first position's DriveFields)
    volume: object            # _FieldSamples / _CellSamples of the drive points
    eps_points: dict
    eps_in_plane: dict        # in-plane cell eps {'xx': ...} the triangle moving boundary's D reads ({} otherwise)
    interfaces: object        # _Interfaces or None
    fallbacks: int            # drive points whose eps is not their own material's (module header)
    point_laws: object        # int64 (n_points,) law index per drive point (-1 none), or None (reference 'none')
    laws: tuple               # ((longitudinal law, shear law, density), ...)
    raster: object            # _Raster or None


@dataclass(frozen=True, eq=False)
class _Bound:
    """A bind's result besides the columns: what record / result / summary read."""
    planar: bool
    azimuthal_order: int
    n_parts: int
    scale: float
    kappa: float
    wavelength: float
    incident_power: float
    collection_power: float
    torsion: float
    norms: dict
    q_mean: tuple
    laws: tuple
    lines: tuple      # per position ((law, q (m,), weight (m,)), ...) (reference local) or ()
    rasters: tuple    # per position kappa sigma~ (3, n_x, n_y) (reference homogeneous) or ()
    raster: object    # _Raster or None
    n_points: int
    n_edge_points: int


@dataclass(frozen=True, eq=False)
class _CellPoints:
    """Drive points of a rectangular-cell solid: one per selected window cell (flat C-order index `elements`, also the
    row of its material values) at the cell centre, weight dA [m^2]."""
    elements: np.ndarray
    weights: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.elements.size)


@dataclass(frozen=True, eq=False)
class _FacePoints:
    """Moving-boundary Yee faces of a rectangular-cell solid, x faces first: unit normals (k, 2) (+x / +y out of the lo
    cell, side 0), lengths (k,) [m] and the flat indices of the x faces into (n_x - 1, n_y) and of the y faces into
    (n_x, n_y - 1)."""
    normals: np.ndarray
    lengths: np.ndarray
    x_faces: np.ndarray
    y_faces: np.ndarray


def _materials_differ(solid, first, second) -> np.ndarray:
    """Bool per pair of element (rectangular cells: cell) indices first / second: the solid members flag or a material
    value differs (NaN equals NaN)."""
    members = np.asarray(solid.solid, dtype=bool)
    differ = members[first] != members[second]
    for values in solid.materials.values():
        a, b = values[first], values[second]
        differ |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
    return differ


def _eps_differ(eps_0, eps_1) -> np.ndarray:
    """Bool: two eps samples differ (an optical interface) by more than _EPS_RTOL of the larger."""
    return np.abs(eps_0 - eps_1) > _EPS_RTOL * np.maximum(np.abs(eps_0), np.abs(eps_1))


def _material_groups(solid, elements) -> np.ndarray:
    """int64 material group per listed mesh element: equal solid members flag and equal material values (NaN = NaN,
    -0 = 0; bit-exact otherwise), numbered within this call."""
    columns = [np.asarray(solid.solid, dtype=np.float64)[elements]]
    columns += [np.asarray(values, dtype=np.float64)[elements] for values in solid.materials.values()]
    table = np.stack(columns, axis=1) + 0.0   # -0.0 -> 0.0
    table[np.isnan(table)] = np.nan           # one NaN bit pattern
    _, groups = np.unique(np.ascontiguousarray(table).view(np.int64), axis=0, return_inverse=True)
    return np.asarray(groups, dtype=np.int64).reshape(-1)


class _ElementGeometry:
    """Triangle solids (SolidDriveInputs): the fem.elastic_drives DrivePoints of the selected elements, the
    phase-corrected samplers of the optical grids, the own-material optical cells, the moving-boundary mesh edges,
    int B^T sigma / int N^T t assembly and the raster cells located in the mesh (module header)."""

    def __init__(self, solid, selection, label, repeated):
        self.solid, self.label = solid, label
        self.repeated = repeated   # several stress drives on these points: keep their strain operator (bind's life)
        try:
            self.points = drive_points(solid.dofs, axisymmetric=solid.axisymmetric,
                                       azimuthal_order=solid.azimuthal_order, elements=selection)
        except ValueError as exc:
            raise ValueError(f'{label}: {exc}') from exc
        self.n_vector = self.points.components * solid.dofs.n_dofs

    def sampler(self, x, y, grids, what) -> _FieldSamples:
        return _FieldSamples(x, y, grids, self.points.points, what, axis=self.solid.axisymmetric)

    def own_cells(self, x, y, volume) -> tuple:
        """(flat optical cell per drive point, bool matched) (module header: the own-material cell)."""
        return self._own_cells(x, y, volume.centre, self.points.elements)

    def _own_cells(self, x, y, stencil, elements) -> tuple:
        """(flat cell, matched) per point of a cell-centre _GridSamples stencil whose solid elements are `elements`:
        the largest-weight stencil cell whose centre lies in an element of the point's material group, else the
        largest-weight cell (matched False)."""
        order = np.argsort(-stencil.weight, axis=1, kind='stable')
        candidates = np.take_along_axis(stencil.index, order, axis=1)
        cells, inverse = np.unique(candidates, return_inverse=True)
        n_y = len(y)
        located = np.asarray(self.solid.locate(np.column_stack([x[cells // n_y], y[cells % n_y]])),
                             dtype=np.int64).reshape(-1)
        elements = np.asarray(elements, dtype=np.int64).reshape(-1)
        inside = located >= 0
        involved, position = np.unique(np.concatenate([elements, located[inside]]), return_inverse=True)
        groups = _material_groups(self.solid, involved)[np.asarray(position).reshape(-1)]
        cell_groups = np.full(cells.size, -1, dtype=np.int64)
        cell_groups[inside] = groups[elements.size:]
        same = cell_groups[np.asarray(inverse).reshape(candidates.shape)] == groups[:elements.size, None]
        matched = same.any(axis=1)
        chosen = candidates[np.arange(len(candidates)), np.where(matched, np.argmax(same, axis=1), 0)]
        return chosen, matched

    def stress_drive(self, stress) -> np.ndarray:
        """(n_vector,) complex int B^T stress dV of a stress (rows, n_points)."""
        return assemble_stress_drive(self.solid.dofs, self.points, stress, keep_operator=self.repeated)

    def traction_drive(self, interfaces, values) -> np.ndarray:
        """(n_vector,) complex int N^T n f ds of a normal traction value f (k, q) on the interface edges."""
        traction = values[..., None] * interfaces.points.normals
        return assemble_edge_drive(self.solid.dofs, interfaces.points, traction)

    def _side_eps(self, edge_points, x, y, isotropic, what) -> tuple:
        """((isotropic eps of side 0, of side 1) (k, q) of the side materials, int64 (k, q) count of the sides whose
        eps fell back to the staircase cell) per Gauss point of EdgeDrivePoints (module header); every value is per
        point, so a subset of the edges reads its rows."""
        shape = tuple(edge_points.weights.shape)
        flat = edge_points.points.reshape(-1, 2)
        normals = edge_points.normals.reshape(-1, 2)
        stencil = _GridSamples(x, y, flat, what)
        diagonal = np.hypot(*_centre_spacings(x, y, flat))
        values, fallbacks = [], np.zeros(len(flat), dtype=np.int64)
        for side, sign in ((0, -1.0), (1, 1.0)):   # normals out of side 0: side 0 lies at -n
            cells, matched = self._own_cells(x, y, stencil, np.repeat(edge_points.sides[:, side], shape[1]))
            missed = ~matched
            if missed.any():
                cells[missed] = _nearest_cells(x, y, flat[missed]
                                               + sign * diagonal[missed, None] * normals[missed])
                fallbacks += missed
            values.append(isotropic.reshape(-1)[cells].reshape(shape))
        return tuple(values), fallbacks.reshape(shape)

    def interfaces(self, selection, x, y, eps):
        """_Interfaces of the moving boundary (module header), None when no interface touches the selection."""
        solid, label = self.solid, self.label
        dofs = solid.dofs
        mesh = dofs.mesh
        pairs = np.asarray(mesh.edge_elements, dtype=np.int64)
        interior = np.flatnonzero(pairs[:, 1] >= 0)
        first, second = pairs[interior, 0], pairs[interior, 1]
        window = np.asarray(solid.window, dtype=bool)
        touching = (selection[first] | selection[second]) & window[first] & window[second]
        edges = interior[touching & _materials_differ(solid, first, second)]
        if not edges.size:
            return None
        isotropic = sum(eps.values()) / len(eps)
        what = f'{label}: moving-boundary points'

        def points_of(chosen):
            return edge_drive_points(dofs, chosen, axisymmetric=solid.axisymmetric,
                                     azimuthal_order=solid.azimuthal_order)
        edge_points = points_of(edges)
        if not len(edge_points.edges):
            return None
        (eps_0, eps_1), fallbacks = self._side_eps(edge_points, x, y, isotropic, what)
        optical = _eps_differ(eps_0, eps_1).any(axis=1)
        if not optical.any():
            return None
        if not optical.all():   # edges of equal optical materials dropped (the same rows of the per-edge arrays)
            edge_points = points_of(edge_points.edges[optical])
            eps_0, eps_1, fallbacks = eps_0[optical], eps_1[optical], fallbacks[optical]
        shape = tuple(edge_points.weights.shape)
        sampler = _InterfaceSamples(x, y, edge_points.points.reshape(-1, 2), what, not solid.axisymmetric)
        return _Interfaces(points=edge_points, sampler=sampler, sides=None, eps_sides=(eps_0, eps_1), shape=shape,
                           fallbacks=int(fallbacks.sum()))

    def raster_cells(self, x, y, selection) -> tuple:
        """(flat optical cells whose centres lie in the selection, their mesh elements)."""
        centres = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1).reshape(-1, 2)
        elements = np.asarray(self.solid.locate(centres), dtype=np.int64)
        inside = elements >= 0
        inside[inside] = selection[elements[inside]]
        cells = np.flatnonzero(inside)
        return cells, elements[cells]


class _CellGeometry:
    """Rectangular-cell (Q1) solids (CellDriveInputs) on the optics' own mesh: one point per selected window cell, the
    cells' own (cell-centre) optical values, the moving-boundary Yee faces, brillouin.cell_stress_drive / face_drive and
    the selected cells as the homogeneous raster (module header)."""

    def __init__(self, solid, selection, label):
        self.solid, self.label = solid, label
        self.shape = (int(np.asarray(solid.x).size), int(np.asarray(solid.y).size))
        n_x, n_y = self.shape
        if (np.shape(solid.dx) != (n_x,) or np.shape(solid.dy) != (n_y,) or selection.shape != (n_x * n_y,)
                or solid.n_unknowns != 2 * (n_x + 1) * (n_y + 1)):
            raise ValueError(f'{label}: the rectangular-cell solid inputs do not fit its {n_x} x {n_y} window cells '
                             f'(widths {np.shape(solid.dx)} / {np.shape(solid.dy)}, {selection.size} cells, '
                             f'{solid.n_unknowns} unknowns; a bug)')
        cells = np.flatnonzero(selection)
        self.points = _CellPoints(elements=cells, weights=np.outer(solid.dx, solid.dy).reshape(-1)[cells])
        self.n_vector = 2 * (n_x + 1) * (n_y + 1)

    def sampler(self, x, y, grids, what) -> _CellSamples:
        solid = self.solid
        if not (np.array_equal(x, solid.x) and np.array_equal(y, solid.y)):
            raise ValueError(f'{what}: the optics run on other window cells ({len(x)} x {len(y)}) than the '
                             f'rectangular-cell solid ({self.shape[0]} x {self.shape[1]}): the drive reads the '
                             'optical cells as the solid cells, so both physics need one tensor mesh')
        return _CellSamples(self.points.elements)

    def own_cells(self, x, y, volume) -> tuple:
        """(the drive cells themselves, all matched): a solid cell is its optical cell."""
        return volume.cell, np.ones(volume.cell.size, dtype=bool)

    def stress_drive(self, stress) -> np.ndarray:
        """(n_vector,) complex sum dA B_c^T stress of a stress (3, n_points) at the selected cells."""
        n_x, n_y = self.shape
        raster = np.zeros((3, n_x * n_y), dtype=np.complex128)
        raster[:, self.points.elements] = stress
        return cell_stress_drive(raster.reshape(3, n_x, n_y), self.solid.dx, self.solid.dy).reshape(-1)

    def traction_drive(self, interfaces, values) -> np.ndarray:
        """(n_vector,) complex face sum of a Johnson value f (k,) per interface face: f l_face on the face's normal,
        half on each corner node."""
        faces = interfaces.points
        loads = np.asarray(values, dtype=np.complex128) * faces.lengths
        n_x, n_y = self.shape
        count = faces.x_faces.size
        face_x = np.zeros((n_x - 1) * n_y, dtype=np.complex128)
        face_y = np.zeros(n_x * (n_y - 1), dtype=np.complex128)
        face_x[faces.x_faces] = loads[:count]
        face_y[faces.y_faces] = loads[count:]
        return face_drive(face_x.reshape(n_x - 1, n_y), face_y.reshape(n_x, n_y - 1)).reshape(-1)

    def interfaces(self, selection, x, y, eps):
        """_Interfaces of the moving boundary on the interior Yee faces (module header), None when none."""
        solid = self.solid
        n_x, n_y = self.shape
        index = np.arange(n_x * n_y, dtype=np.int64).reshape(n_x, n_y)
        members = np.asarray(solid.solid, dtype=bool)
        isotropic = (sum(eps.values()) / len(eps)).reshape(-1)
        found = []   # per axis (face indices, lo cells, hi cells)
        for axis in (0, 1):
            lo = (index[:-1, :] if axis == 0 else index[:, :-1]).reshape(-1)
            hi = (index[1:, :] if axis == 0 else index[:, 1:]).reshape(-1)
            faces = np.flatnonzero((selection[lo] | selection[hi]) & (members[lo] | members[hi]))
            lo, hi = lo[faces], hi[faces]
            keep = _materials_differ(solid, lo, hi) & _eps_differ(isotropic[lo], isotropic[hi])
            found.append((faces[keep], lo[keep], hi[keep]))
        (x_faces, x_lo, x_hi), (y_faces, y_lo, y_hi) = found
        if not (x_faces.size or y_faces.size):
            return None
        lengths = np.concatenate([np.asarray(solid.dy, dtype=np.float64)[x_faces % n_y],
                                  np.asarray(solid.dx, dtype=np.float64)[y_faces // (n_y - 1)]])
        normals = np.zeros((x_faces.size + y_faces.size, 2))
        normals[:x_faces.size, 0] = 1.0
        normals[x_faces.size:, 1] = 1.0
        sides = (_CellSamples(np.concatenate([x_lo, y_lo])), _CellSamples(np.concatenate([x_hi, y_hi])))
        return _Interfaces(points=_FacePoints(normals=normals, lengths=lengths, x_faces=x_faces, y_faces=y_faces),
                           sampler=None, sides=sides, eps_sides=tuple(side.cells(isotropic) for side in sides),
                           shape=(len(lengths),), fallbacks=0)

    def raster_cells(self, x, y, selection) -> tuple:
        """(flat optical cells of the selection, their material rows): the solid's own cells."""
        cells = self.points.elements
        return cells, cells


def _geometry(solid, selection, label, repeated):
    """The drive geometry of the solid's inputs: _ElementGeometry of SolidDriveInputs, else _CellGeometry (bind checks
    the type first); repeated: several stress drives on the points (_ElementGeometry keeps their strain operator)."""
    if isinstance(solid, SolidDriveInputs):
        return _ElementGeometry(solid, selection, label, repeated)
    return _CellGeometry(solid, selection, label)


def _labelled(label, check, *args):
    """check(*args), its ValueError prefixed with label."""
    try:
        return check(*args)
    except ValueError as exc:
        raise ValueError(f'{label}: {exc}') from exc


class BrillouinDrive:
    """One brillouin_scattering feature on a solid: its settings (module header) and, once bound by the study hook,
    the drive columns and per-frequency records."""

    def __init__(self, name, optical_physics, collection_physics, incident_source, collection_source, detection,
                 temperature, photoelastic, moving_boundary, axial_length, scan_offsets, reference, fit_peak,
                 element_mask, mirror_factor, crystal_regions):
        label = f"brillouin_scattering '{name}'"
        if not isinstance(optical_physics, str) or not optical_physics:
            raise ValueError(f'{label}: optical_physics must name an optical physics, got {optical_physics!r}')
        for what, value in (('collection_physics', collection_physics), ('incident_source', incident_source),
                            ('collection_source', collection_source)):
            if not isinstance(value, str):
                raise ValueError(f"{label}: {what} must be a name ('' = the default), got {value!r}")
        detection = _labelled(label, _checked_detection, detection)
        temperature = _labelled(label, _temperature, temperature)
        if reference not in REFERENCES:
            raise ValueError(f"{label}: reference must be one of {', '.join(REFERENCES)}, got {reference!r}")
        axial_length, mirror_factor = float(axial_length), float(mirror_factor)
        if not (axial_length > 0.0 and math.isfinite(axial_length)):
            raise ValueError(f'{label}: axial_length must be > 0 m, got {axial_length!r}')
        if not (mirror_factor >= 1.0 and math.isfinite(mirror_factor)):
            raise ValueError(f'{label}: mirror_factor must be >= 1 (2 per model mirror plane), got {mirror_factor!r}')
        if not (photoelastic or moving_boundary):
            raise ValueError(f'{label}: photoelastic and moving_boundary are both off: nothing drives the solid')
        offsets = np.array(scan_offsets, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets) or not np.isfinite(offsets).all():
            raise ValueError(f'{label}: scan_offsets must be (n >= 1, 2) finite focus offsets [m] '
                             f'(brillouin.scan_positions), got shape {offsets.shape}')
        offsets.flags.writeable = False
        self.name = name
        self.optical_physics = optical_physics
        self.collection_physics = collection_physics or optical_physics
        self.incident_source = incident_source
        self.collection_source = collection_source
        self.detection = detection
        self.temperature = temperature
        self.photoelastic = bool(photoelastic)
        self.moving_boundary = bool(moving_boundary)
        self.axial_length = axial_length
        self.scan_offsets = offsets
        self.reference = reference
        self.fit_peak = bool(fit_peak)
        self.element_mask = None if element_mask is None else np.array(element_mask, dtype=bool)
        self.mirror_factor = mirror_factor
        self.crystal_regions = _labelled(label, crystal_regions_checked, tuple(
            (np.array(mask, dtype=bool), np.array(tensor, dtype=np.float64)) for mask, tensor in crystal_regions))
        self._bound = None    # _Bound of the last bind (None: not bound)
        self._records = []    # [(frequency [Hz], psd per position (n_positions,))] of the frequency_response run
        self._result = None   # result() of the current _bound and _records (None: not built); _changed() drops it

    @property
    def n_positions(self) -> int:
        return len(self.scan_offsets)

    def _changed(self) -> None:
        """Drop the kept result: every writer of _bound / _records calls it."""
        self._result = None

    @property
    def _label(self) -> str:
        return f"brillouin_scattering '{self.name}'"

    def _require_bound(self) -> _Bound:
        if self._bound is None:
            raise ValueError(f'{self._label}: not bound yet (the frequency_response study binds the drive after its '
                             'background stage)')
        return self._bound

    # ---- bind ----
    def bind(self, solid, physics_by_name, progress=None) -> tuple:
        """(columns complex128 (n_positions x n_parts, solid.n_unknowns): position-major, each the drive kappa (w_pe +
        w_mb) on the u rows; [(log level, text)]) of the solid (SolidDriveInputs) against the solved optics of
        physics_by_name (module header); progress(done, n_positions) after each position (it may raise the study's
        Cancelled: the previous bound state is kept). ValueError naming the value for every mismatch."""
        label = self._label
        if not isinstance(solid, (SolidDriveInputs, CellDriveInputs)):   # _geometry relies on it
            raise TypeError(f'{label}: bind takes the SolidDriveInputs (triangles) or CellDriveInputs (rectangular '
                            f'cells) of the solid, got {type(solid).__name__}')
        planar = not solid.axisymmetric
        parts = self._parts(solid)
        optics, collection, incident, collected = self._resolve(physics_by_name)
        if self.reference == 'homogeneous' and (not planar or self.mirror_factor > 1.0):
            raise ValueError(f"{label}: reference 'homogeneous' needs a planar solid without model mirror planes (its "
                             "plane-wave sum takes the unfolded drive); use reference 'local'")
        if self.crystal_regions and not planar:
            raise ValueError(f'{label}: crystal photoelastic tensors are planar only (an axisymmetric drive needs an '
                             'isotropic, rotationally invariant p)')
        selection = self._selection(solid)
        geometry = _geometry(solid, selection, label, self.photoelastic and self.n_positions * len(parts) > 1)
        points, n_vector = geometry.points, geometry.n_vector
        if solid.n_unknowns < n_vector:
            raise ValueError(f'{label}: the solid has {solid.n_unknowns} unknowns, fewer than its {n_vector} '
                             'displacement dofs (a bug)')
        p11, p12, p44 = self._photoelastic(solid, points.elements)
        crystals = [(mask[points.elements], tensor) for mask, tensor in _labelled(
            f'{label} (masks per solid mesh element)', crystal_regions_checked, self.crystal_regions, selection.shape)]
        scale = self.mirror_factor / self.axial_length if planar else self.mirror_factor
        same_field = collected is incident
        collection_grid, collection_check = None, None
        if not same_field:   # the collection field stays at its own focus (R10: the scan moves the illumination)
            fields = self._optical(self.collection_physics, lambda: collection.drive_fields(
                collection.solve_moved_beam(collected, (0.0, 0.0))))
            collection_grid, _ = _optical_arrays(fields, planar, label)
            collection_check = (float(fields.omega), np.asarray(fields.x, dtype=np.float64),
                                np.asarray(fields.y, dtype=np.float64), collection_grid.grids)
            del fields
        n_parts = len(parts)
        columns = np.zeros((self.n_positions * n_parts, solid.n_unknowns), dtype=np.complex128)
        context = kappa = None
        torsion, norms, q_mean, lines, rasters = 0.0, {}, [], [], []
        for position, offset in enumerate(self.scan_offsets):
            offset = (float(offset[0]), float(offset[1]))
            fields = self._optical(self.optical_physics, lambda offset=offset: optics.drive_fields(
                optics.solve_moved_beam(incident, offset)))
            grid, eps = _optical_arrays(fields, planar, label)
            if context is None:
                context = self._context(solid, geometry, selection, fields, grid, eps, planar, collection_check)
                kappa = amplitude_prefactor(context.omega, incident.incident_power, collected.incident_power)
                if not self.photoelastic and context.interfaces is None:
                    raise ValueError(f'{label}: photoelastic is off and no optical interface touches the drive '
                                     'selection: nothing drives the solid (turn photoelastic on or select the '
                                     'interfaces)')
            elif not (float(fields.omega) == context.omega and np.array_equal(fields.x, context.x)
                      and np.array_equal(fields.y, context.y) and _same_grids(grid.grids, context.grids)):
                raise ValueError(f'{label}: the optics changed their grid or wavelength between scan positions (a bug)')
            del fields, eps   # the solution of this position: only its window samples are kept below
            partner = grid if same_field else collection_grid
            incident_points = context.volume.fields(grid)
            collection_points = incident_points if same_field else context.volume.fields(partner)
            if planar:
                stresses = [solid_rows(optical_stress(field_products(incident_points, collection_points,
                                                                     context.eps_points), p11, p12, p44, crystals))]
            else:
                split, share = azimuthal_parts(incident_points, collection_points, context.eps_points, self.detection,
                                               p11, p12, p44, weights=points.weights / solid.ring_angle)
                stresses = [split[key] for key in parts]
                torsion = max(torsion, share)
            boundary = self._boundary_values(context, grid, partner, same_field, parts, planar)
            squares = [0.0, 0.0]
            for index, stress in enumerate(stresses):
                column = columns[position * n_parts + index, :n_vector]
                if self.photoelastic:
                    drive = geometry.stress_drive(kappa * stress)
                    column += drive
                    squares[0] += float(np.vdot(drive, drive).real)
                if boundary is not None:
                    drive = geometry.traction_drive(context.interfaces, kappa * boundary[index])
                    column += drive
                    squares[1] += float(np.vdot(drive, drive).real)
            if position == 0:
                norms = {'photoelastic': math.sqrt(squares[0]), 'moving_boundary': math.sqrt(squares[1])}
            mean, position_lines = self._local_lines(context, points, grid, partner, stresses, kappa, planar)
            q_mean.append(mean)
            if self.reference == 'local':
                lines.append(position_lines)
            elif self.reference == 'homogeneous':
                rasters.append(self._raster_values(context.raster, grid, partner, same_field, kappa))
            if progress is not None:
                progress(position + 1, self.n_positions)
        interfaces = context.interfaces
        self._bound = _Bound(
            planar=planar, azimuthal_order=solid.azimuthal_order, n_parts=n_parts, scale=scale, kappa=kappa,
            wavelength=float(incident.wavelength), incident_power=float(incident.incident_power),
            collection_power=float(collected.incident_power), torsion=torsion, norms=norms, q_mean=tuple(q_mean),
            laws=context.laws, lines=tuple(lines), rasters=tuple(rasters), raster=context.raster,
            n_points=points.n_points, n_edge_points=0 if interfaces is None else int(np.prod(interfaces.shape)))
        self._records = []
        self._changed()
        messages = [('info', self.summary())]
        if torsion > TORSION_WARNING:
            messages.append(('warning', f'{label}: the dropped azimuthal order 0 torsion holds {torsion:.3g} of the '
                                        'drive (cross-polarised detection; not modelled: no Brillouin peak in an '
                                        'isotropic liquid, but a solid with shear peaks is under-counted)'))
        if self.moving_boundary and interfaces is None:
            messages.append(('info', f'{label}: no optical interface touches the drive selection (no moving-boundary '
                                     'part)'))
        fallbacks = context.fallbacks + (0 if interfaces is None else interfaces.fallbacks)
        if fallbacks:
            messages.append(('warning', f'{label}: {context.fallbacks:,} of {points.n_points:,} drive points and '
                                        f'{0 if interfaces is None else interfaces.fallbacks:,} interface side samples '
                                        'found no optical cell of their own material (equal solid material values) '
                                        'among the four nearest cell centres: a solid region thinner than an optical '
                                        'cell (refine the optical mesh there), or solid material values that vary in '
                                        'space; they take the staircase permittivity of the nearest cell'))
        return columns, messages

    def _parts(self, solid) -> tuple:
        """Azimuthal parts (m, parity) of the solid's order ((None,) planar); ValueError for an order / detection with
        none."""
        label = self._label
        if not solid.axisymmetric:
            if self.detection != 'co_polarized':
                raise ValueError(f'{label}: detection {self.detection} needs em_fdfd_bor optics and an axisymmetric '
                                 'solid (a planar drive is co-polarised)')
            return (None,)
        order = solid.azimuthal_order
        if order not in (0, 2):
            raise ValueError(f'{label}: an axisymmetric solid of azimuthal_order {order}: two order-1 optical fields '
                             'drive orders 0 and 2 only')
        parts = tuple(key for key in AZIMUTHAL_PARTS[self.detection] if key[0] == order)
        if not parts:
            raise ValueError(f'{label}: detection {self.detection} drives no part of azimuthal order {order} '
                             '(co_polarized: orders 0 and 2, even; cross_polarized: order 2, odd; the order-0 torsion '
                             'is not modelled)')
        return parts

    def _resolve(self, physics_by_name) -> tuple:
        """(optics, collection physics, incident beam, collection beam) by name (module header)."""
        optics = self._optics(physics_by_name, self.optical_physics, 'optical_physics')
        collection = self._optics(physics_by_name, self.collection_physics, 'collection_physics')
        incident = self._source(optics, self.incident_source, 'incident_source', self.optical_physics)
        if collection is optics:
            if self.collection_source and self.collection_source != incident.name:
                raise ValueError(f"{self._label}: collection_source '{self.collection_source}' is another beam of the "
                                 f"optical physics '{self.optical_physics}', whose one solution sums every beam: give "
                                 'the collection beam its own collection_physics, or leave collection_source empty '
                                 '(confocal: the illumination is the collection field)')
            return optics, collection, incident, incident
        collected = self._source(collection, self.collection_source, 'collection_source', self.collection_physics)
        return optics, collection, incident, collected

    def _optics(self, physics_by_name, name, what):
        solver = physics_by_name.get(name)
        if solver is None:
            raise ValueError(f"{self._label}: {what} '{name}' is not a physics of the model (physics: "
                             f"{', '.join(physics_by_name)})")
        if not isinstance(solver, DriveOpticsSolver):
            raise ValueError(f"{self._label}: {what} '{name}' ({type(solver).__name__}) has no focused-beam drive: "
                             'name an em_fdfd_ez, em_fdfd_hz or em_fdfd_bor physics with a focused_beam feature')
        kind = getattr(solver.domain, 'grid_kind', 'tensor')
        if kind != 'tensor':
            raise ValueError(f"{self._label}: {what} '{name}' runs on a {kind} mesh; the drive reads optics on a "
                             'rectangular (tensor) mesh only')
        return solver

    def _source(self, optics, wanted, what, physics):
        sources = optics.drive_sources()
        if wanted:
            if wanted not in sources:
                raise ValueError(f"{self._label}: {what} '{wanted}' is not an enabled focused beam of physics "
                                 f"'{physics}' (its focused beams: {', '.join(sources) or 'none'})")
            return sources[wanted]
        if len(sources) != 1:
            remedy = f'name one in {what}' if sources else 'add a focused_beam feature'
            raise ValueError(f"{self._label}: physics '{physics}' has {len(sources)} enabled focused beams "
                             f"({', '.join(sources) or 'none'}): {remedy}")
        return next(iter(sources.values()))

    def _optical(self, physics, call):
        """call() (an optical solve / field read), its ValueError prefixed with the feature and physics names."""
        try:
            return call()
        except ValueError as exc:
            raise ValueError(f"{self._label}: optical physics '{physics}': {exc} (the drive reads the optics as the "
                             "study's background stage solved them and never re-solves them)") from exc

    def _selection(self, solid) -> np.ndarray:
        """Bool per mesh element: the drive selection within the solved members."""
        selection = np.asarray(solid.solid, dtype=bool)
        if self.element_mask is not None:
            if self.element_mask.shape != selection.shape:
                raise ValueError(f'{self._label}: element_mask has {self.element_mask.size} values, the solid mesh '
                                 f'{selection.size} elements')
            selection = selection & self.element_mask
        if not selection.any():
            raise ValueError(f'{self._label}: the drive selection holds no element the solid solves (a void or '
                             'unselected region)')
        return selection

    def _material(self, solid, name) -> np.ndarray:
        values = solid.materials.get(name)
        if values is None:
            raise ValueError(f'{self._label}: the solid has no material property {name} (a bug: it is in the solid '
                             'material schema)')
        return values

    def _photoelastic(self, solid, rows) -> tuple:
        """(p11, p12, p44 or None (unset: (p11 - p12) / 2)) of the solid's material rows (elements / cells)."""
        p44 = solid.materials.get('photoelastic_p44')
        return (self._material(solid, 'photoelastic_p11')[rows], self._material(solid, 'photoelastic_p12')[rows],
                None if p44 is None else p44[rows])

    def _context(self, solid, geometry, selection, fields, optical, eps, planar, collection_check) -> _Context:
        """Sampling state of the optical grid of the first position: the volume sampler, own-material eps at the points
        (PEC refused), the interfaces, the reference laws per point and the homogeneous raster cells."""
        label = self._label
        x, y = np.asarray(fields.x, dtype=np.float64), np.asarray(fields.y, dtype=np.float64)
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f'{label}: the optical window has {len(x)} x {len(y)} cells (at least 2 x 2)')
        omega = float(fields.omega)
        if collection_check is not None:
            collection_omega, collection_x, collection_y, collection_grids = collection_check
            if abs(collection_omega - omega) > _OMEGA_RTOL * omega:
                raise ValueError(f"{label}: collection_physics '{self.collection_physics}' solves omega "
                                 f'{collection_omega:.9g} rad/s, the optics {omega:.9g} rad/s: one wavelength')
            if not (np.array_equal(collection_x, x) and np.array_equal(collection_y, y)
                    and _same_grids(collection_grids, optical.grids)):
                raise ValueError(f"{label}: collection_physics '{self.collection_physics}' runs on another optical "
                                 'grid: the drive multiplies both fields cell by cell (put both on one mesh)')
        volume = geometry.sampler(x, y, optical.grids, f'{label}: drive points')
        cells, matched = geometry.own_cells(x, y, volume)
        eps_points = {key: values.reshape(-1)[cells] for key, values in eps.items()}
        if fields.pec is not None:
            inside = np.asarray(fields.pec, dtype=bool).reshape(-1)[cells]
            if inside.any():
                first = int(cells[int(np.argmax(inside))])
                where = (x[first // len(y)], y[first % len(y)])
                raise ValueError(f'{label}: {int(inside.sum())} drive points lie in PEC cells of the optics (first in '
                                 f'the cell at ({where[0]:.6g}, {where[1]:.6g}) m): keep the drive selection out of '
                                 'conductors')
        del cells
        interfaces = geometry.interfaces(selection, x, y, eps) if self.moving_boundary else None
        eps_in_plane = {}
        if interfaces is not None and interfaces.sampler is not None:   # the triangle moving boundary's D = eps E
            eps_in_plane = {name + name: eps[name + name] for name in interfaces.sampler.in_plane
                            if name in optical.centred}
        point_laws, laws, raster = None, (), None
        if self.reference != 'none':
            element_laws, laws = self._reference_laws(solid, selection)
            point_laws = element_laws[geometry.points.elements]
            if self.reference == 'homogeneous':
                raster = self._raster_cells(solid, geometry, selection, x, y, eps, element_laws)
        return _Context(x=x, y=y, omega=omega, grids=optical.grids, volume=volume, eps_points=eps_points,
                        eps_in_plane=eps_in_plane, interfaces=interfaces, fallbacks=int((~matched).sum()),
                        point_laws=point_laws, laws=laws, raster=raster)

    def _boundary_values(self, context, incident, partner, same_field, parts, planar):
        """[Johnson value f (k, q) per part] of one position (incident / partner: _Optical), None without
        interfaces."""
        interfaces = context.interfaces
        if interfaces is None:
            return None
        normals = interfaces.points.normals
        if interfaces.sampler is not None:   # triangles: E_par and D_n at the interface points themselves
            sampler = interfaces.sampler
            fields_i, displacement_i = sampler.fields(incident, context.eps_in_plane, interfaces.shape)
            fields_c, displacement_c = ((fields_i, displacement_i) if same_field else
                                        sampler.fields(partner, context.eps_in_plane, interfaces.shape))
            if planar:
                return [interface_traction(normals, fields_i, fields_c, displacement_i, displacement_c,
                                           interfaces.eps_sides)]
            values = azimuthal_interface_traction(normals, fields_i, fields_c, displacement_i, displacement_c,
                                                  interfaces.eps_sides, self.detection)
            return [values[key] for key in parts]

        def sides(values):   # rectangular cells (CellDriveInputs: planar)
            return tuple(side.fields(values) for side in interfaces.sides)
        incident_sides = sides(incident)
        collection_sides = incident_sides if same_field else sides(partner)
        return [moving_boundary_traction(normals, incident_sides, collection_sides, interfaces.eps_sides)]

    def _reference_laws(self, solid, selection) -> tuple:
        """(int64 law index per mesh element (-1 outside the selection), ((longitudinal law, shear law, density),
        ...)) of the drive selection: one law per viscoelastic parameter group and density."""
        label = self._label
        moduli = solid.moduli
        anisotropic = selection & ~(moduli.youngs_route | moduli.moduli_route)
        if anisotropic.any():
            raise ValueError(f"{label}: reference '{self.reference}' needs isotropic moduli with a finite bulk "
                             f"modulus, but {int(anisotropic.sum())} drive element(s) take neither the youngs_modulus "
                             'nor the shear_modulus / bulk_modulus route (a material of the c_ij route, or an '
                             'incompressible / viscosity-only liquid without bulk_modulus): set reference '
                             "'none' (or give them youngs_modulus / poisson_ratio, or a bulk_modulus)")
        density = self._material(solid, 'rho')
        index = np.full(selection.size, -1, dtype=np.int64)
        laws = []
        for group in parameter_groups(moduli, selection, lambda: label):
            if group.bulk.incompressible:
                raise ValueError(f"{label}: reference '{self.reference}' needs a finite bulk modulus in the drive "
                                 "selection (an incompressible material has no longitudinal sound wave): set "
                                 "reference 'none' or give it a bulk_modulus")
            shear = group.shear.at
            longitudinal = longitudinal_function(group.bulk.at, shear)
            elements = np.flatnonzero(group.elements)
            values, inverse = np.unique(density[elements], return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            for value_index, value in enumerate(values):
                index[elements[inverse == value_index]] = len(laws)
                laws.append((longitudinal, shear, float(value)))
        return index, tuple(laws)

    def _local_lines(self, context, points, grid, partner, stresses, kappa, planar) -> tuple:
        """(q_mean [1/m], ((law, q, weight), ...) compressed lines or ()) of one position: the local wavevector of P =
        E_c . E_i (cell-centre values), sigma_L = q^T sigma~ q / q^2 per point and part, weights |kappa sigma_L|^2 dV
        summed over parts."""
        product = sum(partner.centred[name] * values for name, values in grid.centred.items())
        q_x, q_y = (context.volume.sample(values) for values in local_wavenumbers(product, context.x, context.y))
        q_squared = q_x * q_x + q_y * q_y
        safe = np.where(q_squared > 0.0, q_squared, 1.0)
        aa, bb, ab = _PLANE_ROWS[not planar]
        weights = np.zeros(points.n_points)
        for stress in stresses:
            longitudinal = (q_x * q_x * stress[aa] + q_y * q_y * stress[bb] + 2.0 * q_x * q_y * stress[ab]) / safe
            weights += np.abs(kappa * longitudinal) ** 2
        weights *= points.weights
        weights[~(q_squared > 0.0)] = 0.0
        q = np.sqrt(q_squared)
        total = float(weights.sum())
        mean = float(weights @ q) / total if total > 0.0 else math.nan
        if self.reference != 'local':
            return mean, ()
        keep = (weights > 0.0) & (context.point_laws >= 0)
        lines = []
        for law in np.unique(context.point_laws[keep]):
            selected = keep & (context.point_laws == law)
            values, masses = q[selected], weights[selected]
            bins = np.floor(np.log(values) / math.log1p(LINE_BIN)).astype(np.int64)
            unique, inverse = np.unique(bins, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            mass = np.bincount(inverse, weights=masses, minlength=len(unique))
            moment = np.bincount(inverse, weights=masses * values, minlength=len(unique))
            lines.append((int(law), moment / mass, mass))
        return mean, tuple(lines)

    def _raster_cells(self, solid, geometry, selection, x, y, eps, element_laws) -> _Raster:
        """_Raster of the optical cells whose centres lie in the drive selection (one uniform isotropic material)."""
        label = self._label
        if any(mask[selection].any() for mask, _ in self.crystal_regions):
            raise ValueError(f"{label}: reference 'homogeneous' needs an isotropic medium (a crystal region lies in "
                             "the drive selection): use reference 'local'")
        cells, cell_elements = geometry.raster_cells(x, y, selection)
        if not cells.size:
            raise ValueError(f"{label}: reference 'homogeneous': no optical cell centre lies in the drive selection")
        cell_laws = np.unique(element_laws[cell_elements])
        if cell_laws.size != 1 or cell_laws[0] < 0:
            raise ValueError(f"{label}: reference 'homogeneous' needs one uniform material over the drive selection, "
                             f"got {cell_laws.size} (moduli, loss and density sets): use reference 'local'")
        i, j = np.unravel_index(cells, (len(x), len(y)))
        i0, j0 = int(i.min()), int(j.min())
        shape = (int(i.max()) - i0 + 1, int(j.max()) - j0 + 1)
        dx = _uniform_spacing(x[i0:i0 + shape[0]], f'{label} x')
        dy = _uniform_spacing(y[j0:j0 + shape[1]], f'{label} y')
        p11, p12, p44 = self._photoelastic(solid, cell_elements)
        return _Raster(cells=cells, i=i - i0, j=j - j0, shape=shape, dx=dx, dy=dy, law=int(cell_laws[0]), p11=p11,
                       p12=p12, p44=p44, eps={key: values.reshape(-1)[cells] for key, values in eps.items()})

    def _raster_values(self, raster, grid, partner, same_field, kappa) -> np.ndarray:
        """kappa sigma~ (3, n_x, n_y) rows (xx, yy, xy) of one position on the raster box (0 outside the region), of the
        cell-centre fields (the edge grids' phase-corrected half steps: no face mean)."""
        incident = {name: values.reshape(-1)[raster.cells] for name, values in grid.centred.items()}
        collection = incident if same_field else {name: values.reshape(-1)[raster.cells]
                                                  for name, values in partner.centred.items()}
        stress = solid_rows(optical_stress(field_products(incident, collection, raster.eps), raster.p11, raster.p12,
                                           raster.p44))
        image = np.zeros((3,) + raster.shape, dtype=np.complex128)
        image[:, raster.i, raster.j] = kappa * stress
        return image

    # ---- records and result ----
    def record(self, frequency, readouts) -> None:
        """Record one frequency [Hz]: readouts w^H u (complex, one per column in bind's order) -> psd per position
        (fdt_psd x scale, parts summed); ValueError on an active readout."""
        bound = self._require_bound()
        readouts = np.asarray(readouts, dtype=np.complex128).reshape(-1)
        if readouts.size != self.n_positions * bound.n_parts:
            raise ValueError(f'{self._label}: {readouts.size} readouts, the drive has '
                             f'{self.n_positions * bound.n_parts} columns')
        frequency = float(frequency)
        try:
            psd = fdt_psd(2.0 * math.pi * frequency, readouts, self.temperature, bound.scale)
        except ValueError as exc:
            raise ValueError(f'{self._label} at {frequency:.6g} Hz: {exc}') from exc
        self._records.append((frequency, psd.reshape(self.n_positions, bound.n_parts).sum(axis=1)))
        self._changed()

    def reset_records(self) -> None:
        """Forget the per-frequency records (the bound columns stay)."""
        self._records = []
        self._changed()

    def result(self) -> dict:
        """The 'xy' result (brillouin.brillouin_result) of the recorded frequencies (ascending), with the reference
        and the extras; ValueError before a frequency_response run or for a frequency solved twice. Built once per
        bind and records (the references and peak fits are its cost) and handed out as a deep copy (nested dicts,
        lists and arrays: callers may change what they get)."""
        if self._result is None:
            self._result = self._built_result()
        return copy.deepcopy(self._result)

    def _built_result(self) -> dict:
        bound = self._require_bound()
        if not self._records:
            raise ValueError('not solved yet: the Brillouin spectrum needs a frequency_response study')
        frequencies = np.array([frequency for frequency, _ in self._records], dtype=np.float64)
        order = np.argsort(frequencies, kind='stable')
        frequencies = frequencies[order]
        repeated = frequencies[1:][np.diff(frequencies) <= 0.0]
        if repeated.size:
            listed = ', '.join(f'{value:.6g}' for value in repeated[:5])
            raise ValueError(f'{self._label}: frequencies solved twice ({listed} Hz): list each frequency once')
        psd = np.stack([self._records[index][1] for index in order], axis=1)   # (positions, frequencies)
        several = self.n_positions > 1
        extras = {'incident_power': bound.incident_power, 'collection_power': bound.collection_power,
                  'wavelength': bound.wavelength, 'mirror_factor': self.mirror_factor,
                  'azimuthal_order': bound.azimuthal_order,
                  'axial_length': self.axial_length if bound.planar else math.nan,
                  'torsion_drive_fraction': bound.torsion, 'drive_norms': dict(bound.norms)}
        for index, value in enumerate(bound.q_mean):
            extras[f'q_mean_{index + 1}' if several else 'q_mean'] = value
        return brillouin_result(BrillouinSpectra(frequencies=frequencies, psd=psd,
                                                 reference=self._reference(bound, frequencies),
                                                 offsets=self.scan_offsets, temperature=self.temperature,
                                                 fit_peak=self.fit_peak, extras=extras))

    def _reference(self, bound, frequencies):
        """(n_positions, n_f) reference psd [1/Hz] (scale applied), None for reference 'none'."""
        if self.reference == 'none':
            return None
        reference = np.zeros((self.n_positions, len(frequencies)))
        for position in range(self.n_positions):
            if self.reference == 'local':
                for law, q, weights in bound.lines[position]:
                    longitudinal, _, density = bound.laws[law]
                    reference[position] += local_reference(frequencies, weights, q, longitudinal, density,
                                                           self.temperature, bound.scale)
            else:
                raster = bound.raster
                longitudinal, shear, density = bound.laws[raster.law]
                reference[position] = homogeneous_reference(frequencies, bound.rasters[position], raster.dx, raster.dy,
                                                            longitudinal, shear, density, self.temperature,
                                                            bound.scale)
        return reference

    def summary(self) -> str:
        """One line on the bound drive (columns, points, powers, wavelength, kappa, scale, drive norms of position 1,
        the dropped torsion share)."""
        if self._bound is None:
            return f'{self._label}: not bound yet'
        bound = self._bound
        unit = 'W/m' if bound.planar else 'W'
        text = (f'{self._label}: {self.n_positions} scan position(s) x {bound.n_parts} part(s) = '
                f'{self.n_positions * bound.n_parts} drive column(s) on {bound.n_points:,} drive points and '
                f'{bound.n_edge_points:,} interface points; incident {bound.incident_power:.6g} {unit}, collection '
                f'{bound.collection_power:.6g} {unit}, wavelength {bound.wavelength:.6g} m, kappa {bound.kappa:.6g}, '
                f'scale {bound.scale:.6g}; |w| photoelastic {bound.norms["photoelastic"]:.4g}, moving boundary '
                f'{bound.norms["moving_boundary"]:.4g} (position 1)')
        if not bound.planar:
            text += f'; dropped order-0 torsion share {bound.torsion:.3g}'
        return text

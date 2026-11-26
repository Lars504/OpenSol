"""Runtime of the brillouin_scattering feature (frequency_response): the bound optical fields of every scan position
sampled at the solid's quadrature points, the complex drive columns, the per-frequency FDT records and the result."""
# The constructor is the FROZEN contract of the feature's apply (definition_brillouin); the math is brillouin.py, the
# FEM assembly fem.elastic_drives. Constructor values (all checked here, ValueError naming the value):
# - name: the feature name; optical_physics: the em_fdfd_ez / _hz / _bor physics name; collection_physics: '' = the
#   optical physics; incident_source / collection_source: focused_beam feature names ('' = the only enabled one);
# - detection: brillouin.DETECTIONS; temperature [K] > 0; photoelastic / moving_boundary: bools, at least one True;
# - axial_length [m] > 0 (planar solids only: scale mirror_factor / L; axisymmetric solids ignore it);
# - scan_offsets: (n, 2) focus offsets [m] in position order, brillouin.scan_positions(offsets_x, offsets_y) (x outer,
#   y inner; BOR: (0, dz)); position 1 = row 0 = the shown solution and the reduced-sweep load;
# - reference: brillouin.REFERENCES; fit_peak: bool;
# - element_mask: bool per solid mesh element of the drive selection (None = every solid element);
# - mirror_factor: 2 ** (model mirror planes) (core.mirror.mirror_faces(spec)), >= 1;
# - crystal_regions: ((bool mask per solid mesh element, (6, 6) Voigt p in model axes), ...) (planar only).
# Flow (SolidMechanicsFEMPhysics.bind_physics_drives, called by the frequency_response study after its background
# stage and the static states, before reset_frequency_response):
# - bind(SolidDriveInputs, physics_by_name, progress) -> (columns (n_positions x n_parts, n_unknowns) complex128 on the
#   u rows (mixed: the pressure rows stay 0), [(log level, text)]): per scan position one optical solve on the optics'
#   kept factor (solve_moved_beam: only the illumination moves; the collection field is the collection physics'
#   solution, fixed: transmission detection does not co-move) and its DriveFields, then the column kappa (int B^T
#   sigma~ dV + int N^T n f ds). The optics are read as the background stage left them: a system or source changed
#   since (a coupling or material update) raises in solve_moved_beam / drive_fields; nothing here re-solves them.
#   Columns per position: planar one; axisymmetric one per part of brillouin.AZIMUTHAL_PARTS[detection] at the solid's
#   azimuthal order (0: (0, even); 2: (2, even) co- / (2, odd) cross-polarised), summed per position in record.
# - Sampling: fields bilinear between the window cell centres at the DrivePoints (points of the half cell between the
#   outermost centres and the window edge, i.e. within one centre spacing of the centres' hull, take the hull value;
#   farther ones raise: brillouin.bilinear_weights), eps the staircase value of the containing (nearest-centre) cell;
#   a drive point in a PEC cell raises. The clamped edge half cell is the even mirror rule on a mirror-plane edge
#   (bilinear between a centre and its even image); em_fdfd_bor adds the axis rule on x_lo (r = 0): the order-1 E_z
#   ~ r is scaled by r / r_0 below the first centre r_0 (E_r, A_phi stay finite there). An odd field component on a
#   mirror plane keeps its edge value in that half cell (no mirror signs here: an O(half cell) error of the strip).
# - Moving boundary: interior edges between two window elements, touching the drive selection, whose sides differ in
#   a solid material value (the conforming solid mesh marks material interfaces exactly; an optical interface inside
#   one solid material, or between two materials of equal mechanics, is not found) AND in the optical eps of the cells
#   containing the points one local cell diagonal off the edge midpoint into each side (so the centre-tested staircase
#   cell lies on that side of a locally straight interface); per Gauss point the fields and eps of each side are read
#   at those shifted points, f = the Johnson value (isotropic eps per side: the mean of the component eps).
# - record(frequency, readouts w^H u per column): psd per position (brillouin.fdt_psd x scale, parts summed; an active
#   readout raises). result(): frequencies ascending, the references describe the photoelastic drive: local = per
#   position and material group the DHO lines of the points, compressed in relative |q| bins of LINE_BIN with the
#   weighted-mean q per bin (the line shape is smooth in q: a 1e-4 bin shifts no line by more than ~1e-8 of its
#   width); homogeneous = the kappa sigma~ raster of the optical cells inside the drive selection (planar, no mirror
#   planes: the FFT needs the unfolded drive, one uniform isotropic material, uniform cells over its box).
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ...fem.elastic_drives import assemble_edge_drive, assemble_stress_drive, drive_points, edge_drive_points
from .brillouin import (AZIMUTHAL_PARTS, DETECTIONS, PLANAR_COMPONENTS, REFERENCES, BrillouinSpectra,
                        amplitude_prefactor, azimuthal_moving_boundary, azimuthal_parts, bilinear_sample,
                        bilinear_weights, brillouin_result, fdt_psd, field_products, homogeneous_reference,
                        local_reference, local_wavenumbers, longitudinal_function, modulus_function,
                        moving_boundary_traction, optical_stress, solid_rows)
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
# (aa, bb, ab) rows of the solid stress the longitudinal projection reads: planar (xx, yy, xy), axisymmetric
# (rr, zz, rz)
_PLANE_ROWS = {False: (0, 1, 2), True: (0, 1, 3)}


@runtime_checkable
class DriveOptics(Protocol):
    """An optical physics a solid drive reads (em_fdfd_ez / _hz, em_fdfd_bor): its focused beams by name, one solve
    with a beam moved on its kept factor and the host DriveFields of a solution (em focused_beam.DriveFields)."""
    domain: object
    def drive_sources(self): ...
    def solve_moved_beam(self, source, offset): ...
    def drive_fields(self, solution): ...


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


class _GridSamples:
    """Bilinear samples of window cell arrays (n_x, n_y) at fixed points (brillouin.bilinear_weights, the edge half
    cells clamped: module header) and the containing (largest-weight) cell of each point; axis=True (em_fdfd_bor): the
    axis rule in the half cell 0 <= r < r_0 of the first centre, where the order-1 E_z (~ r) is scaled by r / r_0."""

    def __init__(self, x, y, points, what, axis=False):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        try:
            self.index, self.weight = bilinear_weights(x, y, _clamped(x, y, points))
        except ValueError as exc:
            raise ValueError(f'{what}: {exc}') from exc
        self.cell = self.index[np.arange(len(self.index)), np.argmax(self.weight, axis=1)]
        self.axis_factor = None   # (n,) r / r_0 below the first centre (1 elsewhere), or None
        if axis and (points[:, 0] < x[0]).any():
            self.axis_factor = np.clip(points[:, 0] / x[0], 0.0, 1.0)

    def sample(self, values) -> np.ndarray:
        return bilinear_sample(values, self.index, self.weight)

    def fields(self, components) -> dict:
        """{name: samples} of brillouin field components (the axis rule on 'z' of a BOR sampler)."""
        samples = {name: self.sample(values) for name, values in components.items()}
        if self.axis_factor is not None and 'z' in samples:
            samples['z'] = samples['z'] * self.axis_factor
        return samples

    def cells(self, values) -> np.ndarray:
        return np.asarray(values).reshape(-1)[self.cell]


def _clamped(x, y, points) -> np.ndarray:
    """(n, 2) copy of points with coordinates within one centre spacing outside the centres' hull moved onto it."""
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    for axis, centres in enumerate((x, y)):
        low, high = centres[0], centres[-1]
        values = points[:, axis]
        inside = (values >= low - (centres[1] - centres[0])) & (values <= high + (centres[-1] - centres[-2]))
        points[:, axis] = np.where(inside, np.clip(values, low, high), values)
    return points


def _side_shift(x, y, points) -> np.ndarray:
    """(n,) diagonal of the local cell-centre spacing at points (the side-sample distance off an interface)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    widths = []
    for axis, centres in enumerate((x, y)):
        index = np.clip(np.searchsorted(centres, points[:, axis]) - 1, 0, len(centres) - 2)
        widths.append(centres[index + 1] - centres[index])
    return np.hypot(*widths)


def _uniform_spacing(centres, what) -> float:
    """The cell-centre spacing of a run of centres; ValueError when it is not uniform or shorter than 2 cells."""
    steps = np.diff(np.asarray(centres, dtype=np.float64))
    if not steps.size or float(np.ptp(steps)) > _SPACING_RTOL * float(steps.mean()):
        raise ValueError(f'{what}: the homogeneous reference needs uniform optical cells (>= 2 per axis) over the '
                         "drive region's box; use reference 'local'")
    return float(steps.mean())


def _optical_arrays(fields, planar, label) -> tuple:
    """({brillouin component: complex (n_x, n_y)}, {'cc': complex eps (n_x, n_y)}) of DriveFields; ValueError when
    its components do not fit the solid's form."""
    names = _PLANAR_FIELDS.get(fields.polarization) if planar else _BOR_FIELDS
    keys = None if names is None else {key for key, _ in names}
    if keys is None or set(fields.fields) != keys or set(fields.eps) != keys:
        wanted = ('a planar solid: em_fdfd_ez {ez} or em_fdfd_hz {ex, ey}' if planar
                  else 'an axisymmetric solid: em_fdfd_bor {r, phi, z}')
        raise ValueError(f"{label}: the optical drive fields ({fields.polarization}: "
                         f"{', '.join(sorted(fields.fields))}) do not fit {wanted}")
    components = {name: np.asarray(fields.fields[key], dtype=np.complex128) for key, name in names}
    eps = {name + name: np.asarray(fields.eps[key], dtype=np.complex128) for key, name in names}
    return components, eps


@dataclass(frozen=True, eq=False)
class _Interfaces:
    """Moving-boundary samples: the EdgeDrivePoints, the two side samplers (side 0 = -shift n, side 1 = +shift n) of
    their flattened Gauss points, the isotropic eps of each side (k, q) and the (k, q) shape."""
    points: object
    sides: tuple
    eps_sides: tuple
    shape: tuple


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
    volume: _GridSamples
    eps_points: dict
    interfaces: object        # _Interfaces or None
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
        if detection not in DETECTIONS:
            raise ValueError(f"{label}: detection must be one of {', '.join(DETECTIONS)}, got {detection!r}")
        if reference not in REFERENCES:
            raise ValueError(f"{label}: reference must be one of {', '.join(REFERENCES)}, got {reference!r}")
        temperature, axial_length, mirror_factor = float(temperature), float(axial_length), float(mirror_factor)
        if not (temperature > 0.0 and math.isfinite(temperature)):
            raise ValueError(f'{label}: temperature must be > 0 K, got {temperature!r}')
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
        self.crystal_regions = tuple((np.array(mask, dtype=bool), np.array(tensor, dtype=np.float64))
                                     for mask, tensor in crystal_regions)
        for mask, tensor in self.crystal_regions:
            if tensor.shape != (6, 6) or mask.ndim != 1:
                raise ValueError(f'{label}: a crystal region is (mask per solid element, (6, 6) p), got '
                                 f'{mask.shape} and {tensor.shape}')
        self._bound = None    # _Bound of the last bind (None: not bound)
        self._records = []    # [(frequency [Hz], psd per position (n_positions,))] of the frequency_response run

    @property
    def n_positions(self) -> int:
        return len(self.scan_offsets)

    @property
    def n_columns(self) -> int:
        """Drive columns of the last bind (positions x azimuthal parts); ValueError before."""
        return self.n_positions * self._require_bound().n_parts

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
        if not isinstance(solid, SolidDriveInputs):
            raise TypeError(f'{self._label}: bind takes the SolidDriveInputs of the solid, got {type(solid).__name__}')
        label = self._label
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
        dofs = solid.dofs
        try:
            points = drive_points(dofs, axisymmetric=solid.axisymmetric, azimuthal_order=solid.azimuthal_order,
                                  elements=selection)
        except ValueError as exc:
            raise ValueError(f'{label}: {exc}') from exc
        n_vector = points.components * dofs.n_dofs
        if solid.n_unknowns < n_vector:
            raise ValueError(f'{label}: the solid has {solid.n_unknowns} unknowns, fewer than its {n_vector} '
                             'displacement dofs (a bug)')
        p11 = self._material(solid, 'photoelastic_p11')[points.elements]
        p12 = self._material(solid, 'photoelastic_p12')[points.elements]
        p44 = solid.materials.get('photoelastic_p44')
        p44 = None if p44 is None else p44[points.elements]
        crystals = []
        for mask, tensor in self.crystal_regions:
            if mask.shape != selection.shape:
                raise ValueError(f'{label}: a crystal mask has {mask.size} values, the solid mesh {selection.size} '
                                 'elements')
            crystals.append((mask[points.elements], tensor))
        scale = self.mirror_factor / self.axial_length if planar else self.mirror_factor
        same_field = collected is incident
        collection_grid, collection_check = None, None
        if not same_field:   # the collection field stays at its own focus (R10: the scan moves the illumination)
            fields = self._optical(self.collection_physics, lambda: collection.drive_fields(
                collection.solve_moved_beam(collected, (0.0, 0.0))))
            collection_grid, _ = _optical_arrays(fields, planar, label)
            collection_check = (float(fields.omega), np.asarray(fields.x, dtype=np.float64),
                                np.asarray(fields.y, dtype=np.float64))
            del fields
        n_parts = len(parts)
        columns = np.zeros((self.n_positions * n_parts, solid.n_unknowns), dtype=np.complex128)
        context = kappa = None
        torsion, norms, q_mean, lines, rasters = 0.0, {}, [], [], []
        for position, offset in enumerate(self.scan_offsets):
            offset = (float(offset[0]), float(offset[1]))
            fields = self._optical(self.optical_physics, lambda offset=offset: optics.drive_fields(
                optics.solve_moved_beam(incident, offset)))
            if context is None:
                context = self._context(solid, points, selection, fields, planar, collection_check)
                kappa = amplitude_prefactor(context.omega, incident.incident_power, collected.incident_power)
                if not self.photoelastic and context.interfaces is None:
                    raise ValueError(f'{label}: photoelastic is off and no optical interface touches the drive '
                                     'selection: nothing drives the solid (turn photoelastic on or select the '
                                     'interfaces)')
            elif not (float(fields.omega) == context.omega and np.array_equal(fields.x, context.x)
                      and np.array_equal(fields.y, context.y)):
                raise ValueError(f'{label}: the optics changed their grid or wavelength between scan positions (a bug)')
            grid, _ = _optical_arrays(fields, planar, label)
            del fields   # the solution of this position: only its window samples are kept below
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
            boundary = self._boundary_values(context.interfaces, grid, partner, same_field, parts, planar)
            squares = [0.0, 0.0]
            for index, stress in enumerate(stresses):
                column = columns[position * n_parts + index, :n_vector]
                if self.photoelastic:
                    drive = assemble_stress_drive(dofs, points, kappa * stress)
                    column += drive
                    squares[0] += float(np.vdot(drive, drive).real)
                if boundary is not None:
                    traction = (kappa * boundary[index])[..., None] * context.interfaces.points.normals
                    drive = assemble_edge_drive(dofs, context.interfaces.points, traction)
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
        messages = [('info', self.summary())]
        if torsion > TORSION_WARNING:
            messages.append(('warning', f'{label}: the dropped azimuthal order 0 torsion holds {torsion:.3g} of the '
                                        'drive (cross-polarised detection; not modelled: no Brillouin peak in an '
                                        'isotropic liquid, but a solid with shear peaks is under-counted)'))
        if self.moving_boundary and interfaces is None:
            messages.append(('info', f'{label}: no optical interface touches the drive selection (no moving-boundary '
                                     'part)'))
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
        if not isinstance(solver, DriveOptics):
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

    def _context(self, solid, points, selection, fields, planar, collection_check) -> _Context:
        """Sampling state of the optical grid of the first position: the volume sampler, eps at the points (PEC
        refused), the interfaces, the reference laws per point and the homogeneous raster cells."""
        label = self._label
        x, y = np.asarray(fields.x, dtype=np.float64), np.asarray(fields.y, dtype=np.float64)
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f'{label}: the optical window has {len(x)} x {len(y)} cells (at least 2 x 2)')
        omega = float(fields.omega)
        if collection_check is not None:
            collection_omega, collection_x, collection_y = collection_check
            if abs(collection_omega - omega) > _OMEGA_RTOL * omega:
                raise ValueError(f"{label}: collection_physics '{self.collection_physics}' solves omega "
                                 f'{collection_omega:.9g} rad/s, the optics {omega:.9g} rad/s: one wavelength')
            if not (np.array_equal(collection_x, x) and np.array_equal(collection_y, y)):
                raise ValueError(f"{label}: collection_physics '{self.collection_physics}' runs on another optical "
                                 'grid: the drive multiplies both fields cell by cell (put both on one mesh)')
        volume = _GridSamples(x, y, points.points, f'{label}: drive points', axis=not planar)
        _, eps = _optical_arrays(fields, planar, label)
        eps_points = {key: volume.cells(values) for key, values in eps.items()}
        if fields.pec is not None:
            inside = np.asarray(fields.pec, dtype=bool).reshape(-1)[volume.cell]
            if inside.any():
                first = points.points[int(np.argmax(inside))]
                raise ValueError(f'{label}: {int(inside.sum())} drive points lie in PEC cells of the optics (first at '
                                 f'({first[0]:.6g}, {first[1]:.6g}) m): keep the drive selection out of conductors')
        isotropic = sum(eps.values()) / len(eps)
        interfaces = self._interfaces(solid, selection, x, y, isotropic, planar) if self.moving_boundary else None
        point_laws, laws, raster = None, (), None
        if self.reference != 'none':
            element_laws, laws = self._reference_laws(solid, selection)
            point_laws = element_laws[points.elements]
            if self.reference == 'homogeneous':
                raster = self._raster_cells(solid, selection, x, y, eps, element_laws)
        return _Context(x=x, y=y, omega=omega, volume=volume, eps_points=eps_points, interfaces=interfaces,
                        point_laws=point_laws, laws=laws, raster=raster)

    def _interfaces(self, solid, selection, x, y, isotropic, planar):
        """_Interfaces of the moving boundary (module header), None when no interface touches the selection."""
        label = self._label
        dofs = solid.dofs
        mesh = dofs.mesh
        pairs = np.asarray(mesh.edge_elements, dtype=np.int64)
        interior = np.flatnonzero(pairs[:, 1] >= 0)
        first, second = pairs[interior, 0], pairs[interior, 1]
        window = np.asarray(solid.window, dtype=bool)
        members = np.asarray(solid.solid, dtype=bool)
        touching = (selection[first] | selection[second]) & window[first] & window[second]
        differ = members[first] != members[second]
        for values in solid.materials.values():
            a, b = values[first], values[second]
            differ |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
        edges = interior[touching & differ]
        if not edges.size:
            return None
        ends = np.asarray(mesh.points, dtype=np.float64)[np.asarray(mesh.edges)[edges]]   # (k, 2, 2)
        middle = ends.mean(axis=1)
        tangent = ends[:, 1] - ends[:, 0]
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1) / np.linalg.norm(tangent, axis=1)[:, None]
        shift = _side_shift(x, y, middle)[:, None]
        what = f'{label}: moving-boundary side samples'
        eps_0 = _GridSamples(x, y, middle - shift * normal, what).cells(isotropic)
        eps_1 = _GridSamples(x, y, middle + shift * normal, what).cells(isotropic)
        optical = np.abs(eps_0 - eps_1) > _EPS_RTOL * np.maximum(np.abs(eps_0), np.abs(eps_1))
        if not optical.any():
            return None
        edge_points = edge_drive_points(dofs, edges[optical], axisymmetric=solid.axisymmetric,
                                        azimuthal_order=solid.azimuthal_order)
        if not len(edge_points.edges):
            return None
        shape = tuple(edge_points.weights.shape)
        flat = edge_points.points.reshape(-1, 2)
        normals = edge_points.normals.reshape(-1, 2)
        shift = _side_shift(x, y, flat)[:, None]
        sides = tuple(_GridSamples(x, y, flat + sign * shift * normals, what, axis=not planar) for sign in (-1.0, 1.0))
        return _Interfaces(points=edge_points, sides=sides,
                           eps_sides=tuple(side.cells(isotropic).reshape(shape) for side in sides), shape=shape)

    def _boundary_values(self, interfaces, grid, partner, same_field, parts, planar):
        """[Johnson value f (k, q) per part] of one position, None without interfaces."""
        if interfaces is None:
            return None

        def sides(values):
            return tuple({name: array.reshape(interfaces.shape) for name, array in side.fields(values).items()}
                         for side in interfaces.sides)
        incident = sides(grid)
        collection = incident if same_field else sides(partner)
        normals = interfaces.points.normals
        if planar:
            return [moving_boundary_traction(normals, incident, collection, interfaces.eps_sides, PLANAR_COMPONENTS)]
        values = azimuthal_moving_boundary(normals, incident, collection, interfaces.eps_sides, self.detection)
        return [values[key] for key in parts]

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
            shear = modulus_function(group.shear)
            longitudinal = longitudinal_function(modulus_function(group.bulk), shear)
            elements = np.flatnonzero(group.elements)
            values, inverse = np.unique(density[elements], return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            for value_index, value in enumerate(values):
                index[elements[inverse == value_index]] = len(laws)
                laws.append((longitudinal, shear, float(value)))
        return index, tuple(laws)

    def _local_lines(self, context, points, grid, partner, stresses, kappa, planar) -> tuple:
        """(q_mean [1/m], ((law, q, weight), ...) compressed lines or ()) of one position: the local wavevector of P =
        E_c . E_i, sigma_L = q^T sigma~ q / q^2 per point and part, weights |kappa sigma_L|^2 dV summed over parts."""
        product = sum(partner[name] * values for name, values in grid.items())
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

    def _raster_cells(self, solid, selection, x, y, eps, element_laws) -> _Raster:
        """_Raster of the optical cells whose centres lie in the drive selection (one uniform isotropic material)."""
        label = self._label
        if any(mask[selection].any() for mask, _ in self.crystal_regions):
            raise ValueError(f"{label}: reference 'homogeneous' needs an isotropic medium (a crystal region lies in "
                             "the drive selection): use reference 'local'")
        centres = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1).reshape(-1, 2)
        elements = np.asarray(solid.locate(centres), dtype=np.int64)
        inside = elements >= 0
        inside[inside] = selection[elements[inside]]
        cells = np.flatnonzero(inside)
        if not cells.size:
            raise ValueError(f"{label}: reference 'homogeneous': no optical cell centre lies in the drive selection")
        cell_elements = elements[cells]
        cell_laws = np.unique(element_laws[cell_elements])
        if cell_laws.size != 1 or cell_laws[0] < 0:
            raise ValueError(f"{label}: reference 'homogeneous' needs one uniform material over the drive selection, "
                             f"got {cell_laws.size} (moduli, loss and density sets): use reference 'local'")
        i, j = np.unravel_index(cells, (len(x), len(y)))
        i0, j0 = int(i.min()), int(j.min())
        shape = (int(i.max()) - i0 + 1, int(j.max()) - j0 + 1)
        dx = _uniform_spacing(x[i0:i0 + shape[0]], f'{label} x')
        dy = _uniform_spacing(y[j0:j0 + shape[1]], f'{label} y')
        p44 = solid.materials.get('photoelastic_p44')
        return _Raster(cells=cells, i=i - i0, j=j - j0, shape=shape, dx=dx, dy=dy, law=int(cell_laws[0]),
                       p11=self._material(solid, 'photoelastic_p11')[cell_elements],
                       p12=self._material(solid, 'photoelastic_p12')[cell_elements],
                       p44=None if p44 is None else p44[cell_elements],
                       eps={key: values.reshape(-1)[cells] for key, values in eps.items()})

    def _raster_values(self, raster, grid, partner, same_field, kappa) -> np.ndarray:
        """kappa sigma~ (3, n_x, n_y) rows (xx, yy, xy) of one position on the raster box (0 outside the region)."""
        incident = {name: values.reshape(-1)[raster.cells] for name, values in grid.items()}
        collection = incident if same_field else {name: values.reshape(-1)[raster.cells]
                                                  for name, values in partner.items()}
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

    def reset_records(self) -> None:
        """Forget the per-frequency records (the bound columns stay)."""
        self._records = []

    def result(self) -> dict:
        """The 'xy' result (brillouin.brillouin_result) of the recorded frequencies (ascending), with the reference
        and the extras; ValueError before a frequency_response run or for a frequency solved twice."""
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

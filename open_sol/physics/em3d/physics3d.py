"""EMPhysics3D: the PhysicsSolver of em_fdtd_3d (engine3d on a Domain3D) with dipole / plane-wave / mode sources, point
probes, flux planes / boxes, DFT layers, mode monitors, far fields, PEC (conformal: conformal_pec3d), poles and the
harmonic FDTD hooks."""
import math
from collections.abc import Mapping

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, EPS0, MU0
from ...core.cut_cells import union_test
from ...core.mirror import MIRROR_SIGNS, plane_wave_mirror_issue
from ...core.solver_base import PhysicsSolver
from ..em import _make_temporal, _pole_label, _pole_step_term, _resolve_frequencies, _resolve_frequency
from ..em.materials import DrudePole, LorentzPole
from ..em.monitors import dft_sample_interval
from .boundaries3d import AXES, solver_boundary_config_3d
from .conformal_pec3d import STABILITY_BAND_CELLS, build_conformal_pec_3d
from .engine3d import COMPONENT_AXES, MAX_MATERIAL_IDS, FDTDEngine3D
from .harmonic3d import HarmonicAccumulator3D
from .mode_ports3d import AXIS_PERMITTIVITY, MODE_DIRECTIONS, ModeMonitor3D, ModeSource3D
from .monitors3d import DFTPlane3D, FluxBox3D, PlaneFlux3D, point_probe
from .near_to_far3d import NearToFar3D
from .renderer3d import EMRenderer3D
from .sources3d import DIPOLE_ORIENTATIONS, DipoleSource3D, nearest_face
from .subpixel_smoothing3d import SubpixelSmoothing3D
from .tfsf3d import PLANE_WAVE_DIRECTIONS, PlaneWaveSource3D

# MaterialGrid arrays the engine may read: eps_r, sigma_e, mu_r always (material_schema), eps_xx / eps_yy / sigma_m when
# the model allocated them (optional_material_schema)
MATERIAL_ARRAYS = ('eps_r', 'eps_xx', 'eps_yy', 'sigma_e', 'mu_r', 'sigma_m')
PERMITTIVITY_SLOT = 'eps_r_update'
PERMITTIVITY_SLOTS = {PERMITTIVITY_SLOT: 'eps_r', 'eps_xx_update': 'eps_xx', 'eps_yy_update': 'eps_yy'}
HALF_STORAGE_PEAK = 16.0   # stored units of the estimated peak field: ~4000x headroom below the fp16 max


class EM3DSlots:
    """permittivity_arrays and inject_source of the 3D EM physics (em_fdtd_3d, em_fdfd_3d on tensor grids and
    tetrahedra); type_label names the type in messages."""

    type_label = ''

    @property
    def permittivity_arrays(self) -> tuple:
        """Permittivity arrays the solver reads: eps_r (E_z, and E_x / E_y unless eps_xx / eps_yy are allocated)."""
        allocated = self.domain.materials.property_names()
        return tuple(array for array in PERMITTIVITY_SLOTS.values() if array in allocated)

    def inject_source(self, name, array):
        """Copy a coupling's array into the permittivity array of slot `name` (eps_r_update, eps_xx_update /
        eps_yy_update when allocated), then materials_changed."""
        array_name = PERMITTIVITY_SLOTS.get(name)
        if array_name not in self.permittivity_arrays:
            valid = [slot for slot, array in PERMITTIVITY_SLOTS.items() if array in self.permittivity_arrays]
            raise ValueError(f"{self.type_label} accepts the source slots of its allocated permittivity arrays "
                             f"({', '.join(valid)}), got {name!r}")
        target = getattr(self.domain.materials, array_name)
        xp = self.domain.xp
        xp.copyto(target, xp.asarray(array, dtype=target.dtype))
        self.materials_changed((array_name,))


class StaticEM3DContract(EM3DSlots):
    """The quasi-static contract of the eigen-solved 3D EM physics (em_fdfd_3d: tensor grids and tetrahedra): no time
    stepping, state = the permittivity arrays, sigma_e and 'q_accum': None."""

    def prepare(self, domain):
        return

    def step(self, t=None, dt=None):
        """Nothing: the eigenfrequency study drives this physics (solve_eigenfrequencies)."""
        return

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """The permittivity arrays, sigma_e and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        state = {name: getattr(materials, name) for name in self.permittivity_arrays}
        state['sigma_e'] = materials.sigma_e
        state['q_accum'] = None
        return state


def complex_cell_materials(materials, omega, pole_regions) -> tuple:
    """([eps_x, eps_y, eps_z] host complex128 = background - j sigma_e / (omega eps0) + chi of each pole on its region,
    the host float64 backgrounds (AXIS_PERMITTIVITY: eps_r where eps_xx / eps_yy are not allocated), host complex128 mu
    - j sigma_m / (omega mu0)) of em_fdfd_3d cells or elements; pole_regions = [(pole, index of its region (window
    slices or Ellipsis), its mask)]."""
    allocated = materials.property_names()

    def host(name):
        return np.asarray(to_host(getattr(materials, name)), dtype=np.float64)

    common = -1j * host('sigma_e') / (omega * EPS0)   # common to the three components
    for pole, index, mask in pole_regions:
        common[index] += complex(pole.susceptibility(omega)) * np.asarray(to_host(mask), dtype=np.float64)
    backgrounds = [host(name if name in allocated else 'eps_r') for name in AXIS_PERMITTIVITY]
    mu = host('mu_r').astype(np.complex128)
    if 'sigma_m' in allocated:
        mu -= 1j * host('sigma_m') / (omega * MU0)
    return [background + common for background in backgrounds], backgrounds, mu


def check_pole(pole):
    """TypeError unless pole is a DrudePole or LorentzPole."""
    if not isinstance(pole, (DrudePole, LorentzPole)):
        raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')


class _PhysicalFieldState(Mapping):
    """state() of a half_storage engine: the six components as float32 copies in V/m / A/m, made on access only."""

    def __init__(self, state, engine):
        self._state = state
        self._engine = engine

    def __getitem__(self, name):
        value = self._state[name]
        if name not in COMPONENT_AXES:
            return value
        return value.astype(DTYPE) * DTYPE(self._engine.field_scale(name))

    def __iter__(self):
        return iter(self._state)

    def __len__(self):
        return len(self._state)


class EMPhysics3D(EM3DSlots, PhysicsSolver):
    """3D FDTD on a Domain3D: engine, sources, monitors; materials_changed marks the material index dirty (rebuilt
    with the source calibrations before the next step). EM loss protocol: take_absorbed_power / state()['q_accum']."""

    name = 'em'
    type_label = 'em_fdtd_3d'
    # cylindrical (r, phi, z) sectors (symmetry 2.21, scripts only: the em_fdtd_3d PhysicsType stays Cartesian until
    # the model path exists; engine3d header): dipoles, probes, DFT layers, PEC, poles, losses and harmonic FDTD; the
    # items whose placement or weights read Cartesian widths refuse a sector (_require_cartesian)
    coordinate_grids = ('cartesian', 'polar')

    def __init__(self, domain, precision='single'):
        if getattr(domain, 'dimensions', 2) != 3:
            raise ValueError('EMPhysics3D needs a Domain3D (a model with domain size_z)')
        self.domain = domain
        self.boundary_config = solver_boundary_config_3d(domain)
        self.engine = FDTDEngine3D(domain, self.boundary_config, precision)
        self.fields = self.engine.fields
        self.sources = []
        self.monitors = []
        self._flux_boxes = {}          # (edge bounds, frequencies) -> FluxBox3D shared by flux boxes and far fields
        self.dispersive_regions = []   # [(pole, window slices, cropped bool mask)]
        self.finalize_warnings = []
        self.runtime_warnings = []
        self.harmonic = None
        self._materials_dirty = True
        self._quantised_arrays = set()   # arrays whose re-quantisation was already reported
        self._reported_warnings = set()  # source / monitor calibration warnings already reported
        self.port_mode_solves = None     # mode_ports3d solves shared by the ports of one _refresh_materials round
        # Conformal PEC (solver_defaults.conformal_boundaries): the model's 3D geometry locator, the regions of
        # add_conformal_pec_region [(window, cropped mask, inside(x, y, z))], their ConformalPEC3D once finalized
        self.conformal_locator = None
        self.conformal_pec = None
        self._conformal_regions = []

    # ---- PhysicsSolver contract (+ EM3DSlots) ----
    def prepare(self, domain):
        pass

    def step(self, t=None, dt=None):
        if self._materials_dirty:   # a recalibration's new source / monitor notes too (e.g. a lossy TFSF background)
            self.runtime_warnings.extend(self._refresh_materials() + self._item_warnings(only_new=True))
        self.engine.advance(self.sources, self.monitors)

    def dt_limit(self) -> float:
        return float(self.engine.dt)

    def state(self):
        fields = self.fields
        self.engine.deposit_pole_work()
        state = {'ex': fields.ex, 'ey': fields.ey, 'ez': fields.ez, 'hx': fields.hx, 'hy': fields.hy, 'hz': fields.hz,
                 'eps_r': self.domain.materials.eps_r, 'q_accum': self.engine.q_accum}
        return state if self.engine.precision == 'single' else _PhysicalFieldState(state, self.engine)

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: mean absorbed power density [W/m^3] since the last call (None for an empty window);
        merge_previous also averages over the last completed window (end-of-run flush)."""
        return self.engine.take_absorbed_power(merge_previous)

    def finalize(self):
        """Material index, stability check (ValueError naming the stable courant_factor), source calibration, DFT
        sample interval and the half_storage field scale (before the first step only)."""
        if self.engine.n_step == 0:
            self._set_field_scale()
        # plus the plane-wave outline / port mode notes of this calibration
        self.finalize_warnings = self._refresh_materials() + self._item_warnings(only_new=False)
        if self._conformal_regions and self.conformal_pec is None:
            self._apply_conformal_pec()
        conformal_note = self._conformal_pec_note()
        if conformal_note is not None:
            self.finalize_warnings.append(conformal_note)
        pole_terms = {}   # id(region) -> (region, summed leap-frog pole term of its poles)
        for pole, _, mask in self.dispersive_regions:
            region = self.engine.region_by_mask[id(mask)]
            previous = pole_terms.get(id(region), (region, 0.0))[1]
            pole_terms[id(region)] = (region, previous + _pole_step_term(pole, self.engine.dt))
        issue = self.engine.stability_issue(list(pole_terms.values()))
        if issue is not None:
            labels = ', '.join(_pole_label(pole) for pole, _, _ in self.dispersive_regions)
            raise ValueError(issue + (f' (poles: {labels})' if labels else ''))
        if self.engine.n_step == 0:
            spectral = [monitor for monitor in self.monitors if hasattr(monitor, 'update')]
            interval = dft_sample_interval(self.engine.dt, self.sources, [monitor.frequencies for monitor in spectral])
            for monitor in spectral:
                monitor.sample_interval = interval

    def _set_field_scale(self):
        """half_storage: a power of two placing the sources' largest field near HALF_STORAGE_PEAK stored units (fp16
        max 65504, normal down to 6e-5), then every monitor reads back in physical units."""
        engine = self.engine
        if engine.precision == 'single':
            return
        peaks = [source.peak_field() for source in self.sources if hasattr(source, 'peak_field')]
        peak = max((value for value in peaks if value > 0.0), default=HALF_STORAGE_PEAK)
        engine.set_field_scale(2.0 ** round(math.log2(peak / HALF_STORAGE_PEAK)))
        for monitor in self.monitors:
            if hasattr(monitor, 'set_field_scales'):
                monitor.set_field_scales(engine.e_scale, engine.h_scale)
            elif hasattr(monitor, 'value_scale'):   # FieldProbe
                monitor.value_scale = engine.field_scale(monitor.field_component)

    def _require_single(self, what):
        if self.engine.precision != 'single':
            raise ValueError(f'{what} needs single precision fields (stepper precision {self.engine.precision}): '
                             'its kernels read and write float32 fields in physical units')

    def _require_cartesian(self, what):
        if self.engine.polar:
            raise NotImplementedError(f'em_fdtd_3d: {what} is not available on a cylindrical (r, phi, z) sector yet '
                                      '(its placement or weights use Cartesian cell widths); sectors take dipoles, '
                                      'probes, DFT layers, PEC, dispersive / lossy media and harmonic FDTD')

    def materials_changed(self, names):
        if not set(MATERIAL_ARRAYS).isdisjoint(names):
            self._materials_dirty = True

    def renderer(self):
        return EMRenderer3D(self)

    def set_subpixel_smoothing(self, geometry):
        """Average the current eps on the 3D interface geometry (igor 1.5, model.subpixel_geometry) at every material
        rebuild: face overrides the E coefficients, pole / dipole cb and mode ports read; None switches it off."""
        if geometry is not None:
            self._require_cartesian('subpixel smoothing')
        self.engine.subpixel_smoothing = (None if geometry is None
                                          else SubpixelSmoothing3D(geometry, self.engine.shape))
        self._materials_dirty = True

    def _refresh_materials(self):
        """Rebuild the material index and recalibrate the sources; returns a note per array newly re-quantised."""
        engine = self.engine
        engine.rebuild_materials()
        self.port_mode_solves = {}   # ports on equal layers share a mode solve per omega (this round only)
        try:
            for source in self.sources:
                source.calibrate(engine)
            for monitor in self.monitors:   # mode monitors solve their modes until the first step
                if hasattr(monitor, 'calibrate'):
                    monitor.calibrate(engine)
        finally:
            self.port_mode_solves = None
        self._materials_dirty = False
        notes = []
        for name, error in engine.quantisation_errors.items():
            if name not in self._quantised_arrays:
                self._quantised_arrays.add(name)
                notes.append(f'{name} has more distinct values than the uint16 material index holds: re-quantised to '
                             f'at most {MAX_MATERIAL_IDS - 1} levels (max deviation {error:.3g}, refreshed at every '
                             'material change)')
        return notes

    def _item_warnings(self, only_new):
        """Warnings of the sources' and monitors' last calibration (only_new: those not reported before)."""
        warnings = []
        for item in self.sources + self.monitors:
            for warning in getattr(item, 'warnings', ()):
                if not (only_new and warning in self._reported_warnings):
                    warnings.append(warning)
                self._reported_warnings.add(warning)
        return warnings

    # ---- frequency_domain study (harmonic FDTD) ----
    def attach_harmonic(self, frequencies):
        self._require_single('harmonic FDTD (frequency_domain study)')
        self.harmonic = HarmonicAccumulator3D(self, frequencies)
        self.monitors.append(self.harmonic)
        return self.harmonic

    def select_harmonic_frequency(self, index):
        self.harmonic.select(index)

    # ---- materials ----
    def add_dispersive(self, pole, window, mask):
        """Drude / Lorentz pole on the cropped bool mask of window (the permittivity arrays are its background)."""
        self.engine.add_pole(pole, window, mask)
        self.dispersive_regions.append((pole, window, mask))

    def add_pec(self, window, mask):
        """PEC cells on a cropped bool mask (faces between two PEC cells are pinned)."""
        self.engine.add_pec(window, mask)
        self._materials_dirty = True

    def set_conformal_boundaries(self, locator):
        """solver_defaults.conformal_boundaries: pec_domain features call add_conformal_pec_region with
        locator.inside([object]) (model.boundary_locator.GeometryLocator in 3D, duck-typed); None = staircased PEC."""
        self.conformal_locator = locator

    def add_conformal_pec_region(self, window, mask, inside):
        """PEC cells on a cropped bool mask whose drawn outline is inside(x, y, z) -> bool (host): the staircase cells
        at once (sources, ports, poles read them), the Dey-Mittra H faces of conformal_pec3d at finalize."""
        self._require_cartesian('conformal PEC')
        self.add_pec(window, mask)
        if 0 not in mask.shape:
            self._conformal_regions.append((window, mask, inside))

    def _apply_conformal_pec(self):
        """ConformalPEC3D of the conformal regions on their bounding box grown by the stability check's band (host: one
        transfer of the box's material ids and masks), handed to the engine."""
        engine = self.engine
        windows = [window for window, _, _ in self._conformal_regions]
        margin = STABILITY_BAND_CELLS + 3
        box = tuple(slice(max(min(window[axis].start for window in windows) - margin, 0),
                          min(max(window[axis].stop for window in windows) + margin, count))
                    for axis, count in enumerate(engine.shape))
        origin = tuple(cells.start for cells in box)
        pec = np.asarray(to_host(engine.material_id[box] == engine.pec_id), dtype=bool)
        conformal = np.zeros(pec.shape, dtype=bool)
        for window, mask, _ in self._conformal_regions:
            local = tuple(slice(cells.start - start, cells.stop - start) for cells, start in zip(window, origin))
            conformal[local] |= np.asarray(to_host(mask), dtype=bool)
        centres = tuple(np.asarray(axis_centres, dtype=np.float64) for axis_centres in self.domain.cell_centers())
        self.conformal_pec = build_conformal_pec_3d(
            engine.widths, centres, self.domain.physical_window(), origin, pec, conformal,
            union_test([inside for _, _, inside in self._conformal_regions]), engine.dt)
        engine.set_conformal_faces(self.conformal_pec.face_cell, self.conformal_pec.face_axis,
                                   self.conformal_pec.face_extra)

    def _conformal_pec_note(self):
        """Warning (the ConformalPEC3D summary) when the stability floor limited most cut faces or row_scale fell to 0,
        i.e. near staircase accuracy; None otherwise or without conformal PEC."""
        conformal = self.conformal_pec
        if conformal is None or conformal.face_cell.size == 0:
            return None
        if conformal.row_scale > 0.0 and 2 * int(np.count_nonzero(conformal.floored)) <= conformal.face_cell.size:
            return None
        return (f'{conformal.summary()}: the stability floor keeps most cut faces near staircase accuracy; a lower '
                'courant_factor leaves more of the conformal correction')

    # ---- sources ----
    def add_dipole(self, x, y, z, orientation='z', dipole_moment=1e-12, wavelength=None, frequency=None,
                   bandwidth=0.5, waveform='gaussian'):
        """Electric point dipole of moment I l [A m] along x / y / z on the E face nearest (x, y, z): the face along
        the dipole axis nearest the coordinate, the containing cell on the other two axes."""
        if orientation not in DIPOLE_ORIENTATIONS:
            raise ValueError(f"dipole orientation must be one of {DIPOLE_ORIENTATIONS}, got {orientation!r}")
        domain = self.domain
        domain.validate_physical_pos(x, y, z, 'Dipole')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        axis = AXES.index(orientation)
        cell = list(domain.pos_to_cell(x, y, z))
        cell[axis] = nearest_face(domain.cell_edges()[axis], (x, y, z)[axis], domain.physical_window()[axis])
        if self.engine.pec_wall_cell(cell, tuple(other for other in range(3) if other != axis)):
            raise ValueError(f'dipole at ({x:.4g}, {y:.4g}, {z:.4g}) m lies in the outermost cell layer of a pec face '
                             f'parallel to it, where its tangential E is pinned to 0; move it inside')
        source = DipoleSource3D('e' + orientation, cell, temporal, dipole_moment,
                                self.engine.face_dual_volume(axis, cell))
        self.sources.append(source)
        self._materials_dirty = True
        return source

    # ---- monitors ----
    def add_field_probe(self, x, y, z, component='ez'):
        """Raw Yee samples of one component at the cell containing (x, y, z)."""
        if component not in COMPONENT_AXES:
            raise ValueError(f'probe component must be one of {list(COMPONENT_AXES)}, got {component!r}')
        self.domain.validate_physical_pos(x, y, z, 'Field probe')
        probe = point_probe(self.domain.pos_to_cell(x, y, z), component)
        self.monitors.append(probe)
        return probe

    def add_flux_plane(self, normal, position, wavelengths=None, frequencies=None):
        """Flux through the cell-edge plane nearest `position` [m] along `normal` ('x' / 'y' / 'z') across the whole
        physical window; > 0 toward +normal."""
        self._require_cartesian('a flux plane')
        axis = AXES.index(normal)
        bounds = list(self._window_edges())
        bounds[axis] = (self._nearest_edge(axis, position),) * 2
        monitor = PlaneFlux3D(axis, bounds[axis][0], bounds, _resolve_frequencies(wavelengths, frequencies),
                              self.engine.dt, self.engine.widths, self.engine.shape, self.engine.periodic,
                              xp=self.engine.xp)
        self.monitors.append(monitor)
        return monitor

    def add_flux_box(self, lo, hi, wavelengths=None, frequencies=None):
        """Net outward flux through the box between the cell edges nearest the corners lo, hi [m]."""
        self._require_cartesian('a flux box')
        return self._shared_flux_box(self._box_edges(lo, hi, 'flux box'), _resolve_frequencies(wavelengths,
                                                                                               frequencies))

    def _shared_flux_box(self, edge_bounds, frequencies):
        """The stepped FluxBox3D of these edge bounds and frequencies, created once: flux boxes and far fields on one
        surface share its DFT accumulation."""
        key = (tuple(tuple(int(edge) for edge in bounds) for bounds in edge_bounds),
               tuple(np.atleast_1d(np.asarray(frequencies, dtype=np.float64)).tolist()))
        box = self._flux_boxes.get(key)
        if box is None:
            box = FluxBox3D(edge_bounds, frequencies, self.engine.dt, self.engine.widths, self.engine.shape,
                            self.engine.periodic, xp=self.engine.xp)
            self._flux_boxes[key] = box
            self.monitors.append(box)
        return box

    def _box_edges(self, lo, hi, label):
        """Per axis (first, last) window cell edges nearest the corners lo, hi [m]; ValueError for an empty span."""
        bounds = []
        for axis in range(3):
            first, last = self._nearest_edge(axis, min(lo[axis], hi[axis])), self._nearest_edge(axis, max(lo[axis],
                                                                                                         hi[axis]))
            if last <= first:
                raise ValueError(f'{label} spans no cell along {AXES[axis]}: corners {lo[axis]:.4g} and '
                                 f'{hi[axis]:.4g} m snap to the same cell edge')
            bounds.append((first, last))
        return bounds

    def add_dft_plane(self, z, wavelengths=None, frequencies=None):
        """DFT of E (cell centres) over the window layer containing height z [m]."""
        domain = self.domain
        domain.validate_physical_pos(0.0, 0.0, z, 'DFT plane')
        monitor = DFTPlane3D(domain.physical_window(), domain.pos_to_cell_z(z),
                             _resolve_frequencies(wavelengths, frequencies), self.engine.dt, self.engine.periodic,
                             self.engine.nz, xp=self.engine.xp, wrap_angles=self.engine.bloch_angles)
        self.monitors.append(monitor)
        return monitor

    def _window_edges(self):
        """Per axis (first, last) cell-edge index of the physical window."""
        return tuple((cells.start, cells.stop) for cells in self.domain.physical_window())

    def _nearest_edge(self, axis, coordinate):
        """Grid edge index nearest `coordinate` [m] among the physical window's edges."""
        first, last = self._window_edges()[axis]
        edges = self.domain.cell_edges()[axis]
        return first + int(np.argmin(np.abs(edges[first:last + 1] - float(coordinate))))

    # ---- plane wave, mode ports, far field (tfsf3d, mode_ports3d, near_to_far3d) ----
    def add_plane_wave(self, lo, hi, direction='+z', polarization='x', amplitude=1.0, wavelength=None,
                       frequency=None, bandwidth=0.5, waveform='gaussian'):
        """TFSF plane wave (incident E along `polarization`, peak `amplitude` V/m) along `direction` on the total-field
        box between the window cell edges nearest lo, hi [m]; a periodic axis spans its period, and with both
        transverse axes periodic a downstream face on the window edge opens the box into the CPML (one-sided)."""
        self._require_single('a plane wave source')
        self._require_cartesian('a plane wave source')
        if self.boundary_config.has_bloch_phase:
            raise ValueError('a TFSF plane wave needs Bloch k = 0: its incident field is a normal-incidence wave, '
                             'which a Bloch lattice with k != 0 does not hold (no oblique 3D plane wave yet)')
        if direction not in PLANE_WAVE_DIRECTIONS:
            raise ValueError(f'plane wave direction must be one of {PLANE_WAVE_DIRECTIONS}, got {direction!r}')
        if polarization not in AXES or polarization == direction[1]:
            raise ValueError(f'plane wave polarization must be an axis other than the propagation axis '
                             f'{direction[1]}, got {polarization!r}')
        axis_d, sign = AXES.index(direction[1]), (1 if direction[0] == '+' else -1)
        periodic = self.engine.periodic
        if periodic[axis_d]:
            raise ValueError(f'a plane wave cannot travel along the periodic axis {direction[1]}')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        one_sided = all(periodic[axis] for axis in range(3) if axis != axis_d)
        mirrors = self.boundary_config.ghost_faces
        bounds = []
        for axis, (start, stop) in enumerate(self._window_edges()):
            if periodic[axis]:
                bounds.append((None, None))
                continue
            first = self._nearest_edge(axis, min(lo[axis], hi[axis]))
            last = self._nearest_edge(axis, max(lo[axis], hi[axis]))
            if last <= first:
                raise ValueError(f'plane wave box spans no cell along {AXES[axis]}')
            if axis == axis_d and one_sided:
                first = None if sign < 0 and first == start else first
                last = None if sign > 0 and last == stop else last
            # a face on a mirror plane of the wave's parity opens (None: the box runs through the ghost, its image
            # completes it); core.mirror.plane_wave_mirror_issue refuses the other parity and the propagation axis
            for side, edge, window_edge in (('lo', first, start), ('hi', last, stop)):
                face = f'{AXES[axis]}_{side}'
                if edge is not None and edge == window_edge and face in mirrors:
                    problem = plane_wave_mirror_issue(face, MIRROR_SIGNS[getattr(self.boundary_config, face)],
                                                      AXES[axis_d], polarization)
                    if problem is not None:
                        raise ValueError(f'plane wave box {problem}')
                    first, last = (None, last) if side == 'lo' else (first, None)
            if (first is not None and first < start + 1) or (last is not None and last > stop - 1):
                raise ValueError(f'plane wave box face along {AXES[axis]} lies on the domain edge: keep one '
                                 f'scattered-field cell outside every box face (a downstream face may reach the edge '
                                 f'only with both transverse axes periodic, a transverse face a mirror plane of the '
                                 f"wave's parity)")
            bounds.append((first, last))
        source = PlaneWaveSource3D(self.engine, bounds, axis_d, sign, AXES.index(polarization), temporal, amplitude)
        self.sources.append(source)
        self._materials_dirty = True
        return source

    def add_mode_source(self, normal, position, direction='positive', mode_number=1, power=1.0, wavelength=None,
                        frequency=None, bandwidth=0.3, waveform='gaussian'):
        """Guided mode of `power` W launched along +-normal from the cell layer containing `position` [m] (total field
        from it downstream), solved on the whole grid cross-section of that layer at the carrier wavelength."""
        self._require_single('a mode source')
        self._require_cartesian('a mode source')
        if normal not in AXES:
            raise ValueError(f'mode source normal must be one of {AXES}, got {normal!r}')
        if direction not in MODE_DIRECTIONS:
            raise ValueError(f'mode source direction must be one of {MODE_DIRECTIONS}, got {direction!r}')
        if waveform not in ('gaussian', 'cw'):
            raise ValueError(f"mode source waveform must be 'gaussian' or 'cw', got {waveform!r}")
        axis = AXES.index(normal)
        if self.engine.periodic[axis]:
            raise ValueError(f'a mode source cannot launch along the periodic axis {normal}')
        start, stop = self._window_edges()[axis]
        cell = getattr(self.domain, f'pos_to_cell_{normal}')(position)
        if (direction == 'positive' and cell < start + 1) or (direction == 'negative' and cell > stop - 2):
            raise ValueError(f'mode source at {normal} = {position:.4g} m: its interface face lies on the window edge '
                             f'(inside a CPML layer or on the wall); move it at least one cell inside')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        source = ModeSource3D(self, axis, cell, direction, mode_number, temporal, power,
                              f'mode source ({normal} = {position:.4g} m)')
        self.sources.append(source)
        self._materials_dirty = True
        return source

    def add_mode_monitor(self, normal, position, mode_number=1, wavelengths=None, frequencies=None):
        """Forward / backward power of mode `mode_number` through the cell-edge plane nearest `position` [m] along
        `normal` across the physical window (modes solved per frequency at finalize)."""
        self._require_single('a mode monitor')
        self._require_cartesian('a mode monitor')
        if normal not in AXES:
            raise ValueError(f'mode monitor normal must be one of {AXES}, got {normal!r}')
        axis = AXES.index(normal)
        edge = self._nearest_edge(axis, position)
        bounds = list(self._window_edges())
        bounds[axis] = (edge, edge)
        monitor = ModeMonitor3D(self, axis, edge, bounds, _resolve_frequencies(wavelengths, frequencies), mode_number,
                                f'mode monitor ({normal} = {position:.4g} m)')
        self.monitors.append(monitor)
        self._materials_dirty = True
        return monitor

    def add_near_to_far(self, lo, hi, wavelengths=None, frequencies=None):
        """Near-to-far transform over the closed box between the window cell edges nearest lo, hi [m] (vacuum
        outside it), unfolded over the mirror faces (their images; ValueError for mirrors on both faces of an axis or
        a periodic axis). Not in self.monitors: its FluxBox3D (shared with a flux box of the same surface and
        frequencies) is."""
        self._require_cartesian('a far-field box')
        periodic = [axis for axis in AXES if self.boundary_config.periodic(axis)]
        if periodic:
            raise ValueError(f"a far-field box needs free space around it, but axis {', '.join(periodic)} is periodic: "
                             'the box currents would include the neighbour cells (no array factor, no Floquet orders)')
        domain = self.domain
        window = self._window_edges()
        mirrors = [(AXES.index(face[0]), window[AXES.index(face[0])][0 if face.endswith('lo') else 1],
                    1 if getattr(self.boundary_config, face) == 'symmetric' else -1)
                   for face in self.boundary_config.ghost_faces]
        box = self._shared_flux_box(self._box_edges(lo, hi, 'far-field box'),
                                    _resolve_frequencies(wavelengths, frequencies))
        return NearToFar3D(box, domain.cell_edges(), domain.cell_centers(), mirrors=mirrors)


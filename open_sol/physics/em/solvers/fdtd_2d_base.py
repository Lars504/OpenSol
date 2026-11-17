"""_YeeEngine2DBase: shared part of the 2D FDTD Yee engines (curl spacings, field sets, solver contract, J.E window)
plus PEC-mask and dispersive-region helpers."""
import math
import warnings

import numpy as np
from ....core.backend import cp, to_host

from ....core.constants import DTYPE, MU0
from ....core.monitors import FieldProbe, FieldProbeRows
from ....core.solver_base import PhysicsSolver
from ....core.sources import PointSource
from ..materials import register_em_arrays
from ..placement import PERMITTIVITY_SLOTS
from ..sources import ElectricDipoleSource, MagneticLineCurrentSource
from ..yee_grid import grid_cell_widths, yee_inverse_spacings
from . import fdtd_ez_2d_kernels as _kernels

# gpu: the fixed-argument launches of a step (H: Yee + conformal + H CPML + fused H rows of every field set; E: Yee +
# E CPML + fused E rows per set, with the pole updates / corrections when nothing eager sits between) replay as CUDA
# graphs (as em3d engine3d), the FieldProbes' record (core.monitors.FieldProbeRows) at the end of the real set's E graph;
# the wall re-pins after the sources (skipped when _walls_need_repin finds them no-ops), accumulating / media pole
# paths, sources, the probe re-sample after them and the other monitors stay eager (and the row copies of a grid too small for the fused rows: a column-row copy
# overlaps its source in cupy's bounds check, which copies the input: an allocation inside a capture)
STEP_GRAPHS = True
_MAX_STEP_GRAPHS = 8   # graph keys kept (stale ones belong to replaced conformal tables)


def _bounding_box(weight):
    """(slice, slice) bounding the entries where `weight` > 0, None when there are none (host sync: build time only)."""
    positive = weight > 0
    rows = np.flatnonzero(to_host(positive.any(axis=1)))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(to_host(positive.any(axis=0)))
    return (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1))


def _check_cell_array(array, shape, dtype, what, xp=cp):
    """TypeError unless `array` is an `xp` (cupy default, numpy on the cpu backend) `dtype` array of exactly the cell
    grid `shape` (no broadcast, no cast)."""
    if not isinstance(array, xp.ndarray) or array.dtype != dtype or array.shape != shape:
        raise TypeError(f"{what} must be a {getattr(xp, '__name__', 'cupy')} {np.dtype(dtype).name} array of shape "
                        f"{shape}, got {type(array).__name__} {getattr(array, 'dtype', '')} "
                        f"{getattr(array, 'shape', '')}")


def record_dispersive_region(regions, pole, mask):
    """Append (pole, bounding box, mask cropped to it) to `regions`; mask None = everywhere, an empty mask adds nothing
    (one host sync)."""
    if mask is None:
        regions.append((pole, None, None))
        return
    box = _bounding_box(mask)
    if box is not None:
        regions.append((pole, box, mask[box].copy()))


def add_pec_cells(pec_cells, mask):
    """Union PEC cell mask: a copy of `mask` when `pec_cells` is None, else `pec_cells` |= mask in place."""
    if pec_cells is None:
        return mask.copy()
    pec_cells |= mask
    return pec_cells


def _union_box(box_a, box_b):
    """Smallest (slice, slice) covering both boxes (either may be None)."""
    if box_a is None or box_b is None:
        return box_a if box_b is None else box_b
    return tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(box_a, box_b))


def _point_source_cells(source):
    """(i, j) cells / faces a point-like source writes (exactly PointSource, MagneticLineCurrentSource or
    ElectricDipoleSource), None for any other source (lines, TFSF boxes, mode ports, Bloch lines, subclasses)."""
    source_type = type(source)
    if source_type is ElectricDipoleSource:
        return [(i, j) for _, i, j, _ in source.faces]
    if source_type is PointSource or source_type is MagneticLineCurrentSource:
        return [(source.ix, source.iy)]
    return None


def _curl_spacings(cell_dx, cell_dy, cpml, xp, dtype=DTYPE):
    """Run-backend (inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h) of yee_inverse_spacings in the field `dtype`, CPML kappa
    folded in (cpml None: bulk only); *_e at cell centres, *_h at hi faces (last entry = periodic wrap)."""
    spacings = yee_inverse_spacings(cell_dx, cell_dy)
    curl_arrays = []
    for name in ('inv_dx_e', 'inv_dx_h', 'inv_dy_e', 'inv_dy_h'):
        bulk = getattr(spacings, name).astype(dtype)
        curl_arrays.append(xp.asarray(bulk, dtype=dtype) if cpml is None else getattr(cpml, f'build_{name}')(bulk))
    return tuple(curl_arrays)


class _FieldSet:
    """One real field set the Yee kernels step: index (0 real part, 1 imaginary part), its fields, its BoundaryManager
    (own CPML psi), the partner set's fields read at Bloch wraps and the signed wrap sines (+sin(k L) real set,
    -sin(k L) imaginary set; 0 and partner = itself in a real-only engine), in the field dtype."""

    __slots__ = ('index', 'fields', 'boundary_mgr', 'partner_fields', 'wrap_sin_x', 'wrap_sin_y')

    def __init__(self, index, fields, boundary_mgr, partner_fields, wrap_sin_x, wrap_sin_y):
        self.index = index
        self.fields = fields
        self.boundary_mgr = boundary_mgr
        self.partner_fields = partner_fields
        real = fields.dtype.type
        self.wrap_sin_x = real(wrap_sin_x)
        self.wrap_sin_y = real(wrap_sin_y)


class _YeeEngine2DBase(PhysicsSolver):
    """Shared part of the 2D FDTD engines: curl spacings, launch scalars, field sets (Bloch), da / db, solver contract,
    J.E window. Subclasses set polarization, the coefficient material sets, _rebuild_e_coefficients and
    _accumulate_e_new_half. Field dtype (stepper precision) = the fields' (float32 single, float64 double): every
    field-side array (coefficients, spacings, CPML, poles) and kernel scalar is in it; materials, PEC keep masks and
    q_accum stay float32."""

    name = "em"
    polarization = None
    # MaterialGrid arrays the E / H update coefficients depend on (materials_changed).
    _E_COEFFICIENT_MATERIALS = frozenset()
    _H_COEFFICIENT_MATERIALS = frozenset({'mu_r', 'sigma_m'})
    # field components the E half of a step writes (the eager work after the E graph may change them: probe re-sample)
    _E_COMPONENTS = frozenset()

    def __init__(self, grid, fields, materials, boundary_manager):
        # CPML psi signs are polarisation-specific: a manager of the other polarisation would amplify in the PML
        if boundary_manager.polarization != self.polarization:
            raise ValueError(f'{type(self).__name__} needs a BoundaryManager built with polarization='
                             f'{self.polarization!r}, got {boundary_manager.polarization!r}')
        # Run backend = the MaterialGrid's (core sub file "Backends"): fields and CPML must live on it too; every kernel
        # is a core.kernels twin, so the same calls run cupy or numpy.
        self.xp = materials.backend.xp
        if fields.xp is not self.xp or boundary_manager.xp is not self.xp:
            raise ValueError(f'{type(self).__name__}: fields ({fields.xp.__name__}), BoundaryManager '
                             f'({boundary_manager.xp.__name__}) and materials ({self.xp.__name__}) must share one '
                             f'backend')
        # Stepper precision = the field dtype; the CPML psi recursion must run in it too.
        self.dtype = fields.dtype
        self._real = self.dtype.type
        if boundary_manager.dtype != self.dtype:
            raise ValueError(f'{type(self).__name__}: fields ({self.dtype.name}) and BoundaryManager '
                             f'({boundary_manager.dtype.name}) need one dtype')
        self.grid = grid
        self.fields = fields
        self.materials = materials
        self.boundary_mgr = boundary_manager
        # Arrays the model build already registered are kept (hz: new eps_xx / eps_yy start from eps_r when it exists).
        register_em_arrays(materials, self.polarization)

        self.nx = grid.nx
        self.ny = grid.ny
        self.dt = grid.dt
        # kernel scalars in the field dtype (a float32 dt in a float64 kernel would round the time step)
        self._dt_real = self._real(self.dt)
        self._inv_dt_real = self._real(1.0 / float(self.dt))

        # Host float64 Yee cell widths (yee_grid); non-uniform spacing and CPML kappa live entirely in the 1D curl
        # arrays, so the same kernels serve every grid.
        self._cell_dx, self._cell_dy = grid_cell_widths(grid)
        (self._inv_dx_e_1d, self._inv_dx_h_1d,
         self._inv_dy_e_1d, self._inv_dy_h_1d) = _curl_spacings(self._cell_dx, self._cell_dy, boundary_manager.cpml,
                                                                 self.xp, self.dtype)

        # Fused-kernel launch config; RawKernel scalars are numpy scalars of the declared C type. Periodic / k = 0
        # Bloch axes wrap inside the Yee kernels (period nx / ny cells).
        self._k_grid, self._k_block = _kernels.launch(self.nx, self.ny)
        self._nx_i32 = np.int32(self.nx)
        self._ny_i32 = np.int32(self.ny)
        self._periodic_x = bool(boundary_manager.config.periodic_x)
        self._periodic_y = bool(boundary_manager.config.periodic_y)
        self._periodic_x_i32 = np.int32(self._periodic_x)
        self._periodic_y_i32 = np.int32(self._periodic_y)

        # Bloch k != 0 (A6): F = F_re + j F_im with F(x + L) = F(x) e^{-j k L}, L = the meshed period (sum of the grid
        # widths: a periodic axis has no PML). The update is real and linear, so each part steps with the real
        # kernels; only the wraps mix them (kernel headers). The imaginary set (fields.imag, own CPML psi via
        # BoundaryManager.partner) exists only then; sources drive the real set (Bloch line sources both, with
        # e^{-j k s}), monitors with update() add j F_im, everything else reads the real set.
        config = boundary_manager.config
        phase_x, phase_y = config.bloch_phase_angles(float(np.sum(self._cell_dx)), float(np.sum(self._cell_dy)))
        self._wrap_cos_x, self._wrap_cos_y = self._real(math.cos(phase_x)), self._real(math.cos(phase_y))
        if config.has_bloch_phase:
            imaginary = type(fields)(self.nx, self.ny, xp=self.xp, dtype=self.dtype)
            fields.imag = imaginary
            sin_x, sin_y = math.sin(phase_x), math.sin(phase_y)
            self._field_sets = (_FieldSet(0, fields, boundary_manager, imaginary, sin_x, sin_y),
                                _FieldSet(1, imaginary, boundary_manager.partner(), fields, -sin_x, -sin_y))
        else:
            fields.imag = None
            self._field_sets = (_FieldSet(0, fields, boundary_manager, fields, 0.0, 0.0),)

        # da / db per cell, rewritten in place by one fused kernel (material updates never reallocate them).
        self.da = self.xp.empty((self.nx, self.ny), dtype=self.dtype)
        self.db = self.xp.empty((self.nx, self.ny), dtype=self.dtype)

        self.n_step = 0
        # Host float64 clock: float32 accumulation quantizes CW source phase (omega ~ 1e15) once t reaches tens of ps.
        self.time = 0.0

        # J.E accumulator for heat coupling: integral of the absorbed power density over the current window [J/m^3]
        # (divide by q_accum_time, host float64), allocated at the first enable so pure-EM runs skip the loss passes.
        self._enable_q_accum = False
        self.q_accum = None
        self.q_accum_time = 0.0
        # Last completed accumulation window, saved by reset_q_accum (buffer allocated at the first non-empty reset).
        self.last_q_accum = None
        self.last_q_accum_time = 0.0

        # C5 Kerr / TPA / free-carrier media (fdtd_nonlinear.NonlinearMedia), created by the first add_nonlinear, and
        # four-level gain regions (fdtd_gain.GainRegion, add_gain); the subclass supplies _nonlinear_components (its E
        # component sets) / _nonlinear_loss_target and calls both around its E update.
        self.nonlinear = None
        self.gain_media = []

        # (phase key) -> [graph or None after its eager run, kept inputs] (_fixed_block; gpu only); n_step_graphs =
        # captures made
        self.use_step_graphs = STEP_GRAPHS and self.xp is not np
        self._step_graphs = {}
        self.n_step_graphs = 0
        # (source tuple, whether its injections can reach a wall row) of the last source list (_walls_need_repin)
        self._repin_check = None
        # gpu: the FieldProbes among the monitors, sampled by one launch per step at the end of the real set's E graph
        # (taken over at their first step; (id, length) of the monitor list last scanned); _step_probes = this step's
        # launch until the E graph takes it, _probes_recorded = taken this step (stamp in _finish_step)
        self._probe_rows = None
        self._probe_scan = None
        self._step_probes = None
        self._probes_recorded = False

    # --- PhysicsSolver contract -----------------------------------------------

    def prepare(self, domain):
        # The engine is fully constructed at __init__; nothing to do.
        return

    def dt_limit(self) -> float:
        return float(self.dt)

    def step(self, t: float = None, dt: float = None) -> None:
        """One Yee iteration without sources or monitors (dt is advisory: the engine keeps its Courant dt)."""
        self.advance()

    def refresh_coefficients(self) -> None:
        """Rebuild coefficients left stale by a lazy slot write (no host sync); nothing pending by default."""
        return

    def inject_source(self, name: str, array) -> None:
        """'<out-of-plane>_source' (ez_source / hz_source): run-backend float32 (nx, ny) array kept by reference and
        added to the out-of-plane field every step (None clears it); a permittivity slot: copied into its array, then
        _permittivity_written() (ez rebuilds the E coefficients at once, hz at the next step)."""
        slots = PERMITTIVITY_SLOTS[self.polarization]
        source_slot = f'{self.polarization}_source'
        if name == source_slot:
            if array is not None:
                _check_cell_array(array, (self.nx, self.ny), DTYPE, f"inject_source('{source_slot}', ...) array",
                                  xp=self.xp)
            self._out_of_plane_source = array
        elif name in slots:
            if getattr(array, 'shape', None) != (self.nx, self.ny):
                raise ValueError(f"inject_source({name!r}, ...) array must have shape {(self.nx, self.ny)}, got "
                                 f"{getattr(array, 'shape', None)}")
            self.xp.copyto(getattr(self.materials, slots[name]), self.xp.asarray(array, dtype=DTYPE))
            self._permittivity_written()
        else:
            accepted = ', '.join(repr(slot) for slot in (source_slot, *slots))
            raise ValueError(f"{type(self).__name__} accepts sources {accepted}; got '{name}'")

    def _permittivity_written(self) -> None:
        """A permittivity slot was written: rebuild the E coefficients (a lazy engine marks them dirty instead)."""
        self._rebuild_e_coefficients()

    def finalize(self) -> None:
        self._rebuild_e_coefficients()
        self._rebuild_h_coefficients()
        self._sigma_e_changed()

    def materials_changed(self, names) -> None:
        """Rebuild the E and/or H update coefficients after in-place writes of the arrays they depend on."""
        if not self._E_COEFFICIENT_MATERIALS.isdisjoint(names):
            self._rebuild_e_coefficients()
        if not self._H_COEFFICIENT_MATERIALS.isdisjoint(names):
            self._rebuild_h_coefficients()
        if 'sigma_e' in names:
            self._sigma_e_changed()

    def _sigma_e_changed(self) -> None:
        """Hook after finalize and a sigma_e write: refresh what depends on the sigma_e > 0 cells (loss boxes)."""
        return

    def _rebuild_e_coefficients(self):
        raise NotImplementedError

    def _rebuild_h_coefficients(self):
        _kernels.h_coefficients(
            self.materials.sigma_m, self.materials.mu_r,
            self._dt_real, self._real(MU0),
            self.da, self.db)

    @property
    def precision(self) -> str:
        """Stepper precision of the fields: 'single' (float32) or 'double' (float64)."""
        return 'double' if self.dtype == np.float64 else 'single'

    def _require_single(self, what):
        """ValueError naming `what` on a double engine (C5 media keep float32 state and kernels)."""
        if self.dtype != DTYPE:
            raise ValueError(f'{what} runs with the single-precision stepper only (this engine steps float64 '
                             f'fields: stepper precision double)')

    @property
    def complex_fields(self) -> bool:
        """Whether a Bloch phase k != 0 made the engine step an imaginary field set (fields.imag)."""
        return len(self._field_sets) > 1

    def update_e(self):
        """Per field set: poles from E^n (+ their E^n loss half while accumulating), Yee E update with CPML / BC post,
        pole corrections (_update_e_set); the E^{n+1} loss half follows in advance()."""
        for field_set in self._field_sets:
            self._update_e_set(field_set)

    def add_nonlinear(self, mask, **params):
        """chi2 / Kerr / TPA / free-carrier medium on the run-backend bool (nx, ny) cell `mask` (fdtd_nonlinear.
        add_nonlinear_region: n2, tpa_beta, carrier_lifetime, fca_cross_section, fcd_coefficient, pump_wavelength,
        d_eff, poling; SI), on the E component sets of _nonlinear_components; single-precision stepper only."""
        from .fdtd_nonlinear import add_nonlinear_region   # it imports this module
        self._require_single('nonlinear_medium')
        add_nonlinear_region(self, mask, **params)
        self._nonlinear_added(mask)

    def _nonlinear_added(self, mask) -> None:
        """Hook after add_nonlinear (hz: loss-split boxes and cells)."""
        return

    def add_gain(self, mask, **params):
        """Four-level gain medium on the run-backend bool (nx, ny) cell `mask` (fdtd_gain.add_gain_region; SI), per E
        component set; single-precision stepper only."""
        from .fdtd_gain import add_gain_region   # it imports this module
        self._require_single('gain_medium')
        add_gain_region(self, mask, **params)

    def _walls_need_repin(self, sources) -> bool:
        """Whether the wall re-pin after injection can change a field: False only with no source array, pole, gain or
        nonlinear medium (their writes after the fused rows may reach a row) and every source a point cell clear of the
        pinned rows (BoundaryManager.clear_of_walls); the source check is kept per source list."""
        if (self._out_of_plane_source is not None or self.drude_poles or self.lorentz_poles or self.gain_media
                or self.nonlinear is not None):
            return True
        checked = self._repin_check
        if (checked is None or len(checked[0]) != len(sources)
                or any(kept is not source for kept, source in zip(checked[0], sources))):
            needed = not all(self._source_clear_of_walls(source) for source in sources)
            checked = self._repin_check = (tuple(sources), needed)
        return checked[1]

    def _source_clear_of_walls(self, source) -> bool:
        """True for a point-like source whose every cell is clear of every field set's wall rows."""
        cells = _point_source_cells(source)
        return cells is not None and all(field_set.boundary_mgr.clear_of_walls(i, j)
                                         for field_set in self._field_sets for i, j in cells)

    def _inject_imaginary_sources(self, sources, t_half) -> None:
        """Electric injection of the imaginary set (Bloch line sources, inject_electric_imaginary) at t_half."""
        if len(self._field_sets) > 1:
            imaginary = self._field_sets[1].fields
            for source in sources:
                if hasattr(source, 'inject_electric_imaginary'):
                    source.inject_electric_imaginary(imaginary, t_half)

    # --- loss accounting ------------------------------------------------------

    @property
    def enable_q_accum(self) -> bool:
        """Whether ohmic and pole losses accumulate into q_accum (allocated at the first enable)."""
        return self._enable_q_accum

    @enable_q_accum.setter
    def enable_q_accum(self, enabled) -> None:
        if enabled and self.complex_fields:
            raise NotImplementedError('absorbed power (track_absorption, em_heat_joule, q_accum) is not available with '
                                      'a Bloch phase k != 0: the complex field mixes the +k and -k responses')
        self._enable_q_accum = bool(enabled)
        if self._enable_q_accum:
            if self.q_accum is None:
                self.q_accum = self.xp.zeros((self.nx, self.ny), dtype=DTYPE)
            self._q_accum_enabled()

    def _q_accum_enabled(self) -> None:
        """Hook after every enable: allocate engine-specific loss buffers."""
        return

    def reset_q_accum(self) -> None:
        """Start a new accumulation window; a non-empty one is kept as last_q_accum / last_q_accum_time."""
        if self.q_accum_time > 0.0:
            if self.last_q_accum is None:
                self.last_q_accum = self.xp.empty_like(self.q_accum)
            self.xp.copyto(self.last_q_accum, self.q_accum)
            self.last_q_accum_time = self.q_accum_time
        if self.q_accum is not None:
            self.q_accum.fill(DTYPE(0.0))
        self.q_accum_time = 0.0

    def take_absorbed_power(self, merge_previous: bool = False):
        """Mean absorbed power density [W/m^3] since the last call (new array), then a window reset; None for an empty
        window or before accumulation was enabled. merge_previous (end-of-run flush) first folds in the last window."""
        window = float(self.q_accum_time)
        if self.q_accum is None or window <= 0.0:
            return None
        if merge_previous and self.last_q_accum is not None:
            self.q_accum += self.last_q_accum
            window += float(self.last_q_accum_time)
            self.q_accum_time = window
        power_density = self.q_accum * DTYPE(1.0 / window)
        self.reset_q_accum()
        return power_density

    # --- time stepping --------------------------------------------------------

    def _finish_step(self, t_full, monitors, resample_probes=False) -> None:
        """End of advance: E^{n+1} loss half while accumulating, clock to t_full, then the monitors at t_full (the
        FieldProbeRows row recorded in the E graph: re-sampled first when resample_probes, then stamped)."""
        if self._enable_q_accum:
            self._accumulate_e_new_half()
        self.n_step += 1
        self.time = t_full
        if self._probes_recorded:
            self._probes_recorded = False
            if resample_probes:
                self._probe_rows.rewrite_last_row()
            self._probe_rows.stamp(t_full)
        fields = self.fields
        for monitor in monitors:
            if not hasattr(monitor, 'update'):
                if not getattr(monitor, 'recorded_by_rows', False):
                    monitor.record(fields, t_full)
            elif self.n_step % monitor.sample_interval == 0:   # DFT monitors sample every sample_interval steps
                monitor.update(fields, t_full)

    def _take_probe_launch(self, monitors) -> None:
        """Start of advance (gpu): this step's probe_rows launch over the FieldProbes among monitors (taken over at
        their first step: float32 / float64 cupy fields of this engine), None without."""
        self._step_probes = None
        if self.xp is np:
            return
        scan = (id(monitors), len(monitors))
        if scan != self._probe_scan:
            self._probe_scan = scan
            for monitor in monitors:
                if isinstance(monitor, FieldProbe) and not monitor.recorded_by_rows:
                    if self._probe_rows is None:
                        self._probe_rows = FieldProbeRows(self.fields, self.dtype)
                    if self._probe_rows.accepts(monitor):
                        self._probe_rows.add(monitor)
        if self._probe_rows is not None:
            self._step_probes = self._probe_rows.next_launch()

    def _e_graph(self, k, key, block) -> None:
        """_fixed_block of field set k's E launches; the real set's (k 0) ends with this step's probe record (its
        launch id in the key, the launch kept with the entry)."""
        launch = self._step_probes if k == 0 else None
        if launch is None:
            self._fixed_block(key, (), block)
            return
        self._step_probes = None
        self._probes_recorded = True
        kernel, grid, threads, args = launch

        def recorded_block():
            block()
            kernel(grid, threads, args)

        self._fixed_block(key + (id(launch),), (launch,), recorded_block)

    def _probes_need_resample(self, sources, t_half, repin_walls, source_array_writes_e) -> bool:
        """Whether this step's probe record (inside the E graph) may be stale: a probe reads an E component and an
        eager write after the graph may have changed one (wall re-pin, PEC keep, non-fused rows, poles with the loss
        halves, media, a source array, or a source injection; an exact point source only while its float32 value at
        t_half is non-zero, the value its injection skips at)."""
        if not self._probes_recorded or self._probe_rows.components.isdisjoint(self._E_COMPONENTS):
            return False
        if (repin_walls or source_array_writes_e or self._pec_box is not None or self.gain_media
                or self.nonlinear is not None or not self._field_sets[0].boundary_mgr.fused_walls
                or ((self.drude_poles or self.lorentz_poles) and self._enable_q_accum)):
            return True
        return not all(self._e_injection_skipped(source, t_half) for source in sources)

    def _e_injection_skipped(self, source, t) -> bool:
        """True when the source's E-side injection at t writes nothing: no E-side injection (hz: no inject_electric),
        or an exact PointSource (ez) / ElectricDipoleSource (hz) whose DTYPE(amplitude temporal(t)) is 0."""
        source_type = type(source)
        if self.polarization == 'hz':
            if not hasattr(source, 'inject_electric'):
                return True
            if source_type is not ElectricDipoleSource:
                return False
        elif source_type is not PointSource:
            return False
        return DTYPE(source.amplitude * source.temporal(t)) == 0

    def _cpml_post(self, field_set, kind, *arrays):
        """CPML psi update + correction of one field set (BoundaryManager.apply_*_post without its outer-face rows):
        kind 'h' / 'e' (ez) or 'hz' / 'exy' (hz), arrays as the CPMLSided method takes them; nothing without CPML."""
        cpml = field_set.boundary_mgr.cpml
        if cpml:
            getattr(cpml, f'apply_{kind}')(*arrays)

    def _fixed_block(self, key, kept, block):
        """Run block() (launches with fixed arguments, no sync, no allocation): eagerly on the cpu, on the first step of
        a key and when capture failed, else replayed from its CUDA graph. key holds the ids of the replaceable inputs,
        kept alive with the entry (an id is never reused while its graph lives); fields, CPML psi, coefficients and the
        1D spacings are rewritten in place, never reassigned (physics/em sub file "FDTD step graphs")."""
        if not self.use_step_graphs:
            block()
            return
        entry = self._step_graphs.get(key)
        if entry is None:   # eager run first: compiles / loads the kernels outside the capture
            if len(self._step_graphs) >= _MAX_STEP_GRAPHS:
                self._step_graphs.clear()
            self._step_graphs[key] = [None, kept]
            block()
            return
        if entry[0] is None:
            entry[0] = self._capture(block)
            if entry[0] is None:
                block()
                return
        entry[0].launch(self.xp.cuda.get_current_stream())

    def _capture(self, block):
        """CUDA graph of block() (None, with a warning and eager steps from now on, when capture fails)."""
        cupy = self.xp
        stream = cupy.cuda.Stream(non_blocking=True)
        try:
            with stream:
                stream.begin_capture()
                try:
                    block()
                finally:
                    graph = stream.end_capture()
        except Exception as error:   # capture unsupported here: the same kernels launched one by one
            self.use_step_graphs = False
            self._step_graphs.clear()
            warnings.warn(f'em_fdtd_{self.polarization}: CUDA graph capture of the step kernels failed '
                          f'({type(error).__name__}: {error}); launching them one by one from now on', RuntimeWarning,
                          stacklevel=4)
            return None
        self.n_step_graphs += 1
        return graph

    # --- masks ----------------------------------------------------------------

    def _set_buffers(self, shape):
        """One zeroed pole buffer of `shape` in the field dtype per field set."""
        return [self.xp.zeros(shape, dtype=self.dtype) for _ in self._field_sets]

    def _check_cell_mask(self, mask, what):
        """TypeError unless `mask` is a run-backend bool (nx, ny) cell mask."""
        _check_cell_array(mask, (self.nx, self.ny), np.bool_, what, xp=self.xp)

    def _check_weight_shape(self, weight, what):
        """ValueError unless the host weight array has the (nx, ny) cell shape and lies in [0, 1]."""
        if weight.shape != (self.nx, self.ny) or not (np.all(weight >= 0.0) and np.all(weight <= 1.0)):
            raise ValueError(f"{what} must be a host (nx, ny) = ({self.nx}, {self.ny}) array in [0, 1], got shape "
                             f"{weight.shape}")

    def _pole_cell_mask(self, mask):
        """Checked run-backend bool cell mask of a pole; None = all cells."""
        if mask is None:
            return self.xp.ones((self.nx, self.ny), dtype=np.bool_)
        self._check_cell_mask(mask, 'Pole mask')
        return mask

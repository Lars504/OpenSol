"""EMPhysics: 2D FDTD in 'ez' (E_z, H_x, H_y) or 'hz' (H_z, E_x, E_y) polarisation with calibrated sources, monitors,
mode ports and material helpers; re-exports FDFDPhysics, ModePhysics and the EM building blocks."""
import math

import numpy as np
from ...core.backend import to_host

from ...core.constants import C0, DTYPE, ETA0
from ...core.solver_base import PhysicsSolver
from .fields import Fields2D_Ez, Fields2D_Hz
from .materials import DrudePole, LorentzPole, tfsf_outline_warnings
from .boundaries import BCType, BoundaryConfig, BoundaryManager, CPMLSided, solver_boundary_config
from .sources import (
    GaussianPulse, ContinuousWave, RickerWavelet,
    PointSource, LineSource, TFSFSource, TFSFSourceHz,
    MagneticLineCurrentSource, ElectricDipoleSource, TangentialSheetCurrentSource,
    PointCurrentCalibration, SheetCurrentCalibration, DipoleCalibration,
    BlochLineSource, BlochTangentialSheetCurrentSource, bloch_line_phase,
)
from .monitors import (FieldProbe, DFTMonitor, FluxMonitor, NearToFarField, NONLINEAR_HARMONIC_ORDER,
                       _check_frequencies, dft_sample_interval)
from .harmonic_fdtd import HarmonicAccumulator
from .mode_ports import ModeMonitor, ModePortSetup, ModeSource
from .placement import (PERMITTIVITY_ARRAYS, PERMITTIVITY_SLOTS, box_cells, box_window_faces, electric_dipole_faces,
                        flux_line_cells, mirror_planes, near_to_far_box_cells, point_source_cell, port_cells, probe_cell,
                        require_polarization, sheet_line, tfsf_box_cells, tfsf_open_faces)
from .renderer import EMRenderer, EMRendererHz
from .solvers import FDTDEngine2D_Ez, FDTDEngine2D_Hz
from .solvers.fdtd_2d_base import _bounding_box, add_pec_cells, record_dispersive_region
from .yee_grid import cpml_grading_cell_size, grid_cell_widths
from .subpixel_smoothing import SubpixelSmoothing
from .conformal_pec import ConformalPEC, build_conformal_pec
from .conformal_poles import conformal_pole_weights
from ...core.cut_cells import union_test
# Last: fdfd_2d and modes_2d import many sibling modules (renderer, harmonic, ...), so solvers/__init__ leaves them out.
from .solvers.fdfd_2d import FDFDPhysics
from .solvers.modes_2d import ModePhysics


__all__ = [
    'EMPhysics', 'FDFDPhysics', 'ModePhysics', 'FDTDEngine2D_Ez', 'FDTDEngine2D_Hz',
    'Fields2D_Ez', 'Fields2D_Hz',
    'DrudePole', 'LorentzPole',
    'BCType', 'BoundaryConfig', 'BoundaryManager', 'CPMLSided',
    'GaussianPulse', 'ContinuousWave', 'RickerWavelet',
    'PointSource', 'LineSource', 'TFSFSource', 'TFSFSourceHz',
    'FieldProbe', 'DFTMonitor', 'FluxMonitor', 'NearToFarField',
    'ModeSource', 'ModeMonitor',
    'EMRenderer', 'EMRendererHz',
]

# Per polarisation: (fields, engine, renderer) classes (permittivity arrays / slots and probe components: placement).
_POLARIZATION_CLASSES = {'ez': (Fields2D_Ez, FDTDEngine2D_Ez, EMRenderer),
                         'hz': (Fields2D_Hz, FDTDEngine2D_Hz, EMRendererHz)}
# MaterialGrid arrays the calibrated source injection depends on (cb / db and the local impedance at omega0).
_CALIBRATION_MATERIALS = {'ez': frozenset({'eps_r', 'sigma_e', 'mu_r', 'sigma_m'}),
                          'hz': frozenset({'eps_xx', 'eps_yy', 'sigma_e', 'mu_r', 'sigma_m'})}
# explicit_kernels stepper precision -> field dtype of the 2D engines (half_storage: em_fdtd_3d and others only)
_FIELD_DTYPES = {'single': np.dtype(np.float32), 'double': np.dtype(np.float64)}


class EMPhysics(PhysicsSolver):
    """2D FDTD on a Domain, polarisation 'ez' (E_z, H_x, H_y) or 'hz' (H_z, E_x, E_y): engine, BCs, calibrated sources,
    monitors and material helpers."""
    # hz reads eps_xx / eps_yy, never eps_r: direct MaterialGrid writes target those, then materials_changed. Losses:
    # take_absorbed_power(); state()['q_accum'] is None until engine.enable_q_accum is set (ez jz_total: None without
    # a Drude pole).

    name = "em"
    subpixel_smoothing = None   # SubpixelSmoothing of the model-wide switch (set_subpixel_smoothing); mode ports read it

    def __init__(self, domain, polarization: str = 'ez', boundary_config=None, precision: str = 'single'):
        if polarization not in _POLARIZATION_CLASSES:
            raise ValueError(f"polarization must be 'ez' (E_z, H_x, H_y) or 'hz' (H_z, E_x, E_y), "
                             f"got {polarization!r}")
        if precision not in _FIELD_DTYPES:
            raise ValueError(f"EMPhysics precision must be 'single' (float32 fields) or 'double' (float64), got "
                             f"{precision!r}")

        self.domain = domain
        self.polarization = polarization
        # Stepper precision (explicit_kernels): fields, coefficients, CPML, poles and sources in this dtype; materials,
        # coupling slots, q_accum, probe traces and box DFT spectra stay float32 / complex64.
        self.precision = precision
        self.dtype = dtype = _FIELD_DTYPES[precision]
        bc = solver_boundary_config(domain, boundary_config, 'EMPhysics')
        self.boundary_config = bc
        # Run backend of the Domain (cupy on 'gpu', numpy on 'cpu'): every array below lives there and every kernel is
        # a core.kernels twin, so one code path serves both.
        self.xp = xp = domain.xp

        grid = domain.grid
        fields_class, engine_class, self._renderer_class = _POLARIZATION_CLASSES[polarization]
        self.fields = fields_class(grid.nx, grid.ny, xp=xp, dtype=dtype)

        # CPML grading uses the PML cell size (dx_base on graded grids); graded spacings enter through dx_arr / dy_arr.
        graded_widths = ((np.asarray(grid.dx_arr), np.asarray(grid.dy_arr)) if domain.non_uniform
                         else (None, None))
        self.boundary_mgr = BoundaryManager(
            bc, grid.nx, grid.ny, cpml_grading_cell_size(domain), float(grid.dt),
            dx_arr=graded_widths[0], dy_arr=graded_widths[1], polarization=polarization, xp=xp, dtype=dtype,
        )

        self.engine = engine_class(
            grid, self.fields, domain.materials,
            boundary_manager=self.boundary_mgr,
        )

        self.sources = []
        self.monitors = []
        # Calibrated sources store their injection from cb / db / eps / mu_r (sheets: the complex eps(omega0) and mu
        # of their line cells, dispersive_regions included); the Python-bool dirty flag defers the recalibration to
        # the next step (or finalize) without device syncs. Mode ports join through ModePortSetup (modes solved by the
        # calibrations before the first step, later re-solved when their line's materials drifted).
        self._calibrations = []
        self._calibration_dirty = False
        # Read by the mode solves and sheet calibrations: every dispersive region as (pole, bounding-box slices, xp
        # bool mask cropped to the box; box and mask None = everywhere), the union PEC cell mask (xp bool, None until
        # a PEC region).
        self.dispersive_regions = []
        # Courant check of the poles: (pole, box, cropped weight) per region, the dispersive_regions mask or the
        # conformal pole weights (conformal_poles stability shares, which may reach a cell outside the mask).
        self._pole_stability_regions = []
        self.pec_cells = None
        # Conformal PEC (solver_defaults.conformal_boundaries): the model's geometry locator (set_conformal_boundaries),
        # the union cell mask and outline tests of add_conformal_pec_region, applied once at finalize (conformal_pec).
        self.conformal_locator = None
        self.conformal_pec = None
        self._conformal_mask = None
        self._conformal_tests = []
        # Mode-port and TFSF-outline warnings of the last finalize (model.builder logs them, Simulation.run warns them).
        self.finalize_warnings = []
        # Append-only warnings raised while stepping (mode-port re-solve notes and failures); the
        # time_dependent study logs and Simulation.run warns the entries added during a run.
        self.runtime_warnings = []
        # Window-sized running DFT of a frequency_domain study (attach_harmonic), else None.
        self.harmonic = None
        # phase_change.PhaseChangeMedium regions (eps / sigma_e written into the arrays; heat_em_phase_change steps them)
        self.phase_change_media = []

    def attach_harmonic(self, frequencies):
        """Append a HarmonicAccumulator at `frequencies` [Hz] to the monitors before the first step (frequency_domain
        study: harmonic_fdtd.py)."""
        self.harmonic = HarmonicAccumulator(self, frequencies)
        self.monitors.append(self.harmonic.monitor)
        return self.harmonic

    def select_harmonic_frequency(self, index):
        """Renderer harmonic fields show study frequency `index` (after the pulse is normalised)."""
        self.harmonic.select(index)

    @property
    def permittivity_arrays(self) -> tuple:
        """MaterialGrid permittivity arrays the engine reads: ('eps_r',) for ez, ('eps_xx', 'eps_yy') for hz."""
        return PERMITTIVITY_ARRAYS[self.polarization]

    # --- PhysicsSolver contract (delegate to engine) ---

    def prepare(self, domain):
        self.engine.prepare(domain)

    def step(self, t: float = None, dt: float = None):
        # Pending (lazy) coefficient rebuilds first, so a recalibration never reads stale cb / eps.
        self.engine.refresh_coefficients()
        if self._calibration_dirty:
            self._calibrate_sources()
        self.engine.advance(self.sources, self.monitors)

    def dt_limit(self) -> float:
        return self.engine.dt_limit()

    def state(self) -> dict:
        return self.engine.state()

    def inject_source(self, name: str, array):
        self.engine.inject_source(name, array)
        if name in PERMITTIVITY_SLOTS[self.polarization]:
            self._calibration_dirty = True

    def finalize(self) -> None:
        """Rebuild E/H coefficients from the final materials, check Courant stability, calibrate the sources now (mode
        ports solve their modes once), set the DFT sample interval before the first step; finalize_warnings = the mode
        ports' solve and placement warnings plus non-vacuum TFSF box edges."""
        self.finalize_warnings = []
        if self._conformal_tests and self.conformal_pec is None:
            self._apply_conformal_pec()
        self.engine.finalize()
        self._check_courant_stability()
        self._calibrate_sources()
        for port in (*self.sources, *self.monitors):
            if isinstance(port, (ModeSource, ModeMonitor)):
                self.finalize_warnings.extend(port.warnings)
        self.finalize_warnings.extend(tfsf_outline_warnings(
            self, [source for source in self.sources if isinstance(source, (TFSFSource, TFSFSourceHz))]))
        if self.engine.n_step == 0:   # a running DFT keeps its sample grid
            dft_monitors = [monitor for monitor in self.monitors if hasattr(monitor, 'update')]
            sample_interval = dft_sample_interval(float(self.engine.dt), self.sources,
                                                  [monitor.frequencies for monitor in dft_monitors],
                                                  NONLINEAR_HARMONIC_ORDER
                                                  if getattr(self.engine, 'nonlinear', None) is not None else 1)
            for monitor in dft_monitors:
                monitor.sample_interval = sample_interval

    def materials_changed(self, names) -> None:
        names = frozenset(names)
        self.engine.materials_changed(names)
        if not _CALIBRATION_MATERIALS[self.polarization].isdisjoint(names):
            self._calibration_dirty = True

    def set_subpixel_smoothing(self, geometry):
        """Average the current eps on the interface locations of `geometry` (core.interface_geometry, built once by
        model.subpixel_geometry) in every E-coefficient rebuild and in the mode-port lines; None switches it off."""
        self.subpixel_smoothing = None if geometry is None else SubpixelSmoothing(geometry, self.polarization)
        self.engine.subpixel_smoothing = self.subpixel_smoothing
        self.materials_changed(self.permittivity_arrays)

    def renderer(self):
        return self._renderer_class(self)

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: mean absorbed power density [W/m^3] since the last call (None for an empty window);
        merge_previous also averages over the last completed window (end-of-run flush)."""
        return self.engine.take_absorbed_power(merge_previous)

    def _check_courant_stability(self):
        """ValueError when eps * mu_r < (C0 dt)^2 (1/dx_i^2 + 1/dy_j^2) + mu_r * pole term in any cell, PML included
        (one host sync); eps = eps_r (ez) or min(eps_xx, eps_yy) (hz), pole term from _pole_stability_term."""
        dt = float(self.engine.dt)
        dx_cells, dy_cells = self.domain.cell_widths()
        light_step_squared = (C0 * dt) ** 2
        xp = self.xp
        required_x = xp.asarray((light_step_squared / dx_cells ** 2).astype(DTYPE))
        required_y = xp.asarray((light_step_squared / dy_cells ** 2).astype(DTYPE))
        materials = self.domain.materials
        if self.polarization == 'ez':
            permittivity, permittivity_label = materials.eps_r, 'eps_r'
        else:
            permittivity, permittivity_label = xp.minimum(materials.eps_xx, materials.eps_yy), 'min(eps_xx, eps_yy)'
        pole_term = self._pole_stability_term(dt)
        requirement = required_x[:, None] + required_y[None, :]
        if pole_term is not None:
            requirement = requirement + materials.mu_r * pole_term
        stability_ratio = permittivity * materials.mu_r
        stability_ratio /= requirement
        worst_ratio = float(stability_ratio.min())
        if worst_ratio >= 1.0 - 1e-6:
            return
        ix, iy = np.unravel_index(int(xp.argmin(stability_ratio)), stability_ratio.shape)
        x_centers, y_centers = self.domain.cell_centers()
        mu_r = float(materials.mu_r[ix, iy])
        eps_mu = float(permittivity[ix, iy]) * mu_r
        required = light_step_squared * (1.0 / dx_cells[ix] ** 2 + 1.0 / dy_cells[iy] ** 2)
        pole_text = ''
        if pole_term is not None and float(pole_term[ix, iy]) > 0.0:
            pole_labels = ', '.join(_pole_label(pole) for pole, box, cropped_weight in self._pole_stability_regions
                                    if _region_contains(box, cropped_weight, ix, iy))
            pole_text = (f' + mu_r * pole term {mu_r * float(pole_term[ix, iy]):.4g} of {pole_labels} (the leap-frog '
                         f'pole update adds mu_r (omega_p dt)^2 / 4 per Drude pole)')
        # Both terms scale with dt^2 (Lorentz about), so the factor scales with sqrt(worst_ratio); rounded down.
        stable_factor = math.floor(float(self.domain.grid.courant) * math.sqrt(max(worst_ratio, 0.0)) * 1e3) / 1e3
        raise ValueError(f'{permittivity_label} * mu_r = {eps_mu:.4g} at ({x_centers[ix]:.4g}, '
                         f'{y_centers[iy]:.4g}) m is below (C0 dt)^2 (1/dx^2 + 1/dy^2) = {required:.4g}{pole_text} '
                         f'there, so the time step is unstable; lower courant_factor to <= {stable_factor:.3f}, or '
                         f'raise {" / ".join(self.permittivity_arrays)} / mu_r')

    def _pole_stability_term(self, dt):
        """Per-cell sum of _pole_step_term x the region weight over _pole_stability_regions (run-backend DTYPE (nx, ny);
        None without poles), 0 on PEC cells; hz uses cell weights (a face's averaged pole weight over its averaged eps
        never exceeds a cell's)."""
        if not self._pole_stability_regions:
            return None
        grid = self.domain.grid
        pole_term = self.xp.zeros((grid.nx, grid.ny), dtype=DTYPE)
        for pole, box, cropped_weight in self._pole_stability_regions:
            step_term = DTYPE(_pole_step_term(pole, dt))
            if box is None:
                pole_term += step_term
            else:
                pole_term[box] += cropped_weight.astype(DTYPE) * step_term
        if self.pec_cells is not None:
            pole_term *= ~self.pec_cells
        return pole_term

    def _calibrate_sources(self):
        """Recompute every calibrated source's injection from the current engine coefficients and materials."""
        engine = self.engine
        materials = self.domain.materials
        for calibration in self._calibrations:
            calibration.apply(engine, materials)
        self._calibration_dirty = False

    def _add_calibrated(self, source, calibration):
        self.sources.append(source)
        self._calibrations.append(calibration)
        self._calibration_dirty = True
        return source

    # --- material helpers ---

    def add_dielectric_rectangle(self, x0, y0, x1, y1, eps_r, sigma_e=0.0):
        self._set_dielectric(self.domain.rect_mask(x0, y0, x1, y1), eps_r, sigma_e)

    def add_dielectric_circle(self, cx, cy, radius, eps_r, sigma_e=0.0):
        self._set_dielectric(self.domain.circle_mask(cx, cy, radius), eps_r, sigma_e)

    def _set_dielectric(self, mask, eps_r, sigma_e):
        """Isotropic relative permittivity eps_r into every permittivity array of this polarisation, plus sigma_e."""
        permittivity = {name: eps_r for name in self.permittivity_arrays}
        self.domain.materials.set_mask(mask, sigma_e=sigma_e, **permittivity)
        self.materials_changed(self.permittivity_arrays + ('sigma_e',))

    def add_drude_rectangle(self, x0, y0, x1, y1, drude: DrudePole):
        self._add_drude_background(drude, self.domain.rect_mask(x0, y0, x1, y1))

    def add_drude_circle(self, cx, cy, radius, drude: DrudePole):
        self._add_drude_background(drude, self.domain.circle_mask(cx, cy, radius))

    def _add_drude_background(self, drude, mask):
        """Drude pole whose eps_inf overwrites the permittivity arrays under the mask (E coefficients rebuilt)."""
        self.engine.add_drude(drude, mask, set_background=True)
        record_dispersive_region(self.dispersive_regions, drude, mask)
        record_dispersive_region(self._pole_stability_regions, drude, mask)
        self._calibration_dirty = True

    def add_dispersive(self, pole, mask, weights=None):
        """Dispersive pole on `mask` with the permittivity arrays kept as the background permittivity; weights = the
        region's conformal_poles.ConformalPoleWeights (cut-cell pole shares; dispersive_regions keeps the mask)."""
        engine_weight = None if weights is None else (
            weights.cell if self.polarization == 'ez' else (weights.face_x, weights.face_y))
        if isinstance(pole, DrudePole):
            self.engine.add_drude(pole, mask, set_background=False, weight=engine_weight)
        elif isinstance(pole, LorentzPole):
            self.engine.add_lorentz(pole, mask, weight=engine_weight)
        else:
            raise TypeError(f"add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}")
        record_dispersive_region(self.dispersive_regions, pole, mask)
        if weights is None:
            record_dispersive_region(self._pole_stability_regions, pole, mask)
        else:
            box = _bounding_box(weights.stability)
            if box is not None:
                self._pole_stability_regions.append((pole, box, self.xp.asarray(weights.stability[box], dtype=DTYPE)))
        self._calibration_dirty = True   # sheet sources calibrate on eps(omega0), poles included

    def conformal_pole_weights(self, mask, inside):
        """solver_defaults.conformal_boundaries: ConformalPoleWeights of a dispersive region (run-backend bool cell
        mask, drawn outline inside(x, y)) for add_dispersive (host bisection, one transfer)."""
        return conformal_pole_weights(self.domain, self.polarization, to_host(mask), inside, self._periodic_axes())

    def _periodic_axes(self) -> tuple:
        """(periodic_x, periodic_y) bools of the boundary configuration."""
        config = self.boundary_config
        return bool(getattr(config, 'periodic_x', False)), bool(getattr(config, 'periodic_y', False))

    def add_pec_region(self, mask):
        """Perfect electric conductor on a run-backend bool (nx, ny) mask; sources / ports recalibrate before the next
        step (their lines may now hold PEC cells)."""
        self.engine.add_pec_mask(mask)
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self._calibration_dirty = True

    def set_conformal_boundaries(self, locator):
        """solver_defaults.conformal_boundaries: pec_domain features call add_conformal_pec_region with
        locator.inside(entities) (model.boundary_locator.GeometryLocator, duck-typed); None = staircased PEC."""
        self.conformal_locator = locator

    def add_conformal_pec_region(self, mask, inside):
        """PEC on the run-backend bool (nx, ny) mask whose drawn outline is inside(x, y) -> bool (host): pec_cells now, the
        engine at finalize with conformal_pec (ez: exact wall distances, hz: free face lengths / areas)."""
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self._conformal_mask = add_pec_cells(self._conformal_mask, mask)
        self._conformal_tests.append(inside)
        self._calibration_dirty = True
        if self.polarization == 'ez':   # E_z keeps the cell mask; only the H links to it change
            self.engine.add_pec_mask(mask)

    def _apply_conformal_pec(self):
        """Build the ConformalPEC of the conformal regions (host) and hand it to the engine (one transfer each way)."""
        conformal = build_conformal_pec(self.domain, self.polarization, to_host(self.pec_cells),
                                        to_host(self._conformal_mask), union_test(self._conformal_tests),
                                        float(self.engine.dt), self._periodic_axes())
        if self.polarization == 'ez':
            self.engine.set_conformal_faces(conformal.face_cell, conformal.face_axis, conformal.face_extra)
        else:
            self.engine.add_pec_mask(self._conformal_mask, keep=(conformal.keep_x, conformal.keep_y))
            self.engine.set_conformal_cells(conformal.cell, conformal.cell_extra)
            conformal = ConformalPEC('hz', cell=conformal.cell, cell_extra=conformal.cell_extra,
                                     area_fraction=conformal.area_fraction)   # the keep masks live on the engine
        self.conformal_pec = conformal
        self._conformal_mask = None

    # --- sources ---

    def add_point_source(self, x, y, wavelength=None, frequency=None,
                         bandwidth=0.5, current=1e-3, waveform='gaussian'):
        """z-directed filament current `current` [A] at (x, y): J = I / (dx_i dy_j), injected as cb * J per step."""
        require_polarization(self.polarization, 'ez', 'add_point_source')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        ix, iy, cell_area = point_source_cell(self.domain, x, y, "Point source")
        source = PointSource(ix, iy, temporal, amplitude=1.0, weight=self.xp.zeros(1, dtype=DTYPE))
        return self._add_calibrated(source, PointCurrentCalibration(source, current, cell_area))

    def add_magnetic_line_current(self, x, y, wavelength=None, frequency=None,
                                  bandwidth=0.5, magnetic_current=1.0, waveform='gaussian'):
        """z-directed magnetic line current K_m = `magnetic_current` [V] at (x, y) (hz only), injected as db * K_m /
        (dx_i dy_j) per step; in a uniform medium |H_z| = omega eps0 eps_r K_m / 4 |H0^(2)(k r)|."""
        require_polarization(self.polarization, 'hz', 'add_magnetic_line_current')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        ix, iy, cell_area = point_source_cell(self.domain, x, y, "Magnetic line current")
        source = MagneticLineCurrentSource(ix, iy, temporal, amplitude=1.0, weight=self.xp.zeros(1, dtype=DTYPE))
        return self._add_calibrated(source, PointCurrentCalibration(source, magnetic_current, cell_area,
                                                                    coefficient='db'))

    def add_electric_dipole(self, x, y, wavelength=None, frequency=None, bandwidth=0.5,
                            dipole_moment=1e-3, orientation=0.0, waveform='gaussian'):
        """In-plane electric dipole (hz only), M = `dipole_moment` [A] (integral of J over the cross-section) along
        `orientation` [deg, ccw from +x] on the faces around the cell; |H_z| = k M/4 |H1^(2)(k r)| |sin(phi - theta)|."""
        require_polarization(self.polarization, 'hz', 'add_electric_dipole')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        ix, iy, _ = point_source_cell(self.domain, x, y, "Electric dipole")
        faces = electric_dipole_faces(self.domain, self.boundary_config, ix, iy, dipole_moment, orientation)
        source = ElectricDipoleSource(ix, iy, temporal, faces, xp=self.xp)
        return self._add_calibrated(source, DipoleCalibration(source))

    def add_line_source(self, axis, position, wavelength=None, frequency=None,
                        bandwidth=0.5, amplitude=1.0, waveform='gaussian', span=None):
        """Sheet current radiating `amplitude` [V/m] to each side in the local medium (|eta| at the centre frequency,
        sigma_e and poles included); axis 'x' = line along x at y = position. span = (start, end) physical -> cells
        [cell(start), cell(end)+1), an end on a cpml window edge continued through the PML; None = whole grid line."""
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        return self._add_sheet_current(axis, position, temporal, amplitude, span, "Line source")

    def add_gaussian_beam(self, axis, position, waist, center, wavelength=None, frequency=None,
                          bandwidth=0.5, amplitude=1.0, waveform='gaussian', span=None):
        """Line source with transverse factor exp(-((s - center)/waist)^2); `amplitude` = peak field [V/m] at the waist."""
        if not waist > 0:
            raise ValueError(f"Gaussian beam waist must be > 0, got {waist}")
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        return self._add_sheet_current(axis, position, temporal, amplitude, span, "Gaussian beam",
                                       waist=waist, center=center)

    def _add_sheet_current(self, axis, position, temporal, amplitude, span, label, waist=None, center=None):
        """ez: E_z sheet through the line cells' centres; hz: E_x (axis 'x') / E_y (axis 'y') on their hi faces."""
        sheet = sheet_line(self.domain, self.boundary_config, self.polarization, axis, position, span, label,
                           waist=waist, center=center)
        weight = self.xp.asarray(2.0 * sheet.transverse_shape / (float(ETA0) * sheet.normal_width), dtype=DTYPE)
        profile = self.xp.zeros(weight.size, dtype=DTYPE)
        if self.engine.complex_fields:   # Bloch k != 0: the sheet carries e^{-j k.r} (an oblique wave)
            source_class = BlochLineSource if self.polarization == 'ez' else BlochTangentialSheetCurrentSource
        else:
            source_class = LineSource if self.polarization == 'ez' else TangentialSheetCurrentSource
        source = source_class(axis, sheet.line_cell, temporal, amplitude, span=sheet.span_cells, profile=profile)
        if self.engine.complex_fields:
            source.set_bloch_phase(bloch_line_phase(self.domain.grid, self.boundary_config, axis, sheet.line_cell,
                                                    sheet.span_cells))
        return self._add_calibrated(source, SheetCurrentCalibration(source, weight, self, sheet.normal_width))

    def add_tfsf(self, x0, y0, x1, y1, wavelength=None, frequency=None,
                 bandwidth=0.5, amplitude=1.0, waveform='gaussian'):
        """+x plane wave of incident peak `amplitude` [V/m] (E_z on ez; E_y on hz, H_z = E_y / eta0) inside the box
        (uniform or graded grids: the incident line runs on the grid's x widths; box edges in vacuum, else finalize
        warns); a box reaching a y mirror face of the wave's parity opens onto it (placement.tfsf_box_cells)."""
        d = self.domain
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        cells = tfsf_box_cells(d, x0, y0, x1, y1, boundary_config=self.boundary_config,
                               polarization=self.polarization)
        open_faces = tfsf_open_faces(self.boundary_config, box_window_faces(d, cells), self.polarization)
        tfsf_class = TFSFSource if self.polarization == 'ez' else TFSFSourceHz
        source = tfsf_class(*cells, temporal, float(d.dx), float(d.grid.dt), amplitude,
                            cell_widths=grid_cell_widths(d.grid), open_faces=open_faces, xp=self.xp, dtype=self.dtype)
        self.sources.append(source)
        return source

    # --- monitors ---

    def add_field_probe(self, x, y, field_component=None):
        """Raw Yee samples of one field component at the cell containing (x, y); None = the out-of-plane field."""
        ix, iy, field_component = probe_cell(self.domain, self.polarization, x, y, field_component)
        probe = FieldProbe(ix, iy, field_component)
        self.monitors.append(probe)
        return probe

    def add_dft_monitor(self, x0, y0, x1, y1, wavelengths=None, frequencies=None, downsample=1):
        """Running DFT over the cell box; downsample s > 1 records every s-th cell along x and y (memory / s^2)."""
        ix0, ix1, iy0, iy1 = box_cells(self.domain, x0, y0, x1, y1, "DFT monitor")
        bc = self.boundary_config
        monitor = DFTMonitor(slice(ix0, ix1), slice(iy0, iy1),
                             _resolve_frequencies(wavelengths, frequencies), float(self.domain.grid.dt),
                             polarization=self.polarization, periodic_x=bc.periodic_x, periodic_y=bc.periodic_y,
                             downsample=downsample, xp=self.xp)
        self.monitors.append(monitor)
        return monitor

    def add_flux_monitor(self, axis, position, span, wavelengths=None, frequencies=None):
        """Poynting flux through a line (axis 'x' = along x at y = position, as add_line_source; flux > 0 toward +y);
        span = (start, end) along the line [m]. The monitor's own `axis` is the normal."""
        normal_axis, line_cell, span_cells, along_widths = flux_line_cells(self.domain, axis, position, span)
        monitor = FluxMonitor(normal_axis, line_cell, span_cells, _resolve_frequencies(wavelengths, frequencies),
                              float(self.domain.grid.dt), along_widths, polarization=self.polarization, xp=self.xp)
        self.monitors.append(monitor)
        return monitor

    def add_near_to_far(self, x0, y0, x1, y1, wavelengths=None, frequencies=None):
        """Near-to-far-field box: the out-of-plane field and the tangential in-plane field (face mean) sampled on the
        centres of columns x0, x1 and rows y0, y1 of the half-open cell box, corners included; the pattern unfolds
        over the mirror faces."""
        d = self.domain
        ix0, ix1, iy0, iy1 = near_to_far_box_cells(d, x0, y0, x1, y1)
        monitor = NearToFarField(ix0, ix1, iy0, iy1, _resolve_frequencies(wavelengths, frequencies),
                                 float(d.grid.dt), d.cell_centers(), polarization=self.polarization,
                                 mirrors=mirror_planes(d, self.boundary_config), xp=self.xp)
        self.monitors.append(monitor)
        return monitor

    # --- mode ports (slab modes along a line, solved at finalize / the first step, re-solved on material drift) ---

    def add_mode_source(self, axis, position, span, wavelength=None, frequency=None, bandwidth=0.5, power=1.0,
                        mode_number=1, direction='positive', waveform='gaussian'):
        """One-sided TFSF source of slab mode `mode_number` carrying `power` [W/m] toward +normal ('positive') or
        -normal on a line (axis 'x' = along x at y = position); span = (start, end) along the line [m]."""
        if waveform not in ('gaussian', 'cw'):
            raise ValueError(f"mode source waveform must be 'gaussian' or 'cw', got {waveform!r}")
        power = float(power)
        if not (math.isfinite(power) and power > 0):
            raise ValueError(f"mode source power must be finite and > 0 [W/m], got {power}")
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        line_cell, span_cells, label = port_cells(self.domain, axis, position, span, 'Mode source')
        source = ModeSource(self, axis, line_cell, span_cells, temporal, power, mode_number, direction, label)
        return self._add_calibrated(source, ModePortSetup(self, source))

    def add_mode_monitor(self, axis, position, span, wavelengths=None, frequencies=None, mode_number=1):
        """Powers of slab mode `mode_number` travelling toward +normal / -normal through a line (axis as
        add_mode_source) at each monitor frequency (results.mode_monitor_result)."""
        line_cell, span_cells, label = port_cells(self.domain, axis, position, span, 'Mode monitor')
        monitor = ModeMonitor(self, axis, line_cell, span_cells, _resolve_frequencies(wavelengths, frequencies),
                              mode_number, label)
        self.monitors.append(monitor)
        self._calibrations.append(ModePortSetup(self, monitor))
        self._calibration_dirty = True
        return monitor


def _resolve_frequency(wavelength, frequency) -> float:
    """Source frequency [Hz] from exactly one of wavelength [m] / frequency [Hz] (positive, finite)."""
    if (wavelength is None) == (frequency is None):
        raise ValueError(f"pass exactly one of wavelength or frequency, got wavelength={wavelength}, "
                         f"frequency={frequency}")
    value = float(frequency if frequency is not None else wavelength)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"source wavelength / frequency must be finite and > 0, got {value}")
    return value if frequency is not None else C0 / value


def _resolve_frequencies(wavelengths, frequencies):
    """Monitor frequencies [Hz] (1D float64) from exactly one of wavelengths [m] / frequencies [Hz]."""
    if (wavelengths is None) == (frequencies is None):
        raise ValueError("pass exactly one of wavelengths or frequencies")
    if frequencies is not None:
        return _check_frequencies(frequencies)
    return C0 / _check_frequencies(wavelengths, 'wavelengths')


def _make_temporal(waveform, frequency, bandwidth):
    if waveform == 'gaussian':
        return GaussianPulse(frequency, bandwidth)
    if waveform == 'cw':
        return ContinuousWave(frequency)
    if waveform == 'ricker':
        return RickerWavelet(frequency)
    raise ValueError(f"unknown waveform {waveform!r}; valid: 'gaussian', 'cw', 'ricker'")


def _pole_step_term(pole, dt) -> float:
    """Pole term of the leap-frog bound (z = -1 mode, independent of gamma and sigma_e): Drude (omega_p dt)^2 / 4,
    Lorentz delta_eps (omega_0 dt)^2 / (4 - (omega_0 dt)^2); ValueError when omega_0 dt >= 2."""
    if isinstance(pole, DrudePole):
        return (float(pole.omega_p) * dt) ** 2 / 4.0
    resonance_step_squared = (float(pole.omega_0) * dt) ** 2
    if resonance_step_squared >= 4.0:
        raise ValueError(f'{_pole_label(pole)}: omega_0 dt = {math.sqrt(resonance_step_squared):.4g} >= 2, so the '
                         f'Lorentz update is unstable; lower courant_factor or refine the mesh')
    return float(pole.delta_eps) * resonance_step_squared / (4.0 - resonance_step_squared)


def _pole_label(pole) -> str:
    if isinstance(pole, DrudePole):
        return f'Drude pole omega_p = {float(pole.omega_p):.4g} rad/s'
    return f'Lorentz pole delta_eps = {float(pole.delta_eps):.4g}, omega_0 = {float(pole.omega_0):.4g} rad/s'


def _region_contains(box, cropped_mask, ix, iy) -> bool:
    """Whether a dispersive_regions entry (box None = everywhere) covers cell (ix, iy) (host sync)."""
    if box is None:
        return True
    rows, columns = box
    if not (rows.start <= ix < rows.stop and columns.start <= iy < columns.stop):
        return False
    return bool(cropped_mask[ix - rows.start, iy - columns.start])

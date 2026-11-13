"""NonlinearHarmonicPhysics (physics type em_fdfd_nonlinear): one private FDFDPhysics per harmonic channel on a
shared Domain, chi(2) / chi(3) bulk media, nonlinear sheets and the mixing table, solved as the undepleted cascade or the
pump-depleted self-consistent solution (Picard / Anderson or Newton-Krylov)."""
# Composition (nonlinear_optics_plan decision 1): every channel keeps its own HarmonicSystem, factor, sources, monitors
# and ports; features apply to channel(name). Nonlinear terms are NonlinearPolarizationPhasor sources appended to the
# target channels at finalize (or the first solve), reading the ChannelFieldStore snapshot of the source channels;
# pump-depleted Kerr terms also get a KerrOperatorTerm (lagged operator permittivity). Only this physics is registered
# in the Simulation; the shared MaterialGrid arrays are written once and every channel is told. Every array is on the
# run backend (self.xp = domain.xp: cupy or numpy, as the channels' FDFDPhysics).
import copy
import math

from ...core.backend import to_host
from ...core.constants import C0, DTYPE
from ...core.solver_base import PhysicsSolver
from ...solvers.sparse_direct import SparseDirectSolver
from ..em.boundaries import BCType, solver_boundary_config
from ..em.harmonic import cell_permittivity
from ..em.placement import PERMITTIVITY_ARRAYS, PERMITTIVITY_SLOTS, sheet_line
from ..em.adjoint import is_objective
from ..em.design_density import DesignRegion, paint_regions
from ..em.solvers.fdfd_2d import FDFDPhysics
from ..em.yee_grid import grid_cell_widths
from .absorption import absorbed_power_density, carrier_density, kerr_density, kerr_terms, sheet_kerr_sites, work_fields
from .adjoint import cascade_sensitivity
from .iteration import (ITERATION_METHODS, PumpDepletedSettings, solve_newton_krylov, solve_pump_depleted,
                        solve_undepleted)
from .media import bulk_medium
from .mixing import (MILLER_REFERENCE_HARMONICS, build_mixing_table, describe_mixing, inexact_term_issues,
                     miller_scales, resolve_channels, scaled_table, solve_order)
from .renderer import NonlinearRenderer
from .sheets import line_sheet, sheet_site_work, surface_sheet
from .sources import (ChannelFieldStore, FreeCarriers, KerrOperatorTerm, NewtonLinearization,
                      NonlinearPolarizationPhasor, cell_areas, delivered_power)

COUPLING_MODES = ('undepleted', 'pump_depleted')
STATE_POLARIZATIONS = ('ez', 'hz')   # state() carries both phasor pairs (zeros for an absent polarisation)
_PERMITTIVITY_NAMES = ('eps_r', 'eps_xx', 'eps_yy')
_PERMITTIVITY_SLOTS = {**PERMITTIVITY_SLOTS['ez'], **PERMITTIVITY_SLOTS['hz']}   # slot -> MaterialGrid array
_NONLINEAR_SOURCES = (NonlinearPolarizationPhasor, KerrOperatorTerm)


def channel_boundary_config(base, channel):
    """Copy of the shared BoundaryConfig for one mixing.Channel: its Bloch k on the periodic axes (bloch faces where
    k != 0) and its mirror parity (an even channel, parity_exponent 0, turns antisymmetric faces symmetric)."""
    config = copy.copy(base)
    for axis, wavenumber in zip('xy', channel.bloch_k):
        if getattr(base, f'periodic_{axis}') and wavenumber != 0.0:
            setattr(config, f'{axis}_lo', 'bloch')
            setattr(config, f'{axis}_hi', 'bloch')
            setattr(config, f'bloch_k{axis}', float(wavenumber))
    if channel.parity_exponent == 0:
        for face in base.ghost_faces:
            if getattr(base, face) == BCType.ANTISYMMETRIC:
                setattr(config, face, 'symmetric')
    config._validate()
    return config


def channel_plane_signs(base, channel):
    """{mirror face: +1 symmetric / -1 antisymmetric} of a channel's E field (channel_boundary_config)."""
    return {face: (-1 if getattr(base, face) == BCType.ANTISYMMETRIC and channel.parity_exponent else 1)
            for face in base.ghost_faces}


class NonlinearHarmonicPhysics(PhysicsSolver):
    """Cascaded frequency-domain chi(n) mixing between declared harmonic channels (mixing.ChannelSpec), each solved by
    its own FDFDPhysics; undepleted = channels solved once in rank order with the nonlinear sources of lower ranks,
    pump_depleted = every term (+ steady-state free carriers), block Gauss-Seidel sweeps with Anderson acceleration
    (iteration_method 'picard') or Newton-Krylov ('newton') to the self-consistent solution."""

    name = 'nl'

    def __init__(self, domain, channel_specs, boundary_config=None, linear_solver='auto', coupling_mode='undepleted',
                 tolerance=1e-8, max_iterations=50, relaxation=1.0, anderson_depth=5, continuation_steps=1,
                 iteration_method='picard'):
        if coupling_mode not in COUPLING_MODES:
            raise ValueError(f"coupling_mode must be one of {', '.join(COUPLING_MODES)}, got {coupling_mode!r}")
        if iteration_method not in ITERATION_METHODS:
            raise ValueError(f"iteration_method must be one of {', '.join(ITERATION_METHODS)}, got "
                             f"{iteration_method!r}")
        self.domain = domain
        self.xp = domain.xp
        self.coupling_mode = coupling_mode
        self.iteration_settings = PumpDepletedSettings(float(tolerance), int(max_iterations), float(relaxation),
                                                       int(anderson_depth), int(continuation_steps),
                                                       iteration_method)
        self.channels, self.input_names = resolve_channels(channel_specs)
        # one sparse_direct runtime shared by the channels (each keeps its own factor): the build's 'linear' solver,
        # or for a back_end choice (scripts) a private one whose counters work_counters reports
        self._owns_linear_solver = not isinstance(linear_solver, SparseDirectSolver)
        if self._owns_linear_solver:
            linear_solver = SparseDirectSolver(back_end=linear_solver, backend=domain.backend.name)
        self.linear_solver = linear_solver
        # the shared faces (k = 0, the inputs' mirror parity); each channel solves on its own copy (Bloch k, parity)
        self.boundary_config = solver_boundary_config(domain, boundary_config, 'NonlinearHarmonicPhysics')
        if self.boundary_config.has_bloch_phase:
            raise ValueError('em_fdfd_nonlinear: give the Bloch k per input channel (ChannelSpec.bloch_k; generated '
                             'channels follow their combination), not on the boundary_config')
        self.channel_physics = []
        for channel in self.channels:
            physics = FDFDPhysics(domain, channel.polarization, channel_boundary_config(self.boundary_config, channel),
                                  wavelength=2.0 * math.pi * C0 / channel.omega, linear_solver=linear_solver)
            physics.omega = channel.omega   # exact combination frequency (no wavelength round trip)
            self.channel_physics.append(physics)
        self.polarizations = tuple(dict.fromkeys(channel.polarization for channel in self.channels))
        self.store = ChannelFieldStore(self.channels, domain, self.boundary_config,
                                       [physics.lower_wrap_factors for physics in self.channel_physics])
        self.media = []               # media.BulkMedium
        self.sheets = []              # sheets.SheetMedium (surface chi(2), 2D-material sheets)
        self.design_regions = []      # em design_density.DesignRegion (optimization study; writes the shared arrays)
        self._sensitivity = None      # (key, adjoint.EpsSensitivity): eps_sensitivity() cache
        self.adjoint_record = None    # pump_depleted adjoint GMRES of the last gradient (adjoint.depleted_cotangents)
        self._adjoint_products = 0    # summed transposed sweeps of every pump_depleted gradient (work_counters)
        self.mixing = None           # mixing.MixingTable, built at finalize / the first solve
        self.applied_terms = ()       # the terms acting in coupling_mode (undepleted: the rank-filtered cascade)
        self.cascade = None           # iteration.CascadeRecord of the last solve
        self.finalize_warnings = []
        self.enable_q_accum = False   # loss-protocol flag set by the runner; q_abs comes from every channel solve
        first_input = next(channel for channel in self.channels if channel.frequency_name == self.input_names[0])
        self.selected_channel = first_input.name
        self.reference_omega = first_input.omega   # chi(3) n2 / beta conversion default
        self._sources = {}            # target channel index -> applied NonlinearPolarizationPhasor
        self._diagnostic_sources = {}   # target index -> phasor of every table term (power-exchange identity)
        self._kerr_terms = {}         # target index -> KerrOperatorTerm (pump_depleted picard)
        self.free_carriers = None     # sources.FreeCarriers (pump_depleted with a chi(3) carrier model)
        self.linearization = None     # sources.NewtonLinearization (pump_depleted newton)
        self._power_scalars = None    # (store versions, scalars) cache
        self._cell_area = None
        self._q_abs_total = None
        self._q_abs_key = None        # (store versions, channel n_solves) of _q_abs_total
        self._dirty = True
        self._zeros = self.xp.zeros((domain.grid.nx, domain.grid.ny), dtype=DTYPE)   # absent polarisation (read only)

    # --- channels and media ---

    @property
    def permittivity_arrays(self) -> tuple:
        """Registered permittivity arrays among eps_r, eps_xx, eps_yy (at least the channels' own; the model registers
        all three), which thermo-optic and the eps slots write."""
        registered = self.domain.materials.property_names()
        own = {name for polarization in self.polarizations for name in PERMITTIVITY_ARRAYS[polarization]}
        return tuple(name for name in _PERMITTIVITY_NAMES if name in registered or name in own)

    @property
    def host_factor_bytes(self):
        """Summed host factor bytes of the channels (SuperLU L + U, MUMPS / PARDISO; Simulation.memory_report), None
        before any factor."""
        sizes = [physics.host_factor_bytes for physics in self.channel_physics if physics.host_factor_bytes is not None]
        return sum(sizes) if sizes else None

    def channel_index(self, name):
        """Index of channel `name` in self.channels; ValueError listing the names."""
        for index, channel in enumerate(self.channels):
            if channel.name == name:
                return index
        raise ValueError(f"no harmonic channel {name!r}; channels: {', '.join(c.name for c in self.channels)}")

    def channel(self, name):
        """The FDFDPhysics of channel `name` (feature apply target)."""
        return self.channel_physics[self.channel_index(name)]

    @property
    def selected_physics(self):
        return self.channel(self.selected_channel)

    def bloch_periods(self):
        """(L_x, L_y) [m] meshed grid lengths of the periodic axes (None on the others): the Bloch periods."""
        widths = grid_cell_widths(self.domain.grid)
        return tuple(float(axis_widths.sum()) if periodic else None
                     for axis_widths, periodic in zip(widths, (self.boundary_config.periodic_x,
                                                               self.boundary_config.periodic_y)))

    @property
    def has_odd_mirror(self):
        """True when a shared mirror face is antisymmetric (the input channels' parity -1 there)."""
        return any(getattr(self.boundary_config, face) == BCType.ANTISYMMETRIC
                   for face in self.boundary_config.ghost_faces)

    def plane_signs(self, name):
        """{mirror face: E parity +1 / -1} of channel `name` (generated even combinations are +1 everywhere)."""
        return channel_plane_signs(self.boundary_config, self.channels[self.channel_index(name)])

    def add_bulk_medium(self, name, order, tensor, mask, poling=None, cell_scale=None, process_scales=(),
                        miller_reference_omega=None, carriers=None, carrier_index_squared=None):
        """chi(n) medium (model-axes tensor [(m/V)^(n-1)]) on a bool full-grid mask (run backend); chi(2) poling =
        full-grid real values (sign per cell) or None; cell_scale = scalar or full-grid real factor per cell (chi(3)
        n0^2) or None; process_scales ((process, factor), ...); Miller's rule relative to miller_reference_omega [rad/s]
        (None = off); carriers = media.CarrierModel (chi(3), n0^2 = carrier_index_squared); the mixing table is rebuilt
        at the next solve."""
        self._check_new_medium(name, mask)
        medium = bulk_medium(name, order, tensor, mask, poling, cell_scale, process_scales=process_scales,
                             miller_reference_omega=miller_reference_omega, carriers=carriers,
                             carrier_index_squared=carrier_index_squared)
        self.media.append(medium)
        self.mixing = None
        self._dirty = True
        return medium

    def _check_new_medium(self, name, *masks):
        """ValueError for a taken medium / sheet name, TypeError for a mask that is not a bool full-grid array of the
        run backend."""
        if any(medium.name == name for medium in (*self.media, *self.sheets)):
            raise ValueError(f"nonlinear medium {name!r} exists already")
        for mask in masks:
            self._check_mask(mask, 'medium mask')

    def _check_mask(self, mask, what):
        xp = self.xp
        if not isinstance(mask, xp.ndarray) or mask.dtype != xp.bool_ or mask.shape != self._zeros.shape:
            raise TypeError(f"{what} must be a {xp.__name__} bool array of shape {self._zeros.shape}")

    def _check_sheet_bloch(self, name):
        """NotImplementedError when a channel carries a Bloch phase: the sheet sites sample and deposit across the
        periodic wrap without it."""
        if any(physics.boundary_config.has_bloch_phase for physics in self.channel_physics):
            raise NotImplementedError(f'nonlinear sheet {name!r}: sheets need Bloch k = 0 on every channel (their '
                                      'sites take no Bloch wrap phase)')

    def _add_sheet(self, sheet):
        self.sheets.append(sheet)
        self.mixing = None
        self._dirty = True
        return sheet

    def add_surface_sheet(self, name, local_tensor, mask, outside_mask=None, selvedge='inside', boundary_test=None):
        """Surface chi(2) (sheets.surface_local_tensor [m^2/V], local (n, t, z) frame) on the faces between the cells of
        a bool full-grid mask (run backend) and unselected cells (outside_mask: only those); boundary_test = host (x, y)
        -> bool of the mask's drawn shape in Domain coordinates (smoothed normals and lengths) or None (staircase);
        selvedge 'inside' | 'outside' = the cell whose normal E the sites sample and receive."""
        self._check_new_medium(name, mask, *(() if outside_mask is None else (outside_mask,)))
        self._check_sheet_bloch(name)
        config = self.boundary_config
        x_edges, y_edges = self.domain.cell_edges()
        return self._add_sheet(surface_sheet(
            name, local_tensor, to_host(mask), None if outside_mask is None else to_host(outside_mask),
            grid_cell_widths(self.domain.grid), (bool(config.periodic_x), bool(config.periodic_y)), selvedge,
            boundary_test, (x_edges[0], y_edges[0]), xp=self.xp))

    def add_line_sheet(self, name, axis, position, span, order, local_tensor, response='polarization',
                       process_scales=()):
        """2D-material sheet on an axis-aligned line (axis 'x' = along x at y = position; span as add_line_source):
        order-n sheet-frame tensor (sheets.sheet_chi2_local [m^2/V], response 'polarization'; sheet_sigma3_local [S
        m^2/V^2], response 'current': K = sigma E E E, P_s = K / (j omega)); sites as conducting_sheet."""
        self._check_new_medium(name)
        self._check_sheet_bloch(name)
        polarization = 'hz' if 'hz' in self.polarizations else 'ez'   # hz: refuses the outer grid wall
        line = sheet_line(self.domain, self.boundary_config, polarization, axis, position, span,
                          f'Nonlinear sheet {name}')
        return self._add_sheet(line_sheet(name, order, local_tensor, axis, line.line_cell, line.along,
                                          self._zeros.shape, grid_cell_widths(self.domain.grid), response,
                                          process_scales, xp=self.xp))

    def background_index_squared(self, omega):
        """float32 full-grid n0^2 = (Re sqrt(eps(omega)))^2 of the isotropic permittivity (eps_r, else eps_xx) with the
        channels' poles at omega [rad/s]: the chi(3) n2 / beta conversion per cell."""
        materials = self.domain.materials
        name = 'eps_r' if 'eps_r' in materials.property_names() else 'eps_xx'
        eps = cell_permittivity(materials, name, self.channel_physics[0].dispersive_regions, float(omega), xp=self.xp)
        return (self.xp.sqrt(eps).real ** 2).astype(DTYPE)

    def add_dispersive(self, pole, mask):
        """Drude / Lorentz pole on a bool cell mask (run backend) for every channel (each evaluates it at its own
        omega)."""
        for physics in self.channel_physics:
            physics.add_dispersive(pole, mask)

    def add_pec_region(self, mask):
        """Perfect electric conductor on a bool (nx, ny) cell mask (run backend) in every channel."""
        for physics in self.channel_physics:
            physics.add_pec_region(mask)

    def _medium_chi1(self, medium, omega):
        """Re chi1 = mean Re eps_r(omega) - 1 over the medium's cells (poles included; host sync): Miller's rule."""
        materials = self.domain.materials
        name = 'eps_r' if 'eps_r' in materials.property_names() else 'eps_xx'
        xp = self.xp
        eps = cell_permittivity(materials, name, self.channel_physics[0].dispersive_regions, float(omega), xp=xp)
        inside = medium.cell_factor != 0
        n_inside = int(xp.sum(inside))
        if n_inside == 0:   # e.g. an n2 / beta medium on cells with n0^2 = 0 (metal): 0 / 0 otherwise
            raise ValueError(f"nonlinear medium {medium.name!r}: Miller's rule needs cells with a nonzero nonlinear "
                             'factor (n0^2 = (Re sqrt eps)^2 is 0 on every selected cell)')
        return float(xp.sum(xp.where(inside, eps[medium.box].real, 0.0))) / n_inside - 1.0

    def _miller_table(self, table):
        """The table with Miller's-rule factors on the terms of media with a Miller reference (chi1 per channel)."""
        chi1_by_medium = {}
        for medium in self.media:
            if medium.miller_reference_omega is None:
                continue
            reference_omega = medium.miller_reference_omega
            target, sources = MILLER_REFERENCE_HARMONICS[medium.order]
            reference = math.prod(self._medium_chi1(medium, harmonic * reference_omega)
                                  for harmonic in (target, *sources))
            if reference == 0.0:
                raise ValueError(f"nonlinear medium {medium.name!r}: Miller's rule needs chi1 = eps - 1 != 0 at the "
                                 'reference frequencies')
            chi1 = [self._medium_chi1(medium, channel.omega) for channel in self.channels]
            chi1_by_medium[medium.name] = (chi1, reference)
        return scaled_table(table, miller_scales(table, chi1_by_medium)) if chi1_by_medium else table

    def _build_mixing(self):
        """Mixing table of the channels and media (Miller factors applied); (re)attach the applied phasors, free
        carriers and the pump-depleted Kerr operator terms (picard) or the Newton linearisation (newton)."""
        table = build_mixing_table(self.channels, [medium.mixing_tensor() for medium in (*self.media, *self.sheets)])
        if not table.terms:
            reasons = '; '.join(table.dropped[:3]) or 'no medium, or no exact frequency combination of the channels'
            raise ValueError(f"no process couples the channels ({reasons})")
        problems = inexact_term_issues(table, self.bloch_periods(), self.has_odd_mirror)
        if problems:
            raise ValueError('; '.join(problems))
        table = self._miller_table(table)
        pump_depleted = self.coupling_mode == 'pump_depleted'
        newton = pump_depleted and self.iteration_settings.iteration_method == 'newton'
        applied = table.terms if pump_depleted else table.undepleted_terms()
        if not applied:
            raise ValueError("undepleted: no process feeds a channel from channels of lower rank (only back-action or "
                             "Kerr terms); declare generated channels as combinations of the inputs, or use "
                             "coupling_mode 'pump_depleted'")
        for physics in self.channel_physics:
            physics.sources = [source for source in physics.sources if not isinstance(source, _NONLINEAR_SOURCES)]
            if physics.extra_permittivity is not None:
                physics.set_extra_permittivity(None)
        self._sources, self._diagnostic_sources, self._kerr_terms = {}, {}, {}
        bulk_names = {medium.name for medium in self.media}   # sheet SPM / XPM stay in the rhs (no operator term)
        kerr_applied = tuple(term for term in applied if term.kerr and term.medium in bulk_names)
        carriers = pump_depleted and kerr_applied and any(medium.carriers is not None for medium in self.media)
        self.free_carriers = FreeCarriers(kerr_applied, self.store, self.media) if carriers else None
        for target in range(len(self.channels)):
            target_terms = tuple(term for term in applied if term.target == target)
            if target_terms or self.free_carriers is not None:
                source = NonlinearPolarizationPhasor(target, target_terms, self.store, self.media,
                                                     skip_unsolved=pump_depleted, free_carriers=self.free_carriers,
                                                     sheets=self.sheets)
                self.channel_physics[target]._add_source(source)
                self._sources[target] = source
            all_terms = table.terms_for(target)
            if pump_depleted and target in self._sources:
                self._diagnostic_sources[target] = self._sources[target]   # the same terms
            elif all_terms:
                self._diagnostic_sources[target] = NonlinearPolarizationPhasor(target, all_terms, self.store,
                                                                               self.media, skip_unsolved=True,
                                                                               sheets=self.sheets)
            kerr = tuple(term for term in target_terms if term.kerr and term.medium in bulk_names)
            if pump_depleted and not newton and (kerr or self.free_carriers is not None):
                self._kerr_terms[target] = KerrOperatorTerm(target, kerr, self.store, self.media,
                                                            free_carriers=self.free_carriers)
                self.channel_physics[target]._add_source(self._kerr_terms[target])
        self.linearization = (NewtonLinearization(self._sources, self.store, self.media, self.free_carriers,
                                                  sheets=self.sheets) if newton else None)
        self.mixing = table
        self.applied_terms = applied
        self._power_scalars = None

    def mixing_lines(self):
        """describe_mixing lines of the applied terms (study log), [] before the table exists."""
        return [] if self.mixing is None else describe_mixing(self.mixing, self.applied_terms)

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Solve the nonlinear problem when a medium, material, source or channel changed since the last solve."""
        if self._dirty or any(physics._solution_dirty for physics in self.channel_physics):
            self.solve_nonlinear()

    def dt_limit(self) -> float:
        return math.inf

    def reset_records(self) -> None:
        """RerunSolver (reused sweep build): the next study run solves afresh on the kept channel factors (monitors keep
        one value per omega, so the rerun overwrites them) from zero fields, as a fresh build: the same result per value
        (no hysteresis branch) and, measured on the SHG sweep, fewer Newton products than a warm start."""
        self.store.clear()
        self._dirty = True

    def state(self) -> dict:
        """Selected channel's float32 <pol>_re / _im of both polarisations (zeros for the other one), q_abs, the
        permittivity arrays, sigma_e and 'q_accum': None (loss-protocol key)."""
        selected = self.selected_physics
        selected_state = selected.state()
        state = {}
        for polarization in STATE_POLARIZATIONS:
            for part in ('re', 'im'):
                key = f'{polarization}_{part}'
                state[key] = selected_state[key] if polarization == selected.polarization else self._zeros
        state['q_abs'] = selected_state['q_abs']
        materials = self.domain.materials
        for name in self.permittivity_arrays:
            state[name] = getattr(materials, name)
        state['sigma_e'] = materials.sigma_e
        state['q_accum'] = None
        return state

    def inject_source(self, name: str, array):
        """Copy a coupling's array into the permittivity array of slot `name` ('eps_r_update', 'eps_xx_update',
        'eps_yy_update') once and tell every channel (those reading the array reassemble)."""
        array_name = _PERMITTIVITY_SLOTS.get(name)
        if array_name is None or array_name not in self.permittivity_arrays:
            slots = [slot for slot, array in _PERMITTIVITY_SLOTS.items() if array in self.permittivity_arrays]
            raise ValueError(f"NonlinearHarmonicPhysics accepts sources {', '.join(slots)}; got '{name}'")
        self.xp.copyto(getattr(self.domain.materials, array_name), self.xp.asarray(array, dtype=DTYPE))
        self.materials_changed({array_name})

    def finalize(self) -> None:
        """Finalize every channel (back end, size rule, port modes; warnings prefixed with the channel) and build the
        mixing table (ValueError when no process couples the channels)."""
        self.finalize_warnings = []
        for channel, physics in zip(self.channels, self.channel_physics):
            physics.finalize()
            self.finalize_warnings.extend(f'channel {channel.name}: {warning}' for warning in physics.finalize_warnings)
        self._build_mixing()

    def materials_changed(self, names) -> None:
        for physics in self.channel_physics:
            physics.materials_changed(names)

    def renderer(self):
        return NonlinearRenderer(self)

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: absorbed power [W/m^3] of the last solve, float32 full grid (cached per solve, never
        written by the caller): the channels' linear q_abs plus the exact two-photon and free-carrier absorption of
        the applied terms, the same for picard and newton (absorption.absorbed_power_density); None before a solve."""
        if self.cascade is None:
            return None
        key = (tuple(self.store.versions), tuple(physics.n_solves for physics in self.channel_physics))
        if self._q_abs_total is None:
            self._q_abs_total = self.xp.zeros(self._zeros.shape, dtype=DTYPE)
        if self._q_abs_key != key:
            absorbed_power_density(self, self._q_abs_total)
            self._q_abs_key = key
        return self._q_abs_total

    # --- study hooks ---

    def solve_nonlinear(self, before_channel=None, after_sweep=None):
        """Solve the channels in coupling_mode -> CascadeRecord (undepleted: once each in rank order; pump_depleted:
        sweeps to the tolerance, converged False after max_iterations); before_channel(channel) runs before each channel
        solve (progress, cancel), after_sweep(step, sweep, residual) after each pump-depleted sweep."""
        if self.mixing is None:
            self._build_mixing()
        order = solve_order(self.channels)
        if self.coupling_mode == 'undepleted':
            self.cascade = solve_undepleted(self.channel_physics, self.channels, order, self.store, before_channel)
        elif self.linearization is not None:
            self.cascade = solve_newton_krylov(self.channel_physics, self.channels, order, self.store,
                                               self.linearization, self.iteration_settings, _NONLINEAR_SOURCES,
                                               before_channel, after_sweep)
        else:
            self.cascade = solve_pump_depleted(self.channel_physics, self.channels, order, self.store,
                                               self._kerr_terms, self.iteration_settings, _NONLINEAR_SOURCES,
                                               before_channel, after_sweep)
        self._dirty = False
        return self.cascade

    def select_channel(self, name):
        """Show channel `name` in state() and the renderer (snapshots of the nonlinear study)."""
        self.channel_index(name)
        self.selected_channel = name

    # --- adjoint sensitivity and design regions (optimization study; adjoint.py) ---

    def solve_design(self):
        """optimization study hook: solve the cascade at the current design (channels record their objectives)."""
        self.solve_nonlinear()

    def eps_sensitivity(self):
        """adjoint.cascade_sensitivity of the channels' objectives at the last solve (undepleted cascade or pump_depleted
        fixed point), cached until the next solve or objective change."""
        objectives = tuple(monitor for physics in self.channel_physics for monitor in physics.monitors
                           if is_objective(monitor))
        key = (tuple(physics.n_solves for physics in self.channel_physics), objectives)
        if self._sensitivity is None or self._sensitivity[0] != key:
            self._sensitivity = None   # free the previous gradient first
            self.adjoint_record = None
            self._sensitivity = (key, cascade_sensitivity(self))
            if self.adjoint_record is not None:
                self._adjoint_products += self.adjoint_record['gmres_products'] + 2   # + the rhs and gradient sweeps
        return self._sensitivity[1]

    def add_design_region(self, mask, eps_min, eps_max, initial_density=0.5, filter_radius=0.0, projection_beta=0.0,
                          projection_eta=0.5):
        """Density design region (em design_density.DesignRegion) on a bool (nx, ny) mask (run backend) writing eps
        into every permittivity array (eps_r, eps_xx, eps_yy: isotropic); ValueError when it overlaps an earlier
        region."""
        self._check_mask(mask, 'design region mask')
        region = DesignRegion(self.domain, mask, self.permittivity_arrays, eps_min, eps_max, initial_density,
                              filter_radius, projection_beta, projection_eta)
        if any(region.overlaps(other) for other in self.design_regions):
            raise ValueError('design regions of one physics must not share cells')
        self.design_regions.append(region)
        region.write()
        self.materials_changed(self.permittivity_arrays)
        return region

    def design_density(self):
        """float32 (nx, ny) projected density rho_bar of every design region, 0 elsewhere (fresh array)."""
        return paint_regions(self.design_regions, self._zeros.shape, self.xp)

    def polarization_magnitude(self, index):
        """float32 full-grid |P_NL| [C/m^2] of the applied source of channel `index` (zeros without one)."""
        source = self._sources.get(index)
        if source is None or self.cascade is None:
            return self._zeros
        return source.magnitude()

    def carrier_density_map(self):
        """float32 full-grid steady-state carrier density N [1/m^3] of the stored fields (zeros without carriers)."""
        if self.free_carriers is None or self.cascade is None:
            return self._zeros
        return self.free_carriers.density_map(self._zeros.shape)

    def scalars(self) -> dict:
        """power_exchange_<channel> = W_k [W/m] of the applied terms incl. free carriers (2D: per unit length in z),
        power_exchange_imbalance = |sum W| / sum |W| of every table term, nonlinear_absorption = -W of the applied Kerr
        terms (bulk media and sheets: TPA), free_carrier_absorption = -W of the carrier terms, peak_carrier_density
        [1/m^3], iterations (sweeps / Newton steps) and residual of the last solve."""
        if self.cascade is None:
            return {}
        key = tuple(self.store.versions)
        if self._power_scalars is None or self._power_scalars[0] != key:
            scalars = {f'power_exchange_{channel.name}': 0.0 for channel in self.channels}
            powers = {}   # id(source) -> W: pump_depleted diagnostic sources are the applied ones (host syncs once)

            def power_of(source):
                if id(source) not in powers:
                    powers[id(source)] = self._delivered_power(source)
                return powers[id(source)]
            for target, source in self._sources.items():
                scalars[f'power_exchange_{self.channels[target].name}'] = power_of(source)
            exchange = [power_of(source) for source in self._diagnostic_sources.values()]
            total = sum(abs(power) for power in exchange)
            scalars['power_exchange_imbalance'] = abs(sum(exchange)) / total if total > 0.0 else 0.0
            scalars['nonlinear_absorption'] = (-sum(self._kerr_work(target, source)
                                                    for target, source in self._sources.items())
                                               if self.coupling_mode == 'pump_depleted' else 0.0)
            scalars['free_carrier_absorption'] = -sum(self._carrier_work(target) for target in self._sources)
            scalars['peak_carrier_density'] = (0.0 if self.free_carriers is None
                                               else self.free_carriers.peak_density())
            scalars['iterations'] = float(self.cascade.sweeps)
            scalars['residual'] = float(self.cascade.residual)
            self._power_scalars = (key, scalars)
        return dict(self._power_scalars[1])

    @property
    def cell_area(self):
        """float64 full-grid Yee cell areas [m^2] (cached)."""
        if self._cell_area is None:
            self._cell_area = cell_areas(self.domain)
        return self._cell_area

    def _delivered_power(self, source):
        """W [W/m] of a phasor on its target channel's stored field (bulk centre + sheet site work)."""
        return source.power(self.cell_area)

    def _kerr_work(self, target, source):
        """W [W/m] of the applied SPM / XPM terms (bulk centre work + sheet site work) of channel `target`'s phasor on
        the stored fields (< 0: two-photon absorption)."""
        field = self.store.centre_vector(target)
        if field is None:
            return 0.0
        omega = self.channels[target].omega
        work = 0.0
        kerr = kerr_terms(source)
        if kerr:
            density = kerr_density(self, target, kerr, work_fields(self, target, kerr, carriers=False))
            work += delivered_power(omega, field, density, self.cell_area)
        sheet_part = sheet_kerr_sites(self, target, kerr_terms(source, sheet=True))
        if sheet_part is not None:
            sites, sheets, grids = sheet_part
            work += sheet_site_work(sites, sheets, grids[target], omega, self.xp)
        return work

    def _carrier_work(self, target):
        """W [W/m] of the free-carrier polarisation of channel `target` on the stored fields (< 0: absorption)."""
        if self.free_carriers is None:
            return 0.0
        field = self.store.centre_vector(target)
        if field is None:
            return 0.0
        density = carrier_density(self, target, work_fields(self, target, ()))
        return delivered_power(self.channels[target].omega, field, density, self.cell_area)

    def solve_summary(self):
        """One line per channel solve summary for the study log; None before the first solve."""
        if self.cascade is None:
            return None
        return '; '.join(f'channel {channel.name} ({channel.polarization}): {physics.solve_summary()}'
                         for channel, physics in zip(self.channels, self.channel_physics))

    def work_counters(self):
        """Channels, sweeps (Newton steps), Newton GMRES products, pump_depleted adjoint transposed sweeps and summed
        channel solves (summary 'solvers' physics counters); factorisations / recycled GMRES are the 'linear' role's
        record, or added here for a private (script) solver."""
        cascade = self.cascade
        counters = {'channels': len(self.channels), 'sweeps': 0 if cascade is None else cascade.sweeps,
                    'newton_gmres_iterations': 0 if cascade is None else cascade.gmres_iterations,
                    'adjoint_sweeps': self._adjoint_products,
                    'solves': sum(physics.n_solves for physics in self.channel_physics)}
        if self._owns_linear_solver:
            counters.update(self.linear_solver.work_counters())
        return counters

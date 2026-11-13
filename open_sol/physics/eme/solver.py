"""EMEPhysics: bidirectional eigenmode expansion of a 2D in-plane model along x or y (eme_plan stage 1): sections from
the mesh, slab line modes per section (PML modes kept), mode-matched interfaces, the S-matrix cascade, port results and
the field reconstructed over the window. EMEBase holds the solve flow shared with the 3D EMEPhysics3D (solver3d)."""
# FDFDPhysics attribute set (domain, polarization, boundary_config, sources, monitors, dispersive_regions, pec_cells,
# finalize_warnings, runtime_warnings, permittivity_arrays, enable_q_accum, solved_omega, n_solves), so PortLine,
# couplings and the Visualizer treat it alike. Quasi-static: step solves at omega when dirty, solve_harmonic always.
# Material vocabulary 'modes' (eps_r, eps_xx, eps_yy all exist); a 2D solve reads ez eps_r / hz eps_xx, eps_yy.
# Faces along the propagation axis are semi-infinite ports at the window planes (their bc and PML are not read).
# Bends (add_bend): the slices whose centre lies in a bend range are the straightened bend (arc length at the reference
# radius along the axis); their slice rows carry a bend label row, so sections split at the range ends and the cache
# keys differ. Periodic block (add_periodic): the sections of [start, stop] are the unit cell, drawn once; the
# interface after it becomes (J * cell)^(n - 1) * exit (J = the junction last -> first cell section), so the ports see
# n periods and the drawn cell shows the first one. Adjoint (adjoint.py): EMEBase add_objective / eps_sensitivity
# (each subclass its _cascade_sensitivity), 2D also add_design_region, the FDFDPhysics protocol of the optimization
# study; the cascade (and, with a periodic link, the cell cascade and exit interface: periodic_adjoint) is kept while an
# objective exists.
import math
import time

import numpy as np
from ...core.backend import to_host

from ...core.constants import C0, DTYPE, EPS0
from ...core.solver_base import PhysicsSolver
from ...core.wavelengths import wavelength_label
from ..em.adjoint import EpsSensitivity, sensitivity_density
from ..em.boundaries import WALL_SIGNS, BCType, MIRROR_TYPES, solver_boundary_config
from ..em.design_density import DesignRegion, paint_regions
from ..em.materials import DrudePole, LorentzPole, register_em_arrays
from ..em.mode_ports import PortLine
from ..em.operators import check_zero_bloch
from ..em.slab_modes import solve_slab_modes
from ..em.solvers.fdtd_2d_base import _check_cell_array, add_pec_cells, record_dispersive_region
from ..em.yee_grid import grid_cell_widths
from .adjoint import (EMEObjective, PeriodicAdjointFields, face_gradient_to_cells, objective_cotangents,
                      section_gradient)
from .cascade import (PORT_FACES, CascadeBuilder, SMatrix, bloch_modes, cell_cascade, forward_bloch, repeated_link,
                      total_smatrix, unit_cell)
from .interfaces import interface_defects, interface_smatrix
from .mode_sets import INPUT_POWER_FLOOR, bent_line, gauss_mean, gauss_positions, mode_set, section_fields
from .renderer import EMERenderer
from .sections import slice_stack, stack_sections

EME_POLARIZATIONS = ('ez', 'hz')
PROPAGATION_AXES = ('x', 'y')
EME_PERMITTIVITY_ARRAYS = ('eps_r', 'eps_xx', 'eps_yy')
EME_PERMITTIVITY_SLOTS = {f'{array}_update': array for array in EME_PERMITTIVITY_ARRAYS}
# MaterialGrid arrays a 2D solve reads (as the FDFD system / PortLine rows)
_READ_ARRAYS = {'ez': ('eps_r', 'sigma_e', 'mu_r', 'sigma_m'), 'hz': ('eps_xx', 'eps_yy', 'sigma_e', 'mu_r', 'sigma_m')}
# hz permittivity arrays per propagation axis: (cells of the line, eps_t; faces along the line, eps_l) = PortLine rows
_HZ_LINE_ARRAYS = {'x': ('eps_yy', 'eps_xx'), 'y': ('eps_xx', 'eps_yy')}
MODE_CACHE_BYTES = 1 << 30       # mode sets kept per solve; the rest are re-solved for the field reconstruction
LENGTH_SWEEP_BATCH_BYTES = 1 << 26   # host complex blocks of the stretch factors re-cascaded together (one chunk)
LENGTH_SWEEP_LIVE_BLOCKS = 16        # (N, N) complex128 blocks per stretch factor alive inside a stacked star product


def length_sweep_chunk(n_modes):
    """Stretch factors the length sweep re-cascades together at n_modes modes per section (>= 1)."""
    return max(1, int(LENGTH_SWEEP_BATCH_BYTES // (LENGTH_SWEEP_LIVE_BLOCKS * 16 * max(int(n_modes), 1) ** 2)))
_RECIPROCITY_WARN = 1e-3         # interface |s12 - s21^T| / |s21| above which a runtime warning asks for more modes
# Wall cell outside the line solve range = sign x its inner neighbour (out-of-plane field): em.boundaries.WALL_SIGNS
SECTION_TABLE_COLUMNS = ('section', 'start', 'length', 'n_slices', 'n_modes', 'n_eff_1', 'n_eff_2', 'power_defect',
                         'reciprocity_defect')
SECTION_TABLE_UNITS = ('1', 'm', 'm', '1', '1', '1', '1', '1', '1')
PORT_TABLE_COLUMNS = ('output_face', 'mode_number', 'n_eff', 'n_eff_imag', 's_abs2', 'phase_deg', 'power_fraction')
PORT_TABLE_UNITS = ('', '1', '1', '1', '1', 'deg', '1')
BLOCH_TABLE_COLUMNS = ('bloch_mode', 'n_bloch', 'n_bloch_imag', 'multiplier_abs', 'phase_per_period_deg',
                       'dominant_mode')
BLOCH_TABLE_UNITS = ('1', '1', '1', '1', 'deg', '1')


def positive_wavelength_omega(wavelength, label):
    """omega = 2 pi C0 / wavelength; ValueError unless the wavelength is finite and > 0 [m]."""
    wavelength = float(wavelength)
    if not (math.isfinite(wavelength) and wavelength > 0.0):
        raise ValueError(f'{label} wavelength must be finite and > 0 [m], got {wavelength!r}')
    return 2.0 * math.pi * C0 / wavelength


def slice_positions(widths, window):
    """Full-axis float64 slice centres [m from the window's lo edge] (NaN outside the window)."""
    widths = np.asarray(widths, dtype=np.float64)
    positions = np.full(widths.size, np.nan)
    inside = widths[window]
    positions[window] = np.cumsum(inside) - 0.5 * inside
    return positions


class EMEBase(PhysicsSolver):
    """Solve flow of both EME physics: sections (_find_sections), cached section mode sets (_solve_section),
    interfaces, cascade, port scalars, field (_reconstruct), monitors; excitation, bends, periodic block, study hooks."""

    name = 'eme'
    tracks_modes = False   # True: _solve_section starts from the previous wavelength's entry (3D shift-invert)
    objective_unit = '1'   # J = port power fractions (the optimization study labels; FDFD objectives are W/m)

    def __init__(self, domain, wavelength, n_modes, section_tolerance, read_arrays, label):
        self.domain = domain
        self._omega = positive_wavelength_omega(wavelength, label)
        self.n_modes = int(n_modes)
        self.section_tolerance = float(section_tolerance)
        self._read_arrays = tuple(read_arrays)
        self.sources = []   # none (placement / Visualizer protocol)
        self.monitors = []  # EMEPortsMonitor / EMESectionTableMonitor / EMELengthSweepMonitor / EMEPeriodicMonitor /
        #                     adjoint.EMEObjective
        self.dispersive_regions = []
        self.pec_cells = None
        self.finalize_warnings = []
        self.runtime_warnings = []    # append-only; the studies log new entries
        self.enable_q_accum = False   # loss-protocol flag only: every solve computes q_abs
        self.excitation = ('lo', 1, 1.0)   # (input face, 1-based port mode, power [W/m in 2D, W in 3D])
        self.n_port_modes = 4              # port modes reported per face (scalars, port table)
        self.bends = []                    # (start, stop, radius, reference) [m]; ranges along the axis
        self.periodic = None               # EMEPeriodicMonitor of the periodic block, or None

        self.sections = []            # sections.Section of the last solve
        self.port_modes = {}          # face -> ModeSet of the last solve
        self.smatrix = None           # cascade.SMatrix between the lo and hi window planes (all modes)
        self.section_rows = []        # SECTION_TABLE_COLUMNS rows of the last solve
        self.sweep_data = None        # (k per section, interfaces) of the last solve while a length sweep is attached
        self.periodic_data = None     # unit cell of the last solve (EMEPeriodicMonitor), or None
        self.cascade = None           # cascade.Cascade of the last solve while an EMEObjective is attached (adjoint)
        # ((first, last), n_periods, cell Cascade, exit interface) of a periodic link of n_periods > 1 while the
        # cascade is kept, or None
        self.periodic_adjoint = None
        self.solved_omega = None
        self.n_solves = 0
        self.last_solve_seconds = None
        self._counters = {'solves': 0, 'sections': 0, 'mode_solves': 0, 'cache_hits': 0, 'mode_resolves': 0,
                          'tracked_solves': 0, 'interfaces': 0, 'solve_seconds': 0.0}
        self._last_counts = {}
        self._scalars = {}
        self._mode_cache = {}         # section key -> solve entry (last item a ModeSet) of the last solve's omega
        self._cache_omega = None
        self._link = None             # (first, last, n_periods, junction) of the last solve's periodic link, or None
        self._sensitivity = None      # ((n_solves, objectives), EpsSensitivity) of the last eps_sensitivity
        self._dirty = True

    @property
    def permittivity_arrays(self) -> tuple:
        """MaterialGrid permittivity arrays of the 'modes' vocabulary."""
        return EME_PERMITTIVITY_ARRAYS

    @property
    def omega(self) -> float:
        return self._omega

    @omega.setter
    def omega(self, omega):
        omega = float(omega)
        if not (math.isfinite(omega) and omega > 0.0):
            raise ValueError(f'EME omega must be finite and > 0 [rad/s], got {omega!r}')
        if omega != self._omega:
            self._omega = omega
            self._dirty = True

    def set_wavelength(self, wavelength):
        """Vacuum wavelength [m] of the next stationary solve (a reused sweep build); the mode cache is kept (3D
        solves start from the previous wavelength's modes)."""
        self.omega = positive_wavelength_omega(wavelength, type(self).__name__)

    def reset_records(self):
        """RerunSolver (reused sweep build): nothing to drop (every monitor keeps only the last solve); the next step
        solves when the wavelength or another input changed."""

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Solve at omega when materials, PEC, poles or the excitation changed (stationary study, flush)."""
        if self._dirty:
            self._solve()

    def dt_limit(self) -> float:
        return math.inf

    def inject_source(self, name: str, array):
        """Copy a coupling's array into the permittivity array of slot `name`; the next step re-solves when it is one
        the solve reads."""
        if name not in EME_PERMITTIVITY_SLOTS:
            raise ValueError(f"EME accepts sources {', '.join(EME_PERMITTIVITY_SLOTS)}; got '{name}'")
        array_name = EME_PERMITTIVITY_SLOTS[name]
        xp = self.domain.xp
        xp.copyto(getattr(self.domain.materials, array_name), xp.asarray(array, dtype=DTYPE))
        if array_name in self._read_arrays:
            self._dirty = True

    def finalize(self) -> None:
        self.finalize_warnings = []

    def materials_changed(self, names) -> None:
        if not set(self._read_arrays).isdisjoint(names):
            self._dirty = True

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: q_abs [W/m^3] of the last solve (float32 full grid on domain.xp, overwritten by the next), None
        before the first solve; merge_previous is ignored (no accumulation window)."""
        return None if self.solved_omega is None else self.q_abs

    # --- study hooks ---

    def solve_harmonic(self, omega):
        """Solve now at `omega` [rad/s] (frequency_domain study hook) and record the monitors."""
        self.omega = omega
        self._solve()

    def scalars(self) -> dict:
        """transmission_<k>, transmission, reflection of the last solve (E5 scalar outputs; {} before it)."""
        return dict(self._scalars)

    def work_counters(self):
        """Cumulative work (study summary 'solvers'): solves, sections, mode solves, cache hits, re-solves for the
        field, solves started from the previous wavelength's modes (3D), interfaces, host wall seconds."""
        return dict(self._counters)

    def solve_summary(self):
        """One line on the last solve for the study log; None before it."""
        if self.last_solve_seconds is None:
            return None
        counts = self._last_counts
        return (f"{counts['sections']} sections at {wavelength_label(2.0 * math.pi * C0 / self.solved_omega)}: "
                f"{counts['mode_solves']} mode solves ({counts['cache_hits']} cached, {counts['mode_resolves']} "
                f"re-solved for the field, {counts['tracked_solves']} started from the previous wavelength), up to "
                f"{counts['max_modes']} modes per section, {self.last_solve_seconds:.3g} s")

    # --- excitation, bends, periodic block, monitors ---

    def set_excitation(self, input_face='lo', input_mode=1, input_power=1.0, n_port_modes=4):
        """Field of `input_power` [W/m in 2D, W in 3D] in port mode `input_mode` (1-based) entering at `input_face`;
        n_port_modes port modes per face in the scalars and the port table."""
        if input_face not in PORT_FACES:
            raise ValueError(f'input_face must be one of {PORT_FACES}, got {input_face!r}')
        if int(input_mode) != input_mode or input_mode < 1 or int(n_port_modes) != n_port_modes or n_port_modes < 1:
            raise ValueError(f'input_mode and n_port_modes must be integers >= 1, got {input_mode!r}, {n_port_modes!r}')
        input_power = float(input_power)
        if not (math.isfinite(input_power) and input_power > 0.0):
            raise ValueError(f'input_power must be finite and > 0, got {input_power!r}')
        self.excitation = (input_face, int(input_mode), input_power)
        self.n_port_modes = int(n_port_modes)
        self._dirty = True

    def add_ports(self, input_face='lo', input_mode=1, input_power=1.0, n_port_modes=4):
        """set_excitation plus a monitor of the port S-matrix (ports_result)."""
        self.set_excitation(input_face, input_mode, input_power, n_port_modes)
        return self._add_monitor(EMEPortsMonitor())

    def add_section_table(self):
        """Monitor of the per-section table (section_table_result)."""
        return self._add_monitor(EMESectionTableMonitor())

    def add_length_sweep(self, start, stop, stretch_factors):
        """Monitor re-cascading the sections whose centre lies in [start, stop] [m from the window's lo edge along the
        propagation axis] with their lengths times each stretch factor (length_sweep_result); the next solve keeps
        the interfaces."""
        monitor = EMELengthSweepMonitor(start, stop, stretch_factors)
        self.monitors.append(monitor)
        self._dirty = True   # a solve before did not keep its interfaces
        return monitor

    def add_bend(self, start, stop, radius, reference):
        """Slices whose centre lies in [start, stop] [m from the window's lo edge along the axis] form a bend of
        `radius` [m] about the transverse coordinate `reference` [m] (2D: the line axis; 3D: x); ranges never overlap."""
        start, stop, radius, reference = float(start), float(stop), float(radius), float(reference)
        if not all(math.isfinite(value) for value in (start, stop, radius, reference)) or not stop > start:
            raise ValueError(f'bend: needs finite start < stop, radius and reference [m], got {start!r}, {stop!r}, '
                             f'{radius!r}, {reference!r}')
        if radius == 0.0:
            raise ValueError('bend radius must be non-zero')
        for other in self.bends:
            if start <= other[1] and other[0] <= stop:
                raise ValueError(f'bend range [{start:.6g}, {stop:.6g}] m overlaps [{other[0]:.6g}, {other[1]:.6g}] m')
        self.bends.append((start, stop, radius, reference))
        self._mode_cache, self._cache_omega = {}, None   # the label rows do not encode the radius
        self._dirty = True

    def add_periodic(self, start, stop, n_periods, n_bloch_modes=4):
        """The sections whose centre lies in [start, stop] [m] are a unit cell repeated n_periods times (drawn once),
        plus a monitor of its Bloch modes (periodic_result); at most one per physics."""
        if self.periodic is not None:
            raise ValueError('only one periodic block per EME physics')
        self.periodic = EMEPeriodicMonitor(start, stop, n_periods, n_bloch_modes)
        self.monitors.append(self.periodic)
        self._dirty = True
        return self.periodic

    def _add_monitor(self, monitor):
        self.monitors.append(monitor)
        if self.solved_omega is not None:
            monitor.record(self)
        return monitor

    # --- adjoint sensitivity (adjoint.py) ---

    def add_objective(self, quantity, mode_number=1, weight=1.0):
        """Objective term weight * quantity, a port power fraction of the excitation (adjoint.EME_OBJECTIVE_QUANTITIES;
        mode_number of the _mode ones); J sums every term. The next step re-solves (keeping the cascade)."""
        self._dirty = True
        return self._add_monitor(EMEObjective(quantity, mode_number, weight))

    def eps_sensitivity(self):
        """em.adjoint.EpsSensitivity (J [1], dJ/d eps per cell of the permittivity arrays the solve reads) of the
        objectives at the last solve (cached until the next solve); zeros without objectives or before a solve."""
        objectives = tuple(monitor for monitor in self.monitors if isinstance(monitor, EMEObjective))
        if not objectives or self.solved_omega is None or self.cascade is None:
            return EpsSensitivity(0.0, None, {}, self.domain.xp.zeros(self.q_abs.shape, dtype=DTYPE))
        key = (self.n_solves, objectives)
        if self._sensitivity is None or self._sensitivity[0] != key:
            self._sensitivity = None   # free the previous gradient first
            self._sensitivity = (key, self._cascade_sensitivity(objectives))
        return self._sensitivity[1]

    def _cascade_sensitivity(self, objectives):
        """EpsSensitivity of the last solve by the cascade adjoint."""
        raise NotImplementedError

    def _bend_index(self, position):
        """Index of the bend whose range contains `position` [m from the window's lo edge], or None."""
        for index, (start, stop, _, _) in enumerate(self.bends):
            if start <= position <= stop:
                return index
        return None

    def _bend_labels(self, positions):
        """(n_bends, n) float64 rows: 1 where a slice centre lies in bend k's range (NaN positions never do)."""
        positions = np.asarray(positions, dtype=np.float64)
        rows = np.zeros((len(self.bends), positions.size))
        for index, (start, stop, _, _) in enumerate(self.bends):
            rows[index] = (positions >= start) & (positions <= stop)
        return rows

    def _periodic_range(self, sections):
        """(first, last) section indices of the periodic block, None without one; ValueError when no section centre
        lies in its range or, with n_periods > 1, no section follows it."""
        if self.periodic is None:
            return None
        start, stop = self.periodic.start, self.periodic.stop
        inside = [index for index, section in enumerate(sections)
                  if start <= section.start + 0.5 * section.length <= stop]
        if not inside:
            raise ValueError(f'periodic block [{start:.6g}, {stop:.6g}] m holds no section centre')
        if self.periodic.n_periods > 1 and inside[-1] == len(sections) - 1:
            raise ValueError('periodic block: a section must follow the unit cell (draw an output guide after it)')
        return inside[0], inside[-1]

    def assemble_total(self, phases):
        """Total SMatrix of the kept interfaces (sweep_data) with section propagation factors `phases`, the periodic
        link rebuilt from them (length-sweep re-cascade)."""
        _, interfaces = self.sweep_data
        linked = list(interfaces)
        if self._link is not None:
            first, last, n_periods, junction = self._link
            linked[last] = repeated_link(unit_cell(phases, interfaces, first, last, junction), n_periods,
                                         interfaces[last])
        return total_smatrix(phases, linked)

    # --- solve ---

    def _find_sections(self):
        raise NotImplementedError

    def _solve_section(self, section, omega, previous_entry):
        """Solve entry (tuple, last item the ModeSet) of a section at omega; previous_entry = its entry of the last
        solve at another omega (start vectors), or None."""
        raise NotImplementedError

    def _reconstruct(self, cascade, omega, previous_cache, cache, counts):
        raise NotImplementedError

    def _entry_bytes(self, entry):
        """Host bytes a cached solve entry holds."""
        return entry[-1].nbytes

    def _section_modes(self, section, omega, previous_cache, cache, counts, tracking=None, for_field=False):
        """Solve entry of a section: from this solve's cache, the previous one (same omega), else _solve_section
        started from `tracking` (the last solve's entries at another omega)."""
        entry = cache.get(section.key)
        if entry is None and section.key in previous_cache:
            entry = previous_cache[section.key]
            self._keep_entry(cache, section.key, entry, counts)
        if entry is not None:
            if not for_field:
                counts['cache_hits'] += 1
            return entry
        previous_entry = tracking.get(section.key) if self.tracks_modes and tracking else None
        entry = self._solve_section(section, omega, previous_entry)
        counts['mode_resolves' if for_field else 'mode_solves'] += 1
        counts['tracked_solves'] += int(previous_entry is not None)
        self._keep_entry(cache, section.key, entry, counts)
        return entry

    def _keep_entry(self, cache, key, entry, counts):
        """Cache a solved or carried-over entry while this solve's cache stays within MODE_CACHE_BYTES."""
        size = self._entry_bytes(entry)
        if counts['cache_bytes'] + size <= MODE_CACHE_BYTES:
            cache[key] = entry
            counts['cache_bytes'] += size

    def _solve(self):
        """Sections, mode sets, interfaces and cascade at omega (periodic link included); port S-matrix, scalars,
        field, monitors."""
        start = time.perf_counter()
        omega = self._omega
        label = wavelength_label(2.0 * math.pi * C0 / omega)
        same_omega = self._cache_omega == omega
        previous_cache = self._mode_cache if same_omega else {}
        tracking = {} if same_omega else self._mode_cache
        cache = {}
        counts = {'sections': 0, 'mode_solves': 0, 'cache_hits': 0, 'mode_resolves': 0, 'tracked_solves': 0,
                  'cache_bytes': 0, 'max_modes': 0}
        sections = self._find_sections()
        block = self._periodic_range(sections)
        builder = CascadeBuilder()
        keep_all = any(isinstance(monitor, EMELengthSweepMonitor) for monitor in self.monitors)
        phases, wavenumbers, kept, link = [], [], {}, None
        rows, worst_reciprocity, previous, first_modes, cell_first_modes = [], (0.0, None), None, None, None
        self.periodic_data = None
        # (id previous, id modes) -> (previous, modes, SMatrix, defects): a repeated section pair (grating, array) reuses
        # its interface (omega and materials are fixed within one solve; the mode cache shares ModeSets by Section.key)
        interfaces = {}
        for index, section in enumerate(sections):
            modes = self._section_modes(section, omega, previous_cache, cache, counts, tracking)[-1]
            wavenumbers.append(modes.k)
            if previous is not None:
                pair = interfaces.get((id(previous), id(modes)))
                if pair is None or pair[0] is not previous or pair[1] is not modes:
                    smatrix = interface_smatrix(previous, modes)
                    pair = (previous, modes, smatrix, interface_defects(smatrix, previous, modes))
                    interfaces[(id(previous), id(modes))] = pair
                smatrix, (power_defect, reciprocity) = pair[2], pair[3]
                if keep_all or (block is not None and block[0] <= index - 1 <= block[1]):
                    kept[index - 1] = smatrix
                if block is not None and index - 1 == block[1] and self.periodic.n_periods > 1:
                    # junction and cell of the unit cell were built by _periodic_record at index block[1]
                    link = (block[0], block[1], self.periodic.n_periods, self.periodic_data['junction'])
                    builder.add_interface(repeated_link(self.periodic_data['cell'], link[2], smatrix))
                else:
                    builder.add_interface(smatrix)
                rows[-1][-2:] = [power_defect, reciprocity]
                if reciprocity > worst_reciprocity[0]:
                    worst_reciprocity = (reciprocity, index)
            if block is not None and index == block[0]:
                cell_first_modes = modes
            phase = np.exp(-1j * modes.k * section.length)
            phases.append(phase)
            builder.add_section(phase)
            n_eff = [complex(value).real for value in modes.n_eff[:2]] + [math.nan] * max(0, 2 - modes.n_modes)
            rows.append([index + 1, section.start, section.length, section.n_slices, modes.n_modes, *n_eff,
                         math.nan, math.nan])
            counts['max_modes'] = max(counts['max_modes'], modes.n_modes)
            first_modes = modes if index == 0 else first_modes
            if block is not None and index == block[1]:
                self.periodic_data = _periodic_record(sections, phases, wavenumbers, kept, block, modes,
                                                      cell_first_modes)
            previous = modes
        cascade = builder.finish()
        counts['sections'] = len(sections)
        self.sections = sections
        keep_cascade = any(isinstance(monitor, EMEObjective) for monitor in self.monitors)
        self.cascade = cascade if keep_cascade else None
        self.periodic_adjoint = None
        if keep_cascade and link is not None:
            self.periodic_adjoint = ((link[0], link[1]), link[2],
                                     cell_cascade(cascade.phases, kept, link[0], link[1], link[3]), kept[link[1]])
        self._link = link
        self.sweep_data = (wavenumbers, [kept[k] for k in range(len(sections) - 1)]) if keep_all else None
        self.port_modes = {'lo': first_modes, 'hi': previous}
        self.smatrix = cascade.total
        self.section_rows = rows
        self.solved_omega = omega
        if worst_reciprocity[0] > _RECIPROCITY_WARN:
            self.runtime_warnings.append(
                f'{label}: the interface before section {worst_reciprocity[1] + 1} breaks reciprocity by '
                f'{worst_reciprocity[0]:.2g} (|S12 - S21^T| / |S21|): its mode basis is truncated; raise n_modes '
                f'until the port results converge (with cpml faces radiation needs the PML modes)')
        self._scalars = self._port_scalars(label)
        self._reconstruct(cascade, omega, previous_cache, cache, counts)
        self._mode_cache, self._cache_omega = cache, omega
        self._dirty = False
        self.n_solves += 1
        for monitor in self.monitors:
            monitor.record(self)
        self.last_solve_seconds = time.perf_counter() - start
        self._last_counts = counts
        for name in ('sections', 'mode_solves', 'cache_hits', 'mode_resolves', 'tracked_solves'):
            self._counters[name] += counts[name]
        self._counters['interfaces'] += max(len(sections) - 1, 0)
        self._counters['solves'] += 1
        self._counters['solve_seconds'] += self.last_solve_seconds

    def input_amplitude(self):
        """(incident amplitudes of the input face's port modes, unit-amplitude power of the input mode, or None when
        the input face has fewer modes)."""
        face, number, power = self.excitation
        modes = self.port_modes[face]
        incident = np.zeros(modes.n_modes, dtype=np.complex128)
        if number > modes.n_modes:
            return incident, None
        unit_power = float(modes.power[number - 1])
        incident[number - 1] = math.sqrt(power / unit_power) if unit_power > INPUT_POWER_FLOOR else 1.0
        return incident, unit_power

    def port_blocks(self, smatrix=None):
        """(reflection block, transmission block, input ModeSet, output ModeSet) of the excitation's input face in
        smatrix (default: the last solve's)."""
        smatrix = self.smatrix if smatrix is None else smatrix
        if self.excitation[0] == 'lo':
            return smatrix.s11, smatrix.s21, self.port_modes['lo'], self.port_modes['hi']
        return smatrix.s22, smatrix.s12, self.port_modes['hi'], self.port_modes['lo']

    def port_fractions(self, smatrix=None):
        """{transmission_<k> (k <= n_port_modes, modes found), transmission, reflection} power fractions of the
        excitation in smatrix (default: the last solve's); {} when the input mode is missing or carries no power."""
        _, number, _ = self.excitation
        reflection, transmission, input_modes, output_modes = self.port_blocks(smatrix)
        _, unit_power = self.input_amplitude()
        if unit_power is None or unit_power <= INPUT_POWER_FLOOR:
            return {}
        column = number - 1
        transmitted = np.abs(transmission[:, column]) ** 2 * output_modes.power / unit_power
        reflected = np.abs(reflection[:, column]) ** 2 * input_modes.power / unit_power
        fractions = {f'transmission_{k + 1}': float(transmitted[k])
                     for k in range(min(self.n_port_modes, transmitted.size))}
        fractions['transmission'] = float(np.sum(transmitted))
        fractions['reflection'] = float(np.sum(reflected))
        return fractions

    def _port_scalars(self, label):
        """port_fractions of the last solve; a runtime warning when there are none."""
        fractions = self.port_fractions()
        if not fractions:
            _, unit_power = self.input_amplitude()
            number = self.excitation[1]
            reason = ('has fewer modes' if unit_power is None else f'mode {number} carries no power (evanescent)')
            self.runtime_warnings.append(f'{label}: the {self.excitation[0]} port {reason}: no port power fractions; '
                                         f'check input_mode (field shown at unit amplitude)')
        return fractions


def _periodic_record(sections, phases, wavenumbers, kept, block, last_modes, first_modes):
    """periodic_data of a unit cell: its SMatrix (hi end of the last cell section to the next period's) and junction
    (last -> first cell section, reused by the periodic link), the port basis
    (last cell section's modes), period [m], length-weighted mean Re k of mode 1 [1/m], 1-based section range."""
    first, last = block
    junction = interface_smatrix(last_modes, first_modes)
    cell = unit_cell(phases, kept, first, last, junction)
    period = float(sum(sections[k].length for k in range(first, last + 1)))
    reference = sum(float(wavenumbers[k][0].real) * sections[k].length for k in range(first, last + 1)) / period
    return {'cell': cell, 'junction': junction, 'modes': last_modes, 'period': period,
            'reference_k': reference, 'sections': (first + 1, last + 1)}


class EMEPhysics(EMEBase):
    """2D eigenmode expansion at omega: n_modes slab modes per section (0 = every active line cell), S-matrix cascade,
    the excitation's field over the window; ports = the lo / hi window planes along the propagation axis."""

    def __init__(self, domain, polarization='ez', propagation_axis='x', boundary_config=None, wavelength=1.55e-6,
                 n_modes=20, section_tolerance=0.0, discrete_spacing=None):
        if polarization not in EME_POLARIZATIONS:
            raise ValueError(f"EMEPhysics polarization must be 'ez' or 'hz', got {polarization!r}")
        if propagation_axis not in PROPAGATION_AXES:
            raise ValueError(f"EMEPhysics propagation_axis must be 'x' or 'y', got {propagation_axis!r}")
        if int(n_modes) != n_modes or int(n_modes) < 0:
            raise ValueError(f'EMEPhysics n_modes must be an integer >= 0 (0 = every active cell), got {n_modes!r}')
        super().__init__(domain, wavelength, n_modes, section_tolerance, _READ_ARRAYS[polarization], 'EMEPhysics')
        self.polarization = polarization
        self.propagation_axis = propagation_axis
        self.line_axis = 'y' if propagation_axis == 'x' else 'x'
        self._axis_index = PROPAGATION_AXES.index(propagation_axis)
        bc = solver_boundary_config(domain, boundary_config, 'EMEPhysics')
        check_zero_bloch(bc, 'EMEPhysics')
        for side in ('lo', 'hi'):
            face_type = getattr(bc, f'{propagation_axis}_{side}')
            if face_type == BCType.PERIODIC or face_type in MIRROR_TYPES:
                raise ValueError(f'EMEPhysics: the {propagation_axis}_{side} face along the propagation axis is a '
                                 f'semi-infinite port; {face_type.value} is not supported there')
        self.boundary_config = bc
        self.discrete_spacing = None if discrete_spacing is None else float(discrete_spacing)
        if self.discrete_spacing is not None:
            widths = domain.cell_widths()[self._axis_index][domain.physical_window()[self._axis_index]]
            if not np.allclose(widths, self.discrete_spacing, rtol=1e-9, atol=0.0):
                raise ValueError(f'EMEPhysics discrete_spacing {self.discrete_spacing!r} m needs uniform window cells '
                                 f'of that width along {propagation_axis}')
        register_em_arrays(domain.materials, 'modes')
        shape = (domain.grid.nx, domain.grid.ny)
        # float32 full-grid fields and q_abs [W/m^3] (domain.xp) of the last solve, rewritten in place (zeros before
        # the first)
        xp = domain.xp
        self.field_arrays = {name: xp.zeros(shape, dtype=DTYPE) for name in ('re', 'im', 'e_intensity', 'h_abs')}
        self.q_abs = xp.zeros(shape, dtype=DTYPE)
        self._zeros = xp.zeros(shape, dtype=DTYPE)
        self.design_regions = []   # design_density.DesignRegion writing _design_arrays (optimization study)

    def state(self) -> dict:
        """Float32 full-grid ez_re / _im, hz_re / _im (the other polarisation reads zeros), e_intensity, q_abs
        [W/m^3], the permittivity arrays, sigma_e and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        state = {}
        for polarization in EME_POLARIZATIONS:
            own = polarization == self.polarization
            state[f'{polarization}_re'] = self.field_arrays['re'] if own else self._zeros
            state[f'{polarization}_im'] = self.field_arrays['im'] if own else self._zeros
        state['e_intensity'] = self.field_arrays['e_intensity']
        state['q_abs'] = self.q_abs
        for name in EME_PERMITTIVITY_ARRAYS:
            state[name] = getattr(materials, name)
        state['sigma_e'] = materials.sigma_e
        state['q_accum'] = None
        return state

    def renderer(self):
        return EMERenderer(self)

    # --- material helpers ---

    def add_dispersive(self, pole, mask):
        """Record a Drude / Lorentz pole on a run-backend bool cell mask (None = everywhere); its chi(omega) joins every
        solve."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        if mask is not None:
            _check_cell_array(mask, self._zeros.shape, np.bool_, 'Pole mask', xp=self.domain.xp)
        record_dispersive_region(self.dispersive_regions, pole, mask)
        self._dirty = True

    def add_pec_region(self, mask):
        """Perfect electric conductor on a run-backend bool (nx, ny) cell mask."""
        _check_cell_array(mask, self._zeros.shape, np.bool_, 'PEC mask', xp=self.domain.xp)
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self._dirty = True

    # --- adjoint sensitivity and design regions (adjoint.py, optimization study) ---

    @property
    def _design_arrays(self) -> tuple:
        """Permittivity arrays a design region writes: ez eps_r; hz eps_xx and eps_yy (an isotropic change)."""
        return ('eps_r',) if self.polarization == 'ez' else ('eps_xx', 'eps_yy')

    def add_design_region(self, mask, eps_min, eps_max, initial_density=0.5, filter_radius=0.0, projection_beta=0.0,
                          projection_eta=0.5):
        """Density design region (_design_arrays) on the window cells of a run-backend bool (nx, ny) mask; ValueError
        when it touches the first or last window slice (the ports) or overlaps an earlier region."""
        _check_cell_array(mask, self._zeros.shape, np.bool_, 'Design region mask', xp=self.domain.xp)
        region = DesignRegion(self.domain, mask, self._design_arrays, eps_min, eps_max, initial_density, filter_radius,
                              projection_beta, projection_eta, objective_unit=self.objective_unit)
        window = self.domain.physical_window()[self._axis_index]
        along = region.box[self._axis_index]
        if along.start <= window.start or along.stop >= window.stop:
            raise ValueError(f'em_eme design region touches the first or last cell layer along {self.propagation_axis}'
                             ' (the port sections, held fixed by the adjoint): keep it inside the window')
        if any(region.overlaps(other) for other in self.design_regions):
            raise ValueError('design regions of one physics must not share cells')
        self.design_regions.append(region)
        region.write()
        self.materials_changed(self._design_arrays)
        return region

    def design_density(self):
        """float32 (nx, ny) projected density rho_bar of every design region, 0 elsewhere (fresh array)."""
        return paint_regions(self.design_regions, self._zeros.shape, self.domain.xp)

    def _cascade_sensitivity(self, objectives):
        """EpsSensitivity (ez eps_r, hz eps_xx and eps_yy) of the last solve by the cascade adjoint (section_gradient on
        the slices of the design regions, every window slice without one; the cell of a periodic block sums its
        periods); NotImplementedError for the discrete variant."""
        if self.discrete_spacing is not None:
            raise NotImplementedError('em_eme adjoint sensitivity: not with the discrete propagation variant')
        omega = self.solved_omega
        face, number, _ = self.excitation
        _, unit_power = self.input_amplitude()
        if unit_power is None or unit_power <= INPUT_POWER_FLOOR:
            raise ValueError(f'em_eme adjoint sensitivity: input mode {number} of the {face} port is missing or '
                             'carries no power')
        total, cotangents = objective_cotangents(objectives, face, self.port_blocks(), number - 1, unit_power)
        # (cells array, faces array or None): ez eps_r of the line cells; hz eps_t of the cells, eps_l of the faces
        cell_name, face_name = (('eps_r', None) if self.polarization == 'ez'
                                else _HZ_LINE_ARRAYS[self.propagation_axis])
        host = {name: np.zeros(self._zeros.shape) for name in (cell_name, face_name) if name is not None}
        if cotangents:
            self._add_section_gradients(host, cell_name, face_name, omega, face, number, cotangents)
        gradients = {name: self.domain.xp.asarray(values) for name, values in host.items()}
        return EpsSensitivity(total, float(omega), gradients, sensitivity_density(self.domain, gradients))

    def _add_section_gradients(self, host, cell_name, face_name, omega, face, number, cotangents):
        """Add dJ/d eps of every needed section's slices into the host (nx, ny) arrays host[cell_name] (line cells) and
        host[face_name] (line faces mapped to their cells; hz only)."""
        incident = np.zeros(self.port_modes[face].n_modes, dtype=np.complex128)
        incident[number - 1] = 1.0
        periodic = None
        if self.periodic_adjoint is not None:
            block, n_periods, cell, exit_interface = self.periodic_adjoint
            periodic = PeriodicAdjointFields(self.cascade, block, cell, exit_interface, n_periods, face, incident,
                                             cotangents)
        widths = self.domain.cell_widths()[self._axis_index]
        line_index = 1 - self._axis_index
        along_widths = grid_cell_widths(self.domain.grid)[line_index]   # PortLine's face-mean widths
        line_periodic = (self.boundary_config.periodic_x, self.boundary_config.periodic_y)[line_index]
        needed = self._sensitivity_slices()
        counts = {'cache_hits': 0, 'mode_resolves': 0, 'tracked_solves': 0, 'cache_bytes': MODE_CACHE_BYTES}
        for index, section in enumerate(self.sections):
            slices = slice(section.first_slice, section.first_slice + section.n_slices)
            if not needed[slices].any():
                continue
            port, line, face_scale, modes = self._section_modes(section, omega, self._mode_cache, {}, counts,
                                                                for_field=True)
            cell_rows, face_rows = section_gradient(self.cascade, index, line, modes, widths[slices], section.length,
                                                    omega, face, incident, cotangents,
                                                    self._centre_scale(section, port), face_scale, periodic)
            cell_rows[~needed[slices]] = 0.0
            self._place_rows(host[cell_name], slices, slice(*port.solve_range), cell_rows)
            if face_name is not None:
                a, b = port.solve_range   # local face f = hi face of grid cell a - 1 + f (PortLine._face_index)
                grid_faces = np.arange(a - 1, b)
                face_index = grid_faces % along_widths.size if line_periodic else np.clip(grid_faces, 0,
                                                                                          along_widths.size - 1)
                face_rows[~needed[slices]] = 0.0
                self._place_rows(host[face_name], slices, slice(None),
                                 face_gradient_to_cells(face_rows, face_index, along_widths, line_periodic))
        self._counters['mode_resolves'] += counts['mode_resolves']

    def _place_rows(self, array, slices, cells, rows):
        """array[slices, cells] += rows (propagation x) or array[cells, slices] += rows.T (y)."""
        if self.propagation_axis == 'x':
            array[slices, cells] += rows
        else:
            array[cells, slices] += rows.T

    def _sensitivity_slices(self):
        """(n,) bool along the propagation axis: the window slices of the design region boxes (every window slice
        without a region)."""
        axis = self._axis_index
        needed = np.zeros(self._zeros.shape[axis], dtype=bool)
        if not self.design_regions:
            needed[self.domain.physical_window()[axis]] = True
        for region in self.design_regions:
            needed[region.box[axis]] = True
        return needed

    def _centre_scale(self, section, port):
        """(m,) r / R at the line cell centres of a bent section (mode_sets.bent_line), None when straight."""
        bend = self._bend_index(self._slice_positions()[section.first_slice])
        if bend is None:
            return None
        _, _, radius, reference = self.bends[bend]
        a, b = port.solve_range
        return 1.0 + (self.domain.cell_centers()[1 - self._axis_index][a:b] - reference) / radius

    def _pole_region(self, box, mask):
        """Host float64 (nx, ny) region of a pole: 1 inside."""
        shape = self._zeros.shape
        if box is None:
            return np.ones(shape)
        region = np.zeros(shape)
        region[box] = to_host(mask)
        return region

    def _loss(self, omega):
        """Host float64 (nx, ny) -Im eps_c(omega) = sigma_e / (omega eps0) - sum Im chi_pole on its region; 0 on PEC
        cells (as 3D _layer_loss: a lossy material painted under a PEC region absorbs nothing)."""
        loss = to_host(self.domain.materials.sigma_e).astype(np.float64) / (float(omega) * EPS0)
        for pole, box, mask in self.dispersive_regions:
            loss -= complex(pole.susceptibility(float(omega))).imag * self._pole_region(box, mask)
        if self.pec_cells is not None:
            loss[to_host(self.pec_cells).astype(bool)] = 0.0
        return loss

    # --- solve ---

    def _slice_positions(self):
        axis = self._axis_index
        return slice_positions(self.domain.cell_widths()[axis], self.domain.physical_window()[axis])

    def _host_stack(self):
        """slice_stack of the arrays a solve reads, the pole regions, the PEC cells and the bend labels (host)."""
        materials = self.domain.materials
        shape = self._zeros.shape
        arrays = [to_host(getattr(materials, name)) for name in self._read_arrays]
        arrays += [self._pole_region(box, mask) for _pole, box, mask in self.dispersive_regions]
        if self.pec_cells is not None:
            arrays.append(to_host(self.pec_cells))
        for labels in self._bend_labels(self._slice_positions()):
            arrays.append(np.broadcast_to(labels[:, None] if self._axis_index == 0 else labels[None, :], shape))
        return slice_stack(arrays, self.propagation_axis)

    def _find_sections(self):
        widths = self.domain.cell_widths()[self._axis_index]
        return stack_sections(self._host_stack(), widths, self.domain.physical_window()[self._axis_index],
                              self.section_tolerance)

    def _solve_section(self, section, omega, previous_entry):
        """(PortLine, SlabLine, bend face scale or None, ModeSet) of the section's first slice (direct line solves:
        previous_entry is always None)."""
        window = self.domain.physical_window()[1 - self._axis_index]
        port = PortLine(self, self.line_axis, section.first_slice, (window.start, window.stop),
                        f'EME section at {self.propagation_axis} cell {section.first_slice}')
        port.read_materials(self)
        line = port.slab_line(omega)
        face_scale = None
        bend = self._bend_index(self._slice_positions()[section.first_slice])
        if bend is not None:
            _, _, radius, reference = self.bends[bend]
            line_index = 1 - self._axis_index
            a, b = port.solve_range
            line, face_scale = bent_line(line, radius, self.domain.cell_centers()[line_index][a:b],
                                         self.domain.cell_edges()[line_index][a:b + 1], reference)
        n_modes = self.n_modes or int(np.count_nonzero(line.active))
        modes = solve_slab_modes(line, omega, n_modes, normal_spacing=self.discrete_spacing, drop_pml_modes=False)
        return port, line, face_scale, mode_set(line, modes, self.discrete_spacing)

    def _reconstruct(self, cascade, omega, previous_cache, cache, counts):
        """Field of the excitation in every section (cell centres) into the float32 field arrays; q_abs from the
        3-point Gauss cell mean of |E|^2 along the axis (the cell integral of the continuum propagation)."""
        face = self.excitation[0]
        incident, _ = self.input_amplitude()
        widths = self.domain.cell_widths()[self._axis_index]
        shape = self._zeros.shape
        out_of_plane = np.zeros(shape, dtype=np.complex128)
        in_plane = np.zeros(shape, dtype=np.float64)
        e_mean = np.zeros(shape, dtype=np.float64)
        solve_range = None
        for index, section in enumerate(self.sections):
            port, line, face_scale, modes = self._section_modes(section, omega, previous_cache, cache, counts,
                                                                for_field=True)
            forward, backward = cascade.section_amplitudes(index, face, incident)
            n_slices = section.n_slices
            positions, _ = gauss_positions(widths[section.first_slice:section.first_slice + n_slices])
            u_points, in_plane_points = section_fields(line, modes, forward, backward, positions, section.length,
                                                       omega, face_scale)
            e_points = np.abs(u_points) ** 2 if self.polarization == 'ez' else in_plane_points
            values = (u_points[n_slices:2 * n_slices], in_plane_points[n_slices:2 * n_slices], gauss_mean(e_points))
            slices = slice(section.first_slice, section.first_slice + n_slices)
            cells = slice(*port.solve_range)
            index_tuple = (slices, cells) if self.propagation_axis == 'x' else (cells, slices)
            for array, value in zip((out_of_plane, in_plane, e_mean), values):
                array[index_tuple] = value if self.propagation_axis == 'x' else value.T
            solve_range = port.solve_range
        self._fill_wall_cells((out_of_plane,), (in_plane, e_mean), solve_range)
        if self.polarization == 'ez':
            e_intensity, h_abs = np.abs(out_of_plane) ** 2, np.sqrt(in_plane)
        else:
            e_intensity, h_abs = in_plane, np.abs(out_of_plane)
        xp = self.domain.xp   # one host-to-device copy per array on the gpu
        for name, values in (('re', out_of_plane.real), ('im', out_of_plane.imag), ('e_intensity', e_intensity),
                             ('h_abs', h_abs)):
            self.field_arrays[name][...] = xp.asarray(values, dtype=DTYPE)
        loss = self._loss(omega)
        if np.any(loss != 0.0):
            self.q_abs[...] = xp.asarray(0.5 * float(omega) * EPS0 * loss * e_mean, dtype=DTYPE)
        else:
            self.q_abs.fill(0)

    def _fill_wall_cells(self, signed, unsigned, solve_range):
        """Wall cells beside the line solve range: sign x inner neighbour (signed arrays), |sign| x (unsigned)."""
        n_line = signed[0].shape[1 - self._axis_index]
        for side, outer, inner in (('lo', 0, 1), ('hi', n_line - 1, n_line - 2)):
            inside = solve_range[0] <= outer < solve_range[1]
            if inside or not 0 <= inner < n_line:
                continue
            sign = WALL_SIGNS[self.polarization].get(getattr(self.boundary_config, f'{self.line_axis}_{side}'), 0.0)
            for array, factor in [(array, sign) for array in signed] + [(array, abs(sign)) for array in unsigned]:
                if self.propagation_axis == 'x':
                    array[:, outer] = factor * array[:, inner]
                else:
                    array[outer, :] = factor * array[inner, :]


class EMEPortsMonitor:
    """Port S-matrix of the last solve, restricted to n_port_modes per face, and the excitation."""

    def __init__(self):
        self.record_data = None

    def record(self, physics):
        face, number, power = physics.excitation
        reflection, transmission, input_modes, output_modes = physics.port_blocks()
        _, unit_power = physics.input_amplitude()
        n_lo = min(physics.n_port_modes, physics.port_modes['lo'].n_modes)
        n_hi = min(physics.n_port_modes, physics.port_modes['hi'].n_modes)
        s = physics.smatrix
        port_matrix = np.block([[s.s11[:n_lo, :n_lo], s.s12[:n_lo, :n_hi]], [s.s21[:n_hi, :n_lo], s.s22[:n_hi, :n_hi]]])
        self.record_data = {'omega': physics.solved_omega, 'face': face, 'number': number, 'power': power,
                            'unit_power': unit_power, 'reflection': reflection, 'transmission': transmission,
                            'input_modes': input_modes, 'output_modes': output_modes,
                            'n_report': physics.n_port_modes, 'port_matrix': port_matrix, 'port_counts': (n_lo, n_hi)}


class EMESectionTableMonitor:
    """SECTION_TABLE_COLUMNS rows of the last solve."""

    def __init__(self):
        self.rows = None
        self.omega = None

    def record(self, physics):
        self.rows = [list(row) for row in physics.section_rows]
        self.omega = physics.solved_omega


class EMELengthSweepMonitor:
    """Port power fractions of the last solve with the sections whose centre lies in [start, stop] stretched by each
    factor (re-cascaded from the kept interfaces, periodic link rebuilt: no mode solve)."""

    def __init__(self, start, stop, stretch_factors):
        self.start, self.stop = float(start), float(stop)
        if not (math.isfinite(self.start) and math.isfinite(self.stop) and self.stop > self.start):
            raise ValueError(f'length sweep: needs finite start < stop [m], got {start!r}, {stop!r}')
        self.stretch_factors = [float(factor) for factor in stretch_factors]
        if not self.stretch_factors or not all(math.isfinite(factor) and factor > 0.0
                                               for factor in self.stretch_factors):
            raise ValueError(f'length sweep: stretch_factors must be a non-empty list of finite values > 0, got '
                             f'{stretch_factors!r}')
        self.record_data = None

    def record(self, physics):
        if physics.sweep_data is None:   # solved before the monitor was attached: the next solve records
            return
        wavenumbers, _ = physics.sweep_data
        inside = [self.start <= section.start + 0.5 * section.length <= self.stop for section in physics.sections]
        if not any(inside):
            label = wavelength_label(2.0 * math.pi * C0 / physics.solved_omega)
            physics.runtime_warnings.append(f'{label}: no section centre lies in the length sweep range '
                                            f'[{self.start:.6g}, {self.stop:.6g}] m: every stretch gives the same '
                                            f'result')
        base_length = sum(section.length for section, flag in zip(physics.sections, inside) if flag)
        names = [f'transmission_{number}' for number in range(1, physics.n_port_modes + 1)]
        series = {name: [] for name in (*names, 'transmission', 'reflection')}
        factors = np.asarray(self.stretch_factors, dtype=np.float64)
        # stretch factors re-cascaded together (stacked star products), chunked so a chunk's blocks stay small
        chunk = length_sweep_chunk(max(int(np.asarray(k).size) for k in wavenumbers))
        for begin in range(0, factors.size, chunk):
            stretch = factors[begin:begin + chunk]
            unstretched = np.ones_like(stretch)
            phases = [np.exp((-1j * k * section.length)[None, :] * (stretch if flag else unstretched)[:, None])
                      for k, section, flag in zip(wavenumbers, physics.sections, inside)]
            total = physics.assemble_total(phases)
            blocks = [np.broadcast_to(block, (stretch.size,) + block.shape[-2:])
                      for block in (total.s11, total.s12, total.s21, total.s22)]
            for index in range(stretch.size):
                fractions = physics.port_fractions(SMatrix(*(block[index] for block in blocks)))
                for name, values in series.items():
                    values.append(fractions.get(name, math.nan))
        self.record_data = {'omega': physics.solved_omega, 'lengths': [factor * base_length
                                                                        for factor in self.stretch_factors],
                            'series': series, 'n_stretched': int(sum(inside))}


class EMEPeriodicMonitor:
    """Unit cell of the periodic block (start, stop [m], n_periods) and the forward Bloch modes of the last solve."""

    def __init__(self, start, stop, n_periods, n_bloch_modes=4):
        self.start, self.stop = float(start), float(stop)
        if not (math.isfinite(self.start) and math.isfinite(self.stop) and self.stop > self.start):
            raise ValueError(f'periodic block: needs finite start < stop [m], got {start!r}, {stop!r}')
        if int(n_periods) != n_periods or n_periods < 1 or int(n_bloch_modes) != n_bloch_modes or n_bloch_modes < 1:
            raise ValueError(f'periodic block: n_periods and n_bloch_modes must be integers >= 1, got {n_periods!r}, '
                             f'{n_bloch_modes!r}')
        self.n_periods, self.n_bloch_modes = int(n_periods), int(n_bloch_modes)
        self.record_data = None

    def record(self, physics):
        data = physics.periodic_data
        if data is None:
            return
        multipliers, forward, backward = bloch_modes(data['cell'])
        flags = forward_bloch(multipliers, forward, backward, data['modes'].power)
        multipliers, forward, backward = multipliers[flags], forward[:, flags], backward[:, flags]
        order = np.argsort(-np.abs(multipliers), kind='stable')
        period, k0 = data['period'], physics.solved_omega / C0
        rows = []
        for number, column in enumerate(order[:self.n_bloch_modes]):
            value = complex(multipliers[column])
            phase = -math.atan2(value.imag, value.real)                    # Re K period in (-pi, pi]
            zone = round((data['reference_k'] * period - phase) / (2.0 * math.pi))
            weights = np.abs(forward[:, column]) ** 2 + np.abs(backward[:, column]) ** 2
            rows.append([number + 1, (phase + 2.0 * math.pi * zone) / (period * k0),
                         math.log(abs(value)) / (period * k0), abs(value), math.degrees(phase),
                         int(np.argmax(weights)) + 1])
        self.record_data = {'omega': physics.solved_omega, 'rows': rows, 'period': period,
                            'sections': data['sections'], 'multipliers': multipliers[order]}


def length_sweep_result(monitor) -> dict:
    """'xy' of an EMELengthSweepMonitor vs the stretched length [m] (factor order): series transmission_<k>,
    transmission, reflection; extras stretch_factors, wavelength, n_sections_stretched."""
    data = monitor.record_data
    if data is None:
        raise _not_solved('eme_length_sweep')
    return {'kind': 'xy', 'x': np.asarray(data['lengths'], dtype=np.float64), 'x_label': 'Stretched length',
            'x_unit': 'm', 'y_label': 'Power fraction',
            'series': {name: np.asarray(values, dtype=np.float64) for name, values in data['series'].items()},
            'series_units': {name: '1' for name in data['series']},
            'stretch_factors': list(monitor.stretch_factors),
            'wavelength': float(2.0 * math.pi * C0 / data['omega']), 'n_sections_stretched': data['n_stretched']}


def periodic_result(monitor) -> dict:
    """'table' of an EMEPeriodicMonitor (BLOCH_TABLE_COLUMNS, least attenuated first); extras wavelength, period [m],
    n_periods, first_section / last_section (1-based) of the unit cell."""
    data = monitor.record_data
    if data is None:
        raise _not_solved('eme_periodic')
    return {'kind': 'table', 'columns': list(BLOCH_TABLE_COLUMNS), 'units': list(BLOCH_TABLE_UNITS),
            'rows': data['rows'], 'wavelength': float(2.0 * math.pi * C0 / data['omega']), 'period': data['period'],
            'n_periods': monitor.n_periods, 'first_section': data['sections'][0], 'last_section': data['sections'][1]}


def _not_solved(what):
    return ValueError(f'not solved yet: the {what} has no solve (run a stationary or frequency_domain study)')


def ports_result(monitor) -> dict:
    """'table' of an EMEPortsMonitor: reflected (input face) and transmitted (other face) rows per port mode; extras
    wavelength, input_face, input_mode, transmission, reflection, port S-matrix (real / imag parts) and its mode counts."""
    data = monitor.record_data
    if data is None:
        raise _not_solved('eme_ports')
    column = data['number'] - 1
    unit_power = data['unit_power']
    has_power = unit_power is not None and unit_power > INPUT_POWER_FLOOR
    output_face = 'hi' if data['face'] == 'lo' else 'lo'
    rows, totals = [], {}
    for face, block, modes, total_name in ((data['face'], data['reflection'], data['input_modes'], 'reflection'),
                                           (output_face, data['transmission'], data['output_modes'], 'transmission')):
        values = (block[:, column] if unit_power is not None else np.full(modes.n_modes, np.nan + 0j))
        fractions = (np.abs(values) ** 2 * modes.power / unit_power if has_power
                     else np.full(modes.n_modes, np.nan))
        totals[total_name] = float(np.sum(fractions))
        for k in range(min(data['n_report'], modes.n_modes)):
            n_eff = complex(modes.n_eff[k])
            rows.append([face, k + 1, n_eff.real, n_eff.imag, float(abs(values[k]) ** 2),
                         float(np.degrees(np.angle(values[k]))), float(fractions[k])])
    return {'kind': 'table', 'columns': list(PORT_TABLE_COLUMNS), 'units': list(PORT_TABLE_UNITS), 'rows': rows,
            'wavelength': float(2.0 * math.pi * C0 / data['omega']), 'input_face': data['face'],
            'input_mode': data['number'], 'input_power': data['power'], **totals,
            's_matrix_real': data['port_matrix'].real.tolist(), 's_matrix_imag': data['port_matrix'].imag.tolist(),
            's_matrix_modes': {'lo': data['port_counts'][0], 'hi': data['port_counts'][1]}}


def section_table_result(monitor) -> dict:
    """'table' of an EMESectionTableMonitor (SECTION_TABLE_COLUMNS); extra wavelength."""
    if monitor.rows is None:
        raise _not_solved('eme_section_table')
    return {'kind': 'table', 'columns': list(SECTION_TABLE_COLUMNS), 'units': list(SECTION_TABLE_UNITS),
            'rows': monitor.rows, 'wavelength': float(2.0 * math.pi * C0 / monitor.omega)}

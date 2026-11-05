"""Cross-section eigenmodes on a Domain (xy = waveguide cross-section, propagation along z): ModePhysics solves the
full-vector modes on the host (cross_section_modes) and exposes one of them; table / dispersion monitors record every
solve."""
import math
import time

import numpy as np
from ....core.backend import to_host

from ....core.constants import C0, DTYPE, EPS0, MU0
from ....core.cut_cells import dual_cell_cuts, union_test
from ....core.error_indicators import energy_interpolation_indicator
from ....core.gpu_memory import format_bytes
from ....core.solver_base import PhysicsSolver
from ....core.wavelengths import wavelength_label
from ..boundaries import solver_boundary_config
from ..cross_section_modes import (BEND_MODELS, GROUP_RELATIVE_STEP, LINE_BLOCK_BOTH_PERIODIC_TEXT,
                                   CrossSectionMaterials, bloch_wrap_factors, centre_fields, line_block_layout,
                                   mode_group_indices, mode_overlap, mode_point_materials,
                                   prepare_cross_section_modes, size_issue_for, solve_prepared_modes)
from ..fdfd_line_block import MAX_BATCH_WAVELENGTHS
from ....solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver
from ....solvers.line_block import LineBlockSolver, RejectedLineBlockSolve, batch_factorizations
from ....solvers.sparse_direct import scipy_available
from ..harmonic import cell_permeability, cell_permittivity, write_float32
from ..materials import DrudePole, LorentzPole, register_em_arrays
from ..nonlinear_overlap import ShgOverlapMonitor
from ..renderer import ModeRenderer
from ..subpixel_smoothing import SubpixelSmoothing
from ..yee_grid import cpml_grading_cell_size, grid_cell_widths
from .fdtd_2d_base import _check_cell_array, add_pec_cells, record_dispersive_region

MODE_POLARIZATION = 'modes'   # EM_MATERIAL_SCHEMAS key; no in-plane polarisation (placement helpers refuse it)
MODE_PERMITTIVITY_ARRAYS = ('eps_r', 'eps_xx', 'eps_yy')   # eps_r holds eps_zz (as for the E_z types)
MODE_PERMITTIVITY_SLOTS = {f'{array}_update': array for array in MODE_PERMITTIVITY_ARRAYS}
MODE_FIELD_NAMES = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz', 'e_intensity', 'h_abs', 'sz')   # ModeRenderer's mode fields
MODE_TABLE_COLUMNS = ('mode_number', 'n_eff', 'n_eff_imag', 'loss_db_per_cm', 'n_group', 'te_fraction',
                      'effective_area', 'pml_fraction', 'selected')
MODE_TABLE_UNITS = ('1', '1', '1', 'dB/cm', '1', '1', 'm^2', '1', '')
_DB_PER_NEPER = 20.0 / math.log(10.0)   # power loss in dB of a field decaying by e^-1


def bend_factor_message(x_edges, radius, reference_x):
    """Problem text when the bend scale factor 1 + (x - reference_x) / radius is <= 0 at a grid edge x, else None."""
    if radius == 0.0:
        return None
    edge = float(np.min(x_edges)) if radius > 0.0 else float(np.max(x_edges))
    factor = 1.0 + (edge - reference_x) / radius
    if factor > 0.0:
        return None
    return (f'bend_radius {radius:.4g} m with reference x {reference_x:.4g} m maps the grid edge x = {edge:.4g} m '
            f'(PML included) to a scale factor 1 + (x - x_ref) / R = {factor:.3g} <= 0: the centre of curvature lies '
            f'inside the grid; use |bend_radius| > {abs(edge - reference_x):.4g} m or move the reference')


def _complex_text(value):
    """'2.441780' or '2.441780 - 1.2e-05j' of a complex n_eff."""
    text = f'{value.real:.6f}'
    if value.imag != 0.0:
        text += f" {'-' if value.imag < 0 else '+'} {abs(value.imag):.3g}j"
    return text


class _ModeStudyMixin:
    """Mode-study members of ModePhysics and FEMModePhysics: the solve_modes / solve_harmonic hooks, the post-solve
    bookkeeping (warnings, counters, selection, monitors, timings), n_eff texts, scalars and the table / dispersion
    monitors; a subclass gives _solve_once(omega) -> (modes, solve info, warnings) and select_mode(number)."""

    def _init_mode_study(self, wavelength, n_modes, mode_index, n_eff_guess, group_index):
        """Mode-study settings and the empty solve record."""
        self.omega = 2.0 * math.pi * C0 / _positive(wavelength, 'wavelength [m]')
        self.n_modes = _positive_int(n_modes, 'n_modes')
        self.mode_index = _positive_int(mode_index, 'mode_index')
        self.n_eff_guess = None if n_eff_guess is None else _positive(n_eff_guess, 'n_eff_guess')
        self.group_index = bool(group_index)
        self.modes = []               # modes of the last solve, descending Re beta^2
        self.selected_mode = None     # 1-based number shown (None: no mode)
        self.solved_omega = None
        self.last_solve_info = None   # solve info of the last solve
        self.host_factor_bytes = None   # SuperLU factor of the last solve (Simulation.memory_report)
        self.last_solve_seconds = None
        self.total_solve_seconds = 0.0
        self.n_solves = 0
        self.n_modes_found = 0        # summed over every solve (work_counters)

    def solve_modes(self) -> int:
        """Solve at the wavelength setting (mode_analysis study hook); the number of modes found."""
        self._solve()
        return len(self.modes)

    def solve_harmonic(self, omega):
        """Solve at `omega` [rad/s] (frequency_domain study hook) and select mode_index."""
        self.omega = float(omega)
        self._solve()

    def n_eff_text(self, number):
        """'n_eff = 2.441780' (complex when lossy) of mode `number` (1-based) of the last solve."""
        return f'n_eff = {_complex_text(complex(self.modes[number - 1].n_eff))}'

    def selection_label(self):
        """'mode 1: n_eff = ... (lambda = 1550 nm)' of the selected mode; None before a solve or without modes."""
        if self.selected_mode is None:
            return None
        return (f'mode {self.selected_mode}: {self.n_eff_text(self.selected_mode)} '
                f'({wavelength_label(2.0 * math.pi * C0 / self.solved_omega)})')

    def scalars(self) -> dict:
        """{'n_eff_k': Re n_eff of mode k} of the last solve, modes found only (E5 scalar outputs)."""
        return {f'n_eff_{number}': complex(mode.n_eff).real for number, mode in enumerate(self.modes, start=1)}

    def _solve(self):
        """Solve at self.omega on the current materials (group indices when enabled), select mode_index, record the
        monitors."""
        start = time.perf_counter()
        omega = self.omega
        modes, info, warnings = self._solve_once(omega)
        label = wavelength_label(2.0 * math.pi * C0 / omega)
        self.runtime_warnings.extend(f'{label}: {warning}' for warning in warnings)
        self.modes = modes
        self.solved_omega = omega
        self.last_solve_info = info
        self.host_factor_bytes = info['factor_host_bytes']
        self.n_solves += 1
        self.n_modes_found += len(modes)
        if self.mode_index > len(modes):
            shown = f'mode {len(modes)} is shown' if modes else 'nothing is shown'
            self.runtime_warnings.append(
                f'{label}: {len(modes)} mode(s) found, fewer than mode_index {self.mode_index}: {shown} (raise '
                f'n_modes, set n_eff_guess near the wanted mode, or widen the domain: modes mostly in the PML are '
                f'dropped)')
        self.select_mode(self.mode_index)
        for monitor in self.monitors:
            monitor.record(self)
        self.last_solve_seconds = time.perf_counter() - start
        self.total_solve_seconds += self.last_solve_seconds

    def add_mode_table(self):
        """Scalar data of every mode of the last solve (mode_table_result)."""
        return self._add_monitor(ModeTableMonitor())

    def add_mode_dispersion(self, mode_number=1):
        """Scalar data of mode `mode_number` per solved wavelength, followed by overlap (mode_dispersion_result)."""
        return self._add_monitor(ModeDispersionMonitor(mode_number))

    def _add_monitor(self, monitor):
        """Append a monitor; it records the last solve at once when there is one."""
        self.monitors.append(monitor)
        if self.solved_omega is not None:
            monitor.record(self)
        return monitor


class ModePhysics(_ModeStudyMixin, PhysicsSolver):
    """Full-vector cross-section modes at omega: n_modes modes (descending Re beta^2) of the diagonal-anisotropic eps
    and mu, solved by the study hooks solve_modes / solve_harmonic; the selected one feeds the renderer and state."""
    # Attribute set of FDFDPhysics (domain, polarization, boundary_config, sources, monitors, dispersive_regions,
    # pec_cells, finalize_warnings, permittivity_arrays, enable_q_accum, solved_omega, n_solves), so placement,
    # couplings and the Visualizer treat it alike. Every hook call re-solves (materials are read fresh); step never
    # solves. Monitors follow record(physics) after every solve.
    # A LineBlockSolver as shifted_solver factors the shift-invert operators instead of host SuperLU: the solve's own
    # operators (omega and, with group_index, omega (1 +- step)) and, in a frequency_domain study
    # (plan_harmonic_frequencies), those of the next planned wavelengths, up to MAX_BATCH_WAVELENGTHS operators per
    # factorize_operators batch; the assembled problems are kept with the batch (_material_serial guards it) and a
    # rejected solve redoes that eigen-solve on SuperLU (run-log warning). Detached solves (shg_overlap, bend_overlap)
    # take the eigen solver's own factor (SuperLU, or cuDSS LU with factorization cudss: no host size rule then).

    name = 'modes'

    def __init__(self, domain, boundary_config=None, wavelength=1.55e-6, n_modes=4, mode_index=1, n_eff_guess=None,
                 bend_radius=0.0, bend_reference_x=None, group_index=False, eigen_solver=None, bend_model='exact',
                 shifted_solver=None):
        self.domain = domain
        # the build's 'eigen' solver (ctx.solvers), else a private one whose counters work_counters reports
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        # the build's 'shifted' role: a LineBlockSolver factors the shift-invert operators, anything else = SuperLU
        self.shifted_solver = shifted_solver
        self._line_block = shifted_solver if isinstance(shifted_solver, LineBlockSolver) else None
        self._planned_omegas = ()
        self._plan_serial = None
        self._material_serial = 0     # bumped by every material / geometry change (inject_source, materials_changed)
        self._batch = {}              # omega -> line_block.LineBlockFactorization of the kept batch
        self._batch_problems = {}     # omega -> its unsolved cross_section_modes.CrossSectionProblem
        self._batch_serial = None
        self.last_batch_size = None   # operators of the last line-block batch
        self.n_line_block_fallbacks = 0   # eigen-solves the line-block factor rejected (redone on SuperLU)
        self.polarization = MODE_POLARIZATION
        bc = solver_boundary_config(domain, boundary_config, 'ModePhysics')
        self.boundary_config = bc   # Bloch k: phased operator wrap entries (cross_section_modes header)
        self._init_mode_study(wavelength, n_modes, mode_index, n_eff_guess, group_index)
        bend_radius = float(bend_radius)
        x_edges = domain.cell_edges()[0]
        reference_x = 0.5 * float(domain.size_x) if bend_reference_x is None else float(bend_reference_x)
        message = bend_factor_message(x_edges, bend_radius, reference_x)
        if message is not None:
            raise ValueError(f'ModePhysics: {message}')
        if bend_model not in BEND_MODELS:
            raise ValueError(f"ModePhysics bend_model must be one of {', '.join(BEND_MODELS)}, got {bend_model!r}")
        self._bend = None if bend_radius == 0.0 else (bend_radius, reference_x, x_edges, bend_model)
        self._cell_widths = grid_cell_widths(domain.grid)
        self._pml_dx = cpml_grading_cell_size(domain)
        self.wrap_factors = bloch_wrap_factors(self._cell_widths, bc)   # (1, 1) unless Bloch k != 0

        # eps_xx / eps_yy new to the MaterialGrid start from eps_r (an E_z physics or script registered it).
        register_em_arrays(domain.materials, MODE_POLARIZATION)

        self.sources = []   # none (placement / Visualizer protocol)
        self.monitors = []  # ModeTableMonitor / ModeDispersionMonitor
        self.dispersive_regions = []
        self.pec_cells = None
        self.finalize_warnings = []
        self.runtime_warnings = []    # append-only (fewer modes than mode_index); the studies log new entries
        self.enable_q_accum = False   # loss-protocol flag only: em_modes has no absorbed power
        self.subpixel_smoothing = None   # SubpixelSmoothing ('modes') of the model-wide switch (set_subpixel_smoothing)
        # Conformal PEC (solver_defaults.conformal_boundaries): the builder's geometry locator, the union mask and
        # outline tests of add_conformal_pec_region, their dual-cell cuts (host, built at the first solve)
        self.conformal_locator = None
        self._conformal_mask = None
        self._conformal_tests = []
        self._pec_cuts = None

        shape = (domain.grid.nx, domain.grid.ny)
        # float32 full-grid fields (run backend: domain.xp) of the selected mode at the cell centres, rewritten in
        # place (zeros before a solve)
        self.mode_fields = {name: domain.xp.zeros(shape, dtype=DTYPE) for name in MODE_FIELD_NAMES}

    @property
    def permittivity_arrays(self) -> tuple:
        """MaterialGrid permittivity arrays the solve reads: eps_r (= eps_zz), eps_xx, eps_yy."""
        return MODE_PERMITTIVITY_ARRAYS

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Nothing: modes are solved only by the mode_analysis / frequency_domain hooks."""
        return

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Float32 full-grid e_intensity [V^2/m^2] and sz [W/m^2] of the selected mode, the permittivity arrays,
        sigma_e and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        state = {'e_intensity': self.mode_fields['e_intensity'], 'sz': self.mode_fields['sz']}
        for name in MODE_PERMITTIVITY_ARRAYS:
            state[name] = getattr(materials, name)
        state['sigma_e'] = materials.sigma_e
        state['q_accum'] = None
        return state

    def inject_source(self, name: str, array):
        """Copy a coupling's array into the permittivity array of slot `name` (eps_r_update = eps_zz, eps_xx_update,
        eps_yy_update); the next solve reads it."""
        if name not in MODE_PERMITTIVITY_SLOTS:
            raise ValueError(f"ModePhysics accepts sources {', '.join(MODE_PERMITTIVITY_SLOTS)}; got '{name}'")
        xp = self.domain.xp
        xp.copyto(getattr(self.domain.materials, MODE_PERMITTIVITY_SLOTS[name]), xp.asarray(array, dtype=DTYPE))
        self._material_serial += 1

    @property
    def _on_device(self) -> bool:
        """The eigen solver factors on the GPU (cudss): no host SuperLU size rule."""
        return getattr(self.eigen_solver, 'factorization', 'superlu') in CUDSS_FACTORIZATIONS

    def finalize(self) -> None:
        """ValueError without scipy, for a grid over the solver's size limit (SuperLU; none for cudss) or two periodic
        axes (line_block); finalize_warnings = its size warning."""
        self.finalize_warnings = []
        if not scipy_available():
            raise ValueError('em_modes needs scipy (host sparse eigensolver), which is not installed: pip install scipy')
        bc = self.boundary_config
        if self._line_block is not None:
            if bc.periodic_x and bc.periodic_y:
                raise ValueError(f'ModePhysics: {LINE_BLOCK_BOTH_PERIODIC_TEXT}')
            return
        if self._on_device:   # cuDSS guards the device memory itself after planning
            return
        issue = size_issue_for(self.domain.grid.nx, self.domain.grid.ny)
        if issue is not None:
            if issue[0] == 'error':
                raise ValueError(issue[1])
            self.finalize_warnings.append(issue[1])

    def materials_changed(self, names) -> None:
        self._material_serial += 1   # every solve reads the MaterialGrid afresh; a kept line-block batch is stale

    def renderer(self):
        return ModeRenderer(self)

    def set_subpixel_smoothing(self, geometry):
        """Average the current eps on the interface geometry (igor 1.5, model.subpixel_geometry) in every solve and in
        point_materials (E_x / E_y faces anisotropic, E_z cells arithmetic); None switches it off."""
        self.subpixel_smoothing = None if geometry is None else SubpixelSmoothing(geometry, MODE_POLARIZATION)
        self._material_serial += 1

    def set_conformal_boundaries(self, locator):
        """solver_defaults.conformal_boundaries: pec_domain features call add_conformal_pec_region with
        locator.inside(entities) (model.boundary_locator, duck-typed); None = staircased."""
        self.conformal_locator = locator

    def add_conformal_pec_region(self, mask, inside):
        """PEC on a run-backend bool (nx, ny) cell mask whose drawn outline is inside(x, y) (host): every solve then uses
        the conformal Faraday operators of cross_section_modes (core.cut_cells.dual_cell_cuts of the union)."""
        self.add_pec_region(mask)
        self._conformal_mask = add_pec_cells(self._conformal_mask, mask)
        self._conformal_tests.append(inside)
        self._pec_cuts = None

    def _dual_cuts(self):
        """(link_x, link_y, area) of the conformal PEC regions on the full grid (cached), None without any."""
        if not self._conformal_tests:
            return None
        if self._pec_cuts is None:
            x_centres, y_centres = self.domain.cell_centers()
            self._pec_cuts = dual_cell_cuts(x_centres, y_centres, to_host(self._conformal_mask),
                                            union_test(self._conformal_tests))
        return self._pec_cuts

    # --- study hooks ---

    def plan_harmonic_frequencies(self, omegas):
        """frequency_domain study hook: the omegas [rad/s] solve_harmonic follows, in order; a line_block shifted solver
        then batches the shift-invert operators of the next wavelengths too; () ends the plan."""
        self._planned_omegas = tuple(float(omega) for omega in omegas)
        self._plan_serial = self._material_serial
        if not self._planned_omegas:
            self._release_batch()

    def select_mode(self, number):
        """Show mode `number` (1-based, clamped to the modes found) in mode_fields / state; zeros without modes."""
        if not self.modes:
            self.selected_mode = None
            for values in self.mode_fields.values():
                values.fill(0)
            return
        number = min(max(int(number), 1), len(self.modes))
        bc = self.boundary_config
        centres = centre_fields(self.modes[number - 1], bool(bc.periodic_x), bool(bc.periodic_y), self.wrap_factors)
        host_fields = {name: centres[name].real for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')}
        host_fields['e_intensity'] = sum(np.abs(centres[name]) ** 2 for name in ('ex', 'ey', 'ez'))
        host_fields['h_abs'] = np.sqrt(sum(np.abs(centres[name]) ** 2 for name in ('hx', 'hy', 'hz')))
        host_fields['sz'] = 0.5 * np.real(centres['ex'] * np.conj(centres['hy'])
                                          - centres['ey'] * np.conj(centres['hx']))
        for name, values in host_fields.items():
            write_float32(self.mode_fields[name], values)
        self.selected_mode = number

    @property
    def bend(self):
        """The cross_section_modes bend tuple (radius, reference_x, full-grid x edges, model) its solves use; None
        straight."""
        return self._bend

    def point_materials(self, omega=None):
        """cross_section_modes.ModePointMaterials of the current MaterialGrid at omega [rad/s], default the last solve's
        (the eps its modes saw while the materials are unchanged); ValueError without either."""
        omega = self.solved_omega if omega is None else float(omega)
        if omega is None:
            raise ValueError(f"em_modes '{self.name}' has no mode solve yet")
        return mode_point_materials(self._cell_widths, self.boundary_config, self._pml_dx,
                                    self._cross_section_materials(omega), self._bend)

    def error_indicator(self):
        """S7.7 hook (mesh plan A2): energy-weighted interpolation eta^2 parts (2, *window) of the selected mode (after a
        study the mode_index one): E components x eps0 |eps_ii|, H x mu0 (core.error_indicators)."""
        if self.selected_mode is None:
            raise ValueError(f"em_modes '{self.name}' has no solved mode")
        bc = self.boundary_config
        centres = centre_fields(self.modes[self.selected_mode - 1], bool(bc.periodic_x), bool(bc.periodic_y),
                                self.wrap_factors)
        window = tuple(self.domain.physical_window())
        widths = [np.asarray(axis_widths, dtype=np.float64)[axis_window]
                  for axis_widths, axis_window in zip(self.domain.cell_widths(), window)]
        materials = self.domain.materials
        components = [(np.asarray(centres[component])[window],
                       EPS0 * np.abs(np.asarray(to_host(getattr(materials, array)[window]), dtype=np.float64)))
                      for component, array in (('ex', 'eps_xx'), ('ey', 'eps_yy'), ('ez', 'eps_r'))]
        components += [(np.asarray(centres[component])[window], MU0) for component in ('hx', 'hy', 'hz')]
        return energy_interpolation_indicator(components, widths)

    def solve_summary(self):
        """One line on the last solve for the study log (modes found, unknowns, wall times, factor size); None before."""
        if self.last_solve_seconds is None:
            return None
        info = self.last_solve_info
        kind = info.get('factor_kind', 'SuperLU')
        if kind == 'SuperLU':
            factor = (f'SuperLU factorisation {info["factor_seconds"]:.3g} s, L + U factors '
                      f'{format_bytes(info["factor_host_bytes"])} of host memory')
        elif kind.startswith('cuDSS'):   # the eigen solver's factorization cudss
            factor = f'{kind} factorisation on the GPU {info["factor_seconds"]:.3g} s'
        else:
            batch = '' if self.last_batch_size is None else f', one batch of {self.last_batch_size} operators'
            factor = f'{info["factor_kind"]} factor of the shifted solver ({self._line_block.precision}{batch})'
        return (f'{len(self.modes)} of {self.n_modes} modes at '
                f'{wavelength_label(2.0 * math.pi * C0 / self.solved_omega)}: {info["n_unknowns"]:,} unknowns, '
                f'solve {self.last_solve_seconds:.3g} s{" with the group index" if self.group_index else ""} '
                f'({factor}; eigs {info["eigen_seconds"]:.3g} s, '
                f'{info["operator_applications"]} operator applications, {info["eigenvalues_converged"]} of '
                f'{info["eigenvalues_requested"]} eigenvalues, {info["pml_modes_dropped"]} PML modes dropped)')

    def work_counters(self):
        """Work of every solve so far (summary 'solvers' physics counters; the eigen role's ARPACK counts, group-index
        solves included, are its own record, or added here for a private solver); solve_seconds is host wall time."""
        counters = {'solves': self.n_solves, 'modes_found': self.n_modes_found,
                    'solve_seconds': self.total_solve_seconds}
        if self._line_block is not None:
            counters['line_block_fallbacks'] = self.n_line_block_fallbacks
        if self._owns_eigen_solver:
            counters.update(self.eigen_solver.work_counters())
        return counters

    def _solve_once(self, omega):
        """(modes, solve info, new warnings) of one solve at omega (group indices when enabled; a line-block batch whose
        operators are all solved is released)."""
        solve = self._solve_at(omega, self.n_modes)
        modes = list(solve.modes)
        warnings = [warning for warning in solve.info['warnings'] if warning not in self.finalize_warnings]
        if self.group_index and modes:
            modes, group_info = mode_group_indices(self._cell_widths, self.boundary_config, self._pml_dx,
                                                   self._cross_section_materials, omega, modes,
                                                   n_eff_guess=self.n_eff_guess, bend=self._bend,
                                                   relative_step=GROUP_RELATIVE_STEP, eigen_solver=self.eigen_solver,
                                                   solve_at=self._solve_at)
            warnings.extend(group_info['warnings'])
        if self._line_block is not None and not self._batch_problems:
            self._release_batch()   # every operator of the batch solved
        return modes, solve.info, warnings

    def _solve_at(self, omega, n_modes, start_vector=None):
        """CrossSectionSolve at omega of the current materials: the eigen solver's factor (host SuperLU or cuDSS), or
        the line-block batch entry of omega (a rejected solve: SuperLU on the same problem, run-log warning)."""
        if self._line_block is None:
            problem = self._prepared(omega, self.n_eff_guess, self._bend, check_size=not self._on_device)
            return solve_prepared_modes(problem, n_modes, start_vector=start_vector, eigen_solver=self.eigen_solver)
        problem, factorization = self._batched_problem(omega)
        try:
            return solve_prepared_modes(problem, n_modes, start_vector=start_vector, eigen_solver=self.eigen_solver,
                                        factorization=factorization)
        except RejectedLineBlockSolve as exc:
            self.n_line_block_fallbacks += 1
            self.runtime_warnings.append(f'{wavelength_label(2.0 * math.pi * C0 / omega)}: {exc}; shift-invert by host '
                                         'SuperLU instead')
            return solve_prepared_modes(problem, n_modes, start_vector=start_vector, eigen_solver=self.eigen_solver)

    def _problem_omegas(self, omega):
        """Omegas of the shift-invert operators one solve at omega needs: omega, then omega (1 +- step) with the group
        index (mode_group_indices order)."""
        if not self.group_index:
            return [omega]
        return [omega, omega * (1.0 + GROUP_RELATIVE_STEP), omega * (1.0 - GROUP_RELATIVE_STEP)]

    def _problem_sequence(self):
        """Operator omegas in solve order from the current solve on: its own, then those of the later planned
        wavelengths (a plan of the current material state that contains self.omega)."""
        centre = self.omega
        wavelengths = [centre]
        if self._plan_serial == self._material_serial and centre in self._planned_omegas:
            wavelengths = list(dict.fromkeys(self._planned_omegas[self._planned_omegas.index(centre):]))
        return list(dict.fromkeys(omega for wavelength in wavelengths for omega in self._problem_omegas(wavelength)))

    def _prepared(self, omega, n_eff_guess, bend, check_size=False):
        """cross_section_modes.CrossSectionProblem at omega of the current materials (check_size: the host SuperLU size
        rule)."""
        return prepare_cross_section_modes(self._cell_widths, self.boundary_config, self._pml_dx,
                                           self._cross_section_materials(omega), omega, n_eff_guess=n_eff_guess,
                                           bend=bend, check_size=check_size)

    def _batched_problem(self, omega):
        """(CrossSectionProblem, LineBlockFactorization) of omega, taken out of the kept batch; a new batch of the next
        operators of _problem_sequence (at most MAX_BATCH_WAVELENGTHS, what line_block.MEMORY_FRACTION of the free
        memory holds, one layout) when the batch lacks it or is stale."""
        if self._batch:
            entry = next(iter(self._batch.values()))
            if not entry.valid or self._batch_serial != self._material_serial:
                self._release_batch()
        if omega not in self._batch_problems:
            self._release_batch()   # before sizing the next batch: its inverses count as used memory
            sequence = self._problem_sequence()
            upcoming = sequence[sequence.index(omega):] if omega in sequence else [omega]
            first = self._prepared(omega, self.n_eff_guess, self._bend)
            layout = line_block_layout(first)
            n_batch = min(len(upcoming), MAX_BATCH_WAVELENGTHS,
                          self._line_block.batch_size(len(upcoming), layout, first.operator.nnz))
            problems = [first]
            for other in upcoming[1:n_batch]:
                problem = self._prepared(other, self.n_eff_guess, self._bend)
                if not np.array_equal(problem.active_index, first.active_index):
                    break
                problems.append(problem)
            labels = [f'{self._line_block.label()} ({wavelength_label(2.0 * math.pi * C0 / problem.omega)})'
                      for problem in problems]
            entries = batch_factorizations(self._line_block, [problem.operator for problem in problems], layout,
                                           labels, owner=self)
            self._batch = {problem.omega: entry for problem, entry in zip(problems, entries)}
            self._batch_problems = {problem.omega: problem for problem in problems}
            self._batch_serial = self._material_serial
            self.last_batch_size = len(problems)
        return self._batch_problems.pop(omega), self._batch[omega]

    def _release_batch(self):
        """Drop the kept line-block batch (device inverses, operator caches) and its unsolved problems."""
        for entry in self._batch.values():
            entry.release()   # one shared LineBlockFactor: the first call frees it
            break
        self._batch = {}
        self._batch_problems = {}
        self._batch_serial = None

    def _cross_section_materials(self, omega):
        """CrossSectionMaterials at omega: host complex128 eps_xx / eps_yy / eps_zz (sigma_e, poles) and mu (sigma_m)
        of every cell, host bool PEC cells, the subpixel-smoothing deltas of the background eps when enabled, the
        conformal PEC cuts when any."""
        materials = self.domain.materials
        xp = self.domain.xp
        eps = {name: to_host(cell_permittivity(materials, name, self.dispersive_regions, omega, xp=xp))
               for name in MODE_PERMITTIVITY_ARRAYS}
        pec = (np.zeros((self.domain.grid.nx, self.domain.grid.ny), dtype=bool) if self.pec_cells is None
               else to_host(self.pec_cells).copy())
        deltas = None if self.subpixel_smoothing is None else self.subpixel_smoothing.point_deltas(materials)
        return CrossSectionMaterials(eps_xx=eps['eps_xx'], eps_yy=eps['eps_yy'], eps_zz=eps['eps_r'],
                                     mu=to_host(cell_permeability(materials, omega, xp=xp)), pec=pec,
                                     eps_point_deltas=deltas, pec_cuts=self._dual_cuts())

    def solve_detached(self, omega, n_modes, n_eff_guess=None, straight=False):
        """CrossSectionSolve at omega of the current materials (same grid, walls, bend unless straight, eigen solver),
        leaving modes, selection and monitors untouched (the 2 omega modes of shg_overlap, the straight guide of
        bend_overlap)."""
        problem = self._prepared(float(omega), n_eff_guess, None if straight else self._bend,
                                 check_size=not self._on_device)
        return solve_prepared_modes(problem, int(n_modes), eigen_solver=self.eigen_solver)

    # --- material helpers ---

    def add_dispersive(self, pole, mask):
        """Record a Drude / Lorentz pole on a run-backend bool cell mask (None = everywhere): its chi(omega) joins every
        solve; the permittivity arrays stay the background."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f"add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}")
        if mask is not None:
            _check_cell_array(mask, self.mode_fields['ex'].shape, np.bool_, 'Pole mask', xp=self.domain.xp)
        record_dispersive_region(self.dispersive_regions, pole, mask)
        self._material_serial += 1

    def add_pec_region(self, mask):
        """Perfect electric conductor on a run-backend bool (nx, ny) cell mask."""
        _check_cell_array(mask, self.mode_fields['ex'].shape, np.bool_, 'PEC mask', xp=self.domain.xp)
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self._material_serial += 1

    # --- monitors (+ the mode table / dispersion of _ModeStudyMixin) ---

    def add_shg_overlap(self, d_tensor, cell_mask, pump_mode=1, harmonic_mode=1, n_harmonic_modes=4,
                        harmonic_n_eff_guess=None, ring_crystal='local'):
        """SHG coupled-mode overlap of pump mode / 2 omega harmonic mode per solve on the device or host bool (nx, ny)
        chi2 cells, d_tensor (3, 3, 3) [m/V] in model axes, ring_crystal 'local' | 'fixed' (nonlinear_overlap);
        ValueError with a mirror face, a Bloch k != 0 or a fixed ring crystal without a bend."""
        if self.boundary_config.ghost_faces:
            raise ValueError('shg_overlap needs the whole cross-section: mirror faces '
                             f"{', '.join(self.boundary_config.ghost_faces)} keep only one parity of the overlap")
        if self.boundary_config.has_bloch_phase:
            raise ValueError('shg_overlap needs Bloch k = 0: the 2 omega modes would need the Bloch wave vector 2 k')
        if ring_crystal == 'fixed' and (self.bend is None or float(self.bend[0]) == 0.0):
            raise ValueError("shg_overlap ring_crystal 'fixed' needs a bend (bend_radius != 0)")
        mask = np.asarray(to_host(cell_mask), dtype=bool)
        return self._add_monitor(ShgOverlapMonitor(d_tensor, mask, pump_mode, harmonic_mode, n_harmonic_modes,
                                                   harmonic_n_eff_guess, ring_crystal))

    def add_bend_overlap(self, n_straight_modes=4, straight_n_eff_guess=None):
        """Straight-to-bend junction per solve (BendOverlapMonitor); ValueError for a straight guide."""
        if self.bend is None:
            raise ValueError('bend_overlap needs a bent guide (bend_radius != 0): a straight guide matches itself')
        return self._add_monitor(BendOverlapMonitor(n_straight_modes, straight_n_eff_guess))


def _positive(value, what):
    """float(value), ValueError unless finite and > 0."""
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f'ModePhysics {what} must be finite and > 0, got {value!r}')
    return value


def _positive_int(value, what):
    """int(value), ValueError unless >= 1."""
    if int(value) != value or int(value) < 1:
        raise ValueError(f'ModePhysics {what} must be an integer >= 1, got {value!r}')
    return int(value)


def _mode_scalars(mode):
    """[Re n_eff, Im n_eff, loss dB/cm, n_group (NaN when not computed), te_fraction, effective_area [m^2],
    pml_fraction] of a CrossSectionMode."""
    n_eff = complex(mode.n_eff)
    n_group = math.nan if mode.n_group is None else complex(mode.n_group).real
    loss_db_per_cm = -_DB_PER_NEPER * complex(mode.beta).imag / 100.0   # Im beta < 0 = loss
    return [n_eff.real, n_eff.imag, loss_db_per_cm, n_group, float(mode.te_fraction), float(mode.effective_area),
            float(mode.pml_fraction)]


class ModeTableMonitor:
    """Rows of every mode of the last solve (MODE_TABLE_COLUMNS) and the selected number."""

    def __init__(self):
        self.rows = None
        self.omega = None

    def record(self, physics):
        self.omega = physics.solved_omega
        selected = physics.selected_mode
        self.rows = [[number, *_mode_scalars(mode), number == selected]
                     for number, mode in enumerate(physics.modes, start=1)]


class ModeDispersionMonitor:
    """Scalars of one mode per solved omega: mode_number at the first solve with that many modes, then the mode of
    largest |mode_overlap| with the previously followed one."""

    def __init__(self, mode_number=1):
        self.mode_number = _positive_int(mode_number, 'mode_dispersion mode_number')
        self._records = {}     # omega [rad/s] -> _mode_scalars + [overlap with the previously followed mode]
        self._followed = None  # CrossSectionMode of the last record

    def record(self, physics):
        modes = physics.modes
        if self._followed is None:
            index, overlap = (self.mode_number - 1, 1.0) if self.mode_number <= len(modes) else (None, math.nan)
        elif modes:
            overlaps = [mode_overlap(self._followed, mode) for mode in modes]
            index = int(np.argmax(overlaps))
            overlap = overlaps[index]
        else:
            index, overlap = None, math.nan
        if index is None:   # not found (yet): NaN row, keep following the previous mode
            self._records[float(physics.solved_omega)] = [math.nan] * (len(MODE_TABLE_COLUMNS) - 2) + [overlap]
            return
        self._followed = modes[index]
        self._records[float(physics.solved_omega)] = _mode_scalars(modes[index]) + [overlap]

    def recorded(self):
        """(omega float64 in ascending wavelength order, float64 (n, 8) rows); ValueError before the first solve."""
        if not self._records:
            raise ValueError('not solved yet: the mode dispersion has no solve (run a mode_analysis or '
                             'frequency_domain study)')
        omegas = sorted(self._records, reverse=True)
        return (np.array(omegas, dtype=np.float64),
                np.array([self._records[omega] for omega in omegas], dtype=np.float64))


class BendOverlapMonitor:
    """Straight-to-bend junction of the last solve: the straight guide's modes (solve_detached straight, same
    materials) and per bent mode the straight mode of largest mode_overlap (power coupling at the junction plane)."""

    def __init__(self, n_straight_modes=4, straight_n_eff_guess=None):
        self.n_straight_modes = _positive_int(n_straight_modes, 'bend_overlap n_straight_modes')
        self.straight_n_eff_guess = (None if straight_n_eff_guess is None
                                     else _positive(straight_n_eff_guess, 'bend_overlap straight_n_eff_guess'))
        self.rows = None
        self.omega = None
        self.straight_modes = []   # CrossSectionMode of the straight guide at the last solve

    def record(self, physics):
        straight = physics.solve_detached(physics.solved_omega, self.n_straight_modes, self.straight_n_eff_guess,
                                          straight=True).modes
        rows = []
        for number, bent in enumerate(physics.modes, start=1):
            overlaps = [mode_overlap(bent, mode) for mode in straight]
            best = int(np.argmax(overlaps)) if overlaps else None
            overlap = math.nan if best is None else float(overlaps[best])
            same = float(overlaps[number - 1]) if number <= len(overlaps) else math.nan
            loss_db = -10.0 * math.log10(overlap) if overlap > 0.0 else math.nan
            n_eff = complex(bent.n_eff)
            rows.append([number, n_eff.real, n_eff.imag, math.nan if best is None else best + 1,
                         math.nan if best is None else complex(straight[best].n_eff).real, overlap, loss_db, same])
        self.omega, self.rows, self.straight_modes = physics.solved_omega, rows, straight


BEND_OVERLAP_COLUMNS = ('mode_number', 'n_eff', 'n_eff_imag', 'straight_mode', 'n_eff_straight', 'overlap',
                        'junction_loss_db', 'overlap_same_rank')
BEND_OVERLAP_UNITS = ('1', '1', '1', '1', '1', '1', 'dB', '1')


def bend_overlap_result(monitor) -> dict:
    """'table' of a BendOverlapMonitor (BEND_OVERLAP_COLUMNS, one row per bent mode); extras wavelength [m],
    n_straight_found; ValueError before the first solve."""
    if monitor.rows is None:
        raise ValueError('not solved yet: bend_overlap has no solve (run a mode_analysis or frequency_domain study)')
    return {'kind': 'table', 'columns': list(BEND_OVERLAP_COLUMNS), 'units': list(BEND_OVERLAP_UNITS),
            'rows': [list(row) for row in monitor.rows], 'wavelength': float(2.0 * math.pi * C0 / monitor.omega),
            'n_straight_found': len(monitor.straight_modes)}


def mode_table_result(monitor) -> dict:
    """'table' of a ModeTableMonitor (last solve, descending Re beta^2); extras wavelength [m], selected_mode;
    ValueError before the first solve."""
    if monitor.rows is None:
        raise ValueError('not solved yet: the mode table has no solve (run a mode_analysis or frequency_domain study)')
    selected = next((row[0] for row in monitor.rows if row[-1]), None)
    return {'kind': 'table', 'columns': list(MODE_TABLE_COLUMNS), 'units': list(MODE_TABLE_UNITS),
            'rows': [list(row) for row in monitor.rows], 'wavelength': float(2.0 * math.pi * C0 / monitor.omega),
            'selected_mode': selected}


def mode_dispersion_result(monitor) -> dict:
    """'xy' of a ModeDispersionMonitor vs ascending wavelength [m]: series n_eff (Re), n_eff_imag, loss_db_per_cm,
    n_group, te_fraction, effective_area, pml_fraction, overlap; extras frequency, wavelength, mode_number."""
    omegas, values = monitor.recorded()
    names = MODE_TABLE_COLUMNS[1:-1] + ('overlap',)
    units = MODE_TABLE_UNITS[1:-1] + ('1',)
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Effective index',
            'series': {name: values[:, column] for column, name in enumerate(names)},
            'series_units': dict(zip(names, units)), 'frequency': frequency, 'wavelength': wavelength.copy(),
            'mode_number': monitor.mode_number}

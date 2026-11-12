"""2D FDFD electromagnetics: one sparse Helmholtz solve per angular frequency for E_z ('ez') or H_z ('hz') on the A1
Yee operators (Bloch k != 0 on the wrap entries), with phasor sources, harmonic monitors, conducting sheets and the EM
loss protocol."""
import cmath
import math
import time

import numpy as np
from ....core.backend import to_host

from ....core.constants import C0, DTYPE, EPS0, MU0
from ....core.error_indicators import energy_interpolation_indicator
from ....core.gpu_memory import format_bytes
from ....core.solver_base import PhysicsSolver
from ..adjoint import (OBJECTIVE_QUANTITIES, EpsSensitivity, FluxObjective, ModePowerObjective, eps_sensitivity,
                       is_objective)
from ....core.cut_cells import union_test
from ....core.metric import window_metric
from ..boundaries import MIRROR_TYPES, WALL_SIGNS, BCType, solver_boundary_config
from ..conformal_pec import build_conformal_fdfd
from ..design_density import DesignRegion, paint_regions
from ..eigenfrequency import solve_resonances
from ....solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, LDLT_FACTORIZATIONS, ArpackShiftInvertSolver
from ....solvers.host_direct import HOST_DIRECT_BACK_ENDS
from ....solvers.line_block import (LineBlockFactorization, LineBlockSolver, RejectedLineBlockSolve,
                                    operator_shift_bytes)
from ....solvers.sparse_direct import (CUDSS_BACK_ENDS, SparseDirectSolver, host_size_back_end, qr_workspace_bytes,
                                       size_issue)
from ..fdfd_line_block import LINE_BLOCK_TEXT, MAX_BATCH_WAVELENGTHS, fdfd_block_axis, fdfd_layout
from ..fdfd_monitors import HarmonicFluxMonitor, HarmonicNearToFar, HarmonicProbe
from ..fdfd_ports import HarmonicModeMonitor, ModeSourcePhasor
from ..fdfd_sources import (ElectricDipolePhasor, MagneticCurrentPhasor, PlaneWavePhasor, PointCurrentPhasor,
                            SheetCurrentPhasor)
from ..harmonic import HarmonicSystem, absorbed_power_ez, absorbed_power_hz, cell_permeability, cell_permittivity
from ..index_reference import OFF_DIAGONAL_ARRAY
from ..materials import DrudePole, LorentzPole, register_em_arrays, tfsf_outline_warnings
from ..operators import (active_coupling_faces, bloch_wrap_phases, helmholtz_operator, off_diagonal_face_couplings,
                         pin_image_cells, stretched_cell_areas, yee_operators)
from ..placement import (PERMITTIVITY_ARRAYS, box_window_faces, electric_dipole_faces,
                         flux_line_cells, line_mirror_factor, mirror_planes, near_to_far_box_cells, point_source_cell,
                         port_cells, probe_cell, require_polarization, sheet_line, tfsf_box_cells, tfsf_open_faces)
from ..renderer import FDFDRendererEz, FDFDRendererHz
from ..sheet_conductivity import ConductingSheet, SheetConductivity
from ..sources import bloch_line_phase
from ..subpixel_smoothing import SubpixelSmoothing
from ..yee_grid import cell_row, cpml_grading_cell_size, face_average, face_pec, grid_cell_widths
from .fdtd_2d_base import _check_cell_array, add_pec_cells, record_dispersive_region
from .harmonic_shell import COMPONENT_UNITS, HarmonicEMShell

# Wall / mirror faces: outermost cell = sign x its inner neighbour, as the FDTD BoundaryManager writes the row (sign 0:
# pinned to 0). Even faces (sign +1) also get face weight 0 on the inner face: zero flux, and a face current there drops.
_WALL_SIGNS = WALL_SIGNS
_REJECTED = 'rejected'   # _line_factor of a system whose line-block solve missed the accept tolerance
_SPARSE_DIRECT_TEXT = 'sparse direct'


class FDFDPhysics(HarmonicEMShell, PhysicsSolver):
    """Frequency-domain EM on a Domain, polarisation 'ez' (E_z, H_x, H_y) or 'hz' (H_z, E_x, E_y): assembles the complex
    Helmholtz system at omega, factorises it once per (omega, materials) and solves for the summed phasor sources."""
    # Dirty flags are host bools: _system_dirty (materials, eps slots, PEC, poles, a new omega) reassembles and
    # refactorises (scipy_splu at an unchanged omega: GMRES on the old factor first, see _solve_linear);
    # _solution_dirty (that, or a new source) re-solves in step(). Port protocol (fdfd_ports mode source /
    # monitor): a source / monitor with solve(physics, omega) and `warnings` re-solves its modes at finalize and after
    # every reassembly (before any add_to_rhs / record); finalize collects the warnings.

    name = 'em'
    subpixel_smoothing = None   # SubpixelSmoothing of the model-wide switch (set_subpixel_smoothing); ports read it
    objective_unit = 'W/m'      # J of the adjoint objectives (optimization study labels, design_region result)
    # Polar (r, phi) Domains (symmetry 2.16 item 5 / 2.18, scripts only: the em_fdfd_* PhysicsTypes stay Cartesian until
    # the model path exists). x = r - radius_min, y = phi; the metric sits in the A1 ops (operators.yee_operators), so
    # face fields are E_r / E_phi (hz) or H_r / H_phi (ez) under their _x / _y names. Faces: x_lo a pec / pmc wall at
    # radius_min > 0, or the origin (radius_min 0, set 'pmc': zero face length, no wall row); x_hi cpml (cylindrical PML,
    # complex r) / pec / pmc; y faces pec / pmc / mirrors (rays are mirror lines: C_nv) or periodic / bloch on a domain
    # whose angle_wrap joins them, bloch_ky = the azimuthal order m [1/rad] (F(phi + span) = F(phi) e^{-j m span}: the
    # C_N sector phase of COMSOL's Floquet periodicity; the domain's angle_wrap only says the faces are joined).
    coordinate_grids = ('cartesian', 'polar')

    def __init__(self, domain, polarization='ez', boundary_config=None, wavelength=1.55e-6, linear_solver='auto',
                 eigen_solver=None):
        if polarization not in PERMITTIVITY_ARRAYS:
            raise ValueError(f"polarization must be 'ez' (E_z, H_x, H_y) or 'hz' (H_z, E_x, E_y), "
                             f"got {polarization!r}")
        self.domain = domain
        self.polarization = polarization
        self.xp = domain.xp   # run backend arrays: cupy + cupyx CSR ('gpu') or numpy + scipy CSR ('cpu')

        bc = solver_boundary_config(domain, boundary_config, 'FDFDPhysics')
        self.boundary_config = bc
        self._wall_rows = _outer_wall_rows(bc, polarization)
        # core.metric.WindowMetric of the whole grid (real radii, PML cells included) on a polar Domain, else None
        self.metric = None
        if domain.coordinates.polar:
            _check_polar_boundaries(domain, bc)
            self.metric = window_metric(*grid_cell_widths(domain.grid), domain.coordinates)
            if domain.coordinates.radius_min == 0.0:   # the origin: zero face length, no wall row
                self._wall_rows = tuple(row for row in self._wall_rows if not (row[0] == 0 and row[1] == 0))
        # Bloch k != 0 (operators header): the phasor one period below index 0 is the stored last value times e^{+j k L}
        # per axis (1 on other axes): face-to-centre means, wrapped dipole faces and port neighbours use it.
        self.lower_wrap_factors = tuple(cmath.exp(1j * phase)
                                        for phase in bloch_wrap_phases(grid_cell_widths(domain.grid), bc))

        # omega, the float32 full-grid state (_field_re / _field_im / _q_abs, rewritten in place by every solve, zeros
        # before the first) and the hz eps_xy_update change (HarmonicEMShell)
        self._init_harmonic_shell(wavelength, (domain.grid.nx, domain.grid.ny))
        # The build's 'linear' / 'eigen' solvers (ctx.solvers); scripts pass a back_end choice, 'line_block' or nothing
        # and get private ones, whose counters work_counters then reports (a role runtime's are the summary's role
        # record). A line_block linear solver (fdfd_line_block) keeps one block factor per system; its rejected and
        # transposed (adjoint) solves go to a private host-reference SparseDirectSolver (_reference_solver, lazy).
        self._private_solvers = []
        if isinstance(linear_solver, str) and linear_solver == 'line_block':
            linear_solver = LineBlockSolver(backend=domain.backend.name)
            self._private_solvers.append(linear_solver)
        elif not isinstance(linear_solver, (SparseDirectSolver, LineBlockSolver)):
            linear_solver = SparseDirectSolver(back_end=linear_solver, backend=domain.backend.name)
            self._private_solvers.append(linear_solver)
        self._line_block = linear_solver if isinstance(linear_solver, LineBlockSolver) else None
        self._line_layout = None        # fdfd_line_block.fdfd_layout of the grid (first line-block factorisation)
        self._line_factor = None        # LineBlockFactor of system.operator, _REJECTED or None
        self._line_factor_index = 0     # shift index of system.operator in _line_factor (a wavelength batch holds several)
        # Wavelength batches (Part II stage 3, plan_harmonic_frequencies): the study's omegas in solve order, the
        # _material_serial they were planned at, and the kept factorize_operators batch {omega: shift index}.
        self._planned_omegas = ()
        self._plan_serial = None
        self._material_serial = 0       # bumped by every system change except a new omega
        self._batch_factor = None
        self._batch_omegas = {}
        self._batch_serial = None
        self.last_batch_size = None     # wavelengths of the last batched line-block factorisation
        self._reference_solver = None   # line_block: SparseDirectSolver of rejected / non-symmetric transposed solves
        self._reference_factorization = None   # its Factorization of system.operator
        self._adjoint_symmetry = None   # line_block: (system, host D or None) of _adjoint_symmetrizer
        self.n_line_block_fallbacks = 0   # driven systems the line-block factor rejected (solved by the reference)
        self.last_linear_text = None    # 'line-block' / 'sparse direct' of the last driven solve
        self.runtime_warnings = []      # append-only (rejected line-block solves); the studies log new entries
        if eigen_solver is None:
            eigen_solver = ArpackShiftInvertSolver()
            self._private_solvers.append(eigen_solver)
        self.linear_solver = linear_solver   # back end resolved at finalize, else at the first solve
        self.eigen_solver = eigen_solver

        # hz: eps_xx / eps_yy are what E_x / E_y see (as FDTDEngine2D_Hz): arrays the model build registered are kept,
        # new ones start from eps_r when it exists.
        register_em_arrays(domain.materials, polarization)

        self.sources = []
        self.monitors = []
        # (pole, bounding-box slices, xp bool mask cropped to the box; box and mask None = everywhere) per dispersive
        # region, and the union PEC cell mask (xp bool, None until a PEC region): read by the assembly and the ports.
        self.dispersive_regions = []
        self.pec_cells = None
        # Conformal PEC (solver_defaults.conformal_boundaries): the model's geometry locator (set_conformal_boundaries),
        # the union cell mask and outline tests of add_conformal_pec_region, and their conformal_pec.ConformalFDFD on the
        # run backend (built at finalize or the first assembly; None again after a new PEC region).
        self.conformal_locator = None
        self.conformal_pec = None
        self._conformal_mask = None
        self._conformal_tests = []
        self.design_regions = []      # design_density.DesignRegion (optimization study; eps written into the arrays)
        self.phase_change_media = []  # phase_change.PhaseChangeMedium (eps / sigma_e written into the arrays)
        self.conducting_sheets = []   # sheet_conductivity.ConductingSheet: operator terms + loss, never in the arrays
        self._extra_permittivity = None   # (box | None, ez array | hz (xx, yy)): see set_extra_permittivity
        self.finalize_warnings = []
        self.enable_q_accum = False   # loss-protocol flag set by the runner; q_abs is computed by every solve

        self.solution = None          # xp complex128 (nx, ny) E_z / H_z of the last solve
        self.system = None            # HarmonicSystem of the last assembly
        self.solved_omega = None      # rad/s of the last solve
        self.host_factor_bytes = None   # host factor bytes (SuperLU L + U, MUMPS / PARDISO; Simulation.memory_report)
        self.last_factor_seconds = None   # host wall time of the last factorisation
        self.last_solve_seconds = None    # host wall time of the last whole solve (incl. any refactorisation)
        self.last_recycled_iterations = None   # GMRES iterations of the last solve if it reused an older factor
        self.n_solves = 0             # completed solves: cadenced couplings skip a transfer while it is unchanged
        self.eigen_solve = None       # eigenfrequency.EigenSolve of the last solve_eigenfrequencies
        self.resonance_region = None  # eigenfrequency.ResonanceRegion of a resonance_table selection, None = none
        self.selected_resonance = None   # 1-based resonance shown in solution / state, None = the driven solve
        self._factorization = None
        self._factorization_current = False   # factor of system.operator itself (else of an earlier one at omega)
        self._factored_omega = None
        # cuDSS back ends: the factor of an older system, refactorised in place by the next factorisation (plan reused
        # while the pattern holds); a general LU factor's adjoint solves use a factor of the transpose (lazy, alike)
        self._cudss_previous = None
        self._transposed_factorization = None
        self._transposed_previous = None
        self._yee_ops = None          # (omega, YeeOperators2D): materials-independent, reused while omega is unchanged
        self._face_fields = None      # cached face_fields() of the last solve
        self._sensitivity = None      # ((n_solves, objectives), EpsSensitivity): eps_sensitivity() cache
        self._system_dirty = True
        self._solution_dirty = True
        self._port_modes_current = False

    # --- PhysicsSolver contract (+ HarmonicEMShell) ---

    def finalize(self) -> None:
        """Resolve the linear back end (ValueError naming 'pip install scipy' or the other choice; also for a grid over
        its size limit) and solve the port modes at omega; finalize_warnings = the ports' warnings plus non-uniform TFSF
        box edges and (hz) port / sheet-source lines on eps_xy cells."""
        self.finalize_warnings = []
        self._conformal_arrays()    # built with the model (steady memory), not at the first solve
        self._check_solver_size()   # resolves the back end first
        self._solve_port_modes()
        for port in self._mode_ports():
            self.finalize_warnings.extend(port.warnings)
        self.finalize_warnings.extend(tfsf_outline_warnings(
            self, [source for source in self.sources if isinstance(source, PlaneWavePhasor)], uniform=True))
        self.finalize_warnings.extend(self._off_diagonal_line_warnings())

    def _off_diagonal_line_warnings(self):
        """hz: one warning per mode port / line source / beam whose line cells hold a non-zero eps_xy of the assembly
        (material eps_xy now; a later eps_xy_update coupling is not checked): they solve with the diagonal eps only."""
        if self.polarization != 'hz':
            return []
        cell_eps_xy = self.assembly_eps_xy()
        if cell_eps_xy is None:
            return []
        warnings = []
        for item in (*self.sources, *self.monitors):
            if isinstance(item, SheetCurrentPhasor):
                label, line = item.label, item.line
                consequence = ('its launch is calibrated with the face permittivity eps_ii - eps_xy^2 / eps_jj and '
                               'ignores the eps_xy coupling, so the radiated amplitude is approximate')
            elif hasattr(item, 'port') and hasattr(item, 'solve'):   # mode source / monitor / mode-power objective
                label, line = item.port.label, item.port.line_index(item.position)
                consequence = ('its slab modes are solved with eps_xx / eps_yy only, so the injected or measured mode '
                               'is not the guided mode of the anisotropic line and powers / fractions are approximate')
            else:
                continue
            largest = float(self.xp.abs(cell_eps_xy[line]).max())   # one host sync per line, at finalize only
            if largest > 0.0:
                warnings.append(f"{label}: its line cells have an off-diagonal eps_xy (max |eps_xy| {largest:.3g}); "
                                f"{consequence}. Put the line in cells without eps_xy (isotropic or unrotated)")
        return warnings

    def set_subpixel_smoothing(self, geometry):
        """Average the current background eps on the interface locations of `geometry` (core.interface_geometry) in
        every assembly and in the port lines, as EMPhysics does; None switches it off."""
        if geometry is not None:
            self._require_cartesian('subpixel smoothing')
        self.subpixel_smoothing = None if geometry is None else SubpixelSmoothing(geometry, self.polarization)
        self._mark_system_dirty()

    def renderer(self):
        renderer_class = FDFDRendererEz if self.polarization == 'ez' else FDFDRendererHz
        return renderer_class(self)

    def plan_harmonic_frequencies(self, omegas):
        """frequency_domain study hook: the omegas [rad/s] solve_harmonic follows, in order; a line_block linear solver
        then factorises up to MAX_BATCH_WAVELENGTHS of them per batch (Part II stage 3); () ends the plan."""
        self._planned_omegas = tuple(float(omega) for omega in omegas)
        self._plan_serial = self._material_serial
        if not self._planned_omegas:
            self._release_batch()

    def set_resonance_region(self, region):
        """Keep the ResonanceRegion (host bool (nx, ny) cells) whose energy fraction the eigen solve reports / filters;
        ValueError for another shape."""
        if region.cells.shape != self._field_re.shape:
            raise ValueError(f'resonance region cells {region.cells.shape} do not match the grid {self._field_re.shape}')
        self.resonance_region = region

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances nearest the wavelength setting (shift-invert on the
        system assembled there, eigenfrequency.solve_resonances; a line_block linear solver's factor of that system,
        host SuperLU otherwise or after it rejects a solve) -> EigenSolve, kept as eigen_solve."""
        self.eigen_solve = None
        # SuperLU: the scipy_splu size rule below; line_block: the block-axis rule
        system = self._refresh_system(check_driven_size=self._line_block is not None)
        self._release_kept_cudss()   # no driven factorisation follows here: free the device factors kept for a plan
        shape = self._field_re.shape
        pinned = self._pinned_cells(system.pec_cells)   # identity rows: PEC and the outermost wall / mirror cells
        cell_mass = system.cell_eps if system.polarization == 'ez' else system.cell_mu
        if system.cell_area_fractions is not None:   # hz conformal PEC: the free-area mass of the cut cells
            cell_mass = cell_mass * system.cell_area_fractions
        mass = np.asarray(to_host(cell_mass), dtype=np.complex128).ravel().copy()
        mass[np.asarray(to_host(pinned), dtype=bool).ravel()] = 0.0
        # cell areas of the PML-share filter = the outer product of these factors (polar: r_c dr x dphi)
        area_factors = (self.domain.cell_widths() if self.metric is None
                        else (self.metric.volume.along_x, self.metric.volume.along_y))
        cell_areas = np.outer(*(np.asarray(factor, dtype=np.float64) for factor in area_factors))
        in_window = np.zeros(shape, dtype=bool)
        in_window[self.domain.physical_window()] = True
        arguments = (_host_sparse(system.operator), mass, system.omega, cell_areas, in_window, n_resonances)
        options = {'field_of': lambda vector: vector.reshape(shape), 'eigen_solver': self.eigen_solver,
                   'symmetrizer': self._eigen_symmetrizer(system.omega)}
        if self.resonance_region is not None:
            options.update(in_region=self.resonance_region.cells.ravel(),
                           min_region_fraction=self.resonance_region.min_fraction)
        kept = self._system_line_factor() if self._line_block is not None else None
        factorization = (None if kept is None else
                         LineBlockFactorization(kept[0], kept[1], f'{self._line_block.label()} (shift-invert)'))
        try:
            self.eigen_solve = solve_resonances(*arguments, factorization=factorization, **options,
                                                warnings=self._eigen_size_warnings() if factorization is None else ())
        except RejectedLineBlockSolve as exc:
            self.n_line_block_fallbacks += 1
            self.runtime_warnings.append(f'{exc}; shift-invert by host SuperLU instead')
            self._drop_line_factor()
            self._line_factor = _REJECTED
            self.eigen_solve = solve_resonances(*arguments, **options, warnings=self._eigen_size_warnings())
        return self.eigen_solve

    def _eigen_symmetrizer(self, omega):
        """_symmetrizer for an eigen solver whose own factorization is an LDL^T (cuDSS, MUMPS, PARDISO), else None."""
        if getattr(self.eigen_solver, 'factorization', None) not in LDLT_FACTORIZATIONS:
            return None
        return self._symmetrizer(omega)

    def _symmetrizer(self, omega):
        """Stretched cell areas D of the system at omega (D A complex symmetric: a cuDSS / MUMPS / PARDISO factor of
        D A is an LDL^T), None with a Bloch phase."""
        if self.boundary_config.has_bloch_phase:
            return None
        domain = self.domain
        cell_widths = grid_cell_widths(domain.grid)
        polar = self.metric is not None   # as _assemble: PML graded on the outermost r width
        pml_dx = float(cell_widths[0][-1]) if polar else cpml_grading_cell_size(domain)
        return stretched_cell_areas(cell_widths, self.boundary_config, pml_dx, omega,
                                    polar_radius_min=domain.coordinates.radius_min if polar else None)

    def _eigen_size_warnings(self):
        """[warning] of the host size rule of the shift-invert factor (whatever the linear back end; SuperLU's, also
        for MUMPS / PARDISO), [] below it or with a cuDSS factorization (GPU); ValueError beyond its limit."""
        factorization = getattr(self.eigen_solver, 'factorization', None)
        if factorization in CUDSS_FACTORIZATIONS:
            return []
        issue = size_issue(host_size_back_end(factorization), *self._field_re.shape, False,
                           False)   # periodicity matters to cusolver_qr only
        if issue is None:
            return []
        if issue[0] == 'error':
            raise ValueError(f'eigenfrequency solve: {issue[1]}')
        return [issue[1]]

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last eigenfrequency solve in solution, q_abs and the state arrays
        (in-plane fields from the curl at the target omega); the next step() solves the driven problem again."""
        resonance = self.eigen_solve.resonances[number - 1]
        xp = self.xp
        if self.solution is None:
            self.solution = xp.empty(self._field_re.shape, dtype=xp.complex128)
        xp.copyto(self.solution, xp.asarray(resonance.field))
        self.selected_resonance = number
        self._refresh_state(self.system)
        self._solution_dirty = True

    def face_curls(self):
        """Fresh complex128 (hi-y, hi-x) (dy_h u, dx_h u) of the last solve; hz adds the sources' face currents (dy_h H_z
        + J_x, dx_h H_z - J_y = D_x, -D_y up to j omega eps0): what face_fields weights and scales."""
        if self.solution is None or self.system is None:
            raise ValueError("not solved yet: FDFDPhysics face fields need a solve (step or solve_harmonic) first")
        system = self.system
        ops = system.ops
        solution = self.solution.ravel()
        shape = self.solution.shape
        y_face_curls = (ops.dy_h @ solution).reshape(shape)   # d/dy on the hi-y faces
        x_face_curls = (ops.dx_h @ solution).reshape(shape)   # d/dx on the hi-x faces
        if system.polarization == 'hz':
            # Right-hand-side-only sources (TFSF / mode) have no face currents; a shown resonance is source-free.
            for source in (self.sources if self.selected_resonance is None else ()):
                if hasattr(source, 'face_currents'):
                    jx_faces, jy_faces = source.face_currents(self, system)
                    if jx_faces is not None:
                        y_face_curls += jx_faces
                    if jy_faces is not None:
                        x_face_curls -= jy_faces
        return y_face_curls, x_face_curls

    def face_fields(self):
        """complex128 (nx, ny) in-plane fields of the last solve: ez (hx on hi-y, hy on hi-x faces), hz (ex on hi-y, ey
        on hi-x faces, 0 on PEC faces); face_curls weighted (+ the eps_xy couplings) and scaled, cached per solve."""
        if self._face_fields is None:
            y_face_fields, x_face_fields = self.face_curls()
            system = self.system
            shape = self.solution.shape
            if system.polarization == 'ez':
                # Faraday: H_x = -(dy_h E_z) / (j omega mu0 mu_face), H_y = (dx_h E_z) / (j omega mu0 mu_face)
                curl_scale = 1.0 / (1j * system.omega * MU0)
                y_scale, x_scale = -curl_scale, curl_scale
            else:
                # Ampere: E_x = (dy_h H_z + J_xF) / (j omega eps0 eps_xface), E_y = -(dx_h H_z - J_yF) / (j omega eps0
                # eps_yface); on a sheet face the curl alone would be ~2/(k w_perp) times the field. Weight 0 on PEC
                # faces gives E = 0 there.
                curl_scale = 1.0 / (1j * system.omega * EPS0)
                y_scale, x_scale = curl_scale, -curl_scale
            # eps_xy (harmonic.py header): D_x = y_face_fields, D_y = -x_face_fields; E_x += X_xy D_y, E_y += X_yx D_x
            coupled = None
            if system.off_diagonal is not None:
                y_faces_from_x_faces, x_faces_from_y_faces = system.off_diagonal
                coupled = ((y_faces_from_x_faces @ x_face_fields.ravel()).reshape(shape),
                           (x_faces_from_y_faces @ y_face_fields.ravel()).reshape(shape))
            y_face_fields *= system.weight_hi_y_faces
            x_face_fields *= system.weight_hi_x_faces
            if coupled is not None:   # E_x = y_scale (V D_x - X_xy x_faces), E_y = x_scale (V x_faces - X_yx D_x)
                y_face_fields -= coupled[0]
                x_face_fields -= coupled[1]
            y_face_fields *= y_scale
            x_face_fields *= x_scale
            self._face_fields = (y_face_fields, x_face_fields)
        return self._face_fields

    def error_indicator(self):
        """S7.7 hook (mesh plan A2): energy-weighted interpolation eta^2 parts (2, *window) of the shown solution (driven
        or the selected resonance): out-of-plane field and face fields, E x eps0 |eps| of the cells, H x mu0."""
        if self.solution is None or self.system is None:
            raise ValueError(f"em_fdfd_{self.polarization} '{self.name}' has no solution yet")
        system = self.system
        window = tuple(self.domain.physical_window())
        widths = [np.asarray(axis_widths, dtype=np.float64)[axis_window]
                  for axis_widths, axis_window in zip(self.domain.cell_widths(), window)]

        def host(values):
            return np.asarray(to_host(values))[window]

        y_face_fields, x_face_fields = self.face_fields()   # ez: (H_x, H_y), hz: (E_x, E_y)
        if self.polarization == 'ez':
            components = [(host(self.solution), EPS0 * np.abs(host(system.cell_eps))),
                          (host(y_face_fields), MU0), (host(x_face_fields), MU0)]
        else:
            components = [(host(self.solution), MU0),
                          (host(y_face_fields), EPS0 * np.abs(host(system.cell_eps_xx))),
                          (host(x_face_fields), EPS0 * np.abs(host(system.cell_eps_yy)))]
        return energy_interpolation_indicator(components, widths)

    # --- solve ---

    def _mark_system_dirty(self, ports=True, omega_only=False):
        """Reassemble and re-solve at the next solve; ports=False keeps the port modes (a change they do not see);
        omega_only: a new omega alone (planned wavelength batches stay valid)."""
        self._system_dirty = True
        self._solution_dirty = True
        if not omega_only:
            self._material_serial += 1
        if ports:
            self._port_modes_current = False

    def _check_solver_size(self):
        """sparse_direct size check of the full grid (ValueError before an oversized factorisation); line_block: its
        block-axis rule (ValueError for two periodic axes)."""
        if self._line_block is not None:
            fdfd_block_axis(*self._grid_shape_and_periodicity())
            return
        self.linear_solver.check_size(*self._grid_shape_and_periodicity())

    def _grid_shape_and_periodicity(self):
        """(nx, ny, periodic_x, periodic_y) of the full grid: the sparse_direct size and QR-order arguments."""
        grid, bc = self.domain.grid, self.boundary_config
        return grid.nx, grid.ny, bool(bc.periodic_x), bool(bc.periodic_y)

    def _mode_ports(self):
        """Sources and monitors following the port protocol (solve(physics, omega), warnings)."""
        return [port for port in (*self.sources, *self.monitors) if hasattr(port, 'solve')]

    def _solve_port_modes(self):
        """Re-solve the port modes at omega unless they are current (finalize already solved them)."""
        if self._port_modes_current:
            return
        for port in self._mode_ports():
            port.solve(self, self.omega)
        self._port_modes_current = True

    def record_monitors(self):
        """Record the last solve in every monitor (after _solve(record_monitors=False), e.g. a converged NLFD channel)."""
        if self.solution is None or self.system is None:
            raise ValueError("not solved yet: FDFDPhysics.record_monitors needs a solve first")
        for monitor in self.monitors:
            monitor.record(self, self.system)

    def _solve(self, record_monitors=True):
        """Solve at self.omega: reassemble and refactorise only when the system changed, sum the source currents,
        refresh the solution, q_abs and the state arrays, then record every monitor (unless record_monitors is False:
        an iterate of the nonlinear physics; n_solves counts it anyway)."""
        solve_start = time.perf_counter()
        xp = self.xp
        system = self._refresh_system(check_driven_size=True)
        self._solve_port_modes()

        rhs = xp.zeros(self._field_re.shape, dtype=xp.complex128)
        for source in self.sources:
            source.add_to_rhs(rhs, self, system)
        if system.pec_cells is not None:
            # ez: pinned rows (E_z = 0); hz: the uncoupled k0^2 mu row of a PEC cell then gives H_z = 0
            xp.copyto(rhs, rhs.dtype.type(0), where=system.pec_cells)
        for axis, outer, _, _, _ in self._wall_rows:   # pinned outermost wall / mirror cells
            cell_row(rhs, axis, outer)[...] = 0
        # Kept complex128: face fields, flux, q_abs and port amplitudes difference neighbours of it.
        solution = self._solve_linear(rhs.ravel()).reshape(rhs.shape)
        del rhs
        if self.solution is None:
            self.solution = xp.empty(solution.shape, dtype=xp.complex128)
        xp.copyto(self.solution, solution)
        del solution
        self.selected_resonance = None
        self._refresh_state(system)
        self._solution_dirty = False
        self.n_solves += 1
        if record_monitors:
            self.record_monitors()
        self.last_solve_seconds = time.perf_counter() - solve_start

    def _refresh_system(self, check_driven_size):
        """HarmonicSystem at self.omega, reassembled when dirty (old system and any non-recyclable factor freed first);
        check_driven_size: the back end's size rule per factorisation (scripts skip finalize, free memory changes)."""
        if self._system_dirty:
            self.system = None
            self._face_fields = None
            factorization = self._factorization
            if factorization is not None and not (factorization.recyclable and self._factored_omega == self.omega):
                self._factorization = None
                if _is_cudss(factorization):   # kept for its plan: the next factorisation refactorises it in place
                    self._cudss_previous = factorization
            if self._transposed_factorization is not None:
                self._transposed_previous, self._transposed_factorization = self._transposed_factorization, None
            self._factorization_current = False
            self._drop_line_factor()
            self._reference_factorization = None
            if check_driven_size:
                self._check_solver_size()
            self.system = self._assemble(self.omega)
            self._system_dirty = False
        return self.system

    def _refresh_state(self, system):
        """Wall rows of self.solution, then q_abs and the float32 state arrays from it (face-field cache reset)."""
        self._fill_wall_rows(system.pec_cells)
        self._face_fields = None
        self.solved_omega = system.omega
        if system.polarization == 'ez':
            power_density = absorbed_power_ez(system.omega, system.cell_eps, self.solution)
        else:
            ex_faces, ey_faces = self.face_fields()
            power_density = absorbed_power_hz(system.omega, system.cell_eps_xx, system.cell_eps_yy, ex_faces, ey_faces,
                                              system.periodic_x, system.periodic_y, system.face_fractions)
        xp = self.xp
        xp.copyto(self._q_abs, power_density)
        del power_density
        self._add_sheet_losses(system)
        xp.copyto(self._field_re, self.solution.real)
        xp.copyto(self._field_im, self.solution.imag)

    def _add_sheet_losses(self, system):
        """q_abs += 1/2 Re sigma_s |E_t|^2 / (line-cell width) on the line cells of every conducting sheet (ez: E_z of the
        cells; hz: the tangential E on their hi faces), so q_abs dA sums to the sheet's operator loss."""
        if not self.conducting_sheets:
            return
        face_fields = self.face_fields() if system.polarization == 'hz' else None
        for sheet in self.conducting_sheets:
            if face_fields is None:
                tangential = self.solution[sheet.line]
            else:   # E_x on the hi-y faces (sheet along x), E_y on the hi-x faces (sheet along y)
                tangential = face_fields[0 if sheet.axis == 'x' else 1][sheet.line]
            loss_scale = 0.5 * sheet.sigma(system.omega).real / sheet.cell_width
            self._q_abs[sheet.line] += (loss_scale * self.xp.abs(tangential) ** 2).astype(DTYPE)

    def _fill_wall_rows(self, fixed_cells):
        """Outermost wall / mirror cells (solved as 0) = sign x their inner neighbour, as the FDTD rows; x faces first,
        so a corner takes sign_x sign_y of its diagonal neighbour; the fixed PEC cells (HarmonicSystem.pec_cells) back
        to 0."""
        filled = False
        xp = self.xp
        for axis, outer, inner, _, sign in self._wall_rows:
            if sign != 0.0:
                xp.multiply(cell_row(self.solution, axis, inner), sign, out=cell_row(self.solution, axis, outer))
                filled = True
        if filled and fixed_cells is not None:
            xp.copyto(self.solution, self.solution.dtype.type(0), where=fixed_cells)

    def _drop_line_factor(self):
        """Release the line-block factor of the previous system (none without a line_block solver; a wavelength batch
        stays for its other omegas)."""
        if (self._line_factor is not None and self._line_factor != _REJECTED
                and self._line_factor is not self._batch_factor):
            self._line_factor.release()
        self._line_factor = None

    def _release_batch(self):
        """Drop the kept wavelength batch (its device inverses and operator caches)."""
        if self._batch_factor is not None:
            self._batch_factor.release()
        self._batch_factor = None
        self._batch_omegas = {}

    def _batch_factor_of_system(self):
        """(LineBlockFactor, shift index) of system.omega in a wavelength batch (a new batch of the next planned omegas
        from this one when it is planned but not in the kept batch), None: not planned, materials changed since the
        plan, or fewer than two omegas fit (the single-system factor then)."""
        omega = self.system.omega
        batch = self._batch_factor
        if batch is not None and (not batch.valid or self._batch_serial != self._material_serial):
            if not batch.valid and self._line_block.owner is not self:
                self._planned_omegas = ()   # a physics sharing the solver re-bound it: batches would refactor each omega
            self._release_batch()
        if self._batch_factor is not None and omega in self._batch_omegas:
            return self._batch_factor, self._batch_omegas[omega]
        if self._plan_serial != self._material_serial or omega not in self._planned_omegas:
            return None
        self._release_batch()   # before sizing the next batch: its inverses count as used memory
        upcoming = list(dict.fromkeys(self._planned_omegas[self._planned_omegas.index(omega):]))
        layout = self._line_layout
        host_operator = _host_sparse(self.system.operator)
        per_operator = operator_shift_bytes(layout.n_blocks, layout.block_size, layout.n_unknowns, host_operator.nnz,
                                            self._line_block.precision)
        n_batch = min(len(upcoming), MAX_BATCH_WAVELENGTHS,
                      self._line_block.chunk_size(len(upcoming), 1, layout, per_operator, kept=True))
        if n_batch < 2:
            return None
        batch_omegas = upcoming[:n_batch]   # the other omegas assembled here and again at their solve (deterministic)
        operators = [host_operator] + [_host_sparse(self._assemble(other).operator) for other in batch_omegas[1:]]
        self._batch_factor = self._line_block.factorize_operators(operators, layout, owner=self)
        self._batch_omegas = {value: index for index, value in enumerate(batch_omegas)}
        self._batch_serial = self._material_serial
        self.last_batch_size = n_batch
        return self._batch_factor, 0

    def _system_line_factor(self):
        """(LineBlockFactor, shift index) of system.operator: a planned wavelength batch entry, else bound and
        factorised on first use (re-bound when the shared solver holds another operator); None once this system was
        rejected."""
        if self._line_factor == _REJECTED:
            return None
        if self._line_factor is not None and not self._line_factor.valid:
            self._line_factor = None
        if self._line_factor is None:
            start = time.perf_counter()
            if self._line_layout is None:
                self._line_layout = fdfd_layout(*self._grid_shape_and_periodicity())
            batched = self._batch_factor_of_system()
            if batched is not None:
                self._line_factor, self._line_factor_index = batched
            else:
                self._line_block.set_operator(_host_sparse(self.system.operator), self._line_layout, (), owner=self)
                self._line_factor = self._line_block.factorize(np.zeros((1, 0), dtype=np.complex128))
                self._line_factor_index = 0
                self.last_batch_size = None
            if batched is None or batched[1] == 0:   # a reused batch entry factorised nothing
                self.last_factor_seconds = time.perf_counter() - start
        return self._line_factor, self._line_factor_index

    def _solve_line_block(self, flat_rhs):
        """x by the line-block factor of system.operator (_system_line_factor), None when it rejects the solve (this
        system then goes to the reference solver until it changes; run-log warning)."""
        kept = self._system_line_factor()
        if kept is None:
            return None
        solution, relative, accepted = kept[0].solve(to_host(flat_rhs), kept[1])
        if accepted:
            return self.xp.asarray(solution)
        self.n_line_block_fallbacks += 1
        self.runtime_warnings.append(
            f'{self._line_block.label()}: line-block relative residual {relative:.3g} above the accept tolerance '
            f'{self._line_block.accept_tolerance:g} at lambda = {2.0 * math.pi * C0 / self.system.omega * 1e9:.6g} nm; '
            'solved by a host sparse direct factor instead (precision double rarely rejects)')
        self._drop_line_factor()
        self._line_factor = _REJECTED
        return None

    def _adjoint_symmetrizer(self):
        """Host D of the current system when D A is complex symmetric (so A^T = D A D^-1), else None (Bloch phase,
        eps_xy, or a measured asymmetry above cudss_factor.SYMMETRY_TOLERANCE); checked once per system."""
        system = self.system
        if self._adjoint_symmetry is not None and self._adjoint_symmetry[0] is system:
            return self._adjoint_symmetry[1]
        symmetrizer = None if system.off_diagonal is not None else self._symmetrizer(system.omega)
        if symmetrizer is not None:
            import scipy.sparse
            from ....solvers.cudss_factor import SYMMETRY_TOLERANCE, relative_asymmetry
            operator = system.operator
            host = operator.get() if hasattr(operator, 'get') else operator
            if relative_asymmetry(scipy.sparse.diags(symmetrizer) @ host) > SYMMETRY_TOLERANCE:
                symmetrizer = None
        self._adjoint_symmetry = (system, symmetrizer)
        return symmetrizer

    def _line_block_adjoint(self, flat_rhs):
        """x of operator^T x = flat_rhs on the kept line-block factor of the operator: x = D solve(flat_rhs / D) when
        D A is complex symmetric; None otherwise or on a rejection (the reference factor solves it)."""
        symmetrizer = self._adjoint_symmetrizer()
        if symmetrizer is None:
            return None
        scale = self.xp.asarray(symmetrizer)
        solution = self._solve_line_block(flat_rhs / scale)
        return None if solution is None else solution * scale

    def _solve_reference(self, flat_rhs, transpose):
        """x of the line_block physics' private SparseDirectSolver ('auto'): one factor per system, transposed too."""
        if self._reference_solver is None:
            self._reference_solver = SparseDirectSolver(backend=self.domain.backend.name,
                                                        label=lambda: f'{self._line_block.label()} reference')
            self._private_solvers.append(self._reference_solver)
        if self._reference_factorization is None:
            self._reference_solver.check_size(*self._grid_shape_and_periodicity())
            self._reference_factorization = self._reference_solver.factorize(self.system.operator,
                                                                             *self._grid_shape_and_periodicity())
            self.host_factor_bytes = self._reference_factorization.host_bytes
        return self._reference_factorization.solve(flat_rhs, transpose=transpose)

    def _solve_linear(self, flat_rhs, transpose=False):
        """x of system.operator x = flat_rhs (transpose: operator^T, the adjoint solve): line_block: its factor (a
        transposed or rejected solve: the reference sparse direct factor); sparse_direct: GMRES preconditioned by a
        SuperLU factor of an earlier operator at this omega while it reaches sparse_direct.RECYCLE_RTOL, else
        (re)factorise and solve directly."""
        if not transpose:
            self.last_recycled_iterations = None
        if self._line_block is not None:
            if transpose:
                solution = self._line_block_adjoint(flat_rhs)
            else:
                solution = self._solve_line_block(flat_rhs)
                self.last_linear_text = LINE_BLOCK_TEXT if solution is not None else _SPARSE_DIRECT_TEXT
            return solution if solution is not None else self._solve_reference(flat_rhs, transpose)
        if self._factorization is not None and not self._factorization_current:
            initial_guess = self.solution if self.solved_omega == self.omega and not transpose else None
            solution, iterations = self.linear_solver.solve_recycled(self._factorization, self.system.operator,
                                                                     flat_rhs, initial_guess, transpose=transpose)
            if solution is not None:
                if not transpose:
                    self.last_recycled_iterations = iterations
                return solution
            self._factorization = None   # too far from the factored operator: refactorise
        if self._factorization is None:
            options = {}
            if self.linear_solver.back_end in CUDSS_BACK_ENDS:   # LDL^T of D A (LU when not symmetric), plan reused
                options = {'previous': self._cudss_previous, 'symmetric': True,
                           'symmetrizer': self._symmetrizer(self.system.omega)}
                self._cudss_previous = None
            elif self.linear_solver.back_end in HOST_DIRECT_BACK_ENDS:   # host LDL^T of D A (LU when not symmetric):
                self._release_kept_cudss()   # recyclable and transposes natively, like SuperLU
                options = {'symmetric': True, 'symmetrizer': self._symmetrizer(self.system.omega)}
            else:   # SuperLU (also after an 'auto' runtime fell back from cuDSS: its kept factors are freed)
                self._release_kept_cudss()
            self._factorization = self.linear_solver.factorize(self.system.operator,
                                                               *self._grid_shape_and_periodicity(), **options)
            self.last_factor_seconds = self.linear_solver.last_factor_seconds
            self.host_factor_bytes = self._factorization.host_bytes
            self._factored_omega = self.system.omega
            self._factorization_current = True
        if transpose and _is_cudss(self._factorization) and not self._factorization.symmetric:
            return self._transposed_solve(flat_rhs)
        return self._factorization.solve(flat_rhs, transpose=transpose)

    def _release_kept_cudss(self):
        """Release the cuDSS factors kept only for their plan (_cudss_previous, _transposed_previous)."""
        for name in ('_cudss_previous', '_transposed_previous'):
            factorization = getattr(self, name)
            if factorization is not None:
                factorization.release()
                setattr(self, name, None)

    def _transposed_solve(self, flat_rhs):
        """x of system.operator^T x = flat_rhs by a cuDSS LU of the transpose (the adjoint solve when the driven factor
        is a general LU: Bloch phase, hz eps_xy couplings), factorised on first use per system, plan reused."""
        if self._transposed_factorization is None:
            previous, self._transposed_previous = self._transposed_previous, None
            self._transposed_factorization = self.linear_solver.factorize(
                self.system.operator.T.tocsr(), *self._grid_shape_and_periodicity(), previous=previous)
        return self._transposed_factorization.solve(flat_rhs)

    def solve_summary(self):
        """One line on the last solve for the study log (back end, unknowns, wall times, factor size); None before."""
        if self.last_solve_seconds is None:
            return None
        grid = self.domain.grid
        linear = self.linear_solver
        if self._line_block is not None:
            layout = self._line_layout
            blocks = '' if layout is None else f', {layout.n_blocks} blocks of b = {layout.block_size}'
            text = (f'{grid.nx * grid.ny:,} unknowns, line-block solver ({linear.precision}{blocks}), last solve '
                    f'{self.last_linear_text}, solve {self.last_solve_seconds:.3g} s')
            if self.last_factor_seconds is not None:
                batch = ('' if self.last_batch_size is None
                         else f', one batch of {self.last_batch_size} wavelengths: later ones in it are solve-only')
                text += f' (last line-block factorisation {self.last_factor_seconds:.3g} s{batch})'
            if self.n_line_block_fallbacks:
                text += f'; {self.n_line_block_fallbacks} system(s) rejected and solved by host sparse direct'
            return text
        text = (f'{grid.nx * grid.ny:,} unknowns, sparse direct back end {linear.back_end}, solve '
                f'{self.last_solve_seconds:.3g} s')
        if linear.back_end == 'cusolver_qr':   # keyed on the back end: a device factor (cuDSS) also has no host bytes
            workspace = qr_workspace_bytes(*self._grid_shape_and_periodicity())
            return f'{text} (GPU QR refactorised in every solve, estimated workspace {format_bytes(workspace)})'
        if linear.n_recycled_solves:   # counts of the solver (shared by the NLFD channels)
            last = (f'{self.last_recycled_iterations} GMRES iterations'
                    if self.last_recycled_iterations is not None else 'factorised')
            text = (f'{text}; {linear.n_factorizations} factorisations, {linear.n_recycled_solves} solves '
                    f'GMRES-preconditioned by an earlier factor; last solve {last}')
        factor_seconds = 'n/a' if self.last_factor_seconds is None else f'{self.last_factor_seconds:.3g} s'
        if linear.back_end == 'scipy_splu':
            return (f'{text} (SuperLU factorisation {factor_seconds}, L + U factors '
                    f'{format_bytes(self.host_factor_bytes or 0)} of host memory)')
        description = getattr(self._factorization, 'description', None) or linear.back_end
        if linear.back_end in HOST_DIRECT_BACK_ENDS:   # library factor outside numpy: its own memory report
            return (f'{text} ({description} factorisation {factor_seconds}, factor '
                    f'{format_bytes(self.host_factor_bytes or 0)} of host memory)')
        if linear.back_end in CUDSS_BACK_ENDS:   # counts of the role solver (adjoint transposes included)
            return (f'{text} ({description} factorisation {factor_seconds}; {linear.n_factorizations} cuDSS '
                    f'factorisations on {linear.n_cudss_plans} plan(s))')
        return f'{text} ({description} factorisation {factor_seconds})'

    def work_counters(self):
        """Solves so far (summary 'solvers' physics counters), plus the counters of private (script) linear / eigen
        solvers; a build's role solvers report their own."""
        counters = {'solves': self.n_solves}
        if self._line_block is not None:   # the reference sparse direct (rejected / adjoint solves) is private
            counters['line_block_fallbacks'] = self.n_line_block_fallbacks
        for solver in self._private_solvers:
            counters.update(solver.work_counters())
        return counters

    def _assemble(self, omega):
        """HarmonicSystem of the current materials (+ extra_permittivity) at omega: A1 operators with the complex PML
        stretch, complex128 Helmholtz operator (ez: PEC cells pinned; hz: PEC face weights 0; wall / mirror rows:
        _WALL_SIGNS; conformal_pec weights); conducting sheets enter the operator only, not cell / face eps."""
        domain = self.domain
        materials = domain.materials
        xp = self.xp
        bc = self.boundary_config
        periodic_x, periodic_y = bool(bc.periodic_x), bool(bc.periodic_y)
        cell_dx, cell_dy = grid_cell_widths(domain.grid)
        if self._yee_ops is None or self._yee_ops[0] != omega:
            self._yee_ops = None   # free the previous omega's operators first
            polar = self.metric is not None   # polar: PML cells graded on the outermost r width (no mesh_config)
            pml_dx = float(cell_dx[-1]) if polar else cpml_grading_cell_size(domain)
            self._yee_ops = (omega, yee_operators((cell_dx, cell_dy), bc, pml_dx, omega=omega, xp=xp,
                                                  polar_radius_min=domain.coordinates.radius_min if polar else None))
        ops = self._yee_ops[1]
        k0_squared = (omega / C0) ** 2
        cell_mu = cell_permeability(materials, omega, xp=xp)
        pec_cells = self.pec_cells
        conformal = self._conformal_arrays()
        if self.polarization == 'ez':
            cell_eps = cell_permittivity(materials, 'eps_r', self.dispersive_regions, omega, xp=xp)
            self._add_extra_permittivity((cell_eps,))
            if self.subpixel_smoothing is not None:   # averaged background eps on the interface cells
                self.subpixel_smoothing.add_deltas('cells', cell_eps, materials)
            # W = 1 / mu_face on both face sets, mu_face = the face's lower cell = the cell itself (A3 convention)
            weight_hi_x_faces = 1.0 / cell_mu
            if conformal is not None:   # E_z = 0 at the drawn wall: each link into the metal over t d (conformal_pec)
                weight_hi_y_faces = weight_hi_x_faces * conformal.face_scale_hi_y
                weight_hi_x_faces *= conformal.face_scale_hi_x
            else:
                weight_hi_y_faces = weight_hi_x_faces.copy() if self._wall_rows else weight_hi_x_faces
            self._zero_even_wall_faces(weight_hi_x_faces, weight_hi_y_faces)
            cell_term = k0_squared * cell_eps
            for sheet in self.conducting_sheets:   # diagonal -j omega mu0 sigma_s / w_perp on the line cells
                cell_term[sheet.line] += k0_squared * sheet.permittivity_change(omega)
            operator = helmholtz_operator(ops, weight_hi_x_faces, weight_hi_y_faces, cell_term, xp=xp)
            del cell_term
            operator = self._pin_outer_cells(operator, pec_cells)
            return HarmonicSystem(
                polarization='ez', omega=omega, ops=ops, operator=operator, weight_hi_x_faces=weight_hi_x_faces,
                weight_hi_y_faces=weight_hi_y_faces, cell_eps=cell_eps, cell_eps_xx=None, cell_eps_yy=None,
                cell_mu=cell_mu, eps_hi_x_faces=None, eps_hi_y_faces=None, pec_cells=pec_cells,
                periodic_x=periodic_x, periodic_y=periodic_y)

        cell_eps_xx = cell_permittivity(materials, 'eps_xx', self.dispersive_regions, omega, xp=xp)
        cell_eps_yy = cell_permittivity(materials, 'eps_yy', self.dispersive_regions, omega, xp=xp)
        self._add_extra_permittivity((cell_eps_xx, cell_eps_yy))
        cell_eps_xy = self.assembly_eps_xy()   # kept on the system: the adjoint chain rule reads it
        if cell_eps_xy is not None and self.metric is not None:
            raise NotImplementedError('FDFDPhysics: eps_xy on a polar grid is not supported (its face couplings use '
                                      'Cartesian cell areas; components there are radial / angular, so a Cartesian '
                                      'crystal is off-diagonal everywhere)')
        along_x, along_y, eta_xy = cell_eps_xx, cell_eps_yy, None
        if cell_eps_xy is not None:   # E_x / E_y see 1 / eta_xx, 1 / eta_yy along their faces (harmonic.py header)
            squared = cell_eps_xy * cell_eps_xy
            along_x = cell_eps_xx - squared / cell_eps_yy
            along_y = cell_eps_yy - squared / cell_eps_xx
            eta_xy = -cell_eps_xy / (cell_eps_xx * cell_eps_yy - squared)
            del squared
        eps_hi_y_faces = face_average(along_x, 1, periodic_y, cell_dy, xp=xp)   # eps_xx at E_x
        eps_hi_x_faces = face_average(along_y, 0, periodic_x, cell_dx, xp=xp)   # eps_yy at E_y
        del along_x, along_y
        if self.subpixel_smoothing is not None:   # averaged background eps on the interface faces (as the FDTD)
            self.subpixel_smoothing.add_deltas('faces_y', eps_hi_y_faces, materials)
            self.subpixel_smoothing.add_deltas('faces_x', eps_hi_x_faces, materials)
        weight_hi_x_faces = 1.0 / eps_hi_x_faces   # V_x
        weight_hi_y_faces = 1.0 / eps_hi_y_faces   # V_y
        for sheet in self.conducting_sheets:   # tangential face: eps_face - j sigma_s / (omega eps0 w_dual)
            sheet_weights = weight_hi_y_faces if sheet.axis == 'x' else weight_hi_x_faces   # E_x / E_y faces
            sheet_weights[sheet.line] = 1.0 / (1.0 / sheet_weights[sheet.line] + sheet.permittivity_change(omega))
        face_fractions = cell_area_fractions = None
        if conformal is not None:   # Dey-Mittra: only the metal cells without a free face stay H_z = 0 rows
            pec_cells = conformal.fixed_cells
            face_fractions = (conformal.fraction_hi_x, conformal.fraction_hi_y)
            cell_area_fractions = conformal.area_fraction
            for weights, fractions in zip((weight_hi_x_faces, weight_hi_y_faces), face_fractions):
                xp.copyto(weights, weights.dtype.type(0), where=fractions == 0.0)   # E faces wholly in the metal
        if pec_cells is not None:
            xp.copyto(weight_hi_x_faces, weight_hi_x_faces.dtype.type(0),
                      where=face_pec(pec_cells, 0, periodic_x, xp=xp))
            xp.copyto(weight_hi_y_faces, weight_hi_y_faces.dtype.type(0),
                      where=face_pec(pec_cells, 1, periodic_y, xp=xp))
        self._zero_even_wall_faces(weight_hi_x_faces, weight_hi_y_faces)
        if conformal is None:
            operator = helmholtz_operator(ops, weight_hi_x_faces, weight_hi_y_faces, k0_squared * cell_mu, xp=xp)
        else:   # area-scaled: K (free lengths l / L on the faces) - k0^2 mu diag(A_free), so still complex symmetric
            operator = helmholtz_operator(ops, weight_hi_x_faces * face_fractions[0],
                                          weight_hi_y_faces * face_fractions[1],
                                          k0_squared * cell_mu * cell_area_fractions, xp=xp)
        off_diagonal = None
        if eta_xy is not None:   # E faces with weight 0 (PEC, even walls) and never-updated last faces carry no E
            active_hi_x_faces, active_hi_y_faces = active_coupling_faces(weight_hi_x_faces, weight_hi_y_faces,
                                                                         periodic_x, periodic_y)
            off_diagonal = off_diagonal_face_couplings((cell_dx, cell_dy), (periodic_x, periodic_y),
                                                       self.lower_wrap_factors, eta_xy, active_hi_x_faces,
                                                       active_hi_y_faces, xp=xp)
            del eta_xy, active_hi_x_faces, active_hi_y_faces
            y_faces_from_x_faces, x_faces_from_y_faces = off_diagonal
            operator = (operator - ops.dx_e @ x_faces_from_y_faces @ ops.dy_h
                        - ops.dy_e @ y_faces_from_x_faces @ ops.dx_h).tocsr()
        operator = self._pin_outer_cells(operator, None)
        return HarmonicSystem(
            polarization='hz', omega=omega, ops=ops, operator=operator, weight_hi_x_faces=weight_hi_x_faces,
            weight_hi_y_faces=weight_hi_y_faces, cell_eps=None, cell_eps_xx=cell_eps_xx, cell_eps_yy=cell_eps_yy,
            cell_mu=cell_mu, eps_hi_x_faces=eps_hi_x_faces, eps_hi_y_faces=eps_hi_y_faces, pec_cells=pec_cells,
            periodic_x=periodic_x, periodic_y=periodic_y, off_diagonal=off_diagonal, cell_eps_xy=cell_eps_xy,
            face_fractions=face_fractions, cell_area_fractions=cell_area_fractions)

    def assembly_eps_xy(self):
        """hz float64 (nx, ny) eps_xy the assembly uses = the material eps_xy + the eps_xy_update change, 0 in PEC cells
        and in the outer two cell rows of every wall / mirror face (pinned images assume 5 points); None when all 0."""
        materials = self.domain.materials
        parts = [array for array in ((getattr(materials, OFF_DIAGONAL_ARRAY)
                                      if OFF_DIAGONAL_ARRAY in materials.property_names() else None),
                                     self._coupled_eps_xy) if array is not None]
        if not parts:
            return None
        xp = self.xp
        cell_eps_xy = parts[0].astype(xp.float64)
        for part in parts[1:]:
            cell_eps_xy += part
        if self.pec_cells is not None:
            xp.copyto(cell_eps_xy, 0.0, where=self.pec_cells)
        for axis, outer, inner, _, _ in self._wall_rows:
            cell_row(cell_eps_xy, axis, outer)[...] = 0.0
            cell_row(cell_eps_xy, axis, inner)[...] = 0.0
        if not bool(xp.any(cell_eps_xy != 0.0)):   # one host sync per assembly: an all-zero eps_xy keeps 5 points
            return None
        return cell_eps_xy

    def _add_extra_permittivity(self, cell_eps_arrays):
        """Add extra_permittivity in place to the assembly's cell eps(omega) (ez (eps,), hz (eps_xx, eps_yy))."""
        if self._extra_permittivity is None:
            return
        box, values = self._extra_permittivity
        for cell_eps, extra in zip(cell_eps_arrays, values):
            if extra is not None:
                region = cell_eps if box is None else cell_eps[box]   # a view: += writes into cell_eps
                region += extra

    def _zero_even_wall_faces(self, weight_hi_x_faces, weight_hi_y_faces):
        """Face weight 0 on the inner face of each even (sign +1) wall / mirror row: zero flux, E (hz) / H (ez) = 0."""
        for axis, _, _, inner_face, sign in self._wall_rows:
            if sign == 1.0:
                cell_row(weight_hi_x_faces if axis == 0 else weight_hi_y_faces, axis, inner_face)[...] = 0

    def _pinned_cells(self, pec_cells):
        """New run-backend bool (nx, ny) of the cells solved as identity rows: pec_cells (None = none) and the
        outermost wall / mirror cells."""
        xp = self.xp
        pinned = xp.zeros(self._field_re.shape, dtype=xp.bool_) if pec_cells is None else pec_cells.copy()
        for axis, outer, _, _, _ in self._wall_rows:
            cell_row(pinned, axis, outer)[...] = True
        return pinned

    def _pin_outer_cells(self, operator, pec_cells):
        """pin_image_cells of the outermost wall / mirror cells (u = sign u_inner) and of `pec_cells` (ez, sign 0);
        the operator itself when there is neither."""
        if not self._wall_rows and pec_cells is None:
            return operator
        xp = self.xp
        pinned = self._pinned_cells(pec_cells)
        image_signs = xp.zeros(self._field_re.shape, dtype=xp.float64)
        for axis, outer, _, _, sign in self._wall_rows:
            cell_row(image_signs, axis, outer)[...] = sign
        if pec_cells is not None:   # an E_z PEC cell on a wall row is 0 (the FDTD's PEC keep), no image
            xp.copyto(image_signs, 0.0, where=pec_cells)
        return pin_image_cells(operator, pinned, image_signs, xp=xp)

    # --- material helpers ---

    def add_dispersive(self, pole, mask):
        """Record a Drude / Lorentz pole on a bool cell mask of the run backend (None = everywhere) for the assembly (no
        ADE); the permittivity arrays stay the background."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f"add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}")
        if mask is not None:
            _check_cell_array(mask, self._field_re.shape, self.xp.bool_, 'Pole mask', xp=self.xp)
        record_dispersive_region(self.dispersive_regions, pole, mask)
        self._mark_system_dirty()

    def add_pec_region(self, mask):
        """Perfect electric conductor on a bool (nx, ny) cell mask of the run backend."""
        _check_cell_array(mask, self._field_re.shape, self.xp.bool_, 'PEC mask', xp=self.xp)
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self.conformal_pec = None   # the free cells changed
        self._mark_system_dirty()

    def set_conformal_boundaries(self, locator):
        """solver_defaults.conformal_boundaries: pec_domain features call add_conformal_pec_region with
        locator.inside(entities) (model.boundary_locator.GeometryLocator, duck-typed); None = staircased PEC."""
        self.conformal_locator = locator

    def add_conformal_pec_region(self, mask, inside):
        """PEC on the run-backend bool mask whose drawn outline is inside(x, y) -> bool (host): pec_cells now (ports,
        sources read the staircase), the assembly by conformal_pec (ez: wall distances, hz: free lengths / areas)."""
        self._require_cartesian('conformal PEC')
        self.add_pec_region(mask)
        self._conformal_mask = add_pec_cells(self._conformal_mask, mask)
        self._conformal_tests.append(inside)

    def _conformal_arrays(self):
        """conformal_pec.ConformalFDFD of the conformal regions on the run backend (one host build per PEC change), None
        without one."""
        if not self._conformal_tests:
            return None
        if self.conformal_pec is None:
            config = self.boundary_config
            host_arrays = build_conformal_fdfd(self.domain, self.polarization, to_host(self.pec_cells),
                                               to_host(self._conformal_mask), union_test(self._conformal_tests),
                                               (bool(config.periodic_x), bool(config.periodic_y)))
            self.conformal_pec = host_arrays.on_backend(self.xp)
        return self.conformal_pec

    @property
    def extra_permittivity(self):
        """None or (box slices | None = full grid, ez (array,) | hz (xx | None, yy | None)) complex128 relative eps added
        to the cell eps(omega) of the assembly before the face averaging (set_extra_permittivity)."""
        return self._extra_permittivity

    def set_extra_permittivity(self, values, box=None):
        """Physics-private permittivity change on `box` (2 slices; None = full grid), never written into the
        MaterialGrid (e.g. an NLFD channel's Kerr eps): ez one complex xp array, hz (xx, yy) (None = no change);
        None clears it. Sources and q_abs see it (Im < 0 = loss); port modes do not (they stay current)."""
        if values is None:
            self._extra_permittivity = None
            self._mark_system_dirty(ports=False)
            return
        values = (values,) if self.polarization == 'ez' else tuple(values)
        if len(values) != len(self.permittivity_arrays):
            raise ValueError(f"extra_permittivity of an {self.polarization!r} physics needs "
                             f"{len(self.permittivity_arrays)} array(s) ({', '.join(self.permittivity_arrays)}), got "
                             f"{len(values)}")
        grid_shape = self._field_re.shape
        if box is None:
            box_shape = grid_shape
        else:
            box = tuple(box)
            if len(box) != 2 or not all(isinstance(part, slice) and part.step in (None, 1) for part in box):
                raise ValueError(f"extra_permittivity box must be two step-1 slices, got {box!r}")
            box_shape = tuple(len(range(*part.indices(size))) for part, size in zip(box, grid_shape))
        xp = self.xp
        for extra in values:
            if extra is not None and (not isinstance(extra, xp.ndarray) or extra.shape != box_shape):
                raise ValueError(f"extra_permittivity arrays must be {xp.__name__} arrays of the box shape {box_shape}, "
                                 f"got {type(extra).__name__} {getattr(extra, 'shape', None)}")
        self._extra_permittivity = (box, tuple(None if extra is None else extra.astype(xp.complex128, copy=False)
                                               for extra in values))
        self._mark_system_dirty(ports=False)

    def add_conducting_sheet(self, axis, position, span=None, conductivity=0j):
        """Zero-thickness conducting sheet (2D material) on an axis-aligned line (axis 'x' = along x at y = position;
        span as add_line_source): only the tangential E conducts, `conductivity` = sigma_s [S] (complex number) or a
        callable omega -> complex (sheet_conductivity.SheetConductivity); exp(+j omega t), inductive Im < 0."""
        self._require_cartesian('add_conducting_sheet')
        if not callable(conductivity):
            conductivity = SheetConductivity('constant', value=complex(conductivity))
        line = sheet_line(self.domain, self.boundary_config, self.polarization, axis, position, span,
                          "Conducting sheet")
        dx_cells, dy_cells = grid_cell_widths(self.domain.grid)
        cell_width = float((dy_cells if axis == 'x' else dx_cells)[line.line_cell])
        # ez: E_z at the line-cell centres, the delta spread over the cell (Yee width); hz: the tangential E on their
        # hi faces, spread over the dual face spacing (sheet_line's normal_width there)
        normal_width = cell_width if self.polarization == 'ez' else line.normal_width
        sheet = ConductingSheet(axis, line.line_cell, line.along, normal_width, cell_width, conductivity)
        sheet.sigma(self.omega)   # fail now on a non-finite model value
        self.conducting_sheets.append(sheet)
        self._mark_system_dirty(ports=False)   # port modes do not see a sheet
        return sheet

    # --- sources (FDTD amplitude meanings; phase [deg], 0 = cos carrier) ---

    def _add_source(self, source):
        self.sources.append(source)
        self._solution_dirty = True
        if hasattr(source, 'solve'):
            self._port_modes_current = False
        return source

    def add_point_source(self, x, y, current=1e-3, phase=0.0):
        """z-directed filament current `current` [A] at (x, y) (ez only): J = I e^{j phase} / (dx_i dy_j) (polar: (x,
        y) = (r - radius_min, phi), the sector area r_c dr dphi)."""
        require_polarization(self.polarization, 'ez', 'add_point_source')
        ix, iy, cell_area = self._point_cell(x, y, "Point source")
        return self._add_source(PointCurrentPhasor(ix, iy, current, phase, cell_area))

    def add_magnetic_line_current(self, x, y, magnetic_current=1.0, phase=0.0):
        """z-directed magnetic line current K_m = `magnetic_current` [V] at (x, y) (hz only; polar as add_point_source);
        in a uniform medium |H_z| = omega eps0 eps_r K_m / 4 |H0^(2)(k r)|."""
        require_polarization(self.polarization, 'hz', 'add_magnetic_line_current')
        ix, iy, cell_area = self._point_cell(x, y, "Magnetic line current")
        return self._add_source(MagneticCurrentPhasor(ix, iy, magnetic_current, phase, cell_area))

    def _point_cell(self, x, y, label):
        """(ix, iy, cell area [m^2]) of a point source: placement.point_source_cell, the metric area on a polar grid."""
        ix, iy, cell_area = point_source_cell(self.domain, x, y, label)
        if self.metric is not None:
            cell_area = float(self.metric.volume.along_x[ix] * self.metric.volume.along_y[iy])
        return ix, iy, cell_area

    def _require_cartesian(self, label):
        """NotImplementedError on a polar Domain for a feature whose placement / weights assume Cartesian cells."""
        if self.metric is not None:
            raise NotImplementedError(
                f'FDFDPhysics: {label} is not available on a polar (r, phi) grid yet (its placement or weights use '
                'Cartesian cell widths); polar runs take point / magnetic line currents, probes, flux monitors, PEC, '
                'poles and the eigenfrequency solve')

    def add_electric_dipole(self, x, y, dipole_moment=1e-3, orientation=0.0, phase=0.0):
        """In-plane electric dipole (hz only), M = `dipole_moment` [A] along `orientation` [deg, ccw from +x] on the
        faces around the cell; |H_z| = k M/4 |H1^(2)(k r)| |sin(phi - theta)|."""
        require_polarization(self.polarization, 'hz', 'add_electric_dipole')
        self._require_cartesian('add_electric_dipole')
        ix, iy, _ = point_source_cell(self.domain, x, y, "Electric dipole")
        faces = electric_dipole_faces(self.domain, self.boundary_config, ix, iy, dipole_moment, orientation)
        # The lo face of cell 0 wraps to the last face, one period up: Bloch factor e^{-j k L} = 1 / lower wrap factor.
        phased_faces = []
        for component, i, j, current_density in faces:
            if component == 'ey' and i > ix:
                current_density /= self.lower_wrap_factors[0]
            elif component == 'ex' and j > iy:
                current_density /= self.lower_wrap_factors[1]
            phased_faces.append((component, i, j, current_density))
        return self._add_source(ElectricDipolePhasor(ix, iy, phased_faces, phase))

    def add_line_source(self, axis, position, amplitude=1.0, phase=0.0, span=None):
        """Sheet current radiating `amplitude` [V/m] to each side in the local medium; axis 'x' = line along x at
        y = position. span = (start, end) physical -> cells [cell(start), cell(end)+1), an end on a cpml window edge
        continued through the PML; None = whole grid line."""
        return self._add_sheet_current(axis, position, amplitude, phase, span, "Line source")

    def add_gaussian_beam(self, axis, position, waist, center, amplitude=1.0, phase=0.0, span=None):
        """Line source with transverse factor exp(-((s - center)/waist)^2); `amplitude` = peak field [V/m] on the
        waist line."""
        if not waist > 0:
            raise ValueError(f"Gaussian beam waist must be > 0, got {waist}")
        return self._add_sheet_current(axis, position, amplitude, phase, span, "Gaussian beam", waist=waist,
                                       center=center)

    def _add_sheet_current(self, axis, position, amplitude, phase, span, label, waist=None, center=None):
        """ez: J_z through the line cells' centres; hz: J_x (axis 'x') / J_y (axis 'y') on their hi faces; Bloch k != 0:
        times e^{-j k.r} along the line (sources.bloch_line_phase, as the FDTD Bloch line sources)."""
        self._require_cartesian(label)
        sheet = sheet_line(self.domain, self.boundary_config, self.polarization, axis, position, span, label,
                           waist=waist, center=center)
        line_phase = None
        if self.boundary_config.has_bloch_phase:
            line_phase = bloch_line_phase(self.domain.grid, self.boundary_config, axis, sheet.line_cell,
                                          sheet.span_cells)
        return self._add_source(SheetCurrentPhasor(axis, sheet, amplitude, phase, self.polarization, line_phase,
                                                   label=label))

    def add_tfsf(self, x0, y0, x1, y1, amplitude=1.0, phase=0.0):
        """+x plane wave of incident peak `amplitude` [V/m] (E_z on ez; in-plane |E| on hz, H_z = n |E| / eta0) inside
        the box (uniform or graded grids; box edges in one uniform background medium, else finalize warns; on a periodic
        y axis the box may span every y cell), scattered field outside it; a Bloch ky != 0 makes it the oblique wave at
        ky (uniform y cells only: ValueError otherwise)."""
        self._require_cartesian('add_tfsf')
        d = self.domain
        bc = self.boundary_config
        periodic_y = bc.periodic_y
        # a box reaching a y mirror face of the wave's parity opens onto it: the pinned ghost row and the zero face
        # weight of an even mirror (_WALL_SIGNS +1) leave no correction there, so nothing else changes
        cells = tfsf_box_cells(d, x0, y0, x1, y1, full_period_y=periodic_y, boundary_config=bc,
                               polarization=self.polarization)
        ix0, ix1, iy0, iy1 = cells
        full_period_y = periodic_y and iy0 == 0 and iy1 == d.grid.ny
        return self._add_source(PlaneWavePhasor(ix0, ix1, iy0, iy1, amplitude, phase, self.polarization,
                                                grid_cell_widths(d.grid), full_period_y=full_period_y,
                                                bloch_ky=bc.bloch_ky if bc.bloch_y else 0.0,
                                                open_faces=tfsf_open_faces(bc, box_window_faces(d, cells),
                                                                           self.polarization)))

    def add_mode_source(self, axis, position, span, power=1.0, mode_number=1, direction='positive', phase=0.0):
        """One-sided source of slab mode `mode_number` carrying `power` [W/m] toward +normal ('positive') or -normal on
        a line (axis 'x' = along x at y = position); span = (start, end) along the line [m]."""
        self._require_cartesian('add_mode_source')
        power = float(power)
        if not (math.isfinite(power) and power > 0):
            raise ValueError(f"mode source power must be finite and > 0 [W/m], got {power}")
        line_cell, span_cells, label = port_cells(self.domain, axis, position, span, 'Mode source')
        return self._add_source(ModeSourcePhasor(self, axis, line_cell, span_cells, power, mode_number, direction,
                                                 phase, label))

    # --- monitors ---

    def _add_monitor(self, monitor):
        """Append a monitor (its device arrays moved to the run backend first); it records the last solve at once when
        there is one."""
        if hasattr(monitor, 'use_array_module'):
            monitor.use_array_module(self.xp)
        self.monitors.append(monitor)
        if hasattr(monitor, 'solve'):
            self._port_modes_current = False
        if self.solution is not None and self.system is not None:
            monitor.record(self, self.system)
        return monitor

    def add_field_probe(self, x, y, field_component=None):
        """Complex phasor of one component at the cell containing (x, y) per solve; None = the out-of-plane field."""
        ix, iy, field_component = probe_cell(self.domain, self.polarization, x, y, field_component)
        return self._add_monitor(HarmonicProbe(ix, iy, field_component, COMPONENT_UNITS[field_component]))

    def add_flux_monitor(self, axis, position, span):
        """Time-averaged Poynting power [W/m] through a line (axis 'x' = along x at y = position, as add_line_source;
        > 0 toward +y); span = (start, end) along the line [m]. The monitor's own `axis` is the normal. Polar: axis 'x'
        = a ray at phi = position (power toward +phi), axis 'y' = the arc through the hi r faces of the line cells
        (outward; arc lengths r_f dphi, so a closed circle gives the exact discrete power balance)."""
        normal_axis, line_cell, span_cells, along_widths = flux_line_cells(self.domain, axis, position, span)
        if self.metric is not None and normal_axis == 'x':
            along_widths = self.metric.x_face_area.along_x[line_cell + 1] * self.metric.x_face_area.along_y
        return self._add_monitor(HarmonicFluxMonitor(normal_axis, line_cell, span_cells, along_widths,
                                                     self.polarization))

    def add_mode_monitor(self, axis, position, span, mode_number=1):
        """Powers of slab mode `mode_number` travelling toward +normal / -normal through a line (axis as
        add_mode_source) per solve (fdfd_ports.harmonic_mode_monitor_result)."""
        self._require_cartesian('add_mode_monitor')
        line_cell, span_cells, label = port_cells(self.domain, axis, position, span, 'Mode monitor')
        return self._add_monitor(HarmonicModeMonitor(self, axis, line_cell, span_cells, mode_number, label))

    def add_near_to_far(self, x0, y0, x1, y1):
        """Near-to-far-field box: the out-of-plane field and the tangential in-plane field (face mean) of every solve
        sampled on the centres of columns x0, x1 and rows y0, y1 of the half-open cell box, corners included."""
        self._require_cartesian('add_near_to_far')
        d = self.domain
        ix0, ix1, iy0, iy1 = near_to_far_box_cells(d, x0, y0, x1, y1)
        return self._add_monitor(HarmonicNearToFar(ix0, ix1, iy0, iy1, d.cell_centers(), polarization=self.polarization,
                                                   mirrors=mirror_planes(d, self.boundary_config)))

    # --- adjoint sensitivity (adjoint.py) ---

    def add_objective(self, axis, position, span, quantity='flux', mode_number=1, weight=1.0):
        """Objective term weight * value on a line (axis as add_flux_monitor): 'flux' [W/m] through it, or
        'mode_power_positive' / 'mode_power_negative' = |a|^2 [W/m] of mode_number; J sums every term."""
        self._require_cartesian('add_objective')
        if quantity not in OBJECTIVE_QUANTITIES:
            raise ValueError(f"objective quantity must be one of {', '.join(OBJECTIVE_QUANTITIES)}, got {quantity!r}")
        if quantity == 'flux':
            normal_axis, line_cell, span_cells, along_widths = flux_line_cells(self.domain, axis, position, span)
            mirror_factor = line_mirror_factor(self.domain, self.boundary_config, normal_axis, span_cells)
            return self._add_monitor(FluxObjective(normal_axis, line_cell, span_cells, along_widths,
                                                   self.polarization, weight, mirror_factor))
        line_cell, span_cells, label = port_cells(self.domain, axis, position, span, 'Objective')
        return self._add_monitor(ModePowerObjective(self, axis, line_cell, span_cells, mode_number, label,
                                                    quantity.rsplit('_', 1)[1], weight))

    def solve_adjoint(self, flat_rhs):
        """x of operator^T x = flat_rhs (unconjugated) for the system of the last solve, on its factorisation (SuperLU
        trans='T', MUMPS / PARDISO transposed solves; GMRES-recycled or refactorised like a forward solve)."""
        if self.solution is None or self.system is None:
            raise ValueError("not solved yet: FDFDPhysics.solve_adjoint needs a solve (step or solve_harmonic) first")
        return self._solve_linear(self.xp.asarray(flat_rhs, dtype=self.xp.complex128), transpose=True)

    def eps_sensitivity(self):
        """adjoint.EpsSensitivity of the objectives at the last driven solve (one transposed solve, cached per solve and
        objectives; hz also dJ/d eps_xy); zeros without objectives, before a solve or while a resonance is shown."""
        objectives = tuple(monitor for monitor in self.monitors if is_objective(monitor))
        if self.selected_resonance is not None or not objectives:
            grid = self.domain.grid
            return EpsSensitivity(0.0, None, {}, self.xp.zeros((grid.nx, grid.ny), dtype=DTYPE))
        key = (self.n_solves, objectives)
        if self._sensitivity is None or self._sensitivity[0] != key:
            self._sensitivity = None   # free the previous gradient first
            self._sensitivity = (key, eps_sensitivity(self, self._wall_rows))
        return self._sensitivity[1]

    # --- design regions (design_density.py, optimization study) ---

    def add_design_region(self, mask, eps_min, eps_max, initial_density=0.5, filter_radius=0.0, projection_beta=0.0,
                          projection_eta=0.5):
        """Density design region on the window cells of a bool (nx, ny) mask of the run backend; writes eps_min + rho_bar
        (eps_max - eps_min) of initial_density into the permittivity arrays (ValueError when it overlaps an earlier
        region)."""
        self._require_cartesian('add_design_region')
        _check_cell_array(mask, self._field_re.shape, self.xp.bool_, 'Design region mask', xp=self.xp)
        region = DesignRegion(self.domain, mask, self.permittivity_arrays, eps_min, eps_max, initial_density,
                              filter_radius, projection_beta, projection_eta, objective_unit=self.objective_unit)
        if any(region.overlaps(other) for other in self.design_regions):
            raise ValueError('design regions of one physics must not share cells')
        self.design_regions.append(region)
        region.write()
        self.materials_changed(self.permittivity_arrays)
        return region

    def design_density(self):
        """float32 (nx, ny) projected density rho_bar of every design region, 0 elsewhere (fresh array)."""
        return paint_regions(self.design_regions, self._field_re.shape, self.xp)


def _host_sparse(operator):
    """Host scipy copy of a cupyx sparse matrix; a scipy matrix (cpu backend) itself."""
    return operator if isinstance(operator.data, np.ndarray) else operator.get()


def _is_cudss(factorization) -> bool:
    """Whether a driven factor is a cudss_factor.CudssFactorization (both precisions: back_end 'cudss')."""
    return getattr(factorization, 'back_end', None) == 'cudss'


def _outer_wall_rows(boundary_config, polarization):
    """(axis, outermost index, inner neighbour index, inner face index, _WALL_SIGNS sign) per wall / mirror face, x
    faces first (indices along the axis; hi side negative)."""
    rows = []
    for axis, axis_name in ((0, 'x'), (1, 'y')):
        for side, outer, inner, inner_face in (('lo', 0, 1, 0), ('hi', -1, -2, -2)):
            sign = _WALL_SIGNS[polarization].get(getattr(boundary_config, f'{axis_name}_{side}'))
            if sign is not None:
                rows.append((axis, outer, inner, inner_face, sign))
    return tuple(rows)


def _check_polar_boundaries(domain, bc):
    """ValueError for faces a polar (r, phi) FDFD cannot take (FDFDPhysics class comment): r is no periodic, mirror or
    x_lo cpml axis, phi no cpml axis, the origin needs 'pmc', periodic / bloch phi faces need joined domain faces."""
    coordinates = domain.coordinates
    if coordinates.line:
        raise ValueError('FDFDPhysics: a 1D line window (symmetry 2.20) has no EM form; use a 2D polar Domain')
    faces = {face: getattr(bc, face) for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    refused = [f'{face} {value.value}' for face, value in faces.items()
               if (face != 'x_hi' and value == BCType.CPML) or (face[0] == 'x' and (
                   value in MIRROR_TYPES or value in (BCType.PERIODIC, BCType.BLOCH)))]
    if refused:
        raise ValueError(f"FDFDPhysics on a polar grid: {', '.join(refused)} is not allowed (cpml only on x_hi = "
                         'the outer circle; an r = const circle is no mirror or period; phi faces take pec / pmc / '
                         'mirrors or periodic / bloch)')
    if coordinates.radius_min == 0.0 and faces['x_lo'] != BCType.PMC:
        raise ValueError(f"FDFDPhysics on a polar grid from the origin (radius_min 0): x_lo must be 'pmc', the "
                         f"placeholder of the origin (zero face length, no wall row), got {faces['x_lo'].value!r}")
    if bc.periodic_y != (coordinates.angle_wrap != 'none'):
        raise ValueError(f"FDFDPhysics on a polar grid: periodic / bloch phi faces (bc y_lo {faces['y_lo'].value}) "
                         f"need a domain whose angle_wrap joins them and the reverse (angle_wrap "
                         f"{coordinates.angle_wrap!r}); the EM phase is bloch_ky = azimuthal order m [1/rad]")

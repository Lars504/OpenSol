"""2D FDFD on a TriangleDomain (em_fdfd_ez / em_fdfd_hz on a 'triangles' mesh): FEMFDFDPhysics solves physics/em/fem_fdfd
on the host, driven (point sources, probes; stationary / frequency_domain) or eigen (eigenfrequency), and exposes the
solution per element (state) and as nodal fem_field coefficients (renderer, Session)."""
import math
import time

import numpy as np

from ....core.backend import to_host
from ....core.constants import C0, EPS0, MU0
from ....solvers.host_direct import HOST_DIRECT_BACK_ENDS, is_host_direct_factor
from ....solvers.sparse_direct import CUDSS_BACK_ENDS, SYMMETRIC_OPTION_BACK_ENDS, SparseDirectSolver
from ..eigenfrequency import element_region_rows
from ..fdfd_monitors import _ascending_wavelength
from ..fdfd_sources import phase_factor
from ..fem_fdfd import (FEMFDFDMaterials, assemble_fem_fdfd, dipole_load, element_state, fem_fdfd_error_indicator,
                        fem_fdfd_layout, fem_fdfd_size_issue, nodal_fields, point_fields, point_load,
                        solve_fem_fdfd_resonances)
from ..placement import PERMITTIVITY_ARRAYS, PROBE_COMPONENTS, require_polarization
from ..renderer import FDFDRendererEz, FDFDRendererHz
from .fem_em_base import TriangleEMPhysics
from .harmonic_shell import COMPONENT_UNITS, HarmonicEMShell, harmonic_state_names

# renderer fields served on triangles (the Yee renderer's names; adjoint / design / phase-change fields stay tensor)
_TRIANGLE_FIELDS = {
    'ez': ('ez_re', 'ez_im', 'ez_abs', 'ez_phase', 'e_abs', 'e_intensity', 'h_abs', 'eps_r', 'sigma_e',
           'absorbed_power'),
    'hz': ('hz_re', 'hz_im', 'hz_abs', 'hz_phase', 'ex_re', 'ex_im', 'ey_re', 'ey_im', 'e_abs', 'e_intensity', 'h_abs',
           'eps_xx', 'eps_yy', 'sigma_e', 'absorbed_power', 'eps_xy'),
}


class _PointExcitation:
    """One point source of the FEM physics: load(physics, system) -> complex (n_free,) (fem_fdfd point / dipole load);
    x, y [m] and the FDTD amplitude meaning of its feature."""

    def __init__(self, x, y, kind, amplitude, phase, orientation=0.0):
        self.x, self.y = float(x), float(y)
        self.kind = kind                 # 'current' (ez I [A]), 'magnetic_current' (hz K_m [V]), 'dipole' (hz M [A])
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        self.orientation = float(orientation)   # dipole direction [deg, ccw from +x]
        self._phasor = self.amplitude * phase_factor(phase, 'Point source')

    def load(self, physics, system):
        point = (self.x, self.y)
        if self.kind == 'current':     # (K - k0^2 M) E_z = j omega mu0 I phi(x0)
            return point_load(system.layout, point, 1j * system.omega * MU0 * self._phasor)
        if self.kind == 'magnetic_current':
            return point_load(system.layout, point, 1j * system.omega * EPS0 * self._phasor)
        angle = math.radians(self.orientation)
        return dipole_load(system.layout, system.materials, point,
                           (self._phasor * math.cos(angle), self._phasor * math.sin(angle)))


class FEMHarmonicProbe:
    """Complex phasor of one component at a point per solve (fem_fdfd.point_fields; the out-of-plane field exact, the
    in-plane one from the gradient in the containing element); fdfd_monitors.harmonic_probe_result reads it."""

    def __init__(self, x, y, field_component, unit):
        self.x, self.y = float(x), float(y)
        self.field_component = field_component
        self.unit = unit
        self._values = {}   # omega [rad/s] -> complex

    def record(self, physics, system):
        self._values[float(system.omega)] = point_fields(system, physics.solution, (self.x, self.y))[
            self.field_component]

    def recorded(self):
        """(omega float64 in ascending wavelength order, complex128 phasors); ValueError before the first solve."""
        omegas, values = _ascending_wavelength(
            self._values, f'harmonic probe {self.field_component} at ({self.x:.6g}, {self.y:.6g}) m')
        return omegas, np.array(values, dtype=np.complex128)


class FEMFDFDPhysics(HarmonicEMShell, TriangleEMPhysics):
    """Frequency-domain E_z ('ez') or H_z ('hz') on a fem.domain.TriangleDomain: Lagrange elements of degree 1 / 2 with a
    PML ring; driven solves of point sources factorised once per (omega, materials) by the 'linear' role on the host,
    resonances by the 'eigen' role; state fields and source slots per element (field_dofs: P0)."""
    # FDFDPhysics' attribute set for the studies, placement and couplings (polarization, sources / monitors, poles,
    # pec_cells, solution, solved_omega, eigen_solve / selected_resonance, the EM loss protocol), minus the Yee-only
    # pieces (sheets, TFSF, ports, flux / far-field monitors, adjoint, design regions: refused on triangles by validation).
    # solution = complex128 host coefficients of the layout's Lagrange map (fixed dofs 0); every assembly re-reads the
    # MaterialGrid. HarmonicEMShell gives the contract members it shares with FDFDPhysics, TriangleEMPhysics the
    # per-element ones.

    tensor_class = 'FDFDPhysics'

    def __init__(self, domain, polarization, boundary_config, wavelength=1.55e-6, linear_solver='auto',
                 eigen_solver=None, order=2):
        if polarization not in PERMITTIVITY_ARRAYS:
            raise ValueError(f"polarization must be 'ez' or 'hz', got {polarization!r}")
        self._init_triangle(domain, boundary_config, polarization, polarization, order, eigen_solver)
        self._init_harmonic_shell(wavelength, fields=self.element_fields)
        if isinstance(linear_solver, str):
            linear_solver = SparseDirectSolver(back_end=linear_solver, backend=domain.backend.name)
            self._private_solvers.append(linear_solver)
        elif not isinstance(linear_solver, SparseDirectSolver):
            raise TypeError(f'em_fdfd_{polarization} on triangles factorises with a sparse_direct linear solver, got '
                            f'{type(linear_solver).__name__} (line_block runs on tensor meshes only)')
        self.linear_solver = linear_solver

        self.solution = None          # host complex128 coefficients of the last solve / shown resonance
        self.system = None            # fem_fdfd.FEMFDFDSystem of the last assembly
        self.eigen_solve = None
        self.resonance_region = None  # eigenfrequency.ResonanceRegion (bool per element) of a resonance_table, or None
        self.selected_resonance = None
        self.last_factor_seconds = None
        self._factorization = None
        # cuDSS / MUMPS / PARDISO factor of an older system, refactorised in place by the next one (plan / analysis)
        self._cudss_previous = None
        self._system_dirty = True
        self._solution_dirty = True

    @property
    def type_label(self) -> str:
        return f'em_fdfd_{self.polarization}'

    @property
    def state_fields(self) -> tuple:
        return harmonic_state_names(self.polarization)

    @property
    def material_names(self) -> tuple:
        return (*self.permittivity_arrays, 'sigma_e')

    @property
    def renderer_field_infos(self) -> dict:
        """The Yee renderer's FieldInfos of the fields that exist on triangles."""
        renderer_class = FDFDRendererEz if self.polarization == 'ez' else FDFDRendererHz
        return {name: renderer_class.field_infos[name] for name in _TRIANGLE_FIELDS[self.polarization]
                if name in renderer_class.field_infos}

    # --- PhysicsSolver contract (+ HarmonicEMShell, TriangleEMPhysics) ---

    def _check_back_end(self):
        _ = self.linear_solver.back_end   # resolves the choice (ValueError without scipy)

    def _size_issue(self, layout, factorization=None):
        """(level, text) of the host SuperLU size rule (the eigen factorization unless given; MUMPS / PARDISO keep it),
        None for cuDSS."""
        if factorization is None:
            factorization = getattr(self.eigen_solver, 'factorization', 'superlu')
        return fem_fdfd_size_issue(layout.n_unknowns, self.order, factorization)

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the shown solution for mesh refinement (fem_fdfd tangential
        in-plane jumps); ValueError before a solve."""
        if self.solution is None or self.system is None:
            raise ValueError(f'em_fdfd_{self.polarization} on triangles: no solution to estimate the mesh error of')
        return fem_fdfd_error_indicator(self.system, self.solution)

    # --- study hooks ---

    def set_resonance_region(self, region):
        """Keep the ResonanceRegion (host bool per mesh element) whose energy fraction the eigen solve reports /
        filters; ValueError for another shape."""
        expected = (int(self.mesh.n_elements),)
        if region.cells.shape != expected:
            raise ValueError(f'resonance region cells {region.cells.shape} do not match the {expected[0]} elements')
        self.resonance_region = region

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances nearest the wavelength setting (fem_fdfd on the
        current materials, one assembly), kept as eigen_solve."""
        self.eigen_solve = None
        system = self._refresh_system()
        if self._cudss_previous is not None:   # no driven factorisation follows here: free the factor kept for a plan
            self._cudss_previous.release()
            self._cudss_previous = None
        issue = self._size_issue(system.layout)
        if issue is not None and issue[0] == 'error':
            raise ValueError(f'eigenfrequency solve: {issue[1]}')
        region = self.resonance_region
        in_region = (None if region is None
                     else element_region_rows(system.layout.dofs, system.layout.free, region.cells))
        self.eigen_solve = solve_fem_fdfd_resonances(system, int(n_resonances), eigen_solver=self.eigen_solver,
                                                     warnings=() if issue is None else (issue[1],),
                                                     in_region=in_region,
                                                     min_region_fraction=0.0 if region is None else region.min_fraction)
        self.eigen_solve.info.update(n_elements=int(self.mesh.n_elements), order=self.order,
                                     assembly_seconds=system.assembly_seconds)
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last eigenfrequency solve in solution and the per-element state
        (in-plane fields at the target omega); the next step() solves the driven problem again."""
        resonance = self.eigen_solve.resonances[number - 1]
        self._show(self.system.layout.coefficients(resonance.field), self.system)
        self.selected_resonance = number
        self._solution_dirty = True

    def selection_label(self):
        """'resonance k: lambda = ..., Q = ...' while a resonance is shown, else 'lambda = <nm> nm' of the last solve."""
        if self.selected_resonance is not None:
            return self._resonance_label()
        if self.solved_omega is None:
            return None
        return f'lambda = {2.0 * math.pi * C0 / self.solved_omega * 1e9:.6g} nm'

    def solve_summary(self):
        """One line on the last driven solve for the study log (unknowns, elements, times, factor size); None before."""
        if self.last_solve_seconds is None or self.system is None:
            return None
        linear = self.linear_solver
        cudss = linear.back_end in CUDSS_BACK_ENDS
        host_direct = linear.back_end in HOST_DIRECT_BACK_ENDS
        factor_text = ((getattr(self._factorization, 'description', None) or linear.back_end)
                       if cudss or host_direct else 'host SuperLU')
        text = (f'{self.system.layout.n_unknowns:,} FEM unknowns ({self.mesh.n_elements:,} triangles, degree '
                f'{self.order}), sparse direct back end {linear.back_end} ({factor_text} on triangles), '
                f'solve {self.last_solve_seconds:.3g} s')
        if self.last_factor_seconds is not None:
            host_bytes = f'{(self.host_factor_bytes or 0) / 2 ** 20:.4g} MiB of host memory'
            factor_size = (f'{linear.n_factorizations} factorisations on {linear.n_cudss_plans} plan(s)' if cudss
                           else f'factor {host_bytes}' if host_direct else f'L + U factors {host_bytes}')
            text += (f' (assembly {self.system.assembly_seconds:.3g} s, {factor_text} factorisation '
                     f'{self.last_factor_seconds:.3g} s, {factor_size})')
        return text

    # --- solve ---

    def _mark_system_dirty(self, ports=True, omega_only=False):
        """Reassemble and re-solve at the next solve (ports / omega_only: FDFDPhysics' keywords, nothing to keep here)."""
        self._system_dirty = True
        self._solution_dirty = True

    def _system_changed(self):
        self._mark_system_dirty()

    def _solve(self):
        """Driven solve at omega: reassemble / refactorise when the system changed, sum the point loads, refresh the
        solution and state, record every probe."""
        started = time.perf_counter()
        system = self._refresh_system()
        if self._factorization is None:
            back_end = self.linear_solver.back_end
            keeps_previous = back_end in SYMMETRIC_OPTION_BACK_ENDS
            # cuDSS: no host size rule; MUMPS / PARDISO: SuperLU's rule, texts naming the library
            issue = self._size_issue(system.layout, back_end if keeps_previous else 'superlu')
            if issue is not None and issue[0] == 'error':
                raise ValueError(issue[1])
            # host SuperLU (cusolver_qr too), or the cuDSS / MUMPS / PARDISO LDL^T of the complex symmetric pencil with
            # its equilibration, refactorising the previous system's factor in place (plan / analysis reused; a factor
            # of another library is released by host_direct_factorization); SuperLU ignores these options
            previous, self._cudss_previous = self._cudss_previous, None
            if not keeps_previous and previous is not None:   # an 'auto' runtime fell back to SuperLU (B1b): free it
                previous.release()
                previous = None
            self._factorization = self.linear_solver.factorize_host(system.operator, symmetric=True,
                                                                    row_scale=system.equilibration, previous=previous)
            self.last_factor_seconds = self.linear_solver.last_factor_seconds
            self.host_factor_bytes = self._factorization.host_bytes
        rhs = np.zeros(system.layout.n_unknowns, dtype=np.complex128)
        for source in self.sources:
            rhs += source.load(self, system)
        free_values = np.asarray(self._factorization.solve_host(rhs), dtype=np.complex128)
        self._show(system.layout.coefficients(free_values), system)
        self.selected_resonance = None
        self._solution_dirty = False
        self.n_solves += 1
        for monitor in self.monitors:
            monitor.record(self, system)
        self.last_solve_seconds = time.perf_counter() - started

    def _refresh_system(self):
        """fem_fdfd.FEMFDFDSystem at omega, reassembled (and the driven factor dropped) when dirty."""
        if self._system_dirty or self.system is None:
            self.system = None
            if (getattr(self._factorization, 'back_end', None) == 'cudss'
                    or is_host_direct_factor(self._factorization)):   # kept for its plan / analysis (_solve)
                self._cudss_previous = self._factorization
            self._factorization = None
            self.system = assemble_fem_fdfd(self._current_layout(), self._materials(self._omega), self._omega)
            self._system_dirty = False
        return self.system

    def _show(self, coefficients, system):
        """solution = coefficients; per-element state (centroid values, q_abs) refreshed, nodal cache reset."""
        self.solution = coefficients
        self.solved_omega = system.omega
        centroid_values, absorbed = element_state(system, coefficients)
        re_name, im_name, _ = harmonic_state_names(self.polarization)
        self._write_element_fields({re_name: centroid_values.real, im_name: centroid_values.imag, 'q_abs': absorbed})

    def _build_layout(self, pec):
        return fem_fdfd_layout(self.mesh, self.order, self.boundary_config, self.polarization, pec)

    def _materials(self, omega):
        """fem_fdfd.FEMFDFDMaterials at omega: host complex eps (ez eps_r; hz the in-plane tensor with eps_xy) and mu
        (sigma_m), sigma_e and poles included; host bool PEC elements."""
        eps, mu, pec = self._element_materials(omega)
        if self.polarization == 'ez':
            return FEMFDFDMaterials(eps=eps['eps_r'], mu=mu, pec=pec)
        tensor = np.zeros((self.mesh.n_elements, 2, 2), dtype=np.complex128)
        tensor[:, 0, 0], tensor[:, 1, 1] = eps['eps_xx'], eps['eps_yy']
        off_diagonal = np.asarray(to_host(self.eps_xy()), dtype=np.float64)
        off_diagonal[pec] = 0.0
        tensor[:, 0, 1] = tensor[:, 1, 0] = off_diagonal
        return FEMFDFDMaterials(eps=tensor, mu=mu, pec=pec)

    # --- display ---

    def _has_shown_field(self) -> bool:
        return self.solution is not None

    def _last_unknowns(self):
        return None if self.system is None else self.system.layout.n_unknowns

    def _shown_nodal_fields(self, nodal_dofs) -> dict:
        return _shown_parts(nodal_fields(self.system, self.solution, nodal_dofs), self.polarization)

    def fem_field(self, component) -> tuple:
        """(Lagrange DofMap, host float64 coefficients) of a renderer field (fem.renderer contract): materials, eps_xy
        and absorbed_power per element (P0); phasor fields at the nodes of the element degree (the out-of-plane unknown
        exact, in-plane fields averaged over the non-PEC elements of a node); zeros before a solve."""
        if component == 'eps_xy':
            return self.field_dofs('sigma_e'), np.asarray(to_host(self.eps_xy()), dtype=np.float64)
        return super().fem_field(component)

    # --- sources, probes ---

    def _add_source(self, source):
        self.sources.append(source)
        self._solution_dirty = True
        return source

    def add_point_source(self, x, y, current=1e-3, phase=0.0):
        """z-directed filament current `current` [A] at (x, y) (ez only): load j omega mu0 I e^{j phase} phi_i(x, y)."""
        require_polarization(self.polarization, 'ez', 'add_point_source')
        return self._add_source(_PointExcitation(x, y, 'current', current, phase))

    def add_magnetic_line_current(self, x, y, magnetic_current=1.0, phase=0.0):
        """z-directed magnetic line current K_m [V] at (x, y) (hz only): load j omega eps0 K_m e^{j phase} phi_i."""
        require_polarization(self.polarization, 'hz', 'add_magnetic_line_current')
        return self._add_source(_PointExcitation(x, y, 'magnetic_current', magnetic_current, phase))

    def add_electric_dipole(self, x, y, dipole_moment=1e-3, orientation=0.0, phase=0.0):
        """In-plane electric current moment M [A] along orientation [deg, ccw from +x] at (x, y) (hz only): load (R eta
        p) . grad phi_i (fem_fdfd.dipole_load)."""
        require_polarization(self.polarization, 'hz', 'add_electric_dipole')
        return self._add_source(_PointExcitation(x, y, 'dipole', dipole_moment, phase, orientation))

    def add_field_probe(self, x, y, field_component=None):
        """Complex phasor of one component at (x, y) per solve; None = the out-of-plane field."""
        field_component = self.polarization if field_component is None else field_component
        if field_component not in PROBE_COMPONENTS[self.polarization]:
            raise ValueError(f"em_fdfd_{self.polarization} probe component must be one of "
                             f"{PROBE_COMPONENTS[self.polarization]}, got {field_component!r}")
        probe = FEMHarmonicProbe(x, y, field_component, COMPONENT_UNITS[field_component])
        self.monitors.append(probe)
        if self.solution is not None and self.system is not None and self.selected_resonance is None:
            probe.record(self, self.system)
        return probe


def _shown_parts(nodal, polarization) -> dict:
    """Host float64 nodal values of every renderer phasor field from fem_fdfd.nodal_fields ('u', 'fx', 'fy')."""
    u, fx, fy = nodal['u'], nodal['fx'], nodal['fy']
    shown = {f'{polarization}_re': u.real, f'{polarization}_im': u.imag, f'{polarization}_abs': np.abs(u),
             f'{polarization}_phase': np.angle(u)}
    if polarization == 'hz':
        shown.update(ex_re=fx.real, ex_im=fx.imag, ey_re=fy.real, ey_im=fy.imag)
    in_plane = np.abs(fx) ** 2 + np.abs(fy) ** 2
    out_of_plane = np.abs(u) ** 2
    shown['e_intensity'] = out_of_plane if polarization == 'ez' else in_plane
    shown['e_abs'] = np.sqrt(shown['e_intensity'])
    shown['h_abs'] = np.sqrt(in_plane if polarization == 'ez' else out_of_plane)
    return shown

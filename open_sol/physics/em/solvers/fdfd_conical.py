"""Conical FDFD on a Domain (symmetry 2.9: a z-invariant structure at a fixed out-of-plane wave number k_z, all six
components): ConicalPhysics assembles and solves on the host (physics/em/conical) and keeps float32 centre fields (run
backend) for the renderer."""
import cmath
import math
import time

import numpy as np
from ....core.backend import to_host

from ....core.constants import C0, EPS0
from ....core.solver_base import PhysicsSolver
from ..boundaries import BCType, solver_boundary_config
from ..conical import (COMPONENTS, FIELD_COMPONENTS, assemble_conical, centre_phasors, conical_fields,
                       conical_row_scale, driven_rhs, line_flux, point_current_terms, size_issue_for,
                       solve_conical_resonances, source_power)
from ..cross_section_modes import CrossSectionMaterials
from ..fdfd_monitors import _ascending_wavelength, probe_phasor_result
from ..harmonic import cell_permeability, cell_permittivity, shown_phasor_fields
from ..materials import register_em_arrays
from ..placement import flux_line_cells, line_mirror_factor
from ..renderer_conical import ConicalRenderer
from ..subpixel_smoothing import SubpixelSmoothing
from ..yee_grid import cpml_grading_cell_size, grid_cell_widths
from ....solvers.host_direct import is_host_direct_factor
from .host_fdfd import HostFDFDBase, direct_text

CONICAL_POLARIZATION = 'conical'        # no in-plane polarisation (sheet placement helpers refuse it)
CONICAL_MATERIAL_SCHEMA_KEY = 'modes'   # EM_MATERIAL_SCHEMAS key: eps_r (E_z), eps_xx (E_x), eps_yy (E_y), mu, sigmas
CONICAL_PERMITTIVITY_ARRAYS = ('eps_r', 'eps_xx', 'eps_yy')   # eps_r holds eps_zz (as em_modes)
CONICAL_PERMITTIVITY_SLOTS = {f'{array}_update': array for array in CONICAL_PERMITTIVITY_ARRAYS}
# float32 full-grid fields ConicalRenderer reads: Re and modulus of the six centre phasors, |E|^2, |H|, S_z, q_abs
CONICAL_STATE_NAMES = (FIELD_COMPONENTS + tuple(f'{name}_abs' for name in FIELD_COMPONENTS)
                       + ('e_intensity', 'h_abs', 'sz', 'q_abs'))
COMPONENT_UNITS = {'ex': 'V/m', 'ey': 'V/m', 'ez': 'V/m', 'hx': 'A/m', 'hy': 'A/m', 'hz': 'A/m'}
_CELL_EPS = {'x': 'eps_xx', 'y': 'eps_yy', 'z': 'eps_zz'}   # CrossSectionMaterials cell eps per E component


class ConicalPhysics(HostFDFDBase, PhysicsSolver):
    """Frequency-domain EM of a z-invariant structure at fixed k_z (fields ~ exp(j (omega t - k_z z))): host driven
    solves and shift-invert resonances; the shown solution feeds the float32 centre fields."""
    # FDFDPhysics attribute set (domain, polarization, boundary_config, sources, monitors, dispersive_regions,
    # pec_cells, finalize_warnings, permittivity_arrays, enable_q_accum, solved_omega, n_solves, eigen_solve,
    # selected_resonance, linear_solver, eigen_solver). A material / omega / k_z change re-reads the MaterialGrid (five
    # host transfers) into a new ConicalSystem; the driven SuperLU factor is kept while the system stays, and after a
    # material change at the same omega, k_z and active set (thermo-optic Picard) GMRES on the old factor runs first
    # (sparse_direct RECYCLE_*), refactorising only when it misses. 'linear' back_end cudss / cudss_mixed: an LDL^T of D A
    # (system.symmetrizer; a general LU with a Bloch phase) with the eigen-solve's row scale instead of SuperLU; not
    # recyclable, so a replaced factor is kept as _cudss_previous and the next factorisation (new omega, k_z, material)
    # refactorises it in place: one cuDSS plan while the pattern holds. mumps / pardiso (+ _mixed): the same LDL^T
    # options on the host, recyclable like SuperLU and under SuperLU's size rule.

    name = 'em'
    type_label = 'em_fdfd_conical'
    permittivity_arrays = CONICAL_PERMITTIVITY_ARRAYS   # eps_r (= eps_zz, E_z), eps_xx (E_x), eps_yy (E_y)
    permittivity_slots = CONICAL_PERMITTIVITY_SLOTS
    state_names = CONICAL_STATE_NAMES
    field_components = FIELD_COMPONENTS
    component_units = COMPONENT_UNITS

    def __init__(self, domain, boundary_config=None, wavelength=1.55e-6, k_z=0.0, linear_solver=None,
                 eigen_solver=None):
        if getattr(domain, 'dimensions', 2) != 2:
            raise ValueError('ConicalPhysics needs a 2D domain: the drawing is the cross-section of a z-invariant '
                             'structure')
        self.domain = domain
        self.xp = domain.xp   # run backend of the fields / masks (cupy 'gpu', numpy 'cpu'); the solves are host
        self.polarization = CONICAL_POLARIZATION
        self.boundary_config = solver_boundary_config(domain, boundary_config, 'ConicalPhysics')
        self._k_z = _real_wavenumber(k_z)
        # omega, the linear / eigen solvers, the source / monitor / solve record and the float32 display fields
        # (CONICAL_STATE_NAMES) of HostFDFDBase; sources: ConicalCurrent (point and line sources); monitors:
        # CentrePhasorProbe, ConicalFluxMonitor
        self._init_host_fdfd(domain, wavelength, linear_solver, eigen_solver)
        self._cell_widths = grid_cell_widths(domain.grid)   # the cross_section_modes layout widths (as em_modes)
        self.pml_dx = cpml_grading_cell_size(domain)
        register_em_arrays(domain.materials, CONICAL_MATERIAL_SCHEMA_KEY)
        self.subpixel_smoothing = None   # SubpixelSmoothing ('modes') of the model-wide switch

        self.system = None            # conical.ConicalSystem of the current materials, omega and k_z
        self.last_materials = None    # CrossSectionMaterials (host) of the system
        self.assembly_count = 0
        self.yee_fields = None        # host conical_fields of the driven solve (monitors)
        self.last_source_power = None   # W/m delivered by the sources in the last driven solve
        self._eigen_materials = None

    @property
    def k_z(self) -> float:
        """Out-of-plane wave number k_z [rad/m] of every solve (fields ~ exp(-j k_z z))."""
        return self._k_z

    @k_z.setter
    def k_z(self, k_z):
        k_z = _real_wavenumber(k_z)
        if k_z != self._k_z:
            self._k_z = k_z
            self._mark_assembly_dirty()

    # --- PhysicsSolver contract (HostFDFDBase) and study hooks ---

    def _grid_size_issue(self):
        return size_issue_for(self.domain.grid.nx, self.domain.grid.ny)

    def renderer(self):
        return ConicalRenderer(self)

    def set_subpixel_smoothing(self, geometry):
        """Average the current eps on the interface geometry (igor 1.5) in every solve as em_modes does (E_x / E_y faces
        anisotropic, E_z cells arithmetic); None switches it off."""
        self.subpixel_smoothing = (None if geometry is None
                                   else SubpixelSmoothing(geometry, CONICAL_MATERIAL_SCHEMA_KEY))
        self._mark_assembly_dirty()

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances at k_z nearest the wavelength setting
        (conical.solve_conical_resonances on the current materials), kept as eigen_solve."""
        self.eigen_solve = None
        self._release_kept_cudss()   # the eigen-solve's own factor needs the device memory
        self._eigen_materials = self._conical_materials(self.omega)
        self.eigen_solve = solve_conical_resonances(self._cell_widths, self.boundary_config, self.pml_dx,
                                                    self._eigen_materials, self.k_z, int(n_resonances), self.omega,
                                                    eigen_solver=self.eigen_solver)
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last eigenfrequency solve in the fields and state; the next step()
        solves the driven problem again."""
        resonance = self.eigen_solve.resonances[number - 1]
        self.selected_resonance = number
        self._show(self.eigen_solve.system, resonance.fields, self._eigen_materials)
        self._solution_dirty = True

    def solve_summary(self):
        """One line on the last driven solve for the study log (unknowns, k_z, wall times, factor size); None before."""
        if self.last_solve_seconds is None:
            return None
        text = f'{self.system.n_unknowns:,} unknowns (E_x, E_y, E_z) at k_z = {self.k_z:.6g} rad/m'
        if self._linear_on_cudss:
            return self._cudss_summary(f'{text}, {getattr(self._factorization, "description", "cuDSS")}, solve '
                                       f'{self.last_solve_seconds:.3g} s')
        return (f'{text}, {direct_text(self._factorization)}, solve {self.last_solve_seconds:.3g} s'
                + self._host_factor_summary(is_host_direct_factor(self._factorization)))

    def scalars(self) -> dict:
        """{'source_power': W/m delivered by the sources in the last driven solve} (E5 scalar outputs; {} before)."""
        return {} if self.last_source_power is None else {'source_power': self.last_source_power}

    # --- solve ---

    def _mark_assembly_dirty(self, omega_only=False):
        """Materials, omega or k_z changed: re-read the materials into a new system at the next solve."""
        self._system_dirty = True
        self._solution_dirty = True

    def _refresh_system(self):
        """ConicalSystem at self.omega and k_z on the current materials when dirty; the driven factor stays as a
        recycling preconditioner only for a material change at the same omega, k_z and active set."""
        if self._system_dirty or self.system is None:
            previous = self.system
            self.system = None
            self.last_materials = self._conical_materials(self.omega)
            self.system = assemble_conical(self._cell_widths, self.boundary_config, self.pml_dx, self.last_materials,
                                           self.omega, self.k_z, check_size=not self._linear_on_cudss)
            self.assembly_count += 1
            self._system_dirty = False
            self._operator_replaced(previous is not None and previous.omega == self.system.omega
                                    and previous.k_z == self.system.k_z
                                    and np.array_equal(previous.active_index, self.system.active_index))
        return self.system

    def _solve_linear(self, system, rhs):
        """x of system.matrix x = rhs (HostFDFDBase._solve_kept_factor; a cuDSS factor is the LDL^T of D A with
        conical_row_scale)."""
        initial_guess = (self.solution if self.solution is not None and self.solution.size == rhs.size
                         and self.solved_omega == system.omega else None)
        return self._solve_kept_factor(system.matrix, rhs, initial_guess, lambda: self._factorize_operator(
            system.matrix, lambda: {'symmetrizer': system.symmetrizer, 'row_scale': conical_row_scale(system)}))

    def _solve(self):
        """Driven solve at self.omega and k_z: (re)assemble when dirty, solve (recycled or factorised), show the solution,
        record the source power and the monitors."""
        start = time.perf_counter()
        issue = None if self._linear_on_cudss else size_issue_for(self.domain.grid.nx, self.domain.grid.ny)
        if issue is not None and issue[0] == 'error':
            raise ValueError(issue[1])
        system = self._refresh_system()
        terms = [term for source in self.sources for term in source.terms]
        self.solution = self._solve_linear(system, driven_rhs(system, terms=terms))
        self.selected_resonance = None
        self.yee_fields = conical_fields(system, self.solution)
        self.last_source_power = source_power(system, self.yee_fields, terms)
        self._show(system, self.yee_fields, self.last_materials)
        self._solution_dirty = False
        self.n_solves += 1
        for monitor in self.monitors:
            monitor.record(self)
        self.last_solve_seconds = time.perf_counter() - start

    def _show(self, system, yee_fields, materials):
        """Centre phasors of the Yee fields into centre_phasors and the float32 fields (16 host-to-device copies on the
        gpu); q_abs = 1/2 omega eps0 sum_c (-Im eps_c of the cell) <|E_c|^2> of the cell."""
        phasors, e_squared = centre_phasors(system, yee_fields)
        host_fields = shown_phasor_fields(phasors, FIELD_COMPONENTS[:3], FIELD_COMPONENTS[3:], sz=True)
        host_fields['q_abs'] = 0.5 * system.omega * EPS0 * sum(
            -np.asarray(getattr(materials, _CELL_EPS[component])).imag * e_squared[component]
            for component in COMPONENTS)
        self._show_fields(host_fields, phasors, system.omega)

    def _conical_materials(self, omega):
        """CrossSectionMaterials at omega: host complex128 eps_xx / eps_yy / eps_zz (sigma_e, poles) and mu (sigma_m) of
        every cell, host bool PEC cells, the subpixel-smoothing deltas of the background eps when enabled."""
        materials = self.domain.materials
        xp = self.xp
        eps = {name: to_host(cell_permittivity(materials, name, self.dispersive_regions, omega, xp=xp))
               for name in CONICAL_PERMITTIVITY_ARRAYS}
        pec = (np.zeros((self.domain.grid.nx, self.domain.grid.ny), dtype=bool) if self.pec_cells is None
               else np.array(to_host(self.pec_cells), dtype=bool))
        deltas = None if self.subpixel_smoothing is None else self.subpixel_smoothing.point_deltas(materials)
        return CrossSectionMaterials(eps_xx=eps['eps_xx'], eps_yy=eps['eps_yy'], eps_zz=eps['eps_r'],
                                     mu=to_host(cell_permeability(materials, omega, xp=xp)), pec=pec,
                                     eps_point_deltas=deltas)

    # --- sources and monitors (poles / PEC masks, field probes: HostFDFDBase) ---

    def add_point_source(self, x, y, component='z', current=1e-3, phase=0.0):
        """Current element (conical.point_current_terms) along `component` 'x' | 'y' | 'z' in the cell containing (x, y)
        carrying `current` [A] (integral of J over the cell), phasor e^{j phase} (phase in degrees, FDTD sign)."""
        self.domain.validate_physical_pos(x, y, 'Point source')
        ix, iy = self.domain.pos_to_cell(x, y)
        phasor = float(current) * cmath.exp(1j * math.radians(float(phase)))
        source = ConicalCurrent(point_current_terms(self._cell_widths, self.boundary_config, ix, iy,
                                                    _component(component), phasor))
        return self._add_source(source)

    def add_line_source(self, axis, position, span, component='z', amplitude=1.0, phase=0.0):
        """Sheet of surface current K = amplitude [A/m] along `component` on an axis-aligned line (axis 'x': along x at
        y = position) over span [m]: per cell K x its length, times exp(-j k s) along a Bloch axis (_line_cells)."""
        component = _component(component)
        if axis not in ('x', 'y'):
            raise ValueError(f"Line source axis (along the line) must be 'x' or 'y', got {axis!r}")
        line_cell, along_cells = self._line_cells(axis, position, span, 'Line source')
        along = 0 if axis == 'x' else 1
        widths = self._cell_widths[along]
        centres = np.cumsum(widths) - 0.5 * widths
        bloch = self.boundary_config.bloch_x if axis == 'x' else self.boundary_config.bloch_y
        wavenumber = (self.boundary_config.bloch_kx if axis == 'x' else self.boundary_config.bloch_ky) if bloch else 0.0
        phasor = float(amplitude) * cmath.exp(1j * math.radians(float(phase)))
        edges = np.concatenate(([0.0], np.cumsum(widths)))
        terms = []
        for cell in range(*along_cells):
            ix, iy = (cell, line_cell) if axis == 'x' else (line_cell, cell)
            if component != axis or not wavenumber:
                current = phasor * widths[cell] * cmath.exp(-1j * float(wavenumber) * centres[cell])
                terms.extend(point_current_terms(self._cell_widths, self.boundary_config, ix, iy, component, current))
                continue
            # an along-line current sits on the cell's two faces: each takes the Bloch phase of its own position (the
            # centre phase would leave the sheet low by cos(k dx / 2)); the wrapped lo face of cell 0 is at x = 0
            for name, i, j, density in point_current_terms(self._cell_widths, self.boundary_config, ix, iy, component,
                                                           phasor * widths[cell]):
                face = i if axis == 'x' else j
                position = edges[face + 1] - (edges[-1] if face > cell else 0.0)
                terms.append((name, i, j, density * cmath.exp(-1j * float(wavenumber) * position)))
        return self._add_source(ConicalCurrent(terms))

    def _line_cells(self, axis, position, span, label):
        """(line cell across the line, [start, stop) cells along it) of a line source; a span end on a cpml window edge
        extends to the grid end (a uniform sheet then does not diffract like a slit)."""
        if span is None or len(span) != 2:
            raise ValueError(f'{label} needs span = (start, end) along the line [m], got {span!r}')
        start, end = sorted((float(span[0]), float(span[1])))
        domain = self.domain
        if axis == 'x':
            domain.validate_physical_pos(start, position, f'{label} start')
            domain.validate_physical_pos(end, position, f'{label} end')
            line_cell = domain.pos_to_cell_y(position)
            lo, hi = domain.pos_to_cell_x(start), domain.pos_to_cell_x(end) + 1
        else:
            domain.validate_physical_pos(position, start, f'{label} start')
            domain.validate_physical_pos(position, end, f'{label} end')
            line_cell = domain.pos_to_cell_x(position)
            lo, hi = domain.pos_to_cell_y(start), domain.pos_to_cell_y(end) + 1
        window = domain.physical_window()[0 if axis == 'x' else 1]
        n_along = domain.grid.nx if axis == 'x' else domain.grid.ny
        if lo == window.start and getattr(self.boundary_config, f'{axis}_lo') == BCType.CPML:
            lo = 0
        if hi == window.stop and getattr(self.boundary_config, f'{axis}_hi') == BCType.CPML:
            hi = n_along
        return line_cell, (lo, hi)

    def add_flux_monitor(self, axis, position, span):
        """Power per unit z length [W/m] through a line per driven solve (axis 'x' = along x at y = position: flux along
        +y; 'y': flux along +x), span = (start, end) along it [m]; conical.line_flux on the hi face of the line cells."""
        normal_axis, line_cell, span_cells, _ = flux_line_cells(self.domain, axis, position, span)
        n_normal = self.domain.grid.ny if normal_axis == 'y' else self.domain.grid.nx
        periodic = self.boundary_config.periodic_y if normal_axis == 'y' else self.boundary_config.periodic_x
        if line_cell >= n_normal - 1 and not periodic:
            raise ValueError(f'Flux monitor: the line cell {line_cell} is the last grid cell along {normal_axis}, so its '
                             'hi face is the outer grid edge; move the line inside')
        mirror_factor = line_mirror_factor(self.domain, self.boundary_config, normal_axis, span_cells)
        return self._add_monitor(ConicalFluxMonitor(normal_axis, line_cell, span_cells, mirror_factor))

    def _add_monitor(self, monitor):
        """Append a monitor; it records the shown driven solve at once when there is one."""
        self.monitors.append(monitor)
        if self.yee_fields is not None and self.selected_resonance is None:
            monitor.record(self)
        return monitor


def _real_wavenumber(k_z):
    """float(k_z) [rad/m], ValueError unless finite."""
    k_z = float(k_z)
    if not math.isfinite(k_z):
        raise ValueError(f'ConicalPhysics k_z must be a finite wave number [rad/m], got {k_z!r}')
    return k_z


def _component(component):
    """The current direction 'x' | 'y' | 'z', ValueError otherwise."""
    if component not in COMPONENTS:
        raise ValueError(f'source component must be one of {COMPONENTS}, got {component!r}')
    return component


class ConicalCurrent:
    """Point terms [(component, i, j, J_F [A/m^2])] of a point or line source (conical.point_current_terms)."""

    def __init__(self, terms):
        self.terms = list(terms)


class ConicalFluxMonitor:
    """Power per unit z length [W/m] through the hi face of a line of cells per driven solve, > 0 toward +normal
    (conical.line_flux); mirror_factor 2 per mirror face the line reaches (unfolded total)."""

    def __init__(self, normal, line_cell, span_cells, mirror_factor):
        self.normal = normal
        self.line_cell = int(line_cell)
        self.span = (int(span_cells[0]), int(span_cells[1]))
        self.mirror_factor = float(mirror_factor)
        self._power = {}   # omega -> W/m (drawn part)

    def record(self, physics):
        self._power[float(physics.solved_omega)] = line_flux(physics.system, physics.yee_fields, self.normal,
                                                             self.line_cell, self.span)

    def recorded(self):
        """(omega float64 in ascending wavelength order, power float64 [W/m] of the drawn part); ValueError before."""
        omegas, powers = _ascending_wavelength(self._power,
                                               f'flux monitor ({self.normal} flux, line cell {self.line_cell})')
        return omegas, np.array(powers, dtype=np.float64)


def conical_probe_result(probe) -> dict:
    """fdfd_monitors.probe_phasor_result of a conical field probe (y label: the component)."""
    return probe_phasor_result(probe, ConicalRenderer.field_infos[probe.field_component].label.replace('Re ', ''))


def conical_flux_result(monitor) -> dict:
    """'flux_spectrum' of a ConicalFluxMonitor: time-averaged power per unit z length [W/m] vs ascending wavelength
    (x mirror_factor: the unfolded total), extra mirror_factor."""
    omegas, powers = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    return {'kind': 'flux_spectrum', 'frequency': frequency, 'wavelength': C0 / frequency,
            'power': monitor.mirror_factor * powers, 'unit': 'W/m', 'mirror_factor': monitor.mirror_factor}

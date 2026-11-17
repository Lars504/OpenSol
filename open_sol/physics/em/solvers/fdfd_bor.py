"""Body-of-revolution FDFD on a Domain (x = r from the x_lo window edge = the axis, y = z; fields ~ exp(-j m phi)):
BORPhysics assembles and solves on the host (body_of_revolution) and keeps float32 centre fields (run backend) for the
renderer."""
import cmath
import math
import time

import numpy as np
from ....core.backend import to_host

from ....core.constants import C0, EPS0, FACES
from ....core.solver_base import PhysicsSolver
from ..body_of_revolution import (BOR_FIELD_COMPONENTS, BOR_SOURCE_COMPONENTS, BORAssembly, BORMaterials, axis_class,
                                  bor_row_scale, centre_fields, distinct_orders, far_field_power, far_field_samples,
                                  line_flux, merge_order_solves, order_pair_fields, order_sum_far_field,
                                  point_current_rhs, size_issue_for, solve_bor_resonances, yee_centres, yee_fields)
from ..bor_line_block import bor_layout, driven_factorization, order_factorizations, wavelength_factorizations
from ..boundaries import BCType, solver_boundary_config
from ..fdfd_line_block import MAX_BATCH_WAVELENGTHS
from ....solvers.line_block import LineBlockFactorization, LineBlockSolver, RejectedLineBlockSolve
from ....solvers.host_direct import HOST_DIRECT_BACK_ENDS
from ....solvers.sparse_direct import CUDSS_BACK_ENDS
from ..bor_ports import BORModeMonitor, BORModeSource
from ..eigenfrequency import resonance_table_result
from ..fdfd_monitors import _ascending_wavelength, probe_phasor_result
from ..harmonic import cell_permeability, cell_permittivity, shown_phasor_fields
from ..materials import register_em_arrays
from ..placement import box_cells, flux_line_cells, port_cells
from ..renderer_bor import BORRenderer
from ..yee_grid import cpml_grading_cell_size
from .host_fdfd import HostFDFDBase, direct_text, is_cudss

BOR_POLARIZATION = 'bor'          # no in-plane polarisation (sheet placement helpers refuse it)
BOR_MATERIAL_SCHEMA_KEY = 'modes'   # EM_MATERIAL_SCHEMAS key: eps_r (E_phi), eps_xx (E_r), eps_yy (E_z), mu_r, sigmas
BOR_PERMITTIVITY_ARRAYS = ('eps_r', 'eps_xx', 'eps_yy')
BOR_PERMITTIVITY_SLOTS = {'eps_r_update': 'eps_r', 'eps_xx_update': 'eps_xx', 'eps_yy_update': 'eps_yy'}
# float32 full-grid fields BORRenderer reads: Re and modulus of the six centre phasors, |E|^2, |H|, q_abs
BOR_STATE_NAMES = (BOR_FIELD_COMPONENTS + tuple(f'{name}_abs' for name in BOR_FIELD_COMPONENTS)
                   + ('e_intensity', 'h_abs', 'q_abs'))
COMPONENT_UNITS = {'er': 'V/m', 'ephi': 'V/m', 'ez': 'V/m', 'hr': 'A/m', 'hphi': 'A/m', 'hz': 'A/m'}
FAR_FIELD_QUANTITY = 'abs(E)^2'
LINE_BLOCK_TEXT = 'line-block factor (refined against the complex128 operator)'   # solve_summary solver texts
_REJECTED = 'rejected'   # _line_factor of a system the line-block solver rejected: SuperLU until the system changes
_MIRROR_SIGNS = {BCType.PMC: 1.0, BCType.PEC: -1.0}   # z_lo walls that mirror the far-field box (E parity)


def bor_boundary_config(domain, boundary_config, label):
    """solver_boundary_config with the body-of-revolution rules (BORPhysics, BORFDTDPhysics): x_lo the axis (no cpml /
    periodic), no mirror faces (pec / pmc walls on the grid edge are the mirrors)."""
    config = solver_boundary_config(domain, boundary_config, label)
    if config.cpml_sides['x_lo'] or config.periodic_x:
        raise ValueError(f'{label}: x_lo is the axis r = 0, so it must be no cpml or periodic face')
    mirrors = [face for face in FACES if getattr(config, face) in (BCType.SYMMETRIC, BCType.ANTISYMMETRIC)]
    if mirrors:
        raise ValueError(f"{label}: mirror faces ({', '.join(mirrors)}) are not supported: its walls sit on the grid "
                         "edge, so a mirror's padded ghost cell would move the plane one cell out of the window; use "
                         'pec / pmc walls (a pec / pmc z_lo wall is the mirror plane)')
    return config


def bor_flux_line(domain, axis, position, span):
    """(normal 'z' | 'r', line cell, span cells) of a BOR flux line (axis 'x' = along r at z = position: flux along
    +z; 'y' = along z at r = position: flux along +r); ValueError when its hi face is the outer grid edge."""
    normal_axis, line_cell, span_cells, _ = flux_line_cells(domain, axis, position, span)
    normal = 'z' if normal_axis == 'y' else 'r'
    n_normal = domain.grid.ny if normal == 'z' else domain.grid.nx
    if line_cell >= n_normal - 1:
        raise ValueError(f'Flux monitor: the line cell {line_cell} is the last grid cell along {normal}, so its hi '
                         'face is the outer grid edge; move the line inside')
    return normal, line_cell, span_cells


def bor_far_field_box(domain, boundary_config, x0, y0, x1, y1):
    """((ix0, ix1), (iy0, iy1), mirror_sign) of a BOR far-field box covering [x0, x1] x [y0, y1] (r, z): a bottom on
    a pec / pmc z_lo wall is mirrored (sign -1 / +1), else None; ValueError without a window cell beyond r_hi, z_hi
    and (unmirrored) z_lo."""
    ix0, ix1, iy0, iy1 = box_cells(domain, x0, y0, x1, y1, 'Far-field box')
    window_r, window_z = domain.physical_window()
    mirror_sign = _MIRROR_SIGNS.get(boundary_config.y_lo) if iy0 == 0 else None
    if ix1 >= window_r.stop or iy1 >= window_z.stop or (mirror_sign is None and iy0 <= window_z.start):
        raise ValueError(f'Far-field box cells [{ix0}, {ix1}) x [{iy0}, {iy1}) need one window cell beyond r_hi, '
                         f'z_hi and z_lo (window [{window_r.start}, {window_r.stop}) x [{window_z.start}, '
                         f'{window_z.stop})); a box bottom on a pec / pmc z_lo wall needs none')
    return (ix0, ix1), (iy0, iy1), mirror_sign


class BORPhysics(HostFDFDBase, PhysicsSolver):
    """Frequency-domain EM of a body of revolution at azimuthal order m: driven solves (point currents, mode ports) at
    omega and shift-invert resonances, on the host; the shown solution feeds the float32 centre fields."""
    # FDFDPhysics attribute set (domain, polarization, boundary_config, sources, monitors, dispersive_regions,
    # pec_cells, finalize_warnings, permittivity_arrays, enable_q_accum, solved_omega, n_solves, eigen_solve,
    # selected_resonance, linear_solver, eigen_solver) plus axisymmetric = True (Session ring-weights field derived
    # values). Levels: a material / omega change re-reads the MaterialGrid (five host transfers, kept as last_materials
    # for the ports) into a new BORAssembly; an azimuthal_order change only takes assembly.system(m) (an m sweep
    # assembles once). The driven factor is kept while omega, m and materials stay; after a material change at the same
    # omega, m and active set (thermo-optic Picard) GMRES on the old factor runs first (sparse_direct RECYCLE_*), a
    # refactorisation only when it misses. Permittivity: eps_xx = eps_rr (E_r), eps_r = eps_phiphi (E_phi), eps_yy =
    # eps_zz (E_z, along the axis). A LineBlockSolver as shifted_solver factors every driven system and the
    # shift-invert operators of the eigen-solve (all azimuthal_orders of an axis class in one batch) on its backend; a
    # solve it rejects goes to host SuperLU (run-log warning). Anything else as shifted_solver: host SuperLU only.
    # Wavelength batches (plan_harmonic_frequencies, frequency_domain): a driven single-order solve at a planned omega
    # not in the kept batch factorises the next up to MAX_BATCH_WAVELENGTHS planned omegas in one factorize_operators
    # batch; their host materials and assemblies stay with the batch (_batch_assemblies) and serve their own solve while
    # (_material_serial, order) is unchanged.
    # Driven order sum (driven_orders non-empty): every point source is a 3D element at phi = 0 (BORPointCurrent.
    # term_rhs); per listed m one factor of A(m) solves +m and -m (body_of_revolution.order_pair_fields), monitors
    # accumulate per order (record_order) and total (record); shown fields = the phi = 0 sum, q_abs = its phi mean; no
    # factor is kept (a re-solve factorises every order again), mode ports are refused.
    # 'linear' back_end cudss / cudss_mixed: an LDL^T of the complex symmetric operator (row scale (|diag K| + k0^2
    # |M|)^-1/2, as the eigen-solve) instead of SuperLU; not recyclable, so a replaced factor is kept as _cudss_previous
    # and the next factorisation (new omega, material, order, order-sum term) refactorises it in place: one cuDSS plan
    # while the pattern holds. mumps / pardiso (+ _mixed): the same LDL^T options on the host, recyclable like SuperLU
    # (GMRES after a material change) and under SuperLU's size rule.

    name = 'em'
    type_label = 'em_fdfd_bor'
    permittivity_arrays = BOR_PERMITTIVITY_ARRAYS   # eps_r (E_phi), eps_xx (E_r), eps_yy (E_z)
    permittivity_slots = BOR_PERMITTIVITY_SLOTS
    state_names = BOR_STATE_NAMES
    field_components = BOR_FIELD_COMPONENTS
    component_units = COMPONENT_UNITS

    def __init__(self, domain, boundary_config=None, wavelength=1.55e-6, azimuthal_order=0, linear_solver=None,
                 eigen_solver=None, shifted_solver=None, azimuthal_orders=None, driven_orders=None):
        if getattr(domain, 'dimensions', 2) != 2:
            raise ValueError('BORPhysics needs a 2D domain: the drawing is the (r, z) half plane')
        self.domain = domain
        self.xp = domain.xp   # run backend of the fields / masks (cupy 'gpu', numpy 'cpu'); the solves are host
        self.polarization = BOR_POLARIZATION
        self.boundary_config = bor_boundary_config(domain, boundary_config, 'BORPhysics')
        axis_class(azimuthal_order)   # integer check
        self._azimuthal_order = int(azimuthal_order)
        self.azimuthal_orders = azimuthal_orders
        # omega, the linear / eigen solvers, the source / monitor / solve record and the float32 display fields
        # (BOR_STATE_NAMES) of HostFDFDBase; sources: BORPointCurrent, bor_ports.BORModeSource; monitors:
        # CentrePhasorProbe, BORFluxMonitor, BORNearToFar, bor_ports.BORModeMonitor
        self._init_host_fdfd(domain, wavelength, linear_solver, eigen_solver)
        self.shifted_solver = shifted_solver   # the build's 'shifted' role (line_block: device factors), may be None
        self._line_block = shifted_solver if isinstance(shifted_solver, LineBlockSolver) else None
        self._cell_widths = domain.cell_widths()   # float64 (dx, dy): r from the x_lo grid edge, as Session rings
        self.pml_dx = cpml_grading_cell_size(domain)
        self._window = domain.physical_window()
        register_em_arrays(domain.materials, BOR_MATERIAL_SCHEMA_KEY)
        self.axisymmetric = True

        self.assembly = None          # body_of_revolution.BORAssembly of the current materials and omega
        self.system = None            # body_of_revolution.BORSystem of the assembly at azimuthal_order
        self.last_materials = None    # BORMaterials (host) of the last assembly
        self.assembly_count = 0       # systems built (assembly or order changes): ports re-solve their modes
        # bor_ports: ((omega, order, assembly_count), {line key: PortModes}): ports on identical lines share one solve
        self.port_mode_solves = (None, {})
        self.last_linear_text = None   # 'host SuperLU' / 'line-block' of the last driven solve
        self._line_factor = None      # line_block.LineBlockFactorization of system, 'rejected', or None
        # Wavelength batches: the planned omegas in solve order, the _material_serial they were planned at, and the
        # kept batch {omega: LineBlockFactorization} with its (material serial, azimuthal order)
        self._planned_omegas = ()
        self._plan_serial = None
        self._material_serial = 0     # bumped by every assembly change except a new omega
        self._batch = {}
        self._batch_key = None
        self._batch_assemblies = {}   # omega -> (BORMaterials, BORAssembly) of the later batch members
        self.last_batch_size = None   # wavelengths of the last batched line-block factorisation
        self._eigen_assembly = None   # BORAssembly of the last eigen-solve (select_resonance rebuilds its orders)
        self._eigen_system = None
        self._yee = None             # body_of_revolution.yee_fields of the last driven solve (monitors)
        self._assembly_dirty = True
        self.order_sum_info = None    # dict (orders, unknowns, linear texts) of the last driven order sum
        self.driven_orders = driven_orders

    @property
    def azimuthal_order(self) -> int:
        """Azimuthal order m of the next solve (fields ~ exp(-j m phi))."""
        return self._azimuthal_order

    @azimuthal_order.setter
    def azimuthal_order(self, order):
        """Switch m: the next solve takes assembly.system(m) (no material re-read) and factorises it anew."""
        axis_class(order)
        if int(order) != self._azimuthal_order:
            self._azimuthal_order = int(order)
            self._system_dirty = True
            self._solution_dirty = True
            self._retire_factorization()   # another operator: nothing to recycle (a cuDSS plan stays)
            self._drop_line_factor()

    @property
    def azimuthal_orders(self) -> tuple:
        """Orders the eigen-solve covers (fields ~ exp(-j m phi) each): the order list, else (azimuthal_order,)."""
        return self._azimuthal_orders or (self._azimuthal_order,)

    @azimuthal_orders.setter
    def azimuthal_orders(self, orders):
        """Distinct integer orders of solve_eigenfrequencies (None / empty: azimuthal_order alone)."""
        self._azimuthal_orders = distinct_orders(orders, 'azimuthal_orders')

    @property
    def driven_orders(self) -> tuple:
        """Orders m >= 0 of the driven order sum (each with its -m partner), () = one driven solve at azimuthal_order."""
        return self._driven_orders

    @driven_orders.setter
    def driven_orders(self, orders):
        """Distinct integer orders >= 0 of driven solves summed over m (None / empty: azimuthal_order alone)."""
        self._driven_orders = distinct_orders(orders, 'driven_orders', non_negative=True)
        self._solution_dirty = True

    # --- PhysicsSolver contract (HostFDFDBase) and study hooks ---

    def _grid_size_issue(self):
        return size_issue_for(self.domain.grid.nx, self.domain.grid.ny)

    def renderer(self):
        return BORRenderer(self)

    def plan_harmonic_frequencies(self, omegas):
        """frequency_domain study hook: the omegas [rad/s] solve_harmonic follows, in order; a line_block shifted solver
        then factorises up to MAX_BATCH_WAVELENGTHS of them per batch (single-order driven solves); () ends the plan."""
        self._planned_omegas = tuple(float(omega) for omega in omegas)
        self._plan_serial = self._material_serial
        if not self._planned_omegas:
            self._release_batch()

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances nearest the wavelength setting per order of
        azimuthal_orders, all from one assembly (body_of_revolution.solve_bor_resonances; line-block factors batched per
        axis class when the shifted solver is one) -> one EigenSolve (order by order), kept as eigen_solve."""
        self.eigen_solve = None
        system = self._refresh_system()
        self._release_kept_cudss()   # the eigen-solve's own factor needs the device memory
        assembly = self.assembly
        orders = self.azimuthal_orders
        factorizations = self._order_factorizations(assembly, orders)
        solves, warnings = {}, []
        for order, factorization in factorizations:
            order_system = system if order == system.azimuthal_order else assembly.system(order)
            try:
                solves[order] = solve_bor_resonances(order_system, n_resonances, eigen_solver=self.eigen_solver,
                                                     factorization=factorization)
            except RejectedLineBlockSolve as exc:
                warnings.append(f'm = {order}: {exc}; shift-invert by host SuperLU instead')
                solves[order] = solve_bor_resonances(order_system, n_resonances, eigen_solver=self.eigen_solver)
        self._eigen_assembly, self._eigen_system = assembly, system
        self.eigen_solve = merge_order_solves([solves[order] for order in orders], warnings)
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last eigenfrequency solve in the fields and state (in-plane H
        from the curl at the target omega, its own order's system); the next step() solves the driven problem again."""
        resonance = self.eigen_solve.resonances[number - 1]
        if self._eigen_system.azimuthal_order != resonance.azimuthal_order:
            self._eigen_system = self._eigen_assembly.system(resonance.azimuthal_order)
        self.solution = resonance.field
        self.selected_resonance = number
        self._yee = None
        self._show(self._eigen_system, resonance.field)
        self._solution_dirty = True

    def solve_summary(self):
        """One line on the last driven solve for the study log (unknowns, wall times, factor size); None before."""
        if self.last_solve_seconds is None:
            return None
        if self._driven_orders and self.order_sum_info is not None:
            info = self.order_sum_info
            return (f"driven order sum m = {', '.join(map(str, info['orders']))} (each m > 0 with -m), up to "
                    f"{info['unknowns']:,} unknowns (E_r, E_phi, E_z) per order, {' / '.join(info['texts'])} (one "
                    f'factor per order), solve {self.last_solve_seconds:.3g} s')
        linear = self.linear_solver
        text = (f'{self.system.n_unknowns:,} unknowns (E_r, E_phi, E_z), azimuthal order {self.azimuthal_order}, '
                f'{self.last_linear_text}, solve {self.last_solve_seconds:.3g} s')
        if self.last_linear_text == LINE_BLOCK_TEXT:
            if self.last_batch_size is not None:
                text += f' (one batch of {self.last_batch_size} wavelengths: later ones in it are solve-only)'
            return text
        if linear.back_end in CUDSS_BACK_ENDS:
            return self._cudss_summary(text)
        return text + self._host_factor_summary(linear.back_end in HOST_DIRECT_BACK_ENDS)

    def yee(self):
        """body_of_revolution.yee_fields of the last driven solve (host, cached per solve; an order sum: the phi = 0
        sum of its terms); ValueError before it."""
        if self.selected_resonance is not None or (self.solution is None and self._yee is None):
            raise ValueError('BORPhysics.yee: no driven solve is shown')
        if self._yee is None:
            self._yee = yee_fields(self.system, self.solution)
        return self._yee

    # --- solve ---

    def _mark_assembly_dirty(self, omega_only=False):
        """Materials or omega changed: re-read the materials into a new assembly at the next solve; omega_only: a new
        omega alone (planned wavelength batches stay valid)."""
        self._assembly_dirty = True
        self._system_dirty = True
        self._solution_dirty = True
        if not omega_only:
            self._material_serial += 1

    def _refresh_system(self):
        """BORSystem at self.omega and azimuthal_order: a new assembly when materials / omega changed, a new system
        when the order changed. The driven factor stays as a recycling preconditioner only for a material change at
        the same omega, order and active set."""
        previous = None   # (omega, order, active set) of the system the kept factor belongs to
        if self._assembly_dirty or self.assembly is None:
            if self.system is not None:
                previous = (self.system.omega, self.system.azimuthal_order, self.system.unknown_index)
            self.assembly = self.system = None
            kept = (self._batch_assemblies.pop(self.omega, None)
                    if self._batch_key == (self._material_serial, self.azimuthal_order) else None)
            if kept is not None:   # assembled with its wavelength batch at the same materials
                self.last_materials, self.assembly = kept
            else:
                self.last_materials = self._bor_materials(self.omega)
                self.assembly = BORAssembly(self._cell_widths, self.boundary_config, self.pml_dx, self.last_materials,
                                            self.omega, window=self._window)
            self._assembly_dirty = False
            self._system_dirty = True
        if self._system_dirty or self.system is None:
            self.system = self.assembly.system(self.azimuthal_order)
            self.assembly_count += 1
            self._system_dirty = False
            self._drop_line_factor()
            self._operator_replaced(previous is not None and previous[0] == self.system.omega
                                    and previous[1] == self.system.azimuthal_order
                                    and np.array_equal(previous[2], self.system.unknown_index))
        return self.system

    def _factorize_host(self, system):
        """Factorization of system.operator by the 'linear' solver (HostFDFDBase._factorize_operator; cuDSS row scale
        bor_row_scale); a line-block fallback onto host SuperLU first checks the size rule _solve skipped."""
        if self._line_block is not None and not self._linear_on_cudss:
            issue = size_issue_for(self.domain.grid.nx, self.domain.grid.ny)
            if issue is not None and issue[0] == 'error':
                raise ValueError(f'the line-block solve was rejected and its host SuperLU fallback is too '
                                 f'large: {issue[1]}')
        # C^T W C - k0^2 M: complex symmetric by construction (unconjugated transpose, diagonal W and M)
        return self._factorize_operator(system.operator, lambda: {'row_scale': bor_row_scale(system)},
                                        symmetric_by_construction=True)

    def _order_factorizations(self, assembly, orders):
        """(order, line-block factorization) per order of `orders`, batched per axis class when the shifted solver is a
        line_block (re-bound to the order families), else (order, None): host factors."""
        if self._line_block is None:
            return ((order, None) for order in orders)
        self._drop_line_factor()   # the shared solver is re-bound to the order families
        self._release_batch()
        return order_factorizations(self._line_block, assembly, orders, owner=self)

    def _drop_line_factor(self):
        """Release the driven line-block factor (another system, or the solver re-bound elsewhere); a wavelength batch
        stays for its other omegas."""
        if (self._line_factor is not None and self._line_factor != _REJECTED
                and not any(entry is self._line_factor for entry in self._batch.values())):
            self._line_factor.release()
        self._line_factor = None

    def _release_batch(self):
        """Drop the kept wavelength batch (its device inverses and operator caches)."""
        for entry in self._batch.values():
            entry.release()   # one shared LineBlockFactor: the first call frees it
            break
        self._batch = {}
        self._batch_key = None
        self._batch_assemblies = {}

    def _batch_factorization(self, system):
        """LineBlockFactorization of system in a wavelength batch (a new batch of the next planned omegas from
        system.omega when it is planned but not in the kept batch), None: not planned, materials changed since the plan,
        or fewer than two omegas fit (the single-system factor then)."""
        omega = system.omega
        if self._batch:
            entry = next(iter(self._batch.values()))
            if not entry.valid or self._batch_key != (self._material_serial, system.azimuthal_order):
                if not entry.valid and self._line_block.owner is not self:
                    self._planned_omegas = ()   # another user re-bound the shared solver: batches would refactor
                self._release_batch()
        if omega in self._batch:
            return self._batch[omega]
        if self._plan_serial != self._material_serial or omega not in self._planned_omegas:
            return None
        self._release_batch()   # before sizing the next batch: its inverses count as used memory
        upcoming = list(dict.fromkeys(self._planned_omegas[self._planned_omegas.index(omega):]))
        layout = bor_layout(system.shape, system.unknown_index)
        n_batch = min(len(upcoming), MAX_BATCH_WAVELENGTHS,
                      self._line_block.batch_size(len(upcoming), layout, system.operator.nnz))
        if n_batch < 2:
            return None
        systems, assemblies = [system], {}
        for other in upcoming[1:n_batch]:   # kept for their own solve (_refresh_system)
            materials = self._bor_materials(other)
            assembly = BORAssembly(self._cell_widths, self.boundary_config, self.pml_dx, materials, other,
                                   window=self._window)
            other_system = assembly.system(system.azimuthal_order)
            if not np.array_equal(other_system.unknown_index, system.unknown_index):
                break
            systems.append(other_system)
            assemblies[other] = (materials, assembly)
        if len(systems) < 2:
            return None
        entries = wavelength_factorizations(self._line_block, systems, owner=self)
        self._batch = {member.omega: entry for member, entry in zip(systems, entries)}
        self._batch_key = (self._material_serial, system.azimuthal_order)
        self._batch_assemblies = assemblies
        self.last_batch_size = len(systems)
        return entries[0]

    def _solve_line_block(self, system, rhs):
        """x by the line-block factor of system (a planned wavelength batch, else factorised on first use), None when it
        rejects the solve (then this system goes to host SuperLU until it changes; run-log warning)."""
        if self._line_factor == _REJECTED:
            return None
        if self._line_factor is not None and not self._line_factor.valid:
            self._line_factor = None   # the shared solver holds another operator now
        try:
            if self._line_factor is None:
                start = time.perf_counter()
                kept = system.omega in self._batch   # a batch entry factorised earlier (unless the batch is stale)
                batched = self._batch_factorization(system)
                if batched is not None:
                    self._line_factor = batched
                else:
                    self._line_factor = driven_factorization(self._line_block, system, owner=self)
                    self.last_batch_size = None
                if batched is None or not kept:
                    self.last_factor_seconds = time.perf_counter() - start
            return self._line_factor.solve_host(rhs)
        except RejectedLineBlockSolve as exc:
            self.runtime_warnings.append(f'{exc}; solved by host SuperLU instead')
            self._drop_line_factor()
            self._line_factor = _REJECTED
            return None

    def _solve_linear(self, system, rhs):
        """x of system.operator x = rhs: the line-block factor when the shifted solver is one (rejected: SuperLU), else
        GMRES on the factor of an earlier operator (same omega, order, active set) while it reaches
        sparse_direct.RECYCLE_RTOL, else (re)factorise and solve directly."""
        self.last_recycled_iterations = None
        if self._line_block is not None:
            solution = self._solve_line_block(system, rhs)
            if solution is not None:
                self.last_linear_text = LINE_BLOCK_TEXT
                return solution
        initial_guess = (self.solution if self.solution is not None and self.solution.size == rhs.size
                         and self.selected_resonance is None and self.solved_omega == system.omega else None)
        solution = self._solve_kept_factor(system.operator, rhs, initial_guess, lambda: self._factorize_host(system))
        # recycled solves too: the kept factor's kind (SuperLU, MUMPS / PARDISO; cuDSS factors are never recycled)
        self.last_linear_text = direct_text(self._factorization)
        return solution

    def _solve(self):
        """Driven solve at self.omega: (re)assemble when dirty, solve (recycled or factorised), show the solution,
        record the monitors; the order sum when driven_orders is set."""
        start = time.perf_counter()
        issue = (None if self.linear_solver.back_end in CUDSS_BACK_ENDS or self._line_block is not None
                 else size_issue_for(self.domain.grid.nx, self.domain.grid.ny))   # line_block: _factorize_host checks
        if issue is not None and issue[0] == 'error':
            raise ValueError(issue[1])
        if self._driven_orders:
            self._solve_order_sum(start)
            return
        system = self._refresh_system()
        rhs = np.zeros(system.n_unknowns, dtype=np.complex128)
        for source in self.sources:
            rhs += source.rhs(self, system)
        self.solution = self._solve_linear(system, rhs)
        self.selected_resonance = None
        self._yee = None
        self._show(system, self.solution)
        self._solution_dirty = False
        self.n_solves += 1
        for monitor in self.monitors:
            monitor.record(self)
        self.last_solve_seconds = time.perf_counter() - start

    def _solve_order_sum(self, start):
        """Driven order sum at self.omega over driven_orders (class comment): one factor per order (line-block batches
        per axis class when the shifted solver is one), the +m / -m terms of every point source, monitors fed per order,
        the phi = 0 sum shown with the phi-mean q_abs."""
        ports = sorted({type(item).__name__ for item in self.sources + self.monitors
                        if isinstance(item, (BORModeSource, BORModeMonitor))})
        if ports:
            raise ValueError(f"BORPhysics driven order sum: mode ports ({', '.join(ports)}) belong to one azimuthal "
                             'order; clear driven_orders or remove them')
        system = self._refresh_system()   # the assembly at omega (its azimuthal_order system is reused when listed)
        if is_cudss(self._factorization):
            self._retire_factorization()   # the order terms refactorise it (one device factor at a time)
        assembly = self.assembly
        orders = self._driven_orders
        factorizations = self._order_factorizations(assembly, orders)
        total, mean_e_squared, texts, unknowns = None, None, set(), 0
        for order, factorization in factorizations:
            order_system = system if order == system.azimuthal_order else assembly.system(order)
            rhs = np.zeros(order_system.n_unknowns, dtype=np.complex128)
            minus_rhs = np.zeros_like(rhs)
            for source in self.sources:
                rhs += source.term_rhs(order_system)
                if order > 0:
                    minus_rhs += source.term_rhs(order_system, order=-order)
            factor = _OrderFactor(self, order_system, factorization)
            terms = order_pair_fields(order_system, rhs, factor.solve_host, minus_rhs=minus_rhs)
            self._keep_cudss_plan(factor.factorization)   # the next order refactorises it
            texts |= factor.texts
            unknowns = max(unknowns, order_system.n_unknowns)
            for _, yee in terms:
                _, e_squared = yee_centres(yee)
                if total is None:
                    total, mean_e_squared = dict(yee), dict(e_squared)
                else:
                    total = {name: total[name] + yee[name] for name in total}
                    mean_e_squared = {name: mean_e_squared[name] + e_squared[name] for name in mean_e_squared}
            for monitor in self.monitors:
                monitor.record_order(self, order_system, terms)
        self.solution = None
        self.selected_resonance = None
        self._yee = total
        self._show_phasors(system, yee_centres(total)[0], mean_e_squared)
        self._solution_dirty = False
        self.n_solves += 1
        self.order_sum_info = {'orders': orders, 'unknowns': unknowns, 'texts': sorted(texts)}
        self.last_linear_text = ' / '.join(sorted(texts))
        for monitor in self.monitors:
            monitor.record(self)
        self.last_solve_seconds = time.perf_counter() - start

    def _show(self, system, solution):
        """Centre phasors of `solution` into centre_phasors and the float32 fields (15 host-to-device copies on gpu)."""
        phasors, e_squared = centre_fields(system, solution)
        self._show_phasors(system, phasors, e_squared)

    def _show_phasors(self, system, phasors, e_squared):
        """Centre phasors and per-component cell mean |E_c|^2 (q_abs) into centre_phasors and the float32 fields."""
        host_fields = shown_phasor_fields(phasors, ('er', 'ephi', 'ez'), ('hr', 'hphi', 'hz'))
        host_fields['q_abs'] = 0.5 * system.omega * EPS0 * sum(-system.cell_eps[name].imag * e_squared[name]
                                                               for name in ('r', 'phi', 'z'))
        self._show_fields(host_fields, phasors, system.omega)

    def _bor_materials(self, omega):
        """BORMaterials at omega: host complex128 eps of the three components (permittivity array, sigma_e, poles)
        and mu (sigma_m), host bool PEC cells."""
        materials = self.domain.materials
        eps = {name: to_host(cell_permittivity(materials, name, self.dispersive_regions, omega, xp=self.xp))
               for name in BOR_PERMITTIVITY_ARRAYS}
        pec = (np.zeros((self.domain.grid.nx, self.domain.grid.ny), dtype=bool) if self.pec_cells is None
               else to_host(self.pec_cells))
        return BORMaterials(eps=eps['eps_r'], mu=to_host(cell_permeability(materials, omega, xp=self.xp)), pec=pec,
                            eps_rr=eps['eps_xx'], eps_zz=eps['eps_yy'])

    # --- sources and monitors (poles / PEC masks, field probes: HostFDFDBase) ---

    def add_point_source(self, x, y, component='z', current=1e-3, phase=0.0):
        """Current element (body_of_revolution.point_current_rhs) of `component` 'r' | 'phi' | 'z' and current [A] in
        the cell containing (x = r, y = z), phasor current e^{j phase} (phase in degrees, FDTD sign)."""
        if component not in BOR_SOURCE_COMPONENTS:
            raise ValueError(f'point source component must be one of {BOR_SOURCE_COMPONENTS}, got {component!r}')
        self.domain.validate_physical_pos(x, y, 'Point source')
        ix, iy = self.domain.pos_to_cell(x, y)
        return self._add_source(BORPointCurrent(ix, iy, component,
                                                float(current) * cmath.exp(1j * math.radians(float(phase)))))

    def add_mode_source(self, position, span, power=1.0, mode_number=1, direction='positive', phase=0.0):
        """Port source (bor_ports.BORModeSource) on the z = position line over r = span (from the axis, or annular):
        `power` [W] of guided mode `mode_number` toward +z ('positive') or -z, phase [deg]."""
        row, span_cells, label = port_cells(self.domain, 'x', position, span, 'Mode source')
        return self._add_source(BORModeSource(self, row, span_cells, power, mode_number, direction, phase, label))

    def add_flux_monitor(self, axis, position, span):
        """Power [W] through a line per driven solve (axis 'x' = along r at z = position: flux along +z; 'y' = along z
        at r = position: flux along +r), span = (start, end) along it [m]; body_of_revolution.line_flux."""
        return self._add_monitor(BORFluxMonitor(*bor_flux_line(self.domain, axis, position, span)))

    def add_near_to_far(self, x0, y0, x1, y1):
        """Far-field monitor (body_of_revolution.far_field_samples) on the box of cells covering [x0, x1] x [y0, y1]
        (r, z); a box bottom on a pec / pmc z_lo wall is mirrored."""
        return self._add_monitor(BORNearToFar(*bor_far_field_box(self.domain, self.boundary_config, x0, y0, x1, y1)))

    def add_mode_monitor(self, position, span, mode_number=1):
        """Port monitor (bor_ports.BORModeMonitor) on the z = position line over r = span (from the axis, or annular)."""
        row, span_cells, label = port_cells(self.domain, 'x', position, span, 'Mode monitor')
        return self._add_monitor(BORModeMonitor(self, row, span_cells, mode_number, label))

    def _add_monitor(self, monitor):
        """Append a monitor; record the shown driven solve at once when there is one (an order sum: at the next solve,
        which needs its per-order terms)."""
        self.monitors.append(monitor)
        if self._driven_orders:
            self._solution_dirty = True
        elif (self.centre_phasors is not None and self.selected_resonance is None and self.system is not None
              and self.solution is not None):   # solution None: an order sum shown (driven_orders cleared since)
            monitor.record(self)
        return monitor


class _OrderFactor:
    """solve_host of one order's operator in a driven order sum: its line-block factorization until that rejects a
    solve (then the 'linear' solver, run-log warning), else a 'linear' solver factor (host SuperLU or cuDSS) made on
    first use; texts = kinds used."""

    def __init__(self, physics, system, factorization):
        self.physics, self.system, self.factorization = physics, system, factorization
        self.texts = set()

    def solve_host(self, rhs):
        if self.factorization is None:
            self.factorization = self.physics._factorize_host(self.system)
        try:   # a deferred cuDSS check missed: SuperLU on an 'auto' runtime (HostFDFDBase._checked_solve_host)
            solution, self.factorization = self.physics._checked_solve_host(
                self.factorization, rhs, lambda: self.physics._factorize_host(self.system))
        except RejectedLineBlockSolve as exc:
            self.physics.runtime_warnings.append(f'm = {self.system.azimuthal_order}: {exc}; solved by the linear '
                                                 'solver instead')
            self.factorization = None
            return self.solve_host(rhs)
        self.texts.add(LINE_BLOCK_TEXT if isinstance(self.factorization, LineBlockFactorization)
                       else direct_text(self.factorization))
        return solution


class BORPointCurrent:
    """Current element in cell (ix, iy): component 'r' | 'phi' | 'z', complex phasor current [A]."""

    def __init__(self, ix, iy, component, current):
        self.ix, self.iy = int(ix), int(iy)
        self.component = component
        self.current = complex(current)

    def rhs(self, physics, system):
        return point_current_rhs(system, self.ix, self.iy, self.component, self.current)

    def term_rhs(self, system, order=None):
        """rhs of the exp(-j m phi) Fourier term at m = order (default the system's) of this element as a 3D point
        current at phi = 0 of moment I d (d = cell height for 'z', width for 'r' / 'phi'): the ring current I (r, z),
        I d / (2 pi r) (phi); the same for every m except an axis-cell 'r' at |m| = 1 (point_current_rhs)."""
        scale = 1.0
        if self.component == 'phi':
            scale = system.r_widths[self.ix] / (2.0 * math.pi * system.r_centres[self.ix])
        return point_current_rhs(system, self.ix, self.iy, self.component, self.current * scale, order=order)


class BORFluxMonitor:
    """Power [W] through the hi face of a line of cells per driven solve: normal 'z' (annulus over r cells) or 'r'
    (cylinder over z cells), > 0 toward +normal (body_of_revolution.line_flux)."""

    def __init__(self, normal, line_cell, span_cells):
        self.normal = normal
        self.line_cell = int(line_cell)
        self.span = (int(span_cells[0]), int(span_cells[1]))
        self._power = {}         # omega -> W
        self._order_power = {}   # omega -> (orders, float64 (n_orders,) W per listed m with its -m) of an order sum
        self._pending = {}       # |m| -> W of the order sum being solved

    def record_order(self, physics, system, terms):
        """Flux [W] of one order's terms (+m, -m) of a driven order sum (orthogonal in phi: they add)."""
        self._pending[abs(terms[0][0])] = sum(line_flux(system, fields, self.normal, self.line_cell, self.span)
                                             for _, fields in terms)

    def record(self, physics):
        omega = float(physics.solved_omega)
        if physics.driven_orders:
            per_order = np.array([self._pending[order] for order in physics.driven_orders], dtype=np.float64)
            self._pending = {}
            self._order_power[omega] = (tuple(physics.driven_orders), per_order)
            self._power[omega] = float(np.sum(per_order))
            return
        self._order_power.pop(omega, None)
        self._power[omega] = line_flux(physics.system, physics.yee(), self.normal, self.line_cell, self.span)

    def recorded(self):
        """(omega float64 in ascending wavelength order, power float64 [W]); ValueError before the first solve."""
        omegas, powers = _ascending_wavelength(self._power,
                                               f'flux monitor ({self.normal} flux, line cell {self.line_cell})')
        return omegas, np.array(powers, dtype=np.float64)

    def recorded_orders(self):
        """(orders, float64 (n_wavelengths, n_orders) W per listed m with its -m, ascending wavelength) when every
        recorded solve is an order sum over the same orders, else None."""
        omegas, _ = self.recorded()
        records = [self._order_power.get(float(omega)) for omega in omegas]
        if any(record is None for record in records) or len({record[0] for record in records}) != 1:
            return None
        return records[0][0], np.array([record[1] for record in records], dtype=np.float64)


class BORNearToFar:
    """Equivalent-current samples of a closed (r, z) cell box per driven solve (body_of_revolution.far_field_samples)
    with the azimuthal order solved; mirror_sign +1 / -1 mirrors a box bottom on a pmc / pec z_lo wall."""

    def __init__(self, r_cells, z_cells, mirror_sign):
        self.r_cells = (int(r_cells[0]), int(r_cells[1]))
        self.z_cells = (int(z_cells[0]), int(z_cells[1]))
        self.mirror_sign = mirror_sign
        # omega -> (azimuthal order, far_field_samples tuple), or of an order sum (orders tuple, [(signed order,
        # far_field_samples)])
        self._samples = {}
        self._pending = []   # [(signed order, far_field_samples)] of the order sum being solved

    def record_order(self, physics, system, terms):
        """Equivalent-current samples of one order's terms (+m, -m) of a driven order sum; the first listed order
        starts a new sum (a failed solve leaves its terms behind)."""
        if abs(terms[0][0]) == physics.driven_orders[0]:
            self._pending = []
        self._pending.extend((order, far_field_samples(system, fields, self.r_cells, self.z_cells, self.mirror_sign))
                             for order, fields in terms)

    def record(self, physics):
        if physics.driven_orders:
            self._samples[float(physics.solved_omega)] = (tuple(physics.driven_orders), self._pending)
            self._pending = []
            return
        system = physics.system
        self._samples[float(physics.solved_omega)] = (system.azimuthal_order, far_field_samples(
            system, physics.yee(), self.r_cells, self.z_cells, self.mirror_sign))

    def recorded(self):
        """(omega float64 in ascending wavelength order, list of (azimuthal order, far_field_samples) or of an order
        sum (orders tuple, [(signed order, far_field_samples)])); ValueError before a solve."""
        return _ascending_wavelength(self._samples, f'far-field box cells {self.r_cells} x {self.z_cells}')


def bor_probe_result(probe) -> dict:
    """fdfd_monitors.probe_phasor_result of a BOR field probe (y label: the component)."""
    return probe_phasor_result(probe, BORRenderer.field_infos[probe.field_component].label.replace('Re ', ''))


def bor_flux_result(monitor) -> dict:
    """'flux_spectrum' of a BORFluxMonitor: time-averaged power [W] (2 pi ring weights) vs ascending wavelength; of a
    driven order sum the total, extras power_orders (n_wavelengths, n_orders: per listed m with its -m) and
    azimuthal_orders."""
    omegas, powers = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    result = {'kind': 'flux_spectrum', 'frequency': frequency, 'wavelength': C0 / frequency, 'power': powers,
              'unit': 'W'}
    order_powers = monitor.recorded_orders()
    if order_powers is not None:
        result['azimuthal_orders'], result['power_orders'] = list(order_powers[0]), order_powers[1]
    return result


def bor_resonance_table_result(table, physics) -> dict:
    """eigenfrequency.resonance_table_result plus the last column azimuthal_order (the order of each resonance; rows
    order by order as the eigen-solve lists them) and the extra azimuthal_orders (the solved list)."""
    result = resonance_table_result(table, physics)
    resonances = physics.eigen_solve.resonances
    result['columns'] = list(result['columns']) + ['azimuthal_order']
    result['units'] = list(result['units']) + ['1']
    result['rows'] = [list(row) + [resonance.azimuthal_order] for row, resonance in zip(result['rows'], resonances)]
    result['azimuthal_orders'] = list(physics.eigen_solve.info['azimuthal_orders'])
    return result


def bor_far_field_angles(n_angles):
    """(angles (n,) [rad, counter-clockwise from +r toward +z in the drawing plane], polar angles theta from +z)."""
    n_angles = int(n_angles)
    if n_angles < 1:
        raise ValueError(f'n_angles must be >= 1, got {n_angles}')
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    return angles, np.arccos(np.clip(np.sin(angles), -1.0, 1.0))


def bor_far_field_result(monitor, n_angles=360) -> dict:
    """'far_field' of a BORNearToFar vs ascending wavelength: 'power' (n_wavelengths, n_angles) radiated power per
    steradian r^2 |E|^2 / (2 eta0) [W/sr] in the meridional plane (angle from +r toward +z; the half with cos < 0 is
    phi = pi, the same power), extras power_theta / power_phi (the two polarisations), azimuthal_order (of the last
    solved wavelength). A driven order sum: the summed field's pattern (cos < 0 at phi = pi, a different power), extras
    power_azimuthal_mean (phi average), power_orders (n_wavelengths, n_orders, n_angles: phi average per listed m with
    its -m) and azimuthal_orders instead of azimuthal_order."""
    omegas, samples = monitor.recorded()
    angles, theta = bor_far_field_angles(n_angles)
    frequency = omegas / (2.0 * math.pi)
    result = {'kind': 'far_field', 'frequency': frequency, 'wavelength': C0 / frequency, 'angles': angles,
              'unit': 'W/sr', 'quantity': FAR_FIELD_QUANTITY, 'angle_axes': ('r', 'z')}
    order_sums = [isinstance(entry[0], tuple) for entry in samples]
    if not any(order_sums):
        parts = [far_field_power(order, omega, sample, theta) for omega, (order, sample) in zip(omegas, samples)]
        result['azimuthal_order'] = int(samples[-1][0])
    elif all(order_sums) and len({entry[0] for entry in samples}) == 1:
        phi = np.where(np.cos(angles) < 0.0, math.pi, 0.0)
        parts = [order_sum_far_field(terms, omega, theta, phi) for omega, (_, terms) in zip(omegas, samples)]
        orders = samples[0][0]
        result.update(azimuthal_orders=list(orders),
                      power_azimuthal_mean=np.array([sum(part[3].values()) for part in parts]),
                      power_orders=np.array([[part[3][order] for order in orders] for part in parts]))
    else:
        raise ValueError(f'far-field box cells {monitor.r_cells} x {monitor.z_cells}: its wavelengths mix single-order '
                         'and order-sum solves (or different order lists)')
    result.update(power=np.array([part[0] for part in parts]), power_theta=np.array([part[1] for part in parts]),
                  power_phi=np.array([part[2] for part in parts]))
    return result

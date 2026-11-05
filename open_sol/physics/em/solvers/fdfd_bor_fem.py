"""Body-of-revolution resonances on a TriangleDomain (em_fdfd_bor on a 'triangles' mesh, eigenfrequency study only):
FEMBORPhysics solves physics/em/fem_bor per azimuthal order on the host and exposes the selected resonance per element
(state) and as nodal fem_field coefficients (renderer, Session)."""
import time

from ..body_of_revolution import BORMaterials, axis_class, distinct_orders, merge_order_solves
from ..fem_bor import (FEM_BOR_SIZE_REMEDY, FEMBORAssembly, element_means, fem_bor_layout, nodal_fields,
                       resonance_coefficients, solve_fem_bor_resonances)
from ..harmonic import shown_phasor_fields
from ..renderer_bor import BORRenderer
from .fdfd_bor import BOR_MATERIAL_SCHEMA_KEY, BOR_PERMITTIVITY_ARRAYS, BOR_PERMITTIVITY_SLOTS, BOR_POLARIZATION
from .fem_em_base import TriangleEMPhysics


class FEMBORPhysics(TriangleEMPhysics):
    """Resonances of a body of revolution on a fem.domain.TriangleDomain (x = r from the x_lo window edge = the axis,
    y = z): per azimuthal order of azimuthal_orders the n resonances nearest the wavelength (fem_bor, one assembly per
    solve); the selected one feeds the per-element state and fem_field (nodal values in the element degree)."""
    # BORPhysics' attribute set for the eigenfrequency study, placement and couplings (polarization 'bor', axisymmetric
    # ring weights, eigen_solve / selected_resonance for the resonance_table), minus every driven piece (sources,
    # monitors, ports, line-block factors: refused on triangles by validation).

    type_label = 'em_fdfd_bor'
    tensor_class = 'BORPhysics'
    boundary_hint = '; x_lo = the axis'
    state_fields = ('e_intensity', 'q_abs')
    permittivity_arrays = BOR_PERMITTIVITY_ARRAYS   # eps_r (E_phi), eps_xx (E_r), eps_yy (E_z)
    permittivity_slots = BOR_PERMITTIVITY_SLOTS
    material_names = BORRenderer.material_names
    renderer_field_infos = BORRenderer.field_infos
    size_remedy = FEM_BOR_SIZE_REMEDY

    def __init__(self, domain, boundary_config, wavelength=1.55e-6, azimuthal_order=0, azimuthal_orders=None,
                 eigen_solver=None, order=2):
        self._init_triangle(domain, boundary_config, BOR_POLARIZATION, BOR_MATERIAL_SCHEMA_KEY, order, eigen_solver)
        self.omega = self._omega_of(wavelength)
        axis_class(azimuthal_order)   # integer check
        self.azimuthal_order = int(azimuthal_order)
        self.azimuthal_orders = azimuthal_orders
        self.axisymmetric = True      # Session ring weights
        self.eigen_solve = None       # eigenfrequency.EigenSolve of the last solve_eigenfrequencies
        self.selected_resonance = None   # 1-based
        self._assembly = None         # fem_bor.FEMBORAssembly of the last eigen solve
        self._coefficients = None     # stacked [e; w] of the selected resonance
        self._selected_order = None

    @property
    def assembly(self):
        """fem_bor.FEMBORAssembly of the last eigen solve (None before one)."""
        return self._assembly

    @property
    def azimuthal_orders(self) -> tuple:
        """Orders the eigen-solve covers: the order list, else (azimuthal_order,)."""
        return self._azimuthal_orders or (self.azimuthal_order,)

    @azimuthal_orders.setter
    def azimuthal_orders(self, orders):
        self._azimuthal_orders = distinct_orders(orders, 'azimuthal_orders')

    # --- study hooks ---

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances nearest the wavelength setting per order of
        azimuthal_orders, all from one assembly (fem_bor) -> one EigenSolve (order by order), kept as eigen_solve."""
        start = time.perf_counter()
        self.eigen_solve = None
        omega = self.omega
        materials = self._bor_materials(omega)
        layout = self._current_layout(materials.pec)
        issue = self._size_issue(layout)
        if issue is not None and issue[0] == 'error':
            raise ValueError(f'eigenfrequency solve: {issue[1]}')
        assembly = FEMBORAssembly(layout, materials, omega)
        solves = [solve_fem_bor_resonances(assembly.system(order), n_resonances, eigen_solver=self.eigen_solver)
                  for order in self.azimuthal_orders]
        self.eigen_solve = merge_order_solves(solves, () if issue is None else (issue[1],))
        self.eigen_solve.info.update(n_elements=int(self.mesh.n_elements), order=self.order,
                                     assembly_seconds=assembly.assembly_seconds)
        self._assembly = assembly
        self.selected_resonance = self._coefficients = self._nodal = None
        self.host_factor_bytes = self.eigen_solve.info['factor_host_bytes']
        self.n_solves += 1
        self.last_solve_seconds = time.perf_counter() - start
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last eigenfrequency solve in the per-element state and fem_field (H
        from the curl at the target omega)."""
        resonance = self.eigen_solve.resonances[number - 1]
        assembly = self._assembly
        self._selected_order = resonance.azimuthal_order
        self._coefficients = resonance_coefficients(assembly.layout, resonance.azimuthal_order, resonance.field)
        intensity, absorbed = element_means(assembly, self._coefficients, resonance.azimuthal_order)
        self._write_element_fields({'e_intensity': intensity, 'q_abs': absorbed})
        self.selected_resonance = number
        self.solved_omega = assembly.omega

    def selection_label(self):
        """'resonance k: m = <m>, lambda = ..., Q = ...' while a resonance is shown, else None."""
        return self._resonance_label()

    # --- TriangleEMPhysics hooks ---

    def _has_shown_field(self) -> bool:
        return self.selected_resonance is not None

    def _shown_nodal_fields(self, nodal_dofs) -> dict:
        fields = nodal_fields(self._assembly, self._coefficients, self._selected_order, nodal_dofs)
        return shown_phasor_fields(fields, ('er', 'ephi', 'ez'), ('hr', 'hphi', 'hz'))   # BORRenderer fields

    def _last_unknowns(self):
        return None if self.eigen_solve is None else self.eigen_solve.info['n_unknowns']

    def _layout_unknowns(self, layout) -> int:
        return max(free.size for free in layout.free_by_class)   # the largest free set (one factor per axis class)

    def _build_layout(self, pec):
        return fem_bor_layout(self.mesh, self.order, self.boundary_config, pec)

    def _bor_materials(self, omega):
        """BORMaterials at omega: host complex128 per-element eps of the three components (permittivity array, sigma_e,
        poles) and mu (sigma_m), host bool PEC elements."""
        eps, mu, pec = self._element_materials(omega)
        return BORMaterials(eps=eps['eps_r'], mu=mu, pec=pec, eps_rr=eps['eps_xx'], eps_zz=eps['eps_yy'])

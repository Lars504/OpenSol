"""Conical resonances on a TriangleDomain (em_fdfd_conical on a 'triangles' mesh, eigenfrequency study only):
FEMConicalPhysics solves physics/em/fem_conical at the k_z setting on the host and exposes the selected resonance per
element (state) and as nodal fem_field coefficients (renderer, Session)."""
import time

from ..conical import COMPONENTS
from ..fem_conical import (FEM_CONICAL_SIZE_REMEDY, absorbed_power_means, assemble_fem_conical, fem_conical_layout,
                           resonance_field, solve_fem_conical_resonances)
from ..fem_modes import FEMModeMaterials, element_means, mode_error_indicator, nodal_fields
from ..harmonic import shown_phasor_fields
from ..renderer_conical import ConicalRenderer
from .fdfd_conical import (CONICAL_MATERIAL_SCHEMA_KEY, CONICAL_PERMITTIVITY_ARRAYS, CONICAL_PERMITTIVITY_SLOTS,
                           CONICAL_POLARIZATION, _real_wavenumber)
from .fem_em_base import TriangleEMPhysics


_E_NAMES = tuple(f'e{component}' for component in COMPONENTS)
_H_NAMES = tuple(f'h{component}' for component in COMPONENTS)


class FEMConicalPhysics(TriangleEMPhysics):
    """Resonances of a z-invariant structure at fixed k_z on a fem.domain.TriangleDomain: the n nearest the wavelength
    setting (fem_conical, one assembly per solve); the selected one feeds the per-element state and fem_field (nodal
    values in the element degree)."""
    # ConicalPhysics' attribute set for the eigenfrequency study, placement and couplings (polarization 'conical',
    # eigen_solve / selected_resonance for the resonance_table), minus every driven piece (sources, monitors: refused on
    # triangles by validation).

    type_label = 'em_fdfd_conical'
    tensor_class = 'ConicalPhysics'
    state_fields = ('e_intensity', 'q_abs')
    permittivity_arrays = CONICAL_PERMITTIVITY_ARRAYS   # eps_r (= eps_zz, E_z), eps_xx (E_x), eps_yy (E_y)
    permittivity_slots = CONICAL_PERMITTIVITY_SLOTS
    material_names = ConicalRenderer.material_names
    renderer_field_infos = ConicalRenderer.field_infos
    size_remedy = FEM_CONICAL_SIZE_REMEDY

    def __init__(self, domain, boundary_config, wavelength=1.55e-6, k_z=0.0, eigen_solver=None, order=2):
        self._init_triangle(domain, boundary_config, CONICAL_POLARIZATION, CONICAL_MATERIAL_SCHEMA_KEY, order,
                            eigen_solver)
        self.omega = self._omega_of(wavelength)
        self.k_z = _real_wavenumber(k_z)
        self.eigen_solve = None       # eigenfrequency.EigenSolve of the last solve_eigenfrequencies
        self.selected_resonance = None   # 1-based
        self._system = None           # fem_conical.FEMConicalSystem of the last eigen solve
        self._field = None            # fem_conical.FEMConicalField of the selected resonance

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the shown resonance for mesh refinement: fem_modes'
        mode_error_indicator (tangential H / normal D jumps); ValueError before a selection."""
        if self.selected_resonance is None:
            raise ValueError('em_fdfd_conical on triangles: no selected resonance to estimate the mesh error of')
        return mode_error_indicator(self._system.layout, self._field, self._system.materials)

    # --- study hooks ---

    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances at k_z nearest the wavelength setting (fem_conical
        on the current materials, one assembly), kept as eigen_solve."""
        start = time.perf_counter()
        self.eigen_solve = None
        omega = self.omega
        materials = self._materials(omega)
        layout = self._current_layout(materials.pec)
        issue = self._size_issue(layout)
        if issue is not None and issue[0] == 'error':
            raise ValueError(f'eigenfrequency solve: {issue[1]}')
        system = assemble_fem_conical(layout, materials, omega, self.k_z)
        self.eigen_solve = solve_fem_conical_resonances(system, int(n_resonances), eigen_solver=self.eigen_solver,
                                                        warnings=() if issue is None else (issue[1],))
        self.eigen_solve.info.update(n_elements=int(self.mesh.n_elements), order=self.order, k_z=self.k_z,
                                     assembly_seconds=system.assembly_seconds)
        self._system = system
        self.selected_resonance = self._field = self._nodal = None
        self.host_factor_bytes = self.eigen_solve.info['factor_host_bytes']
        self.n_solves += 1
        self.last_solve_seconds = time.perf_counter() - start
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last eigenfrequency solve in the per-element state and fem_field (H
        from the curl at the target omega)."""
        system = self._system
        self._field = resonance_field(system, self.eigen_solve.resonances[number - 1].field)
        intensity, _ = element_means(system.layout, self._field, system.materials.mu)
        self._write_element_fields({'e_intensity': intensity, 'q_abs': absorbed_power_means(system, self._field)})
        self.selected_resonance = number
        self.solved_omega = system.omega

    def selection_label(self):
        """'resonance k: lambda = ..., Q = ...' while a resonance is shown, else None."""
        return self._resonance_label()

    # --- TriangleEMPhysics hooks ---

    def _has_shown_field(self) -> bool:
        return self.selected_resonance is not None

    def _shown_nodal_fields(self, nodal_dofs) -> dict:
        fields = nodal_fields(self._system.layout, self._field, self._system.materials.mu, nodal_dofs)
        return shown_phasor_fields(fields, _E_NAMES, _H_NAMES, sz=True)   # ConicalRenderer fields

    def _last_unknowns(self):
        return None if self.eigen_solve is None else self.eigen_solve.info['n_unknowns']

    def _build_layout(self, pec):
        return fem_conical_layout(self.mesh, self.order, self.boundary_config, pec)

    def _materials(self, omega):
        """FEMModeMaterials at omega: host complex128 per-element eps_xx / eps_yy / eps_zz (sigma_e, poles) and mu
        (sigma_m), host bool PEC elements."""
        eps, mu, pec = self._element_materials(omega)
        return FEMModeMaterials(eps_xx=eps['eps_xx'], eps_yy=eps['eps_yy'], eps_zz=eps['eps_r'], mu=mu, pec=pec)

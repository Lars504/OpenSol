"""Cross-section eigenmodes on a TriangleDomain (em_modes on a 'triangles' mesh, mesh plan 2.6): FEMModePhysics solves
physics/em/fem_modes on the host and exposes the selected mode per element (state) and as nodal / per-element
fem_field coefficients (renderer, Session); the ModePhysics table / dispersion monitors record every solve."""
import math

from ....core.constants import C0
from ....core.gpu_memory import format_bytes
from ....core.wavelengths import wavelength_label
from ..cross_section_modes import GROUP_RELATIVE_STEP
from ..fem_modes import (FEMModeMaterials, element_means, fem_group_indices, fem_mode_layout, mode_error_indicator,
                         nodal_fields, solve_fem_modes)
from ..harmonic import shown_phasor_fields
from ..renderer import ModeRenderer
from .fem_em_base import TriangleEMPhysics
from .modes_2d import (MODE_PERMITTIVITY_ARRAYS, MODE_PERMITTIVITY_SLOTS, MODE_POLARIZATION, _ModeStudyMixin)



class FEMModePhysics(_ModeStudyMixin, TriangleEMPhysics):
    """Full-vector cross-section modes at omega on a fem.domain.TriangleDomain: n_modes modes (descending Re beta^2) of
    per-element diagonal eps and mu, solved by the study hooks solve_modes / solve_harmonic; the selected one feeds
    the per-element state and fem_field (nodal values in the element degree)."""
    # ModePhysics' attribute set and study hooks (placement / couplings / studies treat both alike), minus the tensor
    # extras (bend, Bloch, line-block batches, shg / bend overlaps, point_materials: refused on triangles by
    # validation). State fields and source slots are per element (field_dofs: P0).

    name = 'modes'
    type_label = 'em_modes'
    tensor_class = 'ModePhysics'
    state_fields = ('e_intensity', 'sz')   # per-element state fields of the selected mode
    permittivity_arrays = MODE_PERMITTIVITY_ARRAYS   # eps_r (= eps_zz), eps_xx, eps_yy
    permittivity_slots = MODE_PERMITTIVITY_SLOTS
    material_names = ModeRenderer.material_names
    renderer_field_infos = ModeRenderer.field_infos

    def __init__(self, domain, boundary_config, wavelength=1.55e-6, n_modes=4, mode_index=1, n_eff_guess=None,
                 group_index=False, eigen_solver=None, order=2):
        self._init_triangle(domain, boundary_config, MODE_POLARIZATION, MODE_POLARIZATION, order, eigen_solver)
        self._init_mode_study(wavelength, n_modes, mode_index, n_eff_guess, group_index)
        self._solve_layout = None     # layout of self.modes
        self._solved_materials = None   # FEMModeMaterials of the last solve (mu: recovery of H; error indicator)

    # --- study hooks (+ _ModeStudyMixin) ---

    def select_mode(self, number):
        """Show mode `number` (1-based, clamped to the modes found) in the per-element state and fem_field; zeros
        without modes."""
        self._nodal = None
        if not self.modes:
            self.selected_mode = None
            for values in self.element_fields.values():
                values.fill(0)
            return
        number = min(max(int(number), 1), len(self.modes))
        intensity, flow = element_means(self._solve_layout, self.modes[number - 1], self._solved_materials.mu)
        self._write_element_fields({'e_intensity': intensity, 'sz': flow})
        self.selected_mode = number

    def solve_summary(self):
        """One line on the last solve for the study log; None before."""
        if self.last_solve_seconds is None:
            return None
        info = self.last_solve_info
        degree = 'quadratic' if info['order'] == 2 else 'linear'
        kind = info.get('factor_kind', 'SuperLU')
        if kind.startswith('cuDSS'):   # the eigen solver's factorization cudss / cudss_mixed
            factor = (f'{kind} on the GPU {info["factor_seconds"]:.3g} s, factor + operator '
                      f'{format_bytes(info["factor_device_bytes"])} of device memory')
            if 'single_residual' in info:
                factor += f', unrefined residual {info["single_residual"]:.2g}'
            if 'rayleigh_change' in info:
                factor += (f', loose solves + Rayleigh quotients (n_eff^2 moved by up to '
                           f'{info["rayleigh_change"]:.2g})')
        else:
            factor = (f'SuperLU factorisation {info["factor_seconds"]:.3g} s, L + U factors '
                      f'{format_bytes(info["factor_host_bytes"])} of host memory')
        return (f'{len(self.modes)} of {self.n_modes} modes at '
                f'{wavelength_label(2.0 * math.pi * C0 / self.solved_omega)}: '
                f'{info["n_unknowns"]:,} unknowns ({degree} elements on {info["n_elements"]:,} triangles), solve '
                f'{self.last_solve_seconds:.3g} s{" with the group index" if self.group_index else ""} ({factor}; '
                f'eigs {info["eigen_seconds"]:.3g} s, {info["operator_applications"]} operator applications, '
                f'{info["eigenvalues_converged"]} of {info["eigenvalues_requested"]} eigenvalues, '
                f'{info["pml_modes_dropped"]} PML modes dropped)')

    def error_indicator(self):
        """eta^2 per element (float64 (n_elements,)) of the shown mode for mesh refinement (S7.7, mesh plan A3):
        fem_modes.mode_error_indicator; ValueError before a solve or without modes."""
        if self.selected_mode is None:
            raise ValueError('em_modes on triangles: no solved mode to estimate the mesh error of')
        return mode_error_indicator(self._solve_layout, self.modes[self.selected_mode - 1], self._solved_materials)

    def _solve_once(self, omega):
        """(modes, solve info, warnings) of one solve at omega on the current materials (group indices when enabled)."""
        materials = self._materials(omega)
        layout = self._current_layout(materials.pec)
        solve = solve_fem_modes(layout, materials, omega, self.n_modes, n_eff_guess=self.n_eff_guess,
                                eigen_solver=self.eigen_solver)
        modes = list(solve.modes)
        warnings = list(solve.info['warnings'])
        if self.group_index and modes:   # one solve: the Hellmann-Feynman form of the complex symmetric pencil
            modes, group_warnings = fem_group_indices(layout, self._materials, omega, modes, GROUP_RELATIVE_STEP)
            warnings.extend(group_warnings)
        self._solve_layout = layout
        self._solved_materials = materials
        return modes, solve.info, warnings

    # --- TriangleEMPhysics hooks ---

    def _has_shown_field(self) -> bool:
        return self.selected_mode is not None

    def _shown_nodal_fields(self, nodal_dofs) -> dict:
        fields = nodal_fields(self._solve_layout, self.modes[self.selected_mode - 1], self._solved_materials.mu,
                              nodal_dofs)   # ModeRenderer fields: Re of the six components, moduli, S_z
        return shown_phasor_fields(fields, ('ex', 'ey', 'ez'), ('hx', 'hy', 'hz'), abs_parts=False, sz=True)

    def _last_unknowns(self):
        return None if self.last_solve_info is None else self.last_solve_info['n_unknowns']

    def _extra_counters(self) -> dict:
        return {'modes_found': self.n_modes_found, 'solve_seconds': self.total_solve_seconds}

    def _build_layout(self, pec):
        return fem_mode_layout(self.mesh, self.order, self.boundary_config, pec)

    def _materials(self, omega):
        """FEMModeMaterials at omega: host complex128 per-element eps_xx / eps_yy / eps_zz (sigma_e, poles) and mu
        (sigma_m), host bool PEC elements."""
        eps, mu, pec = self._element_materials(omega)
        return FEMModeMaterials(eps_xx=eps['eps_xx'], eps_yy=eps['eps_yy'], eps_zz=eps['eps_r'], mu=mu, pec=pec)

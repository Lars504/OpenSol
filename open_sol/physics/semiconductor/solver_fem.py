"""drift_diffusion on a triangle mesh: FEMDriftDiffusionPhysics, the DriftDiffusionPhysics API over the box-method
FEMDriftDiffusionSystem (P1 nodes of the window elements). State arrays per element (D12: V, N, P, phi at the element
centroid, E / J element vectors); V, N, P, phi_n, phi_p rendered nodal (P1), the rest per element."""
import time

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE
from ...fem.diffusion import ring_dof_locations
from ...fem.dofs import lagrange_dofs
from ...fem.renderer import FEMFieldRenderer
from ...fem.sampling import evaluate
from ...solvers.host_direct import factor_label
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, SPLU_LIMIT_UNKNOWNS, SPLU_WARN_UNKNOWNS
from .drift_diffusion import COMPONENTS
from .drift_diffusion_fem import FEMDriftDiffusionSystem
from .renderer import SEMICONDUCTOR_FIELD_INFOS
from .solver import DriftDiffusionPhysics

NODAL_FIELDS = ('V', 'N', 'P', 'phi_n', 'phi_p')   # P1 nodal in fem_field; their state arrays = centroid means
_CARRIER_FIELDS = ('N', 'P', 'phi_n', 'phi_p')     # 0 in insulator elements
_SMALLER_MODEL = 'coarsen the triangle mesh (larger elements away from junctions) or shrink the domain'


def fem_linear_size_issue(back_end, n_nodes):
    """(level, text) of the 3-unknown nodal system of n_nodes P1 nodes on the 'linear' back end, or None (always on
    cuDSS: guarded by its own device estimate); cusolver_qr is refused on triangles (no grid index order); MUMPS /
    PARDISO keep the SuperLU limits (uncalibrated), named by their library. CPU only."""
    if back_end in CUDSS_BACK_ENDS:
        return None
    if back_end == 'cusolver_qr':
        return 'error', ("the sparse_direct back end 'cusolver_qr' needs a tensor mesh (a grid index order); on a "
                         "triangle mesh use Automatic or back_end 'auto' / 'scipy_splu' (host SuperLU)")
    unknowns = COMPONENTS * int(n_nodes)
    text = f'{unknowns:,} unknowns ({int(n_nodes):,} mesh nodes x 3: potential, electrons, holes)'
    library = factor_label(back_end)
    if unknowns > SPLU_LIMIT_UNKNOWNS:
        return 'error', (f'{text} exceed the {SPLU_LIMIT_UNKNOWNS:,} limit of the host {library} solver: its factors '
                         f'would fill the host memory; {_SMALLER_MODEL}')
    if unknowns > SPLU_WARN_UNKNOWNS:
        return 'warning', (f'{text} exceed ~{SPLU_WARN_UNKNOWNS:,}, the practical size of host {library}; '
                           f'{_SMALLER_MODEL}')
    return None


class FEMDriftDiffusionPhysics(DriftDiffusionPhysics):
    """2D stationary drift-diffusion on a fem.domain.TriangleDomain (box method on the P1 nodes), planar or
    axisymmetric (r, z); the feature API, results and counters of DriftDiffusionPhysics."""

    def prepare(self, domain) -> None:
        """Register the material arrays, allocate the per-element fields and the nodal system (features come later)."""
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError(f"drift_diffusion '{self.name}': FEMDriftDiffusionPhysics needs a triangle mesh")
        self.domain = domain
        self.xp = domain.xp
        mesh = domain.mesh
        domain.materials.add_properties(self.material_schema)
        self.window = slice(None)   # every per-element array is whole (the base class indexes with it)
        self.fields = {name: self.xp.zeros(mesh.n_elements, dtype=DTYPE) for name in SEMICONDUCTOR_FIELD_INFOS}
        tolerance, max_iterations, max_bias_step = self._newton_settings
        self.system = FEMDriftDiffusionSystem(mesh, label=lambda: f"drift_diffusion '{self.name}'",
                                              tolerance=tolerance, max_newton_iterations=max_iterations,
                                              max_bias_step=max_bias_step, axisymmetric=self.axisymmetric,
                                              linear_solver=self.linear_solver)
        self.full = lagrange_dofs(mesh, 1, geometry_order=1)          # nodal fields (the P1 nodes)
        self.element_map = lagrange_dofs(mesh, 0, geometry_order=1)   # per-element fields and slots
        self.nodal = {name: np.zeros(mesh.n_points) for name in NODAL_FIELDS}
        self._ring = None
        self.runtime_warnings = []   # appended once per new count of negative edge coefficients
        self._reported_negative_faces = 0
        self._dirty = True

    def finalize(self) -> None:
        """Size rule of the nodal system on the resolved back end (ValueError for cusolver_qr or over the limit)."""
        issue = fem_linear_size_issue(self.linear_solver.back_end, self.system.n_cells)
        if issue is not None and issue[0] == 'error':
            raise ValueError(issue[1])
        self._finalized = True
        self._dirty = True

    def renderer(self):
        return FEMFieldRenderer(self, 'semiconductor', SEMICONDUCTOR_FIELD_INFOS, lambda: f'solve {self.n_solves}')

    def field_dofs(self, name):
        """fem DofMap of state field or source slot `name`: every one is per element (P0)."""
        if name not in self.fields:
            raise ValueError(f"drift_diffusion '{self.name}' has no state field or source slot '{name}'")
        return self.element_map

    def fem_field(self, component) -> tuple:
        """(DofMap, host float64 coefficients): V, N, P, phi_n, phi_p nodal (P1), the rest per element."""
        if component in self.nodal:
            return self.full, self.nodal[component]
        if component in self.fields:
            return self.element_map, np.asarray(to_host(self.fields[component]), dtype=np.float64)
        raise KeyError(f"drift_diffusion '{self.name}' has no field '{component}'")

    def set_conformal_boundaries(self, locator) -> None:
        """Nothing to do: the mesh follows the drawn outlines."""

    # --- internals ------------------------------------------------------------

    def _solve(self):
        if not self._finalized:
            self.finalize()
        start = time.perf_counter()
        xp = self.xp
        system = self.system
        materials = self.domain.materials
        host_materials = {name: np.asarray(to_host(getattr(materials, name)), dtype=np.float64)
                          for name in self.material_schema}
        generation = (np.asarray(to_host(self.fields['generation']), dtype=np.float64)
                      + np.asarray(to_host(self.fields['generation_ext']), dtype=np.float64))
        system.set_inputs(host_materials, generation, self.thermal_voltage)
        system.solve()
        nodal, element = system.outputs()
        if self._ring is None:
            self._ring = ring_dof_locations(self.full)
        ring_ids, ring_location = self._ring
        for name, values in nodal.items():
            if ring_ids.size:   # PML ring nodes: the window field at the clamped position (edge extension)
                values[ring_ids] = evaluate(self.full, values, location=ring_location)
            self.nodal[name] = values
        # per-element state: centroid means of the nodal fields (carriers 0 in insulator elements, ring elements by
        # their extruded material)
        triangles = self.domain.mesh.triangles
        semiconductor = host_materials['n_intrinsic'] > 0.0
        for name in NODAL_FIELDS:
            centroid = self.nodal[name][triangles].mean(axis=1)
            if name in _CARRIER_FIELDS:
                centroid = np.where(semiconductor, centroid, 0.0)
            self.fields[name][...] = xp.asarray(centroid, dtype=DTYPE)
        for name, values in element.items():
            self.fields[name][...] = xp.asarray(values, dtype=DTYPE)
        if system.negative_faces != self._reported_negative_faces:   # runtime_warnings is append-only (study log)
            self._reported_negative_faces = system.negative_faces
            if system.negative_faces:
                self.runtime_warnings.append(
                    f'{system.negative_faces} mesh edge coefficient(s) are negative (non-Delaunay edges, typically an '
                    'obtuse triangle at a material interface): carrier densities may undershoot there; refine the mesh')
        self.n_solves += 1
        self._dirty = False
        self._solve_seconds += time.perf_counter() - start

    def _device_window_mask(self, mask, label='mask'):
        """bool per-element mask on the run backend; ValueError for another shape."""
        mask = self.xp.asarray(mask)
        count = self.domain.mesh.n_elements
        if mask.shape != (count,):
            raise ValueError(f'{label}: mask must be a per-element ({count},) boolean array, got shape {mask.shape}')
        return mask.astype(np.bool_)

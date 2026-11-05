"""FEMResonancePhysics3D: em_fdfd_3d on a 'tetrahedra' mesh (igor3d_plan T4b): the PhysicsSolver around
fem3d.FEMPhysics3D with per-element materials read from the TetrahedronDomain's MaterialGrid at every assembly, the
eigenfrequency study hooks of FDFDPhysics3D and FEMRenderer3D (point samples for the Session)."""
# Materials per element at omega_t (host): eps_xx / eps_yy / eps_r (eps_r for an unallocated component) - j sigma_e /
# (omega eps0) + each pole's chi on its element mask, mu_r - j sigma_m / (omega mu0), PEC = the union of the pec_domain
# element masks (their elements and every dof they touch drop out). Subpixel smoothing and conformal boundaries do not
# apply (schema: element meshes follow the outlines; gmsh conforms to spheres / boxes / cylinders only).
import numpy as np

from ...core.backend import to_host
from ...core.solver_base import PhysicsSolver
from ...fem.mesh3d import POOR_RADIUS_RATIO
from .fem3d import FEMPhysics3D
from .physics3d import MATERIAL_ARRAYS, StaticEM3DContract, check_pole, complex_cell_materials
from .renderer_fem3d import FEMRenderer3D


class FEMResonancePhysics3D(StaticEM3DContract, PhysicsSolver):
    """3D resonances on a fem.domain3d.TetrahedronDomain (BoundaryConfig3D faces cpml / pec / pmc / mirrors /
    periodic with Bloch k on matching mesh sides): solve_eigenfrequencies / select_resonance as FDFDPhysics3D, PEC and
    dispersive poles on per-element masks, permittivity slots for the thermo-optic coupling."""

    name = 'em'
    type_label = 'em_fdfd_3d'

    def __init__(self, domain, boundary_config, wavelength=1.55e-6, eigen_solver=None, element_order=2):
        if getattr(domain, 'grid_kind', 'tensor') != 'tetrahedra':
            raise ValueError('FEMResonancePhysics3D needs a fem.domain3d.TetrahedronDomain (a tetrahedra mesh); '
                             'tensor grids take FDFDPhysics3D')
        boundary_config.validate()
        self.domain = domain
        self.xp = domain.xp
        self.boundary_config = boundary_config
        self.pec_masks = []            # run-backend bool (n_elements,) PEC element masks
        self.dispersive_regions = []   # [(pole, run-backend bool (n_elements,) mask)]
        self.finalize_warnings = []
        self.eigen_solve = None
        self.selected_resonance = None
        self.host_factor_bytes = None
        self.fem = FEMPhysics3D(domain.mesh, boundary_config, materials=self._element_materials,
                                element_order=element_order, wavelength=wavelength, eigen_solver=eigen_solver)

    @property
    def omega(self) -> float:
        """Target angular frequency [rad/s] (the wavelength setting)."""
        return self.fem.omega

    # ---- PhysicsSolver contract (+ StaticEM3DContract) ----
    def finalize(self):
        """finalize_warnings = a note on slivers (TetrahedronQuality.n_poor: radius ratio below POOR_RADIUS_RATIO and an
        obtuse dihedral angle)."""
        quality = self.domain.mesh.quality
        self.finalize_warnings = [] if not quality.n_poor else [
            f'{quality.n_poor:,} of {self.domain.n_elements:,} tetrahedra have a radius ratio below '
            f'{POOR_RADIUS_RATIO:g} and an obtuse dihedral angle (smallest ratio {quality.smallest:.3g}): such '
            'slivers worsen the conditioning and the accuracy; check thin gaps in the geometry or set h_min']

    def materials_changed(self, names):
        if not set(MATERIAL_ARRAYS).isdisjoint(names):
            self.fem.invalidate()

    def renderer(self):
        return FEMRenderer3D(self)

    def work_counters(self) -> dict:
        """Assemblies and their host seconds (+ the eigen solver's counters when it is private), the free dofs."""
        counters = self.fem.work_counters()
        if self.fem.system is not None:
            counters['unknowns'] = int(len(self.fem.system.free))
        return counters

    # ---- model items ----
    def add_pec(self, window, mask):
        """PEC elements on a run-backend bool per-element mask (window: all elements)."""
        self.pec_masks.append(mask)
        self.fem.invalidate()

    def set_resonance_region(self, region):
        """resonance_table selection: an eigenfrequency.ResonanceRegion with one bool per element (FEMPhysics3D
        reports the share on the dofs those elements touch and drops resonances below region.min_fraction)."""
        if region.cells.shape != (self.domain.n_elements,):
            raise ValueError(f'resonance region cells {region.cells.shape} do not match the {self.domain.n_elements} '
                             'elements')
        self.fem.resonance_region = region

    def set_subpixel_smoothing(self, geometry):
        """Ignored: element meshes follow the material outlines (schema refuses it on tetrahedra)."""
        return

    def add_dispersive(self, pole, window, mask):
        """Drude / Lorentz pole on a per-element bool mask (window: all elements); the arrays are its background."""
        check_pole(pole)
        self.dispersive_regions.append((pole, mask))
        self.fem.invalidate()

    # ---- eigenfrequency study ----
    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: FEMPhysics3D.solve_eigenfrequencies on the current materials -> EigenSolve."""
        self.eigen_solve = None
        self.selected_resonance = None
        self.eigen_solve = self.fem.solve_eigenfrequencies(n_resonances)
        self.host_factor_bytes = self.eigen_solve.info['factor_host_bytes']
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last solve (the renderer samples its Nedelec field)."""
        self.fem.select_resonance(number)
        self.selected_resonance = number

    def _element_materials(self, omega):
        """((m, 3) complex eps, (m,) complex 1 / mu, (m,) bool pec) per element at omega (module header)."""
        cell_eps, _, mu = complex_cell_materials(self.domain.materials, omega,
                                                 [(pole, Ellipsis, mask) for pole, mask in self.dispersive_regions])
        eps = np.stack(cell_eps, axis=1)
        pec = np.zeros(self.domain.n_elements, dtype=bool)
        for mask in self.pec_masks:
            pec |= np.asarray(to_host(mask), dtype=bool)
        return eps, 1.0 / mu, pec

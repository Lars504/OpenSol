"""FEMPhysics3D: 3D EM resonances on a TetrahedronMesh (igor3d_plan section 7): Nedelec first kind degree 1 / 2,
materials and a Cartesian PML shell frozen at the target omega, resonances by the em_fdfd_3d eigen layer
(eigenfrequency.solve_resonances, squared Cayley map, sparse mass, host SuperLU or a cuDSS factor). Scripts use it
directly; em_fdfd_3d on a 'tetrahedra' mesh wraps it (fem_solver3d.FEMResonancePhysics3D, igor3d_plan T4b)."""
# Formulation (exp(+j omega t), unconjugated Galerkin: complex symmetric with PML / loss): curl (Lambda / mu)^-1 curl E
# = k^2 eps Lambda E, k = omega / C0, Lambda = diag(s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z) of the complex
# coordinate stretches s = kappa + sigma / (alpha + j omega_t eps0) (em3d CPML grading over the shell width beyond each
# cpml window face, sigma_max = cpml_sigma_max at width / cpml_npml); Lambda = 1 in the window. K = int curl W . nu'
# curl W, M = int W . eps' W; A = -K + k_t^2 M on the free dofs (eigenfrequency.py sign, B = -M sparse). Walls:
# pec / antisymmetric faces and the outer side of a PML shell (COMSOL's default exterior) fix the tangential dofs;
# pmc / symmetric faces are natural (mirror planes = window faces); PEC elements drop out with every dof they touch.
# Periodic faces (Bloch, igor3d_plan T6): the mesh's two sides must match node for node with the same node order (the
# builtin Kuhn split does; fem.dofs3d.periodic_dof_partners raises otherwise); each hi-side dof stands for its lo-side
# master times e^{-j k L} per crossed period (fem.dofs3d.bloch_masters: F(x + L) = F(x) e^{-j k L}, L = the node
# extent), u = P u_free, and the test space is the conjugate one, so A_free = P^H A P (likewise K, M, W): Hermitian
# without loss / PML, not complex symmetric once a phase is complex (LU, no LDL^T or Rayleigh polish); k = 0 keeps P
# real and the system symmetric. A master is pinned with any dof it stands for. Elements hold the material of their
# region (fem.mesh3d: the last solid at the centroid clamped into the window, so the shell
# extrudes the window-edge materials). PML share: the per-row energy of W = int W . |Re eps| W (no stretch), window
# rows = dofs whose entity centre lies in the closed window. Equilibration of a cuDSS factor: (|diag K| + k_t^2 |diag
# M|)^-1/2 (cudss_factor header). Rule degree 2 x degree, + 2 in PML elements (varying Lambda).
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...fem.assembly import pattern_sum
from ...fem.assembly3d import assemble_nedelec_3d
from ...fem.dofs3d import bloch_masters, nedelec_dofs_3d
from ...fem.mesh3d import OUTER_FACES_3D
from ...fem.sampling3d import PointLocator3D, evaluate_nedelec_3d
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver, pattern_combination
from ...solvers.sparse_direct import vector_3d_size_issue
from ..em.boundaries import _grading_values, cpml_sigma_max
from ..em.eigenfrequency import solve_resonances
from .fdfd3d import ARNOLDI_BASIS_PER_WANTED, ARPACK_TOLERANCE_FLOOR

ELEMENT_ORDERS = (1, 2)
_FIXED_FACE_TYPES = frozenset(('pec', 'antisymmetric', 'cpml'))   # tangential E 0 on the mesh's outer side
# host SuperLU size rule: FEM unknowns x this ~ the Yee E unknowns of equal fill (denser rows: ~18 / ~45 nonzeros vs
# Yee's 13); uncalibrated first figures (igor3d_plan 7.2 T8)
YEE_EQUIVALENT_UNKNOWNS = {1: 1.5, 2: 3.5}


@dataclass(frozen=True)
class FEMMaterial3D:
    """Material of a mesh region: eps_r (number or (xx, yy, zz) diagonal), sigma_e [S/m], mu_r, sigma_m [ohm/m];
    pec=True makes the region a perfect conductor (its elements and every dof they touch drop out)."""
    eps_r: object = 1.0
    sigma_e: float = 0.0
    mu_r: float = 1.0
    sigma_m: float = 0.0
    pec: bool = False

    def eps_at(self, omega) -> np.ndarray:
        """(3,) complex eps_r - j sigma_e / (omega eps0) on the diagonal."""
        return np.broadcast_to(np.asarray(self.eps_r, dtype=np.complex128), (3,)) - 1j * self.sigma_e / (omega * EPS0)

    def mu_at(self, omega) -> complex:
        return complex(self.mu_r) - 1j * self.sigma_m / (omega * MU0)


@dataclass(frozen=True, eq=False)
class PMLShell3D:
    """Complex coordinate stretch of the mesh's PML shell: thickness and peak conductivity per outer face (0 without a
    shell there) with the em3d CPML grading; tensor() = Lambda at points."""
    window: tuple                 # (x0, y0, z0, x1, y1, z1) [m]
    widths: dict                  # face -> shell thickness [m]
    sigma_max: dict               # face -> peak conductivity [S/m]
    kappa_max: float
    alpha_max: float              # [S/m]
    grading_order: int

    @property
    def active(self) -> bool:
        return any(width > 0.0 for width in self.widths.values())

    def stretches(self, points, omega) -> tuple:
        """Complex128 (s_x, s_y, s_z) of shape points.shape[:-1] (1 inside the window)."""
        points = np.asarray(points, dtype=np.float64)
        result = []
        for axis, name in enumerate('xyz'):
            coordinate = points[..., axis]
            stretch = np.ones(coordinate.shape, dtype=np.complex128)
            for face, depth in ((f'{name}_lo', self.window[axis] - coordinate),
                                (f'{name}_hi', coordinate - self.window[3 + axis])):
                width = self.widths[face]
                inside = depth > 0.0
                if width <= 0.0 or not inside.any():
                    continue
                sigma, kappa, alpha = _grading_values(np.clip(depth[inside] / width, 0.0, 1.0), self.sigma_max[face],
                                                      self.kappa_max, self.alpha_max, self.grading_order)
                stretch[inside] = kappa + sigma / (alpha + 1j * float(omega) * EPS0)
            result.append(stretch)
        return tuple(result)

    def tensor(self, points, omega) -> np.ndarray:
        """(..., 3) diagonal Lambda = (s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z) at points."""
        s_x, s_y, s_z = self.stretches(points, omega)
        return np.stack([s_y * s_z / s_x, s_x * s_z / s_y, s_x * s_y / s_z], axis=-1)


def pml_shell_3d(mesh, boundary_config) -> PMLShell3D:
    """PMLShell3D of the mesh's shell (node extent beyond the window per face) with the CPML settings of a
    BoundaryConfig3D; ValueError when a shell face and the cpml faces disagree."""
    bounds = mesh.bounds
    window = tuple(float(value) for value in mesh.window)
    slack = 1e-9 * max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    widths, sigma_max = {}, {}
    for axis, name in enumerate('xyz'):
        extents = ((f'{name}_lo', window[axis] - bounds[axis]), (f'{name}_hi', bounds[3 + axis] - window[3 + axis]))
        for face, width in extents:
            width = width if width > slack else 0.0
            if (width > 0.0) != (getattr(boundary_config, face) == 'cpml'):
                raise ValueError(f"tetrahedron mesh PML shell on {face}: {'present' if width > 0.0 else 'missing'}, "
                                 f"but the face is {getattr(boundary_config, face)}; mesh and boundary settings "
                                 'disagree')
            widths[face] = width
            sigma_max[face] = cpml_sigma_max(boundary_config, width / int(boundary_config.cpml_npml)) if width else 0.0
    return PMLShell3D(window=window, widths=widths, sigma_max=sigma_max,
                      kappa_max=float(boundary_config.cpml_kappa_max), alpha_max=float(boundary_config.cpml_alpha_max),
                      grading_order=int(boundary_config.cpml_order))


@dataclass(frozen=True, eq=False)
class BlochIdentification:
    """Periodic dof identification (module header): master (n_dofs,) int64 = the dof each dof stands for (itself off
    the hi sides), phase (n_dofs,) complex128 with u = phase x u_master."""
    master: np.ndarray
    phase: np.ndarray

    @property
    def complex_phase(self) -> bool:
        return bool(np.any(self.phase != 1.0))


def bloch_identification(dofs, boundary_config):
    """BlochIdentification of the periodic axes of a BoundaryConfig3D (bloch_k [rad/m], period = the node extent
    along the axis); None without periodic faces; ValueError when the mesh sides do not match."""
    axes = [axis for axis, name in enumerate('xyz') if boundary_config.periodic(name)]
    if not axes:
        return None
    master, phase = bloch_masters(dofs, {axis: float(boundary_config.bloch_k[axis]) for axis in axes})
    return BlochIdentification(master=master, phase=phase)


@dataclass(frozen=True, eq=False)
class FEMSystem3D:
    """Host system of one assembly at omega [rad/s] on the free dofs: operator A, curl-curl K, mass M, energy matrix W
    (CSR), in_window (n,), equilibration (n,), free (n,) int64 dof indices (Bloch: the free masters), prolongation
    (n_dofs, n) CSR u = P u_free with the Bloch phases (None: plain selection of `free`), symmetric (A^T = A)."""
    omega: float
    free: np.ndarray
    operator: object
    curl: object
    mass: object
    energy: object
    in_window: np.ndarray
    equilibration: np.ndarray
    prolongation: object = None
    symmetric: bool = True


class FEMPhysics3D:
    """Resonances of a TetrahedronMesh (module header): materials {region name: FEMMaterial3D} (background for
    untagged elements) or a callable omega -> ((m, 3) complex eps, (m,) complex 1 / mu, (m,) bool pec) per element
    (the model path, em3d fem_solver3d), a BoundaryConfig3D (cpml faces need the mesh's PML shell, periodic faces
    matching mesh sides; bloch_k as em_fdfd_3d), element_order
    1 / 2, the target wavelength [m]; solve_eigenfrequencies / select_resonance as em_fdfd_3d, fields_at / plane_cut
    sample the selected resonance; invalidate() after a material change."""

    def __init__(self, mesh, boundary_config, materials=None, background=None, element_order=2, wavelength=1.55e-6,
                 eigen_solver=None):
        if element_order not in ELEMENT_ORDERS:
            raise ValueError(f'FEMPhysics3D element_order must be 1 or 2, got {element_order!r}')
        wavelength = float(wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f'FEMPhysics3D wavelength must be finite and > 0 [m], got {wavelength!r}')
        unknown = [] if callable(materials) else sorted(set(materials or {}) - set(mesh.region_names))
        if unknown:
            raise ValueError(f'FEMPhysics3D: materials for regions the mesh does not have: {unknown}')
        self.mesh = mesh
        self.boundary_config = boundary_config
        self.materials = materials if callable(materials) else dict(materials or {})
        self.background = FEMMaterial3D() if background is None else background
        self.element_order = int(element_order)
        self.omega = 2.0 * math.pi * C0 / wavelength
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.dofs = nedelec_dofs_3d(mesh, self.element_order)
        self.pml = pml_shell_3d(mesh, boundary_config)
        self.bloch = bloch_identification(self.dofs, boundary_config)   # None without periodic faces
        self.system = None             # FEMSystem3D of the last assembly
        self.eigen_solve = None
        self.selected_resonance = None  # 1-based
        self.resonance_region = None    # eigenfrequency.ResonanceRegion (cells = bool per element), None = none
        self.coefficients = None        # complex (n_dofs,) Nedelec coefficients of the selected resonance
        self._inverse_mu = None         # (m,) complex 1 / mu per element when it was selected
        self._locator = None
        self._counters = {'assemblies': 0, 'assembly_seconds': 0.0}

    # ---- materials and walls ----
    def _element_materials(self):
        """((m, 3) complex eps, (m,) complex 1 / mu, (m,) bool pec) per element at self.omega."""
        if callable(self.materials):
            return self.materials(self.omega)
        names = self.mesh.region_names
        table = [self.materials.get(name, self.background) for name in names] + [self.background]
        region = np.where(self.mesh.region >= 0, self.mesh.region, len(names))
        eps = np.array([material.eps_at(self.omega) for material in table])[region]
        inverse_mu = (1.0 / np.array([material.mu_at(self.omega) for material in table]))[region]
        pec = np.array([material.pec for material in table])[region]
        return eps, inverse_mu, pec

    def _pinned(self, pec_elements):
        """Sorted int64 pinned dofs: those of PEC elements, of no active element and on fixed outer faces."""
        dofs = self.dofs
        pinned = [dofs.element_dofs_of(pec_elements)]
        used = np.zeros(dofs.n_dofs, dtype=bool)
        used[dofs.element_dofs[~pec_elements].ravel()] = True
        pinned.append(np.flatnonzero(~used))
        for face in OUTER_FACES_3D:
            if getattr(self.boundary_config, face) in _FIXED_FACE_TYPES:
                pinned.append(dofs.face_dofs(self.mesh.face_tags[face]))
        return np.unique(np.concatenate(pinned))

    # ---- assembly ----
    def invalidate(self):
        """Materials changed: the next solve re-assembles (the shown resonance stays until then)."""
        self.system = None

    def _refresh_system(self):
        if self.system is None:
            self.system = self._assemble()
        return self.system

    def _assemble(self) -> FEMSystem3D:
        """FEMSystem3D at self.omega (module header)."""
        started = time.perf_counter()
        omega, mesh, dofs = self.omega, self.mesh, self.dofs
        eps, inverse_mu, pec = self._element_materials()
        in_pml = ~mesh.window_elements() if self.pml.active else np.zeros(mesh.n_elements, dtype=bool)
        degree = 2 * self.element_order
        curl, mass = assemble_nedelec_3d(dofs, inverse_mu, eps, elements=~pec & ~in_pml, degree=degree)
        if in_pml.any():
            def stretched_nu(points, elements):
                return inverse_mu[elements][:, None, None] / self.pml.tensor(points, omega)

            def stretched_eps(points, elements):
                return eps[elements][:, None, :] * self.pml.tensor(points, omega)

            curl_pml, mass_pml = assemble_nedelec_3d(dofs, stretched_nu, stretched_eps, elements=~pec & in_pml,
                                                     degree=degree + 2)
            curl, mass = pattern_sum((curl, curl_pml)), pattern_sum((mass, mass_pml))   # value-free pattern
        _, energy = assemble_nedelec_3d(dofs, None, np.abs(eps.real), elements=~pec, degree=degree)
        pinned = self._pinned(pec)
        prolongation = None
        if self.bloch is None:
            free = np.setdiff1d(np.arange(dofs.n_dofs), pinned)
            curl, mass, energy = (matrix[free][:, free].tocsr() for matrix in (curl, mass, energy))
        else:
            free, prolongation = self._bloch_prolongation(pinned)
            adjoint = prolongation.conj().T.tocsr()
            curl, mass, energy = ((adjoint @ matrix @ prolongation).tocsr() for matrix in (curl, mass, energy))
        target_squared = (omega / C0) ** 2
        operator = pattern_combination((mass, curl), (target_squared, -1.0))   # cuDSS plan reuse across sweeps
        equilibration = 1.0 / np.sqrt(np.abs(curl.diagonal()) + target_squared * np.abs(mass.diagonal()))
        in_window = mesh.in_window(dofs.dof_points()[free])
        self._counters['assemblies'] += 1
        self._counters['assembly_seconds'] += time.perf_counter() - started
        return FEMSystem3D(omega=omega, free=free, operator=operator, curl=curl, mass=mass, energy=energy,
                           in_window=in_window, equilibration=equilibration, prolongation=prolongation,
                           symmetric=self.bloch is None or not self.bloch.complex_phase)

    def _bloch_prolongation(self, pinned):
        """(free masters (n,) int64, CSR (n_dofs, n) u = P u_free with the Bloch phases): a master is pinned with any
        dof it stands for (a PEC element or wall touching either side)."""
        import scipy.sparse
        master, phase = self.bloch.master, self.bloch.phase
        free = np.setdiff1d(np.unique(master), master[pinned])
        column = np.full(self.dofs.n_dofs, -1, dtype=np.int64)
        column[free] = np.arange(free.size)
        rows = np.flatnonzero(column[master] >= 0)
        prolongation = scipy.sparse.csr_matrix((phase[rows], (rows, column[master[rows]])),
                                               shape=(self.dofs.n_dofs, free.size))
        return free, prolongation

    # ---- eigenfrequency study ----
    def solve_eigenfrequencies(self, n_resonances):
        """Up to n_resonances resonances nearest the wavelength (squared-Cayley shift-invert of A on the free dofs;
        host SuperLU under the 3D size rule at YEE_EQUIVALENT_UNKNOWNS, or a cuDSS factor) -> EigenSolve."""
        self.eigen_solve = None
        system = self._refresh_system()
        n_free = len(system.free)
        issue = (None if self.eigen_solver.factorization in CUDSS_FACTORIZATIONS
                 else vector_3d_size_issue(int(n_free * YEE_EQUIVALENT_UNKNOWNS[self.element_order])))
        if issue is not None and issue[0] == 'error':
            raise ValueError(f'eigenfrequency solve ({n_free:,} Nedelec degree-{self.element_order} unknowns): '
                             f'{issue[1]}')
        region = self.resonance_region
        region_options = {}
        if region is not None:   # the free rows (Bloch: masters) of every dof the region's elements touch
            touched = np.zeros(self.dofs.n_dofs, dtype=bool)
            touched[self.dofs.element_dofs[np.asarray(region.cells, dtype=bool)].ravel()] = True
            region_options = {'in_region': touched[system.free], 'min_region_fraction': region.min_fraction}
        self.eigen_solve = solve_resonances(
            system.operator, system.mass, system.omega, system.energy, system.in_window, n_resonances,
            spectral_map='squared_cayley', warnings=() if issue is None else (issue[1],), eigen_solver=self.eigen_solver,
            tolerance_floor=ARPACK_TOLERANCE_FLOOR if self.pml.active else 0.0,
            basis_per_wanted=ARNOLDI_BASIS_PER_WANTED, symmetric=system.symmetric, equilibration=system.equilibration,
            **region_options)
        return self.eigen_solve

    def select_resonance(self, number):
        """Resonance `number` (1-based) of the last solve becomes the sampled field (Bloch: hi-side dofs phased)."""
        resonance = self.eigen_solve.resonances[number - 1]
        if self.system.prolongation is not None:
            self.coefficients = np.asarray(self.system.prolongation @ resonance.field, dtype=np.complex128)
        else:
            coefficients = np.zeros(self.dofs.n_dofs, dtype=np.complex128)
            coefficients[self.system.free] = resonance.field
            self.coefficients = coefficients
        self.selected_resonance = number
        _, self._inverse_mu, _ = self._element_materials()   # of the solved materials: H of fields_at

    @property
    def resonance_omega(self) -> complex:
        """Complex omega [rad/s] of the selected resonance."""
        return complex(self.eigen_solve.resonances[self.selected_resonance - 1].omega)

    def fields_at(self, points) -> dict:
        """{'e': (p, 3), 'h': (p, 3)} complex fields of the selected resonance at points (p, 3) [m] (NaN outside the
        mesh): E from the coefficients, H = j curl E / (omega mu0 mu) at the resonance omega (unstretched: window)."""
        if self._locator is None:
            self._locator = PointLocator3D(self.mesh)
        return self.fields_located(*self._locator.locate(points))

    def fields_located(self, elements, reference) -> dict:
        """fields_at of points already located (PointLocator3D.locate: elements (p,), reference (p, 3))."""
        if self.coefficients is None:
            raise RuntimeError('FEMPhysics3D: fields before select_resonance')
        e_field, curl = evaluate_nedelec_3d(self.dofs, self.coefficients, elements, reference)
        element_inverse_mu = np.where(elements >= 0, self._inverse_mu[np.maximum(elements, 0)], np.nan)
        h_field = 1j * curl * element_inverse_mu[:, None] / (self.resonance_omega * MU0)
        return {'e': e_field, 'h': h_field}

    def plane_cut(self, axis, position, counts=(64, 64)) -> dict:
        """Fields of the selected resonance on a counts grid over the window cut normal to axis (0 / 1 / 2) at
        position [m]: {'u', 'v' (1D coordinates), 'e', 'h' ((n_u, n_v, 3)), 'e_abs' ((n_u, n_v))}."""
        window = self.mesh.window
        others = [k for k in range(3) if k != axis]
        u = np.linspace(window[others[0]], window[3 + others[0]], int(counts[0]))
        v = np.linspace(window[others[1]], window[3 + others[1]], int(counts[1]))
        grid_u, grid_v = np.meshgrid(u, v, indexing='ij')
        points = np.empty(grid_u.shape + (3,))
        points[..., axis] = float(position)
        points[..., others[0]], points[..., others[1]] = grid_u, grid_v
        fields = self.fields_at(points.reshape(-1, 3))
        shape = grid_u.shape + (3,)
        e_field, h_field = fields['e'].reshape(shape), fields['h'].reshape(shape)
        return {'u': u, 'v': v, 'e': e_field, 'h': h_field, 'e_abs': np.sqrt((np.abs(e_field) ** 2).sum(axis=-1))}

    def work_counters(self) -> dict:
        """Assemblies and their host seconds (+ the eigen solver's counters when it is private)."""
        counters = dict(self._counters)
        if self._owns_eigen_solver:
            counters.update(self.eigen_solver.work_counters())
        return counters

    @property
    def n_unknowns(self) -> int:
        """Free Nedelec dofs of the last assembly (assembles when needed)."""
        return int(len(self._refresh_system().free))

"""FDFDPhysics3D: the PhysicsSolver of em_fdfd_3d (igor3d_plan): the 3D Yee curl-curl operator (operators3d) on the live
E faces of a Domain3D, materials and PML frozen at the target omega, resonances by shift-invert on a host SuperLU or
cuDSS factor (eigenfrequency.solve_resonances, squared Cayley map) and the selected resonance's E / H on the run backend."""
# Materials (host, at omega_t = 2 pi C0 / wavelength): per E face the width-weighted mean (yee_grid.face_shares) of its
# two cells' complex eps of that component (eps_xx / eps_yy / eps_r - j sigma_e / (omega eps0) + every pole's chi on its
# region) under the PEC rule of the em_fdtd_3d kernels (kernels3d.pec_face_values); per H edge its lower cell's mu_r -
# j sigma_m / (omega mu0) (engine3d rule). Read straight from the MaterialGrid arrays (the float32 values; em_fdtd_3d
# steps the same values through its uint16 index unless it re-quantises beyond 65534 distinct ones). Subpixel smoothing
# (set_subpixel_smoothing): on each interface face of the build's InterfaceGeometry3D the background eps of its
# component is replaced by subpixel_smoothing3d.interface_eps of the two representative cells (sigma_e and poles keep
# their means; no override when a face or representative cell is PEC), the em_fdtd_3d face override. Operator on the
# live unknowns: A = -K + k_t^2 diag(eps_face) (K = curl_h diag(1 / mu) curl_e, operators3d.curl_curl_operator; the
# sign convention of eigenfrequency.py), assembled once per (omega, materials). The gradient null space of K (k = 0)
# is why the solve uses the squared Cayley map. PML share: energy weights |Re eps| x the E dual volume (half on an
# antisymmetric plane), window = E faces of window cells and the plane faces.
# Selected resonance: E = the prolongation of its eigenvector (mirror ghost images included), H = j h_keep curl_e E /
# (omega mu0 mu) at the resonance's complex omega; complex64 full-grid arrays on the run backend (renderer_fdfd3d).
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import C0, MU0
from ...core.solver_base import PhysicsSolver
from ...solvers.arpack_shift_invert import (CUDSS_FACTORIZATIONS, LDLT_FACTORIZATIONS, ArpackShiftInvertSolver,
                                            pattern_combination)
from ...solvers.sparse_direct import vector_3d_size_issue
from ..em.eigenfrequency import solve_resonances
from ..em.yee_grid import face_partner, face_shares
from .boundaries3d import AXES, solver_boundary_config_3d
from .kernels3d import pec_face_values
from .operators3d import (curl_curl_operator, face_dual_volumes, face_stretches, live_unknowns_3d,
                          nested_dissection_ordering, with_pec_walls, yee_operators_3d)
from .physics3d import MATERIAL_ARRAYS, StaticEM3DContract, check_pole, complex_cell_materials
from .renderer_fdfd3d import FDFD_3D_COMPONENTS, FDFDRenderer3D
from .subpixel_smoothing3d import FACE_KINDS, interface_eps

# ARPACK of the resonance solve: each application is two triangular solves on a large 3D factor, and with CPML the
# wanted eigenvalues sit in a cluster of PML modes (slow restarts), so a larger basis than scipy's 2 k + 1 (unless the
# eigen node sets subspace_size) and, with a cpml face only, a tolerance floor (Q finite anyway: relative Q error ~2
# floor Q); closed models keep machine precision (lossless Q = inf). Measured on one factor of a 17,172-unknown eighth
# Mie sphere (a removed one-off self-test), operator applications: machine precision + scipy basis 528, floor alone
# 638, 3 k alone 435, floor + 3 k 336, floor + 5 k 307; the basis memory stays far below the 3D factor's
ARPACK_TOLERANCE_FLOOR = 1e-10
ARNOLDI_BASIS_PER_WANTED = 5
_INTERFACE_ARRAYS = ('location', 'cell_a', 'cell_b', 'fill', 'normal_square')   # InterfaceSet3D fields read


@dataclass(frozen=True, eq=False)
class FDFDSystem3D:
    """Host system of one assembly at omega [rad/s]: ops (YeeOperators3D), live (LiveUnknowns3D), operator A (n, n) CSR
    complex128, mass = the live face eps, inverse_mu (3 N,) 1 / mu of every H edge, energy_weights / in_window (n,),
    elimination_order (n,) of the host factor (operators3d.nested_dissection_ordering; None on a cuDSS / MUMPS / PARDISO
    factorization), symmetrizer (n,) complex D with D A complex symmetric (stretched dual volumes, half on
    antisymmetric plane faces; None with a Bloch phase)."""
    omega: float
    ops: object
    live: object
    operator: object
    mass: np.ndarray
    inverse_mu: np.ndarray
    energy_weights: np.ndarray
    in_window: np.ndarray
    elimination_order: np.ndarray | None
    symmetrizer: np.ndarray | None


def face_permittivities(cell_eps, pec, cell_widths, periodic):
    """(3 N,) complex128 stacked E-face eps: per component c the face_shares mean of cell_eps[c] (nx, ny, nz) and its
    hi partner along c under the PEC rule (kernels3d.pec_face_values); faces between two PEC cells keep their mean."""
    faces = []
    for axis in range(3):
        lower_share, upper_share = face_shares(cell_widths[axis], periodic[axis], np.float64)
        shape = [1, 1, 1]
        shape[axis] = -1
        lower, upper = pec_face_values(cell_eps[axis], face_partner(cell_eps[axis], axis, periodic[axis]), pec,
                                       face_partner(pec, axis, periodic[axis]))
        faces.append((lower_share.reshape(shape) * lower + upper_share.reshape(shape) * upper).ravel())
    return np.concatenate(faces)


def smoothing_deltas(interface_faces, background_eps, pec, cell_widths, periodic):
    """(3 N,) float64 averaged minus face-mean background eps on the subpixel interface faces (0 elsewhere and where a
    face or representative cell is PEC); interface_faces = per component the host _INTERFACE_ARRAYS of its
    InterfaceSet3D, background_eps = per component the real (nx, ny, nz) eps it averages."""
    shape = pec.shape
    n_cells = math.prod(shape)
    flat_pec = pec.ravel()
    deltas = np.zeros(3 * n_cells)
    for axis, (location, cell_a, cell_b, fill, normal_square) in enumerate(interface_faces):
        if location.size == 0:
            continue
        eps = background_eps[axis].ravel()
        stride = math.prod(shape[axis + 1:])
        index = (location // stride) % shape[axis]
        last = index == shape[axis] - 1
        partner = location + stride   # face_partner: the last face wraps (periodic) or takes its own cell
        partner[last] = location[last] - index[last] * stride if periodic[axis] else location[last]
        lower_share, upper_share = face_shares(cell_widths[axis], periodic[axis], np.float64)
        mean = lower_share[index] * eps[location] + upper_share[index] * eps[partner]
        smoothed = interface_eps(eps[cell_a], eps[cell_b], fill.astype(np.float64), normal_square.astype(np.float64))
        free = ~(flat_pec[location] | flat_pec[partner] | flat_pec[cell_a] | flat_pec[cell_b])
        deltas[axis * n_cells + location[free]] = (smoothed - mean)[free]
    return deltas


class FDFDPhysics3D(StaticEM3DContract, PhysicsSolver):
    """3D frequency-domain EM on a Domain3D (BoundaryConfig3D faces cpml / pec / pmc / periodic with Bloch k / mirrors):
    the curl-curl operator at the wavelength setting, solve_eigenfrequencies / select_resonance (eigenfrequency study);
    PEC cells and dispersive poles on cropped masks, permittivity slots for the thermo-optic coupling."""

    name = 'em'
    type_label = 'em_fdfd_3d'

    def __init__(self, domain, wavelength=1.55e-6, eigen_solver=None):
        if getattr(domain, 'dimensions', 2) != 3:
            raise ValueError('FDFDPhysics3D needs a Domain3D (a model with domain size_z)')
        if domain.coordinates.polar:
            raise NotImplementedError('FDFDPhysics3D runs on Cartesian grids only (no cylindrical sectors)')
        self.domain = domain
        self.xp = domain.xp
        self.boundary_config = solver_boundary_config_3d(domain)
        wavelength = float(wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f'FDFDPhysics3D wavelength must be finite and > 0 [m], got {wavelength!r}')
        self._omega = 2.0 * math.pi * C0 / wavelength
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        grid = domain.grid
        self.shape = tuple(int(count) for count in grid.shape)
        # the kernel (float32 grid) widths, as em_fdtd_3d; the CPML grading cell = the larger outermost cell per axis
        self.widths = tuple(np.asarray(widths, dtype=np.float64) for widths in (grid.dx_arr, grid.dy_arr, grid.dz_arr))
        self.periodic = tuple(self.boundary_config.periodic(axis) for axis in AXES)
        self._pml_cell_sizes = tuple(float(max(widths[0], widths[-1])) for widths in self.widths)
        self.dispersive_regions = []   # [(pole, window slices, cropped run-backend bool mask)]
        self.pec_regions = []          # [(window slices, cropped run-backend bool mask)]
        self.interface_faces = None    # host _INTERFACE_ARRAYS per component of the subpixel geometry (None = off)
        self.finalize_warnings = []
        self.system = None             # FDFDSystem3D of the last assembly
        self.eigen_solve = None        # eigenfrequency.EigenSolve of the last solve_eigenfrequencies
        self.selected_resonance = None   # 1-based resonance shown in the fields
        self.resonance_region = None   # eigenfrequency.ResonanceRegion of a resonance_table selection, None = none
        self.host_factor_bytes = None  # L + U of the last shift-invert factor (Simulation.memory_report)
        self._system_dirty = True
        self._counters = {'assemblies': 0, 'assembly_seconds': 0.0}
        # complex64 full-grid E / H of the selected resonance (zeros before the first selection)
        self.fields = {name: self.xp.zeros(self.shape, dtype=np.complex64) for name in FDFD_3D_COMPONENTS}

    @property
    def omega(self) -> float:
        """Target angular frequency [rad/s] (the wavelength setting)."""
        return self._omega

    # ---- PhysicsSolver contract (+ StaticEM3DContract) ----
    def finalize(self):
        self.finalize_warnings = []

    def materials_changed(self, names):
        if not set(MATERIAL_ARRAYS).isdisjoint(names):
            self._system_dirty = True

    def renderer(self):
        return FDFDRenderer3D(self)

    def work_counters(self) -> dict:
        """Assemblies and their host seconds (+ the eigen solver's counters when it is private)."""
        counters = dict(self._counters)
        if self._owns_eigen_solver:
            counters.update(self.eigen_solver.work_counters())
        return counters

    # ---- model items ----
    def add_pec(self, window, mask):
        """PEC cells on a cropped run-backend bool mask of the cell box `window` (faces between two PEC cells drop)."""
        self.pec_regions.append((window, mask))
        self._system_dirty = True

    def set_subpixel_smoothing(self, geometry):
        """Model-wide subpixel smoothing (igor 1.5): host copies of the InterfaceGeometry3D faces whose background eps
        every assembly averages (module header); None = off."""
        self.interface_faces = None if geometry is None else tuple(
            tuple(np.asarray(to_host(getattr(geometry.kind(kind), name))) for name in _INTERFACE_ARRAYS)
            for kind in FACE_KINDS)
        self._system_dirty = True

    def set_resonance_region(self, region):
        """resonance_table selection: an eigenfrequency.ResonanceRegion of host bool full-grid cells; the solve
        reports the share of each resonance on their E faces and drops those below region.min_fraction."""
        if region.cells.shape != self.shape:
            raise ValueError(f'resonance region cells {region.cells.shape} do not match the grid {self.shape}')
        self.resonance_region = region

    def add_dispersive(self, pole, window, mask):
        """Drude / Lorentz pole on the cropped bool mask of window (the permittivity arrays are its background)."""
        check_pole(pole)
        self.dispersive_regions.append((pole, window, mask))
        self._system_dirty = True

    # ---- eigenfrequency study ----
    def solve_eigenfrequencies(self, n_resonances):
        """eigenfrequency study hook: up to n_resonances resonances nearest the wavelength setting (squared-Cayley
        shift-invert on the system assembled there; host SuperLU (or MUMPS / PARDISO, the same rule) under
        sparse_direct.vector_3d_size_issue, or cuDSS under its own device check) -> EigenSolve."""
        self.eigen_solve = None
        system = self._refresh_system()
        issue = (None if self.eigen_solver.factorization in CUDSS_FACTORIZATIONS
                 else vector_3d_size_issue(system.live.n_unknowns))
        if issue is not None and issue[0] == 'error':
            raise ValueError(f'eigenfrequency solve: {issue[1]}')
        region = self.resonance_region
        region_options = {} if region is None else {   # each cell owns its +x / +y / +z E faces (as in_window)
            'in_region': np.tile(region.cells.ravel(), 3)[system.live.full_index],
            'min_region_fraction': region.min_fraction}
        self.eigen_solve = solve_resonances(system.operator, system.mass, system.omega, system.energy_weights,
                                            system.in_window, n_resonances, spectral_map='squared_cayley',
                                            warnings=() if issue is None else (issue[1],),
                                            eigen_solver=self.eigen_solver, ordering=system.elimination_order,
                                            tolerance_floor=(ARPACK_TOLERANCE_FLOOR if self.boundary_config.needs_cpml
                                                             else 0.0),
                                            basis_per_wanted=ARNOLDI_BASIS_PER_WANTED, symmetrizer=system.symmetrizer,
                                            **region_options)
        self.host_factor_bytes = self.eigen_solve.info['factor_host_bytes']
        return self.eigen_solve

    def select_resonance(self, number):
        """Show resonance `number` (1-based) of the last solve: E from its unknowns, H from the curl at its complex
        omega, written into the complex64 fields."""
        resonance = self.eigen_solve.resonances[number - 1]
        system = self.system
        e_stacked = system.live.prolongation @ resonance.field
        edge_weights = np.where(system.live.h_keep, system.inverse_mu, 0.0)
        h_stacked = 1j * edge_weights * (system.ops.curl_e @ e_stacked) / (complex(resonance.omega) * MU0)
        n_cells = math.prod(self.shape)
        for index, name in enumerate(FDFD_3D_COMPONENTS):
            stacked = e_stacked if name[0] == 'e' else h_stacked
            component = stacked[(index % 3) * n_cells:(index % 3 + 1) * n_cells].reshape(self.shape)
            self.xp.copyto(self.fields[name], self.xp.asarray(component.astype(np.complex64)))
        self.selected_resonance = number

    # ---- assembly ----
    def _refresh_system(self):
        if self._system_dirty or self.system is None:
            self.system = self._assemble()
            self._system_dirty = False
        return self.system

    def _pec_cells(self):
        """Host bool (nx, ny, nz) PEC cells: the pec regions and the pec wall layers."""
        pec = np.zeros(self.shape, dtype=bool)
        for window, mask in self.pec_regions:
            pec[window] |= np.asarray(to_host(mask), dtype=bool)
        return with_pec_walls(self.shape, self.boundary_config, self.domain.physical_window(), pec)

    def _assemble(self):
        """FDFDSystem3D at self.omega (module header)."""
        import scipy.sparse
        started = time.perf_counter()
        omega = self._omega
        config = self.boundary_config
        window = self.domain.physical_window()
        pec = self._pec_cells()
        ops = yee_operators_3d(self.widths, config, self._pml_cell_sizes, omega)
        live = live_unknowns_3d(self.shape, config, window, pec)
        eps_faces, inverse_mu = self._materials_at(omega, pec)
        mass = eps_faces[live.full_index]
        target_squared = (omega / C0) ** 2
        operator = pattern_combination(   # value-free pattern: cuDSS plan reuse across sweep values
            (scipy.sparse.diags(target_squared * mass), curl_curl_operator(ops, live, inverse_mu)), (1.0, -1.0))
        dual_volumes = face_dual_volumes(self.widths, self.periodic)[live.full_index]
        dual_volumes[live.plane_faces] *= 0.5   # the image weighting: also the plane rows' share of the symmetrizer
        symmetrizer = None if config.has_bloch_phase else (
            dual_volumes * face_stretches(self.shape, config, self._pml_cell_sizes, omega)[live.full_index])
        in_window_cells = np.zeros(self.shape, dtype=bool)
        in_window_cells[window] = True
        in_window = np.tile(in_window_cells.ravel(), 3)[live.full_index] | live.plane_faces
        # cuDSS / MUMPS / PARDISO reorder themselves: the host nested dissection only for a SuperLU factor
        elimination_order = (None if self.eigen_solver.factorization in LDLT_FACTORIZATIONS
                             else nested_dissection_ordering(live))
        self._counters['assemblies'] += 1
        self._counters['assembly_seconds'] += time.perf_counter() - started
        return FDFDSystem3D(omega=omega, ops=ops, live=live, operator=operator, mass=mass, inverse_mu=inverse_mu,
                            energy_weights=np.abs(mass.real) * dual_volumes, in_window=in_window,
                            elimination_order=elimination_order, symmetrizer=symmetrizer)

    def _materials_at(self, omega, pec):
        """((3 N,) complex128 face eps, (3 N,) 1 / mu of every H edge) at omega (module header)."""
        cell_eps, background_eps, mu = complex_cell_materials(self.domain.materials, omega, self.dispersive_regions)
        eps_faces = face_permittivities(cell_eps, pec, self.widths, self.periodic)
        del cell_eps
        if self.interface_faces is not None:
            eps_faces += smoothing_deltas(self.interface_faces, background_eps, pec, self.widths, self.periodic)
        return eps_faces, np.tile((1.0 / mu).ravel(), 3)

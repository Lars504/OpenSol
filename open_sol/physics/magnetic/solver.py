"""2D magnetostatics on the DiffusionSystem: planar A_z, or axisymmetric flux function psi = r A_phi on the PLANAR
operator with c = nu / r_c; stranded coils, held regions, a uniform external field (held outer ring) and remanence as
face sources. Per unit z length (planar) or whole rings (axisymmetric)."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES, MU0
from ...core.damped_iteration import DampedIteration, DampingSettings
from ...core.fv_diffusion import (BCSide, DiffusionPhysicsBase, DiffusionSystem, ScalarBoundaryConfig,
                                  edge_extension_views)
from .fields import MagneticFields2D
from .kept_preconditioner import KeptPreconditionerSolver, kept_preconditioner_solver
from .materials import MAGNETIC_SCHEMA, PERMEABILITY_ARRAYS, REMANENCE_ARRAYS, SATURATION_ARRAY
from .renderer import MagneticRenderer
from .saturation import (ANDERSON_DEPTH, TENSOR_NONLINEAR_METHODS, bh_curve, refuse_curve_conflicts,
                         saturation_reluctivity, tabulated_reluctivity)

_ANCHOR_TEXT = 'magnetic insulation face, held potential or external-field ring'
_REMEDY_TEXT = 'Keep magnetic insulation (the default) on at least one face, or add an external field.'
_DERIVED_FIELDS = ('B_x', 'B_y', 'B_abs', 'H_x', 'H_y', 'w_mag')


@dataclass(frozen=True, eq=False)
class CoilRegion:
    """Runtime object of a stranded coil: window cells, turns, current [A] (> 0 along +z / +phi), cross-section area
    [m^2] (the r-z area when axisymmetric)."""
    name: str
    mask_window: object     # bool window mask of the run backend
    turns: int
    current: float
    area: float


def face_drive_terms(system, drive_x, drive_y):
    """Face terms [c u] (toward +x (nx+1, ny), toward +y (nx, ny+1)) of a cell-centred drive [u / m] whose physical
    flux is -c (grad u - drive): flux * face length = conductance * (u_lower - u_upper) + term. Held half cells carry
    none, outer faces only through their face-term conductance (Neumann: none)."""
    xp = system.xp
    free = ~system.fixed_user
    dtype = system.dtype
    half_x = xp.where(free, 0.5 * drive_x * system.dx_window_device[:, None], 0.0).astype(dtype, copy=False)
    half_y = xp.where(free, 0.5 * drive_y * system.dy_window_device[None, :], 0.0).astype(dtype, copy=False)
    conductance_x, conductance_y = system.face_conductances
    nx, ny = system.window_shape
    face_drive_x = xp.zeros((nx + 1, ny), dtype=dtype)   # toward +x, per face [c u]
    face_drive_x[1:-1, :] = conductance_x * (half_x[:-1, :] + half_x[1:, :])
    face_drive_y = xp.zeros((nx, ny + 1), dtype=dtype)
    face_drive_y[:, 1:-1] = conductance_y * (half_y[:, :-1] + half_y[:, 1:])
    for face, (cells, conductance, _) in system.face_terms.items():
        if face == 'x_lo':
            face_drive_x[0, :] = conductance * half_x[cells]
        elif face == 'x_hi':
            face_drive_x[-1, :] = conductance * half_x[cells]
        elif face == 'y_lo':
            face_drive_y[:, 0] = conductance * half_y[cells]
        else:
            face_drive_y[:, -1] = conductance * half_y[cells]
    return face_drive_x, face_drive_y


def face_divergence_source(face_drive_x, face_drive_y):
    """Window rhs [c u] of face_drive_terms: minus their net outflow per cell."""
    return -(face_drive_x[1:, :] - face_drive_x[:-1, :]) - (face_drive_y[:, 1:] - face_drive_y[:, :-1])


class MagneticPhysics(DiffusionPhysicsBase):
    """2D magnetostatics -curl(nu curl A) = J on the physical window (c_xx = nu_yy / r, c_yy = nu_xx / r; r = 1
    planar): one DiffusionSystem solve per change; A, B, H and w_mag edge-extended into the PML ring."""

    name = 'magnetic'
    type_label = 'magnetostatics'           # message prefix (MagneticACPhysics: magnetic_fields_ac)
    _base_material_schema = MAGNETIC_SCHEMA
    _probe_field = 'A'
    _probe_label = 'vector potential probe'
    coordinate_grids = ('cartesian',)       # B, H, coils and face sources read Cartesian cell widths

    def __init__(self, domain, axisymmetric: bool = False, tol: float = 1e-8, max_iter: int = 5000,
                 compute_residual: bool = False, precision: str = 'float64', nonlinear_tolerance: float = 1e-6,
                 nonlinear_max_iterations: int = 100, nonlinear_relaxation: float = 0.5,
                 nonlinear_method: str = 'auto', linear_solver=None):
        if domain.dimensions != 2:
            raise ValueError(f'{self.type_label} needs a 2D domain (a 3D vector potential needs edge elements)')
        self._axisymmetric = bool(axisymmetric)
        self.material_schema = dict(self._base_material_schema)
        # planar operator in both geometries: the ring-weighted one would cancel the 1 / r of c; linear_solver (the
        # build's pcg) replaces precision / tol / max_iter; a multigrid / direct one keeps its build across the
        # saturation loop (kept_preconditioner)
        system = DiffusionSystem(
            domain, precision, tol, max_iter, label=lambda: f"{self.type_label} '{self.name}'",
            unknown_name='vector potential', anchor_text=_ANCHOR_TEXT, remedy_text=_REMEDY_TEXT,
            compute_residual=compute_residual, axisymmetric=False,
            linear_solver=kept_preconditioner_solver(linear_solver))
        insulation = ScalarBoundaryConfig(**{face: BCSide('dirichlet', value=0.0) for face in FACES})
        if self._axisymmetric:
            # Axis rule: the x_lo half cell's series resistance int r / nu dr puts c at r_c / 2, doubling the planar
            # Dirichlet half-cell conductance (psi = 0 on the axis)
            insulation.x_lo = BCSide('dirichlet', value=0.0, conductance_factor=2.0)
        self._init_diffusion(domain, insulation, system)
        self.coils = []
        self._source_dirty = True
        self._remanence_stale = True
        self._remanence_rhs = None
        self._remanence_drives = None   # face_drive_terms of the remanence (None: no remanence arrays)
        # B-H saturation (stage 4): fixed point on ln of the secant reluctivities (nu_xx, nu_yy) of the window, float64
        # (2, wx, wy), Anderson-mixed or damped Picard
        nonlinear_method = 'anderson' if nonlinear_method == 'auto' else nonlinear_method
        if nonlinear_method not in TENSOR_NONLINEAR_METHODS:
            raise ValueError(f"{self.type_label}: nonlinear_method on a tensor mesh must be 'auto' or one of "
                             f"{TENSOR_NONLINEAR_METHODS}, got {nonlinear_method!r} (newton runs on triangle meshes)")
        self.runtime_warnings = []      # append-only (unconverged saturation loops); the studies log new ones
        self.picard = DampedIteration(
            DampingSettings(nonlinear_tolerance, nonlinear_max_iterations, nonlinear_relaxation,
                            anderson_depth=ANDERSON_DEPTH if nonlinear_method == 'anderson' else 0),
            label=lambda: f"{self.type_label} '{self.name}' saturation")
        # ln nu loop state (2, k) on the saturating cells only (warm start of the next loop); None = linear nu
        self._log_reluctivity_state = None
        self._saturation = None         # float64 J_s [T] of the saturating cells (k,), None = no J_s > 0
        self._bh_curves = []            # (window bool mask, cupy saturation.BHCurve) per add_bh_curve, later wins
        self._nonlinear = False         # J_s > 0 or a B-H table somewhere: the saturation loop runs
        self._nonlinear_cells = None    # window bool mask of the last refresh's saturating cells
        self._nonlinear_flat = None     # their flat window indices (k,): the compact loop state's cells
        self._saturation_stale = True
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    @property
    def axisymmetric(self) -> bool:
        """True in (r, z) geometry (x = r from the window's x_lo edge); the system itself stays planar."""
        return self._axisymmetric

    def prepare(self, domain) -> None:
        """Register the permeability arrays, prepare the window system and allocate the fields; assembly is deferred."""
        self.domain = domain
        self.xp = domain.xp   # numpy (cpu backend) or cupy
        domain.materials.add_properties(self.material_schema)
        system = self.system
        system.prepare(domain)
        if isinstance(system.linear_solver, KeptPreconditionerSolver):
            system.linear_solver.bind_window(system.window_shape)

        nx, ny = domain.grid.nx, domain.grid.ny
        if self.fields is None or (self.fields.nx, self.fields.ny) != (nx, ny):
            self.fields = MagneticFields2D(nx, ny, xp=self.xp)
        window = system.window
        if min(system.window_shape) < 3:
            raise ValueError(f"magnetostatics '{self.name}' needs at least 3 x 3 window cells, got "
                             f"{system.window_shape}")
        self._A_window = self.fields.A[window]
        self._source_window = self.fields.J_coupled[window]
        self._applied_source_window = self.fields.J_applied[window]
        x_range, y_range = self._bind_window(self.fields.A)
        self._derived_pml_copies = [
            pair for field_name in _DERIVED_FIELDS
            for pair in edge_extension_views(getattr(self.fields, field_name), x_range, y_range)]
        self._bind_geometry()
        self._source_dirty = True
        self._remanence_stale = True
        self._log_reluctivity_state = None
        self._nonlinear_cells = None
        self._saturation_stale = True

    def _bind_geometry(self):
        """Window cell edges / centres from the window's x_lo / y_lo edge (host float64; x = r when axisymmetric),
        r_c as a solve-dtype (wx, 1) device column (1 planar) and the zero cross term."""
        system = self.system
        self._x_faces = np.concatenate(([0.0], np.cumsum(system.dx_window)))
        self._y_faces = np.concatenate(([0.0], np.cumsum(system.dy_window)))
        self._x_centres = 0.5 * (self._x_faces[:-1] + self._x_faces[1:])
        self._y_centres = 0.5 * (self._y_faces[:-1] + self._y_faces[1:])
        radius = self._x_centres if self._axisymmetric else np.ones_like(self._x_centres)
        self._radius_window = self.xp.asarray(radius, dtype=system.dtype)[:, None]
        self._zero_cross_term = self.xp.zeros((), dtype=system.dtype)

    def finalize(self) -> None:
        """Assemble once (fails early on non-positive permeability or a floating region)."""
        self._assemble()

    def materials_changed(self, names) -> None:
        names = set(names)
        if PERMEABILITY_ARRAYS & names:
            self.system.operator_dirty = True
        if names.intersection(REMANENCE_ARRAYS):
            self._remanence_stale = True
            self._source_dirty = True
        if SATURATION_ARRAY in names:
            self._saturation_stale = True
            self.system.operator_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the operator or a source changed (always at the first call; a saturation loop when a
        material sets saturation_polarization or a B-H curve); probes sample after a re-solve and at every dt = inf
        call, never on unchanged time steps."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._source_dirty or system.n_solves == 0
        if re_solve:
            if self._saturation_stale:
                self._refresh_saturation()
            if self._nonlinear:
                self._solve_saturated()
            else:
                if system.operator_dirty:
                    self._assemble()
                self._solve_once()
            self._source_dirty = False
        if re_solve or dt is None or not math.isfinite(dt):
            for probe in self.probes:   # a re-solve at the end time plus the flush keep one sample (as potential)
                probe.record(self.fields, t, replace_same_time=t > 0.0)

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        fields = self.fields
        materials = self.domain.materials
        state = {name: getattr(fields, name) for name in ('A', *_DERIVED_FIELDS, 'J_coupled', 'J_applied')}
        for array_name in self.material_schema:
            state[array_name] = getattr(materials, array_name)
        return state

    def inject_source(self, name: str, array) -> None:
        if name != 'current_density':
            raise ValueError(f"magnetostatics '{self.name}' accepts source 'current_density'; got '{name}'")
        self.xp.copyto(self.fields.J_coupled, self.xp.asarray(array, dtype=DTYPE))
        self._source_dirty = True

    def renderer(self):
        return MagneticRenderer(self)

    def scalars(self) -> dict:
        """energy [J/m planar, J axisymmetric], per coil <name>_flux_linkage / <name>_inductance; {} before a solve."""
        if self.system.n_solves == 0:
            return {}
        values = {'energy': self.field_energy()}
        for coil in self.coils:
            result = self.coil_results(coil)
            values[f'{coil.name}_flux_linkage'] = result['flux_linkage']
            values[f'{coil.name}_inductance'] = result['inductance']
        return values

    def work_counters(self) -> dict:
        """The saturation Picard loop's counters (nonlinear_runs / _iterations / _unconverged) and its kept-preconditioner
        solves once it ran, else {}."""
        if not self.picard.n_runs:
            return {}
        counters = self.picard.work_counters()
        if isinstance(self.system.linear_solver, KeptPreconditionerSolver):
            counters.update(self.system.linear_solver.work_counters())
        return counters

    # --- magnetics API (shared settings, add_boundary_flux, probes: DiffusionPhysicsBase) ---------------------

    def set_face_bc(self, face: str, bc: BCSide) -> None:
        """As DiffusionPhysicsBase; axisymmetric x_lo is the axis (psi = 0) and cannot be changed."""
        if self._axisymmetric and face == 'x_lo':
            raise ValueError(f"magnetostatics '{self.name}': x_lo is the symmetry axis in axisymmetric geometry "
                             '(psi = r A_phi = 0 there); its condition is fixed')
        super().set_face_bc(face, bc)

    def add_fixed_potential(self, mask, value) -> None:
        """Hold the unknown (planar A_z [T m], axisymmetric psi = r A_phi [T m^2]; scalar or full-grid array) on the
        window cells under mask."""
        system = self.system
        mask_window = system.window_mask(mask, 'fixed potential')
        system.set_fixed(mask_window, system.window_values(value, 'fixed potential', system.dtype))

    def add_source(self, mask, density) -> None:
        """Add a current density J_z / J_phi [A/m^2] (scalar or full-grid array) to J_applied on the masked cells."""
        mask_window = self.system.window_mask(mask, 'current source')
        density_window = self.system.window_values(density, 'current source density', DTYPE)
        self._applied_source_window += self.xp.where(mask_window, density_window, DTYPE(0.0))
        self._source_dirty = True

    def add_coil(self, name: str, mask, turns: int, current: float) -> CoilRegion:
        """Stranded coil: J = turns * current / area on the masked window cells (area = their r-z / x-y area)."""
        if int(turns) < 1:
            raise ValueError(f"coil '{name}': turns must be >= 1, got {turns}")
        system = self.system
        mask_window = system.window_mask(mask, f"coil '{name}'").copy()
        xp = self.xp
        area = float(xp.sum(xp.where(mask_window, system.cell_area, 0.0)))
        if not area > 0.0:
            raise ValueError(f"coil '{name}' selects no window cell")
        coil = CoilRegion(name, mask_window, int(turns), float(current), area)
        self._applied_source_window += xp.where(mask_window, DTYPE(coil.turns * coil.current / area), DTYPE(0.0))
        self._source_dirty = True
        self.coils.append(coil)
        return coil

    def add_bh_curve(self, mask, rows) -> bool:
        """Tabulated anhysteretic B-H curve [(H [A/m], B [T]), ...] (saturation.bh_curve rules) on the window cells
        under the full-grid bool mask: nu_xx = nu_yy = H(|B|) / |B| there, mu_dc ignored; a later curve wins shared
        cells. False (nothing added) when the mask covers no window cell."""
        curve = bh_curve(rows, self.xp)
        mask_window = self.system.window_mask(mask, 'B-H curve').copy()
        if not bool(self.xp.any(mask_window)):
            return False
        self._bh_curves.append((mask_window, curve))
        self._log_reluctivity_state = None   # the next loop starts from every curve's B = 0 limit
        self._saturation_stale = True
        self.system.operator_dirty = True
        return True

    def set_external_field(self, B_x: float, B_y: float) -> None:
        """Uniform external flux density [T] by holding the outermost window cells at the external potential evaluated
        at their faces toward the free cells (axisymmetric: B_y = B_z only, x_lo is the axis)."""
        values, mask = self._external_ring(float(B_x), float(B_y))
        system = self.system
        system.set_fixed(self.xp.asarray(mask), self.xp.asarray(values, dtype=system.dtype))

    def coil_results(self, coil: CoilRegion) -> dict:
        """current, turns, flux_linkage [Wb/m planar, Wb axisymmetric] = turns / area * integral of A (2 pi psi) over
        the coil, inductance = flux_linkage / current [H/m, H] (nan at zero current)."""
        system = self.system
        if system.n_solves == 0:
            raise RuntimeError(f"magnetostatics '{self.name}': coil '{coil.name}' needs a solve first")
        weight = 2.0 * math.pi if self._axisymmetric else 1.0
        integral = float(self.xp.sum(self.xp.where(coil.mask_window, system.unknown2d * system.cell_area, 0.0)))
        linkage = coil.turns * weight * integral / coil.area
        inductance = linkage / coil.current if coil.current != 0.0 else float('nan')
        return {'current': coil.current, 'turns': coil.turns, 'flux_linkage': linkage, 'inductance': inductance}

    def field_energy(self) -> float:
        """Energy (1/2) u^T K u of the free cells [J/m planar, J axisymmetric (x 2 pi)]: every face touching a free cell
        (held-free faces: the free half cell) and outer faces of free cells (the axis face with its factor 2); with
        remanence the energy of B alone (not the magnet co-energy)."""
        system = self.system
        xp = self.xp
        unknown = system.unknown2d
        free = ~system.fixed_user
        conductance_x, conductance_y = system.face_conductances
        total = (xp.sum(xp.where(free[:-1, :] | free[1:, :],
                                 conductance_x * (unknown[:-1, :] - unknown[1:, :]) ** 2, 0.0))
                 + xp.sum(xp.where(free[:, :-1] | free[:, 1:],
                                   conductance_y * (unknown[:, :-1] - unknown[:, 1:]) ** 2, 0.0)))
        for cells, conductance, rhs in system.face_terms.values():
            conducting = (conductance > 0) & free[cells]
            total += xp.sum(xp.where(conducting, (conductance * unknown[cells] - rhs) ** 2
                                     / xp.where(conducting, conductance, 1.0), 0.0))
        return 0.5 * (2.0 * math.pi if self._axisymmetric else 1.0) * float(total)

    def rebuild_operator(self) -> None:
        """Re-assemble the operator, held-cell elimination, anchor check and Jacobi preconditioner."""
        self._assemble()

    def warm_start_fields(self) -> dict:
        """Host copies of the window unknown (A_z or psi) and coupled source, to seed a build on the same mesh."""
        return {'unknown': np.array(to_host(self.system.unknown2d)), 'source': np.array(to_host(self._source_window))}

    def seed_warm_start(self, fields) -> tuple:
        """Write warm_start_fields() of another build (unknown on free cells, A, source); returns the slots seeded
        (B, H and w_mag wait for the next solve)."""
        self.system.seed_unknown(fields['unknown'])
        self._write_potential_field()
        self.xp.copyto(self._source_window, self.xp.asarray(fields['source'], dtype=DTYPE))
        self._source_dirty = True
        return ('A', 'J_coupled')

    # --- internals ------------------------------------------------------------

    def _coefficients(self, read_permeability=True):
        """(c_xx, c_yy) = (nu_yy / r, nu_xx / r) window arrays in the solve dtype; keeps nu_xx / nu_yy (fields) and
        mu_dc_xx / _yy (saturation curve; read and checked unless read_permeability is False: the kept copies);
        nu = exp of the saturation loop state when set, else 1 / (mu0 mu_dc)."""
        dtype = self.system.dtype
        if read_permeability:
            materials = self.domain.materials
            window = self.system.window
            mu_xx = materials.mu_dc_xx[window].astype(dtype)
            mu_yy = materials.mu_dc_yy[window].astype(dtype)
            xp = self.xp
            if bool(xp.any(~(mu_xx > 0) | ~(mu_yy > 0) | ~xp.isfinite(mu_xx) | ~xp.isfinite(mu_yy))):
                raise ValueError(f"magnetostatics '{self.name}': mu_dc_xx / mu_dc_yy must be finite and > 0 in every "
                                 'window cell')
            self._permeability = (mu_xx, mu_yy)
        mu_xx, mu_yy = self._permeability
        self._nu_xx = 1.0 / (MU0 * mu_xx)
        self._nu_yy = 1.0 / (MU0 * mu_yy)
        if self._log_reluctivity_state is not None:   # the saturating cells: exp of the compact loop state
            flat = self._nonlinear_flat
            self._nu_xx.reshape(-1)[flat] = self.xp.exp(self._log_reluctivity_state[0]).astype(dtype)
            self._nu_yy.reshape(-1)[flat] = self.xp.exp(self._log_reluctivity_state[1]).astype(dtype)
        return self._nu_yy / self._radius_window, self._nu_xx / self._radius_window

    def _assemble(self, full=True):
        """Assemble with the current nu; full=False (later saturation iterations: only nu changed, every c > 0) reuses
        the checked mu_dc copies and skips the anchor check."""
        c_xx, c_yy = self._coefficients(read_permeability=full)
        self.system.assemble(c_xx, c_yy, self._zero_cross_term, self.bcs, 0.0, check_anchors=full)
        self._remanence_stale = True

    def _solve_once(self):
        """One linear solve on the assembled operator: rhs (sources, coils, remanence), CG, fields."""
        system = self.system
        system.build_rhs(self._source_window, self._applied_source_window)
        self._add_remanence_rhs()
        system.solve()
        self._update_fields()

    def _refresh_saturation(self):
        """Window J_s of the saturating cells (None without J_s > 0) and whether the loop runs (J_s or a B-H curve);
        the loop state is dropped when the saturating set changes or vanishes; a nonlinear cell with remanence, or a
        B-H curve cell with J_s > 0, raises (one sync per check)."""
        self._saturation_stale = False
        self._saturation = None
        materials = self.domain.materials
        names = materials.property_names()
        window = self.system.window
        xp = self.xp
        nonlinear_cells = xp.zeros(self.system.window_shape, dtype=bool)
        saturation = None
        if SATURATION_ARRAY in names:
            saturation = materials.saturation_polarization[window].astype(np.float64)
            nonlinear_cells |= saturation > 0.0
            if bool(xp.any(nonlinear_cells)):
                self._saturation = saturation   # window here, compact below
        if self._bh_curves:
            tabulated_cells = xp.zeros_like(nonlinear_cells)
            for mask_window, _ in self._bh_curves:
                tabulated_cells |= mask_window
            refuse_curve_conflicts(f"magnetostatics '{self.name}'", 'cell', curve_and_polarization=(
                self._saturation is not None and bool(xp.any(tabulated_cells & nonlinear_cells))))
            nonlinear_cells |= tabulated_cells
        self._nonlinear = self._saturation is not None or bool(self._bh_curves)
        if not self._nonlinear:
            self._log_reluctivity_state = None   # back to the linear 1 / (mu0 mu_dc)
            self._nonlinear_cells = None
            return
        if self._nonlinear_cells is None or not bool(xp.array_equal(nonlinear_cells, self._nonlinear_cells)):
            self._log_reluctivity_state = None   # new saturating set: restart from every curve's B = 0 limit
            self._nonlinear_flat = xp.flatnonzero(nonlinear_cells.ravel())
        self._nonlinear_cells = nonlinear_cells
        flat = self._nonlinear_flat
        if self._saturation is not None:
            self._saturation = self._saturation.reshape(-1)[flat]
        # the B-H curve cells of each table among the saturating cells (compact masks)
        self._compact_curves = [(mask_window.reshape(-1)[flat], curve) for mask_window, curve in self._bh_curves]
        remanent = xp.zeros_like(nonlinear_cells)
        for name in REMANENCE_ARRAYS:
            if name in names:
                remanent |= getattr(materials, name)[window] != 0
        refuse_curve_conflicts(f"magnetostatics '{self.name}'", 'cell',
                               saturating_remanence=bool(xp.any(nonlinear_cells & remanent)))

    def _solve_saturated(self):
        """Fixed point on ln nu (core.damped_iteration, Anderson-mixed or damped Picard): each iteration assembles
        with the current nu (the first one also checks mu_dc and anchors), solves warm-started at the previous A and
        proposes ln nu of the curves (J_s: nu(|B|), tables: |H| / B(|H|)); the change of ln nu is the relative change
        of nu (scale 1). The first state is every curve's B = 0 limit (the linear 1 / (mu0 mu_dc) for J_s curves)."""
        first = [True]   # the first proposal of each loop assembles fully
        if self._log_reluctivity_state is None:
            self._coefficients()   # reads and checks mu_dc
            zeros = self.xp.zeros(self._nonlinear_flat.size, dtype=np.float64)
            self._log_reluctivity_state = self._saturated_log_reluctivity(zeros, zeros)
        state = self._log_reluctivity_state

        def propose(log_reluctivity):   # the state itself, read by _coefficients through the attribute
            if isinstance(self.system.linear_solver, KeptPreconditionerSolver):   # later iterations may keep its build
                self.system.linear_solver.offer_state(log_reluctivity, fresh=first[0])
            self._assemble(full=first[0])
            first[0] = False
            self._solve_once()
            flat = self._nonlinear_flat
            flux_magnitude = self._flux_density_magnitude.reshape(-1)[flat].astype(np.float64)
            # |H| = nu |B| in the (isotropic) table cells: B = H / nu there (no remanence), read by the tables only
            field_magnitude = (self._nu_xx.reshape(-1)[flat].astype(np.float64) * flux_magnitude if self._bh_curves
                               else None)
            return self._saturated_log_reluctivity(flux_magnitude, field_magnitude)

        # the operator and fields stay those of the last iterate's solve; the state warm-starts the next loop
        if not self.picard.run(state, propose, 1.0):
            self.runtime_warnings.append(self.picard.last_warning)

    def _saturated_log_reluctivity(self, flux_magnitude, field_magnitude):
        """ln (nu_xx, nu_yy) float64 (2, k) of the curves on the saturating cells at their float64 cell-centre |B| =
        flux_magnitude (k,): the J_s curve per axis (no J_s: linear 1 / (mu0 mu_dc)), then each B-H table on both axes
        of its cells at the centre |H| = field_magnitude (H-based: the B-based proposal diverges on the table's steep
        mu0 extension)."""
        xp = self.xp
        flat = self._nonlinear_flat
        mu_xx, mu_yy = (mu.reshape(-1)[flat].astype(np.float64) for mu in self._permeability)
        if self._saturation is not None:
            nu_xx = saturation_reluctivity(flux_magnitude, mu_xx, self._saturation, xp)
            nu_yy = saturation_reluctivity(flux_magnitude, mu_yy, self._saturation, xp)
        else:
            nu_xx, nu_yy = 1.0 / (MU0 * mu_xx), 1.0 / (MU0 * mu_yy)
        for mask, curve in self._compact_curves:
            tabulated = tabulated_reluctivity(field_magnitude, curve, xp)
            nu_xx = xp.where(mask, tabulated, nu_xx)
            nu_yy = xp.where(mask, tabulated, nu_yy)
        return xp.log(xp.stack((nu_xx, nu_yy)))

    def _add_remanence_rhs(self):
        """Add the remanence face source to the rhs built by build_rhs (free rows) and to rhs_full; no-op without
        allocated remanence arrays."""
        if self._remanence_stale:
            self._remanence_drives = self._remanence_face_drives()
            self._remanence_rhs = (None if self._remanence_drives is None
                                   else face_divergence_source(*self._remanence_drives))
            self._remanence_stale = False
        if self._remanence_rhs is None:
            return
        system = self.system
        system.rhs2d += system.free * self._remanence_rhs
        system.rhs_full += self._remanence_rhs

    def _remanence_face_drives(self):
        """face_drive_terms of H = nu (B - B_r): drive (-B_ry, B_rx) planar, (r B_rz, 0) axisymmetric; None when no
        remanence array is allocated."""
        materials = self.domain.materials
        allocated = [name for name in REMANENCE_ARRAYS if name in materials.property_names()]
        if not allocated:
            return None
        system = self.system
        window = system.window
        dtype = system.dtype
        zeros = self.xp.zeros(system.window_shape, dtype=dtype)
        remanence_x = materials.remanence_x[window].astype(dtype) if 'remanence_x' in allocated else zeros
        remanence_y = materials.remanence_y[window].astype(dtype) if 'remanence_y' in allocated else zeros
        if self._axisymmetric:
            if bool(self.xp.any(remanence_x != 0)):
                raise NotImplementedError(f"magnetostatics '{self.name}': radial remanence (remanence_x) is not "
                                          'available in axisymmetric geometry; magnetise along z (remanence_y)')
            face_drive_x, face_drive_y = face_drive_terms(system, self._radius_window * remanence_y, zeros)
            # axis half cell: int_0^r_c r B dr = B r_c^2 / 2 is half of v_c dr / 2 (its conductance carries the 2)
            face_drive_x[0, :] *= 0.5
            return face_drive_x, face_drive_y
        return face_drive_terms(system, -remanence_y, remanence_x)

    def _external_ring(self, B_x, B_y):
        """(values, mask) host window arrays of the held outer ring: planar A = B_x y - B_y x at the ring cells' faces
        toward the free cells; axisymmetric psi = B_z r^2 / 2 with the exact half-cell value on x_hi."""
        x_faces, y_faces, x_centres, y_centres = self._x_faces, self._y_faces, self._x_centres, self._y_centres
        shape = self.system.window_shape
        values = np.zeros(shape)
        mask = np.zeros(shape, dtype=bool)
        if self._axisymmetric:
            if B_x != 0.0:
                raise ValueError(f"magnetostatics '{self.name}': an axisymmetric external field is axial only "
                                 f'(B_x = B_r must be 0, got {B_x})')
            inner_radius, ring_face = x_centres[-2], x_faces[-2]   # free neighbour centre, shared face
            values[-1, :] = 0.5 * B_y * inner_radius * (2.0 * ring_face - inner_radius)
            values[:, 0] = values[:, -1] = 0.5 * B_y * x_centres ** 2
            mask[-1, :] = True
        else:
            def potential(x, y):
                return B_x * y - B_y * x
            values[0, :] = potential(x_faces[1], y_centres)
            values[-1, :] = potential(x_faces[-2], y_centres)
            values[:, 0] = potential(x_centres, y_faces[1])
            values[:, -1] = potential(x_centres, y_faces[-2])
            mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        return values, mask

    def _write_potential_field(self):
        """Field A from the unknown (axisymmetric A_phi = psi / r_c), edge-extended into the PML ring."""
        unknown = self.system.unknown2d
        self.xp.copyto(self._A_window, unknown / self._radius_window if self._axisymmetric else unknown)
        self._extend_into_pml()

    def _update_fields(self):
        """A from the unknown; H from the physical face fluxes (operator flux + remanence face terms) averaged to the
        centres, B = H / nu + B_r (both 0 in held cells), w_mag = (nu_xx B_x^2 + nu_yy B_y^2) / 2; edge-extended."""
        system = self.system
        xp = self.xp
        window = system.window
        fields = self.fields
        self._write_potential_field()

        flux_x, flux_y = system.flux_at_centres()   # -c grad u; held cells 0
        free = ~system.fixed_user
        materials = self.domain.materials
        names = materials.property_names()
        remanence_x = xp.where(free, materials.remanence_x[window], 0.0) if 'remanence_x' in names else 0.0
        remanence_y = xp.where(free, materials.remanence_y[window], 0.0) if 'remanence_y' in names else 0.0
        if self._remanence_drives is not None:   # flux * length = operator part + face term
            face_drive_x, face_drive_y = self._remanence_drives
            flux_x += xp.where(free, 0.5 * (face_drive_x[:-1, :] + face_drive_x[1:, :])
                               / system.dy_window_device[None, :], 0.0)
            flux_y += xp.where(free, 0.5 * (face_drive_y[:, :-1] + face_drive_y[:, 1:])
                               / system.dx_window_device[:, None], 0.0)
        sign = -1.0 if self._axisymmetric else 1.0   # flux = (H_y, -H_x) planar, (-H_z, H_r) axisymmetric
        H_x = -sign * flux_y
        H_y = sign * flux_x
        B_x = H_x / self._nu_xx + remanence_x
        B_y = H_y / self._nu_yy + remanence_y
        fields.H_x[window] = H_x
        fields.H_y[window] = H_y
        fields.B_x[window] = B_x
        fields.B_y[window] = B_y
        self._flux_density_magnitude = xp.hypot(B_x, B_y)   # solve dtype: the saturation curve reads it
        fields.B_abs[window] = self._flux_density_magnitude
        fields.w_mag[window] = 0.5 * (self._nu_xx * B_x ** 2 + self._nu_yy * B_y ** 2)
        for destination, source in self._derived_pml_copies:
            xp.copyto(destination, source)

    def _applied_source_changed(self):
        self._source_dirty = True

    def _indicator_flux(self):
        """error_indicator's flux: (H_y, H_x) of the window, the physical flux (+-H_y, -+H_x) with the remanence face
        terms (signs drop out of the squared jumps)."""
        window = self.system.window
        return self.fields.H_y[window], self.fields.H_x[window]

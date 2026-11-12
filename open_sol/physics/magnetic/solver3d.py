"""3D magnetostatics by the reduced magnetic scalar potential on the 7-point DiffusionSystem (magnetostatics_3d_plan):
unknown u = mu0 Omega [T m] with c = mu_dc per axis, so the physical face flux c (w - grad u) is B itself; the drive w =
mu0 T + B_r / mu_dc enters as face terms (stranded coils by T-Omega: T along the coil axis, J = curl T), symmetry (u =
0) / insulation (Neumann) faces, ideal-iron held regions, a uniform external field on the held outer ring and a corner
anchor for all-Neumann models; saturating materials (J_s curve, B-H tables) by a fixed point on ln mu; energy and coil
flux linkage from the face flows. Cartesian Domain3D only."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES_3D, MU0
from ...core.damped_iteration import DampedIteration, DampingSettings
from ...core.fv_diffusion import DiffusionPhysicsBase, DiffusionSystem, ScalarBoundaryConfig
from ...core.fv_diffusion_3d import (centre_gradient_3d, edge_extension_views_3d, face_drive_divergence_3d,
                                     face_drive_integrals_3d, face_drive_terms_3d, face_flow_energy_3d, face_flows_3d,
                                     face_terms_at_centres_3d)
from ...core.monitors import FieldProbe
from .coils3d import coil_drive_per_ampere
from .kept_preconditioner import KeptPreconditionerSolver, kept_preconditioner_solver
from .fields import (MAGNETIC_3D_CURRENT_NAMES, MAGNETIC_3D_FIELD_NAMES, MAGNETIC_3D_PERMEABILITY_NAME,
                     MAGNETIC_3D_PROBE_COMPONENTS, MagneticFields3D)
from .materials import MAGNETIC_3D_SCHEMA, PERMEABILITY_3D_ARRAYS, REMANENCE_3D_ARRAYS, SATURATION_ARRAY
from .renderer import MagneticRenderer3D
from .saturation import (ANDERSON_DEPTH, TENSOR_NONLINEAR_METHODS, bh_curve, refuse_curve_conflicts,
                         saturation_permeability, tabulated_reluctivity)

_ANCHOR_TEXT = 'symmetry face, ideal iron region, external-field ring or the corner anchor'
_REMEDY_TEXT = 'Give every window cell mu_dc > 0.'
_PERMEABILITY_NAMES = ('mu_dc_xx', 'mu_dc_yy', 'mu_dc_zz')
_AXES = 'xyz'


@dataclass(frozen=True, eq=False)
class CoilRegion3D:
    """Runtime object of a stranded coil: axis index (0, 1, 2), turns, current [A] (> 0 counter-clockwise seen from
    +axis), axial length h [m] and T per ampere [1/m] (float64 window array of the run backend, along the axis)."""
    name: str
    axis_index: int
    turns: int
    current: float
    axial_length: float
    per_ampere: object


class MagneticPhysics3D(DiffusionPhysicsBase):
    """3D magnetostatics div(mu_dc (grad u - w)) = 0 on the physical window of a Cartesian Domain3D: one DiffusionSystem
    solve per change (a saturation loop when a cell saturates); phi_m = u / mu0, B from the face fluxes, H = (B - B_r)
    / (mu0 mu_dc), J = curl T, edge-extended into any PML shell."""

    name = 'magnetic'
    type_label = 'magnetostatics_3d'
    material_schema = MAGNETIC_3D_SCHEMA
    _probe_field = 'B_abs'
    _probe_label = 'magnetic probe'
    coordinate_grids = ('cartesian',)   # the ring potential, the drives and the coil slices read Cartesian widths

    def __init__(self, domain, tol: float = 1e-8, max_iter: int = 5000, compute_residual: bool = False,
                 precision: str = 'float64', nonlinear_tolerance: float = 1e-6, nonlinear_max_iterations: int = 100,
                 nonlinear_relaxation: float = 0.5, nonlinear_method: str = 'auto', linear_solver=None):
        if domain.dimensions != 3:
            raise ValueError(f'{self.type_label} needs a 3D domain (2D models: magnetostatics)')
        system = DiffusionSystem(
            domain, precision, tol, max_iter, label=lambda: f"{self.type_label} '{self.name}'",
            unknown_name='magnetic scalar potential', anchor_text=_ANCHOR_TEXT, remedy_text=_REMEDY_TEXT,
            compute_residual=compute_residual,
            linear_solver=kept_preconditioner_solver(linear_solver))   # kept across the saturation loop
        self._init_diffusion(domain, ScalarBoundaryConfig(), system)   # every face insulated (Neumann, decision 4)
        self.coils = []
        self._source_dirty = True
        self._drive_stale = True
        self._face_drives = None   # face_drive_terms_3d of w = mu0 T + B_r / mu_dc (None: no coil, no remanence)
        self._drive_rhs = None
        self._corner_anchored = False   # window cell (0, 0, 0) held at u = 0 by the type (all-Neumann models)
        # saturation: fixed point on ln of the isotropic secant mu_r of the saturating cells (float64 window, 0
        # elsewhere), Anderson-mixed or damped Picard, as the 2D tensor loop
        nonlinear_method = 'anderson' if nonlinear_method == 'auto' else nonlinear_method
        if nonlinear_method not in TENSOR_NONLINEAR_METHODS:
            raise ValueError(f"{self.type_label}: nonlinear_method must be 'auto' or one of {TENSOR_NONLINEAR_METHODS}, "
                             f'got {nonlinear_method!r} (no Newton tangent on tensor grids)')
        self.runtime_warnings = []      # append-only (unconverged saturation loops); the studies log new ones
        self.picard = DampedIteration(
            DampingSettings(nonlinear_tolerance, nonlinear_max_iterations, nonlinear_relaxation,
                            anderson_depth=ANDERSON_DEPTH if nonlinear_method == 'anderson' else 0),
            label=lambda: f"{self.type_label} '{self.name}' saturation")
        self._log_permeability_state = None   # loop state (warm start of the next loop); None = linear mu_dc
        self._saturation = None         # float64 window J_s [T], None = no J_s > 0
        self._bh_curves = []            # (window bool mask, saturation.BHCurve) per add_bh_curve, later wins
        self._nonlinear_cells = None    # bool window of the J_s > 0 and curve cells; None: no loop
        self._nonlinear_flat = None     # their flat window indices (k,): the compact loop state's cells
        self._saturation_stale = True
        self._permeability = None       # checked mu_dc_xx / _yy / _zz window views of the last full assembly
        self.prepare(domain)

    # --- PhysicsSolver contract ---------------------------------------------

    def prepare(self, domain) -> None:
        """Register the permeability arrays, prepare the window system and allocate the fields; assembly is deferred."""
        self.domain = domain
        self.xp = domain.xp   # numpy (cpu backend) or cupy
        domain.materials.add_properties(self.material_schema)
        system = self.system
        system.prepare(domain)
        if isinstance(system.linear_solver, KeptPreconditionerSolver):
            system.linear_solver.bind_window(system.window_shape)
        if min(system.window_shape) < 3:
            raise ValueError(f"{self.type_label} '{self.name}' needs at least 3 x 3 x 3 window cells, got "
                             f'{system.window_shape}')
        shape = tuple(domain.materials.shape)
        if self.fields is None or self.fields.shape != shape:
            self.fields = MagneticFields3D(shape, xp=self.xp)
        window = system.window
        self._phi_window = self.fields.phi_m[window]
        self._bind_window(self.fields.phi_m)
        self._derived_pml_copies = [pair for field_name in MAGNETIC_3D_FIELD_NAMES[1:]
                                    for pair in edge_extension_views_3d(getattr(self.fields, field_name), window)]
        self._zero_source = self.xp.zeros(system.window_shape, dtype=DTYPE)   # build_rhs input: no volume sources
        # window face / centre coordinates [m] from the window's lower corner (host float64), the external ring's frame
        self._faces = tuple(np.concatenate(([0.0], np.cumsum(widths)))
                            for widths in (system.dx_window, system.dy_window, system.dz_window))
        self._centres = tuple(0.5 * (faces[:-1] + faces[1:]) for faces in self._faces)
        self._source_dirty = True
        self._drive_stale = True
        self._corner_anchored = False
        self._log_permeability_state = None
        self._saturation_stale = True

    def finalize(self) -> None:
        """Assemble once (fails early on a non-positive permeability)."""
        self._assemble()

    def materials_changed(self, names) -> None:
        names = set(names)
        if PERMEABILITY_3D_ARRAYS & names:
            self.system.operator_dirty = True
        if names.intersection(REMANENCE_3D_ARRAYS):
            self._drive_stale = True
            self._source_dirty = True
        if SATURATION_ARRAY in names:
            self._saturation_stale = True
            self.system.operator_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Re-solve only when the operator or a drive changed (always at the first call; a saturation loop when a
        material sets saturation_polarization or a B-H curve); probes sample after a re-solve and at every dt = inf
        call, never on unchanged time steps."""
        t = 0.0 if t is None else float(t)
        system = self.system
        re_solve = system.operator_dirty or self._source_dirty or system.n_solves == 0
        if re_solve:
            if self._saturation_stale:
                self._refresh_saturation()
            if self._nonlinear_cells is not None:
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
        names = (MAGNETIC_3D_FIELD_NAMES + (MAGNETIC_3D_CURRENT_NAMES if fields.has_current_density else ())
                 + ((MAGNETIC_3D_PERMEABILITY_NAME,) if fields.has_effective_permeability else ()))
        state = {name: getattr(fields, name) for name in names}
        for array_name in self.material_schema:
            state[array_name] = getattr(self.domain.materials, array_name)
        return state

    def inject_source(self, name: str, array) -> None:
        raise ValueError(f"{self.type_label} '{self.name}' has no source slots; got '{name}'")

    def renderer(self):
        return MagneticRenderer3D(self)

    def scalars(self) -> dict:
        """energy [J], per coil <name>_flux_linkage [Wb] / <name>_inductance [H]; {} before a solve."""
        if self.system.n_solves == 0:
            return {}
        flows = self._total_face_flows()
        values = {'energy': self.field_energy(flows)}
        for coil in self.coils:
            result = self.coil_results(coil, flows)
            values[f'{coil.name}_flux_linkage'] = result['flux_linkage']
            values[f'{coil.name}_inductance'] = result['inductance']
        return values

    def work_counters(self) -> dict:
        """The saturation loop's counters (nonlinear_runs / _iterations / _unconverged) and its kept-preconditioner
        solves once it ran, else {}."""
        if not self.picard.n_runs:
            return {}
        counters = self.picard.work_counters()
        if isinstance(self.system.linear_solver, KeptPreconditionerSolver):
            counters.update(self.system.linear_solver.work_counters())
        return counters

    # --- magnetics API (settings, set_face_bc: DiffusionPhysicsBase; BCSide('neumann', 0) = magnetic insulation, the
    # default; BCSide('dirichlet', 0) = symmetry, u = 0: field normal to the face) ----------------------------------

    def add_boundary_flux(self, weights, value, outline=False) -> None:
        raise NotImplementedError(f"{self.type_label} '{self.name}' takes no boundary flux (a normal B through a face "
                                  'is not a magnetic source)')

    def add_ideal_iron(self, mask, potential=0.0) -> None:
        """Hold Omega = potential [A] (u = mu0 Omega; scalar or full-grid array) on the window cells under mask: ideal
        iron (mu -> inf: H = 0 inside, B normal to its surface); B and H read 0 there."""
        self._release_corner_anchor()   # a later held region decides the anchor again
        system = self.system
        mask_window = system.window_mask(mask, 'ideal iron')
        values = system.window_values(potential, 'ideal iron potential', system.dtype)
        system.set_fixed(mask_window, MU0 * values)   # operator_dirty: the re-assembly refreshes the drives too

    def set_external_field(self, B_x: float, B_y: float, B_z: float) -> None:
        """Uniform external flux density [T]: the outermost window cells are held at u = -B . r evaluated at their faces
        toward the free cells (r from the window's lower corner), so a uniform field is exact on any widths."""
        self._release_corner_anchor()
        values, mask = self._external_ring((float(B_x), float(B_y), float(B_z)))
        self.system.set_fixed(mask, values)

    def add_coil(self, name: str, mask, axis: str, turns: int, current: float) -> CoilRegion3D:
        """Stranded coil around axis 'x' / 'y' / 'z' on the masked winding cells (closed around a hole in every slice
        normal to the axis): T = turns current / h * s along the axis (coils3d), J = curl T."""
        if int(turns) < 1:
            raise ValueError(f"coil '{name}': turns must be >= 1, got {turns}")
        if axis not in _AXES:
            raise ValueError(f"coil '{name}': axis must be 'x', 'y' or 'z', got {axis!r}")
        system = self.system
        mask_window = system.window_mask(mask, f"coil '{name}'")
        per_ampere, axial_length = coil_drive_per_ampere(
            to_host(mask_window), axis, (system.dx_window, system.dy_window, system.dz_window), int(turns),
            f"coil '{name}'")
        coil = CoilRegion3D(name, _AXES.index(axis), int(turns), float(current), axial_length,
                            self.xp.asarray(per_ampere, dtype=np.float64))
        self.coils.append(coil)
        self._write_current_density()
        self._drive_stale = True
        self._source_dirty = True
        return coil

    def add_bh_curve(self, mask, rows) -> bool:
        """Tabulated anhysteretic B-H curve [(H [A/m], B [T]), ...] (saturation.bh_curve rules) on the window cells
        under the full-grid bool mask: isotropic mu_r = B(|H|) / (mu0 |H|) there, mu_dc ignored; a later curve wins
        shared cells. False (nothing added) when the mask covers no window cell."""
        curve = bh_curve(rows, self.xp)
        mask_window = self.system.window_mask(mask, 'B-H curve').copy()
        if not bool(self.xp.any(mask_window)):
            return False
        self._bh_curves.append((mask_window, curve))
        self._log_permeability_state = None   # the next loop starts from every curve's H = 0 limit
        self._saturation_stale = True
        self.system.operator_dirty = True
        return True

    def coil_results(self, coil: CoilRegion3D, flows=None) -> dict:
        """current, turns, flux_linkage [Wb] = sum over the faces normal to the coil axis of B_n A times the integral of
        T / I across the face (with a ring or held cells, up to their boundary term: decision 5), inductance =
        flux_linkage / current [H] (nan at zero current), energy [J] (field_energy); flows: _total_face_flows() when
        already computed."""
        system = self.system
        if system.n_solves == 0:
            raise RuntimeError(f"{self.type_label} '{self.name}': coil '{coil.name}' needs a solve first")
        flows = self._total_face_flows() if flows is None else flows
        integral = face_drive_integrals_3d(coil.per_ampere, coil.axis_index, system.geometry_3d, ~system.fixed_user)
        linkage = float(self.xp.sum(flows[coil.axis_index] * integral, dtype=self.xp.float64))
        inductance = linkage / coil.current if coil.current != 0.0 else float('nan')
        return {'current': coil.current, 'turns': coil.turns, 'flux_linkage': linkage, 'inductance': inductance,
                'energy': self.field_energy(flows)}

    def field_energy(self, flows=None) -> float:
        """Energy [J] = sum over the faces touching a free cell of (B_n A)^2 / (2 mu0 conductance), i.e. of B_n^2 / (2 mu0
        mu_face) times the face volume A d (held-free faces: the free half cell; held cells and the ring excluded; with
        remanence the energy of B alone, not the magnet's co-energy)."""
        system = self.system
        flows = self._total_face_flows() if flows is None else flows
        return face_flow_energy_3d(flows, system.face_conductances, system.face_terms, ~system.fixed_user) / (2.0 * MU0)

    def add_probe(self, ix: int, iy: int, iz: int, component: str = 'B_abs') -> FieldProbe:
        """Record field `component` (B_abs [T] or phi_m [A]) at full-grid cell (ix, iy, iz) whenever step records."""
        if component not in MAGNETIC_3D_PROBE_COMPONENTS:
            raise ValueError(f'{self.type_label} probe component must be one of {MAGNETIC_3D_PROBE_COMPONENTS}, got '
                             f'{component!r}')
        self.system.require_window_cell(ix, iy, self._probe_label, iz)
        probe = FieldProbe(ix, iy, component, iz=iz)
        self.probes.append(probe)
        return probe

    def rebuild_operator(self) -> None:
        """Re-assemble the operator, held-cell elimination and preconditioner."""
        self._assemble()

    def warm_start_fields(self) -> dict:
        """Host copy of the window unknown u, to seed a build on the same mesh."""
        return {'unknown': np.array(to_host(self.system.unknown2d))}

    def seed_warm_start(self, fields) -> tuple:
        """Write warm_start_fields() of another build (unknown on free cells, phi_m); B and H wait for the next solve."""
        self.system.seed_unknown(fields['unknown'])
        self._write_potential_field()
        self._source_dirty = True
        return ('phi_m',)

    # --- internals ------------------------------------------------------------

    def _assemble(self, full=True):
        """Assembly with c = mu_dc per axis, exp of the saturation state in the saturating cells (the system keeps
        solve-dtype copies: H and the remanence drive read them); full=False (later saturation iterations: only those
        c > 0 changed) skips the corner anchor, the mu_dc read / check and the anchor check."""
        if full:
            self._update_corner_anchor()
            self._read_permeability()
        coefficients = self._permeability
        if self._log_permeability_state is not None:   # the compact state (k,) on the saturating cells
            secant = self.xp.exp(self._log_permeability_state).astype(self.system.dtype)
            updated = []
            for mu in coefficients:
                coefficient = mu.astype(self.system.dtype)   # a contiguous copy of the window view
                coefficient.reshape(-1)[self._nonlinear_flat] = secant
                updated.append(coefficient)
            coefficients = tuple(updated)
        self.system.assemble(coefficients[0], coefficients[1], None, self.bcs, 0.0, c_zz=coefficients[2],
                             check_anchors=full)
        self._drive_stale = True

    def _read_permeability(self):
        """Keep the window views of mu_dc_xx / _yy / _zz, checked finite and > 0."""
        materials = self.domain.materials
        window = self.system.window
        xp = self.xp
        permeability = tuple(getattr(materials, name)[window] for name in _PERMEABILITY_NAMES)
        if any(bool(xp.any(~(mu > 0) | ~xp.isfinite(mu))) for mu in permeability):
            raise ValueError(f"{self.type_label} '{self.name}': mu_dc_xx / mu_dc_yy / mu_dc_zz must be finite and > 0 "
                             'in every window cell')
        self._permeability = permeability

    def _refresh_saturation(self):
        """Window J_s and the saturating cells (J_s > 0 or a B-H curve; None: no loop, mu_eff stays unallocated); a
        J_s cell with mu_dc differing between axes, a curve cell with J_s > 0 or a saturating cell with a remanence
        raises (one sync per check); the loop state is dropped when the saturating set changes or vanishes."""
        self._saturation_stale = False
        self._saturation = None
        previous_cells = self._nonlinear_cells
        self._nonlinear_cells = None
        materials = self.domain.materials
        names = materials.property_names()
        window = self.system.window
        xp = self.xp
        nonlinear_cells = xp.zeros(self.system.window_shape, dtype=bool)
        if SATURATION_ARRAY in names:
            saturation = materials.saturation_polarization[window].astype(np.float64)
            saturating = saturation > 0.0
            if bool(xp.any(saturating)):
                mu_xx, mu_yy, mu_zz = (getattr(materials, name)[window] for name in _PERMEABILITY_NAMES)
                if bool(xp.any(saturating & ((mu_xx != mu_yy) | (mu_xx != mu_zz)))):
                    raise ValueError(f"{self.type_label} '{self.name}': a cell with saturation_polarization > 0 has "
                                     'different mu_dc_xx / mu_dc_yy / mu_dc_zz (the saturating curve is isotropic: '
                                     'give one mu_dc)')
                self._saturation = saturation
                nonlinear_cells |= saturating
        if self._bh_curves:
            tabulated_cells = xp.zeros_like(nonlinear_cells)
            for mask_window, _ in self._bh_curves:
                tabulated_cells |= mask_window
            refuse_curve_conflicts(f"{self.type_label} '{self.name}'", 'cell', curve_and_polarization=(
                self._saturation is not None and bool(xp.any(tabulated_cells & nonlinear_cells))))
            nonlinear_cells |= tabulated_cells
        if self._saturation is None and not self._bh_curves:
            self._log_permeability_state = None   # back to the linear mu_dc (_assemble reads the state alone)
            return
        remanent = xp.zeros_like(nonlinear_cells)
        for name in REMANENCE_3D_ARRAYS:
            if name in names:
                remanent |= getattr(materials, name)[window] != 0
        refuse_curve_conflicts(f"{self.type_label} '{self.name}'", 'cell',
                               saturating_remanence=bool(xp.any(nonlinear_cells & remanent)))
        if previous_cells is None or not bool(xp.array_equal(nonlinear_cells, previous_cells)):
            self._log_permeability_state = None   # new saturating set: restart from every curve's H = 0 limit
            self._nonlinear_flat = xp.flatnonzero(nonlinear_cells.ravel())
        self._nonlinear_cells = nonlinear_cells
        if self._saturation is not None:   # compact J_s and curve masks on the saturating cells (C order)
            self._saturation = self._saturation[nonlinear_cells]
        self._compact_curves = [(mask_window[nonlinear_cells], curve) for mask_window, curve in self._bh_curves]
        fields = self.fields
        if not fields.has_effective_permeability:
            fields.allocate_effective_permeability()
            self._permeability_pml_copies = edge_extension_views_3d(fields.mu_eff, window)

    def _solve_saturated(self):
        """Fixed point on ln mu_r of the saturating cells (core.damped_iteration, Anderson-mixed or damped Picard): each
        iteration assembles with c = exp(state) there (the first one fully), solves warm-started at the previous u and
        proposes ln of the curves' secant mu_r at the centre |H| = |B| / (mu0 mu_r), H-based for both curves (|B| from
        the face fluxes, decision 7); scale 1 (relative change of mu). The first state is every curve's H = 0 limit."""
        xp = self.xp
        first = [True]   # the first proposal of each loop assembles fully
        if self._log_permeability_state is None:
            self._read_permeability()
            self._log_permeability_state = self._saturated_log_permeability(
                xp.zeros(self._nonlinear_flat.size, dtype=np.float64))
        state = self._log_permeability_state

        def propose(log_permeability):   # the state itself, read by _assemble through the attribute
            if isinstance(self.system.linear_solver, KeptPreconditionerSolver):   # later iterations may keep its build
                self.system.linear_solver.offer_state(log_permeability, fresh=first[0])
            self._assemble(full=first[0])
            first[0] = False
            self._solve_once()
            flux_magnitude = self._flux_density_magnitude.ravel()[self._nonlinear_flat].astype(np.float64)
            field_magnitude = flux_magnitude / (MU0 * xp.exp(log_permeability))
            return self._saturated_log_permeability(field_magnitude)

        # the operator and fields stay those of the last iterate's solve; the state warm-starts the next loop
        if not self.picard.run(state, propose, 1.0):
            self.runtime_warnings.append(self.picard.last_warning)

    def _saturated_log_permeability(self, field_magnitude):
        """ln of the secant mu_r (float64 (k,)) of the curves on the saturating cells at their float64 centre |H| =
        field_magnitude (k,) (J_s curve with mu_dc_xx, isotropic there; each B-H table on its cells)."""
        xp = self.xp
        permeability = self._permeability[0].ravel()[self._nonlinear_flat].astype(np.float64)
        secant = (permeability if self._saturation is None
                  else saturation_permeability(field_magnitude, permeability, self._saturation, xp))
        for mask, curve in self._compact_curves:
            secant = xp.where(mask, 1.0 / (MU0 * tabulated_reluctivity(field_magnitude, curve, xp)), secant)
        return xp.log(secant)

    def _update_corner_anchor(self):
        """Hold window cell (0, 0, 0) at u = 0 when no face is a symmetry face and no cell is held (the all-Neumann
        problem is consistent, unique up to a constant); release it once a symmetry face exists. Held regions and the
        ring release it themselves before they hold."""
        dirichlet_face = any(getattr(self.bcs, face).kind != 'neumann' for face in FACES_3D)
        if self._corner_anchored:
            if dirichlet_face:
                self._release_corner_anchor()
            return
        system = self.system
        if dirichlet_face or bool(self.xp.any(system.fixed_user)):
            return
        corner = self.xp.zeros(system.window_shape, dtype=bool)
        corner[0, 0, 0] = True
        system.set_fixed(corner, self.xp.zeros((), dtype=system.dtype))
        self._corner_anchored = True

    def _release_corner_anchor(self):
        if not self._corner_anchored:
            return
        system = self.system
        system.fixed_user[0, 0, 0] = False
        system.fixed_user_values[0, 0, 0] = 0.0
        system.operator_dirty = True
        self._corner_anchored = False

    def _external_ring(self, field):
        """(values, mask) window device arrays of the held outer ring: u = -B . r on each outer cell layer at its face
        toward the free cells (x layers first, then y, z: the shared edges touch no free cell)."""
        xp = self.xp
        system = self.system
        values = xp.zeros(system.window_shape, dtype=system.dtype)
        mask = xp.zeros(system.window_shape, dtype=bool)
        for axis in range(3):
            for edge, face_index in ((0, 1), (-1, -2)):   # the layer's face shared with the free neighbour layer
                coordinates = list(self._centres)
                coordinates[axis] = self._faces[axis][[face_index]]
                grids = np.meshgrid(*coordinates, indexing='ij')
                layer = np.take(-(field[0] * grids[0] + field[1] * grids[1] + field[2] * grids[2]), 0, axis=axis)
                index = [slice(None)] * 3
                index[axis] = edge
                values[tuple(index)] = xp.asarray(layer, dtype=system.dtype)
                mask[tuple(index)] = True
        return values, mask

    def _solve_once(self):
        """One linear solve on the assembled operator: rhs (coil and remanence face drives), CG, fields."""
        system = self.system
        system.build_rhs(self._zero_source, self._zero_source)
        self._add_drive_rhs()
        system.solve()
        self._update_fields()

    def _add_drive_rhs(self):
        """Add the face drive's divergence to the rhs built by build_rhs (free rows) and to rhs_full; no-op without a
        coil or remanence array."""
        if self._drive_stale:
            self._face_drives = self._drive_terms()
            self._drive_rhs = None if self._face_drives is None else face_drive_divergence_3d(self._face_drives)
            self._drive_stale = False
        if self._drive_rhs is None:
            return
        system = self.system
        system.rhs2d += system.free * self._drive_rhs
        system.rhs_full += self._drive_rhs

    def _drive_terms(self):
        """face_drive_terms_3d of w = B_r / mu_dc (the assembled c) + mu0 I T per coil along its axis; None without a
        coil or an allocated remanence array."""
        materials = self.domain.materials
        names = materials.property_names()
        if not self.coils and not any(name in names for name in REMANENCE_3D_ARRAYS):
            return None
        system = self.system
        coefficients = (system.assembled_c_xx, system.assembled_c_yy, system.assembled_c_zz)
        drives = [getattr(materials, name)[system.window] / coefficient if name in names else None
                  for name, coefficient in zip(REMANENCE_3D_ARRAYS, coefficients)]
        for coil in self.coils:
            coil_drive = (MU0 * coil.current) * coil.per_ampere
            axis = coil.axis_index
            drives[axis] = coil_drive if drives[axis] is None else drives[axis] + coil_drive
        zero = self.xp.zeros((), dtype=system.dtype)
        return face_drive_terms_3d(tuple(zero if drive is None else drive for drive in drives),
                                   system.face_conductances, system.face_terms, system.geometry_3d, ~system.fixed_user)

    def _total_face_flows(self):
        """Face flows [T m^2] = B_n A toward +x / +y / +z of the last solve: operator flows plus the drive terms."""
        system = self.system
        flows = face_flows_3d(system.unknown2d, system.face_conductances, system.face_terms)
        if self._face_drives is None:
            return flows
        return tuple(flow + drive for flow, drive in zip(flows, self._face_drives))

    def _write_current_density(self):
        """J = curl T [A/m^2] of every coil (T along axis a: J_i = eps_ija dT/dx_j, the transverse centre gradients),
        written to the window and edge-extended; allocates J with the first coil."""
        fields = self.fields
        system = self.system
        window = system.window
        if not fields.has_current_density:
            fields.allocate_current_density()
            self._current_pml_copies = [pair for field_name in MAGNETIC_3D_CURRENT_NAMES
                                        for pair in edge_extension_views_3d(getattr(fields, field_name), window)]
        density = [0.0, 0.0, 0.0]
        for coil in self.coils:
            potential = coil.current * coil.per_ampere   # T [A/m]
            axis = coil.axis_index
            for gradient_axis in range(3):
                if gradient_axis == axis:
                    continue
                component = 3 - axis - gradient_axis
                sign = 1.0 if (gradient_axis - component) % 3 == 1 else -1.0   # Levi-Civita eps(component, j, axis)
                density[component] = density[component] + sign * centre_gradient_3d(potential, gradient_axis,
                                                                                    system.geometry_3d)
        for name, values in zip(MAGNETIC_3D_CURRENT_NAMES, density):
            getattr(fields, name)[window] = values
        for destination, source in self._current_pml_copies:
            self.xp.copyto(destination, source)

    def _write_potential_field(self):
        """phi_m = u / mu0 [A] on the window, edge-extended into the PML shell."""
        self.xp.copyto(self._phi_window, self.system.unknown2d / MU0)
        self._extend_into_pml()

    def _update_fields(self):
        """phi_m; B = operator face flux + drive face terms averaged to the centres (0 in held cells), H = (B - B_r) /
        (mu0 c) with the assembled c (secant mu in saturating cells), B_abs (kept in the solve dtype for the saturation
        proposal), w_mag = sum B_c^2 / (2 mu0 mu_c) (the energy scalar's density, as 2D), mu_eff = the mean of the
        three c when allocated; edge-extended."""
        system = self.system
        xp = self.xp
        window = system.window
        fields = self.fields
        self._write_potential_field()
        free = ~system.fixed_user
        flux = system.flux_at_centres()   # -c grad u per axis [T]
        if self._face_drives is not None:
            drive = face_terms_at_centres_3d(self._face_drives, system.geometry_3d, free)
            flux = tuple(operator_part + drive_part for operator_part, drive_part in zip(flux, drive))
        names = self.domain.materials.property_names()
        coefficients = (system.assembled_c_xx, system.assembled_c_yy, system.assembled_c_zz)
        magnitude_squared = 0.0
        energy = 0.0
        for axis, flux_density, coefficient, remanence_name in zip(_AXES, flux, coefficients, REMANENCE_3D_ARRAYS):
            remanence = (xp.where(free, getattr(self.domain.materials, remanence_name)[window], 0.0)
                         if remanence_name in names else 0.0)
            field_strength = (flux_density - remanence) / (MU0 * coefficient)
            getattr(fields, f'B_{axis}')[window] = flux_density
            getattr(fields, f'H_{axis}')[window] = field_strength
            magnitude_squared = magnitude_squared + flux_density * flux_density
            energy = energy + flux_density * flux_density / (MU0 * coefficient)   # B^2 / (mu0 mu): as 2D and the energy
        magnitude = xp.sqrt(magnitude_squared)
        fields.B_abs[window] = magnitude
        fields.w_mag[window] = 0.5 * energy
        self._flux_density_magnitude = magnitude if self._nonlinear_cells is not None else None
        for destination, source in self._derived_pml_copies:
            xp.copyto(destination, source)
        if fields.has_effective_permeability:
            fields.mu_eff[window] = sum(coefficients) / 3.0
            for destination, source in self._permeability_pml_copies:
                xp.copyto(destination, source)

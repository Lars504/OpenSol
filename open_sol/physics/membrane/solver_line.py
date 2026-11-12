"""membrane in a 1D model (MembraneLinePhysics): complex line elements (fem.line_fem) on the cells of the line for D (1 +
j eta) (w'''' - 2 k_y^2 w'' + k_y^4 w) - (T_xx (1 + j eta) w')' + T_yy (1 + j eta) k_y^2 w - rho_s omega^2 w = p (a strip
uniform along y, or a guide along y with axial wavenumber k_y; Lagrange P1 / P2, Hermite cubic with bending), and
LineHarmonicSystem: fem.harmonic.HarmonicSystemBase on a line."""
# Dofs ascend along the line (fem.line_fem: P1 node k; P2 node 2 k, midpoint 2 e + 1; Hermite w 2 k, w' 2 k + 1); the
# system numbering is the dofs of the active (non-hole) elements in that order. Hermite slope dofs are scaled by the
# mean cell width h (the system solves for h w': dof_scale 1 / h, the slope basis / h), so the w and w' rows of the SI
# operator, ~1 / h^2 apart, are
# equilibrated for the factor and ARPACK; `values` and resonance fields are physical. Units per unit length along y:
# loads N/m (end forces, point forces), pressure Pa. Bending: clamped ends hold w and w', simply supported ends w only,
# symmetry ends w' only;
# the plate Poisson ratio enters only free-end moments, which the energy D (W''^2 + 2 k_y^2 W'^2 + k_y^4 W^2) leaves out
# (exact for clamped / simply supported ends).
from dataclasses import dataclass, fields

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE
from ...fem.assembly import congruence, restrict
from ...fem.fixed_pattern import scatter_add
from ...fem.harmonic import (HarmonicCondition, HarmonicEigenPencil, HarmonicSystemBase, dirichlet_values, is_zero,
                             system_positions)
from ...fem.line_fem import (active_dofs, assemble_line_bending, assemble_line_load, assemble_line_mass,
                             assemble_line_stiffness, assemble_point_load, dof_points, end_dofs, evaluate_line,
                             line_dofs, line_evaluation_matrix)
from .renderer import MEMBRANE_FIELD_INFOS, PHASOR_FIELD_NAMES, MembraneLineRenderer
from .solver_common import MembraneBase, complex_amplitude

LINE_FACES = ('x_lo', 'x_hi')
_CLAMPED = HarmonicCondition('dirichlet', 0.0)


def _point_matrix(positions, values, size):
    """CSR (size, size) holding values at the diagonal positions (end admittance / damping)."""
    from scipy.sparse import csr_matrix
    positions = np.asarray(positions, dtype=np.int64)
    values = np.broadcast_to(np.asarray(values), positions.shape)
    return csr_matrix((values, (positions, positions)), shape=(size, size))


def _diagonal(values):
    from scipy.sparse import diags
    return diags(np.asarray(values, dtype=np.float64)).tocsr()


@dataclass(frozen=True, eq=False)
class ScaledEigenPencil(HarmonicEigenPencil):
    """HarmonicEigenPencil of a scaled system: spread returns physical full-numbering fields (x dof_scale)."""
    dof_scale: object = None

    def spread(self, vector) -> np.ndarray:
        full = super().spread(vector)
        return full if self.dof_scale is None else full * self.dof_scale


class LineHarmonicSystem(HarmonicSystemBase):
    """Complex scalar harmonic system bending (u'')'' - (a u')' + (b + j omega c - omega^2 m) u = f on Lagrange P1 / P2
    or Hermite cubic line elements (fem.line_fem) over cell edges [m]; faces 'x_lo' / 'x_hi' are the end value dofs
    (Neumann / Robin values are point terms there), no PML. Coefficients: number or (n_elements,) arrays; bending only on
    Hermite. Hermite slope unknowns are scaled by the mean cell width (module header)."""
    faces = LINE_FACES

    def __init__(self, edges, order=2, linear_solver=None, *, family='lagrange', symmetric=True,
                 label=lambda: 'line harmonic fem'):
        if (family, order) not in (('lagrange', 1), ('lagrange', 2), ('hermite', 3)):
            raise ValueError(f'{label()}: line elements must be lagrange 1 / 2 or hermite 3, got {family} {order!r}')
        self.order = int(order)
        self.family = family
        self.dofs = line_dofs(edges, family, self.order)
        self.dof_scale = np.ones(self.dofs.n_dofs)
        if family == 'hermite':
            self.dof_scale[1::2] = 1.0 / float(np.mean(self.dofs.widths))   # unknown h w' (physical = scaled x this)
        self.bending = 0.0
        super().__init__(self.dofs.n_dofs, linear_solver, None, 'dirichlet', symmetric, label)

    def set_bending(self, bending) -> None:
        """Bending coefficient D (number or (n_elements,), complex D (1 + j eta)) of (D u'')'' (Hermite only)."""
        if self.family != 'hermite' and not is_zero(bending):
            raise ValueError(f'{self.label()}: bending needs Hermite (C1) line elements')
        self.bending = bending
        self.operator_dirty = True

    def active_elements(self) -> np.ndarray:
        if self.active is None:
            return np.ones(self.dofs.n_elements, dtype=bool)
        if self.active.shape != (self.dofs.n_elements,):
            raise ValueError(f'{self.label()}: active mask has shape {self.active.shape}, expected '
                             f'({self.dofs.n_elements},)')
        return self.active

    def face_entities(self, face) -> np.ndarray:
        """Full-numbering value dof of the end node of a face."""
        return end_dofs(self.dofs, face)

    def _assemble_terms(self) -> dict:
        dofs = self.dofs
        active = self.active_elements()
        if not active.any():
            raise ValueError(f'{self.label()}: no active element')
        global_ids = np.flatnonzero(active_dofs(dofs, active)).astype(np.int64)
        size = global_ids.size
        coefficients = self.coefficients
        stiffness = [restrict(assemble_line_stiffness(dofs, coefficients['a'], elements=active), global_ids)]
        if not is_zero(self.bending):
            stiffness.append(restrict(assemble_line_bending(dofs, self.bending, elements=active), global_ids))
        damping, mass = [], []
        for key, terms in (('b', stiffness), ('damping', damping), ('mass', mass)):
            if not is_zero(coefficients[key]):
                terms.append(restrict(assemble_line_mass(dofs, coefficients[key], elements=active), global_ids))
        points = dof_points(dofs)
        scale = self.dof_scale[global_ids]
        boundary_load = np.zeros(size, dtype=np.complex128)
        constraints = []
        for entities, condition, sequence in self._condition_items():
            positions, _ = system_positions(global_ids, entities)
            if not positions.size:
                continue
            if condition.kind == 'dirichlet':
                values = dirichlet_values(condition.value, points[global_ids[positions]][:, None])
                constraints.append((sequence, positions, values / scale[positions]))
                continue
            if condition.kind == 'robin':
                if not is_zero(condition.admittance):
                    stiffness.append(_point_matrix(positions, condition.admittance, size))
                if not is_zero(condition.damping):
                    damping.append(_point_matrix(positions, condition.damping, size))
            if callable(condition.value):
                raise ValueError(f'{self.label()}: a line end takes a number as its Neumann / Robin value')
            if not is_zero(condition.value):
                scatter_add(boundary_load, positions, np.full(np.size(positions), complex(condition.value)))
        transform = _diagonal(scale)
        stiffness, damping, mass = ([congruence(transform, matrix) for matrix in terms]
                                    for terms in (stiffness, damping, mass))
        return {'global_ids': global_ids, 'stiffness': stiffness, 'damping': damping, 'mass': mass,
                'boundary_load': boundary_load * scale, 'constraints': constraints,
                'in_window': np.ones(size, dtype=bool)}

    def _source_load(self) -> np.ndarray:
        dofs, global_ids = self.dofs, self.global_ids
        active = self.active_elements()
        full = np.zeros(dofs.n_dofs, dtype=np.complex128)
        if self.source is not None and not is_zero(self.source):
            full += assemble_line_load(dofs, self.source, elements=active)
        if self.point_loads:
            positions = np.array([point[0] for point, _ in self.point_loads], dtype=np.float64)
            values = np.array([value for _, value in self.point_loads], dtype=np.complex128)
            point_load = assemble_point_load(dofs, positions, values)
            outside = ~active_dofs(dofs, active) & (point_load != 0.0)
            if outside.any():
                raise ValueError(f'{self.label()}: point load(s) on dofs of no active element (a hole) at x = '
                                 f'{dof_points(dofs)[outside].tolist()} m')
            full += point_load
        return (full * self.dof_scale)[global_ids]

    def _physical_values(self, full) -> np.ndarray:
        """The physical (unscaled) full-numbering values of scaled system values (HarmonicSystemBase hook)."""
        return full * self.dof_scale

    def eigen_pencil(self, region_elements=None) -> ScaledEigenPencil:
        """The scaled eigen pencil whose spread gives physical fields."""
        pencil = super().eigen_pencil(region_elements)
        return ScaledEigenPencil(**{item.name: getattr(pencil, item.name) for item in fields(pencil)},
                                 dof_scale=self.dof_scale)

    def _element_rows(self, element_mask) -> np.ndarray:
        """Bool per system dof touched by the masked active elements."""
        mask = np.asarray(element_mask, dtype=bool) & self.active_elements()
        return active_dofs(self.dofs, mask)[self.global_ids]


class MembraneLinePhysics(MembraneBase):
    """Membrane in a 1D model (a padding-free tensor Domain one cell across): line elements on the window cells (Hermite
    cubic when a material sets bending_stiffness), the axial wavenumber k_y as the T_yy k_y^2 (and D k_y^2 / k_y^4)
    terms; ends x_lo / x_hi clamped, simply supported or free (a force per length there)."""

    _valid_faces = LINE_FACES

    def __init__(self, domain, *, linear_solver=None, eigen_solver=None, target_frequency=0.0,
                 discretization='quadratic', axial_wavenumber=0.0):
        coordinates = getattr(domain, 'coordinates', None)
        if getattr(domain, 'grid_kind', 'tensor') != 'tensor' or not getattr(coordinates, 'line', False):
            raise ValueError('membrane: MembraneLinePhysics needs a 1D model (DomainSpec size_y null); a 2D model '
                             'runs on a triangle mesh (MembraneFEMPhysics)')
        if not np.isfinite(float(axial_wavenumber)):
            raise ValueError(f'membrane: axial_wavenumber must be finite, got {axial_wavenumber!r}')
        self.axial_wavenumber = float(axial_wavenumber)
        super().__init__(domain, linear_solver=linear_solver, eigen_solver=eigen_solver,
                         target_frequency=target_frequency, discretization=discretization)

    # --- geometry hooks -------------------------------------------------------------
    def _prepare_geometry(self) -> None:
        window_x, window_y = self.domain.physical_window()
        self._window = (window_x, window_y)
        edges = np.asarray(self.domain.cell_edges()[0], dtype=np.float64)[window_x.start:window_x.stop + 1]
        self.line_family, self.line_order = ('hermite', 3) if self.bending else ('lagrange', self.order)
        self.line_dofs = line_dofs(edges, self.line_family, self.line_order)
        self._centres = 0.5 * (edges[:-1] + edges[1:])
        self._centre_evaluation = None   # (line dofs, centre CSR, values, centre values) of _element_phasors
        self._face_clamps = set()
        self._face_supports = set()  # simply supported ends (w held, slope free)
        self._face_symmetries = set()  # even mirror ends (w' held with bending, w free)
        self._face_forces = {}    # face -> complex force per length along y [N/m]
        self._point_forces = []     # (x, complex force per length along y [N/m])
        self._pressure = np.zeros(self._centres.size, dtype=np.complex128)   # [Pa] per cell

    def _element_text(self) -> str:
        return 'Hermite cubic' if self.line_family == 'hermite' else f'P{self.order}'

    def _displacement_dofs(self):
        return slice(0, None, 2) if self.line_family == 'hermite' else slice(None)

    def _field_shape(self) -> tuple:
        return tuple(self.domain.materials.shape)

    def _n_elements(self) -> int:
        return int(self._centres.size)

    def _n_full(self) -> int:
        return int(self.line_dofs.n_dofs)

    def _element_values(self, array) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)[self._window].reshape(-1)

    def _field_array(self, values) -> np.ndarray:
        full = np.zeros(self._field_shape())
        full[self._window] = np.asarray(values).reshape(-1, 1)
        return full

    def _new_system(self, pml):
        system = LineHarmonicSystem(self.line_dofs.edges, self.line_order, self.linear_solver,
                                    family=self.line_family, label=self._label)
        for face in LINE_FACES:
            if face in self._face_clamps or face in self._face_supports:
                system.set_face_condition(face, _CLAMPED)
            elif face in self._face_forces:
                system.set_face_condition(face, HarmonicCondition('neumann', self._face_forces[face]))
            slope_held = face in self._face_clamps or face in self._face_symmetries
            if slope_held and self.line_family == 'hermite':
                system.add_boundary_condition(end_dofs(system.dofs, face, value=False, slope=True), _CLAMPED)
        for x, force in self._point_forces:
            system.add_point_load((x,), force)
        if np.any(self._pressure != 0.0):
            system.set_source(self._pressure.copy())
        return system

    def _apply_coefficients(self, system, coefficients) -> None:
        """a = (T_xx + 2 D k_y^2) (1 + j eta), b = (T_yy k_y^2 + D k_y^4) (1 + j eta), bending D (1 + j eta), mass
        rho_s per cell; holes inactive (tension_xy unused)."""
        factor = (1.0 + 1j * coefficients.loss) if coefficients.lossy else 1.0
        k_squared = self.axial_wavenumber ** 2
        bending = coefficients.bending
        stiffness = (coefficients.tension_xx + 2.0 * bending * k_squared) * factor
        transverse = (coefficients.tension_yy * k_squared + bending * k_squared ** 2) * factor
        system.set_coefficients(a=stiffness, b=transverse if k_squared != 0.0 else 0.0,
                                mass=coefficients.density.copy(), damping=0.0)
        if self.line_family == 'hermite':
            system.set_bending(coefficients.complex_bending())
        system.set_active(~coefficients.hole)

    def _probe_value(self, probe, values) -> complex:
        return complex(evaluate_line(self.line_dofs, values, [probe.location])[0])

    def _element_phasors(self) -> np.ndarray:
        """Complex128 w at the cell centres (0 in holes): the centre evaluation matrix kept per line dof map, the
        centre values per shown solution (`values` is replaced, never written in place, per show)."""
        coefficients = self._read_coefficients()
        kept = self._centre_evaluation
        if kept is None or kept[0] is not self.line_dofs:
            kept = self._centre_evaluation = (self.line_dofs, line_evaluation_matrix(self.line_dofs, self._centres),
                                              None, None)
        if kept[2] is not self.values:
            kept = self._centre_evaluation = kept[:2] + (self.values, kept[1] @ self.values)
        return np.where(coefficients.hole, 0.0, kept[3])

    # --- membrane API (the definition callbacks) --------------------------------------
    def add_pressure(self, mask, pressure, phase=0.0) -> None:
        """Add a pressure amplitude [Pa] (number or full-grid array) at phase [deg] on the cells under the mask."""
        cells = np.asarray(to_host(mask), dtype=bool)[self._window].reshape(-1)
        values = np.asarray(to_host(pressure), dtype=np.float64)
        values = values[self._window].reshape(-1) if values.ndim else np.broadcast_to(values, cells.shape)
        self._pressure += np.where(cells, complex_amplitude(values, phase), 0.0)
        self._discard_system()

    def add_point_force(self, point, force, phase=0.0) -> None:
        """Add a transverse force per length along y [N/m] at phase [deg] on the line through x = point[0]."""
        self._point_forces.append((float(point[0]), complex(complex_amplitude(force, phase))))
        self._discard_system()

    def add_probe_at(self, name, x):
        """Harmonic probe of w at x [m] on the line."""
        x = float(x)
        edges = self.line_dofs.edges
        if not edges[0] - 1e-9 * self.line_dofs.length <= x <= edges[-1] + 1e-9 * self.line_dofs.length:
            raise ValueError(f'membrane probe at x = {x} m lies off the line [{edges[0]}, {edges[-1]}] m')
        return self._new_probe(name, x)

    # --- display ----------------------------------------------------------------------
    def renderer(self):
        return MembraneLineRenderer(self)

    def window_field(self, component):
        """Float32 (nx, 1) run-backend array of a display field at the window cell centres."""
        if component in PHASOR_FIELD_NAMES:
            phasors = self._element_phasors()
            parts = {'w_re': phasors.real, 'w_im': phasors.imag, 'w_abs': np.abs(phasors),
                     'w_phase': np.angle(phasors)}
            values = parts[component]
        elif component in MEMBRANE_FIELD_INFOS:
            values = self._display_values(component)
        else:
            raise KeyError(f"{self._label()} has no field '{component}'")
        return self.xp.asarray(np.asarray(values).reshape(-1, 1), dtype=DTYPE)

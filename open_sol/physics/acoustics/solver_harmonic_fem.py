"""pressure_acoustics on triangle meshes (PressureAcousticsFEMPhysics) and the dimension-free AcousticHarmonicBase it
shares with solver_harmonic_fem3d: the frequency-domain Helmholtz equation of a fluid by complex Lagrange P1 / P2 on
fem.harmonic (planar or axisymmetric (r, z)), rigid obstacles, boundary conditions, point sources, a fem.pml ring, the
scattered-field background pressure and complex probe / surface records per frequency (core.harmonic_probe)."""
# Equation (exp(+j omega t)): -div(grad p / rho) - omega^2 p / (rho c^2) = j omega q, so fem.harmonic's a = 1 / rho and
# mass = 1 / (rho c^2); q = volume source rate (point source Q delta). Velocity v = j grad p / (omega rho). Conditions
# (fluxes n_out . grad p / rho): hard 0, motion = (j omega)^power x the inward normal amplitude (acceleration 0, velocity
# 1, displacement 2), impedance Z: -j omega p / Z (Robin damping 1 / Z; default rho c of the adjacent fluid), soft p = 0,
# pressure p = value. Outer faces act on their window edge only (a wall continues along a PML side as sound hard);
# conditions on object outlines add up (held values: the later feature wins); a PML face takes none.
# Scattered field (background p_b; the unknown is p_s, total p = p_s + p_b in the window, p_s in the PML): p_b solves
# the background medium (rho_b, c_b) exactly, so with b_K = a_b, m_b of it the total-field weak form leaves the source
# f_s = [K_s - omega^2 M_s] p_b,h (p_b,h = nodal values) with the element coefficients -(a - a_b), -(m - m_b) on fluid
# window elements (|difference| <= CONTRAST_RTOL relative: none) and +a_b, +m_b on rigid-obstacle window elements (their
# outline flux), plus -int n . a_b grad p_b over every non-PML outer window face (0 on a symmetry plane of p_b), the
# total-field Robin / Dirichlet terms (impedance load -j omega p_b / Z, held values given - p_b) and no PML-interface
# term (the PML absorbs p_s only). All per frequency: extra loads and held-value shifts on the assembled pencil
# (HarmonicSystemBase.set_frequency_load / set_held_offsets), no re-assembly. Probes and surface means use p_b evaluated
# exactly at their points.
import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE
from ...core.harmonic_probe import HarmonicProbeRecord, harmonic_probe_result
from ...core.renderer import FieldInfo
from ...core.solver_base import PhysicsSolver
from ...fem.assembly import _edge_quadrature, assemble_edge_load, assemble_mass, assemble_stiffness
from ...fem.diffusion import window_face_edges
from ...fem.dofs import lagrange_dofs
from ...fem.elements import lagrange_basis, lagrange_values
from ...fem.harmonic import FACES_2D, RING_ANGLE, HarmonicCondition, ScalarHarmonicSystem
from ...fem.mesher import MERGE_TOLERANCE
from ...fem.pml import DEFAULT_PML_REFLECTION, pml_ring
from ...fem.reduced_outputs import field_policy, wants_fields
from ...fem.renderer import FEMFieldRenderer
from ...fem.sampling import Location, evaluate, locator

ACOUSTIC_HARMONIC_SCHEMA = {'rho': 0.0, 'sound_speed': 0.0}   # the elastic_waves MaterialGrid arrays of the same names
DISCRETIZATIONS = {'quadratic': 2, 'linear': 1}
REFERENCE_PRESSURE = 20e-6     # Pa rms: 0 dB of the sound pressure level
SPL_FLOOR = -120.0             # dB: the level shown where |p| = 0
CONTRAST_RTOL = 1e-6           # relative coefficient difference that still counts as the background medium (float32)
GRADIENT_STEP_RTOL = 1e-6      # central-difference step of an expression background, x the longest window side
BOUNDARY_KINDS = ('hard', 'soft', 'pressure', 'impedance', 'motion')
BACKGROUND_KINDS = ('plane_wave', 'standing_wave', 'expression')
SIDES = ('all', '+x', '-x', '+y', '-y', '+z', '-z')
FACE_NORMALS = {'x_lo': (-1.0, 0.0, 0.0), 'x_hi': (1.0, 0.0, 0.0), 'y_lo': (0.0, -1.0, 0.0), 'y_hi': (0.0, 1.0, 0.0),
                'z_lo': (0.0, 0.0, -1.0), 'z_hi': (0.0, 0.0, 1.0)}
NODAL_FIELDS = ('p_abs', 'p_re', 'p_im', 'spl', 'p_s_abs', 'p_s_re', 'p_s_im')
VELOCITY_FIELDS = ('v_abs', 'v_x_re', 'v_x_im', 'v_y_re', 'v_y_im', 'v_z_re', 'v_z_im')
THREE_D_ONLY_FIELDS = ('v_z_re', 'v_z_im')
STATE_NAMES = ('p_abs', 'p_re', 'p_im')   # per-element centroid values of the total pressure (D12)


def _pressure_info(name, label, signed=False, primary=False):
    return FieldInfo(name, label, unit='Pa', signed=signed, cmap='RdBu_r' if signed else 'viridis', primary=primary,
                     parity='even' if signed else '')


def _velocity_info(name, label):
    return FieldInfo(name, label, unit='m/s', signed=True, cmap='RdBu_r')


FIELD_INFOS = {
    'p_abs': _pressure_info('p_abs', 'Pressure amplitude', primary=True),
    'p_re': _pressure_info('p_re', 'Pressure, real part', signed=True),
    'p_im': _pressure_info('p_im', 'Pressure, imaginary part', signed=True),
    'spl': FieldInfo('spl', 'Sound pressure level', unit='dB', cmap='magma'),
    'p_s_abs': _pressure_info('p_s_abs', 'Scattered pressure amplitude'),
    'p_s_re': _pressure_info('p_s_re', 'Scattered pressure, real part', signed=True),
    'p_s_im': _pressure_info('p_s_im', 'Scattered pressure, imaginary part', signed=True),
    'v_abs': FieldInfo('v_abs', 'Particle velocity amplitude', unit='m/s', cmap='viridis'),
    'v_x_re': _velocity_info('v_x_re', 'Particle velocity x (axisymmetric: r), real part'),
    'v_x_im': _velocity_info('v_x_im', 'Particle velocity x (axisymmetric: r), imaginary part'),
    'v_y_re': _velocity_info('v_y_re', 'Particle velocity y (axisymmetric: z), real part'),
    'v_y_im': _velocity_info('v_y_im', 'Particle velocity y (axisymmetric: z), imaginary part'),
    'v_z_re': _velocity_info('v_z_re', 'Particle velocity z (3D), real part'),
    'v_z_im': _velocity_info('v_z_im', 'Particle velocity z (3D), imaginary part'),
    'rho': FieldInfo('rho', 'Density', unit='kg/m^3', cmap='cividis'),
    'sound_speed': FieldInfo('sound_speed', 'Sound speed', unit='m/s', cmap='cividis'),
}


def _host(array) -> np.ndarray:
    return np.asarray(to_host(array), dtype=np.float64)


def _entity_values(values, entities):
    """A number, or a per-mesh-entity array taken at `entities` as (k, 1) (edge / face coefficient forms)."""
    return values if np.ndim(values) == 0 else np.asarray(values)[entities][:, None]


def _as_entity_callable(value):
    """Edge / face coefficient form f(points (k, q, d), entities) of a number or a function of points (..., d)."""
    if callable(value):
        return lambda points, entities: value(points)
    return value


# ---- background pressure ----
@dataclass(frozen=True, eq=False)
class BackgroundPressure:
    """Background pressure p_b of the scattered-field formulation (module header): kind (BACKGROUND_KINDS), complex
    amplitude p0 [Pa], unit propagation direction (d,), rigid-plane coordinate [m] (standing_wave), background density
    [kg/m^3] and sound speed [m/s]; 'expression': real / imaginary parts as functions of points (..., d) [m]."""
    kind: str
    amplitude: complex = 1.0
    direction: tuple = (0.0, -1.0)
    plane_position: float = 0.0
    density: float = 1.2
    sound_speed: float = 343.0
    real_part: object = None
    imaginary_part: object = None
    gradient_step: float = 0.0   # [m] central differences of an expression (0: set by the physics)

    def __post_init__(self):
        if self.kind not in BACKGROUND_KINDS:
            raise ValueError(f"background kind must be one of {', '.join(BACKGROUND_KINDS)}, got {self.kind!r}")
        if not (self.density > 0.0 and self.sound_speed > 0.0):
            raise ValueError(f'background density and sound speed must be > 0, got {self.density!r} kg/m^3, '
                             f'{self.sound_speed!r} m/s')
        if self.kind == 'expression' and not (callable(self.real_part) and callable(self.imaginary_part)):
            raise ValueError("an 'expression' background needs real_part and imaginary_part functions")
        if self.kind != 'expression' and not np.linalg.norm(self.direction) > 0.0:
            raise ValueError(f'background direction must not be zero, got {self.direction!r}')

    @property
    def unit_direction(self) -> np.ndarray:
        direction = np.asarray(self.direction, dtype=np.float64)
        return direction / np.linalg.norm(direction)

    @property
    def axis(self) -> int:
        """Coordinate index of a standing wave's axis (its direction must lie along one)."""
        return int(np.argmax(np.abs(self.direction)))

    def values(self, points, omega) -> np.ndarray:
        """Complex128 p_b (...) at points (..., d) [m] and angular frequency omega [rad/s]."""
        points = np.asarray(points, dtype=np.float64)
        k = omega / self.sound_speed
        if self.kind == 'plane_wave':
            return self.amplitude * np.exp(-1j * k * (points @ self.unit_direction[:points.shape[-1]]))
        if self.kind == 'standing_wave':
            return 2.0 * self.amplitude * np.cos(k * (points[..., self.axis] - self.plane_position)) + 0j
        return (np.asarray(self.real_part(points), dtype=np.float64)
                + 1j * np.asarray(self.imaginary_part(points), dtype=np.float64))

    def gradients(self, points, omega) -> np.ndarray:
        """Complex128 grad p_b (..., d) at points (..., d) [m] (an expression: central differences of gradient_step)."""
        points = np.asarray(points, dtype=np.float64)
        dimensions = points.shape[-1]
        k = omega / self.sound_speed
        if self.kind == 'plane_wave':
            direction = self.unit_direction[:dimensions]
            return (-1j * k * self.values(points, omega))[..., None] * direction
        result = np.zeros(points.shape, dtype=np.complex128)
        if self.kind == 'standing_wave':
            result[..., self.axis] = -2.0 * self.amplitude * k * np.sin(k * (points[..., self.axis]
                                                                             - self.plane_position))
            return result
        step = self.gradient_step
        for axis in range(dimensions):
            shift = np.zeros(dimensions)
            shift[axis] = step
            result[..., axis] = (self.values(points + shift, omega) - self.values(points - shift, omega)) / (2.0 * step)
        return result

    def text(self) -> str:
        if self.kind == 'expression':
            return 'background pressure expression'
        return (f'background {self.kind.replace("_", " ")} |p0| {abs(self.amplitude):.4g} Pa along '
                f'{tuple(round(float(value), 6) for value in self.unit_direction)}, c_b {self.sound_speed:.6g} m/s')


# ---- runtime records ----
@dataclass(frozen=True, eq=False)
class AcousticBoundary:
    """One boundary feature: target = outer face name or int64 mesh entities (2D edges / 3D faces), kind
    (BOUNDARY_KINDS), value (pressure: complex or f(points (k, d)) -> (k,); motion: complex or f(points (..., d)) ->
    (...)), power (motion: 0 acceleration, 1 velocity, 2 displacement), impedance [Pa s/m] (None: rho c)."""
    target: object
    kind: str
    value: object = 0.0
    power: int = 0
    impedance: object = None

    def __post_init__(self):
        if self.kind not in BOUNDARY_KINDS:
            raise ValueError(f"boundary kind must be one of {', '.join(BOUNDARY_KINDS)}, got {self.kind!r}")
        if self.impedance is not None and not float(self.impedance) > 0.0:
            raise ValueError(f'impedance must be > 0 Pa s/m, got {self.impedance!r}')


@dataclass(frozen=True, eq=False)
class FrequencyProbe:
    """Point probe: point [m], containing mesh element, reference coordinates, its HarmonicProbeRecord (p, p_s) and the
    full-numbering dofs / Lagrange basis weights of the element at the point (p = values[dofs] @ weights)."""
    point: np.ndarray
    element: int
    reference: np.ndarray
    record: HarmonicProbeRecord
    dofs: np.ndarray = None
    weights: np.ndarray = None


@dataclass(frozen=True, eq=False)
class SurfaceMonitor:
    """Surface mean / weighted integral of p over mesh entities (2D edges, 3D faces) chosen on one side: unit / weight
    loads (full numbering), measure and weight integral (whole rings / per metre of depth in planar 2D), records."""
    name: str
    side: str
    entities: np.ndarray
    weight: object
    unit_load: np.ndarray
    weight_load: np.ndarray
    area: float
    weight_integral: float
    mean_record: HarmonicProbeRecord
    weighted_record: HarmonicProbeRecord
    quadrature: object = None   # (points (k, q, d), unit weights (k, q), weight x unit weights) of p_b, None: loads


@dataclass(frozen=True, eq=False)
class _BackgroundTerms:
    """Frequency-independent parts of the background source: a_b, (K_s, M_s, full ids) of the contrast / obstacle
    elements (None: none), (face, window entities) of the non-PML outer faces, full dof points and their window mask."""
    background_admittance: float
    operators: object
    face_entities: tuple
    dof_points: np.ndarray
    in_window: np.ndarray
    n_source_elements: int


# ---- dimension-free physics ----
class AcousticHarmonicBase(PhysicsSolver):
    """Frequency-domain pressure acoustics on an element mesh (module header): boundaries, obstacles, point sources,
    background, probes and surface monitors are recorded by the definition callbacks; the system is built at finalize
    (and after a material change) and solved once per frequency_response frequency. Subclasses supply the mesh-kind
    pieces (_new_system, _window_elements, _make_pml, _window_face_entities, _entity_load, ...)."""

    name = 'pressure_acoustics'
    type_label = 'pressure_acoustics'
    material_schema = dict(ACOUSTIC_HARMONIC_SCHEMA)
    coordinate_grids = ('cartesian',)
    conformal_locator = None
    grid_kind = 'triangles'
    dimensions = 2
    faces = FACES_2D

    def __init__(self, domain, order=2, linear_solver=None, *, axisymmetric=False, pml_faces=(),
                 pml_reflection=DEFAULT_PML_REFLECTION, pml_frequency=None):
        if order not in (1, 2):
            raise ValueError(f'{self.type_label}: element order must be 1 (linear) or 2 (quadratic), got {order!r}')
        pml_faces = tuple(pml_faces)
        unknown = [face for face in pml_faces if face not in self.faces]
        if unknown:
            raise ValueError(f"{self.type_label}: PML faces must be among {', '.join(self.faces)}, got {unknown}")
        if axisymmetric and 'x_lo' in pml_faces:
            raise ValueError(f'{self.type_label}: an axisymmetric PML cannot lie on x_lo (the axis)')
        self.order = int(order)
        self.linear_solver = linear_solver
        self._axisymmetric = bool(axisymmetric)
        self.pml_faces = pml_faces
        self.pml_reflection = float(pml_reflection)
        self.pml_frequency = None if pml_frequency is None else float(pml_frequency)
        self.finalize_notes = []
        self.prepare(domain)

    def _label(self) -> str:
        return f"{self.type_label} '{self.name}'"

    # --- PhysicsSolver contract ---------------------------------------------
    def prepare(self, domain) -> None:
        """Register rho / sound_speed, allocate the per-element state (clears boundaries, obstacles, sources, the
        background, probes and monitors); the system is built at finalize."""
        if getattr(domain, 'grid_kind', 'tensor') != self.grid_kind:
            raise ValueError(f'{self._label()}: {type(self).__name__} needs a {self.grid_kind} mesh, got '
                             f"{getattr(domain, 'grid_kind', 'tensor')}")
        self.domain = domain
        self.xp = domain.xp
        self.mesh = domain.mesh
        domain.materials.add_properties(self.material_schema)
        n_elements = self.mesh.n_elements
        self.full, self.element_map = self._dof_maps()
        self.fields = {name: self.xp.zeros(n_elements, dtype=DTYPE) for name in STATE_NAMES}
        self.boundaries = []
        self.obstacle = np.zeros(n_elements, dtype=bool)
        self.point_sources = []          # (point (d,), complex flow rate [m^3/s; planar m^2/s])
        self.background = None
        self.frequency_probes = []
        self.surface_monitors = []
        self.system = None
        self.pml = None
        self._system_dirty = True
        self._frequency_loads = []       # (complex full load, power): (j omega)^power x load per frequency
        self._impedance_terms = []       # (entities, damping 1 / Z): the background's -j omega p_b / Z loads
        self._background_terms = None
        self._active = self._window = None
        self._rho = self._a = self._m = self._impedance_z = None
        self._active_dofs = None
        self._scattered = self._total = self._background_nodal = None
        self._velocity = None
        self._omega = None
        self.last_frequency = None
        self.n_frequency_solves = 0
        self.assembly_seconds = 0.0      # host wall time of _build_system (PML, conditions, assembly, background terms)
        self.reduced_sweep = None        # info of plan_frequency_response, or None
        self._field_policy = None        # fem.reduced_outputs field policy of that plan
        self._unpublished_omega = None   # [rad/s] solved without showing its fields (field policy)
        self._background_cache = None    # (omega, background load) of the last _background_at
        self._background_nodal_cache = None   # (omega, p_b nodal) of the last _background_nodal_at
        self._background_device = None   # (omega, cupy p_b nodal) of the last device publish
        self._device_centroid_maps = None   # (system, cupy active dofs, element dofs, centroid basis, active elements)

    def finalize(self) -> None:
        """Build and assemble the system: a fluid element without rho / sound_speed or a PML without a reference
        frequency fails before the study."""
        self._build_system()
        system = self.system
        parts = [f'P{self.order} {self.grid_kind}, {system.pencil.free.size:,} unknowns']
        if self.pml is not None:
            parts.append(f'PML {self.pml.text()} at {self.pml_frequency:.6g} Hz')
        if self.background is not None:
            parts.append(f'{self.background.text()}, {self._background_terms.n_source_elements:,} source elements')
        self.finalize_notes = ['; '.join(parts)]

    def materials_changed(self, names) -> None:
        if set(names) & set(self.material_schema):
            self._system_dirty = True

    def step(self, t: float = 0.0, dt: float = math.inf) -> None:
        """Nothing: the frequency_response study drives the solves (solve_frequency_response)."""

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        state = dict(self.fields)
        for array_name in self.material_schema:
            state[array_name] = getattr(self.domain.materials, array_name)
        return state

    def set_conformal_boundaries(self, locator) -> None:
        """solver_defaults.conformal_boundaries: nothing to do (the elements follow the drawn outlines)."""

    @property
    def axisymmetric(self) -> bool:
        return self._axisymmetric

    @property
    def ring_angle(self) -> float:
        return 2.0 * math.pi

    def _ring(self) -> float:
        return RING_ANGLE if self._axisymmetric else 1.0

    def work_counters(self) -> dict:
        """Frequency solves, assemblies and assembly_seconds (system build wall time; factorisations are the 'linear'
        role's record)."""
        return {'frequency_solves': self.n_frequency_solves,
                'assemblies': 0 if self.system is None else self.system.n_assemblies,
                'assembly_seconds': self.assembly_seconds}

    # --- definition API -----------------------------------------------------
    def add_boundary(self, target, kind, value=0.0, power=0, impedance=None) -> None:
        """Boundary condition on an outer face (name) or mesh entities (module header kinds)."""
        if isinstance(target, str):
            if target not in self.faces:
                raise ValueError(f"{self._label()}: unknown face {target!r}; expected one of {', '.join(self.faces)}")
            if target in self.pml_faces:
                raise ValueError(f'{self._label()}: {target} carries a PML (its inner side), so it takes no boundary '
                                 'condition: turn the PML off there or remove the feature')
        else:
            target = np.unique(np.asarray(target, dtype=np.int64))
        self.boundaries.append(AcousticBoundary(target, kind, value, int(power), impedance))
        self._system_dirty = True

    def add_obstacle(self, mask) -> None:
        """Make the masked elements a rigid obstacle (excluded: no pressure inside, a sound-hard outline)."""
        mask = np.asarray(to_host(mask), dtype=bool).reshape(-1)
        if mask.size != self.mesh.n_elements:
            raise ValueError(f'{self._label()}: obstacle mask has {mask.size} values, expected one per element '
                             f'({self.mesh.n_elements})')
        self.obstacle |= mask
        self._system_dirty = True

    def add_point_source(self, point, flow_rate) -> None:
        """Volume source of complex flow rate [m^3/s; planar 2D m^2/s per metre; axisymmetric the whole ring] at a
        point [m] (an inactive element there fails at the build)."""
        self.point_sources.append((np.asarray(point, dtype=np.float64).reshape(self.dimensions), complex(flow_rate)))
        self._system_dirty = True

    def set_background(self, background) -> None:
        """Scattered-field background pressure (BackgroundPressure; None = total-field formulation)."""
        if background is not None and background.kind == 'expression' and not background.gradient_step > 0.0:
            bounds = np.asarray(self.mesh.window, dtype=np.float64).reshape(2, -1)
            background = replace(background, gradient_step=GRADIENT_STEP_RTOL * float(np.max(bounds[1] - bounds[0])))
        if background is not None and background.kind != 'expression' and len(background.direction) != self.dimensions:
            raise ValueError(f'{self._label()}: the background direction needs {self.dimensions} components, got '
                             f'{background.direction!r}')
        self.background = background
        self._system_dirty = True

    def add_frequency_probe(self, point, label='') -> HarmonicProbeRecord:
        """Record the complex total p and scattered p_s [Pa] at a point [m] at every solved frequency."""
        point = np.asarray(point, dtype=np.float64).reshape(self.dimensions)
        element, reference = self._locate(point)
        record = HarmonicProbeRecord(('p', 'p_s'), unit='Pa', label=label or 'Pressure')
        dofs, weights = self._point_weights(element, reference)   # once: each record is then one dot product
        self.frequency_probes.append(FrequencyProbe(point, element, reference, record, dofs, weights))
        return record

    def add_surface_monitor(self, name, faces=(), entities=None, member=None, side='all', weight=1.0) -> SurfaceMonitor:
        """Mean and weight-weighted integral of p over outer faces (their window side) and object entities, on the
        entities whose outward normal (out of the member elements; else out of their first element) points to `side`;
        weight: a number or a function of points (..., d)."""
        if side not in SIDES:
            raise ValueError(f"{self._label()}: side must be one of {', '.join(SIDES)}, got {side!r}")
        axis, sign = (None, 0.0) if side == 'all' else ('xyz'.index(side[1]), 1.0 if side[0] == '+' else -1.0)
        if axis is not None and axis >= self.dimensions:
            raise ValueError(f'{self._label()}: side {side} needs a 3D model')
        parts = []
        for face in faces:
            if axis is None or sign * FACE_NORMALS[face][axis] > 0.0:
                parts.append(self._window_face_entities(face))
        if entities is not None and len(entities):
            entities = np.unique(np.asarray(entities, dtype=np.int64))
            if axis is not None:
                normals = self._entity_normals(entities, None if member is None
                                               else np.asarray(to_host(member), dtype=bool).reshape(-1))
                entities = entities[sign * normals[:, axis] > 1e-9]
            parts.append(entities)
        selected = np.unique(np.concatenate(parts)).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)
        if not selected.size:
            raise ValueError(f"{self._label()}: surface '{name}' selects no boundary on side {side}")
        unit_load = np.real(self._entity_load(selected, 1.0))
        weight_load = self._entity_load(selected, _as_entity_callable(weight))
        quadrature = self._entity_quadrature(selected)
        if quadrature is not None:   # B6: the background integrals of _record become dot products
            points, unit_weights = quadrature
            quadrature = (points, unit_weights,
                          unit_weights * _weight_values(_as_entity_callable(weight), points, selected))
        ring = self._ring()
        integral_unit = 'Pa m' if self.dimensions == 2 and not self._axisymmetric else 'Pa m^2'
        monitor = SurfaceMonitor(
            name=name, side=side, entities=selected, weight=weight, unit_load=unit_load, weight_load=weight_load,
            area=ring * float(unit_load.sum()), weight_integral=ring * complex(weight_load.sum()).real,
            mean_record=HarmonicProbeRecord('p_mean', unit='Pa', label=f'Mean pressure on {name}'),
            weighted_record=HarmonicProbeRecord('p_weighted', unit=integral_unit,
                                                label=f'Weighted pressure integral on {name}'),
            quadrature=quadrature)
        self.surface_monitors.append(monitor)
        return monitor

    def surface_result(self, monitor) -> dict:
        """'xy' result of a SurfaceMonitor: series p_mean_re / _im / _abs / _phase [Pa] and p_weighted_* (Pa m planar
        2D, else Pa m^2) versus frequency; extras area, weight_integral, side."""
        result = harmonic_probe_result(monitor.mean_record)
        weighted = harmonic_probe_result(monitor.weighted_record)
        result['series'].update(weighted['series'])
        result['series_units'].update(weighted['series_units'])
        result['components'] = list(result['components']) + list(weighted['components'])
        result.update(area=monitor.area, weight_integral=monitor.weight_integral, side=monitor.side)
        return result

    # --- frequency_response study hooks ---------------------------------------
    def reset_frequency_response(self):
        """Forget the probe and surface records of an earlier frequency_response run (the study calls it first)."""
        for probe in self.frequency_probes:
            probe.record.clear()
        for monitor in self.surface_monitors:
            monitor.mean_record.clear()
            monitor.weighted_record.clear()
        self.reduced_sweep, self._field_policy, self._unpublished_omega = None, None, None
        if self.system is not None:
            self.system.clear_reduced()

    def plan_frequency_response(self, frequencies, options):
        """ReducedSweepSolver hook (after reset_frequency_response): options {method, tolerance, moments, max_points,
        modal_count, fields}; 'direct' -> None (only the field policy, R6), else the system's reduced model over
        frequencies [Hz] with the source loads as (j omega)^power terms and the background as a non-polynomial load +
        held offsets -> its info dict."""
        from ...fem.reduced_harmonic import HarmonicLoad, PowerCoefficient
        from ...solvers.batched_direct import BatchedDirectSolver
        method = options.get('method', 'direct')
        batched = method == 'direct' and isinstance(self.linear_solver, BatchedDirectSolver)
        if method == 'direct' and not batched:
            self._field_policy = field_policy(options, frequencies)
            return None
        frequencies = np.asarray(frequencies, dtype=np.float64).ravel()
        if not (frequencies.size and np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)):
            raise ValueError(f'{self._label()}: a reduced or batched frequency sweep needs frequencies > 0 Hz')
        if self._system_dirty or self.system is None:
            self._build_system()
        system = self.system
        system.set_frequency_load(None)   # the model / plan carries them (extra loads / held offsets per omega)
        system.set_held_offsets(None)
        extra_loads = [HarmonicLoad(vector, PowerCoefficient((1j) ** power, power))
                       for vector, power in self._frequency_loads]
        held_offsets_of = None
        if self._background_terms is not None:
            extra_loads.append(HarmonicLoad(vector_of=lambda omega: self._background_at(omega)[1]))
            held_offsets_of = lambda omega: -self._background_nodal_at(omega)   # noqa: E731 (p_s = given - p_b)
        if batched:   # solvers.batched_direct: the same loads, solved in batches of frequencies (None: no reduced model)
            system.plan_batched(2.0 * math.pi * frequencies, extra_loads=extra_loads, held_offsets_of=held_offsets_of)
            self._field_policy = field_policy(options, frequencies)
            return None
        info = system.plan_reduced(
            2.0 * math.pi * frequencies, method, tolerance=options.get('tolerance', 1e-6),
            moments=options.get('moments', 4), max_points=options.get('max_points', 40),
            modal_count=options.get('modal_count', 20), extra_loads=extra_loads, held_offsets_of=held_offsets_of)
        self._field_policy = field_policy(options, frequencies)
        self.reduced_sweep = info
        return info

    def solve_frequency_response(self, frequency):
        """frequency_response study hook: the driven pressure field at frequency [Hz] > 0 (one complex factor of the
        pencil assembled once, plus the per-frequency source / background terms; or the plan_frequency_response model);
        records probes and surfaces and shows the solution where the field policy asks."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'{self._label()}: a frequency response needs a frequency > 0 Hz, got {frequency!r}')
        if self._system_dirty or self.system is None:
            self._build_system()
        system = self.system
        omega = 2.0 * math.pi * frequency
        fields = wants_fields(self._field_policy, frequency)
        if system.reduced_model is not None or system.batched_plan is not None:   # loads per omega in the plan
            system.solve(omega, fields=fields)
        else:
            load = np.zeros(self.full.n_dofs, dtype=np.complex128)
            for vector, power in self._frequency_loads:
                load += (1j * omega) ** power * vector
            offsets = None
            if self._background_terms is not None:
                background_nodal, background_load = self._background_at(omega)
                load += background_load
                offsets = -background_nodal
            system.set_frequency_load(load)
            system.set_held_offsets(offsets)
            system.solve(omega)
        self.n_frequency_solves += 1
        if fields:
            self._show_solution(frequency, omega)
        self._unpublished_omega = None if fields else omega
        self._record(frequency, omega)

    def publish_frequency_response(self):
        """ReducedSweepSolver hook: show the last solved frequency when the field policy skipped it (direct values, or
        V y of a fields-free reduced solve)."""
        omega, self._unpublished_omega = self._unpublished_omega, None
        if omega is None or self.system is None:
            return
        self.system.publish_fields()
        self._show_solution(omega / (2.0 * math.pi), omega)

    def _show_solution(self, frequency, omega) -> None:
        """The system's values as the shown scattered field at frequency [Hz] (+ p_b) and the published state."""
        self._scattered = self.system.values.copy()
        self._background_nodal = (self._background_nodal_at(omega) if self._background_terms is not None
                                  else np.zeros(self.full.n_dofs, dtype=np.complex128))
        self._omega = omega
        self.last_frequency = frequency
        self._publish()

    def _background_at(self, omega) -> tuple:
        """(p_b nodal, background load) at omega, the load cached for the last omega (nodal: _background_nodal_at)."""
        omega = float(omega)
        cached = self._background_cache
        if cached is None or cached[0] != omega:
            cached = self._background_cache = (omega, self._background_load(omega))
        return self._background_nodal_at(omega), cached[1]

    def _background_nodal_at(self, omega) -> np.ndarray:
        """p_b at the window dofs (0 elsewhere) at omega, cached for the last omega: readers of the nodal values alone
        (held offsets, the shown field) never assemble the load."""
        omega = float(omega)
        cached = self._background_nodal_cache
        if cached is None or cached[0] != omega:
            terms = self._background_terms
            cached = self._background_nodal_cache = (
                omega, np.where(terms.in_window, self.background.values(terms.dof_points, omega), 0.0))
        return cached[1]

    # --- build ---------------------------------------------------------------
    def _build_system(self) -> None:
        """Coefficients from rho / sound_speed, the PML, conditions, frequency loads, the assembly and the background
        terms (module header); timed into assembly_seconds."""
        started = time.perf_counter()
        mesh = self.mesh
        materials = self.domain.materials
        rho, speed = _host(materials.rho), _host(materials.sound_speed)
        window = self._window_elements()
        fluid = np.isfinite(rho) & np.isfinite(speed) & (rho > 0.0) & (speed > 0.0)
        pml = None
        if self.pml_faces:
            layer = ~window & ~self.obstacle & fluid
            if not layer.any():
                raise ValueError(f'{self._label()}: the PML layers hold no fluid element (rho and sound_speed > 0)')
            if not (self.pml_frequency is not None and self.pml_frequency > 0.0):
                raise ValueError(f'{self._label()}: a PML needs a reference frequency > 0 Hz (setting pml_frequency, '
                                 'else the lowest frequency of the frequency_response study)')
            pml = self._make_pml(float(speed[layer].max()), 2.0 * math.pi * self.pml_frequency)
        region = np.ones(mesh.n_elements, dtype=bool) if pml is not None else window
        active = region & ~self.obstacle
        if not active.any():
            raise ValueError(f'{self._label()}: no fluid element is left (every element is a rigid obstacle)')
        missing = active & ~fluid
        if missing.any():
            raise ValueError(f'{self._label()}: {int(missing.sum()):,} of the {int(active.sum()):,} fluid elements '
                             'have no density rho > 0 or sound_speed > 0: give every fluid region a material with both '
                             '(air 1.2 kg/m^3, 343 m/s; water 998, 1480), or select solid regions as rigid_obstacle')
        safe_rho, safe_speed = np.where(active, rho, 1.0), np.where(active, speed, 1.0)
        self._rho, self._active, self._window = rho, active, window
        self._a = np.where(active, 1.0 / safe_rho, 0.0)
        self._m = np.where(active, 1.0 / (safe_rho * safe_speed ** 2), 0.0)
        self._impedance_z = np.where(active, safe_rho * safe_speed, 0.0)
        system = self._new_system(pml)
        system.set_coefficients(a=self._a, mass=self._m)
        system.set_active(active)
        self.system, self.pml = system, pml
        self._frequency_loads, self._impedance_terms = [], []
        self._apply_boundaries(system)
        for point, flow_rate in self.point_sources:
            self._frequency_loads.append((self._point_load(point, flow_rate), 1))
        system.assemble()
        self._active_dofs = np.zeros(self.full.n_dofs, dtype=bool)
        self._active_dofs[system.global_ids] = True
        self._background_terms = self._background_preparation() if self.background is not None else None
        self._background_cache = self._background_nodal_cache = self._background_device = None
        self._system_dirty = False
        self.assembly_seconds += time.perf_counter() - started

    def _apply_boundaries(self, system) -> None:
        """Every boundary in feature order on the window entities of its faces (the last feature per face only) or on
        its object entities; impedances keep their damping for the background term, motions their loads."""
        last_on_face = {boundary.target: index for index, boundary in enumerate(self.boundaries)
                        if isinstance(boundary.target, str)}
        for index, boundary in enumerate(self.boundaries):
            if isinstance(boundary.target, str):
                if last_on_face[boundary.target] != index:
                    continue
                entities = self._window_face_entities(boundary.target)
            else:
                entities = boundary.target
            if not len(entities) or boundary.kind == 'hard':
                continue
            if boundary.kind == 'soft':
                condition = HarmonicCondition('dirichlet', 0.0)
            elif boundary.kind == 'pressure':
                condition = HarmonicCondition('dirichlet', boundary.value)
            elif boundary.kind == 'impedance':
                damping = self._impedance_damping(entities, boundary.impedance)
                condition = HarmonicCondition('robin', damping=damping)
                self._impedance_terms.append((entities, damping))
            else:   # motion: natural operator, (j omega)^power x the inward normal amplitude as a load
                self._frequency_loads.append((self._entity_load(entities, _as_entity_callable(boundary.value)),
                                              boundary.power))
                continue
            system.add_boundary_condition(entities, condition)

    def _impedance_damping(self, entities, impedance):
        """1 / Z: a number for a given impedance, else per mesh entity 1 / (rho c) of its adjacent fluid element."""
        if impedance is not None:
            return 1.0 / float(impedance)
        pairs = self._entity_pairs(entities)
        first, second = pairs[:, 0], pairs[:, 1]
        second_active = (second >= 0) & self._active[np.maximum(second, 0)]
        element = np.where(self._active[first], first, np.where(second_active, second, -1))
        damping = np.zeros(self._n_entities())
        touching = element >= 0
        damping[entities[touching]] = 1.0 / self._impedance_z[element[touching]]
        return damping

    def _background_preparation(self) -> _BackgroundTerms:
        """Contrast / obstacle source operators, the non-PML outer faces and the dof points of the background."""
        background = self.background
        admittance = 1.0 / background.density
        compliance = 1.0 / (background.density * background.sound_speed ** 2)
        fluid_window = self._active & self._window
        stiffness_difference = np.where(fluid_window, self._a - admittance, 0.0)
        mass_difference = np.where(fluid_window, self._m - compliance, 0.0)
        stiffness_difference[np.abs(stiffness_difference) <= CONTRAST_RTOL * admittance] = 0.0
        mass_difference[np.abs(mass_difference) <= CONTRAST_RTOL * compliance] = 0.0
        void_window = self.obstacle & self._window
        stiffness_coefficient = -stiffness_difference + np.where(void_window, admittance, 0.0)
        mass_coefficient = -mass_difference + np.where(void_window, compliance, 0.0)
        source_elements = (stiffness_coefficient != 0.0) | (mass_coefficient != 0.0)
        operators = (self._source_operators(stiffness_coefficient, mass_coefficient, source_elements)
                     if source_elements.any() else None)
        faces = tuple(face for face in self.faces
                      if face not in self.pml_faces and not (self._axisymmetric and face == 'x_lo'))
        face_entities = tuple((face, self._window_face_entities(face)) for face in faces)
        dof_points = self.full.dof_points()
        return _BackgroundTerms(admittance, operators, face_entities, dof_points, self._in_window(dof_points),
                                int(source_elements.sum()))

    def _background_load(self, omega) -> np.ndarray:
        """The scattered-field source load of the background at omega (module header)."""
        terms, background = self._background_terms, self.background
        nodal = self._background_nodal_at(omega)
        load = np.zeros(self.full.n_dofs, dtype=np.complex128)
        if terms.operators is not None:
            stiffness, mass, ids = terms.operators
            local = nodal[ids]
            load[ids] += stiffness @ local - omega * omega * (mass @ local)
        for face, entities in terms.face_entities:
            if not len(entities):
                continue
            normal = np.asarray(FACE_NORMALS[face][:self.dimensions])

            def flux(points, kept, normal=normal):
                return -terms.background_admittance * (background.gradients(points, omega) @ normal)
            load += self._entity_load(entities, flux)
        for entities, damping in self._impedance_terms:
            def admittance_load(points, kept, damping=damping):
                return -1j * omega * _entity_values(damping, kept) * background.values(points, omega)
            load += self._entity_load(entities, admittance_load)
        return load

    # --- outputs ---------------------------------------------------------------
    def _publish(self) -> None:
        """Per-element centroid state of the total field (p_s + p_b on the fluid dofs): on the device from the system's
        device_values after a direct cuDSS solve on a cupy domain (R6), else on the host; the host total and the
        velocities are formed on read (_host_total, _element_velocity)."""
        self._total = None
        self._velocity = None
        xp = self.xp
        device = self.system.device_values() if xp is not np else None
        if device is not None:
            centroid, module = self._device_centroids(device), xp
        else:
            centroid, module = np.where(self._active, self._centroid_values(self._host_total()), 0.0), np
        for name, values in (('p_abs', module.abs(centroid)), ('p_re', centroid.real), ('p_im', centroid.imag)):
            xp.copyto(self.fields[name], xp.asarray(values, dtype=DTYPE))

    def _host_total(self):
        """Host complex total field p_s + p_b on the fluid dofs (0 elsewhere) of the shown solution, formed on first
        read; None before a solve."""
        if self._total is None and self._scattered is not None:
            self._total = np.where(self._active_dofs, self._scattered + self._background_nodal, 0.0)
        return self._total

    def _device_centroids(self, scattered):
        """cupy complex centroid values of p_s (cupy full values) + p_b, 0 off the fluid: element maps uploaded once
        per system, p_b nodal once per frequency."""
        xp = self.xp
        maps = self._device_centroid_maps
        if maps is None or maps[0] is not self.system:
            maps = self._device_centroid_maps = (self.system, xp.asarray(self._active_dofs),
                                                 xp.asarray(self.full.element_dofs),
                                                 xp.asarray(self._centroid_basis(), dtype=np.float64),
                                                 xp.asarray(self._active))
        _, active_dofs, element_dofs, basis, active = maps
        total = scattered
        if self._background_terms is not None:
            cached = self._background_device
            if cached is None or cached[0] != self._omega:
                cached = self._background_device = (self._omega, xp.asarray(self._background_nodal))
            total = scattered + cached[1]
        total = xp.where(active_dofs, total, 0.0)
        return xp.where(active, total[element_dofs] @ basis, 0.0)

    def _record(self, frequency, omega) -> None:
        """Probe (exact p_b at the point) and surface records of the last solve (linear outputs of the system values:
        from L V after a fields-free reduced solve)."""
        background = self.background
        output = self.system.linear_output
        for index, probe in enumerate(self.frequency_probes):
            if not self._active[probe.element]:
                probe.record.record(frequency, {'p': 0j, 'p_s': 0j})
                continue
            scattered = complex(output(('probe', index), lambda values, probe=probe: self._point_value(values, probe)))
            incident = 0j
            if background is not None and bool(self._in_window(probe.point[None])[0]):
                incident = complex(background.values(probe.point[None], omega)[0])
            probe.record.record(frequency, {'p': scattered + incident, 'p_s': scattered})
        ring = self._ring()
        for monitor in self.surface_monitors:
            mean = complex(output(('surface_mean', monitor.name), lambda values, load=monitor.unit_load: load @ values))
            weighted = complex(output(('surface_weighted', monitor.name),
                                      lambda values, load=monitor.weight_load: load @ values))
            if background is not None:
                background_mean, background_weighted = self._surface_background(monitor, omega)
                mean += background_mean
                weighted += background_weighted
            measure = monitor.area / ring
            monitor.mean_record.record(frequency, mean / measure if measure > 0.0 else 0j)
            monitor.weighted_record.record(frequency, ring * weighted)

    def _surface_background(self, monitor, omega) -> tuple:
        """(int p_b, int weight p_b) over a monitor's entities at omega (no ring factor): dot products on its
        precomputed rule, else sums of assembled entity loads."""
        background = self.background
        if monitor.quadrature is not None:
            points, unit_weights, weighted_weights = monitor.quadrature
            values = background.values(points, omega)
            return complex(np.sum(unit_weights * values)), complex(np.sum(weighted_weights * values))
        weight = _as_entity_callable(monitor.weight)
        mean = complex(np.sum(self._entity_load(
            monitor.entities, lambda points, kept: background.values(points, omega))))
        weighted = complex(np.sum(self._entity_load(
            monitor.entities, lambda points, kept: _weight_values(weight, points, kept)
            * background.values(points, omega))))
        return mean, weighted

    def _element_velocity(self):
        """Complex (n_elements, d) particle velocity j grad p / (omega rho) at the element centroids (0 off the fluid);
        None before a solve."""
        if self._host_total() is None:
            return None
        if self._velocity is None:
            gradients = self._centroid_gradients(self._total)
            scale = np.where(self._active, 1.0 / (self._omega * np.where(self._active, self._rho, 1.0)), 0.0)
            self._velocity = np.where(self._active[:, None], 1j * gradients * scale[:, None], 0.0)
        return self._velocity

    def _component_field(self, component) -> tuple:
        """(DofMap, host float64 coefficients) of a display component: pressures nodal, velocities and materials per
        element (zeros before a solve)."""
        if component in NODAL_FIELDS:
            if self._host_total() is None:
                return self.full, np.zeros(self.full.n_dofs)
            if component == 'spl':
                amplitude = np.abs(self._total)
                level = 20.0 * np.log10(np.maximum(amplitude, 1e-300) / (math.sqrt(2.0) * REFERENCE_PRESSURE))
                return self.full, np.maximum(level, SPL_FLOOR)
            values = self._scattered if component.startswith('p_s_') else self._total
            part = component.rsplit('_', 1)[1]
            return self.full, {'re': values.real, 'im': values.imag, 'abs': np.abs(values)}[part]
        if component in VELOCITY_FIELDS:
            axis = None if component == 'v_abs' else 'xyz'.index(component[2])
            if axis is not None and axis >= self.dimensions:
                raise KeyError(f"{self._label()}: field '{component}' exists in 3D models only")
            velocity = self._element_velocity()
            if velocity is None:
                return self.element_map, np.zeros(self.mesh.n_elements)
            if axis is None:
                return self.element_map, np.sqrt(np.sum(np.abs(velocity) ** 2, axis=1))
            values = velocity[:, axis]
            return self.element_map, values.real if component.endswith('_re') else values.imag
        if component in self.material_schema:
            return self.element_map, _host(getattr(self.domain.materials, component))
        raise KeyError(f"{self._label()} has no field '{component}'")

    def _field_infos(self) -> dict:
        """The display FieldInfos of this instance (v_z_* in 3D only)."""
        return {name: info for name, info in FIELD_INFOS.items()
                if self.dimensions == 3 or name not in THREE_D_ONLY_FIELDS}

    def _time_label(self) -> str:
        if self.last_frequency is None:
            return 'not solved'
        return f'f = {self.last_frequency:.6g} Hz, solve {self.n_frequency_solves}'

    def field_dofs(self, name):
        """fem DofMap of state field `name` (the transfer contract): every one is per element (P0)."""
        if name not in self.state():
            raise ValueError(f"{self._label()} has no state field '{name}'")
        return self.element_map

    # --- subclass hooks ----------------------------------------------------------
    def _dof_maps(self) -> tuple:
        raise NotImplementedError

    def _new_system(self, pml):
        raise NotImplementedError

    def _window_elements(self) -> np.ndarray:
        raise NotImplementedError

    def _make_pml(self, wave_speed, omega_reference):
        raise NotImplementedError

    def _window_face_entities(self, face) -> np.ndarray:
        raise NotImplementedError

    def _entity_load(self, entities, value) -> np.ndarray:
        raise NotImplementedError

    def _entity_quadrature(self, entities):
        """(points (k, q, d), weights (k, q)) with sum_i _entity_load(entities, f)_i = sum weights f(points), or None
        (surface monitors then sum assembled loads)."""
        return None

    def _entity_pairs(self, entities) -> np.ndarray:
        raise NotImplementedError

    def _n_entities(self) -> int:
        raise NotImplementedError

    def _entity_normals(self, entities, member) -> np.ndarray:
        raise NotImplementedError

    def _source_operators(self, stiffness_coefficient, mass_coefficient, elements) -> tuple:
        raise NotImplementedError

    def _in_window(self, points) -> np.ndarray:
        raise NotImplementedError

    def _locate(self, point) -> tuple:
        raise NotImplementedError

    def _point_load(self, point, value) -> np.ndarray:
        raise NotImplementedError

    def _point_value(self, coefficients, probe):
        """Value of full-numbering coefficients at a probe's point (its precomputed element dofs and basis weights)."""
        return np.asarray(coefficients)[probe.dofs] @ probe.weights

    def _point_weights(self, element, reference) -> tuple:
        """(int64 full-numbering dofs of the element, float64 Lagrange basis values at the reference point)."""
        raise NotImplementedError

    def _centroid_values(self, coefficients) -> np.ndarray:
        raise NotImplementedError

    def _centroid_basis(self) -> np.ndarray:
        """(n,) Lagrange basis values of self.full at the reference centroid (the device publish)."""
        raise NotImplementedError

    def _centroid_gradients(self, coefficients) -> np.ndarray:
        raise NotImplementedError


def _weight_values(weight, points, kept):
    """Weight of a surface monitor at points (k, q, d): a number, or its edge / face coefficient form."""
    return weight(points, kept) if callable(weight) else weight


def oriented_normals(normals, midpoints, pairs, member, centroids) -> np.ndarray:
    """Unit normals (k, d) pointing out of the member element of each entity (pairs (k, 2), second -1 outside; no
    member mask or no member side: out of the first element)."""
    first, second = pairs[:, 0], pairs[:, 1]
    inside = first
    if member is not None:
        second_member = (second >= 0) & member[np.maximum(second, 0)]
        inside = np.where(~member[first] & second_member, second, first)
    toward = centroids[inside] - midpoints
    flip = np.sum(toward * normals, axis=1) > 0.0
    normals = np.where(flip[:, None], -normals, normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


# ---- triangles ----
class PressureAcousticsFEMPhysics(AcousticHarmonicBase):
    """pressure_acoustics on a fem.domain.TriangleDomain (module header): planar or axisymmetric (x = r), the PML on
    the mesh's layer ring (fem.pml.pml_ring), outer faces on their window edges (fem.diffusion.window_face_edges)."""

    grid_kind = 'triangles'
    dimensions = 2
    faces = FACES_2D

    @property
    def geometry_order(self) -> int:
        return 1 if self.order == 1 else 2

    def renderer(self):
        return FEMFieldRenderer(self, self.name, self._field_infos(), self._time_label)

    def fem_field(self, component) -> tuple:
        """(Lagrange DofMap, host coefficients) of a display component (the Session's triangle reads)."""
        return self._component_field(component)

    def _dof_maps(self) -> tuple:
        return (lagrange_dofs(self.mesh, self.order, geometry_order=self.geometry_order),
                lagrange_dofs(self.mesh, 0, geometry_order=self.geometry_order))

    def _new_system(self, pml):
        return ScalarHarmonicSystem(self.mesh, self.order, self.linear_solver, axisymmetric=self._axisymmetric, pml=pml,
                                    pml_outer='dirichlet', symmetric=True, label=self._label)

    def _window_elements(self) -> np.ndarray:
        return self.mesh.pml_faces == 0

    def _make_pml(self, wave_speed, omega_reference):
        try:
            return pml_ring(self.mesh, wave_speed, omega_reference, self.pml_reflection, faces=self.pml_faces)
        except ValueError as exc:
            raise ValueError(f'{self._label()}: {exc} (another physics on this mesh may ring other faces)') from None

    def _window_face_entities(self, face) -> np.ndarray:
        return window_face_edges(self.mesh, face)

    def _entity_load(self, entities, value) -> np.ndarray:
        return assemble_edge_load(self.full, np.asarray(entities, dtype=np.int64), value,
                                  axisymmetric=self._axisymmetric)

    def _entity_quadrature(self, entities) -> tuple:
        """assemble_edge_load's default rule on self.full, read from its edge-rule cache (fem.assembly._edge_quadrature,
        so the rule cannot drift; the basis sums to 1): (points (k, q, 2), Gauss weight x |dx/dt| (x r axisymmetric));
        every edge has an element of the full map, so none is dropped."""
        _, _, _, weights, lengths, points = _edge_quadrature(self.full, np.asarray(entities, dtype=np.int64),
                                                             self.order + 2 + int(self._axisymmetric))
        line_weights = weights[None, :] * lengths
        if self._axisymmetric:
            line_weights = line_weights * points[..., 0]
        return points, line_weights

    def _entity_pairs(self, entities) -> np.ndarray:
        return self.mesh.edge_elements[np.asarray(entities, dtype=np.int64)]

    def _n_entities(self) -> int:
        return self.mesh.n_edges

    def _entity_normals(self, entities, member) -> np.ndarray:
        mesh = self.mesh
        ends = mesh.points[mesh.edges[entities]]          # (k, 2, 2)
        tangent = ends[:, 1] - ends[:, 0]
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        return oriented_normals(normals, ends.mean(axis=1), mesh.edge_elements[entities], member, mesh.centroids())

    def _source_operators(self, stiffness_coefficient, mass_coefficient, elements) -> tuple:
        dofs = lagrange_dofs(self.mesh, self.order, elements, geometry_order=self.geometry_order)
        stiffness = assemble_stiffness(dofs, stiffness_coefficient, axisymmetric=self._axisymmetric).tocsr()
        mass = assemble_mass(dofs, mass_coefficient, axisymmetric=self._axisymmetric).tocsr()
        return stiffness, mass, np.asarray(dofs.global_ids, dtype=np.int64)

    def _in_window(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        x0, y0, x1, y1 = self.mesh.window
        slack = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
        return ((points[:, 0] >= x0 - slack) & (points[:, 0] <= x1 + slack)
                & (points[:, 1] >= y0 - slack) & (points[:, 1] <= y1 + slack))

    def _locate(self, point) -> tuple:
        location = locator(self.mesh, self.geometry_order).locate(np.asarray(point, dtype=np.float64).reshape(1, 2))
        element = int(location.element[0])
        if element < 0:
            raise ValueError(f'{self._label()}: point {tuple(float(value) for value in point)} m lies outside the mesh')
        return element, np.asarray(location.reference[0], dtype=np.float64)

    def _point_load(self, point, value) -> np.ndarray:
        element, reference = self._locate(point)
        if not self._active[element]:
            raise ValueError(f'{self._label()}: point source at {tuple(float(item) for item in point)} m lies outside '
                             'the fluid (in a rigid obstacle or a non-fluid element)')
        basis = lagrange_basis(self.order, reference[None, :])[0][0]
        vector = np.zeros(self.full.n_dofs, dtype=np.complex128)
        np.add.at(vector, self.full.element_dofs[element], basis * (value / self._ring()))
        return vector

    def _point_weights(self, element, reference) -> tuple:
        """The full map's element dofs and its Lagrange values at the reference point (fem.sampling.evaluate's basis)."""
        return (np.asarray(self.full.element_dofs[element], dtype=np.int64),
                np.asarray(lagrange_values(self.order, np.asarray(reference, dtype=np.float64)[None, None, :])[0, 0],
                           dtype=np.float64))

    def _centroid_values(self, coefficients) -> np.ndarray:
        return np.asarray(coefficients)[self.full.element_dofs] @ self._centroid_basis()

    def _centroid_basis(self) -> np.ndarray:
        return lagrange_basis(self.order, np.array([[1.0 / 3.0, 1.0 / 3.0]]))[0][0]

    def _centroid_gradients(self, coefficients) -> np.ndarray:
        count = self.mesh.n_elements
        location = Location(np.arange(count, dtype=np.int64), np.full((count, 2), 1.0 / 3.0))
        _, gradients = evaluate(self.full, coefficients, location=location, gradient=True)
        return np.nan_to_num(gradients)

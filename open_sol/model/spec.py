"""Declarative, JSON-serialisable model specification: parameters, geometry, selections, materials, physics, study.
Numeric fields hold ``float | int | str`` (str = expression of parameters); renames keep every reference consistent."""
import numbers
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields, is_dataclass

from ..core.constants import FACES, FACES_3D
from .expressions import (ExpressionError, is_identifier, is_reserved_name, referenced_names,
                          rename_in_expression)
from .schema import INTERFACE_VARIABLES, MESH_PARAMS, output_field_names, scalar_output_names, state_field_names

# version 2 adds DomainSpec.size_z and the geometry z params (3D); version 1 files load unchanged (read as version 2)
SPEC_VERSION = 2
SUPPORTED_SPEC_VERSIONS = (1, 2)
OUTER_FACES = FACES                  # the outer faces of a 2D model (z_lo / z_hi exist only in 3D: outer_faces(spec))
OUTER_FACES_3D = FACES_3D
RESERVED_ENTITY_NAMES = OUTER_FACES_3D + ('all',)


OUTER_FACES_1D = FACES[:2]           # x_lo / x_hi: the two ends of a 1D model (DomainSpec.size_y null)


def outer_faces(spec) -> tuple:
    """Outer face names of the model: OUTER_FACES in 2D, OUTER_FACES_3D when DomainSpec.size_z is set, OUTER_FACES_1D
    when size_y is null."""
    if spec.domain.size_y is None:
        return OUTER_FACES_1D
    return OUTER_FACES_3D if spec.domain.size_z is not None else OUTER_FACES

# Param kinds whose values are numeric expressions that may reference parameters
_PARAMETER_EXPRESSION_KINDS = ('float', 'int', 'expression', 'float_list', 'points', 'table')
# Param kinds whose values may reference physics fields as '<physics>_<field>'
_STATE_EXPRESSION_KINDS = ('float', 'expression')


@dataclass
class ParameterSpec:
    name: str
    expression: float | int | str
    description: str = ''


@dataclass
class GeometrySpec:
    name: str
    kind: str                      # key of geometry.GEOMETRY_KINDS
    params: dict = field(default_factory=dict)


@dataclass
class SelectionSpec:
    name: str
    kind: str                      # 'domain' | 'boundary' | 'point' | 'line' | 'interface'
    entities: list = field(default_factory=list)
    side: str = ''                 # kind 'interface': 'inside' | 'outside' (schema.INTERFACE_SIDES), else ''


@dataclass
class MaterialSpec:
    """Properties override library values; dispersion None inherits the library's poles, [] means none."""
    name: str
    library: str | None = None
    properties: dict = field(default_factory=dict)
    dispersion: list | None = None
    selection: list = field(default_factory=list)


@dataclass
class FeatureSpec:
    name: str
    type: str
    selection: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PhysicsSpec:
    name: str
    type: str
    settings: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    every: int = 1
    solvers: dict = field(default_factory=dict)   # solver role -> SolverSpec name; a missing role = Automatic
    mesh: str = ''                                 # MeshSpec name ('' = the first mesh, ModelSpec.meshes[0])

    def feature(self, name) -> FeatureSpec:
        return _find(self.features, name, f'feature in physics {self.name!r}')


@dataclass
class CouplingSpec:
    """every None -> max(every of the source physics, every of the target physics)."""
    name: str
    type: str
    source: str
    target: str
    every: int | None = None
    params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class RefinementSpec:
    object: str
    dx: float | int | str
    margin: float | int | str | None = None
    dx_x: float | int | str | None = None     # per-axis cell sizes (S7.1 anisotropic); None = dx
    dx_y: float | int | str | None = None
    dx_z: float | int | str | None = None
    automatic: bool = False                   # filled in from the auto-mode sizes (GUI); auto mode ignores it
    where: str = 'inside'                     # REFINEMENT_PLACES: 'edges' / 'surface' on gmsh tetrahedra only (D3)


REFINEMENT_PLACES = ('inside', 'edges', 'surface')   # RefinementSpec.where: in the object, along its edges, at its surface


@dataclass
class MeshBandSpec:
    """Coordinate band [lo, hi] of one MeshAxisSpec axis refined to dx."""
    lo: float | int | str
    hi: float | int | str
    dx: float | int | str


@dataclass
class MeshSegmentSpec:
    """One piece of a custom axis: from the previous end (first: 0) to `end` (None: the axis end) in `cells` cells."""
    cells: float | int | str
    end: float | int | str | None = None
    profile: str = 'uniform'                  # core.mesh.SEGMENT_PROFILES
    ratio: float | int | str = 1.0


@dataclass
class MeshAxisSpec:
    """Per-axis mesh options (nonuniform / auto modes): max_ratio, coordinate bands, custom edges or segments."""
    axis: str
    max_ratio: float | int | str | None = None
    bands: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    segments: list = field(default_factory=list)


@dataclass
class MeshFaceSizeSpec:
    """Element size near a plane face of a tetrahedron mesh (new_modules_plan G1): an outer face, or the bounding-box
    faces of a domain object (face '' = all six), size within `distance` of the face, growing by max_ratio beyond."""
    entity: str                                  # outer face 'x_lo' .. 'z_hi' or a domain object
    size: float | int | str                      # element size at the face [m]
    face: str = ''                               # objects: one face of the bounding box ('x_lo' .. 'z_hi'), '' = all
    distance: float | int | str = 0.0            # half-thickness [m] of the band at `size` around the face


_MESH_DEFAULTS = {param.name: param.default for param in MESH_PARAMS}


@dataclass
class MeshSpec:
    """A named mesh (S7.2): the fields of MESH_PARAMS (defaults and help there; kind 'tensor' | 'triangles' |
    'tetrahedra') plus the refinements and the per-axis options (tensor meshes and builtin tetrahedra)."""
    name: str = 'mesh'
    kind: str = _MESH_DEFAULTS['kind']          # 'tensor' | 'triangles' | 'tetrahedra'
    mode: str = _MESH_DEFAULTS['mode']          # 'uniform' | 'nonuniform' | 'auto'
    dx: float | int | str = _MESH_DEFAULTS['dx']
    max_ratio: float | int | str | None = _MESH_DEFAULTS['max_ratio']   # None = schema.MESH_KIND_DEFAULTS
    wavelength_min: float | int | str | None = _MESH_DEFAULTS['wavelength_min']
    points_per_wavelength: float | int | str | None = _MESH_DEFAULTS['points_per_wavelength']
    refinements: list = field(default_factory=list)
    outer_dx: float | int | str | None = _MESH_DEFAULTS['outer_dx']
    outer_ratio: float | int | str | None = _MESH_DEFAULTS['outer_ratio']
    axes: list = field(default_factory=list)   # MeshAxisSpec, at most one per axis
    h_min: float | int | str | None = _MESH_DEFAULTS['h_min']                 # triangles (S7.5)
    curvature_factor: float | int | str = _MESH_DEFAULTS['curvature_factor']
    mesher: str = _MESH_DEFAULTS['mesher']
    sweep_axis: str = _MESH_DEFAULTS['sweep_axis']   # tetrahedra: 'z' = swept prism layers (plan D5)
    pml_layers: int | str = _MESH_DEFAULTS['pml_layers']
    # triangles: the half (quarter) window meshed and mirrored about the window centre line x / y (new_modules_plan G2)
    mirror_x: bool = _MESH_DEFAULTS['mirror_x']
    mirror_y: bool = _MESH_DEFAULTS['mirror_y']
    # element-size sources [m] graded by max_ratio: triangles [x, y, h] (written by adaptive refinement, S7.7 A3),
    # tetrahedra [x, y, z, h]
    size_points: list = field(default_factory=list)
    face_sizes: list = field(default_factory=list)   # tetrahedra: MeshFaceSizeSpec (sizes near plane faces)


@dataclass
class AdaptationSpec:
    """Mesh refinement of a study (S7.7): 'adaptive' (error-indicator driven) or 'uniform' (a mesh-convergence
    sequence) until the derived value `target` changes by <= tolerance (relative) between levels."""
    target: str
    method: str = 'adaptive'                       # schema.ADAPTATION_METHODS
    tolerance: float | int = 1e-3
    max_levels: int = 2
    growth: float | int = 1.7                      # cell / element count growth per level
    meshes: list = field(default_factory=list)     # mesh names ([] = every mesh of the study's physics)
    max_cells: int | None = None                   # stop before a level above this cell / element count
    anisotropic: bool = True                       # tensor: x / y bands from the dominant indicator part
    keep_levels: bool = False                      # store every level's snapshots, not only the last


@dataclass
class EstimatedParameterSpec:
    """A model parameter the estimation adjusts: start value and optional bounds (numbers or expressions of the base
    parameters)."""
    name: str
    initial: float | int | str | None = None       # None = the parameter's own value
    lower: float | int | str | None = None         # None = unbounded below
    upper: float | int | str | None = None         # None = unbounded above


@dataclass
class EstimationTargetSpec:
    """A measured value of a derived value; its residual is (model - value) / sigma."""
    derived: str
    value: float | int | str
    sigma: float | int | str | None = None         # None = |value| (relative misfit; 1 when value is 0)


@dataclass
class EstimationCurveSpec:
    """Measured samples (x, y) of a 1D result sample or a derived curve; the model is interpolated onto x."""
    source: str                                    # 'physics.feature:sample' or a derived curve name
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    sigma: float | int | str | list | None = None  # one value, one per point, or None = the RMS of y


@dataclass
class EstimationSpec:
    """Parameter estimation (COMSOL Parameter Estimation): the study runs per evaluation until the weighted misfit of
    targets and curves is least, then once more at the best point."""
    parameters: list = field(default_factory=list)   # EstimatedParameterSpec
    targets: list = field(default_factory=list)      # EstimationTargetSpec
    curves: list = field(default_factory=list)       # EstimationCurveSpec
    method: str = 'least_squares'                    # model.estimation.ESTIMATION_METHODS
    max_evaluations: int = 50                        # study runs of the search (the best-point run not counted)
    diff_step: float | int = 1e-2                    # least_squares: relative finite-difference step of the Jacobian
    tolerance: float | int = 1e-6                    # stop when the cost / step changes by less (relative)


@dataclass
class SolverSpec:
    """A solver node: a registered SolverType with its settings, assigned to (physics, role) pairs by
    PhysicsSpec.solvers (one node may serve several)."""
    name: str
    type: str
    settings: dict = field(default_factory=dict)


@dataclass
class SolverDefaultsSpec:
    """Model-wide solver choices (the Solvers node); backend 'auto' = 'gpu' when cupy is installed, else 'cpu'
    (schema.solver_backend; api / CLI backend= overrides it per run)."""
    backend: str = 'auto'          # 'auto' | 'gpu' | 'cpu'
    # 'auto' (each solver's own default) | a PRECISIONS key: every solver that offers it and sets none of its own
    # (Automatic and nodes without a precision setting; schema.with_default_precision, solver_plan S9)
    precision: str = 'auto'
    subpixel_smoothing: bool = False   # material-interface averaging (igor 1.5), honoured per PhysicsType flag
    conformal_boundaries: bool = False   # exact curved walls (solver_plan S7.6), honoured per PhysicsType flag
    # 'error' | 'warning': a memory estimate over the GPU / host budget refuses the model or only warns (schema.MEMORY_CHECKS)
    memory_check: str = 'error'


SWEEP_SPACINGS = ('list', 'linear', 'log')   # SweepSpec.spacing: the values as listed, or count points lower..upper


@dataclass
class SweepSpec:
    """spacing 'list': the values in order; 'linear' / 'log': count points from lower to upper. keep_fields (default
    True): keep the snapshots of every sweep value (summary sweep 'snapshots'), False only the last one's. reuse_build:
    build once and re-set the following feature values per value (model.sweep_reuse)."""
    parameter: str
    values: list = field(default_factory=list)
    keep_fields: bool = True
    spacing: str = 'list'
    lower: float | int | str | None = None
    upper: float | int | str | None = None
    count: int = 5
    reuse_build: bool = False

    def point_values(self) -> list:
        """Sweep values (numbers or expressions) in run order: the listed values, or count evenly / geometrically
        spaced points with exact end points (expressions of lower and upper)."""
        if self.spacing not in ('linear', 'log'):
            return list(self.values) if isinstance(self.values, list) else []
        if not isinstance(self.count, int) or self.count < 1 or self.lower is None or self.upper is None:
            return []
        last = self.count - 1
        lower, upper = f'({self.lower})', f'({self.upper})'
        points = []
        for index in range(self.count):
            if index == 0:
                points.append(self.lower)
            elif index == last:
                points.append(self.upper)
            elif self.spacing == 'linear':
                points.append(f'{lower} + {index}/{last}*({upper} - {lower})')
            else:
                points.append(f'{lower}*({upper}/{lower})**({index}/{last})')
        return points


@dataclass
class StudySpec:
    """settings = values of the StudyType params; stored_fields 'physics.component' ([] -> every output field);
    stored_cuts: 3D cut planes [axis, position] snapshots store besides the mid-height z cut."""
    type: str = 'time_dependent'
    settings: dict = field(default_factory=dict)
    stored_fields: list = field(default_factory=list)
    sweep: SweepSpec | None = None
    stored_cuts: list = field(default_factory=list)   # 3D: [['x' | 'y' | 'z', position [m]], ...] ([] = none)
    adaptation: AdaptationSpec | None = None          # mesh refinement levels (S7.7); None = the model's meshes
    estimation: EstimationSpec | None = None          # parameter estimation (model.study_estimation); None = one run
    # physics names the study itself solves (COMSOL's physics in study step); the others run as stationary background
    # physics. [] = every physics whose solver has the study's solve (validate: studies with a background step only)
    solved_physics: list = field(default_factory=list)

    def solves(self, physics_name) -> bool:
        """False when solved_physics lists other physics only (physics_name then runs as a background physics)."""
        return not self.solved_physics or physics_name in self.solved_physics


@dataclass
class DerivedValueSpec:
    """source None: expression of fields reduced over selection (operation 'expression': one number of parameters,
    scalars, t and earlier derived values); source 'physics.feature': over that 1D result; '<curve name>': that curve."""
    name: str
    expression: str
    selection: list = field(default_factory=list)
    operation: str = 'mean'
    source: str | None = None
    at: float | int | str | None = None            # operation 'at': abscissa value of the source result
    reference: float | int | str | None = None     # analytic value (parameters only)
    tolerance: float | int | None = None           # selftest --examples: passes when derived_error <= tolerance
    window: list = field(default_factory=list)     # source: [lo, hi] abscissa bounds of the reduction ([] = all)
    wavelength: float | int | str | None = None    # far_field source: wavelength [m] of the pattern (None = first)
    fit: str = ''                                  # fit operations: fitted quantity (polyfit 'c<k>', 'fwhm', ...)
    degree: int = 1                                # polyfit: polynomial degree; peak fits: number of lines
    point: str = ''                                # field operations 'at' / 'mode_volume': point object ('' = none)


@dataclass
class AnalyticCurveSpec:
    """Analytic curve drawn over a simulated one; target 'physics.feature', 'sweep' or 'line' (line + field)."""
    name: str
    target: str
    expression: str
    series: str = ''
    line: list = field(default_factory=list)       # target 'line': [x0, y0, x1, y1] (3D or x0..z0, x1..z1) in m
    field: str = ''                                # target 'line': output field 'physics.component'
    x_column: str = ''                             # 'table' target: numeric abscissa column ('' = row index)
    plane: float | int | str | None = None         # target 'line', 3D: cut height z [m] (None = default cut)
    snapshot: int | str | None = None              # target 'line': snapshot index or label (None = final state)
    tolerance: float | int | None = None           # selftest --examples: passes when curve_error max <= tolerance
    interface: str = ''                            # target 'line' (2D): 'interface' selection sampled one-sided
    sampling: str = 'nearest'                      # target 'line', 3D segment: 'nearest' cell | 'trilinear'
    projection: str = ''                           # with interface: '' (field as is) | 'normal' | 'tangential'


@dataclass
class DerivedCurveSpec:
    """Curve computed after the run from 1D feature results: expression of input aliases, x and parameters, drawn
    with the other curves of its plot."""
    name: str
    expression: str
    inputs: dict = field(default_factory=dict)     # alias -> 'physics.feature:sample' (xy series, table column ...)
    plot: str = ''                                 # plot shared with other curves ('' = a plot of its own)
    label: str = ''                                # legend text ('' = name)
    unit: str = ''                                 # unit of the values
    log_x: bool = False                            # the first curve of a plot sets its axes
    log_y: bool = False
    reference: float | int | str | None = None     # analytic curve of x and parameters (curve_error, dotted)
    tolerance: float | int | None = None           # selftest --examples: passes when curve_error max <= tolerance


@dataclass
class DomainSpec:
    """size_z None = 2D model; a value makes the model 3D (layout x layer stack, geometry z params). coordinates
    'polar' / 'spherical' (2D): size_x = radial depth [m] from radius_min, size_y = angle span [rad] from angle_min;
    'polar' with size_z = a cylindrical (r, phi, z) sector, z along the axis (symmetry 2.21, heat). size_y None = 1D model along x (symmetry 2.20): cartesian slab, polar cylinder, spherical sphere (r from radius_min)."""
    size_x: float | int | str = 4e-6
    size_y: float | int | str | None = 4e-6
    size_z: float | int | str | None = None
    coordinates: str = 'cartesian'                 # geometry.DOMAIN_COORDINATES
    radius_min: float | int | str = 0.0           # polar / spherical: r of the x_lo face [m]
    angle_min: float | int | str = 0.0            # polar: phi of the y_lo face, spherical: theta [rad]
    angle_wrap: str = 'none'                       # polar: core.metric.ANGLE_WRAPS (y_hi joined to y_lo)


@dataclass
class ModelSpec:
    title: str = 'Untitled'
    description: str = ''                          # free text for readers (GUI, examples); ignored by build / run
    version: int = SPEC_VERSION
    parameters: list = field(default_factory=list)
    domain: DomainSpec = field(default_factory=DomainSpec)
    geometry: list = field(default_factory=list)
    selections: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    physics: list = field(default_factory=list)
    couplings: list = field(default_factory=list)
    meshes: list = field(default_factory=lambda: [MeshSpec()])   # MeshSpec, >= 1, unique names; first = default
    solvers: list = field(default_factory=list)
    solver_defaults: SolverDefaultsSpec = field(default_factory=SolverDefaultsSpec)
    study: StudySpec = field(default_factory=StudySpec)
    derived_values: list = field(default_factory=list)
    analytic_curves: list = field(default_factory=list)
    derived_curves: list = field(default_factory=list)
    # ModelSpec(mesh=m) (scripts, self-tests) puts m in place of meshes[0]: a keyword of the __init__ wrapper below the
    # class, not a field or InitVar (dataclasses.replace would pass the old meshes[0] back in and undo a new one)

    # ---- serialisation ----
    def to_dict(self) -> dict:
        """Plain JSON-compatible dict (nested dataclasses -> dicts, tuples -> lists)."""
        return _to_plain(self, 'model')

    @classmethod
    def from_dict(cls, data) -> 'ModelSpec':
        """Rebuild from a to_dict() result; ValueError on unknown/missing keys or an unsupported version. A legacy
        'mesh' object (files before S7.2) is read as meshes = [that mesh] (the one reading rule, D17)."""
        if not isinstance(data, dict):
            raise ValueError(f'model: expected an object, got {type(data).__name__}')
        version = data.get('version', SPEC_VERSION)
        if isinstance(version, bool) or version not in SUPPORTED_SPEC_VERSIONS:
            raise ValueError(f'unsupported model version {version!r} (supported: '
                             f"{', '.join(map(str, SUPPORTED_SPEC_VERSIONS))})")
        if 'mesh' in data:
            if 'meshes' in data:
                raise ValueError("model: give 'meshes' (a list) or the legacy 'mesh' object, not both")
            data = {('meshes' if key == 'mesh' else key): ([value] if key == 'mesh' else value)
                    for key, value in data.items()}
        spec = _from_plain(cls, data, 'model')
        spec.version = SPEC_VERSION   # version 1 is a subset of version 2
        return spec

    @property
    def is_3d(self) -> bool:
        """True when DomainSpec.size_z is set (spec v2 3D model)."""
        return self.domain.size_z is not None

    @property
    def dimensions(self) -> int:
        """1 when DomainSpec.size_y is null (a 1D model), 3 when size_z is set, else 2."""
        if self.domain.size_y is None:
            return 1
        return 3 if self.is_3d else 2

    def copy(self) -> 'ModelSpec':
        return deepcopy(self)

    # ---- lookups (KeyError listing the known names) ----
    def geometry_object(self, name) -> GeometrySpec:
        return _find(self.geometry, name, 'geometry object')

    def selection(self, name) -> SelectionSpec:
        return _find(self.selections, name, 'selection')

    def material(self, name) -> MaterialSpec:
        return _find(self.materials, name, 'material')

    def physics_by_name(self, name) -> PhysicsSpec:
        return _find(self.physics, name, 'physics')

    def coupling(self, name) -> CouplingSpec:
        return _find(self.couplings, name, 'coupling')

    def solver(self, name) -> SolverSpec:
        return _find(self.solvers, name, 'solver')

    def derived_value(self, name) -> DerivedValueSpec:
        return _find(self.derived_values, name, 'derived value')

    def analytic_curve(self, name) -> AnalyticCurveSpec:
        return _find(self.analytic_curves, name, 'analytic curve')

    def derived_curve(self, name) -> DerivedCurveSpec:
        return _find(self.derived_curves, name, 'derived curve')

    def parameter(self, name) -> ParameterSpec:
        return _find(self.parameters, name, 'parameter')

    def mesh_by_name(self, name) -> MeshSpec:
        return _find(self.meshes, name, 'mesh')

    # ---- meshes per physics (S7.2) ----
    def mesh_of(self, physics) -> MeshSpec:
        """MeshSpec a physics (PhysicsSpec or name) runs on: PhysicsSpec.mesh, '' = meshes[0]; KeyError for an unknown
        physics or mesh name."""
        physics_spec = physics if isinstance(physics, PhysicsSpec) else self.physics_by_name(physics)
        if not physics_spec.mesh:
            if not self.meshes:
                raise KeyError('the model has no mesh')
            return self.meshes[0]
        return self.mesh_by_name(physics_spec.mesh)

    def find_mesh(self, physics):
        """mesh_of, None for an unknown physics or mesh name (or a malformed mesh entry): validation reports those."""
        try:
            return self.mesh_of(physics)
        except (KeyError, AttributeError):
            return None

    def physics_on(self, mesh_name) -> list:
        """PhysicsSpecs (spec order) whose mesh is mesh_name; physics naming an unknown mesh belong to none."""
        on_mesh = []
        for physics_spec in self.physics:
            try:
                if self.mesh_of(physics_spec).name == mesh_name:
                    on_mesh.append(physics_spec)
            except KeyError:   # reported by validate
                continue
        return on_mesh

    def used_meshes(self) -> list:
        """MeshSpecs (spec order) some physics runs on; [meshes[0]] when no physics names a known mesh (a model
        without physics still has its default mesh), [] without meshes."""
        used, seen = [], set()
        for mesh_spec in self.meshes:   # a repeated name (a validation error) counts once, as _find resolves it
            if mesh_spec.name not in seen and self.physics_on(mesh_spec.name):
                used.append(mesh_spec)
            seen.add(mesh_spec.name)
        return used or self.meshes[:1]

    def all_names(self) -> set:
        """Names of every node (parameters, geometry, selections, materials, physics, features, couplings, solvers,
        derived values, analytic and derived curves)."""
        names = set()
        for items in (self.parameters, self.geometry, self.selections, self.materials, self.physics,
                      self.couplings, self.solvers, self.derived_values, self.analytic_curves, self.derived_curves):
            names.update(item.name for item in items)
        for physics in self.physics:
            names.update(feature.name for feature in physics.features)
        return names

    def unique_name(self, prefix) -> str:
        """prefix + the smallest integer suffix >= 1 not used by any node."""
        taken = self.all_names()
        index = 1
        while f'{prefix}{index}' in taken:
            index += 1
        return f'{prefix}{index}'

    # ---- renames / removal ----
    def rename_geometry(self, old, new, registry):
        """Rename a geometry object in entity lists, composites, refinements and typed 'ref' params."""
        obj = self.geometry_object(old)
        if new == old:
            return
        self._check_entity_name(new)
        obj.name = new
        self._map_entity_lists(lambda entities: [new if entity == old else entity for entity in entities])
        for values, params in self._geometry_param_dicts():
            _map_refs(values, params, 'geometry', old, new)
        for refinement in (refinement for mesh_spec in self.meshes for refinement in mesh_spec.refinements):
            if refinement.object == old:
                refinement.object = new
        for face_size in (face_size for mesh_spec in self.meshes for face_size in mesh_spec.face_sizes):
            if face_size.entity == old:
                face_size.entity = new
        for derived in self.derived_values:
            if derived.point == old:
                derived.point = new
        for values, params in self._typed_param_dicts(registry):
            _map_refs(values, params, 'geometry', old, new)

    def rename_selection(self, old, new):
        """Rename a named selection in every entity list and analytic curve interface."""
        selection = self.selection(old)
        if new == old:
            return
        self._check_entity_name(new)
        selection.name = new
        self._map_entity_lists(lambda entities: [new if entity == old else entity for entity in entities])
        for curve in self.analytic_curves:
            if curve.interface == old:
                curve.interface = new

    def rename_derived_value(self, old, new):
        """Rename a derived value and its references (later derived-value expressions, the adaptation target, the
        estimation targets, 'sweep' analytic-curve series)."""
        derived = self.derived_value(old)
        if new == old:
            return
        _check_new_name(new, [other.name for other in self.derived_values], 'derived value')
        index = self.derived_values.index(derived)
        derived.name = new
        for later in self.derived_values[index + 1:]:
            later.expression = _rename_value(later.expression, lambda name: new if name == old else None)
        adaptation = self.study.adaptation
        if adaptation is not None and adaptation.target == old:
            adaptation.target = new
        for target in self._estimation_items('targets'):
            if target.derived == old:
                target.derived = new
        for curve in self.analytic_curves:
            if curve.target == 'sweep' and curve.series == old:
                curve.series = new

    def rename_mesh(self, old, new):
        """Rename a mesh and its references (PhysicsSpec.mesh, AdaptationSpec.meshes)."""
        mesh_spec = self.mesh_by_name(old)
        if new == old:
            return
        _check_new_name(new, [other.name for other in self.meshes], 'mesh')
        mesh_spec.name = new
        for physics in self.physics:
            if physics.mesh == old:
                physics.mesh = new
        adaptation = self.study.adaptation
        if adaptation is not None and isinstance(adaptation.meshes, list):
            adaptation.meshes = [new if name == old else name for name in adaptation.meshes]

    def remove_mesh(self, name):
        """Delete a mesh no physics uses (ValueError naming them otherwise, and for the last mesh); it also leaves
        AdaptationSpec.meshes."""
        mesh_spec = self.mesh_by_name(name)
        users = [physics.name for physics in self.physics_on(name)]
        if users:
            raise ValueError(f"mesh '{name}' is used by {', '.join(users)}: assign them another mesh first")
        if len(self.meshes) == 1:
            raise ValueError('the model needs at least one mesh')
        self.meshes = [other for other in self.meshes if other is not mesh_spec]
        adaptation = self.study.adaptation
        if adaptation is not None and isinstance(adaptation.meshes, list):
            adaptation.meshes = [other for other in adaptation.meshes if other != name]

    def rename_material(self, old, new, registry):
        """Rename a material and typed 'ref' params of ref_kind 'material'."""
        material = self.material(old)
        if new == old:
            return
        _check_new_name(new, [other.name for other in self.materials], 'material')
        material.name = new
        for values, params in self._typed_param_dicts(registry):
            _map_refs(values, params, 'material', old, new)

    def rename_physics(self, old, new, registry):
        """Rename a physics in couplings, stored fields, result keys ('<old>.<feature>') and '<old>_<field>'
        expression names. Only the type's state / output field and scalar output names and float / expression /
        'physics'-ref params change."""
        physics = self.physics_by_name(old)
        if new == old:
            return
        _check_new_name(new, [other.name for other in self.physics], 'physics')
        try:
            physics_type = registry.physics_type(physics.type)
            field_names = {info.name for info in physics_type.state_fields + physics_type.output_fields}
            field_names.update(scalar for physics_name, scalar in scalar_output_names(self, registry).values()
                               if physics_name == old)
        except KeyError:
            field_names = None
        parameter_names = {parameter.name for parameter in self.parameters}
        old_prefix = old + '_'
        # '<old>_x_<field>' belongs to a physics named '<old>_x' when one exists
        longer_physics_prefixes = tuple(other.name + '_' for other in self.physics
                                        if other.name != old and other.name.startswith(old_prefix))

        def rename(name):
            if (not name.startswith(old_prefix) or name in parameter_names
                    or name.startswith(longer_physics_prefixes)):
                return None
            field_name = name[len(old_prefix):]
            if field_names is not None and field_name not in field_names:
                return None
            return f'{new}_{field_name}'

        physics.name = new
        for coupling in self.couplings:
            if coupling.source == old:
                coupling.source = new
            if coupling.target == old:
                coupling.target = new
        self.study.stored_fields = [_rename_dotted(entry, old, new) for entry in self.study.stored_fields]
        if isinstance(self.study.solved_physics, list):
            self.study.solved_physics = [new if name == old else name for name in self.study.solved_physics]
        for derived in self.derived_values:
            derived.source = _rename_dotted(derived.source, old, new)
        for curve in self.analytic_curves:
            curve.target = _rename_dotted(curve.target, old, new)
            curve.field = _rename_dotted(curve.field, old, new)
        for curve in self.derived_curves:
            curve.inputs = {alias: _rename_dotted(reference, old, new) for alias, reference in curve.inputs.items()}
        for curve in self._estimation_items('curves'):
            curve.source = _rename_dotted(curve.source, old, new)
        self._rename_value_expressions(rename)
        for values, params in self._typed_param_dicts(registry):
            _rename_typed(values, params, _STATE_EXPRESSION_KINDS, rename)
            _map_refs(values, params, 'physics', old, new)

    def rename_parameter(self, old, new, registry):
        """Rename a parameter in every numeric expression (str/choice values are never touched); curve variables and
        result samples that shadow it keep their name (xy / table sourced expressions are left alone)."""
        parameter = self.parameter(old)
        if new == old:
            return
        _check_new_name(new, [other.name for other in self.parameters], 'parameter')
        if is_reserved_name(new):
            raise ValueError(f'{new!r} is a reserved name')
        if new in state_field_names(self, registry) or new in output_field_names(self, registry):
            raise ValueError(f'{new!r} is a physics field name')
        if new in scalar_output_names(self, registry):
            raise ValueError(f'{new!r} is a physics scalar output name')
        shadows = {id(item): self._parameter_shadows(item, registry)
                   for item in self.analytic_curves + self.derived_values + self.derived_curves}
        for item in self.analytic_curves + self.derived_values + self.derived_curves:
            shadowing = shadows[id(item)]
            if (shadowing is not None and new in shadowing and old not in shadowing
                    and old in _expression_names(item.expression)):
                raise ValueError(f'{new!r} is a variable of the expression of {item.name!r}, which reads parameter '
                                 f'{old!r}: choose another name')

        def rename(name):
            return new if name == old else None

        def shadowed(item):
            return shadows[id(item)]

        parameter.name = new
        for other in self.parameters:
            other.expression = _rename_value(other.expression, rename)
        self.domain.size_x = _rename_value(self.domain.size_x, rename)
        self.domain.size_y = _rename_value(self.domain.size_y, rename)
        self.domain.size_z = _rename_value(self.domain.size_z, rename)
        self.domain.radius_min = _rename_value(self.domain.radius_min, rename)
        self.domain.angle_min = _rename_value(self.domain.angle_min, rename)
        for mesh_spec in self.meshes:
            _rename_typed(vars(mesh_spec), MESH_PARAMS, _PARAMETER_EXPRESSION_KINDS, rename)
            for refinement in mesh_spec.refinements:
                for name in ('dx', 'margin', 'dx_x', 'dx_y', 'dx_z'):
                    setattr(refinement, name, _rename_value(getattr(refinement, name), rename))
            for face_size in mesh_spec.face_sizes:
                face_size.size = _rename_value(face_size.size, rename)
                face_size.distance = _rename_value(face_size.distance, rename)
            for axis_spec in mesh_spec.axes:
                axis_spec.max_ratio = _rename_value(axis_spec.max_ratio, rename)
                axis_spec.edges = _rename_value(axis_spec.edges, rename)
                for item in axis_spec.bands + axis_spec.segments:
                    for name, value in list(vars(item).items()):
                        if name != 'profile':
                            setattr(item, name, _rename_value(value, rename))
        sweep = self.study.sweep
        if sweep is not None:
            if sweep.parameter == old:
                sweep.parameter = new
            sweep.values = _rename_value(sweep.values, rename)
            sweep.lower, sweep.upper = _rename_value(sweep.lower, rename), _rename_value(sweep.upper, rename)
        for estimated in self._estimation_items('parameters'):
            if estimated.name == old:
                estimated.name = new
            for name in ('initial', 'lower', 'upper'):
                setattr(estimated, name, _rename_value(getattr(estimated, name), rename))
        for item in self._estimation_items('targets') + self._estimation_items('curves'):
            if hasattr(item, 'value'):
                item.value = _rename_value(item.value, rename)
            item.sigma = _rename_value(item.sigma, rename)
        self.study.stored_cuts = [[cut[0], _rename_value(cut[1], rename)]
                                  if isinstance(cut, (list, tuple)) and len(cut) == 2 else cut
                                  for cut in self.study.stored_cuts]
        for values, params in self._geometry_param_dicts():
            _rename_typed(values, params, _PARAMETER_EXPRESSION_KINDS, rename)
        self._rename_value_expressions(rename, shadowed)
        for values, params in self._typed_param_dicts(registry):
            _rename_typed(values, params, _PARAMETER_EXPRESSION_KINDS, rename)
            _map_refs(values, params, 'parameter', old, new)

    def _parameter_shadows(self, item, registry):
        """Names shadowing parameters in the expression of an analytic curve, derived value or derived curve; None when
        unknown before the run (xy / table samples, unresolved target)."""
        from .analytic import DERIVED_CURVE_VARIABLES, RESULT_SAMPLE_NAMES, curve_variables, is_curve_source
        if isinstance(item, DerivedCurveSpec):
            return frozenset(item.inputs) | frozenset(DERIVED_CURVE_VARIABLES)
        if isinstance(item, AnalyticCurveSpec):
            kind = item.target if item.target in ('sweep', 'line') else self._result_kind(item.target, registry)
            try:
                return frozenset(curve_variables(kind, item.series))
            except KeyError:
                return None
        if item.source is None:
            return frozenset(INTERFACE_VARIABLES)
        if is_curve_source(item.source):
            return frozenset((item.source, 'x'))
        sample_names = RESULT_SAMPLE_NAMES.get(self._result_kind(item.source, registry), (None, None))[1]
        return None if sample_names is None else frozenset(sample_names)

    def _result_kind(self, key, registry):
        """result_kind of the feature behind a 'physics.feature' key; None when it does not resolve."""
        physics_name, _, feature_name = key.partition('.') if isinstance(key, str) else ('', '', '')
        try:
            physics = self.physics_by_name(physics_name)
            feature = physics.feature(feature_name)
            return registry.physics_type(physics.type).feature_type(feature.type).result_kind
        except (KeyError, AttributeError):
            return None

    def rename_result_key(self, old_key, new_key):
        """Replace the feature result key 'physics.feature' old_key by new_key in derived value sources, analytic
        curve targets and derived curve inputs (a renamed feature)."""
        for derived in self.derived_values:
            if derived.source == old_key:
                derived.source = new_key
        for curve in self.analytic_curves:
            if curve.target == old_key:
                curve.target = new_key
        for curve in self.derived_curves:
            curve.inputs = {alias: (new_key + reference[len(old_key):]
                                    if isinstance(reference, str) and reference.startswith(old_key + ':')
                                    else reference)
                            for alias, reference in curve.inputs.items()}
        for curve in self._estimation_items('curves'):
            if isinstance(curve.source, str) and curve.source.startswith(old_key + ':'):
                curve.source = new_key + curve.source[len(old_key):]

    def _estimation_items(self, kind) -> list:
        """The StudySpec.estimation entries of kind 'parameters' | 'targets' | 'curves' that are spec objects ([]
        without an estimation; malformed entries are left to validation)."""
        estimation = self.study.estimation
        items = getattr(estimation, kind, None) if estimation is not None else None
        classes = {'parameters': EstimatedParameterSpec, 'targets': EstimationTargetSpec,
                   'curves': EstimationCurveSpec}
        return [item for item in items if isinstance(item, classes[kind])] if isinstance(items, list) else []

    def remove_geometry(self, name, registry):
        """Delete a geometry object and strip every reference (entity lists, composites, refinements, typed refs)."""
        obj = self.geometry_object(name)
        self.geometry = [other for other in self.geometry if other is not obj]
        self._map_entity_lists(lambda entities: [entity for entity in entities if entity != name])
        for values, params in self._geometry_param_dicts():
            _map_refs(values, params, 'geometry', name, None)
        for mesh_spec in self.meshes:
            mesh_spec.refinements = [refinement for refinement in mesh_spec.refinements if refinement.object != name]
            mesh_spec.face_sizes = [face_size for face_size in mesh_spec.face_sizes if face_size.entity != name]
        for derived in self.derived_values:
            if derived.point == name:
                derived.point = ''
        for values, params in self._typed_param_dicts(registry):
            _map_refs(values, params, 'geometry', name, None)

    def remove_selection(self, name):
        """Delete a named selection and strip it from every entity list and analytic curve interface."""
        selection = self.selection(name)
        self.selections = [other for other in self.selections if other is not selection]
        self._map_entity_lists(lambda entities: [entity for entity in entities if entity != name])
        for curve in self.analytic_curves:
            if curve.interface == name:
                curve.interface = ''
                curve.projection = ''

    def remove_physics(self, name, registry):
        """Delete a physics with its couplings, stored fields and typed 'physics' refs."""
        physics = self.physics_by_name(name)
        self.physics = [other for other in self.physics if other is not physics]
        self.couplings = [coupling for coupling in self.couplings
                          if coupling.source != name and coupling.target != name]
        self.study.stored_fields = [entry for entry in self.study.stored_fields
                                    if not (isinstance(entry, str) and entry.startswith(name + '.'))]
        if isinstance(self.study.solved_physics, list):
            self.study.solved_physics = [other for other in self.study.solved_physics if other != name]
        for values, params in self._typed_param_dicts(registry):
            _map_refs(values, params, 'physics', name, None)

    def rename_solver(self, old, new):
        """Rename a solver node and its role assignments (PhysicsSpec.solvers)."""
        solver = self.solver(old)
        if new == old:
            return
        _check_new_name(new, [other.name for other in self.solvers], 'solver')
        solver.name = new
        for physics in self.physics:
            if isinstance(physics.solvers, dict):
                physics.solvers = {role: new if node == old else node for role, node in physics.solvers.items()}

    def remove_solver(self, name):
        """Delete a solver node; the physics roles it filled return to Automatic."""
        solver = self.solver(name)
        self.solvers = [other for other in self.solvers if other is not solver]
        for physics in self.physics:
            if isinstance(physics.solvers, dict):
                physics.solvers = {role: node for role, node in physics.solvers.items() if node != name}

    # ---- internal helpers ----
    def _check_entity_name(self, new):
        """Geometry objects and selections share one namespace that avoids the reserved entity names."""
        taken = [obj.name for obj in self.geometry] + [selection.name for selection in self.selections]
        _check_new_name(new, taken, 'geometry/selection')
        if new in RESERVED_ENTITY_NAMES:
            raise ValueError(f'{new!r} is reserved (reserved entity names: {list(RESERVED_ENTITY_NAMES)})')

    def _map_entity_lists(self, transform):
        """Replace every entity list (selections, materials, features, derived values) by transform(list)."""
        for selection in self.selections:
            selection.entities = transform(list(selection.entities))
        for material in self.materials:
            material.selection = transform(list(material.selection))
        for physics in self.physics:
            for feature in physics.features:
                feature.selection = transform(list(feature.selection))
        for derived in self.derived_values:
            derived.selection = transform(list(derived.selection))

    def _rename_value_expressions(self, rename, shadowed=None):
        """Apply rename(name) to material property / dispersion values, derived value expressions / at / reference /
        window / wavelength, analytic curve expressions / lines and derived curve expressions / references;
        shadowed(item) -> names an item's expression keeps (None: the whole expression; default: derived curve aliases)."""
        def item_rename(item):
            keep = (frozenset(item.inputs) if isinstance(item, DerivedCurveSpec) else frozenset()) if shadowed is None \
                else shadowed(item)
            return (lambda name: None) if keep is None else (lambda name: None if name in keep else rename(name))

        for material in self.materials:
            material.properties = {key: _rename_value(value, rename) for key, value in material.properties.items()}
            if material.dispersion is not None:
                material.dispersion = [
                    {key: value if key == 'model' else _rename_value(value, rename) for key, value in pole.items()}
                    if isinstance(pole, dict) else pole
                    for pole in material.dispersion]
        for derived in self.derived_values:
            derived.expression = _rename_value(derived.expression, item_rename(derived))
            derived.at = _rename_value(derived.at, rename)
            derived.reference = _rename_value(derived.reference, rename)
            derived.window = _rename_value(derived.window, rename)
            derived.wavelength = _rename_value(derived.wavelength, rename)
        for curve in self.analytic_curves:
            curve.expression = _rename_value(curve.expression, item_rename(curve))
            curve.line = _rename_value(curve.line, rename)
            curve.plane = _rename_value(curve.plane, rename)
        for curve in self.derived_curves:   # an input alias (and x) shadows a parameter of the same name
            curve.expression = _rename_value(curve.expression, item_rename(curve))
            curve.reference = _rename_value(curve.reference, rename)

    def _geometry_param_dicts(self):
        """Yield (params dict, Param tuple) of every geometry object with a known kind."""
        from .geometry import GEOMETRY_KINDS   # geometry imports this module
        for obj in self.geometry:
            kind = GEOMETRY_KINDS.get(obj.kind)
            if kind is not None and isinstance(obj.params, dict):
                yield obj.params, kind.params

    def _typed_param_dicts(self, registry):
        """Yield (values dict, Param tuple) for settings, features, couplings, solvers and study of registered types."""
        for physics in self.physics:
            try:
                physics_type = registry.physics_type(physics.type)
            except KeyError:
                continue
            if isinstance(physics.settings, dict):
                yield physics.settings, physics_type.settings
            for feature in physics.features:
                try:
                    feature_type = physics_type.feature_type(feature.type)
                except KeyError:
                    continue
                if isinstance(feature.params, dict):
                    yield feature.params, feature_type.params
        for coupling in self.couplings:
            try:
                coupling_type = registry.coupling_type(coupling.type)
            except KeyError:
                continue
            if isinstance(coupling.params, dict):
                yield coupling.params, coupling_type.params
        for solver in self.solvers:
            try:
                solver_type = registry.solver_type(solver.type)
            except KeyError:
                continue
            if isinstance(solver.settings, dict):
                yield solver.settings, solver_type.params
        try:
            study_type = registry.study_type(self.study.type)
        except KeyError:
            return
        if isinstance(self.study.settings, dict):
            yield self.study.settings, study_type.params


def _default_mesh(spec):
    return spec.meshes[0]


def _set_default_mesh(spec, mesh_spec):
    spec.meshes = [mesh_spec] + list(spec.meshes[1:])


ModelSpec.mesh = property(_default_mesh, _set_default_mesh, doc='meshes[0], the default mesh (physics with mesh '
                                                                "''); one-mesh scripts and self-tests read / set it")
_dataclass_init = ModelSpec.__init__


def _init_with_default_mesh(self, *args, mesh=None, **kwargs):
    """The dataclass __init__ plus the keyword mesh: m in place of meshes[0] (None: meshes as given)."""
    _dataclass_init(self, *args, **kwargs)
    if mesh is not None:
        self.meshes = [mesh] + list(self.meshes[1:])


ModelSpec.__init__ = _init_with_default_mesh


# ---- module helpers ----
def _find(items, name, category):
    for item in items:
        if item.name == name:
            return item
    raise KeyError(f'no {category} named {name!r}; known: {[item.name for item in items]}')


def _check_new_name(new, taken, category):
    if not is_identifier(new):
        raise ValueError(f'{new!r} is not a valid {category} name (letters, digits and _, not starting with a digit)')
    if new in taken:
        raise ValueError(f'a {category} named {new!r} already exists')


def _rename_dotted(value, old, new):
    """'<old>.<rest>' -> '<new>.<rest>' (stored fields, result keys); any other value unchanged."""
    return new + value[len(old):] if isinstance(value, str) and value.startswith(old + '.') else value


def _expression_names(value) -> frozenset:
    """Free names of an expression string; empty for numbers and unparsable text."""
    if not isinstance(value, str):
        return frozenset()
    try:
        return referenced_names(value)
    except ExpressionError:
        return frozenset()


def _rename_value(value, rename):
    """Rename names in an expression string or (nested) list via rename(name) -> new name | None."""
    if isinstance(value, str):
        for name in sorted(_expression_names(value)):
            new_name = rename(name)
            if new_name is not None and new_name != name:
                value = rename_in_expression(value, name, new_name)
        return value
    if isinstance(value, (list, tuple)):
        return [_rename_value(item, rename) for item in value]
    return value


def _rename_typed(values, params, kinds, rename):
    """Rename expression names in the entries of values whose Param kind is in kinds (in place)."""
    for param in params:
        if param.kind in kinds and param.name in values:
            values[param.name] = _rename_value(values[param.name], rename)


def _map_refs(values, params, ref_kind, name, replacement):
    """'ref' / 'names' params of the given ref_kind holding name (in place): a 'ref' becomes replacement, a 'names' list
    has it replaced (replacement None: dropped)."""
    for param in params:
        if param.ref_kind != ref_kind or param.name not in values:
            continue
        value = values[param.name]
        if param.kind == 'ref' and value == name:
            values[param.name] = replacement
        elif param.kind == 'names' and isinstance(value, (list, tuple)):
            values[param.name] = [replacement if item == name else item for item in value
                                  if item != name or replacement is not None]


def _to_plain(value, where):
    """JSON-compatible deep copy: dataclasses -> dicts, tuples -> lists, numeric scalars -> int/float."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name), f'{where}.{f.name}') for f in fields(value)}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, dict):
        plain = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{where}: key {key!r} is not text (JSON object keys must be strings)')
            plain[key] = _to_plain(item, f'{where}.{key}')
        return plain
    if isinstance(value, (list, tuple)):
        return [_to_plain(item, f'{where}[{index}]') for index, item in enumerate(value)]
    raise ValueError(f'{where}: value of type {type(value).__name__} is not JSON-compatible')


# Nested dataclass fields: field name -> (dataclass, 'list' | 'one' | 'optional')
_NESTED_FIELDS = {
    ModelSpec: {'parameters': (ParameterSpec, 'list'), 'domain': (DomainSpec, 'one'),
                'geometry': (GeometrySpec, 'list'), 'selections': (SelectionSpec, 'list'),
                'materials': (MaterialSpec, 'list'), 'physics': (PhysicsSpec, 'list'),
                'couplings': (CouplingSpec, 'list'), 'meshes': (MeshSpec, 'list'),
                'solvers': (SolverSpec, 'list'), 'solver_defaults': (SolverDefaultsSpec, 'one'),
                'study': (StudySpec, 'one'),
                'derived_values': (DerivedValueSpec, 'list'), 'analytic_curves': (AnalyticCurveSpec, 'list'),
                'derived_curves': (DerivedCurveSpec, 'list')},
    PhysicsSpec: {'features': (FeatureSpec, 'list')},
    MeshSpec: {'refinements': (RefinementSpec, 'list'), 'axes': (MeshAxisSpec, 'list'),
               'face_sizes': (MeshFaceSizeSpec, 'list')},
    MeshAxisSpec: {'bands': (MeshBandSpec, 'list'), 'segments': (MeshSegmentSpec, 'list')},
    StudySpec: {'sweep': (SweepSpec, 'optional'), 'adaptation': (AdaptationSpec, 'optional'),
                'estimation': (EstimationSpec, 'optional')},
    EstimationSpec: {'parameters': (EstimatedParameterSpec, 'list'), 'targets': (EstimationTargetSpec, 'list'),
                     'curves': (EstimationCurveSpec, 'list')},
}
_LIST_OR_NONE_FIELDS = {(MaterialSpec, 'dispersion')}
# Fields annotated str (or str | None: null while the GUI has not chosen one) take only text, bool fields (switches
# read by truthiness downstream: "false" would leave a feature on) only JSON true / false. The annotations are real
# types (no `from __future__ import annotations` here), so every new field is covered.
_TEXT_TYPES = (str, str | None)


def _from_plain(cls, data, where):
    """Build dataclass cls from a plain dict; ValueError naming the location on unknown/missing keys or bad types."""
    if not isinstance(data, dict):
        raise ValueError(f'{where}: expected an object, got {type(data).__name__}')
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(str(key) for key in data if key not in known)
    if unknown:
        raise ValueError(f'{where}: unknown keys {unknown}; allowed: {sorted(known)}')
    missing = [name for name, f in known.items()
               if name not in data and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError(f'{where}: missing keys {missing}')
    nested = _NESTED_FIELDS.get(cls, {})
    kwargs = {}
    for key, value in data.items():
        location = f'{where}.{key}'
        if key in nested:
            child_cls, arity = nested[key]
            if arity == 'list':
                if not isinstance(value, list):
                    raise ValueError(f'{location}: expected a list, got {type(value).__name__}')
                kwargs[key] = [_from_plain(child_cls, item, f'{location}[{index}]')
                               for index, item in enumerate(value)]
            elif arity == 'optional' and value is None:
                kwargs[key] = None
            else:
                kwargs[key] = _from_plain(child_cls, value, location)
            continue
        default_factory = known[key].default_factory
        if default_factory in (list, dict) and not isinstance(value, default_factory):
            raise ValueError(f'{location}: expected a {default_factory.__name__}, got {type(value).__name__}')
        if (cls, key) in _LIST_OR_NONE_FIELDS and value is not None and not isinstance(value, list):
            raise ValueError(f'{location}: expected a list or null, got {type(value).__name__}')
        if known[key].type in _TEXT_TYPES and value is not None and not isinstance(value, str):
            raise ValueError(f'{location}: expected text, got {type(value).__name__}')
        if known[key].type is bool and not isinstance(value, bool):
            raise ValueError(f'{location}: expected true or false, got {value!r}')
        kwargs[key] = _to_plain(value, location)
    return cls(**kwargs)

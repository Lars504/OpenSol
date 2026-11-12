"""CPU-only model validation: every problem of a ModelSpec becomes an Issue at its GUI node path.
validate() never raises for model errors; physics / feature / coupling rules come from the types' validate callbacks."""
import math
import os
import traceback
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..core.gpu_memory import format_bytes
from ..materials import REGISTRY as MATERIAL_LIBRARY, crystal_entries
from .analytic import (COMPLEX_FIT_OPERATIONS, CURVE_FUNCTION_ARITY, CURVE_RESULT_KINDS, CURVE_VARIABLES,
                       DERIVED_CURVE_VARIABLES, DERIVED_RESULT_KINDS, FFT_SERIES, FIT_QUANTITIES, RATIO_SERIES,
                       RESULT_SAMPLE_NAMES, SEGMENT_SAMPLINGS, complex_fit_series, curve_variables, fit_problem,
                       is_curve_source, is_segment_3d, split_curve_input, split_key)
from .expressions import (COORDINATE_NAMES, RESERVED_NAMES, THREE_D_COORDINATE_NAMES, ExpressionError,
                          compile_expression, error_message, evaluate_parameters, evaluate_scalar, is_identifier,
                          is_reserved_name, referenced_names)
from .definition_helpers import is_axisymmetric
from .interface_sampling import interface_selection, partner_component
from .geometry import (DOMAIN_COORDINATES, GEOMETRY_KINDS, array_overlap, boundary_entities, bounding_box,
                       box_window, composite_inputs,
                       consumed_objects, domain_entities, domain_map, domain_object_rasterizer, expand_entities,
                       interface_entities, line_entities, material_paint_order, paint_material_index, point_entities,
                       preview_pixel_centers, rasterize, rasterize_box, rasterize_window, resolve_geometry,
                       Z_PARAM_NAMES,
                       z_param_set)
from .meshing import (MESH_ESTIMATE_ERRORS, PHYSICS_BASE_DIVISOR, PHYSICS_OBJECT_DIVISOR,
                      backend_memory, boundary_config_difference_text, estimate_memory, estimate_mesh, estimate_meshes,
                      awaits_built_mesh, has_estimated_counts, largest_adjacent_ratio, size_points_problem)
from .schema import (ADAPTATION_METHODS, BACKENDS, DERIVED_OPERATIONS, FIELD_DERIVED_OPERATIONS,
                     GLOBAL_DERIVED_OPERATION, INTERFACE_OPERATIONS, SOURCE_FREE_DERIVED_OPERATIONS,
                     INTERFACE_PROJECTIONS, INTERFACE_SIDES, INTERFACE_VARIABLES, FREE_MESH_KINDS, MESH_KIND_DEFAULTS,
                     MEMORY_CHECKS, MESH_KINDS, MESH_PARAMS,
                     PERMITTIVITY_PROPERTY_NAMES,
                     PRECISIONS, SELECTION_KINDS, SOURCE_DERIVED_OPERATIONS, SOLVER_BACKENDS, SOLVER_DEFAULT_PRECISIONS,
                     SOLVER_ROLES, Issue, ValidationContext, check_value,
                     conformal_boundaries_physics, coupling_role_text, coupling_type_accepts, coupling_writes,
                     default_precision_ignored, effective_coupling_every, with_default_precision, with_node_precision,
                     feature_precision_misfit, fill_defaults, get_registry, memory_check_level,
                     output_field_names, param_visible, short_repr, solver_backend, solver_grid, solver_misfit,
                     state_field_names,
                     subpixel_smoothing_physics, valid_every)
from .spec import OUTER_FACES_3D, REFINEMENT_PLACES, RESERVED_ENTITY_NAMES, SWEEP_SPACINGS, outer_faces
from .symmetry_check import symmetry_issues
from .triangle_meshing import mirror_mismatch

MEMORY_WARN_FRACTION = 0.8      # warn when the estimated GPU peak / host memory exceeds this share of its budget
LARGE_HOST_BYTES = 8 * 2 ** 30  # without a host budget: warn above this estimated host memory (CPU factorisations)
MAX_COVERAGE_TEST_CELLS = 1e6   # empty-object / override checks: more bounding-box cells to rasterise are skipped
_COUPLING_EXPRESSION_NAMES = frozenset({'value'})    # coupling 'expression' params see the transferred field value
_EXPRESSION_KINDS = frozenset({'float', 'int', 'expression'})   # Param kinds whose string values are expressions
_ENTITY_KIND_TEXT = {'domain': "a domain object or 'all'", 'boundary': 'an outer face or a domain object perimeter',
                     'point': 'a point object', 'line': 'a line object', 'interface': 'a domain object outline'}
_ENTITY_LISTINGS = {'domain': domain_entities, 'boundary': boundary_entities, 'point': point_entities,
                    'line': line_entities, 'interface': interface_entities}   # selection kind -> entity listing
# studies whose physics share one mesh (S7.2 1.2: they combine the physics' solutions on one grid)
_ONE_MESH_STUDIES = ('thermal_noise', 'optomechanics', 'nonlinear_frequency_domain')
# studies whose run picks its solved physics by StudySpec.solved_physics (study._solver_roles / the optomechanics roles)
_SOLVED_PHYSICS_STUDIES = ('frequency_domain', 'mode_analysis', 'eigenfrequency', 'frequency_response', 'optomechanics')
_GMSH_ENVIRONMENT = 'OPEN_SOL_GMSH'   # path of the gmsh program (MeshSpec.mesher 'gmsh'), else the PATH


def validate(spec, registry=None, mesh_estimate=None, mesh_error=None, memory_budget=None, memory_estimate=None,
             host_budget=None, memory_parts=None):
    """Issues of the model in GUI tree order (errors block building); CPU only, never raises for model errors.
    mesh_estimate / mesh_error: cached meshing estimates {mesh name: estimate} / failure messages {mesh name: text}, or
    one estimate / message of the default mesh; the missing used meshes are estimated here; memory_budget: GPU pool
    budget [bytes] (None: no GPU memory check); memory_estimate: a cached meshing.estimate_memory result of those
    meshes (else computed); host_budget: available host RAM [bytes] (core.host_memory.available_bytes; None: only the
    fixed LARGE_HOST_BYTES warning); memory_parts: that estimate's by_mesh parts, naming the mesh a budget issue is
    reported at (None: computed when one is reported)."""
    registry = get_registry() if registry is None else registry
    return _Validator(spec, registry, mesh_estimate, mesh_error, memory_budget, memory_estimate, host_budget,
                      memory_parts).run()


@dataclass(frozen=True)
class MemoryVerdict:
    """Budget judgement of a memory estimate: the used mesh its issues are reported at (the largest part), the budgets
    it exceeds, and the used element meshes not yet counted on their built mesh (pending, meshing.awaits_built_mesh:
    a build meshes them first and judges on the built counts and factor fills)."""
    mesh: str
    gpu_over: bool
    host_over: bool
    pending: tuple

    @property
    def over(self) -> bool:
        return self.gpu_over or self.host_over


def memory_verdict(spec, mesh_estimates, estimate, memory_budget=None, host_budget=None, parts=None) -> MemoryVerdict:
    """MemoryVerdict of a meshing.estimate_memory result of the used meshes {name: estimate} against the GPU budget
    (gpu backend) and the available host RAM (None: not judged); parts: its by_mesh ({} / None: the first mesh)."""
    held = backend_memory(spec, estimate)
    budget = None if solver_backend(spec) == 'cpu' else memory_budget
    gpu_over = bool(budget) and held.device_peak > budget
    host_over = host_budget is not None and held.host > host_budget
    by_device = bool(budget) and (gpu_over or not host_over)
    part_bytes = {}
    for name, part in (parts or {}).items():
        if name in mesh_estimates:
            held_part = backend_memory(spec, part)
            part_bytes[name] = held_part.device_peak if by_device else held_part.host
    mesh = max(part_bytes, key=part_bytes.get) if part_bytes else next(iter(mesh_estimates), '')
    pending = tuple(name for name, mesh_estimate in mesh_estimates.items() if awaits_built_mesh(mesh_estimate))
    return MemoryVerdict(mesh, gpu_over, host_over, pending)


def refused_on_estimated_counts(spec, mesh_estimates, estimate, memory_budget=None, host_budget=None) -> bool:
    """True when the budget check would refuse the model (memory_check 'error') on element counts that are only
    size-based: the caller builds those meshes (meshing.built_mesh_estimates) and judges on their counts."""
    if (not mesh_estimates or estimate is None or memory_check_level(spec) != 'error'
            or any(mesh is None for mesh in mesh_estimates.values())):
        return False
    verdict = memory_verdict(spec, mesh_estimates, estimate, memory_budget, host_budget)
    return verdict.over and bool(verdict.pending)


class _Validator:
    """One validation pass: partial evaluation (parameters, geometry, mesh) in __init__, the checks in run()."""

    def __init__(self, spec, registry, mesh_estimate, mesh_error, memory_budget=None, memory_estimate=None,
                 host_budget=None, memory_parts=None):
        self.spec = spec
        self.registry = registry
        self.memory_budget = memory_budget
        self.host_budget = host_budget
        self.memory_estimate = memory_estimate
        self.memory_parts = memory_parts
        self.issues = []
        self.n_errors = 0
        parameter_errors, geometry_errors = {}, {}
        parameters = evaluate_parameters(spec.parameters, errors=parameter_errors)
        shapes = resolve_geometry(spec, parameters, errors=geometry_errors)
        meshes, mesh_errors = _mesh_estimates(spec, registry, parameters, mesh_estimate, mesh_error, shapes)
        default_name = next(iter(meshes), '')
        self.vctx = ValidationContext(spec=spec, registry=registry, parameters=parameters,
                                      parameter_errors=parameter_errors, shapes=shapes,
                                      geometry_errors=geometry_errors, mesh=meshes.get(default_name),
                                      mesh_error=mesh_errors.get(default_name), study_type=spec.study.type,
                                      state_names=state_field_names(spec, registry), meshes=meshes,
                                      mesh_errors=mesh_errors, mesh_name=default_name)
        self.state_names = frozenset(self.vctx.state_names)
        self.output_names = output_field_names(spec, registry)   # {field name: (physics, field)}, read-only
        self.objects = _first_by_name(spec.geometry)
        self.selections = _first_by_name(spec.selections)
        self.physics_by_name = _first_by_name(spec.physics)
        self.consumed = consumed_objects(spec)
        self.physics_entries = []   # (PhysicsSpec, PhysicsType, settings values or None) of registered physics
        self.scalar_names = set()   # '<physics>_<scalar>' of valid scalar outputs (check_physics; derived values)
        self.mesh_inputs_ok = True  # False when physics settings / boundary providers (mesh estimate inputs) fail
        self.empty_object_messages = {}   # (mesh name, domain object) -> warning when it selects no cell, else None

    def run(self):
        self.check_names()
        self.check_parameters()
        domain_ok = self.check_domain()
        self.check_geometry()
        self.check_selections()
        self.check_materials()
        self.check_physics()
        self.check_symmetry()
        self.check_features()
        self.check_couplings()
        self.check_meshes(domain_ok)
        self.check_memory()
        self.check_solvers()
        self.check_study()
        self.check_derived_values()
        self.check_analytic_curves()
        self.check_derived_curves()
        return self.issues

    # ---- issue collection ----
    def append(self, issue):
        self.issues.append(issue)
        if issue.level == 'error':
            self.n_errors += 1

    def add(self, level, path, message):
        self.append(Issue(level, tuple(path), message))

    def error(self, path, message):
        self.add('error', path, message)

    def warning(self, path, message):
        self.add('warning', path, message)

    def info(self, path, message):
        self.add('info', path, message)

    def _call(self, path, owner, callback, *args):
        """(callback(*args), True); an exception becomes an error Issue at path and gives (None, False)."""
        try:
            return callback(*args), True
        except Exception as exc:   # a crashing definition callback must not take down validation (GUI, build)
            self.error(path, internal_error_text(owner, exc))
            return None, False

    def _delegate_issues(self, path, owner, callback, *args):
        """Append the Issues returned by a type's validate callback (iterated inside the exception guard)."""
        result, ok = self._call(path, owner, lambda: list(callback(*args) or ()))
        if not ok:
            return
        for issue in result:
            if isinstance(issue, Issue):
                self.append(issue)
            else:
                self.error(path, f'internal error in {owner}: returned {issue!r} instead of an Issue')

    def check_symmetry(self):
        """Symmetry 2.7: geometry drawn beyond a mirror plane / the (r, z) axis vs its mirror image (warnings); none on a
        polar / spherical domain (no mirror plane, its axis is a coordinate line)."""
        if self.vctx.domain_map is not None and self.vctx.domain_map.mapped:
            return
        self._delegate_issues(('model',), 'the symmetry check', symmetry_issues, self.vctx)

    # ---- names (rules of brief 6.3) ----
    def check_names(self):
        spec = self.spec
        entity_names = set()   # geometry objects and selections share one namespace
        for head, category, items in (('geometry', 'geometry object', spec.geometry),
                                      ('selection', 'selection', spec.selections)):
            for item in items:
                path = (head, item.name)
                problem = _name_problem(item.name, category)
                if problem is None and item.name in RESERVED_ENTITY_NAMES:
                    problem = f"'{item.name}' is reserved (outer faces {', '.join(OUTER_FACES_3D)} and 'all')"
                if problem is None and item.name in entity_names:
                    problem = f"duplicate name '{item.name}' (geometry objects and selections share one namespace)"
                if problem is not None:
                    self.error(path, problem)
                entity_names.add(item.name)
        for head, category, items in (('material', 'material', spec.materials), ('physics', 'physics', spec.physics),
                                      ('coupling', 'coupling', spec.couplings), ('solver', 'solver', spec.solvers),
                                      ('derived', 'derived value', spec.derived_values),
                                      ('curve', 'analytic curve', spec.analytic_curves),
                                      ('derived_curve', 'derived curve', spec.derived_curves)):
            self._check_unique_names(items, (head,), category)
        for physics in spec.physics:
            self._check_unique_names(physics.features, ('feature', physics.name), 'feature')

    def _check_unique_names(self, items, path_head, category):
        seen = set()
        for item in items:
            problem = _name_problem(item.name, category)
            if problem is None and item.name in seen:
                problem = f"duplicate {category} name '{item.name}'"
            if problem is not None:
                self.error(path_head + (item.name,), problem)
            seen.add(item.name)

    # ---- parameters and domain ----
    def check_parameters(self):
        path = ('parameters',)
        parameter_errors = self.vctx.parameter_errors
        order = {}
        for index, parameter in enumerate(self.spec.parameters):
            order.setdefault(parameter.name if isinstance(parameter.name, str) else repr(parameter.name), index)
        for name, message in sorted(parameter_errors.items(), key=lambda item: order.get(item[0], len(order))):
            self.error(path, f"parameter '{name}': {message}")
        field_names = self.state_names | frozenset(self.output_names)
        for parameter in self.spec.parameters:
            if parameter.name in field_names and parameter.name not in parameter_errors:
                self.error(path, f"parameter '{parameter.name}' has the name of a physics field; rename it")

    def check_domain(self):
        """Domain size > 0 (size_y null: a 1D model); True when the sizes are valid."""
        domain_ok = True
        for axis in ('size_x', 'size_y'):
            if axis == 'size_y' and self.spec.domain.size_y is None:
                continue   # a 1D model (symmetry 2.20): geometry.domain_map checks the rest
            if self._check_number(('geometry',), f'domain {axis}', getattr(self.spec.domain, axis)) is None:
                domain_ok = False
        if self.spec.domain.size_z is not None:   # spec v2: a 3D model
            if self._check_number(('geometry',), 'domain size_z', self.spec.domain.size_z,
                                  extra=' (or null for a 2D model)') is None:
                domain_ok = False
        return self._check_coordinates() and domain_ok

    def _check_coordinates(self):
        """DomainSpec coordinates: a known grid, its radius_min / angle_min, the angle range (geometry.domain_map);
        a warning for radius_min / angle_min set on a Cartesian grid. True when they are valid."""
        domain = self.spec.domain
        if domain.coordinates not in DOMAIN_COORDINATES:
            self.error(('geometry',), f"domain coordinates {short_repr(domain.coordinates)} is not one of "
                                      f"{', '.join(DOMAIN_COORDINATES)}")
            return False
        if domain.angle_wrap != 'none' and domain.coordinates != 'polar':
            self.error(('geometry',), f"domain angle_wrap {short_repr(domain.angle_wrap)} joins the angle faces of a "
                                      f"polar domain; the {domain.coordinates} grid has none: set it to 'none'")
            return False
        # 1D (size_y None): geometry.domain_map refuses size_z, wraps, angle_min and a slab radius_min below
        if domain.coordinates == 'cartesian' and domain.size_y is not None:
            unused = [name for name in ('radius_min', 'angle_min') if getattr(domain, name) not in (0, 0.0)]
            if unused:
                self.warning(('geometry',), f"domain {' / '.join(unused)} only apply to polar / spherical "
                                            'coordinates; they are ignored on the Cartesian grid')
            return True
        for name in ('radius_min', 'angle_min'):
            _, message = self._scalar(getattr(domain, name))
            if message is not None:
                self.error(('geometry',), f'domain {name}: {message}')
                return False
        try:
            window_map = domain_map(self.spec, self.vctx.parameters)
        except ValueError as exc:
            if self.vctx.domain_size is not None:   # else reported above
                self.error(('geometry',), error_message(exc))
            return False
        return self._check_angle_wrap(window_map)

    def _check_angle_wrap(self, window_map):
        """A phi wrap repeats the sector 2 pi / size_y times: a warning when that is no whole number, an error for an
        antiperiodic wrap unless it is even (the angular Bloch phase pi of a C_N sector). True unless an error."""
        if window_map.angle_wrap == 'none':
            return True
        sectors = 2.0 * math.pi / window_map.size_y
        whole = round(sectors)
        is_whole = abs(sectors - whole) <= 1e-6 * sectors
        if window_map.angle_wrap == 'antiperiodic' and not (is_whole and whole % 2 == 0):
            self.error(('geometry',), f"domain angle_wrap 'antiperiodic' (u changes sign across the wrap: the angular "
                                      f'Bloch phase pi of a C_N sector) needs an even number N = 2 pi / size_y of '
                                      f'sectors, got {sectors:.6g}')
            return False
        if not is_whole:
            self.warning(('geometry',), f"domain angle_wrap {window_map.angle_wrap!r}: 2 pi / size_y = {sectors:.6g} "
                                        'is no whole number of sectors, so the wrapped solution belongs to no full '
                                        'circle (a C_N sector spans 2 pi / N)')
        return True

    # ---- geometry and selections ----
    def check_geometry(self):
        name_counts = Counter(obj.name for obj in self.spec.geometry)
        composite_users = {}   # input object -> composites reading it, in order
        for obj in self.spec.geometry:
            path = ('geometry', obj.name)
            kind = GEOMETRY_KINDS.get(obj.kind)
            own_problem = False
            if kind is not None and isinstance(obj.params, dict):
                own_problem = not self._check_values(path, kind.params, obj.params, 'parameter', frozenset(),
                                                     check_refs=False)
                if kind.composite:
                    for input_name in composite_inputs(kind, obj.params):
                        composite_users.setdefault(input_name, []).append(obj.name)
            if not own_problem and name_counts[obj.name] == 1 and obj.name in self.vctx.geometry_errors:
                self.error(path, self.vctx.geometry_errors[obj.name])
            if kind is not None and isinstance(obj.params, dict):
                self._check_object_dimensions(path, obj, kind)
            shape = self.vctx.shapes.get(obj.name)
            if shape is not None and shape.kind == 'array' and name_counts[obj.name] == 1:
                overlap = array_overlap(obj.name, self.vctx.shapes)
                if overlap is not None:
                    self.warning(path, overlap)
        for input_name, users in composite_users.items():
            for user in users[1:]:
                if user == users[0]:
                    self.error(('geometry', user), f"lists '{input_name}' more than once")
                else:
                    self.error(('geometry', user), f"'{input_name}' is already an input of '{users[0]}'; "
                                                   f"an object can feed one composite only")

    def _check_object_dimensions(self, path, obj, kind):
        """3D-only kinds and set z params in a 2D model are errors; in 3D a point / line z must lie in [0, size_z]."""
        if not self.spec.is_3d:
            if 2 not in kind.dimensions:
                kinds_2d = ', '.join(name for name, other in GEOMETRY_KINDS.items() if 2 in other.dimensions)
                self.error(path, f'a {kind.name} exists only in 3D models; set the domain size_z (3D) or use a '
                                 f'2D kind ({kinds_2d})')
                return
            set_z = [name for name in Z_PARAM_NAMES if z_param_set(name, obj.params.get(name))]
            if set_z:
                self.error(path, f"{', '.join(set_z)} {'is' if len(set_z) == 1 else 'are'} only used in 3D models; "
                                 f"clear {'it' if len(set_z) == 1 else 'them'} or set the domain size_z")
            return
        size_z = self.vctx.size_z
        height, message = (self._scalar(obj.params['z']) if obj.params.get('z') is not None else (None, None))
        if size_z is not None and message is None and height is not None:
            tolerance = 1e-9 * size_z
            if not -tolerance <= height <= size_z + tolerance:
                self.error(path, f'z = {height:g} m lies outside the domain height 0 .. {size_z:g} m')

    def check_selections(self):
        for selection in self.spec.selections:
            path = ('selection', selection.name)
            if selection.kind not in SELECTION_KINDS:
                self.error(path, f"unknown selection kind {selection.kind!r}; valid: {', '.join(SELECTION_KINDS)}")
                continue
            problems, _ = self._entity_problems(selection.entities, selection.kind)
            for problem in problems:
                self.error(path, problem)
            if selection.kind == 'interface':
                if selection.side not in INTERFACE_SIDES:
                    self.error(path, f"side: an interface selection reads one side of the outline: "
                                     f"{' or '.join(INTERFACE_SIDES)}, got {short_repr(selection.side)}")
                if self.spec.dimensions != 2:
                    self.error(path, f'interface selections need a 2D model ({self.spec.dimensions}D: not yet)')
            elif selection.side != '':
                self.error(path, "side applies to 'interface' selections only; clear it")

    def _entity_problems(self, entities, kind, object_kinds=(), allow_outer_faces=True, allow_interior=True):
        """(messages, expanded entity list or None) for an entity list that must hold `kind` entities."""
        if not isinstance(entities, (list, tuple)):
            return [f'expected a list of entity names, got {short_repr(entities)}'], None
        wrong_kind = [name for name in entities
                      if isinstance(name, str) and name in self.selections and self.selections[name].kind != kind]
        if wrong_kind:
            valid = self._valid_entities_text(kind, object_kinds, allow_outer_faces, allow_interior)
            return [f"selection '{name}' is a {self.selections[name].kind} selection; {kind} entities are needed; "
                    f"{valid}" for name in wrong_kind], None
        problems = []
        try:
            expanded = expand_entities(self.spec, entities)
        except ValueError as exc:   # selection cycle or a non-text entity
            return [str(exc)], None
        for name in expanded:
            problem = self._entity_problem(name, kind, object_kinds, allow_outer_faces, allow_interior)
            if problem is not None:
                problems.append(problem)
        return problems, expanded

    def _entity_problem(self, name, kind, object_kinds, allow_outer_faces, allow_interior):
        """Message when one expanded entity does not fit a `kind` selection (ending in the valid names), else None."""
        def with_valid_names(message):
            return f'{message}; {self._valid_entities_text(kind, object_kinds, allow_outer_faces, allow_interior)}'

        if name == 'all':
            if kind != 'domain':
                return with_valid_names(f"'all' (the whole domain) is not {_ENTITY_KIND_TEXT[kind]}")
            if object_kinds:
                return with_valid_names(f"'all' is not allowed here; select a {' or '.join(object_kinds)} object")
            return None
        if name in outer_faces(self.spec):
            if kind != 'boundary':
                return with_valid_names(f"outer face '{name}' is not {_ENTITY_KIND_TEXT[kind]}")
            if not allow_outer_faces:
                return with_valid_names(f"outer face '{name}' is not allowed here; select a domain object perimeter")
            return None
        obj = self.objects.get(name)
        if obj is None:
            return with_valid_names(f"unknown entity '{name}'")
        geometry_kind = GEOMETRY_KINDS.get(obj.kind)
        if geometry_kind is None:
            return None   # unknown geometry kind: reported at the geometry object
        needed_entity_kind = 'domain' if kind in ('domain', 'boundary', 'interface') else kind
        if geometry_kind.entity_kind != needed_entity_kind:
            return with_valid_names(f"'{name}' is a {obj.kind}, not {_ENTITY_KIND_TEXT[kind]}")
        if kind == 'boundary' and not allow_interior:
            return with_valid_names(f"'{name}': object perimeters are not allowed here; select an outer face")
        if name in self.consumed:
            return with_valid_names(f"'{name}' is an input of a composite object and cannot be selected on its own")
        if object_kinds and obj.kind not in object_kinds:
            return with_valid_names(f"'{name}' is a {obj.kind}; allowed here: {', '.join(object_kinds)}")
        return None

    def _valid_entities_text(self, kind, object_kinds, allow_outer_faces, allow_interior):
        """'valid <kind> entities: a, b' (or 'none') for the end of a selection message."""
        valid = self._valid_entity_names(kind, object_kinds, allow_outer_faces, allow_interior)
        return f"valid {kind} entities: {', '.join(valid) or 'none'}"

    def _valid_entity_names(self, kind, object_kinds, allow_outer_faces, allow_interior):
        """Sorted names a `kind` entity list accepts here: 'all', outer faces and objects that fit, plus the
        selections of that kind."""
        valid = set()
        for name in _ENTITY_LISTINGS[kind](self.spec):
            if name == 'all':
                accepted = not object_kinds
            elif name in OUTER_FACES_3D:
                accepted = allow_outer_faces
            else:   # a geometry object of the right entity kind (composite inputs already left out)
                accepted = ((kind != 'boundary' or allow_interior)
                            and (not object_kinds or self.objects[name].kind in object_kinds))
            if accepted and isinstance(name, str):
                valid.add(name)
        valid.update(selection.name for selection in self.spec.selections
                     if selection.kind == kind and isinstance(selection.name, str))
        return sorted(valid)

    # ---- materials ----
    def check_materials(self):
        property_params = {param.name: param for _, param in self.registry.material_property_params()}
        present_reads = self._present_material_reads()
        expanded_selections = []
        for material in self.spec.materials:
            path = ('material', material.name)
            library = material.library   # None or '' = no library (as materials.effective_material_data)
            if library not in (None, '') and (not isinstance(library, str) or library not in MATERIAL_LIBRARY):
                self.error(path, f"unknown library material {library!r}; available: {', '.join(MATERIAL_LIBRARY)}")
            properties = material.properties if isinstance(material.properties, dict) else {}
            if properties is not material.properties:
                self.error(path, f'properties must be a mapping of name -> value, got '
                                 f'{short_repr(material.properties)}')
            if material.dispersion is not None and not isinstance(material.dispersion, (list, tuple)):
                self.error(path, f'dispersion must be a list of pole dicts or None, got '
                                 f'{short_repr(material.dispersion)}')
            elif crystal_entries(material.dispersion):
                self._check_crystal_entries(path, crystal_entries(material.dispersion))
            for key, raw in properties.items():
                param = property_params.get(key)
                if param is None:
                    known = ', '.join(property_params) or 'none (no physics type registered)'
                    self.error(path, f"unknown material property {key!r}; known: {known}")
                    continue
                if raw is None:
                    continue   # unset: the library value (if any) applies
                message = self._value_message(param, raw, self.state_names)
                if message is not None:
                    self.error(path, f"property '{key}': {message}")
            if present_reads is not None:
                self._check_unread_properties(path, properties, property_params, *present_reads)
            problems, expanded = self._entity_problems(material.selection, 'domain')
            for problem in problems:
                self.error(path, f'selection: {problem}')
            if expanded == [] and not problems:
                self.warning(path, 'not assigned to any domain; it has no effect')
            if expanded and not problems:
                self._check_object_cells(path, expanded)
            expanded_selections.append(None if problems else expanded)
        self._check_material_overrides(expanded_selections)

    def _check_crystal_entries(self, path, entries):
        """Errors for more than one crystal entry, a malformed one, and a plane that does not fit the model: 'sagittal'
        (the default) in 2D, 'full' (written out) in 3D. Called only with entries: the import is local so the selftest
        cache ties crystal_tensors to crystal models only."""
        from ..materials.crystal_tensors import FULL_PLANE, SAGITTAL_PLANE, entry_settings
        if len(entries) > 1:
            self.error(path, f'only one crystal entry per material, got {len(entries)}')
        for entry in entries:
            try:
                _, _, plane = entry_settings(entry)
            except (ValueError, TypeError) as exc:   # TypeError: an unhashable value from a hand-edited file
                self.error(path, str(exc))
                continue
            if self.spec.is_3d and entry.get('plane') != FULL_PLANE:
                self.error(path, f"crystal entry: a 3D model expands the crystal in all three axes: set 'plane': "
                                 f"'{FULL_PLANE}' (model x, y, z = the rotated X1', X2', X3'; got "
                                 f"{entry.get('plane', SAGITTAL_PLANE)!r})")
            elif not self.spec.is_3d and plane != SAGITTAL_PLANE:
                self.error(path, f"crystal entry: plane '{plane}' is for 3D models; a 2D model takes 'plane': "
                                 f"'{SAGITTAL_PLANE}' (model x, y, z = X1', X3', -X2')")

    def _present_material_reads(self):
        """(present physics type names, material property names they read), or None when the model has no physics or
        an unregistered physics type (then nothing is known to be unread)."""
        present_types = {}
        for physics in self.spec.physics:
            try:
                present_types.setdefault(physics.type, self.registry.physics_type(physics.type))
            except (KeyError, TypeError):   # unknown type: reported by check_physics
                return None
        if not present_types:
            return None
        read_names = list(dict.fromkeys(param.name for physics_type in present_types.values()
                                        for param in physics_type.material_properties))
        return list(present_types), read_names

    def _check_unread_properties(self, path, properties, property_params, type_names, read_names):
        """Warn for user-set properties no present physics type declares (the builder ignores them); permittivity
        names are left to the EM types' own hint when any present type reads permittivity."""
        reads_permittivity = any(name in PERMITTIVITY_PROPERTY_NAMES for name in read_names)
        unread = [key for key, raw in properties.items()
                  if raw is not None and key in property_params and key not in read_names
                  and not (reads_permittivity and key in PERMITTIVITY_PROPERTY_NAMES)]
        if unread:
            self.warning(path, f"{', '.join(unread)} {'is' if len(unread) == 1 else 'are'} not read by the present "
                               f"physics ({', '.join(type_names)}) and ignored; they read: "
                               f"{', '.join(read_names) or 'no material property'}")

    def _check_material_overrides(self, expanded_selections):
        """Warn for materials whose cells are all taken by winning materials (geometry.material_paint_order)."""
        materials = self.spec.materials
        for index, (material, entities) in enumerate(zip(materials, expanded_selections)):
            if not entities:
                continue
            path = ('material', material.name)
            is_background = 'all' in entities
            if is_background:
                later_background = next((later.name for later, later_entities
                                         in zip(materials[index + 1:], expanded_selections[index + 1:])
                                         if later_entities and 'all' in later_entities), None)
                if later_background is not None:
                    self.warning(path, f"fully overridden by the later background material '{later_background}' "
                                       f"(both select 'all'; the later one wins)")
                    continue
            preview = self.vctx.material_map()
            if preview is not None and np.any(preview[0] == index + 1):
                continue   # wins a preview pixel
            overridden, ok = self._call(path, 'the material override check', self._overridden_on_mesh, index,
                                        expanded_selections)
            if not ok:
                continue
            reason = ("fully covered by geometry materials (a material selecting 'all' fills only the cells no other "
                      "material covers)" if is_background else
                      'fully covered by later materials (later materials win where selections overlap)')
            if overridden is not None:
                if overridden:
                    self.warning(path, f'{reason}; it has no effect')
                continue
            if preview is None or self.spec.dimensions == 3:   # the preview is an xy footprint: no z ends
                continue
            index_image, _, extent = preview   # too many mesh cells to test: judge on the coarse preview
            if self._preview_coverage(entities, index_image.shape, extent).any():
                pixel_width = (extent[1] - extent[0]) / index_image.shape[0]
                self.warning(path, f'{reason} on the validation preview (pixels ~{pixel_width:.2g} m, coarser than '
                                   f'the mesh); it probably has no effect')

    def _overridden_on_mesh(self, index, expanded_selections):
        """True when the materials painted after material index take every window cell centre of the mesh estimate it
        selects (the build's rule), False when it keeps one or selects none, None without a mesh estimate or above
        MAX_COVERAGE_TEST_CELLS rasterised cells."""
        mesh = self._coverage_mesh()
        if mesh is None:
            return None
        shapes = self.vctx.shapes
        x_edges, y_edges = _window_edges(mesh)
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        own_entities = expanded_selections[index]
        order = material_paint_order(expanded_selections)
        winner_entities = [entity for later in order[order.index(index) + 1:] for entity in expanded_selections[later]]
        if 'all' in winner_entities:
            return True
        own_objects = [] if 'all' in own_entities else self._domain_objects(own_entities)
        winner_objects = list(dict.fromkeys(self._domain_objects(winner_entities)))
        if mesh.domain_map is not None:   # polar / spherical: every window cell at its drawing-plane centre
            if 'all' not in own_entities and not own_objects:
                return False
            x_mapped, y_mapped = _mapped_window_centres(mesh)
            if x_mapped.size * (1 + len(own_objects) + len(winner_objects)) > MAX_COVERAGE_TEST_CELLS:
                return None
            kept = np.ones(x_mapped.shape, dtype=bool)
            if own_objects:
                kept = np.any([rasterize(name, shapes, x_mapped, y_mapped, np) for name in own_objects], axis=0)
            if not kept.any():
                return False
            for name in winner_objects:
                kept &= ~rasterize(name, shapes, x_mapped, y_mapped, np)
            return not kept.any()
        if mesh.z_edges is not None:   # 3D: objects end in z, so a full footprint need not cover every layer
            return self._overridden_in_3d(mesh, x_centres, y_centres, own_entities, own_objects, winner_objects)
        if 'all' not in own_entities:
            if not own_objects:
                return False   # no resolved domain object: reported by the selection / geometry checks
            boxes = [bounding_box(name, shapes) for name in own_objects]   # crop to the material's bounding box
            x_centres = x_centres[np.searchsorted(x_centres, min(box[0] for box in boxes), side='left'):
                                  np.searchsorted(x_centres, max(box[2] for box in boxes), side='right')]
            y_centres = y_centres[np.searchsorted(y_centres, min(box[1] for box in boxes), side='left'):
                                  np.searchsorted(y_centres, max(box[3] for box in boxes), side='right')]
            if len(x_centres) == 0 or len(y_centres) == 0:
                return False   # outside the domain: the empty-object check reports it
        tested_cells = len(x_centres) * len(y_centres) + sum(
            _rasterised_cell_count(name, shapes, x_centres, y_centres) for name in own_objects + winner_objects)
        if tested_cells > MAX_COVERAGE_TEST_CELLS:
            return None
        if own_objects:
            kept = np.zeros((len(x_centres), len(y_centres)), dtype=bool)
            for name in own_objects:
                window, mask = rasterize_window(name, shapes, x_centres, y_centres, np)
                kept[window] |= mask
        else:
            kept = np.ones((len(x_centres), len(y_centres)), dtype=bool)
        if not kept.any():
            return False   # selects no cell centre: the empty-object check reports it
        for name in winner_objects:
            window, mask = rasterize_window(name, shapes, x_centres, y_centres, np)
            kept[window] &= ~mask
        return not kept.any()

    def _overridden_in_3d(self, mesh, x_centres, y_centres, own_entities, own_objects, winner_objects):
        """_overridden_on_mesh on a 3D tensor window: the same test on the (x, y, z) window cell centres, one
        rasterize_box per object; None above MAX_COVERAGE_TEST_CELLS."""
        if 'all' not in own_entities and not own_objects:
            return False   # no resolved domain object: reported by the selection / geometry checks
        shapes = self.vctx.shapes
        sides = mesh.npml_sides
        z_edges = mesh.z_edges[sides['z_lo']:mesh.nz - sides['z_hi'] + 1]
        centres = (x_centres.astype(np.float32), y_centres.astype(np.float32),
                   (0.5 * (z_edges[:-1] + z_edges[1:])).astype(np.float32))
        window_shape = tuple(len(axis) for axis in centres)
        object_cells = sum(int(np.prod([axis.stop - axis.start for axis in box_window(name, shapes, *centres)]))
                           for name in own_objects + winner_objects)
        if int(np.prod(window_shape)) + object_cells > MAX_COVERAGE_TEST_CELLS:
            return None
        kept = np.zeros(window_shape, dtype=bool) if own_objects else np.ones(window_shape, dtype=bool)
        for name in own_objects:
            window, mask = rasterize_box(name, shapes, *centres, np)
            kept[window] |= mask
        if not kept.any():
            return False   # selects no cell centre: the empty-object check reports it
        for name in winner_objects:
            window, mask = rasterize_box(name, shapes, *centres, np)
            kept[window] &= ~mask
        return not kept.any()

    def _domain_objects(self, entities):
        """Resolved domain-object names among expanded entities ('all', outer faces and failed objects dropped)."""
        shapes = self.vctx.shapes
        return [name for name in entities
                if name in shapes and GEOMETRY_KINDS[shapes[name].kind].entity_kind == 'domain']

    def _preview_coverage(self, entities, shape, extent):
        """Bool raster (material_preview grid) of the cells the expanded entities cover."""
        x_centres, y_centres = preview_pixel_centers(extent, shape)
        object_mask = domain_object_rasterizer(self.vctx.shapes, x_centres, y_centres, np)
        return paint_material_index([entities], object_mask, shape, np) > 0

    # ---- objects that select no mesh cell ----
    def _coverage_mesh(self):
        """The tensor estimate of vctx.mesh (the default mesh, or the current physics' inside on_mesh) the coverage
        checks rasterise on; None without one (failed estimate, a triangle mesh)."""
        mesh = self.vctx.mesh
        return mesh if mesh is not None and mesh.kind == 'tensor' else None

    def _check_object_cells(self, path, entities):
        """Warn at path for each selected domain object that contains no window cell centre of the mesh estimate."""
        if self._coverage_mesh() is None:
            return
        for name in entities:
            shape = self.vctx.shapes.get(name)
            if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
                continue   # 'all', outer faces, failed objects: reported elsewhere or always non-empty
            key = (self.vctx.mesh_name, name)
            if key not in self.empty_object_messages:
                message, _ = self._call(path, 'the empty-object check', self._empty_object_message, name)
                self.empty_object_messages[key] = message
            if self.empty_object_messages[key] is not None:
                self.warning(path, self.empty_object_messages[key])

    def _empty_object_message(self, name):
        """Warning text when domain object name lies outside the domain or contains no window cell centre (float64
        centres of the mesh estimate), else None; None also above MAX_COVERAGE_TEST_CELLS rasterised cells."""
        if self.vctx.mesh.domain_map is not None:
            return self._empty_mapped_object_message(name)
        x_edges, y_edges = _window_edges(self.vctx.mesh)
        size_x, size_y = float(x_edges[-1]), float(y_edges[-1])
        x0, y0, x1, y1 = bounding_box(name, self.vctx.shapes)
        if x1 < 0.0 or y1 < 0.0 or x0 > size_x or y0 > size_y:
            return (f"'{name}' lies outside the domain (x 0 to {size_x:.4g} m, y 0 to {size_y:.4g} m), so it "
                    f"selects no cells; move it inside")
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        if _rasterised_cell_count(name, self.vctx.shapes, x_centres, y_centres) > MAX_COVERAGE_TEST_CELLS:
            return None   # far too many cells to be empty in practice; not worth the CPU time
        _, mask = rasterize_window(name, self.vctx.shapes, x_centres, y_centres, np)
        if mask.any():
            return None
        column = int(np.clip(np.searchsorted(x_edges, 0.5 * (x0 + x1), side='right') - 1, 0, len(x_centres) - 1))
        row = int(np.clip(np.searchsorted(y_edges, 0.5 * (y0 + y1), side='right') - 1, 0, len(y_centres) - 1))
        cell_width = float(x_edges[column + 1] - x_edges[column])
        cell_height = float(y_edges[row + 1] - y_edges[row])
        cell_text = (f'{cell_width:.2g} m' if math.isclose(cell_width, cell_height, rel_tol=1e-6)
                     else f'{cell_width:.2g} x {cell_height:.2g} m')
        return (f"'{name}' contains no cell centre (cells there ~{cell_text}), so it selects no cells; enlarge it "
                f"or refine the mesh around it (a mesh refinement on '{name}', nonuniform / auto mode)")

    def _empty_mapped_object_message(self, name):
        """_empty_object_message on a polar / spherical estimate (drawing-plane cell centres of the whole window)."""
        mesh = self.vctx.mesh
        x_mapped, y_mapped = _mapped_window_centres(mesh)
        if x_mapped.size > MAX_COVERAGE_TEST_CELLS:
            return None
        if rasterize(name, self.vctx.shapes, x_mapped, y_mapped, np).any():
            return None
        x0, y0, x1, y1 = bounding_box(name, self.vctx.shapes)
        corners_x, corners_y = np.array([x0, x1, x0, x1, 0.5 * (x0 + x1)]), np.array([y0, y0, y1, y1, 0.5 * (y0 + y1)])
        if not mesh.domain_map.contains(corners_x, corners_y).any():
            return (f"'{name}' lies outside the {mesh.domain_map.kind} domain (r {mesh.domain_map.radius_min:.4g} to "
                    f'{mesh.domain_map.radius_max:.4g} m), so it selects no cells; move it inside')
        return (f"'{name}' contains no cell centre (cells ~{mesh.dx_min:.2g} to {mesh.dx_max:.2g} m), so it selects "
                'no cells; enlarge it or refine the mesh (smaller dx, or a mesh axis x band)')

    # ---- physics and features ----
    def _physics_mesh(self, physics, physics_type=None):
        """(mesh name the physics' checks see, or None: unknown mesh name or a mesh kind its type (and feature_type,
        when given) does not run on, so no estimate)."""
        mesh_spec = self.spec.find_mesh(physics)
        if mesh_spec is None:
            return None
        if physics_type is not None and mesh_spec.kind not in physics_type.grids:
            return None
        return mesh_spec.name

    def check_physics(self):
        boundary_providers = {}   # mesh name -> [(physics name, boundary config)] of the physics defining one
        for physics in self.spec.physics:
            path = ('physics', physics.name)
            try:
                physics_type = self.registry.physics_type(physics.type)
            except KeyError as exc:
                self.error(path, error_message(exc))
                continue
            model_dimensions = self.spec.dimensions
            if model_dimensions not in physics_type.dimensions:
                self.mesh_inputs_ok = False
                remedy = {3: 'clear the domain size_z (2D model)', 1: 'set the domain size_y (2D model)'}.get(
                    model_dimensions, 'set the domain size_z (3D model)' if 3 in physics_type.dimensions
                    else 'clear the domain size_y (1D model)')
                self.error(path, f"physics type '{physics_type.name}' runs only in "
                                 f"{' / '.join(f'{n}D' for n in physics_type.dimensions)} models, this model is "
                                 f'{model_dimensions}D; {remedy} or use a {model_dimensions}D type')
                continue
            self._check_every(path, physics.every, allow_none=False)
            mesh_name = self._check_physics_mesh(path, physics, physics_type)
            values = None
            if self._check_values(path, physics_type.settings, physics.settings, 'setting', self.state_names):
                values = self.vctx.values(physics_type.settings, physics.settings)
            self.physics_entries.append((physics, physics_type, values))
            if values is None:
                self.mesh_inputs_ok = False
                continue
            self._check_coordinate_grid(path, physics_type, values)
            if physics_type.scalar_outputs is not None:
                self._check_scalar_outputs(path, physics, physics_type, values)
            boundary_problem = None
            if physics_type.domain_requirements is not None:
                boundary_config, boundary_problem = self._boundary_config_status(path, physics_type, values)
                if boundary_config is not None:
                    boundary_providers.setdefault(mesh_name, []).append((physics.name, boundary_config))
            first_new_issue = len(self.issues)
            if physics_type.validate is not None:
                with self.vctx.on_mesh(self._physics_mesh(physics, physics_type)):
                    self._delegate_issues(path, f"physics type '{physics_type.name}' validate",
                                          physics_type.validate, self.vctx, physics, values)
            if boundary_problem is not None:
                self.mesh_inputs_ok = False   # the mesh estimate fails on the same problem: report it here only
                if not any(issue.level == 'error' and issue.path == path
                           for issue in self.issues[first_new_issue:]):   # the type's validate did not explain it
                    self.error(path, f'invalid boundary configuration: {boundary_problem}')
        for providers in boundary_providers.values():   # per mesh: its physics merge into one boundary config
            first_name, first_config = providers[0]
            for name, boundary_config in providers[1:]:
                path = ('physics', name)
                difference_text, ok = self._call(path, 'the boundary configuration comparison',
                                                 boundary_config_difference_text, first_config, boundary_config)
                if not ok:
                    self.mesh_inputs_ok = False   # the mesh estimate hits the same exception
                elif difference_text:
                    self.mesh_inputs_ok = False
                    self.error(path, f"its boundary configuration differs from that of '{first_name}' in "
                                     f"{difference_text}; physics sharing a mesh must use equal boundaries "
                                     f"and PML settings (bc_* and pml_* settings)")

    def _check_physics_mesh(self, path, physics, physics_type):
        """PhysicsSpec.mesh names a mesh (error otherwise) of a kind the type runs on (PhysicsType.grids); returns the
        mesh name (None for an unknown one)."""
        if not isinstance(physics.mesh, str):
            self.error(path, f'mesh must be a mesh name (text), got {short_repr(physics.mesh)}')
            return None
        try:
            mesh_spec = self.spec.mesh_of(physics)
        except KeyError:
            self.error(path, f"mesh {physics.mesh!r} does not exist; meshes: "
                             f"{_name_list(mesh_spec.name for mesh_spec in self.spec.meshes)} ('' = the first)")
            return None
        if mesh_spec.kind in MESH_KINDS and mesh_spec.kind not in physics_type.grids:
            self.error(path, f"{physics_type.label} does not run on a {mesh_spec.kind} mesh (mesh "
                             f"'{mesh_spec.name}'); it runs on {' or '.join(physics_type.grids)} meshes: assign it "
                             'another mesh (physics mesh) or change the mesh kind')
        return mesh_spec.name

    def _check_coordinate_grid(self, path, physics_type, values):
        """A polar / spherical domain needs a type whose coordinate_grids hold 'polar', set axisymmetric exactly on a
        spherical one (the space fixes the drawing's map)."""
        window_map = self.vctx.domain_map
        if window_map is None or not window_map.mapped:
            return
        kind = window_map.kind
        if window_map.kind == 'cartesian':   # a 1D slab: Cartesian cells, planar
            if is_axisymmetric(values):
                self.error(path, "a 1D slab (coordinates cartesian) needs geometry 'planar' in every physics; a 1D "
                                 'cylinder is coordinates polar')
            return
        if 'polar' not in physics_type.coordinate_grids:
            polar_types = ('heat (a 3D cylindrical (r, phi, z) sector)' if self.spec.is_3d
                           else 'heat, electric_currents, electrostatics')
            self.error(path, f"physics type '{physics_type.name}' solves on Cartesian grids only; this domain has "
                             f"coordinates '{kind}': set the domain coordinates to cartesian or use a type that "
                             f'supports polar grids ({polar_types})')
        elif is_axisymmetric(values) != window_map.axisymmetric:
            wanted = "'axisymmetric' (spherical (r, theta))" if window_map.axisymmetric else "'planar' (polar (r, phi))"
            self.error(path, f"a domain with coordinates '{kind}' needs geometry {wanted} in every physics: the "
                             'coordinates fix how the drawing extends into 3D')

    def _boundary_config_status(self, path, physics_type, values):
        """(boundary_config of the domain_requirements or None, ValueError message of an invalid one or None)."""
        try:
            requirements = physics_type.domain_requirements(values)
        except ValueError as exc:
            return None, str(exc)
        except Exception as exc:
            self.mesh_inputs_ok = False   # the mesh estimate hits the same exception
            self.error(path, internal_error_text(f"physics type '{physics_type.name}' domain_requirements", exc))
            return None, None
        return getattr(requirements, 'boundary_config', None), None

    def _check_scalar_outputs(self, path, physics, physics_type, values):
        """Scalar output names (E5): identifiers, unique, not a field of the same type, no parameter named
        '<physics>_<scalar>'; valid ones join self.scalar_names."""
        owner = f"physics type '{physics_type.name}' scalar_outputs"
        scalars, ok = self._call(path, owner, lambda: list(physics_type.scalar_outputs(values, physics)))
        if not ok:
            return
        field_names = {info.name for info in physics_type.state_fields + physics_type.output_fields}
        parameter_names = {parameter.name for parameter in self.spec.parameters}
        seen = set()
        for scalar in scalars:
            if not is_identifier(scalar) or scalar in seen or scalar in field_names:
                problem = ('not an identifier' if not is_identifier(scalar) else
                           'listed twice' if scalar in seen else 'also a field of this physics type')
                self.error(path, f'internal error in {owner}: scalar {short_repr(scalar)} is {problem}')
                continue
            seen.add(scalar)
            qualified = f'{physics.name}_{scalar}'
            if qualified in parameter_names:
                self.error(('parameters',), f"parameter '{qualified}' has the name of a scalar output of physics "
                                            f"'{physics.name}'; rename it")
            self.scalar_names.add(qualified)

    def check_features(self):
        for physics, physics_type, _ in self.physics_entries:
            field_names = {info.name for info in physics_type.output_fields}
            mesh_name = self._physics_mesh(physics, physics_type)
            for feature in physics.features:
                if not feature.enabled:
                    continue
                path = ('feature', physics.name, feature.name)
                try:
                    feature_type = physics_type.feature_type(feature.type)
                except KeyError as exc:
                    self.error(path, error_message(exc))
                    continue
                if feature_type.result_kind is not None and feature.name in field_names:
                    self.error(path, f"a result feature named like the output field '{physics.name}."
                                     f"{feature.name}' (one key for figures/{physics.name}.{feature.name}.png and "
                                     f"Results.figure); rename the feature")
                errors_before = self.n_errors
                kind = self.spec.mesh_by_name(mesh_name).kind if mesh_name is not None else None
                if kind is not None and kind not in feature_type.grids:
                    self.error(path, f"{feature_type.label} does not work on a {kind} mesh (mesh '{mesh_name}' of "
                                     f"'{physics.name}'); it works on {' or '.join(feature_type.grids)} meshes")
                with self.vctx.on_mesh(mesh_name if kind is None or kind in feature_type.grids else None):
                    self._check_values(path, feature_type.params, feature.params, 'parameter', self.state_names)
                    self._check_feature_selection(path, feature_type, feature.selection)
                    self._check_physics_refs(path, physics, feature_type, feature.params)
                    if (feature_type.requires_uniform_mesh and self.vctx.mesh is not None
                            and not self.vctx.mesh_uniform):
                        self.error(path, f"{feature_type.label} needs a uniform mesh; set the mode of mesh "
                                         f"'{mesh_name}' to uniform")
                    if self.n_errors == errors_before and feature_type.validate is not None:
                        values = self.vctx.values(feature_type.params, feature.params)
                        if values is not None:
                            self._delegate_issues(path, f"feature type '{feature_type.name}' validate",
                                                  feature_type.validate, self.vctx, physics, feature, values)

    def _check_physics_refs(self, path, physics, feature_type, raw_values):
        """S7.2: a feature Param naming another physics (ref_kind 'physics') reads it on its own grid, so both physics
        must share one mesh."""
        if not isinstance(raw_values, dict):
            return
        own_mesh = self._mesh_name_of(physics)
        for param in feature_type.params:
            if param.ref_kind != 'physics' or param.name not in raw_values:
                continue
            value = raw_values[param.name]
            for other_name in value if isinstance(value, (list, tuple)) else (value,):
                if not isinstance(other_name, str):   # a malformed value: reported by _check_values
                    continue
                other = self.physics_by_name.get(other_name)
                other_mesh = self._mesh_name_of(other) if other is not None else None
                if own_mesh is not None and other_mesh is not None and other_mesh != own_mesh:
                    self.error(path, f"{param.name} '{other_name}' runs on mesh '{other_mesh}', '{physics.name}' on "
                                     f"'{own_mesh}': a feature reading another physics needs both on one mesh "
                                     f"(set the mesh of '{other_name}' to '{own_mesh}' or the reverse)")

    def _mesh_name_of(self, physics):
        """Name of the physics' mesh, None for an unknown mesh name (reported at the physics)."""
        mesh_spec = self.spec.find_mesh(physics)
        return None if mesh_spec is None else mesh_spec.name

    def _check_feature_selection(self, path, feature_type, selection):
        kind = feature_type.selection_kind
        if kind is None:
            if selection:
                self.warning(path, f'{feature_type.label} takes no selection; the selection is ignored')
            return
        problems, expanded = self._entity_problems(selection, kind, feature_type.object_kinds,
                                                   feature_type.allow_outer_faces, feature_type.allow_interior)
        for problem in problems:
            self.error(path, f'selection: {problem}')
        if expanded is None:
            return
        if not problems and kind in ('domain', 'boundary'):
            self._check_object_cells(path, expanded)
        count = len(expanded)
        wanted = (' or '.join(feature_type.object_kinds) + ' object') if feature_type.object_kinds \
            else _ENTITY_KIND_TEXT[kind]
        if count == 0 and feature_type.min_entities > 0:
            valid = self._valid_entities_text(kind, feature_type.object_kinds, feature_type.allow_outer_faces,
                                              feature_type.allow_interior)
            self.error(path, f'selection is empty; select {wanted} ({valid})')
        elif count < feature_type.min_entities:
            self.error(path, f'selection needs at least {feature_type.min_entities} entities, has {count}')
        if feature_type.max_entities is not None and count > feature_type.max_entities:
            self.error(path, f'selection allows at most {feature_type.max_entities} '
                             f"entit{'y' if feature_type.max_entities == 1 else 'ies'} ({wanted}), has {count}")

    # ---- couplings ----
    def check_couplings(self):
        slot_writers = {}   # (target, slot) -> [(coupling spec, contribution)] of the couplings that pass their checks
        for coupling in self.spec.couplings:
            if not coupling.enabled:
                continue
            path = ('coupling', coupling.name)
            try:
                coupling_type = self.registry.coupling_type(coupling.type)
            except KeyError as exc:
                self.error(path, error_message(exc))
                continue
            errors_before = self.n_errors
            for role, physics_name in (('source', coupling.source), ('target', coupling.target)):
                physics = self.physics_by_name.get(physics_name)
                if physics is None:
                    problem = (f'no {role} physics chosen' if physics_name is None
                               else f"{role} physics '{physics_name}' does not exist")
                    self.error(path, f'{problem}; {self._physics_choices(role, coupling_type)}')
                    continue
                if not coupling_type_accepts(coupling_type, role, physics.type):
                    self.error(path, f"the {role} must be a {coupling_role_text(coupling_type, role)} physics; "
                                     f"'{physics_name}' is '{physics.type}'; "
                                     f"{self._physics_choices(role, coupling_type)}")
            self._check_every(path, coupling.every, allow_none=True)
            self._check_values(path, coupling_type.params, coupling.params, 'parameter', _COUPLING_EXPRESSION_NAMES)
            if self.n_errors == errors_before:
                self._check_coupling_meshes(path, coupling, coupling_type)
            if self.n_errors == errors_before and coupling_type.validate is not None:
                values = self.vctx.values(coupling_type.params, coupling.params)
                if values is not None:
                    source = self.physics_by_name[coupling.source]
                    with self.vctx.on_mesh(self._physics_mesh(source, self._registered_type(source))):
                        self._delegate_issues(path, f"coupling type '{coupling_type.name}' validate",
                                              coupling_type.validate, self.vctx, coupling, values)
            if self.n_errors == errors_before:
                written, ok = self._call(path, f"coupling type '{coupling_type.name}' writes", coupling_writes,
                                         self.vctx, coupling)
                for slot, contribution in (written or {}).items() if ok else ():
                    slot_writers.setdefault((coupling.target, slot), []).append((coupling, contribution))
        self._check_slot_writers(slot_writers)

    def _registered_type(self, physics):
        """PhysicsType of a physics spec, None for an unknown type (reported at the physics)."""
        try:
            return self.registry.physics_type(physics.type)
        except KeyError:
            return None

    def _check_coupling_meshes(self, path, coupling, coupling_type):
        """S7.2: source and target on different meshes need a CouplingType.mesh_transfer (error naming both physics and
        the fix); with one, an info names the transfer the build makes."""
        source_mesh = self._mesh_name_of(self.physics_by_name[coupling.source])
        target_mesh = self._mesh_name_of(self.physics_by_name[coupling.target])
        if source_mesh is None or target_mesh is None or source_mesh == target_mesh:
            return
        if coupling_type.mesh_transfer is None:
            self.error(path, f"source '{coupling.source}' runs on mesh '{source_mesh}', target '{coupling.target}' on "
                             f"'{target_mesh}': a {coupling_type.label} coupling works within one mesh only; put both "
                             'physics on one mesh (physics mesh)')
            return
        how = ('the source field is interpolated onto the target mesh' if coupling_type.mesh_transfer == 'field'
               else 'its result moves conservatively (totals kept)' if coupling_type.value_kind == 'conservative'
               else 'its result is interpolated onto the target mesh')
        self.info(path, f"transfers from mesh '{source_mesh}' to mesh '{target_mesh}': {how}")

    def _check_slot_writers(self, slot_writers):
        """Writer rule (E13): a slot takes one overwriting coupling, or any number of contributors of one kind ('sum' /
        'index') sharing one every; errors at each coupling involved."""
        for (target, slot), writers in slot_writers.items():
            if len(writers) < 2:
                continue
            names = _name_list(coupling.name for coupling, _ in writers)
            where = f"slot '{slot}' of '{target}' (written by {names})"
            kinds = {contribution for _, contribution in writers}
            if '' in kinds:
                message = (f'{where}: an overwriting coupling replaces the slot at every transfer, so it cannot share '
                           'it; keep one coupling per slot, or only couplings that add up (e.g. em_heat_joule with '
                           'currents_heat_joule, heat_em_thermo_optic with another index coupling)')
            elif len(kinds) > 1:
                message = f"{where}: contributions of different kinds ({', '.join(sorted(kinds))}) cannot be combined"
            else:
                everies = {coupling.name: effective_coupling_every(coupling, self.physics_by_name[coupling.source],
                                                                   self.physics_by_name[target])
                           for coupling, _ in writers}
                if len(set(everies.values())) <= 1:
                    continue
                message = (f"{where}: contributors to one slot share one every, got "
                           f"{', '.join(f'{name} {every}' for name, every in everies.items())} (leave every blank on "
                           'all or set the same value)')
            for coupling, _ in writers:
                self.error(('coupling', coupling.name), message)

    def _physics_choices(self, role, coupling_type):
        """'valid <role> physics [of type ...]: names' over the model's physics the coupling type accepts in role."""
        names = _name_list(physics.name for physics in self.spec.physics
                           if coupling_type_accepts(coupling_type, role, physics.type))
        any_type = getattr(coupling_type, f'{role}_type') is None
        type_text = '' if any_type else f' of type {coupling_role_text(coupling_type, role)}'
        return f'valid {role} physics{type_text}: {names}'

    # ---- meshes ----
    def check_meshes(self, domain_ok):
        """ModelSpec.meshes (S7.2): names, each mesh's settings, refinements, axes and kind rules, its estimate
        (failure, notes, auto rule) at ('mesh', name), unused meshes (warning) and the one-mesh rules."""
        spec = self.spec
        if not spec.meshes:
            self.error(('meshes',), 'the model has no mesh: add one (meshes)')
            return
        self._check_unique_names(spec.meshes, ('mesh',), 'mesh')
        used = {mesh_spec.name for mesh_spec in spec.used_meshes()}
        checked = set()
        for mesh_spec in spec.meshes:
            if isinstance(mesh_spec.name, str) and mesh_spec.name not in checked:   # a repeated name: reported above
                checked.add(mesh_spec.name)
                self._check_mesh(('mesh', mesh_spec.name), mesh_spec, domain_ok, mesh_spec.name in used)
        self._check_one_mesh_rules(used)

    def _check_mesh(self, path, mesh_spec, domain_ok, used):
        errors_before = self.n_errors
        raw_values = {param.name: getattr(mesh_spec, param.name) for param in MESH_PARAMS}
        self._check_values(path, MESH_PARAMS, raw_values, 'mesh setting', frozenset(), check_refs=False)
        if mesh_spec.dx is None and mesh_spec.mode != 'auto':
            self.error(path, f"the cell size (dx) is required in mode {mesh_spec.mode} (empty only in auto mode, where "
                             'the automatic rule alone then sizes the cells)')
        triangles = mesh_spec.kind == 'triangles'
        element_kind = mesh_spec.kind if mesh_spec.kind in FREE_MESH_KINDS else None
        # builtin tetrahedra are graded by the tensor planner (meshing.graded_window_edges): tensor refinement rules
        # (swept prism layers too, along z: meshing.swept_meshing refuses dx_x / dx_y unlike dx)
        sized_kind = None if mesh_spec.kind == 'tetrahedra' and (mesh_spec.mesher == 'builtin'
                                                                 or mesh_spec.sweep_axis == 'z') else element_kind
        if mesh_spec.refinements:
            if mesh_spec.mode == 'uniform':
                self.info(path, 'refinements are ignored in uniform mode (used in nonuniform and auto modes)')
            else:   # meshing applies explicit refinements in nonuniform and auto mode
                for index, refinement in enumerate(mesh_spec.refinements):
                    self._check_refinement(path, index, refinement, sized_kind)
        if mesh_spec.axes and mesh_spec.mode == 'uniform' and sized_kind is None:   # meshing checks axis entries
            self.info(path, 'mesh axes options are ignored in uniform mode (used in nonuniform and auto modes)')
        if mesh_spec.size_points:
            problem = size_points_problem(mesh_spec.size_points, 3 if mesh_spec.kind == 'tetrahedra' else 2)
            if element_kind is None:
                self.error(path, 'size_points apply to triangle ([x, y, h]) and tetrahedron ([x, y, z, h]) meshes '
                                 f'only (adaptive refinement writes them on triangles); remove them for kind '
                                 f'{mesh_spec.kind}')
            elif problem:
                self.error(path, problem)
        mirrors = [f'mirror_{axis}' for axis in ('x', 'y') if getattr(mesh_spec, f'mirror_{axis}') is True]
        if mirrors and not triangles:
            self.error(path, f"{' / '.join(mirrors)} apply to triangle meshes only; set them false for kind "
                             f'{mesh_spec.kind}')
        if mesh_spec.sweep_axis not in ('none', None) and mesh_spec.kind != 'tetrahedra':
            self.error(path, f"sweep_axis applies to tetrahedron meshes only; set it to 'none' for kind "
                             f'{mesh_spec.kind}')
        if mesh_spec.face_sizes:
            if mesh_spec.kind != 'tetrahedra':
                self.error(path, f'face_sizes apply to tetrahedron meshes only; remove them for kind {mesh_spec.kind} '
                                 '(use refinements)')
            else:
                for index, face_size in enumerate(mesh_spec.face_sizes):
                    self._check_face_size(path, index, face_size)
        if triangles:
            self._check_triangle_mesh(path, mesh_spec)
        elif mesh_spec.kind == 'tetrahedra':
            self._check_tetrahedron_mesh(path, mesh_spec)
        if not used:
            self.warning(path, 'no physics uses this mesh, so it is not built: assign it to a physics (physics mesh) '
                               'or delete it')
            return
        # the estimate also fails on bad domain size / physics settings, which are reported at their own nodes
        mesh_error = self.vctx.mesh_errors.get(mesh_spec.name)
        if mesh_error is not None and domain_ok and self.mesh_inputs_ok and self.n_errors == errors_before:
            self.error(path, f'mesh estimate failed: {mesh_error}')
        mesh = self.vctx.meshes.get(mesh_spec.name)
        if mesh is None:
            return
        if mesh.note:   # meshing: why the grid differs from the requested mode
            self.info(path, mesh.note)
        if (mesh.kind == 'tetrahedra' and mesh.sizing.mesher == 'builtin' and not mesh.sizing.sweep_axis
                and mesh.sizing.edge_sizes):
            self.warning(path, f"mesher 'builtin' ignores the edge sizes the physics ask for along "
                               f"{', '.join(repr(name) for name in sorted(mesh.sizing.edge_sizes))} (e.g. electrode "
                               "edges): refine those objects (per-axis sizes) or use mesher 'gmsh'")
        for axis in mesh.custom_axes:   # custom edges / segments are not graded by max_ratio
            axis_spec = next(entry for entry in mesh_spec.axes if entry.axis == axis)
            raw_limit = next(value for value in (axis_spec.max_ratio, mesh_spec.max_ratio,
                                                 MESH_KIND_DEFAULTS['tensor']['max_ratio']) if value is not None)
            limit, _ = self._scalar(raw_limit)
            ratio = largest_adjacent_ratio(mesh, axis)
            if limit is not None and ratio > limit * (1.0 + 1e-6):
                self.warning(path, f'custom {axis} cells: neighbouring cells differ by up to {ratio:.3g}x, above the '
                                   f'maximum growth ratio {limit:g} (wave physics reflect at such jumps)')
        if mesh.wavelength_source not in ('', 'mesh'):
            self.info(path, f"auto mesh resolves wavelength_min = {mesh.wavelength_min:.4g} m, the shortest "
                            f"wavelength of physics '{mesh.wavelength_source}' (set wavelength_min to override)")
        if mesh.auto_rule == 'physics':
            base_dx = mesh.auto_dx if mesh.auto_dx is not None else mesh.dx_max   # dx_max: the outer cell with outer_dx
            self.info(path, f'auto mesh without a wave physics on it: base size ~{base_dx:.4g} m (longest domain '
                            f'side / {PHYSICS_BASE_DIVISOR:g}), the objects its physics use refined to their smallest '
                            f'side / {PHYSICS_OBJECT_DIVISOR:g} (set wavelength_min for the wavelength rule)')

    def _check_triangle_mesh(self, path, mesh_spec):
        """Kind 'triangles': a 2D Cartesian model, no tensor-only options (axes error, outer grading warning), the
        gmsh program when chosen; subpixel smoothing / conformal boundaries do not apply (info)."""
        self._check_element_mesh_space(path, 'triangles', 2)
        if mesh_spec.axes:
            self.error(path, 'mesh axes options apply to tensor meshes only; remove them for triangles')
        outer = [name for name in ('outer_dx', 'outer_ratio') if getattr(mesh_spec, name) is not None]
        if outer and mesh_spec.mode != 'uniform':
            self.warning(path, f"{' / '.join(outer)} grade tensor meshes only and are ignored on triangles (the "
                               'element sizes grow by max_ratio up to dx)')
        self._check_gmsh_found(path, mesh_spec, '')
        self._model_wide_flags_info(path, 'triangles', 'the elements follow material boundaries and curved walls')
        sizes = self.vctx.domain_size
        if sizes is None or self.spec.dimensions != 2:
            return
        for axis in ('x', 'y'):
            if getattr(mesh_spec, f'mirror_{axis}') is not True:
                continue
            mismatch = mirror_mismatch(self.spec, self.vctx.shapes, sizes, 'xy'.index(axis))
            if mismatch is not None:
                objects, fraction = mismatch
                self.warning(path, f'mirror_{axis}: the mesh is mirror symmetric about {axis} = size_{axis} / 2, but '
                                   f"the materials drawn are not ({', '.join(repr(name) for name in objects)} differ "
                                   f'from their mirror image on {100.0 * fraction:.0f} % of the checked samples); '
                                   'the mesh conforms to them and their images, but the solution is not symmetric: '
                                   'draw the structure symmetric or switch the mirror off')

    def _check_tetrahedron_mesh(self, path, mesh_spec):
        """Kind 'tetrahedra': a 3D Cartesian model; gmsh: no tensor-only options (axes error, outer grading warning)
        and the gmsh program; builtin (tensor-graded bricks): no size_points, an info; subpixel smoothing / conformal
        boundaries do not apply (info)."""
        self._check_element_mesh_space(path, 'tetrahedra', 3)
        swept = mesh_spec.sweep_axis == 'z'   # prism layers (plan D5): meshing refuses their other rules
        builtin = mesh_spec.mesher == 'builtin' and not swept
        if swept:
            self._check_swept_mesh(path, mesh_spec)
        if mesh_spec.axes and not (builtin or swept):
            self.error(path, "mesh axes options apply to tensor meshes and builtin tetrahedra only; remove them or use "
                             "mesher 'builtin'")
        outer = [name for name in ('outer_dx', 'outer_ratio') if getattr(mesh_spec, name) is not None]
        if outer and mesh_spec.mode != 'uniform' and not (builtin or swept):
            self.warning(path, f"{' / '.join(outer)} grade tensor meshes and builtin tetrahedra only and are ignored "
                               'by gmsh (the element sizes grow by max_ratio up to dx)')
        if builtin and mesh_spec.size_points:
            self.error(path, "size_points need mesher 'gmsh' (builtin tetrahedra are graded along whole axes by "
                             'refinements, face_sizes and the per-axis options); remove them or use mesher gmsh')
        if not self._check_gmsh_found(path, mesh_spec, ' (bricks split into tetrahedra)') and builtin:
            self.info(path, "mesher 'builtin': rectilinear bricks split into 6 tetrahedra each (no slivers at any "
                            'aspect ratio), graded like a tensor mesh along whole axes (refinements with per-axis '
                            'sizes, face_sizes, per-axis options; then flat object faces are element faces, else '
                            'equal cubes); curved objects are tagged per element (a staircase; h_min unused); mesher '
                            "'gmsh' follows spheres and cylinders")
        self._model_wide_flags_info(path, 'tetrahedra', 'gmsh elements follow spheres, boxes and cylinders')

    def _check_element_mesh_space(self, path, kind, dimensions):
        """Error unless the model is a dimensions-D Cartesian one (kind 'triangles' 2D / 'tetrahedra' 3D), naming the
        kind that fits."""
        if self.spec.dimensions == dimensions and self.spec.domain.coordinates == 'cartesian':
            return
        fitting = {2: 'triangles', 3: 'tetrahedra'}.get(self.spec.dimensions, 'tensor')
        noun = 'triangle' if kind == 'triangles' else 'tetrahedron'
        self.error(path, f'{noun} meshes need a {dimensions}D Cartesian domain (this model: {self.spec.dimensions}D, '
                         f"coordinates '{self.spec.domain.coordinates}'); use kind "
                         f"{fitting if fitting != kind else 'tensor'}")

    def _check_gmsh_found(self, path, mesh_spec, builtin_note) -> bool:
        """Error (True) when mesher 'gmsh' is chosen and the gmsh program is not found; builtin_note follows the
        'use mesher builtin' remedy."""
        if mesh_spec.mesher != 'gmsh' or _gmsh_found():
            return False
        self.error(path, f"mesher 'gmsh': the gmsh program was not found (set {_GMSH_ENVIRONMENT} to its path or "
                         f"put it on the PATH), or use mesher 'builtin'{builtin_note}")
        return True

    def _model_wide_flags_info(self, path, kind, reason):
        """Info that subpixel_smoothing / conformal_boundaries (when on) do not apply on an element mesh kind."""
        flags = [name for name in ('subpixel_smoothing', 'conformal_boundaries')
                 if getattr(self.spec.solver_defaults, name, False) is True]
        if flags:
            self.info(path, f"{' and '.join(flags)} do not apply on {kind}: {reason}")

    def _check_swept_mesh(self, path, mesh_spec):
        """sweep_axis 'z' (swept prism layers, plan D5): the meshing rules (swept_mesh_problem) as an error, an info,
        a warning naming the primitives tagged per element (no vertical side walls)."""
        from .geometry import is_array_copy
        from .swept_meshing import footprint_shapes, swept_mesh_problem
        problem = swept_mesh_problem(mesh_spec, self.spec.dimensions == 3)
        if problem:
            self.error(path, problem)
            return
        self.info(path, f"sweep_axis 'z': {mesh_spec.mesher} triangles in the x-y plane stacked in z layers, each "
                        'prism split into 3 tetrahedra; layers from the z grading (refinement dx_z / dx, z axis '
                        'options, z face sizes), object top / bottom faces are layer planes')
        unswept = [name for name in footprint_shapes(self.vctx.shapes).unswept if not is_array_copy(name)]
        if unswept:
            self.warning(path, f"sweep_axis 'z': {', '.join(repr(name) for name in unswept)} have no vertical side "
                               'walls (spheres, tapers, x / y extruded shapes other than rectangles) and are tagged '
                               "per element (a staircase); use sweep_axis 'none' with mesher gmsh to follow them")

    def _check_one_mesh_rules(self, used):
        """S7.2 1.2: physics that must share one mesh (every family 'em' physics, every physics of a study in
        _ONE_MESH_STUDIES; couplings and feature refs: check_couplings / check_features)."""
        if len(used) < 2:
            return
        meshes = {}   # physics name -> mesh name (known meshes only)
        for physics in self.spec.physics:
            name = self._mesh_name_of(physics)
            if name is not None:
                meshes[physics.name] = name
        em_physics = [physics.name for physics, physics_type, _ in self.physics_entries
                      if physics_type.family == 'em' and physics.name in meshes]
        for name in em_physics[1:]:
            first = em_physics[0]
            if meshes[name] != meshes[first]:
                self.error(('physics', name), f"EM physics share one mesh (common boundaries, PML and absorbed-power "
                                              f"windows): '{name}' runs on mesh '{meshes[name]}', '{first}' on "
                                              f"'{meshes[first]}'; set the mesh of '{name}' to '{meshes[first]}'")
        if self.spec.study.type in _ONE_MESH_STUDIES and len(set(meshes.values())) > 1:
            self.error(('study',), f"the {self.spec.study.type} study combines its physics on one grid: put every "
                                   f"physics on one mesh (now: {', '.join(sorted(set(meshes.values())))})")

    def check_memory(self):
        """meshing.estimate_memory of every used mesh against memory_budget and host_budget (above: an issue at
        memory_check_level, error by default; warning above MEMORY_WARN_FRACTION; no GPU budget: no GPU check; no host
        budget: LARGE_HOST_BYTES warning), at the ('mesh', name) of memory_verdict (the largest part); on the cpu
        backend the device estimate counts against host_budget too. An error on size-based element counts says so:
        build always meshes first and judges the built counts, api.check when refused_on_estimated_counts."""
        if not self.vctx.meshes or any(mesh is None for mesh in self.vctx.meshes.values()):
            return
        whole_estimate = self.memory_estimate
        if whole_estimate is None:
            try:
                whole_estimate = estimate_memory(self.spec, self.vctx.parameters, self.registry, self.vctx.meshes,
                                                 shapes=self.vctx.shapes)
            except MESH_ESTIMATE_ERRORS:
                return   # the failing input is reported at its own node
            except Exception as exc:   # a crashing memory_estimate callback must not take down validation
                self.error(('meshes',), internal_error_text('the memory estimate', exc))
                return
        estimate = backend_memory(self.spec, whole_estimate)   # cpu: the device arrays live in RAM, one host budget
        budget = None if solver_backend(self.spec) == 'cpu' else self.memory_budget
        host_budget = self.host_budget
        gpu_close = bool(budget) and estimate.device_peak > MEMORY_WARN_FRACTION * budget
        host_close = host_budget is not None and estimate.host > MEMORY_WARN_FRACTION * host_budget
        host_large = host_budget is None and estimate.host > LARGE_HOST_BYTES
        if not (gpu_close or host_close or host_large):
            return
        verdict = memory_verdict(self.spec, self.vctx.meshes, whole_estimate, self.memory_budget, host_budget,
                                 self._memory_parts())
        path = ('mesh', verdict.mesh)
        refuses = memory_check_level(self.spec) == 'error'
        over_issue = self.error if refuses else self.warning
        if not refuses:
            override = " (Solvers memory_check 'warning': running anyway, the runtime memory guards decide)"
        elif verdict.pending:   # not final yet: element counts from the sizes or factor fills without the mesh
            pending_names = ', '.join(f"'{name}'" for name in verdict.pending)
            sized = any(has_estimated_counts(self.vctx.meshes[name]) for name in verdict.pending)
            what = ('element count' if sized else 'direct-solver factor size')
            override = (f'. The {what} of mesh {pending_names} is estimated without the built mesh: a build makes the '
                        f'mesh first and judges on its real counts (GUI: Build mesh on the mesh node)')
        else:
            override = ''
        if gpu_close:
            over = verdict.gpu_over
            largest = '; '.join(f'{label} {format_bytes(size)}' for label, size in sorted(
                estimate.device_items(), key=lambda item: -item[1])[:3])
            external = (f', cuDSS factors outside the pool {format_bytes(estimate.external_peak)}'
                        if estimate.external_peak else '')
            text = (f'estimated GPU memory {format_bytes(estimate.device_peak)} at the peak (build peak '
                    f'{format_bytes(estimate.gpu_build_peak)}, {format_bytes(estimate.gpu_steady + estimate.gpu_run)} '
                    f'while running{external}) '
                    f"{'exceeds' if over else 'is close to'} the GPU budget {format_bytes(budget)}; largest: "
                    f'{largest}. Reduce the cell count (mesh cell size, domain size), DFT / far-field boxes or their '
                    f'n_wavelengths, close other GPU programs, or set the Solvers memory_check to warning (GUI: '
                    f'Override memory warning, on this mesh node) when the estimate is known to be high')
            (over_issue if over else self.warning)(path, text + (override if over else ''))
        if host_close:
            over = verdict.host_over
            (over_issue if over else self.warning)(path, _host_memory_text(estimate, host_budget)
                                                   + (override if over else ''))
        elif host_large:
            self.warning(path, _host_memory_text(estimate, None))

    def _memory_parts(self):
        """{used mesh name: its part of the memory estimate} for memory_verdict: the caller's memory_parts, {} with
        one mesh (the issue is its own), else estimated here ({} when that fails: the first used mesh)."""
        if self.memory_parts is not None or len(self.vctx.meshes) < 2:
            return self.memory_parts or {}
        parts = {}
        try:
            estimate_memory(self.spec, self.vctx.parameters, self.registry, self.vctx.meshes, by_mesh=parts,
                            shapes=self.vctx.shapes)
        except Exception:   # check_memory already has the whole estimate: only the node of its issue is at stake
            return {}
        return parts

    def _check_refinement(self, path, index, refinement, element_kind=None):
        """One mesh refinement; element_kind 'triangles' / 'tetrahedra': a maximum element size (no per-axis sizes,
        margin unused)."""
        where = f'refinement {index + 1}'
        if refinement.object not in self.objects:
            self.error(path, f'{where}: unknown geometry object {refinement.object!r}; valid geometry objects: '
                             f'{_name_list(self.objects)}')
        if element_kind is not None:   # a maximum element size in / along / at the object
            per_axis = [f'dx_{axis}' for axis in 'xyz' if getattr(refinement, f'dx_{axis}') is not None]
            if per_axis:
                self.error(path, f"{where}: {' / '.join(per_axis)} apply to tensor meshes only; give dx (the maximum "
                                 'element size of the object)')
            if refinement.margin is not None:
                self.info(path, f'{where}: margin is unused on {element_kind} (sizes grow by max_ratio from the '
                                'object)')
        place = getattr(refinement, 'where', 'inside')
        if place not in REFINEMENT_PLACES:
            self.error(path, f"{where}: where must be one of {', '.join(REFINEMENT_PLACES)}, got {place!r}")
        elif place != 'inside':
            self._check_refinement_place(path, where, refinement, element_kind)
        self._check_number(path, f'{where} cell size', refinement.dx)
        if refinement.margin is not None:
            self._check_number(path, f'{where} margin', refinement.margin, strict=False)

    def _check_refinement_place(self, path, label, refinement, element_kind):
        """RefinementSpec.where 'edges' / 'surface': gmsh tetrahedra only (error); a warning when gmsh gets nothing to
        size (no domain object, primitives without a gmsh solid, spheres for 'edges')."""
        place = refinement.where
        if element_kind != 'tetrahedra':   # _check_mesh passes None for tensor meshes and builtin tetrahedra
            self.error(path, f"{label}: where '{place}' applies to tetrahedron meshes with mesher 'gmsh' only (a size "
                             "field along the object's edges / at its surface); use 'inside'")
            return
        shape = self.vctx.shapes.get(refinement.object) if isinstance(refinement.object, str) else None
        if shape is None:
            return
        if GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            self.warning(path, f"{label}: where '{place}' sizes the edges / surface of domain objects; "
                               f"{refinement.object!r} is a {shape.kind}, so nothing is refined (use 'inside')")
            return
        from .tetrahedral_meshing import primitives_without_solids
        missing = primitives_without_solids(refinement.object, self.vctx.shapes, edges=place == 'edges')
        if missing:
            kinds = 'spheres have no edges; ' if place == 'edges' else ''
            self.warning(path, f"{label}: where '{place}': gmsh sizes nothing along "
                               f"{', '.join(repr(name) for name in missing)} ({kinds}ellipses, polygons, x / y "
                               'circles and tapered prisms get no gmsh solid and are tagged per element); refine '
                               "them with where 'inside' or face_sizes")

    def _check_face_size(self, path, index, face_size):
        """One MeshFaceSizeSpec of a tetrahedron mesh: an outer face (face '') or a domain object (face '' or one of
        its bounding-box faces), size > 0, distance >= 0."""
        where = f'face size {index + 1}'
        entity, face = face_size.entity, face_size.face
        shape = self.vctx.shapes.get(entity) if isinstance(entity, str) else None
        if entity in OUTER_FACES_3D:
            if face:
                self.error(path, f"{where}: face applies to geometry objects only; leave it '' for the outer face "
                                 f'{entity}')
        elif entity not in self.objects:
            self.error(path, f'{where}: unknown entity {entity!r}; give an outer face ({", ".join(OUTER_FACES_3D)}) '
                             f'or a domain object ({_name_list(self.objects)})')
        elif shape is not None and GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            self.error(path, f'{where}: {entity!r} is a {shape.kind} object; face sizes refine outer faces or the '
                             'bounding-box faces of domain objects')
        if entity not in OUTER_FACES_3D and face not in ('', *OUTER_FACES_3D):
            self.error(path, f"{where}: face must be '' (all six bounding-box faces) or one of "
                             f"{', '.join(OUTER_FACES_3D)}, got {short_repr(face)}")
        self._check_number(path, f'{where} size', face_size.size)
        self._check_number(path, f'{where} distance', face_size.distance, strict=False)

    # ---- solvers ----
    def check_solvers(self):
        """Solver defaults, solver nodes (type, settings, unused) and the physics' role assignments (solver_plan S3.6)."""
        spec = self.spec
        if spec.solver_defaults.backend not in SOLVER_BACKENDS:
            self.error(('solvers',), f'backend {short_repr(spec.solver_defaults.backend)} is not one of '
                                     f"{', '.join(SOLVER_BACKENDS)}")
            return
        backend = solver_backend(spec)
        if spec.solver_defaults.memory_check not in MEMORY_CHECKS:
            self.error(('solvers',), f'memory_check {short_repr(spec.solver_defaults.memory_check)} is not one of '
                                     f"{', '.join(MEMORY_CHECKS)}")
        self._check_default_precision()
        self._check_subpixel_smoothing()
        self._check_conformal_boundaries()
        node_values = {}   # solver node name -> (SolverType, resolved values) of the nodes without errors
        for node in spec.solvers:
            path = ('solver', node.name)
            try:
                solver_type = self.registry.solver_type(node.type)
            except KeyError as exc:
                self.error(path, error_message(exc))
                continue
            if not self._check_values(path, solver_type.params, node.settings, 'setting', frozenset()):
                continue
            values = self.vctx.values(solver_type.params, with_default_precision(spec, solver_type, node.settings))
            if values is None:
                continue
            node_values[node.name] = (solver_type, values)
            if solver_type.validate is not None:
                self._delegate_issues(path, f"solver type '{solver_type.name}' validate", solver_type.validate,
                                      self.vctx, values, path)
        for physics, physics_type, _settings_values in self.physics_entries:
            if backend not in physics_type.backends:   # solver_plan S6: the physics' own arrays and kernels
                self.error(('physics', physics.name),
                           f'{physics_type.label} runs on the {BACKENDS[physics_type.backends[0]]} only so far (its '
                           f'fields, materials and kernels have no {BACKENDS[backend]} path yet): set the Solvers '
                           "backend to 'auto' or " + ' or '.join(f"'{name}'" for name in physics_type.backends))
                continue
            errors_before = self.n_errors
            self._check_physics_solvers(physics, physics_type, node_values, backend)
            if self.n_errors == errors_before:
                self._check_automatic_solvers(physics, physics_type, backend)
            if self.n_errors == errors_before:
                self._check_feature_precisions(physics, physics_type)
        assigned = {name for physics in spec.physics if isinstance(physics.solvers, dict)
                    for name in physics.solvers.values()}
        for node in spec.solvers:
            if node.name not in assigned:
                self.warning(('solver', node.name), 'no physics uses this solver: assign it to a physics role (physics '
                                                    'solvers {role: name}) or delete it')

    def _check_default_precision(self):
        """solver_defaults.precision: 'auto' or a precision; when set, an info naming the (physics, role) solvers that
        keep their own precision and why (solver_plan S9)."""
        value = self.spec.solver_defaults.precision
        if value not in SOLVER_DEFAULT_PRECISIONS:
            self.error(('solvers',), f'precision {short_repr(value)} is not one of '
                                     f"{', '.join(SOLVER_DEFAULT_PRECISIONS)}")
            return
        ignored = default_precision_ignored(self.spec, self.registry, self.vctx.study_type)
        if ignored:
            reasons = '; '.join(f"'{physics}' {role}: {reason}" for physics, role, reason in ignored)
            self.info(('solvers',), f'the default {PRECISIONS[value]} is not used by {reasons}')

    def _check_subpixel_smoothing(self):
        self._check_model_wide_flag('subpixel_smoothing', subpixel_smoothing_physics,
                                    'smoothing of material interfaces is ignored by',
                                    'em_fdtd_ez / _hz, em_fdfd_ez / _hz, em_modes and em_fdtd_3d outside an '
                                    'optimization study')

    def _check_conformal_boundaries(self):
        self._check_model_wide_flag('conformal_boundaries', conformal_boundaries_physics,
                                    'conformal boundaries are ignored by',
                                    'em_fdtd_ez / _hz and em_modes (pec_domain), laminar_flow (solid walls), heat and '
                                    'the potential types (held regions, insulated walls, perimeter fluxes) in 2D '
                                    'Cartesian models and em_fdtd_3d (pec_domain) in 3D models')

    def _check_model_wide_flag(self, name, honouring_physics, ignored_text, honoured_by):
        """solver_defaults.<name>: a bool; when on, an info '<ignored_text> <physics and why>', a warning when none
        honours it (only honoured_by do)."""
        value = getattr(self.spec.solver_defaults, name)
        if not isinstance(value, bool):
            self.error(('solvers',), f'{name} must be true or false, got {short_repr(value)}')
            return
        honouring, ignoring = honouring_physics(self.spec, self.registry)
        if ignoring:
            reasons = '; '.join(f"'{physics}': {reason}" for physics, reason in ignoring.items())
            self.info(('solvers',), f'{ignored_text} {reasons}')
        if value and not honouring:
            self.warning(('solvers',), f'{name} is on but no physics of this model honours it (only {honoured_by} '
                                       'do): switch it off')

    def _check_physics_solvers(self, physics, physics_type, node_values, backend):
        """PhysicsSpec.solvers: known role and node, the role used in this study (else a warning) and solver_misfit (an
        error naming the fitting choices)."""
        path = ('physics', physics.name)
        if not isinstance(physics.solvers, dict):
            self.error(path, 'solvers must be a mapping of solver role -> solver name, got '
                             f'{short_repr(physics.solvers)}')
            return
        node_names = [node.name for node in self.spec.solvers]
        study = self.vctx.study_type
        for role, node_name in physics.solvers.items():
            if role not in SOLVER_ROLES:
                self.error(path, f"solvers: unknown solver role {short_repr(role)}; valid: {', '.join(SOLVER_ROLES)}")
                continue
            if node_name not in node_names:
                self.error(path, f'{role} solver: unknown solver {short_repr(node_name)}; valid: '
                                 f'{_name_list(node_names)} (or leave the role out for Automatic)')
                continue
            if node_name not in node_values:
                continue   # reported at the solver node
            solver_type, values = node_values[node_name]
            solver_role = physics_type.solver_roles.get(role)
            if solver_role is not None and not solver_role.used_in(study):
                self.warning(path, f"{role} solver '{node_name}': {physics_type.label} does not use a {role} solver in "
                                   f'a {study} study; the assignment is ignored')
                continue
            if solver_role is not None:   # the default precision applies per role (schema.resolved_solvers)
                settings = self.spec.solver(node_name).settings
                role_values = self.vctx.values(solver_type.params, with_node_precision(
                    self.spec, solver_type, settings, solver_role, backend, solver_grid(self.spec, physics), study))
                values = values if role_values is None else role_values
            misfit = solver_misfit(solver_type, values, physics_type, role, study, backend,
                                   solver_grid(self.spec, physics))
            if misfit:
                self.error(path, f"{role} solver '{node_name}': {misfit}; "
                                 + self._fitting_solvers_text(physics, physics_type, role, node_values, study, backend))

    def _check_automatic_solvers(self, physics, physics_type, backend):
        """The Automatic roles (no node assigned) against solver_misfit on the run backend (explicit_kernels on the
        cpu): error at the physics."""
        resolved = self.vctx.solvers(physics)
        for role, entry in (resolved or {}).items():
            if entry.node is not None:
                continue
            misfit = solver_misfit(entry.solver_type, entry.values, physics_type, role, self.vctx.study_type, backend,
                                   solver_grid(self.spec, physics))
            if misfit:
                self.error(('physics', physics.name), f"Automatic {role} solver '{entry.solver_type.name}': {misfit}; "
                                                      'assign a solver node that fits')

    def _fitting_solvers_text(self, physics, physics_type, role, node_values, study, backend):
        """'fitting: ...' (Automatic and the solver nodes that fit), or the remedy when the type has no such role."""
        solver_role = physics_type.solver_roles.get(role)
        if solver_role is None:
            return f'remove the {role} entry from the physics solvers'
        grid = solver_grid(self.spec, physics)
        fitting = [f"'{name}'" for name, (solver_type, values) in node_values.items()
                   if not solver_misfit(solver_type, values, physics_type, role, study, backend, grid)]
        automatic_type = solver_role.automatic(backend, grid, study)[0]
        return 'fitting: ' + ', '.join([f"Automatic ('{automatic_type}': leave the role out)"] + fitting)

    def _check_feature_precisions(self, physics, physics_type):
        """Enabled features declaring FeatureType.precisions against the resolved stepper precision (S3.3): error at
        the feature."""
        if 'stepper' not in physics_type.solver_roles:
            return
        restricted = []   # (feature spec, FeatureType) of enabled features that do not take every precision
        for feature in physics.features:
            try:
                feature_type = physics_type.feature_type(feature.type)
            except KeyError:   # reported by check_features
                continue
            if feature.enabled and set(feature_type.precisions) != set(PRECISIONS):
                restricted.append((feature, feature_type))
        if not restricted:
            return
        resolved = self.vctx.solvers(physics)
        if resolved is None:   # reported at the physics / solver node
            return
        for feature, feature_type in restricted:
            misfit = feature_precision_misfit(feature_type, resolved)
            if misfit:
                self.error(('feature', physics.name, feature.name),
                           f'{misfit}: choose a stepper solver of that precision or disable the feature')

    # ---- study ----
    def check_study(self):
        study = self.spec.study
        path = ('study',)
        if not self.spec.physics:
            self.error(path, 'the model has no physics; add one to compute anything')
        try:
            study_type = self.registry.study_type(study.type)
        except KeyError as exc:
            self.error(path, error_message(exc))
            study_type = None
        if study_type is not None:
            background = study_type.background_study
            for physics, physics_type, _ in self.physics_entries:
                supported_studies = physics_type.supported_studies
                if study.type in supported_studies or (background and background in supported_studies):
                    continue
                supported = ', '.join(supported_studies) or 'none'
                problem = (f'supports neither the {study_type.label} study nor, as a background physics, the '
                           f'{background} study' if background else f'does not support the {study_type.label} study')
                self.error(path, f"physics '{physics.name}' ({physics_type.label}) {problem} (supported: {supported})")
            if self._check_values(path, study_type.params, study.settings, 'study setting', frozenset()):
                self._check_study_values(path, study_type, study.settings)
            if study_type.requires_time_stepping:
                self._check_time_stepping(path, study_type)
            self._check_solved_physics(path, study.solved_physics, study_type)
            if background:
                self._check_background_study(path, study_type)
        self._check_stored_fields(path, study.stored_fields)
        self._check_stored_cuts(path, study.stored_cuts)
        self._check_sweep(path, study.sweep)
        self._check_adaptation(path, study.adaptation, study_type)
        if study.estimation is not None:
            self._check_estimation(path, study.estimation)

    def _check_estimation(self, path, estimation):
        """StudySpec.estimation: no sweep or adaptation, scipy installed, and model.estimation's checks of the
        parameters, targets, curves and settings."""
        import importlib.util

        from .estimation import resolve_estimation   # only models with an estimation load it (selftest cache)
        from .spec import EstimationSpec
        if not isinstance(estimation, EstimationSpec):
            self.error(path, f'estimation must be an EstimationSpec object, got {short_repr(estimation)}')
            return
        if self.spec.study.sweep is not None:
            self.error(path, 'estimation: a parameter estimation within a parameter sweep is not supported; remove the '
                             'sweep or the estimation')
        if self.spec.study.adaptation is not None:
            self.error(path, 'estimation: a parameter estimation with mesh refinement levels is not supported; clear '
                             'adaptation (refine the mesh first, then estimate on it) or the estimation')
        if importlib.util.find_spec('scipy') is None:
            self.error(path, 'estimation needs scipy (least_squares / Nelder-Mead), which is not installed: pip install '
                             'scipy')
        _, errors, warnings = resolve_estimation(self.spec, self.vctx.parameters)
        for message in errors:
            self.error(path, f'estimation: {message}')
        for message in warnings:
            self.warning(path, f'estimation: {message}')

    def _check_adaptation(self, path, adaptation, study_type):
        """StudySpec.adaptation (S7.7): a study type with supports_adaptation, no sweep or parity sectors (v1), a known
        method, target = a derived value, tolerance > 0, max_levels >= 1, growth > 1, meshes = used mesh names,
        max_cells null or >= 1, bool switches."""
        if adaptation is None:
            return
        if study_type is not None and not study_type.supports_adaptation:
            supported = _name_list(other.name for other in self.registry.study_types() if other.supports_adaptation)
            self.error(path, f'adaptation: the {study_type.label} study does not refine meshes (studies that do: '
                             f'{supported}); clear adaptation')
        if self.spec.study.sweep is not None:
            self.error(path, 'adaptation: mesh refinement levels within a parameter sweep are not supported yet; '
                             'remove the sweep or the adaptation')
        if isinstance(self.spec.study.settings, dict) and self.spec.study.settings.get('parity_sectors') is True:
            self.error(path, 'adaptation: mesh refinement levels with parity sectors are not supported yet; switch '
                             'parity_sectors off or clear adaptation')
        if adaptation.method not in ADAPTATION_METHODS:
            self.error(path, f"adaptation method {short_repr(adaptation.method)} is not one of "
                             f"{', '.join(ADAPTATION_METHODS)}")
        elif adaptation.method == 'adaptive' and self.spec.dimensions == 3:   # mesh_refinement.refined_axes
            self.error(path, "adaptation: adaptive refinement of 3D meshes is not supported yet; use method 'uniform'")
        derived_names = [derived.name for derived in self.spec.derived_values]
        if adaptation.target not in derived_names:
            self.error(path, f'adaptation target {short_repr(adaptation.target)} is not a derived value; derived '
                             f'values: {_name_list(derived_names)} (the refinement stops when it converges)')
        for name, lower, inclusive, integer in (('tolerance', 0.0, False, False), ('max_levels', 1, True, True),
                                                ('growth', 1.0, False, False)):
            value = getattr(adaptation, name)
            number_ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if not number_ok or (integer and not isinstance(value, int)) or not (
                    value >= lower if inclusive else value > lower):
                bound = f"{'>=' if inclusive else '>'} {lower:g}"
                self.error(path, f"adaptation {name} must be {'an integer' if integer else 'a number'} {bound}, got "
                                 f'{short_repr(value)}')
        max_cells = adaptation.max_cells
        if max_cells is not None and (isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1):
            self.error(path, f'adaptation max_cells must be null or an integer >= 1, got {short_repr(max_cells)}')
        for name in ('anisotropic', 'keep_levels'):
            if not isinstance(getattr(adaptation, name), bool):
                self.error(path, f'adaptation {name} must be true or false, got {short_repr(getattr(adaptation, name))}')
        used = [mesh_spec.name for mesh_spec in self.spec.used_meshes()]
        if not isinstance(adaptation.meshes, list):
            self.error(path, f'adaptation meshes must be a list of mesh names, got {short_repr(adaptation.meshes)}')
            return
        for name in adaptation.meshes:
            if name not in used:
                self.error(path, f'adaptation mesh {short_repr(name)} is not a mesh the physics use; used meshes: '
                                 f'{_name_list(used)} ([] = all of them)')

    def _check_study_values(self, path, study_type, settings):
        """The study type's own validate callback over the resolved settings."""
        if study_type.validate is None:
            return
        values = self.vctx.values(study_type.params, settings)
        if values is not None:
            self._delegate_issues(path, f"study type '{study_type.name}' validate", study_type.validate, self.vctx,
                                  values)

    def _check_solved_physics(self, path, solved_physics, study_type):
        """StudySpec.solved_physics: a list of distinct physics names, only in a study of _SOLVED_PHYSICS_STUDIES, each
        physics supporting the study itself; every physics left out must support its background study."""
        if isinstance(solved_physics, list) and not solved_physics:
            return
        if not isinstance(solved_physics, list) or not all(isinstance(name, str) for name in solved_physics):
            self.error(path, f'solved_physics must be a list of physics names, got {short_repr(solved_physics)}')
            return
        if study_type.name not in _SOLVED_PHYSICS_STUDIES or not study_type.background_study:
            self.error(path, f'solved_physics applies to the {", ".join(_SOLVED_PHYSICS_STUDIES)} studies (the physics '
                             f'it leaves out run as stationary background physics); the {study_type.label} study solves '
                             'every physics itself: clear solved_physics')
            return
        duplicates = sorted(name for name, count in Counter(solved_physics).items() if count > 1)
        if duplicates:
            self.error(path, f"solved_physics lists {', '.join(duplicates)} more than once")
        entries = {physics.name: physics_type for physics, physics_type, _ in self.physics_entries}
        for name in dict.fromkeys(solved_physics):
            if name not in self.physics_by_name:
                self.error(path, f"solved_physics names '{name}', which is no physics of the model; physics: "
                                 f"{_name_list(physics.name for physics in self.spec.physics)}")
            elif name in entries and study_type.name not in entries[name].supported_studies:
                self.error(path, f"solved_physics names '{name}' ({entries[name].label}), which the {study_type.label} "
                                 'study cannot solve (its type does not list the study): leave it out, it then runs as '
                                 f'a {study_type.background_study} background physics')
        background = study_type.background_study
        for name, physics_type in entries.items():
            if name in solved_physics or background in physics_type.supported_studies:
                continue
            if study_type.name in physics_type.supported_studies:   # else reported by check_study
                self.error(path, f"physics '{name}' ({physics_type.label}) is not in solved_physics, so it would run as "
                                 f'a {background} background physics, which its type does not support: add it to '
                                 'solved_physics')

    def _solved_by_study(self, study_type):
        """Names of the physics the study solves itself: those listed in StudySpec.solved_physics (when set and a list),
        else every physics whose type lists the study."""
        selection = self.spec.study.solved_physics
        listed = selection if isinstance(selection, list) and selection else None
        return [physics.name for physics, physics_type, _ in self.physics_entries
                if study_type.name in physics_type.supported_studies
                and (listed is None or physics.name in listed)]

    def _check_background_study(self, path, study_type):
        """A study with background physics: >= 1 physics supporting the study itself (harmonic / mode physics); nothing
        may read such a physics (enabled couplings, material properties), except with StudyType.solution_feedback a
        coupling writing a background physics (info: the run alternates background and study solves) and with
        StudyType.coupled_points any reader when every physics supports the background study (info: coupled points).
        With StudySpec.solved_physics only the listed physics count as solved by the study."""
        if not self.physics_entries or len(self.physics_entries) != len(self.spec.physics):
            return   # no physics, or an unknown type: reported at ('study',) / the physics
        harmonic = self._solved_by_study(study_type)
        if not harmonic:
            capable = _name_list(physics_type.name for physics_type in self.registry.physics_types()
                                 if study_type.name in physics_type.supported_studies)
            self.error(path, f'the {study_type.label} study needs at least one physics that supports it (types: '
                             f'{capable}); every physics here would only run as a background physics (use a '
                             f'{study_type.background_study} study)')
            return
        alternative = f'; {study_type.feedback_alternative}' if study_type.feedback_alternative else ''
        # coupled points: physics that cannot join the stationary fixed point of every physics at each point
        unsteady = [physics.name for physics, physics_type, _ in self.physics_entries
                    if study_type.background_study not in physics_type.supported_studies]
        coupled_problem = (f': a coupled {study_type.label} run solves every point as a {study_type.background_study} '
                           f"fixed point of every physics, which {', '.join(unsteady)} cannot join"
                           if study_type.coupled_points and unsteady else '')
        feedback, readers = [], []
        for coupling in self.spec.couplings:
            if not (coupling.enabled and coupling.source in harmonic):
                continue
            if study_type.solution_feedback and coupling.target not in harmonic:
                feedback.append(coupling.name)
            elif study_type.solution_feedback:
                self.error(('coupling', coupling.name),
                           f"source '{coupling.source}' and target '{coupling.target}' are both solved by the "
                           f'{study_type.label} study itself: its solutions are fed back only into background physics '
                           f'(those solved by the {study_type.background_study} background step)')
            elif study_type.coupled_points and not unsteady:
                readers.append(coupling.name)
            else:
                self.error(('coupling', coupling.name),
                           f"source '{coupling.source}' is solved by the {study_type.label} study itself (not as a "
                           f'background physics), whose solutions are not fed back into other physics'
                           f'{coupled_problem}{alternative}')
        if feedback:
            self.info(path, f"coupling(s) {', '.join(feedback)} feed the {study_type.label} solution back into "
                            'background physics: the run alternates background and study solves until every primary '
                            'field changes by at most the tolerance between rounds (study setting feedback_rounds)')
        harmonic_fields = {name: physics for name, (physics, _) in self.vctx.state_names.items()
                           if physics in harmonic}
        for material in self.spec.materials:
            for what, raw in _material_property_values(material):
                read = sorted(_referenced_or_empty(raw).intersection(harmonic_fields))
                if not read:
                    continue
                if study_type.coupled_points and not unsteady:
                    readers.append(f'material {material.name}')
                    continue
                self.error(('material', material.name),
                           f"{what} {short_repr(raw)} uses {', '.join(read)} of the physics "
                           f"'{harmonic_fields[read[0]]}', which the {study_type.label} study solves itself and "
                           f'does not feed back into materials{coupled_problem}{alternative}')
        if readers:
            self.info(path, f"coupled {study_type.label} run: {', '.join(dict.fromkeys(readers))} read physics the "
                            f'study solves, so each point (wavelength) is a {study_type.background_study} fixed point '
                            'of every physics (max_iterations, tolerance, relaxation), warm-started from the previous '
                            'point in list order')

    def _check_time_stepping(self, path, study_type):
        """Error when no physics steps in time (skipped while any physics settings are invalid)."""
        entries = self.physics_entries
        if not entries or any(values is None for _, _, values in entries):
            return
        for physics, physics_type, values in entries:
            if physics_type.steps_in_time is None:
                continue
            steps, ok = self._call(('physics', physics.name), f"physics type '{physics_type.name}' steps_in_time",
                                   physics_type.steps_in_time, values)
            if ok and steps:
                return
        self.error(path, f'the {study_type.label} study needs at least one physics that steps in time; none does '
                         f'(use a stationary study, or switch a physics to a time-stepping mode)')

    def _check_stored_fields(self, path, stored_fields):
        for entry in stored_fields:
            if not isinstance(entry, str) or entry.count('.') != 1:
                self.error(path, f"stored field {entry!r}: expected 'physics.component'")
                continue
            physics_name, component = entry.split('.')
            physics = self.physics_by_name.get(physics_name)
            if physics is None:
                self.error(path, f"stored field '{entry}': no physics named '{physics_name}'; valid physics: "
                                 f"{_name_list(other.name for other in self.spec.physics)}")
                continue
            try:
                physics_type = self.registry.physics_type(physics.type)
            except KeyError:
                continue   # reported at the physics
            components = [info.name for info in physics_type.output_fields]
            if component not in components:
                self.error(path, f"stored field '{entry}': physics '{physics_name}' has no output field "
                                 f"'{component}'; available: {', '.join(components)}")
                continue
            required_study = physics_type.output_fields[components.index(component)].requires_study
            if required_study and required_study != self.spec.study.type:
                self.error(path, f"stored field '{entry}' exists only in a {required_study} study (this study: "
                                 f'{self.spec.study.type}); remove it or change the study type')

    def _check_stored_cuts(self, path, stored_cuts):
        """Errors at path for stored cut planes outside a 3D model, not [axis, position] or off the domain."""
        if stored_cuts and not self.spec.is_3d:
            self.error(path, 'stored_cuts: cut planes apply to 3D models only (a 2D model stores its whole plane); '
                             'clear them')
            return
        sizes = {'x': self.spec.domain.size_x, 'y': self.spec.domain.size_y, 'z': self.spec.domain.size_z}
        for entry in stored_cuts:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] in sizes
                    and isinstance(entry[1], (int, float, str)) and not isinstance(entry[1], bool)):
                self.error(path, f"stored cut {short_repr(entry)}: expected ['x' | 'y' | 'z', position in m] (the "
                                 'plane normal to that axis through the window cell containing the position)')
                continue
            axis, raw = entry
            if axis != 'z' and self.spec.domain.coordinates == 'polar':
                self.error(path, f"stored cut {axis}: a cylindrical (r, phi, z) sector (coordinates polar) is cut at "
                                 "constant z only; use ['z', height]")
                continue
            position, message = self._scalar(raw)
            if message is not None:
                self.error(path, f'stored cut {axis}: {message}')
                continue
            size = self._scalar(sizes[axis])[0]
            if size is not None and not -1e-9 * size <= position <= size * (1.0 + 1e-9):
                self.error(path, f'stored cut {axis} = {position:.6g} m lies outside the domain (0 .. {size:.6g} m '
                                 f'along {axis})')

    def _check_sweep(self, path, sweep):
        if sweep is None:
            return
        parameter_names = [parameter.name for parameter in self.spec.parameters]
        if sweep.parameter not in parameter_names:
            self.error(path, f'sweep parameter {sweep.parameter!r} is not a model parameter; valid parameters: '
                             f'{_name_list(parameter_names)}')
        if not isinstance(sweep.keep_fields, bool):
            self.error(path, f'sweep keep_fields must be true or false, got {short_repr(sweep.keep_fields)}')
        if not isinstance(sweep.reuse_build, bool):
            self.error(path, f'sweep reuse_build must be true or false, got {short_repr(sweep.reuse_build)}')
        elif sweep.reuse_build:
            from .sweep_reuse import reuse_issues   # only reused sweeps load it
            issues, ok = self._call(path, 'the reuse_build check', reuse_issues, self.spec, self.registry)
            for issue_path, message in (issues if ok else ()):
                self.error(issue_path, message)
        if not isinstance(sweep.values, list):
            self.error(path, f'the sweep values must be a list of numbers, got {short_repr(sweep.values)}')
            return
        if sweep.spacing not in SWEEP_SPACINGS:
            self.error(path, f'sweep spacing {short_repr(sweep.spacing)} is not one of {_name_list(SWEEP_SPACINGS)}')
            return
        if sweep.spacing != 'list' and not self._sweep_range_ok(path, sweep):
            return
        sweep_points = sweep.point_values()
        if not sweep_points:
            self.error(path, 'the sweep needs at least one value')
            return
        base_ok = (sweep.parameter in parameter_names and not self.vctx.parameter_errors
                   and not self.vctx.geometry_errors and bool(self.vctx.meshes)
                   and all(mesh is not None for mesh in self.vctx.meshes.values()))
        for index, raw in enumerate(sweep_points):
            value, message = self._scalar(raw)
            if message is not None:
                self.error(path, f'sweep value {index + 1}: {message}')
            elif base_ok and value != self.vctx.parameters.get(sweep.parameter):
                problem, ok = self._call(path, 'the sweep value check', self._sweep_value_problem, sweep.parameter,
                                         value)
                if ok and problem is not None:
                    text, level = problem
                    (self.error if level == 'error' else self.warning)(
                        path, f'sweep value {index + 1} ({sweep.parameter} = {value:g}): {text}')

    def _sweep_range_ok(self, path, sweep):
        """Report a bad linear / log sweep range (count, lower / upper, a log range through or at zero); True when ok."""
        if isinstance(sweep.count, bool) or not isinstance(sweep.count, int) or sweep.count < 1:
            self.error(path, f'sweep count must be a whole number of at least 1, got {short_repr(sweep.count)}')
            return False
        bounds = {}
        for name in ('lower', 'upper'):
            raw = getattr(sweep, name)
            if raw is None:
                self.error(path, f'the {sweep.spacing} sweep needs a {name} value')
                return False
            bounds[name], message = self._scalar(raw)
            if message is not None:
                self.error(path, f'sweep {name}: {message}')
                return False
        if sweep.spacing == 'log' and bounds['lower'] is not None and bounds['upper'] is not None \
                and not bounds['lower'] * bounds['upper'] > 0.0:
            self.error(path, f"a log sweep needs lower and upper of the same sign and non-zero, got "
                             f"{bounds['lower']:g} .. {bounds['upper']:g}")
            return False
        return True

    def _sweep_value_problem(self, parameter, value):
        """(text, issue level) of the first model problem the sweep value brings to a valid base model (parameters,
        geometry, domain size, mesh estimate: 'error'; memory estimate over memory_budget / host_budget:
        memory_check_level), as the study builds it (expression replaced by the value); None without one."""
        parameter_errors, geometry_errors = {}, {}
        parameters = evaluate_parameters([(item.name, repr(float(value)) if item.name == parameter else item.expression)
                                          for item in self.spec.parameters], errors=parameter_errors)
        if parameter_errors:
            name, message = next(iter(parameter_errors.items()))
            return f"parameter '{name}': {message}", 'error'
        shapes = resolve_geometry(self.spec, parameters, errors=geometry_errors)
        if geometry_errors:
            name, message = next(iter(geometry_errors.items()))
            return f"geometry object '{name}': {message}", 'error'
        try:
            domain_map(self.spec, parameters)   # the size, and a polar / spherical angle range
        except ValueError as exc:   # ExpressionError included
            return error_message(exc), 'error'
        try:
            meshes = estimate_meshes(self.spec, parameters, self.registry, shapes)
        except MESH_ESTIMATE_ERRORS as exc:
            return f'mesh estimate failed: {error_message(exc)}', 'error'
        gpu_budget = None if solver_backend(self.spec) == 'cpu' else self.memory_budget   # check_memory's rule
        if not gpu_budget and self.host_budget is None:
            return None
        try:
            estimate = backend_memory(self.spec, estimate_memory(self.spec, parameters, self.registry, meshes,
                                                                 shapes=shapes))
        except MESH_ESTIMATE_ERRORS as exc:
            return f'memory estimate failed: {error_message(exc)}', 'error'
        level = memory_check_level(self.spec)
        if gpu_budget and estimate.device_peak > gpu_budget:
            largest = '; '.join(f'{label} {format_bytes(size)}' for label, size in sorted(
                estimate.device_items(), key=lambda item: -item[1])[:2])
            return (f'estimated GPU memory {format_bytes(estimate.device_peak)} at the peak exceeds the GPU budget '
                    f'{format_bytes(self.memory_budget)} (largest: {largest}); narrow the sweep range or reduce the '
                    f'cell count'), level
        if self.host_budget is not None and estimate.host > self.host_budget:
            return f'{_host_memory_text(estimate, self.host_budget)}, or narrow the sweep range', level
        return None

    # ---- derived values and analytic curves ----
    def check_derived_values(self):
        parameter_names = frozenset(self.vctx.parameters)
        field_names = (parameter_names | COORDINATE_NAMES   # x, y, z (3D only, checked below), t (Session.evaluate)
                       | frozenset(self.output_names) | frozenset(self.scalar_names))
        curve_names = [curve.name for curve in self.spec.derived_curves]
        derived_names = [derived.name for derived in self.spec.derived_values]
        for index, derived in enumerate(self.spec.derived_values):
            path = ('derived', derived.name)
            earlier_names = frozenset(name for name in derived_names[:index] if isinstance(name, str))
            self._check_derived_references(path, derived, earlier_names, frozenset(derived_names[index + 1:]),
                                           field_names)
            global_value = derived.source is None and derived.operation == GLOBAL_DERIVED_OPERATION
            interface = (interface_selection(self.spec, derived.selection)
                         if derived.source is None and not global_value else None)
            available_names = field_names | (frozenset(INTERFACE_VARIABLES) if interface is not None else frozenset())
            if global_value:   # fields, coordinates and itself: _check_global_expression says why
                available_names = available_names | {derived.name}
            source_kind = None   # result kind of the source ('xy' for a derived curve), None: none or unresolved
            if is_curve_source(derived.source):
                if derived.source in curve_names:
                    available_names, source_kind = parameter_names | {derived.source, 'x'}, 'xy'
                else:
                    available_names = None
                    self.error(path, f"source: no derived curve named {derived.source!r} (a feature result is "
                                     f"'physics.feature'); derived curves: {_name_list(curve_names)}")
            elif derived.source is not None:
                feature_type = self._result_feature(path, 'source', derived.source, DERIVED_RESULT_KINDS)
                source_kind = None if feature_type is None else feature_type.result_kind
                sample_names = None if feature_type is None else RESULT_SAMPLE_NAMES[source_kind][1]
                # xy series / table columns are known only at run time (a missing name gives NaN + a warning)
                available_names = None if sample_names is None else parameter_names | frozenset(sample_names)
            if available_names is not None:
                available_names = available_names | earlier_names
            if derived.operation not in DERIVED_OPERATIONS:
                self.error(path, f"unknown operation {derived.operation!r}; valid: {', '.join(DERIVED_OPERATIONS)}")
            elif derived.source is None and derived.operation not in SOURCE_FREE_DERIVED_OPERATIONS:
                self.error(path, f"operation '{derived.operation}' needs a source (feature result or derived curve); "
                                 f"over fields: {', '.join(FIELD_DERIVED_OPERATIONS)}; without fields: "
                                 f"'{GLOBAL_DERIVED_OPERATION}'")
            elif derived.source is not None and derived.operation == GLOBAL_DERIVED_OPERATION:
                self.error(path, f"operation '{GLOBAL_DERIVED_OPERATION}' evaluates one number of parameters, scalar "
                                 'outputs, t and the derived values listed above, not a source: clear the source')
            elif derived.source is not None and derived.operation not in SOURCE_DERIVED_OPERATIONS:
                self.error(path, f"operation '{derived.operation}' reduces fields (source null); over a source: "
                                 f"{', '.join(SOURCE_DERIVED_OPERATIONS)}")
            elif interface is not None and derived.operation not in INTERFACE_OPERATIONS:
                self.error(path, f"operation '{derived.operation}' over the interface selection '{interface.name}'; "
                                 f"valid: {', '.join(INTERFACE_OPERATIONS)}")
            self._check_derived_reduction(path, derived, source_kind)
            self._check_expression_text(path, 'expression', derived.expression, available_names)
            if global_value:
                self._check_global_expression(path, derived, field_names, earlier_names)
            elif derived.source is None:
                self._check_height_in_2d(path, 'expression', derived.expression)
                self._check_derived_meshes(path, derived)
            if global_value:
                if derived.selection:
                    self.error(path, f"selection: operation '{GLOBAL_DERIVED_OPERATION}' reads no fields; clear the "
                                     'selection')
            elif derived.source is None and interface is None:   # an interface: checked at its selection node
                problems, expanded = self._entity_problems(derived.selection, 'domain')
                for problem in problems:
                    self.error(path, f'selection: {problem}')
                if expanded and not problems:
                    self._check_object_cells(path, expanded)
            elif derived.source is not None and derived.selection:
                self.error(path, 'selection: a derived value of a feature result reduces over its samples; '
                                 'clear the selection')
            self._check_derived_point(path, derived)
            if derived.source is not None and derived.operation == 'at' and derived.at is None:
                self.error(path, "at: operation 'at' needs the abscissa value at (m for wavelength, s for t, the x "
                                 'unit of an xy result, the 0-based row of a table)')
            elif derived.at is not None and (derived.operation != 'at' or derived.source is None):
                self.warning(path, "at is used only by operation 'at' over a source (over fields: point)")
            for what, raw in (('at', derived.at), ('reference', derived.reference)):
                message = None if raw is None else self._scalar(raw)[1]
                if message is not None:
                    self.error(path, f'{what}: {message}')
            self._check_tolerance(path, derived.tolerance)
            if derived.tolerance is not None and derived.reference is None:
                self.warning(path, 'tolerance: applies to derived_error, which needs a reference; set one or clear '
                                   'the tolerance')

    def _check_derived_meshes(self, path, derived):
        """S7.2 D3: an expression naming fields of physics on several meshes is evaluated on the mesh of the first of
        those physics in model order, the others interpolated onto it (info)."""
        names = _referenced_or_empty(derived.expression)
        field_physics = self.output_names
        physics_order = [physics.name for physics in self.spec.physics]
        named = sorted({field_physics[name][0] for name in names if name in field_physics},
                       key=lambda name: physics_order.index(name) if name in physics_order else len(physics_order))
        meshes = {name: self._mesh_name_of(self.physics_by_name[name]) for name in named
                  if name in self.physics_by_name}
        if len({mesh for mesh in meshes.values() if mesh is not None}) < 2:
            return
        first = next(name for name in named if meshes.get(name) is not None)
        others = ', '.join(f"{name} (mesh '{mesh}')" for name, mesh in meshes.items()
                           if mesh is not None and mesh != meshes[first])
        self.info(path, f"evaluated on mesh '{meshes[first]}' of '{first}' (the first of its physics in model order); "
                        f'the fields of {others} are interpolated onto it')

    def _check_derived_references(self, path, derived, earlier_names, later_names, field_names):
        """Errors at path for derived values its expression names that are listed after it, or that it could not
        read because a parameter / field / scalar / reserved name shadows them."""
        try:
            names = referenced_names(derived.expression)
        except ExpressionError:
            return   # reported by _check_expression_text
        for name in sorted(names & later_names - earlier_names):
            self.error(path, f"expression: derived value '{name}' is listed after this one; move it above to use "
                             'its value here')
        for name in sorted(names & earlier_names):
            if name in field_names or name in RESERVED_NAMES or name in INTERFACE_VARIABLES:
                self.error(path, f"expression: '{name}' names both a derived value and a parameter, field, scalar "
                                 'output or reserved name; rename the derived value to use it here')

    def _check_global_expression(self, path, derived, field_names, earlier_names):
        """Errors at path for an operation 'expression' value naming fields / coordinates (nothing reduces them) or
        itself; forward and unknown names are reported by _check_derived_references / _check_expression_text."""
        names = _referenced_or_empty(derived.expression)
        plain_names = frozenset(self.vctx.parameters) | frozenset(self.scalar_names) | {'t'} | earlier_names
        for name in sorted(names & field_names - plain_names):
            self.error(path, f"expression: '{name}' is a field or coordinate, and operation "
                             f"'{GLOBAL_DERIVED_OPERATION}' reduces no fields; reduce it in a derived value listed "
                             'above (e.g. operation integral) and name that value here')
        if derived.name in names and derived.name not in plain_names | field_names:
            self.error(path, f"expression: names '{derived.name}' itself; a derived value reads only the values "
                             'listed above it')

    def _check_derived_point(self, path, derived):
        """Errors at path for DerivedValueSpec.point: a point object, only for field operations 'at' (required) and
        'mode_volume'."""
        uses_point = derived.source is None and derived.operation in ('at', 'mode_volume')
        if derived.point in ('', None):
            if derived.source is None and derived.operation == 'at':
                self.error(path, "point: operation 'at' over fields reads the cell holding a point object; set "
                                 f"point (point objects: {_name_list(point_entities(self.spec))})")
            return
        if not uses_point:
            self.error(path, "point applies to the field operations 'at' and 'mode_volume' only; clear it")
            return
        obj = self.objects.get(derived.point)
        if obj is None or obj.kind != 'point':
            self.error(path, f"point: {short_repr(derived.point)} is no point object; point objects: "
                             f"{_name_list(point_entities(self.spec))}")

    def _check_derived_reduction(self, path, derived, source_kind):
        """Errors / warnings at path for window, wavelength, fit and degree of a derived value (source_kind: the
        source's result kind, None when there is none or it is unresolved)."""
        operation = derived.operation
        if derived.window != []:
            if not isinstance(derived.window, list) or len(derived.window) != 2:
                self.error(path, f'window: expected [lo, hi] in the abscissa unit or [], got '
                                 f'{short_repr(derived.window)}')
            elif derived.source is None:
                self.error(path, 'window: applies to a source result or derived curve only; clear it')
            else:
                bounds = [self._scalar(raw) for raw in derived.window]
                for (value, message), side in zip(bounds, ('lo', 'hi')):
                    if message is not None:
                        self.error(path, f'window {side}: {message}')
                if all(message is None for _value, message in bounds) and bounds[0][0] == bounds[1][0]:
                    self.error(path, f'window: lo and hi are equal ({bounds[0][0]:g}): the window is empty')
                if operation in ('last', 'at'):
                    self.warning(path, f"window is not used by operation '{operation}'")
        if derived.wavelength is not None:
            value, message = self._scalar(derived.wavelength)
            if message is not None:
                self.error(path, f'wavelength: {message}')
            elif not value > 0.0:
                self.error(path, f'wavelength: must be > 0 m, got {value:g}')
            if source_kind is not None and source_kind != 'far_field':
                self.error(path, f'wavelength: applies to far_field sources only ({derived.source} is a '
                                 f'{source_kind} result); clear it')
        if operation in FIT_QUANTITIES:
            problem = fit_problem(operation, derived.fit, derived.degree)
            if problem is not None:
                self.error(path, problem)
        elif operation in DERIVED_OPERATIONS and (derived.fit or derived.degree != 1):
            self.warning(path, f"fit and degree are used only by the fit operations "
                               f"({', '.join(FIT_QUANTITIES)}); operation '{operation}' ignores them")
        if operation in COMPLEX_FIT_OPERATIONS and derived.source is not None:
            # the '<name>_im' sibling and its x samples are known only at run time (xy series): result_value checks them
            if is_curve_source(derived.source):
                self.error(path, f"operation '{operation}' pairs the '<name>_re' and '<name>_im' series of one "
                                 "feature result; a derived curve has a single series (use a 'physics.feature' xy "
                                 "source)")
            elif source_kind is not None and source_kind != 'xy':
                self.error(path, f"operation '{operation}' needs an xy source with '<name>_re' / '<name>_im' series "
                                 f'({derived.source} is a {source_kind} result)')
            try:
                complex_fit_series(derived.expression, operation)
            except ValueError as exc:
                self.error(path, f'expression: {exc}')

    def check_analytic_curves(self):
        parameter_names = frozenset(self.vctx.parameters)
        for curve in self.spec.analytic_curves:
            path = ('curve', curve.name)
            variables = None   # None: target unresolved, its variables unknown
            target_kind = None
            if curve.target == 'sweep':
                variables = CURVE_VARIABLES['sweep']
                if self.spec.study.sweep is None:
                    self.warning(path, 'target sweep: the study has no parameter sweep, so the curve is not drawn')
            elif curve.target == 'line':
                variables = CURVE_VARIABLES['line']
                self._check_curve_line(path, curve)
                self._check_curve_cut(path, curve)
                self._check_height_in_2d(path, 'expression', curve.expression)
            else:
                feature_type = self._result_feature(path, 'target', curve.target, CURVE_RESULT_KINDS,
                                                    alternatives="'sweep', 'line' or ")
                if feature_type is not None:
                    target_kind = feature_type.result_kind
                    variables = curve_variables(target_kind, curve.series)
            if curve.target != 'line' and (curve.line or curve.field or curve.plane is not None
                                           or curve.snapshot is not None):
                self.error(path, "line, field, plane and snapshot apply to target 'line' only; clear them")
            if curve.target != 'line' and curve.sampling != 'nearest':
                self.error(path, "sampling applies to target 'line' only; set it to 'nearest'")
            if curve.interface != '':
                self._check_curve_interface(path, curve)
            if curve.projection != '':
                self._check_curve_projection(path, curve)
            if curve.x_column and target_kind != 'table' and (target_kind is not None
                                                               or curve.target in ('sweep', 'line')):
                self.error(path, f"x_column applies to 'table' targets only; {curve.target} is not one: clear it")
            self._check_expression_text(path, 'expression', curve.expression,
                                        None if variables is None else parameter_names | frozenset(variables))
            self._check_tolerance(path, curve.tolerance)
            if curve.tolerance is not None and curve.series in (FFT_SERIES, RATIO_SERIES):
                self.warning(path, f"tolerance: a curve with series '{curve.series}' is never compared (curve_error "
                                   f'note), so selftest --examples fails it; clear the tolerance')

    def _check_curve_interface(self, path, curve):
        """Errors at path for an analytic curve interface: target 'line' of a 2D model only, naming an 'interface'
        selection."""
        interfaces = [selection.name for selection in self.spec.selections if selection.kind == 'interface']
        if curve.target != 'line' or self.spec.is_3d:
            self.error(path, "interface applies to target 'line' in 2D models only; clear it")
        elif curve.interface not in interfaces:
            self.error(path, f'interface: {short_repr(curve.interface)} is no interface selection; interface '
                             f'selections: {_name_list(interfaces)}')

    def _check_curve_projection(self, path, curve):
        """Errors at path for an analytic curve projection: one of INTERFACE_PROJECTIONS, with an interface, on an x
        component whose y partner (partner_component) is an output field of the same physics; a warning when a
        snapshot curve's partner is not among explicit stored fields."""
        if curve.projection not in INTERFACE_PROJECTIONS:
            self.error(path, f"projection: expected '' or one of {', '.join(INTERFACE_PROJECTIONS)}, got "
                             f'{short_repr(curve.projection)}')
            return
        if curve.interface == '':
            self.error(path, 'projection: needs an interface (the outline giving the normal); set interface or '
                             'clear projection')
            return
        try:
            physics_name, component = split_key(curve.field, 'physics.component')
            physics = self.physics_by_name[physics_name]
            components = [info.name for info in self.registry.physics_type(physics.type).output_fields]
        except (ValueError, KeyError):   # reported by _check_curve_line / at the physics
            return
        partner = partner_component(component)
        if partner not in components:
            x_components = [name for name in components if partner_component(name) in components]
            self.error(path, f"projection: field '{curve.field}' is no x component of an in-plane vector whose y "
                             f"component is a field of '{physics_name}' too ('ex_re' -> 'ey_re', 'E_x' -> 'E_y'); x "
                             f"components: {', '.join(f'{physics_name}.{name}' for name in x_components) or 'none'}")
            return
        stored = self.spec.study.stored_fields
        if curve.snapshot is not None and stored and f'{physics_name}.{partner}' not in stored:
            self.warning(path, f"projection: the y component '{physics_name}.{partner}' is not a stored field of the "
                               'study, so the snapshot profile fails')

    def check_derived_curves(self):
        parameter_names = frozenset(self.vctx.parameters)
        for curve in self.spec.derived_curves:
            path = ('derived_curve', curve.name)
            aliases = []
            if not isinstance(curve.inputs, dict) or not curve.inputs:
                self.error(path, "inputs: give at least one alias -> 'physics.feature:sample' entry")
            else:
                for alias, reference in curve.inputs.items():
                    if not is_identifier(alias) or is_reserved_name(alias):
                        self.error(path, f'inputs: alias {alias!r} must be an identifier and not a reserved name')
                        continue
                    aliases.append(alias)
                    try:
                        key, _sample = split_curve_input(reference)
                    except ValueError as exc:
                        self.error(path, f'inputs {alias}: {exc}')
                        continue
                    # the sample names of xy / table results are known only at run time (an error text then)
                    self._result_feature(path, f'inputs {alias}', key, DERIVED_RESULT_KINDS)
            self._check_expression_text(path, 'expression', curve.expression,
                                        parameter_names | frozenset(aliases) | frozenset(DERIVED_CURVE_VARIABLES),
                                        functions=CURVE_FUNCTION_ARITY)
            if curve.plot and not is_identifier(curve.plot):
                self.error(path, f'plot: {curve.plot!r} must be an identifier (it names figures/derived_<plot>.png)')
            if curve.reference is not None and not isinstance(curve.reference, (int, float)):
                self._check_expression_text(path, 'reference', curve.reference,
                                            parameter_names | frozenset(DERIVED_CURVE_VARIABLES))
            elif isinstance(curve.reference, bool):
                self.error(path, f'reference: expected a number or an expression of x and parameters, got '
                                 f'{curve.reference!r}')
            self._check_tolerance(path, curve.tolerance)
            if curve.tolerance is not None and curve.reference is None:
                self.warning(path, 'tolerance: applies to the curve_error against a reference; set one or clear '
                                   'the tolerance')

    def _check_tolerance(self, path, tolerance):
        """Error at path unless tolerance is None or a finite number >= 0."""
        if tolerance is None:
            return
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance < 0):
            self.error(path, f'tolerance: expected a finite number >= 0 (relative error) or null, got '
                             f'{short_repr(tolerance)}')

    def _check_curve_line(self, path, curve):
        """Errors at path for an analytic curve line that is not 4 finite coordinates of two distinct points or a
        field that is no output field 'physics.component' of the model."""
        try:
            physics_name, component = split_key(curve.field, 'physics.component')
        except ValueError as exc:
            self.error(path, f'field: {exc}')
        else:
            physics = self.physics_by_name.get(physics_name)
            try:
                components = ([] if physics is None else
                              [info.name for info in self.registry.physics_type(physics.type).output_fields])
            except KeyError:   # unknown physics type: reported at the physics
                components = None
            if components is not None and component not in components:
                self.error(path, f"field: no output field '{curve.field}'; fields: "
                                 f"{', '.join(f'{physics_name}.{name}' for name in components) or 'none'}")
        segment = is_segment_3d(curve.line)
        if not isinstance(curve.line, list) or not (len(curve.line) == 4 or segment):
            self.error(path, f'line: expected [x0, y0, x1, y1] (3D models also the segment [x0, y0, z0, x1, y1, z1]) '
                             f'in m, got {short_repr(curve.line)}')
            return
        if segment and not self.spec.is_3d:
            self.error(path, 'line: the 3D segment form [x0, y0, z0, x1, y1, z1] needs a 3D model (domain size_z); '
                             'use [x0, y0, x1, y1]')
            return
        if segment and self.spec.domain.coordinates == 'polar':
            self.error(path, 'line: the 3D segment form samples Cartesian 3D models; on a cylindrical (r, phi, z) '
                             'sector give [x0, y0, x1, y1] with plane = the cut height')
            return
        labels = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1') if segment else ('x0', 'y0', 'x1', 'y1')
        coordinates = []
        for label, raw in zip(labels, curve.line):
            value, message = self._scalar(raw)
            if message is not None:
                self.error(path, f'line {label}: {message}')
            coordinates.append(value)
        half = len(coordinates) // 2
        if None not in coordinates and coordinates[:half] == coordinates[half:]:
            self.error(path, 'line: the start and end point coincide')

    def _check_curve_cut(self, path, curve):
        """Errors at path for a line curve plane (3D models only, a number, not with a snapshot or a 3D segment) or
        snapshot (an int or a label); a warning when the snapshot field of a 2D-form line is not among explicit stored
        fields (a 3D segment's profile is stored with every snapshot); an error for a sampling other than
        SEGMENT_SAMPLINGS or trilinear without a 3D segment."""
        if curve.sampling not in SEGMENT_SAMPLINGS:
            self.error(path, f"sampling: expected one of {', '.join(SEGMENT_SAMPLINGS)}, got "
                             f'{short_repr(curve.sampling)}')
        elif curve.sampling != 'nearest' and not is_segment_3d(curve.line):
            self.error(path, f"sampling '{curve.sampling}' applies to a 3D segment line [x0, y0, z0, x1, y1, z1] "
                             "only (a line in a cut samples its containing cells); set sampling 'nearest'")
        segment = is_segment_3d(curve.line)
        if segment and curve.plane is not None:
            self.error(path, 'plane: a 3D segment line gives its own z; clear plane')
        if curve.plane is not None:
            if not self.spec.is_3d:
                self.error(path, 'plane: a cut height applies to 3D models only; clear it')
            message = self._scalar(curve.plane)[1]
            if message is not None:
                self.error(path, f'plane: {message}')
            if curve.snapshot is not None:
                self.error(path, 'plane: snapshots hold the mid-height cut only; clear plane or snapshot')
        snapshot = curve.snapshot
        if snapshot is None:
            return
        if isinstance(snapshot, bool) or not isinstance(snapshot, (int, str)) or snapshot == '':
            self.error(path, f'snapshot: expected an index (int, negative from the end) or a snapshot label such as '
                             f"'mode 1' or 'resonance 2', got {short_repr(snapshot)}")
        stored = self.spec.study.stored_fields
        if stored and curve.field not in stored and not segment:
            self.warning(path, f"snapshot: field '{curve.field}' is not a stored field of the study (stored: "
                               f"{', '.join(map(str, stored))}), so the profile fails")

    def _result_feature(self, path, what, key, kinds, alternatives=''):
        """FeatureType of the feature named by the result key 'physics.feature' when its result kind is in kinds,
        else None after an error at path (a disabled feature only warns: it has no result)."""
        try:
            physics_name, feature_name = split_key(key)
        except ValueError:
            self.error(path, f"{what}: expected {alternatives}'physics.feature', got {short_repr(key)}")
            return None
        physics = self.physics_by_name.get(physics_name)
        if physics is None:
            known = ', '.join(self.physics_by_name) or 'none'
            self.error(path, f"{what}: no physics '{physics_name}'; physics: {known}")
            return None
        feature = next((item for item in physics.features if item.name == feature_name), None)
        if feature is None:
            names = ', '.join(item.name for item in physics.features) or 'none'
            self.error(path, f"{what}: physics '{physics_name}' has no feature '{feature_name}'; features: {names}")
            return None
        try:
            feature_type = self.registry.physics_type(physics.type).feature_type(feature.type)
        except KeyError:   # unknown types are reported at the physics / feature
            return None
        if feature_type.result_kind not in kinds:
            self.error(path, f"{what}: '{key}' ({feature_type.name}) gives {feature_type.result_kind or 'no'} result; "
                             f"valid result kinds: {', '.join(kinds)}")
            return None
        if not feature.enabled:
            self.warning(path, f"{what}: feature '{key}' is disabled, so it has no result")
        return feature_type

    def _check_expression_text(self, path, what, raw, available_names, functions=None):
        """Error at path when raw is no expression text or (available_names not None) uses other names; functions:
        extra call names {name: (min, max)} it may use."""
        message = None
        if not isinstance(raw, str):
            message = f'must be text, got {short_repr(raw)}'
        else:
            try:
                expression = compile_expression(raw, functions)
                if available_names is not None:
                    expression.check_names(available_names)
            except ExpressionError as exc:
                message = self._failed_parameter_message(raw) or str(exc)
        if message is not None:
            self.error(path, f'{what}: {message}')

    def _check_height_in_2d(self, path, what, raw):
        """Error at path when the expression text raw uses z in a 2D model."""
        message = self._height_in_2d_message(raw)
        if message is not None:
            self.error(path, f'{what}: {message}')

    # ---- value helpers ----
    def _check_values(self, path, params, raw_values, what, allowed_names, check_refs=True):
        """Report unknown keys and problems of every visible param; True when there are none."""
        if not isinstance(raw_values, dict):
            self.error(path, f'{what}s must be a mapping of name -> value, got {short_repr(raw_values)}')
            return False
        known = [param.name for param in params]
        unknown = sorted(str(key) for key in raw_values if key not in known)
        values_ok = True
        if unknown:
            self.error(path, f"unknown {what}{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}; "
                             f"valid: {', '.join(known) or 'none'}")
            values_ok = False
        filled = fill_defaults(params, raw_values)
        for param in params:
            if not param_visible(param, filled):
                continue
            raw = filled[param.name]
            message = self._value_message(param, raw, allowed_names)
            if message is None and check_refs:
                message = self._reference_problem(param, raw)
            if message is not None:
                where = f'{param.label} [{param.name}]' if param.label and param.label != param.name else param.name
                self.error(path, f'{where}: {message}')
                values_ok = False
        return values_ok

    def _value_message(self, param, raw, allowed_names):
        """check_value message, replaced by a pointer to the failed parameter the value depends on; z in a 2D model
        (expression kinds only: a choice / text value such as axis 'z' is no coordinate)."""
        message = check_value(param, raw, self.vctx.parameters, allowed_names)
        if message is None:
            return self._height_in_2d_message(raw) if param.kind in _EXPRESSION_KINDS else None
        return self._failed_parameter_message(raw) or message

    def _height_in_2d_message(self, raw):
        """Error text when the expression text raw uses z (THREE_D_COORDINATE_NAMES) in a 2D model, else None."""
        if self.spec.is_3d or not isinstance(raw, str):
            return None
        try:
            used = sorted(referenced_names(raw) & THREE_D_COORDINATE_NAMES)
        except ExpressionError:
            return None   # reported by the value check
        if not used:
            return None
        return (f"uses {', '.join(used)}, the height of 3D models; this model is 2D (no domain size_z): use x, y "
                f'only, or set size_z')

    def _check_number(self, path, label, raw, strict=True, extra=''):
        """The scalar of raw when it is > 0 (strict) or >= 0, else None with the error '<label>: <problem>' at path
        (extra: text after the bound in the message)."""
        value, message = self._scalar(raw)
        if message is None and not (value > 0 if strict else value >= 0):
            message = f"must be {'>' if strict else '>='} 0{extra}, got {value:g}"
        if message is not None:
            self.error(path, f'{label}: {message}')
            return None
        return value

    def _scalar(self, raw):
        """(float, None) for a valid number / parameter expression, else (None, message)."""
        try:
            return evaluate_scalar(raw, self.vctx.parameters), None
        except ValueError as exc:   # ExpressionError
            return None, self._failed_parameter_message(raw) or str(exc)

    def _failed_parameter_message(self, raw):
        """'depends on parameter p, which has an error' when raw references a failed parameter, else None."""
        parameter_errors = self.vctx.parameter_errors
        if not parameter_errors:
            return None
        failed, pending = set(), [raw]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                try:
                    failed.update(referenced_names(item).intersection(parameter_errors))
                except ExpressionError:
                    pass
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
        return f"depends on parameter '{min(failed)}', which has an error" if failed else None

    def _reference_problem(self, param, raw):
        """Message when a 'ref' / 'names' value names a material, geometry, physics or parameter that does not exist."""
        if not param.ref_kind or param.kind not in ('ref', 'names') or raw is None or raw == '':
            return None
        known = {'material': self.spec.materials, 'geometry': self.spec.geometry, 'physics': self.spec.physics,
                 'parameter': self.spec.parameters}[param.ref_kind]
        known_names = [item.name for item in known]
        missing = [name for name in ([raw] if param.kind == 'ref' else raw) if name not in known_names]
        if missing:
            return (f"unknown {param.ref_kind} {', '.join(repr(name) for name in missing)}; "
                    f"valid: {_name_list(known_names)}")
        return None

    def _check_every(self, path, every, allow_none):
        if every is None and allow_none:
            return
        if not valid_every(every):
            self.error(path, f'every must be an integer >= 1, got {every!r}')


# ---- module helpers ----
def _mesh_estimates(spec, registry, parameters, mesh_estimate, mesh_error, shapes=None):
    """({used mesh name: estimate | None}, {name: failure message | None}) in spec.used_meshes order: the given ones (a
    dict, or one estimate / message of the default mesh; a given estimate wins over a message), the rest estimated
    here on shapes (a crashing definition callback becomes an internal-error message)."""
    names = [mesh_spec.name for mesh_spec in spec.used_meshes()]
    default = names[0] if names else ''
    given = (mesh_estimate if isinstance(mesh_estimate, dict) else {} if mesh_estimate is None
             else {getattr(mesh_estimate, 'name', default): mesh_estimate})
    given_errors = mesh_error if isinstance(mesh_error, dict) else {default: mesh_error} if mesh_error else {}
    meshes, errors = {}, {}
    for name in names:
        estimate, message = given.get(name), given_errors.get(name) or None
        if estimate is None and message is None:
            try:
                estimate = estimate_mesh(spec, parameters, registry, name, shapes)
            except MESH_ESTIMATE_ERRORS as exc:
                message = error_message(exc)
            except Exception as exc:   # a crashing definition callback (domain_requirements) must not escape
                message = internal_error_text('the mesh estimate', exc)
        meshes[name], errors[name] = estimate, None if estimate is not None else message
    return meshes, errors


def _gmsh_found():
    """True when the gmsh program is available: OPEN_SOL_GMSH names an existing file, or gmsh is on the PATH."""
    import shutil
    configured = os.environ.get(_GMSH_ENVIRONMENT)
    return os.path.isfile(configured) if configured else shutil.which('gmsh') is not None


def _host_memory_text(estimate, host_budget):
    """Issue text of a large host estimate, judged against the available RAM host_budget (None: unknown)."""
    largest = '; '.join(f'{label} {format_bytes(size)}' for label, size in estimate.largest_items(2, host=True))
    if host_budget is None:
        judged = ': the CPU factorisation may exhaust the RAM and swap'
    else:
        verb = 'exceeds' if estimate.host > host_budget else 'is close to'
        judged = f' {verb} the available RAM {format_bytes(host_budget)}: the CPU factorisation may fail or swap'
    return (f'estimated host memory {format_bytes(estimate.host)} ({largest}; factor fill estimated a priori)'
            f'{judged}; reduce the cell count, use a GPU linear solver where offered or close other programs')


def _name_list(names):
    """', '-joined text names in the given order (non-text names skipped), or 'none'."""
    return ', '.join(name for name in names if isinstance(name, str)) or 'none'


def _material_property_values(material):
    """[(description, raw value)] of a material's property values ([] for a malformed properties dict: reported at
    the material); dispersion pole values take parameters only, so they never read a field."""
    if not isinstance(material.properties, dict):
        return []
    return [(f"property '{key}'", raw) for key, raw in material.properties.items()]


def _referenced_or_empty(raw):
    """Names an expression value references; empty for numbers, None and malformed values."""
    try:
        return referenced_names(raw)
    except ExpressionError:
        return frozenset()


def _mapped_window_centres(mesh):
    """(x, y) float64 (nx, ny) drawing-plane cell centres [m] of a polar / spherical MeshEstimate (no PML)."""
    x_edges, y_edges = mesh.x_edges, mesh.y_edges
    return mesh.domain_map.to_physical(0.5 * (x_edges[:-1] + x_edges[1:])[:, None],
                                       0.5 * (y_edges[:-1] + y_edges[1:])[None, :])


def _window_edges(mesh):
    """(x edges, y edges) float64 [m] of the physical window cells of a MeshEstimate (PML cells dropped)."""
    sides = mesh.npml_sides
    return (mesh.x_edges[sides['x_lo']:mesh.nx - sides['x_hi'] + 1],
            mesh.y_edges[sides['y_lo']:mesh.ny - sides['y_hi'] + 1])


def _rasterised_cell_count(name, shapes, x_centres, y_centres):
    """Cells rasterize_window tests for domain object name: the bounding-box sub-grids of its primitives, summed."""
    shape = shapes[name]
    kind = GEOMETRY_KINDS[shape.kind]
    if kind.composite:
        return sum(_rasterised_cell_count(item, shapes, x_centres, y_centres)
                   for item in composite_inputs(kind, shape.values))
    x0, y0, x1, y1 = bounding_box(name, shapes)
    n_columns = int(np.searchsorted(x_centres, x1, side='right') - np.searchsorted(x_centres, x0, side='left'))
    n_rows = int(np.searchsorted(y_centres, y1, side='right') - np.searchsorted(y_centres, y0, side='left'))
    return max(n_columns, 0) * max(n_rows, 0)


def _first_by_name(items):
    """name -> first item with that name."""
    table = {}
    for item in items:
        table.setdefault(item.name, item)
    return table


def _name_problem(name, category):
    """Message when `name` is not a valid node name, else None."""
    if name is None or name == '':
        return f'the {category} has no name'
    if not is_identifier(name):
        return f'{name!r} is not a valid {category} name (letters, digits and _, not starting with a digit)'
    return None


def internal_error_text(owner, exc):
    """Issue text for an exception raised inside a definition callback: type, message and raising file:line."""
    frames = traceback.extract_tb(exc.__traceback__)
    location = f' ({os.path.basename(frames[-1].filename)}:{frames[-1].lineno})' if frames else ''
    return f'internal error in {owner}: {type(exc).__name__}: {exc}{location}'

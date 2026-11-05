"""Model layer: a declarative, JSON-serialisable model spec built into GPU solvers through a physics registry.
Only the CPU-safe modules are imported here (no GPU work); builder/session/study/worker are imported explicitly."""
from .expressions import (
    ExpressionError, Expression, compile_expression, evaluate_scalar, evaluate_parameters,
    referenced_names, is_identifier, rename_in_expression,
)
from .schema import (
    Param, FeatureType, DomainRequirements, PhysicsType, CouplingType, StudyType, Issue,
    TemplateEntry, SelfTestCase, SelfTestVariant, RESULT_KIND_DOCS, node_label,
    ValidationContext, Registry, get_registry, state_field_names, output_field_names,
    resolve_values, check_value, param_visible,
)
from .spec import (
    SPEC_VERSION, OUTER_FACES, OUTER_FACES_3D, RESERVED_ENTITY_NAMES, outer_faces,
    ParameterSpec, GeometrySpec, SelectionSpec, MaterialSpec, FeatureSpec, PhysicsSpec, CouplingSpec,
    RefinementSpec, MeshSpec, SweepSpec, StudySpec, DerivedValueSpec, AnalyticCurveSpec, DerivedCurveSpec, DomainSpec,
    MeshFaceSizeSpec, ModelSpec,
)
from .geometry import (
    GEOMETRY_KINDS, GeometryKind, ResolvedShape, resolve_geometry, expand_entities,
    material_preview,
)

__all__ = [
    'ExpressionError', 'Expression', 'compile_expression', 'evaluate_scalar', 'evaluate_parameters',
    'referenced_names', 'is_identifier', 'rename_in_expression',
    'Param', 'FeatureType', 'DomainRequirements', 'PhysicsType', 'CouplingType', 'StudyType', 'Issue',
    'TemplateEntry', 'SelfTestCase', 'SelfTestVariant', 'RESULT_KIND_DOCS', 'node_label',
    'ValidationContext', 'Registry', 'get_registry', 'state_field_names', 'output_field_names',
    'resolve_values', 'check_value', 'param_visible',
    'SPEC_VERSION', 'OUTER_FACES', 'OUTER_FACES_3D', 'RESERVED_ENTITY_NAMES', 'outer_faces',
    'ParameterSpec', 'GeometrySpec', 'SelectionSpec', 'MaterialSpec', 'FeatureSpec', 'PhysicsSpec',
    'CouplingSpec', 'RefinementSpec', 'MeshSpec', 'MeshFaceSizeSpec', 'SweepSpec', 'StudySpec', 'DerivedValueSpec', 'AnalyticCurveSpec',
    'DerivedCurveSpec', 'DomainSpec', 'ModelSpec',
    'GEOMETRY_KINDS', 'GeometryKind', 'ResolvedShape', 'resolve_geometry', 'expand_entities',
    'material_preview',
]

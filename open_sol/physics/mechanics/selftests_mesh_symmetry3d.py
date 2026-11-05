"""Electrode-edge mesh sizes and the mirror-plane check of solid_mechanics_3d (piezo_3d_plan stage 6): gmsh3d edge
distances, MathEval edge fields and a gmsh mesh refined along an electrode box (skipped without gmsh), the
electrode_edge_size setting through tetrahedron_sizing and the mesh estimate, crystal_tensors.mirror_violation and the
Roller warning. definition3d registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import lithium_niobate_entry
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec

UM = 1e-6
_SOLID = 'solid'


def _box(name, low, high):
    return GeometrySpec(name, 'box', {'x0': low[0], 'y0': low[1], 'z0': low[2], 'x1': high[0], 'y1': high[1],
                                      'z1': high[2]})


# ---- fem_gmsh3d_edge_sizes: edge distances, the .geo edge fields, a gmsh mesh refined along an electrode box ----
# Box (0, 0, 0)-(1, 1, 1) L and a z cylinder (r 0.5 L, height L): Solid.edge_distance at points with closed-form
# distances (exact to round-off); geo_text writes one MathEval per box edge (12) and cylinder rim (2), all in the Min
# field. With gmsh: a 1 x 1 x 0.3 um window, h_max 0.1 um, with a 0.4 x 0.4 x 0.1 um pad box on top (edge_size 20 nm,
# growth 1.3): every tetrahedron with a node within edge_size / 2 of a pad edge has edges <= _EDGE_RATIO x edge_size
# (gmsh meets its target size within ~1.5; first estimate), and the mesh has more elements than without the edge size.
_DISTANCE_TOLERANCE = 1e-12
_EDGE_H_MAX = 0.1 * UM
_EDGE_SIZE = 0.02 * UM
_EDGE_GROWTH = 1.3
_EDGE_RATIO = 3.0
_EDGE_WINDOW = (0.0, 0.0, 0.0, 1.0 * UM, 1.0 * UM, 0.3 * UM)
_EDGE_PAD = (0.3 * UM, 0.3 * UM, 0.2 * UM, 0.7 * UM, 0.7 * UM, 0.3 * UM)


def _check_edge_sizes(_):
    """Closed-form edge distances, the MathEval rows of geo_text and (gmsh) the element sizes along the pad edges."""
    from ...fem.gmsh3d import Solid, geo_text, mesh_solids_with_gmsh
    from ...fem.gmsh_io import find_gmsh
    errors, details = {}, {}
    box = Solid('box', 'box', (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), edge_size=0.1)
    cylinder = Solid('rod', 'cylinder', (0.0, 0.0, 0.0, 0.5, 1.0), edge_size=0.1)
    cases = [(box, (0.5, 0.5, 0.5), math.sqrt(0.5)), (box, (2.0, 0.5, 0.0), 1.0), (box, (0.5, 0.5, 2.0), math.sqrt(1.25)),
             (box, (1.0, 0.3, 1.0), 0.0), (cylinder, (0.0, 0.0, 0.5), math.sqrt(0.5)),
             (cylinder, (0.5, 0.0, 1.0), 0.0), (cylinder, (1.0, 0.0, -1.0), math.sqrt(1.25))]
    errors['distance'] = max(abs(float(solid.edge_distance(np.array(point))) - expected)
                             for solid, point, expected in cases)
    text = geo_text((0.0, 0.0, 0.0, 2.0, 2.0, 2.0), (0.0, 0.0, 0.0, 2.0, 2.0, 2.0), [box, cylinder], 1.0, 0.01, 2.0,
                    growth=_EDGE_GROWTH)
    n_math = text.count('= MathEval;')
    details['math_eval_fields'] = n_math
    errors['math_eval_fields'] = float(n_math != 14 or f'Field[{n_math + 1}] = Min;' not in text)
    if find_gmsh() is None:
        details['gmsh'] = 'skipped: gmsh not found (set OPEN_SOL_GMSH or put gmsh on PATH)'
    else:
        refined = Solid('pad', 'box', _EDGE_PAD, edge_size=_EDGE_SIZE)
        mesh = mesh_solids_with_gmsh(_EDGE_WINDOW, [refined], _EDGE_H_MAX, growth=_EDGE_GROWTH)
        plain = mesh_solids_with_gmsh(_EDGE_WINDOW, [Solid('pad', 'box', _EDGE_PAD)], _EDGE_H_MAX,
                                      growth=_EDGE_GROWTH)
        near = refined.edge_distance(mesh.points) <= 0.5 * _EDGE_SIZE
        tets = mesh.tets[near[mesh.tets].any(axis=1)]
        corners = mesh.points[tets]
        lengths = [np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)]
        largest = float(np.max(lengths)) / _EDGE_SIZE
        details.update(n_elements=mesh.n_elements, n_elements_plain=plain.n_elements, near_tets=int(len(tets)),
                       largest_edge_over_size=largest)
        errors['edge_ratio'] = max(largest / _EDGE_RATIO - 1.0, 0.0)
        errors['refines'] = float(not mesh.n_elements > plain.n_elements)
    worst = float(max(errors.values()))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _DISTANCE_TOLERANCE,
            'details': {**details, 'errors': errors}}


# ---- solid_fem3d_electrode_edge_size: the setting through meshing (CPU, no run) ----
# 1 x 1 x 0.3 um piezoelectric LN film with a pad box electrode on top, gmsh tetrahedra (h_max 0.1 um): with
# electrode_edge_size 20 nm tetrahedron_sizing names {'pad': 2e-8} and the gmsh estimate counts more elements than at 0;
# off (0), piezoelectric off or the builtin mesher: no edge sizes (builtin: an info names the unused setting).
def _edge_spec(edge_size, piezoelectric=True, mesher='gmsh'):
    settings = {'piezoelectric': piezoelectric, 'electrode_edge_size': edge_size}
    features = [FeatureSpec('ground', 'electrode', ['z_lo'], {'connection': 'voltage', 'voltage': 0.0}),
                FeatureSpec('drive', 'electrode', ['pad'], {'connection': 'voltage', 'voltage': 1.0}),
                FeatureSpec('clamp', 'fixed', ['x_lo'])]
    return ModelSpec(
        title='electrode edge sizes',
        domain=DomainSpec(size_x=_EDGE_WINDOW[3], size_y=_EDGE_WINDOW[4], size_z=_EDGE_WINDOW[5]),
        geometry=[_box('film', _EDGE_WINDOW[:3], (_EDGE_WINDOW[3], _EDGE_WINDOW[4], _EDGE_PAD[2])),
                  _box('pad', _EDGE_PAD[:3], _EDGE_PAD[3:])],
        materials=[MaterialSpec('ln', dispersion=[lithium_niobate_entry((0.0, 0.0, 0.0))], selection=['film'])],
        physics=[PhysicsSpec(_SOLID, 'solid_mechanics_3d', settings=settings, features=features)],
        meshes=[MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=_EDGE_H_MAX, mesher=mesher)])


def _check_edge_setting(_):
    """Edge sizes of the sizing per variant, the estimate growth, the unused-setting info."""
    from ...model.meshing import estimate_mesh
    from ...model.schema import get_registry
    from ...model.validate import validate
    registry = get_registry()
    errors, details = {}, {}
    refined = estimate_mesh(_edge_spec(_EDGE_SIZE), {}, registry)
    plain = estimate_mesh(_edge_spec(0.0), {}, registry)
    details.update(edge_sizes=dict(refined.sizing.edge_sizes), n_elements=refined.n_elements,
                   n_elements_plain=plain.n_elements)
    errors['named'] = float(refined.sizing.edge_sizes != {'pad': _EDGE_SIZE})
    errors['estimate_grows'] = float(not refined.n_elements > plain.n_elements)
    for label, spec in (('off', _edge_spec(0.0)), ('not_piezo', _edge_spec(_EDGE_SIZE, piezoelectric=False))):
        errors[label] = float(bool(estimate_mesh(spec, {}, registry).sizing.edge_sizes))
    builtin = _edge_spec(_EDGE_SIZE, mesher='builtin')
    errors['builtin_info'] = float(not any(issue.level == 'info' and 'electrode_edge_size' in issue.message
                                           for issue in validate(builtin, registry)))
    worst = float(max(errors.values()))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 0.5,
            'details': {**details, 'errors': errors}}


# ---- solid_fem3d_mirror_parity_check: crystal_tensors.mirror_violation and the Roller warning (CPU) ----
# LN in its standard axes (class 3m, mirror plane normal to X1): no entry flips under x -> -x, c14 (and e15, e22, ...)
# flip under y and z; an isotropic set never flips. Validation: rollers on x_lo (none) and y_lo (warning naming c14 or
# another flipping entry) of a slab of that crystal.
def _mirror_spec():
    features = [FeatureSpec('roll_x', 'roller', ['x_lo']), FeatureSpec('roll_y', 'roller', ['y_lo']),
                FeatureSpec('roll_z', 'roller', ['z_lo'])]
    return ModelSpec(
        title='mirror parity',
        domain=DomainSpec(size_x=1.0 * UM, size_y=1.0 * UM, size_z=0.3 * UM),
        materials=[MaterialSpec('ln', dispersion=[lithium_niobate_entry((0.0, 0.0, 0.0))], selection=['all'])],
        physics=[PhysicsSpec(_SOLID, 'solid_mechanics_3d', settings={}, features=features)],
        meshes=[MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=0.25 * UM, mesher='builtin')])


def _check_mirror(_):
    """mirror_violation per axis of standard-axes LN and of an isotropic set; the Roller warnings of _mirror_spec."""
    from ...materials.crystal_tensors import crystal_properties, mirror_violation
    from ...model.schema import get_registry
    from ...model.validate import validate
    errors, details = {}, {}
    crystal = crystal_properties(lithium_niobate_entry((0.0, 0.0, 0.0)))
    violations = {'xyz'[axis]: mirror_violation(crystal, axis) for axis in range(3)}
    details['violations'] = {axis: [float(relative), name] for axis, (relative, name) in violations.items()}
    errors['x_symmetric'] = float(bool(violations['x'][1]))
    errors['y_broken'] = float(not violations['y'][1])
    errors['z_broken'] = float(not violations['z'][1])
    isotropic = {'c11': 2.0, 'c12': 1.0, 'c13': 1.0, 'c22': 2.0, 'c23': 1.0, 'c33': 2.0, 'c44': 0.5, 'c55': 0.5,
                 'c66': 0.5, 'eps_dc_xx': 3.0, 'eps_dc_yy': 3.0, 'eps_dc_zz': 3.0}
    errors['isotropic'] = float(any(mirror_violation(isotropic, axis)[1] for axis in range(3)))
    warned = {issue.path[-1] for issue in validate(_mirror_spec(), get_registry())
              if issue.level == 'warning' and 'mirror plane' in issue.message}
    details['warned_features'] = sorted(warned)
    errors['warnings'] = float(warned != {'roll_y', 'roll_z'})
    worst = float(max(errors.values()))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 0.5,
            'details': {**details, 'errors': errors}}


MESH_SYMMETRY_3D_CASES = (
    SelfTestCase(
        name='fem_gmsh3d_edge_sizes', label='fem gmsh3d: element sizes along solid edges',
        description=('fem.gmsh3d.Solid.edge_distance of a box (12 edges) and a z cylinder (2 rims) at points with '
                     f'closed-form distances within {_DISTANCE_TOLERANCE:g}; geo_text writes one MathEval size field '
                     'per edge, all in the Min background field; with gmsh (else skipped and noted) a 1 x 1 x 0.3 um '
                     'window with a 0.4 x 0.4 x 0.1 um pad box, h_max 0.1 um, edge_size 20 nm, growth 1.3: the '
                     f'tetrahedra touching the pad edges have edges <= {_EDGE_RATIO:g} x edge_size and the mesh has '
                     'more elements than without (first estimates, not yet run).'),
        specs=dict, check=_check_edge_sizes),
    SelfTestCase(
        name='solid_fem3d_electrode_edge_size', label='Solid mechanics 3D: electrode_edge_size through meshing',
        description=('CPU, no run: a piezoelectric LN film with a pad box electrode on a gmsh tetrahedra mesh (h_max '
                     '0.1 um): electrode_edge_size 20 nm gives tetrahedron_sizing edge_sizes {pad: 2e-8} (the '
                     "physics' mesh_edge_sizes hook) and a larger gmsh element estimate; 0 or piezoelectric off give "
                     'none; the builtin mesher gets an info naming the unused setting. error = 1 per failed item, '
                     'tolerance 0.5.'),
        specs=dict, check=_check_edge_setting),
    SelfTestCase(
        name='solid_fem3d_mirror_parity_check', label='Solid mechanics 3D: mirror-plane check of crystal tensors',
        description=('crystal_tensors.mirror_violation: lithium niobate in its standard axes (3m, mirror normal to X1) '
                     'keeps every c / e / eps entry under x -> -x and flips some under y and z; an isotropic set flips '
                     'none. Validation of a slab of that crystal warns on the Roller features on y_lo and z_lo (a '
                     'possible half-model plane that is no symmetry of the crystal), not on x_lo. error = 1 per failed '
                     'item, tolerance 0.5.'),
        specs=dict, check=_check_mirror),
)

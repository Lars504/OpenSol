"""Self-tests of the azimuthal order m >= 1 of the axisymmetric triangle solid (new_modules_plan K7): the order-m operator
identities (CPU, no model: rigid motions, K_sigma of the tilt, mass totals, the rigid-mode count) and the out-of-plane
modes of a thin annular plate, free and clamped at its inner rim, against the Kirchhoff table of the Basiri plan (B-E1
brackets); definition registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result, worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

UM = 1e-6
_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_TABLE = 'modes'

# ---- solid_fem_azimuthal_operator (CPU, no model): a 1 x 0.5 um (r, z) block from r = 0.5 um of 0.125 um structured
# right triangles (P2), isotropic (E 100 GPa, nu 0.3, rho 2000). Order 1: the sideways translation (U_r, U_z, U_phi) =
# (1, 0, -1) and the tilt (z, -r, -z) have zero strain, so K x = 0, and zero (translation) or pi int (sigma_rr +
# sigma_tt + 2 sigma_zz) r dA (tilt: sigma : (W^T W) of the rotation, phi integrated) geometric energy for a uniform
# prescribed stress (the translation's 0 relative to x^T |K_sigma| x: its terms cancel); orders 2 and 3: the same vectors strain (x^T K x > 0). Every K, K_sigma symmetric; the translation
# mass x^T M x = 2 pi rho int r dA; the order-1 element strains of the tilt vanish; ElasticFEMSystem counts 2 free rigid
# motions at order 1 (rigid_motions_allowed), none at order 2, and refuses the unsupported order-1 block otherwise.
_STEP = 0.125 * UM
_R0 = 0.5 * UM
_SIZE = (1.0 * UM, 0.5 * UM)
_ELASTIC = (100e9, 0.3, 2000.0)
_PRESTRESS = (2e8, -1e8, 5e7, 3e7)   # rr, zz, tt, rz [Pa]
_OPERATOR_TOLERANCE = 1e-10


def _check_operator(_):
    """Order-m rigid null space, tilt K_sigma identity, mass total, element strains, rigid-motion counts."""
    from ...fem.dofs import lagrange_dofs
    from ...fem.elasticity import (StressField, assemble_azimuthal_geometric_stiffness, assemble_azimuthal_mass,
                                   assemble_azimuthal_stiffness, azimuthal_element_strains, local_dof_points)
    from ...fem.selftests_harmonic import _rectangle_mesh
    from .selftests_fem import _isotropic_coefficients
    from .system_fem import ElasticFEMSystem
    x_edges = _R0 + np.arange(0.0, _SIZE[0] + 0.5 * _STEP, _STEP)
    y_edges = np.arange(0.0, _SIZE[1] + 0.5 * _STEP, _STEP)
    mesh = _rectangle_mesh(x_edges, y_edges)
    youngs, poisson, density = _ELASTIC
    coefficients = _isotropic_coefficients(mesh.n_elements, youngs, poisson)
    dofs = lagrange_dofs(mesh, 2)
    points = dofs.dof_points()
    radius, height = points[:, 0], points[:, 1]
    translation = np.column_stack([np.ones_like(radius), np.zeros_like(radius), -np.ones_like(radius)]).reshape(-1)
    tilt = np.column_stack([height, -radius, -height]).reshape(-1)
    errors, details = {}, {}
    prestress = np.tile(np.array(_PRESTRESS), (mesh.n_elements, 1))
    r_area = 0.5 * (x_edges[-1] ** 2 - x_edges[0] ** 2) * _SIZE[1]   # int r dA
    for order in (1, 2, 3):
        stiffness = assemble_azimuthal_stiffness(dofs, coefficients, order)
        geometric = assemble_azimuthal_geometric_stiffness(dofs, StressField(prescribed=prestress), order)
        scale = float(abs(stiffness).max())
        errors[f'm{order} stiffness symmetric'] = float(abs(stiffness - stiffness.T).max()) / scale
        errors[f'm{order} geometric symmetric'] = (float(abs(geometric - geometric.T).max())
                                                   / float(abs(geometric).max()))
        for name, vector in (('translation', translation), ('tilt', tilt)):
            energy = float(vector @ (stiffness @ vector)) / (scale * float(vector @ vector))
            if order == 1:
                errors[f'm1 {name} null'] = float(np.abs(stiffness @ vector).max()) / (scale * np.abs(vector).max())
            else:   # a rigid motion only at order 1
                errors[f'm{order} {name} strains'] = float(energy <= 1e-8)
                details[f'm{order} {name} energy'] = energy
        if order == 1:
            sigma_rr, sigma_zz, sigma_tt, _ = _PRESTRESS
            expected_tilt = math.pi * (sigma_rr + sigma_tt + 2.0 * sigma_zz) * r_area
            errors['m1 tilt geometric energy'] = (abs(float(tilt @ (geometric @ tilt)) - expected_tilt)
                                                  / abs(expected_tilt))
            # the unit translation's energy cancels terms far above the small tilt value: relative to their size
            translation_terms = float(np.abs(translation) @ (abs(geometric) @ np.abs(translation)))
            errors['m1 translation geometric energy'] = (abs(float(translation @ (geometric @ translation)))
                                                         / translation_terms)
            details['m1 tilt geometric expected'] = expected_tilt
    mass = assemble_azimuthal_mass(dofs, np.full(mesh.n_elements, density))
    expected_mass = 2.0 * math.pi * density * r_area
    errors['translation mass'] = abs(float(translation @ (mass @ translation)) / expected_mass - 1.0)
    _, strains = azimuthal_element_strains(dofs, tilt.reshape(-1, 3), local_dof_points(2), 1)
    errors['m1 tilt strains'] = float(np.abs(strains).max())   # a unit rotation: strains O(1) if wrong
    solid = np.ones(mesh.n_elements, dtype=bool)
    counts = {}
    for order in (1, 2):
        system = ElasticFEMSystem(mesh, 2, True, lambda: 'azimuthal operator', azimuthal_order=order)
        system.rigid_motions_allowed = True
        system.assemble(coefficients, solid)
        counts[order] = system.n_rigid_motions
    errors['rigid motion counts'] = float(counts != {1: 2, 2: 0})
    refused = ElasticFEMSystem(mesh, 2, True, lambda: 'azimuthal operator', azimuthal_order=1)
    try:
        refused.assemble(coefficients, solid)
        errors['unsupported order 1 refused'] = 1.0
    except ValueError:
        errors['unsupported order 1 refused'] = 0.0
    details['rigid motion counts'] = counts
    return worst_result(errors, _OPERATOR_TOLERANCE, **details)


# ---- solid_fem_azimuthal_annular_plate: the silica annulus of the Basiri microdisk (a = 148 um, b = 82 um, t = 1.8
# um; E 73 GPa, nu 0.17, rho 2200) as an (r, z) cross-section on 0.45 um P2 triangles (4 across t), the window from the
# axis (void below r = b): the lowest out-of-plane mode of each order m = 1 .. 4, free-free (order 1: target > 0 drops
# the sideways translation and the tilt) and clamped at r = b (a Fixed (domain) rim r in [b - 1 um, b] of the same
# material), against the Kirchhoff plate table of the plan (section 6, 1.5 % as its E1 tolerance, t / (a - b) = 0.027).
# Each run targets 0.9 x the expected frequency and keeps the nearest mode.
_OUTER = 148.0 * UM
_INNER = 82.0 * UM
_THICKNESS = 1.8 * UM
_RIM = 1.0 * UM
_SILICA = {'youngs_modulus': 73e9, 'poisson_ratio': 0.17, 'rho': 2200.0}
_PLATE_H = 0.45 * UM
_KIRCHHOFF = {'free': {1: 412.1e3, 2: 94.2e3, 3: 254.5e3, 4: 473.6e3},     # Hz, basiri plan section 6
              'clamped': {1: 367.7e3, 2: 411.0e3, 3: 504.1e3, 4: 659.6e3}}
_PLATE_TOLERANCE = 1.5e-2


def _plate_spec(support, order):
    geometry = [GeometrySpec('annulus', 'rectangle', {'x0': _INNER, 'y0': 0.0, 'x1': _OUTER, 'y1': _THICKNESS})]
    selection = ['annulus']
    features = [FeatureSpec(_TABLE, 'resonance_table')]
    if support == 'clamped':
        geometry.append(GeometrySpec('rim', 'rectangle', {'x0': _INNER - _RIM, 'y0': 0.0, 'x1': _INNER,
                                                          'y1': _THICKNESS}))
        selection.append('rim')
        features.append(FeatureSpec('clamp', 'fixed_domain', ['rim']))
    settings = {'geometry': 'axisymmetric', 'azimuthal_order': order, 'discretization': 'quadratic',
                'target_frequency': 0.9 * _KIRCHHOFF[support][order]}
    return ModelSpec(
        title=f'Self-test: {support} annular plate, azimuthal order {order}',
        domain=DomainSpec(size_x=_OUTER, size_y=_THICKNESS), geometry=geometry,
        materials=[MaterialSpec('silica', properties=dict(_SILICA), selection=selection)],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings=settings, features=features)],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_PLATE_H)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}))


def _plate_specs():
    return {f'{support}_m{order}': _plate_spec(support, order) for support in _KIRCHHOFF for order in range(1, 5)}


def _check_plate(results_by_name):
    """The kept mode of every run against the Kirchhoff table."""
    errors, tolerances, details = {}, {}, {}
    for support, table in _KIRCHHOFF.items():
        for order, expected in table.items():
            run = f'{support}_m{order}'
            result = results_by_name[run].result(_SOLID, _TABLE)
            columns = result['columns']
            frequencies = [float(row[columns.index('frequency')]) for row in result['rows']]
            measured = frequencies[0] if frequencies else math.nan
            errors[run], tolerances[run] = abs(measured / expected - 1.0), _PLATE_TOLERANCE
            details[run] = {'measured': measured, 'kirchhoff': expected}
    return scaled_worst_result(errors, tolerances, **details)


AZIMUTHAL_CASES = (
    SelfTestCase(
        name='solid_fem_azimuthal_operator',
        label='Solid mechanics on triangles: azimuthal order m operator identities',
        description=('fem.elasticity order-m form directly (no model run, P2 on a structured (r, z) block from r = 0.5 '
                     'um): at m = 1 the sideways translation (1, 0, -1) and the tilt (z, -r, -z) are in the null space '
                     'of K, the tilt geometric energy is pi int (sigma_rr + sigma_tt + 2 sigma_zz) r dA and the '
                     'translation one 0 for a uniform prescribed stress, the tilt strains vanish; at m = 2, 3 both '
                     'strain; K and K_sigma symmetric, the translation mass 2 pi rho int r dA; ElasticFEMSystem counts 2 '
                     'free rigid motions at m = 1, none at m = 2, and refuses an unsupported m = 1 block; within '
                     f'{_OPERATOR_TOLERANCE:g} (first estimate, not yet run).'),
        specs=dict, check=_check_operator),
    SelfTestCase(
        name='solid_fem_azimuthal_annular_plate',
        label='Solid mechanics on triangles: annular plate modes of azimuthal order 1 .. 4 (free and clamped)',
        description=('The silica annulus of the Basiri microdisk (a 148 um, b 82 um, t 1.8 um) as an axisymmetric (r, z) '
                     'section on 0.45 um P2 triangles: the lowest out-of-plane mode of each azimuthal_order 1 .. 4, '
                     'free-free (order 1 drops its 2 rigid motions) and clamped at r = b by a Fixed (domain) rim, '
                     f'against the Kirchhoff plate table of the plan within {_PLATE_TOLERANCE:g} (8 runs; first '
                     'estimate, not yet run). measured = error in tolerance units, tolerance 1.'),
        specs=_plate_specs, check=_check_plate),
)

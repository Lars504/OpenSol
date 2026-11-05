"""Bloch / Floquet unit cells of solid_mechanics_3d on tetrahedra (piezo_3d_plan stage 5): the reduced pencil
identities on a two-material slab cell (CPU, no model), bulk bands of all-periodic cells against the Christoffel
equation (isotropic, rotated LN, a k = 0 cell with its rigid modes dropped), the piezoelectrically stiffened LN bands,
gmsh periodic surfaces (skipped without gmsh) and the 'solid_bloch_band_diagram_3d' template. definition3d registers
them."""
import math

import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, lithium_niobate_entry, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                           PhysicsSpec, StudySpec, SweepSpec)
from .selftests import _elastic, _lame

# definition3d imports this module for its template: the sibling 3D selftests modules load inside the checks only
UM = 1e-6
_SOLID = 'solid'
_TABLE = 'bands'
_EULER = [31.0, 47.0, 73.0]   # deg: lithium niobate with every c, e and eps entry non-zero
_CHRISTOFFEL_X = [0, 5, 4]    # Voigt columns (xx, xy, xz) of the displacement components along a wave on x


# ---- solid_fem3d_bloch_operator: the reduced pencil of a two-material slab cell (CPU, no model) ----
# Cell 0.5 x 0.25 x 0.25 um of 0.125 um Kuhn cubes (P2), periodic along x: isotropic material A (E 100 GPa, nu 0.3, rho
# 2000) on the first half of the period, the rotated LN of _EULER on the second; Fixed on y_lo, rollers on z_lo / z_hi.
# Items: P^H K P Hermitian to round-off at k = 0.3 pi / a; the bands at k and k + 2 pi / a equal; zone folding: the
# cell's bands at k = 0 and k = pi / a (real phase -1) together are the two-period supercell's bands at k = 0; the cell
# shifted by a / 4 (one cube, the slabs wrapped) has the same bands. Rigid check on rollers y / z only: at k = 0 the
# slide along x is refused (allowed: counted, 1 motion), at k != 0 none is left.
_CELL_PERIOD = 0.5 * UM
_CELL_STEP = 0.125 * UM
_CELL_SIDE = 0.25 * UM
_CELL_MODES = 6
_CELL_K = 0.3 * math.pi / _CELL_PERIOD
_CELL_A = (100e9, 0.3, 2000.0)
_HERMITIAN_TOLERANCE = 1e-12   # relative to the largest entry (round-off of P^H K P; first estimate)
_BAND_TOLERANCE = 1e-8         # relative frequency between equal discrete problems (ARPACK at machine tolerance)
_CELL_HOLDS = [('y_lo', {'u_x': 0.0, 'u_y': 0.0, 'u_z': 0.0}), ('z_lo', {'u_z': 0.0}), ('z_hi', {'u_z': 0.0})]
_ROLLER_HOLDS = [('y_lo', {'u_y': 0.0}), ('y_hi', {'u_y': 0.0}), ('z_lo', {'u_z': 0.0}), ('z_hi', {'u_z': 0.0})]


def _cell_system(periods, shift_cubes, holds, wavenumber, rigid_allowed=False):
    """(assembled ElasticFEMSystem3D with its BlochCell3D, consistent mass CSR) of `periods` repeats of the slab cell
    (material A where the centroid lies in the first half of a period after a shift by shift_cubes cubes)."""
    from ...fem.elasticity3d import isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from ...materials.crystal_tensors import full_crystal
    from .bloch3d import bloch_cell_3d
    from .system_fem3d import ElasticFEMSystem3D
    edges_x = np.linspace(0.0, periods * _CELL_PERIOD, int(round(periods * _CELL_PERIOD / _CELL_STEP)) + 1)
    edges_side = np.linspace(0.0, _CELL_SIDE, int(round(_CELL_SIDE / _CELL_STEP)) + 1)
    mesh = structured_box_mesh(edges_x, edges_side, edges_side)
    system = ElasticFEMSystem3D(mesh, 2, lambda: 'Bloch operator self-test')
    system.bloch = bloch_cell_3d(system.full, (True, False, False), (wavenumber, 0.0, 0.0))
    system.rigid_motions_allowed = rigid_allowed
    for face, components in holds:
        system.hold(system.full.face_dofs(system.face_faces(face)), **components)
    count = mesh.n_elements
    in_a = np.mod(mesh.centroids()[:, 0] - shift_cubes * _CELL_STEP, _CELL_PERIOD) < 0.5 * _CELL_PERIOD
    crystal = full_crystal(lithium_niobate_entry(_EULER))
    lame_lambda, mu = _lame(*_CELL_A[:2])
    isotropic = isotropic_matrix_3d(np.full(count, lame_lambda), np.full(count, mu))
    stiffness = np.where(in_a[:, None, None], isotropic, crystal.stiffness[None, :, :])
    density = np.where(in_a, _CELL_A[2], float(crystal.properties['rho']))
    system.assemble(stiffness, np.ones(count, dtype=bool))
    return system, system.mass(density)


def _cell_prolongation(system):
    from .bloch3d import bloch_prolongation
    return bloch_prolongation(*system.bloch.vector_masters(), system.free)[1]


def _cell_frequencies(system, mass, count=_CELL_MODES):
    """Sorted lowest `count` band frequencies [Hz] of an assembled cell (modal.solve_modes on its prolongation)."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .modal import solve_modes
    solve = solve_modes(system.operator, mass, None, 0.0, count, 3, ArpackShiftInvertSolver(),
                        prolongation=_cell_prolongation(system))
    return np.sort([resonance.frequency for resonance in solve.resonances])


def _band_error(measured, expected):
    if len(measured) != len(expected):
        return math.inf
    return float(np.max(np.abs(np.asarray(measured) / np.asarray(expected) - 1.0)))


def _check_operator(_):
    """Hermitian reduced stiffness, the 2 pi / a shift, zone folding, a shifted cell and the periodic rigid check."""
    errors, tolerances, details = {}, {}, {}
    system, mass = _cell_system(1, 0, _CELL_HOLDS, _CELL_K)
    prolongation = _cell_prolongation(system)
    reduced = (prolongation.conj().T @ system.operator @ prolongation).tocsr()
    peak = float(np.max(np.abs(reduced.data)))
    errors['hermitian'] = float(np.max(np.abs((reduced - reduced.conj().T).data), initial=0.0)) / peak
    tolerances['hermitian'] = _HERMITIAN_TOLERANCE
    details['complex_pencil'] = bool(np.iscomplexobj(reduced.data))
    bands = _cell_frequencies(system, mass)
    shifted_k = _cell_frequencies(*_cell_system(1, 0, _CELL_HOLDS, _CELL_K + 2.0 * math.pi / _CELL_PERIOD))
    errors['k_plus_2pi_over_a'] = _band_error(shifted_k, bands)
    shifted_cell = _cell_frequencies(*_cell_system(1, 1, _CELL_HOLDS, _CELL_K))
    errors['shifted_cell'] = _band_error(shifted_cell, bands)
    gamma = _cell_frequencies(*_cell_system(1, 0, _CELL_HOLDS, 0.0))
    edge = _cell_frequencies(*_cell_system(1, 0, _CELL_HOLDS, math.pi / _CELL_PERIOD))
    folded = np.sort(np.concatenate([gamma, edge]))[:_CELL_MODES]
    supercell = _cell_frequencies(*_cell_system(2, 0, _CELL_HOLDS, 0.0))
    errors['zone_folding'] = _band_error(supercell, folded)
    for item in ('k_plus_2pi_over_a', 'shifted_cell', 'zone_folding'):
        tolerances[item] = _BAND_TOLERANCE
    details.update({'bands_Hz': bands.tolist(), 'gamma_Hz': gamma.tolist(), 'zone_edge_Hz': edge.tolist(),
                    'supercell_Hz': supercell.tolist()})
    rigid = {}
    try:   # rollers on y / z only, k = 0: the slide along x stays free
        _cell_system(1, 0, _ROLLER_HOLDS, 0.0)
        rigid['refused'] = False
    except ValueError as exc:
        rigid['refused'] = 'slide along x' in str(exc)
        rigid['message'] = str(exc)
    rigid['counted'] = _cell_system(1, 0, _ROLLER_HOLDS, 0.0, rigid_allowed=True)[0].n_rigid_motions
    try:
        rigid['bloch_k_motions'] = _cell_system(1, 0, _ROLLER_HOLDS, _CELL_K)[0].n_rigid_motions
    except ValueError as exc:
        rigid['bloch_k_motions'] = str(exc)
    passed = rigid['refused'] and rigid['counted'] == 1 and rigid['bloch_k_motions'] == 0
    errors['rigid_check'], tolerances['rigid_check'] = (0.0 if passed else math.inf), 1.0
    details['rigid'] = rigid
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_bloch_bulk_bands: all-periodic cells, a plane wave along x (isotropic, crystal, gamma, piezo) ----
# A 1 x 0.25 x 0.25 um cell of 62.5 nm Kuhn cubes (P2), every axis periodic, bloch_kx = 0.4 pi / a: the three lowest
# bands are the bulk plane waves exp(-j k x) (the folded ones lie above 1.6 pi / a), rho omega^2 = the eigenvalues of
# the Christoffel matrix Gamma_ik = c_i1k1 (Voigt columns xx, xy, xz) times k^2 (isotropic: c_T k twice, c_L k);
# piezoelectric (no electrode: k != 0 excludes a constant phi): D_x = 0 stiffens it, Gamma + gamma gamma^T / (eps0
# eps_xx) with gamma_i = e_1J(i) (piezo_e11, _e16, _e15). Run gamma (isotropic, k = 0, target half the first folded
# band): the 4-fold transverse band c_T 2 pi / a, the three rigid translations dropped. P2 with k h <= 0.4 (first
# estimates).
_BULK_SIZE = (1.0 * UM, 0.25 * UM, 0.25 * UM)
_BULK_CELL = 0.0625 * UM
_BULK_K = 0.4 * math.pi / _BULK_SIZE[0]
_BULK_ISOTROPIC = (100e9, 0.25, 2000.0)
_BULK_MODES = 3
_GAMMA_MODES = 4
_BULK_TOLERANCE = 1e-4      # relative frequency, k h = 0.08
_GAMMA_TOLERANCE = 1e-3     # relative frequency, k h = 0.39
_PERIODIC_ALL = {f'periodic_{axis}': True for axis in 'xyz'}
_BULK_RUNS = ('isotropic', 'crystal', 'gamma', 'piezo')   # piezo: the crystal cell with piezoelectric on


def _isotropic_properties():
    youngs_modulus, poisson_ratio, density = _BULK_ISOTROPIC
    return _elastic(youngs_modulus, poisson_ratio, rho=density)


def _bulk_spec(run):
    from .selftests_fem3d import _solid_spec
    piezoelectric = run == 'piezo'
    material = (MaterialSpec('medium', properties=_isotropic_properties(), selection=['all'])
                if run in ('isotropic', 'gamma')
                else MaterialSpec('crystal', dispersion=[lithium_niobate_entry(_EULER)], selection=['all']))
    settings = {**_PERIODIC_ALL, 'bloch_kx': 0.0 if run == 'gamma' else _BULK_K, 'piezoelectric': piezoelectric}
    count = _BULK_MODES
    if run == 'gamma':
        settings['target_frequency'] = 0.5 * _gamma_reference()
        count = _GAMMA_MODES
    return _solid_spec(f'Self-test: Bloch bulk bands on tetrahedra ({run})', _BULK_SIZE, [material],
                       [FeatureSpec(_TABLE, 'resonance_table')], cell=_BULK_CELL,
                       study=StudySpec(type='eigenfrequency', settings={'n_resonances': count}), settings=settings)


def _float32_constants(properties, names):
    return {name: float32_value(properties.get(name, 0.0)) for name in names}


def _isotropic_voigt():
    """(6 x 6 Voigt c [Pa], rho) of _BULK_ISOTROPIC as the float32 MaterialGrid stores it."""
    youngs_modulus, poisson_ratio, density = (float32_value(value) for value in _BULK_ISOTROPIC)
    lame_lambda, mu = _lame(youngs_modulus, poisson_ratio)
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = lame_lambda
    stiffness[np.arange(3), np.arange(3)] += 2.0 * mu
    stiffness[np.arange(3, 6), np.arange(3, 6)] = mu
    return stiffness, density


def _crystal_voigt():
    """(6 x 6 Voigt c, 3 x 6 e, eps_xx (relative), rho) of the rotated LN as the float32 MaterialGrid stores them."""
    from ...fem.elasticity3d import STIFFNESS_NAMES
    from ...materials.crystal_tensors import full_crystal
    properties = full_crystal(lithium_niobate_entry(_EULER)).properties
    stiffness = np.zeros((6, 6))
    for name, value in _float32_constants(properties, STIFFNESS_NAMES).items():
        row, column = int(name[1]) - 1, int(name[2]) - 1
        stiffness[row, column] = stiffness[column, row] = value
    piezo = np.zeros((3, 6))
    for field in range(3):
        for voigt in range(6):
            piezo[field, voigt] = float32_value(properties.get(f'piezo_e{field + 1}{voigt + 1}', 0.0))
    return stiffness, piezo, float32_value(properties['eps_dc_xx']), float32_value(properties['rho'])


def _christoffel_frequencies(stiffness, density, wavenumber, piezo=None, permittivity_xx=None):
    """Sorted plane-wave frequencies [Hz] along x of a Voigt c (piezo: stiffened by gamma gamma^T / (eps0 eps_xx))."""
    christoffel = stiffness[np.ix_(_CHRISTOFFEL_X, _CHRISTOFFEL_X)].copy()
    if piezo is not None:
        coupling = piezo[0, _CHRISTOFFEL_X]
        christoffel += np.outer(coupling, coupling) / (EPS0 * permittivity_xx)
    velocities = np.sqrt(np.linalg.eigvalsh(christoffel) / density)
    return np.sort(velocities * wavenumber / (2.0 * math.pi))


def _gamma_reference():
    """[Hz] of the first folded transverse band at k = 0: c_T 2 pi / a."""
    stiffness, density = _isotropic_voigt()
    return math.sqrt(stiffness[3, 3] / density) / _BULK_SIZE[0]


def _bulk_expected(run):
    """Expected band frequencies [Hz] of a bulk run (piezo: stiffened by gamma gamma^T / (eps0 eps_xx))."""
    if run == 'gamma':
        return [_gamma_reference()] * _GAMMA_MODES
    if run == 'isotropic':
        return _christoffel_frequencies(*_isotropic_voigt(), _BULK_K).tolist()
    stiffness, piezo, permittivity_xx, density = _crystal_voigt()
    if run == 'piezo':
        return _christoffel_frequencies(stiffness, density, _BULK_K, piezo, permittivity_xx).tolist()
    return _christoffel_frequencies(stiffness, density, _BULK_K).tolist()


def _table_frequencies(results):
    return sorted(float(row[1]) for row in results.result(_SOLID, _TABLE)['rows'])


def _check_bulk(results_by_name):
    """Table frequencies of each run against its Christoffel plane waves (gamma: the 4-fold folded band; piezo: the
    stiffened waves)."""
    errors, tolerances, details = {}, {}, {}
    for run, results in results_by_name.items():
        expected = _bulk_expected(run)
        measured = _table_frequencies(results)
        errors[run] = _band_error(measured, expected)
        tolerances[run] = _GAMMA_TOLERANCE if run == 'gamma' else _BULK_TOLERANCE
        details[run] = {'measured_Hz': measured, 'expected_Hz': expected}
    return scaled_worst_result(errors, tolerances, **details)


# ---- fem_gmsh3d_periodic: gmsh periodic surfaces of a holed, ribbed cell (skipped without gmsh) ----
# 0.5 x 0.5 x 0.2 um cell, periodic along x and y, with a z cylinder (r 0.15 um) in the middle and a box rib across the
# whole x period (its faces split both x sides alike): every hi-side node must have its lo-side partner at the
# translated position (periodic_node_partners) and the P2 Bloch masters must build (the Lagrange edge partners need no
# node order). error = the largest partner mismatch over the side length, tolerance 1e-9.
_GMSH_SIDE = 0.5 * UM
_GMSH_THICKNESS = 0.2 * UM
_GMSH_H_MAX = 0.08 * UM
_GMSH_TOLERANCE = 1e-9


def _check_gmsh_periodic(_):
    """Node partners along x and y of the gmsh cell, their position mismatch and the P2 Bloch masters."""
    from ...fem.dofs3d import bloch_masters, lagrange_dofs_3d
    from ...fem.gmsh3d import Solid, mesh_solids_with_gmsh
    from ...fem.gmsh_io import find_gmsh
    from ...fem.mesh3d import periodic_node_partners
    if find_gmsh() is None:
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _GMSH_TOLERANCE,
                'details': {'skipped': 'gmsh not found (set OPEN_SOL_GMSH or put gmsh on PATH)'}}
    solids = [Solid('hole', 'cylinder', (0.5 * _GMSH_SIDE, 0.5 * _GMSH_SIDE, 0.0, 0.15 * UM, _GMSH_THICKNESS)),
              Solid('rib', 'box', (0.0, 0.03 * UM, 0.0, _GMSH_SIDE, 0.08 * UM, _GMSH_THICKNESS))]
    mesh = mesh_solids_with_gmsh((0.0, 0.0, 0.0, _GMSH_SIDE, _GMSH_SIDE, _GMSH_THICKNESS), solids, _GMSH_H_MAX,
                                 periodic_axes=('x', 'y'))
    errors, details = {}, {'n_points': mesh.n_points, 'n_elements': mesh.n_elements}
    for axis in (0, 1):
        name = 'xyz'[axis]
        try:
            partners = periodic_node_partners(mesh, axis)
        except ValueError as exc:
            errors[name], details[name] = math.inf, str(exc)
            continue
        hi_nodes = np.flatnonzero(partners >= 0)
        shift = np.zeros(3)
        shift[axis] = _GMSH_SIDE
        mismatch = mesh.points[hi_nodes] - shift - mesh.points[partners[hi_nodes]]
        errors[name] = float(np.max(np.abs(mismatch))) / _GMSH_SIDE
        details[name] = {'pairs': int(hi_nodes.size)}
    try:
        master, _ = bloch_masters(lagrange_dofs_3d(mesh, 2), {0: math.pi / _GMSH_SIDE, 1: 0.0})
        details['p2_masters_moved'] = int(np.count_nonzero(master != np.arange(master.size)))
        errors['p2_masters'] = 0.0
    except ValueError as exc:
        errors['p2_masters'], details['p2_masters'] = math.inf, str(exc)
    worst = float(np.max(list(errors.values())))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _GMSH_TOLERANCE,
            'details': {**details, 'errors': errors}}


# ---- template 'solid_bloch_band_diagram_3d': a corrugated silicon nanobeam cell, bands over k_x ----
BAND_DIAGRAM_3D_LABEL = 'Phononic nanobeam band diagram (3D)'
BAND_DIAGRAM_3D_DESCRIPTION = (
    'Solid mechanics 3D on builtin tetrahedra: one unit cell (a = 0.5 um) of a free-standing silicon nanobeam (0.25 x '
    '0.25 um core) with a 0.75 um wide wing over the middle half of the period, periodic along x with bloch_kx = '
    'k_fraction 2 pi / a swept over 0-0.5 (Gamma-X); an eigenfrequency study per k lists the 8 bands nearest 4 GHz '
    '(resonance_table; target > 0 drops the rigid modes at Gamma). The bands per k form the band diagram; gaps between '
    'them are the phononic band gaps of the corrugation.')
_TEMPLATE_PERIOD = 0.5e-6
_TEMPLATE_WIDTH = 0.75e-6        # wing width (domain y)
_TEMPLATE_CORE = 0.25e-6         # core width and the film thickness
_TEMPLATE_K_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
_TEMPLATE_TARGET = 4e9           # Hz
_TEMPLATE_BANDS = 8


def solid_bloch_band_diagram_3d_spec():
    """Template 'solid_bloch_band_diagram_3d': corrugated nanobeam cell, k sweep along x, eigenfrequency study."""
    core_low = 0.5 * (_TEMPLATE_WIDTH - _TEMPLATE_CORE)
    geometry = [GeometrySpec('core', 'box', {'x0': 0.0, 'y0': core_low, 'z0': 0.0, 'x1': _TEMPLATE_PERIOD,
                                             'y1': core_low + _TEMPLATE_CORE, 'z1': _TEMPLATE_CORE}),
                GeometrySpec('wing', 'box', {'x0': 0.25 * _TEMPLATE_PERIOD, 'y0': 0.0, 'z0': 0.0,
                                             'x1': 0.75 * _TEMPLATE_PERIOD, 'y1': _TEMPLATE_WIDTH,
                                             'z1': _TEMPLATE_CORE})]
    settings = {'discretization': 'quadratic', 'periodic_x': True, 'bloch_kx': 'k_fraction * 2 * pi / a',
                'target_frequency': _TEMPLATE_TARGET}
    return ModelSpec(
        title=BAND_DIAGRAM_3D_LABEL,
        parameters=[ParameterSpec('a', _TEMPLATE_PERIOD, 'lattice constant [m]'),
                    ParameterSpec('k_fraction', _TEMPLATE_K_FRACTIONS[-1],
                                  'Bloch kx in units of 2 pi / a (Gamma-X: 0-0.5)')],
        domain=DomainSpec(size_x=_TEMPLATE_PERIOD, size_y=_TEMPLATE_WIDTH, size_z=_TEMPLATE_CORE),
        geometry=geometry,
        materials=[MaterialSpec('silicon', properties=_elastic(169e9, 0.28, rho=2329.0), selection=['core', 'wing'])],
        physics=[PhysicsSpec(_SOLID, 'solid_mechanics_3d', settings=settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=62.5e-9, mesher='builtin')],
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _TEMPLATE_BANDS},
                        sweep=SweepSpec('k_fraction', list(_TEMPLATE_K_FRACTIONS))))


BLOCH_FEM_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_bloch_operator', label='Solid mechanics 3D on tetrahedra: Bloch cell operator identities',
        description=('bloch3d and modal.solve_modes directly (no model run, P2): a 0.5 x 0.25 x 0.25 um slab cell of '
                     '0.125 um Kuhn cubes, isotropic material (E 100 GPa, nu 0.3) on the first half of the x period and '
                     f'lithium niobate at euler_deg {_EULER} on the second, periodic along x, Fixed y_lo, rollers z_lo / '
                     f'z_hi: P^H K P Hermitian (relative {_HERMITIAN_TOLERANCE:g}); the {_CELL_MODES} lowest bands at k '
                     'and k + 2 pi / a, the cell shifted by a / 4 and the two-period supercell at k = 0 against the '
                     'cell at k = 0 and pi / a (zone folding) within '
                     f'{_BAND_TOLERANCE:g}; the periodic rigid-mode check (rollers on y / z only: the slide along x '
                     'refused at k = 0, counted as 1 motion when allowed, none at k != 0). measured = error = the worst '
                     'deviation in tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=dict, check=_check_operator),
    SelfTestCase(
        name='solid_fem3d_bloch_bulk_bands', label='Solid mechanics 3D on tetrahedra: Bloch bulk bands (Christoffel)',
        description=('eigenfrequency study of solid_mechanics_3d on quadratic builtin tetrahedra (62.5 nm Kuhn cubes) of '
                     'a 1 x 0.25 x 0.25 um cell, periodic along x, y and z with bloch_kx = 0.4 pi / a: the 3 lowest '
                     'bands are the plane waves along x, rho omega^2 = eig(Gamma) k^2 with the Christoffel matrix '
                     'Gamma_ik = c_i1k1. Runs isotropic (E 100 GPa, nu 0.25, rho 2000: c_T twice and c_L), crystal '
                     f'(lithium niobate at euler_deg {_EULER}: three quasi-waves) within {_BULK_TOLERANCE:g}, and gamma '
                     '(isotropic, k = 0, target half the first folded band: its 4 modes at c_T 2 pi / a, the 3 rigid '
                     f'translations dropped) within {_GAMMA_TOLERANCE:g}, and piezo (the crystal run with piezoelectric '
                     'on, no electrode: k != 0 excludes a constant potential; D_x = 0 stiffens the Christoffel matrix by '
                     'gamma gamma^T / (eps0 eps_xx), gamma = (e11, e16, e15) of the rotated crystal) within '
                     f'{_BULK_TOLERANCE:g}. Pins the periodic identification over three axes (edges and corners), the '
                     'complex Hermitian eigen path, the rigid-mode drop and the coupled Bloch pencil (phi with the u '
                     'phase, massless phi rows; the former solid_fem3d_bloch_piezo). measured = error = the worst '
                     'deviation in tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {run: _bulk_spec(run) for run in _BULK_RUNS}, check=_check_bulk),
    SelfTestCase(
        name='fem_gmsh3d_periodic', label='fem gmsh3d: periodic surfaces of a holed cell',
        description=('fem.gmsh3d.mesh_solids_with_gmsh with periodic_axes x and y (needs gmsh, else skipped and noted) '
                     'on a 0.5 x 0.5 x 0.2 um cell with a z cylinder in the middle and a box rib across the x period: '
                     'every hi-side node on x_hi / y_hi has its lo-side partner (fem.mesh3d.periodic_node_partners) at '
                     f'the translated position within {_GMSH_TOLERANCE:g} of the side, and the P2 Bloch masters build '
                     '(Lagrange edge partners without the node-order rule). Pins the .geo Periodic Surface rows.'),
        specs=dict, check=_check_gmsh_periodic),
)

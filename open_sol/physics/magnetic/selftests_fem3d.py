"""Self-tests of magnetostatics_3d on tetrahedron meshes (MagneticFEMPhysics3D; builtin Kuhn meshes on the cpu backend,
no gmsh unless noted): the coil edge drive (CPU check), an empty box in a uniform field, layered and saturating slabs
against the continuous 1D solutions (exact for P1 / P2 on aligned elements), the coil energy identity, a permeable
sphere on gmsh tetrahedra (heavy, skipped without gmsh) and cpu vs gpu. Specs built fresh; registered by
definition3d."""
import numpy as np

from ...core.constants import MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import last_value, scaled_worst_result, worst_error
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, SolverDefaultsSpec, StudySpec)
from .selftests import _NONLINEAR_SETTINGS as _SATURATION_SETTINGS, _nonlinear_iterations
from .selftests_3d import (_HELD_POTENTIAL, _SATURATING_CURVES, _SATURATING_MATERIALS, _SATURATING_ORIENTATIONS,
                           _SATURATING_POTENTIAL, _SLAB_BOX, _SLAB_CELL, _SLAB_PERMEABILITY, _SLAB_RANGE, _UNIFORM_BOX,
                           _UNIFORM_CELL, _UNIFORM_FIELDS, MAGNETOSTATICS_3D_TYPE_NAME, _box, _inverse_curve, _report,
                           _slab_axes, _slab_errors, _slab_parts, _volume)

_MAGNETIC = 'magnetic'
_TOLERANCE = 1e-6   # exact FEM solutions, float32 sample storage (first estimate)
_AXES = 'xyz'
_ORDERS = ('quadratic', 'linear')


# ---- shared helpers ----
def _tetrahedra_mesh(cell):
    return MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=cell, mesher='builtin')


def _fem_spec(title, size, features, materials=(), geometry=(), cell=_SLAB_CELL, discretization='quadratic',
              settings=None, backend='cpu', mesh=None):
    """Stationary magnetostatics_3d-only model on a builtin tetrahedra mesh (Automatic linear solver: host SuperLU)."""
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size[0], size_y=size[1], size_z=size[2]),
        geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_3D_TYPE_NAME,
                             settings={'discretization': discretization, **(settings or {})},
                             features=list(features))],
        meshes=[mesh if mesh is not None else _tetrahedra_mesh(cell)],
        solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type='stationary'))


# ---- magnet3d_fem_coil_drive (CPU) ----
_DRIVE_CELL = 1e-4                                      # m, Kuhn cubes
_DRIVE_CELLS = (10, 10, 12)
_DRIVE_OUTER = ((2, 2, 3), (8, 8, 9))                   # cells: square winding around z (outer minus inner box)
_DRIVE_INNER = ((4, 4, 3), (6, 6, 9))
_DRIVE_TURNS = 50
_DRIVE_TOLERANCE = 1e-10                                # float64 sums of edge integrals


def _line_circulation(mesh, per_ampere, x, y):
    """Sum of the edge integrals along the node line (x, y) parallel to z, oriented toward +z."""
    tolerance = 1e-6 * _DRIVE_CELL
    on_line = (np.abs(mesh.points[:, 0] - x) < tolerance) & (np.abs(mesh.points[:, 1] - y) < tolerance)
    edges = np.flatnonzero(on_line[mesh.edges[:, 0]] & on_line[mesh.edges[:, 1]])
    rise = mesh.points[mesh.edges[edges, 1], 2] - mesh.points[mesh.edges[edges, 0], 2]
    return float(np.sum(np.sign(rise) * per_ampere[edges]))


def _check_coil_drive(results_by_name):
    """Edge drive of a square winding on a Kuhn mesh: axial length, circulation along the axis = turns, 0 outside the
    winding, the ramp s in [0, 1] on the z edges within the winding's axial extent; error = largest error over its
    tolerance (failed sub-checks)."""
    from ...fem.mesh3d import structured_box_mesh
    from ...fem.sampling3d import PointLocator3D
    from .coils3d import coil_edge_drive_per_ampere
    mesh = structured_box_mesh(*(np.arange(count + 1) * _DRIVE_CELL for count in _DRIVE_CELLS))
    centroids = mesh.centroids()

    def inside(box):
        low, high = (np.asarray(corner, dtype=np.float64) * _DRIVE_CELL for corner in box)
        return np.all((centroids > low) & (centroids < high), axis=1)

    winding = inside(_DRIVE_OUTER) & ~inside(_DRIVE_INNER)
    per_ampere, length = coil_edge_drive_per_ampere(mesh, winding, 'z', _DRIVE_TURNS, 'coil',
                                                    PointLocator3D(mesh).locate)
    expected_length = (_DRIVE_OUTER[1][2] - _DRIVE_OUTER[0][2]) * _DRIVE_CELL
    centre = 0.5 * (_DRIVE_INNER[0][0] + _DRIVE_INNER[1][0]) * _DRIVE_CELL
    axis_circulation = _line_circulation(mesh, per_ampere, centre, centre)
    outside_circulation = _line_circulation(mesh, per_ampere, _DRIVE_CELL, _DRIVE_CELL)
    start, end = mesh.points[mesh.edges[:, 0]], mesh.points[mesh.edges[:, 1]]
    axial = (np.all(np.abs(end[:, :2] - start[:, :2]) < 1e-6 * _DRIVE_CELL, axis=1)
             & (np.minimum(start[:, 2], end[:, 2]) >= _DRIVE_OUTER[0][2] * _DRIVE_CELL - 1e-6 * _DRIVE_CELL)
             & (np.maximum(start[:, 2], end[:, 2]) <= _DRIVE_OUTER[1][2] * _DRIVE_CELL + 1e-6 * _DRIVE_CELL))
    ramp = per_ampere[axial] / ((_DRIVE_TURNS / length) * (end[axial, 2] - start[axial, 2]))
    errors = {'axial_length': abs(length - expected_length) / expected_length,
              'axis_circulation': abs(axis_circulation - _DRIVE_TURNS) / _DRIVE_TURNS,
              'outside_circulation': abs(outside_circulation) / _DRIVE_TURNS,
              'ramp_bounds': max(0.0, float(ramp.max()) - 1.0, -float(ramp.min()))}
    return scaled_worst_result(errors, {name: _DRIVE_TOLERANCE for name in errors}, axis_circulation=axis_circulation,
                               axial_length=length, n_axial_edges=int(axial.sum()))


# ---- magnet3d_fem_uniform_field ----
_UNIFORM_RUNS = {f'{order}_{name}': (order, name) for order in _ORDERS for name in ('z', 'oblique')}


def _uniform_spec(order, name, backend='cpu'):
    """Empty box in the named applied field on Kuhn tetrahedra, a B_abs probe at the box centre."""
    centre = dict(zip(_AXES, (0.5 * size for size in _UNIFORM_BOX)))
    return _fem_spec(
        f'3D empty box in a uniform field on tetrahedra ({order}, {name})', _UNIFORM_BOX,
        [FeatureSpec('applied', 'external_field', ['all'], dict(zip(('B_x', 'B_y', 'B_z'), _UNIFORM_FIELDS[name]))),
         FeatureSpec('probe', 'probe', ['probe_point'], {'component': 'B_abs'})],
        geometry=[GeometrySpec('probe_point', 'point', centre)], cell=_UNIFORM_CELL, discretization=order,
        backend=backend)


def _uniform_errors(results, field):
    """Deviations of B, mu0 H (relative to |B_ext|) and phi_m (relative to max |phi_m|) from the uniform field on the
    sample lattice, and of the centre probe."""
    applied = np.asarray(field, dtype=np.float64)
    scale = float(np.linalg.norm(applied))
    errors = {}
    centres = None
    for index, axis in enumerate(_AXES):
        flux_density, centres = _volume(results, f'B_{axis}')
        field_strength, _ = _volume(results, f'H_{axis}')
        errors[f'B_{axis}'] = float(np.max(np.abs(flux_density - applied[index]))) / scale
        errors[f'H_{axis}'] = float(np.max(np.abs(MU0 * field_strength - applied[index]))) / scale
    potential, _ = _volume(results, 'phi_m')
    grids = np.meshgrid(*centres, indexing='ij')
    expected = -(applied[0] * grids[0] + applied[1] * grids[1] + applied[2] * grids[2]) / MU0
    errors['phi_m'] = float(np.max(np.abs(potential - expected))) / float(np.max(np.abs(expected)))
    errors['probe_B_abs'] = abs(last_value(results, _MAGNETIC, 'probe') - scale) / scale
    return errors


def _check_uniform(results_by_name):
    errors = {f'{run}_{key}': value for run, (_, name) in _UNIFORM_RUNS.items()
              for key, value in _uniform_errors(results_by_name[run], _UNIFORM_FIELDS[name]).items()}
    return _report(errors)


# ---- magnet3d_fem_layered_slab / magnet3d_fem_saturating_slab ----
def _slab_spec(slab_axis, drive_axis, order, slab_properties=None, potential=_HELD_POTENTIAL, settings=None,
               kind='permeable'):
    """The magnet3d_layered_slab layout (selftests_3d._slab_parts) on Kuhn tetrahedra of the same cell."""
    features, materials, geometry = _slab_parts(slab_axis, drive_axis, slab_properties, potential)
    return _fem_spec(f'3D {kind} slab on tetrahedra normal to {_AXES[slab_axis]}, driven along {_AXES[drive_axis]} '
                     f'({order})', _SLAB_BOX, features, materials=materials, geometry=geometry, discretization=order,
                     settings=settings)


def _slab_layout(slab_axis, drive_axis, centres):
    """(iron mask, held mask, drive coordinate, (slab low, high)) on the sample lattice."""
    grids = np.meshgrid(*centres, indexing='ij')
    low, high = (fraction * _SLAB_BOX[slab_axis] for fraction in _SLAB_RANGE)
    iron = (grids[slab_axis] >= low) & (grids[slab_axis] < high)   # sample centres never lie on the slab faces
    position = grids[drive_axis]
    return iron, position < _SLAB_CELL, position, (low, high)


def _series_potential(position, field_air, field_iron, bounds, potential):
    """Omega [A] along the drive axis from the held layer face (Omega_0) through air, the iron band, air."""
    low, high = bounds
    drop = (field_air * (np.clip(position, _SLAB_CELL, low) - _SLAB_CELL)
            + field_iron * (np.clip(position, low, high) - low) + field_air * (np.clip(position, high, None) - high))
    return potential - drop


def _slab_expected(slab_axis, drive_axis, centres, iron_field_of_flux=None, iron_flux_of_field=None,
                   potential=_HELD_POTENTIAL):
    """Continuous (B_d, H_d, phi_m, iron mask) on the sample lattice: normal (slab axis = drive axis) one B through the
    series air / iron / air with Omega_0 = sum of length H; tangential H = Omega_0 / (L - layer) everywhere; B, H = 0
    and phi_m = Omega_0 in the held layer. Iron: mu_dc 1e5, or the curve (H(B) / B(H) callables)."""
    iron, held, position, (low, high) = _slab_layout(slab_axis, drive_axis, centres)
    free_length = _SLAB_BOX[drive_axis] - _SLAB_CELL
    if iron_field_of_flux is None:
        iron_field_of_flux = lambda flux: flux / (MU0 * _SLAB_PERMEABILITY)   # noqa: E731
        iron_flux_of_field = lambda field: MU0 * _SLAB_PERMEABILITY * field   # noqa: E731
    if slab_axis == drive_axis:
        air_length, iron_length = free_length - (high - low), high - low

        def magnetomotive_force(flux):
            return air_length * flux / MU0 + iron_length * iron_field_of_flux(flux)

        lower, upper = 0.0, MU0 * potential / air_length   # the air alone bounds B
        for _ in range(200):
            middle = 0.5 * (lower + upper)
            lower, upper = (middle, upper) if magnetomotive_force(middle) < potential else (lower, middle)
        flux = 0.5 * (lower + upper)
        field_air, field_iron = flux / MU0, iron_field_of_flux(flux)
        flux_density = np.full(position.shape, flux)
        field_strength = np.where(iron, field_iron, field_air)
        omega = _series_potential(position, field_air, field_iron, (low, high), potential)
    else:
        field = potential / free_length
        field_strength = np.full(position.shape, field)
        flux_density = np.where(iron, iron_flux_of_field(field), MU0 * field)
        omega = potential * (_SLAB_BOX[drive_axis] - position) / free_length
    return (np.where(held, 0.0, flux_density), np.where(held, 0.0, field_strength), np.where(held, potential, omega),
            iron & ~held)


def _layered_runs():
    runs = {f'{name}_quadratic': _slab_spec(*axes, 'quadratic') for name, axes in _slab_axes().items()}
    runs['normal_x_linear'] = _slab_spec(0, 0, 'linear')
    return runs


def _check_layered(results_by_name):
    axes_by_name = _slab_axes()
    errors = {}
    for run, results in results_by_name.items():
        slab_axis, drive_axis = axes_by_name[run.rsplit('_', 1)[0]]
        centres = _volume(results, 'phi_m')[1]
        expected = _slab_expected(slab_axis, drive_axis, centres)
        for key, value in _slab_errors(results, drive_axis, expected, _HELD_POTENTIAL, shape=None).items():
            errors[f'{run}_{key}'] = value
    return _report(errors)


def _saturating_runs():
    return {f'{material}_{orientation}': _slab_spec(*axes, 'linear', slab_properties=properties(),
                                                    potential=_SATURATING_POTENTIAL, settings=_SATURATION_SETTINGS,
                                                    kind=f'saturating ({material})')
            for material, properties in _SATURATING_MATERIALS.items()
            for orientation, axes in _SATURATING_ORIENTATIONS.items()}


def _check_saturating(results_by_name):
    """B, H, phi_m per run against the continuous series / imposed-H solution on the iron curve, and mu_eff = B / (mu0
    H) in the free iron."""
    errors, iterations = {}, {}
    for material in _SATURATING_MATERIALS:
        curve = _SATURATING_CURVES[material]
        for orientation, axes in _SATURATING_ORIENTATIONS.items():
            run = f'{material}_{orientation}'
            results = results_by_name[run]
            centres = _volume(results, 'phi_m')[1]
            expected = _slab_expected(*axes, centres, iron_field_of_flux=lambda flux: _inverse_curve(curve, flux),
                                      iron_flux_of_field=curve, potential=_SATURATING_POTENTIAL)
            for key, value in _slab_errors(results, axes[1], expected, _SATURATING_POTENTIAL, shape=None).items():
                errors[f'{run}_{key}'] = value
            permeability, _ = _volume(results, 'mu_eff')
            free_iron = expected[3]
            expected_permeability = expected[0][free_iron] / (MU0 * expected[1][free_iron])
            errors[f'{run}_mu_eff'] = float(np.max(np.abs(permeability[free_iron] / expected_permeability - 1.0)))
            iterations[run] = _nonlinear_iterations(results)
    return _report(errors, nonlinear_iterations=iterations)


# ---- magnet3d_fem_coil_energy_identity ----
_IDENTITY_CELL = 4e-5                                   # m, Kuhn cubes
_IDENTITY_CELLS = (12, 10, 10)
_IDENTITY_BOXES = {   # name: ((low cells), (high cells))
    'coil_outer': ((4, 2, 2), (8, 8, 8)),               # square winding around x: outer minus inner
    'coil_inner': ((4, 4, 4), (8, 6, 6)),
    'core': ((2, 4, 4), (10, 6, 6)),                    # fills the hole, beyond both coil ends
    'block': ((1, 7, 1), (3, 9, 3)),                    # off-axis permeable block: no symmetry
}
_IDENTITY_PERMEABILITY = {'core': 50.0, 'block': 5.0}
_IDENTITY_TURNS, _IDENTITY_CURRENT = 50, 0.2            # -, A
_IDENTITY_TOLERANCE = 1e-9                              # float64 quadrature of one direct solve (first estimate)


def _identity_spec(order, backend='cpu'):
    """Square coil around x with a permeable core and an off-axis block, every face insulation (the corner anchor)."""
    geometry = [_box(name, *(tuple(index * _IDENTITY_CELL for index in corner) for corner in corners))
                for name, corners in _IDENTITY_BOXES.items()]
    geometry.append(GeometrySpec('winding', 'difference', {'base': 'coil_outer', 'subtract': ['coil_inner']}))
    return _fem_spec(
        f'3D coil energy identity on tetrahedra ({order})', tuple(count * _IDENTITY_CELL for count in _IDENTITY_CELLS),
        [FeatureSpec('coil', 'coil', ['winding'], {'axis': 'x', 'turns': _IDENTITY_TURNS,
                                                   'current': _IDENTITY_CURRENT})],
        materials=[MaterialSpec(name, properties={'mu_dc': value}, selection=[name])
                   for name, value in _IDENTITY_PERMEABILITY.items()],
        geometry=geometry, cell=_IDENTITY_CELL, discretization=order, backend=backend)


def _identity_errors(results):
    """Relative errors of W = Lambda I / 2 and L = Lambda / I from the coil table result."""
    result = results.result(_MAGNETIC, 'coil')
    energy, linkage, inductance = (float(result[name]) for name in ('energy', 'flux_linkage', 'inductance'))
    return {'identity': abs(energy - 0.5 * linkage * _IDENTITY_CURRENT) / energy,
            'inductance': abs(inductance - linkage / _IDENTITY_CURRENT) / abs(inductance)}


def _check_identity(results_by_name):
    errors = {f'{run}_{key}': value for run, results in results_by_name.items()
              for key, value in _identity_errors(results).items()}
    inductances = {run: float(results.result(_MAGNETIC, 'coil')['inductance'])
                   for run, results in results_by_name.items()}
    return _report(errors, _IDENTITY_TOLERANCE, inductances=inductances)


# ---- magnet3d_fem_permeable_sphere (heavy, gmsh) ----
_SPHERE_RADIUS = 1e-4                                   # m
_SPHERE_BOX = 12.0 * _SPHERE_RADIUS                     # m: far-boundary error ~ (1/6)^3
_SPHERE_PERMEABILITY = 1000.0
_SPHERE_FIELD = 1e-3                                    # T, applied B_z
_SPHERE_SIZES = (2.0 * _SPHERE_RADIUS, _SPHERE_RADIUS / 3.0)   # m: maximum element size, on the sphere
_SPHERE_TOLERANCE = 0.03                                # faceted sphere, far boundary, P2 (first estimate)


def _sphere_runs():
    """The quadratic sphere run on gmsh tetrahedra; {} without gmsh (the check then passes as skipped)."""
    from ...fem.gmsh_io import find_gmsh
    if find_gmsh() is None:
        return {}
    centre = 0.5 * _SPHERE_BOX
    mesh = MeshSpec(name='tetrahedra', kind='tetrahedra', mode='nonuniform', dx=_SPHERE_SIZES[0], max_ratio=1.3,
                    mesher='gmsh', refinements=[RefinementSpec('sphere', _SPHERE_SIZES[1])])
    probes = {'probe_centre': centre, 'probe_inner': centre + 0.5 * _SPHERE_RADIUS}
    return {'quadratic': _fem_spec(
        '3D permeable sphere in a uniform field on gmsh tetrahedra', (_SPHERE_BOX,) * 3,
        [FeatureSpec('applied', 'external_field', ['all'], {'B_x': 0.0, 'B_y': 0.0, 'B_z': _SPHERE_FIELD}),
         *(FeatureSpec(name, 'probe', [f'{name}_point'], {'component': 'B_abs'}) for name in probes)],
        materials=[MaterialSpec('air', properties={'mu_dc': 1.0}, selection=['all']),
                   MaterialSpec('sphere_material', properties={'mu_dc': _SPHERE_PERMEABILITY}, selection=['sphere'])],
        geometry=[GeometrySpec('sphere', 'sphere', {'cx': centre, 'cy': centre, 'cz': centre, 'r': _SPHERE_RADIUS}),
                  *(GeometrySpec(f'{name}_point', 'point', {'x': centre, 'y': centre, 'z': z})
                    for name, z in probes.items())],
        mesh=mesh)}


def _check_sphere(results_by_name):
    """B_abs at the centre and at z = a / 2 against B_in = 3 mu_r / (mu_r + 2) B_0; skipped without gmsh."""
    if not results_by_name:
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _SPHERE_TOLERANCE,
                'details': {'skipped': 'gmsh not found (set OPEN_SOL_GMSH or put gmsh on PATH)'}}
    expected = 3.0 * _SPHERE_PERMEABILITY / (_SPHERE_PERMEABILITY + 2.0) * _SPHERE_FIELD
    results = results_by_name['quadratic']
    errors = {name: abs(last_value(results, _MAGNETIC, name) - expected) / expected
              for name in ('probe_centre', 'probe_inner')}
    return _report(errors, _SPHERE_TOLERANCE, expected_B_in=expected)


# ---- magnet3d_fem_cpu_gpu ----
def _backend_runs():
    """The quadratic oblique box and the quadratic coil identity on the cpu (host SuperLU) and on the gpu (Automatic:
    cuDSS with nvmath-python, else host SuperLU)."""
    return {f'{case}_{backend}': factory(backend) for backend in ('cpu', 'gpu')
            for case, factory in (('uniform', lambda name: _uniform_spec('quadratic', 'oblique', name)),
                                  ('coil', lambda name: _identity_spec('quadratic', name)))}


def _check_backends(results_by_name):
    """Exact errors of the cpu runs, and (gpu runs present) max |B_abs,cpu - B_abs,gpu| / max |B_abs|."""
    errors = {'uniform_cpu': worst_error(_uniform_errors(results_by_name['uniform_cpu'], _UNIFORM_FIELDS['oblique'])),
              'coil_cpu': worst_error(_identity_errors(results_by_name['coil_cpu']))}
    gpu_runs = [case for case in ('uniform', 'coil') if f'{case}_gpu' in results_by_name]
    for case in gpu_runs:
        magnitudes = [_volume(results_by_name[f'{case}_{backend}'], 'B_abs')[0] for backend in ('cpu', 'gpu')]
        errors[f'{case}_B_abs_cpu_gpu'] = (float(np.max(np.abs(magnitudes[0] - magnitudes[1])))
                                           / float(np.max(np.abs(magnitudes[1]))))
    return _report(errors, gpu_runs=gpu_runs)


FEM_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='magnet3d_fem_coil_drive',
        label='Magnetostatics 3D on tetrahedra: coil edge drive',
        description=(f'CPU check of coils3d.coil_edge_drive_per_ampere on a Kuhn mesh of {_DRIVE_CELLS[0]} x '
                     f'{_DRIVE_CELLS[1]} x {_DRIVE_CELLS[2]} cubes of {_DRIVE_CELL * 1e3:g} mm: a square winding around '
                     f'z (box difference, {_DRIVE_TURNS} turns). The axial length is the winding extent; the edge '
                     'integrals of T per ampere summed along the node line through the hole centre give turns (the '
                     'Ampere circulation, exact for s = 1 in the hole and the axial clipping), along a line outside '
                     'the winding 0, and s = integral / (turns / h * dz) lies in [0, 1] on the z edges within the '
                     f'winding extent. error = the largest error over {_DRIVE_TOLERANCE:g}; tolerance 1. Not run yet.'),
        specs=lambda: {}, check=_check_coil_drive),
    SelfTestCase(
        name='magnet3d_fem_uniform_field',
        label='Magnetostatics 3D on tetrahedra: empty box in a uniform field',
        description=(f'magnetostatics_3d on a builtin tetrahedra mesh (Kuhn cubes of {_UNIFORM_CELL * 1e6:g} um) of an '
                     f'empty {_UNIFORM_BOX[0] * 1e3:g} x {_UNIFORM_BOX[1] * 1e3:g} x {_UNIFORM_BOX[2] * 1e3:g} mm box, '
                     'quadratic and linear elements, external_field along z and oblique: phi_m = -B . r / mu0 on '
                     'the window-face dofs is in the element space, so B, mu0 H (element means) and phi_m are exact on '
                     'the sample lattice, and the B_abs probe at the centre. measured = largest deviation relative to '
                     f'|B_ext| (phi_m: to max |phi_m|), tolerance {_TOLERANCE:g} (float32 samples; first estimate). '
                     'Not run yet.'),
        specs=lambda: {run: _uniform_spec(*settings) for run, settings in _UNIFORM_RUNS.items()},
        check=_check_uniform),
    SelfTestCase(
        name='magnet3d_fem_layered_slab',
        label='Magnetostatics 3D on tetrahedra: permeable slab normal and tangential to the field',
        description=('The magnet3d_layered_slab layout (mu_dc 1e5 slab, ideal-iron first layer at Omega_0, symmetry '
                     f'face) on Kuhn tetrahedra of {_SLAB_CELL * 1e6:g} um (quadratic for normal_ / tangential_ x, y, '
                     'z; linear for normal_x). The slab faces and the held layer lie on element faces, so the FEM '
                     'holds the continuous solution exactly: normal B = mu0 Omega_0 / sum of length / mu over the free '
                     'length, tangential H = Omega_0 / (L - layer); phi_m piecewise linear, B, H = 0 and phi_m = '
                     'Omega_0 in the held layer. measured = largest deviation of B, H per component (relative to their '
                     f'maxima) and phi_m (relative to Omega_0) on the sample lattice, tolerance {_TOLERANCE:g} (first '
                     'estimate). Not run yet.'),
        specs=_layered_runs, check=_check_layered, heavy=True),
    SelfTestCase(
        name='magnet3d_fem_saturating_slab',
        label='Magnetostatics 3D on tetrahedra: saturating slab (J_s curve and B-H table)',
        description=('The magnet3d_saturating_slab layout and iron (J_s curve or its bh_curve table; ideal-iron layer '
                     f'at Omega = {_SATURATING_POTENTIAL:g} A) on Kuhn tetrahedra, linear elements: normal (drive along '
                     'the slab normal x: one B with air length B / mu0 + iron length H(B) = Omega_0, by bisection) and '
                     'tangential (drive along y: H = Omega_0 / (L - layer), iron on its curve). The FEM holds these '
                     'continuous solutions exactly once the per-element secant loop converged (loop tolerance '
                     f'{_SATURATION_SETTINGS["nonlinear_tolerance"]:g}, Anderson). measured = largest deviation of B, '
                     'H, phi_m (as magnet3d_fem_layered_slab) and of mu_eff / (B / (mu0 H)) - 1 in the free iron, '
                     f'tolerance {_TOLERANCE:g} (first estimate). Not run yet.'),
        specs=_saturating_runs, check=_check_saturating),
    SelfTestCase(
        name='magnet3d_fem_coil_energy_identity',
        label='Magnetostatics 3D on tetrahedra: coil energy identity W = Lambda I / 2',
        description=(f'magnetostatics_3d on Kuhn tetrahedra of {_IDENTITY_CELL * 1e6:g} um ({_IDENTITY_CELLS[0]} x '
                     f'{_IDENTITY_CELLS[1]} x {_IDENTITY_CELLS[2]} cubes), quadratic and linear: a square coil around '
                     f'x ({_IDENTITY_TURNS} turns x {_IDENTITY_CURRENT:g} A, edge drive) with a core of mu_dc '
                     f"{_IDENTITY_PERMEABILITY['core']:g} filling its hole and an off-axis block of mu_dc "
                     f"{_IDENTITY_PERMEABILITY['block']:g}, every face insulation (the corner anchor dof, no held "
                     'boundary term). Galerkin orthogonality with exact quadrature makes the energy int B^2 / (2 mu0 '
                     'mu) equal Lambda I / 2 of the linkage int T . B / I; L = Lambda / I. measured = the largest '
                     f'relative error (coil table result), tolerance {_IDENTITY_TOLERANCE:g} (first estimate). Not run '
                     'yet.'),
        specs=lambda: {order: _identity_spec(order) for order in _ORDERS}, check=_check_identity),
    SelfTestCase(
        name='magnet3d_fem_permeable_sphere',
        label='Magnetostatics 3D on tetrahedra: permeable sphere on gmsh tetrahedra',
        description=(f'Heavy, needs gmsh (else skipped and noted): a sphere of radius {_SPHERE_RADIUS * 1e3:g} mm and '
                     f'mu_dc {_SPHERE_PERMEABILITY:g} at the centre of a {_SPHERE_BOX * 1e3:g} mm cube in B_z = '
                     f'{_SPHERE_FIELD:g} T (external field on the window faces), quadratic elements on gmsh tetrahedra '
                     f'({_SPHERE_SIZES[0] * 1e3:g} mm maximum, {_SPHERE_SIZES[1] * 1e3:.3g} mm on the sphere, growth '
                     '1.3): B_abs probes at the centre and at z = a / 2 against B_in = 3 mu_r / (mu_r + 2) B_0. '
                     f'Tolerance {_SPHERE_TOLERANCE:g} (faceted sphere, far boundary ~(1/6)^3; first estimate). Not '
                     'run yet.'),
        specs=_sphere_runs, check=_check_sphere),
    SelfTestCase(
        name='magnet3d_fem_cpu_gpu',
        label='Magnetostatics 3D on tetrahedra: CPU backend vs GPU',
        description=('The quadratic oblique magnet3d_fem_uniform_field box and the quadratic coil identity on '
                     'solver_defaults.backend cpu (Automatic host SuperLU) and gpu (Automatic sparse_direct: a double '
                     'cuDSS Cholesky with nvmath-python, else SuperLU; the gpu run keeps the state on the device): the '
                     'cpu runs against their exact values, max |B_abs| difference relative to the gpu '
                     f'maximum. Without a usable GPU the gpu runs are skipped. Tolerance {_TOLERANCE:g}. Not run yet.'),
        specs=_backend_runs, check=_check_backends, backends=('cpu', 'gpu')),
)

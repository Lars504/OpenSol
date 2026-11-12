"""Analytic self-tests of magnetostatics_3d (stages 1-3 of magnetostatics_3d_plan): an empty box in a uniform field on
a graded mesh, layered slabs along every axis and saturating slabs (exact 1D discrete solutions), permeable and
magnetised spheres (closed
forms, a coarse / fine pair), a thick solenoid's axis field and Ampere total, the coil energy identity, a cored coil
against 2D axisymmetric magnetostatics, half models on symmetry / insulation planes against the full model and the
cpu backend; plus the 'needle_3d' template. Specs built fresh; registered by definition3d."""
import math

import numpy as np

from ...core.constants import MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre, scaled_worst_result, solver_role, with_exact_cg, worst_error
from ...model.spec import (AnalyticCurveSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshAxisSpec, MeshBandSpec, MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec,
                           SolverSpec, StudySpec)
from .selftests import (_IRON_PERMEABILITY, _IRON_SATURATION, _NONLINEAR_SETTINGS as _SATURATION_SETTINGS,
                        MAGNETOSTATICS_TYPE_NAME, _curve_flux_density, _nonlinear_iterations, _table_flux_density,
                        _table_rows)

MAGNETOSTATICS_3D_TYPE_NAME = 'magnetostatics_3d'   # = the definition's type name (definition3d imports this module)
_MAGNETIC = 'magnetic'                              # physics instance name = the type's default instance
_TOLERANCE = 1e-6   # exact discrete identities: float64 CG 1e-12, float32 field storage
_AXES = 'xyz'
_BOX_KEYS = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1')


# ---- shared helpers ----
def _magnetic_spec(title, size, features, materials=(), geometry=(), mesh=None, preconditioner=None, settings=None):
    """Stationary magnetostatics_3d-only model on the exact pcg node (float64, 1e-12)."""
    return with_exact_cg(ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size[0], size_y=size[1], size_z=size[2]),
        geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_3D_TYPE_NAME, settings=dict(settings or {}),
                             features=list(features))],
        mesh=mesh if mesh is not None else MeshSpec(),
        study=StudySpec(type='stationary')), preconditioner=preconditioner)


def _box(name, low, high):
    """GeometrySpec box between the corners low and high [m] (3-tuples)."""
    return GeometrySpec(name, 'box', dict(zip(_BOX_KEYS, (*low, *high))))


def _volume(results, component):
    """(float64 window values, (x, y, z) float64 cell centres [m]) of a magnetic field."""
    volume = results.volume(f'{_MAGNETIC}.{component}')
    return (np.asarray(volume.values, dtype=np.float64),
            tuple(np.asarray(centres, dtype=np.float64) for centres in volume.centres))


def _report(errors, tolerance=_TOLERANCE, **details):
    error = worst_error(errors)
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': tolerance,
            'details': {'errors': errors, **details}}


def _derived(results, name):
    return float(results.summary()['derived'][name])   # float() also reads 'nan'


def _circle(name, centre, radius, z_range):
    """GeometrySpec circle prism along z: centre (x, y) [m], radius [m], z_range (z0, z1) [m]."""
    return GeometrySpec(name, 'circle', {'cx': centre[0], 'cy': centre[1], 'r': radius,
                                         'z0': z_range[0], 'z1': z_range[1]})


def _ring(name, centre, radii, z_range):
    """Winding ring around z: the circle prism of radii[1] minus that of radii[0] (two consumed inputs + the
    difference)."""
    return [_circle(f'{name}_outer', centre, radii[1], z_range), _circle(f'{name}_inner', centre, radii[0], z_range),
            GeometrySpec(name, 'difference', {'base': f'{name}_outer', 'subtract': [f'{name}_inner']})]


def _ampere_total(results, coil_axis, axis_point):
    """Coil current [A] through the half plane normal to axis c = coil_axis + 2 (cyclic) from the coil axis toward +b
    (b = coil_axis + 1): J_c summed over the cells with centres beyond axis_point[0] along b, in the cell row just
    above axis_point[1] along c, times the centre spacings along the coil axis and b (exact on the uniform cells J
    lives in); > 0 for a counter-clockwise current."""
    along, across = (coil_axis + 1) % 3, (coil_axis + 2) % 3
    density, centres = _volume(results, f'J_{_AXES[across]}')
    row = int(np.searchsorted(centres[across], axis_point[1]))
    plane = np.moveaxis(density, (coil_axis, along, across), (0, 1, 2))[:, :, row]
    beyond = centres[along] > axis_point[0]
    spacing_axial, spacing_along = np.gradient(centres[coil_axis]), np.gradient(centres[along])
    return float(np.sum(plane[:, beyond] * spacing_axial[:, None] * spacing_along[None, beyond]))


# ---- magnet3d_uniform_field ----
_UNIFORM_BOX = (6e-4, 5e-4, 4e-4)                       # m
_UNIFORM_CELL = 5e-5                                    # m, base cells
_UNIFORM_FINE = 1.25e-5                                 # m, refinement of the 'refined' box: x, y and z graded
_UNIFORM_REFINED = ((2e-4, 1.5e-4, 1e-4), (3.5e-4, 3e-4, 2.5e-4))
_UNIFORM_FIELDS = {'x': (1e-3, 0.0, 0.0), 'y': (0.0, 1e-3, 0.0), 'z': (0.0, 0.0, 1e-3),
                   'oblique': (2e-3, -1e-3, 3e-3)}      # T
_GRADED_MIN_RATIO = 2.0                                 # widest / narrowest cell spacing per axis


def _uniform_spec(name):
    """Empty box in the named applied field, the 'refined' box graded in, a B_abs probe at the box centre."""
    field = _UNIFORM_FIELDS[name]
    centre = dict(zip(_AXES, (0.5 * size for size in _UNIFORM_BOX)))
    return _magnetic_spec(
        f'3D empty box in a uniform field ({name})', _UNIFORM_BOX,
        [FeatureSpec('applied', 'external_field', ['all'], dict(zip(('B_x', 'B_y', 'B_z'), field))),
         FeatureSpec('probe', 'probe', ['probe_point'], {'component': 'B_abs'})],
        geometry=[_box('refined', *_UNIFORM_REFINED), GeometrySpec('probe_point', 'point', centre)],
        mesh=MeshSpec(mode='nonuniform', dx=_UNIFORM_CELL, refinements=[RefinementSpec('refined', _UNIFORM_FINE, 0.0)]))


def _uniform_runs():
    return {name: _uniform_spec(name) for name in _UNIFORM_FIELDS}


def _uniform_errors(results, field):
    """Deviations of B, mu0 H and phi_m from the uniform field over the free cells (the held outer ring excluded),
    relative to |B_ext| / max |phi_m|; ValueError unless every axis is graded."""
    applied = np.asarray(field, dtype=np.float64)
    scale = float(np.linalg.norm(applied))
    interior = (slice(1, -1),) * 3
    errors = {}
    centres = None
    for index, axis in enumerate(_AXES):
        flux_density, centres = _volume(results, f'B_{axis}')
        field_strength, _ = _volume(results, f'H_{axis}')
        errors[f'B_{axis}'] = float(np.max(np.abs(flux_density[interior] - applied[index]))) / scale
        errors[f'H_{axis}'] = float(np.max(np.abs(MU0 * field_strength[interior] - applied[index]))) / scale
    for axis, axis_centres in zip(_AXES, centres):
        spacing = np.diff(axis_centres)
        if not np.max(spacing) >= _GRADED_MIN_RATIO * np.min(spacing):
            raise ValueError(f'expected graded {axis} cells (widest / narrowest >= {_GRADED_MIN_RATIO:g}), got '
                             f'{np.min(spacing):.4g} .. {np.max(spacing):.4g} m')
    potential, _ = _volume(results, 'phi_m')
    grids = np.meshgrid(*centres, indexing='ij')
    expected = -(applied[0] * grids[0] + applied[1] * grids[1] + applied[2] * grids[2]) / MU0
    errors['phi_m'] = (float(np.max(np.abs(potential[interior] - expected[interior])))
                       / float(np.max(np.abs(expected))))
    samples = np.asarray(results.result(_MAGNETIC, 'probe')['values'], dtype=np.float64)
    errors['probe_B_abs'] = abs(float(samples[-1]) - scale) / scale if samples.size else math.inf
    return errors


def _check_uniform(results_by_name):
    errors = {f'{name}_{key}': value for name, field in _UNIFORM_FIELDS.items()
              for key, value in _uniform_errors(results_by_name[name], field).items()}
    return _report(errors)


# ---- magnet3d_layered_slab ----
_SLAB_CELL = 1e-5                                       # m, uniform cubic cells
_SLAB_CELLS = (20, 10, 15)                              # the slab range below falls on cell edges along every axis
_SLAB_BOX = tuple(count * _SLAB_CELL for count in _SLAB_CELLS)
_SLAB_RANGE = (0.4, 0.6)                                # slab extent along its normal, fractions of the box
_SLAB_PERMEABILITY = 1e5
_HELD_POTENTIAL = 1.0                                   # A, Omega of the ideal-iron first cell layer


def _slab_axes():
    """{run: (slab normal axis, drive axis)}: drive along the normal (B uniform) and along the next axis (H
    uniform)."""
    runs = {}
    for axis in range(3):
        runs[f'normal_{_AXES[axis]}'] = (axis, axis)
        runs[f'tangential_{_AXES[axis]}'] = (axis, (axis + 1) % 3)
    return runs


def _slab_parts(slab_axis, drive_axis, slab_properties=None, potential=_HELD_POTENTIAL):
    """(features, materials, geometry) of the layered slab (also selftests_fem3d's): slab normal to slab_axis across
    the box (mu 1e5 unless slab_properties are given); the first cell layer along drive_axis is ideal iron at
    `potential` [A], the drive axis' hi face a symmetry face (u = 0), the other faces insulated."""
    low, high = [0.0] * 3, list(_SLAB_BOX)
    low[slab_axis], high[slab_axis] = (fraction * _SLAB_BOX[slab_axis] for fraction in _SLAB_RANGE)
    layer_high = list(_SLAB_BOX)
    layer_high[drive_axis] = _SLAB_CELL
    properties = {'mu_dc': _SLAB_PERMEABILITY} if slab_properties is None else dict(slab_properties)
    return ([FeatureSpec('iron', 'ideal_iron', ['iron_layer'], {'potential': potential}),
             FeatureSpec('mirror', 'symmetry', [f'{_AXES[drive_axis]}_hi'])],
            [MaterialSpec('air', properties={'mu_dc': 1.0}, selection=['all']),
             MaterialSpec('iron', properties=properties, selection=['slab'])],
            [_box('slab', low, high), _box('iron_layer', (0.0, 0.0, 0.0), layer_high)])


def _slab_spec(slab_axis, drive_axis, slab_properties=None, potential=_HELD_POTENTIAL, settings=None,
               kind='permeable'):
    """The _slab_parts model on uniform cubic cells."""
    features, materials, geometry = _slab_parts(slab_axis, drive_axis, slab_properties, potential)
    return _magnetic_spec(f'3D {kind} slab normal to {_AXES[slab_axis]}, driven along {_AXES[drive_axis]}', _SLAB_BOX,
                          features, materials=materials, geometry=geometry,
                          mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL), settings=settings)


def _slab_runs():
    return {name: _slab_spec(*axes) for name, axes in _slab_axes().items()}


def _slab_expected(slab_axis, drive_axis):
    """Exact discrete (B_d [T], H_d [A/m], phi_m [A]) window arrays of a layered-slab run: per row along the drive axis
    the series half-cell resistances w / mu from the held layer's face (u = mu0 Omega_0) to the symmetry face (u =
    0); B, H = 0 in the held layer."""
    centres = cell_centre(np.arange(_SLAB_CELLS[slab_axis]), _SLAB_CELL)
    low, high = (fraction * _SLAB_BOX[slab_axis] for fraction in _SLAB_RANGE)
    profile = np.where((centres >= low) & (centres < high), _SLAB_PERMEABILITY, 1.0)
    view = [1, 1, 1]
    view[slab_axis] = -1
    permeability = np.moveaxis(np.broadcast_to(profile.reshape(view), _SLAB_CELLS), drive_axis, 0)
    resistance = _SLAB_CELL / permeability
    resistance[0] = 0.0   # the held layer
    held_value = MU0 * _HELD_POTENTIAL
    flux = held_value / resistance.sum(axis=0)
    potential = held_value - flux * (np.cumsum(resistance, axis=0) - 0.5 * resistance)
    potential[0] = held_value
    flux_density = np.broadcast_to(flux, resistance.shape).copy()
    flux_density[0] = 0.0
    field_strength = flux_density / (MU0 * permeability)
    return tuple(np.moveaxis(array, 0, drive_axis) for array in (flux_density, field_strength, potential / MU0))


def _slab_errors(results, drive_axis, expected, held_potential=_HELD_POTENTIAL, shape=_SLAB_CELLS):
    """B and H per component against the (B_d, H_d, phi_m, ...) arrays `expected` (relative to max B_d / max H_d),
    phi_m relative to Omega_0 = held_potential; the window shape is checked unless shape is None (selftests_fem3d:
    the tetrahedra sample lattice)."""
    expected_flux, expected_field, expected_potential = expected[:3]
    errors = {}
    for index, axis in enumerate(_AXES):
        flux_density, _ = _volume(results, f'B_{axis}')
        if shape is not None and flux_density.shape != shape:
            raise ValueError(f'expected the window shape {shape}, got {flux_density.shape}')
        field_strength, _ = _volume(results, f'H_{axis}')
        driven = index == drive_axis
        errors[f'B_{axis}'] = (float(np.max(np.abs(flux_density - (expected_flux if driven else 0.0))))
                               / float(np.max(expected_flux)))
        errors[f'H_{axis}'] = (float(np.max(np.abs(field_strength - (expected_field if driven else 0.0))))
                               / float(np.max(expected_field)))
    potential, _ = _volume(results, 'phi_m')
    errors['phi_m'] = float(np.max(np.abs(potential - expected_potential))) / held_potential
    return errors


def _check_slabs(results_by_name):
    errors = {f'{name}_{key}': value for name, axes in _slab_axes().items()
              for key, value in _slab_errors(results_by_name[name], axes[1], _slab_expected(*axes)).items()}
    return _report(errors)


# ---- magnet3d_saturating_slab ----
# The layered-slab layout with the 2D test iron (Frohlich-Kennelly mu_dc 1000, J_s 1 T; or its bh_curve table, rows to
# 1500 A/m) in the slab. normal: drive along the slab normal x; the air (15 free cells) dominates the series reluctance,
# so the drive is nearly flux-imposed (H-based Picard factor (1 - s)(1 - w) ~ 0.8 at B ~ 0.59 T, Anderson-mixed), and
# the row's B sits inside the table rows. tangential: drive along y, H = Omega_0 / L = 7.8e5 A/m in every row, the iron
# deep in saturation (~2 T; the table on its mu0 extension): the H-based proposal is exact from the first solve.
_SATURATING_POTENTIAL = 70.0                            # A, Omega of the ideal-iron layer
_SATURATING_ORIENTATIONS = {'normal': (0, 0), 'tangential': (0, 1)}   # (slab normal axis, drive axis)
_SATURATING_MATERIALS = {'curve': lambda: {'mu_dc': _IRON_PERMEABILITY, 'saturation_polarization': _IRON_SATURATION},
                         'table': lambda: {'bh_curve': _table_rows()}}
_SATURATING_CURVES = {'curve': _curve_flux_density, 'table': _table_flux_density}   # forward B(H) of the checks


def _saturating_runs():
    return {f'{material}_{orientation}': _slab_spec(*axes, slab_properties=properties(),
                                                    potential=_SATURATING_POTENTIAL, settings=_SATURATION_SETTINGS,
                                                    kind=f'saturating ({material})')
            for material, properties in _SATURATING_MATERIALS.items()
            for orientation, axes in _SATURATING_ORIENTATIONS.items()}


def _inverse_curve(flux_density_of_field, flux_density):
    """H [A/m] at B >= 0 [T] of a monotone curve with B(H) >= mu0 H: bisection."""
    low, high = 0.0, flux_density / MU0
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if flux_density_of_field(middle) < flux_density else (low, middle)
    return 0.5 * (low + high)


def _saturating_row(iron, flux_density_of_field):
    """(B [T], H [A/m] per free cell) of one drive row of uniform free cells (bool iron flags): sum of w H(B) =
    Omega_0, by bisection on B."""
    def field_of(flux_density):
        return np.where(iron, _inverse_curve(flux_density_of_field, flux_density), flux_density / MU0)

    def magnetomotive_force(flux_density):
        return _SLAB_CELL * float(np.sum(field_of(flux_density)))

    low, high = 0.0, 1.0
    while magnetomotive_force(high) < _SATURATING_POTENTIAL:
        low, high = high, 2.0 * high
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if magnetomotive_force(middle) < _SATURATING_POTENTIAL else (low, middle)
    flux_density = 0.5 * (low + high)
    return flux_density, field_of(flux_density)


def _saturating_expected(material, slab_axis, drive_axis):
    """Exact discrete (B_d [T], H_d [A/m], phi_m [A], iron mask) window arrays of a saturating slab run: per drive row
    one face flux B (no transverse flux) with sum over its free cells of w H(B) = Omega_0 (air B / mu0, iron the
    curve's inverse), so every iron cell sits on its curve at the row's B; B, H = 0 in the held layer."""
    centres = cell_centre(np.arange(_SLAB_CELLS[slab_axis]), _SLAB_CELL)
    low, high = (fraction * _SLAB_BOX[slab_axis] for fraction in _SLAB_RANGE)
    view = [1, 1, 1]
    view[slab_axis] = -1
    profile = ((centres >= low) & (centres < high)).reshape(view)
    iron = np.moveaxis(np.broadcast_to(profile, _SLAB_CELLS), drive_axis, 0)
    rows = iron.reshape(iron.shape[0], -1)
    flux = np.zeros(rows.shape[1])
    field = np.zeros(rows.shape)
    solved = {}
    for column in range(rows.shape[1]):
        key = rows[1:, column].tobytes()
        if key not in solved:
            solved[key] = _saturating_row(rows[1:, column], _SATURATING_CURVES[material])
        flux[column], field[1:, column] = solved[key]
    field = field.reshape(iron.shape)
    flux_density = np.broadcast_to(flux.reshape(iron.shape[1:]), iron.shape).copy()
    flux_density[0] = 0.0
    drop = MU0 * _SLAB_CELL * field             # u drop across each free cell (0 in the held layer)
    held_value = MU0 * _SATURATING_POTENTIAL
    potential = held_value - (np.cumsum(drop, axis=0) - 0.5 * drop)
    potential[0] = held_value
    return tuple(np.moveaxis(array, 0, drive_axis)
                 for array in (flux_density, field, potential / MU0, np.array(iron)))


def _check_saturating(results_by_name):
    """B, H, phi_m per run against _saturating_expected, and mu_eff = B / (mu0 H) in the free iron cells."""
    errors, iterations = {}, {}
    for material in _SATURATING_MATERIALS:
        for orientation, axes in _SATURATING_ORIENTATIONS.items():
            name = f'{material}_{orientation}'
            results = results_by_name[name]
            *expected, iron = _saturating_expected(material, *axes)
            for key, value in _slab_errors(results, axes[1], expected, _SATURATING_POTENTIAL).items():
                errors[f'{name}_{key}'] = value
            permeability, _ = _volume(results, 'mu_eff')
            free_iron = iron & (expected[0] > 0.0)
            expected_permeability = expected[0][free_iron] / (MU0 * expected[1][free_iron])
            errors[f'{name}_mu_eff'] = float(np.max(np.abs(permeability[free_iron] / expected_permeability - 1.0)))
            iterations[name] = _nonlinear_iterations(results)
    return _report(errors, nonlinear_iterations=iterations)


# ---- magnet3d_permeable_sphere / magnet3d_magnetized_sphere ----
_SPHERE_RADIUS = 1e-4                                   # m
_SPHERE_BOX = 12.0 * _SPHERE_RADIUS                     # m, cube side: far-ring error ~ (a / 6 a)^3
_SPHERE_RESOLUTIONS = {'coarse': 10, 'fine': 20}        # cells per radius in the refinement
_SPHERE_TOLERANCE = 0.02                                # staircase sphere and far ring
# mu_r >> 1: a staircased sphere's excess polarizability is first order in h (3 % at a / 20), so the permeable case
# compares the first-order Richardson value of the coarse / fine pair; left: far ring ~0.4 % + staircase scatter
_SPHERE_EXTRAPOLATED_TOLERANCE = 0.01
_SPHERE_PERMEABILITY = 1000.0
_SPHERE_FIELD = 1e-3                                    # T, applied B_z
_SPHERE_REMANENCE = 1.0                                 # T, B_r along z
_CORE_NAMES = ('B_core', 'H_core')                      # derived means over the central half radius


def _sphere_spec(title, resolution, features, sphere_properties):
    """Sphere of radius a at the cube centre in 12 a of air; refinement a / resolution, max_ratio 1.3 to a / 2 cells;
    derived values B_core / H_core = mean B_z / H_z over the sphere of radius a / 2."""
    centre = 0.5 * _SPHERE_BOX
    centre_values = {'cx': centre, 'cy': centre, 'cz': centre}
    spec = _magnetic_spec(
        title, (_SPHERE_BOX,) * 3, features,
        materials=[MaterialSpec('air', properties={'mu_dc': 1.0}, selection=['all']),
                   MaterialSpec('sphere_material', properties=dict(sphere_properties), selection=['sphere'])],
        geometry=[GeometrySpec('sphere', 'sphere', {**centre_values, 'r': _SPHERE_RADIUS}),
                  GeometrySpec('core', 'sphere', {**centre_values, 'r': 0.5 * _SPHERE_RADIUS})],
        mesh=MeshSpec(mode='nonuniform', dx=0.5 * _SPHERE_RADIUS, max_ratio=1.3, refinements=[
            RefinementSpec('sphere', _SPHERE_RADIUS / resolution, 0.25 * _SPHERE_RADIUS)]),
        preconditioner='multigrid')
    spec.derived_values = [DerivedValueSpec(name, f'{_MAGNETIC}_{component}', ['core'], 'mean')
                           for name, component in zip(_CORE_NAMES, ('B_z', 'H_z'))]
    return spec


def _permeable_sphere_runs():
    return {run: _sphere_spec(f'3D permeable sphere in a uniform field ({run})', resolution,
                              [FeatureSpec('applied', 'external_field', ['all'],
                                           {'B_x': 0.0, 'B_y': 0.0, 'B_z': _SPHERE_FIELD})],
                              {'mu_dc': _SPHERE_PERMEABILITY})
            for run, resolution in _SPHERE_RESOLUTIONS.items()}


def _magnetized_sphere_runs():
    return {run: _sphere_spec(f'3D uniformly magnetised sphere ({run})', resolution, [],
                              {'remanence_z': _SPHERE_REMANENCE})
            for run, resolution in _SPHERE_RESOLUTIONS.items()}


def _check_sphere(results_by_name, expected, extrapolate=False):
    """Largest relative deviation of the core means from `expected` ({derived name: value}) per run; error = the fine
    run's (extrapolate: of the first-order Richardson values), inf unless the fine run beats the coarse one."""
    run_errors, measured = {}, {}
    for run in _SPHERE_RESOLUTIONS:
        derived = results_by_name[run].summary()['derived']
        measured[run] = {name: float(derived[name]) for name in expected}   # float() reads 'nan'
        run_errors[run] = max(abs(measured[run][name] - value) / abs(value) for name, value in expected.items())
    falls = run_errors['fine'] < run_errors['coarse']
    details = {'run_errors': run_errors, 'measured': measured, 'expected': expected, 'error_falls': falls}
    compared, tolerance = run_errors['fine'], _SPHERE_TOLERANCE
    if extrapolate:
        refinement = _SPHERE_RESOLUTIONS['fine'] / _SPHERE_RESOLUTIONS['coarse']   # cell size ratio, 2
        extrapolated = {name: (refinement * measured['fine'][name] - measured['coarse'][name]) / (refinement - 1.0)
                        for name in expected}
        compared = max(abs(extrapolated[name] - value) / abs(value) for name, value in expected.items())
        tolerance = _SPHERE_EXTRAPOLATED_TOLERANCE
        details.update(extrapolated=extrapolated, extrapolated_error=compared)
    error = compared if falls and math.isfinite(compared) else math.inf
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': tolerance, 'details': details}


def _check_permeable_sphere(results_by_name):
    """B_in = 3 mu_r / (mu_r + 2) B_0, H_in = B_in / (mu0 mu_r)."""
    flux_density = 3.0 * _SPHERE_PERMEABILITY / (_SPHERE_PERMEABILITY + 2.0) * _SPHERE_FIELD
    return _check_sphere(results_by_name, {'B_core': flux_density,
                                           'H_core': flux_density / (MU0 * _SPHERE_PERMEABILITY)}, extrapolate=True)


def _check_magnetized_sphere(results_by_name):
    """B_in = 2 B_r / 3, H_in = -B_r / (3 mu0)."""
    return _check_sphere(results_by_name, {'B_core': 2.0 * _SPHERE_REMANENCE / 3.0,
                                           'H_core': -_SPHERE_REMANENCE / (3.0 * MU0)})


# ---- magnet3d_solenoid_axis ----
_SOLENOID_BOX = 6e-3                                    # m, cube side
_SOLENOID_RADII = (4e-4, 6e-4)                          # m, winding r1, r2
_SOLENOID_HALF_LENGTH = 5e-4                            # m, b: the winding spans z_c +- b
_SOLENOID_CELL = 2.5e-5                                 # m, uniform band cells around the winding
_SOLENOID_BAND = (7.5e-4, 6.5e-4)                       # m, band half widths (x and y, z) about the centre
_SOLENOID_TURNS, _SOLENOID_CURRENT = 100, 0.1           # -, A
_SOLENOID_SAMPLED = 0.8                                 # axis cells within this fraction of b are compared
_SOLENOID_TOLERANCES = {'axis_B_z': 0.02, 'ampere': 1e-5}   # staircase ring / float32 J sums


def _solenoid_spec():
    """Thick solenoid around z at the cube centre, every face insulation (the corner anchor), 25 um bands graded by
    1.3 to 0.4 mm cells; multigrid pcg."""
    centre = 0.5 * _SOLENOID_BOX
    lateral, axial = _SOLENOID_BAND
    winding = (centre - _SOLENOID_HALF_LENGTH, centre + _SOLENOID_HALF_LENGTH)
    bands = {'x': lateral, 'y': lateral, 'z': axial}
    return _magnetic_spec(
        '3D thick solenoid', (_SOLENOID_BOX,) * 3,
        [FeatureSpec('coil', 'coil', ['winding'], {'axis': 'z', 'turns': _SOLENOID_TURNS,
                                                   'current': _SOLENOID_CURRENT})],
        geometry=_ring('winding', (centre, centre), _SOLENOID_RADII, winding),
        mesh=MeshSpec(mode='nonuniform', dx=4e-4, max_ratio=1.3, axes=[
            MeshAxisSpec(axis, bands=[MeshBandSpec(centre - half, centre + half, _SOLENOID_CELL)])
            for axis, half in bands.items()]),
        preconditioner='multigrid')


def _solenoid_axis_field(offset):
    """Closed-form B_z [T] on the axis of the uniform-J thick solenoid at offset z - z_c [m]."""
    inner, outer = _SOLENOID_RADII
    density = _SOLENOID_TURNS * _SOLENOID_CURRENT / (2.0 * _SOLENOID_HALF_LENGTH * (outer - inner))

    def end_term(distance):
        return distance * math.log((outer + math.hypot(outer, distance)) / (inner + math.hypot(inner, distance)))

    return 0.5 * MU0 * density * (end_term(_SOLENOID_HALF_LENGTH - offset) + end_term(_SOLENOID_HALF_LENGTH + offset))


def _check_solenoid(results_by_name):
    """B_z on the axis (bilinear in x, y) against the closed form over |z - z_c| <= 0.8 b, relative to the centre
    value; the Ampere total of J through the half plane from the axis against turns * current."""
    results = results_by_name['run']
    centre = 0.5 * _SOLENOID_BOX
    flux_density, (x_centres, y_centres, z_centres) = _volume(results, 'B_z')
    i = int(np.searchsorted(x_centres, centre)) - 1
    j = int(np.searchsorted(y_centres, centre)) - 1
    tx = (centre - x_centres[i]) / (x_centres[i + 1] - x_centres[i])
    ty = (centre - y_centres[j]) / (y_centres[j + 1] - y_centres[j])
    on_axis = ((1 - tx) * (1 - ty) * flux_density[i, j] + tx * (1 - ty) * flux_density[i + 1, j]
               + (1 - tx) * ty * flux_density[i, j + 1] + tx * ty * flux_density[i + 1, j + 1])
    sampled = np.abs(z_centres - centre) <= _SOLENOID_SAMPLED * _SOLENOID_HALF_LENGTH
    expected = np.array([_solenoid_axis_field(z - centre) for z in z_centres[sampled]])
    errors = {'axis_B_z': float(np.max(np.abs(on_axis[sampled] - expected))) / _solenoid_axis_field(0.0)}
    total = _ampere_total(results, 2, (centre, centre))
    ampere_turns = _SOLENOID_TURNS * _SOLENOID_CURRENT
    errors['ampere'] = abs(total - ampere_turns) / ampere_turns
    return scaled_worst_result(errors, _SOLENOID_TOLERANCES, B_centre=_solenoid_axis_field(0.0), ampere_total=total)


# ---- magnet3d_coil_energy_identity ----
_IDENTITY_CELL = 2e-5                                   # m, uniform cubic cells
_IDENTITY_CELLS = (24, 20, 20)
_IDENTITY_BOXES = {   # name: ((low cells), (high cells))
    'coil_outer': ((8, 5, 5), (16, 15, 15)),            # square winding around x: outer minus inner
    'coil_inner': ((8, 8, 8), (16, 12, 12)),
    'core': ((3, 9, 9), (21, 11, 11)),                  # through the hole, beyond both coil ends
    'block': ((2, 14, 2), (6, 18, 6)),                  # off-axis permeable block: no symmetry
}
_IDENTITY_AXIS = (10, 10)                               # cells: the coil axis (y, z), on cell edges
_IDENTITY_PERMEABILITY = {'core': 50.0, 'block': 5.0}
_IDENTITY_TURNS, _IDENTITY_CURRENT = 50, 0.2            # -, A
_IDENTITY_TOLERANCES = {'identity': 1e-9, 'ampere': 1e-6}   # float64 scalars / float32 J on uniform cells


def _coil_energy_spec():
    """Square coil around x with a permeable core through its hole and an off-axis block, every face insulation (the
    corner anchor); derived W, Lambda, L from the scalars."""
    geometry = [_box(name, *(tuple(index * _IDENTITY_CELL for index in corner) for corner in corners))
                for name, corners in _IDENTITY_BOXES.items()]
    geometry.append(GeometrySpec('winding', 'difference', {'base': 'coil_outer', 'subtract': ['coil_inner']}))
    spec = _magnetic_spec(
        '3D coil energy identity', tuple(count * _IDENTITY_CELL for count in _IDENTITY_CELLS),
        [FeatureSpec('coil', 'coil', ['winding'], {'axis': 'x', 'turns': _IDENTITY_TURNS,
                                                   'current': _IDENTITY_CURRENT})],
        materials=[MaterialSpec(name, properties={'mu_dc': value}, selection=[name])
                   for name, value in _IDENTITY_PERMEABILITY.items()],
        geometry=geometry, mesh=MeshSpec(mode='uniform', dx=_IDENTITY_CELL))
    spec.derived_values = [DerivedValueSpec('W', f'{_MAGNETIC}_energy', [], 'max'),
                           DerivedValueSpec('Lambda', f'{_MAGNETIC}_coil_flux_linkage', [], 'max'),
                           DerivedValueSpec('L', f'{_MAGNETIC}_coil_inductance', [], 'max')]
    return spec


def _coil_energy_errors(results):
    """Relative errors of W = Lambda I / 2 (with L = Lambda / I) and of the Ampere total against turns * current."""
    energy, linkage, inductance = (_derived(results, name) for name in ('W', 'Lambda', 'L'))
    ampere_turns = _IDENTITY_TURNS * _IDENTITY_CURRENT
    total = _ampere_total(results, 0, tuple(index * _IDENTITY_CELL for index in _IDENTITY_AXIS))
    return {'identity': max(abs(energy - 0.5 * linkage * _IDENTITY_CURRENT) / energy,
                            abs(inductance - linkage / _IDENTITY_CURRENT) / abs(inductance)),
            'ampere': abs(total - ampere_turns) / ampere_turns}


def _check_coil_energy(results_by_name):
    results = results_by_name['run']
    return scaled_worst_result(_coil_energy_errors(results), _IDENTITY_TOLERANCES,
                               energy=_derived(results, 'W'), inductance=_derived(results, 'L'))


# ---- magnet3d_vs_axisymmetric ----
_AXI_RADIUS = 4e-3                                      # m, 2D domain radius = 3D half side
_AXI_HEIGHT = 1.2e-2                                    # m
_AXI_CORE = (2.5e-4, 1.5e-3)                            # m, core radius and half length (centred on the coil)
_AXI_COIL_RADII = (3.5e-4, 5.5e-4)                      # m
_AXI_COIL_HALF_LENGTH = 5e-4                            # m
_AXI_SENSOR = (5e-5, 1e-4, 2.5e-5)                      # m, sensor radius, gap below the core tip, half height
_AXI_PERMEABILITY = 100.0
_AXI_TURNS, _AXI_CURRENT = 100, 0.1                     # -, A
_AXI_CELLS = {'3d': 2.5e-5, '2d': 1.25e-5}              # m, 3D band cells / 2D uniform cells
_AXI_TOLERANCE = 0.02                                   # staircase core and winding in 3D (first estimate)


def _axi_ranges():
    """{part: (radius or radii, (z0, z1))} about the window height's centre."""
    centre = 0.5 * _AXI_HEIGHT
    core_radius, core_half = _AXI_CORE
    sensor_radius, gap, sensor_half = _AXI_SENSOR
    sensor_centre = centre - core_half - gap
    return {'core': (core_radius, (centre - core_half, centre + core_half)),
            'winding': (_AXI_COIL_RADII, (centre - _AXI_COIL_HALF_LENGTH, centre + _AXI_COIL_HALF_LENGTH)),
            'sensor': (sensor_radius, (sensor_centre - sensor_half, sensor_centre + sensor_half))}


def _axi_derived(axial_component):
    return [DerivedValueSpec('L', f'{_MAGNETIC}_coil_inductance', [], 'max'),
            DerivedValueSpec('B_sensor', f'{_MAGNETIC}_{axial_component}', ['sensor'], 'mean')]


def _axi_3d_spec():
    """The cored coil in a 2R x 2R x height box (every face insulation), 25 um bands graded by 1.3; multigrid pcg."""
    ranges = _axi_ranges()
    axis = (_AXI_RADIUS, _AXI_RADIUS)
    core_radius, core_range = ranges['core']
    sensor_radius, sensor_range = ranges['sensor']
    lateral = _AXI_COIL_RADII[1] + 1e-4
    axial = (sensor_range[0] - 1e-4, core_range[1] + 1e-4)
    spec = _magnetic_spec(
        '3D cored coil (vs axisymmetric)', (2.0 * _AXI_RADIUS, 2.0 * _AXI_RADIUS, _AXI_HEIGHT),
        [FeatureSpec('coil', 'coil', ['winding'], {'axis': 'z', 'turns': _AXI_TURNS, 'current': _AXI_CURRENT})],
        materials=[MaterialSpec('core', properties={'mu_dc': _AXI_PERMEABILITY}, selection=['core'])],
        geometry=[_circle('core', axis, core_radius, core_range), _circle('sensor', axis, sensor_radius, sensor_range),
                  *_ring('winding', axis, *ranges['winding'])],
        mesh=MeshSpec(mode='nonuniform', dx=4e-4, max_ratio=1.3, axes=[
            MeshAxisSpec('x', bands=[MeshBandSpec(_AXI_RADIUS - lateral, _AXI_RADIUS + lateral, _AXI_CELLS['3d'])]),
            MeshAxisSpec('y', bands=[MeshBandSpec(_AXI_RADIUS - lateral, _AXI_RADIUS + lateral, _AXI_CELLS['3d'])]),
            MeshAxisSpec('z', bands=[MeshBandSpec(*axial, _AXI_CELLS['3d'])])]),
        preconditioner='multigrid')
    spec.derived_values = _axi_derived('B_z')
    return spec


def _axi_2d_spec():
    """The same coil, core and sensor in axisymmetric 2D magnetostatics (x = r, y = z; insulation faces), uniform
    12.5 um cells; multigrid pcg."""
    ranges = _axi_ranges()

    def rectangle(name, radii, z_range):
        return GeometrySpec(name, 'rectangle', {'x0': radii[0], 'y0': z_range[0], 'x1': radii[1], 'y1': z_range[1]})

    core_radius, core_range = ranges['core']
    sensor_radius, sensor_range = ranges['sensor']
    spec = with_exact_cg(ModelSpec(
        title='Cored coil, axisymmetric 2D (vs 3D)',
        domain=DomainSpec(size_x=_AXI_RADIUS, size_y=_AXI_HEIGHT),
        geometry=[rectangle('core', (0.0, core_radius), core_range),
                  rectangle('sensor', (0.0, sensor_radius), sensor_range),
                  rectangle('winding', *ranges['winding'])],
        materials=[MaterialSpec('core', properties={'mu_dc': _AXI_PERMEABILITY}, selection=['core'])],
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_TYPE_NAME, settings={'geometry': 'axisymmetric'}, features=[
            FeatureSpec('coil', 'coil', ['winding'], {'turns': _AXI_TURNS, 'current': _AXI_CURRENT})])],
        mesh=MeshSpec(mode='uniform', dx=_AXI_CELLS['2d']),
        study=StudySpec(type='stationary')), preconditioner='multigrid')
    spec.derived_values = _axi_derived('B_y')
    return spec


def _check_vs_axisymmetric(results_by_name):
    """Relative 3D - 2D differences of the inductance and the sensor mean of the axial B."""
    values = {run: {name: _derived(results_by_name[run], name) for name in ('L', 'B_sensor')} for run in ('3d', '2d')}
    errors = {name: abs(values['3d'][name] - values['2d'][name]) / abs(values['2d'][name]) for name in ('L', 'B_sensor')}
    return scaled_worst_result(errors, {name: _AXI_TOLERANCE for name in errors}, values=values)


# ---- magnet3d_mirror_parity ----
_PARITY_CELL = 1e-5                                     # m, uniform cubic cells
_PARITY_CELLS = (16, 12, 16)
_PARITY_MAGNET = ((6, 4, 6), (10, 8, 10))               # cells: centred cube, its edges on cell edges
_PARITY_PERMEABILITY = 2.0                              # H != B / mu0 in the magnet
_PARITY_REMANENCE = 1.0                                 # T along z: B_z even, phi_m odd about the z midplane
_PARITY_HALVES = {'half_z': (2, 'symmetry'), 'half_y': (1, 'magnetic_insulation')}
_PARITY_FIELDS = ('phi_m', 'B_x', 'B_y', 'B_z', 'H_x', 'H_y', 'H_z')


def _parity_spec(title, cells, extra_features):
    """Magnetised cube (remanence_z, mu 2) on uniform cells, x faces symmetry (the anchor), plus extra_features."""
    size = tuple(count * _PARITY_CELL for count in cells)
    low, high = (tuple(index * _PARITY_CELL for index in corner) for corner in _PARITY_MAGNET)
    return _magnetic_spec(
        title, size, [FeatureSpec('x_mirror', 'symmetry', ['x_lo', 'x_hi']), *extra_features],
        materials=[MaterialSpec('magnet', properties={'mu_dc': _PARITY_PERMEABILITY,
                                                      'remanence_z': _PARITY_REMANENCE}, selection=['cube'])],
        geometry=[_box('cube', low, high)], mesh=MeshSpec(mode='uniform', dx=_PARITY_CELL))


def _parity_runs():
    runs = {'full': _parity_spec('3D magnetised cube, full model', _PARITY_CELLS, [])}
    for name, (axis, feature_type) in _PARITY_HALVES.items():
        cells = list(_PARITY_CELLS)
        cells[axis] //= 2
        runs[name] = _parity_spec(f'3D magnetised cube, lower half along {_AXES[axis]} ({feature_type})', cells,
                                  [FeatureSpec('half_plane', feature_type, [f'{_AXES[axis]}_hi'])])
    return runs


def _check_parity(results_by_name):
    """Each half model's fields against the full model's lower half; per field relative to its group maximum (phi_m,
    the B components, the H components of the full model)."""
    full = {name: _volume(results_by_name['full'], name)[0] for name in _PARITY_FIELDS}
    scales = {group: max(float(np.max(np.abs(full[name]))) for name in _PARITY_FIELDS if name.startswith(group))
              for group in ('phi', 'B', 'H')}
    errors = {}
    for half, (axis, _) in _PARITY_HALVES.items():
        index = [slice(None)] * 3
        index[axis] = slice(0, _PARITY_CELLS[axis] // 2)
        for name in _PARITY_FIELDS:
            values = _volume(results_by_name[half], name)[0]
            reference = full[name][tuple(index)]
            if values.shape != reference.shape:
                raise ValueError(f'{half} {name}: window {values.shape}, expected {reference.shape}')
            errors[f'{half}_{name}'] = (float(np.max(np.abs(values - reference)))
                                        / scales[name.split('_')[0]])
    return _report(errors, scales=scales)


# ---- magnet3d_cpu_gpu ----
_BACKEND_RUNS = {   # run -> (case, spec factory, closed-form arguments)
    'uniform_oblique': ('uniform', lambda: _uniform_spec('oblique'), 'oblique'),
    'slab_normal_z': ('slab', lambda: _slab_spec(2, 2), (2, 2)),
    'slab_tangential_x': ('slab', lambda: _slab_spec(0, 1), (0, 1)),
    'coil_energy': ('coil', _coil_energy_spec, None),
}
_PARITY_RUNS = ('uniform_oblique', 'slab_normal_z', 'coil_energy')   # also on the gpu for the cpu vs gpu comparison


def _on_cpu(spec):
    """spec on the cpu backend with a sparse_direct 'linear' node (host SuperLU of the 3D window; Automatic there is
    multigrid pcg)."""
    spec.solver_defaults.backend = 'cpu'
    spec.solvers = [SolverSpec('cpu_direct', 'sparse_direct', {})]
    for physics in spec.physics:
        physics.solvers = {'linear': 'cpu_direct'}
    return spec


def _backend_runs():
    runs = {f'{name}_cpu': _on_cpu(factory()) for name, (_case, factory, _arguments) in _BACKEND_RUNS.items()}
    for name in _PARITY_RUNS:
        spec = _BACKEND_RUNS[name][1]()
        spec.solver_defaults.backend = 'gpu'
        runs[f'{name}_gpu'] = spec
    return runs


def _check_backends(results_by_name):
    """Closed-form errors (coil: the energy identity) of the cpu runs and (gpu runs present) max |B_abs,cpu -
    B_abs,gpu| / max |B_abs|; raises unless each cpu run recorded a cpu sparse_direct linear solver."""
    errors = {}
    for name, (case, _factory, arguments) in _BACKEND_RUNS.items():
        results = results_by_name[f'{name}_cpu']
        role = solver_role(results, _MAGNETIC)
        if role.get('backend') != 'cpu' or role.get('type') != 'sparse_direct':
            raise ValueError(f"run '{name}_cpu': expected a cpu sparse_direct linear solver, summary 'solvers' "
                             f'records {role!r}')
        if case == 'uniform':
            case_errors = _uniform_errors(results, _UNIFORM_FIELDS[arguments])
        elif case == 'slab':
            case_errors = _slab_errors(results, arguments[1], _slab_expected(*arguments))
        else:   # the energy identity only: the Ampere total is a float32 sum (its own tolerance)
            case_errors = {'identity': _coil_energy_errors(results)['identity']}
        errors[f'{name}_cpu'] = worst_error(case_errors)
    gpu_runs = [name for name in _PARITY_RUNS if f'{name}_gpu' in results_by_name]   # absent without a GPU
    for name in gpu_runs:
        magnitudes = [_volume(results_by_name[f'{name}_{backend}'], 'B_abs')[0] for backend in ('cpu', 'gpu')]
        errors[f'{name}_B_abs_cpu_gpu'] = (float(np.max(np.abs(magnitudes[0] - magnitudes[1])))
                                           / float(np.max(np.abs(magnitudes[1]))))
    return _report(errors, gpu_runs=gpu_runs)


SELFTEST_CASES_3D = (
    SelfTestCase(
        name='magnet3d_uniform_field',
        label='Magnetostatics 3D: empty box in a uniform field (graded)',
        description=(f'magnetostatics_3d in an empty {_UNIFORM_BOX[0] * 1e3:g} x {_UNIFORM_BOX[1] * 1e3:g} x '
                     f'{_UNIFORM_BOX[2] * 1e3:g} mm vacuum box, mesh nonuniform: base {_UNIFORM_CELL * 1e6:g} um, a '
                     f'box refined to {_UNIFORM_FINE * 1e6:g} um (x, y and z graded; the check raises unless every '
                     f'axis spans a width ratio >= {_GRADED_MIN_RATIO:g}); external_field along x, y, z and oblique '
                     f"({', '.join(f'{name} {field}' for name, field in _UNIFORM_FIELDS.items())} T). The held outer "
                     'ring takes u = -B . r at its faces toward the free cells, so u is linear and B = B_ext, H = '
                     'B_ext / mu0, phi_m = -B_ext . r / mu0 are exact in every free cell (the ring excluded) on any '
                     f'widths. measured = largest deviation relative to |B_ext| (phi_m: to max |phi_m|), tolerance '
                     f'{_TOLERANCE:g}, float64 CG 1e-12. Not run yet.'),
        specs=_uniform_runs, check=_check_uniform),
    SelfTestCase(
        name='magnet3d_layered_slab',
        label='Magnetostatics 3D: permeable slab normal and tangential to the field, along x, y, z',
        description=(f'magnetostatics_3d on {_SLAB_CELLS[0]} x {_SLAB_CELLS[1]} x {_SLAB_CELLS[2]} uniform '
                     f'{_SLAB_CELL * 1e6:g} um cells: a slab of mu_dc {_SLAB_PERMEABILITY:g} across the box, normal '
                     f'to x, y or z over {_SLAB_RANGE[0]:g} .. {_SLAB_RANGE[1]:g} of that length (cell edges). The '
                     'first cell layer along the drive axis is ideal_iron at Omega = '
                     f'{_HELD_POTENTIAL:g} A, the drive axis hi face a symmetry face (u = 0), the other faces '
                     'insulation. Runs normal_<axis> (drive along the slab normal: B uniform) and tangential_<axis> '
                     '(drive along the next axis: H uniform, B = mu B_0 in the slab). Exact: per row the series '
                     'half-cell resistances w / mu between the held face and the symmetry face (B = mu0 Omega_0 / '
                     'sum, phi_m from the partial sums, B, H = 0 in the held layer). measured = largest deviation of '
                     'B and H per component (relative to max B_d / max H_d) and of phi_m (relative to Omega_0), '
                     f'tolerance {_TOLERANCE:g}. Pins c = mu_dc per axis (no xx / yy swap), the ideal-iron held '
                     'value mu0 Omega, the symmetry face on every axis and B from the face fluxes. Not run yet.'),
        specs=_slab_runs, check=_check_slabs),
    SelfTestCase(
        name='magnet3d_saturating_slab',
        label='Magnetostatics 3D: saturating slab (J_s curve and B-H table), normal and tangential',
        description=(f'The magnet3d_layered_slab layout (slab normal to x) with the 2D test iron in the slab: the '
                     f'Frohlich-Kennelly curve mu_dc {_IRON_PERMEABILITY:g}, J_s {_IRON_SATURATION:g} T (curve_*) or '
                     'its bh_curve table sampled to 1500 A/m (table_*); ideal_iron layer at Omega = '
                     f'{_SATURATING_POTENTIAL:g} A. normal: drive along x, the air dominates the series reluctance '
                     '(nearly flux-imposed: the H-based proposal contracts by ~(1 - s), the Anderson loop does the '
                     'work), B ~ 0.59 T inside the table rows; tangential: drive along y, H = Omega_0 / L in every row, '
                     'the iron near 2 T (the table on its mu0 extension), exact from the first proposal. Exact: per '
                     'row one face flux B with sum of w H(B) = Omega_0 (bisection on B, the curve inverted by '
                     'bisection), every iron cell on its curve at that B. measured = largest deviation of B, H per '
                     'component, phi_m (as magnet3d_layered_slab) and of mu_eff / (B / (mu0 H)) - 1 in the free iron '
                     f'cells; tolerance {_TOLERANCE:g}; loop tolerance {_SATURATION_SETTINGS["nonlinear_tolerance"]:g} '
                     '(default Anderson). Pins the 3D saturation loop, the H-based J_s and table proposals, add_bh_curve '
                     'on 3D region masks and mu_eff. Not run yet.'),
        specs=_saturating_runs, check=_check_saturating),
    SelfTestCase(
        name='magnet3d_permeable_sphere',
        label='Magnetostatics 3D: permeable sphere in a uniform field',
        description=(f'magnetostatics_3d: a sphere of radius {_SPHERE_RADIUS * 1e3:g} mm and mu_dc '
                     f'{_SPHERE_PERMEABILITY:g} at the centre of a {_SPHERE_BOX * 1e3:g} mm vacuum cube in B_z = '
                     f'{_SPHERE_FIELD:g} T (external_field ring), mesh nonuniform refined on the sphere to a / '
                     f"{_SPHERE_RESOLUTIONS['coarse']} (coarse) and a / {_SPHERE_RESOLUTIONS['fine']} (fine), "
                     'graded by 1.3 to a / 2; exact pcg node with the multigrid preconditioner. Closed form B_in = 3 '
                     'mu_r / (mu_r + 2) B_0, H_in = B_in / (mu0 mu_r), compared with the derived means of B_z and H_z '
                     'over the central half radius. At mu_r >> 1 the staircase raises the core field to first order '
                     'in h (~6 % coarse, ~3 % fine), so measured = the largest relative deviation of the first-order '
                     'Richardson values (2 fine - coarse), inf unless the fine run beats the coarse one; tolerance '
                     f'{_SPHERE_EXTRAPOLATED_TOLERANCE:g} (far ring error ~ (1/6)^3, staircase scatter).'),
        specs=_permeable_sphere_runs, check=_check_permeable_sphere, heavy=True),
    SelfTestCase(
        name='magnet3d_magnetized_sphere',
        label='Magnetostatics 3D: uniformly magnetised sphere',
        description=(f'magnetostatics_3d: the magnet3d_permeable_sphere layout with remanence_z = '
                     f'{_SPHERE_REMANENCE:g} T and mu_dc 1 in the sphere, no external field and every face '
                     'insulation, so the type holds the corner cell (the all-Neumann anchor). Closed form B_in = 2 '
                     'B_r / 3, H_in = -B_r / (3 mu0) against the derived core means of B_z and H_z; measured = the '
                     'fine run\'s largest relative deviation (no contrast: no first-order staircase term), inf unless '
                     f'below the coarse run\'s; tolerance {_SPHERE_TOLERANCE:g}. Pins the remanence face drives in 3D '
                     'and B, H from the face fluxes inside a magnet.'),
        specs=_magnetized_sphere_runs, check=_check_magnetized_sphere, heavy=True),
    SelfTestCase(
        name='magnet3d_mirror_parity',
        label='Magnetostatics 3D: half models on symmetry and insulation planes',
        description=(f'magnetostatics_3d on uniform {_PARITY_CELL * 1e6:g} um cells: a centred cube magnet '
                     f'(remanence_z {_PARITY_REMANENCE:g} T, mu_dc {_PARITY_PERMEABILITY:g}) in a '
                     f'{_PARITY_CELLS[0]} x {_PARITY_CELLS[1]} x {_PARITY_CELLS[2]} cell box with symmetry x faces. '
                     'half_z: the lower half along z with a symmetry z_hi face (phi_m odd, B normal to the '
                     'midplane); half_y: the lower half along y with magnetic_insulation on y_hi (phi_m even, B '
                     'tangential). The full model restricted to each half is the discrete solution of the half '
                     '(face conductances and remanence face terms coincide on the plane). measured = largest '
                     'deviation of phi_m, B and H per component relative to the full model\'s maximum of its group, '
                     f'tolerance {_TOLERANCE:g}. Pins the face kinds behind core.mirror.FACE_FEATURE_SIGNS; the '
                     'unfold of a model mirror plane itself is not exercised. Not run yet.'),
        specs=_parity_runs, check=_check_parity),
    SelfTestCase(
        name='magnet3d_cpu_gpu',
        label='Magnetostatics 3D: CPU backend vs GPU',
        description=('magnetostatics_3d on solver_defaults.backend cpu with a sparse_direct linear solver node (host '
                     'SuperLU of the 3D window in nested-dissection order; Automatic there is multigrid pcg): '
                     'magnet3d_uniform_field (oblique) and magnet3d_layered_slab '
                     '(normal_z, tangential_x) against their exact solutions, magnet3d_coil_energy_identity (W = '
                     'Lambda I / 2), and uniform_oblique / slab_normal_z / coil_energy also on the gpu (exact pcg '
                     'node): max |B_abs| difference relative to the gpu maximum. The check raises unless each cpu run '
                     'recorded a cpu sparse_direct linear solver. Without a usable GPU the gpu runs are skipped. '
                     f'tolerance {_TOLERANCE:g}. Not run yet.'),
        specs=_backend_runs, check=_check_backends),
    SelfTestCase(
        name='magnet3d_solenoid_axis',
        label='Magnetostatics 3D: thick solenoid axis field and Ampere total',
        description=(f'magnetostatics_3d: a thick solenoid around z (r1 {_SOLENOID_RADII[0] * 1e3:g} mm, r2 '
                     f'{_SOLENOID_RADII[1] * 1e3:g} mm, length {2e3 * _SOLENOID_HALF_LENGTH:g} mm, {_SOLENOID_TURNS} '
                     f'turns x {_SOLENOID_CURRENT:g} A; a circle-prism difference) at the centre of a '
                     f'{_SOLENOID_BOX * 1e3:g} mm cube with every face insulation (the corner anchor), '
                     f'{_SOLENOID_CELL * 1e6:g} um bands graded by 1.3 to 0.4 mm, multigrid pcg. B_z on the axis '
                     '(bilinear in x, y) against the uniform-J closed form mu0 J / 2 [G(b - z) + G(b + z)], G(s) = s '
                     f'ln((r2 + sqrt(r2^2 + s^2)) / (r1 + sqrt(r1^2 + s^2))), over |z| <= {_SOLENOID_SAMPLED:g} b, '
                     f"relative to the centre value (tolerance {_SOLENOID_TOLERANCES['axis_B_z']:g}: staircase ring, "
                     'insulated box); Ampere: J_y summed over the half plane y = y_c, x > x_c (uniform cells) against '
                     f"turns * current (tolerance {_SOLENOID_TOLERANCES['ampere']:g}). error = the largest error over "
                     'its tolerance; tolerance 1. Pins the coil T (hole fill, distance ramp, axial length), the face '
                     'drive of T and J = curl T. Not run yet.'),
        specs=lambda: {'run': _solenoid_spec()}, check=_check_solenoid),
    SelfTestCase(
        name='magnet3d_coil_energy_identity',
        label='Magnetostatics 3D: coil energy identity W = Lambda I / 2',
        description=(f'magnetostatics_3d on {_IDENTITY_CELLS[0]} x {_IDENTITY_CELLS[1]} x {_IDENTITY_CELLS[2]} '
                     f'uniform {_IDENTITY_CELL * 1e6:g} um cells: a square coil around x (box difference, '
                     f'{_IDENTITY_TURNS} turns x {_IDENTITY_CURRENT:g} A) with a core of mu_dc '
                     f"{_IDENTITY_PERMEABILITY['core']:g} through its hole and an off-axis block of mu_dc "
                     f"{_IDENTITY_PERMEABILITY['block']:g}, every face insulation (the corner anchor, so no held "
                     'boundary term). The discrete energy (sum of face flows^2 / (2 mu0 conductance)) equals Lambda I '
                     '/ 2 of the face-flow linkage exactly, and L = Lambda / I (relative, tolerance '
                     f"{_IDENTITY_TOLERANCES['identity']:g}, float64 scalars); Ampere: J_z over the half plane z = "
                     f"z_c, y > y_c against turns * current (tolerance {_IDENTITY_TOLERANCES['ampere']:g}). error = "
                     'the largest error over its tolerance; tolerance 1. Not run yet.'),
        specs=lambda: {'run': _coil_energy_spec()}, check=_check_coil_energy),
    SelfTestCase(
        name='magnet3d_vs_axisymmetric',
        label='Magnetostatics 3D: cored coil against 2D axisymmetric magnetostatics',
        description=(f'A coil around z (r {_AXI_COIL_RADII[0] * 1e3:g} - {_AXI_COIL_RADII[1] * 1e3:g} mm, length '
                     f'{2e3 * _AXI_COIL_HALF_LENGTH:g} mm, {_AXI_TURNS} turns x {_AXI_CURRENT:g} A) on a cylindrical '
                     f'core (r {_AXI_CORE[0] * 1e3:g} mm, length {2e3 * _AXI_CORE[1]:g} mm, mu_dc '
                     f'{_AXI_PERMEABILITY:g}), insulation faces: magnetostatics_3d in a {2e3 * _AXI_RADIUS:g} x '
                     f'{2e3 * _AXI_RADIUS:g} x {_AXI_HEIGHT * 1e3:g} mm box ({_AXI_CELLS["3d"] * 1e6:g} um bands, '
                     'graded by 1.3) against axisymmetric magnetostatics (r up to '
                     f'{_AXI_RADIUS * 1e3:g} mm, uniform {_AXI_CELLS["2d"] * 1e6:g} um cells): the coil inductance '
                     f'and the mean axial B over a sensor of radius {_AXI_SENSOR[0] * 1e6:g} um, '
                     f'{_AXI_SENSOR[1] * 1e6:g} um below the core tip; multigrid pcg. error = the largest relative '
                     f'difference over {_AXI_TOLERANCE:g} (staircased core and winding, box vs cylinder outer face; '
                     'first estimate); tolerance 1. Not run yet.'),
        specs=lambda: {'3d': _axi_3d_spec(), '2d': _axi_2d_spec()}, check=_check_vs_axisymmetric, heavy=True),
)


# ---- 'needle_3d' template ----
NEEDLE_3D_LABEL = 'Flux concentrator needle (3D)'
NEEDLE_3D_DESCRIPTION = (
    'Magnetostatics (3D, stationary): a 15 mm x 125 um x 25 um Metglas needle (mu_dc 1e5) along z in a uniform axial '
    'field B_ext (external_field, 10 x 10 x 35 mm window held at the applied potential), the geometry of Carey et al. '
    '2026 on coarse 25 um cells around the needle and its lower tip (graded by 1.3 to 1 mm, by 1.5 to 2 mm outside). '
    'Derived values: gain_h50 = mean |B| / B_ext over a sensor box 50 um below the tip and EF_h50 = 20 log10(gain) '
    'against the prolate-spheroid estimate (Eq. 2 with m = l / sqrt(w t)); line curve B_axis: |B| on the needle axis '
    'below the tip against Eq. 2. The converged, paper-matched run is experiments/Ben_flux/needle_3d_scalar_potential.')


def needle_3d_spec():
    """'needle_3d' template: the rectangular needle in an axial external field, graded bands at 25 um."""
    parameters = [
        ParameterSpec('B_ext', 2.1e-5, 'applied axial flux density [T]'),
        ParameterSpec('mu_needle', 1e5, 'relative permeability of the needle'),
        ParameterSpec('l_needle', '15*mm', 'needle length (z)'),
        ParameterSpec('w_needle', '125*um', 'needle width (x)'),
        ParameterSpec('t_needle', '25*um', 'needle thickness (y)'),
        ParameterSpec('Lx', '10*mm', 'window size x'),
        ParameterSpec('Ly', '10*mm', 'window size y'),
        ParameterSpec('Lz', '35*mm', 'window size z (along the field)'),
        ParameterSpec('x_c', 'Lx/2', 'needle axis x'),
        ParameterSpec('y_c', 'Ly/2', 'needle axis y'),
        ParameterSpec('z_tip', '10*mm', 'z of the lower needle tip'),
        ParameterSpec('z_top', 'z_tip + l_needle', 'z of the upper needle tip'),
        ParameterSpec('h_sensor', '50*um', 'sensor height below the tip'),
        ParameterSpec('m_aspect', 'l_needle/sqrt(w_needle*t_needle)', 'needle aspect ratio m = l / sqrt(w t)'),
        ParameterSpec('s_eq2', 'l_needle/m_aspect', 'decay length of Eq. 2, sqrt(w t)'),
        ParameterSpec('N_demag', '(m_aspect/sqrt(m_aspect**2 - 1)*log(m_aspect + sqrt(m_aspect**2 - 1)) - 1)'
                                 '/(m_aspect**2 - 1)', 'demagnetization factor of a prolate spheroid of aspect m'),
        ParameterSpec('G_tip', 'mu_needle/(N_demag*(mu_needle - 1) + 1)/m_aspect', 'Eq. 1-2 tip gain B0 / (m B_ext)'),
    ]
    eq2 = '(G_tip + (1 - G_tip)*({h})/sqrt(({h})**2 + s_eq2**2))'
    return ModelSpec(
        title='Flux concentrator needle in 3D',
        parameters=parameters,
        domain=DomainSpec(size_x='Lx', size_y='Ly', size_z='Lz'),
        geometry=[
            GeometrySpec('needle', 'box', {'x0': 'x_c - w_needle/2', 'y0': 'y_c - t_needle/2', 'z0': 'z_tip',
                                           'x1': 'x_c + w_needle/2', 'y1': 'y_c + t_needle/2', 'z1': 'z_top'}),
            GeometrySpec('sensor', 'box', {'x0': 'x_c - 68*um', 'y0': 'y_c - 43*um', 'z0': 'z_tip - h_sensor - 12.5*um',
                                           'x1': 'x_c + 68*um', 'y1': 'y_c + 43*um',
                                           'z1': 'z_tip - h_sensor + 12.5*um'})],
        materials=[MaterialSpec('air', properties={'mu_dc': 1.0}, selection=['all']),
                   MaterialSpec('metglas', properties={'mu_dc': 'mu_needle'}, selection=['needle'])],
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_3D_TYPE_NAME, features=[
            FeatureSpec('applied', 'external_field', ['all'], {'B_x': 0.0, 'B_y': 0.0, 'B_z': 'B_ext'})])],
        mesh=MeshSpec(mode='nonuniform', dx='1*mm', max_ratio=1.3, outer_dx='2*mm', outer_ratio=1.5, axes=[
            MeshAxisSpec('x', bands=[MeshBandSpec('x_c - 0.1*mm', 'x_c + 0.1*mm', '25*um')]),
            MeshAxisSpec('y', bands=[MeshBandSpec('y_c - 0.05*mm', 'y_c + 0.05*mm', '25*um')]),
            MeshAxisSpec('z', bands=[MeshBandSpec('z_tip - 0.3*mm', 'z_tip + 50*um', '25*um'),
                                     MeshBandSpec('z_tip', 'z_top', '0.5*mm'),
                                     MeshBandSpec('z_top - 50*um', 'z_top + 50*um', '50*um')])]),
        study=StudySpec(type='stationary', stored_fields=[f'{_MAGNETIC}.B_abs', f'{_MAGNETIC}.phi_m'],
                        stored_cuts=[['y', 'y_c']]),
        derived_values=[
            DerivedValueSpec('gain_h50', f'{_MAGNETIC}_B_abs/B_ext', ['sensor'], 'mean'),
            DerivedValueSpec('EF_h50', '20*log10(gain_h50)', [], 'mean',
                             reference='20*log10' + eq2.format(h='h_sensor'))],
        analytic_curves=[AnalyticCurveSpec(
            'B_axis', 'line', 'B_ext*' + eq2.format(h='z_tip - z'),
            line=['x_c + 1*um', 'y_c + 1*um', 'z_tip - 0.3*mm', 'x_c + 1*um', 'y_c + 1*um', 'z_tip - 12.5*um'],
            field=f'{_MAGNETIC}.B_abs')])

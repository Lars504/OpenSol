"""Self-tests of solid_membrane_tension (new_modules_plan S9): a uniformly stressed plane-stress film whose exact
tension sets the anisotropic modes of the clamped membrane rectangle (plus the coupling's validation refusals), the
Mauranyapin M-E1 perforated cell <sigma> / sigma_0 = 0.7186 and its Bloch band against c_eff = sqrt(<N> / <rho_s>);
eigenfrequency studies with solved_physics ['membrane']."""
import math

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import float32_value, scaled_worst_result
from ..model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                          MeshSpec, ModelSpec, PhysicsSpec, SolverDefaultsSpec, StudySpec)

TENSION_COUPLING = 'solid_membrane_tension'   # = definitions_membrane.TENSION_COUPLING_NAME (not imported)
_SOLID, _SOLID_TYPE = 'solid', 'solid_mechanics'
_MEMBRANE, _MEMBRANE_TYPE = 'membrane', 'membrane'
_COUPLING = 'film_tension'
_TABLE = 'modes'
_FREQUENCY_COLUMN = 1   # resonance_table rows: number, frequency, frequency_imag, q_factor, pml_fraction
_FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')


def _film_model(title, size, mesh_h, materials, solid_features, solid_settings, coupling_params, n_modes,
                geometry=()):
    """Plane-stress solid + clamped membrane on one uniform triangle mesh (cpu), the tension coupling, an
    eigenfrequency study solving the membrane only, and the window integrals of the membrane tension."""
    membrane_features = [FeatureSpec('frame', 'clamped', list(_FACES)), FeatureSpec(_TABLE, 'resonance_table')]
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]), geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings={'out_of_plane': 'plane_stress', **solid_settings},
                             features=list(solid_features)),
                 PhysicsSpec(_MEMBRANE, _MEMBRANE_TYPE, features=membrane_features)],
        couplings=[CouplingSpec(_COUPLING, TENSION_COUPLING, _SOLID, _MEMBRANE, params=dict(coupling_params))],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=mesh_h)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': n_modes}, solved_physics=[_MEMBRANE]),
        derived_values=[DerivedValueSpec('N_xx_integral', f'{_MEMBRANE}_tension_xx', [], 'integral'),
                        DerivedValueSpec('N_yy_integral', f'{_MEMBRANE}_tension_yy', [], 'integral')])


def _derived(results, name):
    return float(results.summary()['derived'][name])   # float() also reads 'nan'


def _table_frequencies(results):
    """Sorted resonance_table frequencies [Hz] of the membrane."""
    return sorted(float(row[_FREQUENCY_COLUMN]) for row in results.result(_MEMBRANE, _TABLE)['rows'])


# ---- membrane_film_tension_uniform: an a x b plane-stress film (E, nu, h = 50 nm as the solid thickness setting,
# coupling thickness empty) with an equibiaxial initial stress sigma0, rollers on x_lo / x_hi / y_lo and a traction S on
# y_hi: the exact uniform state eps_xx = 0, sigma_yy = S, sigma_xx = sigma0 + nu (S - sigma0), sigma_xy = 0 (linear u,
# exact in P2). The coupling gives N_xx = h sigma_xx, N_yy = h S, and the clamped rectangle has f_mn = (1 / 2) sqrt((N_xx
# (m / a)^2 + N_yy (n / b)^2) / rho_s) (P2 at h = a / 80: the accuracy of the membrane template rectangle). Validation: a
# plane-strain solid, no thickness, and solved_physics [] are refused at the coupling.
_STRIP = (1.0e-3, 0.7e-3)
_STRIP_MESH_H = _STRIP[0] / 80.0
_STRIP_FILM = 50e-9            # m (solid thickness setting)
_STRIP_PRESTRESS = 1e9         # Pa, equibiaxial initial stress (float32-exact)
_STRIP_TRACTION = 4e8          # Pa on y_hi (float32-exact)
_STRIP_ELASTIC = (250e9, 0.25)
_STRIP_AREAL_DENSITY = 1.55e-4  # kg/m^2
_STRIP_MODES = 4
_STRIP_FREQUENCY_TOLERANCE = 1e-5   # relative (first estimate: membrane_emfdfd_rectangle's tolerance)
_STRIP_TENSION_TOLERANCE = 1e-6     # relative mean tension (float32 stress and slot storage)


def _strip_spec():
    youngs_modulus, poisson_ratio = _STRIP_ELASTIC
    film = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'initial_stress_xx': _STRIP_PRESTRESS,
            'initial_stress_yy': _STRIP_PRESTRESS, 'areal_density': _STRIP_AREAL_DENSITY}
    features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi']), FeatureSpec('bottom', 'roller', ['y_lo']),
                FeatureSpec('pull', 'boundary_load', ['y_hi'], {'t_x': 0.0, 't_y': _STRIP_TRACTION})]
    return _film_model('Self-test: uniform film tension -> anisotropic membrane rectangle modes', _STRIP,
                       _STRIP_MESH_H, [MaterialSpec('film', properties=film, selection=['all'])], features,
                       {'thickness': _STRIP_FILM}, {}, _STRIP_MODES)


def _strip_tensions():
    """(N_xx, N_yy) [N/m] of the exact uniform state (stored float32 constants)."""
    poisson_ratio = float32_value(_STRIP_ELASTIC[1])
    stress_xx = _STRIP_PRESTRESS + poisson_ratio * (_STRIP_TRACTION - _STRIP_PRESTRESS)
    return _STRIP_FILM * stress_xx, _STRIP_FILM * _STRIP_TRACTION


def _strip_frequencies():
    tension_xx, tension_yy = _strip_tensions()
    size_x, size_y = _STRIP
    density = float32_value(_STRIP_AREAL_DENSITY)
    values = [0.5 * math.sqrt((tension_xx * (m / size_x) ** 2 + tension_yy * (n / size_y) ** 2) / density)
              for m in range(1, 9) for n in range(1, 9)]
    return sorted(values)[:_STRIP_MODES]


def _validation_refusals():
    """{variant: True when validation reports an error at the coupling}: plane-strain solid, no thickness, and an
    empty solved_physics (the study would solve the solid itself)."""
    from ..model.validate import validate
    path = ('coupling', _COUPLING)

    def plane_strain(spec):
        spec.physics[0].settings['out_of_plane'] = 'plane_strain'

    def no_thickness(spec):
        spec.physics[0].settings.pop('thickness')

    def solid_solved(spec):
        spec.study.solved_physics = []

    refused = {}
    for name, change in (('plane_strain', plane_strain), ('no_thickness', no_thickness),
                         ('solid_solved', solid_solved)):
        spec = _strip_spec()
        change(spec)
        refused[name] = any(issue.level == 'error' and tuple(issue.path) == path for issue in validate(spec))
    return refused


def _check_strip(results_by_name):
    """Membrane table vs the anisotropic continuum, mean tensions vs the exact state, one snapshot per mode, and the
    validation refusals."""
    results = results_by_name['strip']
    errors, tolerances, details = {}, {}, {}
    expected = _strip_frequencies()
    measured = _table_frequencies(results)
    errors['frequency'] = (max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))
                           if len(measured) >= len(expected) else math.inf)
    tolerances['frequency'] = _STRIP_FREQUENCY_TOLERANCE
    area = _STRIP[0] * _STRIP[1]
    tension_xx, tension_yy = _strip_tensions()
    mean_xx, mean_yy = _derived(results, 'N_xx_integral') / area, _derived(results, 'N_yy_integral') / area
    errors['N_xx'], errors['N_yy'] = abs(mean_xx / tension_xx - 1.0), abs(mean_yy / tension_yy - 1.0)
    tolerances['N_xx'] = tolerances['N_yy'] = _STRIP_TENSION_TOLERANCE
    labels = results.summary().get('snapshot_labels') or []
    errors['snapshots'], tolerances['snapshots'] = float(abs(len(labels) - _STRIP_MODES)), 0.5
    refused = _validation_refusals()
    errors['validation'] = 0.0 if all(refused.values()) else math.inf
    tolerances['validation'] = 1.0
    details.update(frequency_measured=measured, frequency_expected=expected, mean_tension_measured=[mean_xx, mean_yy],
                   mean_tension_expected=[tension_xx, tension_yy], snapshot_labels=labels, refused=refused)
    return scaled_worst_result(errors, tolerances, **details)


# ---- the perforated cell (Mauranyapin plan E1(a) through the coupling; checked by the Bloch band case): a p = 3 um
# cell with a centred a = 1 um square hole (phi = 1/9, sharp corners), 80 nm Si3N4 (E 250 GPa, nu 0.22) with sigma0 = 1 GPa
# equibiaxial initial stress, plane stress, rollers on all four faces (= periodicity at zero macroscopic strain for this
# mirror-symmetric cell), the hole void for the solid and a hole for the membrane (areal_density 0). The stationary
# background solve relaxes the stress around the hole; <sigma_ii> = (1 / p^2) int N_ii dA / h over the membrane's
# coupled tension (holes 0). Reference 0.7186 (Q4 N = 48 / 96 / 192 Richardson, order ~1.07 from the re-entrant
# corners); P2 triangles at h = p / 60 over-estimate it by a few 1e-3 at most (first estimate), tolerance 1 %.
_CELL_PITCH = 3e-6
_CELL_HOLE = 1e-6
_CELL_FILM = 80e-9              # m (coupling thickness param; solid thickness setting empty)
_CELL_PRESTRESS = 1e9           # Pa
_CELL_ELASTIC = (250e9, 0.22)
_CELL_DENSITY = 3200.0          # kg/m^3
_CELL_MESH_H = _CELL_PITCH / 60.0
_CELL_REFERENCE = 0.7186        # <sigma> / sigma0, sharp corners (mauranyapin2021 plan, Section 6)
_CELL_TOLERANCE = 1e-2


def _cell_parts():
    """(materials, geometry) of the perforated cell: the prestressed film everywhere, the centred hole void with
    areal_density 0 (a membrane hole)."""
    youngs_modulus, poisson_ratio = _CELL_ELASTIC
    low, high = 0.5 * (_CELL_PITCH - _CELL_HOLE), 0.5 * (_CELL_PITCH + _CELL_HOLE)
    film = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'initial_stress_xx': _CELL_PRESTRESS,
            'initial_stress_yy': _CELL_PRESTRESS, 'areal_density': _CELL_DENSITY * _CELL_FILM}
    materials = [MaterialSpec('film', properties=film, selection=['all']),
                 MaterialSpec('hole_void', properties={'youngs_modulus': 0.0, 'areal_density': 0.0},
                              selection=['hole'])]
    return materials, [GeometrySpec('hole', 'rectangle', {'x0': low, 'y0': low, 'x1': high, 'y1': high})]


# ---- membrane_film_tension_cell_bloch_band (Mauranyapin plan E1(b), r_c = 0): the perforated cell above, the relaxed
# solid as background, then the membrane Bloch cell (periodic_x / _y, no clamp, no bending) at |k| p = 0.1 and 0.2 along
# x and along the diagonal. Long-wave homogenisation (plan Section 4: div N = 0 with traction-free holes, so the corrector
# is constant) gives omega = c_eff |k| with c_eff^2 = n.<N>.n / <rho_s> (cell averages, holes 0; <rho_s> = rho h (1 -
# phi)); the lowest band's phase velocity 2 pi f / |k| against c_eff from the same run's tension / density integrals
# within 0.3 % (plan: (a) <-> (b)), axis vs diagonal within 0.3 %, and c_eff at the axis' smallest k vs the plan's
# 502.6 m/s within 0.5 % (half the static 1 % on <sigma>).
# Solid periodicity: the 2D solid_mechanics has no periodic / Bloch setting (only solid_mechanics_3d on tetrahedra has),
# so the solid keeps the rollers on all four faces of the static case. The mesh is meshed mirror_x / mirror_y: the mirror
# image of each x_lo node is the x_hi node at the same y (the matching side nodes the membrane's periodic axes need), and
# the discrete roller solution is the mirror-symmetric one, u_n = 0 and zero shear traction on the faces, i.e. the
# periodic solution at zero mean strain. Its opposite-face roller reactions are equal and opposite, so they cancel
# against every periodic membrane test function: the discrete equilibrium the homogenised c_eff rests on holds. A plain
# (non-mirrored) mesh would make the roller state differ from the periodic one at the discretisation level.
# Residual errors: the (k p)^2 dispersion of the band (the k^2 extrapolation of the two k is in details) and the element
# mean N (P0 coefficient) against the linear P2 element stress (quadratic in the small mismatch). Tolerances first
# estimates.
_BAND_KP = (0.1, 0.2)            # |k| p
_BAND_DIRECTIONS = {'x': (1.0, 0.0), 'diagonal': (math.sqrt(0.5), math.sqrt(0.5))}
_BAND_MODES = 2
_BAND_TOLERANCE = 3e-3           # band phase velocity vs c_eff of the same run, and axis vs diagonal
_BAND_REFERENCE_TOLERANCE = 5e-3  # c_eff vs the plan's extrapolated 502.6 m/s
_BAND_REFERENCE_SPEED = 502.6    # m/s (mauranyapin2021 plan E1 target; also recomputed from 0.7186 in the details)


def _band_run_name(direction, index):
    return f'{direction}_{index}'


def _band_spec(direction, index):
    """The perforated cell (mirror_x / mirror_y mesh) with the membrane Bloch-periodic along x and y at |k| =
    _BAND_KP[index] / p along the direction; derived integrals of the membrane tension components and areal density."""
    unit_x, unit_y = _BAND_DIRECTIONS[direction]
    wavenumber = _BAND_KP[index] / _CELL_PITCH
    materials, geometry = _cell_parts()
    membrane_settings = {'periodic_x': True, 'periodic_y': True, 'bloch_kx': wavenumber * unit_x,
                         'bloch_ky': wavenumber * unit_y}
    return ModelSpec(
        title=f'Self-test: perforated film cell Bloch band ({direction}, k p = {_BAND_KP[index]:g})',
        domain=DomainSpec(size_x=_CELL_PITCH, size_y=_CELL_PITCH), geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings={'out_of_plane': 'plane_stress'},
                             features=[FeatureSpec('rollers', 'roller', list(_FACES))]),
                 PhysicsSpec(_MEMBRANE, _MEMBRANE_TYPE, settings=membrane_settings,
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        couplings=[CouplingSpec(_COUPLING, TENSION_COUPLING, _SOLID, _MEMBRANE, params={'thickness': _CELL_FILM})],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_CELL_MESH_H, mirror_x=True,
                         mirror_y=True)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _BAND_MODES}, solved_physics=[_MEMBRANE]),
        derived_values=[DerivedValueSpec('N_xx_integral', f'{_MEMBRANE}_tension_xx', [], 'integral'),
                        DerivedValueSpec('N_yy_integral', f'{_MEMBRANE}_tension_yy', [], 'integral'),
                        DerivedValueSpec('N_xy_integral', f'{_MEMBRANE}_tension_xy', [], 'integral'),
                        DerivedValueSpec('rho_s_integral', f'{_MEMBRANE}_areal_density', [], 'integral')])


def _band_specs():
    return {_band_run_name(direction, index): _band_spec(direction, index)
            for direction in _BAND_DIRECTIONS for index in range(len(_BAND_KP))}


def _homogenised_speed(results, direction):
    """c_eff [m/s] = sqrt(n.<N>.n / <rho_s>) along the direction from the run's cell integrals (NaN when not positive)."""
    unit_x, unit_y = _BAND_DIRECTIONS[direction]
    tension = (unit_x * unit_x * _derived(results, 'N_xx_integral')
               + unit_y * unit_y * _derived(results, 'N_yy_integral')
               + 2.0 * unit_x * unit_y * _derived(results, 'N_xy_integral'))
    density = _derived(results, 'rho_s_integral')
    return math.sqrt(tension / density) if tension > 0.0 and density > 0.0 else math.nan


def _check_band(results_by_name):
    """Lowest band phase velocity per run vs that run's c_eff, axis vs diagonal at the smallest k, and c_eff vs the
    plan's 502.6 m/s."""
    errors, tolerances, details = {}, {}, {}
    band_speeds, homogenised_speeds = {}, {}
    for direction in _BAND_DIRECTIONS:
        for index, kp in enumerate(_BAND_KP):
            name = _band_run_name(direction, index)
            results = results_by_name[name]
            frequencies = _table_frequencies(results)
            wavenumber = kp / _CELL_PITCH
            band_speeds[name] = 2.0 * math.pi * frequencies[0] / wavenumber if frequencies else math.nan
            homogenised_speeds[name] = _homogenised_speed(results, direction)
            errors[name] = abs(band_speeds[name] / homogenised_speeds[name] - 1.0)
            tolerances[name] = _BAND_TOLERANCE
            details[name] = {'band_frequencies_hz': frequencies, 'band_speed': band_speeds[name],
                             'c_eff': homogenised_speeds[name],
                             'mean_tension': [_derived(results, f'N_{part}_integral') / _CELL_PITCH ** 2
                                              for part in ('xx', 'yy', 'xy')],
                             'mean_areal_density': _derived(results, 'rho_s_integral') / _CELL_PITCH ** 2}
    axis, diagonal = _band_run_name('x', 0), _band_run_name('diagonal', 0)
    errors['axis_vs_diagonal'] = abs(band_speeds[diagonal] / band_speeds[axis] - 1.0)
    tolerances['axis_vs_diagonal'] = _BAND_TOLERANCE
    errors['c_eff_reference'] = abs(homogenised_speeds[axis] / _BAND_REFERENCE_SPEED - 1.0)
    tolerances['c_eff_reference'] = _BAND_REFERENCE_TOLERANCE
    scale = _CELL_FILM * _CELL_PRESTRESS * _CELL_PITCH * _CELL_PITCH   # <N_ii> / (h sigma0) of the relaxed cell
    for part in ('xx', 'yy'):
        ratio = _derived(results_by_name[axis], f'N_{part}_integral') / scale
        errors[f'ratio_{part}'], tolerances[f'ratio_{part}'] = abs(ratio / _CELL_REFERENCE - 1.0), _CELL_TOLERANCE
        details[f'ratio_{part}'] = ratio
    k_low, k_high = _BAND_KP[0] ** 2, _BAND_KP[-1] ** 2   # c^2(k) = c0^2 + b k^2: two-point k^2 extrapolation
    extrapolated = {}
    for direction in _BAND_DIRECTIONS:
        low = band_speeds[_band_run_name(direction, 0)]
        high = band_speeds[_band_run_name(direction, len(_BAND_KP) - 1)]
        square = (k_high * low * low - k_low * high * high) / (k_high - k_low)
        extrapolated[direction] = math.sqrt(square) if square > 0.0 else math.nan
    open_fraction = (_CELL_HOLE / _CELL_PITCH) ** 2
    details.update(kp=list(_BAND_KP), band_speed_k0_extrapolated=extrapolated, c_eff_plan=_BAND_REFERENCE_SPEED,
                   c_eff_from_reference_ratio=math.sqrt(_CELL_REFERENCE * _CELL_PRESTRESS
                                                        / (_CELL_DENSITY * (1.0 - open_fraction))),
                   mesh_h=_CELL_MESH_H)
    return scaled_worst_result(errors, tolerances, **details)


MEMBRANE_TENSION_CASES = (
    SelfTestCase(
        name='membrane_film_tension_uniform',
        label='Coupling: uniform plane-stress film tension -> anisotropic membrane rectangle modes',
        description=('A 1 x 0.7 mm plane-stress film (h 50 nm from the solid thickness setting, E 250 GPa, nu 0.25, '
                     f'initial stress {_STRIP_PRESTRESS:g} Pa equibiaxial; rollers x_lo / x_hi / y_lo, traction '
                     f'{_STRIP_TRACTION:g} Pa on y_hi) and a clamped membrane on the same P2 triangles (h = a / 80), '
                     "eigenfrequency study with solved_physics ['membrane']: the exact uniform state sigma_xx = sigma0 + "
                     'nu (S - sigma0), sigma_yy = S gives N = h sigma through solid_membrane_tension; the mean tensions '
                     f'(window integrals of tension_xx / _yy) within {_STRIP_TENSION_TOLERANCE:g}, the {_STRIP_MODES} '
                     'lowest table frequencies against f_mn = (1 / 2) sqrt((N_xx (m / a)^2 + N_yy (n / b)^2) / rho_s) '
                     f'within {_STRIP_FREQUENCY_TOLERANCE:g}, one snapshot per mode; validation errors at the coupling '
                     'for a plane-strain solid, no thickness and an empty solved_physics. measured = error in tolerance '
                     'units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {'strip': _strip_spec()}, check=_check_strip),
    SelfTestCase(
        name='membrane_film_tension_cell_bloch_band',
        label='Coupling: relaxed perforated film cell tension and Bloch band vs homogenised c_eff (Mauranyapin E1)',
        description=('A 3 um cell with a centred 1 um square hole (sharp corners), 80 nm Si3N4 film (E 250 GPa, nu '
                     '0.22, sigma0 = 1 GPa equibiaxial initial stress; hole: areal_density 0) on a mirror_x / mirror_y '
                     'P2 triangle mesh h = p / 60: the plane-stress solid '
                     '(no periodic setting in 2D, so mirror rollers on all faces: on the mirror-symmetric mesh the '
                     'discrete periodic state at zero mean strain) relaxes as background (solved_physics '
                     "['membrane']), solid_membrane_tension (thickness param 80 nm) writes N = h sigma: <N_ii> / (h "
                     'sigma0) = int tension_ii dA / (h sigma0 p^2) of the first run against the extrapolated Q4 '
                     f'reference {_CELL_REFERENCE:g} (mauranyapin2021 plan E1, Section 6) within {_CELL_TOLERANCE:g} '
                     'for xx and yy; and the membrane eigen solve runs on '
                     'the Bloch cell (periodic_x / _y, no clamp, no bending) at |k| p = '
                     f"{', '.join(f'{value:g}' for value in _BAND_KP)} along x and the diagonal: the lowest band "
                     'phase velocity 2 pi f / |k| against c_eff = sqrt(n.<N>.n / <rho_s>) from the same run\'s cell '
                     f'integrals within {_BAND_TOLERANCE:g}, axis vs diagonal within {_BAND_TOLERANCE:g}, and c_eff '
                     f'against the plan\'s {_BAND_REFERENCE_SPEED:g} m/s within {_BAND_REFERENCE_TOLERANCE:g} '
                     '(mauranyapin2021 plan E1). measured = error in tolerance units, tolerance 1 (first estimates, not '
                     'yet run).'),
        specs=_band_specs, check=_check_band, heavy=True),
)

"""Self-test of the membrane squeeze_film_damping feature (experiment plan F4): the clamped drum's (0, 1) mode over a
vented gas gap against the closed-form Reynolds pressure, full disc and a half model with a sealed symmetry face."""
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests import (_AREAL_DENSITY, _DRUM_H, _DRUM_RADIUS, _DRUM_ZEROS, _FILM, _MEMBRANE_TYPE, _PHYSICS, _TABLE,
                        _solver_defaults, _triangles)

_FILM_TYPE = 'squeeze_film_damping'   # = definition._SQUEEZE_FILM (selftests never import the definition)
_VISCOSITY = 1.81e-5            # Pa s
_AMBIENT_PRESSURE = 101325.0    # Pa
# incompressible gap 2 um; compressible 15 um: squeeze term 12 mu omega / (H^2 p_a) ~ 1.1 (j01 / R)^2 at f_01 = 435 kHz
_FILMS = {'film_incompressible': (2e-6, 0.0, 'open_edges'), 'film_vented_all': (2e-6, 0.0, 'all'),
          'film_compressible': (15e-6, _AMBIENT_PRESSURE, 'open_edges')}
_DRUM_TOLERANCE = 1e-3          # c, k, gamma, m_eff vs the closed form on P2 h = R / 20 (plan F4; first estimate)
_SAME_TOLERANCE = 1e-10         # open_edges = all on the drum (its rim is a hole outline), table totals = film sums


def _film_features(names):
    return [FeatureSpec(name, _FILM_TYPE, ['drum'], params={'gap': gap, 'viscosity': _VISCOSITY,
                                                             'ambient_pressure': ambient, 'slip': 'none',
                                                             'vents': vents})
            for name, (gap, ambient, vents) in _FILMS.items() if name in names]


def _drum_film_specs():
    """'full': the clamped drum (holes around it) with the three films; 'half': x <= centre, x_hi a symmetry face, the
    rim clamped through a frame (the window square minus the disc, whose outline is the arc only) and the two
    open_edges films. A point at the centre puts a mesh node at the mode's peak."""
    side = 2.2 * _DRUM_RADIUS
    centre = 0.5 * side
    disc = {'cx': centre, 'cy': centre, 'r': _DRUM_RADIUS}
    full = ModelSpec(
        title='Self-test: squeeze film under a clamped circular membrane',
        domain=DomainSpec(size_x=side, size_y=side),
        geometry=[GeometrySpec('drum', 'circle', dict(disc)),
                  GeometrySpec('centre', 'point', {'x': centre, 'y': centre})],
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['drum'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('rim', 'clamped', ['drum']), FeatureSpec(_TABLE, 'resonance_table'),
            *_film_features(_FILMS)])],
        meshes=[_triangles(_DRUM_H)], solver_defaults=_solver_defaults('cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}))
    half = ModelSpec(
        title='Self-test: squeeze film under a half clamped circular membrane (symmetry face sealed)',
        domain=DomainSpec(size_x=centre, size_y=side),
        geometry=[GeometrySpec('drum', 'circle', dict(disc)), GeometrySpec('cut', 'circle', dict(disc)),
                  GeometrySpec('square', 'rectangle', {'x0': -side, 'y0': -side, 'x1': 2.0 * side, 'y1': 2.0 * side}),
                  GeometrySpec('frame', 'difference', {'base': 'square', 'subtract': ['cut']}),
                  GeometrySpec('centre', 'point', {'x': centre, 'y': centre})],
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['drum'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('rim', 'clamped', ['frame']), FeatureSpec('mirror', 'symmetry', ['x_hi']),
            FeatureSpec(_TABLE, 'resonance_table'),
            *_film_features(('film_incompressible', 'film_compressible'))])],
        meshes=[_triangles(_DRUM_H)], solver_defaults=_solver_defaults('cpu'),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}))
    return {'full': full, 'half': half}


def _first_row(results, feature):
    """{column: value} of the first row of a table result."""
    table = results.result(_PHYSICS, feature)
    return dict(zip(table['columns'], table['rows'][0]))


def _closed_form(gap, ambient, frequency):
    """(c [kg/s], k [N/m], m_eff [kg]) of the full drum's (0, 1) mode, peak 1 m: p = -(12 mu / H^3) J0(k r) / (k^2 + j
    sigma~), sigma~ = 12 mu omega / (H^2 p_a), F = int p J0 dA = -(12 mu / H^3) pi a^2 J1(j01)^2 / (k^2 + j sigma~)."""
    from scipy.special import j1
    zero = _DRUM_ZEROS[0]
    omega = 2.0 * math.pi * frequency
    wavenumber_squared = (zero / _DRUM_RADIUS) ** 2
    squeeze = 0.0 if not ambient > 0.0 else 12.0 * _VISCOSITY * omega / (gap ** 2 * ambient)
    overlap = math.pi * _DRUM_RADIUS ** 2 * float(j1(zero)) ** 2
    force = -(12.0 * _VISCOSITY / gap ** 3) * overlap / complex(wavenumber_squared, squeeze)
    return -force.real, omega * force.imag, _AREAL_DENSITY * overlap


def _check_drum_film(results_by_name):
    errors, details = {}, {}
    for run, share in (('full', 1.0), ('half', 0.5)):
        results = results_by_name[run]
        names = _FILMS if run == 'full' else ('film_incompressible', 'film_compressible')
        for name in names:
            if name == 'film_vented_all':
                continue
            gap, ambient, _ = _FILMS[name]
            row = _first_row(results, name)
            damping, stiffness, mass = _closed_form(gap, ambient, row['frequency'])
            measured = {'gas_damping': row['gas_damping'] / share, 'effective_mass': row['effective_mass'] / share,
                        'gamma_gas': row['gamma_gas']}
            expected = {'gas_damping': damping, 'effective_mass': mass, 'gamma_gas': damping / mass}
            if ambient > 0.0:
                measured['gas_stiffness'] = row['gas_stiffness'] / share
                expected['gas_stiffness'] = stiffness
            else:   # incompressible: no gas spring (k relative to omega c)
                omega = 2.0 * math.pi * row['frequency']
                errors[f'{run}_{name}_no_spring'] = abs(row['gas_stiffness']) / (omega * abs(row['gas_damping']))
            for column, value in measured.items():
                errors[f'{run}_{name}_{column}'] = abs(value / expected[column] - 1.0)
            details[f'{run}_{name}'] = {'measured': measured, 'expected': expected}
    full = results_by_name['full']
    vented, open_edges = _first_row(full, 'film_vented_all'), _first_row(full, 'film_incompressible')
    errors['open_edges_equals_all'] = abs(open_edges['gas_damping'] / vented['gas_damping'] - 1.0)
    table = _first_row(full, _TABLE)
    film_sum = sum(_first_row(full, name)['gas_damping'] for name in _FILMS)
    errors['table_total'] = abs(table.get('gas_damping', math.nan) / film_sum - 1.0)
    tolerances = {name: (_SAME_TOLERANCE if name in ('open_edges_equals_all', 'table_total')
                         or name.endswith('no_spring') else _DRUM_TOLERANCE) for name in errors}
    return scaled_worst_result(errors, tolerances, **details)


MEMBRANE_SQUEEZE_FILM_CASES = (
    SelfTestCase(
        'membrane_squeeze_film_drum', 'Membrane: squeeze-film damping of the clamped drum mode (0, 1)',
        'The clamped drum of membrane_drum_modes (R = 0.5 mm, P2 h = R / 20, a node at the centre) over a gas gap, rim '
        'vented (a hole outline), slip none: per film the (0, 1) row against the closed form p = -(12 mu / H^3) J0(k '
        'r) / (k^2 + j 12 mu omega / (H^2 p_a)), k = j01 / R (p(R) = 0): c = -Re F, k_gas = omega Im F with F = -(12 mu '
        '/ H^3) pi R^2 J1(j01)^2 / (k^2 + j sigma~), m_eff = rho_s pi R^2 J1(j01)^2, gamma = c / m_eff (incompressible '
        f'2 um: gamma = 12 mu R^2 / (rho_s H^3 j01^2), no gas spring; compressible 15 um at 1 atm), {_DRUM_TOLERANCE:g} '
        'relative; vents open_edges = all on the drum and the resonance table total = the sum of the films '
        f'({_SAME_TOLERANCE:g}); the half model x <= centre with a symmetry face (sealed) gives half c, k, m_eff and the '
        f'same gamma ({_DRUM_TOLERANCE:g}; first estimates).',
        _drum_film_specs, _check_drum_film),
)

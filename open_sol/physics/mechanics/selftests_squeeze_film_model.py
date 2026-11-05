"""Model-run self-test of the solid_mechanics_3d squeeze_film_damping feature (new_modules_plan S6w): the thickness mode
of a roller-guided column (a rigid piston on its top face) under three films against the strip, compressible-strip and
vented-rectangle closed forms, and the resonance-table totals. definition3d registers it."""
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result
from ...model.spec import FeatureSpec, MaterialSpec, StudySpec
from .selftests import _elastic
from .selftests_fem3d import _solid_spec

UM = 1e-6
_SOLID = 'solid'
_TABLE = 'modes'
_FILM_TYPE = 'squeeze_film_damping'
# A column Lx x Ly x Lz (1 x 0.5 x 2 um, 0.125 um Kuhn cubes, P2), E 100 GPa, nu 0.25, rho 2000, fixed on z_lo, rollers
# on the four side faces: u = (0, 0, sin(pi z / (2 Lz))) is an exact continuum mode (lateral strains 0, no shear), f1 =
# c_L / (4 Lz), c_L = sqrt((lambda + 2 mu) / rho) (~968 MHz; the lowest: modes varying across the 1 um width start near
# c_T / (2 Lx) ~ 2.2 GHz). Its top face z_hi moves as a rigid piston of peak 1 m (w = -1), m_eff = rho Lx Ly Lz / 2.
# The discrete mode is near-uniform on the top face (P2, k h ~ 0.1); the deviations below are first estimates.
_COLUMN_SIZE = (1.0 * UM, 0.5 * UM, 2.0 * UM)
_CELL = 0.125 * UM
_MATERIAL = (100e9, 0.25, 2000.0)      # E [Pa], nu, rho [kg/m^3]
# films on z_hi: gap 2 um of air (mu, p_a, mean free path at p_a): squeeze number ~3 at f1 over the 1 um vent distance
_GAP = 2.0 * UM
_VISCOSITY = 1.81e-5
_AMBIENT_PRESSURE = 101325.0
_MEAN_FREE_PATH = 68e-9
_FILMS = {
    # vents on x_lo / x_hi (y rims sealed), incompressible, no slip: c = mu Ly Lx^3 / H^3 (quadratic p: exact in P2)
    'film_strip': {'ambient_pressure': 0.0, 'slip': 'none', 'vents': 'y_sealed'},
    # the same strip in compressible air with Veijola slip: F = -Ly / (j omega C) (Lx - (2 / kappa) tanh(kappa Lx / 2))
    'film_air': {'ambient_pressure': _AMBIENT_PRESSURE, 'slip': 'veijola', 'mean_free_path': _MEAN_FREE_PATH,
                 'vents': 'y_sealed'},
    # the whole rim vented, incompressible: the rectangle series (short side Ly)
    'film_plate': {'ambient_pressure': 0.0, 'slip': 'none', 'vents': 'all'},
}
_SERIES_TERMS = 200                    # odd n of the rectangle series (converged far below 1e-12)
_FREQUENCY_TOLERANCE = 1e-4            # thickness mode vs the continuum
_MASS_TOLERANCE = 1e-3                 # m_eff vs rho V / 2
_STRIP_TOLERANCE = 1e-3                # incompressible strip c (exact in P2 for a uniform w)
_COMPRESSIBLE_TOLERANCE = 2e-3         # compressible strip c / k (8 P2 cells over the vent distance)
_PLATE_TOLERANCE = 2e-3                # vented rectangle c
_IDENTITY_TOLERANCE = 1e-12            # gamma = c / m_eff, Q = omega m_eff / c, table totals
_INCOMPRESSIBLE_STIFFNESS = 1e-9       # |k| / (omega c) of an incompressible film


def _column_spec():
    youngs_modulus, poisson_ratio, density = _MATERIAL
    films = [FeatureSpec(name, _FILM_TYPE, ['z_hi'], params={'gap': _GAP, 'viscosity': _VISCOSITY, **values})
             for name, values in _FILMS.items()]
    return _solid_spec('Self-test: squeeze-film damping of a piston column (tetrahedra)', _COLUMN_SIZE,
                       [MaterialSpec('column', properties=_elastic(youngs_modulus, poisson_ratio, rho=density),
                                     selection=['all'])],
                       [FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                        FeatureSpec('clamp', 'fixed', ['z_lo']), FeatureSpec(_TABLE, 'resonance_table'), *films],
                       cell=_CELL, study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}))


def _stored_material():
    """(E, nu, rho) as the float32 MaterialGrid keeps them."""
    return tuple(float32_value(value) for value in _MATERIAL)


def _thickness_frequency() -> float:
    """f1 = sqrt((lambda + 2 mu) / rho) / (4 Lz) [Hz] of the stored material."""
    youngs_modulus, poisson_ratio, density = _stored_material()
    modulus = youngs_modulus * (1.0 - poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    return math.sqrt(modulus / density) / (4.0 * _COLUMN_SIZE[2])


def _strip_damping(viscosity) -> float:
    """Incompressible c [kg/s] of the strip vented at x_lo / x_hi: mu Ly Lx^3 / H^3."""
    length_x, length_y, _ = _COLUMN_SIZE
    return viscosity * length_y * length_x ** 3 / _GAP ** 3


def _plate_damping() -> float:
    """Incompressible c [kg/s] of the fully vented Lx x Ly rectangle: mu a b^3 / H^3 (1 - (192 b / (pi^5 a)) sum_n odd
    tanh(n pi a / (2 b)) / n^5), b = Ly the short side."""
    long_side, short_side, _ = _COLUMN_SIZE
    series = sum(math.tanh(n * math.pi * long_side / (2.0 * short_side)) / n ** 5
                 for n in range(1, 2 * _SERIES_TERMS, 2))
    correction = 1.0 - 192.0 * short_side / (math.pi ** 5 * long_side) * series
    return _VISCOSITY * long_side * short_side ** 3 / _GAP ** 3 * correction


def _compressible_strip(omega) -> tuple:
    """(c [kg/s], k [N/m]) of the vented strip in compressible air with Veijola slip at omega [rad/s]."""
    import cmath
    length_x, length_y, _ = _COLUMN_SIZE
    effective = _VISCOSITY / (1.0 + 9.638 * (_MEAN_FREE_PATH / _GAP) ** 1.159)
    conductance, compressibility = _GAP ** 3 / (12.0 * effective), _GAP / _AMBIENT_PRESSURE
    kappa = cmath.sqrt(1j * omega * compressibility / conductance)
    force = -length_y / (1j * omega * compressibility) * (length_x - 2.0 / kappa * cmath.tanh(kappa * length_x / 2.0))
    return -force.real, omega * force.imag


def _film_row(results, name) -> dict:
    """The single row of a film table as {column: value}."""
    table = results.result(_SOLID, name)
    return dict(zip(table['columns'], table['rows'][0]))


def _check_column(results_by_name):
    results = results_by_name['column']
    modes = results.result(_SOLID, _TABLE)
    columns, rows = list(modes['columns']), modes['rows']
    errors, tolerances = {}, {}
    if len(rows) != 1 or 'gas_damping' not in columns:
        errors['table'], tolerances['table'] = math.inf, 1.0
        return scaled_worst_result(errors, tolerances, columns=columns, rows=rows)
    mode = dict(zip(columns, rows[0]))
    films = {name: _film_row(results, name) for name in _FILMS}
    length_x, length_y, length_z = _COLUMN_SIZE
    reference_mass = _stored_material()[2] * length_x * length_y * length_z / 2.0
    omega = 2.0 * math.pi * mode['frequency']
    strip, air, plate = films['film_strip'], films['film_air'], films['film_plate']
    air_damping, air_stiffness = _compressible_strip(2.0 * math.pi * air['frequency'])
    errors.update({
        'frequency': abs(mode['frequency'] / _thickness_frequency() - 1.0),
        'effective_mass': abs(strip['effective_mass'] / reference_mass - 1.0),
        'strip_damping': abs(strip['gas_damping'] / _strip_damping(_VISCOSITY) - 1.0),
        'strip_stiffness': abs(strip['gas_stiffness']) / (omega * strip['gas_damping']),
        'air_damping': abs(air['gas_damping'] / air_damping - 1.0),
        'air_stiffness': abs(air['gas_stiffness'] / air_stiffness - 1.0),
        'plate_damping': abs(plate['gas_damping'] / _plate_damping() - 1.0),
        'gamma_identity': max(abs(row['gamma_gas'] * row['effective_mass'] / row['gas_damping'] - 1.0)
                              for row in films.values()),
        'q_identity': max(abs(row['q_gas'] * row['gas_damping'] / (2.0 * math.pi * row['frequency']
                                                                   * row['effective_mass']) - 1.0)
                          for row in films.values()),
        'table_damping': abs(mode['gas_damping'] / sum(row['gas_damping'] for row in films.values()) - 1.0),
        'table_stiffness': abs(mode['gas_stiffness'] - sum(row['gas_stiffness'] for row in films.values()))
                           / abs(air['gas_stiffness']),
        'table_gamma': abs(mode['gamma_gas'] / sum(row['gamma_gas'] for row in films.values()) - 1.0),
        'table_q': abs(mode['q_gas'] * mode['gas_damping'] / (omega * strip['effective_mass']) - 1.0)})
    tolerances.update({'frequency': _FREQUENCY_TOLERANCE, 'effective_mass': _MASS_TOLERANCE,
                       'strip_damping': _STRIP_TOLERANCE, 'strip_stiffness': _INCOMPRESSIBLE_STIFFNESS,
                       'air_damping': _COMPRESSIBLE_TOLERANCE, 'air_stiffness': _COMPRESSIBLE_TOLERANCE,
                       'plate_damping': _PLATE_TOLERANCE, 'gamma_identity': _IDENTITY_TOLERANCE,
                       'q_identity': _IDENTITY_TOLERANCE, 'table_damping': _IDENTITY_TOLERANCE,
                       'table_stiffness': _IDENTITY_TOLERANCE, 'table_gamma': _IDENTITY_TOLERANCE,
                       'table_q': _IDENTITY_TOLERANCE})
    details = {'frequency_Hz': mode['frequency'], 'continuum_Hz': _thickness_frequency(),
               'effective_mass_kg': strip['effective_mass'], 'reference_mass_kg': reference_mass,
               'films': films, 'strip_closed_form': _strip_damping(_VISCOSITY),
               'air_closed_form': {'damping': air_damping, 'stiffness': air_stiffness},
               'plate_closed_form': _plate_damping(), 'resonance_row': mode}
    return scaled_worst_result(errors, tolerances, **details)


SQUEEZE_FILM_MODEL_CASES = (
    SelfTestCase(
        name='solid_fem3d_squeeze_film_column',
        label='Solid mechanics 3D on tetrahedra: squeeze-film damping of a piston column',
        description=('eigenfrequency study of solid_mechanics_3d (cpu, builtin Kuhn tetrahedra, P2, host SuperLU) on a 1 '
                     'x 0.5 x 2 um column (E 100 GPa, nu 0.25, rho 2000) fixed on z_lo with rollers on the four sides, '
                     'whose lowest mode is the exact thickness mode sin(pi z / (2 Lz)) at c_L / (4 Lz) (within '
                     f'{_FREQUENCY_TOLERANCE:g}; m_eff = rho V / 2 within {_MASS_TOLERANCE:g}), so its top face is a '
                     'rigid piston of 1 m. Three squeeze_film_damping features on z_hi, gap 2 um: vents on the x rims '
                     f'(y_sealed), incompressible, no slip: c = mu Ly Lx^3 / H^3 within {_STRIP_TOLERANCE:g} and k = 0; '
                     'the same strip in air (p_a 101325 Pa, Veijola slip, mean free path 68 nm): c and k of the '
                     'compressible strip -Ly / (j omega C) (Lx - (2 / kappa) tanh(kappa Lx / 2)) within '
                     f'{_COMPRESSIBLE_TOLERANCE:g}; the whole rim vented: the rectangle series mu a b^3 / H^3 (1 - (192 '
                     f'b / (pi^5 a)) sum tanh(n pi a / 2 b) / n^5) within {_PLATE_TOLERANCE:g}; gamma_gas = c / m_eff, '
                     'q_gas = omega m_eff / c and the Resonance table columns gas_damping / gas_stiffness / gamma_gas / '
                     f'q_gas = the totals of the three films within {_IDENTITY_TOLERANCE:g}. measured = error = the '
                     'worst deviation in tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {'column': _column_spec()}, check=_check_column),
)

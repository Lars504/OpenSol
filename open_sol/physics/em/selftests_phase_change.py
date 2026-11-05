"""C5 phase-change self-tests: the crystallisation kinetics and mixing on the CPU (isothermal JMAK, melting, linear
quench against the rate integral, Lorentz-Lorenz mixing), and an isothermal em_fdtd_hz anneal driven by transient heat
through heat_em_phase_change (fast test kinetics) against the closed-form JMAK crystallinity and the mixed permittivity;
plus the 'phase_change_gst_switch' template. Specs are built fresh (the em_fdfd_ez anneal is the example
multiphysics/analytic_phase_change_jmak_anneal)."""
import math

import numpy as np

from ...core.constants import C0, ELEMENTARY_CHARGE, EPS0, K_B
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y
from ...model.spec import (CouplingSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)

PHASE_CHANGE_COUPLING = 'heat_em_phase_change'   # = coupling/definitions_phase_change (not imported)
_EM, _HEAT, _MEDIUM = 'em', 'heat', 'pcm'
_BOLTZMANN_EV = K_B / ELEMENTARY_CHARGE   # eV/K
# GST-225 values written out (the feature defaults, physics/em/phase_change.GST_225; the CPU case compares both)
_GST_OPTICS = {'n_amorphous': 3.94, 'k_amorphous': 0.045, 'n_crystalline': 6.11, 'k_crystalline': 0.83,
               'wavelength': 1.55e-6}
_GST_KINETICS = {'crystallization_temperature': 423.0, 'fast_crystallization_temperature': 650.0,
                 'melting_temperature': 889.0, 'min_crystallization_time': 50e-9, 'activation_energy': 2.24,
                 'avrami_exponent': 2.5}
_KINETICS_TOLERANCE = 1e-3
_ANNEAL_TOLERANCE = 1e-4

# anneal runs: (EM type, kinetics, anneal temperature [K], heat time_step [s], heat every, duration [s], cell [m], size)
_ANNEALS = {
    # test kinetics: at 600 K > fast temperature the rate is the cap 1 / 1e-13 s, Avrami 1, over ~1e-13 s of FDTD
    'fdtd_hz': ('em_fdtd_hz', {'crystallization_temperature': 400.0, 'fast_crystallization_temperature': 500.0,
                               'melting_temperature': 900.0, 'min_crystallization_time': 1e-13,
                               'activation_energy': 2.24, 'avrami_exponent': 1.0},
                600.0, 1e-12, 100, 1e-13, 20e-9, (0.4e-6, 0.1e-6)),
}


def jmak_rate(temperature, kinetics):
    """Capped-Arrhenius rate k(T) [1/s] of a kinetics dict (closed form of phase_change.Kinetics.rate)."""
    if not kinetics['crystallization_temperature'] <= temperature < kinetics['melting_temperature']:
        return 0.0
    exponent = -kinetics['activation_energy'] / _BOLTZMANN_EV * (1.0 / temperature
                                                                - 1.0 / kinetics['fast_crystallization_temperature'])
    return math.exp(min(exponent, 0.0)) / kinetics['min_crystallization_time']


def lorentz_lorenz(crystallinity, optics):
    """Complex permittivity (n - j k)^2 mix of the two phases with (eps - 1) / (eps + 2) linear in the crystallinity."""
    polarizabilities = []
    for phase in ('amorphous', 'crystalline'):
        eps = complex(optics[f'n_{phase}'], -optics[f'k_{phase}']) ** 2
        polarizabilities.append((eps - 1.0) / (eps + 2.0))
    mixed = (1.0 - crystallinity) * polarizabilities[0] + crystallinity * polarizabilities[1]
    return (1.0 + 2.0 * mixed) / (1.0 - mixed)


# ---- CPU: kinetics and mixing ----
def _simpson(function, low, high, n_intervals=2000):
    """Composite Simpson integral of a vectorised numpy function over [low, high]."""
    points = np.linspace(low, high, n_intervals + 1)
    weights = np.ones(n_intervals + 1)
    weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
    return float(np.sum(weights * function(points))) * (high - low) / (3.0 * n_intervals)


def _quench_theta(kinetics, rate, start, stop, n_steps):
    """theta of phase_change.Kinetics.advance over a linear cooling start -> stop [K] at `rate` [K/s] (end-of-step T)."""
    theta = np.zeros(1)
    elapsed = (start - stop) / rate / n_steps
    for step in range(1, n_steps + 1):
        kinetics.advance(theta, np.array([start - (start - stop) * step / n_steps]), elapsed)
    return float(theta[0])


def _check_kinetics(results_by_name):
    """Isothermal JMAK, melting, quench integral and the two quench extremes, mixing endpoints / midpoint."""
    from .phase_change import GST_225, Kinetics, Optics   # physics module (cupy at import): inside the check only
    kinetics = Kinetics(GST_225['crystallization_temperature'], GST_225['fast_crystallization_temperature'],
                        GST_225['melting_temperature'], GST_225['min_crystallization_time'],
                        GST_225['activation_energy'], GST_225['avrami_exponent'])
    optics = Optics(GST_225['n_amorphous'], GST_225['k_amorphous'], GST_225['n_crystalline'],
                    GST_225['k_crystalline'], GST_225['wavelength'])
    errors, details = {}, {}
    defaults = {**_GST_OPTICS, **_GST_KINETICS}
    errors['defaults'] = max(abs(GST_225[name] - value) / abs(value) for name, value in defaults.items())

    # isothermal anneal in 40 uneven steps: additivity makes the JMAK form exact for any step split
    anneal_temperature, duration = 600.0, 1.2e-6
    steps = np.linspace(1.0, 3.0, 40)
    steps *= duration / steps.sum()
    theta = np.zeros(1)
    for elapsed in steps:
        kinetics.advance(theta, np.array([anneal_temperature]), elapsed)
    measured_x = float(kinetics.crystallinity(theta)[0])
    anneal_theta = duration * jmak_rate(anneal_temperature, _GST_KINETICS)
    expected_x = -math.expm1(-anneal_theta ** _GST_KINETICS['avrami_exponent'])
    errors['isothermal'] = abs(measured_x - expected_x) / expected_x
    details.update(isothermal_crystallinity=measured_x, expected_isothermal_crystallinity=expected_x)

    # melting resets a fully crystalline cell
    theta = np.full(1, math.inf)
    kinetics.advance(theta, np.array([_GST_KINETICS['melting_temperature'] + 1.0]), 1e-9)
    errors['melting'] = 0.0 if float(theta[0]) == 0.0 else math.inf

    # linear quench from the melting point: theta = (1 / rate) * integral of k(T) dT over [onset, melt]
    melt, onset = _GST_KINETICS['melting_temperature'], _GST_KINETICS['crystallization_temperature']
    fast, tau = _GST_KINETICS['fast_crystallization_temperature'], _GST_KINETICS['min_crystallization_time']
    ratio = _GST_KINETICS['activation_energy'] / _BOLTZMANN_EV
    integral = (melt - fast) / tau + _simpson(lambda t: np.exp(-ratio * (1.0 / t - 1.0 / fast)) / tau, onset, fast)
    quench_rate = 5e9
    measured_theta = _quench_theta(kinetics, quench_rate, melt, 300.0, 10000)
    expected_theta = integral / quench_rate
    errors['quench_integral'] = abs(measured_theta - expected_theta) / expected_theta
    details.update(quench_theta=measured_theta, expected_quench_theta=expected_theta)
    extremes = {}
    for label, rate in (('fast', 1e11), ('slow', 1e8)):
        extremes[label] = float(kinetics.crystallinity(np.array([_quench_theta(kinetics, rate, melt, 300.0, 4000)]))[0])
    details['quench_crystallinity'] = {'fast_1e11_K_per_s': extremes['fast'], 'slow_1e8_K_per_s': extremes['slow']}
    errors['quench_extremes'] = 0.0 if extremes['fast'] < 0.01 and extremes['slow'] > 0.99 else math.inf

    # Lorentz-Lorenz mixing: endpoints are the phases, the midpoint averages (eps - 1) / (eps + 2)
    mixed = optics.permittivity(np.array([0.0, 1.0, 0.5]))
    eps_a, eps_c = optics.phase_permittivity(False), optics.phase_permittivity(True)
    midpoint_polarizability = (complex(mixed[2]) - 1.0) / (complex(mixed[2]) + 2.0)
    expected_polarizability = 0.5 * ((eps_a - 1.0) / (eps_a + 2.0) + (eps_c - 1.0) / (eps_c + 2.0))
    errors['mixing'] = max(abs(complex(mixed[0]) - eps_a) / abs(eps_a), abs(complex(mixed[1]) - eps_c) / abs(eps_c),
                           abs(midpoint_polarizability - expected_polarizability) / abs(expected_polarizability))
    details['midpoint_permittivity'] = [float(mixed[2].real), float(mixed[2].imag)]
    error = max(errors.values())
    return {'measured': measured_theta, 'expected': expected_theta,
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _KINETICS_TOLERANCE,
            'details': {'errors': errors, **details}}


# ---- GPU: isothermal anneals through transient heat ----
def _anneal_spec(case):
    """Uniform GST strip (window and PML) at the anneal temperature: insulated transient heat with T0 = T_anneal (no
    source, so T stays T0 exactly) drives the phase_change_medium through heat_em_phase_change."""
    em_type, kinetics, temperature, time_step, heat_every, duration, cell, (size_x, size_y) = _ANNEALS[case]
    medium_values = {**_GST_OPTICS, **kinetics, 'initial_crystallinity': 0.0, 'mixing': 'lorentz_lorenz'}
    return ModelSpec(
        title=f'Self-test: phase-change anneal ({em_type})', domain=DomainSpec(size_x=size_x, size_y=size_y),
        materials=[MaterialSpec('gst', library='gst_225_amorphous', selection=['all'])],
        physics=[PhysicsSpec(_EM, em_type, settings=dict(PERIODIC_Y),
                             features=[FeatureSpec(_MEDIUM, 'phase_change_medium', ['all'], medium_values)]),
                 PhysicsSpec(_HEAT, 'heat', every=heat_every,
                             settings={'time_mode': 'transient', 'time_step': time_step, 'T0': temperature})],
        couplings=[CouplingSpec('phase', PHASE_CHANGE_COUPLING, _HEAT, _EM)],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='time_dependent', settings={'duration': duration}))


def _check_anneal(results_by_name, case):
    """Mean crystallinity at the last heat time vs 1 - exp(-(k(T) t)^n); eps / sigma_e of every window cell vs the
    Lorentz-Lorenz mix at that crystallinity (relative errors)."""
    em_type, kinetics, temperature, *_ = _ANNEALS[case]
    results = results_by_name['run']
    series = results.result(_EM, _MEDIUM)
    times, values = np.asarray(series['t'], dtype=np.float64), np.asarray(series['values'], dtype=np.float64)
    if times.size < 2:
        return {'measured': math.nan, 'expected': math.nan, 'error': math.inf, 'tolerance': _ANNEAL_TOLERANCE,
                'details': {'n_updates': int(times.size)}}
    expected_x = -math.expm1(-(times[-1] * jmak_rate(temperature, kinetics)) ** kinetics['avrami_exponent'])
    errors = {'crystallinity': abs(values[-1] - expected_x) / expected_x}
    permittivity = lorentz_lorenz(expected_x, _GST_OPTICS)
    conductivity = -permittivity.imag * 2.0 * math.pi * C0 / _GST_OPTICS['wavelength'] * EPS0
    components = ('eps_r',) if em_type.endswith('_ez') else ('eps_xx', 'eps_yy')
    for component in components:
        cells = np.asarray(results.field(f'{_EM}.{component}').values, dtype=np.float64)
        errors[component] = float(np.max(np.abs(cells - permittivity.real))) / permittivity.real
    cells = np.asarray(results.field(f'{_EM}.sigma_e').values, dtype=np.float64)
    errors['sigma_e'] = float(np.max(np.abs(cells - conductivity))) / conductivity
    crystallinity = np.asarray(results.field(f'{_EM}.crystallinity').values, dtype=np.float64)
    errors['crystallinity_field'] = float(np.max(np.abs(crystallinity - expected_x))) / expected_x
    error = max(errors.values())
    return {'measured': float(values[-1]), 'expected': expected_x,
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _ANNEAL_TOLERANCE,
            'details': {'errors': errors, 'last_heat_time': float(times[-1]), 'n_updates': int(times.size),
                        'expected_eps': [permittivity.real, permittivity.imag], 'expected_sigma_e': conductivity}}


def _anneal_case(case, name, label, text):
    return SelfTestCase(
        name=name, label=label,
        description=f'{text} The heat is transient and insulated with T0 = the anneal temperature and no source, so T '
                    'stays T0 exactly; heat_em_phase_change hands T and the heat time to the phase_change_medium after '
                    'every heat step. JMAK additivity makes X = 1 - exp(-(k(T) t)^n) exact for any step split, t = the '
                    'last heat time (the last point of the medium result), so the tolerance 1e-4 covers only float32 '
                    'storage of X, eps and sigma_e; eps / sigma_e / the crystallinity field are checked in every window '
                    'cell against the Lorentz-Lorenz mix of (n - j k)^2 at X (sigma_e = -Im eps omega eps0 at 1.55 um).',
        specs=lambda: {'run': _anneal_spec(case)},
        check=lambda results_by_name: _check_anneal(results_by_name, case))


# ---- template ----
GST_SWITCH_LABEL = 'GST phase-change switch'
GST_SWITCH_DESCRIPTION = ('EM (E_z, frequency domain at 1.55 um) + transient heat, time dependent: a cw line source '
                          '(3 MV/m) heats a 100 nm GST-225 film on 0.5 um of silica over a 293 K heat sink; '
                          'em_heat_joule feeds the absorbed power to heat, heat_em_phase_change crystallises the film '
                          'from its temperature history (more absorbing as it crystallises), and the FDFD re-solves '
                          'after every heat step; crystallinity and T maps, film temperature and crystallinity traces.')
GST_SWITCH_SUMMARY = ('A light beam heats a thin phase-change film until it crystallises; the crystalline film absorbs '
                      'more, heats faster and may melt. Light solved in the frequency domain, heat in time.')


def phase_change_gst_switch_spec():
    """Template 'phase_change_gst_switch' (em_fdfd_ez + transient heat, both couplings)."""
    size_x, size_y = 3.0e-6, 0.1e-6
    film = (2.0e-6, 2.1e-6)
    return ModelSpec(
        title=GST_SWITCH_LABEL, description=GST_SWITCH_DESCRIPTION, domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_line', 'line', {'x0': 1.0e-6, 'y0': 0.0, 'x1': 1.0e-6, 'y1': size_y}),
                  GeometrySpec('film', 'rectangle', {'x0': film[0], 'y0': 0.0, 'x1': film[1], 'y1': size_y}),
                  GeometrySpec('substrate', 'rectangle', {'x0': film[1], 'y0': 0.0, 'x1': size_x, 'y1': size_y}),
                  GeometrySpec('film_point', 'point', {'x': 0.5 * (film[0] + film[1]), 'y': 0.5 * size_y})],
        materials=[MaterialSpec('gst', library='gst_225_amorphous', selection=['film']),
                   MaterialSpec('silica', library='silica', selection=['substrate'])],
        physics=[PhysicsSpec(_EM, 'em_fdfd_ez', settings={**PERIODIC_Y, 'wavelength': 1.55e-6},
                             features=[FeatureSpec('source', 'line_source', ['source_line'], {'amplitude': 3e6}),
                                       FeatureSpec(_MEDIUM, 'phase_change_medium', ['film'], {})]),
                 PhysicsSpec(_HEAT, 'heat', settings={'time_mode': 'transient', 'time_step': 5e-9, 'T0': 293.15},
                             features=[FeatureSpec('sink', 'temperature', ['x_hi'], {'temperature': 293.15}),
                                       FeatureSpec('film_temperature', 'temperature_probe', ['film_point'])])],
        couplings=[CouplingSpec('joule', 'em_heat_joule', _EM, _HEAT),
                   CouplingSpec('phase', PHASE_CHANGE_COUPLING, _HEAT, _EM)],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='time_dependent', settings={'duration': 1e-6},
                        stored_fields=[f'{_EM}.crystallinity', f'{_HEAT}.T']))


PHASE_CHANGE_SELFTEST_CASES = (
    SelfTestCase(
        name='phase_change_kinetics', label='Phase-change kinetics and mixing (CPU)',
        description='CPU check of physics/em/phase_change (no model run) with the GST-225 defaults: (1) the defaults '
                    'equal the values written out here; (2) an isothermal 600 K anneal of 1.2 us in 40 uneven steps '
                    'gives the JMAK X = 1 - exp(-(k t)^n) exactly (additivity); (3) a crystalline cell above the '
                    'melting point resets to theta = 0; (4) a linear cooling at 5e9 K/s from the melting point in 10000 '
                    'end-of-step increments gives theta = (1 / rate) * integral of k(T) dT (the cap part exact, the '
                    'Arrhenius part by Simpson) to O(dT): the first-order step error is ~2e-4 (10000 steps not yet '
                    'confirmed by a run); (5) a melt-quench at '
                    '1e11 K/s stays amorphous (X < 0.01) and one at 1e8 K/s recrystallises (X > 0.99); (6) the '
                    'Lorentz-Lorenz mix returns each phase at X = 0 / 1 and averages (eps - 1) / (eps + 2) at X = 0.5. '
                    f'error = the largest relative error (inf for a failed check); tolerance {_KINETICS_TOLERANCE:g}.',
        specs=lambda: {}, check=_check_kinetics),
    _anneal_case('fdtd_hz', 'em_hz_phase_change_anneal', 'EM H_z phase-change anneal (time domain + heat)',
                 'em_fdtd_hz (eps_xx, eps_yy and sigma_e rewritten every heat step = 100 FDTD steps) on a y-periodic '
                 '0.4 x 0.1 um GST-225 strip '
                 'at 600 K for 1e-13 s with test kinetics: above its fast temperature 500 K the rate is the cap 1 / '
                 '1e-13 s, Avrami exponent 1 (X ~ 0.64).'),
)

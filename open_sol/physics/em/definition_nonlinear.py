"""C5 model features of the FDTD types em_fdtd_ez / _hz: 'nonlinear_medium' (chi2 with poling, Kerr, two-photon
absorption and TPA free carriers; solvers/fdtd_nonlinear) and 'gain_medium' (four-level gain; solvers/fdtd_gain) on the
selected domains, their
validation (time_dependent only, Bloch k = 0), memory estimates, template and self-tests. The FD types (em_fdfd_*,
em_modes) have no such features: they stay linear."""
import math

from ...core.constants import C0, DTYPE, EPS0
from ...core.memory_estimate import BOOL_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.schema import FeatureType, Issue, Param, feature_path
from .definition_bloch import bloch_active_axes
from .definition_common import FREQUENCY_DOMAIN_STUDY, _source_values, raw_settings
from .selftests_nonlinear import TEMPLATE_DESCRIPTION, TEMPLATE_LABEL, nonlinear_silicon_slab_spec
from .solvers.fdtd_gain import gain_coupling

FEATURE_NAME = 'nonlinear_medium'
GAIN_FEATURE_NAME = 'gain_medium'
_ENGINE_PARAMS = ('n2', 'tpa_beta', 'carrier_lifetime', 'fca_cross_section', 'fcd_coefficient', 'pump_wavelength',
                  'd_eff')
_GAIN_ENGINE_PARAMS = ('wavelength', 'linewidth', 'density', 'pump_rate', 'tau_32', 'tau_21', 'tau_10',
                       'initial_upper_fraction')
_PUMP_MISMATCH = 0.01   # relative pump / source wavelength difference that gets an info
_MIN_STEPS_PER_LIFETIME = 10   # gain rate equations: explicit flows, stable for dt << every lifetime

_COMPONENT_TEXT = {
    'ez': 'E_z at the cell centres, n0^2 = eps_r of the cell',
    'hz': ('each in-plane component on its own faces (E_x and E_y, face mean of the cell values as for eps), n0^2 = the '
           'face permittivity; |E|^2 is taken per component (E_x^2 on E_x faces, E_y^2 on E_y faces) and the carriers '
           'live on the faces of the component whose TPA generated them; chi2 acts per component too (P_x = eps0 chi2 '
           'E_x^2 on E_x faces, P_y = eps0 chi2 E_y^2 on E_y faces): exact for a field along one in-plane axis '
           '(slab modes, plane waves), approximate for mixed in-plane polarisation (the cross-component chi2, Kerr and '
           'TPA terms are dropped)'),
}


def _params():
    """n2, TPA beta, carrier lifetime, FCA cross-section, FCD coefficient, pump wavelength, d_eff, poling (SI)."""
    return (
        Param('n2', 'float', 0.0, label='Kerr index n2', unit='m^2/W',
              tip='Intensity-dependent refractive index: the index rises by n2 times the light intensity.\n\n'
                  'Silicon about 6e-18, silicon nitride about 2.5e-19, silica about 2.6e-20 square metres per watt. '
                  '0 switches the Kerr effect off.',
              help='Kerr coefficient: dn = n2 I with I = n0 c eps0 |E|^2 / 2 the cycle-mean intensity; stepped as the '
                   'instantaneous chi3 = (4/3) n0^2 eps0 c n2 [m^2/V^2] (D = eps0 (eps + chi3 E^2) E), so a cw '
                   'field also drives its third harmonic. Silicon ~6e-18, Si3N4 ~2.5e-19, silica ~2.6e-20 m^2/W at '
                   '1.55 um; negative = self-defocusing. Keep chi3 E^2 << eps (the implicit update assumes a '
                   'monotonic D(E)).'),
        Param('tpa_beta', 'float', 0.0, label='Two-photon absorption', unit='m/W', min=0.0,
              tip='Two-photon absorption coefficient: the loss per length grows with the intensity.\n\n'
                  'Silicon at 1550 nm about 8e-12 metres per watt (0.8 cm/GW). 0 switches it off.',
              help='Two-photon absorption: dI/dz = -beta I^2 for a plane wave, stepped as the conductivity sigma = '
                   '(2/3) beta (n0 c eps0)^2 E^2 (its cycle mean gives beta I^2 for a pure cosine). Silicon ~8e-12 '
                   'm/W (0.8 cm/GW) at 1.55 um, GaAs ~3e-10. Its power counts as absorbed power (q_accum, '
                   'em_heat_joule) and, with free-carrier parameters, generates carriers.'),
        Param('carrier_lifetime', 'float', 0.0, label='Carrier lifetime', unit='s', min=0.0,
              tip='How long the electron-hole pairs created by two-photon absorption live.\n\n'
                  'Silicon waveguides about 1e-9 seconds (shorter with carrier sweep-out). Needed when either '
                  'free-carrier value is set.',
              help='Recombination time tau of dN/dt = beta I^2 / (2 hbar omega) - N / tau (generation = the '
                   'instantaneous TPA power / (2 hbar omega), stepped with an implicit Euler step in float64). Must be '
                   '> 0 when fca_cross_section or fcd_coefficient is set; silicon ~0.5-3 ns. Carriers build up over '
                   'about tau, so a pulse train shorter than tau sees only part of the steady density.'),
        Param('fca_cross_section', 'float', 0.0, label='Free-carrier absorption cross-section', unit='m^2', min=0.0,
              tip='Absorption per free carrier: the loss per length is this value times the carrier density.\n\n'
                  'Silicon at 1550 nm about 1.45e-21 square metres. 0 switches it off.',
              help='Free-carrier absorption alpha = sigma N (power loss per length), stepped as the conductivity sigma '
                   'N n0 eps0 c. Silicon ~1.45e-21 m^2 at 1.55 um (scales ~lambda^2). Its power counts as absorbed '
                   'power.'),
        Param('fcd_coefficient', 'float', 0.0, label='Free-carrier index change', unit='m^3', min=0.0,
              tip='Index drop per free carrier: the refractive index falls by this value times the carrier '
                  'density.\n\nSilicon at 1550 nm about 1.35e-27 cubic metres. 0 switches it off.',
              help='Free-carrier dispersion dn = -k N (k >= 0), stepped as delta eps = -2 n0 k N (first order) in the '
                   'implicit update. Silicon ~1.35e-27 m^3 at 1.55 um, a common linearisation of the Soref-Bennett '
                   'electron + hole terms (the hole term is sublinear, so k depends on the density range).'),
        Param('pump_wavelength', 'float', 1.55e-6, label='Pump wavelength', unit='m', min=0.0, min_exclusive=True,
              tip='Vacuum wavelength of the light whose two-photon absorption creates the carriers; it sets the '
                  'photon energy.\n\nUse the source wavelength. Only used with the free-carrier values.',
              help='Vacuum wavelength of the TPA photons: each absorbed pair of photons (energy 2 hbar omega, omega = '
                   '2 pi c / pump_wavelength) creates one carrier pair. Set it to the source centre wavelength '
                   '(validation notes a mismatch); unused without free-carrier parameters.'),
        Param('d_eff', 'float', 0.0, label='Second-order coefficient d', unit='m/V',
              tip='Strength of the second-order nonlinearity (second harmonic, sum and difference frequencies) in metres '
                  'per volt: 1 pm/V is 1e-12.\n\nLithium niobate d33 about 2.5e-11, gallium arsenide about 1.2e-10. 0 '
                  'switches it off.',
              help='Instantaneous second-order response P = eps0 chi2 E^2 with chi2 = 2 d_eff (not scaled by n0), stepped '
                   'implicitly with the Kerr term: a cw field E = Re(E1 exp(j omega t)) drives P(2 omega) = eps0 d_eff '
                   'E1^2 (the SHG phasor of em_fdfd_nonlinear) and the DC rectification eps0 d_eff |E1|^2. One '
                   'scalar per field component, no tensor: E_z types take P_z = eps0 chi2 E_z^2 (d33 of a crystal with '
                   'its c axis along z), H_z types P_x = eps0 chi2 E_x^2 and P_y = eps0 chi2 E_y^2 (no cross terms). LN '
                   'd33 ~2.5e-11, GaAs d14 ~1.2e-10 m/V; keep |chi2 E| << eps. Use em_fdfd_nonlinear for crystal '
                   'tensors; this term is for pulsed time-domain cross-checks.'),
        Param('poling', 'float', 1.0, label='Poling', unit='1', field_dependent=True,
              tip='Sign of the crystal domain at each point for the second-order term: 1 keeps it, -1 flips it. A '
                  'formula such as sign(sin(2*pi*x/period)) gives periodic poling for quasi phase matching.',
              help='Domain sign of chi2 per cell, an expression of x, y and parameters: values >= 0 keep d_eff, < 0 '
                   'invert it (Kerr, TPA and carriers are unchanged). Periodic poling: sign(sin(2 * pi * x / Lambda)) '
                   'with Lambda = lambda / (2 |n(2 omega) - n(omega)|); H_z face values average the two cells, so a '
                   'domain wall on a cell edge gives 0 on the face between them. 1 = unpoled.'),
    )


def _apply(ctx, solver, feature_spec, values, selection):
    """engine.add_nonlinear on the selected cells (PML included under 'all'); a non-constant poling (with d_eff != 0)
    goes as a full-grid array, a constant < 0 as -1 everywhere."""
    mask = ctx.domain_mask(selection.entities)
    poling = None
    if float(values['d_eff']) != 0.0:
        poling = ctx.eval_field(values['poling'], mask=mask)
        if not hasattr(poling, 'shape'):   # a constant sign: unpoled, or inverted everywhere
            poling = None if poling >= 0.0 else -mask.astype(DTYPE)   # float32, as an evaluated poling
    solver.engine.add_nonlinear(mask, poling=poling, **{name: float(values[name]) for name in _ENGINE_PARAMS})
    return None


def _time_domain_issues(vctx, physics_spec, path, feature_type):
    """Errors for a frequency_domain study (superposition broken) and for a Bloch wave vector k != 0."""
    issues = []
    if vctx.study_type == FREQUENCY_DOMAIN_STUDY:
        issues.append(Issue('error', path, f'a {feature_type} breaks superposition, so the frequency_domain study (one '
                                           'linear pulse divided by its spectrum) cannot use it; run a time_dependent '
                                           'study'))
    physics_type = vctx.registry.physics_type(physics_spec.type)
    settings = vctx.values(physics_type.settings, raw_settings(physics_spec))
    if settings is not None and bloch_active_axes(settings):
        issues.append(Issue('error', path, f'a {feature_type} is not available with a Bloch wave vector k != 0 (its '
                                           'response to the field intensity would mix the +k and -k parts of the '
                                           'complex field); set bloch_kx / bloch_ky to 0'))
    return issues


def _validate(vctx, physics_spec, feature_spec, values):
    """Errors: frequency_domain study, Bloch k != 0, free-carrier values without a lifetime; warnings / infos for a
    medium that does nothing, carriers without TPA, an unused lifetime, a pump wavelength off every source."""
    path = feature_path(physics_spec, feature_spec)
    issues = _time_domain_issues(vctx, physics_spec, path, FEATURE_NAME)
    physics_type = vctx.registry.physics_type(physics_spec.type)
    numbers = {name: values.get(name) for name in _ENGINE_PARAMS}
    if not all(isinstance(value, (int, float)) for value in numbers.values()):
        return issues   # reported by the param checks
    carriers = numbers['fca_cross_section'] > 0 or numbers['fcd_coefficient'] > 0
    if numbers['n2'] == 0 and numbers['tpa_beta'] == 0 and numbers['d_eff'] == 0:
        issues.append(Issue('warning', path, 'n2, tpa_beta and d_eff are 0: the medium is linear (free carriers come '
                                             'only from two-photon absorption)'))
    elif carriers and numbers['tpa_beta'] == 0:
        issues.append(Issue('warning', path, 'free-carrier values without tpa_beta: no carriers are generated'))
    if carriers and not numbers['carrier_lifetime'] > 0:
        issues.append(Issue('error', path, 'fca_cross_section / fcd_coefficient need carrier_lifetime > 0 (the '
                                           'recombination time, silicon ~1e-9 s)'))
    elif numbers['carrier_lifetime'] > 0 and not carriers:
        issues.append(Issue('info', path, 'carrier_lifetime is unused: set fca_cross_section and / or '
                                          'fcd_coefficient for free-carrier effects'))
    if carriers and numbers['tpa_beta'] > 0:
        source_wavelengths = [source_values['wavelength']
                              for _, source_values in _source_values(vctx, physics_spec, physics_type)]
        pump = numbers['pump_wavelength']
        if source_wavelengths and all(abs(wavelength - pump) > _PUMP_MISMATCH * pump
                                      for wavelength in source_wavelengths):
            issues.append(Issue('info', path, f'pump_wavelength {pump:.4g} m differs from every source wavelength '
                                              f'({", ".join(f"{wavelength:.4g}" for wavelength in source_wavelengths)}'
                                              ' m): the carrier generation uses its photon energy'))
    return issues


def _memory_estimate(polarization):
    """FeatureType.memory_estimate: per E component set (ez 1, hz 2) on the selection's box the E^n buffer and three
    coefficient arrays (Kerr, TPA, chi2), with carriers four more float32 + the float64 density; build: the full-grid
    union / coefficient temporaries (hz face averages), with d_eff the poling values. The hz pole cell mask / power
    totals it joins: fdtd_memory_estimate."""
    n_sets = 1 if polarization == 'ez' else 2

    def estimate(values, n_selected_cells, mesh):
        carriers = any(isinstance(values.get(name), (int, float)) and values[name] > 0
                       for name in ('fca_cross_section', 'fcd_coefficient'))
        chi2 = isinstance(values.get('d_eff'), (int, float)) and values['d_eff'] != 0
        n_coefficients = 3 + (4 if carriers else 0)
        per_cell = FLOAT32_BYTES * (1 + n_coefficients) + (FLOAT64_BYTES if carriers else 0)
        n_cells = int(mesh.n_cells)
        items = [(f'nonlinear medium ({n_sets} x (E^n + {n_coefficients} coefficients'
                  f'{" + carrier density" if carriers else ""}))', n_sets * per_cell * int(n_selected_cells))]
        face_temporaries = 4 if polarization == 'hz' else 0   # face_average partner, products and result
        poling_temporaries = 2 if chi2 else 0   # eval_field values + the constant-sign fallback (float32 both)
        build_extra = n_cells * (2 * BOOL_BYTES + FLOAT32_BYTES * (1 + n_coefficients + face_temporaries
                                                                   + poling_temporaries))
        return MemoryEstimate.from_items(items, build_extra=build_extra)
    return estimate


def nonlinear_medium_feature(polarization):
    """FeatureType 'nonlinear_medium' of em_fdtd_<polarization>."""
    return FeatureType(
        FEATURE_NAME, 'Nonlinear medium', 'domain_condition', 'domain', params=_params(),
        summary='Makes the selected domains respond to intense light: an intensity-dependent refractive index (Kerr), '
                'two-photon absorption, and the free carriers it creates, which absorb and lower the index, and '
                'optionally a second-order response (second harmonic generation, with periodic poling). Time domain '
                'only.',
        description='chi2 (d_eff, poling), Kerr, two-photon absorption (TPA) and TPA free carriers on the selected cells '
                    '(later features win on overlaps), on top of the linear material (eps, sigma_e, poles). Each step '
                    f'corrects {_COMPONENT_TEXT[polarization]} after the linear update by solving D = eps0 (eps + chi2 '
                    'E + chi3 E^2 + delta eps(N)) E with the TPA + FCA conductivity implicitly (one fused kernel on the '
                    'bounding box '
                    'of the selection; n0 is read live, so thermo-optic eps changes carry over). The carrier density '
                    'N follows dN/dt = TPA power / (2 hbar omega) - N / tau (output field carrier_density, 1/m^3) and '
                    'adds the absorption sigma_fca N and the index change -k N. TPA and FCA power count as absorbed '
                    'power. Nonlinear media break superposition: time_dependent study only (the frequency_domain '
                    'harmonic FDTD and the FDFD / em_modes types are linear), and not with a Bloch wave vector k != '
                    '0. Resolve the harmonics it generates (>= 10 cells per wavelength at wavelength / 2 with d_eff, / '
                    '3 for a strong Kerr medium) or expect them to alias; the DFT monitors then sample often enough '
                    'for up to three times the source band (no harmonic folds into a monitor band in time). '
                    'Single-precision stepper only (its kernels '
                    'and state are float32; a double stepper solver is a validation error).',
        apply=_apply, validate=_validate, memory_estimate=_memory_estimate(polarization), precisions=('single',))


# ---- gain_medium (four-level, two-electron system) ----
def _gain_params():
    """Transition wavelength and linewidth, density, pump rate, three lifetimes, initial upper-level fraction (SI)."""
    return (
        Param('wavelength', 'float', 1.55e-6, label='Transition wavelength', unit='m', min=0.0, min_exclusive=True,
              tip='Vacuum wavelength at which the medium amplifies most (the centre of its gain line).',
              help='Vacuum wavelength lambda_a of the lasing transition 2 -> 1: the polarisation obeys d2P/dt2 + gamma '
                   'dP/dt + omega_a^2 P = kappa (N1 - N2) E, omega_a = 2 pi c / lambda_a.'),
        Param('linewidth', 'float', 1e14, label='Gain linewidth', unit='rad/s', min=0.0,
              tip='Width of the gain line as an angular frequency in rad/s (not Hz).\n\n'
                  '1e14 rad/s is about 130 nm wide at 1550 nm. Smaller values give a narrower, higher gain peak.',
              help='Full width gamma of the Lorentzian gain line in angular frequency (rad/s, not Hz; FWHM in '
                   'wavelength ~ lambda_a^2 gamma / (2 pi c)). The peak gain scales as 1 / gamma.'),
        Param('density', 'float', 1e24, label='Active ion density', unit='1/m^3', min=0.0, min_exclusive=True,
              tip='Number of active atoms or ions per cubic metre: the total over the four levels.',
              help='Total density N0 + N1 + N2 + N3 of the active centres [1/m^3] (conserved). Peak small-signal power '
                   'gain ~ k0 delta_eps omega_a / (gamma n0) per metre with delta_eps = kappa (N2 - N1) / (eps0 '
                   'omega_a^2) (validation reports it).'),
        Param('pump_rate', 'float', 0.0, label='Pump rate', unit='1/s', min=0.0,
              tip='Rate at which the pump lifts atoms from the ground level to the top level, per atom and second.\n\n'
                  "0 means no pump: then set an initial inversion with 'Initial upper-level fraction'.",
              help='Pump rate W_p of N0 -> N3 per ground-state atom [1/s]. Steady small-signal inversion N2 - N1 = '
                   'W_p (tau_21 - tau_10) N, N = density / (1 + W_p (tau_32 + tau_21 + tau_10)); it builds up over '
                   '~tau_21 (often ns = millions of steps), so short runs start from initial_upper_fraction instead. '
                   'Explicit rate step: W_p dt must stay << 1.'),
        Param('tau_32', 'float', 1e-13, label='Lifetime level 3 -> 2', unit='s', min=0.0, min_exclusive=True,
              tip='How fast atoms drop from the top (pumped) level to the upper lasing level.',
              help='Non-radiative lifetime of level 3 into level 2 [s]; >= 10 time steps (explicit rate step).'),
        Param('tau_21', 'float', 1e-9, label='Lifetime level 2 -> 1', unit='s', min=0.0, min_exclusive=True,
              tip='Lifetime of the upper lasing level; it also sets how strongly the transition couples to light.',
              help='Lifetime of the upper lasing level [s], also taken as the radiative lifetime in kappa = 6 pi eps0 '
                   'c^3 / (omega_a^2 tau_21) (Chang and Taflove 2004): a longer tau_21 means a weaker coupling per '
                   'atom but a larger steady inversion.'),
        Param('tau_10', 'float', 1e-13, label='Lifetime level 1 -> 0', unit='s', min=0.0, min_exclusive=True,
              tip='How fast the lower lasing level empties into the ground level; short for good gain.',
              help='Lifetime of the lower lasing level into the ground state [s]; >= 10 time steps.'),
        Param('initial_upper_fraction', 'float', 0.0, label='Initial upper-level fraction', unit='1', min=0.0,
              max=1.0,
              tip='Share of the atoms in the upper lasing level at the start, 0 to 1; the rest start in the ground '
                  'level.\n\nUse it to start from an inverted, amplifying medium without waiting for the pump.',
              help='Fraction of density in level 2 at t = 0 (the rest in level 0): an initial inversion N2 - N1 = '
                   'fraction x density, e.g. for runs much shorter than tau_21. 0 = all in the ground level.'),
    )


def _apply_gain(ctx, solver, feature_spec, values, selection):
    """engine.add_gain on the selected cells (PML included under 'all')."""
    mask = ctx.domain_mask(selection.entities)
    solver.engine.add_gain(mask, **{name: float(values[name]) for name in _GAIN_ENGINE_PARAMS})
    return None


def _validate_gain(vctx, physics_spec, feature_spec, values):
    """Errors: frequency_domain study, Bloch k != 0, lifetimes below 10 time steps, W_p dt >= 0.1; info: the peak
    small-signal gain of the initial and of the pumped steady inversion."""
    path = feature_path(physics_spec, feature_spec)
    issues = _time_domain_issues(vctx, physics_spec, path, GAIN_FEATURE_NAME)
    numbers = {name: values.get(name) for name in _GAIN_ENGINE_PARAMS}
    if not all(isinstance(value, (int, float)) for value in numbers.values()):
        return issues   # reported by the param checks
    dt = getattr(vctx.mesh, 'dt', None)
    if isinstance(dt, (int, float)) and dt > 0:
        short = [name for name in ('tau_32', 'tau_21', 'tau_10')
                 if numbers[name] < _MIN_STEPS_PER_LIFETIME * dt]
        if short:
            issues.append(Issue('error', path, f'{", ".join(short)} below {_MIN_STEPS_PER_LIFETIME} time steps '
                                               f'({_MIN_STEPS_PER_LIFETIME * dt:.3g} s): the explicit rate step would '
                                               'overshoot; use longer lifetimes or a finer mesh'))
        if numbers['pump_rate'] * dt >= 0.1:
            issues.append(Issue('error', path, f'pump_rate x dt = {numbers["pump_rate"] * dt:.3g} >= 0.1: the explicit '
                                               'rate step would overshoot'))
    if numbers['pump_rate'] == 0 and numbers['initial_upper_fraction'] == 0:
        issues.append(Issue('warning', path, 'no pump and no initial inversion: the medium stays in its ground state '
                                             'and neither absorbs nor amplifies'))
        return issues
    omega = 2.0 * math.pi * C0 / numbers['wavelength']
    eps_per_n = gain_coupling(numbers['wavelength'], numbers['tau_21']) / (EPS0 * omega ** 2)
    pumped = numbers['pump_rate'] * numbers['density'] / (
        1.0 + numbers['pump_rate'] * (numbers['tau_32'] + numbers['tau_21'] + numbers['tau_10']))
    inversions = (('initial', numbers['initial_upper_fraction'] * numbers['density']),
                  ('pumped steady', pumped * (numbers['tau_21'] - numbers['tau_10'])))
    if numbers['linewidth'] > 0:
        # line-centre power gain 2 k0 Im n = k0 |delta eps| omega_a / gamma (n0 = 1)
        gain_per_inversion = 2.0 * math.pi / numbers['wavelength'] * eps_per_n * omega / numbers['linewidth']
        texts = [f'{label} N2 - N1 = {inversion:.3g} 1/m^3: peak power gain ~{gain_per_inversion * inversion:.3g} 1/m'
                 for label, inversion in inversions if inversion != 0]
        if texts:
            issues.append(Issue('info', path, '; '.join(texts) + ' (small signal, n0 = 1; divide by the background '
                                                                 'index)'))
    return issues


def _gain_memory_estimate(polarization):
    """FeatureType.memory_estimate of a gain_medium: per E component set (ez 1, hz 2) on the selection's box the
    weight, P, dP, E^n (float32) and four float64 populations; build: the cell weight and hz face-average temporaries."""
    n_sets = 1 if polarization == 'ez' else 2

    def estimate(values, n_selected_cells, mesh):
        per_cell = 4 * FLOAT32_BYTES + 4 * FLOAT64_BYTES
        face_temporaries = 4 if polarization == 'hz' else 0
        build_extra = (int(mesh.n_cells) * FLOAT32_BYTES * (1 + face_temporaries)
                       + 2 * FLOAT64_BYTES * int(n_selected_cells))   # the weight > 0 density temporaries
        return MemoryEstimate.from_items(
            ((f'gain medium ({n_sets} x (weight, P, dP, E^n + 4 float64 populations))',
              n_sets * per_cell * int(n_selected_cells)),), build_extra=build_extra)
    return estimate


def gain_medium_feature(polarization):
    """FeatureType 'gain_medium' of em_fdtd_<polarization>."""
    where = ('E_z at the cell centres' if polarization == 'ez' else
             'each in-plane E face set separately (E_x and E_y faces, weight = face mean of the selected cells; each '
             'face set keeps its own populations, so a field along one in-plane axis depletes only that set)')
    return FeatureType(
        GAIN_FEATURE_NAME, 'Gain medium', 'domain_condition', 'domain', params=_gain_params(),
        summary='Makes the selected domains amplify light near a transition wavelength, as a pumped laser medium '
                'with four energy levels. Time domain only.',
        description='Four-level two-electron gain medium (Chang and Taflove, Opt. Express 12, 3827, 2004) on the '
                    f'selected cells, acting on {where}: a Lorentz polarisation d2P/dt2 + gamma dP/dt + omega_a^2 P = '
                    'kappa (N1 - N2) E, kappa = 6 pi eps0 c^3 / (omega_a^2 tau_21), coupled to the rate equations of '
                    'the populations N0..N3 (pump W_p N0 -> N3, decays N3 -> N2 -> N1 -> N0, stimulated transitions '
                    '(E . dP/dt) / (hbar omega_a)); N2 > N1 amplifies, N1 > N2 absorbs. Regions add (overlaps are two '
                    'species); the polarisation and populations live on the bounding box of the selection. Output '
                    'field population_inversion = N2 - N1 (1/m^3). Not counted in absorbed power; not part of the '
                    'build Courant check (keep |delta eps| (omega_a dt)^2 / 4 << 1). Time_dependent study only, Bloch '
                    'k = 0 only and single-precision stepper only (as nonlinear_medium).',
        apply=_apply_gain, validate=_validate_gain, memory_estimate=_gain_memory_estimate(polarization),
        precisions=('single',))


def _selftest_cases():
    """The Kerr / TPA / free-carrier / gain SelfTestCases of both types."""
    from .selftests_nonlinear import NONLINEAR_SELFTEST_CASES
    return tuple(NONLINEAR_SELFTEST_CASES)


def register(registry):
    """Register the 'nonlinear_silicon_slab' template and the nonlinear / gain selftest loader."""
    registry.register_template(
        'nonlinear_silicon_slab', TEMPLATE_LABEL, TEMPLATE_DESCRIPTION, nonlinear_silicon_slab_spec,
        summary='An intense light pulse crosses a silicon slab: the Kerr effect shifts its phase, two-photon '
                'absorption and the free carriers it creates take away power.')
    registry.register_selftests(_selftest_cases)

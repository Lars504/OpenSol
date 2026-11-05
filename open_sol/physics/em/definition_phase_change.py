"""C5 model feature 'phase_change_medium' of the in-plane EM types (em_fdtd_* / em_fdfd_*): a crystallinity state
(physics/em/phase_change.py) whose mixed amorphous / crystalline permittivity replaces eps / sigma_e on the selected
cells, stepped by the heat temperature history through the coupling heat_em_phase_change; register() adds its template
and self-tests."""
from ...core.memory_estimate import BOOL_BYTES, COMPLEX128_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.definition_helpers import study_duration
from ...model.schema import FeatureType, Issue, Param, feature_path
from .definition_common import _source_values, auto_mesh_issues, resolved_solved_wavelengths
from .phase_change import (FEATURE_NAME, GST_225, GST_225_REFERENCES, MIXING_RULES, Kinetics, Optics,
                           add_phase_change_medium)
from .placement import PERMITTIVITY_ARRAYS
from .selftests_phase_change import (GST_SWITCH_DESCRIPTION, GST_SWITCH_LABEL, GST_SWITCH_SUMMARY,
                                     phase_change_gst_switch_spec)

PHASE_CHANGE_COUPLING = 'heat_em_phase_change'   # = selftests_phase_change.PHASE_CHANGE_COUPLING (not imported)
_WAVELENGTH_MISMATCH = 0.01   # relative reference / used wavelength difference that gets an info
_SHORT_RUN_FRACTION = 0.01    # study duration / min_crystallization_time below which a driven medium barely changes


def _params():
    """Optical constants of both phases, mixing rule, initial state and the JMAK kinetics (GST-225 defaults)."""
    def index_param(name, label, phase, part):
        extinction = part == 'k'
        return Param(name, 'float', GST_225[name], label=label, unit='1', min=0.0, min_exclusive=not extinction,
                     tip=(f'{"Extinction coefficient" if extinction else "Refractive index"} of the {phase} phase at '
                          "the 'Reference wavelength'. The default is Ge2Sb2Te5 at 1550 nm."),
                     help=(f'{"Extinction coefficient k" if extinction else "Refractive index n"} of the {phase} '
                           'phase at wavelength (eps = (n - j k)^2, exp(+j omega t)); default GST-225 at 1.55 um '
                           f"(library material gst_225_{phase})."))
    return (
        index_param('n_amorphous', 'Amorphous refractive index', 'amorphous', 'n'),
        index_param('k_amorphous', 'Amorphous extinction coefficient', 'amorphous', 'k'),
        index_param('n_crystalline', 'Crystalline refractive index', 'crystalline', 'n'),
        index_param('k_crystalline', 'Crystalline extinction coefficient', 'crystalline', 'k'),
        Param('wavelength', 'float', GST_225['wavelength'], label='Reference wavelength', unit='m', min=0.0,
              min_exclusive=True,
              tip='Vacuum wavelength at which the refractive indices and extinction coefficients above apply.\n\n'
                  'Use the wavelength of the light you simulate: the absorption is exact only there.',
              help='Vacuum wavelength of n and k: the absorption enters as sigma_e = -Im(eps) omega eps0 at this '
                   'omega, exact there (FDTD pulses see sigma_e / (omega eps0) away from it); validation notes a '
                   'source / solved wavelength more than 1 % off.'),
        Param('mixing', 'choice', 'lorentz_lorenz', label='Mixing rule', choices=MIXING_RULES,
              choice_labels=(('lorentz_lorenz', 'Lorentz-Lorenz (effective medium)'),
                             ('linear', 'Linear in permittivity')),
              tip='How the permittivity of a partly crystalline cell is formed from the two phases.\n\n'
                  '- Lorentz-Lorenz: the usual effective-medium choice for phase-change alloys.\n'
                  '- Linear: the permittivity changes in proportion to the crystallinity.',
              help="Permittivity of crystallinity X: 'lorentz_lorenz' (eps - 1) / (eps + 2) = (1 - X) (eps_a - 1) / "
                   "(eps_a + 2) + X (eps_c - 1) / (eps_c + 2) (complex), or 'linear' eps = (1 - X) eps_a + X eps_c."),
        Param('initial_crystallinity', 'float', 0.0, label='Initial crystallinity', unit='1', min=0.0, max=1.0,
              tip='Crystalline share of every selected cell at the start: 0 amorphous, 1 fully crystalline.\n\n'
                  'Without a phase-change coupling from a heat physics it stays at this value.',
              help='Crystallinity X at t = 0 on every selected cell (theta = (-ln(1 - X))^(1/n)); without an enabled '
                   f'{PHASE_CHANGE_COUPLING} into this physics (time_dependent study) X stays here, which also gives '
                   'a static, partially crystalline medium for any study.'),
        Param('crystallization_temperature', 'float', GST_225['crystallization_temperature'],
              label='Crystallisation onset temperature', unit='K', min=0.0, min_exclusive=True,
              tip='Below this temperature the material does not crystallise at all.\n\n'
                  'Ge2Sb2Te5: about 423 kelvin (150 degrees Celsius).',
              help='Onset T_x [K]: k(T) = 0 below it (room-temperature drift switched off). GST-225 ~423 K (150 C, '
                   'slow anneals; Friedrich et al. 2000).'),
        Param('fast_crystallization_temperature', 'float', GST_225['fast_crystallization_temperature'],
              label='Fastest-crystallisation temperature', unit='K', min=0.0, min_exclusive=True,
              tip="Above this temperature crystallisation proceeds at its fastest rate, set by 'Fastest "
                  "crystallisation time'; below it, it slows down steeply.",
              help='T_fast [K]: the rate is capped at 1 / min_crystallization_time from T_fast up to the melting '
                   'point (the Arrhenius law would otherwise give picosecond crystallisation near melting, so no '
                   'melt-quench could stay amorphous). Illustrative GST-225 value 650 K.'),
        Param('min_crystallization_time', 'float', GST_225['min_crystallization_time'],
              label='Fastest crystallisation time', unit='s', min=0.0, min_exclusive=True,
              tip='Time scale of crystallisation at its fastest, just below melting.\n\n'
                  'Ge2Sb2Te5: tens of nanoseconds. A melted cell that cools through the crystallisation range much '
                  'faster than this stays amorphous.',
              help='tau_min [s]: JMAK rate k = 1 / tau_min between fast_crystallization_temperature and the melting '
                   'point; a melt-quench crossing that range in much less than tau_min stays amorphous (with the '
                   'defaults a 1e11 K/s quench keeps X < 0.01, a 1e8 K/s one recrystallises fully). Illustrative '
                   'GST-225 value 50 ns (tens of ns in SET pulses).'),
        Param('activation_energy', 'float', GST_225['activation_energy'], label='Activation energy', unit='eV',
              min=0.0,
              tip='How steeply crystallisation slows down below the fastest-crystallisation temperature, in '
                  'electronvolts.\n\nGe2Sb2Te5: about 2.2 eV.',
              help='Ea [eV] of the Arrhenius rate k(T) = exp(-Ea / kB (1/T - 1/T_fast)) / tau_min between the onset and '
                   'T_fast. GST-225 2.24 eV (Friedrich et al. 2000); with the default cap it gives ~100 s at the '
                   '423 K onset.'),
        Param('avrami_exponent', 'float', GST_225['avrami_exponent'], label='Avrami exponent', unit='1', min=0.0,
              min_exclusive=True,
              tip='Shape of the crystallisation curve in time: larger values give a slower start and a sharper '
                  'switch.\n\nTypically 2 to 4.',
              help='JMAK exponent n: X = 1 - exp(-theta^n), theta = integral of k(T) dt (the additivity rule, exact '
                   'for isothermal steps of any length); 2-4 for GST (2.5 default).'),
        Param('melting_temperature', 'float', GST_225['melting_temperature'], label='Melting temperature', unit='K',
              min=0.0, min_exclusive=True,
              tip='At and above this temperature the material melts and becomes amorphous; it crystallises again '
                  'only if it cools slowly.\n\nGe2Sb2Te5: about 889 kelvin.',
              help='T_melt [K]: a cell at T >= T_melt is molten: theta = 0, X = 0 (optically amorphous; the liquid '
                   'optical constants and the latent heat are not modelled). GST-225 889 K (616 C, Yamada et al. '
                   '1991).'),
    )


def kinetics_from_values(values):
    """phase_change.Kinetics of resolved feature values."""
    return Kinetics(float(values['crystallization_temperature']), float(values['fast_crystallization_temperature']),
                    float(values['melting_temperature']), float(values['min_crystallization_time']),
                    float(values['activation_energy']), float(values['avrami_exponent']))


def optics_from_values(values):
    """phase_change.Optics of resolved feature values."""
    return Optics(float(values['n_amorphous']), float(values['k_amorphous']), float(values['n_crystalline']),
                  float(values['k_crystalline']), float(values['wavelength']), values['mixing'])


def _apply(ctx, solver, feature_spec, values, selection):
    """PhaseChangeMedium on the selected cells (PML included under 'all'): its initial eps / sigma_e written now."""
    mask = ctx.domain_mask(selection.entities)
    return add_phase_change_medium(solver, mask, kinetics_from_values(values), optics_from_values(values),
                                   float(values['initial_crystallinity']))


def _result(session, solver, medium, feature_spec, options):
    return medium.result()


def _used_wavelengths(vctx, physics_spec, physics_type):
    """Wavelengths [m] the physics works at: FDFD the solved ones, FDTD its sources' centre wavelengths."""
    if any(param.name == 'wavelength' for param in physics_type.settings):
        return resolved_solved_wavelengths(vctx, physics_spec, physics_type)
    return [values['wavelength'] for _, values in _source_values(vctx, physics_spec, physics_type)]


def _validate(vctx, physics_spec, feature_spec, values):
    """Errors: temperatures out of order, Re eps < 1 in a time-stepped type; infos: phase permittivities, wavelength
    mismatch, no driving coupling; warnings: fast temperature >= melting, auto mesh, a driven FDTD run far shorter than
    tau_min."""
    path = feature_path(physics_spec, feature_spec)
    physics_type = vctx.registry.physics_type(physics_spec.type)
    if not all(isinstance(value, (int, float)) for name, value in values.items() if name != 'mixing'):
        return []   # reported by the param checks
    issues = []
    onset, fast = values['crystallization_temperature'], values['fast_crystallization_temperature']
    melt = values['melting_temperature']
    if not onset < melt:
        issues.append(Issue('error', path, f'crystallization_temperature ({onset:.6g} K) must be below '
                                           f'melting_temperature ({melt:.6g} K)'))
    if not fast >= onset:
        issues.append(Issue('error', path, f'fast_crystallization_temperature ({fast:.6g} K) must not be below '
                                           f'crystallization_temperature ({onset:.6g} K)'))
    if fast >= melt and onset < melt:
        issues.append(Issue('warning', path, f'fast_crystallization_temperature ({fast:.6g} K) is not below '
                                             f'melting_temperature ({melt:.6g} K): the capped rate 1 / '
                                             'min_crystallization_time is never reached (molten cells do not '
                                             'crystallize), so the fastest crystallization is slower than that'))
    optics = optics_from_values(values)
    phases = {'amorphous': optics.phase_permittivity(False), 'crystalline': optics.phase_permittivity(True)}
    sigma_factor = optics.conductivity_factor()
    issues.append(Issue('info', path, '; '.join(f'{phase}: eps_r = {eps.real:.4g}, sigma_e = '
                                                f'{-eps.imag * sigma_factor:.4g} S/m' for phase, eps in phases.items())
                        + f" at {values['wavelength']:.4g} m"))
    time_stepped = physics_type.steps_in_time is not None
    low = [phase for phase, eps in phases.items() if eps.real < 1.0]
    if time_stepped and low:
        issues.append(Issue('error', path, f"{' and '.join(low)} phase with Re eps = n^2 - k^2 < 1: the time step is "
                                           'checked only at the build (initial state), and Re eps >= 1 keeps every '
                                           'later mix stable; this medium needs a frequency-domain type'))
    issues.extend(auto_mesh_issues(vctx, physics_spec, path, 'the auto mesh sizes cells from the material indices, not '
                                                              f"from n_crystalline = {values['n_crystalline']:.4g}: "
                                                              'give the selected domains a material of that index '
                                                              '(e.g. gst_225_crystalline) or use a uniform / graded '
                                                              'mesh'))
    used = _used_wavelengths(vctx, physics_spec, physics_type)
    reference = values['wavelength']
    if used and all(abs(wavelength - reference) > _WAVELENGTH_MISMATCH * reference for wavelength in used):
        issues.append(Issue('info', path, f'reference wavelength {reference:.4g} m differs from every source / solved '
                                          f"wavelength ({', '.join(f'{wavelength:.4g}' for wavelength in used)} m): "
                                          'the absorption sigma_e is exact only at the reference wavelength'))
    driven = any(coupling.enabled and coupling.type == PHASE_CHANGE_COUPLING and coupling.target == physics_spec.name
                 for coupling in vctx.spec.couplings)
    if not driven:
        issues.append(Issue('info', path, f'no enabled {PHASE_CHANGE_COUPLING} coupling (heat -> '
                                          f"'{physics_spec.name}'): the crystallinity stays at initial_crystallinity "
                                          f"({values['initial_crystallinity']:.3g})"))
    duration = study_duration(vctx) if vctx.study_type == 'time_dependent' else None
    if (driven and time_stepped and isinstance(duration, (int, float))
            and duration < _SHORT_RUN_FRACTION * values['min_crystallization_time']):
        issues.append(Issue('warning', path, f'the run ({duration:.3g} s) is shorter than '
                                             f'{100 * _SHORT_RUN_FRACTION:g} % of min_crystallization_time '
                                             f"({values['min_crystallization_time']:.3g} s), so the crystallinity "
                                             'barely changes (only melting acts this fast): time-domain EM steps are '
                                             '~1e-17 s; switching on heat time scales needs em_fdfd_* in a '
                                             'time_dependent study'))
    return issues


def _memory_estimate(values, n_selected_cells, mesh):
    """theta (float64), crystallinity (float32), mask and mean weights on the selection's box; build / update: the
    full-grid domain mask and the complex mixing temporaries on the box."""
    n_cells = int(n_selected_cells)
    per_cell = FLOAT64_BYTES + FLOAT32_BYTES + BOOL_BYTES + FLOAT64_BYTES
    temporaries = n_cells * (3 * COMPLEX128_BYTES + 4 * FLOAT64_BYTES + 2 * FLOAT32_BYTES)
    return MemoryEstimate.from_items(
        (('phase-change medium (theta, crystallinity, mask, mean weights)', per_cell * n_cells),),
        build_extra=int(mesh.n_cells) * BOOL_BYTES + temporaries)


def phase_change_medium_feature(polarization):
    """FeatureType 'phase_change_medium' of an in-plane EM type of polarization 'ez' | 'hz'."""
    arrays = ' / '.join(PERMITTIVITY_ARRAYS[polarization])
    return FeatureType(
        FEATURE_NAME, 'Phase-change medium', 'domain_condition', 'domain', params=_params(), result_kind='timeseries',
        summary='Makes the selected domains a phase-change alloy such as GST: it crystallises when heated, melts and '
                'turns amorphous when very hot, and its refractive index and absorption follow. The temperature comes '
                'from a heat physics through the phase-change coupling.',
        description='Phase-change medium (e.g. GST-225, the defaults) on the selected cells: a per-cell crystallinity '
                    'X in [0, 1] (output field crystallinity) sets eps(X) by the mixing rule of the amorphous and '
                    f'crystalline (n - j k)^2, written as {arrays} = Re eps and sigma_e = -Im eps omega eps0 (omega of '
                    'wavelength) over the material there (later features win on overlaps; dispersion poles of the '
                    'material stay; a state-dependent permittivity expression of a material overwrites it). X follows '
                    f'the heat temperature history only through {PHASE_CHANGE_COUPLING} (time_dependent study, '
                    'transient heat): after each heat step theta += k(T) dt with T the end-of-step temperature, X = 1 - '
                    'exp(-theta^n) (JMAK with the additivity rule), k(T) = exp(-Ea / kB (1/T - 1/T_fast)) / tau_min '
                    'between crystallization_temperature and fast_crystallization_temperature, 1 / tau_min from there '
                    'to the melting point, 0 below the onset; T >= melting_temperature melts the cell (theta = 0, X = '
                    '0), and whether it recrystallises while cooling follows from the same rate (a fast quench stays '
                    'amorphous). Each update rewrites the arrays on the bounding box of the selection and calls '
                    'materials_changed (FDTD: coefficients rebuilt, sources recalibrated; FDFD: re-solved at its next '
                    'step). Not modelled: latent heat, the liquid optical constants, thermal properties following X '
                    '(heat uses the material, e.g. gst_225_amorphous), nucleation statistics. Result: timeseries of '
                    'the mean X over the selected cells at every update (t = heat time). Linear at optical time scales, '
                    'so it works in every study (frequency_domain, FDFD, Bloch k) as a static medium at '
                    'initial_crystallinity. Time scales: GST crystallises in tens of ns, ~1e9 FDTD steps; switching '
                    'runs use em_fdfd_* + transient heat in a time_dependent study (the FDFD re-solves after every '
                    f'heat step). {GST_225_REFERENCES}',
        apply=_apply, validate=_validate, result=_result, memory_estimate=_memory_estimate)


def _selftest_cases():
    """The phase-change SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_phase_change import PHASE_CHANGE_SELFTEST_CASES
    return PHASE_CHANGE_SELFTEST_CASES


def register(registry):
    """Register the 'phase_change_gst_switch' template and the phase-change selftest loader."""
    registry.register_template('phase_change_gst_switch', GST_SWITCH_LABEL, GST_SWITCH_DESCRIPTION,
                               phase_change_gst_switch_spec, summary=GST_SWITCH_SUMMARY)
    registry.register_selftests(_selftest_cases)

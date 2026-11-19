"""Built-in coupling types: EM Joule heating -> heat, resistive Joule heating (electric currents -> heat), heat -> EM
thermo-optics, generic field transfer, the E13 contributor helpers; also registers the coupled-chain selftests of
coupling/selftests.py, selftests_slot_sum.py and selftests_mesh_transfer.py (per-physics meshes)."""
import importlib
import math
import pkgutil
from dataclasses import dataclass

from ..core.constants import C0, DTYPE
from ..model.definition_helpers import evaluates_to, selected_material_data, study_duration
from ..model.expressions import FIELD_COORDINATE_NAMES, compile_expression
from ..model.schema import (TIME_DEPENDENT_STUDY, CouplingType, Issue, Param, coupling_reads_absorbed_power,
                            effective_coupling_every, valid_every)
from ..physics.heat.coupling import DNDT_ARRAYS
from .coupling_term import CouplingTerm
from .index_terms import after_source_step, index_slots, physics_and_type, writes_index_slots
from .standard import THERMO_OPTIC_SLOTS, joule_heating, thermo_optic

HEAT_TYPE_NAME = 'heat'
CURRENTS_TYPE_NAME = 'electric_currents'   # = physics.potential.definition.CURRENTS_TYPE_NAME (not imported here)
# EM loss protocol (every solver of a family-'em' type): state()['q_accum'] always present (None allowed); a settable
# getattr(solver, 'engine', solver).enable_q_accum; take_absorbed_power(merge_previous=False) -> FDTD: mean W/m^3 since
# the last call, then reset (merge_previous: also over the last completed window), None for an empty window; FDFD: q_abs
# of its last solve (never reset, None before the first). Each take resets an FDTD window: one reader per time-stepped
# source (CouplingType.reads_absorbed_power, absorbed_power_reader_issues).
_EM_FAMILY = 'em'
_TRANSFER_NAMES = FIELD_COORDINATE_NAMES | {'value'}   # a field_transfer expression's names besides parameters
_CW_RIPPLE_WARNING = 0.01   # relative cell-to-cell error of a CW Joule window mean above which validation warns
# EM types whose plane is a cross-section transverse to the propagation (thermo-optic expansion on eps_zz only); a new
# cross-section EM type joins it
_CROSS_SECTION_EM_TYPES = frozenset({'em_modes'})
_STRAIN_OPTIC_TYPE_NAME = 'solid_em_strain_optic'   # = physics.mechanics.selftests.STRAIN_OPTIC_COUPLING (not imported)
_EXTENSION_PREFIX = 'definitions_'   # coupling/definitions_<topic>.py: register(registry, RegisteredPhysics)


def solid_supplies_deformation(spec, em_name):
    """True when an enabled solid_em_strain_optic with deformation on targets EM physics em_name: its strain image
    replaces the (1 + alpha dT) factors of heat_em_thermo_optic there."""
    for coupling in spec.couplings:
        params = coupling.params if isinstance(coupling.params, dict) else {}
        if (coupling.enabled and coupling.type == _STRAIN_OPTIC_TYPE_NAME and coupling.target == em_name
                and params.get('deformation', True) is not False):
            return True
    return False


# ---- shared help texts (coupling/definitions_*.py import them) ----
HEAT_SOURCE_SUM_TEXT = (
    'Additive heat source (volumetric_power): every em_heat_joule / currents_heat_joule / fluid_heat_convection / '
    'magnetic_heat_induction coupling into one heat physics sums, all with the same every (validation error '
    'otherwise); no field_transfer into volumetric_power beside them.')
INDEX_SUM_TEXT = (
    'An E13 index contributor: the slot gets eps = (n0 + the sum of dn of every index coupling into it)^2, n0 the '
    'shared reference index recorded before the first transfer, all with one every (validation error otherwise); no '
    'overwriting field_transfer or state-dependent permittivity property on the slot.')


# ---- E13 contributor helpers (index ones: coupling/index_terms.py) ----
def writes_heat_source(vctx, coupling_spec):
    """CouplingType.writes of a heat-source contributor: {'volumetric_power': 'sum'}."""
    return {'volumetric_power': 'sum'}


# ---- create callbacks (worker thread, GPU) ----
def _create_currents_joule(ctx, coupling_spec, values, every):
    """'sum' contributor copying q_joule into volumetric_power (no transform)."""
    return CouplingTerm(coupling_spec.source, 'q_joule', coupling_spec.target, 'volumetric_power', every=every,
                        name=coupling_spec.name, contribution='sum')


def _create_em_joule(ctx, coupling_spec, values, every):
    """standard.joule_heating between the spec's physics."""
    return joule_heating(em=coupling_spec.source, heat=coupling_spec.target, every=every, name=coupling_spec.name)


def _create_thermo_optic(ctx, coupling_spec, values, every):
    """standard.thermo_optic over the target solver's permittivity arrays (validation guarantees thermo-optic slots);
    without the alpha factors when a solid strain-optic coupling maps the deformation onto the same EM physics, and
    only on eps_zz of an em_modes cross-section (the transverse-stretch image leaves eps_xx / eps_yy unchanged)."""
    target_solver = ctx.sim.physics[coupling_spec.target]
    expansion = not solid_supplies_deformation(ctx.spec, coupling_spec.target)
    if expansion and ctx.spec.physics_by_name(coupling_spec.target).type in _CROSS_SECTION_EM_TYPES:
        expansion = ('eps_r',)
    return thermo_optic(heat=coupling_spec.source, em=coupling_spec.target,
                        permittivity_arrays=getattr(target_solver, 'permittivity_arrays', ()), every=every,
                        after_source_step=after_source_step(ctx, coupling_spec, every), name=coupling_spec.name,
                        with_expansion=expansion)


def _create_field_transfer(ctx, coupling_spec, values, every):
    """CouplingTerm whose transform evaluates the expression (value = source field) into a persistent float32
    full-grid buffer (no allocation per transfer beyond the expression's own temporaries)."""
    expression = compile_expression(values['expression'])
    expression.check_names(_TRANSFER_NAMES.union(ctx.parameters))
    namespace = dict(ctx.parameters)
    if not expression.names.isdisjoint(FIELD_COORDINATE_NAMES):   # clamped into the PML ring; 3D: (X, Y, Z)
        namespace.update(zip(('x', 'y', 'z'), ctx.domain.coordinate_grids(True)))
    xp = ctx.xp   # the run backend's array module
    transferred = xp.empty(ctx.domain.materials.shape, dtype=DTYPE)   # (nx, ny) or 3D (nx, ny, nz)
    coupling_name = coupling_spec.name

    def transform(field, source_solver, target_solver):
        namespace['value'] = field
        result = expression.evaluate(namespace, xp=xp)
        if getattr(result, 'ndim', 0) and result.shape != transferred.shape:
            raise ValueError(f"coupling '{coupling_name}': expression {expression.text!r} gives shape "
                             f"{result.shape}, expected the full grid {transferred.shape}")
        transferred[...] = result   # casts to float32; a scalar (value unused) fills the grid
        return transferred

    return CouplingTerm(coupling_spec.source, values['source_field'], coupling_spec.target, values['target_slot'],
                        transform, every=every, name=coupling_name,
                        after_source_step=after_source_step(ctx, coupling_spec, every))


# ---- validation helpers (CPU only) ----
def _writes_field_transfer(vctx, coupling_spec):
    """CouplingType.writes of field_transfer: {target_slot: ''} (overwrites); {} when the param is not text."""
    slot = coupling_spec.params.get('target_slot') if isinstance(coupling_spec.params, dict) else None
    return {slot: ''} if isinstance(slot, str) and slot else {}


def source_steps_in_time(vctx, physics_spec):
    """True when the physics' type steps in time with its settings (FDTD); False when unknown."""
    try:
        physics_type = vctx.registry.physics_type(physics_spec.type)
    except KeyError:
        return False
    if physics_type.steps_in_time is None:
        return False
    settings = vctx.values(physics_type.settings, physics_spec.settings)
    return bool(settings is not None and physics_type.steps_in_time(settings))


def absorbed_power_reader_issues(vctx, coupling_spec):
    """Error when another enabled reads_absorbed_power coupling takes the absorbed power of the same time-stepped
    source: each take_absorbed_power() resets its window, so each reader would get only part of it."""
    try:
        source_spec = vctx.spec.physics_by_name(coupling_spec.source)
    except KeyError:
        return []
    if not source_steps_in_time(vctx, source_spec):
        return []
    others = [other.name for other in vctx.spec.couplings
              if other.enabled and other.name != coupling_spec.name and other.source == coupling_spec.source
              and coupling_reads_absorbed_power(vctx.registry, other)]
    if not others:
        return []
    noun, verb = ('couplings', 'take') if len(others) > 1 else ('coupling', 'takes')
    return [Issue('error', ('coupling', coupling_spec.name),
                  f"{noun} {', '.join(others)} also {verb} the absorbed power of the "
                  f"time-stepped '{source_spec.name}': each transfer resets its averaging window, so each reader "
                  'would get only part of it; keep one reader (or use an em_fdfd type)')]


def _pulsed_sources(em_spec, em_type):
    """Names of enabled EM source features whose waveform is not 'cw'."""
    names = []
    for feature in em_spec.features:
        if not feature.enabled:
            continue
        try:
            feature_type = em_type.feature_type(feature.type)
        except KeyError:
            continue
        waveform_param = next((param for param in feature_type.params if param.name == 'waveform'), None)
        if feature_type.category != 'source' or waveform_param is None:
            continue
        params = feature.params if isinstance(feature.params, dict) else {}
        if params.get('waveform', waveform_param.default) != 'cw':
            names.append(feature.name)
    return names


def _cw_ripple_issues(vctx, path, em_spec, em_type, every, em_dt):
    """Warning when the window every * dt_EM is not a whole number of half periods of a cw source, so its mean of
    cos^2 power is off by |sin(omega W)| / (omega W) > 1 %, varying from cell to cell with the phase."""
    worst = None   # (relative error, source name, half period in EM steps)
    for feature in em_spec.features:
        if not feature.enabled or not isinstance(feature.params, dict):
            continue
        try:
            feature_type = em_type.feature_type(feature.type)
        except KeyError:
            continue
        if feature_type.category != 'source':
            continue
        source_values = vctx.values(feature_type.params, feature.params)
        if not source_values or source_values.get('waveform') != 'cw':
            continue
        wavelength = source_values.get('wavelength')
        if not isinstance(wavelength, (int, float)) or not wavelength > 0:
            continue
        half_period_steps = wavelength / (2.0 * C0 * em_dt)
        omega_window = math.pi * every / half_period_steps
        window_error = abs(math.sin(omega_window)) / omega_window
        if worst is None or window_error > worst[0]:
            worst = (window_error, feature.name, half_period_steps)
    if worst is None or worst[0] <= _CW_RIPPLE_WARNING:
        return []
    window_error, source_name, half_period_steps = worst
    snapped_every = max(1, round(max(1, round(every / half_period_steps)) * half_period_steps))
    return [Issue('warning', path,
                  f"every = {every} EM steps is not a whole number of half periods ({half_period_steps:.4g} steps) "
                  f"of cw source '{source_name}': the window-averaged Joule power is off by up to "
                  f'{100 * window_error:.2g} %, varying from cell to cell with the optical phase (set the heat '
                  f'every to {snapped_every}, a whole number of half periods, or to many periods)')]


def _validate_joule(vctx, coupling_spec, values):
    """Warnings: every differing from the heat every, pulsed EM feeding a steady heat solve, transient heat target,
    duration shorter than one window, cw ripple of a window that is not a whole number of half periods."""
    path = ('coupling', coupling_spec.name)
    em_spec, em_type = physics_and_type(vctx, coupling_spec.source)
    heat_spec, heat_type = physics_and_type(vctx, coupling_spec.target)
    if em_spec is None or heat_spec is None:
        return []
    heat_values = vctx.values(heat_type.settings, heat_spec.settings) or {}   # bad settings: param checks report
    issues = []
    heat_mode = heat_values.get('time_mode') if vctx.study_type == TIME_DEPENDENT_STUDY else 'stationary'
    if heat_mode == 'stationary':
        pulsed = _pulsed_sources(em_spec, em_type)
        if pulsed:
            issues.append(Issue('warning', path,
                                f"heat '{heat_spec.name}' is solved quasi-statically while EM sources "
                                f"{', '.join(pulsed)} are not 'cw': a steady solve of a pulse's window-averaged "
                                'power is not physical (use waveform cw)'))
    elif heat_mode == 'transient' and em_type.steps_in_time is not None:   # FDFD: stroboscopic, heat time scales
        issues.append(Issue('warning', path,
                            f"heat '{heat_spec.name}' is transient: EM advances in femtosecond steps, so transient "
                            'heat barely moves during the run (use time_mode stationary)'))
    duration = study_duration(vctx)
    every = effective_coupling_every(coupling_spec, em_spec, heat_spec)
    if every is not None and valid_every(em_spec.every) and valid_every(heat_spec.every) and every != heat_spec.every:
        issues.append(Issue('warning', path,
                            f"every = {every} differs from the every = {heat_spec.every} of the heat physics "
                            f"'{heat_spec.name}': the absorbed power is averaged over a window that does not match "
                            f'the heat solves (leave every blank or set it to {heat_spec.every})'))
    # the EM Courant step is the tick only for a time-stepped source (FDFD in time_dependent: another physics sets it)
    em_dt = getattr(vctx.mesh, 'dt', None) if em_type.steps_in_time is not None else None
    if duration is not None and every is not None and em_dt and duration < every * em_dt:
        issues.append(Issue('warning', path,
                            f'study duration {duration:.4g} s is shorter than every * dt_EM = {every} * '
                            f"{em_dt:.4g} s: the Joule power reaches '{heat_spec.name}' only in the final flush"))
    if vctx.study_type == TIME_DEPENDENT_STUDY and every is not None and em_dt:
        issues.extend(_cw_ripple_issues(vctx, path, em_spec, em_type, every, em_dt))
    return issues + absorbed_power_reader_issues(vctx, coupling_spec)


def _validate_field_transfer(vctx, coupling_spec, values):
    """source_field must be a state field of the source type, target_slot a source slot of the target type (other
    writers: the writer rule of model.validate); warning for the accumulated q_accum."""
    path = ('coupling', coupling_spec.name)
    issues = []
    source_spec, source_type = physics_and_type(vctx, coupling_spec.source)
    if source_type is not None:
        state_fields = [info.name for info in source_type.state_fields]
        if values['source_field'] not in state_fields:
            issues.append(Issue('error', path,
                                f"source_field {values['source_field']!r} is not a state field of physics "
                                f"'{source_spec.name}' ({source_type.name}); valid: {', '.join(state_fields) or 'none'}"))
        elif source_type.family == _EM_FAMILY and values['source_field'] == 'q_accum':
            issues.append(Issue('warning', path,
                                "q_accum is the J.E energy density [J/m^3] accumulated since the last em_heat_joule "
                                'transfer, not a power: only em_heat_joule averages and resets it (without one it '
                                'grows through the whole run); use em_heat_joule for the absorbed power density'))
    target_spec, target_type = physics_and_type(vctx, coupling_spec.target)
    if target_type is not None:
        if values['target_slot'] not in target_type.source_slots:
            issues.append(Issue('error', path,
                                f"target_slot {values['target_slot']!r} is not a source slot of physics "
                                f"'{target_spec.name}' ({target_type.name}); valid: "
                                f"{', '.join(target_type.source_slots) or 'none'}"))
    return issues


def _dispersive_thermo_optic_issues(vctx, slots, path):
    """Warning per selected dispersive material with a non-zero dndT of a written slot (or alpha, with a Lorentz
    pole): the transforms shift only the background permittivity, never the pole susceptibility."""
    parameters = vctx.parameters
    dndT_names = sorted({DNDT_ARRAYS[slot] for slot in slots})
    issues = []
    for material, properties, poles in selected_material_data(vctx.spec):
        if not poles:
            continue
        names = [name for name in dndT_names
                 if evaluates_to(properties.get(name, properties.get('dndT')), parameters, lambda value: value != 0.0)]
        if any(isinstance(pole, dict) and pole.get('model') == 'lorentz' for pole in poles):
            names += [name for name in ('alpha_xx', 'alpha_yy')
                      if evaluates_to(properties.get(name, properties.get('alpha')), parameters,
                                      lambda value: value != 0.0)]
        if names:
            issues.append(Issue('warning', path,
                                f"material '{material.name}' is dispersive and sets {', '.join(names)}: the "
                                'thermo-optic transfer changes only its background permittivity (n0 = sqrt of eps_r / '
                                'eps_inf), never the pole susceptibility, so the shift is too small by about '
                                'n(omega) / n0 (scale dndT by that ratio at the wavelength of interest)'))
    return issues


def _validate_thermo_optic(vctx, coupling_spec, values):
    """Error when the target type has no index slot the thermo-optic transforms write (other writers: the writer rule
    of model.validate); warnings for dispersive materials with dndT / alpha."""
    path = ('coupling', coupling_spec.name)
    target_spec, target_type = physics_and_type(vctx, coupling_spec.target)
    if target_type is None:
        return []
    slots = index_slots(target_type)
    if slots:
        return _dispersive_thermo_optic_issues(vctx, slots, path)
    return [Issue('error', path,
                  f"target physics '{target_spec.name}' ({target_type.name}) has none of the thermo-optic slots "
                  f"{', '.join(THERMO_OPTIC_SLOTS)}; its source slots: "
                  f"{', '.join(target_type.source_slots) or 'none'}")]


# ---- types ----
def _currents_joule_type():
    """currents_heat_joule: the q_joule of an electric_currents physics copied into heat's volumetric_power."""
    return CouplingType(
        'currents_heat_joule', 'Joule heating (electric currents -> heat)', CURRENTS_TYPE_NAME, HEAT_TYPE_NAME,
        summary='Turns the resistive heating of the electric currents into a heat source of the heat physics. It adds '
                'up with the other heating couplings of that heat physics, such as absorbed light.',
        description='Resistive heating: q_joule [W/m^3] of the last electric_currents solve (J.E per cell: each '
                    "face's dissipation split between its two cells by their half-cell resistances, outer Dirichlet "
                    'faces to their cell) becomes the heat source Q (heat ignores the PML ring). '
                    + HEAT_SOURCE_SUM_TEXT + ' Its window integral equals the sum of V * current over the voltage / '
                    'terminal features exactly (planar: per m of depth). Fixed point currents -> Q -> T -> '
                    'sigma_dc(T): give the material a sigma_dc expression of heat_T (no coupling needed; the currents '
                    're-solve after each heat step that changes it). Time-dependent study: leave every null or equal '
                    'to the heat every and the currents every at 1. Transient heat: each step uses the q_joule at its '
                    'start temperature; the first step has no Joule source.',
        create=_create_currents_joule,
        writes=writes_heat_source,
        mesh_transfer='value')


def _coupling_types(em_types, absorbing_em_types, has_currents):
    """Coupling types for the registered EM type names (sorted tuples; em_types targets of heat_em_thermo_optic,
    absorbing_em_types sources of em_heat_joule; the EM couplings only when they are non-empty) and, with has_currents,
    currents_heat_joule; field_transfer always."""
    field_transfer = CouplingType(
        'field_transfer', 'Field transfer', None, None,
        params=(
            Param('source_field', 'str', '', label='Source field',
                  help='State field of the source physics read at every transfer (full grid, PML ring included), '
                       'e.g. T of heat. EM q_accum is an accumulated energy density, not a power: use em_heat_joule.',
                  tip='Field of the source physics that is read at every transfer, for example the temperature of a '
                      'heat physics.'),
            Param('target_slot', 'str', '', label='Target input',
                  help='Source slot of the target physics, overwritten at every transfer: no other coupling may '
                       'write it, not even an additive one (validation error).',
                  tip='Input of the target physics that the result replaces at every transfer, for example the heat '
                      'source of a heat physics.\n\nNo other coupling may write the same input.'),
            Param('expression', 'expression', 'value', label='Transform expression',
                  help='Of value (the source field), x, y (3D: z; cell centres, PML cells take the nearest window '
                       'centre) and parameters, e.g. 1e-3*value; no other fields, no t. Written as float32 into one '
                       'persistent full-grid buffer (a constant fills it).',
                  tip='Expression applied to the source field, which it calls value; it may also use the position x, '
                      'y (and the height z in 3D models) and model parameters.\n\nThe default, value, copies the '
                      'field unchanged.')),
        summary='Copies a field of one physics, optionally transformed by an expression, into an input of another '
                'physics.',
        description='Generic transfer: the expression is written into target_slot of the target physics, replacing '
                    'it (heat volumetric_power replaces Q; the EM permittivity slots replace their whole array, PML '
                    'included; EM ez_source / hz_source add the buffer to E_z / H_z every EM step until the next '
                    'transfer: a raw V/m or A/m increment per step, so its effect scales with 1/dt). Slot units: the '
                    "target physics type's source slots.",
        create=_create_field_transfer, validate=_validate_field_transfer, writes=_writes_field_transfer,
        mesh_transfer='field')
    currents_types = (_currents_joule_type(),) if has_currents else ()
    if not em_types:
        return (*currents_types, field_transfer)
    em_heat_joule = CouplingType(
        'em_heat_joule', 'Joule heating (electromagnetic waves -> heat)', absorbing_em_types, HEAT_TYPE_NAME,
        summary='Turns the light absorbed by a wave physics into a heat source of the heat physics. With a '
                "frequency-domain wave physics, use a 'Stationary' study for one wavelength or a 'Frequency domain' "
                'study for a list of them.',
        description='The absorbed power density J.E [W/m^3] (sigma_e conduction plus Drude / Lorentz currents) of '
                    'the EM physics (take_absorbed_power: the mean over the EM steps since the last transfer; FDFD: '
                    'the time average of its last solve) becomes the heat source Q, e.g. E_z plus H_z physics for '
                    'unpolarised light. ' + HEAT_SOURCE_SUM_TEXT + ' FDFD: a stationary study for one wavelength, a '
                    'frequency_domain study for a list (each wavelength a stationary fixed point of every physics, '
                    'continued from the previous one; no em_modes or time-domain physics beside it). em_fdtd_hz puts '
                    'each E_x / E_y face loss into its lossy neighbour cells (sigma_e by conductivity, poles to the '
                    "dispersive cells); em_fdfd_hz uses each cell's loss with |E_x|^2, |E_y|^2 averaged over its two "
                    'faces. Time-stepped EM: leave every null or equal to the heat every (validation warns otherwise); '
                    'the final flush sends the last partial window merged with the previous full one (a study shorter '
                    'than every * dt_EM delivers its power only there). A cw window W = every * dt_EM averages cos^2 '
                    'exactly only over whole half optical periods, else up to T_opt / (2 pi W) off, varying per cell '
                    '(validation warns above 1 % naming the nearest whole-half-period every). Transient heat lags one '
                    'window: its step over [kW, (k+1)W] uses the power of [(k-1)W, kW] (no source in the first, the '
                    'last window never stepped). Pulsed EM sources feeding steady heat, transient heat or such a short '
                    'study: validation warning. A second reader of the absorbed power of the same time-stepped EM '
                    'physics (em_heat_joule, em_semiconductor_generation): validation error (each transfer resets '
                    'the window).',
        create=_create_em_joule,
        validate=_validate_joule,
        writes=writes_heat_source,
        mesh_transfer='value', reads_absorbed_power=True)
    heat_em_thermo_optic = CouplingType(
        'heat_em_thermo_optic', 'Thermo-optic effect (heat -> electromagnetic waves)', HEAT_TYPE_NAME, em_types,
        summary='Updates the refractive index of a wave physics from the temperature of the heat physics. It uses '
                "each material's thermo-optic and thermal expansion coefficients.",
        description='Each transfer rewrites the EM permittivity from the heat temperature: dT = T - T0 (T0 of the '
                    'heat physics), n0 = sqrt of the built permittivity component (dispersive: of eps_inf). '
                    + INDEX_SUM_TEXT + ' Alone: E_z types eps_zz = (n0_zz + dndT_zz dT)^2 (1 + alpha_xx dT)(1 + '
                    'alpha_yy dT) (eps_r_update); H_z types eps_xx and eps_yy likewise with n0_xx / dndT_xx and n0_yy '
                    '/ dndT_yy (eps_xx_update / eps_yy_update); em_modes all three, the alpha factor on eps_zz only '
                    '(eps_xx = (n0_xx + dndT_xx dT)^2; transverse to the propagation the exact stretch image also '
                    'scales mu_zz, which cannot be held, so the transverse expansion is only partly counted; the axial '
                    'path term n_eff alpha_zz dT is post-processing). The alpha factor maps the in-plane expansion '
                    'onto the fixed grid (a structure stretched by s_x s_y is a medium with eps_zz s_x s_y), so the '
                    'optical path grows as the real one; H_z: only the phase index matches (the exact image scales '
                    'mu_zz). Per cell it is first order, exact only for alpha uniform over the mode: cells with alpha 0 (cladding, '
                    'air) miss their share, under-counting the expansion shift by the mode energy fraction outside the '
                    'expanding material. An enabled solid_em_strain_optic (deformation on) into the same EM physics '
                    'drops the alpha factor (the solid strain supplies the deformation). Dispersive materials: the '
                    'pole susceptibility never changes, so dndT shifts eps by 2 n0 dndT dT instead of 2 n(omega) dndT '
                    'dT (scale dndT by n(omega) / n0) and alpha scales only the background; validation warns for a '
                    'dndT on them or an alpha with a Lorentz pole. An isotropic dndT sets dndT_xx, _yy and _zz; no '
                    'off-diagonal dndT / alpha. time_dependent (every = the heat every): right after each heat step, '
                    "so one heat interval's EM steps see the temperature of the previous interval's power; stationary: "
                    'per fixed-point iteration; frequency_domain: once after the heat solve, before the wavelengths '
                    '(a coupled wavelength scan: per iteration of each wavelength); mode_analysis: once before the '
                    'mode solve.',
        create=_create_thermo_optic,
        validate=_validate_thermo_optic,
        writes=writes_index_slots,
        mesh_transfer='field', partner_materials=True)   # the transform reads heat's dndT / alpha arrays
    return (em_heat_joule, heat_em_thermo_optic, *currents_types, field_transfer)


def _selftest_cases():
    """The coupled-chain SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_backend import CHAIN_BACKEND_CASES
    from .selftests_mesh_transfer import MESH_TRANSFER_CASES
    from .selftests_slot_sum import SLOT_SUM_CASES
    return (*SELFTEST_CASES, *SLOT_SUM_CASES, *CHAIN_BACKEND_CASES, *MESH_TRANSFER_CASES)


@dataclass(frozen=True)
class RegisteredPhysics:
    """The registered physics types as the coupling definition modules read them (sorted name tuples)."""
    names: frozenset
    em_physics_types: tuple        # PhysicsType of family 'em', registration order
    em_types: tuple
    absorbing_em_types: tuple      # with an 'absorbed_power' output field
    # frequency-domain EM types that may join a time_dependent study (em_fdfd_*): the acoustics optics couplings
    optical_types: tuple
    absorbing_optical_types: tuple


def registered_physics(registry):
    """RegisteredPhysics of the physics types registered so far (all of them: couplings register last)."""
    physics_types = registry.physics_types()
    em_physics_types = tuple(physics_type for physics_type in physics_types if physics_type.family == _EM_FAMILY)
    absorbing = {physics_type.name for physics_type in em_physics_types
                 if any(info.name == 'absorbed_power' for info in physics_type.output_fields)}
    optical_types = tuple(sorted(physics_type.name for physics_type in em_physics_types
                                 if physics_type.steps_in_time is None
                                 and TIME_DEPENDENT_STUDY in physics_type.supported_studies))
    return RegisteredPhysics(
        names=frozenset(physics_type.name for physics_type in physics_types), em_physics_types=em_physics_types,
        em_types=tuple(sorted(physics_type.name for physics_type in em_physics_types)),
        absorbing_em_types=tuple(sorted(absorbing)), optical_types=optical_types,
        absorbing_optical_types=tuple(name for name in optical_types if name in absorbing))


def _extension_module_names():
    """Sorted coupling/definitions_*.py module names (each has register(registry, physics))."""
    package = importlib.import_module(__package__)
    return sorted(f'{__package__}.{info.name}' for info in pkgutil.iter_modules(package.__path__)
                  if info.name.startswith(_EXTENSION_PREFIX) and not info.ispkg)


def register(registry):
    """Register the coupling types (heat_em_thermo_optic accepts every registered physics type of family 'em',
    em_heat_joule those with an 'absorbed_power' output field; currents_heat_joule only when electric_currents is
    registered), the coupled-chain selftest loader, then register(registry, physics) of every definitions_*.py in
    name order, each registering only what its partner physics allow (they import this module's E13 helpers, so they
    are imported here, never at module level)."""
    physics = registered_physics(registry)
    for coupling_type in _coupling_types(physics.em_types, physics.absorbing_em_types,
                                         CURRENTS_TYPE_NAME in physics.names):
        registry.register_coupling(coupling_type)
    registry.register_selftests(_selftest_cases)
    for module_name in _extension_module_names():
        module = importlib.import_module(module_name)
        register_extension = getattr(module, 'register', None)
        if register_extension is None:
            raise AttributeError(f'coupling definition module {module_name} has no register(registry, physics)')
        register_extension(registry, physics)

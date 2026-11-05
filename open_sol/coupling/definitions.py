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
from ..model.schema import CouplingType, Issue, Param, effective_coupling_every, valid_every
from .coupling_term import CouplingTerm
from .index_terms import after_source_step, index_slots, physics_and_type, writes_index_slots
from .standard import THERMO_OPTIC_SLOTS, joule_heating, thermo_optic

HEAT_TYPE_NAME = 'heat'
CURRENTS_TYPE_NAME = 'electric_currents'   # = physics.potential.definition.CURRENTS_TYPE_NAME (not imported here)
# EM loss protocol (every solver of a family-'em' type): state()['q_accum'] always present (None allowed); a settable
# getattr(solver, 'engine', solver).enable_q_accum; take_absorbed_power(merge_previous=False) -> mean W/m^3 since the
# last call (merge_previous: also over the last completed window) or None.
_EM_FAMILY = 'em'
_TRANSFER_NAMES = FIELD_COORDINATE_NAMES | {'value'}   # a field_transfer expression's names besides parameters
_CW_RIPPLE_WARNING = 0.01   # relative cell-to-cell error of a CW Joule window mean above which validation warns
_THERMO_OPTIC_DNDT = {'eps_r_update': 'dndT_zz', 'eps_xx_update': 'dndT_xx', 'eps_yy_update': 'dndT_yy'}
# EM types whose plane is a cross-section transverse to the propagation (thermo-optic expansion on eps_zz only)
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
    heat_mode = heat_values.get('time_mode') if vctx.study_type == 'time_dependent' else 'stationary'
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
    if vctx.study_type == 'time_dependent' and every is not None and em_dt:
        issues.extend(_cw_ripple_issues(vctx, path, em_spec, em_type, every, em_dt))
    return issues


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
    dndT_names = sorted({_THERMO_OPTIC_DNDT[slot] for slot in slots})
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
        description='Resistive (Joule) heating: q_joule in W/m^3 of the last electric_currents solve (J.E per cell: '
                    'the dissipation of every face split between its two cells by their half-cell resistances, '
                    'outer Dirichlet faces to their cell; a DC solve is time-averaged by construction) becomes the heat source Q '
                    '(volumetric_power; heat ignores the edge-extended PML ring). It is an additive contribution: Q '
                    '= the sum of every em_heat_joule / currents_heat_joule / fluid_heat_convection / '
                    'magnetic_heat_induction coupling into that heat physics, all with '
                    'the same every (validation error otherwise); a field_transfer into volumetric_power cannot be '
                    'combined with them. '
                    'Its integral over the window equals the sum over the voltage / terminal features of V * current exactly, so the '
                    'derived integral of currents_q_joule is the electrical power per m of depth. The stationary '
                    'fixed point is currents -> Q -> T -> sigma_dc(T) -> currents: a temperature-dependent '
                    'conductivity is a sigma_dc expression of heat_T on the material (no coupling needed; the '
                    'currents re-solve whenever it is rewritten, right after each heat step). In a time-dependent '
                    'study leave every null or equal to the heat every, and the currents every at 1 (it re-solves '
                    'only when something changed), so the q_joule sent before each heat step already uses the '
                    'sigma_dc(T) of the previous one. Transient heat: each step uses the q_joule of the temperature '
                    'at its start (explicit in the source), except the first step, which has no Joule source (the '
                    'first transfer follows it).',
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
                  help="State field of the source physics read at every transfer (full grid, PML ring included); "
                       "valid names are the source type's state fields, listed by describe (e.g. T of heat). EM "
                       'q_accum is an accumulated energy density, not a power: use em_heat_joule for that.',
                  tip='Field of the source physics that is read at every transfer, for example the temperature of a '
                      'heat physics.'),
            Param('target_slot', 'str', '', label='Target input',
                  help="Source slot of the target physics written at every transfer; valid names are the target "
                       "type's source slots, listed by describe. Every transfer overwrites the slot, so no other "
                       'coupling may write it, not even an additive one such as em_heat_joule or heat_em_thermo_optic '
                       '(validation error).',
                  tip='Input of the target physics that the result replaces at every transfer, for example the heat '
                      'source of a heat physics.\n\nNo other coupling may write the same input.'),
            Param('expression', 'expression', 'value', label='Transform expression',
                  help='Expression of value (the source field), x, y (in 3D models also z; cell centres, PML cells get '
                       'the nearest window centre) and parameters, e.g. 1e-3*value; no other field names and no t. '
                       'The result is written as float32 into one persistent full-grid buffer (a constant fills the '
                       'grid).',
                  tip='Expression applied to the source field, which it calls value; it may also use the position x, '
                      'y (and the height z in 3D models) and model parameters.\n\nThe default, value, copies the '
                      'field unchanged.')),
        summary='Copies a field of one physics, optionally transformed by an expression, into an input of another '
                'physics.',
        description='Generic transfer: at every transfer the expression of value (source_field of the source '
                    'physics), x, y (3D: z) and parameters is evaluated into one persistent float32 full-grid buffer '
                    'and written into target_slot of the target physics. Slots overwrite (heat volumetric_power '
                    'replaces Q; the EM permittivity slots eps_r_update / eps_xx_update / eps_yy_update replace their '
                    'whole array, PML included; EM ez_source / hz_source keep the buffer and add it to E_z / H_z '
                    'every EM step until the next transfer: a raw V/m or A/m increment per step, so its effect '
                    'scales with 1/dt), so no other coupling may write that slot, not even an additive one '
                    '(e.g. em_heat_joule, currents_heat_joule, fluid_heat_convection, magnetic_heat_induction, '
                    'heat_em_thermo_optic, em_solid_optical_force). Slot '
                    'units: describe, per physics '
                    'type.',
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
        description='Joule heating: the absorbed power density J.E in W/m^3 (sigma_e conduction plus Drude / '
                    f"Lorentz currents) of an EM physics type with an absorbed_power output "
                    f"({', '.join(absorbing_em_types)}; not em_modes, which solves eigenmodes without a source), "
                    'taken through its '
                    'take_absorbed_power() as the mean over the EM steps since the last transfer (an FDFD physics: '
                    'the time-averaged power density of its last solve), becomes the heat source Q '
                    '(volumetric_power) as an additive contribution: Q = the sum of every em_heat_joule / '
                    'currents_heat_joule / fluid_heat_convection / magnetic_heat_induction coupling into that heat '
                    'physics (e.g. E_z and H_z '
                    'physics for unpolarised light, or optical plus resistive heating), all with the same every '
                    '(validation error otherwise); '
                    'a field_transfer into volumetric_power cannot be combined with them. With FDFD use a stationary '
                    'study for one wavelength, or a frequency_domain study for a list (coupled wavelength scan: each '
                    'wavelength a stationary fixed point of every physics, continued from the previous one in list '
                    'order; no em_modes or time-domain physics beside it). em_fdtd_hz '
                    'evaluates the loss on the cell faces that carry E_x / E_y and assigns each face loss to its lossy '
                    'neighbour cell(s): the sigma_e part by conductivity, the Drude / Lorentz part to the dispersive '
                    "cell(s); em_fdfd_hz uses each cell's own loss with |E_x|^2 and |E_y|^2 averaged over its two "
                    'faces. Leave every null or equal to the heat every so each heat solve gets exactly the steps '
                    'since the previous one (validation warns otherwise); runner.flush() at the end of the run sends '
                    'the mean over the last partial window merged with the previous full one, so the final Q does '
                    'not depend on where the run stops within a window (a study shorter than every * dt_EM has no '
                    'full window and delivers its power only in that flush). A cw window W = every * dt_EM averages '
                    'the cos^2 power exactly only over a whole number of half optical periods, else the mean is off by '
                    'up to T_opt / (2 pi W), varying from cell to cell with the phase (validation warns above 1 % and '
                    'names the nearest whole-half-period every). Transient heat lags one window: its step over '
                    '[kW, (k+1)W] uses the power of [(k-1)W, kW], so the first interval has no source and the last '
                    'window (and the flushed final Q) is never stepped (heat gets (K-1)/K of the energy of K windows). '
                    'Pulsed (non-cw) EM sources feeding a '
                    'steady heat solve, transient heat or such a short study give a validation warning.',
        create=_create_em_joule,
        validate=_validate_joule,
        writes=writes_heat_source,
        mesh_transfer='value')
    heat_em_thermo_optic = CouplingType(
        'heat_em_thermo_optic', 'Thermo-optic effect (heat -> electromagnetic waves)', HEAT_TYPE_NAME, em_types,
        summary='Updates the refractive index of a wave physics from the temperature of the heat physics. It uses '
                "each material's thermo-optic and thermal expansion coefficients.",
        description='Thermo-optic effect: each transfer rewrites the permittivity arrays of the EM physics from the '
                    'heat temperature T, with dT = T - T0 (T0 of the heat physics) and n0 = sqrt of the built '
                    'permittivity component (the background eps_inf for dispersive materials), recorded before the '
                    'first transfer. It is an additive index contribution: it hands dn = n - n0 per component, and '
                    'eps = (n0 + the sum of dn of every index coupling into that slot)^2 (e.g. two heat physics, or '
                    'later electro-optic couplings), all with the same every (validation error otherwise); alone it '
                    'gives the formulas below. E_z types (em_fdtd_ez, em_fdfd_ez): eps_zz = (n0_zz + '
                    'dndT_zz dT)^2 (1 + alpha_xx dT)(1 + alpha_yy dT) through eps_r_update. H_z types (em_fdtd_hz, '
                    'em_fdfd_hz): eps_xx = (n0_xx + dndT_xx dT)^2 (1 + alpha_xx dT)(1 + alpha_yy dT) through '
                    'eps_xx_update, and eps_yy likewise with n0_yy and dndT_yy through eps_yy_update. em_modes (a '
                    'cross-section, E along x, y and z): all three, eps_zz through eps_r_update, eps_xx and eps_yy '
                    'through their slots; there the plane is transverse to the propagation, and the exact image of a '
                    'transverse stretch leaves eps_xx / eps_yy unchanged (isotropic alpha) and scales eps_zz and mu_zz '
                    'by s_x s_y, so the alpha factor goes on eps_zz only (eps_xx = (n0_xx + dndT_xx dT)^2): mu_zz '
                    'cannot be held, so the transverse expansion is only partly counted, and the axial path term '
                    '(n_eff alpha_zz dT on the phase per length) is a post-processing factor. The alpha factor '
                    'is the in-plane thermal expansion mapped onto the fixed grid (transformation optics: a structure '
                    'stretched by s_x s_y is a medium with eps_zz s_x s_y), so n grows by about (1 + alpha dT) and the '
                    'optical path as the real structure does; for H_z the exact image scales mu_zz instead, so only the '
                    'phase index matches there. Applied per cell it is a first-order local approximation, exact only '
                    'for an alpha uniform over the mode: cells with alpha 0 (cladding, air) miss their share of the '
                    'stretch, so the expansion shift is under-counted by the mode energy fraction outside the '
                    'expanding material. With an enabled solid_em_strain_optic (deformation on) into the same EM '
                    'physics the alpha factor is dropped (dn = dndT dT): the solid strain then supplies the '
                    'deformation, constraints and cladding included. Dispersive materials: the Drude / Lorentz susceptibility is never changed, '
                    'so dndT shifts eps by 2 n0 dndT dT instead of 2 n(omega) dndT dT (too small by about n(omega) / '
                    'n0; scale dndT by that ratio at the wavelength of interest) and alpha scales only the background '
                    '(negligible for metals); validation warns for a dndT on them, or an alpha with a Lorentz pole. '
                    'An isotropic dndT sets dndT_xx, dndT_yy and dndT_zz; off-diagonal dndT / alpha terms are not '
                    'modelled. Each '
                    'slot overwrites its whole array including the PML, so it cannot be combined with a field_transfer '
                    'into that slot or a state-dependent permittivity property (validation error). Time-dependent study '
                    '(every = the heat every): applied right after each heat step, so the EM steps of one heat '
                    'interval see the temperature of the absorbed power of the previous one; stationary study: one '
                    'fixed-point iteration; frequency-domain study: applied once after the heat solve, before the '
                    'wavelengths (a coupled wavelength scan, e.g. with em_heat_joule from that EM physics: in every '
                    'fixed-point iteration of each wavelength, as in the stationary study); mode_analysis study: '
                    'applied once after the heat solve, before the mode solve.',
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
                                 and 'time_dependent' in physics_type.supported_studies))
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

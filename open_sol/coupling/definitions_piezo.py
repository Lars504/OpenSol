"""Coupling type electrostatics_acoustics_piezo (piezoelectric drive: the electrostatics E into the elastic_waves
electric_field_x / _y slots; two-way by default: the bound charge of the strain back into the electrostatics, re-solved
inside each transfer), its self-tests and the 'yz_ln_saw_idt' / 'tfln_acousto_optic_modulator' templates; discovered
and registered by definitions.register when elastic_waves and electrostatics are."""
from ..model.definition_helpers import is_axisymmetric, selected_material_sets
from ..model.schema import CouplingType, Issue, Param, coupling_writes, effective_coupling_every
from ..physics.acoustics.couplings import bound_charge_transform
from .coupling_term import CouplingGroup, CouplingTerm
from .index_terms import after_source_step, physics_and_type
from .selftests_piezo import (TFLN_AOM_DESCRIPTION, TFLN_AOM_LABEL, TFLN_AOM_SUMMARY, YZ_LN_SAW_IDT_DESCRIPTION,
                              YZ_LN_SAW_IDT_LABEL, YZ_LN_SAW_IDT_SUMMARY, tfln_acousto_optic_modulator_spec,
                              yz_ln_saw_idt_spec)

PIEZO_TYPE_NAME = 'electrostatics_acoustics_piezo'   # = selftests_piezo.PIEZO_TYPE_NAME (not imported)
ACOUSTICS_TYPE_NAME = 'elastic_waves'      # = physics.acoustics.definition.PHYSICS_TYPE_NAME
ELECTROSTATICS_TYPE_NAME = 'electrostatics'
FIELD_SLOTS = {'E_x': 'electric_field_x', 'E_y': 'electric_field_y'}   # = physics.acoustics.solver.ELECTRIC_FIELD_SLOTS
PIEZO_NAMES = ('piezo_e11', 'piezo_e12', 'piezo_e16', 'piezo_e21', 'piezo_e22', 'piezo_e26')   # = solver.PIEZO_ARRAYS
CHARGE_SLOT = 'charge_density'             # = the electrostatics KindSpec slot
PERMITTIVITY_NAMES = ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_xy')   # electrostatics state(): relative coefficient arrays
BACK_ACTIONS = ('bound_charge', 'none')
_AOM_OPTICAL_TYPE = 'em_fdfd_hz'   # the tfln_acousto_optic_modulator template's optical physics


class _CoupledResolve:
    """Group member of the two-way coupling: re-solves the electrostatics on the bound charge just injected, at the
    acoustics' current time, so the field terms after it hand over E of this very step."""

    contribution = ''

    def __init__(self, electrostatics, acoustics, name):
        self.target = electrostatics           # an attached physics (runner check); writes no slot
        self.acoustics = acoustics
        self.name = name
        self.source_fields = frozenset()

    def _source_changed(self, physics_by_name):
        return False                           # the group's charge term decides

    def apply_now(self, physics_by_name):
        physics_by_name[self.target].solve_coupled_source(physics_by_name[self.acoustics].time)

    flush = apply_now


# ---- create (worker thread, GPU) ----
def _create_piezo(ctx, coupling_spec, values, every):
    """CouplingGroup of the two field terms (the acoustics applies -e^T dE at each), right after the electrostatics
    steps when their strides match; two-way: bound charge -> electrostatics re-solve -> the field terms, and the
    stiffened acoustic dt."""
    transfers_after_source = after_source_step(ctx, coupling_spec, every)
    terms = [CouplingTerm(coupling_spec.source, field, coupling_spec.target, slot, every=1,
                          name=f'{coupling_spec.name}.{slot}', after_source_step=transfers_after_source)
             for field, slot in FIELD_SLOTS.items()]
    acoustics = ctx.sim.physics[coupling_spec.target]
    if values['back_action'] == 'bound_charge' and acoustics.piezoelectric:
        state = ctx.sim.physics[coupling_spec.source].state()
        acoustics.enable_bound_charge(*(state[name] for name in PERMITTIVITY_NAMES))
        charge = CouplingTerm(coupling_spec.target, 'sxx', coupling_spec.source, CHARGE_SLOT,
                              transform=bound_charge_transform, every=1, name=f'{coupling_spec.name}.{CHARGE_SLOT}',
                              after_source_step=transfers_after_source)
        terms = [charge, _CoupledResolve(coupling_spec.source, coupling_spec.target,
                                         f'{coupling_spec.name}.re_solve'), *terms]
    return CouplingGroup(terms, every=every, name=coupling_spec.name, after_source_step=transfers_after_source)


def _writes_fields(vctx, coupling_spec):
    return dict.fromkeys(FIELD_SLOTS.values(), '')


# ---- validation (CPU only) ----
def _physics_axisymmetric(vctx, physics_name):
    physics_spec, physics_type = physics_and_type(vctx, physics_name)
    return physics_type is not None and is_axisymmetric(vctx.values(physics_type.settings, physics_spec.settings) or {})


def _two_way_issues(vctx, coupling_spec, path):
    """Two-way rules: the coupling's every equals the acoustics' (one re-solve per acoustic step), no other writer of
    the electrostatics charge_density, and the cost info."""
    issues = []
    try:
        source_spec = vctx.spec.physics_by_name(coupling_spec.source)
        target_spec = vctx.spec.physics_by_name(coupling_spec.target)
    except KeyError:
        return issues
    every = effective_coupling_every(coupling_spec, source_spec, target_spec)
    if every is not None and every != target_spec.every:
        issues.append(Issue('error', path, f"back_action bound_charge: every {every} must equal the every of "
                                           f"'{target_spec.name}' ({target_spec.every}): the bound charge of each "
                                           'acoustic step is re-solved before the next one (leave every blank)'))
    for other in vctx.spec.couplings:
        if other.enabled and other.name != coupling_spec.name and other.target == coupling_spec.source \
                and CHARGE_SLOT in coupling_writes(vctx, other):
            issues.append(Issue('error', path, f"back_action bound_charge writes '{CHARGE_SLOT}' of "
                                               f"'{coupling_spec.source}', which coupling '{other.name}' also writes; "
                                               "keep one, or set back_action 'none'"))
    issues.append(Issue('info', path,
                        f"two-way: after every step of '{coupling_spec.target}' the bound charge -div(e S) of its "
                        f"strain re-solves '{coupling_spec.source}' (one CG solve per acoustic step) and E goes back "
                        'as the stress -e^T E: piezoelectric stiffening included, the acoustic dt uses the stiffened '
                        'c^D = c^E + e^T eps^-1 e'))
    return issues


def _validate_piezo(vctx, coupling_spec, values):
    """Error for an axisymmetric partner, warning without a piezoelectric material, the two-way rules or the one-way
    info."""
    path = ('coupling', coupling_spec.name)
    axisymmetric = [name for name in (coupling_spec.source, coupling_spec.target) if _physics_axisymmetric(vctx, name)]
    if axisymmetric:
        return [Issue('error', path, f"{', '.join(axisymmetric)}: geometry axisymmetric; the piezoelectric coupling "
                                     'is planar (sagittal plane) only')]
    issues = []
    if not selected_material_sets(vctx, PIEZO_NAMES):
        issues.append(Issue('warning', path, "no selected material sets piezo_e* entries (or a {'model': 'crystal'} "
                                             'entry): the coupling stresses nothing'))
    if values['back_action'] == 'bound_charge':
        return issues + _two_way_issues(vctx, coupling_spec, path)
    issues.append(Issue('info', path,
                        f"one-way (back_action none): '{coupling_spec.target}' takes the stress -e^T E of the field "
                        f"of '{coupling_spec.source}'; no piezoelectric stiffening (wave speeds ~K^2 / 2 low, ~2 % for "
                        'a YZ lithium niobate SAW) and no bound charge back to the electrostatics'))
    return issues


# ---- type, registration ----
PIEZO_COUPLING = CouplingType(
    PIEZO_TYPE_NAME, 'Piezoelectric coupling (electrostatics <-> elastic waves)', ELECTROSTATICS_TYPE_NAME,
    ACOUSTICS_TYPE_NAME,
    params=(
        Param('back_action', 'choice', 'bound_charge', label='Back-action on the field', choices=BACK_ACTIONS,
              choice_labels=(('bound_charge', 'Two-way (bound charge, stiffening)'),
                             ('none', 'One-way (drive only)')),
              help="bound_charge (two-way): after every acoustic step the bound charge rho_b = -div(e S) of the "
                   "strain goes into the electrostatics slot 'charge_density', the electrostatics re-solves "
                   '(warm-started CG; with driven terminals only the DC part) and its E goes back in the same '
                   'transfer, so the waves feel the piezoelectric stiffening without lag and the acoustic dt uses c^D '
                   '= c^E + e^T eps^-1 e; costs one CG solve per acoustic step, needs the coupling every equal to the '
                   "acoustics' every and writes charge_density alone. none (one-way): only the field is handed over "
                   '(wave speeds about K^2 / 2 low, no charge back).',
              tip='Whether the waves act back on the electric field.\n\n- Two-way: the strain of the waves charges '
                  'the crystal, which changes the field and stiffens the material (correct wave speeds, one '
                  'electrostatics solve per time step).\n- One-way: the field only drives the waves (faster, wave '
                  'speeds a few percent low).'),),
    summary='Couples the electric field of the electrostatics with the waves in a piezoelectric material of the '
            'elastic waves, as an interdigital transducer launching surface waves; by default the waves act back on '
            'the field (piezoelectric stiffening).',
    description='Piezoelectric coupling (planar, sagittal plane): each transfer hands the electrostatics E_x / E_y '
                '(cell centres) to the electric_field_x / _y slots of elastic_waves, which adds T = -e^T (E - '
                'E_last) to its stresses at once (piezo_e* material arrays in reduced Voigt form, usually from a '
                'crystal entry; the first transfer only records the field, the initial state being its '
                'equilibrium). Transfers right after the physics step (every 1 with both at every 1); a '
                'time-varying field comes from voltage / terminal waveforms (superposed static solves, the field at '
                'the end time of each step), so the stress at t_n uses E(t_n). back_action bound_charge (default, '
                'two-way): before the field, the bound charge rho_b = -div(e S) of the accumulated strain S (face '
                'form: the mean P = e S of the two cells per face; at a Dirichlet / Robin outer face of the '
                'electrostatics the cell P, its surface charge landing on the electrode; insulating outer faces none) '
                "goes into the electrostatics slot 'charge_density' and the electrostatics re-solves at the "
                'acoustics time, so D = eps E + e S is source-free and the waves are stiffened without lag (dt from '
                'c^D = c^E + e^T eps^-1 e). back_action none: one-way, wave speeds about K^2 / 2 low (~2 % for a YZ '
                'lithium niobate SAW). Not modelled: pyroelectric and thermal-strain charge, the anti-plane coupling '
                'of cuts whose model plane is no crystal mirror plane (validation warns).',
    create=_create_piezo,
    validate=_validate_piezo,
    writes=_writes_fields)


def _selftest_cases():
    """The piezo SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_piezo import PIEZO_CASES
    return PIEZO_CASES


def register(registry, physics):
    """When elastic_waves and electrostatics are registered: the coupling type, its templates (+ the acousto-optic
    modulator when em_fdfd_hz is an absorbing optical type, so definitions_acoustics / _pockels register the
    acousto-optic and Pockels types it uses), then its selftest loader; physics: definitions.RegisteredPhysics."""
    if not {ACOUSTICS_TYPE_NAME, ELECTROSTATICS_TYPE_NAME} <= physics.names:
        return
    registry.register_coupling(PIEZO_COUPLING)
    registry.register_template('yz_ln_saw_idt', YZ_LN_SAW_IDT_LABEL, YZ_LN_SAW_IDT_DESCRIPTION, yz_ln_saw_idt_spec,
                               summary=YZ_LN_SAW_IDT_SUMMARY)
    if _AOM_OPTICAL_TYPE in physics.absorbing_optical_types:
        registry.register_template('tfln_acousto_optic_modulator', TFLN_AOM_LABEL, TFLN_AOM_DESCRIPTION,
                                   tfln_acousto_optic_modulator_spec, summary=TFLN_AOM_SUMMARY)
    registry.register_selftests(_selftest_cases)

"""Coupling type electrostatics_em_pockels (applied static field -> EM index, an E13 'index' contributor), its
self-test and the 'ln_phase_modulator' template; discovered and registered by definitions.register."""
from ..model.definition_helpers import evaluates_to, selected_material_data
from ..model.schema import CouplingType, Issue, Param
from ..physics.potential.pockels import (APPLIED_FIELDS, POCKELS_ARRAYS, POCKELS_SHEAR_ROW, POCKELS_TENSOR_ROWS,
                                         make_pockels_transform, pockels_shear_shift)
from .definitions_acoustics import crystal_off_diagonal_issues
from .index_terms import (has_off_diagonal_slot, index_contributor, index_slots, target_type,
                          writes_index_and_off_diagonal_slots)
from .selftests_pockels import (LN_PHASE_MODULATOR_DESCRIPTION, LN_PHASE_MODULATOR_LABEL, LN_PHASE_MODULATOR_SUMMARY,
                                ln_phase_modulator_spec)

POCKELS_TYPE_NAME = 'electrostatics_em_pockels'   # = selftests_pockels.POCKELS_TYPE_NAME (not imported)
ELECTROSTATICS_TYPE_NAME = 'electrostatics'   # = physics.potential.definition.ELECTROSTATICS_TYPE_NAME
_POCKELS_PROPERTIES = {array: ('r', array) for array in POCKELS_ARRAYS.values()}   # array -> (isotropic, component)


# ---- create callback (worker thread, GPU) ----
def _create_pockels(ctx, coupling_spec, values, every):
    """One 'index' contributor per index slot of the target type (dn from the chosen E component; + the 'sum'
    eps_xy_update term of em_fdfd_hz from the tensor xy row), a CouplingGroup when several; validation guarantees
    index slots."""
    return index_contributor(ctx, coupling_spec, every, APPLIED_FIELDS[values['field_component']],
                             make_pockels_transform, off_diagonal_transform=pockels_shear_shift)


# ---- validation (CPU only) ----
def _pockels_material_issues(vctx, slots, path, off_diagonal):
    """Warning when no selected material has a non-zero r or tensor entry of a written slot (the coupling changes
    nothing); warnings per selected material that is dispersive (only its background index shifts) or sets both r /
    r_ii and tensor entries (added twice); off_diagonal: the target's eps_xy row counts too."""
    arrays = sorted(POCKELS_ARRAYS[slot] for slot in slots)
    tensor_names = ([name for slot in slots for name in POCKELS_TENSOR_ROWS[slot]]
                    + (list(POCKELS_SHEAR_ROW) if off_diagonal else []))
    issues, active = [], False

    def non_zero(value):   # field-dependent counts
        return evaluates_to(value, vctx.parameters, lambda number: number != 0.0)
    for material, properties, poles in selected_material_data(vctx.spec, unresolved=True):
        if properties is None:   # reported at the material
            active = True
            continue
        names = []
        for array in arrays:
            isotropic, component = _POCKELS_PROPERTIES[array]
            if non_zero(properties.get(component, properties.get(isotropic))):
                names.append(component)
        tensor = [name for name in tensor_names if non_zero(properties.get(name))]
        active = active or bool(names) or bool(tensor)
        if (names or tensor) and poles:
            issues.append(Issue('warning', path,
                                f"material '{material.name}' is dispersive and sets {', '.join(names + tensor)}: the "
                                'Pockels transfer shifts only its background index (n0 = sqrt of eps_r / eps_inf), '
                                'never the pole susceptibility'))
        if names and tensor:
            issues.append(Issue('warning', path,
                                f"material '{material.name}' sets both {', '.join(names)} and the tensor entries "
                                f"{', '.join(tensor)}: both index changes are added (r_ii E_k + r_i1 E_x + r_i2 E_y); "
                                'keep one description of the crystal'))
    if not active:
        issues.append(Issue('warning', path,
                            f"no selected material sets a non-zero {' / '.join(arrays)} (or r, or the tensor entries "
                            "pockels_r*): the coupling changes no index (library lithium_niobate_x_cut carries r, a "
                            "{'model': 'crystal'} entry the tensor; other materials need r set)"))
    return issues


def _validate_pockels(vctx, coupling_spec, values):
    """Error when the target type has no index slot; material warnings (no r anywhere, r on a dispersive material).
    The writer rule (E13) is checked by model.validate through `writes`."""
    path = ('coupling', coupling_spec.name)
    physics_type = target_type(vctx, coupling_spec)
    if physics_type is None:
        return []
    slots = index_slots(physics_type)
    if not slots:
        return [Issue('error', path,
                      f"target physics '{coupling_spec.target}' ({physics_type.name}) has no permittivity slot "
                      f"({', '.join(POCKELS_ARRAYS)}); its source slots: "
                      f"{', '.join(physics_type.source_slots) or 'none'}")]
    return (_pockels_material_issues(vctx, slots, path, has_off_diagonal_slot(physics_type))
            + crystal_off_diagonal_issues(vctx, path, 'pockels', 'Pockels response', physics_type))


# ---- type ----
def pockels_coupling_type(em_types):
    """electrostatics_em_pockels from electrostatics to every registered family-'em' type (sorted names em_types)."""
    return CouplingType(
        POCKELS_TYPE_NAME, 'Pockels effect (electrostatics -> electromagnetic waves)', ELECTROSTATICS_TYPE_NAME,
        em_types,
        params=(
            Param('field_component', 'choice', 'x', label='Applied field direction', choices=tuple(APPLIED_FIELDS),
                  choice_labels=(('x', 'Along x'), ('y', 'Along y')),
                  help="Component E_k of the electrostatic field (E_x or E_y of the source) that drives every "
                       "index change dn_ii = -1/2 n0_ii^3 r_ii E_k; the other component is ignored, so align the "
                       "material's r_ii with this direction (x-cut lithium niobate in top view: x, its optic axis). "
                       'The crystal tensor entries pockels_r11 .. _r62 always use both components.',
                  tip='Direction of the applied static field that changes the refractive index; the field '
                      'component across it is ignored.\n\n'
                      'For lithium niobate this is the direction of its optic axis.'),),
        summary='Changes the refractive index of a wave physics in proportion to the static electric field of an '
                'electrostatics physics, as in an electro-optic modulator. It uses the electro-optic coefficients of '
                'the materials.',
        description='Pockels (linear electro-optic) effect: each transfer writes, for every permittivity component '
                    'ii the EM type has (E_z types: zz through eps_r_update; H_z types: xx and yy through '
                    'eps_xx_update / eps_yy_update; em_modes: all three), the index change dn_ii = -1/2 n0_ii^3 r_ii '
                    'E_k with E_k = E_x or E_y of the electrostatics physics (field_component; V/m, E = -grad V) and '
                    'r_xx / r_yy / r_zz the material Pockels coefficients [m/V] of the electrostatics vocabulary '
                    '(isotropic r sets all three; 0 where unset). n0_ii = sqrt of the permittivity component before '
                    'any coupling rewrites it (the E13 shared reference index; dispersive cells: sqrt of eps_r / '
                    'eps_inf, the pole susceptibility never changes: validation warning). It is an E13 index '
                    'contributor: with heat_em_thermo_optic on the same physics the slot gets eps = (n0 + dn_Pockels + '
                    'dn_thermal)^2, so both couplings need the same every (validation error otherwise); an '
                    'overwriting field_transfer on the slot, or a state-dependent permittivity property, cannot be '
                    'combined with it. This is the diagonal of the Pockels tensor for one applied direction: x-cut '
                    "lithium niobate in top view (crystal z along model x, electrodes either side of a waveguide "
                    "along y, field_component x) has r_xx = r33 = 30.8e-12 (E_x of the H_z types sees n_e) and r_yy "
                    '= r_zz = r13 = 8.6e-12 (clamped r^S, library lithium_niobate_x_cut); the sign follows E: a '
                    'positive E_k with r > '
                    '0 lowers n. A crystal (a {"model": "crystal"} entry: the clamped r^S rotated into model axes, '
                    'the tensor entries pockels_r11 .. _r62) adds -1/2 n0_ii^3 (r_i1 E_x + r_i2 E_y) from both field '
                    'components whatever field_component, so the opposite-sign r22 terms of a rotated cut are '
                    'included; clamped because the strain of a piezoelectric elastic_waves run adds its own index '
                    'change through acoustics_em_acousto_optic (without that coupling, a slow drive of a free crystal '
                    'sees r^T, larger for lithium niobate). em_fdfd_hz also gets the xy row in its eps_xy_update slot '
                    '(a sum contributor): Delta eps_xy = -n0_xx^2 n0_yy^2 (r_61 E_x + r_62 E_y) (r51 of YZ lithium '
                    'niobate); other in-plane EM types drop it, and every type drops the anti-plane change '
                    '(Delta(1/n^2)_yz, _xz; validation warns for crystals). The applied field is the one of the '
                    'last electrostatics solve: a stationary study applies it each fixed-point iteration (one '
                    'iteration behind within the loop, as the thermo-optic coupling), a frequency_domain or '
                    'mode_analysis study once after the background solve, before the wavelengths / modes. eps_dc '
                    'differences matter: a dielectric with a high eps_dc (lithium niobate eps_c 28) between low-eps_dc '
                    'cladding (silica 3.9) takes only a small share of the voltage.',
        create=_create_pockels,
        validate=_validate_pockels,
        writes=writes_index_and_off_diagonal_slots)


def _selftest_cases():
    """The Pockels SelfTestCases (imported on the first registry.selftests(), not by get_registry): pockels_cpu_gpu
    only; its wrapped pockels_phase_index_sum is not registered alone (examples check both of its parts)."""
    from .selftests_backend import POCKELS_BACKEND_CASES
    return POCKELS_BACKEND_CASES


def register(registry, physics):
    """The coupling type, its selftest loader (+ the thermo-optic index sum) and the 'ln_phase_modulator' template,
    when an EM type and electrostatics are registered (physics: definitions.RegisteredPhysics)."""
    if not physics.em_types or ELECTROSTATICS_TYPE_NAME not in physics.names:
        return
    registry.register_coupling(pockels_coupling_type(physics.em_types))
    registry.register_template('ln_phase_modulator', LN_PHASE_MODULATOR_LABEL, LN_PHASE_MODULATOR_DESCRIPTION,
                               ln_phase_modulator_spec, summary=LN_PHASE_MODULATOR_SUMMARY)
    registry.register_selftests(_selftest_cases)

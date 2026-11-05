"""Coupling type solid_mechanics -> membrane: solid_membrane_tension, the relaxed plane-stress film stress times its
thickness (N_ij = h sigma_ij per element) overwriting the membrane's coupled tension slots (new_modules_plan S9);
discovered and registered by definitions.register."""
from ..core.backend import array_module_of
from ..core.constants import DTYPE, FACES
from ..model.definition_helpers import evaluates_to, is_axisymmetric, selected_material_data
from ..model.schema import CouplingType, Issue, Param
from .coupling_term import CouplingGroup, CouplingTerm
from .index_terms import after_source_step, physics_and_type

TENSION_COUPLING_NAME = 'solid_membrane_tension'   # = selftests_membrane.TENSION_COUPLING (not imported)
SOLID_TYPE_NAME = 'solid_mechanics'                # = physics.mechanics.selftests.SOLID_TYPE_NAME (not imported)
MEMBRANE_TYPE_NAME = 'membrane'                    # = physics.membrane.definition.MEMBRANE_TYPE_NAME (not imported)
# solid state field (per element on triangles, the centroid value = the element mean of P1 / P2 stress) -> membrane
# source slot (physics.membrane.materials.COUPLED_TENSION_SLOTS, repeated here)
STRESS_TO_TENSION = (('stress_xx', 'coupled_tension_xx'), ('stress_yy', 'coupled_tension_yy'),
                     ('stress_xy', 'coupled_tension_xy'))
_PLANE_STRESS = 'plane_stress'                     # = physics.mechanics.solver_common PLANE_STRESS (not imported)
_TRIANGLES = 'triangles'
_STUDY_SOLVED_SOLID_STUDIES = ('eigenfrequency', 'frequency_response')   # where the solid would be solved itself
_MEMBRANE_TENSION_PROPERTIES = ('tension', 'tension_xx', 'tension_yy', 'tension_xy')


class ScaledStress:
    """Transform h * stress of the solid's per-element state into one persistent float32 buffer (allocated on the
    first transfer on the field's array module, so it runs on the run backend)."""

    def __init__(self, thickness, coupling_name):
        self.thickness = DTYPE(thickness)
        self.coupling_name = coupling_name
        self.buffer = None

    def __call__(self, field, source_solver, target_solver):
        if getattr(field, 'ndim', 0) != 1:
            raise ValueError(f"coupling '{self.coupling_name}': the solid stress must be one value per triangle "
                             f"element, got shape {getattr(field, 'shape', ())} (the solid must run on the membrane's "
                             'triangle mesh)')
        xp = array_module_of(field)
        if self.buffer is None or self.buffer.shape != field.shape:
            self.buffer = xp.empty(field.shape, dtype=DTYPE)
        xp.multiply(field, self.thickness, out=self.buffer)
        return self.buffer


def _film_thickness(coupling_thickness, solid_thickness):
    """The coupling's thickness param, else the solid's thickness setting (None when neither is set)."""
    return coupling_thickness if coupling_thickness is not None else solid_thickness


# ---- create callback (worker thread) ----
def _create_tension(ctx, coupling_spec, values, every):
    """CouplingGroup of three overwriting terms stress_ij -> coupled_tension_ij, each scaled by the film thickness."""
    solid = ctx.sim.physics[coupling_spec.source]
    thickness = _film_thickness(values['thickness'], getattr(solid, 'thickness', None))
    if thickness is None:
        raise ValueError(f"coupling '{coupling_spec.name}': no film thickness (set the coupling's thickness or the "
                         f"thickness setting of solid '{coupling_spec.source}')")
    terms = [CouplingTerm(coupling_spec.source, stress, coupling_spec.target, slot,
                          ScaledStress(thickness, coupling_spec.name), every=1, name=f'{coupling_spec.name}.{slot}')
             for stress, slot in STRESS_TO_TENSION]
    return CouplingGroup(terms, every=every, name=coupling_spec.name,
                         after_source_step=after_source_step(ctx, coupling_spec, every))


def _writes_tension(vctx, coupling_spec):
    """CouplingType.writes: every coupled tension slot overwritten ('')."""
    return {slot: '' for _, slot in STRESS_TO_TENSION}


# ---- validation (CPU only) ----
def _material_tension_issues(vctx, path, membrane_name):
    """Info when a selected material also sets a membrane tension: the coupled tension adds to it."""
    names = [material.name for material, properties, _ in selected_material_data(vctx.spec)
             if any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value != 0.0)
                    for name in _MEMBRANE_TENSION_PROPERTIES)]
    if not names:
        return []
    return [Issue('info', path, f"material(s) {', '.join(names)} also set a membrane tension: membrane "
                                f"'{membrane_name}' adds the coupled film tension to it (leave the material tension "
                                'unset when the solid supplies the whole prestress)')]


def _validate_tension(vctx, coupling_spec, values):
    """Errors: the solid not planar plane stress, no film thickness, not on a triangle mesh; warnings: two different
    thicknesses, a membrane PML (the solid's ring is void); infos: the solid solved by the study itself (remedy
    solved_physics), a material tension that the coupled one adds to."""
    path = ('coupling', coupling_spec.name)
    solid_spec, solid_type = physics_and_type(vctx, coupling_spec.source)
    membrane_spec, membrane_type = physics_and_type(vctx, coupling_spec.target)
    if solid_spec is None or membrane_spec is None:
        return []
    issues = []
    solid_settings = vctx.values(solid_type.settings, solid_spec.settings) or {}   # bad settings: param checks report
    if is_axisymmetric(solid_settings) or solid_settings.get('out_of_plane') != _PLANE_STRESS:
        issues.append(Issue('error', path, f"solid '{solid_spec.name}' must be a planar plane-stress film (setting "
                                           'out_of_plane plane_stress, geometry planar): the membrane tension is the '
                                           'in-plane film stress times its thickness'))
    solid_thickness = solid_settings.get('thickness')
    coupling_thickness = values.get('thickness')
    if _film_thickness(coupling_thickness, solid_thickness) is None:
        issues.append(Issue('error', path, f"no film thickness: set the coupling's thickness or the thickness setting "
                                           f"of solid '{solid_spec.name}'"))
    elif coupling_thickness is not None and solid_thickness is not None and coupling_thickness != solid_thickness:
        issues.append(Issue('warning', path, f'thickness {coupling_thickness:.4g} m differs from the thickness setting '
                                             f"{solid_thickness:.4g} m of solid '{solid_spec.name}': the coupling uses "
                                             'its own (leave it empty to use the solid one)'))
    try:
        mesh_kind = vctx.spec.mesh_of(solid_spec).kind
    except KeyError:   # reported at the physics
        mesh_kind = None
    if mesh_kind is not None and mesh_kind != _TRIANGLES:
        issues.append(Issue('error', path, f"solid '{solid_spec.name}' must run on a triangle mesh (the membrane's): "
                                           'the tension is written per element'))
    membrane_settings = vctx.values(membrane_type.settings, membrane_spec.settings) or {}
    pml = [face for face in FACES if membrane_settings.get(f'pml_{face}')]
    if pml:
        issues.append(Issue('warning', path, f"membrane '{membrane_spec.name}' has a PML on {', '.join(pml)}: the "
                                             'solid is void in the PML ring, so the coupled tension is 0 there and the '
                                             'layer has no stiffness from it'))
    if vctx.study_type in _STUDY_SOLVED_SOLID_STUDIES and vctx.spec.study.solves(solid_spec.name):
        issues.append(Issue('info', path, f"the {vctx.study_type} study solves solid '{solid_spec.name}' itself, so "
                                          'this coupling cannot read its static state: set the study solved_physics to '
                                          f"['{membrane_spec.name}'] (the solid then runs as a stationary background "
                                          'physics first)'))
    issues.extend(_material_tension_issues(vctx, path, membrane_spec.name))
    return issues


# ---- type ----
_THICKNESS_PARAM = Param(
    'thickness', 'float', None, label='Film thickness', unit='m', min=0.0, min_exclusive=True, optional=True,
    help='Film thickness h [m] in N_ij = h sigma_ij. Empty (default): the thickness setting of the solid (validation '
         'error when that is empty too); a value overrides it (a warning when both are set and differ).',
    tip='Thickness of the film that turns its stress into the membrane tension.\n\nEmpty uses the thickness set on '
        'the solid mechanics.')

SOLID_MEMBRANE_TENSION = CouplingType(
    TENSION_COUPLING_NAME, 'Film prestress (solid mechanics -> membrane)', SOLID_TYPE_NAME, MEMBRANE_TYPE_NAME,
    params=(_THICKNESS_PARAM,),
    summary='Sets the tension of the membrane from the stress of a thin film computed by the solid mechanics, for '
            'example a prestressed film that relaxes around holes before its vibration modes are found. Both physics '
            'share one triangle mesh; list only the membrane as solved in the study.',
    description='Membrane tension from a plane-stress solid on the same triangle mesh: each transfer writes N_xx = h '
                'stress_xx, N_yy = h stress_yy and N_xy = h stress_xy [N/m] per element into the membrane slots '
                'coupled_tension_xx / _yy / _xy (overwriting: no other coupling may write them), which the membrane '
                'adds to its material tension (leave that unset when the solid gives the whole prestress; validation '
                'info otherwise). stress_ij is the total static stress of the solid (initial_stress + C (strain - '
                'thermal strain): the relaxed state of a prestressed film with holes) at the element centroid, the '
                'element mean for P1 / P2 on straight elements, so the cell integral of the tension is exact; void '
                'elements (no youngs_modulus) give 0, so with areal_density 0 there they are membrane holes (free '
                'edges); a void element with areal_density > 0 is mass without stiffness. h = the thickness param, '
                'else the thickness setting of the solid. The solid must be planar plane_stress on the membrane\'s '
                'triangle mesh (validation errors; no mesh transfer). Run order: in an eigenfrequency or '
                'frequency_response study set StudySpec.solved_physics to the membrane only: the solid then runs as a '
                'stationary background physics (validated and estimated as in a stationary study), its static solve '
                'comes first, the coupling transfers once, then the membrane resonances / responses are solved; '
                'without solved_physics the study solves the solid itself and the coupling is a validation error. A '
                'membrane PML ring gets no coupled tension (the solid is void there: warning). Linearised about the '
                'relaxed state: no stress stiffening of the solid itself and no bending.',
    create=_create_tension,
    validate=_validate_tension,
    writes=_writes_tension)


def _selftest_cases():
    """The film-tension SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_membrane import MEMBRANE_TENSION_CASES
    return MEMBRANE_TENSION_CASES


def register(registry, physics):
    """solid_membrane_tension (+ its selftest loader) when solid_mechanics and membrane are registered (physics:
    definitions.RegisteredPhysics)."""
    if {SOLID_TYPE_NAME, MEMBRANE_TYPE_NAME} <= physics.names:
        registry.register_coupling(SOLID_MEMBRANE_TENSION)
        registry.register_selftests(_selftest_cases)

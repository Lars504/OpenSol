"""Model-layer rules of the solids' viscoelastic moduli and formulation setting (FEM meshes, frequency_response): the
validation of the relaxed shear / bulk moduli, the frequency-dependent properties (viscosity, relaxation branches) and
the mixed u-p formulation; imported by definition.py and definition3d.py."""
# Rule numbers V1 .. V22 of the design (viscoelastic.py header for the law): physics-level rules at the physics path,
# material rules at the material. Materials: definition_helpers.selected_material_data (the builder's merge; user-set =
# MaterialSpec.properties). A material is 'frequency dependent' when it may set a VISCOELASTIC_ARRAYS value != 0; a
# member of the displacement formulation needs a relaxed modulus (a viscosity-only liquid is void there).
from ...model.definition_helpers import (evaluates_to, is_axisymmetric, plain_number, reports_model_checks,
                                         selected_material_data)
from ...model.schema import FREQUENCY_RESPONSE_STUDY, Issue
from .definition_common import LOCKING_POISSON_RATIO
from .materials import (BULK_BRANCH_ARRAYS, LOSS_ARRAYS, MODULUS_ARRAYS, SHEAR_BRANCH_ARRAYS, STIFFNESS_TENSOR_ARRAYS,
                        STIFFNESS_TENSOR_ARRAYS_3D, VISCOELASTIC_ARRAYS, VISCOELASTIC_PROPERTY_NAMES)
from .viscoelastic import DISPLACEMENT, MAX_GROUPS, MIXED

_FREQUENCY_NAMES = tuple(name for name in VISCOELASTIC_PROPERTY_NAMES if name in VISCOELASTIC_ARRAYS)
_LAMINAR_FLOW_TYPE = 'laminar_flow'   # shares the viscosity property (> 0 there: a fluid cell)
_REDUCED_SWEEPS = ('adaptive', 'modal')
_PLANE_STRESS = 'plane_stress'
# values of a material's parameter tuple (V20: one viscoelastic group per distinct tuple)
_GROUP_NAMES = ('youngs_modulus', 'poisson_ratio', *MODULUS_ARRAYS, LOSS_ARRAYS[0], *_FREQUENCY_NAMES)


class _Material:
    """The rule predicates of one selected material (merged properties; user-set ones apart)."""

    def __init__(self, material, properties, parameters, tensor_names):
        self.name = material.name
        self.path = ('material', material.name)
        self.properties = properties
        self.user = dict(material.properties or {})
        self.parameters = parameters
        self.youngs = properties.get('youngs_modulus') is not None
        self.poisson = properties.get('poisson_ratio') is not None
        self.moduli = any(properties.get(name) is not None for name in MODULUS_ARRAYS)
        self.shear_set = properties.get('shear_modulus') is not None
        self.bulk_set = properties.get('bulk_modulus') is not None
        self.tensor = any(properties.get(name) is not None for name in tensor_names)
        self.frequency_dependent = any(self.nonzero(name) for name in _FREQUENCY_NAMES)
        self.user_frequency = [name for name in _FREQUENCY_NAMES
                               if evaluates_to(self.user.get(name), parameters, lambda value: value != 0.0)]
        self.viscous = self.positive('viscosity')
        self.shear_branch = any(self.positive(modulus) for modulus, _ in SHEAR_BRANCH_ARRAYS)
        self.bulk_parts = self.positive('bulk_viscosity') or any(self.positive(modulus)
                                                                 for modulus, _ in BULK_BRANCH_ARRAYS)
        shear = plain_number(properties.get('shear_modulus'), parameters)
        # G0 = 0: no youngs_modulus or c_ij entries and shear_modulus unset or a plain 0 (an expression may be > 0)
        self.shear_zero = not self.youngs and not self.tensor and (not self.shear_set or shear == 0.0)
        self.shear_alive = (not self.shear_zero) or self.viscous or self.shear_branch
        self.stiff = self.youngs or self.moduli or self.tensor   # a relaxed modulus: a displacement-formulation member

    def nonzero(self, name) -> bool:
        return evaluates_to(self.properties.get(name), self.parameters, lambda value: value != 0.0)

    def positive(self, name) -> bool:
        return evaluates_to(self.properties.get(name), self.parameters, lambda value: value > 0.0)

    def group_tuple(self):
        """The plain values of _GROUP_NAMES (None unset), or None when one is not a plain number."""
        values = []
        for name in _GROUP_NAMES:
            value = self.properties.get(name)
            number = plain_number(value, self.parameters)
            if value is not None and number is None:
                return None
            values.append(number)
        return tuple(values)


def viscoelastic_issues(vctx, physics_spec, settings_values, path, fem, dimensions) -> list:
    """The viscoelastic / formulation rules V1 .. V22 of a solid (vctx in the study the solid takes, fem: triangles /
    tetrahedra, dimensions 2 or 3)."""
    study = vctx.study_type
    response = study == FREQUENCY_RESPONSE_STUDY
    formulation = settings_values.get('formulation', DISPLACEMENT)
    mixed = formulation == MIXED
    parameters = vctx.parameters
    tensor_names = STIFFNESS_TENSOR_ARRAYS_3D if dimensions == 3 else STIFFNESS_TENSOR_ARRAYS
    materials = [_Material(material, properties, parameters, tensor_names)
                 for material, properties, _ in selected_material_data(vctx.spec)]
    issues = []
    issues.extend(_physics_issues(vctx, settings_values, path, fem, dimensions, materials, response, mixed))
    if reports_model_checks(vctx, physics_spec, physics_spec.type):
        issues.extend(_material_route_issues(materials))
        issues.extend(_laminar_flow_issues(vctx, materials))
    issues.extend(_formulation_material_issues(materials, study, response, mixed, fem, physics_spec.name))
    return issues


def _members(materials, mixed):
    """The selected materials the formulation solves (mixed: G* not identically 0; displacement: a relaxed modulus)."""
    return [material for material in materials if (material.shear_alive and (material.stiff or material.viscous)
                                                   if mixed else material.stiff)]


def _physics_issues(vctx, settings_values, path, fem, dimensions, materials, response, mixed) -> list:
    """V1 .. V7, V20, V22 at the physics."""
    issues = []
    members = _members(materials, mixed)
    frequency = response and any(material.frequency_dependent for material in members)
    if mixed:
        if not fem:   # V1
            issues.append(Issue('error', path, 'formulation mixed needs a triangle or tetrahedron mesh (the '
                                               'rectangular-cell solid solves the displacement only)'))
        if not response:   # V2
            issues.append(Issue('error', path, f'the mixed formulation runs in {FREQUENCY_RESPONSE_STUDY} only (this '
                                               f'model runs {vctx.study_type}): set formulation displacement'))
        if settings_values.get('discretization', 'quadratic') != 'quadratic':   # V3
            issues.append(Issue('error', path, 'formulation mixed needs discretization quadratic (Taylor-Hood: P2 '
                                               'displacement, P1 pressure)'))
        if any(material.tensor for material in members):   # isotropic only (no deviatoric split of a c_ij)
            names = ', '.join(material.name for material in members if material.tensor)
            issues.append(Issue('error', path, f'formulation mixed takes isotropic materials only; {names} set(s) c_ij '
                                               'entries (or a crystal entry)'))
    what = ('formulation mixed is' if mixed
            else 'frequency-dependent moduli (viscosity, relaxation branches) in a frequency response are')
    if mixed or frequency:
        if settings_values.get('piezoelectric'):   # V4
            issues.append(Issue('error', path, f'{what} not available with piezoelectric on'))
        if (dimensions == 2 and not is_axisymmetric(settings_values)
                and settings_values.get('out_of_plane') == _PLANE_STRESS):   # V5
            issues.append(Issue('error', path, f'{what} not available with plane_stress: its condensation sigma_zz = '
                                               '0 is not linear in the moduli (only a plain mechanical_loss_factor '
                                               'keeps it); use plane_strain'))
        if settings_values.get('stress_stiffening') or settings_values.get('geometric_nonlinearity'):   # V6
            issues.append(Issue('error', path, f'{what} not available with stress_stiffening or '
                                               'geometric_nonlinearity (no static state of the frequency-dependent '
                                               'moduli): turn them off'))
    sweep = (vctx.spec.study.settings if isinstance(vctx.spec.study.settings, dict) else {}).get('sweep_method',
                                                                                                'direct')
    if response and ((frequency and sweep in _REDUCED_SWEEPS) or (mixed and sweep == 'modal')):   # V7
        issues.append(Issue('error', path, f"sweep_method '{sweep}' needs a pencil polynomial in omega with a modal "
                                           'basis (frequency-dependent moduli change with omega; the mixed pressure '
                                           "block is massless): use sweep_method 'direct'"))
    grouped = members if mixed else [material for material in members if material.frequency_dependent]
    tuples = {material.group_tuple() for material in grouped if response and fem} - {None}
    if len(tuples) > MAX_GROUPS:   # V20 (the solver repeats it for expressions)
        issues.append(Issue('error', path, f'{len(tuples)} distinct viscoelastic material parameter sets exceed '
                                           f'{MAX_GROUPS} (each is a fixed term of the frequency response): merge '
                                           'materials of equal values'))
    if mixed:   # V22
        pml = sorted(key for key, value in settings_values.items()
                     if key.startswith('pml_') and key not in ('pml_cells', 'pml_reflection') and value)
        periodic = sorted(key for key in ('periodic_x', 'periodic_y', 'periodic_z') if settings_values.get(key))
        if pml or periodic:
            issues.append(Issue('error', path, f"formulation mixed is not available with {', '.join(pml + periodic)} "
                                               '(an elastic PML runs in the eigenfrequency study only; the harmonic '
                                               'output rows assume no periodic map)'))
    return issues


def _material_route_issues(materials) -> list:
    """V8 .. V11 (settings free: reported once per model)."""
    issues = []
    for material in materials:
        if (material.youngs or material.poisson) and material.moduli:   # V8
            issues.append(Issue('error', material.path, f"material '{material.name}' sets youngs_modulus / "
                                                        'poisson_ratio and shear_modulus / bulk_modulus: one route '
                                                        'per material (E and nu, or G0 and K0)'))
        if material.tensor and (material.moduli or material.frequency_dependent):   # V9
            issues.append(Issue('error', material.path, f"material '{material.name}' sets c_ij entries (or a crystal "
                                                        'entry) and isotropic or viscoelastic moduli (shear_modulus, '
                                                        'bulk_modulus, viscosity, relaxation branches): the '
                                                        'viscoelastic law is isotropic'))
        for part, branches in (('shear', SHEAR_BRANCH_ARRAYS), ('bulk', BULK_BRANCH_ARRAYS)):   # V10
            for modulus, time in branches:
                if material.positive(modulus) and not material.positive(time):
                    issues.append(Issue('error', material.path, f"material '{material.name}' sets {modulus} > 0 "
                                                                f'without {time} > 0 (the {part} branch relaxes '
                                                                'over that time)'))
                time_value = plain_number(material.properties.get(time), material.parameters)
                modulus_value = plain_number(material.properties.get(modulus), material.parameters)
                unset = material.properties.get(modulus) is None
                if (time_value is not None and time_value > 0.0
                        and (unset or (modulus_value is not None and modulus_value <= 0.0))):
                    issues.append(Issue('error', material.path, f"material '{material.name}' sets {time} without "
                                                                f'{modulus} > 0'))
        if material.bulk_parts and not (material.bulk_set or material.youngs):   # V11
            issues.append(Issue('error', material.path, f"material '{material.name}' sets bulk_viscosity or a bulk "
                                                        'relaxation branch: it needs a finite bulk modulus '
                                                        '(bulk_modulus or youngs_modulus)'))
        bulk = plain_number(material.properties.get('bulk_modulus'), material.parameters)
        if bulk is not None and bulk <= 0.0:
            issues.append(Issue('error', material.path, f"material '{material.name}' has bulk_modulus {bulk:g}: it "
                                                        'must be > 0 (leave it unset for an incompressible material '
                                                        'of formulation mixed)'))
    return issues


def _laminar_flow_issues(vctx, materials) -> list:
    """V19: a laminar_flow physics shares viscosity (> 0 there: a fluid cell)."""
    if not any(physics.type == _LAMINAR_FLOW_TYPE for physics in vctx.spec.physics):
        return []
    return [Issue('warning', material.path, f"material '{material.name}' sets a viscosity > 0 and a solid modulus: "
                                            f'{_LAMINAR_FLOW_TYPE} treats these cells as fluid (its viscosity is the '
                                            'same property)')
            for material in materials if material.viscous and material.stiff]


def _formulation_material_issues(materials, study, response, mixed, fem, physics_name) -> list:
    """V12 .. V18 and V21 at the materials."""
    issues = []
    for material in materials:
        name, material_path = material.name, material.path
        solid_vocabulary = (material.bulk_set or material.shear_set or material.shear_branch or material.bulk_parts)
        if not fem:   # V18
            user = sorted(key for key in (*MODULUS_ARRAYS, *_FREQUENCY_NAMES) if material.user.get(key) is not None)
            if user:
                issues.append(Issue('error', material_path, f"material '{name}' sets {', '.join(user)}: triangle / "
                                                            'tetrahedron meshes only; a tensor mesh reads '
                                                            'youngs_modulus / poisson_ratio and the plain '
                                                            'mechanical_loss_factor'))
            continue
        if not mixed and material.shear_set and not material.bulk_set and not material.youngs:   # V12
            issues.append(Issue('error', material_path, f"material '{name}' sets shear_modulus without bulk_modulus: "
                                                        f'{physics_name} needs bulk_modulus (or formulation mixed in '
                                                        'a frequency response: unset = incompressible)'))
        elif response and not mixed and material.shear_zero and solid_vocabulary:   # V13
            issues.append(Issue('error', material_path, f"material '{name}' has no shear stiffness (bulk_modulus "
                                                        'only, shear_modulus 0 or relaxation branches alone): a '
                                                        'liquid cell, use formulation mixed'))
        if response and mixed and (material.bulk_set or material.bulk_parts) and not material.shear_alive:   # V14
            issues.append(Issue('error', material_path, f"material '{name}' has no shear stiffness or viscosity "
                                                        '(bulk_modulus only): G* = 0, nothing for the mixed '
                                                        'formulation to solve there'))
        if response and not mixed and material.viscous and not material.stiff:   # V15
            issues.append(Issue('warning', material_path, f"material '{name}' sets a viscosity but no modulus: void "
                                                          f'for {physics_name} (a liquid joins only with formulation '
                                                          'mixed)'))
        if not response and material.user_frequency and material.stiff:   # V16
            issues.append(Issue('warning', material_path, f"material '{name}' sets "
                                                          f"{', '.join(material.user_frequency)}: ignored by the "
                                                          f'{study} study, the relaxed moduli G0, K0 are used '
                                                          f'(only {FREQUENCY_RESPONSE_STUDY} reads them)'))
        if not response and material.shear_zero and (material.bulk_set or material.shear_set):   # V17
            issues.append(Issue('error', material_path, f"material '{name}' has no shear stiffness (shear_modulus 0 "
                                                        f'or bulk_modulus only): a liquid has no static stiffness, '
                                                        f'the {study} study cannot hold it'))
        poisson = plain_number(material.properties.get('poisson_ratio'), material.parameters)
        if response and not mixed and poisson is not None and poisson > LOCKING_POISSON_RATIO:   # V21
            issues.append(Issue('info', material_path, f"material '{name}': poisson_ratio {poisson:g} is nearly "
                                                       'incompressible; consider formulation mixed (locking-free) '
                                                       'for its frequency response'))
    return issues

"""Model-layer rules and estimate counts of the solids' viscoelastic moduli and formulation setting (FEM meshes,
frequency_response): the validation of the relaxed shear / bulk moduli, the frequency-dependent properties (viscosity,
relaxation branches) and the mixed u-p formulation, and the memory / run-time items of their group terms and saddle-point
blocks; imported by definition.py and definition3d.py."""
# Rule numbers V1 .. V22 of the design (viscoelastic.py header for the law): physics-level rules at the physics path,
# material rules at the material. Materials: definition_helpers.selected_material_data (the builder's merge; user-set =
# MaterialSpec.properties). A material is 'frequency dependent' when it may set a VISCOELASTIC_ARRAYS value != 0; a
# member of the displacement formulation needs a relaxed modulus (a viscosity-only liquid is void there).
# Estimates (viscoelastic_counts, viscoelastic_memory_items, viscoelastic_time_items): counted from the solver code
# (solver_common.mixed_harmonic_operators, frequency_response.HarmonicSystem, fem.elasticity / elasticity3d group
# assemblers, fem.mixed_elasticity / mixed_elasticity3d) on the mesh estimate's counts; one group per distinct material
# parameter tuple, each spread over its painted element share (interface duplicates of the pressure ignored: the
# estimate stays just below the measured bytes). Uncalibrated (mechanics sub file "Memory estimates").
import math
from dataclasses import dataclass

from ...core import time_estimate as te
from ...core.memory_estimate import BOOL_BYTES, COMPLEX128_BYTES, FLOAT64_BYTES, FLOAT64_CSR_ENTRY_BYTES, INT32_BYTES
from ...fem.assembly import COO_BATCH_ENTRIES
from ...fem.diffusion import painted_fraction
from ...model.definition_helpers import (azimuthal_order_of, evaluates_to, is_axisymmetric, plain_number,
                                         reports_model_checks, selected_material_data)
from ...model.schema import FREQUENCY_RESPONSE_STUDY, Issue
from .definition_common import LOCKING_POISSON_RATIO
from .materials import (BULK_BRANCH_ARRAYS, LOSS_ARRAYS, MODULUS_ARRAYS, SHEAR_BRANCH_ARRAYS, STIFFNESS_TENSOR_ARRAYS,
                        STIFFNESS_TENSOR_ARRAYS_3D, VISCOELASTIC_ARRAYS, VISCOELASTIC_BRANCHES,
                        VISCOELASTIC_PROPERTY_NAMES)
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
        self.incompressible = not (self.youngs or self.bulk_set or self.tensor)   # mixed: 1 / K* = 0

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
    elif (response and mixed and sweep == 'adaptive' and dimensions == 2 and is_axisymmetric(settings_values)
          and azimuthal_order_of(settings_values) == 1
          and any(material.incompressible for material in members)):   # V7b
        names = ', '.join(material.name for material in members if material.incompressible)
        issues.append(Issue('error', path, "sweep_method 'adaptive' with formulation mixed at azimuthal_order 1 cannot "
                                           f'take an incompressible material ({names}: no bulk_modulus): its zero '
                                           'pressure block is kept through the axis ties as a per-frequency term, '
                                           "which a reduced sweep refuses; use sweep_method 'direct' or give the "
                                           'material a bulk_modulus'))
    grouped = members if mixed else [material for material in members if material.frequency_dependent]
    tuples = {material.group_tuple() for material in grouped if response and fem} - {None}
    if len(tuples) > MAX_GROUPS:   # V20 (the solver repeats it for expressions)
        issues.append(Issue('error', path, f'{len(tuples)} distinct viscoelastic material parameter sets exceed '
                                           f'{MAX_GROUPS} (each is a fixed term of the frequency response): merge '
                                           'materials of equal values'))
    if mixed:   # V22 (2D triangles: the elastic PML stretches the mixed blocks too, solver_fem._mixed_operators)
        pml = sorted(key for key, value in settings_values.items()
                     if key.startswith('pml_') and key not in ('pml_cells', 'pml_reflection') and value
                     and dimensions == 3)
        periodic = sorted(key for key in ('periodic_x', 'periodic_y', 'periodic_z') if settings_values.get(key))
        if pml or periodic:
            issues.append(Issue('error', path, f"formulation mixed is not available with {', '.join(pml + periodic)} "
                                               '(an elastic PML of a 3D solid runs in the eigenfrequency study only; '
                                               'the harmonic output rows assume no periodic map)'))
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


# ---- memory and run-time estimates (both FEM solids; root "Estimates mirror the solver") ----
_INT64_BYTES = 8
# viscoelastic.ElementModuli per element: G0, K0, eta, mu_s, mu_b and (modulus, time) of every shear and bulk branch
# (float64), the two route masks (bool)
_ELEMENT_MODULI_BYTES = (5 + 4 * VISCOELASTIC_BRANCHES) * FLOAT64_BYTES + 2 * BOOL_BYTES
# COO entry of the 2D assemblers (fem.assembly.append_blocks: broadcast int64 rows / columns, float64 values, held to
# the end) and of the 3D batches (fem.assembly.CsrAccumulator: int32 rows / columns, float64 values)
_COO_BYTES_2D = 2 * _INT64_BYTES + FLOAT64_BYTES
_COO_BYTES_3D = 2 * INT32_BYTES + FLOAT64_BYTES
# fem.assembly.pattern_sum of non-matching patterns: tocoo of every part, then _to_csr's concatenated int32 rows /
# columns and float64 values (two copies of the parts' entries at the peak)
_PATTERN_SUM_BYTES = 2 * (2 * INT32_BYTES + FLOAT64_BYTES)
# frequency_response._pattern_positions per call: the int64 keys of the summed pattern (np.repeat, the product, the
# int64 columns and their sum: three int64 arrays live) and of the term (the same), its canonical copy
_POSITION_KEY_BYTES = 3 * _INT64_BYTES
# host bytes moved per stored term entry and frequency (HarmonicSystem._pencil_extra / the host route's values[positions]
# += c values): index and value reads, the complex product written and read, the gathered and scattered complex value
_TERM_VALUE_BYTES = 2 * _INT64_BYTES + 4 * COMPLEX128_BYTES
# host bytes moved per summed-pattern entry of one _pattern_positions key build (repeat write, multiply read / write,
# int64 cast read / write, add read / write) and per term entry of its search (the same keys, the copy and duplicate sum,
# the equality check); the searchsorted reads log2(entries) keys per term entry
_KEY_BUILD_BYTES = 60
_TERM_SEARCH_BYTES = _KEY_BUILD_BYTES + 4 * FLOAT64_CSR_ENTRY_BYTES + 2 * _INT64_BYTES


@dataclass(frozen=True)
class ViscoelasticCounts:
    """What a FEM solid's frequency response assembles beyond K, L, M for its viscoelastic group terms and the mixed u-p
    blocks (viscoelastic_counts): `moduli` the ElementModuli is built (a modulus or viscoelastic property set, or mixed);
    `groups` parameter groups assembled (mixed: every member, the pressure zones), `group_matrices` / `group_entries` /
    `group_coo` their CSRs (groups x parts), stored entries and assembly COO entries, `grouped_elements` the elements in
    a group; `terms` / `term_entries` the FrequencyTerms and their stored entries (u and pressure blocks); mixed:
    `pressure_dofs` n_p (P1 nodes of the member share), `coupling_entries` nnz(B), `pressure_entries` sum nnz(M_p,g),
    `pressure_coo` their assembly COO entries, `column_map` (azimuthal order 1: axis ties + the pressure selection);
    `displacement_unknowns` / `displacement_entries` the u block the caller counts (its K)."""
    mixed: bool = False
    moduli: bool = False
    groups: int = 0
    group_matrices: int = 0
    group_entries: int = 0
    group_coo: int = 0
    grouped_elements: int = 0
    terms: int = 0
    term_entries: int = 0
    pressure_dofs: int = 0
    coupling_entries: int = 0
    pressure_entries: int = 0
    pressure_coo: int = 0
    column_map: bool = False
    displacement_unknowns: int = 0
    displacement_entries: int = 0

    @property
    def active(self) -> bool:
        """The harmonic system carries terms or the saddle-point blocks."""
        return self.mixed or self.terms > 0

    @property
    def unknowns(self) -> int:
        """Unknowns of the factorised pencil: u (+ p mixed)."""
        return self.displacement_unknowns + self.pressure_dofs

    @property
    def entries(self) -> int:
        """Stored entries of the summed pencil pattern: K's + 2 nnz(B) + the pressure blocks (mixed)."""
        return self.displacement_entries + 2 * self.coupling_entries + self.pressure_entries

    @property
    def fill_scale(self) -> float:
        """Factor fill of the saddle point over the u-only fill at the same nodes: ((c n_u + n_p) / (c n_u))^2 (the
        nested-dissection node pairs carry (dofs per node)^2 entries; a P1 pressure adds one dof at every vertex)."""
        return (self.unknowns / max(self.displacement_unknowns, 1)) ** 2


def viscoelastic_counts(spec, parameters, settings_values, mesh, components, local, dimensions, displacement_unknowns,
                        displacement_entries) -> ViscoelasticCounts:
    """ViscoelasticCounts of a FEM solid (spec bound to the study the solid takes: definition_common.background_context)
    on an element-mesh estimate (n_nodes, n_edges, n_elements; tetrahedra n_faces or Euler; material_elements for the
    painted shares, else every element), `components` per node (2, 3 at azimuthal order m >= 1 or in 3D) and `local`
    displacement dofs per element; the u block (displacement_unknowns, displacement_entries) as the caller counts K.
    Groups: one per distinct plain parameter tuple of the grouped materials (an expression-valued one counts once),
    at most MAX_GROUPS; mixed: the members, else the frequency-dependent members of a frequency response."""
    tensor_names = STIFFNESS_TENSOR_ARRAYS_3D if dimensions == 3 else STIFFNESS_TENSOR_ARRAYS
    materials = [_Material(material, properties, parameters or {}, tensor_names)
                 for material, properties, _ in selected_material_data(spec)]
    response = spec.study.type == FREQUENCY_RESPONSE_STUDY
    mixed = response and (settings_values or {}).get('formulation', DISPLACEMENT) == MIXED
    viscoelastic_names = MODULUS_ARRAYS + _FREQUENCY_NAMES
    moduli = mixed or any(material.properties.get(name) is not None for material in materials
                          for name in viscoelastic_names)
    base = dict(moduli=moduli, displacement_unknowns=int(displacement_unknowns),
                displacement_entries=int(displacement_entries))
    if not response:
        return ViscoelasticCounts(**base)
    members = _members(materials, mixed)
    grouped = members if mixed else [material for material in members if material.frequency_dependent]
    shares = {}   # parameter tuple -> (painted share, representative material)
    for material in grouped:
        key = material.group_tuple() or ('expression', material.name)
        share = painted_fraction(mesh, [material.name])
        old = shares.get(key)
        shares[key] = (share + (old[0] if old else 0.0), material)
    groups = list(shares.values())[:MAX_GROUPS]
    total = sum(share for share, _ in groups)
    if total > 1.0:   # overlapping paints (or no material_elements: every share 1): the shares split the mesh
        groups = [(share / total, material) for share, material in groups]
    elements = int(mesh.n_elements)
    element_coo = (components * local) ** 2
    column_map = dimensions == 2 and is_axisymmetric(settings_values or {}) and \
        azimuthal_order_of(settings_values) == 1
    grouped_share = min(total, 1.0)
    pressure = {}
    if mixed:
        nodes, edges = int(mesh.n_nodes), int(mesh.n_edges)
        cells = (int(getattr(mesh, 'n_faces', 0)) or edges + elements - nodes + 1) if dimensions == 3 else elements
        # (P1 vertex, P2 dof) pairs sharing an element: V + 2 E vertex pairs, 2 E endpoint and 3 F opposite (vertex,
        # edge) pairs (F = triangles in 2D, faces in 3D); P1 pairs V + 2 E (fem.diffusion.lagrange_entries_per_dof)
        pressure_local = 4 if dimensions == 3 else 3   # P1 corners per element
        pressure = dict(pressure_dofs=int(round(nodes * grouped_share)),
                        coupling_entries=int(round(components * (nodes + 4 * edges + 3 * cells) * grouped_share)),
                        pressure_entries=int(round((nodes + 2 * edges) * grouped_share)),
                        pressure_coo=int(round(elements * grouped_share
                                               * (pressure_local * components * local + pressure_local ** 2))))
        parts = 1   # the unit-shear D_g of every group
        terms = sum(1 for _, material in groups if material.viscous or material.shear_branch)
        term_entries = sum(share * displacement_entries for share, material in groups
                           if material.viscous or material.shear_branch)
        pressure_terms = [(share, material) for share, material in groups
                          if material.bulk_parts or (material.incompressible and column_map)]
        terms += len(pressure_terms)
        term_entries += sum(share for share, _ in pressure_terms) * (nodes + 2 * edges)
    else:
        shear = any(material.viscous or material.shear_branch for _, material in groups)
        bulk = any(material.bulk_parts for _, material in groups)
        parts = int(shear) + int(bulk)
        per_group = [(share, int(material.viscous or material.shear_branch) + int(material.bulk_parts))
                     for share, material in groups]
        terms = sum(count for _, count in per_group)
        term_entries = sum(share * count for share, count in per_group) * displacement_entries
    return ViscoelasticCounts(
        mixed=mixed, groups=len(groups), group_matrices=len(groups) * parts,
        group_entries=int(round(grouped_share * parts * displacement_entries)),
        group_coo=int(round(grouped_share * parts * elements * element_coo)),
        grouped_elements=int(round(grouped_share * elements)), terms=terms, term_entries=int(round(term_entries)),
        column_map=column_map, **pressure, **base)


def scaled_fill(fill, counts):
    """The factor fill argument of a mixed saddle point: a counted cudss_factor.FactorFill of the u block (definition
    factor_fill on the u unknowns) times counts.fill_scale; a growth-law dimension (an int) unchanged (the true
    unknowns and entries carry it)."""
    entries = getattr(fill, 'entries', None)
    if entries is None or not counts.mixed:
        return fill
    from ...solvers.cudss_factor import FactorFill
    return FactorFill(entries * counts.fill_scale)


def viscoelastic_memory_items(counts, *, elements, local, scalar_dofs, matrix_rows, mass_entries, back_end, loss,
                              dimensions, stretched=False) -> tuple:
    """(host items, run items) of the viscoelastic group terms and the mixed blocks of counts (viscoelastic_counts) on
    top of the caller's frequency-response items (whose unknowns / entries take counts.unknowns / counts.entries):
    elements / local / scalar_dofs of the mesh (scalar_dofs: the full-mesh nodal display map), matrix_rows the u rows
    of the group matrices (2D: c x the solid map's dofs; 3D: 3 x the full map's), mass_entries the u mass's stored
    entries, back_end the resolved 'linear' back end (cuDSS: the device slots; MUMPS / PARDISO: a HostPencil), loss
    (a loss factor: the loss sum), stretched (an elastic PML of the 2D frequency response: the group matrices, B,
    M_p,g, the saddle blocks and the term values complex128, ElasticFEMSystem.stretch_arguments). Every item labelled
    from the code that allocates it."""
    from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS, LOWER_STORED_FRACTION
    host, run = [], []
    if counts.moduli:
        host.append(('viscoelastic element moduli (G0, K0, eta, viscosities and branches per element)',
                     _ELEMENT_MODULI_BYTES * elements))
    if not counts.active:
        return host, run
    index = INT32_BYTES
    value = COMPLEX128_BYTES if stretched else FLOAT64_BYTES   # matrix values (stretched: complex symmetric)
    csr = value + index
    kind = 'complex128' if stretched else 'float64'
    pattern_sum_bytes = _PATTERN_SUM_BYTES + 2 * (value - FLOAT64_BYTES)
    n_p = counts.pressure_dofs
    n_total = matrix_rows + n_p
    n_free = counts.unknowns
    if counts.group_matrices:
        coo_value = value - FLOAT64_BYTES   # the extra value bytes of a stretched (complex) COO entry
        coo = ((_COO_BYTES_3D + coo_value) * min(counts.group_coo, counts.group_matrices * COO_BATCH_ENTRIES)
               if dimensions == 3 else (_COO_BYTES_2D + coo_value) * counts.group_coo)
        host.extend([('viscoelastic group assembly COO (every group and part until its CSR)', coo),
                     (f'viscoelastic group matrices ({kind} CSR per group and part)',
                      csr * counts.group_entries + index * (matrix_rows + 1) * counts.group_matrices)])
    if counts.mixed:
        groups, grouped = counts.groups, counts.grouped_elements
        if dimensions == 3:   # PressureSpace3D: the element table, per group its elements and nodes
            space = _INT64_BYTES * (4 * elements + grouped + n_p)
            coo = _COO_BYTES_3D * min(counts.pressure_coo, (1 + groups) * COO_BATCH_ENTRIES)
        else:   # PressureSpace: the element table, per group a P1 DofMap (elements, element dofs, ids, positions)
            space = _INT64_BYTES * (3 * elements + 4 * grouped + n_p + groups * elements)
            coo = (_COO_BYTES_2D + value - FLOAT64_BYTES) * counts.pressure_coo
        coupling, pressure_mass, deviatoric = counts.coupling_entries, counts.pressure_entries, counts.group_entries
        # mixed_harmonic_operators: -B and -B^T embedded and their sum (saddle_coupling), per group D_g and M_p,g
        # embedded (index shifts: copies) and the scaled constant copies, the constant operator sum (saddle + D_g + M_p),
        # the loss sum (loss factor: the D_g and pressure blocks), the embedded mass; one indptr (n_total + 1) each
        blocks = (4 * coupling + 3 * deviatoric + 3 * pressure_mass + (2 * coupling + deviatoric + pressure_mass)
                  + mass_entries + ((deviatoric + pressure_mass) if loss else 0))
        matrices = 6 + 4 * groups
        host.extend([
            ('mixed pressure space (element table, per-group P1 dof maps)', space),
            ('mixed coupling B and pressure mass assembly COO', coo),
            (f'mixed coupling B and pressure masses M_p,g ({kind} CSR)',
             csr * (coupling + pressure_mass) + index * (n_p + 1) * (1 + groups)),
            (f'mixed saddle-point blocks in the [u; p] numbering ({kind} CSR: -B, -B^T, embedded D_g and M_p,g, scaled '
             'copies, the operator, loss and mass)', csr * blocks + index * (n_total + 1) * matrices),
            ('mixed pattern sums of the saddle-point operator (COO of every part)',
             pattern_sum_bytes * (2 * coupling + deviatoric + pressure_mass)),
            ('mixed Schur row scale (B embedded over every unknown and its squared free-dof reduction, float64 CSR; '
             'the per-frequency inverse diagonal and proxy)',
             (csr + FLOAT64_CSR_ENTRY_BYTES) * coupling + index * (n_total + n_free + 2) + 2 * FLOAT64_BYTES * n_free),
            ('mixed shown pressure (complex), its nodal display fields and point values',
             COMPLEX128_BYTES * (n_p + elements * (local + 1)) + (COMPLEX128_BYTES + 2 * FLOAT64_BYTES) * scalar_dofs),
            ('mixed padded load and held values (float64)', 2 * FLOAT64_BYTES * n_total)])
        if counts.column_map:   # m = 1: block_diag(axis ties (the system prolongation), free pressure selection)
            host.append(('mixed column map (axis ties and the free-pressure selection, float64 CSR)',
                         csr * (2 * counts.displacement_unknowns + 2 * n_p) + index * 2 * (n_total + 1)))
        if back_end not in CUDSS_BACK_ENDS and back_end not in HOST_DIRECT_BACK_ENDS:
            # host SuperLU factors the equilibrated R A R (sparse_direct._equilibrated_host): its complex values beside
            # A's, and the numpy gathers of scaled_values (int64 rows, float64 column scales)
            host.append(('mixed equilibrated saddle point of the SuperLU route (complex128 values, scaling gathers)',
                         (COMPLEX128_BYTES + _INT64_BYTES + FLOAT64_BYTES) * counts.entries))
    if counts.terms:
        terms, term_entries, entries = counts.terms, counts.term_entries, counts.entries
        reduced = (2 if counts.column_map else 1) * term_entries   # a congruence (m = 1) holds its product too
        host.extend([
            (f'frequency response reduced frequency terms (free-dof {kind} CSR per term)',
             csr * reduced + index * (n_free + 1) * terms),
            (f'frequency response term positions and values on the summed pattern (int64 + {kind})',
             (_INT64_BYTES + value) * term_entries),
            ('frequency response summed pattern with the term zeros and the position search (int64 keys)',
             csr * term_entries + pattern_sum_bytes * term_entries + _POSITION_KEY_BYTES * (entries + term_entries)),
            ('frequency response term diagonals and held-value products (float64 and complex per term)',
             (FLOAT64_BYTES + COMPLEX128_BYTES) * n_total * terms)])
        if back_end in CUDSS_BACK_ENDS or back_end in HOST_DIRECT_BACK_ENDS:   # the pencil route: slots mapped once
            union = int(math.ceil(term_entries * (LOWER_STORED_FRACTION if back_end in CUDSS_BACK_ENDS else 1.0)))
            host.append(('frequency response term slots on the pencil (union slots, per-term index and values, one '
                         "frequency's values; the pattern's int64 rows / columns while mapping)",
                         (_INT64_BYTES + _INT64_BYTES + value + COMPLEX128_BYTES) * union
                         + 2 * _INT64_BYTES * entries))
            if back_end in CUDSS_BACK_ENDS:   # factorize_pencil(extra=) uploads both per factorisation
                run.append(('frequency response per-frequency term values and slots (device)',
                            (_INT64_BYTES + COMPLEX128_BYTES) * union))
    return host, run


def viscoelastic_time_items(counts, *, local, factorizations, kind, field_points, back_end, assembly_bytes_per_coo,
                            assembly_calls, pattern_sum_stream, output_ops, loss) -> tuple:
    """core.time_estimate TimeItems of the viscoelastic group terms and the mixed blocks of counts (op counts from the
    code, never seconds): the group assembly (one strain operator per chunk, a material product per part), the mixed
    coupling / pressure-mass assembly and saddle-point blocks, the term positions and pencil slots (once), per
    factorisation the term values, the Schur row scale (on host SuperLU also the scaled copy; on cuDSS the upload)
    and per field frequency the complex-correct stress pair and the pressure outputs. assembly_bytes_per_coo /
    assembly_calls: the caller's host assembly price per COO entry and per assembly; pattern_sum_stream(parts,
    value_bytes, entries): its HostStream of a pattern sum; output_ops: its per-solve output ops (the stress pair costs
    about half of them again); loss: a mechanical_loss_factor is set (the solvers' _harmonic_moduli build the (C', C'')
    pair for it too, as the memory item booked beside the frequency-response items)."""
    from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS, LOWER_STORED_FRACTION
    items = []
    if field_points and output_ops and (counts.active or loss):   # the caller's outputs priced its real-moduli pass
        items.append(te.TimeItem('complex stress outputs (storage and loss moduli)', 'frequency', 0.5 * field_points,
                                 'exact', tuple(output_ops)))
    if not counts.active:
        return tuple(items)
    n_free = counts.unknowns
    if counts.group_matrices:
        parts = counts.group_matrices // max(counts.groups, 1)
        single = counts.group_coo / max(parts, 1)   # B and the weights once per chunk, the material product per part
        items.append(te.TimeItem('viscoelastic group stiffness assembly', 'assembly', 1.0, 'exact', (
            te.HostStream(assembly_bytes_per_coo * single * (1.0 + 0.5 * (parts - 1))),
            te.PythonCalls(assembly_calls + 20 * counts.group_matrices))))
    if counts.mixed:
        blocks = 2 * counts.coupling_entries + counts.group_entries + counts.pressure_entries
        items += [te.TimeItem('mixed pressure coupling and mass assembly', 'assembly', 1.0, 'exact', (
                      te.HostStream(assembly_bytes_per_coo * counts.pressure_coo),
                      te.PythonCalls(assembly_calls + 10 * counts.groups))),
                  te.TimeItem('mixed saddle-point blocks (embedding, scaled copies, pattern sums; once)', 'assembly',
                              1.0, 'exact', (pattern_sum_stream(2, FLOAT64_BYTES, blocks),
                                             te.HostStream(4 * FLOAT64_CSR_ENTRY_BYTES * blocks),
                                             te.PythonCalls(20 * counts.groups + 20)))]
    per_factor = []
    if counts.terms:
        entries, term_entries = counts.entries, counts.term_entries
        search = ((counts.terms + 3) * _KEY_BUILD_BYTES * entries
                  + term_entries * (_TERM_SEARCH_BYTES + _INT64_BYTES * math.log2(max(entries, 2))))
        pencil = back_end in CUDSS_BACK_ENDS
        union = term_entries * (LOWER_STORED_FRACTION if pencil else 1.0)
        once = [pattern_sum_stream(1, FLOAT64_BYTES, term_entries), te.HostStream(search),
                te.PythonCalls(10 * counts.terms)]
        if pencil or back_end in HOST_DIRECT_BACK_ENDS:   # the pencil route maps the slots once
            once.append(te.HostStream(2 * _INT64_BYTES * entries + 6 * _INT64_BYTES * term_entries))
        items.append(te.TimeItem('harmonic frequency terms on the summed pattern (positions, slots; once)', 'assembly',
                                 1.0, 'exact', tuple(once)))
        per_factor += [te.HostStream(_TERM_VALUE_BYTES * union
                                     + (3 * COMPLEX128_BYTES + 3 * FLOAT64_BYTES) * n_free * counts.terms),
                       te.PythonCalls(10 * counts.terms)]
        if pencil:   # factorize_pencil(extra=): slots and values up, the scattered add on the device
            per_factor += [te.Transfer((_INT64_BYTES + COMPLEX128_BYTES) * union),
                           te.Elementwise(3 * COMPLEX128_BYTES * union, COMPLEX128_BYTES * union, launches=3)]
    if counts.mixed:   # the Schur proxy (one product of the squared B), its inverse diagonal and the row scale
        per_factor += [te.HostStream(FLOAT64_CSR_ENTRY_BYTES * counts.coupling_entries + 6 * FLOAT64_BYTES * n_free),
                       te.PythonCalls(10)]
        if back_end not in CUDSS_BACK_ENDS and back_end not in HOST_DIRECT_BACK_ENDS:
            # host SuperLU: E = R A R entry by entry (values in / out, the row and column gathers) and the scaled solve
            # vectors
            per_factor.append(te.HostStream(2 * (COMPLEX128_BYTES + FLOAT64_BYTES) * counts.entries
                                            + _INT64_BYTES * counts.entries))
    if per_factor:
        items.append(te.TimeItem('harmonic per-frequency terms and row scales', 'factorisation', factorizations, kind,
                                 tuple(per_factor)))
    if field_points and counts.mixed:   # nodal_pressure and pressure_at of every output pass (complex)
        items.append(te.TimeItem('mixed pressure outputs', 'frequency', field_points, 'exact', (
            te.HostStream(4 * COMPLEX128_BYTES * (counts.pressure_dofs + counts.grouped_elements * (local + 1))),
            te.PythonCalls(20))))
    return tuple(items)

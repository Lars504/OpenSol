"""Model-layer definition of the 'magnetostatics' physics type (settings, features, material vocabulary, validation,
memory estimate, scalars) plus its templates and selftests; registered through definition.py. Its face / held-region
features, material Params and validate_magnetic_common are shared with definition_ac. On a 'triangles' mesh the
create builds MagneticFEMPhysics (stage 5, solver_fem.py). Imports no cupy at module level."""
from ...core.memory_estimate import (BOOL_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate, combine,
                                     diffusion_system_estimate, window_cell_count)
from ...core.time_estimate import HeldValues, diffusion_time_estimate
from ...fem.diffusion import cudss_back_end, cudss_diffusion_estimate, fem_diffusion_estimate, fem_time_system
from ...fem.magnetostatics import PLANAR_DEGREES
from ...fem.quadrature import triangle_rule
from ...model.definition_helpers import (AXISYMMETRIC_SPACE_SETTINGS, CG_DIFFUSION_LINEAR_ROLE, DISCRETIZATION_PARAM,
                                         DISCRETIZATION_SETTING, axisymmetric_issues, cg_solver_issues, element_order,
                                         evaluates_to, face_conflicts, geometry_param, held_object_conflicts,
                                         is_axisymmetric, linear_solver_direct, linear_solver_mixed,
                                         linear_solver_multigrid, linear_solver_precision, make_point_probe,
                                         on_triangles, reports_model_checks, selected_material_data,
                                         validate_points_inside)
from ...model.geometry import expand_entities
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path, resolve_values
from .materials import (BH_CURVE_PROPERTY, MAGNETIC_PROPERTY_ARRAYS, MAGNETIC_SCHEMA, OPTIONAL_MAGNETIC_SCHEMA,
                        REMANENCE_ARRAYS, SATURATION_ARRAY)
from .renderer import MAGNETIC_FIELD_INFOS
from .saturation import (ANDERSON_DEPTH, NONLINEAR_METHODS, TENSOR_NONLINEAR_METHODS, TRIANGLE_NONLINEAR_METHODS,
                         bh_curve)
from .selftests import FLUX_CONCENTRATOR_DESCRIPTION, FLUX_CONCENTRATOR_LABEL, flux_concentrator_needle_spec
from .selftests_fem import IRON_C_CORE_DESCRIPTION, IRON_C_CORE_LABEL, iron_c_core_spec

MAGNETOSTATICS_TYPE_NAME = 'magnetostatics'   # = selftests.MAGNETOSTATICS_TYPE_NAME (selftests never import this)
_OUTER_FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
_FACE_FEATURES = frozenset({'magnetic_insulation', 'symmetry'})
_ANCHOR_FEATURES = frozenset({'external_field', 'fixed_potential'})
_BOTH_GRIDS = ('tensor', 'triangles')   # FeatureType.grids of the features MagneticFEMPhysics applies too


# ---- settings ----
_GEOMETRY_PARAM = geometry_param(
    'A = A_z of currents along z, per unit z length (energy J/m, flux linkage Wb/m, inductance H/m)',
    'currents J_phi circle the axis; the solve is for the flux function psi = r A_phi on a planar operator with '
    'coefficients nu / r (2 pi psi = the flux through the circle of radius r), field A reports A_phi = psi / r, totals '
    'are whole rings (energy J, flux linkage Wb, inductance H)',
    'currents flow out of the drawing and results are per metre of depth.',
    'Currents circle the axis.',
    axis_text='psi = 0 there with the exact half-cell conductance of the axis cell; any face feature selecting x_lo '
              'is a validation error')
_SATURATION_TEXT = 'materials with saturation_polarization > 0 or a bh_curve'
_NONLINEAR_SETTINGS = (   # loop of saturating materials (core.damped_iteration; plans_overview decision 7)
    Param('nonlinear_tolerance', 'float', 1e-6, label='Saturation iteration tolerance', unit='1', min=0.0,
          min_exclusive=True,
          help=f'Stop of the saturation loop of {_SATURATION_TEXT}. Tensor mesh: converged when one proposal changes '
               'ln of the secant reluctivity nu = H / B of every cell by at most this value (a relative change); each '
               'iteration is one assembly and one CG solve warm-started at the previous A, so keep it above the CG '
               'accuracy. Triangle mesh (Newton): converged when the last step changes A by at most this fraction of '
               'max |A| and the largest free residual of the weak form is at most this fraction of the largest load '
               'or internal force. Ignored without a saturating material.',
          tip='How accurately the saturation of magnetic materials is converged, as a relative change.\n\nKeep '
              '1e-6. Only used when a material has a saturation polarisation or a B-H curve.'),
    Param('nonlinear_max_iterations', 'int', 100, label='Saturation iteration limit', min=1,
          help='Saturation iteration (tensor: fixed-point proposals, triangles: Newton steps) limit per solve of a '
               'saturating model; an unconverged loop keeps the fields of its last iterate and logs a warning (study '
               'log). Ignored without a saturating material.',
          tip='Largest number of saturation iterations; each one is a full solve.\n\n'
              'A loop that reaches the limit keeps its unfinished result and logs a warning.'),
    Param('nonlinear_method', 'choice', 'auto', label='Saturation iteration method', choices=NONLINEAR_METHODS,
          choice_labels=(('auto', 'Automatic'), ('anderson', 'Accelerated fixed point (Anderson)'),
                         ('picard', 'Damped fixed point (Picard)'), ('newton', 'Newton (triangle mesh)')),
          help=f"Solver of {_SATURATION_TEXT}. 'auto' (default): 'anderson' on a tensor mesh, 'newton' on a triangle "
               f"mesh. Tensor mesh: a fixed point on ln nu; 'anderson' mixes the last {ANDERSON_DEPTH} iterates "
               '(Anderson acceleration with the fixed relaxation below), which converges also where plain Picard '
               "oscillates (currents imposing H on iron past the knee); 'picard' applies the damped proposal with "
               "automatic halving. Triangle mesh: 'newton' minimises the convex magnetic energy with the exact "
               'element tangent (symmetric positive definite for any increasing B-H curve) and a backtracking line '
               'search on the energy, globally convergent and quadratic near the solution (piecewise-linear '
               "bh_curve tables: superlinear across their rows); one factorisation per step. 'newton' on a tensor "
               "mesh and 'anderson' / 'picard' on a triangle mesh are validation errors.",
          tip='How the saturation iteration is driven.\n\nKeep Automatic: an accelerated fixed point on rectangular '
              'cells, Newton on triangles.'),
    Param('nonlinear_relaxation', 'float', 0.5, label='Saturation iteration damping', unit='1', min=1.0 / 64.0,
          max=1.0,
          help="Fraction of each proposed change of ln nu applied, [1/64, 1] (tensor mesh only; Newton on triangles "
               "finds its step by the energy line search). 'picard': it halves automatically (down to "
               '1/64) while the change grows and recovers up to this value once it shrinks; Picard converges without '
               'damping where the flux is imposed (an air gap, an applied field) but oscillates where the current '
               'imposes H on iron past the knee of the curve: it needs a value below 2 / (1 + d ln nu / d ln B) '
               "(about 0.5 at B ~ 0.7 J_s), lower it (0.2) deep into saturation. 'anderson': the fixed mixing factor "
               '(0.5 - 1).',
          tip='Fraction of each correction that is applied, between 0 and 1.\n\nKeep 0.5; lower it if the '
              'iteration does not settle. Not used on a triangle mesh.'),
)
_METHOD_LABELS = dict(next(param for param in _NONLINEAR_SETTINGS if param.name == 'nonlinear_method').choice_labels)
_SETTINGS = (
    _GEOMETRY_PARAM,
    DISCRETIZATION_PARAM,
    *_NONLINEAR_SETTINGS,
)


# ---- material vocabulary ----
def _material_property(name, label, unit, help_text, tip, positive=False, arrays=MAGNETIC_PROPERTY_ARRAYS):
    """Magnetic material property Param (default None: unset properties are never written); arrays = the property ->
    MaterialGrid arrays map (2D, or magnetostatics_3d's)."""
    return Param(name, 'float', None, label=label, unit=unit, min=0.0 if positive else None,
                 min_exclusive=positive, help=help_text, tip=tip, optional=True, field_dependent=True,
                 state_dependent=True, expands_to=arrays[name])


_MU_DC_HELP = (
    'Isotropic static relative permeability (> 0): sets mu_dc_xx and mu_dc_yy (component entries override it); unset '
    '= 1 (vacuum), so every cell takes part in the solve. NOT the optical mu_r of the wave physics (kept 1 there). '
    'Typical: air, copper, aluminium 1; ferrite 1e3 - 1e4; electrical steel 4e3; Metglas (amorphous Co / Fe) 1e4 - '
    '1e5; mu-metal 5e4. A large contrast is solved exactly by the finite volumes but slows Jacobi-PCG (iterations '
    'grow with the contrast): tighten the tolerance (1e-10). With saturation_polarization it is the initial '
    '(small-field) permeability of the B-H curve.')
_MU_DC_TIP = ('How strongly the material concentrates magnetic flux; empty means vacuum, 1.\n\n'
              'Typical: air and copper 1, ferrite 1000 to 10000, Metglas up to 100000.')
_MU_COMPONENT_TIP = ("Static relative permeability along {axis} of a material that magnetises differently along x and "
                     "y. Replaces 'Static relative permeability' for this direction.")
_REMANENCE_HELP = (
    'Remanent flux density B_r [T] along {axis} of a permanent magnet: H = (B - B_r) / (mu0 mu_dc), entered as face '
    'sources at the magnet boundary (exact for a uniformly magnetised region, also across its edges). Typical: '
    'NdFeB 1.2 - 1.4 T with mu_dc 1.05, SmCo 1.0 T, ferrite 0.4 T with mu_dc 1.1. Unset = 0 (the remanence arrays '
    'are allocated only when a material sets one).{extra}')
_SATURATION_HELP = (
    'Saturation polarisation J_s [T] (> 0) of an anhysteretic B-H curve (magnetostatics only): the Frohlich-Kennelly '
    'curve B = mu0 H + J_s a H / (1 + a H) with a = mu0 (mu_dc - 1) / J_s, i.e. initial permeability mu_dc, B -> mu0 '
    'H + J_s in strong fields, no hysteresis. Tensor mesh: per axis the same curve with mu_dc_xx / mu_dc_yy and the '
    'cell-centre |B|, solved by a fixed point on ln of the secant reluctivity nu = H(|B|) / |B|, Anderson-accelerated '
    '(settings nonlinear_method, nonlinear_tolerance, nonlinear_max_iterations, nonlinear_relaxation); the energy '
    'scalar and coil inductances are then secant values '
    '((1/2) A^T K A, flux linkage / current), not the integral of H dB. Triangle mesh: H = nu(|B|) B at every '
    'quadrature point (isotropic: mu_dc_xx must equal mu_dc_yy, validation error otherwise), solved by Newton; the '
    'energy scalar is the integral of H dB. Unset = linear (the array is allocated only '
    'when a material sets it); mu_dc <= 1 stays linear. Not combinable with a remanence in the same material '
    '(validation error); ignored by magnetic_fields_ac. Typical: iron 2.15, electrical steel 2.0, Metglas 0.6 - 1.6, '
    'Mn-Zn ferrite 0.4 - 0.5, NiFe (mu-metal) 0.8.')
_BH_CURVE_HELP = (
    'Tabulated anhysteretic B-H curve (magnetostatics only): rows [H, B] with H [A/m] and B [T] both strictly '
    'increasing, H, B > 0 except an optional first row [0, 0] (the curve always starts at the origin), e.g. [[100, '
    '0.5], [300, 1.0], [1000, 1.4], [5000, 1.7]]. H(B) is interpolated linearly between the rows; beyond the last row '
    'B = B_n + mu0 (H - H_n) (fully saturated), so end the table well into saturation. The cells follow the secant '
    'reluctivity nu = |H| / B(|H|) on both axes (isotropic; mu_dc / mu_dc_xx / mu_dc_yy are ignored there), solved '
    'by the same fixed point on ln nu as saturation_polarization (settings nonlinear_*; each proposal reads the '
    'cell-centre |H|, which stays stable also beyond the last row); the first iterate is the '
    'first segment\'s slope B_1 / H_1. Energy and inductances are secant values on a tensor mesh. On a triangle mesh '
    'Newton reads the curve B-indexed (H(B) linear per segment, the segment slope as tangent, the exact energy '
    'integral). Not combinable with '
    'saturation_polarization or a remanence in the same material (validation error); ignored by magnetic_fields_ac. '
    'A constant table per material (no x, y or field dependence; entries may be parameter expressions).')
_BH_CURVE = Param(BH_CURVE_PROPERTY, 'table', None, label='B-H curve', min=0.0, optional=True, increasing=True,
                  columns=(('H', 'A/m'), ('B', 'T')), help=_BH_CURVE_HELP,
                  tip='Measured magnetisation curve of a soft magnetic material: one pair of field strength H and flux '
                      'density B per line, both increasing.\n\nThe curve starts at zero; above the last pair the '
                      'material is treated as fully saturated. Replaces the static relative permeability.')
_MATERIAL_PROPERTIES = (
    _material_property('mu_dc', 'Static relative permeability', '1', _MU_DC_HELP, _MU_DC_TIP, positive=True),
    _material_property('mu_dc_xx', 'Static relative permeability along x', '1',
                       'Static relative permeability along x (axisymmetric: r), > 0; overrides mu_dc for this axis.',
                       _MU_COMPONENT_TIP.format(axis='x'), positive=True),
    _material_property('mu_dc_yy', 'Static relative permeability along y', '1',
                       'Static relative permeability along y (axisymmetric: z), > 0; overrides mu_dc for this axis.',
                       _MU_COMPONENT_TIP.format(axis='y'), positive=True),
    _material_property('remanence_x', 'Remanence along x', 'T',
                       _REMANENCE_HELP.format(axis='x', extra=' Axisymmetric geometry: not available (validation '
                                                              'error; magnetise along z with remanence_y).'),
                       'Built-in magnetisation of a permanent magnet along x, as a flux density.\n\n'
                       'Typical: neodymium magnets about 1.3. Not available in axisymmetric models.'),
    _material_property('remanence_y', 'Remanence along y', 'T',
                       _REMANENCE_HELP.format(axis='y (axisymmetric: along the axis z)', extra=''),
                       'Built-in magnetisation of a permanent magnet along y (along the axis in axisymmetric models), '
                       'as a flux density.\n\nTypical: neodymium magnets about 1.3.'),
    _material_property(SATURATION_ARRAY, 'Saturation polarisation', 'T', _SATURATION_HELP,
                       'Flux density the magnetisation of the material approaches in strong fields; empty means a '
                       'linear material.\n\nTypical: iron 2.1, electrical steel 2.0, Metglas 0.6 to 1.6, ferrite 0.4.',
                       positive=True),
    _BH_CURVE,
)


# ---- feature texts ----
_INSULATION_DESCRIPTION = (
    'Magnetic insulation A = 0 on the selected outer faces (the default of every face no feature sets): the normal '
    'flux density is 0, field lines run along the face. It is the mirror image of an ODD current distribution (J_z '
    'changes sign across the plane, e.g. the midplane between a conductor and its return current) and stands in for a '
    'distant boundary far from the sources. Axisymmetric: psi = r A_phi = 0, so no net flux crosses the circle of an '
    'x_hi face (the return flux lies inside it); x_lo is the axis (psi = 0 already; selecting it is a validation '
    'error). Anchors the solve.')
_SYMMETRY_DESCRIPTION = (
    'Symmetry dA/dn = 0 (Neumann) on the selected outer faces: the tangential flux density is 0, field lines cross the '
    'face at right angles. It is the mirror image of an EVEN current distribution (J_z the same on both sides, e.g. '
    'the midplane of two parallel conductors carrying the same current) and models a face next to an ideal iron yoke. '
    'Does not anchor the solve: a model whose four faces are all symmetry needs an external_field or fixed_potential. '
    'Axisymmetric: not allowed on x_lo (the axis).')
_EXTERNAL_FIELD_DESCRIPTION = (
    "Uniform applied flux density (B_x, B_y) [T], selection 'all' only, at most one per physics: the outermost window "
    'cells are held at the external potential (planar A = B_x y - B_y x evaluated at their faces toward the free '
    'cells, so a uniform field is reproduced exactly; axisymmetric psi = B_y r^2 / 2 on the z faces and the exact '
    'half-cell value on x_hi, B_x must be 0; exact on uniform r cells only: graded r cells add a first-order error '
    'where the radial width changes). Inside, objects perturb the field; keep them several object sizes '
    'away from the domain edges, since the ring holds the undisturbed field there. The ring cells leave the solve: '
    'B and H read 0 in them, and face conditions of the outer faces have no effect (the axis stays psi = 0). '
    'Anchors the solve. Typical: the Earth field ~5e-5 T, a Helmholtz coil 1e-4 - 1e-2 T.')
_FIXED_POTENTIAL_DESCRIPTION = (
    'Holds the vector potential on the selected cells for the whole run (they leave the solve; B and H read 0 in '
    'them): planar A_z [T m], axisymmetric the flux function psi = r A_phi [T m^2] (2 pi psi = the flux through the '
    'circle of radius r). A region held at one value excludes all flux, like a perfect diamagnet or superconductor; '
    'two regions held at A_1 and A_2 carry the flux (A_1 - A_2) per m of depth between them (planar). Anchors the '
    'solve.')
_CURRENT_SOURCE_DESCRIPTION = (
    'Applied current density J [A/m^2] on the selected cells, added to field J_applied (several sources and coils '
    'add; the coupled slot current_density, field J_coupled, is separate): planar J_z out of the drawing (> 0 along '
    '+z: B circles counter-clockwise), axisymmetric J_phi circling the axis (> 0 counter-clockwise seen from +z: B_z > '
    '0 inside). May depend on x and y.')
_COIL_DESCRIPTION = (
    'Stranded coil (many thin insulated turns): J = turns * current / area spread uniformly over the selected cells '
    '(area = their x-y, axisymmetric r-z, cross-section; skin and proximity effects need the solid conductors of '
    'magnetic_fields_ac). current [A] > 0 along +z (axisymmetric +phi). A planar coil without a return is a net '
    'current per m of depth that returns through the insulation faces; model a go / return pair as two coil features '
    'with opposite currents. Result \'table\': one row [feature, current, turns, flux_linkage, inductance] of the '
    'latest solve and the same extras: flux_linkage = turns / area * integral of A over the coil (axisymmetric '
    '2 pi psi: the flux through each turn) in Wb/m (planar) or Wb, inductance = flux_linkage / current in H/m or H '
    '(nan at zero current). For coils in series carrying +I and -I the inductances of the features add up to the '
    'loop inductance 2 W / I^2 (W the energy scalar); with several independent currents a coil\'s value includes the '
    'flux of the others (mutual terms). Scalars <feature>_flux_linkage and <feature>_inductance carry the same '
    'numbers.')
_PROBE_DESCRIPTION = ('Vector potential A (planar A_z, axisymmetric A_phi) [T m] of the cell containing the point, one '
                      'sample per re-solve plus one at the end of the run (stationary: per fixed-point iteration, all '
                      'at t = 0). On a triangle mesh: A at the mesh node of the point, once per solve.')
_SLOT_DOCS = ('A/m^2', 'Coupled current density J_z (axisymmetric J_phi; field J_coupled), added to the '
                       'current_source and coil features (field J_applied); each transfer replaces it.')
_SCALARS_HELP = (
    "energy: magnetic field energy of the free cells, (1/2) A^T K A of the discrete solution = integral of B^2 / "
    "(2 mu0 mu_dc) (J/m planar, J axisymmetric; with remanence the energy of B alone, not the magnet's co-energy; held "
    "cells and the external-field ring excluded, held-free faces count their free half cell; with saturating "
    "materials the secant value at the converged nu, not the integral of H dB; on a triangle mesh the integral of the "
    "energy density w(B) over the window elements outside fixed_potential regions, i.e. the integral of H dB also "
    "when saturating); "
    "<coil>_flux_linkage (Wb/m planar, Wb axisymmetric) and <coil>_inductance (H/m, H) of every enabled coil "
    "feature, as its table result.")
_PHYSICS_DESCRIPTION = (
    '2D magnetostatics on the physical window (finite volumes, Jacobi-PCG): -div(nu grad A) = J with nu = 1 / (mu0 '
    'mu_dc) (planar A_z [T m] of currents J_z along z; c_xx = nu_yy, c_yy = nu_xx, since B_y = -dA/dx sees mu_yy), or '
    'axisymmetric the flux function psi = r A_phi of currents J_phi on the planar operator with coefficients nu / r '
    '(setting geometry). B = curl A (planar B_x = dA/dy, B_y = -dA/dx; axisymmetric B_r = -(1/r) dpsi/dz, B_z = (1/r) '
    'dpsi/dr) and H = nu (B - B_r) from the face fluxes averaged to the cell centres; w_mag = (nu_xx B_x^2 + nu_yy '
    'B_y^2) / 2. Every outer face is magnetic insulation (A = 0) unless a feature sets it; sources are current_source '
    'and coil features, the coupled slot current_density and material remanence; external_field holds the outer '
    'ring at a uniform applied field. mu_dc may depend on x, y and state fields (a stationary study iterates); a '
    'material with saturation_polarization (analytic) or a bh_curve (table) follows an anhysteretic B-H curve, solved '
    'by an Anderson-accelerated (or '
    'damped Picard) fixed point on ln nu (settings nonlinear_*; iterations in the study summary as '
    'nonlinear_iterations, an unconverged loop logs a '
    'warning). A time-dependent study re-solves only when mu_dc, remanence, a source or a held value changed. Fields '
    'are edge-extended into an EM PML ring. Scalars: energy and per coil flux linkage and inductance. On a triangle '
    'mesh (MeshSpec kind triangles) finite elements of the discretization setting (P2 default, P1) solve for planar A_z '
    'or axisymmetric A_phi itself (not psi) in B-space: B = curl A at quadrature points, H = diag(nu) (B - B_r) '
    '(no xx / yy swap there), saturating materials isotropic H = nu(|B|) B by exact-tangent Newton with an energy line '
    'search (nonlinear_method auto = newton); faces act on the window edge, external_field holds A_ext on the '
    'window-edge nodes (the whole window is free), fixed_potential holds psi / r on the element nodes, state fields '
    'are per element (A at the centroid, B / H / w_mag element means; plots show B and H averaged to the nodes), the '
    'energy scalar is the integral of H dB, and an axisymmetric radial remanence is allowed. The instance is usually '
    'named magnetic.')
_PHYSICS_SUMMARY = ('Static magnetic fields of coils, currents and permanent magnets in magnetic materials, planar or '
                    'rotationally symmetric. Gives the flux density, the field energy and coil inductances.')


# ---- feature apply / result callbacks (worker thread, GPU) ----
def _face_setter(kind):
    """Apply callback of a face feature: Dirichlet ('dirichlet', 0) or Neumann ('neumann') on the selected outer
    faces (2D A: insulation is Dirichlet, axisymmetric x_lo a validation error; 3D phi_m: symmetry is Dirichlet)."""
    def apply_faces(ctx, solver, feature_spec, values, selection):
        from ...core.fv_diffusion import BCSide   # lazy: keeps cupy out of this module's import
        for face in selection.outer_faces:
            solver.set_face_bc(face, BCSide(kind, value=0.0))
    return apply_faces


_apply_insulation, _apply_symmetry = _face_setter('dirichlet'), _face_setter('neumann')


def _apply_external_field(ctx, solver, feature_spec, values, selection):
    solver.set_external_field(values['B_x'], values['B_y'])
    return None


def _apply_fixed_potential(ctx, solver, feature_spec, values, selection):
    solver.add_fixed_potential(ctx.domain_mask(selection.entities), values['potential'])
    return None


def _apply_current_source(ctx, solver, feature_spec, values, selection):
    source_mask = ctx.domain_mask(selection.entities)
    solver.add_source(source_mask, ctx.eval_field(values['density'], mask=source_mask))
    return None


def _apply_coil(ctx, solver, feature_spec, values, selection):
    return solver.add_coil(feature_spec.name, ctx.domain_mask(selection.entities), values['turns'], values['current'])


def _coil_result(session, solver, coil, feature_spec, options):
    """One table row [feature, current, turns, flux_linkage, inductance] of the latest solve (per m of depth on a
    planar 2D solver; whole values axisymmetric and in 3D, whose solvers have no axisymmetric attribute)."""
    result = solver.coil_results(coil)
    linkage_unit, inductance_unit = ('Wb', 'H') if getattr(solver, 'axisymmetric', True) else ('Wb/m', 'H/m')
    return {'kind': 'table', 'columns': ['feature', 'current', 'turns', 'flux_linkage', 'inductance'],
            'units': ['', 'A', '1', linkage_unit, inductance_unit],
            'rows': [[coil.name, result['current'], result['turns'], result['flux_linkage'], result['inductance']]],
            **result}


def _coil_memory(values, n_selected_cells, mesh):
    """The coil's window bool mask (kept for its flux linkage; triangles: a host bool per element)."""
    if mesh.kind == 'triangles':
        return MemoryEstimate.from_items(host_items=(('coil element mask (host bool)', BOOL_BYTES * mesh.n_elements),))
    return MemoryEstimate.from_items((('coil mask (bool window)', BOOL_BYTES * window_cell_count(mesh)),))


_apply_probe, _probe_result = make_point_probe('vector potential', 'T m')


# ---- validation (CPU only) ----
def _selection_entities(vctx, feature):
    try:
        return expand_entities(vctx.spec, feature.selection)
    except (ValueError, KeyError, TypeError):   # bad selection: reported by the selection checks
        return []


def _validate_whole_window(vctx, physics_spec, feature_spec, values):
    """Selection 'all' only (external_field of the 2D and 3D types)."""
    entities = _selection_entities(vctx, feature_spec)
    if entities and entities != ['all']:
        return [Issue('error', feature_path(physics_spec, feature_spec),
                      "external_field holds the outer ring of the whole window: select 'all' only")]
    return []


def _validate_external_field(vctx, physics_spec, feature_spec, values):
    """Selection 'all' only; axisymmetric: B_x = 0."""
    path = feature_path(physics_spec, feature_spec)
    issues = _validate_whole_window(vctx, physics_spec, feature_spec, values)
    try:   # magnetostatics or magnetic_fields_ac (they share this feature)
        settings = vctx.registry.physics_type(physics_spec.type).settings
    except KeyError:
        settings = _SETTINGS
    if is_axisymmetric(vctx.values(settings, physics_spec.settings)) and values.get('B_x', 0.0) != 0.0:
        issues.append(Issue('error', path, 'an axisymmetric external field is axial only: set B_x (= B_r) to 0 and '
                                           'the axial field in B_y'))
    return issues


def _face_kinds(vctx, physics_spec):
    """Outer face -> feature type that sets it last (unset faces: magnetic_insulation, the default)."""
    kinds = dict.fromkeys(_OUTER_FACES, 'magnetic_insulation')
    for feature in physics_spec.features:
        if feature.enabled and feature.type in _FACE_FEATURES:
            for face in _selection_entities(vctx, feature):
                if face in kinds:
                    kinds[face] = feature.type
    return kinds


def _remanence_x_materials(vctx):
    """Names of the selected spec materials (library merged) that may set a non-zero remanence_x."""
    return [material.name for material, properties, _ in selected_material_data(vctx.spec)
            if evaluates_to(properties.get('remanence_x'), vctx.parameters, lambda value: value != 0.0)]


def _axisymmetric_issues(vctx, physics_spec, path, type_text):
    """Shared (r, z) rules (a feature on x_lo, the axis with psi = 0, is an error; planar partners warn) plus, for
    magnetostatics, no radial remanence."""
    issues = axisymmetric_issues(vctx, physics_spec, path, type_text, axis_severity='error',
                                 axis_consequence='psi = r A_phi = 0 is fixed there; remove x_lo from the selection')
    if (physics_spec.type == MAGNETOSTATICS_TYPE_NAME and not on_triangles(vctx, physics_spec)   # FEM: B_r load
            and reports_model_checks(vctx, physics_spec, MAGNETOSTATICS_TYPE_NAME)):
        for material_name in _remanence_x_materials(vctx):
            issues.append(Issue('error', ('material', material_name),
                                f"material '{material_name}' sets remanence_x, which axisymmetric magnetostatics "
                                f"'{physics_spec.name}' does not support (radial magnetisation): magnetise along z "
                                '(remanence_y) or remove it'))
    return issues


def _bh_curve_rows(raw_rows, parameters):
    """Resolved bh_curve rows [(H, B), ...], None when the value fails the table checks (reported at the material)."""
    try:
        return resolve_values((_BH_CURVE,), {BH_CURVE_PROPERTY: raw_rows}, parameters)[BH_CURVE_PROPERTY]
    except (ValueError, TypeError, ArithmeticError):
        return None


def _curve_material_issues(material, properties, parameters, remanence_names, permeability_names):
    """One selected material's B-H curve issues (2D and 3D): a saturation (saturation_polarization or bh_curve) with
    a remanence of remanence_names, both saturations, a bh_curve the curve rules reject (errors); one of
    permeability_names beside a bh_curve (warning: ignored)."""
    path = ('material', material.name)
    issues = []
    saturates = evaluates_to(properties.get(SATURATION_ARRAY), parameters, lambda value: value > 0.0)
    raw_rows = properties.get(BH_CURVE_PROPERTY)
    tabulated = raw_rows is not None
    if (saturates or tabulated) and any(evaluates_to(properties.get(name), parameters, lambda value: value != 0.0)
                                        for name in remanence_names):
        issues.append(Issue('error', path, f"material '{material.name}' sets "
                                           f"{BH_CURVE_PROPERTY if tabulated else SATURATION_ARRAY} and a "
                                           'remanence: a saturating permanent magnet is not modelled (remove one)'))
    if not tabulated:
        return issues
    if saturates:
        issues.append(Issue('error', path, f"material '{material.name}' sets both {SATURATION_ARRAY} and "
                                           f'{BH_CURVE_PROPERTY}: give one B-H curve (remove one)'))
    set_names = [name for name in permeability_names if properties.get(name) is not None]
    if set_names:
        issues.append(Issue('warning', path, f"material '{material.name}': {', '.join(set_names)} "
                                             f'{"is" if len(set_names) == 1 else "are"} ignored in its '
                                             f'cells, where {BH_CURVE_PROPERTY} sets the permeability'))
    rows = _bh_curve_rows(raw_rows, parameters)
    if rows is not None:
        try:
            bh_curve(rows)
        except ValueError as exc:
            issues.append(Issue('error', path, f"property '{BH_CURVE_PROPERTY}': {exc}"))
    return issues


def _saturation_issues(vctx, physics_spec):
    """Once per model, per selected material: _curve_material_issues."""
    if not reports_model_checks(vctx, physics_spec, MAGNETOSTATICS_TYPE_NAME):
        return []
    issues = []
    for material, properties, _ in selected_material_data(vctx.spec):
        issues.extend(_curve_material_issues(material, properties, vctx.parameters, REMANENCE_ARRAYS,
                                             ('mu_dc', 'mu_dc_xx', 'mu_dc_yy')))
    return issues


def _validate_magnetostatics(vctx, physics_spec, settings_values):
    """validate_magnetic_common plus the saturation material rule and the mesh-kind rules of nonlinear_method."""
    return (validate_magnetic_common(vctx, physics_spec, settings_values, 'magnetostatics')
            + _saturation_issues(vctx, physics_spec) + _mesh_kind_issues(vctx, physics_spec, settings_values))


def _selected_material_properties(spec):
    """(material, properties) of the selected materials, library merged (failures are reported at the material)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    selected = []
    for material in spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):
            continue
        if material.selection:
            selected.append((material, properties))
    return selected


def _saturating_materials(spec, parameters):
    """(material, properties) of the selected materials (library merged) with saturation_polarization > 0 or a
    bh_curve."""
    return [(material, properties) for material, properties in _selected_material_properties(spec)
            if properties.get(BH_CURVE_PROPERTY) is not None
            or evaluates_to(properties.get(SATURATION_ARRAY), parameters, lambda value: value > 0.0)]


def _method_misfit(settings_values, triangles):
    """The nonlinear_method set for the other mesh kind ('newton' runs on triangles only, 'anderson' / 'picard' on
    tensor meshes only), else None."""
    method = (settings_values or {}).get('nonlinear_method', 'auto')
    allowed = TRIANGLE_NONLINEAR_METHODS if triangles else TENSOR_NONLINEAR_METHODS
    return method if method != 'auto' and method not in allowed else None


def _anisotropic_saturation(saturating, parameters):
    """Names of the saturating materials with saturation_polarization > 0 and mu_dc_xx != mu_dc_yy (the triangle-mesh
    curve is isotropic)."""
    names = []
    for material, properties in saturating:
        if not evaluates_to(properties.get(SATURATION_ARRAY), parameters, lambda value: value > 0.0):
            continue
        components = [properties.get(name, properties.get('mu_dc')) for name in ('mu_dc_xx', 'mu_dc_yy')]
        if components[0] != components[1]:
            names.append(material.name)
    return names


def _mesh_kind_issues(vctx, physics_spec, settings_values):
    """nonlinear_method per mesh kind (_method_misfit); on triangles an isotropic mu_dc in every saturating material
    (once per model)."""
    triangles = on_triangles(vctx, physics_spec)
    allowed = TRIANGLE_NONLINEAR_METHODS if triangles else TENSOR_NONLINEAR_METHODS
    issues = []
    misfit = _method_misfit(settings_values, triangles)
    if misfit is not None:
        issues.append(Issue('error', ('physics', physics_spec.name),
                            f"nonlinear_method '{misfit}' runs on {'tensor meshes' if triangles else 'triangle meshes'} "
                            f"only: use 'auto' or one of {', '.join(allowed)} on this "
                            f"{'triangle' if triangles else 'tensor'} mesh"))
    if not triangles or not reports_model_checks(vctx, physics_spec, MAGNETOSTATICS_TYPE_NAME):
        return issues
    for name in _anisotropic_saturation(_saturating_materials(vctx.spec, vctx.parameters), vctx.parameters):
        issues.append(Issue('error', ('material', name),
                            f"material '{name}' sets {SATURATION_ARRAY} with mu_dc_xx != mu_dc_yy: on a triangle mesh "
                            'the saturating curve is isotropic (one energy of |B|); give one mu_dc'))
    return issues


def _mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
    """GUI text of the discretisation and saturation method on mesh_kind, plus what that method cannot run here (the
    _mesh_kind_issues rules; on tensor meshes Picard with a saturating material as a caution)."""
    triangles = mesh_kind == 'triangles'
    order = 'quadratic' if element_order(settings_values) == 2 else 'linear'
    text = (f'Finite elements on triangles ({order} elements).' if triangles
            else 'Finite volumes on rectangular cells.')
    saturating = _saturating_materials(spec, parameters)
    method = (settings_values or {}).get('nonlinear_method', 'auto')
    if not saturating:
        text += ' No material saturates, so this is one linear solve and the saturation iteration settings are unused.'
    else:
        names = ', '.join(material.name for material, _ in saturating)
        iteration = ("Newton's method with a line search on the magnetic energy" if triangles
                     else 'a damped fixed-point iteration (Picard)' if method == 'picard'
                     else 'an accelerated fixed-point iteration (Anderson)')
        text += f' Saturation of {names} is solved by {iteration}.'
    problems = []
    misfit = _method_misfit(settings_values, triangles)
    if misfit is not None:
        problems.append(f"Saturation iteration method '{_METHOD_LABELS.get(misfit, misfit)}' runs only on "
                        f"{'rectangular cells' if triangles else 'triangle meshes'}: choose Automatic.")
    if triangles:
        problems.extend(f'Material {name} saturates with different permeabilities along x and y, but on triangles '
                        'the saturation curve is the same in every direction: give it one permeability.'
                        for name in _anisotropic_saturation(saturating, parameters))
    elif saturating and method == 'picard':
        problems.append('The damped fixed point can oscillate where currents drive iron past the knee of its curve; '
                        'Automatic (Anderson) converges there.')
    return text, problems


def validate_magnetic_common(vctx, physics_spec, settings_values, type_text):
    """Anchor (planar: not every face symmetry without a held region), one external field, precision / CG limit (when
    the type has them), face / held-object conflicts, axisymmetric rules; shared with magnetic_fields_ac."""
    path = ('physics', physics_spec.name)
    issues = []
    enabled = [feature for feature in physics_spec.features if feature.enabled]
    n_external = sum(1 for feature in enabled if feature.type == 'external_field')
    if n_external > 1:
        issues.append(Issue('error', path, f'{n_external} external_field features: at most one per physics (each '
                                           'holds the same outer ring)'))
    axisymmetric = is_axisymmetric(settings_values)
    anchored = (axisymmetric or any(feature.type in _ANCHOR_FEATURES for feature in enabled)
                or 'magnetic_insulation' in _face_kinds(vctx, physics_spec).values())
    if not anchored:
        issues.append(Issue('error', path, 'every outer face is a symmetry face and nothing holds the potential, so A '
                                           'is undetermined (singular system): keep magnetic insulation on one face, '
                                           'or add an external_field or fixed_potential'))
    issues.extend(cg_solver_issues(vctx, physics_spec, path))
    issues.extend(face_conflicts(vctx, physics_spec, _FACE_FEATURES, path))
    issues.extend(held_object_conflicts(vctx, physics_spec, frozenset({'fixed_potential'}),
                                        frozenset({'current_source', 'coil'}), path))
    if axisymmetric:
        issues.extend(_axisymmetric_issues(vctx, physics_spec, path, type_text))
    return issues


# ---- physics factory callbacks ----
def _create(ctx, physics_spec, values):
    """MagneticPhysics (tensor) or MagneticFEMPhysics (triangles, the discretization order) plus one add_bh_curve per
    material region with a bh_curve (2D regions keep their masks until the build releases its caches)."""
    nonlinear = {'nonlinear_tolerance': values['nonlinear_tolerance'],
                 'nonlinear_max_iterations': values['nonlinear_max_iterations'],
                 'nonlinear_method': values['nonlinear_method']}
    linear_solver = ctx.solvers(physics_spec.name)['linear']
    if ctx.domain.grid_kind == 'triangles':
        from .solver_fem import MagneticFEMPhysics   # lazy: the solver imports cupy
        solver = MagneticFEMPhysics(ctx.domain, axisymmetric=is_axisymmetric(values),
                                    discretization=values[DISCRETIZATION_SETTING], linear_solver=linear_solver,
                                    **nonlinear)
    else:
        from .solver import MagneticPhysics   # lazy: the solver imports cupy
        solver = MagneticPhysics(ctx.domain, axisymmetric=is_axisymmetric(values), linear_solver=linear_solver,
                                 nonlinear_relaxation=values['nonlinear_relaxation'], **nonlinear)
    for region in ctx.material_regions:
        raw_rows = region.properties.get(BH_CURVE_PROPERTY)
        if raw_rows is not None:
            rows = resolve_values((_BH_CURVE,), {BH_CURVE_PROPERTY: raw_rows}, ctx.parameters)[BH_CURVE_PROPERTY]
            solver.add_bh_curve(region.mask, rows)
    return solver


_N_FIELDS = 9   # A, B_x, B_y, B_abs, H_x, H_y, w_mag, J_coupled, J_applied
# loop state (2), J_s, proposal (2), update (2), |B| and mu copies of one proposal; the state, J_s and the Anderson
# history are compact on the saturating cells (solver._nonlinear_flat): the window is their upper bound here
_SATURATION_WINDOW_ARRAYS = 9
# Anderson history per state component (2 per window cell): dX, dF (depth each), weighted (depth + 1), x, f previous
_ANDERSON_STATE_ARRAYS = 3 * ANDERSON_DEPTH + 3
_CURVE_PROPOSAL_ARRAYS = 4   # float64 window temporaries of one table proposal (|H|, B(|H|), its extension, nu)


def _bh_curve_material_count(spec):
    """Selected materials (library merged) setting a bh_curve: one window mask each in the build."""
    return sum(1 for _, properties in _selected_material_properties(spec)
               if properties.get(BH_CURVE_PROPERTY) is not None)


def _memory_estimate(settings_values, mesh, context):
    """MagneticFields2D (float32 full grid) + the window DiffusionSystem + nu_xx / nu_yy, mu_dc_xx / _yy and |B| (solve
    dtype window) + one bool window mask per bh_curve material; run items: the remanence face terms and rhs when a
    material sets remanence, the saturation loop state, J_s, proposal temporaries and the Anderson history when one
    sets saturation_polarization or a bh_curve; a cuDSS 'linear' factor (fem.diffusion.cudss_diffusion_estimate).
    Triangles: _triangle_memory_estimate."""
    if mesh.kind == 'triangles':
        return _triangle_memory_estimate(settings_values, mesh, context)
    window_cells = window_cell_count(mesh)
    precision = linear_solver_precision(context)
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    remanence = any(context.material_sets(name) for name in REMANENCE_ARRAYS)
    n_curves = _bh_curve_material_count(context.spec)
    saturation = context.material_sets(SATURATION_ARRAY) or n_curves > 0
    saturation_arrays = _SATURATION_WINDOW_ARRAYS + (
        2 * _ANDERSON_STATE_ARRAYS if settings_values.get('nonlinear_method') in ('auto', 'anderson') else 0) + (
        _CURVE_PROPOSAL_ARRAYS if n_curves else 0)
    own = MemoryEstimate.from_items(
        ((f'fields ({_N_FIELDS} x float32)', _N_FIELDS * FLOAT32_BYTES * int(mesh.n_cells)),
         ('reluctivities, permeabilities and |B| (window)', 5 * value_bytes * window_cells),
         ('B-H curve masks (bool window)', BOOL_BYTES * window_cells * n_curves)),
        run_items=(('remanence face terms and rhs (window)', 3 * value_bytes * window_cells * int(remanence)),
                   ('saturation state, J_s, proposal and Anderson history (float64 window)',
                    saturation_arrays * FLOAT64_BYTES * window_cells * int(saturation))))
    parts = [own, diffusion_system_estimate(window_cells, precision, mixed=linear_solver_mixed(context),
                                            multigrid=linear_solver_multigrid(context),
                                            direct=linear_solver_direct(context))]
    cudss = cudss_back_end(context.solvers)
    if cudss:   # DiffusionDirectSolver on the kept A_solve, refactorised per saturation iteration
        parts.append(cudss_diffusion_estimate(window_cells, 5 * window_cells, 2, cudss, input_copy=False,
                                              refactorizes=bool(saturation)))
    return combine(parts)


SATURATION_ITERATIONS_GUESS = 15   # run-time estimate: saturation loop solves (calibrate from the physics counters)


def saturation_iterations(settings_values, context):
    """(guess, limit) of the saturation loop's solves when a material saturates (saturation_polarization or a
    bh_curve), else None (one linear solve); shared by magnetostatics_3d."""
    if not (context.material_sets(SATURATION_ARRAY) or _bh_curve_material_count(context.spec) > 0):
        return None
    limit = int(settings_values['nonlinear_max_iterations'])
    return min(SATURATION_ITERATIONS_GUESS, limit), limit


NEWTON_ITERATIONS_GUESS = 10   # triangles: Newton steps per saturating solve (uncalibrated; the selftests saw 7-10)
# where A_z / psi is held, for the Jacobi-PCG condition estimate: every outer face (insulation) unless a symmetry
# face frees it, the external-field ring on the window edge, fixed_potential regions
_HELD_POTENTIAL = HeldValues(feature_types=frozenset({'magnetic_insulation', 'fixed_potential'}),
                             edge_types=frozenset({'external_field'}), outer_faces_held=True,
                             free_types=frozenset({'symmetry'}))


def _time_estimate(settings_values, mesh, context):
    """The DiffusionSystem recipe on the window (A_z / psi), times the saturation loop; triangles: at the Lagrange
    unknowns with their CSR entries per row (fem.diffusion.fem_time_system), Newton steps + the B = 0 start solve
    (uncalibrated)."""
    iterations = saturation_iterations(settings_values, context)
    if mesh.kind == 'triangles':
        if iterations is not None:
            iterations = (min(NEWTON_ITERATIONS_GUESS, iterations[1]) + 1, iterations[1] + 1)
        unknowns, entries = fem_time_system(mesh, element_order(settings_values))
        return diffusion_time_estimate(context, unknowns, iterations=iterations, every_step=False,
                                       held=_HELD_POTENTIAL, entries_per_row=entries)
    return diffusion_time_estimate(context, context.window_cells(), iterations=iterations, every_step=False,
                                   held=_HELD_POTENTIAL)


_FEM_NODAL_FIELDS = 5   # A and the B_x, B_y, H_x, H_y display averages (full-mesh float64)
_FEM_POINT_STATE = 9    # float64 per quadrature point of one evaluation: B (2), D (4), H (2), w


def _triangle_memory_estimate(settings_values, mesh, context):
    """MagneticFEMPhysics: per-element float32 fields, nodal values and the operator / factor of fem_diffusion_estimate
    (refactorised per Newton step when a material saturates), plus the host curl quadrature cache (operator Q, weights,
    points per rule point: 2 n + 3 float64) and the point state of one Newton evaluation."""
    order = element_order(settings_values)
    points = triangle_rule(PLANAR_DEGREES[order] + int(is_axisymmetric(settings_values))).weights.size
    local = 6 if order == 2 else 3
    elements = int(mesh.n_elements)
    saturation = context.material_sets(SATURATION_ARRAY) or _bh_curve_material_count(context.spec) > 0
    system = fem_diffusion_estimate(mesh, order, element_fields=_N_FIELDS, nodal_fields=_FEM_NODAL_FIELDS,
                                    direct=linear_solver_direct(context), multigrid=linear_solver_multigrid(context),
                                    cudss=cudss_back_end(context.solvers), refactorizes=bool(saturation))
    own = MemoryEstimate.from_items(host_items=(
        ('curl quadrature cache (float64)', (2 * local + 3) * points * FLOAT64_BYTES * elements),
        ('quadrature point state of one evaluation (float64)', _FEM_POINT_STATE * points * FLOAT64_BYTES * elements)))
    return combine((own, system))


def _scalar_outputs(settings_values, physics_spec):
    """energy, then <coil>_flux_linkage / <coil>_inductance per enabled coil feature."""
    names = ['energy']
    for feature in physics_spec.features:
        if feature.enabled and feature.type == 'coil':
            names.extend((f'{feature.name}_flux_linkage', f'{feature.name}_inductance'))
    return tuple(names)


# ---- type ----
_FEATURES = (
    FeatureType('magnetic_insulation', 'Magnetic insulation', 'boundary_condition', 'boundary', allow_interior=False,
                summary='Makes the magnetic field run along the selected domain edges, as on a far boundary or the '
                        'midplane between a current and its return. Every edge without another condition has it.',
                description=_INSULATION_DESCRIPTION, apply=_apply_insulation, grids=_BOTH_GRIDS),
    FeatureType('symmetry', 'Symmetry (field normal)', 'boundary_condition', 'boundary', allow_interior=False,
                summary='Makes the magnetic field cross the selected domain edges at right angles, as on the midplane '
                        'of two equal currents or next to ideal iron.',
                description=_SYMMETRY_DESCRIPTION, apply=_apply_symmetry, grids=_BOTH_GRIDS),
    FeatureType('external_field', 'External magnetic field', 'domain_condition', 'domain', max_entities=1,
                params=(Param('B_x', 'float', 0.0, label='Applied flux density x', unit='T',
                              help='x component of the uniform applied flux density; axisymmetric: must be 0 (B_r).',
                              tip='Strength of the applied field along x.\n\nMust be 0 in axisymmetric models.'),
                        Param('B_y', 'float', 1e-4, label='Applied flux density y', unit='T',
                              help='y component of the uniform applied flux density (axisymmetric: the axial B_z).',
                              tip='Strength of the applied field along y (along the axis in axisymmetric '
                                  'models).\n\nThe Earth field is about 0.00005.')),
                summary='Applies a uniform magnetic field, held at the domain edges, around the objects inside.',
                description=_EXTERNAL_FIELD_DESCRIPTION, apply=_apply_external_field,
                validate=_validate_external_field, grids=_BOTH_GRIDS),
    FeatureType('fixed_potential', 'Fixed vector potential', 'domain_condition', 'domain',
                params=(Param('potential', 'float', 0.0, label='Vector potential', unit='T m',
                              space_units=(('axisymmetric', 'T m^2'),),
                              help='Held value: planar A_z [T m]; axisymmetric the flux function psi = r A_phi '
                                   '[T m^2]. 0 makes the region exclude all flux.',
                              tip='Value the selected domains are held at; 0 keeps all magnetic flux out of them, '
                                  'like a superconductor.'),),
                summary='Holds the magnetic potential in the selected domains, so no flux enters them.',
                description=_FIXED_POTENTIAL_DESCRIPTION, apply=_apply_fixed_potential, grids=_BOTH_GRIDS),
    FeatureType('current_source', 'Current density', 'domain_condition', 'domain',
                params=(Param('density', 'float', 1e6, label='Current density', unit='A/m^2', field_dependent=True,
                              help='J_z (planar, > 0 along +z) or J_phi (axisymmetric, > 0 counter-clockwise seen '
                                   'from +z); may depend on x and y; must be finite on the selected cells.',
                              tip='Current per area flowing out of the drawing (in axisymmetric models: around the '
                                  'axis).'),),
                summary='Drives a given current per area through the selected domains.',
                description=_CURRENT_SOURCE_DESCRIPTION, apply=_apply_current_source, grids=_BOTH_GRIDS),
    FeatureType('coil', 'Coil', 'domain_condition', 'domain', result_kind='table',
                params=(Param('current', 'float', 1.0, label='Current', unit='A',
                              help='Current per turn, > 0 along +z (axisymmetric: counter-clockwise seen from +z).',
                              tip='Current in each turn of the coil; the sign sets the direction.'),
                        Param('turns', 'int', 1, label='Turns', unit='1', min=1,
                              help='Number of turns spread over the selected cross-section (stranded winding).',
                              tip='Number of wire turns in the selected cross-section.')),
                summary='A winding of many thin turns: drives the current through the selected domains and reports '
                        'its flux linkage and inductance.',
                description=_COIL_DESCRIPTION, apply=_apply_coil, result=_coil_result, memory_estimate=_coil_memory,
                grids=_BOTH_GRIDS),
    FeatureType('probe', 'Vector potential probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
                result_kind='timeseries',
                summary='Records the magnetic vector potential at a point after every solve.',
                description=_PROBE_DESCRIPTION, apply=_apply_probe, result=_probe_result,
                validate=validate_points_inside, grids=_BOTH_GRIDS),
)

_FIELD_INFOS = tuple(MAGNETIC_FIELD_INFOS.values())

MAGNETOSTATICS = PhysicsType(
    name=MAGNETOSTATICS_TYPE_NAME,
    label='Magnetostatics (2D)',
    default_instance_name='magnetic',
    family='',
    group='electrical',
    spaces_2d=('planar', 'axisymmetric'),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    description=_PHYSICS_DESCRIPTION,
    summary=_PHYSICS_SUMMARY,
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=dict(MAGNETIC_SCHEMA),
    optional_material_schema=dict(OPTIONAL_MAGNETIC_SCHEMA),
    state_fields=_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    source_slots={'current_density': _SLOT_DOCS},
    supported_studies=('stationary', 'time_dependent'),
    default_study={'type': 'stationary'},
    uses_dispersion=False,
    solver_roles={'linear': CG_DIFFUSION_LINEAR_ROLE},
    create=_create,
    validate=_validate_magnetostatics,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALARS_HELP,
    dimensions=(2,),
    backends=('gpu', 'cpu'),   # MagneticPhysics on the Domain's xp (solver_plan S6 stage 2); FEM: host + run backend
    grids=('tensor', 'triangles'),   # triangles: MagneticFEMPhysics (solver_fem.py, magnetic plan stage 5)
    mesh_method=_mesh_method,
)


def _selftest_cases():
    """The magnetostatics SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_backend import BACKEND_SELFTEST_CASES
    from .selftests_fem import FEM_SELFTEST_CASES
    return SELFTEST_CASES + BACKEND_SELFTEST_CASES + FEM_SELFTEST_CASES


def register(registry):
    """The magnetostatics type, the 'flux_concentrator_needle' and 'iron_c_core_saturation' templates, then the
    selftest loader."""
    registry.register_physics(MAGNETOSTATICS)
    registry.register_template(
        'flux_concentrator_needle', FLUX_CONCENTRATOR_LABEL, FLUX_CONCENTRATOR_DESCRIPTION,
        flux_concentrator_needle_spec,
        summary='A thin, highly permeable needle in a uniform magnetic field gathers the flux and concentrates it at '
                'its tip; the gain below the tip is swept over the distance from the tip. Axisymmetric, steady.')
    registry.register_template(
        'iron_c_core_saturation', IRON_C_CORE_LABEL, IRON_C_CORE_DESCRIPTION, iron_c_core_spec,
        summary='An iron C-core with an air gap, driven by a coil whose current is swept until the iron saturates; '
                'the flux density in the gap against the current shows the knee. Triangle mesh, steady.')
    registry.register_selftests(_selftest_cases)

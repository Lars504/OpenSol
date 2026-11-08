"""Model-layer definition of the 'magnetostatics_3d' physics type (stages 1-3 of magnetostatics_3d_plan: fields of
permeable and saturating materials, permanent magnets, ideal iron, a uniform applied field and stranded coils by the
reduced scalar potential; settings, features, material vocabulary, validation, memory estimate, scalars) plus its
template and selftests; registered through definition.py. On a 'tetrahedra' mesh the create builds MagneticFEMPhysics3D
(solver_fem3d.py). Imports no cupy at module level (the solver inside create)."""
from ...core.constants import MU0
from ...core.memory_estimate import (BOOL_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate, combine,
                                     diffusion_system_estimate, window_cell_count)
from ...core.time_estimate import HeldValues, diffusion_time_estimate
from ...fem.diffusion import cudss_back_end, cudss_diffusion_estimate
from ...fem.diffusion import lagrange_entries_per_dof
from ...fem.diffusion3d import fem_diffusion_estimate_3d
from ...model.definition_helpers import (CG_DIFFUSION_LINEAR_ROLE, DISCRETIZATION_PARAM, cg_solver_issues,
                                         element_order, evaluates_to, face_conflicts, held_object_conflicts,
                                         linear_solver_direct, linear_solver_mixed, linear_solver_multigrid,
                                         linear_solver_precision, point_cells, reports_model_checks,
                                         validate_points_inside)
from ...model.geometry import expand_entities
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path, resolve_values
from ...solvers.roles import resolved_precision
from .definition_magnetostatics import (_ANDERSON_STATE_ARRAYS, _BH_CURVE, _CURVE_PROPOSAL_ARRAYS, _MU_DC_TIP,
                                        _NONLINEAR_SETTINGS, _bh_curve_material_count, _bh_curve_rows, _coil_result,
                                        _material_property as _material_property_2d, _method_misfit, _set_faces,
                                        saturation_iterations)
from .fields import MAGNETIC_3D_CURRENT_NAMES, MAGNETIC_3D_FIELD_NAMES, MAGNETIC_3D_PROBE_COMPONENTS
from .materials import (BH_CURVE_PROPERTY, MAGNETIC_3D_PROPERTY_ARRAYS, MAGNETIC_3D_SCHEMA,
                        OPTIONAL_MAGNETIC_3D_SCHEMA, REMANENCE_3D_ARRAYS, SATURATION_ARRAY)
from .renderer import MAGNETIC_3D_FIELD_INFOS
from .saturation import TENSOR_NONLINEAR_METHODS, bh_curve
from .selftests_3d import NEEDLE_3D_DESCRIPTION, NEEDLE_3D_LABEL, needle_3d_spec

MAGNETOSTATICS_3D_TYPE_NAME = 'magnetostatics_3d'   # = selftests_3d.MAGNETOSTATICS_3D_TYPE_NAME (not imported)
_FACE_FEATURES = frozenset({'magnetic_insulation', 'symmetry'})
_GRIDS = ('tensor', 'tetrahedra')   # PhysicsType / FeatureType grids: MagneticPhysics3D, MagneticFEMPhysics3D


# ---- material vocabulary (the 2D Params' kind / unit / bounds / flags; expands_to and texts of 3D) ----
def _material_property(name, label, unit, help_text, tip, positive=False):
    """The 2D magnetic material Param with the magnetostatics_3d property -> array map."""
    return _material_property_2d(name, label, unit, help_text, tip, positive, arrays=MAGNETIC_3D_PROPERTY_ARRAYS)


_MU_DC_HELP = (
    'Isotropic static relative permeability (> 0): sets mu_dc_xx, mu_dc_yy and mu_dc_zz (component entries override '
    'it); unset = 1 (vacuum). NOT the optical mu_r of the wave physics. Typical: air, copper, aluminium 1; ferrite 1e3 '
    '- 1e4; electrical steel 4e3; Metglas 1e4 - 1e5; mu-metal 5e4. The finite volumes solve a large contrast exactly; '
    "the pcg multigrid (Automatic on the gpu) keeps the iteration count moderate (a few hundred at 1e5). With "
    'saturation_polarization it is the initial (small-field) permeability of the B-H curve.')
_MU_COMPONENT_TIP = ("Static relative permeability along {axis} of a material that magnetises differently along x, y "
                     "and z. Replaces 'Static relative permeability' for this direction.")
_REMANENCE_HELP = (
    'Remanent flux density B_r [T] along {axis} of a permanent magnet: B = mu0 mu_dc H + B_r, entered as face drives '
    'B_r / mu_dc at the magnet faces (exact for a uniformly magnetised region, also across its edges). Typical: NdFeB '
    '1.2 - 1.4 T with mu_dc 1.05, SmCo 1.0 T, ferrite 0.4 T with mu_dc 1.1. Unset = 0 (the remanence arrays are '
    'allocated only when a material sets one).')
_REMANENCE_TIP = ('Built-in magnetisation of a permanent magnet along {axis}, as a flux density.\n\n'
                  'Typical: neodymium magnets about 1.3.')
_SATURATION_HELP = (
    'Saturation polarisation J_s [T] (> 0) of an anhysteretic B-H curve: the Frohlich-Kennelly curve B = mu0 H + J_s a '
    'H / (1 + a H) with a = mu0 (mu_dc - 1) / J_s, i.e. initial permeability mu_dc, B -> mu0 H + J_s in strong fields, '
    'no hysteresis. Isotropic: mu_dc_xx, mu_dc_yy and mu_dc_zz must be equal in the material (validation error). Solved '
    'by a fixed point on ln of the secant permeability mu = B(|H|) / (mu0 |H|) at the cell-centre |H| of each solve, '
    'Anderson-accelerated (settings nonlinear_*); energy and coil inductances are then secant values, not the integral '
    'of H dB. Unset = linear (the array is allocated only when a material sets it); mu_dc <= 1 stays linear. Not '
    'combinable with a remanence or a bh_curve in the same material (validation error). Typical: iron 2.15, electrical '
    'steel 2.0, Metglas 0.6 - 1.6, Mn-Zn ferrite 0.4 - 0.5, NiFe (mu-metal) 0.8.')
_MATERIAL_PROPERTIES = (
    _material_property('mu_dc', 'Static relative permeability', '1', _MU_DC_HELP, _MU_DC_TIP, positive=True),
    *(_material_property(f'mu_dc_{axis}{axis}', f'Static relative permeability along {axis}', '1',
                         f'Static relative permeability along {axis}, > 0; overrides mu_dc for this axis.',
                         _MU_COMPONENT_TIP.format(axis=axis), positive=True) for axis in 'xyz'),
    *(_material_property(name, f'Remanence along {axis}', 'T', _REMANENCE_HELP.format(axis=axis),
                         _REMANENCE_TIP.format(axis=axis)) for name, axis in zip(REMANENCE_3D_ARRAYS, 'xyz')),
    _material_property(SATURATION_ARRAY, 'Saturation polarisation', 'T', _SATURATION_HELP,
                       'Flux density the magnetisation of the material approaches in strong fields; empty means a '
                       'linear material.\n\nTypical: iron 2.1, electrical steel 2.0, Metglas 0.6 to 1.6, ferrite 0.4.',
                       positive=True),
    _BH_CURVE,   # the 2D Param (rows [H, B]); isotropic, mu_dc ignored in its cells
)


# ---- feature texts ----
_INSULATION_DESCRIPTION = (
    'Magnetic insulation B.n = 0 on the selected outer faces (z_lo / z_hi included; the default of every face no '
    'feature sets): field lines run along the face (Neumann on phi_m). It is the mirror plane of a field parallel to '
    'it, and stands in for a distant boundary far from the sources. Does not anchor phi_m: with insulation on every '
    'face, no external field, symmetry face or ideal iron the type holds the window corner cell at phi_m = 0 (the '
    'problem is then unique up to that constant; the corner cell reads B = 0).')
_SYMMETRY_DESCRIPTION = (
    'Symmetry (field normal) on the selected outer faces: phi_m = 0 there (Dirichlet), so the tangential H is 0 and '
    'field lines cross the face at right angles; the mirror plane of a field normal to it (the midplane between a '
    'magnet and its mirror image), or a face next to ideal iron. The opposite face kinds of 2D magnetostatics (there '
    'A = 0 is insulation): this type solves for the scalar potential. Anchors the solve.')
_EXTERNAL_FIELD_DESCRIPTION = (
    "Uniform applied flux density (B_x, B_y, B_z) [T], selection 'all' only, at most one per physics: the outermost "
    'window cells (all six faces) are held at the applied potential phi_m = -B . r / mu0 evaluated at their faces '
    'toward the free cells (r from the window corner), so a uniform field is exact on any cell widths. Inside, objects '
    'perturb the field; keep them several object sizes from the window edges (the ring holds the undisturbed field: '
    'error ~ (object size / distance)^3 of a dipole). The ring cells leave the solve (B, H read 0 there) and face '
    'features have no effect under it. Anchors the solve. Typical: the Earth field ~5e-5 T, a Helmholtz coil 1e-4 - '
    '1e-2 T.')
_IDEAL_IRON_DESCRIPTION = (
    'Holds the magnetic scalar potential phi_m on the selected cells for the whole run: ideal iron (mu -> infinity), '
    'H = 0 inside and B normal to its surface; the cells leave the solve (B and H read 0 in them). Two regions at '
    'different potentials drive the flux between them like the poles of a yoke (the magnetomotive force is their '
    'difference in A). The opposite of the 2D fixed vector potential (a flux excluder). Anchors the solve.')
_COIL_DESCRIPTION = (
    "Stranded coil (many thin insulated turns) wound around the axis 'x', 'y' or 'z' (setting axis) on the selected "
    'winding objects, e.g. a circle prism along that axis minus a smaller one (difference): every window slice normal '
    'to the axis that holds winding cells must close around a hole inside the window, else the build fails naming the '
    'coil. current [A] > 0 circulates counter-clockwise seen from +axis (B along +axis inside). The drive is the coil '
    'field T along the axis (T-Omega, H = T - grad phi_m): T = turns current / h in the enclosed hole, ramping to 0 '
    'across the winding by the distances to the hole and to the outside (uniform J in a circular ring), 0 outside; h = '
    'the winding\'s axial length. J = curl T is shown in fields J_x / J_y / J_z (its total through a half plane from '
    'the axis is turns current). Result \'table\': one row [feature, current, turns, flux_linkage, inductance] of the '
    'latest solve and the same extras: flux_linkage [Wb] = the integral of T . B / current over the coil\'s support '
    '(the flux through each turn, summed), inductance = flux_linkage / current [H] (nan at zero current); with several '
    "coils a coil's value includes the flux of the others (mutual terms). Exact energy identity W = flux_linkage "
    'current / 2 for one coil with no held cells (no ring, ideal iron or symmetry face); held cells add their '
    'boundary term. Scalars <feature>_flux_linkage and <feature>_inductance carry the same numbers. With a core of '
    "mu_dc above ~1e3 in the hole, use pcg precision 'double' or 'mixed' (T - grad phi_m cancels to a few digits "
    "there; validation warns for 'single'). On a tetrahedron mesh T is held as edge integrals (lowest-order edge "
    'elements; the hole / winding / outside ramp from a fine lattice over the winding), so the current through the '
    'coil is exactly turns current and J is constant per element; the winding must be one piece along the axis and '
    'its axial length h is the extent of its elements.')
_PROBE_DESCRIPTION = ('B_abs [T] or phi_m [A] (setting component) of the cell containing the point (its z, '
                      'mid-height when unset; a tetrahedron mesh: evaluated at the point), one sample per re-solve plus '
                      'one at the end of the run (stationary: all at t = 0).')
_SCALARS_HELP = (
    'energy [J]: magnetic field energy of the free cells, the sum over faces of B_n^2 / (2 mu0 mu_face) times the face '
    'volume (held cells and the external-field ring excluded, held-free faces count their free half cell; with '
    "remanence the energy of B alone, not the magnet's co-energy; with saturating materials the secant value at the "
    'converged mu, not the integral of H dB); <coil>_flux_linkage [Wb] and <coil>_inductance [H] of every enabled '
    'coil feature, as its table result (secant values when saturating).')
_PHYSICS_DESCRIPTION = (
    '3D magnetostatics by the reduced magnetic scalar potential (finite volumes, 7-point stencil on the physical window '
    'of a 3D model): H = T - grad phi_m with T the coil fields, B = mu0 mu_dc H + B_r, div B = 0, solved for u = mu0 '
    'phi_m [T m] with the coefficients c = mu_dc per axis (mu_dc_xx, mu_dc_yy, mu_dc_zz; no cross terms) and the face '
    'drive w = mu0 T + B_r / mu_dc, so the face flux is B itself: B is taken from the face fluxes averaged to the cell '
    'centres (accurate in high-mu cells and iron cores too), H = (B - B_r) / (mu0 mu_dc) per axis, w_mag = sum of B^2 '
    '/ (2 mu0 mu_dc) per axis (the density of the energy scalar, as in 2D; not B . H / 2 inside magnets), '
    'phi_m [A], J = curl T [A/m^2]. Every outer face is magnetic insulation (B.n = 0) unless a symmetry feature sets '
    'phi_m = 0 there; sources are coils, an external_field (held outer ring), ideal_iron regions at a potential and '
    'material remanence; with none of the anchors (symmetry face, ring, ideal iron) the type holds the window corner '
    'cell at 0. Saturating materials (saturation_polarization or a bh_curve, isotropic) run a fixed point on ln of '
    'their secant permeability, each iteration one assembly and one warm-started solve (settings nonlinear_*; '
    "'newton' is not available on this grid); field mu_eff shows the assembled permeability. Scalars: energy and per "
    'coil flux linkage and inductance. The Lorentz force is not available yet (magnetostatics_3d_plan stage 4); eddy '
    'currents in 3D are out of scope. Linear solver: pcg (Automatic: multigrid on both backends). A '
    'time-dependent study re-solves only when mu_dc, a saturation or a remanence changed. Fields are edge-extended into '
    'an EM PML shell. On a tetrahedron mesh (MeshSpec kind tetrahedra; 3D Cartesian) the same model by finite elements '
    '(setting discretization: quadratic or linear Lagrange phi_m on affine tetrahedra), solved by the sparse direct '
    'solver (Automatic: cuDSS on the gpu backend with nvmath-python, else host SuperLU; one factorisation per '
    'saturation iteration, cuDSS refreshing its plan): the external field holds phi_m on '
    'the window faces (no held ring: every element stays free and wins over face features there), the corner anchor is '
    'the window node nearest the lower corner, B, H, w_mag, J and mu_eff are element means (constant per element on '
    'the display), phi_m is the element field, energy and flux linkage exact integrals of the element fields; field '
    'derived values are not available on tetrahedra yet (use probes and the coil table). The instance is usually named '
    'magnetic.')
_PHYSICS_SUMMARY = ('Static magnetic fields of coils, permanent magnets, magnetic (also saturating) materials and an '
                    'applied field in 3D, by the magnetic scalar potential. Gives the flux density, the field energy '
                    'and coil inductances.')


# ---- feature apply / result callbacks (worker thread, GPU) ----
def _apply_insulation(ctx, solver, feature_spec, values, selection):
    _set_faces(solver, selection.outer_faces, 'neumann')
    return None


def _apply_symmetry(ctx, solver, feature_spec, values, selection):
    _set_faces(solver, selection.outer_faces, 'dirichlet')
    return None


def _apply_external_field(ctx, solver, feature_spec, values, selection):
    solver.set_external_field(values['B_x'], values['B_y'], values['B_z'])
    return None


def _apply_ideal_iron(ctx, solver, feature_spec, values, selection):
    solver.add_ideal_iron(ctx.domain_mask(selection.entities), values['potential'])
    return None


def _apply_coil(ctx, solver, feature_spec, values, selection):
    return solver.add_coil(feature_spec.name, ctx.domain_mask(selection.entities), values['axis'], values['turns'],
                           values['current'])


def _window_cells_3d(mesh):
    """Physical-window voxels of a 3D MeshEstimate (PML layers excluded along x, y and z)."""
    sides = mesh.npml_sides
    return window_cell_count(mesh) * (int(mesh.nz) - sides.get('z_lo', 0) - sides.get('z_hi', 0))


_LATTICE_POINT_BYTES = 80   # coil lattice per point: points, element, reference, flags, ramp (uncalibrated)
_LOCATE_POINT_BYTES = 400   # PointLocator3D candidates and distances per located point (24 nearest)


def _coil_memory(values, n_selected_cells, mesh):
    """The coil's float64 window T per ampere (device); build extra: the host winding mask, ramp and T copies.
    Tetrahedra: the host float64 edge dofs of T per ampere; build extra the winding lattice at its cap."""
    if mesh.kind == 'tetrahedra':
        from .coils3d import LATTICE_MAX_POINTS, LOCATE_CHUNK   # lazy: coils3d imports scipy
        return MemoryEstimate.from_items(
            build_extra=LATTICE_MAX_POINTS * _LATTICE_POINT_BYTES + LOCATE_CHUNK * _LOCATE_POINT_BYTES,
            host_items=(('coil T per ampere (float64 edge integrals, host)', FLOAT64_BYTES * int(mesh.n_edges)),))
    window_cells = _window_cells_3d(mesh)
    return MemoryEstimate.from_items(
        (('coil T per ampere (float64 window)', FLOAT64_BYTES * window_cells),),
        build_extra=BOOL_BYTES * window_cells,
        host_items=(('coil winding mask, ramp and T (host)', (BOOL_BYTES + 2 * FLOAT64_BYTES) * window_cells),))


def _apply_probe(ctx, solver, feature_spec, values, selection):
    if ctx.domain.grid_kind == 'tetrahedra':   # evaluated at the point
        if len(selection.objects) != 1:
            raise ValueError(f"magnetic probe '{feature_spec.name}' needs exactly one point object, got "
                             f'{len(selection.objects)}')
        return solver.add_probe_at(ctx.point_position(selection.objects[0]), values['component'])
    cells = point_cells(ctx, feature_spec, selection)
    if len(cells) != 1:
        raise ValueError(f"magnetic probe '{feature_spec.name}' needs exactly one point object, got {len(cells)}")
    return solver.add_probe(*cells[0], values['component'])


def _probe_result(session, solver, probe, feature_spec, options):
    from ...core.monitors import probe_result   # lazy: core.monitors imports cupy
    return probe_result(probe, unit=MAGNETIC_3D_FIELD_INFOS[probe.field_component].unit)


# ---- validation (CPU only) ----
def _validate_external_field(vctx, physics_spec, feature_spec, values):
    """Selection 'all' only."""
    try:
        entities = expand_entities(vctx.spec, feature_spec.selection)
    except (ValueError, KeyError, TypeError):   # bad selection: reported by the selection checks
        return []
    if entities and entities != ['all']:
        return [Issue('error', feature_path(physics_spec, feature_spec),
                      "external_field holds the outer ring of the whole window: select 'all' only")]
    return []


_CANCELLATION_PERMEABILITY = 1e3   # decision 7: above it T - grad phi_m in a coil's core loses float32 digits


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


def _table_peak_permeability(raw_rows, parameters):
    """Largest secant B / (mu0 H) over the bh_curve rows, 0 when they do not resolve (reported at the material)."""
    rows = _bh_curve_rows(raw_rows, parameters)
    return max((flux / (MU0 * field) for field, flux in rows or () if field > 0.0), default=0.0)


def _high_permeability_materials(vctx):
    """Names of the selected materials with mu_dc or a component (or not known at validation), or a bh_curve secant,
    above _CANCELLATION_PERMEABILITY."""
    names = []
    for material, properties in _selected_material_properties(vctx.spec):
        raw_rows = properties.get(BH_CURVE_PROPERTY)
        if any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value > _CANCELLATION_PERMEABILITY)
               for name in ('mu_dc', 'mu_dc_xx', 'mu_dc_yy', 'mu_dc_zz')) or (
                raw_rows is not None
                and _table_peak_permeability(raw_rows, vctx.parameters) > _CANCELLATION_PERMEABILITY):
            names.append(material.name)
    return names


def _saturation_issues(vctx, physics_spec, settings_values):
    """nonlinear_method 'newton' (error); once per model, per selected material: a saturation (saturation_polarization
    or bh_curve) with a remanence, both saturations, J_s with mu_dc differing between axes, a bh_curve the curve rules
    reject (errors), mu_dc beside a bh_curve (warning: ignored)."""
    issues = []
    misfit = _method_misfit(settings_values, triangles=False)
    if misfit is not None:
        issues.append(Issue('error', ('physics', physics_spec.name),
                            f"nonlinear_method '{misfit}' is not available in {physics_spec.type} (tensor and "
                            f"tetrahedron meshes): use 'auto' or one of {', '.join(TENSOR_NONLINEAR_METHODS)}"))
    if not reports_model_checks(vctx, physics_spec, MAGNETOSTATICS_3D_TYPE_NAME):
        return issues
    parameters = vctx.parameters
    for material, properties in _selected_material_properties(vctx.spec):
        path = ('material', material.name)
        saturates = evaluates_to(properties.get(SATURATION_ARRAY), parameters, lambda value: value > 0.0)
        raw_rows = properties.get(BH_CURVE_PROPERTY)
        tabulated = raw_rows is not None
        if (saturates or tabulated) and any(
                evaluates_to(properties.get(name), parameters, lambda value: value != 0.0)
                for name in REMANENCE_3D_ARRAYS):
            issues.append(Issue('error', path, f"material '{material.name}' sets "
                                               f"{BH_CURVE_PROPERTY if tabulated else SATURATION_ARRAY} and a "
                                               'remanence: a saturating permanent magnet is not modelled (remove one)'))
        if saturates and not tabulated:
            components = [properties.get(name, properties.get('mu_dc')) for name in ('mu_dc_xx', 'mu_dc_yy', 'mu_dc_zz')]
            if components[1] != components[0] or components[2] != components[0]:
                issues.append(Issue('error', path, f"material '{material.name}' sets {SATURATION_ARRAY} with "
                                                   'different mu_dc_xx / mu_dc_yy / mu_dc_zz: the saturating curve is '
                                                   'isotropic; give one mu_dc'))
        if not tabulated:
            continue
        if saturates:
            issues.append(Issue('error', path, f"material '{material.name}' sets both {SATURATION_ARRAY} and "
                                               f'{BH_CURVE_PROPERTY}: give one B-H curve (remove one)'))
        permeability_names = [name for name in ('mu_dc', 'mu_dc_xx', 'mu_dc_yy', 'mu_dc_zz')
                              if properties.get(name) is not None]
        if permeability_names:
            issues.append(Issue('warning', path, f"material '{material.name}': {', '.join(permeability_names)} "
                                                 f'{"is" if len(permeability_names) == 1 else "are"} ignored in its '
                                                 f'cells, where {BH_CURVE_PROPERTY} sets the permeability'))
        rows = _bh_curve_rows(raw_rows, parameters)
        if rows is not None:
            try:
                bh_curve(rows)
            except ValueError as exc:
                issues.append(Issue('error', path, f"property '{BH_CURVE_PROPERTY}': {exc}"))
    return issues


def _single_precision_coil_issues(vctx, physics_spec, path):
    """Decision 7 warning: a coil with a pcg 'single' linear solver and a material of mu_dc > 1e3 (a core in a coil's
    hole cancels T - grad phi_m to a few digits)."""
    if not any(feature.enabled and feature.type == 'coil' for feature in physics_spec.features):
        return []
    resolved = (vctx.solvers(physics_spec) or {}).get('linear')
    if resolved is None or resolved_precision(resolved.solver_type, resolved.values) != 'single':
        return []
    names = _high_permeability_materials(vctx)
    if not names:
        return []
    return [Issue('warning', path, f"coil with material(s) {', '.join(names)} of mu_dc > "
                                   f"{_CANCELLATION_PERMEABILITY:g} on a 'single' precision linear solver: inside a "
                                   "core the coil field and the potential gradient cancel to a few digits; use "
                                   "precision 'double' or 'mixed'")]


def _validate_magnetostatics_3d(vctx, physics_spec, settings_values):
    """At most one external field; CG limit / direct size, face and held-object conflicts, the coil precision
    warning, the saturation rules."""
    path = ('physics', physics_spec.name)
    issues = []
    n_external = sum(1 for feature in physics_spec.features if feature.enabled and feature.type == 'external_field')
    if n_external > 1:
        issues.append(Issue('error', path, f'{n_external} external_field features: at most one per physics (each '
                                           'holds the same outer ring)'))
    issues.extend(cg_solver_issues(vctx, physics_spec, path))
    issues.extend(face_conflicts(vctx, physics_spec, _FACE_FEATURES, path))
    issues.extend(held_object_conflicts(vctx, physics_spec, frozenset({'ideal_iron'}), frozenset({'coil'}), path))
    issues.extend(_single_precision_coil_issues(vctx, physics_spec, path))
    issues.extend(_saturation_issues(vctx, physics_spec, settings_values))
    return issues


# ---- physics factory and memory estimate ----
def _create(ctx, physics_spec, values):
    """MagneticPhysics3D (tensor) or MagneticFEMPhysics3D (tetrahedra, the discretization order) on ctx.domain with
    the build's 'linear' solver (every face insulated until features apply) plus one add_bh_curve per material region
    with a bh_curve."""
    nonlinear = {name: values[name] for name in ('nonlinear_tolerance', 'nonlinear_max_iterations',
                                                 'nonlinear_relaxation', 'nonlinear_method')}
    linear_solver = ctx.solvers(physics_spec.name)['linear']
    tetrahedra = ctx.domain.grid_kind == 'tetrahedra'
    if tetrahedra:
        from .solver_fem3d import MagneticFEMPhysics3D   # lazy: the solver imports cupy
        solver = MagneticFEMPhysics3D(ctx.domain, order=element_order(values), linear_solver=linear_solver,
                                      **nonlinear)
    else:
        from .solver3d import MagneticPhysics3D   # lazy: the solver imports cupy
        solver = MagneticPhysics3D(ctx.domain, linear_solver=linear_solver, **nonlinear)
    for region in ctx.material_regions:
        raw_rows = region.properties.get(BH_CURVE_PROPERTY)
        if raw_rows is not None:
            rows = resolve_values((_BH_CURVE,), {BH_CURVE_PROPERTY: raw_rows}, ctx.parameters)[BH_CURVE_PROPERTY]
            # tetrahedra: the region's element mask; tensor: a fresh full-grid mask, cropped by the solver
            solver.add_bh_curve(region.mask if tetrahedra else ctx.region_mask(region), rows)
    return solver


_POST_PROCESSING_ARRAYS = 12   # solve-dtype window temporaries of one field update (fluxes, drive means, B, H, sums)
_COIL_UPDATE_ARRAYS = 5   # float64 window temporaries of the J update (T, its gradient, the face gradient, 2 sums)
_SCALAR_ARRAYS = 5        # window temporaries of the energy / linkage (face flows (3), conductance, mask), float64 bound
_SATURATION_ARRAYS = 7    # float64 window: loop state, J_s, proposal, update, |B| and mu_r of one proposal, mu_dc copy
_SATURATION_VALUE_ARRAYS = 5   # solve-dtype window: exp(state), the three assembled-c inputs, the kept |B|
_SATURATION_MASKS = 3     # bool window: the saturating cells, the J_s > 0 cells, the table cells


# where the scalar potential is held, for the Jacobi-PCG condition estimate: symmetry faces (u = 0), ideal iron, the
# external-field ring on the window edge; insulation faces are natural, so with none of these the corner anchor
_HELD_POTENTIAL = HeldValues(feature_types=frozenset({'symmetry', 'ideal_iron'}),
                             edge_types=frozenset({'external_field'}))


def _time_estimate(settings_values, mesh, context):
    """The 3D DiffusionSystem recipe on the window voxels, times the saturation loop; tetrahedra: the recipe at the
    Lagrange unknowns with their CSR entries per row (fem.diffusion3d, uncalibrated)."""
    iterations = saturation_iterations(settings_values, context)
    if mesh.kind == 'tetrahedra':
        order = element_order(settings_values)
        unknowns = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
        return diffusion_time_estimate(context, unknowns, 3, iterations=iterations, every_step=False,
                                       held=_HELD_POTENTIAL, entries_per_row=lagrange_entries_per_dof(mesh, order))
    return diffusion_time_estimate(context, _window_cells_3d(mesh), 3, iterations=iterations, every_step=False,
                                   held=_HELD_POTENTIAL)


# MagneticFEMPhysics3D host arrays (uncalibrated): per element float64 results (8 fields + B_rms + J (3)), mu_dc /
# assembled mu / B_r (9), the Nedelec-1 element dofs (6 int64); per element of one post-processing block the (m, q, .)
# rule temporaries (P2 gradients (q, 10, 3), T, B, H at 4 points)
_FEM_ELEMENT_FLOATS = 12 + 9 + 6
_FEM_BLOCK_BYTES = 2048


def _tetrahedra_memory_estimate(settings_values, mesh, context):
    """MagneticFEMPhysics3D: fem_diffusion_estimate_3d (float32 element state, nodal values, operator / factor) plus
    the host element arrays, one post-processing block and the saturation loop over at most every element."""
    from .solver_fem3d import COIL_FIELD_CACHE_BYTES, POST_CHUNK   # lazy: the solver imports cupy
    elements = int(mesh.n_elements)
    features = [feature for feature in context.physics_spec.features if feature.enabled]
    coils = any(feature.type == 'coil' for feature in features)
    saturation = context.material_sets(SATURATION_ARRAY) or _bh_curve_material_count(context.spec) > 0
    n_fields = len(MAGNETIC_3D_FIELD_NAMES) + len(MAGNETIC_3D_CURRENT_NAMES) * int(coils) + int(saturation)
    system = fem_diffusion_estimate_3d(mesh, element_order(settings_values), element_fields=n_fields,
                                       direct=linear_solver_direct(context), multigrid=linear_solver_multigrid(context),
                                       cudss=cudss_back_end(context.solvers), refactorizes=bool(saturation))
    loop_arrays = int(saturation) * (_SATURATION_ARRAYS + (
        _ANDERSON_STATE_ARRAYS if settings_values.get('nonlinear_method') in ('auto', 'anderson') else 0))
    own = MemoryEstimate.from_items(host_items=(
        ('element results, materials and edge dofs (host)', _FEM_ELEMENT_FLOATS * FLOAT64_BYTES * elements),
        ('post-processing block (rule-point temporaries)', _FEM_BLOCK_BYTES * min(elements, POST_CHUNK)),
        ('saturation state, proposal and Anderson history (host, per element)',
         loop_arrays * FLOAT64_BYTES * elements),
        # total T at 4 rule points x 3 components per element, kept across the saturation iterations up to the cap
        ('coil T rule values (saturation cache, host)',
         int(coils and saturation) * min(4 * 3 * FLOAT64_BYTES * elements, COIL_FIELD_CACHE_BYTES))))
    return combine((own, system))


def _memory_estimate(settings_values, mesh, context):
    """MagneticFields3D (float32 full grid; J with a coil, mu_eff when saturating) + the zero source window + the 3D
    window DiffusionSystem + one bool window mask per bh_curve material; build extra: the external ring values / mask,
    the mu_dc check masks and the J update; run items: the face drives and rhs (coils or remanence), the saturation
    loop (state, proposal, Anderson history; one state component); transient: the field-update and scalar temporaries;
    a cuDSS 'linear' factor (fem.diffusion.cudss_diffusion_estimate). Tetrahedra: _tetrahedra_memory_estimate."""
    if mesh.kind == 'tetrahedra':
        return _tetrahedra_memory_estimate(settings_values, mesh, context)
    window_cells = _window_cells_3d(mesh)
    precision = linear_solver_precision(context)
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    features = [feature for feature in context.physics_spec.features if feature.enabled]
    coils = any(feature.type == 'coil' for feature in features)
    driven = coils or any(context.material_sets(name) for name in REMANENCE_3D_ARRAYS)
    external = any(feature.type == 'external_field' for feature in features)
    n_curves = _bh_curve_material_count(context.spec)
    saturation = context.material_sets(SATURATION_ARRAY) or n_curves > 0
    saturation_bytes = int(saturation) * window_cells * (
        FLOAT64_BYTES * (_SATURATION_ARRAYS + (_CURVE_PROPOSAL_ARRAYS if n_curves else 0)
                         + (_ANDERSON_STATE_ARRAYS if settings_values.get('nonlinear_method') in ('auto', 'anderson')
                            else 0))
        + value_bytes * _SATURATION_VALUE_ARRAYS + BOOL_BYTES * _SATURATION_MASKS)
    n_fields = len(MAGNETIC_3D_FIELD_NAMES) + len(MAGNETIC_3D_CURRENT_NAMES) * int(coils) + int(saturation)
    own = MemoryEstimate.from_items(
        ((f'fields ({n_fields} x float32)', n_fields * FLOAT32_BYTES * int(mesh.n_cells)),
         ('zero source (float32 window)', FLOAT32_BYTES * window_cells),
         ('B-H curve masks (bool window)', BOOL_BYTES * window_cells * n_curves)),
        build_extra=((value_bytes + BOOL_BYTES) * window_cells * int(external) + 2 * BOOL_BYTES * window_cells
                     + _COIL_UPDATE_ARRAYS * FLOAT64_BYTES * window_cells * int(coils)
                     + BOOL_BYTES * int(mesh.n_cells) * int(n_curves > 0)),
        run_items=(('coil and remanence face drives and rhs (window)', 4 * value_bytes * window_cells * int(driven)),
                   ('saturation state, J_s, proposal and Anderson history (window)', saturation_bytes)),
        transient_items=(('field update', 'field update temporaries (window)',
                          _POST_PROCESSING_ARRAYS * value_bytes * window_cells),
                         ('energy and flux linkage', 'energy and flux linkage temporaries (window)',
                          _SCALAR_ARRAYS * FLOAT64_BYTES * window_cells)))
    parts = [own, diffusion_system_estimate(window_cells, precision, dimensions=3, mixed=linear_solver_mixed(context),
                                            multigrid=linear_solver_multigrid(context),
                                            direct=linear_solver_direct(context))]
    cudss = cudss_back_end(context.solvers)
    if cudss:   # DiffusionDirectSolver on the kept A_solve, refactorised per saturation iteration
        parts.append(cudss_diffusion_estimate(window_cells, 7 * window_cells, 3, cudss, input_copy=False,
                                              refactorizes=bool(saturation)))
    return combine(parts)


# ---- type ----
_FEATURES = (
    FeatureType('magnetic_insulation', 'Magnetic insulation', 'boundary_condition', 'boundary', allow_interior=False,
                summary='Makes the magnetic field run along the selected domain faces, as on a far boundary. Every '
                        'face without another condition has it.',
                description=_INSULATION_DESCRIPTION, apply=_apply_insulation, grids=_GRIDS),
    FeatureType('symmetry', 'Symmetry (field normal)', 'boundary_condition', 'boundary', allow_interior=False,
                summary='Makes the magnetic field cross the selected domain faces at right angles, as on the midplane '
                        'between a magnet and its mirror image or next to ideal iron.',
                description=_SYMMETRY_DESCRIPTION, apply=_apply_symmetry, grids=_GRIDS),
    FeatureType('external_field', 'External magnetic field', 'domain_condition', 'domain', max_entities=1,
                params=(Param('B_x', 'float', 0.0, label='Applied flux density x', unit='T',
                              help='x component of the uniform applied flux density.',
                              tip='Strength of the applied field along x.'),
                        Param('B_y', 'float', 0.0, label='Applied flux density y', unit='T',
                              help='y component of the uniform applied flux density.',
                              tip='Strength of the applied field along y.'),
                        Param('B_z', 'float', 1e-4, label='Applied flux density z', unit='T',
                              help='z component of the uniform applied flux density.',
                              tip='Strength of the applied field along z.\n\nThe Earth field is about 0.00005.')),
                summary='Applies a uniform magnetic field, held at the domain faces, around the objects inside.',
                description=_EXTERNAL_FIELD_DESCRIPTION, apply=_apply_external_field,
                validate=_validate_external_field, grids=_GRIDS),
    FeatureType('ideal_iron', 'Ideal iron (fixed magnetic potential)', 'domain_condition', 'domain',
                params=(Param('potential', 'float', 0.0, label='Magnetic scalar potential', unit='A',
                              help='Held phi_m [A] (magnetomotive force against phi_m = 0); 0 for a single iron '
                                   'region.',
                              tip='Magnetic potential the selected domains are held at; the difference between two '
                                  'such regions drives the flux between them.'),),
                summary='Treats the selected domains as ideal iron: no field inside, flux enters and leaves at right '
                        'angles.',
                description=_IDEAL_IRON_DESCRIPTION, apply=_apply_ideal_iron, grids=_GRIDS),
    FeatureType('coil', 'Coil', 'domain_condition', 'domain', result_kind='table',
                params=(Param('axis', 'choice', 'z', label='Coil axis', choices=('x', 'y', 'z'),
                              help='Axis the winding circles; every slice normal to it through the winding must close '
                                   'around a hole inside the window.',
                              tip='Direction of the coil axis: the winding goes around it.'),
                        Param('current', 'float', 1.0, label='Current', unit='A',
                              help='Current per turn, > 0 counter-clockwise seen from +axis (B along +axis inside).',
                              tip='Current in each turn of the coil; the sign sets the direction.'),
                        Param('turns', 'int', 1, label='Turns', unit='1', min=1,
                              help='Number of turns spread over the winding (stranded).',
                              tip='Number of wire turns in the winding.')),
                summary='A winding of many thin turns around an axis: drives the current through the selected ring '
                        'and reports its flux linkage and inductance.',
                description=_COIL_DESCRIPTION, apply=_apply_coil, result=_coil_result, memory_estimate=_coil_memory,
                grids=_GRIDS),
    FeatureType('probe', 'Magnetic probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
                params=(Param('component', 'choice', 'B_abs', label='Quantity', choices=MAGNETIC_3D_PROBE_COMPONENTS,
                              choice_labels=tuple((name, MAGNETIC_3D_FIELD_INFOS[name].label)
                                                  for name in MAGNETIC_3D_PROBE_COMPONENTS),
                              help='Field sampled at the point: B_abs [T] or phi_m [A].',
                              tip='Quantity recorded at the point.'),),
                result_kind='timeseries',
                summary='Records the magnetic flux density magnitude or the scalar potential at a point after every '
                        'solve.',
                description=_PROBE_DESCRIPTION, apply=_apply_probe, result=_probe_result,
                validate=validate_points_inside, grids=_GRIDS),
)

_FIELD_INFOS = tuple(MAGNETIC_3D_FIELD_INFOS.values())


def _scalar_outputs(settings_values, physics_spec):
    """energy, then <coil>_flux_linkage / <coil>_inductance per enabled coil feature."""
    names = ['energy']
    for feature in physics_spec.features:
        if feature.enabled and feature.type == 'coil':
            names.extend((f'{feature.name}_flux_linkage', f'{feature.name}_inductance'))
    return tuple(names)


MAGNETOSTATICS_3D = PhysicsType(
    name=MAGNETOSTATICS_3D_TYPE_NAME,
    label='Magnetostatics (3D)',
    default_instance_name='magnetic',
    family='',
    group='electrical',
    dimensions=(3,),
    description=_PHYSICS_DESCRIPTION,
    summary=_PHYSICS_SUMMARY,
    settings=(*_NONLINEAR_SETTINGS, DISCRETIZATION_PARAM),   # the 2D Param objects (decision 10); 'newton' refused
    features=_FEATURES,
    grids=_GRIDS,   # tetrahedra: MagneticFEMPhysics3D (solver_fem3d.py)
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=dict(MAGNETIC_3D_SCHEMA),
    optional_material_schema=dict(OPTIONAL_MAGNETIC_3D_SCHEMA),
    state_fields=_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    supported_studies=('stationary', 'time_dependent'),
    default_study={'type': 'stationary'},
    uses_dispersion=False,
    solver_roles={'linear': CG_DIFFUSION_LINEAR_ROLE},
    create=_create,
    validate=_validate_magnetostatics_3d,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALARS_HELP,
    backends=('gpu', 'cpu'),   # MagneticPhysics3D on the Domain's xp; DiffusionSystem 3D runs on both (FEM: host)
)


def _selftest_cases():
    """The 3D (tensor and tetrahedra) magnetostatics SelfTestCases (imported on the first registry.selftests())."""
    from .selftests_3d import SELFTEST_CASES_3D
    from .selftests_fem3d import FEM_3D_SELFTEST_CASES
    return (*SELFTEST_CASES_3D, *FEM_3D_SELFTEST_CASES)


def register(registry):
    """The magnetostatics_3d type, the 'needle_3d' template, then the 3D (tensor and tetrahedra) selftest loader."""
    registry.register_physics(MAGNETOSTATICS_3D)
    registry.register_template(
        'needle_3d', NEEDLE_3D_LABEL, NEEDLE_3D_DESCRIPTION, needle_3d_spec,
        summary='A thin, highly permeable needle in a uniform magnetic field concentrates the flux at its tip, in 3D; '
                'gain below the tip against the prolate-spheroid estimate. Steady.')
    registry.register_selftests(_selftest_cases)

"""Model-layer definition of the 'magnetic_fields_ac' physics type (time-harmonic eddy currents at one frequency on
tensor or triangle meshes: settings, coils with solid conductors, material vocabulary, validation, estimates, scalars)
plus its template
and selftests; shares the face / held-region features, mu_dc Params and validate_magnetic_common with
definition_magnetostatics. Imports no cupy at module level."""
from ...core import time_estimate as te
from ...core.memory_estimate import (COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES, FLOAT64_BYTES,
                                     MemoryEstimate, combine, diffusion_system_estimate, host_lu_bytes,
                                     window_cell_count)
from ...fem.diffusion import FEM_FILL_FACTOR, cudss_back_end, lagrange_entries_per_dof
from ...fem.magnetostatics import PLANAR_DEGREES
from ...fem.quadrature import triangle_rule
from ...model.definition_helpers import (AXISYMMETRIC_SPACE_SETTINGS, DISCRETIZATION_PARAM, DISCRETIZATION_SETTING,
                                         MAGNETIC_AC_LINEAR_ROLE, conducts, coupling_target, element_order,
                                         evaluates_to, is_axisymmetric, linear_solver_cudss, make_point_probe,
                                         material_follows_state, optional_material_param, reports_model_checks,
                                         validate_points_inside)
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path
from ...solvers.sparse_direct import RECYCLE_MAX_ITERATIONS, scipy_available, size_issue
from .circuit import COIL_RESULT_KEYS, CONDUCTORS, DRIVES, triangle_size_issue
from .definition_magnetostatics import (_BOTH_GRIDS, _FEATURES as _DC_FEATURES, _GEOMETRY_PARAM,
                                        _MATERIAL_PROPERTIES as _DC_MATERIAL_PROPERTIES, _apply_current_source,
                                        validate_magnetic_common)
from .materials import (BH_CURVE_PROPERTY, MAGNETIC_AC_SCHEMA, MAGNETIC_PROPERTY_ARRAYS, OPTIONAL_MAGNETIC_AC_SCHEMA,
                        REMANENCE_ARRAYS, SATURATION_ARRAY)
from .renderer import MAGNETIC_AC_FIELD_INFOS
from .selftests_ac import (BUSBAR_DESCRIPTION, BUSBAR_LABEL, NEEDLE_AC_DESCRIPTION, NEEDLE_AC_LABEL,
                           busbar_proximity_spec, flux_concentrator_needle_ac_spec)

MAGNETIC_AC_TYPE_NAME = 'magnetic_fields_ac'   # = selftests_ac.MAGNETIC_AC_TYPE_NAME (selftests never import this)
_SHARED_FEATURE_NAMES = ('magnetic_insulation', 'symmetry', 'external_field', 'fixed_potential')


# ---- settings ----
_FREQUENCY_PARAM = Param(
    'frequency', 'float', 50.0, label='Frequency', unit='Hz', min=0.0, min_exclusive=True,
    help='Frequency f of the time-harmonic solve (omega = 2 pi f, phasors exp(+j omega t), peak amplitudes). One '
         'frequency per build: sweep it with a parameter (each value refactorises). Skin depth delta = sqrt(2 / '
         '(omega mu0 mu_dc sigma_dc)): copper 9.3 mm at 50 Hz, 0.66 mm at 10 kHz, 66 um at 1 MHz; resolve delta '
         'with >= 3 cells near conductor surfaces. Quasi-static: displacement current neglected (sizes << the '
         'wavelength c / f).',
    tip='Frequency of the alternating currents and fields.\n\n'
        'Higher frequencies push the current to conductor surfaces (skin effect): copper about 9 mm deep at 50 Hz, '
        '0.7 mm at 10 kHz. Use cells smaller than that depth near conductors.')
_SETTINGS = (_GEOMETRY_PARAM, _FREQUENCY_PARAM, DISCRETIZATION_PARAM)


# ---- material vocabulary ----
def _material_property(name, label, unit, help_text, tip):
    """AC material property Param, >= 0 (sigma_dc entries: kind, unit, bounds and flags as electric_currents')."""
    return optional_material_param(name, label, unit, MAGNETIC_PROPERTY_ARRAYS[name], 0.0, help_text, tip)


_SIGMA_DC_HELP = (
    'Isotropic static conductivity [S/m] carrying the eddy and solid-conductor currents (out of the plane: '
    'magnetic_fields_ac needs sigma_dc_xx = sigma_dc_yy, else a run error); the same arrays as electric_currents, so '
    'a sigma_dc(heat_T) expression reaches both. Unset = 0 (no induced current). Ignored inside stranded coils (no '
    'eddy currents in a stranded winding; there it sets the wire conductivity of the coil resistance). Typical '
    'copper 5.8e7, aluminium 3.8e7, stainless steel 1.4e6, electrical steel 2e6 (laminations: use the stack value '
    'or a lower effective one), sea water 4.8.')
_SIGMA_DC_TIP = ('Electrical conductivity of the material, where alternating fields induce eddy currents; empty means '
                 'an insulator.\n\nTypical: copper 5.8e7, aluminium 3.8e7, steel about 1e6 to 2e6.')
_MU_LOSS_HELP = (
    'Magnetic loss: imaginary part of the relative permeability, mu = mu_dc - j mu_dc_loss (exp(+j omega t)); '
    '>= 0, unset = 0 (the array is allocated only when a material sets it). Dissipates q_magnetic = omega '
    'Im(nu) |B|^2 / 2 with nu = 1 / (mu0 mu). Tabulated data (mu\', mu\'\') at the solve frequency: an expression of '
    'the swept frequency parameter covers a dispersive material. Typical: soft ferrite mu_dc 2000, mu_dc_loss 10 - '
    '200 (rising with f); Metglas ~1e4 / 1e3 at 100 kHz.')
_MATERIAL_PROPERTIES = (
    *_DC_MATERIAL_PROPERTIES[:3],   # mu_dc, mu_dc_xx, mu_dc_yy (the magnetostatics Param objects)
    _material_property('mu_dc_loss', 'Magnetic loss (imaginary permeability)', '1', _MU_LOSS_HELP,
                       'How much magnetic energy the material turns into heat per cycle; empty means none.\n\n'
                       'Typical soft ferrites 10 to 200, rising with frequency.'),
    _material_property('sigma_dc', 'Static conductivity', 'S/m', _SIGMA_DC_HELP, _SIGMA_DC_TIP),
    _material_property('sigma_dc_xx', 'Static conductivity along x', 'S/m',
                       'DC conductivity along x; overrides sigma_dc for this axis (magnetic_fields_ac: must equal '
                       'sigma_dc_yy).',
                       "Static conductivity along x; replaces 'Static conductivity' there."),
    _material_property('sigma_dc_yy', 'Static conductivity along y', 'S/m',
                       'DC conductivity along y; overrides sigma_dc for this axis (magnetic_fields_ac: must equal '
                       'sigma_dc_xx).',
                       "Static conductivity along y; replaces 'Static conductivity' there."),
)


# ---- feature texts ----
_CURRENT_SOURCE_DESCRIPTION = (
    'Applied current density amplitude J [A/m^2] at phase 0 on the selected cells, added to field J_applied_re '
    '(several sources and coils add; the coupled slot current_density, field J_coupled, is separate): planar J_z out '
    'of the drawing, axisymmetric J_phi circling the axis. May depend on x and y. It is an impressed current: '
    'conducting material under it still carries eddy currents (use a stranded coil for a winding).')
_COIL_DESCRIPTION = (
    "Coil or conductor carrying a time-harmonic current. conductor 'stranded': many thin insulated turns, J = turns * "
    'I / area uniform over the selected cells (I = current at phase deg), no eddy currents in its cells (their '
    'sigma_dc is the wire conductivity of the coil resistance, times fill_factor); always current-driven. conductor '
    "'solid': one massive conductor (turns ignored) with J = sigma_dc (v - j omega A) / r, the applied field v uniform "
    "over it (planar v = voltage per m of length; axisymmetric the loop voltage around the ring / (2 pi r)): drive "
    "'current' prescribes the total current (one circuit row; the solve finds v, skin and proximity effects included), "
    "drive 'voltage' the loop voltage [V/m planar, V axisymmetric] (the solve finds the current). Its cells need "
    'sigma_dc > 0. Go / return pairs: two coil features with currents +I / -I (or phase 180). Result \'table\': one '
    'row [feature, ' + ', '.join(COIL_RESULT_KEYS) + '] of the latest solve and the same extras: the complex current '
    'and loop voltage (stranded: R_dc I + j omega flux linkage), Z = V / I: resistance Re Z, inductance Im Z / omega, '
    'power Re(V I*) / 2; planar per m of length (A, V/m, Ohm/m, H/m, W/m), axisymmetric whole rings (A, V, Ohm, H, '
    'W). With several conductors each Z includes the mutual terms (for series +I / -I the resistances and '
    'inductances add to the loop values). Scalars <feature>_<key> carry the same numbers.')
_PROBE_DESCRIPTION = ('Vector potential amplitude |A| (planar A_z, axisymmetric A_phi) [T m] of the cell containing '
                      'the point, one sample per solve plus one at the end of the run.')
_SLOT_DOCS = ('A/m^2', 'Coupled current density J_z (axisymmetric J_phi; field J_coupled), a real amplitude at phase 0 '
                       'added to the current_source and coil features; each transfer replaces it.')
_SCALARS_HELP = (
    'energy: time-averaged magnetic energy (1/4) Re(A^H K A) of the free cells (J/m planar, J axisymmetric); '
    'eddy_loss: time-averaged ohmic loss, the integral of q_eddy = sigma_dc |E|^2 / 2 over the conducting cells '
    '(induced and solid-conductor currents; stranded coils excluded) in W/m or W; magnetic_loss: (omega / 2) Im(A^H K '
    'A), the mu_dc_loss dissipation (W/m, W; the integral of q_magnetic only approximately at material interfaces); '
    'per enabled coil feature <coil>_' + ', <coil>_'.join(COIL_RESULT_KEYS) + ' as its table result.')
_PHYSICS_DESCRIPTION = (
    '2D time-harmonic magnetic fields (eddy currents) at one frequency (setting frequency), quasi-static: curl(nu '
    'curl A) + j omega sigma_dc A = J with nu = 1 / (mu0 (mu_dc - j mu_dc_loss)), phasors exp(+j omega t) of peak '
    'amplitude; planar A_z of currents along z, or axisymmetric the flux function psi = r A_phi of currents J_phi on '
    'the planar operator with coefficients nu / r (setting geometry), as magnetostatics. Solved on the host by one '
    'complex SuperLU factorisation of the finite-volume system (scipy required; solid conductors with a prescribed '
    'current add one bordered circuit row each); a later operator change (sigma_dc(heat_T), mu_dc) first tries GMRES '
    'preconditioned with the old factor, refactorising only when it misses. Fields: A, B (re / im parts and '
    'magnitudes; B from the complex face fluxes), J_conduction = sigma_dc E with E = -j omega A (+ v of solid '
    'conductors), J_total, q_eddy, q_magnetic and w_mag (time averages), J_applied, J_coupled. Faces default to '
    'magnetic insulation (A = 0); external_field holds the outer ring at a uniform applied field amplitude (phase '
    '0, the phase reference of held values). No remanence (a permanent magnet is static: magnetostatics). A '
    'time-dependent study re-solves only when a material, a source or a held value changed. Fields are '
    'edge-extended into an EM PML ring. On a triangle mesh (MeshSpec kind triangles) complex finite elements of the '
    'discretization setting (P2 default, P1) solve for planar A_z or axisymmetric A_phi itself (not psi) in B-space, '
    'with the magnetostatics conditions on triangles (faces on the window edge, external_field on the window-edge '
    'nodes, fixed_potential psi / r on the element nodes) and the same coils, circuit rows, scalars and results; state '
    'fields are per element (A at the centroid, element means of B and the loss / energy densities; plots show A and B '
    'on the nodes); a solid conductor touching the axis is a run error (its applied field v / r is singular there). '
    "There the Automatic linear solver is a double cuDSS LDL^T on the gpu backend with nvmath-python (back_end 'auto' "
    "= 'cudss'; an operator change refactorises on the kept analysis instead of GMRES; a cuDSS refusal switches to "
    'host SuperLU with a warning), else host SuperLU. The instance is usually named magnetic.')
_PHYSICS_SUMMARY = ('Alternating magnetic fields with eddy currents, skin and proximity effects in conductors, planar '
                    'or rotationally symmetric. Gives the field, the losses and coil resistances and inductances.')


# ---- feature callbacks ----
def _apply_coil(ctx, solver, feature_spec, values, selection):
    return solver.add_coil(feature_spec.name, ctx.domain_mask(selection.entities), current=values['current'],
                           phase=values['phase'], turns=values['turns'], conductor=values['conductor'],
                           drive=values['drive'], voltage=values['voltage'], fill_factor=values['fill_factor'])


def _coil_result(session, solver, coil, feature_spec, options):
    """One table row [feature, COIL_RESULT_KEYS...] of the latest solve."""
    result = solver.coil_results(coil)
    units = (('A', 'A', 'V', 'V', 'Ohm', 'H', 'W') if solver.axisymmetric
             else ('A', 'A', 'V/m', 'V/m', 'Ohm/m', 'H/m', 'W/m'))
    return {'kind': 'table', 'columns': ['feature', *COIL_RESULT_KEYS], 'units': ['', *units],
            'rows': [[coil.name, *(result[key] for key in COIL_RESULT_KEYS)]], **result}


_apply_probe, _probe_result = make_point_probe('vector potential amplitude', 'T m')


# ---- validation (CPU only) ----
def _material_issues(vctx, physics_spec, path):
    """Once per model: remanence is ignored (warning per material); no conducting material (info)."""
    if not reports_model_checks(vctx, physics_spec, MAGNETIC_AC_TYPE_NAME):
        return []
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    issues = []
    any_conductor = False
    for material in vctx.spec.materials:
        if not material.selection:
            continue
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            continue
        any_conductor = any_conductor or conducts(properties, vctx.parameters, 'sigma_dc')
        if any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value != 0.0)
               for name in REMANENCE_ARRAYS):
            issues.append(Issue('warning', ('material', material.name),
                                f"material '{material.name}' sets a remanence, which magnetic_fields_ac "
                                f"'{physics_spec.name}' ignores (a permanent magnet is a static field, not a phasor "
                                'at the frequency): solve it with magnetostatics'))
        saturates = evaluates_to(properties.get(SATURATION_ARRAY), vctx.parameters, lambda value: value > 0.0)
        saturation_names = [name for name, sets in ((SATURATION_ARRAY, saturates),
                                                    (BH_CURVE_PROPERTY, properties.get(BH_CURVE_PROPERTY) is not None))
                            if sets]
        if saturation_names:
            issues.append(Issue('warning', ('material', material.name),
                                f"material '{material.name}' sets {' and '.join(saturation_names)}, which "
                                f"magnetic_fields_ac '{physics_spec.name}' ignores (linear phasors at mu_dc; "
                                'saturation needs magnetostatics)'))
    if not any_conductor:
        issues.append(Issue('info', path, 'no selected material sets sigma_dc > 0: without conductors the solve is '
                                          'magnetostatics at phase 0 (no eddy currents; solid coils need sigma_dc)'))
    return issues


def _solver_issues(vctx, physics_spec, settings_values, path):
    """scipy missing (error); a cusolver_qr node (error: host matrices only); the window size (triangles: the dofs +
    circuit rows) against the host SuperLU limits (not on a cuDSS back end: its own estimate guards it)."""
    if not scipy_available():
        return [Issue('error', path, 'magnetic_fields_ac assembles on the host with scipy (SuperLU or cuDSS factor), '
                                     'and scipy is not installed: pip install scipy')]
    solvers = vctx.solvers(physics_spec) or {}
    resolved = solvers.get('linear')
    if resolved is not None and resolved.values.get('back_end') == 'cusolver_qr':
        return [Issue('error', path, f"linear solver '{resolved.node}': magnetic_fields_ac factorises host matrices "
                                     "(SuperLU or cuDSS): use back_end 'auto', 'scipy_splu' or 'cudss', or Automatic")]
    mesh = vctx.mesh
    if mesh is None or linear_solver_cudss(solvers):
        return []
    if mesh.kind == 'triangles':
        order = element_order(settings_values)
        dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
        issue = triangle_size_issue(dofs + _circuit_rows(physics_spec), order)
        return [] if issue is None else [Issue(issue[0], path, issue[1])]
    sides = mesh.npml_sides
    issue = size_issue('scipy_splu', mesh.nx - sides['x_lo'] - sides['x_hi'], mesh.ny - sides['y_lo'] - sides['y_hi'],
                       False, False, remedy='coarsen the mesh away from the conductors or shrink the domain')
    return [] if issue is None else [Issue(issue[0], path, issue[1])]


def _validate_coil(vctx, physics_spec, feature_spec, values):
    """A voltage drive needs a solid conductor."""
    if values.get('drive') == 'voltage' and values.get('conductor') == 'stranded':
        return [Issue('warning', feature_path(physics_spec, feature_spec),
                      "drive 'voltage' applies to conductor 'solid' only: a stranded coil is driven by its current")]
    return []


def _validate_ac(vctx, physics_spec, settings_values):
    path = ('physics', physics_spec.name)
    issues = validate_magnetic_common(vctx, physics_spec, settings_values, MAGNETIC_AC_TYPE_NAME)
    issues.extend(_solver_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_material_issues(vctx, physics_spec, path))
    return issues


# ---- physics factory callbacks ----
def _create(ctx, physics_spec, values):
    """MagneticACPhysics (tensor) or MagneticACFEMPhysics (triangles, the discretization order)."""
    host_solver = ctx.solvers(physics_spec.name)['linear']
    if ctx.domain.grid_kind == 'triangles':
        from .solver_ac_fem import MagneticACFEMPhysics   # lazy: no solver code at definition import time
        return MagneticACFEMPhysics(ctx.domain, values['frequency'], axisymmetric=is_axisymmetric(values),
                                    discretization=values[DISCRETIZATION_SETTING], host_solver=host_solver)
    from .solver_ac import MagneticACPhysics   # lazy: no solver code at definition import time
    return MagneticACPhysics(ctx.domain, values['frequency'], axisymmetric=is_axisymmetric(values),
                             host_solver=host_solver)


_N_FIELDS = len(MAGNETIC_AC_FIELD_INFOS) - 3   # fields.AC_FIELD_NAMES (the infos add mu_dc_xx / _yy, sigma_dc_xx)
_HOST_WINDOW_ARRAYS = 24   # complex128-equivalent host window arrays of one assembly / field update (estimate)
# Triangle meshes (uncalibrated): complex128 values per quadrature point of one post-processing (B (2), A, E, q_eddy,
# q_magnetic, w_mag, power and their element-integral temporaries) and host CSR copies (curl, mass, operator, eliminated
# rows / block, bordered)
_FEM_POINT_VALUES = 12
_FEM_CSR_COPIES = 6


def _cudss_estimate(context, n_unknowns, entries) -> MemoryEstimate:
    """The 'linear' cuDSS LDL^T of the complex symmetric host system (RecycledFactor: factorize_host at the first
    solve, so run items; refactorised on its plan when the operator follows a state: index maps kept, refresh peak
    transient), its solve vectors (transient) and the factor outside the pool; nothing on another back end."""
    from ...solvers.cudss_factor import (factor_device_bytes, index_map_bytes, operator_copy_bytes, refresh_peak_bytes,
                                         solve_buffer_bytes, solve_peak_bytes)
    back_end = cudss_back_end(context.solvers)
    if back_end is None:
        return MemoryEstimate()
    mixed = back_end == 'cudss_mixed'
    run_items = [('cuDSS lower-triangle CSR copies (complex128)',
                  operator_copy_bytes(n_unknowns, entries, COMPLEX128_BYTES, symmetric=True, mixed=mixed)),
                 ('cuDSS rhs buffer and row scale', solve_buffer_bytes(n_unknowns, COMPLEX128_BYTES, mixed))]
    transient_items = [('linear solve', 'cuDSS solve vectors (complex128; mixed: the GMRES basis)',
                        solve_peak_bytes(n_unknowns, COMPLEX128_BYTES, mixed))]
    if material_follows_state(context, _OPERATOR_PROPERTY_NAMES):   # refactorize on the plan: the device route
        run_items.append(('cuDSS index maps of the refactorisations', index_map_bytes(n_unknowns, entries)))
        transient_items.append(('refactorisation', 'cuDSS refactorisation (uploaded values, refresh or probe peak)',
                                refresh_peak_bytes(entries, COMPLEX128_BYTES, symmetry_check=True,
                                                   n_unknowns=n_unknowns, mixed=mixed)))
    permanent, extra = factor_device_bytes(n_unknowns, entries, COMPLEX128_BYTES, 2, mixed)
    return MemoryEstimate.from_items(run_items=run_items, transient_items=transient_items,
                                     external_items=(('cuDSS factor (outside the cupy pool)', permanent),),
                                     external_extra=extra)


def _triangle_memory_estimate(settings_values, mesh, context):
    """MagneticACFEMPhysics: per-element float32 fields on the run backend; host complex CSR copies, the SuperLU factor
    (FEM fill over the 2D heuristic; cuDSS: _cudss_estimate), nodal values, the curl quadrature cache and the point
    values of one post-processing (uncalibrated)."""
    order = element_order(settings_values)
    dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    unknowns = dofs + _circuit_rows(context.physics_spec)
    elements = int(mesh.n_elements)
    points = triangle_rule(PLANAR_DEGREES[order] + int(is_axisymmetric(settings_values))).weights.size
    local = 6 if order == 2 else 3
    entries = int(round(lagrange_entries_per_dof(mesh, order) * dofs))
    factor_items = () if linear_solver_cudss(context.solvers) else (
        ('SuperLU L + U factors (complex128, FEM heuristic, uncalibrated)',
         int(FEM_FILL_FACTOR * host_lu_bytes(unknowns))),)
    own = MemoryEstimate.from_items(
        ((f'element fields ({_N_FIELDS} x float32)', _N_FIELDS * FLOAT32_BYTES * elements),),
        host_items=factor_items + (
                    ('host matrices (complex128 CSR copies)', _FEM_CSR_COPIES * COMPLEX128_CSR_ENTRY_BYTES * entries),
                    ('nodal A, B and loads (complex128)', 5 * COMPLEX128_BYTES * dofs),
                    ('curl quadrature cache (float64)', (2 * local + 3) * points * FLOAT64_BYTES * elements),
                    ('quadrature point values of one post-processing (complex128)',
                     _FEM_POINT_VALUES * points * COMPLEX128_BYTES * elements),
                    ('assembly COO (transient, complex128)', 3 * COMPLEX128_BYTES * local * local * elements)))
    return combine((own, _cudss_estimate(context, unknowns, entries)))


def _memory_estimate(settings_values, mesh, context):
    """Device: the float32 phasor fields + the real window DiffusionSystem (anchor check, held cells) + its nu_xx /
    nu_yy (+ a cuDSS factor: _cudss_estimate); host: SuperLU factor, operator / eliminated / bordered matrices and
    window arrays; run: one window float32 transfer buffer. Triangles: _triangle_memory_estimate."""
    if mesh.kind == 'triangles':
        return _triangle_memory_estimate(settings_values, mesh, context)
    window_cells = window_cell_count(mesh)
    n_rows = _circuit_rows(context.physics_spec)
    factor_items = () if linear_solver_cudss(context.solvers) else (   # a cuDSS factor: _cudss_estimate
        ('SuperLU L + U factors (heuristic)', host_lu_bytes(window_cells + n_rows)),)
    own = MemoryEstimate.from_items(
        ((f'fields ({_N_FIELDS} x float32)', _N_FIELDS * FLOAT32_BYTES * int(mesh.n_cells)),
         ('reluctivities nu_xx / nu_yy (window)', 2 * FLOAT64_BYTES * window_cells)),
        host_items=factor_items + (
                    ('host matrices (kept stencil, operator, eliminated, bordered; complex128)',
                     4 * 5 * COMPLEX128_CSR_ENTRY_BYTES * window_cells),
                    ('host window arrays (complex128)', _HOST_WINDOW_ARRAYS * COMPLEX128_BYTES * window_cells)),
        run_items=(('field transfer buffer (window float32)', FLOAT32_BYTES * window_cells),))
    return combine((own, diffusion_system_estimate(window_cells, 'float64'),
                    _cudss_estimate(context, window_cells + n_rows, 5 * window_cells)))


# run-time estimate guess: GMRES iterations of one recycled-factor solve after an operator change (sigma_dc / mu_dc
# following a state field; calibrate from the 'linear' role counters gmres_iterations / recycled_solves)
RECYCLED_GMRES_ITERATIONS_GUESS = 8
_OPERATOR_PROPERTY_NAMES = tuple(param.name for param in _MATERIAL_PROPERTIES)


def _circuit_rows(physics_spec) -> int:
    """Bordered circuit rows: enabled coils with conductor 'solid' and drive 'current' (raw params, defaults filled)."""
    return sum(1 for feature in physics_spec.features
               if feature.enabled and feature.type == 'coil' and isinstance(feature.params, dict)
               and feature.params.get('conductor', 'stranded') == 'solid'
               and feature.params.get('drive', 'current') == 'current')


def _time_estimate(settings_values, mesh, context):
    """Complex factor of the window system (+ one bordered row per current-driven solid conductor) through the 'linear'
    role (host SuperLU or cuDSS) after the real window assembly and the host build; re-solves reuse the factor, an
    operator following a state is rebuilt and solved by recycled-factor GMRES (cuDSS: refactorised on its plan); per
    step of a time_dependent study when driven. Triangles: the 'linear' work on the FEM unknowns only."""
    rows = _circuit_rows(context.physics_spec)
    rebuilds = material_follows_state(context, _OPERATOR_PROPERTY_NAMES)
    driven = coupling_target(context) or rebuilds
    per_step = context.time_dependent and driven
    if per_step:   # at most one re-solve per step (only after an input changed); the first solve is step 1's
        later, kind = 1.0, 'bounded'
    else:
        solves, kind, _ = context.static_solves() if driven else (1.0, 'exact', None)
        later = solves - 1.0

    # a cuDSS factor is not recyclable: each operator rebuild refactorises on its plan (and solves) instead of GMRES
    on_cudss = linear_solver_cudss(context.solvers)
    refactorised = on_cudss and rebuilds

    def linear_work(unknowns, nnz):
        if refactorised:
            counts = dict(solves=later if per_step else 1.0 + later, solves_kind=kind, per_step=per_step,
                          factorizations=later if per_step else 1.0 + later, factorizations_kind=kind,
                          factorizations_per_step=per_step)
        else:
            counts = dict(solves=1.0 if per_step or rebuilds else 1.0 + later,
                          solves_kind='exact' if rebuilds else kind, per_step=per_step and not rebuilds,
                          factorizations=1.0, factorizations_per_step=False)
        host = {} if on_cudss else {'backend': 'cpu'}   # SuperLU: factorize_host on either backend
        return context.solver_work('linear', unknowns=unknowns, nnz=nnz, value_bytes=16, dimensions=2, **counts,
                                   **host)

    if mesh.kind == 'triangles':
        order = element_order(settings_values)
        dofs = float(int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0))
        skipped = 'triangle assembly and post-processing' + ('' if on_cudss else ' and recycled-factor GMRES')
        return te.combine((te.TimeEstimate(not_estimated=(skipped,)),
                           linear_work(dofs + rows, lagrange_entries_per_dof(mesh, order) * dofs)))
    cells = float(context.window_cells())
    unknowns = cells + rows
    nnz = 5.0 * cells
    # real window DiffusionSystem (anchor check, held cells); host complex operator, eliminated and sliced copies
    build_ops = (te.Elementwise(6 * FLOAT64_BYTES * nnz, 2 * FLOAT64_BYTES * nnz, launches=30), te.PythonCalls(60),
                 te.HostStream(2 * 3 * COMPLEX128_CSR_ENTRY_BYTES * nnz), te.PythonCalls(100))
    # source and bordered system, the host phasor field arrays, one float32 transfer per field
    update_ops = (te.HostStream(2 * _HOST_WINDOW_ARRAYS * COMPLEX128_BYTES * cells
                                + 2 * COMPLEX128_CSR_ENTRY_BYTES * nnz),
                  te.Transfer(_N_FIELDS * FLOAT32_BYTES * cells), te.PythonCalls(80))
    items = [te.TimeItem('window assembly and host build', 'build', 1.0, 'exact', build_ops)]
    items.append(te.TimeItem('source and field update', 'solve', 1.0 if per_step else 1.0 + later, kind, update_ops,
                             per_step=per_step, counter=('solvers', context.name, 'counters', 'solves')))
    linear = linear_work(unknowns, nnz)
    if rebuilds and later > 0 and on_cudss:
        items.append(te.TimeItem('operator rebuild', 'build', later, kind, build_ops, per_step=per_step))
    elif rebuilds and later > 0:
        gmres_ops = (te.HostSolve(unknowns, 2, True),
                     te.HostStream(COMPLEX128_CSR_ENTRY_BYTES * nnz + 4 * COMPLEX128_BYTES * unknowns),
                     # complex Gram-Schmidt against half the Krylov basis on average
                     te.DenseFlops(8.0 * unknowns * 0.5 * RECYCLED_GMRES_ITERATIONS_GUESS, host=True))
        items += [te.TimeItem('operator rebuild', 'build', later, kind, build_ops, per_step=per_step),
                  te.TimeItem('recycled-factor GMRES iterations', 'iteration', later * RECYCLED_GMRES_ITERATIONS_GUESS,
                              'guess', gmres_ops, limit=later * RECYCLE_MAX_ITERATIONS, per_step=per_step,
                              counter=('solvers', context.name, 'roles', 'linear', 'counters', 'gmres_iterations'))]
    return te.combine((te.TimeEstimate(tuple(items)), linear))


def _scalar_outputs(settings_values, physics_spec):
    """energy, eddy_loss, magnetic_loss, then <coil>_<key> per enabled coil feature."""
    names = ['energy', 'eddy_loss', 'magnetic_loss']
    for feature in physics_spec.features:
        if feature.enabled and feature.type == 'coil':
            names.extend(f'{feature.name}_{key}' for key in COIL_RESULT_KEYS)
    return tuple(names)


# ---- type ----
_DC_FEATURES_BY_NAME = {feature.name: feature for feature in _DC_FEATURES}
_FEATURES = (
    *(_DC_FEATURES_BY_NAME[name] for name in _SHARED_FEATURE_NAMES),
    FeatureType('current_source', 'Current density', 'domain_condition', 'domain',
                params=(Param('density', 'float', 1e6, label='Current density amplitude', unit='A/m^2',
                              field_dependent=True,
                              help='J_z (planar) or J_phi (axisymmetric) amplitude at phase 0; may depend on x and y.',
                              tip='Amplitude of the alternating current per area flowing out of the drawing (in '
                                  'axisymmetric models: around the axis).'),),
                summary='Drives a given alternating current per area through the selected domains.',
                description=_CURRENT_SOURCE_DESCRIPTION, apply=_apply_current_source, grids=_BOTH_GRIDS),
    FeatureType('coil', 'Coil or conductor', 'domain_condition', 'domain', result_kind='table',
                params=(Param('conductor', 'choice', 'stranded', label='Conductor', choices=CONDUCTORS,
                              choice_labels=(('stranded', 'Stranded winding'), ('solid', 'Solid conductor')),
                              help="'stranded': uniform J = turns I / area, no eddy currents in it; 'solid': one "
                                   'massive conductor with its own skin and proximity effects (circuit row).',
                              tip='- Stranded winding: many thin turns, the current is spread evenly.\n'
                                  '- Solid conductor: one massive bar or ring; the current redistributes by the skin '
                                  'and proximity effects.'),
                        Param('drive', 'choice', 'current', label='Drive', choices=DRIVES,
                              choice_labels=(('current', 'Current'), ('voltage', 'Voltage')),
                              visible_if=(('conductor', ('solid',)),),
                              help="Solid conductors: 'current' prescribes the total current, 'voltage' the loop "
                                   'voltage (stranded coils are always current-driven).',
                              tip='Whether the current or the voltage of the conductor is given.'),
                        Param('current', 'float', 1.0, label='Current amplitude', unit='A',
                              visible_if=(('conductor', ('stranded',)), ('drive', ('current',))),
                              help='Peak current per turn (solid: the total current), > 0 along +z (axisymmetric: '
                                   'counter-clockwise seen from +z), at the given phase.',
                              tip='Peak value of the alternating current; the sign sets the direction.'),
                        Param('voltage', 'float', 1e-3, label='Voltage amplitude', unit='V/m',
                              space_units=(('axisymmetric', 'V'),), visible_if=(('drive', ('voltage',)),),
                              help="Solid conductor with drive 'voltage': peak loop voltage, in V per m of length "
                                   '(planar) or V around the ring (axisymmetric).',
                              tip='Peak voltage driving the conductor (planar models: per metre of length).'),
                        Param('phase', 'float', 0.0, label='Phase', unit='deg',
                              help='Phase of the current (or voltage) phasor in degrees; held potentials and the '
                                   'external field are at phase 0.',
                              tip='Phase shift of this current in degrees; 180 reverses it.'),
                        Param('turns', 'int', 1, label='Turns', unit='1', min=1,
                              visible_if=(('conductor', ('stranded',)),),
                              help='Number of turns spread over the selected cross-section (stranded only).',
                              tip='Number of wire turns in the selected cross-section.'),
                        Param('fill_factor', 'float', 1.0, label='Copper fill factor', unit='1', min=0.0,
                              max=1.0, min_exclusive=True, visible_if=(('conductor', ('stranded',)),),
                              help='Stranded only: conducting share of the cross-section; the coil resistance is '
                                   'turns^2 / (fill_factor area^2) times the sum of (path length / sigma_dc) area '
                                   'over its cells (0 where sigma_dc is unset: an ideal winding).',
                              tip='Share of the winding cross-section that is wire; sets the coil resistance.')),
                summary='A winding of thin turns or a solid conductor carrying an alternating current; reports its '
                        'current, voltage, resistance, inductance and loss.',
                description=_COIL_DESCRIPTION, apply=_apply_coil, result=_coil_result, validate=_validate_coil,
                grids=_BOTH_GRIDS),
    FeatureType('probe', 'Vector potential probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
                result_kind='timeseries',
                summary='Records the amplitude of the magnetic vector potential at a point after every solve.',
                description=_PROBE_DESCRIPTION, apply=_apply_probe, result=_probe_result,
                validate=validate_points_inside, grids=_BOTH_GRIDS),
)

_FIELD_INFOS = tuple(MAGNETIC_AC_FIELD_INFOS.values())


def _mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
    """GUI text of the discretisation on mesh_kind (no extra problems: validation and the run cover the rest)."""
    if mesh_kind == 'triangles':
        order = 'quadratic' if element_order(settings_values) == 2 else 'linear'
        return (f'Finite elements on triangles ({order} elements) for the complex vector potential, one direct '
                'factorisation per frequency on the host. Every feature works here; a solid conductor must not touch '
                'the symmetry axis.', [])
    return 'Finite volumes on rectangular cells, one direct factorisation per frequency on the host.', []

MAGNETIC_FIELDS_AC = PhysicsType(
    name=MAGNETIC_AC_TYPE_NAME,
    label='Magnetic Fields, AC (2D)',
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
    material_schema=dict(MAGNETIC_AC_SCHEMA),
    optional_material_schema=dict(OPTIONAL_MAGNETIC_AC_SCHEMA),
    state_fields=_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    source_slots={'current_density': _SLOT_DOCS},
    supported_studies=('stationary', 'time_dependent'),
    default_study={'type': 'stationary'},
    uses_dispersion=False,
    create=_create,
    validate=_validate_ac,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALARS_HELP,
    dimensions=(2,),
    solver_roles={'linear': MAGNETIC_AC_LINEAR_ROLE},
    backends=('gpu', 'cpu'),   # host SuperLU either way; fields and the real window system on the Domain's xp
    grids=('tensor', 'triangles'),   # triangles: MagneticACFEMPhysics (solver_ac_fem.py)
    mesh_method=_mesh_method,
)


def _selftest_cases():
    """The magnetic_fields_ac SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_ac_fem import AC_FEM_SELFTEST_CASES
    from .selftests_backend import AC_BACKEND_SELFTEST_CASES
    return AC_BACKEND_SELFTEST_CASES + AC_FEM_SELFTEST_CASES


def register(registry):
    """The magnetic_fields_ac type, the 'busbar_proximity' and 'flux_concentrator_needle_ac' templates, then the
    selftest loader."""
    registry.register_physics(MAGNETIC_FIELDS_AC)
    registry.register_template(
        'busbar_proximity', BUSBAR_LABEL, BUSBAR_DESCRIPTION, busbar_proximity_spec,
        summary='Two copper bars carrying opposite alternating currents: the current crowds onto the facing sides, '
                'raising the resistance with frequency. Planar, frequency sweep.')
    registry.register_template(
        'flux_concentrator_needle_ac', NEEDLE_AC_LABEL, NEEDLE_AC_DESCRIPTION, flux_concentrator_needle_ac_spec,
        summary='The flux-concentrator needle in an alternating field: eddy currents in the needle reduce the gain '
                'below its tip as the frequency rises. Axisymmetric, frequency sweep.')
    registry.register_selftests(_selftest_cases)

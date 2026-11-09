"""Model-layer definition of the physics type 'drift_diffusion' (settings, doping / contact / gate / generation
features, material vocabulary, validation, memory estimate) plus its template and selftests. Loaded lazily by the
registry (physics/semiconductor/__init__.py must not import it); its EM couplings live in
coupling/definitions_semiconductor.py."""
import math
from dataclasses import replace

from ...core import time_estimate as te
from ...core.memory_estimate import (COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, FLOAT64_CSR_ENTRY_BYTES,
                                     MemoryEstimate, combine, host_lu_bytes, phase_before_items, window_cell_count)
from ...model.definition_helpers import (AXISYMMETRIC_SPACE_SETTINGS, CUDSS_LINEAR_ROLE, axisymmetric_issues,
                                         coupling_target, evaluates_to, geometry_param, held_object_conflicts,
                                         host_direct_back_end, is_axisymmetric, linear_solver_cudss,
                                         material_follows_state, numpy_calls, reports_model_checks, resolved_back_end)
from ...model.expressions import is_identifier
from ...model.schema import FeatureType, Issue, Param, PhysicsType, resolve_values
from ...solvers.cudss_factor import (CUDSS_SOLVE_PHASE, factor_device_bytes, index_map_bytes, operator_copy_bytes,
                                     refresh_peak_bytes, solve_buffer_bytes, solve_peak_bytes)
from ...solvers.host_direct import factor_label, host_direct_precision, solver_host_bytes, superlu_bytes_scale
from .drift_diffusion import COMPONENTS
from .renderer import SEMICONDUCTOR_FIELD_INFOS
from .selftests import PN_DIODE_DESCRIPTION, PN_DIODE_LABEL, PN_DIODE_SUMMARY, silicon_pn_photodiode_spec
from .solver import (CONTACT_CURRENT_PREFIX, GATE_CHARGE_PREFIX, GENERATION_SLOT, SEMICONDUCTOR_PROPERTY_ARRAYS,
                     SEMICONDUCTOR_SCHEMA, DriftDiffusionPhysics, linear_size_issue)
from .solver_fem import FEMDriftDiffusionPhysics, fem_linear_size_issue

DRIFT_DIFFUSION_TYPE_NAME = 'drift_diffusion'
_HELD_FEATURES = frozenset({'ohmic_contact', 'gate'})
_BOTH_GRIDS = ('tensor', 'triangles')   # the type and every feature run on triangle meshes too (solver_fem.py)


# ---- settings ----
_SETTINGS = (
    geometry_param(
        '2D drift-diffusion per unit z length (contact currents A/m, gate charges C/m)',
        '(r, z) drift-diffusion; every Poisson conductance, Scharfetter-Gummel face coefficient, face and cell volume '
        'is weighted by 2 pi r (x-faces at their own r, y-faces and volumes at the cell-centre r; 2 pi r_c dr dz is '
        'the exact ring volume), so densities and rates (N, P 1/m^3, J A/m^2, generation and recombination 1/(m^3 '
        's)) keep their units, while contact currents and gate charges (table results, scalar outputs current_ / '
        'charge_<feature>) are those of the whole ring in A and C (a resistance V / I in Ohm, a capacitance Q / V in '
        'F)',
        'contact currents and gate charges are per metre of depth.',
        'Contact currents and gate charges are those of the whole ring, in amperes and coulombs.',
        axis_text='no radial current or D flux, like the insulating outer faces'),
    Param('temperature', 'float', 300.0, label='Temperature', unit='K', min=0.0, min_exclusive=True,
          help='Lattice temperature setting the thermal voltage V_T = k_B T / q (25.85 mV at 300 K) of the Einstein '
               'relation D = mu V_T and the Boltzmann statistics. n_intrinsic is NOT rescaled with it: give a '
               'temperature-dependent n_intrinsic yourself (it may depend on heat_T). Default 300 K, the temperature '
               'of the library semiconductor data (silicon n_intrinsic 1e16 1/m^3), deliberately not heat T0 (293.15 '
               'K, where the optical indices apply): with a heat physics in the model set both to the same value '
               '(one parameter) if they should agree.',
          tip='Temperature of the semiconductor; 300 K is room temperature and the temperature of the library '
              'semiconductor data.\n\n'
              'The intrinsic carrier density of the material does not follow it automatically. The heat physics '
              'starts from 293.15 K by default: use one parameter for both when they should agree.'),
    Param('tolerance', 'float', 1e-8, label='Newton tolerance', unit='1', min=0.0, min_exclusive=True,
          help='Newton stops when the largest potential update is <= tolerance * V_T and the largest carrier update '
               '|dn| / (n + n_i) (holes alike) is <= tolerance over the free unknowns; quadratic convergence makes '
               'the final error far smaller. 1e-8 suits float64.',
          tip='How accurately each solve is converged, as a relative change; keep 1e-8.'),
    Param('max_iterations', 'int', 50, label='Newton iteration limit', min=1,
          help='Newton iterations per continuation step (and for the equilibrium start); a step that does not '
               'converge is halved, down to 1/64 of the first step, then the solve fails (run error).',
          tip='Largest number of Newton steps per continuation step; a step that fails is retried smaller.'),
    Param('max_bias_step', 'float', 0.1, label='Largest bias step', unit='V', min=0.0, min_exclusive=True,
          help='Largest change of any contact or gate voltage per continuation step (the solve ramps every applied '
               'voltage and the generation from the previous solution, the first time from equilibrium at 0 V); '
               'about 4 V_T keeps Newton safe for forward-biased junctions.',
          tip='The applied voltages are raised in steps no larger than this; smaller steps are slower but safer.'),
)


# ---- material vocabulary ----
def _material_property(name, label, unit, minimum=None, help_text='', tip='', min_exclusive=False):
    """Field- / state-dependent drift-diffusion material property (default None: unset properties are never written)."""
    return Param(name, 'float', None, label=label, unit=unit, min=minimum, min_exclusive=min_exclusive,
                 help=help_text, tip=tip, optional=True, field_dependent=True, state_dependent=True,
                 expands_to=SEMICONDUCTOR_PROPERTY_ARRAYS[name])


_EPS_DC_TIP = ('Relative permittivity for static fields; empty means vacuum, 1.\n\n'
               'Often far above the optical value: silica 3.9 against 2.1, silicon 11.7.')
_MATERIAL_PROPERTIES = (
    # eps_dc, eps_dc_xx / _yy: kind, unit, bounds and flags as the electrostatics entries (registry rule)
    _material_property('eps_dc', 'Static relative permittivity', '1', 1.0,
                       'Isotropic static relative permittivity (>= 1) of the Poisson equation: sets eps_dc_xx and '
                       'eps_dc_yy (component entries override it); unset = 1. Silicon 11.7, silica 3.9.',
                       tip=_EPS_DC_TIP),
    _material_property('eps_dc_xx', 'Static relative permittivity along x', '1', 1.0,
                       'Static relative permittivity along x (>= 1); overrides eps_dc for this axis.',
                       tip="Static relative permittivity along x; replaces 'Static relative permittivity' there."),
    _material_property('eps_dc_yy', 'Static relative permittivity along y', '1', 1.0,
                       'Static relative permittivity along y (>= 1); overrides eps_dc for this axis.',
                       tip="Static relative permittivity along y; replaces 'Static relative permittivity' there."),
    _material_property('n_intrinsic', 'Intrinsic carrier density', '1/m^3', 0.0,
                       'Intrinsic carrier density n_i [1/m^3]: cells with n_i > 0 are semiconductor (Poisson + '
                       'carriers), the rest insulators (Poisson only, no carriers). Silicon 1.0e16 at 300 K (1e10 '
                       'cm^-3), GaAs 2.1e12, germanium 2.0e19. Homojunctions only: one band structure, no '
                       'heterojunction band offsets.',
                       tip='Density of free electrons in the pure material; a value above zero makes the material a '
                           'semiconductor.\n\nSilicon about 1e16 per cubic metre.'),
    _material_property('mobility_n', 'Electron mobility', 'm^2/(V s)', 0.0,
                       'Electron mobility [m^2/(V s)], constant (no doping or field dependence; give an expression '
                       'for that). Lightly doped silicon 0.135 (1350 cm^2/Vs), at 1e24 1/m^3 ~0.03.',
                       tip='How easily electrons drift in an electric field; silicon about 0.135.'),
    _material_property('mobility_p', 'Hole mobility', 'm^2/(V s)', 0.0,
                       'Hole mobility [m^2/(V s)], constant. Lightly doped silicon 0.048 (480 cm^2/Vs).',
                       tip='How easily holes drift in an electric field; silicon about 0.048.'),
    _material_property('lifetime', 'Carrier lifetime', 's', 0.0,
                       'Shockley-Read-Hall lifetime (mid-gap trap) of both carriers: sets lifetime_n and lifetime_p; '
                       'unset = 1e-6 s. Silicon 1e-9 (heavily doped) .. 1e-3 (float zone).',
                       tip='Average time before a free electron and hole recombine through defects; 1 microsecond '
                           'when empty.', min_exclusive=True),
    _material_property('lifetime_n', 'Electron lifetime', 's', 0.0,
                       'SRH electron lifetime tau_n [s]; overrides lifetime.',
                       tip="Recombination lifetime of electrons; replaces 'Carrier lifetime'.", min_exclusive=True),
    _material_property('lifetime_p', 'Hole lifetime', 's', 0.0,
                       'SRH hole lifetime tau_p [s]; overrides lifetime.',
                       tip="Recombination lifetime of holes; replaces 'Carrier lifetime'.", min_exclusive=True),
    _material_property('radiative_recombination', 'Radiative recombination coefficient', 'm^3/s', 0.0,
                       'Band-to-band coefficient B [m^3/s] of R_rad = B (n p - n_i^2); unset = 0. Silicon 1.1e-20, '
                       'GaAs 7.2e-16.',
                       tip='Strength of direct (light-emitting) recombination; tiny in silicon, large in gallium '
                           'arsenide.'),
)


# ---- feature callbacks (worker thread, GPU) ----
def _apply_doping(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    solver.add_doping(mask, ctx.eval_field(values['donors'], mask=mask), ctx.eval_field(values['acceptors'], mask=mask))
    return None


def _apply_auger(ctx, solver, feature_spec, values, selection):
    solver.set_auger_recombination(ctx.domain_mask(selection.entities), values['electron_coefficient'],
                                   values['hole_coefficient'])
    return None


def _apply_generation(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    solver.add_generation(mask, ctx.eval_field(values['rate'], mask=mask))
    return None


def _apply_contact(ctx, solver, feature_spec, values, selection):
    region = solver.add_ohmic_contact(feature_spec.name, ctx.domain_mask(selection.entities), values['voltage'])
    ctx.follow_parameters(feature_spec.params, values, ('voltage',),
                          lambda voltage: solver.set_contact_voltage(region, voltage), solver,
                          f"feature '{feature_spec.name}'")
    return region


def _apply_gate(ctx, solver, feature_spec, values, selection):
    region = solver.add_gate(feature_spec.name, ctx.domain_mask(selection.entities), values['voltage'],
                             values['work_function_difference'])
    ctx.follow_parameters(feature_spec.params, values, ('voltage',),
                          lambda voltage: solver.set_gate_voltage(region, voltage), solver,
                          f"feature '{feature_spec.name}'")
    return region


def _contact_result(session, solver, region, feature_spec, options):
    electron, hole = solver.contact_currents(region)
    current = electron + hole
    unit = solver.current_unit
    voltage = solver.contact_voltage(region)
    return {'kind': 'table', 'columns': ['feature', 'voltage', 'current', 'electron_current', 'hole_current'],
            'units': ['', 'V', unit, unit, unit], 'rows': [[region.name, voltage, current, electron, hole]],
            'voltage': voltage, 'current': current, 'electron_current': electron, 'hole_current': hole}


def _gate_result(session, solver, region, feature_spec, options):
    charge = solver.gate_charge(region)
    voltage = solver.gate_voltage(region)
    return {'kind': 'table', 'columns': ['feature', 'voltage', 'charge'], 'units': ['', 'V', solver.charge_unit],
            'rows': [[region.name, voltage, charge]], 'voltage': voltage, 'charge': charge}


# ---- scalar outputs (E5) ----
_SCALAR_PREFIXES = {'ohmic_contact': CONTACT_CURRENT_PREFIX, 'gate': GATE_CHARGE_PREFIX}


def _scalar_outputs(settings_values, physics_spec):
    """current_<contact> per enabled ohmic_contact, charge_<gate> per enabled gate (DriftDiffusionPhysics.scalars);
    invalid or repeated feature names are skipped (validation reports them at the feature)."""
    names, seen = [], set()
    for feature in physics_spec.features:
        prefix = _SCALAR_PREFIXES.get(feature.type)
        if prefix is None or not feature.enabled or not is_identifier(feature.name) or feature.name in seen:
            continue
        seen.add(feature.name)
        names.append(f'{prefix}{feature.name}')
    return tuple(names)


_SCALAR_OUTPUTS_HELP = (
    'current_<contact> (A/m) per enabled ohmic_contact feature: its total current (electron + hole) per m of depth of '
    'the last solve, as the contact table result (positive = conventional current leaving the contact into the '
    'device); charge_<gate> (C/m) per enabled gate feature: its free charge per m of depth, as the gate table result; '
    "with geometry 'axisymmetric' both are whole-ring values in A and C. "
    'Example derived value {"name": "I_anode", "expression": "semiconductor_current_anode"} (operation mean of a '
    'constant is the value itself); a parameter sweep of a contact voltage then gives the I-V curve (with the sweep '
    'setting reuse_build the build is kept and each point continues from the previous converged state).')


# ---- validation (CPU only) ----
def _material_issues(vctx, physics_spec, path):
    """Model-wide (first drift_diffusion physics): error when no material sets n_intrinsic > 0; warning per
    semiconductor material without any mobility."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    if not reports_model_checks(vctx, physics_spec, DRIFT_DIFFUSION_TYPE_NAME):
        return []
    issues, any_semiconductor = [], False
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            any_semiconductor = True
            continue
        if not evaluates_to(properties.get('n_intrinsic'), vctx.parameters, lambda value: value > 0.0):
            continue
        any_semiconductor = any_semiconductor or bool(material.selection)
        if not any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value > 0.0)
                   for name in ('mobility_n', 'mobility_p')):
            issues.append(Issue('warning', ('material', material.name),
                                f"material '{material.name}' sets n_intrinsic but no mobility_n / mobility_p: its "
                                'carriers cannot move (no current), only recombine'))
    if not any_semiconductor:
        issues.append(Issue('error', path, 'no selected material sets n_intrinsic > 0: assign a semiconductor '
                                           '(e.g. library silicon) to the domain'))
    return issues


def _linear_solver_issues(vctx, physics_spec, settings_values, path):
    """The back end's size limit on the 3-unknown window system (scipy availability: the sparse_direct node, or
    Automatic falling back to cusolver_qr)."""
    back_end = resolved_back_end(vctx.solvers(physics_spec))
    mesh = vctx.mesh
    if mesh is None or back_end is None:
        return []
    if mesh.kind == 'triangles':   # nodal box method: 3 unknowns per P1 node, no cusolver_qr
        issue = fem_linear_size_issue(back_end, mesh.n_nodes)
        return [] if issue is None else [Issue(issue[0], path, issue[1])]
    sides = mesh.npml_sides
    issue = linear_size_issue(back_end, mesh.nx - sides['x_lo'] - sides['x_hi'],
                              mesh.ny - sides['y_lo'] - sides['y_hi'])
    return [] if issue is None else [Issue(issue[0], path, issue[1])]


def _validate_physics(vctx, physics_spec, settings_values):
    path = ('physics', physics_spec.name)
    issues = []
    if not any(feature.enabled and feature.type == 'ohmic_contact' for feature in physics_spec.features):
        issues.append(Issue('error', path, 'drift-diffusion needs at least one ohmic_contact: it fixes the Fermi '
                                           'level (a semiconductor without a contact has no unique steady state)'))
    issues.extend(_linear_solver_issues(vctx, physics_spec, settings_values, path))
    issues.extend(held_object_conflicts(vctx, physics_spec, _HELD_FEATURES, frozenset(), path))
    issues.extend(_material_issues(vctx, physics_spec, path))
    if is_axisymmetric(settings_values):
        issues.extend(axisymmetric_issues(vctx, physics_spec, path, 'drift_diffusion'))
    return issues


# ---- factory callbacks ----
def _create(ctx, physics_spec, values):
    physics_class = FEMDriftDiffusionPhysics if ctx.domain.grid_kind == 'triangles' else DriftDiffusionPhysics
    return physics_class(ctx.domain, temperature=values['temperature'],
                         linear_solver=ctx.solvers(physics_spec.name)['linear'],
                         tolerance=values['tolerance'], max_newton_iterations=values['max_iterations'],
                         max_bias_step=values['max_bias_step'], axisymmetric=is_axisymmetric(values))


# Host bytes per window cell of one Newton step: ~49 COO entries (20 per face, 2 faces per cell, 9 per cell) as int64
# rows / cols + float64 values, held about three times (assembly, pinned copy, equilibrated copy).
_COO_ENTRIES_PER_CELL = 49
_COO_COPIES = 3
# CSR entries per window cell of that Jacobian once duplicates are summed (the cuDSS device copies): Poisson row 5 V
# + n + p, each continuity row 5 of its carrier + 5 V + the other carrier (recombination)
_CSR_ENTRIES_PER_CELL = 29


# Triangle meshes (FEMDriftDiffusionSystem): COO entries of one Newton Jacobian per mesh edge (Poisson 4 + electron 8 +
# hole 8) and per node (charge 2 + recombination 4 + diagonal 1); SuperLU fill of the nodal 3 x 3 block system over the
# scalar 5-point heuristic (fem.diffusion FEM_FILL_FACTOR 2 x the tensor's 3): uncalibrated
_COO_ENTRIES_PER_EDGE = 20
_COO_ENTRIES_PER_NODE = 7
_TRIANGLE_LU_FILL = 6.0
_TRIANGLE_HOST_ARRAYS_PER_NODE = 42      # float64 nodal state, inputs and outputs
_TRIANGLE_HOST_ARRAYS_PER_ELEMENT = 42   # float64 gradients, weights, least-squares map, element outputs
# host_lu_bytes prices complex128 entries; the Jacobian LU is real float64 (factorize_host real=True)
_REAL_LU_SCALE = FLOAT64_CSR_ENTRY_BYTES / COMPLEX128_CSR_ENTRY_BYTES


def _cudss_estimate(context, unknowns, csr_entries, poisson_unknowns, poisson_entries) -> MemoryEstimate:
    """cuDSS 'linear' LUs of the Newton systems (made in the study: run items): the coupled one (unknowns, csr_entries)
    and the cold start's Poisson one (released after it); refresh peaks / solve vectors transient, factors external."""
    mixed = resolved_back_end(context.solvers) == 'cudss_mixed'

    def held_bytes(rows, entries):
        """factorize_host keeps the full equilibrated E (+ the float32 copy when mixed), rhs buffer and row scale; the
        later Newton steps refactorise on the device route and keep its index maps (no transpose map: an LU)."""
        return (operator_copy_bytes(rows, entries, FLOAT64_BYTES, symmetric=False, mixed=mixed)
                + solve_buffer_bytes(rows, FLOAT64_BYTES, mixed) + index_map_bytes(rows, entries, symmetry_check=False))

    def refactor_bytes(rows, entries):
        return refresh_peak_bytes(entries, FLOAT64_BYTES, symmetry_check=False, n_unknowns=rows, mixed=mixed)

    coupled = held_bytes(unknowns, csr_entries)
    # the cold start's Poisson LU is released before the coupled one is planned: its whole life is one phase before
    poisson = held_bytes(poisson_unknowns, poisson_entries) + refactor_bytes(poisson_unknowns, poisson_entries)
    permanent, extra = factor_device_bytes(unknowns, csr_entries, FLOAT64_BYTES, 2, mixed, symmetric=False)
    return MemoryEstimate.from_items(
        run_items=(('cuDSS coupled operator copies, rhs buffer, row scale and index maps', coupled),),
        transient_items=(('Poisson cold start', 'cuDSS Poisson LU (copies, maps, refresh), before the coupled one',
                          phase_before_items(poisson, coupled)),
                         ('refactorisation', 'cuDSS refactorisation per Newton step (refresh or probe peak)',
                          refactor_bytes(unknowns, csr_entries)),
                         (CUDSS_SOLVE_PHASE, 'cuDSS solve vectors (one solve or probe)',
                          solve_peak_bytes(unknowns, FLOAT64_BYTES, mixed))),
        host_items=(('host CSR and its equilibrated copy (float64)', 2 * (FLOAT64_BYTES + 4) * csr_entries),),
        external_items=(('cuDSS coupled LU factor (outside the cupy pool; the Poisson one is freed before it)',
                         permanent),), external_extra=extra)


def _triangle_memory_estimate(mesh, context):
    """FEMDriftDiffusionPhysics: float32 per-element fields on the run backend; host Newton COO arrays, float64
    matrix and SuperLU factor of 3 unknowns per P1 node (cuDSS: _cudss_estimate), nodal and element float64 arrays
    (uncalibrated)."""
    nodes, edges, elements = int(mesh.n_nodes), int(mesh.n_edges), int(mesh.n_elements)
    entries = _COO_ENTRIES_PER_EDGE * edges + _COO_ENTRIES_PER_NODE * nodes
    items = [(f'element fields ({len(SEMICONDUCTOR_FIELD_INFOS)} x float32)',
              len(SEMICONDUCTOR_FIELD_INFOS) * FLOAT32_BYTES * elements)]
    host_items = [('Newton COO arrays', _COO_COPIES * 3 * FLOAT64_BYTES * entries),
                  ('nodal and element arrays (float64)', FLOAT64_BYTES * (_TRIANGLE_HOST_ARRAYS_PER_NODE * nodes
                                                                          + _TRIANGLE_HOST_ARRAYS_PER_ELEMENT * elements))]
    if linear_solver_cudss(context.solvers):   # COO entries bound the coupled CSR; Poisson: the P1 node graph
        return combine((MemoryEstimate.from_items(items, host_items=host_items),
                        _cudss_estimate(context, COMPONENTS * nodes, entries, nodes, 2 * edges + nodes)))
    host_items += _host_factor_items(context, 'SuperLU L + U factors (heuristic, uncalibrated on triangles)',
                                     _TRIANGLE_LU_FILL * _REAL_LU_SCALE * host_lu_bytes(COMPONENTS * nodes),
                                     COMPONENTS * nodes, entries)
    host_items.append(('host sparse matrix (float64)', FLOAT64_CSR_ENTRY_BYTES * entries))
    return MemoryEstimate.from_items(items, host_items=host_items)


def _host_factor_items(context, superlu_label, superlu_bytes, unknowns, entries) -> list:
    """Host items of the coupled Newton LU: SuperLU at its heuristic; a MUMPS / PARDISO back end (the kept factor per
    unknown count) that heuristic x host_direct.superlu_bytes_scale (real LU; float32 values when mixed; uncalibrated)
    + its own copies (host_direct.solver_host_bytes); the cold start's Poisson factor is released before the coupled
    one is factored and stays below it."""
    back_end = host_direct_back_end(context.solvers)
    if back_end is None:
        return [(superlu_label, int(superlu_bytes))]
    label, mixed = factor_label(back_end), host_direct_precision(back_end) == 'mixed'
    return [(f"{label} LU factor ({'float32' if mixed else 'float64'}, SuperLU heuristic scaled, uncalibrated)",
             int(superlu_bytes * superlu_bytes_scale(back_end, symmetric=False, complex_values=False))),
            (f'{label} solver copies (equilibrated operator, row scale; mixed: GMRES vectors)',
             solver_host_bytes(unknowns, entries, complex_values=False, mixed=mixed, symmetric=False))]


def _memory_estimate(settings_values, mesh, context):
    """Device: the float32 full-grid fields (+ the cusolver_qr operator); host: the Newton COO arrays, the float64
    sparse matrix, the SuperLU factor of 3 unknowns per window cell (scipy_splu) and the float64 state arrays; cuDSS:
    _cudss_estimate instead of the SuperLU ones; triangle meshes: _triangle_memory_estimate."""
    if mesh.kind == 'triangles':
        return _triangle_memory_estimate(mesh, context)
    window_cells = window_cell_count(mesh)
    unknowns = COMPONENTS * window_cells
    entries = _COO_ENTRIES_PER_CELL * window_cells
    items = [(f'fields ({len(SEMICONDUCTOR_FIELD_INFOS)} x float32)',
              len(SEMICONDUCTOR_FIELD_INFOS) * FLOAT32_BYTES * int(mesh.n_cells))]
    host_items = [('Newton COO arrays', _COO_COPIES * 3 * FLOAT64_BYTES * entries),
                  ('state, inputs and face arrays (float64)', _SOLVE_HOST_ARRAYS * FLOAT64_BYTES * window_cells)]
    if linear_solver_cudss(context.solvers):   # Poisson: the 5-point window system
        return combine((MemoryEstimate.from_items(items, host_items=host_items),
                        _cudss_estimate(context, unknowns, _CSR_ENTRIES_PER_CELL * window_cells, window_cells,
                                        5 * window_cells)))
    if resolved_back_end(context.solvers) == 'cusolver_qr':
        items.append(('cusolver_qr operator (float64 CSR)', 2 * FLOAT64_CSR_ENTRY_BYTES * entries))
    else:
        # dense 3 x 3 cell blocks: ~9 entries per scalar fill entry of the window grid, i.e. ~3x the scalar heuristic
        # at 3 unknowns per cell
        host_items += _host_factor_items(context, 'SuperLU L + U factors (heuristic)',
                                         3 * _REAL_LU_SCALE * host_lu_bytes(unknowns), unknowns,
                                         _CSR_ENTRIES_PER_CELL * window_cells)
        host_items.append(('host sparse matrix (float64)', FLOAT64_CSR_ENTRY_BYTES * entries))
    return MemoryEstimate.from_items(items, host_items=host_items)


# run-time estimate guesses (calibrate from the summary physics counters newton_iterations / continuation_steps /
# cold_starts): Newton iterations per continuation step (quadratic from the converged neighbour; a warm re-solve after
# a generation change is one step) and of the equilibrium start (nonlinear Poisson from charge neutrality)
NEWTON_ITERATIONS_PER_STEP_GUESS = 6
EQUILIBRIUM_NEWTON_ITERATIONS_GUESS = 10
# host bytes moved per Jacobian COO entry by one Newton iteration (value concatenation, the kept-entry take and slot
# bincount of drift_diffusion._PinnedPattern, L1 equilibration, the CSC conversion of SuperLU; calibrate against
# solve_seconds minus the factor_seconds)
NEWTON_ASSEMBLY_BYTES_PER_ENTRY = 150
_POISSON_COO_ENTRIES_PER_CELL = 9   # 4 per interior face (2 faces per cell) + the cell charge term
# numpy calls (definition_helpers.numpy_calls: host_stream latency each) of one Newton iteration, counted from
# drift_diffusion: coupled = _coupled_system ~200 (bernoulli + 2 derivatives ~35, recombination ~27, 27 block
# broadcasts + 1 concatenation) + the _newton update ~55 + _linear_solve ~15 (pinned-set compare, slot bincount, L1
# equilibration, canonical CSR wrap); Poisson = _poisson_system ~35 + the equilibrium update ~25 + _linear_solve ~15
_COUPLED_NEWTON_NUMPY_CALLS = 270
_POISSON_NEWTON_NUMPY_CALLS = 75
_SOLVE_HOST_ARRAYS = 44             # float64 window arrays of set_inputs / outputs per solve (the memory estimate's)
_MATERIAL_PROPERTY_NAMES = tuple(param.name for param in _MATERIAL_PROPERTIES)


def _ramp_steps(settings_values, context):
    """Continuation steps of a cold solve: the largest |voltage| of the enabled contacts / gates over max_bias_step
    (generation alone ramps in one step); None when a voltage does not resolve to a number."""
    largest = 0.0
    for feature in context.physics_spec.features:
        if not (feature.enabled and feature.type in _HELD_FEATURES):
            continue
        try:
            values = resolve_values(DRIFT_DIFFUSION.feature_type(feature.type).params, feature.params,
                                    context.parameters)
            largest = max(largest, abs(float(values['voltage'])))
        except (ValueError, TypeError):   # reported by validate
            return None
    return max(math.ceil(largest / settings_values['max_bias_step'] - 1e-9), 1)


# Newton systems that plan a cuDSS factor per cold solve (equilibrium Poisson n, coupled 3 n: the role's cudss_plans
# counts both; re-solves refactorise the kept coupled plan)
_PLANNED_SYSTEMS = 2


def _counted(estimate, counter, plan_share):
    """Copy with every item's counter set to `counter` (() = not measured on its own); a cuDSS plan item keeps the
    role's cudss_plans counter x plan_share (the plans of every Newton part), dropped at plan_share 0 (kept plan)."""
    items = []
    for item in estimate.items:
        if item.unit != 'plan':
            items.append(replace(item, counter=counter))
        elif plan_share > 0.0:
            items.append(replace(item, counter_scale=plan_share))
    return replace(estimate, items=tuple(items))


def _newton_part(context, label, cells, components, iterations, limit, per_step, counter, plans=True):
    """Newton iterations (items labelled '<label>: ...') on the system of `components` unknowns per window cell: the
    host Jacobian assembly and one factorisation + solve of the 'linear' role each (a real LU: sparse_direct SuperLU or
    cuDSS, one plan per part unless plans=False: a re-solve on the kept plan)."""
    coupled = components == COMPONENTS
    coo_entries = (_COO_ENTRIES_PER_CELL if coupled else _POISSON_COO_ENTRIES_PER_CELL) * cells
    assembly = te.TimeItem('Jacobian assembly', 'iteration', iterations, 'guess',
                           (te.HostStream(NEWTON_ASSEMBLY_BYTES_PER_ENTRY * coo_entries),
                            numpy_calls(_COUPLED_NEWTON_NUMPY_CALLS if coupled else _POISSON_NEWTON_NUMPY_CALLS)),
                           limit=limit, per_step=per_step, counter=counter)
    # priced on the scalar 5-point host LU curve at 3 n unknowns: the 3 x 3 cell blocks fill ~b^1.5 = 5x more (no
    # block size in te.HostLU yet); cuDSS likewise on its 5-point curves
    value_bytes = FLOAT64_BYTES   # real LU on every back end (cuDSS and SuperLU)
    solver = context.solver_work('linear', unknowns=components * cells, nnz=coo_entries, value_bytes=value_bytes,
                                 dimensions=2,
                                 solves=iterations, solves_kind='guess', factorizations=iterations,
                                 factorizations_kind='guess', per_step=per_step,
                                 backend='cpu')   # factorize_host on either backend
    plan_share = 1.0 / _PLANNED_SYSTEMS if plans else 0.0
    return te.combine((te.TimeEstimate((assembly,)), _counted(solver, counter, plan_share))).labelled(label)


def _time_estimate(settings_values, mesh, context):
    """Host Newton: per iteration the Jacobian assembly and one factorisation + solve of the 'linear' role. Cold solve =
    the equilibrium Poisson Newton + the voltage ramp (continuation steps of max_bias_step); a re-solve (coupled
    generation, state-dependent material) = one warm continuation step, per step of a time_dependent study. Triangle
    meshes: not estimated yet."""
    if mesh.kind == 'triangles':
        return te.TimeEstimate(not_estimated=('drift_diffusion on a triangle mesh',))
    cells = float(context.window_cells())
    newton_limit = int(settings_values['max_iterations'])
    ramp = _ramp_steps(settings_values, context) or 1
    driven = coupling_target(context) or material_follows_state(context, _MATERIAL_PROPERTY_NAMES)
    per_step = context.time_dependent and driven
    re_solves = 1.0 if per_step else (context.static_solves()[0] - 1.0 if driven else 0.0)
    newton_counter = ('solvers', context.name, 'counters', 'newton_iterations')
    cold = ramp * NEWTON_ITERATIONS_PER_STEP_GUESS
    io_ops = (te.HostStream(2 * _SOLVE_HOST_ARRAYS * FLOAT64_BYTES * cells),
              te.Transfer((len(SEMICONDUCTOR_FIELD_INFOS) + len(SEMICONDUCTOR_SCHEMA)) * FLOAT32_BYTES * cells),
              te.PythonCalls(150))
    parts = [_newton_part(context, 'equilibrium Newton', cells, 1, float(EQUILIBRIUM_NEWTON_ITERATIONS_GUESS),
                          newton_limit, False, ())]
    if per_step:   # newton_iterations then counts the cold and the per-step iterations together: no counter
        parts += [_newton_part(context, 'Newton ramp', cells, COMPONENTS, float(cold), ramp * newton_limit, False, ()),
                  _newton_part(context, 'Newton re-solve', cells, COMPONENTS, float(NEWTON_ITERATIONS_PER_STEP_GUESS),
                               newton_limit, True, (), plans=False),
                  te.TimeEstimate((te.TimeItem('inputs and outputs', 'solve', 1.0, 'bounded', io_ops, per_step=True),))]
    else:
        iterations = cold + re_solves * NEWTON_ITERATIONS_PER_STEP_GUESS
        parts += [_newton_part(context, 'Newton', cells, COMPONENTS, float(iterations),
                               (ramp + re_solves) * newton_limit, False, newton_counter),
                  te.TimeEstimate((te.TimeItem('inputs and outputs', 'solve', 1.0 + re_solves,
                                               'exact' if re_solves == 0 else 'guess', io_ops),))]
    return te.combine(parts)


def _mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
    """GUI text of the discretisation on mesh_kind (no extra problems: validation covers the linear solver)."""
    if mesh_kind == 'triangles':
        return ('Control volumes around the corners of the triangles, with exponentially fitted currents along every '
                'triangle edge (linear elements; curved outlines become straight segments between nodes). Every '
                'setting and feature works here; the linear solver must be the host direct solver or GPU cuDSS.', [])
    return 'Finite volumes on rectangular cells, with exponentially fitted currents across every cell face.', []


# ---- texts ----
_DESCRIPTION = (
    '2D stationary drift-diffusion per unit z length, or of a body of revolution (setting geometry axisymmetric: x = '
    'r from the x_lo window edge, whole-ring contact currents and gate charges), on the physical window (finite '
    'volumes): Poisson -div(eps0 eps_dc '
    'grad psi) = q (p - n + N_D - N_A) and the steady electron / hole continuity div J_n = q (R - G), div J_p = q (G - '
    'R) with Scharfetter-Gummel face currents (exact for an exponential carrier profile between cell centres; D = mu '
    'V_T, Boltzmann statistics), solved fully coupled by Newton (sparse direct solves of the unsymmetric Jacobian '
    'by the linear solver role, Automatic = sparse_direct) in scaled variables, potential updates log-damped. psi '
    '(output V) is the electrostatic potential referred to the intrinsic level, with the equilibrium Fermi level at '
    '0 V: in neutral n-type silicon at 1e22 1/m^3 V = V_T asinh(N_D / 2 n_i) = +0.36 V, at an ohmic contact V = that '
    'value plus the contact voltage. Cells with n_intrinsic > 0 are semiconductor; the others are insulators (Poisson '
    'only, eps_dc, no carriers, e.g. an oxide or the air). Outer faces are insulating (zero normal D and current). '
    'Doping comes from doping features (they add), contacts and gates from ohmic_contact / gate features (at least '
    'one ohmic contact); recombination is SRH with a mid-gap trap (lifetime_n / _p) plus radiative (B) plus Auger '
    '(auger_recombination features: (C_n n + C_p p) (n p - n_i^2)); generation '
    'is the sum of the generation features and the coupled slot generation (em_semiconductor_generation: G = '
    'absorbed optical power / photon energy). Each solve continues from the last solution, ramping every applied '
    'voltage (at most max_bias_step per step) and the generation, the first one from equilibrium (nonlinear Poisson '
    'at 0 V); a step that fails is halved, and a solve that cannot converge raises. Outputs: V, N, P, E, the '
    'Scharfetter-Gummel currents J (= Jn + Jp; face currents averaged to the cell centres, outer faces counting 0), '
    'the quasi-Fermi potentials phi_n = V - V_T ln(N / n_i), phi_p = V + V_T ln(P / n_i), net doping, space charge '
    'q (P - N + N_D - N_A) and the net recombination R (0 in contacts and insulators), all copied edge-extended into '
    'an EM PML ring. The contact currents are exact face sums (table results); the electron currents into all '
    'contacts sum to q times the integral of G - R. Coupling to optics: semiconductor_em_plasma_dispersion (free '
    'carriers -> EM index, an E13 index contributor) and em_semiconductor_generation (EM absorption -> G); a '
    'stationary study iterates them with the EM solve. LIMITS: homojunctions (no band offsets, no heterojunctions), '
    'constant mobilities (an expression may make them field- or state-dependent), no '
    'Fermi-Dirac statistics or incomplete ionisation, no transient (the time-dependent study re-solves the steady '
    'state when an input changed). The window needs about 1-3 cells per Debye length near junctions (Debye length '
    'sqrt(eps V_T / (q N)): 13 nm at 1e23 1/m^3). The instance is usually named semiconductor.'
    ' On a triangle mesh (MeshSpec kind triangles; planar or axisymmetric) it is the box method on the linear (P1) '
    'nodes of the window elements: each mesh edge carries the Scharfetter-Gummel current with the coefficient sum of '
    'its two elements (cot / 2 edge weight of the linear stiffness times eps or mobility: edge-averaged finite '
    'elements), control volumes are the lumped node shares; curved outlines are faceted at the nodes. Doping, contacts, '
    'gates and generation select elements: a node takes the volume-weighted mean of its semiconductor elements (a pn '
    'junction on element edges has mixed doping at its nodes), an ohmic contact holds every node of its semiconductor '
    'elements, a gate every node of its elements (potential only, an ordinary edge conductance; no current enters); '
    'generation acts on the free semiconductor elements. V, N, P and the quasi-Fermi potentials are nodal fields '
    '(linear across an insulator element next to the semiconductor), E is -grad V per element and J the least-squares '
    "vector of the element's own edge currents; every state slot holds one value per element (centroid means, carriers "
    "0 in insulators). Mesh edges with a negative coefficient (non-Delaunay, e.g. an obtuse triangle at a material "
    "interface) are reported once as a run warning. Linear solver host SuperLU or cuDSS (back_end 'cusolver_qr' needs "
    "a tensor mesh); Automatic is cuDSS on the GPU backend with nvmath-python (a Jacobian cuDSS refuses switches the "
    "run to host SuperLU for good, with a warning), else host SuperLU. A sparse_direct node with back_end 'cudss' / "
    "'cudss_mixed' (GPU backend, nvmath-python) "
    'factorises every Newton Jacobian on the GPU as a real LU, reusing the cuDSS plan (ordering and symbolic phase) '
    'across Newton iterations and continuation steps while the contacts, gates and insulator cells stay the same; '
    'cudss_mixed factorises in single precision with float64 refinement and falls back to double when that misses '
    '1e-12. A sparse_direct node on a MUMPS or PARDISO back end (opt-in, never Automatic) keeps its host real LU the '
    'same way and reuses its analysis; it keeps the host SuperLU size limits and raises on a refusal (no fallback).')
_SLOT_DOCS = {GENERATION_SLOT: ('1/(m^3 s)', 'Coupled electron-hole pair generation rate (field generation, written by '
                                             'em_semiconductor_generation), added to the generation features (field '
                                             'generation_ext); acts on the free semiconductor cells only.')}
_CONTACT_RESULT = (
    "Result 'table' of the latest solve with one row [feature, voltage, current, electron_current, hole_current] "
    "(units '', V, A/m, A/m, A/m; A with geometry axisymmetric) and the same extras: the current per m of depth (the "
    'whole ring when axisymmetric) flowing out of the contact into the device (positive = conventional current '
    'leaving the contact), summed over the Scharfetter-Gummel face currents '
    'between its held cells and their neighbours (float64, from the converged state; on a triangle mesh plus the net '
    "generation in the held nodes' control volumes, i.e. the current through the contact outline).")


def _features():
    return (
        FeatureType('doping', 'Doping', 'domain_condition', 'domain',
                    params=(Param('donors', 'float', 0.0, label='Donor density', unit='1/m^3', min=0.0,
                                  field_dependent=True,
                                  help='Ionised donor density N_D [1/m^3] (1e22 = 1e16 cm^-3); may depend on x, y '
                                       '(e.g. a Gaussian implant profile).',
                                  tip='Density of donor atoms (n-type doping) per cubic metre; 1e22 equals 1e16 '
                                      'per cubic centimetre.'),
                            Param('acceptors', 'float', 0.0, label='Acceptor density', unit='1/m^3', min=0.0,
                                  field_dependent=True,
                                  help='Ionised acceptor density N_A [1/m^3]; may depend on x, y.',
                                  tip='Density of acceptor atoms (p-type doping) per cubic metre.')),
                    summary='Adds donor and / or acceptor doping to the selected domains. Several doping features '
                            'add up.',
                    description='Adds N_D - N_A to the net doping of the selected cells (several features add; fully '
                                'ionised; ignored in insulating cells). A step between two features at a rectangle '
                                'edge is an abrupt junction on a cell edge (triangle meshes: on element edges, the '
                                'nodes there take the mean of both sides).',
                    apply=_apply_doping, grids=_BOTH_GRIDS),
        FeatureType('ohmic_contact', 'Ohmic contact', 'domain_condition', 'domain',
                    params=(Param('voltage', 'float', 0.0, label='Voltage', unit='V', follows_parameters=True,
                                  help='Applied voltage of the contact [V] (the Fermi level of its cells sits at '
                                       '-voltage in energy, so the contact potential is V = voltage + V_T asinh(net '
                                       'doping / 2 n_i)).',
                                  tip='Voltage applied to the contact.'),),
                    result_kind='table',
                    summary='Turns the selected semiconductor cells into an ideal ohmic contact at a fixed voltage and '
                            'reports the current through it, per metre of depth (of the whole ring when '
                            'axisymmetric).',
                    description='Holds every selected semiconductor cell (n_intrinsic > 0; other selected cells stay '
                                'ordinary insulators) at charge-neutral equilibrium with its own doping, shifted by '
                                'the voltage: n p = n_i^2, n - p = N_D - N_A, V = voltage + V_T asinh((N_D - N_A) / '
                                '2 n_i). The held values sit at the cell centres, so the voltage drop starts at the '
                                'centres of the outermost held cells. A later contact or gate takes shared cells '
                                '(validation warns on possible overlaps). At least one ohmic contact is required. '
                                f'{_CONTACT_RESULT}',
                    apply=_apply_contact, result=_contact_result, grids=_BOTH_GRIDS),
        FeatureType('gate', 'Gate', 'domain_condition', 'domain',
                    params=(Param('voltage', 'float', 0.0, label='Voltage', unit='V', follows_parameters=True,
                                  help='Applied gate voltage [V]; a reused sweep build (sweep reuse_build) follows '
                                       'it without rebuilding.', tip='Voltage applied to the gate electrode.'),
                            Param('work_function_difference', 'float', 0.0, label='Work-function difference',
                                  unit='V',
                                  help='phi_ms [V]: gate work function minus the intrinsic-level work function of '
                                       'the semiconductor (electron affinity + E_g / 2 q); the held potential is V = '
                                       'voltage - phi_ms. 0 = a mid-gap gate; n+ polysilicon on silicon ~ -0.56, p+ '
                                       '~ +0.56. Referred to the intrinsic level, not the Fermi level: the flat-band '
                                       'voltage is phi_ms + V_bulk with V_bulk = V_T asinh(net doping / 2 n_i) '
                                       '(p-type silicon at N_A = 1e23 1/m^3: V_bulk = -0.42 V), and the textbook '
                                       'work-function '
                                       'difference (to the Fermi level) is phi_ms + V_bulk.',
                                  tip='Difference between the gate metal and the semiconductor work functions; '
                                      'zero for a mid-gap gate.')),
                    result_kind='table',
                    summary='Turns the selected insulating domains into a gate electrode at a fixed voltage and reports '
                            'its charge, as in a capacitor or MOS modulator.',
                    description='Holds the potential of every selected cell at voltage - work_function_difference (an '
                                'ideal conductor: the value sits on its faces toward the other cells, half-cell '
                                'conductance 2 eps0 eps_dc / dx of the neighbour). The cells must be insulating '
                                '(n_intrinsic 0; run error otherwise): put the gate on an oxide. No current flows '
                                "into a gate. Result 'table' of the latest solve with one row [feature, voltage, "
                                "charge] (units '', V, C/m; C with geometry axisymmetric) and the extras voltage and "
                                'charge: the free charge per m of depth (the whole ring when axisymmetric) on the '
                                'gate, its outward D flux (Gauss).',
                    apply=_apply_gate, result=_gate_result, grids=_BOTH_GRIDS),
        FeatureType('auger_recombination', 'Auger recombination', 'domain_condition', 'domain',
                    params=(Param('electron_coefficient', 'float', 2.8e-43, label='Electron Auger coefficient',
                                  unit='m^6/s', min=0.0,
                                  help='C_n [m^6/s] of R_Auger = (C_n n + C_p p) (n p - n_i^2) (eeh process); '
                                       'silicon 2.8e-43 (2.8e-31 cm^6/s), GaAs ~1e-42. A constant per feature (no '
                                       'x, y dependence): kept in float64 outside the material grid.',
                                  tip='Strength of three-carrier recombination driven by electrons; matters at high '
                                      'carrier densities (heavy doping, strong injection).'),
                            Param('hole_coefficient', 'float', 9.9e-44, label='Hole Auger coefficient',
                                  unit='m^6/s', min=0.0,
                                  help='C_p [m^6/s] (ehh process); silicon 9.9e-44 (9.9e-32 cm^6/s).',
                                  tip='Strength of three-carrier recombination driven by holes.')),
                    summary='Adds Auger (three-carrier) recombination in the selected domains, important in heavily '
                            'doped regions and under strong injection (LEDs, solar cells, forward-biased diodes).',
                    description='Sets the Auger coefficients of the selected semiconductor cells (a later feature '
                                'takes shared cells; unselected cells have none): R_Auger = (C_n N + C_p P) (N P - '
                                'n_i^2) adds to the SRH and radiative recombination (output recombination, the '
                                'continuity rows and the contact currents). The coefficients are float64 constants '
                                '(about 1e-43 m^6/s: a float32 subnormal, so they are no material property). On a '
                                'triangle mesh a node takes the semiconductor-volume mean of its elements.',
                    apply=_apply_auger, grids=_BOTH_GRIDS),
        FeatureType('generation', 'Generation', 'domain_condition', 'domain',
                    params=(Param('rate', 'float', 1e27, label='Generation rate', unit='1/(m^3 s)', min=0.0,
                                  field_dependent=True,
                                  help='Electron-hole pair generation rate [1/(m^3 s)]; may depend on x, y (e.g. '
                                       'G0 * exp(-alpha * (y0 - y)) for light entering from the top). 1 W/m^2 '
                                       'absorbed at 1.55 um is 7.8e18 pairs / (m^2 s).',
                                  tip='Number of electron-hole pairs created per cubic metre and second, for '
                                      'example by absorbed light.'),),
                    summary='Creates electron-hole pairs at a given rate throughout the selected domains, for example '
                            'to mimic absorbed light.',
                    description='Adds the rate to generation_ext on the selected cells (several features add); it '
                                'acts on the free semiconductor cells only. The coupled slot generation (from an EM '
                                'physics) adds to it.',
                    apply=_apply_generation, grids=_BOTH_GRIDS),
    )


_FIELD_INFOS = tuple(SEMICONDUCTOR_FIELD_INFOS.values())
DRIFT_DIFFUSION = PhysicsType(
    name=DRIFT_DIFFUSION_TYPE_NAME,
    label='Semiconductor (drift-diffusion, 2D)',
    default_instance_name='semiconductor',
    family='',
    group='electrical',
    spaces_2d=('planar', 'cross_section', 'axisymmetric'),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    description=_DESCRIPTION,
    summary='Electrons and holes in a semiconductor device in steady state: doping, contacts, junction fields and '
            'currents, light-generated carriers. Can change the refractive index of a wave physics.',
    settings=_SETTINGS,
    features=_features(),
    grids=_BOTH_GRIDS,   # triangles: FEMDriftDiffusionPhysics (solver_fem.py)
    mesh_method=_mesh_method,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=SEMICONDUCTOR_SCHEMA,
    state_fields=_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    source_slots=_SLOT_DOCS,
    supported_studies=('stationary', 'time_dependent'),
    default_study={'type': 'stationary'},
    uses_dispersion=False,
    create=_create,
    validate=_validate_physics,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALAR_OUTPUTS_HELP,
    solver_roles={'linear': CUDSS_LINEAR_ROLE},
    backends=('gpu', 'cpu'),   # host float64 system; output fields on the Domain's xp; its EM couplings stay gpu
)


def _selftest_cases():
    """The drift_diffusion SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_backend import BACKEND_SELFTEST_CASES
    from .selftests_cudss import CUDSS_SELFTEST_CASES
    from .selftests_fem import FEM_SELFTEST_CASES
    return BACKEND_SELFTEST_CASES + CUDSS_SELFTEST_CASES + FEM_SELFTEST_CASES


def register(registry):
    """The drift_diffusion type, then its template and the selftest loader."""
    registry.register_physics(DRIFT_DIFFUSION)
    registry.register_template('silicon_pn_photodiode', PN_DIODE_LABEL, PN_DIODE_DESCRIPTION,
                               silicon_pn_photodiode_spec, summary=PN_DIODE_SUMMARY)
    registry.register_selftests(_selftest_cases)

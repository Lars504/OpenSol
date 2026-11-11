"""Shared pieces of the diffusion-group profilers (heat, potential, magnetic, drift_diffusion): the scaling options
popped from the command line, the spec edit and the part lists of the study loop, the tensor DiffusionSystem and the
FEM diffusion engines (triangles / tetrahedra)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402,F401


def methods(target_class, names, prefix):
    """(target, label) parts of several methods of one 'module:Class'."""
    return tuple((f'{target_class}.{name}', f'{prefix}: {name}') for name in names)


def functions(module, names, prefix):
    """(target, label) parts of several module-level functions."""
    return tuple((f'{module}:{name}', f'{prefix}: {name}') for name in names)


def pop_option(name, default, kind=float):
    """Value of '--name VALUE' removed from sys.argv (the harness parser never sees it), else default."""
    flag = f'--{name}'
    if flag not in sys.argv:
        return default
    position = sys.argv.index(flag)
    value = kind(sys.argv[position + 1])
    del sys.argv[position:position + 2]
    return value


def scaling_options(kinds, default_kind):
    """(kind name, its settings dict, refine, steps, automatic) from --kind / --refine / --steps / --automatic; the
    kind's defaults fill what is not given."""
    kind = pop_option('kind', default_kind, str)
    if kind not in kinds:
        raise SystemExit(f"--kind must be one of {', '.join(kinds)}, got {kind!r}")
    settings = kinds[kind]
    refine = pop_option('refine', settings.get('refine', 1.0))
    steps = pop_option('steps', settings.get('steps', 1.0))
    automatic = bool(pop_option('automatic', settings.get('automatic', 1), int))
    return kind, settings, refine, steps, automatic


def scaled_spec(spec, refine=1.0, steps=1.0, automatic=True):
    """spec with every numeric MeshSpec.dx / refine, every numeric physics time_step / steps (same duration, steps x
    the step count) and, automatic, the selftest exact-CG 'linear' node replaced by the Automatic solver."""
    for mesh in spec.meshes:
        if refine != 1.0 and isinstance(mesh.dx, (int, float)) and not isinstance(mesh.dx, bool):
            mesh.dx = float(mesh.dx) / refine
    for physics in spec.physics:
        time_step = physics.settings.get('time_step')
        if steps != 1.0 and isinstance(time_step, (int, float)) and not isinstance(time_step, bool):
            physics.settings['time_step'] = float(time_step) / steps
    if automatic:
        from open_sol.model.selftest_helpers import EXACT_CG_NAME
        for physics in spec.physics:
            if physics.solvers.get('linear') == EXACT_CG_NAME:
                del physics.solvers['linear']
        spec.solvers = [solver for solver in spec.solvers if solver.name != EXACT_CG_NAME]
    return spec


def run_kind(name, parts, description, kinds, default_kind, note=''):
    """CLI of a profiler with --kind choices {kind: {'case', 'runs', 'refine', 'steps', 'repeat', 'automatic'}}: the
    kind's case through scaled_spec; the harness options (--case, --runs, --repeat, ...) still override."""
    kind, settings, refine, steps, automatic = scaling_options(kinds, default_kind)
    text = (f'kind {kind}: refine {refine:g}, steps x {steps:g}, automatic linear solver {automatic}'
            + (f'; {note}' if note else ''))
    return main(f'{name}_{kind}', parts, description, case=settings['case'], runs=settings.get('runs'),
                spec_edit=lambda spec: scaled_spec(spec, refine, steps, automatic),
                repeat=settings.get('repeat', 1), note=text)


# ---- study loop, transfers, renderers (every diffusion-group profiler) ----
STUDY_LOOP_PARTS = (
    ('open_sol.model.study:_run_built', 'study: _run_built (study type run + records)'),
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record (field reads)'),
    ('open_sol.model.study:_feature_results', 'study: feature results'),
    ('open_sol.model.study:_solver_records', 'study: solver records'),
    ('open_sol.coupling.runner:MultiphysicsRunner.prepare', 'runner: prepare'),
    ('open_sol.coupling.runner:MultiphysicsRunner.advance', 'runner: advance (time stepping)'),
    ('open_sol.coupling.runner:MultiphysicsRunner.flush', 'runner: flush'),
    ('open_sol.core.backend:to_host', 'transfer: core.backend.to_host'),
    ('open_sol.fem.renderer:FEMFieldRenderer.device_field', 'renderer: FEMFieldRenderer.device_field'),
    ('open_sol.fem.renderer3d:LagrangeSampleRenderer.device_field', 'renderer: LagrangeSampleRenderer.device_field'),
    ('open_sol.fem.domain:TriangleDomain.located', 'fem: TriangleDomain.located'),
    ('open_sol.fem.domain3d:TetrahedronDomain.locate', 'fem: TetrahedronDomain.locate'),
    # nonlinear loops of potential (electrolyte Newton), magnetostatics 2D / 3D (saturation)
    ('open_sol.core.damped_iteration:DampedIteration.run', 'damped iteration: run'),
    ('open_sol.core.damped_iteration:AndersonHistory.step', 'damped iteration: Anderson step'),
)
COMMON_PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_LOOP_PARTS

# ---- tensor DiffusionSystem (core.fv_diffusion: heat, electric_currents / electrostatics, magnetostatics 2D / 3D) ----
DIFFUSION_SYSTEM = 'open_sol.core.fv_diffusion:DiffusionSystem'
TENSOR_DIFFUSION_PARTS = methods(DIFFUSION_SYSTEM, (
    'prepare', 'assemble', 'refresh_diagonal', 'refresh_capacity', '_check_anchors', '_label_anchors', 'build_rhs',
    'solve', 'flux_at_centres', 'axis_flux_at_centres', 'cross_flux_at_centres', 'dissipation_density',
    'fixed_region_outflow', 'face_inward_flux', 'flux_jump_indicator', 'update_face_value',
), 'DiffusionSystem') + methods('open_sol.core.fv_diffusion:DiffusionPhysicsBase', (
    '_extend_into_pml', 'error_indicator',
), 'DiffusionPhysicsBase') + functions('open_sol.core.fv_diffusion', (
    'build_anisotropic_laplacian', 'cross_quads',
), 'fv_diffusion') + functions('open_sol.core.fv_diffusion_3d', (
    'build_laplacian', 'anchor_link_graph', 'flux_at_centres', 'face_flows_3d',
), 'fv_diffusion_3d')

# ---- FEM diffusion (fem.diffusion triangles, fem.diffusion3d tetrahedra) + assembly ----
FEM_DIFFUSION_PARTS = methods('open_sol.fem.diffusion:FEMDiffusionSystem', (
    'assemble', '_fixed_values', '_refresh_constant_terms', '_check_anchors', '_factorize', 'solve',
    '_extend_into_ring', 'reaction', 'face_flux', 'inward_flux', 'centroid_values', 'element_integrals',
), 'FEMDiffusionSystem') + methods('open_sol.fem.diffusion:FEMDiffusionPhysicsBase', (
    'nodal_field', 'element_field', '_record_probes',
), 'FEMDiffusionPhysicsBase') + functions('open_sol.fem.diffusion', (
    'floating_region_count', 'ring_dof_locations', 'coefficient_tensor', 'element_means',
), 'fem.diffusion') + methods('open_sol.fem.diffusion3d:FEMDiffusionSystem3D', (
    'assemble', '_factorize', 'solve', '_extend_into_shell', 'reaction', 'face_flux', 'inward_flux', 'value_at',
    'centroid_values', 'element_integrals',
), 'FEMDiffusionSystem3D') + methods('open_sol.fem.diffusion3d:FEMDiffusionPhysicsBase3D', (
    'nodal_field', 'element_field', '_record_probes', '_locate_point',
), 'FEMDiffusionPhysicsBase3D') + functions('open_sol.fem.diffusion3d', (
    'shell_dof_locations', 'coefficient_tensor_3d',
), 'fem.diffusion3d')
FEM_ASSEMBLY_PARTS = functions('open_sol.fem.assembly', (
    'assemble_stiffness', 'assemble_mass', 'assemble_load', 'assemble_edge_mass', 'assemble_edge_load', 'pattern_sum',
    'nodal_average',
), 'fem.assembly') + functions('open_sol.fem.assembly3d', (
    'assemble_lagrange_stiffness_3d', 'assemble_lagrange_mass_3d', 'assemble_lagrange_load_3d',
    'assemble_point_load_3d', 'assemble_lagrange_gradient_load_3d', 'assemble_face_mass_3d', 'assemble_face_load_3d',
    'element_integrals_3d',
), 'fem.assembly3d') + functions('open_sol.fem.dofs', ('lagrange_dofs',), 'fem.dofs') + functions(
    'open_sol.fem.dofs3d', ('lagrange_dofs_3d',), 'fem.dofs3d') + functions(
    'open_sol.fem.sampling', ('evaluate', 'node_values'), 'fem.sampling') + functions(
    'open_sol.fem.sampling3d', ('evaluate_lagrange_3d', 'centroid_values_3d'), 'fem.sampling3d') + functions(
    'open_sol.fem.indicators', ('diffusion_indicator',), 'fem.indicators')

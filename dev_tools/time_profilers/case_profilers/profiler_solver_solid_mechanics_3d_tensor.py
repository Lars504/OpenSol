"""Profile solid_mechanics_3d on tensor bricks (matrix-free PCG, matrix-free shift-invert modes) on the gpu: selftest
runs solid_3d_uniaxial_block_graded (stationary; as is, 4x finer, 4x finer mixed, 4x finer on the Automatic solver),
the solid_3d_multigrid block (4x finer, double and mixed) and bilayer (4x finer, mixed) on pcg preconditioner multigrid,
solid_3d_eigenfrequency_roller_box (eigenfrequency; as is, 3x finer, 3x finer mixed) and
solid_3d_column_z_expressions (BoxValues loads, probes); ~25 s on an RTX 3050 before the Automatic run (the selftest
runs assign the double exact_cg node: ~19.5 s of it). --warmup runs only the small as-is cases (both kernel
precisions compile). Run from any folder:
python profiler_solver_solid_mechanics_3d_tensor.py [--warmup] [--no-cprofile]"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

BLOCK_CASE = 'solid_3d_uniaxial_block_graded'
MODES_CASE = 'solid_3d_eigenfrequency_roller_box'
COLUMN_CASE = 'solid_3d_column_z_expressions'
MULTIGRID_CASE = 'solid_3d_multigrid'   # its block / bilayer runs on pcg preconditioner multigrid
BLOCK_REFINEMENT = 4      # cell widths / 4: ~2e5 cells, ~6e5 dofs (one large static CG solve)
MODES_REFINEMENT = 3      # 27 x 15 x 12 cells, ~1.6e4 free dofs (a PCG solve per Lanczos application)


def _refined(spec, factor):
    """Deep copy of spec with the first mesh's base dx and every refinement dx divided by factor."""
    spec = copy.deepcopy(spec)
    mesh = spec.meshes[0]
    mesh.dx = float(mesh.dx) / factor
    for refinement in mesh.refinements:
        refinement.dx = float(refinement.dx) / factor
    return spec


def _on_automatic(spec):
    """Deep copy of spec with the selftest's exact_cg node (with_exact_cg) dropped, so every role runs Automatic."""
    from open_sol.model.selftest_helpers import EXACT_CG_NAME
    spec = copy.deepcopy(spec)
    spec.solvers = [node for node in spec.solvers if node.name != EXACT_CG_NAME]
    for physics in spec.physics:
        physics.solvers = {role: name for role, name in physics.solvers.items() if name != EXACT_CG_NAME}
    return spec


def _specs():
    """{run: ModelSpec}: the selftest runs (case-prefixed) plus refined copies."""
    block = case_specs(BLOCK_CASE)['run']
    block_mixed = case_specs(BLOCK_CASE, variant='mixed')['run']
    modes = case_specs(MODES_CASE)['lowest']
    modes_mixed = case_specs(MODES_CASE, variant='mixed')['lowest']
    column = case_specs(COLUMN_CASE)['run']
    multigrid = case_specs(MULTIGRID_CASE)
    multigrid_mixed = case_specs(MULTIGRID_CASE, variant='mixed')
    return {'block': block, 'block_fine': _refined(block, BLOCK_REFINEMENT),
            'block_fine_mixed': _refined(block_mixed, BLOCK_REFINEMENT),
            'block_fine_automatic': _on_automatic(_refined(block, BLOCK_REFINEMENT)),
            'block_fine_multigrid': _refined(multigrid['block'], BLOCK_REFINEMENT),
            'block_fine_multigrid_mixed': _refined(multigrid_mixed['block'], BLOCK_REFINEMENT),
            'bilayer_fine_multigrid_mixed': _refined(multigrid_mixed['bilayer'], BLOCK_REFINEMENT),
            'modes': modes, 'modes_fine': _refined(modes, MODES_REFINEMENT),
            'modes_fine_mixed': _refined(modes_mixed, MODES_REFINEMENT), 'column': column}


def _warmup_specs():
    """{run: ModelSpec} of --warmup: the small as-is runs, double and mixed (compiles the float and double brick
    kernels, the multigrid kernels, the mass kernel and the outputs) without the ~40 s of the refined double block."""
    return {'block': case_specs(BLOCK_CASE)['run'], 'block_mixed': case_specs(BLOCK_CASE, variant='mixed')['run'],
            'block_multigrid': case_specs(MULTIGRID_CASE)['block'],
            'block_multigrid_mixed': case_specs(MULTIGRID_CASE, variant='mixed')['block'],
            'modes': case_specs(MODES_CASE)['lowest'], 'column': case_specs(COLUMN_CASE)['run']}


PHYSICS = 'open_sol.physics.mechanics.solver3d:SolidMechanics3DPhysics'
SYSTEM = 'open_sol.physics.mechanics.system3d:ElasticSystem3D'
KERNELS = 'open_sol.physics.mechanics.kernels3d:BrickKernels'
MULTIGRID = 'open_sol.physics.mechanics.multigrid3d:BrickMultigrid'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', 'step', '_assemble', '_element_lame', '_build_rhs', '_solve_system', '_write_outputs',
    'solve_eigenfrequencies', 'select_resonance', 'reaction', 'state', 'add_body_force', 'add_temperature_change',
    'work_counters',
)) + tuple((f'{SYSTEM}.{method}', f'system: {method}') for method in (
    'prepare', 'assemble', '_check_rigid_modes', '_rigid_mode_gram', 'build_rhs', 'solve', 'reaction', 'write_outputs',
    'lowest_modes', 'hold_face', 'add_face_traction',
)) + tuple((f'{KERNELS}.{method}', f'kernels: {method}') for method in (
    'diagonal_blocks', 'reduce_invert_blocks', 'load', 'outputs', 'mass_product',
)) + tuple((f'{MULTIGRID}.{method}', f'multigrid: {method}') for method in (
    'build', '_largest_eigenvalue', '_dense_inverse', 'precondition',
)) + (
    ('open_sol.physics.mechanics.modal:solve_lowest_modes_matrix_free', 'modal: solve_lowest_modes_matrix_free'),
    ('open_sol.physics.mechanics.modal:_ranked_resonances', 'modal: _ranked_resonances'),
    ('open_sol.physics.mechanics.modal:modal_table_result', 'modal: modal_table_result'),
    ('open_sol.physics.mechanics.renderer3d:SolidRenderer3D.device_field', 'renderer: SolidRenderer3D.device_field'),
    ('open_sol.model.study:_run_stationary', 'study: _run_stationary'),
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study_eigenfrequency:_run_eigenfrequency', 'study: _run_eigenfrequency'),
    ('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
    ('open_sol.model.session:Session.field', 'session: Session.field'),
    ('open_sol.model.field_access_tensor:TensorAccess.field', 'field access: TensorAccess.field'),
    ('open_sol.model.field_access_tensor:TensorAccess.display_field', 'field access: TensorAccess.display_field'),
)

if __name__ == '__main__':
    main('solid_mechanics_3d_tensor', PARTS, __doc__, case=None, specs=_specs, warmup_specs=_warmup_specs)

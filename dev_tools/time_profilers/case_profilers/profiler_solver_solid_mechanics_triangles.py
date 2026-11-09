"""Profile solid_mechanics on triangle meshes (P1 / P2 FEM) on the gpu: stationary (Kirsch plate, released initial
stress), eigenfrequency with stress stiffening (lossless and damped), frequency response (161-frequency direct and
adaptive sweeps with probe / reaction records, a shown-fields sweep) and a geometrically nonlinear Newton solve, selftest
specs with refined meshes (~9-11 s gpu total; the selftests themselves run on the cpu backend). Run from any
folder: python profiler_solver_solid_mechanics_triangles.py [--repeat N] [--no-cprofile]"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402


def _scaled(value, factor):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value / factor
    return f'({value}) / {factor}'


def refined(spec, factor):
    """Copy of spec with every mesh size (dx, h_min, refinement / band dx) divided by factor."""
    spec = copy.deepcopy(spec)
    for mesh in spec.meshes:
        mesh.dx = _scaled(mesh.dx, factor)
        mesh.h_min = _scaled(mesh.h_min, factor)
        for refinement in mesh.refinements:
            for name in ('dx', 'dx_x', 'dx_y', 'dx_z'):
                setattr(refinement, name, _scaled(getattr(refinement, name), factor))
        for axis in mesh.axes:
            for band in axis.bands:
                band.dx = _scaled(band.dx, factor)
    return spec


def _specs():
    specs = {'stationary_kirsch': refined(case_specs('solid_fem_kirsch')['plate'], 2)}   # P2, curved hole, ~14k el
    box = case_specs('solid_fem_prestressed_roller_box')   # P2 h 0.25 mm, 3375 elements, as the selftest
    specs['eigen_prestressed'] = box['prestressed']        # K + K_sigma of the static state, eigsh
    specs['eigen_prestressed_lossy'] = box['lossy']         # damped pencil, eigs
    specs['stationary_release'] = box['release']          # initial stress released, probes
    sweep = case_specs('solid_fem_reduced_sweep', runs=['direct', 'adaptive'])   # 161 frequencies, snapshots none
    specs['frequency_direct'] = refined(sweep['direct'], 2)       # one complex factor per frequency, record rows
    specs['frequency_adaptive'] = refined(sweep['adaptive'], 2)   # reduced model (plan_reduced) + rows evaluate
    specs['frequency_shown'] = refined(case_specs('solid_fem_frequency_probe', runs=['fixed'])['fixed'], 2)   # fields
    specs['nonlinear_stretch'] = refined(case_specs('solid_fem_nonlinear_uniform', runs=['stretch'])['stretch'], 16)
    return specs


PHYSICS = 'open_sol.physics.mechanics.solver_fem:SolidMechanicsFEMPhysics'
SYSTEM = 'open_sol.physics.mechanics.system_fem:ElasticFEMSystem'
ELASTICITY = 'open_sol.fem.elasticity'
MODAL = 'open_sol.physics.mechanics.modal'
HARMONIC = 'open_sol.physics.mechanics.frequency_response:HarmonicSystem'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', 'step', '_static_solve', '_nonlinear_state', '_tangent_assembler_2d', 'solve_static_state',
    '_pencil_stiffness', '_initial_stress', '_thermal_rows', '_fixed_stress', 'reaction', 'solve_eigenfrequencies',
    'solve_eigenfrequencies_targets', 'solve_eigenfrequencies_band', '_modal_pencil', 'select_resonance',
    '_stiffness_loss', 'reset_frequency_response', '_harmonic_inputs', '_harmonic_system', '_harmonic_solve',
    '_harmonic_output_rows', 'plan_frequency_response', 'solve_frequency_response', 'publish_frequency_response',
    '_record_harmonic', '_build_harmonic_operators', '_show_harmonic', 'fem_field', 'window_field',
    '_element_stiffness', '_assemble', '_solid_density', '_total_load', '_external_load', '_point_outputs', '_strains',
    '_output_values', '_write_outputs', '_record_probes',
)) + tuple((f'{SYSTEM}.{method}', f'system: {method}') for method in (
    'assemble', 'stiffness', 'fixed_values', 'release_factor', 'solve', 'solve_tangent_reduced', 'solve_last_tangent',
    'accept_solution', 'solve_homogeneous', 'reaction', 'mass', 'geometric_stiffness', '_solve_free',
    '_check_rigid_modes',
)) + tuple((f'{ELASTICITY}:{function}', f'fem: {function}') for function in (
    'assemble_elastic_stiffness', 'assemble_elastic_mass', 'assemble_stress_load', 'assemble_body_load',
    'assemble_edge_traction', 'element_strains', 'assemble_geometric_stiffness', 'initial_stress_force',
    'green_lagrange_strains',
)) + tuple((f'{ELASTICITY}:TangentAssembler.{method}', f'fem: TangentAssembler.{method}') for method in (
    '__init__', 'force', 'force_and_tangent', 'tangent', 'reduced', 'full',
)) + tuple((f'{HARMONIC}.{method}', f'harmonic: {method}') for method in (
    '__init__', '_factorize', 'solve', '_product', '_rows_solve', 'last_solution', 'plan_reduced', 'set_output_rows',
    '_reduced_solve', 'reduced_problem',
)) + tuple((f'{MODAL}:ModalPencil.{method}', f'modal: ModalPencil.{method}') for method in (
    '__init__', '_pencil', 'solve_at', '_shift_solve', '_candidates', '_resonance', 'release',
)) + (
    ('open_sol.fem.assembly:pattern_sum', 'fem: pattern_sum'),
    ('open_sol.fem.assembly:assemble_load', 'fem: assemble_load'),
    ('open_sol.fem.assembly:nodal_average', 'fem: nodal_average'),
    ('open_sol.fem.dofs:lagrange_dofs', 'fem: lagrange_dofs'),
    ('open_sol.fem.reduced_harmonic:ReducedHarmonicModel.train', 'rom: ReducedHarmonicModel.train'),
    ('open_sol.fem.reduced_harmonic:ReducedHarmonicModel.expand', 'rom: ReducedHarmonicModel.expand'),
    ('open_sol.fem.reduced_harmonic:ReducedHarmonicModel.reduced_coordinates', 'rom: reduced_coordinates'),
    ('open_sol.fem.reduced_harmonic:ReducedHarmonicModel.verify', 'rom: ReducedHarmonicModel.verify'),
    ('open_sol.fem.reduced_outputs:ReducedRows.__init__', 'rom: ReducedRows.__init__'),
    ('open_sol.fem.reduced_outputs:ReducedRows.evaluate', 'rom: ReducedRows.evaluate'),
    ('open_sol.physics.mechanics.nonlinear_start:solve_from_start', 'newton: solve_from_start'),
    ('open_sol.physics.mechanics.nonlinear_start:ramp', 'newton: ramp'),
    ('open_sol.physics.mechanics.tangent_update:_solve_tangent', 'newton: _solve_tangent'),
    ('open_sol.physics.mechanics.tangent_update:_full_product', 'newton: _full_product'),
    ('open_sol.core.damped_iteration:DampedIteration.run', 'newton: DampedIteration.run'),
    ('open_sol.physics.mechanics.stiffness:element_stiffness', 'assembly: element_stiffness'),
    ('open_sol.physics.mechanics.stiffness:check_positive_definite', 'assembly: check_positive_definite'),
    ('open_sol.solvers.sparse_direct:DiffusionDirectSolver._factorize_superlu', 'solver: diffusion _factorize_superlu'),
    ('open_sol.solvers.sparse_direct:DevicePencil.__init__', 'solver: DevicePencil.__init__'),
    ('open_sol.solvers.sparse_direct:DevicePencil.combined', 'solver: DevicePencil.combined'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.shift_pencil', 'solver: arpack shift_pencil'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver._factorize_shift', 'solver: arpack _factorize_shift'),
    ('open_sol.solvers.arpack_shift_invert:ShiftPencil.__init__', 'solver: ShiftPencil.__init__'),
    ('open_sol.solvers.arpack_shift_invert:pattern_combination', 'solver: pattern_combination'),
    ('open_sol.model.study:_solve_static_states', 'study: _solve_static_states'),
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record'),
    ('open_sol.model.study:_LiveFeed.update', 'study: live update'),
    ('open_sol.model.study_eigenfrequency:_search_solve', 'study: eigen _search_solve'),
    ('open_sol.model.study_frequency_response:_plan_reduced_sweeps', 'study: _plan_reduced_sweeps'),
    ('open_sol.physics.mechanics.solver_fem:SolidFEMRenderer.device_field', 'render: SolidFEMRenderer.device_field'),
    ('open_sol.physics.mechanics.solver_fem:SolidFEMRenderer.fem_field', 'render: SolidFEMRenderer.fem_field'),
    ('open_sol.model.field_access_triangles:TriangleAccess.field', 'sampling: TriangleAccess.field'),
    ('open_sol.model.field_access_triangles:TriangleAccess.display_field', 'sampling: TriangleAccess.display_field'),
    ('open_sol.model.field_access_triangles:TriangleAccess._field_data', 'sampling: TriangleAccess._field_data'),
    ('open_sol.model.field_access_triangles:TriangleAccess.evaluate', 'sampling: TriangleAccess.evaluate'),
    ('open_sol.model.field_access_triangles:TriangleAccess.values_at', 'sampling: TriangleAccess.values_at'),
)

if __name__ == '__main__':
    main('solid_mechanics_triangles', PARTS, __doc__, case=None, specs=_specs)

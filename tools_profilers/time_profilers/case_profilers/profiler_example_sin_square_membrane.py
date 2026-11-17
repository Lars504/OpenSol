"""Profile the example mechanics/sin_square_membrane_modes_3d.json on the gpu (solid_mechanics_3d, P2 tetrahedra on
swept prisms, stress_stiffening eigenfrequency: static 'linear' cuDSS solve, K_sigma, shift-invert on 'eigen'); the
selftest study took 27.6 s against a 6.6 s estimate. Run from any folder:
python profiler_example_sin_square_membrane.py [--warmup] [--no-cprofile] [--set L_mem=50*um]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import REPOSITORY, SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

EXAMPLE = REPOSITORY / 'examples' / 'mechanics' / 'sin_square_membrane_modes_3d.json'


def _specs():
    """{run: ModelSpec}: the example as the selftest runs it (its own backend 'gpu')."""
    from open_sol import api
    return {'membrane_50um': api.load(EXAMPLE)}


PHYSICS = 'open_sol.physics.mechanics.solver_fem3d:SolidMechanicsFEMPhysics3D'
SYSTEM = 'open_sol.physics.mechanics.system_fem3d:ElasticFEMSystem3D'
PENCIL = 'open_sol.physics.mechanics.modal:ModalPencil'
SHIFT_PENCIL = 'open_sol.solvers.arpack_shift_invert:ShiftPencil'
ARPACK = 'open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver'
CUDSS = 'open_sol.solvers.cudss_factor:CudssFactorization'
DEVICE_PENCIL = 'open_sol.solvers.sparse_direct:DevicePencil'
ELASTICITY = 'open_sol.fem.elasticity3d'
_ROLE_TARGETS = {target for target, _ in SOLVER_ROLE_PARTS}


def _methods(owner, label, names):
    """(target, label) parts of owner's methods, skipping those SOLVER_ROLE_PARTS already time."""
    return tuple((f'{owner}.{name}', f'{label}: {name}') for name in names
                 if f'{owner}.{name}' not in _ROLE_TARGETS)


PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + _methods(PHYSICS, 'physics', (
    'prepare', 'finalize', 'step', '_assemble', '_static_solve', 'solve_static_state', '_static_refused',
    '_geometric_stiffness', '_pencil_stiffness', '_initial_stress', '_total_load', '_external_load',
    '_solid_density', '_stiffness_loss', 'solve_eigenfrequencies', '_modal_pencil', '_evaluated_films',
    'select_resonance', '_write_outputs', '_ensure_outputs', 'lagrange_field', 'state', '_record_probes',
)) + _methods(SYSTEM, 'system', (
    'assemble', '_check_rigid_modes', '_solve_free', 'solve', 'solve_homogeneous', 'mass', 'release_factor',
    'reaction',
)) + _methods(PENCIL, 'modal pencil', (
    '__init__', '_reduced', '_pencil', 'solve_at', '_shift_solve', '_candidates', '_resonance', 'release', '_info',
)) + _methods(SHIFT_PENCIL, 'shift pencil', (
    '__init__', 'device_mass', 'mass_operator', 'operator', 'row_scale',
)) + _methods(ARPACK, 'arpack', (
    'shift_pencil', '_checked_pencil', '_factorize_shift', 'eigs_device', '_host_eigs',
)) + _methods(DEVICE_PENCIL, 'device pencil', (
    '__init__', '_terms_symmetric', 'combined', 'full_values',
)) + _methods(CUDSS, 'cudss', (
    '__init__', 'from_device_values', '_equilibrated', '_plan_solver', '_device_plan',
    '_device_asymmetry', '_upload_values', '_factorize_with_fallbacks', '_factorize_and_probe', '_replace_by_double',
    '_deferred_check_passed', '_probe_errors', '_inner_solve', '_gmres_refine', '_single_solve', 'release',
)) + tuple((f'{ELASTICITY}:{function}', f'fem: {function}') for function in (
    'assemble_elastic_stiffness_3d', 'assemble_elastic_mass_3d', 'assemble_stress_load_3d',
    'assemble_geometric_stiffness_3d', 'assemble_body_load_3d', 'element_strains_3d', 'nodal_average_3d',
)) + (
    ('open_sol.fem.assembly:pattern_sum', 'fem: pattern_sum'),
    ('open_sol.fem.fixed_pattern:FixedPatternCsr.add_elements', 'fem: FixedPatternCsr.add_elements (block scatter)'),
    ('open_sol.solvers.cudss_factor:_DeviceIndexMaps._host_passes', 'cudss: index maps host passes'),
    ('open_sol.solvers.cudss_factor:_DeviceIndexMaps._uploaded', 'cudss: index maps upload'),
    ('open_sol.solvers.cudss_factor:canonical_csr', 'cudss: canonical_csr'),
    ('open_sol.solvers.arpack_shift_invert:pattern_combination', 'arpack: pattern_combination'),
    ('open_sol.fem.sampling3d:evaluate_lagrange_3d', 'fem: evaluate_lagrange_3d'),
    ('open_sol.fem.sampling3d:centroid_values_3d', 'fem: centroid_values_3d'),
    ('open_sol.fem.domain3d:TetrahedronDomain.locate', 'fem: TetrahedronDomain.locate'),
    ('open_sol.fem.renderer3d:LagrangeSampleRenderer.sample', 'renderer: LagrangeSampleRenderer.sample'),
    ('open_sol.physics.mechanics.modal:modal_table_result', 'modal: modal_table_result'),
    ('open_sol.model.study_eigenfrequency:_run_eigenfrequency', 'study: _run_eigenfrequency'),
    ('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),
    ('open_sol.model.study:_solve_static_states', 'study: _solve_static_states'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
)

if __name__ == '__main__':
    main('example_sin_square_membrane', PARTS, __doc__, case=None, specs=_specs)

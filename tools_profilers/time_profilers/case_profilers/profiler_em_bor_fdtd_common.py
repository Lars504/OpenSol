"""Shared part lists of the em BOR / conical / FDTD profilers (profiler_solver_em_*.py): study helpers, Session reads,
runner, transfers, eigen-solve and triangle FEM assembly / sampling, BOR tensor physics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS  # noqa: E402


def methods(owner, label, names):
    """((target, 'label: name'), ...) of method / function names of owner 'module:Class' or 'module'."""
    separator = '.' if ':' in owner else ':'
    return tuple((f'{owner}{separator}{name}', f'{label}: {name}') for name in names)


def take_flag(flag):
    """True (and removed from sys.argv) when the command line holds flag (script options the harness CLI lacks)."""
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


# Study helpers reached through module globals. The StudyType run functions (_run_time_dependent,
# _run_frequency_domain, _run_eigenfrequency, ...) are held by the registry, so wrapping them would time nothing.
COMMON_PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:_run_built', 'study: _run_built (study run after the build)'),
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study:_solve_wavelength_list', 'study: _solve_wavelength_list (per-wavelength loop)'),
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots (resonance / mode loop)'),
    ('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'output: _SnapshotRecorder.record'),
    ('open_sol.model.study:_SnapshotRecorder._field_data', 'output: _SnapshotRecorder._field_data'),
    ('open_sol.model.study:_feature_results', 'output: _feature_results (feature result callbacks)'),
    ('open_sol.model.study:_solver_records', 'output: _solver_records'),
    ('open_sol.model.study:_LiveFeed.update', 'output: _LiveFeed.update'),
    ('open_sol.model.session:Session.field', 'session: Session.field'),
    ('open_sol.model.field_access_tensor:TensorAccess.field', 'session: TensorAccess.field'),
    ('open_sol.model.field_access_tensor:TensorAccess.evaluate', 'session: TensorAccess.evaluate (derived values)'),
    ('open_sol.coupling.runner:MultiphysicsRunner.prepare', 'runner: prepare'),
    ('open_sol.coupling.runner:MultiphysicsRunner.advance', 'runner: advance (time-step chunks)'),
    ('open_sol.coupling.runner:MultiphysicsRunner.flush', 'runner: flush'),
    ('open_sol.core.backend:to_host', 'transfer: core.backend.to_host'),
    ('open_sol.solvers.line_block:LineBlockSolver.set_operator', 'role: LineBlockSolver.set_operator'),
    ('open_sol.solvers.line_block:LineBlockFactorization.solve_host', 'role: LineBlockFactorization.solve_host'),
)

EIGEN_PARTS = methods('open_sol.physics.em.eigenfrequency', 'eigen', (
    'solve_resonances', '_filtered_candidates', 'squared_cayley_roots',
)) + (
    ('open_sol.physics.em.eigenfrequency:_PencilMass.device_times', 'eigen: _PencilMass.device_times'),
)

# triangle FEM host assembly, element bases and point sampling (fem_bor / fem_conical / fem_modes all use them)
FEM_PARTS = methods('open_sol.fem.assembly', 'fem assembly', (
    'assemble_curl_curl', 'assemble_stiffness', 'assemble_mass', 'assemble_vector_mass', 'assemble_gradient_coupling',
    '_assemble', '_tensor_blocks', '_to_csr', 'restrict',
)) + methods('open_sol.fem.elements', 'fem elements', (
    'nedelec_basis', 'lagrange_basis', 'physical_gradients', 'element_geometry',
)) + methods('open_sol.fem.indicators', 'fem sampling', ('nedelec_at', 'lagrange_at')) + (
    ('open_sol.fem.sampling:evaluate', 'fem sampling: evaluate'),
    ('open_sol.model.field_access_triangles:TriangleAccess.fem_field', 'session: TriangleAccess.fem_field'),
    ('open_sol.model.field_access_triangles:TriangleAccess._field_data', 'session: TriangleAccess._field_data'),
)

# em_fdtd_ez / _hz physics, monitors and results (the engine / renderer / TFSF classes are per script)
EM_PHYSICS_PARTS = methods('open_sol.physics.em:EMPhysics', 'em physics', (
    'finalize', '_calibrate_sources', '_check_courant_stability', 'take_absorbed_power', 'materials_changed',
)) + methods('open_sol.physics.em.monitors', 'em monitors', (
    'FluxMonitor.flux', 'NearToFarField.far_field', '_LineSpectra.host',
)) + methods('open_sol.physics.em.results', 'em results', ('flux_result', 'far_field_result', 'dft_result'))

_BOR = 'open_sol.physics.em.solvers.fdfd_bor'
_BOR_OPS = 'open_sol.physics.em.body_of_revolution'
# body_of_revolution helpers shared by em_fdfd_bor and em_fdtd_bor (flux, far field, Yee recovery)
BOR_OPERATOR_PARTS = methods(f'{_BOR_OPS}:BORAssembly', 'bor assembly', (
    '__init__', 'system', '_class_operators', 'order_family', 'curl_parts', 'e_point_means',
)) + methods(_BOR_OPS, 'bor ops', (
    'point_current_rhs', 'yee_fields', 'centre_fields', 'yee_centres', 'line_flux', 'box_flux', 'source_power',
    'order_pair_fields', 'far_field_samples', 'far_field_power', 'order_sum_far_field',
))

# em_fdfd_bor on tensor meshes (BORPhysics): driven and eigen paths
BOR_TENSOR_PARTS = methods(f'{_BOR}:BORPhysics', 'bor physics', (
    'finalize', 'solve_harmonic', 'plan_harmonic_frequencies', 'solve_eigenfrequencies', 'select_resonance',
    '_refresh_system', '_factorize_host', '_batch_factorization', '_solve_line_block', '_solve_linear', '_solve',
    '_solve_order_sum', '_show', '_show_phasors', '_bor_materials', 'yee',
)) + methods(_BOR, 'bor monitors', (
    'BORFluxMonitor.record', 'BORFluxMonitor.record_order',
    'BORNearToFar.record', 'BORNearToFar.record_order', '_OrderFactor.solve_host',
    'bor_probe_result', 'bor_flux_result', 'bor_far_field_result', 'bor_resonance_table_result',
)) + methods('open_sol.physics.em.fdfd_monitors', 'bor monitors', (
    'CentrePhasorProbe.record', 'CentrePhasorProbe.record_order',
)) + methods(_BOR_OPS, 'bor eigen', (
    'solve_bor_resonances', 'order_eigen_solve', 'merge_order_solves',
)) + methods('open_sol.physics.em.bor_line_block', 'bor line block', (
    'order_factorizations', 'driven_factorization', 'wavelength_factorizations',
)) + (
    ('open_sol.physics.em.renderer_bor:BORRenderer.device_field', 'renderer: BORRenderer.device_field'),
) + BOR_OPERATOR_PARTS

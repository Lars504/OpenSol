"""Profile solid_mechanics_3d on tetrahedra (P2 FEM: shift-invert modes on the 'eigen' runtime, complex frequency
response on the 'linear' runtime) on the gpu: solid_fem3d_roller_box_modes runs lowest / shifted (24k free dofs;
lowest also on the mixed cuDSS eigen factor), solid_fem3d_pml_bar_radiation (damped PML pencil) and the elastic
solid_fem3d_frequency_response_bar run as a 30-point sweep on 2x finer cells (snapshot per frequency); ~9-10 s.
--warmup runs only the PML bar (cuDSS load). Run from any folder:
python profiler_solver_solid_mechanics_3d_tetrahedra.py [--warmup] [--no-cprofile]"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

MODES_CASE = 'solid_fem3d_roller_box_modes'
PML_CASE = 'solid_fem3d_pml_bar_radiation'
RESPONSE_CASE = 'solid_fem3d_frequency_response_bar'
SWEEP_POINTS = 30           # frequencies of the response sweep (one complex factor each)
SWEEP_SPAN = 0.1            # +- relative band around the selftest's single elastic frequency
RESPONSE_REFINEMENT = 2     # cell / 2: 4 x 4 x 32 Kuhn cubes, ~2e4 P2 displacement dofs
_MIXED_EIGEN_NODE = 'eigen_mixed_factor'


def _response_sweep():
    """The elastic bar run as a SWEEP_POINTS sweep around its frequency on cells / RESPONSE_REFINEMENT."""
    spec = copy.deepcopy(case_specs(RESPONSE_CASE, runs=['elastic'])['elastic'])
    settings = dict(spec.study.settings)
    centre = float(settings.pop('frequencies')[0])
    settings.update(frequency_start=(1.0 - SWEEP_SPAN) * centre, frequency_stop=(1.0 + SWEEP_SPAN) * centre,
                    n_frequencies=SWEEP_POINTS)
    spec.study.settings = settings
    spec.meshes[0].dx = float(spec.meshes[0].dx) / RESPONSE_REFINEMENT
    return spec


def _mixed_eigen_factor(spec):
    """Deep copy of spec with its 'eigen' role on an arpack_shift_invert node of factorization cudss_mixed (float32
    factor + GMRES refinement per application; Automatic is the double factor)."""
    from open_sol.model.spec import SolverSpec
    spec = copy.deepcopy(spec)
    spec.solvers.append(SolverSpec(_MIXED_EIGEN_NODE, 'arpack_shift_invert', {'factorization': 'cudss_mixed'}))
    spec.physics[0].solvers['eigen'] = _MIXED_EIGEN_NODE
    return spec


def _specs():
    """{run: ModelSpec}: roller-box modes (lowest, also on a mixed eigen factor; shifted), the PML bar and the
    response sweep."""
    modes = case_specs(MODES_CASE, runs=['lowest', 'shifted'])
    pml = case_specs(PML_CASE)
    return {'modes_lowest': modes['lowest'], 'modes_lowest_mixed_factor': _mixed_eigen_factor(modes['lowest']),
            'modes_shifted': modes['shifted'],
            **{f'pml_{name}': spec for name, spec in pml.items()}, 'response_sweep': _response_sweep()}


def _warmup_specs():
    """{run: ModelSpec} of --warmup: the small PML bar runs (cuDSS library load, plan and factor kernels)."""
    return {f'pml_{name}': spec for name, spec in case_specs(PML_CASE).items()}


PHYSICS = 'open_sol.physics.mechanics.solver_fem3d:SolidMechanicsFEMPhysics3D'
SYSTEM = 'open_sol.physics.mechanics.system_fem3d:ElasticFEMSystem3D'
PENCIL = 'open_sol.physics.mechanics.modal:ModalPencil'
HARMONIC = 'open_sol.physics.mechanics.frequency_response:HarmonicSystem'
ELASTICITY = 'open_sol.fem.elasticity3d'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', 'step', '_static_solve', '_assemble', '_pml_shell', '_solid_density',
    'initial_stress_imbalance', 'solve_static_state', '_pencil_stiffness', 'solve_eigenfrequencies', '_modal_pencil',
    '_stiffness_loss', '_evaluated_films', 'select_resonance', 'reset_frequency_response', 'plan_frequency_response',
    'solve_frequency_response', 'publish_frequency_response', '_harmonic_inputs', '_build_harmonic_operators',
    '_harmonic_solve', '_harmonic_output_rows', '_record_harmonic', '_write_complex_outputs', '_total_load',
    '_external_load', '_ensure_outputs', '_write_primary_outputs', '_write_mechanical_outputs',
    '_write_energy_outputs', '_write_electric_outputs', '_point_outputs', '_write_element_field', 'lagrange_field',
    'state', '_record_probes',
)) + tuple((f'{SYSTEM}.{method}', f'system: {method}') for method in (
    'assemble', '_check_rigid_modes', '_solve_free', 'solve', 'mass', 'shell_only_dofs', 'reaction', 'release_factor',
)) + tuple((f'{PENCIL}.{method}', f'modal pencil: {method}') for method in (
    '__init__', '_reduced', '_pencil', 'solve_at', '_shift_solve', '_candidates', '_resonance', 'release', '_info',
)) + tuple((f'{HARMONIC}.{method}', f'harmonic system: {method}') for method in (
    '__init__', '_reduced_free', '_factorize', 'solve', '_product', '_full', '_rows_solve', 'set_output_rows',
    'last_solution',
)) + tuple((f'{ELASTICITY}:{function}', f'fem: {function}') for function in (
    'assemble_elastic_stiffness_3d', 'assemble_elastic_mass_3d', 'assemble_stress_load_3d', 'assemble_body_load_3d',
    'assemble_face_traction_3d', 'element_strains_3d', 'nodal_average_3d',
)) + (
    ('open_sol.fem.assembly:pattern_sum', 'fem: pattern_sum'),
    ('open_sol.fem.sampling3d:evaluate_lagrange_3d', 'fem: evaluate_lagrange_3d'),
    ('open_sol.fem.sampling3d:centroid_values_3d', 'fem: centroid_values_3d'),
    ('open_sol.fem.domain3d:TetrahedronDomain.locate', 'fem: TetrahedronDomain.locate'),
    ('open_sol.fem.renderer3d:LagrangeSampleRenderer.sample', 'renderer: LagrangeSampleRenderer.sample'),
    ('open_sol.fem.renderer3d:LagrangeSampleRenderer.device_field', 'renderer: LagrangeSampleRenderer.device_field'),
    ('open_sol.physics.mechanics.modal:_resonance', 'modal: _resonance'),
    ('open_sol.physics.mechanics.modal:modal_table_result', 'modal: modal_table_result'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.shift_pencil', 'role: shift_pencil'),
    ('open_sol.model.study_eigenfrequency:_run_eigenfrequency', 'study: _run_eigenfrequency'),
    ('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),
    ('open_sol.model.study_frequency_response:_run_frequency_response', 'study: _run_frequency_response'),
    ('open_sol.model.study_frequency_response:_plan_reduced_sweeps', 'study: _plan_reduced_sweeps'),
    ('open_sol.model.study:_solve_static_states', 'study: _solve_static_states'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
    ('open_sol.model.field_access_tetrahedra:TetrahedraAccess._field_data', 'field access: tetrahedra _field_data'),
    ('open_sol.model.field_access_tetrahedra:TetrahedraAccess.segment_profile',
     'field access: tetrahedra segment_profile'),
)

if __name__ == '__main__':
    main('solid_mechanics_3d_tetrahedra', PARTS, __doc__, case=None, specs=_specs, warmup_specs=_warmup_specs)

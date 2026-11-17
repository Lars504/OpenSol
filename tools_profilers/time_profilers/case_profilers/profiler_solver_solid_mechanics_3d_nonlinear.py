"""Profile geometrically nonlinear solid_mechanics_3d on tetrahedra (total-Lagrangian Newton, fixed-pattern tangents)
on the gpu: the solid_fem3d_nonlinear_uniform block made P2 on 62.5 nm cells (~2.6e4 free dofs, device tangents on the
Automatic cuDSS 'linear' back end): stretch, release, stretch with tangent_update every_iteration, a 3-value reuse_build sweep
of the stretch (warm starts), and a host-tangent stretch (scipy_splu, 125 nm cells); ~8-12 s. --warmup runs only the
device stretch on 125 nm cells (cuDSS load). Run from any folder:
python profiler_solver_solid_mechanics_3d_nonlinear.py [--warmup] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

DEVICE_CELL = 0.0625e-6     # m: 16 x 8 x 8 Kuhn cubes, P2
HOST_CELL = 0.125e-6        # m: 8 x 4 x 4 Kuhn cubes, P2 (host SuperLU tangents)
SWEEP_FRACTIONS = (0.4, 0.7, 1.0)   # x the selftest stretch, one reused build
_HOST_NODE = 'host_direct'


def _nonlinear_spec(run, cell, tangent_update=None, host=False):
    """The selftest's uniform-block spec of run ('stretch' | 'release') as P2 on cell, with tangent_update (None: the
    default when_slow); host: a scipy_splu 'linear' node (host TangentAssembler3D)."""
    from open_sol.model.spec import SolverSpec
    from open_sol.physics.mechanics.selftests_nonlinear3d import _uniform_spec
    spec = _uniform_spec(run)
    physics = spec.physics[0]
    physics.settings.update(discretization='quadratic')
    if tangent_update is not None:
        physics.settings['tangent_update'] = tangent_update
    spec.meshes[0].dx = cell
    if host:
        spec.solvers.append(SolverSpec(_HOST_NODE, 'sparse_direct', {'back_end': 'scipy_splu'}))
        physics.solvers['linear'] = _HOST_NODE
    return spec


def _sweep_spec():
    """The device stretch with u_x = parameter 'pull' swept over SWEEP_FRACTIONS on one reused build."""
    from open_sol.model.spec import ParameterSpec, SweepSpec
    from open_sol.physics.mechanics.selftests_nonlinear3d import _UNIFORM_STRETCH
    spec = _nonlinear_spec('stretch', DEVICE_CELL)
    spec.parameters.append(ParameterSpec('pull', _UNIFORM_STRETCH))
    spec.physics[0].feature('pull').params['u_x'] = 'pull'
    spec.study.sweep = SweepSpec('pull', values=[fraction * _UNIFORM_STRETCH for fraction in SWEEP_FRACTIONS],
                                 reuse_build=True)
    return spec


def _specs():
    return {'stretch': _nonlinear_spec('stretch', DEVICE_CELL), 'release': _nonlinear_spec('release', DEVICE_CELL),
            'stretch_full_newton': _nonlinear_spec('stretch', DEVICE_CELL, tangent_update='every_iteration'),
            'stretch_sweep': _sweep_spec(), 'stretch_host': _nonlinear_spec('stretch', HOST_CELL, host=True)}


def _warmup_specs():
    """{run: ModelSpec} of --warmup: the device stretch on the coarse host cells (cuDSS load, tangent kernels)."""
    return {'stretch_coarse': _nonlinear_spec('stretch', HOST_CELL)}


PHYSICS = 'open_sol.physics.mechanics.solver_fem3d:SolidMechanicsFEMPhysics3D'
SYSTEM = 'open_sol.physics.mechanics.system_fem3d:ElasticFEMSystem3D'
ASSEMBLER = 'open_sol.fem.elasticity3d:TangentAssembler3D'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', 'step', '_static_solve', '_assemble', 'initial_stress_imbalance', '_nonlinear_state',
    '_tangent_assembler_3d', '_fixed_stress', '_external_load', '_total_load', '_initial_stress', 'warm_start_fields',
    'seed_warm_start', 'reset_records', 'set_held_values', '_ensure_outputs', '_write_primary_outputs',
    '_write_outputs', '_write_mechanical_outputs', '_write_energy_outputs', '_point_values', '_record_probes',
    'reaction', 'state',
)) + tuple((f'{SYSTEM}.{method}', f'system: {method}') for method in (
    'assemble', '_check_rigid_modes', 'solve_tangent_reduced', 'solve_tangent_device', 'solve_last_tangent',
    'accept_solution', 'fixed_values', 'reaction', 'release_factor',
)) + tuple((f'{ASSEMBLER}.{method}', f'tangent assembler: {method}') for method in (
    '__init__', '_chunk_geometry', 'force', 'force_and_tangent', 'full', 'reduced', 'reduced_values', 'full_product',
    'tangent',
)) + (
    ('open_sol.fem.fixed_pattern:FixedPatternCsr.__init__', 'fixed pattern: FixedPatternCsr.__init__'),
    ('open_sol.fem.fixed_pattern:FixedPatternCsr.reduced', 'fixed pattern: FixedPatternCsr.reduced'),
    ('open_sol.fem.fixed_pattern:DevicePatternCsr.__init__', 'fixed pattern: DevicePatternCsr.__init__'),
    ('open_sol.fem.elasticity3d:green_lagrange_strains_3d', 'fem: green_lagrange_strains_3d'),
    ('open_sol.fem.elasticity3d:assemble_elastic_stiffness_3d', 'fem: assemble_elastic_stiffness_3d'),
    ('open_sol.fem.elasticity3d:nodal_average_3d', 'fem: nodal_average_3d'),
    ('open_sol.fem.sampling3d:evaluate_lagrange_3d', 'fem: evaluate_lagrange_3d'),
    ('open_sol.physics.mechanics.tangent_update:_solve_tangent', 'newton: _solve_tangent'),
    ('open_sol.physics.mechanics.tangent_update:_full_product', 'newton: _full_product (held-dof lift)'),
    ('open_sol.physics.mechanics.tangent_update:TangentPolicy.reuse', 'newton: TangentPolicy.reuse'),
    ('open_sol.physics.mechanics.nonlinear_start:solve_from_start', 'newton: solve_from_start'),
    ('open_sol.physics.mechanics.nonlinear_start:ramp', 'newton: ramp (load steps)'),
    ('open_sol.physics.mechanics.nonlinear_start:NonlinearStart.candidate', 'newton: NonlinearStart.candidate'),
    ('open_sol.core.damped_iteration:DampedIteration.run', 'newton: DampedIteration.run'),
    ('open_sol.model.study:_run_stationary', 'study: _run_stationary'),
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study:_run_sweep', 'study: _run_sweep'),
    ('open_sol.model.study:_run_built', 'study: _run_built'),
    ('open_sol.model.study:_warm_start_state', 'study: _warm_start_state'),
    ('open_sol.model.study:_seed_warm_start', 'study: _seed_warm_start'),
    ('open_sol.model.session:Session.set_sweep_value', 'session: set_sweep_value'),
    ('open_sol.model.session:Session.reset_run_records', 'session: reset_run_records'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
)

if __name__ == '__main__':
    main('solid_mechanics_3d_nonlinear', PARTS, __doc__, case=None, specs=_specs, warmup_specs=_warmup_specs)

"""Profile heat on tensor meshes (HeatPhysics on core.fv_diffusion.DiffusionSystem) on the gpu. Kinds: '2d' (default)
the transient slab run of heat_time_dependent_values refined 32x (640 x 64 cells) with 50x the steps (1000 implicit-
Euler steps, Automatic linear solver; first estimate ~10 s); '3d' heat_3d_z_couplings refined 2x, repeated 2x;
'steady' heat_multigrid_outer_graded (multigrid PCG, 1.2 s) repeated 6x. Run from any folder:
python profiler_solver_heat.py [--kind 2d|3d|steady] [--refine R] [--steps F]
[--automatic 0|1] [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import COMMON_PARTS, TENSOR_DIFFUSION_PARTS, methods, run_kind  # noqa: E402

HEAT = 'open_sol.physics.heat.solver:HeatPhysics'
PARTS = COMMON_PARTS + TENSOR_DIFFUSION_PARTS + methods(HEAT, (
    'prepare', 'finalize', 'step', 'rebuild_operator', '_assemble', '_capacity_over_dt', '_sync_rise_from_temperature',
    'heat_flux_magnitude', 'temperature_rise', '_refresh_post_processing',
), 'heat') + (
    ('open_sol.physics.heat.renderer:HeatRenderer.device_field', 'renderer: HeatRenderer.device_field'),
)
KINDS = {
    '2d': {'case': 'heat_time_dependent_values', 'runs': ['slab'], 'refine': 32.0, 'steps': 50.0},
    '3d': {'case': 'heat_3d_z_couplings', 'refine': 2.0, 'repeat': 2},
    'steady': {'case': 'heat_multigrid_outer_graded', 'repeat': 6, 'automatic': 0},
}

if __name__ == '__main__':
    run_kind('heat', PARTS, __doc__, KINDS, '2d')

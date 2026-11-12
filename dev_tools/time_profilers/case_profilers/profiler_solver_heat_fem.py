"""Profile heat on element meshes (FEMHeatPhysics on fem.diffusion, FEMHeatPhysics3D on fem.diffusion3d) on the gpu.
Kinds: 'triangles' (default) the 'slab' run of heat_fem_transient_slab (P2, Automatic solver) refined 4x with 10x the
steps (300 implicit-Euler steps; first estimate ~10 s); 'tetrahedra' the slab of heat_fem3d_transient_slab refined 2x
with 4x the steps (120 steps); 'steady' heat_fem_cpu_gpu run gpu (the quadratic disc heater, cuDSS) refined 2x,
repeated 4x; 'radiation' heat_radiation_slab_triangles (P2, cpu: one re-assembly per radiation re-linearisation, the
ReducedSplit value-gather path) refined 32x, repeated 2x. Run from any folder:
python profiler_solver_heat_fem.py [--kind triangles|tetrahedra|steady|radiation] [--refine R]
[--steps F] [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import (COMMON_PARTS, FEM_ASSEMBLY_PARTS, FEM_DIFFUSION_PARTS, methods,  # noqa: E402
                                      run_kind)

HEAT_METHODS = ('prepare', 'finalize', 'step', 'rebuild_operator', '_assemble', '_publish', '_values_changed',
                '_linearise_radiation')
PARTS = (COMMON_PARTS + FEM_DIFFUSION_PARTS + FEM_ASSEMBLY_PARTS
         + methods('open_sol.physics.heat.solver_fem:FEMHeatPhysics', HEAT_METHODS + ('fem_field', 'error_indicator'),
                   'heat triangles')
         + methods('open_sol.physics.heat.solver_fem3d:FEMHeatPhysics3D', HEAT_METHODS + ('lagrange_field',),
                   'heat tetrahedra'))
KINDS = {
    'triangles': {'case': 'heat_fem_transient_slab', 'runs': ['slab'], 'refine': 4.0, 'steps': 10.0},
    'tetrahedra': {'case': 'heat_fem3d_transient_slab', 'runs': ['slab'], 'refine': 2.0, 'steps': 4.0},
    'steady': {'case': 'heat_fem_cpu_gpu', 'runs': ['gpu'], 'refine': 2.0, 'repeat': 4},
    'radiation': {'case': 'heat_radiation_slab_triangles', 'runs': ['run'], 'refine': 32.0, 'repeat': 2},
}

if __name__ == '__main__':
    run_kind('heat_fem', PARTS, __doc__, KINDS, 'triangles')

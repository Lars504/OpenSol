"""Profile magnetic_fields_ac on the gpu backend: MagneticACPhysics (tensor: real DiffusionSystem assembly for the
checks, then the host complex operator ac_operator + circuit rows solved by RecycledFactor) and MagneticACFEMPhysics
(triangles, host complex assembly and solve). Kinds: 'tensor' (default) the gpu runs of magnet_ac_cpu_gpu (skin slab,
solid strip) refined 8x, repeated 2x; 'triangles' magnet_ac_fem_skin_slab refined 2x, repeated 2x; 'strip'
magnet_ac_fem_solid_strip refined 2x, repeated 2x (first estimates ~10 s each). Run from any folder:
python profiler_solver_magnetic_fields_ac.py [--kind tensor|triangles|strip] [--refine R]
[--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import (COMMON_PARTS, FEM_ASSEMBLY_PARTS, TENSOR_DIFFUSION_PARTS,  # noqa: E402
                                      functions, methods, run_kind)

AC_PARTS = methods('open_sol.physics.magnetic.solver_ac:MagneticACPhysics', (
    'prepare', 'step', '_assemble', '_complex_reluctivities', '_conductivity', '_build_host_system',
    '_solve_harmonic', '_update_fields', 'scalars', 'coil_results', 'field_energy',
), 'magnetic_fields_ac tensor') + functions('open_sol.physics.magnetic.ac_operator', (
    'assemble_operator', 'face_conductances', 'boundary_terms', 'face_fluxes_at_centres', 'face_power',
), 'ac_operator') + functions('open_sol.physics.magnetic.circuit', ('bordered_system',), 'circuit') + (
    ('open_sol.physics.magnetic.circuit:RecycledFactor.solve', 'circuit: RecycledFactor.solve'),
    ('open_sol.physics.magnetic.renderer:MagneticRenderer.device_field', 'renderer: MagneticRenderer.device_field'),
    ('open_sol.physics.magnetic.solver:MagneticPhysics._external_ring', 'magnetostatics base: _external_ring'),
)
FEM_AC_PARTS = methods('open_sol.physics.magnetic.solver_ac_fem:MagneticACFEMPhysics', (
    'prepare', 'finalize', 'step', '_complex_reluctivities', '_conductivity', '_point_values', '_build_host',
    '_solve', '_publish', 'scalars', 'coil_results', 'field_energy', 'fem_field', 'error_indicator',
), 'magnetic_fields_ac triangles') + functions('open_sol.fem.magnetostatics', (
    'curl_operator', 'curl_quadrature', 'local_flux_density', 'quadrature_flux_density', 'assemble_curl_tangent',
    'block_mass', 'block_load', 'element_integrals',
), 'fem.magnetostatics')
PARTS = COMMON_PARTS + TENSOR_DIFFUSION_PARTS + AC_PARTS + FEM_AC_PARTS + FEM_ASSEMBLY_PARTS
KINDS = {
    'tensor': {'case': 'magnet_ac_cpu_gpu', 'runs': ['skin_slab_gpu', 'solid_strip_gpu'], 'refine': 8.0,
               'repeat': 2},
    'triangles': {'case': 'magnet_ac_fem_skin_slab', 'refine': 2.0, 'repeat': 2},
    'strip': {'case': 'magnet_ac_fem_solid_strip', 'refine': 2.0, 'repeat': 2},
}

if __name__ == '__main__':
    run_kind('magnetic_fields_ac', PARTS, __doc__, KINDS, 'tensor')

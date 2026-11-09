"""Profile 2D magnetostatics on the gpu: MagneticPhysics (A_z / psi on the tensor DiffusionSystem, saturation by the
Picard / Anderson loop) and MagneticFEMPhysics (host Newton on fem.magnetostatics, triangles). Kinds: 'tensor'
(default) magnet_saturating_sheet refined 8x, repeated 2x; 'triangles' magnet_fem_deep_saturation (Newton with the
energy line search) refined 2x, repeated 2x; 'linear' magnet_coil_inductance refined 8x, repeated 2x (first estimates
~10 s each). Run from any folder:
python profiler_solver_magnetostatics.py [--kind tensor|triangles|linear] [--refine R]
[--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import (COMMON_PARTS, FEM_ASSEMBLY_PARTS, TENSOR_DIFFUSION_PARTS,  # noqa: E402
                                      functions, methods, run_kind)

MAGNETIC = 'open_sol.physics.magnetic.solver:MagneticPhysics'
MAGNETIC_PARTS = methods(MAGNETIC, (
    'prepare', 'finalize', 'step', '_assemble', '_coefficients', '_solve_once', '_refresh_saturation',
    '_solve_saturated', '_saturated_log_reluctivity', '_add_remanence_rhs', '_remanence_face_drives', '_external_ring',
    '_write_potential_field', '_update_fields', 'scalars', 'coil_results', 'field_energy',
), 'magnetostatics tensor') + (
    ('open_sol.physics.magnetic.renderer:MagneticRenderer.device_field', 'renderer: MagneticRenderer.device_field'),
)
FEM_MAGNETIC_PARTS = methods('open_sol.physics.magnetic.solver_fem:MagneticFEMPhysics', (
    'prepare', 'finalize', 'step', '_set_materials', '_publish', 'scalars', 'coil_results', 'field_energy',
    'fem_field', 'error_indicator',
), 'magnetostatics triangles') + methods('open_sol.physics.magnetic.system_fem:MagneticFEMSystem', (
    '_constrain', 'solve', '_loads', 'prepare_solve', '_assemble_linear', '_lifted_solve', '_linear_solve', '_newton',
    '_evaluate', '_energy', '_point_state', '_check_anchors', '_factorize', '_solve_factorized', '_extend_into_ring',
    'constitutive', 'element_means', 'stored_energy', 'region_integrals', 'nodal_fields', 'nodal_flux_density',
), 'MagneticFEMSystem') + functions('open_sol.fem.magnetostatics', (
    'curl_operator', 'curl_quadrature', 'local_flux_density', 'quadrature_flux_density', 'curl_vector',
    'assemble_curl_tangent', 'quadrature_integral', 'block_load', 'element_integrals',
), 'fem.magnetostatics')
PARTS = COMMON_PARTS + TENSOR_DIFFUSION_PARTS + MAGNETIC_PARTS + FEM_MAGNETIC_PARTS + FEM_ASSEMBLY_PARTS
KINDS = {
    'tensor': {'case': 'magnet_saturating_sheet', 'refine': 8.0, 'repeat': 2},
    'triangles': {'case': 'magnet_fem_deep_saturation', 'refine': 2.0, 'repeat': 2},
    'linear': {'case': 'magnet_coil_inductance', 'refine': 8.0, 'repeat': 2},
}

if __name__ == '__main__':
    run_kind('magnetostatics', PARTS, __doc__, KINDS, 'tensor')

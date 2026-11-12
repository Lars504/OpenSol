"""Profile magnetostatics_3d on the gpu: MagneticPhysics3D (reduced scalar potential on the 7-point DiffusionSystem,
saturation loop) and MagneticFEMPhysics3D (fem.diffusion3d, tetrahedra). Kinds: 'tensor' (default)
magnet3d_saturating_slab repeated 3x (3.0 s each); 'tetrahedra' magnet3d_fem_saturating_slab (7.9 s); 'sphere'
magnet3d_vs_axisymmetric (8.6 s, also runs 2D axisymmetric magnetostatics); 'coil' magnet3d_solenoid_axis repeated 3x
(3.5 s). Run from any folder:
python profiler_solver_magnetostatics_3d.py [--kind tensor|tetrahedra|sphere|coil]
[--refine R] [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import (COMMON_PARTS, FEM_ASSEMBLY_PARTS, FEM_DIFFUSION_PARTS,  # noqa: E402
                                      TENSOR_DIFFUSION_PARTS, functions, methods, run_kind)

TENSOR_3D_PARTS = methods('open_sol.physics.magnetic.solver3d:MagneticPhysics3D', (
    'prepare', 'finalize', 'step', '_assemble', '_read_permeability', '_refresh_saturation', '_solve_saturated',
    '_saturated_log_permeability', '_update_corner_anchor', '_external_ring', '_solve_once', '_add_drive_rhs',
    '_drive_terms', '_total_face_flows', '_write_current_density', '_write_potential_field', '_update_fields',
    'scalars', 'coil_results', 'field_energy',
), 'magnetostatics_3d tensor') + functions('open_sol.core.fv_diffusion_3d', (
    'face_drive_integrals_3d', 'face_drive_terms_3d', 'centre_gradient_3d', 'face_drive_divergence_3d',
    'face_terms_at_centres_3d', 'face_flow_energy_3d',
), 'fv_diffusion_3d') + functions('open_sol.physics.magnetic.coils3d', (
    'coil_drive_per_ampere', 'coil_edge_drive_per_ampere',
), 'coils3d') + (
    ('open_sol.physics.magnetic.renderer:MagneticRenderer3D.device_field', 'renderer: MagneticRenderer3D.device_field'),
    ('open_sol.physics.magnetic.renderer:MagneticRenderer.device_field', 'renderer: MagneticRenderer.device_field'),
)
FEM_3D_PARTS = methods('open_sol.physics.magnetic.solver_fem3d:MagneticFEMPhysics3D', (
    'prepare', 'finalize', 'step', '_assemble', '_update_constraints', '_read_permeability', '_host_material',
    '_remanence', '_refresh_saturation', '_solve_saturated', '_saturated_log_permeability', '_coil_dofs',
    '_solve_once', '_drive_load', '_saturating_flux_rms', '_update_fields', '_write_current_density',
    '_record_probes', 'lagrange_field', 'scalars', 'coil_results', 'field_energy',
), 'magnetostatics_3d tetrahedra')
# the saturation loops' kept multigrid hierarchy / direct factor (2D and 3D tensor)
KEPT_PARTS = methods('open_sol.physics.magnetic.kept_preconditioner:KeptPreconditionerSolver', (
    'factorize', 'solve', '_build', '_solve_kept_multigrid', '_solve_kept_direct', '_rebuild_and_solve',
), 'kept preconditioner')
# the 'sphere' kind also runs 2D axisymmetric magnetostatics (MagneticPhysics)
AXISYMMETRIC_PARTS = methods('open_sol.physics.magnetic.solver:MagneticPhysics', (
    'prepare', 'finalize', 'step', '_assemble', '_solve_once', '_update_fields',
), 'magnetostatics 2D')
PARTS = (COMMON_PARTS + TENSOR_DIFFUSION_PARTS + TENSOR_3D_PARTS + FEM_DIFFUSION_PARTS + FEM_ASSEMBLY_PARTS
         + FEM_3D_PARTS + AXISYMMETRIC_PARTS + KEPT_PARTS)
KINDS = {
    'tensor': {'case': 'magnet3d_saturating_slab', 'repeat': 3},
    'tetrahedra': {'case': 'magnet3d_fem_saturating_slab'},
    'sphere': {'case': 'magnet3d_vs_axisymmetric'},
    'coil': {'case': 'magnet3d_solenoid_axis', 'repeat': 3},
}

if __name__ == '__main__':
    run_kind('magnetostatics_3d', PARTS, __doc__, KINDS, 'tensor')

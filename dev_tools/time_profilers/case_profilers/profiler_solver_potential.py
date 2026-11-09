"""Profile electric_currents / electrostatics on the gpu: PotentialPhysics on the tensor DiffusionSystem,
FEMPotentialPhysics (fem.diffusion, triangles) and FEMPotentialPhysics3D (fem.diffusion3d, tetrahedra). Kinds:
'tensor' (default) potential_divider_plates_graded refined 8x, repeated 2x; 'electrolyte' the damped-Newton
Poisson-Boltzmann electrostatics_electrolyte_stern_nonlinear refined 4x, repeated 2x; 'waveform' the driven-terminal
superposition electrostatics_waveform_superposition repeated 4x; 'triangles' electrostatics_fem_coax refined 2x,
repeated 2x; 'tetrahedra' electrostatics_fem3d_plates refined 2x, repeated 2x (first estimates ~10 s each). Run from
any folder:
python profiler_solver_potential.py [--kind tensor|electrolyte|waveform|triangles|tetrahedra]
[--refine R] [--steps F] [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import (COMMON_PARTS, FEM_ASSEMBLY_PARTS, FEM_DIFFUSION_PARTS,  # noqa: E402
                                      TENSOR_DIFFUSION_PARTS, methods, run_kind)

PARTS = (COMMON_PARTS + TENSOR_DIFFUSION_PARTS + FEM_DIFFUSION_PARTS + FEM_ASSEMBLY_PARTS
         + methods('open_sol.physics.potential.solver:PotentialPhysics', (
             'prepare', 'finalize', 'step', '_assemble', '_coefficients', '_solve_nonlinear', '_ionic_terms',
             '_write_ionic_diagonal', '_step_driven', '_solve_bases', '_solve_basis', '_superpose',
             '_update_derived_fields', '_record_probes', 'scalars', 'held_flux', '_region_flux',
         ), 'potential tensor')
         + methods('open_sol.physics.potential.solver_fem:FEMPotentialPhysics', (
             'prepare', 'finalize', 'step', 'rebuild_operator', '_publish', 'scalars', 'held_flux', 'fem_field',
             'ion_concentration', '_ionic_loss', 'error_indicator',
         ), 'potential triangles')
         + methods('open_sol.physics.potential.solver_fem3d:FEMPotentialPhysics3D', (
             'prepare', 'finalize', 'step', 'rebuild_operator', '_publish', 'scalars', 'held_flux', 'lagrange_field',
         ), 'potential tetrahedra')
         + (('open_sol.physics.potential.renderer:_PotentialRenderer.device_field',
             'renderer: _PotentialRenderer.device_field'),))
KINDS = {
    'tensor': {'case': 'potential_divider_plates_graded', 'refine': 8.0, 'repeat': 2},
    'electrolyte': {'case': 'electrostatics_electrolyte_stern_nonlinear', 'refine': 4.0, 'repeat': 2},
    'waveform': {'case': 'electrostatics_waveform_superposition', 'repeat': 4},
    'triangles': {'case': 'electrostatics_fem_coax', 'refine': 2.0, 'repeat': 2},
    'tetrahedra': {'case': 'electrostatics_fem3d_plates', 'refine': 2.0, 'repeat': 2},
}

if __name__ == '__main__':
    run_kind('potential', PARTS, __doc__, KINDS, 'tensor')

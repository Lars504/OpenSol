"""Profile em_fdfd_conical on triangles, eigenfrequency (FEMConicalPhysics, gpu): selftest em_fdfd_conical_fem_pipe (PEC
pipe at k_z, quadratic) refined from h = a / 16 to a / 40 by REFINE (2.7 s recorded at a / 16 on the cpu backend; ~5-10 s
expected). Run (from any folder): python profiler_solver_em_fdfd_conical_triangles.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import COMMON_PARTS, EIGEN_PARTS, FEM_PARTS, methods  # noqa: E402
from profiler_harness import main  # noqa: E402

REFINE = 16.0 / 40.0   # element size factor on the case's h = a / 16

PARTS = COMMON_PARTS + EIGEN_PARTS + FEM_PARTS + methods(
    'open_sol.physics.em.solvers.fdfd_conical_fem:FEMConicalPhysics', 'conical fem physics', (
        'finalize', 'solve_eigenfrequencies', 'select_resonance', 'fem_field', '_materials', '_current_layout',
    )) + methods('open_sol.physics.em.fem_conical', 'conical fem', (
        'fem_conical_layout', 'assemble_fem_conical', 'solve_fem_conical_resonances', 'resonance_field',
        'absorbed_power_means',
    )) + methods('open_sol.physics.em.fem_modes', 'fem_modes', (
        'fem_mode_layout', 'pml_ring', '_mode_samples', '_coefficients', 'sample_permeability', '_sample_fields',
        'element_means', 'nodal_fields',
    ))


def refined(spec):
    """spec with its triangle mesh size scaled by REFINE."""
    mesh = spec.meshes[0]
    mesh.dx = float(mesh.dx) * REFINE
    return spec


if __name__ == '__main__':
    main('em_fdfd_conical_triangles', PARTS, __doc__, case='em_fdfd_conical_fem_pipe', spec_edit=refined,
         note=f'triangle mesh dx x {REFINE:g} of the case')

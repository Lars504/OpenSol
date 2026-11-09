"""Profile em_fdfd_bor on triangles, eigenfrequency (FEMBORPhysics, gpu): selftest em_bor_fem_sphere_model (PEC sphere,
m = 1, quadratic) refined from h = a / 16 to a / 40 by REFINE (1.1 s recorded at a / 16; ~5-10 s expected). Run from any
folder: python profiler_solver_em_fdfd_bor_triangles.py (basiri2019 e8_optics_bor runs the same engine)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import COMMON_PARTS, EIGEN_PARTS, FEM_PARTS, methods  # noqa: E402
from profiler_harness import main  # noqa: E402

REFINE = 16.0 / 40.0   # element size factor on the case's h = a / 16

_FEM_BOR = 'open_sol.physics.em.fem_bor'
PARTS = COMMON_PARTS + EIGEN_PARTS + FEM_PARTS + methods(
    'open_sol.physics.em.solvers.fdfd_bor_fem:FEMBORPhysics', 'bor fem physics', (
        'finalize', 'solve_eigenfrequencies', 'select_resonance', 'fem_field', '_bor_materials', '_current_layout',
    )) + methods(f'{_FEM_BOR}:FEMBORAssembly', 'bor fem assembly', ('__init__', 'system')) + methods(_FEM_BOR, 'bor fem', (
        'fem_bor_layout', 'solve_fem_bor_resonances', 'element_means', 'nodal_fields', '_fields_at', '_metric',
    )) + methods('open_sol.physics.em.body_of_revolution', 'bor eigen', ('order_eigen_solve', 'merge_order_solves'))


def refined(spec):
    """spec with its triangle mesh size scaled by REFINE."""
    mesh = spec.meshes[0]
    mesh.dx = float(mesh.dx) * REFINE
    return spec


if __name__ == '__main__':
    main('em_fdfd_bor_triangles', PARTS, __doc__, case='em_bor_fem_sphere_model', spec_edit=refined,
         note=f'triangle mesh dx x {REFINE:g} of the case')

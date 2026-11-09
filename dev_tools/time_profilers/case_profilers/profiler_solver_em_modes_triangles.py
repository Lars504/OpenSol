"""Profile em_modes on triangle meshes (FEMModePhysics, physics/em/solvers/modes_fem + fem_modes: host assembly, the
eigen role's factor, ARPACK) on the gpu backend: run h80nm of selftest em_modes_fem_rib (TFLN rib half structure,
quadratic, 48k unknowns, 2 modes; 19 s on the cpu backend, 11 s of it the SuperLU factorisation; the gpu backend may
resolve Automatic to cuDSS). The case sets solver_defaults.backend 'cpu'; --backend (default gpu) overrides it. Run
from any folder:
python profiler_solver_em_modes_triangles.py [--no-cprofile] [--runs h40nm]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import MODES_TRIANGLES, STUDY_RUNNERS, TRIANGLE_READS  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TRIANGLE_READS + MODES_TRIANGLES + (
    ('open_sol.fem.mesher:mesh_pslg', 'build: fem.mesher.mesh_pslg (builtin mesher)'),
)

if __name__ == '__main__':
    main('em_modes_triangles', PARTS, __doc__, case='em_modes_fem_rib', runs=['h80nm'])

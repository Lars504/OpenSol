"""Profile em_modes on tensor meshes (ModePhysics, physics/em/solvers/modes_2d + cross_section_modes: host SuperLU +
ARPACK) on the gpu: run heated_gpu of selftest em_modes_cpu_gpu (heat with an exact pcg node, heat_em_thermo_optic into
em_modes, mode_analysis of a silicon strip, 9900 cells; ~25 s, of which ~22 s ARPACK: 2463 operator applications and
one unconverged eigen solve in the last selftest report). Run from any folder:
python profiler_solver_em_modes.py [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import COUPLINGS, MODES_TENSOR, STUDY_RUNNERS, TENSOR_READS  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TENSOR_READS + COUPLINGS + MODES_TENSOR

if __name__ == '__main__':
    main('em_modes', PARTS, __doc__, case='em_modes_cpu_gpu', runs=['heated_gpu'])

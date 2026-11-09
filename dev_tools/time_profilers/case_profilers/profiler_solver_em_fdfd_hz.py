"""Profile em_fdfd_hz on tensor meshes (FDFDPhysics, physics/em/solvers/fdfd_2d) on the gpu: the hz gpu runs of
selftest em_fdfd_cpu_gpu (far-field box with the electric dipole, Fresnel strip, mode-port slab at two wavelengths,
TFSF Drude cylinder, pmc / symmetric wall strips, eigenfrequency cavity, graded Fresnel strip, rotated eps_xy slab,
graphene sheet; ~0.8 - 1.6 s each, ~10 s in all). Run from any folder:
python profiler_solver_em_fdfd_hz.py [--repeat N] [--no-cprofile] [--runs NAME ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import FDFD_TENSOR, STUDY_RUNNERS, TENSOR_READS  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

RUNS = ['far_field__hz__gpu', 'fresnel__hz_glass__gpu', 'slab__hz__gpu', 'drude__hz__gpu', 'wall__hz_pmc__gpu',
        'wall__hz_symmetric__gpu', 'cavity__hz__gpu', 'graded__hz_glass__gpu', 'eps_xy__x_rotated__gpu',
        'sheet__hz_device__gpu']
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TENSOR_READS + FDFD_TENSOR

if __name__ == '__main__':
    main('em_fdfd_hz', PARTS, __doc__, case='em_fdfd_cpu_gpu', runs=RUNS)

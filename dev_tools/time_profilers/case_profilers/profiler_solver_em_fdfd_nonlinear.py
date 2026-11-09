"""Profile em_fdfd_nonlinear (NonlinearHarmonicPhysics, physics/nlfd: harmonic channels on tensor FDFDPhysics) on the
gpu: the gpu runs of selftest nlfd_cpu_gpu (undepleted SHG, mixed polarisation, Kerr by Picard and by Newton-Krylov,
surface chi2, nonlinear sheets, hz cascade adjoint gradient, 3-iteration SHG optimisation; ~0.8 - 1.6 s each, ~11 s in
all). Run from any folder:
python profiler_solver_em_fdfd_nonlinear.py [--repeat N] [--no-cprofile] [--runs NAME ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import FDFD_TENSOR, NLFD_PARTS, STUDY_RUNNERS, TENSOR_READS  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

RUNS = ['shg__run__gpu', 'mixed__run__gpu', 'kerr__picard__gpu', 'kerr__newton__gpu', 'surface__run__gpu',
        'sheet__ez_pump__gpu', 'sheet__tmd_hz__gpu', 'adjoint__hz_gradient__gpu', 'optimization__run__gpu']
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TENSOR_READS + NLFD_PARTS + FDFD_TENSOR

if __name__ == '__main__':
    main('em_fdfd_nonlinear', PARTS, __doc__, case='nlfd_cpu_gpu', runs=RUNS)

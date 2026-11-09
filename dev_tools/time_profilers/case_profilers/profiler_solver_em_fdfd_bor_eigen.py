"""Profile em_fdfd_bor on tensor meshes, eigenfrequency (BORPhysics, gpu): selftest em_bor_order_list_model, runs
'superlu' (Automatic eigen factor: may resolve to cuDSS on the gpu) and 'line_block' (shifted role), PEC cylinder, orders
0..2 (~20 s recorded for both: large for a 25 x 25 grid, worth the look). Run from any folder:
python profiler_solver_em_fdfd_bor_eigen.py [--runs superlu]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import BOR_TENSOR_PARTS, COMMON_PARTS, EIGEN_PARTS  # noqa: E402
from profiler_harness import main  # noqa: E402

PARTS = COMMON_PARTS + EIGEN_PARTS + BOR_TENSOR_PARTS

if __name__ == '__main__':
    main('em_fdfd_bor_eigen', PARTS, __doc__, case='em_bor_order_list_model')

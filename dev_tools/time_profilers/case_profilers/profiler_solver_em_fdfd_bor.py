"""Profile em_fdfd_bor on tensor meshes, driven solves (BORPhysics, gpu): selftest em_bor_cudss_driven, 6 runs (a 3-
wavelength m = 0 batch and the off-axis order sum m = 0..3, each by host SuperLU, cudss and cudss_mixed; ~9 s recorded).
Run (from any folder): python profiler_solver_em_fdfd_bor.py [--runs NAME ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import BOR_TENSOR_PARTS, COMMON_PARTS  # noqa: E402
from profiler_harness import main  # noqa: E402

PARTS = COMMON_PARTS + BOR_TENSOR_PARTS

if __name__ == '__main__':
    main('em_fdfd_bor', PARTS, __doc__, case='em_bor_cudss_driven',
         note='runs batch_* = frequency_domain m = 0, orders_* = stationary order sum m = 0..3')

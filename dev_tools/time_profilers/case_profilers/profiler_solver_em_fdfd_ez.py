"""Profile em_fdfd_ez on tensor meshes (FDFDPhysics, physics/em/solvers/fdfd_2d) on the gpu: the ez gpu runs of
selftest em_fdfd_cpu_gpu (far-field box, Fresnel strip, mode-port slab at two wavelengths, TFSF Drude cylinder, pec
wall strip, eigenfrequency cavities, the Joule / thermo-optic fixed point, conducting sheet, odd mirror half; ~0.8 -
1.7 s each, ~12 s in all). Run from any folder:
python profiler_solver_em_fdfd_ez.py [--repeat N] [--no-cprofile] [--runs NAME ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import COUPLINGS, FDFD_TENSOR, STUDY_RUNNERS, TENSOR_READS  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, main  # noqa: E402

RUNS = ['far_field__ez__gpu', 'fresnel__ez_glass__gpu', 'slab__ez__gpu', 'drude__ez__gpu', 'wall__ez_pec__gpu',
        'wall__ez_pec_domain__gpu', 'cavity__ez__gpu', 'cavity__ez_lossy__gpu', 'joule__joule__gpu',
        'sheet__ez_device__gpu', 'mirror__ez_odd_half__gpu']
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TENSOR_READS + COUPLINGS + FDFD_TENSOR

if __name__ == '__main__':
    main('em_fdfd_ez', PARTS, __doc__, case='em_fdfd_cpu_gpu', runs=RUNS)

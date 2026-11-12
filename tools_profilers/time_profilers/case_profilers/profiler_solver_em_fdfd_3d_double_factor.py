"""Profile em_fdfd_3d on the Yee tensor grid on the double cuDSS factor (gpu): the example fdfd_mie_sphere_resonance_3d
with every arpack_shift_invert node on factorization cudss (one solve per application, no mixed refinement); parts of
profiler_solver_em_fdfd_3d.py, whose mie run is the same model on cudss_mixed. Run from any folder:
python profiler_solver_em_fdfd_3d_double_factor.py [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import main  # noqa: E402
from profiler_solver_em_fdfd_3d import NOTE, PARTS, double_factor, load_example  # noqa: E402


def _specs():
    """{'mie': the example on the double cuDSS factor}."""
    return {'mie': double_factor(load_example())}


if __name__ == '__main__':
    main('em_fdfd_3d_double_factor', PARTS, __doc__, specs=_specs, note=NOTE)

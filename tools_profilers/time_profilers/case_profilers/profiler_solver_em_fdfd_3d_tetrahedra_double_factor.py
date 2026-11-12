"""Profile em_fdfd_3d on a 'tetrahedra' mesh on the double cuDSS factor (gpu): the example fem_mie_sphere_resonance_3d
with every arpack_shift_invert node on factorization cudss; parts of profiler_solver_em_fdfd_3d_tetrahedra.py, whose mie
run is the same model on cudss_mixed. Run from any folder:
python profiler_solver_em_fdfd_3d_tetrahedra_double_factor.py [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import main  # noqa: E402
from profiler_solver_em_fdfd_3d import double_factor  # noqa: E402
from profiler_solver_em_fdfd_3d_tetrahedra import NOTE, PARTS, _on_gpu, load_example  # noqa: E402


def _specs():
    """{'mie': the example on the double cuDSS factor}."""
    return {'mie': double_factor(load_example())}


if __name__ == '__main__':
    main('em_fdfd_3d_tetrahedra_double_factor', PARTS, __doc__, specs=_specs, spec_edit=_on_gpu, note=NOTE)

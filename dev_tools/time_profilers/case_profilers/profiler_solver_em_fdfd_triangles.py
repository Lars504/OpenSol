"""Profile em_fdfd_ez / em_fdfd_hz on triangle meshes (FEMFDFDPhysics, physics/em/solvers/fdfd_2d_fem + fem_fdfd)
on the gpu backend: selftest em_fdfd_fem_disk_model (ez and hz PEC-disk eigenfrequency runs, quadratic) and
em_fdfd_fem_driven_model (ez point current in a PML ring, stationary), ~3.3 s per pass on the cpu backend, repeated 3x.
Both cases set solver_defaults.backend 'cpu'; the harness overrides it with --backend (default gpu), so Automatic
resolves as on the gpu (cuDSS when nvmath-python is present). Run from any folder:
python profiler_solver_em_fdfd_triangles.py [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import FDFD_TRIANGLES, STUDY_RUNNERS, TRIANGLE_READS  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TRIANGLE_READS + FDFD_TRIANGLES + (
    ('open_sol.fem.mesher:mesh_pslg', 'build: fem.mesher.mesh_pslg (builtin mesher)'),
)


def specs():
    """{run: spec} of both triangle cases (run names prefixed by their case)."""
    runs = {f'disk_{name}': spec for name, spec in case_specs('em_fdfd_fem_disk_model').items()}
    runs.update({f'driven_{name}': spec for name, spec in case_specs('em_fdfd_fem_driven_model').items()})
    return runs


if __name__ == '__main__':
    main('em_fdfd_triangles', PARTS, __doc__, case='em_fdfd_fem_disk_model', specs=specs, repeat=3)

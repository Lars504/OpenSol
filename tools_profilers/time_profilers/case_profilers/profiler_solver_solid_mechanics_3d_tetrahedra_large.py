"""Profile one large P2 elasticity factor on tetrahedra (solid_mechanics_3d, stationary 'linear' role): the quadratic
run of solid_fem3d_uniaxial_block refined --refine x (default 7: 28 x 14 x 14 Kuhn cubes, ~1.4e5 displacement dofs,
nested-dissection fronts ~2.5e3 dofs), the size where a block low-rank factor (mumps_blr) should beat the full-rank
ones; one factorisation and one solve, so the factor dominates. First estimate (not yet run): mumps ~5-15 s, SuperLU
(--scipy) past the 60 s timeout. --warmup runs the unrefined block (numba / library loads). Run from any folder:
python profiler_solver_solid_mechanics_3d_tetrahedra_large.py [--refine R] [--warmup] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import pop_option  # noqa: E402
from profiler_harness import case_specs, main, refined  # noqa: E402
from profiler_solver_solid_mechanics_3d_tetrahedra import PARTS  # noqa: E402

BLOCK_CASE = 'solid_fem3d_uniaxial_block'
BLOCK_RUN = 'quadratic'
DEFAULT_REFINEMENT = 7.0   # cell 0.25 um / 7: (8 R + 1) (4 R + 1)^2 P2 nodes x 3 dofs


def _block(refinement):
    """{run: ModelSpec} of the quadratic uniaxial block on cells / refinement."""
    spec = case_specs(BLOCK_CASE, runs=[BLOCK_RUN])[BLOCK_RUN]
    return {f'block_refine{refinement:g}': refined(spec, refinement) if refinement != 1.0 else spec}


if __name__ == '__main__':
    refinement = pop_option('refine', DEFAULT_REFINEMENT)
    main(f'solid_mechanics_3d_tetrahedra_large_refine{refinement:g}', PARTS, __doc__, case=None,
         specs=lambda: _block(refinement), warmup_specs=lambda: _block(1.0),
         note=f'uniaxial block refined {refinement:g}x')

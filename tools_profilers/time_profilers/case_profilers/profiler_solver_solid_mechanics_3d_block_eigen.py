"""Profile the opt-in block_eigen eigensolver (LOBPCG) of solid_mechanics_3d on tensor bricks, gpu, on the cases the
tensor script's modes runs do not cover: the roller box of solid_3d_eigenfrequency_roller_box 3x finer (3 multigrid
levels) at precision double and with preconditioner jacobi, and 6x finer (~1.3e5 dofs, 4 levels) mixed / double on the
private hierarchy and mixed on the linear pcg's own multigrid (the shared-hierarchy path); production linear settings.
Every run holds its own block_eigen node, so `--solver-type block_eigen` keeps its settings. --warmup runs the
two-level selftest box first (mixed, double, jacobi: every *_rows kernel compiles). PARTS: the tensor script's. Run
from any folder: python profiler_solver_solid_mechanics_3d_block_eigen.py [--warmup] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import case_specs, main, refined  # noqa: E402
from profiler_solver_solid_mechanics_3d_tensor import (  # noqa: E402
    BLOCK_EIGEN_LEVELS_CASE, MODES_CASE, MODES_REFINEMENT, PARTS, _production)

LEVELS_REFINEMENT = 6     # 54 x 30 x 24 cells, ~1.3e5 dofs: four multigrid levels (27 x 15 x 12, 14 x 8 x 6, 7 x 4 x 3)
BLOCK_EIGEN_NODE = 'block_eigen_modes'   # = selftests_eigenfrequency._BLOCK_EIGEN_NODE (the levels case's node)


def _with_block_eigen(spec, **settings):
    """spec (edited in place, returned) with the eigen role on a block_eigen node holding `settings` (the case's node
    replaced)."""
    from open_sol.model.spec import SolverSpec
    spec.solvers = [node for node in spec.solvers if node.name != BLOCK_EIGEN_NODE]
    spec.solvers.append(SolverSpec(BLOCK_EIGEN_NODE, 'block_eigen', dict(settings)))
    spec.physics[0].solvers['eigen'] = BLOCK_EIGEN_NODE
    return spec


def _specs():
    """{run: ModelSpec}: the refined roller boxes, each with its own block_eigen node, linear on production settings."""
    box = case_specs(MODES_CASE)['lowest']
    fine, levels = refined(box, MODES_REFINEMENT), refined(box, LEVELS_REFINEMENT)
    return {'modes_fine_double': _with_block_eigen(_production(fine), precision='double'),
            'modes_fine_jacobi': _with_block_eigen(_production(fine), preconditioner='jacobi'),
            'modes_levels': _with_block_eigen(_production(levels)),
            'modes_levels_double': _with_block_eigen(_production(levels), precision='double'),
            'modes_levels_shared_multigrid': _with_block_eigen(_production(levels, 'multigrid'))}


def _warmup_specs():
    """{run: ModelSpec} of --warmup: the two-level selftest box with block_eigen mixed, double and jacobi."""
    block = case_specs(BLOCK_EIGEN_LEVELS_CASE)['block']
    return {'levels': _production(block),
            'levels_double': _with_block_eigen(_production(block), precision='double'),
            'levels_jacobi': _with_block_eigen(_production(block), preconditioner='jacobi')}


if __name__ == '__main__':
    main('solid_mechanics_3d_block_eigen', PARTS, __doc__, case=None, specs=_specs, warmup_specs=_warmup_specs)

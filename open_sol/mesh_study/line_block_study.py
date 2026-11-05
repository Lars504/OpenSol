"""S7.8 line_block cost vs refined axis (GPU): the FDFD line-source example on uniform, x-refined, y-refined and
both-refined meshes, solved by sparse_direct (Automatic) and line_block -> records/line_block.jsonl."""
from .common import NM, Variant, graded_mesh, load_example, record_file, run_variants, selected_cases, uniform_mesh

CASE = 'line_block'
SOLVERS = ('sparse_direct', 'line_block')   # sparse_direct = the Automatic 'linear' role of em_fdfd_ez
SPAN = ('x_src - 1.2*um', 'x_src + 1.2*um', 'dx_cell')   # refined band around the source (x_src = y_src axis)


def variants(fine):
    y_span = ('y_src - 1.2*um', 'y_src + 1.2*um', 'dx_cell')
    meshes = {'uniform': lambda: uniform_mesh('dx_cell'),
              'x_refined': lambda: graded_mesh(50 * NM, x_bands=[SPAN]),
              'y_refined': lambda: graded_mesh(50 * NM, y_bands=[y_span]),
              'both_refined': lambda: graded_mesh(50 * NM, x_bands=[SPAN], y_bands=[y_span])}
    for dx in [25, 12.5] + ([6.25] if fine else []):
        for family, mesh in meshes.items():
            for solver in SOLVERS:
                yield Variant(family, dx * NM, {'dx_cell': dx * NM}, mesh(),
                              solver_type=None if solver == 'sparse_direct' else solver, solver_physics=('em',),
                              sweep={'solver': solver})


def run(options):
    if selected_cases(options, CASE, (CASE,)):
        run_variants(CASE, load_example('em/analytic_fdfd_line_source_green_ez.json'), list(variants(options.fine)),
                     record_file(options, CASE), options)

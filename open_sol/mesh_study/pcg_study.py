"""S7.8 PCG iterations vs grading (GPU): Jacobi vs multigrid on the strip heater (planar: contrast, max_ratio, aspect),
the disc heater (axisymmetric: contrast, max_ratio, outer grading) and the polar annulus (metric aspect b / a)
-> records/pcg_<case>.jsonl."""
from .common import (MM, NM, Variant, graded_mesh, load_example, load_model, record_file, run_variants,
                     selected_cases, uniform_mesh)

PRECONDITIONERS = ('jacobi', 'multigrid')
CONTRASTS = [1, 2, 4, 8, 16, 32, 64]           # dx_base / dx_fine (1 = uniform fallback)
RATIOS = [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]  # max_ratio at contrast RATIO_SWEEP_CONTRAST
RATIO_SWEEP_CONTRAST = 32
ASPECTS = [1, 2, 4, 8, 16, 32, 64]             # dx_base / dy_fine of the strip heater rows (rows refined only)
OUTER_EXTENTS = [10, 30, 100, 300, 1000]       # R_out / a_heat of the disc heater with outer grading
OUTER_RATIO = 1.3                              # industry-standard default maximum element growth rate
METRIC_ASPECTS = [3, 10, 30, 100, 300]         # b / a of the polar annulus (= dr / (a dphi) at the inner arc)
MAX_ITERATIONS = 200_000                       # explicit: the count is iterations to tolerance, not the auto cap
STRIP_SNAPS = [(name, 'dx_fine', 0.0) for name in ('heater_snap_lo', 'heater_snap_hi', 'core_snap_lo', 'core_snap_hi',
                                                    'oxide_snap')]
STRIP_BOX = [('heater', 'dx_fine', 'ref_margin'), ('core', 'dx_fine', 'ref_margin')] + STRIP_SNAPS
STRIP_ROW_BANDS = [('hy0 - ref_margin', 'hy1 + ref_margin', 'dx_fine'), ('cy0 - ref_margin', 'cy1 + ref_margin', 'dx_fine')]


def _cg(variant_args, preconditioner, physics='heat'):
    """Variant with the exact pcg node (double, 1e-10, MAX_ITERATIONS, preconditioner) on physics' linear role."""
    family, h, parameters, mesh, sweep = variant_args
    return Variant(family, h, parameters, mesh, solver_type='pcg', solver_physics=(physics,),
                   solver_settings={'precision': 'double', 'tolerance': 1e-10, 'max_iterations': MAX_ITERATIONS,
                                    'preconditioner': preconditioner},
                   sweep={**sweep, 'preconditioner': preconditioner})


def strip_heater_variants():
    dx_fine = 25 * NM
    for preconditioner in PRECONDITIONERS:
        for contrast in CONTRASTS:
            yield _cg(('contrast', dx_fine, {'dx_fine': dx_fine, 'dx_base': contrast * dx_fine},
                       graded_mesh('dx_base', STRIP_BOX), {'contrast': contrast, 'max_ratio': 1.15}), preconditioner)
        for ratio in RATIOS:
            yield _cg(('ratio', dx_fine, {'dx_fine': dx_fine, 'dx_base': RATIO_SWEEP_CONTRAST * dx_fine},
                       graded_mesh('dx_base', STRIP_BOX, max_ratio=ratio),
                       {'contrast': RATIO_SWEEP_CONTRAST, 'max_ratio': ratio}), preconditioner)
        for aspect in ASPECTS:   # x stays at dx_base away from the snaps: aspect = dx / dy inside the row bands
            yield _cg(('aspect', dx_fine, {'dx_fine': dx_fine, 'dx_base': aspect * dx_fine},
                       graded_mesh('dx_base', [('oxide_snap', 'dx_fine', 0.0)], y_bands=STRIP_ROW_BANDS),
                       {'aspect': aspect, 'max_ratio': 1.15}), preconditioner)


def disc_heater_variants():
    a_heat = 1 * MM
    dx_fine = a_heat / 40
    box = [('heater', 'dx_mesh', 'ref_margin')]
    for preconditioner in PRECONDITIONERS:
        for contrast in CONTRASTS:
            yield _cg(('contrast', dx_fine, {'dx_mesh': dx_fine, 'dx_base': contrast * dx_fine},
                       graded_mesh('dx_base', box), {'contrast': contrast, 'max_ratio': 1.15}), preconditioner)
        for ratio in RATIOS:
            yield _cg(('ratio', dx_fine, {'dx_mesh': dx_fine, 'dx_base': RATIO_SWEEP_CONTRAST * dx_fine},
                       graded_mesh('dx_base', box, max_ratio=ratio),
                       {'contrast': RATIO_SWEEP_CONTRAST, 'max_ratio': ratio}), preconditioner)
        for extent in OUTER_EXTENTS:   # base a / 10, heater refined to a / 40, cells grow 1.3 x up to R_out / 20
            yield _cg(('outer', dx_fine, {'dx_mesh': dx_fine, 'R_out': f'{extent}*a_heat', 'dx_base': 'R_out/20'},
                       graded_mesh(a_heat / 10, box, outer_dx='R_out/20', outer_ratio=OUTER_RATIO),
                       {'extent': extent, 'outer_ratio': OUTER_RATIO}), preconditioner)


def polar_annulus_variants():
    a_inner = 1 * MM
    for preconditioner in PRECONDITIONERS:
        for aspect in METRIC_ASPECTS:   # 200 radial cells, ~(pi / 2) 200 b / (b - a) angle cells
            yield _cg(('metric', (aspect - 1) * a_inner / 200,
                       {'b_outer': f'{aspect}*a_inner', 'dx_mesh': '(b_outer - a_inner)/200'},
                       uniform_mesh('dx_mesh'), {'b_over_a': aspect}), preconditioner)


CASES = {   # case -> (model loader, variants)
    'strip_heater': (lambda: load_model('strip_heater'), strip_heater_variants),
    'disc_heater': (lambda: load_model('disc_heater'), disc_heater_variants),
    'polar_annulus': (lambda: load_example('heat/polar_annulus_log_profile.json'), polar_annulus_variants),
}


def run(options):
    """Run the cases --case selects (default all)."""
    for case in selected_cases(options, 'pcg', CASES):
        load, variants = CASES[case]
        run_variants(f'pcg_{case}', load(), list(variants()), record_file(options, 'pcg', case), options)

"""S7.8 accuracy per cell (GPU): every mesh variant of the five reference problems -> records/accuracy_<case>.jsonl.
Families: uniform, graded_box (object refinements), anisotropic (S7.1 axis bands), subpixel / conformal (S7.6)."""
from .common import (MM, NM, Variant, graded_mesh, load_example, load_model, record_file, run_variants,
                     selected_cases, uniform_mesh)

SUBPIXEL = {'subpixel_smoothing': True}
CONFORMAL = {'conformal_boundaries': True}


def rib_variants(fine):
    """Uniform dx divides every rib dimension; graded / anisotropic keep a 50 nm base."""
    snaps = [('snap_film', 'dx_fine', 0.0), ('snap_slab', 'dx_fine', 0.0)]
    box = [('rib', 'dx_fine', 'ref_margin')] + snaps
    x_bands = [('x_top_edge - ref_margin', 'x_foot + ref_margin', 'dx_fine')]   # columns over the sidewall
    y_bands = [('t_box - ref_margin', 'y_film + ref_margin', 'dx_fine')]        # rows over the film
    for dx in [50, 25, 20, 12.5, 10, 6.25] + ([5] if fine else []):
        yield Variant('uniform', dx * NM, {'dx_fine': dx * NM}, uniform_mesh('dx_fine'))
    for dx in [25, 20, 12.5, 10] + ([6.25] if fine else []):
        yield Variant('subpixel', dx * NM, {'dx_fine': dx * NM}, uniform_mesh('dx_fine'), SUBPIXEL)
    for dx in [25, 12.5, 10, 6.25] + ([5, 3.125] if fine else []):
        parameters = {'dx_fine': dx * NM, 'dx_base': 50 * NM}
        yield Variant('graded_box', dx * NM, parameters, graded_mesh('dx_base', box))
        anisotropic = graded_mesh('dx_base', snaps, x_bands=x_bands, y_bands=y_bands)
        yield Variant('anisotropic', dx * NM, parameters, anisotropic)
        yield Variant('anisotropic_subpixel', dx * NM, parameters, anisotropic, SUBPIXEL)


def disk_variants(fine):
    box = [('disk', 'dx_fine', 'ref_margin')]
    for dx in [40, 30, 20, 15, 10, 7.5] + ([5] if fine else []):
        yield Variant('uniform', dx * NM, {'dx_fine': dx * NM}, uniform_mesh('dx_fine'))
    for dx in [40, 30, 20, 15, 10] + ([7.5] if fine else []):
        yield Variant('subpixel', dx * NM, {'dx_fine': dx * NM}, uniform_mesh('dx_fine'), SUBPIXEL)
    for dx in [20, 15, 10, 7.5, 5] + ([3.75] if fine else []):
        parameters = {'dx_fine': dx * NM, 'dx_base': 50 * NM}
        yield Variant('graded_box', dx * NM, parameters, graded_mesh('dx_base', box))
        yield Variant('graded_box_subpixel', dx * NM, parameters, graded_mesh('dx_base', box), SUBPIXEL)


def strip_heater_variants(fine):
    """Uniform dx on the 150 nm edge lattice; graded variants on a 600 nm base."""
    snaps = [(name, 'dx_fine', 0.0) for name in ('heater_snap_lo', 'heater_snap_hi', 'core_snap_lo', 'core_snap_hi',
                                                  'oxide_snap')]
    box = [('heater', 'dx_fine', 'ref_margin'), ('core', 'dx_fine', 'ref_margin')] + snaps
    y_bands = [('hy0 - ref_margin', 'hy1 + ref_margin', 'dx_fine'), ('cy0 - ref_margin', 'cy1 + ref_margin', 'dx_fine')]
    x_bands = [(f'{edge} - ref_margin', f'{edge} + ref_margin', 'dx_fine') for edge in ('hx0', 'hx1', 'cx0', 'cx1')]
    for dx in [150, 75, 50, 37.5, 30, 25, 18.75] + ([15] if fine else []):
        yield Variant('uniform', dx * NM, {'dx_fine': dx * NM}, uniform_mesh('dx_fine'))
    for dx in [75, 50, 37.5, 25, 18.75, 12.5] + ([7.5] if fine else []):
        yield Variant('graded_box', dx * NM, {'dx_fine': dx * NM}, graded_mesh('dx_base', box))
        yield Variant('anisotropic', dx * NM, {'dx_fine': dx * NM},
                      graded_mesh('dx_base', snaps, x_bands=x_bands, y_bands=y_bands))


def disc_heater_variants(fine):
    a_heat = 1 * MM
    box = [('heater', 'dx_mesh', 'ref_margin')]
    for n in [10, 20, 40, 80] + ([160] if fine else []):
        parameters = {'dx_mesh': a_heat / n, 'dx_base': a_heat / n}
        yield Variant('uniform', a_heat / n, parameters, uniform_mesh('dx_mesh'))
        yield Variant('conformal', a_heat / n, parameters, uniform_mesh('dx_mesh'), CONFORMAL)
    for n in [10, 20, 40, 80, 160] + ([320] if fine else []):
        parameters = {'dx_mesh': a_heat / n, 'dx_base': a_heat / 2}
        yield Variant('graded_box', a_heat / n, parameters, graded_mesh('dx_base', box))
        yield Variant('graded_box_conformal', a_heat / n, parameters, graded_mesh('dx_base', box), CONFORMAL)


def coax_variants(fine):
    """a / dx = n keeps the axis on a cell vertex (L_box / dx = 10 n). Graded meshes cannot help (both circles refine
    their bounding squares, nearly the whole box): conformal and the quadtree / FEM prototypes are the comparison."""
    a_core = 0.5 * MM
    for n in [5, 10, 20, 40, 80] + ([160] if fine else []):
        yield Variant('uniform', a_core / n, {'dx_cell': a_core / n}, uniform_mesh('dx_cell'))
        yield Variant('conformal', a_core / n, {'dx_cell': a_core / n}, uniform_mesh('dx_cell'), CONFORMAL)


CASES = {   # case -> (model loader, variants)
    'rib': (lambda: load_model('rib_modes'), rib_variants),
    'disk': (lambda: load_model('disk_eigen'), disk_variants),
    'strip_heater': (lambda: load_model('strip_heater'), strip_heater_variants),
    'disc_heater': (lambda: load_model('disc_heater'), disc_heater_variants),
    'coax': (lambda: load_example('potential/coax_capacitor_electrostatics.json'), coax_variants),
}


def run(options):
    """Run the cases --case selects (default all)."""
    for case in selected_cases(options, 'accuracy', CASES):
        load, variants = CASES[case]
        run_variants(case, load(), list(variants(options.fine)), record_file(options, 'accuracy', case), options)

"""Profile em_fdfd_conical on tensor meshes (ConicalPhysics, gpu): the em_fdfd_conical_cudss runs (current sheet,
frequency_domain at 3 wavelengths, by host SuperLU / cudss / cudss_mixed) with the 2-cell y period widened to SHEET_Y_CELLS
cells (SUPERLU_SHEET_Y_CELLS for the host SuperLU reference), plus 'eigen_lattice' = em_fdfd_conical_lattice (Bloch
eigenfrequency, 'auto': cuDSS on the gpu since definition_conical's auto_factorizations, host SuperLU before) with LATTICE_SCALE x more cells per axis. The recorded cases take < 1 s; ~10 s
expected at these sizes (a previous run at y 240 / lattice x 10: 400 s + 211 s in SuperLU). Run from any folder:
python profiler_solver_em_fdfd_conical.py [--runs NAME ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import COMMON_PARTS, EIGEN_PARTS, methods  # noqa: E402
from profiler_harness import case_specs, main  # noqa: E402

SHEET_Y_CELLS = 240          # y period of the cuDSS sheet runs (case: 2 cells; x is 160 cells incl. the cpml)
SUPERLU_SHEET_Y_CELLS = 16   # ... of 'sheet_sparse_direct' (host SuperLU: 240 cells took ~133 s per factorisation)
SUPERLU_SHEET_RUN = 'sheet_sparse_direct'
LATTICE_SCALE = 4            # cells per axis x this of the 8 x 6 Bloch cell (host SuperLU: x 10 took 211 s)

_CONICAL = 'open_sol.physics.em.solvers.fdfd_conical'
PARTS = COMMON_PARTS + EIGEN_PARTS + methods(f'{_CONICAL}:ConicalPhysics', 'conical physics', (
    'finalize', 'solve_harmonic', 'solve_eigenfrequencies', 'select_resonance', '_refresh_system', '_factorize',
    '_solve_linear', '_solve', '_show', '_conical_materials',
)) + methods(_CONICAL, 'conical monitors', (
    'ConicalProbe.record', 'ConicalFluxMonitor.record', 'conical_probe_result', 'conical_flux_result',
)) + methods('open_sol.physics.em.conical', 'conical ops', (
    'assemble_conical', '_symmetrizer', 'conical_fields', 'centre_phasors', 'driven_rhs', 'solve_conical_driven',
    'line_flux', 'source_power', 'solve_conical_resonances',
)) + (
    ('open_sol.physics.em.renderer_conical:ConicalRenderer.device_field', 'renderer: ConicalRenderer.device_field'),
)


def widened_sheet(spec, y_cells=SHEET_Y_CELLS):
    """Sheet spec with a y_cells-cell y period (lines and the medium rectangle span it)."""
    size_y = y_cells * float(spec.meshes[0].dx)
    spec.domain.size_y = size_y
    for shape in spec.geometry:
        if shape.kind in ('line', 'rectangle'):
            shape.params['y1'] = size_y
    return spec


def scaled_lattice(spec):
    """Bloch-cell spec with LATTICE_SCALE x the domain and medium rectangle (same cell, Bloch k and target)."""
    spec.domain.size_x = float(spec.domain.size_x) * LATTICE_SCALE
    spec.domain.size_y = float(spec.domain.size_y) * LATTICE_SCALE
    for shape in spec.geometry:
        if shape.kind == 'rectangle':
            shape.params['x1'] = float(shape.params['x1']) * LATTICE_SCALE
            shape.params['y1'] = float(shape.params['y1']) * LATTICE_SCALE
    return spec


def conical_specs():
    """{run: ModelSpec}: the widened cuDSS sheet runs ({} without nvmath) and the scaled Bloch lattice."""
    specs = {name: widened_sheet(spec, SUPERLU_SHEET_Y_CELLS if name == SUPERLU_SHEET_RUN else SHEET_Y_CELLS)
             for name, spec in case_specs('em_fdfd_conical_cudss').items()}
    specs.update({f'eigen_{name}': scaled_lattice(spec) for name, spec in case_specs('em_fdfd_conical_lattice').items()})
    return specs


if __name__ == '__main__':
    main('em_fdfd_conical', PARTS, __doc__, specs=conical_specs,
         note=f'sheet y period {SHEET_Y_CELLS} cells ({SUPERLU_SHEET_RUN}: {SUPERLU_SHEET_Y_CELLS}), '
              f'lattice x {LATTICE_SCALE} per axis')

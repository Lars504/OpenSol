"""Profile the 1D membrane line engine (MembraneLinePhysics / LineHarmonicSystem, fem.line_fem P2 / Hermite) on the
gpu: the lossy string driven over 51 frequencies (direct and adaptive) and as a complex resonance, the P2 strip with
an axial wavenumber and the 2000-cell Hermite plate strip at two k_y; small cases repeated 8x. Run from any
folder: python profiler_solver_membrane_line.py [--runs ...] [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402


def _specs():
    """{run: spec} prefixed per case (run names of different cases clash)."""
    loss = case_specs('membrane_line_loss')
    plate = case_specs('membrane_plate_strip_dispersion')
    return {
        'string_direct': loss['response'],
        'string_modes': loss['modes'],
        'string_adaptive': case_specs('membrane_line_loss_adaptive')['response'],
        'strip_dispersion': case_specs('membrane_line_dispersion')['strip'],
        'plate_strip_ky_0': plate['ky_0'],
        'plate_strip_ky_1': plate['ky_1'],
    }


MEMBRANE = 'open_sol.physics.membrane'
PHYSICS = f'{MEMBRANE}.solver_line:MembraneLinePhysics'
LINE_SYSTEM = f'{MEMBRANE}.solver_line:LineHarmonicSystem'
SYSTEM = 'open_sol.fem.harmonic:HarmonicSystemBase'
REDUCED = 'open_sol.fem.reduced_harmonic:ReducedHarmonicModel'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', '_read_coefficients', '_ready_system', '_new_system', '_apply_coefficients',
    'solve_eigenfrequencies', 'select_resonance', 'reset_frequency_response', 'plan_frequency_response',
    'solve_frequency_response', 'publish_frequency_response', '_publish', '_element_phasors', '_probe_value',
    'window_field', 'state',
)) + tuple((f'{LINE_SYSTEM}.{method}', f'line system: {method}') for method in (
    '_assemble_terms', '_source_load', 'eigen_pencil',
)) + tuple((f'{SYSTEM}.{method}', f'fem system: {method}') for method in (
    'assemble', 'load_vector', 'solve', '_solve_free', '_factorize', '_store_solution', 'publish_fields',
    'linear_output', 'plan_reduced', 'reduced_terms',
)) + (
    ('open_sol.fem.harmonic:HarmonicPencil.of', 'fem pencil: HarmonicPencil.of'),
    ('open_sol.fem.harmonic:HarmonicPencil.reduced', 'fem pencil: HarmonicPencil.reduced (per frequency)'),
    ('open_sol.fem.line_fem:line_dofs', 'line fem: line_dofs'),
    ('open_sol.fem.line_fem:assemble_line_stiffness', 'line fem: assemble_line_stiffness'),
    ('open_sol.fem.line_fem:assemble_line_mass', 'line fem: assemble_line_mass'),
    ('open_sol.fem.line_fem:assemble_line_bending', 'line fem: assemble_line_bending'),
    ('open_sol.fem.line_fem:evaluate_line', 'line fem: evaluate_line (probes, publish)'),
    ('open_sol.fem.harmonic_resonances:solve_system_resonances', 'fem eigen: solve_system_resonances'),
    ('open_sol.fem.harmonic_resonances:solve_harmonic_resonances', 'fem eigen: solve_harmonic_resonances'),
    ('open_sol.fem.harmonic_resonances:resonance_table', 'fem eigen: resonance_table'),
) + tuple((f'{REDUCED}.{method}', f'reduced sweep: {method}') for method in (
    'train', 'expand', 'residuals', 'reduced_coordinates', 'verify', '_append',
)) + (
    (f'{MEMBRANE}.renderer:MembraneLineRenderer.device_field', 'render: MembraneLineRenderer.device_field'),
    ('open_sol.model.study_frequency_response:_plan_reduced_sweeps', 'study: _plan_reduced_sweeps'),
    ('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'output: _SnapshotRecorder.record'),
    ('open_sol.model.study:_LiveFeed.update', 'output: _LiveFeed.update'),
)

if __name__ == '__main__':
    main('membrane_line', PARTS, __doc__, specs=_specs, repeat=8)

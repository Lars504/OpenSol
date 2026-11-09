"""Profile membrane film on triangles (MembraneFEMPhysics / MembraneTriangleSystem) on the gpu: eigen (template
rectangle h = a / 80), direct / adaptive / modal sweeps (rectangle window, direct cut to 61 frequencies), film ports
with mode projections (port tunnelling t11 at 3 frequencies) and a film PML guide. Run from any folder:
python profiler_solver_membrane_film.py [--runs ...] [--repeat N] [--no-cprofile]"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

DIRECT_FREQUENCIES = 61                 # of the case's 201 (cpu 0.19 s / frequency, mostly the factor)
PORT_FREQUENCY_FACTORS = (0.99, 1.0, 1.01)   # around the t11 drive (5.5 MHz): per-frequency DtN terms


def _with_settings(spec, **settings):
    return replace(spec, study=replace(spec.study, settings={**spec.study.settings, **settings}))


def _specs():
    """{run: spec} prefixed per case (run names of different cases clash)."""
    specs = {'eigen_rectangle': case_specs('membrane_emfdfd_rectangle', runs=['membrane'])['membrane']}
    for method, spec in case_specs('membrane_rectangle_reduced_sweep').items():
        if method == 'direct':
            spec = _with_settings(spec, n_frequencies=DIRECT_FREQUENCIES)
        specs[f'sweep_{method}'] = spec
    port = case_specs('membrane_port_tunnelling', runs=['t11'])['t11']
    base = float(port.study.settings['frequencies'][0])
    specs['port_t11'] = _with_settings(port, frequencies=[base * factor for factor in PORT_FREQUENCY_FACTORS])
    specs['guide_pml'] = case_specs('membrane_guide_pml')['guide']
    return specs


MEMBRANE = 'open_sol.physics.membrane'
PHYSICS = f'{MEMBRANE}.solver_fem:MembraneFEMPhysics'
TRIANGLE_SYSTEM = f'{MEMBRANE}.solver_fem:MembraneTriangleSystem'
SYSTEM = 'open_sol.fem.harmonic:HarmonicSystemBase'
REDUCED = 'open_sol.fem.reduced_harmonic:ReducedHarmonicModel'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', '_read_coefficients', '_ready_system', '_new_system', '_apply_coefficients',
    '_set_port_medium', '_pml', 'solve_eigenfrequencies', '_multi_shift_solve', '_shift_solver', 'select_resonance',
    'reset_frequency_response', 'plan_frequency_response', 'solve_frequency_response', 'publish_frequency_response',
    '_record_frequency', '_publish', '_element_phasors', 'fem_field', 'state', 'add_mode_projection', 'port_medium',
)) + tuple((f'{TRIANGLE_SYSTEM}.{method}', f'triangle system: {method}') for method in (
    '_assemble_terms', '_port_terms', 'frequency_terms', 'port_outputs',
)) + tuple((f'{SYSTEM}.{method}', f'fem system: {method}') for method in (
    'assemble', 'load_vector', 'solve', '_solve_free', '_factorize', '_device_terms', '_extra_on_device',
    '_keep_device_values', '_store_solution', 'publish_fields', 'linear_output', 'plan_reduced', 'reduced_terms',
    'modal_vectors', 'eigen_pencil', '_spread',
)) + (
    ('open_sol.fem.harmonic:ScalarHarmonicSystem._assemble_terms', 'fem system: ScalarHarmonicSystem._assemble_terms'),
    ('open_sol.fem.harmonic:ScalarHarmonicSystem._source_load', 'fem system: ScalarHarmonicSystem._source_load'),
    ('open_sol.fem.harmonic:HarmonicPencil.of', 'fem pencil: HarmonicPencil.of'),
    ('open_sol.fem.harmonic:HarmonicPencil.reduced', 'fem pencil: HarmonicPencil.reduced (host route)'),
    ('open_sol.fem.assembly:assemble_stiffness', 'fem assembly: assemble_stiffness'),
    ('open_sol.fem.assembly:assemble_mass', 'fem assembly: assemble_mass'),
    ('open_sol.fem.assembly:pattern_sum', 'fem assembly: pattern_sum'),
    ('open_sol.fem.assembly:_to_csr', 'fem assembly: _to_csr (COO -> CSR, also per-frequency port terms)'),
    ('open_sol.fem.dofs:lagrange_dofs', 'fem dofs: lagrange_dofs'),
    ('open_sol.fem.pml:pml_ring', 'fem pml: pml_ring'),
    ('open_sol.fem.harmonic_resonances:solve_system_resonances', 'fem eigen: solve_system_resonances'),
    ('open_sol.fem.harmonic_resonances:solve_harmonic_resonances', 'fem eigen: solve_harmonic_resonances'),
    ('open_sol.fem.harmonic_resonances:resonance_table', 'fem eigen: resonance_table'),
) + tuple((f'{REDUCED}.{method}', f'reduced sweep: {method}') for method in (
    'train', 'fit_modal', 'evaluate', 'expand', 'residuals', 'reduced_coordinates', 'verify', '_append',
)) + (
    (f'{MEMBRANE}.ports:MembranePort.assemble', 'ports: MembranePort.assemble'),
    (f'{MEMBRANE}.ports:MembranePort.terms', 'ports: MembranePort.terms'),
    (f'{MEMBRANE}.ports:MembranePort.outputs', 'ports: MembranePort.outputs'),
    (f'{MEMBRANE}.ports:ModeProjector.apply', 'ports: ModeProjector.apply'),
    (f'{MEMBRANE}.ports:mode_projector', 'ports: mode_projector'),
    (f'{MEMBRANE}.ports:port_edges', 'ports: port_edges'),
    ('open_sol.model.study_frequency_response:_plan_reduced_sweeps', 'study: _plan_reduced_sweeps'),
    ('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'output: _SnapshotRecorder.record'),
    ('open_sol.model.study:_LiveFeed.update', 'output: _LiveFeed.update'),
    ('open_sol.model.field_access_triangles:TriangleAccess._field_data', 'render: TriangleAccess._field_data'),
    ('open_sol.fem.sampling:evaluate', 'render: fem.sampling.evaluate (raster)'),
    ('open_sol.fem.sampling:display_triangulation', 'render: fem.sampling.display_triangulation'),
    ('open_sol.fem.renderer:FEMFieldRenderer.fem_field', 'render: FEMFieldRenderer.fem_field'),
    ('open_sol.fem.renderer:FEMFieldRenderer.device_field', 'render: FEMFieldRenderer.device_field'),
)

if __name__ == '__main__':
    main('membrane_film', PARTS, __doc__, specs=_specs)

"""Profile the membrane plate (C0 interior penalty P2, bending_fem; plate PML; plate ports, plate_ports) on the gpu:
eigen of the simply supported rectangle (h = a / 60), the complex Bloch plate strip cell, the plate PML guide (3
frequencies) and the plate strip between two plate ports (4 frequencies). Optional large run 'plate_film_limit' (the
tunnelling model with D -> 0 plate ports, cpu 49 s): add it with --runs. Run from any folder:
python profiler_solver_membrane_plate.py [--runs ...] [--repeat N] [--no-cprofile]"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

PML_FREQUENCY_FACTORS = (0.99, 1.0, 1.01)            # the PML reference stays the first frequency solved
PORT_FREQUENCY_FACTORS = (0.98, 0.99, 1.0, 1.01)     # per-frequency strip eig + DtN of both plate ports
DEFAULT_RUNS = ['plate_eigen', 'plate_bloch', 'plate_pml', 'plate_port']


def _with_frequencies(spec, factors):
    base = float(spec.study.settings['frequencies'][0])
    settings = {**spec.study.settings, 'frequencies': [base * factor for factor in factors]}
    return replace(spec, study=replace(spec.study, settings=settings))


def _specs():
    """{run: spec} prefixed per case (run names of different cases clash)."""
    return {
        'plate_eigen': case_specs('membrane_plate_supported_rectangle', runs=['fine'])['fine'],
        'plate_bloch': case_specs('membrane_plate_strip_bloch', runs=['ky_1'])['ky_1'],
        'plate_pml': _with_frequencies(case_specs('membrane_plate_guide_pml')['guide'], PML_FREQUENCY_FACTORS),
        'plate_port': _with_frequencies(case_specs('membrane_plate_port_supported')['strip'], PORT_FREQUENCY_FACTORS),
        'plate_film_limit': case_specs('membrane_plate_port_film_limit', runs=['plate'])['plate'],
    }


MEMBRANE = 'open_sol.physics.membrane'
PHYSICS = f'{MEMBRANE}.solver_fem:MembraneFEMPhysics'
TRIANGLE_SYSTEM = f'{MEMBRANE}.solver_fem:MembraneTriangleSystem'
SYSTEM = 'open_sol.fem.harmonic:HarmonicSystemBase'
PLATE_PORT = f'{MEMBRANE}.plate_ports:PlatePort'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', '_read_coefficients', '_ready_system', '_new_system', '_apply_coefficients',
    '_set_port_medium', '_plate_wall_kinds', '_pml', 'solve_eigenfrequencies', '_shift_solver', '_bloch_shift_solver',
    'select_resonance', 'reset_frequency_response', 'plan_frequency_response', 'solve_frequency_response',
    'publish_frequency_response', '_record_frequency', '_publish', 'fem_field', 'state', 'port_medium',
)) + tuple((f'{TRIANGLE_SYSTEM}.{method}', f'triangle system: {method}') for method in (
    '_assemble_terms', '_port_terms', 'frequency_terms', 'port_outputs',
)) + tuple((f'{SYSTEM}.{method}', f'fem system: {method}') for method in (
    'assemble', 'load_vector', 'solve', '_solve_free', '_factorize', '_device_terms', '_extra_on_device',
    '_keep_device_values', '_store_solution', 'linear_output', 'eigen_pencil',
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
    ('open_sol.fem.dofs:bloch_masters', 'fem dofs: bloch_masters'),
    ('open_sol.fem.dofs:bloch_prolongation', 'fem dofs: bloch_prolongation'),
    ('open_sol.fem.pml:pml_ring', 'fem pml: pml_ring'),
    ('open_sol.fem.harmonic_resonances:solve_system_resonances', 'fem eigen: solve_system_resonances'),
    ('open_sol.fem.harmonic_resonances:solve_harmonic_resonances', 'fem eigen: solve_harmonic_resonances'),
    ('open_sol.fem.harmonic_resonances:resonance_table', 'fem eigen: resonance_table'),
    (f'{MEMBRANE}.bending_fem:assemble_plate_bending', 'plate: assemble_plate_bending (volume + IP edges)'),
    (f'{MEMBRANE}.bending_fem:assemble_seam_bending', 'plate: assemble_seam_bending (Bloch seam)'),
    (f'{MEMBRANE}.bending_fem:_volume_terms', 'plate: _volume_terms (one point on straight elements)'),
    (f'{MEMBRANE}.bending_fem:side_values', 'plate: side_values (edge slopes / normal moments)'),
    (f'{MEMBRANE}.bending_fem:p2_derivatives', 'plate: p2_derivatives (isoparametric, per point)'),
    (f'{MEMBRANE}.bending_fem:edge_blocks', 'plate: edge_blocks (IP edge matrices)'),
    (f'{PLATE_PORT}.assemble', 'plate ports: PlatePort.assemble (trace map)'),
    (f'{PLATE_PORT}.modes', 'plate ports: PlatePort.modes'),
    (f'{PLATE_PORT}.terms', 'plate ports: PlatePort.terms'),
    (f'{PLATE_PORT}.outputs', 'plate ports: PlatePort.outputs'),
    (f'{MEMBRANE}.plate_ports:strip_modes', 'plate ports: strip_modes (dense companion eig)'),
    (f'{MEMBRANE}.plate_ports:plate_strip', 'plate ports: plate_strip'),
    (f'{MEMBRANE}.ports:port_edges', 'ports: port_edges'),
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
    main('membrane_plate', PARTS, __doc__, runs=DEFAULT_RUNS, specs=_specs)

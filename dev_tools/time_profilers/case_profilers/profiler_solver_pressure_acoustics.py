"""Profile pressure_acoustics on triangles (frequency response) on the gpu: selftest pressure_acoustics_rigid_disk run
(axisymmetric, refined mesh, background + PML, 2 frequencies, ~3.7 s) plus the 4 runs of
pressure_acoustics_reduced_sweep (duct, 61 frequencies direct / adaptive, velocity and background drives) on a 10x
finer mesh (dx 2.5 mm, ~2e4 P2 dofs; ~1.5-2 s each). Run from any folder:
python profiler_solver_pressure_acoustics.py [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

SWEEP_ELEMENT = 0.0025   # m (selftest: 0.025)

PHYSICS = 'open_sol.physics.acoustics.solver_harmonic_fem:AcousticHarmonicBase'
TRIANGLES = 'open_sol.physics.acoustics.solver_harmonic_fem:PressureAcousticsFEMPhysics'
SYSTEM = 'open_sol.fem.harmonic:HarmonicSystemBase'
REDUCED = 'open_sol.fem.reduced_harmonic:ReducedHarmonicModel'
# StudyType.run (_run_frequency_response) and feature apply callbacks are stored in the registry before the parts are
# wrapped: time what they call instead
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study_frequency_response:_plan_reduced_sweeps', 'study: _plan_reduced_sweeps'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record (raster reads)'),
    ('open_sol.model.study:_LiveFeed.update', 'study: live feed update'),
) + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', '_build_system', '_apply_boundaries', '_impedance_damping', '_background_preparation',
    '_background_at', '_background_nodal_at', '_background_load', 'plan_frequency_response',
    'solve_frequency_response', 'publish_frequency_response', '_show_solution', '_record', '_publish', '_host_total',
    '_surface_background', 'surface_result', '_component_field', '_element_velocity', '_device_centroids',
)) + tuple((f'{TRIANGLES}.{method}', f'triangles: {method}') for method in (
    '_entity_load', '_source_operators', '_point_value', '_centroid_values', '_locate', 'fem_field',
)) + tuple((f'{SYSTEM}.{method}', f'fem system: {method}') for method in (
    'assemble', 'load_vector', 'solve', '_solve_free', '_factorize', '_device_terms', '_extra_on_device',
    '_keep_device_values', '_store_solution', 'publish_fields', '_spread', 'linear_output', 'plan_reduced',
    'reduced_terms', '_reduced_held',
)) + (
    ('open_sol.fem.harmonic:ScalarHarmonicSystem._assemble_terms', 'fem system: ScalarHarmonicSystem._assemble_terms'),
    ('open_sol.fem.harmonic:ScalarHarmonicSystem._source_load', 'fem system: ScalarHarmonicSystem._source_load'),
    ('open_sol.fem.harmonic:ScalarHarmonicSystem.nodal_field', 'fem system: ScalarHarmonicSystem.nodal_field'),
    ('open_sol.fem.harmonic:HarmonicPencil.of', 'fem pencil: HarmonicPencil.of (canonical pattern, blocks)'),
    ('open_sol.fem.harmonic:HarmonicPencil.coupling', 'fem pencil: coupling (host A_fc(omega) per frequency)'),
    ('open_sol.fem.harmonic:HarmonicPencil.reduced', 'fem pencil: reduced (host A(omega))'),
    ('open_sol.fem.assembly:assemble_stiffness', 'fem assembly: assemble_stiffness'),
    ('open_sol.fem.assembly:assemble_mass', 'fem assembly: assemble_mass'),
    ('open_sol.fem.assembly:assemble_edge_mass', 'fem assembly: assemble_edge_mass'),
    ('open_sol.fem.assembly:assemble_edge_load', 'fem assembly: assemble_edge_load'),
    ('open_sol.fem.assembly:pattern_sum', 'fem assembly: pattern_sum'),
    ('open_sol.fem.dofs:lagrange_dofs', 'fem dofs: lagrange_dofs'),
    ('open_sol.fem.pml:pml_ring', 'fem pml: pml_ring'),
    ('open_sol.fem.sampling:PointLocator.locate', 'fem sampling: PointLocator.locate'),
    ('open_sol.fem.sampling:RasterSampler.sample', 'fem sampling: RasterSampler.sample'),
    ('open_sol.solvers.sparse_direct:DevicePencil.combined', 'role: DevicePencil.combined'),
) + tuple((f'{REDUCED}.{method}', f'reduced: {method}') for method in (
    'train', 'evaluate', 'expand', 'add_vectors', 'reduced_coordinates', 'residuals', 'verify',
))


def profiled_specs():
    """{run: spec}: the rigid disk run plus the reduced_sweep runs on the finer duct mesh."""
    specs = {f'disk_{name}': spec for name, spec in case_specs('pressure_acoustics_rigid_disk').items()}
    for name, spec in case_specs('pressure_acoustics_reduced_sweep').items():
        for mesh in spec.meshes:
            mesh.dx = SWEEP_ELEMENT
        specs[f'sweep_{name}'] = spec
    return specs


if __name__ == '__main__':
    main('pressure_acoustics', PARTS, __doc__, case='pressure_acoustics_rigid_disk', specs=profiled_specs)

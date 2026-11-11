"""Profile pressure_acoustics on tetrahedra (solver_harmonic_fem3d, frequency response) on the gpu: the 4 runs of
selftest pressure_acoustics_duct_3d (impedance, PML shell, soft, velocity ends; builtin Kuhn tetrahedra) on a 4x finer
mesh (dx 6.25 mm: ~3e4 P2 tetrahedra, ~5e4 dofs) at 5 frequencies each, plus a 'background' run (the impedance duct
under a +x background plane wave: per-frequency background loads and held offsets). Expected ~2-3 s per run, ~12 s.
Run from any folder:
python profiler_solver_pressure_acoustics_tetrahedra.py [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

DUCT_ELEMENT = 0.00625                              # m (selftest: 0.025)
FREQUENCIES = [500.0, 550.0, 600.0, 650.0, 700.0]   # Hz (selftest: 600 only)
AIR_DENSITY, AIR_SPEED = 1.2, 343.0

PHYSICS = 'open_sol.physics.acoustics.solver_harmonic_fem:AcousticHarmonicBase'
TETRAHEDRA = 'open_sol.physics.acoustics.solver_harmonic_fem3d:PressureAcousticsFEMPhysics3D'
SYSTEM = 'open_sol.fem.harmonic:HarmonicSystemBase'
SYSTEM_3D = 'open_sol.fem.harmonic3d:ScalarHarmonicSystem3D'
# StudyType.run (_run_frequency_response) and feature apply callbacks are stored in the registry before the parts are
# wrapped: time what they call instead
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record (plane cuts)'),
    ('open_sol.model.study:_LiveFeed.update', 'study: live feed update'),
    ('open_sol.model.tetrahedral_meshing:tetrahedralize', 'build: tetrahedralize'),
    ('open_sol.fem.mesh3d:structured_box_mesh', 'build: structured_box_mesh (Kuhn split)'),
    ('open_sol.fem.mesh3d:build_tetrahedron_mesh', 'build: build_tetrahedron_mesh (faces, adjacency)'),
) + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', '_build_system', '_apply_boundaries', '_impedance_damping', '_background_preparation',
    '_background_at', '_background_nodal_at', '_background_load', 'plan_frequency_response',
    'solve_frequency_response', 'publish_frequency_response', '_show_solution', '_record', '_publish', '_host_total',
    '_surface_background', '_component_field', '_element_velocity', '_device_centroids',
)) + tuple((f'{TETRAHEDRA}.{method}', f'tetrahedra: {method}') for method in (
    '_dof_maps', '_make_pml', '_window_face_entities', '_entity_load', '_source_operators', '_point_value',
    '_centroid_values', '_centroid_gradients', '_locate', 'lagrange_field',
)) + tuple((f'{SYSTEM}.{method}', f'fem system: {method}') for method in (
    'assemble', 'load_vector', 'solve', '_solve_free', '_factorize', '_device_terms', '_extra_on_device',
    '_keep_device_values', '_store_solution', 'publish_fields', 'linear_output',
)) + tuple((f'{SYSTEM_3D}.{method}', f'fem system 3D: {method}') for method in (
    '_assemble_terms', '_source_load', 'face_entities', 'lagrange_field',
)) + (
    ('open_sol.fem.harmonic:HarmonicPencil.of', 'fem pencil: HarmonicPencil.of (canonical pattern, blocks)'),
    ('open_sol.fem.harmonic:HarmonicPencil.coupling', 'fem pencil: coupling (host A_fc(omega) per frequency)'),
    ('open_sol.fem.harmonic:HarmonicPencil.reduced', 'fem pencil: reduced (host A(omega))'),
    ('open_sol.fem.assembly3d:assemble_lagrange_stiffness_3d', 'fem assembly 3D: assemble_lagrange_stiffness_3d'),
    ('open_sol.fem.assembly3d:assemble_lagrange_mass_3d', 'fem assembly 3D: assemble_lagrange_mass_3d'),
    ('open_sol.fem.assembly3d:assemble_face_load_3d', 'fem assembly 3D: assemble_face_load_3d'),
    ('open_sol.fem.assembly:pattern_sum', 'fem assembly: pattern_sum'),
    ('open_sol.fem.dofs3d:lagrange_dofs_3d', 'fem dofs: lagrange_dofs_3d'),
    ('open_sol.fem.pml:pml_shell', 'fem pml: pml_shell'),
    ('open_sol.fem.domain3d:TetrahedronDomain.locate', 'fem sampling: TetrahedronDomain.locate'),
    ('open_sol.fem.sampling3d:PointLocator3D.locate', 'fem sampling: PointLocator3D.locate'),
    ('open_sol.fem.sampling3d:evaluate_lagrange_3d', 'fem sampling: evaluate_lagrange_3d'),
    ('open_sol.model.field_access_tetrahedra:TetrahedraAccess._plane', 'session: tetrahedra plane cut'),
)


def profiled_specs():
    """{run: spec}: the duct_3d runs on the finer mesh at FREQUENCIES, plus the impedance duct under a background."""
    from open_sol.model.spec import FeatureSpec
    specs = case_specs('pressure_acoustics_duct_3d')
    specs['background'] = case_specs('pressure_acoustics_duct_3d', ['impedance'])['impedance']
    specs['background'].physics[0].features.append(FeatureSpec('incident', 'background_pressure_field', [], {
        'wave': 'plane_wave', 'amplitude': 0.5, 'direction': '+x', 'background_density': AIR_DENSITY,
        'background_sound_speed': AIR_SPEED}))
    for spec in specs.values():
        for mesh in spec.meshes:
            mesh.dx = DUCT_ELEMENT
        spec.study.settings = {'frequencies': list(FREQUENCIES)}
    return specs


if __name__ == '__main__':
    main('pressure_acoustics_tetrahedra', PARTS, __doc__, case='pressure_acoustics_duct_3d', specs=profiled_specs)

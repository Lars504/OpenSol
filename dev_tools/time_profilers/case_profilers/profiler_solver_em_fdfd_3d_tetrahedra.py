"""Profile em_fdfd_3d on a 'tetrahedra' mesh (Nedelec FEM eigenfrequency) on the gpu: the example
fem_mie_sphere_resonance_3d (gmsh sphere, PML shell, cudss_mixed; builtin mesher when gmsh is missing) and the
em3d_fem_model_bloch run moved to the gpu (Bloch LU, derived values on tetrahedra); the example on the double cuDSS
factor: profiler_solver_em_fdfd_3d_tetrahedra_double_factor.py. Run from any folder:
python profiler_solver_em_fdfd_3d_tetrahedra.py [--runs mie bloch__run] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import methods  # noqa: E402
from profiler_harness import REPOSITORY, SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

EXAMPLE = REPOSITORY / 'examples' / 'em3d' / 'fem_mie_sphere_resonance_3d.json'
NOTE = 'eigs excl = host ARPACK + per-application transfers; renderer sample re-evaluates E per component'


def load_example():
    """The fem_mie_sphere_resonance_3d example spec (cudss_mixed), on mesher builtin when gmsh is missing."""
    from open_sol import api
    from open_sol.fem.gmsh_io import find_gmsh
    example = api.load(EXAMPLE)
    if find_gmsh() is None:
        print('gmsh not found (OPEN_SOL_GMSH / PATH): the mie run uses mesher builtin', flush=True)
        example.meshes[0].mesher = 'builtin'
    return example


def _specs():
    """{'mie': the example, 'bloch__run': the em3d_fem_model_bloch run}."""
    runs = {'mie': load_example()}
    runs.update({f'bloch__{run}': spec for run, spec in case_specs('em3d_fem_model_bloch').items()})
    return runs


def _on_gpu(spec):
    """The cpu self-test specs (and a --case fallback) on the gpu backend."""
    spec.solver_defaults.backend = 'gpu'
    return spec


EM3D = 'open_sol.physics.em3d'
EIGEN = 'open_sol.physics.em.eigenfrequency'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
    ('open_sol.model.study_eigenfrequency:_search_solve', 'study: eigenfrequency _search_solve'),
    ('open_sol.model.field_access_tetrahedra:TetrahedraAccess.field', 'session: TetrahedraAccess.field'),
    ('open_sol.model.field_access_tetrahedra:TetrahedraAccess.evaluate', 'session: TetrahedraAccess.evaluate'),
    ('open_sol.model.tetrahedral_meshing:tetrahedralize', 'mesh: tetrahedralize'),
    ('open_sol.fem.gmsh3d:mesh_solids_with_gmsh', 'mesh: mesh_solids_with_gmsh'),
    ('open_sol.fem.gmsh3d:_run_gmsh', 'mesh: gmsh process'),
    ('open_sol.fem.dofs3d:nedelec_dofs_3d', 'fem: nedelec_dofs_3d'),
) + methods(f'{EM3D}.fem_solver3d:FEMResonancePhysics3D', 'physics', (
    'finalize', 'solve_eigenfrequencies', 'select_resonance', '_element_materials', 'state')
) + methods(f'{EM3D}.fem3d:FEMPhysics3D', 'fem physics', (
    '_assemble', '_pinned', '_bloch_prolongation', 'select_resonance', 'fields_located')
) + (
    (f'{EM3D}.fem3d:pml_shell_3d', 'fem: pml_shell_3d'),
    (f'{EM3D}.fem3d:bloch_identification', 'fem: bloch_identification'),
    (f'{EM3D}.fem3d:PMLShell3D.tensor', 'fem: PMLShell3D.tensor'),
    ('open_sol.fem.assembly3d:assemble_nedelec_3d', 'fem: assemble_nedelec_3d (host)'),
    ('open_sol.fem.sampling3d:evaluate_nedelec_3d', 'fem: evaluate_nedelec_3d (host field samples)'),
    ('open_sol.fem.sampling3d:PointLocator3D.locate', 'fem: PointLocator3D.locate'),
    ('open_sol.fem.domain3d:TetrahedronDomain.locate', 'fem: TetrahedronDomain.locate'),
    (f'{EIGEN}:solve_resonances', 'eigen: solve_resonances (host glue, transfers, ARPACK reverse calls)'),
    (f'{EIGEN}:_filtered_candidates', 'eigen: _filtered_candidates (Rayleigh polish, PML filter)'),
    (f'{EIGEN}:squared_cayley_roots', 'eigen: squared_cayley_roots'),
    (f'{EIGEN}:_resonance', 'eigen: _resonance'),
    (f'{EIGEN}:_PencilMass.device_times', 'eigen: _PencilMass.device_times'),
) + methods(f'{EM3D}.renderer_fem3d:FEMRenderer3D', 'renderer', ('sample', 'device_field'))

if __name__ == '__main__':
    main('em_fdfd_3d_tetrahedra', PARTS, __doc__, specs=_specs, spec_edit=_on_gpu, note=NOTE)

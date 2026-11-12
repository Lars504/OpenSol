"""Profile em_fdfd_3d on the Yee tensor grid (eigenfrequency, cudss_mixed eigen node) on the gpu: the example
fdfd_mie_sphere_resonance_3d (open CPML eighth sphere, subpixel smoothing, ~1e5 unknowns), the same on the double cuDSS
factor (mie_double_factor: one solve per application against the mixed factor's refinement) and the
em3d_fdfd_bragg_cell k_z sweep (2.3 s). Run from any folder (scale the example with --set dx_cell=30*nm):
python profiler_solver_em_fdfd_3d.py [--runs mie mie_double_factor bragg__sweep] [--no-cprofile]"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import REPOSITORY, SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

EXAMPLE = REPOSITORY / 'examples' / 'em3d' / 'fdfd_mie_sphere_resonance_3d.json'


def double_factor(spec):
    """Deep copy of spec with every arpack_shift_invert node on factorization cudss (the double cuDSS factor)."""
    spec = copy.deepcopy(spec)
    for node in spec.solvers:
        if node.type == 'arpack_shift_invert':
            node.settings = {**dict(node.settings), 'factorization': 'cudss'}
    return spec


def _specs():
    """{'mie': the example, 'mie_double_factor': it on the double factor, 'bragg__sweep': the em3d_fdfd_bragg_cell
    sweep run}."""
    from open_sol import api
    example = api.load(EXAMPLE)
    runs = {'mie': example, 'mie_double_factor': double_factor(example)}
    runs.update({f'bragg__{run}': spec for run, spec in case_specs('em3d_fdfd_bragg_cell').items()})
    return runs


def _methods(owner, label, names):
    return tuple((f'{owner}.{name}', f'{label}: {name}') for name in names)


EM3D = 'open_sol.physics.em3d'
EIGEN = 'open_sol.physics.em.eigenfrequency'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:_run_sweep', 'study: _run_sweep'),
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
    ('open_sol.model.study_eigenfrequency:_search_solve', 'study: eigenfrequency _search_solve'),
    ('open_sol.model.subpixel_geometry:build_interface_geometry_3d', 'build: subpixel build_interface_geometry_3d'),
) + _methods(f'{EM3D}.fdfd3d:FDFDPhysics3D', 'physics', (
    'finalize', 'set_subpixel_smoothing', 'solve_eigenfrequencies', 'select_resonance', '_refresh_system',
    '_assemble', '_materials_at', '_pec_cells', 'state')
) + (
    (f'{EM3D}.fdfd3d:face_permittivities', 'assembly: face_permittivities'),
    (f'{EM3D}.fdfd3d:smoothing_deltas', 'assembly: smoothing_deltas'),
    (f'{EM3D}.subpixel_smoothing3d:interface_eps', 'assembly: subpixel interface_eps'),
    (f'{EM3D}.operators3d:yee_operators_3d', 'assembly: yee_operators_3d (scipy kron curls)'),
    (f'{EM3D}.operators3d:live_unknowns_3d', 'assembly: live_unknowns_3d'),
    (f'{EM3D}.operators3d:curl_curl_operator', 'assembly: curl_curl_operator'),
    (f'{EM3D}.operators3d:face_stretches', 'assembly: face_stretches'),
    (f'{EM3D}.operators3d:face_dual_volumes', 'assembly: face_dual_volumes'),
    (f'{EM3D}.operators3d:nested_dissection_ordering', 'assembly: nested_dissection_ordering'),
    (f'{EIGEN}:solve_resonances', 'eigen: solve_resonances (host glue, transfers, ARPACK reverse calls)'),
    (f'{EIGEN}:_filtered_candidates', 'eigen: _filtered_candidates (Rayleigh polish, PML filter)'),
    (f'{EIGEN}:squared_cayley_roots', 'eigen: squared_cayley_roots'),
    (f'{EIGEN}:_resonance', 'eigen: _resonance'),
    (f'{EIGEN}:_PencilMass.device_times', 'eigen: _PencilMass.device_times'),
) + _methods(f'{EM3D}.renderer_fdfd3d:FDFDRenderer3D', 'renderer', ('device_field', 'device_plane', '_field',
                                                                    '_centre'))

if __name__ == '__main__':
    main('em_fdfd_3d', PARTS, __doc__, specs=_specs,
         note='CudssFactorization.solve and ArpackShiftInvertSolver.eigs are role parts: eigs excl = host ARPACK '
              '+ the per-application asnumpy / asarray of solve_resonances')

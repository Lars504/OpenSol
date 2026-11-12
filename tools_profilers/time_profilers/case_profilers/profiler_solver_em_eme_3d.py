"""Profile em_eme_3d (3D eigenmode expansion along z) on the gpu: the PEC step (SuperLU and cuDSS eigen node), the
lossy film stack and the template thermo_optic_phase_shifter_eme_3d (heat + thermo-optic + 10-mode EME) at a coarser
SHIFTER_DX. Run from any folder:
python profiler_solver_em_eme_3d.py [--runs NAME ...] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import methods  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

SHIFTER_DX = 80e-9   # template: 50 nm (80 x 60 x 400 cells, 4800-cell cross-sections); 80 nm ~ 50 x 38 x 250
# (case, runs or None = all); eme_3d_cudss_pec_model has no run without nvmath
CASES = (('eme_cpu_gpu', ['pec_3d_gpu']), ('eme_3d_cudss_pec_model', None), ('eme_3d_lossy_film_model', None))


def _specs():
    """{case__run: spec} of the CASES runs + 'shifter': the template on SHIFTER_DX cells."""
    from open_sol import api
    runs = {f'{case}__{run}': spec for case, names in CASES for run, spec in case_specs(case, names).items()}
    shifter = api.template('thermo_optic_phase_shifter_eme_3d')
    shifter.meshes[0].dx = SHIFTER_DX
    runs['shifter'] = shifter
    return runs


EME = 'open_sol.physics.eme'
MODES = 'open_sol.physics.em.cross_section_modes'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
) + methods('open_sol.coupling.coupling_term:CouplingTerm', 'coupling', ('apply', 'apply_now', 'flush')
) + methods('open_sol.physics.heat.solver:HeatPhysics', 'heat', ('finalize', 'step')
) + methods(f'{EME}.solver:EMEBase', 'eme', (
    'step', 'solve_harmonic', '_solve', '_section_modes', 'eps_sensitivity', '_port_scalars', 'port_fractions')
) + methods(f'{EME}.solver3d:EMEPhysics3D', 'eme 3d', (
    '_find_sections', '_layer_rows', '_layer_materials', '_layer_loss', '_layer_pec', '_stretched_areas',
    '_solve_section', '_reconstruct', '_cascade_sensitivity', 'state')
) + (
    (f'{EME}.solver:EMEPortsMonitor.record', 'monitor: EMEPortsMonitor.record'),
    (f'{EME}.solver:EMESectionTableMonitor.record', 'monitor: EMESectionTableMonitor.record'),
    (f'{EME}.sections:find_sections', 'sections: find_sections (streams z layers)'),
    (f'{MODES}:prepare_cross_section_modes', 'modes: prepare_cross_section_modes'),
    (f'{MODES}:assemble_cross_section', 'modes: assemble_cross_section'),
    (f'{MODES}:solve_prepared_modes', 'modes: solve_prepared_modes'),
    (f'{MODES}:_build_mode', 'modes: _build_mode'),
    (f'{EME}.mode_sets:cross_section_mode_set', 'modes: cross_section_mode_set'),
    (f'{EME}.mode_sets:stretched_areas', 'modes: stretched_areas'),
    (f'{EME}.mode_sets:cross_section_fields', 'field: cross_section_fields (host)'),
    (f'{EME}.interfaces:interface_smatrix', 'cascade: interface_smatrix'),
    (f'{EME}.interfaces:interface_defects', 'cascade: interface_defects'),
    (f'{EME}.cascade:CascadeBuilder.finish', 'cascade: CascadeBuilder.finish'),
    (f'{EME}.cascade:Cascade.section_amplitudes', 'cascade: Cascade.section_amplitudes'),
    (f'{EME}.adjoint:cross_section_gradient', 'adjoint: cross_section_gradient'),
) + methods(f'{EME}.renderer3d:EMERenderer3D', 'renderer', ('device_field', 'device_plane', '_field'))

if __name__ == '__main__':
    main('em_eme_3d', PARTS, __doc__, specs=_specs,
         note=f'shifter template at dx {SHIFTER_DX * 1e9:g} nm (edit SHIFTER_DX for the template mesh)')

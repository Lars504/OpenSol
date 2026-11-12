"""Profile em_fdtd_3d (Yee time stepping) on the gpu: dipole far field, mode ports, Bloch monitors, heat chain and TFSF box
self-test runs (~9.5 s of selftest wall). Run from any folder:
python profiler_solver_em_fdtd_3d.py [--runs NAME ...] [--no-cprofile] [--no-sync]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import methods  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

# (case, selftest wall s): CPML + dipole + flux box + far field; mode source / monitors (host cross-section solves);
# Bloch k != 0 imaginary set + flux planes + DFT layer; ohmic loss / q_accum -> heat -> thermo-optic rebuilds; TFSF box
CASES = (('em3d_far_field_dipole', 0.86), ('em3d_mode_source_transmission', 3.57), ('em3d_bloch_array_monitors', 1.26),
         ('em3d_heat_chain', 2.41), ('em3d_tfsf_closed_box', 1.28))


def _specs():
    """{case__run: spec} of every case in CASES."""
    return {f'{case}__{run}': spec for case, _ in CASES for run, spec in case_specs(case).items()}


EM3D = 'open_sol.physics.em3d'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
    ('open_sol.model.study:_LiveFeed.update', 'study: _LiveFeed.update'),
) + methods('open_sol.coupling.runner:MultiphysicsRunner', 'runner', ('prepare', 'advance', 'flush')
) + methods('open_sol.coupling.coupling_term:CouplingTerm', 'coupling', ('apply', 'apply_now', 'flush')
) + methods('open_sol.physics.heat.solver:HeatPhysics', 'heat', ('finalize', 'step')
) + methods(f'{EM3D}.physics3d:EMPhysics3D', 'physics', (
    'finalize', 'step', '_refresh_materials', 'state', 'take_absorbed_power', 'inject_source')
) + methods(f'{EM3D}.engine3d:FDTDEngine3D', 'engine', (
    'advance', 'rebuild_materials', '_refresh_pole_cb', '_refresh_ohmic_box', 'stability_issue', 'deposit_pole_work',
    'reset_q_accum', 'take_absorbed_power')
) + (
    (f'{EM3D}.material_index3d:build_material_index', 'engine: material_index3d.build_material_index'),
    (f'{EM3D}.sources3d:DipoleSource3D.calibrate', 'source: DipoleSource3D.calibrate'),
    (f'{EM3D}.tfsf3d:PlaneWaveSource3D.calibrate', 'source: PlaneWaveSource3D.calibrate'),
    (f'{EM3D}.mode_ports3d:solve_port_modes', 'ports: solve_port_modes'),
    (f'{EM3D}.mode_ports3d:port_materials', 'ports: port_materials (layer host reads)'),
    (f'{EM3D}.mode_ports3d:ModeSource3D.calibrate', 'ports: ModeSource3D.calibrate'),
    (f'{EM3D}.mode_ports3d:ModeMonitor3D.calibrate', 'ports: ModeMonitor3D.calibrate (per-frequency solves)'),
    (f'{EM3D}.mode_ports3d:ModeMonitor3D.amplitudes', 'ports: ModeMonitor3D.amplitudes'),
    ('open_sol.physics.em.cross_section_modes:prepare_cross_section_modes', 'modes: prepare_cross_section_modes'),
    ('open_sol.physics.em.cross_section_modes:assemble_cross_section', 'modes: assemble_cross_section'),
    ('open_sol.physics.em.cross_section_modes:solve_prepared_modes', 'modes: solve_prepared_modes'),
    ('open_sol.physics.em.cross_section_modes:mode_overlap', 'modes: mode_overlap'),
    (f'{EM3D}.monitors3d:PlaneFlux3D.update', 'monitor: PlaneFlux3D.update (per sample, per plane)'),
    (f'{EM3D}.monitors3d:PlaneFlux3D.flux', 'monitor: PlaneFlux3D.flux'),
    (f'{EM3D}.monitors3d:PlaneFlux3D.host_pairs', 'monitor: PlaneFlux3D.host_pairs'),
    (f'{EM3D}.monitors3d:DFTPlane3D.update', 'monitor: DFTPlane3D.update (per sample)'),
    (f'{EM3D}.monitors3d:DFTPlane3D.spectrum', 'monitor: DFTPlane3D.spectrum'),
    (f'{EM3D}.near_to_far3d:NearToFar3D.radiation_vectors', 'far field: NearToFar3D.radiation_vectors'),
    (f'{EM3D}.near_to_far3d:NearToFar3D.far_field_amplitude', 'far field: NearToFar3D.far_field_amplitude'),
) + methods(f'{EM3D}.renderer3d:EMRenderer3D', 'renderer', ('device_field', 'device_plane', '_field', '_centre'))

if __name__ == '__main__':
    main('em_fdtd_3d', PARTS, __doc__, specs=_specs,
         note='engine.advance is timed per step (one sync each): compare its excl s and cpu/wall with --no-sync')

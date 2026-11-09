"""Profile em_fdtd_hz (FDTDEngine2D_Hz, time stepping, gpu): selftest em_hz_mie_drude_cylinder run 'device' (TFSF plane
wave, Drude cylinder at 2.5 nm with face poles, two closed flux boxes at 241 wavelengths, track_absorption; both runs
19.1 s recorded). --per-step also times the per-step calls (one GPU sync each). Run from any folder:
python profiler_solver_em_fdtd_hz.py [--per-step] [--runs device reference]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import COMMON_PARTS, EM_PHYSICS_PARTS, methods, take_flag  # noqa: E402
from profiler_harness import main  # noqa: E402

PER_STEP = take_flag('--per-step')

_EM = 'open_sol.physics.em'
_ENGINE = f'{_EM}.solvers.fdtd_hz_2d:FDTDEngine2D_Hz'
PARTS = COMMON_PARTS + EM_PHYSICS_PARTS + methods(_ENGINE, 'engine', (
    'finalize', '_rebuild_e_coefficients', '_rebuild_h_coefficients', 'add_drude', 'add_pec_mask',
    '_ensure_pole_power_totals', 'take_absorbed_power', 'reset_q_accum',
)) + (
    (f'{_EM}.renderer:EMRendererHz.device_field', 'renderer: EMRendererHz.device_field'),
    (f'{_EM}.renderer:EMRendererHz._centred_e', 'renderer: EMRendererHz._centred_e'),
)
if PER_STEP:
    PARTS += methods(f'{_EM}:EMPhysics', 'per step', ('step',)) + methods(_ENGINE, 'per step', (
        'refresh_coefficients', 'advance', 'update_h', 'update_e', '_accumulate_e_new_half',
    )) + methods(f'{_EM}.monitors', 'per step', ('FluxMonitor.update', 'DFTMonitor.update', 'NearToFarField.update')) + (
        methods(f'{_EM}.sources:TFSFSourceHz', 'per step', ('inject_magnetic', 'inject_electric')))

if __name__ == '__main__':
    main('em_fdtd_hz' + ('_per_step' if PER_STEP else ''), PARTS, __doc__, case='em_hz_mie_drude_cylinder',
         runs=['device'], note='per-step parts timed' if PER_STEP else 'per-step calls untimed (in runner: advance)')

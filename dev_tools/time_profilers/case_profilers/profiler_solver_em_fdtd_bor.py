"""Profile em_fdtd_bor (BORFDTDPhysics, time stepping, gpu): selftest em_fdtd_bor_offaxis_dipole_model (driven order sum
m = 0..3: ~7 lockstep steppers, flux lines and a far-field box; 6.6 s recorded on the cpu backend). --per-step also times
the per-step calls (one GPU sync each: compare with the default run). Run from any folder:
python profiler_solver_em_fdtd_bor.py [--per-step] [--case em_fdtd_bor_model_ringdown]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import BOR_OPERATOR_PARTS, COMMON_PARTS, methods, take_flag  # noqa: E402
from profiler_harness import main  # noqa: E402

PER_STEP = take_flag('--per-step')

_PHYSICS = 'open_sol.physics.em.solvers.fdtd_bor'
_STEPPER = 'open_sol.physics.em.fdtd_bor'
PARTS = COMMON_PARTS + BOR_OPERATOR_PARTS + methods(f'{_PHYSICS}:BORFDTDPhysics', 'fdtd bor physics', (
    'finalize', '_refresh', '_host_materials', '_host_poles', 'stepped_materials', 'centre_samples', 'attach_harmonic',
)) + methods(_PHYSICS, 'fdtd bor monitors', (
    'BORCurrentSource.attach', 'BORFluxSpectrum.attach', 'BORFarFieldSpectrum.attach', 'shared_spectrum',
    'BORFluxSpectrum.order_powers', 'BORFluxSpectrum.powers', 'BORFarFieldSpectrum.term_samples',
    'fdtd_bor_flux_result', 'fdtd_bor_far_field_result', 'BORHarmonic.normalize',
)) + methods(_STEPPER, 'fdtd bor stepper', (
    'stretched_assembly', 'stable_time_step', 'order_time_step_limit', 'source_spectrum', 'combine_terms',
    'order_term_fields', 'BORTimeStepper.__init__', 'BORTimeStepper.refresh_coefficients',
    'BORTimeStepper.field_energy', 'BORTimeStepper.spectra_terms', 'BORTimeStepper.harmonic_yee',
    'BOROrderSum.__init__', 'BOROrderSum.refresh_coefficients', 'BOROrderSum.field_energy',
    'BOROrderSum.spectra_terms', 'BOROrderSum.harmonic_yee', 'BOROrderSum._phi_zero_sum', 'BORSpectra.full_phasors',
)) + (
    ('open_sol.physics.em.renderer_fdtd_bor:BORFDTDRenderer.device_field', 'renderer: BORFDTDRenderer.device_field'),
)
if PER_STEP:
    PARTS += methods(f'{_PHYSICS}:BORFDTDPhysics', 'per step', ('step',)) + methods(_STEPPER, 'per step', (
        'BOROrderSum.step', 'BORTimeStepper.step', '_LockstepBatch.step', '_BatchSide.advance',
        '_SpectraGroup.accumulate',
    ))

if __name__ == '__main__':
    main('em_fdtd_bor' + ('_per_step' if PER_STEP else ''), PARTS, __doc__, case='em_fdtd_bor_offaxis_dipole_model',
         note='per-step parts timed' if PER_STEP else 'per-step calls untimed (in runner: advance)')

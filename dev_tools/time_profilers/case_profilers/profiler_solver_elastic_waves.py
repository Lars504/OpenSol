"""Profile elastic_waves (tensor velocity-stress time stepping) on the gpu: the 6 gpu runs of selftest
elastic_waves_cpu_gpu (P / S strips, fluid-fluid reflection, thermoelastic heat pulse, plate resonances with the
resonance finder; 1.2e3-6.4e3 cells, 0.9-4.6e3 ticks, ~0.7-1.3 s each: launch-bound, shows the per-tick host work) plus
'p_wave_wide' = run p_wave of elastic_wave_speeds widened across its periodic y to WIDE_CELLS cells (500 x 2048 cells,
1171 ticks, ~1.5 s: kernel-bound). ~8 s in all. `step` is timed per tick (a GPU sync each): compare with --no-sync.
Run from any folder:
python profiler_solver_elastic_waves.py [--repeat N] [--no-cprofile] [--no-sync]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

WIDE_CELLS = 2048   # cells across the periodic strip of 'p_wave_wide' (selftest: 4)
GPU_RUNS = ['gpu__elastic_wave_speeds__p_wave', 'gpu__elastic_wave_speeds__s_wave', 'gpu__elastic_wave_speeds__p_wave_y',
            'gpu__acoustic_normal_reflection__run', 'gpu__acoustic_thermoelastic_slab__run',
            'gpu__acoustic_plate_resonances__run']

# StudyType.run / feature apply / create callbacks are stored in the registry before the parts are wrapped: time the
# methods they call (runner, physics) instead of those module functions
PHYSICS = 'open_sol.physics.acoustics.solver:AcousticPhysics'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record'),
    ('open_sol.model.study:_LiveFeed.update', 'study: live feed update'),
    ('open_sol.coupling.runner:MultiphysicsRunner.prepare', 'runner: prepare'),
    ('open_sol.coupling.runner:MultiphysicsRunner.advance', 'runner: advance (per chunk)'),
    ('open_sol.coupling.runner:MultiphysicsRunner.flush', 'runner: flush'),
    ('open_sol.physics.acoustics.cpml:AcousticAxisLayers.configure', 'physics: CPML configure'),
) + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    '__init__', 'finalize', 'step', '_update_time_step', '_rebuild_coefficients', '_configure', '_set_storage_scales',
    'pressure', 'elastic_strain', 'physical_array', 'add_traction_line', 'add_heat_pulse', 'add_force_source',
    'add_probe',
)) + (
    ('open_sol.physics.acoustics.renderer:AcousticRenderer.device_field', 'renderer: device_field'),
    ('open_sol.physics.acoustics.sources:AcousticProbe.get_data', 'results: AcousticProbe.get_data'),
    ('open_sol.core.resonance:ResonanceMonitor.result', 'results: ResonanceMonitor.result'),
    ('open_sol.core.resonance:harmonic_inversion', 'results: harmonic_inversion (host)'),
)


def profiled_specs():
    """{run: spec}: the gpu runs of elastic_waves_cpu_gpu plus the widened P-wave strip."""
    specs = case_specs('elastic_waves_cpu_gpu', GPU_RUNS)
    wide = case_specs('elastic_wave_speeds', ['p_wave'])['p_wave']
    width = WIDE_CELLS * float(wide.meshes[0].dx)
    wide.domain.size_y = width
    for shape in wide.geometry:
        if shape.name == 'sheet':
            shape.params['y1'] = width
    specs['p_wave_wide'] = wide
    return specs


if __name__ == '__main__':
    main('elastic_waves', PARTS, __doc__, case='elastic_waves_cpu_gpu', specs=profiled_specs)

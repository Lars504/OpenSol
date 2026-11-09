"""Profile laminar_flow (D2Q9 lattice Boltzmann) on the gpu, transient time stepping and the steady march:
- transient: 'tg_wide' = fluid_taylor_green_decay on a TG_CELLS^2 box (selftest 64^2) for 4x its 620 steps (~2.5 s,
  kernel-bound); fluid_time_dependent_drives 'force' / 'wall' (feature values of t re-set every tick: 500 / ~3400
  ticks on 16 / 48 cells, launch-bound, ~1 s); fluid_poiseuille_startup_half_storage (half_storage populations).
- steady: the gpu runs of fluid_cpu_gpu (force / pressure channels, Bouzidi channel: host wall links; ~0.8 s each)
  plus 'force_wide' = the force channel 1024 x 40 cells (selftest 8 x 10; drive / 16: same peak speed), a ~1.4e4-step
  march (~1.5 s).
~9 s in all. Transient `step` is timed per tick (a GPU sync each): compare with --no-sync. Run from any folder:
python profiler_solver_laminar_flow.py [--repeat N] [--no-cprofile] [--no-sync]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

TG_CELLS = 1024          # Taylor-Green box side (cells)
TG_DURATION_FACTOR = 4   # x the selftest's 620 lattice steps
FORCE_CELLS = (1024, 40)  # force_wide channel (length periodic, height between walls)
STEADY_RUNS = ['gpu__fluid_poiseuille_force__run', 'gpu__fluid_poiseuille_pressure__run',
               'gpu__fluid_bouzidi_channel__run']

# StudyType.run / feature apply / create callbacks are stored in the registry before the parts are wrapped: time the
# methods they call (runner, physics) instead of those module functions
PHYSICS = 'open_sol.physics.fluid.solver:FluidPhysics'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record'),
    ('open_sol.model.study:_LiveFeed.update', 'study: live feed update'),
    ('open_sol.coupling.runner:MultiphysicsRunner.prepare', 'runner: prepare'),
    ('open_sol.coupling.runner:MultiphysicsRunner.advance', 'runner: advance (per chunk)'),
    ('open_sol.coupling.runner:MultiphysicsRunner.flush', 'runner: flush'),
    ('open_sol.coupling.material_update:TimeDependentUpdate.apply_now', 'runner: value of t update (per tick)'),
    ('open_sol.physics.fluid.bouzidi:wall_link_fractions', 'physics: Bouzidi wall_link_fractions (host bisection)'),
) + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', 'step', '_refresh_lattice', '_refresh_wall_links', '_prepare_lattice', '_update_relaxation',
    '_update_face_values', '_rescale_populations', '_start_from_initial_flow', '_march_to_steady', '_velocity_change',
    '_publish', 'set_face_value', 'set_initial_flow', 'add_body_force', 'vorticity_window',
)) + (
    ('open_sol.physics.fluid.renderer:FluidRenderer.device_field', 'renderer: device_field'),
    ('open_sol.core.monitors:FieldProbe.get_data', 'results: FieldProbe.get_data'),
)


def _taylor_green_wide():
    """fluid_taylor_green_decay on a TG_CELLS^2 box (wavenumber rescaled to the box) for TG_DURATION_FACTOR x its
    steps."""
    spec = case_specs('fluid_taylor_green_decay')['run']
    old_size = float(spec.domain.size_x)
    new_size = TG_CELLS * float(spec.meshes[0].dx)
    spec.domain.size_x = spec.domain.size_y = new_size
    for feature in spec.physics[0].features:
        if feature.type == 'initial_velocity':
            feature.params = {key: value.replace(repr(old_size), repr(new_size)) if isinstance(value, str) else value
                              for key, value in feature.params.items()}
    spec.study.settings['duration'] *= TG_DURATION_FACTOR
    return spec


def _force_wide():
    """The fluid_cpu_gpu force channel at FORCE_CELLS, the drive scaled by (old / new height)^2 (same peak speed)."""
    spec = case_specs('fluid_cpu_gpu', ['gpu__fluid_poiseuille_force__run'])['gpu__fluid_poiseuille_force__run']
    cell = float(spec.meshes[0].dx)
    old_height = float(spec.domain.size_y)
    spec.domain.size_x, spec.domain.size_y = (cells * cell for cells in FORCE_CELLS)
    for feature in spec.physics[0].features:
        if feature.type == 'body_force':
            feature.params['f_x'] = float(feature.params['f_x']) * (old_height / spec.domain.size_y) ** 2
    return spec


def profiled_specs():
    """{run: spec}: transient runs first, then the steady marches."""
    specs = {'tg_wide': _taylor_green_wide()}
    specs.update({f'drives_{name}': spec for name, spec in case_specs('fluid_time_dependent_drives').items()})
    specs['startup_half'] = case_specs('fluid_poiseuille_startup_half_storage')['run']
    specs.update(case_specs('fluid_cpu_gpu', STEADY_RUNS))
    specs['force_wide'] = _force_wide()
    return specs


if __name__ == '__main__':
    main('laminar_flow', PARTS, __doc__, case='fluid_taylor_green_decay', specs=profiled_specs)

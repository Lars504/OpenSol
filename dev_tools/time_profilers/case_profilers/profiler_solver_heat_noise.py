"""Profile the thermal_noise study of heat (thermal_noise.py: host SuperLU per (frequency, axial) shift on the
'heat' physics vs the GPU line_block solver on 'heat_line_block', one em_modes solve) on the gpu: selftest
heat_noise_line_block (40.5 s with 8 frequencies) cut to --frequencies 2 (first estimate ~10 s). Run from any
folder:
python profiler_solver_heat_noise.py [--frequencies N] [--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import (COMMON_PARTS, TENSOR_DIFFUSION_PARTS, functions, main, methods,  # noqa: E402
                                      pop_option)

NOISE = 'open_sol.physics.heat.thermal_noise'
OPERATOR = f'{NOISE}:_HarmonicDiffusionOperator'
LINE_BLOCK = 'open_sol.solvers.line_block:LineBlockSolver'
PARTS = (COMMON_PARTS + TENSOR_DIFFUSION_PARTS + (
    ('open_sol.physics.heat.solver:HeatPhysics.solve_thermal_noise', 'heat: solve_thermal_noise'),
    ('open_sol.physics.heat.solver:HeatPhysics.rebuild_operator', 'heat: rebuild_operator'),
    ('open_sol.physics.em.solvers.modes_2d:ModePhysics.solve_modes', 'em_modes: solve_modes'),
    (f'{NOISE}:HarmonicHeatOperator.__init__', 'noise: HarmonicHeatOperator build (host operator copy)'),
    (f'{NOISE}:HarmonicParticleOperator.__init__', 'noise: HarmonicParticleOperator build'),
) + methods(OPERATOR, (
    'shifted_spectral_densities', 'spectral_densities', '_line_block_solver', 'variance_per_length',
    'static_variance_per_length',
), 'noise operator') + functions(NOISE, (
    'solve_noise_monitor', 'resolve_mode_channels', 'mode_pair_weighting', 'illumination_fields', 'fundamental_mode',
    'axial_samples', 'thermal_noise_result', '_contribution_weights', '_band_integral',
), 'noise') + methods(LINE_BLOCK, (
    'set_operator', '_bind_layout', '_block_cache', '_solve_chunk', '_factor_chunk', '_refine', '_residual', 'release',
), 'line_block'))


def fewer_frequencies(spec, count):
    """spec with its thermal_noise study cut to `count` frequencies over the same band."""
    spec.study.settings['n_frequencies'] = int(count)
    return spec


if __name__ == '__main__':
    frequencies = pop_option('frequencies', 2, int)
    main('heat_noise', PARTS, __doc__, case='heat_noise_line_block',
         spec_edit=lambda spec: fewer_frequencies(spec, frequencies), note=f'{frequencies} frequencies')

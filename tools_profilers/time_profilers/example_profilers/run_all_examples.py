"""Run every example_profiler.py group in turn on run_all_profilers.py's machinery (solver flags, backends, timeout,
history, profile points and plot), all of it in out/ here (out/run_all.log, out/history/<timestamp>/,
out/profile_points.json / .png); with no solver flag every role runs Automatic. Run from any folder: python run_all_examples.py [--only SUBSTRING ...]
[--cudss --cudss-mixed --scipy --pardiso --mumps | --all] [--backend gpu cpu] [--timeout S] [--new-first] [--plot-only]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'case_profilers'))
import run_all_profilers as suite  # noqa: E402
from example_profiler import GROUPS  # noqa: E402

suite.COMMANDS = tuple(GROUPS)
suite.OUT = HERE / 'out'
suite.POINTS_PATH = suite.OUT / 'profile_points.json'
suite.PLOT_PATH = suite.OUT / 'profile_points.png'
suite.PLOT_TITLE = 'example profilers per group'
suite.command_line = lambda entry: [str(HERE / 'example_profiler.py'), *entry.split()]
# every variant starts from Automatic (the examples' own solver nodes dropped): no flag is Automatic's pick, a group
# flag changes only what its group sets (back end, precision)
suite.SOLVERS = {solver: {**variant, 'arguments': ('--automatic-solvers', *variant['arguments'])}
                 for solver, variant in suite.SOLVERS.items()}
# cudss_mixed runs at --precision single: magnetostatics' float32 CG is ~8 % off with iron (mu_dc 100; validation
# warns, definition_helpers.single_precision_contrast_issues), so that row times a wrong answer
suite.SOLVER_EXCLUDED_COMMANDS = {'cudss_mixed': ('magnetic',)}

if __name__ == '__main__':
    suite.main(__doc__)

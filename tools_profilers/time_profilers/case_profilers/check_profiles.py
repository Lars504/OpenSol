"""Batch --if-changed check of profiler commands in one process (run_all_profilers calls it: open_sol, the registry and
the package snapshot load once, not ~8 s per command). Reads a JSON {key: [script, arguments...]}, runs each script's
main() in profiler_harness CHECK_ONLY mode and writes {key: {'lines': stdout lines of its kept profile (or of no run
taking the compared solver), or None: profile it, 'error': text when the check itself failed}}.
python check_profiles.py COMMANDS.json RESULTS.json"""
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profiler_harness  # noqa: E402  (the module the scripts' own `from profiler_harness import` binds)


def main():
    commands_path, results_path = Path(sys.argv[1]), Path(sys.argv[2])
    commands = json.loads(commands_path.read_text(encoding='utf-8'))
    profiler_harness.CHECK_ONLY = True
    results = {}
    for key, arguments in commands.items():
        start = time.perf_counter()
        sys.argv = list(arguments)   # the scripts and the harness parser read their options from it
        try:
            runpy.run_path(arguments[0], run_name='__main__')
        except profiler_harness.ProfileChecked as checked:
            results[key] = {'lines': checked.lines}
        except (Exception, SystemExit) as error:   # a broken check: the command runs on its own and shows the error
            results[key] = {'lines': None, 'error': ''.join(traceback.format_exception_only(error)).strip()}
        else:
            results[key] = {'lines': None, 'error': 'the script never reached the harness check'}
        outcome = 'kept' if results[key]['lines'] else results[key].get('error') or 'needs profiling'
        print(f'{time.perf_counter() - start:6.2f} s  {key}: {outcome}', flush=True)
    results_path.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

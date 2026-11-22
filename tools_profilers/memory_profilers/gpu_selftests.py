"""Run the GPU runs of the selftest cases and examples of a checkout (cpu runs skipped) for compare_vram_reports.py:
light (default) the heaviest-measured few per selftests module / example folder, --heavy every GPU case and example.
Run: python gpu_selftests.py <checkout folder> [--heavy] [--pick-from <heavy run's report folder>] [--out DIR]"""
import argparse
import functools
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

SELECTION_FILE = Path(__file__).resolve().parent / 'out' / 'gpu_light_selection.json'   # shared by both checkouts
EXAMPLE_PREFIX = 'example '


def import_api(checkout):
    """open_sol.api imported from `checkout` (raises when another open_sol, e.g. the editable install, wins)."""
    sys.path.insert(0, str(checkout))
    api = importlib.import_module('open_sol.api')
    imported_checkout = Path(api.__file__).resolve().parent.parent   # open_sol/api.py -> repository folder
    if imported_checkout != checkout:
        raise ImportError(f'open_sol imported from {imported_checkout}, not from {checkout}')
    return api


def gpu_case_names(api, registry):
    """Registered cases with at least one model run (own or variant) on the gpu; direct-check cases have no vram."""
    return [case.name for case in registry.selftests() if api._case_estimate_specs(case, ('gpu',), registry)]


def gpu_example_names(api):
    """Selftest examples (with an analytic comparison) whose run resolves to the gpu backend."""
    names = []
    for path in api._selected_example_paths(None):
        spec = api.load_model(path)   # an unloadable example raises here: fix it before comparing
        if api.has_analytic_comparison(spec) and api.solver_backend(spec) == 'gpu':
            names.append(path.relative_to(api.EXAMPLES_DIR).as_posix())
    return names


def case_module(case):
    """Name of the selftests module that defines a case's check (partials unwrapped)."""
    function = case.check
    while isinstance(function, functools.partial):
        function = function.func
    return getattr(function, '__module__', None) or '(unknown module)'


def measured_peaks(report_dir):
    """({case name: peak bytes}, {example path: peak bytes}) over the runs of a report folder (largest measured part)."""
    from compare_vram_reports import measured_runs, peak_part   # after import_api: it imports no open_sol
    case_peaks, example_peaks = defaultdict(int), defaultdict(int)
    for key, entry in measured_runs(report_dir).items():
        peak = peak_part(entry)[1] or 0
        if key.startswith(EXAMPLE_PREFIX):
            path = key[len(EXAMPLE_PREFIX):].rsplit(' [', 1)[0]
            example_peaks[path] = max(example_peaks[path], peak)
        else:
            name = key.split(' ', 1)[0]   # 'case / run' or 'case [variant] / run'
            case_peaks[name] = max(case_peaks[name], peak)
    if not case_peaks and not example_peaks:
        raise ValueError(f'no GPU vram record in the reports of {report_dir}')
    return case_peaks, example_peaks


def heaviest_per_group(peaks, group_of, per_group):
    """The `per_group` names of largest peak in each group, heaviest first within a group."""
    groups = defaultdict(list)
    for name, peak in peaks.items():
        groups[group_of(name)].append((peak, name))
    return [name for group in sorted(groups) for _, name in sorted(groups[group], reverse=True)[:per_group]]


def build_selection(api, registry, report_dir, per_group):
    """Light selection from a heavy run's reports -> SELECTION_FILE: per selftests module / example folder the
    heaviest `per_group` cases / examples by measured peak."""
    case_peaks, example_peaks = measured_peaks(report_dir)
    modules = {case.name: case_module(case) for case in registry.selftests()}
    known_peaks = {name: peak for name, peak in case_peaks.items() if name in modules}
    selection = {'picked_from': str(report_dir), 'per_group': per_group,
                 'cases': heaviest_per_group(known_peaks, modules.__getitem__, per_group),
                 'examples': heaviest_per_group(example_peaks, lambda path: path.split('/', 1)[0], per_group)}
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_FILE.write_text(json.dumps(selection, indent=1), encoding='utf-8')
    print(f"selection: {len(selection['cases'])} cases, {len(selection['examples'])} examples -> {SELECTION_FILE}")
    return selection


def light_names(api, registry, args):
    """(case names, example paths) of the light selection that exist in this checkout (dropped ones printed)."""
    if args.pick_from:
        selection = build_selection(api, registry, Path(args.pick_from).resolve(), args.per_group)
    elif SELECTION_FILE.exists():
        selection = json.loads(SELECTION_FILE.read_text(encoding='utf-8'))
    else:
        raise FileNotFoundError(f'no {SELECTION_FILE}: pass --pick-from <report folder of a --heavy run> once')
    gpu_cases = set(gpu_case_names(api, registry))
    gpu_examples = set([] if args.no_examples else gpu_example_names(api))
    case_names = [name for name in selection['cases'] if name in gpu_cases]
    example_names = [path for path in selection['examples'] if path in gpu_examples]
    dropped = sorted(set(selection['cases']) - gpu_cases) + (
        [] if args.no_examples else sorted(set(selection['examples']) - gpu_examples))
    if dropped:
        print(f"not GPU selftests in this checkout (dropped): {', '.join(dropped)}")
    return case_names, example_names


def print_progress(event):
    if event.get('event') == 'begin':
        print(f"{event['phase']}: {event['n_run']} to run", flush=True)
    elif event.get('event') == 'start':
        print(f"[{event['index']}/{event['n_total']}] {event['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='GPU runs of the selftest cases and examples of a checkout.')
    parser.add_argument('checkout', help='repository folder whose open_sol is run (e.g. ..\\open_sol_main)')
    parser.add_argument('--heavy', action='store_true', help='every GPU case and example (default: the light selection)')
    parser.add_argument('--pick-from', metavar='DIR',
                        help=f'light: rebuild the selection from the reports of a --heavy run (-> {SELECTION_FILE.name})')
    parser.add_argument('--per-group', type=int, default=2,
                        help='light, with --pick-from: heaviest cases per selftests module / examples per folder (2)')
    parser.add_argument('--out', help='report folder (default <checkout>/selftest_gpu_light_out or selftest_gpu_out)')
    parser.add_argument('--no-examples', action='store_true', help='registered cases only')
    args = parser.parse_args()
    if args.heavy and args.pick_from:
        parser.error('--pick-from builds the light selection; it does not go with --heavy')
    checkout = Path(args.checkout).resolve()
    default_out = 'selftest_gpu_out' if args.heavy else 'selftest_gpu_light_out'   # *_out/ is gitignored
    out_dir = Path(args.out).resolve() if args.out else checkout / default_out
    api = import_api(checkout)
    if 'gpu' not in api._available_backends():
        raise RuntimeError('no usable GPU (cupy and a CUDA device)')
    api._available_backends = lambda: ('gpu',)   # cpu runs of mixed cases are skipped (their checks may then fail)
    registry = api.get_registry()
    if args.heavy:
        case_names = gpu_case_names(api, registry)
        example_names = [] if args.no_examples else gpu_example_names(api)
    else:
        case_names, example_names = light_names(api, registry, args)
    print(f"{checkout} ({'heavy' if args.heavy else 'light'}): {len(case_names)} gpu cases, {len(example_names)} gpu "
          f"examples -> {out_dir}", flush=True)
    start = time.perf_counter()
    # named runs never reuse cached records, so both checkouts measure fresh runs
    if case_names:
        api.selftest(out_dir=out_dir, names=case_names, progress=print_progress)
    if example_names:
        api.selftest_examples(out_dir=out_dir, names=example_names, progress=print_progress)
    print(f'done in {time.perf_counter() - start:.0f} s; reports in {out_dir}')


if __name__ == '__main__':
    main()

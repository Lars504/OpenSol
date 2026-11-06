"""Command line python -m open_sol describe | template | check | profile | run | show | selftest | mesh-study; every
command leaves a file.
Exit codes: 0 success, 1 model errors / run not completed / selftest not passed, 2 usage errors (incl. missing files)."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__, api
from .core import host_memory
from .io.results_io import json_safe, load_summary, write_json
from .model.describe import UnknownTopicError, reference_files
from .model.expressions import error_message, evaluate_parameters
from .model.schema import get_registry
from .model.structure_index import CONVENTIONS_PATH, STRUCTURE_INDEX_PATH, structure_index_text

_MAX_ISSUE_LINES = 30
_MAX_ERROR_LINES = 10
_OUTLINE_KEYS = ('title', 'description', 'parameters', 'domain', 'geometry', 'selections', 'materials', 'physics',
                 'couplings', 'meshes', 'solvers', 'solver_defaults', 'study', 'derived_values', 'analytic_curves',
                 'derived_curves')


def main(argv=None) -> int:
    """Parse argv (default sys.argv[1:]), run one command and return its exit code."""
    _configure_streams()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:   # argparse: usage error (2) or --help (0)
        return 0 if exc.code is None else int(exc.code)
    try:
        return args.handler(args)
    except OSError as exc:   # missing / unreadable input file or unwritable output
        print(f'open_sol: {exc}', file=sys.stderr)
        return 2


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m open_sol',
        description='open_sol headless interface: every command writes its report next to the model (or to --out).')
    commands = parser.add_subparsers(dest='command', required=True, metavar='command')

    describe_parser = commands.add_parser('describe', help='API reference generated from the registry')
    describe_parser.add_argument('topic', nargs='?',
                                 help="one topic, e.g. 'spec', 'em_fdtd_ez', 'em_fdtd_ez.flux_monitor', 'templates'")
    target = describe_parser.add_mutually_exclusive_group()
    target.add_argument('--out', metavar='FILE', help='write the text to FILE (UTF-8, LF line ends)')
    target.add_argument('--update-reference', action='store_true',
                        help=f'rewrite {api.API_REFERENCE_PATH.name} and {api.API_TEMPLATES_PATH.name} next to the '
                             f'package (full text only) and, in a repository checkout, {STRUCTURE_INDEX_PATH.name}')
    describe_parser.add_argument('--json', action='store_true', help='structured JSON instead of markdown text')
    describe_parser.set_defaults(handler=_describe)

    template_parser = commands.add_parser('template', help='write a template model file to start from')
    template_parser.add_argument('key', help="template key (python -m open_sol describe templates lists them)")
    template_parser.add_argument('file', nargs='?', help='model JSON file to write (default ./<key>.json)')
    template_parser.add_argument('--force', action='store_true', help='overwrite an existing file')
    template_parser.set_defaults(handler=_template)

    check_parser = commands.add_parser('check', help='CPU-only validation, mesh and step estimate -> check.json')
    check_parser.add_argument('model', help='model JSON file')
    check_parser.add_argument('--out', metavar='DIR', help='report directory (default <model dir>/<model stem>_out)')
    _add_set_argument(check_parser)
    check_parser.add_argument('--max-wall-time', type=float, metavar='S',
                              help='warn when even the shortest estimated run time exceeds S seconds')
    check_parser.add_argument('--json', action='store_true', help='print the report as JSON')
    check_parser.set_defaults(handler=_check)

    profile_parser = commands.add_parser('profile', help='measure this machine for the run-time estimates -> '
                                                         'time_profiles/time_profile_<backend>.json')
    profile_parser.add_argument('--backend', choices=('gpu', 'cpu'),
                                help='backend to measure (default: every available backend, cpu and gpu when a '
                                     'CUDA device answers)')
    profile_parser.add_argument('--quick', action='store_true', help='coarser curves in ~5 s instead of ~30 s')
    profile_parser.set_defaults(handler=_profile)

    run_parser = commands.add_parser('run', help='build and run a model -> summary.json, log.txt, results.npz, '
                                                 'model.json, figures/')
    run_parser.add_argument('model', help='model JSON file')
    run_parser.add_argument('--out', metavar='DIR', help='report directory (default <model dir>/<model stem>_out)')
    run_parser.add_argument('--figures', action='store_true',
                            help='write figures/*.png: <key>.png per saved field and feature result, '
                                 'derived_<plot>.png, <result>_vs_<column>.png (table curves), curve_<name>.png '
                                 '(line curves), scene_3d.png (3D models: objects and the first saved field on '
                                 'x, y, z cut planes)')
    run_parser.add_argument('--max-wall-time', type=float, metavar='S',
                            help='stop after S seconds of wall time (status timeout, partial results)')
    run_parser.add_argument('--no-arrays', action='store_true', help='do not write results.npz')
    _add_set_argument(run_parser)
    run_parser.add_argument('--json', action='store_true', help='print summary.json on stdout (progress on stderr)')
    run_parser.set_defaults(handler=_run)

    show_parser = commands.add_parser('show', help='outline of a model file or summary of a results directory '
                                                   '-> show.json')
    show_parser.add_argument('path', help='model JSON file, or results directory (holding summary.json)')
    show_parser.add_argument('--out', metavar='DIR', help='show.json directory (default <model dir>/<model stem>_out '
                                                          'for a model, the directory itself for results)')
    show_parser.add_argument('--figures', action='store_true',
                             help='results directory: load results.npz and regenerate <out>/figures/*.png (else only '
                                  'summary.json is read); a sweep also gets its sweep_*.png overlays. 3D model '
                                  'file: write <out>/figures/scene_3d.png (perspective picture of the objects, no '
                                  'run); a 3D results directory also gets scene_3d.png with the saved cut planes')
    show_parser.add_argument('--series', metavar='NAME',
                             help='with --figures: draw only this series of xy results and of the sweep result '
                                  'overlays (e.g. fraction_positive of a mode monitor)')
    _add_set_argument(show_parser)
    show_parser.add_argument('--json', action='store_true', help='print show.json')
    show_parser.set_defaults(handler=_show)

    selftest_parser = commands.add_parser('selftest', help='built-in checks and analytic validation cases '
                                                           '-> selftest_report_<subset>.json')
    selftest_parser.add_argument('names', nargs='*', metavar='NAME',
                                 help='cases to run, heavy or not, built-in checks only when named (default the built-in '
                                      'checks and the non-heavy cases)')
    subset_group = selftest_parser.add_mutually_exclusive_group()
    subset_group.add_argument('--heavy', action='store_true',
                              help='run only the heavy registered cases (>= 5 s each) instead of the non-heavy ones')
    subset_group.add_argument('--all', action='store_true',
                              help='run everything (unchanged passing ones reused unless --reset): every registered '
                                   'case, heavy or not (gpu runs and variants with a '
                                   'usable GPU), then the --examples run -> the light and heavy reports + '
                                   'selftest_report_analytic.json')
    subset_group.add_argument('--examples', action='store_true',
                              help='instead: run every examples/**/*.json with a reference or analytic curve once, on '
                                   'its own backend (NAMEs: only these, relative path or stem) -> '
                                   'selftest_report_analytic.json')
    selftest_parser.add_argument('--out', metavar='DIR', default='selftest_out',
                                 help='report directory (default ./selftest_out); its selftest_cache.json lets a case '
                                      'or example that passed reuse its record while the code it ran is unchanged')
    selftest_parser.add_argument('--reset', action='store_true',
                                 help='rerun every selected case / example, also unchanged ones that passed (named '
                                      'ones always rerun); refreshes their cache entries')
    selftest_parser.add_argument('--json', action='store_true', help='print the report as JSON (case lines on stderr)')
    selftest_parser.add_argument('--list', action='store_true', help='list the case names and exit')
    selftest_parser.set_defaults(handler=_selftest)

    mesh_study_parser = commands.add_parser(
        'mesh-study', help='solver_plan S7.8 mesh study (mesh_study_plan.md): runs, records, gate report -> '
                           '<out>/mesh_study_report.json')
    mesh_study_parser.add_argument('parts', nargs='*', metavar='PART',
                                   help='per_physics, audit, prototypes (CPU), accuracy, pcg, line_block (GPU), '
                                        'analyse (default: all, in this order)')
    mesh_study_parser.add_argument('--out', metavar='DIR', default='mesh_study_out',
                                   help='study directory: records/, runs/, tables/, figures/ (default ./mesh_study_out)')
    mesh_study_parser.add_argument('--case', nargs='+', default=[], metavar='CASE',
                                   help='only these cases (accuracy: rib disk strip_heater disc_heater coax; pcg: '
                                        'strip_heater disc_heater polar_annulus; line_block); other parts skip')
    mesh_study_parser.add_argument('--fine', action='store_true', help='add the slowest accuracy / line_block variants')
    rerun_group = mesh_study_parser.add_mutually_exclusive_group()
    rerun_group.add_argument('--force', action='store_true', help='re-run variants already recorded')
    rerun_group.add_argument('--retry-failed', action='store_true', help='re-run recorded variants that did not complete')
    mesh_study_parser.add_argument('--max-cells', type=float, default=2e7, metavar='N',
                                   help='skip variants with more cells (default 2e7; not recorded)')
    mesh_study_parser.add_argument('--max-wall-time', type=float, metavar='S', help='stop each run after S seconds')
    mesh_study_parser.set_defaults(handler=_mesh_study)
    return parser


def _add_set_argument(parser):
    parser.add_argument('--set', action='append', default=[], metavar='NAME=EXPR', dest='overrides',
                        type=_override_pair, help='override a model parameter expression (repeatable), e.g. --set P_in=2e3')
    parser.add_argument('--backend', choices=('auto', 'gpu', 'cpu'),
                        help="override the model's solver_defaults.backend (where the run computes)")


def _override_pair(text):
    """argparse type of --set NAME=EXPR: (name, expression); a malformed one is a usage error (exit 2)."""
    name, equals, expression = text.partition('=')
    if not equals or not name.strip() or not expression.strip():
        raise argparse.ArgumentTypeError(f'--set takes NAME=EXPR, got {text!r}')
    return name.strip(), expression.strip()


# ---- commands ----
def _describe(args):
    """describe: markdown (or --json) to stdout, to --out FILE, or into api_reference.md + api_templates.md (+
    structure_index.md in a checkout) (--update-reference)."""
    if args.update_reference and (args.topic is not None or args.json):
        return _usage_error('--update-reference writes the full text reference; it takes no topic and no --json')
    try:
        if args.update_reference:
            texts = {api.API_REFERENCE_PATH.parent / name: text for name, text in reference_files().items()}
            if CONVENTIONS_PATH.is_file():   # the module index lives beside structure_conventions.md
                texts[STRUCTURE_INDEX_PATH] = structure_index_text()
        elif args.json:
            text = json.dumps(json_safe(api.describe(args.topic, as_dict=True)), indent=2) + '\n'
        else:
            text = api.describe(args.topic)
    except UnknownTopicError as exc:   # other errors propagate with a traceback (exit 1)
        return _usage_error(str(exc))
    except RuntimeError as exc:   # describe's documentation check: help text missing or extra in the registry tables
        print(f'open_sol describe: {exc}', file=sys.stderr)
        return 1
    if not (args.update_reference or args.out is not None):
        sys.stdout.write(text)
        return 0
    if not args.update_reference:
        texts = {Path(args.out): text}
    for target, target_text in texts.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(target_text, encoding='utf-8', newline='\n')
        print(f'wrote {target}')
    return 0


def _template(args):
    """template: write the template KEY as a model JSON file (an existing file only with --force)."""
    target = Path(args.file) if args.file else Path(f'{args.key}.json')
    if target.exists() and not args.force:
        return _usage_error(f'{target} exists; pass --force to overwrite it or name another FILE')
    try:
        spec = api.template(args.key)
    except KeyError as exc:   # unknown key: the message lists the keys
        return _usage_error(error_message(exc))
    target.parent.mkdir(parents=True, exist_ok=True)
    api.save(spec, target)
    print(f'wrote {target}')
    return 0


def _check(args):
    """check: CPU-only report -> <out>/check.json; exit 1 on model errors."""
    overrides = dict(args.overrides)   # --set pairs, the last one wins
    model_path = Path(args.model)
    out_dir = _default_out(model_path, args.out)
    report = api.check(model_path, out_dir=out_dir, parameters=overrides, host_budget=host_memory.available_bytes(),
                       backend=args.backend, max_wall_time=args.max_wall_time)
    if args.json:
        _print_json(report)
    else:
        for line in _check_lines(report, model_path):
            print(line)
        print(f"report: {out_dir / 'check.json'}")
    return 1 if report['n_errors'] else 0


def _profile(args):
    """profile: run the machine benchmarks of the run-time estimate and store the profile of the backend, or of every
    available backend without --backend (exit 0)."""
    from .core.backend import gpu_available
    backends = [args.backend] if args.backend else ['cpu'] + (['gpu'] if gpu_available() else [])
    for backend in backends:
        print(f'profiling {backend}:')
        record = api.profile_machine(backend, quick=args.quick, progress=lambda text: print(f'  {text}'))
        print(record['text'])
        print(f"wrote {record['path']}")
    return 0


def _run(args):
    """run: blocking api.run with progress lines; the Results is closed before exiting (exit 0 only if completed)."""
    if args.max_wall_time is not None and not args.max_wall_time > 0.0:
        return _usage_error(f'--max-wall-time must be > 0 s, got {args.max_wall_time:g}')
    overrides = dict(args.overrides)   # --set pairs, the last one wins
    model_path = Path(args.model)
    out_dir = _default_out(model_path, args.out)
    stream = sys.stderr if args.json else sys.stdout
    try:
        results = api.run(model_path, out_dir=out_dir, max_wall_time=args.max_wall_time,
                          progress=api.ProgressPrinter(stream), figures=args.figures, save_arrays=not args.no_arrays,
                          parameters=overrides, backend=args.backend)
    except KeyboardInterrupt:   # api.run has closed the session and written its minimal report
        print(f'interrupted; the partial report is in {out_dir}', file=sys.stderr)
        return 1
    try:
        summary = results.summary()
        if args.json:
            _print_json(summary)
        print(f'run {model_path}', file=stream)
        for line in _summary_lines(summary):
            print(f'  {line}', file=stream)
        print(f'report: {out_dir}', file=stream)
    finally:
        n_log_lines = len(results.log)
        results.close(copy_to_host=False)   # the process exits; nothing reads a host copy
        for line in results.log[n_log_lines:]:   # close-time warnings (also appended to log.txt)
            print(line, file=sys.stderr)
    return 0 if results.status == 'completed' else 1


def _show(args):
    """show: summary of a results directory or outline of a model file -> show.json."""
    path = Path(args.path)
    if not path.exists():
        print(f'open_sol: no such file or directory: {path}', file=sys.stderr)
        return 2
    overrides = dict(args.overrides)   # --set pairs, the last one wins
    if path.is_dir():
        if overrides:
            print('--set applies to model files only', file=sys.stderr)
        return _show_results(path, args)
    return _show_model(path, args, overrides)


def _show_results(path, args):
    stream = sys.stderr if args.json else sys.stdout
    out_dir = Path(args.out) if args.out else path
    try:   # results.npz and model.json are read only to draw figures
        results = api.Results.load(path) if args.figures else None
        summary = results.summary() if results is not None else load_summary(path)
    except OSError:   # no summary.json (not a results directory) or unreadable: exit 2 in main
        raise
    except Exception as exc:   # corrupt report files (e.g. a truncated results.npz): show.json says so
        error_text = f'{type(exc).__name__}: {error_message(exc)}'
        print(f'open_sol: cannot load the results in {path}: {error_text}', file=sys.stderr)
        _write_show_report(out_dir, _show_report('results', path, None, None, {'error': error_text}), args.json,
                           stream)
        return 1
    print(f'results {path}', file=stream)
    for line in _summary_lines(summary):
        print(f'  {line}', file=stream)
    if args.figures:   # the figures of api.run(figures=True); --series only where the result has it
        for name, draw in results._figure_jobs(args.series):
            _write_figure(name, draw, out_dir / 'figures' / f'{name}.png', stream)
    _write_show_report(out_dir, _show_report('results', path, None, None, summary), args.json, stream)
    return 0


def _show_model(path, args, overrides):
    stream = sys.stderr if args.json else sys.stdout
    check_report = api.check(path, parameters=overrides, host_budget=host_memory.available_bytes(),
                             backend=args.backend)
    try:
        spec = api.load(path)
    except ValueError:   # invalid content: the check report holds the message
        spec = None
    if spec is not None:
        try:
            api._apply_parameters(spec, overrides)
        except KeyError:   # unknown name: the check report holds the issue; the outline shows the file as is
            pass
    outline = None if spec is None else _model_outline(spec)
    counts = {key: check_report[key] for key in ('n_errors', 'n_warnings', 'n_infos')}
    print(f'model {path}', file=stream)
    for line in [] if outline is None else _outline_lines(outline):
        print(f'  {line}', file=stream)
    print(f"check: {counts['n_errors']} errors, {counts['n_warnings']} warnings, {counts['n_infos']} infos",
          file=stream)
    for line in _issue_lines(check_report['issues']):
        print(f'  {line}', file=stream)
    out_dir = _default_out(path, args.out)
    if args.figures and spec is not None and spec.domain.size_z is not None:
        _write_figure('scene_3d', lambda target: api.scene_figure(spec, target), out_dir / 'figures' / 'scene_3d.png',
                      stream)
    elif args.figures:
        print('--figures applies to results directories and 3D model files only', file=sys.stderr)
    _write_show_report(out_dir, _show_report('model', path, outline, counts, None), args.json, stream)
    return 0 if spec is not None else 1


def _write_figure(name, draw, target, stream):
    """draw(target), then 'wrote <target>' or why it was not written (never raises: the other figures go on)."""
    try:
        draw(target)
    except Exception as exc:   # report it and go on with the other figures
        print(f'  figure {name} not written: {type(exc).__name__}: {error_message(exc)}', file=stream)
    else:
        print(f'  wrote {target}', file=stream)


def _selftest(args):
    """selftest: a cached-count line per phase, two lines per run case (start, result), then 'N/M passed'; exit 1
    unless every case passed."""
    if args.examples:
        return _selftest_examples(args)
    if args.list:
        print('built-in (run first, except in a named run that does not name them):')
        for name in api.BUILTIN_SELFTESTS:
            print(f'  {name}')
        print('registered:')
        for case in get_registry().selftests():
            print(f"  {case.name}  {case.label}{'  (heavy)' if case.heavy else ''}")
        return 0
    out_dir = Path(args.out)
    stream = sys.stderr if args.json else sys.stdout
    heavy = True if args.heavy else None if args.all else False
    if args.all and args.names:
        return _usage_error('--all runs everything; give NAMEs without --all')
    try:
        report = api.selftest(out_dir=out_dir, names=args.names or None, heavy=heavy, reset=args.reset,
                              progress=lambda record: _print_selftest_record(record, stream))
    except KeyError as exc:   # unknown case name, raised before any case runs
        return _usage_error(error_message(exc))
    except KeyboardInterrupt as interrupt:   # api.selftest has written the partial report
        _print_interrupted_counts(getattr(interrupt, 'partial_report', None))
        print(f"interrupted; the partial report is in {out_dir}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(report)
    files = ', '.join(str(out_dir / name) for name in report['report_files'])
    print(f"{report['n_passed']}/{report['n_cases']} passed ({report['subset']} cases, {report['n_skipped']} "
          f"registered cases not run, {report['n_reused']} unchanged since passing: reused, --reset reruns them)",
          file=stream)
    if 'examples' in report:
        print(f"examples: {_examples_counts_text(report['examples'])}", file=stream)
    print(f'report: {files}', file=stream)
    return 0 if report['passed'] else 1


def _selftest_examples(args):
    """selftest --examples: the cached count, two lines per example run (start, result), then the counts; exit 1 on a
    failed or erroring example."""
    if args.list:
        print('use --examples without --list: the examples are run, not listed', file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    stream = sys.stderr if args.json else sys.stdout
    try:
        report = api.selftest_examples(out_dir=out_dir, names=args.names or None, reset=args.reset,
                                       progress=lambda record: _print_selftest_record(record, stream))
    except KeyError as exc:   # unknown example name, raised before any example runs
        return _usage_error(error_message(exc))
    except KeyboardInterrupt as interrupt:   # api.selftest_examples has written the partial report
        _print_interrupted_counts(getattr(interrupt, 'partial_report', None))
        print(f'interrupted; the partial report is in {out_dir}', file=sys.stderr)
        return 1
    if args.json:
        _print_json(report)
    print(f"{_examples_counts_text(report)}; report: {out_dir / report['report_files'][0]}", file=stream)
    return 0 if report['passed'] else 1


def _print_interrupted_counts(report):
    """Passed / failed / error counts of the tests finished before an interrupt (stderr); nothing without a report."""
    if report is None:
        return
    if 'n_cases' in report:   # api.selftest (an 'all' run may hold the examples finished so far too)
        print(f"interrupted after {report['n_cases']} cases: {report['n_passed']} passed, {report['n_failed']} "
              f"failed, {report['n_error']} errors", file=sys.stderr)
        if 'examples' in report:
            print(f"examples: {_examples_counts_text(report['examples'])}", file=sys.stderr)
    else:   # api.selftest_examples
        print(f'interrupted: {_examples_counts_text(report)}', file=sys.stderr)


def _examples_counts_text(counts):
    """'N/M examples passed, ... reused' of an analytic report (or its counts in a selftest --all report)."""
    not_run_here = (f"{counts['n_skipped_backend']} gpu examples not run: no usable GPU; "
                    if counts['n_skipped_backend'] else '')
    return (f"{counts['n_passed']}/{counts['n_examples']} examples passed, {counts['n_failed']} failed, "
            f"{counts['n_error']} errors, {counts['n_no_tolerance']} without a tolerance ({not_run_here}"
            f"{counts['n_reused']} unchanged: reused)")


def _mesh_study(args):
    """mesh-study: the named parts (default all) -> <out>/records, tables, figures, mesh_study_report.json; exit 1 when
    a part raised (its traceback is printed, the other parts still ran)."""
    from . import mesh_study   # imports api and every study part: only this command needs it
    unknown = [part for part in args.parts if part not in mesh_study.PARTS]
    if unknown:
        return _usage_error(f"unknown mesh-study part {', '.join(unknown)}; parts: {', '.join(mesh_study.PARTS)}")
    if args.max_wall_time is not None and not args.max_wall_time > 0.0:
        return _usage_error(f'--max-wall-time must be > 0 s, got {args.max_wall_time:g}')
    options = mesh_study.StudyOptions(out_dir=Path(args.out), force=args.force, retry_failed=args.retry_failed,
                                      fine=args.fine, cases=tuple(args.case), max_cells=args.max_cells,
                                      max_wall_time=args.max_wall_time)
    try:
        failed = mesh_study.run_study(args.parts, options)
    except KeyboardInterrupt:   # every finished run is already a record line
        print(f'interrupted; the records so far are in {options.records_dir} (re-run to resume)', file=sys.stderr)
        return 1
    if failed:
        print(f"mesh study: part(s) {', '.join(failed)} raised (tracebacks above)", file=sys.stderr)
    return 1 if failed else 0


# ---- output helpers ----
_RESULT_INDENT = '      '   # second line of a test (its result), under the start line


def _print_example(record, stream):
    """'PASS|FAIL|ERROR|NOTOL  worst error / tolerance R  [flags]  (time split)  vram D  counts D (item)' plus a
    failure message."""
    tag = {'passed': 'PASS', 'failed': 'FAIL', 'no_tolerance': 'NOTOL'}.get(record['status'], 'ERROR')
    flags = f"  [{', '.join(record['flags'])}]" if record['flags'] else ''
    print(f"{_RESULT_INDENT}{tag:5} worst error / tolerance {_number(record['worst_ratio'])}{flags}  "
          f"({_time_split_text(record)}){_deviations_text(record)}", file=stream, flush=True)
    if record['status'] in ('failed', 'error') and record['message']:
        print(f"{_RESULT_INDENT * 2}{record['message']}", file=stream, flush=True)


def _print_selftest_record(record, stream):
    """api.selftest / selftest_examples progress: a 'begin' / 'start' / 'skipped' event, an example record or a case
    record."""
    if 'event' in record:
        _print_event(record, stream)
    else:
        (_print_example if 'example' in record else _print_case)(record, stream)


def _print_event(event, stream):
    """'begin': '<phase>: N to run, M cached'; 'start': 'i/n  ~T left  name  (last T, estimate low - high)';
    'skipped': 'SKIP reason'."""
    if event['event'] == 'begin':
        text = (f"{event['phase']}: {event['n_run']} to run, {event['n_reused']} cached (unchanged since their last "
                'run; --reset reruns them)')
    elif event['event'] == 'start':
        untimed = f" + {event['n_untimed']} untimed" if event['n_untimed'] else ''
        last = 'never timed' if event['last_wall_time'] is None else _duration(event['last_wall_time'])
        text = (f"{event['index']}/{event['n_total']}  ~{_duration(event['remaining'])} left{untimed}  "
                f"{event['name']}  (last {last}{_estimate_before_text(event.get('time_estimate'))})")
    else:
        text = f"{_RESULT_INDENT}SKIP  {event['message']}"
    print(text, file=stream, flush=True)


def _estimate_before_text(estimate):
    """', estimate low - high' (the span of the guessed counts; one value when every count is exact) of a 'start'
    event's time_estimate ('+' when part of the work has no recipe, 'partly' when some runs could not be estimated);
    ', no estimate' without a usable machine profile; ' (outdated gpu profile)' appended when priced on an out-of-date
    one; '' without runs."""
    if estimate is None:
        return ''
    if not estimate['n_estimated']:
        message = estimate['message']
        reason = ('' if 'profile' not in message else ' (no machine profile)' if 'no machine profile' in message
                  else ' (machine profile unreadable)')
        return ', no estimate' + reason
    low, high = estimate['seconds_low'], estimate['seconds_high']
    span = _duration(low) if low == high else f'{_duration(low)} - {_duration(high)}'
    extra = '+' if estimate['not_estimated'] else ''
    partly = ' partly' if estimate['n_estimated'] < estimate['n_runs'] else ''
    return f", estimate{partly} {span}{extra}{estimate.get('outdated', '')}"


def _time_split_text(record):
    """'9.3 s: build 6.1 [est 5.0], study 1.5 [est 2.1], output 1.4 [est 1.1], other 0.3' of a finished case /
    example record (study estimate at the runs' measured counts; no [est] without one); 'wall s' without a split."""
    split = (record.get('time_estimate') or {}).get('split')
    if split is None:
        return f"{record['wall_time']:.1f} s"
    parts = []
    for phase in ('build', 'study', 'output', 'other'):
        estimate = split[phase]['estimate']
        parts.append(f"{phase} {split[phase]['seconds']:.1f}" + ('' if estimate is None else f' [est {estimate:.1f}]'))
    return f"{split['total']:.1f} s: " + ', '.join(parts)


def _deviations_text(record):
    """'  vram -40% max 1200 MB  counts +200% (CG iterations)': the worst measured / estimated - 1 of the GPU memory (-
    the estimate books too much, + more held), its largest measured part and the worst solver count; each when measured."""
    vram, counts = record.get('vram_deviation'), record.get('count_deviation')
    text = ''
    if vram is not None:
        text = f'  vram {vram:+.0%}'
        max_bytes = record.get('vram_max_bytes')
        if max_bytes is not None:
            max_megabytes = max_bytes / 2 ** 20
            text += f' max {max_megabytes:.1f} MB' if max_megabytes < 1.0 else f' max {max_megabytes:.0f} MB'
    if counts is not None:
        item = record.get('count_item')
        text += f'  counts {counts:+.0%}' + ('' if item is None else f' ({item})')
    return text


def _duration(seconds):
    """'12.3 s', '4 min 05 s' or '1 h 02 min'."""
    if seconds < 60.0:
        return f'{seconds:.1f} s'
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f'{minutes} min {rest:02d} s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} h {minutes:02d} min'


def _print_case(record, stream):
    """'PASS|FAIL  error E <= / > tolerance T  (time split)  vram D  counts D (item)' or 'ERROR  (time split) ...', plus
    the message of a case that did not pass."""
    status = record['status']
    tag = {'passed': 'PASS', 'failed': 'FAIL'}.get(status, 'ERROR')
    comparison = ''
    if tag != 'ERROR':
        relation = '<=' if status == 'passed' else '>'
        comparison = f" error {_number(record['error'])} {relation} tolerance {_number(record['tolerance'])}"
    print(f"{_RESULT_INDENT}{tag:5}{comparison}  ({_time_split_text(record)}){_deviations_text(record)}",
          file=stream, flush=True)
    if record['message']:
        print(f"{_RESULT_INDENT * 2}{record['message']}", file=stream, flush=True)


def _check_lines(report, model_path):
    """'check <file>: N errors, M warnings, K infos; mesh ...' then the first issues."""
    counts = f"{report['n_errors']} errors, {report['n_warnings']} warnings, {report['n_infos']} infos"
    mesh, study = report['mesh'], report['study'] or {}
    if mesh is None:
        mesh_text = f"mesh not estimated ({report['mesh_error'] or 'see the issues'})"
    else:
        mesh_text = (f"mesh {api._cells_text(mesh['nx'], mesh['ny'], mesh.get('nz'), mesh['n_cells'])}"
                     if mesh.get('nx') is not None else f"mesh ~{mesh['n_cells']:,} {mesh.get('kind') or 'elements'}")
        if len(report.get('meshes') or []) > 1:
            mesh_text += f" (mesh '{mesh['name']}'; {len(report['meshes'])} meshes)"
        if study.get('dt_estimate') is not None:
            mesh_text += f", dt {study['dt_estimate']:.3g} s"
        if study.get('total_steps') is not None:
            mesh_text += f", ~{study['total_steps']:,} steps"
    if study.get('n_wavelengths') is not None:   # frequency_domain study
        mesh_text += f", {study['n_wavelengths']} wavelengths"
    return [f'check {model_path}: {counts}; {mesh_text}', *_issue_lines(report['issues'])]


def _summary_lines(summary):
    """Readable lines of a run summary (the minimal fallback form included)."""
    message = summary.get('message')
    lines = [f"status {summary.get('status')}" + (f': {message}' if message else '')]
    if summary.get('wall_time') is not None:
        study_text = (f" (study {_number(summary['study_seconds'])} s)"
                      if summary.get('study_seconds') is not None else '')
        lines.append(f"wall time {_number(summary['wall_time'])} s{study_text}")
    build = summary.get('build')
    if build:
        dt_text = '' if build.get('dt') is None else f"dt {_number(build['dt'])} s, "   # time-stepped physics only
        lines.append(f"mesh {api._cells_text(build['nx'], build['ny'], build.get('nz'), build['n_cells'])}, "
                     f"{dt_text}built in {_number(build['build_seconds'])} s")
    references, errors = summary.get('derived_reference') or {}, summary.get('derived_error') or {}
    for name, value in (summary.get('derived') or {}).items():
        comparison = (f' (reference {_number(references[name])}, relative error {_number(errors.get(name))})'
                      if name in references else '')
        lines.append(f'{name} = {_number(value)}{comparison}')
    for name, record in (summary.get('curve_error') or {}).items():
        lines.append(f'curve {name}: {_curve_error_text(record)}')
    for name, record in (summary.get('derived_curves') or {}).items():
        state = f"error: {record['error']}" if 'error' in record else f"{len(record.get('x') or [])} points"
        lines.append(f"derived curve {name} (plot {record.get('plot')}): {state}")
    sweep = summary.get('sweep')
    if sweep:
        lines.append(f"sweep {sweep['parameter']}: " + ', '.join(_number(value) for value in sweep['values']))
        for name, values in sweep['derived'].items():
            lines.append(f'  {name}: ' + ', '.join(_number(value) for value in values))
            for label, key in (('reference', 'reference'), ('relative error', 'error')):
                table = sweep.get(key) or {}
                if name in table:
                    lines.append(f'  {name} {label}: ' + ', '.join(_number(value) for value in table[name]))
        for name, records in (sweep.get('curve_error') or {}).items():
            lines.append(f'  curve {name} max relative error: '
                         + ', '.join(_number(record.get('max')) for record in records))
    for kind, digests in (('field', summary.get('fields')), ('result', summary.get('results'))):
        for key, digest in (digests or {}).items():
            lines.append(f'{kind} {key}: {_digest_text(digest)}')
    warnings, errors = summary.get('warnings') or [], summary.get('errors') or []
    lines.append(f'{len(warnings)} warnings, {len(errors)} errors')
    lines.extend(f'error: {text}' for text in errors[:_MAX_ERROR_LINES])
    lines.extend(_issue_lines(summary.get('issues') or []))
    files = summary.get('files') or {}
    written = [name for key, name in files.items() if key != 'figures' and name]
    figures = files.get('figures') or []
    if written or figures:
        lines.append('files: ' + ', '.join(written + ([f'{len(figures)} figures'] if figures else [])))
    return lines


def _curve_error_text(record):
    """'max relative error M, rms R over N points' of a curve_error record, or why it was not compared."""
    if not record.get('n_points'):
        return f"not compared ({record.get('note') or 'no finite points'})"
    return (f"max relative error {_number(record.get('max'))}, rms {_number(record.get('rms'))} over "
            f"{record['n_points']} points")


def _digest_text(digest):
    """Scalar entries of a field / result digest; just the error when it failed."""
    if 'error' in digest:
        return f"error: {digest['error']}"
    return ', '.join(f'{key} {_number(value)}' for key, value in digest.items()
                     if key not in ('kind', 'physics', 'component', 'label') and not isinstance(value, (list, dict)))


def _issue_lines(issues):
    """'[level] node: message' of the first 30 issue records."""
    lines = [f"[{issue['level']}] {issue['node']}: {issue['message']}" for issue in issues[:_MAX_ISSUE_LINES]]
    if len(issues) > _MAX_ISSUE_LINES:
        lines.append(f'... {len(issues) - _MAX_ISSUE_LINES} more issues in the report')
    return lines


def _model_outline(spec):
    """show.json outline: the spec's JSON content (version left out) with each parameter's evaluated value."""
    plain = spec.to_dict()
    values = evaluate_parameters(spec.parameters, errors={})
    for parameter in plain['parameters']:
        parameter['value'] = values.get(parameter['name'])
    return {key: plain[key] for key in _OUTLINE_KEYS}


def _outline_lines(outline):
    """Readable lines of a model outline."""
    lines = [f"title: {outline['title']}"]
    if outline['description']:
        lines.append(f"description: {outline['description']}")
    for parameter in outline['parameters']:
        value = '' if parameter['value'] is None else f" (= {_number(parameter['value'])})"
        lines.append(f"parameter {parameter['name']} = {parameter['expression']}{value}")
    lines.append(f"domain: {outline['domain']['size_x']} x {outline['domain']['size_y']} m")
    for item in outline['geometry']:
        lines.append(f"geometry {item['name']} ({item['kind']}): {item['params']}")
    for item in outline['selections']:
        lines.append(f"selection {item['name']} ({item['kind']}): {_names_text(item['entities'])}")
    for item in outline['materials']:
        library = '' if item['library'] is None else f" [{item['library']}]"
        properties = f" {item['properties']}" if item['properties'] else ''
        lines.append(f"material {item['name']}{library} on {_names_text(item['selection'])}{properties}")
    for physics in outline['physics']:
        solvers = f", solvers {physics['solvers']}" if physics['solvers'] else ''
        lines.append(f"physics {physics['name']} ({physics['type']}, every {physics['every']}): "
                     f"settings {physics['settings']}{solvers}")
        for feature in physics['features']:
            disabled = '' if feature['enabled'] else ' [disabled]'
            lines.append(f"  feature {feature['name']} ({feature['type']}) on {_names_text(feature['selection'])}"
                         f"{disabled}: {feature['params']}")
    for coupling in outline['couplings']:
        disabled = '' if coupling['enabled'] else ' [disabled]'
        lines.append(f"coupling {coupling['name']} ({coupling['type']}): {coupling['source']} -> "
                     f"{coupling['target']}{disabled} {coupling['params']}")
    for mesh in outline['meshes']:
        refinements = f", {len(mesh['refinements'])} refinements" if mesh['refinements'] else ''
        lines.append(f"mesh {mesh['name']} ({mesh['kind']}): {mesh['mode']}, dx {mesh['dx']}{refinements}")
    for solver in outline['solvers']:
        lines.append(f"solver {solver['name']} ({solver['type']}): {solver['settings']}")
    if outline['solver_defaults']['backend'] != 'auto':
        lines.append(f"solver backend: {outline['solver_defaults']['backend']}")
    if outline['solver_defaults'].get('precision', 'auto') != 'auto':
        lines.append(f"solver precision: {outline['solver_defaults']['precision']}")
    study, sweep = outline['study'], outline['study']['sweep']
    if sweep is None:
        sweep_text = ''
    elif sweep.get('spacing', 'list') == 'list':
        sweep_text = f"; sweep {sweep['parameter']} over {len(sweep['values'])} values"
    else:
        sweep_text = (f"; sweep {sweep['parameter']} {sweep['lower']} .. {sweep['upper']}, {sweep['count']} "
                      f"{sweep['spacing']} values")
    lines.append(f"study: {study['type']} {study['settings']}{sweep_text}")
    for derived in outline['derived_values']:
        over = (f"result {derived['source']}" if derived['source'] is not None
                else _names_text(derived['selection']) or 'the window')
        at = f" at {derived['at']}" if derived['operation'] == 'at' else ''
        reference = '' if derived['reference'] is None else f", reference {derived['reference']}"
        if derived.get('tolerance') is not None:
            reference += f", tolerance {derived['tolerance']}"
        lines.append(f"derived {derived['name']} = {derived['operation']}({derived['expression']}){at} over "
                     f"{over}{reference}")
    for curve in outline['analytic_curves']:
        cut = ''.join(f", {key} {curve[key]}" for key in ('plane', 'snapshot') if curve.get(key) is not None)
        series = ', '.join(f"{key} {curve[key]}" for key in ('x_column', 'series') if curve.get(key))
        where = (f" of {curve['field']} along {curve['line']}{cut}" if curve['target'] == 'line'
                 else f' ({series})' if series else '')
        tolerance = '' if curve.get('tolerance') is None else f" (tolerance {curve['tolerance']})"
        lines.append(f"analytic curve {curve['name']} on {curve['target']}{where}: {curve['expression']}{tolerance}")
    for curve in outline['derived_curves']:
        inputs = ', '.join(f'{alias} = {reference}' for alias, reference in curve['inputs'].items())
        lines.append(f"derived curve {curve['name']} in plot {curve['plot'] or curve['name']}: {curve['expression']} "
                     f"with {inputs}")
    return lines


def _names_text(names):
    return ', '.join(str(name) for name in names)


def _show_report(kind, path, outline, check_counts, summary):
    """show.json content (decisions: kind 'model' | 'results')."""
    return {'open_sol_version': __version__, 'kind': kind, 'path': str(path), 'outline': outline,
            'check': check_counts, 'summary': summary}


def _write_show_report(out_dir, report, as_json, stream):
    report_path = out_dir / 'show.json'
    write_json(report_path, report)
    if as_json:
        _print_json(report)
    print(f'report: {report_path}', file=stream)


def _default_out(model_path, out):
    """--out, else '<model dir>/<model stem>_out'."""
    return Path(out) if out else model_path.parent / f'{model_path.stem}_out'


def _number(value):
    """Compact text of a report value (None -> 'n/a'; digest strings such as 'nan' unchanged)."""
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def _print_json(data):
    print(json.dumps(json_safe(data), indent=2))


def _usage_error(message):
    print(f'open_sol: error: {message}', file=sys.stderr)
    return 2


def _configure_streams():
    """stdout / stderr escape characters the console cannot encode instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            try:
                reconfigure(errors='backslashreplace')
            except ValueError:   # a detached or replaced stream: keep it as it is
                pass


if __name__ == '__main__':
    sys.exit(main())

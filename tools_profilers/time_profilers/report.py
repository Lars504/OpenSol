"""Machine record, per-record summaries (min / median / max per phase over the warm repeats, cold repeat, CPU over
wall time, eigenvalue / derived-value difference against the baseline) and the markdown comparison tables of the
solver profiler."""
import importlib.metadata
import os
import platform
import statistics
import sys

from tools_profilers.time_profilers.case_runner import RSS_BOUND, RSS_PROCESS, RSS_RAISED, RSS_RESET
from tools_profilers.time_profilers.runners import ACCURACY_MARGIN

# optional packages whose versions go into the machine record (besides time_profile.library_versions)
OPTIONAL_DISTRIBUTIONS = ('python-mumps', 'pydiso', 'mkl', 'threadpoolctl', 'numba', 'cupy', 'cupy-cuda12x',
                          'nvmath-python')
PHASES = ('analysis', 'factor', 'solve', 'refactor', 'total', 'sweep',   # sweep = refactor + solve of a repeat
          'build', 'study', 'output')                                     # model cases (model_cases module notes)
MIB = 2 ** 20
MEMORY_METHODS = {RSS_RESET: 'Linux VmHWM reset after the assembly: exact',
                  RSS_RAISED: 'the solver phase raised the process peak above the assembly peak: exact',
                  RSS_BOUND: 'the assembly set the process peak: "<= +X" = the solver phase added at most X',
                  RSS_PROCESS: '"(process)" = whole-process peak, imports and assembly included'}


def _windows_cpu_model():
    """ProcessorNameString of CPU 0 from the registry, None off Windows."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return str(winreg.QueryValueEx(key, 'ProcessorNameString')[0]).strip()
    except OSError:
        return None


def _cpu_model() -> str:
    """CPU model name (/proc/cpuinfo on Linux, the registry on Windows, else platform.processor())."""
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if line.lower().startswith(('model name', 'hardware', 'cpu model')):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return _windows_cpu_model() or platform.processor() or platform.machine() or 'unknown'


def _windows_core_classes():
    """{efficiency class: physical cores} from GetLogicalProcessorInformationEx (RelationProcessorCore; the highest
    class = performance cores on a hybrid CPU), None off Windows or on failure."""
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    try:
        kernel32 = ctypes.WinDLL('kernel32')
    except OSError:
        return None
    function = kernel32.GetLogicalProcessorInformationEx
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    function.restype = wintypes.BOOL
    length = wintypes.DWORD(0)
    function(0, None, ctypes.byref(length))   # fails, setting the buffer length it needs
    if length.value == 0:
        return None
    buffer = ctypes.create_string_buffer(length.value)
    if not function(0, buffer, ctypes.byref(length)):
        return None
    raw = buffer.raw[:length.value]
    classes, offset = {}, 0
    while offset + 10 <= len(raw):   # record: DWORD Relationship, DWORD Size, BYTE Flags, BYTE EfficiencyClass, ...
        size = int.from_bytes(raw[offset + 4:offset + 8], 'little')
        if size <= 0:
            break
        classes[raw[offset + 9]] = classes.get(raw[offset + 9], 0) + 1
        offset += size
    return classes or None


def _linux_physical_cores():
    """Distinct (physical id, core id) pairs of /proc/cpuinfo, None when unknown."""
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError:
        return None
    cores = set()
    for block in text.split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if 'core id' in fields:
            cores.add((fields.get('physical id', '0'), fields['core id']))
    return len(cores) or None


def machine_info(threads) -> dict:
    """CPU, cores (Windows: per efficiency class), RAM, OS, the requested thread count and library versions (open_sol,
    numpy / scipy / numba via core.time_profile.library_versions, the optional wrappers) and which host direct
    libraries are usable."""
    import open_sol
    from open_sol.core.host_memory import available_bytes, total_bytes
    from open_sol.core.time_profile import library_versions
    from open_sol.solvers.host_direct import HOST_DIRECT_LIBRARIES, host_direct_available
    versions = dict(library_versions('cpu'))
    versions['open_sol'] = open_sol.__version__
    for name in OPTIONAL_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    core_classes = _windows_core_classes()
    physical = sum(core_classes.values()) if core_classes else _linux_physical_cores()
    return {'cpu': _cpu_model(), 'machine': platform.machine(), 'platform': platform.platform(),
            'logical_cores': os.cpu_count(), 'physical_cores': physical,
            'core_classes': {str(level): count for level, count in sorted((core_classes or {}).items())},
            'ram_total_bytes': total_bytes(), 'ram_available_bytes': available_bytes(), 'threads_requested': threads,
            'versions': versions,
            'host_direct_usable': {library: host_direct_available(library) for library in HOST_DIRECT_LIBRARIES}}


def _phase_value(repeat, phase):
    phases = repeat.get('phases') or {}
    if phase == 'sweep':
        refactor, solve = phases.get('refactor'), phases.get('solve')
        return None if refactor is None or solve is None else refactor + solve
    return phases.get(phase)


def summarise(record) -> dict:
    """{phase: {'min', 'median', 'max', 'count'}} over the warm repeats (all repeats when none is warm; phases left
    None skipped), cold_total, cpu_over_wall of the warm repeats, the residual maximum and its tolerance."""
    repeats = list(record.get('repeats', ()))
    warm = [repeat for repeat in repeats if not repeat.get('cold')]
    timed = warm or repeats
    summary = {'warm_repeats': len(warm)}
    for phase in PHASES:
        values = [value for value in (_phase_value(repeat, phase) for repeat in timed) if value is not None]
        if values:
            summary[phase] = {'min': min(values), 'median': statistics.median(values), 'max': max(values),
                              'count': len(values)}
    cold = [repeat for repeat in repeats if repeat.get('cold')]
    if cold and warm:
        summary['cold_total'] = _phase_value(cold[0], 'total')
    wall = sum(repeat.get('wall_seconds') or 0.0 for repeat in timed)
    cpu = [repeat.get('cpu_seconds') for repeat in timed if repeat.get('cpu_seconds') is not None]
    if cpu and wall > 0.0:
        summary['cpu_over_wall'] = sum(cpu) / wall
    residuals = [repeat.get('residual') for repeat in repeats if repeat.get('residual') is not None]
    if residuals:
        summary['residual_max'] = max(residuals)
    tolerances = [repeat.get('tolerance') for repeat in repeats if repeat.get('tolerance') is not None]
    if tolerances:
        summary['tolerance'] = tolerances[-1]
    return summary


def eigen_difference(record, reference):
    """Max over the reference resonances of min |omega - omega_ref| / |omega_ref| among the record's resonances (last
    repeat of each); None when either found none."""
    def omegas(entry):
        repeats = entry.get('repeats') or ()
        if not repeats:
            return []
        return [complex(re, im) for re, im in repeats[-1]['details'].get('eigenvalues', ())]
    found, wanted = omegas(record), omegas(reference)
    if not found or not wanted:
        return None
    return max(min(abs(omega - target) for omega in found) / abs(target) for target in wanted)


def derived_difference(record, reference):
    """Max over the derived values both model runs report (last repeat of each) of |v - v_ref| / |v_ref| (absolute
    when v_ref is 0); None when they share none."""
    def derived(entry):
        repeats = entry.get('repeats') or ()
        return (repeats[-1].get('details') or {}).get('derived') or {} if repeats else {}
    found, wanted = derived(record), derived(reference)
    differences = []
    for name, target in wanted.items():
        value = found.get(name)
        if isinstance(value, (int, float)) and isinstance(target, (int, float)):
            differences.append(abs(value - target) / abs(target) if target else abs(value))
    return max(differences) if differences else None


def add_gpu(machine, records) -> None:
    """machine['gpu']: the device the first gpu child reported ({'name', 'total_bytes'}), None without gpu runs."""
    machine['gpu'] = next((record['gpu'] for record in records if record.get('gpu')), None)


def _seconds(value) -> str:
    return '-' if value is None else f'{value:.3g} s'


def _ratio(value) -> str:
    """Speed-up text: '160x', '3.5x', '0.0089x'."""
    return f'{value:.0f}x' if value >= 10.0 else f'{value:.2g}x'


def _mib(value) -> str:
    mib = value / MIB
    return f'{mib:.0f} MiB' if mib >= 10.0 else f'{mib:.2g} MiB'


def _median(record, phase):
    return (record or {}).get('summary', {}).get(phase, {}).get('median')


def spread_text(record, phase) -> str:
    """' +-4%' = half the min-max range of the warm repeats in % of the median ('' with one repeat)."""
    entry = record.get('summary', {}).get(phase)
    if not entry or entry['count'] < 2 or entry['median'] <= 0.0:
        return ''
    return f" +-{50.0 * (entry['max'] - entry['min']) / entry['median']:.0f}%"


def _last_details(record) -> dict:
    return ((record.get('repeats') or [{}])[-1]).get('details') or {}


def _time_cell(record, baseline) -> str:
    status = record.get('status')
    total = _median(record, 'total')
    if total is None:
        return status or '-'
    text = _seconds(total) + spread_text(record, 'total')
    reference = _median(baseline, 'total')
    if reference and total > 0.0:
        text += f', {_ratio(reference / total)}'
    sweep = _median(record, 'sweep')
    if sweep is not None:
        text += f'; sweep {_seconds(sweep)}'
        reference_sweep = _median(baseline, 'sweep')
        if reference_sweep and sweep > 0.0:
            text += f', {_ratio(reference_sweep / sweep)}'
    if status != 'ok':
        text += f' ({status})'
    return text


def _memory_cell(record) -> str:
    parts = []
    if record.get('rss_solver_bytes') is not None:
        parts.append(f"+{_mib(record['rss_solver_bytes'])}")
    elif record.get('rss_solver_bound_bytes') is not None:
        parts.append(f"<= +{_mib(record['rss_solver_bound_bytes'])}")
    elif record.get('rss_peak_bytes') is not None:
        parts.append(f"{_mib(record['rss_peak_bytes'])} (process)")
    details = _last_details(record)
    if details.get('factor_bytes'):
        parts.append(f"factor {_mib(details['factor_bytes'])}")
    if details.get('device_bytes'):
        parts.append(f"device factor {_mib(details['device_bytes'])}")
    if record.get('backend') == 'gpu' and details.get('gpu_peak_bytes'):
        parts.append(f"gpu peak {_mib(details['gpu_peak_bytes'])}")
    return '; '.join(parts) or '-'


def _work_text(record) -> str:
    """'; work a=1, b=2' of the profiled physics' solver_work counters of a model run (at most 4), '' otherwise."""
    work = (_last_details(record).get('solver_work') or {}).get(record.get('physics')) or {}
    items = [f'{name}={value}' for name, value in work.items() if isinstance(value, (int, float))][:4]
    return f"; work {', '.join(items)}" if items else ''


def _detail_cell(record) -> str:
    summary = record.get('summary', {})
    if _median(record, 'total') is None:
        reason = ' '.join((record.get('reason') or '').split()).replace('|', '/')   # one table cell
        return (reason[:60] + '...') if len(reason) > 63 else (reason or '-')
    parts = []
    for phase, short in (('analysis', 'a'), ('factor', 'f'), ('solve', 's'), ('refactor', 'rf'), ('build', 'b'),
                         ('study', 'st'), ('output', 'o')):
        value = _median(record, phase)
        if value is not None:
            parts.append(f'{short} {value:.3g}')
    text = ' / '.join(parts)
    accuracy, tolerance = summary.get('residual_max'), summary.get('tolerance')
    if accuracy is not None:
        text += f'; r {accuracy:.1e}'
        if tolerance is not None and accuracy > ACCURACY_MARGIN * tolerance:
            text += f' > tol {tolerance:.0e}'
    details = _last_details(record)
    if details.get('residual_own_operator') is not None:
        text += f"; own {details['residual_own_operator']:.1e}"
    difference = record.get('eigen_difference')
    if difference is not None:
        text += f'; dw {difference:.1e}'
    if details.get('reference_error') is not None:
        text += f"; ex {details['reference_error']:.1e}"
    if record.get('derived_difference') is not None:
        text += f"; dd {record['derived_difference']:.1e}"
    errors = [value for value in (details.get('derived_error') or {}).values() if isinstance(value, (int, float))]
    if errors:
        text += f'; err {max(errors):.1e}'
    text += _work_text(record)
    if 'iterations' in details:
        text += f"; it {details['iterations']}"
    if details.get('operator_applications') is not None:
        text += f"; op {details['operator_applications']}"
    if details.get('preconditioner_applications') is not None:
        text += f"; pc {details['preconditioner_applications']}"
    if details.get('solve_refinements'):
        text += f"; gm {details['solve_refinements']}"
    if summary.get('cold_total') is not None:
        text += f"; cold {summary['cold_total']:.3g}"
    if summary.get('cpu_over_wall') is not None:
        text += f"; cpu/wall {summary['cpu_over_wall']:.2g}"
    return text


def _machine_line(machine) -> str:
    versions = machine['versions']
    ram = (machine['ram_total_bytes'] or 0) / 2 ** 30
    threads = machine['threads_requested'] or 'library defaults'
    physical = machine.get('physical_cores')
    cores = f"{machine['logical_cores']} logical" + (f' / {physical} physical' if physical else '') + ' cores'
    classes = machine.get('core_classes') or {}
    if len(classes) == 2:   # hybrid: the higher efficiency class is the performance cores
        efficiency, performance = (classes[key] for key in sorted(classes, key=int))
        cores += f' ({performance} performance + {efficiency} efficiency)'
    gpu = machine.get('gpu')
    gpu_text = f", GPU {gpu['name']} ({gpu['total_bytes'] / 2 ** 30:.1f} GiB)" if gpu else ''
    return (f"{machine['cpu']}, {cores}, {ram:.1f} GiB RAM{gpu_text}, threads: {threads}; open_sol "
            f"{versions.get('open_sol')}, numpy {versions.get('numpy')}, scipy {versions.get('scipy')}, python-mumps "
            f"{versions.get('python-mumps')}, pydiso {versions.get('pydiso')}, cupy "
            f"{versions.get('cupy') or versions.get('cupy-cuda12x')}, "
            f"nvmath-python {versions.get('nvmath-python')}")


def _legend(records) -> list:
    methods = sorted({record.get('rss_method') for record in records if record.get('rss_method') in MEMORY_METHODS})
    memory = '; '.join(MEMORY_METHODS[method] for method in methods) or 'not measured'
    lines = ['Time cells: median total (factor + solve; eigen: whole solve; model: the whole api.run) over the warm '
             'repeats, +-half the min-max range in % of the median, speed-up vs the baseline (host SuperLU on the cpu '
             'backend, else the cpu Automatic); sweep = refactor + solve (one more point of a sweep on the same '
             'pattern) and its speed-up. (auto) = what Automatic resolves to on that backend; @gpu = the gpu backend '
             '(operator and rhs on the device, every phase timed after a device synchronise).', '',
             f'Memory cells: +X = host RSS the solver phase added ({memory}; device memory not included); factor = '
             'the host factor size the library reports, device factor = a cuDSS factor (outside the cupy pool), gpu '
             'peak = the build ledger peak of a gpu model run (a gpu vibration-mode problem: the cupy pool high water '
             'of the eigen solve over its live arrays at the start).', '',
             'Phase cells (median seconds): a analysis, f factor / set-up (analysis included), s solve, rf value-only '
             'refactorisation; r max relative residual (eigen: max eigen-pair residual), "> tol" when above '
             f'{ACCURACY_MARGIN:g}x the tolerance (status inaccurate), own = the residual against the solver\'s own '
             'float32 operator, dw max |omega - omega_baseline| / |omega| (the baseline: SuperLU, or the cpu '
             'Automatic), ex max relative omega error against the exact discrete modes (vibration-mode problems), it '
             'CG iterations (vibration modes: arpack its inner CG iterations, block_eigen its block iterations), op '
             'operator applications (arpack: shifted-inverse applications, i.e. factor or inner PCG solves; '
             'block_eigen: stiffness products of block rows), pc preconditioner applications (block_eigen), gm GMRES '
             'refinement iterations of the solve, cold = total of the first (cold) repeat, cpu/wall = process CPU over '
             'wall time of the warm repeats (about the threads that worked).', '']
    if any(record.get('kind') == 'model' for record in records):
        lines += ['Model rows: b build (every build of the study: mesh, materials, assembly, solver set-up), st study '
                  '(the solves and everything between them), o output (validation, results written); dd max relative '
                  'difference of the derived values from the baseline run, err max derived_error against the '
                  "model's analytic references, work = the physics' solver_work counters (summary.json).", '']
    return lines


def _column_header(problem_class, size, records) -> str:
    """'n = 64 (4,096 unknowns)' of a synthetic size, 'as written' / 'NAME = v' of a model column (+ its cells)."""
    if problem_class.kind != 'model':
        unknowns = next((record['n_unknowns'] for record in records if record.get('n_unknowns')), None)
        return f'n = {size} ({unknowns:,} unknowns)' if unknowns else f'n = {size}'
    cells = next((_last_details(record).get('n_cells') for record in records if _last_details(record).get('n_cells')),
                 None)
    text = 'as written' if size is None else f'{problem_class.scan_parameter} = {size}'
    return text + (f' ({cells:,} cells)' if isinstance(cells, int) else '')


def markdown(problem_classes, candidates_by_problem, records, sizes_by_problem, machine) -> str:
    """Markdown report: machine line, legend, then per problem a time / speed-up, a memory and a phase / accuracy
    table (rows solvers, baseline first, Automatic marked; columns sizes with their unknown counts)."""
    lines = ['# Solver profile', '', _machine_line(machine), ''] + _legend(records)
    for name, problem_class in problem_classes.items():
        sizes = sizes_by_problem[name]
        by_key = {(record['row'], record['size']): record for record in records if record['problem'] == name}
        header = ' | '.join(_column_header(problem_class, size, [record for record in by_key.values()
                                                                 if record['size'] == size]) for size in sizes)
        columns = 'model' if problem_class.kind == 'model' else 'n'
        lines += [f'## {name}', '', problem_class.description, '',
                  f'physics {problem_class.physics_type}, role {problem_class.role}, study {problem_class.study}, '
                  f'grid {problem_class.grid}; {columns} = {problem_class.size_text}', '']
        rows = list(candidates_by_problem[name])
        baseline_row = next((candidate.row for candidate in rows if candidate.baseline), None)
        for title, cell in (('Time and speed-up', 'time'), ('Memory', 'memory'), ('Phases and accuracy', 'detail')):
            lines += [f'**{title}**', '', f'| solver | {header} |', '|---|' + '---|' * len(sizes)]
            for candidate in rows:
                label = candidate.row + (' (auto)' if candidate.automatic else '')
                cells = []
                for size in sizes:
                    record = by_key.get((candidate.row, size))
                    if record is None:
                        cells.append('-')
                    elif cell == 'time':
                        cells.append(_time_cell(record, by_key.get((baseline_row, size))))
                    elif cell == 'memory':
                        cells.append(_memory_cell(record))
                    else:
                        cells.append(_detail_cell(record))
                lines.append(f"| {label} | {' | '.join(cells)} |")
            lines.append('')
    return '\n'.join(lines) + '\n'

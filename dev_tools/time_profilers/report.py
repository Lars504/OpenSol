"""Machine record, per-record summaries (min / median per phase, eigenvalue difference against SuperLU) and the
markdown comparison tables of the CPU solver profiler."""
import importlib.metadata
import os
import platform
import statistics

# optional packages whose versions go into the machine record (besides time_profile.library_versions)
OPTIONAL_DISTRIBUTIONS = ('python-mumps', 'pydiso', 'mkl', 'threadpoolctl', 'numba')
PHASES = ('analysis', 'factor', 'solve', 'refactor', 'total')
MIB = 2 ** 20


def _cpu_model() -> str:
    """CPU model name (/proc/cpuinfo on Linux, else platform.processor())."""
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if line.lower().startswith(('model name', 'hardware', 'cpu model')):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or 'unknown'


def _physical_cores():
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
    """CPU, cores, RAM, OS, the requested thread count and library versions (open_sol, numpy / scipy / numba via
    core.time_profile.library_versions, the optional wrappers) and which host direct libraries are usable."""
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
    return {'cpu': _cpu_model(), 'machine': platform.machine(), 'platform': platform.platform(),
            'logical_cores': os.cpu_count(), 'physical_cores': _physical_cores(), 'ram_total_bytes': total_bytes(),
            'ram_available_bytes': available_bytes(), 'threads_requested': threads, 'versions': versions,
            'host_direct_usable': {library: host_direct_available(library) for library in HOST_DIRECT_LIBRARIES}}


def summarise(record) -> dict:
    """{phase: {'min', 'median'}} over the record's repeats (phases a repeat left None are skipped), the residual
    maximum and the per-repeat totals."""
    summary = {}
    for phase in PHASES:
        values = [repeat['phases'].get(phase) for repeat in record.get('repeats', ())]
        values = [value for value in values if value is not None]
        if values:
            summary[phase] = {'min': min(values), 'median': statistics.median(values)}
    residuals = [repeat.get('residual') for repeat in record.get('repeats', ()) if repeat.get('residual') is not None]
    if residuals:
        summary['residual_max'] = max(residuals)
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


def _seconds(value) -> str:
    return '-' if value is None else f'{value:.3g} s'


def _memory(record) -> str:
    solver_bytes = record.get('rss_solver_bytes')
    peak = solver_bytes if solver_bytes is not None else record.get('rss_peak_bytes')
    if peak is None:
        return '? MiB'
    return f"{peak / MIB:.0f} MiB{'' if solver_bytes is not None else ' (process)'}"


def _median(record, phase):
    return record.get('summary', {}).get(phase, {}).get('median')


def _time_cell(record, baseline) -> str:
    status = record.get('status')
    total = _median(record, 'total')
    if total is None:
        return status or '-'
    text = f'{_seconds(total)}, {_memory(record)}'
    reference = None if baseline is None else _median(baseline, 'total')
    if reference and total > 0.0:
        text += f', {reference / total:.2g}x'
    if status != 'ok':
        text += f' ({status})'
    return text


def _detail_cell(record) -> str:
    total = _median(record, 'total')
    if total is None:
        reason = ' '.join((record.get('reason') or '').split()).replace('|', '/')   # one table cell
        return (reason[:60] + '...') if len(reason) > 63 else (reason or '-')
    parts = []
    for phase, short in (('analysis', 'a'), ('factor', 'f'), ('solve', 's'), ('refactor', 'rf')):
        value = _median(record, phase)
        if value is not None:
            parts.append(f'{short} {value:.3g}')
    accuracy = record.get('summary', {}).get('residual_max')
    text = ' / '.join(parts) + (f'; r {accuracy:.1e}' if accuracy is not None else '')
    difference = record.get('eigen_difference')
    if difference is not None:
        text += f'; dw {difference:.1e}'
    details = (record.get('repeats') or [{}])[-1].get('details', {})
    if 'iterations' in details:
        text += f"; it {details['iterations']}"
    if details.get('solve_refinements'):
        text += f"; gm {details['solve_refinements']}"
    return text


def _machine_line(machine) -> str:
    versions = machine['versions']
    ram = (machine['ram_total_bytes'] or 0) / 2 ** 30
    threads = machine['threads_requested'] or 'library defaults'
    return (f"{machine['cpu']}, {machine['logical_cores']} logical / {machine['physical_cores']} physical cores, "
            f'{ram:.1f} GiB RAM, threads: {threads}; open_sol {versions.get("open_sol")}, numpy '
            f'{versions.get("numpy")}, scipy {versions.get("scipy")}, python-mumps {versions.get("python-mumps")}, '
            f'pydiso {versions.get("pydiso")}')


def markdown(problem_classes, candidates_by_problem, records, sizes_by_problem, machine) -> str:
    """Markdown report: machine line, then per problem a time / memory / speed-up table and a phase / accuracy table
    (rows solvers, baseline first, Automatic marked; columns sizes with their unknown counts)."""
    lines = ['# CPU solver profile', '', _machine_line(machine), '',
             'Time cells: median total (factor + solve; eigen: whole solve) over the repeats, peak RSS the solver '
             'phase added (Linux VmHWM; "(process)": whole-process peak), speed-up vs host SuperLU. Phase cells '
             '(median seconds): a analysis, f factor / set-up (analysis included), s solve, rf value-only '
             'refactorisation; r max relative residual (eigen: max eigen-pair residual), dw max |omega - '
             'omega_SuperLU| / |omega|, it CG iterations, gm GMRES refinement iterations of the solve. '
             '(auto) = what Automatic resolves to on the CPU backend.', '']
    for name, problem_class in problem_classes.items():
        sizes = sizes_by_problem[name]
        by_key = {(record['solver'], record['size']): record for record in records if record['problem'] == name}
        unknowns = {record['size']: record['n_unknowns'] for record in by_key.values() if record.get('n_unknowns')}
        header = ' | '.join(f'n = {size} ({unknowns[size]:,} unknowns)' if size in unknowns else f'n = {size}'
                            for size in sizes)
        lines += [f'## {name}', '', problem_class.description, '',
                  f'physics {problem_class.physics_type}, role {problem_class.role}, study {problem_class.study}, '
                  f'grid {problem_class.grid}; n = {problem_class.size_text}', '']
        rows = list(candidates_by_problem[name])
        baseline_label = next((candidate.label for candidate in rows if candidate.baseline), None)
        for title, cell in (('Time, memory, speed-up', 'time'), ('Phases and accuracy', 'detail')):
            lines += [f'**{title}**', '', f'| solver | {header} |', '|---|' + '---|' * len(sizes)]
            for candidate in rows:
                label = candidate.label + (' (auto)' if candidate.automatic else '')
                cells = []
                for size in sizes:
                    record = by_key.get((candidate.label, size))
                    if record is None:
                        cells.append('-')
                    elif cell == 'time':
                        cells.append(_time_cell(record, by_key.get((baseline_label, size))))
                    else:
                        cells.append(_detail_cell(record))
                lines.append(f"| {label} | {' | '.join(cells)} |")
            lines.append('')
    return '\n'.join(lines) + '\n'

"""Run every profiler_solver_*.py command in turn (a failing one is logged, the rest continue) -> out/run_all.log
with each command's exit code, wall time and report path; each report is also kept in out/history/<timestamp>/.
Run (from any folder): python run_all_profilers.py [--only SUBSTRING ...]"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]

COMMANDS = (
    'pressure_acoustics', 'pressure_acoustics_tetrahedra', 'elastic_waves', 'laminar_flow',
    'em_fdfd_ez', 'em_fdfd_hz', 'em_fdfd_triangles', 'em_fdfd_nonlinear', 'em_modes', 'em_modes_triangles',
    'em_fdfd_bor', 'em_fdfd_bor_eigen', 'em_fdfd_bor_triangles', 'em_fdfd_conical', 'em_fdfd_conical_triangles',
    'em_fdtd_bor', 'em_fdtd_ez', 'em_fdtd_hz',
    'em_fdtd_3d', 'em_fdfd_3d', 'em_fdfd_3d_tetrahedra', 'em_eme', 'em_eme_3d',
    'heat --kind 2d', 'heat --kind 3d', 'heat --kind steady',
    'heat_fem --kind triangles', 'heat_fem --kind tetrahedra', 'heat_fem --kind steady', 'heat_noise',
    'potential --kind tensor', 'potential --kind electrolyte', 'potential --kind waveform',
    'potential --kind triangles', 'potential --kind tetrahedra',
    'magnetostatics --kind tensor', 'magnetostatics --kind triangles', 'magnetostatics --kind linear',
    'magnetic_fields_ac --kind tensor', 'magnetic_fields_ac --kind triangles', 'magnetic_fields_ac --kind strip',
    'magnetostatics_3d --kind tensor', 'magnetostatics_3d --kind tetrahedra', 'magnetostatics_3d --kind sphere',
    'magnetostatics_3d --kind coil',
    'drift_diffusion --kind tensor', 'drift_diffusion --kind triangles',
    'solid_mechanics_tensor', 'solid_mechanics_triangles', 'solid_mechanics_3d_tensor --warmup',
    'solid_mechanics_3d_tetrahedra', 'solid_mechanics_3d_nonlinear',
    'membrane_film', 'membrane_plate', 'membrane_line',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='*', default=None, help='run only commands containing one of these')
    options = parser.parse_args()
    commands = [entry for entry in COMMANDS if not options.only or any(text in entry for text in options.only)]
    log_path = HERE / 'out' / 'run_all.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_folder = HERE / 'out' / 'history' / time.strftime('%Y-%m-%d_%H%M%S')
    history_folder.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")}: {len(commands)} commands, '
                  f'reports kept in {history_folder} =====\n')
        for entry in commands:
            script, *arguments = entry.split()
            command = [sys.executable, str(HERE / f'profiler_solver_{script}.py'), *arguments]
            print(f'>>> {entry}', flush=True)
            start = time.perf_counter()
            completed = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True)
            wall = time.perf_counter() - start
            report = [line for line in completed.stdout.splitlines() if line.startswith('report: ')]
            if report:
                report_path = Path(report[-1][len('report: '):].strip())
                if report_path.is_file():
                    shutil.copyfile(report_path, history_folder / f'{report_path.parent.name}.txt')
            tail = '' if completed.returncode == 0 else '\n    ' + '\n    '.join(completed.stderr.splitlines()[-8:])
            line = (f'{entry}: exit {completed.returncode}, {wall:.1f} s, '
                    f'{report[-1] if report else "no report"}{tail}')
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
    print(f'log: {log_path}')


if __name__ == '__main__':
    main()

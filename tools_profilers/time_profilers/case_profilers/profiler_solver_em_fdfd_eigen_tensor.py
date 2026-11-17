"""Profile the em_fdfd_ez / em_fdfd_hz eigenfrequency study on tensor meshes (FDFDPhysics.solve_eigenfrequencies) on
the gpu: every run on Automatic (arpack_shift_invert, a shift-invert factor), so run_all's no-flag / --cudss lines are
the factor and --time-filter (harness --solver-type time_filter, default node) the factor-free FDTD + Rayleigh-Ritz on
the same runs. Runs, each sized so the default time_filter band (+-10 %) holds its requested resonances:
cavity_ez / cavity_hz / cavity_ez_lossy (selftest em_fdfd_eigenfrequency_pec_cavity boxes, sides x 3: closed, no PML,
3 resonances; its time_filter twin needs bandwidth 0.4 on the small boxes), disk_ez / disk_hz (the
disk_resonator_eigenmodes template, ez and as em_fdfd_hz: open, CPML, 6 resonances, ~53k unknowns; the selftest
em_fdfd_time_filter_open_disc's 0.6 um disc needs bandwidth 0.15) and disk_ez_fine (the template on 3x finer cells,
~4e5 unknowns: where the factor gets expensive). --warmup runs cavity_ez and disk_hz first (cuDSS load, the ez / hz
stepper, CPML and DFT kernels). Run from any folder:
python profiler_solver_em_fdfd_eigen_tensor.py [--warmup] [--solver-type time_filter] [--runs NAME ...] [--no-cprofile]"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em2d_parts import FDFD_TENSOR, STUDY_RUNNERS, TENSOR_READS, parts  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main, refined  # noqa: E402

CAVITY_CASE = 'em_fdfd_eigenfrequency_pec_cavity'   # 40 nm PEC boxes ez (51 x 31 cells), hz (52 x 32), ez_lossy
CAVITY_SCALE = 3          # box sides x 3: ~11 cavity modes within +-10 % of the target, the 3 nearest well inside
DISK_REFINEMENT = 3       # template cells 20 nm / 3: (600 + 30)^2 ~ 4e5 unknowns


def _enlarged(spec, factor):
    """Deep copy of spec with both domain sides x factor (same cells, factor x the count per side)."""
    spec = copy.deepcopy(spec)
    spec.domain.size_x = factor * float(spec.domain.size_x)
    spec.domain.size_y = factor * float(spec.domain.size_y)
    return spec


def _hz(spec):
    """Deep copy of spec whose physics run em_fdfd_hz (the same settings and features)."""
    spec = copy.deepcopy(spec)
    for physics_spec in spec.physics:
        physics_spec.type = 'em_fdfd_hz'
    return spec


def _specs():
    """{run: ModelSpec} of the closed cavities, the open disk (ez, hz) and the refined disk, all on Automatic."""
    from open_sol.physics.em.selftests_eigenfrequency import disk_resonator_eigenmodes_spec
    cavities = case_specs(CAVITY_CASE)
    disk = disk_resonator_eigenmodes_spec()
    return {'cavity_ez': _enlarged(cavities['ez'], CAVITY_SCALE), 'cavity_hz': _enlarged(cavities['hz'], CAVITY_SCALE),
            'cavity_ez_lossy': _enlarged(cavities['ez_lossy'], CAVITY_SCALE),
            'disk_ez': disk, 'disk_hz': _hz(disk), 'disk_ez_fine': refined(disk, DISK_REFINEMENT)}


def _warmup_specs():
    """{run: ModelSpec} of --warmup: one closed ez and one open hz run (both steppers, CPML, DFT kernels, cuDSS)."""
    runs = _specs()
    return {'cavity_ez': runs['cavity_ez'], 'disk_hz': runs['disk_hz']}


TIME_FILTER_EIGEN = 'open_sol.physics.em.time_filter_eigen'
TIME_FILTER = 'open_sol.solvers.time_filter:TimeFilterEigenSolver'
# the time_filter internals (no EMPhysics.step: one sync per time step; filtered_fields' exclusive time is the stepping
# loop, whose queued kernels the next DFTMonitor.update entry sync drains into it); _ritz_block runs for the polish
# residuals and the final Ritz vectors alike, ritz_pairs' exclusive time is the polish Gram-Schmidt
TIME_FILTER_PARTS = (
    (f'{TIME_FILTER_EIGEN}:solve_time_filter_resonances', 'time filter: solve_time_filter_resonances'),
    (f'{TIME_FILTER_EIGEN}:filtered_fields', 'time filter: filtered_fields (excl: FDTD stepping loop)'),
    (f'{TIME_FILTER_EIGEN}:_source_cells', 'time filter: _source_cells'),
    (f'{TIME_FILTER_EIGEN}:_relative_residual', 'time filter: _relative_residual (host, per candidate)'),
    ('open_sol.physics.em:EMPhysics.__init__', 'time filter fdtd: EMPhysics.__init__'),
    ('open_sol.physics.em:EMPhysics.finalize', 'time filter fdtd: EMPhysics.finalize'),
    ('open_sol.physics.em.monitors:DFTMonitor.__init__', 'time filter dft: DFTMonitor.__init__ (accumulators)'),
    ('open_sol.physics.em.monitors:DFTMonitor.update', 'time filter dft: DFTMonitor.update (running DFT)'),
    (f'{TIME_FILTER}.ritz_pairs', 'time filter rr: ritz_pairs (excl: polish Gram-Schmidt)'),
    (f'{TIME_FILTER}._orthonormal_rows', 'time filter rr: _orthonormal_rows (Gram + host eigh)'),
    (f'{TIME_FILTER}._project', 'time filter rr: _project (projection + host eig)'),
    (f'{TIME_FILTER}._ritz_block', 'time filter rr: _ritz_block (polish residuals, Ritz vectors)'),
)
# the resonance filter shared with the shift-invert path (solve_resonances is in FDFD_TENSOR)
RESONANCE_FILTER = parts('em eigen', 'open_sol.physics.em.eigenfrequency', (
    '_filtered_candidates', '_ResonanceFilter.__init__', '_ResonanceFilter.measures',
)) + (('open_sol.model.study_eigenfrequency:_solve_resonance_snapshots', 'study: _solve_resonance_snapshots'),)
PARTS = (STUDY_PARTS + SOLVER_ROLE_PARTS + STUDY_RUNNERS + TENSOR_READS + FDFD_TENSOR + TIME_FILTER_PARTS
         + RESONANCE_FILTER)

if __name__ == '__main__':
    main('em_fdfd_eigen_tensor', PARTS, __doc__, case=None, specs=_specs, warmup_specs=_warmup_specs)

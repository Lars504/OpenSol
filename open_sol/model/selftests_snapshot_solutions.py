"""Snapshot solutions self-test (GUI runs: RunControl.keep_solutions): the em_fdtd_3d dipole template with snapshots
every 10 fs keeps a solution per snapshot, a restored snapshot renders its stored field bit for bit, an unstored field
on a vertical cut renders at the snapshot's time, and the live state is unchanged afterwards; registered by
study.register."""
import numpy as np

from .schema import SelfTestCase

_OUTPUT_INTERVAL = 10e-15   # s: snapshots at 0, 10, 20 fs + the final one (28 fs)
_STORED = 'em.ez'


def _run(spec):
    """run_study with snapshot solutions kept -> (session, summary); RuntimeError on a failed study."""
    from .study import RunControl, StudyFailed, StudyFinished, run_study
    events = []
    session = run_study(spec, events.append, RunControl(live_results=False, keep_solutions=True))
    failed = [event for event in events if isinstance(event, StudyFailed)]
    finished = [event for event in events if isinstance(event, StudyFinished)]
    if failed or not finished:
        if session is not None:
            from .study import release_session
            release_session(None, session)
        raise RuntimeError(f'study failed: {failed[-1].message if failed else "no StudyFinished"}')
    return session, finished[-1].summary


def _check_snapshot_solutions(_results_by_name):
    """Failed checks of the kept solutions (error = their count, tolerance 0)."""
    from ..physics.em3d.selftests import dipole_selftest_spec
    from .study import release_session
    spec = dipole_selftest_spec()
    spec.title = 'Self-test: snapshot solutions (em_fdtd_3d dipole)'
    spec.study.settings['output_interval'] = _OUTPUT_INTERVAL
    spec.study.stored_fields = [_STORED]
    session, summary = _run(spec)
    failed = []
    try:
        solutions = summary.get('snapshot_solutions') or []
        if len(solutions) != len(session.snapshots) or len(solutions) < 3:
            failed.append(f'{len(solutions)} snapshot_solutions entries for {len(session.snapshots)} snapshots')
        missing = [index for index, names in enumerate(solutions) if 'em' not in names]
        if missing:
            failed.append(f'snapshots {missing} keep no solution of em')
        x_centres = session.cut_centers('x')
        side_cut = ('x', float(x_centres[len(x_centres) // 3]))
        live_before = session.canvas_field('e_abs', 'em', plane=side_cut).values.copy()   # as snapshot_field reads
        physics, component = _STORED.split('.')
        for index, snapshot in enumerate(session.snapshots[:-1]):
            with session.restored_snapshot(index, [physics]):
                restored = session.field(component, physics).values
            if not np.array_equal(restored, snapshot.fields[_STORED].values):
                failed.append(f'snapshot {index + 1}: restored {_STORED} differs from the stored one by '
                              f'{float(np.max(np.abs(restored - snapshot.fields[_STORED].values))):.3g}')
        early = session.snapshot_field(1, 'e_abs', physics, plane=side_cut)
        if abs(early.time - session.snapshots[1].time) > 1e-21:
            failed.append(f'snapshot field time {early.time} != snapshot time {session.snapshots[1].time}')
        if np.array_equal(early.values, live_before):
            failed.append('e_abs of snapshot 2 on the x cut equals the final state (nothing restored)')
        live_after = session.canvas_field('e_abs', 'em', plane=side_cut).values
        if not np.array_equal(live_after, live_before):
            failed.append('the live state changed after the snapshot reads')
        final = session.snapshot_field(len(session.snapshots) - 1, 'e_abs', physics, plane=side_cut)
        if not np.array_equal(final.values, live_before):
            failed.append('the final snapshot (LIVE_SOLUTION) does not read the live state')
    finally:
        release_session(None, session)
    return {'measured': len(failed), 'expected': 0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': {'failed_checks': failed, 'snapshot_solutions': summary.get('snapshot_solutions')}}


SNAPSHOT_SOLUTION_SELFTEST_CASES = (
    SelfTestCase(
        name='snapshot_solutions',
        label='Snapshot solutions: any field and cut of a time snapshot (em_fdtd_3d)',
        description='The em_fdtd_3d dipole self-test model (28 fs) with snapshots every 10 fs, stored field em.ez only, '
                    'run with RunControl(keep_solutions=True) as the GUI does. Checks: every snapshot keeps a solution '
                    'of em (summary snapshot_solutions); inside Session.restored_snapshot the mid-height em.ez equals '
                    'the stored snapshot field bit for bit; Session.snapshot_field of the unstored e_abs on an x cut '
                    'carries the snapshot time and differs from the final state; the final snapshot reads the live '
                    'state; the live e_abs is unchanged after all reads. error = failed checks, tolerance 0.',
        specs=lambda: {},
        check=_check_snapshot_solutions),
)

"""Qt adapter of model.worker.BackendWorker: a 30 ms timer drains its events and latest-value slots (progress, live
output, GPU memory) and re-emits them as GUI-thread signals; build/run jobs are tracked by their final events."""
import contextlib
import queue
import sys
from collections import deque
from dataclasses import dataclass

from PySide6 import QtCore

from ..model.study import (BuildFailed, Built, LogEvent, StudyFailed, StudyFinished, StudyStarted, SweepStep,
                           SweepValueDone)
from ..model.worker import BackendWorker, DeviceInfo, TaskFailed, TaskResult

POLL_INTERVAL_MS = 30
MAX_EVENTS_PER_POLL = 500
# event that ends a submitted job of each kind (a run job also emits Built / BuildFailed on its way)
_JOB_END_EVENTS = {'build': (Built, BuildFailed), 'run': (StudyFinished, StudyFailed)}


@dataclass
class _Job:
    kind: str                # 'build' | 'run'
    spec: object             # copy of the submitted ModelSpec (results_spec once built)
    generation: int          # bridge generation at submission (advance_generation)
    started: bool = False    # run job: StudyStarted seen (pause is honoured from then on)


class WorkerBridge(QtCore.QObject):
    """GUI-thread facade of the backend worker: submits jobs and calls, re-emits every worker event as a signal and
    keeps the worker's DeviceInfo and newest MemoryStatus (CPU dataclasses; the GUI thread never queries the GPU)."""

    built = QtCore.Signal(object)             # BuildSummary
    build_failed = QtCore.Signal(object)      # BuildFailed
    progress = QtCore.Signal(object)          # Progress
    live_frame = QtCore.Signal(object)        # FieldData
    live_results = QtCore.Signal(object)      # {(physics, feature): timeseries dict}
    log = QtCore.Signal(str, str)             # level, message
    study_started = QtCore.Signal()
    study_finished = QtCore.Signal(object)    # StudyFinished
    study_failed = QtCore.Signal(str, str)    # message, traceback
    sweep_step = QtCore.Signal(object)        # SweepStep
    sweep_value_done = QtCore.Signal(object)  # SweepValueDone
    busy_changed = QtCore.Signal(bool)
    paused_changed = QtCore.Signal(bool)
    task_failed = QtCore.Signal(str)          # failure of a call() without on_error (message + traceback)
    device_info = QtCore.Signal(object)       # DeviceInfo, once at worker start-up
    memory_status = QtCore.Signal(object)     # MemoryStatus, after every job and about 1 Hz during a study

    def __init__(self, worker=None):
        super().__init__()
        self._worker = worker if worker is not None else BackendWorker()
        self._jobs = deque()                  # submitted build/run jobs not yet ended, in worker order
        self._paused = False
        self._polling = False                 # a handler running a nested event loop must not re-enter _poll
        self._callbacks = {}                  # task id -> (on_result, on_error)
        self.results_spec = None              # copy of the spec the worker session was built from
        self.last_build_summary = None        # BuildSummary of that session (Built)
        self.last_study_summary = None        # StudyFinished.summary of its last study (None: built only / failed)
        self._session_serial = None           # Built.session_serial of the session the views show (kept on failure)
        self.device = None                    # DeviceInfo of the worker's GPU, None until reported
        self.last_memory_status = None        # newest MemoryStatus, None until reported
        self.generation = 0                   # bumped by advance_generation (document replaced)
        self._event_generation = 0            # generation of the job whose events / live values go out
        self._event_handlers = {
            Built: self._on_built, BuildFailed: self._on_build_failed, StudyStarted: self._on_study_started,
            StudyFinished: self._on_study_finished, StudyFailed: self._on_study_failed, LogEvent: self._on_log,
            SweepStep: self._on_sweep_step, SweepValueDone: self.sweep_value_done.emit,
            TaskResult: self._on_task_result, TaskFailed: self._on_task_failed, DeviceInfo: self._on_device_info,
        }
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

    @property
    def busy(self) -> bool:
        """True from build()/run() until that job's final event (Built/BuildFailed, StudyFinished/StudyFailed) or
        until the worker goes idle without one."""
        return bool(self._jobs)

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def event_is_current(self) -> bool:
        """False while emitting events / live values of a job submitted before the last advance_generation."""
        return self._event_generation == self.generation

    def advance_generation(self):
        """Document replaced: jobs still queued or running keep running, but their events stop being current."""
        self.generation += 1

    # ---- jobs and study control ----
    def build(self, spec):
        """Queue a build of spec (copied on this thread); results_spec becomes a copy of it once it is built."""
        self._worker.build(spec)
        self._submit(_Job('build', spec.copy(), self.generation))

    def run(self, spec):
        """Queue a study run of spec (copied on this thread; it rebuilds first)."""
        self._worker.run(spec)
        self._submit(_Job('run', spec.copy(), self.generation))

    def pause(self):
        """Pause the running study at its next checkpoint (logged and ignored before study_started)."""
        if self._paused:
            return
        job = self._jobs[0] if self._jobs else None
        if job is None or job.kind != 'run' or not job.started:
            self.log.emit('info', 'Pause applies once a study is running (not while building)')
            return
        self._worker.pause()
        self._paused = True
        self.paused_changed.emit(True)

    def resume(self):
        """Continue a paused study."""
        if not self._paused:
            return
        self._worker.resume()
        self._paused = False
        self.paused_changed.emit(False)

    def stop(self):
        """Cancel the running or queued study (it ends with status 'cancelled'); a build is logged as not cancellable.
        Queued jobs still run to their final events."""
        with self._state_signals():
            if not self._worker.cancel() and self._jobs and self._jobs[0].kind == 'build':
                self.log.emit('info', 'A build cannot be cancelled; it runs to completion')
            if self._paused:
                self._worker.resume()
                self._paused = False

    def set_live_field(self, physics, component, max_pixels=250_000):
        """Field streamed as live_frame while a study runs."""
        self._worker.set_live_field(physics, component, max_pixels)

    def call(self, fn, on_result=None, on_error=None) -> int:
        """Run fn(session) on the worker; on_result(value) / on_error(message) run later on the GUI thread. A session
        built after the last delivered Built is refused (on_error). Returns the task id."""
        task_id = self._worker.call(fn, session_serial=self._session_serial)
        self._callbacks[task_id] = (on_result, on_error)
        return task_id

    def call_without_session(self, fn, on_result=None, on_error=None) -> int:
        """Run fn() (host work, no session: e.g. a triangle mesh preview) on the worker, between jobs or at study
        checkpoints; callbacks as call(). Returns the task id."""
        task_id = self._worker.call(fn, needs_session=False)
        self._callbacks[task_id] = (on_result, on_error)
        return task_id

    def shutdown(self, timeout=2.0) -> bool:
        """Stop polling and the worker thread; True when the thread stopped within `timeout` seconds."""
        self._poll_timer.stop()
        return self._worker.shutdown(timeout)

    # ---- polling ----
    def _poll(self):
        """Timer slot; skipped while a previous poll is still dispatching (a handler ran a nested event loop)."""
        if self._polling:
            return
        self._polling = True
        try:
            self._deliver_events()
        finally:
            self._polling = False

    def _deliver_events(self):
        """Deliver queued events in order; the latest-value slots go out just before the event ending the only
        tracked job (no later job can have set them), else after the batch."""
        worker_idle = not self._worker.busy   # read first: an idle worker has queued every final event already
        jobs_before = list(self._jobs)
        events, drained = [], False
        while not drained and len(events) < MAX_EVENTS_PER_POLL:
            try:
                events.append(self._worker.events.get_nowait())
            except queue.Empty:
                drained = True
        latest = self._worker.take_latest()
        for event in events:
            if latest and len(self._jobs) == 1 and self._ends_current_job(event):
                self._emit_latest(latest)
                latest = None
            self._dispatch(event)
            if not self._poll_timer.isActive():
                return                        # shut down by a handler: no further signals
        if latest:
            self._emit_latest(latest)
        if worker_idle and drained:
            self._retire_unended_jobs(jobs_before)

    def _emit_latest(self, latest):
        if self._jobs:                        # else the last dispatched job's generation
            self._event_generation = self._jobs[0].generation
        progress = latest.get('progress')
        if progress is not None:
            self.progress.emit(progress)
        frame = latest.get('live_frame')
        if frame is not None:
            self.live_frame.emit(frame.field)
        results = latest.get('live_results')
        if results is not None:
            self.live_results.emit(results.results)
        status = latest.get('memory')
        if status is not None:
            self.last_memory_status = status
            self.memory_status.emit(status)

    def _dispatch(self, event):
        handler = self._event_handlers.get(type(event))
        if handler is None:
            self.log.emit('error', f'unexpected worker event {type(event).__name__}: {event!r}')
            return
        if self._jobs:                        # job events belong to the oldest unended job (worker order)
            self._event_generation = self._jobs[0].generation
        handler(event)

    def _submit(self, job):
        with self._state_signals():
            self._jobs.append(job)

    def _ends_current_job(self, event) -> bool:
        return bool(self._jobs) and isinstance(event, _JOB_END_EVENTS[self._jobs[0].kind])

    def _end_job_on(self, event):
        """Retire the oldest job when `event` ends it; a finished run releases the pause, also a pause that reached
        the next queued run in the worker (it was sent after this run had already ended there)."""
        if self._ends_current_job(event):
            job = self._jobs.popleft()
            if job.kind == 'run' and self._paused:
                self._worker.resume()
                self._paused = False

    def _retire_unended_jobs(self, jobs_before):
        """Forget jobs of `jobs_before` the idle worker ended without a final event (a worker contract violation,
        logged as an error) so busy cannot stay True for good."""
        unended = [job for job in self._jobs if any(job is old_job for old_job in jobs_before)]
        if not unended:
            return
        with self._state_signals():
            self._jobs = deque(job for job in self._jobs if all(job is not old_job for old_job in unended))
            self._paused = False              # jobs left were submitted during this poll: none has started
        self.log.emit('error', 'The backend went idle without a final event for the '
                               f'{", ".join(job.kind for job in unended)} job; its results may be missing')

    @contextlib.contextmanager
    def _state_signals(self):
        """Emit busy_changed / paused_changed after the block when it changed them."""
        was_busy, was_paused = self.busy, self._paused
        try:
            yield
        finally:
            if self.busy != was_busy:
                self.busy_changed.emit(self.busy)
            if self._paused != was_paused:
                self.paused_changed.emit(self._paused)

    # ---- event handlers ----
    def _on_built(self, event):
        self.results_spec = self._jobs[0].spec if self._jobs else None
        self._session_serial = event.session_serial
        self.last_build_summary, self.last_study_summary = event.summary, None
        with self._state_signals():
            self._end_job_on(event)
            self.built.emit(event.summary)

    def _on_build_failed(self, event):
        self.results_spec = None              # the worker dropped its session before building
        self.last_study_summary = None
        with self._state_signals():
            self._end_job_on(event)
            self.build_failed.emit(event)

    def _on_study_started(self, event):
        if self._jobs and self._jobs[0].kind == 'run':
            self._jobs[0].started = True
        self.study_started.emit()

    def _on_study_finished(self, event):
        if not (event.summary or {}).get('session_live', True):
            self.results_spec = None          # a sweep's last value failed: the worker holds no session
        self.last_study_summary = event.summary if self.results_spec is not None else None
        with self._state_signals():
            self._end_job_on(event)
            self.study_finished.emit(event)

    def _on_study_failed(self, event):
        if not event.session_live:
            self.results_spec = None          # closed (OOM) or never built / released: the worker holds no session
        self.last_study_summary = None
        with self._state_signals():
            self._end_job_on(event)
            self.study_failed.emit(str(event.message), str(event.traceback or ''))

    def _on_log(self, event):
        self.log.emit(str(event.level), str(event.message))

    def _on_sweep_step(self, event):
        self.sweep_step.emit(event)

    def _on_device_info(self, event):
        self.device = event
        self.device_info.emit(event)

    def _on_task_result(self, event):
        on_result, _ = self._callbacks.pop(event.task_id, (None, None))
        if on_result is not None:
            _invoke(on_result, event.value)

    def _on_task_failed(self, event):
        _, on_error = self._callbacks.pop(event.task_id, (None, None))
        if on_error is not None:
            _invoke(on_error, event.message)
        else:
            details = f'{event.message}\n{event.traceback}' if event.traceback else event.message
            self.task_failed.emit(details)


def _invoke(callback, argument):
    """Call a GUI callback; an exception goes to sys.excepthook so the remaining events are still delivered."""
    try:
        callback(argument)
    except Exception:
        sys.excepthook(*sys.exc_info())

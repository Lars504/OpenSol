"""Backend worker: one daemon thread owns the Session and every GPU call (budget, DeviceInfo, MemoryStatus slot); the
GUI talks to it through queues only, and constructing it or calling its public methods makes no GPU call."""
import collections
import dataclasses
import itertools
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass

from ..core import gpu_memory
from ..core.backend import gpu_available
from ..core.gpu_memory import DeviceInfo, MemoryStatus
from ..solvers.cudss_factor import remember_cudss_orderings
from .meshing import remember_element_meshes
from .schema import Issue, solver_backend
from .study import (
    BuildFailed, Built, Cancelled, LiveFrame, LiveResults, LogEvent, Progress, RunControl, StudyFailed,
    StudyFinished, StudyStarted, SweepStep, SweepValueDone, build_session, release_session, run_study,
)

__all__ = [
    'BackendWorker', 'TaskResult', 'TaskFailed',
    'Progress', 'LiveFrame', 'LiveResults', 'LogEvent', 'SweepStep', 'SweepValueDone', 'Built', 'BuildFailed',
    'StudyStarted',
    'StudyFinished', 'StudyFailed', 'Cancelled', 'RunControl', 'DeviceInfo', 'MemoryStatus',
]

_MEMORY_STATUS_PERIOD = 1.0   # s of wall time between MemoryStatus updates at study checkpoints
_BUDGET_LOG_CHANGE = 0.02     # relative change of the per-job GPU budget against the last logged one that is logged


@dataclass
class TaskResult:
    """Return value of a call(fn) task (CPU data only)."""
    task_id: int
    value: object


@dataclass
class TaskFailed:
    """A call(fn) task raised, or no session existed to run it on."""
    task_id: int
    message: str
    traceback: str


# Event type -> take_latest() slot (overwritten, not queued)
_LATEST_SLOTS = {Progress: 'progress', LiveFrame: 'live_frame', LiveResults: 'live_results', MemoryStatus: 'memory'}


class BackendWorker:
    """One daemon thread owns every GPU call; the GUI talks to it through queues only."""

    def __init__(self):
        self.events = queue.Queue()            # discrete events (everything except the latest slots)
        self.control = RunControl()
        self.session = None                    # worker-thread use only
        self._session_serial = 0               # number of the newest session (worker thread; reported in Built)
        self._condition = threading.Condition()
        self._jobs = collections.deque()       # ('build' | 'run', spec copy)
        self._calls = collections.deque()      # (task_id, fn, expected session serial or None, needs_session)
        self._task_ids = itertools.count(1)
        self._running_job = None               # kind of the job being executed, None between jobs
        self._cancel_next_run = False          # cancel() hit a queued run: cancel it once it is popped
        self._stopping = False
        self._latest = {}
        self._latest_lock = threading.Lock()
        self._last_memory_status = -math.inf   # time.monotonic() of the last MemoryStatus (worker thread)
        self._memory_status_failed = False     # a failing memory_status() was logged (once)
        self._logged_budget = None             # GPU budget [bytes] of the last budget log line (worker thread)
        self._device_kind = 'gpu'              # DeviceInfo.kind of _start_device: 'cpu' without a usable GPU
        self._thread = threading.Thread(target=self._serve, name='open_sol-backend', daemon=True)
        self._thread.start()

    @property
    def busy(self):
        """True while a build/run job runs or is queued (call tasks do not count)."""
        with self._condition:
            return self._running_job is not None or bool(self._jobs)

    def take_latest(self):
        """{'progress' | 'live_frame' | 'live_results' | 'memory': latest Progress / LiveFrame / LiveResults /
        MemoryStatus}, only the slots updated since the previous call."""
        with self._latest_lock:
            latest, self._latest = self._latest, {}
        return latest

    def build(self, spec):
        """Queue a build of a copy of `spec` (taken now); the job closes the previous session first."""
        self._submit('build', spec.copy())

    def run(self, spec):
        """Queue a study run of a copy of `spec` (taken now): control.reset(), then run_study."""
        self._submit('run', spec.copy())

    def call(self, fn, session_serial=None, needs_session=True):
        """Queue fn(session) -> CPU-only value (cupy data fails the task; between jobs and at study checkpoints); with
        session_serial (from Built) it fails instead of running on a later session; needs_session False runs fn()
        without one (host work off the GUI thread, e.g. a triangle mesh preview). Returns the TaskResult / TaskFailed
        id."""
        if not callable(fn):
            raise TypeError(f'call needs a callable fn(session), got {type(fn).__name__}')
        with self._condition:
            self._require_running()
            task_id = next(self._task_ids)
            self._calls.append((task_id, fn, session_serial, needs_session))
            self._condition.notify()
        return task_id

    def pause(self):
        with self._condition:
            if self._running_job == 'run':
                self.control.pause()

    def resume(self):
        with self._condition:
            if self._running_job == 'run':
                self.control.resume()

    def cancel(self):
        """Cancel the running study, else the next queued one (it ends 'cancelled' at its first checkpoint); a build
        cannot be cancelled. Queued jobs still run to their final events. Returns False when no run was affected."""
        with self._condition:
            if self._running_job == 'run':
                self.control.cancel()
                return True
            if any(kind == 'run' for kind, _ in self._jobs):
                self._cancel_next_run = True
                return True
            return False

    def set_live_field(self, physics, component, max_pixels=250_000):
        self.control.set_live_field(physics, component, max_pixels)

    def shutdown(self, timeout=2.0):
        """Cancel the study, drop queued jobs and calls, join the thread; True when it stopped within timeout."""
        with self._condition:
            self._stopping = True
            self._jobs.clear()
            self._calls.clear()
            self._cancel_next_run = False
            self.control.cancel()
            self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _submit(self, kind, spec):
        with self._condition:
            self._require_running()
            self._jobs.append((kind, spec))
            self._condition.notify()

    def _require_running(self):
        """Raise once shutdown() was called (caller holds the condition)."""
        if self._stopping:
            raise RuntimeError('the backend worker has been shut down')

    # ---- worker thread ----
    def _serve(self):
        """Thread loop: GPU start-up, then pending calls first, then the next job, until shutdown; closes the
        session at exit."""
        remember_element_meshes()   # a Build mesh preview or build meshes once for the next build
        remember_cudss_orderings()   # a re-run of the same mesh skips cuDSS's host reordering (cudss_factor)
        self._start_device()
        while True:
            with self._condition:
                while not (self._stopping or self._jobs or self._calls):
                    self._condition.wait()
                if self._stopping:
                    break
                job = None if self._calls else self._jobs.popleft()
                if job is not None:
                    self._running_job = job[0]
                    if job[0] == 'run':
                        self.control.reset()
                        if self._cancel_next_run:
                            self._cancel_next_run = False
                            self.control.cancel()
            if job is None:
                if self._service_calls():
                    self._emit_memory_status(force=True)
                continue
            try:
                self._run_job(*job)
            finally:
                with self._condition:
                    self._running_job = None
                    self.control.reset()
            self._emit_memory_status(force=True)
        try:
            release_session(self._set_session, self.session, self._emit)
        except Exception as exc:   # shutting down: log only
            self._emit(LogEvent('error', f'freeing GPU memory at shutdown: {type(exc).__name__}: {exc}'))

    def _start_device(self):
        """First thing on the thread: DeviceInfo event, GPU memory budget, first MemoryStatus and one log line;
        a failure is logged (every job recomputes the budget and fails without one). Without a usable GPU (no cupy or
        no CUDA device): DeviceInfo kind 'cpu' (host RAM), host MemoryStatus, no budget."""
        if not gpu_available():
            self._device_kind = 'cpu'
            device = gpu_memory.cpu_device_info()
            self._emit(device)
            self._emit_memory_status(force=True)
            self._emit(LogEvent('info', f'No usable GPU: models run on the CPU backend; '
                                        f'{gpu_memory.format_bytes(device.free_bytes)} of '
                                        f'{gpu_memory.format_bytes(device.total_bytes)} host memory available'))
            return
        try:
            device = gpu_memory.device_info()
            self._emit(device)
            budget = gpu_memory.set_budget()
            self._logged_budget = budget
            self._emit_memory_status(force=True)
            self._emit(LogEvent('info', f'GPU {device.name}: {gpu_memory.format_bytes(device.total_bytes)} total, '
                                        f'{gpu_memory.format_bytes(device.free_bytes)} free; memory budget '
                                        f'{gpu_memory.format_bytes(budget)}'))
        except Exception as exc:
            self._emit(LogEvent('error', f'GPU start-up: {type(exc).__name__}: {exc}'))

    def _run_job(self, kind, spec):
        """Close the previous session, recompute the GPU budget, then execute one queued job; a budget failure and
        errors escaping run_study / build_session become BuildFailed / StudyFailed."""
        failure = None   # (message, traceback text, out_of_memory): the exception itself is not kept
        issues = []
        try:
            release_session(self._set_session, self.session, self._emit)
            budget_error = self._refresh_budget() if solver_backend(spec) == 'gpu' else None   # cpu: host RAM only
            if budget_error is not None:
                failure, issues = (budget_error, '', False), [Issue('error', ('model',), budget_error)]
            elif kind == 'build':
                build_session(spec, self._emit, self._set_session)
            else:
                run_study(spec, self._emit, self.control, service=self._checkpoint_service,
                          on_session=self._set_session)
        except Exception as exc:
            failure = (f'{type(exc).__name__}: {exc}', traceback.format_exc(), gpu_memory.is_out_of_memory(exc))
        if failure is not None:
            message, failure_traceback, out_of_memory = failure
            if kind == 'build' or issues:   # a run job reports the budget failure as its build's too
                self._emit(BuildFailed(message, failure_traceback, issues, out_of_memory=out_of_memory))
            if kind == 'run':   # release_session ran on_session(None) before any failure it raised
                self._emit(StudyFailed(message, failure_traceback, out_of_memory=out_of_memory,
                                       session_live=self.session is not None))

    def _refresh_budget(self):
        """gpu_memory.set_budget() between jobs (the pool is nearly empty after release_session, so free + reserved
        is the current headroom); logs a noticeable change; returns the failure text, else None."""
        try:
            budget = gpu_memory.set_budget()
        except Exception as exc:   # the job must not run without a pool limit (it would spill into system memory)
            return f'GPU memory budget not set: {type(exc).__name__}: {exc}'
        if self._logged_budget is None or abs(budget - self._logged_budget) > _BUDGET_LOG_CHANGE * self._logged_budget:
            self._logged_budget = budget
            self._emit(LogEvent('info', f'GPU memory budget {gpu_memory.format_bytes(budget)}'))
        return None

    def _service_calls(self):
        """Run the calls queued so far against the current session; returns how many ran."""
        with self._condition:
            pending = list(self._calls)
            self._calls.clear()
        for task_id, fn, session_serial, needs_session in pending:
            if needs_session and self.session is None:
                self.events.put(TaskFailed(task_id, 'no model has been built yet (build or compute first)', ''))
                continue
            if needs_session and session_serial is not None and session_serial != self._session_serial:
                self.events.put(TaskFailed(task_id, 'the session changed since this request (the model was '
                                                    'rebuilt); not run on the new session', ''))
                continue
            try:
                value = fn(self.session) if needs_session else fn()
            except Exception as exc:
                self.events.put(TaskFailed(task_id, f'{type(exc).__name__}: {exc}', traceback.format_exc()))
            else:   # cupy data on the GUI thread would outlive release_session and leak pool memory
                if gpu_memory.owned_bytes(value, max_depth=4):
                    self.events.put(TaskFailed(task_id, 'call fn returned GPU (cupy) data; return numpy', ''))
                else:
                    self.events.put(TaskResult(task_id, value))
        return len(pending)

    def _checkpoint_service(self):
        """Study checkpoint service: queued calls, then a MemoryStatus at most once per second."""
        self._service_calls()
        self._emit_memory_status()

    def _emit_memory_status(self, force=False):
        """MemoryStatus into the 'memory' slot (rate-limited to 1 Hz unless forced); a failing query is logged once
        and its status skipped."""
        now = time.monotonic()
        if not force and now - self._last_memory_status < _MEMORY_STATUS_PERIOD:
            return
        self._last_memory_status = now
        try:
            status = gpu_memory.cpu_memory_status() if self._device_kind == 'cpu' else gpu_memory.memory_status()
        except Exception as exc:
            if not self._memory_status_failed:
                self._memory_status_failed = True
                self._emit(LogEvent('warning', f'GPU memory status unavailable: {type(exc).__name__}: {exc}'))
            return
        self._emit(status)

    def _set_session(self, session):
        self.session = session
        if session is not None:
            self._session_serial += 1

    def _emit(self, event):
        if type(event) is Built:   # emitted right after on_session(session): the serial is the new session's
            event = dataclasses.replace(event, session_serial=self._session_serial)
        slot = _LATEST_SLOTS.get(type(event))
        if slot is None:
            self.events.put(event)
            return
        with self._latest_lock:
            self._latest[slot] = event

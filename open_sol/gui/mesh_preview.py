"""Element meshes for display (mesh plan T6, igor3d_plan T4b): the worker session's built triangle / tetrahedron meshes
and mesher previews of the mesh form ('Build mesh'), kept as host data (viz.common.TriangleMeshLines, or
TetrahedronMeshData the canvas cuts by its shown plane) with a key of the inputs they were made from, so a view shows a
mesh only while the model would still give it. GUI thread; work runs on the worker."""
import time
from dataclasses import dataclass

from PySide6 import QtCore

from ..core.gpu_memory import ledger_time_text
from ..model.expressions import error_message
from ..model.geometry import resolve_geometry
from ..model.meshing import element_mesh_key, element_window_size
from ..model.schema import FREE_MESH_KINDS
from ..viz.common import tetrahedron_mesh_data, triangle_mesh_lines

_KEY_ERRORS = (ValueError, KeyError, TypeError, ArithmeticError)
AUTO_PREVIEW_MAX_ELEMENTS = 200_000   # the mesh view meshes triangle estimates up to this size by itself; larger: Build
AUTO_PREVIEW_MAX_TETRAHEDRA = 50_000  # ... tetrahedron estimates (gmsh -3 is slower)
_AUTO_PREVIEW_DELAY_MS = 400          # quiet time after the last edit before an automatic preview starts
_ELEMENT_NAMES = {'triangles': 'triangles', 'tetrahedra': 'tetrahedra'}


@dataclass(frozen=True, eq=False)
class MeshDrawing:
    """One element mesh ready to draw: its host data and the key of the inputs it was made from."""
    mesh_name: str
    key: str
    lines: object               # viz.common.TriangleMeshLines (triangles) | TetrahedronMeshData (tetrahedra)
    source: str                 # 'built' (the worker session's mesh) | 'preview' (the mesh form's Build mesh)
    seconds: float | None = None   # mesher wall time of a preview
    kind: str = 'triangles'


def _mesh_data(mesh, kind):
    return tetrahedron_mesh_data(mesh) if kind == 'tetrahedra' else triangle_mesh_lines(mesh)


def _built_mesh_drawings(session) -> dict:
    """{mesh name: (key, kind, drawing data)} of the session's element meshes (worker thread, host data only)."""
    from ..model.meshing import tetrahedron_sizing, triangle_sizing

    drawings, shapes = {}, None
    for name, domain in session.domains.items():
        kind = getattr(domain, 'grid_kind', 'tensor')
        if kind not in FREE_MESH_KINDS:
            continue
        if shapes is None:
            shapes = resolve_geometry(session.spec, session.parameters)
        sizing = (tetrahedron_sizing if kind == 'tetrahedra' else triangle_sizing)(
            session.spec, session.spec.mesh_by_name(name), session.parameters, session.registry, shapes)
        size = element_window_size(session.spec, session.parameters, kind)
        drawings[name] = (element_mesh_key(sizing, size, shapes), kind, _mesh_data(domain.mesh, kind))
    return drawings


def _has_element_mesh(spec) -> bool:
    return spec is not None and any(mesh_spec.kind in FREE_MESH_KINDS for mesh_spec in spec.used_meshes())


class ElementMeshStore(QtCore.QObject):
    """Built and previewed triangle / tetrahedron meshes per mesh name: the canvas draws them (mesh view, 'Show
    mesh'), the mesh form starts previews and shows their state. Built meshes are fetched after every Built /
    StudyFinished."""

    changed = QtCore.Signal(str)   # mesh name whose drawings or preview state changed
    message = QtCore.Signal(str)   # status text (preview finished / failed)
    log = QtCore.Signal(str, str)  # level, Messages line (a Build mesh preview's result and stage times)

    def __init__(self, document, bridge, parent=None):
        super().__init__(parent)
        self._document = document
        self._bridge = bridge
        self._built = {}              # mesh name -> MeshDrawing of the worker session
        self._previews = {}           # mesh name -> MeshDrawing of the last finished preview
        self._pending = {}            # mesh name -> key of the preview running now
        self._failures = {}           # mesh name -> (key, message) of the last failed preview
        self._log_requests = {}       # mesh name -> key of the preview Build mesh asked to log
        self._keys = (None, {})       # (document revision, {mesh name: (key or None, problem text)})
        self._auto_names = set()      # mesh names the mesh view asked to mesh once edits pause
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(_AUTO_PREVIEW_DELAY_MS)
        self._auto_timer.timeout.connect(self._run_auto_previews)
        document.replaced.connect(self._on_document_replaced)
        bridge.built.connect(self._on_built)
        bridge.build_failed.connect(lambda _failure: self._clear_built())
        bridge.study_finished.connect(self._on_study_finished)
        bridge.study_failed.connect(lambda _message, _traceback: self._on_study_failed())

    # ---- queries ----
    def current_key(self, mesh_name) -> tuple:
        """(key or None, problem text) of the document's element mesh mesh_name (cached per revision)."""
        revision = self._document.revision
        if self._keys[0] != revision:
            self._keys = (revision, {})
        cache = self._keys[1]
        if mesh_name not in cache:
            cache[mesh_name] = self._document_key(mesh_name)
        return cache[mesh_name]

    def drawing(self, mesh_name):
        """The built mesh when the document still gives it, else such a preview; None otherwise."""
        key = self.current_key(mesh_name)[0]
        if key is None:
            return None
        for drawings in (self._built, self._previews):
            drawing = drawings.get(mesh_name)
            if drawing is not None and drawing.key == key:
                return drawing
        return None

    def built_drawing(self, mesh_name):
        """The worker session's mesh mesh_name (the mesh its fields live on), whatever the document says now."""
        return self._built.get(mesh_name)

    def outdated(self, mesh_name) -> bool:
        """True when a built or previewed mesh exists that the document no longer gives."""
        return self.drawing(mesh_name) is None and (mesh_name in self._built or mesh_name in self._previews)

    def pending(self, mesh_name) -> bool:
        return mesh_name in self._pending

    def failure(self, mesh_name) -> str:
        """Message of the last preview failure of the current inputs, '' otherwise."""
        key, text = self._failures.get(mesh_name, (None, ''))
        return text if key is not None and key == self.current_key(mesh_name)[0] else ''

    # ---- previews ----
    def request_preview(self, mesh_name, log=False) -> str:
        """Mesh the document's element mesh mesh_name on the worker (no session needed; the worker keeps the mesh for
        the next build, meshing.element_mesh); log True (the Build mesh button) also logs its stage times like a
        build. Returns a refusal text, '' when started (or already running for these inputs)."""
        key, problem = self.current_key(mesh_name)
        if key is None:
            return f"Mesh '{mesh_name}' cannot be meshed: {problem}"
        if log:
            self._log_requests[mesh_name] = key
        if self._pending.get(mesh_name) == key:
            return ''
        spec, parameters = self._document.spec.copy(), dict(self._document.parameters())
        kind = spec.mesh_by_name(mesh_name).kind
        generation = self._bridge.generation

        def run():
            from ..model.meshing import element_mesh
            start = time.perf_counter()
            shapes = resolve_geometry(spec, parameters)
            resolved = time.perf_counter()
            mesh, reused = element_mesh(spec, parameters, None, mesh_name, shapes=shapes)
            meshed = time.perf_counter()
            data = _mesh_data(mesh, kind)
            seconds = [['geometry', resolved - start], ['mesh', meshed - resolved],
                       ['display data', time.perf_counter() - meshed]]
            return data, meshed - start, seconds, reused

        def done(result):
            if self._bridge.generation != generation:
                return
            data, seconds, stage_seconds, reused = result
            self._finish_preview(mesh_name, key)
            self._previews[mesh_name] = MeshDrawing(mesh_name, key, data, 'preview', seconds, kind)
            self.changed.emit(mesh_name)
            text = (f"Mesh '{mesh_name}': {data.n_elements:,} {_ELEMENT_NAMES[kind]}, {data.n_points:,} nodes"
                    + (' (reused: made earlier from the same mesh settings and geometry)' if reused else ''))
            self.message.emit(f'{text} ({seconds:.2g} s)')
            if self._take_log_request(mesh_name, key):
                self.log.emit('info', text)
                self.log.emit('info', ledger_time_text({'seconds': stage_seconds},
                                                       title=f"Build mesh '{mesh_name}' time by stage"))

        def failed(error):
            if self._bridge.generation != generation:
                return
            self._finish_preview(mesh_name, key)
            self._failures[mesh_name] = (key, str(error).splitlines()[0] if error else 'failed')
            self.changed.emit(mesh_name)
            self.message.emit(f"Mesh '{mesh_name}' failed: {self._failures[mesh_name][1]}")
            if self._take_log_request(mesh_name, key):
                self.log.emit('error', f"Build mesh '{mesh_name}' failed: {error or 'failed'}")

        self._pending[mesh_name] = key
        self._failures.pop(mesh_name, None)
        self._bridge.call_without_session(run, on_result=done, on_error=failed)
        self.changed.emit(mesh_name)
        return ''

    def request_auto_preview(self, mesh_name) -> bool:
        """Mesh view without current elements: preview mesh_name once edits pause (each call restarts the wait).
        False when its estimate exceeds AUTO_PREVIEW_MAX_ELEMENTS (tetrahedra: AUTO_PREVIEW_MAX_TETRAHEDRA) or the
        last preview of these inputs failed."""
        estimate = self._document.mesh_estimate(mesh_name)[0]
        if estimate is None or estimate.kind not in FREE_MESH_KINDS or self.failure(mesh_name):
            return False
        limit = AUTO_PREVIEW_MAX_TETRAHEDRA if estimate.kind == 'tetrahedra' else AUTO_PREVIEW_MAX_ELEMENTS
        if estimate.n_elements > limit:
            return False
        self._auto_names.add(mesh_name)
        self._auto_timer.start()
        return True

    def _run_auto_previews(self):
        """Start the asked-for previews whose inputs have no drawing yet; a running preview of older inputs goes first
        (its finish redraws the mesh view, which asks again)."""
        names, self._auto_names = sorted(self._auto_names), set()
        for name in names:
            if self.drawing(name) is None and not self.pending(name) and not self.failure(name):
                self.request_preview(name)

    def _finish_preview(self, mesh_name, key):
        if self._pending.get(mesh_name) == key:
            del self._pending[mesh_name]

    def _take_log_request(self, mesh_name, key) -> bool:
        """True (once) when Build mesh asked for the preview of these inputs to be logged."""
        if self._log_requests.get(mesh_name) != key:
            return False
        del self._log_requests[mesh_name]
        return True

    # ---- built meshes ----
    def _on_built(self, _summary):
        if not self._bridge.event_is_current:
            return
        if _has_element_mesh(self._bridge.results_spec):
            self._fetch_built()
        else:
            self._clear_built()

    def _on_study_finished(self, finished):
        """A run ends on its last session (sweep value, refinement level): its meshes replace the built ones."""
        if not self._bridge.event_is_current:
            return
        if not (finished.summary or {}).get('session_live', True) or not _has_element_mesh(self._bridge.results_spec):
            self._clear_built()
            return
        self._fetch_built()

    def _on_study_failed(self):
        if self._bridge.results_spec is None:
            self._clear_built()

    def _fetch_built(self):
        generation = self._bridge.generation

        def done(drawings):
            if self._bridge.generation != generation:
                return
            names = set(self._built) | set(drawings)
            self._built = {name: MeshDrawing(name, key, data, 'built', kind=kind)
                           for name, (key, kind, data) in drawings.items()}
            for name in sorted(names):
                self.changed.emit(name)

        def failed(error):   # a sweep rebuilt meanwhile: StudyFinished fetches again
            if self._bridge.generation == generation and 'session changed' not in str(error):
                self.message.emit(f'Built element meshes not shown: {str(error).splitlines()[0]}')
        self._bridge.call(_built_mesh_drawings, on_result=done, on_error=failed)

    def _clear_built(self):
        names = sorted(self._built)
        self._built = {}
        for name in names:
            self.changed.emit(name)

    def _on_document_replaced(self):
        names = sorted(set(self._built) | set(self._previews) | set(self._pending) | set(self._failures))
        self._built, self._previews, self._pending, self._failures, self._log_requests = {}, {}, {}, {}, {}
        self._keys = (None, {})
        self._auto_names = set()
        self._auto_timer.stop()
        for name in names:
            self.changed.emit(name)

    # ---- helpers ----
    def _document_key(self, mesh_name):
        estimate, problem = self._document.mesh_estimate(mesh_name)
        if estimate is None:
            return None, problem or 'no mesh estimate'
        if estimate.kind not in FREE_MESH_KINDS:
            return None, 'it is not a triangle or tetrahedron mesh'
        try:
            parameters = self._document.parameters()
            spec = self._document.spec
            return element_mesh_key(estimate.sizing, element_window_size(spec, parameters, estimate.kind),
                                    resolve_geometry(spec, parameters)), ''
        except _KEY_ERRORS as exc:
            return None, error_message(exc)

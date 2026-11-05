"""CouplingTerm (source state field -> optional transform -> target inject_source slot), CouplingGroup of multi-slot
couplings and TransferredPhysics, a physics seen from another mesh (S7.2)."""
from collections.abc import Mapping

from ..core.backend import array_module_of
from ..core.constants import DTYPE

CONTRIBUTIONS = ('sum', 'index')   # E13 contributor kinds (slot_sum.SlotSum combines them)


class TransferredPhysics:
    """A physics seen from another mesh by a coupling transform running there (S7.2): state() fields interpolated by
    `transfer` onto that mesh on first access after refresh() (transfer None: state() raises, a 'value' transform
    reads only its source), `domain` = that mesh's Domain (its MaterialGrid holds this physics' material arrays when the
    coupling type sets partner_materials); every other attribute is the solver's own (T0, time, omega, ...)."""

    def __init__(self, solver, transfer, domain):
        self._solver = solver
        self._transfer = transfer
        self.domain = domain
        self._buffers = {}   # state name -> persistent target-shaped float32 buffer
        self._fresh = {}     # state name -> value transferred since the last refresh
        self._revision = None   # state_revisions of the solver the fresh values were taken at

    def __getattr__(self, name):
        solver = self.__dict__.get('_solver')
        if solver is None:
            raise AttributeError(name)
        return getattr(solver, name)

    def refresh(self, revision=None):
        """Forget the transferred values (the next state() access interpolates again), unless `revision` (the
        solver's state_revisions) is the one they were taken at: terms sharing this view then interpolate once."""
        if revision is None or revision != self._revision:
            self._fresh.clear()
        self._revision = revision

    def state(self):
        if self._transfer is None:
            raise RuntimeError(f"physics '{self._solver.name}' runs on another mesh: a coupling transform of a 'value' "
                               'mesh transfer reads only its source physics')
        return _TransferredState(self)

    def transferred(self, name):
        """State field `name` of the solver on this view's mesh (grid arrays interpolated, None / scalars as they are;
        grid arrays = ndim >= the transfer's source_ndim: 2 on tensor grids, 1 on a triangle mesh)."""
        if name not in self._fresh:
            value = self._solver.state()[name]
            if value is not None and getattr(value, 'ndim', 0) >= getattr(self._transfer, 'source_ndim', 2):
                value = self._buffers[name] = self._transfer.apply(value, out=self._buffers.get(name))
            self._fresh[name] = value
        return self._fresh[name]


class _TransferredState(Mapping):
    """state() mapping of a TransferredPhysics: each field interpolated on access."""

    def __init__(self, view):
        self._view = view

    def __getitem__(self, name):
        return self._view.transferred(name)

    def __iter__(self):
        return iter(self._view._solver.state())

    def __len__(self):
        return len(self._view._solver.state())


class CouplingTerm:
    """Copies `source.state()[source_field]` (optionally transformed) into `target.inject_source(target_input)`;
    `flush_transform` (default transform) only at the end-of-run flush; every None = runner.prepare sets max(strides).
    contribution 'sum' / 'index' (E13): the value goes into the term's own `buffer`; a SlotSum injects the slot.
    also_reads: further (physics, field) pairs the transform reads; they join source_fields and the change check.
    Source and target on different meshes (S7.2, set_mesh_transfers): source_transfer = the source seen from the
    target mesh (transform there), value_transfer = the transform's result moved from the source mesh."""

    def __init__(self, source: str, source_field: str,
                 target: str, target_input: str,
                 transform=None, every: int = None, name: str = '', flush_transform=None,
                 after_source_step: bool = False, contribution: str = '', also_reads=(),
                 source_transfer=None, value_transfer=None):
        if every is not None and int(every) < 1:
            raise ValueError(f"CouplingTerm every must be >= 1 or None, got {every}")
        if contribution not in ('', *CONTRIBUTIONS):
            raise ValueError(f"CouplingTerm contribution must be '' or one of {', '.join(CONTRIBUTIONS)}, got "
                             f"{contribution!r}")
        self.contribution = contribution
        self.buffer = None   # contributor: persistent full-grid float32 copy of the last value (None before one)
        self.source = source
        self.source_field = source_field
        self.target = target
        self.target_input = target_input
        self.transform = transform
        self.flush_transform = transform if flush_transform is None else flush_transform
        self.every = None if every is None else int(every)
        self.name = name
        self.after_source_step = bool(after_source_step)
        self.source_fields = frozenset({(source, source_field), *also_reads})
        # physics whose state_revisions trigger a transfer: the source first, then the other read physics
        self._read_physics = (source, *sorted({physics for physics, _ in also_reads} - {source}))
        self._sent_revision = None   # state_revisions of _read_physics at the last transfer
        self.source_transfer = self.value_transfer = None
        self._views = {}             # physics name -> TransferredPhysics (S7.2)
        self._moved = None           # value_transfer output buffer
        self.set_mesh_transfers(source_transfer, value_transfer)

    def set_mesh_transfers(self, source_transfer=None, value_transfer=None, views=None):
        """S7.2 source and target on different meshes, at most one GridTransfer: source_transfer ('field') interpolates
        the source state onto the target mesh, where the transform runs; value_transfer ('value') moves the result of
        the transform (on the source mesh, the target seen there without state) onto the target mesh. views: a dict
        shared by the terms of one coupling (same transfer), so a source field is interpolated once per revision."""
        if source_transfer is not None and value_transfer is not None:
            raise ValueError(f"coupling '{self.name}': give a source_transfer or a value_transfer, not both")
        self.source_transfer = source_transfer
        self.value_transfer = value_transfer
        self._views = {} if views is None else views
        self._moved = None

    def apply(self, physics_by_name: dict):
        """Cadenced transfer (the runner picks the ticks): apply_now unless the source has not solved / stepped since
        the last transfer."""
        if self._source_changed(physics_by_name):
            self.apply_now(physics_by_name)

    def apply_now(self, physics_by_name: dict):
        """Unconditional transfer; a transform returning None means no new data (nothing injected)."""
        self._transfer(physics_by_name, self.transform)

    def flush(self, physics_by_name: dict):
        """End-of-run transfer (runner.flush): apply_now with flush_transform."""
        self._transfer(physics_by_name, self.flush_transform)

    def transfer_value(self, physics_by_name: dict, transform=None):
        """The value apply_now would inject (transform default: self.transform); None = nothing to inject. May be the
        source's live state array: copy before keeping it. Target-shaped also across meshes (S7.2)."""
        transform = self.transform if transform is None else transform
        src = _lookup(physics_by_name, self.source, self)
        tgt = _lookup(physics_by_name, self.target, self)
        if self.source_transfer is not None:   # 'field': the transform runs on the target mesh
            src = self._view(src, self.source_transfer, tgt)
            src.refresh(state_revisions(physics_by_name, (self.source,)))
        elif self.value_transfer is not None:   # 'value': it runs on the source mesh
            tgt = self._view(tgt, None, src)
        state = src.state()
        if self.source_field not in state:
            raise KeyError(f"coupling '{self.name}': physics '{self.source}' has no state field "
                           f"'{self.source_field}'; available: {list(state)}")
        field = state[self.source_field]
        value = field if transform is None else transform(field, src, tgt)
        if value is None or self.value_transfer is None:
            return value
        self._moved = self.value_transfer.apply(value, out=self._moved)
        return self._moved

    def _view(self, solver, transfer, host):
        """Cached TransferredPhysics of solver on the mesh of `host` (a solver there)."""
        view = self._views.get(solver.name)
        if view is None:
            view = self._views[solver.name] = TransferredPhysics(solver, transfer, getattr(host, 'domain', None))
        return view

    def write(self, physics_by_name: dict, value):
        """Inject value into the target slot; a contributor copies it into its buffer instead (float32)."""
        if not self.contribution:
            _lookup(physics_by_name, self.target, self).inject_source(self.target_input, value)
            return
        shape = getattr(value, 'shape', ())
        if len(shape) not in (1, 2, 3) or (self.buffer is not None and shape != self.buffer.shape):
            expected = 'a full-grid (nx, ny), 3D (nx, ny, nz) or per-element array' if self.buffer is None else \
                f'shape {self.buffer.shape}'
            raise ValueError(f"coupling '{self.name}': contribution to '{self.target_input}' of '{self.target}' has "
                             f"shape {shape}, expected {expected}")
        xp = array_module_of(value)   # the run backend's module (numpy on the cpu)
        if self.buffer is None:
            self.buffer = xp.empty(shape, dtype=DTYPE)
        xp.copyto(self.buffer, value, casting='same_kind')

    def _source_changed(self, physics_by_name: dict) -> bool:
        revision = state_revisions(physics_by_name, self._read_physics)
        return revision is None or revision != self._sent_revision

    def _transfer(self, physics_by_name: dict, transform):
        self._sent_revision = state_revisions(physics_by_name, self._read_physics)
        field = self.transfer_value(physics_by_name, transform)
        if field is not None:
            self.write(physics_by_name, field)


class CouplingGroup:
    """One spec coupling that writes several slots: member terms (their every is unused; only their apply_now is
    called) transfer together, in order, at the group's cadence (None: resolved by runner.prepare as CouplingTerm's)."""

    def __init__(self, terms, every: int = None, name: str = '', after_source_step: bool = False):
        terms = tuple(terms)
        if not terms:
            raise ValueError(f"CouplingGroup '{name}' needs at least one term")
        if every is not None and int(every) < 1:
            raise ValueError(f"CouplingGroup every must be >= 1 or None, got {every}")
        self.terms = terms
        self.every = None if every is None else int(every)
        self.name = name
        self.after_source_step = bool(after_source_step)
        self.source_fields = frozenset().union(*(term.source_fields for term in terms))

    def apply(self, physics_by_name: dict):
        """Cadenced transfer of every term (the runner picks the ticks); skipped while no term's source solved /
        stepped since its last transfer."""
        if any(term._source_changed(physics_by_name) for term in self.terms):
            self.apply_now(physics_by_name)

    def apply_now(self, physics_by_name: dict):
        """Unconditional transfer of every term, in order."""
        for term in self.terms:
            term.apply_now(physics_by_name)

    def flush(self, physics_by_name: dict):
        """End-of-run transfer of every term (their flush), in order."""
        for term in self.terms:
            term.flush(physics_by_name)


def state_revisions(physics_by_name, physics_names):
    """Per-physics counters of produced state (state_revision, else n_solves, else engine.n_step); None (always treated
    as new) without names or when one physics has none. Every state change marks its physics dirty, so shows at its
    next solve (electrostatics with driven terminals: state_revision also counts the per-step superpositions)."""
    if not physics_names:
        return None
    revisions = []
    for physics_name in physics_names:
        solver = physics_by_name.get(physics_name)
        revision = getattr(solver, 'state_revision', None)
        if revision is None:
            revision = getattr(solver, 'n_solves', None)
        if revision is None:
            revision = getattr(getattr(solver, 'engine', None), 'n_step', None)
        if revision is None:
            return None
        revisions.append(revision)
    return tuple(revisions)


def _lookup(physics_by_name, physics_name, coupling):
    try:
        return physics_by_name[physics_name]
    except KeyError:
        raise KeyError(f"coupling '{coupling.name}' references unknown physics '{physics_name}'; "
                       f"attached: {list(physics_by_name)}") from None

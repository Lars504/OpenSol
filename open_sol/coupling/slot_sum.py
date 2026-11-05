"""E13 additive source slots: SlotSum, the one writer of a (target, slot) that combines its contributor CouplingTerms,
and the helpers MultiphysicsRunner.add_coupling routes contributors with."""
from ..core.backend import array_module_of
from ..physics.em.index_reference import INDEX_SLOTS, index_shift_to_permittivity, record_reference_index, \
    reference_index
from .coupling_term import CONTRIBUTIONS, _lookup

# window-averaged EM loss fields (the runner enables their loss accounting; their readers transfer on phase 0)
LOSS_FIELDS = frozenset({'q_accum', 'q_accum_time'})


class SlotSum:
    """Writer of `target_input` on `target` fed by contributor terms: each transfer refreshes their buffers and injects
    'sum' = their sum or 'index' = (n0 + sum dn)^2 (n0 of the component recorded right before its first write)."""

    def __init__(self, target: str, target_input: str, contribution: str):
        if contribution not in CONTRIBUTIONS:
            raise ValueError(f"SlotSum contribution must be one of {', '.join(CONTRIBUTIONS)}, got {contribution!r}")
        if contribution == 'index' and target_input not in INDEX_SLOTS:
            raise ValueError(f"slot '{target_input}' of '{target}' takes no 'index' contributions; index slots: "
                             f"{', '.join(INDEX_SLOTS)}")
        self.target = target
        self.target_input = target_input
        self.contribution = contribution
        self.contributors = ()
        self.every = None             # the contributors' shared every (None: runner.prepare resolves it)
        self._requested_every = None  # every as the contributors passed it (prepare may overwrite self.every)
        self.after_source_step = False
        self._phases = frozenset()    # the contributors' after_source_step values (one when one reads LOSS_FIELDS)
        self._reads_window = False    # a contributor reads LOSS_FIELDS
        self.source_fields = frozenset()
        self.name = f'{target}.{target_input}'
        self._total = None            # combined value buffer ('sum' with several contributors, 'index' dn sum)
        self._permittivity = None     # 'index' output eps buffer
        self._reference_recorded = False

    def add(self, term, every, after_source_step):
        """Append a contributor term with the every / after_source_step of the coupling that carried it (a
        CouplingGroup's); ValueError for another kind, slot or every, or for mixed phases when a contributor reads
        LOSS_FIELDS (the slot would move its loss window one tick late). after_source_step: any contributor's."""
        if (term.target, term.target_input) != (self.target, self.target_input):
            raise ValueError(f"coupling '{term.name}' writes '{term.target_input}' of '{term.target}', not "
                             f"'{self.target_input}' of '{self.target}'")
        if term.contribution != self.contribution:
            raise ValueError(f"coupling '{term.name}' contributes '{term.contribution}' to slot '{self.target_input}' "
                             f"of '{self.target}', but {self.name} contribute{'s' if len(self.contributors) == 1 else ''}"
                             f" '{self.contribution}': one slot combines one kind")
        if self.contributors and every != self._requested_every:
            raise ValueError(f"coupling '{term.name}' (every {every}) and {self.name} (every "
                             f"{self._requested_every}) contribute to slot '{self.target_input}' of '{self.target}': "
                             'contributors to one slot share one every')
        phase = bool(after_source_step)
        reads_window = any(field in LOSS_FIELDS for _, field in term.source_fields)
        if len(self._phases | {phase}) > 1 and (reads_window or self._reads_window):
            after_source = term.name if phase else self.name
            raise ValueError(f"coupling '{term.name}' and {self.name} contribute to slot '{self.target_input}' of "
                             f"'{self.target}' on different phases, and one reads the window-averaged EM loss "
                             f"({', '.join(sorted(LOSS_FIELDS))}): the shared slot would transfer it right after its "
                             f"source steps, one tick into the next window. Give the source physics of '{after_source}' "
                             "an every other than the coupling's (e.g. 1: a quasi-static solver re-solves only when "
                             'its inputs change), so it transfers just before the target steps too')
        self._phases = self._phases | {phase}
        self._reads_window = self._reads_window or reads_window
        self._requested_every = self.every = every
        self.contributors += (term,)
        self.after_source_step = self.after_source_step or bool(after_source_step)
        self.source_fields = self.source_fields | term.source_fields
        self.name = ' + '.join(contributor.name or f'{contributor.source}.{contributor.source_field}'
                               for contributor in self.contributors)

    def apply(self, physics_by_name: dict):
        """Cadenced transfer: refresh the contributors whose source solved / stepped since their last transfer, then
        inject; nothing when none did."""
        changed = [term for term in self.contributors if term._source_changed(physics_by_name)]
        if changed:
            self._record_reference(physics_by_name)
            for term in changed:
                term.apply_now(physics_by_name)
            self._inject(physics_by_name)

    def apply_now(self, physics_by_name: dict):
        """Unconditional transfer: refresh every contributor, inject the combination."""
        self._refresh(physics_by_name, flush=False)
        self._inject(physics_by_name)

    def flush(self, physics_by_name: dict):
        """End-of-run transfer: every contributor's flush, then inject."""
        self._refresh(physics_by_name, flush=True)
        self._inject(physics_by_name)

    def transfer_value(self, physics_by_name: dict, transform=None):
        """Refresh every contributor and return the combined slot value (transform unused; None while no contributor
        has data). A persistent buffer: copy before keeping it."""
        self._refresh(physics_by_name, flush=False)
        return self._combine(physics_by_name)

    def write(self, physics_by_name: dict, value):
        """Inject value into the target slot."""
        _lookup(physics_by_name, self.target, self).inject_source(self.target_input, value)

    def _refresh(self, physics_by_name, flush):
        self._record_reference(physics_by_name)
        for term in self.contributors:
            if flush:
                term.flush(physics_by_name)
            else:
                term.apply_now(physics_by_name)

    def _inject(self, physics_by_name):
        value = self._combine(physics_by_name)
        if value is not None:
            self.write(physics_by_name, value)

    def _record_reference(self, physics_by_name):
        """'index': snapshot n0 of the slot's component once, before this (its only) writer first rewrites it and
        before the contributors read it."""
        if self.contribution == 'index' and not self._reference_recorded:
            materials = _lookup(physics_by_name, self.target, self).domain.materials
            record_reference_index(materials, (INDEX_SLOTS[self.target_input][0],))
            self._reference_recorded = True

    def _combine(self, physics_by_name):
        """Combined slot value from the contributor buffers (no allocation after the first call); None without any."""
        buffers = [term.buffer for term in self.contributors if term.buffer is not None]
        if not buffers:
            return None
        total = buffers[0]
        xp = array_module_of(total)   # the run backend's module (numpy on the cpu)
        if len(buffers) > 1:
            if self._total is None:
                self._total = xp.empty_like(total)
            xp.copyto(self._total, total)
            for buffer in buffers[1:]:
                xp.add(self._total, buffer, out=self._total)
            total = self._total
        if self.contribution == 'sum':
            return total
        if self._permittivity is None:
            self._permittivity = xp.empty_like(total)
        n0 = reference_index(_lookup(physics_by_name, self.target, self).domain.materials, self.target_input)
        return index_shift_to_permittivity(n0, total, self._permittivity)


def slot_contributions(coupling):
    """[(term, every, after_source_step)] of the contributor terms a coupling carries (a contributor CouplingTerm, or
    a CouplingGroup of them with the group's every / after_source_step); [] for a plain coupling."""
    if isinstance(coupling, SlotSum):
        raise TypeError(f"SlotSum '{coupling.name}' is created by add_coupling from contributor terms; add those")
    terms = getattr(coupling, 'terms', None)
    if terms is None:
        if getattr(coupling, 'contribution', ''):
            return [(coupling, coupling.every, coupling.after_source_step)]
        return []
    contributors = [term for term in terms if getattr(term, 'contribution', '')]
    if not contributors:
        return []
    if len(contributors) != len(terms):
        raise ValueError(f"coupling group '{coupling.name}' mixes contributor and overwriting terms; add them as "
                         'separate couplings')
    return [(term, coupling.every, coupling.after_source_step) for term in terms]


def written_slots(coupling) -> frozenset:
    """(target, slot) pairs a coupling injects (CouplingTerm, CouplingGroup terms, SlotSum); empty for a
    DynamicMaterialUpdate."""
    parts = getattr(coupling, 'terms', (coupling,))
    return frozenset((part.target, part.target_input) for part in parts if hasattr(part, 'target_input'))

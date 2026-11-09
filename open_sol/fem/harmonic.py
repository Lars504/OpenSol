"""Complex scalar harmonic FEM on triangle meshes (new_modules_plan S2): ScalarHarmonicSystem solves -div(a grad u) +
(b + j omega c - omega^2 m) u = f by Lagrange P1 / P2 on an element selection, planar or axisymmetric (r, z), with a
fem.pml ring, Dirichlet / Neumann / Robin edge conditions and point loads; the dimension-free HarmonicCondition,
HarmonicPencil (K, C, M on one pattern, one complex factor per frequency), HarmonicSystemBase and HarmonicEigenPencil
(K u = omega^2 M u for fem.harmonic_resonances) serve fem.harmonic3d too. Host numpy, scipy inside functions."""
# Operator (exp(+j omega t)): A(omega) = K + j omega C - omega^2 M on the dofs of the active elements; K = int a grad v
# . grad u + b v u (+ Robin admittance on edges), C = int c v u (+ Robin damping), M = int m v u, all unconjugated (complex
# symmetric for a symmetric a; no Bloch phases here). a: number, (n_elements,), (n_elements, 2) diagonal, (n_elements,
# 2, 2) or a callable (fem.assembly forms); b, c, m: number, (n_elements,) or callable. K, C and M are assembled once
# (per condition / coefficient change) on one CSR pattern (HarmonicPencil), so every frequency's reduced operator has the
# same pattern (cuDSS plan / MUMPS / PARDISO analysis reuse through previous=; on a cuDSS back end the reduced K, C,
# M values stay on the device, sparse_direct.DevicePencil, so a frequency combines, equilibrates and factorises there
# without a host CSR; ports'
# frequency_terms blocks are added there too, at slots mapped once per block pattern (factorize_pencil(extra=), E7-8;
# a block outside the pattern, or unsymmetric with a symmetric pencil, takes factorize_host). A direct solve on a cuDSS
# factor also keeps its full-numbering values on the device (device_values(), R6: device consumers such as a cupy
# _publish skip the upload of values; None after a host solve, a reduced solve or an assembly). An 'auto' cuDSS
# runtime whose deferred check misses at solve falls back to SuperLU and solves the frequency again (B1b). Membrane:
# a = T (1 + j eta), m = rho_s, b = T k_y^2; pressure acoustics: a = 1 / rho, m = 1 / (rho c^2).
# Per-frequency terms without re-assembly (HarmonicSystemBase public API): the subclass hook frequency_terms(omega) ->
# (system CSR added to A(omega), its pattern already in K as explicit zeros; system load) (membrane port DtN), and the
# setters set_frequency_load (full-numbering load) / set_held_offsets (added to the assembled held values) (acoustics).
# Reduced sweeps: plan_reduced(omegas, 'adaptive' | 'modal') builds a fem.reduced_harmonic model from reduced_terms()
# through _factorize (plan reuse); solve() then evaluates it and raises once the operator, load or held values change.
# Non-polynomial per-frequency terms of a reduced sweep go in as plan_reduced(extra_loads=, held_offsets_of=) with
# set_frequency_load / set_held_offsets left None. A4: solve(omega, fields=False) keeps only the reduced coordinates y
# (values stale, fields_current False); linear_output(key, apply) then gives apply(values) from apply of each basis column
# (cached per key until the next plan / clear), publish_fields() writes values = V y.
# Conditions (HarmonicCondition): 'neumann' n_out . (a grad u) = value (the inward flux), 'robin' n_out . (a grad u) =
# value - (admittance + j omega damping) u (a plane-wave absorber: damping = a / c), 'dirichlet' u = value. value: a
# complex number or a callable: Dirichlet f(dof points (k, d)) -> (k,), Neumann / Robin the edge / face coefficient forms
# of fem.assembly / assembly3d. Face conditions act on the outer side of the meshed region on that face (mesh edge /
# face tags: on a face without a layer they also cover the lateral sides of the other faces' layers, so a wall
# continues through a PML); on a face whose layer is not stretched (no pml, or not one of its faces) at the window edge
# instead. A PML face without a condition is held at 0 (pml_outer 'dirichlet') or left natural ('neumann'). Later
# conditions win shared Dirichlet dofs (a sequence counter, as FEMDiffusionSystem); held PML outer sides are the oldest.
# Edge / face terms are not stretched: keep Robin / Neumann boundaries and sources out of the PML.
# Active elements: default every element with a pml, else the window elements (pml_faces == 0); inactive ones (sound-hard
# voids, membrane holes) carry no dofs, their outline is natural (n . a grad u = 0); dofs of no active element stay 0.
# values: complex128 in the full-mesh numbering (nodes, then n_points + edge for P2) as FEMDiffusionSystem. Axisymmetric:
# integrals per radian (weights r, the PML's r~ / r on top), point loads are whole-ring totals / 2 pi; the axis needs no
# condition (natural).
import math
from dataclasses import dataclass

import numpy as np

from .assembly import (_coefficient_values, assemble_edge_load, assemble_edge_mass, assemble_load, assemble_mass,
                       assemble_stiffness, pattern_sum)
from .diffusion import window_face_edges
from .dofs import lagrange_dofs
from .elements import lagrange_values
from .mesh import FACE_BITS
from .mesher import MERGE_TOLERANCE
from .pml import ring_widths
from .quadrature import TRIANGLE_DEGREES
from .sampling import locator

FACES_2D = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
CONDITION_KINDS = ('dirichlet', 'neumann', 'robin')
PML_OUTER_KINDS = ('dirichlet', 'neumann')
RING_ANGLE = 2.0 * math.pi        # whole rings of an axisymmetric (r, z) solve (integrals here are per radian)
PML_EXTRA_DEGREE = 2              # rule degree added for the stretched (rational, position-dependent) coefficients


@dataclass(frozen=True)
class HarmonicCondition:
    """Condition on mesh edges / faces (module header): 'dirichlet' (u = value), 'neumann' (n_out . a grad u = value)
    or 'robin' (n_out . a grad u = value - (admittance + j omega damping) u)."""
    kind: str = 'neumann'
    value: object = 0.0
    admittance: object = 0.0
    damping: object = 0.0

    def __post_init__(self):
        if self.kind not in CONDITION_KINDS:
            raise ValueError(f"HarmonicCondition kind must be one of {', '.join(CONDITION_KINDS)}, got {self.kind!r}")
        if self.kind != 'robin' and not (is_zero(self.admittance) and is_zero(self.damping)):
            raise ValueError(f"HarmonicCondition '{self.kind}': admittance / damping belong to a 'robin' condition")


def is_zero(value) -> bool:
    """True for a number or array that is 0 everywhere (never for a callable)."""
    return (not callable(value)) and not np.any(np.asarray(value) != 0)


def dirichlet_values(value, points) -> np.ndarray:
    """Complex128 (k,) Dirichlet values at dof points (k, d): a number or f(points) -> (k,)."""
    values = value(points) if callable(value) else value
    return np.broadcast_to(np.asarray(values, dtype=np.complex128), (len(points),)).copy()


def system_positions(global_ids, full_ids) -> tuple:
    """(positions (k,) into the sorted system ids global_ids, keep (len(full_ids),) bool) of full-numbering ids; ids
    outside the system are dropped (keep False)."""
    full_ids = np.asarray(full_ids, dtype=np.int64).ravel()
    if not len(global_ids):
        return np.zeros(0, dtype=np.int64), np.zeros(len(full_ids), dtype=bool)
    found = np.minimum(np.searchsorted(global_ids, full_ids), len(global_ids) - 1)
    keep = global_ids[found] == full_ids
    return found[keep].astype(np.int64), keep


# ---- the pencil on one pattern ----
@dataclass(frozen=True, eq=False)
class HarmonicPencil:
    """K, C, M of A(omega) = K + j omega C - omega^2 M on one canonical CSR pattern (system numbering) with the free /
    fixed split and the index maps of the reduced free x free and free x fixed blocks (the same pattern every omega)."""
    shape: tuple
    indices: np.ndarray
    indptr: np.ndarray
    stiffness_data: np.ndarray
    damping_data: np.ndarray
    mass_data: np.ndarray
    free: np.ndarray            # int64 system dofs solved for
    fixed: np.ndarray           # int64 system dofs held (Dirichlet)
    free_pattern: tuple         # (indices, indptr, positions into the data) of the free x free block
    coupling_pattern: tuple     # likewise of the free x fixed block

    @classmethod
    def of(cls, stiffness, damping, mass, fixed_mask) -> 'HarmonicPencil':
        """HarmonicPencil of system-numbering CSR K, C, M (any patterns) and a bool fixed mask per system dof."""
        terms = [matrix.tocsr() for matrix in (stiffness, damping, mass)]
        for matrix in terms:
            matrix.sum_duplicates()   # canonical (sorted, no duplicates): the key order below
        stored = [matrix for matrix in terms if matrix.nnz]
        same = stored and all(np.array_equal(matrix.indptr, stored[0].indptr)
                              and np.array_equal(matrix.indices, stored[0].indices) for matrix in stored[1:])
        union = stored[0] if same else pattern_sum(terms)   # explicit zeros kept; one shared pattern: no sums
        n_rows = union.shape[0]
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(union.indptr))
        union_keys = None

        def aligned(matrix):
            dtype = np.result_type(matrix.dtype, np.float64)   # real stays real, complex stays complex
            if not matrix.nnz:
                return np.zeros(union.nnz, dtype=dtype)
            if matrix is union or (np.array_equal(matrix.indptr, union.indptr)
                                   and np.array_equal(matrix.indices, union.indices)):
                return matrix.data.astype(dtype)   # a copy
            nonlocal union_keys
            if union_keys is None:
                union_keys = rows * union.shape[1] + union.indices   # row-major: sorted for a canonical CSR
            keys = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(matrix.indptr)) * union.shape[1] + matrix.indices
            where = np.minimum(np.searchsorted(union_keys, keys), union.nnz - 1)
            if not np.array_equal(union_keys[where], keys):
                raise RuntimeError('HarmonicPencil: a term did not align with the union pattern')
            data = np.zeros(union.nnz, dtype=dtype)
            data[where] = matrix.data
            return data

        fixed_mask = np.asarray(fixed_mask, dtype=bool)
        free, fixed = np.flatnonzero(~fixed_mask), np.flatnonzero(fixed_mask)
        new_index = np.empty(n_rows, dtype=np.int64)   # position within free (free dofs) or fixed (held dofs)
        new_index[free] = np.arange(free.size)
        new_index[fixed] = np.arange(fixed.size)
        free_row = ~fixed_mask[rows]
        index_dtype = union.indices.dtype

        def block(column_mask, n_block_rows):
            """(indices, indptr, positions into the data) of the free rows x masked columns, canonical order."""
            positions = np.flatnonzero(free_row & column_mask[union.indices])
            counts = np.bincount(new_index[rows[positions]], minlength=n_block_rows)
            indptr = np.concatenate([[0], np.cumsum(counts)]).astype(index_dtype)
            return new_index[union.indices[positions]].astype(index_dtype), indptr, positions

        return cls(shape=union.shape, indices=union.indices, indptr=union.indptr, stiffness_data=aligned(terms[0]),
                   damping_data=aligned(terms[1]), mass_data=aligned(terms[2]), free=free, fixed=fixed,
                   free_pattern=block(~fixed_mask, free.size), coupling_pattern=block(fixed_mask, free.size))

    @property
    def has_damping(self) -> bool:
        return bool(np.any(self.damping_data != 0))

    def _data(self, omega, positions=None) -> np.ndarray:
        stiffness, damping, mass = self.stiffness_data, self.damping_data, self.mass_data
        if positions is not None:
            stiffness, damping, mass = stiffness[positions], damping[positions], mass[positions]
        return stiffness + 1j * omega * damping - omega * omega * mass

    def full(self, omega):
        """Complex CSR A(omega) of every system dof."""
        from scipy.sparse import csr_matrix
        return csr_matrix((self._data(omega), self.indices, self.indptr), shape=self.shape)

    def reduced(self, omega):
        """Complex CSR (n_free, n_free) of A(omega) on the free dofs (the same pattern for every omega)."""
        from scipy.sparse import csr_matrix
        indices, indptr, positions = self.free_pattern
        return csr_matrix((self._data(omega, positions), indices, indptr), shape=(self.free.size, self.free.size))

    def coupling(self, omega):
        """Complex CSR (n_free, n_fixed) of A(omega): the lift of the Dirichlet values."""
        from scipy.sparse import csr_matrix
        indices, indptr, positions = self.coupling_pattern
        return csr_matrix((self._data(omega, positions), indices, indptr), shape=(self.free.size, self.fixed.size))

    def part(self, name):
        """CSR (n_free, n_free) of 'stiffness', 'damping' or 'mass' alone (its own dtype: real stays real)."""
        from scipy.sparse import csr_matrix
        data = {'stiffness': self.stiffness_data, 'damping': self.damping_data, 'mass': self.mass_data}[name]
        indices, indptr, positions = self.free_pattern
        return csr_matrix((data[positions], indices, indptr), shape=(self.free.size, self.free.size))

    def coupling_part(self, name):
        """CSR (n_free, n_fixed) free x fixed block of 'stiffness', 'damping' or 'mass' alone (its own dtype)."""
        from scipy.sparse import csr_matrix
        data = {'stiffness': self.stiffness_data, 'damping': self.damping_data, 'mass': self.mass_data}[name]
        indices, indptr, positions = self.coupling_pattern
        return csr_matrix((data[positions], indices, indptr), shape=(self.free.size, self.fixed.size))


@dataclass(frozen=True, eq=False)
class HarmonicEigenPencil:
    """K u = omega^2 M u on the free dofs (Dirichlet dofs held at 0): full_ids = the full-numbering id per row,
    in_window = row inside the window (False in a PML layer), in_region = rows of a region selection or None."""
    stiffness: object
    mass: object
    full_ids: np.ndarray
    n_full: int
    in_window: np.ndarray
    in_region: object = None

    def spread(self, vector) -> np.ndarray:
        """Complex128 (n_full,) field of a free-row vector (0 on held and inactive dofs)."""
        full = np.zeros(self.n_full, dtype=np.complex128)
        full[self.full_ids] = vector
        return full

    def energy_weights(self) -> np.ndarray:
        """Float64 (n_free,) lumped |M| row sums: the per-row weight of |u|^2 energy shares (0 on massless rows)."""
        return np.asarray(abs(self.mass).sum(axis=1), dtype=np.float64).ravel()


# ---- dimension-free system ----
class HarmonicSystemBase:
    """Dimension-free part of the scalar harmonic systems: coefficients, active selection, conditions by sequence,
    sources, assembly into a HarmonicPencil, the per-frequency factor and solve, reactions and the eigen pencil.
    Subclasses provide face_entities, active_elements, _assemble_terms, _source_load, _element_rows (+ frequency_terms)."""
    faces = FACES_2D

    def __init__(self, n_full, linear_solver, pml, pml_outer, symmetric, label):
        if pml_outer not in PML_OUTER_KINDS:
            raise ValueError(f"pml_outer must be one of {', '.join(PML_OUTER_KINDS)}, got {pml_outer!r}")
        if linear_solver is None:   # scripts / self-tests: host SuperLU ('linear' role runtimes come from the build)
            from ..solvers.sparse_direct import SparseDirectSolver
            linear_solver = SparseDirectSolver(backend='cpu')
        self.linear_solver = linear_solver
        self.pml = pml
        self.pml_outer = pml_outer
        self.symmetric = bool(symmetric)
        self.label = label
        self.values = np.zeros(int(n_full), dtype=np.complex128)
        self.coefficients = {'a': 1.0, 'b': 0.0, 'damping': 0.0, 'mass': 0.0}
        self.active = None              # bool per element, None = the default selection
        self.face_conditions = {}       # face -> (HarmonicCondition, sequence)
        self.boundary_conditions = []   # [entities int64, HarmonicCondition, sequence]
        self.fixed_values = []          # (full ids, complex values, sequence)
        self.source = None              # volume source f (number, per element or callable)
        self.point_loads = []           # (point (d,), complex value)
        self.frequency_load = None      # complex (n_full,) extra load (set_frequency_load) or None
        self.held_offsets = None        # complex (n_full,) added to the assembled held values (set_held_offsets) or None
        self.pencil = None
        self.global_ids = None          # int64 system dof -> full id (sorted)
        self.in_window = None           # bool per system dof
        self.operator_dirty = True
        self.n_solves = 0
        self.n_assemblies = 0
        self.last_omega = None
        self._sequence = 0
        self._boundary_load = None
        self._load = None
        self._assembled_fixed_value = None   # complex per fixed system dof, as assembled
        self._fixed_value = None        # the same + held_offsets
        self._factor = None
        self._device_pencil = None      # (pencil, DevicePencil or None, term names) of _device_terms
        self._extra_slots = None        # (DevicePencil, block indptr, indices, slots) of _extra_on_device
        self._solution = None           # system numbering of the last solve
        self._device_values = None      # cupy full-numbering values of the last direct cuDSS solve (device_values, R6)
        self._device_ids = None         # (pencil, cupy full ids of its free / fixed system dofs) of _keep_device_values
        self._reduced = None            # (ReducedHarmonicModel, its load, its held values) of plan_reduced, or None
        self._held_offsets_of = None    # plan_reduced(held_offsets_of=): f(omega) -> full-numbering held offsets, or None
        self._coordinates = None        # (reduced coordinates y, held values) of the last reduced solve, or None
        self._output_columns = {}       # linear_output key -> (apply of each basis column (k, ...), fixed part or None)
        self.fields_current = True      # values hold the last solve (False after solve(fields=False) of a reduced model)

    # ---- read access ----
    @property
    def solution(self):
        """Complex128 system-numbering u of the last solve (None before one or after a re-assembly)."""
        return self._solution

    @property
    def factor(self):
        """The factor of the last solve's reduced operator (linear_solver.factorize_host result; None before one)."""
        return self._factor

    @property
    def fixed_value(self):
        """Complex128 held values per fixed system dof (pencil.fixed order) of the next solve; assembles if needed."""
        self.ensure_assembled()
        return self._fixed_value

    # ---- settings ----
    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def set_coefficients(self, a=1.0, b=0.0, mass=0.0, damping=0.0) -> None:
        """Volume coefficients of -div(a grad u) + (b + j omega damping - omega^2 mass) u (module header forms)."""
        self.coefficients = {'a': a, 'b': b, 'damping': damping, 'mass': mass}
        self.operator_dirty = True

    def set_active(self, element_mask) -> None:
        """Bool per mesh element of the solved elements (None = the default selection)."""
        self.active = None if element_mask is None else np.asarray(element_mask, dtype=bool).copy()
        self.operator_dirty = True

    def set_face_condition(self, face, condition) -> None:
        """Condition of an outer face (module header: where it acts)."""
        if face not in self.faces:
            raise ValueError(f"unknown face {face!r}; expected one of {', '.join(self.faces)}")
        self.face_conditions[face] = (condition, self._next_sequence())
        self.operator_dirty = True

    def add_boundary_condition(self, entities, condition) -> int:
        """Condition on mesh edges (2D) / faces (3D), e.g. object outlines; returns its handle."""
        self.boundary_conditions.append([np.unique(np.asarray(entities, dtype=np.int64)), condition,
                                         self._next_sequence()])
        self.operator_dirty = True
        return len(self.boundary_conditions) - 1

    def set_boundary_condition(self, handle, condition) -> None:
        """Replace the condition of an add_boundary_condition handle."""
        self.boundary_conditions[handle][1] = condition
        self.operator_dirty = True

    def set_fixed_values(self, full_ids, values) -> None:
        """Hold full-numbering dofs at complex values (number or per id; the latest call wins shared dofs)."""
        full_ids = np.asarray(full_ids, dtype=np.int64).ravel()
        values = np.broadcast_to(np.asarray(values, dtype=np.complex128), full_ids.shape).copy()
        self.fixed_values.append((full_ids, values, self._next_sequence()))
        self.operator_dirty = True

    def set_source(self, source) -> None:
        """Volume source density f (number, per element or callable; None = none); keeps the operator."""
        self.source = source
        self._load = None

    def add_point_load(self, point, value) -> None:
        """Point load of complex `value` at a point [m] (2D: per unit length; axisymmetric: whole-ring total)."""
        self.point_loads.append((np.asarray(point, dtype=np.float64).ravel(), complex(value)))
        self._load = None

    def clear_point_loads(self) -> None:
        self.point_loads = []
        self._load = None

    def set_frequency_load(self, full_load) -> None:
        """Complex (n_full,) full-numbering load added to the assembled one in system units (None = none); keeps the
        operator, survives re-assembly (per-frequency loads without re-assembly)."""
        self.frequency_load = None if full_load is None else self._full_vector(full_load, 'frequency load')
        self._load = None

    def set_held_offsets(self, full_offsets) -> None:
        """Complex (n_full,) full-numbering offsets added to the assembled held values at the held dofs, system units
        (None = the assembled values); keeps the operator, survives re-assembly."""
        self.held_offsets = None if full_offsets is None else self._full_vector(full_offsets, 'held offsets')
        if self.pencil is not None and not self.operator_dirty:
            self._apply_held_offsets()

    def _full_vector(self, values, what) -> np.ndarray:
        values = np.asarray(values, dtype=np.complex128).ravel()
        if values.shape != self.values.shape:
            raise ValueError(f'{self.label()}: {what} needs {self.values.size} full-numbering values, got {values.size}')
        return values

    def _apply_held_offsets(self) -> None:
        fixed_value = self._assembled_fixed_value
        if self.held_offsets is not None:
            fixed_value = fixed_value + self.held_offsets[self.global_ids[self.pencil.fixed]]
        self._fixed_value = fixed_value

    def _condition_items(self):
        """(entities, condition, sequence) of the face conditions (incl. the PML outer sides) and the boundary ones."""
        pml_faces = () if self.pml is None else self.pml.faces
        for face in self.faces:
            entry = self.face_conditions.get(face)
            if entry is None:
                if face not in pml_faces or self.pml_outer != 'dirichlet':
                    continue
                entry = (HarmonicCondition('dirichlet', 0.0), 0)
            yield self.face_entities(face), entry[0], entry[1]
        for entities, condition, sequence in self.boundary_conditions:
            yield entities, condition, sequence

    # ---- assembly ----
    def assemble(self) -> None:
        """Assemble K, C, M, the boundary loads and the held dofs into a HarmonicPencil (no factor yet)."""
        terms = self._assemble_terms()
        global_ids = terms['global_ids']
        n_system = len(global_ids)
        constraints = list(terms['constraints'])
        for full_ids, values, sequence in self.fixed_values:
            positions, keep = system_positions(global_ids, full_ids)
            constraints.append((sequence, positions, values[keep]))
        fixed_mask = np.zeros(n_system, dtype=bool)
        fixed_value = np.zeros(n_system, dtype=np.complex128)
        for _, positions, values in sorted(constraints, key=lambda entry: entry[0]):
            fixed_mask[positions] = True
            fixed_value[positions] = values
        from scipy.sparse import csr_matrix
        shape = (n_system, n_system)

        def summed(matrices):   # one term: a CSR copy, no COO rebuild (E7-4; HarmonicPencil.of canonicalises it)
            if len(matrices) == 1:
                return matrices[0].tocsr(copy=True)
            return pattern_sum(matrices) if matrices else csr_matrix(shape)

        self.pencil = HarmonicPencil.of(summed(terms['stiffness']), summed(terms['damping']), summed(terms['mass']),
                                        fixed_mask)
        self.global_ids = global_ids
        self.in_window = terms['in_window']
        self._boundary_load = terms['boundary_load']
        self._assembled_fixed_value = fixed_value[self.pencil.fixed]
        self._apply_held_offsets()
        self._load = None
        self._solution = None
        self._device_values = self._device_ids = None
        self.clear_reduced()
        self.operator_dirty = False
        self.n_assemblies += 1

    def ensure_assembled(self) -> None:
        """Assemble when a setting changed the operator since the last assembly (or none ran yet)."""
        if self.operator_dirty or self.pencil is None:
            self.assemble()

    def load_vector(self) -> np.ndarray:
        """Complex128 system-numbering load: boundary values + volume source + point loads + set_frequency_load
        (frequency_terms loads not included)."""
        self.ensure_assembled()
        if self._load is None:
            load = self._source_load()
            if self.frequency_load is not None:
                load = load + self.frequency_load[self.global_ids]
            self._load = self._boundary_load + load
        return self._load

    def operator(self, omega):
        """Reduced complex CSR A(omega) on the free dofs (pencil only, frequency_terms not included)."""
        self.ensure_assembled()
        return self.pencil.reduced(float(omega))

    # ---- solve ----
    def frequency_terms(self, omega) -> tuple:
        """Subclass hook: (system CSR (n_system, n_system) added to A(omega) or None, complex system load or None) at
        omega; the matrix pattern must sit in K already (explicit zeros), so the reduced pattern and the plan stay fixed."""
        return None, None

    def solve(self, omega, fields=True) -> np.ndarray:
        """Solve (A(omega) + frequency_terms) u = f at angular frequency omega [rad/s] >= 0 (one complex factor, or the
        plan_reduced model's V y; fields False there: only y, values stay stale until publish_fields); returns `values`."""
        omega = float(omega)
        if not (math.isfinite(omega) and omega >= 0.0):
            raise ValueError(f'{self.label()}: angular frequency must be finite and >= 0, got {omega!r} rad/s')
        if self._reduced is not None:
            _, planned_load, planned_fixed = self._reduced
            if self.operator_dirty or self._load is not planned_load or self._fixed_value is not planned_fixed:
                raise RuntimeError(f'{self.label()}: the operator, loads or held values changed after plan_reduced: '
                                   'plan again or call clear_reduced()')
        load = self.load_vector()
        pencil = self.pencil
        if self._reduced is not None and pencil.free.size:
            model = self._reduced[0]
            held = self._reduced_held(omega)
            self._coordinates = (model.reduced_coordinates([omega])[0], held)
            self.last_omega = omega
            self.n_solves += 1
            self._device_values = None
            if fields:
                self.publish_fields(force=True)
            else:
                self._solution = None
                self.fields_current = False
            return self.values
        extra_matrix, extra_load = self.frequency_terms(omega)
        if extra_load is not None:
            load = load + extra_load
        fixed_value = self._fixed_value
        solution = np.zeros(pencil.shape[0], dtype=np.complex128)
        solution[pencil.fixed] = fixed_value
        free_device = None
        if pencil.free.size:
            rhs = load[pencil.free]
            free_rows = extra = None
            if extra_matrix is not None:
                free_rows = extra_matrix.tocsr()[pencil.free]
                extra = free_rows[:, pencil.free]
            if pencil.fixed.size and np.any(fixed_value != 0.0):   # held at 0 (clamped): no lift
                lift = pencil.coupling(omega) @ fixed_value
                if free_rows is not None:
                    lift = lift + free_rows[:, pencil.fixed] @ fixed_value
                rhs = rhs - lift
            self._factorize(omega, extra)
            solution[pencil.free], free_device = self._solve_free(rhs, omega, extra)
        self._store_solution(solution)
        if free_device is not None:
            self._keep_device_values(free_device, fixed_value)
        self.last_omega = omega
        self.n_solves += 1
        return self.values

    def _solve_free(self, rhs, omega, extra) -> tuple:
        """(host free-dof solution, its cupy copy on a cuDSS factor else None) of the current factor; a deferred check
        of a factorize_pencil factor is judged by this solve: on a miss an 'auto' runtime falls back to SuperLU (B1b,
        SparseDirectSolver.fall_back) and the frequency is factorised and solved again, else ValueError."""
        from ..solvers.cudss_factor import CudssAccuracyError, CudssFactorization
        from ..solvers.host_direct import HostDirectAccuracyError   # host factors probe at factorize: never expected
        for attempt in range(2):
            try:
                if isinstance(self._factor, CudssFactorization):   # R6: the solution stays on the device as well
                    import cupy as cp
                    device = self._factor.solve(rhs)
                    return cp.asnumpy(device), device
                return self._factor.solve_host(rhs), None
            except (CudssAccuracyError, HostDirectAccuracyError) as exc:
                fall_back = getattr(self.linear_solver, 'fall_back', None)
                if attempt or fall_back is None or not fall_back(exc, self.pencil.free.size):
                    raise self._accuracy_error(exc, omega) from None
                self._factor = None   # released by the failed check; SuperLU from now on
                self._factorize(omega, extra)
        raise AssertionError('unreachable')

    def _keep_device_values(self, free_device, fixed_value) -> None:
        """device_values of the last direct solve: the cupy free-dof solution and the held values scattered into the
        full numbering on the device (index arrays uploaded once per assembly); none when a subclass transforms
        values (_physical_values)."""
        if type(self)._physical_values is not HarmonicSystemBase._physical_values:
            return
        import cupy as cp
        pencil = self.pencil
        cached = self._device_ids
        if cached is None or cached[0] is not pencil:
            cached = self._device_ids = (pencil, cp.asarray(self.global_ids[pencil.free]),
                                         cp.asarray(self.global_ids[pencil.fixed]))
        _, free_ids, fixed_ids = cached
        full = cp.zeros(self.values.size, dtype=cp.complex128)
        full[free_ids] = free_device
        if fixed_ids.size:
            full[fixed_ids] = cp.asarray(fixed_value)
        self._device_values = full

    def device_values(self):
        """cupy (n_full,) complex128 values of the last direct solve when its factor solved on the device (cuDSS), the
        same as `values` without a host round trip (R6: e.g. a physics' _publish on a cupy domain); None otherwise."""
        return self._device_values

    def _store_solution(self, solution) -> None:
        """System-numbering solution -> _solution and the full-numbering values (fields_current); device_values
        dropped (the caller keeps them again for a device solve)."""
        self._solution = solution
        self._device_values = None
        self.values[:] = 0.0
        self.values[self.global_ids] = solution
        physical = self._physical_values(self.values)
        if physical is not self.values:   # the default hook returns values itself: no copy
            self.values[:] = physical
        self.fields_current = True

    def _physical_values(self, full) -> np.ndarray:
        """Subclass hook: full-numbering physical values of full-numbering system values (default: unchanged)."""
        return full

    def _device_terms(self):
        """sparse_direct.DevicePencil of the reduced K, C, M (once per assembly) when the linear solver factorises on
        the device (cuDSS back end), else None; with the names of its terms."""
        from ..solvers.sparse_direct import SparseDirectSolver
        pencil = self.pencil
        if self._device_pencil is None or self._device_pencil[0] is not pencil:
            device, names = None, ()
            if isinstance(self.linear_solver, SparseDirectSolver):
                indices, indptr, positions = pencil.free_pattern
                names = tuple(name for name, data in (('stiffness', pencil.stiffness_data),
                                                      ('damping', pencil.damping_data), ('mass', pencil.mass_data))
                              if name == 'stiffness' or np.any(data != 0))
                data = {'stiffness': pencil.stiffness_data, 'damping': pencil.damping_data, 'mass': pencil.mass_data}
                device = self.linear_solver.device_pencil(indptr, indices, [data[name][positions] for name in names],
                                                          symmetric=self.symmetric)
            self._device_pencil = (pencil, device, names)
        return self._device_pencil[1], self._device_pencil[2]

    def _extra_on_device(self, device, extra):
        """(slots, values) of a free x free frequency_terms block in the DevicePencil's stored layout (lower entries of
        a lower-stored one), slots mapped once per (pencil, block pattern); None when the block leaves the pattern or is
        unsymmetric with a symmetric pencil (the host route then)."""
        from ..solvers.cudss_factor import SYMMETRY_TOLERANCE, relative_asymmetry
        block = extra.tocsr(copy=True)
        block.sum_duplicates()
        if device.lower_stored and relative_asymmetry(block) > SYMMETRY_TOLERANCE:
            return None
        rows = np.repeat(np.arange(block.shape[0], dtype=np.int64), np.diff(block.indptr))
        columns, values = block.indices, block.data
        if device.lower_stored:
            keep = columns <= rows
            rows, columns, values = rows[keep], columns[keep], values[keep]
        cached = self._extra_slots
        if (cached is not None and cached[0] is device and np.array_equal(cached[1], block.indptr)
                and np.array_equal(cached[2], block.indices)):
            slots = cached[3]
        else:
            slots = device.slots(rows, columns)
            self._extra_slots = (device, block.indptr.copy(), block.indices.copy(), slots)
        return None if slots is None else (slots, values)

    def _accuracy_error(self, exc, omega) -> ValueError:
        """ValueError of a factor missing its accuracy check (cuDSS, or a MUMPS / PARDISO host factor), named by its
        library."""
        from ..solvers.cudss_factor import CudssAccuracyError
        from ..solvers.host_direct import factor_label
        library = ('cuDSS' if isinstance(exc, CudssAccuracyError)
                   else factor_label(getattr(self.linear_solver, 'back_end', None)))
        return ValueError(f'{self.label()}: the {library} solves of the harmonic operator at f = '
                          f'{omega / (2.0 * math.pi):.6g} Hz miss the accuracy check ({exc}): the operator is '
                          'ill-conditioned, e.g. a drive on a nearly lossless resonance')

    def _factorize(self, omega, extra=None) -> None:
        """Factor of the reduced A(omega) (+ extra, a free x free CSR whose pattern sits in K): from the device pencil
        terms on a cuDSS back end (extra: its values added on the device at slots mapped once, E7-8), else of the host
        matrix (plan reuse through previous= either way; extra outside the pattern, or unsymmetric with a symmetric
        pencil, takes the host route)."""
        from ..solvers.cudss_factor import CudssAccuracyError
        from ..solvers.host_direct import HostDirectAccuracyError   # a RuntimeError: caught before the singular case
        try:
            device, names = self._device_terms()
            added = None if device is None or extra is None else self._extra_on_device(device, extra)
            if device is not None and (extra is None or added is not None):
                powers = {'stiffness': 1.0, 'damping': 1j * omega, 'mass': -omega * omega}
                self._factor = self.linear_solver.factorize_pencil(device, [powers[name] for name in names],
                                                                   previous=self._factor, extra=added)
            else:
                matrix = self.pencil.reduced(omega)
                if extra is not None:
                    matrix = pattern_sum([matrix, extra])
                self._factor = self.linear_solver.factorize_host(matrix, symmetric=self.symmetric,
                                                                 previous=self._factor)
        except (CudssAccuracyError, HostDirectAccuracyError) as exc:
            raise self._accuracy_error(exc, omega) from None
        except RuntimeError as exc:
            raise ValueError(f'{self.label()}: the harmonic operator is singular at f = {omega / (2.0 * math.pi):.6g} '
                             f'Hz ({exc}): a lossless resonance on the drive frequency, or a region with neither a '
                             'Dirichlet / Robin edge nor a b / mass term at omega = 0') from None

    # ---- reduced frequency sweeps (fem.reduced_harmonic) ----
    @property
    def reduced_model(self):
        """The ReducedHarmonicModel solve() evaluates (plan_reduced), or None (direct factorisations)."""
        return None if self._reduced is None else self._reduced[0]

    def clear_reduced(self) -> None:
        """Drop the reduced model: solve() factorises per frequency again (values unpublished by solve(fields=False)
        stay as they were: publish_fields first to keep the last solve)."""
        self._reduced = None
        self.fields_current = True
        self._held_offsets_of = None
        self._coordinates = None
        self._output_columns = {}

    def _reduced_held(self, omega) -> np.ndarray:
        """Complex held values per fixed system dof of a reduced solve at omega (+ held_offsets_of of the plan)."""
        held = self._fixed_value
        if self._held_offsets_of is not None and self.pencil.fixed.size:
            offsets = np.asarray(self._held_offsets_of(omega), dtype=np.complex128).ravel()
            held = held + offsets[self.global_ids[self.pencil.fixed]]
        return held

    def _spread(self, free_vector=None, held=None) -> np.ndarray:
        """Full-numbering physical values of a free-dof vector and / or held values (the other part 0)."""
        pencil = self.pencil
        solution = np.zeros(pencil.shape[0], dtype=np.complex128)
        if free_vector is not None:
            solution[pencil.free] = free_vector
        if held is not None:
            solution[pencil.fixed] = held
        full = np.zeros(self.values.shape, dtype=np.complex128)
        full[self.global_ids] = solution
        return self._physical_values(full)

    def publish_fields(self, force=False) -> np.ndarray:
        """Write values = V y (+ held values) of the last reduced solve when solve(fields=False) left them stale (force:
        always); returns `values`."""
        if (self.fields_current and not force) or self._coordinates is None or self._reduced is None:
            return self.values
        coordinates, held = self._coordinates
        pencil = self.pencil
        solution = np.zeros(pencil.shape[0], dtype=np.complex128)
        solution[pencil.fixed] = held
        solution[pencil.free] = self._reduced[0].basis @ coordinates
        self._store_solution(solution)
        return self.values

    def linear_output(self, key, apply) -> np.ndarray:
        """apply(full-numbering values) of the last solve for a linear map apply (A4: after solve(fields=False) from
        apply of each basis column, computed once per key and plan, and of the held values)."""
        if self.fields_current or self._coordinates is None or self._reduced is None:
            return np.asarray(apply(self.values))
        coordinates, held = self._coordinates
        if key not in self._output_columns:
            basis = self._reduced[0].basis
            columns = [np.asarray(apply(self._spread(free_vector=basis[:, index]))) for index in range(basis.shape[1])]
            constant = None if self._held_offsets_of is not None else np.asarray(apply(self._spread(held=held)))
            self._output_columns[key] = (np.stack(columns) if columns else None, constant)
        columns, constant = self._output_columns[key]
        fixed_part = constant if constant is not None else np.asarray(apply(self._spread(held=held)))
        return fixed_part if columns is None else fixed_part + np.tensordot(coordinates, columns, axes=(0, 0))

    def reduced_terms(self) -> tuple:
        """(HarmonicTerm list, HarmonicLoad list) of A(omega) u_free = f_free - A_fc(omega) g on the free dofs: the
        pencil parts, load_vector and the Dirichlet lift per part (frequency_terms not included)."""
        from .reduced_harmonic import HarmonicLoad, HarmonicTerm, PowerCoefficient
        self.ensure_assembled()
        pencil = self.pencil
        loads = [HarmonicLoad(self.load_vector()[pencil.free], PowerCoefficient(1.0, 0))]
        terms = []
        lifted = pencil.fixed.size and np.any(self._fixed_value != 0.0)
        for name, data, coefficient in (('stiffness', pencil.stiffness_data, PowerCoefficient(1.0, 0)),
                                        ('damping', pencil.damping_data, PowerCoefficient(1j, 1)),
                                        ('mass', pencil.mass_data, PowerCoefficient(-1.0, 2))):
            if name != 'stiffness' and not np.any(data != 0):
                continue
            terms.append(HarmonicTerm(pencil.part(name), coefficient))
            if lifted:
                loads.append(HarmonicLoad(-(pencil.coupling_part(name) @ self._fixed_value), coefficient))
        return terms, loads

    def modal_vectors(self, target_frequency, count, eigen_solver=None) -> tuple:
        """(complex (n_free, m) eigenvectors of K u = omega^2 M u nearest target_frequency [Hz], PML modes dropped; the
        HarmonicResonanceSolve) in the free-dof unknowns of solve (fem.harmonic_resonances)."""
        from .harmonic_resonances import solve_harmonic_resonances
        try:
            eigen = self.eigen_pencil()
        except ValueError as exc:
            raise ValueError(f'{exc} (a modal sweep needs the eigen pencil: use sweep_method adaptive)') from None
        resonance_solve = solve_harmonic_resonances(
            eigen.stiffness, eigen.mass, target_frequency, count, eigen_solver=eigen_solver, in_window=eigen.in_window,
            energy_weights=eigen.energy_weights(), label='modal basis')
        if not resonance_solve.resonances:
            raise ValueError(f'{self.label()}: the modal sweep found no mode near {target_frequency:.6g} Hz')
        vectors = np.column_stack([np.asarray(resonance.field, dtype=np.complex128)
                                   for resonance in resonance_solve.resonances])
        return vectors, resonance_solve

    def plan_reduced(self, omegas, method='adaptive', *, tolerance=1e-6, moments=4, max_points=40, modal_count=20,
                     eigen_solver=None, verify=True, extra_loads=(), held_offsets_of=None) -> dict:
        """Reduced model of the free-dof response over angular frequencies omegas [rad/s] (fem.reduced_harmonic):
        'adaptive' (greedy moment matching) or 'modal' (modal_count modes nearest the band centre + its static
        correction); solve() then evaluates it (no factorisation) until clear_reduced or a change of operator / loads.
        extra_loads: HarmonicLoads of full-numbering vectors (system units) on top of load_vector; held_offsets_of(omega)
        -> full-numbering offsets of the held values (lift included). Refuses frequency_terms (ports). Returns info."""
        from .reduced_harmonic import SWEEP_METHODS, HarmonicLoad, ReducedHarmonicModel
        if method not in SWEEP_METHODS:
            raise ValueError(f"reduced sweep method must be one of {', '.join(SWEEP_METHODS)}, got {method!r}")
        self.clear_reduced()
        omegas = np.atleast_1d(np.asarray(omegas, dtype=np.float64)).ravel()
        if not (omegas.size and np.all(np.isfinite(omegas)) and np.all(omegas > 0.0)):
            raise ValueError(f'{self.label()}: a reduced sweep needs finite angular frequencies > 0 rad/s')
        self.ensure_assembled()
        pencil = self.pencil
        extra_matrix, extra_load = self.frequency_terms(float(omegas[0]))
        if extra_matrix is not None or extra_load is not None:
            raise ValueError(f'{self.label()}: frequency-dependent boundary terms (ports) are not supported by a '
                             f"reduced sweep yet: use sweep_method 'direct'")
        if not pencil.free.size:
            raise ValueError(f'{self.label()}: no free unknowns to reduce')
        terms, loads = self.reduced_terms()
        free_ids = self.global_ids[pencil.free]
        for load in extra_loads:
            if load.vector_of is None:
                loads.append(HarmonicLoad(np.asarray(load.vector, dtype=np.complex128).ravel()[free_ids],
                                          load.coefficient))
            else:
                loads.append(HarmonicLoad(vector_of=lambda omega, of=load.vector_of: np.asarray(
                    of(omega), dtype=np.complex128).ravel()[free_ids]))
        if held_offsets_of is not None and pencil.fixed.size:
            fixed_ids = self.global_ids[pencil.fixed]
            loads.append(HarmonicLoad(vector_of=lambda omega: -(pencil.coupling(omega) @ np.asarray(
                held_offsets_of(omega), dtype=np.complex128).ravel()[fixed_ids])))

        def factorize(omega):
            self._factorize(omega)
            return self._factor
        model = ReducedHarmonicModel(terms, loads, factorize, label=self.label())
        if method == 'adaptive':
            info = model.train(omegas, tolerance=tolerance, moments=moments, max_points=max_points, verify=verify)
        else:
            centre = 0.5 * (omegas.min() + omegas.max()) / (2.0 * math.pi)
            vectors, resonance_solve = self.modal_vectors(centre, modal_count, eigen_solver)
            info = model.fit_modal(vectors, omegas, tolerance=tolerance, verify=verify)
            info.update(modes=len(resonance_solve.resonances),
                        mode_frequencies=[resonance.frequency for resonance in resonance_solve.resonances],
                        eigen_seconds=float(resonance_solve.info['seconds']))
            info['warnings'] = list(info['warnings']) + list(resonance_solve.info['warnings'])
        self._reduced = (model, self._load, self._fixed_value)
        self._held_offsets_of = held_offsets_of
        return info

    def reactions(self) -> np.ndarray:
        """Complex128 (n_full,) A u - f of the last solve on the held dofs (0 elsewhere): the flux the conditions supply
        (2D per unit length / per radian axisymmetric); the pencil terms only (frequency_terms not included)."""
        if self._solution is None:
            raise RuntimeError(f'{self.label()}: solve before reading reactions')
        pencil = self.pencil
        residual = pencil.full(self.last_omega) @ self._solution - self.load_vector()
        result = np.zeros(self.values.shape, dtype=np.complex128)
        result[self.global_ids[pencil.fixed]] = residual[pencil.fixed]
        return result

    def eigen_pencil(self, region_elements=None) -> HarmonicEigenPencil:
        """HarmonicEigenPencil (K, M on the free dofs; Dirichlet held at 0) for fem.harmonic_resonances; region_elements
        (bool per element) marks the rows of a region; ValueError with damping terms (quadratic in omega)."""
        self.ensure_assembled()
        pencil = self.pencil
        if pencil.has_damping:
            raise ValueError(f'{self.label()}: a damping term (volume damping or a Robin damping edge) makes the '
                             'eigenproblem quadratic in omega: use a PML or a complex (loss factor) coefficient instead')
        in_region = None
        if region_elements is not None:
            in_region = self._element_rows(np.asarray(region_elements, dtype=bool))[pencil.free]
        return HarmonicEigenPencil(stiffness=pencil.part('stiffness'), mass=pencil.part('mass'),
                                   full_ids=self.global_ids[pencil.free], n_full=len(self.values),
                                   in_window=self.in_window[pencil.free], in_region=in_region)

    # ---- subclass hooks ----
    def face_entities(self, face) -> np.ndarray:
        raise NotImplementedError

    def active_elements(self) -> np.ndarray:
        raise NotImplementedError

    def _assemble_terms(self) -> dict:
        """{'global_ids', 'stiffness' / 'damping' / 'mass' (lists of system CSR), 'boundary_load', 'constraints'
        [(sequence, system positions, values)], 'in_window'}."""
        raise NotImplementedError

    def _source_load(self) -> np.ndarray:
        raise NotImplementedError

    def _element_rows(self, element_mask) -> np.ndarray:
        raise NotImplementedError


# ---- triangles ----
class ScalarHarmonicSystem(HarmonicSystemBase):
    """Complex scalar harmonic system on a TriangleMesh (module header): Lagrange order 1 / 2 (isoparametric P2 on
    curved edges), planar or axisymmetric (x = r), pml = a 2D fem.pml.PMLStretch of the mesh's ring (pml_ring) or
    None; linear_solver = the 'linear' role runtime (SparseDirectSolver; None = host SuperLU)."""
    faces = FACES_2D

    def __init__(self, mesh, order=2, linear_solver=None, *, axisymmetric=False, pml=None, pml_outer='dirichlet',
                 symmetric=True, label=lambda: 'harmonic fem'):
        if order not in (1, 2):
            raise ValueError(f'{label()}: element order must be 1 or 2, got {order!r}')
        self.mesh = mesh
        self.order = int(order)
        self.geometry_order = 1 if self.order == 1 else 2   # D6: isoparametric P2 on curved elements
        self.axisymmetric = bool(axisymmetric)
        self.full = lagrange_dofs(mesh, self.order, geometry_order=self.geometry_order)   # values / display numbering
        self.element_map = lagrange_dofs(mesh, 0, geometry_order=self.geometry_order)     # P0 fields
        self.dofs = None   # DofMap of the active elements (last assembly)
        super().__init__(self.full.n_dofs, linear_solver, pml, pml_outer, symmetric, label)
        self._check_pml()

    def _check_pml(self) -> None:
        pml = self.pml
        if pml is None:
            return
        if pml.dimensions != 2:
            raise ValueError(f'{self.label()}: a triangle mesh needs a 2D PMLStretch, got {pml.dimensions}D')
        widths = ring_widths(self.mesh)
        missing = [face for face in pml.faces if widths[face] <= 0.0]
        if missing:
            raise ValueError(f"{self.label()}: PML on {', '.join(missing)} but the mesh has no layer ring there")
        if self.axisymmetric and 'x_lo' in pml.faces:
            raise ValueError(f'{self.label()}: an axisymmetric PML cannot lie on x_lo (the axis side, r < 0)')

    def active_elements(self) -> np.ndarray:
        if self.active is not None:
            if self.active.shape != (self.mesh.n_elements,):
                raise ValueError(f'{self.label()}: active mask has shape {self.active.shape}, expected '
                                 f'({self.mesh.n_elements},)')
            return self.active
        if self.pml is not None:
            return np.ones(self.mesh.n_elements, dtype=bool)
        return self.mesh.pml_faces == 0

    def face_entities(self, face) -> np.ndarray:
        """Int64 mesh edges a face condition acts on (module header)."""
        mesh = self.mesh
        ringed = bool(((mesh.pml_faces & FACE_BITS[face]) != 0).any())
        if ringed and (self.pml is None or face not in self.pml.faces):
            return window_face_edges(mesh, face)
        return np.asarray(mesh.edge_tags.get(face, np.zeros(0)), dtype=np.int64)

    def _rule_degree(self, default):
        """None (the assembler's default) without a PML, else default + PML_EXTRA_DEGREE (capped at the largest rule)."""
        if self.pml is None:
            return None
        return min(max(int(default) + PML_EXTRA_DEGREE, 2), TRIANGLE_DEGREES[-1])

    def _stretched(self, coefficient, tensor):
        """The coefficient unchanged without a PML, else a callable of its values x the stretch factors at points."""
        pml, axisymmetric = self.pml, self.axisymmetric
        if pml is None:
            return coefficient

        def at_points(points, elements):
            values = _coefficient_values(coefficient, points, elements, tensor)
            if tensor:
                return np.broadcast_to(values, points.shape[:2] + (2, 2)) * pml.tensor_factor(points, axisymmetric)
            return np.broadcast_to(values, points.shape[:2]) * pml.volume_factor(points, axisymmetric)
        return at_points

    def _assemble_terms(self) -> dict:
        mesh, order, axisymmetric = self.mesh, self.order, self.axisymmetric
        active = self.active_elements()
        if not active.any():
            raise ValueError(f'{self.label()}: no active element')
        dofs = lagrange_dofs(mesh, order, active, geometry_order=self.geometry_order)
        self.dofs = dofs
        axis = int(axisymmetric)
        coefficients = self.coefficients
        # with a PML only the ring elements (a corner beyond a layered face) take the stretched callable at the raised
        # rule; the rest keep the plain coefficient at the default rule (as 3D's SplitCoefficient)
        varying = None if self.pml is None else self.pml.in_layer(mesh.points[mesh.triangles[dofs.elements]]).any(axis=1)

        def assembled(assemble, coefficient, tensor, pml_degree):
            if varying is None:
                return assemble(dofs, coefficient, axisymmetric=axisymmetric)
            parts = [assemble(dofs, self._stretched(coefficient, tensor), degree=pml_degree, axisymmetric=axisymmetric,
                              elements=varying)]
            if not varying.all():
                parts.append(assemble(dofs, coefficient, axisymmetric=axisymmetric, elements=~varying))
            return pattern_sum(parts)   # one pattern whatever the values

        stiffness = [assembled(assemble_stiffness, coefficients['a'], True,
                               self._rule_degree(max(2 * order - 2 + axis, 2)))]
        damping, mass = [], []
        mass_degree = self._rule_degree(2 * order + axis)
        for key, terms in (('b', stiffness), ('damping', damping), ('mass', mass)):
            if not is_zero(coefficients[key]):
                terms.append(assembled(assemble_mass, coefficients[key], False, mass_degree))
        boundary_load = np.zeros(dofs.n_dofs, dtype=np.complex128)
        constraints = []
        points = dofs.dof_points()
        for edges, condition, sequence in self._condition_items():
            if not len(edges):
                continue
            if condition.kind == 'dirichlet':
                ids = dofs.edge_dofs(edges)
                constraints.append((sequence, ids, dirichlet_values(condition.value, points[ids])))
                continue
            if condition.kind == 'robin':
                if not is_zero(condition.admittance):
                    stiffness.append(assemble_edge_mass(dofs, edges, condition.admittance, axisymmetric=axisymmetric))
                if not is_zero(condition.damping):
                    damping.append(assemble_edge_mass(dofs, edges, condition.damping, axisymmetric=axisymmetric))
            if not is_zero(condition.value):
                boundary_load = boundary_load + assemble_edge_load(dofs, edges, condition.value,
                                                                   axisymmetric=axisymmetric)
        x0, y0, x1, y1 = mesh.window
        slack = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
        in_window = ((points[:, 0] >= x0 - slack) & (points[:, 0] <= x1 + slack)
                     & (points[:, 1] >= y0 - slack) & (points[:, 1] <= y1 + slack))
        return {'global_ids': dofs.global_ids, 'stiffness': stiffness, 'damping': damping, 'mass': mass,
                'boundary_load': boundary_load, 'constraints': constraints, 'in_window': in_window}

    def _source_load(self) -> np.ndarray:
        dofs, order, axisymmetric = self.dofs, self.order, self.axisymmetric
        load = np.zeros(dofs.n_dofs, dtype=np.complex128)
        source = self.source
        if source is not None and not is_zero(source):
            support = None
            if not callable(source) and np.ndim(source) == 1:   # per mesh element: integrate only where it acts
                support = np.asarray(source)[dofs.elements] != 0
            load = load + assemble_load(dofs, self._stretched(source, False),
                                        degree=self._rule_degree(2 * order + int(axisymmetric) + 2),
                                        axisymmetric=axisymmetric, elements=support)
        if self.point_loads:
            points = np.array([point for point, _ in self.point_loads], dtype=np.float64).reshape(-1, 2)
            location = locator(self.mesh, self.geometry_order).locate(points)
            basis = lagrange_values(order, location.reference)   # (p, n)
            for index, (point, value) in enumerate(self.point_loads):
                element = int(location.element[index])
                position = int(dofs.element_position[element]) if element >= 0 else -1
                if position < 0:
                    raise ValueError(f'{self.label()}: point load at {point.tolist()} m lies outside the active '
                                     'elements')
                scale = value / (RING_ANGLE if axisymmetric else 1.0)
                load[dofs.element_dofs[position]] += basis[index] * scale   # one element: its dofs are distinct
        return load

    def _element_rows(self, element_mask) -> np.ndarray:
        """Bool per system dof touched by the masked active elements."""
        dofs = self.dofs
        rows = np.zeros(dofs.n_dofs, dtype=bool)
        positions = dofs.element_position[np.asarray(element_mask, dtype=bool)]
        rows[dofs.element_dofs[positions[positions >= 0]].ravel()] = True
        return rows

    # ---- fields ----
    def centroid_values(self) -> np.ndarray:
        """Complex128 u at every element's reference centroid (D12 per-element state; 0 on inactive elements)."""
        basis = lagrange_values(self.order, np.array([[1.0 / 3.0, 1.0 / 3.0]]))[0]
        return self.values[self.full.element_dofs] @ basis

    def nodal_field(self) -> tuple:
        """(full-mesh Lagrange DofMap, complex128 coefficients) of u (renderers split re / im / abs / phase)."""
        return self.full, self.values

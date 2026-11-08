"""CPU memory estimates made before a build (E8): MemoryEstimate, combine and the shared per-cell counters.
Pure Python (no cupy): GUI-thread safe. Physics / feature estimators count their own allocations from the code."""
import math
from dataclasses import dataclass, replace

FLOAT32_BYTES = 4
FLOAT64_BYTES = 8
COMPLEX64_BYTES = 8
COMPLEX128_BYTES = 16
INT32_BYTES = 4
BOOL_BYTES = 1
# One CSR entry of a complex128 sparse matrix: value + int32 column index.
COMPLEX128_CSR_ENTRY_BYTES = COMPLEX128_BYTES + INT32_BYTES
# Host SuperLU fill-in heuristic (2D 5-point stencil, MMD ordering): nnz(L + U) ~ this x n log2 n. Calibrate against
# the FDFD / em_modes host_factor_bytes the study log reports (solvers.sparse_direct.SPLU_LIMIT_UNKNOWNS: ~10 GB at 2e6).
LU_ENTRIES_PER_UNKNOWN_LOG2 = 10.0
# ... of a real 3D 7-point system: nnz(L + U) ~ this x n^(4/3) (nested-dissection order of growth; uncalibrated:
# compare with summary.json solvers <physics>.roles.linear.counters.host_factor_bytes)
LU_3D_ENTRIES_PER_UNKNOWN_4_3 = 30.0
# ... of a complex 3D curl-curl system (em_fdfd_3d, 13 entries per row): the same order of growth, a larger constant
# (uncalibrated guess: compare with the eigen solve's factor_host_bytes in the study log)
LU_VECTOR_3D_ENTRIES_PER_UNKNOWN_4_3 = 60.0
FLOAT64_CSR_ENTRY_BYTES = FLOAT64_BYTES + INT32_BYTES


def phase_before_items(phase_bytes, later_bytes) -> int:
    """Bytes to book for a transient phase that runs before run items of later_bytes exist (a single solve's assembly
    before its factor): run items + the largest phase then stay the true peak, max(early + phase, all + other phases)."""
    return max(int(phase_bytes) - int(later_bytes), 0)


def phase_totals(transient_items) -> dict:
    """{phase: summed bytes} of (phase, label, bytes) items."""
    totals = {}
    for phase, _, size in transient_items:
        totals[phase] = totals.get(phase, 0) + int(size)
    return totals


def largest_phase(transient_items) -> tuple:
    """(phase, bytes) of the run phase whose transient items sum largest (('', 0) without any)."""
    return max(phase_totals(transient_items).items(), key=lambda entry: entry[1], default=('', 0))


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated bytes: device (cupy pool) kept by the build (gpu_steady), its build high-water mark (gpu_build_peak,
    >= gpu_steady), device the study adds while it runs (gpu_run = run_items, kept once allocated, + the largest phase of
    transient_items ((phase, label, bytes): freed when the phase ends; phases never overlap), never seen by the build
    ledger), device outside the pool (gpu_external: cuDSS factors alive through the run, external_transient_items:
    factors alive only in one run phase, + external_extra: the largest factorisation's peak on top), host (CPU
    factorisations, analyses); items / host_items / run_items / external_items = ((label, bytes), ...)."""
    gpu_steady: int = 0
    gpu_build_peak: int = 0
    host: int = 0
    items: tuple = ()
    host_items: tuple = ()
    gpu_run: int = 0
    run_items: tuple = ()
    transient_items: tuple = ()
    gpu_external: int = 0
    external_extra: int = 0
    external_items: tuple = ()
    external_transient_items: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'gpu_steady', int(self.gpu_steady))
        object.__setattr__(self, 'gpu_build_peak', max(int(self.gpu_build_peak), int(self.gpu_steady)))
        for name in ('host', 'gpu_run', 'gpu_external', 'external_extra'):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ('items', 'host_items', 'run_items', 'external_items'):
            object.__setattr__(self, name, tuple((str(label), int(size)) for label, size in getattr(self, name)))
        for name in ('transient_items', 'external_transient_items'):
            object.__setattr__(self, name, tuple((str(phase), str(label), int(size))
                                                 for phase, label, size in getattr(self, name)))

    @classmethod
    def from_items(cls, items=(), build_extra=0, host_items=(), run_items=(), transient_items=(), external_items=(),
                   external_extra=0, external_transient_items=()):
        """Estimate whose gpu_steady / host / gpu_external are the sums of the items (zero-byte items dropped), gpu_run
        = run_items + the largest transient phase; gpu_build_peak = gpu_steady + build_extra, the transient device bytes
        of this part's own build step; external_extra = the transient cuDSS bytes of its largest factorisation;
        external_transient_items ((phase, label, bytes)): factors alive only in that run phase (a factor alive in
        several phases is listed once per phase)."""
        items, host_items, run_items, external_items = (
            tuple((label, int(size)) for label, size in group if int(size) > 0)
            for group in (items, host_items, run_items, external_items))
        transient_items, external_transient_items = (
            tuple((phase, label, int(size)) for phase, label, size in group if int(size) > 0)
            for group in (transient_items, external_transient_items))
        steady = sum(size for _, size in items)
        return cls(steady, steady + max(int(build_extra), 0), sum(size for _, size in host_items), items, host_items,
                   sum(size for _, size in run_items) + largest_phase(transient_items)[1], run_items, transient_items,
                   sum(size for _, size in external_items), max(int(external_extra), 0), external_items,
                   external_transient_items)

    @property
    def build_extra(self) -> int:
        """Transient device bytes on top of gpu_steady during the build."""
        return self.gpu_build_peak - self.gpu_steady

    @property
    def gpu_peak(self) -> int:
        """Pool high-water mark of build and run: max(gpu_build_peak, gpu_steady + gpu_run)."""
        return max(self.gpu_build_peak, self.gpu_steady + self.gpu_run)

    @property
    def external_peak(self) -> int:
        """Device bytes outside the pool at their high-water mark: gpu_external + the largest phase of
        external_transient_items + external_extra."""
        return (self.gpu_external + max(phase_totals(self.external_transient_items).values(), default=0)
                + self.external_extra)

    @property
    def device_peak(self) -> int:
        """Device high-water mark the GPU budget is checked against: the build peak, or steady + run items + per run
        phase its pool transients and the factors alive in it (the largest phase), + the run-long factors and
        external_extra."""
        pool, external = phase_totals(self.transient_items), phase_totals(self.external_transient_items)
        run_phase = max((pool.get(phase, 0) + external.get(phase, 0) for phase in set(pool) | set(external)),
                        default=0)
        run = self.gpu_steady + sum(size for _, size in self.run_items) + run_phase
        return max(self.gpu_build_peak, run) + self.gpu_external + self.external_extra

    def labelled(self, prefix):
        """Copy with every item label (and transient phase) prefixed by '<prefix>: '."""
        def prefixed(group):
            return tuple((f'{prefix}: {label}', size) for label, size in group)

        def phased(group):
            return tuple((f'{prefix}: {phase}', f'{prefix}: {label}', size) for phase, label, size in group)
        return replace(self, items=prefixed(self.items), host_items=prefixed(self.host_items),
                       run_items=prefixed(self.run_items), external_items=prefixed(self.external_items),
                       transient_items=phased(self.transient_items),
                       external_transient_items=phased(self.external_transient_items))

    def largest_items(self, count=3, host=False, run=False, external=False):
        """The `count` largest (label, bytes) items (host / run (transient items included) / external items with host
        / run / external), largest first."""
        group = (self.host_items if host else self.run_items + tuple((label, size) for _, label, size in
                                                                     self.transient_items) if run
                 else self.external_items + self._external_transient_once() if external else self.items)
        return sorted(group, key=lambda item: -item[1])[:count]

    def _external_transient_once(self):
        """(label, bytes) of every phase-bound factor once (a factor listed in several phases: its largest entry)."""
        sizes = {}
        for _, label, size in self.external_transient_items:
            sizes[label] = max(sizes.get(label, 0), size)
        return tuple(sizes.items())

    def device_items(self):
        """Every device item (kept, run, transient and outside the pool) as (label, bytes), a factor of several
        phases once."""
        transient = {}
        for _, label, size in self.transient_items:   # an item listed in several phases (the pencil) once
            transient[label] = max(transient.get(label, 0), size)
        return (self.items + self.run_items + tuple(transient.items()) + self.external_items
                + self._external_transient_once())

    def as_dict(self) -> dict:
        """JSON-ready record (check.json memory_estimate): byte totals and the items, largest first."""
        def records(group):
            return [{'label': label, 'bytes': size} for label, size in sorted(group, key=lambda item: -item[1])]
        phase, phase_bytes = largest_phase(self.transient_items)
        return {'gpu_steady': self.gpu_steady, 'gpu_build_peak': self.gpu_build_peak, 'gpu_run': self.gpu_run,
                'gpu_peak': self.gpu_peak, 'gpu_external': self.gpu_external, 'external_extra': self.external_extra,
                'device_peak': self.device_peak, 'host': self.host, 'items': records(self.items),
                'host_items': records(self.host_items), 'run_items': records(self.run_items),
                'transient_items': [{'phase': phase_name, 'label': label, 'bytes': size} for phase_name, label, size
                                    in sorted(self.transient_items, key=lambda item: -item[2])],
                'largest_phase': {'phase': phase, 'bytes': phase_bytes} if phase_bytes else None,
                'external_items': records(self.external_items),
                'external_transient_items': [{'phase': phase_name, 'label': label, 'bytes': size}
                                             for phase_name, label, size in sorted(self.external_transient_items,
                                                                                   key=lambda item: -item[2])],
                'external_peak': self.external_peak}


def combine(estimates, build_cache_bytes=0) -> MemoryEstimate:
    """Sum of parts built one after another: steady, host, run, external and items add; the build peak = total steady +
    the build caches alive during the whole build (build_cache_bytes) + the largest transient extra of a single part;
    transient run phases (pool and phase-bound factors) and the external factorisation extra: the largest of all parts
    (they never overlap)."""
    estimates = list(estimates)
    steady = sum(estimate.gpu_steady for estimate in estimates)
    largest_extra = max((estimate.build_extra for estimate in estimates), default=0)
    transient_items = tuple(item for estimate in estimates for item in estimate.transient_items)
    run_items = tuple(item for estimate in estimates for item in estimate.run_items)
    external_items = tuple(item for estimate in estimates for item in estimate.external_items)
    return MemoryEstimate(steady, steady + max(int(build_cache_bytes), 0) + largest_extra,
                          sum(estimate.host for estimate in estimates),
                          tuple(item for estimate in estimates for item in estimate.items),
                          tuple(item for estimate in estimates for item in estimate.host_items),
                          sum(size for _, size in run_items) + largest_phase(transient_items)[1],
                          run_items, transient_items,
                          sum(size for _, size in external_items),
                          max((estimate.external_extra for estimate in estimates), default=0), external_items,
                          tuple(item for estimate in estimates for item in estimate.external_transient_items))


def grid_transfer_estimate(source_edges, target_edges, kind) -> tuple:
    """(nnz, bytes) of an S7.2 GridTransfer between two tensor grids: one float32 CSR matrix per axis (value + int32
    column per entry, int32 indptr per target row); edges = per-axis edge sequences (x, y[, z]) in one frame. kind
    'conservative': one entry per piece of the merged edges (overlap weights), 'interpolate': two per target cell."""
    if kind not in ('conservative', 'interpolate'):
        raise ValueError(f"grid transfer kind must be 'conservative' or 'interpolate', got {kind!r}")
    total_nnz, total_bytes = 0, 0
    for source, target in zip(source_edges, target_edges):
        n_target = len(target) - 1
        if kind == 'conservative':
            nnz = len({float(edge) for edge in source} | {float(edge) for edge in target}) - 1
        else:
            nnz = 2 * n_target
        total_nnz += nnz
        total_bytes += nnz * (FLOAT32_BYTES + INT32_BYTES) + (n_target + 1) * INT32_BYTES
    return total_nnz, total_bytes


def window_cell_count(mesh) -> int:
    """Physical-window cells of a MeshEstimate in the xy plane (PML cells excluded; 3D: per z layer, not times nz)."""
    sides = mesh.npml_sides
    return (int(mesh.nx) - sides['x_lo'] - sides['x_hi']) * (int(mesh.ny) - sides['y_lo'] - sides['y_hi'])


def host_lu_bytes(n_unknowns, dimensions=2) -> int:
    """Heuristic host bytes of a complex128 SuperLU factor (L + U) of a 2D 5-point system (n log2 n fill) or a 3D
    curl-curl system (dimensions 3: n^(4/3) fill)."""
    n_unknowns = max(int(n_unknowns), 2)
    if dimensions == 3:
        entries = LU_VECTOR_3D_ENTRIES_PER_UNKNOWN_4_3 * n_unknowns ** (4.0 / 3.0)
    else:
        entries = LU_ENTRIES_PER_UNKNOWN_LOG2 * n_unknowns * math.log2(n_unknowns)
    return int(entries * COMPLEX128_CSR_ENTRY_BYTES)


def host_counted_lu_bytes(triangle_entries, n_unknowns) -> int:
    """Host bytes of a complex128 SuperLU factor (L + U) from counted entries of one factor triangle (diagonal
    included, e.g. fem.dissection pairs x dofs per site^2): both triangles, the diagonal once."""
    n_unknowns = int(n_unknowns)
    return int(max(2.0 * float(triangle_entries) - n_unknowns, n_unknowns) * COMPLEX128_CSR_ENTRY_BYTES)


def host_real_lu_bytes(n_unknowns, dimensions=2) -> int:
    """Heuristic host bytes of a float64 SuperLU factor (L + U) of a 2D 5-point (n log2 n fill) or 3D 7-point (n^(4/3)
    fill) diffusion system."""
    n_unknowns = max(int(n_unknowns), 2)
    if dimensions == 3:
        entries = LU_3D_ENTRIES_PER_UNKNOWN_4_3 * n_unknowns ** (4.0 / 3.0)
    else:
        entries = LU_ENTRIES_PER_UNKNOWN_LOG2 * n_unknowns * math.log2(n_unknowns)
    return int(entries * FLOAT64_CSR_ENTRY_BYTES)


# Window arrays (solve dtype) of the flux post-processing after a solve (DiffusionSystem.flux_at_centres): face flows
# and centre averages with their temporaries; + the cross-flux quads (c_xy); dissipation_density keeps its two
# face share arrays once called (lazy) and takes about as many temporaries as the fluxes
_FLUX_ARRAYS = {2: 6, 3: 9}
_CROSS_FLUX_ARRAYS = 6
_DISSIPATION_SHARE_ARRAYS = 2
# assemble's window locals alive while linear_solver.factorize runs (diagonal_extra, effective_diagonal, fixed_values,
# coupling_to_fixed in the solve dtype; unconnected, fixed, loss_anchor_cells bool); the pattern-build and anchor-check
# temporaries are freed by then
_FACTORIZE_ALIVE_VALUE_ARRAYS = 4
_FACTORIZE_ALIVE_MASKS = 3


def diffusion_system_estimate(window_cells, precision, cross_term=False, transient=False,
                              dimensions=2, mixed=False, multigrid=False, direct=False,
                              azimuthal=False, wrap=False, flux=False, dissipation=False) -> MemoryEstimate:
    """DiffusionSystem (core.fv_diffusion) on window_cells (3D: all window voxels, dimensions=3): prepare buffers, CSR
    pattern, assembled operators and vectors (steady; azimuthal: the kept azimuthal diagonal of an (r, z) order
    m != 0); build extra: the larger of the pattern build + anchor check and (multigrid) the setup while assemble's
    locals live; mixed
    (a 'mixed' pcg, precision 'float64'): its float32 operator copy and refinement vectors; multigrid (the pcg
    multigrid preconditioner): its hierarchy, setup products and host aggregation; direct (a host SuperLU
    sparse_direct linear solver, the 2D cpu Automatic): the host factor and its float64 CSC input copy (a cuDSS one:
    the caller adds fem.diffusion.cudss_diffusion_estimate). wrap (phi wrap of a polar
    window or a cylindrical sector, symmetry 2.21): 2 more slots in the dense build array; its CSR entries stay within
    the 5 / 7 per row. flux (a caller computing flux_at_centres after its solves): its temporaries, transient phase
    'flux post-processing'; dissipation (2D dissipation_density): the share arrays it keeps (run items)."""
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    entries_per_row = 9 if cross_term else 5
    cells = int(window_cells)
    # prepare: cell_area, rhs, unknown, extra_diag / _rhs, fixed values, rhs_full buffer, CG solution + PCGSolver
    # residual / direction / CSR product (solvers.pcg.workspace_bytes) + fixed mask + the cached uint8 anchor code
    # (face-area divisors are separable 1D factors)
    buffers = (11 * value_bytes + BOOL_BYTES + 1) * cells
    # indptr + diagonal positions per row, indices / entry_rows / slot_positions per entry (int32)
    pattern = (2 * INT32_BYTES + 3 * INT32_BYTES * entries_per_row) * cells
    operators = 2 * entries_per_row * value_bytes * cells   # operator_full and A_solve data (shared indices)
    # inverse diagonal, free, rhs_offset, fixed_values, rhs_drive, 2 face conductances, assembled c_xx / c_yy, the
    # conduction diagonal (+ C/dt); the loss-anchor mask (bool); the kept c_xy quad coefficients and their bool mask
    assembly = ((10 + (1 if transient else 0) + (1 if azimuthal else 0) + (1 if cross_term else 0)) * value_bytes
                + (2 if cross_term else 1) * BOOL_BYTES) * cells
    # pattern build (slot mask, int64 flatnonzero), dense slot array (+ 2 phi-wrap slots), assembly temporaries (the
    # kept polar r_c dphi lengths among them), anchor check at 2x resolution (link image, int32 labels and their copy)
    n_slots = entries_per_row + (2 if wrap else 0)   # fv_diffusion_3d._SlotLayout
    build_extra = (n_slots * (BOOL_BYTES + FLOAT64_BYTES) + (n_slots + 7) * value_bytes
                   + 4 * (BOOL_BYTES + 2 * INT32_BYTES) + 30) * cells
    if dimensions == 3:   # 7-point operator (core.fv_diffusion_3d), no cross term, no face-length arrays
        entries_per_row = 7
        # cell volume, rhs, unknown, extra_diag / _rhs, fixed values, rhs_full buffer, 4 CG vectors + fixed mask, the
        # cached uint8 anchor code
        buffers = (11 * value_bytes + BOOL_BYTES + 1) * cells
        pattern = (2 * INT32_BYTES + 3 * INT32_BYTES * entries_per_row) * cells   # slot_positions instead of csr_order
        operators = 2 * entries_per_row * value_bytes * cells
        # as 2D with 3 face conductances and c_zz
        assembly = (12 + (1 if transient else 0)) * value_bytes * cells + BOOL_BYTES * cells
        # pattern build (slot mask, int64 flatnonzero), dense slot array, anchor check at 2x resolution (link image,
        # int32 labels and their copy)
        n_slots = 9 if wrap else entries_per_row   # fv_diffusion_3d._SlotLayout
        build_extra = (n_slots * (BOOL_BYTES + FLOAT64_BYTES) + (n_slots + 6) * value_bytes
                       + 8 * (BOOL_BYTES + 2 * INT32_BYTES) + 30) * cells
    items = [('solver buffers', buffers), ('sparse pattern (int32)', pattern), ('assembled operators', operators),
             ('assembly vectors', assembly)]
    if mixed:
        from ..solvers.pcg import mixed_extra_bytes   # numpy only (the fv_diffusion exception of the core rule)
        items.append(('mixed-precision CG (float32 operator copy, refinement vectors)',
                      mixed_extra_bytes(cells, entries_per_row * cells)))
    host_items = []
    if multigrid:
        from ..solvers import multigrid as amg   # numpy only (the fv_diffusion exception of the core rule)
        items.append(('multigrid hierarchy (operators, prolongations, vectors)',
                      amg.hierarchy_bytes(cells, value_bytes, entries_per_row)))
        alive = (_FACTORIZE_ALIVE_VALUE_ARRAYS * value_bytes + _FACTORIZE_ALIVE_MASKS * BOOL_BYTES) * cells
        build_extra = max(build_extra, alive + amg.setup_extra_bytes(cells, value_bytes, entries_per_row))
        host_items.append(('multigrid aggregation (transient)', amg.aggregation_host_bytes(cells, entries_per_row)))
    if direct:   # solvers.sparse_direct.DiffusionDirectSolver: one factor per assembly, the previous one dropped
        host_items.append(('host SuperLU factor L + U (float64)', host_real_lu_bytes(cells, dimensions)))
        host_items.append(('host CSC copy of the operator (float64)',
                           FLOAT64_CSR_ENTRY_BYTES * entries_per_row * cells + INT32_BYTES * cells))
    run_items, transient_items = [], []
    if flux:   # freed when the caller's post-processing returns
        n_flux = _FLUX_ARRAYS[3 if dimensions == 3 else 2] + (_CROSS_FLUX_ARRAYS if cross_term else 0)
        transient_items.append(('flux post-processing', f'flux post-processing temporaries ({n_flux} window arrays)',
                                n_flux * value_bytes * cells))
    if dissipation and dimensions != 3:
        run_items.append(('dissipation face shares (kept once computed)',
                          _DISSIPATION_SHARE_ARRAYS * value_bytes * cells))
    return MemoryEstimate.from_items(items, build_extra=build_extra, host_items=host_items, run_items=run_items,
                                     transient_items=transient_items)

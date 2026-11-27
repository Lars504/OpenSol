"""CPU memory estimates made before a build (E8): MemoryEstimate, combine and the shared per-cell counters.
Pure Python (no cupy): GUI-thread safe. Physics / feature estimators count their own allocations from the code."""
import math
from dataclasses import dataclass, replace

FLOAT32_BYTES = 4
FLOAT64_BYTES = 8
COMPLEX64_BYTES = 8
COMPLEX128_BYTES = 16
INT32_BYTES = 4
INT64_BYTES = 8
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
# label tag of an item a solver job (role runtime) holds: '<role> solver: <item>' (solver_job_items / solver_job_of)
SOLVER_JOB_TAG = ' solver: '
# MemoryEstimate.peak_items labels of the device bytes no item lists: the build's temporaries on top of the kept
# arrays, and the largest factorisation's bytes on top of its factor (external_extra)
BUILD_TEMPORARIES_LABEL = 'build temporaries (build caches and the largest build step)'
FACTORISATION_EXTRA_LABEL = 'direct-solver factorisation, on top of its factor'


def solver_job_items(role, items) -> list:
    """Copies of (label, bytes) / (phase, label, bytes) items labelled '<role> solver: <label>': memory the role's
    solver job holds while it solves (factor, factor copies, Krylov basis); the solve flow books it at that job."""
    return [(*item[:-2], f'{role}{SOLVER_JOB_TAG}{item[-2]}', item[-1]) for item in items]


def solver_job_of(label) -> tuple:
    """(role, item label) of a solver_job_items label; ('', label) of any other."""
    role, tag, item_label = str(label).partition(SOLVER_JOB_TAG)
    return (role, item_label) if tag and role.isidentifier() else ('', label)


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
        return max(self.gpu_build_peak, self._run_peak()) + self.gpu_external + self.external_extra

    @property
    def run_peak_phase(self) -> str:
        """The run phase device_peak counts: its pool transients + phase-bound factors sum largest (the first of equal
        ones; '' without phases)."""
        pool, external = phase_totals(self.transient_items), phase_totals(self.external_transient_items)
        return max(dict.fromkeys((*pool, *external)), key=lambda phase: pool.get(phase, 0) + external.get(phase, 0),
                   default='')

    def _run_peak(self) -> int:
        """gpu_steady + the run items + the run_peak_phase's pool transients and phase-bound factors."""
        phase = self.run_peak_phase
        return (self.gpu_steady + sum(size for _, size in self.run_items)
                + sum(size for name, _, size in self.transient_items + self.external_transient_items if name == phase))

    @property
    def peak_in_build(self) -> bool:
        """True when device_peak is reached in the build (its high-water mark above the run's), not in the study."""
        return self.gpu_build_peak > self._run_peak()

    def peak_items(self) -> tuple:
        """(label, bytes) device items alive at device_peak, summing to it (from_items / combine totals): kept items,
        run-long factors, run + run_peak_phase (peak_in_build: BUILD_TEMPORARIES_LABEL), FACTORISATION_EXTRA_LABEL."""
        if self.peak_in_build:
            pool_items = self.items + ((BUILD_TEMPORARIES_LABEL, self.build_extra),)
        else:
            phase = self.run_peak_phase
            pool_items = self.items + self.run_items + tuple(
                (label, size) for name, label, size in self.transient_items + self.external_transient_items
                if name == phase)
        extra = ((FACTORISATION_EXTRA_LABEL, self.external_extra),) if self.external_extra else ()
        return pool_items + self.external_items + extra

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
# window locals alive while linear_solver.factorize runs in a re-assembly or an in-place refresh: every value array
# is deleted (or handed to the system) before it (DiffusionSystem.assemble, _assemble_fused, _refresh_diagonal_terms);
# two masks stay (`fixed` and the new anchor code of a deferred gpu anchor check until the next solve; `fixed` and the
# loss anchors of a refresh). A first assembly holds none
_FACTORIZE_ALIVE_VALUE_ARRAYS = 0
_FACTORIZE_ALIVE_MASKS = 2
# window arrays (solve dtype) of DiffusionSystem.refresh_capacity at its largest (the rhs offset): the new C/dt beside
# the old one, the new solve diagonal, the off-diagonal product A_eff u_fix, the boundary rhs and the offset pair (its
# difference and the new offset beside the old one)
_REFRESH_ARRAYS = 6


def _anchor_check_bytes(ndim) -> int:
    """Bytes per window cell of an anchor labelling (DiffusionSystem._label_anchors) on top of the assembly: the bool
    link image and its int32 labels at 2x resolution (2^ndim pixels per cell), the conducting and anchor masks, the
    int32 scan and gathered labels of labels[anchors] (every cell an anchor with C/dt or an azimuthal order)."""
    return 2 ** ndim * (BOOL_BYTES + INT32_BYTES) + 2 * BOOL_BYTES + 2 * INT32_BYTES


def _assembly_stage_bytes(held, value_bytes, ndim, entries_per_row, n_slots, azimuthal=False, maps_kept=False,
                          first=True, host_twins=False) -> dict:
    """{stage: bytes per window cell} a build_laplacian assembly of DiffusionSystem.assemble holds at each stage, in
    the order the code allocates (arrays are counted from their allocation to their del; the solver holds nothing yet:
    the caller adds what it keeps through a re-assembly). held: what is alive when build_laplacian starts (prepare
    buffers, the caller's C/dt and arrays, the owned coefficient copies, c_xy quads). first: the pattern is built here
    and holds one per-entry map through the gather (LaplacianPattern.take_entry_rows / keep_assembly_maps come after
    it), else both kept maps; maps_kept: the system keeps both maps (re-assemblies through build_laplacian), else
    only entry_rows until the elimination; host_twins: the numpy twin of the elimination gathers free[row] /
    free[column] (one entry-sized temporary at a time; the cupy kernel reads them raw)."""
    index = (2 + entries_per_row) * INT32_BYTES   # indptr, diagonal positions, indices: the kept pattern
    one_map = entries_per_row * INT32_BYTES       # slot_positions or entry_rows
    stages = {}
    if first:   # LaplacianPattern: indptr and diagonal positions, then the largest of
        stages['pattern build'] = held + 2 * INT32_BYTES + max(
            # the slot mask, the int32 scan of the device flatnonzero and its int64 positions
            n_slots * (BOOL_BYTES + INT32_BYTES) + entries_per_row * INT64_BYTES,
            # the slot mask, the int64 positions and their int32 copy (slot_positions)
            n_slots * BOOL_BYTES + entries_per_row * (INT64_BYTES + INT32_BYTES),
            # slot_positions, slot % n_slots (then the rows) and the gathered column offsets (the indices)
            3 * one_map)
    # the pattern and its map(s), the face conductances, the dense slot array, the boundary rhs and the gathered
    # A_eff values (the conductance temporaries before the slots: at most 11 arrays with held edges, below these)
    gather_maps = one_map if first else 2 * one_map
    stages['entry gather'] = held + index + gather_maps + (ndim + n_slots + 1 + entries_per_row) * value_bytes
    # from the gather to the solver setup: conductances, the one CSR value array (A_eff, eliminated in place into
    # A_solve), boundary rhs (the rhs drive in place), conduction diagonal, fixed values, azimuthal diagonal; fixed
    # mask, loss anchors, anchor code (the diagonal sums and the pinned-cell test before them: 2 arrays, below the
    # anchor check)
    maps = (2 if maps_kept else 1) * one_map
    checked = (held + index + maps + (ndim + entries_per_row + 3 + int(bool(azimuthal))) * value_bytes
               + 3 * BOOL_BYTES)
    stages['anchor check'] = checked + _anchor_check_bytes(ndim)
    # the free mask and A_eff u_fix (a CSR product: + its canonical-format test, one bool per row), in-place elimination
    stages['elimination'] = (checked + 2 * value_bytes + BOOL_BYTES
                             + (entries_per_row * value_bytes if host_twins else 0))
    # entry_rows released unless both maps are kept; free mask, then 3 more arrays at a time: A_eff u_fix with the
    # two temporaries of the fixed-diagonal update or with the rhs-offset pair, the offset with the Jacobi pair
    stages['rhs offset'] = checked - (0 if maps_kept else one_map) + 4 * value_bytes
    return stages


def diffusion_full_operator_bytes(window_cells, entries_per_row, n_slots, value_bytes) -> int:
    """Transient device bytes of DiffusionSystem.full_operator (thermal noise, parity checks): the dense slot array,
    the rebuilt CSR values, a released pattern's slot_positions rebuild (bool slot mask, int64 flatnonzero, int32 copy)
    and the diagonal with its loss / capacity sum."""
    return int(window_cells) * (int(n_slots) * (int(value_bytes) + BOOL_BYTES)
                                + int(entries_per_row) * (int(value_bytes) + FLOAT64_BYTES + INT32_BYTES)
                                + 2 * int(value_bytes))


def diffusion_system_estimate(window_cells, precision, cross_term=False, transient=False,
                              dimensions=2, mixed=False, multigrid=False, direct=False,
                              azimuthal=False, wrap=False, flux=False, dissipation=False,
                              backend=None, conformal=False, pcg=True, refactorizes=False) -> MemoryEstimate:
    """DiffusionSystem (core.fv_diffusion) on window_cells (3D: all window voxels, dimensions=3): prepare buffers, CSR
    pattern, the one kept operator (A_solve: no A_eff CSR is kept) and assembly vectors (steady; azimuthal: the kept
    azimuthal diagonal of an (r, z) order m != 0; ONE assembly: a re-assembly drops its predecessor first,
    DiffusionSystem._release_assembly); pcg (the resolved 'linear' solver is pcg, fem.diffusion.pcg_linear_solver;
    False: a direct / banded solver, no CG workspace and no kept Jacobi diagonal); build extra:
    the larger of the pattern build + anchor check and (multigrid) the setup while assemble's locals live; conformal
    (solver_defaults.conformal_boundaries reaching a 2D system: cut cells) keeps the assembly maps; mixed
    (a 'mixed' pcg, precision 'float64'): its float32 operator copy and refinement vectors; multigrid (the pcg
    multigrid preconditioner): its hierarchy, setup products and the host aggregation (not on backend 'gpu', which
    aggregates on the device inside the setup's transients; None: booked); direct (a host SuperLU
    sparse_direct linear solver, the 2D cpu Automatic: True or 'scipy_splu'): the host factor and its float64 CSC input
    copy; a host direct back end name (MUMPS / PARDISO, solvers.host_direct.HOST_DIRECT_BACK_ENDS): its heuristic
    factor (host_direct.factor_host_bytes, uncalibrated) and the arrays the factor object keeps beside it
    (solver_host_bytes) instead (a cuDSS one: the caller adds fem.diffusion.cudss_diffusion_estimate). wrap (phi wrap
    of a polar window or a cylindrical sector, symmetry 2.21): 2 more slots in the dense build array; its CSR entries stay within
    the 5 / 7 per row. flux (a caller computing flux_at_centres after its solves): its temporaries, transient phase
    'flux post-processing'; dissipation (2D dissipation_density): the share arrays it keeps (run items); refactorizes
    (the run re-assembles: k / C/dt follow a state or t): a re-assembly's temporaries (phase 're-assembly') and
    (multigrid) the solver re-setup (phase 'refactorisation') beside the run items."""
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    entries_per_row = 9 if cross_term else 5
    cells = int(window_cells)
    # prepare: cell_area, rhs, unknown (the CG iterates on it: no solution vector), extra_diag / _rhs, fixed values,
    # rhs_full buffer (+ pcg: the PCGSolver residual / direction / CSR product, solvers.pcg.workspace_bytes) + fixed
    # mask + the cached uint8 anchor code (face-area divisors are separable 1D factors)
    workspace = 3 if pcg else 0
    buffers = ((7 + workspace) * value_bytes + BOOL_BYTES + 1) * cells
    # (pcg) inverse diagonal, free, rhs_offset, fixed_values, rhs_drive, 2 face conductances, assembled c_xx / c_yy,
    # the conduction diagonal (+ C/dt); the loss-anchor mask (bool); the kept c_xy quad coefficients and their bool mask
    assembly = ((9 + int(pcg) + (1 if transient else 0) + (1 if azimuthal else 0) + (1 if cross_term else 0))
                * value_bytes + (2 if cross_term else 1) * BOOL_BYTES) * cells
    # pattern build (slot mask, int64 flatnonzero), dense slot array (+ 2 phi-wrap slots) and the A_eff values taken
    # from it (an assembly transient now, through the elimination), assembly temporaries (the kept polar r_c dphi
    # lengths among them), anchor check at 2x resolution (link image, int32 labels and their copy)
    n_slots = entries_per_row + (2 if wrap else 0)   # fv_diffusion_3d._SlotLayout
    build_extra = (n_slots * (BOOL_BYTES + FLOAT64_BYTES) + (n_slots + 7 + entries_per_row) * value_bytes
                   + 4 * (BOOL_BYTES + 2 * INT32_BYTES) + 30) * cells
    if dimensions == 3:   # 7-point operator (core.fv_diffusion_3d), no cross term, no face-length arrays
        entries_per_row = 7
        # cell volume, rhs, unknown, extra_diag / _rhs, fixed values, rhs_full buffer (+ pcg: 3 CG vectors) + fixed
        # mask, the cached uint8 anchor code
        buffers = ((7 + workspace) * value_bytes + BOOL_BYTES + 1) * cells
        # as 2D with 3 face conductances and c_zz
        assembly = (11 + int(pcg) + (1 if transient else 0)) * value_bytes * cells + BOOL_BYTES * cells
        # pattern build (slot mask, int64 flatnonzero), dense slot array and the A_eff values, anchor check at 2x
        # resolution (link image, int32 labels and their copy)
        n_slots = 9 if wrap else entries_per_row   # fv_diffusion_3d._SlotLayout
        build_extra = (n_slots * (BOOL_BYTES + FLOAT64_BYTES) + (n_slots + 6 + entries_per_row) * value_bytes
                       + 8 * (BOOL_BYTES + 2 * INT32_BYTES) + 30) * cells
    # A_solve data (the pattern's indices shared); A_eff is never kept (DiffusionSystem.full_operator rebuilds it)
    operators = entries_per_row * value_bytes * cells
    # indptr + diagonal positions per row, indices per entry (int32); the per-entry assembly maps (slot_positions,
    # entry_rows: int32) stay only where re-assemblies run build_laplacian (c_xy, phi wrap, 2D cut cells:
    # DiffusionSystem._fused_pattern), else they live through the first assembly's build only
    pattern = (2 * INT32_BYTES + INT32_BYTES * entries_per_row) * cells
    assembly_maps = 2 * INT32_BYTES * entries_per_row * cells
    if cross_term or wrap or (conformal and dimensions != 3):
        pattern += assembly_maps
    else:
        build_extra += assembly_maps
    reassembly = 0
    if refactorizes:   # a run re-assembly (the pattern kept): the new coefficient copies beside the old ones until the
        # anchor check (DiffusionSystem._release_compared_terms); fused kernels: + A_eff values and the Jacobi output;
        # build_laplacian (c_xy, wrap, cut cells): its build temporaries (an upper bound: no pattern build). The anchor
        # check's labelling runs only when a c > 0 pattern moved (not booked on the fused path)
        ndim = 3 if dimensions == 3 else 2
        fused = not (cross_term or wrap or (conformal and dimensions != 3))
        reassembly = ((entries_per_row + ndim + 1) * value_bytes * cells if fused
                      else build_extra + ndim * value_bytes * cells)
    items = [('solver buffers', buffers), ('sparse pattern (int32)', pattern),
             ('assembled operator (A_solve)', operators), ('assembly vectors', assembly)]
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
        setup = alive + amg.setup_extra_bytes(cells, value_bytes, entries_per_row)
        build_extra = max(build_extra, setup)
        if not (backend == 'gpu' and amg.DEVICE_AGGREGATION):   # the device twin holds no host copy of the CSR
            host_items.append(('multigrid aggregation (transient)', amg.aggregation_host_bytes(cells, entries_per_row)))
        # aggregates and tentative prolongators between setups: numpy arrays (on the cpu backend beside its hierarchy)
        kept_host = ('multigrid aggregates and tentative prolongators (host, kept)', amg.kept_host_bytes(cells))
        (items if backend == 'cpu' else host_items).append(kept_host)
    if direct:   # solvers.sparse_direct.DiffusionDirectSolver: one factor per assembly, the previous one dropped
        from ..solvers import host_direct   # numpy only (a function-level exception of the core rule)
        if direct in host_direct.HOST_DIRECT_BACK_ENDS:   # real SPD: PARDISO Cholesky / MUMPS LDL^T
            mixed = host_direct.host_direct_precision(direct) == 'mixed'
            host_items.append((f"host {host_direct.factor_label(direct)} factor ({'float32' if mixed else 'float64'}, "
                               'heuristic, uncalibrated)',
                               host_direct.factor_host_bytes(cells, dimensions, complex_values=False, symmetric=True,
                                                             mixed=mixed)))
            host_items.append((f'host {host_direct.factor_label(direct)} solver arrays (equilibrated operator, its '
                               'library copy, row scale; mixed: refinement vectors)',
                               host_direct.solver_host_bytes(cells, entries_per_row * cells, complex_values=False,
                                                             mixed=mixed, symmetric=True)))
        else:
            host_items.append(('host SuperLU factor L + U (float64)', host_real_lu_bytes(cells, dimensions)))
            host_items.append(('host CSC copy of the operator (float64)',
                               FLOAT64_CSR_ENTRY_BYTES * entries_per_row * cells + INT32_BYTES * cells))
    run_items, transient_items = [], []
    if reassembly:
        transient_items.append(('re-assembly', 'DiffusionSystem re-assembly temporaries (run)', reassembly))
        if multigrid:   # every re-assembly's factorize (fresh or lagged setup): the build's setup peak, in the run
            transient_items.append(('refactorisation', 'multigrid re-setup (run)', setup))
    if flux:   # freed when the caller's post-processing returns
        n_flux = _FLUX_ARRAYS[3 if dimensions == 3 else 2] + (_CROSS_FLUX_ARRAYS if cross_term else 0)
        transient_items.append(('flux post-processing', f'flux post-processing temporaries ({n_flux} window arrays)',
                                n_flux * value_bytes * cells))
    if dissipation and dimensions != 3:
        run_items.append(('dissipation face shares (kept once computed)',
                          _DISSIPATION_SHARE_ARRAYS * value_bytes * cells))
    return MemoryEstimate.from_items(items, build_extra=build_extra, host_items=host_items, run_items=run_items,
                                     transient_items=transient_items)

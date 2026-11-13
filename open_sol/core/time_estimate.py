"""CPU run-time estimates (time_estimation_plan): the op catalogue, TimeItem / TimeEstimate, pricing by a machine profile
and the post-run time_check. Pure Python (no numpy, no cupy): GUI-thread safe. Recipes count ops, never seconds."""
import math
from dataclasses import dataclass, replace

COUNT_KINDS = ('exact', 'bounded', 'guess')
GUESS_RANGE = 3.0            # a 'guess' count spans count / GUESS_RANGE .. count x GUESS_RANGE (capped by its limit)
UNIT_RATIO_BAND = (1.0 / 3.0, 3.0)   # measured / estimated seconds per unit outside this band: a time_check warning
EXACT_COUNT_TOLERANCE = 0.01  # an 'exact' count off by more than this fraction (and one unit): a warning
MIN_CHECK_SECONDS = 0.5      # groups measuring less than this are recorded, never warned about (timer noise, compiles)
# complex SuperLU vs the real curve of the same size when the profile has no complex curve (4 real flops per complex)
COMPLEX_LU_FACTOR = 4.0
INDEX_BYTES = 4              # int32 CSR column index
# flops (an FMA = 2, as the 'arith_float<bits>' curves count) of one correctly rounded float division in a kernel: a
# reciprocal estimate refined by Newton steps, ~8 FMA-class instructions (an a priori count, never fitted)
DIVISION_FLOPS = 16
# Python calls of one time-following face value per step (builder value_at / expression, material_update apply_now,
# the feature setter, DiffusionSystem.set_face_value / update_face_value; counted from the heat_2d profile call tree)
FACE_VALUE_UPDATE_CALLS = 40


# --- op catalogue (sizes only, machine-independent) ---------------------------------------------------------------

@dataclass(frozen=True)
class Elementwise:
    """Fused elementwise / stencil kernels on the run backend: distinct bytes moved in `launches` equal calls, priced
    at the per-byte rate of all of them together (numba: the cpu backend runs a numba twin when the profile has one).
    flops: arithmetic of all launches in flop_bytes (4 float32 / 8 float64) reals; on the gpu the op takes the slower of
    its bytes and its flops ('arith_float32' / 'arith_float64': consumer GPUs run float64 at 1/32-1/64 rate)."""
    bytes_read: float
    bytes_written: float
    launches: int = 1
    numba: bool = False
    flops: float = 0.0
    flop_bytes: int = 4


@dataclass(frozen=True)
class Spmv:
    """CSR matrix-vector product on the run backend (rows, stored entries, value bytes: 4 / 8 / 16 complex)."""
    rows: float
    nnz: float
    item_bytes: int = 8


@dataclass(frozen=True)
class Reduction:
    """Dot / norm over `elements` (two arrays read) on the run backend; read_back = a host float() of it (a sync)."""
    elements: float
    item_bytes: int = 8
    read_back: bool = False


@dataclass(frozen=True)
class Sync:
    """Device synchronizations (gpu; nothing on the cpu)."""
    count: float = 1


@dataclass(frozen=True)
class Transfer:
    """Host <-> device copy of `bytes` (gpu; nothing on the cpu)."""
    bytes: float


@dataclass(frozen=True)
class HostLU:
    """Host SuperLU factorisation of a 2D 5-point (dimensions 2) or 3D 7-point (3) grid operator; scale = measured
    seconds over that curve for other operators of the same size (several unknowns per cell)."""
    unknowns: float
    dimensions: int = 2
    complex: bool = False
    scale: float = 1.0


@dataclass(frozen=True)
class HostSolve:
    """Triangular solves on a HostLU factor, `rhs` right-hand sides; scale = measured seconds over the grid curve (as
    HostLU.scale, for the factor of another operator of the same size)."""
    unknowns: float
    dimensions: int = 2
    complex: bool = False
    rhs: float = 1
    scale: float = 1.0


@dataclass(frozen=True)
class DevicePlan:
    """cuDSS analysis (host reordering + symbolic factorisation) of a complex symmetric 2D 5-point operator of
    `unknowns` ('cudss_plan_<d>d'); scale = seconds over that curve (FEM fill, a real operator: the caller's ratio)."""
    unknowns: float
    dimensions: int = 2
    scale: float = 1.0


@dataclass(frozen=True)
class DeviceFactor:
    """cuDSS refactorisation on a kept plan (host equilibration, value upload, LDL^T, probe solve) as the 5-point curve
    'cudss_factor_<d>d[_mixed]' (mixed: the complex64 factor); device_values: refreshed on the device (factorize_pencil,
    'cudss_factor_<d>d_device[_mixed]', the host curve on a profile without it); scale as DevicePlan."""
    unknowns: float
    dimensions: int = 2
    mixed: bool = False
    scale: float = 1.0
    device_values: bool = False


@dataclass(frozen=True)
class DeviceSolve:
    """cuDSS solve_host (vector each way, mixed: refined to full accuracy) of `rhs` right-hand sides on a DeviceFactor
    ('cudss_solve_<d>d[_mixed]'); scale = seconds over that curve."""
    unknowns: float
    dimensions: int = 2
    mixed: bool = False
    rhs: float = 1
    scale: float = 1.0


@dataclass(frozen=True)
class DenseFlops:
    """Dense products (block products, Arnoldi orthogonalisation) in real flops, priced at the gemm rate; host=True on
    the host whatever the backend; single: complex64 / float32 work of the run backend ('gemm_single')."""
    flops: float
    host: bool = False
    single: bool = False


@dataclass(frozen=True)
class BlockInverse:
    """Inverses of `batch` dense complex (block, block) matrices on the run backend: one batched call (curves
    'block_inverse[_single]_<b>' over the batch, interpolated in the block size) or, looped, one call per matrix
    ('matrix_inverse[_single]' over the block size); single: complex64."""
    block: float
    batch: float = 1.0
    single: bool = False
    looped: bool = False


@dataclass(frozen=True)
class HostStream:
    """Host numpy passes over `bytes` (host loops of the ARPACK vectors, host assembly)."""
    bytes: float


@dataclass(frozen=True)
class PythonCalls:
    """Per-call host overhead (a loop of many small launches is bound by it)."""
    count: float


@dataclass(frozen=True)
class SolverWork:
    """What a physics asks of one solver role (SolverType.time_estimate): the system (unknowns, stored entries, value
    bytes, grid dimensions for the host factor curves, matrix-free product / callback preconditioner ops per
    application), how many solves / factorisations (per_step: per physics time step) and the CG iteration scale."""
    physics: str
    role: str
    unknowns: float
    nnz: float = 0.0
    value_bytes: int = 8
    dimensions: int = 2
    solves: float = 1.0
    solves_kind: str = 'exact'
    factorizations: float = 1.0
    factorizations_kind: str = 'exact'
    per_step: bool = False
    longest_axis: int = 1          # cells along the longest window axis (auto iteration limit; Jacobi without condition)
    warm_start: bool = False       # solves start from a nearby solution (time steps, nonlinear updates)
    cold_solves: float = 0.0       # warm_start: solves of the series that still start cold (a fixed point's first)
    jacobi_condition: float = 0.0  # condition number of the Jacobi-scaled operator (0: unknown; HeldValues estimate)
    distinct_modes: float = 0.0    # CG finite-termination cap on the iterations of a solve (0: none)
    product_ops: tuple = ()        # matrix-free product per application (empty: a CSR Spmv of nnz)
    preconditioner_ops: tuple = ()  # callback preconditioner per application (empty: Jacobi / the pcg setting)
    refinement_ops: tuple = ()     # pcg mixed: the float64 matrix-free product of an outer refinement (empty: a CSR
                                   # Spmv of nnz); product_ops / preconditioner_ops are then the float32 inner ones
    iteration_ops: tuple = ()      # pcg fused_iteration: one whole CG iteration (and the start) in the physics' fused
                                   # launches (empty: built from product_ops / preconditioner_ops)
    setup_ops: tuple = ()          # pcg: a caller-built preconditioner's setup per factorization (empty: the pcg one)
    setup_counters: tuple = ()     # pcg with setup_ops: (count, seconds) counter names of that setup when the caller
                                   # times it itself (empty: 'factorizations', 'factorize_seconds')
    rhs: float = 1.0              # right-hand sides per solve (line_block: per shift; eigen: solves per application)
    backend: str = 'gpu'
    block_size: int = 0            # line_block: unknowns per grid line (0: sqrt of unknowns)
    batch_shifts: float = 1.0      # line_block: shifts of one solve_shifted chunk (they share its calls and launches)
    eigenpairs: int = 0            # eigen role: eigenpairs per eigen solve (solves = eigen solves)
    default_basis: int = 0         # eigen role: the physics' Arnoldi basis when the node's subspace_size is 0
    factor_scale: float = 1.0      # HostLU / 2D DevicePlan / DeviceFactor scale (operators off the 5- / 7-point curve)
    solve_scale: float = 1.0       # HostSolve / 2D DeviceSolve scale of the solves on those factors
    applications_per_basis: float = 0.0   # eigen role: operator applications per Arnoldi vector (0: the solver's guess)
    factorizations_per_step: bool | None = None   # None: as per_step (a transient system factorises once: False)
    plan_reused: bool = False      # cuDSS factorisations after the first keep its plan (previous=): one plan priced
    device_values: bool = False    # 2D cuDSS refactorisations refresh device values (factorize_pencil): the device curve
    pencil_terms: int = 2          # eigen role, plan_reused: ShiftPencil terms (K, M; a 'loaded' K, B, M: 3)
    ordering_kept: bool = False    # cuDSS plans find a kept ordering of the pattern (a later run on one mesh, an eigen
                                   # role after the driven role): sparse_direct.cudss_plan_fraction
    device_arnoldi: bool = False   # eigen role on a cuDSS factor: the device Krylov-Schur of solve_resonances (no host
                                   # ARPACK basis, no vector transfers; arpack_shift_invert.eigs_device)
    run_backend: str = 'gpu'       # the run's backend even where `backend` says 'cpu' for host solves (TimeContext
                                   # fills it): the eigen role's device Krylov-Schur around a host SuperLU

    @property
    def factor_per_step(self) -> bool:
        return self.per_step if self.factorizations_per_step is None else self.factorizations_per_step

    def counter(self, name) -> tuple:
        """summary.json path of the role runtime's work counter `name`."""
        return ('solvers', self.physics, 'roles', self.role, 'counters', name)

    @property
    def complex(self) -> bool:
        return self.value_bytes >= 16


@dataclass(frozen=True)
class HeldValues:
    """Where a DiffusionSystem physics holds its unknown, for the Jacobi-PCG condition estimate
    (TimeContext.jacobi_conditioning): features holding it on their selection, features holding every outer face,
    outer faces held unless a free_types feature selects them; nothing held = the window corner cell (anchor)."""
    feature_types: frozenset = frozenset()
    edge_types: frozenset = frozenset()       # e.g. an external-field ring: held on the window edge, any selection
    outer_faces_held: bool = False            # default condition of an outer face holds the value (2D A_z insulation)
    free_types: frozenset = frozenset()       # features that free a default-held face (symmetry: natural condition)


def elementwise(elements, arrays_read, arrays_written, item_bytes=4, launches=1, numba=False,
                flops_per_element=0.0) -> Elementwise:
    """Elementwise op of `elements` per array: arrays_read / arrays_written arrays of item_bytes each (and
    flops_per_element arithmetic in item_bytes reals)."""
    return Elementwise(float(elements) * arrays_read * item_bytes, float(elements) * arrays_written * item_bytes,
                       int(launches), numba, float(elements) * flops_per_element, int(item_bytes))


def explicit_step_estimate(context, label, dt_limit, launches, write_fraction=0.5, numba=True,
                           counter=(), numba_items=None, numba_launches=None) -> 'TimeEstimate':
    """Recipe of an explicit time-stepping engine: per step every device array of its memory estimate (context.memory:
    fields, coefficients, CPML, poles) read once and write_fraction of those bytes written, in `launches` kernels
    (+ their Python overhead); in a frequency_domain study the run arrays (harmonic DFT accumulators) are read and
    written too. numba: the cpu twins of its field kernels are numba ones; numba_items (predicate on a memory item's
    own label) / numba_launches name the arrays and launches of those kernels (None: all), the rest (poles,
    monitors, twins without numba) are numpy twins. dt_limit = its time step [s] (None: steps inside a quasi-static
    solve); counter = its own step counter (default: the runner ticks)."""
    memory = context.memory
    steady = float(memory.gpu_steady) if memory is not None else 0.0
    run = float(memory.gpu_run) if memory is not None and context.study.type == 'frequency_domain' else 0.0
    launches = int(launches)
    if not numba or numba_items is None or memory is None:
        fast_steady = steady if numba else 0.0
    else:
        fast_steady = float(sum(size for item_label, size in memory.items
                                if numba_items(item_label.rsplit(': ', 1)[-1])))
    slow_steady = steady - fast_steady
    fast_launches = launches if numba_launches is None else min(int(numba_launches), launches)
    if fast_steady <= 0.0:   # launches price only with bytes: a part without arrays hands its launches over
        fast_launches = 0
    elif slow_steady + run <= 0.0:
        fast_launches = launches
    ops = [PythonCalls(launches)]
    if fast_steady > 0.0:
        ops.append(Elementwise(fast_steady, write_fraction * fast_steady, max(fast_launches, 1), True))
    if slow_steady + run > 0.0:
        ops.append(Elementwise(slow_steady + run, write_fraction * slow_steady + run,
                               max(launches - fast_launches, 1), False))
    return TimeEstimate((TimeItem(f'{label} time step', 'step', 1.0, 'exact', tuple(ops), per_step=True,
                                  counter=counter),),
                        dt_limit=float(dt_limit) if dt_limit else math.inf)


def diffusion_time_estimate(context, window_cells, dimensions=2, cross_term=False, transient=False, time_step=None,
                            iterations=None, every_step=True, held=None, operator_changes=None,
                            entries_per_row=None, face_value_updates=0) -> 'TimeEstimate':
    """Recipe of a DiffusionSystem physics (core.fv_diffusion: heat, electric_currents, electrostatics,
    magnetostatics(_3d)): its 'linear' role on the window system (5 / 9 / 7 entries per row) plus the assembly and the
    right-hand side. Transient (time_dependent, time_step): one solve per step, one assembly; quasi-static: the
    study's static solves (fixed point: the first cold, the rest warm, each re-assembled), in a time_dependent study one
    per step of its stride (every_step False: only when something changed, i.e. when coupled). iterations = (guess,
    limit) of a nonlinear loop (Picard / Newton / Anderson) multiplying the solves; held (HeldValues) gives the
    Jacobi-PCG condition estimate; operator_changes False: coupled quasi-static steps and fixed-point rounds keep the
    build's assembly (None: re-assembled whenever coupled). Fixed-point re-solves are priced warm-started (the first
    cold), transient steps cold (their 'initial_residual' stop asks for the full reduction).
    entries_per_row: an FEM operator (window_cells = its unknowns): its stored entries per row, the host / cuDSS factor
    and solves priced on the stencil curves at the unknowns x entries_per_row over the stencil's entries (equal
    unknowns, never inflated ones: the 3D LU curve grows ~n^2). face_value_updates: feature values of t re-set per
    solve on outer faces (DiffusionSystem.update_face_value, one face row each)."""
    cells = float(window_cells)
    stencil_entries = 7 if dimensions == 3 else 9 if cross_term else 5
    entries = stencil_entries if entries_per_row is None else float(entries_per_row)
    fill_scale = entries / stencil_entries
    nnz = entries * cells
    linear = context.solvers.get('linear')
    precision = linear.values.get('precision', 'double') if linear is not None else 'double'
    # a cuDSS back end: DiffusionDirectSolver factorises the device CSR (factorize_device_csr): the device route
    device_values = linear is not None and linear.values.get('back_end') in ('cudss', 'cudss_mixed')
    value_bytes = 4 if precision == 'single' else 8
    stepping = context.time_dependent and (transient or every_step or context.coupled)
    if stepping:
        solves, kind, limit = 1.0, 'exact', None
        assemblies, assemblies_per_step = 1.0, not transient and (context.coupled if operator_changes is None
                                                                  else operator_changes)
    else:
        solves, kind, limit = context.static_solves()
        # fixed-point rounds re-assemble only when the operator follows the state (else the build's factor serves)
        assemblies, assemblies_per_step = (1.0 if operator_changes is False else solves), False
    if iterations is not None:
        guess, iteration_limit = iterations
        solves, kind = solves * guess, 'guess'
        limit = None if iteration_limit is None else (limit or 1.0) * iteration_limit
        assemblies *= guess
    # the first assembly through build_laplacian; re-assemblies on the kept plain pattern (no c_xy term; FEM operators
    # aside) through core.fv_diffusion_fused: two kernels plus the copies, masks and the anchor pattern test
    fused_share = 0.0
    if entries_per_row is None and not cross_term and assemblies > 0.0:
        fused_share = 1.0 if assemblies_per_step else max(assemblies - 1.0, 0.0) / assemblies
    assembly = (Elementwise((6.0 - 3.0 * fused_share) * nnz * value_bytes, 2.0 * nnz * value_bytes,
                            launches=int(round(30 - fused_share * (30 - 8)))),
                PythonCalls(60 - fused_share * (60 - 30)))
    rhs = (elementwise(cells, 6, 2, value_bytes, launches=6), PythonCalls(15))
    # DiffusionSystem.solve's w = u - u_fix and u = w + u_fix around the linear solve, then the physics' float32 field
    # from the unknown (heat: T from the rise); each probe stores one device scalar (index view + store)
    update = (Elementwise(5.0 * cells * value_bytes, cells * (2.0 * value_bytes + 4.0), launches=3), PythonCalls(10))
    probes = sum(1 for feature in getattr(context.physics_spec, 'features', ())
                 if feature.enabled and feature.type.endswith('_probe'))
    # measured by the linear role: every assembly (and diagonal / capacity refresh) factorises it once, every
    # right-hand side precedes one solve
    role_counter = ('solvers', context.name, 'roles', 'linear', 'counters')
    items = (TimeItem('assembly', 'assembly', assemblies, kind, assembly, limit=limit, per_step=assemblies_per_step,
                      counter=role_counter + ('factorizations',)),
             TimeItem('right-hand side', 'solve', solves, kind, rhs, limit=limit, per_step=stepping,
                      counter=role_counter + ('solves',)),
             TimeItem('solution update', 'solve', solves, kind, update, limit=limit, per_step=stepping,
                      counter=role_counter + ('solves',)))
    if probes:
        items += (TimeItem(f'probe samples ({probes} probes)', 'solve', solves, kind,
                           (Elementwise(8.0 * probes, 4.0 * probes, launches=2 * probes), PythonCalls(4 * probes)),
                           limit=limit, per_step=stepping, counter=role_counter + ('solves',)),)
    if face_value_updates and stepping:
        # per value: the expression, the feature setter chain and the face rhs change (rhs = g (value - reference),
        # its difference, two scatters into rhs_drive / rhs_offset) on one face row
        face_cells = float(context.longest_axis())
        updates = int(face_value_updates)
        items += (TimeItem(f'face value updates ({updates} values of t)', 'solve', solves, kind,
                           (elementwise(face_cells, 5 * updates, 3 * updates, value_bytes, launches=6 * updates),
                            PythonCalls(FACE_VALUE_UPDATE_CALLS * updates)),
                           limit=limit, per_step=stepping, counter=role_counter + ('solves',)),)
    condition, modes = context.jacobi_conditioning(held) if held is not None else (0.0, 0.0)
    solver = context.solver_work('linear', unknowns=cells, nnz=nnz, value_bytes=value_bytes, dimensions=dimensions,
                                 solves=solves, solves_kind=kind, factorizations=assemblies, factorizations_kind=kind,
                                 per_step=stepping, factorizations_per_step=assemblies_per_step,
                                 # a quasi-static coupled step sees an O(1) source change since its last solve: cold;
                                 # a transient step stops at tol x its own initial residual ('initial_residual'
                                 # reference): the full reduction of a cold solve, however close the warm start
                                 longest_axis=context.longest_axis(), warm_start=not transient and solves > 1.0,
                                 cold_solves=0.0 if transient else 1.0, jacobi_condition=condition,
                                 distinct_modes=modes, device_values=device_values, factor_scale=fill_scale,
                                 solve_scale=fill_scale)
    return combine((TimeEstimate(items, dt_limit=float(time_step) if transient and time_step else math.inf),
                    solver))


def _op_pairs(ops):
    """((multiplicity, op), ...) of an ops tuple whose entries are an op or a (multiplicity, op) pair."""
    return tuple(entry if isinstance(entry, tuple) else (1.0, entry) for entry in ops)


def price(op, profile, backend) -> float:
    """Seconds of one op on the machine profile (time_profile.MachineProfile) of the run backend 'gpu' / 'cpu';
    KeyError when the profile lacks the curve the op needs."""
    on_gpu = backend == 'gpu'
    if isinstance(op, Elementwise):
        total = op.bytes_read + op.bytes_written
        if total <= 0:
            return 0.0
        launches = max(int(op.launches), 1)
        curve = 'fused_numba' if (not on_gpu and op.numba and profile.has('fused_numba')) else 'fused'
        # the curve repeats one call on arrays of its size: launches streaming distinct arrays evict each other once
        # their sum leaves the cache, so each pays the rate at the total (cpu L2 / L3 cliffs), at least its own floor
        seconds = max(launches * profile.seconds(curve, total / launches), profile.seconds(curve, total))
        # gpu: arithmetic-bound kernels (float64 on consumer cards); a profile without the curve prices the bytes
        arithmetic = f'arith_float{8 * int(op.flop_bytes)}'
        if on_gpu and op.flops > 0.0 and profile.has(arithmetic):
            seconds = max(seconds, launches * profile.seconds(arithmetic, op.flops / launches))
        return seconds
    if isinstance(op, Spmv):
        size = op.nnz * (op.item_bytes + INDEX_BYTES) + op.rows * (2 * op.item_bytes + INDEX_BYTES)
        return profile.seconds('spmv', size) if size > 0 else 0.0
    if isinstance(op, Reduction):
        seconds = profile.seconds('dot', 2.0 * op.elements * op.item_bytes) if op.elements > 0 else 0.0
        return seconds + (profile.scalar('sync') if op.read_back and on_gpu else 0.0)
    if isinstance(op, Sync):
        return op.count * profile.scalar('sync') if on_gpu else 0.0
    if isinstance(op, Transfer):
        return profile.seconds('transfer', op.bytes) if on_gpu and op.bytes > 0 else 0.0
    if isinstance(op, (HostLU, HostSolve)):
        kind = 'host_lu' if isinstance(op, HostLU) else 'host_solve'
        curve = f'{kind}_{int(op.dimensions)}d'
        factor = 1.0
        if op.complex:
            if profile.has(f'{curve}_complex'):
                curve = f'{curve}_complex'
            else:
                factor = COMPLEX_LU_FACTOR
        seconds = factor * profile.seconds(curve, max(op.unknowns, 1.0))
        return seconds * op.scale * (op.rhs if isinstance(op, HostSolve) else 1.0)
    if isinstance(op, (DevicePlan, DeviceFactor, DeviceSolve)):
        kind = 'plan' if isinstance(op, DevicePlan) else 'factor' if isinstance(op, DeviceFactor) else 'solve'
        mixed = '_mixed' if getattr(op, 'mixed', False) else ''
        curve = f'cudss_{kind}_{int(op.dimensions)}d' + mixed
        if getattr(op, 'device_values', False) and profile.has(f'cudss_{kind}_{int(op.dimensions)}d_device{mixed}'):
            curve = f'cudss_{kind}_{int(op.dimensions)}d_device{mixed}'
        seconds = profile.seconds(curve, max(op.unknowns, 1.0)) * op.scale
        return seconds * (op.rhs if isinstance(op, DeviceSolve) else 1.0)
    if isinstance(op, DenseFlops):
        if op.single and not op.host:
            curve = 'gemm_single'
        else:
            curve = 'host_gemm' if op.host or not on_gpu else 'gemm'
        return profile.seconds(curve, op.flops) if op.flops > 0 else 0.0
    if isinstance(op, BlockInverse):
        suffix = '_single' if op.single else ''
        if op.looped:
            return op.batch * profile.seconds(f'matrix_inverse{suffix}', op.block)
        return _block_inverse_seconds(profile, f'block_inverse{suffix}_', op.block, op.batch)
    if isinstance(op, HostStream):
        return profile.seconds('host_stream', op.bytes) if op.bytes > 0 else 0.0
    if isinstance(op, PythonCalls):
        return op.count * profile.scalar('python_call')
    raise TypeError(f'unknown time op {type(op).__name__}')


def _block_inverse_seconds(profile, prefix, block, batch) -> float:
    """Seconds of one batched inverse of `batch` (block, block) matrices: the curves '<prefix><b>' of the two measured
    block sizes nearest `block`, each at that batch, log-log interpolated in the block size (below the smallest: its
    seconds; above the largest: the last pair's slope); KeyError without such curves."""
    sizes = sorted(int(name[len(prefix):]) for name in profile.curves if name.startswith(prefix))
    if not sizes:
        raise KeyError(f'{prefix}<block>')
    if block <= sizes[0] or len(sizes) == 1:
        return profile.seconds(f'{prefix}{sizes[0]}', batch)
    low, high = next(((low, high) for low, high in zip(sizes, sizes[1:]) if block <= high), tuple(sizes[-2:]))
    seconds_low, seconds_high = (profile.seconds(f'{prefix}{size}', batch) for size in (low, high))
    slope = math.log(seconds_high / seconds_low) / math.log(high / low)
    return seconds_low * (block / low) ** slope


def price_ops(ops, profile, backend) -> float:
    """Seconds of an ops tuple (entries an op or (multiplicity, op))."""
    return sum(multiplicity * price(op, profile, backend) for multiplicity, op in _op_pairs(ops))


# --- estimates ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeItem:
    """count units of work (unit 'step', 'iteration', 'solve', 'factorisation', 'setup', ...), each costing `ops`;
    count_kind 'exact' (fixed by the model), 'bounded' (an upper bound) or 'guess' (heuristic, below `limit`).
    per_step: count is per physics time step (estimate_time multiplies it by the steps). counter / seconds_counter:
    summary.json paths of the measured count (x counter_scale) and of the measured seconds of this item's group (()
    = the part of study_seconds no counter measures)."""
    label: str
    unit: str
    count: float
    count_kind: str = 'exact'
    ops: tuple = ()
    limit: float | None = None
    per_step: bool = False
    counter: tuple = ()
    counter_scale: float = 1.0
    seconds_counter: tuple = ()
    seconds_per_unit: float | None = None

    def __post_init__(self):
        if self.count_kind not in COUNT_KINDS:
            raise ValueError(f"time item '{self.label}': count_kind must be one of {', '.join(COUNT_KINDS)}, "
                             f'got {self.count_kind!r}')

    def counts(self) -> tuple:
        """(low, guess, high) counts: a guess spans GUESS_RANGE either way (high capped by limit), a bound is reached
        at most."""
        count = float(self.count)
        if self.count_kind == 'exact':
            return count, count, count
        high = count if self.count_kind == 'bounded' else count * GUESS_RANGE
        if self.limit is not None:
            high = min(high, float(self.limit))
        return min(count / GUESS_RANGE, high), min(count, high), high   # a limit below count / 3 caps low too


@dataclass(frozen=True)
class TimeEstimate:
    """Estimated run time: items (TimeItem), not_estimated (labels of work without a recipe: never counted as 0 s),
    dt_limit (largest time step of a time-stepped physics part, inf otherwise) and the profile it was priced with
    ({key, date} or None while unpriced)."""
    items: tuple = ()
    not_estimated: tuple = ()
    dt_limit: float = math.inf
    profile: dict | None = None

    def labelled(self, prefix):
        """Copy with every item / not-estimated label prefixed by '<prefix>: '."""
        return replace(self, items=tuple(replace(item, label=f'{prefix}: {item.label}') for item in self.items),
                       not_estimated=tuple(f'{prefix}: {label}' for label in self.not_estimated))

    def with_steps(self, steps, steps_kind, every=1, steps_limit=None, measured_by_ticks=True, ticks=None):
        """Copy with per-step items turned into totals over `steps` physics steps (count kind combined: exact only
        when both are exact); measured by summary 'ticks' x steps / ticks (ticks None: / every; measured_by_ticks:
        ticks are the runner's time steps) unless an item has its own counter."""
        items = []
        for item in self.items:
            if not item.per_step:
                items.append(item)
                continue
            kind = item.count_kind if steps_kind == 'exact' else ('guess' if 'guess' in (item.count_kind, steps_kind)
                                                                  else 'bounded')
            limit = None if steps_limit is None and item.limit is None else (
                (steps_limit if steps_limit is not None else steps) * (item.limit or item.count))
            counter, scale = item.counter, item.counter_scale
            if not counter and measured_by_ticks:
                counter, scale = ('ticks',), (float(item.count) * steps / ticks if ticks else
                                              float(item.count) / max(int(every), 1))
            items.append(replace(item, count=float(item.count) * steps, count_kind=kind, limit=limit, per_step=False,
                                 counter=counter, counter_scale=scale))
        return replace(self, items=tuple(items))

    def priced(self, profile, backend):
        """Copy with seconds_per_unit of every item on the profile; items whose ops the profile cannot price move to
        not_estimated."""
        items, missing = [], list(self.not_estimated)
        for item in self.items:
            try:
                items.append(replace(item, seconds_per_unit=price_ops(item.ops, profile, backend)))
            except KeyError as exc:
                missing.append(f'{item.label} (profile has no {exc.args[0]!r} curve)')
        return replace(self, items=tuple(items), not_estimated=tuple(missing), profile=profile.record())

    def item_seconds(self, item, which=1) -> float:
        """Seconds of one priced item at its low (0) / guess (1) / high (2) count."""
        return item.counts()[which] * (item.seconds_per_unit or 0.0)

    @property
    def seconds_low(self) -> float:
        return sum(self.item_seconds(item, 0) for item in self.items)

    @property
    def seconds_guess(self) -> float:
        return sum(self.item_seconds(item, 1) for item in self.items)

    @property
    def seconds_high(self) -> float:
        return sum(self.item_seconds(item, 2) for item in self.items)

    @property
    def has_guess(self) -> bool:
        return any(item.count_kind != 'exact' for item in self.items)

    def largest_items(self, count=3):
        """The `count` items of the largest guess seconds, largest first."""
        return sorted(self.items, key=lambda item: -self.item_seconds(item))[:count]

    def outdated_text(self) -> str:
        """' (outdated gpu profile)' when priced on an out-of-date stored profile, else ''."""
        if not (self.profile or {}).get('outdated'):
            return ''
        return f" (outdated {self.profile['backend']} profile)"

    def text(self, count=1) -> str:
        """'estimated run time 40 s (guess; 15 s - 2 min), largest: <item> 12 ms x 3400 steps (exact)' (+ not
        estimated work)."""
        if self.profile is None:
            return 'run time not estimated (no machine profile)'
        guess = format_seconds(self.seconds_guess)
        span = (f' (guess; {format_seconds(self.seconds_low)} - {format_seconds(self.seconds_high)})'
                if self.has_guess else '')
        largest = '; '.join(item_text(item) for item in self.largest_items(count))
        text = (f'estimated run time {guess}{span}{self.outdated_text()}'
                + (f', largest: {largest}' if largest else ''))
        if self.not_estimated:
            text += f"; not estimated: {', '.join(self.not_estimated)}"
        return text

    def as_dict(self) -> dict:
        """JSON-ready record (check.json / summary.json time_estimate)."""
        return {'profile': self.profile, 'seconds_low': self.seconds_low, 'seconds_guess': self.seconds_guess,
                'seconds_high': self.seconds_high, 'not_estimated': list(self.not_estimated),
                'items': [{'label': item.label, 'unit': item.unit, 'count': item.count, 'count_kind': item.count_kind,
                           'limit': item.limit, 'seconds_per_unit': item.seconds_per_unit,
                           'seconds': self.item_seconds(item)} for item in self.largest_items(len(self.items))]}


def combine(estimates) -> TimeEstimate:
    """Sum of parts: items and not-estimated labels concatenated, the smallest dt_limit, the first profile."""
    estimates = list(estimates)
    return TimeEstimate(tuple(item for estimate in estimates for item in estimate.items),
                        tuple(label for estimate in estimates for label in estimate.not_estimated),
                        min((estimate.dt_limit for estimate in estimates), default=math.inf),
                        next((estimate.profile for estimate in estimates if estimate.profile is not None), None))


def format_seconds(seconds) -> str:
    """'12 ms', '40 s', '2.5 min', '1.2 h'."""
    seconds = float(seconds)
    if not math.isfinite(seconds):
        return 'inf'
    if seconds < 1e-3:
        return f'{seconds * 1e6:.3g} us'
    if seconds < 1.0:
        return f'{seconds * 1e3:.3g} ms'
    if seconds < 120.0:
        return f'{seconds:.3g} s'
    if seconds < 7200.0:
        return f'{seconds / 60.0:.3g} min'
    return f'{seconds / 3600.0:.3g} h'


def item_text(item) -> str:
    """'<label> 12 ms x 3400 steps (exact)'."""
    return (f'{item.label} {format_seconds(item.seconds_per_unit or 0.0)} x {item.count:.4g} {item.unit}s '
            f'({item.count_kind})')


# --- post-run comparison ------------------------------------------------------------------------------------------

def _summary_value(summary, path):
    """Number at a summary path (keys), None when any key is missing or the value is not a number."""
    value = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def time_check(estimate, summary) -> dict:
    """summary.json time_check of a priced estimate against the run summary (study_seconds, ticks, solvers counters):
    per item predicted vs measured count, per seconds group (items sharing a seconds counter; the rest of
    study_seconds is one group) estimated seconds at the measured counts vs measured seconds."""
    study_seconds = float(summary.get('study_seconds') or 0.0)
    groups, records, record_keys = {}, [], []
    for item in estimate.items:
        measured = _summary_value(summary, item.counter) if item.counter else None
        measured_count = None if measured is None else measured * item.counter_scale
        count = measured_count if measured_count is not None else float(item.count)
        key = tuple(item.seconds_counter)
        group = groups.setdefault(key, {'estimated_seconds': 0.0, 'labels': []})
        group['estimated_seconds'] += count * (item.seconds_per_unit or 0.0)
        group['labels'].append(item.label)
        record_keys.append(key)
        records.append({'label': item.label, 'unit': item.unit, 'count_kind': item.count_kind,
                        'predicted_count': float(item.count), 'measured_count': measured_count,
                        'count_ratio': (measured_count / item.count if measured_count is not None and item.count > 0
                                        else None),
                        'predicted_seconds_per_unit': item.seconds_per_unit, 'group': ' / '.join(key) or 'study'})
    counted_seconds, at_build_seconds = 0.0, 0.0
    start = {'solvers': summary.get('solvers_at_study_start') or {}}
    for key, group in groups.items():
        if key:
            group['measured_seconds'] = _summary_value(summary, key)
            counted_seconds += group['measured_seconds'] or 0.0
            at_build_seconds += _summary_value(start, key) or 0.0   # cumulative timer before the study
    study_seconds += at_build_seconds   # the estimate prices that build-time work as study work
    rest = groups.setdefault((), {'estimated_seconds': 0.0, 'labels': []})
    rest['measured_seconds'] = max(study_seconds - counted_seconds, 0.0)
    for group in groups.values():
        measured, estimated = group.get('measured_seconds'), group['estimated_seconds']
        group['unit_ratio'] = measured / estimated if measured is not None and estimated > 0 else None
    for record, key in zip(records, record_keys):
        group = groups[key]
        record['unit_ratio'] = group['unit_ratio']
        record['measured_seconds_per_unit'] = (record['predicted_seconds_per_unit'] * group['unit_ratio']
                                               if group['unit_ratio'] is not None
                                               and record['predicted_seconds_per_unit'] is not None else None)
    group_records = [{'group': ' / '.join(key) or 'study', 'items': group['labels'],
                      'estimated_seconds': group['estimated_seconds'], 'measured_seconds': group.get('measured_seconds'),
                      'unit_ratio': group['unit_ratio']} for key, group in groups.items()]
    predicted = estimate.seconds_guess
    return {'profile': estimate.profile, 'study_seconds': study_seconds, 'seconds_at_build': at_build_seconds,
            'predicted_seconds': predicted, 'total_ratio': study_seconds / predicted if predicted > 0 else None,
            'unaccounted_seconds': rest['measured_seconds'] - rest['estimated_seconds'],
            'not_estimated': list(estimate.not_estimated), 'groups': group_records, 'items': records}


def merge_time_checks(labelled_records):
    """One time_check of several runs [(run label, record or None)] (sweep values, refinement levels, parity sectors):
    seconds and seconds groups summed, items per run labelled '<run label>: <item>'; None without a record."""
    pairs = [(label, record) for label, record in labelled_records if record]
    if not pairs:
        return None
    groups = {}
    for _, record in pairs:
        for group in record['groups']:
            merged = groups.setdefault(group['group'], {'group': group['group'], 'items': [], 'estimated_seconds': 0.0,
                                                        'measured_seconds': None})
            merged['items'] += [label for label in group['items'] if label not in merged['items']]
            merged['estimated_seconds'] += group['estimated_seconds']
            if group['measured_seconds'] is not None:
                merged['measured_seconds'] = (merged['measured_seconds'] or 0.0) + group['measured_seconds']
    for group in groups.values():
        measured, estimated = group['measured_seconds'], group['estimated_seconds']
        group['unit_ratio'] = measured / estimated if measured is not None and estimated > 0 else None
    study_seconds = sum(record['study_seconds'] for _, record in pairs)
    predicted = sum(record['predicted_seconds'] for _, record in pairs)
    return {'profile': pairs[0][1]['profile'], 'study_seconds': study_seconds,
            'seconds_at_build': sum(record.get('seconds_at_build') or 0.0 for _, record in pairs),
            'predicted_seconds': predicted,
            'total_ratio': study_seconds / predicted if predicted > 0 else None,
            'unaccounted_seconds': sum(record['unaccounted_seconds'] for _, record in pairs),
            'not_estimated': list(dict.fromkeys(label for _, record in pairs for label in record['not_estimated'])),
            'groups': list(groups.values()),
            'items': [dict(item, label=f"{label}: {item['label']}") for label, record in pairs
                      for item in record['items']]}


def time_check_warnings(record) -> list:
    """Warning texts of a time_check record: an 'exact' count that differs, a seconds group whose measured / estimated
    ratio leaves UNIT_RATIO_BAND (speed-ups as well as slow-downs). 'guess' counts never warn."""
    messages = []
    for item in record['items']:
        measured, predicted = item['measured_count'], item['predicted_count']
        if (item['count_kind'] == 'exact' and measured is not None
                and abs(measured - predicted) > max(EXACT_COUNT_TOLERANCE * predicted, 1.0)):
            messages.append(f"run time check: {item['label']} ran {measured:.6g} {item['unit']}s, the model fixes "
                            f"{predicted:.6g}: the run or the time recipe does other work than expected")
    low, high = UNIT_RATIO_BAND
    for group in record['groups']:
        ratio, measured = group['unit_ratio'], group['measured_seconds'] or 0.0
        if ratio is None or measured < MIN_CHECK_SECONDS or low <= ratio <= high:
            continue
        what = ('the rest of the study (no own timer)' if group['group'] == 'study'
                else ', '.join(group['items']))
        hint = ('faster than estimated: check that the work was done (zero right-hand side, an early stop, a skipped '
                'solve)' if ratio < low else 'slower than estimated: unfused path, extra syncs, host transfers, the '
                'other backend, or work without a recipe')
        messages.append(f'run time check: {what} took {ratio:.2g}x the estimated time '
                        f"({format_seconds(measured)} vs {format_seconds(group['estimated_seconds'])}); {hint}")
    return messages

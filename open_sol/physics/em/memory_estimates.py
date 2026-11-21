"""CPU memory estimators (E8) of the EM physics types and their allocating features, counted from the engines'
allocations: fields, coefficients, CPML psi slabs, poles on their boxes, loss accumulators, DFT / line spectra, point
traces, harmonic FDTD accumulators, the FDFD system and the host (eigen) factorisations. GUI-thread safe."""
import math
from functools import lru_cache

from ...core.memory_estimate import (BOOL_BYTES, COMPLEX64_BYTES, COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES,
                                     FLOAT32_BYTES, FLOAT64_BYTES, INT32_BYTES, MemoryEstimate, host_lu_bytes,
                                     phase_before_items, solver_job_items, window_cell_count)
from ...model.definition_helpers import (eigen_factorization, eigen_subspace_size, host_direct_back_end,
                                         linear_solver_cudss, resolved_back_end, resolved_stepper_precision)
from ...model.definition_resonance import point_trace_items
from ...model.schema import EIGENFREQUENCY_STUDY, resolve_values, solver_backend
from ...solvers.arpack_shift_invert import (CUDSS_FACTORIZATIONS, MIXED_EIGEN_MIN_UNKNOWNS, arnoldi_basis_bytes,
                                            arnoldi_basis_size, device_arnoldi_fits)
from ...solvers.cudss_factor import (CUDSS_SOLVE_PHASE, FactorFill, factor_device_bytes, index_map_bytes,
                                     operator_copy_bytes, refresh_peak_bytes, solve_buffer_bytes, solve_peak_bytes)
from ...solvers.device_matrices import device_matrix_bytes
from ...solvers.host_direct import (HOST_DIRECT_FACTORIZATIONS, factor_label, host_direct_precision, solver_host_bytes,
                                    superlu_bytes_scale)
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, HOST_FACTOR_BACK_ENDS
from .definition_bloch import bloch_active_axes
from .definition_common import (FREQUENCY_DOMAIN_STUDY, _q_accum_referenced, _study_wavelengths, eigen_solved,
                                fdtd_parked_material_arrays)
from .solvers.fdtd_2d_base import PARKED_BLOCK_CELLS
from .eigenfrequency import FILTER_DEVICE_BLOCK_COLUMNS
from .materials import GYRATION_Z
from .placement import PERMITTIVITY_ARRAYS

# FDFD kept Helmholtz term plan (operators._HelmholtzPlan): per cell 9 terms (4 per x / y product + the cell term) into
# 5 operator entries; int32: 3 per product term, the first term of each entry, (entry, term) of the other terms (the
# terms: operators._ParkedInt32, pinned host on the gpu, uploaded per assembly) and the pattern (indices + indptr, on
# the run backend)
_FDFD_PLAN_TERMS, _FDFD_OPERATOR_ROW, _FDFD_PRODUCT_TERMS = 9, 5, 8
_INT64_BYTES = 8
_FDFD_PRODUCT_TERM_BYTES = INT32_BYTES * 3 * _FDFD_PRODUCT_TERMS
_FDFD_PLAN_TERM_BYTES = (_FDFD_PRODUCT_TERM_BYTES
                         + INT32_BYTES * (_FDFD_OPERATOR_ROW + 2 * (_FDFD_PLAN_TERMS - _FDFD_OPERATOR_ROW)))
_FDFD_PLAN_PATTERN_BYTES = INT32_BYTES * (_FDFD_OPERATOR_ROW + 1)
# KeptYeeOperators: the int32 2D entry -> 1D entry map of each of the 4 Yee operators (2 entries per row) and the
# indices + indptr of dx_e / dy_e, parked beside the plan's terms and uploaded by every `at` (the systems keep the e
# differences matrix-free: operators.EDifference)
_YEE_ENTRY_MAP_BYTES = 4 * 2 * INT32_BYTES
_YEE_E_PATTERN_BYTES = 2 * (2 + 1) * INT32_BYTES
# the dx_e / dy_e CSR of one assembly: values + indices + indptr (2 entries per row)
_YEE_E_CSR_BYTES = 2 * (2 * COMPLEX128_CSR_ENTRY_BYTES + INT32_BYTES)
# assembly peak per cell: the plan build's sort (int32 rows / columns, int64 order + argsort result, the gathered rows
# per term; first assembly) over a later omega's term buffer (complex128 per term + one gathered 4-per-row factor);
# gpu: the build's product terms beside the sort, the uploaded plan terms beside the term buffer
_FDFD_SORT_BYTES = _FDFD_PLAN_TERMS * (3 * INT32_BYTES + 2 * _INT64_BYTES)
_FDFD_TERM_BUFFER_BYTES = (_FDFD_PLAN_TERMS + 4) * COMPLEX128_BYTES
_FDFD_ASSEMBLY_BYTES = {'cpu': max(_FDFD_SORT_BYTES, _FDFD_TERM_BUFFER_BYTES),
                        'gpu': max(_FDFD_SORT_BYTES + _FDFD_PRODUCT_TERM_BYTES,
                                   _FDFD_TERM_BUFFER_BYTES + _FDFD_PLAN_TERM_BYTES)}
# host transfer per cell of one gpu assembly (definition_fdfd's time recipe): the plan terms, Yee entry maps and e
# patterns up, the host system arrays (cell eps and mu; hz eps_xx / eps_yy and the two E-face eps) down
FDFD_ASSEMBLY_UPLOAD_BYTES = _FDFD_PLAN_TERM_BYTES + _YEE_ENTRY_MAP_BYTES + _YEE_E_PATTERN_BYTES
FDFD_HOST_SYSTEM_ARRAYS = {'ez': 2, 'hz': 5}
FDFD_LOSS_ARRAYS = {'ez': 1, 'hz': 2}   # float32 Im(cell eps) on the device (HarmonicSystem.loss_eps_imag, q_abs)
# Run phases (MemoryEstimate.transient_items) of the frequency-domain engines; never alive at the same time
ASSEMBLY_PHASE = 'assembly'                 # operator assembly temporaries (the system itself is a run item)
SOLVE_PHASE = 'solve'                       # one solve's material temporaries, an own (not kept) factor's copies
EIGEN_PHASE = 'eigen solve'                 # solve_resonances / a mode solve: own factor copies, Krylov basis
# solve_resonances between eigs passes: the resonance filter (one operator upload, column blocks) beside the own
# factor's copies; the Krylov basis waits on the host (eigenfrequency._parked_decomposition)
FILTER_PHASE = 'resonance filter'
REFACTOR_PHASE = 'cuDSS refactorisation'    # refactorize on the planned pattern (a later frequency / order / iteration)
# fdtd_memory_estimate item names (the label before ' (') the cpu numba Yee twins stream (definition_fdtd prices only
# these as numba): fields, update coefficients and the Bloch imaginary set (its bulk is fields)
FDTD_FIELDS_ITEM, FDTD_COEFFICIENTS_ITEM, FDTD_BLOCH_ITEM = 'fields', 'update coefficients', 'Bloch imaginary field set'
FDTD_NUMBA_STREAMED_ITEMS = (FDTD_FIELDS_ITEM, FDTD_COEFFICIENTS_ITEM, FDTD_BLOCH_ITEM)
ADJOINT_PHASE = 'adjoint solve'             # one eps_sensitivity: transposed-solve temporaries and the gradients
# solver role whose job holds a resonance search's own factor, factor copies and Krylov basis (solver_job_items: the
# solve flow books them at that job); the filter, eigenvectors and operator copies stay the physics'
EIGEN_ROLE = 'eigen'
# _ResonanceFilter.measures: device column blocks alive at once, one operator at a time (columns, its product, the
# conjugate, one product temporary and the real energies; + D v with a row scale)
_FILTER_BLOCKS = 5
# its window / region rows and diagonal weights (float32), diagonal mass and row scale (complex128): an upper figure in
# bytes per unknown, kept on the device for the whole search
_FILTER_VECTOR_BYTES = 3 * FLOAT32_BYTES + 2 * COMPLEX128_BYTES
_DFT_SPECTRA = {'ez': 1, 'hz': 3}   # complex64 (n_wavelengths, mx, my) spectra of a DFTMonitor (monitors.py)
# FieldProbe users (core.monitors: growable float32 device trace + float64 host times); counted by the FDTD physics
# estimator because a FeatureType estimator does not see the study duration
_POINT_TRACE_FEATURES = ('field_probe', 'resonance_monitor')
_EIGEN_HOST_VECTORS = 6   # solve_resonances: mass, b_diagonal, start vector (+ rng parts), energy weights / window
# coupling types that write the em_fdfd_hz eps_xy_update slot (coupling/definitions_acoustics, definitions_pockels)
# -> material properties without which they inject nothing (None: always data)
_OFF_DIAGONAL_WRITERS = {'acoustics_em_acousto_optic': None,
                         'electrostatics_em_pockels': ('pockels_r61', 'pockels_r62')}
_OFF_DIAGONAL_EXTRA_FILL = 0.5   # SuperLU L + U of the 9-point eps_xy operator over the 5-point heuristic (rough;
# booked at SuperLU size for a MUMPS / PARDISO factor too)


def _bor_entries_per_row(largest_order) -> float:
    """Full-pattern entries per interior row of the tensor BOR operator C^T W C - k0^2 M (diagonal eps): m = 0 keeps u
    (5: itself, 4 neighbours) apart from E_r / E_z (7: 3 own, 4 of the other via a_phi); |m| >= 1 (the m / r curl
    entries) couples all three, every row 9. Wall rows have fewer: a slight upper bound."""
    return 9.0 if largest_order else 19.0 / 3.0


@lru_cache(maxsize=16)
def _bor_factor_entries(nr, nz, coupled) -> int:
    """Factor entries of one triangle of the tensor BOR operator on an (nr, nz) grid (the full unknown set): the
    fem.dissection count on its cliques, one per curl point (body_of_revolution a_phi: 2 E_r + 2 E_z, a_r: 2 u + E_z,
    a_z: 2 u + E_r; C^T W C couples exactly the unknowns one point differences); coupled False (m = 0): the u block
    and the E_r / E_z block apart, added. Cached per grid (seconds above ~1e6 cells). The n log2 n growth law it
    replaced has no per-row part and read 27 % low (em_bor_cpu_gpu ports, 19,200 unknowns: 13.0 MB of 25.6 MB
    permanent were factor entries); the conical, modes and 2D FDFD cuDSS estimators still use that law."""
    import numpy as np
    from ...fem.dissection import element_dissection_pairs
    n_radial, n_cells = (nr + 1) * nz, nr * nz
    radial_edge, radial_row = np.divmod(np.arange(n_radial), nz)          # E_r on (edge k, centre j)
    cell_column, cell_row = np.divmod(np.arange(n_cells), nz)             # u on (centre i, centre j)
    axial_column, axial_edge = np.divmod(np.arange(nr * (nz + 1)), nz + 1)   # E_z on (centre i, edge l)
    points = np.concatenate((np.stack((radial_edge, radial_row + 0.5), axis=1),
                             np.stack((cell_column + 0.5, cell_row + 0.5), axis=1),
                             np.stack((axial_column + 0.5, axial_edge), axis=1))).astype(np.float64)

    def e_r(edge, row):
        return edge * nz + row

    def u(column, row):
        return n_radial + column * nz + row

    def e_z(column, edge):
        return n_radial + n_cells + column * (nz + 1) + edge

    # a point on the grid's rim misses a neighbour: the index is clipped onto the one it has (a repeated member)
    edge_k, edge_l = (index.ravel() for index in np.meshgrid(np.arange(nr + 1), np.arange(nz + 1), indexing='ij'))
    a_phi = np.stack((e_r(edge_k, np.clip(edge_l - 1, 0, nz - 1)), e_r(edge_k, np.clip(edge_l, 0, nz - 1)),
                      e_z(np.clip(edge_k - 1, 0, nr - 1), edge_l), e_z(np.clip(edge_k, 0, nr - 1), edge_l)), axis=1)
    centre_i, edge_l = (index.ravel() for index in np.meshgrid(np.arange(nr), np.arange(nz + 1), indexing='ij'))
    a_r = np.stack((u(centre_i, np.clip(edge_l - 1, 0, nz - 1)), u(centre_i, np.clip(edge_l, 0, nz - 1)),
                    e_z(centre_i, edge_l)), axis=1)
    edge_k, centre_j = (index.ravel() for index in np.meshgrid(np.arange(nr + 1), np.arange(nz), indexing='ij'))
    a_z = np.stack((u(np.clip(edge_k - 1, 0, nr - 1), centre_j), u(np.clip(edge_k, 0, nr - 1), centre_j),
                    e_r(edge_k, centre_j)), axis=1)
    if coupled:
        cliques = np.vstack((a_phi, a_r[:, [0, 1, 2, 2]], a_z[:, [0, 1, 2, 2]]))
        return element_dissection_pairs(cliques, points)
    return (element_dissection_pairs(a_phi, points)
            + element_dissection_pairs(np.vstack((a_r[:, :2], a_z[:, :2])), points))


def _arnoldi_item(host_items, transient_items, unknowns, basis, context, device_matvec, real=False,
                  single_basis=False):
    """Append an eigs caller's Arnoldi basis: a device EIGEN_PHASE item where the gpu backend runs the device
    Krylov-Schur (arpack_shift_invert.arnoldi_basis_bytes; device_matvec: a cuDSS factor; single_basis: complex64 rows
    and work vectors, eigs(single_basis=True)), else a host item (real: the float64 basis of real ARPACK)."""
    on_device, basis_bytes = arnoldi_basis_bytes(unknowns, basis, solver_backend(context.spec),
                                                 device_matvec=device_matvec, real=real, single_basis=single_basis)
    if on_device:
        row_type = 'complex64' if single_basis else 'complex128'
        transient_items.append((EIGEN_PHASE, f'device Krylov-Schur basis ({basis + 1} {row_type} rows + work '
                                             'vectors)', basis_bytes))
    else:
        host_items.append((f'Arnoldi basis ({basis} {"float64" if real else "complex128"} vectors)', basis_bytes))


def real_cross_section_operator(settings_values, context):
    """True when em_modes' cross-section operator has no imaginary part (estimators, Memory / TimeContext; the solve
    tests the assembled operator): no cpml face, no Bloch phase, no sigma_e / sigma_m and no Drude / Lorentz poles."""
    from .cross_section_modes import cpml_cross_section
    if cpml_cross_section(settings_values) or bloch_active_axes(settings_values):
        return False
    if context.material_sets('sigma_e') or context.material_sets('sigma_m'):
        return False
    from ...materials import effective_material_data   # lazy: materials imports physics.em
    for material in context.spec.materials:
        try:
            _, dispersion = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, KeyError, TypeError):   # reported by validate
            continue
        if dispersion:
            return False
    return True


def resonance_search_items(backend, unknowns, n_wanted, n_basis, own_cudss, filter_entries, scaled_block=None,
                           sparse_mass_entries=0, symmetric_mass=True, single_basis=False, symmetric_operator=None,
                           real_values=False):
    """(host items, transient items) of eigenfrequency.solve_resonances: EIGEN_PHASE the device Krylov-Schur basis
    (first pass; no widening reserve; single_basis: complex64 rows, polished_mixed_search) or ARPACK's host basis (the
    EIGEN_ROLE job's); FILTER_PHASE on the gpu the resonance filter between passes (the basis parked on the host): its
    largest single operator upload (solvers.device_matrices storage: A of filter_entries stored entries, complex; with
    a sparse B of sparse_mass_entries the FEM W of that pattern, float64, and B itself without an own factor) and its
    column blocks (scaled_block: the D v block of a polished search, None = own_cudss); in both its window / weight
    vectors and, an own cuDSS factor with a sparse B, the B copy the solves and the filter share; the host eigenvectors.
    symmetric_mass: B, W stored as lower triangles (solve_resonances(symmetric=True)); symmetric_operator: A too (None:
    = symmetric_mass when B is sparse, the FEM searches). real_values: A and B have no imaginary part (a lossless,
    PML-free model: float64, 12 of 20 B per entry; the caller's materials test), else complex values are booked. The
    own factor's copies: own_cudss_search_items."""
    if symmetric_operator is None:
        symmetric_operator = bool(sparse_mass_entries) and bool(symmetric_mass)
    real_values = bool(real_values)
    value_type = 'float64' if real_values else 'complex'
    gpu = backend == 'gpu'
    vector_bytes = COMPLEX128_BYTES * int(unknowns)
    host_items = [(f'eigenvectors ({n_wanted} x complex128)', n_wanted * vector_bytes)]
    transient_items = []
    if gpu and device_arnoldi_fits(unknowns, n_basis, host_matvec=not own_cudss):
        rows = int(n_basis) + 1   # the first pass only: a widening is checked against the pool room when it happens
        row_type, row_bytes = ('complex64', COMPLEX128_BYTES // 2) if single_basis else ('complex128', COMPLEX128_BYTES)
        transient_items += solver_job_items(EIGEN_ROLE, [
            (EIGEN_PHASE, f'device Krylov-Schur basis ({rows} {row_type} rows + 2 complex128 work vectors; a widened '
                          'search needs about twice as much)', (rows * row_bytes + 2 * COMPLEX128_BYTES) * int(unknowns))])
    else:
        host_items += solver_job_items(EIGEN_ROLE, [(f'Arnoldi basis ({n_basis} vectors)', n_basis * vector_bytes)])
    if not gpu:
        return host_items, transient_items
    vectors_label = 'resonance filter: window, region and diagonal weight / mass / row-scale vectors'
    transient_items += [(phase, vectors_label, _FILTER_VECTOR_BYTES * int(unknowns))
                        for phase in (EIGEN_PHASE, FILTER_PHASE)]
    def stored(symmetric):
        return 'lower triangle + diagonal' if symmetric else 'full CSR'
    if own_cudss and sparse_mass_entries:   # eigenfrequency: device_mass, uploaded once for both
        transient_items += solver_job_items(EIGEN_ROLE, [
            (phase, f'B = -M on the device ({value_type} {stored(symmetric_mass)}; the solves and the resonance filter '
                    'share it)', device_matrix_bytes(unknowns, sparse_mass_entries, symmetric_mass, real_values))
            for phase in (EIGEN_PHASE, FILTER_PHASE)])
    # one operator at a time (_ResonanceFilter.measures): the largest of A, the FEM W and an unshared B
    uploads = [(f'A {value_type} {stored(symmetric_operator)}',
                device_matrix_bytes(unknowns, filter_entries, symmetric_operator, real_values))]
    if sparse_mass_entries:
        uploads.append((f'W float64 {stored(symmetric_mass)}',
                        device_matrix_bytes(unknowns, sparse_mass_entries, symmetric_mass, True)))
        if not own_cudss:
            uploads.append((f'B {value_type} {stored(symmetric_mass)}',
                            device_matrix_bytes(unknowns, sparse_mass_entries, symmetric_mass, real_values)))
    upload_label, upload_bytes = max(uploads, key=lambda upload: upload[1])
    columns = min(FILTER_DEVICE_BLOCK_COLUMNS, max(int(n_wanted), 1))
    blocks = _FILTER_BLOCKS + (1 if (own_cudss if scaled_block is None else scaled_block) else 0)
    transient_items += [
        (FILTER_PHASE, f'resonance filter: one sparse operator uploaded at a time (the largest: {upload_label})',
         upload_bytes),
        (FILTER_PHASE, f'resonance filter: {blocks} column blocks of {columns} Ritz vectors', blocks * columns
         * vector_bytes)]
    return host_items, transient_items


def own_cudss_search_items(label, unknowns, entries, mixed, symmetric=True) -> list:
    """Pool items of solve_resonances' own cuDSS factor (the EIGEN_ROLE job's): its CSR copies, rhs buffer and row scale
    (cudss_kept_bytes) in EIGEN_PHASE and FILTER_PHASE, one solve's vectors in EIGEN_PHASE only (the filter returns the
    cached blocks first). label: what the copies are of."""
    kept = cudss_kept_bytes(unknowns, entries, mixed, symmetric)
    return solver_job_items(EIGEN_ROLE, [
        (EIGEN_PHASE, f'cuDSS copies of {label}, rhs buffer, row scale and one solve\'s vectors',
         kept + solve_peak_bytes(unknowns, COMPLEX128_BYTES, mixed)),
        (FILTER_PHASE, f'cuDSS copies of {label}, rhs buffer and row scale (no solve running)', kept)])


def _resonance_search(context, n_resonances, unknowns, own_cudss, filter_entries, basis_per_wanted=0,
                      scaled_block=None, sparse_mass_entries=0, symmetric_search=False, ldlt=None,
                      polished_double=False, real_values=False):
    """resonance_search_items (real_values passed on) on the run backend of a search for n_resonances: 2 n_resonances
    + 10 wanted (within 1 ..
    unknowns - 2), the Arnoldi basis of the eigen subspace_size, else basis_per_wanted x wanted, else scipy's;
    symmetric_search (solve_resonances(symmetric=True): A itself complex symmetric, FEM and the BOR FIT operator): the
    filter's lower triangles; ldlt (None = symmetric_search; True also for a symmetrized D A without Bloch phase /
    gyration): the own factor is an LDL^T. Complex64 basis rows (every polished search, eigenfrequency
    MIXED_ARPACK_TOLERANCE) on a polished_mixed_search or when polished_double (the caller's test that the search is
    polished on a double factor too: E2, or a floored complex A of an LDL^T)."""
    n_wanted = max(min(2 * n_resonances + 10, unknowns - 2), 1)
    n_basis = arnoldi_basis_size(n_wanted, unknowns, eigen_subspace_size(context.solvers), basis_per_wanted * n_wanted)
    ldlt = symmetric_search if ldlt is None else ldlt
    single_basis = search_single_basis(eigen_factorization(context.solvers), unknowns, own_cudss, ldlt,
                                       polished_double)
    return resonance_search_items(solver_backend(context.spec), unknowns, n_wanted, n_basis, own_cudss=own_cudss,
                                  filter_entries=filter_entries, scaled_block=scaled_block,
                                  sparse_mass_entries=sparse_mass_entries, symmetric_mass=symmetric_search,
                                  single_basis=single_basis, symmetric_operator=symmetric_search,
                                  real_values=real_values)


def cudss_factor_items(label, n_unknowns, entries, dimensions=2, mixed=False, general=False, value_bytes=16):
    """(external items, external extra) of one cuDSS factor (cudss_factor.factor_device_bytes on the full pattern;
    general: an LU, cudss_factor.CUDSS_LU_OVER_LDLT x the symmetric fit)."""
    permanent, extra = factor_device_bytes(n_unknowns, entries, value_bytes, dimensions, mixed, symmetric=not general)
    kind = 'LU' if general else 'LDL^T'
    precision = 'mixed' if mixed else 'double'
    return [(f'cuDSS factor of {label} ({kind}, {precision}; outside the cupy pool)', permanent)], extra


def cudss_kept_bytes(n_unknowns, entries, mixed=False, symmetric=True, value_bytes=COMPLEX128_BYTES) -> int:
    """Pool bytes one complex cuDSS plan keeps: its CSR copies (cudss_factor.operator_copy_bytes), rhs buffer and row
    scale (solve_buffer_bytes)."""
    return (operator_copy_bytes(n_unknowns, entries, value_bytes, symmetric, mixed)
            + solve_buffer_bytes(n_unknowns, value_bytes, mixed))


def cudss_own_factor_bytes(n_unknowns, entries, mixed=False, symmetric=True) -> int:
    """Pool bytes of an own (not kept) cuDSS factor inside one phase: cudss_kept_bytes + one solve's vectors."""
    return (cudss_kept_bytes(n_unknowns, entries, mixed, symmetric)
            + solve_peak_bytes(n_unknowns, COMPLEX128_BYTES, mixed))


def kept_cudss_items(run_items, transient_items, n_unknowns, entries, mixed, refactorises):
    """Append a kept (driven) cuDSS factor's pool items: its copies (run), one solve (CUDSS_SOLVE_PHASE) and, when it
    refactorises on the planned pattern, its index maps (run) and refresh peak (REFACTOR_PHASE)."""
    run_items.append(('cuDSS operator copies, rhs buffer and row scale', cudss_kept_bytes(n_unknowns, entries, mixed)))
    transient_items.append((CUDSS_SOLVE_PHASE, 'cuDSS solve vectors (one solve or probe)',
                            solve_peak_bytes(n_unknowns, COMPLEX128_BYTES, mixed)))
    if refactorises:
        run_items.append(('cuDSS index maps of the refactorisations', index_map_bytes(n_unknowns, entries)))
        transient_items.append((REFACTOR_PHASE, 'cuDSS refactorisation on the planned pattern (refresh peak)',
                                refresh_peak_bytes(entries, n_unknowns=n_unknowns, mixed=mixed)))


def eigen_mixed(factorization, unknowns) -> bool:
    """Whether an eigen runtime's own cuDSS factor of `unknowns` rows is mixed (ArpackShiftInvertSolver.factorization_for:
    a cudss_mixed request below MIXED_EIGEN_MIN_UNKNOWNS factors in double)."""
    return factorization == 'cudss_mixed' and int(unknowns) >= MIXED_EIGEN_MIN_UNKNOWNS


def search_single_basis(factorization, unknowns, own_cudss, ldlt, polished_double=False) -> bool:
    """Whether solve_resonances keeps a complex64 Krylov basis (every polished search: eigenfrequency
    MIXED_ARPACK_TOLERANCE): polished_double (the caller's test of a polish on a double factor too: E2, a floored
    complex A of an LDL^T), else an own cuDSS factor (own_cudss) that is a mixed LDL^T (ldlt; polished_mixed_search).
    The memory items (_resonance_search) and the eigen role's time recipe (SolverWork.single_basis) read it."""
    return bool(polished_double) or bool(own_cudss and polished_mixed_search(factorization, unknowns, symmetric=ldlt))


def polished_mixed_search(factorization, unknowns, symmetric=True) -> bool:
    """Whether solve_resonances on its own cuDSS factor polishes a mixed LDL^T (refine) and so keeps a complex64 Krylov
    basis (eigs_device single_basis): eigen_mixed and a symmetric pencil (an LU is never polished)."""
    return bool(symmetric) and eigen_mixed(factorization, unknowns)


def host_factorization(solvers, eigen) -> str:
    """Factorization of a host-matrix FD physics (FDFD on triangles, conical, BOR): the eigen one when eigen-solved,
    else the cuDSS / MUMPS / PARDISO linear back end, else 'superlu'."""
    return (eigen_factorization(solvers) if eigen
            else resolved_back_end(solvers) if linear_solver_cudss(solvers) or host_direct_back_end(solvers)
            else 'superlu')


def host_factor_items(name, superlu_label, superlu_bytes, unknowns, entries, symmetric=True, eigen=False) -> list:
    """Host items of a factor booked at a SuperLU heuristic: [(superlu_label, superlu_bytes)] unless `name` (the
    resolved back end or eigen factorization) is a host_direct one; then the MUMPS / PARDISO factor = that heuristic x
    host_direct.superlu_bytes_scale (symmetric: an LDL^T of D A / of the complex symmetric pencil; complex64 values when
    mixed, an eigen mixed request below MIXED_EIGEN_MIN_UNKNOWNS factors in double; uncalibrated) + its solver copies
    (host_direct.solver_host_bytes). A resonance search's caller tags them solver_job_items(EIGEN_ROLE, ...)."""
    if name not in HOST_DIRECT_FACTORIZATIONS:
        return [(superlu_label, int(superlu_bytes))]
    if eigen and host_direct_precision(name) == 'mixed' and int(unknowns) < MIXED_EIGEN_MIN_UNKNOWNS:
        name = name.removesuffix('_mixed')   # ArpackShiftInvertSolver.factorization_for (S5c)
    mixed = host_direct_precision(name) == 'mixed'
    library, kind = factor_label(name), 'LDL^T' if symmetric else 'LU'
    return [(f"{library} {kind} factor ({'complex64' if mixed else 'complex128'}, SuperLU heuristic scaled, "
             'uncalibrated)', int(superlu_bytes * superlu_bytes_scale(name, symmetric=symmetric, complex_values=True))),
            (f'{library} solver copies (equilibrated operator, row scale; mixed: GMRES vectors)',
             int(solver_host_bytes(unknowns, entries, complex_values=True, mixed=mixed, symmetric=symmetric)))]


def later_factorisations(context) -> bool:
    """Whether the study factorises the operator again after its first solve (refactorize on the planned pattern):
    more than one study wavelength, an optimization, a time-dependent study or a reused sweep build."""
    study = context.spec.study
    sweep = getattr(study, 'sweep', None)
    if sweep is not None and getattr(sweep, 'reuse_build', False):
        return True
    if study.type == FREQUENCY_DOMAIN_STUDY:
        return _harmonic_wavelength_count(context) > 1
    return study.type in ('optimization', 'time_dependent')


def _off_diagonal_items(context, n_cells, sparse_row):
    """em_fdfd_hz with eps_xy (a material sets it or a coupling writes eps_xy_update) or a material gyration_z: (run
    items, host items) of the coupled change array (eps_xy) / the assembled gyration, the two face couplings, the 4
    extra operator entries per row and the larger factor."""
    def writes_xy(coupling):
        if not (coupling.enabled and coupling.type in _OFF_DIAGONAL_WRITERS
                and coupling.target == context.physics_spec.name):
            return False
        needed = _OFF_DIAGONAL_WRITERS[coupling.type]
        return needed is None or any(context.material_sets(name) for name in needed)
    off_diagonal = context.material_sets('eps_xy') or any(writes_xy(coupling) for coupling in context.spec.couplings)
    gyrotropic = context.material_sets(GYRATION_Z)   # magneto-optic: the same couplings, eta_xy != eta_yx
    if not (off_diagonal or gyrotropic):
        return [], []
    run_items = [('eps_xy coupled change (float32)', FLOAT32_BYTES * n_cells)] if off_diagonal else []
    if gyrotropic:
        run_items.append(('assembled gyration_z kept on the system (float64)', FLOAT64_BYTES * n_cells))
    run_items += [('eps_xy face couplings (2 sparse complex128, 4 entries per row)',
                  2 * (4 * COMPLEX128_CSR_ENTRY_BYTES + sparse_row) * n_cells),
                 ('eps_xy operator entries (4 per row) and eta transients (3 x complex128)',
                  (4 * COMPLEX128_CSR_ENTRY_BYTES + 3 * COMPLEX128_BYTES) * n_cells),
                 ('assembled eps_xy kept on the system (float64)', FLOAT64_BYTES * n_cells)]
    host_items = [('SuperLU fill of the 9-point eps_xy operator (heuristic)',
                   int(_OFF_DIAGONAL_EXTRA_FILL * host_lu_bytes(n_cells)))]
    return run_items, host_items


def _study_values(context):
    """Resolved settings of the model's study (defaults filled), {} when they do not resolve (validate reports)."""
    study = context.spec.study
    try:
        params = context.registry.study_type(study.type).params
        return resolve_values(params, study.settings if isinstance(study.settings, dict) else {}, context.parameters)
    except (KeyError, ValueError, TypeError):
        return {}


def _harmonic_wavelength_count(context):
    """Study wavelengths of a frequency_domain study (0 in other studies or when they do not resolve)."""
    if context.spec.study.type != FREQUENCY_DOMAIN_STUDY:
        return 0
    try:
        return len(_study_wavelengths(context.spec, context.parameters, context.registry))
    except (KeyError, ValueError, TypeError):
        return 0


def _point_trace_items(context, mesh):
    """(build items, run items, host items) of the FieldProbe traces of the enabled field_probe / resonance_monitor
    features (model.definition_resonance.point_trace_items in EM steps)."""
    return point_trace_items(context, mesh, _POINT_TRACE_FEATURES)


def _cpml_psi_bytes(mesh, settings_values, value_bytes=FLOAT32_BYTES):
    """Two psi slabs (e and h rows, value_bytes per entry: float32, float64 for the double stepper) of npml x the other
    axis per CPML side (boundaries.CPMLSided; a mirror face's padding is its ghost cell, no layer)."""
    sides = mesh.npml_sides
    total = 0
    for side, npml in sides.items():
        if settings_values.get(f'bc_{side}', 'cpml') != 'cpml':
            continue
        other_axis = int(mesh.ny) if side.startswith('x') else int(mesh.nx)
        total += 2 * int(npml) * other_axis * value_bytes
    return total


def _pole_regions(context):
    """[(box cells, n_drude, n_lorentz)] of the material regions with poles."""
    return [(region.box_cells, region.n_drude, region.n_lorentz) for region in context.material_regions
            if region.n_drude or region.n_lorentz]


# features that rewrite sigma_e at run time (the FDTD engines then keep a per-cell ca)
_SIGMA_E_FEATURES = frozenset({'phase_change_medium'})


def fdtd_coefficient_arrays(polarization, context, nonlinear=False) -> int:
    """Full-grid coefficient arrays of the 2D FDTD engines (fdtd_2d_base "Coefficient storage"): cb (hz cb_x, cb_y)
    always; ca (hz ca_x, ca_y) per cell when a material sets sigma_e (an expression counts) or a feature rewrites it,
    else the uniform 1; da / db per cell when a material sets mu_r != 1 or sigma_m, else one value; hz eps_x / eps_y
    only with a nonlinear medium."""
    faces = 1 if polarization == 'ez' else 2
    count = faces
    if context.material_sets('sigma_e') or any(feature.enabled and feature.type in _SIGMA_E_FEATURES
                                                for feature in context.physics_spec.features):
        count += faces
    if context.material_sets('mu_r', default=1) or context.material_sets('sigma_m'):
        count += 2
    if nonlinear and polarization == 'hz':
        count += 2
    return count


def fdtd_memory_estimate(polarization):
    """PhysicsType.memory_estimate of em_fdtd_<polarization> (fdtd_ez_2d / fdtd_hz_2d, fdtd_2d_base, boundaries): the
    field-side arrays (fields, coefficients, CPML psi, pole buffers, jz_total, hz pole power totals) in the resolved
    stepper precision (float32 single, float64 double); q_accum, PEC keeps and masks stay float32 / bool."""
    def estimate(settings_values, mesh, context):
        n_cells = int(mesh.n_cells)
        double = resolved_stepper_precision(context.solvers) == 'double'
        real_bytes, real_name = (FLOAT64_BYTES, 'float64') if double else (FLOAT32_BYTES, 'float32')
        regions = _pole_regions(context)
        any_drude = any(n_drude for _, n_drude, _ in regions)
        q_accum = bool(settings_values.get('track_absorption')) or _q_accum_referenced(
            context.spec, context.physics_spec.name, context.registry)
        # C5 nonlinear_medium (its own arrays: the feature estimator): hz adds its cells to the pole cell mask, its
        # power to the pole power totals and needs the face eps
        nonlinear = any(feature.enabled and feature.type == 'nonlinear_medium'
                        for feature in context.physics_spec.features)
        n_coefficients = fdtd_coefficient_arrays(polarization, context, nonlinear)
        items = [(f'{FDTD_FIELDS_ITEM} (3 x {real_name})', 3 * real_bytes * n_cells),
                 (f'{FDTD_COEFFICIENTS_ITEM} ({n_coefficients} x {real_name})',
                  n_coefficients * real_bytes * n_cells),
                 ('CPML psi slabs', _cpml_psi_bytes(mesh, settings_values, real_bytes))]
        if q_accum:   # float32 q_accum (+ hz pole power totals while a pole or nonlinear medium exists)
            pole_totals = 2 if polarization == 'hz' and (regions or nonlinear) else 0
            items.append((f'absorption accumulator (float32 + {pole_totals} x {real_name})',
                          (FLOAT32_BYTES + pole_totals * real_bytes) * n_cells))
        pole_bytes = 0
        for box_cells, n_drude, n_lorentz in regions:
            masks = (n_drude + n_lorentz) * BOOL_BYTES * box_cells   # dispersive_regions cropped masks
            if polarization == 'ez':   # drive + j per Drude, drive + p + dp per Lorentz, on the box
                pole_bytes += masks + (2 * n_drude + 3 * n_lorentz) * real_bytes * box_cells
            else:   # two face sets: shared weights, j per Drude, p + dp per Lorentz
                pole_bytes += masks + 2 * (1 + n_drude + 2 * n_lorentz) * real_bytes * box_cells
        if regions:
            pole_bytes += (real_bytes if polarization == 'ez' and any_drude else 0) * n_cells   # jz_total
        if regions or nonlinear:
            pole_bytes += (BOOL_BYTES if polarization == 'hz' else 0) * n_cells                 # pole cell mask
        items.append(('dispersive poles', pole_bytes))
        if bloch_active_axes(settings_values):   # Bloch k != 0: imaginary fields, CPML psi and pole buffers
            imaginary_pole_bytes = sum((1 if polarization == 'ez' else 2) * (n_drude + 2 * n_lorentz) * real_bytes
                                       * box_cells for box_cells, n_drude, n_lorentz in regions)
            if polarization == 'ez' and any_drude:
                imaginary_pole_bytes += real_bytes * n_cells   # imaginary jz_total
            items.append((FDTD_BLOCH_ITEM, 3 * real_bytes * n_cells
                          + _cpml_psi_bytes(mesh, settings_values, real_bytes) + imaginary_pole_bytes))
        trace_items, run_items, host_items = _point_trace_items(context, mesh)
        transient_items = []
        if q_accum:   # fdtd_2d_base: reset_q_accum parks last_q_accum on the host (pinned on the gpu, lazy);
            # take_absorbed_power returns a new array (+ the uploaded last window at the end-of-run merge)
            host_items = list(host_items) + [('last absorption window (last_q_accum, float32, pinned)',
                                              FLOAT32_BYTES * n_cells)]
            transient_items.append(('absorbed power transfer', 'mean power density of one transfer and the merged '
                                    'last window (2 x float32)', 2 * FLOAT32_BYTES * n_cells))
        n_wavelengths = _harmonic_wavelength_count(context)
        if n_wavelengths:   # harmonic_fdtd.HarmonicAccumulator (attach_harmonic in the study): a window DFTMonitor
            n_spectra = _DFT_SPECTRA[polarization]
            run_items.append((f'harmonic accumulators ({n_spectra} x {n_wavelengths} wavelengths x complex64)',
                              n_spectra * n_wavelengths * COMPLEX64_BYTES * window_cell_count(mesh)))
        # finalize's float32 Courant check (requirement, ratio; hz min(eps)) and two pole weight temporaries (field
        # dtype: the cell weight and its face average); the coefficient layout checks reduce without temporaries
        build_extra = (3 * FLOAT32_BYTES + 2 * real_bytes) * n_cells
        parked = fdtd_parked_material_arrays(polarization, context.spec, context.registry, context.physics_spec,
                                             context.parameters)
        if parked:   # EMPhysics(park_arrays=): on the device through the material painting, on the host from create
            build_extra, extra_items = _parked_fdtd_items(polarization, parked, context, mesh, build_extra,
                                                          3 * real_bytes * n_cells, q_accum)
            items += extra_items
            host_items = list(host_items) + [(f"material arrays parked on the host ({', '.join(parked)}, float32)",
                                              FLOAT32_BYTES * len(parked) * n_cells)]
            window_arrays = [name for name in (*PERMITTIVITY_ARRAYS[polarization], 'mu_r') if name in parked]
            if n_wavelengths and window_arrays:   # the pulse's energy check uploads their windows per chunk
                transient_items.append(('harmonic energy check', f"parked {', '.join(window_arrays)} window uploads "
                                        '(float32)', FLOAT32_BYTES * len(window_arrays) * window_cell_count(mesh)))
        return MemoryEstimate.from_items(items + trace_items, build_extra=build_extra, host_items=host_items,
                                         run_items=run_items, transient_items=transient_items)
    return estimate


def _parked_fdtd_items(polarization, parked, context, mesh, build_extra, field_bytes, q_accum):
    """(build extra, kept items) of em_fdtd_<polarization> with the MaterialGrid arrays `parked` on the host
    (fdtd_2d_base readers): the Courant check's device copies of a varying parked eps_r (ez, any eps counts: poles
    write their background) / mu_r on top of its temporaries, the painted arrays before the fields exist, the hz E
    rebuild's whole-array uploads (the subpixel gathers upload one array at a time after it), the x-row block uploads
    (ez E, both H); kept: ez's sigma_e copy on the conducting cells' box while accumulating (the ohmic loss)."""
    n_cells = int(mesh.n_cells)
    copies = int(polarization == 'ez' and 'eps_r' in parked)
    copies += int('mu_r' in parked and context.material_sets('mu_r', default=1))
    rebuild_uploads = (sum(1 for name in ('eps_xx', 'eps_yy', 'sigma_e') if name in parked)
                       if polarization == 'hz' else 0)
    build_extra = max(build_extra + FLOAT32_BYTES * copies * n_cells,
                      FLOAT32_BYTES * len(parked) * n_cells - field_bytes,
                      FLOAT32_BYTES * max(rebuild_uploads, 1) * n_cells,
                      FLOAT32_BYTES * len(parked) * min(n_cells, PARKED_BLOCK_CELLS))
    items = []
    if polarization == 'ez' and q_accum and 'sigma_e' in parked and context.material_sets('sigma_e'):
        items.append(('sigma_e copy on the box of the conducting cells (float32, ohmic loss)',
                      FLOAT32_BYTES * _property_box_cells(context, 'sigma_e', n_cells)))
    return build_extra, items


def _property_box_cells(context, property_name, n_cells):
    """Full-grid cells of the union bounding box of the material regions setting property_name to a non-zero value
    (n_cells when a region's box does not resolve; 0 without such a region)."""
    from ...materials import effective_material_data   # lazy: materials imports physics.em
    bounds = None
    regions = {region.name: region for region in context.material_regions}
    for material in context.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):   # reported by validate
            continue
        value = properties.get(property_name)
        if value is None or (isinstance(value, (int, float)) and value == 0):
            continue
        region = regions.get(material.name)
        if region is None or not region.box_bounds:
            return n_cells
        bounds = (region.box_bounds if bounds is None else
                  tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(bounds, region.box_bounds)))
    if bounds is None:
        return 0
    return min(n_cells, math.prod(last - first + 1 for first, last in bounds))


def fdfd_memory_estimate(polarization):
    """PhysicsType.memory_estimate of em_fdfd_<polarization> (solvers/fdfd_2d, harmonic, operators; the resolved
    sparse_direct back end of context.solvers): the system and solution are run items (an eigen search on the gpu:
    the system an ASSEMBLY_PHASE item, parked during the search), its assembly an ASSEMBLY_PHASE item; a cuDSS factor
    is external (an adjoint solve on a Bloch-phase LU refactorises it into the transpose's, one LU; with eps_xy a
    second LU is booked). Triangles: _fdfd_fem_memory_estimate."""
    def estimate(settings_values, mesh, context):
        if getattr(mesh, 'kind', 'tensor') == 'triangles':
            return _fdfd_fem_memory_estimate(settings_values, mesh, context)
        n_cells = int(mesh.n_cells)
        sparse_row = INT32_BYTES   # CSR indptr entry per row
        gpu = solver_backend(context.spec) == 'gpu'
        # complex128 face weights and the float32 Im(cell eps) on the run backend; cell eps, cell mu (hz: and the
        # E-face eps) on the host
        n_host_arrays = FDFD_HOST_SYSTEM_ARRAYS[polarization]
        n_loss_arrays = FDFD_LOSS_ARRAYS[polarization]
        n_system_arrays = (4 if polarization == 'ez' else 7) - n_host_arrays
        items = [('state re / im / q_abs (3 x float32)', 3 * FLOAT32_BYTES * n_cells),
                 ('dispersive pole masks', sum((n_drude + n_lorentz) * BOOL_BYTES * box_cells
                                               for box_cells, n_drude, n_lorentz in _pole_regions(context)))]
        solution_bytes = 3 * COMPLEX128_BYTES * n_cells
        run_items = [('solution and face fields (3 x complex128)', solution_bytes)]
        # the device HarmonicSystem: run items, or ASSEMBLY_PHASE items where the eigen search parks it; the plan's
        # terms and the Yee entry maps wait on the host between assemblies on the gpu (operators._ParkedInt32)
        parked_bytes = (_FDFD_PLAN_TERM_BYTES + _YEE_ENTRY_MAP_BYTES + _YEE_E_PATTERN_BYTES) * n_cells
        # the systems keep dx_h / dy_h as CSR, dx_e / dy_e matrix-free (fdfd_2d._drop_e_operators: O(nx + ny)); the
        # e CSR values live in the assembly, their indices on the kept host patterns (cpu) or views of the uploaded
        # park, held by them through the assembly (gpu: entry maps + e patterns)
        assembly_only = _YEE_E_CSR_BYTES - _YEE_E_PATTERN_BYTES
        if gpu:
            assembly_only += _YEE_ENTRY_MAP_BYTES + _YEE_E_PATTERN_BYTES
        system_items = [('Yee operators (dx_h, dy_h sparse complex128; dx_e, dy_e matrix-free)',
                         2 * (2 * COMPLEX128_CSR_ENTRY_BYTES + sparse_row) * n_cells),
                        ('kept Helmholtz pattern (int32; polar grids have none)', _FDFD_PLAN_PATTERN_BYTES * n_cells),
                        ('Helmholtz operator (sparse complex128)',
                         (5 * COMPLEX128_CSR_ENTRY_BYTES + sparse_row) * n_cells),
                        (f'system arrays ({n_system_arrays} x complex128)',
                         n_system_arrays * COMPLEX128_BYTES * n_cells),
                        (f'Im(cell eps) of q_abs ({n_loss_arrays} x float32)', n_loss_arrays * FLOAT32_BYTES * n_cells)]
        if not gpu:
            system_items.append(('kept Helmholtz term plan, Yee kron entry maps and e patterns (int32)', parked_bytes))
        # _assemble: the old system is dropped first; a single solve assembles before its solution and factor exist
        assembly_bytes = (_FDFD_ASSEMBLY_BYTES['gpu' if gpu else 'cpu'] + assembly_only) * n_cells
        single_solve = not later_factorisations(context)
        later_bytes = solution_bytes   # run items made after a single solve's assembly
        transient_items = []
        external_items, external_extra = [], 0
        host_items = []
        off_diagonal_run, off_diagonal_host = (_off_diagonal_items(context, n_cells, sparse_row)
                                               if polarization == 'hz' else ([], []))
        entries = (9 if off_diagonal_run else 5) * n_cells   # eps_xy: 4 more entries per row
        line_block_items = _line_block_run_items(settings_values, mesh, context, polarization)
        operator_copy = (5 * COMPLEX128_CSR_ENTRY_BYTES + sparse_row) * n_cells
        back_end = resolved_back_end(context.solvers)
        superlu_host = False   # host items with a SuperLU factor: the eps_xy fill below adds to them
        eigen = eigen_solved(context.spec, context.physics_spec.name)
        # fdfd_2d._park_device_system: on the gpu without a line_block solver the eigen search frees the device system
        # (and its eps_xy couplings) and reassembles it afterwards, so it is alive only around the assemblies
        parked = eigen and solver_backend(context.spec) == 'gpu' and not line_block_items
        if eigen:   # no driven solve: the shift-invert factor instead
            from .cross_section_modes import cpml_cross_section
            bloch = bool(bloch_active_axes(settings_values))
            host_items, eigen_transient, external_items, external_extra = _eigen_items(   # fdfd_2d._symmetrizer: D A
                context, n_cells, sparse_row, superlu=not line_block_items,              # unless Bloch / gyration_z
                ldlt=not bloch and not context.material_sets(GYRATION_Z),
                # E2 (_adjoint_symmetrizer: no Bloch phase, eps_xy or gyration_z) polishes a complex A on any factor:
                # an open model (a lossy closed one is booked unpolished)
                polished_double=not bloch and not off_diagonal_run and cpml_cross_section(settings_values))
            superlu_host = not line_block_items and eigen_factorization(context.solvers) not in CUDSS_FACTORIZATIONS
            transient_items += eigen_transient
            run_items += line_block_items
        elif back_end in HOST_FACTOR_BACK_ENDS:
            # MUMPS / PARDISO: the LDL^T of D A (an LU with a Bloch phase), recyclable like SuperLU (one kept factor)
            host_items = [*host_factor_items(back_end, 'SuperLU L + U factors (heuristic)', host_lu_bytes(n_cells),
                                             n_cells, entries, symmetric=not bloch_active_axes(settings_values)),
                          ('host operator copy', operator_copy)]
            superlu_host = True
        elif back_end in CUDSS_BACK_ENDS:   # kept between solves (_factorization, _cudss_previous)
            mixed = back_end == 'cudss_mixed'
            # no symmetrizer with a Bloch phase: an LU, and an adjoint solve its transpose's LU; with eps_xy D A passes
            # the asymmetry check (an LDL^T, one plan: em_fdfd_adjoint_cpu_gpu xy runs, 2026-11)
            general = bool(bloch_active_axes(settings_values))
            kept = cudss_kept_bytes(n_cells, entries, mixed, symmetric=not general)
            run_items.append(('cuDSS operator copies, rhs buffer and row scale', kept))
            later_bytes += kept
            transient_items.append((CUDSS_SOLVE_PHASE, 'cuDSS solve vectors (one solve or probe)',
                                    solve_peak_bytes(n_cells, COMPLEX128_BYTES, mixed)))
            external_items, external_extra = cudss_factor_items('the Helmholtz operator', n_cells, entries,
                                                                mixed=mixed, general=general)
            adjoint_lu = general and _has_adjoint_objective(context)
            if adjoint_lu and off_diagonal_run:   # _transposed_solve: the eps_xy pattern may differ from its transpose:
                transposed_kept = cudss_kept_bytes(n_cells, entries, mixed, symmetric=False)   # a second LU (upper
                run_items.append(('cuDSS operator copies of the transposed (adjoint) LU', transposed_kept))   # figure)
                later_bytes += transposed_kept
                transposed, _ = cudss_factor_items('the transposed operator (adjoint)', n_cells, entries, mixed=mixed,
                                                   general=True)
                external_items += transposed
            # else the 5-point Bloch pattern is its transpose's: the adjoint refactorises the one LU in place
            if not single_solve or (adjoint_lu and not off_diagonal_run):   # a refactorisation on the planned pattern
                run_items.append(('cuDSS index maps of the refactorisations', index_map_bytes(n_cells, entries)))
                transient_items.append((REFACTOR_PHASE, 'cuDSS refactorisation on the planned pattern (refresh peak)',
                                        refresh_peak_bytes(entries, n_unknowns=n_cells, mixed=mixed)))
            host_items = [('host operator copies of the cuDSS factorisation (equilibration, symmetry check)',
                           2 * operator_copy)]
        else:
            run_items += line_block_items
        if not eigen:   # fdfd_2d._solve builds the rhs after the factorisation: alive with the solve only
            transient_items.append((CUDSS_SOLVE_PHASE if back_end in CUDSS_BACK_ENDS else SOLVE_PHASE,
                                    'driven rhs (complex128, built after the factorisation)',
                                    COMPLEX128_BYTES * n_cells))
        if parked:   # all but the physics' own eps_xy change array live on the system
            system_items += [item for item in off_diagonal_run if not item[0].startswith('eps_xy coupled change')]
            run_items += [item for item in off_diagonal_run if item[0].startswith('eps_xy coupled change')]
            transient_items += [(ASSEMBLY_PHASE, f'{label} (freed during the eigen search)', size)
                                for label, size in system_items]
        else:
            run_items += system_items + off_diagonal_run
        host_items += off_diagonal_host if superlu_host else []
        host_label = 'cell eps, cell mu' if polarization == 'ez' else 'cell eps_xx / eps_yy, cell mu, E-face eps'
        host_items.append((f'system arrays on the host ({n_host_arrays} x complex128: {host_label})',
                           n_host_arrays * COMPLEX128_BYTES * n_cells))
        if gpu:
            host_items.append(('Helmholtz term plan, Yee kron entry maps and e patterns (int32, pinned between '
                               'assemblies)', parked_bytes))
        transient_items.append((ASSEMBLY_PHASE, 'Helmholtz assembly (term plan build sort / term buffer)'
                                + (' before the solution and factor exist' if single_solve else ''),
                                phase_before_items(assembly_bytes, later_bytes if single_solve else 0)))
        return MemoryEstimate.from_items(items, host_items=host_items, run_items=run_items,
                                         transient_items=transient_items, external_items=external_items,
                                         external_extra=external_extra)
    return estimate


def _has_adjoint_objective(context) -> bool:
    """Whether the physics has an enabled adjoint_objective feature (an eps_sensitivity solve follows the driven one)."""
    return any(feature.enabled and feature.type == 'adjoint_objective' for feature in context.physics_spec.features)


def _fdfd_fem_memory_estimate(settings_values, mesh, context):
    """em_fdfd_* on a triangles mesh (solvers/fdfd_2d_fem, fem_fdfd): 3 float32 per-element state fields; the host
    SuperLU factor (or a cuDSS one: external, pool copies kept / eigen phase), host blocks, vectors, eigen Arnoldi work."""
    from .fem_fdfd import FEM_FDFD_ENTRIES_PER_ROW, fem_fdfd_lu_bytes, scalar_unknowns
    from .fem_modes import DISCRETIZATION_ORDERS
    order = DISCRETIZATION_ORDERS[settings_values.get('discretization', 'quadratic')]
    n_elements = int(mesh.n_elements)
    unknowns = scalar_unknowns(int(mesh.n_nodes), int(mesh.n_edges), order)
    entries = FEM_FDFD_ENTRIES_PER_ROW[order] * unknowns
    eigen = eigen_solved(context.spec, context.physics_spec.name)
    factorization = host_factorization(context.solvers, eigen)
    on_device = factorization in CUDSS_FACTORIZATIONS
    vector_bytes = COMPLEX128_BYTES * unknowns
    # MUMPS / PARDISO: the LDL^T of the complex symmetric pencil (driven: kept and refactorised in place, one factor)
    factor_items = [] if on_device else host_factor_items(
        factorization, 'SuperLU L + U factors (scalar FEM fill, heuristic)', fem_fdfd_lu_bytes(unknowns, order),
        unknowns, entries, symmetric=True, eigen=eigen)
    host_items = (solver_job_items(EIGEN_ROLE, factor_items) if eigen else factor_items) + [
        ('assembled K, M, energy matrix and operator (sparse, rough)', 4 * entries * COMPLEX128_CSR_ENTRY_BYTES),
        ('solution, rhs and dof vectors (complex128)', 4 * vector_bytes),
        ('nodal display fields (3 complex128 + float64 parts)', 8 * FLOAT64_BYTES * unknowns)]
    transient_items = []
    if eigen:   # eigenfrequency.solve_resonances
        n_resonances = int(_study_values(context).get('n_resonances') or 1)
        search_host, transient_items = _resonance_search(context, n_resonances, unknowns, on_device, entries,
                                                         sparse_mass_entries=entries,   # A, B, W: one pattern
                                                         symmetric_search=True,   # lossless, no PML ring: real
                                                         real_values=real_cross_section_operator(settings_values,
                                                                                                 context))
        host_items += [*search_host,
                       (f'resonance vectors ({n_resonances} x complex128)', n_resonances * vector_bytes),
                       ('shifted operator copy (sparse complex128)', entries * COMPLEX128_CSR_ENTRY_BYTES)]
    run_items, external_items, external_extra = [], [], 0
    if on_device:   # cuDSS LDL^T: the lower-triangle CSR copies, solve vectors; the factor outside the pool
        mixed = eigen_mixed(factorization, unknowns) if eigen else factorization == 'cudss_mixed'
        external_items, external_extra = cudss_factor_items('the FEM operator', unknowns, entries, mixed=mixed)
        if eigen:   # solve_resonances' own factor lives for the eigen solve
            transient_items += own_cudss_search_items('the FEM operator', unknowns, entries, mixed)
            external_items = solver_job_items(EIGEN_ROLE, external_items)
        else:       # the driven factor is kept and refactorised in place (fdfd_2d_fem previous=)
            kept_cudss_items(run_items, transient_items, unknowns, entries, mixed, later_factorisations(context))
    return MemoryEstimate.from_items((('state re / im / q_abs (3 x float32 per element)',
                                       3 * FLOAT32_BYTES * n_elements),), host_items=host_items, run_items=run_items,
                                     transient_items=transient_items, external_items=external_items,
                                     external_extra=external_extra)


def _line_block_run_items(settings_values, mesh, context, polarization):
    """Run items of a line_block 'linear' solver (fdfd_line_block): the bound operator and one kept factor; [] for any
    other solver. A rejected or adjoint solve's host SuperLU is not counted, nor a frequency_domain wavelength batch
    (it takes only what line_block.MEMORY_FRACTION of the free memory holds at run time)."""
    resolved = (context.solvers or {}).get('linear')
    if resolved is None or resolved.solver_type.name != 'line_block':
        return []
    from .fdfd_line_block import fdfd_line_block_bytes
    precision = resolved.values.get('precision', 'mixed')
    operator, factor = fdfd_line_block_bytes(mesh.nx, mesh.ny, settings_values['bc_x_lo'] == 'periodic',
                                             settings_values['bc_y_lo'] == 'periodic', precision,
                                             entries_per_row=9 if polarization == 'hz' else 5)
    return [(f'line_block operator ({precision})', operator), (f'line_block factor ({precision})', factor)]


def _eigen_items(context, n_cells, sparse_row, superlu=True, ldlt=True, polished_double=False):
    """(host items, EIGEN_PHASE items, external items, external extra) of eigenfrequency.solve_resonances: its own
    factor (superlu: no line_block solver; SuperLU on the host, a cudss eigen factorization outside the pool with its
    pool copies; ldlt: the LDL^T of D A, so a polished mixed one keeps a complex64 basis, as any search when
    polished_double), the operator copy, the Arnoldi work (resonance_search_items), the resonance fields and vectors."""
    values = _study_values(context)
    n_resonances = int(values.get('n_resonances', values.get('n_optical_resonances')) or 1)
    vector_bytes = COMPLEX128_BYTES * n_cells
    factorization = eigen_factorization(context.solvers)
    own_cudss = superlu and factorization in CUDSS_FACTORIZATIONS   # a symmetrized LDL^T of D A, freed after the solve
    # MUMPS / PARDISO: the symmetrized LDL^T of D A on the host (an LU with a Bloch phase or eps_xy: booked as LDL^T)
    factor = (solver_job_items(EIGEN_ROLE, host_factor_items(
        factorization, 'SuperLU L + U factors (heuristic)', host_lu_bytes(n_cells), n_cells, 5 * n_cells,
        symmetric=True, eigen=True)) if superlu and not own_cudss else [])
    # any factor may be Rayleigh-polished (E2: a complex A with a measured symmetrizer): the D v block booked always
    search_host, transient_items = _resonance_search(context, n_resonances, n_cells, own_cudss, 5 * n_cells,
                                                     scaled_block=True, ldlt=ldlt, polished_double=polished_double)
    external_items, external_extra = [], 0
    if own_cudss:
        mixed = eigen_mixed(factorization, n_cells)
        transient_items += own_cudss_search_items('the shifted operator', n_cells, 5 * n_cells, mixed)
        external_items, external_extra = cudss_factor_items('the shifted operator', n_cells, 5 * n_cells, mixed=mixed)
        external_items = solver_job_items(EIGEN_ROLE, external_items)
    host_items = factor + [
        ('eigen host operator copy', (5 * COMPLEX128_CSR_ENTRY_BYTES + sparse_row) * n_cells), *search_host,
        (f'resonance fields ({n_resonances} x complex128)', n_resonances * vector_bytes),
        ('eigen solve vectors', _EIGEN_HOST_VECTORS * vector_bytes)]
    return host_items, transient_items, external_items, external_extra


def modes_memory_estimate(settings_values, mesh, context):
    """PhysicsType.memory_estimate of em_modes (solvers/modes_2d, cross_section_modes): 9 float32 mode fields (the
    two of state() on the device, the rendered seven host); host SuperLU, operators, Arnoldi basis, modes; a line_block
    batch or a per-solve cuDSS LU (external) instead of SuperLU."""
    if getattr(mesh, 'kind', 'tensor') == 'triangles':
        return _modes_fem_memory_estimate(settings_values, mesh, context)
    n_cells = int(mesh.n_cells)
    unknowns = 2 * n_cells   # mirrors halve it: an upper figure
    n_modes = int(settings_values.get('n_modes', 4))
    from .cross_section_modes import cpml_cross_section, mode_default_basis, mode_eigen_count
    n_eigen = max(mode_eigen_count(n_modes, unknowns), 1)   # wide, an upper figure: lossless solves ask n_modes + 4
    arnoldi_vectors = arnoldi_basis_size(n_eigen, unknowns, eigen_subspace_size(context.solvers),
                                         mode_default_basis(n_eigen, cpml_cross_section(settings_values)))
    factorization = eigen_factorization(context.solvers)
    on_device = factorization in CUDSS_FACTORIZATIONS
    operator_bytes = 2 * 9 * COMPLEX128_CSR_ENTRY_BYTES * unknowns
    host_items = [('operators A, B (sparse complex128)', operator_bytes),
                  (f'mode fields ({n_modes} modes x 6 components)', 6 * n_modes * COMPLEX128_BYTES * n_cells),
                  ('rendered fields of the selected mode (7 x float32: E, H, h_abs)', 7 * FLOAT32_BYTES * n_cells),
                  # ModePhysics._cross_section_materials: built on the host (no device temporaries)
                  ('cross-section materials per solve (6 x float32 copies; eps_xx, eps_yy, eps_zz, mu complex128)',
                   (6 * FLOAT32_BYTES + 4 * COMPLEX128_BYTES) * n_cells)]
    run_items, external_items, external_extra = [], [], 0
    transient_items = []
    shifted = (context.solvers or {}).get('shifted')
    line_block = shifted is not None and shifted.solver_type.name == 'line_block'
    # E1: an own factor of a real operator is float64 and host ARPACK runs real (float64 basis)
    real = not line_block and real_cross_section_operator(settings_values, context)
    _arnoldi_item(host_items, transient_items, unknowns, arnoldi_vectors, context, on_device, real=real)
    if on_device:   # cross_section_modes.solve_prepared_modes: an own LU per solve (no plan reuse), dropped after it
        mixed = eigen_mixed(factorization, unknowns)
        # E1: a real operator factors in float64 (copies and factor at 8 B per value); its solves of the complex Krylov
        # vectors run as two real solves (complex128 vectors either way). B = diag(eps^-1) is a device vector of
        # device_factor_matvec (float64 when real)
        value_bytes = FLOAT64_BYTES if real else COMPLEX128_BYTES
        value_type = 'float64' if real else 'complex128'
        transient_items += [
            (EIGEN_PHASE, f'cuDSS operator copy ({value_type}), rhs buffer, row scale and solve vectors',
             cudss_kept_bytes(unknowns, 9 * unknowns, mixed, symmetric=False, value_bytes=value_bytes)
             + solve_peak_bytes(unknowns, COMPLEX128_BYTES, mixed)),
            (EIGEN_PHASE, f'device mass diagonal B ({value_type} vector) and B x (complex128)',
             (value_bytes + COMPLEX128_BYTES) * unknowns)]
        external_items, external_extra = cudss_factor_items('the shifted mode operator', unknowns, 9 * unknowns,
                                                            mixed=mixed, general=True, value_bytes=value_bytes)
    if line_block:   # ModePhysics line-block batches
        from .cross_section_modes import line_block_bytes
        from .fdfd_line_block import MAX_BATCH_WAVELENGTHS
        per_solve = 3 if settings_values.get('group_index') else 1
        n_operators = min(MAX_BATCH_WAVELENGTHS, per_solve * max(_harmonic_wavelength_count(context), 1))
        precision = shifted.values.get('precision', 'mixed')
        run_items.append((f'line_block batch ({n_operators} shift-invert operators, {precision}; sized to the free '
                          'memory at run time)',
                          line_block_bytes(mesh.nx, mesh.ny, settings_values['bc_x_lo'] == 'periodic',
                                           settings_values['bc_y_lo'] == 'periodic', precision, n_operators)))
        host_items.append((f'kept problems of the batch ({n_operators} x operators A - shift B, A)',
                           (n_operators - 1) * operator_bytes))
    elif not on_device:   # MUMPS / PARDISO: an LU (the Yee cross-section operator is not complex symmetric)
        from ...core.memory_estimate import host_real_lu_bytes
        real_superlu = real and factorization not in HOST_DIRECT_FACTORIZATIONS   # host direct: kept complex (scaled)
        host_items[:0] = host_factor_items(factorization, 'SuperLU L + U factors (float64, heuristic)' if real_superlu
                                           else 'SuperLU L + U factors (heuristic)',
                                           host_real_lu_bytes(unknowns) if real_superlu else host_lu_bytes(unknowns),
                                           unknowns, 9 * unknowns, symmetric=False, eigen=True)
    return MemoryEstimate.from_items((('state fields of the selected mode (e_intensity, sz: 2 x float32)',
                                       2 * FLOAT32_BYTES * n_cells),),
                                     host_items=host_items, run_items=run_items, transient_items=transient_items,
                                     external_items=external_items, external_extra=external_extra)


def _modes_fem_memory_estimate(settings_values, mesh, context):
    """em_modes on a triangles mesh (solvers/modes_fem, fem_modes): 2 float32 per-element state fields; host SuperLU
    (or a per-solve cuDSS LDL^T: external), blocks, Arnoldi basis, sample bases, mode candidates, display fields."""
    from ...solvers.device_matrices import device_matrix_bytes, device_product_bytes
    from .fem_modes import (DISCRETIZATION_ORDERS, FEM_ENTRIES_PER_ROW, SAMPLE_POINTS, cudss_operator_bytes,
                            estimated_unknowns, fem_lu_bytes, operator_bytes, sample_basis_bytes)
    order = DISCRETIZATION_ORDERS[settings_values.get('discretization', 'quadratic')]
    n_elements = int(mesh.n_elements)
    unknowns = estimated_unknowns(n_elements, int(mesh.n_edges), int(mesh.n_nodes), order)
    n_modes = int(settings_values.get('n_modes', 4))
    from .cross_section_modes import mode_eigen_count
    n_eigen = max(mode_eigen_count(n_modes, unknowns), 1)   # fem_modes.solve_fem_modes asks the wide 2 n + 10 too
    arnoldi_vectors = arnoldi_basis_size(n_eigen, unknowns, eigen_subspace_size(context.solvers))
    # candidates of one solve are built as modes before the PML / rank selection: coefficients + 4 sample fields
    mode_bytes = COMPLEX128_BYTES * (unknowns + 4 * SAMPLE_POINTS * n_elements)
    factorization = eigen_factorization(context.solvers)
    on_device = factorization in CUDSS_FACTORIZATIONS
    factor_items = [] if on_device else host_factor_items(   # MUMPS / PARDISO: LDL^T of the complex symmetric pencil
        factorization, 'SuperLU L + U factors (FEM fill, heuristic)', fem_lu_bytes(unknowns), unknowns,
        FEM_ENTRIES_PER_ROW[order] * unknowns, symmetric=True, eigen=True)
    mixed = eigen_mixed(factorization, unknowns)
    # fem_modes.solve_fem_modes: an own LDL^T per solve (no plan reuse), dropped after it, and device_factor_matvec's B
    # (complex symmetric: strict lower triangle + diagonal; real without PML ring and loss: float64; the pencil's pattern
    # an upper figure of B's)
    real_mass = real_cross_section_operator(settings_values, context)
    transient_items = ([(EIGEN_PHASE, 'cuDSS operator copies and solve vectors',
                         cudss_operator_bytes(unknowns, order, mixed)),
                        (EIGEN_PHASE, f'device mass B ({"float64" if real_mass else "complex128"} lower triangle + '
                                      'diagonal)',
                         device_matrix_bytes(unknowns, FEM_ENTRIES_PER_ROW[order] * unknowns, symmetric=True,
                                             real=real_mass)),
                        (EIGEN_PHASE, 'B x of one application (product temporaries)',
                         device_product_bytes(unknowns, symmetric=True, real=real_mass))] if on_device else [])
    external_items, external_extra = (cudss_factor_items('the shifted mode operator', unknowns,
                                                         FEM_ENTRIES_PER_ROW[order] * unknowns, mixed=mixed)
                                      if on_device else ([], 0))
    host_items = factor_items + [
        ('assembled blocks and shifted operator (sparse complex128)', operator_bytes(unknowns, order)),
        ('quadrature-point bases', sample_basis_bytes(n_elements, order)),
        (f'mode candidates ({n_eigen})', n_eigen * mode_bytes),
        ('nodal display fields (10 x float64)',
         10 * FLOAT64_BYTES * (int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)))]
    # a polished mixed LDL^T (cudss_mixed, mumps_mixed / pardiso_mixed: from MIXED_EIGEN_MIN_UNKNOWNS on, below it
    # they factor in double) keeps a complex64 device basis (fem_modes: eigs single_basis)
    polished = int(unknowns) >= MIXED_EIGEN_MIN_UNKNOWNS and (
        factorization == 'cudss_mixed' or (factorization in HOST_DIRECT_FACTORIZATIONS
                                           and host_direct_precision(factorization) == 'mixed'))
    _arnoldi_item(host_items, transient_items, unknowns, arnoldi_vectors, context, on_device, single_basis=polished)
    return MemoryEstimate.from_items((('mode state fields (2 x float32 per element)', 2 * FLOAT32_BYTES * n_elements),),
                                     host_items=host_items, transient_items=transient_items,
                                     external_items=external_items, external_extra=external_extra)


# ---- features ----
def dft_monitor_memory_estimate(polarization):
    """FeatureType.memory_estimate of an FDTD dft_monitor: complex64 (n_wavelengths, box / downsample^2) spectra
    (monitors.DFTMonitor; the stride division assumes a box at least downsample cells wide on each axis)."""
    n_spectra = _DFT_SPECTRA[polarization]

    def estimate(values, n_selected_cells, mesh):
        n_wavelengths = int(values['n_wavelengths'])
        recorded_cells = -(-int(n_selected_cells) // int(values['downsample']) ** 2)
        return MemoryEstimate.from_items(((f'DFT spectra ({n_spectra} x {n_wavelengths} wavelengths x complex64)',
                                           n_spectra * n_wavelengths * COMPLEX64_BYTES * recorded_cells),))
    return estimate


def line_spectra_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of an FDTD flux / mode monitor: E and H complex128 line spectra (monitors._LineSpectra)."""
    n_wavelengths = int(values['n_wavelengths'])
    return MemoryEstimate.from_items(((f'line spectra (2 x {n_wavelengths} wavelengths x complex128)',
                                       2 * n_wavelengths * COMPLEX128_BYTES * int(n_selected_cells)),))


def far_field_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of an FDTD far_field: E and H complex128 spectra on the box outline
    (monitors.NearToFarField); the outline is taken as that of a square box of n_selected_cells."""
    n_wavelengths = int(values['n_wavelengths'])
    side = math.sqrt(max(int(n_selected_cells), 1))
    outline_samples = int(4 * (side + 1))
    return MemoryEstimate.from_items(((f'outline spectra (2 x {n_wavelengths} wavelengths x complex128, approximate)',
                                       2 * n_wavelengths * COMPLEX128_BYTES * outline_samples),))


def adjoint_objective_memory_estimate(polarization):
    """FeatureType.memory_estimate of an FDFD adjoint_objective (adjoint.eps_sensitivity): ADJOINT_PHASE items of the
    one transposed solve all objectives share (its own phase per feature, so the largest counts once)."""
    n_gradients, n_densities, n_temporaries = (1, 1, 6) if polarization == 'ez' else (2, 2, 18)
    n_mean_maps = 0 if polarization == 'ez' else 4   # adjoint._eta_cotangent: 2 entries per row each

    def estimate(values, n_selected_cells, mesh):
        n_cells = int(mesh.n_cells)
        # the float32 EpsSensitivity (gradients + densities) stays cached until the next solve starts (fdfd_2d._solve
        # drops it before the factorisation); small beside the temporaries
        items = [(ADJOINT_PHASE, f'adjoint sensitivity cache ({n_gradients + n_densities} x float32)',
                  (n_gradients + n_densities) * FLOAT32_BYTES * n_cells),
                 (ADJOINT_PHASE, f'adjoint solve temporaries ({n_temporaries} x complex128)',
                  n_temporaries * COMPLEX128_BYTES * n_cells)]
        if n_mean_maps:
            items.append((ADJOINT_PHASE, f'eps_xy chain-rule mean maps ({n_mean_maps} sparse complex128)',
                          n_mean_maps * (2 * COMPLEX128_CSR_ENTRY_BYTES + INT32_BYTES) * n_cells))
        return MemoryEstimate.from_items(transient_items=tuple(items))
    return estimate


def design_region_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of an FDFD design_region (design_density.DesignRegion): the cropped bool mask kept
    (bounding-box cells); the fresh full-grid domain_mask and the float32 box eps values as the extra."""
    box_cells = int(n_selected_cells)
    return MemoryEstimate.from_items((('design region mask (bool, box)', BOOL_BYTES * box_cells),),
                                     build_extra=BOOL_BYTES * int(mesh.n_cells) + FLOAT32_BYTES * box_cells)


def pec_memory_estimate(keep_arrays):
    """FeatureType.memory_estimate of a pec_domain: `keep_arrays` bool keep masks of the engine on the PEC box (FDTD ez
    1, hz 2, FDFD / em_modes 0; taken as the selected cells, the box holds at least those) plus the bool full-grid
    union PEC mask; counted per feature, allocated by the first only."""
    def estimate(values, n_selected_cells, mesh):
        n_cells = int(mesh.n_cells)
        return MemoryEstimate.from_items(
            (('PEC masks', BOOL_BYTES * (keep_arrays * int(n_selected_cells) + n_cells)),),
            build_extra=(1 + keep_arrays) * BOOL_BYTES * n_cells)   # the fresh domain_mask (+ hz face_pec masks)
    return estimate


def conical_memory_estimate(settings_values, mesh, context):
    """PhysicsType.memory_estimate of em_fdfd_conical (solvers/fdfd_conical, conical): 16 float32 centre fields, a
    solve's material temporaries; host SuperLU of 3 unknowns per cell (or a cuDSS LDL^T: external), operators, materials,
    eigen Arnoldi work; a triangles mesh: _conical_fem_memory_estimate."""
    if getattr(mesh, 'kind', 'tensor') == 'triangles':
        return _conical_fem_memory_estimate(settings_values, mesh, context)
    n_cells = int(mesh.n_cells)
    unknowns = 3 * n_cells
    eigen = eigen_solved(context.spec, context.physics_spec.name)
    factorization = host_factorization(context.solvers, eigen)
    on_device = factorization in CUDSS_FACTORIZATIONS
    # MUMPS / PARDISO: the LDL^T of D A (an LU with a Bloch phase) at the SuperLU heuristic x superlu_bytes_scale
    host_items = ([] if on_device else
                  host_factor_items(factorization, 'SuperLU L + U factors (heuristic, x2 for the curl-curl fill)',
                                    2 * host_lu_bytes(unknowns), unknowns, 13 * unknowns,
                                    symmetric=not bloch_active_axes(settings_values), eigen=eigen))
    if eigen:   # the resonance search's own factor
        host_items = solver_job_items(EIGEN_ROLE, host_items)
    host_items += [('curl E, curl H, curl-curl and system matrix (sparse complex128, rough)',
                    30 * COMPLEX128_CSR_ENTRY_BYTES * unknowns),
                   ('cross-section operator parts of the assembly (sparse complex128, rough)',
                    36 * COMPLEX128_CSR_ENTRY_BYTES * n_cells),
                   ('kept materials (eps_xx, eps_yy, eps_zz, mu complex128, pec), Yee fields and centre phasors',
                    (4 + 6 + 6) * COMPLEX128_BYTES * n_cells + n_cells),
                   ('solution and rhs (complex128)', 2 * COMPLEX128_BYTES * unknowns)]
    transient_items = [(SOLVE_PHASE, 'material temporaries per solve (one eps component at a time, mu: 3 x '
                                     'complex128)', 3 * COMPLEX128_BYTES * n_cells)]
    if eigen:   # conical.solve_conical_resonances
        n_resonances = int(_study_values(context).get('n_resonances') or 1)
        from ...solvers.arpack_shift_invert import LDLT_FACTORIZATIONS
        from .cross_section_modes import cpml_cross_section
        ldlt = not bloch_active_axes(settings_values)   # an LDL^T of D A unless a Bloch phase makes it an LU
        search_host, search_transient = _resonance_search(   # an own LDL^T of a complex (open) A: polished (floored)
            context, n_resonances, unknowns, on_device, 13 * unknowns, ldlt=ldlt,
            polished_double=ldlt and factorization in LDLT_FACTORIZATIONS and cpml_cross_section(settings_values))
        transient_items += search_transient
        host_items += [*search_host,
                       (f'resonance fields ({n_resonances} x 6 components complex128)',
                        6 * n_resonances * COMPLEX128_BYTES * n_cells),
                       ('shifted / symmetrized operator copies (sparse complex128, rough)',
                        2 * 13 * COMPLEX128_CSR_ENTRY_BYTES * unknowns)]
    run_items, external_items, external_extra = [], [], 0
    if on_device:   # cuDSS LDL^T (~13 entries per row): the lower-triangle CSR copies, vectors
        mixed = eigen_mixed(factorization, unknowns) if eigen else factorization == 'cudss_mixed'
        external_items, external_extra = cudss_factor_items('the conical curl-curl operator', unknowns, 13 * unknowns,
                                                            mixed=mixed)
        if eigen:   # the resonance search's own factor
            transient_items += own_cudss_search_items('the conical curl-curl operator', unknowns, 13 * unknowns,
                                                      mixed)
            external_items = solver_job_items(EIGEN_ROLE, external_items)
        else:       # host_fdfd keeps the driven factor (_cudss_previous) and refactorises it in place
            kept_cudss_items(run_items, transient_items, unknowns, 13 * unknowns, mixed, later_factorisations(context))
    return MemoryEstimate.from_items((('centre fields (16 x float32)', 16 * FLOAT32_BYTES * n_cells),),
                                     host_items=host_items, run_items=run_items, transient_items=transient_items,
                                     external_items=external_items, external_extra=external_extra)


def _conical_fem_memory_estimate(settings_values, mesh, context):
    """em_fdfd_conical on a triangles mesh (solvers/fdfd_conical_fem, fem_conical; eigenfrequency):
    _fem_resonance_memory_estimate of one search (the resonance search's own cuDSS factor) with the quadrature-point
    bases."""
    return _fem_resonance_memory_estimate(settings_values, mesh, context,
                                          int(_study_values(context).get('n_resonances') or 1), None,
                                          'assembled blocks K / M / W and the restricted operator (sparse, rough)',
                                          'the conical FEM operator', sample_bases=True)


def _fem_resonance_memory_estimate(settings_values, mesh, context, n_resonances, n_orders, blocks_label,
                                   operator_label, basis_per_wanted=0, sample_bases=False):
    """Eigen-solved FEM type on a triangles mesh (fem_modes unknowns; em_fdfd_conical, em_fdfd_bor): 2 float32 state
    fields per element; host SuperLU (or a cuDSS LDL^T: external), blocks, Arnoldi work, resonance vectors (n_orders
    x n_resonances; None: one search, unlabelled), the quadrature-point bases (sample_bases), display fields; one
    order's search and its own factor at a time."""
    from .fem_modes import (DISCRETIZATION_ORDERS, FEM_ENTRIES_PER_ROW, estimated_unknowns, fem_lu_bytes,
                            operator_bytes, sample_basis_bytes)
    order = DISCRETIZATION_ORDERS[settings_values.get('discretization', 'quadratic')]
    n_elements = int(mesh.n_elements)
    unknowns = estimated_unknowns(n_elements, int(mesh.n_edges), int(mesh.n_nodes), order)
    factorization = eigen_factorization(context.solvers)
    on_device = factorization in CUDSS_FACTORIZATIONS
    fem_entries = FEM_ENTRIES_PER_ROW[order] * unknowns
    search_host, search_transient = _resonance_search(   # the filter's A, B and W: one operator's entries each
        context, n_resonances, unknowns, on_device, fem_entries, basis_per_wanted, sparse_mass_entries=fem_entries,
        symmetric_search=True)
    vectors_label, n_vectors = (f'{n_resonances}', n_resonances) if n_orders is None else (
        f'{n_orders} x {n_resonances}', n_orders * n_resonances)
    host_items = ([] if on_device else solver_job_items(EIGEN_ROLE, host_factor_items(
        factorization, 'SuperLU L + U factors (FEM fill, heuristic)', fem_lu_bytes(unknowns), unknowns,
        FEM_ENTRIES_PER_ROW[order] * unknowns, symmetric=True, eigen=True))) + [
        (blocks_label, 2 * operator_bytes(unknowns, order)),
        *search_host,
        (f'resonance vectors ({vectors_label} x complex128)', n_vectors * COMPLEX128_BYTES * unknowns),
        *([('quadrature-point bases', sample_basis_bytes(n_elements, order))] if sample_bases else []),
        ('nodal display fields (6 complex128 + 16 float64)',
         28 * FLOAT64_BYTES * (int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)))]
    transient_items = [(SOLVE_PHASE, 'material temporaries per solve (one eps component at a time, mu: 3 x '
                                     'complex128 per element)', 3 * COMPLEX128_BYTES * n_elements), *search_transient]
    external_items, external_extra = [], 0
    if on_device:
        mixed = eigen_mixed(factorization, unknowns)
        transient_items += own_cudss_search_items(operator_label, unknowns, fem_entries, mixed)
        external_items, external_extra = cudss_factor_items(operator_label, unknowns, fem_entries, mixed=mixed)
        external_items = solver_job_items(EIGEN_ROLE, external_items)
    return MemoryEstimate.from_items((('resonance state fields (2 x float32 per element)',
                                       2 * FLOAT32_BYTES * n_elements),), host_items=host_items,
                                     transient_items=transient_items, external_items=external_items,
                                     external_extra=external_extra)


def bor_memory_estimate(settings_values, mesh, context):
    """PhysicsType.memory_estimate of em_fdfd_bor (solvers/fdfd_bor, body_of_revolution): 15 float32 centre fields, a
    solve's material temporaries; host SuperLU of 3 unknowns per cell (or a cuDSS LDL^T: external, or line_block
    factors), operators, vectors, eigen Arnoldi work, order-sum terms; a triangles mesh: _bor_fem_memory_estimate."""
    if getattr(mesh, 'kind', 'tensor') == 'triangles':
        return _bor_fem_memory_estimate(settings_values, mesh, context)
    n_cells = int(mesh.n_cells)
    unknowns = 3 * n_cells
    study_type = context.spec.study.type
    eigen = study_type == EIGENFREQUENCY_STUDY
    shifted = (context.solvers or {}).get('shifted')
    line_block = shifted is not None and shifted.solver_type.name == 'line_block'
    factorization = host_factorization(context.solvers, eigen)
    on_device = not line_block and factorization in CUDSS_FACTORIZATIONS
    orders = [abs(int(order)) for order in settings_values.get('azimuthal_orders') or ()] or [
        abs(int(settings_values.get('azimuthal_order') or 0))]
    entries = int(math.ceil(_bor_entries_per_row(max(orders)) * unknowns))   # the largest order's pattern
    # MUMPS / PARDISO (not under a line_block shifted solver): the LDL^T of the complex symmetric operator
    host_items = ([] if on_device else
                  host_factor_items(factorization if not line_block else 'superlu',
                                    'SuperLU L + U factors (heuristic, x2 for the curl-curl fill)',
                                    2 * host_lu_bytes(unknowns), unknowns, entries, symmetric=True, eigen=eigen))
    if eigen and not line_block:   # the resonance search's own factor (line_block: the shifted job's fallback)
        host_items = solver_job_items(EIGEN_ROLE, host_items)
    host_items += [('curl, operator and assembly transients (sparse complex128, rough)',
                    30 * COMPLEX128_CSR_ENTRY_BYTES * unknowns),
                   ('kept BORAssembly (curl split C0 / C1, K0 / K1 / K2 of the axis class; sparse complex128, rough)',
                    40 * COMPLEX128_CSR_ENTRY_BYTES * unknowns),
                   ('solution, rhs and centre phasors (complex128)', (4 * 3 + 6) * COMPLEX128_BYTES * n_cells),
                   ('kept materials (eps_rr, eps_phiphi, eps_zz, mu complex128, pec) and Yee fields of the driven solve',
                    (4 + 6) * COMPLEX128_BYTES * n_cells + n_cells)]
    n_listed = len(settings_values.get('azimuthal_orders') or ())
    order_sum = bool(n_listed) and study_type in ('stationary', FREQUENCY_DOMAIN_STUDY)   # the driven order sum
    n_orders = max(n_listed, 1) if eigen or order_sum else 1
    if order_sum:   # solvers/fdfd_bor._solve_order_sum: one order's factor at a time (the item above)
        host_items.append(('driven order sum: Yee fields of the +m / -m terms and their phi = 0 sum (complex128), '
                           'phi-mean |E|^2 (float64)', 3 * 6 * COMPLEX128_BYTES * n_cells + 3 * 8 * n_cells))
    transient_items = [(SOLVE_PHASE, 'material temporaries per solve (one eps component at a time, mu: 3 x '
                                     'complex128)', 3 * COMPLEX128_BYTES * n_cells)]
    if eigen:   # body_of_revolution.solve_bor_resonances, one per order of azimuthal_orders
        n_resonances = int(_study_values(context).get('n_resonances') or 1)
        search_host, search_transient = _resonance_search(   # one order's search at a time (symmetric=True)
            context, n_resonances, unknowns, on_device, entries, symmetric_search=True)
        transient_items += search_transient
        host_items += [*search_host,
                       (f'resonance fields ({n_orders} x {n_resonances} x complex128)',
                        n_orders * n_resonances * COMPLEX128_BYTES * unknowns)]
    run_items, external_items, external_extra = [], [], 0
    if on_device:   # cuDSS LDL^T (_bor_entries_per_row): the lower-triangle CSR copies, vectors
        mixed = eigen_mixed(factorization, unknowns) if eigen else factorization == 'cudss_mixed'
        fill = FactorFill(_bor_factor_entries(int(mesh.nx), int(mesh.ny), bool(max(orders))))
        external_items, external_extra = cudss_factor_items('the BOR curl-curl operator', unknowns, entries,
                                                            dimensions=fill, mixed=mixed)
        if eigen:   # each order's resonance search factors on its own
            transient_items += own_cudss_search_items('the BOR curl-curl operator', unknowns, entries, mixed)
            external_items = solver_job_items(EIGEN_ROLE, external_items)
        else:       # host_fdfd keeps the driven factor and refactorises it at the next order / wavelength
            kept_cudss_items(run_items, transient_items, unknowns, entries, mixed,
                             later_factorisations(context) or (order_sum and n_listed > 1))
    if line_block and (eigen or order_sum):   # bor_line_block.order_factorizations: the bound order family, its factors
        from .bor_line_block import line_block_bytes
        from ...solvers.line_block import MAX_CHUNK_SHIFTS
        operator, factors = line_block_bytes((int(mesh.nx), int(mesh.ny)), shifted.values['precision'],
                                             min(n_orders, MAX_CHUNK_SHIFTS))
        run_items += [('line_block operator caches (K0 - k0^2 M, K1, K2)', operator),
                      (f'line_block kept factors ({min(n_orders, MAX_CHUNK_SHIFTS)} orders, chunked to the budget)',
                       factors)]
    elif line_block:   # fdfd_bor._solve_line_block: whole operators, no order family (K1 / K2 caches) bound
        from .bor_line_block import driven_line_block_bytes
        from .fdfd_line_block import MAX_BATCH_WAVELENGTHS
        n_operators = max(min(_harmonic_wavelength_count(context), MAX_BATCH_WAVELENGTHS), 1)
        label = (f'line_block wavelength batch ({n_operators} operators: block caches, residual CSRs, factors; sized '
                 'to the free memory at run time)' if n_operators > 1 else 'line_block bound operator and its factor')
        run_items.append((label, driven_line_block_bytes((int(mesh.nx), int(mesh.ny)), shifted.values['precision'],
                                                         n_operators, entries)))
    return MemoryEstimate.from_items((('centre fields (15 x float32)', 15 * FLOAT32_BYTES * n_cells),),
                                     host_items=host_items, run_items=run_items, transient_items=transient_items,
                                     external_items=external_items, external_extra=external_extra)


def _bor_fem_memory_estimate(settings_values, mesh, context):
    """em_fdfd_bor on a triangles mesh (solvers/fdfd_bor_fem, fem_bor; eigenfrequency): _fem_resonance_memory_estimate
    per listed order (one order's resonance search and its own cuDSS factor at a time)."""
    from .fem_bor import FEM_BOR_ARNOLDI_BASIS_PER_WANTED
    values = _study_values(context)   # eigenfrequency n_resonances, optomechanics n_optical_resonances
    return _fem_resonance_memory_estimate(
        settings_values, mesh, context, int(values.get('n_resonances') or values.get('n_optical_resonances') or 1),
        max(len(settings_values.get('azimuthal_orders') or ()), 1),
        'assembled blocks K0 / K1 / K2 / M / W and the restricted operator (sparse, rough)', 'the BOR FEM operator',
        basis_per_wanted=FEM_BOR_ARNOLDI_BASIS_PER_WANTED)


def _fdtd_bor_members(settings_values, physics_spec):
    """(orders, steppers) of em_fdtd_bor: one stepper, or per listed order of a driven order sum one per source family
    (fdtd_bor.order_sum_families of the enabled point_source components)."""
    from .fdtd_bor import order_sum_families
    orders = [abs(int(order)) for order in settings_values.get('azimuthal_orders') or ()]
    if not orders:
        return [abs(int(settings_values.get('azimuthal_order') or 0))], 1
    components = {(feature.params or {}).get('component', 'z') for feature in physics_spec.features
                  if feature.enabled and feature.type == 'point_source'}
    return orders, sum(len(order_sum_families(order, components)) for order in orders)


def fdtd_bor_memory_estimate(settings_values, mesh, context):
    """PhysicsType.memory_estimate of em_fdtd_bor (fdtd_bor.BORTimeStepper per order-sum member): float32 fields,
    coefficients, z CPML psi, r_hi layer rows and CSR curl products (fdtd_bor.stepper_bytes), pole buffers on the E
    points of their cells, the other members' monitor spectra, point traces, the harmonic accumulator (run item:
    complex64 of every E and H point per study wavelength and member); on the host the kept real-metric assembly."""
    from .fdtd_bor import layout_sizes, stepper_bytes
    shape = (int(mesh.nx), int(mesh.ny))
    orders, n_members = _fdtd_bor_members(settings_values, context.physics_spec)
    order = max(orders)
    npml = int(settings_values.get('pml_cells') or 0)
    z_pml_rows = 0   # E + a rows of the z CPML layers: ~4 per layer cell
    for face in ('y_lo', 'y_hi'):
        if settings_values.get(f'bc_{face}') == 'cpml':
            z_pml_rows += 4 * npml * (shape[0] + 1)
    radial_rows = 6 * npml * (shape[1] + 1) if settings_values.get('bc_x_hi') == 'cpml' else 0   # ~6 per layer cell
    device, host = stepper_bytes(shape, order, z_pml_rows, radial_rows)
    items = [(f'stepper fields, coefficients, CPML psi and curl products ({n_members} stepper(s))', n_members * device)]
    pole_bytes = mask_bytes = 0   # per pole ~3 E points per box cell: J (+ dP), drive, correction float32, int32 index
    for box_cells, n_drude, n_lorentz in _pole_regions(context):
        pole_bytes += 3 * box_cells * (n_drude * (3 * FLOAT32_BYTES + INT32_BYTES)
                                       + n_lorentz * (4 * FLOAT32_BYTES + INT32_BYTES))
        mask_bytes += (n_drude + n_lorentz) * BOOL_BYTES * box_cells   # dispersive_regions cropped masks
    if pole_bytes:
        items.append((f'dispersive poles ({n_members} stepper(s), E points of their cells; region masks)',
                      n_members * pole_bytes + mask_bytes))
    if n_members > 1:   # the flux / far-field feature estimates count one member's spectra
        monitor_bytes = 0
        for feature in context.physics_spec.features:
            if feature.enabled and feature.type in ('flux_monitor', 'far_field'):
                points = (4 if feature.type == 'flux_monitor' else 12) * (shape[0] + shape[1] + 2)
                count = (feature.params or {}).get('n_wavelengths', 100)
                count = int(count) if isinstance(count, (int, float)) else 100   # an expression: the flux default
                monitor_bytes += points * count * COMPLEX128_BYTES
        items.append((f'monitor spectra of the other {n_members - 1} order-sum steppers (rough)',
                      (n_members - 1) * monitor_bytes))
    # point traces: the BOR step is ~ the 2D one x sqrt(2) / (|m| + 1) for |m| >= 1 (the m / r terms near the axis)
    dt = mesh.dt * min(1.0, math.sqrt(2.0) / (order + 1)) if mesh.dt else None
    trace_items, run_items, host_items = point_trace_items(context, mesh, _POINT_TRACE_FEATURES, dt=dt)
    n_wavelengths = _harmonic_wavelength_count(context)
    if n_wavelengths:   # solvers/fdtd_bor.BORHarmonic
        n_points = sum(sum(sizes) for sizes in layout_sizes(shape))
        run_items.append((f'harmonic accumulator ({n_wavelengths} wavelengths x complex64, every E and H point, '
                          f'{n_members} stepper(s))', n_members * n_wavelengths * COMPLEX64_BYTES * n_points))
    return MemoryEstimate.from_items(items + trace_items, host_items=[('kept real-metric assembly (rough)', host)]
                                     + host_items, run_items=run_items)


def fdtd_bor_flux_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of an em_fdtd_bor flux_monitor: complex128 spectra of its ~4 (n + 1) E / H points."""
    n_wavelengths = int(values['n_wavelengths'])
    return MemoryEstimate.from_items(((f'line spectra ({n_wavelengths} wavelengths x complex128)',
                                       4 * (int(n_selected_cells) + 1) * n_wavelengths * COMPLEX128_BYTES),))


def fdtd_bor_mode_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of an em_fdtd_bor mode_monitor: complex128 spectra of the E_r / u / E_z points of
    its two slices (~6 (n + 1) points)."""
    n_wavelengths = int(values['n_wavelengths'])
    return MemoryEstimate.from_items(((f'slice spectra ({n_wavelengths} wavelengths x complex128)',
                                       6 * (int(n_selected_cells) + 1) * n_wavelengths * COMPLEX128_BYTES),))


def fdtd_bor_far_field_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of an em_fdtd_bor far_field: complex128 spectra of the E / H bands of its faces
    (whole grid rows / columns, ~6 points per cell and face, at most two z and two r faces)."""
    n_wavelengths = int(values['n_wavelengths'])
    n_points = 12 * (int(mesh.nx) + int(mesh.ny) + 2)
    return MemoryEstimate.from_items(((f'face band spectra ({n_wavelengths} wavelengths x complex128)',
                                       n_points * n_wavelengths * COMPLEX128_BYTES),))

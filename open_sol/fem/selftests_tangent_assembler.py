"""Self-test of the fixed-pattern tangent assemblers (fem.elasticity3d.TangentAssembler3D, fem.elasticity.TangentAssembler;
CPU, check({})): internal force, full and free-free tangent against the one-off COO assembly and its slicing, patterns
constant across displacements and equal to the structural pattern of the element blocks. error = failed sub-checks.
GPU twin (plan G5): the device assembler and the device-value cuDSS refactorisation against the host path."""
import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import checks_result

_LENGTH = 1e-6                      # box / rectangle side scale [m]
_RTOL = 1e-13                       # same formulas, other summation order (first estimate)
_DISPLACEMENT = 1e-2 * _LENGTH      # random displacement amplitude: ~1 % strains, Green-Lagrange terms visible
_INITIAL_STRESS = 1e8               # Pa, random prescribed and thermal S0 per element
_FREE_FRACTION = 0.8


def _relative_vector(value, reference) -> float:
    scale = float(np.max(np.abs(reference)))
    return float(np.max(np.abs(value - reference))) / scale if scale > 0.0 else float(np.max(np.abs(value)))


def _relative_matrix(value, reference) -> float:
    scale = float(abs(reference).max())
    difference = (value - reference).tocsr()
    largest = float(abs(difference).max()) if difference.nnz else 0.0
    return largest / scale if scale > 0.0 else largest


def _structural_pattern(n, group_dofs):
    """(indptr, indices) of every element-block entry (values 1, nothing dropped), canonical."""
    import scipy.sparse
    rows = np.concatenate([np.broadcast_to(dofs[:, :, None], dofs.shape + dofs.shape[1:]).ravel()
                           for dofs in group_dofs])
    columns = np.concatenate([np.broadcast_to(dofs[:, None, :], dofs.shape[:1] + dofs.shape[1:] * 2).ravel()
                              for dofs in group_dofs])
    pattern = scipy.sparse.coo_matrix((np.ones(rows.size), (rows, columns)), shape=(n, n)).tocsr()
    pattern.sum_duplicates()
    return pattern.indptr, pattern.indices


def _compare(add, name, assembler, reference_pair, displacement, initial_stress, free, group_dofs, zero):
    """Sub-checks of one assembler against internal_force_and_tangent's (force, CSR) at the displacement."""
    reference_force, reference_tangent = reference_pair
    force, data = assembler.force_and_tangent(displacement, initial_stress=initial_stress)
    full, reduced = assembler.full(data), assembler.reduced(data)
    free_index = np.flatnonzero(free)
    reference_tangent = reference_tangent.tocsr()
    add(f'{name} force', _relative_vector(force, reference_force))
    add(f'{name} force alone', _relative_vector(assembler.force(displacement, initial_stress=initial_stress),
                                                reference_force))
    add(f'{name} full tangent', _relative_matrix(full, reference_tangent))
    add(f'{name} reduced tangent', _relative_matrix(reduced, reference_tangent[free_index][:, free_index].tocsr()))
    rows = np.repeat(np.arange(reduced.shape[0]), np.diff(reduced.indptr))
    same_row = np.diff(rows) == 0   # canonical: column indices strictly increasing within every row
    canonical = bool(np.all(np.diff(reduced.indices.astype(np.int64))[same_row] > 0))
    add(f'{name} reduced canonical', 0.0 if canonical else 1.0, 0.0)
    indptr, indices = _structural_pattern(assembler.n_vector, group_dofs)
    structural = np.array_equal(indptr, full.indptr) and np.array_equal(indices, full.indices)
    add(f'{name} structural pattern', 0.0 if structural else 1.0, 0.0)
    _, zero_data = assembler.force_and_tangent(zero, initial_stress=None)
    same = (zero_data.shape == data.shape and np.array_equal(assembler.full(zero_data).indptr, full.indptr)
            and np.array_equal(assembler.reduced(zero_data).indices, reduced.indices))
    add(f'{name} pattern at u = 0', 0.0 if same else 1.0, 0.0)
    return int(np.count_nonzero(zero_data == 0.0))


def _check_tangent_assembler(_):
    from .dofs import lagrange_dofs
    from .dofs3d import lagrange_dofs_3d
    from .elasticity import StressField, TangentAssembler, internal_force_and_tangent, vector_dofs
    from .elasticity3d import TangentAssembler3D, internal_force_and_tangent_3d, vector_dofs_3d
    from .mesh3d import structured_box_mesh
    from .selftests_harmonic import _rectangle_mesh
    from .tangent_kernel3d import ELEMENT_TANGENT_3D
    checks, details = {}, {}

    def add(name, value, tolerance=_RTOL):
        checks[name] = {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}

    rng = np.random.default_rng(11)
    mesh3 = structured_box_mesh(_LENGTH * np.array([0.0, 0.3, 0.75, 1.2]), _LENGTH * np.array([0.0, 0.45, 0.8]),
                                _LENGTH * np.array([0.0, 0.25, 0.7]))
    factor = rng.standard_normal((6, 6))
    material = np.broadcast_to(1e10 * (factor @ factor.T + 6.0 * np.eye(6)), (mesh3.n_elements, 6, 6)).copy()
    material[: mesh3.n_elements // 3] *= 0.5   # two materials
    solid = np.ones(mesh3.n_elements, dtype=bool)
    solid[-3:] = False                         # a few void elements: their dofs carry no block
    stress3 = StressField(prescribed=_INITIAL_STRESS * rng.standard_normal((mesh3.n_elements, 6)),
                          thermal=_INITIAL_STRESS * rng.standard_normal((mesh3.n_elements, 6)))
    for order in (1, 2):
        dofs = lagrange_dofs_3d(mesh3, order)
        displacement = _DISPLACEMENT * rng.standard_normal((dofs.n_dofs, 3))
        free = rng.random(3 * dofs.n_dofs) < _FREE_FRACTION
        reference = internal_force_and_tangent_3d(dofs, displacement, material, elements=solid,
                                                  initial_stress=stress3)
        assembler = TangentAssembler3D(dofs, material, elements=solid, free=free)
        group_dofs = [vector_dofs_3d(dofs.element_dofs[chunk]) for chunk in assembler.chunks]
        zeros = _compare(add, f'3D P{order}', assembler, reference, displacement, stress3, free, group_dofs,
                         np.zeros_like(displacement))
        uncached = TangentAssembler3D(dofs, material, elements=solid, free=free, cache_bytes=0)
        _, data = assembler.force_and_tangent(displacement, initial_stress=stress3)
        _, data_uncached = uncached.force_and_tangent(displacement, initial_stress=stress3)
        add(f'3D P{order} gradients recomputed', _relative_vector(data_uncached, data))
        # fixed_pattern.kept_pattern: same mesh, elements, chunks and free mask -> the same pattern object
        add(f'3D P{order} pattern kept per mesh', 0.0 if uncached.pattern is assembler.pattern else 1.0, 0.0)
        batched = TangentAssembler3D(dofs, material, elements=solid, free=free, element_kernel=False)
        batched_force, batched_data = batched.force_and_tangent(displacement, initial_stress=stress3)
        kernel_force = assembler.force_and_tangent(displacement, initial_stress=stress3)[0]
        add(f'3D P{order} element kernel vs batched force', _relative_vector(kernel_force, batched_force))
        add(f'3D P{order} element kernel vs batched tangent', _relative_vector(data, batched_data))
        details[f'3D P{order}'] = {'nnz': assembler.pattern.nnz, 'free': int(free.sum()), 'zero entries at u = 0': zeros,
                                   'cached_bytes': assembler.cached_bytes,
                                   'gradients_cached': (assembler.gradients_cached, uncached.gradients_cached),
                                   'element_kernel': assembler.element_kernel,
                                   'host_twin': ELEMENT_TANGENT_3D.host_kind}
    # 2D: P2 triangles, planar monoclinic and axisymmetric (r from 0.2 L: off the axis), an initial stress
    coefficients = {'c11': 2.5e11, 'c12': 1.0e11, 'c22': 2.2e11, 'c66': 0.7e11, 'c16': 0.1e11, 'c26': -0.05e11,
                    'c13': 0.9e11, 'c23': 0.95e11, 'c33': 2.4e11}
    mesh2 = _rectangle_mesh(_LENGTH * np.array([0.2, 0.5, 0.75, 1.2]), _LENGTH * np.array([0.0, 0.45, 0.8, 1.0]))
    for axisymmetric in (False, True):
        rows = 4 if axisymmetric else 3
        name = '2D axisymmetric' if axisymmetric else '2D planar'
        elements = np.ones(mesh2.n_elements, dtype=bool)
        elements[0] = False
        dofs = lagrange_dofs(mesh2, 2, elements=elements)
        displacement = _DISPLACEMENT * rng.standard_normal((dofs.n_dofs, 2))
        free = rng.random(2 * dofs.n_dofs) < _FREE_FRACTION
        stress2 = StressField(prescribed=_INITIAL_STRESS * rng.standard_normal((mesh2.n_elements, rows)))
        element_coefficients = {key: value for key, value in coefficients.items()
                                if axisymmetric or key not in ('c13', 'c23', 'c33')}
        if axisymmetric:   # orthotropic: no c16 / c26
            element_coefficients = {key: value for key, value in element_coefficients.items()
                                    if key not in ('c16', 'c26')}
        reference = internal_force_and_tangent(dofs, displacement, element_coefficients, axisymmetric=axisymmetric,
                                               initial_stress=stress2)
        assembler = TangentAssembler(dofs, element_coefficients, axisymmetric=axisymmetric, free=free)
        group_dofs = [vector_dofs(dofs.element_dofs[group[0]]) for group in assembler.groups]
        zeros = _compare(add, name, assembler, reference, displacement, stress2, free, group_dofs,
                         np.zeros_like(displacement))
        details[name] = {'nnz': assembler.pattern.nnz, 'free': int(free.sum()), 'zero entries at u = 0': zeros}
    return checks_result(checks, **details)


# ---- fem3d_tangent_assembler_gpu (plan G5): the device TangentAssembler3D against the host one on the same box, and
# CudssFactorization.refactorize_device_values against a host-refreshed cuDSS factor and SuperLU
_DEVICE_RTOL = 1e-12   # same formulas, atomic (unordered) summation on the device (first estimate)
_SOLVE_RTOL = 1e-9     # solves of two factors of values equal to round-off: ~ cond x eps (first estimate)
_MIXED_SOLVE_RTOL = 1e-8


def _box_problem(rng):
    """(mesh, material (n, 6, 6), solid mask, StressField S0) of the fem3d_tangent_assembler box."""
    from .elasticity import StressField
    from .mesh3d import structured_box_mesh
    mesh = structured_box_mesh(_LENGTH * np.array([0.0, 0.3, 0.75, 1.2]), _LENGTH * np.array([0.0, 0.45, 0.8]),
                               _LENGTH * np.array([0.0, 0.25, 0.7]))
    factor = rng.standard_normal((6, 6))
    material = np.broadcast_to(1e10 * (factor @ factor.T + 6.0 * np.eye(6)), (mesh.n_elements, 6, 6)).copy()
    material[: mesh.n_elements // 3] *= 0.5
    solid = np.ones(mesh.n_elements, dtype=bool)
    solid[-3:] = False
    stress = StressField(prescribed=_INITIAL_STRESS * rng.standard_normal((mesh.n_elements, 6)),
                         thermal=_INITIAL_STRESS * rng.standard_normal((mesh.n_elements, 6)))
    return mesh, material, solid, stress


def _check_device_assembler(_):
    import scipy.sparse.linalg
    from ..core.backend import cp
    from ..solvers.cudss_factor import CudssFactorization, cudss_available
    from .dofs3d import lagrange_dofs_3d
    from .elasticity3d import TangentAssembler3D, element_strains_3d
    checks, details = {}, {}

    def add(name, value, tolerance=_DEVICE_RTOL):
        checks[name] = {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}

    rng = np.random.default_rng(13)
    mesh, material, solid, stress = _box_problem(rng)
    for order in (1, 2):
        dofs = lagrange_dofs_3d(mesh, order)
        free = rng.random(3 * dofs.n_dofs) < _FREE_FRACTION
        host = TangentAssembler3D(dofs, material, elements=solid, free=free)
        for label, cache_bytes in ((f'P{order}', None), (f'P{order} uncached', 0)):
            device = TangentAssembler3D(dofs, material, elements=solid, free=free, cache_bytes=cache_bytes, device=True)
            for step in range(2):   # the second call reuses the device S0 of the same StressField object
                displacement = _DISPLACEMENT * rng.standard_normal((dofs.n_dofs, 3))
                name = f'{label} call {step + 1}'
                force, data = host.force_and_tangent(displacement, initial_stress=stress)
                device_force, device_data = device.force_and_tangent(displacement, initial_stress=stress)
                add(f'{name} force', _relative_vector(device_force, force))
                add(f'{name} force alone', _relative_vector(device.force(displacement, initial_stress=stress), force))
                add(f'{name} tangent data', _relative_vector(cp.asnumpy(device_data), data))
                full, host_full = device.full(device_data).get(), host.full(data)
                same = np.array_equal(full.indptr, host_full.indptr) and np.array_equal(full.indices, host_full.indices)
                add(f'{name} full pattern', 0.0 if same else 1.0, 0.0)
                add(f'{name} reduced values', _relative_vector(cp.asnumpy(device.reduced_values(device_data)),
                                                               host.reduced(data).data))
                add(f'{name} reduced tangent', _relative_matrix(device.reduced(device_data).get(), host.reduced(data)))
                vector = rng.standard_normal(host.n_vector)
                add(f'{name} full product', _relative_vector(device.full_product(device_data, vector),
                                                             host_full @ vector))
            _, data = host.force_and_tangent(displacement, initial_stress=None)
            _, device_data = device.force_and_tangent(displacement, initial_stress=None)
            add(f'{label} no S0 after a cached one', _relative_vector(cp.asnumpy(device_data), data))
            add(f'{label} tangent()', _relative_matrix(device.tangent(displacement, initial_stress=stress),
                                                       host.tangent(displacement, initial_stress=stress)))
            details[label] = {'device_bytes': device.device_bytes, 'gradients_cached': device.gradients_cached,
                              'element_kernel': device.element_kernel, 'chunk_elements': device.chunk_elements}
        # the CUDA element kernel (the default above) against the batched cupy formulas
        batched = TangentAssembler3D(dofs, material, elements=solid, free=free, device=True, element_kernel=False)
        kernel = TangentAssembler3D(dofs, material, elements=solid, free=free, device=True)
        displacement = _DISPLACEMENT * rng.standard_normal((dofs.n_dofs, 3))
        kernel_force, kernel_data = kernel.force_and_tangent(displacement, initial_stress=stress)
        batched_force, batched_data = batched.force_and_tangent(displacement, initial_stress=stress)
        add(f'P{order} device element kernel vs batched force', _relative_vector(kernel_force, batched_force))
        add(f'P{order} device element kernel vs batched tangent', _relative_vector(cp.asnumpy(kernel_data),
                                                                                   cp.asnumpy(batched_data)))
        points = rng.dirichlet(np.ones(4), size=5)[:, 1:]   # element_strains_3d on cupy (the output twin)
        displacement = _DISPLACEMENT * rng.standard_normal((dofs.n_dofs, 3))
        for green_lagrange, kind in ((False, 'linear'), (True, 'Green-Lagrange')):
            values, strains = element_strains_3d(dofs, displacement, points, elements=solid,
                                                 green_lagrange=green_lagrange)
            device_values, device_strains = element_strains_3d(dofs, cp.asarray(displacement), points, elements=solid,
                                                               green_lagrange=green_lagrange)
            add(f'P{order} {kind} element strains on the device', _relative_vector(cp.asnumpy(device_strains), strains))
            add(f'P{order} {kind} element values on the device', _relative_vector(cp.asnumpy(device_values), values))
    if not cudss_available():
        details['factor'] = 'skipped: nvmath-python (cuDSS) is not installed'
    else:   # P2, free = 80 % of the dofs the solid touches: the reduced tangent is SPD
        dofs = lagrange_dofs_3d(mesh, 2)
        active = np.zeros(dofs.n_dofs, dtype=bool)
        active[dofs.element_dofs[solid].ravel()] = True
        free = (rng.random(3 * dofs.n_dofs) < _FREE_FRACTION) & np.repeat(active, 3)
        host = TangentAssembler3D(dofs, material, elements=solid, free=free)
        device = TangentAssembler3D(dofs, material, elements=solid, free=free, device=True)
        first, second = (_DISPLACEMENT * rng.standard_normal((dofs.n_dofs, 3)) for _ in range(2))
        reduced_first = host.reduced(host.force_and_tangent(first, initial_stress=stress)[1])
        reduced_second = host.reduced(host.force_and_tangent(second, initial_stress=stress)[1])
        device_values = device.reduced_values(device.force_and_tangent(second, initial_stress=stress)[1])
        pattern = host.pattern
        rhs = rng.standard_normal(pattern.n_free)
        superlu = scipy.sparse.linalg.spsolve(reduced_second.tocsc(), rhs)
        reference = CudssFactorization(reduced_second, real=True, positive_definite=True, symmetric_known=True)
        host_solution = reference.solve_host(rhs)
        for precision, tolerance in (('double', _SOLVE_RTOL), ('mixed', _MIXED_SOLVE_RTOL)):
            factor = CudssFactorization(reduced_first, real=True, positive_definite=True, precision=precision,
                                        symmetric_known=True)
            reused = factor.refactorize_device_values(device_values, pattern.reduced_indptr, pattern.reduced_indices)
            solution = factor.solve_host(rhs)
            add(f'{precision} factor: plan reused on the device', 0.0 if reused and factor.n_plans == 1 else 1.0, 0.0)
            add(f'{precision} factor: operator norm', abs(factor.operator_norm / reference.operator_norm - 1.0))
            add(f'{precision} factor: solve vs SuperLU', _relative_vector(solution, superlu), tolerance)
            if precision == 'double':
                add('double factor: solve vs the host-refreshed cuDSS factor', _relative_vector(solution, host_solution),
                    tolerance)
            details[f'{precision} factor'] = {'kind': factor.matrix_kind, 'plans': factor.n_plans,
                                              'factorizations': factor.n_factorizations,
                                              'probe_residual': factor.probe_residual}
            factor.release()
        reference.release()
        details['free_dofs'] = int(pattern.n_free)
    return checks_result(checks, **details)


# ---- fem3d_tangent_kernel_numba: numba vs numpy twin of fem.element_tangent_3d (tangent_kernel3d) on random inputs
_NUMBA_TOLERANCE = 1e-12      # same sums, other order
_NUMBA_ELEMENTS = 64          # compared elements per (order, S0) variant
_NUMBA_SPEED_ELEMENTS = 2048  # P2 elements of the timed call (numpy twin ~86 us and ~117 kB temporaries per element)
_NUMBA_SPEED_CALLS = (1, 5)   # (numpy, numba) timed calls


def _kernel_arguments(rng, elements, n_local, points, has_fixed, want_tangent):
    """The element_tangent_3d args tuple of random P1 / P2-sized inputs: gradients ~1 / L, ~1 % strains, SPD C, S0."""
    columns = 3 * n_local
    factor = rng.standard_normal((elements, 6, 6))
    material = 1e10 * (np.einsum('mij,mkj->mik', factor, factor) + 6.0 * np.eye(6))
    fixed = (_INITIAL_STRESS * rng.standard_normal((elements, points, 6)) if has_fixed else np.zeros((1, 1, 1)))
    return (rng.standard_normal((elements, points, n_local, 3)) / _LENGTH,
            rng.random((elements, points)) * _LENGTH ** 3,
            _DISPLACEMENT * rng.standard_normal((elements, n_local, 3)), material, fixed,
            np.zeros((elements, columns)),
            np.zeros((elements, columns, columns)) if want_tangent else np.zeros((1, 1, 1)),
            np.int32(elements), np.int32(has_fixed), np.int32(want_tangent))


def _check_kernel_numba(_):
    """numba vs numpy element_tangent_3d: P1 (1 point) and P2 (14 points), with and without S0, force alone too."""
    from ..core.kernels import numba_available
    from ..model.selftest_helpers import NUMBA_MISSING_DETAILS, numba_twin_result
    from . import tangent_kernel3d
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _NUMBA_TOLERANCE,
                'details': dict(NUMBA_MISSING_DETAILS)}
    import numba
    numpy_twin = tangent_kernel3d._element_tangent_host
    numba_twin = tangent_kernel3d._element_tangent_numba(numba)
    rng = np.random.default_rng(20261111)
    comparisons = {}
    for n_local, points in ((4, 1), (10, 14)):
        for has_fixed in (0, 1):
            for want_tangent in (1, 0):
                label = f'n{n_local} q{points} S0 {has_fixed} tangent {want_tangent}'
                comparisons[label] = (numpy_twin, numba_twin, _kernel_arguments(rng, _NUMBA_ELEMENTS, n_local, points,
                                                                                has_fixed, want_tangent))
    timings = {'P2 force and tangent': (numpy_twin, numba_twin,
                                        _kernel_arguments(rng, _NUMBA_SPEED_ELEMENTS, 10, 14, 1, 1),
                                        *_NUMBA_SPEED_CALLS)}
    result = numba_twin_result(comparisons, timings, _NUMBA_TOLERANCE, numba)
    result['details']['speed_elements'] = _NUMBA_SPEED_ELEMENTS
    result['details']['registered_host_kind'] = tangent_kernel3d.ELEMENT_TANGENT_3D.host_kind
    return result


# ---- fem_fixed_pattern_row_sum_numba: numba row sum vs bincount of FixedPatternCsr.add (next-gains N2)
_ROW_SUM_NODES = 4000          # scalar nodes of the random block-3 pattern
_ROW_SUM_LOCAL = 10            # nodes per element (P2-sized blocks 30 x 30)
_ROW_SUM_ELEMENTS = 256        # compared elements (real and complex)
_ROW_SUM_SPEED_ELEMENTS = 6144  # timed elements (the nonlinear profiler's 5.5e6 block entries per tangent)
_ROW_SUM_SPEED_CALLS = (2, 10)


def _row_sum_case(rng, elements):
    """(FixedPatternCsr block 3 of `elements` random distinct-node elements, its zero data)."""
    from .fixed_pattern import FixedPatternCsr
    dofs = np.array([rng.choice(_ROW_SUM_NODES, _ROW_SUM_LOCAL, replace=False) for _ in range(elements)])
    pattern = FixedPatternCsr(3 * _ROW_SUM_NODES, [dofs], block=3)
    return pattern, pattern.zeros()


def _check_row_sum_numba(_):
    """numba row sum vs bincount of FixedPatternCsr.add on random real / complex P2-sized blocks."""
    from ..core.kernels import numba_available
    from ..model.selftest_helpers import NUMBA_MISSING_DETAILS, numba_twin_result
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _NUMBA_TOLERANCE,
                'details': dict(NUMBA_MISSING_DETAILS)}
    import numba
    rng = np.random.default_rng(20261112)
    width = 3 * _ROW_SUM_LOCAL
    comparisons, timings = {}, {}
    pattern, data = _row_sum_case(rng, _ROW_SUM_ELEMENTS)

    def twins(target_pattern):
        return (lambda values, blocks: target_pattern.add(values, 0, blocks, numba=False),
                lambda values, blocks: target_pattern.add(values, 0, blocks, numba=True))
    numpy_twin, numba_twin = twins(pattern)
    comparisons['real'] = (numpy_twin, numba_twin, (data, rng.standard_normal((_ROW_SUM_ELEMENTS, width, width))))
    complex_blocks = (rng.standard_normal((_ROW_SUM_ELEMENTS, width, width))
                      + 1j * rng.standard_normal((_ROW_SUM_ELEMENTS, width, width)))

    def as_float(twin):   # complex data summed, then written as float64 pairs (the compared output)
        def run(values, blocks, out):
            twin(values, blocks)
            out[...] = values.view(np.float64)
        return run
    comparisons['complex'] = (as_float(numpy_twin), as_float(numba_twin),
                              (data.astype(np.complex128), complex_blocks, np.zeros(2 * data.size)))
    speed_pattern, speed_data = _row_sum_case(rng, _ROW_SUM_SPEED_ELEMENTS)
    timings['P2 tangent sum'] = (*twins(speed_pattern),
                                 (speed_data, rng.standard_normal((_ROW_SUM_SPEED_ELEMENTS, width, width))),
                                 *_ROW_SUM_SPEED_CALLS)
    result = numba_twin_result(comparisons, timings, _NUMBA_TOLERANCE, numba)
    result['details']['speed_block_entries'] = _ROW_SUM_SPEED_ELEMENTS * width * width
    return result


TANGENT_ASSEMBLER_SELFTEST_CASES = (
    SelfTestCase(
        name='fem3d_tangent_assembler',
        label='FEM: fixed-pattern total-Lagrangian tangent assemblers (3D tetrahedra, 2D triangles)',
        description='CPU checks of fem.elasticity3d.TangentAssembler3D (P1 / P2 on a graded 3 x 2 x 2 Kuhn box, a full '
                    'anisotropic C in two materials, three void elements, a random displacement of ~1 % strain and a '
                    'random prescribed + thermal initial stress, 80 % random free dofs) and of the 2D twin '
                    'fem.elasticity.TangentAssembler (P2 triangles, planar monoclinic and axisymmetric off the axis): '
                    'internal force (with the tangent and alone), full tangent and free-free reduced tangent equal '
                    f'internal_force_and_tangent[_3d] and its [free][:, free] slicing within {_RTOL:g} relative; the '
                    'reduced CSR is canonical; the pattern equals the structural pattern of the element blocks and is '
                    'the same at u = 0; 3D also with the gradient cache off (cache_bytes 0), and the element kernel '
                    '(tangent_kernel3d host twin, the default) against element_kernel=False. error = number of failed '
                    'sub-checks, tolerance 0 (first estimates, not yet run).',
        specs=lambda: {},
        check=_check_tangent_assembler,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem3d_tangent_assembler_gpu',
        label='FEM: device tangent assembler and device-value cuDSS refactorisation (plan G5, GPU)',
        description='The fem3d_tangent_assembler box (P1 / P2, two materials, void elements, random S0): the device '
                    'TangentAssembler3D (gradients cached and recomputed) against the host one at two random '
                    'displacements (the second reusing the device S0), without S0, and tangent(): force (with the '
                    'tangent and alone), tangent data, full pattern, reduced values / CSR and the full product within '
                    f'{_DEVICE_RTOL:g} relative; the CUDA element kernel against the batched cupy formulas '
                    '(element_kernel=False: force, tangent data); element_strains_3d (linear and Green-Lagrange, values and strains) of '
                    'a cupy displacement against the host at random points within the same tolerance. With '
                    'nvmath-python: a P2 SPD reduced tangent planned at one displacement and refreshed at another by CudssFactorization.refactorize_device_values (double and mixed): plan '
                    f'reused, operator norm within {_DEVICE_RTOL:g}, solves against SuperLU within {_SOLVE_RTOL:g} '
                    f'(mixed {_MIXED_SOLVE_RTOL:g}) and against a host-refreshed cuDSS factor. error = number of '
                    'failed sub-checks, tolerance 0 (first estimates, not yet run).',
        specs=lambda: {},
        check=_check_device_assembler,
        backends=('gpu',),
        max_wall_time=120.0),
    SelfTestCase(
        name='fem3d_tangent_kernel_numba',
        label='FEM: numba vs numpy per-element total-Lagrangian force and tangent (CPU)',
        description='The numba twin of fem.element_tangent_3d (fem.tangent_kernel3d: per element H, Green-Lagrange E, '
                    'S = C E + S0, B + B_NL, force, (B + B_NL)^T C (B + B_NL) + K_sigma) against its numpy twin (the '
                    f'batched elasticity3d formulas) on {_NUMBA_ELEMENTS} random elements per variant: P1-sized (4 '
                    'nodes, 1 point) and P2-sized (10 nodes, 14 points), with and without S0, force with the tangent '
                    'and alone; measured = error = the worst max |numba - numpy| relative to the peak of each output, '
                    f'tolerance {_NUMBA_TOLERANCE:g}. Details: ms per call of both on {_NUMBA_SPEED_ELEMENTS} P2 '
                    'elements. Without numba nothing is compared (error 0). Not yet run.',
        specs=lambda: {},
        check=_check_kernel_numba,
        backends=('cpu',),
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_fixed_pattern_row_sum_numba',
        label='FEM: numba row-parallel block sum vs bincount of FixedPatternCsr.add (CPU)',
        description='fem.fixed_pattern.FixedPatternCsr.add with numba=True (prange over global rows, each thread adding '
                    'the block rows of its rows at their kept slots) against numba=False (one bincount) on a random '
                    f'block-3 pattern of {_ROW_SUM_ELEMENTS} elements of {_ROW_SUM_LOCAL} distinct nodes out of '
                    f'{_ROW_SUM_NODES}, real and complex blocks; measured = error = the worst max |numba - bincount| '
                    f'relative to the peak, tolerance {_NUMBA_TOLERANCE:g}. Details: ms per call of both on '
                    f'{_ROW_SUM_SPEED_ELEMENTS} elements (5.5e6 block entries). Without numba nothing is compared. '
                    'Not yet run.',
        specs=lambda: {},
        check=_check_row_sum_numba,
        backends=('cpu',),
        max_wall_time=300.0),
)

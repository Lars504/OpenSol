"""Line-block factors (solvers/line_block) of the body-of-revolution operator family A(m) = base + m K1 + m^2 K2 of one
BORAssembly (symmetry 2.3, solver_plan Part II stage 4): line layout of the (E_r, u, E_z) unknowns, batched order
factors for shift-invert eigen-solves, single driven factors and frequency_domain wavelength batches (Part II stage 3);
numpy at module level."""
# Layout: block_axis 0 = z rows (E_r and u of cell row j, E_z of z-edge l on line l), 1 = r columns (E_r of r-edge k on
# line k, u and E_z of cell column i). Every a point of the curl couples unknowns at most one line apart, so A(m) is
# block-tridiagonal in either; the cheaper n_blocks b^3 is taken (b ~ 3 x the cells across a line).
import math

import numpy as np

from ...core.constants import C0
from ...solvers.line_block import LineBlockFactorization, LineBlockLayout, batch_factorizations
from .body_of_revolution import _full_offsets, axis_class


def bor_line_ids(shape, unknown_index, block_axis):
    """int64 (n,) grid line of each active unknown (body_of_revolution order E_r, u, E_z) for block_axis 0 (z rows) or
    1 (r columns)."""
    nr, nz = shape
    n_er, n_u, n_ez = _full_offsets(shape)
    er_edge, er_row = np.divmod(np.arange(n_er), nz)
    u_column, u_row = np.divmod(np.arange(n_u), nz)
    ez_column, ez_edge = np.divmod(np.arange(n_ez), nz + 1)
    if block_axis == 0:
        full = np.concatenate((er_row, u_row, ez_edge))
    elif block_axis == 1:
        full = np.concatenate((er_edge, u_column, ez_column))
    else:
        raise ValueError(f'block_axis must be 0 or 1, got {block_axis!r}')
    return full[np.asarray(unknown_index)]


def bor_layout(shape, unknown_index, block_axis=None):
    """LineBlockLayout of the active unknowns: blocks = z rows (0) or r columns (1), None = the cheaper n_blocks b^3."""
    def cost(axis):
        sizes = np.bincount(bor_line_ids(shape, unknown_index, axis))
        return np.count_nonzero(sizes) * float(sizes.max()) ** 3

    axis = min((0, 1), key=cost) if block_axis is None else block_axis
    return LineBlockLayout(bor_line_ids(shape, unknown_index, axis), block_axis=axis)


def order_factorizations(solver, assembly, orders, owner=None):
    """Yield (order, LineBlockFactorization) of every order of `orders` on one BORAssembly: one set_operator per axis
    class (base, matrix terms K1 / K2), one kept batch factor per chunk of its orders (released before the next)."""
    by_class = {}
    for order in orders:
        by_class.setdefault(axis_class(order), []).append(int(order))
    for class_index, class_orders in by_class.items():
        base, matrix_terms, unknown_index = assembly.order_family(class_index)
        solver.set_operator(base, bor_layout(assembly.shape, unknown_index), (), owner=owner, matrix_terms=matrix_terms)
        coefficients = (np.array([[order, order ** 2] for order in class_orders], dtype=np.complex128) if matrix_terms
                        else np.zeros((len(class_orders), 0), dtype=np.complex128))
        step = solver.chunk_size(len(class_orders), 1, kept=True)
        for start in range(0, len(class_orders), step):
            factor = solver.factorize(coefficients[start:start + step])
            try:   # also on an exception in the consumer or an early close of the generator
                for index, order in enumerate(class_orders[start:start + step]):
                    yield order, LineBlockFactorization(factor, index, f'{solver.label()} (m = {order})')
            finally:
                factor.release()


def driven_factorization(solver, system, owner=None):
    """LineBlockFactorization of one BORSystem's operator (set_operator with no terms, one shift)."""
    solver.set_operator(system.operator, bor_layout(system.shape, system.unknown_index), (), owner=owner)
    return LineBlockFactorization(solver.factorize(np.zeros((1, 0), dtype=np.complex128)), 0,
                                  f'{solver.label()} (m = {system.azimuthal_order})')


def wavelength_factorizations(solver, systems, owner=None):
    """[LineBlockFactorization] of one factorize_operators batch of BORSystems of one order at several omegas (no shift
    family: A is not affine in omega); ValueError unless their active unknowns agree (one layout)."""
    first = systems[0]
    for system in systems[1:]:
        if system.azimuthal_order != first.azimuthal_order or not np.array_equal(system.unknown_index,
                                                                                 first.unknown_index):
            raise ValueError('wavelength_factorizations: the systems differ in order or active unknowns')
    labels = [f'{solver.label()} (m = {system.azimuthal_order}, lambda = '
              f'{2.0 * math.pi * C0 / system.omega * 1e9:.6g} nm)' for system in systems]
    return batch_factorizations(solver, [system.operator for system in systems],
                                bor_layout(first.shape, first.unknown_index), labels, owner=owner)


def line_block_bytes(shape, precision, n_shifts=1):
    """(operator bytes, kept-factor bytes of n_shifts) of the class |m| >= 1 family on an (nr, nz) grid (CPU, the
    memory estimate; the full unknown set, cheaper axis, ~25 CSR entries per operator row)."""
    from ...solvers.line_block import operator_bytes, shift_bytes
    nr, nz = shape
    n_unknowns = 3 * nr * nz + nr + nz
    n_blocks, block_size = min((nz + 1, 3 * nr + 1), (nr + 1, 3 * nz + 1), key=lambda pair: pair[0] * pair[1] ** 3)
    entries = 25 * n_unknowns
    return (operator_bytes(n_blocks, block_size, n_unknowns, entries, 0, precision, complex_operator=True,
                           n_matrix_terms=2, matrix_entries=2 * entries),
            int(n_shifts) * shift_bytes(n_blocks, block_size, n_unknowns, 1, precision, matrix_terms=True, kept=True))


def driven_line_block_bytes(shape, precision, n_operators, entries):
    """Device bytes of the driven single-order solves on an (nr, nz) grid (CPU, the memory estimate; the full unknown
    set, cheaper axis, `entries` CSR entries per operator): one bound operator and its factor (driven_factorization),
    or from two planned wavelengths their factorize_operators batch (wavelength_factorizations: no bound family)."""
    from ...solvers.line_block import operator_bytes, operator_shift_bytes, shift_bytes
    nr, nz = shape
    n_unknowns = 3 * nr * nz + nr + nz
    n_blocks, block_size = min((nz + 1, 3 * nr + 1), (nr + 1, 3 * nz + 1), key=lambda pair: pair[0] * pair[1] ** 3)
    factor = shift_bytes(n_blocks, block_size, n_unknowns, 1, precision, kept=True)
    if int(n_operators) < 2:
        return operator_bytes(n_blocks, block_size, n_unknowns, entries, 0, precision, complex_operator=True) + factor
    return int(n_operators) * (operator_shift_bytes(n_blocks, block_size, n_unknowns, entries, precision) + factor)

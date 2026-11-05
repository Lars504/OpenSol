"""Line-block layout of the 2D FDFD Helmholtz operator (solvers/line_block in the 'linear' role of em_fdfd_ez / _hz,
solver_plan Part II stages 2 and 3): block axis rule, layout, wavelength batch size and CPU byte counts; numpy at module
level."""
# One unknown per grid cell, flat i * ny + j over the full grid (PML included). block_axis 0 = the cells of one y row j
# (b = nx, lines coupled along y), 1 = one x column i (b = ny). The 5-point stencil (9-point with hz eps_xy) couples
# neighbouring lines only, but a periodic / Bloch wrap along the line axis is not block-tridiagonal, so the periodic
# axis goes inside the blocks (periodic x -> 0, periodic y -> 1) and two periodic axes are refused.
import numpy as np

from ...solvers.line_block import LineBlockLayout, operator_bytes, shift_bytes

BOTH_PERIODIC_TEXT = ('the line_block linear solver cannot take two periodic axes (a periodic wrap along the line axis '
                      'is not block-tridiagonal)')
LINE_BLOCK_TEXT = 'line-block'   # solve_summary / last_linear_text of a line-block solve
# frequency_domain wavelengths factorised in one factorize_operators batch (Part II stage 3), at most what
# line_block.MEMORY_FRACTION of the free memory holds at run time; the memory estimate counts one wavelength
MAX_BATCH_WAVELENGTHS = 8


def fdfd_block_axis(nx, ny, periodic_x, periodic_y, refuse=True):
    """Grid axis inside a block: the periodic one, else the cheaper n_blocks b^3 (ties: 1); ValueError for two periodic
    axes (refuse=False: the cheaper axis, for estimates)."""
    if periodic_x and periodic_y and refuse:
        raise ValueError(f'{BOTH_PERIODIC_TEXT}; use the sparse_direct solver')
    if periodic_x != periodic_y:
        return 0 if periodic_x else 1
    costs = {1: nx * float(ny) ** 3, 0: ny * float(nx) ** 3}
    return min((1, 0), key=costs.get)


def fdfd_layout(nx, ny, periodic_x, periodic_y):
    """LineBlockLayout of the full (nx, ny) grid on fdfd_block_axis."""
    return LineBlockLayout.from_grid_cells(np.arange(int(nx) * int(ny)), (int(nx), int(ny)),
                                           fdfd_block_axis(nx, ny, periodic_x, periodic_y))


def fdfd_line_block_bytes(nx, ny, periodic_x, periodic_y, precision, entries_per_row=5):
    """(operator bytes, kept-factor bytes) of one FDFD operator on the solver's device (CPU, memory estimate)."""
    nx, ny = int(nx), int(ny)
    axis = fdfd_block_axis(nx, ny, periodic_x, periodic_y, refuse=False)
    n_blocks, block_size = (ny, nx) if axis == 0 else (nx, ny)
    n_unknowns = nx * ny
    return (operator_bytes(n_blocks, block_size, n_unknowns, entries_per_row * n_unknowns, 0, precision,
                           complex_operator=True),
            shift_bytes(n_blocks, block_size, n_unknowns, 1, precision))

"""Effective-index (2.5D) collapse of a vertical layer stack (B4): the guided mode of the stack, solved by the A3 slab
mode solver on a fine 1D grid across the layers, gives n_eff; a 2D top view then uses eps = n_eff^2. Host numpy only."""
import functools
import math

import numpy as np

from ...core.constants import C0
from .slab_modes import SlabLine, solve_slab_modes

# Vertical slab polarisation -> SlabLine polarisation of a line along the stack normal z (u out of the slab's (s, z)
# plane): 'te' u = E parallel to the layers (pairs with the in-plane-E top-view types em_*_hz), 'tm' u = H parallel to
# them (E mostly along z, pairs with em_*_ez).
SLAB_POLARIZATIONS = {'te': 'ez', 'tm': 'hz'}
_CELLS_PER_WAVELENGTH = 100        # auto cell size: wavelength / (this * max index)
_CELLS_PER_LAYER = 20              # and at most 1/this of the thinnest film
_DECAY_LENGTHS = 14.0              # cladding depth in field decay lengths 1/gamma: the tail drops to ~1e-6 at the wall
_START_CLADDING_WAVELENGTHS = 1.0  # first cladding depth (each side) in vacuum wavelengths
_MAX_CLADDING_WAVELENGTHS = 50.0   # deeper claddings mean a mode too weakly guided for the effective-index method
_GUIDED_MARGIN = 1e-9              # relative: n_eff must exceed the larger cladding index by this


def _check_stack(layer_indices, layer_thicknesses, wavelength, slab_polarization, mode_number, cell_size):
    """ValueError naming the first bad argument."""
    if slab_polarization not in SLAB_POLARIZATIONS:
        raise ValueError(f"slab polarization must be one of {', '.join(SLAB_POLARIZATIONS)}, got {slab_polarization!r}")
    if len(layer_indices) < 3:
        raise ValueError(f'the layer stack needs at least 3 indices (substrate, one or more films, cover), got '
                         f'{len(layer_indices)}')
    if len(layer_thicknesses) != len(layer_indices) - 2:
        raise ValueError(f'{len(layer_indices)} layer indices need {len(layer_indices) - 2} film thicknesses (the '
                         f'substrate and cover are semi-infinite), got {len(layer_thicknesses)}')
    numbers = (*layer_indices, *layer_thicknesses, wavelength) + (() if cell_size is None else (cell_size,))
    if not all(math.isfinite(number) and number > 0.0 for number in numbers):
        raise ValueError('layer indices, film thicknesses, wavelength and cell size must be finite and > 0')
    if isinstance(mode_number, bool) or int(mode_number) != mode_number or mode_number < 1:
        raise ValueError(f'mode number must be an integer >= 1, got {mode_number!r}')


def _stack_line(layer_indices, layer_thicknesses, cladding_depths, cell_target, polarization):
    """SlabLine across substrate (depth cladding_depths[0]), films, cover (cladding_depths[1]), Dirichlet ends; every
    layer gets whole cells (interfaces on faces)."""
    widths, permittivities = [], []
    for index, thickness in zip(layer_indices, (cladding_depths[0], *layer_thicknesses, cladding_depths[1])):
        n_cells = max(1, math.ceil(thickness / cell_target - 1e-9))
        widths.extend([thickness / n_cells] * n_cells)
        permittivities.extend([index ** 2] * n_cells)
    widths = np.asarray(widths, dtype=np.float64)
    eps = np.asarray(permittivities, dtype=np.float64)
    n_cells = widths.size
    spacing = np.empty(n_cells + 1)
    spacing[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    spacing[0], spacing[-1] = widths[0], widths[-1]
    if polarization == 'ez':   # TE: d2E/dz2 + k0^2 eps E = beta^2 E
        cell_weight = np.ones(n_cells)
        face_weight = np.ones(n_cells + 1)
    else:                      # TM: eps d/dz (1/eps dH/dz) + k0^2 eps H = beta^2 H
        cell_weight = eps
        # (1/eps) dH/dz ~ the E parallel to the layers is continuous on an interface face, so H differs across the
        # face by it times the integral of eps: face weight 1 / (width-weighted mean eps), as yee_grid.face_average
        face_weight = np.empty(n_cells + 1)
        face_weight[1:-1] = spacing[1:-1] / (0.5 * (widths[:-1] * eps[:-1] + widths[1:] * eps[1:]))
        face_weight[0], face_weight[-1] = 1.0 / eps[0], 1.0 / eps[-1]
    return SlabLine(polarization=polarization, cell_widths=widths, cell_stretch=np.ones(n_cells, dtype=np.complex128),
                    cell_weight=cell_weight.astype(np.complex128), index_term=eps.astype(np.complex128),
                    active=np.ones(n_cells, dtype=bool), face_inverse_spacing=1.0 / spacing,
                    face_stretch=np.ones(n_cells + 1, dtype=np.complex128),
                    face_weight=face_weight.astype(np.complex128), lo_end='dirichlet', hi_end='dirichlet')


def effective_index(layer_indices, layer_thicknesses, wavelength, slab_polarization='te', mode_number=1,
                    cell_size=None):
    """n_eff of guided mode mode_number (1 = fundamental) of the stack substrate / films / cover (indices bottom to
    top, film thicknesses [m]) at the vacuum wavelength [m]; ValueError when the stack does not guide it."""
    return _effective_index(tuple(float(index) for index in layer_indices),
                            tuple(float(thickness) for thickness in layer_thicknesses), float(wavelength),
                            slab_polarization, mode_number, None if cell_size is None else float(cell_size))


@functools.lru_cache(maxsize=64)   # validation calls it on every check of an unchanged model
def _effective_index(layer_indices, layer_thicknesses, wavelength, slab_polarization, mode_number, cell_size):
    _check_stack(layer_indices, layer_thicknesses, wavelength, slab_polarization, mode_number, cell_size)
    mode_number = int(mode_number)
    cladding_indices = (layer_indices[0], layer_indices[-1])
    cladding_index = max(cladding_indices)
    if max(layer_indices[1:-1]) <= cladding_index:
        raise ValueError(f'no film index exceeds the larger cladding index {cladding_index:g}: the stack guides no mode')
    if cell_size is None:
        cell_size = min(wavelength / (_CELLS_PER_WAVELENGTH * max(layer_indices)),
                        min(layer_thicknesses) / _CELLS_PER_LAYER)
    k0 = 2.0 * math.pi / wavelength
    omega = k0 * C0
    polarization = SLAB_POLARIZATIONS[slab_polarization]
    depths = [_START_CLADDING_WAVELENGTHS * wavelength] * 2
    max_depth = _MAX_CLADDING_WAVELENGTHS * wavelength
    while True:   # deepen the claddings until the mode's tails have decayed at both walls
        line = _stack_line(layer_indices, layer_thicknesses, depths, cell_size, polarization)
        modes = solve_slab_modes(line, omega, mode_number)
        guided = [mode for mode in modes if mode.n_eff.real > cladding_index * (1.0 + _GUIDED_MARGIN)]
        if len(guided) >= mode_number:
            n_eff = float(guided[mode_number - 1].n_eff.real)
            needed = [_DECAY_LENGTHS / (k0 * math.sqrt(n_eff ** 2 - index ** 2)) for index in cladding_indices]
            if all(need <= depth for need, depth in zip(needed, depths)):
                return n_eff
            depths = [max(depth, 1.2 * need) for depth, need in zip(depths, needed)]
            if max(depths) > max_depth:
                raise ValueError(f'{slab_polarization.upper()} mode {mode_number} reaches more than '
                                 f'{_MAX_CLADDING_WAVELENGTHS:g} wavelengths into the cladding (n_eff {n_eff:.6g} '
                                 f'barely above {cladding_index:g}): too weakly guided for the effective-index method')
        else:   # walls too close can push a weakly guided mode below the cladding index: deepen before giving up
            if max(depths) >= max_depth:
                raise ValueError(f'the stack guides only {len(guided)} {slab_polarization.upper()} mode(s) at '
                                 f'{wavelength:.6g} m, fewer than mode number {mode_number}')
            depths = [min(4.0 * depth, max_depth) for depth in depths]

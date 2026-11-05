"""Acoustic CPML: damping profiles of the absorbing layers in the outermost grid cells, the slab index maps and the
compact psi memory arrays the fused velocity / stress kernels update (host numpy profiles, arrays of the run backend)."""
import math

import numpy as np

from ...core.constants import DTYPE

PML_ORDER = 2   # polynomial grading d(depth) = d0 depth^order


def _layer_depths(edges, positions, n_lo, n_hi):
    """Relative depth 0..1 of `positions` inside the lo / hi layer (0 outside); edges = the (n+1,) cell edges."""
    n = len(edges) - 1
    depth = np.zeros(len(positions))
    if n_lo:
        inner = edges[n_lo]
        depth = np.maximum(depth, np.clip((inner - positions) / (inner - edges[0]), 0.0, 1.0))
    if n_hi:
        inner = edges[n - n_hi]
        depth = np.maximum(depth, np.clip((positions - inner) / (edges[n] - inner), 0.0, 1.0))
    return depth


def _peak_damping(edges, n_cells, side, wave_speed, reflection):
    """d0 [1/s] = (order + 1) c ln(1 / R) / (2 L) of one layer of n_cells cells (L its thickness)."""
    n = len(edges) - 1
    thickness = edges[n_cells] - edges[0] if side == 'lo' else edges[n] - edges[n - n_cells]
    return (PML_ORDER + 1) * wave_speed * math.log(1.0 / reflection) / (2.0 * thickness)


class AcousticAxisLayers:
    """CPML of one axis: absorbing layers of n_lo / n_hi cells at its ends, slab map (int32, -1 outside the layers),
    (4, n) float32 profile rows (b, a at cell centres; b, a at hi faces) refreshed by configure; arrays on xp."""

    def __init__(self, cell_widths, n_lo, n_hi, xp):
        self._xp = xp
        widths = np.asarray(cell_widths, dtype=np.float64)
        n = widths.size
        self.n_lo, self.n_hi = int(n_lo), int(n_hi)
        if self.n_lo + self.n_hi >= n:
            raise ValueError(f'{n} cells cannot hold absorbing layers of {self.n_lo} + {self.n_hi} cells (at least one '
                             'cell must remain between them)')
        self.edges = np.concatenate(([0.0], np.cumsum(widths)))
        self.slab = xp.asarray(slab_map(n, self.n_lo, self.n_hi))
        self.n_slab = self.n_lo + self.n_hi
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self._depth_e = _layer_depths(self.edges, centres, self.n_lo, self.n_hi)
        self._depth_h = _layer_depths(self.edges, self.edges[1:], self.n_lo, self.n_hi)
        self._is_lo = centres < self.edges[n] * 0.5   # which layer's d0 a row takes
        self.profiles = xp.empty((4, n), dtype=DTYPE)

    def configure(self, wave_speed, reflection, dt):
        """Rewrite the (b, a) rows for the reference wave speed [m/s], target reflection and time step [s]."""
        n = len(self.edges) - 1
        d0_lo = _peak_damping(self.edges, self.n_lo, 'lo', wave_speed, reflection) if self.n_lo else 0.0
        d0_hi = _peak_damping(self.edges, self.n_hi, 'hi', wave_speed, reflection) if self.n_hi else 0.0
        peak = np.where(self._is_lo, d0_lo, d0_hi)
        rows = np.empty((4, n), dtype=np.float64)
        for offset, depth in ((0, self._depth_e), (2, self._depth_h)):
            b = np.exp(-peak * depth ** PML_ORDER * float(dt))
            rows[offset], rows[offset + 1] = b, b - 1.0
        self.profiles[...] = self._xp.asarray(rows, dtype=DTYPE)


def slab_map(n, n_lo, n_hi) -> np.ndarray:
    """int32 slab index of each of n rows (-1 outside the layers): 0 .. n_lo - 1 at the lo end, then n_lo .. at the hi
    end."""
    slab = np.full(n, -1, dtype=np.int32)
    slab[:n_lo] = np.arange(n_lo, dtype=np.int32)
    slab[n - n_hi:] = n_lo + np.arange(n_hi, dtype=np.int32)
    return slab


def psi_array(n_slab, n_other, slab_first, xp):
    """Zeroed float32 psi slab on xp: (n_slab, n_other) for an x slab, (n_other, n_slab) for a y slab (>= 1 row so the
    kernel always gets a valid pointer)."""
    n_slab = max(int(n_slab), 1)
    shape = (n_slab, n_other) if slab_first else (n_other, n_slab)
    return xp.zeros(shape, dtype=DTYPE)

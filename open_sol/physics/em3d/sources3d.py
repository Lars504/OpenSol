"""Electric point dipole of em_fdtd_3d: a current moment I l [A m] on the one E face nearest the point, calibrated
on the device from the live face cb (no host sync). TFSF / mode sources are extension points (package sub file)."""
import math

import numpy as np

from ...core.backend import cp

from ...core.constants import DTYPE, EPS0
from ...core.half_storage import STORED_ACCESS, half_bits
from ...core.kernels import twin
from ...core.sources import _add_scaled_profile
from .engine3d import COMPONENT_AXES

DIPOLE_ORIENTATIONS = ('x', 'y', 'z')


# target += scale * profile on a half_storage field (its uint16 bits, core.half_storage.half_bits), summed in float32
def _add_scaled_stored_host(profile, scale, target):
    stored = target.view(np.float16)
    stored[...] = (stored.astype(np.float32) + scale * profile).astype(np.float16)


_add_scaled_stored = twin('em3d.add_scaled_stored', cp.ElementwiseKernel(
    'float32 profile, float32 scale', 'T target',
    'stored_set(target, stored_load(target) + scale * profile)', 'em3d_add_scaled_stored',
    preamble=STORED_ACCESS), _add_scaled_stored_host)


def nearest_face(edges, coordinate, window):
    """Index f of the E face (hi face of cell f, i.e. cell edge f + 1) nearest to `coordinate` [m], kept inside the
    window's faces between its first and last cell (f in [window.start, window.stop - 2])."""
    first, last = window.start, window.stop - 2
    if last < first:
        raise ValueError('a dipole along an axis needs at least 2 window cells on that axis')
    candidates = range(first, last + 1)
    return min(candidates, key=lambda face: abs(float(edges[face + 1]) - coordinate))


class DipoleSource3D:
    """E_component += weight * s(t) on cell `cell`; weight = cb (I l) / (A_perp L) with L the face (dual) spacing
    along the dipole, so the impressed current density integrates to the moment (calibrate after material changes);
    weight on the engine's array module (allocated by the first calibrate)."""

    def __init__(self, component, cell, temporal, moment, volume):
        if component not in ('ex', 'ey', 'ez'):
            raise ValueError(f"dipole component must be 'ex', 'ey' or 'ez', got {component!r}")
        self.component = component
        self.cell = tuple(int(index) for index in cell)
        self.temporal = temporal
        self.moment = float(moment)
        self.volume = float(volume)
        self.weight = None

    def calibrate(self, engine):
        """weight = live cb of the face (box_face_cb_3d) x moment / volume (in stored field units), on the device."""
        if self.weight is None:
            self.weight = engine.xp.zeros(1, dtype=DTYPE)
        engine.face_cb_into(self.weight, COMPONENT_AXES[self.component], self.cell, (1, 1, 1))
        self.weight *= DTYPE(self.moment / self.volume / engine.e_scale)

    def peak_field(self) -> float:
        """Order of the largest E it drives [V/m]: the charge of a quarter period on its face, moment / (eps0 omega
        V) (half_storage field scale; within ~10x)."""
        frequency = getattr(self.temporal, 'f0', None) or getattr(self.temporal, 'f_peak')
        return abs(self.moment) / (EPS0 * 2.0 * math.pi * float(frequency) * self.volume)

    def inject(self, fields, t):
        i, j, k = self.cell
        target = getattr(fields, self.component)[i:i + 1, j, k]
        if target.dtype == DTYPE:
            _add_scaled_profile(self.weight, DTYPE(self.temporal(t)), target)
        else:   # half_storage fields
            _add_scaled_stored(self.weight, DTYPE(self.temporal(t)), half_bits(target))

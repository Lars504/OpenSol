"""Device bulk media of the NLFD-chi solver: one chi(n) tensor in model axes on the bounding box of its selection,
times a per-cell factor (chi(2) poling sign +1 / -1; chi(3) n0^2 of the n2 / beta conversion or 1; 0 outside), plus
the optional Miller reference and steady-state free-carrier model."""
from dataclasses import dataclass, replace

import numpy as np

from ...core.backend import array_module_of, to_host
from ...core.constants import C0
from .mixing import MediumTensor


@dataclass(frozen=True, eq=False)
class CarrierModel:
    """Steady-state free carriers of a chi(3) medium: N = lifetime x two-photon pair generation (no diffusion), dn =
    -fcd_coefficient N and alpha = fca_cross_section N at reference_omega, both x (reference_omega / omega)^2."""
    lifetime: float             # s
    fcd_coefficient: float      # m^3
    fca_cross_section: float    # m^2
    reference_omega: float      # rad/s
    index: object = None        # xp float32 box array n0 (0 outside the selection), set by bulk_medium

    def permittivity_factor(self, omega):
        """Complex delta eps / (n0 N) [m^3] at omega: -s (2 fcd + j c fca / omega), s = (omega_ref / omega)^2
        (exp(+j omega t): absorption Im < 0)."""
        scale = (self.reference_omega / float(omega)) ** 2
        return -scale * (2.0 * self.fcd_coefficient + 1j * C0 * self.fca_cross_section / float(omega))


@dataclass(frozen=True, eq=False)
class BulkMedium:
    """chi(n) bulk medium: `tensor` (host, model axes) on full-grid cells `box`, `cell_factor` per box cell."""
    name: str
    order: int
    tensor: np.ndarray      # (3,)*(order+1) chi(n) [(m/V)^(n-1)], source indices symmetrised
    box: tuple              # (slice x, slice y) on the full grid
    cell_factor: object     # xp box array: int8 poling sign (chi2) or float32 scale (chi3) inside, 0 outside
    process_scales: tuple = ()        # ((process, factor), ...): MediumTensor.process_scales
    miller_reference_omega: float = None   # rad/s: Miller's rule relative to this reference process, None = off
    carriers: CarrierModel = None     # chi(3) steady-state free carriers (pump_depleted), None = off

    def mixing_tensor(self):
        """The MediumTensor the mixing table enumerates (CPU)."""
        return MediumTensor(name=self.name, order=self.order, tensor=self.tensor, process_scales=self.process_scales)


def mask_bounding_box(mask):
    """(slice, slice) bounding the True cells of a bool (nx, ny) numpy / cupy mask, None when empty (host sync)."""
    rows = np.flatnonzero(to_host(mask.any(axis=1)))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(to_host(mask.any(axis=0)))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1)


def union_box(boxes):
    """(slice, slice) bounding every (slice, slice) box."""
    boxes = tuple(boxes)
    return (slice(min(box[0].start for box in boxes), max(box[0].stop for box in boxes)),
            slice(min(box[1].start for box in boxes), max(box[1].stop for box in boxes)))


def bulk_medium(name, order, tensor, mask, poling=None, cell_scale=None, xp=None, process_scales=(),
                miller_reference_omega=None, carriers=None, carrier_index_squared=None):
    """BulkMedium of a full-grid bool selection mask (cell arrays on xp, None = the mask's module); poling = full-grid
    real values (chi(2) only; >= 0 -> +1, < 0 -> -1 per cell) or None (unpoled); cell_scale = full-grid real values
    multiplying the tensor per cell (chi(3) n0^2; or a scalar) or None; carriers = CarrierModel (chi(3) only) with
    n0^2 = carrier_index_squared (scalar or full grid). ValueError for an empty selection, poling of a chi(3) tensor,
    poling and a cell scale both given."""
    xp = array_module_of(mask) if xp is None else xp
    medium_tensor = MediumTensor.from_tensor(name, order, tensor, process_scales)
    box = mask_bounding_box(mask)
    if box is None:
        raise ValueError(f"nonlinear medium {name!r}: its selection covers no cell")
    inside = mask[box]
    if poling is not None and cell_scale is not None:
        raise ValueError(f"nonlinear medium {name!r}: give poling (chi2) or a cell scale (chi3), not both")
    if poling is not None:
        if order % 2 == 1:   # chi(n) has rank n + 1: inversion flips even orders (odd rank) only
            raise ValueError(f"nonlinear medium {name!r}: poling flips odd-rank tensors only (chi(2)); a chi({order}) "
                             f"tensor is unchanged by domain inversion")
        values = xp.asarray(poling)[box]
        cell_factor = xp.where(values < 0, -1, 1).astype(xp.int8) * inside.astype(xp.int8)
    elif cell_scale is not None:
        scale = xp.asarray(cell_scale)
        scale = scale if scale.ndim == 0 else scale[box]
        cell_factor = (scale * inside).astype(xp.float32)
    else:
        cell_factor = inside.astype(xp.int8)
    if carriers is not None:
        if order != 3:
            raise ValueError(f"nonlinear medium {name!r}: free carriers need a chi(3) medium (two-photon absorption)")
        index_squared = xp.asarray(1.0 if carrier_index_squared is None else carrier_index_squared)
        index_squared = index_squared if index_squared.ndim == 0 else index_squared[box]
        carriers = replace(carriers, index=(xp.sqrt(index_squared) * inside).astype(xp.float32))
    return BulkMedium(name=name, order=medium_tensor.order, tensor=medium_tensor.tensor, box=box,
                      cell_factor=cell_factor, process_scales=medium_tensor.process_scales,
                      miller_reference_omega=None if miller_reference_omega is None else float(miller_reference_omega),
                      carriers=carriers)


def poling_map(media, shape, xp):
    """float32 full-grid sum of the chi(2) media's cell signs (renderer field 'poling'; 0 outside them)."""
    total = xp.zeros(shape, dtype=xp.float32)
    for medium in media:
        if medium.order == 2:
            total[medium.box] += medium.cell_factor
    return total

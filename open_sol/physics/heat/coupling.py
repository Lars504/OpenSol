"""Heat coupling transforms: absorbed EM power -> heat source (em_to_heat_power) and temperature -> thermo-optic
index shift with thermal expansion (heat_to_em_dn*, E13 'index' contributors)."""
import numpy as np

from ...core.backend import cp
from ...core.constants import DTYPE
from ...core.kernels import twin
from ..em.index_reference import reference_index


# One permittivity component as an E13 index shift: n_model = (n0 + dndT dT) sqrt((1 + alpha_xx dT)(1 + alpha_yy dT)),
# dn = n_model - n0, so a lone contributor gives eps = (n0 + dn)^2 = (n0 + dndT dT)^2 (1 + alpha_xx dT)(1 + alpha_yy dT).
# The expansion factor is the transformation-optics image of the in-plane stretch s_x = 1 + alpha_xx dT, s_y = 1 +
# alpha_yy dT on the fixed grid (J = diag(1/s_x, 1/s_y, 1): eps_zz -> eps_zz s_x s_y, mu_zz -> s_x s_y): the index
# grows by sqrt(s_x s_y) ~ 1 + alpha dT, so the optical path grows as the real structure does. Applied to n before
# squaring (n0 + dndT dT)(1 + alpha dT)^2 it would count the expansion twice. Per cell this is a first-order local
# approximation, exact only for an alpha uniform over the mode: a per-cell alpha is no coordinate map (cells at alpha
# 0, e.g. cladding, miss their share of the stretch), so the expansion shift is under-counted by the mode's energy
# fraction outside the expanding material. The exact image of an anisotropic stretch would scale eps_xx by s_y / s_x and
# eps_yy by s_x / s_y; s_x s_y on every component assumes alpha_xx = alpha_yy (error O((alpha_xx - alpha_yy) dT)).
# Evaluated without float32 cancellation: dn = dndT dT s + n0 (s - 1), s = sqrt(1 + q), s - 1 = q / (1 + s),
# q = (alpha_xx + alpha_yy) dT + alpha_xx alpha_yy dT^2 (clamped at -1: s >= 0), never (n0 + dndT dT) s - n0.
def _thermo_optic_index_shift_host(temperature, reference_temperature, n0, dndT, alpha_xx, alpha_yy):
    """float32 numpy twin of _thermo_optic_index_shift (core.kernels; keep both in step)."""
    rise = (temperature - reference_temperature).astype(DTYPE, copy=False)
    stretch_excess = np.maximum((alpha_xx + alpha_yy) * rise + alpha_xx * alpha_yy * rise * rise, DTYPE(-1.0))
    stretch_root = np.sqrt(DTYPE(1.0) + stretch_excess)
    return (dndT * rise * stretch_root + n0 * (stretch_excess / (DTYPE(1.0) + stretch_root))).astype(DTYPE, copy=False)


_thermo_optic_index_shift = twin('heat.thermo_optic_index_shift', cp.ElementwiseKernel(
    'float32 temperature, float32 reference_temperature, float32 n0, float32 dndT, '
    'float32 alpha_xx, float32 alpha_yy',
    'float32 index_shift',
    '''
    const float rise = temperature - reference_temperature;
    const float stretch_excess = fmaxf((alpha_xx + alpha_yy) * rise + alpha_xx * alpha_yy * rise * rise, -1.0f);
    const float stretch_root = sqrtf(1.0f + stretch_excess);
    index_shift = dndT * rise * stretch_root + n0 * (stretch_excess / (1.0f + stretch_root));
    ''',
    'heat_thermo_optic_index_shift'), _thermo_optic_index_shift_host)

_DNDT_ARRAYS = {'eps_r_update': 'dndT_zz', 'eps_xx_update': 'dndT_xx', 'eps_yy_update': 'dndT_yy'}


def em_to_heat_power(q_accum, em, heat=None):
    """Mean absorbed power density [W/m^3] of the EM solver since the last transfer (take_absorbed_power); None for
    an empty window."""
    # q_accum is passed by CouplingTerm (its source field makes the runner enable the EM loss accounting); it is
    # unused here and may be None (an H_z engine before accumulation starts).
    return em.take_absorbed_power()


def em_to_heat_power_final(q_accum, em, heat=None):
    """End-of-run (flush) transfer: the partial window merged with the last full one, so the final Q does not depend
    on where the run stops within a window."""
    return em.take_absorbed_power(merge_previous=True)


def heat_to_em_dn(T, heat, em=None):
    """Full-grid index shift dn_zz [1] of eps_r_update from T (E13 'index' contributor; see the kernel note)."""
    return _index_shift(T, heat, 'eps_r_update')


def heat_to_em_dn_xx(T, heat, em=None):
    """dn_xx of eps_xx_update from T. For H_z the exact image of the expansion scales mu_zz; on eps the phase index is
    the same and the impedance is off by O(alpha dT)."""
    return _index_shift(T, heat, 'eps_xx_update')


def heat_to_em_dn_yy(T, heat, em=None):
    """dn_yy of eps_yy_update from T (see heat_to_em_dn_xx)."""
    return _index_shift(T, heat, 'eps_yy_update')


def heat_to_em_dn_without_expansion(slot):
    """Transform of index slot `slot` with dn = dndT dT only (no (1 + alpha dT) factors): used when a solid_mechanics
    strain-optic coupling with deformation maps the solid strain onto the same EM physics."""
    def thermo_optic_only(T, heat, em=None):
        return _index_shift(T, heat, slot, with_expansion=False)
    return thermo_optic_only


def _index_shift(T, heat, slot, with_expansion=True):
    """Thermo-optic (+ expansion) dn of the permittivity component of index slot `slot` (n0 from index_reference)."""
    materials = heat.domain.materials
    alpha_xx, alpha_yy = ((materials.alpha_xx, materials.alpha_yy) if with_expansion
                          else (DTYPE(0.0), DTYPE(0.0)))
    n0 = reference_index(materials, slot)
    dndT = getattr(materials, _DNDT_ARRAYS[slot])
    return _thermo_optic_index_shift(T, DTYPE(heat.T0), n0, dndT, alpha_xx, alpha_yy)

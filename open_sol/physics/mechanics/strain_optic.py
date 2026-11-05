"""Strain-optic index shift of the solid_em_strain_optic coupling (an E13 'index' contributor): photoelastic dn from the
elastic strain plus the transformation-optics image of the deformation; the in-plane tensor and eps_xy kernels of
acousto-optics."""
from ...core.backend import cp
import numpy as np

from ...core.constants import DTYPE
from ...core.kernels import twin
from ..em.index_reference import reference_index


# numpy twins of the three kernels below (core.kernels; output allocated as the ElementwiseKernel without an output
# argument; scalars broadcast): a change to a cupy source changes its twin.
def _strain_optic_index_shift_host(n0, p11, p12, strain_along, strain_across_a, strain_across_b, strain_xx, strain_yy,
                                   deformation_weight):
    photoelastic = DTYPE(-0.5) * n0 * n0 * n0 * (p11 * strain_along + p12 * (strain_across_a + strain_across_b))
    stretch = DTYPE(1.0) + deformation_weight * (strain_xx + strain_yy)
    return ((n0 + photoelastic) * np.sqrt(np.maximum(stretch, DTYPE(0.0))) - n0).astype(DTYPE, copy=False)


def _unset_to(values, fallback):
    """NaN entries (unset tensor coefficients) replaced by fallback, as the kernels' (p != p) ? fallback : p."""
    return np.where(np.isnan(values), fallback, values).astype(DTYPE, copy=False)


def _photoelastic_tensor_index_shift_host(n0, p_xx, fallback_xx, p_yy, fallback_yy, p_xy, strain_xx, strain_yy,
                                          shear_strain):
    a = _unset_to(p_xx, fallback_xx)
    b = _unset_to(p_yy, fallback_yy)
    c = _unset_to(p_xy, DTYPE(0.0))
    return (DTYPE(-0.5) * n0 * n0 * n0 * (a * strain_xx + b * strain_yy + c * shear_strain)).astype(DTYPE, copy=False)


def _photoelastic_shear_permittivity_shift_host(n0_xx, n0_yy, p_xx, p_yy, p_shear, p11, p12, strain_xx, strain_yy,
                                                shear_strain):
    a = _unset_to(p_xx, DTYPE(0.0))
    b = _unset_to(p_yy, DTYPE(0.0))
    c = _unset_to(p_shear, DTYPE(0.5) * (p11 - p12))
    scale = n0_xx * n0_xx * n0_yy * n0_yy
    return (-scale * (a * strain_xx + b * strain_yy + c * shear_strain)).astype(DTYPE, copy=False)

# Index slot -> (elastic strain along the field, the two across it): Delta(1/n^2)_ii = p11 e_ii + p12 (e_jj + e_kk).
STRAIN_OPTIC_COMPONENTS = {
    'eps_r_update': ('elastic_strain_zz', 'elastic_strain_xx', 'elastic_strain_yy'),
    'eps_xx_update': ('elastic_strain_xx', 'elastic_strain_yy', 'elastic_strain_zz'),
    'eps_yy_update': ('elastic_strain_yy', 'elastic_strain_xx', 'elastic_strain_zz'),
}

# n_model = (n0 + dn_photoelastic) sqrt(1 + w (eps_xx + eps_yy)), dn = n_model - n0: the stretch factor is the image of
# the in-plane deformation on the fixed grid (eps_zz det F, the phase-index convention of heat_em_thermo_optic), w = 1
# with deformation, 0 without. Also the acousto-optic transform of physics/acoustics/couplings.py (w = 0).
strain_optic_index_shift = twin('mechanics.strain_optic_index_shift', cp.ElementwiseKernel(
    'float32 n0, float32 p11, float32 p12, float32 strain_along, float32 strain_across_a, float32 strain_across_b, '
    'float32 strain_xx, float32 strain_yy, float32 deformation_weight',
    'float32 index_shift',
    '''
    const float photoelastic = -0.5f * n0 * n0 * n0 * (p11 * strain_along + p12 * (strain_across_a + strain_across_b));
    const float stretch = 1.0f + deformation_weight * (strain_xx + strain_yy);
    index_shift = (n0 + photoelastic) * sqrtf(fmaxf(stretch, 0.0f)) - n0;
    ''',
    'mechanics_strain_optic_index_shift'), _strain_optic_index_shift_host)

# Index slot -> ((p_I1, fallback), (p_I2, fallback), p_I6) of the in-plane model-axis photoelastic row I (1 xx, 2 yy,
# 3 zz) of the elastic_waves tensor arrays (physics/acoustics/solver.PHOTOELASTIC_TENSOR_ARRAYS, default NaN = unset:
# the isotropic p11 / p12 fallback, 0 for the shear column).
PHOTOELASTIC_TENSOR_ROWS = {
    'eps_xx_update': (('photoelastic_p11', 'photoelastic_p11'), ('photoelastic_p12', 'photoelastic_p12'),
                      'photoelastic_p16'),
    'eps_yy_update': (('photoelastic_p21', 'photoelastic_p12'), ('photoelastic_p22', 'photoelastic_p11'),
                      'photoelastic_p26'),
    'eps_r_update': (('photoelastic_p31', 'photoelastic_p12'), ('photoelastic_p32', 'photoelastic_p12'),
                     'photoelastic_p36'),
}

# dn_I = -n0^3 / 2 (p_I1 S_xx + p_I2 S_yy + p_I6 2 S_xy): the diagonal row I of Delta(1/n^2) = p S (no deformation
# term); NaN coefficients take their fallback (shear: 0).
photoelastic_tensor_index_shift = twin('mechanics.photoelastic_tensor_index_shift', cp.ElementwiseKernel(
    'float32 n0, float32 p_xx, float32 fallback_xx, float32 p_yy, float32 fallback_yy, float32 p_xy, '
    'float32 strain_xx, float32 strain_yy, float32 shear_strain',
    'float32 index_shift',
    '''
    const float a = (p_xx != p_xx) ? fallback_xx : p_xx;
    const float b = (p_yy != p_yy) ? fallback_yy : p_yy;
    const float c = (p_xy != p_xy) ? 0.0f : p_xy;
    index_shift = -0.5f * n0 * n0 * n0 * (a * strain_xx + b * strain_yy + c * shear_strain);
    ''',
    'mechanics_photoelastic_tensor_index_shift'), _photoelastic_tensor_index_shift_host)

# (p_61, p_62, p_66) of the xy row (the em_fdfd_hz eps_xy slot), NaN = unset: 0, 0 and (p11 - p12) / 2 (isotropic).
PHOTOELASTIC_SHEAR_ROW = ('photoelastic_p61', 'photoelastic_p62', 'photoelastic_p66')

# Delta eps_xy = -n0_xx^2 n0_yy^2 (p_61 S_xx + p_62 S_yy + p_66 g): first order in Delta(1/n^2)_xy about the diagonal
# reference; NaN coefficients take the isotropic fallback.
photoelastic_shear_permittivity_shift = twin('mechanics.photoelastic_shear_permittivity_shift', cp.ElementwiseKernel(
    'float32 n0_xx, float32 n0_yy, float32 p_xx, float32 p_yy, float32 p_shear, float32 p11, float32 p12, '
    'float32 strain_xx, float32 strain_yy, float32 shear_strain',
    'float32 permittivity_shift',
    '''
    const float a = (p_xx != p_xx) ? 0.0f : p_xx;
    const float b = (p_yy != p_yy) ? 0.0f : p_yy;
    const float c = (p_shear != p_shear) ? 0.5f * (p11 - p12) : p_shear;
    const float scale = n0_xx * n0_xx * n0_yy * n0_yy;
    permittivity_shift = -scale * (a * strain_xx + b * strain_yy + c * shear_strain);
    ''',
    'mechanics_photoelastic_shear_permittivity_shift'), _photoelastic_shear_permittivity_shift_host)


def make_strain_optic_transform(slot, deformation):
    """CouplingTerm transform (strain_xx, solid solver, EM solver) -> full-grid dn of index slot `slot`."""
    along, across_a, across_b = STRAIN_OPTIC_COMPONENTS[slot]
    deformation_weight = np.float32(1.0 if deformation else 0.0)

    # Named apart from the module kernel: a closure called strain_optic_index_shift would call itself.
    def solid_strain_index_shift(strain_xx, solid, em=None):
        fields = solid.fields
        materials = solid.domain.materials
        return strain_optic_index_shift(reference_index(materials, slot), materials.photoelastic_p11,
                                        materials.photoelastic_p12, getattr(fields, along), getattr(fields, across_a),
                                        getattr(fields, across_b), fields.strain_xx, fields.strain_yy,
                                        deformation_weight)
    return solid_strain_index_shift

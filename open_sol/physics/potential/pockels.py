"""Pockels (linear electro-optic) index shift of the electrostatics_em_pockels coupling: dn_ii = -1/2 n0_ii^3 (r_ii E_k
+ r_i1 E_x + r_i2 E_y) per E13 index slot of the EM target (the tensor terms when allocated), n0 from index_reference;
the 'sum' Delta eps_xy = -n0_xx^2 n0_yy^2 (r_61 E_x + r_62 E_y) of the eps_xy slot."""
from ...core.backend import cp
from ...core.constants import DTYPE
from ...core.kernels import twin

from ..em.index_reference import in_plane_reference_indices, reference_index

POCKELS_ARRAYS = {'eps_r_update': 'r_zz', 'eps_xx_update': 'r_xx', 'eps_yy_update': 'r_yy'}   # index slot -> r array
APPLIED_FIELDS = {'x': 'E_x', 'y': 'E_y'}   # field_component -> electrostatics state field
# index slot -> (r_I1, r_I2) of the in-plane tensor row I (= solver.POCKELS_TENSOR_ARRAYS, optional arrays)
POCKELS_TENSOR_ROWS = {'eps_xx_update': ('pockels_r11', 'pockels_r12'), 'eps_yy_update': ('pockels_r21', 'pockels_r22'),
                       'eps_r_update': ('pockels_r31', 'pockels_r32')}
POCKELS_SHEAR_ROW = ('pockels_r61', 'pockels_r62')   # the xy row (eps_xy slot)

# core.kernels twins (output allocated, as the ElementwiseKernel without an output argument): a change to a cupy
# source changes its numpy twin.
def _pockels_shear_permittivity_shift_host(n0_xx, n0_yy, field_x, field_y, tensor_x, tensor_y):
    return (-(n0_xx * n0_xx * n0_yy * n0_yy) * (tensor_x * field_x + tensor_y * field_y)).astype(DTYPE, copy=False)


def _pockels_index_shift_host(applied_field, n0, pockels_coefficient):
    return (DTYPE(-0.5) * n0 * n0 * n0 * pockels_coefficient * applied_field).astype(DTYPE, copy=False)


def _pockels_tensor_index_shift_host(applied_field, n0, pockels_coefficient, field_x, field_y, tensor_x, tensor_y):
    return (DTYPE(-0.5) * n0 * n0 * n0 * (pockels_coefficient * applied_field + tensor_x * field_x
                                          + tensor_y * field_y)).astype(DTYPE, copy=False)


_pockels_shear_permittivity_shift = twin('potential.pockels_shear_permittivity_shift', cp.ElementwiseKernel(
    'float32 n0_xx, float32 n0_yy, float32 field_x, float32 field_y, float32 tensor_x, float32 tensor_y',
    'float32 permittivity_shift',
    'permittivity_shift = -n0_xx * n0_xx * n0_yy * n0_yy * (tensor_x * field_x + tensor_y * field_y);',
    'potential_pockels_shear_permittivity_shift'), _pockels_shear_permittivity_shift_host)

_pockels_index_shift = twin('potential.pockels_index_shift', cp.ElementwiseKernel(
    'float32 applied_field, float32 n0, float32 pockels_coefficient', 'float32 index_shift',
    'index_shift = -0.5f * n0 * n0 * n0 * pockels_coefficient * applied_field;',
    'potential_pockels_index_shift'), _pockels_index_shift_host)

_pockels_tensor_index_shift = twin('potential.pockels_tensor_index_shift', cp.ElementwiseKernel(
    'float32 applied_field, float32 n0, float32 pockels_coefficient, float32 field_x, float32 field_y, '
    'float32 tensor_x, float32 tensor_y', 'float32 index_shift',
    'index_shift = -0.5f * n0 * n0 * n0 * (pockels_coefficient * applied_field + tensor_x * field_x '
    '+ tensor_y * field_y);',
    'potential_pockels_tensor_index_shift'), _pockels_tensor_index_shift_host)


def make_pockels_transform(slot):
    """CouplingTerm transform (applied field E_k, electrostatics solver, EM solver) -> full-grid dn of index slot
    `slot`; ValueError when the MaterialGrid has no r array (no electrostatics schema registered)."""
    coefficient_name = POCKELS_ARRAYS[slot]
    tensor_names = POCKELS_TENSOR_ROWS[slot]

    def pockels_index_shift(applied_field, electrostatics, em=None):
        materials = electrostatics.domain.materials
        names = materials.property_names()
        if coefficient_name not in names:
            raise ValueError(f"the Pockels transform needs the MaterialGrid array {coefficient_name}, which only an "
                             'electrostatics physics registers')
        n0 = reference_index(materials, slot)
        coefficient = getattr(materials, coefficient_name)
        if not any(name in names for name in tensor_names):   # no crystal tensor arrays allocated
            return _pockels_index_shift(applied_field, n0, coefficient)
        fields = electrostatics.fields
        return _pockels_tensor_index_shift(applied_field, n0, coefficient, fields.E_x, fields.E_y,
                                           *(getattr(materials, name) for name in tensor_names))
    return pockels_index_shift


def pockels_shear_shift(applied_field, electrostatics, em=None):
    """CouplingTerm transform -> full-grid Delta eps_xy of the eps_xy_update slot from both E components; None (no
    data, nothing injected) when the crystal tensor arrays are not allocated."""
    materials = electrostatics.domain.materials
    if not all(name in materials.property_names() for name in POCKELS_SHEAR_ROW):
        return None
    n0_xx, n0_yy = in_plane_reference_indices(materials)
    fields = electrostatics.fields
    return _pockels_shear_permittivity_shift(n0_xx, n0_yy, fields.E_x, fields.E_y,
                                             *(getattr(materials, name) for name in POCKELS_SHEAR_ROW))

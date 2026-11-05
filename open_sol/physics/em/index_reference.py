"""Shared reference index n0 of the E13 'index' slots: n0 per permittivity component in the MaterialGrid, the slot ->
(permittivity array, n0 array) map and the eps = (n0 + dn)^2 kernel; cupy + core.constants only."""
from ...core.backend import array_module_of, cp
from ...core.kernels import twin

from ...core.constants import DTYPE

# Permittivity array -> reference-index array (eps_r holds eps_zz for the E_z types and em_modes).
REFERENCE_INDEX_ARRAYS = {'eps_r': 'n0_zz', 'eps_xx': 'n0_xx', 'eps_yy': 'n0_yy'}
# 'index' source slot -> (permittivity array it rewrites, reference-index array of that component).
INDEX_SLOTS = {f'{array}_update': (array, n0_name) for array, n0_name in REFERENCE_INDEX_ARRAYS.items()}
# Off-diagonal in-plane permittivity (em_fdfd_hz): its 'sum' slot takes Delta eps_xy = -n0_xx^2 n0_yy^2 Delta(1/n^2)_xy
# contributions (first order in Delta(1/n^2) about the diagonal reference), added to the material eps_xy.
OFF_DIAGONAL_ARRAY = 'eps_xy'
OFF_DIAGONAL_SLOT = 'eps_xy_update'

def _index_shift_to_permittivity_host(n0, index_shift, eps):
    index = n0 + index_shift
    eps[...] = index * index
    return eps


# A core.kernels twin (SlotSum 'index' slots on both backends): the numpy twin follows the cupy source and returns the
# output as the ElementwiseKernel does.
index_shift_to_permittivity = twin('em.index_shift_to_permittivity', cp.ElementwiseKernel(
    'float32 n0, float32 index_shift', 'float32 eps',
    'const float index = n0 + index_shift; eps = index * index;',
    'em_index_shift_to_permittivity'), _index_shift_to_permittivity_host)


def record_reference_index(materials, arrays=None):
    """Snapshot n0 = sqrt(max(eps, 0)) over the full grid for each permittivity array of `arrays` (default: all of
    REFERENCE_INDEX_ARRAYS) that exists; idempotent while no coupling has rewritten that array yet."""
    property_names = materials.property_names()
    for eps_name in REFERENCE_INDEX_ARRAYS if arrays is None else arrays:
        if eps_name in property_names:
            reference_index = materials.add_property(REFERENCE_INDEX_ARRAYS[eps_name], 1.0)
            permittivity = getattr(materials, eps_name)
            xp = array_module_of(permittivity)
            reference_index[:] = xp.sqrt(xp.maximum(permittivity, DTYPE(0.0)))


def reference_index(materials, slot):
    """The n0 array of index slot `slot`; ValueError for a non-index slot or a component never recorded."""
    if slot not in INDEX_SLOTS:
        raise ValueError(f"slot '{slot}' is not an index slot; index slots: {', '.join(INDEX_SLOTS)}")
    eps_name, n0_name = INDEX_SLOTS[slot]
    if n0_name not in materials.property_names():
        raise ValueError(f'{n0_name} was never recorded: the MaterialGrid has no {eps_name} (no EM physics with that '
                         f'permittivity component), or record_reference_index ran before it existed')
    return getattr(materials, n0_name)


def in_plane_reference_indices(materials):
    """(n0_xx, n0_yy) of an off-diagonal transform, recording a component no index slot sum recorded yet (its array is
    then still unperturbed: the diagonal index terms of the same coupling transfer first)."""
    property_names = materials.property_names()
    missing = tuple(eps_name for eps_name in ('eps_xx', 'eps_yy')
                    if REFERENCE_INDEX_ARRAYS[eps_name] not in property_names)
    if missing:
        record_reference_index(materials, missing)
    return reference_index(materials, 'eps_xx_update'), reference_index(materials, 'eps_yy_update')

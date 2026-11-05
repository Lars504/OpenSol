"""Transforms of the acoustics_em_acousto_optic coupling (E13 'index' contributors: the photoelastic dn of
physics/mechanics/strain_optic.py from the elastic strain, e_zz = 0, no deformation term; the in-plane tensor with shear
when the photoelastic tensor arrays exist; a 'sum' Delta eps_xy of the xy row) and of the two-way
electrostatics_acoustics_piezo (the bound charge of the strain into the electrostatics, duck-typed: no potential
import)."""
import numpy as np

from ...core.constants import DTYPE, FACES
from ..em.index_reference import in_plane_reference_indices, reference_index
from ..mechanics.strain_optic import (PHOTOELASTIC_SHEAR_ROW, PHOTOELASTIC_TENSOR_ROWS,
                                      photoelastic_shear_permittivity_shift, photoelastic_tensor_index_shift,
                                      strain_optic_index_shift)

# electrostatics face BC kinds whose edge face takes the cell's own polarisation (an electrode holds the surface charge)
_ELECTRODE_KINDS = ('dirichlet', 'robin')

_NO_STRAIN = np.float32(0.0)        # out-of-plane elastic strain (plane strain; the thermoelastic part is not split off)
_NO_DEFORMATION = np.float32(0.0)   # deformation weight: photoelastic part only
_UNSET = DTYPE(np.nan)              # tensor entry without an array: the kernel's isotropic fallback

# Index slot -> (strain along the field, the two across it) as indices into (strain_xx, strain_yy, strain_zz).
_SLOT_COMPONENTS = {'eps_r_update': (2, 0, 1), 'eps_xx_update': (0, 1, 2), 'eps_yy_update': (1, 0, 2)}


def make_acousto_optic_transform(slot):
    """CouplingTerm transform (sxx, acoustic solver, EM solver) -> full-grid dn of index slot `slot` (isotropic p11 /
    p12, or the in-plane tensor row of the slot incl. the centre shear strain when the tensor arrays exist)."""
    along, across_a, across_b = _SLOT_COMPONENTS[slot]
    (row_xx, fallback_xx), (row_yy, fallback_yy), row_xy = PHOTOELASTIC_TENSOR_ROWS[slot]

    def acousto_optic_index_shift(stress_xx, acoustics, em=None):
        strain_xx, strain_yy = acoustics.elastic_strain()
        materials = acoustics.materials
        if acoustics.photoelastic_tensor:
            return photoelastic_tensor_index_shift(
                reference_index(materials, slot), getattr(materials, row_xx), getattr(materials, fallback_xx),
                getattr(materials, row_yy), getattr(materials, fallback_yy), getattr(materials, row_xy), strain_xx,
                strain_yy, acoustics.elastic_shear_strain())
        strains = (strain_xx, strain_yy, _NO_STRAIN)
        return strain_optic_index_shift(reference_index(materials, slot), materials.photoelastic_p11,
                                        materials.photoelastic_p12, strains[along], strains[across_a],
                                        strains[across_b], strain_xx, strain_yy, _NO_DEFORMATION)
    return acousto_optic_index_shift


def acousto_optic_shear_shift(stress_xx, acoustics, em=None):
    """CouplingTerm transform (sxx, acoustic solver, EM solver) -> full-grid Delta eps_xy of the eps_xy_update slot:
    -n0_xx^2 n0_yy^2 (p_61 e_xx + p_62 e_yy + p_66 g), isotropic p_66 = (p11 - p12) / 2 without the tensor arrays."""
    strain_xx, strain_yy = acoustics.elastic_strain()
    materials = acoustics.materials
    n0_xx, n0_yy = in_plane_reference_indices(materials)
    shear_row = (tuple(getattr(materials, name) for name in PHOTOELASTIC_SHEAR_ROW) if acoustics.photoelastic_tensor
                 else (_UNSET, _UNSET, _UNSET))
    return photoelastic_shear_permittivity_shift(n0_xx, n0_yy, *shear_row, materials.photoelastic_p11,
                                                 materials.photoelastic_p12, strain_xx, strain_yy,
                                                 acoustics.elastic_shear_strain())


def bound_charge_transform(stress_xx, acoustics, electrostatics):
    """CouplingTerm transform (sxx, acoustic solver, electrostatics) -> full-grid rho_b = -div(e S) [C/m^3] on the
    electrostatics window; its Dirichlet / Robin outer faces take the edge cell's P, the others (insulating) none."""
    own_faces = tuple(getattr(electrostatics.bcs, face).kind in _ELECTRODE_KINDS for face in FACES)
    return acoustics.bound_charge(electrostatics.window, own_faces)

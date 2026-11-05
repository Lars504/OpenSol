"""Element stiffness of the 3D solid on tetrahedra (piezo_3d_plan stage 1): the (n_elements, 6, 6) Voigt matrix from
the isotropic Lame pair or the 21 c_IJ material arrays (per component, NaN = the isotropic value), its positive
definiteness, the thermal stress per kelvin and the stress of a strain. Host numpy float64."""
# Voigt order of the model axes 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy, engineering shear (S_4 = 2 strain_yz):
# T_I = c_IJ S_J - beta_I dT with beta_I = c_IJ alpha_J (J = 1, 2, 3: diagonal expansion).
from dataclasses import dataclass

import numpy as np

from ...fem.elasticity3d import isotropic_matrix_3d
from .materials import STIFFNESS_TENSOR_ARRAYS_3D

_DEFINITE_TOLERANCE = 1e-9   # smallest eigenvalue / largest eigenvalue of a positive definite matrix


@dataclass(frozen=True, eq=False)
class ElementStiffness3D:
    """Per element `matrix` (n_elements, 6, 6) [Pa] (0 in void elements), `solid` (bool, c11 > 0) and `anisotropic`
    (a solid element whose matrix differs from its isotropic Lame matrix)."""
    matrix: np.ndarray
    solid: np.ndarray
    anisotropic: bool


def element_stiffness_3d(lame_lambda, mu, tensor):
    """ElementStiffness3D from the Lame pair (n_elements,) (0 without youngs_modulus) and tensor {c_IJ: host
    (n_elements,) array, NaN = unset; absent names not allocated}: a finite entry wins its component."""
    isotropic = isotropic_matrix_3d(lame_lambda, mu)
    matrix = isotropic.copy()
    for name in STIFFNESS_TENSOR_ARRAYS_3D:
        values = tensor.get(name)
        if values is None:
            continue
        row, column = int(name[1]) - 1, int(name[2]) - 1
        values = np.asarray(values, dtype=np.float64)
        entry = np.where(np.isfinite(values), values, isotropic[:, row, column])
        matrix[:, row, column] = matrix[:, column, row] = entry
    solid = matrix[:, 0, 0] > 0.0
    matrix[~solid] = 0.0
    anisotropic = bool(np.any((matrix != isotropic).any(axis=(1, 2)) & solid))
    return ElementStiffness3D(matrix, solid, anisotropic)


def check_positive_definite_3d(stiffness, label):
    """ValueError when the 6 x 6 matrix of a solid element of an anisotropic ElementStiffness3D is not positive
    definite (isotropic elements are, by the poisson_ratio check)."""
    if not stiffness.anisotropic:
        return
    eigenvalues = np.linalg.eigvalsh(stiffness.matrix[stiffness.solid])
    indefinite = eigenvalues[:, 0] <= _DEFINITE_TOLERANCE * eigenvalues[:, -1]
    if indefinite.any():
        raise ValueError(f'{label()}: {int(np.count_nonzero(indefinite))} solid element(s) have a 6 x 6 stiffness that '
                         f'is not positive definite (smallest eigenvalue {float(eigenvalues[indefinite, 0].min()):.4g} '
                         'Pa); check the c_ij entries (a crystal entry is positive definite, a hand-written set may '
                         'not be)')


def thermal_stress_3d(matrix, alpha):
    """(n, 6) beta_I = c_IJ alpha_J [Pa/K] of Voigt matrices (n, 6, 6) and the expansion alpha_xx / yy / zz (n, 3)."""
    return np.einsum('nij,nj->ni', matrix[:, :, :3], alpha)


def stress_3d(matrix, strains):
    """(m, q, 6) T_I = c_IJ S_J [Pa] of matrices (m, 6, 6) and Voigt strains (m, q, 6) (no thermal part)."""
    return np.einsum('mij,mqj->mqi', matrix, strains)

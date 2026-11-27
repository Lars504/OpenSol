"""Element stiffness of the 3D solid on tetrahedra: the (n_elements, 6, 6) Voigt matrix from
the isotropic Lame pair or the 21 c_IJ material arrays (per component, NaN = the isotropic value), its positive
definiteness, the thermal stress per kelvin and the stress of a strain, the unit-modulus matrices of the viscoelastic
group terms and the storage / loss pair C', C'' of a harmonic output. Host numpy float64."""
# Voigt order of the model axes 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy, engineering shear (S_4 = 2 strain_yz):
# T_I = c_IJ S_J - beta_I dT with beta_I = c_IJ alpha_J (J = 1, 2, 3: diagonal expansion).
from dataclasses import dataclass

import numpy as np

from ...fem.elasticity3d import isotropic_matrix_3d
from .materials import STIFFNESS_TENSOR_ARRAYS_3D
from .stiffness import UNIT_MODULI

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


def unit_modulus_matrix_3d(part, n_elements) -> np.ndarray:
    """(n_elements, 6, 6) Voigt matrix of a unit modulus: part 'shear' (G = 1: 2 dev eps) or 'bulk' (K = 1: tr eps I),
    the element_stiffness_3d matrix of one element as a read-only broadcast view (no per-element copy)."""
    if part not in UNIT_MODULI:
        raise ValueError(f"unit modulus part must be one of {', '.join(UNIT_MODULI)}, got {part!r}")
    lame_lambda, mu = UNIT_MODULI[part]
    single = element_stiffness_3d(np.full(1, lame_lambda), np.full(1, mu), {}).matrix[0]
    return np.broadcast_to(single, (int(n_elements),) + single.shape)


def harmonic_matrix_pair_3d(matrix, shear, bulk, loss, frequency_elements, deviatoric=False) -> tuple:
    """(storage C', loss C'') (n_elements, 6, 6) of C*(omega) = C' + j C'' at one frequency (the complex-correct
    harmonic outputs): elements outside frequency_elements C and eta C (loss = eta per element); the frequency_elements
    (bool, None = none) the isotropic matrices of (Re K* - 2/3 Re G*, Re G*) and of the imaginary parts (shear / bulk:
    complex per element; deviatoric (mixed): lambda = -2/3 G). C' is `matrix` itself where no element is frequency
    dependent."""
    storage = matrix
    loss_matrix = np.asarray(loss, dtype=np.float64)[:, None, None] * matrix
    if frequency_elements is not None and np.any(frequency_elements):
        selected = np.flatnonzero(frequency_elements)
        shear = np.asarray(shear, dtype=np.complex128)[selected]
        volumetric = ((-2.0 / 3.0) * shear if deviatoric
                      else np.asarray(bulk, dtype=np.complex128)[selected] - (2.0 / 3.0) * shear)
        storage = matrix.copy()
        storage[selected] = isotropic_matrix_3d(volumetric.real, shear.real)
        loss_matrix[selected] = isotropic_matrix_3d(volumetric.imag, shear.imag)
    return storage, loss_matrix


def thermal_stress_3d(matrix, alpha):
    """(n, 6) beta_I = c_IJ alpha_J [Pa/K] of Voigt matrices (n, 6, 6) and the expansion alpha_xx / yy / zz (n, 3)."""
    return np.einsum('nij,nj->ni', matrix[:, :, :3], alpha)


def stress_3d(matrix, strains):
    """(m, q, 6) T_I = c_IJ S_J [Pa] of matrices (m, 6, 6) and Voigt strains (m, q, 6) (no thermal part)."""
    return np.einsum('mij,mqj->mqi', matrix, strains)

"""Piezoelectric solid_mechanics_3d on tetrahedra (piezo_3d_plan stages 2 and 3, setting piezoelectric): the block-layout
coupled operator [[K_uu, K_uphi], [K_uphi^T, -K_phiphi]] with the potential on every window element (void ones
dielectric), its massless-potential mass, the per-element e / eps tensors of the material arrays and the active potential
dofs; host numpy, scipy inside functions. The electrode reduction and the static / modal solves are piezo.PiezoSystem
(shared with 2D)."""
# Unknowns [u (3 n, dof 3 d + component as fem.elasticity3d) | phi (n)] on the one full-mesh scalar DofMap3D (n = its
# dofs): K_uu = int B^T c B and K_uphi = int B^T e^T grad N on the solid elements, K_phiphi = int grad N . eps0 eps grad
# N on the window elements. Electric enthalpy, signs, charges and reactions as piezo.py (T = c S - e^T E, D = e S + eps
# E, E = -grad phi; Q = -sum of an electrode's phi rows of A x, reaction = u rows of A x - f).
import numpy as np

from ...core.constants import EPS0
from ...fem.assembly3d import assemble_lagrange_stiffness_3d
from ...fem.elasticity3d import COMPONENTS, assemble_piezo_coupling_3d
from .materials import PERMITTIVITY_TENSOR_ARRAYS_3D

_PERMITTIVITY_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))   # PERMITTIVITY_TENSOR_ARRAYS_3D order


def block_layout(n_scalar):
    """(displacement dofs (3 n,), potential dofs (n,)) int64 of the block layout over n scalar dofs."""
    n_scalar = int(n_scalar)
    return (np.arange(COMPONENTS * n_scalar, dtype=np.int64),
            COMPONENTS * n_scalar + np.arange(n_scalar, dtype=np.int64))


def piezo_tensor_3d(arrays, n_elements) -> np.ndarray:
    """(n_elements, 3, 6) e_iJ [C/m^2] from {'piezo_e<i><J>': (n_elements,)} (missing entries 0)."""
    tensor = np.zeros((int(n_elements), 3, 6))
    for name, values in arrays.items():
        tensor[:, int(name[-2]) - 1, int(name[-1]) - 1] = values
    return tensor


def permittivity_tensor_3d(arrays, n_elements) -> np.ndarray:
    """(n_elements, 3, 3) symmetric relative permittivity from {'eps_dc_<ij>': (n_elements,)} (missing: 1 on the
    diagonal, 0 off it)."""
    tensor = np.zeros((int(n_elements), 3, 3))
    tensor[:, np.arange(3), np.arange(3)] = 1.0
    for name, (row, column) in zip(PERMITTIVITY_TENSOR_ARRAYS_3D, _PERMITTIVITY_PAIRS):
        if name in arrays:
            tensor[:, row, column] = tensor[:, column, row] = arrays[name]
    return tensor


def check_permittivity_3d(permittivity, elements, label) -> None:
    """ValueError when the relative permittivity (n_elements, 3, 3) of an element under the bool mask is not positive
    definite."""
    if not elements.any():
        return
    smallest = np.linalg.eigvalsh(permittivity[elements])[:, 0]
    indefinite = ~(smallest > 0.0)
    if indefinite.any():
        raise ValueError(f'{label()}: {int(indefinite.sum())} window element(s) have a static permittivity that is not '
                         f'positive definite (smallest eigenvalue {float(smallest.min()):.4g}): check eps_dc_xx / _yy / '
                         '_zz > 0 and the off-diagonal eps_dc_yz / _xz / _xy')


def coupled_operator_3d(dofs, stiffness_operator, piezo, permittivity, solid, window):
    """Host CSR (4 n, 4 n) of the block operator from the assembled K_uu (3 n, 3 n), e (n_elements, 3, 6) [C/m^2] on the
    solid elements and the relative permittivity (n_elements, 3, 3) on the window elements (bool masks)."""
    import scipy.sparse
    coupling = assemble_piezo_coupling_3d(dofs, piezo, elements=solid)
    dielectric = assemble_lagrange_stiffness_3d(dofs, EPS0 * permittivity, elements=window)
    return scipy.sparse.bmat([[stiffness_operator, coupling], [coupling.T, -dielectric]], format='csr')


def coupled_mass_3d(mass):
    """Host CSR (4 n, 4 n) of the block layout from the vector mass (3 n, 3 n): the potential rows are massless."""
    import scipy.sparse
    n_scalar = mass.shape[0] // COMPONENTS
    return scipy.sparse.block_diag((mass, scipy.sparse.csr_matrix((n_scalar, n_scalar))), format='csr')


def potential_active(dofs, window) -> np.ndarray:
    """Bool (n,) scalar dofs a window element touches: the potential lives there (a PML shell carries none)."""
    active = np.zeros(dofs.n_dofs, dtype=bool)
    active[dofs.element_dofs[np.asarray(window, dtype=bool)].ravel()] = True
    return active

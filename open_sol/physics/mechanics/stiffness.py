"""Element stiffness of the 2D solid: the isotropic Lame pair or the anisotropic c_ij
material arrays (per component, NaN = the isotropic value), reduced to the operator coefficients of each out-of-plane
condition, the thermal stress per kelvin and the stress of a strain, the unit-modulus (G = 1 deviatoric, K = 1
volumetric) coefficients of the viscoelastic group terms and the storage / loss pair C', C'' of a harmonic output;
elementwise on the solve backend (numpy / cupy)."""
# Voigt order of the model axes (1 xx, 2 yy, 3 zz, 6 xy; engineering shear g = 2 strain_xy): T_I = c_IJ S_J - beta_I dT
# (I, J in 1, 2, 3, 6), beta_I = c_IJ alpha_J (J = 1, 2, 3: diagonal expansion). Plane strain S_3 = 0 (stress_zz =
# T_3); plane stress T_3 = 0 condensed (c'_IJ = c_IJ - c_I3 c_3J / c_33, beta'_I = beta_I - c_I3 beta_3 / c_33); generalized
# plane strain S_3 = one uniform e; axisymmetric 3 = the hoop theta (c13 = c_rt, c23 = c_zt, c33 = c_tt), orthotropic
# only (c16 = c26 = c36 = 0). Industry-standard plane strain: the full 3D law with the out-of-plane strains held at 0.
from dataclasses import dataclass

import numpy as np

from ...core.backend import array_module_of, to_host
from .materials import SHEAR_COUPLING_ARRAYS, STIFFNESS_TENSOR_ARRAYS

_INDICES = {'1': 0, '2': 1, '3': 2, '6': 3}   # Voigt index -> row of the 4 x 4 (1, 2, 3, 6) block
_DEFINITE_TOLERANCE = 1e-9                     # smallest eigenvalue / largest |eigenvalue| of a positive definite block
# fem.elasticity coefficient names of the in-plane operator and (axisymmetric) its hoop entries
OPERATOR_NAMES = ('c11', 'c12', 'c22', 'c66', 'c16', 'c26')
HOOP_NAMES = ('c13', 'c23', 'c33')
# (lambda, mu) of the unit moduli: G = 1 deviatoric (2 dev eps, lambda = -2/3) and K = 1 volumetric (tr eps I), so an
# isotropic C = G C_shear + K C_bulk exactly (the viscoelastic group terms G*_g D_g + K*_g V_g)
UNIT_MODULI = {'shear': (-2.0 / 3.0, 1.0), 'bulk': (1.0, 0.0)}


@dataclass(frozen=True, eq=False)
class ElementStiffness:
    """Window element coefficients [Pa] (solve dtype, 0 in void cells): `values` {c11 .. c66}, `solid` (bool, c11 > 0),
    `anisotropic` (any c_ij entry set in a solid cell) and `monoclinic` (any c16 / c26 / c36 != 0 there)."""
    values: dict
    solid: object
    anisotropic: bool
    monoclinic: bool

    def __getitem__(self, name):
        return self.values[name]

    def entry(self, row, column):
        """c_IJ of Voigt indices row, column ('1', '2', '3', '6'), symmetric."""
        low, high = sorted((row, column), key=lambda index: _INDICES[index])
        return self.values[f'c{low}{high}']


def element_stiffness(lame_lambda, mu, tensor, dtype):
    """ElementStiffness from the window Lame pair (0 in cells without youngs_modulus) and tensor {c_ij: window array or
    None (not allocated)}: a finite entry wins its component, else the isotropic value (c16 / c26 / c36: 0)."""
    xp = array_module_of(mu)
    isotropic = _isotropic_entries(lame_lambda, mu)
    values, set_masks = {}, []
    for name in STIFFNESS_TENSOR_ARRAYS:
        fallback = isotropic.get(name, 0.0)
        array = tensor.get(name)
        if array is None:
            values[name] = xp.zeros_like(mu) + fallback
            continue
        array = array.astype(dtype)
        finite = xp.isfinite(array)
        values[name] = xp.where(finite, array, fallback).astype(dtype, copy=False)
        if name not in SHEAR_COUPLING_ARRAYS:
            set_masks.append(finite)
    solid = values['c11'] > 0
    for name in STIFFNESS_TENSOR_ARRAYS:
        values[name] = xp.where(solid, values[name], 0.0).astype(dtype, copy=False)
    anisotropic = any(bool((mask & solid).any()) for mask in set_masks)
    monoclinic = any(bool((values[name] != 0).any()) for name in SHEAR_COUPLING_ARRAYS)
    return ElementStiffness(values, solid, anisotropic or monoclinic, monoclinic)


def _isotropic_entries(lame_lambda, mu) -> dict:
    """{c11 .. c66} of an isotropic Lame pair (the shear couplings left out: 0)."""
    longitudinal = lame_lambda + 2.0 * mu
    return {'c11': longitudinal, 'c22': longitudinal, 'c33': longitudinal, 'c12': lame_lambda, 'c13': lame_lambda,
            'c23': lame_lambda, 'c66': mu}


def unit_modulus_stiffness(part, n_elements, dtype=np.float64) -> ElementStiffness:
    """Host ElementStiffness of a unit modulus on every element: part 'shear' (G = 1: 2 dev eps) or 'bulk' (K = 1: tr eps
    I); c11 = 4/3 resp. 1 > 0, so `solid` is all True."""
    if part not in UNIT_MODULI:
        raise ValueError(f"unit modulus part must be one of {', '.join(UNIT_MODULI)}, got {part!r}")
    lame_lambda, mu = UNIT_MODULI[part]
    return element_stiffness(np.full(int(n_elements), lame_lambda), np.full(int(n_elements), mu), {}, dtype)


def unit_operator_coefficients(part, n_elements, axisymmetric) -> dict:
    """fem.elasticity coefficients {c11, c12, c22, c66, c16, c26 (+ c13, c23, c33 axisymmetric)} of a unit modulus
    (unit_modulus_stiffness; plane strain: the out-of-plane condition of an omega-dependent modulus, plane stress being
    refused), the construction SolidMechanicsFEMPhysics._assemble uses."""
    stiffness = unit_modulus_stiffness(part, n_elements)
    coefficients = dict(zip(OPERATOR_NAMES, operator_coefficients(stiffness, False)))
    if axisymmetric:
        coefficients.update({name: stiffness[name] for name in HOOP_NAMES})
    return coefficients


def harmonic_stiffness_pair(stiffness, shear, bulk, loss, frequency_elements, deviatoric=False) -> tuple:
    """(storage C', loss C'') ElementStiffness of C*(omega) = C' + j C'' at one frequency (the complex-correct harmonic
    outputs): elements outside frequency_elements C and eta C (loss = eta per element, 0 = lossless); the
    frequency_elements (bool per element, None = none) the isotropic pair of (Re K* - 2/3 Re G*, Re G*) and of the
    imaginary parts (shear / bulk: complex G*, K* per element at that omega; deviatoric (mixed): lambda = -2/3 G, the
    pressure carries the volumetric part). Host float64; the storage values are stiffness's own arrays where no
    element is frequency dependent."""
    loss = np.asarray(loss, dtype=np.float64)
    storage_values = dict(stiffness.values)
    loss_values = {name: np.asarray(values, dtype=np.float64) * loss for name, values in stiffness.values.items()}
    if frequency_elements is not None and np.any(frequency_elements):
        selected = np.asarray(frequency_elements, dtype=bool)
        shear = np.asarray(shear, dtype=np.complex128)
        volumetric = (-2.0 / 3.0) * shear if deviatoric else np.asarray(bulk, dtype=np.complex128) - (2.0 / 3.0) * shear
        for values, part in ((storage_values, np.real), (loss_values, np.imag)):
            isotropic = _isotropic_entries(np.where(selected, part(volumetric), 0.0), np.where(selected, part(shear), 0.0))
            for name in STIFFNESS_TENSOR_ARRAYS:
                values[name] = np.where(selected, isotropic.get(name, 0.0), np.asarray(values[name], dtype=np.float64))
    flags = (stiffness.solid, stiffness.anisotropic, stiffness.monoclinic)
    return ElementStiffness(storage_values, *flags), ElementStiffness(loss_values, *flags)


def check_positive_definite(stiffness, rows, label, what):
    """ValueError when the symmetric block of Voigt `rows` (e.g. '1236') is not positive definite in a solid cell of an
    anisotropic ElementStiffness (isotropic cells are, by the poisson_ratio check)."""
    if not stiffness.anisotropic:
        return
    solid = np.asarray(to_host(stiffness.solid), dtype=bool)
    if not solid.any():
        return
    size = len(rows)
    block = np.empty((int(np.count_nonzero(solid)), size, size))
    for a, row in enumerate(rows):   # one host copy per symmetric entry, mirrored into the lower triangle
        for b in range(a, size):
            block[:, a, b] = np.asarray(to_host(stiffness.entry(row, rows[b])), dtype=np.float64)[solid]
            block[:, b, a] = block[:, a, b]
    eigenvalues = np.linalg.eigvalsh(block)
    scale = np.max(np.abs(eigenvalues), axis=1)
    indefinite = eigenvalues[:, 0] <= _DEFINITE_TOLERANCE * scale
    if indefinite.any():
        names = ', '.join(f'c{row}{column}' for a, row in enumerate(rows) for column in rows[a:])
        raise ValueError(f'{label}: {int(np.count_nonzero(indefinite))} solid cell(s) have a stiffness that is not '
                         f'positive definite ({what}: the symmetric block of {names}); check the c_ij entries (a '
                         'crystal entry is positive definite, a hand-written set may not be)')


def operator_coefficients(stiffness, plane_stress):
    """(c11, c12, c22, c66, c16, c26) of the in-plane operator: the c_ij themselves, or (plane_stress) condensed by
    sigma_zz = 0 (c33 = 1 in void cells)."""
    names = ('11', '12', '22', '66', '16', '26')
    if not plane_stress:
        return tuple(stiffness[f'c{name}'] for name in names)
    c33 = _safe_divisor(stiffness['c33'])
    return tuple(stiffness[f'c{name}'] - stiffness.entry(name[0], '3') * stiffness.entry('3', name[1]) / c33
                 for name in names)


def thermal_coefficients(stiffness, alpha_xx, alpha_yy, alpha_zz, plane_stress=False):
    """(beta_1, beta_2, beta_3, beta_6) [Pa/K] = c_IJ alpha_J; plane_stress: beta_I - c_I3 beta_3 / c_33 (beta_3 kept,
    the eps_zz rule reads it)."""
    betas = tuple(stiffness.entry(index, '1') * alpha_xx + stiffness.entry(index, '2') * alpha_yy
                  + stiffness.entry(index, '3') * alpha_zz for index in '1236')
    if not plane_stress:
        return betas
    beta_3 = betas[2]
    c33 = _safe_divisor(stiffness['c33'])
    return tuple(beta - stiffness.entry(index, '3') * beta_3 / c33 if index != '3' else beta
                 for index, beta in zip('1236', betas))


def plane_stress_strain(stiffness, strain_xx, strain_yy, shear, thermal_stress_3):
    """eps_zz of sigma_zz = 0: (beta_3 dT - c31 S_xx - c32 S_yy - c36 g) / c33 (thermal_stress_3 = beta_3 dT)."""
    return (thermal_stress_3 - stiffness['c13'] * strain_xx - stiffness['c23'] * strain_yy
            - stiffness['c36'] * shear) / _safe_divisor(stiffness['c33'])


def elastic_stress(stiffness, strain_xx, strain_yy, strain_zz, shear):
    """(T_xx, T_yy, T_zz, T_xy) [Pa] = c_IJ S_J of the strain (S_3 = strain_zz, S_6 = shear = 2 strain_xy), no thermal
    or piezoelectric part; a plain scalar 0 strain (plane strain S_3) adds no pass (launch-bound sweeps on the gpu)."""
    strains = [(column, strain) for column, strain in zip('1236', (strain_xx, strain_yy, strain_zz, shear))
               if not (np.isscalar(strain) and strain == 0)] or [('1', strain_xx)]
    stresses = []
    for index in '1236':
        terms = [stiffness.entry(index, column) * strain for column, strain in strains]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        stresses.append(total)
    return tuple(stresses)


def _safe_divisor(values):
    """values with 1 where they are <= 0 (void cells)."""
    xp = array_module_of(values)
    return xp.where(values > 0, values, 1.0)

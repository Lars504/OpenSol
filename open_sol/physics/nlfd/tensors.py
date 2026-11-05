"""Nonlinear susceptibility tensors (numpy only, GUI-thread safe): crystal-class Voigt d -> rank-3 chi(2), rotations to
the model axes, isotropic / cubic chi(3) and the n2 / beta_TPA <-> chi(3) conversion."""
# Conventions (nonlinear_optics_plan 2.1): exp(+j omega t); P_k = eps0 2^(1-n) sum chi(n) : E..E over ordered signed
# source tuples, so d [m/V] = chi(2) / 2 and Im chi(3) < 0 is loss. A rotation R maps crystal axes to model axes
# (column a = crystal axis a in model coordinates): T'_{ij..} = R_ia R_jb .. T_ab..
import itertools
import math

import numpy as np

from ...core.constants import C0, EPS0

# Voigt index l of the source pair (j, k): 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy (0-based below)
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

# Independent d_il [m/V] per crystal class (full crystal symmetry; Kleinman holds when the listed values satisfy it,
# e.g. 6mm d15 = d31, mm2 d15 = d31 and d24 = d32, 32 d14 = 0). 3m ties d15 = d31 (Kleinman) and keeps the mirror
# perpendicular to crystal x (Boyd's table). 'general' = all 18 d_il.
CRYSTAL_CLASSES = {
    '3m': ('d22', 'd31', 'd33'),
    '-43m': ('d14',),
    '6mm': ('d15', 'd31', 'd33'),
    'mm2': ('d15', 'd24', 'd31', 'd32', 'd33'),
    '32': ('d11', 'd14'),
    'general': tuple(f'd{i}{l}' for i in range(1, 4) for l in range(1, 7)),
}

# Crystal c axis (crystal z) along model z / x / y; cyclic axis permutations (proper rotations).
CUT_ROTATIONS = {
    'z_cut': np.eye(3),
    'x_cut': np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),   # crystal x, y, z -> model y, z, x
    'y_cut': np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),   # crystal x, y, z -> model z, x, y
}


def voigt_matrix(crystal_class, coefficients):
    """(3, 6) float64 d_il [m/V] of a crystal class from {name: value} (missing names = 0); ValueError for an unknown
    class or a coefficient the class does not have."""
    if crystal_class not in CRYSTAL_CLASSES:
        raise ValueError(f"crystal_class must be one of {', '.join(CRYSTAL_CLASSES)}, got {crystal_class!r}")
    allowed = CRYSTAL_CLASSES[crystal_class]
    unknown = sorted(set(coefficients) - set(allowed))
    if unknown:
        raise ValueError(f"crystal class {crystal_class} has the coefficients {', '.join(allowed)}; got {unknown}")
    value = {name: float(coefficients.get(name, 0.0)) for name in allowed}
    d = np.zeros((3, 6))
    if crystal_class == 'general':
        for name, entry in value.items():
            d[int(name[1]) - 1, int(name[2]) - 1] = entry
    elif crystal_class == '3m':
        d[0, 4] = d[1, 3] = value['d31']   # d15 = d24 = d31
        d[0, 5] = d[1, 0] = -value['d22']
        d[1, 1] = value['d22']
        d[2, 0] = d[2, 1] = value['d31']
        d[2, 2] = value['d33']
    elif crystal_class == '-43m':
        d[0, 3] = d[1, 4] = d[2, 5] = value['d14']
    elif crystal_class == '6mm':
        d[0, 4] = d[1, 3] = value['d15']
        d[2, 0] = d[2, 1] = value['d31']
        d[2, 2] = value['d33']
    elif crystal_class == 'mm2':
        d[0, 4] = value['d15']
        d[1, 3] = value['d24']
        d[2, 0], d[2, 1], d[2, 2] = value['d31'], value['d32'], value['d33']
    else:   # '32'
        d[0, 0] = value['d11']
        d[0, 1] = d[1, 5] = -value['d11']
        d[0, 3] = value['d14']
        d[1, 4] = -value['d14']
    return d


def chi2_from_voigt(d):
    """(3, 3, 3) chi(2)_ijk = 2 d_il, l = Voigt index of (j, k) (symmetric in j, k) [m/V]."""
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (3, 6):
        raise ValueError(f"Voigt d must have shape (3, 6), got {d.shape}")
    chi2 = np.zeros((3, 3, 3))
    for l, (j, k) in enumerate(VOIGT_PAIRS):
        chi2[:, j, k] = chi2[:, k, j] = 2.0 * d[:, l]
    return chi2


def euler_rotation(alpha, beta, gamma):
    """(3, 3) rotation crystal -> model axes of z-x-z Euler angles [deg]: Rz(alpha) Rx(beta) Rz(gamma)."""
    def about_z(angle):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    c, s = math.cos(math.radians(beta)), math.sin(math.radians(beta))
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return about_z(alpha) @ about_x @ about_z(gamma)


def rotate_tensor(tensor, rotation):
    """Tensor of any rank in the rotated axes: T'_{ij..} = R_ia R_jb .. T_ab.. (rotation (3, 3) orthogonal)."""
    rotated = np.asarray(tensor)
    rotation = np.asarray(rotation, dtype=np.float64)
    for axis in range(rotated.ndim):
        rotated = np.moveaxis(np.tensordot(rotation, rotated, axes=([1], [axis])), 0, axis)
    return rotated


def symmetrize_sources(tensor):
    """Mean over the permutations of the source indices (all but the first): intrinsic permutation symmetry."""
    tensor = np.asarray(tensor)
    order = tensor.ndim - 1
    permutations = list(itertools.permutations(range(1, order + 1)))
    return sum(np.transpose(tensor, (0, *permutation)) for permutation in permutations) / len(permutations)


def kleinman_asymmetry(tensor):
    """max |T - T permuted| / max |T| over every index permutation (0 = full permutation symmetry; 0 for T = 0)."""
    tensor = np.asarray(tensor)
    scale = float(np.max(np.abs(tensor)))
    if scale == 0.0:
        return 0.0
    deviation = max(float(np.max(np.abs(tensor - np.transpose(tensor, permutation))))
                    for permutation in itertools.permutations(range(tensor.ndim)))
    return deviation / scale


def chi3_isotropic(chi_xxxx):
    """(3, 3, 3, 3) isotropic chi(3) [m^2/V^2]: chi_xxyy = chi_xyxy = chi_xyyx = chi_xxxx / 3 (real or complex)."""
    return chi3_cubic(chi_xxxx, 1.0)


def chi3_cubic(chi_xxxx, anisotropy):
    """(3, 3, 3, 3) cubic chi(3) in crystal axes: chi_iiii = chi_xxxx, chi_iijj = chi_ijij = chi_ijji = anisotropy
    chi_xxxx / 3 (anisotropy rho = 3 chi_xxyy / chi_xxxx; 1 = isotropic, Si ~1.27 at 1.55 um)."""
    chi_xxxx = complex(chi_xxxx)
    pair_value = complex(anisotropy) * chi_xxxx / 3.0
    delta = np.eye(3)
    tensor = pair_value * (np.einsum('ij,kl->ijkl', delta, delta) + np.einsum('ik,jl->ijkl', delta, delta)
                           + np.einsum('il,jk->ijkl', delta, delta))
    for i in range(3):
        tensor[i, i, i, i] = chi_xxxx
    return tensor if tensor.imag.any() else tensor.real


def chi3_from_n2_beta(n2, beta_tpa, background_index, reference_omega):
    """Complex chi_xxxx [m^2/V^2] = (4 eps0 c n0^2 / 3) (n2 - j c beta / (2 omega_ref)); n2 [m^2/W], beta [m/W],
    omega_ref [rad/s] (exp(+j omega t): TPA gives Im chi3 < 0)."""
    reference_omega = float(reference_omega)
    if not (math.isfinite(reference_omega) and reference_omega > 0.0):
        raise ValueError(f"reference angular frequency must be finite and > 0 [rad/s], got {reference_omega!r}")
    scale = 4.0 * EPS0 * C0 * float(background_index) ** 2 / 3.0
    return scale * complex(float(n2), -C0 * float(beta_tpa) / (2.0 * reference_omega))


def n2_beta_from_chi3(chi_xxxx, background_index, reference_omega):
    """(n2 [m^2/W], beta_TPA [m/W]) of a complex chi_xxxx, the inverse of chi3_from_n2_beta."""
    chi_xxxx = complex(chi_xxxx)
    scale = 4.0 * EPS0 * C0 * float(background_index) ** 2 / 3.0
    return chi_xxxx.real / scale, -2.0 * float(reference_omega) * chi_xxxx.imag / (scale * C0)

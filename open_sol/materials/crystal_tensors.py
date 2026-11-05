"""Crystal tensors (numpy only, GUI-thread safe): c^E, e, eps^S, photoelastic p and clamped Pockels r per crystal in
its standard axes, Bond rotations into model axes by SAW Euler angles, and the sagittal (2D) or full (3D) expansion of a
crystal entry."""
# Voigt order 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy; engineering shear strain (S4 = 2 S_yz), so the e_iJ of D = e S equal
# the tensor e_ijk and stiffnesses need no factors. Constitutive law T = c^E S - e^T E, D = e S + eps0 eps^S E.
# Optics (Nye): Delta(1/n^2)_I = p_IJ S_J + r_Ik E_k with B_I tensor components (B_4 = B_yz), so p_IJ = p_ijkl and
# r_Ik = r_ijk without factors; rotations p' = M p M^T, r' = M r R^T (M the Bond stress matrix).
# Euler angles (phi, theta, psi) [deg] in the SAW convention: crystal axes -> substrate axes X' = a X with the passive
# a = Rz(psi) Rx(theta) Rz(phi) (rows = X1' propagation, X2', X3' surface normal); YZ = (0, 90, 90), X-cut / Y-
# propagating = (90, 90, 0), 128 deg Y-X = (0, 38, 0). 'sagittal' (2D models): model (x, y, z) = (X1', X3', -X2');
# 'full' (3D models): model (x, y, z) = (X1', X2', X3'), the film in the xy plane with z the surface normal.
import functools
import math
from dataclasses import dataclass

import numpy as np

CRYSTAL_MODEL = 'crystal'
SAGITTAL_PLANE = 'sagittal'
FULL_PLANE = 'full'
PLANES = (SAGITTAL_PLANE, FULL_PLANE)
ENTRY_KEYS = ('model', 'crystal', 'euler_deg', 'plane')
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
IN_PLANE_VOIGT = (0, 1, 5)     # xx, yy, xy of the model plane
ANTI_PLANE_VOIGT = (3, 4)      # yz, xz
# rows: model x, y, z in substrate axes (X1', X2', X3')
SAGITTAL_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
FULL_AXES = np.eye(3)
# 'full' expansion: the 21 c_IJ (I <= J), the 18 piezo_e<i><J> (D_i vs the Voigt strain J) and the six clamped eps^S
FULL_STIFFNESS_PROPERTIES = tuple(f'c{row}{column}' for row in range(1, 7) for column in range(row, 7))
FULL_PIEZO_PROPERTIES = tuple(f'piezo_e{row}{column}' for row in range(1, 4) for column in range(1, 7))
FULL_PERMITTIVITY_PROPERTIES = {'eps_dc_xx': (0, 0), 'eps_dc_yy': (1, 1), 'eps_dc_zz': (2, 2), 'eps_dc_yz': (1, 2),
                                'eps_dc_xz': (0, 2), 'eps_dc_xy': (0, 1)}
ZERO_RELATIVE = 1e-12          # rotated entries below this fraction of the largest of their tensor are set to 0
MIRROR_RELATIVE = 1e-9         # mirror_violation: sign-flipping entries above this fraction break the mirror symmetry
PIEZO_PROPERTIES = ('piezo_e11', 'piezo_e12', 'piezo_e16', 'piezo_e21', 'piezo_e22', 'piezo_e26')
STIFFNESS_PROPERTIES = ('c11', 'c12', 'c22', 'c66')   # always expanded; c16 / c26 only when either is non-zero
# the zz row (always) and c36 (when non-zero), piezo_e13 / _e23 (always): read only by 2D solid_mechanics
OUT_OF_PLANE_STIFFNESS_PROPERTIES = ('c13', 'c23', 'c33', 'c36')
OUT_OF_PLANE_PIEZO_PROPERTIES = ('piezo_e13', 'piezo_e23')
# photoelastic_p<I><J>: index-ellipsoid row I (1 xx, 2 yy, 3 zz, 6 xy) vs the in-plane Voigt strain J (1, 2, 6); p11 /
# p12 always, the other ten (elastic_waves tensor arrays) only when they differ from the isotropic fallback
PHOTOELASTIC_PROPERTIES = ('photoelastic_p11', 'photoelastic_p12', 'photoelastic_p16', 'photoelastic_p21',
                           'photoelastic_p22', 'photoelastic_p26', 'photoelastic_p31', 'photoelastic_p32',
                           'photoelastic_p36', 'photoelastic_p61', 'photoelastic_p62', 'photoelastic_p66')
PHOTOELASTIC_TENSOR_PROPERTIES = PHOTOELASTIC_PROPERTIES[2:]
# pockels_r<I><k>: row I (1 xx, 2 yy, 3 zz, 6 xy) vs the in-plane field k (1 x, 2 y); only when any is non-zero
POCKELS_PROPERTIES = ('pockels_r11', 'pockels_r12', 'pockels_r21', 'pockels_r22', 'pockels_r31', 'pockels_r32',
                      'pockels_r61', 'pockels_r62')
IN_PLANE_ROWS = (0, 1, 2, 5)    # index-ellipsoid rows the EM slots hold (xy: the em_fdfd_hz eps_xy slot)
SHEAR_ROW = 5                   # xy: dropped by EM types without an eps_xy slot
ANTI_PLANE_ROWS = (3, 4)        # yz, xz rows: always dropped


@dataclass(frozen=True)
class CrystalData:
    """One crystal in its standard (IEEE 1949) axes: class, density [kg/m^3], independent c^E [Pa], e [C/m^2],
    relative clamped eps^S (eps11, eps33), optional photoelastic p [1] and Pockels r [m/V], with the data sources."""
    crystal_class: str
    rho: float
    stiffness: dict
    piezo: dict
    permittivity_clamped: tuple
    reference: str
    photoelastic: dict = None
    pockels: dict = None
    optical_reference: str = ''


# Values transcribed without the source at hand (piezoelectric_plan status): verify against the cited tables. Signs:
# for 3m / 32 the sign of c14 and e22 (3m) / e11, e14 (32) depend on the +Y (+X) choice of the axis standard; only the
# products c14 e22 (3m) are convention independent, e15, e31, e33 are not affected.
CRYSTALS = {
    'lithium_niobate': CrystalData(
        '3m', 4700.0,
        {'c11': 2.03e11, 'c12': 0.53e11, 'c13': 0.75e11, 'c14': 0.09e11, 'c33': 2.45e11, 'c44': 0.60e11},
        {'e15': 3.7, 'e22': 2.5, 'e31': 0.2, 'e33': 1.3}, (44.0, 29.0),
        'Warner, Onoe and Coquin 1967 (as tabulated by Weis and Gaylord 1985), IEEE 1949 axes: mirror normal to X, '
        'c14 > 0 with e22 > 0',
        photoelastic={'p11': -0.026, 'p12': 0.090, 'p13': 0.133, 'p14': -0.075, 'p31': 0.179, 'p33': 0.071,
                      'p41': -0.151, 'p44': 0.146},
        pockels={'r13': 8.6e-12, 'r22': 3.4e-12, 'r33': 30.8e-12, 'r51': 28.0e-12},
        optical_reference='p at 633 nm and clamped (strain-free) r^S at 633 nm, from memory: unverified, check against '
                          'Weis and Gaylord 1985 (Appl. Phys. A 37, 191) and Andrushchak et al. 2009 (p); the signs of '
                          'p14, p41 and r22 flip with the +Y choice like c14 / e22 and must share the axis setting of '
                          'the elastic data'),
    'zinc_oxide': CrystalData(
        '6mm', 5680.0,
        {'c11': 2.097e11, 'c12': 1.211e11, 'c13': 1.051e11, 'c33': 2.109e11, 'c44': 0.4247e11},
        {'e15': -0.59, 'e31': -0.61, 'e33': 1.14}, (8.55, 10.2),
        'Auld, Acoustic Fields and Waves in Solids (1973), wurtzite, +Z the Zn face',
        photoelastic={'p11': 0.222, 'p12': 0.099, 'p13': -0.111, 'p31': 0.088, 'p33': -0.235, 'p44': -0.0585},
        pockels={'r13': -1.4e-12, 'r33': 2.6e-12},
        optical_reference='p and r at 633 nm, from memory (signs least certain; r clamped or not not recorded; r51 not '
                          'in the data set, 0): unverified, check against Dixon 1967 (J. Appl. Phys. 38, 5149) and '
                          'Yariv and Yeh, Optical Waves in Crystals, Table 7.2'),
    'aluminum_nitride': CrystalData(
        '6mm', 3260.0,
        {'c11': 3.45e11, 'c12': 1.25e11, 'c13': 1.20e11, 'c33': 3.95e11, 'c44': 1.18e11},
        {'e15': -0.48, 'e31': -0.58, 'e33': 1.55}, (8.0, 9.5),
        'Tsubouchi and Mikoshiba 1985 (sputtered c-axis films), wurtzite, +Z the Al face',
        optical_reference='no photoelastic or Pockels data entered'),
    'quartz': CrystalData(
        '32', 2648.0,
        {'c11': 0.8674e11, 'c12': 0.0699e11, 'c13': 0.1191e11, 'c14': -0.1791e11, 'c33': 1.0720e11,
         'c44': 0.5794e11},
        {'e11': 0.171, 'e14': -0.0406}, (4.43, 4.63),
        'Bechmann 1958, right-handed alpha quartz, IEEE 1949 axes (X the 2-fold axis)',
        photoelastic={'p11': 0.16, 'p12': 0.27, 'p13': 0.27, 'p14': -0.030, 'p31': 0.29, 'p33': 0.10, 'p41': -0.047,
                      'p44': -0.079},
        optical_reference='p at 589-633 nm, from memory (p14 / p41 signs follow the axis setting): unverified, check '
                          'against Narasimhamurty 1981 (Photoelastic and Electro-Optic Properties of Crystals); no '
                          'Pockels data entered'),
    'gallium_arsenide': CrystalData(
        '-43m', 5307.0, {'c11': 1.188e11, 'c12': 0.538e11, 'c44': 0.594e11}, {'e14': -0.16}, (12.9, 12.9),
        'Auld 1973, cubic axes; the e14 sign follows the [111] Ga face convention',
        photoelastic={'p11': -0.165, 'p12': -0.140, 'p44': -0.072},
        pockels={'r41': 1.43e-12},
        optical_reference='p and r41 at 1.15 um, from memory (the r41 sign relative to e14 not checked): unverified, '
                          'check against Dixon 1967 (J. Appl. Phys. 38, 5149) and Yariv and Yeh, Optical Waves in '
                          'Crystals, Table 7.2'),
}
_STIFFNESS_NAMES = {'3m': ('c11', 'c12', 'c13', 'c14', 'c33', 'c44'), '32': ('c11', 'c12', 'c13', 'c14', 'c33', 'c44'),
                    '6mm': ('c11', 'c12', 'c13', 'c33', 'c44'), '-43m': ('c11', 'c12', 'c44')}
_PIEZO_NAMES = {'3m': ('e15', 'e22', 'e31', 'e33'), '32': ('e11', 'e14'), '6mm': ('e15', 'e31', 'e33'),
                '-43m': ('e14',)}
_PHOTOELASTIC_NAMES = {'3m': ('p11', 'p12', 'p13', 'p14', 'p31', 'p33', 'p41', 'p44'),
                       '32': ('p11', 'p12', 'p13', 'p14', 'p31', 'p33', 'p41', 'p44'),
                       '6mm': ('p11', 'p12', 'p13', 'p31', 'p33', 'p44'), '-43m': ('p11', 'p12', 'p44')}
_POCKELS_NAMES = {'3m': ('r13', 'r22', 'r33', 'r51'), '32': ('r11', 'r41'), '6mm': ('r13', 'r33', 'r51'),
                  '-43m': ('r41',)}


def stiffness_voigt(crystal_class, values):
    """(6, 6) c^E [Pa] of a crystal class from its independent constants (c66 = (c11 - c12) / 2 for 3m / 32 / 6mm)."""
    v = {name: float(values.get(name, 0.0)) for name in _STIFFNESS_NAMES[crystal_class]}
    c = np.zeros((6, 6))
    if crystal_class == '-43m':
        c[0, 0] = c[1, 1] = c[2, 2] = v['c11']
        c[0, 1] = c[0, 2] = c[1, 2] = v['c12']
        c[3, 3] = c[4, 4] = c[5, 5] = v['c44']
    else:
        c[0, 0] = c[1, 1] = v['c11']
        c[0, 1] = v['c12']
        c[0, 2] = c[1, 2] = v['c13']
        c[2, 2] = v['c33']
        c[3, 3] = c[4, 4] = v['c44']
        c[5, 5] = 0.5 * (v['c11'] - v['c12'])
        c14 = v.get('c14', 0.0)   # trigonal: c24 = -c14, c56 = c14
        c[0, 3], c[1, 3], c[4, 5] = c14, -c14, c14
    return np.triu(c) + np.triu(c, 1).T


def piezo_voigt(crystal_class, values):
    """(3, 6) e [C/m^2] of a crystal class (3m: mirror normal to X; 32: X the 2-fold axis)."""
    v = {name: float(values.get(name, 0.0)) for name in _PIEZO_NAMES[crystal_class]}
    e = np.zeros((3, 6))
    if crystal_class == '3m':
        e[0, 4] = e[1, 3] = v['e15']
        e[0, 5] = e[1, 0] = -v['e22']
        e[1, 1] = v['e22']
        e[2, 0] = e[2, 1] = v['e31']
        e[2, 2] = v['e33']
    elif crystal_class == '6mm':
        e[0, 4] = e[1, 3] = v['e15']
        e[2, 0] = e[2, 1] = v['e31']
        e[2, 2] = v['e33']
    elif crystal_class == '32':
        e[0, 0], e[0, 1], e[0, 3] = v['e11'], -v['e11'], v['e14']
        e[1, 4], e[1, 5] = -v['e14'], -v['e11']
    else:   # -43m
        e[0, 3] = e[1, 4] = e[2, 5] = v['e14']
    return e


def photoelastic_voigt(crystal_class, values):
    """(6, 6) p_IJ [1] of a crystal class (p66 = (p11 - p12) / 2 for 3m / 32 / 6mm; trigonal p24 = -p14, p65 = p14,
    p42 = -p41, p56 = p41); p is not symmetric (p13 != p31)."""
    v = {name: float(values.get(name, 0.0)) for name in _PHOTOELASTIC_NAMES[crystal_class]}
    p = np.zeros((6, 6))
    if crystal_class == '-43m':
        p[:3, :3] = v['p12']
        p[0, 0] = p[1, 1] = p[2, 2] = v['p11']
        p[3, 3] = p[4, 4] = p[5, 5] = v['p44']
        return p
    p[0, 0] = p[1, 1] = v['p11']
    p[0, 1] = p[1, 0] = v['p12']
    p[0, 2] = p[1, 2] = v['p13']
    p[2, 0] = p[2, 1] = v['p31']
    p[2, 2] = v['p33']
    p[3, 3] = p[4, 4] = v['p44']
    p[5, 5] = 0.5 * (v['p11'] - v['p12'])
    p14, p41 = v.get('p14', 0.0), v.get('p41', 0.0)
    p[0, 3], p[1, 3], p[5, 4] = p14, -p14, p14
    p[3, 0], p[3, 1], p[4, 5] = p41, -p41, p41
    return p


def pockels_voigt(crystal_class, values):
    """(6, 3) r_Ik [m/V] of a crystal class (the pattern of e^T: 3m mirror normal to X, 32 X the 2-fold axis)."""
    v = {name: float(values.get(name, 0.0)) for name in _POCKELS_NAMES[crystal_class]}
    r = np.zeros((6, 3))
    if crystal_class in ('3m', '6mm'):
        r[0, 2] = r[1, 2] = v['r13']
        r[2, 2] = v['r33']
        r[3, 1] = r[4, 0] = v['r51']
        if crystal_class == '3m':
            r[0, 1] = r[5, 0] = -v['r22']
            r[1, 1] = v['r22']
    elif crystal_class == '32':
        r[0, 0], r[1, 0], r[5, 1] = v['r11'], -v['r11'], -v['r11']
        r[3, 0], r[4, 1] = v['r41'], -v['r41']
    else:   # -43m
        r[3, 0] = r[4, 1] = r[5, 2] = v['r41']
    return r


def _crystal_data(name):
    """CRYSTALS entry `name`; ValueError naming the known ones."""
    if name not in CRYSTALS:
        raise ValueError(f"unknown crystal {name!r}; known: {', '.join(sorted(CRYSTALS))}")
    return CRYSTALS[name]


def optical_tensors(name):
    """(p (6, 6) or None, r (6, 3) [m/V] or None) of a CRYSTALS entry in crystal axes (None: no data entered)."""
    data = _crystal_data(name)
    return (None if data.photoelastic is None else photoelastic_voigt(data.crystal_class, data.photoelastic),
            None if data.pockels is None else pockels_voigt(data.crystal_class, data.pockels))


def crystal_tensors(name):
    """(c^E (6, 6) [Pa], e (3, 6) [C/m^2], eps^S (3, 3) relative, rho [kg/m^3]) of a CRYSTALS entry in crystal axes."""
    data = _crystal_data(name)
    permittivity = np.diag([data.permittivity_clamped[0], data.permittivity_clamped[0], data.permittivity_clamped[1]])
    return (stiffness_voigt(data.crystal_class, data.stiffness), piezo_voigt(data.crystal_class, data.piezo),
            permittivity, data.rho)


def substrate_rotation(phi, theta, psi):
    """(3, 3) passive a = Rz(psi) Rx(theta) Rz(phi) [deg]: rows = substrate axes X1', X2', X3' in crystal axes."""
    from ..physics.nlfd.tensors import euler_rotation   # lazy: materials import time stays numpy only
    return euler_rotation(phi, theta, psi).T


def bond_matrix(rotation):
    """(6, 6) Bond stress matrix M of an orthogonal rotation R (T'_ij = R_ia R_jb T_ab): T'_I = M_IJ T_J."""
    rotation = np.asarray(rotation, dtype=np.float64)
    bond = np.zeros((6, 6))
    for row, (i, j) in enumerate(VOIGT_PAIRS):
        for column, (k, l) in enumerate(VOIGT_PAIRS):
            bond[row, column] = rotation[i, k] * rotation[j, l]
            if k != l:
                bond[row, column] += rotation[i, l] * rotation[j, k]
    return bond


def rotate_voigt(stiffness, piezo, permittivity, rotation):
    """(c', e', eps') in rotated axes by Bond matrices: c' = M c M^T, e' = R e M^T, eps' = R eps R^T."""
    bond = bond_matrix(rotation)
    rotation = np.asarray(rotation, dtype=np.float64)
    return bond @ stiffness @ bond.T, rotation @ piezo @ bond.T, rotation @ permittivity @ rotation.T


def rotate_optical(photoelastic, pockels, rotation):
    """(p', r') in rotated axes: p' = M p M^T, r' = M r R^T (None stays None)."""
    bond = bond_matrix(rotation)
    rotation = np.asarray(rotation, dtype=np.float64)
    return (None if photoelastic is None else bond @ photoelastic @ bond.T,
            None if pockels is None else bond @ pockels @ rotation.T)


def full_stiffness(stiffness):
    """(3, 3, 3, 3) c_ijkl of a (6, 6) Voigt stiffness (also p_ijkl of a photoelastic p: no symmetry between pairs
    assumed)."""
    full = np.zeros((3, 3, 3, 3))
    for row, (i, j) in enumerate(VOIGT_PAIRS):
        for column, (k, l) in enumerate(VOIGT_PAIRS):
            for a, b in {(i, j), (j, i)}:
                for c, d in {(k, l), (l, k)}:
                    full[a, b, c, d] = stiffness[row, column]
    return full


def full_piezo(piezo):
    """(3, 3, 3) e_ijk of a (3, 6) Voigt e (engineering shear: no factors)."""
    full = np.zeros((3, 3, 3))
    for column, (j, k) in enumerate(VOIGT_PAIRS):
        full[:, j, k] = full[:, k, j] = piezo[:, column]
    return full


def voigt_stiffness(full):
    """(6, 6) Voigt stiffness of a (3, 3, 3, 3) c_ijkl."""
    return np.array([[full[i, j, k, l] for k, l in VOIGT_PAIRS] for i, j in VOIGT_PAIRS])


def voigt_piezo(full):
    """(3, 6) Voigt e of a (3, 3, 3) e_ijk."""
    return np.array([[full[i, j, k] for j, k in VOIGT_PAIRS] for i in range(3)])


def mirror_violation(values, axis) -> tuple:
    """(relative size, name) of the largest 3D entry of `values` (c_IJ, piezo_e*, eps_dc_*: name -> number) that changes
    sign under the reflection axis -> -axis (0 x, 1 y, 2 z), relative to its tensor's largest entry; (0.0, '') when
    every such entry is below MIRROR_RELATIVE (a mirror plane normal to that axis is then a symmetry of the material)."""
    def count(pair):
        return sum(index == axis for index in pair)

    parities = (
        {name: count(VOIGT_PAIRS[int(name[1]) - 1]) + count(VOIGT_PAIRS[int(name[2]) - 1])
         for name in FULL_STIFFNESS_PROPERTIES},
        {name: (int(name[7]) - 1 == axis) + count(VOIGT_PAIRS[int(name[8]) - 1]) for name in FULL_PIEZO_PROPERTIES},
        {name: count(pair) for name, pair in FULL_PERMITTIVITY_PROPERTIES.items()})
    worst = (0.0, '')
    for tensor_parities in parities:
        magnitudes = {name: abs(float(values[name])) for name in tensor_parities if values.get(name) is not None}
        scale = max(magnitudes.values(), default=0.0)
        for name, magnitude in magnitudes.items():
            if tensor_parities[name] % 2 and scale > 0.0 and magnitude / scale > worst[0]:
                worst = (magnitude / scale, name)
    return worst if worst[0] > MIRROR_RELATIVE else (0.0, '')


def _cleaned(array):
    """Copy with entries below ZERO_RELATIVE of the largest |entry| set to 0 (rotation round-off)."""
    array = np.array(array, dtype=np.float64)
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    array[np.abs(array) <= ZERO_RELATIVE * scale] = 0.0
    return array


@dataclass(frozen=True)
class SagittalCrystal:
    """A 'crystal' entry reduced to the sagittal plane: the expanded material properties (c16 / c26 among them only
    when either is non-zero), the in-plane monoclinic c16 / c26 [Pa], the largest anti-plane coupling ratios, the
    dropped anti-plane (yz, xz) and the xy (eps_xy slot only) shares of the photoelastic / Pockels index change."""
    crystal: str
    euler_deg: tuple
    properties: dict
    c16: float
    c26: float
    antiplane_stiffness_ratio: float
    antiplane_piezo_ratio: float
    antiplane_photoelastic_ratio: float = 0.0
    antiplane_pockels_ratio: float = 0.0
    shear_photoelastic_ratio: float = 0.0
    shear_pockels_ratio: float = 0.0

    @property
    def monoclinic_ratio(self) -> float:
        """max(|c16|, |c26|) / max(c11, c22): the share of the in-plane monoclinic terms."""
        return max(abs(self.c16), abs(self.c26)) / max(self.properties['c11'], self.properties['c22'])


def entry_settings(entry):
    """(crystal, (phi, theta, psi), plane) of a crystal entry dict; ValueError for unknown keys or values."""
    if not isinstance(entry, dict) or entry.get('model') != CRYSTAL_MODEL:
        raise ValueError(f"a crystal entry is a dict with 'model': '{CRYSTAL_MODEL}', got {entry!r}")
    unknown = sorted(set(entry) - set(ENTRY_KEYS))
    if unknown:
        raise ValueError(f"crystal entry takes {', '.join(ENTRY_KEYS)}; unknown {unknown}")
    crystal = entry.get('crystal')
    if crystal not in CRYSTALS:
        raise ValueError(f"crystal entry: crystal must be one of {', '.join(sorted(CRYSTALS))}, got {crystal!r}")
    angles = entry.get('euler_deg', (0.0, 0.0, 0.0))
    if (not isinstance(angles, (list, tuple)) or len(angles) != 3
            or not all(isinstance(angle, (int, float)) and not isinstance(angle, bool) and math.isfinite(angle)
                       for angle in angles)):
        raise ValueError(f'crystal entry: euler_deg must be three plain numbers [phi, theta, psi] in degrees (no '
                         f'parameter expressions), got {angles!r}')
    plane = entry.get('plane', SAGITTAL_PLANE)
    if plane not in PLANES:
        raise ValueError(f"crystal entry: plane must be one of {', '.join(PLANES)}, got {plane!r}")
    return crystal, tuple(float(angle) for angle in angles), plane


def model_tensors(crystal, euler_deg, axes=SAGITTAL_AXES):
    """(c, e, eps, rho) of a crystal in model axes (rows: model x, y, z in substrate axes; default the sagittal (x, y,
    z) = (X1', X3', -X2'), FULL_AXES the identity), c and eps exactly symmetric (M c M^T is only to round-off), entries
    cleaned."""
    stiffness, piezo, permittivity, rho = crystal_tensors(crystal)
    rotation = np.asarray(axes, dtype=np.float64) @ substrate_rotation(*euler_deg)
    stiffness, piezo, permittivity = rotate_voigt(stiffness, piezo, permittivity, rotation)
    stiffness, permittivity = 0.5 * (stiffness + stiffness.T), 0.5 * (permittivity + permittivity.T)
    return _cleaned(stiffness), _cleaned(piezo), _cleaned(permittivity), rho


def model_optical_tensors(crystal, euler_deg):
    """(p (6, 6), r (6, 3)) of a crystal in sagittal model axes, entries cleaned; None where no data was entered."""
    photoelastic, pockels = optical_tensors(crystal)
    rotation = SAGITTAL_AXES @ substrate_rotation(*euler_deg)
    photoelastic, pockels = rotate_optical(photoelastic, pockels, rotation)
    return (None if photoelastic is None else _cleaned(photoelastic), None if pockels is None else _cleaned(pockels))


def _row_ratio(tensor, rows, columns):
    """Largest |entry| of `rows` over the largest of all rows, in the given (in-plane) columns."""
    block = np.abs(tensor[:, list(columns)])
    largest = float(np.max(block))
    return float(np.max(block[list(rows)])) / largest if largest > 0.0 else 0.0


def _photoelastic_properties(photoelastic):
    """photoelastic_p11 / _p12, and the ten tensor entries when any differs from its isotropic fallback (p22 = p11,
    p21 = p31 = p32 = p12, p66 = (p11 - p12) / 2, p16 = p26 = p36 = p61 = p62 = 0)."""
    def entry(name):
        return photoelastic[int(name[-2]) - 1, int(name[-1]) - 1]
    properties = {name: entry(name) for name in PHOTOELASTIC_PROPERTIES[:2]}
    p11, p12 = properties['photoelastic_p11'], properties['photoelastic_p12']
    fallback = {'photoelastic_p16': 0.0, 'photoelastic_p21': p12, 'photoelastic_p22': p11, 'photoelastic_p26': 0.0,
                'photoelastic_p31': p12, 'photoelastic_p32': p12, 'photoelastic_p36': 0.0, 'photoelastic_p61': 0.0,
                'photoelastic_p62': 0.0, 'photoelastic_p66': 0.5 * (p11 - p12)}
    tolerance = ZERO_RELATIVE * float(np.max(np.abs(photoelastic)))
    if any(abs(entry(name) - fallback[name]) > tolerance for name in PHOTOELASTIC_TENSOR_PROPERTIES):
        properties.update({name: entry(name) for name in PHOTOELASTIC_TENSOR_PROPERTIES})
    return properties


def sagittal_crystal(entry):
    """SagittalCrystal of a crystal entry (cached per crystal / angles); ValueError for a malformed entry or plane
    'full' (the 3D expansion: full_crystal)."""
    crystal, euler_deg, plane = entry_settings(entry)
    if plane != SAGITTAL_PLANE:
        raise ValueError(f"crystal entry: plane '{plane}' is the 3D expansion; a 2D model takes 'plane': "
                         f"'{SAGITTAL_PLANE}'")
    return _sagittal(crystal, euler_deg)


@functools.lru_cache(maxsize=None)
def _sagittal(crystal, euler_deg):
    """SagittalCrystal of a crystal and its Euler angles (cached: a tuple of floats)."""
    stiffness, piezo, permittivity, rho = model_tensors(crystal, euler_deg)
    properties = {'rho': float(rho), 'c11': stiffness[0, 0], 'c12': stiffness[0, 1], 'c22': stiffness[1, 1],
                  'c66': stiffness[5, 5]}
    if stiffness[0, 5] != 0.0 or stiffness[1, 5] != 0.0:   # monoclinic in-plane terms (they allocate arrays)
        properties.update(c16=stiffness[0, 5], c26=stiffness[1, 5])
    # the zz row and column (2D solid_mechanics only: stress_zz, plane stress, thermal stress)
    properties.update(c13=stiffness[0, 2], c23=stiffness[1, 2], c33=stiffness[2, 2])
    if stiffness[2, 5] != 0.0:
        properties['c36'] = stiffness[2, 5]
    for name in PIEZO_PROPERTIES + OUT_OF_PLANE_PIEZO_PROPERTIES:   # piezo_e<k><J>: D_k vs the Voigt strain J
        properties[name] = piezo[int(name[-2]) - 1, int(name[-1]) - 1]
    properties.update(eps_dc_xx=permittivity[0, 0], eps_dc_yy=permittivity[1, 1], eps_dc_xy=permittivity[0, 1])
    photoelastic, pockels = model_optical_tensors(crystal, euler_deg)
    ratios = [0.0, 0.0, 0.0, 0.0]   # anti-plane photoelastic, anti-plane Pockels, xy photoelastic, xy Pockels
    if photoelastic is not None:   # strains S_xx, S_yy, 2 S_xy only (plane strain, u_z clamped)
        properties.update(_photoelastic_properties(photoelastic))
        ratios[0] = _row_ratio(photoelastic, ANTI_PLANE_ROWS, IN_PLANE_VOIGT)
        ratios[2] = _row_ratio(photoelastic, (SHEAR_ROW,), IN_PLANE_VOIGT)
    if pockels is not None:        # fields E_x, E_y only (E_z = 0)
        if np.any(pockels[list(IN_PLANE_ROWS), :2] != 0.0):
            properties.update({name: pockels[int(name[-2]) - 1, int(name[-1]) - 1] for name in POCKELS_PROPERTIES})
        ratios[1] = _row_ratio(pockels, ANTI_PLANE_ROWS, (0, 1))
        ratios[3] = _row_ratio(pockels, (SHEAR_ROW,), (0, 1))
    in_plane, anti_plane = list(IN_PLANE_VOIGT), list(ANTI_PLANE_VOIGT)
    stiffness_scale = float(np.max(np.abs(stiffness)))
    piezo_scale = float(np.max(np.abs(piezo)))
    return SagittalCrystal(
        crystal, euler_deg, {name: float(value) for name, value in properties.items()},
        float(stiffness[0, 5]), float(stiffness[1, 5]),
        float(np.max(np.abs(stiffness[np.ix_(in_plane, anti_plane)]))) / stiffness_scale,
        float(np.max(np.abs(piezo[np.ix_([0, 1], anti_plane)]))) / piezo_scale if piezo_scale > 0.0 else 0.0,
        *ratios)


@dataclass(frozen=True, eq=False)
class FullCrystal:
    """A 'crystal' entry expanded in 3D model axes (x, y, z) = (X1', X2', X3'): the material properties (rho, the 21
    c_IJ, 18 piezo_e<i><J>, six eps_dc_*; no optics) and the cleaned tensors c (6, 6) [Pa], e (3, 6) [C/m^2], eps^S
    (3, 3)."""
    crystal: str
    euler_deg: tuple
    properties: dict
    stiffness: np.ndarray
    piezo: np.ndarray
    permittivity: np.ndarray


def full_crystal(entry):
    """FullCrystal of a crystal entry with plane 'full' (cached per crystal / angles); ValueError for a malformed entry
    or another plane."""
    crystal, euler_deg, plane = entry_settings(entry)
    if plane != FULL_PLANE:
        raise ValueError(f"crystal entry: plane '{plane}' is the 2D sagittal reduction; a 3D model takes 'plane': "
                         f"'{FULL_PLANE}'")
    return _full(crystal, euler_deg)


@functools.lru_cache(maxsize=None)
def _full(crystal, euler_deg):
    """FullCrystal of a crystal and its Euler angles (cached)."""
    stiffness, piezo, permittivity, rho = model_tensors(crystal, euler_deg, FULL_AXES)
    properties = {'rho': float(rho)}
    properties.update({name: float(stiffness[int(name[1]) - 1, int(name[2]) - 1])
                       for name in FULL_STIFFNESS_PROPERTIES})
    properties.update({name: float(piezo[int(name[-2]) - 1, int(name[-1]) - 1]) for name in FULL_PIEZO_PROPERTIES})
    properties.update({name: float(permittivity[pair]) for name, pair in FULL_PERMITTIVITY_PROPERTIES.items()})
    return FullCrystal(crystal, euler_deg, properties, stiffness, piezo, permittivity)


def crystal_properties(entry):
    """Material properties a crystal entry expands to: the 'sagittal' (2D) or 'full' (3D) set by its plane; ValueError
    for a malformed entry."""
    _, _, plane = entry_settings(entry)
    return (full_crystal(entry) if plane == FULL_PLANE else sagittal_crystal(entry)).properties


def is_crystal_entry(entry) -> bool:
    """True for a dispersion-list dict with 'model': 'crystal'."""
    return isinstance(entry, dict) and entry.get('model') == CRYSTAL_MODEL

"""Self-test of the crystal photoelastic tensor in the g0 overlap (optomechanics.py): the em_modes / em_fdfd_hz /
em_fdfd_ez photoelastic sums of a rotated lithium niobate region beside isotropic cells against an independent
four-index contraction (CPU); physics and materials modules only inside the check."""
import math
import types

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import no_runs
from ...model.spec import MaterialSpec

_CELLS = (6, 4)
_CELL = 20e-9                          # m
_CRYSTAL_COLUMNS = 3                   # cells i < 3 hold the crystal, the rest the isotropic p11 / p12 (p44 unset)
_CRYSTAL = 'lithium_niobate'
_EULER_DEG = (20.0, 50.0, 70.0)        # generic cut: every rotated p entry non-zero
_P11, _P12 = 0.121, 0.270              # isotropic cells
_EPS = (2.1, 2.4, 2.7)                 # eps_xx, eps_yy, eps_zz (em_modes, hz: xx / yy)
_EZ_EPS = 2.25                         # em_fdfd_ez (one eps)
_FIELD = (0.8 + 0.1j, -0.3 + 0.5j, 0.2 - 0.6j)   # (E_x, E_y, E_z) [V/m]
_WAVELENGTH = 1.55e-6                  # m
_WAVENUMBER = 3.0e6                    # rad/m, backward k_z
# (constant [m], d/dx, d/dy) of u_x, u_y, w
_COEFFICIENTS = ((1e-11, 2e-4, 1e-4), (-2e-11, -0.5e-4, 3e-4), (3e-11, 1.5e-4, -2e-4))
_TOLERANCE = 1e-10                     # relative (rotated entries below 1e-12 of the largest are cleaned to 0)


def _full_tensors():
    """(crystal p_ijkl rotated into model axes by an explicit four-index rotation, isotropic p_ijkl)."""
    from ...materials import crystal_tensors
    data = crystal_tensors.CRYSTALS[_CRYSTAL]
    crystal_axes = crystal_tensors.full_stiffness(crystal_tensors.photoelastic_voigt(data.crystal_class,
                                                                                     data.photoelastic))
    rotation = crystal_tensors.SAGITTAL_AXES @ crystal_tensors.substrate_rotation(*_EULER_DEG)
    rotated = np.einsum('ia,jb,kc,ld,abcd->ijkl', rotation, rotation, rotation, rotation, crystal_axes)
    isotropic = np.zeros((6, 6))
    isotropic[:3, :3] = _P12
    isotropic[np.arange(3), np.arange(3)] = _P11
    isotropic[np.arange(3, 6), np.arange(3, 6)] = 0.5 * (_P11 - _P12)
    return rotated, crystal_tensors.full_stiffness(isotropic)


def _cell_contraction(eps, pump, stokes, tensor, gradient):
    """eps0 E_s^H Delta eps E_p of one cell: Delta(1/eps)_ij = p_ijkl S_kl of gradient[i][j] = d u_j / d x_i,
    Delta eps = -eps Delta(1/eps) eps (diagonal eps)."""
    strain = 0.5 * (gradient + gradient.T)
    permittivity = np.diag(eps)
    change = -permittivity @ np.einsum('ijkl,kl->ij', tensor, strain) @ permittivity
    return EPS0 * complex(np.conj(stokes) @ change @ pump)


def _gradient(centre, backward):
    """(3, 3) d u_j / d x_i at a cell centre: plane strain (u_x, u_y), or the guided (u_x, u_y, -i w), d/dz = i k_z."""
    slopes = [(slope_x, slope_y) for _, slope_x, slope_y in _COEFFICIENTS]
    if not backward:
        return np.array([[slopes[0][0], slopes[1][0], 0.0], [slopes[0][1], slopes[1][1], 0.0], [0.0, 0.0, 0.0]],
                        dtype=np.complex128)
    amplitude = np.array([centre[0], centre[1], -1j * centre[2]])
    return np.array([[slopes[0][0], slopes[1][0], -1j * slopes[2][0]],
                     [slopes[0][1], slopes[1][1], -1j * slopes[2][1]],
                     1j * _WAVENUMBER * amplitude])


def _expected(eps, field, backward, tensors):
    """sum over the cells of the contraction (crystal tensor on i < _CRYSTAL_COLUMNS) times the cell area."""
    rotated, isotropic = tensors
    pump = np.asarray(field, dtype=np.complex128)
    stokes = pump * np.array([1.0, 1.0, -1.0]) if backward else pump
    total = 0.0j
    for i in range(_CELLS[0]):
        for j in range(_CELLS[1]):
            centre = [constant + slope_x * (i + 0.5) * _CELL + slope_y * (j + 0.5) * _CELL
                      for constant, slope_x, slope_y in _COEFFICIENTS]
            tensor = rotated if i < _CRYSTAL_COLUMNS else isotropic
            total += _CELL ** 2 * _cell_contraction(eps, pump, stokes, tensor, _gradient(centre, backward))
    return total


def _overlaps(backward):
    """{path: (OpticalOverlap, eps (xx, yy, zz), field (E_x, E_y, E_z))} on the periodic grid (backward: em_modes)."""
    from ..em import cross_section_modes
    from ..em.boundaries import BoundaryConfig
    from . import optomechanics
    n_x, n_y = _CELLS
    dx, dy = np.full(n_x, _CELL), np.full(n_y, _CELL)
    ones = np.ones(_CELLS, dtype=np.complex128)
    window = (slice(0, n_x), slice(0, n_y))
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    materials = cross_section_modes.CrossSectionMaterials(eps_xx=_EPS[0] * ones, eps_yy=_EPS[1] * ones,
                                                          eps_zz=_EPS[2] * ones, mu=ones,
                                                          pec=np.zeros(_CELLS, dtype=bool))
    points = cross_section_modes.mode_point_materials(
        (dx, dy), BoundaryConfig('periodic', 'periodic', 'periodic', 'periodic'), _CELL, materials)
    area = np.outer(dx, dy)
    mode = types.SimpleNamespace(ex=_FIELD[0] * ones, ey=_FIELD[1] * ones, ez=_FIELD[2] * ones, x_face_area=area,
                                 y_face_area=area, power=1.0, omega=omega, mirror_factor=1.0, n_eff=1.5)
    overlaps = {'cross_section': (optomechanics.cross_section_overlap(mode, points, window, backward=backward), _EPS,
                                  _FIELD)}
    if not backward:
        eps_xx, eps_yy = _EPS[0] * ones, _EPS[1] * ones
        overlaps['hz'] = (optomechanics.in_plane_hz_overlap(ones, _FIELD[0] * ones, _FIELD[1] * ones, eps_xx, eps_yy,
                                                            eps_xx, eps_yy, ones, (dx, dy), window, (True, True),
                                                            omega), _EPS, (_FIELD[0], _FIELD[1], 0.0))
        overlaps['ez'] = (optomechanics.in_plane_ez_overlap(_FIELD[2] * ones, _EZ_EPS * ones, dx, dy, omega),
                          (_EZ_EPS,) * 3, (0.0, 0.0, _FIELD[2]))
    return overlaps


def _definition_violations(model_tensor):
    """crystal_photoelastic_tensor of MaterialSpecs: the entry's model tensor, None with a user photoelastic_p11."""
    from .definition_optomechanics import crystal_photoelastic_tensor
    entry = {'model': 'crystal', 'crystal': _CRYSTAL, 'euler_deg': list(_EULER_DEG), 'plane': 'sagittal'}
    violations = []
    tensor = crystal_photoelastic_tensor(MaterialSpec('crystal', dispersion=[entry]))
    if tensor is None or not np.array_equal(tensor, model_tensor):
        violations.append('crystal_photoelastic_tensor differs from model_optical_tensors')
    if crystal_photoelastic_tensor(MaterialSpec('user', properties={'photoelastic_p11': 0.1},
                                                dispersion=[entry])) is not None:
        violations.append('a user photoelastic_p11 must switch the crystal tensor off')
    return violations


def _check_crystal_contraction(results_by_name):
    """Forward (em_modes, hz, ez) and backward (em_modes) photoelastic sums with a rotated LN region vs the
    four-index contraction; definition helper checks counted as violations."""
    from ...materials.crystal_tensors import model_optical_tensors
    from . import optomechanics
    n_x, n_y = _CELLS
    dx, dy = np.full(n_x, _CELL), np.full(n_y, _CELL)
    x_nodes, y_nodes = np.meshgrid(_CELL * np.arange(n_x + 1), _CELL * np.arange(n_y + 1), indexing='ij')
    nodal = np.stack([constant + slope_x * x_nodes + slope_y * y_nodes
                      for constant, slope_x, slope_y in _COEFFICIENTS], axis=-1)
    solid = np.ones(_CELLS, dtype=bool)
    p11, p12, p44 = np.full(_CELLS, _P11), np.full(_CELLS, _P12), np.full(_CELLS, np.nan)
    model_tensor = model_optical_tensors(_CRYSTAL, _EULER_DEG)[0]
    mask = np.zeros(_CELLS, dtype=bool)
    mask[:_CRYSTAL_COLUMNS, :] = True
    tensors = _full_tensors()
    errors, details = {}, {}
    for backward in (False, True):
        shape = nodal if backward else nodal[..., :2]
        wavenumber = _WAVENUMBER if backward else None
        for path, (overlap, eps, field) in _overlaps(backward).items():
            moving, photo = optomechanics.overlap_integrals(overlap, shape, solid, p11, p12, dx, dy, p44, wavenumber,
                                                            [(mask, model_tensor)])
            expected = _expected(eps, field, backward, tensors)
            key = f"{path} {'backward' if backward else 'forward'}"
            errors[key] = abs(photo - expected) / abs(expected) + abs(moving) / abs(expected)
            details[key] = {'measured': [photo.real, photo.imag], 'expected': [expected.real, expected.imag]}
    violations = _definition_violations(model_tensor)
    worst = max(errors.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst + len(violations), 'tolerance': _TOLERANCE,
            'details': dict(details, errors=errors, violations=violations)}


CRYSTAL_CASES = (
    SelfTestCase(
        name='om_crystal_photoelastic_contraction',
        label='Optomechanics: crystal photoelastic tensor vs the full contraction',
        description=f'CPU: {_CELLS[0]} x {_CELLS[1]} periodic cells ({_CELL * 1e9:g} nm), the first {_CRYSTAL_COLUMNS} '
                    f'columns a {_CRYSTAL} region (crystal_tensors p rotated to the model axes of euler_deg '
                    f'{list(_EULER_DEG)}), the rest isotropic (p11 {_P11:g}, p12 {_P12:g}, p44 unset); a uniform '
                    f'complex E = {_FIELD}, a linear nodal shape (plane strain (u_x, u_y); backward (u_x, u_y, w) at '
                    f'k_z = {_WAVENUMBER:g} rad/m). The photoelastic sum of the em_modes overlap (forward and '
                    f'backward, eps {_EPS}), the em_fdfd_hz overlap (E_x, E_y) and the em_fdfd_ez overlap (E_z, eps '
                    f'{_EZ_EPS:g}) must equal sum eps0 E_s^H Delta eps E_p dA with Delta(1/eps)_ij = p_ijkl S_kl of '
                    'the crystal tensor rotated by an explicit four-index rotation (not the Bond matrices; p not '
                    'symmetric: p14 != p41), Delta eps = -eps Delta(1/eps) eps, E_s = E_p forward, (E_x, E_y, -E_z) '
                    f'backward; moving boundary 0 (relative {_TOLERANCE:g}). Also crystal_photoelastic_tensor of a '
                    'MaterialSpec = model_optical_tensors, None when the material sets photoelastic_p11. Pins the '
                    'Voigt factors, the xz / yz rows with E_z (forward 2 Re, backward 2 i Im) and the crystal axes.',
        specs=no_runs, check=_check_crystal_contraction),
)

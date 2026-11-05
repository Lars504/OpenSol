"""Self-tests of the Brillouin gain column of optomechanical_coupling (optomechanics.py): backward scattering against the
bulk plane-wave g_B (CPU; the backward tensor contraction: selftests_optomechanics_crystal), the forward intramodal
column against eigen-solve differences (CPU), and the backward model path of the optomechanics study (GPU); physics
modules only inside checks."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import no_runs
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_optomechanics import (_FD_STEP, _P11, _P12, _SLAB_PERIOD, _TWIN_CELL, _TWIN_CLADDING,
                                      _TWIN_HALF_WIDTH, _WAVELENGTH, _host_csr, _slab_layout, _slab_materials,
                                      _slab_mode)

_QUALITY = 500.0                       # mechanical Q_m of every gain
_LENGTH = 1e-3                         # m, optomechanical_coupling length (G_B does not depend on it)
_ELASTIC = (1.0e11, 0.25, 3000.0)      # E [Pa], nu, rho [kg/m^3]

# ---- uniform guide: every face periodic (optics), rollers on every face (solid); E_x plane wave of n_x (eps_xx above
# eps_yy = eps_zz: a single fundamental mode), the uniform-w longitudinal phonon at k_z = 2 beta exact in Q1 ----
_BULK_CELLS = (12, 8)
_BULK_CELL = 20e-9                     # m
_BULK_INDICES = (2.2, 2.0)             # n_x, n_o (eps_yy = eps_zz)
_BULK_TOLERANCE = 1e-9                 # relative (exact discrete case; eigenvector round-off)
_RESIDUAL_TOLERANCE = 1e-12            # K Q - Omega^2 M Q of the prescribed phonon (relative)

# ---- forward: TE slab (x periodic one cell, y pec) with the u_y = sin(2 pi j / N) roller mode of the whole line (odd
# about the slab centre, so S_yy is even and overlaps TE0; sin(pi j / N) gives S_yy odd and a zero overlap) ----
_FORWARD_TOLERANCE = 1e-6              # relative (central difference of two mode solves)
_FORWARD_HARMONIC = 2                  # m of u_y = sin(m pi j / N)
_FORWARD_RESIDUAL_TOLERANCE = 1e-11    # relative: K Q cancels to ~ (1 - cos theta), round-off ~ 1e-16 / theta^2

# ---- GPU chain: the uniform guide as a model (em_modes + solid_mechanics, optomechanics study, backward) ----
_CHAIN_TOLERANCE = 1e-5                # relative: float32 materials (rho, p12, eps) and outputs
_CHAIN_TARGET_OFFSET = 1.003           # target_frequency / f_L (never on the resonance: singular factor)
_CHAIN_MODES = 3


def _elastic_constants():
    """(lambda, mu, c11) [Pa] of _ELASTIC."""
    youngs, poisson, _ = _ELASTIC
    mu = youngs / (2.0 * (1.0 + poisson))
    lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return lame_lambda, mu, lame_lambda + 2.0 * mu


def _bulk_gain(omega, index, speed, area):
    """Bulk backward SBS per area [1/(W m)]: g_B / A = Q_m omega n^6 p12^2 / (2 c^2 rho v^2 A), = 2 pi n^7 p12^2 / (c
    lambda^2 rho v Delta nu) / A with Delta nu = Omega / (2 pi Q_m), Omega = 2 n omega v / c."""
    density = _ELASTIC[2]
    return _QUALITY * omega * index ** 6 * _P12 ** 2 / (2.0 * C0 ** 2 * density * speed ** 2 * area)


def _periodic_config():
    from ..em.boundaries import BoundaryConfig
    return BoundaryConfig('periodic', 'periodic', 'periodic', 'periodic')


def _uniform_materials(cross_section_modes, shape, eps_xx, eps_yy, eps_zz):
    """CrossSectionMaterials of uniform diagonal eps on `shape` cells."""
    ones = np.ones(shape, dtype=np.complex128)
    return cross_section_modes.CrossSectionMaterials(eps_xx=eps_xx * ones, eps_yy=eps_yy * ones, eps_zz=eps_zz * ones,
                                                     mu=ones, pec=np.zeros(shape, dtype=bool))


def _uniform_phonon(node_shape):
    """(flat nodal (u_x, u_y, w) = (0, 0, 1) [m], free dofs: rollers on all faces hold u_x on x faces, u_y on y)."""
    nodal = np.zeros((*node_shape, 3))
    nodal[..., 2] = 1.0
    free = np.ones((*node_shape, 3), dtype=bool)
    free[0, :, 0] = free[-1, :, 0] = False
    free[:, 0, 1] = free[:, -1, 1] = False
    return nodal.reshape(-1), free.reshape(-1)


def _check_bulk(results_by_name):
    """Backward G_B of the E_x plane wave and the uniform-w phonon at k_z = 2 beta vs g_B / A."""
    from ..em import cross_section_modes
    from . import optomechanics
    from .guided_modes import guided_operators
    n_x, n_y = _BULK_CELLS
    cell = _BULK_CELL
    dx, dy = np.full(n_x, cell), np.full(n_y, cell)
    index_x, index_o = _BULK_INDICES
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    materials = _uniform_materials(cross_section_modes, _BULK_CELLS, index_x ** 2, index_o ** 2, index_o ** 2)
    config = _periodic_config()
    solve = cross_section_modes.solve_cross_section_modes((dx, dy), config, cell, materials, omega, 2)
    mode = solve.modes[0]
    points = cross_section_modes.mode_point_materials((dx, dy), config, cell, materials)
    window = (slice(0, n_x), slice(0, n_y))
    overlap = optomechanics.cross_section_overlap(mode, points, window, backward=True)

    wavenumber = 2.0 * complex(mode.beta).real
    lame_lambda, mu, c11 = _elastic_constants()
    density = _ELASTIC[2]
    stiffness, mass = guided_operators(np.full(_BULK_CELLS, lame_lambda), np.full(_BULK_CELLS, mu),
                                       np.full(_BULK_CELLS, density), dx, dy, wavenumber)
    node_shape = (n_x + 1, n_y + 1)
    phonon, free = _uniform_phonon(node_shape)
    omega_mechanical = math.sqrt(c11 / density) * wavenumber
    residual = (stiffness @ phonon - omega_mechanical ** 2 * (mass @ phonon))[free]
    residual = float(np.linalg.norm(residual) / np.linalg.norm((stiffness @ phonon)[free]))
    solid = np.ones(_BULK_CELLS, dtype=bool)
    row = optomechanics.mode_coupling_row(
        1, overlap, omega_mechanical, phonon, node_shape, mass, solid, np.full(_BULK_CELLS, _P11),
        np.full(_BULK_CELLS, _P12), dx, dy, _LENGTH, quality_factor=_QUALITY, axial_wavenumber=wavenumber)
    columns = optomechanics.OPTOMECHANICS_TABLE_COLUMNS
    measured = row[columns.index('brillouin_gain')]
    expected = _bulk_gain(omega, index_x, math.sqrt(c11 / density), n_x * n_y * cell ** 2)
    index_error = abs(complex(mode.n_eff).real / index_x - 1.0)
    error = abs(measured / expected - 1.0) + index_error
    violations = [] if residual <= _RESIDUAL_TOLERANCE else [f'phonon residual {residual:.3g}']
    return {'measured': measured, 'expected': expected, 'error': error + len(violations),
            'tolerance': _BULK_TOLERANCE,
            'details': {'n_eff': complex(mode.n_eff).real, 'te_fraction': mode.te_fraction,
                        'axial_wavenumber': wavenumber, 'mechanical_frequency': omega_mechanical / (2.0 * math.pi),
                        'phonon_residual': residual, 'max_abs_ez': float(np.max(np.abs(mode.ez))),
                        'g_om_moving_boundary': row[columns.index('g_om_moving_boundary')],
                        'violations': violations}}


def _check_forward(results_by_name):
    """Forward brillouin_gain of the TE slab and the u_y = sin(2 pi j / N) roller mode (photoelastic only) vs 2 Q_m
    kappa^2 / (omega m' Omega^2) of a central-difference kappa, the 1D Q1 symbol Omega and the analytic Q^T M Q."""
    from ..em import cross_section_modes
    from . import optomechanics
    from .system import mass_stencil, stiffness_stencil
    widths_y, eps_line, _, _ = _slab_layout(_TWIN_CELL, _TWIN_HALF_WIDTH, _TWIN_CLADDING)
    n_y = widths_y.size
    cell_widths = (np.array([_SLAB_PERIOD]), widths_y)
    mode, points = _slab_mode(_slab_materials(cross_section_modes, eps_line, eps_line, eps_line), cell_widths, 'te')
    overlap = optomechanics.cross_section_overlap(mode, points, (slice(0, 1), slice(0, n_y)))

    lame_lambda, mu, c11 = _elastic_constants()
    density = _ELASTIC[2]
    shape = (1, n_y)
    stiffness = _host_csr(stiffness_stencil(np.full(shape, c11), np.full(shape, lame_lambda), np.full(shape, c11),
                                            np.full(shape, mu), cell_widths[0], widths_y, np.float64, xp=np))
    mass = _host_csr(mass_stencil(np.full(shape, density), cell_widths[0], widths_y, np.float64, xp=np))
    node_shape = (2, n_y + 1)
    nodal = np.zeros((*node_shape, 2))
    theta, cell = _FORWARD_HARMONIC * math.pi / n_y, _TWIN_CELL
    nodal[:, :, 1] = np.sin(theta * np.arange(n_y + 1))[None, :]
    free = np.ones((*node_shape, 2), dtype=bool)
    free[..., 0] = False                 # u_x held (x-invariant plane strain)
    free[:, 0, 1] = free[:, -1, 1] = False   # rollers on the y ends
    stiffness_symbol, mass_symbol = 2.0 / cell * (1.0 - math.cos(theta)), cell / 3.0 * (2.0 + math.cos(theta))
    omega_mechanical = math.sqrt(c11 * stiffness_symbol / (density * mass_symbol))
    flat, free = nodal.reshape(-1), free.reshape(-1)
    residual = (stiffness @ flat - omega_mechanical ** 2 * (mass @ flat))[free]
    residual = float(np.linalg.norm(residual) / np.linalg.norm((stiffness @ flat)[free]))
    peak = float(np.max(np.abs(nodal)))   # 1 when 2 m divides N (row scales the peak node to 1)
    mass_per_length = density * _SLAB_PERIOD * mass_symbol * n_y / 2.0 / peak ** 2   # Q^T M Q of the peak-scaled Q

    solid = np.ones(shape, dtype=bool)
    p11, p12 = np.full(shape, _P11), np.full(shape, _P12)
    row = optomechanics.mode_coupling_row(1, overlap, omega_mechanical, flat, node_shape, mass, solid, p11, p12,
                                          cell_widths[0], widths_y, _LENGTH, moving_boundary=False,
                                          p44=np.zeros(shape), quality_factor=_QUALITY)
    strain_xx, strain_yy, _ = optomechanics.element_strains(nodal, cell_widths[0], widths_y)
    changes = [values[0] for values in optomechanics.photoelastic_changes(
        eps_line[None, :], eps_line[None, :], eps_line[None, :], p11, p12, strain_xx, strain_yy)]
    alpha = _FD_STEP / max(float(np.max(np.abs(values / eps_line))) for values in changes)
    betas = []
    for sign in (1.0, -1.0):
        shifted = [eps_line + sign * alpha * values for values in changes]
        betas.append(complex(_slab_mode(_slab_materials(cross_section_modes, *shifted), cell_widths, 'te')[0].beta).real)
    kappa = (betas[0] - betas[1]) / (2.0 * alpha)
    expected = 2.0 * _QUALITY * kappa ** 2 / (mode.omega * mass_per_length * omega_mechanical ** 2)
    columns = optomechanics.OPTOMECHANICS_TABLE_COLUMNS
    measured = row[columns.index('brillouin_gain')]
    error = abs(measured / expected - 1.0)
    violations = [] if residual <= _FORWARD_RESIDUAL_TOLERANCE else [f'mode residual {residual:.3g}']
    return {'measured': measured, 'expected': expected, 'error': error + len(violations),
            'tolerance': _FORWARD_TOLERANCE,
            'details': {'kappa_central_difference': kappa, 'alpha': alpha, 'mode_residual': residual,
                        'effective_mass_per_length': row[columns.index('effective_mass')] / _LENGTH,
                        'expected_mass_per_length': mass_per_length,
                        'mechanical_frequency': omega_mechanical / (2.0 * math.pi), 'violations': violations}}


BRILLOUIN_CASES = (
    SelfTestCase(
        name='om_backward_brillouin_bulk',
        label='Brillouin: backward gain of a uniform guide vs bulk g_B',
        description=f'CPU: {_BULK_CELLS[0]} x {_BULK_CELLS[1]} periodic cells ({_BULK_CELL * 1e9:g} nm) of eps_xx = '
                    f'{_BULK_INDICES[0]:g}^2, eps_yy = eps_zz = {_BULK_INDICES[1]:g}^2 (cross_section_modes: the E_x '
                    'plane wave, beta = k0 n_x exactly, E_z = 0); the guided Q1 pencil (guided_modes.guided_operators, '
                    f'E {_ELASTIC[0]:g} Pa, nu {_ELASTIC[1]:g}, rho {_ELASTIC[2]:g}) at k_z = 2 Re beta with rollers on '
                    'every face, whose uniform-w longitudinal phonon at Omega = v_L k_z is exact (residual <= '
                    f'{_RESIDUAL_TOLERANCE:g}). The backward brillouin_gain (Q_m {_QUALITY:g}, p12 {_P12:g}) must equal '
                    'the bulk g_B = 2 pi n^7 p12^2 / (c lambda^2 rho v Delta nu) per cross-section area (relative '
                    f'{_BULK_TOLERANCE:g}). Pins S_zz = k_z w in the p12 term, the 3-component mass and peak '
                    'normalisation, the |kappa|^2 gain and the 1 W normalisation.',
        specs=no_runs, check=_check_bulk),
    SelfTestCase(
        name='om_forward_brillouin_gain',
        label='Brillouin: forward intramodal gain column vs eigen-solve differences',
        description='CPU: the x-periodic y-pec TE slab of om_photoelastic_hellmann_feynman (cross_section_modes) and '
                    f'the plane-strain Q1 mode u_y = sin({_FORWARD_HARMONIC} pi j / N) of the whole line (u_x held, '
                    'rollers on the y ends; odd about the slab centre so S_yy overlaps TE0; exact: residual <= '
                    f'{_FORWARD_RESIDUAL_TOLERANCE:g}, Omega from the 1D linear-element symbol, Q^T M Q = rho h_x '
                    f'M(theta) N / 2 analytic). The forward brillouin_gain (photoelastic only, Q_m {_QUALITY:g}) must '
                    'equal 2 Q_m '
                    "kappa^2 / (omega m' Omega^2) with kappa = d beta / d alpha from the central difference of two "
                    f'mode solves with eps +/- alpha Delta eps (relative {_FORWARD_TOLERANCE:g}).',
        specs=no_runs, check=_check_forward),
)


# ---- GPU chain: the uniform guide as a model ----
def _chain_size():
    return _BULK_CELLS[0] * _BULK_CELL, _BULK_CELLS[1] * _BULK_CELL


def _chain_frequency():
    """f_L [Hz] of the uniform-w phonon at k_z = 2 k0 n_x."""
    _, _, c11 = _elastic_constants()
    return math.sqrt(c11 / _ELASTIC[2]) * 2.0 * 2.0 * math.pi * _BULK_INDICES[0] / _WAVELENGTH / (2.0 * math.pi)


def _chain_specs():
    """optomechanics study: em_modes on the periodic uniform guide, solid with rollers, backward coupling."""
    size_x, size_y = _chain_size()
    youngs, poisson, density = _ELASTIC
    em_settings = {'wavelength': _WAVELENGTH, 'n_modes': 2, 'group_index': True, 'bc_x_lo': 'periodic',
                   'bc_x_hi': 'periodic', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'}
    features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi']), FeatureSpec('ends', 'roller', ['y_lo', 'y_hi']),
                FeatureSpec('sbs', 'optomechanical_coupling',
                            params={'optical_physics': 'modes', 'optical_mode': 1, 'length': _LENGTH,
                                    'mechanical_quality_factor': _QUALITY, 'scattering': 'backward'})]
    properties = {'eps_r': _BULK_INDICES[1] ** 2, 'eps_xx': _BULK_INDICES[0] ** 2, 'youngs_modulus': youngs,
                  'poisson_ratio': poisson, 'rho': density, 'photoelastic_p11': _P11, 'photoelastic_p12': _P12}
    spec = ModelSpec(
        title='Self-test: backward Brillouin gain of a uniform guide (optomechanics)',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        materials=[MaterialSpec('crystal', properties=properties, selection=['all'])],
        physics=[PhysicsSpec('modes', 'em_modes', settings=em_settings,
                             features=[FeatureSpec('mode_table', 'mode_table')]),
                 PhysicsSpec('solid', 'solid_mechanics',
                             settings={'target_frequency': _CHAIN_TARGET_OFFSET * _chain_frequency()},
                             features=features)],
        mesh=MeshSpec(mode='uniform', dx=_BULK_CELL),
        study=StudySpec(type='optomechanics', settings={'n_mechanical_modes': _CHAIN_MODES}))
    return {'backward': spec}


def _check_chain(results_by_name):
    """Row nearest the longitudinal phonon: frequency vs v_L k_z, brillouin_gain vs g_B / A at the table n_eff."""
    table = results_by_name['backward'].result('solid', 'sbs')
    columns = table['columns']
    frequencies = [float(row[columns.index('frequency')]) for row in table['rows']]
    n_eff = float(table['n_eff'])
    wavenumber = float(table['axial_wavenumber'])
    omega = 2.0 * math.pi * float(table['optical_frequency'])
    _, _, c11 = _elastic_constants()
    speed = math.sqrt(c11 / _ELASTIC[2])
    expected_frequency = speed * wavenumber / (2.0 * math.pi)
    nearest = int(np.argmin([abs(frequency - expected_frequency) for frequency in frequencies]))
    measured = float(table['rows'][nearest][columns.index('brillouin_gain')])
    size_x, size_y = _chain_size()
    expected = _bulk_gain(omega, n_eff, speed, size_x * size_y)
    errors = {'gain': abs(measured / expected - 1.0),
              'frequency': abs(frequencies[nearest] / expected_frequency - 1.0),
              'axial_wavenumber': abs(wavenumber / (2.0 * n_eff * omega / C0) - 1.0)}
    worst = max(errors.values())
    return {'measured': measured, 'expected': expected, 'error': worst, 'tolerance': _CHAIN_TOLERANCE,
            'details': {'errors': errors, 'frequencies': frequencies, 'row': nearest, 'n_eff': n_eff,
                        'axial_wavenumber': wavenumber, 'scattering': table.get('scattering')}}


BRILLOUIN_CHAIN_CASES = (
    SelfTestCase(
        name='om_backward_brillouin_chain',
        label='Brillouin: backward gain through the optomechanics study',
        description='GPU: the uniform guide of om_backward_brillouin_bulk as a model (em_modes with every face '
                    'periodic, eps_r = n_o^2 and eps_xx = n_x^2; solid_mechanics with rollers on every face, target '
                    f'{_CHAIN_TARGET_OFFSET:g} x f_L), optomechanics study, optomechanical_coupling with scattering '
                    "backward: the study sets the solid's k_z = 2 Re beta (extra axial_wavenumber vs 2 n_eff omega / "
                    'c), the row nearest f_L = v_L k_z / (2 pi) must match it and its brillouin_gain the bulk g_B per '
                    f'area at the table n_eff (relative {_CHAIN_TOLERANCE:g}: float32 materials). Pins the model path '
                    '(create guided, study-set k_z, guided overlap data, 3-component rows).',
        specs=_chain_specs, check=_check_chain),
)

"""Self-tests of the optomechanical g0 overlap (optomechanics.py; CPU: photoelastic exactness, Hellmann-Feynman against
eigen-solve differences, moving-boundary convergence to the slab / cavity relations, effective mass identity, shear
isotropy, for ez,
hz and cross-section modes; GPU: the em_modes model chain against strain-optic finite differences) and the
'silicon_ring_breathing_g0' / 'silicon_disk_wgm_g0' templates; physics modules only inside the checks."""
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import no_runs, with_exact_cg
from ...model.spec import (CouplingSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)

_P11, _P12 = 0.121, 0.270              # photoelastic coefficients of the strained cells (silica-like)
_FD_STEP = 1e-5                        # central differences: max |alpha Delta eps / eps|
_HF_TOLERANCE = 1e-6                   # relative, overlap vs central difference (exact first-order perturbation)

# ---- 1: uniform ez cavity under uniform strain: d omega / omega = -Delta eps / (2 eps) identically ----
_UNIFORM_CELLS = (24, 17)
_UNIFORM_CELL = 50e-9                  # m
_UNIFORM_EPS = 2.25
_UNIFORM_STRAIN = (3e-4, -1e-4)        # (S_xx, S_yy) of the prescribed Q = (S_xx x, S_yy y)
_UNIFORM_HZ_STRAIN = 2e-4              # isotropic S_xx = S_yy of the hz run: d eps_xx = d eps_yy, a uniform eps scaling
_UNIFORM_MODES = 3
_UNIFORM_TOLERANCE = 1e-10

# ---- 2: partially filled ez cavity (smooth bump Q) and the x-periodic TE slab of cross_section_modes ----
_CAVITY_CELLS = (20, 16)               # the identity is exact on any grid: small for the six dense eighs
_CAVITY_CELL = 40e-9                   # m
_BLOCK = (slice(6, 14), slice(4, 12))  # dielectric, strained cells
_BLOCK_EPS = 6.25
_WAVELENGTH = 1.55e-6                  # m, cross-section cases
_SLAB_PERIOD = 50e-9                   # m: the single periodic x cell
_SLAB_INDEX, _CLADDING_INDEX = 2.0, 1.444
_TWIN_HALF_WIDTH, _TWIN_CLADDING, _TWIN_CELL = 0.25e-6, 1.0e-6, 10e-9   # m

# ---- 3 / 4: moving boundary vs the continuum slab / cavity relations, two meshes ----
_SLAB_HALF_WIDTH, _SLAB_CLADDING = 0.3e-6, 2.0e-6    # m: e^(-2 gamma D) < 1e-6 of the pec walls
_MESH_CELLS = (10e-9, 5e-9)                          # m: coarse, fine
_CAVITY_INDICES = (2.0, 1.0)                         # slab, outside
_CAVITY_HALF_WIDTH, _CAVITY_HALF_LENGTH = 0.4e-6, 1.0e-6   # m: slab |x| < a, walls (E = 0) at |x| = L
_ORDER_RATIO = 1.6                     # coarse / fine error ratio at least (first order: 2, second: 4)
_ORDER_FLOOR = 1e-6                    # fine errors below this pass without the ratio (round-off)
_MOVING_TOLERANCES = {'te': 1e-2, 'tm': 5e-2, 'ez': 3e-2, 'hz_x': 5e-2, 'hz_y': 5e-2}   # fine-mesh relative error
_HZ_TRANSVERSE = (8, 0.8e-6)           # hz slab: cells across and width W of the pec box along the slab (H_z ~ cos(pi
                                       # t / W): q > k n2, evanescent outside, normal E at the slab faces)
_RESIDUAL_TOLERANCE = 1e-9             # separable hz field in the 2D operator (relative)

# ---- 5: Q1 plate modes (rollers on x_lo / y_lo) ----
_PLATE_CELLS = (10, 6)
_PLATE_CELL = 1e-6                     # m
_PLATE_YOUNGS, _PLATE_POISSON, _PLATE_DENSITY = 1.65e11, 0.22, 2329.0
_PLATE_LENGTH = 1e-5                   # m
_PLATE_MODES = 4
_MASS_TOLERANCE = 1e-8                 # m_eff Omega^2 vs L Q^T K Q (eigsh to round-off)
_INVARIANCE_TOLERANCE = 1e-12          # g0 under the reference-node choice

# ---- shear photoelasticity: uniform E at angle theta, uniform (S_xx, S_yy, S_xy), periodic grid ----
_SHEAR_CELLS = (6, 5)
_SHEAR_CELL = 20e-9                    # m
_SHEAR_EPS = 2.25
_SHEAR_STRAIN = (2e-4, -1e-4, 1.5e-4)  # (S_xx, S_yy, tensor S_xy) of Q = (S_xx x + S_xy y, S_xy x + S_yy y)
_SHEAR_P44 = -0.051                    # explicit (cubic, silicon-like) p44 of the second variant
_SHEAR_ANGLES = (0.3, 1.1, 2.4)        # rad, E = (cos, sin)
_SHEAR_TOLERANCE = 1e-12


def _cavity_stiffness(n_cells, cell):
    """(n, n) dense -d^2/dx^2 of uniform cells with E = 0 on both outer faces (odd image cells)."""
    stiffness = 2.0 * np.eye(n_cells) - np.eye(n_cells, k=1) - np.eye(n_cells, k=-1)
    stiffness[0, 0] = stiffness[-1, -1] = 3.0
    return stiffness / cell ** 2


def _ez_cavity_modes(eps, cell, n_modes):
    """(omega (n_modes,) [rad/s], fields (n_modes, nx, ny)) of -lap E = (omega / C0)^2 eps E on uniform square cells,
    E = 0 on the outer faces (ny = 1: no y dependence): the symmetric ez cavity pencil, dense scipy eigh."""
    import scipy.linalg
    eps = np.asarray(eps, dtype=np.float64)
    n_x, n_y = eps.shape
    stiffness = np.kron(_cavity_stiffness(n_x, cell), np.eye(n_y))
    if n_y > 1:
        stiffness += np.kron(np.eye(n_x), _cavity_stiffness(n_y, cell))
    values, vectors = scipy.linalg.eigh(stiffness, np.diag(eps.ravel()), subset_by_index=[0, n_modes - 1])
    return C0 * np.sqrt(values), vectors.T.reshape(n_modes, n_x, n_y)


def _hz_operator(eps_xx, eps_yy, cell_widths):
    """(ops, curl-curl L (scipy CSR, L H = -k^2 H), eps_xx at the E_x (hi-y) faces, eps_yy at the E_y (hi-x) faces) of
    the FDFD hz discretisation of a pec box: operators.yee_operators / helmholtz_operator, yee_grid.face_average."""
    from ..em.boundaries import BoundaryConfig
    from ..em.operators import helmholtz_operator, yee_operators
    from ..em.yee_grid import face_average
    dx_cells, dy_cells = cell_widths
    ops = yee_operators((dx_cells, dy_cells), BoundaryConfig('pec', 'pec', 'pec', 'pec'), float(dx_cells[0]), xp=np)
    eps_ex_faces = face_average(np.asarray(eps_xx, dtype=np.float64), 1, False, dy_cells)
    eps_ey_faces = face_average(np.asarray(eps_yy, dtype=np.float64), 0, False, dx_cells)
    curl_curl = helmholtz_operator(ops, 1.0 / eps_ey_faces, 1.0 / eps_ex_faces, np.zeros(eps_ex_faces.shape), xp=np)
    return ops, curl_curl, eps_ex_faces, eps_ey_faces


def _hz_face_fields(ops, field, eps_ex_faces, eps_ey_faces, omega):
    """(E_x on the hi-y faces, E_y on the hi-x faces) [V/m] of H_z cells at omega: the curl of FDFDPhysics.face_fields."""
    flat = np.asarray(field, dtype=np.complex128).ravel()
    scale = 1.0 / (1j * omega * EPS0)
    return ((ops.dy_h @ flat).reshape(field.shape) * scale / eps_ex_faces,
            -(ops.dx_h @ flat).reshape(field.shape) * scale / eps_ey_faces)


def _hz_cavity_modes(eps_xx, eps_yy, cell, n_modes):
    """(omega (n_modes,), H_z (n_modes, nx, ny), [(E_x, E_y) faces], eps_xx at E_x, eps_yy at E_y) of the lowest
    non-static H_z resonances of a pec box of uniform square cells (L symmetric there), dense scipy eigh."""
    import scipy.linalg
    n_x, n_y = np.shape(eps_xx)
    ops, curl_curl, eps_ex_faces, eps_ey_faces = _hz_operator(eps_xx, eps_yy, (np.full(n_x, cell), np.full(n_y, cell)))
    stiffness = -curl_curl.toarray()
    values, vectors = scipy.linalg.eigh(0.5 * (stiffness + stiffness.T), subset_by_index=[1, n_modes])   # 0: static
    omegas = C0 * np.sqrt(values)
    fields = vectors.T.reshape(n_modes, n_x, n_y)
    faces = [_hz_face_fields(ops, field, eps_ex_faces, eps_ey_faces, omega) for omega, field in zip(omegas, fields)]
    return omegas, fields, faces, eps_ex_faces, eps_ey_faces


def _full_window(shape):
    return slice(0, shape[0]), slice(0, shape[1])


def _bump_nodal(x_nodes, y_nodes):
    """Smooth non-uniform nodal Q (n_x + 1, n_y + 1, 2) [m] over the node coordinates."""
    span_x, span_y = x_nodes[-1] - x_nodes[0], y_nodes[-1] - y_nodes[0]
    u, v = np.meshgrid((x_nodes - x_nodes[0]) / span_x, (y_nodes - y_nodes[0]) / span_y, indexing='ij')
    return np.stack((0.7 * np.sin(math.pi * u) * np.sin(2.0 * math.pi * v) + 0.2 * u * v,
                     np.cos(1.3 * math.pi * u) * np.sin(math.pi * v) - 0.3 * v ** 2), axis=-1)


def _masked_changes(optomechanics, eps, nodal, solid, p11, p12, dx, dy):
    """(Delta eps_xx, _yy, _zz) of the nodal strain on the solid cells of an isotropic eps."""
    strain_xx, strain_yy, _ = optomechanics.element_strains(nodal, dx, dy)
    return optomechanics.photoelastic_changes(eps, eps, eps, p11, p12, np.where(solid, strain_xx, 0.0),
                                              np.where(solid, strain_yy, 0.0))


def _uniform_strain_nodal(strain_xx, strain_yy, cell):
    """Nodal Q = (S_xx x, S_yy y) [m] on the _UNIFORM_CELLS nodes."""
    n_x, n_y = _UNIFORM_CELLS
    nodal = np.zeros((n_x + 1, n_y + 1, 2))
    nodal[..., 0] = strain_xx * cell * np.arange(n_x + 1)[:, None]
    nodal[..., 1] = strain_yy * cell * np.arange(n_y + 1)[None, :]
    return nodal


def _check_uniform_cavity(results_by_name):
    """g_om photoelastic / omega = eps p12 (S_xx + S_yy) / 2 (ez) and eps (p11 + p12) S / 2 (hz, isotropic S), g_om
    moving boundary = 0, for three modes each."""
    from . import optomechanics
    n_x, n_y = _UNIFORM_CELLS
    cell = _UNIFORM_CELL
    eps = np.full(_UNIFORM_CELLS, _UNIFORM_EPS)
    dx, dy = np.full(n_x, cell), np.full(n_y, cell)
    solid = np.ones(_UNIFORM_CELLS, dtype=bool)
    p11, p12 = np.full(_UNIFORM_CELLS, _P11), np.full(_UNIFORM_CELLS, _P12)
    overlaps = {}
    omegas, fields = _ez_cavity_modes(eps, cell, _UNIFORM_MODES)
    overlaps['ez'] = ([optomechanics.in_plane_ez_overlap(field, eps, dx, dy, omega)
                       for omega, field in zip(omegas, fields)], _uniform_strain_nodal(*_UNIFORM_STRAIN, cell),
                      0.5 * _UNIFORM_EPS * _P12 * sum(_UNIFORM_STRAIN))
    omegas, fields, faces, eps_ex_faces, eps_ey_faces = _hz_cavity_modes(eps, eps, cell, _UNIFORM_MODES)
    overlaps['hz'] = ([optomechanics.in_plane_hz_overlap(field, *face_fields, eps, eps, eps_ex_faces, eps_ey_faces,
                                                         np.ones(_UNIFORM_CELLS), (dx, dy), _full_window(eps.shape),
                                                         (False, False), omega)
                       for omega, field, face_fields in zip(omegas, fields, faces)],
                      _uniform_strain_nodal(_UNIFORM_HZ_STRAIN, _UNIFORM_HZ_STRAIN, cell),
                      0.5 * _UNIFORM_EPS * (_P11 + _P12) * _UNIFORM_HZ_STRAIN)
    errors, details = {}, {}
    for name, (mode_overlaps, nodal, expected_ratio) in overlaps.items():
        mode_errors, measured = [], []
        for overlap in mode_overlaps:
            moving, photo = optomechanics.overlap_integrals(overlap, nodal, solid, p11, p12, dx, dy)
            expected = overlap.omega * expected_ratio
            rate_photo, rate_moving = (overlap.scale * photo).real, (overlap.scale * moving).real
            measured.append(rate_photo)
            mode_errors.append(abs(rate_photo / expected - 1.0) + abs(rate_moving) / abs(expected))
        errors[name] = max(mode_errors)
        details[name] = {'g_om_photoelastic': measured, 'expected_over_omega': expected_ratio,
                         'omega': [overlap.omega for overlap in mode_overlaps], 'errors': mode_errors}
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _UNIFORM_TOLERANCE, 'details': details}


def _slab_layout(cell, half_width, cladding):
    """(dy (ny,) [m], eps (ny,), core bool (ny,), centre y [m]): pec ghost, cladding, core, cladding, pec ghost."""
    n_cladding = int(round(cladding / cell))
    n_core = int(round(2.0 * half_width / cell))
    n_y = 2 + 2 * n_cladding + n_core
    core = np.zeros(n_y, dtype=bool)
    core[1 + n_cladding:1 + n_cladding + n_core] = True
    eps = np.where(core, _SLAB_INDEX ** 2, _CLADDING_INDEX ** 2)
    return np.full(n_y, cell), eps, core, (1 + n_cladding + 0.5 * n_core) * cell


def _slab_materials(cross_section_modes, eps_xx, eps_yy, eps_zz):
    """CrossSectionMaterials of (ny,) component lines on the one periodic x cell."""
    def line(values):
        return np.asarray(values, dtype=np.complex128)[None, :]
    return cross_section_modes.CrossSectionMaterials(
        eps_xx=line(eps_xx), eps_yy=line(eps_yy), eps_zz=line(eps_zz),
        mu=np.ones((1, np.size(eps_zz)), dtype=np.complex128), pec=np.zeros((1, np.size(eps_zz)), dtype=bool))


def _slab_mode(materials, cell_widths, family):
    """(mode, ModePointMaterials) of the first 'te' (te_fraction > 0.5) or 'tm' mode of the x-periodic, y-pec slab."""
    from ..em import cross_section_modes
    from ..em.boundaries import BoundaryConfig
    config = BoundaryConfig('periodic', 'periodic', 'pec', 'pec')
    cell = float(cell_widths[1][0])
    solve = cross_section_modes.solve_cross_section_modes(cell_widths, config, cell, materials,
                                                          2.0 * math.pi * C0 / _WAVELENGTH, 4)
    for mode in solve.modes:
        if (mode.te_fraction > 0.5) == (family == 'te'):
            return mode, cross_section_modes.mode_point_materials(cell_widths, config, cell, materials)
    raise ValueError(f'no {family.upper()} slab mode among {len(solve.modes)} modes')


def _check_hellmann_feynman(results_by_name):
    """Photoelastic g_om (ez cavity) and d beta / d alpha (TE slab) of a smooth bump Q vs central differences of two
    eigen-solves with eps -/+ alpha Delta eps(Q)."""
    from ..em import cross_section_modes
    from . import optomechanics
    errors, details = {}, {}

    n_x, n_y = _CAVITY_CELLS
    cell = _CAVITY_CELL
    eps = np.ones(_CAVITY_CELLS)
    eps[_BLOCK] = _BLOCK_EPS
    solid = np.zeros(_CAVITY_CELLS, dtype=bool)
    solid[_BLOCK] = True
    p11, p12 = np.where(solid, _P11, 0.0), np.where(solid, _P12, 0.0)
    no_shear = np.zeros(_CAVITY_CELLS)   # p44 = 0: the discrete operators hold diagonal eps only
    dx, dy = np.full(n_x, cell), np.full(n_y, cell)
    nodal = _bump_nodal(cell * np.arange(n_x + 1), cell * np.arange(n_y + 1))
    omegas, fields = _ez_cavity_modes(eps, cell, 1)
    overlap = optomechanics.in_plane_ez_overlap(fields[0], eps, dx, dy, omegas[0])
    _, photo = optomechanics.overlap_integrals(overlap, nodal, solid, p11, p12, dx, dy, no_shear)
    rate = (overlap.scale * photo).real
    change = _masked_changes(optomechanics, eps, nodal, solid, p11, p12, dx, dy)[2]
    alpha = _FD_STEP / float(np.max(np.abs(change / eps)))
    plus = _ez_cavity_modes(eps + alpha * change, cell, 1)[0][0]
    minus = _ez_cavity_modes(eps - alpha * change, cell, 1)[0][0]
    difference = (plus - minus) / (2.0 * alpha)
    errors['ez cavity'] = abs(rate / difference - 1.0)
    details['ez cavity'] = {'g_om_photoelastic': rate, 'central_difference': difference, 'alpha': alpha}

    # hz: E_x faces see d eps_xx, E_y faces d eps_yy (p11 / p12 differ, so the component mapping is pinned)
    omegas, fields, faces, eps_ex_faces, eps_ey_faces = _hz_cavity_modes(eps, eps, cell, 1)
    overlap = optomechanics.in_plane_hz_overlap(fields[0], *faces[0], eps, eps, eps_ex_faces, eps_ey_faces,
                                                np.ones(_CAVITY_CELLS), (dx, dy), _full_window(eps.shape),
                                                (False, False), omegas[0])
    _, photo = optomechanics.overlap_integrals(overlap, nodal, solid, p11, p12, dx, dy, no_shear)
    rate = (overlap.scale * photo).real
    change_xx, change_yy, _ = _masked_changes(optomechanics, eps, nodal, solid, p11, p12, dx, dy)
    alpha = _FD_STEP / max(float(np.max(np.abs(values / eps))) for values in (change_xx, change_yy))
    plus = _hz_cavity_modes(eps + alpha * change_xx, eps + alpha * change_yy, cell, 1)[0][0]
    minus = _hz_cavity_modes(eps - alpha * change_xx, eps - alpha * change_yy, cell, 1)[0][0]
    difference = (plus - minus) / (2.0 * alpha)
    errors['hz cavity'] = abs(rate / difference - 1.0)
    details['hz cavity'] = {'g_om_photoelastic': rate, 'central_difference': difference, 'alpha': alpha}

    widths_y, eps_line, core, _ = _slab_layout(_TWIN_CELL, _TWIN_HALF_WIDTH, _TWIN_CLADDING)
    cell_widths = (np.array([_SLAB_PERIOD]), widths_y)
    mode, points = _slab_mode(_slab_materials(cross_section_modes, eps_line, eps_line, eps_line), cell_widths, 'te')
    window = (slice(0, 1), slice(0, widths_y.size))
    overlap = optomechanics.cross_section_overlap(mode, points, window)
    y_nodes = np.concatenate(([0.0], np.cumsum(widths_y)))
    centre, spread = 0.5 * y_nodes[-1], 2.0 * _TWIN_HALF_WIDTH
    nodal = np.zeros((2, widths_y.size + 1, 2))
    nodal[:, :, 1] = ((y_nodes - centre) / spread * np.exp(-((y_nodes - centre) / spread) ** 2))[None, :]
    solid_line = core[None, :]
    p11, p12 = np.where(solid_line, _P11, 0.0), np.where(solid_line, _P12, 0.0)
    _, photo = optomechanics.overlap_integrals(overlap, nodal, solid_line, p11, p12, cell_widths[0], widths_y,
                                               np.zeros(solid_line.shape))
    rate = (overlap.beta_scale * photo).real
    changes = [values[0] for values in _masked_changes(optomechanics, eps_line[None, :], nodal, solid_line, p11, p12,
                                                       cell_widths[0], widths_y)]
    alpha = _FD_STEP / max(float(np.max(np.abs(values / eps_line))) for values in changes)
    betas = []
    for sign in (1.0, -1.0):
        shifted = [eps_line + sign * alpha * values for values in changes]
        betas.append(complex(_slab_mode(_slab_materials(cross_section_modes, *shifted), cell_widths, 'te')[0].beta).real)
    difference = (betas[0] - betas[1]) / (2.0 * alpha)
    errors['te slab'] = abs(rate / difference - 1.0)
    details['te slab'] = {'d_beta_d_alpha': rate, 'central_difference': difference, 'alpha': alpha,
                          'n_eff': complex(mode.n_eff).real}
    worst = max(errors.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _HF_TOLERANCE,
            'details': dict(details, errors=errors)}


def _slab_derivative(family, half_width):
    """(beta [1/m], d beta / d a [1/m^2]) of the fundamental even TE / TM mode of the symmetric slab of half-width a
    in an infinite cladding: kappa tan(kappa a) w_core = gamma w_clad (w = 1 TE, 1 / n^2 TM), implicit derivative."""
    from scipy.optimize import brentq
    k0 = 2.0 * math.pi / _WAVELENGTH
    core_weight, cladding_weight = (1.0, 1.0) if family == 'te' else (1.0 / _SLAB_INDEX ** 2,
                                                                       1.0 / _CLADDING_INDEX ** 2)

    def parts(beta):
        return math.sqrt(k0 ** 2 * _SLAB_INDEX ** 2 - beta ** 2), math.sqrt(beta ** 2 - k0 ** 2 * _CLADDING_INDEX ** 2)

    def relation(beta):
        kappa, gamma = parts(beta)
        return core_weight * kappa * math.tan(kappa * half_width) - cladding_weight * gamma

    lowest = max(k0 * _CLADDING_INDEX,
                 math.sqrt(max(k0 ** 2 * _SLAB_INDEX ** 2 - (math.pi / (2.0 * half_width)) ** 2, 0.0)))
    beta = brentq(relation, lowest * (1.0 + 1e-12), k0 * _SLAB_INDEX * (1.0 - 1e-12), xtol=1e-12, rtol=1e-14)
    kappa, gamma = parts(beta)
    secant_squared = 1.0 / math.cos(kappa * half_width) ** 2
    derivative_a = core_weight * kappa ** 2 * secant_squared
    derivative_beta = (-core_weight * beta / kappa * (math.tan(kappa * half_width) + kappa * half_width * secant_squared)
                       - cladding_weight * beta / gamma)
    return beta, -derivative_a / derivative_beta


def _cavity_derivative():
    """(omega [rad/s], d omega / d a [rad/s/m]) of the fundamental even E_z mode of the slab |x| < a between E = 0
    walls at |x| = L: n1 tan(k n1 a) = n2 cot(k n2 (L - a)), implicit derivative."""
    from scipy.optimize import brentq
    n_core, n_outside = _CAVITY_INDICES
    a, length = _CAVITY_HALF_WIDTH, _CAVITY_HALF_LENGTH

    def relation(k):
        return n_core * math.tan(k * n_core * a) - n_outside / math.tan(k * n_outside * (length - a))

    highest = math.pi / (2.0 * n_core * a) * (1.0 - 1e-12)
    k = brentq(relation, 1e-6 * highest, highest, xtol=1e-6, rtol=1e-14)
    inner, outer = k * n_core * a, k * n_outside * (length - a)
    secant_squared, cosecant_squared = 1.0 / math.cos(inner) ** 2, 1.0 / math.sin(outer) ** 2
    derivative_a = k * (n_core ** 2 * secant_squared - n_outside ** 2 * cosecant_squared)
    derivative_k = n_core ** 2 * a * secant_squared + n_outside ** 2 * (length - a) * cosecant_squared
    return C0 * k, -C0 * derivative_a / derivative_k


def _hz_transverse_wavenumber():
    """q [1/m] of the discrete cos(pi (t + 1/2) / n_t) across the pec box: 2 sin(pi / (2 n_t)) / pitch."""
    n_transverse, width = _HZ_TRANSVERSE
    return 2.0 * math.sin(0.5 * math.pi / n_transverse) * n_transverse / width


def _hz_cavity_derivative():
    """(omega, d omega / d a) of the even fundamental H_z = f(n) cos(q t) of the slab |n| < a between E_t = 0 walls at
    |n| = L: (kappa1 / eps1) tan(kappa1 a) + (kappa2 / eps2) tan(kappa2 (L - a)) = 0, kappa_i^2 = k^2 eps_i - q^2 (the
    imaginary kappa2 is the tanh branch), q the discrete transverse wavenumber; implicit derivative."""
    import cmath
    from scipy.optimize import brentq
    eps_core, eps_outside = (index ** 2 for index in _CAVITY_INDICES)
    a, length = _CAVITY_HALF_WIDTH, _CAVITY_HALF_LENGTH
    transverse = _hz_transverse_wavenumber()

    def kappas(k):
        return cmath.sqrt(k * k * eps_core - transverse ** 2), cmath.sqrt(k * k * eps_outside - transverse ** 2)

    def relation(k):
        kappa_core, kappa_outside = kappas(k)
        return (kappa_core / eps_core * cmath.tan(kappa_core * a)
                + kappa_outside / eps_outside * cmath.tan(kappa_outside * (length - a))).real

    lowest = transverse / _CAVITY_INDICES[0] * (1.0 + 1e-9)
    highest = math.sqrt((math.pi / (2.0 * a)) ** 2 + transverse ** 2) / _CAVITY_INDICES[0] * (1.0 - 1e-12)
    k = brentq(relation, lowest, highest, xtol=1e-6, rtol=1e-14)
    kappa_core, kappa_outside = kappas(k)
    secant_core = 1.0 / cmath.cos(kappa_core * a) ** 2
    secant_outside = 1.0 / cmath.cos(kappa_outside * (length - a)) ** 2
    derivative_a = (kappa_core ** 2 / eps_core * secant_core - kappa_outside ** 2 / eps_outside * secant_outside).real
    derivative_k = (k / kappa_core * (cmath.tan(kappa_core * a) + kappa_core * a * secant_core)
                    + k / kappa_outside * (cmath.tan(kappa_outside * (length - a))
                                           + kappa_outside * (length - a) * secant_outside)).real
    return C0 * k, -C0 * derivative_a / derivative_k


def _hz_slab_cavity(cell, axis):
    """(omega, H_z, cell widths, eps) of that fundamental on `cell` along the slab normal (x for axis 0, y for axis 1)
    and _HZ_TRANSVERSE cells across: f of the separable 1D reduction of the hz pencil times the discrete cosine."""
    import scipy.linalg
    from ..em.yee_grid import face_average
    n_cells = int(round(2.0 * _CAVITY_HALF_LENGTH / cell))
    centres = -_CAVITY_HALF_LENGTH + cell * (np.arange(n_cells) + 0.5)
    eps_line = np.where(np.abs(centres) < _CAVITY_HALF_WIDTH, _CAVITY_INDICES[0] ** 2, _CAVITY_INDICES[1] ** 2)
    n_transverse, width = _HZ_TRANSVERSE
    face_eps = face_average(eps_line, 0, False, np.full(n_cells, cell))[:-1]
    difference = np.eye(n_cells - 1, n_cells, k=1) - np.eye(n_cells - 1, n_cells)
    # cell x [sum V (df / dn)^2 + q^2 sum f^2 / eps] = k^2 cell sum f^2 (the hz Rayleigh quotient of f x cos)
    stiffness = (difference.T @ np.diag(1.0 / (face_eps * cell)) @ difference
                 + np.diag(_hz_transverse_wavenumber() ** 2 * cell / eps_line))
    values, vectors = scipy.linalg.eigh(stiffness / cell, subset_by_index=[0, 0])
    field = np.outer(vectors[:, 0], np.cos(math.pi * (np.arange(n_transverse) + 0.5) / n_transverse))
    eps = np.outer(eps_line, np.ones(n_transverse))
    widths = (np.full(n_cells, cell), np.full(n_transverse, width / n_transverse))
    if axis == 1:
        field, eps, widths = field.T.copy(), eps.T.copy(), widths[::-1]
    return C0 * math.sqrt(values[0]), field, widths, eps


def _hz_breathing_rate(cell, axis):
    """(moving-boundary d omega / d a of a breathing Q on _hz_slab_cavity, relative residual of its field in the 2D
    FDFD hz operator: pins the reduction)."""
    from . import optomechanics
    omega, field, widths, eps = _hz_slab_cavity(cell, axis)
    ops, curl_curl, eps_ex_faces, eps_ey_faces = _hz_operator(eps, eps, widths)
    flat, eigenvalue = field.ravel(), (omega / C0) ** 2
    residual = float(np.linalg.norm(curl_curl @ flat + eigenvalue * flat) / np.linalg.norm(eigenvalue * flat))
    ex_faces, ey_faces = _hz_face_fields(ops, field, eps_ex_faces, eps_ey_faces, omega)
    overlap = optomechanics.in_plane_hz_overlap(field, ex_faces, ey_faces, eps, eps, eps_ex_faces, eps_ey_faces,
                                                np.ones(field.shape), widths, _full_window(field.shape),
                                                (False, False), omega)
    n_x, n_y = field.shape
    nodes = -_CAVITY_HALF_LENGTH + cell * np.arange(field.shape[axis] + 1)
    nodal = np.zeros((n_x + 1, n_y + 1, 2))
    if axis == 0:
        nodal[:, :, 0] = np.sign(nodes)[:, None]
    else:
        nodal[:, :, 1] = np.sign(nodes)[None, :]
    q_x, q_y = optomechanics.face_displacements(nodal)
    return (overlap.scale * (np.sum(q_x * overlap.face_x) + np.sum(q_y * overlap.face_y))).real, residual


def _order_issue(errors, tolerance):
    """Violation text when the fine error exceeds tolerance or the error does not fall by _ORDER_RATIO; None."""
    coarse, fine = errors
    if not fine <= tolerance:
        return f'fine-mesh error {fine:.3g} > {tolerance:g}'
    if fine > _ORDER_FLOOR and not coarse >= _ORDER_RATIO * fine:
        return f'error ratio {coarse / fine:.3g} < {_ORDER_RATIO} (no first-order convergence)'
    return None


def _check_moving_boundary(results_by_name):
    """Moving-boundary rates of a breathing Q (+1 / -1 on the two interfaces) on two meshes: d beta / d a of the TE / TM
    slab modes (cross_section_modes) and d omega / d a of the ez cavity vs the continuum relations."""
    from ..em import cross_section_modes
    from . import optomechanics
    errors, rates, violations = {}, {}, []
    for family in ('te', 'tm'):
        _, expected = _slab_derivative(family, _SLAB_HALF_WIDTH)
        family_errors, family_rates = [], []
        for cell in _MESH_CELLS:
            widths_y, eps_line, _, centre = _slab_layout(cell, _SLAB_HALF_WIDTH, _SLAB_CLADDING)
            cell_widths = (np.array([_SLAB_PERIOD]), widths_y)
            mode, points = _slab_mode(_slab_materials(cross_section_modes, eps_line, eps_line, eps_line), cell_widths,
                                      family)
            overlap = optomechanics.cross_section_overlap(mode, points, (slice(0, 1), slice(0, widths_y.size)))
            y_nodes = cell * np.arange(widths_y.size + 1)
            nodal = np.zeros((2, widths_y.size + 1, 2))
            nodal[:, :, 1] = np.sign(y_nodes - centre)[None, :]
            q_x, q_y = optomechanics.face_displacements(nodal)
            rate = (overlap.beta_scale * (np.sum(q_x * overlap.face_x) + np.sum(q_y * overlap.face_y))).real
            family_rates.append(rate)
            family_errors.append(abs(rate / expected - 1.0))
        errors[family], rates[family] = family_errors, {'expected': expected, 'measured': family_rates}
    _, expected = _cavity_derivative()
    cavity_errors, cavity_rates = [], []
    for cell in _MESH_CELLS:
        n_cells = int(round(2.0 * _CAVITY_HALF_LENGTH / cell))
        centres = -_CAVITY_HALF_LENGTH + cell * (np.arange(n_cells) + 0.5)
        eps = np.where(np.abs(centres) < _CAVITY_HALF_WIDTH, _CAVITY_INDICES[0] ** 2, _CAVITY_INDICES[1] ** 2)[:, None]
        omegas, fields = _ez_cavity_modes(eps, cell, 1)
        widths_x, widths_y = np.full(n_cells, cell), np.array([cell])
        overlap = optomechanics.in_plane_ez_overlap(fields[0], eps, widths_x, widths_y, omegas[0])
        nodal = np.zeros((n_cells + 1, 2, 2))
        nodal[:, :, 0] = np.sign(-_CAVITY_HALF_LENGTH + cell * np.arange(n_cells + 1))[:, None]
        q_x, q_y = optomechanics.face_displacements(nodal)
        rate = (overlap.scale * (np.sum(q_x * overlap.face_x) + np.sum(q_y * overlap.face_y))).real
        cavity_rates.append(rate)
        cavity_errors.append(abs(rate / expected - 1.0))
    errors['ez'], rates['ez'] = cavity_errors, {'expected': expected, 'measured': cavity_rates}
    _, expected = _hz_cavity_derivative()
    for axis, name in ((0, 'hz_x'), (1, 'hz_y')):
        hz_errors, hz_rates, residuals = [], [], []
        for cell in _MESH_CELLS:
            rate, residual = _hz_breathing_rate(cell, axis)
            hz_rates.append(rate)
            hz_errors.append(abs(rate / expected - 1.0))
            residuals.append(residual)
        errors[name] = hz_errors
        rates[name] = {'expected': expected, 'measured': hz_rates, 'operator_residuals': residuals}
        if not max(residuals) <= _RESIDUAL_TOLERANCE:
            violations.append(f'{name}: separable field residual {max(residuals):.3g} > {_RESIDUAL_TOLERANCE:g}')
    for name, values in errors.items():
        issue = _order_issue(values, _MOVING_TOLERANCES[name])
        if issue is not None:
            violations.append(f'{name}: {issue}')
    normalized = max(values[1] / _MOVING_TOLERANCES[name] for name, values in errors.items())
    return {'measured': normalized, 'expected': 0.0, 'error': normalized + len(violations), 'tolerance': 1.0,
            'details': {'relative_errors_coarse_fine': errors, 'rates': rates, 'cells': list(_MESH_CELLS),
                        'violations': violations}}


def _host_csr(stencil):
    """Host scipy CSR of a (nx_nodes, ny_nodes, 2, 3, 3, 2) Q1 stencil (dof = 2 node + component)."""
    import scipy.sparse
    nx_nodes, ny_nodes = stencil.shape[:2]
    node_i, node_j, row_component, offset_i, offset_j, column_component = np.indices(stencil.shape)
    neighbour_i, neighbour_j = node_i + offset_i - 1, node_j + offset_j - 1
    inside = (neighbour_i >= 0) & (neighbour_i < nx_nodes) & (neighbour_j >= 0) & (neighbour_j < ny_nodes)
    rows = (2 * (node_i * ny_nodes + node_j) + row_component)[inside]
    columns = (2 * (neighbour_i * ny_nodes + neighbour_j) + column_component)[inside]
    n_dofs = 2 * nx_nodes * ny_nodes
    return scipy.sparse.csr_matrix((stencil[inside], (rows, columns)), shape=(n_dofs, n_dofs))


def _check_effective_mass(results_by_name):
    """Q1 plate modes (modal.solve_modes on host K, M): m_eff Omega^2 = L Q^T K Q / |Q_ref|^2 and g0 equal for the
    peak-node and a point reference (a synthetic ez overlap with an eps jump)."""
    from . import optomechanics
    from .materials import lame_coefficients
    from .modal import solve_modes
    from .system import mass_stencil, stiffness_stencil
    n_x, n_y = _PLATE_CELLS
    dx, dy = np.full(n_x, _PLATE_CELL), np.full(n_y, _PLATE_CELL)
    lame_lambda, mu = lame_coefficients(np.full(_PLATE_CELLS, _PLATE_YOUNGS), np.full(_PLATE_CELLS, _PLATE_POISSON))
    longitudinal = lame_lambda + 2.0 * mu
    stiffness = _host_csr(stiffness_stencil(longitudinal, lame_lambda, longitudinal, mu, dx, dy, np.float64, xp=np))
    mass = _host_csr(mass_stencil(np.full(_PLATE_CELLS, _PLATE_DENSITY), dx, dy, np.float64, xp=np))
    node_shape = (n_x + 1, n_y + 1)
    free = np.ones((*node_shape, 2), dtype=bool)
    free[0, :, 0] = False    # roller x_lo
    free[:, 0, 1] = False    # roller y_lo
    solve = solve_modes(stiffness, mass, free.ravel(), 0.0, _PLATE_MODES, 2)
    centres_x, centres_y = np.meshgrid((np.arange(n_x) + 0.5) / n_x, (np.arange(n_y) + 0.5) / n_y, indexing='ij')
    field = np.exp(-((centres_x - 0.4) ** 2 + (centres_y - 0.6) ** 2) / 0.1) * np.exp(1j * centres_y)
    eps = np.where(centres_x < 0.5, 4.0, 2.0)
    overlap = optomechanics.in_plane_ez_overlap(field, eps, dx, dy, 1.2e15)
    solid = np.ones(_PLATE_CELLS, dtype=bool)
    p11, p12 = np.full(_PLATE_CELLS, _P11), np.full(_PLATE_CELLS, _P12)
    mass_errors, invariance_errors = [], []
    for number, resonance in enumerate(solve.resonances, start=1):
        displacement = resonance.displacement
        norms = np.sqrt(np.sum(displacement.reshape(*node_shape, 2) ** 2, axis=2))
        reference_node = np.unravel_index(int(np.argsort(norms.ravel())[-5]), node_shape)
        rows = [optomechanics.mode_coupling_row(number, overlap, resonance.omega, displacement, node_shape, mass,
                                                solid, p11, p12, dx, dy, _PLATE_LENGTH, reference)
                for reference in (None, reference_node)]
        work = _PLATE_LENGTH * float(displacement @ (stiffness @ displacement)) / float(np.max(norms)) ** 2
        mass_errors.append(abs(rows[0][2] * resonance.omega ** 2 / work - 1.0))
        g0 = optomechanics.OPTOMECHANICS_TABLE_COLUMNS.index('g0')
        invariance_errors.append(abs(rows[1][g0] - rows[0][g0]) / abs(rows[0][g0]))
    errors = {'effective mass': max(mass_errors) / _MASS_TOLERANCE,
              'g0 invariance': max(invariance_errors) / _INVARIANCE_TOLERANCE}
    worst = max(errors.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'mass_errors': mass_errors, 'invariance_errors': invariance_errors,
                        'frequencies': [resonance.frequency for resonance in solve.resonances],
                        'normalized_errors': errors}}


def _check_shear_rotation(results_by_name):
    """Photoelastic overlap of uniform (S_xx, S_yy, S_xy) and uniform E = (cos, sin) theta on the hz face and the
    cross-section paths: unset p44 (NaN) = eps0 A d eps along E of the rotated diagonal strain (isotropy); an explicit
    p44 = eps0 A (d eps_xx cos^2 + d eps_yy sin^2 + 2 d eps_xy cos sin)."""
    import types
    from ..em import cross_section_modes
    from ..em.boundaries import BoundaryConfig
    from . import optomechanics
    n_x, n_y = _SHEAR_CELLS
    cell = _SHEAR_CELL
    dx, dy = np.full(n_x, cell), np.full(n_y, cell)
    area = n_x * n_y * cell ** 2
    eps = np.full(_SHEAR_CELLS, _SHEAR_EPS, dtype=np.complex128)
    ones = np.ones(_SHEAR_CELLS, dtype=np.complex128)
    solid = np.ones(_SHEAR_CELLS, dtype=bool)
    p11, p12 = np.full(_SHEAR_CELLS, _P11), np.full(_SHEAR_CELLS, _P12)
    strain_xx, strain_yy, strain_xy = _SHEAR_STRAIN
    x_nodes, y_nodes = np.meshgrid(cell * np.arange(n_x + 1), cell * np.arange(n_y + 1), indexing='ij')
    nodal = np.stack((strain_xx * x_nodes + strain_xy * y_nodes, strain_xy * x_nodes + strain_yy * y_nodes), axis=-1)
    window = _full_window(_SHEAR_CELLS)
    materials = cross_section_modes.CrossSectionMaterials(eps_xx=eps, eps_yy=eps, eps_zz=eps, mu=ones,
                                                          pec=np.zeros(_SHEAR_CELLS, dtype=bool))
    points = cross_section_modes.mode_point_materials(
        (dx, dy), BoundaryConfig('periodic', 'periodic', 'periodic', 'periodic'), cell, materials)
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    scale = -EPS0 * area * _SHEAR_EPS ** 2
    errors, details = {}, {}
    for theta in _SHEAR_ANGLES:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        ex, ey = cos_t * ones, sin_t * ones
        mode = types.SimpleNamespace(ex=ex, ey=ey, ez=0.0 * ones, x_face_area=np.outer(dx, dy),
                                     y_face_area=np.outer(dx, dy), power=1.0, omega=omega, mirror_factor=1.0,
                                     n_eff=1.5)
        overlaps = {'hz': optomechanics.in_plane_hz_overlap(ones, ex, ey, eps, eps, eps, eps, ones, (dx, dy), window,
                                                            (True, True), omega),
                    'cross_section': optomechanics.cross_section_overlap(mode, points, window)}
        along = strain_xx * cos_t ** 2 + strain_yy * sin_t ** 2 + 2.0 * strain_xy * cos_t * sin_t
        expected = {'unset': scale * (_P11 * along + _P12 * (strain_xx + strain_yy - along)),
                    'explicit': scale * ((_P11 * strain_xx + _P12 * strain_yy) * cos_t ** 2
                                         + (_P11 * strain_yy + _P12 * strain_xx) * sin_t ** 2
                                         + 4.0 * _SHEAR_P44 * strain_xy * cos_t * sin_t)}
        shear = {'unset': np.full(_SHEAR_CELLS, np.nan), 'explicit': np.full(_SHEAR_CELLS, _SHEAR_P44)}
        for path, overlap in overlaps.items():
            for variant, p44 in shear.items():
                _, photo = optomechanics.overlap_integrals(overlap, nodal, solid, p11, p12, dx, dy, p44)
                key = f'{path} {variant} theta={theta:g}'
                errors[key] = abs(photo - expected[variant]) / abs(expected[variant])
                details[key] = {'measured': [photo.real, photo.imag], 'expected': expected[variant]}
    worst = max(errors.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _SHEAR_TOLERANCE,
            'details': dict(details, errors=errors)}


OPTOMECHANICS_CASES = (
    SelfTestCase(
        name='om_photoelastic_uniform_cavity',
        label='Optomechanics: photoelastic shift of a uniformly strained cavity',
        description=f'CPU: ez cavity of {_UNIFORM_CELLS[0]} x {_UNIFORM_CELLS[1]} uniform {_UNIFORM_CELL * 1e9:g} nm '
                    f'cells filled with eps {_UNIFORM_EPS:g} (dense symmetric -lap E = k^2 eps E, E = 0 on the outer '
                    f'faces), prescribed Q = (S_xx x, S_yy y), S = {_UNIFORM_STRAIN}, p12 {_P12:g}. A uniform eps '
                    'scaling gives d omega / omega = -Delta eps / (2 eps) exactly: the photoelastic g_om of modes 1-'
                    f'{_UNIFORM_MODES} must equal omega eps p12 (S_xx + S_yy) / 2 and the moving-boundary part 0 '
                    f'(relative {_UNIFORM_TOLERANCE:g}); likewise the hz cavity (the FDFD hz operator of '
                    'operators.yee_operators / helmholtz_operator, pec box, the static H_z skipped) under isotropic S = '
                    f'{_UNIFORM_HZ_STRAIN:g}: omega eps (p11 + p12) S / 2. Pins the strain of nodal Q, the tensor sign '
                    'and the ez (electric) / hz (magnetic energy) normalisations.',
        specs=no_runs, check=_check_uniform_cavity),
    SelfTestCase(
        name='om_photoelastic_hellmann_feynman',
        label='Optomechanics: photoelastic overlap vs eigen-solve differences',
        description=f'CPU: (1) ez cavity ({_CAVITY_CELLS[0]} x {_CAVITY_CELLS[1]} cells, a strained eps '
                    f'{_BLOCK_EPS:g} block, smooth bump Q): photoelastic g_om of mode 1 vs (omega(eps + alpha Delta '
                    'eps) - omega(eps - alpha Delta eps)) / (2 alpha) of the dense eigen-solve; (1b) the same for the hz '
                    'cavity (FDFD hz operator: d eps_xx on the E_x, d eps_yy on the E_y faces, face-averaged, p11 != '
                    'p12 pins the component mapping); (2) the x-periodic '
                    f'y-pec slab of cross_section_modes (n {_SLAB_INDEX:g} in {_CLADDING_INDEX:g}, half-width '
                    f'{_TWIN_HALF_WIDTH * 1e6:g} um, {_TWIN_CELL * 1e9:g} nm cells, TE0 = E_x only): d beta / d alpha '
                    'of the photoelastic overlap vs the central difference of two mode solves, all with p44 = 0 (the '
                    'operators hold diagonal eps only; the shear term: om_photoelastic_shear_rotation). Both are exact '
                    'first-order perturbations of their discrete operators (symmetric ez pencil; the TE slab line '
                    'operator with E_x = (k0 / beta) eta0 H_x), so only the central difference (max |alpha Delta '
                    f'eps / eps| = {_FD_STEP:g}) and the eigen-solve round-off remain: relative {_HF_TOLERANCE:g}. '
                    'Not yet run on the 20 x 16 cavity (was 30 x 22).',
        specs=no_runs, check=_check_hellmann_feynman),
    SelfTestCase(
        name='om_moving_boundary_convergence',
        label='Optomechanics: moving-boundary overlap vs slab and cavity relations',
        description='CPU, a breathing Q (+1 / -1 on the two interfaces) on two meshes '
                    f'({_MESH_CELLS[0] * 1e9:g} and {_MESH_CELLS[1] * 1e9:g} nm): d beta / d a of the TE0 and TM0 '
                    f'modes of a symmetric slab (n {_SLAB_INDEX:g} in {_CLADDING_INDEX:g}, half-width '
                    f'{_SLAB_HALF_WIDTH * 1e6:g} um; cross_section_modes, x periodic, pec walls '
                    f'{_SLAB_CLADDING * 1e6:g} um out) vs the implicit derivative of kappa tan(kappa a) = gamma (TM: '
                    'kappa / n1^2, gamma / n2^2), and d omega / d a of the even E_z mode of a slab (n '
                    f'{_CAVITY_INDICES[0]:g} in {_CAVITY_INDICES[1]:g}, |x| < {_CAVITY_HALF_WIDTH * 1e6:g} um) between '
                    f'E = 0 walls at |x| = {_CAVITY_HALF_LENGTH * 1e6:g} um vs n1 tan(k n1 a) = n2 cot(k n2 (L - a)); '
                    'hz: the same slab and walls times cos(q t) across a pec box of '
                    f'{_HZ_TRANSVERSE[1] * 1e6:g} um ({_HZ_TRANSVERSE[0]} cells, q = its discrete wavenumber, '
                    'evanescent outside: tangential E_t and normal D_n at the slab faces), slab normal along x (hz_x) '
                    'and along y (hz_y), the separable field of the FDFD hz operator (residual <= '
                    f'{_RESIDUAL_TOLERANCE:g}) vs (kappa1 / eps1) tan(kappa1 a) + (kappa2 / eps2) tan(kappa2 (L - a)) = '
                    '0, kappa_i^2 = k^2 eps_i - q^2. '
                    'The staircase face rule converges (TM and hz pin the |D_n|^2 term): the fine-mesh relative error '
                    f'must stay below {_MOVING_TOLERANCES} and the error fall by at least {_ORDER_RATIO} from the '
                    f'coarse mesh (unless below {_ORDER_FLOOR:g}); measured = the worst fine error over its '
                    'tolerance, error adds 1 per violation.',
        specs=no_runs, check=_check_moving_boundary),
    SelfTestCase(
        name='om_effective_mass_identity',
        label='Optomechanics: effective mass and normalisation invariance of g0',
        description=f'CPU: Q1 plate of {_PLATE_CELLS[0]} x {_PLATE_CELLS[1]} cells ({_PLATE_CELL * 1e6:g} um, silicon '
                    'elastic constants, rollers on x_lo / y_lo), host K and M from the solver stencils, modal.'
                    f'solve_modes for {_PLATE_MODES} modes. m_eff Omega^2 must equal L Q^T K Q / |Q_peak|^2 (relative '
                    f'{_MASS_TOLERANCE:g}: m_eff uses the mass of the solve), and g0 with a synthetic ez overlap (an eps '
                    'jump, photoelastic cells) must not depend on the reference node (peak node vs another node: '
                    f'relative {_INVARIANCE_TOLERANCE:g}). measured = the worst error in tolerance units.',
        specs=no_runs, check=_check_effective_mass),
    SelfTestCase(
        name='om_photoelastic_shear_rotation',
        label='Optomechanics: shear photoelastic term and isotropy',
        description=f'CPU: {_SHEAR_CELLS[0]} x {_SHEAR_CELLS[1]} periodic cells, eps {_SHEAR_EPS:g}, uniform strain '
                    f'(S_xx, S_yy, S_xy) = {_SHEAR_STRAIN} of a nodal Q, p11 {_P11:g}, p12 {_P12:g}, a uniform in-plane '
                    f'E = (cos theta, sin theta) at theta = {_SHEAR_ANGLES} rad, through the em_fdfd_hz face overlap '
                    'and the em_modes cross-section overlap. Unset p44 (NaN = (p11 - p12) / 2) must give eps0 A d eps '
                    'along E of the rotated diagonal strain, -eps^2 (p11 S_ee + p12 S_tt) (an isotropic tensor is '
                    f'rotation invariant: pins the Voigt factor 2 S_xy, the sign and the default); p44 {_SHEAR_P44:g} '
                    'the component sum with d eps_xy = -eps^2 2 p44 S_xy. Relative '
                    f'{_SHEAR_TOLERANCE:g}. The shear term itself has no discrete-operator reference (diagonal eps).',
        specs=no_runs, check=_check_shear_rotation),
)


# ---- 6 (GPU): om_strain_optic_chain ----
# x-periodic (2 cells) TE slab of em_modes between pec walls in y; the solid fills the domain with one elastic material
# and rollers on all four faces, so its lowest mode is u_y = sin(pi y / L_y) exactly (Q1, any n_y). A lumped body force
# f_y = F sin(pi y / L_y) at the cell centres is proportional to M Q, so the static solve is u = U Q exactly; U = peak
# centre u_y / cos^2(pi dy / (2 L_y)) (n_y even). The core sits below the middle, where S_yy = U (pi / L_y) cos(pi y /
# L_y) > 0. Table dn_eff_dx (photoelastic only) vs (n_eff(+F) - n_eff(-F)) / (2 U) of two mode_analysis runs with
# solid_em_strain_optic (deformation off): TE0 = E_x only, so the overlap is the exact first-order perturbation.
_CHAIN_CELL = 10e-9                          # m
_CHAIN_SIZE = (2 * _CHAIN_CELL, 2.0e-6)      # m: x periodic, y between the pec walls / rollers
_CHAIN_CORE = (0.49e-6, 0.71e-6)             # m: y range of the 220 nm core
_CHAIN_INDICES = (3.48, 1.444)               # core, cladding
_CHAIN_ELASTIC = (1.65e11, 0.22, 2329.0)     # E [Pa], nu, rho [kg/m^3] of every cell
_CHAIN_STRAIN = 1e-3                         # peak S_yy of the static runs
_CHAIN_TOLERANCE = 1e-3                      # relative: float32 eps / strain fields in the difference (~1e-4)
_CHAIN_RUNS = ('optomechanics', 'plus', 'minus')


def _chain_force():
    """F [N/m^3] of a static peak strain _CHAIN_STRAIN: (lambda + 2 mu) (pi / L_y)^2 U (continuum; U is measured)."""
    youngs, poisson, _ = _CHAIN_ELASTIC
    longitudinal = youngs * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return longitudinal * (math.pi / _CHAIN_SIZE[1]) * _CHAIN_STRAIN


def _chain_spec(run):
    """'optomechanics': the g0 table (moving boundary off); 'plus' / 'minus': mode_analysis of the strain-optic slab
    under +F / -F."""
    size_x, size_y = _CHAIN_SIZE
    youngs, poisson, density = _CHAIN_ELASTIC
    mechanics = {'youngs_modulus': youngs, 'poisson_ratio': poisson, 'rho': density, 'photoelastic_p11': _P11,
                 'photoelastic_p12': _P12}
    em_settings = {'wavelength': _WAVELENGTH, 'n_modes': 2, 'group_index': run == 'optomechanics',
                   'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec'}
    features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi']), FeatureSpec('ends', 'roller', ['y_lo', 'y_hi'])]
    couplings = []
    if run == 'optomechanics':
        features.append(FeatureSpec('g0', 'optomechanical_coupling',
                                    params={'optical_physics': 'modes', 'optical_mode': 1, 'length': 1e-5,
                                            'moving_boundary': False}))
        study = StudySpec(type='optomechanics', settings={'n_mechanical_modes': 1})
    else:
        force = _chain_force() * (1.0 if run == 'plus' else -1.0)
        features.append(FeatureSpec('load', 'body_force', ['all'],
                                    {'f_x': 0.0, 'f_y': f'{force!r} * sin(pi * y / {size_y!r})'}))
        couplings.append(CouplingSpec('strain_optic', 'solid_em_strain_optic', 'solid', 'modes',
                                      params={'deformation': False}))
        study = StudySpec(type='mode_analysis')
    spec = ModelSpec(
        title=f'Self-test: optomechanical strain-optic chain ({run})',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('core', 'rectangle', {'x0': 0.0, 'y0': _CHAIN_CORE[0], 'x1': size_x,
                                                     'y1': _CHAIN_CORE[1]})],
        materials=[MaterialSpec('cladding', properties={'eps_r': _CHAIN_INDICES[1] ** 2, **mechanics},
                                selection=['all']),
                   MaterialSpec('core', properties={'eps_r': _CHAIN_INDICES[0] ** 2, **mechanics},
                                selection=['core'])],
        physics=[PhysicsSpec('modes', 'em_modes', settings=em_settings,
                             features=[FeatureSpec('mode_table', 'mode_table')]),
                 PhysicsSpec('solid', 'solid_mechanics', features=features)],
        couplings=couplings,
        mesh=MeshSpec(mode='uniform', dx=_CHAIN_CELL),
        study=study)
    return spec if run == 'optomechanics' else with_exact_cg(spec)


def _chain_specs():
    return {run: _chain_spec(run) for run in _CHAIN_RUNS}


def _check_chain(results_by_name):
    """Table dn_eff_dx of mechanical mode 1 vs the strain-optic central difference."""
    table = results_by_name['optomechanics'].result('solid', 'g0')
    expected = float(table['rows'][0][table['columns'].index('dn_eff_dx')])
    edge_factor = math.cos(0.5 * math.pi * _CHAIN_CELL / _CHAIN_SIZE[1]) ** 2
    n_effs, amplitudes, te_fractions = [], [], []
    for run in ('plus', 'minus'):
        results = results_by_name[run]
        modes = results.result('modes', 'mode_table')
        columns = modes['columns']
        n_effs.append(float(modes['rows'][0][columns.index('n_eff')]))
        te_fractions.append(float(modes['rows'][0][columns.index('te_fraction')]))
        u_y = np.asarray(results.field('solid.u_y').values, dtype=np.float64)
        amplitudes.append(float(np.max(np.abs(u_y))) / edge_factor)
    measured = (n_effs[0] - n_effs[1]) / (amplitudes[0] + amplitudes[1])
    error = abs(measured / expected - 1.0) if expected != 0.0 else math.inf
    return {'measured': measured, 'expected': expected, 'error': error, 'tolerance': _CHAIN_TOLERANCE,
            'details': {'n_eff_plus_minus': n_effs, 'amplitudes': amplitudes, 'te_fractions': te_fractions,
                        'mechanical_frequency': float(table['rows'][0][table['columns'].index('frequency')])}}


OPTOMECHANICS_CHAIN_CASES = (
    SelfTestCase(
        name='om_strain_optic_chain',
        label='Optomechanics: model g0 table vs strain-optic finite difference',
        description=f'GPU, three runs: an x-periodic ({_CHAIN_SIZE[0] * 1e9:g} nm, 2 cells) silicon-like slab (n '
                    f'{_CHAIN_INDICES[0]:g}, y {_CHAIN_CORE[0] * 1e6:g}-{_CHAIN_CORE[1] * 1e6:g} um) in n '
                    f'{_CHAIN_INDICES[1]:g} between em_modes pec walls {_CHAIN_SIZE[1] * 1e6:g} um apart, '
                    f'{_CHAIN_CELL * 1e9:g} nm cells; the solid fills it with one elastic material (E '
                    f'{_CHAIN_ELASTIC[0]:g} Pa, nu {_CHAIN_ELASTIC[1]:g}, p11 {_P11:g}, p12 {_P12:g}) and rollers on '
                    'every face, so its lowest mode is u_y = sin(pi y / L_y) exactly. (1) optomechanics study, '
                    'optomechanical_coupling with moving_boundary off: column dn_eff_dx of mode 1; (2, 3) mode_analysis '
                    'with a static body force f_y = +/-F sin(pi y / L_y) (proportional to M Q, so u = U Q exactly, U '
                    'from the peak cell-centre u_y) and solid_em_strain_optic (deformation off): (n_eff(+) - n_eff(-)) '
                    f'/ (2 U) at peak strain {_CHAIN_STRAIN:g}. TE0 (E_x only) makes the overlap the exact first-order '
                    'perturbation; float32 eps and strain fields leave ~1e-4: relative tolerance '
                    f'{_CHAIN_TOLERANCE:g}. Pins the model path (mode selection, peak normalisation, strain and sign '
                    'conventions) against an existing coupling.',
        specs=_chain_specs, check=_check_chain),
)


# ---- template: silicon_ring_breathing_g0 ----
# Quarter of a suspended 450 x 220 nm silicon strip in air (x_lo / y_lo through its centre): em antisymmetric x_lo and
# symmetric y_lo select the quasi-TE0 mode (E_x even about both planes), solid rollers on the same faces keep the
# vibration modes even; CPML on x_hi / y_hi, uniform 5 nm cells (core 45 x 22). Bend off: R = 10 um enters through
# the length 2 pi R (the straight-guide approximation of the ring cross-section, first order in w / R). No closed form.
SILICON_RING_G0_LABEL = 'Silicon ring breathing g0'
SILICON_RING_G0_DESCRIPTION = (
    'Optomechanics study: quasi-TE0 mode (em_modes, 1.55 um, group index) and the lowest in-plane vibration modes of a '
    'suspended 450 x 220 nm silicon strip (quarter domain with mirrors and rollers), optomechanical_coupling table of '
    'g0 / 2 pi (moving boundary, photoelastic, total) for a ring of R = 10 um (length 2 pi R).')
_RING_DOMAIN = (1.0e-6, 0.8e-6)          # m, quarter cross-section
_RING_CORE = (0.225e-6, 0.11e-6)         # m, half width / half height
_RING_RADIUS = 10e-6                     # m


def silicon_ring_breathing_g0_spec():
    """Template 'silicon_ring_breathing_g0' (em_modes + solid_mechanics, optomechanics study)."""
    em_settings = {'wavelength': 1.55e-6, 'n_modes': 2, 'group_index': True, 'bc_x_lo': 'antisymmetric',
                   'bc_y_lo': 'symmetric', 'bc_x_hi': 'cpml', 'bc_y_hi': 'cpml'}
    return ModelSpec(
        title=SILICON_RING_G0_LABEL,
        domain=DomainSpec(size_x=_RING_DOMAIN[0], size_y=_RING_DOMAIN[1]),
        geometry=[GeometrySpec('core', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _RING_CORE[0], 'y1': _RING_CORE[1]})],
        materials=[MaterialSpec('air', library='air', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['core'])],
        physics=[PhysicsSpec('modes', 'em_modes', settings=em_settings,
                             features=[FeatureSpec('mode_table', 'mode_table')]),
                 PhysicsSpec('solid', 'solid_mechanics',
                             features=[FeatureSpec('mirrors', 'roller', ['x_lo', 'y_lo']),
                                       FeatureSpec('g0', 'optomechanical_coupling',
                                                   params={'optical_physics': 'modes', 'optical_mode': 1,
                                                           'length': 2.0 * math.pi * _RING_RADIUS})])],
        mesh=MeshSpec(mode='uniform', dx=5e-9),
        study=StudySpec(type='optomechanics', settings={'n_mechanical_modes': 6}))


# ---- template: silicon_disk_wgm_g0 ----
# Quarter of a 2D silicon disk (R = 1.5 um, centre at the domain corner) in air: em_fdfd_hz (in-plane E, the TE-like
# whispering-gallery family of a thin disk; the radial E is normal to the rim, so the |D_n|^2 term matters) with
# symmetric mirrors on x_lo / y_lo and CPML on x_hi / y_hi, eigen-solve near 1.55 um; plane-strain solid (a disk seen
# from the top is closer to plane stress: approximate) with rollers on the mirror faces, modes near the radial breathing
# estimate ~2 c / (2 pi R); length = the 220 nm thickness. 20 nm cells (~22 per wavelength in silicon). No closed form.
SILICON_DISK_G0_LABEL = 'Silicon disk breathing g0'
SILICON_DISK_G0_DESCRIPTION = (
    'Optomechanics study: in-plane whispering-gallery resonance (em_fdfd_hz eigen-solve near 1.55 um, quarter domain '
    'with mirrors) and the vibration modes of a 2D silicon disk of R = 1.5 um in air (plane strain, rollers on the '
    'mirror faces), optomechanical_coupling table of g0 / 2 pi (moving boundary, photoelastic, total) for a 220 nm '
    'thick disk (length).')
_DISK_RADIUS = 1.5e-6                    # m
_DISK_DOMAIN = 2.3e-6                    # m, quarter domain side (0.8 um of air before the CPML)
_DISK_THICKNESS = 220e-9                 # m, the out-of-plane length of m_eff
_DISK_TARGET_FREQUENCY = 1.6e9           # Hz, near the radial breathing mode


def silicon_disk_wgm_g0_spec():
    """Template 'silicon_disk_wgm_g0' (em_fdfd_hz + solid_mechanics, optomechanics study)."""
    em_settings = {'wavelength': 1.55e-6, 'bc_x_lo': 'symmetric', 'bc_y_lo': 'symmetric', 'bc_x_hi': 'cpml',
                   'bc_y_hi': 'cpml'}
    return ModelSpec(
        title=SILICON_DISK_G0_LABEL,
        domain=DomainSpec(size_x=_DISK_DOMAIN, size_y=_DISK_DOMAIN),
        geometry=[GeometrySpec('disk', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': _DISK_RADIUS})],
        materials=[MaterialSpec('air', library='air', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['disk'])],
        physics=[PhysicsSpec('em', 'em_fdfd_hz', settings=em_settings),
                 PhysicsSpec('solid', 'solid_mechanics', settings={'target_frequency': _DISK_TARGET_FREQUENCY},
                             features=[FeatureSpec('mirrors', 'roller', ['x_lo', 'y_lo']),
                                       FeatureSpec('g0', 'optomechanical_coupling',
                                                   params={'optical_physics': 'em', 'optical_mode': 1,
                                                           'length': _DISK_THICKNESS})])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='optomechanics', settings={'n_mechanical_modes': 6, 'n_optical_resonances': 4}))

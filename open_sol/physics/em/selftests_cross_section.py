"""CPU self-tests of the full-vector cross-section mode solver (A5): slab limits against slab_modes, discrete analytic
PEC / PMC guides, mirror halves and quarter-domain step-index fibre modes against exact Bessel roots (no model run)."""
import functools
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase

_WAVELENGTH = 1.55e-6                             # m, vacuum
_CLADDING_INDEX = 1.444

# Slab limit: x periodic with one cell, slab normal y between pec walls; V = k0 d/2 sqrt(n1^2 - n2^2) = 2.24
_SLAB_CORE_INDEX = 2.0
_SLAB_THICKNESS = 0.8e-6                          # m: TE0, TE1, TM0, TM1 guided
_SLAB_CORE_CELL = 5e-9                            # m: 160 core cells
_SLAB_CLADDING = 3.0e-6                           # m per side: > 5 decay lengths of TM1 (0.57 um)
_SLAB_GROWTH, _SLAB_MAX_CELL = 1.05, 20e-9        # graded cladding: x1.05 per cell up to 20 nm
_SLAB_PERIOD = 50e-9                              # m: x width of the single periodic cell
_SLAB_MODES = 6                                   # TE0, TM0, TE1, TM1 + 2 cladding modes
_GROUP_STEP = 1e-3
_TE_TOLERANCE = 1e-9                              # |n_eff - slab 'ez'|
_PROFILE_TOLERANCE = 1e-8                         # E_x profile vs slab profile, relative to its peak
_GROUP_TOLERANCE = 1e-7                           # relative group-index difference
_POLARIZATION_PURITY = 1e-10                      # TE modes: 1 - te_fraction; TM modes: te_fraction
_TM_TOLERANCE = 2e-3                              # |n_eff - slab 'hz'|, see the case description

# Discrete analytic guides: (label, x faces, y faces, interior cells (Mx, My), cell (dx, dy) [m], eps, mu)
_GUIDE_CASES = (
    ('hollow_pec', 'pec', 'pec', (40, 30), (50e-9, 40e-9), 1.0, 1.0),
    ('filled_pec_x_pmc_y', 'pec', 'pmc', (20, 20), (50e-9, 40e-9), 2.25, 1.5),
)
_GUIDE_MODES = 6
_GUIDE_TOLERANCE = 1e-9                           # |beta^2 - exact| / (k0^2 eps mu)

# Mirrors: silicon-nitride-like strip (n = 2.0, 1.2 x 0.8 um) in silica, symmetric about faces
_STRIP_CORE_INDEX = 2.0
_STRIP_HALF_WIDTH, _STRIP_HALF_HEIGHT = 0.6e-6, 0.4e-6
_STRIP_CELL = 50e-9                               # the checked identity is exact at any cell: round-off only
_STRIP_PML_CLADDING = 0.6e-6                      # m of uniform cladding between the core and the CPML
_STRIP_PML_CELLS = 12
_STRIP_WALL_CLADDING = 1.0e-6                     # m of graded cladding between the core and the pec wall
_STRIP_GROWTH, _STRIP_MAX_CELL = 1.08, 100e-9
_MIRROR_MODES = 4
_MIRROR_TOLERANCE = 1e-9                          # |n_eff(full) - n_eff(half)|

# Step-index fibre on a quarter domain (mirrors on x_lo / y_lo, pec walls on x_hi / y_hi)
_FIBRE_CORE_INDEX, _FIBRE_CLADDING_INDEX = 1.50, 1.40
_FIBRE_RADIUS = 1.6e-6                            # m: V = 3.49, guided HE11, TE01, TM01, HE21 (next cut-off 3.83)
_FIBRE_CELL = 40e-9                               # m: 40 cells per radius
_FIBRE_EXTENT = 4.4e-6                            # m from the axis to the pec wall (> 3 LP11 decay lengths outside)
_FIBRE_SCAN_POINTS = 4001                         # sign-change scan of the exact relations over n2 < n_eff < n1
_FIBRE_TOLERANCE = 5e-4                           # |n_eff - exact|, see the case description
# (label, x_lo face, y_lo face, modes compared, exact families of those modes)
_FIBRE_SOLVES = (
    ('he11', 'antisymmetric', 'symmetric', 1, ('he11',)),
    ('te01_he21', 'antisymmetric', 'antisymmetric', 2, ('te01', 'he21')),
    ('tm01_he21', 'symmetric', 'symmetric', 2, ('he21', 'tm01')),
)


def _graded_cladding(start_cell, thickness, growth, max_cell):
    """Cell widths growing x growth from start_cell (capped at max_cell) until they cover thickness [m]."""
    widths, width, covered = [], start_cell, 0.0
    while covered < thickness - 1e-3 * start_cell:
        width = min(width * growth, max(max_cell, start_cell))
        widths.append(width)
        covered += width
    return np.array(widths)


def _isotropic(cross_section_modes, eps, mu=1.0):
    """CrossSectionMaterials of an isotropic (nx, ny) permittivity, uniform mu, no PEC."""
    eps = np.asarray(eps, dtype=np.complex128)
    return cross_section_modes.CrossSectionMaterials(
        eps_xx=eps, eps_yy=eps.copy(), eps_zz=eps.copy(), mu=np.full(eps.shape, mu, dtype=np.complex128),
        pec=np.zeros(eps.shape, dtype=bool))


def _slab_setup():
    """(dy (ny,) [m], eps (ny,) complex): ghost, graded cladding, core, cladding, ghost (ghosts copy their neighbour)."""
    cladding = _graded_cladding(_SLAB_CORE_CELL, _SLAB_CLADDING, _SLAB_GROWTH, _SLAB_MAX_CELL)
    n_core = int(round(_SLAB_THICKNESS / _SLAB_CORE_CELL))
    inner = np.concatenate((cladding[::-1], np.full(n_core, _SLAB_CORE_CELL), cladding))
    dy = np.concatenate(([inner[0]], inner, [inner[-1]]))
    eps = np.full(dy.size, _CLADDING_INDEX ** 2, dtype=np.complex128)
    core_start = 1 + cladding.size
    eps[core_start:core_start + n_core] = _SLAB_CORE_INDEX ** 2
    return dy, eps


@functools.lru_cache(maxsize=1)
def _slab_solve():
    """(modes, info) of the slab cross-section: one periodic x cell, pec walls on y; cached so the TE and TM checks
    share one solve (deterministic; callers must not mutate the modes)."""
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    dy, eps = _slab_setup()
    dx = np.array([_SLAB_PERIOD])
    solve = cross_section_modes.solve_cross_section_modes(
        (dx, dy), BoundaryConfig('periodic', 'periodic', 'pec', 'pec'), _SLAB_CORE_CELL,
        _isotropic(cross_section_modes, eps[None, :]), 2.0 * math.pi * C0 / _WAVELENGTH, _SLAB_MODES)
    return solve.modes, solve.info


def _slab_line(polarization):
    """SlabLine of the cross-section cells 1..ny-2 (between the wall planes): slab face k = hi face of cross-section
    cell k; ez 'odd' ends (the pec-wall image of E_x), hz 'neumann' ends with 1/eps of the width-weighted face mean."""
    from . import slab_modes
    from .yee_grid import face_average, yee_inverse_spacings
    dy, eps = _slab_setup()
    n_cells = dy.size - 2
    face_inverse_spacing = yee_inverse_spacings(dy, dy).inv_dx_h[:n_cells + 1]
    ones_cells, ones_faces = np.ones(n_cells, dtype=np.complex128), np.ones(n_cells + 1, dtype=np.complex128)
    if polarization == 'ez':
        cell_weight, face_weight, ends = ones_cells, ones_faces, ('odd', 'odd')
    else:
        cell_weight = eps[1:-1]
        face_weight = 1.0 / face_average(eps, 0, False, dy)[:n_cells + 1]
        ends = ('neumann', 'neumann')
    return slab_modes.SlabLine(
        polarization=polarization, cell_widths=dy[1:-1], cell_stretch=ones_cells, cell_weight=cell_weight,
        index_term=eps[1:-1], active=np.ones(n_cells, dtype=bool), face_inverse_spacing=face_inverse_spacing,
        face_stretch=ones_faces, face_weight=face_weight, lo_end=ends[0], hi_end=ends[1])


def _check_te_slab(results_by_name):
    """TE slab modes (E_x only) against slab_modes 'ez' on the same cells (n_eff, profile, group index, purity), plus
    the TM modes against slab 'hz' (a failed check when off by more than _TM_TOLERANCE)."""
    from . import cross_section_modes, slab_modes
    from .boundaries import BoundaryConfig
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    modes, info = _slab_solve()
    te_modes = [mode for mode in modes if mode.te_fraction > 0.5][:2]
    line = _slab_line('ez')
    references = slab_modes.solve_slab_modes(line, omega, 2)
    errors, profile_errors, group_errors, violations = {}, {}, {}, []
    if len(te_modes) < 2:
        violations.append(f'{len(te_modes)} TE modes returned, expected 2')
    for number, (mode, reference) in enumerate(zip(te_modes, references)):
        errors[f'n_eff_te{number}'] = abs(mode.n_eff - reference.n_eff)
        profile = mode.ex[0, 1:-1] * math.sqrt(_SLAB_PERIOD)   # 1 W over the period vs 1 W/m
        profile_errors[f'te{number}'] = (float(np.max(np.abs(profile - reference.profile)))
                                         / float(np.max(np.abs(reference.profile))))
        if 1.0 - mode.te_fraction > _POLARIZATION_PURITY:
            violations.append(f'TE{number}: te_fraction {mode.te_fraction!r} differs from 1')
    if len(te_modes) == 2:
        dy, eps = _slab_setup()
        materials = _isotropic(cross_section_modes, eps[None, :])
        with_group, _group_info = cross_section_modes.mode_group_indices(
            (np.array([_SLAB_PERIOD]), dy), BoundaryConfig('periodic', 'periodic', 'pec', 'pec'), _SLAB_CORE_CELL,
            lambda _omega: materials, omega, te_modes, relative_step=_GROUP_STEP)
        for number, (mode, reference) in enumerate(zip(with_group, references)):
            expected = slab_modes.group_index(lambda _omega: line, omega, reference, 3, relative_step=_GROUP_STEP)
            group_errors[f'te{number}'] = abs(mode.n_group - expected) / abs(expected)
    violations += [f'{key}: E_x profile differs by {value:.3g} (> {_PROFILE_TOLERANCE})'
                   for key, value in profile_errors.items() if not value <= _PROFILE_TOLERANCE]
    violations += [f'{key}: group index differs by {value:.3g} (> {_GROUP_TOLERANCE})'
                   for key, value in group_errors.items() if not value <= _GROUP_TOLERANCE]
    tm_outcome = _check_tm_slab(results_by_name)
    if not tm_outcome['error'] <= tm_outcome['tolerance']:
        violations.append(f"TM slab: error {tm_outcome['error']:.3g} (> {_TM_TOLERANCE}; see tm details)")
    measured = max(errors.values(), default=float('nan'))
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': _TE_TOLERANCE,
            'details': {'n_eff_errors': errors, 'profile_errors': profile_errors, 'group_index_errors': group_errors,
                        'n_eff': [[mode.n_eff.real, mode.n_eff.imag] for mode in te_modes],
                        'n_eff_slab': [reference.n_eff.real for reference in references],
                        'violations': violations, 'n_unknowns': info['n_unknowns'], 'tm': tm_outcome['details']}}


def _check_tm_slab(results_by_name):
    """TM slab modes (H_x, E_y, E_z) against slab_modes 'hz' (dual point set: discretisation-order agreement); part of
    em_cross_section_te_slab."""
    from . import slab_modes
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    modes, _info = _slab_solve()
    tm_modes = [mode for mode in modes if mode.te_fraction < 0.5][:2]
    references = slab_modes.solve_slab_modes(_slab_line('hz'), omega, 2)
    errors, violations = {}, []
    if len(tm_modes) < 2:
        violations.append(f'{len(tm_modes)} TM modes returned, expected 2')
    for number, (mode, reference) in enumerate(zip(tm_modes, references)):
        errors[f'n_eff_tm{number}'] = abs(mode.n_eff - reference.n_eff)
        if mode.te_fraction > _POLARIZATION_PURITY:
            violations.append(f'TM{number}: te_fraction {mode.te_fraction!r} differs from 0')
    measured = max(errors.values(), default=float('nan'))
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': _TM_TOLERANCE,
            'details': {'n_eff_errors': errors, 'n_eff': [[mode.n_eff.real, mode.n_eff.imag] for mode in tm_modes],
                        'n_eff_slab_hz': [reference.n_eff.real for reference in references],
                        'violations': violations}}


def _wall_wavenumbers_squared(kind, n_interior, cell):
    """(centre K^2, face K^2) of the 1D wall Laplacians between two planes M = n_interior cells apart, K = (2/d)
    sin(p pi / (2M)): pec centre p = 1..M (odd images), face p = 0..M (free); pmc centre p = 0..M-1, face p = 1..M-1."""
    if kind == 'pec':
        centre_orders, face_orders = np.arange(1, n_interior + 1), np.arange(0, n_interior + 1)
    else:
        centre_orders, face_orders = np.arange(0, n_interior), np.arange(1, n_interior)

    def squared(orders):
        return (2.0 / cell * np.sin(orders * math.pi / (2.0 * n_interior))) ** 2

    return squared(centre_orders), squared(face_orders)


def _guide_spectrum(k0, index_squared, x_kind, y_kind, interior, cells):
    """Exact discrete beta^2 of a homogeneous guide between walls, descending: h_x (x centre, y face) and h_y (x face,
    y centre) decouple into separable wall Laplacians, beta^2 = k0^2 eps mu - Kx^2 - Ky^2."""
    x_centre, x_face = _wall_wavenumbers_squared(x_kind, interior[0], cells[0])
    y_centre, y_face = _wall_wavenumbers_squared(y_kind, interior[1], cells[1])
    transverse = np.concatenate(((x_centre[:, None] + y_face[None, :]).ravel(),
                                 (x_face[:, None] + y_centre[None, :]).ravel()))
    return np.sort(k0 ** 2 * index_squared - transverse)[::-1]


def _check_pec_guide(results_by_name):
    """Homogeneous guides between pec / pmc walls against their exact discrete spectrum (the wall-plane rule)."""
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    k0 = 2.0 * math.pi / _WAVELENGTH
    errors, violations, details = {}, [], {}
    for label, x_kind, y_kind, interior, cells, eps, mu in _GUIDE_CASES:
        widths = (np.full(interior[0] + 2, cells[0]), np.full(interior[1] + 2, cells[1]))
        solve = cross_section_modes.solve_cross_section_modes(
            widths, BoundaryConfig(x_kind, x_kind, y_kind, y_kind), cells[0],
            _isotropic(cross_section_modes, np.full((interior[0] + 2, interior[1] + 2), eps), mu),
            C0 * k0, _GUIDE_MODES)
        expected = _guide_spectrum(k0, eps * mu, x_kind, y_kind, interior, cells)[:_GUIDE_MODES]
        measured = np.array([mode.beta ** 2 for mode in solve.modes])
        if measured.size != _GUIDE_MODES:
            violations.append(f'{label}: {measured.size} modes returned, expected {_GUIDE_MODES}')
            continue
        errors[label] = float(np.max(np.abs(measured - expected))) / (k0 ** 2 * eps * mu)
        details[label] = {'beta_squared_over_k0_squared': (measured.real / k0 ** 2).tolist(),
                          'expected_over_k0_squared': (expected / k0 ** 2).tolist()}
    measured_error = max(errors.values(), default=float('nan'))
    return {'measured': measured_error, 'expected': 0.0, 'error': measured_error + len(violations),
            'tolerance': _GUIDE_TOLERANCE, 'details': {'errors': errors, 'spectra': details, 'violations': violations}}


def _mirror_axis_widths(half_core):
    """(widths, L) of a mirror axis symmetric about the hi face of cell L - 1: ghost, graded cladding, core | core,
    cladding, ghost (the ghosts sit outside the pec wall planes)."""
    half = np.concatenate((np.full(int(round(half_core / _STRIP_CELL)), _STRIP_CELL),
                           _graded_cladding(_STRIP_CELL, _STRIP_WALL_CLADDING, _STRIP_GROWTH, _STRIP_MAX_CELL)))
    half = np.append(half, half[-1])
    return np.concatenate((half[::-1], half)), half.size


def _pml_axis_widths(half_core):
    """Uniform widths of a CPML axis: PML, cladding, core, cladding, PML (even count, core centred)."""
    half_cells = _STRIP_PML_CELLS + int(round((_STRIP_PML_CLADDING + half_core) / _STRIP_CELL))
    return np.full(2 * half_cells, _STRIP_CELL)


def _strip_boundary(boundary_config_class, mirror_axis, lo_kind):
    """BoundaryConfig with lo_kind / pec on the mirror axis and _STRIP_PML_CELLS CPML cells on both faces of the other."""
    mirror_faces, pml_faces = (lo_kind, 'pec'), ('cpml', 'cpml')
    config = boundary_config_class(*(mirror_faces + pml_faces if mirror_axis == 'x' else pml_faces + mirror_faces))
    config.cpml_npml = _STRIP_PML_CELLS
    return config


def _centred_coordinates(widths, centre_face):
    """Cell centres [m] relative to the face after cell centre_face - 1 (edge index centre_face)."""
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    return 0.5 * (edges[:-1] + edges[1:]) - edges[centre_face]


def _check_mirrors(results_by_name):
    """Symmetric + antisymmetric half domains reproduce the full-domain strip modes (x mirror with CPML on y, y mirror
    with CPML on x)."""
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    errors, violations, details = {}, [], {}
    for axis in ('x', 'y'):
        if axis == 'x':
            dx, n_half = _mirror_axis_widths(_STRIP_HALF_WIDTH)
            dy = _pml_axis_widths(_STRIP_HALF_HEIGHT)
            x_offsets, y_offsets = _centred_coordinates(dx, n_half), _centred_coordinates(dy, dy.size // 2)
        else:
            dx = _pml_axis_widths(_STRIP_HALF_WIDTH)
            dy, n_half = _mirror_axis_widths(_STRIP_HALF_HEIGHT)
            x_offsets, y_offsets = _centred_coordinates(dx, dx.size // 2), _centred_coordinates(dy, n_half)
        core = (np.abs(x_offsets)[:, None] < _STRIP_HALF_WIDTH) & (np.abs(y_offsets)[None, :] < _STRIP_HALF_HEIGHT)
        eps = np.where(core, _STRIP_CORE_INDEX ** 2, _CLADDING_INDEX ** 2)
        half_slice = (slice(n_half - 1, None), slice(None)) if axis == 'x' else (slice(None), slice(n_half - 1, None))
        half_widths = (dx[n_half - 1:], dy) if axis == 'x' else (dx, dy[n_half - 1:])
        full = cross_section_modes.solve_cross_section_modes(
            (dx, dy), _strip_boundary(BoundaryConfig, axis, 'pec'), _STRIP_CELL, _isotropic(cross_section_modes, eps),
            omega, _MIRROR_MODES).modes
        halves = [cross_section_modes.solve_cross_section_modes(
            half_widths, _strip_boundary(BoundaryConfig, axis, kind), _STRIP_CELL,
            _isotropic(cross_section_modes, eps[half_slice]), omega, _MIRROR_MODES).modes
            for kind in ('symmetric', 'antisymmetric')]
        union = sorted((mode.n_eff for modes in halves for mode in modes), key=lambda n_eff: -(n_eff ** 2).real)
        union = union[:_MIRROR_MODES]
        if len(full) != _MIRROR_MODES or len(union) != _MIRROR_MODES:
            violations.append(f'{axis} mirror: {len(full)} full / {len(union)} half modes, expected {_MIRROR_MODES}')
            continue
        errors[f'{axis}_mirror'] = max(abs(mode.n_eff - n_eff) for mode, n_eff in zip(full, union))
        details[f'{axis}_mirror'] = {'full': [mode.n_eff.real for mode in full],
                                     'symmetric': [mode.n_eff.real for mode in halves[0]],
                                     'antisymmetric': [mode.n_eff.real for mode in halves[1]]}
    measured = max(errors.values(), default=float('nan'))
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': _MIRROR_TOLERANCE,
            'details': {'errors': errors, 'n_eff': details, 'violations': violations}}


def _fibre_mismatch(special, n_eff, vacuum_wavenumber, radius, family):
    """Pole-free exact step-index relation at n_eff (u = k0 a sqrt(n1^2 - n^2), w = k0 a sqrt(n^2 - n2^2)): TE0m
    J1 w K0 + K1 u J0, TM0m with n1^2 / n2^2 weights, HE / EH (order nu) the hybrid relation times (u J w K)^2."""
    n1, n2 = _FIBRE_CORE_INDEX, _FIBRE_CLADDING_INDEX   # n_eff: a float or an array (the scan grid)
    u = vacuum_wavenumber * radius * np.sqrt(n1 ** 2 - n_eff ** 2)
    w = vacuum_wavenumber * radius * np.sqrt(n_eff ** 2 - n2 ** 2)
    if family == 'te':
        return special.j1(u) * w * special.k0(w) + special.k1(w) * u * special.j0(u)
    if family == 'tm':
        return n1 ** 2 * special.j1(u) * w * special.k0(w) + n2 ** 2 * special.k1(w) * u * special.j0(u)
    j, j_prime = special.jv(family, u), special.jvp(family, u)
    k, k_prime = special.kv(family, w), special.kvp(family, w)
    first = j_prime * w * k + k_prime * u * j
    second = n1 ** 2 * j_prime * w * k + n2 ** 2 * k_prime * u * j
    return first * second - (family * n_eff * (1.0 / u ** 2 + 1.0 / w ** 2) * u * w * j * k) ** 2


def _fibre_index(special, optimize, vacuum_wavenumber, radius, family):
    """Largest exact n_eff in (n2, n1) of family 'te', 'tm' or hybrid order nu (int): scan for sign changes, brentq."""
    grid = np.linspace(_FIBRE_CLADDING_INDEX, _FIBRE_CORE_INDEX, _FIBRE_SCAN_POINTS)[1:-1]

    def mismatch(n_eff):
        return _fibre_mismatch(special, float(n_eff), vacuum_wavenumber, radius, family)

    values = _fibre_mismatch(special, grid, vacuum_wavenumber, radius, family)
    changes = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    if changes.size == 0:
        raise ValueError(f'no guided root of fibre family {family!r}')
    index = int(changes[-1])
    return float(optimize.brentq(mismatch, grid[index], grid[index + 1], xtol=1e-15))


def _check_fibre(results_by_name):
    """Quarter-domain step-index fibre: HE11, TE01 / HE21, TM01 / HE21 against the exact relations at the
    area-matched radius of the staircase core."""
    import scipy.optimize
    import scipy.special
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    vacuum_wavenumber = 2.0 * math.pi / _WAVELENGTH
    n_cells = int(round(_FIBRE_EXTENT / _FIBRE_CELL)) + 2
    widths = np.full(n_cells, _FIBRE_CELL)
    offsets = (np.arange(n_cells) - 0.5) * _FIBRE_CELL       # centre minus the mirror plane (hi face of cell 0)
    core = offsets[:, None] ** 2 + offsets[None, :] ** 2 < _FIBRE_RADIUS ** 2
    radius = math.sqrt(4.0 * np.count_nonzero(core[1:-1, 1:-1]) * _FIBRE_CELL ** 2 / math.pi)
    materials = _isotropic(cross_section_modes, np.where(core, _FIBRE_CORE_INDEX ** 2, _FIBRE_CLADDING_INDEX ** 2))
    exact = {'he11': 1, 'he21': 2, 'te01': 'te', 'tm01': 'tm'}
    exact = {name: _fibre_index(scipy.special, scipy.optimize, vacuum_wavenumber, radius, family)
             for name, family in exact.items()}
    errors, measured_n_eff, violations = {}, {}, []
    for label, x_face, y_face, count, families in _FIBRE_SOLVES:
        modes = cross_section_modes.solve_cross_section_modes(
            (widths, widths), BoundaryConfig(x_face, 'pec', y_face, 'pec'), _FIBRE_CELL, materials,
            C0 * vacuum_wavenumber, count).modes
        measured_n_eff[label] = [mode.n_eff.real for mode in modes]
        if len(modes) != count:
            violations.append(f'{label}: {len(modes)} modes returned, expected {count}')
            continue
        expected = sorted((exact[family] for family in families), reverse=True)
        for number, (mode, value) in enumerate(zip(modes, expected)):
            errors[f'{label}_{number}'] = abs(mode.n_eff - value)
    measured = max(errors.values(), default=float('nan'))
    v_number = vacuum_wavenumber * radius * math.sqrt(_FIBRE_CORE_INDEX ** 2 - _FIBRE_CLADDING_INDEX ** 2)
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': _FIBRE_TOLERANCE,
            'details': {'errors': errors, 'n_eff_measured': measured_n_eff, 'n_eff_exact': exact,
                        'area_matched_radius': radius, 'v_number': v_number,
                        'te01_minus_tm01_exact': exact['te01'] - exact['tm01'], 'violations': violations}}


CROSS_SECTION_CPU_SELFTESTS = (
    SelfTestCase(
        name='em_cross_section_te_slab',
        label='EM cross-section modes: TE and TM slab',
        description='CPU check of physics/em/cross_section_modes (no model run) in the slab limit: one periodic x cell '
                    '(50 nm), slab normal y between pec walls, core n = 2.0, d = 0.8 um in n = 1.444 at 1.55 um '
                    '(V = 2.24), 5 nm core cells, cladding graded x1.05 up to 20 nm over 3 um per side. The TE modes '
                    '(E_x, H_y, H_z; te_fraction > 0.5) decouple and their H_y block equals the slab_modes ez '
                    'operator on the cells between the wall planes with odd ends (the pec image rule), so TE0 / TE1 '
                    'n_eff must equal solve_slab_modes ez to round-off, the E_x profile (times sqrt(period): 1 W over '
                    'the period vs 1 W/m) its profile to 1e-8, the group index (central difference, relative step '
                    '1e-3, mode_group_indices with overlap tracking) slab_modes.group_index to 1e-7 relative, and '
                    'te_fraction 1 to 1e-10. error = max |n_eff - slab| + the number of failed checks; tolerance 1e-9 '
                    '(both eigen-solves reach ~eps ||T|| / k0^2 ~ 1e-12). The same solve also checks '
                    'the TM modes (H_x, E_y, E_z; te_fraction '
                    '< 0.5, which must be 0 to 1e-10) against solve_slab_modes hz on the same cells (neumann ends), a '
                    'failed check when |n_eff - slab hz| > 2e-3 or a TM te_fraction is off. '
                    'The point sets are dual (cross-section H_x and the normal E_y share the interface face, the slab '
                    'line has H_z at the centres), so they agree to discretisation order only. The cross-section B '
                    'weight 1/eps_y uses the arithmetic face mean where the consistent weight is mean(1/eps): a '
                    'first-order error, by perturbation theory |dn| ~ n (eps1 - eps2)^2 / (2 eps1 eps2 (eps1 + '
                    'eps2)) |H_if|^2 dx / int |H|^2 / eps dy = 2.6e-4 (TM0) and 6.1e-4 (TM1) at 5 nm, the hz slab '
                    'adding ~1e-4 (a wrong wall rule or coupling term shifts n_eff by > 1e-2). Harmonic face averaging '
                    'of the normal component is a flagged follow-up.',
        specs=lambda: {},
        check=_check_te_slab,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_cross_section_pec_guide',
        label='EM cross-section modes: walled guides',
        description='Homogeneous guides between walls against their exact discrete spectrum (no model run): hollow '
                    'pec rectangle of 40 x 30 interior cells (dx 50 nm, dy 40 nm) and a pec-x / pmc-y guide of 20 x 20 '
                    'cells filled with eps 2.25, mu 1.5 (it carries a TEM mode, beta = k0 sqrt(eps mu), exactly at '
                    'the default shift but for its 1e-6 offset). In a homogeneous medium the H_x (x centre, y face) '
                    'and H_y (x face, y centre) blocks decouple into separable wall Laplacians, so beta^2 = k0^2 eps '
                    'mu - Kx^2 - Ky^2, K = (2/d) sin(p pi / (2M)) with M the cells between the wall planes (inner '
                    'faces of the outermost cells), pec: centre p = 1..M, face p = 0..M; pmc: centre p = 0..M-1, '
                    'face p = 1..M-1. The 6 largest beta^2 (degenerate TE / TM pairs included) must match; a plane '
                    'half a cell off changes Kx^2 by ~2 dx / a ~ 5 %. error = max |beta^2 - exact| / (k0^2 eps mu) + '
                    'failed checks; tolerance 1e-9 (round-off of the shift-invert solve).',
        specs=lambda: {},
        check=_check_pec_guide,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_cross_section_mirrors',
        label='EM cross-section modes: mirror halves',
        description='Mirror rule of physics/em/cross_section_modes (no model run): a strip (n = 2.0, 1.2 x 0.8 um, '
                    f'silica cladding, {_STRIP_CELL * 1e9:g} nm cells) symmetric about a face of a graded grid '
                    f'(cladding x{_STRIP_GROWTH:g} up to {_STRIP_MAX_CELL * 1e9:g} nm, pec walls 1 um from the core) '
                    f'along the mirror axis and with {_STRIP_PML_CELLS} CPML cells along the '
                    'other (complex eigenproblem, PML-mode filter). The 4 leading full-domain modes must equal the '
                    '4 leading modes of the symmetric and antisymmetric half domains together (the full operator '
                    'restricted to even / odd fields is the half operator with the ghost image rule, exactly, mu = 1 '
                    'and symmetric widths), once with the mirror on x_lo, once on y_lo. error = max |n_eff(full) - '
                    'n_eff(halves)| + failed checks; tolerance 1e-9 (round-off of two eigen-solves). Not yet run with '
                    '50 nm cells.',
        specs=lambda: {},
        check=_check_mirrors,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_cross_section_fibre',
        label='EM cross-section modes: step-index fibre',
        description='Vector modes of a step-index fibre (no model run): n1 = 1.50, n2 = 1.40, a = 1.6 um at 1.55 um '
                    '(V = 3.49: HE11, TE01, TM01, HE21 guided), 40 nm cells on a quarter domain with mirrors on x_lo '
                    f'/ y_lo through the axis and pec walls {_FIBRE_EXTENT * 1e6:g} um from it: antisymmetric x / '
                    'symmetric y gives HE11 '
                    '(x-polarised), antisymmetric / antisymmetric TE01 and HE21, symmetric / symmetric TM01 and HE21. '
                    'Exact n_eff from the pole-free relations (TE: J1 w K0 + K1 u J0 = 0, TM with n1^2, n2^2 weights, '
                    'hybrid: (J\'/(uJ) + K\'/(wK)) (n1^2 J\'/(uJ) + n2^2 K\'/(wK)) = nu^2 n^2 (1/u^2 + 1/w^2)^2 '
                    'times (u J w K)^2; scipy Bessel functions, sign-change scan + brentq, largest root) at the '
                    'radius of a disc with the staircase core area. error = max |n_eff - exact| + failed checks; '
                    'tolerance 5e-4: estimated errors are ~2e-5 (HE11) to ~1e-4 (TM01) from the arithmetic face mean '
                    'of the normal eps (first order, ~(eps1 - eps2)^2), up to ~1e-4 from the staircase beyond its '
                    'area, up to ~5e-5 (HE21) from the walls (> 3 decay lengths) and ~1e-5 second-order bulk '
                    'dispersion; the details list the exact TE01 - TM01 splitting a scalar solver would miss. Not yet '
                    'run with the 4.4 um walls.',
        specs=lambda: {},
        check=_check_fibre,
        max_wall_time=120.0),
)

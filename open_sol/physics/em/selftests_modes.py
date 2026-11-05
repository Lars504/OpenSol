"""CPU self-tests of the 1D slab mode solver (A3): analytic slab dispersion, exact discrete eigenvalues and operator
residuals, normalisation and modal projections; no model run (specs() is empty, check({}) computes everything)."""
import cmath
import math
from dataclasses import replace

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import analytic_slab_index

# Silicon slab in silica at 1.55 um: V = k0 d/2 sqrt(n1^2 - n2^2) = 2.56, two guided modes per polarisation
_WAVELENGTH = 1.55e-6                             # m, vacuum
_CORE_INDEX, _CLADDING_INDEX = 3.476, 1.444
_CORE_THICKNESS = 0.4e-6                          # m
_FINE_CELL = 2.5e-9                               # m: 160 core cells
_COARSE_CELL = 5e-9                               # m: lossy lines of 480 cells (dense complex eig path)
_CLADDING_THICKNESS = 2.0e-6                      # m per side: 5.7 decay lengths of the weakest mode (TM1, 0.35 um)
_GRADED_GROWTH, _GRADED_MAX_CELL = 1.05, 10e-9    # graded cladding: x1.05 per cell from the core cell up to 10 nm
_LOSSY_CORE_INDEX = 3.476 - 0.005j
_LOSSY_CLADDING_THICKNESS = 1.0e-6                # m (fundamental-mode decay lengths <= 0.1 um)
_PML_CLADDING_THICKNESS = 0.8e-6                  # m, then _PML_CELLS stretched cells per side (832 cells)
_PML_CELLS, _PML_STRETCH_MAX = 16, 4.0            # s = 1 - 4j depth^3
_GROUP_STEP = 1e-3                                # relative frequency step of the group-index central difference
_INDEX_TOLERANCE = 3e-4                           # |n_eff - analytic|, see the case description
_GROUP_TOLERANCE = 2e-3                           # |n_g - analytic|
_PARITY_TOLERANCE = 1e-8                          # mirror symmetry of modes 1 (even) and 2 (odd)

# Uniform plates with exact discrete eigenvalues
_PLATE_CELLS, _PLATE_CELL = 50, 20e-9             # 1 um wide
_PLATE_PERMITTIVITY, _PLATE_MU = 2.25, 2.0
_PLATE_MODES = 5                                  # ez dirichlet plate: modes 1-2 propagate, 3-5 are evanescent
_FDTD_DT = 1e-17                                  # s (omega dt = 0.012)
_NYQUIST_SPACING = 0.4e-6                         # m: k_t D/2 = 1.72 > 1 for the TEM mode
_RANDOM_SEED = 20261006
_RANDOM_BASE_CELL = 5e-9                          # m; random widths in [1, 2) x this
_RANDOM_STRETCH_CELLS = 8                         # complex stretch on this many outer cells / faces per end
_RANDOM_MODES = 6
_DISCRETE_TOLERANCE = 1e-10

# Projections
_PROJECTION_DT, _PROJECTION_SPACING = 1e-17, 5e-9     # s; normal dual spacing D [m] (offset h = D/2)
_AMPLITUDES = (0.8 - 0.3j, -0.25 + 0.6j)               # (a_positive, a_negative) of mode 1
_CONTAMINATION = (0.5 + 0.5j, 0.3 + 0.0j)              # (a_positive, a_negative) of mode 2
_ROTATION = 1.1                                        # rad
_DISPERSIVE_CELLS = 20
_DISPERSIVE_EPS_INF, _DISPERSIVE_PLASMA_RATIO = 2.25, 0.5   # eps(omega) = 2.25 - (0.5 omega0 / omega)^2
_PROJECTION_TOLERANCE = 1e-9

# Plasmonic hz line (em_slab_spp_discrete): metal eps -2 - 0.2j below air on uniform 5 nm cells, _PML_CELLS stretched
# cells (s = 1 - 4j depth^3) at both ends (metal and air side). n_spp^2 = eps_m / (eps_m + 1) = 1.96 - 0.19j: field
# decay lengths ~89 nm (air) and ~44 nm (metal). (label, metal thickness, air thickness, n_eff tolerance relative to
# the exact discrete surface mode): 'deep' ends ~11 decay lengths from the interface (332 cells: dense eigvals + inverse
# iteration over the whole spectrum incl. the PML modes, end perturbation ~|u_end|^2 ~ 1e-10); 'tight' ends 2.3 decay
# lengths away (92 cells: ~1 % of the mode's weight in the PMLs shifts n_eff by about that).
_SPP_WAVELENGTH = 0.55e-6                         # m
_SPP_METAL_PERMITTIVITY = -2.0 - 0.2j
_SPP_CELL = 5e-9                                  # m
_SPP_LAYOUTS = (('deep', 0.5e-6, 1.0e-6, 1e-7), ('tight', 0.1e-6, 0.2e-6, 0.05))
_SPP_POWER_TOLERANCE = 1e-9                       # |power - 1|: normalised to +1 W/m (forward, Re k_t > 0)

# Degenerate lossy periodic lines (em_slab_degenerate_orthogonal): (label, polarization, cells, cell [m], mu_r); the
# 700-cell line takes the scipy shift-invert path, the 50-cell plates dense numpy eig (periodic corner).
_DEGENERATE_PERMITTIVITY = (1.5 - 0.01j) ** 2
_DEGENERATE_LINES = (('ez_plate', 'ez', _PLATE_CELLS, _PLATE_CELL, _PLATE_MU),
                     ('hz_plate', 'hz', _PLATE_CELLS, _PLATE_CELL, _PLATE_MU),
                     ('ez_long', 'ez', 700, 5e-9, 1.0))
_DEGENERATE_MODES = 5                             # k = 0, then the +-1 and +-2 pairs
_ORTHOGONALITY_TOLERANCE = 1e-8                   # |mode_overlap| between two returned modes
_DEGENERATE_EIGENVALUE_TOLERANCE = 1e-10          # |k_t^2 - exact| / (k0^2 |eps mu_r|)

# (label, polarization, lo end, hi end, permittivity, angle theta_p(p, m) of the exact discrete eigenvector, p >= 1)
_PLATE_CASES = (
    ('ez_dirichlet', 'ez', 'dirichlet', 'dirichlet', _PLATE_PERMITTIVITY, lambda p, m: p * math.pi / (m + 1)),
    ('hz_neumann', 'hz', 'neumann', 'neumann', _PLATE_PERMITTIVITY, lambda p, m: (p - 1) * math.pi / m),
    ('ez_odd', 'ez', 'odd', 'odd', _PLATE_PERMITTIVITY, lambda p, m: p * math.pi / m),
    ('hz_dirichlet_neumann', 'hz', 'dirichlet', 'neumann', _PLATE_PERMITTIVITY,
     lambda p, m: (2 * p - 1) * math.pi / (2 * m + 1)),
    ('ez_odd_neumann', 'ez', 'odd', 'neumann', _PLATE_PERMITTIVITY, lambda p, m: (2 * p - 1) * math.pi / (2 * m)),
    ('ez_periodic', 'ez', 'periodic', 'periodic', _PLATE_PERMITTIVITY, lambda p, m: 2.0 * math.pi * (p // 2) / m),
    ('hz_periodic', 'hz', 'periodic', 'periodic', _PLATE_PERMITTIVITY, lambda p, m: 2.0 * math.pi * (p // 2) / m),
    ('ez_lossy_dirichlet', 'ez', 'dirichlet', 'dirichlet', (1.5 - 0.01j) ** 2, lambda p, m: p * math.pi / (m + 1)),
)
# (polarization, lo end, hi end, cells, PEC cells): 700-cell lines take the scipy shift-invert path
_RANDOM_LINES = (
    ('ez', 'dirichlet', 'neumann', 40, (7, 23)),
    ('hz', 'odd', 'dirichlet', 40, (0, 15)),
    ('ez', 'neumann', 'odd', 40, (39,)),
    ('hz', 'periodic', 'periodic', 40, (12,)),
    ('ez', 'periodic', 'periodic', 40, (0,)),
    ('hz', 'neumann', 'neumann', 40, ()),
    ('ez', 'periodic', 'periodic', 700, (100, 400)),
    ('hz', 'dirichlet', 'odd', 700, (350,)),
)


def _line(slab_modes, polarization, widths, permittivity, ends, mu_r=None, cell_stretch=None, face_stretch=None,
          active=None):
    """SlabLine of isotropic cells: faces take ez 1/mu_r of their lower cell, hz 1/(width-weighted mean permittivity of
    their two cells, as yee_grid.face_average) (0 next to a PEC cell); end faces use the end cell, periodic wrap faces
    cells m-1 and 0."""
    widths = np.asarray(widths, dtype=np.float64)
    n_cells = widths.size
    permittivity = np.ones(n_cells, dtype=np.complex128) * permittivity
    mu_r = np.ones(n_cells, dtype=np.complex128) if mu_r is None else np.asarray(mu_r, dtype=np.complex128)
    cell_stretch = np.ones(n_cells, dtype=np.complex128) if cell_stretch is None else cell_stretch
    face_stretch = np.ones(n_cells + 1, dtype=np.complex128) if face_stretch is None else face_stretch.copy()
    active = np.ones(n_cells, dtype=bool) if active is None else active
    periodic = ends[0] == 'periodic'
    lower = np.concatenate(([n_cells - 1 if periodic else 0], np.arange(n_cells)))   # lower cell of face k
    upper = np.concatenate((np.arange(n_cells), [0 if periodic else n_cells - 1]))  # upper cell of face k
    if periodic:
        face_stretch[-1] = face_stretch[0]
    if polarization == 'ez':
        cell_weight, face_weight = mu_r, 1.0 / mu_r[lower]
    else:
        cell_weight = permittivity
        face_eps = ((permittivity[lower] * widths[lower] + permittivity[upper] * widths[upper])
                    / (widths[lower] + widths[upper]))
        face_weight = np.where(active[lower] & active[upper], 1.0 / face_eps, 0.0)
    return slab_modes.SlabLine(
        polarization=polarization, cell_widths=widths, cell_stretch=cell_stretch, cell_weight=cell_weight,
        index_term=permittivity * mu_r, active=active, face_inverse_spacing=2.0 / (widths[lower] + widths[upper]),
        face_stretch=face_stretch, face_weight=face_weight, lo_end=ends[0], hi_end=ends[1])


def _pml_stretch(n_cells, pml_cells):
    """(cell stretch (m,), face stretch (m + 1,)) of s = 1 - j _PML_STRETCH_MAX depth^3 over pml_cells at both ends."""
    cell_depth = np.zeros(n_cells)
    face_depth = np.zeros(n_cells + 1)
    if pml_cells:
        cell_ramp = (np.arange(pml_cells) + 0.5) / pml_cells     # increasing outward from the PML interface
        face_ramp = np.arange(pml_cells + 1) / pml_cells
        cell_depth[:pml_cells] = cell_ramp[::-1]
        cell_depth[n_cells - pml_cells:] = cell_ramp
        face_depth[:pml_cells + 1] = face_ramp[::-1]
        face_depth[n_cells - pml_cells:] = face_ramp
    return 1.0 - 1j * _PML_STRETCH_MAX * cell_depth ** 3, 1.0 - 1j * _PML_STRETCH_MAX * face_depth ** 3


def _slab_line(slab_modes, polarization, ends, core_cell, cladding_thickness, growth=1.0,
               core_permittivity=_CORE_INDEX ** 2, pml_cells=0):
    """Symmetric slab line: core cells of core_cell, cladding cells growing x growth (capped at _GRADED_MAX_CELL) over
    cladding_thickness per side, then pml_cells stretched cells of the outermost cladding width."""
    cladding = []
    width, covered = core_cell, 0.0
    while covered < cladding_thickness - 1e-3 * core_cell:
        width = min(width * growth, max(_GRADED_MAX_CELL, core_cell))
        cladding.append(width)
        covered += width
    cladding = np.array(cladding)
    n_core = int(round(_CORE_THICKNESS / core_cell))
    pml = np.full(pml_cells, cladding[-1])
    widths = np.concatenate((pml, cladding[::-1], np.full(n_core, core_cell), cladding, pml))
    permittivity = np.full(widths.size, _CLADDING_INDEX ** 2, dtype=np.complex128)
    core_start = pml_cells + cladding.size
    permittivity[core_start:core_start + n_core] = core_permittivity
    cell_stretch, face_stretch = _pml_stretch(widths.size, pml_cells)
    return _line(slab_modes, polarization, widths, permittivity, ends, cell_stretch=cell_stretch,
                 face_stretch=face_stretch)


def _plate_line(slab_modes, polarization, ends, permittivity):
    """Uniform plate of _PLATE_CELLS cells (mu_r = _PLATE_MU)."""
    return _line(slab_modes, polarization, np.full(_PLATE_CELLS, _PLATE_CELL), permittivity, ends,
                 mu_r=np.full(_PLATE_CELLS, _PLATE_MU))


def _slab_mismatch(n_eff, k0, polarization, odd, core_permittivity):
    """Symmetric-slab dispersion residual at a complex n_eff: even x tan x - r y, odd -x cot x - r y with x = kappa d/2,
    y = gamma d/2, r = 1 (ez, TE) or eps_core / eps_cladding (hz, TM)."""
    cladding_permittivity = _CLADDING_INDEX ** 2
    half_thickness_k0 = 0.5 * _CORE_THICKNESS * k0
    x = half_thickness_k0 * np.sqrt(np.complex128(core_permittivity - n_eff ** 2))
    y = half_thickness_k0 * np.sqrt(np.complex128(n_eff ** 2 - cladding_permittivity))
    ratio = 1.0 if polarization == 'ez' else core_permittivity / cladding_permittivity
    return (-x / np.tan(x) if odd else x * np.tan(x)) - ratio * y


def _analytic_slab_index(k0, polarization, odd, core_permittivity=_CORE_INDEX ** 2):
    """Exact n_eff of the symmetric slab: bisection with the lossless core (selftest_helpers), then secant steps on the
    complex relation when the core is lossy."""
    n_eff = complex(analytic_slab_index(k0, _CORE_THICKNESS, float(np.real(core_permittivity)), _CLADDING_INDEX ** 2,
                                        polarization, odd))
    if np.imag(core_permittivity) == 0.0:
        return n_eff
    previous_n, current_n = n_eff, n_eff * (1.0 - 1e-4j)
    previous_residual = _slab_mismatch(previous_n, k0, polarization, odd, core_permittivity)
    for _ in range(60):
        residual = _slab_mismatch(current_n, k0, polarization, odd, core_permittivity)
        if residual == previous_residual:
            break
        next_n = complex(current_n - residual * (current_n - previous_n) / (residual - previous_residual))
        previous_n, previous_residual, current_n = current_n, residual, next_n
        if abs(current_n - previous_n) <= 1e-15 * abs(current_n):
            break
    return current_n


def _analytic_group_index(k0, polarization, odd):
    """C0 d beta / d omega of the analytic slab by the central difference group_index uses (relative step
    _GROUP_STEP)."""
    k_plus, k_minus = k0 * (1.0 + _GROUP_STEP), k0 * (1.0 - _GROUP_STEP)
    return ((_analytic_slab_index(k_plus, polarization, odd) * k_plus
             - _analytic_slab_index(k_minus, polarization, odd) * k_minus) / (k_plus - k_minus))


def _check_analytic(results_by_name):
    """Slab n_eff (uniform, graded, lossy, PML-extended) and group index against the analytic relations plus parity and
    guided-mode count; error = max index error (group error scaled by 0.15) + failed checks."""
    from . import slab_modes
    k0 = 2.0 * math.pi / _WAVELENGTH
    omega = C0 * k0
    end_pairs = {'ez': ('dirichlet', 'dirichlet'), 'hz': ('neumann', 'neumann')}
    index_errors, parity_errors, group_errors = {}, {}, {}
    measured_n_eff, analytic_n_eff = {}, {}
    violations = []

    def compare(key, mode, expected):
        index_errors[key] = abs(mode.n_eff - expected)
        measured_n_eff[key] = [mode.n_eff.real, mode.n_eff.imag]
        analytic_n_eff[key] = [expected.real, expected.imag]

    group_line, group_modes = None, None
    for polarization in ('ez', 'hz'):
        guided = [_analytic_slab_index(k0, polarization, odd) for odd in (False, True)]
        for layout, growth in (('uniform', 1.0), ('graded', _GRADED_GROWTH)):
            line = _slab_line(slab_modes, polarization, end_pairs[polarization], _FINE_CELL, _CLADDING_THICKNESS,
                              growth=growth)
            modes = slab_modes.solve_slab_modes(line, omega, 3)
            for number, parity_sign in ((1, 1.0), (2, -1.0)):
                key = f'{polarization}_{layout}_mode{number}'
                profile = modes[number - 1].profile
                compare(key, modes[number - 1], guided[number - 1])
                parity_errors[key] = (float(np.max(np.abs(profile - parity_sign * profile[::-1])))
                                      / float(np.max(np.abs(profile))))
                if parity_errors[key] > _PARITY_TOLERANCE:
                    violations.append(f"{key} is not {'even' if number == 1 else 'odd'}")
            if not modes[2].n_eff.real < _CLADDING_INDEX:
                violations.append(f'{polarization}_{layout}: mode 3 has Re(n_eff) {modes[2].n_eff.real:.6g} >= n2 '
                                  f'(the slab has 2 guided modes)')
            if (polarization, layout) == ('ez', 'uniform'):
                group_line, group_modes = line, modes
        lossy_line = _slab_line(slab_modes, polarization, end_pairs[polarization], _COARSE_CELL,
                                _LOSSY_CLADDING_THICKNESS, core_permittivity=_LOSSY_CORE_INDEX ** 2)
        compare(f'{polarization}_lossy_mode1', slab_modes.solve_slab_modes(lossy_line, omega, 2)[0],
                _analytic_slab_index(k0, polarization, False, _LOSSY_CORE_INDEX ** 2))
    pml_line = _slab_line(slab_modes, 'ez', end_pairs['ez'], _FINE_CELL, _PML_CLADDING_THICKNESS, pml_cells=_PML_CELLS)
    pml_modes = slab_modes.solve_slab_modes(pml_line, omega, 3)
    for number in (1, 2):
        compare(f'ez_pml_mode{number}', pml_modes[number - 1], _analytic_slab_index(k0, 'ez', number == 2))
    for polarization in ('ez', 'hz'):   # 432 cells: dense complex eig over the whole spectrum incl. the PML modes
        dense_pml_line = _slab_line(slab_modes, polarization, end_pairs[polarization], _COARSE_CELL,
                                    _PML_CLADDING_THICKNESS, pml_cells=_PML_CELLS)
        compare(f'{polarization}_pml_dense_mode1', slab_modes.solve_slab_modes(dense_pml_line, omega, 2)[0],
                _analytic_slab_index(k0, polarization, False))
    for number in (1, 2):
        measured_group = slab_modes.group_index(lambda omega_value: group_line, omega, group_modes[number - 1], 3,
                                                relative_step=_GROUP_STEP)
        group_errors[f'ez_mode{number}'] = abs(measured_group - _analytic_group_index(k0, 'ez', number == 2))

    measured = max(index_errors.values())
    error = max(measured, max(group_errors.values()) * (_INDEX_TOLERANCE / _GROUP_TOLERANCE)) + len(violations)
    return {'measured': measured, 'expected': 0.0, 'error': error, 'tolerance': _INDEX_TOLERANCE,
            'details': {'index_errors': index_errors, 'group_index_errors': group_errors,
                        'parity_errors': parity_errors, 'n_eff_measured': measured_n_eff,
                        'n_eff_analytic': analytic_n_eff, 'violations': violations}}


def _random_line(slab_modes, rng, polarization, n_cells, ends, pec_cells):
    """Random graded lossy line: widths 5-10 nm, eps 2-12 with loss, mu_r 1-2, complex stretch on the outer
    _RANDOM_STRETCH_CELLS cells and faces, PEC cells at pec_cells."""
    widths = _RANDOM_BASE_CELL * (1.0 + rng.random(n_cells))
    permittivity = 2.0 + 10.0 * rng.random(n_cells) - 0.5j * rng.random(n_cells)
    mu_r = 1.0 + rng.random(n_cells)
    cell_stretch = np.ones(n_cells, dtype=np.complex128)
    face_stretch = np.ones(n_cells + 1, dtype=np.complex128)
    depth = _RANDOM_STRETCH_CELLS
    for stretch in (cell_stretch, face_stretch):
        for outer in (slice(0, depth), slice(-depth, None)):
            stretch[outer] = 1.0 + rng.random(depth) - 2.0j * rng.random(depth)
    active = np.ones(n_cells, dtype=bool)
    active[list(pec_cells)] = False
    return _line(slab_modes, polarization, widths, permittivity, ends, mu_r=mu_r, cell_stretch=cell_stretch,
                 face_stretch=face_stretch, active=active)


def _ghost_column(index, n_cells, end):
    """(column, factor) of u_index in the reference operator: inside -> (index, 1); beyond an end -> dirichlet
    (None, 0), odd (mirror cell, -1), periodic (wrapped cell, 1)."""
    if 0 <= index < n_cells:
        return index, 1.0
    if end == 'odd':
        return (0 if index < 0 else n_cells - 1), -1.0
    if end == 'periodic':
        return index % n_cells, 1.0
    return None, 0.0


def _reference_operator(line, omega_tilde):
    """Dense (m, m) operator of the slab_modes contract from explicit ghost-cell rules (independent of its assembly):
    row k = cell_weight / (w s) (F_{k+1} - F_k) + (omega_t / C0)^2 q u_k, F_f = b_f (u_f - u_{f-1})."""
    n_cells = line.cell_widths.size
    operator = np.zeros((n_cells, n_cells), dtype=np.complex128)
    for cell in range(n_cells):
        row_scale = line.cell_weight[cell] / (line.cell_widths[cell] * line.cell_stretch[cell])
        operator[cell, cell] += (omega_tilde / C0) ** 2 * line.index_term[cell]
        for face, face_sign in ((cell + 1, 1.0), (cell, -1.0)):
            end = line.lo_end if face == 0 else line.hi_end if face == n_cells else None
            if end == 'neumann':
                continue
            coefficient_face = 0 if end == 'periodic' else face
            conductance = (line.face_weight[coefficient_face] * line.face_inverse_spacing[coefficient_face]
                           / line.face_stretch[coefficient_face])
            for neighbour, neighbour_sign in ((face, 1.0), (face - 1, -1.0)):
                column, factor = _ghost_column(neighbour, n_cells, end)
                if column is not None:
                    operator[cell, column] += face_sign * neighbour_sign * factor * row_scale * conductance
    return operator


def _check_discrete(results_by_name):
    """(A) plate eigenvalues, plane-wave amplitude, edge amplitude; (B) FDTD dispersion and the arcsin branch; (C)
    random-line residuals; error = max relative error + failed checks."""
    from . import slab_modes
    k0 = 2.0 * math.pi / _WAVELENGTH
    omega = C0 * k0
    errors, violations, all_modes = {}, [], []

    for label, polarization, lo_end, hi_end, permittivity, angle in _PLATE_CASES:
        modes = slab_modes.solve_slab_modes(_plate_line(slab_modes, polarization, (lo_end, hi_end), permittivity),
                                            omega, _PLATE_MODES)
        all_modes.extend(modes)
        if len(modes) != _PLATE_MODES:
            violations.append(f'{label}: {len(modes)} modes returned, expected {_PLATE_MODES}')
            continue
        index_term = permittivity * _PLATE_MU
        exact = [k0 ** 2 * index_term - (2.0 / _PLATE_CELL * math.sin(0.5 * angle(p, _PLATE_CELLS))) ** 2
                 for p in range(1, _PLATE_MODES + 1)]
        errors[f'eigenvalue_{label}'] = (max(abs(mode.k_tilde ** 2 - value) for mode, value in zip(modes, exact))
                                         / (k0 ** 2 * abs(index_term)))
        if lo_end == 'periodic':
            cell_weight = _PLATE_MU if polarization == 'ez' else permittivity
            field_constant = MU0 if polarization == 'ez' else EPS0
            transverse_factor = k0 * math.sqrt(index_term) / (omega * field_constant * cell_weight)
            amplitude = math.sqrt(2.0 / (transverse_factor * _PLATE_CELLS * _PLATE_CELL))
            errors[f'plane_wave_{label}'] = float(np.max(np.abs(modes[0].profile - amplitude))) / amplitude
        if label == 'ez_dirichlet':
            sine = np.sin(math.pi * np.arange(1, _PLATE_CELLS + 1) / (_PLATE_CELLS + 1))
            errors['edge_amplitude'] = abs(modes[0].edge_amplitude - float(sine[0] / np.max(sine)))

    tem_line = _plate_line(slab_modes, 'hz', ('neumann', 'neumann'), _PLATE_PERMITTIVITY)
    omega_tilde = 2.0 / _FDTD_DT * math.sin(0.5 * omega * _FDTD_DT)
    k_tilde = omega_tilde * math.sqrt(_PLATE_PERMITTIVITY * _PLATE_MU) / C0
    for spacing in (_PLATE_CELL, _NYQUIST_SPACING):
        mode = slab_modes.solve_slab_modes(tem_line, omega, 1, dt=_FDTD_DT, normal_spacing=spacing)[0]
        all_modes.append(mode)
        half_spacing = 0.5 * spacing
        argument = k_tilde * half_spacing
        phase = complex(math.asin(argument)) if argument <= 1.0 else complex(0.5 * math.pi, -math.acosh(argument))
        expected_n_eff = phase / half_spacing * C0 / omega
        errors[f'fdtd_n_eff_spacing_{spacing * 1e9:.0f}nm'] = abs(mode.n_eff - expected_n_eff) / abs(expected_n_eff)
        errors[f'fdtd_k_tilde_spacing_{spacing * 1e9:.0f}nm'] = abs(mode.k_tilde - k_tilde) / k_tilde
    errors['fdtd_omega_tilde'] = abs(all_modes[-1].omega_tilde - omega_tilde) / omega_tilde

    rng = np.random.default_rng(_RANDOM_SEED)
    residual = 0.0
    for polarization, lo_end, hi_end, n_cells, pec_cells in _RANDOM_LINES:
        line = _random_line(slab_modes, rng, polarization, n_cells, (lo_end, hi_end), pec_cells)
        modes = slab_modes.solve_slab_modes(line, omega, _RANDOM_MODES)
        all_modes.extend(modes)
        operator = _reference_operator(line, omega)
        active = line.active
        operator_norm = float(np.max(np.sum(np.abs(operator[active]), axis=1)))
        for mode in modes:
            mismatch = operator @ mode.profile - mode.k_tilde ** 2 * mode.profile
            residual = max(residual, float(np.max(np.abs(mismatch[active])))
                           / (operator_norm * float(np.max(np.abs(mode.profile)))))
            if np.any(mode.profile[~active] != 0.0):
                violations.append(f'{polarization} {lo_end}/{hi_end} line of {n_cells} cells: u != 0 on a PEC cell')
    errors['random_line_residual'] = residual

    def off_branch(k_tilde):
        """True unless Re k_t > 0 for Re k_t^2 > 0 and Im k_t <= 0 otherwise (the slab_modes branch contract)."""
        return not (k_tilde.real > 0.0 if (k_tilde ** 2).real > 0.0 else k_tilde.imag <= 0.0)

    branch_failures = sum(1 for mode in all_modes
                          if off_branch(mode.k_tilde) or mode.k_tilde.imag > 0.0 or mode.k.imag > 0.0)
    if branch_failures:
        violations.append(f'{branch_failures} modes off the k_t branch (Re k_t > 0 for Re k_t^2 > 0, else Im k_t < 0) '
                          f'or with Im(k_t) > 0 or Im(k) > 0 on a passive line')

    measured = max(errors.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations),
            'tolerance': _DISCRETE_TOLERANCE,
            'details': {'errors': errors, 'violations': violations}}


def _dispersive_permittivity(omega_value, reference_omega):
    """eps(omega) = _DISPERSIVE_EPS_INF - (_DISPERSIVE_PLASMA_RATIO reference_omega / omega)^2 (lossless Drude-like)."""
    return _DISPERSIVE_EPS_INF - (_DISPERSIVE_PLASMA_RATIO * reference_omega / omega_value) ** 2


def _discrete_plane_wavenumber(omega_value, reference_omega):
    """k = (2/D) arcsin((D/2) sqrt(eps(omega)) omega_t / C0) of the dispersive plane wave (D = _PLATE_CELL)."""
    omega_tilde = 2.0 / _PROJECTION_DT * math.sin(0.5 * omega_value * _PROJECTION_DT)
    half_spacing = 0.5 * _PLATE_CELL
    return math.asin(half_spacing * math.sqrt(_dispersive_permittivity(omega_value, reference_omega)) * omega_tilde
                     / C0) / half_spacing


def _check_projections(results_by_name):
    """Normalisation, phase, orthogonality, mode_amplitudes round trip, align / track, bilinear fallback and the
    dispersive group index; error = max error + failed checks."""
    from . import slab_modes
    k0 = 2.0 * math.pi / _WAVELENGTH
    omega = C0 * k0
    half_spacing = 0.5 * _PROJECTION_SPACING
    errors, violations = {}, []

    for polarization, ends in (('ez', ('dirichlet', 'dirichlet')), ('hz', ('neumann', 'neumann'))):
        line = _slab_line(slab_modes, polarization, ends, _COARSE_CELL, _LOSSY_CLADDING_THICKNESS,
                          core_permittivity=_LOSSY_CORE_INDEX ** 2)
        modes = slab_modes.solve_slab_modes(line, omega, 4, dt=_PROJECTION_DT, normal_spacing=_PROJECTION_SPACING)
        # Discrete Yee flux: t sits D/2 downstream of u, and the FDTD averages E over two steps to the H time.
        errors[f'power_{polarization}'] = max(
            abs(0.5 * float(np.real(np.sum(np.conj(mode.transverse_factor) * np.abs(mode.profile) ** 2
                                           * line.cell_widths * np.exp(1j * np.conj(mode.k) * half_spacing))))
                * math.cos(0.5 * omega * _PROJECTION_DT) - 1.0) for mode in modes[:2])
        for number, mode in enumerate(modes, start=1):
            peak = mode.profile[slab_modes.peak_cell(mode.profile)]
            if not (peak.real > 0.0 and abs(peak.imag) <= 1e-12 * abs(peak)):
                violations.append(f'{polarization} mode {number}: the lo-most largest |u| is not real > 0')

        rho = line.cell_widths * line.cell_stretch / line.cell_weight
        norms = [math.sqrt(float(np.sum(np.abs(mode.profile) ** 2 * np.abs(rho)))) for mode in modes]
        orthogonality = module_overlap = 0.0
        for first in range(len(modes)):
            for second in range(first + 1, len(modes)):
                orthogonality = max(orthogonality, abs(complex(np.sum(modes[first].profile * modes[second].profile
                                                                      * rho))) / (norms[first] * norms[second]))
                module_overlap = max(module_overlap, abs(slab_modes.mode_overlap(line, modes[first], modes[second])))
        errors[f'orthogonality_{polarization}'] = orthogonality
        errors[f'mode_overlap_{polarization}'] = module_overlap

        truth = ((modes[0], _AMPLITUDES), (modes[1], _CONTAMINATION))
        out_of_plane = np.zeros(line.cell_widths.size, dtype=np.complex128)
        transverse_over_sign = np.zeros(line.cell_widths.size, dtype=np.complex128)
        for mode, (positive, negative) in truth:
            out_of_plane += (positive + negative) * mode.profile
            transverse_over_sign += mode.transverse_factor * mode.profile * (
                positive * np.exp(-1j * mode.k * half_spacing) - negative * np.exp(1j * mode.k * half_spacing))
        round_trip = 0.0
        for mode, amplitudes in truth:
            recovered = slab_modes.mode_amplitudes(line, mode, out_of_plane, transverse_over_sign, half_spacing)
            round_trip = max(round_trip, max(abs(value - amplitude) for value, amplitude in zip(recovered, amplitudes))
                             / max(abs(amplitude) for amplitude in amplitudes))
        errors[f'round_trip_{polarization}'] = round_trip

        rotated = replace(modes[1], profile=modes[1].profile * np.exp(1j * _ROTATION))
        aligned = slab_modes.align_phase(line, rotated, modes[1])
        aligned_overlap = slab_modes.mode_overlap(line, aligned, modes[1])
        if not (aligned_overlap.real > 0.0 and abs(aligned_overlap.imag) <= 1e-12 * abs(aligned_overlap)):
            violations.append(f'{polarization}: align_phase left the overlap {aligned_overlap!r} not real > 0')
        errors[f'align_magnitude_{polarization}'] = (float(np.max(np.abs(np.abs(aligned.profile)
                                                                         - np.abs(modes[1].profile))))
                                                     / float(np.max(np.abs(modes[1].profile))))
        shuffled = [modes[2], modes[0], modes[3], modes[1]]
        for target, position in ((0, 1), (1, 3), (2, 0)):
            if slab_modes.track_mode(line, modes[target], shuffled) != position:
                violations.append(f'{polarization}: track_mode did not find mode {target + 1} at position {position}')

    plate = _plate_line(slab_modes, 'ez', ('dirichlet', 'dirichlet'), _PLATE_PERMITTIVITY)
    plate_modes = slab_modes.solve_slab_modes(plate, omega, _PLATE_MODES)
    evanescent = plate_modes[2:]
    if any(mode.k_tilde.real != 0.0 for mode in evanescent):
        violations.append('plate modes 3-5 are not evanescent (Re k_t != 0)')
    errors['evanescent_normalisation'] = max(
        abs(abs(0.5 * complex(np.sum(mode.transverse_factor * mode.profile ** 2 * plate.cell_widths
                                     * plate.cell_stretch))) - 1.0) for mode in evanescent)
    errors['evanescent_power'] = max(abs(mode.power) for mode in evanescent)
    errors['self_overlap'] = abs(slab_modes.mode_overlap(plate, plate_modes[0], plate_modes[0]) - 1.0)
    rotated = replace(plate_modes[0], profile=plate_modes[0].profile * np.exp(1j * _ROTATION))
    restored = slab_modes.align_phase(plate, rotated, plate_modes[0])
    errors['align_restore'] = (float(np.max(np.abs(restored.profile - plate_modes[0].profile)))
                               / float(np.max(np.abs(plate_modes[0].profile))))

    def dispersive_line(omega_value):
        return _line(slab_modes, 'hz', np.full(_DISPERSIVE_CELLS, _PLATE_CELL),
                     _dispersive_permittivity(omega_value, omega), ('periodic', 'periodic'))

    reference = slab_modes.solve_slab_modes(dispersive_line(omega), omega, 3, dt=_PROJECTION_DT,
                                            normal_spacing=_PLATE_CELL)[0]
    measured_group = slab_modes.group_index(dispersive_line, omega, reference, 3, dt=_PROJECTION_DT,
                                            normal_spacing=_PLATE_CELL, relative_step=_GROUP_STEP)
    wavenumber_plus = _discrete_plane_wavenumber(omega * (1.0 + _GROUP_STEP), omega)
    wavenumber_minus = _discrete_plane_wavenumber(omega * (1.0 - _GROUP_STEP), omega)
    expected_group = C0 * (wavenumber_plus - wavenumber_minus) / (2.0 * _GROUP_STEP * omega)
    errors['group_index_dispersive'] = abs(measured_group - expected_group) / expected_group

    measured = max(errors.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations),
            'tolerance': _PROJECTION_TOLERANCE,
            'details': {'errors': errors, 'violations': violations, 'group_index_measured': measured_group.real,
                        'group_index_expected': expected_group}}


def _decay_ratio(eigenvalue, k0, cell, permittivity):
    """Root |r| < 1 of r + 1/r = 2 + cell^2 (eigenvalue - k0^2 permittivity): u_k ~ r^k decays away from the interface
    in a uniform hz half-line."""
    excess = 0.5 * cell ** 2 * (eigenvalue - k0 ** 2 * permittivity)   # half_trace - 1, kept apart (cancellation)
    half_trace = 1.0 + excess
    root = cmath.sqrt(excess * (2.0 + excess))
    ratio = half_trace - root
    return ratio if abs(ratio) < 1.0 else half_trace + root


def _discrete_surface_eigenvalue(k0, cell, eps_metal, eps_dielectric):
    """k_t^2 of the exact discrete hz surface mode of two uniform half-lines (equal cells, interface face eps their
    mean): root of b eps_d - a eps_m - a b eps_f = 0 with a = 1 - 1/r_d, b = 1/r_m - 1 (secant from the continuum)."""
    face_permittivity = 0.5 * (eps_metal + eps_dielectric)

    def mismatch(eigenvalue):
        a = 1.0 - 1.0 / _decay_ratio(eigenvalue, k0, cell, eps_dielectric)
        b = 1.0 / _decay_ratio(eigenvalue, k0, cell, eps_metal) - 1.0
        return b * eps_dielectric - a * eps_metal - a * b * face_permittivity

    previous = complex(k0 ** 2 * eps_metal * eps_dielectric / (eps_metal + eps_dielectric))
    current = previous * (1.0 + 1e-3)
    previous_residual = mismatch(previous)
    for _ in range(60):
        residual = mismatch(current)
        if residual == previous_residual:
            break
        following = current - residual * (current - previous) / (residual - previous_residual)
        previous, previous_residual, current = current, residual, following
        if abs(current - previous) <= 1e-15 * abs(current):
            break
    return current


def _check_spp(results_by_name):
    """Mode 1 of plasmonic hz lines (PML on the metal and the air side) vs the exact discrete surface mode, its +1 W/m
    normalisation; error in tolerance units + failed checks."""
    from . import slab_modes
    k0 = 2.0 * math.pi / _SPP_WAVELENGTH
    omega = C0 * k0
    expected_eigenvalue = _discrete_surface_eigenvalue(k0, _SPP_CELL, _SPP_METAL_PERMITTIVITY, 1.0)
    expected_n_eff = cmath.sqrt(expected_eigenvalue) / k0
    continuum_n_eff = cmath.sqrt(_SPP_METAL_PERMITTIVITY / (_SPP_METAL_PERMITTIVITY + 1.0))
    normalized_errors, details, violations = {}, {}, []
    for label, metal_thickness, air_thickness, tolerance in _SPP_LAYOUTS:
        n_metal = int(round(metal_thickness / _SPP_CELL)) + _PML_CELLS
        n_air = int(round(air_thickness / _SPP_CELL)) + _PML_CELLS
        permittivity = np.concatenate((np.full(n_metal, _SPP_METAL_PERMITTIVITY), np.ones(n_air))).astype(np.complex128)
        cell_stretch, face_stretch = _pml_stretch(permittivity.size, _PML_CELLS)
        line = _line(slab_modes, 'hz', np.full(permittivity.size, _SPP_CELL), permittivity, ('neumann', 'neumann'),
                     cell_stretch=cell_stretch, face_stretch=face_stretch)
        modes = slab_modes.solve_slab_modes(line, omega, 3)
        if not modes:
            violations.append(f'{label}: no mode returned')
            continue
        mode = modes[0]
        n_eff = complex(mode.n_eff)
        normalized_errors[f'{label}_n_eff'] = abs(n_eff / expected_n_eff - 1.0) / tolerance
        normalized_errors[f'{label}_power'] = abs(mode.power - 1.0) / _SPP_POWER_TOLERANCE
        details[f'{label}_n_eff'] = [n_eff.real, n_eff.imag]
        details[f'{label}_cells'] = int(permittivity.size)
        details[f'{label}_n_eff_all_modes'] = [[complex(other.n_eff).real, complex(other.n_eff).imag]
                                              for other in modes]
        if not n_eff.real > 1.0:
            violations.append(f'{label}: mode 1 has Re(n_eff) {n_eff.real:.6g} <= 1 (the surface mode was dropped or '
                              f'ranked below a radiation mode)')
    worst = max(normalized_errors.values()) if normalized_errors else math.inf
    return {'measured': details.get('deep_n_eff', [math.nan])[0], 'expected': expected_n_eff.real,
            'error': worst + len(violations), 'tolerance': 1.0,
            'details': {'expected_n_eff': [expected_n_eff.real, expected_n_eff.imag],
                        'continuum_n_eff': [continuum_n_eff.real, continuum_n_eff.imag],
                        'normalized_errors': normalized_errors, 'violations': violations, **details}}


def _check_degenerate(results_by_name):
    """Lossy periodic lines: eigenvalues vs the exact discrete plane waves, every pair of returned modes bilinear
    orthogonal (the +-k pairs included); error in tolerance units + failed checks."""
    from . import slab_modes
    k0 = 2.0 * math.pi / _WAVELENGTH
    omega = C0 * k0
    normalized_errors, details, violations = {}, {}, []
    for label, polarization, n_cells, cell, mu_r in _DEGENERATE_LINES:
        line = _line(slab_modes, polarization, np.full(n_cells, cell), _DEGENERATE_PERMITTIVITY,
                     ('periodic', 'periodic'), mu_r=np.full(n_cells, mu_r))
        modes = slab_modes.solve_slab_modes(line, omega, _DEGENERATE_MODES)
        if len(modes) != _DEGENERATE_MODES:
            violations.append(f'{label}: {len(modes)} modes returned, expected {_DEGENERATE_MODES}')
            continue
        index_term = _DEGENERATE_PERMITTIVITY * mu_r
        exact = [k0 ** 2 * index_term - (2.0 / cell * math.sin(math.pi * (p // 2) / n_cells)) ** 2
                 for p in range(1, _DEGENERATE_MODES + 1)]
        eigenvalue_error = (max(abs(mode.k_tilde ** 2 - value) for mode, value in zip(modes, exact))
                            / (k0 ** 2 * abs(index_term)))
        overlap = max(abs(slab_modes.mode_overlap(line, modes[first], modes[second]))
                      for first in range(len(modes)) for second in range(first + 1, len(modes)))
        normalized_errors[f'{label}_eigenvalue'] = eigenvalue_error / _DEGENERATE_EIGENVALUE_TOLERANCE
        normalized_errors[f'{label}_orthogonality'] = overlap / _ORTHOGONALITY_TOLERANCE
        details[f'{label}_eigenvalue_error'] = eigenvalue_error
        details[f'{label}_largest_overlap'] = overlap
    worst = max(normalized_errors.values()) if normalized_errors else math.inf
    return {'measured': worst, 'expected': 0.0, 'error': worst + len(violations), 'tolerance': 1.0,
            'details': {'normalized_errors': normalized_errors, 'violations': violations, **details}}


MODE_SELFTEST_CASES = (
    SelfTestCase(
        name='em_slab_modes_analytic',
        label='EM slab modes vs analytic slab',
        description='CPU check of physics/em/slab_modes against the analytic symmetric slab (no model run): silicon '
                    'core n1 = 3.476, d = 0.4 um, in silica n2 = 1.444 at 1.55 um (V = 2.56, two guided modes per '
                    'polarisation). Roots by bisection: ez (TE) even x tan x = y, odd -x cot x = y, hz (TM) with y '
                    'times n1^2/n2^2 (x = kappa d/2, y = gamma d/2); the lossy core n1 = 3.476 - 0.005j by complex '
                    'secant steps. Lines: 2.5 nm cells with 2 um cladding per side (ez dirichlet, hz neumann ends): '
                    'modes 1 and 2 against the even / odd roots, mode 1 even and mode 2 odd to 1e-8 (exact mirror '
                    'symmetry up to eps ||T|| / gap ~ 1e-12), mode 3 below n2; the same with cladding cells growing '
                    'x1.05 up to 10 nm (graded); the lossy core on 5 nm cells with 1 um cladding (dense complex eig); '
                    'ez with 0.8 um cladding plus 16 stretched PML cells per side, s = 1 - 4j depth^3 (832 cells: '
                    'scipy shift-invert path), and mode 1 of both polarisations on the same layout with 5 nm cells '
                    '(432 cells: dense complex eig of the whole spectrum, where the PML modes, Re(n_eff) of order 10, '
                    'must be dropped, where evanescent modes with a complex eigenvalue reach Re(n_eff) ~ 1.9 at '
                    'Im(n_eff) ~ -22 and must rank below the guided modes (ranking by Re k_t^2), and where the guided '
                    "mode's eigenvalue carries an Im of either sign of order 1e-8 from its tail in the PML, which the "
                    'k_t branch (Re k_t > 0 for Re k_t^2 > 0) must not turn into a backward mode). Group index of the '
                    'uniform ez modes 1 and 2 (central difference, '
                    'relative step 1e-3, modes tracked by overlap) against the same central difference of the '
                    'analytic beta. error = max |n_eff - analytic| (+ group-index error x 0.15) + the number of '
                    'failed parity / guided-count checks. Tolerance 3e-4: the exact discrete slab relation, e.g. '
                    'even tan(x (1 + (kappa dx)^2/24)) = r (y/x) (1 - ((kappa dx)^2 + (gamma dx)^2)/8), shifts n_eff '
                    'by 2e-5 (modes 1) to 1.05e-4 (ez mode 2) at 2.5 nm and by ~9e-5 for the fundamental modes at '
                    '5 nm (a half-cell interface misplacement would give ~2.6e-3); the group-index tolerance 2e-3 '
                    'allows 2-4 times the n_eff error carried by n_g = n + omega dn/domega with margin, while a '
                    'tracking failure would give errors of order 100.',
        specs=lambda: {},
        check=_check_analytic,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_slab_modes_discrete',
        label='EM slab modes exact discrete',
        description='Exact discrete checks of physics/em/slab_modes (no model run). (A) Uniform plates of 50 cells of '
                    '20 nm (eps 2.25, mu_r 2): k_t^2 of the first 5 modes equals k0^2 eps mu_r - (4/dx^2) '
                    'sin^2(theta_p/2) with theta_p = p pi/(m+1) (dirichlet, ez), (p-1) pi/m (neumann, hz), p pi/m '
                    '(odd, ez), (2p-1) pi/(2m+1) (dirichlet / neumann, hz), (2p-1) pi/(2m) (odd / neumann, ez), '
                    '2 pi floor(p/2)/m (periodic, ez and hz), and for a lossy dirichlet plate (n = 1.5 - 0.01j, '
                    'dense complex eig); the periodic plane wave is real, constant and |u|^2 = 2/(Re(g) L) (1 W/m); '
                    'the dirichlet mode 1 edge_amplitude is sin(pi/(m+1)) / max sin. (B) FDTD consistency of the hz '
                    'TEM mode with dt = 1e-17 s: normal spacing D = 20 nm gives k = (2/D) arcsin((D/2) n omega_t / '
                    'C0), omega_t = (2/dt) sin(omega dt/2); D = 0.4 um puts k_t D/2 = 1.72 beyond the grid Nyquist '
                    'limit, k = (pi/2 - j acosh(k_t D/2)) / (D/2) (real k_t D/2 > 1 lies on the arcsin cut: the Im k '
                    '< 0 side). (C) Random graded lossy '
                    'lines (widths 5-10 nm, eps 2-12 with loss, mu_r 1-2, complex stretch on the outer 8 cells and '
                    'faces, PEC cells incl. end cells, every end pair, 40 cells dense, 700 cells shift-invert incl. a '
                    'periodic corner): every mode satisfies A u = k_t^2 u for A built from explicit ghost-cell rules '
                    'of the operator contract (relative to ||A|| max |u|) with u = 0 on PEC cells. Every k_t lies on '
                    'the contract branch (Re k_t > 0 when Re k_t^2 > 0, else Im k_t < 0) and, the lines being '
                    'passive, has Im k_t <= 0 and Im k <= 0. error = max relative error + the number of failed '
                    'checks; tolerance '
                    '1e-10: float64 eigen-solves of identical operators (eps ||T|| / k0^2 ~ 1e-13).',
        specs=lambda: {},
        check=_check_discrete,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_slab_modes_projections',
        label='EM slab mode projections',
        description='Normalisation and projections of physics/em/slab_modes (no model run). Lossy silicon slab lines '
                    '(core 3.476 - 0.005j, ez dirichlet and hz neumann ends, 5 nm cells, dt = 1e-17 s, normal spacing '
                    'D = 5 nm): modes 1-2 carry the discrete Yee flux 1/2 Re sum conj(g) |u|^2 w e^{j conj(k) D/2} '
                    'cos(omega dt/2) = 1 W/m, every mode has its largest |u| '
                    'real > 0, the 4 modes are orthogonal in sum a b w s / cell_weight (relative to the |.| norms, '
                    'also through mode_overlap), mode_amplitudes recovers (a+, a-) = (0.8 - 0.3j, -0.25 + 0.6j) of '
                    'mode 1 and (0.5 + 0.5j, 0.3) of mode 2 from U = sum (a+ + a-) u and '
                    "T' = sum g u (a+ exp(-j k h) - a- exp(j k h)) at h = D/2, align_phase makes the overlap real > 0 "
                    'keeping |u|, and track_mode finds the modes in a shuffled list. Lossless ez plate (eps 2.25, '
                    'mu_r 2): the evanescent modes 3-5 take the bilinear normalisation |1/2 sum g u^2 w s| = 1 with '
                    'zero power, a real mode overlaps itself with 1 and align_phase undoes a rotation. Group index of '
                    'a dispersive periodic hz line (eps = 2.25 - (omega0 / (2 omega))^2, dt, D = 20 nm) equals the '
                    'same central difference of the closed-form discrete dispersion k = (2/D) arcsin((D/2) sqrt(eps) '
                    'omega_t / C0). error = max error + the number of failed checks; tolerance 1e-9: all are '
                    'identities up to the eigenvector orthogonality eps ||T|| / eigenvalue gap ~ 1e-12.',
        specs=lambda: {},
        check=_check_projections,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_slab_spp_discrete',
        label='EM slab modes: plasmonic line and PML filter',
        description='CPU check of physics/em/slab_modes on a plasmonic H_z line (no model run): a metal half-line '
                    '(eps -2 - 0.2j, Re eps < 0) below air on uniform 5 nm cells at 550 nm, 16 stretched PML cells '
                    '(s = 1 - 4j depth^3) at both ends, so the PML modes of the metal side and the air side (Re '
                    'k_t^2 far above k0^2) must be dropped by the PML-mode filter (> 50 % of the |rho|-weighted |u|^2 '
                    'in stretched cells, |rho| = w |s| / |eps| underweights the metal) and the surface plasmon must '
                    'be mode 1. Reference: the exact discrete surface mode of two uniform half-lines with the '
                    'interface face eps = (eps_m + 1) / 2, b eps_d - a eps_m - a b eps_f = 0 with a = 1 - 1/r_d, b = '
                    '1/r_m - 1, r + 1/r = 2 + dx^2 (k_t^2 - k0^2 eps) (|r| < 1 per side; continuum limit eps_d kappa_m '
                    '+ eps_m kappa_d = 0, the face mean cancels the first-order error, so the discrete n_eff is within '
                    'O((kappa dx)^2) of sqrt(eps_m / (eps_m + 1)), listed in details). Layouts: deep (0.5 um metal, '
                    '1 um air, ~11 decay lengths: 332 cells, dense eigvals of the whole spectrum + inverse '
                    'iteration) |n_eff / reference - 1| <= 1e-7 (end perturbation ~|u_end|^2 ~ 1e-10); tight (0.1 '
                    'um metal, 0.2 um air, 2.3 decay lengths: about 1 % of the mode in the PMLs) within 5 %, so the '
                    'filter keeps a mode whose tails reach both PMLs. Both: power = +1 W/m to 1e-9 (|eps_m| > 1, '
                    'net forward flux) and Re n_eff > 1; a dropped surface mode or a kept PML mode gives errors of '
                    'order 30 % or more. error = worst error in tolerance units + failed checks, tolerance 1.',
        specs=lambda: {},
        check=_check_spp,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_slab_degenerate_orthogonal',
        label='EM slab modes: degenerate lossy modes orthogonal',
        description='CPU check of physics/em/slab_modes (no model run): on lossy periodic lines (eps (1.5 - '
                    '0.01j)^2) the plane waves +-k are exactly degenerate, and the complex eigensolvers return an '
                    'arbitrary basis of each pair (e.g. e^{+-jks}, whose bilinear self-product sum u^2 rho vanishes '
                    'on a full period); solve_slab_modes recombines clusters closer than 1e-8 ||T|| by a pivoted '
                    'bilinear Gram-Schmidt. Lines: ez and hz plates of 50 cells of 20 nm (mu_r 2; dense numpy eig, '
                    'periodic corner) and an ez line of 700 cells of 5 nm (scipy shift-invert with the corner). The '
                    'first 5 modes (k = 0 and the pairs p = 1, 2) must equal the exact k_t^2 = k0^2 eps mu_r - (4 / '
                    'dx^2) sin^2(pi floor(p/2) / m) to 1e-10 relative and every pair of them must have |mode_overlap| '
                    '<= 1e-8 (distinct eigenvalues: eps ||T|| / gap and the ARPACK residual; recombined pairs: '
                    'round-off). error = worst error in tolerance units + failed checks, tolerance 1.',
        specs=lambda: {},
        check=_check_degenerate,
        max_wall_time=60.0),
)

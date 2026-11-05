"""Self-test of the shared Yee operators (A1) against one real GPU half-step of both FDTD engines, the E7 sign
convention, the face-material helpers and the graded mesher's end cells; no model run (specs() is empty, check({}) computes everything)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import max_relative_error

# (A) uniform periodic grid: Laplacian eigenvalue of a cosine mode
_UNIFORM_SHAPE = (32, 24)
_UNIFORM_CELL = 10e-9                 # m
_UNIFORM_MODES = (3, 5)               # periods per grid length along x, y
_UNIFORM_PHASES = (0.3, 0.7)          # rad
# (A') Bloch phase: kx, ky [rad/m] of 2.3 and -1.7 periods per grid length (off the grid harmonics, both signs)
_BLOCH_K = (2.0 * math.pi * 2.3 / (_UNIFORM_SHAPE[0] * _UNIFORM_CELL),
            -2.0 * math.pi * 1.7 / (_UNIFORM_SHAPE[1] * _UNIFORM_CELL))
# (B, C) random graded grid
_GRADED_SHAPE = (17, 13)
_GRADED_SEED = 20261002
_GRADED_BASE_CELL = 1e-8              # m; widths in [1, 1.5) x this; also the CPML grading cell size
_GRADED_NPML = 4
# (C) sign checks
_SIGN_WAVELENGTH = 1e-6                      # m, vacuum
_DRUDE_PARAMS = (1.37e16, 4.08e13, 9.84)     # omega_p [rad/s], gamma [1/s], eps_inf (gold-like)
_LORENTZ_PARAMS = (1.0, 3e15, 1e14)          # delta_eps, omega_0 [rad/s], gamma [1/s]
_LOSSY_EPS, _LOSSY_SIGMA = 2.0, 1e3          # eps_background, sigma_e [S/m]
# (D) face helpers
_FACE_SHAPE = (5, 4)
# (E) fused H_z face coefficients
_FACE_SIGMA_MAX = 3e5                 # S/m; k = sigma dt / (2 eps) <~ 0.4 at eps 1, so ca stays well away from 0
_FACE_COEFFICIENT_RTOL = 1e-6         # float32; the rebuild should match bit for bit
# em_drude_courant_bound: free-electron aluminium (Ordal) filling an all-PEC box
_ALUMINIUM_DRUDE = (2.24e16, 1.22e14, 1.0)   # omega_p [rad/s], gamma [1/s], eps_inf
_BOUND_CELL = 10e-9                          # m; largest stable courant_factor 0.967 there
_BOUND_BOX = 0.4e-6                          # m, square
_BOUND_STEPS = 300
_UNSTABLE_FACTOR = 0.99                      # the default courant_factor
_STABLE_MARGIN = 0.98                        # stable runs at this fraction of the largest stable factor
_STABLE_GROWTH_LIMIT = 1e3                   # max |E_z| / initial max; an unstable run (|z| ~1.5 / step) overflows
_UNSTABLE_GROWTH_MIN = 1e6
# em_drude_velocity_form: Drude J kernels against the float64 leap-frog recursion at constant E. 'hold': a weakly
# damped metal (gamma dt 2e-6, where a float32 c1 = 1 - gamma dt is off by 1.3 % in gamma) held at its DC current;
# 'transient': a strongly damped one (gamma dt 1e-2) charging from J = 0
_VELOCITY_OMEGA_P = 1.37e16                   # rad/s
_VELOCITY_DT = 1e-17                          # s
_VELOCITY_SCENARIOS = (('hold', 2e11, 10000, True), ('transient', 1e15, 300, False))   # (name, gamma, steps, steady)
_VELOCITY_FIELD = 1.0                         # V/m
_VELOCITY_TOLERANCE = 1e-5                    # relative J; the former c1 J + c2 E form drifts ~3e-4 in 'hold'
_VELOCITY_2D_HOLD_STEPS = 4000                # 2D case: the old form's ~linear drift is still ~1e-4 (10x the tolerance)
# mesh_graded_end_cells: x axes of (whole + fraction) fine cells ending inside a fine band (CPU only)
_END_FINE_CELL = 10e-9                       # m
_END_BASE_CELL = 30e-9                       # m
_END_MAX_RATIO = 1.15
_END_WHOLE_CELLS = 200
_END_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
_END_RTOL = 1e-6                             # float32 cell widths

_TOLERANCE = 1e-10                    # float64 arithmetic of identical stencils
_ENGINE_STEP_SCALE = 1e-5             # maps the float32 tolerance 1e-5 of B onto _TOLERANCE


def _apply(operator, field):
    """operator @ field on the flat index i*ny + j, reshaped to the field's (nx, ny)."""
    return np.asarray(operator @ field.ravel()).reshape(field.shape)


def _laplacian_residual(operators, boundaries):
    """(A) max |L f - lambda f| / (|lambda| max |f|) of a cosine mode on the uniform periodic grid."""
    nx, ny = _UNIFORM_SHAPE
    mode_x, mode_y = _UNIFORM_MODES
    phase_x, phase_y = _UNIFORM_PHASES
    config = boundaries.BoundaryConfig('periodic', 'periodic', 'periodic', 'periodic')
    cell_widths = (np.full(nx, _UNIFORM_CELL), np.full(ny, _UNIFORM_CELL))
    ops = operators.yee_operators(cell_widths, config, _UNIFORM_CELL)
    kx_dx = 2.0 * math.pi * mode_x / nx
    ky_dy = 2.0 * math.pi * mode_y / ny
    field = np.outer(np.cos(kx_dx * np.arange(nx) + phase_x), np.cos(ky_dy * np.arange(ny) + phase_y))
    eigenvalue = (-(2.0 / _UNIFORM_CELL ** 2) * (1.0 - math.cos(kx_dx))
                  - (2.0 / _UNIFORM_CELL ** 2) * (1.0 - math.cos(ky_dy)))
    laplacian_field = _apply(operators.scalar_laplacian(ops), field)
    return (float(np.max(np.abs(laplacian_field - eigenvalue * field)))
            / (abs(eigenvalue) * float(np.max(np.abs(field)))))


def _bloch_residual(operators, boundaries, yee_grid):
    """(A') max relative residual of the Bloch-phased Laplacian on exp(-j (kx x + ky y)) (uniform grid, both axes
    Bloch, k off the grid harmonics) and of the Bloch-wrapped face-to-centre mean of exp(-j k x) on hi faces."""
    nx, ny = _UNIFORM_SHAPE
    kx, ky = _BLOCH_K
    config = boundaries.BoundaryConfig('bloch', 'bloch', 'bloch', 'bloch')
    config.bloch_kx, config.bloch_ky = kx, ky
    cell_widths = (np.full(nx, _UNIFORM_CELL), np.full(ny, _UNIFORM_CELL))
    ops = operators.yee_operators(cell_widths, config, _UNIFORM_CELL)
    x_centres = (np.arange(nx) + 0.5) * _UNIFORM_CELL
    y_centres = (np.arange(ny) + 0.5) * _UNIFORM_CELL
    field = np.outer(np.exp(-1j * kx * x_centres), np.exp(-1j * ky * y_centres))
    eigenvalue = (-(2.0 / _UNIFORM_CELL ** 2) * (1.0 - math.cos(kx * _UNIFORM_CELL))
                  - (2.0 / _UNIFORM_CELL ** 2) * (1.0 - math.cos(ky * _UNIFORM_CELL)))
    laplacian_residual = (float(np.max(np.abs(_apply(operators.scalar_laplacian(ops), field) - eigenvalue * field)))
                          / abs(eigenvalue))
    # hi-x face samples of exp(-j kx x) (faces at x_i + dx/2): centre mean = cos(kx dx/2) exp(-j kx x_i) on every cell
    wrap_factor = np.exp(1j * operators.bloch_wrap_phases(cell_widths, config)[0])
    face_values = field * np.exp(-0.5j * kx * _UNIFORM_CELL)
    centre_mean = 0.5 * (face_values + yee_grid.lower_face_partner(face_values, 0, True, wrap_factor=wrap_factor))
    mean_residual = float(np.max(np.abs(centre_mean - math.cos(0.5 * kx * _UNIFORM_CELL) * field)))
    return max(laplacian_residual, mean_residual)


def _vacuum_engine(polarization, config, grid):
    """FDTD engine of `polarization` filled with vacuum on the bare `grid` (no Domain); `config` must have CPML sigma 0."""
    from ...core.material_grid import MaterialGrid
    from .boundaries import BoundaryManager
    from .fields import Fields2D_Ez, Fields2D_Hz
    from .solvers import FDTDEngine2D_Ez, FDTDEngine2D_Hz
    fields_class, engine_class = ((Fields2D_Ez, FDTDEngine2D_Ez) if polarization == 'ez'
                                  else (Fields2D_Hz, FDTDEngine2D_Hz))
    # sigma = alpha = 0 on the outermost CPML rows: the psi coefficient a = 0 / 0 there is replaced by 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        manager = BoundaryManager(config, grid.nx, grid.ny, _GRADED_BASE_CELL, float(grid.dt),
                                  dx_arr=np.asarray(grid.dx_arr), dy_arr=np.asarray(grid.dy_arr),
                                  polarization=polarization)
    return engine_class(grid, fields_class(grid.nx, grid.ny), MaterialGrid(grid.nx, grid.ny), manager)


def _engine_half_step_residual(engine, ops, rng):
    """(B) max relative error of one real GPU H and one E half-step from zero fields (new field / db or cb = the
    kappa-folded discrete curl, CPML psi = 0 at sigma 0) against the operators."""
    import cupy as cp
    fields = engine.fields
    shape = (engine.nx, engine.ny)

    def load(name):
        """Random float32 values written into fields.<name>, returned as float64."""
        values = rng.standard_normal(shape).astype(np.float32)
        getattr(fields, name)[...] = cp.asarray(values)
        return values.astype(np.float64)

    def host(array):
        return cp.asnumpy(array).astype(np.float64)

    fields.reset()
    if engine.polarization == 'ez':
        ez = load('ez')
        engine.update_h()
        pairs = [(host(fields.hy) / host(engine.db), _apply(ops.dx_h, ez)),
                 (-host(fields.hx) / host(engine.db), _apply(ops.dy_h, ez))]
        fields.reset()
        hx, hy = load('hx'), load('hy')
        engine.update_e()
        pairs.append((host(fields.ez) / host(engine.cb), _apply(ops.dx_e, hy) - _apply(ops.dy_e, hx)))
    else:
        ex, ey = load('ex'), load('ey')
        engine.update_h()
        pairs = [(host(fields.hz) / host(engine.db), _apply(ops.dy_e, ex) - _apply(ops.dx_e, ey))]
        fields.reset()
        hz = load('hz')
        engine.update_e()
        pairs += [(host(fields.ex) / host(engine.cb_x), _apply(ops.dy_h, hz)),
                  (-host(fields.ey) / host(engine.cb_y), _apply(ops.dx_h, hz))]
    return max(max_relative_error(measured, reference) for measured, reference in pairs)


def _sign_violations(operators, boundaries, dispersion, materials, cell_widths):
    """(C) Number of failed exp(+j omega t) sign and build checks at the 1 um angular frequency."""
    omega = 2.0 * math.pi * C0 / _SIGN_WAVELENGTH
    config = boundaries.BoundaryConfig.all_pml(_GRADED_NPML)
    profile = boundaries.cpml_axis_profiles(config, _GRADED_SHAPE[0], 'x', _GRADED_BASE_CELL)
    checks = []
    for sigma, kappa, alpha in ((profile.sigma_e, profile.kappa_e, profile.alpha_e),
                                (profile.sigma_h, profile.kappa_h, profile.alpha_h)):
        stretch = operators.pml_stretch(sigma, kappa, alpha, omega)
        lossy = sigma > 0.0
        checks.append(bool(np.any(lossy)))
        checks.append(bool(np.all(stretch.imag[lossy] < 0.0)))
        checks.append(bool(np.all(stretch[~lossy] == kappa[~lossy])))
    complex_ops = operators.yee_operators(cell_widths, config, _GRADED_BASE_CELL, omega=omega)
    checks.extend(bool(matrix.dtype == np.complex128)
                  for matrix in (complex_ops.dx_e, complex_ops.dx_h, complex_ops.dy_e, complex_ops.dy_h))

    drude = materials.DrudePole(*_DRUDE_PARAMS)
    lorentz = materials.LorentzPole(*_LORENTZ_PARAMS)
    checks.append(drude.susceptibility(omega).imag < 0.0)
    checks.append(lorentz.susceptibility(omega).imag < 0.0)
    drude_spectrum = drude.susceptibility(np.array([0.5 * omega, omega, 2.0 * omega]))
    checks.append(drude_spectrum.dtype == np.complex128 and bool(np.all(drude_spectrum.imag < 0.0)))
    permittivity = dispersion.complex_permittivity(
        np.array([_LOSSY_EPS]), np.array([_LOSSY_SIGMA]), [(drude, None)], omega)
    checks.append(bool(np.all(permittivity.imag < 0.0)))
    checks.append(_even_wall_fold_residual(operators, boundaries, cell_widths) < 1e-12)
    return sum(1 for passed in checks if not passed)


def _even_wall_fold_residual(operators, boundaries, cell_widths):
    """max |A' 1| over the unpinned rows / max |A|, A' = pin_image_cells(Laplacian, outer ring, sign +1) of an all-pmc
    grid: an even image keeps constants in the interior null space (pin_cells alone does not)."""
    ops = operators.yee_operators(cell_widths, boundaries.BoundaryConfig.all_pmc(), _GRADED_BASE_CELL)
    laplacian = operators.scalar_laplacian(ops)
    outer_ring = np.ones((ops.nx, ops.ny), dtype=bool)
    outer_ring[1:-1, 1:-1] = False
    folded = operators.pin_image_cells(laplacian, outer_ring, outer_ring.astype(np.float64))
    row_sums = folded @ np.ones(ops.nx * ops.ny)
    return float(np.max(np.abs(row_sums[~outer_ring.ravel()])) / abs(laplacian).max())


def _face_partner_index(index, axis, periodic):
    """(i, j) of the cell across the hi face of cell `index` along `axis` (explicit reference for part D)."""
    partner = list(index)
    if partner[axis] < _FACE_SHAPE[axis] - 1:
        partner[axis] += 1
    elif periodic:
        partner[axis] = 0
    return tuple(partner)


def _face_helper_violations(yee_grid, rng):
    """(D) Number of face_average (graded widths: dual-cell weighted; uniform widths: bit-equal to the plain mean) /
    face_pec results (axis 0 / 1, periodic or not) that differ from explicit loops."""
    values = rng.standard_normal(_FACE_SHAPE)
    mask = rng.random(_FACE_SHAPE) < 0.5
    violations = 0
    for axis in (0, 1):
        graded_widths = rng.uniform(0.5, 2.0, _FACE_SHAPE[axis])
        uniform_widths = np.full(_FACE_SHAPE[axis], 0.7)
        for periodic in (True, False):
            expected_weighted = np.empty(_FACE_SHAPE)
            expected_mean = np.empty(_FACE_SHAPE)
            expected_pec = np.empty(_FACE_SHAPE, dtype=bool)
            for index in np.ndindex(*_FACE_SHAPE):
                partner = _face_partner_index(index, axis, periodic)
                w_self, w_partner = graded_widths[index[axis]], graded_widths[partner[axis]]
                expected_weighted[index] = (values[index] * w_self + values[partner] * w_partner) / (w_self + w_partner)
                expected_mean[index] = (values[index] + values[partner]) * 0.5
                expected_pec[index] = bool(mask[index]) or bool(mask[partner])
            weighted = yee_grid.face_average(values, axis, periodic, graded_widths)
            violations += int(not np.allclose(weighted, expected_weighted, rtol=1e-12, atol=1e-12))
            violations += int(not np.array_equal(yee_grid.face_average(values, axis, periodic, uniform_widths),
                                                 expected_mean))
            violations += int(not np.array_equal(yee_grid.face_pec(mask, axis, periodic), expected_pec))
            violations += int(yee_grid.face_average(values.astype(np.float32), axis, periodic,
                                                    graded_widths).dtype != np.float32)
    return violations


def _hz_face_coefficient_violations(engine, yee_grid, config, cell_widths, rng):
    """(E) Number of H_z E-face arrays (eps, ca, cb of E_x / E_y) from the fused rebuild that differ from face_average
    of random eps_xx / eps_yy / sigma_e followed by the E_z cell-coefficient kernel."""
    import cupy as cp
    from ...core.constants import DTYPE, EPS0
    from .solvers.fdtd_ez_2d_kernels import e_coefficients
    materials = engine.materials
    shape = (engine.nx, engine.ny)
    for name, low, high in (('eps_xx', 1.0, 12.0), ('eps_yy', 1.0, 12.0), ('sigma_e', 0.0, _FACE_SIGMA_MAX)):
        getattr(materials, name)[...] = cp.asarray(rng.uniform(low, high, shape).astype(np.float32))
    engine.materials_changed({'eps_xx', 'eps_yy', 'sigma_e'})
    eps0 = DTYPE(EPS0)
    dx_cells, dy_cells = cell_widths
    violations = 0
    for axis, periodic, widths, cell_eps, face_arrays in (
            (1, config.periodic_y, dy_cells, materials.eps_xx, (engine.eps_x, engine.ca_x, engine.cb_x)),
            (0, config.periodic_x, dx_cells, materials.eps_yy, (engine.eps_y, engine.ca_y, engine.cb_y))):
        face_eps = yee_grid.face_average(cell_eps, axis, periodic, widths, xp=cp)
        face_sigma = yee_grid.face_average(materials.sigma_e, axis, periodic, widths, xp=cp)
        expected_ca, expected_cb = cp.empty_like(face_eps), cp.empty_like(face_eps)
        e_coefficients(face_sigma, face_eps, DTYPE(engine.dt), eps0, expected_ca, expected_cb)
        for measured, expected in zip(face_arrays, (face_eps * eps0, expected_ca, expected_cb)):
            violations += int(not bool(cp.allclose(measured, expected, rtol=_FACE_COEFFICIENT_RTOL, atol=0.0)))
    return violations


def _check_operators(results_by_name):
    """Parts A-E; error = max(residual A, residual B x 1e-5) + the number of failed checks."""
    from ...core.mesh import NonUniformGrid2D
    from . import boundaries, dispersion, materials, operators, yee_grid

    residual_laplacian = _laplacian_residual(operators, boundaries)
    residual_bloch = _bloch_residual(operators, boundaries, yee_grid)

    rng = np.random.default_rng(_GRADED_SEED)
    nx, ny = _GRADED_SHAPE
    cell_widths = (_GRADED_BASE_CELL * (1.0 + 0.5 * rng.random(nx)),
                   _GRADED_BASE_CELL * (1.0 + 0.5 * rng.random(ny)))
    grid = NonUniformGrid2D(*cell_widths, verbose=False)
    kernel_cell_widths = yee_grid.grid_cell_widths(grid)     # the float32-rounded widths the engines use
    residual_engine_steps = 0.0
    face_coefficient_violations = 0
    for config in (boundaries.BoundaryConfig.all_pml(_GRADED_NPML),
                   boundaries.BoundaryConfig.with_periodic_x(_GRADED_NPML)):
        # CPML sigma 0 keeps psi at 0 (kappa grading unchanged), so a half-step is the kappa-folded curl alone.
        config.cpml_sigma_factor = 0.0
        ops = operators.yee_operators(kernel_cell_widths, config, _GRADED_BASE_CELL)
        for polarization in ('ez', 'hz'):
            engine = _vacuum_engine(polarization, config, grid)
            residual_engine_steps = max(residual_engine_steps, _engine_half_step_residual(engine, ops, rng))
            if polarization == 'hz':
                face_coefficient_violations += _hz_face_coefficient_violations(
                    engine, yee_grid, config, kernel_cell_widths, rng)

    sign_violations = _sign_violations(operators, boundaries, dispersion, materials, cell_widths)
    face_violations = _face_helper_violations(yee_grid, rng)

    measured = max(residual_laplacian, residual_bloch, residual_engine_steps * _ENGINE_STEP_SCALE)
    error = measured + sign_violations + face_violations + face_coefficient_violations
    return {'measured': measured, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'residual_laplacian': residual_laplacian, 'residual_bloch': residual_bloch,
                        'residual_engine_half_steps': residual_engine_steps,
                        'sign_violations': sign_violations,
                        'face_helper_violations': face_violations,
                        'hz_face_coefficient_violations': face_coefficient_violations}}


def _check_cpml_complex_symmetry(results_by_name):
    """relative_asymmetry of diag(D) A, D = stretched_cell_areas, A = a pinned Helmholtz operator with random complex
    face weights / cell term on graded CPML grids at 1 um (Cartesian, walls + CPML, polar)."""
    import scipy.sparse
    from ...solvers.cudss_factor import relative_asymmetry
    from .boundaries import BoundaryConfig
    from .operators import helmholtz_operator, pin_image_cells, stretched_cell_areas, yee_operators
    omega = 2.0 * math.pi * C0 / _SIGN_WAVELENGTH
    rng = np.random.default_rng(_GRADED_SEED + 1)
    nx, ny = _GRADED_SHAPE
    graded = (_GRADED_BASE_CELL * (1.0 + 0.5 * rng.random(nx)), _GRADED_BASE_CELL * (1.0 + 0.5 * rng.random(ny)))
    angles = 0.05 * (1.0 + 0.5 * rng.random(ny))   # polar phi widths [rad]
    cases = {'cartesian_cpml': (BoundaryConfig.all_pml(_GRADED_NPML), graded, None),
             'walls_cpml': (BoundaryConfig('pmc', 'cpml', 'pec', 'cpml'), graded, None),
             'polar_cpml': (BoundaryConfig('pmc', 'cpml', 'pec', 'pec'), (graded[0], angles), 2e-7)}
    asymmetries, unscaled = {}, {}
    for name, (config, widths, radius_min) in cases.items():
        config.cpml_npml = _GRADED_NPML
        ops = yee_operators(widths, config, _GRADED_BASE_CELL, omega=omega, polar_radius_min=radius_min)
        weights = [1.0 / (1.0 + rng.random((nx, ny)) - 0.2j * rng.random((nx, ny))) for _ in range(2)]
        cell_term = (omega / C0) ** 2 * (2.0 + rng.random((nx, ny)) - 0.3j * rng.random((nx, ny)))
        operator = helmholtz_operator(ops, weights[0], weights[1], cell_term)
        pinned = np.zeros((nx, ny), dtype=bool)
        pinned[0, :] = True   # an even image row, as a mirror plane
        operator = pin_image_cells(operator, pinned, pinned.astype(np.float64))
        symmetrizer = stretched_cell_areas(widths, config, _GRADED_BASE_CELL, omega, polar_radius_min=radius_min)
        asymmetries[name] = relative_asymmetry(scipy.sparse.diags(symmetrizer) @ operator)
        unscaled[name] = relative_asymmetry(operator)
    measured = max(asymmetries.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': 1e-12,
            'details': {'asymmetry': asymmetries, 'asymmetry_without_symmetrizer': unscaled}}


def _ez_field_growth(em, rng):
    """max |E_z| after _BOUND_STEPS steps from random E_z (H = J = 0) over the initial max (inf on overflow / NaN)."""
    import cupy as cp
    em.fields.ez[...] = cp.asarray(rng.standard_normal(em.fields.ez.shape).astype(np.float32))
    initial_peak = float(cp.max(cp.abs(em.fields.ez)))
    for _ in range(_BOUND_STEPS):
        em.engine.advance()
    final_peak = float(cp.max(cp.abs(em.fields.ez)))
    return final_peak / initial_peak if math.isfinite(final_peak) else math.inf


def _check_drude_courant_bound(results_by_name):
    """Build check raises / passes around the largest stable factor (ez, hz); ez runs grow / stay bounded likewise."""
    from ...core.domain import Domain
    from . import EMPhysics
    from .boundaries import BoundaryConfig
    from .materials import DrudePole
    omega_p, gamma, eps_inf = _ALUMINIUM_DRUDE
    largest_stable_factor = math.sqrt(eps_inf / (1.0 + (omega_p * _BOUND_CELL) ** 2 / (8.0 * C0 ** 2)))
    stable_factor = _STABLE_MARGIN * largest_stable_factor
    rng = np.random.default_rng(_GRADED_SEED)
    failed, details = 0, {'largest_stable_factor': largest_stable_factor, 'stable_factor': stable_factor}
    for polarization in ('ez', 'hz'):
        for label, courant_factor in (('unstable', _UNSTABLE_FACTOR), ('stable', stable_factor)):
            boundary_config = BoundaryConfig.all_pec()
            domain = Domain.uniform(_BOUND_BOX, _BOUND_BOX, _BOUND_CELL, courant=courant_factor,
                                    boundary_config=boundary_config, verbose=False)
            em = EMPhysics(domain, polarization, boundary_config=boundary_config)
            em.add_drude_rectangle(0.0, 0.0, _BOUND_BOX, _BOUND_BOX, DrudePole(omega_p, gamma, eps_inf))
            try:
                em.finalize()
                raised = False
            except ValueError:
                raised = True
            failed += int(raised != (label == 'unstable'))
            details[f'{polarization}_{label}_check_raised'] = raised
            if polarization == 'ez':
                growth = _ez_field_growth(em, rng)
                details[f'ez_{label}_growth'] = growth
                failed += int(not (growth > _UNSTABLE_GROWTH_MIN if label == 'unstable'
                                   else growth < _STABLE_GROWTH_LIMIT))
    return {'measured': failed, 'expected': 0, 'error': failed, 'tolerance': 0.5, 'details': details}


def drude_velocity_form_errors(make_steppers, hold_steps=None):
    """{'<backend> <kernel> <scenario>': max relative J error} of the steppers `make_steppers(xp)` -> {kernel: (shape,
    step(damping, drive, e, j))} against the float64 recursion J* + (J0 - J*) c1^n, on cpu and a usable gpu;
    hold_steps (None: _VELOCITY_SCENARIOS) overrides the 'hold' step count."""
    from ...core.backend import cp, gpu_available, to_host
    from ...core.constants import EPS0
    from .materials import DrudePole
    errors = {}
    for backend, xp in (('cpu', np), ('gpu', cp)) if gpu_available() else (('cpu', np),):
        for kernel_name, (shape, step) in make_steppers(xp).items():
            for scenario, gamma, steps, start_steady in _VELOCITY_SCENARIOS:
                if scenario == 'hold' and hold_steps is not None:
                    steps = hold_steps
                pole = DrudePole(_VELOCITY_OMEGA_P, gamma)
                damping, drive = pole.ade_coefficients(_VELOCITY_DT)
                gamma_dt = float(pole.gamma) * _VELOCITY_DT
                c1 = (1.0 - gamma_dt / 2.0) / (1.0 + gamma_dt / 2.0)
                steady_current = EPS0 * float(pole.omega_p) ** 2 / float(pole.gamma) * _VELOCITY_FIELD   # sigma_dc E
                start_current = steady_current if start_steady else 0.0
                expected = steady_current + (start_current - steady_current) * c1 ** steps
                e = xp.full(shape, _VELOCITY_FIELD, dtype=np.float32)
                j = xp.full(shape, start_current, dtype=np.float32)
                for _ in range(steps):
                    step(damping, drive, e, j)
                measured = to_host(j).astype(np.float64)
                errors[f'{backend} {kernel_name} {scenario}'] = float(np.max(np.abs(measured - expected))) / expected
    return errors


def _drude_2d_steppers(xp):
    """{kernel: (shape, step)} of the E_z (store / summed total) and H_z (plain / fused power) Drude J kernels."""
    from .solvers import fdtd_ez_2d_kernels as ez_kernels
    from .solvers import fdtd_hz_2d_kernels as hz_kernels
    shape = (4, 4)
    total, power = xp.zeros(shape, dtype=np.float32), xp.zeros(shape, dtype=np.float32)
    weight = xp.ones(shape, dtype=np.float32)

    def ez_total(damping, drive, e, j):
        total.fill(0)
        ez_kernels.drude_j_update_total(damping, drive, e, j, total)
    return {'ez_store': (shape, lambda damping, drive, e, j: ez_kernels.drude_j_update_store(damping, drive, e, j,
                                                                                             total)),
            'ez_total': (shape, ez_total),
            'hz': (shape, lambda damping, drive, e, j: hz_kernels.drude_j_update(damping, drive, weight, e, j)),
            'hz_power': (shape, lambda damping, drive, e, j: hz_kernels.drude_j_update_power(damping, drive, weight,
                                                                                             e, j, power))}


def _check_drude_velocity_form(results_by_name):
    """2D Drude J kernels hold sigma_dc E at gamma dt 2e-6 and follow a damped transient (relative 1e-5)."""
    errors = drude_velocity_form_errors(_drude_2d_steppers, hold_steps=_VELOCITY_2D_HOLD_STEPS)
    measured = max(errors.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': _VELOCITY_TOLERANCE,
            'details': errors}


def _check_mesh_end_cells(results_by_name):
    """Graded x axes ending in a fine band: min cell >= dx_t (1 - dx_t / (2 L)), ratios <= max_ratio, sum = L."""
    from ...core.mesh import MeshConfig
    failed, worst_min_cell, worst_ratio = 0, math.inf, 1.0
    for fraction in _END_FRACTIONS:
        length = (_END_WHOLE_CELLS + fraction) * _END_FINE_CELL
        for band_start in (0.0, 0.5 * length):     # fine everywhere / coarse half graded into the fine end band
            mesh = MeshConfig(length, length, _END_BASE_CELL, max_ratio=_END_MAX_RATIO)
            mesh.add_refinement_region(band_start, 0.0, length, length, _END_FINE_CELL)
            cells = mesh.cell_spacings()[0].astype(np.float64)
            min_cell = float(cells.min()) / _END_FINE_CELL
            ratio = float(np.max(np.maximum(cells[1:] / cells[:-1], cells[:-1] / cells[1:])))
            worst_min_cell, worst_ratio = min(worst_min_cell, min_cell), max(worst_ratio, ratio)
            failed += int(min_cell < 1.0 - 0.5 * _END_FINE_CELL / length - _END_RTOL)
            failed += int(ratio > _END_MAX_RATIO * (1.0 + _END_RTOL))
            failed += int(abs(float(cells.sum()) - length) > _END_RTOL * length)
    return {'measured': failed, 'expected': 0, 'error': failed, 'tolerance': 0.5,
            'details': {'min_cell_over_fine_target': worst_min_cell, 'max_adjacent_ratio': worst_ratio}}


OPERATOR_SELFTEST_CASES = (
    SelfTestCase(
        name='em_operators_consistency',
        label='EM Yee operator consistency',
        description='Check of the shared Yee difference operators against the GPU FDTD engines (no model run). (A) '
                    'Uniform periodic 32 x 24 '
                    'grid (10 nm cells): the scalar Laplacian dx_e dx_h + dy_e dy_h applied to cos(2 pi 3 i/32 + 0.3) '
                    'cos(2 pi 5 j/24 + 0.7) must equal lambda times the field, lambda = -(2/dx^2)(1 - cos(kx dx)) - '
                    '(2/dy^2)(1 - cos(ky dy)). (A\') The same grid Bloch-periodic on both axes with kx, ky of 2.3 and '
                    '-1.7 periods per grid length: the phased-wrap Laplacian on exp(-j (kx x + ky y)) must give the '
                    'same eigenvalue, and the face-to-centre mean of exp(-j kx x) on the hi faces, with the wrapped lo '
                    'face of cell 0 times exp(+j kx L), must equal cos(kx dx/2) exp(-j kx x) on every cell. (B) Random '
                    'graded 17 x 13 grid (widths 10-15 nm) with all-CPML and '
                    'x-periodic boundaries (4 PML cells, kappa graded, sigma 0 so psi stays 0), vacuum: the E_z and '
                    'H_z engines run one real H and one E half-step from random fields (the other set 0); new field / '
                    'db (cb) must equal the kappa-folded operator curl (pins centre / face spacing selection, the '
                    'E_x / E_y orientation and the last face row on periodic axes to the kernels). (C) '
                    'Signs at 1 um in exp(+j omega t): PML stretch Im < 0 inside the layers and exactly kappa outside, '
                    'a complex operator build, Drude / Lorentz susceptibility and lossy complex_permittivity Im < 0, '
                    'and pin_image_cells of an even image on the outer '
                    'ring of an all-pmc graded grid keeps constants in the interior null space of the Laplacian '
                    '(relative row sums < 1e-12). (D) face_average (width-weighted on graded widths, the plain '
                    'mean bit for bit on uniform ones) / face_pec against explicit loops. (E) On the H_z engines of '
                    'B with random eps_xx / eps_yy (1-12) and sigma_e (0-3e5 S/m): the fused face eps / ca / cb '
                    'rebuild must equal face_average followed by the E_z cell-coefficient kernel (relative 1e-6). '
                    'error = '
                    "max(residual A, residual A', residual B x 1e-5) + the number of failed checks. Tolerance 1e-10: "
                    "A and A' compare float64 arithmetic (~1e-15); B compares float32 engine fields whose own "
                    'tolerance 1e-5 is scaled by 1e-5.',
        specs=lambda: {},
        check=_check_operators,
        max_wall_time=60.0, backends=('gpu',)),
    SelfTestCase(
        name='em_fdfd_cpml_complex_symmetry',
        label='EM FDFD symmetrizer with CPML',
        description='CPU check of the 2D FDFD symmetrizer (no model run): on a random graded 17 x 13 grid (widths '
                    '10-15 nm, 4 CPML cells at 1 um, complex stretch) a Helmholtz operator dx_e W_x dx_h + dy_e W_y '
                    'dy_h + diag(c) with random complex face weights and cell term, its x_lo row pinned as an even '
                    'image (pin_image_cells), times D = operators.stretched_cell_areas (dx s_x dy s_y, polar times '
                    'the stretched centre radius) must be complex symmetric: all-CPML, pmc / pec walls + CPML and a '
                    'polar grid (r_min 200 nm, phi widths ~0.05 rad, CPML on r_hi). It is the D with which an own '
                    'cuDSS factor of the eigenfrequency study is an LDL^T (factorization cudss / cudss_mixed). error '
                    '= the worst max |DA - (DA)^T| / max |DA|, tolerance 1e-12; details also give the asymmetry '
                    'without D.',
        specs=lambda: {},
        check=_check_cpml_complex_symmetry,
        max_wall_time=60.0, backends=('cpu',)),
    SelfTestCase(
        name='em_drude_courant_bound',
        label='EM Drude Courant bound',
        description='GPU check of the leap-frog Drude stability bound (no model run): free-electron aluminium '
                    '(omega_p 2.24e16 rad/s, gamma 1.22e14 rad/s, eps_inf 1) fills an all-PEC 0.4 um box of 10 nm '
                    'cells. The bound eps_inf mu_r >= courant_factor^2 + mu_r (omega_p dt)^2 / 4 gives a largest '
                    'stable courant_factor 0.967. EMPhysics.finalize must raise at 0.99 and pass at 0.98 x 0.967 for '
                    'ez and hz; an ez run of 300 steps from random E_z must grow over 1e6 (overflow) at 0.99 and stay '
                    'below 1e3 x its initial peak at 0.98 x 0.967. error = the number of failed checks, tolerance 0.5.',
        specs=lambda: {},
        check=_check_drude_courant_bound,
        max_wall_time=60.0, backends=('gpu',)),
    SelfTestCase(
        name='em_drude_velocity_form',
        label='EM Drude velocity form',
        description='Check of the 2D Drude J kernels (no model run; cpu twins, and the CUDA kernels when a GPU is '
                    'usable): E_z store / summed-total and H_z plain / fused-power updates in float32 at constant E = 1 '
                    'V/m, omega_p 1.37e16 rad/s, dt 1e-17 s. (hold) gamma 2e11 1/s (gamma dt 2e-6), started at the DC '
                    f'current sigma_dc E = eps0 omega_p^2 / gamma E, {_VELOCITY_2D_HOLD_STEPS} steps: J must stay '
                    'there. (transient) gamma '
                    '1e15 1/s (gamma dt 1e-2) from J = 0, 300 steps: J must follow J* (1 - c1^n) of the float64 '
                    'leap-frog c1 = (1 - gamma dt / 2) / (1 + gamma dt / 2). error = the worst relative J error, '
                    'tolerance 1e-5 (the velocity form J += damping J + drive E keeps ~1e-7; a float32 c1 J + c2 E '
                    'drifts about linearly, ~3e-4 over 10000 steps, so ~1e-4 in hold). Not yet run at 4000 hold '
                    'steps.',
        specs=lambda: {},
        check=_check_drude_velocity_form,
        max_wall_time=60.0, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='mesh_graded_end_cells',
        label='Graded mesh end cells',
        description='CPU check of the graded 1D mesher (no model run, no GPU): x axes of 200.1-200.9 fine cells (10 '
                    'nm target, 30 nm base, max_ratio 1.15), fine everywhere or graded from a coarse half into a fine '
                    'end band. The end remainder must be rounded to whole cells, never kept as a short last cell '
                    '(it would set dx_min, hence dt): every cell >= 10 nm (1 - 10 nm / (2 L)), adjacent ratios <= '
                    '1.15 and the widths sum to L (relative 1e-6, float32 widths). error = the number of failed '
                    'checks, tolerance 0.5.',
        specs=lambda: {},
        check=_check_mesh_end_cells,
        max_wall_time=60.0),
)

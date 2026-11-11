"""Self-tests of the guided (2.5D, axial_wavenumber) vibration modes of solid_mechanics: the host operator against its
discrete closed forms (CPU) and eigenfrequency model runs (roller box symbols, the k_z = 0 split into plane strain +
antiplane, a free square rod's flexural / torsional / longitudinal branches); physics modules only inside the CPU check."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import no_runs, scaled_worst_result
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests import _lame
from .selftests_eigenfrequency import _add_rank_items, _symbols, _table_frequencies

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_STUDY = 'eigenfrequency'         # = model.study_eigenfrequency.STUDY_NAME
_TABLE = 'modes'
_YOUNGS_MODULUS = 70e9            # Pa
_POISSON_RATIO = 0.25
_DENSITY = 2200.0                 # kg/m^3
_FREQUENCY_TOLERANCE = 1e-8       # relative (float64 K and M, eigsh to round-off)
_FIELD_TOLERANCE = 1e-5           # mode-1 cell-centre u (peak node |u| = 1 m, float32 outputs)

# ---- roller box: 10 x 4 mm, uniform 0.5 mm cells (20 x 8), rollers on all four faces (u_z free there), k_z = 2 pi / 10
# mm. u_x = A sin(p pi i / N_x) cos(q pi j / N_y), u_y = B cos sin, w = C cos cos (nodal) are exact discrete modes of
# the uniform 2.5D Q1 pencil: per order pair the (up to) 3 x 3 symbol of the 1D linear-element symbols K(t) = (2 / h)(1
# - cos t), M(t) = (h / 3)(2 + cos t), S(t) = sin t with B_xx = c11 Kx My + mu (Mx Ky + k^2 Mx My), B_yy alike, B_ww =
# mu (Kx My + Mx Ky) + c11 k^2 Mx My, B_xy = (lambda + mu) Sx Sy, B_xw = (lambda + mu) k Sx My, B_yw = (lambda + mu) k
# Sy Mx = omega^2 rho Mx My; u_x for 0 < p < N_x, u_y for 0 < q < N_y, w always. Mode 1 = (1, 0) lower branch (u_x and
# w), C / A ~ -0.5, so its peak node is the u_x crest at i = N_x / 2.
_BOX_SIZE = (10e-3, 4e-3)         # m
_BOX_CELL = 0.5e-3                # m
_BOX_CELLS = (20, 8)
_BOX_WAVENUMBER = 2.0 * math.pi / 10e-3   # rad/m
_BOX_LOWEST = 8
_BOX_SHIFTED = 3
_BOX_SHIFT_INDEX = 4              # the shifted target sits 3 % above the 5th lowest mode
_BOX_SHIFT_FACTOR = 1.03

# ---- k_z = 0 split: the same block, Fixed on x_lo (u_x = u_y = u_z = 0), rollers on y_lo / y_hi, x_hi free. The guided
# spectrum must be the plane-strain spectrum (a run without axial_wavenumber) merged with the antiplane one, mu Laplacian
# of w: w = sin(t_x i) cos(t_y j), t_x = (2 m + 1) pi / (2 N_x) (held at i = 0, mirror at i = N_x), t_y = q pi / N_y.
_SPLIT_MODES = 10

# ---- free square rod: 1 x 1 mm, 16 x 16 cells, no supports (k_z > 0 has no rigid motion), k_z = 50 and 100 rad/m (k a
# = 0.05, 0.1). Lowest four: the flexural pair, torsion, longitudinal. omega^2 / k^2 (omega / k^2 flexural) is even in k,
# so (4 s(k) - s(2 k)) / 3 removes the k^2 term: longitudinal -> E / rho exactly (uniaxial stress in the cross-section is
# linear in x, y, inside Q1), flexural -> sqrt(E I / (rho A)), torsion -> G J / (rho I_p) (Saint-Venant J of the square;
# both carry the O(h^2) Q1 error of the anticlastic / warping fields).
_ROD_SIDE = 1e-3                  # m
_ROD_CELLS = 16
_ROD_WAVENUMBERS = (50.0, 100.0)  # rad/m
_ROD_MODES = 4
_ROD_LONGITUDINAL_TOLERANCE = 1e-6
_ROD_FLEXURAL_TOLERANCE = 1e-2
_ROD_TORSION_TOLERANCE = 2e-2
_ROD_DEGENERACY_TOLERANCE = 1e-7

# ---- CPU operator case: 6 x 4 uniform cells, k_z = 800 rad/m (symbols), a graded mesh with a void cell (symmetry)
_HOST_CELLS = (6, 4)
_HOST_CELL = 0.5e-3
_HOST_WAVENUMBER = 800.0
_HOST_SYMMETRY_TOLERANCE = 1e-14
_HOST_RESIDUAL_TOLERANCE = 1e-11
_HOST_SPLIT_TOLERANCE = 1e-13
_HOST_NULL_TOLERANCE = 1e-10      # rigid-mode eigenvalues / largest


def _guided_symbol(theta_x, theta_y, wavenumber, cell):
    """(3 x 3 stiffness symbol over (u_x, u_y, w), mass symbol rho Mx My) of the separable roller-box mode."""
    lame_lambda, mu = _lame(_YOUNGS_MODULUS, _POISSON_RATIO)
    c11, coupling = lame_lambda + 2.0 * mu, lame_lambda + mu
    stiff_x, mass_x, slope_x = _symbols(theta_x, cell)
    stiff_y, mass_y, slope_y = _symbols(theta_y, cell)
    axial = wavenumber ** 2 * mass_x * mass_y
    symbol = np.array([
        [c11 * stiff_x * mass_y + mu * (mass_x * stiff_y + axial), coupling * slope_x * slope_y,
         coupling * wavenumber * slope_x * mass_y],
        [coupling * slope_x * slope_y, c11 * mass_x * stiff_y + mu * (stiff_x * mass_y + axial),
         coupling * wavenumber * slope_y * mass_x],
        [coupling * wavenumber * slope_x * mass_y, coupling * wavenumber * slope_y * mass_x,
         mu * (stiff_x * mass_y + mass_x * stiff_y) + c11 * axial]])
    return symbol, _DENSITY * mass_x * mass_y


def _present(p, q, cells):
    """Components (0 u_x, 1 u_y, 2 w) of order pair (p, q) that are not identically 0 on the roller box."""
    return [component for component, present in ((0, 0 < p < cells[0]), (1, 0 < q < cells[1]), (2, True)) if present]


def _box_modes(wavenumber=_BOX_WAVENUMBER, cells=_BOX_CELLS, cell=_BOX_CELL):
    """Sorted [(frequency [Hz], (p, q, branch), eigenvector over the present components)] of the guided roller box."""
    modes = []
    for p in range(cells[0] + 1):
        for q in range(cells[1] + 1):
            symbol, mass = _guided_symbol(p * math.pi / cells[0], q * math.pi / cells[1], wavenumber, cell)
            present = _present(p, q, cells)
            values, vectors = np.linalg.eigh(symbol[np.ix_(present, present)])
            for branch, value in enumerate(values):
                vector = np.zeros(3)
                vector[present] = vectors[:, branch]
                modes.append((math.sqrt(value / mass) / (2.0 * math.pi), (p, q, branch), vector))
    modes.sort(key=lambda mode: mode[0])
    return modes


def _block_spec(title, features, axial_wavenumber, target_frequency, n_resonances, size=_BOX_SIZE, cell=_BOX_CELL):
    """Uniform isotropic block with a resonance table, eigenfrequency study (axial_wavenumber None = plane strain)."""
    settings = {'target_frequency': target_frequency}
    if axial_wavenumber is not None:
        settings['axial_wavenumber'] = axial_wavenumber
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size[0], size_y=size[1]),
        materials=[MaterialSpec('block', properties={'youngs_modulus': _YOUNGS_MODULUS, 'poisson_ratio': _POISSON_RATIO,
                                                     'rho': _DENSITY}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings=settings,
                             features=list(features) + [FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type=_STUDY, settings={'n_resonances': n_resonances}))


# ---- roller box ----
def _box_shifted_target():
    return _BOX_SHIFT_FACTOR * _box_modes()[_BOX_SHIFT_INDEX][0]


def _box_specs():
    rollers = [FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi'])]
    title = 'Self-test: guided vibration modes of a roller-supported block (solid mechanics, axial_wavenumber)'
    return {'lowest': _block_spec(title, rollers, _BOX_WAVENUMBER, 0.0, _BOX_LOWEST),
            'shifted': _block_spec(title, rollers, _BOX_WAVENUMBER, _box_shifted_target(), _BOX_SHIFTED)}


def _box_mode_one_error(results, vector):
    """max |u - closed form| of mode 1 ((1, 0) lower: u_x = A sin, w = C cos along x, uniform in y) at the cell centres,
    scaled by the peak node (the u_x crest, |A| > |C|): u_x = sin(t (i + 1/2)) cos(t / 2), u_z = (C / A) cos(t (i +
    1/2)) cos(t / 2), u_y = 0."""
    amplitude_x, amplitude_w = vector[0], vector[2]
    if not abs(amplitude_x) > abs(amplitude_w):
        raise ValueError(f'mode 1 has |C| >= |A| ({amplitude_w:.4g}, {amplitude_x:.4g}): its peak node is ambiguous')
    n_x = _BOX_CELLS[0]
    theta = math.pi / n_x
    phase = theta * (np.arange(n_x) + 0.5)
    expected = {'u_x': (np.sin(phase) * math.cos(0.5 * theta))[:, None],
                'u_y': np.zeros((n_x, 1)),
                'u_z': (amplitude_w / amplitude_x * np.cos(phase) * math.cos(0.5 * theta))[:, None]}
    errors = []
    for name, values in expected.items():
        field = np.asarray(results.field(f'{_PHYSICS}.{name}').values, dtype=np.float64)
        if field.shape != _BOX_CELLS:
            raise ValueError(f'{name} window shape {field.shape} differs from the {_BOX_CELLS} block cells')
        errors.append(float(np.max(np.abs(field - values))))
    return max(errors)


def _check_box(results_by_name):
    """Both runs' frequencies in rank order against the 3 x 3 symbols, snapshot counts and the mode-1 shape incl. u_z."""
    modes = _box_modes()
    if modes[0][1] != (1, 0, 0):
        raise ValueError(f'expected the (1, 0) lower branch lowest, got {modes[0][1]}')
    if not modes[1][0] > 1.001 * modes[0][0]:
        raise ValueError('mode 1 is (nearly) degenerate: its shape is not unique')
    target = _box_shifted_target()
    expected_runs = {'lowest': modes[:_BOX_LOWEST],
                     'shifted': sorted(modes, key=lambda mode: abs(mode[0] - target))[:_BOX_SHIFTED]}
    errors, tolerances, details = {}, {}, {}
    for run, expected in expected_runs.items():
        measured, _ = _add_rank_items(run, results_by_name[run], [frequency for frequency, _, _ in expected],
                                      len(expected), _FREQUENCY_TOLERANCE, errors, tolerances)
        details[run] = {'orders': [list(orders) for _, orders, _ in expected],
                        'frequency_expected': [frequency for frequency, _, _ in expected],
                        'frequency_measured': measured}
    errors['lowest mode 1 field'] = _box_mode_one_error(results_by_name['lowest'], modes[0][2])
    tolerances['lowest mode 1 field'] = _FIELD_TOLERANCE
    return scaled_worst_result(errors, tolerances, **details, target_frequency_shifted=target)


# ---- k_z = 0 split ----
def _split_specs():
    features = [FeatureSpec('clamp', 'fixed', ['x_lo']), FeatureSpec('rollers', 'roller', ['y_lo', 'y_hi'])]
    title = 'Self-test: guided modes at k_z = 0 = plane strain + antiplane (solid mechanics)'
    return {'guided': _block_spec(title, features, 0.0, 0.0, _SPLIT_MODES),
            'plane': _block_spec(title, features, None, 0.0, _SPLIT_MODES)}


def _antiplane_frequencies():
    """Sorted antiplane (pure w) frequencies [Hz] of the clamped / rollered block: mu (Kx My + Mx Ky) = omega^2 rho Mx My
    with the quarter-wave x phases (2 m + 1) pi / (2 N_x) and the cos y phases q pi / N_y."""
    _, mu = _lame(_YOUNGS_MODULUS, _POISSON_RATIO)
    n_x, n_y = _BOX_CELLS
    frequencies = []
    for m in range(n_x):
        stiff_x, mass_x, _ = _symbols((2 * m + 1) * math.pi / (2 * n_x), _BOX_CELL)
        for q in range(n_y + 1):
            stiff_y, mass_y, _ = _symbols(q * math.pi / n_y, _BOX_CELL)
            value = mu * (stiff_x * mass_y + mass_x * stiff_y) / (_DENSITY * mass_x * mass_y)
            frequencies.append(math.sqrt(value) / (2.0 * math.pi))
    return sorted(frequencies)


def _check_split(results_by_name):
    """Guided k_z = 0 frequencies = the lowest of plane-strain (the 'plane' run) merged with the antiplane closed form."""
    plane = _table_frequencies(results_by_name['plane'], _SPLIT_MODES, 'plane')
    guided = _table_frequencies(results_by_name['guided'], _SPLIT_MODES, 'guided')
    antiplane = _antiplane_frequencies()
    expected = sorted(plane + antiplane)[:_SPLIT_MODES]
    errors = {'frequency': max(abs(value / reference - 1.0) for value, reference in zip(guided, expected))}
    labels = results_by_name['guided'].summary().get('snapshot_labels') or []
    errors['snapshots'] = abs(len(labels) - _SPLIT_MODES)
    tolerances = {'frequency': _FREQUENCY_TOLERANCE, 'snapshots': 0.5}
    return scaled_worst_result(errors, tolerances, plane=plane, antiplane_lowest=antiplane[:_SPLIT_MODES],
                               guided=guided, expected=expected,
                               n_antiplane_among_expected=sum(1 for value in expected if value not in plane))


# ---- free square rod ----
def _rod_specs():
    title = 'Self-test: guided modes of a free square rod (solid mechanics, axial_wavenumber)'
    return {f'k{index}': _block_spec(title, [], wavenumber, 0.0, _ROD_MODES, size=(_ROD_SIDE, _ROD_SIDE),
                                     cell=_ROD_SIDE / _ROD_CELLS)
            for index, wavenumber in enumerate(_ROD_WAVENUMBERS, start=1)}


def _square_torsion_constant(side):
    """Saint-Venant torsion constant J [m^4] of a square of `side` (series of the rectangle, 50 odd terms)."""
    series = sum(math.tanh(n * math.pi / 2.0) / n ** 5 for n in range(1, 101, 2))
    return side ** 4 / 3.0 * (1.0 - 192.0 / math.pi ** 5 * series)


def _check_rod(results_by_name):
    """k^2-extrapolated branch coefficients of the lowest four modes (flexural pair, torsion, longitudinal) against
    E / rho (exact), sqrt(E I / (rho A)) and G J / (rho I_p); the flexural pair degenerate (square symmetry)."""
    lame_lambda, mu = _lame(_YOUNGS_MODULUS, _POISSON_RATIO)
    k1, k2 = _ROD_WAVENUMBERS
    if abs(k2 - 2.0 * k1) > 1e-12 * k2:
        raise ValueError('the rod runs need k2 = 2 k1 for the k^2 extrapolation')
    omega = {run: [2.0 * math.pi * value for value in _table_frequencies(results_by_name[run], _ROD_MODES, run)]
             for run in ('k1', 'k2')}

    def extrapolated(index, power):
        """(4 s(k1) - s(k2)) / 3 of s = omega^power / k^2 (power 1 flexural, 2 the others)."""
        first, second = (omega[run][index] ** power / wavenumber ** 2
                         for run, wavenumber in (('k1', k1), ('k2', k2)))
        return (4.0 * first - second) / 3.0

    side = _ROD_SIDE
    area, inertia, polar = side ** 2, side ** 4 / 12.0, side ** 4 / 6.0
    flexural = math.sqrt(_YOUNGS_MODULUS * inertia / (_DENSITY * area))
    torsion = mu * _square_torsion_constant(side) / (_DENSITY * polar)
    longitudinal = _YOUNGS_MODULUS / _DENSITY
    measured = {'flexural 1': extrapolated(0, 1), 'flexural 2': extrapolated(1, 1), 'torsion': extrapolated(2, 2),
                'longitudinal': extrapolated(3, 2)}
    expected = {'flexural 1': flexural, 'flexural 2': flexural, 'torsion': torsion, 'longitudinal': longitudinal}
    errors = {name: abs(measured[name] / expected[name] - 1.0) for name in measured}
    errors['flexural degeneracy'] = max(abs(omega[run][1] / omega[run][0] - 1.0) for run in omega)
    tolerances = {'flexural 1': _ROD_FLEXURAL_TOLERANCE, 'flexural 2': _ROD_FLEXURAL_TOLERANCE,
                  'torsion': _ROD_TORSION_TOLERANCE, 'longitudinal': _ROD_LONGITUDINAL_TOLERANCE,
                  'flexural degeneracy': _ROD_DEGENERACY_TOLERANCE}
    return scaled_worst_result(errors, tolerances, omega=omega, measured=measured, expected=expected,
                               lame_lambda=lame_lambda)


# ---- CPU: the host operator ----
def _guided_pencil(cells, dx, dy, wavenumber, solid=None):
    """Host (K, M) of guided_modes.guided_operators on widths dx, dy with the test material (void where solid False)."""
    from .guided_modes import guided_operators
    lame_lambda, mu = _lame(_YOUNGS_MODULUS, _POISSON_RATIO)
    solid = np.ones(cells, dtype=bool) if solid is None else solid
    return guided_operators(np.where(solid, lame_lambda, 0.0), np.where(solid, mu, 0.0),
                            np.where(solid, _DENSITY, 0.0), dx, dy, wavenumber)


def _check_host_operator(results_by_name):
    """guided_operators on the host: symmetric K and M (graded mesh, a void cell); at k_z = 0 the in-plane block equals
    the plane-strain stencil, the antiplane block mu (Kxx + Kyy), no coupling; the free body has 4 rigid modes at k_z =
    0 and none at k_z > 0; the roller-box sin / cos vectors satisfy K x = omega^2 M x on the free dofs for every order
    pair with the symbol eigenvalues."""
    import scipy.sparse
    from .system import ElasticPattern, stiffness_stencil
    lame_lambda, mu = _lame(_YOUNGS_MODULUS, _POISSON_RATIO)
    n_x, n_y = _HOST_CELLS
    errors, details = {}, {}

    # symmetry on a graded mesh with one void cell
    graded_x = _HOST_CELL * np.linspace(0.6, 1.4, n_x)
    graded_y = _HOST_CELL * np.linspace(1.3, 0.7, n_y)
    solid = np.ones(_HOST_CELLS, dtype=bool)
    solid[2, 1] = False
    stiffness, mass = _guided_pencil(_HOST_CELLS, graded_x, graded_y, _HOST_WAVENUMBER, solid)
    scale = abs(stiffness).max()
    errors['symmetry'] = max(abs(stiffness - stiffness.T).max() / scale,
                             abs(mass - mass.T).max() / abs(mass).max())

    # k_z = 0 split against the plane-strain stencil
    split, _ = _guided_pencil(_HOST_CELLS, graded_x, graded_y, 0.0, solid)
    lam_cells, mu_cells = np.where(solid, lame_lambda, 0.0), np.where(solid, mu, 0.0)
    pattern = ElasticPattern(n_x + 1, n_y + 1, components=2, xp=np)
    plane = pattern.csr(stiffness_stencil(lam_cells + 2.0 * mu_cells, lam_cells, lam_cells + 2.0 * mu_cells, mu_cells,
                                          graded_x, graded_y, np.float64, xp=np), scipy.sparse)
    antiplane_stencil = stiffness_stencil(mu_cells, 0.0 * mu_cells, mu_cells, mu_cells, graded_x, graded_y, np.float64,
                                          xp=np)
    antiplane = pattern.csr(antiplane_stencil, scipy.sparse)
    n_nodes = (n_x + 1) * (n_y + 1)
    nodes = np.arange(n_nodes)
    in_plane = np.stack((3 * nodes, 3 * nodes + 1), axis=1).ravel()
    axial = 3 * nodes + 2
    split = split.tocsr()
    errors['k_z = 0 split'] = max(
        abs(split[in_plane][:, in_plane] - plane).max() / scale,
        abs(split[axial][:, axial] - antiplane[2 * nodes][:, 2 * nodes]).max() / scale,
        abs(split[in_plane][:, axial]).max() / scale)

    # rigid modes of the free uniform body: 4 at k_z = 0 (3 in plane + the z translation), none at k_z > 0
    widths_x, widths_y = np.full(n_x, _HOST_CELL), np.full(n_y, _HOST_CELL)
    null_counts = {}
    for wavenumber in (0.0, _HOST_WAVENUMBER):
        free_stiffness, free_mass = _guided_pencil(_HOST_CELLS, widths_x, widths_y, wavenumber)
        mass_dense = free_mass.toarray()
        cholesky = np.linalg.cholesky(mass_dense)
        inverse = np.linalg.inv(cholesky)
        values = np.linalg.eigvalsh(inverse @ free_stiffness.toarray() @ inverse.T)
        null_counts[wavenumber] = int(np.count_nonzero(values < _HOST_NULL_TOLERANCE * values[-1]))
    errors['rigid modes'] = abs(null_counts[0.0] - 4) + abs(null_counts[_HOST_WAVENUMBER])
    details['null_counts'] = {str(key): value for key, value in null_counts.items()}

    # roller box: separable discrete modes on the free dofs
    stiffness, mass = _guided_pencil(_HOST_CELLS, widths_x, widths_y, _HOST_WAVENUMBER)
    held = np.zeros((n_x + 1, n_y + 1, 3), dtype=bool)
    held[[0, -1], :, 0] = True
    held[:, [0, -1], 1] = True
    free = ~held.ravel()
    stiffness_free = stiffness.tocsr()[free][:, free]
    mass_free = mass.tocsr()[free][:, free]
    node_i, node_j = np.meshgrid(np.arange(n_x + 1), np.arange(n_y + 1), indexing='ij')
    residuals = []
    for frequency, (p, q, _), vector in _box_modes(_HOST_WAVENUMBER, _HOST_CELLS, _HOST_CELL):
        theta_x, theta_y = p * math.pi / n_x, q * math.pi / n_y
        shape = np.stack((vector[0] * np.sin(theta_x * node_i) * np.cos(theta_y * node_j),
                          vector[1] * np.cos(theta_x * node_i) * np.sin(theta_y * node_j),
                          vector[2] * np.cos(theta_x * node_i) * np.cos(theta_y * node_j)), axis=-1).ravel()[free]
        applied = stiffness_free @ shape
        residual = applied - (2.0 * math.pi * frequency) ** 2 * (mass_free @ shape)
        residuals.append(float(np.linalg.norm(residual) / max(np.linalg.norm(applied), 1e-300)))
    errors['separable modes'] = max(residuals)
    details['n_separable_modes'] = len(residuals)
    tolerances = {'symmetry': _HOST_SYMMETRY_TOLERANCE, 'k_z = 0 split': _HOST_SPLIT_TOLERANCE,
                  'rigid modes': 0.5, 'separable modes': _HOST_RESIDUAL_TOLERANCE}
    return scaled_worst_result(errors, tolerances, **details)


GUIDED_MODES_CASES = (
    SelfTestCase(
        name='solid_guided_operator_exactness',
        label='Solid mechanics: 2.5D guided-mode operator (host)',
        description='CPU check of physics/mechanics/guided_modes on host arrays (no model run): K and M symmetric on a '
                    f'graded {_HOST_CELLS[0]} x {_HOST_CELLS[1]} mesh with a void cell (relative '
                    f'{_HOST_SYMMETRY_TOLERANCE:g}); at k_z = 0 the (u_x, u_y) block equals the plane-strain Q1 '
                    'stencil, the u_z block mu (Kxx + Kyy) and their coupling vanishes ('
                    f'{_HOST_SPLIT_TOLERANCE:g}); a free uniform body has exactly 4 rigid modes at k_z = 0 and none at '
                    f'k_z = {_HOST_WAVENUMBER:g} rad/m; on a roller box of uniform {_HOST_CELL * 1e3:g} mm cells every '
                    'separable mode u_x = A sin cos, u_y = B cos sin, w = C cos cos with (A, B, C) and omega^2 from the '
                    '3 x 3 symbol of the 1D linear-element symbols satisfies K x = omega^2 M x on the free dofs '
                    f'(relative residual {_HOST_RESIDUAL_TOLERANCE:g}). measured = error = the worst deviation in '
                    'tolerance units, tolerance 1.',
        specs=no_runs, check=_check_host_operator),
    SelfTestCase(
        name='solid_guided_modes_roller_box',
        label='Solid mechanics: guided (2.5D) modes of a roller-supported block',
        description=f'eigenfrequency study of solid_mechanics with axial_wavenumber {_BOX_WAVENUMBER:.6g} rad/m on a '
                    f'{_BOX_SIZE[0] * 1e3:g} x {_BOX_SIZE[1] * 1e3:g} mm block of uniform {_BOX_CELL * 1e3:g} mm '
                    f'cells (E {_YOUNGS_MODULUS:g} Pa, nu {_POISSON_RATIO:g}, rho {_DENSITY:g} kg/m^3), rollers on all '
                    f"four faces (u_z free). Runs 'lowest' (target 0, {_BOX_LOWEST} modes) and 'shifted' (target "
                    f'{_BOX_SHIFT_FACTOR:g} x the {_BOX_SHIFT_INDEX + 1}th lowest, {_BOX_SHIFTED} modes). u_x = sin '
                    'cos, u_y = cos sin, u_z = cos cos are exact discrete modes: per order pair a 3 x 3 symbol with '
                    'the k_z couplings (lambda + mu) k Sx My, (lambda + mu) k Sy Mx. The table frequencies must match '
                    f'it in rank order (relative {_FREQUENCY_TOLERANCE:g}), each run stores one snapshot per mode, and '
                    'the live mode 1 ((1, 0) lower branch) has cell-centre u_x = sin(pi (i + 1/2) / N_x) cos(pi / (2 '
                    'N_x)) and u_z = (C / A) cos(...) cos(pi / (2 N_x)), u_y = 0 within '
                    f'{_FIELD_TOLERANCE:g}: this pins the k_z terms, their sign (u_z = the sin-phase amplitude), the '
                    'three-component mass, the shift and the ranking.',
        specs=_box_specs, check=_check_box),
    SelfTestCase(
        name='solid_guided_modes_plane_strain_limit',
        label='Solid mechanics: guided modes at k_z = 0 = plane strain + antiplane',
        description=f'eigenfrequency study of the same block clamped on x_lo (Fixed: u_x = u_y = u_z = 0) with rollers '
                    f"on y_lo / y_hi: run 'guided' (axial_wavenumber 0) and run 'plane' (no axial_wavenumber), "
                    f'{_SPLIT_MODES} lowest modes each. The guided frequencies must equal the plane-strain ones merged '
                    'with the antiplane closed form (mu Laplacian of u_z: sin((2 m + 1) pi i / (2 N_x)) cos(q pi j / '
                    f'N_y) discrete modes) in rank order (relative {_FREQUENCY_TOLERANCE:g}), one snapshot per mode.',
        specs=_split_specs, check=_check_split),
    SelfTestCase(
        name='solid_guided_modes_free_rod',
        label='Solid mechanics: guided modes of a free square rod',
        description=f'eigenfrequency study of an unsupported {_ROD_SIDE * 1e3:g} mm square rod ({_ROD_CELLS} x '
                    f'{_ROD_CELLS} cells, the block material) at axial_wavenumber {_ROD_WAVENUMBERS[0]:g} and '
                    f'{_ROD_WAVENUMBERS[1]:g} rad/m (k_z > 0: no rigid motion), {_ROD_MODES} lowest modes: the '
                    'flexural pair, torsion, longitudinal. Their k^2-extrapolated coefficients (4 s(k) - s(2 k)) / 3 of '
                    's = omega^2 / k^2 (omega / k^2 flexural) must give E / rho (exact in Q1: the uniaxial-stress '
                    f'cross-section field is linear; relative {_ROD_LONGITUDINAL_TOLERANCE:g}), sqrt(E I / (rho A)) '
                    f'({_ROD_FLEXURAL_TOLERANCE:g}) and G J / (rho I_p) with the Saint-Venant J of the square '
                    f'({_ROD_TORSION_TOLERANCE:g}; both carry the O(h^2) error of the anticlastic / warping fields), '
                    f'the flexural pair degenerate within {_ROD_DEGENERACY_TOLERANCE:g}.',
        specs=_rod_specs, check=_check_rod),
)

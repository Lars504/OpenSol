"""Self-tests of the spherically symmetric FDFD (em_fdfd_sphere): CPU checks of physics/em/spherical_multipoles (Mie
coefficients and cross sections of a dielectric sphere, a PEC sphere cavity vs Bessel zeros, dipole powers in vacuum
and the discrete Poynting identity, open-sphere resonances vs the exact Mie poles) and model runs (plane-wave cross
sections in a frequency_domain study, cavity resonances in an eigenfrequency study); physics imports inside the
checks only."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import EIGENFREQUENCY_STUDY, FREQUENCY_DOMAIN_STUDY, SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import cpu_defaults, float_list, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshAxisSpec, MeshSegmentSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, StudySpec)

_TYPE = 'em_fdfd_sphere'
_PHYSICS = 'em'

# ---- Mie sphere: radius a, eps 4 (n = 2), vacuum outside, size parameter x = k a = pi at 1 um. Orders 1..14 (Wiscombe
# x + 4 x^(1/3) + 2 ~ 11). Interface on a node (a = 400 cells); second order: (k n h)^2 ~ 2.5e-4 at 1.25 nm.
_MIE_RADIUS = 0.5e-6
_MIE_EPS = 4.0
_MIE_OUTER = 0.6e-6
_MIE_CELL = 1.25e-9
_MIE_WAVELENGTH = 1.0e-6
_MIE_ORDERS = tuple(range(1, 15))
_MIE_COEFFICIENT_ORDERS = 6          # |a_l - a_l exact| checked for l <= this (larger l: tiny coefficients)
_MIE_TOLERANCES = {'sigma_ext': 2e-3, 'sigma_sca': 2e-3, 'coefficients': 2e-3, 'absorption': 1e-8}

# ---- PEC sphere cavity: vacuum, radius 1 um, pec at R. Exact: hr (TE) j_l(k a) = 0, er (TM) (x j_l(x))' = 0.
_CAVITY_RADIUS = 1.0e-6
_CAVITY_CELL = _CAVITY_RADIUS / 600
_CAVITY_TARGET = 1.0e-6              # k_t a = 2 pi
_CAVITY_ORDERS = (1, 2, 3)
_CAVITY_RESONANCES = 4
_CAVITY_CHECKED = 2                  # analytic roots nearest the target that must be found per term
_CAVITY_TOLERANCE = 5e-4             # relative k: (k h)^2 / 12 ~ 1e-5, margin for the end cells

# ---- dipoles in vacuum: r_s = 0.2 um (a node), lambda 1 um, open at 0.5 um; orders 1..12 ((k r_s)^(2 l) decay).
_DIPOLE_OUTER = 0.5e-6
_DIPOLE_CELL = 1.0e-9
_DIPOLE_RADIUS = 0.2e-6
_DIPOLE_WAVELENGTH = 1.0e-6
_DIPOLE_ORDERS = tuple(range(1, 13))
_DIPOLE_MOMENT = 1e-12               # A m
_DIPOLE_TOLERANCES = {'power': 1e-3, 'poynting': 1e-9}

# ---- open dielectric sphere resonances: eps 4, radius 1 um, vacuum shell to 1.3 um; order 8 near 1.1 um (a
# whispering-gallery mode of radial order 1). Exact: the Mie poles (denominators of a_l / b_l at complex x).
_OPEN_RADIUS = 1.0e-6
_OPEN_OUTER = 1.3e-6
_OPEN_CELL = 1.0e-9
_OPEN_EPS = 4.0
_OPEN_ORDER = 8
_OPEN_TARGET = 1.1e-6
_OPEN_TOLERANCES = {'k': 5e-4, 'q': 2e-2}

# ---- model runs
_MODEL_CELL = 2.5e-9
_MODEL_ANGLE_CELLS = 16              # theta cells of the (r, theta) Mie run (display only)
_MODEL_WAVELENGTHS = (0.9e-6, 1.0e-6, 1.1e-6)
_MODEL_TOLERANCE = 5e-3              # relative cross sections at 2.5 nm cells
_MODEL_CAVITY_ORDER = 2
_MODEL_CAVITY_CELL = _CAVITY_RADIUS / 400
_MODEL_CAVITY_RESONANCES = 3
_MODEL_CAVITY_TOLERANCE = 1e-3


# ---- analytic references (Bohren & Huffman, exp(-i omega t); ours are their complex conjugates) ----
def _riccati(order, x):
    """(psi, psi', chi, chi') of x j_l(x), x y_l(x) (scipy; x real or complex)."""
    from scipy.special import spherical_jn, spherical_yn
    j, dj = spherical_jn(order, x), spherical_jn(order, x, derivative=True)
    y, dy = spherical_yn(order, x), spherical_yn(order, x, derivative=True)
    return x * j, j + x * dj, x * y, y + x * dy


def _mie_coefficients(order, x, m):
    """Bohren & Huffman (a_l, b_l) of a sphere of relative index m (mu = 1) at size parameter x."""
    psi, dpsi, chi, dchi = _riccati(order, x)
    xi, dxi = psi + 1j * chi, dpsi + 1j * dchi
    psi_m, dpsi_m, _, _ = _riccati(order, m * x)
    a = (m * psi_m * dpsi - psi * dpsi_m) / (m * psi_m * dxi - xi * dpsi_m)
    b = (psi_m * dpsi - m * psi * dpsi_m) / (psi_m * dxi - m * xi * dpsi_m)
    return complex(a), complex(b)


def _mie_cross_sections(wavelength, radius, m, orders):
    """(C_ext, C_sca) [m^2] of the Mie series over `orders` in vacuum."""
    k = 2.0 * math.pi / wavelength
    extinction = scattering = 0.0
    for order in orders:
        a, b = _mie_coefficients(order, k * radius, m)
        weight = 2.0 * math.pi * (2 * order + 1) / k ** 2
        extinction += weight * (a + b).real
        scattering += weight * (abs(a) ** 2 + abs(b) ** 2)
    return extinction, scattering


def _real_roots(function, x_max, step=2e-3):
    """Sign-change roots of a real function on (0.05, x_max) (brentq)."""
    from scipy.optimize import brentq
    samples = np.arange(0.05, x_max, step)
    values = np.array([function(x) for x in samples])
    return [brentq(function, samples[i], samples[i + 1]) for i in range(samples.size - 1)
            if values[i] * values[i + 1] < 0.0]


def _cavity_roots(family, order, x_max):
    """Exact k a of a vacuum PEC sphere: hr j_l(x) = 0, er (x j_l(x))' = 0."""
    from scipy.special import spherical_jn
    if family == 'hr':
        return _real_roots(lambda x: spherical_jn(order, x), x_max)
    return _real_roots(lambda x: spherical_jn(order, x) + x * spherical_jn(order, x, derivative=True), x_max)


def _mie_pole(family, order, x_start, m):
    """Complex size parameter x of a Mie pole near x_start in the exp(+j omega t) convention (Im x > 0 decays): Newton
    on the denominator of a_l ('er') or b_l ('hr') with xi = psi - j chi."""
    def denominator(x):
        psi, dpsi, chi, dchi = _riccati(order, x)
        xi, dxi = psi - 1j * chi, dpsi - 1j * dchi
        psi_m, dpsi_m, _, _ = _riccati(order, m * x)
        if family == 'er':
            return m * psi_m * dxi - xi * dpsi_m
        return psi_m * dxi - m * xi * dpsi_m

    x = complex(x_start)
    for _ in range(60):
        step_size = 1e-7 * abs(x)
        derivative = (denominator(x + step_size) - denominator(x - step_size)) / (2.0 * step_size)
        step = denominator(x) / derivative
        x -= step
        if abs(step) < 1e-14 * abs(x):
            break
    return x


# ---- numerics helpers ----
def _materials(edges, inner_radius, eps_inside, eps_outside=1.0):
    """SphereMaterials: eps_inside for cell centres below inner_radius, else eps_outside; mu 1, no PEC."""
    from .spherical_multipoles import SphereMaterials
    centres = 0.5 * (edges[:-1] + edges[1:])
    eps = np.where(centres < inner_radius, eps_inside, eps_outside).astype(np.complex128)
    return SphereMaterials(eps=eps, mu=np.ones_like(eps), pec=np.zeros(eps.size, dtype=bool))


def _uniform_grid(outer, cell):
    from .spherical_multipoles import sphere_grid
    return sphere_grid(0.0, np.full(int(round(outer / cell)), cell))


def _check_mie_sphere(results_by_name):
    """Plane-wave Mie coefficients and cross sections of the discrete radial solve vs Bohren & Huffman (conjugated);
    C_abs = C_ext - C_sca ~ 0 (lossless: the exterior expansion is exact, the interior conserves flux)."""
    from ...solvers.sparse_direct import factorize_host
    from .spherical_multipoles import (assemble_sphere, mie_cross_sections, outgoing_coefficient,
                                       plane_wave_amplitude, plane_wave_rhs)
    grid = _uniform_grid(_MIE_OUTER, _MIE_CELL)
    materials = _materials(grid.edges, _MIE_RADIUS, _MIE_EPS)
    omega = 2.0 * math.pi * C0 / _MIE_WAVELENGTH
    k = 2.0 * math.pi / _MIE_WAVELENGTH
    m = math.sqrt(_MIE_EPS)
    extinction = scattering = 0.0
    coefficient_error = 0.0
    computed = {}
    for order in _MIE_ORDERS:
        exact = dict(zip(('er', 'hr'), _mie_coefficients(order, k * _MIE_RADIUS, m)))
        for family in ('er', 'hr'):
            system = assemble_sphere(grid, materials, family, order, omega, 'centre', 'radiation')
            alpha = plane_wave_amplitude(system, 1.0)
            values = system.full(factorize_host(system.operator).solve_host(
                system.restricted(plane_wave_rhs(system, alpha))))
            coefficient = -outgoing_coefficient(system, values, alpha) / alpha
            sca, ext = mie_cross_sections(system, coefficient)
            scattering += sca
            extinction += ext
            computed[f'{family}{order}'] = [coefficient.real, coefficient.imag]
            if order <= _MIE_COEFFICIENT_ORDERS:
                coefficient_error = max(coefficient_error, abs(coefficient - exact[family].conjugate()))
    exact_extinction, exact_scattering = _mie_cross_sections(_MIE_WAVELENGTH, _MIE_RADIUS, m, _MIE_ORDERS)
    errors = {'sigma_ext': abs(extinction / exact_extinction - 1.0),
              'sigma_sca': abs(scattering / exact_scattering - 1.0),
              'coefficients': coefficient_error,
              'absorption': abs(extinction - scattering) / exact_extinction}
    return scaled_worst_result(errors, _MIE_TOLERANCES, sigma_ext=extinction, sigma_sca=scattering,
                               exact_sigma_ext=exact_extinction, exact_sigma_sca=exact_scattering,
                               coefficients=computed)


def _check_pec_cavity(results_by_name):
    """Resonances of a vacuum PEC sphere per term (shift-invert, closed: no refinement) vs the Bessel zeros: the
    _CAVITY_CHECKED analytic roots nearest the target found within tolerance, no computed one unmatched."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .spherical_multipoles import assemble_sphere, solve_term_resonances
    grid = _uniform_grid(_CAVITY_RADIUS, _CAVITY_CELL)
    materials = _materials(grid.edges, 0.0, 1.0)
    omega = 2.0 * math.pi * C0 / _CAVITY_TARGET
    k_target = 2.0 * math.pi / _CAVITY_TARGET
    eigen_solver = ArpackShiftInvertSolver()
    worst, details = 0.0, {}
    for order in _CAVITY_ORDERS:
        for family in ('er', 'hr'):
            system = assemble_sphere(grid, materials, family, order, omega, 'centre', 'pec')
            candidates, _ = solve_term_resonances(system, _CAVITY_RESONANCES, eigen_solver)
            computed = [candidate.omega.real / C0 * _CAVITY_RADIUS for candidate in candidates]
            exact = _cavity_roots(family, order, 3.0 * k_target * _CAVITY_RADIUS)
            nearest = sorted(exact, key=lambda x: abs(x - k_target * _CAVITY_RADIUS))[:_CAVITY_CHECKED]
            missed = [min((abs(value - x) / x for value in computed), default=math.inf) for x in nearest]
            unmatched = [min(abs(value - x) / x for x in exact) for value in computed]
            worst = max(worst, *missed, *unmatched)
            details[f'{family}{order}'] = {'computed_ka': float_list(computed), 'exact_ka': float_list(nearest)}
    return scaled_worst_result({'k': worst}, {'k': _CAVITY_TOLERANCE}, terms=details)


def _check_dipoles(results_by_name):
    """Radial and tangential dipoles in vacuum: the emitted power summed over the terms vs eta0 k^2 (I d)^2 / (12 pi),
    and per term the emitted power = the outgoing power of its xi coefficient (the discrete Poynting identity)."""
    from ...solvers.sparse_direct import factorize_host
    from .spherical_multipoles import (SphereDipole, assemble_sphere, homogeneous_dipole_power, outgoing_coefficient,
                                       outgoing_power)
    grid = _uniform_grid(_DIPOLE_OUTER, _DIPOLE_CELL)
    materials = _materials(grid.edges, 0.0, 1.0)
    omega = 2.0 * math.pi * C0 / _DIPOLE_WAVELENGTH
    reference = homogeneous_dipole_power(_DIPOLE_MOMENT, omega, 1.0, 1.0)
    errors = {'power': 0.0, 'poynting': 0.0}
    powers = {}
    for orientation in ('radial', 'tangential'):
        dipole = SphereDipole(radius=_DIPOLE_RADIUS, orientation=orientation, moment=_DIPOLE_MOMENT)
        total = 0.0
        for order in _DIPOLE_ORDERS:
            for family in ('er', 'hr'):
                system = assemble_sphere(grid, materials, family, order, omega, 'centre', 'radiation')
                rhs = system.restricted(dipole.node_rhs(system))
                if not np.any(rhs):
                    continue
                values = system.full(factorize_host(system.operator).solve_host(rhs))
                emitted = dipole.emitted_power(system, values)
                radiated = outgoing_power(system, outgoing_coefficient(system, values))
                total += emitted
                errors['poynting'] = max(errors['poynting'], abs(emitted - radiated) / reference)
        powers[orientation] = total
        errors['power'] = max(errors['power'], abs(total / reference - 1.0))
    # the exact free-space eta0 k^2 (I d)^2 / (12 pi) as a cross-check of homogeneous_dipole_power
    k = 2.0 * math.pi / _DIPOLE_WAVELENGTH
    errors['power'] = max(errors['power'], abs(reference / (ETA0 * k ** 2 * _DIPOLE_MOMENT ** 2 / (12 * math.pi)) - 1))
    return scaled_worst_result(errors, _DIPOLE_TOLERANCES, powers=powers, reference=reference)


def _check_open_resonances(results_by_name):
    """Open dielectric sphere, order 8, both families: the resonance nearest the target (shift-invert + exact
    radiation condition) vs the exact Mie pole found from it (relative k and Q)."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .eigenfrequency import q_factor_of
    from .spherical_multipoles import assemble_sphere, solve_term_resonances
    grid = _uniform_grid(_OPEN_OUTER, _OPEN_CELL)
    materials = _materials(grid.edges, _OPEN_RADIUS, _OPEN_EPS)
    omega = 2.0 * math.pi * C0 / _OPEN_TARGET
    eigen_solver = ArpackShiftInvertSolver()
    errors, details = {'k': 0.0, 'q': 0.0}, {}
    for family in ('er', 'hr'):
        system = assemble_sphere(grid, materials, family, _OPEN_ORDER, omega, 'centre', 'radiation')
        candidates, info = solve_term_resonances(system, 2, eigen_solver)
        if not candidates:
            errors['k'] = math.inf
            details[family] = {'info': {key: value for key, value in info.items() if key != 'warnings'}}
            continue
        computed = complex(candidates[0].omega) / C0 * _OPEN_RADIUS
        exact = _mie_pole(family, _OPEN_ORDER, computed, math.sqrt(_OPEN_EPS))
        q_computed, q_exact = q_factor_of(computed), q_factor_of(exact)
        errors['k'] = max(errors['k'], abs(computed - exact) / abs(exact))
        errors['q'] = max(errors['q'], abs(q_computed / q_exact - 1.0) if math.isfinite(q_exact) else math.inf)
        details[family] = {'computed_ka': [computed.real, computed.imag], 'exact_ka': [exact.real, exact.imag],
                           'q': [q_computed, q_exact], 'refine_iterations': info['refine_iterations']}
    return scaled_worst_result(errors, _OPEN_TOLERANCES, terms=details)


# ---- model runs ----
def _line_domain(outer):
    return DomainSpec(size_x=outer, size_y=None, coordinates='spherical', radius_min=0.0)


def _half_disc_domain(outer):
    """The (r, theta) window over the whole theta = 0 .. pi (the drawing: the meridional half disc)."""
    return DomainSpec(size_x=outer, size_y=math.pi, coordinates='spherical', radius_min=0.0, angle_min=0.0)


def _mie_model_specs():
    """frequency_domain plane-wave run of the Mie sphere on the (r, theta) half disc (cpu backend; radial cells at
    _MODEL_CELL, _MODEL_ANGLE_CELLS theta cells)."""
    return {'mie': ModelSpec(
        title='Self-test: em_fdfd_sphere Mie sphere',
        domain=_half_disc_domain(_MIE_OUTER),
        geometry=[GeometrySpec('core', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': _MIE_RADIUS})],
        materials=[MaterialSpec('glass', properties={'eps_r': _MIE_EPS}, selection=['core'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings={'multipole_orders': [float(order) for order in _MIE_ORDERS]},
                             features=[FeatureSpec('wave', 'plane_wave', [], {'amplitude': 1.0})])],
        mesh=MeshSpec(mode='nonuniform', dx=_MODEL_CELL,
                      axes=[MeshAxisSpec('y', segments=[MeshSegmentSpec(_MODEL_ANGLE_CELLS)])]),
        solver_defaults=cpu_defaults(),
        study=StudySpec(type=FREQUENCY_DOMAIN_STUDY, settings={'wavelengths': list(_MODEL_WAVELENGTHS)}))}


def _check_mie_model(results_by_name):
    """sigma_ext / sigma_sca of the plane_wave result per wavelength vs the Mie series; the summed e_intensity image
    is finite and non-zero inside the half disc (NaN outside it)."""
    results = results_by_name['mie']
    result = results.result(_PHYSICS, 'wave')
    m = math.sqrt(_MIE_EPS)
    errors = {}
    for index, wavelength in enumerate(result['x']):
        exact_extinction, exact_scattering = _mie_cross_sections(float(wavelength), _MIE_RADIUS, m, _MIE_ORDERS)
        errors[f'ext_{index}'] = abs(float(result['series']['sigma_ext'][index]) / exact_extinction - 1.0)
        errors[f'sca_{index}'] = abs(float(result['series']['sigma_sca'][index]) / exact_scattering - 1.0)
    intensity = np.asarray(results.field('e_intensity').values, dtype=np.float64)
    inside = intensity[np.isfinite(intensity)]
    errors['field'] = 0.0 if inside.size and float(inside.max()) > 0.0 else math.inf
    tolerances = {name: _MODEL_TOLERANCE for name in errors}
    return scaled_worst_result(errors, tolerances, wavelengths=float_list(result['x']),
                               sigma_ext=float_list(result['series']['sigma_ext']))


def _cavity_model_specs():
    """eigenfrequency run of the vacuum PEC sphere at one order, both families, with a resonance_table."""
    return {'cavity': ModelSpec(
        title='Self-test: em_fdfd_sphere PEC sphere cavity',
        domain=_line_domain(_CAVITY_RADIUS),
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings={'bc_x_hi': 'pec', 'wavelength': _CAVITY_TARGET,
                                                        'multipole_order': _MODEL_CAVITY_ORDER},
                             features=[FeatureSpec('resonances', 'resonance_table', [], {})])],
        mesh=MeshSpec(mode='uniform', dx=_MODEL_CAVITY_CELL),
        solver_defaults=cpu_defaults(),
        study=StudySpec(type=EIGENFREQUENCY_STUDY, settings={'n_resonances': _MODEL_CAVITY_RESONANCES}))}


def _check_cavity_model(results_by_name):
    """Every resonance_table row (k from its frequency) matches an exact PEC-sphere root of its family; both families
    present; one snapshot per resonance number."""
    results = results_by_name['cavity']
    table = results.result(_PHYSICS, 'resonances')
    columns = table['columns']
    frequency_column, family_column = columns.index('frequency'), columns.index('family')
    errors, families = {}, set()
    for index, row in enumerate(table['rows']):
        family = row[family_column]
        families.add(family)
        ka = 2.0 * math.pi * float(row[frequency_column]) / C0 * _CAVITY_RADIUS
        roots = _cavity_roots(family, _MODEL_CAVITY_ORDER, 2.0 * ka + 2.0)
        errors[f'row_{index}'] = min((abs(ka - x) / x for x in roots), default=math.inf)
    errors['families'] = 0.0 if families == {'er', 'hr'} else math.inf
    labels = results.summary().get('snapshot_labels') or []
    expected_rows = 2 * _MODEL_CAVITY_RESONANCES   # both families
    errors['snapshots'] = 0.0 if len(labels) == len(table['rows']) == expected_rows else math.inf
    tolerances = {name: _MODEL_CAVITY_TOLERANCE for name in errors}
    return scaled_worst_result(errors, tolerances, rows=table['rows'], snapshot_labels=list(labels))


SPHERE_SELFTEST_CASES = (
    SelfTestCase(
        name='em_sphere_mie',
        label='EM sphere: Mie coefficients and cross sections',
        description=(
            'CPU check of physics/em/spherical_multipoles (no model run): a dielectric sphere (eps '
            f'{_MIE_EPS:g}, radius {_MIE_RADIUS * 1e6:g} um, vacuum to {_MIE_OUTER * 1e6:g} um, '
            f'{_MIE_CELL * 1e9:g} nm cells, the interface on a node) lit by a plane wave at '
            f'{_MIE_WAVELENGTH * 1e6:g} um (k a = pi), orders 1..{_MIE_ORDERS[-1]}, both families, exact outgoing '
            'boundary. Normalised errors (pass <= 1): C_ext and C_sca vs the Mie series '
            f"({_MIE_TOLERANCES['sigma_ext']:g} relative), |a_l - conj(a_l Bohren-Huffman)| and the same for b_l for "
            f"l <= {_MIE_COEFFICIENT_ORDERS} ({_MIE_TOLERANCES['coefficients']:g}), |C_ext - C_sca| / C_ext "
            f"({_MIE_TOLERANCES['absorption']:g}: lossless, the discrete flux balance is exact). Pins the er / hr "
            'operators, the dual-cell means at the interface, the radiation term and the incident-wave source.'),
        specs=lambda: {},
        check=_check_mie_sphere,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_sphere_pec_cavity',
        label='EM sphere: PEC sphere cavity resonances',
        description=(
            f'CPU check of spherical_multipoles.solve_term_resonances (no model run): a vacuum sphere of radius '
            f'{_CAVITY_RADIUS * 1e6:g} um with pec at R ({_CAVITY_CELL * 1e9:.4g} nm cells), shift-invert near '
            f'{_CAVITY_TARGET * 1e6:g} um for orders {", ".join(map(str, _CAVITY_ORDERS))} and both families '
            f'({_CAVITY_RESONANCES} each). The {_CAVITY_CHECKED} exact resonances nearest the target (hr: j_l(k a) = '
            "0, er: (x j_l(x))' = 0 at x = k a) must be found and every computed one must match an exact one, within "
            f'{_CAVITY_TOLERANCE:g} relative k. Pins the centre condition, the pec end of both families (Dirichlet / '
            'natural) and the eigen role call.'),
        specs=lambda: {},
        check=_check_pec_cavity,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_sphere_dipoles',
        label='EM sphere: dipole power in vacuum',
        description=(
            f'CPU check of the dipole sources of spherical_multipoles (no model run): a radial and a tangential '
            f'current element of moment {_DIPOLE_MOMENT:g} A m at r = {_DIPOLE_RADIUS * 1e6:g} um in vacuum '
            f'({_DIPOLE_CELL * 1e9:g} nm cells, open at {_DIPOLE_OUTER * 1e6:g} um), {_DIPOLE_WAVELENGTH * 1e6:g} um, '
            f'orders 1..{_DIPOLE_ORDERS[-1]}. Normalised errors: the emitted power summed over the terms vs eta0 k^2 '
            f"(I d)^2 / (12 pi) ({_DIPOLE_TOLERANCES['power']:g} relative: pins the angular couplings sqrt(2 l + 1) / "
            "sqrt((2 l + 1) / 2) and the 1 / (4 pi) of the weak form), and per term the emitted power = the outgoing "
            f"power 2 pi k |c|^2 / (omega mu0) or / (omega eps0) ({_DIPOLE_TOLERANCES['poynting']:g} of the total: "
            'the discrete Poynting identity of the complex symmetric operator).'),
        specs=lambda: {},
        check=_check_dipoles,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_sphere_open_resonances',
        label='EM sphere: open dielectric sphere resonances',
        description=(
            f'CPU check of the resonance refinement of spherical_multipoles (no model run): a dielectric sphere (eps '
            f'{_OPEN_EPS:g}, radius {_OPEN_RADIUS * 1e6:g} um, vacuum to {_OPEN_OUTER * 1e6:g} um, '
            f'{_OPEN_CELL * 1e9:g} nm cells), order {_OPEN_ORDER}, both families, target {_OPEN_TARGET * 1e6:g} um: '
            'the resonance nearest the target (shift-invert with the radiation term frozen at the target, then '
            'Rayleigh-functional iteration with the exact outgoing condition) vs the exact Mie pole found by Newton '
            f"from it: relative complex k ({_OPEN_TOLERANCES['k']:g}) and Q ({_OPEN_TOLERANCES['q']:g})."),
        specs=lambda: {},
        check=_check_open_resonances,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_sphere_mie_model',
        label='EM sphere: Mie cross sections (model run)',
        description=(
            'Model run of em_fdfd_sphere on a spherical (r, theta) domain over theta = 0 .. pi (cpu backend; gpu '
            f'variant): the Mie sphere of em_sphere_mie drawn as a circle, {_MODEL_CELL * 1e9:g} nm radial cells, '
            f'{_MODEL_ANGLE_CELLS} theta cells (mesh axis y segments), a plane_wave in a frequency_domain '
            f"study at {', '.join(f'{value * 1e6:g}' for value in _MODEL_WAVELENGTHS)} um, orders "
            f'1..{_MIE_ORDERS[-1]}, both families. sigma_ext / sigma_sca of the feature result vs the Mie series '
            f'within {_MODEL_TOLERANCE:g} relative; the e_intensity field finite and non-zero.'),
        specs=_mie_model_specs,
        check=_check_mie_model,
        variants=(SelfTestVariant('gpu', tolerance=1.0, backend='gpu'),),
        max_wall_time=120.0),
    SelfTestCase(
        name='em_sphere_cavity_model',
        label='EM sphere: PEC sphere cavity (model run)',
        description=(
            'Model run of em_fdfd_sphere on a 1D spherical line in an eigenfrequency study (cpu backend): the vacuum '
            'PEC sphere of '
            f'em_sphere_pec_cavity at order {_MODEL_CAVITY_ORDER}, both families, {_MODEL_CAVITY_RESONANCES} '
            f'resonances per family, {_MODEL_CAVITY_CELL * 1e9:g} nm cells. Every resonance_table row matches an exact '
            f'root of its family within {_MODEL_CAVITY_TOLERANCE:g} relative k, both families appear, '
            f'{2 * _MODEL_CAVITY_RESONANCES} rows and one snapshot per resonance number.'),
        specs=_cavity_model_specs,
        check=_check_cavity_model,
        max_wall_time=120.0),
)

"""Self-tests of 1D models (symmetry plan 2.20, DomainSpec.size_y null): a heated two-layer slab / cylinder / sphere
and a two-dielectric slab / coax / spherical capacitor against the exact discrete chains and their closed forms, plus
the banded solver on synthetic operators (CPU); registered by study.register."""
import math

import numpy as np

from ..core.constants import EPS0
from .schema import SelfTestCase, SelfTestVariant
from .selftest_helpers import cell_edges, solver_role, worst_error
from .spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                   PhysicsSpec, StudySpec)

_HEAT, _CAPACITOR = 'heat', 'capacitor'
_CELL = 1e-5                       # m, uniform cells along x / r
_OUTER = 1e-3                      # m, b: the x_hi end of both runs
_HEAT_CORE = 4e-4                  # m, the heat run's inner layer 0 < r < r1 (40 cells)
_SHELL_INNER = 2e-4                # m, a = radius_min of the cylinder / sphere capacitor
_CAPACITOR_CORE = 3e-4             # m, the capacitor's inner layer thickness (30 cells from x_lo)
_POWER_DENSITY = 1e8               # W/m^3, uniform heat source
_T_FACE = 300.0                    # K, x_hi temperature = T0 (so T_rise = T - T_face)
_K_CORE, _K_OUTER = 2.0, 0.5       # W/(m K)
_EPS_CORE, _EPS_OUTER = 4.0, 2.0
_VOLTAGE = 1.0                     # V on x_lo of the capacitor
_TOLERANCE = 1e-6                  # float32 field storage ~1e-7 of the rise / potential
# kind -> (geometry setting, area measure M and power p of A_f = M r_f^p, charge unit, analytic bound): the heat
# drops are exact inside the layers (q = Q r / (p + 1) is linear), so only the half cells at x_hi / the interface
# deviate (~(dx / L)^2); the planar capacitor is an exact series of half cells, the radial ones the trapezoid rule
_KINDS = {'cartesian': ('planar', 1.0, 0, 'C/m^2', 1e-6),
          'polar': ('planar', 2.0 * math.pi, 1, 'C/m', 1e-3),
          'spherical': ('axisymmetric', 4.0 * math.pi, 2, 'C', 1e-3)}
_HEAT_BOUND = 1e-3                 # allowed max |T_rise - closed form| / max closed form


def _line_domain(kind, size_x, radius_min=0.0):
    return DomainSpec(size_x=size_x, size_y=None, coordinates=kind, radius_min=radius_min)


def _core(radius):
    """The inner layer: a circle about the origin selects the cells with centre x (r) <= radius."""
    return GeometrySpec('core', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': radius})


def _specs(kind):
    """'heat': uniform Q in 0 < r < b (k_core inside r1), x_hi at T_face, x_lo insulated (slab) or the centre;
    'capacitor': V on x_lo (r = a), 0 V on x_hi, eps_core in the first layer. Both stationary, Automatic solvers."""
    geometry_setting = _KINDS[kind][0]
    heat = ModelSpec(
        title=f'Self-test: 1D {kind} heated two-layer body',
        domain=_line_domain(kind, _OUTER),
        geometry=[_core(_HEAT_CORE)],
        materials=[MaterialSpec('outer', properties={'k': _K_OUTER}, selection=['all']),
                   MaterialSpec('core', properties={'k': _K_CORE}, selection=['core'])],
        physics=[PhysicsSpec('heat', 'heat', settings={'geometry': geometry_setting, 'T0': _T_FACE}, features=[
            FeatureSpec('source', 'heat_source', ['all'], {'Q': _POWER_DENSITY}),
            FeatureSpec('sink', 'temperature', ['x_hi'], {'temperature': _T_FACE})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('source_power', 'heat_Q_ext', [], 'integral'),
                        DerivedValueSpec('core_volume', '1', ['core'], 'integral')])
    radius_min = 0.0 if kind == 'cartesian' else _SHELL_INNER
    capacitor = ModelSpec(
        title=f'Self-test: 1D {kind} two-dielectric capacitor',
        domain=_line_domain(kind, _OUTER - radius_min, radius_min),
        geometry=[_core(radius_min + _CAPACITOR_CORE)],
        materials=[MaterialSpec('outer', properties={'eps_dc': _EPS_OUTER}, selection=['all']),
                   MaterialSpec('core', properties={'eps_dc': _EPS_CORE}, selection=['core'])],
        physics=[PhysicsSpec('electrostatics', 'electrostatics', settings={'geometry': geometry_setting}, features=[
            FeatureSpec('inner', 'voltage', ['x_lo'], {'voltage': _VOLTAGE}),
            FeatureSpec('outer', 'voltage', ['x_hi'], {'voltage': 0.0})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary'))
    return {_HEAT: heat, _CAPACITOR: capacitor}


def _heat_chain(edges, conductivity, power):
    """T_rise at the centres of the exact discrete chain: face flow F_f = Q V(r_f) (conservation, insulated or zero-area
    x_lo), drop F_f / g_f with g_f = A_f / (w_l / 2 k_l + w_r / 2 k_r), the x_hi half cell 2 k A / w; per area measure."""
    widths = np.diff(edges)
    areas = edges ** power
    flows = _POWER_DENSITY * edges ** (power + 1) / (power + 1)   # Q times the volume inside r_f, per measure
    drops = np.empty(widths.size)
    drops[-1] = flows[-1] * widths[-1] / (2.0 * conductivity[-1] * areas[-1])
    drops[:-1] = flows[1:-1] / areas[1:-1] * (widths[:-1] / (2.0 * conductivity[:-1])
                                              + widths[1:] / (2.0 * conductivity[1:]))
    return np.cumsum(drops[::-1])[::-1]


def _heat_closed_form(radius, power):
    """T_rise(r) of uniform Q in 0 < r < b with k_core inside r1: Q / (2 (p + 1) k) (r_hi^2 - r_lo^2) per layer above r."""
    factor = _POWER_DENSITY / (2.0 * (power + 1))
    inner = np.minimum(radius, _HEAT_CORE)
    rise = factor * (_OUTER ** 2 - np.maximum(radius, _HEAT_CORE) ** 2) / _K_OUTER
    return rise + np.where(radius < _HEAT_CORE, factor * (_HEAT_CORE ** 2 - inner ** 2) / _K_CORE, 0.0)


def _capacitor_chain(edges, permittivity, power):
    """(charge per measure M, V at the centres) of the series face conductances eps0 A_f / (w_l / 2 eps_l + w_r / 2
    eps_r), half cells 2 eps0 eps A_f / w on both held faces."""
    widths = np.diff(edges)
    areas = edges ** power
    resistances = np.concatenate(([widths[0] / (2.0 * EPS0 * permittivity[0] * areas[0])],
                                  (widths[:-1] / permittivity[:-1] + widths[1:] / permittivity[1:])
                                  / (2.0 * EPS0 * areas[1:-1]),
                                  [widths[-1] / (2.0 * EPS0 * permittivity[-1] * areas[-1])]))
    charge = _VOLTAGE / float(np.sum(resistances))
    return charge, _VOLTAGE - charge * np.cumsum(resistances)[:-1]


def _capacitor_closed_form(kind, radius_min):
    """Charge of the two-layer capacitor: V / sum of the layer integrals of dr / (eps0 eps M r^p)."""
    measure, power = _KINDS[kind][1], _KINDS[kind][2]
    core_edge = radius_min + _CAPACITOR_CORE

    def layer(low, high, eps):
        if power == 0:
            integral = high - low
        elif power == 1:
            integral = math.log(high / low)
        else:
            integral = 1.0 / low - 1.0 / high
        return integral / (EPS0 * eps * measure)
    return _VOLTAGE / (layer(radius_min, core_edge, _EPS_CORE) + layer(core_edge, _OUTER, _EPS_OUTER))


def _solver_errors(results, physics_name):
    """{'<physics>_banded': 0 / 1}: the Automatic linear solver of a 1D model is banded and factorised."""
    record = solver_role(results, physics_name)
    ok = record.get('type') == 'banded' and (record.get('counters') or {}).get('factorizations', 0) >= 1
    return {f'{physics_name}_banded': 0.0 if ok else 1.0}


def _check(results_by_name, kind):
    """Heat: T_rise vs the discrete chain, the closed form within _HEAT_BOUND, the integral of Q_ext and the core
    volume (metric volumes); capacitor: V and both plate charges vs the discrete chain, the unit, the closed form
    within the kind's bound; both: the banded Automatic solver."""
    _, measure, power, charge_unit, analytic_bound = _KINDS[kind]
    heat, capacitor = results_by_name[_HEAT], results_by_name[_CAPACITOR]
    errors, details = {}, {}

    rise_field = heat.field('heat.T_rise')
    rise = np.asarray(rise_field.values, dtype=np.float64)[:, 0]
    heat_centres = np.asarray(rise_field.x_centers, dtype=np.float64)
    heat_edges = cell_edges(heat_centres, 0.0)
    conductivity = np.where(heat_centres <= _HEAT_CORE, _K_CORE, _K_OUTER)
    expected_rise = _heat_chain(heat_edges, conductivity, power)
    closed_form = _heat_closed_form(heat_centres, power)
    rise_scale = float(np.max(expected_rise))
    closed_deviation = float(np.max(np.abs(rise - closed_form))) / float(np.max(closed_form))
    derived = heat.summary()['derived']
    expected_power = _POWER_DENSITY * measure * _OUTER ** (power + 1) / (power + 1)
    expected_core = measure * _HEAT_CORE ** (power + 1) / (power + 1)
    errors.update(heat_profile=float(np.max(np.abs(rise - expected_rise))) / rise_scale,
                  heat_closed_form_excess=max(0.0, closed_deviation - _HEAT_BOUND),
                  heat_power=abs(float(derived['source_power']) / expected_power - 1.0),
                  core_volume=abs(float(derived['core_volume']) / expected_core - 1.0),
                  heat_cells=0.0 if rise.size == round(_OUTER / _CELL) else 1.0)
    errors.update(_solver_errors(heat, 'heat'))
    details.update(T_rise_max_K=float(rise.max()), T_rise_discrete_K=rise_scale,
                   T_rise_closed_form_K=float(closed_form.max()), heat_closed_form_deviation=closed_deviation,
                   source_power=float(derived['source_power']), source_power_expected=expected_power)

    radius_min = 0.0 if kind == 'cartesian' else _SHELL_INNER
    potential_field = capacitor.field('electrostatics.V')
    potential = np.asarray(potential_field.values, dtype=np.float64)[:, 0]
    centres = np.asarray(potential_field.x_centers, dtype=np.float64)
    edges = cell_edges(centres, radius_min)
    permittivity = np.where(centres <= radius_min + _CAPACITOR_CORE, _EPS_CORE, _EPS_OUTER)
    charge_per_measure, expected_potential = _capacitor_chain(edges, permittivity, power)
    charge = charge_per_measure * measure
    inner = capacitor.result('electrostatics', 'inner')
    inner_charge = float(inner['charge'])
    outer_charge = float(capacitor.result('electrostatics', 'outer')['charge'])
    analytic = _capacitor_closed_form(kind, radius_min)
    analytic_deviation = abs(inner_charge / analytic - 1.0)
    errors.update(potential_profile=float(np.max(np.abs(potential - expected_potential))) / _VOLTAGE,
                  inner_charge=abs(inner_charge - charge) / charge, outer_charge=abs(outer_charge + charge) / charge,
                  charge_unit=0.0 if inner['units'][2] == charge_unit else 1.0,
                  charge_closed_form_excess=max(0.0, analytic_deviation - analytic_bound))
    errors.update(_solver_errors(capacitor, 'electrostatics'))
    details.update(charge=inner_charge, charge_discrete=charge, charge_closed_form=analytic,
                   charge_closed_form_deviation=analytic_deviation, charge_unit=inner['units'][2],
                   capacitor_cells=int(potential.size))
    error = worst_error(errors)   # NaN propagates
    return {'measured': inner_charge, 'expected': charge, 'error': error, 'tolerance': _TOLERANCE,
            'details': dict(details, errors=errors)}


# ---- solver_banded: the runtime on synthetic operators (CPU) ----
def _check_banded(_results_by_name):
    """BandedSolver on scipy CSRs vs numpy dense solves: an SPD tridiagonal with an identity (held) row, an SPD
    pentadiagonal given as unsummed duplicate entries, and the refusals (bandwidth above the limit, indefinite)."""
    try:
        import scipy.sparse as sparse
    except ImportError:
        return {'measured': None, 'expected': None, 'error': math.inf, 'tolerance': 1e-12,
                'details': {'message': 'scipy is not installed (the banded solver needs it)'}}
    from ..solvers.banded import BandedSolver
    rng = np.random.default_rng(2020)
    n = 200
    off = -rng.uniform(0.1, 1.0, n - 1)
    diagonal = np.zeros(n)
    diagonal[:-1] -= off
    diagonal[1:] -= off
    diagonal += rng.uniform(1e-3, 1e-2, n)   # anchored
    tridiagonal = sparse.diags([off, diagonal, off], [-1, 0, 1], format='lil')
    tridiagonal[17, :] = 0.0
    tridiagonal[:, 17] = 0.0
    tridiagonal[17, 17] = 1.0                 # a held cell: identity row, as DiffusionSystem's A_solve
    dense_tri = tridiagonal.toarray()
    second = -rng.uniform(0.05, 0.2, n - 2)
    pentadiagonal = sparse.diags([second, off, diagonal + 1.0, off, second], [-2, -1, 0, 1, 2], format='coo')
    duplicated = sparse.coo_matrix((np.concatenate((0.5 * pentadiagonal.data, 0.5 * pentadiagonal.data)),
                                    (np.concatenate((pentadiagonal.row, pentadiagonal.row)),
                                     np.concatenate((pentadiagonal.col, pentadiagonal.col)))), shape=(n, n))
    rhs = rng.standard_normal(n)
    errors, details = {}, {}
    for name, matrix, dense in (('tridiagonal', tridiagonal.tocsr(), dense_tri),
                                ('pentadiagonal', duplicated.tocsr(), pentadiagonal.toarray())):
        solver = BandedSolver(backend='cpu')
        solver.factorize(matrix)
        solution = np.zeros(n)
        solver.solve(rhs, solution)
        expected = np.linalg.solve(dense, rhs)
        errors[name] = float(np.max(np.abs(solution - expected))) / float(np.max(np.abs(expected)))
        details[f'{name}_bandwidth'] = solver.bandwidth
    errors['bandwidths'] = 0.0 if (details['tridiagonal_bandwidth'], details['pentadiagonal_bandwidth']) == (1, 2) \
        else 1.0
    for name, solver, matrix in (('refuses_wide', BandedSolver(backend='cpu', max_bandwidth=1), duplicated.tocsr()),
                                 ('refuses_indefinite', BandedSolver(backend='cpu'),
                                  sparse.diags([np.full(n - 1, -1.0), np.zeros(n), np.full(n - 1, -1.0)],
                                               [-1, 0, 1], format='csr'))):
        try:
            solver.factorize(matrix)
            errors[name] = 1.0
        except ValueError:
            errors[name] = 0.0
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1e-12,
            'details': dict(details, errors=errors)}


_DESCRIPTION = (
    '1D model (DomainSpec size_y null, coordinates {kind}: {shape}), two stationary runs on uniform {cell:g} um cells '
    'with the Automatic linear solver (banded). heat: uniform Q = {power:g} W/m^3 in 0 < r < b = {outer:g} mm, k = '
    '{k_core:g} W/(m K) inside r1 = {core:g} mm (a circle about the origin), {k_outer:g} outside, x_hi at {face:g} K '
    '(T0), x_lo {x_lo}: T_rise against the exact discrete chain (face flow Q V(r_f), drop F / g_f), the closed form '
    'sum Q / (2 (p + 1) k) (r_hi^2 - r_lo^2) within {heat_bound:g}, the integral of Q_ext (Q M b^(p+1) / (p + 1)) and '
    'the core volume (metric volumes, selection on the drawing x axis). capacitor: {voltage:g} V on x_lo (r = a = '
    '{shell:g} mm), 0 V on x_hi, eps_dc {eps_core:g} in the first {core_thickness:g} mm, {eps_outer:g} beyond: V and '
    'both plate charges against the discrete series chain (half cells on the plates), the unit {unit}, the closed '
    'form {closed} within {bound:g}. Both: the summary records a banded linear solver with a factorisation. error = '
    'the largest relative deviation (0 / 1 flags, excesses over the bounds), tolerance {tolerance:g}')
_SHAPES = {'cartesian': ('a slab, x from 0, per m^2: M = 1, p = 0', 'insulated (the symmetry plane)',
                         'eps0 / (L1 / eps1 + L2 / eps2) (exact)'),
           'polar': ('a cylinder, r from radius_min, per m of length: M = 2 pi, p = 1', 'the axis (no area)',
                     '2 pi eps0 / (ln(r1 / a) / eps1 + ln(b / r1) / eps2)'),
           'spherical': ('a sphere, geometry axisymmetric, whole shells: M = 4 pi, p = 2', 'the centre (no area)',
                         '4 pi eps0 / ((1 / a - 1 / r1) / eps1 + (1 / r1 - 1 / b) / eps2)')}


def _case(kind, label, cpu_variant=False):
    """line_<label> case; cpu_variant repeats it on the cpu backend (one kind suffices: the metric weights are
    backend-independent numpy, the variant re-runs the same assembly with the backend switched)."""
    shape, x_lo, closed = _SHAPES[kind]
    description = _DESCRIPTION.format(
        kind=kind, shape=shape, cell=_CELL * 1e6, power=_POWER_DENSITY, outer=_OUTER * 1e3, k_core=_K_CORE,
        core=_HEAT_CORE * 1e3, k_outer=_K_OUTER, face=_T_FACE, x_lo=x_lo, heat_bound=_HEAT_BOUND, voltage=_VOLTAGE,
        shell=0.0 if kind == 'cartesian' else _SHELL_INNER * 1e3, eps_core=_EPS_CORE,
        core_thickness=_CAPACITOR_CORE * 1e3, eps_outer=_EPS_OUTER, unit=_KINDS[kind][3], closed=closed,
        bound=_KINDS[kind][4], tolerance=_TOLERANCE)
    description += '; repeated on the cpu backend.' if cpu_variant else ' (the cpu backend: line_slab).'
    variants = (SelfTestVariant('cpu', tolerance=_TOLERANCE, backend='cpu'),) if cpu_variant else ()
    return SelfTestCase(name=f'line_{label}', label=f'1D {label}: heated body and capacitor', description=description,
                        specs=lambda: _specs(kind), check=lambda results_by_name: _check(results_by_name, kind),
                        variants=variants)


LINE_SELFTEST_CASES = (
    _case('cartesian', 'slab', cpu_variant=True),
    _case('polar', 'cylinder'),
    _case('spherical', 'sphere'),
    SelfTestCase(
        name='solver_banded', label='Banded solver: 1D operators vs dense solves',
        description=('solvers.banded.BandedSolver (the Automatic linear solver of 1D models) on the cpu backend: an SPD '
                     'tridiagonal scipy CSR with an identity (held-cell) row and an SPD pentadiagonal given as '
                     'duplicate COO entries (summed), each factorised once (banded Cholesky) and solved against '
                     'numpy.linalg.solve; bandwidths 1 / 2; a bandwidth above max_bandwidth and an indefinite operator '
                     'raise ValueError. error = the largest relative solution deviation or 0 / 1 flag, tolerance '
                     '1e-12.'),
        specs=lambda: {}, check=_check_banded),
)

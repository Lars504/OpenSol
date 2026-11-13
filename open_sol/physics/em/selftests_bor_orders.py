"""Self-tests of the body-of-revolution order lists and line-block factors (symmetry 2.3): CPU checks of
bor_line_block on the cpu backend (order family vs the assembled systems, shift-invert resonances vs SuperLU), a
model run of azimuthal_orders with and without a line_block solver node, the driven order sum of off-axis dipoles
(CPU numerics and a model run) and the line-block frequency_domain wavelength batch (model runs); physics imports inside
the checks only."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, norm_relative_error, skip_result, solver_role
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, SolverSpec,
                           StudySpec)
from .selftests import normalized_result
from .selftests_bor import (_BOX_R_CELLS, _BOX_Z_ROWS, _CAVITY_CELL, _CAVITY_CELLS, _CAVITY_TARGET, _DIPOLE_CELL,
                            _DIPOLE_CURRENT, _DIPOLE_PML, _DIPOLE_WAVELENGTH, _DIPOLE_WINDOW, _EIGEN_STUDY, _IDENTITY_SOURCE_ROW,
                            _IDENTITY_TOLERANCE, _PATTERN_TOLERANCE, _PHYSICS, _POWER_TOLERANCE, _RADIATION_THETA,
                            _RECYCLE_CELL, _RECYCLE_EPS, _TABLE, _TYPE, _analytic_cavity, _match,
                            _open_dielectric_assembly, _open_vacuum_grid, _trapezoid)
from .selftests_fdfd import cpu_specs

# ---- line-block solves of the order family A(m) = base + m K1 + m^2 K2 on the anisotropic identity grid of
# em_bor_order_sweep (cylindrical PML, complex symmetric): every order's solution vs the host SuperLU of its own
# assembled system. Orders 0, 1, 2 are the three axis classes, 5 a large shift.
_FAMILY_ORDERS = (0, 1, 2, 5)
_FAMILY_RESIDUAL = {'double': 1e-8, 'mixed': 1e-6}   # ||b - A(m) x|| / ||b|| with the independently assembled A(m)
_FAMILY_DIFFERENCE = 1e-6                            # ||x - x_SuperLU|| / ||x_SuperLU||

# ---- shift-invert resonances of the 1 um PEC cylinder (em_bor_pec_cylinder) on line-block factors vs SuperLU ones, on
# its own 40 nm grid (25 x 25 cells: both factor kinds solve the same discrete operator, no closed form is compared)
_EIGEN_CELL = 40e-9
_EIGEN_CELLS = (25, 25)
_EIGEN_ORDERS = (0, 1, 2)
_EIGEN_RESONANCES = 6
_EIGEN_TOLERANCE = 1e-9                              # relative omega (both exact for the same discrete operator)
# compared: |omega - omega_t| < this x omega_t (|k^2 - k_t^2| < k_t^2), the window plain shift-invert kept free of the
# static gradient cluster; solve_bor_resonances' squared Cayley map removes that cluster, so this is only a margin
_EIGEN_WINDOW = 0.9 * (math.sqrt(2.0) - 1.0)

# ---- model run: the cavity, azimuthal_orders [0, 1, 2], 40 nm cells (Bessel error ~8.5e-4 against 5e-3 and the 3e-3
# spurious band of _match: 50 nm would give ~1.3e-3); a second run with a line_block shifted node
_MODEL_CELL = 40e-9
_MODEL_ORDERS = [0, 1, 2]
_MODEL_RESONANCES = 3
_MODEL_TOLERANCE = 5e-3                              # relative k vs the Bessel closed form
_MODEL_SAME_TOLERANCE = 1e-8                         # line-block vs SuperLU run, relative omega
_LINE_BLOCK_NODE = 'device_line_block'

# ---- driven order sum: 3D point currents at phi = 0, rho_s = 8.5 cells (0.2125 um, k rho_s = 1.34) off the axis on
# the em_bor_radiation grid (vacuum, cpml), as their Fourier terms m = -4..4 (orders 0..4, each m > 0 with -m by
# A(-m) = P A(m) P). Exact: a z or y (= phi at phi = 0) dipole of moment p = I h radiates eta0 k^2 p^2 / (12 pi)
# anywhere in free space; the z dipole's order pair m carries the share (1 + (m > 0)) (3/4) int sin^3 theta J_m(k
# rho_s sin theta)^2 dtheta, phi average U_max sin^2 theta J_m^2 (1 + (m > 0)), and the summed far field is U_max
# sin^2 theta at phi = 0 and pi (displacing a dipole changes only the phase); the y dipole's is U_max there.
_OFFAXIS_CELL = 8
# truncation after m = 4: recorded shares m = 5 / 6 were 1.1e-6 / 1.3e-8 (z) and 1.9e-5 / 3e-7 (y) of the power, so
# power and shares move by ~2e-5 (tolerances 0.04, 0.01); the missing y amplitude ~4.4e-3 adds a cross term ~1e-2 of
# U_max to the summed pattern (+0.2 in pattern tolerance units, z +0.1)
_OFFAXIS_ORDERS = tuple(range(5))
_PARITY_ORDERS = (1, 2)               # their -m terms also solved directly on A(-m)
_PARITY_TOLERANCE = 1e-9              # mirrored vs directly solved -m fields, relative
_SHARE_TOLERANCE = 0.01               # per-order power share, absolute
_ODD_FIELD_TOLERANCE = 1e-6           # max |E_phi| / max |E_z| at phi = 0 of the z dipole (odd in phi: exactly 0)
_OFFAXIS_DOMAIN = (1.2e-6, 1.8e-6)    # = _DIPOLE_WINDOW cells of _DIPOLE_CELL
# closed flux box of the model runs (half height above / below the source row, side radius): the CPU box of
# em_bor_radiation, 12 cells from the PML
_OFFAXIS_BOX = (0.6e-6, 0.9e-6)
_OFFAXIS_FAR_FIELD_BOX = (0.95e-6, 0.65e-6)   # far-field box: r extent, half height about the source row

# ---- wavelength batch (solver_plan Part II stage 3): the axial z dipole (m = 0) in a smaller box of its own (the
# check compares line block with SuperLU only), frequency_domain at two wavelengths (already a multi-shift batch,
# below fdfd_line_block.MAX_BATCH_WAVELENGTHS = 8: one batch), SuperLU vs a line_block shifted node
_BATCH_WAVELENGTHS = (0.95e-6, 1.05e-6)
_BATCH_DOMAIN = (1.0e-6, 1.5e-6)
_BATCH_BOX = (0.4e-6, 0.7e-6)          # half height, side radius
_BATCH_TOLERANCES = {'double': 1e-8, 'mixed': 1e-5}   # box power, relative to the largest SuperLU value

# ---- cuDSS 'linear' back ends (driven): the wavelength batch model (2 wavelengths, m = 0: one plan) and the off-axis
# order sum at orders 0..3 (active sets m = 0 and |m| >= 1: two plans for four factors) per back end vs host SuperLU
_CUDSS_BACK_ENDS = ('cudss', 'cudss_mixed')
_CUDSS_NODE = 'sparse_direct_cudss'
_CUDSS_ORDERS = (0, 1, 2, 3)
_CUDSS_EXPECTED = {'batch': (len(_BATCH_WAVELENGTHS), 1), 'orders': (len(_CUDSS_ORDERS), 2)}   # (factors, plans)
_CUDSS_TOLERANCE = 1e-8   # box powers relative to the largest SuperLU value (measured <= 1.1e-14)


def _superlu_runs_on_cpu(specs):
    """specs with the 'superlu' run on the cpu backend; the line_block runs keep the default (the cupy line block is
    its own implementation)."""
    cpu_specs({name: spec for name, spec in specs.items() if name == 'superlu'})
    return specs


def _check_line_block_family(results_by_name):
    """Order-family line-block factors (cpu backend, double and mixed) vs SuperLU of each assembled system, plus the
    single-system driven factor."""
    from .body_of_revolution import point_current_rhs
    from .bor_line_block import driven_factorization, order_factorizations
    from ...solvers.line_block import LineBlockSolver
    from ...solvers.sparse_direct import factorize_host
    assembly = _open_dielectric_assembly(_RECYCLE_EPS)
    errors, details = {}, {}
    references = {}
    for order in _FAMILY_ORDERS:
        system = assembly.system(order)
        rhs = point_current_rhs(system, _RECYCLE_CELL, _IDENTITY_SOURCE_ROW, 'phi', _DIPOLE_CURRENT)
        references[order] = (system, rhs, factorize_host(system.operator).solve_host(rhs))
    for precision, residual_tolerance in _FAMILY_RESIDUAL.items():
        solver = LineBlockSolver(precision=precision, backend='cpu')
        for order, factorization in order_factorizations(solver, assembly, _FAMILY_ORDERS):
            system, rhs, reference = references[order]
            solution, relative, accepted = factorization.factor.solve(rhs, factorization.index)
            residual = float(np.linalg.norm(rhs - system.operator @ solution) / np.linalg.norm(rhs))
            key = f'{precision} m = {order}'
            errors[f'{key} residual'] = residual / residual_tolerance if accepted else math.inf
            errors[f'{key} difference'] = norm_relative_error(solution, reference) / _FAMILY_DIFFERENCE
            details[key] = {'line_block_residual': relative, 'residual': residual,
                            'difference': norm_relative_error(solution, reference), 'accepted': bool(accepted)}
        system, rhs, reference = references[_FAMILY_ORDERS[-1]]
        driven = driven_factorization(solver, system).solve_host(rhs)
        errors[f'{precision} driven factor difference'] = norm_relative_error(driven, reference) / _FAMILY_DIFFERENCE
        details[f'{precision} counters'] = solver.work_counters()
    worst = max(errors, key=errors.get)
    details.update(normalised_errors=errors, worst_item=worst, unknowns=references[_FAMILY_ORDERS[-1]][0].n_unknowns)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


def _cavity_assembly():
    """BORAssembly of the vacuum PEC cylinder of em_bor_pec_cylinder on _EIGEN_CELLS cells of _EIGEN_CELL at its
    target."""
    from .body_of_revolution import BORAssembly, BORMaterials
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='pec', y_hi='pec')
    shape = _EIGEN_CELLS
    widths = tuple(np.full(count, _EIGEN_CELL) for count in shape)
    materials = BORMaterials(eps=np.ones(shape, dtype=np.complex128), mu=np.ones(shape, dtype=np.complex128),
                             pec=np.zeros(shape, dtype=bool))
    return BORAssembly(widths, config, _EIGEN_CELL, materials, 2.0 * math.pi * C0 / _CAVITY_TARGET)


def _check_line_block_resonances(results_by_name):
    """Shift-invert resonances of the PEC cylinder on batched line-block factors (cpu, double) vs host SuperLU, the
    merged order-list EigenSolve (order tags, order-by-order ranking)."""
    from .body_of_revolution import merge_order_solves, solve_bor_resonances
    from .bor_line_block import order_factorizations
    from ...solvers.line_block import LineBlockSolver
    assembly = _cavity_assembly()
    solver = LineBlockSolver(precision='double', backend='cpu')
    line_solves = {order: solve_bor_resonances(assembly.system(order), _EIGEN_RESONANCES, factorization=factorization)
                   for order, factorization in order_factorizations(solver, assembly, _EIGEN_ORDERS)}
    errors, details, violations = {}, {}, []
    host_solves = []
    omega_target = assembly.omega

    def windowed(solve):
        return sorted((resonance.omega for resonance in solve.resonances
                       if abs(resonance.omega - omega_target) < _EIGEN_WINDOW * omega_target),
                      key=lambda omega: (omega.real, omega.imag))

    for order in _EIGEN_ORDERS:
        host = solve_bor_resonances(assembly.system(order), _EIGEN_RESONANCES)
        host_solves.append(host)
        line, reference = windowed(line_solves[order]), windowed(host)
        if not reference or len(line) != len(reference):
            violations.append(f'm = {order}: {len(line)} line-block vs {len(reference)} SuperLU resonances within '
                              f'{_EIGEN_WINDOW:.3g} omega_t of the target')
            continue
        errors[f'm = {order}'] = max((abs(a - b) / abs(b) for a, b in zip(line, reference)), default=0.0)
        details[f'm = {order}'] = {'k_line_block': float_list([value.real / C0 for value in line]),
                                   'factor_kind': line_solves[order].info['factor_kind']}
    merged = merge_order_solves(host_solves)
    tags = [resonance.azimuthal_order for resonance in merged.resonances]
    expected_tags = [order for order, solve in zip(_EIGEN_ORDERS, host_solves) for _ in solve.resonances]
    if tags != expected_tags or merged.info['azimuthal_orders'] != list(_EIGEN_ORDERS):
        violations.append(f'merged order tags {tags} != {expected_tags}')
    worst_item = max(errors, key=errors.get) if errors else None
    measured = errors[worst_item] if errors else math.inf
    details.update(errors=errors, worst_item=worst_item, violations=violations, counters=solver.work_counters(),
                   merged_summary=merged.summary())
    return {'measured': measured, 'expected': 0.0, 'error': math.inf if violations else measured,
            'tolerance': _EIGEN_TOLERANCE, 'details': details}


def _model_order_list_specs():
    """{'superlu', 'line_block'}: em_fdfd_bor PEC cylinder (1 x 1 um, 25 nm), azimuthal_orders [0, 1, 2],
    eigenfrequency; the second with a line_block (double) node in the shifted role."""
    settings = {'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec', 'azimuthal_orders': list(_MODEL_ORDERS),
                'wavelength': _CAVITY_TARGET}
    size = _CAVITY_CELLS[0] * _CAVITY_CELL
    specs = {}
    for run in ('superlu', 'line_block'):
        line_block = run == 'line_block'
        specs[run] = ModelSpec(
            title=f'em_fdfd_bor PEC cylinder, azimuthal order list ({run})',
            domain=DomainSpec(size_x=size, size_y=size),
            physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=dict(settings),
                                 features=[FeatureSpec(_TABLE, 'resonance_table')],
                                 solvers={'shifted': _LINE_BLOCK_NODE} if line_block else {})],
            solvers=[SolverSpec(_LINE_BLOCK_NODE, 'line_block', {'precision': 'double'})] if line_block else [],
            mesh=MeshSpec(mode='uniform', dx=_MODEL_CELL),
            study=StudySpec(type=_EIGEN_STUDY, settings={'n_resonances': _MODEL_RESONANCES}))
    return specs


def _check_model_order_list(results_by_name):
    """Per order of the list: table rows tagged with their order vs the Bessel closed form, one snapshot per row, and
    the line_block run equal to the SuperLU run."""
    size = _CAVITY_CELLS[0] * _CAVITY_CELL
    k_target = 2.0 * math.pi / _CAVITY_TARGET
    errors, violations, details = {}, [], {}
    tables = {}
    for run, results in results_by_name.items():
        table = results.result(_PHYSICS, _TABLE)
        tables[run] = table
        order_column = table['columns'].index('azimuthal_order')
        labels = results.summary().get('snapshot_labels') or []
        if len(labels) != len(table['rows']):
            violations.append(f'{run}: {len(labels)} snapshots for {len(table["rows"])} table rows')
        for order in _MODEL_ORDERS:
            rows = [row for row in table['rows'] if int(row[order_column]) == order]
            computed = [2.0 * math.pi * float(row[2]) / C0 for row in rows]
            if len(rows) != _MODEL_RESONANCES:
                violations.append(f'{run} m = {order}: {len(rows)} rows, expected {_MODEL_RESONANCES}')
            worst, spurious = _match(computed, _analytic_cavity(order, size, size), k_target,
                                     min(_MODEL_RESONANCES - 1, len(computed)) or 1)
            errors[f'{run} m = {order}'] = worst
            if spurious:
                violations.append(f'{run} m = {order}: {spurious} spurious resonance(s)')
            details[f'{run} m = {order}'] = float_list(computed)
        details[f'{run} labels'] = list(labels)
    pairs = list(zip(tables['superlu']['rows'], tables['line_block']['rows']))
    same = max((abs(float(a[2]) - float(b[2])) / abs(float(b[2])) for a, b in pairs), default=math.inf)
    if not same <= _MODEL_SAME_TOLERANCE:
        violations.append(f'line_block run differs from the SuperLU run by {same:.3g} relative frequency')
    worst = max(errors, key=errors.get)
    details.update(errors=errors, worst_item=worst, line_block_vs_superlu=same, violations=violations)
    return {'measured': errors[worst], 'expected': 0.0, 'error': math.inf if violations else errors[worst],
            'tolerance': _MODEL_TOLERANCE, 'details': details}


def _offaxis_radius():
    return (_OFFAXIS_CELL + 0.5) * _DIPOLE_CELL


def _dipole_constants():
    """(wavenumber [1/m], radiated power P [W], peak U_max [W/sr]) of the z / y dipole of moment I h at the grid
    wavelength."""
    wavenumber = 2.0 * math.pi / _DIPOLE_WAVELENGTH
    moment_squared = (_DIPOLE_CURRENT * _DIPOLE_CELL) ** 2
    return (wavenumber, ETA0 * wavenumber ** 2 * moment_squared / (12.0 * math.pi),
            ETA0 * wavenumber ** 2 * moment_squared / (32.0 * math.pi ** 2))


def _order_shares(theta):
    """Analytic power share per order pair of _OFFAXIS_ORDERS of the off-axis z dipole: (1 + (m > 0)) (3/4) int
    sin^3 theta J_m(k rho_s sin theta)^2 dtheta."""
    import scipy.special
    wavenumber = _dipole_constants()[0]
    argument = wavenumber * _offaxis_radius() * np.sin(theta)
    return np.array([(1.0 + (order > 0)) * 0.75 * _trapezoid(np.sin(theta) ** 3
                                                              * scipy.special.jv(order, argument) ** 2, theta)
                     for order in _OFFAXIS_ORDERS])


def _open_vacuum_assembly():
    """BORAssembly of the em_bor_radiation grid (vacuum, cpml on r_hi and both z faces) at its wavelength."""
    from .body_of_revolution import BORAssembly
    config, widths, window, materials, omega = _open_vacuum_grid()
    return BORAssembly(widths, config, _DIPOLE_CELL, materials, omega, window=window)


def _check_offaxis_dipole(results_by_name):
    """Order sums of an off-axis z and y (phi) dipole: flux identity, power, order shares, summed and per-order far
    field vs the closed forms, the mirrored -m terms vs direct solves on A(-m)."""
    import scipy.special
    from .body_of_revolution import (box_flux, far_field_samples, order_pair_fields, order_sum_far_field,
                                     point_current_rhs, source_power, yee_fields)
    from ...solvers.sparse_direct import factorize_host
    assembly = _open_vacuum_assembly()
    omega = assembly.omega
    wavenumber, expected_power, peak = _dipole_constants()
    source_row = _DIPOLE_PML + _DIPOLE_WINDOW[1] // 2
    z_rows = (_DIPOLE_PML + _BOX_Z_ROWS[0], _DIPOLE_PML + _BOX_Z_ROWS[1])
    radius = _offaxis_radius()
    theta = np.linspace(0.0, math.pi, _RADIATION_THETA)
    # y dipole of moment I h at phi = 0: per order the ring current I h / (2 pi rho_s) (BORPointCurrent.term_rhs)
    currents = {'z': ('z', _DIPOLE_CURRENT), 'y': ('phi', _DIPOLE_CURRENT * _DIPOLE_CELL / (2.0 * math.pi * radius))}
    flux = {name: [] for name in currents}        # per order pair: box flux [W]
    delivered = {name: [] for name in currents}   # per order pair: source power [W]
    samples = {name: [] for name in currents}     # [(signed order, far_field_samples)]
    parity = {}
    for order in _OFFAXIS_ORDERS:
        system = assembly.system(order)
        factor = factorize_host(system.operator)
        rhs = {name: point_current_rhs(system, _OFFAXIS_CELL, source_row, component, current)
               for name, (component, current) in currents.items()}
        for name, vector in rhs.items():
            solutions = []

            def solve(right_hand_side):
                solutions.append(factor.solve_host(right_hand_side))
                return solutions[-1]

            terms = order_pair_fields(system, vector, solve)
            flux[name].append(sum(box_flux(system, fields, _BOX_R_CELLS, z_rows) for _, fields in terms))
            # one solve: P b = +-b, so the -m term delivers what +m does
            delivered[name].append(len(terms) * source_power(system, solutions[0], vector))
            samples[name].extend((signed, far_field_samples(system, fields, _BOX_R_CELLS, z_rows))
                                 for signed, fields in terms)
        if order in _PARITY_ORDERS:   # a mixed z + phi rhs takes the separate P b solve
            mixed = rhs['z'] + rhs['y']
            mirrored = dict(order_pair_fields(system, mixed, factor.solve_host))[-order]
            partner = assembly.system(-order)
            direct = yee_fields(partner, factorize_host(partner.operator).solve_host(mixed))
            parity[f'm = {-order}'] = max(norm_relative_error(mirrored[key], direct[key]) for key in direct)
            if order == 1:   # an axis-cell 'r' element: its moved axis half -b / m flips with m, b(-1) = P b(+1)
                axis_rhs = point_current_rhs(system, 0, source_row, 'r', _DIPOLE_CURRENT)
                axis_minus = point_current_rhs(system, 0, source_row, 'r', _DIPOLE_CURRENT, order=-1)
                mirrored = dict(order_pair_fields(system, axis_rhs, factor.solve_host, minus_rhs=axis_minus))[-1]
                direct = yee_fields(partner, factorize_host(partner.operator).solve_host(
                    point_current_rhs(partner, 0, source_row, 'r', _DIPOLE_CURRENT)))
                parity['m = -1, axis r'] = max(norm_relative_error(mirrored[key], direct[key]) for key in direct)
    errors, details = {}, {'radius_m': radius, 'k_rho': wavenumber * radius, 'analytic_power_W': expected_power}
    for name in currents:
        total = float(sum(flux[name]))
        errors[f'{name}: identity'] = abs(total / sum(delivered[name]) - 1.0) / _IDENTITY_TOLERANCE
        errors[f'{name}: power'] = abs(total / expected_power - 1.0) / _POWER_TOLERANCE
        details[f'{name} box flux per order pair (W)'] = float_list(flux[name])
    shares = np.array(flux['z']) / sum(flux['z'])
    expected_shares = _order_shares(theta)
    errors['z: order shares'] = float(np.max(np.abs(shares - expected_shares))) / _SHARE_TOLERANCE
    details.update(order_shares=float_list(shares), order_shares_expected=float_list(expected_shares))
    sin_squared = np.sin(theta) ** 2

    def pattern_error(computed, expected):   # max deviation / U_max, normalised by the pattern tolerance
        return float(np.max(np.abs(computed - expected))) / peak / _PATTERN_TOLERANCE

    order_means = {}   # per dipole: the phi-free order means of its last order_sum_far_field call
    for name, expected_pattern in (('z', peak * sin_squared), ('y', peak)):
        for side, phi in (('phi = 0', 0.0), ('phi = pi', math.pi)):
            pattern, _, _, order_means[name] = order_sum_far_field(samples[name], omega, theta, phi)
            errors[f'{name}: pattern {side}'] = pattern_error(pattern, expected_pattern)
    argument = wavenumber * radius * np.sin(theta)
    errors['z: order patterns'] = max(
        pattern_error(order_means['z'][order], (1.0 + (order > 0)) * peak * sin_squared
                      * scipy.special.jv(order, argument) ** 2) for order in _OFFAXIS_ORDERS)
    errors.update({f'parity {key}': value / _PARITY_TOLERANCE for key, value in parity.items()})
    worst = max(errors, key=errors.get)
    details.update(parity_differences=parity, normalised_errors=errors, worst_item=worst,
                   tolerances={'identity': _IDENTITY_TOLERANCE, 'power': _POWER_TOLERANCE,
                               'shares': _SHARE_TOLERANCE, 'pattern': _PATTERN_TOLERANCE, 'parity': _PARITY_TOLERANCE})
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


def _dipole_box_model(x_source, domain_height, box):
    """(geometry, features) of a z point current at r = x_source on the middle row of a domain_height tall domain (the
    CPU grid's source row) with flux lines closing a box of box = (half height, side radius) (the
    analytic_bor_dipole_radiation layout: closed pairing)."""
    cell = _DIPOLE_CELL
    z_source = domain_height / 2.0 + cell / 2.0   # a cell centre
    box_half, side_radius = box
    half_height = box_half + cell / 2.0
    top, bottom = z_source + box_half, z_source - box_half - cell
    geometry = [
        GeometrySpec('source_point', 'point', {'x': x_source, 'y': z_source}),
        GeometrySpec('top_line', 'line', {'x0': 0.0, 'y0': top, 'x1': side_radius - cell / 4.0, 'y1': top}),
        GeometrySpec('bottom_line', 'line', {'x0': 0.0, 'y0': bottom, 'x1': side_radius - cell / 4.0, 'y1': bottom}),
        GeometrySpec('side_line', 'line', {'x0': side_radius - cell / 2.0, 'y0': z_source - half_height + cell / 4.0,
                                           'x1': side_radius - cell / 2.0, 'y1': z_source + half_height - cell / 4.0})]
    features = [FeatureSpec('dipole', 'point_source', selection=['source_point'],
                            params={'component': 'z', 'current': _DIPOLE_CURRENT, 'phase': 0.0}),
                *(FeatureSpec(name, 'flux_monitor', selection=[f'{name}_line']) for name in ('top', 'bottom', 'side'))]
    return geometry, features


def _model_offaxis_specs():
    """{'offaxis': em_fdfd_bor stationary driven order sum (azimuthal_orders _OFFAXIS_ORDERS) of a z point current 8.5
    cells off the axis on the em_bor_radiation grid, flux lines closing a box and a far-field box}."""
    cell = _DIPOLE_CELL
    z_source = _OFFAXIS_DOMAIN[1] / 2.0 + cell / 2.0
    geometry, features = _dipole_box_model(_offaxis_radius(), _OFFAXIS_DOMAIN[1], _OFFAXIS_BOX)
    far_radius, far_half = _OFFAXIS_FAR_FIELD_BOX
    geometry.append(GeometrySpec('far_field_box', 'rectangle', {'x0': 0.0, 'y0': z_source - far_half,
                                                                'x1': far_radius, 'y1': z_source + far_half}))
    features.append(FeatureSpec('far', 'far_field', selection=['far_field_box'], params={'n_angles': 360}))
    settings = {'wavelength': _DIPOLE_WAVELENGTH, 'azimuthal_orders': list(_OFFAXIS_ORDERS), 'bc_x_hi': 'cpml',
                'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml'}
    return {'offaxis': ModelSpec(
        title='em_fdfd_bor off-axis dipole, driven order sum',
        domain=DomainSpec(size_x=_OFFAXIS_DOMAIN[0], size_y=_OFFAXIS_DOMAIN[1]),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='stationary', stored_fields=[f'{_PHYSICS}.ephi_abs', f'{_PHYSICS}.ez_abs']))}


def _check_model_offaxis(results_by_name):
    """Box-face powers (total and per order) vs the closed forms, the summed far field over both meridional halves and
    its phi average vs U_max cos^2(angle), E_phi = 0 at phi = 0, the order-list extras."""
    results = results_by_name['offaxis']
    _, expected_power, peak = _dipole_constants()
    faces = {name: results.result(_PHYSICS, name) for name in ('top', 'bottom', 'side')}
    signs = {'top': 1.0, 'bottom': -1.0, 'side': 1.0}
    violations = [f'{name}: azimuthal_orders {face.get("azimuthal_orders")}' for name, face in faces.items()
                  if list(face.get('azimuthal_orders') or []) != list(_OFFAXIS_ORDERS)]
    if violations:
        return {'measured': math.inf, 'expected': 0.0, 'error': math.inf, 'tolerance': 1.0,
                'details': {'violations': violations}}
    total = sum(sign * float(faces[name]['power'][0]) for name, sign in signs.items())
    per_order = sum(sign * np.asarray(faces[name]['power_orders'], dtype=np.float64)[0] for name, sign in signs.items())
    expected_shares = _order_shares(np.linspace(0.0, math.pi, _RADIATION_THETA))
    far = results.result(_PHYSICS, 'far')
    angles = np.asarray(far['angles'], dtype=np.float64)
    expected_pattern = peak * np.cos(angles) ** 2
    pattern = np.asarray(far['power'], dtype=np.float64)[0]
    azimuthal_mean = np.asarray(far['power_azimuthal_mean'], dtype=np.float64)[0]
    ephi = np.asarray(results.field(f'{_PHYSICS}.ephi_abs').values, dtype=np.float64)
    ez = np.asarray(results.field(f'{_PHYSICS}.ez_abs').values, dtype=np.float64)
    odd_ratio = float(np.max(ephi) / np.max(ez))
    errors = {'power': abs(total / expected_power - 1.0) / _POWER_TOLERANCE,
              'order shares': float(np.max(np.abs(per_order / total - expected_shares))) / _SHARE_TOLERANCE,
              'pattern': float(np.max(np.abs(pattern - expected_pattern))) / peak / _PATTERN_TOLERANCE,
              'azimuthal mean': float(np.max(np.abs(azimuthal_mean - expected_pattern))) / peak / _PATTERN_TOLERANCE,
              'E_phi at phi = 0': odd_ratio / _ODD_FIELD_TOLERANCE}
    if np.shape(far['power_orders']) != (1, len(_OFFAXIS_ORDERS), angles.size):
        errors['power_orders shape'] = math.inf
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0,
            'details': {'power_W': total, 'analytic_power_W': expected_power,
                        'order_shares': float_list(per_order / total), 'order_shares_expected': float_list(
                            expected_shares), 'pattern_peak_W_per_sr': float(np.max(pattern)),
                        'analytic_peak_W_per_sr': peak, 'ephi_over_ez': odd_ratio, 'normalised_errors': errors,
                        'worst_item': worst}}


def _model_wavelength_batch_specs():
    """{'superlu', 'line_block_double', 'line_block_mixed'}: em_fdfd_bor frequency_domain (_BATCH_WAVELENGTHS, m = 0)
    of a z point current in the axis cell with the closed flux box; the line_block runs with that node as shifted."""
    settings = {'bc_x_hi': 'cpml', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml'}
    specs = {}
    for run in ('superlu', *(f'line_block_{precision}' for precision in _BATCH_TOLERANCES)):
        line_block = run != 'superlu'
        geometry, features = _dipole_box_model(_DIPOLE_CELL / 2.0, _BATCH_DOMAIN[1], _BATCH_BOX)
        specs[run] = ModelSpec(
            title=f'em_fdfd_bor axial dipole, wavelength batch ({run})',
            domain=DomainSpec(size_x=_BATCH_DOMAIN[0], size_y=_BATCH_DOMAIN[1]),
            geometry=geometry,
            physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=dict(settings), features=features,
                                 solvers={'shifted': _LINE_BLOCK_NODE} if line_block else {})],
            solvers=([SolverSpec(_LINE_BLOCK_NODE, 'line_block', {'precision': run.rsplit('_', 1)[1]})]
                     if line_block else []),
            mesh=MeshSpec(mode='uniform', dx=_DIPOLE_CELL),
            study=StudySpec(type='frequency_domain', settings={'wavelengths': list(_BATCH_WAVELENGTHS)}))
    return specs


def _box_powers(results):
    """float64 (n_wavelengths,) top - bottom + side power [W] of a wavelength batch run."""
    signs = {'top': 1.0, 'bottom': -1.0, 'side': 1.0}
    return sum(sign * np.asarray(results.result(_PHYSICS, name)['power'], dtype=np.float64)
               for name, sign in signs.items())


def _check_model_wavelength_batch(results_by_name):
    """Per precision max |P_line_block - P_SuperLU| / max |P_SuperLU| of the box power over the wavelengths in tolerance
    units, and the shifted record: type line_block, batched_operators = the wavelength count, rejected_factor_solves 0."""
    reference = _box_powers(results_by_name['superlu'])
    scale = float(np.max(np.abs(reference))) if reference.size == len(_BATCH_WAVELENGTHS) else math.nan
    errors, details = {}, {'superlu_power_W': float_list(reference)}
    for precision, tolerance in _BATCH_TOLERANCES.items():
        results = results_by_name[f'line_block_{precision}']
        powers = _box_powers(results)
        deviation = float(np.max(np.abs(powers - reference))) / scale if powers.shape == reference.shape else math.inf
        shifted = solver_role(results, _PHYSICS, 'shifted')
        counters = shifted.get('counters') or {}
        good = (shifted.get('type') == 'line_block'
                and float(counters.get('batched_operators', -1)) == len(_BATCH_WAVELENGTHS)
                and float(counters.get('rejected_factor_solves', math.inf)) == 0.0)
        errors[f'{precision} power'] = deviation / tolerance
        errors[f'{precision} record'] = 0.0 if good else math.inf
        details[precision] = {'power_W': float_list(powers), 'deviation': deviation, 'counters': dict(counters)}
    return normalized_result(errors, details, normalised_errors=errors)


def _cudss_driven_specs():
    """'<model>_sparse_direct' (a scipy_splu node: Automatic may resolve to cuDSS on the gpu) and '<model>_<back end>'
    (a sparse_direct node as 'linear') runs of the wavelength batch model and the off-axis order sum at
    _CUDSS_ORDERS; {} without nvmath-python."""
    from ...solvers.cudss_factor import cudss_available
    if not cudss_available():
        return {}
    specs = {}
    for back_end in ('sparse_direct',) + _CUDSS_BACK_ENDS:
        models = {'batch': _model_wavelength_batch_specs()['superlu'], 'orders': _model_offaxis_specs()['offaxis']}
        models['orders'].physics_by_name(_PHYSICS).settings['azimuthal_orders'] = list(_CUDSS_ORDERS)
        node_back_end = 'scipy_splu' if back_end == 'sparse_direct' else back_end   # the reference: host SuperLU
        for model, spec in models.items():
            spec.solvers.append(SolverSpec(_CUDSS_NODE, 'sparse_direct', {'back_end': node_back_end}))
            spec.physics_by_name(_PHYSICS).solvers['linear'] = _CUDSS_NODE
            specs[f'{model}_{back_end}'] = spec
    return specs


def _cudss_compared_powers(results, model):
    """float64 box powers a run is compared by: per wavelength (batch), total and per order (order sum)."""
    if model == 'batch':
        return _box_powers(results)
    signs = {'top': 1.0, 'bottom': -1.0, 'side': 1.0}
    faces = {name: results.result(_PHYSICS, name) for name in signs}
    total = sum(sign * float(faces[name]['power'][0]) for name, sign in signs.items())
    per_order = sum(sign * np.asarray(faces[name]['power_orders'], dtype=np.float64)[0] for name, sign in signs.items())
    return np.concatenate(([total], per_order))


def _check_cudss_driven(results_by_name):
    """Per model and back end: max |P - P_SuperLU| / max |P_SuperLU| in tolerance units, and the linear record: that
    back end, _CUDSS_EXPECTED factorisations (all LDL^T) on its plans (+ one per cudss_mixed double fallback)."""
    if not results_by_name:
        return skip_result('nvmath-python (cuDSS) is not installed')
    errors, details = {}, {}
    for model, (factors, plans) in _CUDSS_EXPECTED.items():
        reference = _cudss_compared_powers(results_by_name[f'{model}_sparse_direct'], model)
        scale = float(np.max(np.abs(reference)))
        details[f'{model}_superlu_W'] = float_list(reference)
        for back_end in _CUDSS_BACK_ENDS:
            results = results_by_name[f'{model}_{back_end}']
            powers = _cudss_compared_powers(results, model)
            deviation = (float(np.max(np.abs(powers - reference))) / scale if powers.shape == reference.shape
                         else math.inf)
            counters = solver_role(results, _PHYSICS).get('counters') or {}
            expected = {'back_end': back_end, 'factorizations': factors, 'cudss_symmetric_factors': factors,
                        'cudss_plans': plans + int(counters.get('cudss_double_fallbacks') or 0)}
            record = {name: counters.get(name) for name in expected}
            errors[f'{model} {back_end}'] = deviation / _CUDSS_TOLERANCE
            errors[f'{model} {back_end} record'] = 0.0 if record == expected else math.inf
            details[f'{model}_{back_end}'] = {'deviation': deviation, 'power_W': float_list(powers),
                                              'counters': dict(counters)}
    return normalized_result(errors, details, normalised_errors=errors)


BOR_ORDER_SELFTEST_CASES = (
    SelfTestCase(
        name='em_bor_line_block_family',
        label='EM body of revolution: line-block order family',
        description=(
            'CPU check of physics/em/bor_line_block and the matrix terms / kept factors of solvers/line_block (cpu '
            'backend, no model run) on the small em_bor_order_sweep grid (anisotropic dielectric block, cylindrical '
            'PML; every item is algebraic, so any grid with nz > nr will do; not yet confirmed by a run on it): '
            f'orders {", ".join(map(str, _FAMILY_ORDERS))} (the three axis classes and a large shift) as shifts of A(m) = (K0 - k0^2 M) + m K1 + m^2 K2 (one '
            'set_operator per axis class, one batch factor), a ring current solved per order. Normalised errors '
            '(passes at <= 1): the residual against the independently assembled BORSystem operator of that order '
            f'(double {_FAMILY_RESIDUAL["double"]:g}, mixed {_FAMILY_RESIDUAL["mixed"]:g}; a rejected solve fails), '
            f'the solution vs host SuperLU ({_FAMILY_DIFFERENCE:g}) and the single-system driven factor vs SuperLU. '
            'Pins the line layout (block-tridiagonal in z rows or r columns), the per-shift coupling blocks and the '
            'refinement with matrix terms.'),
        specs=lambda: {},
        check=_check_line_block_family,
        max_wall_time=180.0),
    SelfTestCase(
        name='em_bor_line_block_resonances',
        label='EM body of revolution: line-block shift-invert',
        description=(
            'CPU check (no model run): the 1 um PEC cylinder of em_bor_pec_cylinder on its own '
            f'{_EIGEN_CELL * 1e9:g} nm grid ({_EIGEN_CELLS[0]} x {_EIGEN_CELLS[1]} cells: both factor kinds solve the '
            'same discrete operator; not yet confirmed by a run on it), orders '
            f'{", ".join(map(str, _EIGEN_ORDERS))}, {_EIGEN_RESONANCES} resonances each by shift-invert ARPACK on '
            'batched line-block factors (cpu backend, double, bor_line_block.order_factorizations) vs on host SuperLU '
            f'factors: within |omega - omega_t| < {_EIGEN_WINDOW:.3g} omega_t (nearer than the static gradient cluster, '
            'so ARPACK finds all of them on either factor) the same count per order and equal omega within '
            f'{_EIGEN_TOLERANCE:g} relative (the same discrete operator), and merge_order_solves keeping order-by-order ranking with each resonance tagged by its '
            'order. Pins the kept-factor API and the order-list merge.'),
        specs=lambda: {},
        check=_check_line_block_resonances,
        max_wall_time=180.0),
    SelfTestCase(
        name='em_bor_order_list_model',
        label='EM body of revolution: azimuthal order list model run',
        description=(
            f'Model runs of em_fdfd_bor (eigenfrequency, azimuthal_orders {_MODEL_ORDERS}, {_MODEL_RESONANCES} '
            f'resonances per order) of the PEC cylinder on a {_MODEL_CELL * 1e9:g} nm mesh (expected Bessel error '
            '~8.5e-4; not yet confirmed by a run at this mesh), once Automatic (host SuperLU, cpu backend) and once '
            'with a line_block (double) node in the shifted role (default backend): per order the table rows tagged '
            f'azimuthal_order vs the Bessel closed form within {_MODEL_TOLERANCE:g} relative k (no spurious '
            'resonance), one snapshot per row, and the line_block run equal to the SuperLU run within '
            f'{_MODEL_SAME_TOLERANCE:g} relative frequency. Pins the model-level order list (one assembly, one '
            'study run), the resonance table column and the shifted role.'),
        specs=lambda: _superlu_runs_on_cpu(_model_order_list_specs()),
        check=_check_model_order_list),
    SelfTestCase(
        name='em_bor_offaxis_dipole',
        label='EM body of revolution: off-axis dipoles as an order sum',
        description=(
            'CPU check of the driven order sum of physics/em/body_of_revolution (no model run) on the '
            'em_bor_radiation grid (vacuum, cpml; the 1.2 x 1.8 um window and orders 0..4, expected worst item ~0.35 '
            'against 0.13 with orders 0..6 on 2 x 3 um: not yet confirmed by a run): a z and a y (phi at phi = 0) point dipole of moment I h at '
            f'rho_s = {_offaxis_radius() * 1e6:g} um off the axis, orders 0..{_OFFAXIS_ORDERS[-1]} (each m > 0 with '
            '-m from one factor, order_pair_fields), box flux and far field per order. Normalised errors (passes at '
            f'<= 1): summed box flux vs summed source power ({_IDENTITY_TOLERANCE:g}, exact), vs eta0 k^2 (I h)^2 / '
            f'(12 pi) for both dipoles ({_POWER_TOLERANCE:g}), the z dipole share per order pair vs (1 + (m > 0)) '
            f'(3/4) int sin^3 theta J_m(k rho_s sin theta)^2 dtheta ({_SHARE_TOLERANCE:g} absolute), the summed far '
            'field (order_sum_far_field) at phi = 0 and pi vs U_max sin^2 theta (z) and U_max (y) and the per-order '
            f'phi average vs U_max sin^2 theta J_m^2 (1 + (m > 0)) (max deviation / U_max, {_PATTERN_TOLERANCE:g}), '
            f'and for m = {", ".join(str(-order) for order in _PARITY_ORDERS)} the mirrored terms of a mixed z + phi '
            f'source vs direct solves on A(-m) ({_PARITY_TOLERANCE:g} relative), at m = -1 also an axis-cell r '
            "element (its moved axis half flips with m: order_pair_fields minus_rhs; not yet confirmed by a run). "
            'Pins A(-m) = P A(m) P, the 3D point '
            'element normalisation per order, the orthogonal power sum and the phased far-field sum. The sum stops at '
            'm = 4: the recorded m = 5 share of the y dipole, 1.9e-5, leaves a ~1e-2 U_max cross term in its pattern.'),
        specs=lambda: {},
        check=_check_offaxis_dipole,
        max_wall_time=240.0),
    SelfTestCase(
        name='em_bor_offaxis_dipole_model',
        label='EM body of revolution: off-axis dipole order-sum model run',
        description=(
            'Model run of em_fdfd_bor (stationary, azimuthal_orders 0..'
            f'{_OFFAXIS_ORDERS[-1]}: the driven order sum) of a z point_source {_offaxis_radius() * 1e6:g} um off the '
            'axis in vacuum on the em_bor_radiation grid (1.2 x 1.8 um, cpu backend; not yet confirmed by a run '
            f'with this grid and orders), flux_monitor lines closing a box ({_OFFAXIS_BOX[1] * 1e6:g} um side, '
            f'+-{_OFFAXIS_BOX[0] * 1e6:g} um: the analytic_bor_dipole_radiation pairing, the CPU box of '
            'em_bor_offaxis_dipole) and a far_field box. Normalised errors (passes at <= 1): top - '
            f'bottom + side power vs eta0 k^2 (I h)^2 / (12 pi) ({_POWER_TOLERANCE:g}), the per-order shares from '
            f'power_orders vs the Bessel integrals of em_bor_offaxis_dipole ({_SHARE_TOLERANCE:g}), the far-field '
            'power over the full 360 degrees (phi = 0 and phi = pi halves) and power_azimuthal_mean vs U_max '
            f'cos^2(angle) ({_PATTERN_TOLERANCE:g} of U_max), max |E_phi| / max |E_z| of the shown phi = 0 field '
            f'(odd in phi, {_ODD_FIELD_TOLERANCE:g}), the azimuthal_orders / power_orders extras. Pins the model '
            'layer of the order sum (create in a driven study, BORPhysics._solve_order_sum, record_order, results).'),
        specs=lambda: cpu_specs(_model_offaxis_specs()),
        check=_check_model_offaxis),
    SelfTestCase(
        name='em_bor_line_block_wavelengths',
        label='EM body of revolution: line-block wavelength batches',
        description=(
            'solver_plan Part II stage 3: model runs of em_fdfd_bor (frequency_domain at '
            f'{", ".join(f"{wavelength * 1e6:g}" for wavelength in _BATCH_WAVELENGTHS)} um, m = 0) of a z '
            f'point_source in the axis cell of a {_BATCH_DOMAIN[0] * 1e6:g} x {_BATCH_DOMAIN[1] * 1e6:g} um domain '
            f'with a closed flux box ({_BATCH_BOX[1] * 1e6:g} um side, +-{_BATCH_BOX[0] * 1e6:g} um; the check compares '
            'line block with SuperLU only), once Automatic (host SuperLU, cpu backend) and with a line_block shifted '
            'node (default backend) in double and mixed precision: the line block '
            'factorises the planned wavelengths as one batch of whole operators (factorize_operators; A is not '
            'affine in omega) and solves each wavelength from it. Normalised errors (passes at <= 1): max |P - '
            'P_SuperLU| / max |P_SuperLU| of the top - bottom + side power over the wavelengths '
            f'(double {_BATCH_TOLERANCES["double"]:g}, mixed {_BATCH_TOLERANCES["mixed"]:g}), and the run summary '
            'must record the shifted role as line_block with batched_operators = the wavelength count and '
            'rejected_factor_solves 0. Not yet confirmed by a run.'),
        specs=lambda: _superlu_runs_on_cpu(_model_wavelength_batch_specs()),
        check=_check_model_wavelength_batch),
    SelfTestCase(
        name='em_bor_cudss_driven',
        label="EM body of revolution: cuDSS 'linear' back ends vs SuperLU",
        description=(
            'Driven em_fdfd_bor model runs (gpu) by the Automatic sparse_direct (host SuperLU) and by a sparse_direct '
            "node with back_end cudss and cudss_mixed as the 'linear' solver (6 runs): the em_bor_line_block_"
            f'wavelengths model (frequency_domain at {len(_BATCH_WAVELENGTHS)} wavelengths, m = 0) and the '
            f'em_bor_offaxis_dipole_model order sum at azimuthal_orders {list(_CUDSS_ORDERS)}. cuDSS factors the '
            'complex symmetric operator as an LDL^T and refactorises the kept factor for the next wavelength / order '
            '(plan reused while the pattern holds: m = 0 and every |m| >= 1 have one pattern each). Per model and back '
            'end '
            'max |P - P_SuperLU| / max |P_SuperLU| of the box powers (per wavelength; total and per order) below '
            f'{_CUDSS_TOLERANCE:g}, and the linear record: that back end, factorizations = cudss_symmetric_factors = '
            f"{_CUDSS_EXPECTED['batch'][0]} / {_CUDSS_EXPECTED['orders'][0]} on {_CUDSS_EXPECTED['batch'][1]} / "
            f"{_CUDSS_EXPECTED['orders'][1]} plan(s) (+ 1 per cudss_mixed double fallback). measured = error = the "
            'worst item in tolerance units, tolerance 1; passes as skipped without nvmath-python. Run on an RTX 3050 '
            '(8.8 s): deviations 2e-15 - 1e-14 for both back ends, no double fallback, cudss_mixed unrefined '
            'residual 4e-6 (batch) / 5e-5 (orders).'),
        specs=_cudss_driven_specs,
        check=_check_cudss_driven),
)

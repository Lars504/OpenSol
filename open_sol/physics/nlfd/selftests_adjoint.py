"""Self-tests of the em_fdfd_nonlinear adjoint (nonlinear_optics_plan stage 5 (b)): eps_sensitivity against a central
difference of the objective (undepleted ez, hz and mixed-polarisation SHG; pump-depleted SHG and SPM; nonlinear sheets
and free carriers), a short SHG optimization run and the shg_metasurface_optimization template; no physics imports."""
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, cell_centre, normalized_result, single_value, vertical_line
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)

# Strip: y-periodic, 4 rows, CPML along x, uniform 10 nm mesh (window cell k spans [k h, (k + 1) h]), eps_r 2.25
# everywhere; pump channel p (1.55 um) from a line source in cell 100, SH channel sh = 2 p, a chi2 slab on cells
# [200, 265) over the full height, undepleted. Objectives on the line in cell 365 (beyond the slab): J = flux of sh
# (weight 1, ~1e-3 W/m) + 1e-9 x flux of p (~8e-6 W/m: it exercises the pump's own objective source on top of what
# the SH sends back; its first-order change through the lossless block is small). The perturbed block
# = cells [220, 224) over the full height, inside the chi2 slab: the gradient there needs every cascade term (pump
# operator, the nonlinear rhs, hz the W of the stored face fields and of the nonlinear face currents, the C / C*
# transposes). Polarisation sets: ez -> ez (3m z-cut d33), hz -> hz ('general' d22: P_y = eps0 d22 E_y^2) and hz -> ez
# (-43m d14 rotated 45 deg about z: P_z = eps0 d14 (E_y^2 - E_x^2)).
_PHYSICS = 'nl'
_CELL = 10e-9                       # m
_ROWS = 4
_SOURCE_CELL = 100
_SLAB = (200, 265)                  # chi2 slab cells
_OBJECTIVE_CELL = 365
_N_X = 465
_BLOCK = (220, 224)                 # perturbed cells
_WAVELENGTH = 1.55e-6               # m
_EPS_R = 2.25
_AMPLITUDE = 1e7                    # V/m
_D = 25e-12                         # m/V
_PUMP_WEIGHT = 1e-9
_STEP = 0.01                        # eps_r step of the central difference (2.25 +- 0.01)
_TOLERANCE = 2e-3                   # relative, adjoint vs central difference
_GRADIENT_VALUE = 'block_gradient'
_CASES = {'ez': ('ez', 'ez', {'crystal_class': '3m', 'd33': _D, 'orientation': 'z_cut'}),
          'hz': ('hz', 'hz', {'crystal_class': 'general', 'd22': _D, 'orientation': 'z_cut'}),
          'mixed': ('hz', 'ez', {'crystal_class': '-43m', 'd14': _D, 'orientation': 'euler', 'euler_alpha': 45.0})}
_OPTIMIZATION_DESIGN = (205, 245)   # design cells inside the chi2 slab
_OPTIMIZATION_ITERATIONS = 2
_IMPROVEMENT = 1e-6                 # J_final / J_initial - 1 must exceed it


def _strip(name, x0_cell, x1_cell):
    return GeometrySpec(name, 'rectangle', {'x0': x0_cell * _CELL, 'y0': 0.0, 'x1': x1_cell * _CELL,
                                            'y1': _ROWS * _CELL})


def _cell_line(name, cell):
    return vertical_line(name, cell_centre(cell, _CELL), 0.0, _ROWS * _CELL)


def _gradient_derived(block_eps_r):
    """[block eps_sensitivity integral] for the gradient run (block at _EPS_R), [] for the finite-difference runs."""
    if block_eps_r != _EPS_R:
        return []
    return [DerivedValueSpec(_GRADIENT_VALUE, f'{_PHYSICS}_eps_sensitivity', ['block'], 'integral')]


def _cascade_spec(case, block_eps_r, study='nonlinear_frequency_domain', design=False):
    """Strip of polarisation set `case` with the block at block_eps_r; design: a design_region instead of the block
    and an optimization study. Only the unperturbed block carries the eps_sensitivity derived value (an adjoint)."""
    pump_polarization, sh_polarization, chi2 = _CASES[case]
    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _WAVELENGTH,
                                                  'polarization': pump_polarization}),
        FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                   'polarization': sh_polarization}),
        FeatureSpec('chi2', 'chi2_medium', ['chi2_slab'], dict(chi2)),
        FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _AMPLITUDE, 'phase': 0.0}),
        FeatureSpec('sh_flux', 'adjoint_objective', ['objective_line'], {'channel': 'sh', 'quantity': 'flux',
                                                                         'weight': 1.0}),
        FeatureSpec('p_flux', 'adjoint_objective', ['objective_line'], {'channel': 'p', 'quantity': 'flux',
                                                                        'weight': _PUMP_WEIGHT})]
    materials = [MaterialSpec('medium', properties={'eps_r': _EPS_R}, selection=['all'])]
    geometry = [_cell_line('source_line', _SOURCE_CELL), _cell_line('objective_line', _OBJECTIVE_CELL),
                _strip('chi2_slab', *_SLAB)]
    derived = []
    if design:
        geometry.append(_strip('design', *_OPTIMIZATION_DESIGN))
        features.append(FeatureSpec('design', 'design_region', ['design'], {
            'eps_min': _EPS_R, 'eps_max': 4.0, 'initial_density': 0.5, 'filter_radius': 0.0}))
        study_spec = StudySpec(type='optimization', settings={'max_iterations': _OPTIMIZATION_ITERATIONS,
                                                             'tolerance': 0.0})
    else:
        geometry.append(_strip('block', *_BLOCK))
        materials.append(MaterialSpec('block', properties={'eps_r': block_eps_r}, selection=['block']))
        derived = _gradient_derived(block_eps_r)
        study_spec = StudySpec(type=study)
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear cascade adjoint ({case}, {study_spec.type})',
        domain=DomainSpec(size_x=_N_X * _CELL, size_y=_ROWS * _CELL),
        geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=dict(PERIODIC_Y),
                             features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=study_spec, derived_values=derived)


def _gradient_specs():
    specs = {}
    for case in _CASES:
        specs[f'{case}_gradient'] = _cascade_spec(case, _EPS_R)
        specs[f'{case}_plus'] = _cascade_spec(case, _EPS_R + _STEP)
        specs[f'{case}_minus'] = _cascade_spec(case, _EPS_R - _STEP)
    return specs


def _objective(results):
    """J [W/m] = flux(sh) + _PUMP_WEIGHT flux(p) of a single-solve run."""
    values = {name: single_value(results.result(_PHYSICS, name)['series']['value'], f'{name} objective')
              for name in ('sh_flux', 'p_flux')}
    return values['sh_flux'] + _PUMP_WEIGHT * values['p_flux'], values


def _check_gradient(results_by_name):
    """Per polarisation set: block integral of eps_sensitivity vs the central difference of J; measured = error = the
    worst relative deviation / 2e-3, tolerance 1."""
    normalized, details = {}, {'unit': 'W/m per unit eps_r', 'step': _STEP}
    for case in _CASES:
        plus, _ = _objective(results_by_name[f'{case}_plus'])
        minus, _ = _objective(results_by_name[f'{case}_minus'])
        finite_difference = (plus - minus) / (2.0 * _STEP)
        gradient_results = results_by_name[f'{case}_gradient']
        adjoint = float(gradient_results.summary()['derived'][_GRADIENT_VALUE])
        error = abs(adjoint / finite_difference - 1.0) if finite_difference != 0.0 else math.nan
        normalized[case] = error / _TOLERANCE
        objective, parts = _objective(gradient_results)
        details.update({f'{case}_J': objective, f'{case}_objectives': parts, f'{case}_adjoint': adjoint,
                        f'{case}_finite_difference': finite_difference, f'{case}_relative_error': error})
    return normalized_result(normalized, details)


def _optimization_specs():
    return {'run': _cascade_spec('ez', _EPS_R, design=True)}


def _check_optimization(results_by_name):
    """Accepted L-BFGS-B iterates raise J: >= 1 iteration and J_final > J_initial (1 + 1e-6); error = failed checks."""
    record = results_by_name['run'].summary().get('optimization') or {}
    initial, final = record.get('objective_initial', math.nan), record.get('objective_final', math.nan)
    iterations = int(record.get('iterations', 0))
    failed = []
    if iterations < 1:
        failed.append(f'no accepted iteration ({record.get("message")})')
    if not final > initial * (1.0 + _IMPROVEMENT):
        failed.append(f'J did not increase: {initial!r} -> {final!r} W/m')
    gain = final / initial if initial else math.nan
    return {'measured': gain, 'expected': 1.0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': {'optimization': record, 'failed_checks': failed}}


# Pump-depleted fixed-point adjoint (adjoint.depleted_cotangents) on the same strip, block and objective line:
# depleted_shg = ez -> ez SHG with a fictitious d33 = 10 nm/V (|E_2w| ~ 0.2 |E_w| at the slab end: back-action and pump
# depletion change the gradient by ~1e-2, far above the tolerance), picard (chi2 only: no Kerr operator term), J =
# flux(sh) + 0.5 flux(p); depleted_kerr = one ez channel, chi3 slab with (3/4) chi3 |E|^2 = 0.05 (SPM), newton, J =
# flux(p). Forward tolerance 1e-9 (the adjoint GMRES runs to it): its FD error ~tol / (2 step x relative sensitivity)
# <~1e-5 vs the 2e-3 tolerance; central difference as above.
_DEPLETED_D = 10e-9                  # m/V, fictitious: (omega / (n c)) d |E|^2 L ~ 0.18 |E| in the 0.65 um slab
_DEPLETED_CHI3 = 0.05 / (0.75 * _AMPLITUDE ** 2)   # m^2/V^2
_DEPLETED_CASES = {
    'depleted_shg': (('p', 'sh'), FeatureSpec('chi2', 'chi2_medium', ['chi2_slab'], {
        'crystal_class': '3m', 'd33': _DEPLETED_D, 'orientation': 'z_cut'}), 'picard', {'sh_flux': 1.0, 'p_flux': 0.5}),
    'depleted_kerr': (('p',), FeatureSpec('chi3', 'chi3_medium', ['chi2_slab'], {
        'tensor': 'isotropic', 'strength': 'chi3', 'chi3_real': _DEPLETED_CHI3, 'chi3_imag': 0.0}), 'newton',
        {'p_flux': 1.0})}
_DEPLETED_SETTINGS = {'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic', 'coupling_mode': 'pump_depleted',
                      'tolerance': 1e-9, 'max_iterations': 200}


def _depleted_spec(case, block_eps_r):
    """Strip of pump-depleted case `case` (_DEPLETED_CASES) with the block at block_eps_r."""
    channels, medium, method, weights = _DEPLETED_CASES[case]
    features = [FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _WAVELENGTH,
                                                          'polarization': 'ez'})]
    if 'sh' in channels:
        features.append(FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                                   'polarization': 'ez'}))
    features += [medium, FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _AMPLITUDE,
                                                                              'phase': 0.0})]
    features += [FeatureSpec(name, 'adjoint_objective', ['objective_line'], {
        'channel': name.split('_')[0], 'quantity': 'flux', 'weight': weight}) for name, weight in weights.items()]
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear pump-depleted adjoint ({case})',
        domain=DomainSpec(size_x=_N_X * _CELL, size_y=_ROWS * _CELL),
        geometry=[_cell_line('source_line', _SOURCE_CELL), _cell_line('objective_line', _OBJECTIVE_CELL),
                  _strip('chi2_slab', *_SLAB), _strip('block', *_BLOCK)],
        materials=[MaterialSpec('medium', properties={'eps_r': _EPS_R}, selection=['all']),
                   MaterialSpec('block', properties={'eps_r': block_eps_r}, selection=['block'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings={**_DEPLETED_SETTINGS, 'iteration_method': method},
                             features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='nonlinear_frequency_domain'),
        derived_values=_gradient_derived(block_eps_r))


def _depleted_specs():
    specs = {}
    for case in _DEPLETED_CASES:
        specs[f'{case}_gradient'] = _depleted_spec(case, _EPS_R)
        specs[f'{case}_plus'] = _depleted_spec(case, _EPS_R + _STEP)
        specs[f'{case}_minus'] = _depleted_spec(case, _EPS_R - _STEP)
    return specs


def _weighted_objective(results, weights):
    """J [W/m] = sum weight x flux objective of a single-solve run, and the values."""
    values = {name: single_value(results.result(_PHYSICS, name)['series']['value'], f'{name} objective')
              for name in weights}
    return sum(weight * values[name] for name, weight in weights.items()), values


def _check_depleted_gradient(results_by_name):
    return _check_weighted_gradients(results_by_name, {case: entry[3] for case, entry in _DEPLETED_CASES.items()})


def _check_weighted_gradients(results_by_name, weights_by_case):
    """Per case: block integral of eps_sensitivity vs the central difference of J (weights {objective: weight}), and
    converged forward solves; measured = error = the worst relative deviation / 2e-3, tolerance 1."""
    normalized, details = {}, {'unit': 'W/m per unit eps_r', 'step': _STEP}
    for case, weights in weights_by_case.items():
        plus, _ = _weighted_objective(results_by_name[f'{case}_plus'], weights)
        minus, _ = _weighted_objective(results_by_name[f'{case}_minus'], weights)
        finite_difference = (plus - minus) / (2.0 * _STEP)
        gradient_results = results_by_name[f'{case}_gradient']
        adjoint = float(gradient_results.summary()['derived'][_GRADIENT_VALUE])
        error = abs(adjoint / finite_difference - 1.0) if finite_difference != 0.0 else math.nan
        solves = [record.get('converged') for name in ('gradient', 'plus', 'minus')
                  for record in (results_by_name[f'{case}_{name}'].summary().get('nonlinear_solve') or {}).values()]
        normalized[case] = error / _TOLERANCE if solves and all(solves) else math.inf
        objective, parts = _weighted_objective(gradient_results, weights)
        details.update({f'{case}_J': objective, f'{case}_objectives': parts, f'{case}_adjoint': adjoint,
                        f'{case}_finite_difference': finite_difference, f'{case}_relative_error': error,
                        f'{case}_forward_converged': solves})
    return normalized_result(normalized, details)


# Sheet and free-carrier transposes on the same strip, block, objective line and central difference: sheet_shg = a D3h
# line sheet (armchair along the line) in cell 240, E_z pump -> H_z SH, undepleted, J = flux(sh) + 1e-9 flux(p);
# sheet_kerr = one E_z channel through a complex sigma3 Kerr sheet in cell 240 ((3/4) eta0 |sigma3| |E|^2 ~ 0.05),
# pump_depleted newton, J = flux(p); carriers = one E_z channel in the chi3 slab of depleted_kerr with two-photon
# absorption (chi3_imag = -0.2 chi3_real) and free carriers (lifetime 0.1 ns: delta eps ~ 1e-2 from N), newton, J =
# flux(p).
_SHEET_LINE_CELL = 240
_SHEET_CHI = 2e-19                  # m^2/V
_SHEET_KERR_SIGMA = (1e-18, 1.5e-18)   # S m^2/V^2 (real, imag)
_CARRIER_LIFETIME = 1e-10           # s
_NEWTON_SETTINGS = {**_DEPLETED_SETTINGS, 'iteration_method': 'newton'}
_EXTENSION_CASES = {   # case -> ({channel: polarization}, nonlinear feature, physics settings, objective weights)
    'sheet_shg': ({'p': 'ez', 'sh': 'hz'}, FeatureSpec('sheet', 'sheet_nonlinearity', ['sheet_line'], {
        'sheet_model': 'D3h', 'chi_s': _SHEET_CHI, 'armchair_angle': 0.0}),
        dict(PERIODIC_Y), {'sh_flux': 1.0, 'p_flux': _PUMP_WEIGHT}),
    'sheet_kerr': ({'p': 'ez'}, FeatureSpec('sheet', 'sheet_nonlinearity', ['sheet_line'], {
        'sheet_model': 'sigma3', 'sigma3_real': _SHEET_KERR_SIGMA[0], 'sigma3_imag': _SHEET_KERR_SIGMA[1]}),
        _NEWTON_SETTINGS, {'p_flux': 1.0}),
    'carriers': ({'p': 'ez'}, FeatureSpec('chi3', 'chi3_medium', ['chi2_slab'], {
        'tensor': 'isotropic', 'strength': 'chi3', 'chi3_real': _DEPLETED_CHI3, 'chi3_imag': -0.2 * _DEPLETED_CHI3,
        'free_carriers': True, 'carrier_lifetime': _CARRIER_LIFETIME, 'background_index': math.sqrt(_EPS_R)}),
        _NEWTON_SETTINGS, {'p_flux': 1.0})}   # fixed carrier n0: the block's eps must not change it (adjoint holds it)


def _extension_spec(case, block_eps_r):
    """Strip of sheet / carrier case `case` (_EXTENSION_CASES) with the block at block_eps_r."""
    channels, medium, settings, weights = _EXTENSION_CASES[case]
    features = [FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _WAVELENGTH,
                                                          'polarization': channels['p']})]
    if 'sh' in channels:
        features.append(FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                                   'polarization': channels['sh']}))
    features += [medium, FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _AMPLITUDE,
                                                                              'phase': 0.0})]
    features += [FeatureSpec(name, 'adjoint_objective', ['objective_line'], {
        'channel': name.split('_')[0], 'quantity': 'flux', 'weight': weight}) for name, weight in weights.items()]
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear sheet / carrier adjoint ({case})',
        domain=DomainSpec(size_x=_N_X * _CELL, size_y=_ROWS * _CELL),
        geometry=[_cell_line('source_line', _SOURCE_CELL), _cell_line('objective_line', _OBJECTIVE_CELL),
                  _cell_line('sheet_line', _SHEET_LINE_CELL), _strip('chi2_slab', *_SLAB),
                  _strip('block', *_BLOCK)],
        materials=[MaterialSpec('medium', properties={'eps_r': _EPS_R}, selection=['all']),
                   MaterialSpec('block', properties={'eps_r': block_eps_r}, selection=['block'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=dict(settings), features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='nonlinear_frequency_domain'),
        derived_values=_gradient_derived(block_eps_r))


def _extension_specs():
    specs = {}
    for case in _EXTENSION_CASES:
        specs[f'{case}_gradient'] = _extension_spec(case, _EPS_R)
        specs[f'{case}_plus'] = _extension_spec(case, _EPS_R + _STEP)
        specs[f'{case}_minus'] = _extension_spec(case, _EPS_R - _STEP)
    return specs


def _check_extension_gradient(results_by_name):
    return _check_weighted_gradients(results_by_name, {case: entry[3] for case, entry in _EXTENSION_CASES.items()})


# Symmetrized transposed solve (adjoint.transposed_channel_solve, the route of a factor without a transposed solve, e.g.
# cuDSS): a stub channel on a graded walls + CPML Helmholtz operator with random complex face weights / cell term and
# an even pinned row, its forward solves by SuperLU; no model run.
_TRANSPOSE_SHAPE = (14, 11)
_TRANSPOSE_CELL = 20e-9              # m, base width (graded x 1 .. 1.5) and CPML grading cell
_TRANSPOSE_NPML = 4
_TRANSPOSE_SEED = 23
_TRANSPOSE_TOLERANCE = 1e-9          # relative vs scipy's solve of A^T (first estimate; round-off ~1e-13 expected)


class _StubChannel:
    """The FDFDPhysics attributes transposed_channel_solve reads; solve_adjoint raises as a cuDSS factor does."""

    def __init__(self, domain, config, omega, operator):
        import types
        import numpy as np
        import scipy.sparse.linalg
        self.domain, self.boundary_config, self.metric, self.xp = domain, config, None, np
        self.system = types.SimpleNamespace(omega=omega, operator=operator)
        self._lu = scipy.sparse.linalg.splu(operator.tocsc())

    def solve_adjoint(self, flat_rhs):
        raise NotImplementedError('stub factor: no transposed solve')

    def _solve_linear(self, flat_rhs):
        return self._lu.solve(flat_rhs)


def _check_symmetrized_transpose(_results_by_name):
    """transposed_channel_solve vs scipy's A^T solve on the stub channel; failed checks: no NotImplementedError with a
    Bloch phase, no RuntimeError for an operator D does not symmetrize."""
    import types
    import numpy as np
    import scipy.sparse.linalg
    from ...core.constants import C0
    from ..em.boundaries import BoundaryConfig
    from ..em.operators import helmholtz_operator, pin_image_cells, yee_operators
    from .adjoint import transposed_channel_solve
    rng = np.random.default_rng(_TRANSPOSE_SEED)
    nx, ny = _TRANSPOSE_SHAPE
    widths = (_TRANSPOSE_CELL * (1.0 + 0.5 * rng.random(nx)), _TRANSPOSE_CELL * (1.0 + 0.5 * rng.random(ny)))
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    config = BoundaryConfig('pmc', 'cpml', 'pec', 'cpml')
    config.cpml_npml = _TRANSPOSE_NPML
    ops = yee_operators(widths, config, _TRANSPOSE_CELL, omega=omega)
    weights = [1.0 / (1.0 + rng.random((nx, ny)) - 0.2j * rng.random((nx, ny))) for _ in range(2)]
    cell_term = (omega / C0) ** 2 * (2.0 + rng.random((nx, ny)) - 0.3j * rng.random((nx, ny)))
    pinned = np.zeros((nx, ny), dtype=bool)
    pinned[0, :] = True   # an even image row, as a mirror plane
    operator = pin_image_cells(helmholtz_operator(ops, weights[0], weights[1], cell_term), pinned,
                               pinned.astype(np.float64)).tocsr()
    domain = types.SimpleNamespace(non_uniform=True, mesh_config=types.SimpleNamespace(dx_base=_TRANSPOSE_CELL),
                                   grid=types.SimpleNamespace(nx=nx, ny=ny, dx_arr=widths[0], dy_arr=widths[1]))
    rhs = rng.standard_normal(nx * ny) + 1j * rng.standard_normal(nx * ny)
    solution = transposed_channel_solve(_StubChannel(domain, config, omega, operator), rhs)
    reference = scipy.sparse.linalg.spsolve(operator.T.tocsc(), rhs)
    error = float(np.linalg.norm(solution - reference) / np.linalg.norm(reference))
    failed = []
    bloch = BoundaryConfig.with_bloch_y(_TRANSPOSE_NPML, ky=1e6)
    try:
        transposed_channel_solve(_StubChannel(domain, bloch, omega, operator), rhs)
        failed.append('Bloch phase not refused')
    except NotImplementedError:
        pass
    skewed = operator.tolil()
    skewed[7 * ny, 2 * ny + 5] += 0.3 * abs(operator).max()   # cells (7, 0) <- (2, 5): no transposed partner
    try:
        transposed_channel_solve(_StubChannel(domain, config, omega, skewed.tocsr()), rhs)
        failed.append('unsymmetrizable operator not refused')
    except RuntimeError:
        pass
    return {'measured': error, 'expected': 0.0, 'error': error + len(failed),
            'tolerance': _TRANSPOSE_TOLERANCE, 'details': {'relative_error': error, 'failed': failed}}


# shg_metasurface_optimization template: y-periodic 0.5 um cell (below the SH wavelength in silica: zeroth order only),
# CPML along x, uniform 20 nm mesh (window cell k spans [20 k, 20 (k + 1)] nm: layers on cell faces, lines at cell
# centres). Air x < 2.3 um with the pump line source at 0.51 um, a 0.8 um design layer x 1.5..2.3 um (air to silicon
# nitride), a 0.3 um z-cut LN film x 2.3..2.6 um (d33 25 pm/V, E_z along c), silica x > 2.6 um, the SH objective line at
# x = 4.01 um. Non-dispersive (the film is far thinner than a coherence length); coefficients illustrative.
SHG_METASURFACE_LABEL = 'Inverse-designed SHG metasurface'
SHG_METASURFACE_DESCRIPTION = (
    'em_fdfd_nonlinear (E_z channels p at 1.55 um and sh = 2 p, undepleted cascade), optimization study: a y-periodic '
    '0.5 um cell with a 0.3 um z-cut lithium niobate film (d33 25 pm/V, eps 4.58) on silica, pumped at normal '
    'incidence by a 1e7 V/m line source in air. A 0.8 um design layer on the film (density from air to silicon nitride, '
    'eps 1 .. 4, filter radius 60 nm, tanh projection reaching beta 8 over 3 continuation stages, at most 30 L-BFGS-B '
    'iterations each) is shaped to maximise the second-harmonic flux transmitted into the silica (adjoint_objective of '
    'channel sh): the layer focuses and phase-matches the pump in the film. The design_density field is stored per '
    'iteration; the design_region result shows J and the grayness versus iteration. 2D: powers per unit length in z.')
SHG_METASURFACE_SUMMARY = ('Lets the optimiser shape a patterned layer on a thin lithium niobate film so that it '
                           'converts as much 1550 nm light as possible into its second harmonic at 775 nm.')
_META_CELL = 20e-9                   # m
_META_PERIOD = 0.5e-6                # m
_META_SIZE_X = 5.0e-6                # m
_META_DESIGN = (1.5e-6, 2.3e-6)      # m, x range of the design layer
_META_FILM = (2.3e-6, 2.6e-6)        # m, x range of the LN film
_META_SOURCE_X, _META_OBJECTIVE_X = 0.51e-6, 4.01e-6   # m
_META_EPS = {'air': 1.0, 'nitride': 4.0, 'ln': 4.58, 'silica': 2.1}


def shg_metasurface_optimization_spec():
    """Template 'shg_metasurface_optimization': inverse-designed SHG layer on a thin chi2 film (cascade adjoint)."""
    def layer(name, x0, x1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': 0.0, 'x1': x1, 'y1': _META_PERIOD})

    def line(name, x):
        return GeometrySpec(name, 'line', {'x0': x, 'y0': 0.0, 'x1': x, 'y1': _META_PERIOD})

    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _WAVELENGTH,
                                                  'polarization': 'ez'}),
        FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                   'polarization': 'ez'}),
        FeatureSpec('ln_chi2', 'chi2_medium', ['film'], {'crystal_class': '3m', 'd33': 25e-12, 'orientation': 'z_cut'}),
        FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _AMPLITUDE, 'phase': 0.0}),
        FeatureSpec('sh_flux', 'adjoint_objective', ['objective_line'], {'channel': 'sh', 'quantity': 'flux',
                                                                         'weight': 1.0}),
        FeatureSpec('design', 'design_region', ['design'], {
            'eps_min': _META_EPS['air'], 'eps_max': _META_EPS['nitride'], 'initial_density': 0.5,
            'filter_radius': 60e-9, 'projection_beta': 8.0, 'projection_eta': 0.5})]
    return ModelSpec(
        title='Inverse-designed SHG metasurface (E_z, 1.55 um -> 775 nm)',
        domain=DomainSpec(size_x=_META_SIZE_X, size_y=_META_PERIOD),
        geometry=[layer('design', *_META_DESIGN), layer('film', *_META_FILM), layer('substrate', _META_FILM[1],
                                                                                    _META_SIZE_X),
                  line('source_line', _META_SOURCE_X), line('objective_line', _META_OBJECTIVE_X)],
        materials=[MaterialSpec('air', properties={'eps_r': _META_EPS['air']}, selection=['all']),
                   MaterialSpec('lithium_niobate_e', properties={'eps_r': _META_EPS['ln']}, selection=['film']),
                   MaterialSpec('silica', properties={'eps_r': _META_EPS['silica']}, selection=['substrate'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=dict(PERIODIC_Y),
                             features=features)],
        mesh=MeshSpec(mode='uniform', dx=_META_CELL),
        study=StudySpec(type='optimization', settings={'max_iterations': 30, 'tolerance': 1e-6,
                                                       'continuation_stages': 3, 'beta_growth': 2.0}))


ADJOINT_TEMPLATES = (   # (key, label, description, factory, summary) for registry.register_template
    ('shg_metasurface_optimization', SHG_METASURFACE_LABEL, SHG_METASURFACE_DESCRIPTION,
     shg_metasurface_optimization_spec, SHG_METASURFACE_SUMMARY),
)


CASCADE_ADJOINT_SELFTESTS = (
    SelfTestCase(
        name='nlfd_cascade_adjoint_gradient',
        label='NLFD cascade adjoint sensitivity vs finite difference',
        description='y-periodic strip (4 rows, CPML along x, uniform 10 nm mesh, eps_r 2.25), pump channel p at 1.55 '
                    'um (line source 1e7 V/m in cell 100), SH channel sh = 2 p, a 65-cell chi2 slab from cell 200, '
                    'undepleted, nonlinear_frequency_domain study. Two adjoint_objective features on the line in cell '
                    '365: J = flux(sh) + 1e-9 flux(p). Per polarisation set (ez -> ez with 3m d33; hz -> hz with '
                    "'general' d22; hz pump -> ez SH with -43m d14 rotated 45 deg) a gradient run integrates "
                    'eps_sensitivity over a 4-cell full-height block inside the chi2 slab (derived value, W/m per '
                    'unit eps_r) and two runs with the block at eps_r 2.25 +- 0.01 (no derived value: no adjoint) '
                    'give the central difference of J. '
                    'measured = error = max |adjoint / finite difference - 1| / 2e-3, tolerance 1. Budget: the '
                    'adjoint is the exact derivative of the discrete cascade (one transposed solve per channel in '
                    'reverse solve order, the polynomial, C and C* transposed, hz the W of the stored face fields and '
                    'of the nonlinear face currents), so only the O(step^2) difference error (~1e-5) and float32 '
                    'storage remain; it pins every back-propagation term of adjoint.py and the em adjoint pieces it '
                    'reuses.',
        specs=_gradient_specs,
        check=_check_gradient),
    SelfTestCase(
        name='nlfd_shg_optimization_run',
        label='NLFD SHG optimization run (cascade adjoint in L-BFGS-B)',
        description='The ez strip of nlfd_cascade_adjoint_gradient with a design_region on cells [205, 245) of the '
                    'chi2 slab (eps 2.25 .. 4, initial density 0.5, no filter or projection) in an optimization study '
                    f'({_OPTIMIZATION_ITERATIONS} L-BFGS-B iterations, tolerance 0): every accepted iterate lowers -J, '
                    'so the run must accept at least one iteration and end with objective_final > objective_initial '
                    '(1 + 1e-6). error = the number of failed checks, tolerance 0; measured = J_final / J_initial. '
                    'Iterations 2 (was 3): not yet confirmed by a run.',
        specs=_optimization_specs,
        check=_check_optimization),
    SelfTestCase(
        name='nlfd_depleted_adjoint_gradient',
        label='NLFD pump-depleted adjoint sensitivity vs finite difference',
        description='The strip of nlfd_cascade_adjoint_gradient (block, objective line and central difference alike) '
                    'in coupling_mode pump_depleted (forward tolerance 1e-9; was 1e-11, not yet confirmed by a run): '
                    'depleted_shg = ez -> ez SHG with a '
                    'fictitious d33 = 10 nm/V (|E_2w| ~ 0.2 |E_w|: back-action and pump depletion matter), picard, J = '
                    'flux(sh) + 0.5 flux(p); depleted_kerr = one ez channel in a chi3 slab ((3/4) chi3 |E|^2 = 0.05: '
                    'SPM), newton, J = flux(p). The adjoint is the transposed fixed point (I - T) nu = T(c) by real '
                    'GMRES plus one gradient sweep (adjoint.depleted_cotangents), exact for the converged discrete '
                    'fixed point. measured = error = max |adjoint / finite difference - 1| / 2e-3 (inf when a forward '
                    'solve did not converge), tolerance 1.',
        specs=_depleted_specs,
        check=_check_depleted_gradient),
    SelfTestCase(
        name='nlfd_sheet_carrier_adjoint_gradient',
        label='NLFD adjoint of nonlinear sheets and free carriers vs finite difference',
        description='The strip of nlfd_cascade_adjoint_gradient (block, objective line and central difference alike): '
                    f'sheet_shg = a D3h sheet_nonlinearity (chi_s {_SHEET_CHI:g} m^2/V, armchair along the line) on '
                    f'the vertical line in cell {_SHEET_LINE_CELL}, E_z pump -> H_z SH, undepleted, J = flux(sh) + 1e-9 '
                    'flux(p); sheet_kerr = one E_z channel through a sigma3 Kerr sheet on that line (sigma3 '
                    f'{_SHEET_KERR_SIGMA[0]:g} + {_SHEET_KERR_SIGMA[1]:g} j S m^2/V^2), pump_depleted newton, J = '
                    'flux(p); carriers = one E_z channel in the chi3 slab of nlfd_depleted_adjoint_gradient with '
                    'two-photon absorption (chi3_imag = -0.2 chi3_real) and free carriers (lifetime '
                    f'{_CARRIER_LIFETIME:g} s), newton, J = flux(p); pump-depleted forward tolerance '
                    f'{_DEPLETED_SETTINGS["tolerance"]:g} (not yet confirmed by a run). The gradient back-propagates '
                    'through the site transposes (sheets.sheet_cotangents) and the carrier transpose '
                    '(adjoint.carrier_cotangents). measured = error = max |adjoint / finite difference - 1| / 2e-3 '
                    '(inf when a forward solve did not converge), tolerance 1.',
        specs=_extension_specs,
        check=_check_extension_gradient),
    SelfTestCase(
        name='nlfd_symmetrized_transpose',
        label='NLFD adjoint: transposed solve through the symmetrizer',
        description='CPU check of adjoint.transposed_channel_solve (no model run), the route the channel adjoint takes '
                    'when the channel factor has no transposed solve (a cuDSS back end): x = D A^-1 D^-1 b with D the '
                    'stretched cell areas, on a stub channel with a graded 14 x 11 Helmholtz operator (pmc / pec walls, '
                    'CPML on x_hi / y_hi at 1.55 um, random complex face weights and cell term, an even pinned row) '
                    "and SuperLU forward solves, vs scipy's solve of A^T. measured = relative error; error = measured "
                    '+ failed checks (a Bloch-phase channel must raise NotImplementedError, an operator with one '
                    f'unpartnered entry RuntimeError); tolerance {_TRANSPOSE_TOLERANCE:g} (an estimate). Not yet '
                    'confirmed by a run.',
        specs=lambda: {},
        check=_check_symmetrized_transpose),
)

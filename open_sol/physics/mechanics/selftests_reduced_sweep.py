"""Self-tests of the solid reduced frequency sweeps (experiment plan A3 / A4): frequency_response.HarmonicSystem's
reduced model and its precomputed record rows against direct solves (CPU, synthetic pencil), and the driven lossy bar on
triangles with sweep_method adaptive / modal against the direct sweep (probe, reaction table, live field)."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result
from ...model.spec import DerivedValueSpec, FeatureSpec, PhysicsSpec, StudySpec
from .selftests_prestress_modes import (_BAR, _BAR_H, _BAR_MATERIAL, _BAR_POINT, _BAR_TRACTION, _LOSS_FACTOR, _SOLID,
                                        _SOLID_TYPE, _point_2d, _triangle_model)

# ---- solid_harmonic_reduced_rows (CPU, specs {}): a chain pencil of 80 dofs (K SPD tridiagonal x 1e3, M diagonal, loss
# eta K), three held dofs with non-zero values, a resistive j b / omega diagonal term and an extra rhs (a floating
# charge): the adaptive model (training residual 1e-12, exact sketch) against a direct HarmonicSystem at every 7th of
# 300 frequencies over 5 - 60 rad/s (the lowest modes), and solve(fields=False) at the record rows (free, held, the
# resistive dof) against the full solve: x and A x.
_CHAIN_DOFS = 80
_CHAIN_HELD = (0, 37, 79)
_CHAIN_HELD_VALUES = (1e-3, 0.5e-3, -2e-3)
_CHAIN_RESISTIVE = (10, 3.0)       # (dof, b): + j b / omega on its diagonal
_CHAIN_CHARGE = (20, 0.1)          # (dof, value) subtracted from the rhs
_CHAIN_ROWS = (3, 10, 37, 50)
_CHAIN_OMEGAS = (5.0, 60.0, 300)
_CHAIN_TOLERANCE = 1e-8            # relative to |x| (residual 1e-12 x Q ~ 100; first estimate)
_CHAIN_ROWS_TOLERANCE = 1e-11      # fields=False rows against the planned full solve (same coordinates)


def _chain_pencil():
    """(K, loss, M, free mask, held values, load) of the synthetic chain."""
    import scipy.sparse
    generator = np.random.default_rng(5)
    n = _CHAIN_DOFS
    diagonal = 2.0 + generator.random(n)
    coupling = -0.9 * generator.random(n - 1)
    stiffness = 1e3 * scipy.sparse.diags([coupling, diagonal, coupling], [-1, 0, 1], format='csr')
    mass = scipy.sparse.diags(1.0 + generator.random(n), 0, format='csr')
    free = np.ones(n, dtype=bool)
    free[list(_CHAIN_HELD)] = False
    held = np.zeros(n, dtype=np.complex128)
    held[list(_CHAIN_HELD)] = _CHAIN_HELD_VALUES
    return stiffness, _LOSS_FACTOR * stiffness, mass, free, held, generator.standard_normal(n)


def _check_reduced_rows(_):
    from .frequency_response import HarmonicSystem
    stiffness, loss, mass, free, held, load = _chain_pencil()
    loads, charges = [_CHAIN_RESISTIVE], [_CHAIN_CHARGE]
    direct = HarmonicSystem(stiffness, loss, mass, free, loads=loads)
    planned = HarmonicSystem(stiffness, loss, mass, free, loads=loads)
    omegas = np.linspace(*_CHAIN_OMEGAS)
    info = planned.plan_reduced(omegas, 'adaptive', held, load, charges, tolerance=1e-12, moments=4, max_points=40)
    rows = np.array(_CHAIN_ROWS)
    planned.set_output_rows(rows)
    full_error = rows_error = 0.0
    for omega in omegas[::7]:
        exact, exact_product = direct.solve(omega, held, load, charges)
        reduced, reduced_product = planned.solve(omega, held, load, charges)
        sparse, sparse_product = planned.solve(omega, held, load, charges, fields=False)
        full_error = max(full_error, float(np.linalg.norm(reduced - exact) / np.linalg.norm(exact)))
        rows_error = max(rows_error,
                         float(np.linalg.norm(sparse[rows] - reduced[rows]) / np.linalg.norm(reduced[rows])),
                         float(np.linalg.norm(sparse_product[rows] - reduced_product[rows])
                               / np.linalg.norm(reduced_product[rows])))
    return scaled_worst_result({'reduced vs direct': full_error, 'rows vs full': rows_error,
                                'converged': 0.0 if info['converged'] else math.inf},
                               {'reduced vs direct': _CHAIN_TOLERANCE, 'rows vs full': _CHAIN_ROWS_TOLERANCE,
                                'converged': 1.0},
                               basis_size=info['basis_size'], factorizations=info['factorizations'],
                               max_residual=info['max_residual'], verified_error=info['verified_error'])


# ---- solid_fem_reduced_sweep: selftests_prestress_modes' driven lossy bar (fixed x_lo, P2, h 0.25 mm, eta 1e-2)
# over 120 - 280 kHz (161 frequencies around the first resonance ~194 kHz): direct, adaptive (rom_tolerance 1e-10) and
# modal (6 modes + static correction) runs with snapshots 'none' (A4: records from the reduced coordinates, the fields
# of the last frequency only). The frequency_probe phasors and the x_lo reaction table against the direct sweep, and the
# live u_x at the probe point (the last frequency's published field).
_SWEEP_HZ = (120e3, 280e3, 161)
_SWEEP_ROM_TOLERANCE = 1e-10
_SWEEP_MODAL_COUNT = 6
_SWEEP_ADAPTIVE_TOLERANCE = 1e-6   # relative to the peak (first estimate)
_SWEEP_MODAL_TOLERANCE = 1e-3      # truncated modes + static correction (first estimate)
_SWEEP_MAX_FACTORIZATIONS = 41     # rom_max_points + the verification
_SWEEP_METHODS = ('direct', 'adaptive', 'modal')


def _sweep_spec(method):
    youngs_modulus, poisson_ratio, density = _BAR_MATERIAL
    material = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'rho': density,
                'mechanical_loss_factor': _LOSS_FACTOR}
    settings = {'frequency_start': _SWEEP_HZ[0], 'frequency_stop': _SWEEP_HZ[1], 'n_frequencies': _SWEEP_HZ[2],
                'snapshots': 'none', 'sweep_method': method}
    if method == 'adaptive':
        settings['rom_tolerance'] = _SWEEP_ROM_TOLERANCE
    elif method == 'modal':
        settings['modal_count'] = _SWEEP_MODAL_COUNT
    features = [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('sides', 'roller', ['y_lo', 'y_hi']),
                FeatureSpec('drive', 'boundary_load', ['x_hi'], {'t_x': _BAR_TRACTION, 't_y': 0.0}),
                FeatureSpec('tip', 'frequency_probe', ['p'])]
    spec = _triangle_model(f'Self-test: driven lossy bar on triangles, {method} frequency sweep', _BAR, _BAR_H,
                           [PhysicsSpec(_SOLID, _SOLID_TYPE, features=features)],
                           StudySpec(type='frequency_response', settings=settings), material,
                           geometry=[_point_2d('p', *_BAR_POINT)])
    spec.derived_values = [DerivedValueSpec('u_x_tip', 'u_x', [], 'at', point='p')]
    return spec


def _sweep_specs():
    return {method: _sweep_spec(method) for method in _SWEEP_METHODS}


def _sweep_records(results):
    """(frequencies, complex probe u_x, complex x_lo reaction F_x, live u_x at the probe) of one run."""
    probe = results.result(_SOLID, 'tip')
    phasors = (np.asarray(probe['series']['u_x_re'], dtype=np.float64)
               + 1j * np.asarray(probe['series']['u_x_im'], dtype=np.float64))
    table = results.result(_SOLID, 'left')
    columns = table['columns']
    reactions = np.array([float(row[columns.index('F_x_re')]) + 1j * float(row[columns.index('F_x_im')])
                          for row in table['rows']])
    return (np.asarray(probe['x'], dtype=np.float64), phasors, reactions,
            float(results.summary()['derived']['u_x_tip']))


def _check_sweep(results_by_name):
    frequencies, probe, reaction, live = _sweep_records(results_by_name['direct'])
    probe_scale, reaction_scale = float(np.max(np.abs(probe))), float(np.max(np.abs(reaction)))
    errors = {'points': abs(len(frequencies) - _SWEEP_HZ[2])}
    tolerances = {'points': 0.5}
    details = {'peak_u_x': probe_scale, 'peak_F_x': reaction_scale}
    for method, tolerance in (('adaptive', _SWEEP_ADAPTIVE_TOLERANCE), ('modal', _SWEEP_MODAL_TOLERANCE)):
        results = results_by_name[method]
        _, measured, measured_reaction, measured_live = _sweep_records(results)
        same = measured.shape == probe.shape and measured_reaction.shape == reaction.shape
        errors[f'{method} probe'] = float(np.max(np.abs(measured - probe))) / probe_scale if same else math.inf
        errors[f'{method} reaction'] = (float(np.max(np.abs(measured_reaction - reaction))) / reaction_scale if same
                                        else math.inf)
        errors[f'{method} live u_x'] = abs(measured_live - live) / probe_scale
        for item in ('probe', 'reaction', 'live u_x'):
            tolerances[f'{method} {item}'] = tolerance
        record = ((results.summary().get('frequency_sweep') or {}).get('physics') or {}).get(_SOLID) or {}
        details[f'{method}_sweep'] = record
        if method == 'adaptive':
            errors['adaptive factorizations'] = float(record.get('factorizations', math.inf))
            tolerances['adaptive factorizations'] = float(_SWEEP_MAX_FACTORIZATIONS)
    return scaled_worst_result(errors, tolerances, **details)


REDUCED_SWEEP_CASES = (
    SelfTestCase(
        name='solid_harmonic_reduced_rows',
        label='Solid mechanics: reduced harmonic model and precomputed record rows (synthetic pencil)',
        description=(f'frequency_response.HarmonicSystem on a {_CHAIN_DOFS}-dof chain pencil (K SPD, loss '
                     f'{_LOSS_FACTOR:g} K, diagonal M) with {len(_CHAIN_HELD)} non-zero held dofs, a resistive j b / '
                     'omega term and a floating-charge rhs: sweep_method adaptive (residual 1e-12) against direct '
                     f'solves within {_CHAIN_TOLERANCE:g} relative, and solve(fields=False) at the record rows (x and A '
                     f'x) against the full reduced solve within {_CHAIN_ROWS_TOLERANCE:g}. CPU; measured = error in '
                     'tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=lambda: {}, check=_check_reduced_rows),
    SelfTestCase(
        name='solid_fem_reduced_sweep',
        label='Solid mechanics on triangles: adaptive and modal frequency sweeps of a driven lossy bar vs direct',
        description=(f'The driven lossy bar of solid_fem_frequency_probe (fixed x_lo, eta {_LOSS_FACTOR:g}) over '
                     f'{_SWEEP_HZ[0] / 1e3:g} - {_SWEEP_HZ[1] / 1e3:g} kHz ({_SWEEP_HZ[2]} frequencies, snapshots none): '
                     'the frequency_probe phasors, the x_lo reaction table and the live u_x at the probe (the last '
                     f'frequency) of sweep_method adaptive (rom_tolerance {_SWEEP_ROM_TOLERANCE:g}) against the direct '
                     f'sweep within {_SWEEP_ADAPTIVE_TOLERANCE:g} of the peak with at most {_SWEEP_MAX_FACTORIZATIONS} '
                     f'factorisations, and of modal ({_SWEEP_MODAL_COUNT} modes + static correction) within '
                     f'{_SWEEP_MODAL_TOLERANCE:g}. measured = error in tolerance units, tolerance 1 (first estimates, '
                     'not yet run).'),
        specs=_sweep_specs, check=_check_sweep),
)

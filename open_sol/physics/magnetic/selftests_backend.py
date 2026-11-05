"""CPU vs GPU backend cases (solver_plan S6 stage 2). magnetostatics: the remanent slab, the coil pair and the
tabulated B-H sheet (Anderson loop) on the cpu backend with the Automatic linear solver (host SuperLU) against their
exact discrete solutions, the sheet also on the gpu (exact pcg node) against its closed form and the cpu run (registered by
definition_magnetostatics.register). magnetic_fields_ac: the skin slab and the solid strip on both backends (registered
by definition_ac.register). api.selftest skips the gpu runs without a usable GPU."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import relative_error, solver_role
from . import selftests as magnetic_selftests
from . import selftests_ac as magnetic_ac_selftests

_REMANENT_CPU = 'remanent_slab_cpu_direct'
_COIL_CPU = 'coil_pair_cpu_direct'
_SHEET_RUNS = {'cpu': 'tabulated_sheet_cpu_cg', 'gpu': 'tabulated_sheet_gpu_cg'}
_TOLERANCE = 1e-6   # exact discrete solutions: SuperLU (cpu) / pcg 1e-12 (gpu sheet) solves inside a 1e-8 Anderson loop
_MAGNETIC = 'magnetic'   # the physics instance name of the magnetic selftests models


def _on_backend(spec, backend):
    spec.solver_defaults.backend = backend
    return spec


def _automatic(spec):
    """spec without solver nodes or assignments (every role Automatic: host SuperLU on the cpu)."""
    spec.solvers = []
    for physics in spec.physics:
        physics.solvers = {}
    return spec


def _backend_runs():
    run = magnetic_selftests._RUN
    return {
        _REMANENT_CPU: _on_backend(_automatic(magnetic_selftests._remanent_slab_specs()[run]), 'cpu'),
        _COIL_CPU: _on_backend(_automatic(magnetic_selftests._coil_pair_specs()[run]), 'cpu'),
        _SHEET_RUNS['cpu']: _on_backend(_automatic(magnetic_selftests._tabulated_sheet_specs()[run]), 'cpu'),
        _SHEET_RUNS['gpu']: _on_backend(magnetic_selftests._tabulated_sheet_specs()[run], 'gpu'),
    }


def _scaled(outcome):
    """A magnetic check's error scaled by its own tolerance onto _TOLERANCE."""
    return outcome['error'] / outcome['tolerance'] * _TOLERANCE


def _tabulated_sheet_check(results):
    """The magnet_saturating_sheet check of one tabulated-sheet run against the forward table B(H)."""
    return magnetic_selftests._check_saturating_sheet({magnetic_selftests._RUN: results},
                                                      curve_flux_density=magnetic_selftests._table_flux_density)


def _check_backends(results_by_name):
    """Largest of: the cpu remanent slab, coil pair and tabulated sheet (and the gpu sheet) against their exact
    solutions and (with the gpu run) max |B_y,cpu - B_y,gpu| / max |B_y| of the sheet; raises unless each cpu run
    recorded a cpu sparse_direct (Automatic SuperLU) linear solver."""
    run = magnetic_selftests._RUN
    cpu_runs = (_REMANENT_CPU, _COIL_CPU, _SHEET_RUNS['cpu'])
    for name in cpu_runs:
        role = solver_role(results_by_name[name], _MAGNETIC)
        if role.get('backend') != 'cpu' or role.get('type') != 'sparse_direct':
            raise ValueError(f"run '{name}': expected a cpu sparse_direct linear solver, summary 'solvers' records "
                             f'{role!r}')
    sheets = {backend: _tabulated_sheet_check(results_by_name[name])
              for backend, name in _SHEET_RUNS.items() if name in results_by_name}   # gpu absent without a GPU
    errors = {'remanent_slab': _scaled(magnetic_selftests._check_remanent_slab({run: results_by_name[_REMANENT_CPU]})),
              'coil_pair': _scaled(magnetic_selftests._check_coil_pair({run: results_by_name[_COIL_CPU]})),
              **{f'tabulated_sheet_{backend}': _scaled(outcome) for backend, outcome in sheets.items()}}
    if 'gpu' in sheets:
        flux_density = {backend: magnetic_selftests._window_field(results_by_name[name], 'B_y')[0]
                        for backend, name in _SHEET_RUNS.items()}
        scale = float(np.max(np.abs(flux_density['gpu'])))
        errors['sheet_B_y_cpu_gpu'] = float(np.max(np.abs(flux_density['cpu'] - flux_density['gpu']))) / scale
    finite = all(np.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
    error = max(errors.values()) if finite else float('nan')
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'gpu_run': 'gpu' in sheets,
                        'sheet_nonlinear_iterations': {backend: outcome['details'].get('nonlinear_iterations')
                                                       for backend, outcome in sheets.items()},
                        'cpu_solvers': {name: solver_role(results_by_name[name], _MAGNETIC) for name in cpu_runs}}}


BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='magnet_cpu_gpu',
        label='Magnetostatics: CPU backend vs GPU',
        description=('Magnetostatics on solver_defaults.backend cpu (numpy arrays, scipy CSR) with the Automatic cpu '
                     'linear solver (host SuperLU): magnet_remanent_slab, magnet_coil_inductance and the tabulated B-H '
                     'sheet against their exact discrete solutions; the sheet also on the gpu backend with the exact '
                     'pcg node (float64, 1e-12; pcg warm starts inside the loop) against its closed form and the cpu '
                     'run (max B_y difference / max |B_y|). Tabulated sheet: magnet_saturating_sheet (J_z = '
                     f'{magnetic_selftests._SATURATING_CURRENT_DENSITY:g} A/m^2) with the iron given by a bh_curve '
                     'table instead of saturation_polarization: the test curve sampled at H = '
                     f"{', '.join(f'{field:g}' for field in magnetic_selftests._TABLE_FIELDS)} A/m (no origin row: "
                     'the solver prepends it); H reaches 2000 A/m, past the last row, so the cells left of the band '
                     "sit on the mu0 extension and the band cells cross every row; nonlinear_method 'anderson' at "
                     f"relaxation {magnetic_selftests._ANDERSON_SETTINGS['nonlinear_relaxation']:g}, tolerance "
                     f"{magnetic_selftests._ANDERSON_SETTINGS['nonlinear_tolerance']:g}. Exact: face H_y = -(current to "
                     'the right) by Ampere whatever nu, and each cell B_y = B(H_y) of the piecewise-linear table at its centre mean H (numpy '
                     "forward interpolation, independent of the solver's). The table proposal is H-based (nu = |H| / "
                     'B(|H|)): the B-based one would face d ln nu / d ln B ~ 260 on the mu0 extension. Each '
                     f'closed-form error is scaled by its own case tolerance onto {_TOLERANCE:g}; the sheet iterations '
                     "are reported. Pins the backends and the 'table' material property path (validation, build "
                     'regions, add_bh_curve), the origin row, the interpolated secant nu(|H|), the saturated '
                     'extension and the loop on a table. Without a usable GPU the gpu run is skipped. Not run yet '
                     'with the cpu sheet on SuperLU (was numpy pcg) and the gpu sheet closed-form check.'),
        specs=_backend_runs, check=_check_backends, heavy=True),
)


# ---- magnetic_fields_ac ----
# (case, spec factory, closed-form check, compared complex fields) per AC model
_AC_MODELS = (('skin_slab', magnetic_ac_selftests._skin_specs, magnetic_ac_selftests._check_skin, ('A', 'B_y')),
              ('solid_strip', magnetic_ac_selftests._strip_specs, magnetic_ac_selftests._check_strip,
               ('A', 'J_conduction')))
_AC_BACKENDS = ('cpu', 'gpu')


def _ac_run_name(model, backend):
    return f'{model}_{backend}'


def _ac_backend_runs():
    run = magnetic_selftests._RUN
    return {_ac_run_name(model, backend): _on_backend(factory()[run], backend)
            for model, factory, _check, _fields in _AC_MODELS for backend in _AC_BACKENDS}


def _check_ac_backends(results_by_name):
    """Largest of the cpu closed-form errors and (gpu runs present) the cpu vs gpu complex field / eddy_loss
    differences; raises unless each cpu run recorded a cpu sparse_direct linear solver."""
    run = magnetic_selftests._RUN
    errors = {}
    for model, _factory, check, field_names in _AC_MODELS:
        cpu_name, gpu_name = (_ac_run_name(model, backend) for backend in _AC_BACKENDS)
        role = solver_role(results_by_name[cpu_name], _MAGNETIC)
        if role.get('backend') != 'cpu' or role.get('type') != 'sparse_direct':
            raise ValueError(f"run '{cpu_name}': expected a cpu sparse_direct linear solver, summary 'solvers' records "
                             f'{role!r}')
        errors[model] = _scaled(check({run: results_by_name[cpu_name]}))
        if gpu_name not in results_by_name:   # absent without a usable GPU (api.selftest skipped it)
            continue
        for field_name in field_names:
            cpu_field, gpu_field = (magnetic_ac_selftests._complex_field(results_by_name[name], field_name)[0]
                                    for name in (cpu_name, gpu_name))
            scale = float(np.max(np.abs(gpu_field)))
            errors[f'{model}_{field_name}_cpu_gpu'] = float(np.max(np.abs(cpu_field - gpu_field))) / scale
        losses = [magnetic_selftests._derived_value(results_by_name[name], 'P') for name in (cpu_name, gpu_name)]
        errors[f'{model}_eddy_loss_cpu_gpu'] = relative_error(*losses)
    finite = all(np.isfinite(value) for value in errors.values())
    error = max(errors.values()) if finite else float('nan')
    gpu_runs = [name for name in results_by_name if name.endswith('_gpu')]
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'gpu_runs': gpu_runs,
                        'cpu_solvers': {model: solver_role(results_by_name[_ac_run_name(model, 'cpu')], _MAGNETIC)
                                        for model, *_rest in _AC_MODELS}}}


AC_BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='magnet_ac_cpu_gpu',
        label='Magnetic fields AC: CPU backend vs GPU',
        description=('magnetic_fields_ac on the 1 x 0.1 mm window of uniform 10 um cells (symmetry y faces, 1D in x, '
                     'copper sigma_dc 5.8e7 S/m), on solver_defaults.backend cpu (numpy fields and real window system, '
                     'the host SuperLU of the Automatic sparse_direct linear solver) and gpu. skin_slab: copper fills '
                     'the window, fixed_potential A = 1e-6 T m on the first cell column, insulation x_hi, f = 440 kHz '
                     '(delta ~10 cells); exact complex tridiagonal of the free cells: A, B_y, q_eddy and eddy_loss '
                     '(pins the j omega sigma diagonal and its sign, the held-cell elimination on the host, the complex '
                     'B reconstruction, the loss). solid_strip: a solid conductor (drive current, 1 A at 30 deg) on the '
                     'band [0.4, 0.6] mm, insulation x faces, f = 1.75 MHz; exact dense bordered system: A, '
                     'J_conduction, voltage, eddy_loss, plus the circuit row (current = I) and Poynting (eddy_loss = '
                     'Re(V I*) / 2) identities to 1e-10 (pins the circuit row, the drive phase, the conductor results). '
                     f'The cpu runs against these exact solutions (each error scaled by its tolerance onto '
                     f'{_TOLERANCE:g}); the gpu runs: max complex A, B_y (slab) / A, J_conduction (strip) difference '
                     'relative to the gpu maximum and the eddy_loss relative difference. The solve is host float64 on '
                     'both backends, so the parity is the float32 field storage at most. Without a usable GPU the gpu '
                     'runs are skipped.'),
        specs=_ac_backend_runs, check=_check_ac_backends),
)

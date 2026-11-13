"""CPU vs GPU backend case of the DiffusionSystem physics (solver_plan S6 stage 2): heat and electric_currents models
solved on solver_defaults.backend 'cpu' (numpy / scipy: Automatic = host SuperLU, or a pcg node on numpy) against the
same models on 'gpu' and the analytic slab peak; registered by definition.register. api.selftest skips the gpu runs without a usable GPU."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role
from ..potential import selftests as potential_selftests
from . import selftests as heat_selftests

_SLAB_CPU = 'heated_slab_cpu_direct'
_POINT_RUNS = {'cpu': 'point_source_cpu_cg', 'gpu': 'point_source_gpu_cg'}
_DIVIDER_RUNS = {'cpu': 'divider_cpu_direct', 'gpu': 'divider_gpu_cg'}
_CURRENTS_POTENTIAL_FIELD = 'currents.V'
_TOLERANCE = 1e-6   # float32 fields of solves converged to 1e-12 (pcg) or exact (SuperLU): agreement ~1e-7


def _on_backend(spec, backend):
    spec.solver_defaults.backend = backend
    return spec


def _automatic(spec):
    """spec without solver nodes or assignments (every role Automatic)."""
    spec.solvers = []
    for physics in spec.physics:
        physics.solvers = {}
    return spec


def _backend_runs():
    """The heated slab template on the cpu (Automatic linear solver), the exact-CG point source on both backends and
    the terminal divider (cpu: Automatic SuperLU, gpu: its exact pcg node)."""
    point_source = heat_selftests._point_source_convection_runs
    divider = potential_selftests._terminal_divider_specs
    run = heat_selftests.RUN_NAME
    return {
        _SLAB_CPU: _on_backend(heat_selftests.heated_slab_spec(), 'cpu'),
        _POINT_RUNS['cpu']: _on_backend(point_source()[run], 'cpu'),
        _POINT_RUNS['gpu']: _on_backend(point_source()[run], 'gpu'),
        _DIVIDER_RUNS['cpu']: _on_backend(_automatic(divider()[potential_selftests._RUN]), 'cpu'),
        _DIVIDER_RUNS['gpu']: _on_backend(divider()[potential_selftests._RUN], 'gpu'),   # its exact pcg node
    }


def _field(results, name):
    return np.asarray(results.field(name).values, dtype=np.float64)


def _check_backends(results_by_name):
    """Largest of: the cpu slab peak rise vs Q L^2 / (8 k), max |T_cpu - T_gpu| / max rise and the energy balance of
    both point-source runs (scaled onto _TOLERANCE), max |V_cpu - V_gpu| / 1 V and the terminal current difference of
    the divider; raises when the cpu runs did not record backend 'cpu' (and SuperLU for the Automatic ones)."""
    for run, physics, solver_type in ((_SLAB_CPU, heat_selftests.PHYSICS_NAME, 'sparse_direct'),
                                      (_POINT_RUNS['cpu'], heat_selftests.PHYSICS_NAME, 'pcg'),
                                      (_DIVIDER_RUNS['cpu'], 'currents', 'sparse_direct')):
        role = solver_role(results_by_name[run], physics)
        if role.get('backend') != 'cpu' or role.get('type') != solver_type:
            raise ValueError(f"run '{run}': expected a cpu {solver_type} linear solver, summary 'solvers' records "
                             f'{role!r}')
    slab = heat_selftests._check_heated_slab({heat_selftests.RUN_NAME: results_by_name[_SLAB_CPU]})
    if _POINT_RUNS['gpu'] not in results_by_name:   # no usable GPU: api.selftest skipped the gpu runs
        return _check_cpu_only(results_by_name, slab)
    point_balance = {backend: _point_balance_error(results_by_name[run]) for backend, run in _POINT_RUNS.items()}
    temperature = {backend: _field(results_by_name[run], heat_selftests.TEMPERATURE_FIELD)
                   for backend, run in _POINT_RUNS.items()}
    rise = float(np.max(temperature['gpu']) - heat_selftests.DEFAULT_T0)
    point_error = float(np.max(np.abs(temperature['cpu'] - temperature['gpu']))) / rise
    potential = {backend: _field(results_by_name[run], _CURRENTS_POTENTIAL_FIELD)
                 for backend, run in _DIVIDER_RUNS.items()}
    potential_error = float(np.max(np.abs(potential['cpu'] - potential['gpu']))) / potential_selftests._VOLTAGE
    current = {backend: float(results_by_name[run].summary()['derived']['I_left'])
               for backend, run in _DIVIDER_RUNS.items()}
    current_error = abs(current['cpu'] - current['gpu']) / abs(current['gpu'])
    errors = {'slab_peak': slab['error'], 'point_source_T': point_error, 'divider_V': potential_error,
              'divider_current': current_error, 'point_source_balance_cpu': point_balance['cpu'],
              'point_source_balance_gpu': point_balance['gpu']}
    finite = all(np.isfinite(value) for value in errors.values())   # max() alone would drop a NaN
    error = max(errors.values()) if finite else float('nan')
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'slab_peak_rise_K': slab['measured'], 'divider_current_A_per_m': current,
                        'cpu_solvers': {run: solver_role(results_by_name[run], physics) for run, physics in (
                            (_SLAB_CPU, heat_selftests.PHYSICS_NAME), (_DIVIDER_RUNS['cpu'], 'currents'))}}}


def _point_balance_error(results):
    """Energy-balance error of one point-source run (heat _check_point_source_convection) scaled onto _TOLERANCE."""
    point = heat_selftests._check_point_source_convection({heat_selftests.RUN_NAME: results})
    return point['error'] / point['tolerance'] * _TOLERANCE


def _check_cpu_only(results_by_name, slab):
    """Without the gpu runs: the slab peak plus the point-source and divider analytic checks of the cpu runs, each
    error scaled by its own tolerance onto _TOLERANCE."""
    divider = potential_selftests._check_terminal_divider(
        {potential_selftests._RUN: results_by_name[_DIVIDER_RUNS['cpu']]})
    errors = {'slab_peak': slab['error'],
              'point_source_balance_cpu': _point_balance_error(results_by_name[_POINT_RUNS['cpu']]),
              'divider_analytic': divider['error'] / divider['tolerance'] * _TOLERANCE}
    finite = all(np.isfinite(value) for value in errors.values())
    error = max(errors.values()) if finite else float('nan')
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'gpu_runs': 'skipped (no usable GPU)', 'slab_peak_rise_K': slab['measured']}}


# ---- diffusion_fused_reassembly (cpu, gpu) ----
_FUSED_TOLERANCE = 1e-12   # float64: the fused kernels differ from build_laplacian only by FMA / summation order
_FUSED_REFERENCE = 0.3     # unknown reference of the face rhs terms
_FUSED_SEED = 7


def _fused_domain(backend, widths):
    """Padding-free Domain (2 widths) or Domain3D (3 widths) of these host cell widths on `backend`."""
    from ...core.constants import FACES, FACES_3D
    if len(widths) == 3:
        from ...core.domain3d import Domain3D
        return Domain3D.from_widths(*widths, {face: 0 for face in FACES_3D}, size_x=float(np.sum(widths[0])),
                                    size_y=float(np.sum(widths[1])), size_z=float(np.sum(widths[2])),
                                    backend=backend)
    from ...core.domain import Domain
    from ...core.material_grid import MaterialGrid
    from ...core.mesh import NonUniformGrid2D
    grid = NonUniformGrid2D(*widths, verbose=False)
    return Domain(grid, MaterialGrid(grid.nx, grid.ny), size_x=grid.size_x, size_y=grid.size_y, non_uniform=True,
                  npml_sides={face: 0 for face in FACES}, backend=backend)


def _fused_system(domain, fused, axisymmetric=False, azimuthal_order=0):
    """Prepared float64 DiffusionSystem on domain with the fused re-assembly on / off."""
    from ...core.fv_diffusion import DiffusionSystem
    system = DiffusionSystem(domain, 'float64', 1e-12, 1000, lambda: 'fused re-assembly check', 'u',
                             'needs an anchor', 'hold a cell', axisymmetric=axisymmetric,
                             azimuthal_order=azimuthal_order)
    system.fused_reassembly = fused
    system.prepare(domain)
    return system


def _fused_inputs(system, rng, three_d):
    """(coefficient sets of two assemblies (same c > 0 cells), bcs, capacity or None) and the held box / loss set on
    system (identical for every system of one window: rng is reseeded per system by the caller)."""
    from ...core.fv_diffusion import BCSide, ScalarBoundaryConfig
    xp = system.xp
    shape = tuple(system.window_shape)
    held = np.zeros(shape, dtype=bool)
    held[(slice(1, 3), slice(shape[1] - 3, shape[1] - 1)) + ((slice(None),) if three_d else ())] = True
    system.set_fixed(xp.asarray(held), xp.asarray(rng.uniform(-1.0, 1.0, shape)))
    loss = np.zeros(shape, dtype=bool)
    loss[-2:] = True
    system.add_diagonal_loss(xp.asarray(loss), xp.asarray(rng.uniform(1e9, 2e9, shape)), xp.asarray(0.4),
                             _FUSED_REFERENCE)
    zero = (3, 1) + ((1,) if three_d else ())   # a pinned cell: every coefficient 0, no loss or capacity
    sets = []
    for _ in range(2):
        coefficients = [rng.uniform(0.5, 2.0, shape) for _ in range(3 if three_d else 2)]
        for values in coefficients:
            values[zero] = 0.0
        sets.append([xp.asarray(values) for values in coefficients])
    faces = dict(x_lo=BCSide('dirichlet', value=1.5, conductance_factor=2.0),
                 x_hi=BCSide('robin', h=3.0, external=0.7), y_lo=BCSide('neumann', value=0.25),
                 y_hi=BCSide('dirichlet', value=-0.5))
    if three_d:
        faces.update(z_lo=BCSide('robin', h=0.5, external=-0.2), z_hi=BCSide('neumann', value=-0.1))
    capacity = xp.asarray(rng.uniform(0.1, 1.0, shape)) if three_d else None
    if capacity is not None:
        capacity[zero] = 0.0
    return sets, ScalarBoundaryConfig(**faces), capacity


def _fused_differences(fused, reference):
    """{array: max |fused - reference| / max |reference|} over everything the two assemblies store."""
    from ...core.backend import to_host

    def pairs():
        yield 'operator', fused.operator_full.data, reference.operator_full.data
        yield 'solve_operator', fused.A_solve.data, reference.A_solve.data
        for name in ('inverse_diagonal', 'rhs_offset', 'rhs_drive', 'fixed_values', 'free', '_conduction_diagonal',
                     'unknown'):
            yield name, getattr(fused, name), getattr(reference, name)
        for face, (_, conductance, rhs) in reference.face_terms.items():
            yield f'{face}_conductance', fused.face_terms[face][1], conductance
            yield f'{face}_rhs', fused.face_terms[face][2], rhs
        for axis, conductance in enumerate(reference.face_conductances):
            yield f'interior_{"xyz"[axis]}', fused.face_conductances[axis], conductance

    differences = {}
    for name, fused_values, reference_values in pairs():
        fused_values = np.asarray(to_host(fused_values), dtype=np.float64)
        reference_values = np.asarray(to_host(reference_values), dtype=np.float64)
        if fused_values.shape != reference_values.shape:
            differences[name] = float('inf')
            continue
        scale = max(float(np.max(np.abs(reference_values), initial=0.0)), 1e-300)
        differences[name] = float(np.max(np.abs(fused_values - reference_values), initial=0.0)) / scale
    return differences


def _fused_window(checks, backend, label, widths, axisymmetric=False, azimuthal_order=0):
    """One window: fused vs reference systems through two assemblies; differences and counters into checks."""
    three_d = len(widths) == 3
    domain = _fused_domain(backend, widths)
    systems = {}
    for fused in (True, False):
        system = _fused_system(domain, fused, axisymmetric, azimuthal_order)
        sets, bcs, capacity = _fused_inputs(system, np.random.default_rng(_FUSED_SEED), three_d)
        for number, coefficients in enumerate(sets):
            if number:
                system.mark_coefficients_changed()
            c_zz = coefficients[2] if three_d else None
            system.assemble(coefficients[0], coefficients[1], None, bcs, _FUSED_REFERENCE, capacity, c_zz=c_zz)
        systems[fused] = system
    for name, difference in _fused_differences(systems[True], systems[False]).items():
        checks.add_error(f'{backend}_{label}_{name}', difference, _FUSED_TOLERANCE)
    checks.add(f'{backend}_{label}_counters', systems[True].n_fused_assemblies == 1
               and systems[False].n_fused_assemblies == 0 and systems[True].n_anchor_shortcuts == 1,
               value={'fused': systems[True].n_fused_assemblies, 'reference': systems[False].n_fused_assemblies,
                      'anchor_shortcuts': systems[True].n_anchor_shortcuts})


def _floating_after_coefficient_change():
    """cpu: a coefficient-only change cutting the window at column 3 (x_lo the only anchor) must still raise."""
    from ...core.fv_diffusion import BCSide, ScalarBoundaryConfig
    domain = _fused_domain('cpu', (np.full(8, 1e-5), np.full(5, 1e-5)))
    system = _fused_system(domain, True)
    bcs = ScalarBoundaryConfig(x_lo=BCSide('dirichlet', value=1.0))
    ones = np.ones(system.window_shape)
    system.assemble(ones, ones, None, bcs, 0.0)
    cut = ones.copy()
    cut[3] = 0.0
    system.mark_coefficients_changed()
    system.assemble(cut, cut, None, bcs, 0.0)


def _check_fused_reassembly():
    """Sub-checks of every window on the cpu (and the gpu when usable) plus the floating-region refusal."""
    from ...core.backend import gpu_available
    from ...model.selftest_helpers import SubChecks
    checks = SubChecks()
    graded_x = 1e-5 * 1.15 ** np.arange(9)
    graded_y = 2e-5 * 0.9 ** np.arange(7)
    backends = ('cpu', 'gpu') if gpu_available() else ('cpu',)
    for backend in backends:
        _fused_window(checks, backend, 'planar', (graded_x, graded_y))
        _fused_window(checks, backend, 'rz_order1', (graded_x, graded_y), axisymmetric=True, azimuthal_order=1)
        _fused_window(checks, backend, '3d', (graded_x[:6], graded_y[:5], 1.5e-5 * 1.1 ** np.arange(4)))
    checks.add_raises('cpu_floating_region_after_coefficient_change', _floating_after_coefficient_change)
    return checks.result(backends=list(backends))


BACKEND_SELFTEST_CASES = (
    SelfTestCase(
        name='diffusion_cpu_gpu',
        label='Heat and electric currents: CPU backend vs GPU',
        description=('The DiffusionSystem physics on solver_defaults.backend cpu (numpy arrays, scipy CSR): the '
                     'heated_slab template with the Automatic cpu linear solver (host SuperLU) against the peak rise '
                     'Q L^2 / (8 k); the point heat source under convection (exact pcg node on numpy) and the '
                     'electric_currents terminal divider (cpu: Automatic SuperLU, gpu: exact pcg) against the same '
                     f'models on the gpu backend (max field difference, terminal current). Tolerance {_TOLERANCE:g}: '
                     'float32 fields of solves converged to 1e-12 or exact. Both point-source runs are also checked '
                     f'against their exact discrete energy balance ({heat_selftests.BOX_SIZE * 1e3:g} mm square, k_xx '
                     f'= {heat_selftests.THERMAL_CONDUCTIVITY:g}, k_yy = {heat_selftests.POINT_K_YY:g} W/(m*K), point '
                     f'P = {heat_selftests.POINT_POWER:g} W/m, inward heat_flux on x_lo, convection h = '
                     f'{heat_selftests.POINT_H:g} W/(m^2*K) on x_hi, y_lo, y_hi: the Robin outflow, sum (T_cell - '
                     'T_ext) w / (1 / h + w / (2 k_normal)), equals P + flux * L; tolerance '
                     f'{heat_selftests.EXACT_IDENTITY_TOLERANCE:g}, scaled onto {_TOLERANCE:g}; h w / (2 k) = 0.25 '
                     'on x_hi, so a wrong half-cell series term or k_normal fails; this pins the point source P / '
                     'cell area, the Neumann flux and the Robin faces in 2D). Without a usable GPU the gpu runs are '
                     'skipped and the cpu runs are checked against their closed forms instead (each error scaled '
                     'by its own tolerance).'),
        specs=_backend_runs, check=_check_backends),
    SelfTestCase(
        name='diffusion_fused_reassembly',
        label='DiffusionSystem: fused re-assembly kernels vs build_laplacian (CPU, GPU)',
        description=('Scripted float64 DiffusionSystems on padding-free graded windows, each assembled twice (the '
                     'second after mark_coefficients_changed with new coefficient values, same c > 0 cells): the '
                     'fused re-assembly (core.fv_diffusion_fused: numpy twins on the cpu, CUDA kernels on the gpu) '
                     'against the build_laplacian path. Windows: 2D Cartesian (held box, a pinned c = 0 cell, '
                     'dirichlet x2 factor / robin / neumann / dirichlet faces, a volumetric loss), 2D (r, z) at '
                     'azimuthal order 1 (zero-area axis face), 3D Cartesian with C/dt and robin / neumann z faces. '
                     'Compared: A_eff and A_solve values, Jacobi diagonal, rhs offset / drive, fixed values, free '
                     'mask, conduction diagonal, the unknown, every face term and interior face conductance (max '
                     f'difference / max |reference|, tolerance {_FUSED_TOLERANCE:g}), plus the fused / anchor-shortcut '
                     'counters, and on the cpu that a coefficient-only change opening a floating region still raises. '
                     'The gpu half is skipped without a usable GPU.'),
        specs=lambda: {}, check=lambda _results: _check_fused_reassembly(), backends=('cpu', 'gpu')),
)

"""CPU vs GPU backend cases of laminar_flow (solver_plan S6, core.kernels twins of the lattice step incl. Bouzidi
links, the lattice <-> SI kernels and the convection transform): the analytic fluid cases on solver_defaults.backend
'cpu' (numpy / numba twins) and 'gpu' through physics/em/selftests_backend.backend_parity_case, plus the CPU case
fluid_step_numba (numba vs numpy step twins); registered by definition.register."""
import dataclasses

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cases_named
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)
from ..em.selftests_backend import PARITY_TOLERANCE, backend_parity_case
from .selftests import FLUID_CASES, FLUID_TYPE_NAME, startup_case

_FLUID = 'fluid'
_RUN = 'run'
# Bouzidi channel: force-driven periodic channel of 1 um cells between solid rectangles drawn off the cell faces (the
# conformal on only; walls 0.2 / 0.8 of a link from the nearest fluid centres)
_CELL = 1e-6
_CHANNEL_SIZE = 20e-6       # window height (y)
_CHANNEL_LENGTH = 8e-6      # periodic x: the flow is x-invariant; 8 columns keep the periodic-seam links
_WALL_LO, _WALL_HI = 1.3e-6, 18.3e-6
_VISCOSITY, _DENSITY = 1e-3, 1000.0
_PEAK_VELOCITY = 5e-3
_DRIVE = 8.0 * _VISCOSITY * _PEAK_VELOCITY / (_WALL_HI - _WALL_LO) ** 2   # N/m^3
_BOUZIDI_TOLERANCE = 1e-2   # linear Bouzidi + TRT: O(dx^2) slip at 17 cells


def _bouzidi_specs():
    """'run': the periodic channel between the drawn walls, conformal_boundaries on."""
    walls = [GeometrySpec('wall_lo', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _CHANNEL_LENGTH, 'y1': _WALL_LO}),
             GeometrySpec('wall_hi', 'rectangle', {'x0': 0.0, 'y0': _WALL_HI, 'x1': _CHANNEL_LENGTH,
                                                   'y1': _CHANNEL_SIZE})]
    return {_RUN: ModelSpec(
        title='Self-test: Poiseuille channel between drawn walls (laminar flow, Bouzidi links)',
        domain=DomainSpec(size_x=_CHANNEL_LENGTH, size_y=_CHANNEL_SIZE),
        geometry=walls,
        materials=[MaterialSpec('liquid', properties={'viscosity': _VISCOSITY, 'rho': _DENSITY}, selection=['all']),
                   MaterialSpec('solid', properties={'viscosity': 0.0, 'rho': _DENSITY},
                                selection=['wall_lo', 'wall_hi'])],
        physics=[PhysicsSpec(_FLUID, FLUID_TYPE_NAME, settings={'tolerance': 1e-6}, features=[
            FeatureSpec('periodic_x', 'periodic', ['x_lo', 'x_hi']),
            FeatureSpec('walls', 'wall', ['y_lo', 'y_hi']),
            FeatureSpec('drive', 'body_force', ['all'], {'f_x': _DRIVE, 'f_y': 0.0})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
        study=StudySpec(type='stationary'))}


def _check_bouzidi(results_by_name):
    """u_x against G (y - y0) (y1 - y) / (2 mu) on the fluid rows, u_y and the solid rows against 0, over the peak."""
    results = results_by_name[_RUN]
    field = results.field(f'{_FLUID}.u_x')
    velocity_x = np.asarray(field.values, dtype=np.float64)
    velocity_y = np.asarray(results.field(f'{_FLUID}.u_y').values, dtype=np.float64)
    y = np.asarray(field.y_centers, dtype=np.float64)
    fluid = (y > _WALL_LO) & (y < _WALL_HI)
    expected = _DRIVE / (2.0 * _VISCOSITY) * (y - _WALL_LO) * (_WALL_HI - y)
    errors = {'u_x': float(np.max(np.abs(velocity_x[:, fluid] - expected[None, fluid]))) / _PEAK_VELOCITY,
              'u_y': float(np.max(np.abs(velocity_y))) / _PEAK_VELOCITY,
              'solid_u_x': float(np.max(np.abs(velocity_x[:, ~fluid]))) / _PEAK_VELOCITY}
    return {'measured': float(np.max(velocity_x)), 'expected': float(np.max(np.where(fluid, expected, 0.0))),
            'error': max(errors.values()), 'tolerance': _BOUZIDI_TOLERANCE,
            'details': {'errors': errors, 'walls_m': [_WALL_LO, _WALL_HI]}}


_BOUZIDI_CASE = SelfTestCase(
    name='fluid_bouzidi_channel', label='Laminar flow: Poiseuille channel between drawn walls (Bouzidi)',
    description=(f'laminar_flow force-driven periodic {_CHANNEL_LENGTH * 1e6:g} x {_CHANNEL_SIZE * 1e6:g} um channel of '
                 '1 um cells, solids drawn below y = 1.3 um and above y = 18.3 um, conformal_boundaries on (Bouzidi '
                 'links): u_x against the parabola on the drawn walls, u_y and the solid cells against 0, relative to '
                 f'the peak; tolerance {_BOUZIDI_TOLERANCE:g}; measured = max u_x. Not yet confirmed by a run.'),
    specs=_bouzidi_specs, check=_check_bouzidi)


# numba vs numpy step twins: random lattice states on a small window with solid cells, per face set (x_lo, x_hi, y_lo,
# y_hi) covering every face rule and both corner precedences, with and without Bouzidi links
_NUMBA_FACE_SETS = (('periodic', 'periodic', 'wall', 'wall'), ('velocity', 'pressure', 'slip', 'wall'),
                    ('pressure', 'pressure', 'velocity', 'pressure'), ('slip', 'pressure', 'periodic', 'periodic'))
_NUMBA_WINDOW = (11, 9)
_NUMBA_TOLERANCE = 1e-5     # relative to the peak output: float32 round-off of the same operation order
_SPEED_WINDOW = (128, 128)  # force-driven periodic channel for the ms-per-step measurement (details only)
_SPEED_STEPS = {'numpy': 3, 'numba': 100}


def _random_step_arguments(rng, kinds, window, lattice, bouzidi, uniform=False):
    """(g_in, the outputs, arguments(source, outputs)) of one random lbm_trt_step call at STORE_PUBLISHED; outputs = the
    target populations, lattice macros and the four full-grid SI fields (window at (1, 1)); uniform: relaxation and
    force read at stride 0 (1-element rates, element 0 of the forces)."""
    wx, wy = window
    fluid = rng.random(window) > 0.2
    g_in = (rng.standard_normal((lattice.N_DIRECTIONS, wx, wy)) * 1e-2 * fluid).astype(np.float32)
    omega_plus = rng.uniform(0.6, 1.8, (1, 1) if uniform else window).astype(np.float32)
    omega_minus = (1.0 / (0.5 + lattice.MAGIC_LAMBDA / (1.0 / omega_plus - 0.5))).astype(np.float32)
    force_x, force_y = (rng.standard_normal((2, wx, wy)) * 1e-4).astype(np.float32)
    mass_density = rng.uniform(500.0, 1500.0, window).astype(np.float32)
    face_kind = np.array([{name: code for code, name in lattice.FACE_KIND_NAMES.items()}[kind] for kind in kinds],
                         dtype=np.int32)
    face_length = max(wx, wy)
    face_values = (rng.standard_normal((4, 3, face_length)) * 1e-2).astype(np.float32)
    wall_row, wall_fraction = np.full(1, -1, dtype=np.int32), np.zeros(lattice.N_DIRECTIONS - 1, dtype=np.float32)
    if bouzidi:   # rows on fluid cells next to an in-window solid, random link fractions on both sides of 1/2
        padded_solid = np.pad(~fluid, 1)
        near_solid = np.zeros(window, dtype=bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                near_solid |= padded_solid[1 + di:1 + di + wx, 1 + dj:1 + dj + wy]
        owners = np.flatnonzero(fluid & near_solid)
        wall_row = np.full(window, -1, dtype=np.int32)
        wall_row.ravel()[owners] = np.arange(owners.size, dtype=np.int32)
        if owners.size:   # else the zeros stand-in, as FluidPhysics passes
            wall_fraction = rng.uniform(0.05, 0.95, (owners.size, lattice.N_DIRECTIONS - 1)).astype(np.float32)
    full_grid = (wx + 2, wy + 2)   # published SI fields: the ring around the window stays untouched
    outputs = (rng.standard_normal((lattice.N_DIRECTIONS, wx, wy)).astype(np.float32),   # untouched in solids
               (rng.standard_normal(window) * 1e-2).astype(np.float32),                  # previous u_x, u_y
               (rng.standard_normal(window) * 1e-2).astype(np.float32),
               np.zeros(window, dtype=np.float32),
               *(rng.standard_normal(full_grid).astype(np.float32) for _ in range(4)))   # u_x, u_y, u_abs, pressure
    stride = np.int32(0 if uniform else 1)

    def arguments(source, step_outputs):
        g_out, velocity_x, velocity_y, deviation, *published = step_outputs
        return (source, g_out, velocity_x, velocity_y, deviation, fluid, omega_plus, omega_minus, force_x, force_y,
                face_kind, face_values, np.int32(wx), np.int32(wy), np.int32(face_length), wall_row, wall_fraction,
                stride, stride, np.int32(lattice.STORE_PUBLISHED), *published, mass_density, np.int32(full_grid[1]),
                np.int32(1), np.int32(1), np.float32(3e-3), np.float32(2e-6))
    return g_in, outputs, arguments


def _step_ms(step, g_in, outputs, arguments, n_steps):
    """Wall time per call [ms] of n_steps ping-pong steps (populations swapped like FluidPhysics)."""
    import time
    source, target = g_in.copy(), outputs[0].copy()
    rest = tuple(array.copy() for array in outputs[1:])
    start = time.perf_counter()
    for _ in range(n_steps):
        step(*arguments(source, (target, *rest)))
        source, target = target, source
    return 1e3 * (time.perf_counter() - start) / n_steps


def _registered_numba_step(kernel):
    """Host call of a registered raw twin (its numba twin, compiled once per process and shared with fluid_cpu_gpu)."""
    return lambda *arguments: kernel(None, None, arguments)


def _check_step_numba(results_by_name):
    """numba vs numpy lbm_trt_step ('single', with / without Bouzidi) on random states; ms per step on _SPEED_WINDOW."""
    from ...core.kernels import numba_available
    from . import lattice
    if not numba_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _NUMBA_TOLERANCE,
                'details': {'numba': 'not installed (or OPEN_SOL_CPU_KERNELS=numpy): nothing compared'}}
    import numba
    rng = np.random.default_rng(20261018)
    errors, worst = {}, 0.0
    kernels = {bouzidi: lattice.step_kernel('single', bouzidi) for bouzidi in (False, True)}
    steps = {bouzidi: (kernel.cpu, _registered_numba_step(kernel)) for bouzidi, kernel in kernels.items()}
    for bouzidi, (numpy_step, numba_step) in steps.items():
        for set_index, kinds in enumerate(_NUMBA_FACE_SETS):
            uniform = set_index == 0   # relaxation / force at stride 0 (the gpu arguments of one fluid, no force)
            g_in, outputs, arguments = _random_step_arguments(rng, kinds, _NUMBA_WINDOW, lattice, bouzidi, uniform)
            numpy_out, numba_out = [array.copy() for array in outputs], [array.copy() for array in outputs]
            numpy_step(*arguments(g_in, numpy_out))
            numba_step(*arguments(g_in, numba_out))
            difference = max(float(np.max(np.abs(a.astype(np.float64) - b))) for a, b in zip(numpy_out, numba_out))
            peak = max(float(np.max(np.abs(a))) for a in numpy_out)
            error = difference / peak if np.isfinite(difference) and peak > 0.0 else float('inf')
            errors[f"{'bouzidi' if bouzidi else 'halfway'}:{'/'.join(kinds)}{':stride0' if uniform else ''}"] = error
            worst = max(worst, error)
    # a numba twin that failed falls back to numpy for good (the comparison above was numpy vs numpy)
    host_kinds = {f"single{'_bouzidi' if bouzidi else ''}": kernel.host_kind for bouzidi, kernel in kernels.items()}
    if any(kind != 'numba' for kind in host_kinds.values()):
        worst = float('inf')
    g_in, outputs, arguments = _random_step_arguments(rng, ('periodic', 'periodic', 'wall', 'wall'), _SPEED_WINDOW,
                                                      lattice, False)
    numpy_step, numba_step = steps[False]
    _step_ms(numba_step, g_in, outputs, arguments, 2)   # warm-up (compiled above)
    numpy_ms, numba_ms = (_step_ms(numpy_step, g_in, outputs, arguments, _SPEED_STEPS['numpy']),
                          _step_ms(numba_step, g_in, outputs, arguments, _SPEED_STEPS['numba']))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _NUMBA_TOLERANCE,
            'details': {'relative_differences': errors, 'speed_window': list(_SPEED_WINDOW),
                        'numpy_ms_per_step': numpy_ms, 'numba_ms_per_step': numba_ms,
                        'numba_speedup': numpy_ms / numba_ms if numba_ms > 0.0 else None,
                        'numba_threads': int(numba.get_num_threads()), 'registered_host_kinds': host_kinds}}


_STEP_NUMBA_CASE = SelfTestCase(
    name='fluid_step_numba', label='Laminar flow: numba vs numpy lattice step (CPU)',
    description=('The numba twin of the laminar_flow lattice step (fluid.lbm_trt_step_single, halfway and Bouzidi) '
                 f'against its numpy twin on random {_NUMBA_WINDOW[0]} x {_NUMBA_WINDOW[1]} states with solid cells, '
                 'for face sets covering wall, slip, velocity, pressure (TRT shear term) and periodic faces and both '
                 'corner precedences, at store mode published (SI fields fused in) and on the first face set with the '
                 'relaxation and force read at stride 0; measured = error = the worst max |numba - numpy| over the '
                 'populations, velocities, density deviation and published fields relative to their peak, tolerance '
                 f'{_NUMBA_TOLERANCE:g} (the '
                 'registered twins; error inf when a numba twin fell back to numpy). Details: '
                 f'ms per step of both on a {_SPEED_WINDOW[0]} x {_SPEED_WINDOW[1]} channel. Without numba nothing is '
                 'compared (error 0).'),
    specs=lambda: {}, check=_check_step_numba, max_wall_time=600.0, heavy=True, backends=('cpu',))


def _fluid_parity_case(name, label, wrapped_cases, max_wall_time):
    """backend_parity_case of the fluid cases, with a description of the twinned fluid kernels."""
    case = backend_parity_case(name, label, wrapped_cases, heavy=True, max_wall_time=max_wall_time)
    case_names = ', '.join(wrapped.name for wrapped in wrapped_cases)
    return dataclasses.replace(case, description=(
        f'The analytic laminar_flow cases {case_names} run on solver_defaults.backend cpu (core.kernels twins of '
        'the lattice step with its face rules and Bouzidi links (numba when installed, else numpy), the numpy '
        'lattice <-> SI kernels and convection transform) and on gpu. Each case\'s own check grades the cpu runs, and its measured value must match the gpu '
        f"runs' within a relative {PARITY_TOLERANCE:g} (float32 on both; steady marches stop within their tolerance "
        'and the float32 floor, ~1e-5). error = the worst of cpu error / case tolerance and difference / parity '
        'tolerance, tolerance 1. Without a usable GPU only the cpu checks count.'))


FLUID_BACKEND_SELFTEST_CASES = (
    _fluid_parity_case(
        'fluid_cpu_gpu', 'Laminar flow: CPU backend vs GPU',
        # periodic x, walls, Guo force; pressure faces with the TRT shear term (FLUID_CASES incl. the cases run only
        # inside these parities)
        (*cases_named(FLUID_CASES, 'fluid_poiseuille_force', 'fluid_poiseuille_pressure'),
         startup_case('single'),                        # transient stepping, dt rescaling
         _BOUZIDI_CASE),                                # interpolated wall links
        max_wall_time=1800.0),
    _fluid_parity_case(
        'fluid_heat_convection_cpu_gpu', 'Laminar flow + heat: CPU backend vs GPU (convection)',
        cases_named(FLUID_CASES, 'fluid_heat_convection_plug'),  # slip faces, velocity inlet, convection, heat
        max_wall_time=3600.0),
    _STEP_NUMBA_CASE,
)

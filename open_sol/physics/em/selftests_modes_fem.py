"""Self-tests of em_modes on triangle meshes (mesh plan stage T3: fem_modes, solvers/modes_fem): CPU cases on fem_modes
directly against closed forms (walled slab, PML leaky slab) and cpu-backend model runs (PEC circular pipe, the S7.8
TFLN rib as the S7.5 gain measurement, a heated strip with heat on a tensor mesh and em_modes on triangles)."""
import math

import numpy as np

from ...core.constants import C0, FACES
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import skip_without_cudss
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec, SolverSpec, StudySpec)
from .selftests import normalized_result

_TYPE = 'em_modes'      # = definition_modes.EM_MODES.name (definition_modes imports this module)
_PHYSICS = 'modes'
_TABLE = 'table'
_N_EFF_COLUMN = 1       # mode_table row index of n_eff


def _report(normalized, details):
    """Self-test result of {item: error / its tolerance}: measured = error = the worst item, tolerance 1."""
    return normalized_result(normalized, dict(details))


def _bisect(function, low, high, steps=200):
    """Root of a real function with a sign change on [low, high] by bisection."""
    value_low = function(low)
    for _ in range(steps):
        middle = 0.5 * (low + high)
        value = function(middle)
        if value_low * value <= 0.0:
            high = middle
        else:
            low, value_low = middle, value
    return 0.5 * (low + high)


def _highest_root(function, top, bottom, samples=4000):
    """Root of the first sign change of function scanning from top down to bottom (the fundamental mode's beta^2)."""
    grid = top - (top - bottom) * np.arange(1, samples) / samples
    values = [function(value) for value in grid]
    for index in range(1, len(grid)):
        if values[index - 1] * values[index] <= 0.0:
            return _bisect(function, grid[index], grid[index - 1])
    raise ValueError('no sign change of the dispersion relation in the scanned range')


def _sizing(h_max, pml_widths=None, pml_layers=8):
    """Explicit meshing.TriangleSizing of a uniform-size mesh with an optional PML ring {face: thickness}."""
    from ...model.meshing import TriangleSizing

    return TriangleSizing(h_max=h_max, h_min=h_max / 100.0, growth=1.3, curvature_factor=0.3, object_sizes={},
                          pml_widths=dict(pml_widths or {}), pml_layers=pml_layers, mesher='builtin')


def _direct_solve(spec, sizing, region_indices, background_index, faces, wavelength, n_modes, order,
                  n_eff_guess=None, cpml=None, mesh=None, eigen_solver=None):
    """(CrossSectionSolve, layout) of fem_modes on the triangulated spec (or the given mesh of it, triangulate being
    deterministic): isotropic eps = index^2 per region (else the background), mu 1, BoundaryConfig of faces {face:
    type} and cpml {BoundaryConfig attribute: value}; eigen_solver None = default (host SuperLU)."""
    from ...model.triangle_meshing import triangulate
    from .boundaries import BoundaryConfig
    from .fem_modes import FEMModeMaterials, fem_mode_layout, solve_fem_modes

    if mesh is None:
        mesh = triangulate(spec, {}, sizing=sizing)
    eps = np.full(mesh.n_elements, background_index ** 2, dtype=np.complex128)
    for name, index in region_indices.items():
        eps[mesh.region == mesh.region_names.index(name)] = index ** 2
    config = BoundaryConfig(**faces)
    for name, value in (cpml or {}).items():
        setattr(config, name, value)
    config._validate()
    materials = FEMModeMaterials(eps_xx=eps, eps_yy=eps, eps_zz=eps, mu=np.ones(mesh.n_elements, dtype=np.complex128),
                                 pec=np.zeros(mesh.n_elements, dtype=bool))
    layout = fem_mode_layout(mesh, order, config)
    return solve_fem_modes(layout, materials, 2.0 * math.pi * C0 / wavelength, n_modes, n_eff_guess=n_eff_guess,
                           eigen_solver=eigen_solver), layout


# ---- em_modes_fem_slab: a core layer x in [L - d, L + d] between pec x walls at 0 and 2 L, y-invariant: y faces pec
# keep the TE family (E_y(x) only), pmc the TM family (H_y(x), E_x, E_z). Exact even fundamental modes: TE kx tan(kx d) =
# gamma coth(gamma (L - d)), TM (kx / n1^2) tan(kx d) = (gamma / n2^2) tanh(gamma (L - d)), kx^2 = k0^2 n1^2 - beta^2,
# gamma^2 = beta^2 - k0^2 n2^2. Interfaces are mesh edges, so quadratic elements reach ~1e-8 at h = 15 nm.
_SLAB_WAVELENGTH = 1.55e-6
_SLAB_INDICES = (2.0, 1.45)          # core, cladding
_SLAB_HALF_WIDTH = 0.25e-6           # d [m]
_SLAB_WALL = 0.9e-6                  # L: wall distance from the core centre [m]
_SLAB_HEIGHT = 0.09e-6               # y extent [m]
_SLAB_H = 15e-9
_SLAB_LINEAR_SIZES = (30e-9, 15e-9)  # linear-element convergence pair
_SLAB_TOLERANCE = 1e-7               # relative n_eff (quadratic)
_LINEAR_MIN_ORDER = 1.75             # observed order in h of the linear n_eff error (expected 2)


def _slab_spec():
    spec = ModelSpec(title='Self-test: FEM cross-section slab',
                     domain=DomainSpec(size_x=2.0 * _SLAB_WALL, size_y=_SLAB_HEIGHT))
    spec.geometry = [GeometrySpec('core', 'rectangle', {'x0': _SLAB_WALL - _SLAB_HALF_WIDTH, 'y0': 0.0,
                                                        'x1': _SLAB_WALL + _SLAB_HALF_WIDTH, 'y1': _SLAB_HEIGHT})]
    return spec


def _slab_index(family):
    """Exact n_eff of the fundamental TE ('te') or TM ('tm') slab mode between the pec x walls."""
    k0 = 2.0 * math.pi / _SLAB_WAVELENGTH
    core, cladding = _SLAB_INDICES
    gap = _SLAB_WALL - _SLAB_HALF_WIDTH

    def mismatch(beta_squared):
        kx = math.sqrt(max(k0 ** 2 * core ** 2 - beta_squared, 0.0))
        gamma = math.sqrt(max(beta_squared - k0 ** 2 * cladding ** 2, 0.0))
        inside, outside = kx * math.sin(kx * _SLAB_HALF_WIDTH), gamma * math.cos(kx * _SLAB_HALF_WIDTH)
        if family == 'te':
            return inside * math.sinh(gamma * gap) - outside * math.cosh(gamma * gap)
        return inside / core ** 2 * math.cosh(gamma * gap) - outside / cladding ** 2 * math.sinh(gamma * gap)
    return math.sqrt(_highest_root(mismatch, (k0 * core) ** 2, (k0 * cladding) ** 2)) / k0


def _check_slab(results_by_name):
    """Quadratic TE / TM n_eff vs the exact roots (1e-7), |power| 1, te_fraction 0 / 1; linear TE order in h >= 1.75."""
    from ...model.triangle_meshing import triangulate

    normalized, details = {}, {}
    core, cladding = _SLAB_INDICES
    spec = _slab_spec()
    meshes = {size: triangulate(spec, {}, sizing=_sizing(size)) for size in {_SLAB_H, *_SLAB_LINEAR_SIZES}}
    te_index = _slab_index('te')
    for family, walls, te_fraction in (('te', 'pec', 0.0), ('tm', 'pmc', 1.0)):
        faces = {'x_lo': 'pec', 'x_hi': 'pec', 'y_lo': walls, 'y_hi': walls}
        solve, layout = _direct_solve(spec, None, {'core': core}, cladding, faces, _SLAB_WAVELENGTH, 2, 2,
                                      mesh=meshes[_SLAB_H])
        mode = solve.modes[0]
        expected = te_index if family == 'te' else _slab_index(family)
        normalized[f'{family} n_eff'] = abs(mode.n_eff.real / expected - 1.0) / _SLAB_TOLERANCE
        normalized[f'{family} power'] = abs(abs(mode.power) - 1.0) / 1e-9
        normalized[f'{family} te_fraction'] = abs(mode.te_fraction - te_fraction) / 1e-6
        details[family] = {'n_eff': [mode.n_eff.real, mode.n_eff.imag], 'n_eff_exact': expected,
                           'te_fraction': mode.te_fraction, 'power': mode.power, 'unknowns': layout.n_unknowns,
                           'elements': int(layout.mesh.n_elements)}
    errors = []
    for size in _SLAB_LINEAR_SIZES:
        solve, _ = _direct_solve(spec, None, {'core': core}, cladding,
                                 {'x_lo': 'pec', 'x_hi': 'pec', 'y_lo': 'pec', 'y_hi': 'pec'}, _SLAB_WAVELENGTH, 2, 1,
                                 mesh=meshes[size])
        errors.append(abs(solve.modes[0].n_eff.real / te_index - 1.0))
    order = math.log(errors[0] / errors[1]) / math.log(_SLAB_LINEAR_SIZES[0] / _SLAB_LINEAR_SIZES[1])
    normalized['te linear order'] = _LINEAR_MIN_ORDER / order if order > 0.0 else math.inf
    details['linear'] = {'sizes': list(_SLAB_LINEAR_SIZES), 'errors': errors, 'order': order}
    return _report(normalized, details)


# ---- em_modes_fem_tem_on_guess: the slab window filled uniformly (n 1.5) between pec x walls, pmc y walls: the
# parallel-plate TEM (e_t = constant E_x, e_z = 0) is an exact discrete mode, beta = k0 n, and the default guess max Re
# sqrt(eps mu) = n puts the shift on it before the 1e-6 offset of prepare_fem_modes (else a singular factor).
_TEM_INDEX = 1.5
_TEM_TOLERANCE = 1e-9   # relative n_eff


def _check_tem_on_guess(results_by_name):
    """Default guess on the TEM: mode 1 n_eff = n (1e-9 relative), |power| 1, every eigenvalue converged."""
    solve, layout = _direct_solve(_slab_spec(), _sizing(_SLAB_LINEAR_SIZES[0]), {'core': _TEM_INDEX}, _TEM_INDEX,
                                  {'x_lo': 'pec', 'x_hi': 'pec', 'y_lo': 'pmc', 'y_hi': 'pmc'}, _SLAB_WAVELENGTH, 2, 2)
    if not solve.modes:
        raise ValueError('no mode found with the guess on the TEM mode')
    mode, info = solve.modes[0], solve.info
    converged = info['eigenvalues_converged'] >= info['eigenvalues_requested']
    normalized = {'TEM n_eff': abs(complex(mode.n_eff) / _TEM_INDEX - 1.0) / _TEM_TOLERANCE,
                  'TEM power': abs(abs(mode.power) - 1.0) / 1e-9, 'converged': 0.0 if converged else math.inf}
    return _report(normalized, {'n_eff': [complex(mode.n_eff).real, complex(mode.n_eff).imag],
                                'shift_index': complex(info['shift_index']).real, 'unknowns': layout.n_unknowns,
                                'operator_applications': info['operator_applications'],
                                'warnings': list(info['warnings'])})


# ---- em_modes_fem_leaky: x-invariant layers along y (E_x(y) family, pec x walls): a substrate (n 1.8, continued into
# the y_lo PML ring), an air gap, a core (n 1.6) and air up to a pec y_hi wall. The core mode (n_eff ~1.33 < 1.8) leaks
# through the gap into the substrate. Exact leaky root: transfer of (E, dE/dy) from E = 0 at the top wall down to the
# substrate, outgoing there (dE/dy = j q E, q = sqrt(k0^2 n_s^2 - beta^2), Re q > 0), complex secant from the bound mode
# of an air substrate. PML: pml_kappa_max 1 (a radiating wave), 24 layers over 1.6 um.
_LEAKY_WAVELENGTH = 1.55e-6
_LEAKY_INDICES = {'substrate': 1.8, 'core': 1.6}   # air (1.0) elsewhere
_LEAKY_LAYERS = (0.3e-6, 0.4e-6, 0.4e-6, 0.8e-6)   # substrate (window part), gap, core, cover [m]
_LEAKY_WIDTH = 0.2e-6
_LEAKY_H = 25e-9
_LEAKY_RING = (1.6e-6, 24)                          # y_lo ring thickness [m], layers
_LEAKY_CPML = {'cpml_npml': 8, 'cpml_kappa_max': 1.0, 'cpml_sigma_factor': 1.0}
_LEAKY_TOLERANCES = (1e-5, 1e-3)                    # relative Re n_eff, relative Im n_eff


def _leaky_spec():
    substrate, gap, core, cover = _LEAKY_LAYERS
    spec = ModelSpec(title='Self-test: FEM leaky slab', domain=DomainSpec(size_x=_LEAKY_WIDTH,
                                                                          size_y=substrate + gap + core + cover))
    spec.geometry = [
        GeometrySpec('substrate', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _LEAKY_WIDTH, 'y1': substrate}),
        GeometrySpec('core', 'rectangle', {'x0': 0.0, 'y0': substrate + gap, 'x1': _LEAKY_WIDTH,
                                           'y1': substrate + gap + core})]
    return spec


def _leaky_index():
    """Exact complex n_eff (Im < 0) of the leaky core mode of _leaky_spec."""
    k0 = 2.0 * math.pi / _LEAKY_WAVELENGTH
    _, gap, core, cover = _LEAKY_LAYERS
    layers = ((1.0, cover), (_LEAKY_INDICES['core'], core), (1.0, gap))

    def substrate_top(beta_squared):
        field, slope = 0j, 1.0 + 0j   # E = 0 at the pec top wall
        for index, thickness in layers:
            k = np.sqrt(complex(k0 ** 2 * index ** 2 - beta_squared))
            if k == 0:
                field = field - slope * thickness
                continue
            field, slope = (field * np.cos(k * thickness) - slope * np.sin(k * thickness) / k,
                            field * k * np.sin(k * thickness) + slope * np.cos(k * thickness))
        return field, slope

    def bound(beta_squared):   # air below: field decaying downward, dE/dy = gamma E
        field, slope = substrate_top(beta_squared)
        return float(np.real(slope - np.sqrt(complex(beta_squared - k0 ** 2)) * field))

    def leaky(beta_squared):
        field, slope = substrate_top(beta_squared)
        return slope - 1j * np.sqrt(complex(k0 ** 2 * _LEAKY_INDICES['substrate'] ** 2 - beta_squared)) * field

    previous = _highest_root(bound, (k0 * _LEAKY_INDICES['core']) ** 2, k0 ** 2) + 0j
    current = previous * (1.0 - 1e-6)
    value_previous, value_current = leaky(previous), leaky(current)
    for _ in range(100):
        step = value_current * (current - previous) / (value_current - value_previous)
        previous, value_previous = current, value_current
        current = current - step
        value_current = leaky(current)
        if abs(step) <= 1e-15 * abs(current):
            break
    beta = np.sqrt(current)
    return complex(beta if beta.real > 0 else -beta) / k0


def _check_leaky(results_by_name):
    """The found mode nearest the exact leaky n_eff: Re (1e-5) and Im (1e-3) relative."""
    expected = _leaky_index()
    solve, layout = _direct_solve(
        _leaky_spec(), _sizing(_LEAKY_H, {'y_lo': _LEAKY_RING[0]}, _LEAKY_RING[1]), _LEAKY_INDICES, 1.0,
        {'x_lo': 'pec', 'x_hi': 'pec', 'y_lo': 'cpml', 'y_hi': 'pec'}, _LEAKY_WAVELENGTH, 4, 2,
        n_eff_guess=expected.real, cpml=_LEAKY_CPML)
    if not solve.modes:
        raise ValueError('no mode found near the leaky mode')
    mode = min(solve.modes, key=lambda candidate: abs(candidate.n_eff - expected))
    measured = complex(mode.n_eff)
    normalized = {'Re n_eff': abs(measured.real / expected.real - 1.0) / _LEAKY_TOLERANCES[0],
                  'Im n_eff': abs(measured.imag / expected.imag - 1.0) / _LEAKY_TOLERANCES[1]}
    return _report(normalized, {'n_eff': [measured.real, measured.imag], 'n_eff_exact': [expected.real, expected.imag],
                                'pml_fraction': mode.pml_fraction, 'unknowns': layout.n_unknowns,
                                'modes_found': [[complex(m.n_eff).real, complex(m.n_eff).imag] for m in solve.modes]})


# ---- em_modes_fem_cudss: the leaky-slab pencil (complex symmetric, PML ring) factorised by cuDSS vs host SuperLU.
# Only the leaky core mode is compared: the other modes of the list lie far from the shift at the edge of the wanted set
# (one with n_eff ~1e-7, whose relative error is meaningless), so which ones ARPACK keeps depends on the factor rounding.
_CUDSS_FACTORIZATIONS = ('cudss', 'cudss_mixed')
_CUDSS_TOLERANCE = 1e-9   # relative n_eff difference of the leaky mode to the SuperLU solve (first estimate)


def _check_fem_cudss(results_by_name):
    """The SuperLU mode nearest the exact leaky n_eff matched by the nearest cudss / cudss_mixed mode (relative); both
    factors must be the LDL^T; every mode found, refinement steps and device bytes recorded."""
    from ...model.triangle_meshing import triangulate
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    skipped = skip_without_cudss()
    if skipped is not None:
        return skipped
    spec = _leaky_spec()
    mesh = triangulate(spec, {}, sizing=_sizing(_LEAKY_H, {'y_lo': _LEAKY_RING[0]}, _LEAKY_RING[1]))
    faces = {'x_lo': 'pec', 'x_hi': 'pec', 'y_lo': 'cpml', 'y_hi': 'pec'}
    guess = _leaky_index().real

    def solve(eigen_solver):
        return _direct_solve(spec, None, _LEAKY_INDICES, 1.0, faces, _LEAKY_WAVELENGTH, 4, 2, n_eff_guess=guess,
                             cpml=_LEAKY_CPML, mesh=mesh, eigen_solver=eigen_solver)[0]

    superlu_modes = [complex(mode.n_eff) for mode in solve(None).modes]
    if not superlu_modes:
        raise ValueError('the SuperLU solve found no mode near the leaky mode')
    exact = _leaky_index()
    reference = min(superlu_modes, key=lambda value: abs(value - exact))
    normalized, details = {}, {'n_eff_superlu': [[value.real, value.imag] for value in superlu_modes],
                               'leaky_superlu': [reference.real, reference.imag]}
    for factorization in _CUDSS_FACTORIZATIONS:
        eigen_solver = ArpackShiftInvertSolver(factorization=factorization)
        result = solve(eigen_solver)
        found = [complex(mode.n_eff) for mode in result.modes]
        error = min(abs(value - reference) for value in found) / abs(reference) if found else math.inf
        normalized[f'{factorization} n_eff'] = error / _CUDSS_TOLERANCE
        normalized[f'{factorization} LDL^T'] = 0.0 if eigen_solver.symmetric_factor else math.inf
        details[factorization] = {'relative_error': error, 'symmetric_factor': bool(eigen_solver.symmetric_factor),
                                  'refinement_steps': int(eigen_solver.refinement_steps),
                                  'device_bytes': int(eigen_solver.device_bytes),
                                  'n_eff_found': [[value.real, value.imag] for value in found],
                                  'single_residual': result.info.get('single_residual'),
                                  'rayleigh_change': result.info.get('rayleigh_change'),
                                  'operator_applications': result.info['operator_applications'],
                                  'factor_seconds': result.info['factor_seconds'],
                                  'eigen_seconds': result.info['eigen_seconds']}
    normalized['cudss_mixed polished'] = 0.0 if details['cudss_mixed']['rayleigh_change'] is not None else math.inf
    return _report(normalized, details)


# ---- em_modes_fem_pipe: PEC circular pipe (radius a = 0.6 um, a / h = 16) as a pec_domain around a circle, vacuum,
# 1.2 um: n_eff^2 = 1 - (k_c / k0)^2 with k_c a = 1.841184 (TE11, a degenerate pair: modes 1 and 2) and 2.404826
# (TM01, mode 3). The curved wall is isoparametric (fem_nedelec_pipe_cutoff: k_c to 1e-6 at a / h = 16). The window
# keeps 0.05 um (~1.3 h) of metal outside the circle: the metal elements are only eliminated by the PEC closure.
_PIPE_RADIUS = 0.6e-6
_PIPE_WINDOW = 1.3e-6
_PIPE_WAVELENGTH = 1.2e-6
_PIPE_CUTOFFS = (1.8411837813406593, 1.8411837813406593, 2.404825557695773)   # k_c a of modes 1..3
_PIPE_TOLERANCE = 1e-6


def _pipe_spec():
    centre = 0.5 * _PIPE_WINDOW
    settings = {f'bc_{face}': 'pec' for face in FACES}
    settings.update(wavelength=_PIPE_WAVELENGTH, n_modes=len(_PIPE_CUTOFFS), discretization='quadratic')
    spec = ModelSpec(
        title='em_modes on triangles: PEC circular pipe',
        domain=DomainSpec(size_x=_PIPE_WINDOW, size_y=_PIPE_WINDOW),
        geometry=[GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _PIPE_WINDOW, 'y1': _PIPE_WINDOW}),
                  GeometrySpec('pipe', 'circle', {'cx': centre, 'cy': centre, 'r': _PIPE_RADIUS}),
                  GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['pipe']})],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec(_TABLE, 'mode_table'), FeatureSpec('wall', 'pec_domain', ['metal'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_PIPE_RADIUS / 16.0)],
        study=StudySpec(type='mode_analysis'))
    spec.solver_defaults.backend = 'cpu'
    return spec


def _check_pipe(results_by_name):
    """n_eff of modes 1..3 vs TE11, TE11, TM01 (relative 1e-6)."""
    rows = results_by_name['run'].result(_PHYSICS, _TABLE)['rows']
    if len(rows) < len(_PIPE_CUTOFFS):
        raise ValueError(f'expected {len(_PIPE_CUTOFFS)} modes in the table, got {len(rows)}')
    k0 = 2.0 * math.pi / _PIPE_WAVELENGTH
    expected = [math.sqrt(1.0 - (cutoff / (k0 * _PIPE_RADIUS)) ** 2) for cutoff in _PIPE_CUTOFFS]
    measured = [float(row[_N_EFF_COLUMN]) for row in rows[:len(_PIPE_CUTOFFS)]]
    normalized = {f'mode {number} n_eff': abs(value / reference - 1.0) / _PIPE_TOLERANCE
                  for number, (value, reference) in enumerate(zip(measured, expected), start=1)}
    return _report(normalized, {'n_eff': measured, 'n_eff_exact': expected})


# ---- em_modes_fem_rib: the S7.8 TFLN rib (mesh_study/models/rib_modes.json: 600 nm x-cut film, 300 nm etch, 1.2 um
# top, 60 deg sidewalls, 1.55 um, quasi-TE0 under an antisymmetric x_lo mirror through the rib centre, here the window
# edge) on triangles, rib and slab at 80 / 40 nm (120 nm elsewhere; a 20 nm level took 74 of 104 s and moved n_eff_1 by
# 1.3e-6, 40 nm is converged far below the tolerance). Reference: the order-1 extrapolation of the uniform 10 / 6.25 nm
# tensor runs of mesh_study_out/tables/accuracy_table.csv. The three-point Richardson fit of mesh_study analysis
# (1.9083084, order 1.57) is pre-asymptotic: the local order falls from ~2 (25 / 20 / 12.5 nm) to 1.57, the subpixel
# family's finest three give 1.05, and order-1 extrapolations of uniform (1.907948), subpixel (1.907895) and
# anisotropic_subpixel (1.907957) agree to 3e-5 with these triangles (1.907986). The tensor family to beat is
# anisotropic_subpixel: its cells at 1e-4 and order in cells = analysis.power_fit of its four rows against this reference.
_RIB_REFERENCE = 1.9079481313
_RIB_TENSOR_CELLS = 1.772e5                # anisotropic_subpixel cells at relative error 1e-4
_RIB_TENSOR_ORDER = 0.9754                 # its order in cells
_RIB_TARGETS = (1e-4, 1e-5)
_RIB_GATE = 3.0                             # S7.5 gate: tensor cells / elements at the target
_RIB_LEVELS = (80e-9, 40e-9)                # element size in the rib and slab [m]
_RIB_H_MAX = 120e-9
_RIB_TOLERANCE = 1e-4                       # relative n_eff_1 at the finest level
_UM = 1e-6


def _rib_run(size):
    return f'h{round(size * 1e9)}nm'


def _rib_spec(size):
    """Half TFLN rib (mirror at x = 0) on a triangles mesh with rib / slab elements of `size`."""
    box, film, etch, cover, window = 1.5 * _UM, 0.6 * _UM, 0.3 * _UM, 1.0 * _UM, 3.0 * _UM
    y_slab, y_film = box + film - etch, box + film
    x_top = 0.6 * _UM
    x_foot = x_top + etch / math.tan(math.radians(60.0))
    n_o, n_e = 2.211, 2.138
    settings = {'bc_x_lo': 'antisymmetric', 'bc_x_hi': 'cpml', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml',
                'pml_sigma_factor': 0.7, 'pml_cells': 4, 'wavelength': 1.55e-6, 'n_modes': 2,
                'discretization': 'quadratic'}
    spec = ModelSpec(
        title=f'em_modes on triangles: TFLN rib ({size * 1e9:g} nm)',
        domain=DomainSpec(size_x=window, size_y=box + film + cover),
        geometry=[GeometrySpec('top_region', 'rectangle', {'x0': 0.0, 'y0': box, 'x1': window,
                                                           'y1': box + film + cover}),
                  GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': box, 'x1': window, 'y1': y_slab}),
                  GeometrySpec('rib', 'polygon', {'points': [(0.0, box), (x_foot, box), (x_foot, y_slab),
                                                             (x_top, y_film), (0.0, y_film)]})],
        materials=[MaterialSpec('buried_oxide', library='silica', selection=['all']),
                   MaterialSpec('top_cladding', properties={'eps_r': 1.0}, selection=['top_region']),
                   MaterialSpec('lithium_niobate_x_cut', properties={'eps_xx': n_e ** 2, 'eps_yy': n_o ** 2,
                                                                     'eps_zz': n_o ** 2}, selection=['slab', 'rib'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[FeatureSpec(_TABLE, 'mode_table')])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='nonuniform', dx=_RIB_H_MAX,
                         refinements=[RefinementSpec('rib', size), RefinementSpec('slab', size)])],
        study=StudySpec(type='mode_analysis'),
        derived_values=[DerivedValueSpec('n_eff_1', f'{_PHYSICS}_n_eff_1')])
    spec.solver_defaults.backend = 'cpu'
    return spec


def _check_rib(results_by_name):
    """Relative n_eff_1 error per level vs the reference (finest <= 1e-4); elements / unknowns at 1e-4 and 1e-5 from the
    two finest levels' order, the gains over anisotropic_subpixel cells (and 2 unknowns per cell) recorded."""
    levels = []
    for size in _RIB_LEVELS:
        summary = results_by_name[_rib_run(size)].summary()
        n_eff = float(summary['derived']['n_eff_1'])
        counters = ((summary.get('solvers') or {}).get(_PHYSICS) or {}).get('counters') or {}
        levels.append({'element_size': size, 'n_eff_1': n_eff, 'error': abs(n_eff / _RIB_REFERENCE - 1.0),
                       'elements': float(counters.get('elements', math.nan)),
                       'unknowns': float(counters.get('unknowns', math.nan))})
    coarse, fine = levels[-2], levels[-1]
    order = (math.log(coarse['error'] / fine['error']) / math.log(fine['elements'] / coarse['elements'])
             if coarse['error'] > 0.0 and fine['error'] > 0.0 else math.nan)
    gains = {}
    for target in _RIB_TARGETS:
        elements = fine['elements'] * (fine['error'] / target) ** (1.0 / order) if order > 0.0 else math.nan
        tensor_cells = _RIB_TENSOR_CELLS * (1e-4 / target) ** (1.0 / _RIB_TENSOR_ORDER)
        gains[f'{target:g}'] = {'elements': elements, 'unknowns': elements * fine['unknowns'] / fine['elements'],
                                'tensor_cells': tensor_cells, 'cell_gain': tensor_cells / elements,
                                'unknown_gain': 2.0 * tensor_cells / (elements * fine['unknowns'] / fine['elements']),
                                'gate_passed': bool(tensor_cells / elements >= _RIB_GATE)}
    return _report({'finest n_eff_1': fine['error'] / _RIB_TOLERANCE},
                   {'reference': _RIB_REFERENCE, 'levels': levels, 'order_in_elements': order, 'at_targets': gains})


# ---- em_modes_fem_heated_strip: a PEC box (1.0 x 0.5 um) filled with n0 = 1.5 (dndT 2e-4 /K, no expansion), heat on a
# tensor mesh with every face held at T0 + dT (uniform T), heat_em_thermo_optic into em_modes on triangles. Uniform
# eps: beta^2 = k0^2 eps - (a discrete cutoff independent of eps), so n_eff_hot^2 - n_eff_cold^2 = eps_hot - eps_cold
# = (n0 + dndT dT)^2 - n0^2 exactly on one mesh; float32 eps arrays leave ~2e-5 of the difference.
_STRIP_SIZE = (1.0e-6, 0.5e-6)
_STRIP_WAVELENGTH = 1.0e-6
_STRIP_INDEX = 1.5
_STRIP_DNDT = 2e-4
_STRIP_RISE = 10.0                # K
_STRIP_T0 = 293.15                # heat T0 default [K]
_STRIP_H = 50e-9                  # triangles (optics) and tensor (thermal) sizes [m]
_STRIP_TOLERANCES = (1e-4, 1e-6)  # relative n_eff^2 difference, relative mean T


def _heated_strip_spec(rise):
    settings = {f'bc_{face}': 'pec' for face in FACES}
    settings.update(wavelength=_STRIP_WAVELENGTH, n_modes=1, discretization='quadratic')
    spec = ModelSpec(
        title=f'em_modes on triangles heated by heat on a tensor mesh (dT = {rise:g} K)',
        domain=DomainSpec(size_x=_STRIP_SIZE[0], size_y=_STRIP_SIZE[1]),
        materials=[MaterialSpec('filling', properties={'eps_r': _STRIP_INDEX ** 2, 'k': 1.0, 'dndT': _STRIP_DNDT},
                                selection=['all'])],
        physics=[PhysicsSpec('heat', 'heat', mesh='thermal', features=[
                     FeatureSpec('walls', 'temperature', list(FACES), {'temperature': _STRIP_T0 + rise})]),
                 PhysicsSpec(_PHYSICS, _TYPE, settings=settings, mesh='optics',
                             features=[FeatureSpec(_TABLE, 'mode_table')])],
        couplings=[CouplingSpec('thermo_optic', 'heat_em_thermo_optic', 'heat', _PHYSICS)],
        meshes=[MeshSpec(name='optics', kind='triangles', mode='uniform', dx=_STRIP_H),
                MeshSpec(name='thermal', kind='tensor', mode='uniform', dx=_STRIP_H)],
        study=StudySpec(type='mode_analysis'),
        derived_values=[DerivedValueSpec('n_eff_1', f'{_PHYSICS}_n_eff_1'),
                        DerivedValueSpec('T_mean', 'T', ['all'], 'mean')])
    spec.solver_defaults.backend = 'cpu'
    return spec


def _check_heated_strip(results_by_name):
    """n_eff^2 difference hot - cold vs the float32 eps difference (1e-4 relative), mean T of the hot run (1e-6)."""
    derived = {name: results_by_name[name].summary()['derived'] for name in ('cold', 'hot')}
    index = np.float32(_STRIP_INDEX)
    eps_cold = float(index ** 2)
    eps_hot = float((index + np.float32(_STRIP_DNDT) * np.float32(_STRIP_RISE)) ** 2)
    difference = float(derived['hot']['n_eff_1']) ** 2 - float(derived['cold']['n_eff_1']) ** 2
    t_mean = float(derived['hot']['T_mean'])
    normalized = {'n_eff^2 difference': abs(difference / (eps_hot - eps_cold) - 1.0) / _STRIP_TOLERANCES[0],
                  'mean T': abs(t_mean / (_STRIP_T0 + _STRIP_RISE) - 1.0) / _STRIP_TOLERANCES[1]}
    return _report(normalized, {'n_eff_cold': float(derived['cold']['n_eff_1']),
                                'n_eff_hot': float(derived['hot']['n_eff_1']), 'difference': difference,
                                'eps_difference': eps_hot - eps_cold, 'T_mean': t_mean})


# ---- em_modes_fem_cudss_model: a dielectric-filled pec box (1.0 x 0.4 um, n 1.5, 1.0 um: TE10, TE20, TE01 guided,
# none degenerate) on 40 nm triangles, model runs with host SuperLU (cpu backend) and an eigen node cudss_mixed (gpu) ----
_BOX_SIZE = (1.0e-6, 0.4e-6)
_BOX_INDEX = 1.5
_BOX_WAVELENGTH = 1.0e-6
_BOX_MODES = 3
_BOX_H = 40e-9
_BOX_NODE = 'eigen_cudss_mixed'
_BOX_TOLERANCE = 1e-9   # relative n_eff to the SuperLU run (first estimate, as em_modes_fem_cudss)


def _cudss_box_spec(factorization):
    """The pec box on triangles; 'superlu' on the cpu backend, else an arpack_shift_invert node of that factorization."""
    settings = {f'bc_{face}': 'pec' for face in FACES}
    settings.update(wavelength=_BOX_WAVELENGTH, n_modes=_BOX_MODES, discretization='quadratic')
    spec = ModelSpec(
        title=f'em_modes on triangles: dielectric pec box ({factorization})',
        domain=DomainSpec(size_x=_BOX_SIZE[0], size_y=_BOX_SIZE[1]),
        materials=[MaterialSpec('filling', properties={'eps_r': _BOX_INDEX ** 2}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[FeatureSpec(_TABLE, 'mode_table')])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_BOX_H)],
        study=StudySpec(type='mode_analysis'))
    if factorization == 'superlu':
        spec.solver_defaults.backend = 'cpu'
    else:
        spec.solvers = [SolverSpec(_BOX_NODE, 'arpack_shift_invert', {'factorization': factorization})]
        spec.physics[0].solvers = {'eigen': _BOX_NODE}
    return spec


def _cudss_box_specs():
    """{'superlu', 'cudss_mixed'} runs when nvmath-python is installed, else {} (the check skips)."""
    from ...solvers.cudss_factor import cudss_available
    if not cudss_available():
        return {}
    return {name: _cudss_box_spec(name) for name in ('superlu', 'cudss_mixed')}


def _check_cudss_box(results_by_name):
    """Every SuperLU mode-table n_eff matched by the nearest cudss_mixed one; the eigen role factorised an LDL^T on
    the device (role counters)."""
    if 'cudss_mixed' not in results_by_name:
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': 1.0,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    n_eff = {name: [float(row[_N_EFF_COLUMN]) for row in results_by_name[name].result(_PHYSICS, _TABLE)['rows']]
             for name in ('superlu', 'cudss_mixed')}
    if not n_eff['superlu']:
        raise ValueError('the SuperLU run found no mode')
    error = max((min(abs(value - expected) for value in n_eff['cudss_mixed']) / abs(expected)
                 if n_eff['cudss_mixed'] else math.inf) for expected in n_eff['superlu'])
    record = (results_by_name['cudss_mixed'].summary().get('solvers') or {}).get(_PHYSICS) or {}
    counters = ((record.get('roles') or {}).get('eigen') or {}).get('counters') or {}
    normalized = {'n_eff': error / _BOX_TOLERANCE,
                  'device factor': 0.0 if int(counters.get('eigen_factor_device_bytes', 0)) > 0 else math.inf,
                  'LDL^T': 0.0 if int(counters.get('eigen_factor_symmetric', 0)) == 1 else math.inf}
    return _report(normalized, {'n_eff': n_eff, 'relative_error': error, 'eigen_counters': counters,
                                'physics_counters': record.get('counters') or {}})


MODES_FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='em_modes_fem_slab',
        label='EM modes on triangles: walled slab',
        description='CPU (fem_modes directly, mesh plan T3): a 0.5 um core (n 2.0) centred between pec x walls 1.8 um '
                    'apart in n 1.45, y-invariant (0.09 um high), 1.55 um. y faces pec keep the TE family (E_y(x)), '
                    'pmc the TM family (H_y(x), E_x, E_z); expected: the fundamental n_eff of each = the exact '
                    'transcendental root (TE kx tan(kx d) = gamma coth(gamma (L - d)), TM with 1 / n^2 and tanh) to '
                    f'{_SLAB_TOLERANCE:g} relative with quadratic elements at h = 15 nm (interfaces on mesh edges), '
                    '|power| = 1 W, te_fraction 0 (TE) / 1 (TM); linear elements at h = 30 / 15 nm converge with order '
                    f'>= {_LINEAR_MIN_ORDER:g} in h. measured = error = the worst item in tolerance units, tolerance '
                    '1. Written, not run.',
        specs=lambda: {},
        check=_check_slab),
    SelfTestCase(
        name='em_modes_fem_tem_on_guess',
        label='EM modes on triangles: guess exactly on a TEM mode',
        description='CPU (fem_modes directly): the em_modes_fem_slab window (1.8 x 0.09 um) filled uniformly with n '
                    f'{_TEM_INDEX:g} between pec x walls and pmc y walls, 1.55 um, quadratic elements at 30 nm. The '
                    'parallel-plate TEM (constant E_x) is an exact discrete mode with n_eff = n, and the default '
                    'guess (max Re sqrt(eps mu) = n) puts the shift on it before the 1e-6 relative offset of '
                    f'prepare_fem_modes. Expected: mode 1 n_eff = {_TEM_INDEX:g} to {_TEM_TOLERANCE:g} relative, '
                    '|power| = 1 W, every requested eigenvalue converged. measured = error = the worst item in '
                    'tolerance units, tolerance 1. Written, not run.',
        specs=lambda: {},
        check=_check_tem_on_guess),
    SelfTestCase(
        name='em_modes_fem_leaky',
        label='EM modes on triangles: PML leaky mode',
        description='CPU (fem_modes directly): x-invariant layers along y between pec x walls: substrate n 1.8 '
                    '(continued into a 1.6 um y_lo PML ring of 24 layers, pml_kappa_max 1), 0.4 um air gap, 0.4 um '
                    'core n 1.6, 0.8 um air to a pec top wall, 1.55 um, quadratic elements at 25 nm. The core mode '
                    '(n_eff ~1.33) leaks through the gap into the substrate. Expected: the found mode nearest the '
                    'exact leaky root (layer transfer with an outgoing substrate wave, complex secant) to '
                    f'{_LEAKY_TOLERANCES[0]:g} relative in Re n_eff and {_LEAKY_TOLERANCES[1]:g} relative in Im n_eff '
                    '(< 0: loss). Deviation from the plan: the reference is this closed form, not a tensor run. '
                    'measured = error = the worst item in tolerance units, tolerance 1. Written, not run.',
        specs=lambda: {},
        check=_check_leaky),
    SelfTestCase(
        name='em_modes_fem_cudss',
        label='EM modes on triangles: cuDSS LDL^T factor',
        description='GPU (fem_modes directly, no model run): the em_modes_fem_leaky pencil (complex symmetric with the '
                    'PML ring) solved with the eigen factorization superlu, cudss and cudss_mixed. Expected: the '
                    'SuperLU mode nearest the exact leaky n_eff matched by the nearest mode of each cuDSS solve to '
                    f'{_CUDSS_TOLERANCE:g} relative (the other modes lie far from the shift at the edge of the wanted '
                    'set and are only recorded; a first estimate: cudss = ARPACK at machine precision on solves '
                    'accurate to the probe residual; cudss_mixed = one float64 refinement step per solve, ARPACK to '
                    '1e-6, then the float64 Rayleigh quotients of the complex symmetric pencil, error ~ (1e-6)^2), both '
                    'cuDSS factors the LDL^T (symmetry check passed, fem_modes row_scale equilibration) and the '
                    'cudss_mixed solve polished. Recorded: unrefined complex64 residual, Rayleigh change of n_eff^2, '
                    'refinement steps of a full-accuracy solve, device bytes, applications and times. Skipped without '
                    'nvmath-python. measured = error = the worst item in tolerance units, tolerance 1. Written, not '
                    'run.',
        specs=lambda: {},
        check=_check_fem_cudss,
        backends=('gpu',)),
    SelfTestCase(
        name='em_modes_fem_cudss_model',
        label='EM modes on triangles: cudss_mixed eigen node',
        description='Model runs (no gmsh): a pec box 1.0 x 0.4 um filled with n 1.5 at 1.0 um on 40 nm quadratic '
                    'triangles, 3 modes (TE10, TE20, TE01), once with the default eigen solver on the cpu backend (host '
                    'SuperLU) and once with an arpack_shift_invert node of factorization cudss_mixed on the gpu backend '
                    '(complex64 LDL^T, loose solves, float64 Rayleigh quotients). Expected: every SuperLU mode-table '
                    f'n_eff matched by the nearest cudss_mixed one to {_BOX_TOLERANCE:g} relative (first estimate), '
                    'and the eigen role counters show a device factor that is an LDL^T. Pins the role plumbing '
                    '(eigen_role mixed_factorization, no host size rule, the summary text). Skipped without '
                    'nvmath-python. measured = error = the worst item in tolerance units, tolerance 1. Written, not '
                    'run.',
        specs=_cudss_box_specs,
        check=_check_cudss_box,
        backends=('gpu',)),
    SelfTestCase(
        name='em_modes_fem_pipe',
        label='EM modes on triangles: PEC circular pipe',
        description='em_modes model run on the cpu backend on a triangles mesh (h = a / 16, discretization '
                    'quadratic): a vacuum circle of radius a = 0.6 um inside a pec_domain (difference of the '
                    f'{_PIPE_WINDOW * 1e6:g} um window box and the circle), pec faces, 1.2 um. Expected n_eff^2 = 1 - '
                    '(k_c / k0)^2 with k_c a = 1.841184 for modes 1 and 2 (the degenerate TE11 pair) and 2.404826 for '
                    f'mode 3 (TM01), relative {_PIPE_TOLERANCE:g}: pins the builder path (element PEC masks, curved '
                    'isoparametric wall, mode table). measured = error = the worst item in tolerance units, tolerance '
                    '1. Not yet run with the 1.3 um window.',
        specs=lambda: {'run': _pipe_spec()},
        check=_check_pipe,
        heavy=True),
    SelfTestCase(
        name='em_modes_fem_rib',
        label='EM modes on triangles: TFLN rib (S7.5 gain)',
        description='S7.5 gain measurement (solver_plan S7.8): the TFLN rib of mesh_study rib_modes (half structure, '
                    'antisymmetric x_lo mirror on the window edge, cpml elsewhere with 4 x 120 nm rings) on a '
                    'triangles mesh with rib and slab elements of 80 and 40 nm (120 nm elsewhere), quadratic, '
                    f'cpu backend. Expected: quasi-TE0 n_eff_1 within {_RIB_TOLERANCE:g} of the reference '
                    f'{_RIB_REFERENCE} (order-1 extrapolation of the uniform 10 / 6.25 nm tensor runs; staircase and '
                    'subpixel interface errors are first order) at the finest level (measured = error = that '
                    'relative error over the tolerance). Recorded in details: per level n_eff_1, error, elements and '
                    'unknowns (work counters); elements / unknowns at 1e-4 and 1e-5 from the order of the two '
                    f'levels; the gains over the anisotropic_subpixel tensor cells ({_RIB_TENSOR_CELLS:.4g} at 1e-4, '
                    f'order {_RIB_TENSOR_ORDER:.3g} in cells) and the gate flag (cell gain >= {_RIB_GATE:g}).',
        specs=lambda: {_rib_run(size): _rib_spec(size) for size in _RIB_LEVELS},
        check=_check_rib,
        heavy=True),
    SelfTestCase(
        name='em_modes_fem_heated_strip',
        label='EM modes on triangles heated from a tensor mesh',
        description='Per-physics meshes (S7.2) with S7.5: heat on a uniform 50 nm tensor mesh with every face held at '
                    'T0 + dT, heat_em_thermo_optic into em_modes on a 50 nm triangles mesh, a pec box 1.0 x 0.5 um '
                    'filled with n0 = 1.5, dndT 2e-4 /K, 1.0 um, cpu backend; runs cold (dT 0) and hot (dT 10 K). A '
                    'uniform eps shifts beta^2 by k0^2 delta eps on any mesh, so expected n_eff_hot^2 - n_eff_cold^2 = '
                    f'(n0 + dndT dT)^2 - n0^2 (float32) to {_STRIP_TOLERANCES[0]:g} relative, and the hot mean T = T0 '
                    f'+ dT to {_STRIP_TOLERANCES[1]:g}. measured = error = the worst item in tolerance units, '
                    'tolerance 1. Written, not run.',
        specs=lambda: {'cold': _heated_strip_spec(0.0), 'hot': _heated_strip_spec(_STRIP_RISE)},
        check=_check_heated_strip),
)

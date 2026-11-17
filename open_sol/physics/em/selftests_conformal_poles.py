"""Self-tests of the 2D FDTD conformal pole weights (solver_defaults.conformal_boundaries on dispersive materials,
solver_plan S7.6; physics/em/conformal_poles): the cut-cell geometry of a Drude disk (CPU) and the normal-incidence
transmission of a thin Drude film whose faces lie off the cell faces (em_fdtd_ez / _hz runs)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, checks_result, float_list, vertical_line
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

_PHYSICS = 'em'
_FLUX = 'transmitted'

# ---- weights (CPU): a disk of radius 11.37 unit cells on a 32 x 32 grid, centre off the cell centres ----
_GRID_CELLS = 32
_DISK = (16.21, 15.83, 11.37)          # cx, cy, r [cells]
_AREA_TOLERANCE = 1e-3                 # disk area from the cell / dual-cell fractions (32 midpoint rows)
_BOUND_TOLERANCE = 1e-12               # face weight <= the larger stability weight of its two cells
_FAR_CELLS = 2.0                       # cells farther than this from the outline keep the staircase mask exactly

# ---- film: Drude film 23.7 nm thick, faces 0.26 / 0.48 of a 5 nm cell into its boundary cells ----
_CELL = 5e-9
_SIZE_X, _SIZE_Y = 1.6e-6, 20e-9       # y-periodic strip, 4 cells across
_SOURCE_X, _FLUX_X = 0.3e-6, 1.3e-6
_FILM = (0.8037e-6, 0.8274e-6)         # x0, x1: 4.74 cells; its centres hold 4 cells (staircase 20 nm)
_OMEGA_P, _GAMMA, _EPS_INF = 1.37e16, 3.2e13, 1.0   # silver-like Drude; eps ~ -52 at 1 um, skin depth ~22 nm
_PULSE = {'waveform': 'gaussian', 'wavelength': 1e-6, 'bandwidth': 0.5, 'amplitude': 1.0}
_SPECTRAL = {'wavelength_start': 0.9e-6, 'wavelength_stop': 1.1e-6, 'n_wavelengths': 5}
# s: the pulse ends at 19 fs and reaches the flux line ~3 fs later; the film (no restoring force) has no in-band ringing
_DURATION = 50e-15
_FILM_TOLERANCE = 0.04                 # relative T, worst wavelength / polarisation (estimated, not measured)
_COMPONENTS = {'ez': 'em_fdtd_ez', 'hz': 'em_fdtd_hz'}


# ---- weights ----
class _UniformGrid:
    """The grid attributes yee_grid.grid_cell_widths reads (graded form)."""

    def __init__(self, n):
        self.nx = self.ny = n
        self.dx_arr = np.ones(n)
        self.dy_arr = np.ones(n)


class _UnitDomain:
    """Duck-typed Domain of n x n unit cells without padding (cell_edges, physical_window, grid)."""

    def __init__(self, n):
        self.grid = _UniformGrid(n)
        self._edges = np.arange(n + 1, dtype=np.float64)

    def cell_edges(self):
        return self._edges, self._edges

    def physical_window(self):
        return slice(0, self.grid.nx), slice(0, self.grid.ny)


def _check_pole_weights(results_by_name):
    """Disk area from the E_z cell weights and from both H_z face weight sets (dual cells of unit area), the face
    weights bounded by their cells' stability weights, the staircase kept away from the outline."""
    from .conformal_poles import conformal_pole_weights
    n = _GRID_CELLS
    cx, cy, radius = _DISK
    domain = _UnitDomain(n)
    centres = np.arange(n, dtype=np.float64) + 0.5

    def disk(x, y):
        return (np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2 <= radius ** 2
    region = disk(centres[:, None], centres[None, :])
    ez = conformal_pole_weights(domain, 'ez', region, disk)
    hz = conformal_pole_weights(domain, 'hz', region, disk)
    area = math.pi * radius ** 2
    distance = np.abs(np.hypot(centres[:, None] - cx, centres[None, :] - cy) - radius)
    far = distance > _FAR_CELLS
    bound_x = hz.face_x[:, :-1] - np.maximum(hz.stability[:, :-1], hz.stability[:, 1:])
    bound_y = hz.face_y[:-1, :] - np.maximum(hz.stability[:-1, :], hz.stability[1:, :])
    staircase_change = max(float(np.max(np.abs(ez.cell - region)[far])),
                           float(np.max(np.abs(hz.stability - region)[far])))
    checks = {'ez_area': (abs(float(ez.cell.sum()) / area - 1.0), _AREA_TOLERANCE),
              'hz_x_area': (abs(float(hz.face_x[:, :-1].sum()) / area - 1.0), _AREA_TOLERANCE),
              'hz_y_area': (abs(float(hz.face_y[:-1, :].sum()) / area - 1.0), _AREA_TOLERANCE),
              'hz_bound': (max(float(bound_x.max()), float(bound_y.max())), _BOUND_TOLERANCE),
              'staircase_far': (staircase_change, 0.0)}
    entries = {name: {'value': float(value), 'tolerance': tolerance, 'passed': bool(value <= tolerance)}
               for name, (value, tolerance) in checks.items()}
    return checks_result(entries, staircase_area_ratio=float(region.sum()) / area, cut_cells=ez.summary(),
                         cut_faces=hz.summary())


# ---- film ----
def _film_spec(polarization, run):
    """y-periodic strip: gaussian line source at 0.3 um, flux line at 1.3 um; run 'reference' (vacuum) or 'on' (the
    Drude film, conformal_boundaries on)."""
    geometry = [vertical_line('source_line', _SOURCE_X, 0.0, _SIZE_Y),
                vertical_line('flux_line', _FLUX_X, 0.0, _SIZE_Y)]
    materials = []
    if run != 'reference':
        geometry.append(GeometrySpec('film', 'rectangle', {'x0': _FILM[0], 'y0': 0.0, 'x1': _FILM[1], 'y1': _SIZE_Y}))
        materials.append(MaterialSpec('drude_film', properties={}, selection=['film'], dispersion=[
            {'model': 'drude', 'omega_p': _OMEGA_P, 'gamma': _GAMMA, 'eps_inf': _EPS_INF}]))
    return ModelSpec(
        title=f'Self-test: thin Drude film transmission ({polarization}, {run})',
        domain=DomainSpec(size_x=_SIZE_X, size_y=_SIZE_Y),
        geometry=geometry,
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS, _COMPONENTS[polarization], settings=dict(PERIODIC_Y), features=[
            FeatureSpec('source', 'line_source', ['source_line'], dict(_PULSE)),
            FeatureSpec(_FLUX, 'flux_monitor', ['flux_line'], dict(_SPECTRAL))])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=run == 'on'),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))


def _film_specs():
    return {f'{polarization}_{run}': _film_spec(polarization, run)
            for polarization in _COMPONENTS for run in ('reference', 'on')}


def _film_transmission(wavelengths, thickness):
    """|t|^2 of a Drude film of `thickness` [m] in vacuum at normal incidence (exp(-i omega t): eps = eps_inf - omega_p^2
    / (omega^2 + i gamma omega), n with Im n >= 0, t = 4 n / ((1 + n)^2 e^{-i delta} - (1 - n)^2 e^{i delta}), delta = n
    k0 d)."""
    omega = 2.0 * math.pi * C0 / np.asarray(wavelengths, dtype=np.float64)
    index = np.sqrt(_EPS_INF - _OMEGA_P ** 2 / (omega ** 2 + 1j * _GAMMA * omega))
    delta = index * omega / C0 * thickness
    transmission = 4.0 * index / ((1.0 + index) ** 2 * np.exp(-1j * delta) - (1.0 - index) ** 2 * np.exp(1j * delta))
    return np.abs(transmission) ** 2


def _measured_transmission(results_by_name, polarization, run):
    """(wavelengths, film flux / vacuum flux) of one run."""
    reference = results_by_name[f'{polarization}_reference'].result(_PHYSICS, _FLUX)
    film = results_by_name[f'{polarization}_{run}'].result(_PHYSICS, _FLUX)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.asarray(film['power'], dtype=np.float64) / np.asarray(reference['power'], dtype=np.float64)
    return np.asarray(reference['wavelength'], dtype=np.float64), ratio


def _check_film(results_by_name):
    """Conformal T vs the Airy |t|^2 of the drawn thickness, worst wavelength and polarisation; the Airy |t|^2 of the
    20 nm staircase thickness in the details."""
    worst, worst_label, details = -1.0, '', {}
    for polarization in _COMPONENTS:
        wavelengths, conformal = _measured_transmission(results_by_name, polarization, 'on')
        exact = _film_transmission(wavelengths, _FILM[1] - _FILM[0])
        errors = np.abs(conformal / exact - 1.0)
        error = float(np.max(errors)) if np.all(np.isfinite(errors)) else math.inf
        if error > worst:
            worst, worst_label = error, polarization
        details[polarization] = {
            'wavelengths': float_list(wavelengths), 'T_exact': float_list(exact),
            'T_conformal': float_list(conformal), 'conformal_error': error,
            'T_20nm_film': float_list(_film_transmission(wavelengths, 4 * _CELL))}
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _FILM_TOLERANCE,
            'details': {'worst_polarization': worst_label, **details}}


CONFORMAL_POLE_SELFTEST_CASES = (
    SelfTestCase(
        name='em_conformal_pole_weights',
        label='Conformal poles: cut-cell weights of a Drude disk',
        description='CPU (physics/em/conformal_poles): the pole weights of a disk of radius 11.37 unit cells (centre '
                    'off the cell centres) on a 32 x 32 grid. The E_z cell weights and the H_z E_x / E_y face weights '
                    f'(dual cells of unit area) each sum to the disk area pi r^2 ({_AREA_TOLERANCE:g} relative: 32 '
                    'midpoint rows per cut cell); every face weight is at most the larger Courant-check weight of its '
                    'two cells (the bound the hz stability check relies on); cells more than 2 cells from the '
                    'outline keep the staircase mask exactly. error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_pole_weights),
    SelfTestCase(
        name='em_conformal_drude_film',
        label='Conformal poles: thin Drude film transmission',
        description='em_fdtd_ez and em_fdtd_hz, y-periodic strip (1.6 um, 5 nm cells, CPML along x, '
                    f'{_DURATION * 1e15:g} fs), gaussian pulse (1 um, '
                    'bandwidth 0.5) at normal incidence (E tangential to the film in both polarisations) through a '
                    'Drude film (omega_p 1.37e16 rad/s, gamma 3.2e13 rad/s, eps_inf 1: eps ~ -52, skin depth ~22 '
                    'nm) 23.7 nm thick whose faces lie 0.26 / 0.48 of a cell into their boundary cells (the cell '
                    'centres see 20 nm). T = flux behind the film / flux of a vacuum run, 5 wavelengths 0.9-1.1 um, '
                    'against the Airy |t|^2 of the drawn thickness with the continuum Drude eps; conformal_boundaries '
                    'on weights each boundary cell / E face pole by its inside area share (a staircase run, '
                    'conformal_boundaries off, is estimated ~30 % off; not run, the Airy T of the 20 nm staircase '
                    f'thickness is in the details). Tolerance {_FILM_TOLERANCE:g} relative (estimated, not measured: '
                    'second-order cut-cell error at dx / skin depth ~0.23, the 5-point dispersion in the metal ~0.5 '
                    '%). error = the worst relative error. The 1.6 um strip and 50 fs not yet confirmed by a run.',
        specs=_film_specs,
        check=_check_film),
)

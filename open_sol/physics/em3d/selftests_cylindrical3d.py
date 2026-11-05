"""Self-test of em_fdtd_3d on cylindrical (r, phi, z) sectors (symmetry 2.21): free ring-downs of a PEC coaxial annulus
(z-invariant TM_z / TE_z modes) on the engine of a polar Domain3D against the closed-form Bessel cross-product
resonances, the phi wrap as a full period, a real antiperiodic C_4 sector and a complex Floquet sector, an r_hi CPML
decay and the sector refusals (scripts, no model)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase

_INNER = 1.0e-6              # m: inner PEC wall = the centre of the x_lo pec layer cell (tangential E pinned there)
_N_R = 33                    # radial cells: walls at the centres of cells 0 and 32, so outer - inner = 32 dr
_DR = 0.5e-6 / (_N_R - 1)    # m: 15.6 nm, k dr ~ 0.1 at k ~ 6.5 / um
_OUTER = _INNER + (_N_R - 1) * _DR
_RADIUS_MIN = _INNER - 0.5 * _DR   # the grid's x_lo edge
_D_PHI = math.pi / 32        # rad: 64 cells per turn (r dphi ~ 98 nm at the inner wall)
_DZ = 0.2e-6                 # m: one z-periodic cell (the z-invariant modes; dz only enters the CFL step)
_BAND_K = (5.5e6, 7.2e6)     # 1/m: the first radial TM / TE order of |m| <= 3 (TM_41 ~7.05 / um is never excited)
_STEPS = 4500                # ~66 decimated samples of the band (~65 periods)
_CPML_STEPS = 2000
_CPML_CELLS = 12
_TOLERANCE = 2e-3            # second-order dr and dt errors ~4e-4 (phi is exact: nu below); a missing r factor >= 1 %
_ABSENT_WINDOW = 5e-3        # relative: no resonance this close to a mode the wrap phase excludes
_GROWTH_LIMIT = 2.0          # closed lossless runs: late / early probe peak
# r_hi CPML run: late / early probe peak. Measured 1.23e-2 (cpu = gpu, 2000 steps): the d/dr-only stretch (no
# complex 1 / r~) is approximate at small k r, presumably reflecting the k r < 1 content of the uniform initial E_z
# (late_k_r_outer in the details); a missing absorber leaves ~1 (the closed runs 0.97-1.07)
_CPML_DECAY = 5e-2
# run -> (field, phi cells, bloch_ky [1/rad], azimuthal orders initialised (None = uniform in phi), expected orders,
# orders the wrap phase excludes); bloch_ky m gives F(phi + span) = F(phi) e^{-j m span}: a quarter turn at m = 2 is
# antiperiodic (real wrap, no imaginary set), at m = 1 a complex Floquet phase (allowed orders 1 + 4 n: 1, -3)
_RUNS = {
    'tm_full_turn': ('ez', 64, 0.0, (0, 2, 3), (0, 2, 3), ()),
    'te_full_turn': ('hz', 64, 0.0, (1, 3), (1, 3), ()),
    'tm_antiperiodic_quarter': ('ez', 16, 2.0, None, (2,), (0, 1, 3)),
    'tm_floquet_quarter': ('ez', 16, 1.0, None, (1, 3), (0, 2)),
}
_PROBE_CELLS = ((9, 5), (16, 23), (24, 41))   # (i, j); j taken modulo the phi cells


def _sector_engine(backend, n_phi, bloch_ky=0.0, x_hi='pec'):
    """Vacuum FDTDEngine3D on a padding-free (bar an r_hi CPML) annular sector of n_phi cells, phi and z periodic, pec
    x_lo and the given x_hi."""
    from ...core.domain3d import Domain3D
    from ...core.metric import Coordinates
    from .boundaries3d import BoundaryConfig3D
    from .engine3d import FDTDEngine3D
    faces = {'x_lo': 'pec', 'x_hi': x_hi, 'y_lo': 'periodic', 'y_hi': 'periodic', 'z_lo': 'periodic',
             'z_hi': 'periodic'}
    config = BoundaryConfig3D(faces, npml=_CPML_CELLS, bloch_k=(0.0, bloch_ky, 0.0))
    sides = config.npml_sides
    n_r = _N_R + sides['x_hi']
    domain = Domain3D.from_widths(np.full(n_r, _DR), np.full(n_phi, _D_PHI), np.full(1, _DZ), sides,
                                  size_x=_N_R * _DR, size_y=n_phi * _D_PHI, size_z=_DZ, boundary_config=config,
                                  coordinates=Coordinates('polar', radius_min=_RADIUS_MIN, angle_wrap='periodic'),
                                  backend=backend)
    domain.materials.add_properties({'eps_r': 1.0, 'sigma_e': 0.0, 'mu_r': 1.0})
    engine = FDTDEngine3D(domain, config)
    engine.rebuild_materials()
    return engine


def _initial_field(engine, field, orders):
    """Set E_z (at r_c, phi_c) or H_z (at r_f[i + 1], phi_f[j + 1]) to the first radial PEC profile times the sum of
    cos(m phi) over orders (uniform in phi for None); zero outside the annulus."""
    radii = engine.radii
    n_phi = engine.ny
    if field == 'ez':
        radius, phi = radii.centre, (np.arange(n_phi) + 0.5) * _D_PHI
        profile = np.sin(math.pi * (radius - _INNER) / (_OUTER - _INNER))   # E_z = 0 on both walls
    else:
        radius, phi = radii.face[1:], (np.arange(n_phi) + 1.0) * _D_PHI
        profile = np.cos(math.pi * (radius - _INNER) / (_OUTER - _INNER))   # d H_z / d r = 0 on both walls
    profile = np.where((radius >= _INNER) & (radius <= _OUTER), profile, 0.0)
    pattern = np.ones(n_phi) if orders is None else sum(np.cos(order * phi) for order in orders)
    values = (profile[:, None] * pattern[None, :])[:, :, None] * np.ones((1, 1, engine.nz))
    getattr(engine.fields, field)[...] = engine.xp.asarray(values.astype(np.float32))


def _ring_down(engine, field, n_steps):
    """(n_probes, n_steps) float64 host trace of the field at the probe cells after every step (a flat-index take into
    a preallocated row: no temporary per step)."""
    from ...core.backend import to_host
    xp = engine.xp
    component = getattr(engine.fields, field)
    cells = [(i, j % engine.ny, 0) for i, j in _PROBE_CELLS]
    flat = xp.asarray(np.ravel_multi_index(tuple(np.array([cell[axis] for cell in cells]) for axis in range(3)),
                                           component.shape))
    samples = component.reshape(-1)   # a view: the fields are updated in place
    trace = xp.zeros((n_steps, len(cells)), dtype=component.dtype)
    for step in range(n_steps):
        engine.advance((), ())
        xp.take(samples, flat, out=trace[step])
    return np.ascontiguousarray(np.asarray(to_host(trace), dtype=np.float64).T)


def _annulus_wavenumbers(kind, orders, inner, outer, k_min, k_max):
    """{m: [k [1/m]]} of the PEC annulus modes in [k_min, k_max], semi-discrete in phi (the FDFD polar tests' rule: the
    phi difference of exp(-j m phi) acts as the order nu = (2 / dphi) sin(m dphi / 2)): TM_z J_nu(ka) Y_nu(kb) =
    J_nu(kb) Y_nu(ka), TE_z the same in J'_nu / Y'_nu (z-invariant; sign changes on a fine grid refined by brentq)."""
    from scipy.optimize import brentq
    from scipy.special import jv, jvp, yv, yvp
    first, second = (jv, yv) if kind == 'tm' else (jvp, yvp)
    roots = {}
    grid = np.linspace(k_min, k_max, 4001)
    for order in orders:
        def cross(k, nu=2.0 / _D_PHI * math.sin(0.5 * order * _D_PHI)):
            return first(nu, k * inner) * second(nu, k * outer) - first(nu, k * outer) * second(nu, k * inner)
        values = cross(grid)
        roots[order] = [brentq(cross, lo, hi, xtol=1e-14 * k_max)
                        for lo, hi, value_lo, value_hi in zip(grid[:-1], grid[1:], values[:-1], values[1:])
                        if value_lo * value_hi < 0.0]
    return roots


def _peak_ratio(trace):
    """Late / early probe peak over the last and first tenth of the trace."""
    tenth = max(1, trace.shape[1] // 10)
    early = float(np.abs(trace[:, :tenth]).max())
    return float(np.abs(trace[:, -tenth:]).max()) / early if early > 0.0 else math.inf


def _mode_run(backend, name):
    """(error, details) of one ring-down: the worst |f / f_exact - 1| of the expected modes; inf for a missing mode,
    a resonance near an excluded one, a growing trace or the wrong field-set count."""
    from ...core.resonance import harmonic_inversion
    field, n_phi, bloch_ky, initial_orders, expected, excluded = _RUNS[name]
    engine = _sector_engine(backend, n_phi, bloch_ky)
    _initial_field(engine, field, initial_orders)
    trace = _ring_down(engine, field, _STEPS)
    band = tuple(C0 * k / (2.0 * math.pi) for k in _BAND_K)
    found = np.asarray(harmonic_inversion(trace, engine.dt, *band)['frequency'], dtype=np.float64)
    kind = 'tm' if field == 'ez' else 'te'
    wavenumbers = _annulus_wavenumbers(kind, expected + excluded, _INNER, _OUTER, *_BAND_K)
    errors, modes = [], {}
    for order in expected:
        if not wavenumbers[order]:
            errors.append(math.inf)   # the band must hold every expected order
        for k in wavenumbers[order]:
            exact = C0 * k / (2.0 * math.pi)
            measured = float(found[np.argmin(np.abs(found - exact))]) if found.size else math.nan
            error = abs(measured / exact - 1.0) if found.size else math.inf
            errors.append(error)
            modes[f'{kind.upper()}_m{order}_k{k * 1e-6:.4f}'] = {'frequency': measured, 'expected': exact,
                                                                 'error': error}
    intruders = []
    for order in excluded:
        for k in wavenumbers[order]:
            exact = C0 * k / (2.0 * math.pi)
            if found.size and np.min(np.abs(found / exact - 1.0)) <= _ABSENT_WINDOW:
                intruders.append(f'{kind.upper()}_m{order}')
    growth = _peak_ratio(trace)
    imaginary_set = engine.fields.imag is not None
    wants_imaginary = name == 'tm_floquet_quarter'
    error = max(errors)
    if intruders or growth > _GROWTH_LIMIT or imaginary_set != wants_imaginary:
        error = math.inf
    return error, {'modes': modes, 'found': found.tolist(), 'excluded_found': intruders, 'peak_growth': growth,
                   'imaginary_field_set': imaginary_set, 'dt': engine.dt}


def _cpml_run(backend):
    """(error, details): the uniform TM ring-down of a periodic quarter sector radiating into an r_hi CPML decays below
    _CPML_DECAY of its first peak (0 when it does, inf otherwise); k r_outer of the late half's strongest line too."""
    engine = _sector_engine(backend, 16, 0.0, x_hi='cpml')
    _initial_field(engine, 'ez', None)
    trace = _ring_down(engine, 'ez', _CPML_STEPS)
    ratio = _peak_ratio(trace)
    late = trace[:, trace.shape[1] // 2:]
    spectrum = np.abs(np.fft.rfft(late, axis=1)).sum(axis=0)
    frequency = float(np.fft.rfftfreq(late.shape[1], engine.dt)[int(np.argmax(spectrum))])
    return (0.0 if ratio <= _CPML_DECAY else math.inf), {'late_over_early_peak': ratio,
                                                         'late_k_r_outer': 2.0 * math.pi * frequency / C0 * _OUTER}


def _refusals():
    """Names of the sector set-ups that were not refused (radius_min 0, x_lo cpml, unjoined periodic phi faces, a flux
    plane on a sector)."""
    from ...core.domain3d import Domain3D
    from ...core.metric import Coordinates
    from .boundaries3d import BoundaryConfig3D
    from .engine3d import FDTDEngine3D
    from .physics3d import EMPhysics3D

    def domain(config, radius_min=_RADIUS_MIN, angle_wrap='periodic'):
        sides = config.npml_sides
        widths = (np.full(4 + sides['x_lo'] + sides['x_hi'], _DR), np.full(8, _D_PHI), np.full(1, _DZ))
        return Domain3D.from_widths(*widths, sides, size_x=4 * _DR, size_y=8 * _D_PHI, size_z=_DZ,
                                    boundary_config=config, backend='cpu',
                                    coordinates=Coordinates('polar', radius_min=radius_min, angle_wrap=angle_wrap))
    periodic_phi = {'y_lo': 'periodic', 'y_hi': 'periodic', 'z_lo': 'periodic', 'z_hi': 'periodic'}
    closed = BoundaryConfig3D({'x_lo': 'pec', 'x_hi': 'pec', **periodic_phi})
    inner_pml = BoundaryConfig3D({'x_lo': 'cpml', 'x_hi': 'pec', **periodic_phi}, npml=4)
    cases = {'radius_min_0': lambda: FDTDEngine3D(domain(closed, radius_min=0.0), closed),
             'x_lo_cpml': lambda: FDTDEngine3D(domain(inner_pml), inner_pml),
             'periodic_phi_without_wrap': lambda: FDTDEngine3D(domain(closed, angle_wrap='none'), closed)}
    missed = []
    for label, build in cases.items():
        try:
            build()
        except ValueError:
            continue
        missed.append(label)
    sector = domain(closed)
    sector.materials.add_properties({'eps_r': 1.0, 'sigma_e': 0.0, 'mu_r': 1.0})
    try:
        EMPhysics3D(sector).add_flux_plane('z', 0.5 * _DZ)
        missed.append('flux_plane')
    except NotImplementedError:
        pass
    return missed


def _check_cylindrical_sector(_results):
    """Every run on the cpu backend (and the gpu one when usable): worst mode error, inf for a failed sub-check."""
    from ...core.backend import gpu_available
    details, worst = {'refusals_missed': _refusals()}, 0.0
    if details['refusals_missed']:
        worst = math.inf
    for backend in ('cpu', 'gpu') if gpu_available() else ('cpu',):
        for name in _RUNS:
            error, run_details = _mode_run(backend, name)
            details[f'{backend}:{name}'] = run_details
            worst = max(worst, error)
        error, run_details = _cpml_run(backend)
        details[f'{backend}:tm_r_hi_cpml'] = run_details
        worst = max(worst, error)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _TOLERANCE, 'details': details}


CYLINDRICAL_3D_CASES = (
    SelfTestCase(
        name='em3d_cylindrical_sector_cavity',
        label='EM 3D FDTD: cylindrical (r, phi, z) sector cavity',
        description=('em_fdtd_3d engine on polar Domain3D windows (scripts, symmetry 2.21): a vacuum PEC coaxial annulus '
                     f'{_INNER * 1e6:g}-{_OUTER * 1e6:g} um (walls at the centres of the pec layer cells; {_N_R} '
                     f'radial cells of {_DR * 1e9:.4g} nm, phi cells of pi/32, one z-periodic cell) rings down from a '
                     'first-radial-order E_z (TM_z) or H_z (TE_z) pattern for '
                     f'{_STEPS} steps; harmonic inversion of three probes in k = 5.5-7.2 / um vs the Bessel '
                     'cross-product roots J_nu(ka) Y_nu(kb) = J_nu(kb) Y_nu(ka) (TE: J_nu\', Y_nu\'), semi-discrete '
                     'in phi: nu = (2 / dphi) sin(m dphi / 2). Runs: full turn '
                     'periodic, TM orders 0, 2, 3 and TE 1, 3; a quarter turn with bloch_ky = 2 (antiperiodic, the '
                     'real wrap: no imaginary field set), only TM_2 (TM_0, _1, _3 absent); a quarter turn with '
                     'bloch_ky = 1 (complex Floquet phase pi/2), TM_1 and TM_3 (orders 1 + 4n) with TM_0 / TM_2 '
                     f'absent; an r_hi CPML quarter turn whose ring-down must fall below {_CPML_DECAY:g} (measured '
                     '1.2e-2: the d/dr-only stretch is approximate at small k r; no absorber leaves ~1); refusals '
                     '(radius_min 0, x_lo cpml, periodic phi faces without angle_wrap, a flux plane). error = the '
                     f'worst |f / f_exact - 1| (inf for a missing / intruding mode, late / early peak > '
                     f'{_GROWTH_LIMIT:g}, a failed refusal), tolerance {_TOLERANCE:g}: dr and dt errors are ~4e-4, '
                     'a missing r factor in a curl shifts f by >= 1 %; measured <= 1e-4. cpu backend, plus the gpu '
                     'one when usable.'),
        specs=lambda: {}, check=_check_cylindrical_sector, heavy=True, max_wall_time=900.0, backends=('cpu', 'gpu')),
)

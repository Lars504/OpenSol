"""Material self-tests of em_fdtd_3d (B2 materials): dipole power in a uniaxial crystal (eps_xx / eps_yy / eps_zz), the
3D EM + heat chain (em_heat_joule absorbed power of a weak absorber against the exact dipole near field,
heat_em_thermo_optic through a uniformly heated medium) and the uint16 index re-quantisation of a continuous eps_r."""
import math

import numpy as np

from ...core.constants import C0, EPS0, ETA0
from ...model.schema import SelfTestCase
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec,
                           ModelSpec, PhysicsSpec, StudySpec)
from .selftests import DEFAULT_COURANT, cubic_time_step, dipole_flux_errors

_EM = 'em'
_HEAT = 'heat'
_EM_TYPE = 'em_fdtd_3d'               # literals (selftests never import the definitions)
_FLUX_FEATURE = 'radiated_power'
_CELL = 40e-9                         # m, cubic cells
_PML_CELLS = 10
_DT = cubic_time_step(_CELL)         # the 3D Courant step of the uniform grid (default courant_factor)
_MOMENT = 1e-12                       # A m (= selftests._MOMENT, which dipole_flux_errors assumes)
_TOLERANCE = 0.05

# ---- uniaxial crystal: eps_xx = eps_yy = eps_t, eps_zz = eps_z everywhere (CPML included) ----
_EPS_T, _EPS_Z = 2.25, 4.0
_UNIAXIAL_SIZE = 1.2e-6               # m cube (30 cells), dipole at the centre, flux box 0.2 - 1.0 um
_UNIAXIAL_BOX = (0.2e-6, 1.0e-6)
_UNIAXIAL_WAVELENGTH = 2.0e-6         # m vacuum: >= 22 cells per wavelength in the crystal (n <= 2)
_UNIAXIAL_BAND = (1.8e-6, 2.2e-6, 3)
# s: pulse 2 t0 = 38 fs at 2 um plus the box transit at n = 2 (2.7 fs); validation bound 2 t0 + diagonal x 2 / c = 52 fs
_UNIAXIAL_DURATION = 60e-15


def _uniaxial_power_ratio(axis):
    """Radiated power of a dipole along axis in the uniaxial crystal over the vacuum value eta0 k0^2 (I l)^2 /
    (12 pi): z (optic axis) sqrt(eps_t); x or y 3/4 sqrt(eps_t) + 1/4 eps_z / sqrt(eps_t) (ordinary + extraordinary)."""
    if axis == 'z':
        return math.sqrt(_EPS_T)
    return 0.75 * math.sqrt(_EPS_T) + 0.25 * _EPS_Z / math.sqrt(_EPS_T)


def _flux_box_spec(title, size, materials, wavelength, band, duration, axis='z', box=None, extra_physics=(),
                   couplings=()):
    """em_fdtd_3d cube of `size` with a gaussian dipole along axis at the centre and a closed flux box."""
    centre = 0.5 * size
    lo, hi = box if box is not None else (size / 6.0, size * 5.0 / 6.0)
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size, size_y=size, size_z=size),
        geometry=[GeometrySpec('dipole_point', 'point', {'x': centre, 'y': centre, 'z': centre}),
                  GeometrySpec('flux_surface', 'box', {'x0': lo, 'y0': lo, 'z0': lo, 'x1': hi, 'y1': hi, 'z1': hi})],
        materials=materials,
        physics=[*extra_physics, PhysicsSpec(_EM, _EM_TYPE, settings={'pml_cells': _PML_CELLS}, features=[
            FeatureSpec('dipole', 'dipole_source', ['dipole_point'],
                        {'dipole_axis': axis, 'waveform': 'gaussian', 'wavelength': wavelength, 'bandwidth': 0.5,
                         'current_moment': _MOMENT}),
            FeatureSpec(_FLUX_FEATURE, 'flux_box', ['flux_surface'],
                        {'wavelength_start': band[0], 'wavelength_stop': band[1], 'n_wavelengths': band[2]})])],
        couplings=list(couplings),
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': duration}))


def _uniaxial_specs():
    crystal = [MaterialSpec('crystal', properties={'eps_xx': _EPS_T, 'eps_yy': _EPS_T, 'eps_zz': _EPS_Z},
                            selection=['all'])]
    return {f'dipole_{axis}': _flux_box_spec(f'Self-test: {axis} dipole in a uniaxial crystal (3D)', _UNIAXIAL_SIZE,
                                             crystal, _UNIAXIAL_WAVELENGTH, _UNIAXIAL_BAND, _UNIAXIAL_DURATION, axis,
                                             _UNIAXIAL_BOX)
            for axis in ('z', 'x')}


def _check_uniaxial(results_by_name):
    errors, details = {}, {}
    for axis in ('z', 'x'):
        errors[axis], details[axis] = dipole_flux_errors(results_by_name[f'dipole_{axis}'],
                                                         _uniaxial_power_ratio(axis))
        details[axis]['power_ratio'] = _uniaxial_power_ratio(axis)
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'relative_errors': errors, **details}}


# ---- 3D EM + heat chain ----
_T0 = 300.0                           # K
_CONDUCTIVITY = 1.0                   # W/(m*K) everywhere (heat only needs an anchored conductor)
# joule run: cw z dipole in vacuum, weak absorber cube in its equatorial near field, stationary heat every _HEAT_EVERY
_JOULE_SIZE = 1.6e-6                  # m cube (40 cells): the absorber stays 8 cells from the CPML
_JOULE_WAVELENGTH = 1.0e-6            # m
_DIPOLE_POSITION = (0.82e-6, 0.82e-6, 0.80e-6)   # cell centre in x, y (containing cell of the mid point), z face
_ABSORBER = ((1.12e-6, 1.28e-6), (0.72e-6, 0.92e-6), (0.68e-6, 0.92e-6))   # m, cell edges: 4 x 5 x 6 cells
_SIGMA_OVER_OMEGA_EPS0 = 0.05         # weak loss: the field inside differs by ~|3 / (eps_c + 2)|^2 - 1 ~ 3e-4
# heat every = 24 half periods of the cw source in EM steps (rounded): the window mean of cos^2 is off by ~4e-4
_HEAT_HALF_PERIODS = 24
_HEAT_EVERY = int(round(_HEAT_HALF_PERIODS * _JOULE_WAVELENGTH / (2.0 * C0 * _DT)))
# the last window starts after 24 periods: cw ramp 1 - exp(-t / 3T) leaves exp(-8) ~3e-4 in amplitude, ~7e-4 in power
_JOULE_WINDOWS = 3
# thermo-optic run: glass n0 = 1.2 heated uniformly by dT through heat walls: n = n0 + dndT dT = 1.5
_GLASS_INDEX = 1.2
_DNDT = 1e-3                          # 1/K
_RISE = 300.0                         # K
_THERMO_WAVELENGTH = 1.5e-6           # m vacuum: 25 cells per wavelength at n = 1.5
_THERMO_BAND = (1.35e-6, 1.65e-6, 3)
_THERMO_DURATION = 45e-15             # s: pulse 2 t0 = 29 fs at 1.5 um; validation bound 2 t0 + diagonal x 1.5 / c = 39 fs


def _sigma():
    return _SIGMA_OVER_OMEGA_EPS0 * 2.0 * math.pi * C0 / _JOULE_WAVELENGTH * float(EPS0)


def _heat_physics(wall_temperature, every):
    return PhysicsSpec(_HEAT, 'heat', settings={'T0': _T0, 'time_mode': 'stationary'}, every=every, features=[
        FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi'],
                    {'temperature': wall_temperature})])


def _joule_spec():
    """cw z dipole + absorber cube, em_heat_joule into stationary heat; derived absorbed power of EM and heat Q."""
    size = _JOULE_SIZE
    (x0, x1), (y0, y1), (z0, z1) = _ABSORBER
    x, y, z = _DIPOLE_POSITION
    return ModelSpec(
        title='Self-test: 3D dipole near field heating a weak absorber (em_heat_joule)',
        domain=DomainSpec(size_x=size, size_y=size, size_z=size),
        geometry=[GeometrySpec('dipole_point', 'point', {'x': x, 'y': y, 'z': z}),
                  GeometrySpec('absorber', 'box', {'x0': x0, 'y0': y0, 'z0': z0, 'x1': x1, 'y1': y1, 'z1': z1})],
        materials=[MaterialSpec('vacuum', properties={'k': _CONDUCTIVITY}, selection=['all']),
                   MaterialSpec('absorber', properties={'k': _CONDUCTIVITY, 'sigma_e': _sigma()},
                                selection=['absorber'])],
        physics=[PhysicsSpec(_EM, _EM_TYPE, settings={'pml_cells': _PML_CELLS}, features=[
                     FeatureSpec('dipole', 'dipole_source', ['dipole_point'],
                                 {'dipole_axis': 'z', 'waveform': 'cw', 'wavelength': _JOULE_WAVELENGTH,
                                  'current_moment': _MOMENT})]),
                 _heat_physics(_T0, _HEAT_EVERY)],
        couplings=[CouplingSpec('joule', 'em_heat_joule', _EM, _HEAT)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _JOULE_WINDOWS * _HEAT_EVERY * _DT}),
        derived_values=[DerivedValueSpec('P_heat', 'Q', ['absorber'], 'integral'),
                        DerivedValueSpec('P_em', 'absorbed_power', ['absorber'], 'integral')])


def _thermo_optic_spec():
    glass = [MaterialSpec('glass', properties={'eps_r': _GLASS_INDEX ** 2, 'k': _CONDUCTIVITY, 'dndT': _DNDT},
                          selection=['all'])]
    return _flux_box_spec('Self-test: 3D dipole in uniformly heated glass (heat_em_thermo_optic)', 1.2e-6, glass,
                          _THERMO_WAVELENGTH, _THERMO_BAND, _THERMO_DURATION,
                          extra_physics=(_heat_physics(_T0 + _RISE, 1000),),
                          couplings=(CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _EM),))


def _chain_specs():
    return {'joule': _joule_spec(), 'thermo_optic': _thermo_optic_spec()}


def _dipole_field(x, y, z, wavenumber):
    """Phasor E_x, E_y, E_z [V/m] of a z Hertzian dipole _MOMENT at the origin in vacuum (exact, e^{j omega t})."""
    r = np.sqrt(x * x + y * y + z * z)
    rho = np.sqrt(x * x + y * y)
    kr = wavenumber * r
    phase = np.exp(-1j * kr)
    cos_t, sin_t = z / r, rho / r
    e_r = float(ETA0) * _MOMENT * cos_t / (2.0 * math.pi * r ** 2) * (1.0 + 1.0 / (1j * kr)) * phase
    e_t = (1j * float(ETA0) * wavenumber * _MOMENT * sin_t / (4.0 * math.pi * r)
           * (1.0 + 1.0 / (1j * kr) - 1.0 / kr ** 2) * phase)
    e_rho = e_r * sin_t + e_t * cos_t
    return e_rho * x / rho, e_rho * y / rho, e_r * cos_t - e_t * sin_t


def _expected_absorbed_power():
    """sum over the absorber cells of V 0.5 sigma sum_c (<E_c^2> on the hi face + on the lo face), <E^2> = |E|^2 / 2:
    the cell split of the FDTD ohmic loss with the exact vacuum dipole field at the face positions."""
    wavenumber = 2.0 * math.pi / _JOULE_WAVELENGTH
    centres = []
    for lo, hi in _ABSORBER:
        first, last = int(round(lo / _CELL)), int(round(hi / _CELL))
        centres.append((np.arange(first, last) + 0.5) * _CELL)
    grids = np.meshgrid(*centres, indexing='ij')
    relative = [grid - origin for grid, origin in zip(grids, _DIPOLE_POSITION)]
    mean_square = 0.0
    for axis in range(3):
        for offset in (-0.5 * _CELL, 0.5 * _CELL):
            shifted = list(relative)
            shifted[axis] = relative[axis] + offset
            mean_square = mean_square + 0.5 * np.abs(_dipole_field(*shifted, wavenumber)[axis]) ** 2
    return float(np.sum(0.5 * _sigma() * mean_square) * _CELL ** 3)


def _check_chain(results_by_name):
    """Absorbed power of the joule run (heat Q and EM absorbed_power integrals over the absorber) against the exact
    near-field value, and the thermo-optic run's radiated power against n eta0 k0^2 (I l)^2 / (12 pi)."""
    derived = results_by_name['joule'].summary()['derived']
    power_heat, power_em = float(derived['P_heat']), float(derived['P_em'])   # float() also reads 'nan'
    expected = _expected_absorbed_power()
    errors = {'joule_absorbed': abs(power_heat / expected - 1.0), 'joule_transfer': abs(power_em / power_heat - 1.0)}
    index = _GLASS_INDEX + _DNDT * _RISE
    errors['thermo_optic'], thermo_details = dipole_flux_errors(results_by_name['thermo_optic'], index)
    errors = {name: (value if math.isfinite(value) else math.inf) for name, value in errors.items()}
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'relative_errors': errors, 'P_heat_W': power_heat, 'P_em_W': power_em,
                        'P_expected_W': expected, 'heat_every': _HEAT_EVERY, 'heated_index': index,
                        'thermo_optic': thermo_details}}


# ---- re-quantisation of a continuous eps_r (engine only, no model) ----
_QUANTISATION_CELLS = 48              # cube of 48^3 = 110592 cells, more distinct values than the uint16 index holds
_QUANTISATION_SPREAD = (2.0, 2.5)     # random eps_r range; the first x layer stays exactly 1.0 (air)
_INDEX_LIMIT = 65534                  # distinct combinations of the index (engine3d.MAX_MATERIAL_IDS - 1)


def _check_requantisation(_results):
    """Random eps_r in [2, 2.5] on 48^3 cells (pmc box) plus one layer of exactly 1.0, on the cpu backend and (with a
    usable GPU) the gpu: the rebuilt index must fit, keep 1.0 exact and deviate by at most the bin width ~ spread /
    limit (10 % bisection slack + 25 %); the worst backend counts."""
    from ...core.backend import gpu_available, to_host
    from ...core.constants import FACES_3D
    from ...core.domain3d import Domain3D
    from .boundaries3d import BoundaryConfig3D
    from .engine3d import FDTDEngine3D

    config = BoundaryConfig3D({face: 'pmc' for face in FACES_3D})
    size = _QUANTISATION_CELLS * _CELL
    low, high = _QUANTISATION_SPREAD
    tolerance = 1.25 * 1.1 * (high - low) / (_INDEX_LIMIT - 1)
    measured, error, details = 0.0, 0.0, {}
    for backend in ('cpu', 'gpu') if gpu_available() else ('cpu',):
        domain = Domain3D.uniform(size, size, size, _CELL, npml_sides={}, courant=DEFAULT_COURANT,
                                  boundary_config=config)
        domain.use_backend(backend)
        domain.materials.add_properties({'eps_r': 1.0, 'sigma_e': 0.0, 'mu_r': 1.0})
        values = np.random.default_rng(7).uniform(low, high, domain.materials.shape).astype(np.float32)
        values[0] = 1.0
        domain.materials.eps_r[...] = domain.xp.asarray(values)
        engine = FDTDEngine3D(domain, config)
        engine.rebuild_materials()
        table_values = to_host(engine.table_eps_z[engine.material_id])
        deviation = float(np.abs(table_values - values).max())
        air_exact = bool((table_values[0] == 1.0).all())
        n_ids = int(engine.table_eps_z.size) - 1
        reported = engine.quantisation_errors.get('eps_r', math.nan)
        consistent = air_exact and n_ids <= _INDEX_LIMIT and abs(reported - deviation) <= 1e-6
        measured = max(measured, deviation)
        error = max(error, deviation if consistent else math.inf)
        details[backend] = {'deviation': deviation, 'n_material_ids': n_ids, 'reported_error': reported,
                            'air_layer_exact': air_exact}
    details['distinct_input_values'] = int(np.unique(values).size)
    return {'measured': measured, 'expected': 0.0, 'error': error, 'tolerance': tolerance, 'details': details}


# ---- Drude velocity form of the em3d pole kernel (the 2D case's scenarios, em.selftests_operators) ----
def _drude_3d_steppers(xp):
    """{kernel: (shape, step)} of kernels3d.drude_j_weighted at face weight 1."""
    from .kernels3d import drude_j_weighted
    shape = (3, 3, 3)
    weight = xp.ones(shape, dtype=np.float32)
    return {'em3d_weighted': (shape, lambda damping, drive, e, j: drude_j_weighted(damping, drive, weight, e, j))}


def _check_drude_velocity_form(_results):
    """em3d Drude J kernel holds sigma_dc E at gamma dt 2e-6 and follows a damped transient (relative 1e-5)."""
    from ..em.selftests_operators import drude_velocity_form_errors
    errors = drude_velocity_form_errors(_drude_3d_steppers)
    measured = max(errors.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': 1e-5, 'details': errors}


MATERIAL_CASES = (
    SelfTestCase(
        name='em3d_uniaxial_dipole_power',
        label='EM 3D: dipole power in a uniaxial crystal (eps_xx, eps_yy, eps_zz)',
        description=(f'em_fdtd_3d cube 1.2 um (40 nm cells, 10 CPML cells per face) filled with a uniaxial crystal '
                     f'eps_xx = eps_yy = {_EPS_T:g}, eps_zz = {_EPS_Z:g} (CPML included), a gaussian dipole (I l = '
                     f'1e-12 A m, 2 um, bandwidth 0.5) at the centre along z (run dipole_z) or x (dipole_x), flux box '
                     f'0.2-1.0 um, {_UNIAXIAL_DURATION * 1e15:g} fs. power_per_cw at 1.8, 2.0, 2.2 um vs the '
                     'spectral-integral closed forms over '
                     'the vacuum eta0 k0^2 (I l)^2 / (12 pi): z (optic axis) sqrt(eps_t) = 1.5, x 3/4 sqrt(eps_t) + '
                     '1/4 eps_z / sqrt(eps_t) = 1.79 (ordinary + extraordinary waves). Error = worst relative error, '
                     'tolerance 5e-2 (lattice dispersion at >= 22 cells per wavelength ~1-2 %). Isotropic eps 2.25 or '
                     '4 everywhere, or swapped components, miss the x run by >= 16 % or the z run by 33 %. Not yet '
                     f'run at {_UNIAXIAL_DURATION * 1e15:g} fs.'),
        specs=_uniaxial_specs, check=_check_uniaxial),
    SelfTestCase(
        name='em3d_heat_chain',
        label='EM 3D + heat: absorbed near-field power and thermo-optic index',
        description=('Run joule: em_fdtd_3d vacuum cube 1.6 um (40 nm cells, 10 CPML), cw z dipole (1e-12 A m, 1 um) '
                     'at (0.82, 0.82, 0.80) um, a 4 x 5 x 6 cell absorber (sigma_e = 0.05 omega eps0) 7.5-11.5 cells '
                     'away in the equatorial plane, em_heat_joule into stationary 3D heat (every = '
                     f'{_HEAT_HALF_PERIODS} half periods, {_JOULE_WINDOWS} windows, walls at T0; the last window\'s cw '
                     'ramp residual ~7e-4 in power). The heat Q integral over the absorber vs the exact Hertzian-dipole '
                     'field (all near- and far-field terms) at the E face positions, split per cell as the FDTD '
                     'ohmic loss: sum V sigma/2 sum_c (<E^2>_hi + <E^2>_lo); also the EM absorbed_power integral vs '
                     'the heat Q (transfer). Run thermo_optic: glass n0 = 1.2, dndT = 1e-3 /K everywhere, heat walls '
                     'at T0 + 300 K (uniform T), heat_em_thermo_optic: the flux-box power of a gaussian dipole (1.5 '
                     'um) vs n eta0 k0^2 (I l)^2 / (12 pi) with n = 1.5. Error = worst relative error, tolerance '
                     '5e-2 (FDTD near field at >= 7.5 cells ~1 %, CPML reflection, lattice dispersion); a missing '
                     'or doubled cell share, a wrong window mean or an unheated index (n = 1.2: 20 %) fails. Not yet '
                     f'run at {_HEAT_HALF_PERIODS} half periods per window / thermo_optic '
                     f'{_THERMO_DURATION * 1e15:g} fs.'),
        specs=_chain_specs, check=_check_chain),
    SelfTestCase(
        name='em3d_material_requantisation',
        label='EM 3D: uint16 material index re-quantisation of a continuous eps_r',
        description=('Engine only (no model run): a 48^3 pmc box with eps_r uniformly random in [2, 2.5] (~110k '
                     'distinct float32 values, more than the 65534 index entries) and one x layer of exactly 1.0. '
                     'rebuild_materials must re-quantise eps_r (occupied bins of the smallest width that fits, '
                     'member mean as value): measured = max |table eps - eps_r|, tolerance 1.375 x 0.5 / 65533 '
                     '(bin width with the 10 % bisection slack); error infinite unless the 1.0 layer stays exact, '
                     'the id count fits and quantisation_errors reports the measured deviation. Runs on the cpu '
                     'backend (numpy) and, with a usable GPU, on the gpu; the worst counts.'),
        specs=lambda: {}, check=_check_requantisation, backends=('cpu', 'gpu')),
    SelfTestCase(
        name='em3d_drude_velocity_form',
        label='EM 3D: Drude velocity form of the pole kernel',
        description=('Kernel only (no model run): the em3d drude_j_weighted update (face weight 1, float32, cpu twin '
                     'and, with a usable GPU, the CUDA kernel) in the scenarios of em_drude_velocity_form: constant E '
                     '= 1 V/m, omega_p 1.37e16 rad/s, dt 1e-17 s; gamma dt 2e-6 held at sigma_dc E for 10000 steps, '
                     'and gamma dt 1e-2 charging from J = 0 for 300 steps against the float64 leap-frog recursion. '
                     'error = the worst relative J error, tolerance 1e-5 (a float32 c1 J + c2 E drifts ~3e-4).'),
        specs=lambda: {}, check=_check_drude_velocity_form, backends=('cpu', 'gpu')),
)

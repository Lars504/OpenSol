"""C2 stage 1 free carriers in silicon: Soref-Bennett plasma-dispersion eps_r / sigma_e expressions, the two pn phase
shifter templates (prescribed depletion carriers; carriers from the electrostatics V) and their em_modes self-tests."""
import math

import numpy as np

from ...core.constants import C0, ELEMENTARY_CHARGE, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result, with_exact_cg
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, StudySpec, SweepSpec)

_MODES_TYPE = 'em_modes'               # = definition_modes.EM_MODES.name (definition_modes imports this module)
_ELECTROSTATICS_TYPE = 'electrostatics'
_MODES = 'modes'                       # physics instance names = the types' default instances
_ELECTROSTATICS = 'electrostatics'
_TABLE = 'table'
_RUN = 'run'

# ---- Soref-Bennett plasma dispersion of crystalline silicon at 1.55 um ----
# R. A. Soref and B. R. Bennett, "Electrooptical effects in silicon", IEEE J. Quantum Electron. 23, 123 (1987); the
# 1.55 um fit as quoted by G. T. Reed et al., Nat. Photonics 4, 518 (2010). Densities N (electrons), P (holes) in
# cm^-3: dn = -(8.8e-22 N + 8.5e-18 P^0.8), dalpha = 8.5e-18 N + 6.0e-18 P [1/cm] (power absorption). The EM material
# takes n = n0 + dn as eps_r = n^2 and alpha [1/m] as sigma_e = n alpha eps0 c, i.e. the complex eps n^2 - j n alpha /
# k0 = (n - j kappa)^2 + kappa^2 with kappa = alpha / (2 k0): exact up to kappa^2 (~1e-8). sigma_e is wavelength
# independent, so alpha stays at its 1.55 um value across a frequency_domain sweep (the fit is a 1.55 um fit anyway).
SOREF_BENNETT_1550 = {'electron_index': 8.8e-22, 'hole_index': 8.5e-18, 'hole_exponent': 0.8,
                      'electron_absorption': 8.5e-18, 'hole_absorption': 6.0e-18}
_N_SILICON = 3.476                     # undoped silicon index at 1.55 um (library silicon eps_r 12.08)
_WAVELENGTH = 1.55e-6                  # m
_THERMAL_VOLTAGE = 0.025852            # V, kT / q at 300 K
_INTRINSIC_DENSITY = 1e16              # m^-3, silicon at 300 K (1e10 cm^-3)
_EPS_SILICON = 11.7                    # static relative permittivity (= library silicon eps_dc)


def plasma_dispersion(n0, electrons, holes):
    """(eps_r, sigma_e [S/m]) of silicon of index n0 with electron / hole densities [m^-3] (numbers or numpy arrays)."""
    coefficients = SOREF_BENNETT_1550
    electrons_cm, holes_cm = 1e-6 * electrons, 1e-6 * holes
    index = (n0 - coefficients['electron_index'] * electrons_cm
             - coefficients['hole_index'] * holes_cm ** coefficients['hole_exponent'])
    absorption = 100.0 * (coefficients['electron_absorption'] * electrons_cm
                          + coefficients['hole_absorption'] * holes_cm)   # 1/m
    return index ** 2, index * absorption * EPS0 * C0


def plasma_dispersion_expressions(n0, electrons=None, holes=None):
    """(eps_r, sigma_e) expression texts of plasma_dispersion; n0 and the densities [m^-3] are expression texts, None =
    no such carriers."""
    coefficients = SOREF_BENNETT_1550
    index_terms, absorption_terms = [], []
    if electrons is not None:
        index_terms.append(f"{coefficients['electron_index']!r} * (1e-6 * ({electrons}))")
        absorption_terms.append(f"{coefficients['electron_absorption']!r} * (1e-6 * ({electrons}))")
    if holes is not None:
        index_terms.append(f"{coefficients['hole_index']!r} * (1e-6 * ({holes}))**{coefficients['hole_exponent']!r}")
        absorption_terms.append(f"{coefficients['hole_absorption']!r} * (1e-6 * ({holes}))")
    if not index_terms:
        raise ValueError('plasma_dispersion_expressions: give electrons and / or holes')
    index = f"({n0} - {' - '.join(index_terms)})"
    return f'{index}**2', f"{index} * 100 * ({' + '.join(absorption_terms)}) * eps0_const * c_const"


# ---- abrupt pn junction, depletion approximation (shared by the templates and carriers_depletion_junction) ----
# p side x < x_j (acceptors N_A), n side x >= x_j (donors N_D). V_bi = V_T ln(N_A N_D / n_i^2), W = sqrt(2 eps_s
# (V_bi + V_R) / q (N_A + N_D) / (N_A N_D)), x_n = W N_A / (N_A + N_D), x_p = W N_D / (N_A + N_D); space charge
# -q N_A on [x_j - x_p, x_j), +q N_D on [x_j, x_j + x_n). Electrostatics: p contact at 0 V, n contact at V_bi + V_R (the
# electrostatic potential, so the neutral regions stay field-free). Carriers: prescribed = 0 inside the depletion
# region, the doping outside; from V = Boltzmann with the quasi-Fermi level of each side flat at its contact (reverse
# bias), p = N_A exp(min(-V, 0) / V_T), n = N_D exp(min(V - V_bi - V_R, 0) / V_T): the doping in the neutral regions,
# the Debye tails of the depletion edges inside (the V_T -> 0 limit is the prescribed profile).
_ACCEPTOR_CHARGE = 'where(x >= x_j - x_p, -q_e * N_A, 0)'   # C/m^3 on the p objects
_DONOR_CHARGE = 'where(x < x_j + x_n, q_e * N_D, 0)'        # C/m^3 on the n objects
_PRESCRIBED_CARRIERS = {'electrons': 'where(x < x_j + x_n, 0, N_D)', 'holes': 'where(x >= x_j - x_p, 0, N_A)'}
_BOLTZMANN_CARRIERS = {'electrons': f'N_D * exp(min({_ELECTROSTATICS}_V - V_bi - V_R, 0) / V_T)',
                       'holes': f'N_A * exp(min(-{_ELECTROSTATICS}_V, 0) / V_T)'}


def _junction_parameters(acceptors, donors, reverse_bias, junction_x):
    """ParameterSpecs of the abrupt junction (doping [m^-3], V_R [V], x_j [m]) and its depletion quantities."""
    return [ParameterSpec('N_A', acceptors, 'acceptor density of the p side [m^-3] (1e24 m^-3 = 1e18 cm^-3)'),
            ParameterSpec('N_D', donors, 'donor density of the n side [m^-3]'),
            ParameterSpec('V_R', reverse_bias, 'reverse bias [V] (> 0 widens the depletion region)'),
            ParameterSpec('x_j', junction_x, 'junction position [m] (p side below, n side above)'),
            ParameterSpec('n_Si', _N_SILICON, 'refractive index of undoped silicon at 1.55 um'),
            ParameterSpec('eps_si', _EPS_SILICON, 'static relative permittivity of silicon'),
            ParameterSpec('n_i', _INTRINSIC_DENSITY, 'intrinsic carrier density of silicon at 300 K [m^-3]'),
            ParameterSpec('V_T', _THERMAL_VOLTAGE, 'thermal voltage kT / q at 300 K [V]'),
            ParameterSpec('q_e', ELEMENTARY_CHARGE, 'elementary charge [C]'),
            ParameterSpec('V_bi', 'V_T * log(N_A * N_D / n_i**2)', 'built-in potential [V]'),
            ParameterSpec('W_dep', 'sqrt(2 * eps_si * eps0_const * (V_bi + V_R) / q_e * (N_A + N_D) / (N_A * N_D))',
                          'depletion width [m] (abrupt junction, depletion approximation)'),
            ParameterSpec('x_n', 'W_dep * N_A / (N_A + N_D)', 'depletion depth on the n side [m]'),
            ParameterSpec('x_p', 'W_dep * N_D / (N_A + N_D)', 'depletion depth on the p side [m]')]


def _doped_silicon_materials(carriers, p_objects, n_objects):
    """Library silicon with Soref-Bennett eps_r / sigma_e of the holes on p_objects and the electrons on n_objects."""
    eps_p, sigma_p = plasma_dispersion_expressions('n_Si', holes=carriers['holes'])
    eps_n, sigma_n = plasma_dispersion_expressions('n_Si', electrons=carriers['electrons'])
    return [MaterialSpec('p_silicon', library='silicon', properties={'eps_r': eps_p, 'sigma_e': sigma_p},
                         selection=list(p_objects)),
            MaterialSpec('n_silicon', library='silicon', properties={'eps_r': eps_n, 'sigma_e': sigma_n},
                         selection=list(n_objects))]


def _depletion_electrostatics(p_objects, n_objects):
    """Electrostatics of the depletion charge between the terminals contact_p (0 V) and contact_n (V_bi + V_R)."""
    return PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS_TYPE, features=[
        FeatureSpec('contact_p', 'terminal', ['contact_p'], {'voltage': 0.0}),
        FeatureSpec('contact_n', 'terminal', ['contact_n'], {'voltage': 'V_bi + V_R'}),
        FeatureSpec('acceptors', 'charge_density', list(p_objects), {'density': _ACCEPTOR_CHARGE}),
        FeatureSpec('donors', 'charge_density', list(n_objects), {'density': _DONOR_CHARGE})])


def _rectangle(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _pec_mode_physics():
    """em_modes with pec on every face (no PML), one mode, mode table 'table'."""
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings.update(wavelength=_WAVELENGTH, n_modes=1)
    return PhysicsSpec(_MODES, _MODES_TYPE, settings=settings, features=[FeatureSpec(_TABLE, 'mode_table')])


# ---- templates: lateral pn junction across a silicon rib (cross-section, em_modes) ----
# 500 x 220 nm rib on a 90 nm slab (rib x 1.75..2.25 um, slab y 0.60..0.69 um, rib top 0.82 um) in silica, junction
# at the rib centre x_j = 2.0 um, N_A = N_D = 5e23 m^-3 (5e17 cm^-3): V_bi ~0.92 V, W ~69 nm at 0 V and ~159 nm at
# 4 V reverse bias (x_n = x_p = W / 2). Uniform 10 nm cells (3-8 cells per depletion depth), 4.0 x 1.6 um window +
# 15 PML cells per side = 430 x 190 cells (163,400 em_modes unknowns, far below the 1e6 scipy_splu warning); every
# geometry edge on a cell edge. The slab runs into the PML on both sides; the electrostatics variant holds the slab
# contacts x 0.4..0.9 um (p) and 3.1..3.6 um (n). Estimate: the depletion edges move ~45 nm each way between 0 and 4 V
# through the mode centre, dn_eff ~1e-4 (~0.6 rad per mm at 1.55 um, V_pi L ~2 V cm); doped-silicon loss ~7 / cm
# (power, ~30 dB/cm) times the confinement.
_SHIFTER_CELL = 10e-9                          # m
_SHIFTER_DOMAIN = (4.0e-6, 1.6e-6)             # m
_SHIFTER_SLAB_Y = (0.60e-6, 0.69e-6)           # m
_SHIFTER_RIB = (1.75e-6, 2.25e-6, 0.82e-6)     # m, (x0, x1, top y)
_SHIFTER_JUNCTION_X = 2.0e-6                   # m
_SHIFTER_CONTACTS_X = {'p': (0.4e-6, 0.9e-6), 'n': (3.1e-6, 3.6e-6)}   # m
_SHIFTER_DOPING = 5e23                         # m^-3, N_A = N_D
_SHIFTER_BIASES = (0.0, 2.0, 4.0)              # V, sweep of V_R
_SHIFTER_SETTINGS = {'wavelength': _WAVELENGTH, 'n_modes': 2, 'pml_sigma_factor': 0.7}   # 0.7 ~ 1 / n of silica
_SHIFTER_P_OBJECTS = ('slab_p', 'rib_p')
_SHIFTER_N_OBJECTS = ('slab_n', 'rib_n')

PN_PRESCRIBED_LABEL = 'PN phase shifter (prescribed carriers)'
PN_PRESCRIBED_DESCRIPTION = (
    'EM cross-section modes, mode_analysis with a sweep over the reverse bias V_R (0, 2, 4 V): a 500 x 220 nm silicon '
    'rib on a 90 nm slab in silica with a lateral pn junction at its centre (N_A = N_D = 5e17 cm^-3). The carrier '
    'densities are prescribed by the depletion approximation in closed form (none inside the depletion width W(V_R), '
    'the doping outside) and enter eps_r / sigma_e of silicon through the Soref-Bennett plasma dispersion at 1.55 um; '
    'derived n_eff_1 and loss_1 (dB/cm) per bias (dn_eff ~1e-4 between 0 and 4 V).')
PN_ELECTROSTATICS_LABEL = 'PN phase shifter (electrostatics)'
PN_ELECTROSTATICS_DESCRIPTION = (
    'Electrostatics + EM cross-section modes, mode_analysis with a sweep over V_R (0, 2, 4 V): the rib of '
    "'pn_phase_shifter_depletion', but the electrostatics solves V of the depletion-approximation space charge "
    'between slab contacts at 0 V and V_bi + V_R, and the carriers follow V by Boltzmann statistics (n = N_D exp((V - '
    'V_bi - V_R) / V_T), p = N_A exp(-V / V_T), clamped at the doping): smooth depletion edges and the 2D field of the '
    'rib, read by eps_r / sigma_e (Soref-Bennett) as state-dependent materials; derived n_eff_1 and loss_1.')


def _pn_phase_shifter_spec(from_electrostatics):
    """Rib cross-section with a lateral pn junction (em_modes, mode_analysis, sweep V_R); carriers prescribed or from
    the electrostatics V."""
    size_x, size_y = _SHIFTER_DOMAIN
    slab_y0, slab_y1 = _SHIFTER_SLAB_Y
    rib_x0, rib_x1, rib_y1 = _SHIFTER_RIB
    junction_x = _SHIFTER_JUNCTION_X
    geometry = [_rectangle('slab_p', 0.0, slab_y0, junction_x, slab_y1),
                _rectangle('slab_n', junction_x, slab_y0, size_x, slab_y1),
                _rectangle('rib_p', rib_x0, slab_y0, junction_x, rib_y1),
                _rectangle('rib_n', junction_x, slab_y0, rib_x1, rib_y1)]
    physics = [PhysicsSpec(_MODES, _MODES_TYPE, settings=dict(_SHIFTER_SETTINGS),
                           features=[FeatureSpec('mode_table', 'mode_table')])]
    if from_electrostatics:
        geometry += [_rectangle(f'contact_{side}', x0, slab_y0, x1, slab_y1)
                     for side, (x0, x1) in _SHIFTER_CONTACTS_X.items()]
        physics.insert(0, _depletion_electrostatics(_SHIFTER_P_OBJECTS, _SHIFTER_N_OBJECTS))
    carriers = _BOLTZMANN_CARRIERS if from_electrostatics else _PRESCRIBED_CARRIERS
    return ModelSpec(
        title=PN_ELECTROSTATICS_LABEL if from_electrostatics else PN_PRESCRIBED_LABEL,
        parameters=_junction_parameters(_SHIFTER_DOPING, _SHIFTER_DOPING, _SHIFTER_BIASES[-1], junction_x),
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   *_doped_silicon_materials(carriers, _SHIFTER_P_OBJECTS, _SHIFTER_N_OBJECTS)],
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_SHIFTER_CELL),
        study=StudySpec(type='mode_analysis', sweep=SweepSpec('V_R', list(_SHIFTER_BIASES))),
        derived_values=[DerivedValueSpec('n_eff_1', 'n_eff', operation='at', source=f'{_MODES}.mode_table', at=0),
                        DerivedValueSpec('loss_1', 'loss_db_per_cm', operation='at', source=f'{_MODES}.mode_table',
                                         at=0)])


def pn_phase_shifter_depletion_spec():
    """Template 'pn_phase_shifter_depletion': prescribed depletion-approximation carriers (no electrostatics)."""
    return _pn_phase_shifter_spec(False)


def pn_phase_shifter_electrostatics_spec():
    """Template 'pn_phase_shifter_electrostatics': carriers from the electrostatics V of the depletion charge."""
    return _pn_phase_shifter_spec(True)


# ---- carriers_depletion_junction: 1D abrupt junction, electrostatics -> Boltzmann carriers -> eps_r / sigma_e ----
# A 0.5 um x 50 nm silicon strip of 5 nm cells (100 x 10, pec em_modes: no PML), insulated y faces (1D in x); contacts
# = terminals on cells 0..9 (p, 0 V) and 90..99 (n, V_bi + V_R), junction x_j = 0.25 um. N_A = 2e24, N_D = 1e24 m^-3
# and V_R chosen (in Python) so that W = 60 nm: x_p = 20 nm, x_n = 40 nm, both depletion edges on cell edges, so the
# rasterised space charge is exactly neutral (4 cells x 2e24 = 8 cells x 1e24) and the neutral regions field-free.
# The 3-point finite-volume solve is then exact for the discrete profile computed below (face D from the cumulative
# charge, V from the half-cell held faces); it differs from the continuous parabola by ~rho h^2 / (8 eps) per charge
# step, so the check uses the discrete one.
_JUNCTION_CELL = 5e-9                   # m
_JUNCTION_CELLS = (100, 10)
_JUNCTION_LENGTH = _JUNCTION_CELLS[0] * _JUNCTION_CELL   # 0.5 um
_JUNCTION_HEIGHT = _JUNCTION_CELLS[1] * _JUNCTION_CELL   # 50 nm, the depth w of the held charges
_JUNCTION_CONTACT = 10 * _JUNCTION_CELL                  # m, held length at each end
_JUNCTION_X = 0.25e-6                   # m
_JUNCTION_DOPING = (2e24, 1e24)         # m^-3, (N_A, N_D)
_JUNCTION_DEPTHS = (20e-9, 40e-9)       # m, (x_p, x_n): cell edges
_JUNCTION_TOLERANCES = {'potential': 1e-6, 'charge_p': 1e-6, 'charge_n': 1e-6, 'eps_r': 1e-6, 'sigma_e': 1e-4}


def _junction_bias():
    """(V_bi, V_R) [V] of the junction whose depletion width is sum(_JUNCTION_DEPTHS)."""
    acceptors, donors = _JUNCTION_DOPING
    width = sum(_JUNCTION_DEPTHS)
    total = (ELEMENTARY_CHARGE * width ** 2 * acceptors * donors
             / (2.0 * _EPS_SILICON * EPS0 * (acceptors + donors)))
    built_in = _THERMAL_VOLTAGE * math.log(acceptors * donors / _INTRINSIC_DENSITY ** 2)
    return built_in, total - built_in


def _depletion_junction_specs():
    """'run': electrostatics (float64 CG to 1e-12) + pec em_modes on the junction strip, mode_analysis."""
    _, reverse_bias = _junction_bias()
    length, height, contact, junction_x = _JUNCTION_LENGTH, _JUNCTION_HEIGHT, _JUNCTION_CONTACT, _JUNCTION_X
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: depletion junction, electrostatics V -> carriers -> eps (em_modes)',
        parameters=_junction_parameters(*_JUNCTION_DOPING, reverse_bias, junction_x),
        domain=DomainSpec(size_x=length, size_y=height),
        geometry=[_rectangle('p_side', 0.0, 0.0, junction_x, height),
                  _rectangle('n_side', junction_x, 0.0, length, height),
                  _rectangle('contact_p', 0.0, 0.0, contact, height),
                  _rectangle('contact_n', length - contact, 0.0, length, height)],
        materials=_doped_silicon_materials(_BOLTZMANN_CARRIERS, ['p_side'], ['n_side']),
        physics=[_depletion_electrostatics(['p_side'], ['n_side']), _pec_mode_physics()],
        mesh=MeshSpec(mode='uniform', dx=_JUNCTION_CELL),
        study=StudySpec(type='mode_analysis')))}


def _junction_discrete_profile(centres):
    """(V [V] on the free cells, free-cell mask, D_x [C/m^2] at the p-contact face) of the discrete depletion solve on
    the uniform cells with these float64 centres."""
    acceptors, donors = _JUNCTION_DOPING
    depth_p, depth_n = _JUNCTION_DEPTHS
    built_in, reverse_bias = _junction_bias()
    free = (centres >= _JUNCTION_CONTACT) & (centres < _JUNCTION_LENGTH - _JUNCTION_CONTACT)
    p_depleted = (centres >= _JUNCTION_X - depth_p) & (centres < _JUNCTION_X)
    n_depleted = (centres >= _JUNCTION_X) & (centres < _JUNCTION_X + depth_n)
    charge = ELEMENTARY_CHARGE * (np.where(n_depleted, donors, 0.0) - np.where(p_depleted, acceptors, 0.0))[free]
    n_free = charge.size
    cumulative = np.concatenate(([0.0], np.cumsum(charge * _JUNCTION_CELL)))   # D - D_lo at faces 0..n_free
    permittivity = _EPS_SILICON * EPS0
    # V_n - V_p = -(h / eps) (n_free D_lo + (S_0 + S_n) / 2 + sum S_1..S_(n-1)) fixes D_lo
    flux_lo = (-(built_in + reverse_bias) * permittivity / _JUNCTION_CELL
               - 0.5 * (cumulative[0] + cumulative[-1]) - float(np.sum(cumulative[1:-1]))) / n_free
    flux = flux_lo + cumulative
    drop = 0.5 * flux[0] + np.concatenate(([0.0], np.cumsum(flux[1:-1])))
    return -_JUNCTION_CELL / permittivity * drop, free, flux_lo


def _check_depletion_junction(results_by_name):
    """V vs the discrete depletion profile, both contact charges vs 0 (neutral, field-free contacts), eps_r / sigma_e
    of every cell vs Soref-Bennett of the Boltzmann carriers of the measured V."""
    results = results_by_name[_RUN]
    potential_data = results.field(f'{_ELECTROSTATICS}.V')
    potential = np.asarray(potential_data.values, dtype=np.float64)
    if potential.shape != _JUNCTION_CELLS:
        raise ValueError(f'V: expected the window shape {_JUNCTION_CELLS}, got {potential.shape}')
    centres = np.asarray(potential_data.x_centers, dtype=np.float64)
    expected_free, free, flux_lo = _junction_discrete_profile(centres)
    built_in, reverse_bias = _junction_bias()
    acceptors, donors = _JUNCTION_DOPING
    depletion_charge = ELEMENTARY_CHARGE * acceptors * _JUNCTION_DEPTHS[0] * _JUNCTION_HEIGHT   # C/m per side
    charge_p = float(results.result(_ELECTROSTATICS, 'contact_p')['charge'])
    charge_n = float(results.result(_ELECTROSTATICS, 'contact_n')['charge'])
    p_side = (centres < _JUNCTION_X)[:, None]
    holes = np.where(p_side, acceptors * np.exp(np.minimum(-potential, 0.0) / _THERMAL_VOLTAGE), 0.0)
    electrons = np.where(p_side, 0.0, donors * np.exp(np.minimum(potential - built_in - reverse_bias, 0.0)
                                                      / _THERMAL_VOLTAGE))
    eps_expected, sigma_expected = plasma_dispersion(_N_SILICON, electrons, holes)
    eps_field = np.asarray(results.field(f'{_MODES}.eps_r').values, dtype=np.float64)
    sigma_field = np.asarray(results.field(f'{_MODES}.sigma_e').values, dtype=np.float64)
    total = built_in + reverse_bias
    errors = {'potential': float(np.max(np.abs(potential[free] - expected_free[:, None]))) / total,
              'charge_p': abs(charge_p) / depletion_charge,
              'charge_n': abs(charge_n) / depletion_charge,
              'eps_r': float(np.max(np.abs(eps_field - eps_expected) / eps_expected)),
              'sigma_e': float(np.max(np.abs(sigma_field - sigma_expected))) / float(np.max(sigma_expected))}
    probe = int(np.searchsorted(centres, _JUNCTION_X))   # first n-side cell
    return scaled_worst_result(errors, _JUNCTION_TOLERANCES, V_bi=built_in, V_R=reverse_bias, V_total=total,
                               Q_contact_p_C_per_m=charge_p, Q_contact_n_C_per_m=charge_n,
                               Q_depletion_C_per_m=depletion_charge, D_lo_discrete_C_per_m2=flux_lo,
                               V_junction_cell=float(potential[probe, 0]),
                               electrons_junction_cell_per_m3=float(electrons[probe, 0]),
                               sigma_e_range_S_per_m=[float(np.min(sigma_field)), float(np.max(sigma_field))],
                               n_cells=list(potential.shape))


CARRIER_SELFTEST_CASES = (
    SelfTestCase(
        name='carriers_depletion_junction',
        label='Carriers: depletion junction, electrostatics to optical permittivity',
        description=(f'A {_JUNCTION_LENGTH * 1e6:g} um x {_JUNCTION_HEIGHT * 1e9:g} nm silicon strip of uniform '
                     f'{_JUNCTION_CELL * 1e9:g} nm cells ({_JUNCTION_CELLS[0]} x {_JUNCTION_CELLS[1]}, insulated y '
                     'faces: 1D in x), electrostatics (float64 CG to 1e-12) + em_modes (pec, mode_analysis: the '
                     'electrostatics is the background solve). Abrupt junction at x_j = '
                     f'{_JUNCTION_X * 1e6:g} um, N_A = {_JUNCTION_DOPING[0]:g}, N_D = {_JUNCTION_DOPING[1]:g} m^-3; the '
                     'template parameters give V_bi = V_T ln(N_A N_D / n_i^2), W, x_n, x_p (depletion approximation) '
                     'and the charge_density features -q N_A / +q N_D on [x_j - x_p, x_j) / [x_j, x_j + x_n); '
                     'terminals contact_p (cells 0..9, 0 V) and contact_n (cells 90..99, V_bi + V_R), V_R chosen so '
                     f'that x_p = {_JUNCTION_DEPTHS[0] * 1e9:g} nm and x_n = {_JUNCTION_DEPTHS[1] * 1e9:g} nm fall on '
                     'cell edges (neutral rasterised charge). Checks: V on the free cells vs the discrete profile '
                     '(face D from the cumulative charge, V stepping by D h / eps from the half-cell held faces; '
                     'exact for the 3-point scheme, the continuous parabola differs by ~rho h^2 / (8 eps)), relative '
                     'to V_bi + V_R, tolerance 1e-6; both terminal charges 0 relative to q N_A x_p w, tolerance 1e-6 '
                     '(only consistent V_bi / W / x_n / x_p strings leave the contacts field-free); eps_r / sigma_e '
                     'of every cell (state-dependent materials of electrostatics_V, applied by the background solve) '
                     'vs Soref-Bennett of the Boltzmann carriers p = N_A exp(min(-V, 0) / V_T), n = N_D exp(min(V - '
                     'V_bi - V_R, 0) / V_T) of the measured V, tolerance 1e-6 (eps_r) / 1e-4 of the largest sigma_e '
                     '(the float32 exponent of the device evaluation). error = the worst error over its tolerance, '
                     'tolerance 1.'),
        specs=_depletion_junction_specs, check=_check_depletion_junction),
)

"""Analytic self-tests of the electrolyte of electrostatics (igor 1.6): linearised (Debye-Hueckel) slit between two
plates and charged wall anchored by the electrolyte alone (planar and axisymmetric pore); nonlinear (Poisson-Boltzmann)
Gouy-Chapman wall (1:1, Grahame and the tanh profile) and a 2:1 wall (the first-integral quadrature); an electrode
behind a Stern layer (Gouy-Chapman-Stern, linear and nonlinear); the c_ion_k concentration fields; registered by
definition."""
import math

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS0, K_B, N_A
from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import with_exact_cg, worst_error
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

ELECTROSTATICS_TYPE_NAME = 'electrostatics'   # = definition.ELECTROSTATICS_TYPE_NAME (definition imports this module)
_PHYSICS = 'electrostatics'
_RUN = 'run'

# Layout of every case: 100 x 10 uniform 1 nm cells (L = 100 nm along x, H = 10 nm along y, insulated y faces: 1D in x),
# water eps_dc 80 everywhere, one electrolyte on 'all' at 298.15 K, float64 CG to 1e-12. The finite volumes give a
# tridiagonal system per row (face conductances c H / dx, half-cell Dirichlet faces 2 c H / dx, cell terms k dx H;
# axisymmetric: x-faces x 2 pi r_face, cells x 2 pi r_c), solved here in float64 on the same cells: the profile, the
# held charges and the ionic charge are exact discrete identities up to the CG residual and float32 storage (~1e-7).
_CELL = 1e-9                                  # m
_N_CELLS = (100, 10)
_LENGTH = _N_CELLS[0] * _CELL                 # L
_HEIGHT = _N_CELLS[1] * _CELL                 # H
_EPS_DC = 80.0
_COEFFICIENT = EPS0 * _EPS_DC                 # c [F/m]
_TEMPERATURE = 298.15                         # K
_TOLERANCE = 1e-6
_ANALYTIC_BOUND = 1e-3                        # allowed deviation from the continuum forms ((dx / lambda_D)^2 ~ 1e-3)


def _screening(concentrations, valences):
    """k = sum z_i^2 e^2 1000 N_A c_i / (kB T) [F/m^3] (c_i in mol/L)."""
    weighted = sum(valence ** 2 * concentration for concentration, valence in zip(concentrations, valences))
    return weighted * 1e3 * N_A * ELEMENTARY_CHARGE ** 2 / (K_B * _TEMPERATURE)


# Slit: plates at V0 on x_lo and x_hi, 1:1 electrolyte 0.1 mM (lambda_D = 30.7 nm, L / (2 lambda_D) = 1.63): the
# continuum V = V0 cosh((x - L/2) / lambda_D) / cosh(L / (2 lambda_D)), plate charge eps0 eps_dc V0 tanh(L / (2
# lambda_D)) / lambda_D per area, total ionic charge -(sum of the plate charges) (Gauss).
_SLIT_SPECIES = ([1e-4, 1e-4], [1.0, -1.0])
_SLIT_VOLTAGE = 0.01                          # V
_SLIT_SCREENING = _screening(*_SLIT_SPECIES)
_SLIT_DEBYE_LENGTH = math.sqrt(_COEFFICIENT / _SLIT_SCREENING)
_SLIT_IONIC_STRENGTH = 1e-4                   # mol/L

# Charged wall: surface_charge sigma on x_lo (planar) or on the pore wall x_hi (axisymmetric, axis x_lo), no voltage or
# terminal (the electrolyte is the only anchor), 2:1 electrolyte (Ca2+ 50 uM, Cl- 100 uM: sum z^2 c = 3e-4 mol/L,
# I = 1.5e-4 mol/L, lambda_D = 25.1 nm). Continuum: planar V = sigma lambda_D / c cosh((L - x) / lambda_D) /
# sinh(L / lambda_D) (insulated x_hi), axisymmetric V = sigma lambda_D / c I0(r / lambda_D) / I1(R / lambda_D);
# total ionic charge -sigma H (planar, C/m) or -sigma 2 pi R H (whole ring, C).
_WALL_SPECIES = ([5e-5, 1e-4], [2.0, -1.0])
_WALL_SURFACE_CHARGE = 2e-4                   # C/m^2 (~8 mV at the wall)
_WALL_SCREENING = _screening(*_WALL_SPECIES)
_WALL_DEBYE_LENGTH = math.sqrt(_COEFFICIENT / _WALL_SCREENING)
_WALL_IONIC_STRENGTH = 1.5e-4                 # mol/L

# Nonlinear (Poisson-Boltzmann) walls: 300 x 4 cells of 0.1 nm (L = 30 nm), surface_charge on x_lo, x_hi insulated, the
# electrolyte the only anchor, Newton to 1e-7 V_T (above the CG accuracy in V; quadratic convergence leaves far less).
# Expectation: the exact discrete nonlinear rows (as the linear cases, cell terms dx H rho(V)) by float64 Newton.
# 1:1 10 mM (lambda_D = 3.04 nm, L / lambda_D ~ 10), sigma = 0.04 C/m^2: Grahame sigma = sqrt(8 c n kB T) sinh(psi0 /
# (2 V_T)) gives psi0 ~ 3.9 V_T (Debye-Hueckel would give 6.7 V_T), profile 4 V_T artanh(tanh(psi0 / (4 V_T)) exp(-x /
# lambda_D)). 2:1 (Ca2+ 5 mM, Cl- 10 mM; lambda_D = 2.48 nm), sigma = -0.01 C/m^2: the first integral (c / 2) V'^2 =
# kB T sum n_i (exp(-z_i V / V_T) - 1) gives psi0 (generalised Grahame, bisection) and the profile (RK4).
_PB_CELL = 1e-10                              # m
_PB_CELLS = (300, 4)
_PB_LENGTH = _PB_CELLS[0] * _PB_CELL
_PB_HEIGHT = _PB_CELLS[1] * _PB_CELL
_PB_SETTINGS = {'nonlinear_tolerance': 1e-7}
_PB_CONTINUUM_BOUND = 3e-3                    # continuum deviation ((dx / lambda_eff)^2 / 24, lambda_eff ~ 0.9 nm)
_THERMAL_VOLTAGE = K_B * _TEMPERATURE / ELEMENTARY_CHARGE
_GOUY_CHAPMAN_SPECIES = ([1e-2, 1e-2], [1.0, -1.0])
_GOUY_CHAPMAN_SURFACE_CHARGE = 0.04           # C/m^2
_GOUY_CHAPMAN_IONIC_STRENGTH = 1e-2           # mol/L
_ASYMMETRIC_SPECIES = ([5e-3, 1e-2], [2.0, -1.0])
_ASYMMETRIC_SURFACE_CHARGE = -0.01            # C/m^2
_ASYMMETRIC_IONIC_STRENGTH = 1.5e-2           # mol/L
_RK4_STEPS_PER_CELL = 4                       # even (a sample at (i + 1/2) dx); RK4 error ~(h / lambda_D)^4 ~ 1e-8
_BISECTIONS = 60                              # halvings of the root brackets: 2^-60 of the bracket, below float64

# Gouy-Chapman-Stern: the Poisson-Boltzmann layout, 1:1 10 mM on all cells, voltage 'electrode' on x_lo behind a Stern
# layer d_S = 0.3 nm, eps_S = 10 (C_S = eps0 eps_S / d_S = 0.295 F/m^2, near the diffuse C_d = c / lambda_D = 0.233),
# x_hi insulated. The face is a Robin row: conductance H / (1 / C_S + dx / (2 c)), drive that times V0. Continuum:
# linear (V0 = 10 mV) psi_d = V0 C_S / (C_S + C_d), C_d = c tanh(L / lambda_D) / lambda_D, V = psi_d cosh((L - x) /
# lambda_D) / cosh(L / lambda_D); nonlinear (V0 = 0.2 V ~ 7.8 V_T) V0 = psi_d + sigma(psi_d) / C_S with Grahame sigma =
# sqrt(8 c n kB T) sinh(psi_d / (2 V_T)) (psi_d ~ 3.5 V_T, bisection) and the tanh profile from the face.
_STERN_SPECIES = _GOUY_CHAPMAN_SPECIES
_STERN_THICKNESS = 3e-10                      # m
_STERN_PERMITTIVITY = 10.0
_STERN_CAPACITANCE = EPS0 * _STERN_PERMITTIVITY / _STERN_THICKNESS   # F/m^2
_STERN_VOLTAGES = {'linear': 0.01, 'nonlinear': 0.2}                # V


def _bessel_i(order, argument):
    """Modified Bessel I_order(argument) by its power series (40 terms: exact to double precision for argument < 10)."""
    half = 0.5 * argument
    return sum(half ** (2 * k + order) / (math.factorial(k) * math.factorial(k + order)) for k in range(40))


def _discrete_profile(face_conductances, cell_conductances, drive_lo, drive_hi, lo_conductance=0.0,
                      hi_conductance=0.0):
    """V at the cell centres of the 1D row system: interior face and cell conductances, outer face conductances and
    drives (Dirichlet g V0, or an inward flux with conductance 0); float64 dense solve."""
    n_cells = cell_conductances.size
    matrix = np.diag(np.asarray(cell_conductances, dtype=np.float64))
    for index, conductance in enumerate(face_conductances):
        matrix[index, index] += conductance
        matrix[index + 1, index + 1] += conductance
        matrix[index, index + 1] -= conductance
        matrix[index + 1, index] -= conductance
    matrix[0, 0] += lo_conductance
    matrix[-1, -1] += hi_conductance
    rhs = np.zeros(n_cells)
    rhs[0] += drive_lo
    rhs[-1] += drive_hi
    return np.linalg.solve(matrix, rhs)


def _centres():
    return (np.arange(_N_CELLS[0]) + 0.5) * _CELL


def _slit_solution():
    """(V at the centres, plate charge per m of depth) of the discrete slit."""
    face = _COEFFICIENT * _HEIGHT / _CELL
    dirichlet = 2.0 * face
    cell_terms = np.full(_N_CELLS[0], _SLIT_SCREENING * _CELL * _HEIGHT)
    potential = _discrete_profile(np.full(_N_CELLS[0] - 1, face), cell_terms, dirichlet * _SLIT_VOLTAGE,
                                  dirichlet * _SLIT_VOLTAGE, dirichlet, dirichlet)
    return potential, dirichlet * (_SLIT_VOLTAGE - potential[0])


def _wall_solution(axisymmetric):
    """(V at the centres, total wall charge [C/m planar | C whole ring]) of the discrete charged wall / pore."""
    centres = _centres()
    if axisymmetric:
        face_weights = 2.0 * math.pi * np.arange(1, _N_CELLS[0]) * _CELL
        cell_weights = 2.0 * math.pi * centres
        wall_charge = _WALL_SURFACE_CHARGE * 2.0 * math.pi * _LENGTH * _HEIGHT
        drives = (0.0, wall_charge)
    else:
        face_weights = np.ones(_N_CELLS[0] - 1)
        cell_weights = np.ones(_N_CELLS[0])
        wall_charge = _WALL_SURFACE_CHARGE * _HEIGHT
        drives = (wall_charge, 0.0)
    potential = _discrete_profile(face_weights * _COEFFICIENT * _HEIGHT / _CELL,
                                  cell_weights * _WALL_SCREENING * _CELL * _HEIGHT, *drives)
    return potential, wall_charge


def _wall_continuum(axisymmetric):
    """Continuum V at the cell centres of the charged wall / pore."""
    scale = _WALL_SURFACE_CHARGE * _WALL_DEBYE_LENGTH / _COEFFICIENT
    centres = _centres()
    if axisymmetric:
        return scale * np.array([_bessel_i(0, r / _WALL_DEBYE_LENGTH) for r in centres]) / _bessel_i(
            1, _LENGTH / _WALL_DEBYE_LENGTH)
    return scale * np.cosh((_LENGTH - centres) / _WALL_DEBYE_LENGTH) / math.sinh(_LENGTH / _WALL_DEBYE_LENGTH)


def _pb_centres():
    return (np.arange(_PB_CELLS[0]) + 0.5) * _PB_CELL


def _boltzmann(potential, species):
    """(rho [C/m^3], -d rho / dV [F/m^3]) of point-charge Boltzmann ions at V (V_bulk 0), numpy arrays."""
    charge = np.zeros_like(potential)
    screening = np.zeros_like(potential)
    for concentration, valence in zip(*species):
        density = 1e3 * N_A * concentration
        factor = np.exp(-valence * potential / _THERMAL_VOLTAGE)
        charge += valence * ELEMENTARY_CHARGE * density * factor
        screening += valence ** 2 * ELEMENTARY_CHARGE * density / _THERMAL_VOLTAGE * factor
    return charge, screening


def _stern_face_conductance():
    """Conductance [F/m] of the x_lo Robin row behind the Stern layer: H / (1 / C_S + dx / (2 c))."""
    return _PB_HEIGHT / (1.0 / _STERN_CAPACITANCE + 0.5 * _PB_CELL / _COEFFICIENT)


def _pb_discrete_wall(species, surface_charge, electrode=None, model='nonlinear'):
    """V at the centres of the discrete wall (sigma H into the x_lo cell; electrode = (x_lo face conductance, voltage)
    adds that Robin row; x_hi insulated): float64 Newton on the finite-volume rows, steps capped at 4 V_T, to 1e-13
    V_T; model 'linear' uses the Debye-Hueckel cell terms (one Newton step is exact)."""
    n_cells = _PB_CELLS[0]
    face = _COEFFICIENT * _PB_HEIGHT / _PB_CELL
    area = _PB_CELL * _PB_HEIGHT
    laplacian = np.zeros((n_cells, n_cells))
    lower = np.arange(n_cells - 1)
    laplacian[lower, lower] += face
    laplacian[lower + 1, lower + 1] += face
    laplacian[lower, lower + 1] -= face
    laplacian[lower + 1, lower] -= face
    drive = np.zeros(n_cells)
    drive[0] = surface_charge * _PB_HEIGHT
    if electrode is not None:
        laplacian[0, 0] += electrode[0]
        drive[0] += electrode[0] * electrode[1]
    potential = np.zeros(n_cells)
    linear_screening = np.full(n_cells, _screening(*species))
    for _ in range(200):
        if model == 'linear':
            charge, screening = -linear_screening * potential, linear_screening
        else:
            charge, screening = _boltzmann(potential, species)
        step = np.linalg.solve(laplacian + np.diag(area * screening), drive + area * charge - laplacian @ potential)
        largest = float(np.max(np.abs(step)))
        potential += step * min(1.0, 4.0 * _THERMAL_VOLTAGE / largest) if largest > 0.0 else 0.0
        if largest <= 1e-13 * _THERMAL_VOLTAGE:
            return potential
    raise ValueError('the reference Newton of the discrete Poisson-Boltzmann wall did not converge')


def _gouy_chapman_continuum():
    """(Grahame wall potential, continuum V at the centres) of the semi-infinite 1:1 wall."""
    wall = 2.0 * _THERMAL_VOLTAGE * math.asinh(_GOUY_CHAPMAN_SURFACE_CHARGE / _grahame_amplitude(_GOUY_CHAPMAN_SPECIES))
    debye = math.sqrt(_COEFFICIENT / _screening(*_GOUY_CHAPMAN_SPECIES))
    profile = 4.0 * _THERMAL_VOLTAGE * np.arctanh(math.tanh(wall / (4.0 * _THERMAL_VOLTAGE))
                                                  * np.exp(-_pb_centres() / debye))
    return wall, profile


def _grahame_amplitude(species):
    """sqrt(8 c n kB T) [C/m^2] of a symmetric 1:1 electrolyte (sigma = it x sinh(psi_d / (2 V_T)))."""
    return math.sqrt(8.0 * _COEFFICIENT * 1e3 * N_A * species[0][0] * K_B * _TEMPERATURE)


def _stern_continuum(model):
    """(diffuse-layer potential psi_d at the face, electrode charge per area sigma, continuum V at the centres) of the
    Gouy-Chapman-Stern electrode: linear finite slab (insulated x_hi) or nonlinear semi-infinite (bisection)."""
    voltage = _STERN_VOLTAGES[model]
    debye = math.sqrt(_COEFFICIENT / _screening(*_STERN_SPECIES))
    centres = _pb_centres()
    if model == 'linear':
        diffuse_capacitance = _COEFFICIENT * math.tanh(_PB_LENGTH / debye) / debye
        diffuse = voltage * _STERN_CAPACITANCE / (_STERN_CAPACITANCE + diffuse_capacitance)
        profile = diffuse * np.cosh((_PB_LENGTH - centres) / debye) / math.cosh(_PB_LENGTH / debye)
        return diffuse, diffuse_capacitance * diffuse, profile
    amplitude = _grahame_amplitude(_STERN_SPECIES)

    def charge(diffuse_potential):
        return amplitude * math.sinh(diffuse_potential / (2.0 * _THERMAL_VOLTAGE))

    low, high = 0.0, voltage   # voltage - psi_d - sigma(psi_d) / C_S falls monotonically in psi_d
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if middle + charge(middle) / _STERN_CAPACITANCE < voltage else (low, middle)
    diffuse = 0.5 * (low + high)
    profile = 4.0 * _THERMAL_VOLTAGE * np.arctanh(math.tanh(diffuse / (4.0 * _THERMAL_VOLTAGE))
                                                  * np.exp(-centres / debye))
    return diffuse, charge(diffuse), profile


def _first_integral(potential, species):
    """(dV/dx)^2 = 2 kB T / c sum n_i (exp(-z_i V / V_T) - 1) of the semi-infinite wall (clipped at 0)."""
    total = sum(1e3 * N_A * concentration * (math.exp(-valence * potential / _THERMAL_VOLTAGE) - 1.0)
                for concentration, valence in zip(*species))
    return max(0.0, 2.0 * K_B * _TEMPERATURE / _COEFFICIENT * total)


def _integrated_continuum(species, surface_charge):
    """(wall potential, continuum V at the centres) of a semi-infinite wall: c sqrt(first integral) = |sigma| by
    bisection (generalised Grahame), then RK4 of dV/dx = -sign(sigma) sqrt(first integral) from the wall."""
    sign = math.copysign(1.0, surface_charge)
    wall_slope = abs(surface_charge) / _COEFFICIENT
    low, high = 0.0, _THERMAL_VOLTAGE
    while math.sqrt(_first_integral(sign * high, species)) < wall_slope:
        high *= 2.0
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if math.sqrt(_first_integral(sign * middle, species)) < wall_slope \
            else (low, middle)
    wall = sign * 0.5 * (low + high)

    def slope(value):
        return -sign * math.sqrt(_first_integral(value, species))

    step = _PB_CELL / _RK4_STEPS_PER_CELL
    value, profile = wall, []
    for index in range(1, _RK4_STEPS_PER_CELL * _PB_CELLS[0] + 1):
        k1 = slope(value)
        k2 = slope(value + 0.5 * step * k1)
        k3 = slope(value + 0.5 * step * k2)
        k4 = slope(value + step * k3)
        value += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if index % _RK4_STEPS_PER_CELL == _RK4_STEPS_PER_CELL // 2:   # at x = (i + 1/2) dx
            profile.append(value)
    return wall, np.asarray(profile)


# ---- specs ----
def _model(title, species, features, settings=None, derived_values=(), model='linear', fine=None, backend='cpu'):
    """Stationary electrostatics with water and an electrolyte 'ions' on 'all' on `backend`: the 100 x 10 nm layout or
    the 30 x 0.4 nm Poisson-Boltzmann one (fine; default: model 'nonlinear')."""
    concentrations, valences = species
    electrolyte = FeatureSpec('ions', 'electrolyte', ['all'], {'concentrations': list(concentrations),
                                                               'valences': list(valences),
                                                               'temperature': _TEMPERATURE, 'model': model})
    nonlinear = model == 'nonlinear' if fine is None else fine
    return with_exact_cg(ModelSpec(
        title=title,
        domain=DomainSpec(size_x=_PB_LENGTH, size_y=_PB_HEIGHT) if nonlinear
        else DomainSpec(size_x=_LENGTH, size_y=_HEIGHT),
        materials=[MaterialSpec('water', properties={'eps_dc': _EPS_DC}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, ELECTROSTATICS_TYPE_NAME, settings=dict(settings or {}),
                             features=[*features, electrolyte])],
        mesh=MeshSpec(mode='uniform', dx=_PB_CELL if nonlinear else _CELL),
        solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('Q_ion', 'rho_ion', [], 'integral'),
                        DerivedValueSpec('debye_length', f'{_PHYSICS}_debye_length_ions', [], 'mean'),
                        DerivedValueSpec('ionic_strength', f'{_PHYSICS}_ionic_strength_ions', [], 'mean'),
                        *derived_values]))


def _slit_specs():
    """'run': plates plate_lo / plate_hi at V0 on x_lo / x_hi around a 0.1 mM 1:1 electrolyte."""
    return {_RUN: _model(
        'Self-test: electrolyte slit (electrostatics)', _SLIT_SPECIES,
        [FeatureSpec('plate_lo', 'voltage', ['x_lo'], {'voltage': _SLIT_VOLTAGE}),
         FeatureSpec('plate_hi', 'voltage', ['x_hi'], {'voltage': _SLIT_VOLTAGE})],
        derived_values=[DerivedValueSpec('Q_lo', f'{_PHYSICS}_charge_plate_lo', [], 'mean')])}


def _wall_specs(axisymmetric):
    """'run': surface_charge 'wall' on x_lo (planar) or x_hi (axisymmetric pore), the electrolyte the only anchor."""
    face = 'x_hi' if axisymmetric else 'x_lo'
    title = 'Self-test: charged pore wall, axisymmetric' if axisymmetric else 'Self-test: charged wall'
    return {_RUN: _model(
        f'{title} (electrostatics electrolyte)', _WALL_SPECIES,
        [FeatureSpec('wall', 'surface_charge', [face], {'surface_charge': _WALL_SURFACE_CHARGE})],
        settings={'geometry': 'axisymmetric'} if axisymmetric else None)}


def _boltzmann_wall_specs(species, surface_charge, title):
    """'run': nonlinear electrolyte, surface_charge 'wall' on x_lo, the electrolyte the only anchor."""
    return {_RUN: _model(
        f'Self-test: {title} (electrostatics nonlinear electrolyte)', species,
        [FeatureSpec('wall', 'surface_charge', ['x_lo'], {'surface_charge': surface_charge})],
        settings=_PB_SETTINGS, model='nonlinear')}


def _stern_specs(model):
    """'run': voltage 'electrode' on x_lo behind the Stern layer, the 1:1 10 mM electrolyte (model) on the fine
    layout; linear on the cpu backend, nonlinear on the default (gpu) one."""
    electrode = FeatureSpec('electrode', 'voltage', ['x_lo'], {'voltage': _STERN_VOLTAGES[model],
                                                               'stern_thickness': _STERN_THICKNESS,
                                                               'stern_permittivity': _STERN_PERMITTIVITY})
    return {_RUN: _model(
        f'Self-test: Gouy-Chapman-Stern electrode (electrostatics {model} electrolyte)', _STERN_SPECIES, [electrode],
        settings=_PB_SETTINGS if model == 'nonlinear' else None, model=model, fine=True,
        backend='cpu' if model == 'linear' else 'auto',
        derived_values=[DerivedValueSpec('Q_electrode', f'{_PHYSICS}_charge_electrode', [], 'mean')])}


# ---- checks ----
def _potential(results, n_cells=_N_CELLS):
    """float64 window V, ValueError unless the run has n_cells."""
    potential = np.asarray(results.field(f'{_PHYSICS}.V').values, dtype=np.float64)
    if potential.shape != tuple(n_cells):
        raise ValueError(f'expected the window shape {tuple(n_cells)}, got {potential.shape}')
    return potential


def _derived(results, name):
    return float(results.summary()['derived'][name])   # float() also reads 'nan'


def _scalar_errors(results, debye_length, ionic_strength):
    """Relative errors of the debye_length / ionic_strength scalar outputs."""
    return {'debye_length': abs(_derived(results, 'debye_length') / debye_length - 1.0),
            'ionic_strength': abs(_derived(results, 'ionic_strength') / ionic_strength - 1.0)}


def _concentration_errors(results, species, potential, model):
    """Errors of the c_ion_k fields against c_k exp(-z_k V / V_T) (linear: c_k (1 - z_k V / V_T)) of the returned V
    (relative to their maximum) and of sum z_k e 1000 N_A c_ion_k against the field rho_ion (relative to max |rho|)."""
    charge = np.zeros_like(potential)
    errors = {}
    for index, (concentration, valence) in enumerate(zip(*species), start=1):
        field = np.asarray(results.field(f'{_PHYSICS}.c_ion_{index}').values, dtype=np.float64)
        reduced = valence * potential / _THERMAL_VOLTAGE
        expected = concentration * (np.exp(-reduced) if model == 'nonlinear' else 1.0 - reduced)
        errors[f'c_ion_{index}'] = float(np.max(np.abs(field - expected)) / np.max(np.abs(expected)))
        charge += valence * ELEMENTARY_CHARGE * 1e3 * N_A * field
    rho = np.asarray(results.field(f'{_PHYSICS}.rho_ion').values, dtype=np.float64)
    errors['rho_from_concentrations'] = float(np.max(np.abs(charge - rho)) / np.max(np.abs(rho)))
    return errors


def _check_slit(results_by_name):
    """Profile, plate charges (table and scalar output), Gauss with the ionic charge and the scalars against the discrete
    slit; midplane and plate charge against the continuum within _ANALYTIC_BOUND."""
    results = results_by_name[_RUN]
    potential = _potential(results)
    expected, plate_charge = _slit_solution()
    charge_lo = float(results.result(_PHYSICS, 'plate_lo')['charge'])
    charge_hi = float(results.result(_PHYSICS, 'plate_hi')['charge'])
    ionic_charge = _derived(results, 'Q_ion')
    half_width = 0.5 * _LENGTH / _SLIT_DEBYE_LENGTH
    midplane = 0.5 * (potential[49] + potential[50]).mean()
    midplane_continuum = _SLIT_VOLTAGE * math.cosh(0.5 * _CELL / _SLIT_DEBYE_LENGTH) / math.cosh(half_width)
    charge_continuum = _COEFFICIENT * _SLIT_VOLTAGE * math.tanh(half_width) / _SLIT_DEBYE_LENGTH * _HEIGHT
    deviations = {'midplane': abs(midplane / midplane_continuum - 1.0),
                  'plate_charge': abs(charge_lo / charge_continuum - 1.0)}
    errors = {'profile': float(np.max(np.abs(potential - expected[:, None]))) / _SLIT_VOLTAGE,
              'plate_lo': abs(charge_lo / plate_charge - 1.0),
              'plate_hi': abs(charge_hi / plate_charge - 1.0),
              'plate_lo_scalar_output': abs(_derived(results, 'Q_lo') / plate_charge - 1.0),
              'gauss': abs(charge_lo + charge_hi + ionic_charge) / plate_charge,
              **_scalar_errors(results, _SLIT_DEBYE_LENGTH, _SLIT_IONIC_STRENGTH),
              **_concentration_errors(results, _SLIT_SPECIES, potential, 'linear'),
              **{f'{name}_excess': max(0.0, value - _ANALYTIC_BOUND) for name, value in deviations.items()}}
    return {'measured': charge_lo, 'expected': plate_charge, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'continuum_deviations': deviations, 'Q_lo': charge_lo, 'Q_hi': charge_hi,
                        'Q_ion': ionic_charge, 'Q_discrete': plate_charge, 'Q_continuum': charge_continuum,
                        'V_midplane': midplane, 'V_midplane_continuum': midplane_continuum,
                        'debye_length_m': _SLIT_DEBYE_LENGTH, 'n_cells': list(potential.shape)}}


def _check_wall(results_by_name, axisymmetric):
    """Profile, Gauss (ionic charge = -wall charge) and the scalars against the discrete wall / pore; the wall
    potential against the continuum within _ANALYTIC_BOUND."""
    results = results_by_name[_RUN]
    potential = _potential(results)
    expected, wall_charge = _wall_solution(axisymmetric)
    continuum = _wall_continuum(axisymmetric)
    wall_cell = -1 if axisymmetric else 0
    ionic_charge = _derived(results, 'Q_ion')
    scale = float(np.max(np.abs(expected)))
    deviation = abs(potential[wall_cell].mean() / continuum[wall_cell] - 1.0)
    errors = {'profile': float(np.max(np.abs(potential - expected[:, None]))) / scale,
              'gauss': abs(ionic_charge + wall_charge) / wall_charge,
              **_scalar_errors(results, _WALL_DEBYE_LENGTH, _WALL_IONIC_STRENGTH),
              'wall_potential_excess': max(0.0, deviation - _ANALYTIC_BOUND)}
    return {'measured': float(potential[wall_cell].mean()), 'expected': float(expected[wall_cell]),
            'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'wall_potential_continuum': float(continuum[wall_cell]),
                        'continuum_deviation': deviation, 'Q_ion': ionic_charge, 'Q_wall': wall_charge,
                        'charge_unit': 'C' if axisymmetric else 'C/m', 'debye_length_m': _WALL_DEBYE_LENGTH,
                        'n_cells': list(potential.shape)}}


def _check_boltzmann_wall(results_by_name, species, surface_charge, ionic_strength, symmetric):
    """Profile and Gauss against the discrete nonlinear wall, the scalars; the continuum profile and the wall potential
    (wall cell + dx/2 sigma / c) against Grahame within _PB_CONTINUUM_BOUND."""
    results = results_by_name[_RUN]
    potential = _potential(results, _PB_CELLS)
    expected = _pb_discrete_wall(species, surface_charge)
    wall, continuum = _gouy_chapman_continuum() if symmetric else _integrated_continuum(species, surface_charge)
    wall_charge = surface_charge * _PB_HEIGHT
    ionic_charge = _derived(results, 'Q_ion')
    debye = math.sqrt(_COEFFICIENT / _screening(*species))
    wall_estimate = float(potential[0].mean()) + 0.5 * _PB_CELL * surface_charge / _COEFFICIENT
    deviations = {'profile': float(np.max(np.abs(potential.mean(axis=1) - continuum))) / abs(wall),
                  'wall_potential': abs(wall_estimate / wall - 1.0)}
    errors = {'profile': float(np.max(np.abs(potential - expected[:, None]))) / abs(expected[0]),
              'gauss': abs(ionic_charge + wall_charge) / abs(wall_charge),
              **_scalar_errors(results, debye, ionic_strength),
              **_concentration_errors(results, species, potential, 'nonlinear'),
              **{f'{name}_excess': max(0.0, value - _PB_CONTINUUM_BOUND) for name, value in deviations.items()}}
    return {'measured': float(potential[0].mean()), 'expected': float(expected[0]), 'error': worst_error(errors),
            'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'continuum_deviations': deviations, 'wall_potential': wall_estimate,
                        'wall_potential_continuum': wall, 'wall_potential_thermal_voltages': wall / _THERMAL_VOLTAGE,
                        'wall_potential_debye_hueckel': surface_charge * debye / _COEFFICIENT, 'Q_ion': ionic_charge,
                        'Q_wall': wall_charge, 'debye_length_m': debye, 'n_cells': list(potential.shape)}}


def _check_stern(results_by_name, model):
    """Profile, electrode charge (table and scalar output), Gauss, the scalars and the concentration fields against the
    discrete Robin-row electrode; psi_d (first cell + dx sigma / (2 c)), sigma and the profile against the
    Gouy-Chapman-Stern continuum within _PB_CONTINUUM_BOUND."""
    results = results_by_name[_RUN]
    potential = _potential(results, _PB_CELLS)
    voltage = _STERN_VOLTAGES[model]
    conductance = _stern_face_conductance()
    expected = _pb_discrete_wall(_STERN_SPECIES, 0.0, (conductance, voltage), model)
    electrode_charge = conductance * (voltage - expected[0])   # C/m
    charge = float(results.result(_PHYSICS, 'electrode')['charge'])
    ionic_charge = _derived(results, 'Q_ion')
    diffuse, surface_charge, continuum = _stern_continuum(model)
    measured_surface_charge = charge / _PB_HEIGHT
    diffuse_estimate = float(potential[0].mean()) + 0.5 * _PB_CELL * measured_surface_charge / _COEFFICIENT
    debye = math.sqrt(_COEFFICIENT / _screening(*_STERN_SPECIES))
    deviations = {'profile': float(np.max(np.abs(potential.mean(axis=1) - continuum))) / diffuse,
                  'diffuse_potential': abs(diffuse_estimate / diffuse - 1.0),
                  'surface_charge': abs(measured_surface_charge / surface_charge - 1.0)}
    errors = {'profile': float(np.max(np.abs(potential - expected[:, None]))) / abs(expected[0]),
              'electrode_charge': abs(charge / electrode_charge - 1.0),
              'electrode_scalar_output': abs(_derived(results, 'Q_electrode') / electrode_charge - 1.0),
              'gauss': abs(charge + ionic_charge) / electrode_charge,
              **_scalar_errors(results, debye, _GOUY_CHAPMAN_IONIC_STRENGTH),
              **_concentration_errors(results, _STERN_SPECIES, potential, model),
              **{f'{name}_excess': max(0.0, value - _PB_CONTINUUM_BOUND) for name, value in deviations.items()}}
    return {'measured': charge, 'expected': electrode_charge, 'error': worst_error(errors), 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'continuum_deviations': deviations, 'Q_electrode': charge,
                        'Q_discrete': electrode_charge, 'Q_ion': ionic_charge,
                        'surface_charge': measured_surface_charge, 'surface_charge_continuum': surface_charge,
                        'diffuse_potential': diffuse_estimate, 'diffuse_potential_continuum': diffuse,
                        'diffuse_potential_thermal_voltages': diffuse / _THERMAL_VOLTAGE,
                        'stern_drop': voltage - diffuse_estimate, 'stern_capacitance_F_m2': _STERN_CAPACITANCE,
                        'debye_length_m': debye, 'n_cells': list(potential.shape)}}


_LAYOUT_TEXT = (f'{_N_CELLS[0]} x {_N_CELLS[1]} uniform {_CELL * 1e9:g} nm cells (L = {_LENGTH * 1e9:g} nm, insulated y '
                f'faces), water eps_dc {_EPS_DC:g}, an electrolyte on all cells at {_TEMPERATURE:g} K, stationary, '
                'float64 CG to 1e-12 on the cpu backend (numpy, the exact pcg node)')
_NOT_YET_RUN_CPU = ' Not yet run on the cpu backend.'
_EXACT_TEXT = ('The expectation is the exact discrete solution of the finite-volume rows (face conductances eps0 '
               'eps_dc H / dx, half-cell Dirichlet faces, cell terms k dx H with k = sum z_i^2 e^2 1000 N_A c_i / (kB '
               'T){ring}) solved in float64 on the same cells')
_PB_LAYOUT_TEXT = (f'{_PB_CELLS[0]} x {_PB_CELLS[1]} uniform {_PB_CELL * 1e9:g} nm cells (L = {_PB_LENGTH * 1e9:g} nm, '
                   f'insulated y faces and x_hi), water eps_dc {_EPS_DC:g}, a nonlinear electrolyte on all cells at '
                   f'{_TEMPERATURE:g} K and surface_charge on x_lo (the only anchor), stationary, float64 CG to 1e-12, '
                   f"Newton to {_PB_SETTINGS['nonlinear_tolerance']:g} V_T, cpu backend (numpy, the exact pcg node)")
_STERN_LAYOUT_TEXT = (f'{_PB_CELLS[0]} x {_PB_CELLS[1]} uniform {_PB_CELL * 1e9:g} nm cells (L = '
                      f'{_PB_LENGTH * 1e9:g} nm, insulated y faces and x_hi), water eps_dc {_EPS_DC:g}, a {{model}} '
                      f'electrolyte on all cells at {_TEMPERATURE:g} K, stationary, float64 CG to 1e-12 (nonlinear: '
                      f"Newton to {_PB_SETTINGS['nonlinear_tolerance']:g} V_T), {{backend}}")
_STERN_BACKEND_TEXT = {  # model -> (backend sentence of the layout, closing sentence)
    'linear': ('cpu backend (numpy, the exact pcg node)', ' The cpu Robin face is also run by '
               'electrostatics_electrolyte_stern_nonlinear[cpu].' + _NOT_YET_RUN_CPU),
    'nonlinear': ('default (gpu) backend', ' Variant cpu: the same run on the cpu backend (numpy, the exact pcg '
                  'node), same tolerance: the gpu / cpu pin of the damped Newton with its CG warm start.')}
_PB_EXACT_TEXT = ('The expectation is the exact discrete solution of the nonlinear finite-volume rows (cell terms dx H '
                  'sum z_i e n_i exp(-z_i V / V_T)) by float64 Newton on the same cells. error = max(max |V - V_j| / '
                  '|V_j(wall cell)|, Gauss |integral rho_ion + sigma H| / |sigma H|, the debye_length / ionic_strength '
                  'scalar outputs, the c_ion_k fields against c_k exp(-z_k V / V_T) of the returned V and their charge '
                  f'sum against rho_ion, the excess over {_PB_CONTINUUM_BOUND:g} of {{continuum}}), tolerance '
                  f'{_TOLERANCE:g}.')

ELECTROLYTE_SELFTEST_CASES = (
    SelfTestCase(
        name='electrostatics_electrolyte_slit', label='Electrostatics: electrolyte slit between two plates',
        description=(f'Electrostatics on {_LAYOUT_TEXT}: 0.1 mM 1:1 (lambda_D = {_SLIT_DEBYE_LENGTH * 1e9:.4g} nm), '
                     f'voltages plate_lo / plate_hi = {_SLIT_VOLTAGE:g} V on x_lo / x_hi. '
                     + _EXACT_TEXT.format(ring='')
                     + '. error = max(max |V - V_j| / V0, both plate charges and the plate_lo scalar output relative to '
                       'the discrete one, Gauss |Q_lo + Q_hi + integral rho_ion| / Q, the debye_length / '
                       'ionic_strength scalar outputs, the c_ion_1 / c_ion_2 fields against c (1 -+ V / V_T) of the '
                       'returned V and their charge sum against rho_ion, the excess over '
                       f'{_ANALYTIC_BOUND:g} of the midplane V against V0 cosh(dx / (2 lambda_D)) / cosh(L / (2 '
                       'lambda_D)) and of the plate charge against eps0 eps_dc V0 tanh(L / (2 lambda_D)) H / '
                       f'lambda_D), tolerance {_TOLERANCE:g}. Pins the diagonal term, the field rho_ion and the '
                       'electrolyte scalar outputs.' + _NOT_YET_RUN_CPU),
        specs=_slit_specs, check=_check_slit),
    SelfTestCase(
        name='electrostatics_electrolyte_charged_wall', label='Electrostatics: charged wall in an electrolyte',
        description=(f'Electrostatics on {_LAYOUT_TEXT}: CaCl2 50 uM (valences 2, -1; lambda_D = '
                     f'{_WALL_DEBYE_LENGTH * 1e9:.4g} nm), surface_charge {_WALL_SURFACE_CHARGE:g} C/m^2 on x_lo, x_hi '
                     'insulated, no voltage or terminal (the electrolyte is the only anchor). '
                     + _EXACT_TEXT.format(ring='')
                     + '. error = max(max |V - V_j| / max V_j, Gauss |integral rho_ion + sigma H| / (sigma H), the '
                       f'scalar outputs, the excess over {_ANALYTIC_BOUND:g} of the wall-cell V against sigma lambda_D '
                       f'/ c cosh((L - x) / lambda_D) / sinh(L / lambda_D)), tolerance {_TOLERANCE:g}. Pins the '
                       'electrolyte anchor (validation and runtime) and the valence squares.' + _NOT_YET_RUN_CPU),
        specs=lambda: _wall_specs(False), check=lambda results_by_name: _check_wall(results_by_name, False)),
    SelfTestCase(
        name='electrostatics_electrolyte_pore_axisymmetric',
        label='Electrostatics: charged cylindrical pore in an electrolyte (axisymmetric)',
        description=(f'Axisymmetric electrostatics (x = r, axis on x_lo) on {_LAYOUT_TEXT}: the charged-wall '
                     f'electrolyte and surface_charge {_WALL_SURFACE_CHARGE:g} C/m^2 on the pore wall x_hi (R = '
                     f'{_LENGTH * 1e9:g} nm). '
                     + _EXACT_TEXT.format(ring='; x-faces x 2 pi r_face, cells x 2 pi r_c')
                     + '. error = max(max |V - V_j| / max V_j, Gauss |ring integral of rho_ion + sigma 2 pi R H| / '
                       f'(sigma 2 pi R H), the scalar outputs, the excess over {_ANALYTIC_BOUND:g} of the wall-cell V '
                       'against sigma lambda_D / c I0(r / lambda_D) / I1(R / lambda_D)), tolerance '
                       f'{_TOLERANCE:g}. Pins the ring-volume weighting of the diagonal term.' + _NOT_YET_RUN_CPU),
        specs=lambda: _wall_specs(True), check=lambda results_by_name: _check_wall(results_by_name, True)),
    SelfTestCase(
        name='electrostatics_electrolyte_gouy_chapman',
        label='Electrostatics: Gouy-Chapman wall (nonlinear Poisson-Boltzmann, 1:1)',
        description=(f'Electrostatics on {_PB_LAYOUT_TEXT}: 1:1 10 mM, surface_charge '
                     f'{_GOUY_CHAPMAN_SURFACE_CHARGE:g} C/m^2 (wall potential ~3.9 V_T; Debye-Hueckel would give '
                     '6.7 V_T). ' + _PB_EXACT_TEXT.format(
                         continuum='the profile against 4 V_T artanh(tanh(psi0 / (4 V_T)) exp(-x / lambda_D)) and of '
                                   'the wall potential (wall cell + dx sigma / (2 c)) against Grahame psi0 = 2 V_T '
                                   'asinh(sigma / sqrt(8 c n kB T))')
                     + ' Pins the Boltzmann charge, its Newton tangent and the damped loop.' + _NOT_YET_RUN_CPU),
        specs=lambda: _boltzmann_wall_specs(_GOUY_CHAPMAN_SPECIES, _GOUY_CHAPMAN_SURFACE_CHARGE, 'Gouy-Chapman wall'),
        check=lambda results_by_name: _check_boltzmann_wall(results_by_name, _GOUY_CHAPMAN_SPECIES,
                                                            _GOUY_CHAPMAN_SURFACE_CHARGE,
                                                            _GOUY_CHAPMAN_IONIC_STRENGTH, True)),
    SelfTestCase(
        name='electrostatics_electrolyte_boltzmann_2_1',
        label='Electrostatics: charged wall in a 2:1 electrolyte (nonlinear Poisson-Boltzmann)',
        description=(f'Electrostatics on {_PB_LAYOUT_TEXT}: CaCl2 (Ca2+ 5 mM, Cl- 10 mM), surface_charge '
                     f'{_ASYMMETRIC_SURFACE_CHARGE:g} C/m^2 (counter-ion Ca2+). ' + _PB_EXACT_TEXT.format(
                         continuum=f'the profile against the RK4 quadrature ({_RK4_STEPS_PER_CELL} steps per cell) of '
                                   'the first integral (c / 2) (dV/dx)^2 = kB T sum n_i (exp(-z_i V / V_T) - 1) and of '
                                   f'the wall potential against its generalised Grahame root ({_BISECTIONS} bisections)')
                     + ' Pins the per-species valences of the Boltzmann sum and the asymmetric screening. Not yet run '
                       f'on the cpu backend with {_RK4_STEPS_PER_CELL} RK4 steps per cell and {_BISECTIONS} '
                       'bisections.'),
        specs=lambda: _boltzmann_wall_specs(_ASYMMETRIC_SPECIES, _ASYMMETRIC_SURFACE_CHARGE, '2:1 charged wall'),
        check=lambda results_by_name: _check_boltzmann_wall(results_by_name, _ASYMMETRIC_SPECIES,
                                                            _ASYMMETRIC_SURFACE_CHARGE, _ASYMMETRIC_IONIC_STRENGTH,
                                                            False)),
    *(SelfTestCase(
        name=f'electrostatics_electrolyte_stern_{model}',
        label=f'Electrostatics: electrode behind a Stern layer (Gouy-Chapman-Stern, {model} electrolyte)',
        description=('Electrostatics on '
                     + _STERN_LAYOUT_TEXT.format(model=model, backend=_STERN_BACKEND_TEXT[model][0])
                     + f': 1:1 10 mM, voltage {_STERN_VOLTAGES[model]:g} V on x_lo behind a Stern layer stern_thickness '
                     f'{_STERN_THICKNESS * 1e9:g} nm, stern_permittivity {_STERN_PERMITTIVITY:g} (C_S '
                     f'= {_STERN_CAPACITANCE:.3g} F/m^2). The expectation is the exact discrete solution of the '
                     'finite-volume rows with the x_lo Robin row H / (1 / C_S + dx / (2 c)) (Newton in float64 on the '
                     'same cells). error = max(max |V - V_j| / V_j(first cell), the electrode charge (table and '
                     'charge_electrode scalar output) against conductance x (V0 - V_j(first cell)), Gauss |Q + integral '
                     'rho_ion| / Q, the debye_length / ionic_strength scalar outputs, the c_ion_1 / c_ion_2 fields '
                     'against c exp(-+ V / V_T) ' + ('(linearised: c (1 -+ V / V_T)) ' if model == 'linear' else '')
                     + 'of the returned V and their charge sum against rho_ion, the excess over '
                     f'{_PB_CONTINUUM_BOUND:g} of the profile, psi_d (first cell + dx sigma / (2 c)) and sigma against '
                     + ('psi_d = V0 C_S / (C_S + C_d), C_d = c tanh(L / lambda_D) / lambda_D, V = psi_d cosh((L - x) / '
                        'lambda_D) / cosh(L / lambda_D)' if model == 'linear' else
                        'the Gouy-Chapman-Stern root V0 = psi_d + sqrt(8 c n kB T) sinh(psi_d / (2 V_T)) / C_S and '
                        'the tanh profile from the face')
                     + f'), tolerance {_TOLERANCE:g}. Pins the stern_thickness / stern_permittivity Robin face, the '
                       'held flux through it and the concentration fields.' + _STERN_BACKEND_TEXT[model][1]),
        specs=lambda model=model: _stern_specs(model),
        check=lambda results_by_name, model=model: _check_stern(results_by_name, model),
        variants=(SelfTestVariant('cpu', tolerance=_TOLERANCE, backend='cpu'),) if model == 'nonlinear' else ())
      for model in ('linear', 'nonlinear')),
)

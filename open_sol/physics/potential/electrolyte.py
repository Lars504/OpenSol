"""Electrolytes of electrostatics (Poisson-Boltzmann, linearised or nonlinear): screening coefficient, ionic strength,
Debye length, Stern capacitance (host math) and the Boltzmann charge / screening / species concentrations of an ion
species list (array-module agnostic, no cupy import; shared by the solver, renderer and definition)."""
import math

from ...core.constants import ELEMENTARY_CHARGE, EPS0, K_B, N_A

LITRES_PER_CUBIC_METRE = 1e3
ELECTROLYTE_MODELS = ('linear', 'nonlinear')
# |z (V - V_bulk)| / V_T beyond which the Boltzmann factor is clipped (e^80 ~ 5.5e34: finite in float32 diagonals).
MAX_BOLTZMANN_EXPONENT = 80.0
# Output fields c_ion_1 .. c_ion_<MAX_SPECIES_FIELDS>: species k (the k-th list entry) of every electrolyte [mol/L].
MAX_SPECIES_FIELDS = 4
ION_CONCENTRATION_FIELDS = tuple(f'c_ion_{index}' for index in range(1, MAX_SPECIES_FIELDS + 1))


def stern_capacitance(thickness, relative_permittivity) -> float:
    """C_S = eps0 eps_S / d_S [F/m^2] of a compact (Stern) layer; 0 for thickness 0 (no layer)."""
    return EPS0 * float(relative_permittivity) / float(thickness) if float(thickness) > 0.0 else 0.0


def screening_coefficient(concentrations, valences, temperature) -> float:
    """k = sum z_i^2 e^2 n_i / (kB T) [F/m^3] (rho_ion = -k (V - V_bulk)), n_i = 1000 N_A c_i from c_i [mol/L]."""
    weighted_sum = sum(valence * valence * concentration for concentration, valence in zip(concentrations, valences))
    return weighted_sum * LITRES_PER_CUBIC_METRE * N_A * ELEMENTARY_CHARGE ** 2 / (K_B * temperature)


def ionic_strength(concentrations, valences) -> float:
    """I = 1/2 sum z_i^2 c_i [mol/L]."""
    return 0.5 * sum(valence * valence * concentration for concentration, valence in zip(concentrations, valences))


def debye_length(relative_permittivity, coefficient) -> float:
    """lambda_D = sqrt(eps0 eps_dc / k) [m] of a screening coefficient k [F/m^3]; inf for k = 0."""
    return math.sqrt(EPS0 * relative_permittivity / coefficient) if coefficient > 0.0 else math.inf


def number_densities(concentrations) -> tuple:
    """n_i = 1000 N_A c_i [1/m^3] of molar concentrations c_i [mol/L]."""
    return tuple(LITRES_PER_CUBIC_METRE * N_A * float(concentration) for concentration in concentrations)


def thermal_voltage(temperature) -> float:
    """V_T = kB T / e [V] (25.69 mV at 298.15 K)."""
    return K_B * float(temperature) / ELEMENTARY_CHARGE


def boltzmann_terms(potential, bulk_potential, densities, valences, temperature, xp):
    """(rho_ion [C/m^3], screening -d rho_ion / dV [F/m^3]) = (sum z_i e n_i b_i, sum z_i^2 e^2 n_i b_i / (kB T)),
    b_i = exp(-z_i (V - V_bulk) / V_T) with the exponent clipped at +-MAX_BOLTZMANN_EXPONENT; xp numpy or cupy."""
    volts = thermal_voltage(temperature)
    reduced = (potential - bulk_potential) / volts
    charge = xp.zeros_like(reduced)
    screening = xp.zeros_like(reduced)
    for density, valence in zip(densities, valences):
        boltzmann = xp.exp(xp.clip(-valence * reduced, -MAX_BOLTZMANN_EXPONENT, MAX_BOLTZMANN_EXPONENT))
        charge += (valence * ELEMENTARY_CHARGE * density) * boltzmann
        screening += (valence * valence * ELEMENTARY_CHARGE * density / volts) * boltzmann
    return charge, screening


def species_concentration(potential, bulk_potential, concentration, valence, temperature, model, xp):
    """Local concentration [mol/L] of one species of bulk concentration c: nonlinear c exp(-z (V - V_bulk) / V_T)
    (exponent clipped as in boltzmann_terms), linear c (1 - z (V - V_bulk) / V_T) (its rho_ion; < 0 beyond V_T / |z|)."""
    reduced = (potential - bulk_potential) / thermal_voltage(temperature)
    if model == 'nonlinear':
        return concentration * xp.exp(xp.clip(-valence * reduced, -MAX_BOLTZMANN_EXPONENT, MAX_BOLTZMANN_EXPONENT))
    return concentration * (1.0 - valence * reduced)

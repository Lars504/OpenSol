"""conducting_sheet self-tests (nonlinear_optics_plan self-test 10): the graphene Kubo conductivity against its T -> 0
closed forms (CPU) and the normal-incidence transmission / absorption of a sheet in em_fdfd_ez / em_fdfd_hz against
the discrete delta-potential closed form (cpu model runs)."""
import math

import numpy as np

from ...core.constants import C0, ELEMENTARY_CHARGE, HBAR, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_fdfd import cpu_specs

_PHYSICS_NAME = 'em'
_FLUX = 'transmitted'
_SIGMA_0 = ELEMENTARY_CHARGE ** 2 / (4.0 * HBAR)   # S

# em_graphene_kubo_limits: at 1 K (kT = 8.6e-5 eV) the smeared Kubo terms equal their T -> 0 forms to ~(kT / (a - 2
# |mu|))^2 ~ 1e-8: intraband e^2 |E_F| tau / (pi hbar^2 (1 + j omega tau)), interband sigma_0 [step(a - 2 |mu|) - (j / pi)
# ln|(a - 2 |mu|) / (a + 2 |mu|)|], a = hbar omega; one photon energy above, one below the 2 |E_F| edge.
_KUBO_FERMI_ENERGY = 0.2               # eV
_KUBO_TAU = 1e-13                      # s
_KUBO_TEMPERATURE = 1.0                # K
_KUBO_PHOTON_ENERGIES = (1.0, 0.2)     # eV
_KUBO_TOLERANCE = 1e-6                 # |sigma - closed form| / sigma_0

# em_fdfd_conducting_sheet: y-periodic vacuum strip 150 x 10 cells of wavelength / 50 (CPML along x), a vertical line
# source at cell 30, the sheet at cell 75 (device runs), a flux line at cell 120 (positions at cell centres). ez: a
# constant sigma_s at 1 um; hz: graphene (Kubo, E_F 0.4 eV, tau 0.1 ps, 300 K) at 50 um, where it is inductive
# (Drude, omega tau = 3.8). Vacuum FDFD is scale-invariant, so the two wavelengths see the same discretisation.
_CELLS_PER_WAVELENGTH = 50
_LENGTH_CELLS, _PERIOD_CELLS = 150, 10
_LINE_CELLS = {'source': 30, 'sheet': 75, 'flux': 120}
_SHEET_CASES = {
    'ez': (1e-6, {'model': 'constant', 'sigma_real': 2e-3, 'sigma_imag': -1e-3}),
    'hz': (50e-6, {'model': 'graphene_kubo', 'fermi_energy': 0.4, 'scattering_time': 1e-13, 'temperature': 300.0}),
}
_SHEET_TOLERANCE = 1e-3                # absolute T and A: CPML reflection floor (~1e-4) between sheet and PML


def _check_kubo_limits(results_by_name):
    """Kubo intraband / interband at 1 K vs their T -> 0 closed forms; error = max |difference| / sigma_0."""
    from .sheet_conductivity import graphene_interband, graphene_intraband
    errors, details = [], {}
    for photon_energy in _KUBO_PHOTON_ENERGIES:
        omega = photon_energy * ELEMENTARY_CHARGE / HBAR
        intraband = graphene_intraband(omega, _KUBO_FERMI_ENERGY, _KUBO_TAU, _KUBO_TEMPERATURE)
        drude = (ELEMENTARY_CHARGE ** 2 * _KUBO_FERMI_ENERGY * ELEMENTARY_CHARGE * _KUBO_TAU
                 / (math.pi * HBAR ** 2 * (1.0 + 1j * omega * _KUBO_TAU)))
        interband = graphene_interband(omega, _KUBO_FERMI_ENERGY, _KUBO_TEMPERATURE)
        edge = 2.0 * _KUBO_FERMI_ENERGY
        closed = _SIGMA_0 * ((1.0 if photon_energy > edge else 0.0)
                             - (1j / math.pi) * math.log(abs((photon_energy - edge) / (photon_energy + edge))))
        errors += [abs(intraband - drude) / _SIGMA_0, abs(interband - closed) / _SIGMA_0]
        details[f'{photon_energy:g}_eV'] = {'intraband': [intraband.real, intraband.imag],
                                            'intraband_closed_form': [drude.real, drude.imag],
                                            'interband': [interband.real, interband.imag],
                                            'interband_closed_form': [closed.real, closed.imag]}
    return {'measured': max(errors), 'expected': 0.0, 'error': max(errors), 'tolerance': _KUBO_TOLERANCE,
            'details': details}


def _sheet_spec(polarization, run_name):
    """'reference' (vacuum) or 'device' (with the conducting sheet) strip of polarization, stationary at its wavelength."""
    wavelength, sheet_values = _SHEET_CASES[polarization]
    cell = wavelength / _CELLS_PER_WAVELENGTH
    size_x, size_y = _LENGTH_CELLS * cell, _PERIOD_CELLS * cell

    def vertical_line(name):
        x = (_LINE_CELLS[name] + 0.5) * cell
        return GeometrySpec(f'{name}_line', 'line', {'x0': x, 'y0': 0.0, 'x1': x, 'y1': size_y})
    geometry = [vertical_line('source'), vertical_line('flux')]
    features = [FeatureSpec('source', 'line_source', ['source_line'], {'amplitude': 1.0, 'phase': 0.0}),
                FeatureSpec(_FLUX, 'flux_monitor', ['flux_line'])]
    if run_name == 'device':
        geometry.append(vertical_line('sheet'))
        features.append(FeatureSpec('sheet', 'conducting_sheet', ['sheet_line'], dict(sheet_values)))
    return ModelSpec(
        title=f'Self-test: conducting sheet transmission ({polarization}, {run_name})',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS_NAME, f'em_fdfd_{polarization}', features=features,
                             settings={'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic', 'wavelength': wavelength})],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='stationary'))


def _sheet_specs():
    """'<polarization>_reference' and '<polarization>_device' of every _SHEET_CASES entry (cpu; the gpu sheet
    assembly is compared in em_fdfd_cpu_gpu)."""
    return cpu_specs({f'{polarization}_{run_name}': _sheet_spec(polarization, run_name)
                      for polarization in _SHEET_CASES for run_name in ('reference', 'device')})


def sheet_transmission(sigma, wavelength, cell):
    """Discrete normal-incidence amplitude transmission of a sheet on the 3-point stencil (both polarisations: the
    E_z cell / the H_z tangential-E face): 1 / (1 + omega mu0 sigma w / (2 sin(k w))), (2 / w) sin(k w / 2) = k0."""
    k0 = 2.0 * math.pi / wavelength
    width = float(np.float32(cell))   # the float32 grid widths of the operator
    half_phase = math.asin(0.5 * k0 * width)
    return 1.0 / (1.0 + 2.0 * math.pi * C0 / wavelength * MU0 * sigma * width / (2.0 * math.sin(2.0 * half_phase)))


def _check_sheet(results_by_name):
    """T = device / reference flux and A = integral of absorbed_power / reference flux vs |t|^2 and 1 - |t|^2 - |t -
    1|^2 of sheet_transmission, per polarisation; error = the worst absolute deviation."""
    from .sheet_conductivity import SheetConductivity
    errors, details = [], {}
    for polarization, (wavelength, sheet_values) in _SHEET_CASES.items():
        if sheet_values['model'] == 'constant':
            conductivity = SheetConductivity('constant', value=complex(sheet_values['sigma_real'],
                                                                       sheet_values['sigma_imag']))
        else:
            conductivity = SheetConductivity('graphene_kubo', fermi_energy=sheet_values['fermi_energy'],
                                             scattering_time=sheet_values['scattering_time'],
                                             temperature=sheet_values['temperature'])
        sigma = conductivity(2.0 * math.pi * C0 / wavelength)
        cell = wavelength / _CELLS_PER_WAVELENGTH
        amplitude = sheet_transmission(sigma, wavelength, cell)
        expected_transmission = abs(amplitude) ** 2
        expected_absorption = 1.0 - expected_transmission - abs(amplitude - 1.0) ** 2
        reference_power = float(np.asarray(results_by_name[f'{polarization}_reference'].result(
            _PHYSICS_NAME, _FLUX)['power'], dtype=np.float64)[0])
        device = results_by_name[f'{polarization}_device']
        device_power = float(np.asarray(device.result(_PHYSICS_NAME, _FLUX)['power'], dtype=np.float64)[0])
        absorbed_density = np.asarray(device.field(f'{_PHYSICS_NAME}.absorbed_power').values, dtype=np.float64)
        width = float(np.float32(cell))
        transmission = device_power / reference_power
        absorption = float(np.sum(absorbed_density)) * width * width / reference_power
        errors += [abs(transmission - expected_transmission), abs(absorption - expected_absorption)]
        details[polarization] = {'sigma': [sigma.real, sigma.imag], 'transmission': transmission,
                                 'expected_transmission': expected_transmission, 'absorption': absorption,
                                 'expected_absorption': expected_absorption, 'reference_power': reference_power}
    worst = math.inf if any(math.isnan(error) for error in errors) else max(errors)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _SHEET_TOLERANCE,
            'details': {**details, 'errors': float_list(errors)}}


CONDUCTING_SHEET_SELFTEST_CASES = (
    SelfTestCase(
        name='em_graphene_kubo_limits',
        label='EM graphene Kubo conductivity limits',
        description='CPU: graphene sheet conductivity (sheet_conductivity.graphene_intraband / graphene_interband, '
                    'exp(+j omega t)) at E_F = 0.2 eV, tau = 0.1 ps, T = 1 K and photon energies 1.0 eV (above the '
                    '2 |E_F| = 0.4 eV interband edge) and 0.2 eV (below) against the T -> 0 closed forms: intraband '
                    'e^2 |E_F| tau / (pi hbar^2 (1 + j omega tau)), interband sigma_0 [step(hbar omega - 2 |E_F|) - '
                    '(j / pi) ln|(hbar omega - 2 |E_F|) / (hbar omega + 2 |E_F|)|], sigma_0 = e^2 / (4 hbar) (the '
                    'universal 6.085e-5 S above the edge, capacitive Im > 0 below it). Pins units, the conjugated '
                    'sign convention and the principal-value integral. error = max |sigma - closed form| / sigma_0, '
                    'tolerance 1e-6 (thermal corrections ~(kT / |hbar omega - 2 E_F|)^2 ~ 1e-8).',
        specs=lambda: {},
        check=_check_kubo_limits,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_fdfd_conducting_sheet',
        label='EM FDFD conducting sheet transmission and absorption',
        description='A vertical line source in a y-periodic vacuum strip (CPML along x, wavelength / 50 cells) '
                    'illuminates a conducting_sheet across the strip at normal incidence: em_fdfd_ez with a '
                    'constant sigma_s = 2e-3 - 1e-3j S at 1 um, em_fdfd_hz with graphene (Kubo: E_F 0.4 eV, tau 0.1 '
                    'ps, 300 K, inductive) at 50 um. T = flux behind the sheet / flux of a vacuum reference run and '
                    'A = integral of absorbed_power / reference flux must equal |t|^2 and 1 - |t|^2 - |t - 1|^2 of '
                    'the discrete delta potential t = 1 / (1 + omega mu0 sigma_s w / (2 sin(k w))), (2 / w) sin(k w / '
                    '2) = k0 (continuum 1 / (1 + eta0 sigma_s / 2)). Pins the ez operator diagonal, the hz face '
                    'weight, the sheet loss in q_abs and the graphene model path. error = the worst absolute '
                    'deviation of T and A; tolerance 1e-3 (CPML reflections between sheet and PML, ~1e-4). cpu '
                    'backend (not yet confirmed by a run there).',
        specs=_sheet_specs,
        check=_check_sheet),
)

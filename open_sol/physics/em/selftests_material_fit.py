"""CPU self-test of the A7 nk_fit (physics/em/material_fit, materials.effective_material_data): analytic round trip
and the library tables (no model run)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase

# synthetic metal: eps_inf, Drude (omega_p, gamma), Lorentz (delta_eps, omega_0, gamma) [rad/s]; one Lorentz in the
# band (343 nm), one above it (209 nm)
_ROUND_TRIP_EPS_INF = 2.0
_ROUND_TRIP_DRUDE = (1.3e16, 6.0e13)
_ROUND_TRIP_LORENTZ = ((1.5, 5.5e15, 6.0e14), (0.8, 9.0e15, 2.0e15))
_ROUND_TRIP_BAND = (300e-9, 900e-9, 61)   # start, stop [m], samples
_ROUND_TRIP_TOLERANCE = 1e-3
# library material, n_lorentz, fit range [m]
_LIBRARY_FITS = (('silver', 2, 400e-9, 900e-9), ('gold', 3, 500e-9, 1000e-9), ('aluminum', 3, 400e-9, 1000e-9),
                 ('silicon', 2, 400e-9, 1000e-9))
_LIBRARY_ERROR_LIMIT = 0.1   # max |eps error| / |eps| a library fit must reach (validation warns above 0.02)
_CONSISTENCY_TOLERANCE = 1e-5   # expanded poles (DrudePole / LorentzPole chi) vs the PoleFit's own error


def _check_nk_fit(results):
    from ...materials import REGISTRY, effective_material_data, fitted_library_poles
    from .definition_common import make_pole
    from .dispersion import complex_permittivity
    from .material_fit import drude_lorentz_permittivity, fit_drude_lorentz, table_samples
    violations = []

    # 1. round trip: model eps -> n, k -> eps -> fit with the same pole count -> eps
    start, stop, count = _ROUND_TRIP_BAND
    wavelengths = np.linspace(start, stop, count)
    omega = 2.0 * math.pi * C0 / wavelengths
    exact = drude_lorentz_permittivity(omega, _ROUND_TRIP_EPS_INF, _ROUND_TRIP_DRUDE, _ROUND_TRIP_LORENTZ)
    root = np.sqrt(exact)   # principal root: Re >= 0, Im <= 0 for lossy eps, i.e. n - j k
    n, k = root.real, -root.imag
    round_trip_eps = (n - 1j * k) ** 2
    conversion_error = float(np.max(np.abs(round_trip_eps - exact) / np.abs(exact)))
    if conversion_error > 1e-12:
        violations.append(f'n, k conversion error {conversion_error:.3g}')
    fit = fit_drude_lorentz(wavelengths, round_trip_eps, len(_ROUND_TRIP_LORENTZ))
    if fit.drude is None:
        violations.append('round trip: the Drude pole was not detected (Re eps < 0 in the band)')
    round_trip_error = fit.relative_error

    # 2. library tables through effective_material_data: passive poles, eps_inf >= 1, the expanded poles reproduce the
    # PoleFit error with the DrudePole / LorentzPole chi of dispersion.complex_permittivity, error below the limit
    library_errors, library_loss_errors, library_poles = {}, {}, {}
    for name, n_lorentz, fit_start, fit_stop in _LIBRARY_FITS:
        entry = {'model': 'nk_fit', 'n_lorentz': n_lorentz, 'wavelength_start': fit_start,
                 'wavelength_stop': fit_stop}
        try:
            properties, poles = effective_material_data(name, {}, [entry])
        except ValueError as exc:
            violations.append(f'{name}: {exc}')
            continue
        library_poles[name] = poles
        if any(value < 0 for pole in poles for key, value in pole.items() if key != 'model'):
            violations.append(f'{name}: a negative pole value (not passive): {poles}')
        drude = [pole for pole in poles if pole['model'] == 'drude']
        eps_inf = drude[0]['eps_inf'] if drude else properties.get('eps_r')
        if eps_inf is None or eps_inf < 1.0:
            violations.append(f'{name}: background eps_inf {eps_inf} missing or below 1')
            continue
        if not drude and name == 'silicon' and properties.get('eps_r') == REGISTRY['silicon'].optical['eps_r']:
            violations.append('silicon: the library eps_r was not replaced by the fitted eps_inf')
        table = REGISTRY[name].nk_table
        samples, target = table_samples(table.wavelengths, table.n, table.k, fit_start, fit_stop)
        built = [(make_pole(pole, float), None) for pole in poles]
        expanded = np.array([complex(complex_permittivity(eps_inf, 0.0, built, 2.0 * math.pi * C0 / wavelength))
                             for wavelength in samples])
        error = float(np.max(np.abs(expanded - target) / np.abs(target)))
        reference = fitted_library_poles(name, n_lorentz, fit_start, fit_stop)   # the cached fit behind the expansion
        library_errors[name] = error
        library_loss_errors[name] = reference.loss_error
        if abs(error - reference.relative_error) > _CONSISTENCY_TOLERANCE:
            violations.append(f'{name}: expanded poles give error {error:.6g}, the fit reports '
                              f'{reference.relative_error:.6g}')
        if error > _LIBRARY_ERROR_LIMIT:
            violations.append(f'{name}: fit error {error:.3g} above {_LIBRARY_ERROR_LIMIT}')
    return {'measured': round_trip_error, 'expected': 0.0, 'error': round_trip_error + len(violations),
            'tolerance': _ROUND_TRIP_TOLERANCE,
            'details': {'round_trip_relative_error': round_trip_error, 'round_trip_loss_error': fit.loss_error,
                        'round_trip_fit': {'eps_inf': fit.eps_inf, 'drude': fit.drude, 'lorentz': fit.lorentz},
                        'library_relative_errors': library_errors, 'library_loss_errors': library_loss_errors,
                        'library_poles': library_poles, 'violations': violations}}


MATERIAL_FIT_SELFTEST_CASES = (
    SelfTestCase(
        name='em_nk_fit',
        label='EM tabulated optical data: Drude-Lorentz fit',
        description='CPU check of the nk_fit dispersion model (no model run). Round trip: eps of eps_inf 2 + Drude '
                    '(omega_p 1.3e16, gamma 6e13 rad/s) + Lorentz (1.5, 5.5e15, 6e14) and (0.8, 9e15, 2e15) at 61 '
                    'wavelengths 300-900 nm, converted to n, k ((n - j k)^2, principal root) and back, fitted with '
                    'Drude + 2 Lorentz poles: the exact model is in the search space, so the max |eps error| / |eps| '
                    'must vanish up to the float32 rounding of the pole values and the solver tolerance. Library: '
                    'silver (2 Lorentz, 400-900 nm), gold (3, 500-1000 nm), aluminum (3, 400-1000 nm), silicon (2, '
                    '400-1000 nm) expanded by materials.effective_material_data must give passive poles (all values '
                    '>= 0), eps_inf >= 1 (silicon: replacing the library eps_r), the PoleFit error again through '
                    'make_pole and dispersion.complex_permittivity (the FDFD / FDTD chi, to 1e-5) and a max relative '
                    f'error below {_LIBRARY_ERROR_LIMIT}. error = round-trip error + failed checks; tolerance '
                    f'{_ROUND_TRIP_TOLERANCE}. Not yet run with the closed-form fit Jacobian.',
        specs=lambda: {},
        check=_check_nk_fit,
        heavy=True,
    ),
)

"""Specs and checks of the magnetic_fields_ac 1D models (skin-depth slab driven by a held potential; current-driven solid
strip; run by selftests_backend magnet_ac_cpu_gpu and selftests_coupled) and the 'busbar_proximity' /
'flux_concentrator_needle_ac' templates; specs built fresh. Expectations are the exact finite-volume solutions on the
run's cells (1D complex systems solved densely here, float64)."""
import math

import numpy as np

from ...core.constants import MU0
from ...model.selftest_helpers import max_relative_error, relative_error
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, RefinementSpec, StudySpec, SweepSpec)
from .selftests import (_BAND, _CELL, _GAP_PARAMETER, _LENGTH, _MAGNETIC, _RUN, _TOLERANCE, _WIDTH, _band_rectangle,
                        _derived_value, _in_range, _mirror_y, _report, _window_field, needle_model)

MAGNETIC_AC_TYPE_NAME = 'magnetic_fields_ac'   # = the definition's type name (the definition imports this module)

# 1D layout of the DC cases (1 x 0.1 mm, 10 um cells, symmetry y faces). Copper sigma_dc; the frequencies put the skin
# depth delta = sqrt(2 / (omega mu0 sigma)) at ~10 cells (slab) and ~5 cells (strip half width 0.1 mm).
_COPPER = 5.8e7                   # S/m
_SKIN_FREQUENCY = 4.4e5           # Hz: delta ~ 99.6 um, the slab is ~10 delta deep
_DRIVE_POTENTIAL = 1e-6           # T m held on the first cell column
_STRIP_FREQUENCY = 1.75e6         # Hz: delta ~ 50 um
_STRIP_CURRENT = 1.0              # A (amplitude)
_STRIP_PHASE = 30.0               # deg
_IDENTITY_TOLERANCE = 1e-10       # circuit-row and Poynting identities (host float64)


def _skin_depth(frequency):
    return math.sqrt(2.0 / (2.0 * math.pi * frequency * MU0 * _COPPER))


# ---- exact discrete solutions (numpy, complex128) ----
def _line_operator(widths, sigma, omega, held_first=False):
    """Per unit y length 1D operator of the free cells between A = 0 x faces (or a held first cell): conductances
    2 nu / (w_i + w_i+1), Dirichlet / held half cells 2 nu / w, diagonal j omega sigma w; returns (matrix, first face
    conductance, face conductances of the free cells' n + 1 faces)."""
    reluctivity = 1.0 / MU0
    cells = slice(1, None) if held_first else slice(None)
    free_widths = widths[cells]
    n = free_widths.size
    faces = np.empty(n + 1)
    faces[0] = 2.0 * reluctivity / free_widths[0]
    faces[1:-1] = 2.0 * reluctivity / (free_widths[:-1] + free_widths[1:])
    faces[-1] = 2.0 * reluctivity / free_widths[-1]
    matrix = np.diag(faces[:-1] + faces[1:] + 1j * omega * sigma[cells] * free_widths).astype(np.complex128)
    matrix -= np.diag(faces[1:-1], 1) + np.diag(faces[1:-1], -1)
    return matrix, faces


def _centre_flux_density(potential, faces, first_value):
    """B_y = (face flux mean) / nu at the free cell centres: face flux g (u_L - u_R), u_L of the first face first_value,
    u_R of the last face 0."""
    padded = np.concatenate(([first_value], potential, [0.0]))
    face_flux = faces * (padded[:-1] - padded[1:])
    return MU0 * 0.5 * (face_flux[:-1] + face_flux[1:])


def _skin_solution(edges):
    """Free cells 1.. of the held-driven slab: A, B_y, q_eddy and the loss per m of depth (times the window height)."""
    widths = np.diff(edges)
    omega = 2.0 * math.pi * _SKIN_FREQUENCY
    sigma = np.full(widths.size, _COPPER)
    matrix, faces = _line_operator(widths, sigma, omega, held_first=True)
    rhs = np.zeros(widths.size - 1, dtype=np.complex128)
    rhs[0] = faces[0] * _DRIVE_POTENTIAL
    potential = np.linalg.solve(matrix, rhs)
    loss_density = 0.5 * _COPPER * omega ** 2 * np.abs(potential) ** 2
    return {'A': potential, 'B_y': _centre_flux_density(potential, faces, _DRIVE_POTENTIAL), 'q_eddy': loss_density,
            'loss': _WIDTH * float(np.sum(loss_density * widths[1:]))}


def _strip_solution(edges):
    """Current-driven strip between A = 0 faces: A, J = sigma (v - j omega A), v [V/m], loss per m of depth."""
    widths = np.diff(edges)
    omega = 2.0 * math.pi * _STRIP_FREQUENCY
    sigma = np.where(_in_range(edges, *_BAND), _COPPER, 0.0)
    matrix, _faces = _line_operator(widths, sigma, omega)
    weights = sigma * widths   # per unit y length
    n = widths.size
    bordered = np.zeros((n + 1, n + 1), dtype=np.complex128)
    bordered[:n, :n] = matrix
    bordered[:n, n] = -weights
    bordered[n, :n] = -1j * omega * weights
    bordered[n, n] = np.sum(weights)
    rhs = np.zeros(n + 1, dtype=np.complex128)
    rhs[n] = _STRIP_CURRENT * np.exp(1j * math.radians(_STRIP_PHASE)) / _WIDTH
    solution = np.linalg.solve(bordered, rhs)
    potential, drive = solution[:n], solution[n]
    density = sigma * (drive - 1j * omega * potential)
    return {'A': potential, 'J': density, 'v': drive,
            'loss': _WIDTH * float(np.sum(0.5 * np.abs(density) ** 2 / np.where(sigma > 0, sigma, 1.0) * widths
                                          * (sigma > 0)))}


# ---- specs ----
def _ac_model(title, features, materials, geometry, frequency, derived_values=()):
    """Stationary magnetic_fields_ac-only model on the uniform _CELL mesh of the DC cases."""
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=_LENGTH, size_y=_WIDTH), geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_MAGNETIC, MAGNETIC_AC_TYPE_NAME, settings={'frequency': frequency},
                             features=list(features))],
        mesh=MeshSpec(mode='uniform', dx=_CELL), study=StudySpec(type='stationary'), derived_values=list(derived_values))


def _loss_value():
    return DerivedValueSpec('P', f'{_MAGNETIC}_eddy_loss', [], 'max')


def _skin_specs():
    """'run': copper everywhere, A = 1e-6 T m held on the first cell column, insulation x_hi, symmetry y faces."""
    return {_RUN: _ac_model(
        'Self-test: skin-depth slab (magnetic_fields_ac)',
        [FeatureSpec('drive', 'fixed_potential', ['drive_strip'], {'potential': _DRIVE_POTENTIAL}),
         FeatureSpec('wall', 'magnetic_insulation', ['x_hi']), _mirror_y()],
        [MaterialSpec('copper', properties={'sigma_dc': _COPPER}, selection=['slab'])],
        [_band_rectangle('slab', 0.0, _LENGTH), _band_rectangle('drive_strip', 0.0, _CELL)],
        _SKIN_FREQUENCY, [_loss_value()])}


def _strip_specs():
    """'run': solid copper strip 'bar' on the band, current 1 A at 30 deg, insulation x faces (default)."""
    return {_RUN: _ac_model(
        'Self-test: current-driven solid strip (magnetic_fields_ac)',
        [_mirror_y(),
         FeatureSpec('bar', 'coil', ['band'], {'conductor': 'solid', 'drive': 'current', 'current': _STRIP_CURRENT,
                                               'phase': _STRIP_PHASE})],
        [MaterialSpec('copper', properties={'sigma_dc': _COPPER}, selection=['band'])],
        [_band_rectangle('band', *_BAND)], _STRIP_FREQUENCY, [_loss_value()])}


# ---- checks ----
def _complex_field(results, component):
    real, edges, _ = _window_field(results, f'{component}_re')
    imaginary, _, _ = _window_field(results, f'{component}_im')
    return real + 1j * imaginary, edges


def _check_skin(results_by_name):
    """A, B_y and q_eddy in the free cells and the eddy_loss scalar against the exact discrete slab."""
    results = results_by_name[_RUN]
    potential, edges = _complex_field(results, 'A')
    flux_density, _ = _complex_field(results, 'B_y')
    loss_density, _, _ = _window_field(results, 'q_eddy')
    exact = _skin_solution(edges)
    loss = _derived_value(results, 'P')
    free = slice(1, None)
    field_scale = float(np.max(np.abs(exact['B_y'])))
    errors = {'A': max_relative_error(potential[free], exact['A'][:, None], _DRIVE_POTENTIAL),
              'B_y': max_relative_error(flux_density[free], exact['B_y'][:, None], field_scale),
              'q_eddy': max_relative_error(loss_density[free], exact['q_eddy'][:, None], float(np.max(exact['q_eddy']))),
              'eddy_loss': relative_error(loss, exact['loss'])}
    delta = _skin_depth(_SKIN_FREQUENCY)
    return _report(loss, exact['loss'], errors, skin_depth_m=delta, B_surface_T=field_scale,
                   continuous_B_surface_T=_DRIVE_POTENTIAL * math.sqrt(2.0) / delta, n_cells=list(potential.shape))


def _check_strip(results_by_name):
    """A, J_conduction and v against the exact bordered system; the circuit row (current = I) and the Poynting
    identity (eddy_loss = Re(V I*) / 2) to _IDENTITY_TOLERANCE, entered as error * _TOLERANCE / _IDENTITY_TOLERANCE."""
    results = results_by_name[_RUN]
    potential, edges = _complex_field(results, 'A')
    density, _ = _complex_field(results, 'J_conduction')
    exact = _strip_solution(edges)
    row = results.result(_MAGNETIC, 'bar')
    current = complex(row['current_re'], row['current_im'])
    voltage = complex(row['voltage_re'], row['voltage_im'])
    prescribed = _STRIP_CURRENT * np.exp(1j * math.radians(_STRIP_PHASE))
    loss = _derived_value(results, 'P')
    identity_scale = _TOLERANCE / _IDENTITY_TOLERANCE
    errors = {'A': max_relative_error(potential, exact['A'][:, None], float(np.max(np.abs(exact['A'])))),
              'J_conduction': max_relative_error(density, exact['J'][:, None], float(np.max(np.abs(exact['J'])))),
              'voltage': abs(voltage - exact['v']) / abs(exact['v']),
              'eddy_loss': relative_error(loss, exact['loss']),
              'current_row_scaled': identity_scale * abs(current - prescribed) / abs(prescribed),
              'poynting_scaled': identity_scale * relative_error(loss, float(row['power']))}
    return _report(abs(current), _STRIP_CURRENT, errors, resistance_ohm_per_m=float(row['resistance']),
                   dc_resistance_ohm_per_m=1.0 / (_COPPER * (_BAND[1] - _BAND[0]) * _WIDTH),
                   inductance_H_per_m=float(row['inductance']), skin_depth_m=_skin_depth(_STRIP_FREQUENCY),
                   n_cells=list(potential.shape))


# ---- template ----
# busbar_proximity: two copper bars 10 x 2 mm, wide faces 2 mm apart, go (+I) and return (-I), in a 40 x 30 mm box of
# magnetic insulation (>= 13 mm from the bars: the loop field decays as a dipole). Mesh: base 1 mm, the bars at
# 0.1 mm (delta at 20 kHz ~ 0.47 mm). Not run yet: cell count, factor size and the R_ac / R_dc values unverified.
BUSBAR_LABEL = 'Busbar pair, proximity effect'
BUSBAR_DESCRIPTION = (
    'Magnetic fields AC, planar, stationary sweep over the frequency f = 50 Hz .. 20 kHz: two copper bars 10 x 2 mm '
    '(wide faces 2 mm apart) carry +I and -I (solid conductors, prescribed current 100 A): the current crowds onto '
    'the facing surfaces. Derived R_loop, L_loop (per m of length) and the loss P per m.')
_BAR_SIZE = (10e-3, 2e-3)          # m
_BAR_GAP = 2e-3                    # m
_BUS_DOMAIN = (40e-3, 30e-3)       # m
_BUS_FREQUENCIES = [50.0, 500.0, 2000.0, 5000.0, 20000.0]   # Hz


def busbar_proximity_spec():
    """Template 'busbar_proximity' (planar magnetic_fields_ac, stationary sweep over f)."""
    size_x, size_y = _BUS_DOMAIN
    x0 = 0.5 * (size_x - _BAR_SIZE[0])
    y_go = 0.5 * (size_y - _BAR_GAP) - _BAR_SIZE[1]
    y_back = 0.5 * (size_y + _BAR_GAP)

    def bar(name, y0):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x0 + _BAR_SIZE[0], 'y1': y0 + _BAR_SIZE[1]})

    def solid(name, current):
        return FeatureSpec(name, 'coil', [name], {'conductor': 'solid', 'drive': 'current', 'current': current})

    return ModelSpec(
        title=BUSBAR_LABEL, description=BUSBAR_DESCRIPTION,
        parameters=[ParameterSpec('f', 1000.0, 'frequency [Hz]'), ParameterSpec('I', 100.0, 'bar current [A]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[bar('go', y_go), bar('back', y_back)],
        materials=[MaterialSpec('copper', properties={'sigma_dc': _COPPER}, selection=['go', 'back'])],
        physics=[PhysicsSpec(_MAGNETIC, MAGNETIC_AC_TYPE_NAME, settings={'frequency': 'f'},
                             features=[solid('go', 'I'), solid('back', '-I')])],
        mesh=MeshSpec(mode='nonuniform', dx=1e-3,
                      refinements=[RefinementSpec('go', 0.1e-3, 1e-3), RefinementSpec('back', 0.1e-3, 1e-3)]),
        study=StudySpec(type='stationary', sweep=SweepSpec('f', list(_BUS_FREQUENCIES))),
        derived_values=[DerivedValueSpec('R_loop', f'{_MAGNETIC}_go_resistance + {_MAGNETIC}_back_resistance', [],
                                         'max'),
                        DerivedValueSpec('L_loop', f'{_MAGNETIC}_go_inductance + {_MAGNETIC}_back_inductance', [],
                                         'max'),
                        DerivedValueSpec('P', f'{_MAGNETIC}_eddy_loss', [], 'max')])


# flux_concentrator_needle_ac: the DC needle model (selftests.needle_model) as magnetic_fields_ac with the needle's
# conductivity and magnetic loss, gap fixed, swept over f. Metglas 2714A-like: resistivity ~1.4 uOhm m (7e5 S/m), skin
# depth sqrt(2 / (omega mu0 mu sigma)) at mu 1e5: 60 um at 1 kHz (~ the 31 um radius: the shielding knee), 11 um at
# 30 kHz (~2 of the 5 um cells: the top of the resolved range). Not run yet: cell count, factor size and gains unverified.
NEEDLE_AC_LABEL = 'Flux-concentrator needle, AC'
NEEDLE_AC_DESCRIPTION = (
    'Magnetic fields AC, axisymmetric, stationary sweep over the frequency f = 10 Hz .. 30 kHz: the flux_concentrator_'
    'needle model (Metglas needle r = 31 um, l = 15 mm, mu_dc 1e5, applied axial field amplitude B_ext = 1e-4 T) with '
    'the needle conductivity sigma_needle = 7e5 S/m and magnetic loss mu_dc_loss = loss_tangent mu_dc: eddy currents '
    'screen the needle once the skin depth falls below its radius (~1 kHz), so the gain = B_sensor / B_ext (mean |B| '
    'over a 31 um disc a gap = 100 um below the tip; EF = 20 log10(gain) dB) drops with frequency.')
_NEEDLE_FREQUENCIES = [10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0]   # Hz


def flux_concentrator_needle_ac_spec():
    """Template 'flux_concentrator_needle_ac' (axisymmetric magnetic_fields_ac, stationary sweep over f)."""
    return needle_model(
        NEEDLE_AC_LABEL, NEEDLE_AC_DESCRIPTION, MAGNETIC_AC_TYPE_NAME, {'frequency': 'f'},
        [_GAP_PARAMETER, ('f', 1000.0, 'frequency [Hz]'), ('sigma_needle', 7e5, 'needle conductivity [S/m]'),
         ('loss_tangent', 0.01, 'magnetic loss tangent mu_dc_loss / mu_dc of the needle')],
        {'sigma_dc': 'sigma_needle', 'mu_dc_loss': 'loss_tangent * mu_needle'},
        SweepSpec('f', list(_NEEDLE_FREQUENCIES)))

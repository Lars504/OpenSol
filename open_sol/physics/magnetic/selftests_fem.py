"""Self-tests of magnetostatics on triangle meshes (magnetic plan section 10.4, MagneticFEMPhysics): the finite-difference
check of residual and tangent with a scripted Newton run, the planar current sheet, the axisymmetric uniform field,
the coil pair, the remanent slab, the flux-driven saturating slab, the current-driven saturating sheets (J_s curve deep
in saturation, tabulated curve), triangles vs the tensor grid on the iron C-core, and the 'iron_c_core_saturation' template. Every
polynomial case is exact for P2 (the solution lies in the space); expectations are the continuous closed forms.
Registered by definition_magnetostatics."""
import math

import numpy as np

from ...core.constants import MU0
from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import last_value, scaled_worst_result
from ...model.spec import (AnalyticCurveSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec, SolverDefaultsSpec,
                           StudySpec, SweepSpec)
from .selftests import (_APPLIED_FIELD, _BAND, _COIL_CURRENT, _COIL_RANGES, _COIL_TURNS, _CURRENT_DENSITY,
                        _IRON_PERMEABILITY, _IRON_SATURATION, _LENGTH, _MAGNETIC, _REMANENCE, _RUN,
                        _SATURATING_CURRENT_DENSITY,
                        _SATURATING_FIELD, _UNIFORM_DOMAIN, _WIDTH, MAGNETOSTATICS_TYPE_NAME, _band_rectangle,
                        _curve_field, _curve_flux_density, _derived_value, _energy_value, _iron, _mirror_y,
                        _nonlinear_iterations, _table_flux_density, _table_rows)

_ELEMENT = 1e-5                    # m, uniform triangle size of the 1 x 0.1 mm layout (as the tensor cells)
_EXACT = 1e-9                      # relative, float64 results of the polynomial cases (energy, scalars)
_FLOAT32 = 1e-6                    # relative, float32 probe / field values of the polynomial cases
_PROBE_X = (0.2e-3, 0.5e-3, 0.8e-3)   # m: left of, inside and right of the band, at mid height
_BAND_CENTRE = 0.5 * (_BAND[0] + _BAND[1])   # m: B_y of the current sheet is odd about it (band mean of B_y = 0)
_FEM_NONLINEAR = {'nonlinear_tolerance': 1e-9, 'nonlinear_max_iterations': 50}
_DEEP_CURRENT_DENSITY = 1e8        # A/m^2: H to 2e4 A/m, B ~ J_s + mu0 H, s = d ln B / d ln H ~ 0.03
_SHEET_TOLERANCES = {'curve': 1e-4, 'table': 1e-3}   # relative A at the probes (first estimates, not run)
# Newton steps from the B = 0 limit start (B ~3.5x the saturated value): seen sheet 10, deep 10, table 7, no backtracks
_NEWTON_LIMITS = {'slab': 3, 'sheet': 12, 'deep': 15}
_REFERENCE_INTERVALS = 4000        # Gauss-Legendre sub-intervals of the reference A = -int B_y dx
_MAGNET_PERMEABILITY = 1.05        # mu_dc of the remanent band (NdFeB-like: the load int Q^T nu B_r sees nu != nu0)

# ---- finite-difference check (CPU, scripted on the shared fem disc mesh) ----
_TANGENT_ELEMENT = 0.12e-6         # m: fem.selftests_mesher disc mesh (2 x 2 um, disc r 0.6 um, curved edges)
_TANGENT_FIELD = 1.2               # T scale of the test potential (|B| crosses the knee of both curves)
_TANGENT_STEP = 1e-7               # central-difference step relative to max |a| (table kinks: crossing chance ~1e-3)
_TANGENT_TOLERANCE = 1e-6
_OUTER_PERMEABILITY = (3.0, 5.0)   # anisotropic linear mu_dc_xx / _yy outside the disc
_OUTER_REMANENCE = (0.2, -0.1)     # T, remanence outside the disc (a load: checks the W / R pairing)
_NEWTON_CURRENT_DENSITY = 1e10     # A/m^2 in the disc: H ~ J r / 2 ~ 3000 A/m at its edge
_NEWTON_MAX_STEPS = 10
_QUADRATIC_ORDER = 1.5             # log r_(k+1) / log r_k of the last steps between 1e-2 and 1e-12

# ---- iron C-core (template, triangles vs tensor) ----
IRON_C_CORE_LABEL = 'Iron C-core saturation'
IRON_C_CORE_DESCRIPTION = (
    'Magnetostatics on a triangle mesh, planar, stationary sweep: an iron C-core (80 x 60 mm, 10 mm limbs, mu_dc 4000, '
    'saturation_polarization 1.8 T) with a gap_width = 2 mm air gap in its right limb, driven by a 200-turn coil '
    'around the left limb (go and return sides as two coil features); I_coil swept 0.5 .. 32 A. Derived B_gap = mean '
    '|B| in the gap against the ideal mu0 N I / g (analytic curve, valid below the knee: the core reluctance takes ~3 '
    '%); Newton (nonlinear_method auto) solves the saturation.')
_CORE_DOMAIN = (160e-3, 120e-3)    # m
_CORE_CURRENTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]   # A
_VS_TENSOR_TOLERANCE = 1e-2        # relative B_gap triangles vs graded tensor (first estimate, not run)


# ---- exact continuous 1D solutions (numpy) ----
def _gauss_integral(function, breaks, points=3):
    """int function dx over [breaks[0], breaks[-1]] by Gauss-Legendre on every interval between the breaks."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    total = 0.0
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        half = 0.5 * (upper - lower)
        total += half * float(np.sum(weights * function(lower + half * (nodes + 1.0))))
    return total


def _line_solution(pieces, length=_LENGTH):
    """Continuous planar mu = 1 solution between A = 0 at x = 0 and x = length (symmetry y faces) of current densities
    pieces [(x0, x1, J)]: (A(x) callable [T m], B_y(x) callable [T], energy per m of depth per m of y width [J/m^2])."""
    reluctivity = 1.0 / MU0

    def enclosed(x):   # int_0^x J
        return sum(density * np.clip(x - x0, 0.0, x1 - x0) for x0, x1, density in pieces)

    def enclosed_integral(x):   # int_0^x int_0^s J
        return sum(density * (0.5 * np.clip(x - x0, 0.0, x1 - x0) ** 2 + (x1 - x0) * np.maximum(x - x1, 0.0))
                   for x0, x1, density in pieces)

    field_0 = -enclosed_integral(length) / length   # H_y(0), from A(length) = 0

    def potential(x):
        return -(field_0 * x + enclosed_integral(x)) / reluctivity

    def flux_y(x):
        return (field_0 + enclosed(x)) / reluctivity

    breaks = sorted({0.0, length, *(x for x0, x1, _ in pieces for x in (x0, x1))})
    energy = _gauss_integral(lambda x: 0.5 * reluctivity * flux_y(x) ** 2, breaks)
    return potential, flux_y, energy


def _sheet_reference(density, flux_of_field, positions):
    """A [T m] at positions of the current-driven sheet (J in the band, A = 0 on x_lo, symmetry x_hi): H_y(x) = -int_x^L
    J, B_y = curve(H_y), A = -int_0^x B_y by composite Gauss-Legendre."""
    lower, upper = _BAND

    def flux_y(x):
        field = -density * (upper - np.clip(x, lower, upper))
        return flux_of_field(field)

    return [-_gauss_integral(flux_y, np.linspace(0.0, x, _REFERENCE_INTERVALS + 1), points=5) for x in positions]


# ---- specs ----
def _fem_model(title, features, materials=(), geometry=(), derived_values=(), axisymmetric=False,
               size=(_LENGTH, _WIDTH), settings=None, discretization='quadratic', element=_ELEMENT):
    """Stationary magnetostatics-only model on a uniform 'triangles' mesh, cpu backend (Automatic: host SuperLU)."""
    settings = dict(settings or {}, discretization=discretization)
    if axisymmetric:
        settings['geometry'] = 'axisymmetric'
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]), geometry=list(geometry),
        materials=list(materials),
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_TYPE_NAME, settings=settings, features=list(features))],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=element)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'), study=StudySpec(type='stationary'),
        derived_values=list(derived_values))


def _probes():
    """(probe points, probe features) at _PROBE_X, mid height."""
    points = [GeometrySpec(f'p{index}', 'point', {'x': x, 'y': 0.5 * _WIDTH}) for index, x in enumerate(_PROBE_X)]
    features = [FeatureSpec(f'probe{index}', 'probe', [point.name]) for index, point in enumerate(points)]
    return points, features


def _probe_values(results):
    """float64 A [T m] of the probes' last samples."""
    return np.array([last_value(results, _MAGNETIC, f'probe{index}') for index in range(len(_PROBE_X))])


def _current_sheet_specs():
    """'run': J = 1e6 A/m^2 in the band between insulation x faces (P2), probes, derived W and the band mean of
    (x - x_c) B_y."""
    points, probes = _probes()
    return {_RUN: _fem_model(
        'Self-test: planar current sheet on triangles (magnetostatics)',
        [FeatureSpec('walls', 'magnetic_insulation', ['x_lo', 'x_hi']), _mirror_y(),
         FeatureSpec('sheet', 'current_source', ['band'], {'density': _CURRENT_DENSITY}), *probes],
        geometry=[_band_rectangle('band', *_BAND), *points],
        derived_values=[_energy_value(),
                        DerivedValueSpec('B_moment', f'B_y * (x - {_BAND_CENTRE!r})', ['band'], 'mean')])}


def _uniform_field_specs():
    """'linear' / 'quadratic': empty axisymmetric box in an applied axial B = 1e-3 T, P1 and P2."""
    derived = [_energy_value, lambda: DerivedValueSpec('Bz_max', 'B_y', ['all'], 'max'),
               lambda: DerivedValueSpec('Bz_min', 'B_y', ['all'], 'min'),
               lambda: DerivedValueSpec('Br_max', 'abs(B_x)', ['all'], 'max')]
    return {order: _fem_model(
        'Self-test: uniform axial field on triangles (axisymmetric magnetostatics)',
        [FeatureSpec('applied', 'external_field', ['all'], {'B_x': 0.0, 'B_y': _APPLIED_FIELD})],
        derived_values=[make() for make in derived], axisymmetric=True, size=_UNIFORM_DOMAIN, discretization=order,
        element=2.5e-5) for order in ('linear', 'quadratic')}


def _coil_pair_specs():
    """'run': coils go (+I) and back (-I), 10 turns each, between insulation x faces (P2); derived W, L_go, L_back."""
    return {_RUN: _fem_model(
        'Self-test: go / return coil pair on triangles (magnetostatics)',
        [_mirror_y(),
         FeatureSpec('go', 'coil', ['go'], {'current': _COIL_CURRENT, 'turns': _COIL_TURNS}),
         FeatureSpec('back', 'coil', ['back'], {'current': -_COIL_CURRENT, 'turns': _COIL_TURNS})],
        geometry=[_band_rectangle(name, *span) for name, span in _COIL_RANGES.items()],
        derived_values=[_energy_value(), DerivedValueSpec('L_go', f'{_MAGNETIC}_go_inductance', [], 'max'),
                        DerivedValueSpec('L_back', f'{_MAGNETIC}_back_inductance', [], 'max')])}


def _remanent_slab_specs():
    """'linear' / 'quadratic': remanence_y with mu_dc 1.05 in the band between insulation x faces, symmetry y faces,
    probes (p1 at the band centre), derived W, H_y max / min and B_y at p1."""
    def model(order):
        points, probes = _probes()
        return _fem_model(
            'Self-test: remanent slab on triangles (magnetostatics)',
            [FeatureSpec('walls', 'magnetic_insulation', ['x_lo', 'x_hi']), _mirror_y(), *probes],
            materials=[MaterialSpec('magnet', properties={'remanence_y': _REMANENCE, 'mu_dc': _MAGNET_PERMEABILITY},
                                    selection=['band'])],
            geometry=[_band_rectangle('band', *_BAND), *points],
            derived_values=[_energy_value(), DerivedValueSpec('Hy_max', 'H_y', ['all'], 'max'),
                            DerivedValueSpec('Hy_min', 'H_y', ['all'], 'min'),
                            DerivedValueSpec('B_magnet', 'B_y', [], 'at', point='p1')],
            discretization=order)
    return {order: model(order) for order in ('linear', 'quadratic')}


def _saturating_slab_specs():
    """'run': saturating iron band across the full y span in an applied B_x = 0.8 T (normal to its faces), Newton;
    H_x read at the band centre (a band mean would take the nodal averages smeared across the iron / air outline)."""
    return {_RUN: _fem_model(
        'Self-test: saturating slab in an applied field on triangles (magnetostatics)',
        [FeatureSpec('applied', 'external_field', ['all'], {'B_x': _SATURATING_FIELD, 'B_y': 0.0})],
        materials=[_iron(['band'])],
        geometry=[_band_rectangle('band', *_BAND),
                  GeometrySpec('centre', 'point', {'x': _BAND_CENTRE, 'y': 0.5 * _WIDTH})],
        derived_values=[DerivedValueSpec('Bx_max', 'B_x', ['all'], 'max'),
                        DerivedValueSpec('Bx_min', 'B_x', ['all'], 'min'),
                        DerivedValueSpec('H_iron', 'H_x', [], 'at', point='centre')],
        settings=_FEM_NONLINEAR)}


def _sheet_specs(density, material):
    """'run': J in the band, the material everywhere, insulation x_lo, symmetry x_hi and y faces (P2, Newton)."""
    points, probes = _probes()
    return {_RUN: _fem_model(
        'Self-test: current sheet in saturating iron on triangles (magnetostatics)',
        [FeatureSpec('wall', 'magnetic_insulation', ['x_lo']),
         FeatureSpec('mirror', 'symmetry', ['x_hi', 'y_lo', 'y_hi']),
         FeatureSpec('sheet', 'current_source', ['band'], {'density': density}), *probes],
        materials=[material], geometry=[_band_rectangle('band', *_BAND), *points], settings=_FEM_NONLINEAR)}


def _table_iron():
    return MaterialSpec('iron_table', properties={'bh_curve': _table_rows()}, selection=['all'])


def iron_c_core_model(title, description, linear=False, triangles=True, sweep=True):
    """C-core model: iron (mu_dc 4000, J_s 1.8 T unless linear), gap_width gap in the right limb, 200-turn coil (go /
    return) around the left limb, insulation faces, derived B_gap; triangles (refined in the gap) or a graded tensor
    mesh; sweep: I_coil over _CORE_CURRENTS with the ideal-gap analytic curve."""
    mm = 1e-3

    def rectangle(name, x0, y0, x1, y1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    iron = {'mu_dc': 4000.0} if linear else {'mu_dc': 4000.0, 'saturation_polarization': 1.8}
    if triangles:
        mesh = MeshSpec(name='triangles', kind='triangles', mode='nonuniform', dx=3 * mm,
                        refinements=[RefinementSpec('gap_probe', 0.4 * mm)])
    else:
        mesh = MeshSpec(name='cells', mode='nonuniform', dx=1 * mm, refinements=[RefinementSpec('gap_probe', 0.25 * mm)])
    return ModelSpec(
        title=title, description=description,
        parameters=[ParameterSpec('I_coil', 1.0, 'coil current per turn [A]'),
                    ParameterSpec('N_turns', 200, 'turns of the coil'),
                    ParameterSpec('gap_width', 2 * mm, 'air gap height [m]')],
        domain=DomainSpec(size_x=_CORE_DOMAIN[0], size_y=_CORE_DOMAIN[1]),
        geometry=[rectangle('outer', 40 * mm, 30 * mm, 120 * mm, 90 * mm),
                  rectangle('window', 50 * mm, 40 * mm, 110 * mm, 80 * mm),
                  rectangle('gap', 110 * mm, '60e-3 - gap_width / 2', 120 * mm, '60e-3 + gap_width / 2'),
                  GeometrySpec('core', 'difference', {'base': 'outer', 'subtract': ['window', 'gap']}),
                  rectangle('gap_probe', 112 * mm, '60e-3 - gap_width / 4', 118 * mm, '60e-3 + gap_width / 4'),
                  rectangle('coil_in', 51 * mm, 45 * mm, 57 * mm, 75 * mm),
                  rectangle('coil_out', 33 * mm, 45 * mm, 39 * mm, 75 * mm)],
        materials=[MaterialSpec('iron', properties=iron, selection=['core'])],
        physics=[PhysicsSpec(_MAGNETIC, MAGNETOSTATICS_TYPE_NAME, features=[
            FeatureSpec('coil_go', 'coil', ['coil_in'], {'current': 'I_coil', 'turns': 'N_turns'}),
            FeatureSpec('coil_return', 'coil', ['coil_out'], {'current': '-I_coil', 'turns': 'N_turns'})])],
        meshes=[mesh], solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary', sweep=SweepSpec('I_coil', list(_CORE_CURRENTS)) if sweep else None),
        derived_values=[DerivedValueSpec('B_gap', 'B_abs', ['gap_probe'], 'mean')],
        analytic_curves=([AnalyticCurveSpec('ideal_gap', 'sweep', 'mu0_const * N_turns * I_coil / gap_width',
                                            series='B_gap')] if sweep else []))


def iron_c_core_spec():
    """Template 'iron_c_core_saturation' (triangles, planar, Newton, I_coil sweep)."""
    return iron_c_core_model(IRON_C_CORE_LABEL, IRON_C_CORE_DESCRIPTION)


def _vs_tensor_specs():
    """'triangles' / 'tensor': the linear C-core at I_coil = 1 A on both mesh kinds."""
    return {name: iron_c_core_model(f'Self-test: linear iron C-core on {name}', '', linear=True,
                                    triangles=name == 'triangles', sweep=False) for name in ('triangles', 'tensor')}


# ---- checks ----
def _check_current_sheet(results_by_name):
    """A at the probes (float32), the energy (float64) and the band mean of (x - x_c) B_y (the plain mean is 0 by
    symmetry) against the continuous solution."""
    results = results_by_name[_RUN]
    potential, flux_y, energy_per_width = _line_solution([(*_BAND, _CURRENT_DENSITY)])
    expected_potential = potential(np.array(_PROBE_X))
    measured_potential = _probe_values(results)
    expected_moment = (_gauss_integral(lambda x: flux_y(x) * (x - _BAND_CENTRE), list(_BAND))
                       / (_BAND[1] - _BAND[0]))
    energy = _derived_value(results, 'W')
    moment = _derived_value(results, 'B_moment')
    errors = {'A_probes': float(np.max(np.abs(measured_potential - expected_potential))
                                / np.max(np.abs(expected_potential))),
              'energy': abs(energy / (energy_per_width * _WIDTH) - 1.0),
              'B_moment': abs(moment / expected_moment - 1.0)}
    return scaled_worst_result(errors, {'A_probes': _FLOAT32, 'energy': _EXACT, 'B_moment': _FLOAT32},
                               A_measured=measured_potential.tolist(), A_expected=expected_potential.tolist(),
                               energy=energy, energy_expected=energy_per_width * _WIDTH,
                               B_moment=moment, B_moment_expected=expected_moment)


def _check_uniform_field(results_by_name):
    """P1 and P2: B_z = B everywhere (max / min samples), B_r = 0, energy B^2 / (2 mu0) pi R^2 Z (float64)."""
    radius, height = _UNIFORM_DOMAIN
    expected_energy = _APPLIED_FIELD ** 2 / (2.0 * MU0) * math.pi * radius ** 2 * height
    errors, details = {}, {}
    for order, results in results_by_name.items():
        values = {name: _derived_value(results, name) for name in ('W', 'Bz_max', 'Bz_min', 'Br_max')}
        errors[f'{order}_Bz'] = max(abs(values['Bz_max'] / _APPLIED_FIELD - 1.0),
                                    abs(values['Bz_min'] / _APPLIED_FIELD - 1.0))
        errors[f'{order}_Br'] = values['Br_max'] / _APPLIED_FIELD
        errors[f'{order}_energy'] = abs(values['W'] / expected_energy - 1.0)
        details[order] = values
    tolerances = {name: _EXACT if name.endswith('energy') else _FLOAT32 for name in errors}
    return scaled_worst_result(errors, tolerances, measured=details, energy_expected=expected_energy)


def _check_coil_pair(results_by_name):
    """W, L_go, L_back against the continuous solution of the two strips (exact for P2), L_go + L_back = 2 W / I^2."""
    results = results_by_name[_RUN]
    densities = {name: sign * _COIL_TURNS * _COIL_CURRENT / ((x1 - x0) * _WIDTH)
                 for (name, (x0, x1)), sign in zip(_COIL_RANGES.items(), (1.0, -1.0))}
    potential, _, energy_per_width = _line_solution([(*_COIL_RANGES[name], density)
                                                     for name, density in densities.items()])
    expected = {'W': energy_per_width * _WIDTH}
    for (name, (x0, x1)), sign in zip(_COIL_RANGES.items(), (1.0, -1.0)):
        linkage = _COIL_TURNS * _gauss_integral(potential, [x0, x1]) / (x1 - x0)
        expected[f'L_{name}'] = linkage / (sign * _COIL_CURRENT)
    measured = {name: _derived_value(results, name) for name in expected}
    errors = {name: abs(measured[name] / expected[name] - 1.0) for name in expected}
    errors['energy_identity'] = abs((measured['L_go'] + measured['L_back']) * _COIL_CURRENT ** 2
                                    / (2.0 * measured['W']) - 1.0)
    return scaled_worst_result(errors, dict.fromkeys(errors, _EXACT), measured=measured, expected=expected)


def _check_remanent_slab(results_by_name):
    """P1 and P2: H_y = H0 everywhere (curl H = 0, int B_y dx = 0 from A = 0 on both x faces), B_y = mu0 mu H0 + B_r in
    the band, A piecewise linear (in both spaces): probes, H max / min, B at p1 (float32) and W (float64)."""
    lower, upper = _BAND
    thickness = upper - lower
    permeability = float(np.float32(_MAGNET_PERMEABILITY))   # the float32 MaterialGrid value (1.05: -4.5e-8 relative)
    field = -_REMANENCE * thickness / (MU0 * (permeability * thickness + _LENGTH - thickness))
    magnet_flux, outer_flux = MU0 * permeability * field + _REMANENCE, MU0 * field
    positions = np.array(_PROBE_X)
    expected_potential = -(outer_flux * np.minimum(positions, lower) + magnet_flux * np.clip(positions - lower, 0.0,
                           thickness) + outer_flux * np.maximum(positions - upper, 0.0))
    expected_energy = _WIDTH * (thickness * magnet_flux ** 2 / (2.0 * MU0 * permeability)
                                + (_LENGTH - thickness) * outer_flux ** 2 / (2.0 * MU0))
    errors, details = {}, {}
    for order, results in results_by_name.items():
        values = {name: _derived_value(results, name) for name in ('W', 'Hy_max', 'Hy_min', 'B_magnet')}
        measured_potential = _probe_values(results)
        errors[f'{order}_A_probes'] = float(np.max(np.abs(measured_potential - expected_potential))
                                            / np.max(np.abs(expected_potential)))
        errors[f'{order}_H'] = max(abs(values['Hy_max'] / field - 1.0), abs(values['Hy_min'] / field - 1.0))
        errors[f'{order}_B_magnet'] = abs(values['B_magnet'] / magnet_flux - 1.0)
        errors[f'{order}_energy'] = abs(values['W'] / expected_energy - 1.0)
        details[order] = dict(values, A_probes=measured_potential.tolist())
    tolerances = {name: _EXACT if name.endswith('energy') else _FLOAT32 for name in errors}
    return scaled_worst_result(errors, tolerances, measured=details, H_expected=field, B_magnet_expected=magnet_flux,
                               A_expected=expected_potential.tolist(), energy_expected=expected_energy)


def _check_saturating_slab(results_by_name):
    """B_x = 0.8 T everywhere, H at the band centre = H(0.8 T) of the curve (bisection), Newton steps <= 3."""
    results = results_by_name[_RUN]
    values = {name: _derived_value(results, name) for name in ('Bx_max', 'Bx_min', 'H_iron')}
    expected_field = _curve_field(_SATURATING_FIELD)
    iterations = _nonlinear_iterations(results)
    errors = {'B': max(abs(values['Bx_max'] / _SATURATING_FIELD - 1.0), abs(values['Bx_min'] / _SATURATING_FIELD - 1.0)),
              'H_iron': abs(values['H_iron'] / expected_field - 1.0),
              'newton_steps': 0.0 if iterations is not None and iterations <= _NEWTON_LIMITS['slab'] else math.inf}
    return scaled_worst_result(errors, {'B': _FLOAT32, 'H_iron': _FLOAT32, 'newton_steps': 1.0},
                               measured=values, H_expected=expected_field, newton_iterations=iterations)


def _make_sheet_check(density, flux_of_field, tolerance, step_limit):
    """check of a current-driven sheet: A at the probes against _sheet_reference (relative to max |A|), Newton
    steps <= step_limit, no unconverged loop."""
    def check(results_by_name):
        results = results_by_name[_RUN]
        expected = np.array(_sheet_reference(density, flux_of_field, _PROBE_X))
        measured = _probe_values(results)
        record = (results.summary().get('solvers') or {}).get(_MAGNETIC, {}).get('counters') or {}
        iterations = record.get('nonlinear_iterations')
        converged = record.get('nonlinear_unconverged', 1) == 0
        errors = {'A_probes': float(np.max(np.abs(measured - expected)) / np.max(np.abs(expected))),
                  'newton': 0.0 if converged and iterations is not None and iterations <= step_limit else math.inf}
        return scaled_worst_result(errors, {'A_probes': tolerance, 'newton': 1.0},
                                   A_measured=measured.tolist(), A_expected=expected.tolist(), counters=record)
    return check


def _check_vs_tensor(results_by_name):
    """B_gap of the linear C-core: triangles vs the graded tensor grid (relative)."""
    values = {name: _derived_value(results, 'B_gap') for name, results in results_by_name.items()}
    errors = {'B_gap': abs(values['triangles'] / values['tensor'] - 1.0)}
    return scaled_worst_result(errors, {'B_gap': _VS_TENSOR_TOLERANCE}, B_gap=values)


# ---- finite-difference tangent check (CPU, scripted) ----
def _tangent_materials(mesh, curve_kind):
    """(reluctivity (n, 2), remanence (n, 2), curves, curve_index, disc mask) of the check: the test iron's J_s curve
    or table in the disc, linear anisotropic mu and remanence outside."""
    from ...fem.selftests_mesher import _DISC
    from .saturation import bh_curve, saturation_curve_terms, tabulated_curve_terms
    cx, cy, radius = _DISC
    centroids = mesh.points[mesh.triangles].mean(axis=1)
    disc = np.hypot(centroids[:, 0] - cx, centroids[:, 1] - cy) < radius
    permeability = np.where(disc[:, None], _IRON_PERMEABILITY, np.array(_OUTER_PERMEABILITY)[None, :])
    remanence = np.where(disc[:, None], 0.0, np.array(_OUTER_REMANENCE)[None, :])
    mu = permeability[:, 0]
    polarization = np.full(mesh.n_elements, _IRON_SATURATION)
    table = bh_curve(_table_rows())

    def curve(magnitude, elements):
        if curve_kind == 'curve':
            return saturation_curve_terms(magnitude, mu[elements][:, None], polarization[elements][:, None], np)
        return tabulated_curve_terms(magnitude, table, np)
    return 1.0 / (MU0 * permeability), remanence, (curve,), np.where(disc, 0, -1), disc


def _test_potential(points):
    """B0 x (0.5 + 0.3 sin(2 pi y / L) + 0.2 cos(2 pi x / L)) at dof points (0 on the axis x = 0), |B| ~ B0."""
    scale = 2.0e-6
    x, y = points[:, 0], points[:, 1]
    return _TANGENT_FIELD * x * (0.5 + 0.3 * np.sin(2.0 * math.pi * y / scale) + 0.2 * np.cos(2.0 * math.pi * x / scale))


def _derivative_errors(system, rng):
    """(relative error of R . d against the central difference of W, of K_t d against that of R) at the test
    potential along a random direction."""
    potential = _test_potential(system.dofs.dof_points())
    direction = rng.standard_normal(potential.size) * float(np.abs(potential).max())
    step = _TANGENT_STEP
    energy, residual, tangent = system.weak_form(potential)
    energy_plus, residual_plus, _ = system.weak_form(potential + step * direction, tangent=False)
    energy_minus, residual_minus, _ = system.weak_form(potential - step * direction, tangent=False)
    slope = float(residual @ direction)
    slope_difference = (energy_plus - energy_minus) / (2.0 * step)
    product = tangent @ direction
    product_difference = (residual_plus - residual_minus) / (2.0 * step)
    return (abs(slope_difference - slope) / abs(slope),
            float(np.linalg.norm(product_difference - product) / np.linalg.norm(product)))


def _newton_run(mesh):
    """Scripted P2 planar Newton on the disc (J_s iron, J inside, A = 0 faces): (converged, steps, relative residuals,
    energies)."""
    from .system_fem import MagneticFEMSystem
    reluctivity, _, curves, curve_index, disc = _tangent_materials(mesh, 'curve')
    system = MagneticFEMSystem(mesh, 2, False, lambda: 'magnet_fem_tangent_check newton')
    system.set_materials(reluctivity, None, curves, curve_index)
    converged = system.solve(np.where(disc, _NEWTON_CURRENT_DENSITY, 0.0), 1e-10, 50)
    return converged, system.last_iterations, list(system.last_residuals), list(system.last_energies)


def _check_tangent(_results_by_name):
    """Finite differences of W and R against R . d and K_t d for P1 / P2, planar / axisymmetric, J_s curve / table;
    the scripted Newton run converges in <= _NEWTON_MAX_STEPS with decreasing W and a quadratic tail. error = the
    largest derivative error over its tolerance + 1 per failed Newton sub-check; tolerance 1."""
    from ...fem.selftests_mesher import disc_mesh
    from .system_fem import MagneticFEMSystem
    mesh = disc_mesh(_TANGENT_ELEMENT)
    rng = np.random.default_rng(20261022)
    errors = {}
    for order in (1, 2):
        for axisymmetric in (False, True):
            for curve_kind in ('curve', 'table'):
                reluctivity, remanence, curves, curve_index, _ = _tangent_materials(mesh, curve_kind)
                system = MagneticFEMSystem(mesh, order, axisymmetric, lambda: 'magnet_fem_tangent_check')
                system.set_materials(reluctivity, remanence, curves, curve_index)
                key = f"P{order}_{'axisymmetric' if axisymmetric else 'planar'}_{curve_kind}"
                errors[f'{key}_residual'], errors[f'{key}_tangent'] = _derivative_errors(system, rng)
    converged, steps, residuals, energies = _newton_run(mesh)
    rises = [later - earlier for earlier, later in zip(energies[:-1], energies[1:])
             if later > earlier + 1e-12 * abs(earlier)]
    orders = [math.log(later) / math.log(earlier) for earlier, later in zip(residuals[:-1], residuals[1:])
              if earlier < 1e-2 and 1e-12 < later < 1.0]
    newton = {'converged': converged, 'steps': steps, 'residuals': residuals, 'energy_rises': rises,
              'observed_orders': orders}
    failures = (int(not converged) + int(steps > _NEWTON_MAX_STEPS) + int(bool(rises))
                + int(bool(orders) and orders[-1] < _QUADRATIC_ORDER))
    worst = max(errors.values()) / _TANGENT_TOLERANCE
    error = worst + failures
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'derivative_errors': errors, 'tolerance': _TANGENT_TOLERANCE, 'newton': newton}}


FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='magnet_fem_tangent_check', label='Magnetostatics on triangles: residual and tangent by finite differences',
        description=(f'CPU, scripted on the fem disc mesh (h {_TANGENT_ELEMENT * 1e6:g} um, curved disc edges): the '
                     'test iron (mu_dc 1000, J_s 1 T curve or its bh_curve table) in the disc, linear anisotropic mu '
                     f'{_OUTER_PERMEABILITY} with remanence {_OUTER_REMANENCE} T outside; at a smooth test potential '
                     f'(|B| up to ~{_TANGENT_FIELD:g} T) and a random direction d, central differences of W (step '
                     f'{_TANGENT_STEP:g} max |a|) against R . d and of R against K_t d, for P1 / P2, planar / '
                     'axisymmetric, curve / table (relative, tolerance 1e-6 each). Plus a scripted P2 planar Newton '
                     f'solve (J = {_NEWTON_CURRENT_DENSITY:g} A/m^2 in the disc, A = 0 faces): converged in <= '
                     f'{_NEWTON_MAX_STEPS} steps, W never rising, the last residual ratio log r_(k+1) / log r_k >= '
                     f'{_QUADRATIC_ORDER:g}. error = the largest derivative error / 1e-6 + 1 per failed Newton check; '
                     'tolerance 1. Not run yet.'),
        specs=lambda: {}, check=_check_tangent),
    SelfTestCase(
        name='magnet_fem_current_sheet', label='Magnetostatics on triangles: planar current sheet',
        description=('P2 on uniform 10 um triangles of the 1 x 0.1 mm layout: J = 1e6 A/m^2 in the band [0.4, 0.6] mm '
                     '(its outline edges are mesh lines) between insulation x faces, symmetry y faces. The continuous '
                     'piecewise-quadratic A lies in the P2 space: A at three probe nodes (float32, 1e-6), the energy '
                     '(float64, 1e-9) and the band mean of (x - x_c) B_y (1e-6; B_y is odd about the band centre x_c, '
                     'so its plain mean is 0) against the closed form. error = the largest error over its tolerance; '
                     'tolerance 1. Variant gpu: the same on the gpu backend (the solve on the run backend, same '
                     'tolerance).'),
        specs=_current_sheet_specs, check=_check_current_sheet,
        variants=(SelfTestVariant('gpu', 1.0, backend='gpu'),)),
    SelfTestCase(
        name='magnet_fem_uniform_field_axisymmetric', label='Magnetostatics on triangles: uniform axial field (r, z)',
        description=('Axisymmetric empty 0.5 x 0.2 mm box in an applied axial B = 1e-3 T (external_field: A_phi = B r / '
                     '2 on the window-edge nodes), P1 and P2: A_phi = B r / 2 is linear, so B_z = B exactly in every '
                     'element including the axis ones (max / min samples, 1e-6) and B_r = 0; energy B^2 / (2 mu0) pi R^2 '
                     'Z (1e-9). Pins the A_phi formulation (decision 1). error = the largest error over its tolerance; '
                     'tolerance 1. Not run yet.'),
        specs=_uniform_field_specs, check=_check_uniform_field),
    SelfTestCase(
        name='magnet_fem_coil_inductance', label='Magnetostatics on triangles: go / return coil pair',
        description=('P2: coils go [0.2, 0.3] mm (+0.1 A) and back [0.7, 0.8] mm (-0.1 A), 10 turns each, insulation x '
                     'faces, symmetry y faces: the continuous piecewise-quadratic A is in the P2 space, so the energy, '
                     'L_go and L_back equal the closed form (1e-9) and L_go + L_back = 2 W / I^2 (1e-9). error = the '
                     'largest relative error over 1e-9; tolerance 1. Not run yet.'),
        specs=_coil_pair_specs, check=_check_coil_pair),
    SelfTestCase(
        name='magnet_fem_remanent_slab', label='Magnetostatics on triangles: remanent slab between insulating faces',
        description=(f'P1 and P2 on uniform 10 um triangles of the 1 x 0.1 mm layout: remanence_y = {_REMANENCE:g} T '
                     f'with mu_dc {_MAGNET_PERMEABILITY:g} in the band [0.4, 0.6] mm, insulation x faces, symmetry y '
                     'faces. Without currents H_y = H0 is uniform (H0 = -B_r t / (mu0 (mu t + L - t)) from A = 0 on both '
                     'faces), B_y = mu0 mu H0 + B_r in the band and A is piecewise linear, so both spaces hold it: A at '
                     'the three probe nodes, H_y max / min over the window and B_y at the band centre (float32, 1e-6), '
                     'the energy int B^2 / (2 mu0 mu) (float64, 1e-9); the closed forms take the float32 mu_dc the '
                     'MaterialGrid holds (1.05 as float64 left W 5.5e-8 off). Pins the remanence load int Q^T nu B_r '
                     'dV. error = the largest error over its tolerance; tolerance 1.'),
        specs=_remanent_slab_specs, check=_check_remanent_slab),
    SelfTestCase(
        name='magnet_fem_saturating_slab', label='Magnetostatics on triangles: flux-driven saturating slab',
        description=(f'Iron (mu_dc {_IRON_PERMEABILITY:g}, J_s {_IRON_SATURATION:g} T) band across the full y span in an '
                     f'applied B_x = {_SATURATING_FIELD:g} T: A = B_x y is linear and in the space, so B_x = 0.8 T '
                     'everywhere (max / min, 1e-6) and H_x at the band centre = H(0.8 T) of the curve (bisection, '
                     '1e-6; not a band mean: the nodal H averages smear across the iron / air outline); Newton '
                     f"(nonlinear_method auto) in <= {_NEWTON_LIMITS['slab']} steps. error = the largest error over "
                     'its tolerance; tolerance 1.'),
        specs=_saturating_slab_specs, check=_check_saturating_slab),
    SelfTestCase(
        name='magnet_fem_deep_saturation', label='Magnetostatics on triangles: deeply saturated current sheet',
        description=(f'J = {_DEEP_CURRENT_DENSITY:g} A/m^2 in the band, J_s iron everywhere, insulation x_lo, symmetry '
                     'x_hi and y faces, P2 on 10 um triangles (H to 2e4 A/m, s = d ln B / d ln H ~ 0.03, far below the '
                     '~0.12 the tensor Anderson loop covers, which is not expected to converge here): H_y(x) = '
                     '-(current right of x) by Ampere, A = -int B(H_y) dx by composite Gauss-Legendre; A at three '
                     f"probe nodes (relative to max |A|, {_SHEET_TOLERANCES['curve']:g}) and Newton converged in <= "
                     f"{_NEWTON_LIMITS['deep']} steps (the energy line search keeps W decreasing). error = the largest "
                     'error over its tolerance; tolerance 1. Not run yet.'),
        specs=lambda: _sheet_specs(_DEEP_CURRENT_DENSITY, _iron(['all'])),
        check=_make_sheet_check(_DEEP_CURRENT_DENSITY, _curve_flux_density, _SHEET_TOLERANCES['curve'],
                                _NEWTON_LIMITS['deep'])),
    SelfTestCase(
        name='magnet_fem_tabulated_bh_sheet', label='Magnetostatics on triangles: current sheet in a tabulated B-H curve',
        description=(f'The {_SATURATING_CURRENT_DENSITY:g} A/m^2 sheet in the bh_curve table of the test iron (H past '
                     'the last row: the mu0 extension), B-indexed curve with kinks at the rows: A at the probes against '
                     f"the forward table reference ({_SHEET_TOLERANCES['table']:g}: P2 across the kinks, a first "
                     f"estimate) and Newton converged in <= {_NEWTON_LIMITS['sheet']} steps (backtracks in the "
                     'details). error = the largest error over its tolerance; tolerance 1. Not run yet.'),
        specs=lambda: _sheet_specs(_SATURATING_CURRENT_DENSITY, _table_iron()),
        check=_make_sheet_check(_SATURATING_CURRENT_DENSITY, _table_flux_density, _SHEET_TOLERANCES['table'],
                                _NEWTON_LIMITS['sheet'])),
    SelfTestCase(
        name='magnet_fem_vs_tensor', label='Magnetostatics: iron C-core on triangles vs rectangular cells',
        description=('The iron_c_core_saturation geometry with linear iron (mu_dc 4000) at I_coil = 1 A: B_gap (mean '
                     '|B| in the gap) on triangles (P2, 3 mm, 0.4 mm in the gap) against the graded tensor grid (1 mm, '
                     f'0.25 mm in the gap), relative {_VS_TENSOR_TOLERANCE:g} (first estimate). error = the relative '
                     'difference over its tolerance; tolerance 1. Not run yet.'),
        specs=_vs_tensor_specs, check=_check_vs_tensor),
)

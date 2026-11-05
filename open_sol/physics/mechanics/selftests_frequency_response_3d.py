"""Frequency-response self-tests of solid_mechanics_3d on tetrahedra (piezo_3d_plan stage 4): the lossy z-rotated lithium
niobate thickness bar driven by a voltage against the continuum admittance with its BVD fit against the short / open
thickness modes, an elastic forced run against the continuum response, and the cuDSS complex LDL^T of the harmonic
pencil against host SuperLU with plan reuse (GPU, no model). definition3d registers them."""
import math

import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, lithium_niobate_entry, scaled_worst_result
from ...model.spec import FeatureSpec, MaterialSpec, StudySpec
from .selftests import _lame
from .selftests_fem3d import _MODES_CELL, _MODES_DENSITY, _MODES_ELASTIC, _MODES_SIZE, _continuum_modes, _solid_spec
from .selftests_piezo_fem3d import (_BAR_EULER, _PIEZO, _THICKNESS_CELL, _THICKNESS_SIZE, _electrode,
                                    _thickness_references)

_SOLID = 'solid'
_STUDY = 'frequency_response'          # = model.study_frequency_response.STUDY_NAME

# ---- solid_fem3d_frequency_response_bar: the thickness bar of solid_fem3d_piezo_thickness_modes with losses ----
# Rollers on the four sides and z_lo, ground z_lo, drive z_hi at V. With c' = c33 (1 + j eta), eps' = eps0 eps33 (1 - j
# tan delta), c^D' = c' + e33^2 / eps' and k = omega sqrt(rho / c^D'): u = B sin(k z), D uniform, T_zz(L) = 0 and V =
# (e33 u(L) - D L) / eps' give B = -e33 V / (L (c^D' k cos kL - e33^2 sin kL / (eps' L))), D = (e33 B sin kL - eps' V) /
# L, Q_drive = -D A and Y = j omega Q / V (the continuum; P2 on 62.5 nm differs by far less than the tolerance). The BVD
# fit of the sweep against the lossless short / open mode 1 (the thickness-mode case's roots). elastic: piezoelectric off,
# traction t on z_hi at 1.6 x the lowest c^E mode: u_z = t sin(k z) / (c' k cos kL), k = omega sqrt(rho / c'), on a cut.
_LOSS_FACTOR = float32_value(2e-3)     # float32-exact material values
_LOSS_TANGENT = float32_value(1e-3)
_DRIVE = 1.0                           # V
_N_SWEEP = 93                          # spacing ~5.5e-4 f: ~3.6 points per linewidth eta f (one SuperLU each, ~0.17 s)
_BAND_MARGIN = 0.02                    # x f_short (10 linewidths) below the resonance and above the antiresonance
                                       # (was 201 points / 0.05: BVD k_eff^2 error 1e-4, 35 s of factorisations)
_TRACTION = 1e6                        # Pa on z_hi (elastic run)
_ELASTIC_FACTOR = 1.6                  # x the lowest elastic thickness mode (between modes 1 and 2)
_ADMITTANCE_TOLERANCE = 1e-3           # |dY| / max(|Y|, omega C0) per frequency (continuum reference; first estimate)
_FIELD_TOLERANCE = 1e-3                # elastic cut: relative to the peak |u_z|; P2 between nodes ~(k h)^3 / 16 ~ 2e-4
_BVD_FREQUENCY_TOLERANCE = 5e-3        # as the 2D bar: a single-mode BVD of a distributed resonator
_BVD_COUPLING_TOLERANCE = 5e-2
_IDENTITY_TOLERANCE = 1e-12            # kappa_e = load_impedance / L1 of the same result
_BAR_RUNS = ('piezo', 'elastic')


def _sweep_band():
    """(start, stop) [Hz] around the lossless short and open mode 1 of the bar."""
    expected, _ = _thickness_references()
    short, open_circuit = expected['short'][0], expected['open'][0]
    return (1.0 - _BAND_MARGIN) * short, open_circuit + _BAND_MARGIN * short


def _elastic_frequency(constants):
    """The forced frequency [Hz] of the elastic run: _ELASTIC_FACTOR x v_E / (4 L)."""
    return _ELASTIC_FACTOR * math.sqrt(constants['c33'] / constants['rho']) / (4.0 * _THICKNESS_SIZE[2])


def _bar_spec(run):
    features = [FeatureSpec('sides', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                FeatureSpec('anchor', 'roller', ['z_lo'])]
    properties = {'mechanical_loss_factor': _LOSS_FACTOR}
    if run == 'piezo':
        properties['dielectric_loss_tangent'] = _LOSS_TANGENT
        features += [_electrode('ground', 'z_lo'), _electrode('drive', 'z_hi', voltage=_DRIVE),
                     FeatureSpec('admittance', 'admittance')]
        start, stop = _sweep_band()
        study = {'frequency_start': start, 'frequency_stop': stop, 'n_frequencies': _N_SWEEP}
    else:
        features.append(FeatureSpec('push', 'boundary_load', ['z_hi'], {'t_x': 0.0, 't_y': 0.0, 't_z': _TRACTION}))
        study = {'frequencies': [_elastic_frequency(_thickness_references()[1])]}
    return _solid_spec(
        f'Self-test: LN thickness bar driven harmonically on tetrahedra ({run})', _THICKNESS_SIZE,
        [MaterialSpec('crystal', properties=properties, dispersion=[lithium_niobate_entry(_BAR_EULER)],
                      selection=['all'])],
        features, cell=_THICKNESS_CELL, study=StudySpec(type=_STUDY, settings=study),
        settings=_PIEZO if run == 'piezo' else None)


def _continuum_admittance(frequencies, constants):
    """Complex Y [S] of the lossy continuum bar at frequencies [Hz] (any V: it cancels)."""
    width, depth, length = _THICKNESS_SIZE
    omega = 2.0 * math.pi * np.asarray(frequencies, dtype=np.float64)
    stiffness = constants['c33'] * (1.0 + 1j * _LOSS_FACTOR)
    permittivity = EPS0 * constants['eps33'] * (1.0 - 1j * _LOSS_TANGENT)
    piezo_e = constants['e33']
    stiffened = stiffness + piezo_e ** 2 / permittivity
    wavenumber = omega * np.sqrt(constants['rho'] / stiffened)   # the sign of the root cancels in B sin(k z)
    phase = wavenumber * length
    amplitude = -piezo_e / (length * (stiffened * wavenumber * np.cos(phase)
                                      - piezo_e ** 2 * np.sin(phase) / (permittivity * length)))
    flux = (piezo_e * amplitude * np.sin(phase) - permittivity) / length
    return 1j * omega * (-flux * width * depth)


def _cut_coordinates(field_data):
    """{axis name: (nu, nv) centre coordinates [m]} of a FieldData cut."""
    values = np.asarray(field_data.values)
    u0, u1, v0, v1 = field_data.extent
    count_u, count_v = values.shape
    u = (np.asarray(field_data.x_centers, dtype=np.float64) if field_data.x_centers is not None
         else u0 + (np.arange(count_u) + 0.5) * (u1 - u0) / count_u)
    v = (np.asarray(field_data.y_centers, dtype=np.float64) if field_data.y_centers is not None
         else v0 + (np.arange(count_v) + 0.5) * (v1 - v0) / count_v)
    return dict(zip(field_data.axes, np.meshgrid(u, v, indexing='ij')))


def _check_bar(results_by_name):
    """piezo: Y(f) against the continuum, the BVD fit against the lossless short / open mode 1, kappa_e = Z / L1;
    elastic: Re u_z and u_abs of the live field on the y = W / 2 cut against the continuum response."""
    expected, constants = _thickness_references()
    errors, tolerances, details = {}, {}, {'constants': constants}

    def add(item, error, tolerance):
        errors[item], tolerances[item] = error, tolerance

    if 'piezo' in results_by_name:
        piezo = results_by_name['piezo'].result(_SOLID, 'admittance')
        frequencies = np.asarray(piezo['x'], dtype=np.float64)
        measured = np.asarray(piezo['series']['re']) + 1j * np.asarray(piezo['series']['im'])
        reference = _continuum_admittance(frequencies, constants)
        width, depth, length = _THICKNESS_SIZE
        capacitive = 2.0 * math.pi * frequencies * EPS0 * constants['eps33'] * width * depth / length
        add('piezo Y', float(np.max(np.abs(measured - reference) / np.maximum(np.abs(reference), capacitive))),
            _ADMITTANCE_TOLERANCE)
        if frequencies.size != _N_SWEEP:
            add('piezo frequency count', abs(frequencies.size - _N_SWEEP), 0.5)
        short, open_circuit = expected['short'][0], expected['open'][0]
        add('BVD f_s', abs(piezo['f_s'] / short - 1.0), _BVD_FREQUENCY_TOLERANCE)
        add('BVD f_p', abs(piezo['f_p'] / open_circuit - 1.0), _BVD_FREQUENCY_TOLERANCE)
        add('BVD k_eff2', abs(piezo['k_eff2'] / ((open_circuit ** 2 - short ** 2) / open_circuit ** 2) - 1.0),
            _BVD_COUPLING_TOLERANCE)
        add('kappa_e', abs(piezo['kappa_e'] * piezo['L1'] / piezo['load_impedance'] - 1.0), _IDENTITY_TOLERANCE)
        details['piezo'] = {'short_Hz': short, 'open_Hz': open_circuit,
                            'fit': {key: piezo.get(key) for key in ('f_s', 'f_p', 'k_eff2', 'Q_s', 'C0', 'C1', 'L1',
                                                                    'R1', 'tan_delta_0', 'fit_error', 'fit_warning',
                                                                    'kappa_e', 'omega_C0_Z')}}
    if 'elastic' in results_by_name:
        results = results_by_name['elastic']
        frequency = _elastic_frequency(constants)
        omega = 2.0 * math.pi * frequency
        stiffness = constants['c33'] * (1.0 + 1j * _LOSS_FACTOR)
        wavenumber = omega * math.sqrt(constants['rho']) / np.sqrt(stiffness)
        cut = ('y', 0.5 * _THICKNESS_SIZE[1])
        field = results.field(f'{_SOLID}.u_z', plane=cut)
        height = _cut_coordinates(field)['z']
        response = _TRACTION * np.sin(wavenumber * height) / (stiffness * wavenumber
                                                              * np.cos(wavenumber * _THICKNESS_SIZE[2]))
        peak = float(np.max(np.abs(response)))
        for name, values in (('u_z', response.real), ('u_abs', np.abs(response))):
            sampled = np.asarray(results.field(f'{_SOLID}.{name}', plane=cut).values, dtype=np.float64)
            add(f'elastic {name}', float(np.nanmax(np.abs(sampled - values))) / peak, _FIELD_TOLERANCE)
        details['elastic'] = {'frequency_Hz': frequency, 'peak_m': peak}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_harmonic_cudss: cuDSS complex LDL^T of the harmonic pencil against host SuperLU (GPU) ----
# The roller-box pencil of solid_fem3d_modes_cudss on 2x coarser cells with eta K loss, a fixed pseudo-random load on the
# free dofs, five frequencies around the lowest mode through one HarmonicSystem per back end: one cuDSS plan for all five
# (refactorised in place), a symmetric (LDL^T) factor, solutions within relative 1e-9 of SuperLU. On the 62.5 nm cells
# (24237 free dofs) host complex SuperLU took 76 s per frequency (cuDSS 1 s): the coarse box keeps the reference cheap.
_CUDSS_CELL = 2.0 * _MODES_CELL
_CUDSS_LOSS_FACTOR = 1e-3
_CUDSS_FACTORS = (0.5, 0.99, 1.0005, 1.01, 1.5)   # x the lowest continuum roller-box frequency
_CUDSS_TOLERANCE = 1e-9                           # relative max |dx| (both double; first estimate)


def _check_cudss(_):
    """Solutions, plan count and factor kind of the 'cudss' back end against 'scipy_splu' on the same HarmonicSystem
    inputs; passes with a note without nvmath-python."""
    from ...fem.elasticity3d import isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from ...solvers.cudss_factor import cudss_available
    from ...solvers.sparse_direct import SparseDirectSolver
    from .frequency_response import HarmonicSystem
    from .system_fem3d import ElasticFEMSystem3D
    if not cudss_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': 1.0,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    axes = [np.linspace(0.0, side, int(round(side / _CUDSS_CELL)) + 1) for side in _MODES_SIZE]
    mesh = structured_box_mesh(*axes)
    system = ElasticFEMSystem3D(mesh, 2, lambda: 'cuDSS harmonic self-test')
    for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi'):
        system.hold(system.full.face_dofs(system.face_faces(face)), **{f'u_{face[0]}': 0.0})
    lame_lambda, mu = _lame(*_MODES_ELASTIC)
    count = mesh.n_elements
    solid = np.ones(count, dtype=bool)
    stiffness = isotropic_matrix_3d(np.full(count, lame_lambda), np.full(count, mu))
    system.assemble(stiffness, solid)
    loss = _CUDSS_LOSS_FACTOR * system.operator
    mass = system.mass(np.full(count, _MODES_DENSITY))
    load = np.where(system.free, np.random.default_rng(6).standard_normal(system.free.size), 0.0)
    fixed = np.zeros(system.free.size)
    lowest = _continuum_modes()[0]
    solutions, details = {}, {'free_dofs': int(system.free.sum()), 'lowest_Hz': lowest}
    runtimes = {}
    for back_end in ('scipy_splu', 'cudss'):
        runtime = SparseDirectSolver(back_end=back_end, backend='gpu')
        harmonic = HarmonicSystem(system.operator, loss, mass, system.free, runtime=runtime)
        solutions[back_end] = [harmonic.solve(2.0 * math.pi * factor * lowest, fixed, load)[0]
                               for factor in _CUDSS_FACTORS]
        runtimes[back_end] = runtime
        factor = harmonic.factor
        details[back_end] = {'counters': runtime.work_counters(), 'factor': getattr(factor, 'description', 'SuperLU'),
                             **{name: getattr(factor, name) for name in ('symmetric', 'asymmetry', 'device_bytes',
                                                                         'plan_seconds', 'factorize_seconds',
                                                                         'probe_residual', 'refinement_steps')
                                if hasattr(factor, name)}}
        if hasattr(factor, 'release'):
            factor.release()
    errors = {f'x at {factor:g} f_1': float(np.max(np.abs(gpu - cpu)) / np.max(np.abs(cpu)))
              for factor, cpu, gpu in zip(_CUDSS_FACTORS, solutions['scipy_splu'], solutions['cudss'])}
    tolerances = dict.fromkeys(errors, _CUDSS_TOLERANCE)
    errors['cuDSS plans'], tolerances['cuDSS plans'] = abs(runtimes['cudss'].n_cudss_plans - 1), 0.5
    errors['LDL^T'], tolerances['LDL^T'] = 0.0 if details['cudss'].get('symmetric') else 1.0, 0.5
    return scaled_worst_result(errors, tolerances, **details)


FREQUENCY_RESPONSE_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_frequency_response_bar',
        label='Solid mechanics 3D on tetrahedra: harmonically driven piezoelectric bar, admittance',
        description=('frequency_response runs on quadratic builtin tetrahedra of the solid_fem3d_piezo_thickness_modes '
                     f'bar ({_THICKNESS_SIZE[0] * 1e6:g} x {_THICKNESS_SIZE[1] * 1e6:g} x {_THICKNESS_SIZE[2] * 1e6:g} '
                     'um, z-rotated lithium niobate, rollers on the four sides and z_lo) with mechanical_loss_factor '
                     f'{_LOSS_FACTOR:g}. piezo: dielectric_loss_tangent {_LOSS_TANGENT:g}, ground z_lo, drive z_hi at '
                     f'{_DRIVE:g} V, an admittance feature, {_N_SWEEP} frequencies from {1.0 - _BAND_MARGIN:g} f_short '
                     f'to f_open + {_BAND_MARGIN:g} f_short: Y(f) against the lossy continuum thickness-mode admittance '
                     f'(|dY| / max(|Y|, omega C0) within {_ADMITTANCE_TOLERANCE:g}), the BVD f_s / f_p against the '
                     f'lossless short / open mode 1 ({_BVD_FREQUENCY_TOLERANCE:g}), k_eff2 '
                     f'({_BVD_COUPLING_TOLERANCE:g}), kappa_e = load_impedance / L1. elastic (piezoelectric off): a '
                     f'{_TRACTION:g} Pa traction on z_hi at {_ELASTIC_FACTOR:g} x the lowest thickness mode: Re u_z and '
                     f'u_abs on the y = W / 2 cut against the continuum within {_FIELD_TOLERANCE:g} of the peak. '
                     'measured = error = the worst deviation in tolerance units, tolerance 1 (first estimates, not yet '
                     'run). Pins the 3D harmonic pencil (losses, coupled mass), the electrode charge, the 3D admittance '
                     'units and the harmonic outputs.'),
        specs=lambda: {run: _bar_spec(run) for run in _BAR_RUNS}, check=_check_bar),
    SelfTestCase(
        name='solid_fem3d_harmonic_cudss',
        label='Solid mechanics 3D on tetrahedra: cuDSS harmonic factor vs SuperLU',
        description=('The solid_fem3d_modes_cudss roller-box pencil (P2, built directly, no model run) on '
                     f'{_CUDSS_CELL * 1e9:g} nm Kuhn cubes with loss {_CUDSS_LOSS_FACTOR:g} K and a fixed pseudo-random '
                     'load, solved by one '
                     'frequency_response.HarmonicSystem per back end at '
                     f"{', '.join(f'{factor:g}' for factor in _CUDSS_FACTORS)} x the lowest continuum mode: the "
                     "sparse_direct 'cudss' runtime (complex symmetric LDL^T) against 'scipy_splu' (relative max |dx| "
                     f'within {_CUDSS_TOLERANCE:g}), one cuDSS plan for all frequencies (refactorised in place) and a '
                     'symmetric factor. details record the counters, plan / factorise seconds and device bytes. GPU '
                     'only; passes with a note without nvmath-python (62.5 nm cells: worst 5.9e-12 at 1.0005 f_1).'),
        specs=dict, check=_check_cudss, backends=('gpu',)),
)

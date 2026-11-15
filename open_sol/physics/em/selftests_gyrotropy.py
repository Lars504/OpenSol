"""Model-run self-tests of the gyrotropic (magneto-optic) permittivity: em_fdfd_hz gyration_z (non-reciprocal point-source
pairs through a lossless block, reciprocity with the magnetisation reversed, the flux balance around the block; the
non-reciprocal phase shift of a magneto-optic-clad slab guide from Bloch eigenfrequencies) and the em_modes first-order
gyration_x shift of the same guide, both against the exact slab dispersion; no physics imports."""
import math

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import probe_phasor, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)
from .selftests_fdfd import cpu_specs

_HZ = 'em_fdfd_hz'
_MODES_TYPE = 'em_modes'
_WAVELENGTH = 1.55e-6   # m


def exact_slab_beta(k0, eps_substrate, eps_core, eps_cladding, gyration, thickness, direction=1):
    """Propagation constant [rad/m] of the TM0 (H_z) mode of an in-plane em_fdfd_hz slab guide along x: substrate (y <
    core), core of `thickness`, cladding (y > core) with gyration g (eps_xy = j g, eps_yx = -j g), direction +1 (fields
    ~ exp(-j beta x)) or -1 (exp(+j beta x)). H_z and (eps H' + direction beta g H) / det are continuous, so kappa d =
    atan(p_s) + atan(p_c), p_s = eps_core gamma_s / (eps_substrate kappa), p_c = eps_core (eps_c gamma_c - direction
    beta g) / (det_c kappa), det_c = eps_c^2 - g^2, gamma_c^2 = beta^2 - k0^2 det_c / eps_c (scipy brentq)."""
    from scipy.optimize import brentq
    det_cladding = eps_cladding * eps_cladding - gyration * gyration

    def mismatch(beta):
        kappa = math.sqrt(k0 * k0 * eps_core - beta * beta)
        gamma_s = math.sqrt(beta * beta - k0 * k0 * eps_substrate)
        gamma_c = math.sqrt(beta * beta - k0 * k0 * det_cladding / eps_cladding)
        p_s = eps_core * gamma_s / (eps_substrate * kappa)
        p_c = eps_core * (eps_cladding * gamma_c - direction * beta * gyration) / (det_cladding * kappa)
        return kappa * thickness - math.atan(p_s) - math.atan(p_c)

    low = k0 * math.sqrt(max(eps_substrate, det_cladding / eps_cladding)) * (1.0 + 1e-12)
    high = k0 * math.sqrt(eps_core) * (1.0 - 1e-12)
    return brentq(mismatch, low, high, xtol=1e-15 * high, rtol=1e-15)


def exact_slab_omega(beta, eps_substrate, eps_core, eps_cladding, gyration, thickness, direction, omega_guess):
    """Angular frequency [rad/s] at which exact_slab_beta(direction) equals beta (scipy brentq within +-20 %)."""
    from scipy.optimize import brentq
    return brentq(lambda omega: exact_slab_beta(omega / C0, eps_substrate, eps_core, eps_cladding, gyration,
                                                thickness, direction) - beta,
                  0.8 * omega_guess, 1.2 * omega_guess, xtol=1e-14 * omega_guess)


# ---- em_fdfd_hz_gyrotropic_reciprocity: the em_fdfd_hz_anisotropic_reciprocity box (2 x 2 um, 25 nm cells, CPML)
# with a lossless isotropic block (eps 4) of gyration g = +-0.5 between A and B. from_a radiates a magnetic line current
# at A and probes H_z at B, from_b the reverse. The operator of -g is the transpose of that of +g (times the cell
# areas), so H(B; A, +g) = H(A; B, -g) to solver round-off, while H(B; A, +g) != H(A; B, +g); a closed contour of four
# flux lines around the block (from_a, +g) has zero net outflow: Hermitian eps, discretely exact Poynting balance.
_PAIR_CELL = 25e-9
_PAIR_SIZE = 2.0e-6
_PAIR_BLOCK = (0.6e-6, 0.7e-6, 1.4e-6, 1.2e-6)
_PAIR_POINTS = {'a': (0.4125e-6, 0.5125e-6), 'b': (1.5625e-6, 1.4625e-6)}
_PAIR_EPS = 4.0
_PAIR_GYRATION = 0.5
_PAIR_WAVELENGTH = 1e-6
_PAIR_TOLERANCE = 1e-8          # relative |H(B; A, +g) - H(A; B, -g)| (direct solve round-off)
_PAIR_MIN_ASYMMETRY = 1e-2      # |H(B; A, +g) - H(A; B, +g)| / |H(B; A, +g)| must exceed this (non-reciprocal)
# flux contour: a line measures on the hi faces of its cells, so the x lines sit in cell columns 22 / 58 (faces 23 /
# 59) spanning cell rows 27..50, the y lines in rows 26 / 50 (faces 27 / 51) spanning columns 23..58 (coordinates at
# cell centres): a closed ring of faces around the block, 1 - 3 cells clear of it
_FLUX_CELLS = {'flux_x0': ((22.5, 27.5), (22.5, 50.5)), 'flux_x1': ((58.5, 27.5), (58.5, 50.5)),
               'flux_y0': ((23.5, 26.5), (58.5, 26.5)), 'flux_y1': ((23.5, 50.5), (58.5, 50.5))}
_FLUX_TOLERANCE = 1e-9          # |net outflow| / sum |line fluxes|
_FLUX_SIGNS = {'flux_x0': -1.0, 'flux_x1': 1.0, 'flux_y0': -1.0, 'flux_y1': 1.0}   # outward = +x / +y on the hi sides


def _pair_spec(gyration):
    """The block of gyration g with physics from_a (+ the flux contour when g > 0) and from_b."""
    x0, y0, x1, y1 = _PAIR_BLOCK
    geometry = [GeometrySpec('block', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})]
    geometry += [GeometrySpec(f'point_{name}', 'point', {'x': x, 'y': y}) for name, (x, y) in _PAIR_POINTS.items()]
    geometry += [GeometrySpec(name, 'line', {'x0': start[0] * _PAIR_CELL, 'y0': start[1] * _PAIR_CELL,
                                             'x1': end[0] * _PAIR_CELL, 'y1': end[1] * _PAIR_CELL})
                 for name, (start, end) in _FLUX_CELLS.items()]
    physics = []
    for source, target in (('a', 'b'), ('b', 'a')):
        features = [FeatureSpec('source', 'point_source', [f'point_{source}'],
                                {'excitation': 'magnetic_current', 'magnetic_current': 1.0, 'phase': 0.0}),
                    FeatureSpec('probe', 'field_probe', [f'point_{target}'], {'component': 'hz'})]
        if source == 'a' and gyration > 0.0:
            features += [FeatureSpec(name, 'flux_monitor', [name]) for name in _FLUX_CELLS]
        physics.append(PhysicsSpec(f'from_{source}', _HZ, settings={'wavelength': _PAIR_WAVELENGTH},
                                   features=features))
    return ModelSpec(
        title=f'{_HZ} non-reciprocity through a gyrotropic block (gyration_z {gyration:+g})',
        domain=DomainSpec(size_x=_PAIR_SIZE, size_y=_PAIR_SIZE),
        geometry=geometry,
        materials=[MaterialSpec('garnet', properties={'eps_r': _PAIR_EPS, 'gyration_z': gyration},
                                selection=['block'])],
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_PAIR_CELL),
        study=StudySpec(type='stationary'))


def _pair_specs():
    return cpu_specs({'plus': _pair_spec(_PAIR_GYRATION), 'minus': _pair_spec(-_PAIR_GYRATION)})


def _check_pair(results_by_name):
    """Reciprocity with g reversed (error), the asymmetry at fixed g (> floor), the net flux out of the block."""
    plus, minus = results_by_name['plus'], results_by_name['minus']
    forward = probe_phasor(plus, 'from_a', 'probe')
    backward = probe_phasor(plus, 'from_b', 'probe')
    reversed_backward = probe_phasor(minus, 'from_b', 'probe')
    reversed_forward = probe_phasor(minus, 'from_a', 'probe')
    asymmetry = abs(forward - backward) / abs(forward)
    errors = {'h_b_from_a_vs_reversed': abs(forward - reversed_backward) / abs(forward),
              'h_a_from_b_vs_reversed': abs(backward - reversed_forward) / abs(backward),
              'non_reciprocity': 0.0 if asymmetry > _PAIR_MIN_ASYMMETRY else math.inf}
    fluxes = {name: float(plus.result('from_a', name)['power'][0]) for name in _FLUX_CELLS}
    net = sum(_FLUX_SIGNS[name] * value for name, value in fluxes.items())
    errors['flux_balance'] = abs(net) / sum(abs(value) for value in fluxes.values())
    tolerances = {'h_b_from_a_vs_reversed': _PAIR_TOLERANCE, 'h_a_from_b_vs_reversed': _PAIR_TOLERANCE,
                  'non_reciprocity': 1.0, 'flux_balance': _FLUX_TOLERANCE}
    return scaled_worst_result(errors, tolerances, h_b_from_a=[forward.real, forward.imag],
                               h_a_from_b=[backward.real, backward.imag],
                               h_a_from_b_reversed=[reversed_backward.real, reversed_backward.imag],
                               relative_asymmetry=asymmetry, min_asymmetry=_PAIR_MIN_ASYMMETRY, line_fluxes=fluxes,
                               net_outflow=net)


# ---- the magneto-optic-clad slab guide (shared): silica substrate, silicon core 400 nm, cladding eps 4.84 (n 2.2,
# Ce:YIG-like) with an exaggerated gyration so the shift is large against the discretisation error. TM0 (H_z in plane,
# E along y in em_modes) is guided (n_eff ~ 3.05 > 2.2). The em_modes frame (x, y, z) is the in-plane frame (z, y, -x)
# turned (a proper rotation): the in-plane gyration_z g is the cross-section gyration_x -g.
_EPS_SUBSTRATE, _EPS_CORE, _EPS_CLADDING = 2.085, 12.11, 4.84
_CORE_THICKNESS = 0.4e-6
_GUIDE_GYRATION = 0.2
_GUIDE_HEIGHT = 2.0e-6            # along y, CPML beyond; core centre at its middle


def _guide_geometry(size_x):
    """Core rectangle and cladding rectangle (core top to the domain top) across the whole x extent."""
    centre = 0.5 * _GUIDE_HEIGHT
    top = centre + 0.5 * _CORE_THICKNESS
    return [GeometrySpec('core', 'rectangle', {'x0': 0.0, 'y0': centre - 0.5 * _CORE_THICKNESS, 'x1': size_x,
                                               'y1': top}),
            GeometrySpec('cladding', 'rectangle', {'x0': 0.0, 'y0': top, 'x1': size_x, 'y1': _GUIDE_HEIGHT})]


def _guide_materials(gyration_name, gyration):
    return [MaterialSpec('substrate', properties={'eps_r': _EPS_SUBSTRATE}, selection=['all']),
            MaterialSpec('core', properties={'eps_r': _EPS_CORE}, selection=['core']),
            MaterialSpec('cladding', properties={'eps_r': _EPS_CLADDING, gyration_name: gyration},
                         selection=['cladding'])]


# ---- em_fdfd_hz_gyrotropic_nrps: the guide in the plane, x periodic over 2 cells with Bloch kx = +-beta0 (beta0 =
# 3.05 k0 at 1550 nm), CPML along y, an eigenfrequency study near 1550 nm, at 10 and 5 nm cells. The guided TM0
# resonance of each sign is omega(+-beta0): the gyration shifts them apart (the non-reciprocal phase shift seen at fixed
# beta), compared with the exact slab dispersion solved for omega at beta0 in each direction. omega converges at second
# order, the split at first order (the off-diagonal face couplings take plain cell means across the core / cladding
# interface: ~6 % at 10 nm, ~3 % at 5 nm), so the split is checked Richardson-extrapolated, 2 s(5 nm) - s(10 nm).
_BLOCH_CELLS = 2
_BLOCH_RESOLUTIONS = {'fine': 5e-9, 'coarse': 10e-9}
_BETA_INDEX = 3.05                 # beta0 / k0 at 1550 nm
_NRPS_OMEGA_TOLERANCE = 1e-3       # relative error of omega(+-beta0) at 10 nm (Yee dispersion and interfaces)
_NRPS_SPLIT_TOLERANCE = 1e-2       # relative error of the extrapolated split omega(-beta0) - omega(+beta0)


def _bloch_guide_spec(direction, cell):
    """In-plane guide at `cell` [m], gyration_z g on the cladding, Bloch kx = direction beta0, an eigenfrequency
    study."""
    size_x = _BLOCH_CELLS * cell
    beta = _BETA_INDEX * 2.0 * math.pi / _WAVELENGTH
    settings = {'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic', 'bloch_kx': direction * beta,
                'wavelength': _WAVELENGTH}
    return ModelSpec(
        title=f"{_HZ} magneto-optic slab guide, Bloch kx {'+' if direction > 0 else '-'}beta0, {cell * 1e9:g} nm cells",
        domain=DomainSpec(size_x=size_x, size_y=_GUIDE_HEIGHT),
        geometry=_guide_geometry(size_x),
        materials=_guide_materials('gyration_z', _GUIDE_GYRATION),
        physics=[PhysicsSpec('em', _HZ, settings=settings, features=[FeatureSpec('table', 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=cell),
        solver_defaults=SolverDefaultsSpec(subpixel_smoothing=True),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}))


def _bloch_guide_specs():
    return cpu_specs({f'{direction_name}_{level}': _bloch_guide_spec(direction, cell)
                      for level, cell in _BLOCH_RESOLUTIONS.items()
                      for direction_name, direction in (('forward', 1), ('backward', -1))})


def _check_bloch_guide(results_by_name):
    """omega(+-beta0) of the nearest resonance vs the exact slab dispersion (10 nm cells) and the Richardson-
    extrapolated split."""
    beta = _BETA_INDEX * 2.0 * math.pi / _WAVELENGTH
    omega_guess = 2.0 * math.pi * C0 / _WAVELENGTH
    expected = {name: exact_slab_omega(beta, _EPS_SUBSTRATE, _EPS_CORE, _EPS_CLADDING, _GUIDE_GYRATION,
                                       _CORE_THICKNESS, direction, omega_guess)
                for name, direction in (('forward', 1), ('backward', -1))}
    expected_split = expected['backward'] - expected['forward']
    measured, splits = {}, {}
    for level in _BLOCH_RESOLUTIONS:
        for name in expected:
            rows = results_by_name[f'{name}_{level}'].result('em', 'table')['rows']
            measured[f'{name}_{level}'] = 2.0 * math.pi * C0 / float(rows[0][1]) if rows else math.nan
        splits[level] = measured[f'backward_{level}'] - measured[f'forward_{level}']
    extrapolated = 2.0 * splits['fine'] - splits['coarse']
    errors = {f'omega_{name}': abs(measured[f'{name}_coarse'] / expected[name] - 1.0) for name in expected}
    errors['split_extrapolated'] = abs(extrapolated / expected_split - 1.0)
    tolerances = {'omega_forward': _NRPS_OMEGA_TOLERANCE, 'omega_backward': _NRPS_OMEGA_TOLERANCE,
                  'split_extrapolated': _NRPS_SPLIT_TOLERANCE}
    return scaled_worst_result(errors, tolerances, measured_omega=measured, expected_omega=expected,
                               splits=splits, extrapolated_split=extrapolated, expected_split=expected_split,
                               split_errors={level: split / expected_split - 1.0 for level, split in splits.items()},
                               gyration=_GUIDE_GYRATION, beta0=beta)


# ---- em_modes_nrps_first_order: the guide as a cross-section (x periodic over 4 cells, CPML along y, subpixel
# smoothing), cladding gyration_x = -g (the in-plane gyration_z g turned), g = 0.05: delta_beta of TM0 (row 2) vs the
# exact beta_f - beta_b of the slab (first order: error ~ g^2 relative, plus second-order discretisation).
_MODES_GYRATION = 0.05
_MODES_CELL = 10e-9
_MODES_TOLERANCE = 5e-3


def _modes_guide_spec():
    size_x = 4 * _MODES_CELL
    settings = {'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic', 'wavelength': _WAVELENGTH, 'n_modes': 2,
                'n_eff_guess': 3.3}
    spec = ModelSpec(
        title=f'{_MODES_TYPE} first-order non-reciprocal phase shift of a magneto-optic-clad slab',
        domain=DomainSpec(size_x=size_x, size_y=_GUIDE_HEIGHT),
        geometry=_guide_geometry(size_x),
        materials=_guide_materials('gyration_x', -_MODES_GYRATION),
        physics=[PhysicsSpec('modes', _MODES_TYPE, settings=settings, features=[FeatureSpec('table', 'mode_table')])],
        mesh=MeshSpec(mode='uniform', dx=_MODES_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu', subpixel_smoothing=True),
        study=StudySpec(type='mode_analysis'))
    return {'guide': spec}


def _check_modes_guide(results_by_name):
    """delta_beta of TM0 (te_fraction ~ 0) vs the exact split; TE0 (no E_z . E_y overlap) must give ~0."""
    table = results_by_name['guide'].result('modes', 'table')
    columns = table['columns']
    rows = [dict(zip(columns, row)) for row in table['rows']]
    k0 = 2.0 * math.pi / _WAVELENGTH
    expected = (exact_slab_beta(k0, _EPS_SUBSTRATE, _EPS_CORE, _EPS_CLADDING, _MODES_GYRATION, _CORE_THICKNESS, 1)
                - exact_slab_beta(k0, _EPS_SUBSTRATE, _EPS_CORE, _EPS_CLADDING, _MODES_GYRATION, _CORE_THICKNESS, -1))
    tm = [row for row in rows if row['te_fraction'] < 0.5]
    te = [row for row in rows if row['te_fraction'] >= 0.5]
    measured = tm[0]['delta_beta'] if tm else math.nan
    errors = {'tm0_delta_beta': abs(measured / expected - 1.0),
              'te0_delta_beta': abs(te[0]['delta_beta']) / abs(expected) if te else math.inf}
    return scaled_worst_result(errors, {'tm0_delta_beta': _MODES_TOLERANCE, 'te0_delta_beta': 1e-9},
                               measured_delta_beta=measured, expected_delta_beta=expected,
                               n_eff=[row['n_eff'] for row in rows], gyration_x=-_MODES_GYRATION)


GYROTROPY_FDFD_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_hz_gyrotropic_reciprocity',
        label='EM frequency domain: non-reciprocity of a magneto-optic (gyration_z) block',
        description=(f'{_HZ}, the box of em_fdfd_hz_anisotropic_reciprocity (all CPML, 2 x 2 um, 25 nm cells) with a '
                     f'lossless block of eps {_PAIR_EPS:g} and gyration_z +-{_PAIR_GYRATION:g} (eps_xy = +-j g) between '
                     'points A and B: from_a radiates a magnetic line current at A and probes H_z at B, from_b the '
                     'reverse. With the magnetisation reversed the operator is the area-weighted transpose, so '
                     'H(B; A, +g) = H(A; B, -g) and H(A; B, +g) = H(B; A, -g) (tolerance '
                     f'{_PAIR_TOLERANCE:g}, direct-solve round-off), while H(B; A, +g) differs from H(A; B, +g) by '
                     f'more than {_PAIR_MIN_ASYMMETRY:g} (else inf). Four flux lines around the block (from_a, +g): '
                     f'net outflow / sum |fluxes| <= {_FLUX_TOLERANCE:g} (a real g absorbs nothing; the discrete '
                     'Poynting balance is exact). Scaled errors, tolerance 1. cpu backend.'),
        specs=_pair_specs, check=_check_pair),
    SelfTestCase(
        name='em_fdfd_hz_gyrotropic_nrps',
        label='EM frequency domain: non-reciprocal phase shift of a magneto-optic slab guide (Bloch eigen)',
        description=(f'{_HZ}, 10 and 5 nm cells, subpixel smoothing: silica substrate, a {_CORE_THICKNESS * 1e9:g} '
                     f'nm silicon core (eps {_EPS_CORE:g}) and a cladding of eps {_EPS_CLADDING:g} with gyration_z '
                     f'{_GUIDE_GYRATION:g}, x periodic over {_BLOCH_CELLS} cells with Bloch kx = +-beta0 (beta0 = '
                     f'{_BETA_INDEX:g} k0 at 1550 nm), CPML along y; eigenfrequency of the guided TM0 band at each '
                     'sign vs the exact slab dispersion (H_z and (eps H\' + beta g H) / det continuous, solved for '
                     'omega at beta0 forward and backward): relative omega errors at 10 nm (tolerance '
                     f'{_NRPS_OMEGA_TOLERANCE:g}) and the non-reciprocal split omega(-beta0) - omega(+beta0), '
                     'Richardson-extrapolated from both resolutions since it converges at first order (the '
                     'off-diagonal couplings take plain cell means across the interface; tolerance '
                     f'{_NRPS_SPLIT_TOLERANCE:g}). Scaled errors, tolerance 1. cpu backend.'),
        specs=_bloch_guide_specs, check=_check_bloch_guide),
)

GYROTROPY_MODES_SELFTEST_CASES = (
    SelfTestCase(
        name='em_modes_nrps_first_order',
        label='EM modes: first-order non-reciprocal phase shift (gyration_x) vs the exact slab',
        description=(f'{_MODES_TYPE}, the magneto-optic slab guide of em_fdfd_hz_gyrotropic_nrps as a cross-section '
                     '(x periodic over 4 cells of 10 nm, CPML along y, subpixel smoothing) with cladding gyration_x '
                     f'= -{_MODES_GYRATION:g} (the in-plane gyration_z turned into the cross-section frame): '
                     'delta_beta of TM0 vs the exact beta_forward - beta_backward of the slab (relative, tolerance '
                     f'{_MODES_TOLERANCE:g}: first order in g plus second-order discretisation), TE0 (no E_y E_z '
                     'product) ~0. cpu backend.'),
        specs=_modes_guide_spec, check=_check_modes_guide),
)

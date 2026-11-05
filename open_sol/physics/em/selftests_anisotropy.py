"""Model-run self-tests of the em_fdfd_hz off-diagonal permittivity eps_xy: a rotated uniaxial slab against its
unrotated equivalent (exact at normal incidence) and the reciprocity of point-source pairs through a rotated block;
no physics imports."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import max_relative_error, probe_phasor, worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_fdfd import cpu_specs

_TYPE = 'em_fdfd_hz'
_EM = 'em'
_WAVELENGTH = 1e-6                     # m
_ORDINARY, _EXTRAORDINARY = 1.5, 2.0   # indices of the uniaxial crystal
_AXIS_ANGLE = 30.0                     # deg, in-plane optic axis from +x


def _rotated_permittivity():
    """(eps_xx, eps_yy, eps_xy) of the crystal with its optic axis at _AXIS_ANGLE, rounded to float32 (the MaterialGrid
    values the solve sees)."""
    angle = math.radians(_AXIS_ANGLE)
    ordinary, contrast = _ORDINARY ** 2, _EXTRAORDINARY ** 2 - _ORDINARY ** 2
    values = (ordinary + contrast * math.cos(angle) ** 2, ordinary + contrast * math.sin(angle) ** 2,
              contrast * math.sin(angle) * math.cos(angle))
    return tuple(float(np.float32(value)) for value in values)


# em_fdfd_hz_rotated_slab: 20 nm cells, a strip 3 um long (CPML at both ends) and 5 cells wide (periodic), a line source
# across it in vacuum (cell 25) launching a normal-incidence plane wave onto a 1 um slab (cells 50..99) of the rotated
# crystal. Along x: H_z(x) only, so D_x = 0 on every face, E_y = eta_yy D_y with 1 / eta_yy = eps_yy - eps_xy^2 / eps_xx
# and E_x = eta_xy (centre mean of D_y): the discrete solution equals that of an unrotated slab of eps_yy' = eps_yy -
# eps_xy^2 / eps_xx exactly, and inside the slab the collocated probes give E_x / E_y = eta_xy / eta_yy = -eps_xy /
# eps_xx. The 'y' runs are the same strip turned by 90 degrees (eps_xx' = eps_xx - eps_xy^2 / eps_yy, E_y / E_x =
# -eps_xy / eps_yy): the two face couplings of the operator.
_SLAB_CELL = 20e-9
_SLAB_LENGTH, _SLAB_WIDTH = 3.0e-6, 0.1e-6
_SLAB_SPAN = (1.0e-6, 2.0e-6)          # along the propagation axis
_SLAB_SOURCE = 0.51e-6                 # centre of cell 25
_SLAB_PROBE = (1.51e-6, 0.05e-6)       # (along, across): centres of cells 75 and 2
_SLAB_TOLERANCE = 1e-5                 # relative field difference / probe ratio (float32 rounding of eps_yy')


def _slab_spec(axis, rotated):
    """Strip along `axis` with the rotated slab (eps_xy set) or its unrotated equivalent."""
    eps_xx, eps_yy, eps_xy = _rotated_permittivity()
    along_x = axis == 'x'
    size_x, size_y = (_SLAB_LENGTH, _SLAB_WIDTH) if along_x else (_SLAB_WIDTH, _SLAB_LENGTH)

    def point(along, across):
        return (along, across) if along_x else (across, along)
    x0, y0 = point(_SLAB_SPAN[0], 0.0)
    x1, y1 = point(_SLAB_SPAN[1], _SLAB_WIDTH)
    sx0, sy0 = point(_SLAB_SOURCE, 0.0)
    sx1, sy1 = point(_SLAB_SOURCE, _SLAB_WIDTH)
    px, py = point(*_SLAB_PROBE)
    if rotated:
        properties = {'eps_xx': eps_xx, 'eps_yy': eps_yy, 'eps_xy': eps_xy}
    elif along_x:
        properties = {'eps_xx': eps_xx, 'eps_yy': eps_yy - eps_xy ** 2 / eps_xx}
    else:
        properties = {'eps_xx': eps_xx - eps_xy ** 2 / eps_yy, 'eps_yy': eps_yy}
    periodic = ('y_lo', 'y_hi') if along_x else ('x_lo', 'x_hi')
    settings = {'wavelength': _WAVELENGTH, **{f'bc_{face}': 'periodic' for face in periodic}}
    features = [FeatureSpec('source', 'line_source', ['source_line'], {'amplitude': 1.0, 'phase': 0.0}),
                *(FeatureSpec(f'probe_{component}', 'field_probe', ['probe_point'], {'component': component})
                  for component in ('ex', 'ey'))]
    return ModelSpec(
        title=f"{_TYPE} {'rotated' if rotated else 'unrotated equivalent'} uniaxial slab, propagation along {axis}",
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('slab', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}),
                  GeometrySpec('source_line', 'line', {'x0': sx0, 'y0': sy0, 'x1': sx1, 'y1': sy1}),
                  GeometrySpec('probe_point', 'point', {'x': px, 'y': py})],
        materials=[MaterialSpec('crystal', properties=properties, selection=['slab'])],
        physics=[PhysicsSpec(_EM, _TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=StudySpec(type='stationary'))


def _slab_specs():
    return {f'{axis}_{kind}': _slab_spec(axis, kind == 'rotated') for axis in ('x', 'y')
            for kind in ('rotated', 'unrotated')}


def _check_slab(results_by_name):
    """Per axis: rotated vs unrotated H_z (Re and Im over the window) and the in-slab probe ratio vs -eps_xy / eps_ii."""
    eps_xx, eps_yy, eps_xy = _rotated_permittivity()
    errors, details = {}, {}
    for axis, (along, across, eps_along) in (('x', ('ey', 'ex', eps_xx)), ('y', ('ex', 'ey', eps_yy))):
        rotated, unrotated = results_by_name[f'{axis}_rotated'], results_by_name[f'{axis}_unrotated']
        for part in ('re', 'im'):
            errors[f'{axis}_hz_{part}'] = max_relative_error(rotated.field(f'{_EM}.hz_{part}').values,
                                                             unrotated.field(f'{_EM}.hz_{part}').values)
        ratio = probe_phasor(rotated, _EM, f'probe_{across}') / probe_phasor(rotated, _EM, f'probe_{along}')
        expected = -eps_xy / eps_along
        errors[f'{axis}_probe_ratio'] = abs(ratio - expected) / abs(expected)
        details[f'{axis}_probe_ratio'] = [ratio.real, ratio.imag]
        details[f'{axis}_expected_ratio'] = expected
    return worst_result(errors, _SLAB_TOLERANCE, eps_xx_yy_xy=[eps_xx, eps_yy, eps_xy], **details)


# em_fdfd_hz_anisotropic_reciprocity: all-CPML 2 x 2 um box, 25 nm cells, a lossy rotated-crystal block between two
# points A and B (not mirror images of each other); physics 'from_a' radiates a magnetic line current at A and probes
# H_z at B, 'from_b' the reverse. The H_z operator times the cell areas is complex symmetric (the eps_xy couplings are
# each other's area-weighted transposes and the block stays clear of the CPML), so H_z(B; A) = H_z(A; B) to solver
# round-off. The 'diagonal' run (eps_xy = 0) shows the coupling term changes the answer.
_PAIR_CELL = 25e-9
_PAIR_SIZE = 2.0e-6
_PAIR_BLOCK = (0.6e-6, 0.7e-6, 1.4e-6, 1.2e-6)
_PAIR_POINTS = {'a': (0.4125e-6, 0.5125e-6), 'b': (1.5625e-6, 1.4625e-6)}
_PAIR_LOSS = 2e4                       # S/m in the block
_PAIR_TOLERANCE = 1e-8                 # relative |H(B; A) - H(A; B)| (direct solve round-off)
_PAIR_MIN_CHANGE = 1e-3                # the eps_xy term must change H(B; A) by more than this (relative)


def _pair_spec(with_xy):
    """Block with (from_a and from_b physics) or without eps_xy (from_a only: the check reads only its change)."""
    eps_xx, eps_yy, eps_xy = _rotated_permittivity()
    properties = {'eps_xx': eps_xx, 'eps_yy': eps_yy, 'sigma_e': _PAIR_LOSS}
    if with_xy:
        properties['eps_xy'] = eps_xy
    x0, y0, x1, y1 = _PAIR_BLOCK
    geometry = [GeometrySpec('block', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})]
    geometry += [GeometrySpec(f'point_{name}', 'point', {'x': x, 'y': y}) for name, (x, y) in _PAIR_POINTS.items()]
    physics = [PhysicsSpec(f'from_{source}', _TYPE, settings={'wavelength': _WAVELENGTH}, features=[
        FeatureSpec('source', 'point_source', [f'point_{source}'],
                    {'excitation': 'magnetic_current', 'magnetic_current': 1.0, 'phase': 0.0}),
        FeatureSpec('probe', 'field_probe', [f'point_{target}'], {'component': 'hz'})])
        for source, target in ((('a', 'b'), ('b', 'a')) if with_xy else (('a', 'b'),))]
    return ModelSpec(
        title=f"{_TYPE} reciprocity through a rotated crystal block ({'with' if with_xy else 'without'} eps_xy)",
        domain=DomainSpec(size_x=_PAIR_SIZE, size_y=_PAIR_SIZE),
        geometry=geometry,
        materials=[MaterialSpec('crystal', properties=properties, selection=['block'])],
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_PAIR_CELL),
        study=StudySpec(type='stationary'))


def _pair_specs():
    return cpu_specs({'anisotropic': _pair_spec(True), 'diagonal': _pair_spec(False)})


def _check_pair(results_by_name):
    """|H(B; A) - H(A; B)| / |H(B; A)| of the anisotropic run (error), the change against the diagonal run (> floor)."""
    anisotropic, diagonal = results_by_name['anisotropic'], results_by_name['diagonal']
    forward, backward = probe_phasor(anisotropic, 'from_a', 'probe'), probe_phasor(anisotropic, 'from_b', 'probe')
    without_xy = probe_phasor(diagonal, 'from_a', 'probe')
    error = abs(forward - backward) / abs(forward)
    change = abs(forward - without_xy) / abs(forward)
    return {'measured': error, 'expected': 0.0, 'error': error if change > _PAIR_MIN_CHANGE else math.inf,
            'tolerance': _PAIR_TOLERANCE,
            'details': {'h_b_from_a': [forward.real, forward.imag], 'h_a_from_b': [backward.real, backward.imag],
                        'h_b_from_a_without_eps_xy': [without_xy.real, without_xy.imag],
                        'relative_change_by_eps_xy': change, 'min_change': _PAIR_MIN_CHANGE}}


ANISOTROPY_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_hz_rotated_slab', label='EM frequency domain: rotated uniaxial slab (eps_xy) = its equivalent',
        description=('em_fdfd_hz, 20 nm cells: a periodic 5-cell strip, CPML at its ends, a line source in vacuum and a '
                     f'1 um slab of a uniaxial crystal (n_o {_ORDINARY:g}, n_e {_EXTRAORDINARY:g}, optic axis '
                     f'{_AXIS_ANGLE:g} deg from x in the plane: eps_xx, eps_yy, eps_xy) at normal incidence, along x '
                     'and (turned) along y. At normal incidence D along the propagation is 0 on every face, so the '
                     'discrete field equals that of an unrotated slab with eps_yy - eps_xy^2 / eps_xx (along x; '
                     'eps_xx - eps_xy^2 / eps_yy along y) exactly, and inside the slab the collocated probes give E_x / '
                     'E_y = -eps_xy / eps_xx (E_y / E_x = -eps_xy / eps_yy): D_x = 0, which pins the sign, size and '
                     'centre averaging of each face coupling. error = the largest relative H_z (Re, Im) difference '
                     f'and probe-ratio error; tolerance {_SLAB_TOLERANCE:g} (float32 rounding of the equivalent '
                     'permittivity). cpu backend (not yet confirmed by a run there).'),
        specs=lambda: cpu_specs(_slab_specs()), check=_check_slab),
    SelfTestCase(
        name='em_fdfd_hz_anisotropic_reciprocity',
        label='EM frequency domain: reciprocity through an eps_xy crystal',
        description=('em_fdfd_hz, all-CPML 2 x 2 um box, 25 nm cells, a lossy (sigma_e '
                     f'{_PAIR_LOSS:g} S/m) block of the rotated uniaxial crystal of em_fdfd_hz_rotated_slab between '
                     'two points A and B: physics from_a radiates a magnetic line current at A and probes H_z at B, '
                     'from_b the reverse. Lorentz reciprocity of a symmetric eps tensor: H_z(B; A) = H_z(A; B); '
                     'discretely exact because the eps_xy face couplings are area-weighted transposes of each other '
                     '(the operator times the cell areas stays complex symmetric). error = |H(B; A) - H(A; B)| / '
                     f'|H(B; A)|, tolerance {_PAIR_TOLERANCE:g}; inf unless eps_xy changes H(B; A) by more than '
                     f'{_PAIR_MIN_CHANGE:g} against the same block without it (from_a only). cpu backend (not yet '
                     'confirmed by a run there).'),
        specs=_pair_specs, check=_check_pair),
)

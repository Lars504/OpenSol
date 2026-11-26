"""Self-tests of the brillouin_scattering drive of a rectangular-cell (Q1) solid on the optics' own tensor mesh: the
cell and face drive transposes against optomechanics' centre strains and face displacements (CPU), and a loss-factor
glass slab under one focused beam, the tensor solid against P2 triangles (model run on the cpu). definition registers
them; no physics imports at module level (brillouin / optomechanics inside the checks)."""
import math

import numpy as np

from ...model.schema import FREQUENCY_RESPONSE_STUDY, SelfTestCase
from ...model.selftest_helpers import (SubChecks, cell_centre, cpu_defaults, float_list, horizontal_line, no_runs,
                                       norm_relative_error, random_complex, relative_error, scaled_worst_result)
from ...model.spec import (DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec,
                           StudySpec)
from ..em.selftests_focused_beam import BEAM_LINE, focused_beam_feature
from .selftests import SOLID_TYPE_NAME, _elastic, _rectangle
from .selftests_brillouin_spectrum import _FEATURE, _frequencies, _frequency_axis, _series, _spectrum

# ---- brillouin_tensor_transpose (CPU): random complex nodal u, stresses and face loads on non-uniform cells ----
_TRANSPOSE_CELLS = (7, 5)
_TRANSPOSE_RTOL = 1e-13   # sums of a few hundred products (round-off)

# ---- brillouin_tensor_vs_triangles ----
_OPTICS = 'optics'
_TENSOR_SOLID = 'solid_q1'
_TRIANGLE_SOLID = 'solid_p2'
_OPTICS_MESH = 'optics'
_TRIANGLE_MESH = 'phonons'
_FEATURE_TYPE = 'brillouin_scattering'   # = definition_brillouin.BRILLOUIN_FEATURE
_WAVELENGTH = 532e-9                     # m, vacuum
_CELL = 10e-9                            # m, tensor mesh (Q1 cells: 18 per phonon wavelength lambda0 / (2 n) = 183 nm)
_CELLS = (360, 130)                      # window cells (x, y)
_SURFACE = 0.7e-6                        # m, top of the glass (cladding and block); air above
_BLOCK_X = (0.6e-6, 3.0e-6)              # m, the mechanical glass block (y 0 .. _SURFACE)
_DRIVE = (1.0e-6, 0.3e-6, 2.6e-6, 0.7e-6)   # m, x0, y0, x1, y1 of the drive rectangle (top edge on the surface)
_LINE_ROW = 100                          # window row of the beam line (y = 1.005 um, air)
_WAIST = 0.4e-6                          # m
_FOCUS = (1.8e-6, 0.5e-6)                # m, 0.2 um inside the glass
_BEAM_POWER = 1.0                        # W/m
_INDEX = 1.45
_GLASS = _elastic(73e9, 0.17, eps_r=_INDEX ** 2, rho=2200.0, photoelastic_p11=0.121, photoelastic_p12=0.270,
                  mechanical_loss_factor=0.2)
_FINE, _BLOCK_SIZE, _COARSE = 25e-9, 35e-9, 100e-9   # m, triangle sizes (drive, block, elsewhere: void)
_MAX_RATIO = 1.3
_FREQUENCIES = (20e9, 45e9, 21)          # Hz: first, last, count (the line at ~32.6 GHz, ~6.5 GHz wide)
_EXTRAS = ('shift', 'fwhm', 'peak', 'scattered_fraction', 'q_mean', 'reference_shift')
_EXTRA_TOLERANCE = 2e-2
_PSD_TOLERANCE = 5e-2
_PEAK_SHARE = 0.1                        # psd samples above this share of the triangle peak enter the L2


# ---- brillouin_tensor_transpose ----
def _check_transpose(_):
    """w . u of cell_stress_drive / face_drive against the strain work and the face sum of optomechanics for random
    complex u; wrong shapes raise."""
    from .brillouin import cell_stress_drive, face_drive
    from .optomechanics import element_strains, face_displacements

    checks = SubChecks()
    rng = np.random.default_rng(97)
    n_x, n_y = _TRANSPOSE_CELLS
    dx, dy = 0.5 + rng.random(n_x), 0.5 + rng.random(n_y)
    nodal = random_complex(rng, (n_x + 1, n_y + 1, 2))
    stress = random_complex(rng, (3, n_x, n_y))
    strain_xx, strain_yy, strain_xy = element_strains(nodal, dx, dy)
    expected = complex(np.sum(np.outer(dx, dy) * (stress[0] * strain_xx + stress[1] * strain_yy
                                                  + stress[2] * 2.0 * strain_xy)))
    measured = complex(np.sum(cell_stress_drive(stress, dx, dy) * nodal))
    checks.add_bound('cell_stress_transpose', abs(measured - expected) / abs(expected), _TRANSPOSE_RTOL)
    face_x, face_y = random_complex(rng, (n_x - 1, n_y)), random_complex(rng, (n_x, n_y - 1))
    displacement_x, displacement_y = face_displacements(nodal)
    expected = complex(np.sum(displacement_x * face_x) + np.sum(displacement_y * face_y))
    measured = complex(np.sum(face_drive(face_x, face_y) * nodal))
    checks.add_bound('face_transpose', abs(measured - expected) / abs(expected), _TRANSPOSE_RTOL)
    checks.add_raises('stress_shape_raises', lambda: cell_stress_drive(stress[:, :-1], dx, dy))
    checks.add_raises('face_shape_raises', lambda: face_drive(face_x[:-1], face_y))
    return checks.result()


# ---- brillouin_tensor_vs_triangles ----
def _solid(name, mesh):
    """A plane-strain solid fixed on y_lo with the brillouin_scattering feature on the drive rectangle (local
    reference, peak fit)."""
    return PhysicsSpec(name, SOLID_TYPE_NAME, mesh=mesh, features=[
        FeatureSpec('floor', 'fixed', ['y_lo']),
        FeatureSpec(_FEATURE, _FEATURE_TYPE, ['drive'], {'optical_physics': _OPTICS, 'reference': 'local',
                                                         'fit_peak': True})])


def _comparison_specs():
    size_x, size_y = (count * _CELL for count in _CELLS)
    line_y = cell_centre(_LINE_ROW, _CELL)
    geometry = [horizontal_line(BEAM_LINE, line_y, 0.0, size_x),
                _rectangle('cladding', 0.0, 0.0, size_x, _SURFACE),
                _rectangle('block', _BLOCK_X[0], 0.0, _BLOCK_X[1], _SURFACE),
                _rectangle('drive', *_DRIVE)]
    beam = focused_beam_feature(_BEAM_POWER, direction='negative', waist=_WAIST, numerical_aperture=None,
                                focus_x=_FOCUS[0], focus_y=_FOCUS[1])
    spec = ModelSpec(
        title='Self-test: Brillouin spectrum of a glass slab, rectangular cells against P2 triangles',
        domain=DomainSpec(size_x=size_x, size_y=size_y), geometry=geometry,
        materials=[MaterialSpec('air', properties={'eps_r': 1.0}, selection=['all']),
                   MaterialSpec('optical_glass', properties={'eps_r': _INDEX ** 2}, selection=['cladding']),
                   MaterialSpec('glass', properties=dict(_GLASS), selection=['block'])],
        physics=[PhysicsSpec(_OPTICS, 'em_fdfd_ez', settings={'wavelength': _WAVELENGTH}, features=[beam],
                             mesh=_OPTICS_MESH),
                 _solid(_TENSOR_SOLID, _OPTICS_MESH), _solid(_TRIANGLE_SOLID, _TRIANGLE_MESH)],
        meshes=[MeshSpec(name=_OPTICS_MESH, kind='tensor', mode='uniform', dx=_CELL),
                MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='nonuniform', dx=_COARSE, max_ratio=_MAX_RATIO,
                         refinements=[RefinementSpec('drive', _FINE), RefinementSpec('block', _BLOCK_SIZE)])],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type=FREQUENCY_RESPONSE_STUDY, settings={'frequencies': _frequencies(*_FREQUENCIES)}))
    return {'slab': spec}


def _check_comparison(results_by_name):
    """The tensor solid's spectrum against the triangle solid's (case description): the extras, the psd above
    _PEAK_SHARE of the triangle peak (L2) and a moving-boundary part found on the rectangular cells."""
    results = results_by_name['slab']
    tensor, triangles = (_spectrum(results, name) for name in (_TENSOR_SOLID, _TRIANGLE_SOLID))
    errors, tolerances = {}, {}
    for name in _EXTRAS:
        errors[name] = relative_error(float(tensor[name]), float(triangles[name]))
        tolerances[name] = _EXTRA_TOLERANCE
    frequencies = _frequency_axis(triangles)
    psd_tensor, psd_triangles = _series(tensor, 'psd'), _series(triangles, 'psd')
    if psd_tensor.shape != psd_triangles.shape or not np.array_equal(_frequency_axis(tensor), frequencies):
        errors['psd'] = math.inf
    else:
        keep = psd_triangles >= _PEAK_SHARE * float(np.max(psd_triangles))
        errors['psd'] = norm_relative_error(psd_tensor[keep], psd_triangles[keep])
    tolerances['psd'] = _PSD_TOLERANCE
    errors['moving boundary found'] = 0.0 if float(tensor['drive_norms']['moving_boundary']) > 0.0 else math.inf
    tolerances['moving boundary found'] = 1.0
    return scaled_worst_result(
        errors, tolerances, frequencies=float_list(frequencies), psd_tensor=float_list(psd_tensor),
        psd_triangles=float_list(psd_triangles),
        extras={name: [float(tensor[name]), float(triangles[name])] for name in _EXTRAS},
        drive_norms={'tensor': dict(tensor['drive_norms']), 'triangles': dict(triangles['drive_norms'])},
        fit_warnings=[tensor.get('fit_warning', ''), triangles.get('fit_warning', '')])


BRILLOUIN_TENSOR_CASES = (
    SelfTestCase(
        name='brillouin_tensor_transpose', label='Brillouin: rectangular-cell drive transposes',
        description=('CPU check of brillouin.cell_stress_drive and face_drive on 7 x 5 non-uniform cells: for random '
                     'complex nodal u, stresses (xx, yy, xy) per cell and normal face loads, w . u equals sum dA '
                     'sigma . (S_xx, S_yy, 2 S_xy) of optomechanics.element_strains and the face sum of '
                     f'optomechanics.face_displacements ({_TRANSPOSE_RTOL:g}); wrong shapes raise. error = number of '
                     'failed sub-checks, tolerance 0 (first estimates, not yet run).'),
        specs=no_runs, check=_check_transpose, max_wall_time=60.0),
    SelfTestCase(
        name='brillouin_tensor_vs_triangles',
        label='Brillouin: glass slab spectrum, rectangular cells against P2 triangles',
        description=('frequency_response study (cpu) of two plane-strain solid_mechanics reading one em_fdfd_ez '
                     'focused beam (532 nm, 10 nm tensor cells, default CPML, a 3.6 x 1.3 um window) into a glass '
                     'block (E 73 GPa, nu 0.17, rho 2200, p11 0.121, p12 0.270, mechanical_loss_factor 0.2; x 0.6 .. '
                     '3.0 um below y = 0.7 um) under air, fixed on y_lo; an optical cladding of the glass index (no '
                     'mechanics) fills the width below the surface, so the drive touches one optical interface, the '
                     "block's top face (glass / air: a moving-boundary part besides the photoelastic one). The beam "
                     '(waist 0.4 um, toward -y from a line over the window width in air) focuses 0.2 um inside the '
                     'glass; the drive rectangle x 1.0 .. 2.6 um, y 0.3 .. 0.7 um. Q1 cells on the mesh of the optics '
                     'against P2 triangles (25 nm in the drive, 35 nm in the rest of the block, 100 nm elsewhere), the '
                     'same brillouin_scattering drive (local reference, peak fit) over '
                     f'{_FREQUENCIES[2]} frequencies 20 .. 45 GHz around f_B = 2 n v_L / lambda0 = 32.6 GHz (the loss '
                     'factor makes the line ~6.5 GHz wide and damps the phonon within ~0.3 um, so the block walls >= '
                     '0.3 um from the drive return no echo above ~1e-2 of the readout): shift, fwhm, peak, '
                     f'scattered_fraction, q_mean and reference_shift within {_EXTRA_TOLERANCE:g} (the Q1 cells with '
                     'their consistent mass raise the shift by ~(q h)^2 / 24 = 0.5 %; both drives read the optical '
                     'cells exactly for a plane wave, the envelope ~(h / w0)^2 ~ 1e-3; the cells\' moving boundary '
                     'takes the mean of the two straddling cells, about cos(k h) ~ 1.5 % low in that part against '
                     f'the triangles\' surface values), psd above {_PEAK_SHARE:g} of the peak within '
                     f'{_PSD_TOLERANCE:g} (L2: the 0.5 % shift on a line ~20 % of f_B wide), a moving-boundary part '
                     'on the cells. measured = error in tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=_comparison_specs, check=_check_comparison),
)

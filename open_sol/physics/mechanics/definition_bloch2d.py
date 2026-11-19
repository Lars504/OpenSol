"""Model-layer pieces of the 2D solid's Bloch / Floquet unit cells (solid_mechanics settings periodic_x / periodic_y,
bloch_kx / bloch_ky): the settings, their validation rules and the estimator flags; definition.py wires them into the
type (the 3D twins live in definition3d)."""
from ...core.constants import FACES
from ...model.definition_helpers import is_axisymmetric
from ...model.schema import Issue, Param, feature_path
from .definition_common import (EIGENFREQUENCY_STUDY, SUPPORT_FEATURES, outer_faces, support_holders)

AXES_2D = ('x', 'y')
BAND_TEMPLATE_2D = 'solid_bloch_band_diagram'
_ELECTRODE_FEATURES = ('electrode', 'electrode_domain')


def periodic_params_2d():
    """periodic_x / _y and bloch_kx / _ky (Bloch / Floquet unit cells: both mesh kinds, eigenfrequency)."""
    periodic = tuple(Param(
        f'periodic_{axis}', 'bool', False, label=f'Periodic {axis}',
        help=('Eigenfrequency study (a plain periodic cell, every bloch_k 0, on triangles also stationary / '
              'time_dependent): the lo and hi faces of this axis are one Bloch-periodic pair, u (with phi when '
              'piezoelectric, the u_z amplitude with axial_wavenumber) on hi = on lo times exp(-j k L) (k its bloch_k, '
              'L the window size along it, exp(+j omega t)). Tensor mesh: corner nodes paired by index (plane strain or '
              'plane stress, also piezoelectric or axial_wavenumber); triangles: matching side nodes (planar; no PML '
              'ring on these faces, an elastic PML on the other axis makes a surface-wave cell). Modes of P^H K P, P^H '
              'M P (complex Hermitian; real when every k L is a multiple of pi), shown as real part (phase 0) and u_abs '
              '= |u|. Supports and electrodes stay off these faces (one on the other axis may reach their corners); '
              'floating electrodes must not touch them. Every bloch_k 0: a cell without supports keeps its rigid '
              'translations: set target_frequency > 0 (drops them) or hold it; piezoelectric then needs a ground '
              'electrode (k != 0: none). A static triangle cell solves K_r = P^T K P (zero mean strain along the axis: '
              'one cell of an infinite array; hold it, e.g. Fixed on the other axis). Not with axisymmetric, '
              'generalized plane strain, stress_stiffening, geometric_nonlinearity or frequency_response.'),
        tip=(f'Makes the {axis} faces periodic: the model is one cell of an infinite periodic structure, such as a '
             'phononic crystal or one period of a surface acoustic wave device.\n\nFor an eigenfrequency study; '
             'on a triangle mesh without a phase step also for a stationary study.')) for axis in AXES_2D)
    bloch = tuple(Param(
        f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
        visible_if=((f'periodic_{axis}', (True,)),),
        help=('Bloch wave vector of this axis (its periodic setting on): phase step exp(-j k L) per period L; k and k '
              '+ 2 pi / L are the same state; 0 = plain periodic, pi / L the zone edge. One eigenfrequency study per k '
              f'gives a band diagram (template {BAND_TEMPLATE_2D}).'),
        tip=('Phase step of the vibration from one period to the next along this axis, in radians per metre.\n\n0 '
             'means no phase step. Swept, it gives the bands of a phononic crystal.')) for axis in AXES_2D)
    return periodic + bloch


def periodic_axes_2d(settings_values) -> tuple:
    """Axes ('x', 'y') whose periodic_<axis> setting is on."""
    return tuple(axis for axis in AXES_2D if settings_values and settings_values.get(f'periodic_{axis}'))


def bloch_wavenumber_2d(settings_values, axis) -> float:
    value = (settings_values or {}).get(f'bloch_k{axis}', 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def solver_periodic_arguments(settings_values) -> dict:
    """periodic= / bloch_k= of the 2D solid solvers."""
    return {'periodic': tuple(bool(settings_values.get(f'periodic_{axis}')) for axis in AXES_2D),
            'bloch_k': tuple(bloch_wavenumber_2d(settings_values, axis) for axis in AXES_2D)}


def trivial_bloch_phase_2d(settings_values) -> bool:
    """Every periodic axis has bloch_k 0 (also without periodic axes)."""
    return all(bloch_wavenumber_2d(settings_values, axis) == 0.0 for axis in periodic_axes_2d(settings_values))


def complex_bloch_pencil_2d(settings_values) -> bool:
    """A periodic axis with bloch_k != 0: a complex Hermitian eigen pencil (k L = pi is real; the estimate does not
    tell)."""
    return bool(periodic_axes_2d(settings_values)) and not trivial_bloch_phase_2d(settings_values)


STATIC_PERIODIC_STUDIES = ('stationary', 'time_dependent')   # a plain periodic cell on triangles (every bloch_k 0)


def static_periodic_cell(vctx, settings_values, triangles) -> bool:
    """A plain periodic cell solved statically: triangles, every periodic bloch_k 0, a stationary / time-dependent
    study (P^T K P of system_fem; the ordinary support rules apply)."""
    return bool(triangles and periodic_axes_2d(settings_values) and trivial_bloch_phase_2d(settings_values)
                and vctx.study_type in STATIC_PERIODIC_STUDIES)


def bloch_issues_2d(vctx, physics_spec, settings_values, path, held_components, pml_faces, triangles):
    """Periodic axes of the 2D solid: warning per bloch_k on a non-periodic axis (ignored); errors outside the
    eigenfrequency study (a plain periodic cell on triangles also solves stationary / time_dependent), for geometry
    axisymmetric, generalized plane strain, stress_stiffening, geometric
    nonlinearity, a PML on a periodic face, per support / electrode on a periodic face and for a cell free to slide
    with every periodic bloch_k 0 at target_frequency 0 (singular K - 0 M; guided k_z > 0 has no rigid motion)."""
    periodic = periodic_axes_2d(settings_values)
    issues = [Issue('warning', path, f'bloch_k{axis} = {bloch_wavenumber_2d(settings_values, axis):.4g} rad/m is '
                                     f'ignored: periodic_{axis} is off')
              for axis in AXES_2D if axis not in periodic and bloch_wavenumber_2d(settings_values, axis) != 0.0]
    if not periodic:
        return issues
    names = ', '.join(periodic)
    if vctx.study_type != EIGENFREQUENCY_STUDY and not static_periodic_cell(vctx, settings_values, triangles):
        static = (' (a plain periodic cell, every bloch_k 0, on a triangle mesh also solves the stationary / '
                  'time_dependent study)' if vctx.study_type in STATIC_PERIODIC_STUDIES else '')
        issues.append(Issue('error', path, f'periodic axes ({names}) are solved by the {EIGENFREQUENCY_STUDY} study '
                                           f'(Bloch bands per k){static}; the {vctx.study_type} study of this periodic '
                                           'cell is not available: turn periodic_* off'))
    refused = []
    if is_axisymmetric(settings_values):
        refused.append('geometry axisymmetric')
    elif settings_values.get('out_of_plane') == 'generalized_plane_strain':
        refused.append('out_of_plane generalized_plane_strain')
    for setting in ('stress_stiffening', 'geometric_nonlinearity'):
        if settings_values.get(setting):
            refused.append(setting)
    if refused:
        issues.append(Issue('error', path, f"periodic axes ({names}) are not available with {', '.join(refused)} (a "
                                           'planar Bloch cell without a static prestate): turn those off'))
    on_pml = [face for face in pml_faces if face[0] in periodic]
    if on_pml:
        issues.append(Issue('error', path, f"pml on the periodic face(s) {', '.join(on_pml)}: a periodic side is the "
                                           'cell boundary, not an outer face; put the PML on the other axis'))
    for feature in physics_spec.features:
        if not (feature.enabled and (feature.type in SUPPORT_FEATURES or feature.type in _ELECTRODE_FEATURES)):
            continue
        faces = [face for face in outer_faces(vctx, feature, FACES) if face[0] in periodic]
        if faces:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} on the periodic face(s) {', '.join(faces)}: a hold or electrode there "
                                'would contradict its periodic image; select non-periodic faces (or objects inside '
                                'the cell)'))
    guided = vctx.study_type == EIGENFREQUENCY_STUDY and bool(settings_values.get('axial_wavenumber'))
    if (vctx.study_type == EIGENFREQUENCY_STUDY and trivial_bloch_phase_2d(settings_values)
            and not settings_values.get('target_frequency') and not guided):
        held = {component for _, component in support_holders(vctx, physics_spec, held_components)}
        sliding = [axis for axis in AXES_2D if axis not in held]
        if sliding and not pml_faces:
            issues.append(Issue('error', path, f"with every periodic bloch_k 0 the cell can slide along "
                                               f"{', '.join(sliding)} (a rigid mode at 0 Hz): set target_frequency > 0 "
                                               '(the rigid modes are dropped), a bloch_k != 0 or supports'))
    return issues

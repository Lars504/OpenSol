"""Mirror planes on the window edge (symmetry 2.5): the model's mirror faces, the image sign of a field component, the
total factor of a selection touching a plane, TFSF boxes opening onto a plane, the unfolded copy of a field and the
parity-sector runs of a source without definite parity (2.6); numpy only (GUI-thread safe)."""
# A mirror face (EM bc 'symmetric' = tangential E even, 'antisymmetric' = tangential E odd) lies on the window edge:
# the domain pads its ghost cell outside. Image of a component across a plane normal to axis a, with s = +1 symmetric /
# -1 antisymmetric: polar (E-like) tangential s, normal -s; axial (H-like) tangential -s, normal s; 'scalar' s; 'even'
# +1 always. Fields of a physics that does not set the face itself (non-EM physics) take s = +1: their face on the
# plane must be the even mirror (insulated / zero flux; heat and potential validation warn otherwise); the magnetic
# types and membrane take s from their own face feature there (FACE_FEATURE_SIGNS). Besides the EM bc settings, the
# MIRROR_FEATURES of a type (membrane symmetry / antisymmetry) make their outer faces model mirror planes too.
import itertools
from dataclasses import replace

import numpy as np

from .harmonic_probe import phasor_series

MIRROR_SIGNS = {'symmetric': 1, 'antisymmetric': -1}
MIRROR_FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')
_VECTOR_KINDS = tuple(f'{vector}_{axis}' for vector in ('polar', 'axial') for axis in 'xyz')
_PHASE_PREFIX = 'phase_'
# FieldInfo.parity values: 'phase_<kind>' = the argument [rad] of a phasor of that kind (an odd image adds pi)
PARITIES = ('even', 'scalar', *_VECTOR_KINDS, *(_PHASE_PREFIX + kind for kind in ('scalar', *_VECTOR_KINDS)))
# Non-EM types whose own face features fix the plane sign of their fields: {type: ({feature type: sign}, default)}.
# Magnetic 'symmetry' (dA/dn = 0, B_t = 0) keeps tangential A even like an EM 'symmetric' face; 'magnetic_insulation'
# (A = 0, also every unset face) makes it odd. Their field parities: A / J polar_z, B / H axial (physics/magnetic).
# magnetostatics_3d (scalar potential phi_m): the opposite face kinds, so the opposite signs: insulation (Neumann,
# every unset face) keeps phi_m even, symmetry (phi_m = 0) makes it odd; phi_m scalar, B / H polar (its gradient).
# membrane (w scalar): symmetry and a free (unset) face keep w even, antisymmetry and the w = 0 holds make it odd.
FACE_FEATURE_SIGNS = {**{type_name: ({'symmetry': 1, 'magnetic_insulation': -1}, -1)
                         for type_name in ('magnetostatics', 'magnetic_fields_ac')},
                      'magnetostatics_3d': ({'symmetry': -1, 'magnetic_insulation': 1}, 1),
                      'membrane': ({'symmetry': 1, 'antisymmetry': -1, 'clamped': -1, 'simply_supported': -1}, 1)}
# Face features that make their selected outer faces model mirror planes ({type: {feature type: sign}}); parity
# sectors swap them like the EM bc settings (sector_specs; a feature selecting several sector faces is split per face)
MIRROR_FEATURES = {'membrane': {'symmetry': 1, 'antisymmetry': -1}}
# Feature params every parity-sector copy sets ({type: {feature type: {param: value}}}): a membrane port then drives
# an incident mode of a guide ending on a mirror plane in the sector of its parity only, at twice its amplitude
SECTOR_FEATURE_PARAMS = {'membrane': {'port': {'sector_drive': True}}}


def own_mirror_faces(physics_spec) -> dict:
    """{face: sign} of the faces one PhysicsSpec sets to 'symmetric' (+1) / 'antisymmetric' (-1) itself."""
    settings = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
    faces = {}
    for face in MIRROR_FACES:
        value = settings.get(f'bc_{face}')
        sign = MIRROR_SIGNS.get(value) if isinstance(value, str) else None
        if sign is not None:
            faces[face] = sign
    return faces


def _enabled_feature_entities(spec, physics_spec, feature_signs):
    """(feature, sign, expanded entities or None) of the enabled features of a type in feature_signs ({type: sign});
    entities None for a bad selection (reported by the selection checks)."""
    from ..model.geometry import expand_entities   # lazy: the one model import of core (numpy-only module)
    for feature in physics_spec.features:
        sign = feature_signs.get(feature.type)
        if sign is None or not feature.enabled:
            continue
        try:
            entities = expand_entities(spec, feature.selection)
        except (ValueError, TypeError):
            entities = None
        yield feature, sign, entities


def feature_mirror_faces(spec, physics_spec) -> dict:
    """{face: sign} of the outer faces selected by the enabled MIRROR_FEATURES of one PhysicsSpec (the last feature
    selecting a face wins); {} for other types."""
    faces = {}
    for _, sign, entities in _enabled_feature_entities(spec, physics_spec, MIRROR_FEATURES.get(physics_spec.type, {})):
        for name in entities or ():
            if name in MIRROR_FACES:
                faces[name] = sign
    return faces


def physics_mirror_faces(spec, physics_spec) -> dict:
    """{face: sign} of the faces one PhysicsSpec makes mirror planes: its bc_<face> settings and mirror features."""
    return {**own_mirror_faces(physics_spec), **feature_mirror_faces(spec, physics_spec)}


def mirror_faces(spec, physics=None) -> dict:
    """{face: sign} of the model's mirror faces (a physics setting bc_<face> to 'symmetric' / 'antisymmetric', or a
    MIRROR_FEATURES face feature: membrane symmetry +1 / antisymmetry -1); with `physics` (a name) the sign is that
    physics' own where it sets the face, its face feature's for a FACE_FEATURE_SIGNS type, else +1 (the even mirror
    of a non-EM physics)."""
    faces, own, target = {}, {}, None
    for physics_spec in spec.physics:
        for face, sign in physics_mirror_faces(spec, physics_spec).items():
            faces.setdefault(face, sign)
            if physics_spec.name == physics:
                own[face] = sign
        if physics_spec.name == physics:
            target = physics_spec
    if physics is None:
        return faces
    if target is not None and target.type in FACE_FEATURE_SIGNS:
        return _feature_face_signs(spec, target, faces)
    return {face: own.get(face, 1) for face in faces}


def _feature_face_signs(spec, physics_spec, faces) -> dict:
    """{face: sign} on `faces` of a FACE_FEATURE_SIGNS physics: the last enabled face feature selecting the face wins,
    unset faces take the type's default."""
    feature_signs, default = FACE_FEATURE_SIGNS[physics_spec.type]
    signs = dict.fromkeys(faces, default)
    for _, sign, entities in _enabled_feature_entities(spec, physics_spec, feature_signs):
        for face in faces:
            if entities is not None and face in entities:
                signs[face] = sign
    return signs


def parity_tag(spec, physics_names) -> str:
    """Mirror parity of the solutions of the named physics, e.g. 'x_lo symmetric, y_lo antisymmetric' (the
    sector_specs label order); '' without mirror faces; physics with differing faces: '<name> <faces>' joined by '; '."""
    sign_names = {sign: name for name, sign in MIRROR_SIGNS.items()}
    texts = {physics_spec.name: ', '.join(f'{face} {sign_names[sign]}'
                                          for face, sign in sorted(physics_mirror_faces(spec, physics_spec).items()))
             for physics_spec in spec.physics if physics_spec.name in physics_names}
    distinct = set(texts.values())
    if len(distinct) <= 1:
        return next(iter(distinct), '')
    return '; '.join(f"{name} {text or 'no mirror'}" for name, text in texts.items())


def touch_factor(faces, touched) -> float:
    """2 per mirror face of `faces` that is in `touched`: a selection, line, plane or box touching a mirror plane joins
    its image, so its total over the unfolded model is that many times the drawn one (mirror-even integrands)."""
    return float(2 ** sum(1 for face in faces if face in touched))


def plane_wave_mirror_issue(face, plane_sign, propagation_axis, polarization_axis):
    """Problem text when a TFSF box may not open onto the mirror plane `face` (E parity plane_sign) for a plane wave
    along propagation_axis with E along polarization_axis ('x' / 'y' / 'z'), else None (the wave is even there)."""
    normal = face[0]
    if normal == propagation_axis:
        return (f'reaches the {face} mirror plane, which is normal to its propagation axis {propagation_axis} (its '
                'image would travel the other way): keep one scattered-field cell between the box and that plane')
    needed = -1 if normal == polarization_axis else 1   # E_p even about the plane: normal -s = +1, tangential s = +1
    if int(plane_sign) != needed:
        wanted = 'antisymmetric' if needed < 0 else 'symmetric'
        return (f'opens onto the {face} mirror plane, about which its incident wave (E along {polarization_axis}) is '
                f'even, so {face} must be {wanted}; else keep one scattered-field cell between the box and the plane')
    return None


def field_parity(info) -> str:
    """Parity of a FieldInfo: its parity, else 'even' for an unsigned field; ValueError for a signed field without one
    or an unknown value."""
    if info.parity:
        if info.parity not in PARITIES:
            raise ValueError(f"field '{info.name}' declares parity {info.parity!r}; valid: {', '.join(PARITIES)}")
        return info.parity
    if not info.signed:
        return 'even'
    raise ValueError(f"field '{info.name}' ({info.label}) declares no mirror parity, so it cannot be unfolded")


def plane_signs_text(signs) -> str:
    """FieldInfo.plane_signs text 'x_lo:1 y_lo:-1' of {face: sign} (sorted faces)."""
    return ' '.join(f'{face}:{int(sign)}' for face, sign in sorted(signs.items()))


def field_plane_signs(faces, info) -> dict:
    """faces {face: sign} with the FieldInfo's own plane_signs replacing the signs of the faces it names."""
    own = {}
    for entry in str(getattr(info, 'plane_signs', '') or '').split():
        face, _, sign = entry.partition(':')
        own[face] = int(sign)
    return {face: own.get(face, sign) for face, sign in faces.items()}


def image_sign(parity, normal_axis, plane_sign) -> int:
    """+1 / -1 factor of the mirror image of a component of `parity` across a plane normal to normal_axis ('x' / 'y' /
    'z') with E parity plane_sign (+1 symmetric, -1 antisymmetric)."""
    kind = parity[len(_PHASE_PREFIX):] if parity.startswith(_PHASE_PREFIX) else parity
    if kind == 'even':
        return 1
    if kind == 'scalar':
        return int(plane_sign)
    if kind not in _VECTOR_KINDS:
        raise ValueError(f'unknown parity {parity!r}; valid: {", ".join(PARITIES)}')
    vector, axis = kind.split('_')
    polar = -int(plane_sign) if axis == normal_axis else int(plane_sign)
    return polar if vector == 'polar' else -polar


def _image(values, axis, side, sign, staggered):
    """Unsigned mirror image of values across its lo / hi edge along axis 0 / 1 in display order (lo: placed before,
    hi: after). Hi-face samples (staggered) shift by one cell: the plane face is 0 when odd, else its nearest sample,
    and the hi image's last face (the missing lo face of the first cell) repeats the nearest one."""
    n = values.shape[axis]
    if not staggered:
        return np.flip(values, axis=axis)
    if side == 'lo':   # image cell j from the plane holds window face j - 1, j = 0 the plane face
        plane = np.take(values, [0], axis=axis) * (0.0 if sign < 0 else 1.0)
        return np.concatenate((np.flip(np.take(values, np.arange(n - 1), axis=axis), axis=axis), plane), axis=axis)
    # hi: the last window face is the plane; image cell m holds window face n - 2 - m
    return np.concatenate((np.flip(np.take(values, np.arange(n - 1), axis=axis), axis=axis),
                           np.take(values, [0], axis=axis)), axis=axis)


def _signed(values, sign, phase):
    """values times sign; a phase [rad] gains pi instead (wrapped to (-pi, pi])."""
    if sign > 0:
        return values
    if phase:
        return np.where(values > 0, values - np.pi, values + np.pi).astype(values.dtype)
    return -values


def unfold_field(field_data, faces):
    """Copy of a FieldData unfolded over its x / y mirror faces {face: sign} along its `axes` (mirror_faces with the
    field's physics; z faces are ignored, so a 3D side cut unfolds only its x or y axis): values, extent, native
    centres and native_shape take the images (lo before, hi after; both faces of an axis give one image each side;
    FieldInfo.plane_signs replace the signs of the faces they name). Unchanged without such faces or when already
    unfolded (FieldData.unfolded: parity-sector snapshots); ValueError for a field of unknown parity. Exact on native
    cells; staggered samples of a display resampling are approximate."""
    axes = tuple(getattr(field_data, 'axes', ('x', 'y')))
    planar = {face: sign for face, sign in field_plane_signs(faces, field_data.info).items()
              if face[0] in 'xy' and face[0] in axes}
    if not planar or getattr(field_data, 'unfolded', False):
        return field_data
    info = field_data.info
    parity = field_parity(info)
    phase = parity.startswith(_PHASE_PREFIX)
    values = np.asarray(field_data.values)
    extent = list(field_data.extent)
    centres = [field_data.x_centers, field_data.y_centers]
    copies = [1, 1]
    for axis_index, axis in enumerate(axes):
        lo, hi = planar.get(f'{axis}_lo'), planar.get(f'{axis}_hi')
        if lo is None and hi is None:
            continue
        staggered = axis in info.staggered
        start, stop = extent[2 * axis_index], extent[2 * axis_index + 1]
        parts = [values]
        axis_centres = None if centres[axis_index] is None else np.asarray(centres[axis_index], dtype=np.float64)
        centre_parts = [axis_centres]
        if lo is not None:
            sign = image_sign(parity, axis, lo)
            parts.insert(0, _signed(_image(values, axis_index, 'lo', sign, staggered), sign, phase))
            if axis_centres is not None:
                centre_parts.insert(0, 2.0 * start - axis_centres[::-1])
            extent[2 * axis_index] = 2.0 * start - stop
        if hi is not None:
            sign = image_sign(parity, axis, hi)
            parts.append(_signed(_image(values, axis_index, 'hi', sign, staggered), sign, phase))
            if axis_centres is not None:
                centre_parts.append(2.0 * stop - axis_centres[::-1])
            extent[2 * axis_index + 1] = 2.0 * stop - start
        values = np.concatenate(parts, axis=axis_index)
        copies[axis_index] = len(parts)
        if axis_centres is not None:
            centres[axis_index] = np.concatenate(centre_parts)
    native_shape = field_data.native_shape
    if native_shape is not None:
        native_shape = (int(native_shape[0]) * copies[0], int(native_shape[1]) * copies[1])
    return replace(field_data, values=np.ascontiguousarray(values, dtype=np.float32), extent=tuple(extent),
                   x_centers=centres[0], y_centers=centres[1], native_shape=native_shape, unfolded=True)


# ---- parity sectors (symmetry 2.6) ----
# A source without definite parity in a structure symmetric about k mirror planes: the run of the drawn source under
# one sign combination of the planes solves the source plus its images, i.e. 2^k times that parity part, so the field
# of the drawn source alone is the mean of the 2^k unfolded sector fields (linear), and a quadratic total over a region
# or surface symmetric about every plane (power, flux, absorbed power) is the mean of the sector totals / 2^k (the
# cross terms of two sectors are odd about a plane where they differ). Each sector run is a half (quarter, ...) model.
def sector_specs(spec, faces=None) -> dict:
    """{label: spec copy} of the 2^k parity sectors over the model's mirror faces (or `faces`, a subset of them): every
    physics setting a face to a mirror (bc_<face> or a MIRROR_FEATURES feature) gets each symmetric / antisymmetric
    combination, plus the SECTOR_FEATURE_PARAMS; label e.g. 'x_lo symmetric, y_lo antisymmetric' (also appended to the
    title)."""
    present = mirror_faces(spec)
    faces = sorted(present) if faces is None else list(faces)
    missing = [face for face in faces if face not in present]
    if not faces or missing:
        raise ValueError(f"sector_specs needs mirror faces of the model (bc_<face> symmetric / antisymmetric, or a "
                         f"membrane symmetry / antisymmetry face); {', '.join(missing) or 'none'} "
                         f"{'is' if len(missing) == 1 else 'are'} not one; model mirror faces: "
                         f"{', '.join(sorted(present)) or 'none'}")
    sectors = {}
    for values in itertools.product(MIRROR_SIGNS, repeat=len(faces)):
        sector = spec.copy()
        face_values = dict(zip(faces, values))
        for physics_spec in sector.physics:
            settings = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
            for face, value in face_values.items():
                if settings.get(f'bc_{face}') in MIRROR_SIGNS:
                    settings[f'bc_{face}'] = value
            if physics_spec.type in MIRROR_FEATURES:
                _swap_mirror_features(sector, physics_spec, face_values)
            sector_params = SECTOR_FEATURE_PARAMS.get(physics_spec.type, {})
            for feature in physics_spec.features:
                if feature.type in sector_params and isinstance(feature.params, dict):
                    feature.params.update(sector_params[feature.type])
        label = ', '.join(f'{face} {value}' for face, value in zip(faces, values))
        sector.title = f'{spec.title} [{label}]'
        sectors[label] = sector
    return sectors


def _swap_mirror_features(spec, physics_spec, face_values) -> None:
    """In place: the enabled MIRROR_FEATURES of physics_spec take the feature type of each sector face's value
    ({face: 'symmetric' | 'antisymmetric'}); a feature also selecting other entities or faces of both values is split
    (the rest keeps its name and type, the parts are named '<name>_<type>')."""
    feature_types = MIRROR_FEATURES[physics_spec.type]
    type_of_sign = {sign: feature_type for feature_type, sign in feature_types.items()}
    expanded = {id(feature): entities for feature, _, entities in
                _enabled_feature_entities(spec, physics_spec, feature_types)}
    features = []
    for feature in physics_spec.features:
        entities = expanded.get(id(feature))
        if entities is None:   # another type, disabled, or a bad selection
            features.append(feature)
            continue
        flipped = [name for name in entities if name in face_values]
        if not flipped:
            features.append(feature)
            continue
        others = [name for name in entities if name not in face_values]
        groups = {}
        for face in flipped:
            groups.setdefault(type_of_sign[MIRROR_SIGNS[face_values[face]]], []).append(face)
        pieces = [replace(feature, selection=others, params=dict(feature.params))] if others else []
        for feature_type, group_faces in sorted(groups.items()):
            name = feature.name if not pieces else f'{feature.name}_{feature_type}'
            pieces.append(replace(feature, name=name, type=feature_type, selection=list(group_faces),
                                  params=dict(feature.params)))
        features.extend(pieces)
    physics_spec.features = features


def _sector_count(count) -> int:
    """count when it is 2^k >= 2 (the sector runs of k mirror planes), else ValueError."""
    if count < 2 or count & (count - 1):
        raise ValueError(f'parity sectors come in 2^k >= 2 runs (k mirror planes), got {count}')
    return count


def combine_sector_fields(fields):
    """FieldData of the drawn source alone from the unfolded fields (Results.field(unfold=True)) of all its sector runs
    (sector_specs): their mean. Linear signed components only (Re / Im phasors, raw components): ValueError for
    unsigned or phase fields (|E|, intensities, losses do not superpose) and for differing components or grids."""
    fields = list(fields)
    _sector_count(len(fields))
    first = fields[0]
    parity = field_parity(first.info)
    if parity == 'even' or parity.startswith(_PHASE_PREFIX):
        raise ValueError(f"field '{first.component}' ({first.info.label}) is not a linear signed component, so its "
                         'sectors do not superpose; combine its Re / Im phasors (or raw components) instead, and '
                         'quadratic totals with combine_sector_totals')
    for other in fields[1:]:
        if (other.component != first.component or np.shape(other.values) != np.shape(first.values)
                or not np.allclose(other.extent, first.extent, rtol=1e-9, atol=0.0)):
            raise ValueError(f"combine_sector_fields: sector fields differ ('{other.component}' "
                             f"{np.shape(other.values)} vs '{first.component}' {np.shape(first.values)}, or their "
                             'extents): pass the same unfolded component of every sector run')
    values = np.mean([np.asarray(field.values, dtype=np.float64) for field in fields], axis=0)
    return replace(first, values=np.ascontiguousarray(values, dtype=np.float32))


def combine_sector_totals(totals):
    """Quadratic total of the drawn source alone (power, flux, absorbed power; scalars or arrays) over a region or
    surface symmetric about every mirror plane, from the unfolded totals of its 2^k sector runs: mean / 2^k."""
    values = [np.asarray(total, dtype=np.float64) for total in totals]
    return np.mean(values, axis=0) / _sector_count(len(values))


def combine_sector_far_fields(results):
    """'far_field' result of the drawn source alone from the unfolded far fields of its 2^k sector runs
    (Results.sector_result): per angle the mean of the complex amplitudes (amplitude_re / _im), power = |mean|^2
    (summed over a trailing vector axis); power_per_cw rescaled by its per-wavelength ratio to power."""
    results = list(results)
    _sector_count(len(results))
    first = results[0]
    missing = [index for index, result in enumerate(results)
               if result is None or 'amplitude_re' not in result or 'amplitude_im' not in result]
    if missing:
        raise ValueError(f'combine_sector_far_fields needs far_field results with amplitude_re / amplitude_im from '
                         f'every sector run; sector(s) {missing} have none')
    amplitudes = [np.asarray(result['amplitude_re'], dtype=np.float64)
                  + 1j * np.asarray(result['amplitude_im'], dtype=np.float64) for result in results]
    if any(amplitude.shape != amplitudes[0].shape for amplitude in amplitudes):
        raise ValueError('combine_sector_far_fields: the sector far fields differ in shape (wavelengths, angles)')
    amplitude = np.mean(amplitudes, axis=0)
    vector = amplitude.ndim == np.ndim(first['power']) + 1   # 3D: (n_f, n_angles, 3) Cartesian components
    power = np.sum(np.abs(amplitude) ** 2, axis=-1) if vector else np.abs(amplitude) ** 2
    combined = dict(first)
    combined.update(power=power, amplitude_re=amplitude.real, amplitude_im=amplitude.imag)
    if first.get('power_per_cw') is not None:   # power_per_cw = power x a per-wavelength factor (NaN where invalid)
        reference = np.asarray(first['power'], dtype=np.float64)
        per_cw = np.asarray(first['power_per_cw'], dtype=np.float64)
        total = np.sum(reference, axis=-1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            factor = np.where(total > 0.0, np.sum(per_cw, axis=-1, keepdims=True) / np.where(total > 0.0, total, 1.0),
                              np.nan)
        combined['power_per_cw'] = power * factor
    return combined


def combine_sector_result(results, n_faces):
    """Feature result of the drawn source alone from its 2^k sector results (k = n_faces), or None without a rule:
    far_field with amplitudes per angle (combine_sector_far_fields); flux_spectrum on a line / plane / box symmetric
    about every plane (mirror_factor 2^k in each sector) by combine_sector_totals of power (+ power_per_cw); 'xy' with
    sector_rule 'linear' (a point probe: combine_sector_linear) or sector_modes (guide modes of membrane ports and mode
    projections: combine_sector_modes)."""
    results = list(results)
    if len(results) != 2 ** int(n_faces) or any(not isinstance(result, dict) for result in results):
        return None
    kind = results[0].get('kind')
    if any(result.get('kind') != kind for result in results):
        return None
    if kind == 'xy' and all(result.get('sector_rule') == 'linear' for result in results):
        combined = combine_sector_linear(results)
    elif kind == 'xy' and all(result.get('sector_modes') is not None for result in results):
        combined = combine_sector_modes(results)
    elif kind == 'far_field' and all('amplitude_re' in result and 'amplitude_im' in result for result in results):
        combined = combine_sector_far_fields(results)
    elif kind == 'flux_spectrum' and all(result.get('mirror_factor') == 2.0 ** int(n_faces) for result in results):
        combined = dict(results[0])
        for key in ('power', 'power_per_cw'):
            if all(result.get(key) is not None for result in results):
                combined[key] = combine_sector_totals([result[key] for result in results])
            else:
                combined.pop(key, None)
    else:
        return None
    combined['combined_sectors'] = len(results)
    return combined


# ---- 'xy' sector rules of driven point probes and guide modes (membrane field_probe, port, mode_projection) ----
# sector_rule 'linear': the series pair <name>_re / _im of a phasor at a drawn point, so the drawn source's phasor is
# the sector mean. sector_modes: guide-mode amplitudes mode_<k>_re / _im of a sector run whose guide may end on a
# mirror plane: sector mode k is the unfolded mode sector_modes[k - 1] times sector_signs[k - 1] (a free mirror wall
# 2k - 1, a held one 2k, none k), so the unfolded amplitude is the sector mean of the signed amplitudes (0 where a
# sector lacks the mode); port powers are |amplitude|^2 times the power per |amplitude|^2 of the mode (from the sector
# where it is largest), the port's own incident power the sector mean / sector_drive_scale (a sector copy drives a
# mirrored guide's mode at 2 A in the sector of its parity only); incident_power and power_fraction_<n> follow from
# the sum over the ports of one physics (combine_sector_port_totals).
def _same_abscissa(results) -> bool:
    """True when every sector result has the first one's 'x' (frequencies or positions) to 1e-12."""
    first = np.asarray(results[0].get('x'), dtype=np.float64)
    for result in results[1:]:
        other = np.asarray(result.get('x'), dtype=np.float64)
        if other.shape != first.shape or not np.allclose(other, first, rtol=1e-12, atol=0.0):
            return False
    return True


def combine_sector_linear(results) -> dict:
    """'xy' result of a point phasor probe (sector_rule 'linear') of the drawn source alone: per abscissa the mean of
    the sectors' <name>_re + j <name>_im, its _abs / _phase recomputed; other series dropped."""
    if not _same_abscissa(results):
        raise ValueError('combine_sector_linear: the sector results have different abscissae')
    first = results[0]
    names = [name[:-3] for name in first['series'] if name.endswith('_re') and f'{name[:-3]}_im' in first['series']]
    series, units = {}, {}
    for name in names:
        phasors = []
        for result in results:
            values = result.get('series') or {}
            if f'{name}_re' not in values or f'{name}_im' not in values:
                raise ValueError(f"combine_sector_linear: a sector result has no series {name}_re / {name}_im")
            phasors.append(np.asarray(values[f'{name}_re'], dtype=np.float64)
                           + 1j * np.asarray(values[f'{name}_im'], dtype=np.float64))
        phasor_series(name, np.mean(phasors, axis=0), (first.get('series_units') or {}).get(f'{name}_re', ''),
                      series, units)
    combined = dict(first)
    combined.update(series=series, series_units=units)
    return combined


def _mode_phasors(results, container) -> dict:
    """{unfolded mode number: sector mean of the signed complex mode_<k>_re / _im} of each sector's `container` dict
    (its 'series' or an extra such as frequency_series), renumbered by sector_modes / sector_signs."""
    sums = {}
    for result in results:
        values = result.get(container)
        if not isinstance(values, dict):
            raise ValueError(f'combine_sector_modes: a sector result has no {container}')
        numbers = list(result['sector_modes'])
        signs = list(result.get('sector_signs') or [1] * len(numbers))
        for index, (number, sign) in enumerate(zip(numbers, signs), start=1):
            real, imaginary = values.get(f'mode_{index}_re'), values.get(f'mode_{index}_im')
            if real is None or imaginary is None:
                continue
            phasor = float(sign) * (np.asarray(real, dtype=np.float64) + 1j * np.asarray(imaginary, dtype=np.float64))
            sums[int(number)] = sums.get(int(number), 0.0) + phasor
    return {number: sums[number] / len(results) for number in sorted(sums)}


def _mode_powers(results, phasors) -> dict:
    """{unfolded mode number: port power [W]} = |combined amplitude|^2 x the power per |amplitude|^2 of the sector
    where that mode's amplitude is largest (0 where no sector carries it)."""
    largest, ratio = {}, {}
    for result in results:
        values = result['series']
        numbers = list(result['sector_modes'])
        for index, number in enumerate(numbers, start=1):
            power = values.get(f'power_{index}')
            real, imaginary = values.get(f'mode_{index}_re'), values.get(f'mode_{index}_im')
            if power is None or real is None or imaginary is None:
                continue
            squared = np.asarray(real, dtype=np.float64) ** 2 + np.asarray(imaginary, dtype=np.float64) ** 2
            number = int(number)
            if number not in largest:
                largest[number], ratio[number] = np.zeros_like(squared), np.zeros_like(squared)
            better = squared > largest[number]
            ratio[number] = np.where(better, np.asarray(power, dtype=np.float64) / np.where(better, squared, 1.0),
                                     ratio[number])
            largest[number] = np.where(better, squared, largest[number])
    return {number: np.abs(phasor) ** 2 * ratio[number] for number, phasor in phasors.items() if number in ratio}


def _power_fractions(series, units, incident) -> None:
    """In place: incident_power [W] and power_fraction_<n> = power_<n> / incident_power (NaN without incidence)."""
    incident = np.asarray(incident, dtype=np.float64)
    for name in [name for name in series if name.startswith('power_') and not name.startswith('power_fraction_')]:
        number = name.removeprefix('power_')
        with np.errstate(divide='ignore', invalid='ignore'):
            series[f'power_fraction_{number}'] = np.where(incident > 0.0, series[name] / incident, np.nan)
        units[f'power_fraction_{number}'] = '1'
    series['incident_power'] = incident
    units['incident_power'] = 'W'


def combine_sector_modes(results) -> dict:
    """'xy' guide-mode result (sector_modes) of the drawn source alone in the unfolded numbering: amplitudes
    mode_<m>_re / _im / _abs / _phase (+ frequency_series), port power_<m> / port_incident_power (incident_power and
    fractions of this port alone until combine_sector_port_totals), or a mode_projection 'frequency' view (its
    averaged_series: mode_<m>_power, _rms, total_power)."""
    if not _same_abscissa(results):
        raise ValueError('combine_sector_modes: the sector results have different abscissae')
    first = results[0]
    first_units = first.get('series_units') or {}
    combined = dict(first)
    series, units = {}, {}
    if isinstance(first.get('averaged_series'), dict):   # mode_projection result_axis 'frequency'
        phasors = _mode_phasors(results, 'averaged_series')   # (n_frequencies, averaged points) each
        total = 0.0
        averaged = {}
        for number, phasor in phasors.items():
            power = np.mean(np.abs(phasor) ** 2, axis=-1)
            series[f'mode_{number}_power'], units[f'mode_{number}_power'] = power, first_units.get('mode_1_power', 'm^2')
            series[f'mode_{number}_rms'], units[f'mode_{number}_rms'] = np.sqrt(power), first_units.get('mode_1_rms', 'm')
            averaged[f'mode_{number}_re'], averaged[f'mode_{number}_im'] = phasor.real.copy(), phasor.imag.copy()
            total = total + power
        series['total_power'], units['total_power'] = np.asarray(total, dtype=np.float64), first_units.get(
            'total_power', 'm^2')
        combined['averaged_series'] = averaged
    else:
        phasors = _mode_phasors(results, 'series')
        unit = first_units.get('mode_1_re', 'm')
        for number, phasor in phasors.items():
            phasor_series(f'mode_{number}', phasor, unit, series, units)
        if any('power_1' in (result.get('series') or {}) for result in results):   # a port
            for number, power in _mode_powers(results, phasors).items():
                series[f'power_{number}'], units[f'power_{number}'] = power, 'W'
            own = [np.asarray((result.get('series') or {}).get('port_incident_power', 0.0), dtype=np.float64)
                   for result in results]
            own_power = np.mean(np.broadcast_arrays(*own), axis=0) / float(first.get('sector_drive_scale') or 1.0)
            series['port_incident_power'], units['port_incident_power'] = own_power, 'W'
            _power_fractions(series, units, own_power)
        if isinstance(first.get('frequency_series'), dict):   # mode_projection: every recorded frequency
            frequency_series = {}
            for number, phasor in _mode_phasors(results, 'frequency_series').items():
                phasor_series(f'mode_{number}', phasor, unit, frequency_series, {})
            combined['frequency_series'] = frequency_series
    combined.update(series=series, series_units=units, sector_modes=sorted(phasors),
                    sector_signs=[1] * len(phasors))
    return combined


def combine_sector_port_totals(combined) -> None:
    """In place over {'physics.feature': combined result}: incident_power and power_fraction_<n> of every combined
    port (sector_modes with port_incident_power) from the sum of the own incident powers of the ports of its physics."""
    groups = {}
    for key, result in combined.items():
        if (isinstance(result, dict) and result.get('sector_modes') is not None
                and 'port_incident_power' in (result.get('series') or {})):
            groups.setdefault(str(key).partition('.')[0], []).append(result)
    for results in groups.values():
        total = np.sum([np.asarray(result['series']['port_incident_power'], dtype=np.float64) for result in results],
                       axis=0)
        for result in results:
            _power_fractions(result['series'], result['series_units'], total)

"""Symmetry check of mirror and axisymmetric models (symmetry 2.7; CPU, numpy, GUI-thread safe): geometry drawn beyond
a mirror plane or the (r, z) axis must be the mirror image of the half the run solves, else one warning per plane."""
import itertools

import numpy as np

from ..core.mirror import MIRROR_SIGNS, physics_mirror_faces
from .definition_helpers import GEOMETRY_SETTING, is_axisymmetric
from .geometry import (GEOMETRY_KINDS, cross_section_runs, material_paint_order, material_selections,
                       rasterize_window)
from .schema import Issue, known_physics

SAMPLES_PER_SIDE_2D = 160   # material samples per axis on each side of the plane (2D)
SAMPLES_PER_SIDE_3D = 40    # likewise per axis in 3D (z layers included)
EXTENSION_FRACTION = 0.25   # beyond the plane by at most this share of the inside reach: an edge extension
MISMATCH_MARK = 'is not the mirror image of the solved half: '   # precedes the object names (self-test parse)


def symmetry_issues(vctx) -> list:
    """One warning per mirror plane (a physics bc_<face> or mirror feature, core.mirror) or (r, z) axis (x_lo of a physics
    with geometry 'axisymmetric') whose material map drawn beyond it differs from the mirror image of the inside."""
    planes = _symmetry_planes(vctx)
    sizes = vctx.domain_size
    if not planes or sizes is None:
        return []
    three_d = vctx.spec.domain.size_z is not None
    if three_d and vctx.size_z is None:
        return []
    sizes = (*sizes, vctx.size_z) if three_d else tuple(sizes)
    selections = material_selections(vctx.spec)   # None for one that fails to expand (the selection checks)
    names = sorted({entity for entities in selections if entities is not None for entity in entities
                    if entity in vctx.shapes and GEOMETRY_KINDS[vctx.shapes[entity].kind].entity_kind == 'domain'})
    if not names:
        return []
    issues = []
    for face, (path, owner_text, solved_text) in planes.items():
        if face[0] == 'z' and not three_d:
            continue
        mismatch = plane_mismatch(vctx.shapes, selections, names, face, sizes)
        if mismatch is not None:
            objects, fraction, depth = mismatch
            issues.append(Issue('warning', path, (
                f"{owner_text}, but the geometry drawn beyond it {MISMATCH_MARK}"
                f"{', '.join(repr(name) for name in objects)} differ(s) across the plane ({100.0 * fraction:.0f} % "
                f"of the checked samples, up to {depth:.3g} m from it). {solved_text}; draw the structure symmetric "
                'about the plane or move the window edge onto its symmetry plane')))
    return issues


def _symmetry_planes(vctx) -> dict:
    """{face: (issue path, owner sentence, what the run solves)}: mirror faces (first physics setting them) and the
    x_lo axis of axisymmetric physics (2D only; a mirror on x_lo wins)."""
    planes, sign_names = {}, {sign: name for name, sign in MIRROR_SIGNS.items()}
    for physics_spec in vctx.spec.physics:
        for face, sign in physics_mirror_faces(vctx.spec, physics_spec).items():
            value = sign_names[sign]
            if face not in planes:
                planes[face] = (('physics', physics_spec.name),
                                f"{face} is a{'n' if value[0] == 'a' else ''} {value} mirror plane of "
                                f"'{physics_spec.name}'",
                                'The run solves the drawn half and its mirror image, ignoring what lies beyond')
    if vctx.spec.domain.size_z is not None or 'x_lo' in planes:
        return planes
    for physics_spec, physics_type in known_physics(vctx.spec, vctx.registry):
        if not any(param.name == GEOMETRY_SETTING for param in physics_type.settings):
            continue
        if is_axisymmetric(vctx.values(physics_type.settings, physics_spec.settings)):
            planes['x_lo'] = (('physics', physics_spec.name),
                              f"x_lo is the rotation axis (r = 0) of axisymmetric '{physics_spec.name}'",
                              'The run solves the body of revolution of the r >= 0 half, ignoring x < 0')
            break
    return planes


def sample_centres(face, sizes, samples_per_side) -> tuple:
    """Per axis 1D ascending sample centres: the window's along the plane, along its normal the window side and its
    mirror image (exact pairs: index k <-> 2 n - 1 - k); float32 for z (3D cross-section heights)."""
    normal = 'xyz'.index(face[0])
    centres = []
    for axis, size in enumerate(sizes):
        offsets = (np.arange(samples_per_side) + 0.5) * (float(size) / samples_per_side)
        if axis == normal:
            plane = float(size) if face.endswith('hi') else 0.0
            offsets = np.concatenate((plane - offsets[::-1], plane + offsets))
        centres.append(offsets.astype(np.float32) if axis == 2 else offsets)
    return tuple(centres)


def object_masks(shapes, names, centres) -> dict:
    """{name: full bool mask on the sample grid of centres (2 or 3 axes)}; 3D one cross-section per z run."""
    shape = tuple(axis_centres.size for axis_centres in centres)
    masks = {name: np.zeros(shape, dtype=bool) for name in names}
    if len(centres) == 2:
        runs, heights = [(None, None)], [None]
    else:
        runs = cross_section_runs(names, shapes, centres[2])
        heights = [centres[2][k_start] for k_start, _ in runs]
    for (k_start, k_stop), z in zip(runs, heights):
        for name in names:
            window, cropped = rasterize_window(name, shapes, centres[0], centres[1], np, z)
            if 0 in cropped.shape:
                continue
            if z is None:
                masks[name][window] = cropped
            else:
                masks[name][window[0], window[1], k_start:k_stop] = cropped[:, :, None]
    return masks


def paint(selections, masks, shape, objects) -> np.ndarray:
    """int16 image of 1 + winning material index (material_paint_order) painted by the named objects only (no
    background material: 0 elsewhere)."""
    image = np.zeros(shape, dtype=np.int16)
    for material_index in material_paint_order(selections):
        for entity in selections[material_index]:
            if entity in objects and entity in masks:
                image[masks[entity]] = material_index + 1
    return image


def plane_mismatch(shapes, selections, names, face, sizes):
    """(objects, mismatched fraction, largest distance [m]) of the samples beyond face where the drawn object material
    differs from the mirror image of the inside (all objects, or only those reaching beyond: inside-only objects rely
    on the mirror) by more than one sample (staircase ties), or None. Checked: the objects reaching beyond and their
    images; an edge extension (beyond by at most EXTENSION_FRACTION of its inside reach, or running to the far window
    edge) only within its reach."""
    samples = SAMPLES_PER_SIDE_3D if len(sizes) == 3 else SAMPLES_PER_SIDE_2D
    centres = sample_centres(face, sizes, samples)
    masks = object_masks(shapes, names, centres)
    shape = tuple(axis_centres.size for axis_centres in centres)
    normal = 'xyz'.index(face[0])
    hi = face.endswith('hi')
    plane = float(sizes[normal]) if hi else 0.0
    beyond = np.zeros(2 * samples, dtype=bool)
    beyond[slice(samples, None) if hi else slice(0, samples)] = True
    distance = np.abs(centres[normal].astype(np.float64) - plane)   # along the normal, from the plane
    spacing = float(sizes[normal]) / samples
    other_axes = tuple(axis for axis in range(len(shape)) if axis != normal)
    reach_shape = [1] * len(shape)
    reach_shape[normal] = 2 * samples
    crossing, checked = set(), np.zeros(shape, dtype=bool)
    for name, mask in masks.items():
        along = mask.any(axis=other_axes)   # samples along the normal holding the object
        if not (along & beyond).any():
            continue
        crossing.add(name)
        outside_reach = float(distance[along & beyond].max())
        inside_reach = float(distance[along & ~beyond].max()) if (along & ~beyond).any() else 0.0
        extension = (outside_reach <= EXTENSION_FRACTION * inside_reach
                     or inside_reach >= float(sizes[normal]) - spacing)
        region = mask | np.flip(mask, axis=normal)
        if extension:
            region &= (distance <= outside_reach + 0.5 * spacing).reshape(reach_shape)
        checked |= region
    if not crossing:
        return None
    checked &= beyond.reshape(reach_shape)
    labels = paint(selections, masks, shape, set(names))   # beyond the plane only crossing objects paint
    matched = np.zeros(shape, dtype=bool)
    for image in (labels, paint(selections, masks, shape, crossing)):   # inside-only objects may lack an image
        mirrored = np.pad(np.flip(image, axis=normal), 1, mode='edge')
        for offsets in itertools.product((0, 1, 2), repeat=len(shape)):
            matched |= labels == mirrored[tuple(slice(offset, offset + size)
                                                for offset, size in zip(offsets, shape))]
    bad = checked & ~matched
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return None
    bad_image = np.flip(bad, axis=normal)
    objects = [name for name in sorted(crossing) if (masks[name] & (bad | bad_image)).any()]
    depth = float(distance[bad.any(axis=other_axes)].max())
    return objects, n_bad / max(int(np.count_nonzero(checked)), 1), depth

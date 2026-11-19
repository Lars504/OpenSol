"""Mirror-plane factors of the totals the magnetic solvers report themselves (symmetry 2.5): every model mirror plane
(core.mirror.mirror_faces; a magnetic symmetry face is one) doubles the window energy and losses, and a coil reaching a
plane joins its image (stranded: in series, solid: in parallel); numpy only."""
import numpy as np

from ...core.mirror import MIRROR_FACES, touch_factor


class MirrorTotals:
    """Mixin of the magnetic solvers: the model mirror faces {face: plane sign of this physics' fields} given at create
    (set_mirror_faces, before the first coil), the factor of the whole-window totals and the per-coil factor."""

    mirror_faces = {}   # class default (scripts): no mirror plane, every factor 1

    def set_mirror_faces(self, faces) -> None:
        """Model mirror planes on the window edge, {face: sign} (core.mirror.mirror_faces(spec, physics); the sign is
        this physics' own: 2D A even +1 / odd -1, 3D phi_m even +1 / odd -1): energy and losses count every image
        (x 2 per plane), a coil reaching a plane joins its image. Before any coil (its factor is fixed when added);
        never the axisymmetric axis x_lo (the ring weights already count the revolution) nor its r_max cylinder x_hi
        (no mirror image)."""
        faces = {str(face): int(sign) for face, sign in dict(faces).items()}
        bad = [face for face, sign in faces.items() if face not in MIRROR_FACES or sign not in (1, -1)]
        if bad:
            raise ValueError(f'{self.type_label}: mirror faces need a face of {MIRROR_FACES} and a sign +1 / -1, got '
                             f"{', '.join(f'{face}: {faces[face]}' for face in bad)}")
        if getattr(self, 'axisymmetric', False) and ('x_lo' in faces or 'x_hi' in faces):
            raise ValueError(f"{self.type_label} '{self.name}': x_lo (the rotation axis) and x_hi (the r_max cylinder) "
                             'of an axisymmetric model are no mirror planes, got mirror faces '
                             f"{', '.join(sorted(faces))}")
        if self.coils:
            raise ValueError(f"{self.type_label} '{self.name}': set the mirror faces before adding coils (their image "
                             'factor is fixed when they are added)')
        self.mirror_faces = faces

    @property
    def window_mirror_factor(self) -> float:
        """2 per model mirror plane: the window reaches every one, so its energy / losses count each image."""
        return touch_factor(self.mirror_faces, self.mirror_faces)

    def _coil_mirror_factor(self, name, touched, solid=False) -> float:
        """touch_factor of the mirror planes a coil reaches (`touched`: outer faces its cells / elements reach): its
        image joins it. A solid conductor reaching a plane where A is odd raises (its uniform applied field v would
        have to change sign across the plane)."""
        reached = [face for face in self.mirror_faces if face in touched]
        odd = [face for face in reached if self.mirror_faces[face] < 0]
        if solid and odd:
            raise ValueError(f"{self.type_label} '{self.name}': solid conductor '{name}' reaches the mirror plane(s) "
                             f"{', '.join(odd)}, where magnetic insulation makes A (and the conductor's current) odd: "
                             'its image would carry the opposite current at the opposite voltage, which one conductor '
                             'cannot. Keep the conductor off the plane (then its image is a separate conductor), or '
                             'use a symmetry face there')
        return touch_factor(self.mirror_faces, reached)


def tensor_touched_faces(mask) -> set:
    """Outer faces whose window edge layer holds a True cell of the host bool window mask (2D: x / y faces, 3D also
    z)."""
    mask = np.asarray(mask, dtype=bool)
    touched = set()
    for face in MIRROR_FACES[:2 * mask.ndim]:
        layer = np.take(mask, 0 if face.endswith('lo') else -1, axis='xyz'.index(face[0]))
        if bool(layer.any()):
            touched.add(face)
    return touched


def element_touched_faces(elements, owners_by_face) -> set:
    """Outer faces with a boundary entity owned by a True element of the host bool (n_elements,) mask;
    owners_by_face: {face: int owner array of its window-boundary edges / faces (-1 = none)}."""
    elements = np.asarray(elements, dtype=bool)
    touched = set()
    for face, owners in owners_by_face.items():
        owners = np.asarray(owners).ravel()
        owners = owners[owners >= 0]
        if owners.size and bool(elements[owners].any()):
            touched.add(face)
    return touched

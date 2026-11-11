"""3D near-to-far transform of em_fdtd_3d: equivalent currents J = n x H, M = -n x E on the six faces of a closed
cell-edge box (FluxBox3D spectra, each pair on its own sample grid), vacuum Green's function, far-field cut planes."""
# Radiation vectors N = sum J exp(+j k r.r') dA, L = sum M exp(+j k r.r') dA (exp(+j omega t), e^{-jkr} outgoing);
# r E_far = -j k / (4 pi) [eta0 N_perp - r x L] (Balanis E_theta = -jk (L_phi + eta0 N_theta) / (4 pi), ...).
# 'power' = r^2 |E_far|^2 / eta0 per steradian of the dt-weighted spectra (no 1/2, as the flux box), so power_per_cw =
# power / (2 |S|^2) [W/sr] integrates over the sphere to the flux box's power_per_cw / cos(omega dt / 2) (the flux
# spectra carry that re-timing factor, the far field leaves it out on purpose). The phase sums are separable
# over each face's two sample axes (two small matrix products per face, frequency and pair).
# Mirror planes (symmetry 2.5) unfold by images: reflecting about x_a = p maps r' to R r' + 2 p e_a and a current to S J
# (J polar, M axial: core.mirror.image_sign), so N_image(k) = S N_drawn(R k) exp(2 j k0 p k_a); every combination of
# the mirrors adds one image, and box faces on a plane are dropped (they join their image inside the unfolded box).
import itertools

import numpy as np

from ...core.constants import C0, ETA0
from ...core.mirror import image_sign
from ..em.results import _per_cw, _shared_waveform_spectrum
from .boundaries3d import AXES

CUT_PLANES = {'xz': (0, 2), 'yz': (1, 2), 'xy': (0, 1)}   # plane -> (axis at angle 0, axis at angle pi/2)
FAR_FIELD_QUANTITY = 'abs(E)^2'


class NearToFar3D:
    """Far field of a FluxBox3D's spectra (stepped by its owner, possibly shared with a flux box); the medium outside
    the box must be vacuum. cell_edges / cell_centers: Domain3D coordinates [m]. mirrors: ((axis, plane edge, E parity
    +-1), ...) of the mirror planes, at most one per axis: the pattern is that of the unfolded structure."""

    def __init__(self, box, cell_edges, cell_centers, mirrors=()):
        self.box = box
        self.frequencies = self.box.frequencies
        self._edges = tuple(np.asarray(edges, dtype=np.float64) for edges in cell_edges)
        self._centers = tuple(np.asarray(centers, dtype=np.float64) for centers in cell_centers)
        self.mirrors = tuple((int(axis), int(edge), int(sign)) for axis, edge, sign in mirrors)
        mirror_axes = [axis for axis, _, _ in self.mirrors]
        if len(set(mirror_axes)) != len(mirror_axes):
            raise ValueError('the 3D far field cannot unfold mirrors on both faces of one axis (infinitely many '
                             'images): keep one of them a cpml face')

    @property
    def sample_interval(self):
        return self.box.sample_interval

    def radiation_vectors(self, directions):
        """(N, L) complex128 (n_f, n_dir, 3) of the unit `directions` (n_dir, 3): the box currents plus their images
        over every combination of the mirror planes."""
        directions = np.asarray(directions, dtype=np.float64)
        wavenumbers = 2.0 * np.pi * self.frequencies / C0
        radiation_n = np.zeros((self.frequencies.size, directions.shape[0], 3), dtype=np.complex128)
        radiation_l = np.zeros_like(radiation_n)
        on_planes = {(axis, edge) for axis, edge, _ in self.mirrors}
        faces = [(outward, plane, plane.host_pairs()) for outward, plane in self.box.faces   # one transfer each
                 if (plane.normal_axis, plane.edge) not in on_planes]
        for flips in itertools.product((False, True), repeat=len(self.mirrors)):
            image_directions = directions.copy()
            phase = np.ones(radiation_n.shape[:2], dtype=np.complex128)
            signs_j, signs_m = np.ones(3), np.ones(3)
            for (axis, edge, sign), flip in zip(self.mirrors, flips):
                if not flip:
                    continue
                image_directions[:, axis] *= -1.0
                phase *= np.exp(2j * wavenumbers[:, None] * self._edges[axis][edge] * directions[None, :, axis])
                signs_j *= [image_sign(f'polar_{component}', AXES[axis], sign) for component in AXES]
                signs_m *= [image_sign(f'axial_{component}', AXES[axis], sign) for component in AXES]
            drawn_n, drawn_l = self._box_vectors(image_directions, faces)
            radiation_n += phase[:, :, None] * drawn_n * signs_j
            radiation_l += phase[:, :, None] * drawn_l * signs_m
        return radiation_n, radiation_l

    def _box_vectors(self, directions, faces):
        """(N, L) of the drawn box currents alone toward `directions`; faces = [(outward, plane, host_pairs)]."""
        n_f = self.frequencies.size
        radiation_n = np.zeros((n_f, directions.shape[0], 3), dtype=np.complex128)
        radiation_l = np.zeros_like(radiation_n)
        unit = np.eye(3)
        for outward, plane, host_pairs in faces:
            normal = plane.normal_axis
            normal_coordinate = self._edges[normal][plane.edge]
            for pair, e_spectrum, h_spectrum in host_pairs:
                face_axis, cell_axis = pair['face_axis'], pair['cell_axis']
                face_slice = pair['face_slice']
                coordinates = {face_axis: self._edges[face_axis][face_slice.start + 1:face_slice.stop + 1],
                               cell_axis: self._centers[cell_axis][pair['cell_slice']]}
                low, high = sorted((face_axis, cell_axis))
                current_j = outward * np.cross(unit[normal], unit[pair['h_axis']])    # J = n x H
                current_m = -outward * np.cross(unit[normal], unit[pair['e_axis']])   # M = -n x E
                for index, frequency in enumerate(self.frequencies):
                    k0 = 2.0 * np.pi * frequency / C0
                    phase_low = np.exp(1j * k0 * directions[:, low:low + 1] * coordinates[low][None, :])
                    phase_high = np.exp(1j * k0 * directions[:, high:high + 1] * coordinates[high][None, :])
                    phase_normal = np.exp(1j * k0 * directions[:, normal] * normal_coordinate)
                    h_sum = np.sum((phase_low @ (h_spectrum[index] * pair['area'])) * phase_high, axis=1)
                    e_sum = np.sum((phase_low @ (e_spectrum[index] * pair['area'])) * phase_high, axis=1)
                    radiation_n[index] += (phase_normal * h_sum)[:, None] * current_j[None, :]
                    radiation_l[index] += (phase_normal * e_sum)[:, None] * current_m[None, :]
        return radiation_n, radiation_l

    def far_field_amplitude(self, directions):
        """Complex (n_f, n_dir, 3) r E_far / sqrt(eta0) (Cartesian, a.u.) toward the unit directions: the radiated power
        per steradian (dt-weighted, no 1/2) is the sum of its |.|^2 over the components (parity sectors combine it per
        angle)."""
        directions = np.asarray(directions, dtype=np.float64)
        radiation_n, radiation_l = self.radiation_vectors(directions)
        along = np.sum(radiation_n * directions[None], axis=2, keepdims=True)
        field = float(ETA0) * (radiation_n - along * directions[None]) - np.cross(directions[None], radiation_l)
        wavenumbers = 2.0 * np.pi * self.frequencies / C0
        return (wavenumbers / (4.0 * np.pi * np.sqrt(float(ETA0))))[:, None, None] * field


def cut_plane_directions(cut_plane, n_angles):
    """(angles (n,) [rad], unit directions (n, 3)): counter-clockwise from the plane's first axis toward its second."""
    if cut_plane not in CUT_PLANES:
        raise ValueError(f'cut_plane must be one of {tuple(CUT_PLANES)}, got {cut_plane!r}')
    n_angles = int(n_angles)
    if n_angles < 1:
        raise ValueError(f'n_angles must be >= 1, got {n_angles}')
    first, second = CUT_PLANES[cut_plane]
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    directions = np.zeros((n_angles, 3))
    directions[:, first] = np.cos(angles)
    directions[:, second] = np.sin(angles)
    return angles, directions


def far_field_result_3d(monitor, physics, cut_plane='xz', n_angles=360) -> dict:
    """'far_field' in one cut plane: 'angles' from its first axis, 'power' (n_f, n_angles) a.u. per steradian = the sum
    over x, y, z of |amplitude|^2 (amplitude_re / _im (n_f, n_angles, 3)), 'power_per_cw' [W/sr] when every source
    shares one waveform; extras 'cut_plane', 'angle_axes'."""
    angles, directions = cut_plane_directions(cut_plane, n_angles)
    frequency = np.array(monitor.frequencies, dtype=np.float64)
    amplitude = monitor.far_field_amplitude(directions)
    power = np.sum(np.abs(amplitude) ** 2, axis=2)
    first, second = CUT_PLANES[cut_plane]
    result = {'kind': 'far_field', 'frequency': frequency, 'wavelength': C0 / frequency, 'angles': angles,
              'power': power, 'amplitude_re': amplitude.real, 'amplitude_im': amplitude.imag, 'unit': 'a.u.',
              'quantity': FAR_FIELD_QUANTITY, 'cut_plane': cut_plane, 'angle_axes': (AXES[first], AXES[second])}
    shared = _shared_waveform_spectrum(physics, frequency, monitor.sample_interval)
    if shared is not None:
        result['power_per_cw'] = _per_cw(power, *shared, factor=2.0)
    return result

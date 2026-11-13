"""3D boundary configuration of em_fdtd_3d (six faces: cpml, pec, pmc, periodic with an optional Bloch k, symmetric /
antisymmetric mirrors) and its CPML layer rows; the CPML grading layout is physics.em.boundaries.cpml_axis_profiles (one axis at a time
through an axis view)."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import DTYPE, FACES_3D
from ..em.boundaries import MIRROR_GHOST_CELLS, _cpml_recursion, cpml_axis_profiles, cpml_side_rows

MIRROR_TYPES_3D = ('symmetric', 'antisymmetric')   # E parity as the 2D mirrors; one ghost cell outside the window edge
FACE_TYPES_3D = ('cpml', 'pec', 'pmc', 'periodic', *MIRROR_TYPES_3D)
AXES = ('x', 'y', 'z')


class BoundaryConfig3D:
    """Per-face type over FACES_3D plus the CPML settings; value equality, unhashable, differences() as the 2D
    BoundaryConfig (meshing merges equal configs and reads npml_sides incl. z_lo / z_hi)."""
    MODEL_SETTING_NAMES = {**{face: f'bc_{face}' for face in FACES_3D},
                           'cpml_npml': 'pml_cells', 'cpml_order': 'pml_order',
                           'cpml_sigma_factor': 'pml_sigma_factor', 'cpml_kappa_max': 'pml_kappa_max',
                           'cpml_alpha_max': 'pml_alpha_max',
                           **{f'bloch_k{axis}': f'bloch_k{axis}' for axis in AXES}}

    def __init__(self, faces: dict = None, npml: int = 15, order: int = 3, sigma_factor: float = 1.0,
                 kappa_max: float = 15.0, alpha_max: float = 0.0, bloch_k=(0.0, 0.0, 0.0)):
        faces = dict(faces or {})
        unknown = sorted(set(faces) - set(FACES_3D))
        if unknown:
            raise ValueError(f'BoundaryConfig3D: unknown faces {unknown}; valid: {", ".join(FACES_3D)}')
        for face in FACES_3D:
            setattr(self, face, faces.get(face, 'cpml'))
        self.cpml_npml = npml
        self.cpml_order = order
        self.cpml_sigma_factor = sigma_factor
        self.cpml_kappa_max = kappa_max
        self.cpml_alpha_max = alpha_max
        # Bloch wave vector [rad/m] of the periodic axes (symmetry 2.8): F(x + L) = F(x) e^{-j k L}, exp(+j omega t);
        # ignored on non-periodic axes; k != 0 makes the engine step a second, imaginary field set
        self.bloch_kx, self.bloch_ky, self.bloch_kz = (float(value) for value in bloch_k)
        self.validate()

    def validate(self):
        """ValueError for unknown face types, unpaired periodic faces, out-of-range CPML settings or a non-finite
        Bloch k."""
        for face in FACES_3D:
            if getattr(self, face) not in FACE_TYPES_3D:
                raise ValueError(f'{face} must be one of {", ".join(FACE_TYPES_3D)}, got {getattr(self, face)!r}')
        if not all(math.isfinite(value) for value in self.bloch_k):
            raise ValueError(f'bloch_kx / bloch_ky / bloch_kz must be finite [rad/m], got {self.bloch_k}')
        for axis in AXES:
            lo, hi = getattr(self, f'{axis}_lo'), getattr(self, f'{axis}_hi')
            if (lo == 'periodic') != (hi == 'periodic'):
                raise ValueError(f'{axis}: periodic faces must be paired, got {lo} / {hi}')
        if self.needs_cpml:
            if int(self.cpml_npml) != self.cpml_npml or self.cpml_npml < 1:
                raise ValueError(f'cpml_npml must be an integer >= 1 with a cpml face, got {self.cpml_npml!r}')
            if int(self.cpml_order) != self.cpml_order or self.cpml_order < 1:
                raise ValueError(f'cpml_order must be an integer >= 1, got {self.cpml_order!r}')
            if not self.cpml_sigma_factor > 0.0 or not self.cpml_kappa_max >= 1.0 or not self.cpml_alpha_max >= 0.0:
                raise ValueError('CPML needs sigma_factor > 0, kappa_max >= 1 and alpha_max >= 0, got '
                                 f'{self.cpml_sigma_factor!r}, {self.cpml_kappa_max!r}, {self.cpml_alpha_max!r}')

    @property
    def needs_cpml(self) -> bool:
        return any(getattr(self, face) == 'cpml' for face in FACES_3D)

    @property
    def cpml_sides(self) -> dict:
        return {face: getattr(self, face) == 'cpml' for face in FACES_3D}

    @property
    def npml_sides(self) -> dict:
        """Domain padding per face: cpml_npml on cpml faces, the mirror ghost on mirror faces, else 0."""
        return {face: int(self.cpml_npml) if getattr(self, face) == 'cpml'
                else MIRROR_GHOST_CELLS if getattr(self, face) in MIRROR_TYPES_3D else 0 for face in FACES_3D}

    @property
    def ghost_faces(self) -> tuple:
        """Mirror faces, whose padding cell copies the window edge cell's width (core.mesh, x / y / z faces)."""
        return tuple(face for face in FACES_3D if getattr(self, face) in MIRROR_TYPES_3D)

    def periodic(self, axis) -> bool:
        return getattr(self, f'{axis}_lo') == 'periodic'

    @property
    def bloch_k(self) -> tuple:
        """(kx, ky, kz) [rad/m] as set (also on non-periodic axes, where they are ignored)."""
        return (self.bloch_kx, self.bloch_ky, self.bloch_kz)

    @property
    def has_bloch_phase(self) -> bool:
        """Whether a periodic axis carries k != 0 (the engine then steps a second, imaginary field set)."""
        return any(self.periodic(axis) and k != 0.0 for axis, k in zip(AXES, self.bloch_k))

    def bloch_phase_angles(self, periods) -> tuple:
        """Wrap phase k L [rad] per axis for the meshed periods L [m] (0 on non-periodic axes)."""
        return tuple(float(k) * float(period) if self.periodic(axis) else 0.0
                     for axis, k, period in zip(AXES, self.bloch_k, periods))

    def differences(self, other) -> tuple:
        """Names of the settings differing from `other` (('type',) for another class); CPML settings only when
        either needs CPML, bloch_k<axis> only when either is periodic on that axis."""
        if not isinstance(other, BoundaryConfig3D):
            return ('type',)
        names = list(FACES_3D)
        if self.needs_cpml or other.needs_cpml:
            names += ['cpml_npml', 'cpml_order', 'cpml_sigma_factor', 'cpml_kappa_max', 'cpml_alpha_max']
        names += [f'bloch_k{axis}' for axis in AXES if self.periodic(axis) or other.periodic(axis)]
        return tuple(name for name in names if getattr(self, name) != getattr(other, name))

    def __eq__(self, other):
        if not isinstance(other, BoundaryConfig3D):
            return NotImplemented
        return not self.differences(other)

    __hash__ = None

    def __repr__(self):
        faces = ', '.join(f'{face}={getattr(self, face)!r}' for face in FACES_3D)
        bloch = f', bloch_k={self.bloch_k}' if self.has_bloch_phase else ''
        return f'BoundaryConfig3D({faces}{bloch})'


def along(axis, index):
    """Index tuple selecting `index` (int, slice or integer array) along axis of a 3D array."""
    key = [slice(None)] * 3
    key[axis] = index
    return tuple(key)


def mirror_planes(config, window):
    """[(axis index, kind, ghost row, inner row, plane-face row)] of the mirror faces of a BoundaryConfig3D on the
    physical-window slices: the one writer of the mirror rows of the FDTD engine and the FDFD unknowns."""
    planes = []
    for axis, name in enumerate(AXES):
        for side in ('lo', 'hi'):
            kind = getattr(config, f'{name}_{side}')
            if kind not in MIRROR_TYPES_3D:
                continue
            cells = window[axis]
            ghost, inner = (cells.start - 1, cells.start) if side == 'lo' else (cells.stop, cells.stop - 1)
            plane_face = cells.start - 1 if side == 'lo' else cells.stop - 1   # hi-face row on the plane
            planes.append((axis, kind, ghost, inner, plane_face))
    return planes


def pec_wall_layers(config, window):
    """Index tuples of the outermost window cell layer of every 'pec' face of a BoundaryConfig3D (physical-window
    slices): the one writer of the wall layers of the FDTD engine and the FDFD unknowns."""
    layers = []
    for axis, name in enumerate(AXES):
        for side in ('lo', 'hi'):
            if getattr(config, f'{name}_{side}') == 'pec':
                layer = window[axis].start if side == 'lo' else window[axis].stop - 1
                layers.append(along(axis, slice(layer, layer + 1)))
    return layers


def solver_boundary_config_3d(domain):
    """domain.boundary_config checked against the Domain3D padding (ValueError when missing or different)."""
    config = domain.boundary_config
    if not isinstance(config, BoundaryConfig3D):
        raise ValueError(f'em_fdtd_3d needs a Domain3D built with a BoundaryConfig3D, got {type(config).__name__}')
    config.validate()
    if config.npml_sides != domain.npml_sides:
        raise ValueError(f'BoundaryConfig3D PML cells {config.npml_sides} do not match the domain padding '
                         f'{domain.npml_sides}')
    return config


@dataclass(frozen=True)
class _AxisView:
    """One axis of a BoundaryConfig3D seen as the 'x' axis of a 2D config (cpml_axis_profiles reads only these)."""
    cpml_sides: dict
    cpml_npml: int
    cpml_order: int
    cpml_sigma_factor: float
    cpml_kappa_max: float
    cpml_alpha_max: float


def axis_cpml_profile(config, axis, n_cells, pml_cell_size):
    """CPMLAxisProfile (float64 sigma / kappa / alpha of E rows at centres, H rows at hi faces) of one 3D axis."""
    view = _AxisView({'x_lo': config.cpml_sides[f'{axis}_lo'], 'x_hi': config.cpml_sides[f'{axis}_hi']},
                     int(config.cpml_npml), int(config.cpml_order), float(config.cpml_sigma_factor),
                     float(config.cpml_kappa_max), float(config.cpml_alpha_max))
    return cpml_axis_profiles(view, n_cells, 'x', pml_cell_size)


@dataclass(frozen=True, eq=False)
class CPMLAxis3D:
    """The CPML layers of one axis and Yee kind ('e' rows at centres, 'h' rows at hi faces), lo and hi side stacked
    along the axis (psi row = lo rows first): int32 row map (grid row -> psi row, -1 outside the layers), float32
    recursion b and gain = a x bulk inverse spacing per psi row, the two psi arrays (axis dimension = n_rows), on the
    run backend's array module; the fused Yee kernels read them (kernels3d CPML_AXIS_PARAMS)."""
    axis: int
    kind: str
    n_rows: int
    row_map: object
    b: object
    gain: object
    psi_first: object
    psi_second: object


def cpml_slabs(config, shape, bulk_inverse_spacings, pml_cell_sizes, dt, xp):
    """(layers, kappa_e, kappa_h): CPMLAxis3D per axis with a CPML side and kind (psi only inside the layers: rows x
    the other two axes; arrays of module xp) plus per-axis float64 kappa rows to fold into the curl spacings;
    pml_cell_sizes = sigma grading cell size [m] per axis."""
    layers, kappa_e, kappa_h = [], [], []
    npml = int(config.cpml_npml)
    for axis_index, axis in enumerate(AXES):
        n = shape[axis_index]
        profile = axis_cpml_profile(config, axis, n, pml_cell_sizes[axis_index])
        kappa_e.append(profile.kappa_e)
        kappa_h.append(profile.kappa_h)
        inverse_e, inverse_h = bulk_inverse_spacings[axis_index]
        # the cpml_axis_profiles rows (mirror-symmetric layout: outermost E rows 0 and n-1 unstretched)
        sides = [cpml_side_rows(side, n, npml) for side in ('lo', 'hi') if config.cpml_sides[f'{axis}_{side}']]
        if not sides:
            continue
        for kind_index, kind in enumerate(('e', 'h')):
            row_map = np.full(n, -1, dtype=np.int32)
            decays, gains = [], []
            for selected in (rows[kind_index] for rows in sides):
                row_map[selected] = np.arange(len(decays) * npml, (len(decays) + 1) * npml, dtype=np.int32)
                sigma, kappa, alpha = (getattr(profile, f'{name}_{kind}')[selected]
                                       for name in ('sigma', 'kappa', 'alpha'))
                a, b = _cpml_recursion(sigma, kappa, alpha, dt, np)   # float32 host rows
                inverse = (inverse_e if kind == 'e' else inverse_h)[selected].astype(np.float32)
                decays.append(b)
                gains.append(a * inverse)   # float32 product: the former in-kernel a[r] * inv[r]
            n_rows = npml * len(sides)
            psi_shape = list(shape)
            psi_shape[axis_index] = n_rows
            layers.append(CPMLAxis3D(axis_index, kind, n_rows, xp.asarray(row_map),
                                     xp.asarray(np.concatenate(decays)), xp.asarray(np.concatenate(gains)),
                                     xp.zeros(psi_shape, dtype=DTYPE), xp.zeros(psi_shape, dtype=DTYPE)))
    return layers, kappa_e, kappa_h


def cpml_psi_bytes(npml_sides, shape):
    """Device bytes of the psi slabs: 2 kinds x 2 components x npml x (other two axes) float32 per CPML side
    (npml_sides of the CPML faces only: a mirror ghost is no layer)."""
    total = 0
    for face, npml in npml_sides.items():
        axis_index = AXES.index(face[0])
        other = math.prod(count for index, count in enumerate(shape) if index != axis_index)
        total += 4 * int(npml) * other * 4
    return total

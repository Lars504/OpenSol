"""2D EM boundaries: BCType, BoundaryConfig (per-face CPML / PEC / PMC / periodic / Bloch / mirror types),
BoundaryManager (applies them around the Yee kernels) and the CPMLSided convolutional PML layers."""
import math

import numpy as np
from ...core.backend import array_module_of, array_module_or_default, cp
from ...core.kernels import twin
from dataclasses import dataclass
from enum import Enum

from ...core.constants import DTYPE, EPS0
from .yee_grid import yee_inverse_spacings


# Out-of-plane component naming: 'ez' = (E_z, H_x, H_y), 'hz' = (H_z, E_x, E_y).
_POLARIZATIONS = ('ez', 'hz')


# ---- fused in-place CPML kernel ---------------------------------------------
# One launch per active PML side and half-step, no temporaries (ElementwiseKernels update their output args in
# place): the psi recursion and the field correction run together after the Yee kernel, which never writes the curl
# inputs f_hi / f_lo of its own update (H kernel reads E, E kernel reads H), so psi sees the same samples as before it.
# The psi term takes the field's own Yee coefficient (cb / db, sigma factor 1 / (1 + k) included), exactly as the
# kappa-folded curl does; any other factor breaks the coordinate-transform image and the layer reflects. A core.kernels
# twin: the numpy twin follows the cupy source. T = the field dtype (float32, float64 for the double stepper): psi, a, b
# and the spacings are stored in it (CPMLSided dtype).
def _psi_update_apply_host(b, a, f_hi, f_lo, inv, sign, coefficient, psi, field):
    psi *= b
    psi += a * (f_hi - f_lo) * inv
    field += sign * coefficient * psi


_psi_update_apply = twin('em2d.cpml_psi_update_apply', cp.ElementwiseKernel(
    'T b, T a, T f_hi, T f_lo, T inv, T sign, T coefficient',
    'T psi, T field',
    'psi = b * psi + a * (f_hi - f_lo) * inv; field += sign * coefficient * psi',
    'cpml_psi_update_apply'), _psi_update_apply_host)

_SIGNS = (1.0, -1.0)   # psi correction signs (+, -), cast to the CPML dtype per side


class BCType(Enum):
    CPML = 'cpml'
    PEC = 'pec'
    PMC = 'pmc'
    PERIODIC = 'periodic'
    BLOCH = 'bloch'
    SYMMETRIC = 'symmetric'
    ANTISYMMETRIC = 'antisymmetric'


# Mirror faces (symmetry plan 2.5): the plane is the window edge; the domain pads MIRROR_GHOST_CELLS ghost cell outside
# it (npml_sides; graded meshes give it the window edge cell's width, so the image is exact), which the wall rows
# below overwrite with the mirror image. Every other physics sharing the domain has its face on the same plane.
MIRROR_TYPES = frozenset({BCType.SYMMETRIC, BCType.ANTISYMMETRIC})
MIRROR_GHOST_CELLS = 1
# Frequency-domain wall / mirror rows of the out-of-plane field per polarisation: outermost cell = sign x its inner
# neighbour (0: pinned), as the FDTD rows below write it (FDFD operator, EME wall fill).
WALL_SIGNS = {'ez': {BCType.PEC: 0.0, BCType.PMC: 1.0, BCType.SYMMETRIC: 1.0, BCType.ANTISYMMETRIC: -1.0},
              'hz': {BCType.PEC: 1.0, BCType.PMC: 0.0, BCType.SYMMETRIC: -1.0, BCType.ANTISYMMETRIC: 1.0}}
# Face types that put tangential E = 0 on the inner face of the outermost cell in H_z polarisation.
_HZ_PEC_PLANE_TYPES = frozenset({BCType.PEC, BCType.ANTISYMMETRIC})
# Face types that make E_z even about the inner face of the outermost cell (tangential H = 0 there) in E_z
# polarisation: the same row copy serves the pmc wall and the symmetric mirror.
_EZ_EVEN_PLANE_TYPES = frozenset({BCType.PMC, BCType.SYMMETRIC})


def _check_polarization(polarization):
    if polarization not in _POLARIZATIONS:
        raise ValueError(f"polarization must be 'ez' (E_z, H_x, H_y) or 'hz' (H_z, E_x, E_y), got {polarization!r}")


def _face_property(face):
    """BoundaryConfig face attribute coercing every assignment (string or BCType) to a BCType."""
    stored_name = '_' + face

    def get_face(self):
        return getattr(self, stored_name)

    def set_face(self, value):
        try:
            bc_type = BCType(value)
        except ValueError:
            raise ValueError(f"{face} must be one of {[t.value for t in BCType]}, got {value!r}") from None
        setattr(self, stored_name, bc_type)

    return property(get_face, set_face)


class BoundaryConfig:
    # Attribute -> model setting name (definition_common.domain_requirements), for messages naming a difference.
    MODEL_SETTING_NAMES = {'x_lo': 'bc_x_lo', 'x_hi': 'bc_x_hi', 'y_lo': 'bc_y_lo', 'y_hi': 'bc_y_hi',
                           'cpml_npml': 'pml_cells', 'cpml_order': 'pml_order',
                           'cpml_sigma_factor': 'pml_sigma_factor', 'cpml_kappa_max': 'pml_kappa_max',
                           'cpml_alpha_max': 'pml_alpha_max', 'bloch_kx': 'bloch_kx', 'bloch_ky': 'bloch_ky'}
    x_lo = _face_property('x_lo')
    x_hi = _face_property('x_hi')
    y_lo = _face_property('y_lo')
    y_hi = _face_property('y_hi')

    def __init__(self, x_lo='cpml', x_hi='cpml', y_lo='cpml', y_hi='cpml'):
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.y_lo = y_lo
        self.y_hi = y_hi

        self.cpml_npml: int = 15
        self.cpml_order: int = 3
        self.cpml_sigma_factor: float = 1.0
        self.cpml_kappa_max: float = 15.0
        self.cpml_alpha_max: float = 0.0   # S/m; acts only near omega eps0 (~1e4 S/m optical)

        # Bloch wave vector [rad/m] of a bloch axis: F(x + L) = F(x) e^{-j kx L} (exp(+j omega t)); k != 0 makes the FDTD
        # engines step a second (imaginary) field set and the FDFD / cross-section operators phase their wrap entries;
        # EME refuses it (operators.check_zero_bloch).
        self.bloch_kx: float = 0.0
        self.bloch_ky: float = 0.0

        self._validate()

    def _validate(self):
        """ValueError for unpaired periodic / bloch faces, out-of-range CPML settings or a non-finite Bloch k; runs at
        construction and again in solver_boundary_config (settings may be reassigned in between)."""
        if not (math.isfinite(self.bloch_kx) and math.isfinite(self.bloch_ky)):
            raise ValueError(f"bloch_kx / bloch_ky must be finite [rad/m], got {self.bloch_kx!r}, {self.bloch_ky!r}")
        for axis_name, lo, hi in [('x', self.x_lo, self.x_hi),
                                   ('y', self.y_lo, self.y_hi)]:
            periodic_types = {BCType.PERIODIC, BCType.BLOCH}
            if lo in periodic_types or hi in periodic_types:
                if lo != hi:
                    raise ValueError(
                        f"{axis_name}: periodic/bloch BCs must be paired. "
                        f"Got {lo.value}/{hi.value}")

        if self.needs_cpml:
            if int(self.cpml_npml) != self.cpml_npml or self.cpml_npml < 1:
                raise ValueError(f"cpml_npml must be an integer >= 1 with a cpml face, got {self.cpml_npml!r}")
            if int(self.cpml_order) != self.cpml_order or self.cpml_order < 1:
                raise ValueError(f"cpml_order must be an integer >= 1, got {self.cpml_order!r}")
            if not self.cpml_sigma_factor > 0.0:
                raise ValueError(f"cpml_sigma_factor must be > 0, got {self.cpml_sigma_factor!r}")
            if not self.cpml_kappa_max >= 1.0:
                raise ValueError(f"cpml_kappa_max must be >= 1, got {self.cpml_kappa_max!r}")
            if not self.cpml_alpha_max >= 0.0:
                raise ValueError(f"cpml_alpha_max must be >= 0 S/m, got {self.cpml_alpha_max!r}")

    @property
    def needs_cpml(self) -> bool:
        return any(s == BCType.CPML for s in
                   [self.x_lo, self.x_hi, self.y_lo, self.y_hi])

    @property
    def cpml_sides(self):
        return {
            'x_lo': self.x_lo == BCType.CPML,
            'x_hi': self.x_hi == BCType.CPML,
            'y_lo': self.y_lo == BCType.CPML,
            'y_hi': self.y_hi == BCType.CPML,
        }

    @property
    def periodic_x(self) -> bool:
        return self.x_lo in (BCType.PERIODIC, BCType.BLOCH)

    @property
    def periodic_y(self) -> bool:
        return self.y_lo in (BCType.PERIODIC, BCType.BLOCH)

    @property
    def bloch_x(self) -> bool:
        return self.x_lo == BCType.BLOCH

    @property
    def bloch_y(self) -> bool:
        return self.y_lo == BCType.BLOCH

    @property
    def has_bloch_phase(self) -> bool:
        """Whether a bloch axis carries k != 0 (the FDTD engines then step a second, imaginary field set)."""
        return (self.bloch_x and self.bloch_kx != 0.0) or (self.bloch_y and self.bloch_ky != 0.0)

    def bloch_phase_angles(self, period_x, period_y):
        """Wrap phases (kx Lx, ky Ly) [rad] for the periods [m] (the meshed grid lengths); 0 on an axis that is not
        bloch."""
        return (float(self.bloch_kx) * float(period_x) if self.bloch_x else 0.0,
                float(self.bloch_ky) * float(period_y) if self.bloch_y else 0.0)

    @property
    def npml_sides(self):
        """Domain padding cells per face: cpml_npml on cpml faces, MIRROR_GHOST_CELLS on mirror faces, else 0."""
        sides = {}
        for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'):
            bc_type = getattr(self, face)
            sides[face] = (self.cpml_npml if bc_type == BCType.CPML
                           else MIRROR_GHOST_CELLS if bc_type in MIRROR_TYPES else 0)
        return sides

    @property
    def ghost_faces(self):
        """Mirror faces, whose padding cell copies the window edge cell's width (core.mesh cell_spacings)."""
        return tuple(face for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi') if getattr(self, face) in MIRROR_TYPES)

    @classmethod
    def all_pml(cls, npml: int = 15):
        bc = cls('cpml', 'cpml', 'cpml', 'cpml')
        bc.cpml_npml = npml
        return bc

    @classmethod
    def all_pec(cls):
        return cls('pec', 'pec', 'pec', 'pec')

    @classmethod
    def all_pmc(cls):
        return cls('pmc', 'pmc', 'pmc', 'pmc')

    @classmethod
    def with_periodic_x(cls, npml_y: int = 15):
        bc = cls('periodic', 'periodic', 'cpml', 'cpml')
        bc.cpml_npml = npml_y
        return bc

    @classmethod
    def with_periodic_y(cls, npml_x: int = 15):
        bc = cls('cpml', 'cpml', 'periodic', 'periodic')
        bc.cpml_npml = npml_x
        return bc

    @classmethod
    def periodic_xy(cls):
        return cls('periodic', 'periodic', 'periodic', 'periodic')

    @classmethod
    def with_bloch_y(cls, npml_x: int = 15, ky: float = 0.0):
        """Bloch along y with wave vector ky [rad/m] (k = 0 is plain periodic), CPML along x."""
        bc = cls('cpml', 'cpml', 'bloch', 'bloch')
        bc.cpml_npml = npml_x
        bc.bloch_ky = float(ky)
        return bc

    def __repr__(self):
        return (f"BoundaryConfig(x_lo='{self.x_lo.value}', x_hi='{self.x_hi.value}', "
                f"y_lo='{self.y_lo.value}', y_hi='{self.y_hi.value}')")

    def differences(self, other) -> tuple:
        """Names of the settings that differ from `other` in a fixed order (('type',) when it is no BoundaryConfig);
        CPML settings count only when either needs CPML, bloch_kx / bloch_ky only when either has that Bloch axis."""
        if not isinstance(other, BoundaryConfig):
            return ('type',)
        names = ['x_lo', 'x_hi', 'y_lo', 'y_hi']
        if self.needs_cpml or other.needs_cpml:
            names += ['cpml_npml', 'cpml_order', 'cpml_sigma_factor', 'cpml_kappa_max', 'cpml_alpha_max']
        if self.bloch_x or other.bloch_x:
            names.append('bloch_kx')
        if self.bloch_y or other.bloch_y:
            names.append('bloch_ky')
        return tuple(name for name in names if getattr(self, name) != getattr(other, name))

    def __eq__(self, other):
        if not isinstance(other, BoundaryConfig):
            return NotImplemented
        return not self.differences(other)

    # Value equality on mutable settings: instances are unhashable.
    __hash__ = None


def solver_boundary_config(domain, boundary_config, solver_label):
    """boundary_config, else domain.boundary_config, else all-CPML of the domain's equal non-zero padding; ValueError
    (naming solver_label, e.g. 'EMPhysics') when none fits, it fails BoundaryConfig._validate (re-run here) or its PML
    cells differ from the domain padding."""
    bc = boundary_config if boundary_config is not None else domain.boundary_config
    if bc is None:
        widths = set(domain.npml_sides.values())
        if len(widths) != 1 or 0 in widths:
            raise ValueError(f"{solver_label} needs an explicit boundary_config for PML padding {domain.npml_sides} "
                             f"(the all-CPML default needs equal non-zero padding on every side)")
        bc = BoundaryConfig.all_pml(int(widths.pop()))
    bc._validate()
    if bc.npml_sides != domain.npml_sides:
        raise ValueError(f"boundary_config PML cells {bc.npml_sides} do not match the domain padding "
                         f"{domain.npml_sides}; build the Domain with the same boundary_config")
    return bc


class BoundaryManager:
    """Applies the per-side BCs of one polarisation ('ez' or 'hz') to one real field set around the Yee kernels;
    periodic / Bloch wraps live in the kernels (Bloch k != 0: the engine's imaginary set takes `partner()`). `dx` = PML
    cell size (CPML grading); dx_arr / dy_arr = graded spacings (None when uniform); xp = the run backend's array module
    (None = the default backend); dtype = the field dtype (float32 default, float64 for the double stepper)."""

    def __init__(self, config: BoundaryConfig, nx: int, ny: int,
                 dx: float, dt: float,
                 dx_arr: np.ndarray = None, dy_arr: np.ndarray = None, polarization: str = 'ez', xp=None,
                 dtype=None):
        _check_polarization(polarization)
        self.config = config
        self.polarization = polarization
        self.xp = array_module_or_default(xp)
        self.dtype = np.dtype(DTYPE if dtype is None else dtype)
        self._grid_args = (nx, ny, dx, dt, dx_arr, dy_arr)

        self._cpml = None
        if config.needs_cpml:
            self._cpml = CPMLSided(nx, ny, config, dx, dt, dx_arr=dx_arr, dy_arr=dy_arr, polarization=polarization,
                                   xp=self.xp, dtype=self.dtype)

    @property
    def cpml(self):
        return self._cpml

    def partner(self):
        """BoundaryManager of the same configuration and grid with its own CPML psi slabs: the imaginary field set of a
        Bloch k != 0 engine (the psi recursion is linear and real, so each set needs its own history)."""
        nx, ny, dx, dt, dx_arr, dy_arr = self._grid_args
        return BoundaryManager(self.config, nx, ny, dx, dt, dx_arr=dx_arr, dy_arr=dy_arr,
                               polarization=self.polarization, xp=self.xp, dtype=self.dtype)

    # ---- E_z polarisation (E_z at cell centres, H_x on hi-y faces, H_y on hi-x faces) ----------------------------
    # Engine order per step: H kernel, apply_h_post, E kernel, apply_e_post; each post runs the fused CPML psi update
    # + correction first (from the other field, untouched by the kernel), then the outer-face rows (apply_h_walls /
    # apply_ez_walls), which the engine applies again after injecting its sources.
    # Mirrors (inner face of the outermost cell = the window edge, the outermost cell being the padded ghost; E parity
    # as in the H_z block below): symmetric ez[0] = ez[1] (the pmc copy, so the Yee update of the tangential H on the
    # plane vanishes), antisymmetric ez[0] = -ez[1] (E = 0 on the plane; the H face there keeps its Yee update, which
    # differences 2 ez[1]); hi faces alike with rows -1 / -2.

    def apply_h_post(self, ez, hx, hy, db):
        """CPML psi update from E_z + correction of H_x / H_y (db = the engine's live H coefficient), then the
        outer-face rows."""
        if self._cpml:
            self._cpml.apply_h(ez, hx, hy, db)
        self.apply_h_walls(hx, hy)

    def apply_h_walls(self, hx, hy):
        """Tangential H = 0 on the pmc planes (H_y on x faces, H_x on y faces): the inner face of the outermost cell
        (face 0 / n-2, as apply_exy_walls), where apply_ez_walls copies the E_z row."""
        c = self.config

        if c.x_lo == BCType.PMC:
            hy[0, :] = 0
        if c.x_hi == BCType.PMC:
            hy[-2, :] = 0
        if c.y_lo == BCType.PMC:
            hx[:, 0] = 0
        if c.y_hi == BCType.PMC:
            hx[:, -2] = 0

    def apply_e_post(self, hx, hy, ez, cb):
        """CPML psi update from H_x / H_y + correction of E_z (cb = the engine's live E coefficient), then the
        outer-face rows (apply_ez_walls)."""
        if self._cpml:
            self._cpml.apply_e(hx, hy, ez, cb)
        self.apply_ez_walls(ez)

    def apply_ez_walls(self, ez):
        """Outer-face E_z rows: pec (E_z = 0), pmc and symmetric (E_z even: the neighbour's row copied), antisymmetric
        (E_z odd: copied negated); the H face on each plane keeps its Yee update, which vanishes for an even row."""
        c = self.config

        if c.x_lo == BCType.PEC:
            ez[0, :] = 0
        if c.x_hi == BCType.PEC:
            ez[-1, :] = 0
        if c.y_lo == BCType.PEC:
            ez[:, 0] = 0
        if c.y_hi == BCType.PEC:
            ez[:, -1] = 0

        if c.x_lo in _EZ_EVEN_PLANE_TYPES:
            ez[0, :] = ez[1, :]
        if c.x_hi in _EZ_EVEN_PLANE_TYPES:
            ez[-1, :] = ez[-2, :]
        if c.y_lo in _EZ_EVEN_PLANE_TYPES:
            ez[:, 0] = ez[:, 1]
        if c.y_hi in _EZ_EVEN_PLANE_TYPES:
            ez[:, -1] = ez[:, -2]

        # In-place negation (out=) avoids a per-step temporary. A corner cell takes both of its faces' copies; each
        # overwrites the whole row / column from the already pinned neighbour, so the order does not matter.
        xp = self.xp
        if c.x_lo == BCType.ANTISYMMETRIC:
            xp.negative(ez[1, :], out=ez[0, :])
        if c.x_hi == BCType.ANTISYMMETRIC:
            xp.negative(ez[-2, :], out=ez[-1, :])
        if c.y_lo == BCType.ANTISYMMETRIC:
            xp.negative(ez[:, 1], out=ez[:, 0])
        if c.y_hi == BCType.ANTISYMMETRIC:
            xp.negative(ez[:, -2], out=ez[:, -1])

    # ---- H_z polarisation (H_z at cell centres, E_x on hi-y faces, E_y on hi-x faces) ----------------------------
    # Engine order per step: H_z kernel, apply_hz_post, E kernel, apply_exy_post; each post runs the fused CPML psi
    # update + correction first, then the outer-face BCs (apply_hz_walls / apply_exy_walls), which the engine applies
    # again after injecting its sources, so a source sample on a BC-pinned row or face has no effect.
    # Mirror convention (E-field parity, both polarisations): 'symmetric' = E mirror-even = PMC plane (H_z odd),
    # 'antisymmetric' = E mirror-odd = PEC plane (H_z even, tangential E = 0); the plane is the inner face of the
    # outermost (ghost) cell = the window edge, on any face, the same plane as for E_z.
    # Walls: H_z pec = tangential E = 0 on the inner face of the outermost cell (H_z mirrored into that cell); H_z pmc =
    # H_z = 0 at the outermost cell centre. E_z pec sits at the outermost centre and E_z pmc on the inner face, so pec
    # (pmc) walls of the two polarisations are half a cell apart.

    def apply_hz_post(self, ex, ey, hz, db):
        """CPML psi update from E_x / E_y + correction of H_z (db = the engine's live H coefficient), then the
        outer-face rows (apply_hz_walls)."""
        if self._cpml:
            self._cpml.apply_hz(ex, ey, hz, db)
        self.apply_hz_walls(hz)

    def apply_hz_walls(self, hz):
        """Outer-face H_z rows: pec / antisymmetric (H_z even), pmc (H_z = 0) and symmetric (H_z odd) faces."""
        c = self.config

        if c.x_lo in _HZ_PEC_PLANE_TYPES:
            hz[0, :] = hz[1, :]
        if c.x_hi in _HZ_PEC_PLANE_TYPES:
            hz[-1, :] = hz[-2, :]
        if c.y_lo in _HZ_PEC_PLANE_TYPES:
            hz[:, 0] = hz[:, 1]
        if c.y_hi in _HZ_PEC_PLANE_TYPES:
            hz[:, -1] = hz[:, -2]

        if c.x_lo == BCType.PMC:
            hz[0, :] = 0
        if c.x_hi == BCType.PMC:
            hz[-1, :] = 0
        if c.y_lo == BCType.PMC:
            hz[:, 0] = 0
        if c.y_hi == BCType.PMC:
            hz[:, -1] = 0

        # In-place negation (out=) avoids a per-step temporary.
        xp = self.xp
        if c.x_lo == BCType.SYMMETRIC:
            xp.negative(hz[1, :], out=hz[0, :])
        if c.x_hi == BCType.SYMMETRIC:
            xp.negative(hz[-2, :], out=hz[-1, :])
        if c.y_lo == BCType.SYMMETRIC:
            xp.negative(hz[:, 1], out=hz[:, 0])
        if c.y_hi == BCType.SYMMETRIC:
            xp.negative(hz[:, -2], out=hz[:, -1])

    def apply_exy_post(self, hz, ex, ey, cb_x, cb_y):
        """CPML psi update from H_z + correction of E_x / E_y (cb_x / cb_y = the engine's live face coefficients), then
        the outer faces (apply_exy_walls)."""
        if self._cpml:
            self._cpml.apply_exy(hz, ex, ey, cb_x, cb_y)
        self.apply_exy_walls(ex, ey)

    def apply_exy_walls(self, ex, ey):
        """Tangential E = 0 on the pec / antisymmetric faces (E_y on x faces, E_x on y faces)."""
        c = self.config

        if c.x_lo in _HZ_PEC_PLANE_TYPES:
            ey[0, :] = 0
        if c.x_hi in _HZ_PEC_PLANE_TYPES:
            ey[-2, :] = 0
        if c.y_lo in _HZ_PEC_PLANE_TYPES:
            ex[:, 0] = 0
        if c.y_hi in _HZ_PEC_PLANE_TYPES:
            ex[:, -2] = 0


# ---- CPML grading profiles (single source of truth) ------------------------------------------------------------
# Rows along an axis of n cells (npml = config.cpml_npml; relative depth 0 at the outermost window cell centre):
# lo E rows 1..npml (cell centres): depth (npml-1 .. 0)/npml; hi E rows n-npml-1..n-2: (0 .. npml-1)/npml;
# lo H rows 0..npml-1 (hi faces): (npml-1/2 .. 1/2)/npml; hi H rows n-npml-1..n-2: (1/2 .. npml-1/2)/npml.
# The outermost E rows 0 and n-1 (half a cell inside the grid-edge PMC, both updated) and H row n-1 (that PMC face)
# are never stretched: the layout is the mirror image of itself about the grid centre (row j <-> n-1-j for E, n-2-j
# for H), which mirror planes rely on (a half model is the exact restriction of the full one), and the E and H
# gradings vanish at one plane (a one-cell offset reflects a few %). CPMLSided, operators.yee_operators and em3d's
# cpml_slabs read cpml_axis_profiles: never duplicate this layout.

def _grading_values(depth, sigma_max, kappa_max, alpha_max, order):
    """Float64 CPML sigma [S/m], kappa [1] and alpha [S/m] at relative depths 0..1 into the layer."""
    sigma = sigma_max * depth ** order
    kappa = 1.0 + (kappa_max - 1.0) * depth ** order
    alpha = alpha_max * (1.0 - depth)
    return sigma, kappa, alpha



def _cpml_recursion(sigma, kappa, alpha, dt, xp, dtype=DTYPE):
    """CPML psi recursion coefficients (a, b) as `dtype` (float32 default) `xp` arrays from float64 sigma / kappa /
    alpha rows."""
    b = np.exp(-(sigma / kappa + alpha) * float(dt) / EPS0).astype(dtype)
    denom = sigma * kappa + kappa ** 2 * alpha
    a = np.divide(sigma * (b - 1.0), denom, out=np.zeros_like(denom),
                  where=denom > 1e-30).astype(dtype)
    return xp.asarray(a), xp.asarray(b)


def cpml_side_rows(side, n_cells, npml):
    """(E rows, H rows) slices of one CPML side 'lo' / 'hi' along an axis of n_cells (layout above: lo E 1..npml, H
    0..npml-1; hi E and H n-npml-1..n-2): the one writer of the row layout (profiles, CPMLSided, em3d cpml_slabs)."""
    n, npml = int(n_cells), int(npml)
    if side == 'lo':
        return slice(1, npml + 1), slice(0, npml)
    return slice(n - npml - 1, n - 1), slice(n - npml - 1, n - 1)


def cpml_sigma_max(config, pml_dx):
    """Peak CPML conductivity [S/m] = sigma_factor (order + 1) / (150 pi pml_dx), pml_dx = PML cell size [m]: the
    vacuum optimum (no 1/sqrt(eps_r mu_r); a medium of index n in the PML wants sigma_factor ~1/n)."""
    return config.cpml_sigma_factor * (config.cpml_order + 1) / (150.0 * np.pi * float(pml_dx))


@dataclass(frozen=True, eq=False)
class CPMLAxisProfile:
    """Float64 (n,) CPML sigma [S/m], kappa [1], alpha [S/m] of one axis: E rows at cell centres, H rows at hi faces;
    outside the layers sigma = 0, kappa = 1, alpha = 0."""
    sigma_e: np.ndarray
    kappa_e: np.ndarray
    alpha_e: np.ndarray
    sigma_h: np.ndarray
    kappa_h: np.ndarray
    alpha_h: np.ndarray


def cpml_axis_profiles(config, n_cells, axis, pml_dx):
    """CPMLAxisProfile of axis 'x' or 'y' (n_cells rows) with the CPML sides of `config` graded (layout above)."""
    if axis not in ('x', 'y'):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    sides = config.cpml_sides
    lo_active, hi_active = sides[f'{axis}_lo'], sides[f'{axis}_hi']
    npml = int(config.cpml_npml)
    n = int(n_cells)
    n_layers = int(lo_active) + int(hi_active)
    if n_layers and n < n_layers * npml + 2:
        raise ValueError(f"{axis} axis: {n} cells cannot hold {n_layers} CPML layer(s) of {npml} cells plus the 2 "
                         f"unstretched rows; it needs at least {n_layers * npml + 2} cells")
    sigma_e, kappa_e, alpha_e = np.zeros(n), np.ones(n), np.zeros(n)
    sigma_h, kappa_h, alpha_h = np.zeros(n), np.ones(n), np.zeros(n)
    # Axes without CPML sides (periodic, pec, ...; operators.yee_operators asks for them too) keep sigma = 0,
    # kappa = 1, alpha = 0 and never evaluate pml_dx.
    sigma_max = cpml_sigma_max(config, pml_dx) if n_layers else 0.0

    def graded(depth):
        return _grading_values(depth, sigma_max, config.cpml_kappa_max, config.cpml_alpha_max, config.cpml_order)

    if lo_active:
        e_rows, h_rows = cpml_side_rows('lo', n, npml)
        sigma_e[e_rows], kappa_e[e_rows], alpha_e[e_rows] = graded(
            np.arange(npml - 1, -1, -1, dtype=np.float64) / npml)
        sigma_h[h_rows], kappa_h[h_rows], alpha_h[h_rows] = graded(
            (np.arange(npml, 0, -1, dtype=np.float64) - 0.5) / npml)
    if hi_active:
        e_rows, h_rows = cpml_side_rows('hi', n, npml)
        sigma_e[e_rows], kappa_e[e_rows], alpha_e[e_rows] = graded(
            np.arange(0, npml, dtype=np.float64) / npml)
        sigma_h[h_rows], kappa_h[h_rows], alpha_h[h_rows] = graded(
            (np.arange(1, npml + 1, dtype=np.float64) - 0.5) / npml)
    return CPMLAxisProfile(sigma_e=sigma_e, kappa_e=kappa_e, alpha_e=alpha_e,
                           sigma_h=sigma_h, kappa_h=kappa_h, alpha_h=alpha_h)


def _as_column(values):
    """Contiguous (n, 1) view of a 1D row set, broadcasting over an x-side slab (npml, ny)."""
    return array_module_of(values).ascontiguousarray(values).reshape(-1, 1)


def _as_row(values):
    """Contiguous (1, n) view of a 1D row set, broadcasting over a y-side slab (nx, npml)."""
    return array_module_of(values).ascontiguousarray(values).reshape(1, -1)


@dataclass(frozen=True, eq=False)
class _CPMLRowSet:
    """One CPML side's rows of one Yee kind (cell-centre 'e' or hi-face 'h'): broadcast-ready psi recursion b / a and
    curl inverse spacing, the 2D slab index, the 2D source indices of its difference (upper - lower sample: 'e' rows
    r - (r - 1), 'h' rows (r + 1) - r), the 1D rows, host kappa (curl fold) and the psi slab (run-backend arrays; all in
    the CPML dtype)."""
    b: object
    a: object
    inv_spacing: object
    index: tuple
    upper: tuple
    lower: tuple
    rows: slice
    kappa: np.ndarray
    psi: object


@dataclass(frozen=True, eq=False)
class _CPMLSide:
    """One active CPML side: axis 'x' / 'y', correction sign of this polarisation, cell-centre (e) and hi-face (h)
    row sets."""
    axis: str
    sign: np.floating
    e: _CPMLRowSet
    h: _CPMLRowSet


def _cpml_row_set(profile, kind, rows, inverse_spacing, dt, axis, psi_shape, xp, dtype):
    """_CPMLRowSet of `kind` ('e' / 'h') on `rows` of one side in `dtype`; inverse_spacing = float64 Yee row of that
    kind."""
    sigma, kappa, alpha = (getattr(profile, f'{name}_{kind}')[rows] for name in ('sigma', 'kappa', 'alpha'))
    a, b = _cpml_recursion(sigma, kappa, alpha, dt, xp, dtype)
    shift = -1 if kind == 'e' else 1   # the other sample of the difference: e rows read r - 1, h rows r + 1
    other = slice(rows.start + shift, rows.stop + shift)
    upper, lower = (rows, other) if kind == 'e' else (other, rows)

    def along(row_slice):
        return (row_slice, slice(None)) if axis == 'x' else (slice(None), row_slice)
    broadcast = _as_column if axis == 'x' else _as_row
    return _CPMLRowSet(b=broadcast(b), a=broadcast(a),
                       inv_spacing=broadcast(xp.asarray(inverse_spacing[rows], dtype=dtype)), index=along(rows),
                       upper=along(upper), lower=along(lower), rows=rows, kappa=kappa.astype(dtype),
                       psi=xp.zeros(psi_shape, dtype=dtype))


class CPMLSided:
    """Convolutional PML of one polarisation on the CPML sides: psi slabs and recursions per side; kappa folds into
    the curl spacings through build_inv_*. `dx` = PML cell size of the sigma grading (both axes); xp = the run
    backend's array module (None = the default backend); dtype = the field dtype (float32 default, float64)."""

    def __init__(self, nx: int, ny: int, config: BoundaryConfig,
                 dx: float, dt: float,
                 dx_arr: np.ndarray = None, dy_arr: np.ndarray = None, polarization: str = 'ez', xp=None,
                 dtype=None):
        _check_polarization(polarization)
        self.xp = xp = array_module_or_default(xp)
        self.dtype = dtype = np.dtype(DTYPE if dtype is None else dtype)
        self.nx = nx
        self.ny = ny
        self.polarization = polarization
        npml = config.cpml_npml
        self.npml = npml
        self.active = config.cpml_sides

        # Every CPML side is padded toward cells of size `dx` (Domain.from_mesh: dx_base), so x and y share one sigma
        # grading; the actual (max_ratio-graded) Yee spacings come from yee_grid like the kernels' (uniform: dx).
        profiles = {'x': cpml_axis_profiles(config, nx, 'x', dx), 'y': cpml_axis_profiles(config, ny, 'y', dx)}
        spacings = yee_inverse_spacings(np.full(nx, float(dx)) if dx_arr is None else dx_arr,
                                        np.full(ny, float(dx)) if dy_arr is None else dy_arr)
        # psi signs: ez H_y += / H_x -= and E_z += dHy/dx - dHx/dy (x +, y -); hz the opposite (x -, y +).
        plus, minus = (dtype.type(sign) for sign in _SIGNS)
        x_sign, y_sign = (plus, minus) if polarization == 'ez' else (minus, plus)
        # Rows: cpml_side_rows (the profiles' layout). psi slabs: x sides (npml, ny), y sides (nx, npml).
        self._sides = []
        self._launch_arguments = {}   # (row kind, ids of the field / coefficient arrays) -> per-side kernel arguments
        for side_name, active in self.active.items():
            if not active:
                continue
            axis = side_name[0]
            e_rows, h_rows = cpml_side_rows(side_name[2:], nx if axis == 'x' else ny, npml)
            psi_shape = (npml, ny) if axis == 'x' else (nx, npml)
            self._sides.append(_CPMLSide(
                axis=axis, sign=x_sign if axis == 'x' else y_sign,
                e=_cpml_row_set(profiles[axis], 'e', e_rows, getattr(spacings, f'inv_d{axis}_e'), dt, axis, psi_shape,
                                xp, dtype),
                h=_cpml_row_set(profiles[axis], 'h', h_rows, getattr(spacings, f'inv_d{axis}_h'), dt, axis,
                                psi_shape, xp, dtype)))

    def _update_apply(self, row_kind, x_arrays, y_arrays):
        """Fused psi update + field correction on every side's `row_kind` rows; *_arrays = (source, coefficient, field)
        of the x / y sides. The per-side views are built once per set of arrays (fields and Yee coefficients are only
        rewritten in place; a replaced array changes the key) instead of 4 slicings per side and call."""
        key = (row_kind,) + tuple(id(array) for array in x_arrays + y_arrays)
        cached = self._launch_arguments.get(key)
        if cached is None:
            if len(self._launch_arguments) >= 4:   # 2 row kinds x 2 Bloch field sets; else stale (replaced arrays)
                self._launch_arguments.clear()
            arguments = []
            for side in self._sides:
                source, coefficient, field = x_arrays if side.axis == 'x' else y_arrays
                row_set = side.e if row_kind == 'e' else side.h
                arguments.append((row_set.b, row_set.a, source[row_set.upper], source[row_set.lower],
                                  row_set.inv_spacing, side.sign, coefficient[row_set.index], row_set.psi,
                                  field[row_set.index]))
            cached = self._launch_arguments[key] = (x_arrays + y_arrays, arguments)   # the arrays keep the ids valid
        for arguments in cached[1]:
            _psi_update_apply(*arguments)

    def apply_h(self, ez, hx, hy, db):
        """psi of dEz/dx, dEz/dy (H-row coefficients), then H_y += db psi_hxy, H_x -= db psi_hyx on the CPML face rows
        (db = the H Yee coefficient); one fused launch per side."""
        self._update_apply('h', (ez, db, hy), (ez, db, hx))

    def apply_e(self, hx, hy, ez, cb):
        """psi of dHy/dx, dHx/dy (E-row coefficients), then E_z += cb (psi_ezy - psi_ezx) on the CPML cell-centre rows
        (cb = the E Yee coefficient); one fused launch per side."""
        self._update_apply('e', (hy, cb, ez), (hx, cb, ez))

    # ---- H_z polarisation: mu dHz/dt = dEx/dy - dEy/dx, eps_x dEx/dt = dHz/dy, eps_y dEy/dt = -dHz/dx ----------

    def apply_hz(self, ex, ey, hz, db):
        """psi of dEy/dx (x sides) and dEx/dy (y sides) at the H_z cell-centre rows (E-row coefficients), then
        H_z += db (psi_y - psi_x) (db = the H Yee coefficient); one fused launch per side."""
        self._update_apply('e', (ey, db, hz), (ex, db, hz))

    def apply_exy(self, hz, ex, ey, cb_x, cb_y):
        """psi of dHz/dx (E_y, x sides) and dHz/dy (E_x, y sides) at the hi-face rows (H-row coefficients), then
        E_y -= cb_y psi_ey, E_x += cb_x psi_ex (cb_* = the face Yee coefficients); one fused launch per side."""
        self._update_apply('h', (hz, cb_y, ey), (hz, cb_x, ex))

    # ---- kappa coordinate stretch: the PML curl uses 1/(kappa d) instead of 1/d, folded into the full-length 1D
    # inverse spacings the Yee kernels index (kappa = 1 in the bulk); cell-centre rows take kappa_e, hi-face rows
    # kappa_h, from the same rows as the psi coefficients.

    def _fold_kappa(self, axis, row_kind, bulk_inverse_spacing):
        """Run-backend copy in the CPML dtype of the 1D bulk inverse spacing of `axis` with kappa of `row_kind` rows
        divided in."""
        inv = np.array(bulk_inverse_spacing, dtype=self.dtype, copy=True)
        for side in self._sides:
            if side.axis == axis:
                row_set = side.e if row_kind == 'e' else side.h
                inv[row_set.rows] /= row_set.kappa
        return self.xp.asarray(inv)

    def build_inv_dx_e(self, bulk_inv_dx: np.ndarray):
        return self._fold_kappa('x', 'e', bulk_inv_dx)

    def build_inv_dx_h(self, bulk_inv_dx: np.ndarray):
        return self._fold_kappa('x', 'h', bulk_inv_dx)

    def build_inv_dy_e(self, bulk_inv_dy: np.ndarray):
        return self._fold_kappa('y', 'e', bulk_inv_dy)

    def build_inv_dy_h(self, bulk_inv_dy: np.ndarray):
        return self._fold_kappa('y', 'h', bulk_inv_dy)

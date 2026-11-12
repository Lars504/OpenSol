"""Full-vector cross-section eigenmodes (A5): transverse-H finite differences on the projected 3D Yee cell, solved on
the host by scipy shift-invert (never cupy here) on a SuperLU factor or a given one (a solvers/line_block factor:
prepare_cross_section_modes / solve_prepared_modes / line_block_layout). xy = cross-section, fields ~ exp(j omega t - j
beta z), lossy Im n_eff < 0."""
# Layout (3D Yee cell projected; material sampled at E_z = the cell centre): E_z (i, j); H_x, E_y on the hi-y face
# (i, j+1/2); H_y, E_x on the hi-x face (i+1/2, j); H_z on the corner (i+1/2, j+1/2). Every derivative is an A1
# operator (operators.yee_operators): dx_h / dy_h centre -> hi face, dx_e / dy_e hi face -> centre, per axis.
# Generalised eigenproblem A h = beta^2 B h, h = eta0 [H_x; H_y] (Fallahkhair, Li, Murphy, JLT 2008), k0 = omega / C0,
# diagonal mu (mxx at H_x, myy at H_y, mzz at H_z):
#   A_xx = k0^2 mxx + Dy^h ezz^-1 Dy^e + eyy^-1 Dx^e mzz^-1 Dx^h mxx    A_xy = -Dy^h ezz^-1 Dx^e + eyy^-1 Dx^e mzz^-1 Dy^h myy
#   A_yx = -Dx^h ezz^-1 Dy^e + exx^-1 Dy^e mzz^-1 Dx^h mxx             A_yy = k0^2 myy + Dx^h ezz^-1 Dx^e + exx^-1 Dy^e mzz^-1 Dy^h myy
#   B = diag(eyy^-1 at the H_x points, exx^-1 at the H_y points). ezz and mu = cell values; exx / eyy = width-weighted
#   face means (yee_grid.face_average along x / y); eps^-1 = 0 on PEC cells / faces between two PEC cells (B singular: those
#   rows are constraints with infinite beta, theta = 0, never selected by which='LM').
# Recovery: h_z = (Dx^h mxx h_x + Dy^h myy h_y) / (j beta mzz) (discrete div B = 0; Dx^h, Dy^h commute), then
#   E_x = exx^-1 (Dy^e h_z + j beta h_y) / (j k0), E_y = eyy^-1 (-j beta h_x - Dx^e h_z) / (j k0),
#   E_z = ezz^-1 (Dx^e h_y - Dy^e h_x) / (j k0), H = h / eta0.
# Bend about the y axis (symmetry 2.19), s = 1 + (x - x_ref) / R = r / R at each Yee point's own x: 'exact' = the
# straight guide of eps' = eps diag(s, s, 1/s), mu' = mu diag(s, s, 1/s) (transformation optics of (r, y, R phi));
# fields are the mapped components (E'_x = E_r, E'_y = E_y, E'_z = s E_phi, H alike), so power and overlaps keep the
# straight formulas and beta = nu / R. 'conformal' = eps x s^2 on all three components, mu untouched (approximate).
# Walls (pec / pmc / symmetric / antisymmetric): the plane is the inner face of the outermost cell, which is a ghost
# (its material is replaced by its inner neighbour's). Centre-indexed quantities along that axis (along x: h_x, E_y,
# E_z) obey ghost = s x inner neighbour (s = -1 pec / antisymmetric, +1 pmc / symmetric) through a column map on
# Dx^h / Dy^h; plane-face unknowns are free (s = -1) or zero (s = +1, dropped); faces beyond the plane are dropped.
# Half a cell off the FDTD ez pec / hz pmc walls. A natural (cpml) end drops its last face (the never-updated FDTD
# face: PMC half a cell beyond the outermost centres, as the A1 'e' row 0 at the lo end).
# Bloch k on a periodic axis (em_modes, the em_fdtd_3d ports, conical): the operators phase their wrap entries, F(x + L)
# = F(x) e^{-j k L}; walls live only on non-periodic axes, so the column maps are unchanged; face-to-centre means across
# the wrap take bloch_wrap_factors. Callers that keep k = 0 refuse it themselves (EME: operators.check_zero_bloch).
# Conformal PEC (pec_cuts, solver_defaults.conformal_boundaries; Dey-Mittra on the projected Yee cell): only Faraday's
# Dx^h / Dy^h change. On E_z (-> H_y / H_x) a cut centre link uses its free length l = t d (E_z = 0 on the drawn wall);
# on E_y, mu h_x (-> H_z) and E_x, mu h_y the circulation takes the edges' free lengths over the dual cell's free area
# (Qx = A^-1 Dx^h diag(t_y), Qy = A^-1 Dy^h diag(t_x), A the free area fraction). Qx Pz_y = Qy Pz_x still holds, so
# the div B = 0 elimination of h_z and its recovery stay exact; Ampere (D^e, eps^-1, B) is unchanged.
import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ...core.constants import C0, ETA0
from .boundaries import BCType, cpml_axis_profiles
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver, device_factor_matvec
from ...solvers.sparse_direct import size_issue
from .operators import bloch_wrap_phases, pml_stretch, yee_operators
from .slab_modes import _transverse_wavenumber, peak_cell
from .yee_grid import face_average, face_partner, lower_face_partner, yee_inverse_spacings

_WALL_SIGNS = {BCType.PEC: -1.0, BCType.ANTISYMMETRIC: -1.0, BCType.PMC: 1.0, BCType.SYMMETRIC: 1.0}
_MIRROR_TYPES = frozenset({BCType.SYMMETRIC, BCType.ANTISYMMETRIC})
_SHIFT_RELATIVE_OFFSET = 1e-6   # sigma this far above (k0 n_guess)^2: never exactly on an eigenvalue (TEM modes)
_PML_MODE_FRACTION = 0.5        # modes with more of their |h|^2 dA energy on PML-stretched points are dropped
_POWER_FLOOR = 1e-6             # |power| below this share of sum |E||H| dA: bilinear (evanescent) normalisation
_RANK_DECIMALS = 12             # Re beta^2 / k0^2 equal to this many decimals ranks as a tie (then by Im)
_THETA_FLOOR = 1e-13            # |theta| below this share of max |theta|: B-null (PEC) eigenvector, skipped
ROUNDING_RESIDUE_RTOL = 1e-12   # assemble_cross_section: operator entries below this share of their row max dropped
_START_VECTOR_SEED = 4          # ARPACK start vector when the caller gives none
_GROUP_MIN_OVERLAP = 0.9        # mode_group_indices: a tracked mode below this overlap gives n_group NaN
GROUP_RELATIVE_STEP = 1e-3      # mode_group_indices: solves at omega (1 +- step)
# ARPACK tolerance floor of a solve that drops PML modes on a cross-section with CPML: those clustered, non-normal
# eigenvalues restart slowly at machine precision (em_modes_cpu_gpu: the 300-restart cap, 12 of 14 converged); a
# wanted mode keeps |d beta^2| <= floor |beta^2 - shift| (n_eff ~1e-12, n_group by +-GROUP_RELATIVE_STEP ~1e-9)
PML_ARPACK_TOLERANCE_FLOOR = 1e-12
# Arnoldi basis per wanted eigenpair of such a solve (as em3d's ARNOLDI_BASIS_PER_WANTED, fem_bor): scipy's 2 k + 1
# basis restarted ~85 applications per basis vector on the PML clusters (em_modes_cpu_gpu: 2479, the 300-restart cap)
PML_ARNOLDI_BASIS_PER_WANTED = 5
LINE_BLOCK_BOTH_PERIODIC_TEXT = ('the line_block shifted solver cannot take two periodic axes (a periodic wrap along the '
                                 'line axis is not block-tridiagonal); use the sparse_direct solver (Automatic)')
_SIZE_REMEDY = ('coarsen the mesh or shrink the cross-section (a symmetric / antisymmetric mirror face halves it per '
                'symmetry plane)')
BEND_MODELS = ('exact', 'conformal')   # bend = (radius, reference_x, x_edges, model): module header


@dataclass(frozen=True, eq=False)
class CrossSectionMaterials:
    """Host cell values at one omega: complex128 (nx, ny) relative eps_xx, eps_yy, eps_zz (MaterialGrid eps_r) and mu
    (sigma and poles included, Im < 0 lossy), bool (nx, ny) pec; eps_point_deltas = None or (nx, ny) real arrays added
    to the E_x face means, E_y face means and E_z cell values before the bend map (subpixel smoothing, igor 1.5);
    pec_cuts = None or core.cut_cells.dual_cell_cuts (link_x, link_y, area) float64 (nx, ny) (conformal PEC)."""
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_zz: np.ndarray
    mu: np.ndarray
    pec: np.ndarray
    eps_point_deltas: tuple = None
    pec_cuts: tuple = None

    def __post_init__(self):
        shape = np.shape(self.eps_zz)
        if len(shape) != 2 or 0 in shape:
            raise ValueError(f"CrossSectionMaterials arrays must be non-empty (nx, ny), got eps_zz shape {shape}")
        if self.eps_point_deltas is not None and (len(self.eps_point_deltas) != 3 or any(
                np.shape(delta) != shape for delta in self.eps_point_deltas)):
            raise ValueError(f"CrossSectionMaterials eps_point_deltas must be three {shape} arrays (E_x faces, E_y "
                             f"faces, E_z cells)")
        if self.pec_cuts is not None and (len(self.pec_cuts) != 3 or any(
                np.shape(cut) != shape or not np.all(np.asarray(cut) > 0.0) for cut in self.pec_cuts)):
            raise ValueError(f"CrossSectionMaterials pec_cuts must be three positive {shape} arrays (x links, y links, "
                             f"dual-cell areas)")
        for name in ('eps_xx', 'eps_yy', 'mu', 'pec'):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"CrossSectionMaterials {name} shape {np.shape(getattr(self, name))} differs from "
                                 f"eps_zz shape {shape}")
        pec = np.asarray(self.pec)
        if pec.dtype != np.bool_:
            raise TypeError(f"CrossSectionMaterials pec must be a bool array, got dtype {pec.dtype}")
        for name in ('eps_xx', 'eps_yy', 'eps_zz', 'mu'):
            values = np.asarray(getattr(self, name))[~pec]
            if not (np.all(np.isfinite(values)) and np.all(values != 0)):
                raise ValueError(f"CrossSectionMaterials {name} must be finite and non-zero on non-PEC cells")


@dataclass(frozen=True, eq=False)
class CrossSectionMode:
    """One mode, fields ~ exp(-j beta z) as complex128 (nx, ny) on their Yee points (E [V/m], H [A/m]) carrying |power|
    = 1 W (the whole mode when mirrors cut the domain); ghost / beyond-wall points hold the mirror images."""
    beta: complex               # [1/m] at the bend reference x: Re > 0 when Re beta^2 > 0, else Im < 0
    n_eff: complex              # beta / k0
    omega: float                # [rad/s]
    ex: np.ndarray              # hi-x faces (i+1/2, j)
    ey: np.ndarray              # hi-y faces (i, j+1/2)
    ez: np.ndarray              # cell centres
    hx: np.ndarray              # hi-y faces
    hy: np.ndarray              # hi-x faces
    hz: np.ndarray              # corners (i+1/2, j+1/2)
    power: float                # 1/2 Re sum (E x H*).z dA [W] of the whole mode: +-1, ~0 when evanescent
    te_fraction: float          # sum |E_x|^2 dA / sum (|E_x|^2 + |E_y|^2) dA (Lumerical convention)
    effective_area: float       # [m^2] (sum |E|^2 dA)^2 / sum |E|^4 dA at the cell centres, whole mode
    pml_fraction: float         # share of the |h|^2 dA energy on PML-stretched points
    x_face_area: np.ndarray     # (nx, ny) [m^2] dual area of the E_x / H_y points inside the walls (power sums)
    y_face_area: np.ndarray     # (nx, ny) [m^2] dual area of the E_y / H_x points inside the walls
    mirror_factor: float        # 2 per mirror face: the domain holds 1 / mirror_factor of the mode
    n_group: complex | None = None


@dataclass(frozen=True, eq=False)
class CrossSectionSolve:
    """Modes by descending Re beta^2 (fewer than asked may return) and solve info (sizes, timings, counts, warnings)."""
    modes: list
    info: dict


@dataclass(frozen=True, eq=False)
class AxisLayout:
    """Per-axis unknown set and integration lengths: centre / face rows active, lengths [m] times their share inside
    the walls, PML-stretched rows, wall signs (None = no wall) and the (n, n) column map of centre quantities."""
    periodic: bool
    lo_sign: float | None
    hi_sign: float | None
    centre_active: np.ndarray
    face_active: np.ndarray
    centre_length: np.ndarray
    face_length: np.ndarray
    stretched_centres: np.ndarray
    stretched_faces: np.ndarray
    image: object


@dataclass(frozen=True, eq=False)
class CrossSectionLayout:
    """Both axes plus the 2D masks / areas: active over [h_x; h_y] (2 nx ny), areas (nx, ny) [m^2] (x_face_area at
    the E_x / H_y points, y_face_area at E_y / H_x, centre_area at E_z), stretched_hx / _hy at E_y / E_x too."""
    x: AxisLayout
    y: AxisLayout
    active: np.ndarray
    x_face_area: np.ndarray
    y_face_area: np.ndarray
    centre_area: np.ndarray
    stretched_hx: np.ndarray
    stretched_hy: np.ndarray
    mirror_factor: float


@dataclass(frozen=True, eq=False)
class CrossSectionSystem:
    """Assembled full-grid operators and material diagonals (flat i*ny + j) of one omega."""
    a_matrix: object
    b_diagonal: np.ndarray
    dx_e: object
    dy_e: object
    dx_h: object        # with the wall column map
    dy_h: object
    inv_exx: np.ndarray
    inv_eyy: np.ndarray
    inv_ezz: np.ndarray
    mu_x: np.ndarray    # at the H_x points (bend-mapped)
    mu_y: np.ndarray    # at the H_y points
    mu_z: np.ndarray    # at the H_z points
    k0: float
    wrap_factors: tuple = (1.0, 1.0)   # e^{+j k L} of the x / y Bloch wraps (lo-face partner of the first cell)


@dataclass(frozen=True, eq=False)
class CrossSectionProblem:
    """One assembled shift-invert problem (prepare_cross_section_modes) before its factor: host scipy operator A - shift
    B and diagonal B on the active unknowns (active_index into [h_x; h_y]), the shift (k0 guess)^2 (1 + offset)."""
    omega: float
    system: CrossSectionSystem
    layout: CrossSectionLayout
    active_index: np.ndarray
    shape: tuple
    operator: object
    b_active: object
    shift: complex
    guess: complex
    warnings: tuple
    assembly_seconds: float


def size_issue_for(nx, ny):
    """(level, text) of the host SuperLU size rule for 2 nx ny unknowns (sparse_direct.size_issue), None when fine; CPU."""
    return size_issue('scipy_splu', nx, ny, False, False, unknowns_per_cell=2, remedy=_SIZE_REMEDY)


def _axis_layout(boundary_config, axis, widths, face_widths, pml_dx):
    """AxisLayout of axis 'x' / 'y' from its boundary faces (module header rules); ValueError when the walls leave no
    cell between their planes."""
    import scipy.sparse
    n_cells = widths.size
    periodic = boundary_config.periodic_x if axis == 'x' else boundary_config.periodic_y
    profile = cpml_axis_profiles(boundary_config, n_cells, axis, pml_dx)
    centre_active = np.ones(n_cells, dtype=bool)
    face_active = np.ones(n_cells, dtype=bool)
    centre_share = np.ones(n_cells)
    face_share = np.ones(n_cells)
    image_columns = np.arange(n_cells)
    image_values = np.ones(n_cells)
    lo_sign = hi_sign = None
    if not periodic:
        lo_sign = _WALL_SIGNS.get(getattr(boundary_config, f'{axis}_lo'))
        hi_sign = _WALL_SIGNS.get(getattr(boundary_config, f'{axis}_hi'))
        n_ghosts = int(lo_sign is not None) + int(hi_sign is not None)
        if n_cells < n_ghosts + 1:
            raise ValueError(f"{axis} axis: {n_cells} cells leave no cell between the wall planes (the outermost cell "
                             f"of every pec / pmc / mirror face is a ghost outside its plane)")
        face_active[-1] = False             # beyond the hi plane, or the never-updated last face of a natural end
        face_share[-1] = 0.0
        if lo_sign is not None:
            centre_active[0], centre_share[0], face_share[0] = False, 0.0, 0.5
            face_active[0] = lo_sign < 0
            image_columns[0], image_values[0] = 1, lo_sign
        if hi_sign is not None:
            centre_active[-1], centre_share[-1], face_share[-2] = False, 0.0, 0.5
            face_active[-2] = hi_sign < 0
            image_columns[-1], image_values[-1] = n_cells - 2, hi_sign
    image = scipy.sparse.csr_matrix((image_values, (np.arange(n_cells), image_columns)), shape=(n_cells, n_cells))
    return AxisLayout(periodic=periodic, lo_sign=lo_sign, hi_sign=hi_sign, centre_active=centre_active,
                       face_active=face_active, centre_length=widths * centre_share,
                       face_length=face_widths * face_share, stretched_centres=profile.sigma_e > 0.0,
                       stretched_faces=profile.sigma_h > 0.0, image=image)


def cross_section_layout(cell_widths, boundary_config, pml_dx):
    """CrossSectionLayout of the grid and boundary faces (also the conical operator's)."""
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    spacings = yee_inverse_spacings(dx_cells, dy_cells)
    x_axis = _axis_layout(boundary_config, 'x', dx_cells, 1.0 / spacings.inv_dx_h, pml_dx)
    y_axis = _axis_layout(boundary_config, 'y', dy_cells, 1.0 / spacings.inv_dy_h, pml_dx)
    hx_active = np.outer(x_axis.centre_active, y_axis.face_active)     # h_x: x centre, y face
    hy_active = np.outer(x_axis.face_active, y_axis.centre_active)     # h_y: x face, y centre
    n_mirrors = sum(getattr(boundary_config, face) in _MIRROR_TYPES for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'))
    return CrossSectionLayout(
        x=x_axis, y=y_axis, active=np.concatenate((hx_active.ravel(), hy_active.ravel())),
        x_face_area=np.outer(x_axis.face_length, y_axis.centre_length),
        y_face_area=np.outer(x_axis.centre_length, y_axis.face_length),
        centre_area=np.outer(x_axis.centre_length, y_axis.centre_length),
        stretched_hx=x_axis.stretched_centres[:, None] | y_axis.stretched_faces[None, :],
        stretched_hy=x_axis.stretched_faces[:, None] | y_axis.stretched_centres[None, :],
        mirror_factor=float(2 ** n_mirrors))


def _ghost_filled(cell_array, layout, dtype):
    """Copy of a cell array whose wall ghost rows / columns take their inner neighbour's values."""
    array = np.array(cell_array, dtype=dtype, copy=True)
    for axis, axis_layout in ((0, layout.x), (1, layout.y)):
        view = np.moveaxis(array, axis, 0)
        if axis_layout.lo_sign is not None:
            view[0] = view[1]
        if axis_layout.hi_sign is not None:
            view[-1] = view[-2]
    return array


@dataclass(frozen=True, eq=False)
class _BendFactors:
    """(nx,) bend material factors (module header) at the cell centres / hi-x faces: transverse eps_xx / eps_yy,
    longitudinal eps_zz; maps_mu: mu_xx, mu_yy, mu_zz take the same factors (exact model)."""
    centre: np.ndarray
    face: np.ndarray
    z_centre: np.ndarray
    z_face: np.ndarray
    maps_mu: bool


def _x_cell_stretch(boundary_config, n_x, pml_dx, omega):
    """(nx,) complex128 CPML coordinate stretch of the x cells (their E rows; exactly 1 outside the layers), None when
    no x cell is stretched."""
    profile = cpml_axis_profiles(boundary_config, n_x, 'x', pml_dx)
    stretch = pml_stretch(profile.sigma_e, profile.kappa_e, profile.alpha_e, omega)
    return stretch if np.any(stretch != 1.0) else None


def _stretched_x_edges(x_edges, cell_stretch):
    """Complex stretched coordinate x~ = x + integral (s - 1) dx at the cell edges: the CPML layers integrate outward
    from their inner edge (a lo layer toward decreasing x), so x~ = x on the unstretched cells."""
    extra = (np.asarray(cell_stretch, dtype=np.complex128) - 1.0) * np.diff(x_edges)
    cumulative = np.concatenate(([0.0], np.cumsum(extra)))
    unstretched = np.flatnonzero(extra == 0.0)
    # anchor on the median unstretched cell: inside the window, never a layer's unstretched outermost E row
    anchor = int(np.median(unstretched)) if unstretched.size else 0
    return x_edges + cumulative - cumulative[anchor]


def _bend_factors(bend, n_x, cell_stretch=None):
    """_BendFactors of bend = None or (radius, reference_x, x_edges (nx + 1,), model in BEND_MODELS) (all 1 when
    straight or R = 0); ValueError when s = 1 + (x - x_ref) / R <= 0 anywhere on the grid. cell_stretch (nx,) complex
    (_x_cell_stretch): the factors are taken at the CPML-stretched coordinate x~ (analytic continuation, so the layer
    stays matched to the bend's growing index; complex inside the layers)."""
    ones = np.ones(n_x)
    if bend is None:
        return _BendFactors(ones, ones, ones, ones, False)
    if len(bend) != 4:
        raise ValueError(f"bend must be (radius, reference_x, x_edges, model), got {len(bend)} entries")
    radius, reference_x, x_edges, model = bend
    if model not in BEND_MODELS:
        raise ValueError(f"bend model must be one of {', '.join(BEND_MODELS)}, got {model!r}")
    radius, reference_x = float(radius), float(reference_x)
    if radius == 0.0:
        return _BendFactors(ones, ones, ones, ones, False)
    x_edges = np.asarray(x_edges, dtype=np.float64)
    if x_edges.shape != (n_x + 1,):
        raise ValueError(f"bend x_cell_edges must have nx + 1 = {n_x + 1} entries, got shape {x_edges.shape}")
    if not (math.isfinite(radius) and math.isfinite(reference_x)):
        raise ValueError(f"bend radius and reference x must be finite, got {radius!r}, {reference_x!r}")
    edge_scale = 1.0 + (x_edges - reference_x) / radius
    if np.any(edge_scale <= 0.0):
        bad_x = float(x_edges[np.argmin(edge_scale)])
        raise ValueError(f"bend radius {radius:g} m: the bend factor 1 + (x - x_ref)/R reaches "
                         f"{float(np.min(edge_scale)):.3g} <= 0 at x = {bad_x:g} m inside the grid (the centre of "
                         f"curvature x_ref - R = {reference_x - radius:g} m lies in it); use a larger |R|")
    if cell_stretch is not None:
        x_edges = _stretched_x_edges(x_edges, cell_stretch)
        edge_scale = 1.0 + (x_edges - reference_x) / radius
    centre_scale = 1.0 + (0.5 * (x_edges[:-1] + x_edges[1:]) - reference_x) / radius
    face_scale = edge_scale[1:]
    if model == 'conformal':
        return _BendFactors(centre_scale ** 2, face_scale ** 2, centre_scale ** 2, face_scale ** 2, False)
    return _BendFactors(centre_scale, face_scale, 1.0 / centre_scale, 1.0 / face_scale, True)


@dataclass(frozen=True, eq=False)
class ModePointMaterials:
    """The permittivity the solve's E points see (host complex128 (nx, ny)): ghost-filled cell eps_xx / eps_yy /
    eps_zz, E_x hi-x face / E_y hi-y face / E_z centre values (width-weighted face means times the bend factor), bend
    factors (nx,) of eps_xx / eps_yy and of eps_zz at the centres / hi-x faces (1 straight), cell widths and centre
    lengths inside the walls [m]."""
    cell_eps_xx: np.ndarray
    cell_eps_yy: np.ndarray
    cell_eps_zz: np.ndarray
    eps_x_faces: np.ndarray
    eps_y_faces: np.ndarray
    eps_z_centres: np.ndarray
    centre_factor: np.ndarray
    face_factor: np.ndarray
    z_centre_factor: np.ndarray
    z_face_factor: np.ndarray
    dx_cells: np.ndarray
    dy_cells: np.ndarray
    x_centre_length: np.ndarray
    y_centre_length: np.ndarray
    layout: object

    @property
    def periodic(self):
        """(periodic_x, periodic_y)."""
        return self.layout.x.periodic, self.layout.y.periodic

    @property
    def centre_area(self):
        """(nx, ny) [m^2] cell areas inside the walls (0 on wall ghost cells)."""
        return self.layout.centre_area

    def ghost_filled(self, cell_array):
        """Complex copy of a cell array whose wall ghost rows / columns take their inner neighbour's values."""
        return _ghost_filled(cell_array, self.layout, np.complex128)


def _point_materials(cell_widths, layout, materials, factors):
    """ModePointMaterials of CrossSectionMaterials on a CrossSectionLayout with _BendFactors (the eps half of
    assemble_cross_section)."""
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    eps_xx = _ghost_filled(materials.eps_xx, layout, np.complex128)
    eps_yy = _ghost_filled(materials.eps_yy, layout, np.complex128)
    eps_zz = _ghost_filled(materials.eps_zz, layout, np.complex128)
    eps_x_faces = face_average(eps_xx, 0, layout.x.periodic, dx_cells)
    eps_y_faces = face_average(eps_yy, 1, layout.y.periodic, dy_cells)
    eps_z_centres = eps_zz
    if materials.eps_point_deltas is not None:   # subpixel smoothing: averaged - standard eps of the E points
        delta_x, delta_y, delta_z = materials.eps_point_deltas
        eps_x_faces, eps_y_faces, eps_z_centres = eps_x_faces + delta_x, eps_y_faces + delta_y, eps_zz + delta_z
    return ModePointMaterials(
        cell_eps_xx=eps_xx, cell_eps_yy=eps_yy, cell_eps_zz=eps_zz,
        eps_x_faces=eps_x_faces * factors.face[:, None],   # E_x hi-x faces
        eps_y_faces=eps_y_faces * factors.centre[:, None],  # E_y hi-y faces
        eps_z_centres=eps_z_centres * factors.z_centre[:, None], centre_factor=factors.centre, face_factor=factors.face,
        z_centre_factor=factors.z_centre, z_face_factor=factors.z_face, dx_cells=dx_cells, dy_cells=dy_cells,
        x_centre_length=layout.x.centre_length, y_centre_length=layout.y.centre_length, layout=layout)


def mode_point_materials(cell_widths, boundary_config, pml_dx, materials, bend=None):
    """ModePointMaterials the solve_cross_section_modes of these arguments assembles (same ghost fill, face means and
    bend map): the permittivity of the E points for overlaps / perturbation integrals of its modes."""
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    layout = cross_section_layout((dx_cells, dy_cells), boundary_config, pml_dx)
    return _point_materials((dx_cells, dy_cells), layout, materials, _bend_factors(bend, dx_cells.size))


def _inverse_or_zero(values, zero_mask):
    """1 / values, 0 where zero_mask (PEC)."""
    inverse = np.zeros(values.shape, dtype=np.complex128)
    np.divide(1.0, values, out=inverse, where=~zero_mask)
    return inverse


def _wall_neutral_cuts(pec_cuts, layout):
    """Copies of the pec_cuts (link_x, link_y, area) with the links / dual cells crossing a wall plane or the grid end
    set to 1 (staircased there) and the links lying in a wall ghost row / column copied from its inner neighbour (the
    mirror image, so the wall column maps commute with the cut lengths)."""
    link_x, link_y, area = (np.array(cut, dtype=np.float64, copy=True) for cut in pec_cuts)
    for axis, axis_layout in ((0, layout.x), (1, layout.y)):
        along, across = (link_x, link_y) if axis == 0 else (link_y, link_x)
        along_view, across_view, area_view = (np.moveaxis(array, axis, 0) for array in (along, across, area))
        along_view[-1] = 1.0   # no in-grid link beyond the last cell (a periodic seam stays staircased)
        area_view[-1] = 1.0
        if axis_layout.lo_sign is not None:
            along_view[0] = area_view[0] = 1.0
            across_view[0] = across_view[1]
        if axis_layout.hi_sign is not None:
            along_view[-2] = area_view[-2] = 1.0
            across_view[-1] = across_view[-2]
    return link_x, link_y, area


def assemble_cross_section(cell_widths, boundary_config, pml_dx, materials, omega, bend, layout):
    """CrossSectionSystem of the module-header operator at omega (complex PML stretch), wall column maps folded into
    Dx^h / Dy^h (with pec_cuts the conformal Faraday operators of the module header); conical reuses its parts."""
    import scipy.sparse
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    nx, ny = dx_cells.size, dy_cells.size
    ops = yee_operators((dx_cells, dy_cells), boundary_config, pml_dx, omega)   # Bloch: phased wrap entries
    image_x = scipy.sparse.kron(layout.x.image, scipy.sparse.identity(ny), format='csr')
    image_y = scipy.sparse.kron(scipy.sparse.identity(nx), layout.y.image, format='csr')
    dx_h = (ops.dx_h @ image_x).tocsr()
    dy_h = (ops.dy_h @ image_y).tocsr()
    dx_h_centres, dy_h_centres = dx_h, dy_h   # Faraday on E_z (-> H_y / H_x); dx_h / dy_h act on E_y, h_x / E_x, h_y
    if materials.pec_cuts is not None:
        link_x, link_y, area = _wall_neutral_cuts(materials.pec_cuts, layout)
        inverse_area = scipy.sparse.diags(1.0 / area.ravel(), 0, format='csr')
        dx_h_centres = (scipy.sparse.diags(1.0 / link_x.ravel(), 0, format='csr') @ dx_h).tocsr()
        dy_h_centres = (scipy.sparse.diags(1.0 / link_y.ravel(), 0, format='csr') @ dy_h).tocsr()
        dx_h = (inverse_area @ ops.dx_h @ scipy.sparse.diags(link_y.ravel(), 0, format='csr') @ image_x).tocsr()
        dy_h = (inverse_area @ ops.dy_h @ scipy.sparse.diags(link_x.ravel(), 0, format='csr') @ image_y).tocsr()
    dx_e, dy_e = ops.dx_e, ops.dy_e
    pec = _ghost_filled(materials.pec, layout, bool)
    mu = _ghost_filled(materials.mu, layout, np.complex128)
    factors = _bend_factors(bend, nx, None if bend is None else _x_cell_stretch(boundary_config, nx, pml_dx, omega))
    points = _point_materials((dx_cells, dy_cells), layout, materials, factors)
    eps_x_faces, eps_y_faces, eps_zz = points.eps_x_faces, points.eps_y_faces, points.eps_z_centres
    if factors.maps_mu:   # H_x at the centres, H_y / H_z on the hi-x faces (x of each point)
        mu_x, mu_y, mu_z = (mu * factor[:, None] for factor in (factors.centre, factors.face, factors.z_face))
    else:
        mu_x = mu_y = mu_z = mu
    # E_x / E_y sit on faces normal to them: a face is PEC only inside the conductor (both cells), never yee_grid.face_pec
    # (either cell, the H_z engine's tangential rule), which would zero the normal E on a conductor surface.
    inv_ezz = _inverse_or_zero(eps_zz, pec).ravel()
    inv_exx = _inverse_or_zero(eps_x_faces, pec & face_partner(pec, 0, layout.x.periodic)).ravel()
    inv_eyy = _inverse_or_zero(eps_y_faces, pec & face_partner(pec, 1, layout.y.periodic)).ravel()
    mu_x, mu_y, mu_z = (values.ravel() for values in (mu_x, mu_y, mu_z))
    k0 = float(omega) / C0

    def diagonal(values):
        return scipy.sparse.diags(values, 0, format='csr')

    mu_x_matrix, mu_y_matrix, inv_mu_z = diagonal(mu_x), diagonal(mu_y), diagonal(1.0 / mu_z)
    inv_ezz_matrix, inv_exx_matrix, inv_eyy_matrix = diagonal(inv_ezz), diagonal(inv_exx), diagonal(inv_eyy)
    a_xx = (k0 ** 2 * mu_x_matrix + dy_h_centres @ inv_ezz_matrix @ dy_e
            + inv_eyy_matrix @ dx_e @ inv_mu_z @ dx_h @ mu_x_matrix)
    a_xy = -(dy_h_centres @ inv_ezz_matrix @ dx_e) + inv_eyy_matrix @ dx_e @ inv_mu_z @ dy_h @ mu_y_matrix
    a_yx = -(dx_h_centres @ inv_ezz_matrix @ dy_e) + inv_exx_matrix @ dy_e @ inv_mu_z @ dx_h @ mu_x_matrix
    a_yy = (k0 ** 2 * mu_y_matrix + dx_h_centres @ inv_ezz_matrix @ dx_e
            + inv_exx_matrix @ dy_e @ inv_mu_z @ dy_h @ mu_y_matrix)
    a_matrix = scipy.sparse.bmat([[a_xx, a_xy], [a_yx, a_yy]], format='csr')
    # a_xy / a_yx vanish exactly where eps_yy = eps_zz and mu is uniform (x / y differences commute on a tensor grid),
    # but with dx != dy the two products round apart: ~1e-16 residue couples h_x, h_y everywhere and inflates the
    # SuperLU fill (graded rib: 1.6x fill per unknown); a real interface coupling is ~delta eps / eps relative
    row_max = np.asarray(abs(a_matrix).max(axis=1).todense()).ravel()
    entry_rows = np.repeat(np.arange(a_matrix.shape[0]), np.diff(a_matrix.indptr))
    a_matrix.data[np.abs(a_matrix.data) <= ROUNDING_RESIDUE_RTOL * row_max[entry_rows]] = 0.0
    a_matrix.eliminate_zeros()
    return CrossSectionSystem(a_matrix=a_matrix, b_diagonal=np.concatenate((inv_eyy, inv_exx)), dx_e=dx_e, dy_e=dy_e,
                   dx_h=dx_h, dy_h=dy_h, inv_exx=inv_exx, inv_eyy=inv_eyy, inv_ezz=inv_ezz, mu_x=mu_x, mu_y=mu_y,
                   mu_z=mu_z, k0=k0, wrap_factors=bloch_wrap_factors((dx_cells, dy_cells), boundary_config))


def bloch_wrap_factors(cell_widths, boundary_config):
    """(e^{+j k_x L_x}, e^{+j k_y L_y}) complex: the phasor one period below index 0 over the stored last value (1 off
    the Bloch axes); the wrap_factor of to_centres / centre_fields."""
    phase_x, phase_y = bloch_wrap_phases(cell_widths, boundary_config)
    return complex(np.exp(1j * phase_x)), complex(np.exp(1j * phase_y))


def _default_index(materials):
    """max Re sqrt(eps mu) over the non-PEC cells and the three permittivity components (the straight structure)."""
    usable = ~np.asarray(materials.pec)
    if not np.any(usable):
        raise ValueError("solve_cross_section_modes: every cell is PEC")
    mu = np.asarray(materials.mu, dtype=np.complex128)[usable]
    return max(float(np.max(np.real(np.sqrt(np.asarray(eps, dtype=np.complex128)[usable] * mu))))
               for eps in (materials.eps_xx, materials.eps_yy, materials.eps_zz))


def _start_vector(start_vector, active_index, shape):
    """ARPACK v0 on the active unknowns: None, a flat array of their count, or the summed eta0 [H_x; H_y] of a
    CrossSectionMode / sequence of them (None when that sum vanishes)."""
    if start_vector is None:
        return None
    if isinstance(start_vector, CrossSectionMode):
        start_vector = [start_vector]
    if isinstance(start_vector, (list, tuple)) and all(isinstance(mode, CrossSectionMode) for mode in start_vector):
        total = np.zeros(active_index.size, dtype=np.complex128)
        for mode in start_vector:
            if mode.hx.shape != shape:
                raise ValueError(f"start_vector mode shape {mode.hx.shape} differs from the grid {shape}")
            total += ETA0 * np.concatenate((mode.hx.ravel(), mode.hy.ravel()))[active_index]
        return total if np.any(total != 0) else None
    vector = np.asarray(start_vector, dtype=np.complex128).ravel()
    if vector.size != active_index.size:
        raise ValueError(f"start_vector needs {active_index.size} entries (info['n_unknowns']), got {vector.size}")
    return vector


def fill_images(field, layout, x_centred, y_centred):
    """In place: wall ghost (centre-indexed) and beyond-plane (face-indexed) points take their mirror images."""
    for axis, axis_layout, centred in ((0, layout.x, x_centred), (1, layout.y, y_centred)):
        view = np.moveaxis(field, axis, 0)
        if centred:
            if axis_layout.lo_sign is not None:
                view[0] = axis_layout.lo_sign * view[1]
            if axis_layout.hi_sign is not None:
                view[-1] = axis_layout.hi_sign * view[-2]
        elif axis_layout.hi_sign is not None and view.shape[0] >= 3:
            view[-1] = -axis_layout.hi_sign * view[-3]   # face parity = -s (pec faces even, pmc faces odd)


def _pml_fraction(layout, hx, hy):
    """Share of sum |h|^2 dA on PML-stretched h_x / h_y points (1 for a vanishing vector)."""
    weight_x = np.abs(hx) ** 2 * layout.y_face_area
    weight_y = np.abs(hy) ** 2 * layout.x_face_area
    total = float(np.sum(weight_x) + np.sum(weight_y))
    if total <= 0.0:
        return 1.0
    return float(np.sum(weight_x[layout.stretched_hx]) + np.sum(weight_y[layout.stretched_hy])) / total


def to_centres(face_values, axis, periodic, wrap_factor=1.0):
    """Plain mean of each cell's hi and lo face values along axis (yee_grid.lower_face_partner convention; wrap_factor
    = e^{+j k L} on a Bloch axis)."""
    return 0.5 * (face_values + lower_face_partner(face_values, axis, periodic, wrap_factor=wrap_factor))


def _build_mode(system, layout, hx, hy, beta, omega):
    """Normalised CrossSectionMode of the full-grid h_x, h_y (nx, ny) at beta: fields, images, 1 W, phase, metrics."""
    shape = hx.shape
    hx_flat, hy_flat = hx.ravel(), hy.ravel()
    j_beta, j_k0 = 1j * beta, 1j * system.k0
    hz = (system.dx_h @ (system.mu_x * hx_flat) + system.dy_h @ (system.mu_y * hy_flat)) / (j_beta * system.mu_z)
    fields = {
        'ex': system.inv_exx * (system.dy_e @ hz + j_beta * hy_flat) / j_k0,
        'ey': system.inv_eyy * (-j_beta * hx_flat - system.dx_e @ hz) / j_k0,
        'ez': system.inv_ezz * (system.dx_e @ hy_flat - system.dy_e @ hx_flat) / j_k0,
        'hx': hx_flat / ETA0, 'hy': hy_flat / ETA0, 'hz': hz / ETA0}
    # (x centred, y centred) of each component, for the ghost / beyond-wall images
    centring = {'ex': (False, True), 'ey': (True, False), 'ez': (True, True), 'hx': (True, False),
                'hy': (False, True), 'hz': (False, False)}
    for name in fields:
        fields[name] = np.array(fields[name].reshape(shape), dtype=np.complex128)
        fill_images(fields[name], layout, *centring[name])
    area_x, area_y, mirror = layout.x_face_area, layout.y_face_area, layout.mirror_factor

    def cross_power(conjugate):
        hx_values = np.conj(fields['hx']) if conjugate else fields['hx']
        hy_values = np.conj(fields['hy']) if conjugate else fields['hy']
        return 0.5 * mirror * complex(np.sum(fields['ex'] * hy_values * area_x - fields['ey'] * hx_values * area_y))

    power = cross_power(True).real
    magnitude_scale = 0.5 * mirror * float(np.sum(np.abs(fields['ex'] * fields['hy']) * area_x
                                                  + np.abs(fields['ey'] * fields['hx']) * area_y))
    norm = abs(power) if abs(power) >= _POWER_FLOOR * magnitude_scale else abs(cross_power(False))
    if norm > 0.0:
        scale = 1.0 / math.sqrt(norm)
    else:
        peak_e = max(float(np.max(np.abs(fields[name]))) for name in ('ex', 'ey', 'ez'))
        scale = 1.0 / peak_e if peak_e > 0.0 else 1.0
    energy_x = float(np.sum(np.abs(fields['ex']) ** 2 * area_x))
    energy_y = float(np.sum(np.abs(fields['ey']) ** 2 * area_y))
    reference, reference_area = (fields['ex'], area_x) if energy_x >= energy_y else (fields['ey'], area_y)
    masked = np.where(reference_area > 0.0, reference, 0.0).ravel()
    phase = 1.0
    if np.any(masked != 0):
        peak_value = masked[peak_cell(masked)]
        phase = np.conj(peak_value) / abs(peak_value)
    for name in fields:
        fields[name] *= scale * phase
    wrap_x, wrap_y = system.wrap_factors
    centre_intensity = (np.abs(to_centres(fields['ex'], 0, layout.x.periodic, wrap_x)) ** 2
                        + np.abs(to_centres(fields['ey'], 1, layout.y.periodic, wrap_y)) ** 2
                        + np.abs(fields['ez']) ** 2)
    quartic = float(np.sum(centre_intensity ** 2 * layout.centre_area))
    effective_area = (mirror * float(np.sum(centre_intensity * layout.centre_area)) ** 2 / quartic
                      if quartic > 0.0 else float('nan'))
    transverse_energy = energy_x + energy_y
    return CrossSectionMode(
        beta=complex(beta), n_eff=complex(beta / system.k0), omega=float(omega), power=cross_power(True).real,
        te_fraction=energy_x / transverse_energy if transverse_energy > 0.0 else 0.0,
        effective_area=effective_area, pml_fraction=_pml_fraction(layout, hx, hy), x_face_area=area_x,
        y_face_area=area_y, mirror_factor=mirror, **fields)


def solve_cross_section_modes(cell_widths, boundary_config, pml_dx, materials, omega, n_modes, n_eff_guess=None,
                              bend=None, start_vector=None, eigen_solver=None, drop_pml_modes=True):
    """CrossSectionSolve of up to n_modes modes found by shift-invert near (k0 n_eff_guess)^2 (default max Re sqrt(eps
    mu)), ranked by descending Re beta^2, PML modes dropped unless drop_pml_modes is False (EME keeps them as
    radiation); cell_widths = (dx_cells, dy_cells) [m], pml_dx = CPML grading cell [m]; bend: None or (radius [m],
    reference_x [m], full-grid x edges, model in BEND_MODELS); eigen_solver: solvers.arpack_shift_invert runtime
    (None = default settings)."""
    problem = prepare_cross_section_modes(cell_widths, boundary_config, pml_dx, materials, omega,
                                          n_eff_guess=n_eff_guess, bend=bend)
    return solve_prepared_modes(problem, n_modes, start_vector=start_vector, eigen_solver=eigen_solver,
                                drop_pml_modes=drop_pml_modes)


def prepare_cross_section_modes(cell_widths, boundary_config, pml_dx, materials, omega, n_eff_guess=None, bend=None,
                                check_size=True):
    """CrossSectionProblem of solve_cross_section_modes (assembly, active unknowns, shifted operator) before any factor;
    check_size: the host SuperLU size rule (a line-block factor has its own)."""
    started = time.perf_counter()
    import scipy.sparse
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"solve_cross_section_modes needs a finite angular frequency omega > 0 [rad/s], got {omega!r}")
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    shape = (dx_cells.size, dy_cells.size)
    if np.shape(materials.eps_zz) != shape:
        raise ValueError(f"materials shape {np.shape(materials.eps_zz)} does not match the {shape[0]} x {shape[1]} "
                         f"cell widths")
    warnings = []
    issue = size_issue_for(*shape) if check_size else None
    if issue is not None:
        if issue[0] == 'error':
            raise ValueError(issue[1])
        warnings.append(issue[1])
    layout = cross_section_layout((dx_cells, dy_cells), boundary_config, pml_dx)
    system = assemble_cross_section((dx_cells, dy_cells), boundary_config, pml_dx, materials, omega, bend, layout)
    active_index = np.flatnonzero(layout.active)
    n_unknowns = int(active_index.size)
    if n_unknowns < 4:
        raise ValueError(f"solve_cross_section_modes: only {n_unknowns} unknowns remain inside the walls")
    a_active = system.a_matrix[active_index][:, active_index]
    b_active = scipy.sparse.diags(system.b_diagonal[active_index], 0, format='csr')
    guess = _default_index(materials) if n_eff_guess is None else complex(n_eff_guess)
    if not (np.isfinite(guess) and guess != 0):
        raise ValueError(f"n_eff_guess must be finite and non-zero, got {n_eff_guess!r}")
    shift = (system.k0 * guess) ** 2 * (1.0 + _SHIFT_RELATIVE_OFFSET)
    return CrossSectionProblem(omega=omega, system=system, layout=layout, active_index=active_index, shape=shape,
                               operator=(a_active - shift * b_active).tocsr(), b_active=b_active, shift=shift,
                               guess=guess, warnings=tuple(warnings), assembly_seconds=time.perf_counter() - started)


def solve_prepared_modes(problem, n_modes, start_vector=None, eigen_solver=None, drop_pml_modes=True,
                         factorization=None):
    """CrossSectionSolve of a CrossSectionProblem (solve_cross_section_modes' ranking and PML rule) on `factorization`
    of problem.operator (solve_host, host_bytes, description: e.g. a line-block factor; None = host SuperLU); ARPACK
    at machine precision, PML_ARPACK_TOLERANCE_FLOOR when PML modes are dropped on a cross-section with CPML."""
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
        raise ValueError(f"n_modes must be an integer >= 1, got {n_modes!r}")
    started = time.perf_counter()
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    system, layout, active_index, shape = problem.system, problem.layout, problem.active_index, problem.shape
    omega, shift, guess, b_active = problem.omega, problem.shift, problem.guess, problem.b_active
    n_unknowns = int(active_index.size)
    warnings = list(problem.warnings)
    own_factor = factorization is None
    if own_factor:   # host SuperLU, or the eigen solver's cuDSS factorization (description)
        factorization = eigen_solver.factorize_host(problem.operator)
    factor_kind = getattr(factorization, 'description', 'SuperLU' if own_factor else 'external')
    factorized = time.perf_counter()
    n_wanted = min(2 * int(n_modes) + 10, n_unknowns - 2)
    initial = _start_vector(start_vector, active_index, shape)
    if initial is None:   # deterministic (reproducible solves and run times), not ARPACK's process-wide random state
        random = np.random.default_rng(_START_VECTOR_SEED)
        initial = random.standard_normal(n_unknowns) + 1j * random.standard_normal(n_unknowns)
    pml_clusters = drop_pml_modes and bool(layout.stretched_hx.any() or layout.stretched_hy.any())
    theta, vectors, applications, converged = eigen_solver.eigs(   # a gpu runtime: the device Krylov-Schur
        lambda vector: factorization.solve_host(b_active @ vector), n_unknowns, n_wanted, initial,
        tolerance_floor=PML_ARPACK_TOLERANCE_FLOOR if pml_clusters else 0.0,
        default_basis=mode_default_basis(n_wanted, pml_clusters),
        device_matvec=device_factor_matvec(factorization, b_active))
    if not converged:
        warnings.append(f'ARPACK converged {np.size(theta)} of {n_wanted} eigenvalues near n_eff = {guess:.6g}; modes '
                        f'may be missing (set the effective index guess near the wanted mode)')
    eigen_done = time.perf_counter()
    theta = np.asarray(theta, dtype=np.complex128)
    usable = np.isfinite(theta) & (np.abs(theta) > _THETA_FLOOR * float(np.max(np.abs(theta), initial=0.0)))
    beta_squared = shift + 1.0 / theta[usable]
    beta = _transverse_wavenumber(beta_squared)
    vectors = np.asarray(vectors)[:, usable]
    candidates = []
    n_pml_dropped = 0
    n_grid = shape[0] * shape[1]
    for column in range(beta.size):
        if beta[column] == 0:
            continue
        full = np.zeros(2 * n_grid, dtype=np.complex128)
        full[active_index] = vectors[:, column]
        hx, hy = full[:n_grid].reshape(shape), full[n_grid:].reshape(shape)
        if drop_pml_modes and _pml_fraction(layout, hx, hy) > _PML_MODE_FRACTION:
            n_pml_dropped += 1
            continue
        candidates.append((beta_squared[column], beta[column], hx, hy))
    values = np.array([candidate[0] for candidate in candidates], dtype=np.complex128)
    order = np.lexsort((-values.imag, -np.round(values.real / system.k0 ** 2, _RANK_DECIMALS)))[:int(n_modes)]
    modes = [_build_mode(system, layout, candidates[index][2], candidates[index][3], candidates[index][1], omega)
             for index in order]
    finished = time.perf_counter()
    info = {'n_unknowns': n_unknowns, 'factor_host_bytes': factorization.host_bytes or 0,
            'assembly_seconds': problem.assembly_seconds, 'factor_seconds': factorized - started,
            'eigen_seconds': eigen_done - factorized, 'seconds': problem.assembly_seconds + finished - started,
            'operator_applications': applications, 'eigenvalues_requested': n_wanted,
            'eigenvalues_converged': int(theta.size), 'pml_modes_dropped': n_pml_dropped,
            'shift_index': guess, 'factor_kind': factor_kind, 'arnoldi': eigen_solver.last_arnoldi,
            'warnings': warnings}
    return CrossSectionSolve(modes=modes, info=info)


def mode_default_basis(n_wanted, pml_clusters):
    """Arnoldi default_basis of a cross-section mode solve (CPU, also the estimators): PML_ARNOLDI_BASIS_PER_WANTED x
    n_wanted when PML modes are dropped on a CPML cross-section, else 0 (scipy's)."""
    return PML_ARNOLDI_BASIS_PER_WANTED * int(n_wanted) if pml_clusters else 0


def cpml_cross_section(settings_values):
    """True when an em_modes settings dict puts a cpml on any outer face (estimators: mode_default_basis)."""
    return any(settings_values.get(f'bc_{face}') == 'cpml' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'))


def line_block_layout(problem):
    """solvers.line_block LineBlockLayout of a problem's active [h_x; h_y] unknowns: blocks = grid rows j (block_axis
    0, x inside) or columns i (1); the periodic axis inside the blocks, else the cheaper n_blocks b^3; ValueError with
    two periodic axes (a wrap along the line axis is not block-tridiagonal)."""
    from ...solvers.line_block import LineBlockLayout
    nx, ny = problem.shape
    column, row = np.divmod(problem.active_index % (nx * ny), ny)   # i (x), j (y) of every active unknown
    axis = line_block_axis(nx, ny, problem.layout.x.periodic, problem.layout.y.periodic)
    return LineBlockLayout(row if axis == 0 else column, block_axis=axis)


def line_block_axis(nx, ny, periodic_x, periodic_y):
    """Grid axis inside a cross-section line block: the periodic one, else the cheaper n_blocks b^3 (b = 2 x the cells
    across); ValueError for two periodic axes (CPU, also the estimators)."""
    if periodic_x and periodic_y:
        raise ValueError(LINE_BLOCK_BOTH_PERIODIC_TEXT)
    if periodic_x != periodic_y:
        return 0 if periodic_x else 1
    return 0 if ny * float(nx) ** 3 < nx * float(ny) ** 3 else 1


def line_block_bytes(nx, ny, periodic_x, periodic_y, precision, n_operators=1):
    """Device bytes of a factorize_operators batch of n_operators cross-section shift-invert operators (2 unknowns per
    cell, ~9 CSR entries per row; CPU, the memory estimate; two periodic axes: the cheaper axis)."""
    from ...solvers.line_block import operator_shift_bytes, shift_bytes
    nx, ny = int(nx), int(ny)
    axis = 1 if periodic_x and periodic_y else line_block_axis(nx, ny, periodic_x, periodic_y)
    n_blocks, block_size = (ny, 2 * nx) if axis == 0 else (nx, 2 * ny)
    n_unknowns = 2 * nx * ny
    per_operator = (operator_shift_bytes(n_blocks, block_size, n_unknowns, 9 * n_unknowns, precision)
                    + shift_bytes(n_blocks, block_size, n_unknowns, 1, precision, kept=True))
    return int(n_operators) * per_operator


def centre_fields(mode, periodic_x, periodic_y, wrap_factors=(1.0, 1.0)):
    """Dict ex, ey, ez, hx, hy, hz of the mode collocated to the cell centres (plain face means, complex128 (nx, ny));
    wrap_factors = bloch_wrap_factors of a Bloch mode."""
    wrap_x, wrap_y = wrap_factors
    return {'ex': to_centres(mode.ex, 0, periodic_x, wrap_x), 'ey': to_centres(mode.ey, 1, periodic_y, wrap_y),
            'ez': np.array(mode.ez, copy=True), 'hx': to_centres(mode.hx, 1, periodic_y, wrap_y),
            'hy': to_centres(mode.hy, 0, periodic_x, wrap_x),
            'hz': to_centres(to_centres(mode.hz, 0, periodic_x, wrap_x), 1, periodic_y, wrap_y)}


def mode_overlap(mode_a, mode_b):
    """|c_ab c_ba| / |c_aa c_bb| with c_ab = 1/2 sum (E_a x H_b*).z dA: 1 for a mode with itself, ~0 between distinct
    lossless modes (power orthogonality), the power coupling fraction of normalised modes."""
    if mode_a.ex.shape != mode_b.ex.shape:
        raise ValueError(f"mode_overlap: grids differ ({mode_a.ex.shape} vs {mode_b.ex.shape})")
    area_x, area_y = mode_a.x_face_area, mode_a.y_face_area

    def cross(first, second):
        return complex(np.sum(first.ex * np.conj(second.hy) * area_x - first.ey * np.conj(second.hx) * area_y))

    denominator = abs(cross(mode_a, mode_a) * cross(mode_b, mode_b))
    if denominator == 0.0:
        return 0.0
    return abs(cross(mode_a, mode_b) * cross(mode_b, mode_a)) / denominator


def mode_group_indices(cell_widths, boundary_config, pml_dx, materials_at, omega, modes, n_eff_guess=None, bend=None,
                       relative_step=GROUP_RELATIVE_STEP, n_modes=None, eigen_solver=None, solve_at=None):
    """(copies of modes with n_group = C0 d beta / d omega, info): central difference over materials_at(omega (1 +-
    step)) -> CrossSectionMaterials, modes as ARPACK v0 and tracked by mode_overlap (NaN below 0.9); solve_at(omega,
    n_modes, start_vector) -> CrossSectionSolve replaces solve_cross_section_modes (ModePhysics line-block batches)."""
    relative_step = float(relative_step)
    if not 0.0 < relative_step < 1.0:
        raise ValueError(f"relative_step must be in (0, 1), got {relative_step!r}")
    modes = list(modes)
    if not modes:
        return [], {'solves': [], 'warnings': []}
    omega = float(omega)
    n_solve = len(modes) + 2 if n_modes is None else int(n_modes)
    tracked_betas, solve_infos, warnings = [], [], []
    for step in (relative_step, -relative_step):
        shifted_omega = omega * (1.0 + step)
        if solve_at is not None:
            solve = solve_at(shifted_omega, n_solve, modes)
        else:
            solve = solve_cross_section_modes(cell_widths, boundary_config, pml_dx, materials_at(shifted_omega),
                                              shifted_omega, n_solve, n_eff_guess=n_eff_guess, bend=bend,
                                              start_vector=modes, eigen_solver=eigen_solver)
        solve_infos.append(solve.info)
        betas = []
        for number, mode in enumerate(modes):
            overlaps = [mode_overlap(mode, candidate) for candidate in solve.modes]
            best = int(np.argmax(overlaps)) if overlaps else -1
            if best < 0 or overlaps[best] < _GROUP_MIN_OVERLAP:
                warnings.append(f'mode {number + 1} (n_eff = {mode.n_eff:.6g}) not tracked at omega x {1 + step:g} '
                                f'(best overlap {max(overlaps, default=0.0):.3g} < {_GROUP_MIN_OVERLAP}): n_group NaN')
                betas.append(complex(float('nan'), float('nan')))
            else:
                betas.append(solve.modes[best].beta)
        tracked_betas.append(betas)
    with_group = [replace(mode, n_group=complex(C0 * (beta_plus - beta_minus) / (2.0 * relative_step * omega)))
                  for mode, beta_plus, beta_minus in zip(modes, *tracked_betas)]
    return with_group, {'solves': solve_infos, 'warnings': warnings}

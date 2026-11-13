"""3D Yee FDTD engine of em_fdtd_3d: six float32 fields, a uint16 material index + per-id coefficient tables
(material_index3d), per-axis 1D curl spacings (CPML kappa folded in), CPML psi on the layer slabs only, dispersive
poles on region boxes and the J.E loss accounting (q_accum) of the EM loss protocol."""
# Layout (cell (i, j, k), flat index (i ny + j) nz + k, z contiguous; the A5 cross-section layout extended along z):
#   E_x on the hi-x face, E_y on the hi-y face, E_z on the hi-z face of the cell (each E normal to its face, material =
#   width-weighted mean of the two cells' eps of that component (eps_xx / eps_yy / eps_r arrays; eps_r alone when the
#   component arrays are not allocated), yee_grid.face_shares; first order for the normal component; subpixel
#   smoothing overrides the eps of interface faces through a uint16 face index, subpixel_smoothing3d);
#   H_x on the (hi y, hi z) edge, H_y on (hi x, hi z), H_z on (hi x, hi y); mu_r and sigma_m of the lower cell (i, j, k).
# Along each axis the profile layout of physics.em.boundaries.cpml_axis_profiles holds: 'e' rows = cell centres (E
# components tangential to that axis' faces), 'h' rows = hi faces. Non-periodic axes: ih[n-1] = 0 and the missing lo
# H of cell 0 reads 0, so the grid edge is PMC half a cell beyond the outermost centres (bc 'pmc' = nothing to do).
# bc 'pec' gives the outermost window cell layer the PEC id: tangential E = 0 at its centres (two-PEC-cell faces).
# Mirrors (symmetry 2.5, E parity as 2D): the domain pads one ghost cell outside the window edge, so the plane is the
# window edge = the hi face of the lo ghost / of the last window cell. symmetric (E even): the tangential H and the
# normal E on that face are pinned to 0; antisymmetric (E odd): the ghost's tangential E = -its inner neighbour's, so
# the plane's tangential H and normal E update with the PEC image. Re-pinned after the H injections and the E sources.
# Step: H Yee with the fused H CPML (+ the sparse Dey-Mittra faces of conformal PEC, conformal_pec3d), source inject_magnetic at t (TFSF / mode-port H corrections), H mirror rows, pole J / P from
# E^n (+ their E^n work half), E Yee with the fused E CPML, pole corrections, E sources at t + dt/2, E mirror rows, loss accounting
# (pole E^{n+1} work half, ohmic sigma E^2), monitors at t + dt (E at t + dt, H at t + dt/2).
# Bloch k != 0 on a periodic axis (symmetry 2.8, as 2D A6): F(x + L) = F(x) e^{-j k L}, L = the meshed period; a second,
# imaginary set fields.imag (own CPML psi and pole states) steps with the same kernels, the wraps mix the two
# (kernels3d _WRAP_LOAD); walls / mirrors act on both, sources drive the real set only, DFT monitors add j ph F_im;
# q_accum is refused.
# Cylindrical (r, phi, z) sectors (symmetry 2.21; a polar Domain3D: x = r - radius_min, y = phi [rad]): the same layout
# read as E_r / E_phi / E_z, H_r / H_phi / H_z; the integral-form curls fold the radii (core.metric.yee_radii) into
# per-i 1D factors of the update kernels (kernels3d header; all 1 on a Cartesian grid), dt is the CFL step of the
# innermost arc (core.grid3d.finest_physical_widths_3d). Annular sectors only (radius_min > 0: no axis cells); r faces
# pec / pmc (pmc = H tangential 0 on the r_lo / r_hi grid face itself) or cpml on r_hi (d/dr stretch only:
# approximate); phi faces pec / pmc / mirrors (rays are mirror planes: C_nv) or periodic on a domain whose angle_wrap
# joins them, with the angular Floquet phase bloch_ky = azimuthal order m [1/rad] (F(phi + span) = F(phi) e^{-j m
# span}, the FDFD polar rule); a phase of 0 or pi mod 2 pi steps the real set alone (c = +-1, s = 0).
# Backends (solver_plan S6): every array lives on the Domain's xp (cupy on 'gpu', numpy on 'cpu') and every kernel is
# a core.kernels twin (kernels3d, the shared 2D pole corrections), so this one code path runs on both.
import math
import warnings
from dataclasses import replace

import numpy as np
from ...core.backend import array_module_of, to_host

from ...core.constants import C0, DTYPE, ETA0
from ...core.grid3d import finest_physical_widths_3d
from ...core.half_storage import half_bits
from ...core.metric import yee_radii
from ...core.monitors import FieldProbe, FieldProbeRows
from ..em.solvers.fdtd_conformal_kernels import sparse_launch
from ..em.solvers.fdtd_ez_2d_kernels import drude_e_correction, lorentz_e_correction
from ..em.materials import DrudePole, LorentzPole
from ..em.yee_grid import face_shares
from .boundaries3d import AXES, MIRROR_TYPES_3D, along, cpml_slabs, mirror_planes, pec_wall_layers
from .kernels3d import (box_face_cb_3d, drude_j_weighted, field_kernels, launch, lorentz_p_weighted, pole_work_half)
from .material_index3d import build_material_index

MAX_MATERIAL_IDS = 65535       # uint16 index; the last id is the PEC entry
# gpu: the fixed-argument launches of a step (H Yee with its CPML + conformal; E Yee with its CPML) replay as two
# CUDA graphs after one eager step per input key (a field kernel launch with ~40 arguments costs ~15 us of host time)
STEP_GRAPHS = True
_MAX_STEP_GRAPHS = 8           # graph keys kept (stale ones belong to replaced tables / scales)
_INDEX_CHUNK_CELLS = 1 << 22   # x-slab size of the material index build (bounds its int64 temporaries)
# MaterialGrid arrays of the index: always eps_r (E_z, and E_x / E_y without their own array), sigma_e, mu_r; the
# optional eps_xx, eps_yy, sigma_m only when the model allocated them (em3d definition: optional_material_schema)
REQUIRED_ARRAYS = ('eps_r', 'sigma_e', 'mu_r')
OPTIONAL_ARRAYS = ('eps_xx', 'eps_yy', 'sigma_m')
COMPONENT_AXES = {'ex': 0, 'ey': 1, 'ez': 2, 'hx': 0, 'hy': 1, 'hz': 2}
_E_PROBE_COMPONENTS = frozenset({'ex', 'ey', 'ez'})   # written after the E graph (probe re-sample)
# stepper precision -> field storage dtype (half_storage: fp16 in memory, float32 arithmetic in the kernels)
FIELD_DTYPES = {'single': np.float32, 'half_storage': np.float16}


_REAL_PHASE_TOLERANCE = 1e-5   # rad: a wrap phase this close to a multiple of pi is real (c = +-1, s = 0); float32
                               # angle widths sum to the sector span within ~1e-6 rad


def _real_phases(angles) -> bool:
    """Whether every wrap phase is 0 or pi mod 2 pi (the real set alone carries the field)."""
    return all(abs(angle / math.pi - round(angle / math.pi)) * math.pi <= _REAL_PHASE_TOLERANCE for angle in angles)


def _wrap_arguments(angles, set_sign, real=False):
    """Kernel wrap scalars of one field set (+1 real, -1 imaginary): H (c, s) per axis of the hi wraps (s = +-sin kL),
    E those of the lo wraps (s = -+sin kL); c = 1, s = 0 on axes without a phase; real: c = +-1 exactly, s = 0."""
    h_args, e_args = [], []
    for angle in angles:
        cosine, sine = (math.cos(angle), math.sin(angle)) if angle else (1.0, 0.0)
        if real:
            cosine, sine = float(round(cosine)), 0.0
        h_args += [DTYPE(cosine), DTYPE(set_sign * sine)]
        e_args += [DTYPE(cosine), DTYPE(-set_sign * sine)]
    return tuple(h_args), tuple(e_args)


def _check_cylindrical(domain, config):
    """ValueError for a cylindrical sector the engine cannot step (module header): radius_min 0, r faces other than
    pec / pmc (+ cpml on r_hi), phi cpml faces, or periodic phi faces without the domain's angle_wrap (or the reverse)."""
    coordinates = domain.coordinates
    if not coordinates.radius_min > 0.0:
        raise ValueError('em_fdtd_3d on a cylindrical sector needs radius_min > 0 (an annular sector): the Yee cells '
                         'have no axis treatment, and the CFL step shrinks with the innermost arc r_min dphi')
    phi_faces = ('pec', 'pmc', 'periodic', *MIRROR_TYPES_3D)
    allowed = {'x_lo': ('pec', 'pmc'), 'x_hi': ('pec', 'pmc', 'cpml'), 'y_lo': phi_faces, 'y_hi': phi_faces}
    refused = [f'{face} {getattr(config, face)}' for face, kinds in allowed.items()
               if getattr(config, face) not in kinds]
    if refused:
        raise ValueError(f"em_fdtd_3d on a cylindrical sector: {', '.join(refused)} is not allowed (r faces pec / pmc, "
                         'cpml only on x_hi = the outer cylinder: an r = const cylinder is no mirror or period; phi '
                         'faces pec / pmc / mirrors / periodic, no cpml)')
    if config.periodic('y') != (coordinates.angle_wrap != 'none'):
        raise ValueError(f"em_fdtd_3d on a cylindrical sector: periodic phi faces (bc y_lo {config.y_lo}) need a domain "
                         f"whose angle_wrap joins them and the reverse (angle_wrap {coordinates.angle_wrap!r}); the EM "
                         'phase is bloch_ky = the azimuthal order m [1/rad]')


def _mirror_rows(domain, config):
    """Wall rows of the mirror faces (module header): [(H name, plane index)] and [(E name, plane index)] zeroed on a
    symmetric face, [(E name, ghost index, inner index)] imaged with -1 on an antisymmetric one."""
    h_zero, e_zero, e_images = [], [], []
    for axis_index, kind, ghost, inner, plane_face in mirror_planes(config, domain.physical_window()):
        axis = AXES[axis_index]
        others = [name for name in AXES if name != axis]
        if kind == 'symmetric':
            h_zero.extend((f'h{name}', along(axis_index, plane_face)) for name in others)
            e_zero.append((f'e{axis}', along(axis_index, plane_face)))
        else:
            e_images.extend((f'e{name}', along(axis_index, ghost), along(axis_index, inner)) for name in others)
    return h_zero, e_zero, e_images


class Fields3D:
    """Six (nx, ny, nz) Yee components (layout in the module header): float32, or float16 under half_storage (stored
    in the engine's field units: FDTDEngine3D.field_scale) of array module xp; `imag` = the imaginary set under Bloch k
    != 0, else None."""

    def __init__(self, shape, dtype, xp):
        for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
            setattr(self, name, xp.zeros(shape, dtype=dtype))
        self.imag = None


class _PoleRegion:
    """Poles of one material region on its box extended one cell to the lo side (or spanning a periodic axis,
    _pole_region): per E component the face weight and live cb (box arrays), Drude J / Lorentz P, dP, face work."""

    def __init__(self, box, weights, window, mask):
        self.box = box              # (slice x, slice y, slice z) of the extended box
        self.weights = weights      # 3 float32 box arrays
        self.window = window        # the region's own window and cropped bool mask (kept alive by the physics)
        self.mask = mask
        self.cb = [array_module_of(weight).zeros_like(weight) for weight in weights]
        self.drude = []             # [(damping, drive, [j_x, j_y, j_z])]
        self.lorentz = []           # [(damping, restoring, drive, [p...], [dp...])]
        self.imag_drude = []        # Bloch k != 0: [j_x, j_y, j_z] of the imaginary field set per drude entry
        self.imag_lorentz = []      # likewise ([p...], [dp...]) per lorentz entry
        self.work = None            # 3 float32 box arrays of face work [J/m^3] since the last deposit (accumulating)


class FDTDEngine3D:
    """Yee stepping on a Domain3D (BoundaryConfig3D faces cpml / pec / pmc / periodic / mirrors); materials from the
    MaterialGrid arrays eps_r, sigma_e, mu_r (+ eps_xx, eps_yy, sigma_m when allocated) through the material index
    (rebuild_materials after any change). precision 'single' | 'half_storage' (fp16 fields, see set_field_scale)."""

    def __init__(self, domain, config, precision='single'):
        if precision not in FIELD_DTYPES:
            raise ValueError(f"em_fdtd_3d stepper precision must be one of {', '.join(FIELD_DTYPES)}, got "
                             f'{precision!r}')
        grid = domain.grid
        self.domain = domain
        self.xp = xp = domain.xp
        self.config = config
        self.precision = precision
        self.kernels = field_kernels(precision)
        # stored field = physical / scale (E [V/m], H [A/m]); 1 in single, set once before the first step otherwise
        self.e_scale = 1.0
        self.h_scale = 1.0
        self.shape = grid.shape
        self.nx, self.ny, self.nz = grid.shape
        self.dt = float(grid.dt)
        self._dt32 = DTYPE(grid.dt)
        self.periodic = tuple(config.periodic(axis) for axis in AXES)
        self.widths = tuple(np.asarray(widths, dtype=np.float64) for widths in (grid.dx_arr, grid.dy_arr, grid.dz_arr))
        # cylindrical sector (module header): YeeRadii of the grid's x cells (no x_lo padding: r_lo is a wall), the
        # per-i curl factors of the kernels (all 1 on a Cartesian grid)
        self.polar = domain.coordinates.polar
        self.radii = None
        if self.polar:
            _check_cylindrical(domain, config)
            self.radii = yee_radii(self.widths[0], domain.coordinates.radius_min)
        self._finest_widths = finest_physical_widths_3d(*self.widths,
                                                        domain.coordinates.radius_min if self.polar else None)
        bulk = []
        for widths, periodic in zip(self.widths, self.periodic):
            inverse_h = 1.0 / (0.5 * (widths + np.roll(widths, -1)))
            if not periodic:
                inverse_h[-1] = 0.0   # the last hi-face row is never updated (non-periodic edge)
            bulk.append((1.0 / widths, inverse_h))
        pml_cell_sizes = tuple(float(max(widths[0], widths[-1])) for widths in self.widths)
        self.cpml, kappa_e, kappa_h = cpml_slabs(config, self.shape, bulk, pml_cell_sizes, self.dt, xp)
        self.inv_e = tuple(xp.asarray((inverse_e / kappa).astype(np.float32))
                           for (inverse_e, _), kappa in zip(bulk, kappa_e))
        self.inv_h = tuple(xp.asarray((inverse_h / kappa).astype(np.float32))
                           for (_, inverse_h), kappa in zip(bulk, kappa_h))
        self.shares = tuple(tuple(xp.asarray(share) for share in face_shares(widths, periodic, np.float32))
                            for widths, periodic in zip(self.widths, self.periodic))
        self._radial_h, self._radial_e = self._radial_factors()
        self.fields = Fields3D(self.shape, FIELD_DTYPES[precision], xp)
        # Bloch k != 0 (symmetry 2.8): the imaginary set (own CPML psi, pole states), wraps mixing the two (kernels3d);
        # a sector whose phases are all 0 or pi steps the real set alone
        self.bloch_angles = config.bloch_phase_angles(tuple(float(np.sum(widths)) for widths in self.widths))
        real_wrap = self.polar and _real_phases(self.bloch_angles)
        self.cpml_imag = []
        if config.has_bloch_phase and not real_wrap:
            self.fields.imag = Fields3D(self.shape, FIELD_DTYPES[precision], xp)
            self.cpml_imag = [replace(layer, psi_first=xp.zeros_like(layer.psi_first),
                                      psi_second=xp.zeros_like(layer.psi_second)) for layer in self.cpml]
        self._wrap_args = {set_sign: _wrap_arguments(self.bloch_angles, set_sign, real=real_wrap)
                           for set_sign in (1, -1)}
        # set sign -> {'h' / 'e': the trailing CPML arguments of update_h_3d / update_e_3d}: fixed for the engine's
        # life (psi rewritten in place), so no step-graph key part
        self._cpml_args = {1: self._cpml_kernel_args(self.cpml), -1: self._cpml_kernel_args(self.cpml_imag)}
        self.material_id = xp.zeros(self.shape, dtype=np.uint16)
        # float32 per material id (last id = PEC: eps -1, sigma 0, mu 1, sigma_m 0)
        self.table_eps_x = self.table_eps_y = self.table_eps_z = None
        self.table_sigma = self.table_mu = self.table_sigma_m = None
        self.quantisation_errors = {}   # array -> max |value - table value| of the re-quantised arrays (last rebuild)
        self._quantisation_deltas = {}  # array -> bin width of the last re-quantisation (the next rebuild's hint)
        self.pec_regions = []        # [(window slices, cropped bool mask)]
        self.pole_regions = []
        self.region_by_mask = {}     # id(cropped mask) -> _PoleRegion (the physics keeps the masks alive)
        self.time = 0.0
        self.n_step = 0
        # J.E accounting (EM loss protocol): q_accum [J/m^3] over the current window (allocated at the first enable),
        # the last completed window, the box of the sigma_e > 0 cells (refreshed per rebuild) and pending pole work
        self._enable_q_accum = False
        self.q_accum = None
        self.q_accum_time = 0.0
        self.last_q_accum = None
        self.last_q_accum_time = 0.0
        self._ohmic_box = None       # (origin, shape, launch) of the conducting cells, None without any
        self._pole_work_pending = False
        self._ints = tuple(np.int32(value) for value in (self.nx, self.ny, self.nz))
        self._periodic_ints = tuple(np.int32(flag) for flag in self.periodic)
        self._grid_launch = launch(self.shape)
        self._mirror_h_zero, self._mirror_e_zero, self._mirror_e_images = _mirror_rows(domain, config)
        self._e_to_h = DTYPE(1.0)    # e_scale / h_scale and its inverse: the curl factors of the stored fields
        self._h_to_e = DTYPE(1.0)
        # (phase 'h' / 'e', ids of the replaceable inputs, curl factor) -> [graph or None after its eager run, kept
        # inputs] (_fixed_block; gpu only); n_step_graphs = captures made
        self.use_step_graphs = STEP_GRAPHS and xp is not np
        self._step_graphs = {}
        self.n_step_graphs = 0
        # gpu: the FieldProbes among the monitors (real set), sampled by one launch at the end of the E graph (taken
        # over at their first step; (id, length) of the monitor list last scanned)
        self._probe_rows = None
        self._probe_scan = None
        # igor 1.5: subpixel_smoothing3d.SubpixelSmoothing3D (EMPhysics3D.set_subpixel_smoothing), refreshed by every
        # rebuild_materials; the kernels read its face index through _face_args (1-element placeholders when off; the
        # face id one is 3D like the real index, else the numba twins indexing fid[i, j, k] fail to type)
        self.subpixel_smoothing = None
        self._no_face_id = xp.zeros((1, 1, 1), dtype=np.uint16)
        self._no_face_table = xp.zeros(3, dtype=DTYPE)
        # solver_plan S7.6: (face cells int64, H components int32, (n, 4) weight - 1 float32, grid, block, int32 count)
        # of the Dey-Mittra H faces (set_conformal_faces), None without conformal PEC
        self._conformal = None

    def _radial_factors(self):
        """Device float32 (r_c, 1 / r_c, 1 / r_link) of the H kernel and (r_f[i + 1], 1 / r_f[i + 1], 1 / r_c) of the
        E kernel (kernels3d header; the r axis' CPML psi reads them too): ones on a Cartesian grid."""
        xp = self.xp

        def device(values):
            return xp.asarray(np.asarray(values, dtype=np.float32))
        if not self.polar:
            ones = device(np.ones(self.nx))
            return (ones,) * 3, (ones,) * 3
        radii = self.radii
        centre, inverse_centre, face_hi = device(radii.centre), device(1.0 / radii.centre), device(radii.face[1:])
        inverse_link = device(1.0 / radii.link)
        return (centre, inverse_centre, inverse_link), (face_hi, device(1.0 / radii.face[1:]), inverse_centre)

    def _cpml_kernel_args(self, layers):
        """{'h' / 'e': the trailing CPML arguments of update_h_3d / update_e_3d}: per axis row map, b, gain, psi first /
        second and int32 psi rows (kernels3d CPML_AXIS_PARAMS); 1-element placeholders and 0 rows on axes without CPML
        (the numba twins type them: 1D int32 / float32, 3D float32 psi)."""
        xp = self.xp
        no_rows, no_values = xp.full(1, -1, dtype=np.int32), xp.zeros(1, dtype=DTYPE)
        no_psi = xp.zeros((1, 1, 1), dtype=DTYPE)
        by_axis_kind = {(layer.axis, layer.kind): layer for layer in layers}
        arguments = {}
        for kind in ('h', 'e'):
            values = []
            for axis in range(3):
                layer = by_axis_kind.get((axis, kind))
                values += ((no_rows, no_values, no_values, no_psi, no_psi, np.int32(0)) if layer is None else
                           (layer.row_map, layer.b, layer.gain, layer.psi_first, layer.psi_second,
                            np.int32(layer.n_rows)))
            arguments[kind] = tuple(values)
        return arguments

    def face_dual_volume(self, axis, cell) -> float:
        """Dual volume [m^3] of the E face along axis of cell (i, j, k): the edge between its two centres times the
        cell face (a sector: E_r r_f[i + 1] dphi dz x dr', E_phi dr dz x r_c dphi', E_z r_c dr dphi x dz')."""
        lengths = []
        for other, widths in enumerate(self.widths):
            index = cell[other]
            # the hi partner: wrapped on a periodic axis (as inverse_h / face_shares), the own cell at a grid edge
            partner = (index + 1) % widths.size if self.periodic[other] else min(index + 1, widths.size - 1)
            lengths.append(0.5 * (widths[index] + widths[partner]) if other == axis else widths[index])
        volume = math.prod(lengths)
        if self.polar:
            volume *= self.radii.face[cell[0] + 1] if axis == 0 else self.radii.centre[cell[0]]
        return float(volume)

    # ---- field units ----
    def set_field_scale(self, e_scale):
        """half_storage: store E / e_scale and H / (e_scale / eta0) (a plane wave's E and H alike in size); only
        before the first step. single keeps 1 (unchanged arithmetic)."""
        if self.precision == 'single':
            return
        if self.n_step:
            raise RuntimeError('the field scale of a half_storage engine is fixed once it has stepped')
        self.e_scale = float(e_scale)
        self.h_scale = self.e_scale / ETA0
        self._e_to_h = DTYPE(self.e_scale / self.h_scale)
        self._h_to_e = DTYPE(self.h_scale / self.e_scale)

    def field_scale(self, component) -> float:
        """Physical value of one stored unit of a field component ('ex' .. 'hz')."""
        return self.e_scale if component[0] == 'e' else self.h_scale

    # ---- materials ----
    @property
    def material_arrays(self) -> tuple:
        """Names of the MaterialGrid arrays the index reads (the optional ones only when allocated)."""
        allocated = self.domain.materials.property_names()
        return REQUIRED_ARRAYS + tuple(name for name in OPTIONAL_ARRAYS if name in allocated)

    def rebuild_materials(self):
        """Material index + tables (material_index3d: continuous values re-quantised to fit the uint16 index, errors
        in quantisation_errors), PEC regions and pec walls, the subpixel face overrides, then the pole cb arrays and
        the ohmic loss box."""
        materials = self.domain.materials
        arrays = {name: getattr(materials, name) for name in self.material_arrays}
        tables, self.quantisation_errors, deltas = build_material_index(
            arrays, self._x_chunks(), self.material_id, MAX_MATERIAL_IDS - 1, self._quantisation_deltas)
        self._quantisation_deltas.update(deltas)
        n_ids = int(tables['eps_r'].size)
        xp = self.xp

        def with_pec(values, pec_value):
            return xp.concatenate((values, xp.full(1, pec_value, dtype=DTYPE))).astype(DTYPE)

        self.table_eps_z = with_pec(tables['eps_r'], -1.0)
        self.table_eps_x = with_pec(tables.get('eps_xx', tables['eps_r']), -1.0)
        self.table_eps_y = with_pec(tables.get('eps_yy', tables['eps_r']), -1.0)
        self.table_sigma = with_pec(tables['sigma_e'], 0.0)
        self.table_mu = with_pec(tables['mu_r'], 1.0)
        self.table_sigma_m = with_pec(tables.get('sigma_m', xp.zeros(n_ids, dtype=DTYPE)), 0.0)
        pec_id = np.uint16(n_ids)
        for window, mask in self.pec_regions:
            xp.copyto(self.material_id[window], pec_id, where=mask)
        for window in self._pec_wall_layers():
            self.material_id[window] = pec_id
        if self.subpixel_smoothing is not None:   # averaged face eps of the new tables (the pole cb reads them)
            self.subpixel_smoothing.refresh(self)
            if self.subpixel_smoothing.quantisation_error is not None:
                self.quantisation_errors['subpixel-smoothed face eps'] = self.subpixel_smoothing.quantisation_error
        self._refresh_pole_cb()
        if self._enable_q_accum:
            self._refresh_ohmic_box()

    def _x_chunks(self):
        per_chunk = max(1, _INDEX_CHUNK_CELLS // (self.ny * self.nz))
        return [(slice(start, min(start + per_chunk, self.nx)),) for start in range(0, self.nx, per_chunk)]

    def _pec_wall_layers(self):
        """Window slices of the outermost window cell layer of every 'pec' face (boundaries3d.pec_wall_layers)."""
        return pec_wall_layers(self.config, self.domain.physical_window())

    def pec_wall_cell(self, cell, axes=(0, 1, 2)):
        """Whether a cell lies in the pec wall layer of a face normal to one of `axes` (tangential E pinned there)."""
        return any(layer[axis].start is not None and layer[axis].start <= cell[axis] < layer[axis].stop
                   for layer in self._pec_wall_layers() for axis in axes)

    def add_pec(self, window, mask):
        """PEC cells on a cropped bool mask (applied by every rebuild_materials)."""
        self.pec_regions.append((window, mask))

    @property
    def pec_id(self) -> int:
        """Material id of the PEC entry (the last table id; valid after rebuild_materials)."""
        return int(self.table_eps_z.shape[0]) - 1

    def set_conformal_faces(self, face_cell, face_axis, face_extra):
        """Dey-Mittra H faces (conformal_pec3d.ConformalPEC3D, host arrays): flat lower-corner cells, H components and
        (n, 4) weight - 1 of their E edges, corrected right after every H Yee update of each field set."""
        count = int(np.asarray(face_cell).size)
        if count == 0:
            self._conformal = None
            return
        xp = self.xp
        self._conformal = (xp.asarray(face_cell, dtype=np.int64), xp.asarray(face_axis, dtype=np.int32),
                           xp.ascontiguousarray(xp.asarray(face_extra, dtype=DTYPE)), *sparse_launch(count))

    def _conformal_h(self, fields):
        """Sparse Dey-Mittra correction of one field set's H (after its Yee H kernel, before the CPML)."""
        face_cell, face_axis, face_extra, grid, block, count = self._conformal
        self.kernels['conformal_h_3d'](grid, block, (
            fields.hx, fields.hy, fields.hz, fields.ex, fields.ey, fields.ez, self.material_id, self.table_mu,
            self.table_sigma_m, *self.inv_h, face_cell, face_axis, face_extra, count, self._ints[1], self._ints[2],
            self._dt32, self._e_to_h))

    def coefficient_args(self):
        """Material and share arguments of box_face_cb_3d after the comp / origin arguments."""
        return (self.material_id, self.table_eps_x, self.table_eps_y, self.table_eps_z, self.table_sigma,
                *self._face_args(), *self.shares[0], *self.shares[1], *self.shares[2], *self._ints,
                *self._periodic_ints, self._dt32)

    def _face_args(self):
        """(face id, face table, int32 use flag) kernel arguments of the subpixel face overrides."""
        smoothing = self.subpixel_smoothing
        if smoothing is None or smoothing.face_id is None:
            return self._no_face_id, self._no_face_table, np.int32(0)
        return smoothing.face_id, smoothing.table, np.int32(1)

    def face_cb_into(self, out, component, origin, box_shape):
        """cb of E component 0 / 1 / 2 on the box at `origin` (contiguous float32 out of box_shape), on the device."""
        grid, block = launch(box_shape)
        box_face_cb_3d(grid, block, (out, np.int32(component), *(np.int32(value) for value in origin),
                                     *(np.int32(value) for value in box_shape), *self.coefficient_args()))

    # ---- dispersive poles ----
    def add_pole(self, pole, window, mask):
        """Drude / Lorentz pole on the cropped bool mask of a region window (box + face weights shared per mask)."""
        if self.precision != 'single':
            raise ValueError(f'dispersive poles need single precision fields (the pole kernels read float32 E), the '
                             f'engine stores {self.precision}')
        region = self.region_by_mask.get(id(mask))
        if region is None:
            region = self._pole_region(window, mask)
            self.region_by_mask[id(mask)] = region
            self.pole_regions.append(region)
        box_shape = region.weights[0].shape
        bloch = self.fields.imag is not None
        xp = self.xp

        def box_arrays():
            return [xp.zeros(box_shape, dtype=DTYPE) for _ in range(3)]
        if isinstance(pole, DrudePole):
            region.drude.append((*pole.ade_coefficients(self.dt), box_arrays()))
            if bloch:
                region.imag_drude.append(box_arrays())
        elif isinstance(pole, LorentzPole):
            region.lorentz.append((*pole.ade_coefficients(self.dt), box_arrays(), box_arrays()))
            if bloch:
                region.imag_lorentz.append((box_arrays(), box_arrays()))
        else:
            raise TypeError(f'add_pole needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        if self._enable_q_accum and region.work is None:
            region.work = box_arrays()

    def _pole_region(self, window, mask):
        """_PoleRegion on the window extended one cell to the lo side (the whole axis on a periodic axis the window
        starts at cell 0 of: its lo face is the wrap face); weight_c = wl_c m(cell) + wu_c m(hi face partner)."""
        box = tuple(slice(0, count) if periodic and cells.start == 0 else slice(max(cells.start - 1, 0), cells.stop)
                    for cells, count, periodic in zip(window, self.shape, self.periodic))
        box_shape = tuple(cells.stop - cells.start for cells in box)
        xp = self.xp
        membership = xp.zeros(tuple(count + 1 for count in box_shape), dtype=DTYPE)
        offset = tuple(cells.start - box_cells.start for cells, box_cells in zip(window, box))
        membership[tuple(slice(start, start + count) for start, count in zip(offset, mask.shape))] = mask
        for axis_index, cells in enumerate(box):   # partner of the last grid face: cell 0 (periodic) or the cell itself
            if cells.stop != self.shape[axis_index] or (self.periodic[axis_index] and cells.start != 0):
                continue   # inside the grid the padding cell lies outside the region (0)
            pad, partner = [slice(None)] * 3, [slice(None)] * 3
            pad[axis_index] = box_shape[axis_index]
            partner[axis_index] = 0 if self.periodic[axis_index] else box_shape[axis_index] - 1
            membership[tuple(pad)] = membership[tuple(partner)]
        weights = []
        for axis_index in range(3):
            lower_share, upper_share = (share[box[axis_index]] for share in self.shares[axis_index])
            broadcast = [1, 1, 1]
            broadcast[axis_index] = -1
            own = membership[:box_shape[0], :box_shape[1], :box_shape[2]]
            shifted = [slice(0, count) for count in box_shape]
            shifted[axis_index] = slice(1, box_shape[axis_index] + 1)
            weights.append(xp.ascontiguousarray(lower_share.reshape(broadcast) * own
                                                + upper_share.reshape(broadcast) * membership[tuple(shifted)]))
        return _PoleRegion(box, weights, window, mask)

    def _refresh_pole_cb(self):
        for region in self.pole_regions:
            origin = tuple(cells.start for cells in region.box)
            for component, cb in enumerate(region.cb):
                self.face_cb_into(cb, component, origin, cb.shape)

    # ---- stepping ----
    def advance(self, sources, monitors):
        """One Yee step (order in the module header); monitors with update() sample every sample_interval steps."""
        fields = self.fields
        t = self.time
        accumulate = self._enable_q_accum
        face_args = self._face_args()
        probe_launch = self._take_probe_launch(monitors)   # recorded at the end of the E graph
        # (set, partner, CPML kernel arguments, wrap scalars): the real set alone (partner itself), or both under Bloch
        # k != 0
        sets = (((fields, fields, self._cpml_args[1], self._wrap_args[1]),) if fields.imag is None else
                ((fields, fields.imag, self._cpml_args[1], self._wrap_args[1]),
                 (fields.imag, fields, self._cpml_args[-1], self._wrap_args[-1])))
        self._fixed_block(('h', id(self.table_mu), id(self.table_sigma_m), id(self._conformal), float(self._e_to_h)),
                          (self.table_mu, self.table_sigma_m, self._conformal), lambda: self._h_block(sets))
        for source in sources:   # H-type injections (TFSF / mode-port corrections) at the H update centre t
            inject_magnetic = getattr(source, 'inject_magnetic', None)
            if inject_magnetic is not None:
                inject_magnetic(fields, t)
        for field_set, _, _, _ in sets:
            for name, index in self._mirror_h_zero:
                half_bits(getattr(field_set, name))[index] = 0   # bits 0 = +0.0 also under half_storage
        for region in self.pole_regions:
            self._pole_drive(region)
            if accumulate:
                self._pole_work(region)
        e_tables = (self.table_eps_x, self.table_eps_y, self.table_eps_z, self.table_sigma)
        # the probe launch (None without probes) joins the key and the kept inputs (replaced when its table grows)
        self._fixed_block(('e', *map(id, e_tables), id(face_args[0]), id(face_args[1]), int(face_args[2]),
                           float(self._h_to_e), id(probe_launch)), (*e_tables, face_args, probe_launch),
                          lambda: self._e_block(sets, face_args, probe_launch))
        for region in self.pole_regions:
            self._pole_correct(region)
        for source in sources:
            source.inject(fields, t + 0.5 * self.dt)
        for field_set, _, _, _ in sets:
            for name, index in self._mirror_e_zero:
                half_bits(getattr(field_set, name))[index] = 0
            for name, ghost, inner in self._mirror_e_images:   # in place: no temporary per step
                component = getattr(field_set, name)
                if component.dtype == np.float16:   # sign-bit flip = exact negation, no cupy float16 kernel
                    bits = half_bits(component)
                    self.xp.bitwise_xor(bits[inner], np.uint16(0x8000), out=bits[ghost])
                else:
                    self.xp.negative(component[inner], out=component[ghost])
        if accumulate:
            self._accumulate_e_new()
        self.time = t + self.dt
        self.n_step += 1
        if probe_launch is not None:
            if self._probes_need_resample(sources, t + 0.5 * self.dt):
                self._probe_rows.rewrite_last_row()
            self._probe_rows.stamp(self.time)
        for monitor in monitors:
            if hasattr(monitor, 'update'):
                if self.n_step % monitor.sample_interval == 0:
                    monitor.update(fields, self.time)
            elif not getattr(monitor, 'recorded_by_rows', False):
                monitor.record(fields, self.time)

    def _take_probe_launch(self, monitors):
        """This step's probe_rows launch over the FieldProbes among monitors (gpu: taken over at their first step, the
        real set's float32 / float16 fields), None without."""
        if self.xp is np:
            return None
        scan = (id(monitors), len(monitors))
        if scan != self._probe_scan:
            self._probe_scan = scan
            for monitor in monitors:
                if isinstance(monitor, FieldProbe) and not monitor.recorded_by_rows:
                    if self._probe_rows is None:
                        self._probe_rows = FieldProbeRows(self.fields, self.fields.ex.dtype)
                    if self._probe_rows.accepts(monitor):
                        self._probe_rows.add(monitor)
        return None if self._probe_rows is None else self._probe_rows.next_launch()

    def _probes_need_resample(self, sources, t_half) -> bool:
        """Whether this step's probe record (end of the E graph) may be stale: a probe reads E and an eager write after
        the graph may have changed it (pole corrections, E mirror rows, a source; a DipoleSource3D only while
        DTYPE(temporal(t_half)) != 0: at 0 its add leaves every value equal)."""
        if self._probe_rows.components.isdisjoint(_E_PROBE_COMPONENTS):
            return False
        if self.pole_regions or self._mirror_e_zero or self._mirror_e_images:
            return True
        from .sources3d import DipoleSource3D   # it imports this module
        return not all(type(source) is DipoleSource3D and DTYPE(source.temporal(t_half)) == 0 for source in sources)

    def _h_block(self, sets):
        """H Yee with the fused H CPML (+ the conformal faces, outside the layers: additive, so the order is free) of
        every field set: fixed arguments only (E is not written in the H phase, so the set order does not matter)."""
        nx, ny, nz = self._ints
        grid, block = self._grid_launch
        for field_set, partner, cpml_args, (h_wrap, _) in sets:
            self.kernels['update_h_3d'](grid, block, (
                field_set.hx, field_set.hy, field_set.hz, field_set.ex, field_set.ey, field_set.ez, partner.ex,
                partner.ey, partner.ez, self.material_id, self.table_mu, self.table_sigma_m, *self.inv_h,
                *self._radial_h, nx, ny, nz, self._dt32, self._e_to_h, *h_wrap, *cpml_args['h']))
            if self._conformal is not None:
                self._conformal_h(field_set)

    def _e_block(self, sets, face_args, probe_launch=None):
        """E Yee with the fused E CPML of every field set, then the probe record (real set fields): fixed arguments
        only."""
        self._e_updates(sets, face_args)
        if probe_launch is not None:
            kernel, grid, block, args = probe_launch
            kernel(grid, block, args)

    def _e_updates(self, sets, face_args):
        """E Yee with the fused E CPML of every field set."""
        nx, ny, nz = self._ints
        grid, block = self._grid_launch
        for field_set, partner, cpml_args, (_, e_wrap) in sets:
            self.kernels['update_e_3d'](grid, block, (
                field_set.ex, field_set.ey, field_set.ez, field_set.hx, field_set.hy, field_set.hz, partner.hx,
                partner.hy, partner.hz, self.material_id, self.table_eps_x, self.table_eps_y, self.table_eps_z,
                self.table_sigma, *face_args, *self.inv_e, *self._radial_e, *self.shares[0], *self.shares[1],
                *self.shares[2], nx, ny, nz,
                *self._periodic_ints, self._dt32, self._h_to_e, *e_wrap, *cpml_args['e']))

    def _fixed_block(self, key, kept, block):
        """Run block() (launches with fixed arguments, no sync, no allocation): eagerly on the cpu, on the first step of
        a key and when capture failed, else replayed from its CUDA graph. key holds the ids of the replaceable inputs
        (kept alive with the entry, so an id is never reused while its graph lives); fields, CPML psi, index and the
        1D arrays are rewritten in place, never reassigned (em3d sub file "Engine")."""
        if not self.use_step_graphs:
            block()
            return
        entry = self._step_graphs.get(key)
        if entry is None:   # eager run first: loads the kernel modules outside the capture
            if len(self._step_graphs) >= _MAX_STEP_GRAPHS:
                self._step_graphs.clear()
            self._step_graphs[key] = [None, kept]
            block()
            return
        if entry[0] is None:
            entry[0] = self._capture(block)
            if entry[0] is None:
                block()
                return
        entry[0].launch(self.xp.cuda.get_current_stream())

    def _capture(self, block):
        """CUDA graph of block() (None, with a warning and eager steps from now on, when capture fails)."""
        cp = self.xp
        stream = cp.cuda.Stream(non_blocking=True)
        try:
            with stream:
                stream.begin_capture()
                try:
                    block()
                finally:
                    graph = stream.end_capture()
        except Exception as error:   # capture unsupported here: the same kernels launched one by one
            self.use_step_graphs = False
            self._step_graphs.clear()
            warnings.warn(f'em_fdtd_3d: CUDA graph capture of the step kernels failed ({type(error).__name__}: '
                          f'{error}); launching them one by one from now on', RuntimeWarning, stacklevel=4)
            return None
        self.n_step_graphs += 1
        return graph

    def _e_views(self, region, fields=None):
        fields = self.fields if fields is None else fields
        return [fields.ex[region.box], fields.ey[region.box], fields.ez[region.box]]

    def _pole_sets(self, region):
        """[(E views, drude [(damping, drive, J)], lorentz [(damping, restoring, drive, P, dP)])] of the real set and,
        under Bloch k != 0, of the imaginary set (its own pole states, same coefficients)."""
        sets = [(self._e_views(region), region.drude, region.lorentz)]
        if self.fields.imag is not None:
            sets.append((self._e_views(region, self.fields.imag),
                         [(damping, drive, currents) for (damping, drive, _), currents
                          in zip(region.drude, region.imag_drude)],
                         [(damping, restoring, drive, polarisations, steps) for (damping, restoring, drive, _, _),
                          (polarisations, steps) in zip(region.lorentz, region.imag_lorentz)]))
        return sets

    def _pole_drive(self, region):
        """J^{n+1/2} / P^{n+1} of every pole of the region from E^n (before the Yee update), per field set."""
        for views, drude, lorentz in self._pole_sets(region):
            for damping, drive, currents in drude:
                for weight, e, current in zip(region.weights, views, currents):
                    drude_j_weighted(damping, drive, weight, e, current)
            for damping, restoring, drive, polarisations, steps in lorentz:
                for weight, e, polarisation, step in zip(region.weights, views, polarisations, steps):
                    lorentz_p_weighted(damping, restoring, drive, weight, e, polarisation, step)

    def _pole_correct(self, region):
        """E^{n+1} -= cb J^{n+1/2} (Drude), -= (cb / dt) (P^{n+1} - P^n) (Lorentz) after the Yee + CPML update."""
        inverse_dt = DTYPE(1.0 / self.dt)
        for views, drude, lorentz in self._pole_sets(region):
            for _, _, currents in drude:
                for cb, e, current in zip(region.cb, views, currents):
                    drude_e_correction(current, cb, e)
            for _, _, _, _, steps in lorentz:
                for cb, e, step in zip(region.cb, views, steps):
                    lorentz_e_correction(step, cb, inverse_dt, e)

    # ---- loss accounting (EM loss protocol) ----
    @property
    def enable_q_accum(self) -> bool:
        """Whether ohmic and pole losses accumulate into q_accum (allocated at the first enable)."""
        return self._enable_q_accum

    @enable_q_accum.setter
    def enable_q_accum(self, enabled) -> None:
        if enabled and self.fields.imag is not None:
            raise ValueError('absorbed power (q_accum) is not available with a Bloch wave vector k != 0: the complex '
                             'field mixes the +k and -k responses')
        self._enable_q_accum = bool(enabled)
        if not self._enable_q_accum:
            return
        if self.q_accum is None:
            self.q_accum = self.xp.zeros(self.shape, dtype=DTYPE)
        for region in self.pole_regions:
            if region.work is None:
                region.work = [self.xp.zeros_like(weight) for weight in region.weights]
        if self.table_sigma is not None:
            self._refresh_ohmic_box()

    def _refresh_ohmic_box(self):
        """Bounding box of the cells with sigma_e > 0 (x-slab chunks, a few host syncs), None without any."""
        any_x = self.xp.zeros(self.nx, dtype=bool)
        any_y = self.xp.zeros(self.ny, dtype=bool)
        any_z = self.xp.zeros(self.nz, dtype=bool)
        for chunk in self._x_chunks():
            lossy = self.table_sigma[self.material_id[chunk]] > 0
            any_x[chunk] = lossy.any(axis=(1, 2))
            any_y |= lossy.any(axis=(0, 2))
            any_z |= lossy.any(axis=(0, 1))
        extents = [np.flatnonzero(to_host(flags)) for flags in (any_x, any_y, any_z)]
        if any(indices.size == 0 for indices in extents):
            self._ohmic_box = None
            return
        origin = tuple(int(indices[0]) for indices in extents)
        shape = tuple(int(indices[-1]) - start + 1 for indices, start in zip(extents, origin))
        self._ohmic_box = (tuple(np.int32(value) for value in (*origin, *shape)), launch(shape))

    def _pole_work(self, region):
        """E^n half of the time-centred pole work, right after the pole drive (E not yet updated)."""
        views = self._e_views(region)
        half_dt = DTYPE(0.5 * self.dt)
        for _, _, currents in region.drude:
            for work, current, e in zip(region.work, currents, views):
                pole_work_half(half_dt, current, e, work)
        for _, _, _, _, steps in region.lorentz:
            for work, step, e in zip(region.work, steps, views):
                pole_work_half(DTYPE(0.5), step, e, work)

    def _accumulate_e_new(self):
        """End of a step while accumulating: E^{n+1} pole work halves, the ohmic loss of the conducting box, dt."""
        for region in self.pole_regions:
            self._pole_work(region)
        if self.pole_regions:
            self._pole_work_pending = True
        if self._ohmic_box is not None:
            box_ints, (grid, block) = self._ohmic_box
            fields = self.fields
            self.kernels['ohmic_loss_3d'](
                grid, block, (self.q_accum, fields.ex, fields.ey, fields.ez, self.material_id, self.table_sigma,
                              *box_ints, *self._ints, *self._periodic_ints, DTYPE(0.5 * self.dt * self.e_scale ** 2)))
        self.q_accum_time += self.dt

    def deposit_pole_work(self):
        """Move the pending pole face work into the cells of q_accum: a cell gets 0.5 m (W_hi / w_hi + W_lo / w_lo) per
        component (m membership, w face weight), which sums exactly to the face work times its dual volume."""
        if not self._pole_work_pending:
            return
        xp = self.xp
        for region in self.pole_regions:
            offset = tuple(cells.start - box_cells.start for cells, box_cells in zip(region.window, region.box))
            members = region.mask.astype(DTYPE)
            deposit = xp.zeros(region.mask.shape, dtype=DTYPE)
            own = tuple(slice(start, start + count) for start, count in zip(offset, region.mask.shape))
            for axis_index, (work, weight) in enumerate(zip(region.work, region.weights)):
                ratio = xp.where(weight > 0, work / xp.maximum(weight, DTYPE(1e-30)), DTYPE(0.0))
                deposit += ratio[own]
                count = region.mask.shape[axis_index]
                lower = list(own)
                if offset[axis_index] >= 1:   # the lo faces of the window cells lie inside the box
                    lower[axis_index] = slice(offset[axis_index] - 1, offset[axis_index] - 1 + count)
                    deposit += ratio[tuple(lower)]
                    work.fill(DTYPE(0.0))
                    continue
                # window starts at grid cell 0: its first cell's lo face is the wrap face (periodic: the box spans
                # the axis, _pole_region) or none
                target = [slice(None)] * 3
                lower[axis_index] = slice(0, count - 1)
                target[axis_index] = slice(1, count)
                deposit[tuple(target)] += ratio[tuple(lower)]
                if self.periodic[axis_index]:
                    last_face = self.shape[axis_index] - 1
                    lower[axis_index] = slice(last_face, last_face + 1)
                    target[axis_index] = slice(0, 1)
                    deposit[tuple(target)] += ratio[tuple(lower)]
                work.fill(DTYPE(0.0))
            self.q_accum[region.window] += DTYPE(0.5) * members * deposit
        self._pole_work_pending = False

    def reset_q_accum(self) -> None:
        """Start a new accumulation window; a non-empty one is kept as last_q_accum / last_q_accum_time."""
        self.deposit_pole_work()
        if self.q_accum_time > 0.0:
            if self.last_q_accum is None:
                self.last_q_accum = self.xp.empty_like(self.q_accum)
            self.xp.copyto(self.last_q_accum, self.q_accum)
            self.last_q_accum_time = self.q_accum_time
        if self.q_accum is not None:
            self.q_accum.fill(DTYPE(0.0))
        self.q_accum_time = 0.0

    def take_absorbed_power(self, merge_previous: bool = False):
        """Mean absorbed power density [W/m^3] since the last call (new full-grid array), then a window reset; None for
        an empty window or before accumulation; merge_previous (end-of-run flush) first folds in the last window."""
        window = float(self.q_accum_time)
        if self.q_accum is None or window <= 0.0:
            return None
        self.deposit_pole_work()
        if merge_previous and self.last_q_accum is not None:
            self.q_accum += self.last_q_accum
            window += float(self.last_q_accum_time)
            self.q_accum_time = window
        power_density = self.q_accum * DTYPE(1.0 / window)
        self.reset_q_accum()
        return power_density

    # ---- checks ----
    def stability_issue(self, pole_terms):
        """Error text when eps mu < (C0 dt)^2 (1/dx^2 + 1/dy^2 + 1/dz^2) (finest physical widths: a sector's innermost
        arc) + mu P anywhere (P = the region's pole term on its cells, pole_terms = [(region, P)]), else None (host
        syncs)."""
        requirement = (C0 * self.dt) ** 2 * sum(1.0 / width ** 2 for width in self._finest_widths)
        xp = self.xp
        table_eps = xp.minimum(xp.minimum(self.table_eps_x, self.table_eps_y), self.table_eps_z)   # PEC stays -1
        bulk = table_eps[:-1] * self.table_mu[:-1]
        worst = float(bulk.min()) / requirement if bulk.size else math.inf
        text = None
        if worst < 1.0 - 1e-6:
            text = (f'min(eps_xx, eps_yy, eps_zz) * mu_r = {worst * requirement:.4g} is below (C0 dt)^2 sum(1/d^2) = '
                    f'{requirement:.4g}')
        for region, pole_term in pole_terms:
            ids = self.material_id[region.box]
            covered = (region.weights[0] + region.weights[1] + region.weights[2]) > 0
            eps, mu = table_eps[ids], self.table_mu[ids]
            ratio = xp.where(covered & (eps > 0), eps * mu / (requirement + mu * DTYPE(pole_term)), np.inf)
            region_worst = float(ratio.min())
            if region_worst < 1.0 - 1e-6 and region_worst < worst:
                worst = region_worst
                text = (f'a dispersive region needs eps_r * mu_r >= (C0 dt)^2 sum(1/d^2) + mu_r * pole term '
                        f'({requirement:.4g} + mu_r * {pole_term:.4g}; leap-frog Drude (omega_p dt)^2 / 4, Lorentz '
                        f'delta_eps (omega_0 dt)^2 / (4 - (omega_0 dt)^2))')
        if text is None:
            return None
        stable_factor = math.floor(float(self.domain.grid.courant) * math.sqrt(max(worst, 0.0)) * 1e3) / 1e3
        return (f'{text}, so the time step is unstable (finest cell widths used everywhere); lower courant_factor to '
                f'<= {stable_factor:.3f} or refine the mesh')

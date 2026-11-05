"""Slab-mode ports of the 2D FDTD engines (A3): port-line extraction for slab_modes, the one-sided TFSF ModeSource and
the ModeMonitor; ModePortSetup.apply(engine, materials) solves a port's modes (finalize or the first step) and re-solves
them when the line's materials drift later. PortLine and mode_table_rows are shared with the FDFD ports."""
# Line axis = direction ALONG the line (LineSource semantics), normal = the other axis. u = out-of-plane field at the
# line-cell centres, t = its transverse partner on the hi face along the normal (_TRANSVERSE_COMPONENT); t = sign_n g u
# for the mode travelling toward +normal. After the first step an FDTD port follows material changes (thermo-optic,
# Pockels slot writes) only through ModePortSetup: each recalibration compares the line's material rows with those of
# the last solve and re-solves above _RESOLVE_DRIFT, tracking the previous mode (the source reads db / cb live, so the
# update coefficients are always current). PortLine reads only physics.domain, boundary_config, polarization,
# dispersive_regions and pec_cells (EMPhysics and FDFDPhysics alike), never engine.
import cmath
import math

import numpy as np
from ...core.backend import array_module_of, cp, to_host
from ...core.kernels import twin

from ...core.constants import C0, DTYPE, MU0
from ...core.sources import WaveformSpectrumCache
from .boundaries import MIRROR_TYPES, BCType, cpml_axis_profiles
from .dispersion import complex_permittivity
from .monitors import _check_frequencies, _LineSpectra
from .operators import pml_stretch
from .slab_modes import (SlabLine, align_phase, group_index, mode_amplitudes, mode_overlap, solve_slab_modes,
                         track_mode, track_slab_mode)
from .yee_grid import cpml_grading_cell_size, face_average, face_pec, grid_cell_widths, yee_inverse_spacings

__all__ = ['PortLine', 'mode_table_rows', 'ModeSource', 'ModeMonitor', 'ModePortSetup']

_DIRECTIONS = ('positive', 'negative')
# (polarization, normal axis) -> transverse partner t (hi face along the normal) and sign_n (t = sign_n g u for the
# positive mode). sign_n equals the polarisation sign s_pol of the TFSF corrections derived from the kernel stencils,
# so a correction's sign is sigma = sign_n * direction sign.
_TRANSVERSE_COMPONENT = {('ez', 'x'): 'hy', ('ez', 'y'): 'hx', ('hz', 'x'): 'ey', ('hz', 'y'): 'ex'}
_NORMAL_SIGN = {('ez', 'x'): -1.0, ('ez', 'y'): 1.0, ('hz', 'x'): 1.0, ('hz', 'y'): -1.0}
# Grid-edge wall BC -> end condition of the line operator (the outermost cell, pinned by the wall, is not solved).
_WALL_ENDS = {'ez': {BCType.PEC: 'dirichlet', BCType.PMC: 'neumann', BCType.SYMMETRIC: 'neumann',
                     BCType.ANTISYMMETRIC: 'odd'},
              'hz': {BCType.PEC: 'neumann', BCType.ANTISYMMETRIC: 'neumann', BCType.PMC: 'dirichlet',
                     BCType.SYMMETRIC: 'odd'}}
# PEC just beyond an end: E_z = 0 in the next cell (ez), tangential E = 0 on the end face (hz).
_METAL_ENDS = {'ez': 'dirichlet', 'hz': 'neumann'}
# A 'pml' end sits on the grid edge, where the kernels hold the missing / never-updated end face at 0 (ez tangential
# H: PMC, hz tangential E: PEC): zero end flux in both polarisations.
_GRID_EDGE_END = 'neumann'
# Wall BCs re-applied after injection along the normal (as definition_feature_validation._PINNED_CELL_BCS /
# _PINNED_FACE_BCS, the model-path twins): per side those overwriting the outermost cell's out-of-plane field, and per polarisation those
# pinning the in-plane field on its inner face (lo face 0, hi face n - 2).
_PINNED_CELL_WALLS = {side: (BCType.PEC, BCType.PMC, BCType.SYMMETRIC, BCType.ANTISYMMETRIC) for side in ('lo', 'hi')}
_PINNED_FACE_WALLS = {'ez': {side: (BCType.PMC, BCType.SYMMETRIC) for side in ('lo', 'hi')},
                      'hz': {side: (BCType.PEC, BCType.ANTISYMMETRIC) for side in ('lo', 'hi')}}
# Port role -> (name of its face, consequence of a wall-pinned row, of a cell-width change); as
# definition_feature_validation.
_PLACEMENT_ROLES = {
    'source': ('interface', 'the launch is neither complete nor one-sided', 'the one-sided launch leaks backward'),
    'monitor': ('measuring', 'the decomposition reads the wall value instead of the field',
                'the forward / backward separation degrades'),
}
_WIDTH_SPREAD_LIMIT = 0.01   # largest relative normal width change over the line cell and its neighbours
# Port role -> consequence of an interface / measuring face inside a CPML layer (the outermost window cell next to a
# cpml face): the TFSF correction has no psi term and the mode is solved for an unstretched normal spacing.
_STRETCHED_FACE_CONSEQUENCES = {
    'source': 'the one-sided launch ignores, so it leaks backward and misses power',
    'monitor': 'the decomposition ignores, so its forward / backward powers are wrong',
}
# (polarization, line axis) -> MaterialGrid rows a port reads: the permittivity rows (ez eps_r; hz eps_t of the
# transverse partner, then eps_l of the E component along the normal on the faces along the line), sigma_e, mu_r,
# sigma_m (mu = mu_r - j sigma_m / (omega mu0), as harmonic.cell_permeability).
_MATERIAL_ROWS = {('ez', 'x'): ('eps_r', 'sigma_e', 'mu_r', 'sigma_m'),
                  ('ez', 'y'): ('eps_r', 'sigma_e', 'mu_r', 'sigma_m'),
                  ('hz', 'x'): ('eps_xx', 'eps_yy', 'sigma_e', 'mu_r', 'sigma_m'),
                  ('hz', 'y'): ('eps_yy', 'eps_xx', 'sigma_e', 'mu_r', 'sigma_m')}
_N_NON_PERMITTIVITY_ROWS = 3   # sigma_e, mu_r, sigma_m after the permittivity rows
_NOT_SOLVED = 'mode not solved yet: finalize or step first'
# GaussianPulse bandwidth above which ModeSource warns: env cos is no Hilbert partner of env sin (band-edge error
# 2 exp(-4 f / (B^2 f0)) relative, DC 2 exp(-1 / B^2) of the peak; 0.6: 8e-4 at the lower band edge, DC 12 %).
_QUADRATURE_BANDWIDTH_LIMIT = 0.6
# |mode_overlap| between neighbouring monitor frequencies below which ModeMonitor warns that tracking may have jumped
# to another mode (track_slab_mode's own acceptance threshold).
_TRACKING_WARN_OVERLAP = 0.9
# Relative change of a port line's material rows since the last solve (per row, against its largest magnitude) above
# which an FDTD port re-solves after the first step; the mismatch errors (upstream leakage, projection) grow ~ drift^2,
# so <= 1e-6 between re-solves (thermo-optic silicon: one re-solve per ~10 K).
_RESOLVE_DRIFT = 1e-3

# One TFSF correction on a (possibly strided) port-line view: incident(t) = Re(phasor) s(t) + Im(phasor) s_q(t). A
# core.kernels twin (also em3d's mode ports): the numpy twin follows the cupy source. T = the field / coefficient dtype
# (float32, float64 for the double 2D stepper); the incident phasor and waveform samples float32.
def _port_correction_host(coefficient, incident_re, incident_im, signal, quadrature, field):
    field += coefficient * (incident_re * signal + incident_im * quadrature)


_port_correction = twin('em.mode_port_correction', cp.ElementwiseKernel(
    'T coefficient, float32 incident_re, float32 incident_im, float32 signal, float32 quadrature',
    'T field',
    'field += coefficient * (T)(incident_re * signal + incident_im * quadrature)',
    'mode_port_correction'), _port_correction_host)


def _check_mode_number(mode_number) -> int:
    """mode_number as an int >= 1 (1 = largest Re(k_t^2): the most strongly guided propagating mode); ValueError
    otherwise."""
    if isinstance(mode_number, bool) or not isinstance(mode_number, (int, np.integer)) or mode_number < 1:
        raise ValueError(f"mode_number must be an int >= 1 (1 = the most strongly guided mode, largest Re(k^2)), "
                         f"got {mode_number!r}")
    return int(mode_number)


def _mode_count(mode_number):
    """Modes solved per frequency: max(mode_number + 2, 4) (mode-table rows and tracking candidates)."""
    return max(mode_number + 2, 4)


def _quadrature_warnings(temporal, label):
    """Warning for a gaussian above _QUADRATURE_BANDWIDTH_LIMIT, whose cos partner is too far from a Hilbert partner."""
    bandwidth = getattr(temporal, 'bandwidth', None)
    if bandwidth is None or bandwidth <= _QUADRATURE_BANDWIDTH_LIMIT:
        return []
    band_edge_error = 2.0 * math.exp(-4.0 * (1.0 - 0.5 * bandwidth) / bandwidth ** 2)
    return [f"{label}: bandwidth {bandwidth:.3g} > {_QUADRATURE_BANDWIDTH_LIMIT}: the quadrature (cos) partner that "
            f"drives the imaginary part of the mode profile errs by {band_edge_error:.1e} of the pulse spectrum at the "
            f"lower band edge (not normalised out of fraction_* / phase_*) and carries a DC part of "
            f"{2.0 * math.exp(-1.0 / bandwidth ** 2):.0%} of its peak (a slow static wake for lossy or PML-tailed "
            f"profiles); keep the bandwidth <= {_QUADRATURE_BANDWIDTH_LIMIT} (0.1-0.3 typical)"]


def mode_table_rows(port, modes, omega, mode_number, dt, normal_spacing, selected_number=None):
    """Rows [mode number, Re n_eff, Im n_eff, loss dB/cm, Re n_group, edge field, selected] of `modes` solved on `port`
    at omega [rad/s] with dt / normal_spacing; selected marks row selected_number (default mode_number)."""
    n_modes = _mode_count(mode_number)
    selected_number = mode_number if selected_number is None else selected_number
    rows = []
    for index, mode in enumerate(modes):
        n_group = group_index(port.slab_line, omega, mode, n_modes, dt=dt, normal_spacing=normal_spacing)
        n_eff, k = complex(mode.n_eff), complex(mode.k)
        rows.append([index + 1, n_eff.real, n_eff.imag, -20.0 * k.imag / math.log(10.0) / 100.0,
                     complex(n_group).real, float(mode.edge_amplitude), index + 1 == selected_number])
    return rows


class PortLine:
    """Geometry of a port line (solve range, end conditions, spacings and CPML profile along it, normal spacings);
    read_materials caches the host material rows, slab_line(omega) then builds SlabLines without device access."""

    def __init__(self, physics, axis, line_cell, span_cells, label):
        if axis not in ('x', 'y'):
            raise ValueError(f"{label} axis must be 'x' or 'y', got {axis!r}")
        domain = physics.domain
        config = physics.boundary_config
        self.polarization = physics.polarization
        self.axis = axis
        self.normal = 'y' if axis == 'x' else 'x'
        self.line_cell = int(line_cell)
        self.label = label
        self.normal_sign = _NORMAL_SIGN[(self.polarization, self.normal)]

        dx_cells, dy_cells = grid_cell_widths(domain.grid)
        spacings = yee_inverse_spacings(dx_cells, dy_cells)
        along_axis = 0 if axis == 'x' else 1
        along_widths = (dx_cells, dy_cells)[along_axis]
        inv_along_h = (spacings.inv_dx_h, spacings.inv_dy_h)[along_axis]
        self._inv_normal_h = (spacings.inv_dx_h, spacings.inv_dy_h)[1 - along_axis]
        self._normal_widths = (dx_cells, dy_cells)[1 - along_axis]
        self._n_normal = self._normal_widths.size
        self._periodic_normal = (config.periodic_x, config.periodic_y)[1 - along_axis]
        self._normal_walls = {side: getattr(config, f'{self.normal}_{side}') for side in ('lo', 'hi')}
        self._periodic = (config.periodic_x, config.periodic_y)[along_axis]
        n_along = along_widths.size
        self._n_along = n_along
        self._along_widths = along_widths   # whole grid line: weights the hz face eps (face_average)

        span_lo, span_hi = int(span_cells[0]), int(span_cells[1])
        if not 0 <= span_lo < span_hi <= n_along:
            raise ValueError(f"{label}: span cells [{span_lo}, {span_hi}) must be a non-empty range inside the "
                             f"{n_along} grid cells along {axis}")
        self.span_cells = (span_lo, span_hi)
        if self._periodic and span_lo == 0 and span_hi == n_along:
            a, b = 0, n_along
            self.lo_end = self.hi_end = self.lo_kind = self.hi_kind = 'periodic'
        else:
            window = domain.physical_window()[along_axis]
            a, self.lo_end, self.lo_kind = self._resolve_end(span_lo, window.start, 0, 1,
                                                             getattr(config, f'{axis}_lo'))
            b, self.hi_end, self.hi_kind = self._resolve_end(span_hi, window.stop, n_along, n_along - 1,
                                                             getattr(config, f'{axis}_hi'))
        if b <= a:
            raise ValueError(f"{label}: no cells left to solve along the line once the wall cells are excluded; "
                             f"lengthen the line")
        self.solve_range = (a, b)
        self.cell_widths = along_widths[a:b].copy()

        # Local face k = lo face of local cell k = hi face of grid cell a + k - 1 (k = 0..m); a face beyond a
        # non-periodic grid edge (only at a 'pml' end, neumann) takes the clipped neighbour's finite values, unused.
        grid_faces = np.arange(a - 1, b)
        self._face_index = grid_faces % n_along if self._periodic else np.clip(grid_faces, 0, n_along - 1)
        self.face_inverse_spacing = inv_along_h[self._face_index]

        profile = cpml_axis_profiles(config, n_along, axis, cpml_grading_cell_size(domain))
        self._cell_pml = (profile.sigma_e[a:b], profile.kappa_e[a:b], profile.alpha_e[a:b])
        self._face_pml = (profile.sigma_h[self._face_index], profile.kappa_h[self._face_index],
                          profile.alpha_h[self._face_index])
        # Normal faces (face k = hi face of cell k) inside a CPML layer (placement_warnings).
        normal_profile = cpml_axis_profiles(config, self._n_normal, self.normal, cpml_grading_cell_size(domain))
        self._stretched_normal_faces = (normal_profile.sigma_h > 0.0) | (normal_profile.kappa_h != 1.0)
        self._last_solve = None   # (key, host rows, poles, SlabLine, SlabModes) of the last solve_modes

    def _resolve_end(self, span_edge, window_edge, grid_edge, wall_solve_edge, face_type):
        """(solve-range edge, end condition, end kind) of one line end (rules of the A3 contract, section 4)."""
        metal = _METAL_ENDS[self.polarization]
        if span_edge == window_edge and face_type == BCType.CPML:
            return grid_edge, _GRID_EDGE_END, 'pml'
        # a mirror ghost is the one padding cell, so its window edge is the wall solve edge
        if not self._periodic and (span_edge == grid_edge or (span_edge == window_edge and face_type in MIRROR_TYPES)):
            return wall_solve_edge, _WALL_ENDS[self.polarization][face_type], 'wall'
        return span_edge, metal, 'truncated'

    def normal_face(self, face):
        """Grid face index `face` along the normal (hi face of that cell, wrapped on a periodic axis); ValueError when
        the face does not exist or is never updated (non-periodic face -1 or last face)."""
        if self._periodic_normal:
            return face % self._n_normal
        if not 0 <= face <= self._n_normal - 2:
            raise ValueError(f"{self.label}: needs the {self.normal} face {face} (hi face of cell {face}), which "
                             f"does not exist or is never updated on the non-periodic {self.normal} axis; move the "
                             f"line at least one cell away from the grid edge")
        return face

    def normal_spacing(self, face):
        """Dual spacing [m] across normal face `face` (1 / inv_d{normal}_h, no CPML kappa)."""
        return 1.0 / float(self._inv_normal_h[face])

    def placement_warnings(self, face, role):
        """Warnings for a line cell or normal `face` a wall BC pins every step, a `face` inside a CPML layer and > 1%
        normal width changes around the line cell (the model path's _mode_line_messages); role 'source' | 'monitor'."""
        face_name, pinned_consequence, spread_consequence = _PLACEMENT_ROLES[role]
        warnings = []
        if not self._periodic_normal:
            transverse = _TRANSVERSE_COMPONENT[(self.polarization, self.normal)]
            pinned = []
            for side, outer_cell, wall_face in (('lo', 0, 0), ('hi', self._n_normal - 1, self._n_normal - 2)):
                wall = self._normal_walls[side]
                if self.line_cell == outer_cell and wall in _PINNED_CELL_WALLS[side]:
                    out_of_plane = 'E_z' if self.polarization == 'ez' else 'H_z'
                    pinned.append(f"{out_of_plane} of the line cell ({self.normal}_{side} = {wall.value})")
                if face == wall_face and wall in _PINNED_FACE_WALLS[self.polarization][side]:
                    pinned.append(f"{transverse[0].upper()}_{transverse[1]} on its {face_name} face "
                                  f"({self.normal}_{side} = {wall.value})")
            if pinned:
                warnings.append(f"{self.label}: a wall BC pins {' and '.join(pinned)} every step, so "
                                f"{pinned_consequence}; move the line one cell further from the wall")
            if self._stretched_normal_faces[face]:
                side = 'lo' if 2 * face < self._n_normal else 'hi'
                warnings.append(f"{self.label}: its {face_name} face ({transverse[0].upper()}_{transverse[1]}) lies "
                                f"inside the {self.normal}_{side} CPML layer (the line is in the outermost window "
                                f"cell), whose stretch {_STRETCHED_FACE_CONSEQUENCES[role]}; move the line one cell "
                                f"further into the window")
        neighbours = np.arange(self.line_cell - 1, self.line_cell + 2)
        neighbours = (neighbours % self._n_normal if self._periodic_normal
                      else neighbours[(neighbours >= 0) & (neighbours < self._n_normal)])
        local_widths = self._normal_widths[neighbours]
        spread = float(local_widths.max() / local_widths.min()) - 1.0
        if spread > _WIDTH_SPREAD_LIMIT:
            warnings.append(f"{self.label}: the cell widths along {self.normal} change by {100.0 * spread:.2g}% around "
                            f"the line cell (more than {100.0 * _WIDTH_SPREAD_LIMIT:g}%): the mode is matched to one "
                            f"spacing, so {spread_consequence}; place the line on a locally uniform mesh")
        return warnings

    def line_index(self, normal_cell):
        """Index tuple of the solve range along the line at cell / face `normal_cell` along the normal."""
        cells = slice(*self.solve_range)
        return (normal_cell, cells) if self.normal == 'x' else (cells, normal_cell)

    def _grid_row(self, array):
        """View of a full-grid array along the whole grid line through the line cell."""
        return array[:, self.line_cell] if self.axis == 'x' else array[self.line_cell, :]

    def _region_row(self, box, mask, xp):
        """Float64 xp row of a dispersive region along the grid line, None when the region misses the line."""
        if box is None:
            return xp.ones(self._n_along, dtype=xp.float64)
        along_box, normal_box = (box[0], box[1]) if self.axis == 'x' else (box[1], box[0])
        if not normal_box.start <= self.line_cell < normal_box.stop:
            return None
        offset = self.line_cell - normal_box.start
        row = xp.zeros(self._n_along, dtype=xp.float64)
        row[along_box] = mask[:, offset] if self.axis == 'x' else mask[offset, :]
        return row

    def _host_rows(self, physics):
        """(host float64 material, pole-mask, PEC and subpixel-smoothing rows along the whole grid line, poles,
        has_pec); one device-to-host transfer, plus one per permittivity row under physics.subpixel_smoothing (the
        averaged - standard eps of the faces / cells the engine gives that component, _smoothing_rows)."""
        names = _MATERIAL_ROWS[(self.polarization, self.axis)]
        materials = physics.domain.materials
        xp = physics.domain.xp
        rows = [self._grid_row(getattr(materials, name)) for name in names]
        poles = []
        for pole, box, mask in physics.dispersive_regions:
            row = self._region_row(box, mask, xp)
            if row is not None:
                poles.append(pole)
                rows.append(row)
        has_pec = physics.pec_cells is not None
        if has_pec:
            rows.append(self._grid_row(physics.pec_cells))
        host_rows = to_host(xp.stack([row.astype(xp.float64) for row in rows]))
        smoothing = getattr(physics, 'subpixel_smoothing', None)
        if smoothing is not None:
            host_rows = np.vstack([host_rows] + [smoothing.line_delta(name, materials, self.axis, self.line_cell)
                                                 for name in names[:len(names) - _N_NON_PERMITTIVITY_ROWS]])
        return host_rows, poles, has_pec

    def material_drift(self, physics):
        """Largest relative change of the line's rows since read_materials (per row, against its largest magnitude;
        inf when the poles or PEC on the line changed)."""
        host_rows, poles, _ = self._host_rows(physics)
        cached_rows, cached_poles = self._cached_rows
        if host_rows.shape != cached_rows.shape or any(a is not b for a, b in zip(poles, cached_poles)):
            return math.inf
        change = np.abs(host_rows - cached_rows).max(axis=1)
        scale = np.maximum(np.abs(host_rows).max(axis=1), np.abs(cached_rows).max(axis=1))
        relative = np.divide(change, scale, out=np.zeros_like(change), where=scale > 0.0)
        return float(relative.max())

    def read_materials(self, physics):
        """Cache host float64 material, pole-mask and PEC rows along the whole grid line (one device-to-host
        transfer)."""
        names = _MATERIAL_ROWS[(self.polarization, self.axis)]
        host_rows, poles, has_pec = self._host_rows(physics)
        self._cached_rows = (host_rows, poles)
        n_permittivity = len(names) - _N_NON_PERMITTIVITY_ROWS
        self._permittivity_rows = tuple(host_rows[:n_permittivity])
        self._sigma_row, self._mu_row, self._sigma_m_row = host_rows[n_permittivity:len(names)]
        self._pole_rows = list(zip(poles, host_rows[len(names):len(names) + len(poles)]))
        n_device = len(names) + len(poles) + int(has_pec)
        self._pec_row = host_rows[n_device - 1] > 0.5 if has_pec else np.zeros(self._n_along, dtype=bool)
        # subpixel smoothing: averaged - standard eps per permittivity row (transverse: added to the line cells, as
        # the engine's normal faces; hz longitudinal: added after the face mean), zeros without smoothing
        self._smoothing_rows = (tuple(host_rows[n_device:]) if host_rows.shape[0] > n_device
                                else (np.zeros(self._n_along),) * n_permittivity)

    def slab_line(self, omega):
        """SlabLine of the solve range at angular frequency omega [rad/s] from the cached rows (read_materials)."""
        cells = slice(*self.solve_range)
        face_index = self._face_index
        mu = self._mu_row - 1j * self._sigma_m_row / (float(omega) * MU0)   # complex128, magnetic loss Im < 0
        permittivity = [complex_permittivity(row, self._sigma_row, self._pole_rows, omega)
                        for row in self._permittivity_rows]
        permittivity[0] = permittivity[0] + self._smoothing_rows[0]
        if self.polarization == 'ez':
            cell_weight = mu[cells]
            index_term = permittivity[0][cells] * mu[cells]
            face_weight = 1.0 / mu[face_index]   # lower grid cell of each face
        else:
            eps_transverse, eps_longitudinal = permittivity
            cell_weight = eps_transverse[cells]
            index_term = eps_transverse[cells] * mu[cells]
            face_eps = (face_average(eps_longitudinal, 0, self._periodic, self._along_widths)
                        + self._smoothing_rows[1])[face_index]
            face_is_pec = face_pec(self._pec_row, 0, self._periodic)[face_index]
            face_weight = np.where(face_is_pec, 0.0, 1.0 / face_eps)
        return SlabLine(polarization=self.polarization, cell_widths=self.cell_widths,
                        cell_stretch=pml_stretch(*self._cell_pml, omega), cell_weight=cell_weight,
                        index_term=index_term, active=~self._pec_row[cells],
                        face_inverse_spacing=self.face_inverse_spacing,
                        face_stretch=pml_stretch(*self._face_pml, omega), face_weight=face_weight,
                        lo_end=self.lo_end, hi_end=self.hi_end)

    def solve_modes(self, physics, omega, mode_number, dt, normal_spacing):
        """(SlabLine, SlabModes) at omega [rad/s] on the current materials (read_materials, slab_line, then
        _mode_count(mode_number) modes with dt / normal_spacing); the last result again when the arguments, host rows
        and poles equal the last solve's (an FDFD reassembly that left the line's materials unchanged)."""
        self.read_materials(physics)
        host_rows, poles = self._cached_rows
        key = (float(omega), _mode_count(mode_number), dt, normal_spacing)
        last = self._last_solve
        if (last is not None and last[0] == key and last[1].shape == host_rows.shape
                and np.array_equal(last[1], host_rows) and len(last[2]) == len(poles)
                and all(a is b for a, b in zip(last[2], poles))):
            return last[3], last[4]
        line = self.slab_line(omega)
        modes = solve_slab_modes(line, omega, key[1], dt=dt, normal_spacing=normal_spacing)
        self._last_solve = (key, host_rows, poles, line, modes)
        return line, modes

    def select(self, line, modes, mode_number, face, role):
        """(modes[mode_number - 1], placement + mode warnings for normal `face` and role 'source' | 'monitor');
        ValueError naming how many modes the solve found."""
        if len(modes) < mode_number:
            a, b = self.solve_range
            raise ValueError(f"{self.label}: mode_number {mode_number} requested, but the solve found only "
                             f"{len(modes)} mode(s) on the {b - a}-cell line")
        mode = modes[mode_number - 1]
        return mode, self.placement_warnings(face, role) + self.mode_warnings(line, mode, mode_number)

    def mode_warnings(self, line, mode, mode_number):
        """Warnings for a selected mode: not carrying +1 W/m (evanescent or backward), field left at an artificial
        (truncated) line end, and Re(n_eff) not above the index of the open (pml / truncated) span-end cells."""
        warnings = []
        if not mode.power > 0.5:
            warnings.append(f"{self.label}: mode {mode_number} carries {mode.power:.3g} W/m toward its direction at "
                            f"unit amplitude instead of 1 W/m (an evanescent or backward mode), so the source power "
                            f"and the monitor powers / fractions are not normalised to its power flow; check "
                            f"mode_number")
        field = np.abs(np.asarray(mode.profile))
        peak = float(field.max())
        for kind, edge_field, side in ((self.lo_kind, field[0], 'start'), (self.hi_kind, field[-1], 'end')):
            if kind == 'truncated' and edge_field > 1e-2 * peak:
                warnings.append(f"{self.label}: mode {mode_number} keeps {edge_field / peak:.2g} of its peak field "
                                f"at the {side} of the line, where the solve ends on an artificial metal wall; widen "
                                f"the line so the mode decays inside it")
        a = self.solve_range[0]
        open_cells = [cell - a for cell, kind in ((self.span_cells[0], self.lo_kind),
                                                  (self.span_cells[1] - 1, self.hi_kind))
                      if kind in ('pml', 'truncated')]
        if open_cells:
            cladding_index = max(cmath.sqrt(complex(line.index_term[cell])).real for cell in open_cells)
            n_eff = complex(mode.n_eff).real
            if n_eff <= cladding_index:
                warnings.append(f"{self.label}: mode {mode_number} has Re(n_eff) = {n_eff:.5g}, not above the index "
                                f"{cladding_index:.5g} at the line ends, so it is not guided (likely a cladding or "
                                f"radiation mode); check mode_number and the line span")
        return warnings


class _LineCorrection:
    """One TFSF correction on the port line: field += coefficient * (re * s(t) + im * s_q(t)), the coefficient a live
    view of the engine's db / cb along the line, re / im the scaled incident phasor (float32 on the coefficient's
    backend)."""

    def __init__(self, component, index, coefficient, scaled_incident):
        self.component = component
        self.index = index
        self.coefficient = coefficient
        xp = array_module_of(coefficient)
        self.incident_re = xp.asarray(np.real(scaled_incident), dtype=DTYPE)
        self.incident_im = xp.asarray(np.imag(scaled_incident), dtype=DTYPE)

    def add(self, fields, temporal, time):
        _port_correction(self.coefficient, self.incident_re, self.incident_im, DTYPE(temporal(time)),
                         DTYPE(temporal.quadrature(time)), getattr(fields, self.component)[self.index])


class ModeSource:
    """One-sided TFSF slab-mode source on a line: total field downstream from the line cell, scattered field upstream;
    injects `power` [W/m] of mode `mode_number` toward +normal ('positive') or -normal ('negative')."""

    def __init__(self, physics, axis, line_cell, span_cells, temporal, power, mode_number, direction, label):
        if direction not in _DIRECTIONS:
            raise ValueError(f"{label} direction must be one of {_DIRECTIONS}, got {direction!r}")
        if not hasattr(temporal, 'quadrature'):
            raise TypeError(f"{label} needs a waveform with a quadrature signal (GaussianPulse or ContinuousWave), "
                            f"got {type(temporal).__name__}")
        self.port = PortLine(physics, axis, line_cell, span_cells, label)
        self.axis = axis
        self.position = self.port.line_cell
        self.span = self.port.solve_range
        self.temporal = temporal
        self.power = float(power)
        self.mode_number = _check_mode_number(mode_number)
        self.tracked_mode_number = self.mode_number   # rank of the followed mode in the latest solve
        self.direction = direction
        self.warnings = []
        self.modes = None   # SlabModes at the carrier frequency once solved; mode = the selected one
        self.mode = None
        self._direction_sign = 1.0 if direction == 'positive' else -1.0
        # Interface face (scattered side): lo face of the line cell for 'positive', its hi face for 'negative'; the
        # mode is solved FDTD-consistent with its dual spacing and the engine dt.
        self.interface_face = self.port.normal_face(self.position - 1 if direction == 'positive' else self.position)
        self._normal_spacing = self.port.normal_spacing(self.interface_face)
        self._omega = 2.0 * math.pi * float(temporal.f0)
        self._dt = float(physics.engine.dt)
        self._magnetic = None
        self._electric = None
        self._table = None
        self._spectrum_cache = WaveformSpectrumCache()   # waveform_spectrum's incremental S(omega)

    @property
    def solved(self) -> bool:
        return self.modes is not None

    def solve(self, physics):
        """Solve the modes at the carrier frequency on the current materials, select mode_number and fill the incident
        line arrays."""
        port = self.port
        line, modes = port.solve_modes(physics, self._omega, self.mode_number, self._dt, self._normal_spacing)
        mode, warnings = port.select(line, modes, self.mode_number, self.interface_face, 'source')
        self.tracked_mode_number = self.mode_number
        self._set_mode(physics, mode, modes, warnings)

    def resolve(self, physics):
        """Re-solve on the current materials, keeping the mode that continues the selected one (largest overlap); its
        rank there becomes tracked_mode_number."""
        port = self.port
        line, modes = port.solve_modes(physics, self._omega, self.mode_number, self._dt, self._normal_spacing)
        self.tracked_mode_number = track_mode(line, self.mode, modes) + 1
        mode = modes[self.tracked_mode_number - 1]
        warnings = (port.placement_warnings(self.interface_face, 'source')
                    + port.mode_warnings(line, mode, self.tracked_mode_number))
        self._set_mode(physics, mode, modes, warnings)

    def _set_mode(self, physics, mode, modes, warnings):
        """Store the selected mode, its warnings and the incident line corrections built from it."""
        port = self.port
        self.warnings = warnings + _quadrature_warnings(self.temporal, port.label)

        # Incident phasors of the mode scaled to `power` (modes carry 1 W/m): u at the line cells, t on the interface
        # face, upstream of the line-cell centre by half its dual spacing.
        sigma = self._direction_sign * port.normal_sign
        u_incident = math.sqrt(self.power) * np.asarray(mode.profile)
        t_incident = (sigma * np.asarray(mode.transverse_factor) * u_incident
                      * np.exp(0.5j * complex(mode.k) * self._normal_spacing))
        engine = physics.engine
        normal = port.normal
        cell_index = port.line_index(self.position)
        face_index = port.line_index(self.interface_face)
        inv_centre = float(getattr(engine, f'_inv_d{normal}_e_1d')[self.position])
        inv_face = float(getattr(engine, f'_inv_d{normal}_h_1d')[self.interface_face])
        transverse = _TRANSVERSE_COMPONENT[(port.polarization, normal)]
        if port.polarization == 'ez':
            self._magnetic = _LineCorrection(transverse, face_index, engine.db[face_index],
                                             sigma * inv_face * u_incident)
            self._electric = _LineCorrection('ez', cell_index, engine.cb[cell_index],
                                             sigma * inv_centre * t_incident)
        else:
            face_cb = engine.cb_y if normal == 'x' else engine.cb_x
            self._magnetic = _LineCorrection('hz', cell_index, engine.db[cell_index],
                                             sigma * inv_centre * t_incident)
            self._electric = _LineCorrection(transverse, face_index, face_cb[face_index],
                                             sigma * inv_face * u_incident)
        self.mode = mode
        self.modes = modes
        self._table = None

    def inject_magnetic(self, fields, t: float):
        """Magnetic correction from the incident E-type field at t, the H update's centre (nothing before the solve)."""
        if self._magnetic is not None:
            self._magnetic.add(fields, self.temporal, t)

    def inject_electric(self, fields, t: float):
        """Electric correction from the incident H-type field at t, the E update's centre (nothing before the solve)."""
        if self._electric is not None:
            self._electric.add(fields, self.temporal, t)

    def mode_table(self):
        """Rows [mode number, Re n_eff, Im n_eff, loss dB/cm, Re n_group, edge field, selected] of the solved modes
        (n_group computed at the first call, cached); ValueError before the solve."""
        if not self.solved:
            raise ValueError(_NOT_SOLVED)
        if self._table is None:
            self._table = mode_table_rows(self.port, self.modes, self._omega, self.mode_number, self._dt,
                                          self._normal_spacing, selected_number=self.tracked_mode_number)
        return self._table

    def waveform_spectrum(self, frequencies, n_step, sample_interval=1):
        """S(omega) = sum of s(n dt) exp(-j omega n dt) m dt over n = m, 2m, ... <= n_step (m = sample_interval, the
        monitor's DFT grid), complex128 at `frequencies` [Hz]; cached per (frequency set, m), a later call adds only
        the steps run since (core.sources.WaveformSpectrumCache)."""
        return self._spectrum_cache.spectrum(self.temporal, self._dt, frequencies, n_step, sample_interval)


class ModeMonitor:
    """Mode decomposition on a line: running DFTs of u at the line cells and of t on their hi faces (normal) over the
    solve range; the modes at every monitor frequency come from the last (re-)solve (ModePortSetup)."""

    def __init__(self, physics, axis, line_cell, span_cells, frequencies, mode_number, label):
        self.port = PortLine(physics, axis, line_cell, span_cells, label)
        self.axis = axis
        self.position = self.port.line_cell
        self.span = self.port.solve_range
        self.mode_number = _check_mode_number(mode_number)
        self.tracked_mode_number = self.mode_number   # rank of the followed centre mode in the latest solve
        self.polarization = physics.polarization
        self.warnings = []
        self.frequencies = _check_frequencies(frequencies)
        self.n_freqs = self.frequencies.size
        self.dt = float(physics.engine.dt)
        self.measuring_face = self.port.normal_face(self.position)

        # Spectra 0 = u, 1 = t, both phased at the E sample time t; the H-type one (ez: t, hz: u) is re-timed in
        # amplitudes().
        n_cells = self.span[1] - self.span[0]
        line = self.port.line_index(self.position)
        t_component = _TRANSVERSE_COMPONENT[(self.polarization, self.port.normal)]
        self._spectra = _LineSpectra(((self.polarization, line, None, n_cells), (t_component, line, None, n_cells)),
                                     self.frequencies, self.dt, xp=physics.domain.xp)
        self.sample_interval = 1   # as DFTMonitor
        self.lines = None   # SlabLine and selected SlabMode per frequency once solved
        self.modes = None
        self._centre = None   # frequency index of the full (centre) solve

    @property
    def solved(self) -> bool:
        return self.modes is not None

    def update(self, fields, t: float):
        self._spectra.update(fields, t, self.sample_interval)

    def solve(self, physics, centre_reference=None):
        """Modes at every frequency: mode_number (or, with centre_reference, the mode overlapping it most) picks the mode
        at the centre frequency (full solve), which is followed (track_slab_mode) and phase-aligned outward in
        frequency, one neighbour at a time (doubtful steps warn)."""
        port = self.port
        port.read_materials(physics)
        normal_spacing = port.normal_spacing(self.measuring_face)
        omegas = 2.0 * np.pi * self.frequencies
        lines = [port.slab_line(omega) for omega in omegas]
        n_modes = _mode_count(self.mode_number)
        order = np.argsort(self.frequencies)
        middle = self.n_freqs // 2
        centre = order[middle]
        selected = [None] * self.n_freqs
        centre_modes = solve_slab_modes(lines[centre], omegas[centre], n_modes, dt=self.dt,
                                        normal_spacing=normal_spacing)
        if centre_reference is None:
            selected[centre], warnings = port.select(lines[centre], centre_modes, self.mode_number,
                                                     self.measuring_face, 'monitor')
            self.tracked_mode_number = self.mode_number
        else:
            self.tracked_mode_number = track_mode(lines[centre], centre_reference, centre_modes) + 1
            selected[centre] = centre_modes[self.tracked_mode_number - 1]
            warnings = (port.placement_warnings(self.measuring_face, 'monitor')
                        + port.mode_warnings(lines[centre], selected[centre], self.tracked_mode_number))
        doubtful = []   # (frequency [Hz], |overlap| with the neighbour's mode, mode_warnings there)
        for neighbours in (order[middle + 1:], order[:middle][::-1]):
            reference = selected[centre]
            for index in neighbours:
                tracked = track_slab_mode(lines[index], omegas[index], reference, n_modes, dt=self.dt,
                                          normal_spacing=normal_spacing)
                overlap = abs(mode_overlap(lines[index], reference, tracked))
                tracked_issues = port.mode_warnings(lines[index], tracked, self.tracked_mode_number)
                if overlap < _TRACKING_WARN_OVERLAP or tracked_issues:
                    doubtful.append((float(self.frequencies[index]), overlap, tracked_issues))
                reference = selected[index] = align_phase(lines[index], tracked, reference)
        if doubtful:
            warnings.append(self._tracking_warning(doubtful))
        self.warnings = warnings
        self.lines = lines
        self.modes = selected
        self._centre = centre

    def resolve(self, physics):
        """Re-solve every frequency on the current materials, the centre mode continuing the previous one."""
        self.solve(physics, centre_reference=self.modes[self._centre])

    def _tracking_warning(self, doubtful):
        """One warning for the frequencies where the tracked mode overlaps its neighbour below _TRACKING_WARN_OVERLAP
        or fails mode_warnings."""
        wavelengths_nm = [C0 / frequency * 1e9 for frequency, _, _ in doubtful]
        lowest_overlap = min(overlap for _, overlap, _ in doubtful)
        first_issue = next((issues[0] for _, _, issues in doubtful if issues), None)
        example = f"; e.g. {first_issue}" if first_issue else ''
        return (f"{self.port.label}: following mode {self.tracked_mode_number} away from the centre frequency is "
                f"doubtful at "
                f"{len(doubtful)} of {self.n_freqs - 1} wavelengths ({min(wavelengths_nm):.4g}-"
                f"{max(wavelengths_nm):.4g} nm; lowest overlap with the neighbouring wavelength's mode "
                f"{lowest_overlap:.2f}){example}. Past a cutoff or an anticrossing the monitor follows another "
                f"(cladding / radiation) mode, so its powers, fractions and n_eff there are not the selected mode's; "
                f"narrow the wavelength band or check the result's n_eff")

    def n_eff(self):
        """Effective index of the selected mode per frequency, complex128 (n_f,); ValueError before the solve."""
        if not self.solved:
            raise ValueError(_NOT_SOLVED)
        return np.array([complex(mode.n_eff) for mode in self.modes], dtype=np.complex128)

    def amplitudes(self):
        """(a_positive, a_negative) complex128 (n_f,) in units of the normalised mode (one host transfer of the
        spectra); ValueError before the solve."""
        if not self.solved:
            raise ValueError(_NOT_SOLVED)
        # The H-type field (ez: t, hz: u) was sampled dt/2 before the E sample time its phases used.
        u_spectra = self._spectra.host(0, retime_h=self.polarization == 'hz')
        t_spectra = self._spectra.host(1, retime_h=self.polarization == 'ez')
        offset = 0.5 * self.port.normal_spacing(self.measuring_face)
        sign_n = self.port.normal_sign
        a_positive = np.empty(self.n_freqs, dtype=np.complex128)
        a_negative = np.empty(self.n_freqs, dtype=np.complex128)
        for index, (line, mode) in enumerate(zip(self.lines, self.modes)):
            a_positive[index], a_negative[index] = mode_amplitudes(line, mode, u_spectra[index],
                                                                   t_spectra[index] / sign_n, offset)
        return a_positive, a_negative


class ModePortSetup:
    """Calibration-style setup of a ModeSource / ModeMonitor: solves its modes on the current materials until the
    first step; later re-solves them (tracking the selected mode) once the line's materials drifted > _RESOLVE_DRIFT."""

    def __init__(self, physics, port_object):
        self.physics = physics
        self.port_object = port_object
        self.resolve_count = 0             # re-solves after the first step
        self._resolve_failed = False       # a failed re-solve keeps the previous modes and stops following
        self._warnings_reported = False    # mode warnings of a re-solve are appended once

    def apply(self, engine, materials):
        port_object = self.port_object
        if not port_object.solved or engine.n_step == 0:
            port_object.solve(self.physics)
            return
        if self._resolve_failed:
            return
        drift = port_object.port.material_drift(self.physics)   # one host transfer of the line rows
        if drift <= _RESOLVE_DRIFT:
            return
        label = port_object.port.label
        previous_warnings = list(port_object.warnings)
        try:
            port_object.resolve(self.physics)
        except ValueError as error:
            self._resolve_failed = True
            self.physics.runtime_warnings.append(
                f"{label}: re-solving its modes after the materials on its line changed by up to {drift:.2g} "
                f"(relative) failed at step {engine.n_step} ({error}); it keeps its previous modes and no longer "
                f"follows material changes")
            return
        self.resolve_count += 1
        notes = []
        if self.resolve_count == 1 and isinstance(port_object, ModeMonitor):
            notes.append(f"{label}: the materials on its line changed by up to {drift:.2g} (relative) during the run, "
                         f"so its modes were re-solved at step {engine.n_step} (and are at every further change above "
                         f"{_RESOLVE_DRIFT:g}); its spectra cover the whole run but are decomposed with the modes of "
                         f"the latest materials, which fits only fields that passed after the medium settled")
        new_warnings = [warning for warning in port_object.warnings if warning not in previous_warnings]
        if new_warnings and not self._warnings_reported:
            self._warnings_reported = True
            notes.extend(f"after a re-solve at step {engine.n_step}: {warning}" for warning in new_warnings)
        self.physics.runtime_warnings.extend(notes)

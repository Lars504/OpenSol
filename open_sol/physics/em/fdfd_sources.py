"""Phasor sources of the FDFD physics: add_to_rhs(rhs, physics, system) adds one feature's current at system.omega to
the complex128 (nx, ny) right-hand side of the run backend; glyph attributes as the FDTD sources (the Visualizer draws
them)."""
# Signs (derivation in harmonic.py): an FDTD current J_F / M_F is minus the textbook impressed current, so the FDFD
# keeps the FDTD amplitude meanings with Ampere j omega eps0 eps E = curl H + J_F, Faraday -j omega mu0 mu H = curl E
# - M_F:
#   ez rhs += -j omega mu0 J_F at the cells; hz rhs += dx_e (J_yF / eps_yface) - dy_e (J_xF / eps_xface)
#   (harmonic.face_current_rhs) - j omega eps0 M_F.
# H_z face sources also expose face_currents(physics, system) -> (J_x on hi-y, J_y on hi-x faces), which
# FDFDPhysics.face_fields adds to the curl (on a sheet face the curl alone is ~2/(k w_perp) times the field).
# A phasor P e^{j phase} is the real signal Re(P e^{j (omega t + phase)}): phase 0 = cos carrier (the FDTD sin carrier
# is phase -90). Total-field / scattered-field sources (PlaneWavePhasor, the mode source phasor) add only to the rhs,
# through system.operator; they have no face currents, so monitors must stay off their interface.
import cmath
import math

import numpy as np
from ...core.backend import array_module_of

from ...core.constants import C0, EPS0, ETA0, MU0
from ...core.mesh import has_uniform_spacing
from .harmonic import face_current_rhs
from .operators import axis_difference_weights
from .yee_grid import cpml_grading_cell_size, grid_cell_widths, sheet_launch_factor


def _on_backend(cached, host_value, xp):
    """`cached` when it already lives on xp's backend, else xp.asarray(host_value) (one transfer per source, at its
    first solve: sources are built before they know the run backend)."""
    if cached is not None and array_module_of(cached) is xp:
        return cached
    return xp.asarray(host_value)


def phase_factor(phase, label):
    """e^{j phase} of a finite phase [deg]; ValueError naming `label` otherwise."""
    phase = float(phase)
    if not math.isfinite(phase):
        raise ValueError(f"{label} phase must be a finite angle [deg], got {phase!r}")
    return cmath.exp(1j * math.radians(phase))


def _require_polarization(system, polarization, label):
    """ValueError when a source of one polarisation is solved in a system of the other."""
    if system.polarization != polarization:
        raise ValueError(f"{label} is an {polarization!r} source, but the system is {system.polarization!r}")


class PointCurrentPhasor:
    """E_z filament current I [A] in cell (ix, iy) (ez): rhs += -j omega mu0 I e^{j phase} / (dx_i dy_j), the cell
    area from yee_grid.grid_cell_widths."""

    def __init__(self, ix, iy, current, phase, cell_area):
        self.ix = int(ix)
        self.iy = int(iy)
        self.current = float(current)
        self.phase = float(phase)
        self._current_density = self.current * phase_factor(phase, 'Point source') / float(cell_area)   # J [A/m^2]

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, 'ez', 'PointCurrentPhasor')
        rhs[self.ix, self.iy] += -1j * system.omega * MU0 * self._current_density


class MagneticCurrentPhasor:
    """H_z magnetic line current K_m [V] in cell (ix, iy) (hz): rhs += -j omega eps0 K_m e^{j phase} / (dx_i dy_j), the
    cell area from yee_grid.grid_cell_widths."""

    def __init__(self, ix, iy, magnetic_current, phase, cell_area):
        self.ix = int(ix)
        self.iy = int(iy)
        self.magnetic_current = float(magnetic_current)
        self.phase = float(phase)
        # M [V/m^2]
        self._current_density = self.magnetic_current * phase_factor(phase, 'Magnetic line current') / float(cell_area)

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, 'hz', 'MagneticCurrentPhasor')
        rhs[self.ix, self.iy] += -1j * system.omega * EPS0 * self._current_density


class ElectricDipolePhasor:
    """In-plane electric dipole of cell (ix, iy) (hz): `faces` = placement.electric_dipole_faces (('ex' | 'ey', i, j,
    J [A/m^2], complex on a Bloch-wrapped face), ...) times e^{j phase}, through harmonic.face_current_rhs."""

    def __init__(self, ix, iy, faces, phase):
        faces = tuple((component, int(i), int(j), complex(current_density))
                      for component, i, j, current_density in faces)
        if not faces or any(face[0] not in ('ex', 'ey') for face in faces):
            raise ValueError(f"ElectricDipolePhasor needs at least one ('ex' | 'ey', i, j, J) face, got {faces}")
        self.ix = int(ix)
        self.iy = int(iy)
        self.faces = faces
        self.phase = float(phase)
        phasor = phase_factor(phase, 'Electric dipole')
        # component -> (face i, face j, complex J) host arrays, moved to the run backend at the first solve
        # (_device_currents): one scatter per component and solve.
        self._host_face_currents = {}
        for component in ('ex', 'ey'):
            selected = [face for face in faces if face[0] == component]
            if selected:
                self._host_face_currents[component] = (
                    np.asarray([face[1] for face in selected], dtype=np.int64),
                    np.asarray([face[2] for face in selected], dtype=np.int64),
                    np.asarray([face[3] * phasor for face in selected], dtype=np.complex128))
        self._device_currents = {}

    def _face_array(self, component, shape, xp):
        """complex128 (nx, ny) face currents of one component (E_x: hi-y faces, E_y: hi-x faces), None without any."""
        if component not in self._host_face_currents:
            return None
        self._device_currents[component] = tuple(
            _on_backend(cached, host, xp) for cached, host in zip(self._device_currents.get(component, (None,) * 3),
                                                                   self._host_face_currents[component]))
        face_i, face_j, currents = self._device_currents[component]
        face_currents = xp.zeros(shape, dtype=xp.complex128)
        face_currents[face_i, face_j] = currents   # distinct faces per component
        return face_currents

    def face_currents(self, physics, system):
        """(J_x on hi-y faces, J_y on hi-x faces) complex128 (nx, ny) FDTD-sign currents [A/m^2], None = none."""
        _require_polarization(system, 'hz', 'ElectricDipolePhasor')
        shape = (system.ops.nx, system.ops.ny)
        xp = array_module_of(system.cell_mu)
        return self._face_array('ex', shape, xp), self._face_array('ey', shape, xp)

    def add_to_rhs(self, rhs, physics, system):
        rhs += face_current_rhs(system, *self.face_currents(physics, system))


class SheetCurrentPhasor:
    """Sheet current radiating `amplitude` [V/m] to each side in the local medium (line source / gaussian beam): ez J_z
    through the line-cell centres, hz J_x (axis 'x') / J_y (axis 'y') on their hi faces (hi_face True); line_phase =
    host k.r [rad] per line cell under Bloch k != 0 (current times e^{-j k.r}: an oblique wave), None = 0."""

    def __init__(self, axis, sheet_line, amplitude, phase, polarization, line_phase=None, label='Sheet current'):
        if axis not in ('x', 'y'):
            raise ValueError(f"SheetCurrentPhasor axis must be 'x' or 'y', got {axis!r}")
        if polarization not in ('ez', 'hz'):
            raise ValueError(f"SheetCurrentPhasor polarization must be 'ez' or 'hz', got {polarization!r}")
        self.axis = axis
        self.label = label   # warnings (FDFDPhysics.finalize)
        self.position = int(sheet_line.line_cell)
        self.span = sheet_line.span_cells
        self.line = ((sheet_line.along, self.position) if axis == 'x' else (self.position, sheet_line.along))
        self.hi_face = polarization == 'hz'
        self.polarization = polarization
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        # 2 amplitude shape e^{j phase} / (eta0 w_perp); times the complex sqrt(eps / mu) of the system at omega
        # (sigma_e, sigma_m and poles included) and the discrete cos(kappa/2) (yee_grid.sheet_launch_factor): eta(omega)
        # K / (2 cos(kappa/2)) = amplitude exactly; the FDTD calibrations use its modulus at the centre frequency.
        peak_weight = 2.0 * self.amplitude * phase_factor(phase, 'Sheet current') / float(ETA0)
        self._normal_spacing = float(sheet_line.normal_width)
        line_weight = peak_weight * sheet_line.transverse_shape / self._normal_spacing
        if line_phase is not None:
            line_phase = np.asarray(line_phase, dtype=np.float64)
            if line_phase.shape != line_weight.shape:
                raise ValueError(f"SheetCurrentPhasor line_phase has {line_phase.size} cells, the line "
                                 f"{line_weight.size}")
            line_weight = line_weight * np.exp(-1j * line_phase)
        self._host_line_weight = np.asarray(line_weight, dtype=np.complex128)
        self._line_weight = None   # on the run backend from the first solve

    def _line_currents(self, line_eps, line_mu, omega):
        """complex128 sheet current densities [A/m^2] of the line cells / faces with complex eps and mu at omega."""
        xp = array_module_of(line_eps)
        self._line_weight = _on_backend(self._line_weight, self._host_line_weight, xp)
        launch_factor = sheet_launch_factor(line_eps * line_mu, omega, self._normal_spacing, xp=xp)
        return self._line_weight * xp.sqrt(line_eps / line_mu) * launch_factor

    def face_currents(self, physics, system):
        """hz: (J_x, None) for axis 'x' or (None, J_y) for axis 'y', complex128 (nx, ny) FDTD-sign face currents
        [A/m^2] from the system's complex face eps and the line cells' complex mu."""
        _require_polarization(system, 'hz', 'SheetCurrentPhasor face_currents')
        line = self.line
        face_eps = system.eps_hi_y_faces if self.axis == 'x' else system.eps_hi_x_faces   # at E_x / at E_y
        xp = array_module_of(face_eps)
        currents = xp.zeros((system.ops.nx, system.ops.ny), dtype=xp.complex128)
        currents[line] = self._line_currents(face_eps[line], system.cell_mu[line], system.omega)
        return (currents, None) if self.axis == 'x' else (None, currents)

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, self.polarization, 'SheetCurrentPhasor')
        if self.polarization == 'hz':
            rhs += face_current_rhs(system, *self.face_currents(physics, system))
            return
        line = self.line
        sheet_current = self._line_currents(system.cell_eps[line], system.cell_mu[line], system.omega)
        rhs[line] += (-1j * system.omega * MU0) * sheet_current


class PlaneWavePhasor:
    """+x plane wave on the total-field cell box [x0, x1) x [y0, y1) (uniform or graded grids, box edges in one uniform
    medium, that of outline cell (x0 - 1, y0)): incident E_z (ez) or in-plane |E| (hz, H_z = sqrt(eps / mu) |E| /
    eta0) = amplitude e^{j phase} [V/m] at the centre of column x0 and y = 0 (grid lo edge); cell_widths = host
    (dx_cells, dy_cells) of the operator (yee_grid.grid_cell_widths); full_period_y = every y cell of a periodic y axis
    (no y edges); bloch_ky [rad/m] != 0 tilts it to the oblique wave e^{-j (k_x x + ky y)} (uniform y cells only).
    open_faces: y sides opening onto an even mirror plane (no outline there; the pinned ghost row and the zero
    plane-face weight already leave them uncorrected)."""
    # TFSF (spec section 0): the unknown u is the total field u_tot on the box Q and the scattered field u_tot - u_inc
    # outside. Substituting u = u_tot - (1 - Q) u_inc into A u_tot = b gives A u = b - A u_inc + A (Q u_inc).
    # A u_inc is kept on the total-field rows and dropped on the scattered-field rows:
    #   b += A (Q u_inc) - Q (A u_inc) = (1 - Q) A (Q u_inc) - Q A ((1 - Q) u_inc),
    # evaluated in the second form (no cancellation inside the box); non-zero only on the rows along the box edge (a
    # full-period box has no y edges: the operator wraps along y). The scattered field u - Q u_inc then solves A u_s =
    # b - Q A u_inc, exact whenever u_inc is a homogeneous solution of the background rows of Q; it reads u_inc only on
    # columns x0 - 1 .. x1 and rows y0 - 1 .. y1, so u_inc = f(x_i) g(y_j) with g = 1 (ky = 0: a y-constant column
    # cancels the y couplings on any widths) or e^{-j ky y_j} (Bloch ky: the eigenvector of the uniform periodic y line,
    # lambda_y = (2 / dy)^2 sin^2(ky dy / 2)) and f the x line of the solved operator (w_e, w_h of
    # operators.axis_difference_weights, background face weight w_x and cell term c):
    #   w_x w_e_i [w_h_i (f_{i+1} - f_i) - w_h_{i-1} (f_i - f_{i-1})] + (c - w_y lambda_y) f_i = 0.
    # Uniform columns x0 - 1 .. x1: the closed form f = e^{-j k_x (x_i - x_c0)}, (2 / dx)^2 sin^2(k_x dx / 2) = (c -
    # w_y lambda_y) / w_x. Graded: f(x0) = 1 and f(x0 - 1) = e^{j k_x h} (k_x of the uniform wave on their centre spacing
    # h: exactly right-going when the upstream cells are uniform, else a small backward part, still exact), then f_{i+1}
    # from row i up to x1 (the closed form avoids the march's growth of round-off in a lossy background).

    def __init__(self, x0, x1, y0, y1, amplitude, phase, polarization, cell_widths, full_period_y=False,
                 bloch_ky=0.0, open_faces=()):
        self.open_faces = frozenset(open_faces)
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        y_lo_ok = y0 == 0 if full_period_y else y0 >= 1
        if not (1 <= x0 < x1 and y_lo_ok and y0 < y1):
            raise ValueError(f"TFSF box [{x0}, {x1}) x [{y0}, {y1}) must be non-empty and keep one scattered-field "
                             f"cell on the lo sides (x0 >= 1, y0 >= 1; y0 = 0 with full_period_y)")
        if polarization not in ('ez', 'hz'):
            raise ValueError(f"PlaneWavePhasor polarization must be 'ez' or 'hz', got {polarization!r}")
        dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
        if not (dx_cells.ndim == dy_cells.ndim == 1 and x1 < dx_cells.size and np.all(np.isfinite(dx_cells))
                and np.all(dx_cells > 0.0) and np.all(np.isfinite(dy_cells)) and np.all(dy_cells > 0.0)):
            raise ValueError(f"PlaneWavePhasor cell_widths must be finite positive (nx,) / (ny,) widths [m] reaching "
                             f"column x1 = {x1}, got shapes {dx_cells.shape} / {dy_cells.shape}")
        self.bloch_ky = float(bloch_ky)
        if not math.isfinite(self.bloch_ky):
            raise ValueError(f"PlaneWavePhasor bloch_ky must be finite [rad/m], got {bloch_ky!r}")
        if self.bloch_ky != 0.0 and not has_uniform_spacing(dy_cells):
            raise ValueError(f"Plane wave: an oblique TFSF wave (Bloch ky = {self.bloch_ky:.6g} rad/m) needs uniform y "
                             f"cells (got {dy_cells.min():.4g} .. {dy_cells.max():.4g} m): e^(-j ky y) is no Bloch wave "
                             f"of a graded periodic y line; refine along x only, or set bloch_ky = 0")
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.full_period_y = bool(full_period_y)
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        self.polarization = polarization
        self._line_widths = dx_cells[x0 - 1:x1 + 1].copy()   # columns x0 - 1 .. x1 (the ones u_inc is read on)
        self._uniform_line = has_uniform_spacing(self._line_widths)
        self.row_spacing = float(dy_cells[0])   # lambda_y of an oblique wave (uniform y then)
        self._row_centres = np.cumsum(dy_cells) - 0.5 * dy_cells   # from the grid lo edge, as sources.bloch_line_phase
        self._line_weights = None   # (omega, w_e, w_h) of columns x0 - 1 .. x1 (axis_difference_weights)
        self._incident_field = self.amplitude * phase_factor(phase, 'Plane wave')   # E_z (ez) / |E| (hz) [V/m]

    def _normal_wavenumber(self, normal_squared, spacing, background_wavenumber):
        """Complex k_x [rad/m] of (2 / h)^2 sin^2(k_x h / 2) = normal_squared on spacing h; ValueError for an evanescent
        oblique wave or at pi or fewer cells of the largest line width per wavelength along x."""
        if self.bloch_ky != 0.0 and normal_squared.real <= 0.0:
            raise ValueError(f"Plane wave: Bloch ky = {self.bloch_ky:.6g} rad/m reaches or exceeds the wavenumber "
                             f"{background_wavenumber:.6g} rad/m of the background, so the incident wave is evanescent "
                             f"(no real incidence angle); lower bloch_ky or the wavelength")
        root = cmath.sqrt(normal_squared)   # principal branch: Re > 0, Im <= 0 when lossy
        largest = float(self._line_widths.max())
        if 0.5 * largest * root.real >= 1.0:
            wavelength = 2.0 * math.pi / root.real
            raise ValueError(f"Plane wave: {math.pi / (0.5 * largest * root.real):.3g} cells per wavelength along x "
                             f"(largest cell {largest:.4g} m on the box columns, wavelength along x in the background "
                             f"{wavelength:.4g} m) is at most pi, where the FDFD grid carries no propagating wave; "
                             f"refine the mesh (>= 10 cells per wavelength)")
        return 2.0 / spacing * cmath.asin(0.5 * spacing * root)

    def _incident_line(self, physics, omega, face_weight, normal_squared, background_wavenumber):
        """Host complex128 f on columns x0 - 1 .. x1, f(x0) = 1: the closed form on uniform columns, else marched on the
        solved operator's x line (class comment)."""
        widths = self._line_widths
        if self._uniform_line:
            spacing = float(widths[1])
            wavenumber = self._normal_wavenumber(normal_squared, spacing, background_wavenumber)
            return np.exp(-1j * wavenumber * spacing * (np.arange(widths.size) - 1.0))
        if self._line_weights is None or self._line_weights[0] != omega:
            weights_e, weights_h = axis_difference_weights(grid_cell_widths(physics.domain.grid),
                                                           physics.boundary_config,
                                                           cpml_grading_cell_size(physics.domain), 'x', omega)
            line = slice(self.x0 - 1, self.x1 + 1)
            self._line_weights = (omega, weights_e[line].copy(), weights_h[line].copy())
        _, weights_e, weights_h = self._line_weights
        spacing = 0.5 * float(widths[0] + widths[1])   # centre spacing of the start pair x0 - 1, x0
        wavenumber = self._normal_wavenumber(normal_squared, spacing, background_wavenumber)
        cell_term = normal_squared * face_weight   # c - w_y lambda_y
        line = np.empty(widths.size, dtype=np.complex128)
        line[0], line[1] = cmath.exp(1j * wavenumber * spacing), 1.0
        for k in range(1, widths.size - 1):   # row of column x0 - 1 + k gives column x0 + k
            step = (weights_h[k - 1] * (line[k] - line[k - 1])
                    - cell_term * line[k] / (face_weight * weights_e[k])) / weights_h[k]
            line[k + 1] = line[k] + step
        return line

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, self.polarization, 'PlaneWavePhasor')
        nx, ny = system.ops.nx, system.ops.ny
        if self.full_period_y and not (system.periodic_y and self.y1 == ny):
            raise ValueError(f"TFSF box [{self.x0}, {self.x1}) x [{self.y0}, {self.y1}) spans the y period, which needs "
                             f"a periodic y axis of {self.y1} cells (got {ny}, periodic_y={system.periodic_y})")
        if self.x1 >= nx or (self.y1 >= ny and not self.full_period_y):
            raise ValueError(f"TFSF box [{self.x0}, {self.x1}) x [{self.y0}, {self.y1}) needs a scattered-field cell on "
                             f"the hi sides of a {nx} x {ny} grid")
        # Background = the complex eps and mu of outline cell (x0 - 1, y0) at omega: ez eps_r with face weights 1 / mu
        # and cell term k0^2 eps; hz eps_yy (E_y, x faces) / eps_xx (E_x, y faces) weights and cell term k0^2 mu.
        reference_cell = (self.x0 - 1, self.y0)
        mu_background = complex(system.cell_mu[reference_cell])
        if self.polarization == 'ez':
            eps_background = complex(system.cell_eps[reference_cell])
            x_weight = y_weight = 1.0 / mu_background
            cell_term = (system.omega / C0) ** 2 * eps_background
        else:
            eps_background = complex(system.cell_eps_yy[reference_cell])
            x_weight, y_weight = 1.0 / eps_background, 1.0 / complex(system.cell_eps_xx[reference_cell])
            cell_term = (system.omega / C0) ** 2 * mu_background
        transverse = (2.0 / self.row_spacing * math.sin(0.5 * self.bloch_ky * self.row_spacing)) ** 2
        normal_squared = (cell_term - y_weight * transverse) / x_weight   # (2 / dx)^2 sin^2(k_x dx / 2) when uniform
        background_wavenumber = (system.omega / C0 * cmath.sqrt(eps_background * mu_background)).real
        line = self._incident_line(physics, system.omega, x_weight, normal_squared, background_wavenumber)
        # The unknown at the centre of column x0: E_z (ez) or H_z = sqrt(eps / mu) |E| / eta0 (hz; exact for the
        # discrete wave, oblique too: |E| = (|K| / (omega eps0 eps)) |H_z| with |K|^2 = eps mu k0^2) [A/m].
        reference_phasor = (self._incident_field if self.polarization == 'ez' else
                            self._incident_field * cmath.sqrt(eps_background / mu_background) / float(ETA0))
        # u_inc[i, j] = reference f(x_i) e^{-j ky y_j} on columns x0 - 1 .. x1 (0 elsewhere: never read)
        xp = array_module_of(rhs)
        incident_columns = xp.zeros(nx, dtype=xp.complex128)
        incident_columns[self.x0 - 1:self.x1 + 1] = xp.asarray(reference_phasor * line)
        row_phases = xp.asarray(np.exp(-1j * self.bloch_ky * self._row_centres))
        incident = xp.outer(incident_columns, row_phases)
        box = (slice(self.x0, self.x1), slice(self.y0, self.y1))
        inside_incident = xp.zeros((nx, ny), dtype=xp.complex128)   # Q u_inc
        inside_incident[box] = incident[box]
        incident -= inside_incident
        outside_incident = incident   # (1 - Q) u_inc, in place (no extra full-grid temporary)
        operator = system.operator
        correction = (operator @ inside_incident.ravel()).reshape(nx, ny)   # kept on the scattered-field rows
        correction[box] = -(operator @ outside_incident.ravel()).reshape(nx, ny)[box]
        rhs += correction

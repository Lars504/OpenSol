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
# is phase -90). Total-field / scattered-field sources (PlaneWavePhasor, FocusedBeamPhasor, the mode source phasor) add
# only to the rhs, through system.operator; they have no face currents, so monitors must stay off their interface.
import cmath
import math

import numpy as np
from ...core.backend import array_module_of, to_host

from ...core.constants import C0, EPS0, ETA0, MU0
from ...core.mesh import has_uniform_spacing
from .focused_beam import (FOCUS_TOLERANCE, NORMAL_RANGE, beam_rows, discrete_waves, interface_power, line_medium)
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
    amplitude_params = ('current', 'phase')   # feature values of set_amplitude, in its argument order (sweep reuse)

    def __init__(self, ix, iy, current, phase, cell_area):
        self.ix = int(ix)
        self.iy = int(iy)
        self._cell_area = float(cell_area)
        self.set_amplitude(current, phase)

    def set_amplitude(self, current, phase):
        """New current [A] and phase [deg] in place (a reused sweep build)."""
        self.current = float(current)
        self.phase = float(phase)
        self._current_density = self.current * phase_factor(phase, 'Point source') / self._cell_area   # J [A/m^2]

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, 'ez', 'PointCurrentPhasor')
        rhs[self.ix, self.iy] += -1j * system.omega * MU0 * self._current_density


class MagneticCurrentPhasor:
    """H_z magnetic line current K_m [V] in cell (ix, iy) (hz): rhs += -j omega eps0 K_m e^{j phase} / (dx_i dy_j), the
    cell area from yee_grid.grid_cell_widths."""
    amplitude_params = ('magnetic_current', 'phase')   # as PointCurrentPhasor

    def __init__(self, ix, iy, magnetic_current, phase, cell_area):
        self.ix = int(ix)
        self.iy = int(iy)
        self._cell_area = float(cell_area)
        self.set_amplitude(magnetic_current, phase)

    def set_amplitude(self, magnetic_current, phase):
        """New magnetic current [V] and phase [deg] in place (a reused sweep build)."""
        self.magnetic_current = float(magnetic_current)
        self.phase = float(phase)
        # M [V/m^2]
        self._current_density = (self.magnetic_current * phase_factor(phase, 'Magnetic line current')
                                 / self._cell_area)

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, 'hz', 'MagneticCurrentPhasor')
        rhs[self.ix, self.iy] += -1j * system.omega * EPS0 * self._current_density


class ElectricDipolePhasor:
    """In-plane electric dipole of cell (ix, iy) (hz): `faces` = placement.electric_dipole_faces (('ex' | 'ey', i, j,
    J [A/m^2], complex on a Bloch-wrapped face), ...) times e^{j phase}, through harmonic.face_current_rhs; dipole_moment
    [A] = the moment the faces were placed for (the faces scale linearly with it: set_amplitude)."""
    amplitude_params = ('dipole_moment', 'phase')   # as PointCurrentPhasor

    def __init__(self, ix, iy, faces, phase, dipole_moment=1.0):
        faces = tuple((component, int(i), int(j), complex(current_density))
                      for component, i, j, current_density in faces)
        if not faces or any(face[0] not in ('ex', 'ey') for face in faces):
            raise ValueError(f"ElectricDipolePhasor needs at least one ('ex' | 'ey', i, j, J) face, got {faces}")
        self.ix = int(ix)
        self.iy = int(iy)
        self._placed_faces = faces
        self._placed_moment = float(dipole_moment)
        if not (math.isfinite(self._placed_moment) and self._placed_moment != 0.0):
            raise ValueError(f"ElectricDipolePhasor dipole_moment must be finite and non-zero [A], got {dipole_moment!r}")
        self.set_amplitude(dipole_moment, phase)

    def set_amplitude(self, dipole_moment, phase):
        """New dipole moment [A] and phase [deg] in place (a reused sweep build); the next solve uploads the currents."""
        self.dipole_moment = float(dipole_moment)
        self.phase = float(phase)
        scale = self.dipole_moment / self._placed_moment
        self.faces = tuple((component, i, j, current_density * scale)
                           for component, i, j, current_density in self._placed_faces)
        phasor = phase_factor(phase, 'Electric dipole')
        # component -> (face i, face j, complex J) host arrays, moved to the run backend at the first solve
        # (_device_currents): one scatter per component and solve.
        self._host_face_currents = {}
        for component in ('ex', 'ey'):
            selected = [face for face in self.faces if face[0] == component]
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
    amplitude_params = ('amplitude', 'phase')   # as PointCurrentPhasor

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
        # 2 amplitude shape e^{j phase} / (eta0 w_perp); times the complex sqrt(eps / mu) of the system at omega
        # (sigma_e, sigma_m and poles included) and the discrete cos(kappa/2) (yee_grid.sheet_launch_factor): eta(omega)
        # K / (2 cos(kappa/2)) = amplitude exactly; the FDTD calibrations use its modulus at the centre frequency.
        self._normal_spacing = float(sheet_line.normal_width)
        unit_weight = 2.0 * sheet_line.transverse_shape / (float(ETA0) * self._normal_spacing)
        if line_phase is not None:
            line_phase = np.asarray(line_phase, dtype=np.float64)
            if line_phase.shape != unit_weight.shape:
                raise ValueError(f"SheetCurrentPhasor line_phase has {line_phase.size} cells, the line "
                                 f"{unit_weight.size}")
            unit_weight = unit_weight * np.exp(-1j * line_phase)
        self._unit_line_weight = np.asarray(unit_weight, dtype=np.complex128)   # per unit amplitude at phase 0
        self.set_amplitude(amplitude, phase)

    def set_amplitude(self, amplitude, phase):
        """New amplitude [V/m] and phase [deg] in place (a reused sweep build); the next solve uploads the weights."""
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        self._host_line_weight = self._unit_line_weight * (self.amplitude * phase_factor(phase, 'Sheet current'))
        self._line_weight = None   # on the run backend from the next solve

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
    amplitude_params = ('amplitude', 'phase')   # as PointCurrentPhasor
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
        self.polarization = polarization
        self._line_widths = dx_cells[x0 - 1:x1 + 1].copy()   # columns x0 - 1 .. x1 (the ones u_inc is read on)
        self._uniform_line = has_uniform_spacing(self._line_widths)
        self.row_spacing = float(dy_cells[0])   # lambda_y of an oblique wave (uniform y then)
        self._row_centres = np.cumsum(dy_cells) - 0.5 * dy_cells   # from the grid lo edge, as sources.bloch_line_phase
        self._line_weights = None   # (omega, w_e, w_h) of columns x0 - 1 .. x1 (axis_difference_weights)
        self.set_amplitude(amplitude, phase)

    def set_amplitude(self, amplitude, phase):
        """New incident amplitude [V/m] and phase [deg] in place (a reused sweep build)."""
        self.amplitude = float(amplitude)
        self.phase = float(phase)
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


_MEDIUM_RTOL = 1e-9   # largest relative spread of an operator value over the rows a focused beam reads


class FocusedBeamPhasor:
    """One-sided focused beam on a line (focused_beam.py: a Gaussian spectrum of the discrete plane waves of the line
    medium, all in phase at `focus` = drawing-frame (x, y) [m]): `power` [W/m] in the unfolded structure toward its
    direction, incident out-of-plane field (ez E_z, hz H_z) at the focus = |u| e^{j phase}; `line` = its
    focused_beam.FocusedBeamLine (TFSF rows: the line cells, total field, and their upstream neighbours). The incident
    rows are computed per solved omega and line medium (read from the assembled operator on the rows line_cell +-
    NORMAL_RANGE: one uniform medium, no PEC, conducting sheet, eps_xy or gyration there, ValueError otherwise) and kept
    on the run backend. Attributes the Brillouin drive reads: name, incident_power, focus, direction, wavelength,
    polarization; axis / position / span are the line glyph (as the mode source)."""
    amplitude_params = ('power', 'phase')   # as PointCurrentPhasor
    # Right-hand side (the ModeSourcePhasor identity, TFSF sign of PlaneWavePhasor): b = A (Q u_inc) - Q (A u_inc), Q =
    # the total-field half plane, u_inc on the line row c and the upstream row c' only. Adjacent-row stencils (5 points,
    # 9 with eps_xy) make b non-zero on those two rows alone: b[c'] = (A u_c)[c'], b[c] = (A u_c)[c] - (A (u_c +
    # u_c'))[c] (two operator products of the flat incident vector).

    def __init__(self, line, power, waist, focus, numerical_aperture, phase, polarization, name):
        label = line.label
        self.line = line
        self.label = label
        self.name = str(name)
        self.polarization = polarization
        self.direction = line.direction
        self.axis, self.position, self.span = line.axis, line.line_cell, line.span
        waist = float(waist)
        if not (math.isfinite(waist) and waist > 0.0):
            raise ValueError(f"{label} waist must be finite and > 0 [m], got {waist!r}")
        self.waist = waist
        if numerical_aperture is not None:
            numerical_aperture = float(numerical_aperture)
            if not (math.isfinite(numerical_aperture) and numerical_aperture > 0.0):
                raise ValueError(f"{label} numerical_aperture must be finite and > 0 (or None), got "
                                 f"{numerical_aperture!r}")
        self.numerical_aperture = numerical_aperture
        focus = tuple(float(value) for value in focus)
        if len(focus) != 2 or not all(math.isfinite(value) for value in focus):
            raise ValueError(f"{label} focus must be finite (x, y) [m], got {focus!r}")
        self.focus = focus
        along_focus = line.along_focus(focus)
        for plane in line.mirror_planes:
            if abs(along_focus - plane) > FOCUS_TOLERANCE * line.along_spacing:
                raise ValueError(f"{label}: its line ends on the mirror plane at {line.axis} = {plane:.6g} m, so the "
                                 f"focus must lie on it (focus {line.axis} = {along_focus:.6g} m): the image completes "
                                 'the beam only when both halves meet there')
        self.solved_omega = None   # rad/s of the last right-hand side
        self._unit = None          # (omega, LineMedium, unit line row, unit upstream row, unit power) on the host
        self._device_rows = None   # (line values, upstream values) of the run backend at power / phase
        self._device_index = None  # (line index, upstream index) of the run backend
        self.set_amplitude(power, phase)

    @property
    def incident_power(self) -> float:
        """Incident power [W/m] of the beam in the unfolded structure (exact: the scale of the injected rows)."""
        return self.power

    @property
    def wavelength(self) -> float:
        """Vacuum wavelength [m] of the last right-hand side; ValueError before the first solve."""
        if self.solved_omega is None:
            raise ValueError(f'{self.label}: not solved yet (no wavelength)')
        return 2.0 * math.pi * float(C0) / self.solved_omega

    def set_amplitude(self, power, phase):
        """New power [W/m] (finite, > 0: ValueError otherwise) and phase [deg] in place (a reused sweep build); the
        next right-hand side rescales the kept unit rows."""
        power = float(power)
        if not (math.isfinite(power) and power > 0.0):
            raise ValueError(f"{self.label} power must be finite and > 0 [W/m], got {power!r}")
        self.power = power
        self.phase = float(phase)
        self._phasor = phase_factor(phase, self.label)
        self._device_rows = None

    def moved(self, offset):
        """New FocusedBeamPhasor on the same line with the focus moved by offset = (dx, dy) [m] (no snapping: only the
        wave phases change); ValueError for an along-line offset when the line ends on a mirror plane."""
        dx, dy = (float(value) for value in offset)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValueError(f"{self.label}: a focus offset must be finite (dx, dy) [m], got {offset!r}")
        along_shift = dx if self.axis == 'x' else dy
        if self.line.mirror_planes and along_shift != 0.0:
            raise ValueError(f"{self.label}: an offset along the line ({self.axis} {along_shift:.6g} m) moves the focus "
                             'off the mirror plane the line ends on; scan across the line only, or model the whole '
                             'structure')
        moved = FocusedBeamPhasor(self.line, self.power, self.waist, (self.focus[0] + dx, self.focus[1] + dy),
                                  self.numerical_aperture, self.phase, self.polarization, self.name)
        moved._device_index = self._device_index   # same rows: no new index upload per scan position
        return moved

    def _medium(self, physics, system):
        """LineMedium of the assembled operator on the rows the beam reads (uniform values; ValueError otherwise)."""
        line = self.line
        c = line.line_cell
        rows = slice(c - NORMAL_RANGE, c + NORMAL_RANGE + 1)
        inner_rows = slice(c - NORMAL_RANGE, c + NORMAL_RANGE)     # the hi faces between those rows
        span = slice(*line.span)
        inner_span = slice(line.span[0], line.span[1] - 1)         # the hi faces between the span cells

        def cells(along, normal):
            return (along, normal) if line.axis == 'x' else (normal, along)

        hi_normal, hi_along = ((system.weight_hi_y_faces, system.weight_hi_x_faces) if line.normal == 'y'
                               else (system.weight_hi_x_faces, system.weight_hi_y_faces))
        cell_material = system.cell_eps if system.polarization == 'ez' else system.cell_mu
        blocks = {'face weight across the line': hi_normal[cells(span, inner_rows)],
                  'face weight along the line': hi_along[cells(inner_span, rows)],
                  'eps_r' if system.polarization == 'ez' else 'mu_r': cell_material[cells(span, rows)]}
        if system.face_fractions is not None:   # hz conformal PEC: the operator scales weights and mass by them
            fraction_x, fraction_y = system.face_fractions
            normal_fraction, along_fraction = ((fraction_y, fraction_x) if line.normal == 'y'
                                               else (fraction_x, fraction_y))
            blocks['free face fraction across the line'] = normal_fraction[cells(span, inner_rows)]
            blocks['free face fraction along the line'] = along_fraction[cells(inner_span, rows)]
        if system.cell_area_fractions is not None:
            blocks['free cell area (conformal PEC)'] = system.cell_area_fractions[cells(span, rows)]
        where = (f'on the cells {line.axis} [{line.span[0]}, {line.span[1]}) x {line.normal} [{rows.start}, '
                 f'{rows.stop}) around the line')
        if physics.pec_cells is not None and bool(physics.xp.any(physics.pec_cells[cells(span, rows)])):
            raise ValueError(f'{self.label}: PEC cells {where}; keep the line in one dielectric medium')
        for name, values in (('eps_xy', system.cell_eps_xy), ('gyration_z', system.cell_gyration)):
            if values is not None and bool(physics.xp.any(values[cells(span, rows)] != 0.0)):
                raise ValueError(f'{self.label}: a non-zero {name} {where}: the beam is a sum of plane waves of a '
                                 'diagonal medium; keep the line in an isotropic, unrotated medium')
        for sheet in getattr(physics, 'conducting_sheets', ()):
            if _slices_overlap(sheet.line, cells(span, rows), line.shape):
                raise ValueError(f'{self.label}: a conducting sheet crosses the cells {where}; move the line off it')
        values = {}
        for name, block in blocks.items():
            host = np.asarray(to_host(block), dtype=np.complex128).ravel()
            reference = host[0]
            spread = float(np.max(np.abs(host - reference)))
            if reference == 0.0 or not (np.all(np.isfinite(host)) and spread <= _MEDIUM_RTOL * abs(reference)):
                raise ValueError(f'{self.label}: the line medium is not uniform {where} ({name} varies by '
                                 f'{spread:.3g} around {complex(reference):.6g}): the launch is exact only in one '
                                 'homogeneous medium there (interfaces, poles of another material, subpixel smoothing '
                                 'of a nearby interface, wall rows); move the line into a uniform region')
            values[name] = complex(reference)
        for name in [name for name in values if name.startswith('free ')]:   # conformal fractions: whole cells only
            if values[name] != 1.0:
                raise ValueError(f'{self.label}: conformal PEC cut cells {where} ({name} {values[name].real:.3g}); '
                                 'keep the line away from the metal')
        weights = list(values.values())   # across, along, cell material (insertion order)
        return line_medium(system.polarization, system.omega, weights[0], weights[1], weights[2])

    def _unit_rows(self, physics, system):
        """(LineMedium, unit line row, unit upstream row, unit power [W/m]) of the system's omega and line medium
        (recomputed when either changed)."""
        medium = self._medium(physics, system)
        unit = self._unit
        if unit is not None and unit[0] == system.omega and unit[1] == medium:
            return unit[1:]
        line = self.line
        waves = discrete_waves(line.span[1] - line.span[0], line.along_spacing, line.normal_spacing, medium,
                               self.waist, self.numerical_aperture, self.label)
        line_row, upstream_row = beam_rows(waves, line.along_centres, (line.line_coordinate, line.upstream_coordinate),
                                           line.along_focus(self.focus), line.normal_focus(self.focus), line.sign)
        unit_power = interface_power(line, medium, line_row, upstream_row)
        if not (math.isfinite(unit_power) and unit_power > 0.0):
            raise ValueError(f'{self.label}: the incident beam carries no power toward {self.direction} '
                             f'({unit_power:.3g} W/m per unit spectrum); check the line medium (lossless, eps > 0)')
        self._unit = (float(system.omega), medium, line_row, upstream_row, unit_power)
        self._device_rows = None
        return self._unit[1:]

    def _scale(self, unit_power):
        """Complex amplitude of the unit spectrum: sqrt(power / unit power) e^{j phase}."""
        return math.sqrt(self.power / unit_power) * self._phasor

    def incident_rows(self, physics, system, normal_cells):
        """Host complex128 (len(normal_cells), n) incident field u_inc at power / phase on the span cells of the given
        full-grid normal cells (the discrete waves continued off the TFSF rows: self-tests, references)."""
        _require_polarization(system, self.polarization, 'FocusedBeamPhasor')
        medium, _, _, unit_power = self._unit_rows(physics, system)
        line = self.line
        waves = discrete_waves(line.span[1] - line.span[0], line.along_spacing, line.normal_spacing, medium,
                               self.waist, self.numerical_aperture, self.label)
        normal_centres = physics.domain.cell_centers()['xy'.index(line.normal)]
        coordinates = np.asarray(normal_centres, dtype=np.float64)[np.asarray(normal_cells, dtype=np.int64)]
        return self._scale(unit_power) * beam_rows(waves, line.along_centres, coordinates,
                                                   line.along_focus(self.focus), line.normal_focus(self.focus),
                                                   line.sign)

    def add_to_rhs(self, rhs, physics, system):
        _require_polarization(system, self.polarization, 'FocusedBeamPhasor')
        line = self.line
        if tuple(rhs.shape) != line.shape or not rhs.flags.c_contiguous:
            raise ValueError(f'{self.label}: rhs must be a C-contiguous array of the grid {line.shape} of its line, got '
                             f'shape {tuple(rhs.shape)}')
        _, line_row, upstream_row, unit_power = self._unit_rows(physics, system)
        xp = array_module_of(rhs)
        if self._device_rows is None or array_module_of(self._device_rows[0]) is not xp:
            scale = self._scale(unit_power)
            self._device_rows = (xp.asarray(scale * line_row), xp.asarray(scale * upstream_row))
        cached = self._device_index or (None, None)
        self._device_index = (_on_backend(cached[0], line.line_index, xp), _on_backend(cached[1], line.upstream_index,
                                                                                        xp))
        line_index, upstream_index = self._device_index
        line_values, upstream_values = self._device_rows
        operator = system.operator
        flat_rhs = rhs.reshape(-1)   # a view of the C-contiguous (nx, ny) rhs
        incident = xp.zeros(flat_rhs.size, dtype=xp.complex128)
        incident[line_index] = line_values
        inside = operator @ incident            # A (Q u_inc): u_inc on the line row
        incident[upstream_index] = upstream_values
        both = operator @ incident              # A u_inc of both rows
        del incident
        flat_rhs[upstream_index] += inside[upstream_index]
        flat_rhs[line_index] += inside[line_index] - both[line_index]
        self.solved_omega = float(system.omega)


def _slices_overlap(first, second, shape):
    """Whether two (index | slice, index | slice) cell selections of an (nx, ny) grid share a cell."""
    for part_a, part_b, size in zip(first, second, shape):
        lo_a, hi_a = _index_range(part_a, size)
        lo_b, hi_b = _index_range(part_b, size)
        if hi_a <= lo_b or hi_b <= lo_a:
            return False
    return True


def _index_range(part, size):
    """Half-open (lo, hi) of an int or step-1 slice along an axis of `size` cells."""
    if isinstance(part, slice):
        lo, hi, _ = part.indices(size)
        return lo, hi
    index = int(part)
    return index, index + 1

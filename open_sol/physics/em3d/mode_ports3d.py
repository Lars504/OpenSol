"""Mode ports of em_fdtd_3d on a plane normal to x, y or z: A5 cross-section modes (physics.em.cross_section_modes) of
one cell layer, the one-sided TFSF ModeSource3D, the ModeMonitor3D decomposition and their result dicts."""
# Port frame: normal axis d, transverse (u, v) = ((d + 1) % 3, (d + 2) % 3) = the solver's local (x, y), local z = d
# (cyclic: right-handed). A layer of a 3D array in (u, v) order is the grid slice, transposed for d = 1 (_plane). The
# engine3d layout projects onto the solver layout (E_u / H_v on hi-u faces, E_v / H_u on hi-v faces, E_d and mu at the
# cell): materials = the layer's cells (eps_r - j sigma_e / (omega eps0) + pole chi, mu_r, PEC from the material table).
# Solved at the numerical frequency (2/dt) sin(omega dt/2) with the materials at omega: the solver's beta is then the
# discrete K = (2/D) sin(beta_d D/2) of the normal spacing D, so beta_d = (2/D) arcsin(K D/2) (n_eff = beta_d / k0).
# Boundaries: cpml / periodic faces as such (a periodic axis with its Bloch k: the Bloch mode at that k, which the
# source drives into the real set like a dipole; DFT monitors read its +k response), pec / pmc faces as natural ends
# (a pec wall's layer is PEC cells); the engine still updates the last E face row of a non-periodic axis, which the
# solver drops: a mode reaching a pec / pmc face is approximate there. A plane (PlaneFlux3D) reads E averaged across it (cos(beta_d D/2) of the centre field) and
# H on it, so a mode of solver power 1 W reads cos(omega dt/2) cos(beta_d D/2) W of discrete flux: the source scales by
# its inverse at f0 (a P W source reads P on flux planes), the monitor reports that flux-equivalent power.
import cmath
import hashlib
import math
from dataclasses import dataclass

import numpy as np
from ...core.backend import to_host
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver

from ...core.constants import C0, DTYPE, EPS0, MU0
from ..em.boundaries import BCType
from ..em.cross_section_modes import CrossSectionMaterials, mode_overlap, solve_cross_section_modes
from ..em.mode_ports import _port_correction, _quadrature_warnings
from ..em.results import _normalizing_spectrum, _per_cw, _shared_waveform_spectrum
from .boundaries3d import AXES
from .monitors3d import PlaneFlux3D

MODE_DIRECTIONS = ('positive', 'negative')
MODE_TABLE_COLUMNS = ('mode_number', 'n_eff', 'n_eff_imag', 'loss_db_per_cm', 'te_fraction', 'effective_area',
                      'pml_fraction', 'selected')
MODE_TABLE_UNITS = ('1', '1', '1', 'dB/cm', '1', 'm^2', '1', '')
_TRACKING_WARN_OVERLAP = 0.9   # monitor: a followed mode overlapping its neighbour frequency's below this warns
_PML_WARN_FRACTION = 1e-2      # a selected mode with more of its energy in the transverse PML warns
_NOT_SOLVED = 'mode not solved yet: finalize or step first'
AXIS_PERMITTIVITY = ('eps_xx', 'eps_yy', 'eps_r')   # MaterialGrid array of E_x / E_y / E_z (eps_r if not allocated)


def port_axes(normal_axis):
    """(u, v) transverse grid axes of a port normal to normal_axis (the solver's local x, y)."""
    return (normal_axis + 1) % 3, (normal_axis + 2) % 3


def _plane(array, normal_axis, index):
    """View of layer `index` along normal_axis in (u, v) order (cupy or numpy)."""
    key = [slice(None)] * 3
    key[normal_axis] = int(index)
    plane = array[tuple(key)]
    return plane.T if normal_axis == 1 else plane


@dataclass(frozen=True)
class _PortBoundary:
    """2D boundary view of the port plane read by cross_section_modes / operators (faces: BCType or None = natural
    end, no wall)."""
    x_lo: object
    x_hi: object
    y_lo: object
    y_hi: object
    cpml_sides: dict
    cpml_npml: int
    cpml_order: int
    cpml_sigma_factor: float
    cpml_kappa_max: float
    cpml_alpha_max: float
    periodic_x: bool
    periodic_y: bool
    bloch_x: bool = False
    bloch_y: bool = False
    bloch_kx: float = 0.0
    bloch_ky: float = 0.0

    @property
    def has_bloch_phase(self) -> bool:
        """Whether a bloch axis carries k != 0 (as BoundaryConfig.has_bloch_phase)."""
        return (self.bloch_x and self.bloch_kx != 0.0) or (self.bloch_y and self.bloch_ky != 0.0)

    def bloch_phase_angles(self, period_x, period_y):
        """Wrap phases (kx Lx, ky Ly) [rad]; 0 on an axis that is not bloch (as BoundaryConfig.bloch_phase_angles)."""
        return (float(self.bloch_kx) * float(period_x) if self.bloch_x else 0.0,
                float(self.bloch_ky) * float(period_y) if self.bloch_y else 0.0)


def port_boundary(config, normal_axis):
    """_PortBoundary of the BoundaryConfig3D faces of the two transverse axes (pec / pmc -> natural ends; mirrors as
    the cross-section mirrors: the same plane, the ghost cell's inner face, and E parity, so the mode is the whole
    mirrored mode and its power the unfolded one; a periodic axis carries its Bloch k, so the mode is the Bloch mode
    the engine's wrap holds)."""
    faces, sides = {}, {}
    for local, axis in zip(('x', 'y'), port_axes(normal_axis)):
        for side in ('lo', 'hi'):
            face_type = getattr(config, f'{AXES[axis]}_{side}')
            faces[f'{local}_{side}'] = {'cpml': BCType.CPML, 'periodic': BCType.PERIODIC,
                                        'symmetric': BCType.SYMMETRIC,
                                        'antisymmetric': BCType.ANTISYMMETRIC}.get(face_type)
            sides[f'{local}_{side}'] = face_type == 'cpml'
    u, v = port_axes(normal_axis)
    periodic_u, periodic_v = config.periodic(AXES[u]), config.periodic(AXES[v])
    return _PortBoundary(**faces, cpml_sides=sides, cpml_npml=int(config.cpml_npml), cpml_order=int(config.cpml_order),
                         cpml_sigma_factor=float(config.cpml_sigma_factor),
                         cpml_kappa_max=float(config.cpml_kappa_max), cpml_alpha_max=float(config.cpml_alpha_max),
                         periodic_x=periodic_u, periodic_y=periodic_v, bloch_x=periodic_u, bloch_y=periodic_v,
                         bloch_kx=float(config.bloch_k[u]), bloch_ky=float(config.bloch_k[v]))


def axis_permittivity(materials, axis):
    """MaterialGrid permittivity array of the E component along `axis` (eps_xx / eps_yy when allocated, else eps_r)."""
    name = AXIS_PERMITTIVITY[axis]
    return getattr(materials, name if name in materials.property_names() else 'eps_r')


def port_materials(physics, normal_axis, layer, omega):
    """CrossSectionMaterials of cell layer `layer` along normal_axis at omega [rad/s] from the engine's (re-quantised)
    material tables (host; a few plane transfers; PEC cells keep the MaterialGrid values): local eps_xx / eps_yy /
    eps_zz = the E_u / E_v / E_d permittivity, sigma_e and pole chi added, mu with sigma_m."""
    materials, engine = physics.domain.materials, physics.engine
    xp = engine.xp
    omega = float(omega)
    u, v = port_axes(normal_axis)
    allocated = materials.property_names()
    ids = _plane(engine.material_id, normal_axis, layer)
    pec_cells = engine.table_eps_z[ids] < 0

    def layer_values(table, grid_array):
        """Host float64 table values of the layer (MaterialGrid array values on the PEC cells)."""
        return to_host(xp.where(pec_cells, _plane(grid_array, normal_axis, layer), table[ids])).astype(np.float64)

    loss = layer_values(engine.table_sigma, materials.sigma_e) / (omega * EPS0)
    tables = (engine.table_eps_x, engine.table_eps_y, engine.table_eps_z)
    permittivities = []
    for axis in (u, v, normal_axis):
        permittivity = layer_values(tables[axis], axis_permittivity(materials, axis)).astype(np.complex128) - 1j * loss
        for pole, window, mask in physics.dispersive_regions:
            start = window[normal_axis].start or 0
            if not start <= layer < window[normal_axis].stop:
                continue
            region = permittivity[window[u], window[v]]   # a view: += writes into permittivity
            region += complex(pole.susceptibility(omega)) * to_host(_plane(mask, normal_axis, layer - start))
        permittivities.append(permittivity)
    permeability = layer_values(engine.table_mu, materials.mu_r).astype(np.complex128)
    if 'sigma_m' in allocated:
        permeability = permeability - 1j * layer_values(engine.table_sigma_m, materials.sigma_m) / (omega * MU0)
    return CrossSectionMaterials(eps_xx=permittivities[0], eps_yy=permittivities[1], eps_zz=permittivities[2],
                                 mu=permeability, pec=to_host(pec_cells),
                                 eps_point_deltas=_smoothing_deltas(engine, normal_axis, layer, ids, pec_cells))


def _smoothing_deltas(engine, normal_axis, layer, ids, pec_cells):
    """None without subpixel face overrides, else the host float64 (n_u, n_v) eps_point_deltas of a port layer: the
    engine's averaged eps of the E_u / E_v faces minus the width-weighted mean of their two layer cells, of E_d minus
    the cell value; 0 where no override applies (none, or a face touching PEC)."""
    smoothing = engine.subpixel_smoothing
    if smoothing is None or smoothing.face_id is None:
        return None
    u, v = port_axes(normal_axis)
    xp = engine.xp
    face_ids = _plane(smoothing.face_id, normal_axis, layer)
    tables = (engine.table_eps_x, engine.table_eps_y, engine.table_eps_z)
    deltas = []
    for local, axis in enumerate((u, v, normal_axis)):
        smoothed = smoothing.plane_values(face_ids, axis)
        cells = tables[axis][ids]
        applies = (smoothed > 0) & ~pec_cells
        standard = cells
        if local < 2:   # in-plane face: the hi partner along the local axis (the cell itself at a non-periodic end)
            def partner(values, axis=axis, local=local):
                shifted = xp.roll(values, -1, axis=local)
                if not engine.periodic[axis]:
                    last = (slice(None),) * local + (-1,)
                    shifted[last] = values[last]
                return shifted
            lower, upper = (share.reshape((-1, 1) if local == 0 else (1, -1)) for share in engine.shares[axis])
            standard = lower * cells + upper * partner(cells)
            applies &= ~partner(pec_cells)
        deltas.append(to_host(xp.where(applies, smoothed - standard, 0.0)).astype(np.float64))
    return tuple(deltas)


def _materials_digest(materials) -> bytes:
    """blake2b digest of every array of a CrossSectionMaterials (shapes included): equal layers give equal modes."""
    digest = hashlib.blake2b(digest_size=16)
    arrays = [materials.eps_xx, materials.eps_yy, materials.eps_zz, materials.mu, materials.pec]
    arrays += list(materials.eps_point_deltas or ()) + list(materials.pec_cuts or ())
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(repr((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.digest()


def solve_port_modes(physics, normal_axis, layer, omega, n_modes, start_vector=None):
    """CrossSectionSolve of the layer at the FDTD numerical frequency of omega (materials at omega); within one
    calibration round (physics.port_mode_solves) ports whose layers hold equal materials share one solve per omega."""
    engine = physics.engine
    u, v = port_axes(normal_axis)
    widths = (engine.widths[u], engine.widths[v])
    pml_dx = max(float(max(axis_widths[0], axis_widths[-1])) for axis_widths in widths)
    numerical_omega = 2.0 / engine.dt * math.sin(0.5 * float(omega) * engine.dt)
    materials = port_materials(physics, normal_axis, layer, omega)
    solves = getattr(physics, 'port_mode_solves', None)
    # same normal axis = same transverse widths and boundary; omega to 12 digits (a source f0 and a monitor frequency
    # from wavelengths may differ in the last bits); the start vector only seeds Arnoldi
    key = (int(normal_axis), f'{float(omega):.12e}', int(n_modes), _materials_digest(materials))
    if solves is not None and key in solves:
        return solves[key]
    # no 'eigen' role (em3d sub file): a private host SuperLU runtime, its Arnoldi on the device on the gpu backend
    eigen_solver = ArpackShiftInvertSolver(device_arnoldi=engine.xp is not np)
    solve = solve_cross_section_modes(widths, port_boundary(engine.config, normal_axis), pml_dx, materials,
                                      numerical_omega, n_modes, start_vector=start_vector, eigen_solver=eigen_solver)
    if solves is not None:
        solves[key] = solve
    return solve


def discrete_beta(mode, spacing):
    """FDTD propagation constant beta_d [1/m] of a mode solved at the numerical frequency: (2/D) arcsin(K D / 2)."""
    return 2.0 / spacing * cmath.asin(0.5 * complex(mode.beta) * spacing)


def _check_mode_number(mode_number):
    if isinstance(mode_number, bool) or not isinstance(mode_number, (int, np.integer)) or mode_number < 1:
        raise ValueError(f'mode_number must be an int >= 1 (1 = largest Re(beta^2)), got {mode_number!r}')
    return int(mode_number)


def _mode_warnings(label, mode, mode_number):
    """Warnings for a selected mode that sits largely in the transverse PML or carries no forward power."""
    warnings = []
    if mode.pml_fraction > _PML_WARN_FRACTION:
        warnings.append(f'{label}: mode {mode_number} has {mode.pml_fraction:.1%} of its energy in the transverse PML '
                        f'(a cladding / radiation mode, or a plane too small for the guide)')
    if mode.power <= 0.0:
        warnings.append(f'{label}: mode {mode_number} carries no forward power (n_eff = {complex(mode.n_eff):.4g}: '
                        f'evanescent or backward)')
    return warnings


class ModeSource3D:
    """One-sided TFSF plane launching `power` [W] of mode `mode_number` along +-normal from cell layer `cell` (total
    field from it downstream); the interface face (scattered side) is the lo face of the cell ('positive') or its hi face."""

    def __init__(self, physics, normal_axis, cell, direction, mode_number, temporal, power, label):
        if direction not in MODE_DIRECTIONS:
            raise ValueError(f'{label} direction must be one of {MODE_DIRECTIONS}, got {direction!r}')
        if not hasattr(temporal, 'quadrature'):
            raise TypeError(f'{label} needs a gaussian or cw waveform (the complex profile needs the quadrature '
                            f'carrier), got {type(temporal).__name__}')
        if not float(power) > 0.0:
            raise ValueError(f'{label} power must be > 0 W, got {power!r}')
        self.physics = physics
        self.label = label
        self.normal_axis = int(normal_axis)
        self.cell = int(cell)
        self.direction = direction
        self.mode_number = _check_mode_number(mode_number)
        self.temporal = temporal
        self.power = float(power)
        self._sign = 1.0 if direction == 'positive' else -1.0
        self.h_layer = self.cell - 1 if direction == 'positive' else self.cell   # the interface face layer
        widths = physics.engine.widths[self.normal_axis]
        self.spacing = 0.5 * float(widths[self.h_layer] + widths[self.h_layer + 1])   # dual spacing at the interface
        self._omega = 2.0 * math.pi * float(temporal.f0)
        self.modes = None
        self.mode = None
        self.beta = None
        self.warnings = []
        self._incident = None       # [(component, layer, coefficient key, re, im)] device float32 planes
        self._coefficients = {}     # key -> live db / cb plane (u, v)

    @property
    def solved(self) -> bool:
        return self.mode is not None

    def calibrate(self, engine):
        """Solve the mode until the first step (later material changes keep it), then refresh the live db / cb
        planes of the four corrections."""
        if self.mode is None or engine.n_step == 0:
            self._solve(engine)
        d = self.normal_axis
        u, v = port_axes(d)
        for key, axis in (('cb_u', u), ('cb_v', v)):
            shape = list(engine.shape)
            shape[d] = 1
            origin = [0, 0, 0]
            origin[d] = self.cell
            box = engine.xp.empty(shape, dtype=DTYPE)
            engine.face_cb_into(box, axis, origin, shape)
            self._coefficients[key] = _plane(box, d, 0)
        ids = _plane(engine.material_id, d, self.h_layer)   # H edges take their lower cell (kernels3d)
        mu = engine.table_mu[ids] * DTYPE(MU0)
        loss = engine.table_sigma_m[ids] * DTYPE(0.5 * engine.dt) / mu
        self._coefficients['db'] = ((DTYPE(engine.dt) / mu) / (DTYPE(1.0) + loss)).astype(DTYPE)

    def _solve(self, engine):
        d = self.normal_axis
        u, v = port_axes(d)
        solve = solve_port_modes(self.physics, d, self.cell, self._omega, max(self.mode_number + 2, 4))
        if len(solve.modes) < self.mode_number:
            raise ValueError(f'{self.label}: the port plane has {len(solve.modes)} modes, mode_number '
                             f'{self.mode_number} does not exist (enlarge the plane or pick a lower mode)')
        mode = solve.modes[self.mode_number - 1]
        beta = discrete_beta(mode, self.spacing)
        flux_factor = math.cos(0.5 * self._omega * engine.dt) * math.cos(0.5 * beta.real * self.spacing)
        amplitude = math.sqrt(self.power / flux_factor)
        h_scale = self._sign * amplitude * cmath.exp(0.5j * beta * self.spacing)   # H one half spacing upstream
        e_u, e_v = amplitude * mode.ex, amplitude * mode.ey
        h_u, h_v = h_scale * mode.hx, h_scale * mode.hy
        widths = engine.widths[d]
        inverse_h, inverse_e = 1.0 / self.spacing, 1.0 / float(widths[self.cell])
        sign = self._sign
        corrections = (('h' + AXES[u], self.h_layer, 'db', -sign * inverse_h * e_v),
                       ('h' + AXES[v], self.h_layer, 'db', sign * inverse_h * e_u),
                       ('e' + AXES[u], self.cell, 'cb_u', sign * inverse_e * h_v),
                       ('e' + AXES[v], self.cell, 'cb_v', -sign * inverse_e * h_u))
        self._incident = [(component, layer, key, engine.xp.asarray(np.real(values), dtype=DTYPE),
                           engine.xp.asarray(np.imag(values), dtype=DTYPE))
                          for component, layer, key, values in corrections]
        self.warnings = (list(solve.info.get('warnings', [])) + _mode_warnings(self.label, mode, self.mode_number)
                         + _quadrature_warnings(self.temporal, self.label))
        self.modes, self.mode, self.beta = solve.modes, mode, beta

    def inject_magnetic(self, fields, t):
        """H corrections from the incident E at t (E^n time); nothing before the solve."""
        self._correct(fields, t, 'h')

    def inject(self, fields, t):
        """E corrections from the incident H at t (the E update centre)."""
        self._correct(fields, t, 'e')

    def _correct(self, fields, t, kind):
        if self._incident is None:
            return
        signal, quadrature = DTYPE(self.temporal(t)), DTYPE(self.temporal.quadrature(t))
        for component, layer, key, incident_re, incident_im in self._incident:
            if component[0] == kind:
                _port_correction(self._coefficients[key], incident_re, incident_im, signal, quadrature,
                                 _plane(getattr(fields, component), self.normal_axis, layer))

    def mode_table(self):
        """Rows per solved mode (MODE_TABLE_COLUMNS); ValueError before the solve."""
        if not self.solved:
            raise ValueError(_NOT_SOLVED)
        k0 = self._omega / C0
        rows = []
        for index, mode in enumerate(self.modes):
            beta = discrete_beta(mode, self.spacing)
            rows.append([index + 1, beta.real / k0, beta.imag / k0, -20.0 * beta.imag / math.log(10.0) / 100.0,
                         float(mode.te_fraction), float(mode.effective_area), float(mode.pml_fraction),
                         index + 1 == self.mode_number])
        return rows


class ModeMonitor3D:
    """Forward / backward amplitudes of mode `mode_number` on the cell-edge plane `edge` along normal_axis over the
    physical window (PlaneFlux3D spectra); modes solved per frequency at finalize, followed outward by overlap."""

    def __init__(self, physics, normal_axis, edge, bounds, frequencies, mode_number, label):
        engine = physics.engine
        self.physics = physics
        self.label = label
        self.normal_axis = int(normal_axis)
        self.edge = int(edge)
        self.mode_number = _check_mode_number(mode_number)
        self.plane = PlaneFlux3D(self.normal_axis, self.edge, bounds, frequencies, engine.dt, engine.widths,
                                 engine.shape, engine.periodic, xp=engine.xp)
        self.frequencies = self.plane.frequencies
        self.dt = float(engine.dt)
        widths = engine.widths[self.normal_axis]
        self.spacing = 0.5 * float(widths[self.edge - 1] + widths[self.edge])
        self.modes = None
        self.betas = None
        self.warnings = []

    @property
    def sample_interval(self):
        return self.plane.sample_interval

    @sample_interval.setter
    def sample_interval(self, value):
        self.plane.sample_interval = value

    @property
    def solved(self) -> bool:
        return self.modes is not None

    def update(self, fields, t):
        self.plane.update(fields, t)

    def calibrate(self, engine):
        """Solve the modes at every frequency until the first step (cell layer above the plane)."""
        if self.modes is None or engine.n_step == 0:
            self._solve()

    def _solve(self):
        omegas = 2.0 * np.pi * self.frequencies
        order = np.argsort(self.frequencies)
        position = self.frequencies.size // 2
        middle = order[position]
        n_modes = max(self.mode_number + 2, 4)
        selected = [None] * self.frequencies.size
        solve = solve_port_modes(self.physics, self.normal_axis, self.edge, omegas[middle], n_modes)
        if len(solve.modes) < self.mode_number:
            raise ValueError(f'{self.label}: the plane has {len(solve.modes)} modes, mode_number {self.mode_number} '
                             f'does not exist (enlarge the plane or pick a lower mode)')
        selected[middle] = solve.modes[self.mode_number - 1]
        warnings = list(solve.info.get('warnings', [])) + _mode_warnings(self.label, selected[middle],
                                                                          self.mode_number)
        doubtful = []
        for neighbours in (order[position + 1:], order[:position][::-1]):
            reference = selected[middle]
            for index in neighbours:
                candidates = solve_port_modes(self.physics, self.normal_axis, self.edge, omegas[index], n_modes,
                                              start_vector=reference).modes
                overlaps = [mode_overlap(reference, candidate) for candidate in candidates]
                if not overlaps:
                    raise ValueError(f'{self.label}: no mode found at {C0 / self.frequencies[index]:.4g} m')
                best = int(np.argmax(overlaps))
                if overlaps[best] < _TRACKING_WARN_OVERLAP:
                    doubtful.append(float(C0 / self.frequencies[index]))
                reference = selected[index] = candidates[best]
        if doubtful:
            warnings.append(f'{self.label}: following mode {self.mode_number} away from the centre wavelength is '
                            f'doubtful at {len(doubtful)} wavelength(s) ({min(doubtful):.4g}-{max(doubtful):.4g} m: '
                            f'overlap with the neighbouring wavelength below {_TRACKING_WARN_OVERLAP}); narrow the band')
        self.modes = selected
        self.betas = np.array([discrete_beta(mode, self.spacing) for mode in selected], dtype=np.complex128)
        self.warnings = warnings

    def n_eff(self):
        """Complex128 (n_f,) beta_d / k0 of the followed mode; ValueError before the solve."""
        if not self.solved:
            raise ValueError(_NOT_SOLVED)
        return self.betas / (2.0 * np.pi * self.frequencies / C0)

    def power_factors(self):
        """(n_f,) flux-equivalent power (a.u., no 1/2) of a unit amplitude: 2 cos(omega dt/2) Re cos(beta_d D/2)."""
        return (2.0 * np.cos(np.pi * self.frequencies * self.dt)
                * np.real(np.cos(0.5 * self.betas * self.spacing)))

    def amplitudes(self):
        """(a_positive, a_negative) complex128 (n_f,) in units of the solver-normalised mode (one transfer per
        spectrum): a = (X / c +- Y) / (2 N), X = int E x H_m, Y = int E_m x H, N = int E_m x H_m (unconjugated)."""
        if not self.solved:
            raise ValueError(_NOT_SOLVED)
        pairs = self.plane.host_pairs()
        positive = np.empty(self.frequencies.size, dtype=np.complex128)
        negative = np.empty(self.frequencies.size, dtype=np.complex128)
        for index, (mode, beta) in enumerate(zip(self.modes, self.betas)):
            cross_field, cross_mode, norm = 0j, 0j, 0j
            for pair, e_spectrum, h_spectrum in pairs:
                mode_e, mode_h = self._mode_samples(mode, pair)
                weights = pair['sign'] * pair['area']
                cross_field += np.sum(e_spectrum[index] * mode_h * weights)
                cross_mode += np.sum(mode_e * h_spectrum[index] * weights)
                norm += np.sum(mode_e * mode_h * weights)
            centre_to_plane = cmath.cos(0.5 * beta * self.spacing)
            positive[index] = (cross_field / centre_to_plane + cross_mode) / (2.0 * norm)
            negative[index] = (cross_field / centre_to_plane - cross_mode) / (2.0 * norm)
        return positive, negative

    def _mode_samples(self, mode, pair):
        """Mode E / H of a PlaneFlux3D pair restricted to its samples, in the pair's grid axis order."""
        u, v = port_axes(self.normal_axis)
        local = {u: ('ex', 'hx'), v: ('ey', 'hy')}
        rows = {pair['face_axis']: pair['face_slice'], pair['cell_axis']: pair['cell_slice']}
        samples = []
        for name in (local[pair['e_axis']][0], local[pair['h_axis']][1]):
            values = getattr(mode, name)[rows[u], rows[v]]
            samples.append(values if u < v else values.T)
        return samples


# ---- results ----
def mode_source_result_3d(source) -> dict:
    """'table' of the modes solved at the source wavelength (MODE_TABLE_COLUMNS); ValueError before the solve."""
    return {'kind': 'table', 'columns': list(MODE_TABLE_COLUMNS), 'units': list(MODE_TABLE_UNITS),
            'rows': [list(row) for row in source.mode_table()], 'wavelength': float(C0 / source.temporal.f0),
            'power': source.power, 'direction': source.direction, 'mode_number': source.mode_number}


def mode_monitor_result_3d(monitor, physics) -> dict:
    """'xy' vs wavelength: power_* a.u. (flux-plane convention), power_per_cw_* [W] with a shared waveform, fraction_*
    = mode power / source power with exactly one ModeSource3D (NaN where its spectrum is too weak)."""
    frequency = np.array(monitor.frequencies, dtype=np.float64)
    wavelength = C0 / frequency
    factors = monitor.power_factors()
    amplitudes = dict(zip(MODE_DIRECTIONS, monitor.amplitudes()))
    series = {f'power_{direction}': factors * np.abs(amplitude) ** 2 for direction, amplitude in amplitudes.items()}
    series_units = dict.fromkeys(series, 'a.u.')
    shared = _shared_waveform_spectrum(physics, frequency, monitor.sample_interval)
    if shared is not None:
        for direction in MODE_DIRECTIONS:
            series[f'power_per_cw_{direction}'] = _per_cw(series[f'power_{direction}'], *shared, factor=2.0)
            series_units[f'power_per_cw_{direction}'] = 'W'
    sources = [source for source in physics.sources if isinstance(source, ModeSource3D)]
    normalized = len(sources) == 1
    if normalized:
        spectrum, valid = _normalizing_spectrum(sources[0], physics, frequency, monitor.sample_interval)
        for direction in MODE_DIRECTIONS:
            per_cw = _per_cw(series[f'power_{direction}'], spectrum, valid, factor=2.0)
            series[f'fraction_{direction}'] = per_cw / sources[0].power
            series_units[f'fraction_{direction}'] = '1'
    n_eff = monitor.n_eff()
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Mode power',
            'series': series, 'series_units': series_units, 'frequency': frequency, 'wavelength': wavelength.copy(),
            'n_eff': n_eff.real.copy(), 'n_eff_imag': n_eff.imag.copy(), 'mode_number': monitor.mode_number,
            'normalized': normalized}

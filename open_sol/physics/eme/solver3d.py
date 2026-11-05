"""EMEPhysics3D: bidirectional eigenmode expansion of a Domain3D along z (eme_plan stage 2): one slice per z cell
layer, full-vector cross-section modes per section (cross_section_modes, PML modes kept), the shared EMEBase cascade and
port results, and the E phasor reconstructed at the cell centres of the whole grid."""
# The xy plane of every z layer is the cross-section: its x / y faces are the EM transverse boundaries (a 2D
# BoundaryConfig view of the BoundaryConfig3D), the z faces are the semi-infinite ports at the window planes (their bc
# and padding are not read; periodic and mirrors refused). Materials: eps_xx / eps_yy / eps_r (= eps_zz), sigma_e, mu_r,
# sigma_m of the layer, complex at omega with the Drude / Lorentz poles of their 3D boxes; PEC cells of the pec boxes
# (cross_section_modes PEC: E_z = 0 at their centres, E_x / E_y = 0 on faces between two of them). Bends: the
# cross_section_modes bend map about x = reference ('exact' model). A new wavelength starts each section's
# shift-invert from the summed physical modes of its last solve (tracks_modes; the start shell keeps H_x, H_y only).
# Factor: the 'eigen' node's factorize_host per section (host SuperLU or cuDSS LU; the transverse-H operator is not
# complex symmetric, so no LDL^T, and cudss_mixed is not offered: ARPACK at machine precision needs refined solves).
# Adjoint (adjoint.cross_section_gradient): eps_sensitivity = dJ/d eps_xx, eps_yy, eps_r of every window cell from the
# kept cascade; no design regions in 3D yet (the optimization study is 2D only).
from dataclasses import replace

import numpy as np
from ...core.backend import to_host

from ...core.constants import DTYPE, EPS0, MU0
from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
from ..em.adjoint import EpsSensitivity
from ..em.boundaries import BoundaryConfig
from ..em.cross_section_modes import CrossSectionMaterials, mode_point_materials, solve_cross_section_modes
from ..em.materials import DrudePole, LorentzPole, register_em_arrays
from .adjoint import PeriodicAdjointFields, cross_section_gradient, objective_cotangents, section_amplitude_pairs
from .mode_sets import (_PML_MODE_FRACTION, cross_section_fields, cross_section_mode_set, gauss_mean, gauss_positions,
                        stretched_areas)
from .renderer3d import EME_3D_COMPONENTS, EMERenderer3D
from .sections import find_sections
from .solver import INPUT_POWER_FLOOR, EME_PERMITTIVITY_ARRAYS, MODE_CACHE_BYTES, EMEBase, slice_positions

_READ_ARRAYS_3D = ('eps_xx', 'eps_yy', 'eps_r', 'sigma_e', 'mu_r', 'sigma_m')
_REFUSED_PORT_FACES = ('periodic', 'symmetric', 'antisymmetric')
BEND_MODEL = 'exact'      # cross_section_modes bend model of the 3D bends
FIELD_BLOCK_SLICES = 64   # z rows reconstructed per host block (n x 3 x nx ny complex128; lossy: 3 Gauss rows per layer)


def cross_section_boundary(config):
    """2D BoundaryConfig of the x / y faces and the CPML settings of a BoundaryConfig3D (the cross-section of every z
    layer)."""
    boundary = BoundaryConfig(config.x_lo, config.x_hi, config.y_lo, config.y_hi)
    boundary.cpml_npml = int(config.cpml_npml)
    boundary.cpml_order = int(config.cpml_order)
    boundary.cpml_sigma_factor = float(config.cpml_sigma_factor)
    boundary.cpml_kappa_max = float(config.cpml_kappa_max)
    boundary.cpml_alpha_max = float(config.cpml_alpha_max)
    boundary._validate()
    return boundary


def _check_box_mask(window, mask, what, xp):
    """window as 3 slices with integer starts; ValueError unless mask is an xp (run backend) bool array of the box
    shape."""
    window = tuple(slice(cells.start or 0, cells.stop) for cells in window)
    if (not isinstance(mask, xp.ndarray) or tuple(mask.shape) != tuple(cells.stop - cells.start for cells in window)
            or mask.dtype != np.bool_):
        raise ValueError(f'{what} mask {type(mask).__name__} shape {tuple(getattr(mask, "shape", ()))} / dtype '
                         f'{getattr(mask, "dtype", "")} does not match its box {window} ({xp.__name__} bool)')
    return window


class EMEPhysics3D(EMEBase):
    """3D eigenmode expansion at omega along z: n_modes cross-section modes per section near (k0 n_eff_guess)^2, the
    S-matrix cascade and the excitation's E over the grid; ports = the z_lo / z_hi window planes."""

    tracks_modes = True

    def __init__(self, domain, boundary_config=None, wavelength=1.55e-6, n_modes=20, n_eff_guess=None,
                 section_tolerance=0.0, eigen_solver=None):
        if getattr(domain, 'dimensions', 2) != 3:
            raise ValueError('EMEPhysics3D needs a Domain3D (size_z set)')
        if int(n_modes) != n_modes or int(n_modes) < 1:
            raise ValueError(f'EMEPhysics3D n_modes must be an integer >= 1, got {n_modes!r}')
        super().__init__(domain, wavelength, n_modes, section_tolerance, _READ_ARRAYS_3D, 'EMEPhysics3D')
        config = domain.boundary_config if boundary_config is None else boundary_config
        if config.npml_sides != domain.npml_sides:
            raise ValueError(f'EMEPhysics3D: boundary padding {config.npml_sides} does not match the domain padding '
                             f'{domain.npml_sides}')
        for side in ('lo', 'hi'):
            if getattr(config, f'z_{side}') in _REFUSED_PORT_FACES:
                raise ValueError(f'EMEPhysics3D: the z_{side} face is a semi-infinite port; '
                                 f'{getattr(config, f"z_{side}")} is not supported there')
        if config.has_bloch_phase:   # the 2D cross-section view carries no k (as EMEPhysics' check_zero_bloch)
            raise NotImplementedError(f'EMEPhysics3D: Bloch k != 0 is not supported (bloch_k = {config.bloch_k}); use '
                                      'k = 0 (periodic)')
        self.boundary_config = config
        self.cross_section_config = cross_section_boundary(config)
        self.polarization = 'modes'
        self.propagation_axis = 'z'
        self.n_eff_guess = None if n_eff_guess is None else float(n_eff_guess)
        self._owns_eigen_solver = eigen_solver is None
        self.eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
        self.pec_regions = []               # (3D cell box, run-backend bool mask) of the PEC boxes
        grid = domain.grid
        # the kernel (float32 grid) widths of the cross-section operator, as em_modes / the em3d mode ports
        self._cell_widths = tuple(np.asarray(axis_widths, dtype=np.float64) for axis_widths in (grid.dx_arr,
                                                                                                 grid.dy_arr))
        self._dz_cells = np.asarray(domain.cell_widths()[2], dtype=np.float64)   # section lengths, as the 2D EME
        self._positions = slice_positions(self._dz_cells, domain.physical_window()[2])
        self._pml_dx = max(float(max(axis_widths[0], axis_widths[-1])) for axis_widths in self._cell_widths)
        self._periodic = (bool(self.cross_section_config.periodic_x), bool(self.cross_section_config.periodic_y))
        self._area_stretch = (None, None)   # (omega, stretched_areas) of the last solve
        self.host_factor_bytes = None       # largest SuperLU factor of the last solve (Simulation.memory_report)
        # cross-section factorisation / ARPACK host wall time and the largest cuDSS factor (cudss eigen node; 0 else)
        self._counters.update({'operator_applications': 0, 'mode_factor_seconds': 0.0, 'mode_eigen_seconds': 0.0,
                               'max_factor_device_bytes': 0})
        register_em_arrays(domain.materials, 'modes')
        shape = (grid.nx, grid.ny, grid.nz)
        # complex64 full-grid E at the cell centres and float32 q_abs [W/m^3] (domain.xp) of the last solve (zeros
        # before)
        xp = domain.xp
        self.field_arrays = {name: xp.zeros(shape, dtype=np.complex64) for name in EME_3D_COMPONENTS}
        self.q_abs = xp.zeros(shape, dtype=DTYPE)

    def state(self) -> dict:
        """The permittivity arrays, sigma_e, q_abs [W/m^3] and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        state = {name: getattr(materials, name) for name in EME_PERMITTIVITY_ARRAYS}
        state['sigma_e'] = materials.sigma_e
        state['q_abs'] = self.q_abs
        state['q_accum'] = None
        return state

    def renderer(self):
        return EMERenderer3D(self)

    def work_counters(self):
        counters = dict(self._counters)
        if self._owns_eigen_solver:
            counters.update(self.eigen_solver.work_counters())
        return counters

    def add_dispersive(self, pole, window, mask):
        """Record a Drude / Lorentz pole on a run-backend bool mask cropped to the 3D cell box `window` (3 slices)."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        self.dispersive_regions.append((pole, _check_box_mask(window, mask, 'pole', self.domain.xp), mask))
        self._dirty = True

    def add_pec(self, window, mask):
        """Perfect electric conductor cells on a run-backend bool mask cropped to the 3D cell box `window` (3
        slices)."""
        self.pec_regions.append((_check_box_mask(window, mask, 'PEC', self.domain.xp), mask))
        self._dirty = True

    # --- solve ---

    def _boxes(self):
        """(3D box, run-backend bool mask) of every pole region, then every PEC region."""
        return [(window, mask) for _pole, window, mask in self.dispersive_regions] + list(self.pec_regions)

    @staticmethod
    def _box_layer(window, mask, layer):
        """(xy slices, run-backend bool (sx, sy) mask) of a 3D box at z layer `layer`, None when the box misses it."""
        if not window[2].start <= layer < window[2].stop:
            return None
        return window[:2], mask[:, :, layer - window[2].start]

    def _layer_rows(self, layer):
        """Host float64 (n_rows, nx ny) material rows of z layer `layer`: the read arrays, one row per pole and PEC box
        (one device-to-host transfer), then one bend label row per bend."""
        materials = self.domain.materials
        xp = self.domain.xp
        rows = [getattr(materials, name)[:, :, layer].astype(np.float64) for name in self._read_arrays]
        for window, mask in self._boxes():
            region = xp.zeros(rows[0].shape, dtype=np.float64)
            cut = self._box_layer(window, mask, layer)
            if cut is not None:
                region[cut[0]] = cut[1]
            rows.append(region)
        host = to_host(xp.stack(rows)).reshape(len(rows), -1)
        labels = self._bend_labels(self._positions[layer:layer + 1])
        return np.vstack([host] + [np.full((1, host.shape[1]), value) for value in labels[:, 0]])

    def _find_sections(self):
        window = self.domain.physical_window()
        materials = self.domain.materials
        xp = self.domain.xp
        scales = []
        for name in self._read_arrays:   # max |value| by two reductions: no window-sized temporary
            values = getattr(materials, name)[window]
            scales.append(max(float(xp.max(values)), -float(xp.min(values))))
        scales += [1.0] * (len(self._boxes()) + len(self.bends))
        return find_sections(self._layer_rows, self._dz_cells, window[2], scales, self.section_tolerance)

    def _layer_pec(self, layer):
        """Host bool (nx, ny) PEC cells of z layer `layer`."""
        pec = np.zeros((self._cell_widths[0].size, self._cell_widths[1].size), dtype=bool)
        for window, mask in self.pec_regions:
            cut = self._box_layer(window, mask, layer)
            if cut is not None:
                pec[cut[0]] |= to_host(cut[1])
        return pec

    def _layer_poles(self, layer, omega):
        """[(chi(omega), xy slices, host bool mask)] of the pole regions crossing z layer `layer`."""
        poles = []
        for pole, window, mask in self.dispersive_regions:
            cut = self._box_layer(window, mask, layer)
            if cut is not None:
                poles.append((complex(pole.susceptibility(float(omega))), cut[0], to_host(cut[1])))
        return poles

    def _host_layer(self, name, layer):
        return to_host(getattr(self.domain.materials, name)[:, :, layer]).astype(np.float64)

    def _layer_materials(self, layer, omega):
        """CrossSectionMaterials of z layer `layer` at omega: eps_xx / eps_yy / eps_zz - j sigma_e / (omega eps0) plus
        the pole chi(omega) on their masks, mu_r - j sigma_m / (omega mu0), the PEC cells."""
        omega = float(omega)
        loss = self._host_layer('sigma_e', layer) / (omega * EPS0)
        poles = self._layer_poles(layer, omega)
        permittivities = {}
        for name in ('eps_xx', 'eps_yy', 'eps_r'):
            permittivity = self._host_layer(name, layer).astype(np.complex128) - 1j * loss
            for chi, cells, mask in poles:
                permittivity[cells] += chi * mask
            permittivities[name] = permittivity
        mu = self._host_layer('mu_r', layer).astype(np.complex128) - 1j * self._host_layer('sigma_m', layer) / (
            omega * MU0)
        return CrossSectionMaterials(eps_xx=permittivities['eps_xx'], eps_yy=permittivities['eps_yy'],
                                     eps_zz=permittivities['eps_r'], mu=mu, pec=self._layer_pec(layer))

    def _layer_loss(self, layer, omega):
        """Host float64 (nx, ny) -Im eps_c(omega) of z layer `layer` (sigma_e and poles; 0 on PEC cells)."""
        loss = self._host_layer('sigma_e', layer) / (float(omega) * EPS0)
        for chi, cells, mask in self._layer_poles(layer, omega):
            loss[cells] -= chi.imag * mask
        loss[self._layer_pec(layer)] = 0.0
        return loss

    def _stretched_areas(self, omega):
        if self._area_stretch[0] != omega:
            self._area_stretch = (omega, stretched_areas(self._cell_widths, self.cross_section_config, self._pml_dx,
                                                         omega))
        return self._area_stretch[1]

    def _entry_bytes(self, entry):
        start = entry[0]
        return entry[-1].nbytes + (0 if start is None else start.hx.nbytes + start.hy.nbytes)

    def _bend_map(self, section):
        """cross_section_modes bend tuple of a bent section, None when straight."""
        bend = self._bend_index(self._positions[section.first_slice])
        if bend is None:
            return None
        _, _, radius, reference = self.bends[bend]
        return radius, reference, self.domain.cell_edges()[0], BEND_MODEL

    def _solve_section(self, section, omega, previous_entry):
        """(start shell or None, ModeSet) of the cross-section of the section's first z layer; previous_entry's shell
        (the summed physical H_x, H_y of the last wavelength) starts the shift-invert."""
        solve = solve_cross_section_modes(self._cell_widths, self.cross_section_config, self._pml_dx,
                                          self._layer_materials(section.first_slice, omega), omega, self.n_modes,
                                          n_eff_guess=self.n_eff_guess, bend=self._bend_map(section),
                                          start_vector=None if previous_entry is None else previous_entry[0],
                                          eigen_solver=self.eigen_solver, drop_pml_modes=False)
        info = solve.info
        self._counters['operator_applications'] += int(info['operator_applications'])
        self._counters['mode_factor_seconds'] += float(info['factor_seconds'])
        self._counters['mode_eigen_seconds'] += float(info['eigen_seconds'])
        self._counters['max_factor_device_bytes'] = max(self._counters['max_factor_device_bytes'],
                                                        int(getattr(self.eigen_solver, 'device_bytes', 0) or 0))
        self.host_factor_bytes = max(self.host_factor_bytes or 0, int(info['factor_host_bytes']))
        for warning in info['warnings']:
            text = f'EME section at z cell {section.first_slice}: {warning}'
            if text not in self.runtime_warnings:
                self.runtime_warnings.append(text)
        if not solve.modes:
            raise ValueError(f'EME section at z cell {section.first_slice}: the cross-section solve found no mode '
                             f'(set n_eff_guess near the wanted modes)')
        physical = [mode for mode in solve.modes if mode.pml_fraction <= _PML_MODE_FRACTION]
        shell = None
        if physical:   # start vector of the next wavelength: the sum of the physical modes' H_x, H_y
            shell = replace(physical[0], hx=sum(mode.hx for mode in physical), hy=sum(mode.hy for mode in physical),
                            ex=None, ey=None, ez=None, hz=None)
        return shell, cross_section_mode_set(solve.modes, self._stretched_areas(omega), self._periodic)

    def _reconstruct(self, cascade, omega, previous_cache, cache, counts):
        """E of the excitation at the cell centres of every window z layer into the complex64 field arrays; on lossy
        sections q_abs from the 3-point Gauss z mean of |E|^2 in each cell (Gauss centre = the stored E)."""
        face = self.excitation[0]
        incident, _ = self.input_amplitude()
        nx, ny = self._cell_widths[0].size, self._cell_widths[1].size
        xp = self.domain.xp
        for array in (*self.field_arrays.values(), self.q_abs):
            array.fill(0)
        for index, section in enumerate(self.sections):
            modes = self._section_modes(section, omega, previous_cache, cache, counts, for_field=True)[-1]
            forward, backward = cascade.section_amplitudes(index, face, incident)
            widths = self._dz_cells[section.first_slice:section.first_slice + section.n_slices]
            offsets = np.concatenate(([0.0], np.cumsum(widths)))
            loss = self._layer_loss(section.first_slice, omega).ravel()
            lossy = bool(np.any(loss != 0.0))
            block_size = FIELD_BLOCK_SLICES // 3 if lossy else FIELD_BLOCK_SLICES
            for block in range(0, section.n_slices, block_size):
                stop = min(block + block_size, section.n_slices)
                n_block = stop - block
                positions, _ = gauss_positions(widths[block:stop])
                if not lossy:
                    positions = positions[n_block:2 * n_block]
                fields = cross_section_fields(modes, forward, backward, offsets[block] + positions, section.length)
                centre = fields[n_block:2 * n_block] if lossy else fields
                layers = slice(section.first_slice + block, section.first_slice + stop)
                for component, name in enumerate(EME_3D_COMPONENTS):
                    values = centre[:, component, :].reshape(n_block, nx, ny).transpose(1, 2, 0)
                    self.field_arrays[name][:, :, layers] = xp.asarray(np.ascontiguousarray(values, dtype=np.complex64))
                if lossy:
                    mean = gauss_mean(np.sum(np.abs(fields) ** 2, axis=1))
                    power = (0.5 * float(omega) * EPS0) * loss[None, :] * mean
                    self.q_abs[:, :, layers] = xp.asarray(np.ascontiguousarray(
                        power.reshape(n_block, nx, ny).transpose(1, 2, 0), dtype=np.float32))

    # --- adjoint sensitivity (adjoint.cross_section_gradient) ---

    def _cascade_sensitivity(self, objectives):
        """EpsSensitivity (float64 dJ/d eps_xx, eps_yy, eps_r per cell; density = their sum per cell volume [1/m^3])
        of the last solve by the cascade adjoint on every window z layer (a periodic cell sums its periods)."""
        omega = self.solved_omega
        face, number, _ = self.excitation
        _, unit_power = self.input_amplitude()
        if unit_power is None or unit_power <= INPUT_POWER_FLOOR:
            raise ValueError(f'em_eme_3d adjoint sensitivity: input mode {number} of the {face} port is missing or '
                             'carries no power')
        total, cotangents = objective_cotangents(objectives, face, self.port_blocks(), number - 1, unit_power)
        host = {name: np.zeros(self.q_abs.shape) for name in EME_PERMITTIVITY_ARRAYS}
        if cotangents:
            self._add_section_gradients(host, omega, face, number, cotangents)
        density = host['eps_xx'] + host['eps_yy']
        density += host['eps_r']
        for axis, widths in enumerate(self.domain.cell_widths()):
            shape = [1, 1, 1]
            shape[axis] = -1
            density /= widths.reshape(shape)
        xp = self.domain.xp
        gradients = {name: xp.asarray(values) for name, values in host.items()}
        return EpsSensitivity(total, float(omega), gradients, xp.asarray(density, dtype=DTYPE))

    def _add_section_gradients(self, host, omega, face, number, cotangents):
        """Add dJ/d eps of every section's z layers into the host (nx, ny, nz) arrays (FIELD_BLOCK_SLICES layers per
        block; uncached sections re-solved, counted as mode_resolves)."""
        incident = np.zeros(self.port_modes[face].n_modes, dtype=np.complex128)
        incident[number - 1] = 1.0
        periodic = None
        if self.periodic_adjoint is not None:
            block, n_periods, cell, exit_interface = self.periodic_adjoint
            periodic = PeriodicAdjointFields(self.cascade, block, cell, exit_interface, n_periods, face, incident,
                                             cotangents)
        area_stretch = self._stretched_areas(omega)
        counts = {'cache_hits': 0, 'mode_resolves': 0, 'tracked_solves': 0, 'cache_bytes': MODE_CACHE_BYTES}
        for index, section in enumerate(self.sections):
            modes = self._section_modes(section, omega, self._mode_cache, {}, counts, for_field=True)[-1]
            points = mode_point_materials(self._cell_widths, self.cross_section_config, self._pml_dx,
                                          self._layer_materials(section.first_slice, omega), self._bend_map(section))
            widths = self._dz_cells[section.first_slice:section.first_slice + section.n_slices]
            offsets = np.concatenate(([0.0], np.cumsum(widths)))
            pairs = section_amplitude_pairs(self.cascade, index, modes.n_modes, face, incident, cotangents, periodic)
            for first in range(0, section.n_slices, FIELD_BLOCK_SLICES):
                stop = min(first + FIELD_BLOCK_SLICES, section.n_slices)
                rows = cross_section_gradient(self.cascade, index, modes, points, area_stretch, widths[first:stop],
                                              section.length, omega, face, incident, cotangents, periodic,
                                              offsets[first], pairs=pairs)
                layers = slice(section.first_slice + first, section.first_slice + stop)
                for name, values in rows.items():
                    host[name][:, :, layers] += values.transpose(1, 2, 0)
        self._counters['mode_resolves'] += counts['mode_resolves']

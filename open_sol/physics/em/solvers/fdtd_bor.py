"""Body-of-revolution FDTD on a Domain (x = r from the x_lo window edge = the axis, y = z; one azimuthal order m, or the
driven order sum of point currents at phi = 0): BORFDTDPhysics steps fdtd_bor.BORTimeStepper / BOROrderSum on the
domain's array module, with Drude / Lorentz poles, point currents, mode ports (fdtd_bor_ports), field probes, flux and
far-field spectra and the frequency_domain harmonic accumulator (BORHarmonic)."""
import math

import numpy as np

from ....core.backend import to_host
from ....core.constants import C0, DTYPE, EPS0, MU0
from ....core.monitors import FieldProbe
from ....core.solver_base import PhysicsSolver
from .. import _make_temporal, _resolve_frequencies, _resolve_frequency
from ..body_of_revolution import (BOR_FIELD_COMPONENTS, BOR_SOURCE_COMPONENTS, BORMaterials, axis_class,
                                  distinct_orders, far_field_power, far_field_samples, line_flux, order_sum_far_field,
                                  yee_centres)
from ..fdtd_bor import BOROrderSum, BORTimeStepper, leapfrog_susceptibility, point_index, source_spectrum
from ..fdtd_bor_ports import (BORFDTDModeMonitor, BORFDTDModeSource, fdtd_bor_mode_monitor_result,
                              stepped_frequency)
from ..harmonic_fdtd import _shared_pulse
from ..materials import DrudePole, LorentzPole, register_em_arrays
from ..placement import port_cells
from ..renderer_fdtd_bor import BORFDTDRenderer
from ..sources import _carrier_angular_frequency
from ..yee_grid import cpml_grading_cell_size
from ....solvers.sparse_direct import scipy_available
from .fdfd_bor import (BOR_MATERIAL_SCHEMA_KEY, BOR_PERMITTIVITY_SLOTS, BOR_POLARIZATION,
                       FAR_FIELD_QUANTITY, bor_boundary_config, bor_far_field_angles, bor_far_field_box,
                       bor_flux_line)
from .fdtd_2d_base import _check_cell_array, record_dispersive_region

# Yee point of each probe component relative to its cell (r, z): E_r / H_z on the +r edge, E_z / H_r on the +z edge,
# H_phi on the +r +z corner, E_phi at the centre
PROBE_OFFSETS = {'er': (1, 0), 'ephi': (0, 0), 'ez': (0, 1), 'hr': (0, 1), 'hz': (1, 0), 'hphi': (1, 1)}
H_COMPONENTS = ('hr', 'hz', 'hphi')   # stepped half a step before the recorded time
_COEFFICIENT_MATERIALS = frozenset({'eps_r', 'eps_xx', 'eps_yy', 'sigma_e', 'mu_r', 'sigma_m'})
_VALID_SPECTRUM_FRACTION = 1e-6   # per-cw values are NaN where |S|^2 < this x |S(carrier)|^2 (as the 2D FDTD)


class BORFDTDPhysics(PhysicsSolver):
    """Time-domain EM of a body of revolution at azimuthal order m, or the driven order sum over azimuthal_orders
    (fdtd_bor module header): the engine is built at finalize from the MaterialGrid (PEC, poles and, for an order sum,
    its members fixed from then on); later material changes refresh its coefficients."""
    # BORPhysics-like attribute set (domain, polarization, boundary_config, sources, monitors, probes,
    # dispersive_regions, pec_cells, finalize_warnings, runtime_warnings, enable_q_accum, axisymmetric = True) plus the
    # FDTD ones the resonance_monitor / harmonic study read (engine with dt / n_step / time, harmonic). The engine is a
    # BORTimeStepper (one order) or a BOROrderSum (azimuthal_orders non-empty: one stepper per order and source family,
    # shown fields / probes / harmonic phasors = the phi = 0 sum, flux / far field = the 3D sums, per order too).
    # sources hold BORCurrentSource and fdtd_bor_ports.BORFDTDModeSource (single order only), monitors also
    # BORFDTDModeMonitor; every source / monitor attaches to the engine at finalize (or at once when it exists).

    name = 'em'

    def __init__(self, domain, boundary_config=None, azimuthal_order=0, courant_factor=0.99, azimuthal_orders=None):
        if getattr(domain, 'dimensions', 2) != 2:
            raise ValueError('BORFDTDPhysics needs a 2D domain: the drawing is the (r, z) half plane')
        axis_class(azimuthal_order)   # integer check
        courant_factor = float(courant_factor)
        if not 0.0 < courant_factor <= 1.0:
            raise ValueError(f'BORFDTDPhysics courant_factor must lie in (0, 1], got {courant_factor!r}')
        orders = distinct_orders(azimuthal_orders, 'BORFDTDPhysics azimuthal_orders', non_negative=True)
        self.domain = domain
        self.polarization = BOR_POLARIZATION
        self.boundary_config = bor_boundary_config(domain, boundary_config, 'BORFDTDPhysics')
        self.azimuthal_order = int(azimuthal_order)
        self.azimuthal_orders = orders   # () = the single order azimuthal_order
        self.courant_factor = courant_factor
        self.xp = domain.xp
        self.pml_dx = cpml_grading_cell_size(domain)
        register_em_arrays(domain.materials, BOR_MATERIAL_SCHEMA_KEY)
        self.sources = []             # BORCurrentSource
        self.monitors = []            # BORFluxSpectrum, BORFarFieldSpectrum
        self.probes = []              # core.monitors.FieldProbe on the engine's Yee views, recorded every step
        self.dispersive_regions = []
        self.pec_cells = None
        self.finalize_warnings = []
        self.runtime_warnings = []    # append-only; the studies log new entries
        self.enable_q_accum = False   # loss-protocol flag (no absorbed_power output: no em_heat_joule)
        self.axisymmetric = True
        self.engine = None            # fdtd_bor.BORTimeStepper or BOROrderSum, built by finalize
        self.harmonic = None          # BORHarmonic of a frequency_domain study
        self._coefficients_dirty = False

    @property
    def permittivity_arrays(self) -> tuple:
        """MaterialGrid permittivity arrays: eps_r (E_phi), eps_xx (E_r), eps_yy (E_z)."""
        return tuple(BOR_PERMITTIVITY_SLOTS.values())

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def finalize(self) -> None:
        """Build the engine (operators, time step, coefficients; an order sum: its members from the source components)
        on the current materials, or refresh its coefficients; ValueError without scipy."""
        self.finalize_warnings = []
        if not scipy_available():
            raise ValueError('em_fdtd_bor needs scipy (host operator assembly and time-step bound), which is not '
                             'installed: pip install scipy')
        if self.engine is not None:
            self._refresh()
            return
        materials, sigma_e, sigma_m = self._host_materials()
        arguments = (self.domain.cell_widths(), self.boundary_config, self.pml_dx, materials)
        options = dict(courant=self.courant_factor, sigma_e=sigma_e, sigma_m=sigma_m,
                       window=self.domain.physical_window(), xp=self.xp, poles=self._host_poles())
        if self.azimuthal_orders:
            components = {source.component for source in self.sources if isinstance(source, BORCurrentSource)}
            self.engine = BOROrderSum(*arguments, self.azimuthal_orders, components, **options)
        else:
            self.engine = BORTimeStepper(*arguments, self.azimuthal_order, **options)
        for source in self.sources:
            source.attach(self.engine)
        for monitor in self.monitors:
            monitor.attach(self.engine)
        self._coefficients_dirty = False

    def step(self, t: float = None, dt: float = None):
        """One leap-frog step of dt_limit(); the probes record E at the new time, H half a step earlier (an order sum:
        the phi = 0 sum)."""
        engine = self._ready()
        engine.step()
        time = engine.time
        for probe in self.probes:
            probe.record_value(engine.probe_value(probe.field_component, (probe.ix, probe.iy)), time)

    def dt_limit(self) -> float:
        return float(self._ready().dt)

    def state(self) -> dict:
        """Float32 full-grid e_intensity [V^2/m^2] of the instantaneous centre fields, eps_r, eps_xx, eps_yy, sigma_e
        and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        centres = self.centre_samples()
        e_intensity = centres['er'] ** 2 + centres['ephi'] ** 2 + centres['ez'] ** 2
        return {'e_intensity': e_intensity, 'eps_r': materials.eps_r, 'eps_xx': materials.eps_xx,
                'eps_yy': materials.eps_yy, 'sigma_e': materials.sigma_e, 'q_accum': None}

    def inject_source(self, name: str, array):
        """Copy a coupling's array into its permittivity array (BOR_PERMITTIVITY_SLOTS); coefficients refresh before
        the next step."""
        if name not in BOR_PERMITTIVITY_SLOTS:
            raise ValueError(f"BORFDTDPhysics accepts sources {', '.join(BOR_PERMITTIVITY_SLOTS)}; got '{name}'")
        target = getattr(self.domain.materials, BOR_PERMITTIVITY_SLOTS[name])
        self.xp.copyto(target, self.xp.asarray(array, dtype=DTYPE))
        self._coefficients_dirty = True

    def materials_changed(self, names) -> None:
        if not _COEFFICIENT_MATERIALS.isdisjoint(names):
            self._coefficients_dirty = True

    def renderer(self):
        return BORFDTDRenderer(self)

    # --- engine ---

    def _ready(self):
        """The stepper (finalize on first use), coefficients refreshed when materials changed."""
        if self.engine is None:
            self.finalize()
        elif self._coefficients_dirty:
            self._refresh()
        return self.engine

    def _refresh(self):
        materials, sigma_e, sigma_m = self._host_materials()
        self.engine.refresh_coefficients(materials, sigma_e, sigma_m)
        self._coefficients_dirty = False

    def _host_materials(self):
        """(BORMaterials of the real eps components / mu_r with the PEC cells, sigma_e, sigma_m): host float64."""
        grid = self.domain.materials

        def host(name):
            return np.asarray(to_host(getattr(grid, name)), dtype=np.float64)

        shape = (self.domain.grid.nx, self.domain.grid.ny)
        pec = np.zeros(shape, dtype=bool) if self.pec_cells is None else np.asarray(to_host(self.pec_cells), bool)
        materials = BORMaterials(eps=host('eps_r'), mu=host('mu_r'), pec=pec, eps_rr=host('eps_xx'),
                                 eps_zz=host('eps_yy'))
        sigma_m = host('sigma_m')
        return materials, host('sigma_e'), (sigma_m if np.any(sigma_m) else None)

    def _host_poles(self):
        """[(pole, host float64 (nx, ny) cell share)] of dispersive_regions (1 in the region's cells)."""
        shape = (self.domain.grid.nx, self.domain.grid.ny)
        poles = []
        for pole, box, cropped_mask in self.dispersive_regions:
            share = np.ones(shape) if box is None else np.zeros(shape)
            if box is not None:
                share[box] = np.asarray(to_host(cropped_mask), dtype=np.float64)
            poles.append((pole, share))
        return poles

    def stepped_materials(self, omega):
        """BORMaterials (host complex, full grid) the stepper realises at omega [rad/s], i.e. the em_fdfd_bor materials
        of its DFT phasors at omega_tilde: eps - j sigma_e cos(omega dt / 2) / (omega_tilde eps0) + the
        leapfrog_susceptibility of each pole on its cells, mu - j sigma_m cos(omega dt / 2) / (omega_tilde mu0)."""
        if self.engine is None:
            self.finalize()
        dt = self.engine.dt
        tilde, centring = stepped_frequency(omega, dt), math.cos(0.5 * float(omega) * dt)
        materials, sigma_e, sigma_m = self._host_materials()
        extra = -1j * sigma_e * centring / (tilde * EPS0)
        for pole, share in self._host_poles():
            extra = extra + leapfrog_susceptibility(pole, omega, dt) * share
        mu = materials.mu.astype(np.complex128)
        if sigma_m is not None:
            mu = mu - 1j * sigma_m * centring / (tilde * MU0)
        return BORMaterials(eps=materials.component('phi') + extra, mu=mu, pec=materials.pec,
                            eps_rr=materials.component('r') + extra, eps_zz=materials.component('z') + extra)

    def centre_samples(self):
        """Full-grid float32 instantaneous fields er, ephi, ez, hr, hphi, hz at the cell centres (Yee points averaged;
        H half a step earlier); zeros before the stepper exists."""
        if self.engine is None:
            shape = (self.domain.grid.nx, self.domain.grid.ny)
            return {name: self.xp.zeros(shape, dtype=DTYPE) for name in BOR_FIELD_COMPONENTS}
        fields, _ = yee_centres(self.engine.yee_views())
        return {name: fields[name].astype(DTYPE, copy=False) for name in BOR_FIELD_COMPONENTS}

    # --- material helpers ---

    def add_dispersive(self, pole, mask):
        """Drude / Lorentz pole on a run-backend bool cell mask (None = everywhere), before finalize: stepped on every
        E point its cells touch (all three components; fdtd_bor module header); the permittivity arrays stay the
        background (eps_inf)."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        if mask is not None:
            _check_cell_array(mask, (self.domain.grid.nx, self.domain.grid.ny), self.xp.bool_, 'Pole mask', xp=self.xp)
        if self.engine is not None:
            raise ValueError('BORFDTDPhysics.add_dispersive: the stepper exists already; add poles before finalize')
        record_dispersive_region(self.dispersive_regions, pole, mask)

    def add_pec_region(self, mask):
        """Perfect electric conductor on a bool (nx, ny) cell mask of the domain's array module (rings in 3D), before
        finalize (it removes E points from the operator)."""
        shape = (self.domain.grid.nx, self.domain.grid.ny)
        if not (isinstance(mask, self.xp.ndarray) and mask.dtype == np.bool_ and mask.shape == shape):
            raise TypeError(f'PEC mask must be a bool array of shape {shape} on the domain backend, got '
                            f"{type(mask).__name__} {getattr(mask, 'dtype', '')} {getattr(mask, 'shape', '')}")
        if self.engine is not None:
            raise ValueError('BORFDTDPhysics.add_pec_region: the stepper exists already; add PEC regions before '
                             'finalize')
        self.pec_cells = mask.copy() if self.pec_cells is None else (self.pec_cells | mask)

    # --- sources and monitors ---

    def add_point_source(self, x, y, component='z', current=1e-3, wavelength=None, frequency=None, bandwidth=0.5,
                         waveform='gaussian'):
        """Current element of `component` 'r' | 'phi' | 'z' and peak current [A] in the cell containing (x = r, y =
        z) with the FDTD waveform (fdtd_bor module header: 'phi' = the em_fdfd_bor phi current at phase +90 deg; in an
        order sum a 3D point current at phi = 0 of moment I d, added before finalize)."""
        if component not in BOR_SOURCE_COMPONENTS:
            raise ValueError(f'point source component must be one of {BOR_SOURCE_COMPONENTS}, got {component!r}')
        self.domain.validate_physical_pos(x, y, 'Point source')
        ix, iy = self.domain.pos_to_cell(x, y)
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        return self._add_source(BORCurrentSource(ix, iy, component, float(current), temporal))

    def add_field_probe(self, x, y, field_component='ez'):
        """Time trace (core.monitors.FieldProbe) of one component on its Yee point of the cell containing (x, y)
        (PROBE_OFFSETS; H half a step before the recorded time)."""
        if field_component not in BOR_FIELD_COMPONENTS:
            raise ValueError(f'field probe component must be one of {BOR_FIELD_COMPONENTS}, got {field_component!r}')
        self.domain.validate_physical_pos(x, y, 'Field probe')
        ix, iy = self.domain.pos_to_cell(x, y)
        offset_r, offset_z = PROBE_OFFSETS[field_component]
        probe = FieldProbe(ix + offset_r, iy + offset_z, field_component)
        self.probes.append(probe)
        return probe

    def add_flux_monitor(self, axis, position, span, wavelengths=None, frequencies=None):
        """Power spectrum through a line (axis 'x' = along r at z = position: flux along +z; 'y' = along z at r =
        position: flux along +r), span = (start, end) along it [m] (body_of_revolution.line_flux on DFT phasors)."""
        return self._add_monitor(BORFluxSpectrum(self, *bor_flux_line(self.domain, axis, position, span),
                                                 _resolve_frequencies(wavelengths, frequencies)))

    def add_near_to_far(self, x0, y0, x1, y1, wavelengths=None, frequencies=None):
        """Far-field spectrum (body_of_revolution.far_field_samples on DFT phasors) of the cell box covering [x0, x1]
        x [y0, y1] (r, z); a box bottom on a pec / pmc z_lo wall is mirrored."""
        return self._add_monitor(BORFarFieldSpectrum(self, *bor_far_field_box(self.domain, self.boundary_config, x0, y0,
                                                                              x1, y1),
                                                     _resolve_frequencies(wavelengths, frequencies)))

    def _refuse_ports_in_order_sum(self, label):
        if self.azimuthal_orders:
            raise ValueError(f'{label}: mode ports need a single azimuthal order (no azimuthal_orders order sum)')

    def add_mode_source(self, position, span, power=1.0, mode_number=1, direction='positive', phase=0.0,
                        wavelength=None, frequency=None, bandwidth=0.5, waveform='gaussian'):
        """Port source (fdtd_bor_ports.BORFDTDModeSource) on the z = position line over r = span (from the axis, or
        annular): `power` [W] of guided mode `mode_number` toward +z ('positive') or -z, phase [deg], the FDTD waveform
        (gaussian or cw: it needs a quadrature); its modes are solved at the carrier when the stepper exists."""
        self._refuse_ports_in_order_sum('Mode source')
        row, span_cells, label = port_cells(self.domain, 'x', position, span, 'Mode source')
        temporal = _make_temporal(waveform, _resolve_frequency(wavelength, frequency), bandwidth)
        return self._add_source(BORFDTDModeSource(self, row, span_cells, temporal, power, mode_number, direction, phase,
                                                  label))

    def _add_source(self, source):
        self.sources.append(source)
        if self.engine is not None:
            source.attach(self.engine)
        return source

    def add_mode_monitor(self, position, span, mode_number=1, wavelengths=None, frequencies=None):
        """Port monitor (fdtd_bor_ports.BORFDTDModeMonitor) on the z = position line over r = span (from the axis, or
        annular) at the monitor wavelengths."""
        self._refuse_ports_in_order_sum('Mode monitor')
        row, span_cells, label = port_cells(self.domain, 'x', position, span, 'Mode monitor')
        return self._add_monitor(BORFDTDModeMonitor(self, row, span_cells,
                                                    _resolve_frequencies(wavelengths, frequencies), mode_number,
                                                    label))

    def _add_monitor(self, monitor):
        self.monitors.append(monitor)
        if self.engine is not None:
            if self.engine.n_step:
                self.runtime_warnings.append(f'{type(monitor).__name__} added at step {self.engine.n_step}: its '
                                             'spectra miss the earlier steps')
            monitor.attach(self.engine)
        return monitor

    # --- frequency_domain study hooks ---

    def attach_harmonic(self, frequencies):
        """BORHarmonic accumulator at the study frequencies [Hz] (before the first step)."""
        self._ready()
        self.harmonic = BORHarmonic(self, frequencies)
        return self.harmonic

    def select_harmonic_frequency(self, index):
        self.harmonic.select(index)


class BORCurrentSource:
    """Current element in cell (ix, iy): component 'r' | 'phi' | 'z', peak current [A], waveform temporal; its
    fdtd_bor injection once the engine exists (an order sum: the list of its members' injections)."""

    def __init__(self, ix, iy, component, current, temporal):
        self.ix, self.iy = int(ix), int(iy)
        self.component = component
        self.current = float(current)
        self.temporal = temporal
        self.injection = None

    def attach(self, engine):
        self.injection = engine.add_current(self.ix, self.iy, self.component, self.current, self.temporal)


def _shared_temporal(physics):
    """The waveform every source of `physics` shares (type and parameters), else None."""
    sources = list(physics.sources)
    if not sources:
        return None
    first = sources[0].temporal
    if any(type(source.temporal) is not type(first) or vars(source.temporal) != vars(first) for source in sources):
        return None
    return first


def shared_spectrum(physics, frequencies):
    """(S(omega) complex128 of the shared waveform at its injection times over the steps run, valid mask where |S|^2 >=
    1e-6 |S(carrier)|^2), None without one shared waveform."""
    temporal = _shared_temporal(physics)
    if temporal is None:
        return None
    engine = physics.engine
    spectrum = source_spectrum(temporal, engine.dt, frequencies, engine.n_step)
    carrier = source_spectrum(temporal, engine.dt, [_carrier_angular_frequency(temporal) / (2.0 * math.pi)],
                              engine.n_step)[0]
    carrier_power = abs(carrier) ** 2
    valid = (np.abs(spectrum) ** 2 >= _VALID_SPECTRUM_FRACTION * carrier_power) & (carrier_power > 0.0)
    return spectrum, valid


def _inverse(spectrum, valid):
    """1 / S where valid, else 0 (complex128)."""
    return np.where(valid, 1.0 / np.where(valid, spectrum, 1.0), 0.0)


class BORFluxSpectrum:
    """DFT of the E points of a line of cells and the H points on their hi face (the line_flux pairing): normal 'z'
    (annulus over r cells) or 'r' (cylinder over z cells), recorded from attach on."""

    def __init__(self, physics, normal, line_cell, span_cells, frequencies):
        self.physics = physics
        self.normal = normal
        self.line_cell = int(line_cell)
        self.span = (int(span_cells[0]), int(span_cells[1]))
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.spectra = None

    def attach(self, engine):
        lo, hi = self.span
        row, edge = self.line_cell, self.line_cell + 1
        if self.normal == 'z':
            selections = {'er': (slice(lo, hi + 1), row), 'ephi': (slice(lo, hi), row),
                          'hphi': (slice(lo, hi + 1), edge), 'hr': (slice(lo, hi), edge)}
        else:
            selections = {'ephi': (row, slice(lo, hi)), 'ez': (row, slice(lo, hi + 1)),
                          'hz': (edge, slice(lo, hi)), 'hphi': (edge, slice(lo, hi + 1))}
        e_index, a_index = point_index(engine.shape, selections)
        self.spectra = engine.add_spectra(self.frequencies, e_index=e_index, a_index=a_index)

    def order_powers(self, scales=None):
        """float64 (n_f, n_orders) pi sum Re(E H*) (line_flux) per engine order (an order sum: each m with its -m, the
        orders orthogonal in phi) of the DFT phasors, each frequency times |scale|^2 (scales None: 1, the raw
        dt-weighted spectra in a.u.; 1 / S: the cw power in W)."""
        engine = self.physics.engine
        position = {order: column for column, order in enumerate(engine.orders)}
        powers = np.zeros((self.frequencies.size, len(engine.orders)), dtype=np.float64)
        for index in range(self.frequencies.size):
            scale = 1.0 if scales is None else scales[index]
            for order, fields in engine.spectra_terms(self.spectra, index, scale):
                column = position[abs(order) if engine.order_sum else order]
                powers[index, column] += line_flux(engine.assembly, fields, self.normal, self.line_cell, self.span)
        return powers

    def powers(self, scales=None):
        """float64 (n_f,) total of order_powers."""
        return self.order_powers(scales).sum(axis=1)


class BORFarFieldSpectrum:
    """DFT of the E / H bands around the faces of a closed (r, z) cell box (body_of_revolution.far_field_samples);
    mirror_sign +1 / -1 mirrors a box bottom on a pmc / pec z_lo wall."""

    def __init__(self, physics, r_cells, z_cells, mirror_sign, frequencies):
        self.physics = physics
        self.r_cells = (int(r_cells[0]), int(r_cells[1]))
        self.z_cells = (int(z_cells[0]), int(z_cells[1]))
        self.mirror_sign = mirror_sign
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.spectra = None

    def attach(self, engine):
        (i0, i1), (j0, j1) = self.r_cells, self.z_cells
        selections = []
        everywhere = slice(None)
        z_edges = [j1] + ([j0] if self.mirror_sign is None else [])
        r_edges = [i1] + ([i0] if i0 > 0 else [])
        for edge in z_edges:   # tangential E interpolated from rows edge - 1, edge; H on the edge row
            selections.append({'er': (everywhere, slice(edge - 1, edge + 1)),
                               'ephi': (everywhere, slice(edge - 1, edge + 1)),
                               'hphi': (everywhere, edge), 'hr': (everywhere, edge)})
        for edge in r_edges:
            selections.append({'ephi': (slice(edge - 1, edge + 1), everywhere),
                               'ez': (slice(edge - 1, edge + 1), everywhere),
                               'hz': (edge, everywhere), 'hphi': (edge, everywhere)})
        indices = [point_index(engine.shape, selection) for selection in selections]
        e_index = np.unique(np.concatenate([pair[0] for pair in indices]))
        a_index = np.unique(np.concatenate([pair[1] for pair in indices]))
        self.spectra = engine.add_spectra(self.frequencies, e_index=e_index, a_index=a_index)

    def term_samples(self, index, scale=1.0):
        """[(signed order, far_field_samples)] of frequency `index` (DFT phasors times scale; engine.spectra_terms)."""
        engine = self.physics.engine
        return [(order, far_field_samples(engine.assembly, fields, self.r_cells, self.z_cells, self.mirror_sign))
                for order, fields in engine.spectra_terms(self.spectra, index, scale)]


def fdtd_bor_flux_result(monitor) -> dict:
    """'flux_spectrum' of a BORFluxSpectrum in monitor order: power pi sum Re(E H*) of the dt-weighted DFT spectra
    ('a.u.'); with one shared source waveform also power_per_cw [W] (normalised by |S(omega)|^2, NaN outside it). An
    order sum: the 3D totals, extras azimuthal_orders, power_orders (and power_orders_per_cw): (n_f, n_orders) per m with
    its -m."""
    frequency = monitor.frequencies.copy()
    order_sum = monitor.physics.engine.order_sum
    order_powers = monitor.order_powers()
    result = {'kind': 'flux_spectrum', 'frequency': frequency, 'wavelength': C0 / frequency,
              'power': order_powers.sum(axis=1), 'unit': 'a.u.'}
    if order_sum:
        result.update(azimuthal_orders=list(monitor.physics.engine.orders), power_orders=order_powers)
    shared = shared_spectrum(monitor.physics, frequency)
    if shared is not None:
        spectrum, valid = shared
        # quadratic in the field: the per-cw power is the raw one times |1 / S|^2 (no second phasor recovery)
        cw_powers = np.where(valid[:, None], order_powers * (np.abs(_inverse(spectrum, valid)) ** 2)[:, None], np.nan)
        result['power_per_cw'] = cw_powers.sum(axis=1)
        if order_sum:
            result['power_orders_per_cw'] = cw_powers
    return result


def fdtd_bor_far_field_result(monitor, n_angles=360) -> dict:
    """'far_field' of a BORFarFieldSpectrum in monitor order (angles as bor_far_field_result): with one shared source
    waveform power per steradian of the cw source [W/sr] (NaN outside its spectrum), else of the raw spectra (a.u.). An
    order sum (as em_fdfd_bor's): the summed field's pattern (cos(angle) >= 0 at phi = 0, else phi = pi), extras
    power_azimuthal_mean, power_orders (n_f, n_orders, n_angles) and azimuthal_orders instead of azimuthal_order."""
    angles, theta = bor_far_field_angles(n_angles)
    frequency = monitor.frequencies.copy()
    shared = shared_spectrum(monitor.physics, frequency)
    scales, valid = ((np.ones(frequency.size), np.ones(frequency.size, dtype=bool)) if shared is None
                     else (_inverse(*shared), shared[1]))
    engine = monitor.physics.engine
    mask = np.where(valid, 1.0, np.nan)[:, None]
    result = {'kind': 'far_field', 'frequency': frequency, 'wavelength': C0 / frequency, 'angles': angles,
              'unit': 'a.u.' if shared is None else 'W/sr', 'quantity': FAR_FIELD_QUANTITY, 'angle_axes': ('r', 'z')}
    if engine.order_sum:
        phi = np.where(np.cos(angles) < 0.0, math.pi, 0.0)
        parts = [order_sum_far_field(monitor.term_samples(index, scales[index]), 2.0 * math.pi * value, theta, phi)
                 for index, value in enumerate(frequency)]
        result.update(azimuthal_orders=list(engine.orders),
                      power_azimuthal_mean=np.array([sum(part[3].values()) for part in parts]) * mask,
                      power_orders=np.array([[part[3][order] for order in engine.orders] for part in parts])
                      * mask[:, :, None])
    else:
        order = monitor.physics.azimuthal_order
        parts = [far_field_power(order, 2.0 * math.pi * value, monitor.term_samples(index, scales[index])[0][1], theta)
                 for index, value in enumerate(frequency)]
        result['azimuthal_order'] = order
    result.update(power=np.array([part[0] for part in parts]) * mask,
                  power_theta=np.array([part[1] for part in parts]) * mask,
                  power_phi=np.array([part[2] for part in parts]) * mask)
    return result


def fdtd_bor_mode_result(monitor, physics) -> dict:
    """fdtd_bor_ports.fdtd_bor_mode_monitor_result of a BORFDTDModeMonitor with the shared source spectrum."""
    return fdtd_bor_mode_monitor_result(monitor, physics, shared_spectrum(physics, monitor.frequencies))


class BORHarmonic:
    """frequency_domain accumulator of a BORFDTDPhysics (the harmonic_fdtd.HarmonicAccumulator protocol: source_end,
    field_energy, normalize, select): complex64 DFT of every E and H point (of every member of an order sum) at the
    study frequencies, divided once by S(omega) of the shared pulse into em_fdfd_bor-convention phasors (at omega_tilde:
    fdtd_bor module header; an order sum: the physical phasors of the phi = 0 field)."""

    def __init__(self, physics, frequencies):
        engine = physics.engine
        if engine.n_step != 0:
            raise ValueError(f'attach_harmonic needs a BOR FDTD physics that has not stepped yet (n_step = '
                             f'{engine.n_step}): S(omega) is summed from the first step')
        self.source_end = float(_shared_pulse(physics).duration)   # [s] 2 t0 of the shared pulse
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.engine = engine
        self.spectra = engine.add_spectra(self.frequencies, dtype=np.complex64)
        self.normalized = False
        self.index = 0

    def field_energy(self, physics):
        return physics.engine.field_energy()

    def normalize(self, physics):
        """Divide every spectrum by S(omega) (once) -> warnings for frequencies where |S|^2 is below 1e-6 of its carrier
        value."""
        if self.normalized:
            return []
        spectrum, valid = shared_spectrum(physics, self.frequencies)
        self.spectra.scale(physics.xp.asarray(_inverse(spectrum, spectrum != 0.0).astype(np.complex64)))
        self.normalized = True
        if bool(np.all(valid)):
            return []
        outside = C0 / self.frequencies[~valid]
        return [f"physics '{getattr(physics, 'name', 'em')}': {outside.size} study wavelength(s) between "
                f'{outside.min():.4g} and {outside.max():.4g} m lie outside the source pulse spectrum (|S|^2 below '
                '1e-6 of its carrier value), so their harmonic fields are noise; move the source wavelength / '
                'bandwidth over the study range']

    def select(self, index):
        index = int(index)
        if not 0 <= index < self.frequencies.size:
            raise IndexError(f'harmonic frequency index {index} out of range for {self.frequencies.size} wavelengths')
        self.index = index

    def yee(self):
        """Yee-point phasor dict (complex, on the array module) of the selected frequency (engine.harmonic_yee);
        ValueError before normalize()."""
        if not self.normalized:
            raise ValueError('harmonic BOR FDTD fields exist only after the pulse of a frequency_domain study')
        return self.engine.harmonic_yee(self.spectra, self.index)

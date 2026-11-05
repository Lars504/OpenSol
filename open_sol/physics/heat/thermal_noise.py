"""Thermorefractive noise by the fluctuation-dissipation theorem on a z-invariant 2D heat cross-section: the one-sided
PSD of X = integral w(r) dT(r) dV is S_X(f) = 4 k_B T0^2 integral dq/2pi |g~(q - q0)|^2 Re sum conj(w) u A, with
(j omega rho cp + k_zz q^2 - div k grad) u = w per axial wavenumber q by the heat 'shifted' solver (sparse_direct: host
SuperLU per (omega, q); line_block: batched GPU solves of all (omega, q) of a channel). Mode-pair
weightings w come from em_modes (Lorentz reciprocity with 1 W modes), illumination weightings from an FDFD field on
the same cross-section (k_z = 0) into an em_modes mode. Particle monitors replace dT by the number-density
fluctuation dc of dilute diffusing scatterers (e.g. proteins): 4 n0 instead of 4 k_B T0^2, (j omega + D q^2 - div D
grad) u = w on the particle cells, dn per particle instead of dndT."""
import math
import time

import numpy as np

from ...core.backend import array_module_of, to_host
from ...core.constants import C0, EPS0, FACES, K_B
from ...solvers.line_block import LineBlockLayout, LineBlockSolver
from ...solvers.sparse_direct import SparseDirectSolver
from ..em.boundaries import MIRROR_TYPES, BCType

WEIGHTINGS = ('mode_pair', 'illumination', 'expression')
DIRECTIONS = ('forward', 'backward')
# collected mode: fundamental quasi-TE / quasi-TM (highest-ranked mode of the family), both, or a mode number
COLLECTED_SELECTIONS = ('te', 'tm', 'te_and_tm', 'number')
INCIDENT_SELECTIONS = ('collected', 'te', 'tm', 'number')   # 'collected': each collected mode scatters into itself
AXIAL_PROFILES = ('uniform', 'gaussian')
OTHER_CONTRIBUTION = 'other'   # contribution of weighted selected cells that lie in no material
# outermost window cell = ghost outside the wall plane (cross_section_modes); mirror ghosts (x2 mode each, MIRROR_TYPES)
# are padded outside the window, so heat's window edge is the mirror plane
_WALL_BC_TYPES = frozenset({BCType.PEC, BCType.PMC})
_ZERO_AXIAL_LENGTHS = 1e-9   # |q| * axial length below this: the axial term relaxes no null mode (q = 0 sample)


class ThermalNoiseMonitor:
    """One thermorefractive_noise feature: the selected window cells, the weighting (a fixed 'expression' array, or a
    mode pair resolved after the mode solve), the axial profile and the last spectrum."""

    def __init__(self, cell_mask, axial_profile, axial_length, axial_points=1, weight=None, axial_wavenumber=0.0,
                 mode_pair=None, contributions=(), particle=None):
        if axial_profile not in AXIAL_PROFILES:
            raise ValueError(f"axial_profile must be one of {', '.join(AXIAL_PROFILES)}, got {axial_profile!r}")
        if not (math.isfinite(axial_length) and axial_length > 0.0):
            raise ValueError(f'axial length must be finite and > 0 m, got {axial_length!r}')
        if (weight is None) == (mode_pair is None):
            raise ValueError('give exactly one of weight (expression weighting) and mode_pair')
        if particle is not None and mode_pair is None:
            raise ValueError('a particle monitor needs a mode_pair (mode-pair or illumination weighting)')
        self.cell_mask = np.asarray(cell_mask, dtype=bool).ravel()   # flat window mask of the selected cells
        self.axial_profile = axial_profile
        self.axial_length = float(axial_length)   # uniform: interaction length L; gaussian: waist w0 of g = exp(-z^2/w0^2)
        self.axial_points = int(axial_points)
        self.weight = None if weight is None else np.asarray(weight, dtype=np.complex128).ravel()
        self.axial_wavenumber = float(axial_wavenumber)
        # dict physics, collected (selection, number, direction), incident (selection, number, direction) or None,
        # illumination (FDFD physics name, then incident is None) or None; resolved per collected mode at the solve
        self.mode_pair = mode_pair
        # [(material name, flat window mask)]: cell-disjoint parts of the selection whose self terms are reported
        self.contributions = [(name, np.asarray(mask, dtype=bool).ravel()) for name, mask in contributions]
        # None (temperature fluctuations) or dict index_increment (dn per particle per m^-3 [m^3]), diffusivity
        # [m^2/s], number_density n0 [1/m^3]: dilute particles diffusing on the selected cells
        self.particle = particle
        self.frequencies = None      # [Hz] of the last solve
        # per collected mode (one for expression weightings): dict tag ('' alone, else 'te' / 'tm'), psd (one-sided
        # PSD of X at self.frequencies), contribution_psd {part name: self-term PSD}, info (q0, variance, mode ...)
        self.channels = []

    def band_fraction(self):
        """Smallest share of the relaxing variance (variance - static_variance) inside the solved band over the channels
        (NaN before a solve or without relaxing variance)."""
        fractions = [channel['info']['band_fraction'] for channel in self.channels]
        return min((fraction for fraction in fractions if not math.isnan(fraction)), default=math.nan)

    def summary(self):
        """One line on the last spectra for the study log; None before a solve."""
        if not self.channels:
            return None
        texts = []
        for channel in self.channels:
            info = channel['info']
            mode_text = ''
            if 'collected_mode' in info:
                mode_text = (f"{channel['tag'] or 'collected'} mode {info['collected_mode']} (n_eff "
                             f"{info['collected_n_eff']:.5g}, te_fraction {info['collected_te_fraction']:.3g}): ")
            particle_text = ''
            if self.particle is not None:
                particle_text = (f", sensing volume {info['sensing_volume']:.4g} m^3 holding "
                                 f"{info['particles_in_sensing_volume']:.4g} particles, bound signal up to "
                                 f"{info['bound_signal_max']:.4g}")
            static_text = ''
            if info['static_variance'] > 0.0:
                static_text = f", static (non-relaxing) share {100.0 * info['static_variance'] / info['variance']:.3g} %"
            texts.append(f"{mode_text}q0 = {info['axial_wavenumber']:.4g} 1/m, total variance {info['variance']:.4g} "
                         f"(band holds {100.0 * info['band_fraction']:.3g} % of the relaxing part{static_text})"
                         f"{particle_text}, "
                         f"{info['factorizations']} factorisations in {info['seconds']:.3g} s")
        return (f"{len(self.frequencies)} frequencies {self.frequencies[0]:.3g}-{self.frequencies[-1]:.3g} Hz; "
                + '; '.join(texts))


class _HarmonicDiffusionOperator:
    """Harmonic noise solve of a z-invariant diffusing field on the solved cells `index`: S = 4 s Re sum conj(w) u A
    with (j omega C A + q^2 a A + K) u = w A, by the heat 'shifted' solver (host SuperLU per shift, or the GPU
    line-block solver); subclasses set K, C, a, s, `anchored` (rows of K with a conductance to a reservoir) and call
    _init_solver."""

    _components = None   # (labels, floating flags, axial sums) of the link graph of K, built on first use

    def _init_solver(self, grid_shape, shifted_solver):
        """Solver state: the window shape of `index` and the shifted solver (solvers.sparse_direct.SparseDirectSolver
        or a LineBlockSolver shared with the other operators of the solve; None: a private SparseDirectSolver)."""
        if shifted_solver is None:
            shifted_solver = SparseDirectSolver()
        if not isinstance(shifted_solver, (SparseDirectSolver, LineBlockSolver)):
            raise TypeError(f'shifted_solver must be a SparseDirectSolver or a LineBlockSolver, got '
                            f'{type(shifted_solver).__name__}')
        self.grid_shape = tuple(int(count) for count in grid_shape)
        self.shifted_solver = shifted_solver
        # SuperLU of the shifts line_block rejects (counted there as rejected_shifts)
        self.reference_solver = (shifted_solver if isinstance(shifted_solver, SparseDirectSolver)
                                 else SparseDirectSolver())
        self.factorizations = 0  # SuperLU factorisations + line-block shift factorisations (per channel info)

    def check_weight(self, weight, label):
        """No check: every solved cell has capacity."""

    def shifted_spectral_densities(self, weights, shifts, on_shift=None):
        """(combined (n_shifts,), parts (n_shifts, n_weights)) of spectral_densities at every (omega, q) of shifts: SuperLU
        per shift, or batched line-block solves (rejected shifts by SuperLU); on_shift() after each shift."""
        n_shifts = len(shifts)
        combined, parts = np.zeros(n_shifts), np.zeros((n_shifts, len(weights)))
        if not isinstance(self.shifted_solver, LineBlockSolver):
            for position, (omega, axial_wavenumber) in enumerate(shifts):
                combined[position], parts[position] = self.spectral_densities(weights, omega, axial_wavenumber)
                if on_shift is not None:
                    on_shift()
            return combined, parts
        drives = np.stack([weight[self.index] * self.cell_area[self.index] for weight in weights])
        total_drive = drives.sum(axis=0)
        scale = 4.0 * self.fluctuation_strength
        coefficients = [(1j * omega, axial_wavenumber ** 2) for omega, axial_wavenumber in shifts]
        for start, responses, _relative, accepted in self._line_block_solver().solve_shifted(coefficients, drives):
            self.factorizations += responses.shape[0]
            for offset, response in enumerate(responses):
                position = start + offset
                if accepted[offset]:
                    parts[position] = scale * np.real(np.sum(np.conj(drives) * response, axis=1))
                    combined[position] = (parts[position, 0] if len(weights) == 1 else
                                          scale * float(np.real(np.vdot(total_drive, response.sum(axis=0)))))
                else:   # counted as the solver's rejected_shifts
                    combined[position], parts[position] = self.spectral_densities(weights, *shifts[position])
                if on_shift is not None:
                    on_shift()
        return combined, parts

    def _line_block_solver(self):
        """The shared LineBlockSolver bound to K with the shift terms (C, a) (re-bound when another operator holds it)."""
        solver = self.shifted_solver
        if solver.owner is not self:
            solver.set_operator(self.conduction, LineBlockLayout.from_grid_cells(self.index, self.grid_shape),
                                (self.capacity, self.axial), owner=self)
        return solver

    def spectral_densities(self, weights, omega, axial_wavenumber):
        """(PSD of sum(weights), [PSD of each weight alone]): per-unit-length one-sided 4 s Re sum conj(w) u A
        [X^2 / (Hz m)] at omega [rad/s] and q [1/m]; one factorisation, one back substitution per weight."""
        import scipy.sparse
        diagonal = 1j * omega * self.capacity + axial_wavenumber ** 2 * self.axial
        factor = self.reference_solver.factorize_host(self.conduction + scipy.sparse.diags(diagonal, format='csc'))
        self.factorizations += 1
        scale = 4.0 * self.fluctuation_strength
        total_drive, total_response, parts = 0.0, 0.0, []
        for weight in weights:
            drive = weight[self.index] * self.cell_area[self.index]
            response = factor.solve_host(drive)
            parts.append(scale * float(np.real(np.vdot(drive, response))))
            total_drive, total_response = total_drive + drive, total_response + response   # linear in w
        if len(weights) == 1:
            return parts[0], parts
        return scale * float(np.real(np.vdot(total_drive, total_response))), parts

    def variance_per_length(self, weight):
        """Sum rule: integral of the PSD over all f = s sum |w|^2 A / C over the solved cells [X^2/m]."""
        solved = self.index
        capacity = self.capacity_density[solved]
        weighted = np.abs(weight[solved]) ** 2 * self.cell_area[solved]   # 0 where C = 0 (check_weight)
        return (self.fluctuation_strength
                * float(np.sum(np.divide(weighted, capacity, out=np.zeros_like(weighted), where=capacity > 0.0))))

    def static_variance_per_length(self, weight, zero_wavenumber):
        """Part of variance_per_length in null modes of K + q^2 a A (u constant on a connected region without anchor,
        at q = 0 or without axial conduction): s |sum w A|^2 / sum C per region, never relaxing (f = 0) [X^2/m]."""
        if self._components is None:
            import scipy.sparse.csgraph
            links = abs(self.conduction).tocsr()
            links.eliminate_zeros()
            count, labels = scipy.sparse.csgraph.connected_components(links, directed=False)
            floating = np.ones(count, dtype=bool)
            floating[labels[self.anchored]] = False
            self._components = (labels, floating, np.bincount(labels, weights=self.axial, minlength=count))
        labels, floating, axial_sums = self._components
        null = floating & (zero_wavenumber | (axial_sums <= 0.0))
        if not null.any():
            return 0.0
        count = null.size
        drive = weight[self.index] * self.cell_area[self.index]
        summed_drive = (np.bincount(labels, weights=drive.real, minlength=count)
                        + 1j * np.bincount(labels, weights=drive.imag, minlength=count))
        capacities = np.bincount(labels, weights=self.capacity, minlength=count)
        shares = np.divide(np.abs(summed_drive) ** 2, capacities, out=np.zeros(count),
                           where=null & (capacities > 0.0))
        return self.fluctuation_strength * float(np.sum(shares))


class HarmonicHeatOperator(_HarmonicDiffusionOperator):
    """Temperature noise of an assembled 2D heat window: conduction + boundary / loss conductances K [W/(m K)] on the
    solved cells (held cells do not fluctuate), capacity rho cp A, axial k_zz A and s = k_B T0^2."""

    def __init__(self, heat, shifted_solver=None):
        system = heat.system
        if system.dimensions != 2 or system.axisymmetric:
            raise ValueError(f"heat '{heat.name}': thermal noise needs a planar 2D heat physics (z-invariant cross-section)")
        if system.operator_dirty or system.operator_full is None:
            heat.rebuild_operator()
        window = system.window
        materials = heat.domain.materials
        self.cell_area = to_host(system.cell_area).astype(np.float64).ravel()
        self.capacity_density = to_host(materials.rho[window].astype(np.float64)
                                        * materials.cp[window].astype(np.float64)).ravel()
        self.axial_conductivity = to_host(materials.k_zz[window]).astype(np.float64).ravel()
        operator_full = system.operator_full   # cupyx CSR on the GPU backend, scipy CSR on the CPU backend
        operator = (operator_full.get() if hasattr(operator_full, 'get') else operator_full).tocsr().astype(np.float64)
        self.held = to_host(system.fixed_user).ravel()
        self.solved = ~self.held & ((operator.diagonal() > 0.0) | (self.capacity_density > 0.0))
        index = np.flatnonzero(self.solved)
        self.index = index
        kept_rows = operator[index]
        self.conduction = kept_rows[:, index].tocsc()
        # anchors (as the DiffusionSystem anchor check): volumetric loss, Dirichlet / h > 0 faces, links to held cells
        anchor_cells = system.extra_diag > 0
        for cells, conductance, _ in system.face_terms.values():
            anchor_cells[cells] |= conductance > 0
        held_index = np.flatnonzero(self.held)
        held_links = (np.asarray(abs(kept_rows[:, held_index]).sum(axis=1)).ravel() if held_index.size
                      else np.zeros(index.size))
        self.anchored = to_host(anchor_cells).ravel()[index] | (held_links > 0.0)
        self.capacity = self.capacity_density[index] * self.cell_area[index]
        self.axial = self.axial_conductivity[index] * self.cell_area[index]
        self.fluctuation_strength = K_B * float(heat.T0) ** 2
        self._init_solver(system.window_shape, shifted_solver)

    def check_weight(self, weight, label):
        """ValueError when the weighting is non-zero on a fluctuating cell without heat capacity (infinite variance)."""
        uncapacitated = (np.abs(weight) > 0.0) & ~self.held & ~(self.capacity_density > 0.0)
        if np.any(uncapacitated):
            share = (float(np.sum(np.abs(weight[uncapacitated]) ** 2 * self.cell_area[uncapacitated]))
                     / float(np.sum(np.abs(weight) ** 2 * self.cell_area)))
            raise ValueError(f'{label}: the weighting is non-zero on {int(np.count_nonzero(uncapacitated))} cells '
                             f'without heat capacity (rho * cp = 0; {100.0 * share:.3g} % of sum |w|^2 dA), whose '
                             'temperature noise would have infinite variance: set rho and cp of every material there, '
                             'or restrict the feature selection')


class HarmonicParticleOperator(_HarmonicDiffusionOperator):
    """Number-density noise of dilute independent particles (Poisson: <dc dc'> = n0 delta) diffusing with D on the
    particle cells of a heat window, no flux through their edges: C = 1, K = -div D grad, a = D, s = n0 [1/m^3]."""

    def __init__(self, heat, particle_cells, diffusivity, number_density, shifted_solver=None):
        from ...core.fv_diffusion import ScalarBoundaryConfig, build_anisotropic_laplacian
        system = heat.system
        if system.dimensions != 2 or system.axisymmetric:
            raise ValueError(f"heat '{heat.name}': particle noise needs a planar 2D heat physics (z-invariant "
                             'cross-section)')
        particle_cells = np.asarray(particle_cells, dtype=bool).ravel()
        if not particle_cells.any():
            raise ValueError(f"heat '{heat.name}': the particle noise selection holds no window cell")
        self.cell_area = to_host(system.cell_area).astype(np.float64).ravel()
        self.capacity_density = particle_cells.astype(np.float64)
        # host coefficient: a scipy CSR on either backend (the shifted solvers take host operators)
        coefficient = np.where(particle_cells, float(diffusivity), 0.0).reshape(system.window_shape)
        operator = build_anisotropic_laplacian(coefficient, coefficient, np.zeros_like(coefficient), system.dx_window,
                                               system.dy_window, ScalarBoundaryConfig(), dtype=np.float64)[0]
        index = np.flatnonzero(particle_cells)   # D = 0 outside: no face link leaves the particle cells
        self.index = index
        self.conduction = operator.tocsr()[index][:, index].tocsc()
        self.capacity = self.cell_area[index]
        self.axial = float(diffusivity) * self.cell_area[index]
        self.anchored = np.zeros(index.size, dtype=bool)   # particles never leave: the count of each region is conserved
        self.fluctuation_strength = float(number_density)
        self._init_solver(system.window_shape, shifted_solver)


def axial_samples(profile, length, points, central_wavenumber):
    """[(|q| [1/m], length weight [m])] with integral dq/2pi |g~(q - q0)|^2 f(q) ~ sum weight f(|q|): uniform g over L
    (f(q0) L, long L), gaussian g = exp(-z^2/w0^2) by Gauss-Hermite (equal |q| merged), and the sum of the weights
    (= integral |g|^2 dz)."""
    if profile == 'uniform':
        return [(abs(central_wavenumber), length)], length
    nodes, weights = np.polynomial.hermite.hermgauss(int(points))
    wavenumbers = np.abs(central_wavenumber + math.sqrt(2.0) * nodes / length)
    order = np.argsort(wavenumbers)
    tolerance = 1e-9 * (abs(central_wavenumber) + 1.0 / length)   # q0 = 0 pairs the symmetric nodes
    samples = []
    for wavenumber, weight in zip(wavenumbers[order], weights[order]):
        length_weight = length / math.sqrt(2.0) * float(weight)
        if samples and wavenumber - samples[-1][0] <= tolerance:
            samples[-1] = (samples[-1][0], samples[-1][1] + length_weight)
        else:
            samples.append((float(wavenumber), length_weight))
    return samples, length * math.sqrt(math.pi / 2.0)


def axial_fourth_moment(profile, length):
    """integral |g|^4 dz [m]: L (uniform), w0 sqrt(pi) / 2 (gaussian g = exp(-z^2/w0^2))."""
    return length if profile == 'uniform' else length * math.sqrt(math.pi) / 2.0


def wall_ghost_cells(heat, boundary_config):
    """Flat window mask of the outermost cells of every em pec / pmc face: ghosts outside the wall plane, which lies on
    their inner face (cross_section_modes)."""
    ghost = np.zeros(heat.system.window_shape, dtype=bool)
    if not boundary_config.periodic_x:
        ghost[0, :] |= getattr(boundary_config, 'x_lo') in _WALL_BC_TYPES
        ghost[-1, :] |= getattr(boundary_config, 'x_hi') in _WALL_BC_TYPES
    if not boundary_config.periodic_y:
        ghost[:, 0] |= getattr(boundary_config, 'y_lo') in _WALL_BC_TYPES
        ghost[:, -1] |= getattr(boundary_config, 'y_hi') in _WALL_BC_TYPES
    return ghost.ravel()


def illumination_fields(heat, illumination_physics):
    """Window ex, ey, ez [V/m] (flat complex128, cell centres) of the last solve of an FDFD physics on the cross-section
    (k_z = 0): ez its E_z solution, hz the face means of its in-plane E (as the renderer)."""
    from ..em.harmonic import face_mean_at_centres
    if illumination_physics.solution is None:
        raise ValueError(f"illumination physics '{illumination_physics.name}' has no solve (it runs as a background "
                         'physics of the thermal_noise study)')
    window = heat.system.window
    size = heat.system.n_window
    zeros = np.zeros(size, dtype=np.complex128)
    if illumination_physics.polarization == 'ez':
        return {'ex': zeros, 'ey': zeros.copy(), 'ez': to_host(illumination_physics.solution[window]).ravel()}
    boundary_config = illumination_physics.boundary_config
    wrap_x, wrap_y = illumination_physics.lower_wrap_factors   # Bloch phase of the wrapped lo face (1 otherwise)
    ex_faces, ey_faces = illumination_physics.face_fields()   # E_x on hi-y faces, E_y on hi-x faces
    xp = array_module_of(ex_faces)
    return {'ex': to_host(face_mean_at_centres(ex_faces, 1, boundary_config.periodic_y, xp=xp,
                                               wrap_factor=wrap_y)[window]).ravel(),
            'ey': to_host(face_mean_at_centres(ey_faces, 0, boundary_config.periodic_x, xp=xp,
                                               wrap_factor=wrap_x)[window]).ravel(),
            'ez': zeros}


def mode_pair_weighting(heat, mode_physics, incident, collected, illumination_physics=None, index_increment=None):
    """(flat window weighting w complex128, q0 [1/m], mirror factor, vacuum wavelength [m]) of the scattered amplitude
    into the collected 1 W mode: w = omega eps0 / 4 sum_i 2 n_i dndT_i E_inc,i E_partner,i, the partner travelling
    against the collected direction (Lorentz reciprocity); E_inc = the incident 1 W mode (w [1/(K m^3)]) or the field
    [V/m] of illumination_physics (k_z = 0; w [W^0.5/(K m^3)], incident None); 0 on wall ghost cells.
    index_increment [m^3] replaces dndT_i (isotropic dn per particle per m^-3: w = amplitude per particle)."""
    from ..em.cross_section_modes import centre_fields
    modes = mode_physics.modes
    roles = (('collected', collected),) if incident is None else (('incident', incident), ('collected', collected))
    for role, (number, _direction) in roles:
        if not 1 <= number <= len(modes):
            raise ValueError(f"{role} mode {number} of '{mode_physics.name}' does not exist: its mode solve found "
                             f'{len(modes)} mode(s) (raise n_modes, set n_eff_guess, or widen the domain)')
    boundary_config = mode_physics.boundary_config
    if boundary_config.has_bloch_phase:   # the weightings pair modes of one k; definition_modes refuses it too
        raise ValueError(f"thermal noise needs the modes of '{mode_physics.name}' at Bloch k = 0 (bloch_kx / "
                         'bloch_ky 0)')
    periodic = (bool(boundary_config.periodic_x), bool(boundary_config.periodic_y))
    window = heat.system.window
    omega = float(mode_physics.solved_omega)

    def oriented_fields(number, sign):
        """Window ex, ey, ez of mode `number` travelling along sign * z (backward: E_z flips, beta -> -beta)."""
        centres = centre_fields(modes[number - 1], *periodic)
        return {name: centres[name][window].ravel() * (sign if name == 'ez' else 1.0) for name in ('ex', 'ey', 'ez')}

    partner_sign = -1.0 if collected[1] == 'forward' else 1.0
    partner_fields = oriented_fields(collected[0], partner_sign)
    if illumination_physics is None:
        incident_sign = 1.0 if incident[1] == 'forward' else -1.0
        incident_fields = oriented_fields(incident[0], incident_sign)
        incident_axial = incident_sign * complex(modes[incident[0] - 1].beta).real
    else:
        if illumination_physics.boundary_config.differences(boundary_config):
            raise ValueError(f"illumination physics '{illumination_physics.name}' and '{mode_physics.name}' need the "
                             'same boundary conditions')
        incident_fields = illumination_fields(heat, illumination_physics)   # raises before a solve
        illumination_omega = float(illumination_physics.solved_omega)
        if abs(illumination_omega - omega) > 1e-9 * omega:
            raise ValueError(f"illumination physics '{illumination_physics.name}' solved at "
                             f'{2.0 * math.pi * C0 / illumination_omega:.6g} m, the modes of \'{mode_physics.name}\' '
                             f'at {2.0 * math.pi * C0 / omega:.6g} m: set equal wavelengths')
        incident_axial = 0.0
    materials = heat.domain.materials
    weight = np.zeros(incident_fields['ex'].size, dtype=np.complex128)
    for component, permittivity, thermo_optic in (('ex', 'eps_xx', 'dndT_xx'), ('ey', 'eps_yy', 'dndT_yy'),
                                                 ('ez', 'eps_r', 'dndT_zz')):
        index = np.sqrt(np.maximum(to_host(getattr(materials, permittivity)[window]).astype(np.float64), 0.0))
        if index_increment is None:
            index_change = to_host(getattr(materials, thermo_optic)[window]).astype(np.float64)
        else:
            index_change = float(index_increment)
        weight += (2.0 * index * index_change).ravel() * incident_fields[component] * partner_fields[component]
    weight *= omega * EPS0 / 4.0
    partner_beta = complex(modes[collected[0] - 1].beta).real
    axial_wavenumber = abs(incident_axial + partner_sign * partner_beta)
    n_mirrors = sum(getattr(boundary_config, face) in MIRROR_TYPES for face in FACES)
    weight[wall_ghost_cells(heat, boundary_config)] = 0.0
    return weight, axial_wavenumber, float(2 ** n_mirrors), 2.0 * math.pi * C0 / omega


def fundamental_mode(mode_physics, family):
    """1-based number of the highest-ranked (largest Re beta^2) mode of family 'te' (te_fraction > 0.5: quasi-TE,
    E_x dominant) or 'tm' (quasi-TM) of a solved em_modes physics; ValueError when none was found."""
    modes = mode_physics.modes
    for number, mode in enumerate(modes, start=1):
        if (float(mode.te_fraction) > 0.5) == (family == 'te'):
            return number
    raise ValueError(f"no quasi-{family.upper()} mode among the {len(modes)} mode(s) of '{mode_physics.name}': raise "
                     'n_modes, lower n_eff_guess, or check the mirror faces (a mirror keeps one family only)')


def resolve_mode_channels(mode_physics, collected, incident):
    """[(tag, incident (number, direction) or None, collected (number, direction))] of a monitor's mode_pair
    selections: one per collected mode, tag '' for a single one, 'te' / 'tm' for te_and_tm."""
    selection, number, direction = collected
    if selection == 'number':
        picks = [('', number)]
    elif selection == 'te_and_tm':
        picks = [(family, fundamental_mode(mode_physics, family)) for family in ('te', 'tm')]
    else:
        picks = [('', fundamental_mode(mode_physics, selection))]
    channels = []
    for tag, collected_number in picks:
        incident_pair = None
        if incident is not None:
            incident_selection, incident_number, incident_direction = incident
            if incident_selection == 'collected':
                incident_number = collected_number
            elif incident_selection != 'number':
                incident_number = fundamental_mode(mode_physics, incident_selection)
            incident_pair = (incident_number, incident_direction)
        channels.append((tag, incident_pair, (collected_number, direction)))
    return channels


def solve_noise_monitor(heat, monitor, physics_by_name, frequencies, operator_for, progress=None):
    """Fill monitor.frequencies / channels: per collected mode (one channel for an expression weighting) the
    weighting (0 on every pec / pmc ghost, which keeps the heat face condition; mirror ghosts lie outside the window),
    then one solve per (frequency, |q|) shift (SuperLU, or line-block batches);
    progress(done, total) runs after every shift of the monitor (it may raise to cancel). particle: None (heat) or (particle cells, diffusivity,
    number density)."""
    label = f"heat '{heat.name}' {'particle' if monitor.particle is not None else 'thermal'} noise"
    particle = monitor.particle
    if monitor.mode_pair is None:
        operator = operator_for()
        plans = [('', monitor.weight, monitor.axial_wavenumber, 1.0, {})]
    else:
        pair = monitor.mode_pair
        mode_physics = physics_by_name.get(pair['physics'])
        if mode_physics is None or not hasattr(mode_physics, 'modes'):
            raise ValueError(f"{label}: no em_modes physics '{pair['physics']}'")
        illumination_physics = None
        if pair.get('illumination') is not None:
            illumination_physics = physics_by_name.get(pair['illumination'])
            if illumination_physics is None or not hasattr(illumination_physics, 'face_fields'):
                raise ValueError(f"{label}: no em_fdfd physics '{pair['illumination']}'")
        operator = operator_for(None if particle is None else (monitor.cell_mask, particle['diffusivity'],
                                                               particle['number_density']))
        plans = []
        for tag, incident, collected in resolve_mode_channels(mode_physics, pair['collected'], pair['incident']):
            weight, axial_wavenumber, mirror_factor, wavelength = mode_pair_weighting(
                heat, mode_physics, incident, collected, illumination_physics,
                index_increment=None if particle is None else particle['index_increment'])
            mode = mode_physics.modes[collected[0] - 1]
            extra = {'wavelength': wavelength, 'collected_mode': collected[0],
                     'collected_n_eff': complex(mode.n_eff).real, 'collected_te_fraction': float(mode.te_fraction)}
            if incident is not None:
                extra['incident_mode'] = incident[0]
            plans.append((tag, weight, axial_wavenumber, mirror_factor, extra))
    frequencies = np.asarray(frequencies, dtype=np.float64)
    sampled = [axial_samples(monitor.axial_profile, monitor.axial_length, monitor.axial_points, plan[2])
               for plan in plans]
    total = frequencies.size * sum(len(samples) for samples, _ in sampled)
    done = [0]

    def shift_done():
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    channels = []
    for (tag, weight, axial_wavenumber, mirror_factor, extra), (samples, axial_norm) in zip(plans, sampled):
        start = time.perf_counter()
        weight = np.where(monitor.cell_mask, weight, 0.0)
        operator.check_weight(weight, label)
        part_names, part_weights = _contribution_weights(monitor, weight)
        count_before = operator.factorizations
        # every (frequency, axial sample) of the channel is one shift of the same operator (batched by line_block)
        shift_frequencies = np.repeat(np.arange(frequencies.size), len(samples))
        shift_length_weights = np.tile([length_weight for _, length_weight in samples], frequencies.size)
        shifts = [(2.0 * math.pi * frequency, wavenumber) for frequency in frequencies for wavenumber, _ in samples]
        combined, parts = operator.shifted_spectral_densities(part_weights or [weight], shifts, shift_done)
        psd = np.zeros(frequencies.size)
        np.add.at(psd, shift_frequencies, shift_length_weights * combined)
        part_psd = np.zeros((len(part_names), frequencies.size))
        for part_index in range(len(part_names)):
            np.add.at(part_psd[part_index], shift_frequencies, shift_length_weights * parts[:, part_index])
        psd *= mirror_factor
        part_psd *= mirror_factor
        variance = mirror_factor * axial_norm * operator.variance_per_length(weight)
        static_variance = mirror_factor * sum(
            length_weight * operator.static_variance_per_length(
                weight, wavenumber * monitor.axial_length <= _ZERO_AXIAL_LENGTHS)
            for wavenumber, length_weight in samples)
        relaxing_variance = variance - static_variance
        info = dict(extra, axial_wavenumber=axial_wavenumber, mirror_factor=mirror_factor, variance=variance,
                    static_variance=static_variance,
                    band_fraction=(_band_integral(frequencies, psd) / relaxing_variance
                                   if relaxing_variance > 1e-12 * variance else math.nan),
                    axial_norm=axial_norm, axial_samples=len(samples),
                    contribution_variances={name: mirror_factor * axial_norm * operator.variance_per_length(part)
                                            for name, part in zip(part_names, part_weights)},
                    factorizations=operator.factorizations - count_before, seconds=time.perf_counter() - start)
        if particle is not None:
            info.update(_particle_info(heat, monitor, weight, mirror_factor, axial_norm))
        channels.append({'tag': tag, 'psd': psd, 'contribution_psd': dict(zip(part_names, part_psd)), 'info': info})
    monitor.frequencies, monitor.channels = frequencies, channels


def _particle_info(heat, monitor, weight, mirror_factor, axial_norm):
    """Single-particle figures: bound (static) signal |w|^2 of one particle at the strongest cell (spot centre) and its
    centre (x, y), sensing volume V_s = (integral |w g|^2 dV)^2 / integral |w g|^4 dV and n0 V_s."""
    cell_area = to_host(heat.system.cell_area).astype(np.float64).ravel()
    intensity = np.abs(weight) ** 2
    second_moment = mirror_factor * axial_norm * float(np.sum(intensity * cell_area))
    fourth_moment = (mirror_factor * axial_fourth_moment(monitor.axial_profile, monitor.axial_length)
                     * float(np.sum(intensity ** 2 * cell_area)))
    sensing_volume = second_moment ** 2 / fourth_moment if fourth_moment > 0.0 else math.nan
    strongest = int(np.argmax(intensity))
    ix, iy = np.unravel_index(strongest, heat.system.window_shape)
    window_x, window_y = heat.system.window
    x_centres, y_centres = heat.domain.cell_centers()
    return {'bound_signal_max': float(intensity[strongest]), 'bound_signal_x': float(x_centres[window_x][ix]),
            'bound_signal_y': float(y_centres[window_y][iy]), 'sensing_volume': sensing_volume,
            'particles_in_sensing_volume': monitor.particle['number_density'] * sensing_volume,
            'index_increment': monitor.particle['index_increment'],
            'number_density': monitor.particle['number_density']}


def _contribution_weights(monitor, weight):
    """(names, weights) of the monitor's contributions, plus 'other' for weighted selected cells of no contribution;
    ([], []) without contributions."""
    if not monitor.contributions:
        return [], []
    names = [name for name, _ in monitor.contributions]
    weights = [np.where(mask, weight, 0.0) for _, mask in monitor.contributions]
    assigned = np.any([mask for _, mask in monitor.contributions], axis=0)
    rest = np.where(assigned, 0.0, weight)
    if np.any(rest != 0.0):
        names.append(OTHER_CONTRIBUTION)
        weights.append(rest)
    return names, weights


def _band_integral(frequencies, psd):
    """Trapezoid integral of psd over the solved frequencies in ln f (0 for a single frequency)."""
    if frequencies.size < 2:
        return 0.0
    log_frequencies = np.log(frequencies)
    values = psd * frequencies
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(log_frequencies)))


_CHANNEL_EXTRAS = ('variance', 'static_variance', 'band_fraction','axial_wavenumber', 'mirror_factor', 'axial_norm', 'wavelength',
                   'collected_mode', 'collected_n_eff', 'collected_te_fraction', 'incident_mode', 'bound_signal_max',
                   'bound_signal_x', 'bound_signal_y', 'sensing_volume', 'particles_in_sensing_volume',
                   'index_increment', 'number_density')


def thermal_noise_result(monitor) -> dict:
    """'xy' of a ThermalNoiseMonitor vs frequency [Hz]: per channel (prefix psd, or psd_<tag> with several collected
    modes) the series <prefix> (one-sided, combined), <prefix>_<part> (self term per contribution), <prefix>_cross
    (minus the self terms) and, for particles, <prefix>_single (per particle in the sensing volume), then 10 log10
    *_db of all but the cross terms; extras below (suffix _<tag> with several channels); ValueError before the first
    solve or on colliding series names."""
    if not monitor.channels:
        raise ValueError('not solved yet: the thermal noise spectrum needs a thermal_noise study')
    linear, cross_names, extras = {}, set(), {}
    for channel in monitor.channels:
        tag, psd, parts, info = channel['tag'], channel['psd'], channel['contribution_psd'], channel['info']
        prefix, suffix = (f'psd_{tag}', f'_{tag}') if tag else ('psd', '')
        _add_series(linear, prefix, psd.copy())
        for name, part in parts.items():
            _add_series(linear, f'{prefix}_{name}', part.copy())
        if parts:
            _add_series(linear, f'{prefix}_cross', psd - np.sum(list(parts.values()), axis=0))
            cross_names.add(f'{prefix}_cross')
            extras[f'contribution_variances{suffix}'] = {name: float(value)
                                                         for name, value in info['contribution_variances'].items()}
        if monitor.particle is not None:
            _add_series(linear, f'{prefix}_single', psd / info['particles_in_sensing_volume'])
        extras.update({f'{key}{suffix}': float(info[key]) for key in _CHANNEL_EXTRAS if key in info})
    series = dict(linear)
    with np.errstate(divide='ignore', invalid='ignore'):
        for name, values in linear.items():
            if name not in cross_names:
                _add_series(series, f'{name}_db', 10.0 * np.log10(values))
    if monitor.mode_pair is None:
        unit = 'X^2/Hz'
    else:
        unit = 'W/Hz' if monitor.mode_pair.get('illumination') is not None else '1/Hz'
    return {'kind': 'xy', 'x': monitor.frequencies.copy(), 'x_label': 'Frequency', 'x_unit': 'Hz',
            'y_label': 'Noise spectral density', 'series': series,
            'series_units': {name: 'dB/Hz' if name not in linear else unit for name in series}, **extras}


def _add_series(series, name, values):
    """series[name] = values; ValueError when a material name made two series share a name."""
    if name in series:
        raise ValueError(f"thermal noise series {name!r} exists twice: rename the material that produces it")
    series[name] = values

"""EM MaterialGrid arrays per polarisation (register_em_arrays), the DrudePole / LorentzPole dispersion poles and the
TFSF box outline warnings."""
import math

import numpy as np
from ...core.backend import array_module_of, host_copy, write_host_into

from ...core.constants import DTYPE, EPS0


# Gyration vector of a magneto-optic material (eps_ij += j e_ijk g_k, model axes; model.schema GYRATION_PROPERTY_NAMES):
# polarisation -> the optional MaterialGrid arrays its solver reads (allocated only when a material sets them). hz: g_z
# (eps_xy = eps_xy + j g_z, eps_yx = eps_xy - j g_z) in the operator; modes: g_x / g_y (eps_yz = j g_x, eps_zx = j g_y)
# in the first-order non-reciprocal phase shift of every mode only.
GYRATION_ARRAYS = {'hz': ('gyration_z',), 'modes': ('gyration_x', 'gyration_y')}
GYRATION_Z = 'gyration_z'

# polarisation -> MaterialGrid arrays (default value) an EM solver reads; hz never reads eps_r; 'modes' (em_modes, the
# cross-section solver) reads all three (eps_r = eps_zz).
EM_MATERIAL_SCHEMAS = {
    'ez': {'eps_r': 1.0, 'mu_r': 1.0, 'sigma_e': 0.0, 'sigma_m': 0.0},
    'hz': {'eps_xx': 1.0, 'eps_yy': 1.0, 'mu_r': 1.0, 'sigma_e': 0.0, 'sigma_m': 0.0},
    'modes': {'eps_r': 1.0, 'eps_xx': 1.0, 'eps_yy': 1.0, 'mu_r': 1.0, 'sigma_e': 0.0, 'sigma_m': 0.0},
}


def register_em_arrays(materials, polarization):
    """Register EM_MATERIAL_SCHEMAS[polarization] on a MaterialGrid (existing arrays kept); new hz / modes eps_xx /
    eps_yy start from eps_r when an E_z physics or script registered it."""
    registered = materials.property_names()
    if polarization in ('hz', 'modes') and 'eps_r' in registered:
        for name in ('eps_xx', 'eps_yy'):
            if name not in registered:
                array_module_of(materials.eps_r).copyto(materials.add_property(name, 1.0), materials.eps_r)
    materials.add_properties(EM_MATERIAL_SCHEMAS[polarization])


def material_solution_state(solver, names) -> dict:
    """Snapshot solutions (materials=True): host copies of the MaterialGrid arrays `names` and of the crystallinity of
    the solver's phase_change_media (both written by couplings)."""
    materials = solver.domain.materials
    return {'arrays': {name: host_copy(getattr(materials, name)) for name in names},
            'crystallinity': [host_copy(medium.crystallinity) for medium in getattr(solver, 'phase_change_media', ())]}


def restore_material_solution(solver, state) -> None:
    """Write a material_solution_state back in place, without materials_changed (a snapshot read only renders and the
    live state is written back the same way afterwards)."""
    materials = solver.domain.materials
    for name, values in state['arrays'].items():
        write_host_into(getattr(materials, name), values)
    media = getattr(solver, 'phase_change_media', ())
    if len(media) != len(state['crystallinity']):
        raise RuntimeError(f"{type(solver).__name__} restore_solution: {len(state['crystallinity'])} phase-change media "
                           f'in the state, {len(media)} in the solver (a state of another build?)')
    for medium, values in zip(media, state['crystallinity']):
        write_host_into(medium.crystallinity, values)


def tfsf_outline_warnings(physics, boxes, uniform=False):
    """Warning per TFSF box (x0, x1, y0, y1 total-field cells) with an outline cell (the two cells across each box edge;
    no y edges with full_period_y, none along an open_faces side) that is not vacuum (FDTD) or, uniform=True (FDFD),
    differs from outline cell (x0 - 1, y0): EM_MATERIAL_SCHEMAS values (as the model's _is_vacuum), a pole or PEC (one
    host sync per box)."""
    if not boxes:
        return []
    materials = physics.domain.materials
    grid = physics.domain.grid
    xp = physics.domain.xp
    pole_cells = []
    for pole, box, cropped_mask in physics.dispersive_regions:
        covered = xp.ones((grid.nx, grid.ny), dtype=xp.bool_)
        if box is not None:
            covered[...] = False
            covered[box] = cropped_mask
        pole_cells.append(covered)
    x_centers, y_centers = physics.domain.cell_centers()
    warnings = []
    for source in boxes:
        x0, x1, y0, y1 = source.x0, source.x1, source.y0, source.y1
        full_period_y = getattr(source, 'full_period_y', False)
        row_start = 0 if full_period_y else y0 - 1
        region = (slice(x0 - 1, x1 + 1), slice(row_start, grid.ny if full_period_y else y1 + 1))
        reference = (0, y0 - row_start)   # outline cell (x0 - 1, y0) in the region
        outline = xp.zeros((x1 - x0 + 2, region[1].stop - row_start), dtype=xp.bool_)
        for name, vacuum_value in EM_MATERIAL_SCHEMAS[physics.polarization].items():
            values = getattr(materials, name)[region]
            if uniform:
                target = values[reference]
                tolerance = DTYPE(1e-6) * xp.abs(target) + DTYPE(1e-12)
            else:
                target, tolerance = DTYPE(vacuum_value), DTYPE(1e-6 * vacuum_value + 1e-12)
            outline |= xp.abs(values - target) > tolerance
        for covered in pole_cells:
            covered = covered[region]
            outline |= (covered != covered[reference]) if uniform else covered
        if physics.pec_cells is not None:
            outline |= physics.pec_cells[region]
        if full_period_y:
            outline[2:-2, :] = False
        else:
            outline[2:-2, 2:-2] = False
        open_faces = getattr(source, 'open_faces', ())   # a y side opening onto a mirror plane is no edge
        if 'y_lo' in open_faces:
            outline[2:-2, :2] = False
        if 'y_hi' in open_faces:
            outline[2:-2, -2:] = False
        n_mismatched = int(outline.sum())
        if not n_mismatched:
            continue
        i, j = np.unravel_index(int(xp.argmax(outline)), outline.shape)
        where = f"({x_centers[x0 - 1 + i]:.4g}, {y_centers[row_start + j]:.4g}) m"
        if uniform:
            warnings.append(f"TFSF box cells [{x0}, {x1}) x [{y0}, {y1}): {n_mismatched} cells along its edges differ "
                            f"from the background cell at ({x_centers[x0 - 1]:.4g}, {y_centers[y0]:.4g}) m "
                            f"(permittivity, mu_r, a conductivity, a dispersive pole or PEC), the first at {where}: "
                            f"the incident wave is computed for that background, so incident light leaks into the "
                            f"scattered field; keep the box edges in one uniform medium")
        else:
            warnings.append(f"TFSF box cells [{x0}, {x1}) x [{y0}, {y1}): {n_mismatched} cells along its edges are "
                            f"not vacuum (permittivity, mu_r, a conductivity, a dispersive pole or PEC), the first at "
                            f"{where}: the incident field is computed for vacuum, so incident light leaks into the "
                            f"scattered field; keep the box edges in vacuum")
    return warnings


def _python_complex_for_scalar(chi):
    """A Python complex for a 0-d result, else the complex128 numpy array."""
    return complex(chi) if np.ndim(chi) == 0 else chi


class DrudePole:

    def __init__(self, omega_p: float, gamma: float, eps_inf: float = 1.0):
        self.omega_p = DTYPE(omega_p)
        self.gamma = DTYPE(gamma)
        self.eps_inf = DTYPE(eps_inf)

    def ade_coefficients(self, dt: float):
        """(damping, drive) of the velocity-form step J += damping J + drive E^n: damping = c1 - 1 of the leap-frog J =
        c1 J + c2 E as its own factor (a float32 c1 ~ 1 keeps gamma dt only to ~6e-8 absolute)."""
        dt = float(dt)
        gamma = float(self.gamma)
        wp = float(self.omega_p)
        denom = 1.0 + gamma * dt / 2.0
        damping = DTYPE(-gamma * dt / denom)
        drive = DTYPE((EPS0 * wp ** 2 * dt) / denom)
        return damping, drive

    def susceptibility(self, omega):
        """Continuous-time chi of dJ/dt + gamma J = eps0 wp^2 E in exp(+j omega t): -wp^2/(omega^2 - j gamma omega)
        (Im < 0); the leap-frog FDTD update of both polarisations differs at O((omega dt)^2). omega [rad/s]: float or
        numpy array."""
        omega = np.asarray(omega, dtype=np.float64)
        omega_p = float(self.omega_p)
        gamma = float(self.gamma)
        return _python_complex_for_scalar(-omega_p ** 2 / (omega ** 2 - 1j * gamma * omega))


class LorentzPole:

    def __init__(self, delta_eps: float, omega_0: float, gamma: float):
        self.delta_eps = DTYPE(delta_eps)
        self.omega_0 = DTYPE(omega_0)
        self.gamma = DTYPE(gamma)

    def ade_coefficients(self, dt: float):
        """(damping, restoring, drive) of the velocity-form step dP += damping dP - restoring P^n + drive E^n, P += dP
        (the centred P'' + gamma P' + w0^2 P = eps0 delta_eps w0^2 E; a1 ~ 2 / a2 ~ -1 in float32 lose w0^2 dt^2)."""
        dt = float(dt)
        w0 = float(self.omega_0)
        g = float(self.gamma)
        de = float(self.delta_eps)
        denom = 1.0 + g * dt / 2.0
        damping = DTYPE(-g * dt / denom)
        restoring = DTYPE(w0 ** 2 * dt ** 2 / denom)
        drive = DTYPE(EPS0 * de * w0 ** 2 * dt ** 2 / denom)
        return damping, restoring, drive

    def susceptibility(self, omega):
        """Continuous-time chi of P'' + gamma P' + w0^2 P = eps0 delta_eps w0^2 E in exp(+j omega t): delta_eps w0^2 /
        (w0^2 - omega^2 + j gamma omega) (Im < 0); the centred FDTD update differs at O((omega dt)^2). omega [rad/s]."""
        omega = np.asarray(omega, dtype=np.float64)
        omega_0 = float(self.omega_0)
        gamma = float(self.gamma)
        delta_eps = float(self.delta_eps)
        return _python_complex_for_scalar(delta_eps * omega_0 ** 2 / (omega_0 ** 2 - omega ** 2 + 1j * gamma * omega))


def _pole_step_term(pole, dt) -> float:
    """Pole term of the leap-frog bound (z = -1 mode, independent of gamma and sigma_e): Drude (omega_p dt)^2 / 4,
    Lorentz delta_eps (omega_0 dt)^2 / (4 - (omega_0 dt)^2); ValueError when omega_0 dt >= 2. Shared by the 2D, BOR and
    3D FDTD engines."""
    if isinstance(pole, DrudePole):
        return (float(pole.omega_p) * dt) ** 2 / 4.0
    resonance_step_squared = (float(pole.omega_0) * dt) ** 2
    if resonance_step_squared >= 4.0:
        raise ValueError(f'{_pole_label(pole)}: omega_0 dt = {math.sqrt(resonance_step_squared):.4g} >= 2, so the '
                         f'Lorentz update is unstable; lower courant_factor or refine the mesh')
    return float(pole.delta_eps) * resonance_step_squared / (4.0 - resonance_step_squared)


def _pole_label(pole) -> str:
    if isinstance(pole, DrudePole):
        return f'Drude pole omega_p = {float(pole.omega_p):.4g} rad/s'
    return f'Lorentz pole delta_eps = {float(pole.delta_eps):.4g}, omega_0 = {float(pole.omega_0):.4g} rad/s'

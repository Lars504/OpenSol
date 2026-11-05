"""EM MaterialGrid arrays per polarisation (register_em_arrays), the DrudePole / LorentzPole dispersion poles and the
TFSF box outline warnings."""
import numpy as np
from ...core.backend import array_module_of

from ...core.constants import DTYPE, EPS0


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

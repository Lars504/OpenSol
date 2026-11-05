"""Time-averaged optical (Maxwell-stress) force density of an em_fdfd_ez solution on the solid: f = -(eps0 / 4) |E_z|^2
grad eps_r (peak phasor, non-magnetic, no electrostriction), integrated per cell face and handed to the solid cells."""
from ...core.backend import array_module_of

from ...core.constants import DTYPE, EPS0
from .optomechanics import face_jump, face_mean


class OpticalForceDensity:
    """Full-grid body_force_x / _y [N/m^3] buffers of the em_solid_optical_force coupling, refreshed once per EM solve:
    each face carries -(eps0 / 4) |E_z|^2 (eps_r jump) times its length, split between its solid cells (the solver's
    c11 > 0; equally when both are solid, dropped when neither is) and divided by their area. Arrays on domain.xp."""

    def __init__(self, domain):
        xp = domain.xp
        dx_cells, dy_cells = domain.cell_widths()
        self._dx = xp.asarray(dx_cells, dtype=DTYPE)[:, None]
        self._dy = xp.asarray(dy_cells, dtype=DTYPE)[None, :]
        grid_shape = (domain.grid.nx, domain.grid.ny)
        self.force_x = xp.zeros(grid_shape, dtype=DTYPE)
        self.force_y = xp.zeros(grid_shape, dtype=DTYPE)
        self._revision = None

    def update(self, em, solid):
        """Recompute from the last EM solve (skipped while em.n_solves is unchanged)."""
        revision = getattr(em, 'n_solves', None)
        if revision is not None and revision == self._revision:
            return
        state = em.state()
        intensity = state['ez_re'] ** 2 + state['ez_im'] ** 2
        eps_r = state['eps_r']
        solid_cells = _solid_cells(solid.domain.materials)
        # +x force per unit depth [N/m] of each x face (between cells i and i + 1), likewise +y of each y face:
        # (eps0 / 4) mean |E_z|^2 (eps_lo - eps_hi) times the face length (the moving-boundary face rule of optomechanics)
        face_x = (0.25 * EPS0) * face_mean(intensity, 0) * face_jump(eps_r, 0) * self._dy
        face_y = (0.25 * EPS0) * face_mean(intensity, 1) * face_jump(eps_r, 1) * self._dx
        self._spread(face_x, solid_cells[:-1, :], solid_cells[1:, :], self.force_x, axis=0)
        self._spread(face_y, solid_cells[:, :-1], solid_cells[:, 1:], self.force_y, axis=1)
        cell_area = self._dx * self._dy
        self.force_x /= cell_area
        self.force_y /= cell_area
        self._revision = revision

    @staticmethod
    def _spread(face_force, solid_lower, solid_upper, out, axis):
        """Split each face force between its solid cells into out (force per cell, overwritten)."""
        solid_count = solid_lower + solid_upper
        share = face_force / array_module_of(out).where(solid_count > 0, solid_count, DTYPE(1.0))
        out.fill(0)
        if axis == 0:
            out[:-1, :] += share * solid_lower
            out[1:, :] += share * solid_upper
        else:
            out[:, :-1] += share * solid_lower
            out[:, 1:] += share * solid_upper


def _solid_cells(materials):
    """Full-grid DTYPE mask of the solid's cells as stiffness.element_stiffness decides them: c11 > 0 where a finite
    c11 entry is set, else youngs_modulus > 0 (the isotropic c11 = lambda + 2 mu)."""
    solid = materials.youngs_modulus > 0
    if 'c11' in set(materials.property_names()):
        c11 = materials.c11
        xp = array_module_of(c11)
        solid = xp.where(xp.isfinite(c11), c11 > 0, solid)
    return solid.astype(DTYPE)


def make_optical_force_transforms(domain):
    """(transform_x, transform_y) of the two CouplingTerms into body_force_x / body_force_y, sharing one buffer pair."""
    density = OpticalForceDensity(domain)

    def force_x(ez_re, em, solid):
        density.update(em, solid)
        return density.force_x

    def force_y(ez_re, em, solid):
        density.update(em, solid)
        return density.force_y
    return force_x, force_y

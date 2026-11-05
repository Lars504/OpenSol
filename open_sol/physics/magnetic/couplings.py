"""Transforms of the magnetic couplings (coupling/definitions_magnetic.py): induction heating q_eddy + q_magnetic into
heat volumetric_power, and the Lorentz force J x B (planar) into solid_mechanics body_force_x / _y; full grid float32
on the magnetic fields' backend (array_module_of)."""
from ...core.backend import array_module_of

from ...core.constants import DTYPE

HEAT_SLOT = 'volumetric_power'
FORCE_SLOTS = ('body_force_x', 'body_force_y')


class InductionHeatSource:
    """CouplingTerm transform (q_eddy, magnetic_fields_ac, heat) -> q_eddy + q_magnetic [W/m^3] in a persistent
    buffer (time averages of the last AC solve)."""

    def __init__(self):
        self.buffer = None

    def __call__(self, q_eddy, magnetic, heat):
        xp = array_module_of(q_eddy)
        if self.buffer is None:
            self.buffer = xp.empty_like(q_eddy, dtype=DTYPE)
        xp.add(q_eddy, magnetic.fields.q_magnetic, out=self.buffer)
        return self.buffer


class LorentzForceDensity:
    """Planar body force f = J_z z x B [N/m^3]: f_x = -J B_y, f_y = J B_x (magnetostatics J = J_coupled + J_applied);
    magnetic_fields_ac the time average Re(J B*) / 2. Recomputed once per magnetic solve into two buffers."""

    def __init__(self):
        self.force_x = None
        self.force_y = None
        self._revision = None

    def update(self, magnetic):
        """Recompute from the last magnetic solve (skipped while magnetic.n_solves is unchanged)."""
        revision = magnetic.n_solves
        if self.force_x is not None and revision == self._revision:
            return
        if magnetic.axisymmetric:
            raise ValueError(f"magnetic physics '{magnetic.name}' is axisymmetric: the Lorentz force has no plane-strain "
                             'image (solid_mechanics is planar)')
        fields = magnetic.fields
        if self.force_x is None:
            xp = array_module_of(fields.B_abs)
            self.force_x = xp.zeros(fields.B_abs.shape, dtype=DTYPE)
            self.force_y = xp.zeros(fields.B_abs.shape, dtype=DTYPE)
        if hasattr(fields, 'J_total_re'):   # magnetic_fields_ac: peak phasors, time average 1/2 Re(J B*)
            self.force_x[...] = -0.5 * (fields.J_total_re * fields.B_y_re + fields.J_total_im * fields.B_y_im)
            self.force_y[...] = 0.5 * (fields.J_total_re * fields.B_x_re + fields.J_total_im * fields.B_x_im)
        else:
            density = fields.J_coupled + fields.J_applied
            self.force_x[...] = -density * fields.B_y
            self.force_y[...] = density * fields.B_x
        self._revision = revision


def make_lorentz_force_transforms():
    """(transform_x, transform_y) of the two CouplingTerms into body_force_x / body_force_y, sharing one density."""
    density = LorentzForceDensity()

    def force_x(B_abs, magnetic, solid):
        density.update(magnetic)
        return density.force_x

    def force_y(B_abs, magnetic, solid):
        density.update(magnetic)
        return density.force_y
    return force_x, force_y

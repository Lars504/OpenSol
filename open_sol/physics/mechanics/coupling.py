"""Heat -> solid mechanics transfer: the heat temperature rise T - T0 into the solid's temperature_rise slot (T0 of the
heat physics is the stress-free temperature)."""
from ...core.constants import DTYPE


def make_temperature_rise_transform(grid_shape, xp):
    """transform(T, heat, solid) of a CouplingTerm: heat.temperature_rise() (solve precision, window) into one
    persistent full-grid float32 buffer of array module xp (the run backend's; PML ring 0: the solid solves on the
    window only); a heat on a triangle / tetrahedron mesh (no window) gives one value per element, copied whole."""
    temperature_rise = xp.zeros(grid_shape, dtype=DTYPE)

    def transform(T, heat, solid=None):
        window = getattr(heat, 'window', None)
        if window is None:
            temperature_rise[...] = heat.temperature_rise()
        else:
            temperature_rise[window] = heat.temperature_rise()
        return temperature_rise
    return transform

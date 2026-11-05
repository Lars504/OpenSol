"""Mesh error indicators of the S7.7 refinement levels (host numpy, no cupy): the default interpolation-error
indicator eta^2 = sum_axis (h_axis^2 d2u / d axis^2)^2 V on graded tensor cells, split per axis; the A2 physics
indicators: Kelly face jumps of cell-centred fluxes / tractions and the energy-weighted interpolation of EM fields."""
import numpy as np


def graded_second_difference(values, widths, axis):
    """d2 values / d axis^2 at the cell centres of graded widths (float64, values' shape); the two end cells copy their
    neighbour's value, an axis of fewer than three cells gives 0."""
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[axis]
    if count < 3:
        return np.zeros_like(values)
    widths = np.asarray(widths, dtype=np.float64)
    gaps = 0.5 * (widths[:-1] + widths[1:])                     # centre-to-centre distances (count - 1,)
    slopes = np.diff(values, axis=axis) / _along(gaps, axis, values.ndim)
    spans = gaps[:-1] + gaps[1:]                                 # c[i + 1] - c[i - 1] of the interior cells
    interior = 2.0 * np.diff(slopes, axis=axis) / _along(spans, axis, values.ndim)
    first = np.take(interior, [0], axis=axis)
    last = np.take(interior, [-1], axis=axis)
    return np.concatenate([first, interior, last], axis=axis)


def interpolation_indicator(values, widths):
    """Per-axis eta^2 parts (n_axes, *values.shape) of one window field on cells of 1D widths per axis:
    (h_a^2 d2u / da^2)^2 V with V the cell volume (product of the widths); non-finite values count as 0."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    widths = [np.asarray(axis_widths, dtype=np.float64) for axis_widths in widths]
    if len(widths) != values.ndim or any(axis_widths.size != size for axis_widths, size in zip(widths, values.shape)):
        raise ValueError(f'interpolation_indicator: widths {[axis_widths.size for axis_widths in widths]} do not '
                         f'match the field shape {values.shape}')
    volume = widths[0]
    for axis_widths in widths[1:]:
        volume = np.multiply.outer(volume, axis_widths)
    parts = np.empty((values.ndim,) + values.shape)
    for axis, axis_widths in enumerate(widths):
        scaled = _along(axis_widths ** 2, axis, values.ndim) * graded_second_difference(values, axis_widths, axis)
        parts[axis] = scaled ** 2 * volume
    return parts


def relative_field(values):
    """values / max |values| over the finite cells (float64; all zeros when that maximum is 0 or none is finite), so
    fields of different units add in one indicator."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    scale = float(np.max(np.abs(values[finite]))) if finite.any() else 0.0
    return np.zeros_like(values) if scale == 0.0 else values / scale


def face_jump_indicator(x_normal, y_normal, coupled_x, coupled_y, x_length, y_length, x_area, y_area,
                        x_compliance, y_compliance):
    """Kelly eta^2 parts (2, nx, ny): the jump J of the normal components (tuples of (nx, ny) arrays, a traction has
    two) across each coupled interior x- (y-) face gives both cells 0.5 w |J|^2 A C (normal width, face area, 1 / c)."""
    shape = np.shape(x_normal[0])
    parts = np.zeros((2,) + tuple(shape))
    for axis, (normal, coupled, length, area, cell_compliance) in enumerate(
            ((x_normal, coupled_x, x_length, x_area, x_compliance),
             (y_normal, coupled_y, y_length, y_area, y_compliance))):
        lower = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        upper = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        jump_squared = sum(np.abs(np.diff(np.asarray(component), axis=axis)) ** 2 for component in normal)
        face_term = np.where(np.asarray(coupled, dtype=bool), jump_squared * np.asarray(area, dtype=np.float64), 0.0)
        cell_weight = 0.5 * np.broadcast_to(np.asarray(length, dtype=np.float64), shape) * np.broadcast_to(
            np.asarray(cell_compliance, dtype=np.float64), shape)
        parts[axis][lower] += cell_weight[lower] * face_term
        parts[axis][upper] += cell_weight[upper] * face_term
    return np.nan_to_num(parts, nan=0.0, posinf=0.0, neginf=0.0)


def compliance(coefficient):
    """1 / coefficient where it is > 0, else 0 (float64): the energy-norm weight of face_jump_indicator."""
    coefficient = np.asarray(coefficient, dtype=np.float64)
    positive = coefficient > 0.0
    return np.where(positive, 1.0 / np.where(positive, coefficient, 1.0), 0.0)


def energy_interpolation_indicator(components, widths):
    """Per-axis eta^2 parts (n_axes, *shape) of field components weighted by their energy density factor: the sum over
    (values, weight) of weight x interpolation_indicator of the real and imaginary parts (weight a scalar or
    values-shaped, e.g. eps0 |eps| for E, mu0 for H)."""
    parts = None
    for values, weight in components:
        values = np.asarray(values)
        own = interpolation_indicator(values.real, widths)
        if np.iscomplexobj(values):
            own += interpolation_indicator(values.imag, widths)
        own *= np.asarray(weight, dtype=np.float64)
        parts = own if parts is None else parts + own
    if parts is None:
        raise ValueError('energy_interpolation_indicator: no field components')
    return parts


def _along(vector, axis, ndim):
    """1D vector shaped to broadcast along `axis` of an ndim array."""
    shape = [1] * ndim
    shape[axis] = vector.size
    return vector.reshape(shape)

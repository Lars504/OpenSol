"""Anhysteretic B-H curves of magnetostatics stage 4 (array-module agnostic, no cupy import): the Frohlich-Kennelly
curve B = mu0 H + J_s a H / (1 + a H), a = mu0 (mu_dc - 1) / J_s, with its closed-form secant reluctivity nu(|B|)
(and the secant permeability of |H| for magnetostatics_3d), and
the tabulated piecewise-linear bh_curve with its secant reluctivity |H| / B(|H|) (forward interpolation). Stage 5
(triangles, Newton): both curves B-indexed with the secant nu, differential nu_d = d|H| / d|B| and energy density
w(|B|) = int_0^B H dB (*_curve_terms)."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import MU0

# setting nonlinear_method: 'auto' = 'anderson' on tensor meshes, 'newton' on triangles; Anderson-mixed or damped
# Picard on ln nu (tensor meshes), exact-tangent Newton with an energy line search (triangles)
NONLINEAR_METHODS = ('auto', 'anderson', 'picard', 'newton')
TENSOR_NONLINEAR_METHODS = ('anderson', 'picard')
TRIANGLE_NONLINEAR_METHODS = ('newton',)
ANDERSON_DEPTH = 5                           # iterates mixed by 'anderson'
_LOG_SERIES_LIMIT = 1e-3                     # x below which x - ln(1 + x) is summed as its series (no cancellation)
# float32 tensor loop: proposals in a row that do not halve the smallest change so far before the loop ends unconverged
FLOAT32_STALL_ITERATIONS = 20


class Float32StallWatch:
    """Ends a single-precision saturation loop whose change stopped falling: the float32 CG stalls near a relative
    residual of ~1e-6, which leaves |B| of high-permeability cells jumping by ~1e-2 per solve (bh example)."""

    def __init__(self):
        self.smallest_change = math.inf
        self.iterations_since = 0
        self.last_change = math.nan
        self.stalled = False

    def check(self, proposal, state) -> bool:
        """True once FLOAT32_STALL_ITERATIONS proposals in a row missed half the smallest change (one sync)."""
        change = float(abs(proposal - state).max())
        self.last_change = change
        if change <= 0.5 * self.smallest_change:
            self.smallest_change, self.iterations_since = change, 0
        else:
            self.iterations_since += 1
        self.stalled = self.iterations_since >= FLOAT32_STALL_ITERATIONS
        return self.stalled

    def record_stop(self, picard) -> None:
        """Count the stopped loop as unconverged on picard (its run returned converged on the repeated state)."""
        picard.n_unconverged += 1
        picard.last_converged = False
        picard.last_warning = (
            f'{picard.label()}: stopped after {picard.last_iterations} iterations at single precision: the change '
            f'({self.last_change:.2e}, tolerance {picard.settings.tolerance:.1e}) did not halve in '
            f'{FLOAT32_STALL_ITERATIONS} iterations; the float32 CG stalls near a relative residual of ~1e-6, which '
            'leaves B in high-permeability cells off by up to percents: use a double or mixed linear solver')


def saturation_reluctivity(flux_density, relative_permeability, saturation_polarization, xp):
    """nu = H / B [m/H] at |B| = flux_density (>= 0) of the curve with initial mu_dc and J_s [T]; the linear 1 / (mu0
    mu_dc) where J_s <= 0 or mu_dc <= 1 (rationalised root: no cancellation at small B, exact limit at B = 0)."""
    saturating = (saturation_polarization > 0.0) & (relative_permeability > 1.0)
    polarization = xp.where(saturating, saturation_polarization, 1.0)
    # a [m/A]; 0 off the curve so the discarded branch stays finite (mu_dc < 1 would give a negative sqrt argument)
    slope = xp.where(saturating, MU0 * (relative_permeability - 1.0) / polarization, 0.0)
    linear_term = MU0 + slope * (polarization - flux_density)       # b of mu0 a H^2 + b H - B = 0
    secant = 2.0 / (linear_term + xp.sqrt(linear_term * linear_term + 4.0 * MU0 * slope * flux_density))
    return xp.where(saturating, secant, 1.0 / (MU0 * relative_permeability))


def saturation_permeability(field_strength, relative_permeability, saturation_polarization, xp):
    """Secant relative permeability B / (mu0 H) = 1 + (mu_dc - 1) / (1 + a H) at |H| = field_strength (>= 0) of the
    same curve (H-based, magnetostatics_3d); mu_dc where J_s <= 0 or mu_dc <= 1."""
    saturating = (saturation_polarization > 0.0) & (relative_permeability > 1.0)
    polarization = xp.where(saturating, saturation_polarization, 1.0)
    slope = xp.where(saturating, MU0 * (relative_permeability - 1.0) / polarization, 0.0)   # a [m/A]
    return xp.where(saturating, 1.0 + (relative_permeability - 1.0) / (1.0 + slope * field_strength),
                    relative_permeability)


def _x_minus_log1p(x, xp):
    """x - ln(1 + x) for x >= 0 without cancellation at small x (series below _LOG_SERIES_LIMIT)."""
    series = x * x * (0.5 - x * (1.0 / 3.0 - x * (0.25 - 0.2 * x)))
    return xp.where(x < _LOG_SERIES_LIMIT, series, x - xp.log1p(xp.where(x < _LOG_SERIES_LIMIT, 0.0, x)))


def saturation_curve_terms(flux_density, relative_permeability, saturation_polarization, xp):
    """(secant nu = H / B, differential nu_d = dH / dB [m/H], energy density w = int_0^B H dB [J/m^3]) at |B| =
    flux_density (>= 0) of the J_s curve (initial mu_dc); linear 1 / (mu0 mu_dc) and nu B^2 / 2 where J_s <= 0 or
    mu_dc <= 1. w = B H - mu0 H^2 / 2 - J_s (a H - ln(1 + a H)) / a at H = nu B."""
    secant = saturation_reluctivity(flux_density, relative_permeability, saturation_polarization, xp)
    saturating = (saturation_polarization > 0.0) & (relative_permeability > 1.0)
    polarization = xp.where(saturating, saturation_polarization, 1.0)
    slope = xp.where(saturating, MU0 * (relative_permeability - 1.0) / polarization, 1.0)   # a [m/A]
    field = secant * flux_density
    scaled = slope * field                                                                  # a H >= 0
    differential = xp.where(saturating, 1.0 / (MU0 + polarization * slope / (1.0 + scaled) ** 2), secant)
    energy = xp.where(saturating,
                      flux_density * field - 0.5 * MU0 * field * field
                      - polarization * _x_minus_log1p(scaled, xp) / slope,
                      0.5 * secant * flux_density * flux_density)
    return secant, differential, energy


@dataclass(frozen=True, eq=False)
class BHCurve:
    """Piecewise-linear anhysteretic B-H table from the origin: strictly increasing float64 H [A/m] / B [T] nodes (xp
    arrays, origin first), the last node and the first segment's secant H_1 / B_1 [m/H] (the |B| -> 0 limit); per
    segment the slope dH / dB [m/H] and the energy int_0^B_k H dB [J/m^3] at each node (B-indexed terms)."""
    field_strength: object
    flux_density: object
    last_field_strength: float
    last_flux_density: float
    initial_reluctivity: float
    segment_slopes: object
    node_energies: object


def bh_curve(rows, xp=np):
    """BHCurve of resolved bh_curve rows [(H, B), ...]: an optional origin row (0, 0) first, then finite H, B > 0,
    both strictly increasing (the origin is prepended); ValueError naming the first offending row otherwise."""
    rows = [(float(row[0]), float(row[1])) for row in rows]
    first = 1 if rows and rows[0] == (0.0, 0.0) else 0
    if len(rows) <= first:
        raise ValueError('bh_curve needs at least one row (H, B) besides the origin (0, 0)')
    previous = (0.0, 0.0)
    for index in range(first, len(rows)):
        field_strength, flux_density = rows[index]
        if not (math.isfinite(field_strength) and math.isfinite(flux_density)):
            raise ValueError(f'bh_curve row {index + 1}: H and B must be finite, got {rows[index]}')
        if not (field_strength > previous[0] and flux_density > previous[1]):
            raise ValueError(f'bh_curve row {index + 1} {rows[index]}: H and B must both increase strictly from '
                             f'{previous} (the curve starts at the origin; only a first row may be (0, 0))')
        previous = rows[index]
    nodes = np.array([(0.0, 0.0)] + rows[first:], dtype=np.float64)
    field_steps, flux_steps = np.diff(nodes[:, 0]), np.diff(nodes[:, 1])
    node_energies = np.concatenate(([0.0], np.cumsum(0.5 * (nodes[:-1, 0] + nodes[1:, 0]) * flux_steps)))
    return BHCurve(xp.asarray(nodes[:, 0]), xp.asarray(nodes[:, 1]), float(nodes[-1, 0]), float(nodes[-1, 1]),
                   float(nodes[1, 0] / nodes[1, 1]), xp.asarray(field_steps / flux_steps), xp.asarray(node_energies))


def tabulated_reluctivity(field_strength, curve, xp):
    """nu = |H| / B(|H|) [m/H] at |H| = field_strength (>= 0, float64) of a BHCurve (its arrays in xp): B interpolated
    linearly between the nodes, beyond the last node B_n + mu0 (H - H_n), at H = 0 the first segment's secant."""
    flux_density = xp.interp(field_strength, curve.field_strength, curve.flux_density)
    beyond = field_strength > curve.last_field_strength
    flux_density = xp.where(beyond, curve.last_flux_density + MU0 * (field_strength - curve.last_field_strength),
                            flux_density)
    positive = field_strength > 0.0
    return xp.where(positive, field_strength / xp.where(positive, flux_density, 1.0), curve.initial_reluctivity)


def tabulated_curve_terms(flux_density, curve, xp):
    """(secant nu, differential nu_d [m/H], energy density w [J/m^3]) at |B| = flux_density (>= 0, float64) of a
    BHCurve, B-indexed: H(B) linear between the nodes (nu_d = the segment slope, w quadratic per segment), beyond the
    last node H_n + (B - B_n) / mu0; at B = 0 nu = the first segment's secant. The slope jumps at the nodes (w is C1)."""
    nodes_b, nodes_h = curve.flux_density, curve.field_strength
    beyond = flux_density > curve.last_flux_density
    segment = xp.clip(xp.searchsorted(nodes_b, flux_density, side='right') - 1, 0, nodes_b.shape[0] - 2)
    step = flux_density - nodes_b[segment]
    slope = curve.segment_slopes[segment]
    field = nodes_h[segment] + slope * step
    energy = curve.node_energies[segment] + step * (nodes_h[segment] + 0.5 * slope * step)
    excess = flux_density - curve.last_flux_density
    field = xp.where(beyond, curve.last_field_strength + excess / MU0, field)
    energy = xp.where(beyond, curve.node_energies[-1] + excess * (curve.last_field_strength + 0.5 * excess / MU0),
                      energy)
    differential = xp.where(beyond, 1.0 / MU0, slope)
    positive = flux_density > 0.0
    secant = xp.where(positive, field / xp.where(positive, flux_density, 1.0), curve.initial_reluctivity)
    return secant, differential, energy


def refuse_curve_conflicts(label, item, curve_and_polarization=False, saturating_remanence=False) -> None:
    """The saturation refusals shared by every magnetostatics solver (label: the physics, item: 'cell' / 'element'):
    ValueError for a B-H curve and saturation_polarization > 0 in one item, NotImplementedError for a saturating item
    with a remanence."""
    article = 'an' if item[0] in 'aeiou' else 'a'
    if curve_and_polarization:
        raise ValueError(f'{label}: {article} {item} has both a B-H curve and saturation_polarization > 0 (give one '
                         'curve per material)')
    if saturating_remanence:
        raise NotImplementedError(f'{label}: {article} {item} sets both a saturation (saturation_polarization or '
                                  'bh_curve) and a remanence (a saturating permanent magnet is not modelled)')

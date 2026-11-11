"""Physics-neutral non-uniform meshing with geometry-aware refinement (cell spacings for any solver)."""

import numpy as np
from dataclasses import dataclass
from typing import List

from .constants import DTYPE, FACES
from .grid import DEFAULT_COURANT_FACTOR, courant_time_step


_MAX_PROFILE_SAMPLES = 2_000_000
UNIFORM_SPACING_RTOL = 1e-6
SNAP_MERGE_FRACTION = 0.25   # rectangle edges closer than this x dx_target to a kept snap / the window edge get no node


def has_uniform_spacing(cell_widths, rel_tol=UNIFORM_SPACING_RTOL):
    """True when max - min cell width <= rel_tol * max width (float64 widths); False when empty."""
    widths = np.asarray(cell_widths, dtype=np.float64)
    return widths.size > 0 and float(widths.max() - widths.min()) <= rel_tol * float(widths.max())


@dataclass
class MeshRegion:
    x0: float
    y0: float
    x1: float
    y1: float
    dx_target: float


@dataclass
class MeshCircle:
    cx: float
    cy: float
    radius: float
    dx_target: float
    margin: float = 0.0


@dataclass
class MeshLine:
    axis: str
    position: float
    dx_target: float
    width: float


@dataclass
class MeshInterval:
    """Band [lo, hi] along one axis refined to dx_target (only that axis: an anisotropic refinement)."""
    lo: float
    hi: float
    dx_target: float


SEGMENT_PROFILES = ('uniform', 'geometric', 'tanh')


class MeshConfig:
    """Graded rectilinear mesh: xy refinements (regions, circles, lines), per-axis bands (z needs size_z), optional
    outer grading beyond the refined core and custom edge lists that replace an axis' march."""

    def __init__(self, size_x: float, size_y: float, dx_base: float, max_ratio: float = 1.15, size_z: float = None,
                 axis_max_ratios=None, outer_dx: float = None, outer_ratio: float = None):
        if not (size_x > 0 and size_y > 0 and dx_base > 0) or (size_z is not None and not size_z > 0):
            raise ValueError(f"MeshConfig needs positive sizes and dx_base, got "
                             f"size=({size_x}, {size_y}, {size_z}), dx_base={dx_base}")
        axis_max_ratios = dict(axis_max_ratios or {})
        named_ratios = [('max_ratio', max_ratio)] + [(f'max_ratio {axis}', r) for axis, r in axis_max_ratios.items()]
        for name, ratio in named_ratios:
            if not ratio >= 1.0:
                raise ValueError(f"MeshConfig {name} must be >= 1, got {ratio}")
        if set(axis_max_ratios) - {'x', 'y', 'z'}:
            raise ValueError(f"MeshConfig axis_max_ratios keys must be 'x', 'y' or 'z', got {sorted(axis_max_ratios)}")
        if outer_dx is not None and not outer_dx > 0:
            raise ValueError(f"MeshConfig outer_dx must be > 0, got {outer_dx}")
        if outer_ratio is not None and not outer_ratio > 1.0:
            raise ValueError(f"MeshConfig outer_ratio must be > 1, got {outer_ratio}")
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z    # None = 2D (no z axis)
        self.dx_base = dx_base
        self.max_ratio = max_ratio            # default of every axis (axis_max_ratios override)
        self.axis_max_ratios = axis_max_ratios
        self.outer_dx = outer_dx              # None: no outer grading
        self.outer_ratio = outer_ratio        # None: the axis max_ratio

        self._regions: List[MeshRegion] = []
        self._circles: List[MeshCircle] = []
        self._lines: List[MeshLine] = []
        self._bands = {'x': [], 'y': [], 'z': []}         # MeshInterval per axis
        self._custom_edges = {}                           # axis -> float64 physical edges (0 .. length)
        self._snap_edges = {'x': [], 'y': [], 'z': []}   # (position, dx_target) of object faces that become cell edges

    def ratio(self, axis):
        """Largest adjacent cell ratio of the graded march along `axis`."""
        return float(self.axis_max_ratios.get(axis, self.max_ratio))

    def axis_outer_ratio(self, axis):
        """Cell growth ratio of the outer grading along `axis` (outer_ratio, else the axis max_ratio)."""
        return float(self.ratio(axis) if self.outer_ratio is None else self.outer_ratio)

    def add_refinement_region(self, x0, y0, x1, y1, dx_target):
        """Refine the box spanned by corners (x0, y0), (x1, y1) (any order) to dx_target."""
        _check_target(dx_target)
        self._regions.append(MeshRegion(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), dx_target))

    def add_refinement_circle(self, cx, cy, radius, dx_target, margin=0.0):
        _check_target(dx_target)
        self._circles.append(MeshCircle(cx, cy, radius, dx_target, margin))

    def add_refinement_line(self, axis, position, dx_target, width):
        _check_target(dx_target)
        if axis not in ('x', 'y'):
            raise ValueError(f"refinement line axis must be 'x' or 'y', got {axis!r}")
        self._lines.append(MeshLine(axis, position, dx_target, width))

    def add_refinement_for_shape(self, shape_type, dx_target, margin=None, **kwargs):
        if margin is None:
            margin = 3.0 * dx_target

        if shape_type == 'circle':
            self.add_refinement_circle(
                kwargs['cx'], kwargs['cy'], kwargs['radius'], dx_target, margin
            )
        elif shape_type == 'rectangle':
            x0, x1 = sorted((kwargs['x0'], kwargs['x1']))
            y0, y1 = sorted((kwargs['y0'], kwargs['y1']))
            self.add_refinement_region(
                x0 - margin, y0 - margin, x1 + margin, y1 + margin, dx_target)
            for axis, edges in (('x', (x0, x1)), ('y', (y0, y1))):   # snapped to cell edges (`snap_positions`)
                self._snap_edges[axis] += [(float(edge), float(dx_target)) for edge in edges]
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

    def add_refinement_band(self, axis, lo, hi, dx_target, snap_edges=()):
        """Refine the band [lo, hi] (any order) along `axis` only to dx_target; snap_edges (object faces) become cell
        edges of that axis. 'z' needs size_z."""
        _check_target(dx_target)
        self._check_axis(axis)
        self._bands[axis].append(MeshInterval(min(lo, hi), max(lo, hi), dx_target))
        self._snap_edges[axis] += [(float(edge), float(dx_target)) for edge in snap_edges]

    def add_snap_edges(self, axis, positions, dx_target):
        """Make positions (object faces) cell edges of `axis` without refining (merge rule of snap_positions at
        dx_target)."""
        _check_target(dx_target)
        self._check_axis(axis)
        self._snap_edges[axis] += [(float(position), float(dx_target)) for position in positions]

    def window_edges(self, axis):
        """Float64 physical edges 0 .. length of `axis`: custom edges, or the graded march with every snap position
        and both ends exact (the float32 widths summed in float64)."""
        self._check_axis(axis)
        if axis in self._custom_edges:
            return self._custom_edges[axis].copy()
        length = float(self._axis_length(axis))
        edges = np.concatenate([[0.0], np.cumsum(self._build_1d_mesh(length, axis).astype(np.float64))])
        edges *= length / edges[-1]
        for position in self.snap_positions(axis):
            edges[int(np.argmin(np.abs(edges - position)))] = position
        edges[0], edges[-1] = 0.0, length
        return edges

    def set_axis_edges(self, axis, edges):
        """Custom cell edges of `axis` (physical frame, increasing); 0 and the axis length are added when missing,
        edges within 1e-9 x length of them or of each other merge. Replaces the graded march of that axis."""
        self._check_axis(axis)
        self._custom_edges[axis] = complete_axis_edges(edges, float(self._axis_length(axis)), axis)

    def custom_axes(self):
        """Axes whose cells come from set_axis_edges."""
        return tuple(self._custom_edges)

    def _check_axis(self, axis):
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"mesh axis must be 'x', 'y' or 'z', got {axis!r}")
        if axis == 'z' and self.size_z is None:
            raise ValueError('a z band or z edges need a MeshConfig with size_z (3D)')

    def _axis_length(self, axis):
        return {'x': self.size_x, 'y': self.size_y, 'z': self.size_z}[axis]

    def snap_positions(self, axis):
        """Sorted snap positions (rectangle edges, band snap faces) along `axis` that become cell edges:
        inside the window by >= SNAP_MERGE_FRACTION x dx_target and that far from the previous kept one (x the finer
        of the two targets), else dropped; none on a custom axis."""
        if axis in self._custom_edges:
            return []
        length = self._axis_length(axis)
        kept, kept_dx = [], []
        for position, dx_target in sorted(self._snap_edges[axis]):
            reach = SNAP_MERGE_FRACTION * dx_target   # an edge at or beyond the window boundary is no interface
            if position < reach or position > length - reach:
                continue
            if kept and position - kept[-1] < SNAP_MERGE_FRACTION * min(dx_target, kept_dx[-1]):
                continue
            kept.append(position)
            kept_dx.append(dx_target)
        return kept

    def _finest_target(self, axis):
        targets = [self.dx_base] + [band.dx_target for band in self._bands[axis]]
        if axis == 'z':
            return min(targets)
        targets += [r.dx_target for r in self._regions]
        targets += [c.dx_target for c in self._circles]
        targets += [line.dx_target for line in self._lines if line.axis == axis]
        return min(targets)

    def outer_core(self, axis):
        """(lo, hi) of the refined core along `axis` (span of every refinement interval, clipped to the axis) beyond
        which the outer grading grows the cells; None without outer grading or refinements on that axis."""
        if self.outer_dx is None or not self.outer_dx > self.dx_base:
            return None
        positions = self._refinement_positions(axis)
        if positions.size == 0:
            return None
        length = float(self._axis_length(axis))
        lo, hi = max(float(positions.min()), 0.0), min(float(positions.max()), length)
        return (lo, hi) if hi > lo else None

    def _target_dx_profile(self, positions, axis):
        """Target cell size at each sorted 1D sample position along `axis` (outer_dx outside the outer core, else
        dx_base); each refinement touches only the samples in its closed interval (found by searchsorted)."""
        target = np.full(positions.shape, float(self.dx_base))
        core = self.outer_core(axis)
        if core is not None:
            target[(positions < core[0]) | (positions > core[1])] = float(self.outer_dx)

        def covered(lo, hi):
            first, stop = np.searchsorted(positions, lo, 'left'), np.searchsorted(positions, hi, 'right')
            return slice(first, stop)

        for band in self._bands[axis]:
            span = covered(band.lo, band.hi)
            np.minimum(target[span], band.dx_target, out=target[span])
        if axis == 'z':   # only the z bands: the xy refinements fix no z extent
            return target

        for r in self._regions:
            lo, hi = (r.x0, r.x1) if axis == 'x' else (r.y0, r.y1)
            span = covered(lo, hi)
            np.minimum(target[span], r.dx_target, out=target[span])

        for c in self._circles:
            center = c.cx if axis == 'x' else c.cy
            span = covered(center - (c.radius + c.margin), center + (c.radius + c.margin))
            np.minimum(target[span], c.dx_target, out=target[span])

        for line in self._lines:
            if line.axis != axis or not line.width > 0:
                continue
            span = covered(line.position - line.width, line.position + line.width)
            distance = np.abs(positions[span] - line.position)
            blended = line.dx_target + (distance / line.width) * (self.dx_base - line.dx_target)
            np.minimum(target[span], blended, out=target[span])
        return target

    def _refinement_positions(self, axis):
        """Float64 ends and centre of every refinement's interval along `axis` (bands; x / y also regions, circles +
        margin, lines), as _target_dx_profile reads them."""
        intervals = [(band.lo, band.hi) for band in self._bands[axis]]
        if axis != 'z':
            intervals += [(r.x0, r.x1) if axis == 'x' else (r.y0, r.y1) for r in self._regions]
            for c in self._circles:
                center = c.cx if axis == 'x' else c.cy
                intervals.append((center - (c.radius + c.margin), center + (c.radius + c.margin)))
            intervals.extend((line.position - line.width, line.position + line.width) for line in self._lines
                             if line.axis == axis and line.width > 0)
        return np.array([value for lo, hi in intervals for value in (lo, 0.5 * (lo + hi), hi)], dtype=np.float64)

    def _build_1d_mesh(self, length, axis):
        """Cells marched on the graded target profile segment by segment between the snap positions (exact cell edges);
        each segment's end remainder is rounded to whole cells and its cells rescaled to its length (`_rescale_segment`).
        A custom axis returns its set_axis_edges widths."""
        if axis in self._custom_edges:
            return np.diff(self._custom_edges[axis]).astype(DTYPE)
        max_ratio = self.ratio(axis)
        segment_edges = np.array([0.0] + self.snap_positions(axis) + [float(length)])

        # Sample finer than the smallest target / segment so narrow refinements are not skipped.
        finest = min(self._finest_target(axis), float(np.diff(segment_edges).min()))
        n_sample = min(max(int(length / (0.2 * finest)), 1000), _MAX_PROFILE_SAMPLES)
        sample_pos = _segment_samples(segment_edges, length / n_sample)
        # Refinement ends and centres are always samples, so a refinement narrower than the capped spacing stays.
        extra = np.clip(self._refinement_positions(axis), 0.0, float(length))
        sample_pos = np.unique(np.concatenate([sample_pos, extra]))
        target = self._target_dx_profile(sample_pos, axis)
        for lo, hi in zip(segment_edges[:-1], segment_edges[1:]):   # a segment shorter than its target gets one cell
            span = slice(np.searchsorted(sample_pos, lo, 'left'), np.searchsorted(sample_pos, hi, 'right'))
            np.minimum(target[span], hi - lo, out=target[span])
        ratios = np.full(sample_pos.shape, max_ratio)   # per sample: the outer grading ratio outside the outer core
        core = self.outer_core(axis)
        if core is not None:
            ratios[(sample_pos < core[0]) | (sample_pos > core[1])] = self.axis_outer_ratio(axis)
        target_dx = _graded_profile(sample_pos, target, ratios)

        cells = []
        n_segments = len(segment_edges) - 1
        for k in range(n_segments):
            lo, hi = float(segment_edges[k]), float(segment_edges[k + 1])
            segment = _march_segment(sample_pos, target_dx, lo, hi, cells[-1] if cells else None, ratios)
            cells.extend(_rescale_segment(segment, hi - lo, snapped_lo=k > 0, snapped_hi=k < n_segments - 1))
        return np.array(cells, dtype=DTYPE)

    def cell_spacings(self, npml_sides=None, pml_dx=None, ghost_faces=()):
        """Full float32 (dx_arr, dy_arr): refined physical cells plus per-side PML padding graded at max_ratio from the
        window edge cell toward pml_dx (default dx_base); the padding of ghost_faces (mirror ghosts) copies the edge
        cell's width."""
        sides = {side: 0 for side in FACES} if npml_sides is None else npml_sides
        for side in FACES:
            if int(sides[side]) < 0:
                raise ValueError(f"npml_sides['{side}'] must be >= 0, got {sides[side]}")
        pml_cell = float(self.dx_base if pml_dx is None else pml_dx)

        dx_phys = self._build_1d_mesh(self.size_x, 'x')
        dy_phys = self._build_1d_mesh(self.size_y, 'y')
        dx_arr = _pad_with_pml(dx_phys, int(sides['x_lo']), int(sides['x_hi']), pml_cell, self.ratio('x'),
                               'x_lo' in ghost_faces, 'x_hi' in ghost_faces)
        dy_arr = _pad_with_pml(dy_phys, int(sides['y_lo']), int(sides['y_hi']), pml_cell, self.ratio('y'),
                               'y_lo' in ghost_faces, 'y_hi' in ghost_faces)
        return dx_arr, dy_arr

    def z_cell_spacings(self, n_lo=0, n_hi=0, pml_dx=None, ghost_faces=()):
        """Full float32 dz_arr (3D): z cells graded on the z bands (base dx_base elsewhere) or custom z edges, plus
        n_lo / n_hi PML cells graded like cell_spacings (z_lo / z_hi in ghost_faces: copies of the edge cell)."""
        if self.size_z is None:
            raise ValueError('z_cell_spacings needs a MeshConfig with size_z (3D)')
        if min(int(n_lo), int(n_hi)) < 0:
            raise ValueError(f'z PML cell counts must be >= 0, got {n_lo}, {n_hi}')
        dz_phys = self._build_1d_mesh(self.size_z, 'z')
        return _pad_with_pml(dz_phys, int(n_lo), int(n_hi), float(self.dx_base if pml_dx is None else pml_dx),
                             self.ratio('z'), 'z_lo' in ghost_faces, 'z_hi' in ghost_faces)

    def build(self, npml=0, pml_dx=None, npml_sides=None, courant=None, verbose=True, ghost_faces=()):
        """NonUniformGrid2D of this mesh; `npml_sides` (x_lo/x_hi/y_lo/y_hi) overrides the uniform `npml`."""
        if npml_sides is None:
            npml_sides = {side: int(npml) for side in FACES}
        dx_arr, dy_arr = self.cell_spacings(npml_sides, pml_dx, ghost_faces)
        return NonUniformGrid2D(dx_arr, dy_arr, courant=courant, verbose=verbose)


def _check_target(dx_target):
    if not dx_target > 0:
        raise ValueError(f"refinement dx_target must be > 0, got {dx_target}")


def complete_axis_edges(edges, length, axis='x'):
    """Float64 edges 0 .. length from increasing custom edges inside [0, length]: 0 and length added when missing,
    edges within 1e-9 x length of them or of each other merged; ValueError otherwise."""
    edges = np.asarray(edges, dtype=np.float64).ravel()
    tolerance = 1e-9 * length
    if edges.size and (not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0)):
        raise ValueError(f'custom {axis} edges must be finite and strictly increasing')
    if edges.size and (edges[0] < -tolerance or edges[-1] > length + tolerance):
        raise ValueError(f'custom {axis} edges must lie in [0, {length:g}] m, got {edges[0]:g} .. {edges[-1]:g}')
    merged = [0.0]
    for edge in edges:
        if tolerance < edge < length - tolerance and edge - merged[-1] > tolerance:
            merged.append(float(edge))
    return np.array(merged + [float(length)])


def segment_widths(length, n_cells, profile='uniform', ratio=1.0):
    """Float64 widths of n_cells summing to length: 'uniform'; 'geometric' (constant growth, last / first = ratio, < 1
    shrinks toward the end); 'tanh' (two-sided clustering toward both ends, centre / end cell = ratio)."""
    n_cells, ratio = int(n_cells), float(ratio)
    if not (length > 0 and n_cells >= 1 and ratio > 0):
        raise ValueError(f'segment needs length > 0, cells >= 1 and ratio > 0, got {length:g}, {n_cells}, {ratio:g}')
    if profile not in SEGMENT_PROFILES:
        raise ValueError(f"segment profile must be one of {', '.join(SEGMENT_PROFILES)}, got {profile!r}")
    if profile == 'uniform' or n_cells == 1 or ratio == 1.0:
        return np.full(n_cells, length / n_cells)
    if profile == 'geometric':
        widths = ratio ** (np.arange(n_cells) / (n_cells - 1))
        return widths * (length / widths.sum())
    if ratio < 1.0:
        raise ValueError(f"tanh profile ratio (centre / end cell) must be >= 1, got {ratio:g}")
    if n_cells == 2:
        return np.full(2, 0.5 * length)   # two cells cannot cluster

    def tanh_widths(stretch):
        edges = 0.5 * (1.0 + np.tanh(stretch * np.linspace(-1.0, 1.0, n_cells + 1)) / np.tanh(stretch))
        return np.diff(edges)

    def tanh_ratio(stretch):
        widths = tanh_widths(stretch)
        with np.errstate(divide='ignore'):   # end cells round to 0 at large stretch: ratio inf
            return widths.max() / widths.min()

    low, high = 1e-6, 1.0
    while tanh_ratio(high) < ratio:   # centre / end ratio grows monotonically with the stretch
        low, high = high, 2.0 * high
        if high > 1e3:
            raise ValueError(f'tanh profile: ratio {ratio:g} is out of reach with {n_cells} cells')
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if tanh_ratio(middle) < ratio else (low, middle)
    return tanh_widths(0.5 * (low + high)) * length


def _graded_profile(positions, target, ratios):
    """Largest profile <= target with local slope (r - 1) / r (r = ratios per sample, the smaller of an interval's two
    ends): cells marched on it change by at most r, so every refinement keeps its target and the ramps lie outside it."""
    ratios = np.broadcast_to(np.asarray(ratios, dtype=np.float64), positions.shape)
    interval_ratio = np.minimum(ratios[1:], ratios[:-1])
    climb = np.concatenate([[0.0], np.cumsum((interval_ratio - 1.0) / interval_ratio * np.diff(positions))])
    descent = climb[-1] - climb                       # slope integral from each sample to the end
    from_left = climb + np.minimum.accumulate(target - climb)
    from_right = descent + np.minimum.accumulate((target - descent)[::-1])[::-1]
    return np.minimum(from_left, from_right)


def _segment_samples(segment_edges, spacing):
    """Sorted sample positions at most `spacing` apart within each segment, every segment edge included exactly once."""
    pieces = [np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / spacing)) + 1))
              for lo, hi in zip(segment_edges[:-1], segment_edges[1:])]
    return np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])


def _profile_at(sample_pos, target_dx, position):
    """(target cell size linearly interpolated at position, index of the sample interval holding it)."""
    idx = max(0, min(int(np.searchsorted(sample_pos, position, side='right')) - 1, len(sample_pos) - 2))
    frac = (position - sample_pos[idx]) / (sample_pos[idx + 1] - sample_pos[idx] + 1e-30)
    return target_dx[idx] + frac * (target_dx[idx + 1] - target_dx[idx]), idx


def _march_segment(sample_pos, target_dx, lo, hi, previous_cell, ratios):
    """Cells lo -> hi at the smaller interpolated target of each cell's two ends, within the local ratio of the one before
    (previous_cell: last cell of the previous segment, or None); the end remainder is rounded to whole cells."""
    cells = []
    pos = lo
    while pos < hi - 1e-15:
        dx_local, idx = _profile_at(sample_pos, target_dx, pos)
        dx_local = min(dx_local, _profile_at(sample_pos, target_dx, min(pos + dx_local, float(sample_pos[-1])))[0])

        last_cell = cells[-1] if cells else previous_cell
        if last_cell is not None:
            max_ratio = float(ratios[idx])
            dx_local = min(dx_local, last_cell * max_ratio)
            dx_local = max(dx_local, last_cell / max_ratio)

        remaining = hi - pos
        if remaining < dx_local * 1.5:
            if remaining >= 0.5 * dx_local or not cells:
                cells.append(dx_local)
            break

        cells.append(dx_local)
        pos += dx_local
    return cells


def _rescale_segment(cells, length, snapped_lo, snapped_hi):
    """Cells rescaled to sum to length; the correction tapers linearly to ~0 over a quarter of the cells toward a snapped
    end, so cells beside a snap node keep their marched ratio (no snapped end: uniform scaling)."""
    cells = np.asarray(cells, dtype=np.float64)
    centre_index = np.arange(len(cells)) + 0.5
    taper_cells = max(1.0, 0.25 * len(cells))
    weight = np.ones(len(cells))
    if snapped_lo:
        weight = np.minimum(weight, centre_index / taper_cells)
    if snapped_hi:
        weight = np.minimum(weight, (len(cells) - centre_index) / taper_cells)
    return cells * (1.0 + weight * (length - cells.sum()) / np.dot(cells, weight))


def _pml_widths_outward(n_cells, edge_cell, pml_cell, max_ratio):
    """PML cell widths ordered outward from the window: geometric steps of max_ratio from edge_cell, stopping at
    pml_cell (a ramp longer than the layer leaves its outer cells finer than pml_cell)."""
    growth = max_ratio ** np.arange(1, n_cells + 1, dtype=np.float64)
    if pml_cell >= edge_cell:
        return np.minimum(edge_cell * growth, pml_cell)
    return np.maximum(edge_cell / growth, pml_cell)


def _pad_with_pml(phys_cells, n_lo, n_hi, pml_cell, max_ratio, ghost_lo=False, ghost_hi=False):
    """Float32 widths: n_lo / n_hi padding cells around phys_cells, graded PML or (ghost_*) copies of the edge cell."""
    lo = (np.full(n_lo, float(phys_cells[0])) if ghost_lo
          else _pml_widths_outward(n_lo, float(phys_cells[0]), pml_cell, max_ratio)[::-1])
    hi = (np.full(n_hi, float(phys_cells[-1])) if ghost_hi
          else _pml_widths_outward(n_hi, float(phys_cells[-1]), pml_cell, max_ratio))
    return np.concatenate([lo, phys_cells.astype(np.float64), hi]).astype(DTYPE)


class NonUniformGrid2D:
    """Rectilinear grid from per-cell widths; dt from the finest cells (courant = fraction of the 2D CFL limit)."""

    def __init__(self, dx_arr, dy_arr, courant=None, verbose=True):
        self.dx_arr = np.asarray(dx_arr, dtype=DTYPE)
        self.dy_arr = np.asarray(dy_arr, dtype=DTYPE)

        self.nx = len(self.dx_arr)
        self.ny = len(self.dy_arr)

        # Total extent including PML; cell coordinates come from Domain.cell_edges/cell_centers.
        self.size_x = float(np.sum(self.dx_arr, dtype=np.float64))
        self.size_y = float(np.sum(self.dy_arr, dtype=np.float64))

        self.dx_min = float(np.min(self.dx_arr))
        self.dy_min = float(np.min(self.dy_arr))
        self.dx_max = float(np.max(self.dx_arr))
        self.dy_max = float(np.max(self.dy_arr))

        self.dx = DTYPE(self.dx_min)
        self.dy = DTYPE(self.dy_min)

        self.dt = courant_time_step(self.dx_min, self.dy_min, courant)
        self.courant = DTYPE(DEFAULT_COURANT_FACTOR if courant is None else courant)

        ratios_x = self.dx_arr[1:] / self.dx_arr[:-1]
        ratios_y = self.dy_arr[1:] / self.dy_arr[:-1]
        self.max_cell_ratio_x = float(np.max(np.maximum(ratios_x, 1.0/ratios_x))) if len(ratios_x) > 0 else 1.0
        self.max_cell_ratio_y = float(np.max(np.maximum(ratios_y, 1.0/ratios_y))) if len(ratios_y) > 0 else 1.0

        if verbose:
            self._print_summary()

    def _print_summary(self):
        print(
            f"NonUniformGrid2D: {self.nx} x {self.ny} cells ({self.nx * self.ny:,} total)\n"
            f"  dx: {self.dx_min*1e9:.1f}-{self.dx_max*1e9:.1f} nm, "
            f"dy: {self.dy_min*1e9:.1f}-{self.dy_max*1e9:.1f} nm\n"
            f"  Max adjacent ratio: x={self.max_cell_ratio_x:.3f}, y={self.max_cell_ratio_y:.3f}\n"
            f"  dt = {float(self.dt)*1e18:.3f} as"
        )
        if self.max_cell_ratio_x > 1.25 or self.max_cell_ratio_y > 1.25:
            print("  WARNING: Adjacent ratio > 1.25 may cause spurious reflections.")

    @property
    def shape(self):
        return (self.nx, self.ny)

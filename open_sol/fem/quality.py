"""Triangle quality measures: element angles, aspect ratios and the MeshQuality record (check.json / self-tests)."""
import math
from dataclasses import asdict, dataclass

import numpy as np

from .mesh import signed_areas

QUALITY_ANGLE = 25.0                                   # deg: the self-tested minimum angle away from small inputs
ASPECT_BIN_EDGES = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, math.inf)


@dataclass(frozen=True)
class MeshQuality:
    """Quality of the interior (non-PML) elements; the PML layer ring only enters min_angle_all."""
    n_elements: int
    n_points: int
    min_angle: float            # deg, interior elements
    mean_min_angle: float       # deg, mean over interior elements of their smallest angle
    min_angle_all: float        # deg, PML ring included
    below_quality_angle: int    # interior elements with a smallest angle < QUALITY_ANGLE
    max_aspect: float           # circumradius / (2 inradius): 1 = equilateral
    aspect_bin_edges: tuple
    aspect_counts: tuple        # interior elements per aspect bin
    max_size_ratio: float       # longest interior edge / h at its midpoint (nan without a size field)
    small_input_angles: tuple   # ((x, y, angle_deg), ...) accepted input angles below 60 deg
    rounds: int                 # mesher refinement rounds (0 for gmsh)
    mesher: str                 # 'builtin' | 'gmsh'

    def as_dict(self) -> dict:
        return asdict(self)


def element_angles(points, triangles) -> np.ndarray:
    """Float64 (n, 3) interior angles [deg]; angle k at node k."""
    corners = points[triangles]
    angles = np.empty(triangles.shape, dtype=np.float64)
    for k in range(3):
        to_next = corners[:, (k + 1) % 3] - corners[:, k]
        to_previous = corners[:, (k + 2) % 3] - corners[:, k]
        cross = to_next[:, 0] * to_previous[:, 1] - to_next[:, 1] * to_previous[:, 0]
        dot = (to_next * to_previous).sum(axis=1)
        angles[:, k] = np.degrees(np.arctan2(np.abs(cross), dot))
    return angles


def aspect_ratios(points, triangles) -> np.ndarray:
    """Circumradius / (2 inradius) per element (1 equilateral, inf degenerate)."""
    corners = points[triangles]
    lengths = np.stack([np.linalg.norm(corners[:, (k + 2) % 3] - corners[:, (k + 1) % 3], axis=1)
                        for k in range(3)], axis=1)
    area = np.abs(signed_areas(points, triangles))
    product = lengths.prod(axis=1)
    perimeter = lengths.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        circumradius = product / (4.0 * area)
        inradius = 2.0 * area / perimeter
        return np.where(area > 0.0, circumradius / (2.0 * inradius), np.inf)


def edge_size_ratios(points, triangles, size_field) -> np.ndarray:
    """Longest-to-size ratio per unique edge of the triangles: length / h(midpoint)."""
    pairs = np.concatenate([triangles[:, [(k + 1) % 3, (k + 2) % 3]] for k in range(3)])
    pairs = np.sort(pairs, axis=1)
    keys = np.unique(pairs[:, 0].astype(np.int64) * len(points) + pairs[:, 1])   # 1D keys: no np.unique(axis=0)
    pairs = np.column_stack([keys // len(points), keys % len(points)])
    starts, ends = points[pairs[:, 0]], points[pairs[:, 1]]
    return np.linalg.norm(ends - starts, axis=1) / size_field(0.5 * (starts + ends))


def mesh_quality(points, triangles, interior=None, *, size_field=None, small_input_angles=(), rounds=0,
                 mesher='builtin') -> MeshQuality:
    """MeshQuality of a mesh; interior = bool mask of the elements judged (None: all)."""
    interior = np.ones(len(triangles), dtype=bool) if interior is None else np.asarray(interior, dtype=bool)
    smallest = element_angles(points, triangles).min(axis=1)
    inner = triangles[interior]
    inner_smallest = smallest[interior]
    aspects = aspect_ratios(points, inner)
    counts, _ = np.histogram(np.minimum(aspects, 1e300), bins=np.array(ASPECT_BIN_EDGES[:-1] + (1e300,)))
    size_ratio = (float(edge_size_ratios(points, inner, size_field).max(initial=0.0))
                  if size_field is not None and len(inner) else math.nan)
    return MeshQuality(
        n_elements=int(len(triangles)), n_points=int(len(points)),
        min_angle=float(inner_smallest.min(initial=math.inf)),
        mean_min_angle=float(inner_smallest.mean()) if len(inner_smallest) else math.nan,
        min_angle_all=float(smallest.min(initial=math.inf)),
        below_quality_angle=int((inner_smallest < QUALITY_ANGLE).sum()),
        max_aspect=float(aspects.max(initial=1.0)), aspect_bin_edges=ASPECT_BIN_EDGES,
        aspect_counts=tuple(int(count) for count in counts), max_size_ratio=size_ratio,
        small_input_angles=tuple(tuple(float(value) for value in entry) for entry in small_input_angles),
        rounds=int(rounds), mesher=mesher)

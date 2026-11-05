"""Core package: re-exports the constants, Grid2D / Grid3D, MeshConfig, MaterialGrid, Domain / Domain3D, envelopes and
sources, FieldProbe, FieldInfo / PhysicsRenderer and the PhysicsSolver ABC."""
from .constants import DTYPE, C0, EPS0, MU0, ETA0, FACES
from .grid import Grid2D
from .mesh import MeshConfig, NonUniformGrid2D, MeshRegion, MeshCircle, MeshLine
from .material_grid import MaterialGrid
from .metric import Coordinates
from .domain import Domain
from .grid3d import Grid3D
from .domain3d import Domain3D
from .sources import (
    GaussianPulse, ContinuousWave, RickerWavelet,
    PointSource, LineSource,
)
from .monitors import FieldProbe
from .renderer import FieldInfo, PhysicsRenderer
from .solver_base import PhysicsSolver

__all__ = [
    'DTYPE', 'C0', 'EPS0', 'MU0', 'ETA0', 'FACES',
    'Grid2D',
    'MeshConfig', 'NonUniformGrid2D', 'MeshRegion', 'MeshCircle', 'MeshLine',
    'MaterialGrid',
    'Coordinates', 'Domain',
    'Grid3D', 'Domain3D',
    'GaussianPulse', 'ContinuousWave', 'RickerWavelet',
    'PointSource', 'LineSource',
    'FieldProbe',
    'FieldInfo', 'PhysicsRenderer',
    'PhysicsSolver',
]

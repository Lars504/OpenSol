"""3D FDTD (B2, physics type em_fdtd_3d): Yee engine on a core.domain3d.Domain3D, and its frequency-domain eigensolver
(em_fdfd_3d: FDFDPhysics3D on the same Yee grid); the model definitions live in definition.py / definition_fdfd3d.py
(never imported here)."""
from .boundaries3d import BoundaryConfig3D
from .fdfd3d import FDFDPhysics3D
from .physics3d import EMPhysics3D

__all__ = ['BoundaryConfig3D', 'EMPhysics3D', 'FDFDPhysics3D']

"""Eigenmode expansion (em_eme / em_eme_3d, eme_plan.md): EMEPhysics (2D) and EMEPhysics3D; the model definitions
live in definition.py / definition3d.py (never imported here)."""
from .solver import EMEPhysics
from .solver3d import EMEPhysics3D

__all__ = ['EMEPhysics', 'EMEPhysics3D']

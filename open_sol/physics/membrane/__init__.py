"""Membrane package (new_modules_plan S4): MembraneFEMPhysics (triangle meshes) and MembraneLinePhysics (1D models);
the model definition lives in definition.py (never imported here)."""
from .solver_fem import MembraneFEMPhysics
from .solver_line import LineHarmonicSystem, MembraneLinePhysics

__all__ = ['MembraneFEMPhysics', 'MembraneLinePhysics', 'LineHarmonicSystem']

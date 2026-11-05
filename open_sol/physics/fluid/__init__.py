"""Laminar flow (D2Q9 lattice Boltzmann) and its convective heat coupling; definition.py is loaded by the registry."""
from .convection import AdvectiveHeatSource
from .solver import FluidPhysics

__all__ = ['AdvectiveHeatSource', 'FluidPhysics']

"""NLFD-chi: cascaded 2D frequency-domain chi(n) mixing on the FDFD core (physics type em_fdfd_nonlinear,
nonlinear_optics_plan); the model definition lives in definition.py (never imported here)."""
from .mixing import ChannelSpec
from .solver import NonlinearHarmonicPhysics

__all__ = ['ChannelSpec', 'NonlinearHarmonicPhysics']

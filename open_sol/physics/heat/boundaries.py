"""Heat boundary conditions are the generic core.fv_diffusion ones (robin external value: `external`)."""
from ...core.fv_diffusion import BCSide, ScalarBoundaryConfig

HeatBoundaryConfig = ScalarBoundaryConfig

__all__ = ['BCSide', 'HeatBoundaryConfig', 'ScalarBoundaryConfig']

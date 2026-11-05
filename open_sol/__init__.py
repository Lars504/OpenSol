"""Open_sol - GPU 2D / 3D multiphysics (FDTD / FDFD electromagnetics, modes, heat, currents / electrostatics,
semiconductors, mechanics, acoustics, fluid): ModelSpec -> validate -> build -> run_study. Headless API open_sol.api +
CLI python -m open_sol; GUI open_sol.gui."""

__version__ = "0.3.0"   # defined before `from . import api` below: api reads it at import

from .simulation import Simulation

from .core.constants import C0, EPS0, MU0, ETA0, DTYPE, FACES
from .core.grid import Grid2D
from .core.mesh import MeshConfig, NonUniformGrid2D, MeshRegion, MeshCircle, MeshLine
from .core.material_grid import MaterialGrid
from .core.domain import Domain
from .core.solver_base import PhysicsSolver
from .core.renderer import FieldInfo, PhysicsRenderer
from .core.monitors import probe_result

from .physics.em import (
    EMPhysics, FDFDPhysics,
    FDTDEngine2D_Ez, FDTDEngine2D_Hz,
    Fields2D_Ez, Fields2D_Hz,
    DrudePole, LorentzPole,
    BCType, BoundaryConfig, BoundaryManager,
    GaussianPulse, ContinuousWave, RickerWavelet,
    PointSource, LineSource, TFSFSource, TFSFSourceHz,
    FieldProbe, DFTMonitor, FluxMonitor, NearToFarField,
)
from .physics.em.results import flux_result, dft_result, far_field_result
from .physics.em.index_reference import record_reference_index

from .physics.heat import (
    HeatPhysics,
    HeatBoundaryConfig, BCSide,
    THERMAL_PROPERTY_ARRAYS,
    attach_thermal_properties,
    em_to_heat_power, heat_to_em_dn, heat_to_em_dn_xx, heat_to_em_dn_yy,
)

from .physics.potential import PotentialPhysics

from .materials import (
    Material,
    GOLD, SILVER, ALUMINUM, SILICON, SILICA, LITHIUM_NIOBATE_Z_CUT, AIR, WATER, HEAVY_WATER,
)

from .coupling import CouplingTerm, CouplingGroup, DynamicMaterialUpdate, MultiphysicsRunner, SlotSum

from .viz import Visualizer, plot_field_array

# model-layer twins of the api names (load_model, save_model, make_template, validate, run_study) stay in their
# submodules: the top level exposes one spelling each
from .model import ModelSpec, get_registry
from .model.builder import ModelError, build
from .model.templates import template_entries
from .io import export_field, export_result

from . import api
from .api import describe, template, load, save, check, run, scene_figure, selftest, Results, RunError

__all__ = [
    'Simulation',
    'C0', 'EPS0', 'MU0', 'ETA0', 'DTYPE', 'FACES',
    'Grid2D',
    'Domain', 'PhysicsSolver', 'FieldInfo', 'PhysicsRenderer', 'probe_result',
    'MeshConfig', 'NonUniformGrid2D', 'MeshRegion', 'MeshCircle', 'MeshLine',
    'MaterialGrid',
    'EMPhysics', 'FDFDPhysics', 'FDTDEngine2D_Ez', 'FDTDEngine2D_Hz',
    'Fields2D_Ez', 'Fields2D_Hz',
    'DrudePole', 'LorentzPole',
    'Material', 'GOLD', 'SILVER', 'ALUMINUM', 'SILICON', 'SILICA', 'LITHIUM_NIOBATE_Z_CUT', 'AIR', 'WATER',
    'HEAVY_WATER',
    'BCType', 'BoundaryConfig', 'BoundaryManager',
    'GaussianPulse', 'ContinuousWave', 'RickerWavelet',
    'PointSource', 'LineSource', 'TFSFSource', 'TFSFSourceHz',
    'FieldProbe', 'DFTMonitor', 'FluxMonitor', 'NearToFarField',
    'flux_result', 'dft_result', 'far_field_result',
    'HeatPhysics',
    'HeatBoundaryConfig', 'BCSide', 'THERMAL_PROPERTY_ARRAYS',
    'attach_thermal_properties', 'record_reference_index',
    'em_to_heat_power', 'heat_to_em_dn', 'heat_to_em_dn_xx', 'heat_to_em_dn_yy',
    'PotentialPhysics',
    'CouplingTerm', 'CouplingGroup', 'DynamicMaterialUpdate', 'MultiphysicsRunner', 'SlotSum',
    'Visualizer', 'plot_field_array',
    'ModelSpec', 'get_registry', 'ModelError', 'build',
    'template_entries',
    'export_field', 'export_result',
    'api', 'describe', 'template', 'load', 'save', 'check', 'run', 'scene_figure', 'selftest', 'Results', 'RunError',
]

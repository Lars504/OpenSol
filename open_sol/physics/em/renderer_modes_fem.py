"""Renderer of em_modes on a triangle mesh (solvers/modes_fem FEMModePhysics): the shared fem.renderer
FEMFieldRenderer (fem_field coefficients; Session rasterises them) with ModeRenderer's field names and FieldInfos."""
from ...fem.renderer import FEMFieldRenderer
from .renderer import ModeRenderer


class FEMModeRenderer(FEMFieldRenderer):
    """FEMFieldRenderer of a FEMModePhysics: physics.fem_field per ModeRenderer component, the selection label."""

    material_names = ModeRenderer.material_names

    def __init__(self, physics):
        super().__init__(physics, 'em', ModeRenderer.field_infos, physics.selection_label)

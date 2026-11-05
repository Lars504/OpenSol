"""Elastic PML ring of solid_mechanics on a triangle mesh (new_modules_plan K5): the fem.pml stretch of the mesh's layer
ring at the eigenfrequency target, its wave speed, the ring-only dofs of the PML-mode filter and the mesh padding the
definition requests. Host numpy."""
# The stretch is fem.pml's: s = 1 - j beta (d / w)^2 at depth d into the ring of face f (thickness w), beta = 3 ln(1 / R)
# c / (2 omega_t w) for the fastest wave c = sqrt(c_II / rho) of the ring's solid elements at the target omega_t (the
# map of physics/mechanics/pml3d): frequency independent, so the stretched pencil K(s) u = omega^2 M(s) u stays linear in
# omega^2. Axisymmetric: the ring weight r -> r~ beyond x_hi (fem.pml radial_ratio) and the hoop strain u_r / r~; no
# layer on x_lo (the axis). The ring's outer side (the mesh's outer face tags) is clamped. The ring carries the material
# of the window edge (clamped-centroid rule); static solves (the prestress state of stress_stiffening) see it as an
# ordinary, unstretched elastic continuation clamped at its outer side, the eigen pencil the stretched K, K_sigma, loss
# and mass.
import math

from ...core.constants import FACES
from ...fem.pml import DEFAULT_PML_REFLECTION, pml_ring, ring_widths
from .pml3d import DEFAULT_PML_CELLS, PMLPadding, ring_only_dofs, ring_wave_speed

__all__ = ('DEFAULT_PML_CELLS', 'DEFAULT_PML_REFLECTION', 'check_ring', 'elastic_pml_ring', 'pml_padding',
           'ring_only_dofs', 'ring_wave_speed')


def pml_padding(faces, cells=DEFAULT_PML_CELLS) -> PMLPadding:
    """DomainRequirements.boundary_config of a 2D elastic PML: npml_sides = the ring thickness in mesh dx (h_max) on
    each face of `faces` (the MeshSpec pml_layers element layers across), 0 on the others."""
    return PMLPadding({face: int(cells) if face in faces else 0 for face in FACES})


def check_ring(mesh, faces, label) -> None:
    """ValueError when the mesh's layer ring and the PML faces differ (e.g. another physics on the mesh rings more
    faces); label() names the solid."""
    for face, width in ring_widths(mesh).items():
        if (width > 0.0) != (face in faces):
            raise ValueError(f"{label()}: triangle mesh PML ring on {face}: {'present' if width > 0.0 else 'missing'}, "
                             f"but pml_{face} is {'off' if width > 0.0 else 'on'}; mesh and PML settings disagree (a "
                             'ring requested by another physics on the same mesh carries no elastic PML)')


def elastic_pml_ring(mesh, faces, omega_target, wave_speed, reflection=DEFAULT_PML_REFLECTION, axisymmetric=False,
                     label=lambda: 'solid mechanics'):
    """fem.pml.PMLStretch of the mesh's ring on `faces` at omega_target [rad/s] > 0 for the fastest wave_speed [m/s];
    ValueError for a target <= 0, an axisymmetric layer on x_lo, or a ring that differs from the faces."""
    if not (math.isfinite(omega_target) and omega_target > 0.0):
        raise ValueError(f'{label()}: an elastic PML needs an eigenfrequency target > 0 (its strength is set there), got '
                         f'omega = {omega_target!r} rad/s')
    if axisymmetric and 'x_lo' in faces:
        raise ValueError(f'{label()}: an axisymmetric solid has no PML on x_lo (the axis r = 0 lies there)')
    check_ring(mesh, faces, label)
    try:
        return pml_ring(mesh, wave_speed, omega_target, reflection, faces=faces)
    except ValueError as exc:
        raise ValueError(f'{label()}: {exc}') from None

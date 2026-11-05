"""Elastic PML shell of solid_mechanics_3d on a tetrahedron mesh (piezo_3d_plan stage 6, D13): complex coordinate
stretches over the mesh's shell beyond the PML faces, their strength from a target reflection at the eigenfrequency
target, and the mesh padding the definition requests (PMLPadding). Host numpy."""
# Stretch (exp(+j omega t)): s_i = 1 - j beta_f (d / w)^2 at depth d into the shell of face f (thickness w) along its
# axis, 1 elsewhere; an outgoing wave exp(-j k x) becomes exp(-j k x - k beta_f d^3 / (3 w^2)). beta_f is frequency
# independent (a fixed complex coordinate map, so K(s) u = omega^2 M(s) u stays linear in omega^2): beta_f = 3 ln(1 /
# reflection) c / (2 omega_t w) gives the round trip exp(-2 k beta_f w / 3) = reflection for the fastest wave (k =
# omega_t / c, c = the largest sqrt(c_II / rho) of the shell's solid elements); slower waves and higher omega decay
# more, modes far below the target less. The stretched weak form: d/dx_i -> d/dx_i / s_i and dV -> s_x s_y s_z dV
# (fem.elasticity3d stretch), unconjugated: K and M complex symmetric. The shell's outer faces are held (u = 0).
import math
from dataclasses import dataclass

import numpy as np

PML_GRADING_ORDER = 2
DEFAULT_PML_REFLECTION = 1e-6
DEFAULT_PML_CELLS = 8


@dataclass(frozen=True)
class PMLPadding:
    """DomainRequirements.boundary_config of an elastic PML: npml_sides = PML element layers per face (x h_max = the
    tetrahedron mesh's shell thickness, model.meshing)."""
    npml_sides: dict

    def differences(self, other) -> tuple:
        """Names that differ from another config ('type' for a non-PMLPadding one)."""
        if not isinstance(other, PMLPadding):
            return ('type',)
        return ('npml_sides',) if dict(self.npml_sides) != dict(other.npml_sides) else ()


@dataclass(frozen=True, eq=False)
class ElasticPMLShell3D:
    """Stretch of the mesh's shell: window box, shell thickness [m] and strength beta per outer face (0 = none)."""
    window: tuple
    widths: dict
    strength: dict

    @property
    def faces(self) -> tuple:
        return tuple(face for face, width in self.widths.items() if width > 0.0)

    def stretches(self, points, elements=None) -> np.ndarray:
        """Complex128 (..., 3) (s_x, s_y, s_z) at points (..., 3) [m] (1 inside the window); `elements` unused (the
        fem.elasticity3d stretch callable form)."""
        points = np.asarray(points, dtype=np.float64)
        result = np.ones(points.shape, dtype=np.complex128)
        for axis, name in enumerate('xyz'):
            coordinate = points[..., axis]
            for face, depth in ((f'{name}_lo', self.window[axis] - coordinate),
                                (f'{name}_hi', coordinate - self.window[3 + axis])):
                width = self.widths[face]
                if width <= 0.0:
                    continue
                inside = depth > 0.0
                fraction = np.clip(depth[inside] / width, 0.0, 1.0)
                result[..., axis][inside] = 1.0 - 1j * self.strength[face] * fraction ** PML_GRADING_ORDER
        return result

    def jacobian(self, points, elements=None) -> np.ndarray:
        """Complex (...,) s_x s_y s_z at points (..., 3) (the stretched volume element)."""
        return np.prod(self.stretches(points), axis=-1)

    def text(self) -> str:
        return ', '.join(f'{face} {self.widths[face]:.4g} m, beta {self.strength[face]:.3g}' for face in self.faces)


def elastic_pml_shell_3d(mesh, faces, omega_target, wave_speed, reflection=DEFAULT_PML_REFLECTION):
    """ElasticPMLShell3D of the mesh's shell on `faces` (module header strength at omega_target [rad/s] > 0 and the
    fastest wave_speed [m/s] of the shell); ValueError when the shell and the PML faces disagree."""
    if not (math.isfinite(omega_target) and omega_target > 0.0):
        raise ValueError(f'an elastic PML needs an eigenfrequency target > 0 (its strength is set there), got omega = '
                         f'{omega_target!r} rad/s')
    if not (0.0 < reflection < 1.0):
        raise ValueError(f'pml_reflection must lie in (0, 1), got {reflection!r}')
    bounds, window = mesh.bounds, tuple(float(value) for value in mesh.window)
    slack = 1e-9 * max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    widths, strength = {}, {}
    for axis, name in enumerate('xyz'):
        for face, width in ((f'{name}_lo', window[axis] - bounds[axis]),
                            (f'{name}_hi', bounds[3 + axis] - window[3 + axis])):
            width = width if width > slack else 0.0
            if (width > 0.0) != (face in faces):
                raise ValueError(f"tetrahedron mesh PML shell on {face}: {'present' if width > 0.0 else 'missing'}, but "
                                 f"pml_{face} is {'off' if width > 0.0 else 'on'}; mesh and PML settings disagree")
            widths[face] = width
            strength[face] = (3.0 * math.log(1.0 / reflection) * wave_speed / (2.0 * omega_target * width)
                              if width else 0.0)
    return ElasticPMLShell3D(window=window, widths=widths, strength=strength)


def ring_wave_speed(diagonal, density, solid, window, label) -> float:
    """Fastest sqrt(c_II / rho) [m/s] over the PML ring / shell's solid elements with rho > 0 (the window's when it has
    none); diagonal = the largest c_II per element [Pa], density [kg/m^3], solid / window bool per element (2D and 3D)."""
    massive = np.asarray(solid, dtype=bool) & (np.asarray(density) > 0.0)
    ring = massive & ~np.asarray(window, dtype=bool)
    chosen = ring if ring.any() else massive
    if not chosen.any():
        raise ValueError(f'{label()}: an elastic PML needs solid elements with a density rho > 0')
    return float(np.sqrt(np.max(np.asarray(diagonal)[chosen] / np.asarray(density)[chosen])))


def ring_only_dofs(element_dofs, window, n_dofs, components) -> np.ndarray:
    """Host bool over the vector dofs (components per scalar dof): dofs of non-window elements (rows of element_dofs,
    window bool per row) that no window element touches (the PML share of a mode; 2D ring and 3D shell)."""
    in_window = np.zeros(n_dofs, dtype=bool)
    in_window[element_dofs[window].ravel()] = True
    in_ring = np.zeros(n_dofs, dtype=bool)
    in_ring[element_dofs[~window].ravel()] = True
    return np.repeat(in_ring & ~in_window, components)

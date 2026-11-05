"""membrane on a triangle mesh (MembraneFEMPhysics): complex Lagrange P1 / P2 w of D (1 + j eta) del^4 w - div(T (1 +
j eta) grad w) - rho_s omega^2 w = p on MembraneTriangleSystem (fem.harmonic.ScalarHarmonicSystem + the C0 interior
penalty plate terms + port Dirichlet-to-Neumann terms per frequency), clamped / simply supported / mirror faces and outlines,
pressure, line and point forces, ports, an optional fem.pml ring, Bloch cells (eigenfrequency); state per element
(D12), display by fem_field."""
# Bloch (periodic_x / _y, fem.dofs.bloch_masters): K and M of the free dofs (+ the bending seam terms) are reduced by
# the prolongation P to P^H K P, P^H M P (complex Hermitian with a complex phase: the 'general' resonance pencil; real
# when every k L is a multiple of pi). Ports (ports.py; a plate face plate_ports.PlatePort, chosen in _new_system from D
# at the port elements, its block on every P2 dof of those elements): the DtN block depends on omega, so it is kept
# out of the HarmonicPencil; its pattern sits in K as explicit zeros, so every frequency's reduced operator keeps one
# pattern (cuDSS plan reuse) and MembraneTriangleSystem.frequency_terms (the HarmonicSystemBase per-omega hook) adds the
# values per frequency.
import math
import warnings

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE, FACES
from ...core.harmonic_probe import HarmonicProbeRecord
from ...fem.assembly import _to_csr
from ...fem.diffusion import window_face_edges
from ...fem.dofs import bloch_masters, bloch_prolongation, lagrange_dofs
from ...fem.elements import lagrange_basis
from ...fem.harmonic import HarmonicCondition, ScalarHarmonicSystem
from ...fem.mesh import FACE_BITS
from ...fem.mesher import MERGE_TOLERANCE
from ...fem.pml import DEFAULT_PML_REFLECTION, pml_ring
from ...fem.renderer import FEMFieldRenderer
from .bending_fem import assemble_plate_bending, assemble_seam_bending, seam_phase
from .plate_ports import PlatePort
from .ports import (PORT_UNIFORM_RTOL, MembranePort, ProjectionRecord, folded_mode, mode_projector, port_edges,
                    unfolded_modes)
from .renderer import MEMBRANE_FIELD_INFOS, PHASOR_FIELD_NAMES, shown_label
from .solver_common import MembraneBase, complex_amplitude, pencil_shift_solver

_CENTROID = np.array([[1.0 / 3.0, 1.0 / 3.0]])
_CLAMPED = HarmonicCondition('dirichlet', 0.0)


class MembraneTriangleSystem(ScalarHarmonicSystem):
    """ScalarHarmonicSystem with the plate bending form (bending_fem: rigidity D (1 + j eta), Poisson ratio, slope-held
    edges) in K and port DtN terms added per frequency (module header)."""

    def __init__(self, mesh, order=2, linear_solver=None, *, pml=None, label=lambda: 'membrane fem'):
        super().__init__(mesh, order, linear_solver, pml=pml, label=label)
        self.bending = None                          # (rigidity (n_elements,), poisson (n_elements,)) or None
        self.slope_edges = np.zeros(0, dtype=np.int64)
        self.ports = []                              # MembranePort
        self._port_layout = None                     # (key, entry slots, indptr, indices) of the DtN pattern

    def set_bending(self, rigidity, poisson) -> None:
        if self.order != 2:
            raise ValueError(f'{self.label()}: plate bending needs quadratic elements (discretization quadratic)')
        self.bending = (np.asarray(rigidity).copy(), np.asarray(poisson, dtype=np.float64).copy())
        self.operator_dirty = True

    def set_slope_edges(self, edges) -> None:
        """Mesh edges whose slope dw/dn is held with bending (clamped / symmetry faces and outlines, PML outer sides)."""
        self.slope_edges = np.unique(np.asarray(edges, dtype=np.int64))
        self.operator_dirty = True

    def add_port(self, port) -> None:
        self.ports.append(port)
        self.operator_dirty = True

    def _assemble_terms(self) -> dict:
        terms = super()._assemble_terms()
        if self.bending is not None:
            rigidity, poisson = self.bending
            terms['stiffness'].append(assemble_plate_bending(self.dofs, rigidity, poisson, self.slope_edges,
                                                             stretch=self.pml))
        if self.ports:   # explicit zeros: the DtN pattern, so the reduced operator's pattern is the same every omega
            rows, cols = [], []
            for port in self.ports:
                port.assemble(self.dofs)
                count = len(port.positions)
                rows.append(np.repeat(port.positions, count))
                cols.append(np.tile(port.positions, count))
            size = self.dofs.n_dofs
            terms['stiffness'].append(_to_csr(rows, cols, [np.zeros(len(part)) for part in rows], (size, size)))
        return terms

    def _port_terms(self, omega) -> tuple:
        """(system CSR of every port's DtN block, system load of the incident modes) at omega [rad/s]: the blocks summed
        into the canonical pattern kept per assembly (only the data per omega; same index arrays every omega)."""
        import scipy.sparse
        size = len(self.global_ids)
        slots, indptr, indices = self._port_pattern(size)
        values = []
        load = np.zeros(size, dtype=np.complex128)
        for port in self.ports:
            block, port_load = port.terms(omega)
            values.append(np.asarray(block, dtype=np.complex128).ravel())
            load[port.positions] += port_load   # positions are unique within a port
        values = np.concatenate(values)
        data = (np.bincount(slots, weights=values.real, minlength=indices.size)
                + 1j * np.bincount(slots, weights=values.imag, minlength=indices.size))
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(size, size)), load

    def _port_pattern(self, size) -> tuple:
        """(slot of every block entry, indptr, indices) of the summed port DtN blocks (block entries row-major, port
        after port), kept while the ports' positions (set per assembly) are unchanged."""
        key = tuple(port.positions for port in self.ports)   # held: identity stays meaningful
        kept = self._port_layout
        if (kept is None or kept[0][0] != size or len(kept[0][1]) != len(key)
                or any(old is not new for old, new in zip(kept[0][1], key))):
            rows = np.concatenate([np.repeat(port.positions, len(port.positions)) for port in self.ports])
            cols = np.concatenate([np.tile(port.positions, len(port.positions)) for port in self.ports])
            entries, slots = np.unique(rows.astype(np.int64) * size + cols, return_inverse=True)
            index_dtype = np.int32 if size < 2 ** 31 else np.int64
            indptr = np.concatenate(([0], np.cumsum(np.bincount(entries // size, minlength=size)))).astype(index_dtype)
            self._port_layout = ((size, key), slots.reshape(-1), indptr, (entries % size).astype(index_dtype))
        return self._port_layout[1:]

    def frequency_terms(self, omega) -> tuple:
        """HarmonicSystemBase hook: the port DtN blocks and incident loads of this omega (pattern in K: module header)."""
        if not self.ports:
            return None, None
        if not omega > 0.0:
            raise ValueError(f'{self.label()}: ports need an angular frequency > 0, got {omega!r} rad/s')
        return self._port_terms(omega)

    def port_outputs(self, omega) -> dict:
        """{port name: MembranePort.outputs} of the last solve."""
        if self.solution is None:
            raise RuntimeError(f'{self.label()}: solve before reading port outputs')
        return {port.name: port.outputs(omega, self.solution) for port in self.ports}


class MembraneFEMPhysics(MembraneBase):
    """Membrane on a fem.domain.TriangleDomain: the window elements without a hole (with a PML: also the ring on the
    pml faces, the stretch set at the reference frequency for the fastest wave of the ring); host complex factors
    through the 'linear' role (driven) and the 'eigen' role (resonances); a plate when a material sets
    bending_stiffness (quadratic only; with a PML the stretched plate form; ports with D > 0 at their face take the
    numeric strip modes of plate_ports.py); Bloch cells on periodic axes (eigenfrequency only)."""

    def __init__(self, domain, *, linear_solver=None, eigen_solver=None, target_frequency=0.0,
                 discretization='quadratic', pml_faces=(), pml_reflection=DEFAULT_PML_REFLECTION, pml_frequency=None,
                 periodic=(False, False), bloch_k=(0.0, 0.0)):
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError('membrane: MembraneFEMPhysics needs a triangle mesh (fem TriangleDomain); a 1D model uses '
                             'MembraneLinePhysics')
        unknown = [face for face in pml_faces if face not in FACES]
        if unknown:
            raise ValueError(f"membrane: unknown PML face(s) {', '.join(unknown)}; expected {', '.join(FACES)}")
        if not 0.0 < float(pml_reflection) < 1.0:
            raise ValueError(f'membrane: pml_reflection must lie in (0, 1), got {pml_reflection!r}')
        self.pml_faces = tuple(face for face in FACES if face in pml_faces)
        self.pml_reflection = float(pml_reflection)
        self.bloch_wavenumbers = {axis: float(bloch_k[axis]) for axis in (0, 1) if periodic[axis]}
        ringed = [face for face in self.pml_faces if 'xy'.index(face[0]) in self.bloch_wavenumbers]
        if ringed:
            raise ValueError(f"membrane: PML on the periodic face(s) {', '.join(ringed)}")
        super().__init__(domain, linear_solver=linear_solver, eigen_solver=eigen_solver,
                         target_frequency=target_frequency, discretization=discretization, pml_frequency=pml_frequency)

    # --- geometry hooks -------------------------------------------------------------
    def _prepare_geometry(self) -> None:
        mesh = self.domain.mesh
        if self.bending and self.order != 2:
            raise ValueError(f'{self._label()}: a plate (bending_stiffness) needs discretization quadratic')
        geometry_order = 1 if self.order == 1 else 2   # D6: isoparametric P2, as ScalarHarmonicSystem
        self.full_dofs = lagrange_dofs(mesh, self.order, geometry_order=geometry_order)
        self.element_map = lagrange_dofs(mesh, 0, geometry_order=geometry_order)
        self._window = np.asarray(mesh.pml_faces == 0, dtype=bool)
        self._face_clamps = set()
        self._face_supports = set()  # simply supported faces (w held, slope free)
        self._face_symmetries = set()  # even mirror faces (slope held with bending, w free)
        self._face_odd_mirrors = set()   # antisymmetry faces (in _face_supports too; a squeeze film vents there)
        self._face_forces = {}      # face -> complex force per length [N/m]
        self._edge_conditions = []  # (mesh edges, HarmonicCondition), in apply order
        self._slope_edge_sets = []  # clamped and symmetry outline edges (slope held with bending)
        self._symmetry_edge_sets = []   # symmetry outline edges (a squeeze film is sealed there)
        self.squeeze_films = []     # squeeze_film.MembraneSqueezeFilm, evaluated after every eigen solve
        self._point_forces = []     # ((x, y), complex force [N])
        # port dicts: name, face, mode_count, incidents [(unfolded mode, complex amplitude [m])], tension_tolerance,
        # sector_drive (a parity-sector copy: core.mirror.SECTOR_FEATURE_PARAMS)
        self._ports = []
        self._port_records = {}     # name -> (amplitude record, power record)
        self._projections = {}      # mode_projection name -> (ModeProjector, ProjectionRecord per driven frequency)
        self._pressure = np.zeros(mesh.n_elements, dtype=np.complex128)   # [Pa] per window element
        self._centroid_basis = lagrange_basis(self.order, _CENTROID)[0][0]
        self._device_phasor_maps = None   # (coefficients, full_dofs, device element dofs, basis, solved) of _publish

    def _field_shape(self) -> tuple:
        return (int(self.domain.mesh.n_elements),)

    def _n_elements(self) -> int:
        return int(self.domain.mesh.n_elements)

    def _n_full(self) -> int:
        return int(self.full_dofs.n_dofs)

    def _element_values(self, array) -> np.ndarray:
        return np.asarray(array, dtype=np.float64).reshape(-1)

    def _field_array(self, values) -> np.ndarray:
        return np.asarray(values)

    def _default_elements(self) -> np.ndarray:
        """Bool per element of the solved region before holes: every element with a PML, else the window ones."""
        if self.pml_faces:
            return np.ones(self._n_elements(), dtype=bool)
        return self._window

    def _new_system(self, pml):
        system = MembraneTriangleSystem(self.domain.mesh, self.order, self.linear_solver, pml=pml, label=self._label)
        for face in FACES:
            if face in self._face_clamps or face in self._face_supports:
                system.set_face_condition(face, _CLAMPED)
            elif face in self._face_forces:
                system.set_face_condition(face, HarmonicCondition('neumann', self._face_forces[face]))
        for edges, condition in self._edge_conditions:
            system.add_boundary_condition(edges, condition)
        for point, force in self._point_forces:
            system.add_point_load(point, force)
        if np.any(self._pressure != 0.0):
            system.set_source(self._pressure.copy())
        if self.bending:   # a PML face's outer side is clamped: w by the system (pml_outer), the slope here
            ringed = set(pml.faces) if pml is not None else set()
            held = ([system.face_entities(face) for face in sorted(self._face_clamps | self._face_symmetries | ringed)]
                    + list(self._slope_edge_sets))
            system.set_slope_edges(np.concatenate(held) if held else np.zeros(0, dtype=np.int64))
        if self._ports:
            coefficients = self._read_coefficients()
            active = self._default_elements() & ~coefficients.hole
            for port in self._ports:
                name, face = port['name'], port['face']
                if bool(((self.domain.mesh.pml_faces & FACE_BITS[face]) != 0).any()):
                    raise ValueError(f"{self._label()}: port '{name}' on {face}, which has a PML ring")
                edges, start, width = port_edges(self.domain.mesh, face, active)
                cross_axis = 1 if face[0] == 'x' else 0
                mirror_walls = self._mirror_walls(cross_axis, start, start + width)
                if self.bending and self._plate_port(edges, coefficients):   # numeric strip modes (plate_ports.py)
                    if port['sector_drive'] and any(wall is not None for wall in mirror_walls):
                        raise ValueError(f"{self._label()}: port '{name}': parity_sectors cannot combine plate ports "
                                         '(their modes are numbered on the drawn half guide); run the full model')
                    walls = self._plate_wall_kinds(cross_axis, start, start + width, edges, f"port '{name}'")
                    system.add_port(PlatePort(name, face, edges, start, width, port['mode_count'], walls,
                                              incidents=port['incidents'], mirror_walls=mirror_walls,
                                              tension_tolerance=port['tension_tolerance']))
                    continue
                free_walls = self._free_walls(cross_axis, start, start + width, f"port '{name}'")
                incidents, scale = self._folded_incidents(port, cross_axis, mirror_walls)
                system.add_port(MembranePort(name, face, edges, start, width, port['mode_count'],
                                             tension_tolerance=port['tension_tolerance'], free_walls=free_walls,
                                             incidents=incidents, mirror_walls=mirror_walls, drive_scale=scale))
        for name, (projector, _) in self._projections.items():
            end = projector.cross_start + projector.width
            projector.set_free_walls(self._free_walls(projector.cross_axis, projector.cross_start, end,
                                                      f"mode_projection '{name}'"))
            projector.mirror_walls = self._mirror_walls(projector.cross_axis, projector.cross_start, end)
        return system

    def _guide_wall_faces(self, cross_axis, start, end) -> tuple:
        """(window face at start or None, at end or None) of a guide cross-section [start, end] [m] along cross_axis
        (0 x / 1 y): the walls lying on the window edge."""
        x0, y0, x1, y1 = self.domain.mesh.window
        low, high = (x0, x1) if cross_axis == 0 else (y0, y1)
        tolerance = 1e3 * MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
        axis = 'xy'[cross_axis]
        return (f'{axis}_lo' if abs(float(start) - low) <= tolerance else None,
                f'{axis}_hi' if abs(float(end) - high) <= tolerance else None)

    def _free_walls(self, cross_axis, start, end, what) -> tuple:
        """(free at start, free at end) of a guide cross-section [start, end] [m] along cross_axis (0 x / 1 y): a wall on
        a window face with symmetry and no hold (clamped / simply supported / antisymmetry) is free, any other clamped."""
        mirrors = self._face_symmetries - self._face_clamps - self._face_supports
        free_walls = tuple(face is not None and face in mirrors
                           for face in self._guide_wall_faces(cross_axis, start, end))
        if all(free_walls):
            raise ValueError(f'{self._label()}: {what} spans two symmetry faces: a guide with two free walls has a '
                             'constant mode (not supported); hold one wall')
        return free_walls

    def _mirror_walls(self, cross_axis, start, end) -> tuple:
        """(sign or None at start, at end) of a guide's walls on mirror faces: +1 a free symmetry face (even), -1 an
        antisymmetry face or a held symmetry face (odd: w = 0); ports.unfolded_modes numbers its modes from them."""
        mirrors = self._face_symmetries | self._face_odd_mirrors
        held = self._face_clamps | self._face_supports
        return tuple(None if face is None or face not in mirrors else (-1 if face in held else 1)
                     for face in self._guide_wall_faces(cross_axis, start, end))

    def _port_elements(self, edges) -> np.ndarray:
        """Window element of each port edge (the cut whose medium the port uses)."""
        pairs = self.domain.mesh.edge_elements[edges]
        return np.where(self._window[pairs[:, 0]], pairs[:, 0], pairs[:, 1])

    def _plate_port(self, edges, coefficients) -> bool:
        """A port is a plate port (plate_ports.PlatePort) when any of its elements has bending_stiffness != 0."""
        return bool(np.any(coefficients.bending[self._port_elements(edges)] != 0.0))

    def _plate_wall_kinds(self, cross_axis, start, end, edges, what) -> tuple:
        """Strip wall kinds ('clamped' | 'supported' | 'symmetry' | 'free', at start, at end) of a plate port guide: the
        hold of the window face a wall lies on, else of the outline edges at the guide's corner node (none: free)."""
        mesh = self.domain.mesh
        nodes = np.unique(mesh.edges[edges])
        along = mesh.points[nodes, cross_axis]
        corners = (int(nodes[np.argmin(along)]), int(nodes[np.argmax(along)]))
        empty = np.zeros(0, dtype=np.int64)
        held = [edges_of for edges_of, condition in self._edge_conditions if condition.kind == 'dirichlet']
        held = np.concatenate(held) if held else empty
        slopes = np.concatenate(self._slope_edge_sets) if self._slope_edge_sets else empty

        def touches(edge_set, corner):
            return bool(edge_set.size) and bool((mesh.edges[edge_set] == corner).any())
        kinds = []
        for face, corner in zip(self._guide_wall_faces(cross_axis, start, end), corners):
            if face is not None:
                if face in self.pml_faces:
                    raise ValueError(f'{self._label()}: {what}: a guide wall on the PML face {face} (the guide must end '
                                     'on held, mirror or free walls)')
                kinds.append('clamped' if face in self._face_clamps else 'supported' if face in self._face_supports
                             else 'symmetry' if face in self._face_symmetries else 'free')
                continue
            value_held, slope_held = touches(held, corner), touches(slopes, corner)
            kinds.append('clamped' if value_held and slope_held else 'supported' if value_held
                         else 'symmetry' if slope_held else 'free')
        return tuple(kinds)

    def _folded_incidents(self, port, cross_axis, mirror_walls) -> tuple:
        """([(half-guide mode, complex amplitude)], drive scale) of a port's incidents given in the unfolded numbering.
        On a guide with one mirror wall a mode of the other parity cannot enter (ValueError; a parity-sector copy,
        sector_drive, leaves it to the sector of its parity), and a sector copy drives 2 A (the sector mean is A)."""
        single = sum(wall is not None for wall in mirror_walls) == 1
        scale = 2.0 if single and port['sector_drive'] else 1.0
        folded = []
        for number, amplitude in port['incidents']:
            mode = folded_mode(mirror_walls, number)
            if mode is None:
                if port['sector_drive']:
                    continue
                side = 0 if mirror_walls[0] is not None else 1
                face = f"{'xy'[cross_axis]}_{('lo', 'hi')[side]}"
                parity = 'odd' if mirror_walls[side] > 0 else 'even'
                raise ValueError(f"{self._label()}: port '{port['name']}': incident mode {number} (numbered on the "
                                 f'unfolded guide) is {parity} about the mirror plane {face}, so it cannot enter this '
                                 'half model; use the study setting parity_sectors (each mode in the run of its '
                                 'parity) or the other mirror feature on that face')
            index, sign = mode
            if index > port['mode_count']:
                raise ValueError(f"{self._label()}: port '{port['name']}': incident mode {number} is mode {index} of "
                                 f"this half guide, beyond its mode_count {port['mode_count']}")
            folded.append((index, sign * scale * amplitude))
        return folded, scale

    def _apply_coefficients(self, system, coefficients) -> None:
        """Tension tensor (1 + j eta) per element, areal density as the mass, the active elements without holes, the
        bending rigidity D (1 + j eta), the port media."""
        n_elements = self._n_elements()
        dtype = np.complex128 if coefficients.lossy else np.float64
        tensor = np.zeros((n_elements, 2, 2), dtype=dtype)
        tensor[:, 0, 0] = coefficients.tension_xx
        tensor[:, 1, 1] = coefficients.tension_yy
        tensor[:, 0, 1] = tensor[:, 1, 0] = coefficients.tension_xy
        if coefficients.lossy:
            tensor *= (1.0 + 1j * coefficients.loss)[:, None, None]
        active = self._default_elements() & ~coefficients.hole
        if not active.any():
            raise ValueError(f'{self._label()}: every element of the solved region is a hole (no tension, no areal '
                             'density)')
        system.set_coefficients(a=tensor, b=0.0, mass=coefficients.density.copy(), damping=0.0)
        system.set_active(active)
        if self.bending:
            system.set_bending(coefficients.complex_bending(), coefficients.poisson)
        for port in system.ports:
            self._set_port_medium(port, coefficients)

    def _set_port_medium(self, port, coefficients) -> None:
        """Mean T_nn, T_ss (1 + j eta), rho_s (a plate port also D (1 + j eta), nu) of the port's elements (the cut);
        ValueError when the density, loss, D or nu varies along the port, or the tension spread / shear exceeds the
        port's tension_tolerance (above PORT_UNIFORM_RTOL within it: a RuntimeWarning and port.medium_warning)."""
        elements = self._port_elements(port.edges)
        plate = isinstance(port, PlatePort)
        if plate != self._plate_port(port.edges, coefficients):
            raise ValueError(f"{self._label()}: port '{port.name}': bending_stiffness at the port changed between 0 and "
                             '> 0 after the system was built (film and plate ports differ); rebuild the model')
        normal, transverse = ((coefficients.tension_xx, coefficients.tension_yy) if port.face[0] == 'x'
                              else (coefficients.tension_yy, coefficients.tension_xx))
        tensions = {'normal tension': normal[elements], 'transverse tension': transverse[elements]}
        others = {'areal density': coefficients.density[elements], 'loss factor': coefficients.loss[elements]}
        if plate:   # the strip modes take one D and nu
            others.update({'bending stiffness': coefficients.bending[elements],
                           'plate Poisson ratio': coefficients.poisson[elements]})
        for what, values in others.items():
            reference = (1.0 if what in ('loss factor', 'plate Poisson ratio')
                         else max(float(np.abs(values).max()), 1e-300))
            if float(np.ptp(values)) > PORT_UNIFORM_RTOL * reference:
                raise ValueError(f"{self._label()}: port '{port.name}': the {what} varies along the port (a port "
                                 'needs a uniform guide at its face)')
        spreads = {what: float(np.ptp(values)) / max(float(np.abs(values).max()), 1e-300)
                   for what, values in tensions.items()}
        spreads['shear tension'] = (float(np.abs(coefficients.tension_xy[elements]).max())
                                    / max(float(np.abs(tensions['normal tension']).max()), 1e-300))
        worst = max(spreads, key=spreads.get)
        spread, tolerance = spreads[worst], port.tension_tolerance
        text = (f'shear tension tension_xy reaches {spread:.3g} of the normal tension at the port (the profiles assume '
                'principal axes along the face)' if worst == 'shear tension'
                else f'the {worst} varies by {spread:.3g} (relative spread) along the port')
        if spread > tolerance:
            raise ValueError(f"{self._label()}: port '{port.name}': {text}, above its tension_tolerance {tolerance:.3g} "
                             '(a port needs a uniform guide at its face; raise tension_tolerance to use the mean '
                             'tension of the port cut, e.g. a tension coupled from a solid)')
        port.tension_spread = spread
        port.medium_warning = ''
        if spread > PORT_UNIFORM_RTOL:
            port.medium_warning = (f"port '{port.name}': {text}; accepted (tension_tolerance {tolerance:.3g}) with the "
                                   'mean over the port cut: expect reflections of the order of the spread')
            warnings.warn(f'{self._label()}: {port.medium_warning}', RuntimeWarning, stacklevel=2)
        factor = 1.0 + 1j * float(np.mean(others['loss factor']))
        medium = (float(np.mean(tensions['normal tension'])) * factor,
                  float(np.mean(tensions['transverse tension'])) * factor, float(np.mean(others['areal density'])))
        if plate:
            port.set_medium(*medium, rigidity=float(np.mean(others['bending stiffness'])) * factor,
                            poisson=float(np.mean(others['plate Poisson ratio'])))
        else:
            port.set_medium(*medium)

    def _pml(self, omega_reference, coefficients):
        """fem.pml ring stretch of the pml faces for the fastest wave of the ring at omega_reference (sqrt(T / rho_s),
        with bending the propagating plate wave: MembraneCoefficients.wave_speed); None without."""
        if not self.pml_faces:
            return None
        if not omega_reference > 0.0:
            raise ValueError(f'{self._label()}: a PML needs a reference frequency > 0: set target_frequency > 0 (or '
                             'pml_frequency) for an eigenfrequency study')
        ring = ~self._window & ~coefficients.hole
        speed = (coefficients.wave_speed(ring, omega_reference)
                 or coefficients.wave_speed(~coefficients.hole, omega_reference))
        if not speed > 0.0:
            raise ValueError(f'{self._label()}: the PML needs elements with areal_density > 0 and tension > 0 or '
                             'bending_stiffness > 0 (its strength is set for their wave speed)')
        return pml_ring(self.domain.mesh, speed, omega_reference, self.pml_reflection, faces=self.pml_faces)

    def _probe_value(self, probe, values) -> complex:
        return complex(values[probe.location])

    def _element_phasors(self) -> np.ndarray:
        """Complex128 w at every element's centroid (0 on holes and elements outside the solved region)."""
        coefficients = self._read_coefficients()
        solved = self._default_elements() & ~coefficients.hole
        return np.where(solved, self.values[self.full_dofs.element_dofs] @ self._centroid_basis, 0.0)

    def _publish(self) -> None:
        """MembraneBase._publish; on a cupy domain the centroid phasors are formed on the device (one upload of values,
        the element maps kept until the coefficients or dofs change; a direct cuDSS solve's device values when kept,
        else one upload of values): a sweep publishes each frequency its field policy asks for."""
        xp = self.xp
        if xp is np:
            return super()._publish()
        coefficients = self._read_coefficients()
        maps = self._device_phasor_maps
        if maps is None or maps[0] is not coefficients or maps[1] is not self.full_dofs:
            solved = self._default_elements() & ~coefficients.hole
            maps = self._device_phasor_maps = (coefficients, self.full_dofs, xp.asarray(self.full_dofs.element_dofs),
                                               xp.asarray(self._centroid_basis, dtype=np.float64), xp.asarray(solved))
        _, _, element_dofs, basis, solved = maps
        values = self._device_values if self._device_values is not None else xp.asarray(self.values)
        phasors = xp.where(solved, values[element_dofs] @ basis, 0.0)
        for name, values in (('w_re', phasors.real), ('w_im', phasors.imag), ('w_abs', xp.abs(phasors))):
            xp.copyto(self.fields[name], values.astype(DTYPE))

    # --- eigenfrequency: Bloch cells ---------------------------------------------------
    def solve_eigenfrequencies(self, n_resonances):
        """MembraneBase.solve_eigenfrequencies, on the Bloch-reduced pencil when an axis is periodic; refused with
        ports (their DtN is not linear in omega^2); then the gas damping of every squeeze film per resonance."""
        self._eigen_preflight()
        if not self.bloch_wavenumbers:
            solve = super().solve_eigenfrequencies(n_resonances)
            self._evaluate_squeeze_films(solve)
            return solve
        self.eigen_solve = None
        reference = self.pml_frequency if self.pml_frequency is not None else self.target_frequency
        system = self._ready_system(2.0 * math.pi * reference)
        solve_at, _ = self._shift_solver(system)
        self.eigen_solve = solve_at(self.target_frequency, n_resonances)
        self.selected_resonance = None
        return self.eigen_solve

    def _multi_shift_solve(self, reference, search):
        """MembraneBase._multi_shift_solve (search targets / band) with solve_eigenfrequencies' port refusal and
        squeeze films (not in a Bloch cell)."""
        self._eigen_preflight()
        solve = super()._multi_shift_solve(reference, search)
        self._evaluate_squeeze_films(solve)
        return solve

    def _eigen_preflight(self) -> None:
        """Refuse ports (their DtN is not linear in omega^2) and squeeze films in a Bloch cell; reset the film rows."""
        if self._ports:
            raise ValueError(f'{self._label()}: ports work in a frequency_response study only (their boundary terms '
                             'depend on the frequency)')
        for film in self.squeeze_films:
            film.rows = None
        if self.bloch_wavenumbers and self.squeeze_films:
            raise ValueError(f'{self._label()}: squeeze_film_damping is not available in a Bloch cell (periodic axes)')

    def _shift_solver(self, system):
        """MembraneBase._shift_solver, or on a Bloch cell that of _bloch_shift_solver."""
        if not self.bloch_wavenumbers:
            return super()._shift_solver(system)
        return self._bloch_shift_solver(system)

    def _bloch_shift_solver(self, system):
        """pencil_shift_solver of P^H K P, P^H M P (+ the bending seam terms) on the free Bloch masters."""
        eigen = system.eigen_pencil()
        pencil, dofs = system.pencil, system.dofs
        master, phase = bloch_masters(dofs, self.bloch_wavenumbers)
        free = np.zeros(dofs.n_dofs, dtype=bool)
        free[pencil.free] = True
        columns, prolongation = bloch_prolongation(master, phase, free)
        reduce = prolongation.tocsr()[pencil.free]
        stiffness = eigen.stiffness
        if self.bending:
            coefficients = self._read_coefficients()
            for axis, wavenumber in sorted(self.bloch_wavenumbers.items()):
                seam = assemble_seam_bending(dofs, coefficients.complex_bending(), coefficients.poisson, axis,
                                             seam_phase(dofs.mesh, axis, wavenumber), stretch=system.pml).tocsr()
                seam = seam[pencil.free][:, pencil.free]
                stiffness = stiffness + (seam if np.any(seam.data.imag) else seam.real)
        adjoint = reduce.conj().T.tocsr()
        reduced_stiffness = (adjoint @ stiffness @ reduce).tocsr()
        reduced_mass = (adjoint @ eigen.mass @ reduce).tocsr()
        kind = 'general' if np.iscomplexobj(reduce.data) else 'auto'
        return pencil_shift_solver(
            reduced_stiffness, reduced_mass, self.eigen_solver, pencil=kind, in_window=system.in_window[columns],
            energy_weights=np.asarray(abs(reduced_mass).sum(axis=1), dtype=np.float64).ravel(),
            field_of=lambda vector: eigen.spread(reduce @ vector))

    # --- squeeze films (squeeze_film.py) -------------------------------------------------
    def add_squeeze_film(self, film):
        """Register a squeeze_film.MembraneSqueezeFilm (evaluated after every eigenfrequency solve); returns it."""
        self.squeeze_films.append(film)
        return film

    def mark_odd_mirror_faces(self, faces) -> None:
        """Outer faces of the antisymmetry feature (held as simply supported): a squeeze film vents there."""
        self._face_odd_mirrors.update(face for face in faces if face in FACES)

    def _film_seals(self) -> tuple:
        """(always-sealed mesh edges: window edges of clamped / supported / symmetry faces and symmetry outlines, held
        edges sealed where membrane lies on both sides) of the squeeze films' 'open_edges' vents."""
        mesh = self.domain.mesh
        faces = self._face_clamps | self._face_symmetries | (self._face_supports - self._face_odd_mirrors)
        sealed = [window_face_edges(mesh, face) for face in sorted(faces)] + list(self._symmetry_edge_sets)
        holds = [edges for edges, condition in self._edge_conditions if condition.kind == 'dirichlet']
        empty = np.zeros(0, dtype=np.int64)
        return (np.concatenate(sealed) if sealed else empty), (np.concatenate(holds) if holds else empty)

    def _evaluate_squeeze_films(self, solve) -> None:
        """Rows per resonance of every squeeze film: the 1 m-peak mode on the film's own numbering, Re omega and m_eff =
        Re(psi^H M psi) with the pencil mass rho_s."""
        if not self.squeeze_films:
            return
        system = self.system
        coefficients = self._read_coefficients()
        membrane_active = self._default_elements() & ~coefficients.hole
        free_ids = system.global_ids[system.pencil.free]
        mass = system.pencil.part('mass')
        shapes, omegas, masses = [], [], []
        for resonance in solve.resonances:
            field = np.array(resonance.field, dtype=np.complex128)
            peak = field[int(np.argmax(np.abs(field)))] if field.size else 0.0
            shape = field / peak if peak != 0.0 else field
            free_shape = shape[free_ids]
            shapes.append(shape)
            omegas.append(float(resonance.omega.real))
            masses.append(float(np.real(np.vdot(free_shape, mass @ free_shape))))
        sealed, holds = self._film_seals()
        mesh = self.domain.mesh
        for film in self.squeeze_films:
            active = film.active_footprint(membrane_active, self._window, self.pml_faces)
            vents = film.vent_edges(mesh, active, sealed, holds, membrane_active)
            film.evaluate(film.build(mesh, active, vents, self.order), shapes, omegas, masses)

    # --- ports ----------------------------------------------------------------------
    def _record_frequency(self, frequency) -> None:
        for name, (projector, record) in self._projections.items():   # A4: from L V after a fields-free reduced solve
            record.record(frequency, self.system.linear_output(('mode_projection', name), projector.apply))
        system = self.system
        if system is None or not system.ports:
            return
        omega = 2.0 * math.pi * float(frequency)
        outputs = system.port_outputs(omega)
        incident = sum(output['incident_power'] for output in outputs.values())
        for name, output in outputs.items():
            amplitudes, powers = self._port_records[name]
            amplitudes.record(frequency, {f'mode_{number}': complex(value)
                                          for number, value in enumerate(output['outgoing'], start=1)})
            values = {f'power_{number}': complex(value) for number, value in enumerate(output['power'], start=1)}
            values['incident_power'] = complex(incident)
            values['port_incident_power'] = complex(output['incident_power'])   # this port's (sector combination)
            powers.record(frequency, values)

    def _reset_records(self) -> None:
        for amplitudes, powers in self._port_records.values():
            amplitudes.clear()
            powers.clear()
        for _, record in self._projections.values():
            record.clear()

    # --- membrane API (the definition callbacks) --------------------------------------
    def symmetry_edges(self, edges) -> None:
        """Even mirror plane on mesh edges (straight object outlines): with bending dw/dn = 0 on each active side."""
        self._slope_edge_sets.append(np.asarray(edges, dtype=np.int64))
        self._symmetry_edge_sets.append(np.asarray(edges, dtype=np.int64))
        self._discard_system()

    def clamp_edges(self, edges) -> None:
        """Hold w = 0 (with bending also dw/dn = 0 on each active side) on mesh edges (object outlines, the constraint
        edges of line objects)."""
        edges = np.asarray(edges, dtype=np.int64)
        self._edge_conditions.append((edges, _CLAMPED))
        self._slope_edge_sets.append(edges)
        self._discard_system()

    def support_edges(self, edges) -> None:
        """Hold w = 0 on mesh edges (object outlines, line objects), their slope free (an interior line: the slope
        stays continuous across it through the interior penalty)."""
        self._edge_conditions.append((np.asarray(edges, dtype=np.int64), _CLAMPED))
        self._discard_system()

    def add_edge_force(self, edges, force) -> None:
        """Add a transverse force per length [N/m] (complex amplitude) on mesh edges (object outlines)."""
        self._edge_conditions.append((np.asarray(edges, dtype=np.int64), HarmonicCondition('neumann', complex(force))))
        self._discard_system()

    def add_pressure(self, mask, pressure, phase=0.0) -> None:
        """Add a pressure amplitude [Pa] (number or per-element array) at phase [deg] on the window elements under the
        element mask (the PML ring takes none)."""
        elements = np.asarray(to_host(mask), dtype=bool).reshape(-1) & self._window
        values = np.asarray(to_host(pressure), dtype=np.float64)
        values = np.broadcast_to(values.reshape(-1) if values.ndim else values, elements.shape)
        self._pressure += np.where(elements, complex_amplitude(values, phase), 0.0)
        self._discard_system()

    def add_point_force(self, point, force, phase=0.0) -> None:
        """Add a transverse point force [N] at phase [deg] at (x, y) [m]."""
        self._point_forces.append((tuple(float(value) for value in point), complex(complex_amplitude(force, phase))))
        self._discard_system()

    def add_probe_node(self, name, node):
        """Harmonic probe of w at a mesh node (a point object's node)."""
        node = int(node)
        if not 0 <= node < self.domain.mesh.n_points:
            raise ValueError(f'membrane probe node {node} is not a mesh node')
        return self._new_probe(name, node)

    def add_port(self, name, face, mode_count, incident_mode=0, amplitude=0.0, phase=0.0,
                 tension_tolerance=PORT_UNIFORM_RTOL, incidents=None, sector_drive=False) -> str:
        """Port on an outer face (ports.py; with bending_stiffness > 0 at the face plate_ports.py, decided at the system
        build): mode_count strip modes (a wall on a symmetry face free) absorbed exactly, incident mode (0 = none; a
        film port numbers it on the unfolded guide when a wall is a mirror face) of amplitude [m] at phase [deg], or
        incidents [(mode, amplitude [m], phase [deg])]; tension spread accepted up to tension_tolerance; sector_drive:
        a parity-sector copy (_folded_incidents). Returns its name."""
        self._check_face(face)
        if any(face == port['face'] for port in self._ports):
            raise ValueError(f'{self._label()}: one port per face ({face} has one already)')
        if 'xy'.index(face[0]) in self.bloch_wavenumbers:
            raise ValueError(f'{self._label()}: port on the periodic face {face}')
        mode_count, tension_tolerance = int(mode_count), float(tension_tolerance)
        if not (math.isfinite(tension_tolerance) and tension_tolerance > 0.0):
            raise ValueError(f'{self._label()}: port tension_tolerance must be a finite value > 0, got '
                             f'{tension_tolerance!r}')
        if incidents is None:
            incidents = [(incident_mode, amplitude, phase)] if int(incident_mode) else []
        drives = []
        for number, value, angle in incidents:
            if float(number) != int(number) or int(number) < 1:
                raise ValueError(f"{self._label()}: port '{name}': incident modes are whole numbers >= 1, got "
                                 f'{number!r}')
            drives.append((int(number), complex(complex_amplitude(value, angle))))
        self._ports.append({'name': str(name), 'face': face, 'mode_count': mode_count, 'incidents': drives,
                            'tension_tolerance': tension_tolerance, 'sector_drive': bool(sector_drive)})
        modes = tuple(f'mode_{number}' for number in range(1, mode_count + 1))
        self._port_records[str(name)] = (
            HarmonicProbeRecord(modes, unit='m', label=f'outgoing mode amplitudes at {name}'),
            HarmonicProbeRecord(tuple(f'power_{number}' for number in range(1, mode_count + 1))
                                + ('incident_power', 'port_incident_power'), unit='W', label=f'mode powers at {name}'))
        self._discard_system()
        return str(name)

    def port_records(self, name) -> tuple:
        """(outgoing amplitude record, power record) of port `name`."""
        return self._port_records[name]

    def port_medium(self, name) -> dict:
        """{'tension_spread' (relative, NaN before a solve), 'tension_tolerance', 'medium_warning' ('' none),
        'free_walls' ([at start, at end]), 'mirror_walls' (sign or None each), 'sector_modes' / 'sector_signs'
        (ports.unfolded_modes; None for a plate port: no sector combination), 'sector_drive_scale', 'plate_walls'
        (strip wall kinds of a plate port, None for a film port)} of port `name` (the result extras; None before a
        solve)."""
        tolerance = next(port['tension_tolerance'] for port in self._ports if port['name'] == name)
        port = next((port for port in (self.system.ports if self.system is not None else ()) if port.name == name), None)
        if port is None:
            return {'tension_spread': math.nan, 'tension_tolerance': tolerance, 'medium_warning': '', 'free_walls': None,
                    'mirror_walls': None, 'sector_modes': None, 'sector_signs': None, 'sector_drive_scale': None,
                    'plate_walls': None}
        plate = isinstance(port, PlatePort)
        numbers, signs = (None, None) if plate else unfolded_modes(port.mirror_walls, port.mode_count)
        return {'tension_spread': float(port.tension_spread), 'tension_tolerance': tolerance,
                'medium_warning': port.medium_warning, 'free_walls': list(port.free_walls),
                'mirror_walls': list(port.mirror_walls), 'sector_modes': None if plate else numbers.tolist(),
                'sector_signs': None if plate else signs.tolist(), 'sector_drive_scale': port.drive_scale,
                'plate_walls': list(port.wall_kinds) if plate else None}

    def add_mode_projection(self, name, guide_axis, cross_start, cross_end, positions, mode_count) -> tuple:
        """Mode projection along guide_axis (0 x / 1 y) at positions [m], located once (ports.mode_projector) and
        recorded per driven frequency; returns (ModeProjector, ProjectionRecord): projector.apply(self.values) = the
        shown solution's a_n. Refused with bending (the plate guide modes are numeric and frequency-dependent)."""
        if self.bending:
            raise ValueError(f"{self._label()}: mode_projection '{name}' projects onto the film strip profiles; with "
                             'bending_stiffness the guide modes are the numeric plate strip modes (per frequency), '
                             'which it does not use: take ports or field probes')
        projector = mode_projector(self.domain.mesh, self.full_dofs, guide_axis, cross_start, cross_end, positions,
                                   mode_count)
        self._projections[str(name)] = (projector, ProjectionRecord())
        return self._projections[str(name)]

    # --- display ----------------------------------------------------------------------
    def renderer(self):
        return FEMFieldRenderer(self, 'membrane', MEMBRANE_FIELD_INFOS, lambda: shown_label(self))

    def field_dofs(self, name):
        """fem DofMap of state field or source slot `name` (the transfer contract): every one is per element (P0)."""
        if name not in self.state() and name not in self.source_slots:
            raise ValueError(f"{self._label()} has no state field or source slot '{name}'")
        return self.element_map

    def fem_field(self, component) -> tuple:
        """(DofMap, host coefficients): the w phasor parts on the full P1 / P2 map, tension / areal density per
        element."""
        if component in PHASOR_FIELD_NAMES:
            values = self.values
            parts = {'w_re': values.real, 'w_im': values.imag, 'w_abs': np.abs(values), 'w_phase': np.angle(values)}
            return self.full_dofs, parts[component]
        if component in MEMBRANE_FIELD_INFOS:
            return self.element_map, self._display_values(component)
        raise KeyError(f"{self._label()} has no field '{component}'")

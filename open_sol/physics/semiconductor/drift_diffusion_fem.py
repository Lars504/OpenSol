"""Drift-diffusion on a triangle mesh: the box method on the P1 nodes of the window elements (vertex-centred finite
volumes = edge-averaged finite elements). Edge coefficients from the affine P1 stiffness of each element (cot / 2
weights x the element's eps or mobility, Scharfetter-Gummel along the edge), control volumes = the lumped P1 mass;
the Newton core is DriftDiffusionCore. Host numpy float64."""
import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS0
from ...fem.assembly import p1_edge_weights, p1_gradients, p1_lumped_weights
from ...fem.diffusion import RING_ANGLE
from .drift_diffusion import DriftDiffusionCore, bernoulli

_NEGATIVE_WEIGHT_RTOL = 1e-9   # face coefficient below -this x the largest one counts as negative (non-Delaunay)


class FEMDriftDiffusionSystem(DriftDiffusionCore):
    """Drift-diffusion on the nodes of the window elements of a fem TriangleMesh: sites = those nodes, faces = mesh
    edges of window elements; doping / contacts / gates per element; outputs nodal and per element (full mesh)."""

    def __init__(self, mesh, label, tolerance=1e-8, max_newton_iterations=50, max_bias_step=0.1, axisymmetric=False,
                 linear_solver=None):
        super().__init__(label, tolerance, max_newton_iterations, max_bias_step, linear_solver)
        self.mesh = mesh
        self.axisymmetric = bool(axisymmetric)
        # whole rings in (r, z): every weight x 2 pi, so contact currents / gate charges are ring totals (A, C)
        self.ring_factor = RING_ANGLE if self.axisymmetric else 1.0
        self.elements = np.flatnonzero(mesh.pml_faces == 0)   # window elements (the PML ring is not solved)
        corner_nodes = mesh.triangles[self.elements]
        self.nodes = np.unique(corner_nodes)                    # site -> mesh node
        node_site = np.full(mesh.n_points, -1, dtype=np.int64)
        node_site[self.nodes] = np.arange(self.nodes.size)
        self.n_cells = int(self.nodes.size)
        self.element_sites = node_site[corner_nodes]            # (m, 3) site of local node k
        local_edges = mesh.element_edges[self.elements]         # (m, 3) mesh edge of local edge k (opposite node k)
        edge_ids = np.unique(local_edges)
        edge_face = np.full(mesh.n_edges, -1, dtype=np.int64)
        edge_face[edge_ids] = np.arange(edge_ids.size)
        self.element_faces = edge_face[local_edges]             # (m, 3) face of local edge k
        self.face_lo = node_site[mesh.edges[edge_ids, 0]]       # mesh edge order: lower node first
        self.face_hi = node_site[mesh.edges[edge_ids, 1]]
        self.n_faces = int(edge_ids.size)
        tangent = mesh.points[mesh.edges[edge_ids, 1]] - mesh.points[mesh.edges[edge_ids, 0]]
        self.face_length = np.hypot(tangent[:, 0], tangent[:, 1])   # edge length h [m] (lo -> hi)
        self.face_tangent = tangent / self.face_length[:, None]
        self.unit_weights = self.ring_factor * p1_edge_weights(mesh, self.elements, axisymmetric=self.axisymmetric)
        self.lumped = self.ring_factor * p1_lumped_weights(mesh, self.elements, axisymmetric=self.axisymmetric)
        self.cell_area = self._site_sum(self.lumped)            # control volume [m^2 per m; m^3 per ring]
        self.gradients, _ = p1_gradients(mesh, self.elements)   # (m, 3, 2): E = -grad psi per element
        # least-squares map of the three edge projections J . t_k to the element vector J (exact for constant J)
        tangents = self.face_tangent[self.element_faces]        # (m, 3, 2)
        self._tangent_solve = np.linalg.solve(np.einsum('mkc,mkd->mcd', tangents, tangents),
                                              np.transpose(tangents, (0, 2, 1)))   # (m, 2, 3)
        self.element_doping = np.zeros(mesh.n_elements)         # N_D - N_A [1/m^3] of the doping features
        self.element_auger = np.zeros((2, mesh.n_elements))     # Auger C_n, C_p [m^6/s] per mesh element
        self._held = []          # ('contact' | 'gate', index, host bool per mesh element), later entries win shared nodes
        self.negative_faces = 0  # faces with a negative Poisson or carrier coefficient at the last set_inputs

    # --- setup ------------------------------------------------------------------

    def _site_sum(self, local_values):
        """Per-site sum of (m, 3) element-corner values."""
        return np.bincount(self.element_sites.ravel(), np.asarray(local_values, dtype=np.float64).ravel(), self.n_cells)

    def _face_sum_of(self, local_values):
        """Per-face sum of (m, 3) element-edge values."""
        return np.bincount(self.element_faces.ravel(), np.asarray(local_values, dtype=np.float64).ravel(), self.n_faces)

    def add_doping(self, net_doping):
        """Add N_D - N_A [1/m^3] per mesh element (0 outside the feature)."""
        self.element_doping += np.asarray(net_doping, dtype=np.float64)

    def clear_doping(self):
        """Net doping 0 everywhere (DriftDiffusionPhysics.set_doping adds every feature again)."""
        self.element_doping[...] = 0.0

    def set_auger(self, mask, electron_coefficient, hole_coefficient):
        """Auger coefficients C_n, C_p [m^6/s] on the masked mesh elements (a later feature takes shared elements)."""
        self.element_auger[:, np.asarray(mask, dtype=bool)] = np.array([[electron_coefficient], [hole_coefficient]],
                                                                      dtype=np.float64)

    def add_contact(self, name, mask, voltage) -> int:
        """Hold every node of the masked semiconductor elements as an ohmic contact at `voltage` [V] (a later contact or
        gate takes shared nodes); returns its index."""
        index = len(self.contact_names)
        self.contact_names.append(name)
        self.contact_voltages.append(float(voltage))
        self._held.append(('contact', index, np.asarray(mask, dtype=bool).copy()))
        return index

    def add_gate(self, name, mask, voltage, work_function_difference) -> int:
        """Hold psi = voltage - work_function_difference [V] on every node of the masked (insulating) elements."""
        index = len(self.gate_names)
        self.gate_names.append(name)
        self.gate_voltages.append(float(voltage))
        self.gate_offsets.append(float(work_function_difference))
        self._held.append(('gate', index, np.asarray(mask, dtype=bool).copy()))
        return index

    # --- inputs -------------------------------------------------------------------

    def set_inputs(self, materials, generation, thermal_voltage):
        """Per-solve inputs: materials {schema array: float64 per mesh element}, generation [1/(m^3 s)] per mesh element
        and V_T [V]; node values are semiconductor-volume means of the element values. ValueError for invalid material
        values or held regions."""
        self.vt = float(thermal_voltage)
        elements = self.elements
        values = {name: np.asarray(array, dtype=np.float64)[elements] for name, array in materials.items()}
        semiconductor_elements = values['n_intrinsic'] > 0.0
        if not semiconductor_elements.any():
            raise ValueError(f'{self.label()}: no window element has n_intrinsic > 0 (assign a semiconductor material)')
        for name in ('mobility_n', 'mobility_p', 'radiative_recombination'):
            if np.any(values[name][semiconductor_elements] < 0.0):
                raise ValueError(f'{self.label()}: {name} must be >= 0 in every semiconductor element')
        for name in ('lifetime_n', 'lifetime_p'):
            if not np.all(values[name][semiconductor_elements] > 0.0):
                raise ValueError(f'{self.label()}: {name} must be > 0 in every semiconductor element')
        self.semiconductor_elements = semiconductor_elements
        semiconductor_weight = np.where(semiconductor_elements[:, None], self.lumped, 0.0)
        semiconductor_volume = self._site_sum(semiconductor_weight)
        semiconductor = semiconductor_volume > 0.0
        safe_volume = np.where(semiconductor, semiconductor_volume, 1.0)

        def node_mean(element_values):
            """Semiconductor-volume mean at the sites (0 off the semiconductor)."""
            total = self._site_sum(semiconductor_weight * np.where(semiconductor_elements, element_values, 0.0)[:, None])
            return np.where(semiconductor, total / safe_volume, 0.0)

        self.semiconductor = semiconductor
        self.charge_fraction = semiconductor_volume / self.cell_area
        self.n_intrinsic = node_mean(values['n_intrinsic'])
        self.doping = node_mean(self.element_doping[elements])
        self.lifetime_n = np.where(semiconductor, node_mean(values['lifetime_n']), 1.0)
        self.lifetime_p = np.where(semiconductor, node_mean(values['lifetime_p']), 1.0)
        self.radiative = node_mean(values['radiative_recombination'])
        self.auger_n, self.auger_p = (node_mean(coefficients[elements]) for coefficients in self.element_auger)
        self._resolve_held(semiconductor_elements)

        # Poisson: P1 stiffness of eps0 diag(eps_xx, eps_yy) on every window element (gates are held nodes)
        permittivity = EPS0 * np.stack([values['eps_dc_xx'], values['eps_dc_yy']], axis=1)
        self.poisson_conductance = self._face_sum_of(self.ring_factor * p1_edge_weights(
            self.mesh, elements, permittivity, axisymmetric=self.axisymmetric))
        # SG face coefficients d = V_T sum over the edge's elements of mu_K w_K [m^2/s] (edge-averaged FE)
        self.mobility_n = np.where(semiconductor_elements, values['mobility_n'], 0.0)
        self.mobility_p = np.where(semiconductor_elements, values['mobility_p'], 0.0)
        self.carrier_n = self.vt * self._face_sum_of(self.mobility_n[:, None] * self.unit_weights)
        self.carrier_p = self.vt * self._face_sum_of(self.mobility_p[:, None] * self.unit_weights)
        self.negative_faces = int(sum(np.count_nonzero(face < -_NEGATIVE_WEIGHT_RTOL * max(float(np.max(face)), 0.0))
                                      for face in (self.poisson_conductance, self.carrier_n, self.carrier_p)))
        # generation of the free semiconductor elements, lumped to the nodes, per control volume
        free_weight = np.where((semiconductor_elements & ~self.contact_elements)[:, None], self.lumped, 0.0)
        element_generation = np.asarray(generation, dtype=np.float64)[elements]
        self.generation = self._site_sum(free_weight * element_generation[:, None]) / self.cell_area
        self._finish_inputs()

    def _resolve_held(self, semiconductor_elements):
        """contact / gate index per site from the held element masks in feature order (later ones win shared nodes);
        contact_elements = the semiconductor elements of the contacts."""
        contact_index = np.full(self.n_cells, -1, dtype=np.int64)
        gate_index = np.full(self.n_cells, -1, dtype=np.int64)
        self.contact_elements = np.zeros(self.elements.size, dtype=bool)
        for kind, index, mask in self._held:
            selected = mask[self.elements]
            if kind == 'contact':
                selected &= semiconductor_elements
                self.contact_elements |= selected
                sites = np.unique(self.element_sites[selected])
                contact_index[sites], gate_index[sites] = index, -1
            else:
                if np.any(selected & semiconductor_elements):
                    raise ValueError(f"{self.label()}: gate '{self.gate_names[index]}' selects semiconductor elements "
                                     '(n_intrinsic > 0); a gate is an electrode on an insulator (use an ohmic contact '
                                     'on a semiconductor)')
                sites = np.unique(self.element_sites[selected])
                gate_index[sites], contact_index[sites] = index, -1
        for index, name in enumerate(self.contact_names):
            if not np.any(contact_index == index):
                raise ValueError(f"{self.label()}: ohmic contact '{name}' holds no semiconductor element (n_intrinsic "
                                 '> 0); select elements of a semiconductor material')
        self.contact_index, self.gate_index = contact_index, gate_index
        self.contact_cells = contact_index >= 0
        self.gate_cells = gate_index >= 0

    # --- outputs --------------------------------------------------------------------

    def outputs(self) -> tuple:
        """(nodal, element) float64 arrays of the last solve on the full mesh: nodal V, N, P, phi_n, phi_p (0 on nodes
        of no window element); per element E, Jn, Jp, J (x, y), net_doping, space_charge and recombination (0 off the
        window / semiconductor elements; J from the element's own Scharfetter-Gummel edge currents)."""
        mesh = self.mesh
        state = self._require_state()
        psi = state[:, 0] * self.vt
        electrons, holes = state[:, 1] * self.scale, state[:, 2] * self.scale
        phi_n, phi_p = self._quasi_fermi(psi, electrons, holes)
        nodal = {}
        for name, values in (('V', psi), ('N', electrons), ('P', holes), ('phi_n', phi_n), ('phi_p', phi_p)):
            nodal[name] = np.zeros(mesh.n_points)
            nodal[name][self.nodes] = values

        sites, semiconductor = self.element_sites, self.semiconductor_elements
        field = -np.einsum('mk,mkc->mc', psi[sites], self.gradients)
        u = state[:, 0]
        delta = (u[self.face_hi] - u[self.face_lo])[self.element_faces]               # (m, 3)
        b_plus = bernoulli(delta)
        b_minus = b_plus + delta
        lo, hi = self.face_lo[self.element_faces], self.face_hi[self.element_faces]
        length = self.face_length[self.element_faces]
        # particle flux densities along t (lo -> hi) of the element's own SG currents; J_n = -q F_n, J_p = q F_p
        flux_n = self.vt * self.mobility_n[:, None] * (electrons[lo] * b_minus - electrons[hi] * b_plus) / length
        flux_p = self.vt * self.mobility_p[:, None] * (holes[lo] * b_plus - holes[hi] * b_minus) / length
        current_n = -ELEMENTARY_CHARGE * np.einsum('mck,mk->mc', self._tangent_solve, flux_n)
        current_p = ELEMENTARY_CHARGE * np.einsum('mck,mk->mc', self._tangent_solve, flux_p)
        rate, _, _ = self._recombination(state[:, 1], state[:, 2], weight=self.semiconductor.astype(np.float64))
        lumped = self.lumped
        # element R = lumped-weighted mean of the node rates: its integral is the sum the continuity rows use
        recombination = np.where(semiconductor, (lumped * rate[sites]).sum(axis=1) / lumped.sum(axis=1), 0.0)
        element_density = np.where(semiconductor, (holes[sites] - electrons[sites]).mean(axis=1), 0.0)
        doping = np.where(semiconductor, self.element_doping[self.elements], 0.0)
        local = {'E_x': field[:, 0], 'E_y': field[:, 1],
                 'Jn_x': current_n[:, 0], 'Jn_y': current_n[:, 1], 'Jp_x': current_p[:, 0], 'Jp_y': current_p[:, 1],
                 'J_x': current_n[:, 0] + current_p[:, 0], 'J_y': current_n[:, 1] + current_p[:, 1],
                 'net_doping': doping, 'space_charge': ELEMENTARY_CHARGE * (element_density + doping),
                 'recombination': recombination * self.scale}
        element = {}
        for name, values in local.items():
            element[name] = np.zeros(mesh.n_elements)
            element[name][self.elements] = np.where(semiconductor | name.startswith('E_'), values, 0.0)
        return nodal, element

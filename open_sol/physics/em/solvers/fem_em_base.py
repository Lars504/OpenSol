"""Shared contract of the EM physics on a TriangleDomain (em_modes, em_fdfd_ez / _hz, em_fdfd_bor and em_fdfd_conical
on a 'triangles' mesh): TriangleEMPhysics holds the per-element state, slots, host materials, the PEC-keyed layout,
poles / PEC masks and the shared triangle renderer; subclasses add the solve and display hooks."""
import math

import numpy as np

from ....core.backend import to_host
from ....core.constants import C0, DTYPE
from ....core.solver_base import PhysicsSolver
from ....fem.dofs import lagrange_dofs
from ....fem.renderer import FEMFieldRenderer
from ....solvers.arpack_shift_invert import ArpackShiftInvertSolver
from ....solvers.sparse_direct import scipy_available
from ..fem_modes import size_issue_for_unknowns
from ..harmonic import cell_permeability, cell_permittivity, write_float32
from ..materials import DrudePole, LorentzPole, register_em_arrays


class TriangleEMPhysics(PhysicsSolver):
    """Contract members of an EM physics on a fem.domain.TriangleDomain: float32 per-element state (element_fields) and
    permittivity slots (P0 field_dofs), nodal fem_field of the shown solution, moot subpixel / conformal options."""
    # Subclasses set the class attributes below, call _init_triangle in __init__ and implement _build_layout(pec),
    # _has_shown_field(), _shown_nodal_fields(nodal_dofs) and _last_unknowns(); _system_changed() is their dirty hook.
    # Every solve re-reads the MaterialGrid; the layout is rebuilt only when the PEC elements change.

    name = 'em'
    type_label = ''            # model type name in messages ('em_modes', ...)
    tensor_class = ''          # the tensor-grid physics class (constructor message)
    boundary_hint = ''         # extra words of the missing boundary_config message
    state_fields = ()          # per-element float32 state fields of the shown solution
    permittivity_arrays = ()   # MaterialGrid permittivity arrays the solve reads
    permittivity_slots = {}    # source slot -> permittivity array
    material_names = ()        # renderer material fields served per element (P0)
    renderer_field_infos = {}  # FieldInfos of the shared triangle renderer
    size_remedy = None         # closing advice of the host SuperLU size rule (None: fem_modes' default)

    def _init_triangle(self, domain, boundary_config, polarization, schema_key, order, eigen_solver=None):
        """Check the mesh kind, boundary_config and order; set the shared attributes and the per-element state."""
        label = type(self).__name__
        if getattr(domain, 'grid_kind', 'tensor') != 'triangles':
            raise ValueError(f'{label} needs a fem.domain.TriangleDomain (a triangles mesh); tensor grids take '
                             f'{self.tensor_class}')
        if boundary_config is None:
            raise ValueError(f'{label} needs a boundary_config (faces and PML settings{self.boundary_hint})')
        boundary_config._validate()
        if order not in (1, 2):
            raise ValueError(f'{label} order must be 1 (linear) or 2 (quadratic), got {order!r}')
        self.domain = domain
        self.mesh = domain.mesh
        self.xp = domain.xp
        self.polarization = polarization
        self.boundary_config = boundary_config
        self.order = int(order)
        self._private_solvers = []    # solvers built here: their work counters are this physics'
        if eigen_solver is None:
            eigen_solver = ArpackShiftInvertSolver()
            self._private_solvers.append(eigen_solver)
        self.eigen_solver = eigen_solver
        register_em_arrays(domain.materials, schema_key)

        self.sources = []             # placement / Visualizer protocol
        self.monitors = []
        self.dispersive_regions = []  # (pole, None, element mask | None): harmonic.cell_permittivity's region form
        self.pec_cells = None         # run-backend bool (n_elements,) PEC elements
        self.finalize_warnings = []
        self.runtime_warnings = []
        self.enable_q_accum = False   # loss-protocol flag set by the runner
        self.conformal_locator = None   # never set: the elements follow the drawn outline (apply_pec duck-types it)
        self.solved_omega = None
        self.n_solves = 0
        self.last_solve_seconds = None
        self.host_factor_bytes = None
        self._layout = None           # (PEC key, layout of the subclass' FEM module)
        self._element_dofs = None     # P0 DofMap of the state fields / slots
        self._nodal_dofs = None       # full-mesh Lagrange DofMap of the element degree (fem_field)
        self._nodal = None            # host float64 nodal renderer fields of the shown solution
        self.element_fields = {name: self.xp.zeros(self.mesh.n_elements, dtype=DTYPE) for name in self.state_fields}

    def _omega_of(self, wavelength) -> float:
        """Angular frequency [rad/s] of a vacuum wavelength [m]; ValueError unless finite and > 0."""
        wavelength = float(wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f'{type(self).__name__} wavelength must be finite and > 0 [m], got {wavelength!r}')
        return 2.0 * math.pi * C0 / wavelength

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Nothing: solved only by the study hooks."""
        return

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Float32 per-element state fields of the shown solution, the permittivity arrays, sigma_e and 'q_accum': None
        (loss-protocol key, never a field)."""
        materials = self.domain.materials
        state = dict(self.element_fields)
        for name in self.permittivity_arrays:
            state[name] = getattr(materials, name)
        state['sigma_e'] = materials.sigma_e
        state['q_accum'] = None
        return state

    def _slot_names(self) -> tuple:
        return tuple(self.permittivity_slots)

    def field_dofs(self, name):
        """fem DofMap of state field or source slot `name` (the transfer contract): one value per element (P0 on all
        elements); ValueError for other names."""
        if name not in (*self.element_fields, *self.permittivity_arrays, 'sigma_e', *self._slot_names()):
            raise ValueError(f"{self.type_label} on triangles has no field or slot '{name}'")
        if self._element_dofs is None:
            self._element_dofs = lagrange_dofs(self.mesh, 0)
        return self._element_dofs

    def fem_field(self, component) -> tuple:
        """(Lagrange DofMap, host float64 coefficients) of a renderer field (fem.renderer contract): materials and
        absorbed_power per element (P0); solution fields at the nodes of the element degree (means over the non-PEC
        elements sharing a node; zeros before a solution is shown)."""
        if component in self.material_names:
            return self.field_dofs(component), np.asarray(to_host(getattr(self.domain.materials, component)),
                                                          dtype=np.float64)
        if component == 'absorbed_power' and 'q_abs' in self.element_fields:
            return self.field_dofs('q_abs'), np.asarray(to_host(self.element_fields['q_abs']), dtype=np.float64)
        if self._nodal_dofs is None:
            self._nodal_dofs = lagrange_dofs(self.mesh, self.order)
        if not self._has_shown_field():
            return self._nodal_dofs, np.zeros(self._nodal_dofs.n_dofs)
        if self._nodal is None:
            self._nodal = self._shown_nodal_fields(self._nodal_dofs)
        if component not in self._nodal:
            raise ValueError(f"{self.type_label} on triangles has no field '{component}'")
        return self._nodal_dofs, self._nodal[component]

    def inject_source(self, name: str, array):
        """Copy a coupling's per-element array into the permittivity array of slot `name`; the next solve reads it."""
        if name not in self.permittivity_slots:
            raise ValueError(f"{type(self).__name__} accepts sources {', '.join(self.permittivity_slots)}; got "
                             f"'{name}'")
        self.xp.copyto(getattr(self.domain.materials, self.permittivity_slots[name]),
                       self.xp.asarray(array, dtype=DTYPE))
        self._system_changed()

    def finalize(self) -> None:
        """ValueError without the solver back end (scipy) or over the host SuperLU size rule (the layout is built here;
        none for a cuDSS eigen factorization); finalize_warnings = its size warning."""
        self.finalize_warnings = []
        self._check_back_end()
        issue = self._size_issue(self._current_layout())
        if issue is not None:
            if issue[0] == 'error':
                raise ValueError(issue[1])
            self.finalize_warnings.append(issue[1])

    def materials_changed(self, names) -> None:
        return   # every solve reads the MaterialGrid afresh

    def renderer(self):
        """Shared triangle renderer with renderer_field_infos; time label = selection_label."""
        return FEMFieldRenderer(self, 'em', self.renderer_field_infos, self.selection_label)

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: per-element q_abs [W/m^3] of the shown solution (float32 run backend), None before one
        (or without q_abs); merge_previous is ignored."""
        if 'q_abs' not in self.element_fields or not self._has_shown_field():
            return None
        return self.element_fields['q_abs']

    def set_subpixel_smoothing(self, geometry):
        """Ignored: the elements follow the material outlines (subpixel smoothing is moot on triangles)."""
        return

    def set_conformal_boundaries(self, locator):
        """Ignored: PEC elements end on the drawn outline (conformal boundaries are moot on triangles)."""
        return

    def solve_summary(self):
        """None: no driven solve (an eigen-solve summary is EigenSolve.summary)."""
        return None

    def work_counters(self):
        """Solves so far, the sizes of the last one (unknowns, elements) and the private solvers' counts."""
        counters = {'solves': self.n_solves, **self._extra_counters()}
        unknowns = self._last_unknowns()
        if unknowns is not None:
            counters.update(unknowns=int(unknowns), elements=int(self.mesh.n_elements))
        for solver in self._private_solvers:
            counters.update(solver.work_counters())
        return counters

    def _extra_counters(self) -> dict:
        return {}

    # --- hooks and helpers of the subclasses ---

    def _system_changed(self):
        """Called after a slot write, pole or PEC change (a cached assembly is stale)."""
        return

    def _check_back_end(self):
        if not scipy_available():
            raise ValueError(f'{self.type_label} needs scipy (host sparse eigensolver), which is not installed: pip '
                             'install scipy')

    def _layout_unknowns(self, layout) -> int:
        return layout.n_unknowns

    def _size_issue(self, layout):
        """(level, text) of the host SuperLU size rule of the eigen factorization, None when fine or for cuDSS."""
        remedy = () if self.size_remedy is None else (self.size_remedy,)
        return size_issue_for_unknowns(self._layout_unknowns(layout),
                                       getattr(self.eigen_solver, 'factorization', 'superlu'), *remedy)

    def _host_pec(self):
        """Host bool (n_elements,) PEC elements."""
        return (np.zeros(self.mesh.n_elements, dtype=bool) if self.pec_cells is None
                else np.array(to_host(self.pec_cells), dtype=bool))

    def _current_layout(self, pec=None):
        """Layout of the current (or given host) PEC elements, rebuilt only when they change."""
        pec = self._host_pec() if pec is None else pec
        key = pec.tobytes()
        if self._layout is None or self._layout[0] != key:
            self._layout = (key, self._build_layout(pec))
        return self._layout[1]

    def _element_materials(self, omega):
        """({permittivity array: host complex128 eps per element (sigma_e, poles)}, host complex128 mu (sigma_m), host
        bool PEC elements) at omega."""
        materials, xp = self.domain.materials, self.xp
        eps = {name: np.asarray(to_host(cell_permittivity(materials, name, self.dispersive_regions, omega, xp=xp)),
                                dtype=np.complex128) for name in self.permittivity_arrays}
        mu = np.asarray(to_host(cell_permeability(materials, omega, xp=xp)), dtype=np.complex128)
        return eps, mu, self._host_pec()

    def _write_element_fields(self, host_values: dict):
        """Write host per-element values into element_fields (float32, one transfer each) and reset the nodal cache."""
        for name, values in host_values.items():
            write_float32(self.element_fields[name], values)
        self._nodal = None

    def _resonance_label(self):
        """'resonance k: <resonance text>' while a resonance is shown, else None."""
        if self.selected_resonance is None:
            return None
        resonance = self.eigen_solve.resonances[self.selected_resonance - 1]
        return f'resonance {self.selected_resonance}: {resonance.text()}'

    # --- material helpers ---

    def add_dispersive(self, pole, mask):
        """Record a Drude / Lorentz pole on a run-backend bool per-element mask (None = everywhere): its chi(omega) joins
        every solve; the permittivity arrays stay the background."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        if mask is None:
            self.dispersive_regions.append((pole, None, None))
        else:
            self._check_element_mask(mask, 'Pole mask')
            if bool(mask.any()):
                self.dispersive_regions.append((pole, None, mask.copy()))
        self._system_changed()

    def add_pec_region(self, mask):
        """Perfect electric conductor on a run-backend bool per-element mask: those elements leave the unknowns, their
        closure holds tangential E at 0 (exact on curved outlines)."""
        self._check_element_mask(mask, 'PEC mask')
        if self.pec_cells is None:
            self.pec_cells = mask.copy()
        else:
            self.pec_cells |= mask
        self._system_changed()

    def _check_element_mask(self, mask, what):
        xp = self.xp
        if not isinstance(mask, xp.ndarray) or mask.dtype != np.bool_ or mask.shape != (self.mesh.n_elements,):
            raise TypeError(f"{what} must be a {getattr(xp, '__name__', 'cupy')} bool array of shape "
                            f'({self.mesh.n_elements},) (one value per element), got {type(mask).__name__} '
                            f"{getattr(mask, 'dtype', '')} {getattr(mask, 'shape', '')}")

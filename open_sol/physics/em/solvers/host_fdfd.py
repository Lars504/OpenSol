"""Shared members of the host-solved Yee harmonic physics (em_fdfd_bor's BORPhysics, em_fdfd_conical's ConicalPhysics):
HostFDFDBase holds omega, the float32 display fields and state, the permittivity slots, the loss protocol, the
stationary / frequency_domain hooks, labels, counters, poles / PEC masks and the kept driven factor (GMRES recycling
after a material change, cuDSS factors kept for their plan)."""
import math

from ....core.constants import C0, DTYPE
from ....core.wavelengths import wavelength_label
from ....solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver
from ....solvers.host_direct import is_host_direct_factor
from ....solvers.sparse_direct import (CUDSS_BACK_ENDS, SYMMETRIC_OPTION_BACK_ENDS, SparseDirectSolver,
                                       scipy_available)
from ..harmonic import write_float32
from ..materials import DrudePole, LorentzPole
from .fdtd_2d_base import _check_cell_array, add_pec_cells, record_dispersive_region

SUPERLU_TEXT = 'host SuperLU'
# MaterialGrid arrays the assembly reads (a change re-reads the materials)
SYSTEM_MATERIALS = frozenset({'eps_r', 'eps_xx', 'eps_yy', 'sigma_e', 'mu_r', 'sigma_m'})


def is_cudss(factorization) -> bool:
    """Whether a factor is a cudss_factor.CudssFactorization (both precisions: back_end 'cudss')."""
    return getattr(factorization, 'back_end', None) == 'cudss'


def direct_text(factorization):
    """solve_summary text of a 'linear' solver factor: its cuDSS / MUMPS / PARDISO description, else host SuperLU."""
    if is_cudss(factorization) or is_host_direct_factor(factorization):
        return factorization.description
    return SUPERLU_TEXT


class HostFDFDBase:
    """Mixin of the host-solved harmonic contract; the class sets type_label / permittivity_slots / state_names and
    gives _solve(), _mark_assembly_dirty(omega_only=False) and _grid_size_issue()."""
    # Driven factor: _factorization (of system's operator itself while _factorization_current, else of an earlier one at
    # the same omega and active set: a GMRES preconditioner, sparse_direct RECYCLE_*); a replaced cuDSS factor is kept
    # as _cudss_previous and refactorised in place by the next factorisation (one plan while the pattern holds). MUMPS /
    # PARDISO factors take the same symmetric options (LDL^T) but are recyclable like SuperLU: never kept as previous.

    type_label = ''              # model type name in messages
    permittivity_arrays = ('eps_r', 'eps_xx', 'eps_yy')
    permittivity_slots = {}      # source slot -> permittivity array
    state_names = ()             # float32 full-grid display fields (fields)
    _line_block = None           # a LineBlockSolver 'shifted' role (BORPhysics), else None

    def _init_host_fdfd(self, domain, wavelength, linear_solver, eigen_solver):
        """omega of the wavelength setting, the 'linear' / 'eigen' solvers (private ones when None: work_counters
        reports them), the empty source / monitor / solve record and the zero display fields."""
        label = type(self).__name__
        wavelength = float(wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f'{label} wavelength must be finite and > 0 [m], got {wavelength!r}')
        self._omega = 2.0 * math.pi * C0 / wavelength
        # The build's 'linear' (sparse_direct: host SuperLU, cuDSS / MUMPS / PARDISO on their back ends, never
        # 'cusolver_qr') / 'eigen' (arpack_shift_invert) solvers; scripts pass nothing and get private ones, reported in
        # work_counters.
        self._private_solvers = []
        if linear_solver is None:
            linear_solver = SparseDirectSolver(back_end='scipy_splu')
            self._private_solvers.append(linear_solver)
        if linear_solver.back_end_choice == 'cusolver_qr':
            raise ValueError(f"{label} factorises a host matrix (SuperLU or cuDSS): a linear solver with back_end "
                             "'cusolver_qr' cannot serve it")
        if eigen_solver is None:
            eigen_solver = ArpackShiftInvertSolver()
            self._private_solvers.append(eigen_solver)
        self.linear_solver = linear_solver
        self.eigen_solver = eigen_solver

        self.sources = []
        self.monitors = []
        self.dispersive_regions = []
        self.pec_cells = None
        self.finalize_warnings = []
        self.runtime_warnings = []    # append-only; the studies log new entries
        self.enable_q_accum = False   # loss-protocol flag set by the runner; q_abs is computed by every solve
        self.solution = None          # host complex128 active vector of the shown driven solve
        self.centre_phasors = None    # host dict of complex128 (nx, ny) centre phasors of the shown solution
        self.solved_omega = None
        self.eigen_solve = None       # EigenSolve of the last solve_eigenfrequencies
        self.selected_resonance = None   # 1-based resonance shown, None = the driven solve
        self.host_factor_bytes = None   # host bytes of the driven factor (SuperLU L + U, MUMPS / PARDISO; memory_report)
        self.last_solve_seconds = None
        self.last_factor_seconds = None
        self.last_recycled_iterations = None   # GMRES iterations of the last solve if it reused an older factor
        self.n_solves = 0
        self._factorization = None
        self._factorization_current = False
        self._cudss_previous = None
        self._system_dirty = True
        self._solution_dirty = True
        shape = (domain.grid.nx, domain.grid.ny)
        # float32 full-grid run-backend fields, rewritten in place by every solve / selection (zeros before the first)
        self.fields = {name: self.xp.zeros(shape, dtype=DTYPE) for name in self.state_names}

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s] of the next solve (the wavelength setting, or the study)."""
        return self._omega

    @omega.setter
    def omega(self, omega):
        omega = float(omega)
        if not (math.isfinite(omega) and omega > 0.0):
            raise ValueError(f'{type(self).__name__} omega must be finite and > 0 [rad/s], got {omega!r}')
        if omega != self._omega:
            self._omega = omega
            self._mark_assembly_dirty(omega_only=True)

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Solve at self.omega when the system or the sources changed since the last solve (stationary study, flush)."""
        if self._solution_dirty:
            self._solve()

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Float32 full-grid e_intensity [V^2/m^2] and q_abs [W/m^3] of the shown solution, eps_r, eps_xx, eps_yy,
        sigma_e and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        return {'e_intensity': self.fields['e_intensity'], 'q_abs': self.fields['q_abs'], 'eps_r': materials.eps_r,
                'eps_xx': materials.eps_xx, 'eps_yy': materials.eps_yy, 'sigma_e': materials.sigma_e,
                'q_accum': None}

    def inject_source(self, name: str, array):
        """Copy a coupling's array into its permittivity array (permittivity_slots); the next solve re-reads it."""
        if name not in self.permittivity_slots:
            raise ValueError(f"{type(self).__name__} accepts sources {', '.join(self.permittivity_slots)}; got "
                             f"'{name}'")
        xp = self.xp
        xp.copyto(getattr(self.domain.materials, self.permittivity_slots[name]), xp.asarray(array, dtype=DTYPE))
        self._mark_assembly_dirty()

    def finalize(self) -> None:
        """ValueError without scipy or for a grid over the host solver's size limit (none for a cuDSS eigen
        factorization, a cuDSS linear back end or a line_block shifted solver: the driven / eigen solve keeps its own
        check); finalize_warnings = its warning."""
        self.finalize_warnings = []
        if not scipy_available():
            raise ValueError(f'{self.type_label} needs scipy (host sparse direct and eigen solvers), which is not '
                             'installed: pip install scipy')
        if (getattr(self.eigen_solver, 'factorization', 'superlu') in CUDSS_FACTORIZATIONS or self._linear_on_cudss
                or self._line_block is not None):
            return
        issue = self._grid_size_issue()
        if issue is not None:
            if issue[0] == 'error':
                raise ValueError(issue[1])
            self.finalize_warnings.append(issue[1])

    def materials_changed(self, names) -> None:
        if not SYSTEM_MATERIALS.isdisjoint(names):
            self._mark_assembly_dirty()

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: q_abs [W/m^3] of the last solve (float32 full-grid run backend, overwritten by the next), None
        before the first solve; merge_previous is ignored (no accumulation window)."""
        return None if self.solved_omega is None else self.fields['q_abs']

    def solve_harmonic(self, omega):
        """Solve now at `omega` [rad/s] (frequency_domain study hook) and record the monitors."""
        self.omega = omega
        self._solve()

    def selection_label(self):
        """'resonance k: lambda = ..., Q = ...' while a resonance is shown, else 'lambda = <nm> nm' of the last
        solve; None before any."""
        if self.selected_resonance is not None:
            resonance = self.eigen_solve.resonances[self.selected_resonance - 1]
            return f'resonance {self.selected_resonance}: {resonance.text()}'
        if self.solved_omega is None:
            return None
        return wavelength_label(2.0 * math.pi * C0 / self.solved_omega)

    def work_counters(self):
        """Driven solves so far (summary 'solvers' physics counters), plus the counters of private (script) linear /
        eigen solvers; a build's role solvers report their own."""
        counters = {'solves': self.n_solves}
        for solver in self._private_solvers:
            counters.update(solver.work_counters())
        return counters

    # --- driven factor ---

    @property
    def _linear_on_cudss(self) -> bool:
        """Whether the 'linear' solver resolves to a cuDSS back end."""
        return self.linear_solver.back_end in CUDSS_BACK_ENDS

    @property
    def _linear_symmetric_options(self) -> bool:
        """Whether the 'linear' solver's factor honours symmetric / symmetrizer / row_scale (cuDSS, MUMPS, PARDISO)."""
        return self.linear_solver.back_end in SYMMETRIC_OPTION_BACK_ENDS

    def _release_kept_cudss(self):
        """Release the cuDSS factor kept only for its plan."""
        if self._cudss_previous is not None:
            self._cudss_previous.release()
            self._cudss_previous = None

    def _keep_cudss_plan(self, factorization):
        """Keep a cuDSS factor no longer solved with as _cudss_previous (an older kept one released); others: no-op."""
        if not is_cudss(factorization) or factorization is self._cudss_previous:
            return
        if self._cudss_previous is not None:
            self._cudss_previous.release()
        self._cudss_previous = factorization

    def _retire_factorization(self):
        """Drop the driven factor; a cuDSS one stays as _cudss_previous for its plan."""
        factorization, self._factorization = self._factorization, None
        self._keep_cudss_plan(factorization)

    def _operator_replaced(self, same_layout):
        """A new operator: the kept factor stays as a recycling preconditioner when it is recyclable and same_layout
        (same omega, order / k_z and active set), else it is retired."""
        if self._factorization is not None and self._factorization.recyclable and same_layout:
            self._factorization_current = False
        else:
            self._retire_factorization()

    def _factorize_operator(self, operator, cudss_options):
        """Factorization of a host operator by the 'linear' solver: cuDSS (LDL^T with cudss_options() refactorising
        _cudss_previous), MUMPS / PARDISO (LDL^T with the same options; the kept cuDSS factor freed) or host SuperLU
        (the kept cuDSS factor freed; also after an 'auto' runtime fell back); host_factor_bytes (None on cuDSS) and
        last_factor_seconds."""
        linear = self.linear_solver
        if self._linear_symmetric_options:
            previous, self._cudss_previous = self._cudss_previous, None
            if not self._linear_on_cudss and previous is not None:   # host back end: no cuDSS plan to refactorise
                previous.release()
                previous = None
            factorization = linear.factorize_host(operator, symmetric=True, previous=previous, **cudss_options())
        else:
            self._release_kept_cudss()
            factorization = linear.factorize_host(operator)
        self.host_factor_bytes = None if is_cudss(factorization) else factorization.host_bytes
        self.last_factor_seconds = linear.last_factor_seconds
        return factorization

    def _solve_kept_factor(self, operator, rhs, initial_guess, factorize):
        """x of operator x = rhs: GMRES on the factor of an earlier operator while it reaches sparse_direct.RECYCLE_RTOL,
        else (re)factorise by factorize() and solve directly."""
        self.last_recycled_iterations = None
        if self._factorization is not None and not self._factorization_current:
            solution, iterations = self.linear_solver.solve_recycled_host(self._factorization, operator, rhs,
                                                                          initial_guess)
            if solution is not None:
                self.last_recycled_iterations = iterations
                return solution
            self._factorization = None   # too far from the factored operator: refactorise
        if self._factorization is None:
            self._factorization = factorize()
            self._factorization_current = True
        return self._factorization.solve_host(rhs)

    def _show_fields(self, host_fields, phasors, omega):
        """Write the display fields (float32, one host-to-device copy each) and keep the centre phasors."""
        for name, values in self.fields.items():
            write_float32(values, host_fields[name])
        self.centre_phasors = phasors
        self.solved_omega = omega

    # --- material helpers ---

    def add_dispersive(self, pole, mask):
        """Record a Drude / Lorentz pole on a run-backend bool cell mask (None = everywhere): its chi(omega) joins every
        solve (all three components); the permittivity arrays stay the background."""
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'add_dispersive needs a DrudePole or LorentzPole, got {type(pole).__name__}')
        if mask is not None:
            _check_cell_array(mask, self.fields['e_intensity'].shape, self.xp.bool_, 'Pole mask', xp=self.xp)
        record_dispersive_region(self.dispersive_regions, pole, mask)
        self._mark_assembly_dirty()

    def add_pec_region(self, mask):
        """Perfect electric conductor on a run-backend bool (nx, ny) cell mask (rings / z-invariant in 3D)."""
        _check_cell_array(mask, self.fields['e_intensity'].shape, self.xp.bool_, 'PEC mask', xp=self.xp)
        self.pec_cells = add_pec_cells(self.pec_cells, mask)
        self._mark_assembly_dirty()

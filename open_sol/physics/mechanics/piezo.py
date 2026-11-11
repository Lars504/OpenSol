"""Piezoelectric plane strain (piezoelectric plan stage 5, setting piezoelectric of solid_mechanics): a nodal Q1 potential
phi on the solid's own nodes, the symmetric coupled operator [[K_uu, K_uphi], [K_phiu, -K_phiphi]], electrodes, the
static host SuperLU solve, the coupled modal pencil and the harmonic solve; host numpy (scipy inside functions).
PiezoSystem and the electrode map work on dof indices, so the 3D tetrahedra solid (piezo_fem3d, block layout) shares
them."""
# 2D unknowns per node (u_x, u_y, phi), dof = 3 node + component (system.ElasticPattern with 3 components). Electric
# enthalpy h = 1/2 S c S + grad(phi) . e S - 1/2 grad(phi) . eps grad(phi), reduced Voigt S = (u_x,x, u_y,y, u_x,y +
# u_y,x), e the in-plane block [[e11, e12, e16], [e21, e22, e26]] [C/m^2] (T = c S - e^T E, D = e S + eps E, E = -grad
# phi), eps = eps0 eps_dc (clamped). Stationarity: K_uu u + K_uphi phi = f (u rows) and K_phiu u - K_phiphi phi = -q (phi
# rows = int grad N . D), so an electrode carries Q = -sum over its nodes of (A x)_phi [C/m] and a support the reaction
# (A x)_u - f. Element integrals exact on a x b rectangles: int N_k,x N_l,x = s_kx s_lx (1 + delta_y) b / (6 a), int
# N_k,y N_l,y = s_ky s_ly (1 + delta_x) a / (6 b), int N_k,x N_l,y = s_kx s_ly / 4 (as system.stiffness_stencil).
# Electrodes: 'voltage' holds phi on its nodes (the voltage in a static solve, 0 in a modal one: short circuit),
# 'floating' ties them to one master phi carrying the net charge (static) or 0 (modal: open circuit). The column map G
# (x = G x_r) does the tying and scales every phi by s = sqrt(max K_uu diagonal / max K_phiphi diagonal), so the reduced
# G^T A G has balanced blocks (c ~ 1e11 Pa against eps ~ 1e-10 F/m) for SuperLU's threshold pivoting; G^T M G = M (the
# mass has no phi part, so phi rows of the pencil are massless: ARPACK mode 3 takes a semi-definite M).
# 'resistive_load' (3D only): tied like a floating electrode, its master row closes through R to ground: j omega Q = -V /
# R, so the reduced master row gains j s^2 / (omega R) x_master (the eigenfrequency study: ModalPencil kind 'loaded'; the
# frequency response: HarmonicSystem's load term); a static solve refuses it.
import math
from dataclasses import dataclass, replace

import numpy as np

from ...core.constants import EPS0
from ...fem.assembly import congruence
from ...solvers.sparse_direct import factorize_host
from .frequency_response import HarmonicSystem
from .modal import ModalPencil, ModalSolve
from .system import LOCAL_NODES, ElasticPattern, mass_stencil, stiffness_stencil

PIEZO_COMPONENTS = 3
DISPLACEMENT_COMPONENTS = 2
ELECTRODE_CONNECTIONS = ('voltage', 'floating')
RESISTIVE_LOAD = 'resistive_load'   # 3D connection: PiezoSystem's load term, not in the 2D choices
_SLOPE_SIGN = (-1.0, 1.0)   # derivative sign of the lower / upper 1D linear shape function (as system)


@dataclass(frozen=True, eq=False)
class Electrode:
    """Runtime object of an electrode feature: name, its potential dofs (host int64, coupled numbering), connection
    ('voltage' | 'floating' | 'resistive_load'), voltage [V] (held, static solves), net charge [C/m 2D, C 3D] (floating,
    static solves) and load_resistance [Ohm] to ground (resistive_load)."""
    name: str
    dofs: np.ndarray
    connection: str
    voltage: float = 0.0
    charge: float = 0.0
    load_resistance: float = 50.0


def interleaved_layout(n_nodes):
    """(displacement dofs (2 n,), potential dofs (n,)) int64 of the 2D layout dof = 3 node + component, node-major."""
    nodes = np.arange(int(n_nodes), dtype=np.int64)
    displacement = (PIEZO_COMPONENTS * nodes[:, None] + np.arange(DISPLACEMENT_COMPONENTS)).reshape(-1)
    return displacement, PIEZO_COMPONENTS * nodes + 2


def node_potential_dofs(nodes):
    """Int64 2D potential dofs of a host bool node mask (any shape, C order = node index)."""
    return PIEZO_COMPONENTS * np.flatnonzero(np.asarray(nodes, dtype=bool).ravel()) + 2


def coupled_stencil(elastic, piezo, permittivity, dx_elements, dy_elements):
    """(nx_nodes, ny_nodes, 3, 3, 3, 3) float64 stencil [per m of depth] of the coupled operator from host window element
    arrays: elastic (c11, c12, c22, c66, c16, c26) [Pa], piezo (e11, e12, e16, e21, e22, e26) [C/m^2], relative
    permittivity (eps_xx, eps_yy, eps_xy), widths [m]."""
    c11, c12, c22, c66, c16, c26 = elastic
    n_x, n_y = c11.shape
    stencil = np.zeros((n_x + 1, n_y + 1, PIEZO_COMPONENTS, 3, 3, PIEZO_COMPONENTS), dtype=np.float64)
    stencil[:, :, :2, :, :, :2] = stiffness_stencil(c11, c12, c22, c66, dx_elements, dy_elements, np.float64, xp=np,
                                                    c16=c16, c26=c26)
    e11, e12, e16, e21, e22, e26 = piezo
    eps_xx, eps_yy, eps_xy = (EPS0 * values for values in permittivity)
    ratio_x = dy_elements[None, :] / (6.0 * dx_elements[:, None])   # b / (6 a)
    ratio_y = dx_elements[:, None] / (6.0 * dy_elements[None, :])   # a / (6 b)
    for k_x, k_y in LOCAL_NODES:
        target = stencil[k_x:k_x + n_x, k_y:k_y + n_y]
        for l_x, l_y in LOCAL_NODES:
            s_kx, s_ky, s_lx, s_ly = (_SLOPE_SIGN[index] for index in (k_x, k_y, l_x, l_y))
            mass_x = 2.0 if k_x == l_x else 1.0   # 1D mass factor (1 + delta)
            mass_y = 2.0 if k_y == l_y else 1.0
            di, dj = l_x - k_x + 1, l_y - k_y + 1
            gradient_xx = s_kx * s_lx * mass_y * ratio_x   # int N_k,x N_l,x
            gradient_yy = s_ky * s_ly * mass_x * ratio_y   # int N_k,y N_l,y
            gradient_xy = 0.25 * s_kx * s_ly               # int N_k,x N_l,y
            gradient_yx = 0.25 * s_ky * s_lx               # int N_k,y N_l,x
            # u test k, phi trial l: int grad N_l . e S(N_k e_a); the phi rows are its transpose
            target[:, :, 0, di, dj, 2] += e11 * gradient_xx + e16 * gradient_yx + e21 * gradient_xy + e26 * gradient_yy
            target[:, :, 1, di, dj, 2] += e12 * gradient_yx + e16 * gradient_xx + e22 * gradient_yy + e26 * gradient_xy
            target[:, :, 2, di, dj, 0] += e11 * gradient_xx + e16 * gradient_xy + e21 * gradient_yx + e26 * gradient_yy
            target[:, :, 2, di, dj, 1] += e12 * gradient_xy + e16 * gradient_xx + e22 * gradient_yy + e26 * gradient_yx
            target[:, :, 2, di, dj, 2] -= (eps_xx * gradient_xx + eps_yy * gradient_yy
                                           + eps_xy * (gradient_xy + gradient_yx))
    return stencil


def coupled_operator(elastic, piezo, permittivity, dx_elements, dy_elements):
    """Host scipy CSR over every dof (dof = 3 node + component (u_x, u_y, phi)) of the coupled operator (arguments as
    coupled_stencil). The loss operator of the frequency response is the same call with eta c, zero piezo and -tan_delta
    eps: [[eta K_uu, 0], [0, +tan_delta K_phiphi]]."""
    import scipy.sparse
    n_x, n_y = elastic[0].shape
    pattern = ElasticPattern(n_x + 1, n_y + 1, components=PIEZO_COMPONENTS, xp=np)
    dx_elements = np.asarray(dx_elements, dtype=np.float64)
    dy_elements = np.asarray(dy_elements, dtype=np.float64)
    return pattern.csr(coupled_stencil(elastic, piezo, permittivity, dx_elements, dy_elements), scipy.sparse)


def coupled_mass(density, dx_elements, dy_elements):
    """Host scipy CSR [kg/m per m of depth] over the coupled dofs: the consistent Q1 mass of window element densities
    [kg/m^3] on u, nothing on phi."""
    import scipy.sparse
    n_x, n_y = density.shape
    dx_elements = np.asarray(dx_elements, dtype=np.float64)
    dy_elements = np.asarray(dy_elements, dtype=np.float64)
    stencil = np.zeros((n_x + 1, n_y + 1, PIEZO_COMPONENTS, 3, 3, PIEZO_COMPONENTS), dtype=np.float64)
    stencil[:, :, :2, :, :, :2] = mass_stencil(density, dx_elements, dy_elements, np.float64, xp=np)
    pattern = ElasticPattern(n_x + 1, n_y + 1, components=PIEZO_COMPONENTS, xp=np)
    return pattern.csr(stencil, scipy.sparse)


def _electrode_map(electrodes, n_dofs, potential_dofs, scale, label, require_ground=True):
    """(G host CSR, held phi dofs, tied phi dofs, master dof per electrode (None for voltage ones; floating and
    resistive_load ones are tied)) over n_dofs coupled dofs, every potential dof scaled by s; ValueError for an empty
    electrode, two electrodes sharing a potential dof or (require_ground) no voltage electrode (phi undetermined)."""
    import scipy.sparse
    owner = np.full(n_dofs, -1, dtype=np.int64)
    columns = np.arange(n_dofs, dtype=np.int64)
    values = np.ones(n_dofs, dtype=np.float64)
    values[potential_dofs] = scale
    held = np.zeros(n_dofs, dtype=bool)
    tied = np.zeros(n_dofs, dtype=bool)
    masters = []
    for index, electrode in enumerate(electrodes):
        dofs = np.unique(np.asarray(electrode.dofs, dtype=np.int64))
        if dofs.size == 0:
            raise ValueError(f"{label()}: electrode '{electrode.name}' covers no node of the solve window")
        shared = dofs[owner[dofs] >= 0]
        if shared.size:
            raise ValueError(f"{label()}: electrodes '{electrodes[int(owner[shared[0]])].name}' and '{electrode.name}' "
                             f'share {shared.size} node(s) (e.g. where two adjacent faces meet): a node belongs to one '
                             'electrode only, so leave a gap or merge them into one feature')
        owner[dofs] = index
        if electrode.connection == 'voltage':
            held[dofs] = True
            masters.append(None)
        else:
            columns[dofs[1:]] = dofs[0]
            tied[dofs[1:]] = True
            masters.append(int(dofs[0]))
    if require_ground and all(master is not None for master in masters):
        raise ValueError(f'{label()}: a piezoelectric solve needs at least one electrode with connection voltage (0 V = '
                         'ground), else the potential has no reference and the system is singular')
    column_map = scipy.sparse.csr_matrix((values, (np.arange(n_dofs), columns)), shape=(n_dofs, n_dofs))
    return column_map, held, tied, tuple(masters)


class PiezoSystem:
    """Coupled host operator of one assembly over explicit displacement / potential dof indices (2D interleaved, 3D
    block), its electrode reduction G and free dofs (u not held and touching a solid element, active phi neither held nor
    tied); the static solve keeps its factor until the next assembly (runtime: the 'linear' sparse_direct runtime's
    factorize_host, SuperLU or cuDSS LDL^T, counted there; None = a private host SuperLU counted as piezo_*);
    require_ground False: no voltage electrode needed (a Bloch cell whose phase excludes a constant phi)."""

    def __init__(self, operator, electrodes, displacement_dofs, potential_dofs, held_displacement, label,
                 potential_active=None, runtime=None, previous=None, require_ground=True):
        operator = operator.tocsr()
        n_dofs = int(operator.shape[0])
        self.displacement_dofs = np.asarray(displacement_dofs, dtype=np.int64)
        self.potential_dofs = np.asarray(potential_dofs, dtype=np.int64)
        diagonal = operator.diagonal()
        stiffness_diagonal, permittivity_diagonal = diagonal[self.displacement_dofs], -diagonal[self.potential_dofs]
        active = (np.ones(self.potential_dofs.size, dtype=bool) if potential_active is None
                  else np.asarray(potential_active, dtype=bool))
        if not np.all(permittivity_diagonal[active] > 0.0):
            raise ValueError(f'{label()}: {int(np.count_nonzero(~(permittivity_diagonal[active] > 0.0)))} node(s) touch '
                             'no element with a positive permittivity (eps_dc)')
        stiffness_peak = float(stiffness_diagonal.max(initial=0.0))
        if not stiffness_peak > 0.0:
            raise ValueError(f'{label()}: no solid cell (youngs_modulus or c11 > 0) in the solve window')
        self.scale = math.sqrt(stiffness_peak / float(permittivity_diagonal[active].max()))
        self.electrodes = tuple(electrodes)
        self.column_map, held_potential, tied, self.masters = _electrode_map(self.electrodes, n_dofs,
                                                                             self.potential_dofs, self.scale, label,
                                                                             require_ground)
        # [(name, master dof, s^2 / R, R)]: the reduced master row gains j s^2 / (omega R) x_master
        self.loads = tuple((electrode.name, master, self.scale ** 2 / float(electrode.load_resistance),
                            float(electrode.load_resistance))
                           for electrode, master in zip(self.electrodes, self.masters)
                           if electrode.connection == RESISTIVE_LOAD)
        self._label = label
        self.operator = operator
        self.reduced = congruence(self.column_map, operator)   # G diagonal (no tied electrode): a scaling
        self.held_displacement = np.asarray(held_displacement, dtype=bool).reshape(-1)
        fixed = np.zeros(n_dofs, dtype=bool)
        fixed[self.displacement_dofs] = self.held_displacement | ~(stiffness_diagonal > 0.0)   # held, or not solid
        fixed[self.potential_dofs[~active]] = True
        self.free = ~(fixed | held_potential | tied)
        self._free_dofs = np.flatnonzero(self.free)
        self._runtime = runtime
        self._factor = None
        self._previous = previous   # an earlier factor the runtime may refactorise (cuDSS plan reuse)
        self._harmonic = None       # frequency_response.HarmonicSystem of (mass, loss) = _harmonic_inputs
        self._harmonic_inputs = (None, None)

    @property
    def factor(self):
        """The static factor of the free reduced operator (None before the first static solve)."""
        return self._factor

    def _fixed_and_load(self, load, held_values):
        """(fixed values, load) host float64 over every dof: held displacements [m] and voltage electrodes at their
        voltage, the load [N/m 2D, N 3D] on the u rows (inputs over displacement_dofs, flattened in their order)."""
        n_dofs = int(self.operator.shape[0])
        fixed = np.zeros(n_dofs, dtype=np.float64)
        fixed[self.displacement_dofs] = np.where(self.held_displacement,
                                                 np.asarray(held_values, dtype=np.float64).reshape(-1), 0.0)
        for electrode in self.electrodes:
            if electrode.connection == 'voltage':
                fixed[np.asarray(electrode.dofs, dtype=np.int64)] = float(electrode.voltage)
        full_load = np.zeros(n_dofs, dtype=np.float64)
        full_load[self.displacement_dofs] = np.asarray(load, dtype=np.float64).reshape(-1)
        return fixed, full_load

    def _charge_terms(self):
        """[(master dof, s Q)] of the floating electrodes: the master row sums the tied phi rows (scaled), so -s Q."""
        return [(master, self.scale * float(electrode.charge))
                for electrode, master in zip(self.electrodes, self.masters) if electrode.connection == 'floating']

    def solve_static(self, load, held_values, counters):
        """(x, A x) host float64 over every dof of the static problem: load and held displacements [m] over
        displacement_dofs, voltage electrodes at their voltage, floating ones carrying their charge."""
        if self.loads:
            raise ValueError(f"{self._label()}: electrode '{self.loads[0][0]}' has connection resistive_load, which "
                             'only the eigenfrequency and frequency_response studies solve (at DC the resistor grounds '
                             'it: use connection voltage, 0 V, for a static solve)')
        n_dofs = int(self.operator.shape[0])
        fixed, full_load = self._fixed_and_load(load, held_values)
        rhs = self.column_map.T @ (full_load - self.operator @ fixed)
        for master, value in self._charge_terms():
            rhs[master] -= value
        if self._factor is None:
            matrix = self.reduced[self._free_dofs][:, self._free_dofs]
            if self._runtime is None:
                self._factor = factorize_host(matrix, real=True)
                counters['piezo_factorizations'] += 1
                counters['piezo_factor_host_bytes'] = int(self._factor.host_bytes or 0)
            else:   # symmetric indefinite: cuDSS LDL^T on its back ends, else host SuperLU
                self._factor = self._runtime.factorize_host(matrix, real=True, symmetric=True, previous=self._previous)
            self._previous = None
        reduced_solution = np.zeros(n_dofs, dtype=np.float64)
        reduced_solution[self._free_dofs] = self._factor.solve_host(rhs[self._free_dofs])
        solution = self.column_map @ reduced_solution + fixed
        counters['piezo_solves'] += 1
        return solution, self.operator @ solution

    def solve_harmonic(self, omega, mass, loss, load, held_values, counters, fields=True):
        """(x, A x) complex host over every dof of (A + j loss - omega^2 M) x = f at omega [rad/s] (frequency_response):
        loads, held displacements, voltages and floating charges as amplitudes at phase 0; one complex factor of the
        HarmonicSystem kept while mass and loss are the same objects (runtime None: a private SuperLU counted as
        piezo_harmonic_*; else the runtime's complex LDL^T / SuperLU, counted there); resistive_load electrodes close
        through their resistor. After plan_harmonic: the reduced model (fields False: x, A x exact only at the
        electrode rows and the extra rows of plan_harmonic)."""
        harmonic = self.harmonic_system(mass, loss)
        fixed, full_load = self._fixed_and_load(load, held_values)
        before = harmonic.factorizations
        solution, product = harmonic.solve(omega, fixed, full_load, self._charge_terms(), fields=fields)
        self._count_harmonic(harmonic, before, counters)
        return solution, product

    def harmonic_system(self, mass, loss):
        """The frequency_response.HarmonicSystem of (A, loss, M), kept while mass and loss are the same objects."""
        harmonic = self._harmonic
        if harmonic is None or self._harmonic_inputs[0] is not mass or self._harmonic_inputs[1] is not loss:
            harmonic = HarmonicSystem(self.operator, loss, mass, self.free, self.column_map, self._runtime,
                                      previous=None if harmonic is None else harmonic.factor,
                                      loads=[(master, value) for _, master, value, _ in self.loads])
            self._harmonic, self._harmonic_inputs = harmonic, (mass, loss)
        return harmonic

    def _count_harmonic(self, harmonic, before, counters) -> None:
        """Private SuperLU factorisations since `before` into piezo_harmonic_* (a runtime counts its own)."""
        if self._runtime is None and harmonic.factorizations > before:
            counters['piezo_harmonic_factorizations'] += harmonic.factorizations - before
            counters['piezo_factor_host_bytes'] = max(counters['piezo_factor_host_bytes'],
                                                      int(getattr(harmonic.factor, 'host_bytes', 0) or 0))

    def clear_harmonic_plan(self) -> None:
        """Drop a plan_harmonic model (solve_harmonic factorises per frequency again)."""
        if self._harmonic is not None:
            self._harmonic.clear_reduced()

    def electrode_rows(self) -> np.ndarray:
        """Int64 dofs of every electrode (the rows electrode_values reads)."""
        rows = [np.asarray(electrode.dofs, dtype=np.int64) for electrode in self.electrodes]
        return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)

    def plan_harmonic(self, omegas, method, mass, loss, load, held_values, counters, rows=(), **options):
        """Reduced model of solve_harmonic over angular frequencies omegas [rad/s] (HarmonicSystem.plan_reduced;
        options tolerance / moments / max_points / modal_count / eigen_solver / label) with output rows = the
        electrode dofs + rows; returns its info."""
        harmonic = self.harmonic_system(mass, loss)
        fixed, full_load = self._fixed_and_load(load, held_values)
        before = harmonic.factorizations
        info = harmonic.plan_reduced(omegas, method, fixed, full_load, self._charge_terms(), **options)
        harmonic.set_output_rows(np.concatenate((self.electrode_rows(), np.asarray(rows, dtype=np.int64).ravel())))
        self._count_harmonic(harmonic, before, counters)
        return info

    def electrode_values(self, solution, product):
        """{electrode name: (V [V], Q [C/m 2D, C 3D])} of a solution x and its product A x (Q = -sum of the phi rows);
        complex for a harmonic solution."""
        scalar = complex if np.iscomplexobj(solution) else float
        values = {}
        for electrode in self.electrodes:
            dofs = np.asarray(electrode.dofs, dtype=np.int64)
            values[electrode.name] = (scalar(solution[dofs[0]]), scalar(-np.sum(product[dofs])))
        return values

    def modal_pencil(self, mass, eigen_solver, prolongation=None, bloch_text='', rigid_motions=0):
        """modal.ModalPencil of the coupled pencil (G^T A G, G^T M G) on the free dofs (short circuit on voltage
        electrodes, open on floating ones, kind 'loaded' with resistive_load electrodes), or on x_G = P x_r of a Bloch
        prolongation over the free dofs (bloch3d; bloch_text labels it); pass its solves through expanded() (x = G x_G:
        physical phi, tied floating electrodes filled; peak node |u| = 1 m read over displacement_dofs)."""
        reduced_mass = congruence(self.column_map, mass)
        components = self.displacement_dofs.size // self.potential_dofs.size   # 2D: 2, 3D: 3 per potential dof
        if self.loads and prolongation is not None:
            raise ValueError(f'{self._label()}: a resistive_load electrode is not available in a Bloch cell')
        return ModalPencil(self.reduced, reduced_mass, self.free, components, eigen_solver, prolongation=prolongation,
                           displacement_dofs=self.displacement_dofs, rigid_motions=rigid_motions,
                           loads=self.loads or None, bloch_text=bloch_text)

    def expanded(self, solve):
        """ModalSolve of a coupled-pencil solve with every resonance over every coupled dof (x = G x_G)."""
        resonances = [replace(resonance, displacement=self.column_map @ resonance.displacement)
                      for resonance in solve.resonances]
        return ModalSolve(resonances=resonances, info={**solve.info, 'piezoelectric': True})

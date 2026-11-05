"""Driven time-harmonic response of the solid (frequency_response study, piezoelectric plan stage 5, piezo_3d_plan stage
4): the reduced pencil of (K + j L - omega^2 M) x = f with one complex symmetric factor per frequency (HarmonicSystem:
private host SuperLU, or a sparse_direct runtime's SuperLU / cuDSS LDL^T with plan reuse; or a fem.reduced_harmonic
model with precomputed record rows), the per-frequency electrode record and the admittance result of an electrode (with
its BVD fit and kappa_e); host numpy (scipy inside functions)."""
# Convention exp(+j omega t) (as the EM frequency_domain study): losses enter as c -> c (1 + j eta) (mechanical loss
# factor) and eps -> eps (1 - j tan delta) (dielectric loss tangent), so L = [[eta K_uu, 0], [0, +tan delta K_phiphi]] on
# the coupled operator [[K_uu, K_uphi], [K_phiu, -K_phiphi]] (piezo.coupled_operator). Every load, prescribed
# displacement, electrode voltage and floating charge is the complex amplitude (phase 0) of the excitation. An electrode
# carries Q = -sum of its phi rows of A x (the discrete Gauss law of the complex D: displacement and loss current), its
# current is I = j omega Q and its admittance Y = I / V [S per m of depth; times the thickness setting in S].
import math
from dataclasses import dataclass

import numpy as np

from ...fem.assembly import congruence, pattern_sum
from .bvd import bvd_admittance, bvd_fit, external_coupling

ADMITTANCE_SERIES = ('abs', 're', 'im', 'phase', 'bvd_abs')
_PHASE_DEGREES = 180.0 / math.pi


class HarmonicSystem:
    """The pencil (operator, loss, mass) of one assembly reduced once to the free dofs (G^T X G with G = column_map, None
    = identity) for the frequency solves; each solve factorises the complex symmetric operator + j loss - omega^2 mass
    of the free dofs: on a cuDSS back end from the term values kept on the device (sparse_direct.DevicePencil,
    factorize_pencil: LDL^T, the previous factor's plan reused), else by the runtime's factorize_host (host SuperLU;
    counted there) or, runtime None, a private host SuperLU. The summed pattern never depends on the
    values (fem.assembly.pattern_sum), so the cuDSS plan holds across frequencies. loads = [(reduced free dof, b)]: + j b
    / omega on that diagonal (a resistive electrode load, piezo.PiezoSystem). Direct solves with fields False after
    set_output_rows (no column map): x and A x only at those rows, the free solution kept (on the device after a cuDSS
    solve) for last_solution (R6)."""

    def __init__(self, operator, loss, mass, free, column_map=None, runtime=None, previous=None, loads=()):
        import scipy.sparse
        self.operator, self.loss, self.mass = operator.tocsr(), None if loss is None else loss.tocsr(), mass.tocsr()
        self.column_map = column_map
        self.free_dofs = np.flatnonzero(np.asarray(free, dtype=bool))
        self._reduced = [self._reduced_free(matrix) for matrix in (self.operator, self.loss, self.mass)]
        self._load = None
        if loads:
            position = np.full(int(self.operator.shape[0]), -1, dtype=np.int64)
            position[self.free_dofs] = np.arange(self.free_dofs.size)
            rows = position[np.array([dof for dof, _ in loads], dtype=np.int64)]
            if np.any(rows < 0):
                raise ValueError('a resistively loaded electrode has no free potential (every node held or inactive)')
            n_free = int(self.free_dofs.size)
            self._load = scipy.sparse.csr_matrix((np.array([value for _, value in loads], dtype=np.float64),
                                                  (rows, rows)), shape=(n_free, n_free))
        # the summed pattern once (pattern_sum's canonical CSR order depends on the patterns only): each term's values
        # aligned on it, so a frequency only combines value arrays (no COO round trip per solve)
        stiffness, loss, mass = self._reduced
        terms = {name: term for name, term in (('stiffness', stiffness), ('loss', loss), ('mass', mass),
                                               ('load', self._load)) if term is not None}
        zeros = {name: term * 0.0 for name, term in terms.items()}
        self._pattern = pattern_sum(list(zeros.values()))
        aligned = {name: pattern_sum([term if other == name else zeros[other] for other in terms]).data
                   for name, term in terms.items()}
        self._fixed_values = aligned['stiffness'].astype(np.complex128)
        if 'loss' in aligned:
            self._fixed_values += 1j * aligned['loss']
        self._mass_values = aligned['mass']
        self._load_values = aligned.get('load')
        # |diag K|, |diag M| of the free dofs: the row scale of every cuDSS factorisation (computed once)
        self._diagonals = (np.abs(stiffness.diagonal()), np.abs(mass.diagonal()))
        self.runtime = runtime
        self._device = None      # sparse_direct.DevicePencil of the terms (cuDSS back end; made on the first factorize)
        self.factor = previous   # the last factor (a CudssFactorization is refactorised in place)
        self.factorizations = 0  # factorisations made (direct solves, reduced-model expansions and verification)
        self._planned = None     # (ReducedHarmonicModel, fixed, load, extra_rhs) of plan_reduced, or None
        self._rows = None        # fem.reduced_outputs.ReducedRows of set_output_rows, or None
        self._direct_rows = None   # (rows, columns, operator / mass / loss row blocks on columns) of direct solves
        self._kept = None        # (free solution, fixed) of the last direct rows-only solve (last_solution)
        self._fixed_terms = None   # (fixed, (K fixed, M fixed, L fixed) or None when fixed is 0) of _fixed_product
        self._free_position = np.full(self._n_reduced(), -1, dtype=np.int64)   # dof -> free index, -1 held
        self._free_position[self.free_dofs] = np.arange(self.free_dofs.size)

    def _reduced_free(self, matrix):
        if matrix is None:
            return None
        reduced = matrix if self.column_map is None else congruence(self.column_map, matrix)
        return reduced.tocsr()[self.free_dofs][:, self.free_dofs].tocsr()

    def _product(self, omega, vector):
        """(operator + j loss - omega^2 mass) vector over every dof."""
        product = _real_times(self.operator, vector) - (omega * omega) * _real_times(self.mass, vector)
        return product if self.loss is None else product + 1j * _real_times(self.loss, vector)

    def _fixed_product(self, omega, fixed):
        """A(omega) fixed from K / M / L times fixed, kept while the held values stay equal (0.0 when they are all 0)."""
        if self._fixed_terms is None or not np.array_equal(self._fixed_terms[0], fixed):
            terms = None if not fixed.any() else tuple(None if matrix is None else _real_times(matrix, fixed)
                                                       for matrix in (self.operator, self.mass, self.loss))
            self._fixed_terms = (fixed.copy(), terms)
        terms = self._fixed_terms[1]
        if terms is None:
            return 0.0
        operator_fixed, mass_fixed, loss_fixed = terms
        product = operator_fixed - (omega * omega) * mass_fixed
        return product if loss_fixed is None else product + 1j * loss_fixed

    def _rows_product(self, omega, values):
        """A x over every dof, exact at the direct output rows and 0 elsewhere, from x at the row blocks' columns."""
        rows, _, (operator_rows, mass_rows, loss_rows) = self._direct_rows
        product = operator_rows @ values - (omega * omega) * (mass_rows @ values)
        if loss_rows is not None:
            product = product + 1j * (loss_rows @ values)
        applied = np.zeros(self._n_reduced(), dtype=np.complex128)
        applied[rows] = product
        return applied

    def _factorize(self, omega):
        """Factor of the free-dof operator + j loss - omega^2 mass (+ j b / omega) at omega [rad/s] (counted)."""
        import scipy.sparse
        from ...solvers.sparse_direct import SparseDirectSolver, factorize_host
        coefficients = [1.0, -omega * omega] + ([] if self._load_values is None else [1j / omega])
        if self._device is None:
            terms = [self._fixed_values, self._mass_values] + ([] if self._load_values is None else [self._load_values])
            self._device = isinstance(self.runtime, SparseDirectSolver) and self.runtime.device_pencil(
                self._pattern.indptr, self._pattern.indices, terms, symmetric=True) or False
        # the scale of diag(K) + omega^2 diag(M) never cancels near a resonance (cuDSS equilibration)
        stiffness_diagonal, mass_diagonal = self._diagonals
        scale = None if self.runtime is None else 1.0 / np.sqrt(stiffness_diagonal + (omega * omega) * mass_diagonal)
        if self._device:   # cuDSS: the pencil values stay on the device (no host CSR per frequency)
            self.factor = self.runtime.factorize_pencil(self._device, coefficients, row_scale=scale,
                                                        previous=self.factor)
        else:
            values = self._fixed_values - (omega * omega) * self._mass_values
            if self._load_values is not None:
                values += (1j / omega) * self._load_values
            matrix = scipy.sparse.csr_matrix((values, self._pattern.indices, self._pattern.indptr),
                                             shape=self._pattern.shape)
            if self.runtime is None:
                self.factor = factorize_host(matrix)
            else:
                self.factor = self.runtime.factorize_host(matrix, symmetric=True, row_scale=scale,
                                                          previous=self.factor)
        self.factorizations += 1
        return self.factor

    def _n_reduced(self) -> int:
        """Unknowns of x_r (every dof without a column map, else its columns)."""
        return int(self.operator.shape[0] if self.column_map is None else self.column_map.shape[1])

    def _full(self, reduced_free_values, fixed) -> np.ndarray:
        """x = G x_r + fixed over every dof of free-dof values x_r."""
        reduced_solution = np.zeros(self._n_reduced(), dtype=np.complex128)
        reduced_solution[self.free_dofs] = reduced_free_values
        return (reduced_solution if self.column_map is None else self.column_map @ reduced_solution) + fixed

    def solve(self, omega, fixed, load, extra_rhs=(), fields=True, product_at_rows=False):
        """(x, A x): complex host vectors over every dof of (operator + j loss - omega^2 mass) x = load with x = G x_r +
        fixed, x_r solved on the free dofs; extra_rhs = [(reduced dof, value)] subtracted from the reduced rhs (a floating
        electrode's charge). After plan_reduced: x_r = V y of the model (fields False: x, A x exact only at the
        set_output_rows rows, 0 elsewhere: no V y). product_at_rows (after set_output_rows): A x exact only at those
        rows, 0 elsewhere, also with fields (no full products per frequency; without row blocks the full A x)."""
        fixed = np.asarray(fixed, dtype=np.complex128)
        if self._planned is not None:
            return self._reduced_solve(omega, fixed, load, extra_rhs, fields, product_at_rows)
        rhs = np.asarray(load, dtype=np.complex128) - self._fixed_product(omega, fixed)
        if self.column_map is not None:
            rhs = self.column_map.T @ rhs
        for dof, value in extra_rhs:
            rhs[dof] -= value
        self._factorize(omega)
        if not fields and self._direct_rows is not None:
            return self._rows_solve(omega, fixed, rhs[self.free_dofs])
        self._kept = None
        solution = self._full(self.factor.solve_host(rhs[self.free_dofs]), fixed)
        if product_at_rows and self._direct_rows is not None:
            return solution, self._rows_product(omega, solution[self._direct_rows[1]])
        return solution, self._product(omega, solution)

    def _rows_solve(self, omega, fixed, free_rhs):
        """(x, A x) exact at the direct output rows, 0 elsewhere: the free solution stays where the factor solved it (a
        cuDSS factor: the device) and is kept; only its values at the row blocks' columns reach the host."""
        rows, columns, _ = self._direct_rows
        on_device = getattr(self.factor, 'back_end', None) == 'cudss'   # a CudssFactorization (both precisions)
        free_solution = self.factor.solve(free_rhs) if on_device else self.factor.solve_host(free_rhs)
        self._kept = (free_solution, fixed)
        positions = self._free_position[columns]
        free = positions >= 0
        values = fixed[columns].copy()   # x = fixed + the free solution scattered (as _full)
        if on_device:
            import cupy as cp
            values[free] += cp.asnumpy(free_solution[cp.asarray(positions[free])])
        else:
            values[free] += free_solution[positions[free]]
        solution = np.zeros(self._n_reduced(), dtype=np.complex128)
        solution[rows] = values[np.searchsorted(columns, rows)]
        return solution, self._rows_product(omega, values)

    @property
    def has_output_rows(self) -> bool:
        """set_output_rows was called (for the reduced model or the direct solves)."""
        return self._rows is not None or self._direct_rows is not None

    def last_solution(self):
        """Host x over every dof of the last direct rows-only solve (its kept free solution: one device transfer after a
        cuDSS solve, R6), None when the last solve was another kind."""
        if self._kept is None:
            return None
        from ...core.backend import to_host
        free_solution, fixed = self._kept
        return self._full(np.asarray(to_host(free_solution)), fixed)

    # ---- reduced sweeps (fem.reduced_harmonic, experiment plan A3 / A4) ----
    @property
    def reduced_model(self):
        """The ReducedHarmonicModel solve() evaluates (plan_reduced), or None (one factorisation per solve)."""
        return None if self._planned is None else self._planned[0]

    def clear_reduced(self) -> None:
        """Drop the reduced model and its output rows: solve() factorises per frequency again (direct output rows
        stay)."""
        self._planned = None
        self._rows = None
        self._kept = None

    def reduced_problem(self, fixed, load, extra_rhs=()) -> tuple:
        """(HarmonicTerm list, HarmonicLoad list) of the free-dof problem of solve (fem.reduced_harmonic): K, j L, -omega^2
        M (+ j b / omega) and the rhs G^T (load - A(omega) fixed) - extra_rhs split by its powers of omega."""
        from ...fem.reduced_harmonic import HarmonicLoad, HarmonicTerm, PowerCoefficient
        stiffness, loss, mass = self._reduced
        terms = [HarmonicTerm(stiffness, PowerCoefficient(1.0, 0))]
        if loss is not None:
            terms.append(HarmonicTerm(loss, PowerCoefficient(1j, 0)))
        terms.append(HarmonicTerm(mass, PowerCoefficient(-1.0, 2)))
        if self._load is not None:
            terms.append(HarmonicTerm(self._load, PowerCoefficient(1j, -1)))
        fixed = np.asarray(fixed, dtype=np.complex128)

        def reduced(vector):
            return np.array(vector if self.column_map is None else self.column_map.T @ vector, dtype=np.complex128)
        constant = reduced(np.asarray(load, dtype=np.complex128) - self.operator @ fixed)
        for dof, value in extra_rhs:
            constant[dof] -= value
        loads = [HarmonicLoad(constant[self.free_dofs], PowerCoefficient(1.0, 0))]
        if np.any(fixed != 0.0):
            loads.append(HarmonicLoad(reduced(self.mass @ fixed)[self.free_dofs], PowerCoefficient(1.0, 2)))
            if self.loss is not None:
                loads.append(HarmonicLoad(-reduced(self.loss @ fixed)[self.free_dofs], PowerCoefficient(1j, 0)))
        return terms, loads

    def plan_reduced(self, omegas, method, fixed, load, extra_rhs=(), *, tolerance=1e-6, moments=4, max_points=40,
                     modal_count=20, eigen_solver=None, verify=True, label='solid') -> dict:
        """Reduced model of solve's free-dof response over angular frequencies omegas [rad/s] for these fixed / load /
        extra_rhs: 'adaptive' (greedy moment matching) or 'modal' (modal_count modes of (K, M) nearest the band centre +
        the static correction; not with electrodes); returns the model's info."""
        from ...fem.harmonic_resonances import solve_harmonic_resonances
        from ...fem.reduced_harmonic import SWEEP_METHODS, ReducedHarmonicModel
        if method not in SWEEP_METHODS:
            raise ValueError(f"reduced sweep method must be one of {', '.join(SWEEP_METHODS)}, got {method!r}")
        self.clear_reduced()
        omegas = np.atleast_1d(np.asarray(omegas, dtype=np.float64)).ravel()
        if not (omegas.size and np.all(np.isfinite(omegas)) and np.all(omegas > 0.0)):
            raise ValueError(f'{label}: a reduced sweep needs finite angular frequencies > 0 rad/s')
        if not self.free_dofs.size:
            raise ValueError(f'{label}: no free unknowns to reduce')
        fixed = np.array(fixed, dtype=np.complex128)
        load = np.array(load, dtype=np.complex128)
        terms, loads = self.reduced_problem(fixed, load, extra_rhs)
        model = ReducedHarmonicModel(terms, loads, self._factorize, label=label)
        if method == 'adaptive':
            info = model.train(omegas, tolerance=tolerance, moments=moments, max_points=max_points, verify=verify)
        else:
            if self.column_map is not None or self._load is not None:
                raise ValueError(f"{label}: sweep_method 'modal' needs a purely elastic pencil (piezoelectric electrodes "
                                 "couple a massless potential): use sweep_method 'adaptive'")
            stiffness, _, mass = self._reduced
            centre = 0.5 * (omegas.min() + omegas.max()) / (2.0 * math.pi)
            resonance_solve = solve_harmonic_resonances(stiffness, mass, centre, modal_count, eigen_solver=eigen_solver,
                                                        label='modal basis')
            if not resonance_solve.resonances:
                raise ValueError(f'{label}: the modal sweep found no mode near {centre:.6g} Hz')
            vectors = np.column_stack([np.asarray(resonance.field, dtype=np.complex128)
                                       for resonance in resonance_solve.resonances])
            info = model.fit_modal(vectors, omegas, tolerance=tolerance, verify=verify)
            info.update(modes=len(resonance_solve.resonances),
                        mode_frequencies=[resonance.frequency for resonance in resonance_solve.resonances],
                        eigen_seconds=float(resonance_solve.info['seconds']))
            info['warnings'] = list(info['warnings']) + list(resonance_solve.info['warnings'])
        self._planned = (model, fixed, load, [(int(dof), value) for dof, value in extra_rhs])
        return info

    def set_output_rows(self, rows) -> None:
        """Rows (over every dof) where solve(fields=False) must give x and A x exactly (records: probes, supports,
        electrodes): after plan_reduced their L V blocks (fem.reduced_outputs.ReducedRows), else for the direct solves
        the operator / mass / loss rows on the columns they touch (no column map; with one, direct solves stay full)."""
        import scipy.sparse
        from ...fem.reduced_outputs import ReducedRows
        if self._planned is None:
            if self.column_map is None:
                rows = np.unique(np.asarray(rows, dtype=np.int64))
                blocks = [None if matrix is None else matrix[rows] for matrix in (self.operator, self.mass, self.loss)]
                columns = np.unique(np.concatenate([rows] + [block.indices for block in blocks if block is not None]))
                self._direct_rows = (rows, columns, tuple(None if block is None else block[:, columns].tocsr()
                                                          for block in blocks))
            return
        model, fixed, _, _ = self._planned
        n_reduced = self._n_reduced()
        selection = scipy.sparse.csr_matrix((np.ones(self.free_dofs.size), (self.free_dofs, np.arange(
            self.free_dofs.size))), shape=(n_reduced, self.free_dofs.size))
        prolongation = selection if self.column_map is None else (self.column_map @ selection).tocsr()
        terms = [(self.operator, lambda omega: 1.0), (self.mass, lambda omega: -omega * omega)]
        if self.loss is not None:
            terms.append((self.loss, lambda omega: 1j))
        self._rows = ReducedRows(rows, prolongation, model.basis, fixed, terms)

    def _reduced_solve(self, omega, fixed, load, extra_rhs, fields, product_at_rows=False):
        model, planned_fixed, planned_load, planned_extra = self._planned
        if not (np.array_equal(planned_fixed, fixed) and np.array_equal(planned_load, np.asarray(load))
                and [(int(dof), value) for dof, value in extra_rhs] == planned_extra):
            raise RuntimeError('the held values, loads or charges changed after plan_reduced: plan again or call '
                               'clear_reduced()')
        coordinates = model.reduced_coordinates([omega])[0]
        if not fields:
            if self._rows is None:
                raise RuntimeError('solve(fields=False) needs set_output_rows after plan_reduced')
            return self._rows.evaluate(omega, coordinates)
        solution = self._full(model.basis @ coordinates, fixed)
        if product_at_rows and self._rows is not None:
            return solution, self._rows.evaluate(omega, coordinates)[1]
        return solution, self._product(omega, solution)


def _real_times(matrix, vector):
    """matrix @ vector; a real sparse matrix times a complex vector as two real products (scipy's sparsetools would
    upcast the matrix values to a complex copy on every call)."""
    if np.iscomplexobj(matrix.data) or not np.iscomplexobj(vector):
        return matrix @ vector
    return matrix @ np.ascontiguousarray(vector.real) + 1j * (matrix @ np.ascontiguousarray(vector.imag))


def solve_harmonic(operator, loss, mass, omega, free, fixed, load, column_map=None, extra_rhs=()):
    """(x, A x, factor host bytes) of one frequency by a private HarmonicSystem (one complex host SuperLU; arguments as
    HarmonicSystem / its solve)."""
    harmonic = HarmonicSystem(operator, loss, mass, free, column_map)
    solution, product = harmonic.solve(omega, fixed, load, extra_rhs)
    return solution, product, int(harmonic.factor.host_bytes or 0)


def reaction_table(record, name) -> dict:
    """'table' of a support's complex reaction per solved frequency from its core.harmonic_probe.HarmonicProbeRecord of
    force components (ascending f): columns frequency, <F>_re, <F>_im per component; extras frequency [Hz], components,
    support. ValueError before the first frequency."""
    frequencies, values = record.recorded()
    columns, units = ['frequency'], ['Hz']
    for component in record.components:
        columns += [f'{component}_re', f'{component}_im']
        units += [record.unit, record.unit]
    rows = [[float(frequency)] + [part for component in record.components
                                  for part in (float(values[component][index].real),
                                               float(values[component][index].imag))]
            for index, frequency in enumerate(frequencies)]
    return {'kind': 'table', 'columns': columns, 'units': units, 'rows': rows, 'frequency': frequencies.tolist(),
            'components': list(record.components), 'support': name}


@dataclass(frozen=True, eq=False)
class FrequencyPoint:
    """One solved frequency [Hz] with {electrode name: (V, Q)} complex amplitudes [V, C/m] (empty without piezo)."""
    frequency: float
    electrodes: dict


@dataclass(frozen=True, eq=False)
class AdmittanceMonitor:
    """Runtime object of an admittance feature: its electrode name ('' = the only driven voltage electrode), the BVD
    fit band [Hz] (None = the whole sweep) and the load impedance [Ohm] of kappa_e."""
    name: str
    electrode: str
    fit_min: float | None
    fit_max: float | None
    load_impedance: float = 50.0


def driven_electrode(monitor, electrodes, label):
    """Name of the monitored electrode: monitor.electrode, or the only voltage electrode with a non-zero voltage;
    ValueError otherwise."""
    voltage = [electrode for electrode in electrodes if electrode.connection == 'voltage']
    if monitor.electrode:
        match = [electrode for electrode in voltage if electrode.name == monitor.electrode]
        if not match:
            raise ValueError(f"{label()}: admittance '{monitor.name}' names electrode '{monitor.electrode}', which is "
                             f"no enabled voltage electrode; voltage electrodes: "
                             f"{', '.join(electrode.name for electrode in voltage) or 'none'}")
        return monitor.electrode
    driven = [electrode.name for electrode in voltage if electrode.voltage != 0.0]
    if len(driven) != 1:
        raise ValueError(f"{label()}: admittance '{monitor.name}' needs its electrode setting when {len(driven)} "
                         f"voltage electrodes have a non-zero voltage ({', '.join(driven) or 'none'})")
    return driven[0]


def admittance_result(monitor, electrode, points, thickness=None):
    """'xy' result of Y(f) = j omega Q / V of `electrode` over the solved FrequencyPoints (ascending f), times the
    thickness [m] (None: per m of depth; a 3D model passes 1.0, its charges are in C): series abs / re / im [S or S/m],
    phase [deg], bvd_abs (the fitted model, NaN without a fit); extras the BVD fit (bvd.FIT_KEYS, fit_warning when it
    failed), kappa_e / omega_C0_Z at monitor.load_impedance (NaN per m of depth: kappa_e_warning), load_impedance,
    bvd_units, electrode."""
    depth = 1.0 if thickness is None else float(thickness)
    points = sorted(points, key=lambda point: point.frequency)
    if not points:
        raise ValueError(f"admittance '{monitor.name}': no frequency was solved (a frequency_response study fills it)")
    frequencies = np.array([point.frequency for point in points], dtype=np.float64)
    admittance = np.empty(frequencies.size, dtype=np.complex128)
    for index, point in enumerate(points):
        voltage, charge = point.electrodes[electrode]
        if voltage == 0:
            raise ValueError(f"admittance '{monitor.name}': electrode '{electrode}' has voltage 0, so Y = I / V is "
                             'undefined: give the driven electrode a non-zero voltage')
        admittance[index] = 2j * math.pi * point.frequency * charge / voltage * depth
    fit = bvd_fit(frequencies, admittance, monitor.fit_min, monitor.fit_max)
    model = (np.abs(bvd_admittance(frequencies, fit)) if math.isfinite(fit['C0'])
             else np.full(frequencies.size, math.nan))
    per_depth = '' if thickness is not None else '/m'
    inverse_depth = '' if thickness is not None else ' m'
    unit = f'S{per_depth}'
    series = {'abs': np.abs(admittance), 're': admittance.real, 'im': admittance.imag,
              'phase': np.angle(admittance) * _PHASE_DEGREES, 'bvd_abs': model}
    units = {'abs': unit, 're': unit, 'im': unit, 'phase': 'deg', 'bvd_abs': unit}
    bvd_units = {'f_s': 'Hz', 'f_p': 'Hz', 'f_peak': 'Hz', 'f_dip': 'Hz', 'k_eff2': '1', 'Q_s': '1',
                 'tan_delta_0': '1', 'C0': f'F{per_depth}', 'C1': f'F{per_depth}', 'L1': f'H{inverse_depth}',
                 'R1': f'Ohm{inverse_depth}', 'fit_error': '1', 'fit_points': '1', 'kappa_e': 'rad/s',
                 'omega_C0_Z': '1', 'load_impedance': 'Ohm'}
    if thickness is None:   # L1 / C0 per m of depth: a load in Ohm has no meaning there
        coupling = {'kappa_e': math.nan, 'omega_C0_Z': math.nan,
                    'kappa_e_warning': 'kappa_e needs the admittance of a finite device: set the thickness setting '
                                       '(2D) so C0 and L1 are in F and H'}
    else:
        coupling = external_coupling(fit, monitor.load_impedance)
    return {'kind': 'xy', 'x': frequencies, 'x_label': 'Frequency', 'x_unit': 'Hz', 'y_label': 'Admittance',
            'series': series, 'series_units': units, 'electrode': electrode, 'bvd_units': bvd_units, **fit,
            **coupling, 'load_impedance': float(monitor.load_impedance)}

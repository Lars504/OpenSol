"""Physics-neutral squeeze-film gas damping (new_modules_plan S6, Basiri E4): SqueezeFilm solves the linearised
compressible Reynolds equation by fem.harmonic.ScalarHarmonicSystem on a planar TriangleMesh (a solid's footprint or a
membrane's own mesh, optionally on an element subset) with vents p = 0 and the Veijola slip viscosity, giving per mode
c_psi, the gas stiffness, gamma_gas = c_psi / m_eff and Q_gas; closed forms (rigid vented annulus, squeeze number)."""
# Equation (exp(+j omega t); gap H, gap-opening displacement w of the surface, gauge pressure p):
#   div(H^3 / (12 mu_eff) grad p) - j omega (H / p_a) p = j omega w
# solved as -div(a grad p) + j omega c p = f with a = H^3 / (12 mu_eff), c = H / p_a (c = 0: ambient_pressure None, the
# incompressible film) per unit gap-opening velocity amplitude (j omega q = 1 m/s, so f = -w): pressure in Pa per m/s.
# The gas pushes the surface along +w, so the modal force per unit velocity is F = int p conj(w) dA [N s/m] = -(c_psi +
# k_gas / (j omega)): c_psi = -Re F [kg/s], k_gas = omega Im F [N/m] (> 0: the trapped-gas spring). c and k are quadratic
# in w (the sign of the opening drops out). With w the mode shape in the normalisation of m_eff: gamma_gas = c_psi /
# m_eff [1/s] (energy decay rate = FWHM in rad/s, Basiri section 3), Q_gas = omega m_eff / c_psi.
# Element subset (active, default all): the film lives on those elements only; vents default to the boundary edges of
# the subset (active_boundary_edges), other boundary edges are sealed (no flux); a closed incompressible film is
# singular (ValueError). w given as coefficients of the film's own full Lagrange numbering (the membrane mode on its own
# mesh, a surface_mesh trace of the same order) goes through the film mass matrix (exact on isoparametric elements);
# anything else (a number, other coefficients, a callable) through fem.surface_mesh.lagrange_field_function (affine).
# An incompressible film is omega-independent: one factor serves every mode. Slip: Veijola et al. 1995, mu_eff = mu / (1
# + 9.638 Kn^1.159), Kn = mean free path at the ambient pressure / H (valid 0 <= Kn <= 880, ~5 %). Planar integrals.
import math
from dataclasses import dataclass

import numpy as np

from .mesh import one_sided

SLIP_MODELS = ('none', 'veijola')
FILM_TABLE_COLUMNS = ('resonance_number', 'frequency', 'effective_mass', 'gas_damping', 'gas_stiffness', 'gamma_gas',
                      'q_gas')
FILM_TABLE_UNITS = ('1', 'Hz', 'kg', 'kg/s', 'N/m', '1/s', '1')
GAS_TABLE_COLUMNS = ('gas_damping', 'gas_stiffness', 'gamma_gas', 'q_gas')   # appended to a resonance table
GAS_TABLE_UNITS = ('kg/s', 'N/m', '1/s', '1')
VEIJOLA_COEFFICIENT = 9.638   # Veijola 1995 slip fit mu_eff = mu / (1 + 9.638 Kn^1.159)
VEIJOLA_EXPONENT = 1.159
VEIJOLA_KNUDSEN_LIMIT = 880.0   # the fit's range 0 <= Kn <= 880


def _plain(values):
    """Float of a 0-d result, else the float64 array."""
    values = np.asarray(values, dtype=np.float64)
    return float(values) if values.ndim == 0 else values


def knudsen_number(mean_free_path, gap):
    """Kn = mean free path [m] / gap [m] (numbers or arrays)."""
    return _plain(np.asarray(mean_free_path, dtype=np.float64) / np.asarray(gap, dtype=np.float64))


def effective_viscosity(viscosity, gap, mean_free_path=0.0, slip='veijola'):
    """Slip-corrected viscosity mu_eff [Pa s] of a gap [m]: 'veijola' mu / (1 + 9.638 Kn^1.159), 'none' mu."""
    if slip not in SLIP_MODELS:
        raise ValueError(f"slip must be one of {', '.join(SLIP_MODELS)}, got {slip!r}")
    gap = np.asarray(gap, dtype=np.float64)
    viscosity = np.broadcast_to(np.asarray(viscosity, dtype=np.float64), gap.shape)
    if slip == 'none':
        return _plain(viscosity)
    knudsen = np.asarray(knudsen_number(mean_free_path, gap))
    return _plain(viscosity / (1.0 + VEIJOLA_COEFFICIENT * knudsen ** VEIJOLA_EXPONENT))


def squeeze_number(viscosity, omega, length, ambient_pressure, gap) -> float:
    """sigma = 12 mu omega L^2 / (p_a H^2) (mu Pa s, omega rad/s, L / H m, p_a Pa): compressibility matters near 1."""
    return 12.0 * float(viscosity) * float(omega) * float(length) ** 2 / (float(ambient_pressure) * float(gap) ** 2)


def rigid_annulus_damping(viscosity, gap, outer_radius, inner_radius=0.0) -> float:
    """Incompressible c_sq [kg/s] of a rigid annular piston vented at both rims: 3 pi mu G / (2 H^3), G = (a^4 - b^4) -
    (a^2 - b^2)^2 / ln(a / b) (inner radius 0: the disc vented at its rim, G = a^4)."""
    outer, inner = float(outer_radius), float(inner_radius)
    if not 0.0 <= inner < outer:
        raise ValueError(f'rigid_annulus_damping: need 0 <= inner < outer radius, got {inner!r}, {outer!r} m')
    moment = outer ** 4 if inner == 0.0 else ((outer ** 4 - inner ** 4)
                                              - (outer ** 2 - inner ** 2) ** 2 / math.log(outer / inner))
    return 3.0 * math.pi * float(viscosity) * moment / (2.0 * float(gap) ** 3)


def active_boundary_edges(mesh, active=None) -> np.ndarray:
    """Int64 mesh edges with exactly one element of the bool element subset `active` beside them (None: the outer
    boundary of the mesh)."""
    if active is None:
        return np.flatnonzero(mesh.edge_elements[:, 1] < 0).astype(np.int64)
    return np.flatnonzero(one_sided(mesh.edge_elements, active)).astype(np.int64)


@dataclass(frozen=True, eq=False)
class SqueezeFilmResponse:
    """Film response per unit gap-opening velocity amplitude (module header): pressure coefficients, modal force, c, k."""
    omega: float            # rad/s
    pressure: np.ndarray    # complex128 full-mesh Lagrange coefficients [Pa per m/s] (0 off the active elements)
    force: complex          # F = int p conj(w) dA [N per m/s]
    damping: float          # c_psi = -Re F [kg/s]
    stiffness: float        # k_gas = omega Im F [N/m]

    def decay_rate(self, effective_mass) -> float:
        """gamma_gas = c_psi / m_eff [1/s] (m_eff in kg, the shape's normalisation)."""
        return self.damping / float(effective_mass)

    def quality_factor(self, effective_mass) -> float:
        """Q_gas = omega m_eff / c_psi (inf without damping)."""
        return math.inf if self.damping == 0.0 else self.omega * float(effective_mass) / self.damping


class SqueezeFilm:
    """Squeeze-film Reynolds solve on a planar TriangleMesh (module header): gap [m] number or per element, viscosity
    [Pa s], ambient_pressure [Pa] (None = incompressible), mean_free_path [m] at p_a with slip 'veijola' / 'none',
    active = bool per element (None: all), vent_edges = boundary edges of the active set held at p = 0 (None: all),
    Lagrange order 1 / 2, linear_solver as ScalarHarmonicSystem."""

    def __init__(self, mesh, gap, viscosity, ambient_pressure=None, *, mean_free_path=0.0, slip='veijola',
                 vent_edges=None, active=None, order=2, linear_solver=None, label=lambda: 'squeeze film'):
        from .harmonic import HarmonicCondition, ScalarHarmonicSystem
        self.mesh = mesh
        self.label = label
        self.active = self._active_mask(active)
        self.gap = self._element_values(gap, 'gap')
        viscosity = float(viscosity)
        if not (math.isfinite(viscosity) and viscosity > 0.0):
            raise ValueError(f'{label()}: viscosity must be > 0 Pa s, got {viscosity!r}')
        if ambient_pressure is not None:
            ambient_pressure = float(ambient_pressure)
            if not (math.isfinite(ambient_pressure) and ambient_pressure > 0.0):
                raise ValueError(f'{label()}: ambient pressure must be > 0 Pa (None = incompressible), got '
                                 f'{ambient_pressure!r}')
        mean_free_path = float(mean_free_path)
        if not (math.isfinite(mean_free_path) and mean_free_path >= 0.0):
            raise ValueError(f'{label()}: mean free path must be >= 0 m, got {mean_free_path!r}')
        self.viscosity = viscosity
        self.ambient_pressure = ambient_pressure
        self.mean_free_path = mean_free_path
        self.effective_viscosity = effective_viscosity(viscosity, self.gap, mean_free_path, slip)
        conductance = _plain(np.asarray(self.gap) ** 3 / (12.0 * np.asarray(self.effective_viscosity)))
        compressibility = 0.0 if ambient_pressure is None else _plain(np.asarray(self.gap) / ambient_pressure)
        system = ScalarHarmonicSystem(mesh, order, linear_solver, label=label)
        system.set_active(self.active)
        system.set_coefficients(a=conductance, damping=compressibility)
        boundary = active_boundary_edges(mesh, self.active)
        vents = boundary if vent_edges is None else np.unique(np.asarray(vent_edges, dtype=np.int64).ravel())
        if not np.isin(vents, boundary).all():
            raise ValueError(f'{label()}: {int((~np.isin(vents, boundary)).sum())} vent edges are not boundary edges '
                             'of the film')
        if ambient_pressure is None and not len(vents):
            raise ValueError(f'{label()}: a closed incompressible film (no vent, no ambient pressure) cannot move gas: '
                             'its pressure solve is singular; vent an edge or set the ambient pressure')
        if len(vents):
            system.add_boundary_condition(vents, HarmonicCondition('dirichlet', 0.0))
        self.vent_edges = vents
        self.system = system
        self._mass = None             # (n_assemblies, film mass CSR on the active dofs)
        self._fixed_factor = None     # (n_assemblies, factor) of the incompressible film

    def _active_mask(self, active) -> np.ndarray:
        """Bool (n_elements,) of the film elements (None: every element)."""
        n_elements = self.mesh.n_elements
        if active is None:
            return np.ones(n_elements, dtype=bool)
        mask = np.asarray(active, dtype=bool).reshape(-1)
        if mask.shape != (n_elements,):
            raise ValueError(f'{self.label()}: active must hold one bool per mesh element ({n_elements},), got shape '
                             f'{mask.shape}')
        if not mask.any():
            raise ValueError(f'{self.label()}: no film element (the active set is empty)')
        return mask.copy()

    def _element_values(self, values, name):
        """Positive float or float64 (n_elements,) array of a per-element input (checked on the active elements;
        inactive ones set to 1)."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim and array.shape != (self.mesh.n_elements,):
            raise ValueError(f'{self.label()}: {name} must be a number or one value per mesh element '
                             f'({self.mesh.n_elements},), got shape {array.shape}')
        used = array[self.active] if array.ndim else array
        if not (np.isfinite(used).all() and (used > 0.0).all()):
            raise ValueError(f'{self.label()}: {name} must be finite and > 0')
        return _plain(np.where(self.active, array, 1.0) if array.ndim else array)

    def shape_function(self, normal_displacement):
        """Callable (points (m, q, 2), mesh elements (m,)) -> (m, q) of w [m]: a number, full-mesh P1 / P2 coefficients
        (affine mesh) or such a callable."""
        if callable(normal_displacement):
            return normal_displacement
        values = np.asarray(normal_displacement)
        if values.ndim == 0:
            constant = values.item()
            return lambda points, elements: np.full(np.shape(points)[:2], constant)
        from .surface_mesh import lagrange_field_function
        return lagrange_field_function(self.mesh, values)

    def _film_mass(self):
        """CSR mass int phi_i phi_j dA on the active film dofs (system numbering) of the current assembly."""
        system = self.system
        system.ensure_assembled()
        if self._mass is None or self._mass[0] != system.n_assemblies:
            from .assembly import assemble_mass
            self._mass = (system.n_assemblies, assemble_mass(system.dofs, 1.0).tocsr())
        return self._mass[1]

    def _solve_pressure(self, omega) -> np.ndarray:
        """System-numbering pressure of the current load at omega; an incompressible film keeps one factor."""
        system = self.system
        if self.ambient_pressure is not None:
            system.solve(omega)
            return system.solution
        system.ensure_assembled()
        pencil = system.pencil
        if self._fixed_factor is None or self._fixed_factor[0] != system.n_assemblies:
            if self._fixed_factor is not None:   # a cuDSS factor lives outside the cupy pool: free it before replacing
                getattr(self._fixed_factor[1], 'release', lambda: None)()
                self._fixed_factor = None
            try:
                factor = system.linear_solver.factorize_host(system.operator(0.0), symmetric=system.symmetric)
            except RuntimeError as exc:
                raise ValueError(f'{self.label()}: the film pressure operator is singular ({exc}): a closed '
                                 'incompressible film or a film region without a vent') from None
            self._fixed_factor = (system.n_assemblies, factor)
        fixed_value = system.fixed_value
        solution = np.zeros(pencil.shape[0], dtype=np.complex128)
        solution[pencil.fixed] = fixed_value
        if pencil.free.size:
            rhs = system.load_vector()[pencil.free]
            if pencil.fixed.size:
                rhs = rhs - pencil.coupling(0.0) @ fixed_value
            solution[pencil.free] = self._fixed_factor[1].solve_host(rhs)
        system.values[:] = 0.0
        system.values[system.global_ids] = solution
        return solution

    def solve(self, normal_displacement, omega) -> SqueezeFilmResponse:
        """Response to the gap-opening shape w [m per unit modal coordinate] at omega [rad/s] >= 0, per unit velocity
        amplitude (module header: the film's own full coefficients go through its mass matrix)."""
        system = self.system
        values = None if callable(normal_displacement) else np.asarray(normal_displacement)
        if values is not None and values.ndim == 1 and values.shape == system.values.shape:
            mass = self._film_mass()
            shape = values.astype(np.complex128)[system.global_ids]
            load = np.zeros(system.values.shape, dtype=np.complex128)
            load[system.global_ids] = -(mass @ shape)
            system.set_source(None)
            system.set_frequency_load(load)
            pressure = self._solve_pressure(float(omega))
            force = complex(pressure @ (mass @ np.conj(shape)))
        else:
            from .assembly import assemble_load
            shape_function = self.shape_function(normal_displacement)
            system.set_frequency_load(None)
            system.set_source(lambda points, elements: -np.asarray(shape_function(points, elements)))
            pressure = self._solve_pressure(float(omega))
            conjugate_shape = assemble_load(system.dofs,
                                            lambda points, elements: np.conj(shape_function(points, elements)))
            force = complex(pressure @ conjugate_shape)
        return SqueezeFilmResponse(omega=float(omega), pressure=system.values.copy(), force=force,
                                   damping=-force.real, stiffness=float(omega) * force.imag)


def mode_gas_damping(film, shapes, omegas, effective_masses) -> list:
    """Per mode (gap-opening shape w, omega [rad/s], m_eff [kg] of the same normalisation) the row {frequency [Hz],
    effective_mass [kg], gas_damping c_psi [kg/s], gas_stiffness k_gas [N/m], gamma_gas [1/s], q_gas}."""
    rows = []
    for shape, omega, effective_mass in zip(shapes, omegas, effective_masses, strict=True):
        response = film.solve(shape, omega)
        rows.append({'frequency': float(omega) / (2.0 * math.pi), 'effective_mass': float(effective_mass),
                     'gas_damping': response.damping, 'gas_stiffness': response.stiffness,
                     'gamma_gas': response.decay_rate(effective_mass),
                     'q_gas': response.quality_factor(effective_mass)})
    return rows


def film_table(rows, **extras) -> dict:
    """'table' result (FILM_TABLE_COLUMNS, one row per mode_gas_damping row, numbered from 1) plus the extras."""
    table_rows = [[number, row['frequency'], row['effective_mass'], row['gas_damping'], row['gas_stiffness'],
                   row['gamma_gas'], row['q_gas']] for number, row in enumerate(rows, start=1)]
    return {'kind': 'table', 'columns': list(FILM_TABLE_COLUMNS), 'units': list(FILM_TABLE_UNITS), 'rows': table_rows,
            **extras}


def gas_damping_totals(films):
    """Per resonance [c, k, gamma, Q] summed over the evaluated films (objects with `rows` of one solve, None = not
    evaluated: c, k and gamma add, 1 / Q adds); None when no film was evaluated."""
    evaluated = [film.rows for film in films if film.rows is not None]
    if not evaluated:
        return None
    totals = []
    for rows in zip(*evaluated, strict=True):
        damping = sum(row['gas_damping'] for row in rows)
        stiffness = sum(row['gas_stiffness'] for row in rows)
        effective_mass = rows[0]['effective_mass']
        omega = 2.0 * math.pi * rows[0]['frequency']
        totals.append([damping, stiffness, damping / effective_mass,
                       math.inf if damping == 0.0 else omega * effective_mass / damping])
    return totals

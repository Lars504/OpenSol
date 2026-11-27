"""Frequency-dependent isotropic moduli of the FEM solids (frequency_response): the generalized Maxwell / Kelvin-Voigt
law G*(omega), K*(omega) per element, the member mask of each formulation and the parameter groups the harmonic system
assembles once (one fixed real term per group and part); host numpy (core.backend.to_host for the material reads)."""
# Law (exp(+j omega t)): G*(w) = G0 (1 + j eta) + j w mu_s + sum_i G_i j w tau_i / (1 + j w tau_i), K* alike with
# bulk_viscosity and the bulk branches; eta = mechanical_loss_factor multiplies the relaxed G0, K0 only. Relaxed moduli
# per element: the youngs_modulus route (G0 = mu, K0 = lambda + 2/3 mu of materials.lame_coefficients, the values the
# solver's stiffness uses) or the shear_modulus / bulk_modulus route (bulk unset = incompressible, K0 = inf: the mixed
# formulation only). Every study but frequency_response reads G0, K0 alone. Displacement formulation: the relaxed part
# G0 (1 + j eta) D + K0 (1 + j eta) V is inside K + j L of the system (an isotropic C = G C_shear + K C_bulk exactly,
# stiffness.UNIT_MODULI), so a group adds only its excess G*(w) - G0 (1 + j eta) (ModulusLaw.excess) on the unit
# matrices D_g, V_g; groups over the omega-dependent members only. Mixed formulation: groups over every member
# (the pressure zones).
import math
from dataclasses import dataclass

import numpy as np

from ...core.backend import to_host
from .materials import (BULK_BRANCH_ARRAYS, LOSS_ARRAYS, MODULUS_ARRAYS, SHEAR_BRANCH_ARRAYS, VISCOELASTIC_ARRAYS,
                        VISCOELASTIC_BRANCHES, lame_coefficients)

DISPLACEMENT = 'displacement'
MIXED = 'mixed'
FORMULATIONS = (DISPLACEMENT, MIXED)   # setting formulation (definition_common.FORMULATION_PARAM)
# distinct parameter tuples a harmonic system assembles as fixed terms (one gradient pass, one term per group and
# part); beyond it a ValueError (spatially varying moduli: make them piecewise constant per region)
MAX_GROUPS = 32
_PARTS = ('shear', 'bulk')
# columns of a parameter row: G0, K0, eta, mu_s, mu_b, then (modulus, time) per shear branch and per bulk branch
_SHEAR_BRANCH_COLUMNS = slice(5, 5 + 2 * VISCOELASTIC_BRANCHES)
_BULK_BRANCH_COLUMNS = slice(5 + 2 * VISCOELASTIC_BRANCHES, 5 + 4 * VISCOELASTIC_BRANCHES)


@dataclass(frozen=True)
class ModulusLaw:
    """One modulus G*(omega) or K*(omega) (module header): relaxed G0 / K0 [Pa] (math.inf = incompressible, a bulk law
    of the mixed formulation), the loss factor eta (on the relaxed modulus only), the dashpot viscosity [Pa s] and the
    Maxwell branches ((modulus [Pa], relaxation time [s]), ...) with modulus > 0."""
    relaxed: float
    loss_factor: float
    viscosity: float
    branches: tuple

    @property
    def frequency_dependent(self) -> bool:
        """A viscosity or a Maxwell branch: the law changes with omega."""
        return self.viscosity > 0.0 or bool(self.branches)

    @property
    def incompressible(self) -> bool:
        return math.isinf(self.relaxed)

    def excess(self, omega):
        """The omega-dependent part j omega mu + sum_i G_i j omega tau_i / (1 + j omega tau_i) [Pa] at omega [rad/s]: a
        number (complex) or an array (complex128 of omega's shape, the same expressions elementwise)."""
        if np.ndim(omega) > 0:
            jw = 1j * np.asarray(omega, dtype=np.float64)
            value = jw * self.viscosity
            for modulus, time in self.branches:
                value = value + modulus * jw * time / (1.0 + jw * time)   # a new array: jw is never aliased
            return np.asarray(value, dtype=np.complex128)
        jw = 1j * float(omega)
        value = jw * self.viscosity
        for modulus, time in self.branches:
            value += modulus * jw * time / (1.0 + jw * time)
        return complex(value)

    def at(self, omega):
        """G*(omega) / K*(omega) [Pa] at omega [rad/s], a number (complex) or an array (complex128 of omega's shape);
        complex(inf) (per element for an array) for an incompressible law."""
        if self.incompressible:
            if np.ndim(omega) > 0:
                return np.full(np.shape(omega), complex(math.inf, 0.0), dtype=np.complex128)
            return complex(math.inf, 0.0)
        return complex(self.relaxed * (1.0 + 1j * self.loss_factor)) + self.excess(omega)

    def compliance(self, omega) -> complex:
        """1 / at(omega) [1/Pa]; 0 for an incompressible law."""
        if self.incompressible:
            return 0j
        return 1.0 / self.at(omega)


@dataclass(frozen=True, eq=False)
class ElementModuli:
    """Host float64 per mesh element (n,), the unset values outside the window: relaxed `shear` G0 (0 unset) and `bulk`
    K0 (inf unset), `loss` eta, `viscosity`, `bulk_viscosity`, the branches (n, VISCOELASTIC_BRANCHES, 2) of (modulus,
    time) (time 0 where the modulus is 0), and the route of each element (bool: youngs_modulus > 0, or a shear_modulus /
    bulk_modulus set; c_ij elements have neither)."""
    shear: np.ndarray
    bulk: np.ndarray
    loss: np.ndarray
    viscosity: np.ndarray
    bulk_viscosity: np.ndarray
    shear_branches: np.ndarray
    bulk_branches: np.ndarray
    youngs_route: np.ndarray
    moduli_route: np.ndarray

    @property
    def frequency_dependent(self) -> np.ndarray:
        """Bool (n,): a viscosity, bulk viscosity or branch in the element."""
        return ((self.viscosity > 0.0) | (self.bulk_viscosity > 0.0) | (self.shear_branches[..., 0] > 0.0).any(axis=1)
                | (self.bulk_branches[..., 0] > 0.0).any(axis=1))

    @property
    def shear_alive(self) -> np.ndarray:
        """Bool (n,): G*(omega) is not identically 0 (G0 > 0, a viscosity or a shear branch)."""
        return (self.shear > 0.0) | (self.viscosity > 0.0) | (self.shear_branches[..., 0] > 0.0).any(axis=1)


@dataclass(frozen=True, eq=False)
class ViscoelasticGroup:
    """Elements (bool per mesh element) sharing one parameter tuple: their shear and bulk ModulusLaw."""
    index: int
    elements: np.ndarray
    shear: ModulusLaw
    bulk: ModulusLaw

    def law(self, part) -> ModulusLaw:
        """The ModulusLaw of part 'shear' / 'bulk'."""
        return self.shear if part == 'shear' else self.bulk


def element_moduli(materials, window, youngs_modulus, poisson_ratio, label, required=False):
    """ElementModuli of a MaterialGrid on the window elements (bool (n,)), youngs_modulus / poisson_ratio the host
    window values the solver reads (0 outside); reads the MODULUS / VISCOELASTIC arrays only when allocated, None when
    none is (and not required: the mixed formulation always needs the moduli). ValueError for both routes in one
    element, a bulk_modulus <= 0, a branch modulus without a relaxation time > 0 (or a time without modulus), and
    bulk_viscosity / a bulk branch without a finite bulk modulus."""
    names = set(materials.property_names())
    if not required and names.isdisjoint(MODULUS_ARRAYS + tuple(sorted(VISCOELASTIC_ARRAYS))):
        return None
    window = np.asarray(window, dtype=bool)
    n_elements = window.size
    # booked (estimate): ElementModuli holds 5 + 4 VISCOELASTIC_BRANCHES float64 + 2 bool per element (kept per assembly)

    def read(name, unset):
        if name not in names:
            return np.full(n_elements, unset)
        return np.where(window, np.asarray(to_host(getattr(materials, name)), dtype=np.float64), unset)

    youngs_modulus = np.asarray(youngs_modulus, dtype=np.float64)
    youngs_route = youngs_modulus > 0.0
    shear_set, bulk_set = read('shear_modulus', math.nan), read('bulk_modulus', math.nan)
    moduli_route = window & (np.isfinite(shear_set) | np.isfinite(bulk_set))
    _refuse(youngs_route & moduli_route, label, 'set youngs_modulus together with shear_modulus / bulk_modulus: one '
                                               'route per material (E / nu or G0 / K0)')
    _refuse(np.isfinite(bulk_set) & ~(bulk_set > 0.0), label, 'have bulk_modulus <= 0 (leave it unset for an '
                                                             'incompressible material of the mixed formulation)')
    lame_lambda, mu = lame_coefficients(youngs_modulus * youngs_route, np.asarray(poisson_ratio) * youngs_route)
    shear = np.where(youngs_route, mu, np.where(np.isfinite(shear_set), shear_set, 0.0))
    bulk = np.where(youngs_route, lame_lambda + (2.0 / 3.0) * mu, np.where(np.isfinite(bulk_set), bulk_set, math.inf))
    branches = {}
    for part, arrays in zip(_PARTS, (SHEAR_BRANCH_ARRAYS, BULK_BRANCH_ARRAYS)):
        stacked = np.zeros((n_elements, VISCOELASTIC_BRANCHES, 2))
        for index, (modulus_name, time_name) in enumerate(arrays):
            modulus, time = read(modulus_name, 0.0), read(time_name, 0.0)
            _refuse((modulus > 0.0) & ~(time > 0.0), label, f'set {modulus_name} > 0 without {time_name} > 0')
            _refuse((time > 0.0) & ~(modulus > 0.0), label, f'set {time_name} without {modulus_name} > 0')
            stacked[:, index, 0] = modulus
            stacked[:, index, 1] = np.where(modulus > 0.0, time, 0.0)
        branches[part] = stacked
    bulk_viscosity = read('bulk_viscosity', 0.0)
    _refuse(((bulk_viscosity > 0.0) | (branches['bulk'][..., 0] > 0.0).any(axis=1)) & ~np.isfinite(bulk), label,
            'set bulk_viscosity or a bulk relaxation branch without a finite bulk modulus (bulk_modulus or '
            'youngs_modulus)')
    return ElementModuli(shear, bulk, read(LOSS_ARRAYS[0], 0.0), read('viscosity', 0.0), bulk_viscosity,
                         branches['shear'], branches['bulk'], youngs_route, moduli_route)


def member_mask(moduli, window, formulation, label, solid):
    """Bool (n,) of the elements the formulation solves (`solid`: the relaxed ElementStiffness c11 > 0 of the window,
    which covers the c_ij elements). displacement: `solid` (a viscosity-only cell is void); ValueError for a
    shear_modulus without bulk_modulus, no shear stiffness (bulk_modulus only, shear_modulus 0) and branches / a bulk
    viscosity on a cell without a relaxed modulus. mixed: every window element whose G*(omega) is not identically 0
    (liquids included); ValueError for c_ij elements (isotropic only) and bulk-only cells."""
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {', '.join(FORMULATIONS)}, got {formulation!r}")
    window = np.asarray(window, dtype=bool)
    solid = np.asarray(solid, dtype=bool) & window
    if moduli is None:
        if formulation == MIXED:
            raise ValueError(f'{label()}: the mixed formulation needs the element moduli (element_moduli required)')
        return solid
    bulk_parts = (moduli.bulk_viscosity > 0.0) | (moduli.bulk_branches[..., 0] > 0.0).any(axis=1)
    shear_branches = (moduli.shear_branches[..., 0] > 0.0).any(axis=1)
    if formulation == DISPLACEMENT:
        _refuse(window & moduli.moduli_route & ~np.isfinite(moduli.bulk), label,
                'set shear_modulus without bulk_modulus: the displacement formulation needs a finite bulk modulus '
                '(formulation mixed takes an unset one as incompressible)')
        _refuse(window & moduli.moduli_route & ~(moduli.shear > 0.0), label,
                'have no shear stiffness (bulk_modulus only or shear_modulus 0): such a liquid cell needs formulation '
                'mixed (frequency_response)')
        _refuse(window & ~solid & (shear_branches | bulk_parts), label,
                'set relaxation branches or bulk_viscosity without a relaxed modulus: the displacement formulation '
                'needs youngs_modulus or shear_modulus with bulk_modulus there')
        return solid
    _refuse(solid & ~moduli.youngs_route & ~moduli.moduli_route, label,
            'take c_ij entries: the mixed formulation is isotropic (youngs_modulus / poisson_ratio or shear_modulus / '
            'bulk_modulus)')
    alive = window & moduli.shear_alive
    _refuse(window & (moduli.moduli_route | bulk_parts) & ~alive, label,
            'have neither shear stiffness nor viscosity (bulk_modulus only): G*(omega) = 0, the mixed formulation has '
            'nothing to solve there')
    return alive


def _refuse(mask, label, text) -> None:
    """ValueError '<label>: N element(s) <text>' when any element of the bool mask is set."""
    count = int(np.count_nonzero(mask))
    if count:
        raise ValueError(f'{label()}: {count} element(s) {text}')


def _parameter_rows(moduli, selected) -> np.ndarray:
    """(m, 5 + 4 VISCOELASTIC_BRANCHES) parameter rows of the element indices selected (columns: module constants)."""
    return np.column_stack([moduli.shear[selected], moduli.bulk[selected], moduli.loss[selected],
                            moduli.viscosity[selected], moduli.bulk_viscosity[selected],
                            moduli.shear_branches[selected].reshape(selected.size, -1),
                            moduli.bulk_branches[selected].reshape(selected.size, -1)])


def _law(row, part) -> ModulusLaw:
    """ModulusLaw of part 'shear' / 'bulk' of one parameter row."""
    columns = _SHEAR_BRANCH_COLUMNS if part == 'shear' else _BULK_BRANCH_COLUMNS
    pairs = np.asarray(row[columns]).reshape(-1, 2)
    branches = tuple((float(modulus), float(time)) for modulus, time in pairs if modulus > 0.0)
    relaxed, viscosity = (row[0], row[3]) if part == 'shear' else (row[1], row[4])
    return ModulusLaw(float(relaxed), float(row[2]), float(viscosity), branches)


def parameter_groups(moduli, elements, label, max_groups=MAX_GROUPS) -> tuple:
    """ViscoelasticGroup per distinct parameter tuple of the elements (bool per mesh element), in ascending row order;
    () without elements. ValueError above max_groups (spatially varying moduli)."""
    elements = np.asarray(elements, dtype=bool)
    selected = np.flatnonzero(elements)
    if not selected.size:
        return ()
    unique, inverse = np.unique(_parameter_rows(moduli, selected), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    if len(unique) > max_groups:
        raise ValueError(f'{label()}: {len(unique)} distinct viscoelastic parameter sets (moduli, loss factor, '
                         f'viscosities, relaxation branches) exceed {max_groups}: each one is a fixed term of the '
                         'frequency response; make the properties piecewise constant per region instead of varying '
                         'with x / y / z or the state')
    groups = []
    for index, row in enumerate(unique):
        mask = np.zeros(elements.size, dtype=bool)
        mask[selected[inverse == index]] = True
        groups.append(ViscoelasticGroup(index, mask, _law(row, 'shear'), _law(row, 'bulk')))
    return tuple(groups)


def group_indices(groups, n_elements) -> np.ndarray:
    """Int64 (n_elements,) group index per mesh element (-1 = in no group): the group assemblers' input."""
    indices = np.full(int(n_elements), -1, dtype=np.int64)
    for group in groups:
        indices[group.elements] = group.index
    return indices


def harmonic_parts(moduli, members, omega) -> tuple:
    """(G*, K*) complex128 (n,) per element at omega [rad/s] of the module law (0 off the members; K* = inf + 0j where
    incompressible): the per-element values the group laws hold, for the complex-correct outputs."""
    jw = 1j * float(omega)
    members = np.asarray(members, dtype=bool)

    def part(relaxed, viscosity, branches):
        finite = np.isfinite(relaxed)
        value = np.where(finite, relaxed, 0.0) * (1.0 + 1j * moduli.loss) + jw * viscosity
        modulus, time = branches[..., 0], branches[..., 1]
        value = value + np.sum(modulus * jw * time / (1.0 + jw * time), axis=1)
        value = np.where(finite, value, np.complex128(math.inf))
        return np.where(members, value, 0.0 + 0.0j)
    return (part(moduli.shear, moduli.viscosity, moduli.shear_branches),
            part(moduli.bulk, moduli.bulk_viscosity, moduli.bulk_branches))

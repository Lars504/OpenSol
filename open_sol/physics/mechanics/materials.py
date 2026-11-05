"""Material vocabulary of the solid (2D plane strain and 3D): Young's modulus, Poisson ratio, the anisotropic c_ij
(2D ten, 3D 21), the piezoelectric e and clamped permittivity (2D in-plane, 3D full), the thermal expansion arrays (the alpha_xx / alpha_yy arrays are shared with heat, whose thermo-optic coupling reads them too). Imports only solvers.pcg (numpy, no cupy)."""
# Solve dtype name (estimators, script constructors) -> solvers.pcg.PCGSolver precision name.
from ...solvers.pcg import DTYPE_PRECISIONS as SOLVER_PRECISIONS

# youngs_modulus = 0 by default: a cell without stiffness is void (air, cladding left out of the solid).
MECHANICAL_SCHEMA = {
    'youngs_modulus': 0.0, 'poisson_ratio': 0.0,
    'alpha_xx': 0.0, 'alpha_yy': 0.0, 'alpha_zz': 0.0,
    'photoelastic_p11': 0.0, 'photoelastic_p12': 0.0,
}

# Mechanical property -> MaterialGrid arrays it writes; isotropic alpha expands to all three expansion arrays.
MECHANICAL_PROPERTY_ARRAYS = {
    'youngs_modulus': ('youngs_modulus',), 'poisson_ratio': ('poisson_ratio',),
    'alpha': ('alpha_xx', 'alpha_yy', 'alpha_zz'), 'alpha_xx': ('alpha_xx',), 'alpha_yy': ('alpha_yy',),
    'alpha_zz': ('alpha_zz',),
    'photoelastic_p11': ('photoelastic_p11',), 'photoelastic_p12': ('photoelastic_p12',),
    'photoelastic_p44': ('photoelastic_p44',),
    'rho': ('rho',),
    'piezo_e11': ('piezo_e11',), 'piezo_e12': ('piezo_e12',), 'piezo_e16': ('piezo_e16',),
    'piezo_e21': ('piezo_e21',), 'piezo_e22': ('piezo_e22',), 'piezo_e26': ('piezo_e26',),
    'eps_dc': ('eps_dc_xx', 'eps_dc_yy'), 'eps_dc_xx': ('eps_dc_xx',), 'eps_dc_yy': ('eps_dc_yy',),
    'eps_dc_xy': ('eps_dc_xy',),
    'piezo_e13': ('piezo_e13',), 'piezo_e23': ('piezo_e23',),
    'piezo_e14': ('piezo_e14',), 'piezo_e15': ('piezo_e15',), 'piezo_e24': ('piezo_e24',), 'piezo_e25': ('piezo_e25',),
    'piezo_e31': ('piezo_e31',), 'piezo_e32': ('piezo_e32',), 'piezo_e33': ('piezo_e33',), 'piezo_e34': ('piezo_e34',),
    'piezo_e35': ('piezo_e35',), 'piezo_e36': ('piezo_e36',),
    'eps_dc_zz': ('eps_dc_zz',), 'eps_dc_xz': ('eps_dc_xz',), 'eps_dc_yz': ('eps_dc_yz',),
    'c11': ('c11',), 'c12': ('c12',), 'c13': ('c13',), 'c22': ('c22',), 'c23': ('c23',), 'c33': ('c33',),
    'c16': ('c16',), 'c26': ('c26',), 'c36': ('c36',), 'c66': ('c66',),
    'c14': ('c14',), 'c15': ('c15',), 'c24': ('c24',), 'c25': ('c25',), 'c34': ('c34',), 'c35': ('c35',),
    'c44': ('c44',), 'c45': ('c45',), 'c46': ('c46',), 'c55': ('c55',), 'c56': ('c56',),
    'mechanical_loss_factor': ('mechanical_loss_factor',), 'dielectric_loss_tangent': ('dielectric_loss_tangent',),
    **{f'initial_stress_{pair}': (f'initial_stress_{pair}',) for pair in ('xx', 'yy', 'zz', 'yz', 'xz', 'xy')},
}
# Piezoelectric plane strain (2D, setting piezoelectric): the reduced in-plane e block (the elastic_waves names and
# order), the out-of-plane column e_k3 (stress_zz only) and the clamped static permittivity (the electrostatics names),
# optional arrays: a material setting a piezo_e* allocates all eleven (any eps_dc entry the three permittivities); the
# solver adds missing ones when piezoelectric.
PIEZO_ARRAYS = ('piezo_e11', 'piezo_e12', 'piezo_e16', 'piezo_e21', 'piezo_e22', 'piezo_e26')
PIEZO_OUT_OF_PLANE_ARRAYS = ('piezo_e13', 'piezo_e23')
PERMITTIVITY_ARRAYS = ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_xy')
_PIEZO_TRIGGERS = PIEZO_ARRAYS + PIEZO_OUT_OF_PLANE_ARRAYS
PIEZO_DEFAULTS = {**{name: 0.0 for name in _PIEZO_TRIGGERS}, 'eps_dc_xx': 1.0, 'eps_dc_yy': 1.0, 'eps_dc_xy': 0.0}
_PERMITTIVITY_TRIGGERS = _PIEZO_TRIGGERS + ('eps_dc', *PERMITTIVITY_ARRAYS)
OPTIONAL_PIEZO_SCHEMA = {name: (default, _PIEZO_TRIGGERS if name in _PIEZO_TRIGGERS else _PERMITTIVITY_TRIGGERS)
                         for name, default in PIEZO_DEFAULTS.items()}
ELECTROMECHANICAL_ARRAYS = frozenset(PIEZO_DEFAULTS)
# Piezoelectric, 3D (solid_mechanics_3d on tetrahedra, setting piezoelectric): the full e_iJ (i = field x / y / z, J =
# Voigt; the 2D names keep their meaning) and the six clamped permittivities; a piezo_e* entry allocates all 18, an
# eps_dc* entry (or a piezo_e*) the six permittivities; the solver adds missing ones when piezoelectric.
PIEZO_TENSOR_ARRAYS_3D = tuple(f'piezo_e{row}{column}' for row in range(1, 4) for column in range(1, 7))
PERMITTIVITY_TENSOR_ARRAYS_3D = ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_zz', 'eps_dc_yz', 'eps_dc_xz', 'eps_dc_xy')
PIEZO_DEFAULTS_3D = {**{name: 0.0 for name in PIEZO_TENSOR_ARRAYS_3D},
                     **{name: (1.0 if name[-1] == name[-2] else 0.0) for name in PERMITTIVITY_TENSOR_ARRAYS_3D}}
_PERMITTIVITY_TRIGGERS_3D = PIEZO_TENSOR_ARRAYS_3D + ('eps_dc', *PERMITTIVITY_TENSOR_ARRAYS_3D)
OPTIONAL_PIEZO_SCHEMA_3D = {
    name: (default, PIEZO_TENSOR_ARRAYS_3D if name in PIEZO_TENSOR_ARRAYS_3D else _PERMITTIVITY_TRIGGERS_3D)
    for name, default in PIEZO_DEFAULTS_3D.items()}
ELECTROMECHANICAL_ARRAYS_3D = frozenset(PIEZO_DEFAULTS_3D)
# Anisotropic stiffness (2D solid_mechanics only; c11 / c12 / c22 / c66 / c16 / c26 are elastic_waves' arrays too) in
# Voigt order 1 xx, 2 yy, 3 zz, 6 xy: optional arrays, any entry set allocates all but c16 / c26 (those: either of them);
# NaN = unset = the isotropic value
# of youngs_modulus / poisson_ratio per component, the shear couplings c16 / c26 / c36 default 0 (physics/mechanics
# stiffness.py).
STIFFNESS_TENSOR_ARRAYS = ('c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c16', 'c26', 'c36', 'c66')
SHEAR_COUPLING_ARRAYS = ('c16', 'c26', 'c36')
# c16 / c26 keep elastic_waves' trigger (either of them only: an allocated pair switches its monoclinic kernel on); an
# unallocated array reads as unset (0 for the shear couplings)
_MONOCLINIC_PAIR = ('c16', 'c26')
OPTIONAL_STIFFNESS_SCHEMA = {name: (0.0 if name in SHEAR_COUPLING_ARRAYS else float('nan'),
                                    _MONOCLINIC_PAIR if name in _MONOCLINIC_PAIR else STIFFNESS_TENSOR_ARRAYS)
                             for name in STIFFNESS_TENSOR_ARRAYS}
# Anisotropic stiffness, 3D (solid_mechanics_3d on tetrahedra): the 21 Voigt entries c_IJ, I <= J (1 xx, 2 yy, 3 zz,
# 4 yz, 5 xz, 6 xy, engineering shear; the 2D names keep their meaning); any entry set allocates all 21. NaN = unset =
# the isotropic value of youngs_modulus / poisson_ratio (stiffness3d.py); the entries outside the isotropic pattern
# default 0 (as the 2D shear couplings).
STIFFNESS_TENSOR_ARRAYS_3D = tuple(f'c{row}{column}' for row in range(1, 7) for column in range(row, 7))
ISOTROPIC_PATTERN_3D = ('c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c44', 'c55', 'c66')
OPTIONAL_STIFFNESS_SCHEMA_3D = {
    name: (float('nan') if name in ISOTROPIC_PATTERN_3D else 0.0, STIFFNESS_TENSOR_ARRAYS_3D)
    for name in STIFFNESS_TENSOR_ARRAYS_3D}
# Losses of the frequency_response study (2D only): c -> c (1 + j eta), eps -> eps (1 - j tan delta); optional arrays
# (each allocated when a material sets it), default 0.
LOSS_ARRAYS = ('mechanical_loss_factor', 'dielectric_loss_tangent')
OPTIONAL_LOSS_SCHEMA = {name: (0.0, (name,)) for name in LOSS_ARRAYS}
# Initial (residual) stress sigma0 [Pa] per cell (new_modules_plan K2; triangle / tetrahedron meshes only): total stress
# = sigma0 + C eps - beta dT; each array allocated when a material sets it, default 0. 2D in fem.elasticity row order:
# planar (xx, yy, xy), axisymmetric (rr = xx, zz = yy, hoop = zz, rz = xy), plus zz (planar out of plane, plane strain
# only); 3D Voigt (xx, yy, zz, yz, xz, xy).
PRESTRESS_ARRAYS = ('initial_stress_xx', 'initial_stress_yy', 'initial_stress_xy', 'initial_stress_zz')
PRESTRESS_ARRAYS_3D = ('initial_stress_xx', 'initial_stress_yy', 'initial_stress_zz', 'initial_stress_yz',
                       'initial_stress_xz', 'initial_stress_xy')
OPTIONAL_PRESTRESS_SCHEMA = {name: (0.0, (name,)) for name in PRESTRESS_ARRAYS}
OPTIONAL_PRESTRESS_SCHEMA_3D = {name: (0.0, (name,)) for name in PRESTRESS_ARRAYS_3D}
PRESTRESS_NAMES = frozenset(PRESTRESS_ARRAYS_3D)
# Density (shared with heat / elastic_waves): only the eigenfrequency solves (2D and 3D) read it (consistent mass).
DENSITY_SCHEMA = {'rho': 0.0}
# Shear photoelastic coefficient (2D only): only the optomechanical g0 overlap reads it; NaN = unset = (p11 - p12) / 2.
SHEAR_PHOTOELASTIC_SCHEMA = {'photoelastic_p44': float('nan')}

# Arrays the assembled stiffness depends on, arrays only the thermal load reads, arrays only the strain-optic coupling
# reads (a change re-solves anyway, so the coupling sees a new revision).
STIFFNESS_ARRAYS = frozenset({'youngs_modulus', 'poisson_ratio', *STIFFNESS_TENSOR_ARRAYS, *STIFFNESS_TENSOR_ARRAYS_3D})
EXPANSION_ARRAYS = frozenset({'alpha_xx', 'alpha_yy', 'alpha_zz'})
PHOTOELASTIC_ARRAYS = frozenset({'photoelastic_p11', 'photoelastic_p12'})


def lame_coefficients(youngs_modulus, poisson_ratio):
    """(lambda, mu) [Pa] of an isotropic solid from E [Pa] and nu (arrays or scalars, -1 < nu < 0.5)."""
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    return lame_lambda, mu

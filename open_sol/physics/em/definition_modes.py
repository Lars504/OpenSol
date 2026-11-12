"""Model definition of 'em_modes' (full-vector cross-section modes: settings, material vocabulary, mode table /
dispersion features) on definition_common, with its templates and self-tests; definition.register calls register."""
from ...core import time_estimate as te
from ...core.memory_estimate import COMPLEX128_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.definition_helpers import eigen_factorization, eigen_role
from ...model.schema import FeatureType, Issue, Param, PhysicsType, SolverRole, feature_path
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, MIXED_FACTORIZATIONS
from ...solvers.line_block import factor_solve_ops
from .definition_common import (BOTH_GRIDS, CPML_BC_HELP, DEFAULT_WAVELENGTH, EM_FAMILY, FREQUENCY_DOMAIN_STUDY,
                                HOST_DIRECT_NAMES, OPTOMECHANICS_STUDY, PERIODIC_BC_HELP, all_solved_wavelengths,
                                apply_pec,
                                boundary_settings_params, eigen_backend_kwargs, em_material_params, fem_size_issues,
                                first_solved_wavelength, kept_line_block_factors, make_validate_physics, mesh_kind,
                                object_result, other_em_physics_labels, physics_counter, raw_settings, role_counter,
                                scipy_issues, shortest_solved_wavelength, source_slot_docs, unused_wavelength_issues,
                                write_dispersive_materials)
from .cross_section_modes import (BEND_MODELS, LINE_BLOCK_BOTH_PERIODIC_TEXT, cpml_cross_section, line_block_axis,
                                  mode_default_basis, size_issue_for)
from .definition_bloch import bloch_active_axes, bloch_params, fdfd_bloch_issues, fdfd_domain_requirements
from .fem_modes import DISCRETIZATION_ORDERS, FEM_ENTRIES_PER_ROW, FEM_LU_FILL_FACTOR, PERIODIC_TEXT, estimated_unknowns
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import modes_memory_estimate, pec_memory_estimate
from .nonlinear_overlap import RING_CRYSTALS, VOIGT_NAMES, shg_overlap_result, voigt_tensor
from .renderer import ModeRenderer
from .selftests_carriers import (PN_ELECTROSTATICS_DESCRIPTION, PN_ELECTROSTATICS_LABEL, PN_PRESCRIBED_DESCRIPTION,
                                 PN_PRESCRIBED_LABEL, pn_phase_shifter_depletion_spec,
                                 pn_phase_shifter_electrostatics_spec)
from .selftests_em_modes import (STRIP_WAVEGUIDE_MODES_DESCRIPTION, STRIP_WAVEGUIDE_MODES_LABEL,
                                 THERMO_OPTIC_PHASE_SHIFTER_DESCRIPTION, THERMO_OPTIC_PHASE_SHIFTER_LABEL,
                                 strip_waveguide_modes_spec, thermo_optic_phase_shifter_spec)
from .solvers.modes_2d import (MODE_PERMITTIVITY_ARRAYS, MODE_PERMITTIVITY_SLOTS, MODE_POLARIZATION, ModePhysics,
                               bend_factor_message, bend_overlap_result, mode_dispersion_result, mode_table_result)
from .solvers.modes_fem import FEMModePhysics

_TYPE_NAME = 'em_modes'
_MODE_ANALYSIS_STUDY = 'mode_analysis'
_THERMAL_NOISE_STUDY = 'thermal_noise'   # solved first there: its modes give heat's mode-pair noise weightings
# EM types that may share a thermal_noise model with em_modes: they solve the same cross-section at k_z = 0
_CROSS_SECTION_ILLUMINATION_TYPES = ('em_fdfd_ez', 'em_fdfd_hz')

# ---- settings ----
_BC_HELP = ' '.join((
    CPML_BC_HELP,
    'In the mode solve it is the complex coordinate stretch s = kappa + sigma / (alpha + j omega eps0) with the FDTD '
    'grading of the pml_* settings: a mode whose tail reaches it gains a small artificial loss (Im n_eff, see '
    'pml_fraction), and modes with more than half of their |H|^2 in PML cells are dropped.',
    PERIODIC_BC_HELP,
    'Walls and mirrors all sit on one plane, the inner face of the outermost cell (the outermost cell is a ghost: '
    'each field component centred along that axis takes -1 (pec, antisymmetric) or +1 (pmc, symmetric) times its '
    'inner neighbour, the components on the plane are free (pec) or 0 (pmc)); that is half a cell off the em_fdtd_ez '
    'pec and em_fdtd_hz pmc walls, so a guide between two pec walls of a domain n cells wide is n - 2 cells wide. '
    'Mirrors, on any face, add their ghost cell outside the domain, so their plane is the domain edge: symmetric = E '
    'mirror-even (a PMC plane), antisymmetric = E mirror-odd (a PEC plane). A '
    'mirror through a symmetric guide halves the unknowns and keeps one parity: on x_lo, antisymmetric keeps the modes '
    'whose E_x is even about the plane (quasi-TE, e.g. TE0 of a strip), symmetric those whose E_y is even (quasi-TM).',
))
_WAVELENGTH_HELP = (
    'Vacuum wavelength lambda0 of the mode solve in a mode_analysis study (omega = 2 pi C0 / lambda0; k0 = omega / '
    'C0). The materials enter at that omega (sigma_e and every Drude / Lorentz pole as a complex permittivity). A '
    'frequency_domain study solves its own wavelengths instead (validation info when this is not the default) and '
    'gives n_eff versus wavelength (mode_dispersion). The mesh needs >= 10 (better 20) cells per lambda0 / |n| in every '
    'material and, in metals, >= 2 (better 4) per skin depth lambda0 / (2 pi |Im n|); validation checks both. A 220 nm '
    'silicon core needs ~20 nm cells (or a refinement) for n_eff to ~1e-2. Typical 1.31e-6 or 1.55e-6 m.')
_SETTINGS = boundary_settings_params(_BC_HELP) + (
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help=_WAVELENGTH_HELP,
          tip='Vacuum wavelength at which the modes are computed, in metres: 1.55e-6 for 1550 nm.\n\n'
              "A 'Frequency domain' study ignores it and computes the modes at its own list of wavelengths. The mesh "
              'needs at least 10 cells per wavelength inside every material.'),
    Param('n_modes', 'int', 4, label='Number of modes', min=1, max=100,
          help='Modes kept per solve: the eigen-solve finds the 2 n_modes + 10 eigenvalues beta^2 nearest the target '
               '(k0 n_eff_guess)^2, drops the modes with more than half of their |H|^2 in PML cells, ranks the rest by '
               'descending Re(beta^2), ties by descending Im(beta^2) (= descending Re n_eff for low-loss guided modes; '
               'evanescent modes, Re(beta^2) < 0, come last) and keeps up to n_modes (fewer may come back, e.g. near '
               'cutoff or when most candidates are PML modes). More modes cost more eigensolver work; 2-10 typical.',
          tip='How many modes to compute, from the most strongly guided one down.\n\n'
              'Fewer may be found when the guide supports fewer. 2 to 10 is typical.'),
    Param('n_eff_guess', 'float', None, label='Effective index guess', unit='1', min=0.0, min_exclusive=True,
          optional=True,
          help='Effective index around which modes are sought: the shift-invert target is sigma = (k0 n_eff_guess)^2, '
               'so the modes whose n_eff^2 lies nearest n_eff_guess^2 are found. Empty: the largest Re sqrt(eps mu) '
               'over the non-PEC cells and the three permittivity components at the solve wavelength (before any '
               'bend map), which finds the most strongly guided modes. Set '
               'it lower to reach higher-order modes, or near the expected value for a surface plasmon or a mode far '
               'below the core index.',
          tip='Effective index near which the modes are searched.\n\n'
              'Empty searches below the highest refractive index of the cross-section, which finds the most strongly '
              'guided modes. Set it to reach higher-order modes.'),
    Param('mode_number', 'int', 1, label='Displayed mode', min=1,
          help='Mode number (1-based rank, descending Re(beta^2), see n_modes; as every mode_number) of the mode shown '
               'after the study: Results.field and the GUI field view, the state fields e_intensity / sz, and selected = True in mode_table. A frequency_domain study '
               'shows the mode of that rank at every wavelength (no tracking; mode_dispersion follows a mode by '
               'overlap). Must not exceed n_modes (validation error); when fewer modes are found the last one is '
               'shown (run-log warning).',
          tip='Which mode the field view shows after the run: 1 is the most strongly guided one.\n\n'
              'Every mode is also stored as its own snapshot.'),
    Param('group_index', 'bool', False, label='Compute group index',
          help='Also compute the group index n_g = C0 d(Re beta) / d(omega) of every mode by a central difference in '
               'omega (relative step 1e-3) with the materials re-evaluated at both frequencies (Drude / Lorentz chi '
               'and sigma_e: material dispersion included; plain eps_r values have none). Tensor meshes: two more '
               'eigen-solves per wavelength, each mode followed by overlap (about 3x the time); triangles meshes: the '
               'first-order perturbation of the complex symmetric pencil, two more assemblies and no eigen-solve. Off: '
               'n_group is NaN in the results.',
          tip='Also compute the group index of every mode, which sets the speed of light pulses in the guide.\n\n'
              'Takes about three times as long on a rectangular grid, a little longer on a triangle mesh.'),
    Param('bend_radius', 'float', 0.0, label='Bend radius', unit='m',
          help='Signed radius R of a waveguide bent in the xz plane (0 = straight). R > 0 puts the centre of curvature '
               'at x = x_ref - R (the outer side of the bend is +x), R < 0 at x_ref + |R| (outer side -x), x_ref = '
               'bend_reference_x; the guide runs around the y axis (rings in the chip plane: y vertical), radius r = |R| s with '
               's = 1 + (x - x_ref) / R at the x of each field point. bend_model sets the form (exact metric or the '
               'conformal map). n_eff is referred to the radius at x_ref: the angular propagation constant (azimuthal '
               'order of a ring) is nu = beta R = k0 n_eff R. Bend modes radiate '
               'toward the outer side, which shows as Im n_eff < 0 / loss_db_per_cm: keep a cpml face and enough '
               'cladding there (validation warns otherwise). s must stay > 0 over the whole grid, PML '
               'included, and the x axis must not be periodic (validation errors). Typical 5e-6 to 1e-4 m.',
          tip='Radius of a bend of the waveguide, in metres; 0 means straight.\n\n'
              'Positive values bend toward -x, so light leaks toward +x; keep an absorbing layer on that side.'),
    Param('bend_reference_x', 'float', None, label='Bend reference x', unit='m', optional=True,
          help='x coordinate in m (physical frame) where s = 1 (bend_radius): the radius to which n_eff is referred, '
               'usually the guide centre. Empty: the domain centre size_x / 2. Unused when bend_radius = 0.',
          tip='Horizontal position where the bend radius is measured, usually the waveguide centre.\n\n'
              'Empty uses the middle of the domain.'),
    Param('bend_model', 'choice', 'exact', label='Bend model', choices=BEND_MODELS,
          choice_labels=(('exact', 'Exact (cylindrical coordinates)'), ('conformal', 'Conformal map (approximate)')),
          help="Form of the bend (unused when bend_radius = 0). 'exact': the bent guide written exactly as a straight "
               'one in (x, y, R phi): eps and mu become eps diag(s, s, 1/s), mu diag(s, s, 1/s) (x, y, z components, '
               'each at its own field point), so the result is the true bend mode to discretisation order whenever '
               'the centre of curvature lies outside the grid; E_x, E_y, H_x, H_y are the physical E_r, E_y, H_r, H_y '
               'and E_z, H_z = s x the physical E_phi, H_phi; power and overlaps use the straight formulas. '
               "'conformal': the approximate map of the bend-loss literature, every permittivity component times "
               's^2, mu untouched: an error second order in w / R (w the mode width; e.g. -5e-4 relative in n_eff for '
               'the whispering-gallery mode of a 50 um metal-rimmed ring at 1.55 um). Both need s > 0 over the grid, '
               'PML included.',
          tip='How the bend is computed: exactly, or by the older approximate conformal map.\n\n'
              'Keep the exact model unless comparing with the approximation.'),
    Param('discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATION_ORDERS),
          choice_labels=(('quadratic', 'Quadratic'), ('linear', 'Linear')),
          help='Element order on a triangles mesh (MeshSpec kind triangles; tensor meshes ignore it and use the Yee '
               'cross-section). quadratic (default, the industry standard): second-order Nedelec edge elements for '
               'E_x, E_y and P2 Lagrange for E_z (Lee-Sun-Cendes mixed form, free of spurious modes), elements with a '
               'curved edge mapped isoparametrically, so circles and ellipses are followed to the mesh; linear: '
               'Whitney edge elements and P1 on straight elements, cheaper, error ~h^2 instead of ~h^4 in beta^2. '
               'About 7 (quadratic) or 2 (linear) unknowns per element, solved by the shift-invert of the eigen '
               'solver: host SuperLU, or with its factorization cudss a cuDSS LDL^T on the GPU (the pencil is complex '
               'symmetric), '
               f'with {HOST_DIRECT_NAMES} a MUMPS / PARDISO LDL^T on the host, '
               'the _mixed ones their complex64 form: one float64 refinement step per solve, ARPACK to 1e-6, then '
               'beta^2 from the float64 Rayleigh quotient of each eigenvector (error ~ the square of the vector '
               'error). '
               'Materials are one value per element (an element takes the material at its centroid; the mesh '
               'follows every drawn outline). On triangles: pec / antisymmetric faces fix tangential E on the window '
               'edge, pmc / symmetric faces are natural (mirror planes are the window edge, no ghost cell); a cpml '
               'face gets a ring of mesh pml_layers structured element layers, pml_cells x the maximum element size '
               'thick, with the stretch of the pml_* settings and a pec outer side (pml_kappa_max 1 suits radiating '
               'leaky modes, the default 15 evanescent tails); pec_domain removes its elements and holds E = 0 on '
               'their outline. Not available on triangles (validation errors): periodic faces, bend_radius, '
               'shg_overlap, bend_overlap, a line_block shifted solver and the thermal_noise / optomechanics '
               'studies; subpixel smoothing and conformal boundaries do not apply.',
          tip='Order of the finite elements when the mesh is made of triangles: quadratic is far more accurate per '
              'element, linear is cheaper.\n\nUnused on a rectangular-cell mesh.'),
) + bloch_params(
    'Bloch wave vector along {axis} in rad/m, used only when bc_{axis}_lo and bc_{axis}_hi are periodic: the modes are '
    'Bloch waves of the periodic cross-section, F({axis} + L) = F({axis}) exp(-j k{axis} L) per period L (the meshed '
    'period; fields ~ exp(j (omega t - beta z))), e.g. one supercell of a photonic-crystal fibre or of a waveguide '
    'array at a transverse phase step k{axis} L per cell; k{axis} and k{axis} + 2 pi / L are the same state, 0 = plain '
    'periodic. The phase sits on the wrap entries of the operator (same unknowns and cost; the operator is no longer '
    'symmetric, which the shift-invert solve handles); the displayed centre fields take it across the wrap. Not with '
    'shg_overlap (its 2 omega modes would need 2 k{axis}) and not in thermal_noise or optomechanics studies (their '
    'overlaps assume k = 0): validation errors. Typical: pi / L at the edge of the first zone.',
    'Phase step of the mode from one period of the cross-section to the next along {axis}, in radians per metre, '
    'for periodic edges.\n\n0 means no phase step. Use pi divided by the period for the edge of the first zone.')

# ---- material vocabulary (definition_common.em_material_params; em_modes reads eps_xx, eps_yy and eps_zz) ----
_COMPLEX_EPS = ('The solve uses the complex permittivity at the solve wavelength: this value - j sigma_e / (omega '
                'eps0) plus the susceptibility chi(omega) of every Drude / Lorentz pole (exp(+j omega t): loss has '
                'Im < 0, giving Im n_eff < 0).')
_BACKGROUND = ('For a dispersive material it is the background eps_inf (an explicit permittivity wins over the Drude '
               'eps_inf). A state-dependent value (e.g. of T) is rewritten once after the background solve, before the '
               'mode solve.')
_FACE_VALUE = ('the width-weighted mean of the two cells sharing that face (solver_defaults.subpixel_smoothing '
               'averages interface faces anisotropically)')
_MATERIAL_PROPERTIES = em_material_params('modes', {
    'eps_r': ('Isotropic relative permittivity (>= 1): sets eps_xx, eps_yy and eps_zz unless the material sets those '
              'components (a component wins over eps_r). n^2 of a lossless dielectric: silica ~2.085, silicon ~12.1 at '
              '1.55 um; may depend on x, y and physics fields. E_z sees the cell value, E_x on the +x and E_y on the +y '
              f'cell faces {_FACE_VALUE}. {_COMPLEX_EPS} {_BACKGROUND}'),
    'eps_zz': ('z component (along the guide) of the relative permittivity (>= 1), the one E_z sees at the cell '
               'centres; overrides eps_r for E_z. Uniaxial crystal with its optic axis along the guide: eps_zz = '
               'n_e^2, eps_r = n_o^2.'),
    'eps_xx': ('x component of the relative permittivity (>= 1), the one E_x sees on the +x cell faces '
               f'({_FACE_VALUE}); overrides eps_r for E_x. E.g. x-cut lithium niobate at 1.55 um (optic axis along '
               'x): eps_xx 4.571 = n_e^2 with eps_r 4.889 = n_o^2 for the other components.'),
    'eps_yy': ('y component of the relative permittivity (>= 1), the one E_y sees on the +y cell faces '
               f'({_FACE_VALUE}); overrides eps_r for E_y. For a uniaxial crystal with its optic axis along y set '
               'eps_yy = n_e^2.'),
    'sigma_e': ('Ohmic loss in S/m on all three E components, default 0: the permittivity at the solve wavelength gains '
                '-j sigma_e / (omega eps0), so the guide gets Im n_eff < 0 and loss_db_per_cm > 0. A weak absorber with '
                'index n and extinction coefficient kappa at angular frequency omega has sigma_e = 2 n kappa omega '
                'eps0; model metals with a Drude pole, not their DC conductivity.'),
    'mu_r': ('Real relative permeability (>= 1) of H_x, H_y and H_z, each taking the value of the cell it belongs to '
             '(H_x on the +y face, H_y on the +x face, H_z at the +x+y corner of that cell); 1 for practically every '
             'optical material. The solve uses mu_r - j sigma_m / (omega mu0).'),
    'sigma_m': ('Magnetic loss on H_x, H_y, H_z in Ohm/m, default 0: mu_r gains -j sigma_m / (omega mu0) (artificial '
                'absorbers only).'),
}, tips={'eps_zz': ('Permittivity for the electric field along z, the direction the light travels.\n\n'
                    "When set it replaces 'Relative permittivity' for that direction.")})

# ---- features ----
_MODE_SCALARS = (
    'n_eff (Re n_eff = Re beta / k0), n_eff_imag (Im n_eff < 0 = loss in the exp(+j omega t), exp(-j beta z) '
    'convention), loss_db_per_cm = -20 log10(e) Im(beta) / 100 (power loss in dB/cm, > 0 for a lossy or leaky mode), '
    'n_group (C0 d(Re beta) / d(omega), NaN unless group_index or when the mode could not be followed to the shifted '
    'frequencies, with a run-log warning), te_fraction = integral |E_x|^2 dA / integral (|E_x|^2 + |E_y|^2) dA (the '
    'industry-standard TE polarisation fraction: 1 = E along x, e.g. the quasi-TE mode of a strip on a substrate '
    'parallel to the xz plane, 0 = E along y), effective_area = (integral |E|^2 dA)^2 / integral |E|^4 dA in m^2 (all '
    'three components at the cell centres; the nonlinear-optics definition; like the 1 W normalisation it covers the '
    'whole mode, mirror images included) and pml_fraction (share of the '
    "mode's |H|^2 in PML cells: kept modes have < 0.5, and above ~1e-2 the PML shapes the mode: widen the cladding)")
_MODE_TABLE_DESCRIPTION = (
    "Mode table of the last solve (a frequency_domain study: its last wavelength). Result 'table', one row per mode "
    f'found, by descending Re(beta^2) (see n_modes): mode_number (1-based rank), {_MODE_SCALARS}, and selected (True for the '
    "mode of the mode_number setting); extras wavelength (m) and selected_mode. For n_eff in a parameter sweep the scalar output "
    "<physics>_n_eff_<k> is simplest; any other column needs a derived value with source '<physics>.<this feature>', "
    "the column name as expression, operation at and at = mode_number - 1 (the row index).")
_MODE_DISPERSION_DESCRIPTION = (
    "Scalars of one mode versus vacuum wavelength. Result 'xy' (x = wavelength in m, ascending; one point per solved "
    'wavelength, so a mode_analysis study gives one; a wavelength solved again keeps its latest solve), series '
    f'{_MODE_SCALARS}, and overlap (|overlap| of the followed mode with the one followed at the previous solve, 1 at '
    'the first; a value well below 1 means the mode changed character or crossed another); extras frequency (Hz), '
    'wavelength (m) and mode_number. The first solve that finds at least mode_number modes picks the mode of that '
    'rank; every later solve follows the mode of largest overlap with the previous one, so it keeps its identity where '
    'modes swap order in n_eff (solve order: the frequency_domain wavelengths list, else wavelength_start to '
    'wavelength_stop). Rows are NaN while the mode is not found.')
_MODE_DISPERSION_PARAMS = (
    Param('mode_number', 'int', 1, label='Mode number', min=1,
          help='1-based rank (descending Re(beta^2), as mode_table) of the followed mode at the first solve that finds that many '
               'modes; later solves follow it by overlap.',
          tip='Which mode to follow across the wavelengths: 1 is the most strongly guided one at the first '
              'wavelength.\n\nThe same mode is then followed even where two modes swap order.'),
)


_SHG_TEXT = (
    "Coupled-mode second-harmonic generation of a 3D waveguide from its cross-section modes. Result 'xy' versus pump "
    'vacuum wavelength (x, m; one point per solve: mode_analysis one, frequency_domain one per wavelength): after '
    'every em_modes solve at omega the feature solves the modes at 2 omega itself (same materials, walls, bend and '
    'eigen solver; a second eigen-solve of the same size, so about twice the time) and pairs mode pump_mode (omega) '
    'with mode harmonic_mode (2 omega), both by rank as in mode_table (no tracking across wavelengths). Series: '
    'n_eff_pump, n_eff_harmonic, delta_beta = Re beta(2 omega) - 2 Re beta(omega) [1/m], coherence_length = pi / '
    '|delta_beta| and poling_period = 2 pi / |delta_beta| [m] (NaN at delta_beta = 0), overlap = |kappa| [V^2/W^1.5], '
    'kappa = integral conj(e_2w) . (d : e_w e_w) dA over the selected domains (1 W modes, fields at the cell centres '
    'by face means), efficiency = (omega eps0 / 2)^2 |kappa|^2 in %/(W cm^2) (P_2w = efficiency P_w^2 L^2 when phase '
    'matched; times sinc^2(delta_beta L / 2) otherwise; a plane-wave mode of area A gives 8 pi^2 d^2 / (eps0 c n_w^2 '
    'n_2w lambda^2 A)) and efficiency_qpm = (2 / pi)^2 efficiency (first-order quasi phase matching, 50 % duty). '
    'Convention: exp(j (omega t - beta z)), SHG P(2 omega)_i = eps0 sum_jk d_ijk E_j E_k with d = chi2 / 2 given as '
    'Voigt d_il in MODEL axes (x, y = cross-section, z = along the guide; l = 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy): '
    'rotate crystal values yourself, e.g. x-cut lithium niobate with its c axis along model x and TE modes (E along '
    'x): d11 = d33(LN) ~ 2.5e-11 m/V, plus d12 = d13 = d31(LN) ~ -4.6e-12 m/V for the minor field components. The '
    'mesh must resolve the 2 omega '
    'modes (>= 10 cells per lambda / (2 n)); no mirror faces (the overlap integrand has no single parity) and Bloch '
    'k = 0 (validation errors); loss enters only through Re beta. With a bend (bend_radius) kappa is the physical overlap per length of '
    'arc at bend_reference_x (where delta_beta and L are measured too): bend_model exact integrates s dA with E_phi = '
    'E_z / s, conformal s^2 dA with the fields taken as physical (exact for E along y, approximate otherwise like the '
    'map), s = 1 + (x - x_ref) / R. ring_crystal local: d stays in the local frame (x radial, y vertical, z along the '
    'guide), i.e. it turns with the guide: right for terms unchanged by rotations about y (e.g. d33, d31, d15 of a '
    'z-cut crystal, c axis along y). ring_crystal fixed (bend only; an x- or y-cut ring): d is given at the '
    'reference cross-section (azimuth phi = 0) and the crystal stays fixed while the guide turns about y, so the '
    'local d(phi) = Q(phi) Q Q : d with Q rows (cos phi, 0, sin phi), (0, 1, 0), (-sin phi, 0, cos phi) and phi = z / R '
    '(signed R as bend_radius, z the arc at x_ref). kappa(phi) = sum over m = -3..3 of kappa_m exp(j m phi) (exact '
    'from 8 azimuths), and order m is phase matched when delta_beta R + m = 0. Extra series: azimuthal_mismatch = '
    'delta_beta R [1] (the azimuthal-order mismatch nu_2w - 2 nu_w of a ring), ring_order m* = round(-delta_beta R) '
    '(NaN when |m*| > 3: no ring harmonic supplies it), ring_mismatch = delta_beta + m* / R [1/m], ring_overlap = '
    '|kappa_m*| and ring_efficiency = (omega eps0 / 2)^2 |kappa_m*|^2 in %/(W cm^2) (0 when |m*| > 3; P_2w = '
    'ring_efficiency P_w^2 L^2 sinc^2(ring_mismatch L / 2) over an arc L), and ring_overlap_minus3 .. ring_overlap_plus3 '
    '= |kappa_m| [V^2/W^1.5]; the plain overlap / efficiency series stay those of the reference cross-section (phi = '
    '0). E.g. x-cut lithium niobate, TE modes: d11 = d33(LN) gives kappa(phi) ~ cos^3 phi, i.e. orders +-1 (3/8) and '
    '+-3 (1/8) of kappa(0).')


def _shg_params():
    """Mode choice, the 2 omega solve and the 18 model-axis d_il."""
    return (
        Param('pump_mode', 'int', 1, label='Pump mode', min=1,
              help='1-based rank (as mode_table) of the pump mode of each em_modes solve; must not exceed n_modes.',
              tip='Which computed mode carries the pump light: 1 is the most strongly guided one.'),
        Param('harmonic_mode', 'int', 1, label='Second-harmonic mode', min=1,
              help='1-based rank (descending Re(beta^2)) of the second-harmonic mode among the n_harmonic_modes modes '
                   'of the 2 omega solve; must not exceed n_harmonic_modes.',
              tip='Which mode at the doubled frequency receives the generated light: 1 is the most strongly guided '
                  'one.'),
        Param('n_harmonic_modes', 'int', 4, label='Modes at the second harmonic', min=1, max=100,
              help='Modes kept by the 2 omega solve (as n_modes of the physics).',
              tip='How many modes to compute at the doubled frequency.'),
        Param('harmonic_n_eff_guess', 'float', None, label='Second-harmonic index guess', unit='1', min=0.0,
              min_exclusive=True, optional=True,
              help='Effective index around which the 2 omega modes are sought (as n_eff_guess); empty = the largest '
                   'index of the cross-section at 2 omega.',
              tip='Effective index near which the second-harmonic modes are searched; empty searches below the highest '
                  'refractive index.'),
        Param('ring_crystal', 'choice', 'local', label='Crystal around the ring', choices=RING_CRYSTALS,
              choice_labels=(('local', 'Turns with the guide'), ('fixed', 'Fixed in the chip (x- or y-cut ring)')),
              help="Bent guides only (bend_radius != 0 for 'fixed', validation error otherwise). 'local': the d_il hold "
                   "in the guide's local frame everywhere. 'fixed': the d_il hold at the reference cross-section and "
                   'the crystal stays fixed while the guide turns about y, so the coupling varies around the ring and '
                   'phase matching picks an azimuthal order m = -3..3 (series azimuthal_mismatch, ring_order, '
                   'ring_mismatch, ring_overlap, ring_efficiency, ring_overlap_minus3 .. _plus3; feature help).',
              tip='How the crystal axes behave around a ring.\n\n- Turns with the guide: the crystal looks the same '
                  'from every point of the ring, for example a crystal axis pointing up out of the chip.\n- Fixed in '
                  'the chip: the crystal axis lies in the chip plane, so the guide sees it from a changing angle as it '
                  'goes around; the results then also list the ring phase matching.'),
        *(Param(name, 'float', 0.0, label=name, unit='m/V',
                help=f'Second-order coefficient {name} in m/V in model axes (Voigt: polarisation component {name[1]} '
                     f'= x, y, z for 1, 2, 3; field pair l = {name[2]}); d = chi2 / 2, 1 pm/V = 1e-12 m/V.',
                tip=f'Nonlinear coefficient {name} in metres per volt along the model axes: 1 pm/V is 1e-12.')
          for name in VOIGT_NAMES),
    )


def _apply_shg_overlap(ctx, solver, feature_spec, values, selection):
    return solver.add_shg_overlap(voigt_tensor(values), ctx.domain_mask(selection.entities), values['pump_mode'],
                                  values['harmonic_mode'], values['n_harmonic_modes'], values['harmonic_n_eff_guess'],
                                  values['ring_crystal'])


def _validate_shg_overlap(vctx, physics_spec, feature_spec, values):
    """Errors: mirror faces, pump_mode above n_modes, harmonic_mode above n_harmonic_modes; warning: all d_il 0."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    settings = vctx.values(EM_MODES.settings, raw_settings(physics_spec))
    if settings is not None:
        mirrors = [face for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')
                   if settings.get(f'bc_{face}') in ('symmetric', 'antisymmetric')]
        if mirrors:
            issues.append(Issue('error', path, f"shg_overlap needs the whole cross-section, but {', '.join(mirrors)} "
                                               'is a mirror (the overlap integrand has no single parity); use pec, '
                                               'pmc, cpml or periodic faces'))
        if values['pump_mode'] > settings['n_modes']:
            issues.append(Issue('error', path, f"pump_mode {values['pump_mode']} exceeds n_modes {settings['n_modes']}"))
        if values['ring_crystal'] == 'fixed' and settings['bend_radius'] == 0.0:
            issues.append(Issue('error', path, "ring_crystal 'fixed' needs a bent guide (bend_radius != 0): a straight "
                                               'guide sees the crystal from one direction only; use local'))
    if values['harmonic_mode'] > values['n_harmonic_modes']:
        issues.append(Issue('error', path, f"harmonic_mode {values['harmonic_mode']} exceeds n_harmonic_modes "
                                           f"{values['n_harmonic_modes']}"))
    if not any(values.get(name) for name in VOIGT_NAMES):
        issues.append(Issue('warning', path, 'every d_il is 0: the efficiency is 0 (delta_beta and the indices are '
                                             'still reported)'))
    return issues


def _shg_memory_estimate(values, n_selected_cells, mesh):
    """Host: the 2 omega modes (6 complex128 fields each), two centre-field sets, the area and mask; the 2 omega factor
    is built after the pump solve's is freed, so the physics estimate covers it."""
    n_cells = int(mesh.n_cells)
    n_modes = int(values.get('n_harmonic_modes', 4))
    return MemoryEstimate.from_items((), build_extra=n_cells, host_items=[   # build: the device selection mask
        (f'shg_overlap 2 omega modes ({n_modes} x 6 components)', 6 * n_modes * COMPLEX128_BYTES * n_cells),
        ('shg_overlap centre fields, area and mask', (12 * COMPLEX128_BYTES + FLOAT64_BYTES + 1) * n_cells)])


_BEND_OVERLAP_TEXT = (
    "Straight-to-bend junction of a bent guide (bend_radius != 0, validation error otherwise). Result 'table' of the "
    'last solve (a frequency_domain study: its last wavelength): after every em_modes solve the feature solves the '
    'same cross-section straight (same materials, walls and eigen solver, n_straight_modes modes near '
    'straight_n_eff_guess; a second eigen-solve of the same size) and pairs each bent mode with the straight mode of '
    'largest power coupling overlap = |c_bs c_sb| / |c_bb c_ss|, c_ab = 1/2 integral (E_a x H_b*).z dA over the '
    'junction plane (the transverse fields of the bent mode are the physical E_r, E_y, H_r, H_y in both bend models). '
    'Columns: mode_number (bent rank), n_eff, n_eff_imag (bent, referred to bend_reference_x), straight_mode (rank of '
    'the best-matching straight mode), n_eff_straight, overlap (1 = a perfect match), junction_loss_db = -10 '
    'log10(overlap) (the power lost into other modes and radiation at one straight-bend junction, ~1 / R^2 for large '
    'R: a ring coupler or an S-bend has two junctions per transition) and overlap_same_rank (with the straight mode of '
    'the same rank, NaN when fewer were found); extras wavelength (m) and n_straight_found. Both guides share the '
    'cross-section, so the junction is centred: the outward shift of the bent mode, which a lateral offset of the '
    'straight guide compensates, is part of the loss.')
_BEND_OVERLAP_PARAMS = (
    Param('n_straight_modes', 'int', 4, label='Straight modes', min=1, max=100,
          help='Modes kept by the straight solve (as n_modes of the physics).',
          tip='How many modes of the straight guide to compute for the comparison.'),
    Param('straight_n_eff_guess', 'float', None, label='Straight index guess', unit='1', min=0.0,
          min_exclusive=True, optional=True,
          help='Effective index around which the straight modes are sought (as n_eff_guess); empty = the largest index '
               'of the cross-section.',
          tip='Effective index near which the straight-guide modes are searched; empty searches below the highest '
              'refractive index.'),
)


def _apply_bend_overlap(ctx, solver, feature_spec, values, selection):
    return solver.add_bend_overlap(values['n_straight_modes'], values['straight_n_eff_guess'])


def _validate_bend_overlap(vctx, physics_spec, feature_spec, values):
    """Error for a straight guide (bend_radius 0)."""
    settings = vctx.values(EM_MODES.settings, raw_settings(physics_spec))
    if settings is not None and settings['bend_radius'] == 0.0:
        return [Issue('error', feature_path(physics_spec, feature_spec),
                      'bend_overlap needs a bent guide: set bend_radius != 0 (a straight guide matches itself)')]
    return []


def _bend_overlap_memory_estimate(values, n_selected_cells, mesh):
    """Host: the straight modes (6 complex128 fields each, kept until the next solve); the straight factor is built
    after the bent solve's is freed, so the physics estimate covers it."""
    n_modes = int(values.get('n_straight_modes', 4))
    return MemoryEstimate.from_items((), host_items=[
        (f'bend_overlap straight modes ({n_modes} x 6 components)', 6 * n_modes * COMPLEX128_BYTES * int(mesh.n_cells))])


def _apply_mode_table(ctx, solver, feature_spec, values, selection):
    return solver.add_mode_table()


def _apply_mode_dispersion(ctx, solver, feature_spec, values, selection):
    return solver.add_mode_dispersion(values['mode_number'])


_FEATURES = (
    FeatureType('mode_table', 'Mode table', 'monitor', None, result_kind='table',
                description=_MODE_TABLE_DESCRIPTION,
                summary='Lists every computed mode with its effective index, loss, polarisation and size. Use it for '
                        'derived values such as the effective index of the first mode in a parameter sweep.',
                apply=_apply_mode_table, result=object_result(mode_table_result), grids=BOTH_GRIDS),
    FeatureType('mode_dispersion', 'Mode dispersion', 'monitor', None, params=_MODE_DISPERSION_PARAMS,
                result_kind='xy', description=_MODE_DISPERSION_DESCRIPTION,
                summary="Follows one mode across the wavelengths of a 'Frequency domain' study: effective index, loss "
                        'and polarisation versus wavelength.',
                apply=_apply_mode_dispersion, result=object_result(mode_dispersion_result), grids=BOTH_GRIDS),
    FeatureType('shg_overlap', 'Second-harmonic overlap', 'monitor', 'domain', params=_shg_params(),
                result_kind='xy', description=_SHG_TEXT,
                summary='Second-harmonic generation in a long waveguide from its modes: the conversion efficiency in '
                        'percent per watt and square centimetre, the phase mismatch and the poling period for quasi '
                        'phase matching. The selected domains are the nonlinear crystal.',
                apply=_apply_shg_overlap, result=object_result(shg_overlap_result), validate=_validate_shg_overlap,
                memory_estimate=_shg_memory_estimate),
    FeatureType('bend_overlap', 'Straight-to-bend overlap', 'monitor', None, params=_BEND_OVERLAP_PARAMS,
                result_kind='table', description=_BEND_OVERLAP_TEXT,
                summary='Loss at the junction between a straight and a bent waveguide: how well each bent mode '
                        'matches the modes of the same guide without the bend, in percent and in decibels.',
                apply=_apply_bend_overlap, result=object_result(bend_overlap_result),
                validate=_validate_bend_overlap, memory_estimate=_bend_overlap_memory_estimate),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor: in every cell whose centre lies in the selected domains E_z = '
                            '0 at the centre and E_x / E_y = 0 on the faces between two such cells (eps^-1 = 0 there), '
                            'staircased to the mesh. With solver_defaults.conformal_boundaries (Dey-Mittra on the '
                            'projected Yee cell) the drawn outline is used instead: E_z = 0 at the bisected wall '
                            'distance along each centre link into the metal, and H_z takes the free lengths of its '
                            'E_x / E_y edges over the free area of its dual cell (free fractions >= 0.02); links and '
                            'dual cells crossing a pec / pmc / mirror wall plane stay staircased. On a triangles mesh '
                            'the elements whose centroid lies in the selected domains leave the unknowns and E_t '
                            'tangential and E_z are 0 on their outline, which the mesh follows (curved outlines '
                            'isoparametric with discretization quadratic).',
                summary='Turns the selected domains into a perfect metal, following the mesh cells. Use it for '
                        'idealised metal walls or hollow metal waveguides.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0), grids=BOTH_GRIDS),
)

# ---- type ----
_DESCRIPTION = (
    'Full-vector finite-difference eigenmode solver of a waveguide cross-section: the xy plane is the cross-section, '
    'the guide runs along z (out of plane), x to the right and y up; fields vary as exp(j (omega t - beta z)) (time '
    'convention exp(+j omega t)) with n_eff = beta / k0, k0 = omega / C0, so a lossy or leaky mode has Im n_eff < 0. '
    'Yee cross-section (a 3D Yee cell projected, transverse-H formulation after Fallahkhair, Li and Murphy, JLT 2008): '
    'E_z at the cell centres, E_x and H_y on the +x cell faces, E_y and H_x on the +y faces, H_z at the +x+y corner; '
    'eps_zz (array eps_r) is the cell value, eps_xx on a +x face and eps_yy on a +y face the width-weighted mean of the '
    'two cells, mu the cell value (solver_defaults.subpixel_smoothing replaces the background eps at interface '
    'locations: E_x / E_y faces 1 / (n^2 <1/eps> + (1 - n^2) / <eps>) with n the in-plane interface normal component, '
    'E_z cells the area mean <eps>); all complex at the solve wavelength (sigma_e, Drude / '
    'Lorentz chi, sigma_m). The generalised eigenproblem A h = beta^2 B h for h = eta0 (H_x, H_y) is solved on the host '
    'by shift-invert Arnoldi (scipy eigs on a SuperLU factor: needs scipy; 2 unknowns per cell, PML included, so '
    'validation warns near and refuses beyond the factor size limit) around (k0 n_eff_guess)^2. Solver roles: eigen '
    '= arpack_shift_invert (tolerance, subspace_size, factorization: superlu, cudss = a cuDSS LU on the GPU without '
    f'the host size limit, guarded by the free device memory; {HOST_DIRECT_NAMES} = a MUMPS / PARDISO factor on '
    'the host under the SuperLU size limit; cudss_mixed and the other _mixed factorizations only on a triangles '
    'mesh, since this operator is not complex symmetric); shifted: Automatic (sparse_direct) = that SuperLU factor, a '
    'line_block node factors the shift-invert operators on its backend instead (unknowns ordered by grid rows or '
    'columns, a periodic axis inside the blocks, two periodic axes an error; mixed or double precision, refined '
    'against the complex128 operator), batched: the omega and, with group_index, omega (1 +- 1e-3) operators of a '
    'solve and, in a frequency_domain study, those of the next wavelengths, up to 8 operators per batch, as many as '
    '60 % of the free memory holds; a rejected solve redoes that eigen-solve on SuperLU (run-log warning); '
    'shg_overlap / bend_overlap solves stay SuperLU. Modes with more than '
    'half of their |H|^2 in PML cells are dropped, the rest ranked by descending Re(beta^2) (ties by Im; = descending '
    'Re n_eff for low-loss guided modes, evanescent ones last), up to n_modes kept. Each '
    'mode carries 1 W, P = 1/2 Re integral (E x H*).z dA (E_x / H_y and E_y / H_x share their points), phased so its '
    'dominant transverse E component is real and positive at its peak (the lowest-index cell of mirror-equal peaks). '
    'Degenerate pairs (the two HE11 polarisations of a round fibre, TE11 / TM11 of a metal pipe) come back as '
    'arbitrary, not necessarily orthogonal, combinations. Outputs (the displayed mode: the mode_number setting after a study, at '
    'the cell centres with the face / corner components averaged there): ex, ey, ez, hx, hy, hz = the real parts in '
    'V/m and A/m, e_abs, e_intensity (|E|^2, the primary field), h_abs and sz = 1/2 Re(E_x H_y* - E_y H_x*) in W/m^2 '
    '(time-averaged power flow along z; its integral over the window equals the 1 W only to discretisation order and '
    'misses any PML share); state fields e_intensity and sz of the displayed mode; scalar outputs n_eff_1 .. '
    'n_eff_<n_modes> (derived values: <physics>_n_eff_<k>). No absorbed power: em_modes is not '
    'an em_heat_joule source. Features: mode_table, mode_dispersion, shg_overlap (coupled-mode SHG efficiency of a '
    'mode pair at omega / 2 omega), bend_overlap (straight-to-bend junction loss of a bent guide), pec_domain. '
    'Periodic faces may carry a Bloch phase (bloch_kx / bloch_ky: Bloch modes of a periodic cross-section). '
    'Studies: mode_analysis (one solve at '
    'the wavelength setting after any background heat / current solve, one snapshot per mode) and frequency_domain '
    '(one solve per wavelength). heat_em_thermo_optic rewrites eps_zz, eps_xx and eps_yy (slots eps_r_update, '
    'eps_xx_update, eps_yy_update). The xy plane here is a cross-section, not a propagation plane, so em_modes cannot '
    "share a model with em_fdtd_* or em_fdfd_* (validation error). On a triangles mesh (MeshSpec kind triangles) the "
    'same physics is a finite-element solve: Nedelec edge elements for E_x, E_y and Lagrange elements for E_z of the '
    'discretization setting (Lee-Sun-Cendes mixed form for beta^2, complex symmetric with PML / loss), materials one '
    'value per element, a structured PML ring; the same outputs on a display raster of the window (state fields one '
    'value per element) and the same table / dispersion results; details and refusals under discretization. The '
    "instance is usually named 'modes'.")


def _create(ctx, physics_spec, settings_values):
    """ModePhysics (tensor grid) or FEMModePhysics (triangle mesh: its walls from the settings) at the first solved
    wavelength with its dispersive poles and their background permittivity written into eps_r (= eps_zz), eps_xx,
    eps_yy; never finalizes."""
    if getattr(ctx.domain, 'grid_kind', 'tensor') == 'triangles':
        solver = FEMModePhysics(ctx.domain, fdfd_domain_requirements(settings_values).boundary_config,
                                wavelength=first_solved_wavelength(ctx, settings_values),
                                n_modes=settings_values['n_modes'], mode_index=settings_values['mode_number'],
                                n_eff_guess=settings_values['n_eff_guess'], group_index=settings_values['group_index'],
                                eigen_solver=ctx.solvers(physics_spec.name)['eigen'],
                                order=DISCRETIZATION_ORDERS[settings_values['discretization']])
    else:
        solver = ModePhysics(ctx.domain, boundary_config=ctx.domain.boundary_config,
                             wavelength=first_solved_wavelength(ctx, settings_values), n_modes=settings_values['n_modes'],
                             mode_index=settings_values['mode_number'], n_eff_guess=settings_values['n_eff_guess'],
                             bend_radius=settings_values['bend_radius'],
                             bend_reference_x=settings_values['bend_reference_x'],
                             group_index=settings_values['group_index'],
                             eigen_solver=ctx.solvers(physics_spec.name)['eigen'],
                             bend_model=settings_values['bend_model'],
                             shifted_solver=ctx.solvers(physics_spec.name).get('shifted'))
    write_dispersive_materials(ctx, physics_spec, solver, MODE_PERMITTIVITY_ARRAYS)
    return solver


def _in_plane_em_issues(vctx, path):
    """Error when another EM type shares the model: its xy plane is a propagation plane, not a cross-section; a
    thermal_noise study accepts em_fdfd_ez / _hz as the cross-section illumination (k_z = 0)."""
    allowed = _CROSS_SECTION_ILLUMINATION_TYPES if vctx.study_type == _THERMAL_NOISE_STUDY else ()
    others = other_em_physics_labels(vctx, _TYPE_NAME, allowed)
    if not others:
        return []
    return [Issue('error', path, f"em_modes treats the xy plane as a waveguide cross-section (propagation along z), "
                                 f"while {', '.join(others)} propagate in the xy plane; they cannot share one model: "
                                 f"compute the modes in a separate model (the in-plane mode ports solve their own "
                                 f"slab modes; a thermal_noise study accepts em_fdfd_ez / em_fdfd_hz as an "
                                 f"illumination of the cross-section)")]


def _bend_issues(vctx, settings_values, path):
    """Bend: no periodic x axis, scale factor > 0 over the meshed grid, a warning without cpml on the outer side."""
    radius = settings_values['bend_radius']
    if radius == 0.0:
        return []
    if settings_values['bc_x_lo'] == 'periodic':
        return [Issue('error', path, 'bend_radius needs a non-periodic x axis (the bend map is not periodic in x); '
                                     'set bc_x_lo / bc_x_hi to cpml or a wall')]
    issues = []
    size = vctx.domain_size
    if mesh_kind(vctx) == 'tensor' and size is not None:   # triangles: refused by _triangle_issues
        reference_x = settings_values['bend_reference_x']
        reference_x = 0.5 * size[0] if reference_x is None else reference_x
        message = bend_factor_message(vctx.mesh.x_edges, radius, reference_x)
        if message is not None:
            issues.append(Issue('error', path, message))
    outer_face = 'x_hi' if radius > 0.0 else 'x_lo'
    if settings_values[f'bc_{outer_face}'] != 'cpml':
        issues.append(Issue('warning', path, f"bend_radius {radius:.4g} m radiates toward {outer_face}, which is "
                                             f"{settings_values[f'bc_{outer_face}']}, not cpml: the radiation is "
                                             f"reflected back, so Im n_eff / loss_db_per_cm miss the bend loss"))
    return issues


def _bloch_issues(vctx, physics_spec, settings_values, path):
    """Bloch k on a non-periodic axis (warning, ignored); with k != 0 errors for shg_overlap features and the
    thermal_noise / optomechanics studies (both refuse it at run time too)."""
    issues = fdfd_bloch_issues(vctx, physics_spec, settings_values)
    if not bloch_active_axes(settings_values):
        return issues
    for feature in physics_spec.features:
        if feature.enabled and feature.type == 'shg_overlap':
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                'shg_overlap needs Bloch k = 0: its 2 omega modes would need the Bloch wave vector '
                                '2 k; set bloch_kx / bloch_ky to 0'))
    if vctx.study_type in (_THERMAL_NOISE_STUDY, OPTOMECHANICS_STUDY):
        issues.append(Issue('error', path, f'a {vctx.study_type} study needs the em_modes modes at Bloch k = 0 (its '
                                           'overlaps assume no phase step across the period); set bloch_kx / bloch_ky '
                                           'to 0'))
    return issues


def _solver_issues(vctx, physics_spec, settings_values, path):
    """scipy missing: error; mode_number above n_modes: error; a mixed eigen factorization (MIXED_FACTORIZATIONS) on a
    tensor mesh: error; a line_block shifted node with two periodic axes: error; else (also under line_block with a
    shg_overlap / bend_overlap feature) the host SuperLU size limit (size_issue_for; none for a cudss factorization,
    MUMPS / PARDISO keep it)."""
    issues = []
    factorization = eigen_factorization(vctx.solvers(physics_spec))
    if factorization in MIXED_FACTORIZATIONS and mesh_kind(vctx) == 'tensor':
        issues.append(Issue('error', path, f'the eigen factorization {factorization} needs the complex symmetric '
                                           'pencil of a triangles mesh (its loose complex64 solves are polished by '
                                           'Rayleigh quotients); the Yee cross-section operator is not complex '
                                           'symmetric: use '
                                           f"'{factorization.removesuffix('_mixed')}' (double) or a triangles mesh"))
    issues.extend(scipy_issues('em_modes', 'host sparse eigensolver', path))
    if settings_values['mode_number'] > settings_values['n_modes']:
        issues.append(Issue('error', path, f"mode_number {settings_values['mode_number']} exceeds n_modes "
                                           f"{settings_values['n_modes']}; raise n_modes or lower mode_number"))
    shifted = (vctx.solvers(physics_spec) or {}).get('shifted')
    prefix = ''
    if shifted is not None and shifted.solver_type.name == 'line_block':   # its own size rule: the memory estimate
        if settings_values['bc_x_lo'] == 'periodic' and settings_values['bc_y_lo'] == 'periodic':
            issues.append(Issue('error', path, LINE_BLOCK_BOTH_PERIODIC_TEXT))
        detached = sorted({feature.type for feature in physics_spec.features
                           if feature.enabled and feature.type in _DETACHED_FEATURES})
        if not detached:
            return issues
        prefix = f"the {' / '.join(detached)} solves use the eigen solver's own factor, not the line block: "
    if mesh_kind(vctx) == 'tensor' and factorization not in CUDSS_FACTORIZATIONS:   # triangles: _triangle_issues
        issue = size_issue_for(int(vctx.mesh.nx), int(vctx.mesh.ny))
        if issue is not None:
            issues.append(Issue(issue[0], path, prefix + issue[1]))
    return issues


def _triangle_issues(vctx, physics_spec, settings_values, path):
    """On a triangles mesh: errors for periodic faces, a bend and the thermal_noise / optomechanics studies (their
    overlaps read the tensor Yee points); the FEM size rule of host SuperLU on the estimated unknowns."""
    if mesh_kind(vctx) != 'triangles':
        return []
    issues = []
    if 'periodic' in (settings_values['bc_x_lo'], settings_values['bc_y_lo']):
        issues.append(Issue('error', path, f'em_modes on a triangles mesh: {PERIODIC_TEXT}'))
    if settings_values['bend_radius'] != 0.0:
        issues.append(Issue('error', path, 'bend_radius is not available on a triangles mesh yet; use a tensor mesh '
                                           'for bent guides'))
    if vctx.study_type in (_THERMAL_NOISE_STUDY, OPTOMECHANICS_STUDY):
        issues.append(Issue('error', path, f'a {vctx.study_type} study needs em_modes on a tensor mesh (its overlaps '
                                           'read the Yee cross-section fields)'))
    return issues + fem_size_issues(vctx, physics_spec, settings_values, path)


def _scalar_outputs(settings_values, physics_spec):
    """n_eff_1 .. n_eff_<n_modes> (ModePhysics.scalars)."""
    return tuple(f'n_eff_{number}' for number in range(1, int(settings_values['n_modes']) + 1))


_SCALAR_OUTPUTS_HELP = (
    'n_eff_<k> for k = 1 .. n_modes (unit 1): Re n_eff of mode k of the last solve, ranked as mode_table (a '
    'frequency_domain study: its last wavelength). A mode that was not found has no value: its derived value is NaN '
    'plus a run-log warning. Example derived value {"name": "n_eff_te0", "expression": "modes_n_eff_1"} (operation '
    'mean of a constant is the value itself); in a parameter sweep it gives n_eff per value.')

_SHARED_VALIDATE = make_validate_physics(MODE_PERMITTIVITY_SLOTS, time_domain=False,
                                         solved_wavelengths=all_solved_wavelengths)


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared EM checks (periodic pairing, materials, slot writers, resolution and skin depth at every solved
    wavelength), the unused wavelength setting, other EM types, the bend, the solver, the Bloch settings and the
    triangles-mesh rules."""
    path = ('physics', physics_spec.name)
    issues = _SHARED_VALIDATE(vctx, physics_spec, settings_values)
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(_in_plane_em_issues(vctx, path))
    issues.extend(_bend_issues(vctx, settings_values, path))
    issues.extend(_solver_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_bloch_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_triangle_issues(vctx, physics_spec, settings_values, path))
    return issues


# run-time estimate: host bytes moved per stored operator entry by one cross-section assembly (scipy curl products, PML
# stretch, active-set slicing, the shifted operator; calibrate against the assembly_seconds of the solve infos)
MODE_ASSEMBLY_BYTES_PER_ENTRY = 800
_MODE_ENTRIES_PER_ROW = 9   # CSR entries per [h_x; h_y] row of the cross-section operator (as the memory estimate)
# SuperLU of the [h_x; h_y] operator on the scalar 2D curve priced at n = 2 x cells: b = 2 unknowns per grid point make
# nested-dissection flops b^3 (n / b)^1.5 = b^1.5 n^1.5 and fill ~ b n log n (theory, as definition_bor's b = 3)
MODE_LU_SCALE = 2.0 ** 1.5
MODE_SOLVE_SCALE = 2.0
_DETACHED_FEATURES = ('shg_overlap', 'bend_overlap')   # each: one more SuperLU eigen-solve per mode solve
# ... of a triangles mesh: host bytes per stored entry of one fem_modes assembly (quadrature-point coefficients, six
# blocks, restriction, the shifted operator) and Python calls; uncalibrated (compare the solve infos' assembly_seconds)
FEM_MODE_ASSEMBLY_BYTES_PER_ENTRY = 2000
FEM_MODE_ASSEMBLY_CALLS = 400


def _time_estimate(settings_values, mesh, context):
    """Per mode solve (one, or one per frequency_domain wavelength) the cross-section assemblies and shift-invert
    eigen-solves of the 'eigen' role (2 n_modes + 10 eigenpairs): the main one, two more with group_index, one per
    shg_overlap / bend_overlap; a line_block 'shifted' node factors the main operators and serves all applications
    (the detached ones are SuperLU: factorised, applications priced alike); triangle meshes: _triangle_time_estimate."""
    if mesh.kind == 'triangles':
        return _triangle_time_estimate(settings_values, mesh, context)
    unknowns = 2.0 * int(mesh.n_cells)   # walls and mirrors remove some: an upper figure
    nnz = _MODE_ENTRIES_PER_ROW * unknowns
    points = float(context.count('harmonic_points', 1)) if context.study.type == FREQUENCY_DOMAIN_STUDY else 1.0
    main = 3.0 if settings_values['group_index'] else 1.0   # omega and omega (1 +- step)
    detached = float(sum(1 for feature in context.physics_spec.features
                         if feature.enabled and feature.type in _DETACHED_FEATURES))
    assembly = (te.HostStream(MODE_ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200))
    solves = te.TimeEstimate((te.TimeItem('cross-section assemblies', 'solve', points, 'exact',
                                          tuple((main + detached, op) for op in assembly),
                                          counter=physics_counter(context, 'solves')),))
    eigenpairs = 2 * int(settings_values['n_modes']) + 10
    common = dict(unknowns=unknowns, nnz=nnz, value_bytes=16, solves=points * (main + detached),
                  eigenpairs=eigenpairs, factor_scale=MODE_LU_SCALE, solve_scale=MODE_SOLVE_SCALE,
                  default_basis=mode_default_basis(eigenpairs, cpml_cross_section(settings_values)))
    # the line_block path prices its own product_ops
    shifted = context.solvers.get('shifted')
    if shifted is None or shifted.solver_type.name != 'line_block':
        return te.combine((solves, context.solver_work('eigen', factorizations=points * (main + detached),
                                                       backend='cpu', **common)))   # host SuperLU shift-invert
    nx, ny = int(mesh.nx), int(mesh.ny)
    periodic = (settings_values['bc_x_lo'] == 'periodic', settings_values['bc_y_lo'] == 'periodic')
    axis = 1 if all(periodic) else line_block_axis(nx, ny, *periodic)   # both periodic: refused by validation
    block = 2 * (nx if axis == 0 else ny)
    parts = [solves, kept_line_block_factors(context, unknowns, nnz, points * main, block),
             context.solver_work('eigen', factorizations=0.0, product_ops=factor_solve_ops(
                 unknowns, block, nnz, shifted.values.get('precision', 'mixed')), **common)]
    if detached:
        parts.append(te.TimeEstimate((te.TimeItem('detached SuperLU factorisations', 'factorisation', points * detached,
                                                  'exact', (te.HostLU(unknowns, 2, True, scale=MODE_LU_SCALE),),
                                                  counter=role_counter(context, 'eigen', 'eigen_factorizations')),)))
    return te.combine(parts)


def _triangle_time_estimate(settings_values, mesh, context):
    """em_modes on a triangles mesh: per mode solve one FEM assembly (+ 2 with group_index: the perturbation pencils)
    and one eigen-solve of the estimated unknowns (host SuperLU priced at FEM_LU_FILL_FACTOR x the 2D curve, uncalibrated; the
    arpack_shift_invert recipe leaves cuDSS in 2D unestimated)."""
    order = DISCRETIZATION_ORDERS[settings_values['discretization']]
    unknowns = float(estimated_unknowns(int(mesh.n_elements), int(mesh.n_edges), int(mesh.n_nodes), order))
    nnz = FEM_ENTRIES_PER_ROW[order] * unknowns
    points = float(context.count('harmonic_points', 1)) if context.study.type == FREQUENCY_DOMAIN_STUDY else 1.0
    assemblies = 3.0 if settings_values['group_index'] else 1.0   # the Hellmann-Feynman pencils at omega (1 +- step)
    assembly = (te.HostStream(FEM_MODE_ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(FEM_MODE_ASSEMBLY_CALLS))
    solves = te.TimeEstimate((te.TimeItem('FEM cross-section assemblies', 'solve', points, 'exact',
                                          tuple((assemblies, op) for op in assembly),
                                          counter=physics_counter(context, 'solves')),))
    host = eigen_backend_kwargs(context)
    return te.combine((solves, context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=16,
                                                   solves=points, factorizations=points,
                                                   eigenpairs=2 * int(settings_values['n_modes']) + 10,
                                                   factor_scale=FEM_LU_FILL_FACTOR, solve_scale=FEM_LU_FILL_FACTOR,
                                                   **host)))


EM_MODES = PhysicsType(
    name=_TYPE_NAME,
    label='Electromagnetic Waves, cross-section modes',
    family=EM_FAMILY,
    group='em',
    spaces_2d=('cross_section',),
    default_instance_name='modes',
    description=_DESCRIPTION,
    summary='Guided modes of a waveguide whose cross-section is the drawn plane, light travelling out of the plane: '
            'effective index, loss, polarisation and field shape of each mode, optionally heated or bent.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS[MODE_POLARIZATION],
    state_fields=tuple(ModeRenderer.field_infos[name]
                       for name in ('e_intensity', 'sz', *MODE_PERMITTIVITY_ARRAYS, 'sigma_e')),
    output_fields=tuple(ModeRenderer.field_infos.values()),
    source_slots=source_slot_docs(MODE_PERMITTIVITY_SLOTS),
    supported_studies=(_MODE_ANALYSIS_STUDY, FREQUENCY_DOMAIN_STUDY, _THERMAL_NOISE_STUDY, OPTOMECHANICS_STUDY),
    default_study={'type': _MODE_ANALYSIS_STUDY},
    uses_dispersion=True,
    create=_create,
    domain_requirements=fdfd_domain_requirements,   # bloch faces where bloch_kx / bloch_ky != 0
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate_physics,
    memory_estimate=modes_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALAR_OUTPUTS_HELP,
    # shifted: the shift-invert factors of the eigen-solve; Automatic sparse_direct = host SuperLU through the eigen
    # role (nothing of its own), line_block = batched device factors (ModePhysics class comment)
    # eigen MIXED_FACTORIZATIONS: the Rayleigh-polished complex64 LDL^T of the triangle pencil; refused on tensor meshes
    solver_roles={'eigen': eigen_role(mixed_factorization=True),
                  'shifted': SolverRole('general', 'sparse_direct', precisions=('mixed', 'double'))},
    subpixel_smoothing=True,
    conformal_boundaries=True,   # pec_domain cut cells (cross_section_modes pec_cuts; 2D Cartesian)
    backends=('gpu', 'cpu'),   # host eigen solve, mode fields on the Domain's xp (solver_plan S6)
    grids=BOTH_GRIDS,         # triangles: solvers/modes_fem (mesh plan 2.6)
)


def _selftest_cases():
    """The CPU cross-section then the model-run em_modes SelfTestCases (imported on the first registry.selftests())."""
    from .selftests_carriers import CARRIER_SELFTEST_CASES
    from .selftests_cross_section import CROSS_SECTION_CPU_SELFTESTS
    from .selftests_em_modes import EM_MODES_SELFTEST_CASES
    from .selftests_modes_backend import MODES_BACKEND_SELFTEST_CASES
    from .selftests_modes_fem import MODES_FEM_SELFTEST_CASES
    from .selftests_shg_overlap import SHG_OVERLAP_SELFTEST_CASES
    return (*CROSS_SECTION_CPU_SELFTESTS, *EM_MODES_SELFTEST_CASES, *CARRIER_SELFTEST_CASES,
            *SHG_OVERLAP_SELFTEST_CASES, *MODES_BACKEND_SELFTEST_CASES, *MODES_FEM_SELFTEST_CASES)


def register(registry):
    """Register em_modes (after the FDTD / FDFD types: their material Params stay first per name), its templates
    (incl. the pn phase shifters of selftests_carriers), then its selftest loader."""
    registry.register_physics(EM_MODES)
    registry.register_template(
        'strip_waveguide_modes', STRIP_WAVEGUIDE_MODES_LABEL, STRIP_WAVEGUIDE_MODES_DESCRIPTION,
        strip_waveguide_modes_spec,
        summary='Guided modes of a 500 by 220 nanometre silicon waveguide in glass at 1550 nm: effective index, '
                'polarisation and field shape of the first four modes.')
    registry.register_template(
        'thermo_optic_phase_shifter', THERMO_OPTIC_PHASE_SHIFTER_LABEL, THERMO_OPTIC_PHASE_SHIFTER_DESCRIPTION,
        thermo_optic_phase_shifter_spec,
        summary='A gold heater warms a silicon waveguide in glass; the effective index of the guided mode rises with '
                'the heater power. Steady heat, then the modes, for three heater powers.')
    registry.register_template(
        'pn_phase_shifter_depletion', PN_PRESCRIBED_LABEL, PN_PRESCRIBED_DESCRIPTION, pn_phase_shifter_depletion_spec,
        summary='A silicon rib waveguide with a junction of p- and n-doped silicon at its centre. Reverse bias widens '
                'the carrier-free zone, which raises the effective index slightly and lowers the loss. The carriers '
                'follow a textbook formula, for three bias voltages.')
    registry.register_template(
        'pn_phase_shifter_electrostatics', PN_ELECTROSTATICS_LABEL, PN_ELECTROSTATICS_DESCRIPTION,
        pn_phase_shifter_electrostatics_spec,
        summary='The doped silicon rib waveguide of the prescribed-carrier phase shifter, but the electric potential '
                'between the two contacts is solved first and the carriers follow it. Three bias voltages.')
    registry.register_selftests(_selftest_cases)

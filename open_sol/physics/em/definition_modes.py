"""Model definition of 'em_modes' (full-vector cross-section modes: settings, material vocabulary, mode table /
dispersion features) on definition_common, with its templates and self-tests; definition.register calls register."""
from ...core import time_estimate as te
from ...core.memory_estimate import COMPLEX128_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.definition_helpers import eigen_factorization, eigen_role, selected_material_sets
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
                                  mode_default_basis, mode_eigen_count, size_issue_for)
from .definition_bloch import bloch_active_axes, bloch_params, fdfd_bloch_issues, fdfd_domain_requirements
from .fem_modes import DISCRETIZATION_ORDERS, FEM_ENTRIES_PER_ROW, FEM_LU_FILL_FACTOR, PERIODIC_TEXT, estimated_unknowns
from .materials import EM_MATERIAL_SCHEMAS, GYRATION_ARRAYS
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
    'Mode-solve CPML: stretch s = kappa + sigma / (alpha + j omega eps0), pml_* grading; a mode tail reaching it gains '
    'a small artificial loss (Im n_eff, see pml_fraction); modes with > half their |H|^2 in PML cells are dropped.',
    PERIODIC_BC_HELP,
    'Walls and mirrors sit on the inner face of the outermost cell (a ghost: each component centred along that axis '
    'takes -1 (pec, antisymmetric) or +1 (pmc, symmetric) times its inner neighbour; components on the plane free '
    '(pec) or 0 (pmc)): half a cell off the em_fdtd_ez pec / em_fdtd_hz pmc walls, so a guide between two pec walls of '
    'an n-cell domain is n - 2 cells wide. Mirrors add their ghost outside the domain (plane = domain edge): symmetric '
    '= E even (PMC plane), antisymmetric = E odd (PEC plane). A mirror through a symmetric guide halves the unknowns '
    'and keeps one parity: on x_lo, antisymmetric keeps E_x even (quasi-TE, e.g. TE0 of a strip), symmetric E_y even '
    '(quasi-TM).',
))
_WAVELENGTH_HELP = (
    'Vacuum wavelength of a mode_analysis solve (k0 = 2 pi / lambda0); materials enter at that omega (sigma_e and '
    'poles as complex eps). frequency_domain solves its own wavelengths instead (validation info when this is not the '
    'default; n_eff versus wavelength: mode_dispersion). Mesh: >= 10 (better 20) cells per lambda0 / |n| and, in '
    'metals, >= 2 (better 4) per skin depth lambda0 / (2 pi |Im n|) (validation checks both); a 220 nm silicon core '
    'needs ~20 nm cells (or refinement) for n_eff to ~1e-2. Typical 1.31e-6 or 1.55e-6 m.')
_SETTINGS = boundary_settings_params(_BC_HELP) + (
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help=_WAVELENGTH_HELP,
          tip='Vacuum wavelength at which the modes are computed, in metres: 1.55e-6 for 1550 nm.\n\n'
              "A 'Frequency domain' study ignores it and computes the modes at its own list of wavelengths. The mesh "
              'needs at least 10 cells per wavelength inside every material.'),
    Param('n_modes', 'int', 4, label='Number of modes', min=1, max=100,
          help='Modes kept per solve: the eigen-solve finds the 2 n_modes + 10 beta^2 nearest the target (k0 '
               'n_eff_guess)^2 (n_modes + 4 on a lossless cross-section without bend or n_eff_guess, back to 2 n_modes '
               '+ 10 when PML modes leave too few), drops PML modes (> half their |H|^2 in PML), ranks by descending '
               'Re(beta^2), ties by descending Im (= descending Re n_eff for low-loss guided modes; evanescent modes '
               'last) and keeps up to n_modes (fewer near cutoff or when most candidates are PML modes). 2-10 typical.',
          tip='How many modes to compute, from the most strongly guided one down.\n\n'
              'Fewer may be found when the guide supports fewer. 2 to 10 is typical.'),
    Param('n_eff_guess', 'float', None, label='Effective index guess', unit='1', min=0.0, min_exclusive=True,
          optional=True,
          help='Shift-invert target sigma = (k0 n_eff_guess)^2: the modes with n_eff^2 nearest it. Empty: the largest '
               'Re sqrt(eps mu) over non-PEC cells and all three components at the solve wavelength (before any bend '
               'map): the most strongly guided modes. Lower it for higher-order modes; set it near the expected value '
               'for a surface plasmon or a mode far below the core index.',
          tip='Effective index near which the modes are searched.\n\n'
              'Empty searches below the highest refractive index of the cross-section, which finds the most strongly '
              'guided modes. Set it to reach higher-order modes.'),
    Param('mode_number', 'int', 1, label='Displayed mode', min=1,
          help='Rank (see n_modes) of the mode shown after the study: Results.field and the GUI view, state fields '
               'e_intensity / sz, selected = True in mode_table. frequency_domain shows that rank at every wavelength '
               '(no tracking; mode_dispersion follows by overlap). > n_modes is a validation error; when fewer modes '
               'are found the last one is shown (run-log warning).',
          tip='Which mode the field view shows after the run: 1 is the most strongly guided one.\n\n'
              'Every mode is also stored as its own snapshot.'),
    Param('group_index', 'bool', False, label='Compute group index',
          help='Group index n_g = C0 d(Re beta) / d(omega) of every mode by a central difference in omega (relative '
               'step 1e-3), materials re-evaluated (pole chi and sigma_e: material dispersion included; plain eps_r has '
               'none). Tensor meshes: two more eigen-solves per wavelength, modes followed by overlap (~3x the time); '
               'triangles: first-order perturbation of the pencil, two more assemblies. Off: n_group NaN.',
          tip='Also compute the group index of every mode, which sets the speed of light pulses in the guide.\n\n'
              'Takes about three times as long on a rectangular grid, a little longer on a triangle mesh.'),
    Param('bend_radius', 'float', 0.0, label='Bend radius', unit='m',
          help='Signed radius R of a guide bent in the xz plane (0 = straight), around the y axis (rings in the chip '
               'plane, y vertical). R > 0: centre of curvature at x_ref - R (outer side +x); R < 0: at x_ref + |R| '
               '(outer side -x); x_ref = bend_reference_x; radius r = |R| s, s = 1 + (x - x_ref) / R. n_eff refers to '
               'the radius at x_ref: angular propagation constant (ring azimuthal order) nu = beta R = k0 n_eff R. '
               'Bend radiation toward the outer side shows as Im n_eff < 0 / loss_db_per_cm: keep a cpml face and '
               'cladding there (validation warns otherwise). s > 0 over the whole grid incl. PML and a non-periodic x '
               'axis are required (validation errors). Typical 5e-6 to 1e-4 m.',
          tip='Radius of a bend of the waveguide, in metres; 0 means straight.\n\n'
              'Positive values bend toward -x, so light leaks toward +x; keep an absorbing layer on that side.'),
    Param('bend_reference_x', 'float', None, label='Bend reference x', unit='m', optional=True,
          help='x (physical frame) where s = 1: the radius n_eff refers to, usually the guide centre. Empty: size_x / '
               '2. Unused when bend_radius = 0.',
          tip='Horizontal position where the bend radius is measured, usually the waveguide centre.\n\n'
              'Empty uses the middle of the domain.'),
    Param('bend_model', 'choice', 'exact', label='Bend model', choices=BEND_MODELS,
          choice_labels=(('exact', 'Exact (cylindrical coordinates)'), ('conformal', 'Conformal map (approximate)')),
          help="Unused when bend_radius = 0. 'exact': the bend written as a straight guide in (x, y, R phi): eps "
               'diag(s, s, 1/s), mu diag(s, s, 1/s), so the true bend mode to discretisation order while the centre of '
               'curvature lies outside the grid; E_x, E_y, H_x, H_y are the physical E_r, E_y, H_r, H_y, E_z, H_z = s x '
               "the physical E_phi, H_phi; power and overlaps use the straight formulas. 'conformal': the approximate "
               'bend-loss map, every eps component times s^2, mu untouched: error second order in w / R (w = mode '
               'width; e.g. -5e-4 relative in n_eff for a 50 um metal-rimmed ring at 1.55 um).',
          tip='How the bend is computed: exactly, or by the older approximate conformal map.\n\n'
              'Keep the exact model unless comparing with the approximation.'),
    Param('discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATION_ORDERS),
          choice_labels=(('quadratic', 'Quadratic'), ('linear', 'Linear')),
          help='Element order on a triangles mesh (tensor meshes use the Yee cross-section). quadratic (industry '
               'standard): second-order Nedelec for E_x, E_y, P2 Lagrange for E_z (Lee-Sun-Cendes mixed form, no '
               'spurious modes), curved edges isoparametric, so circles and ellipses follow the mesh; linear: Whitney '
               '/ P1 on straight elements, cheaper, error ~h^2 instead of ~h^4 in beta^2. ~7 (quadratic) or 2 (linear) '
               'unknowns per element; shift-invert by the eigen solver (host SuperLU; factorization cudss: GPU LDL^T '
               f'of the complex symmetric pencil; {HOST_DIRECT_NAMES}: host MUMPS / PARDISO LDL^T; _mixed: complex64 '
               'with one float64 refinement per solve, ARPACK to 1e-6, beta^2 from float64 Rayleigh quotients, error ~ '
               'the square of the vector error). Materials one value per element (at its centroid). pec / '
               'antisymmetric faces fix tangential E on the window edge, pmc / symmetric are natural (mirror plane = '
               'window edge, no ghost); a cpml face gets mesh pml_layers structured element layers, pml_cells x the '
               'maximum element size thick, pml_* stretch, pec outer side (pml_kappa_max 1 suits leaky modes, the '
               'default 15 evanescent tails); pec_domain removes its elements (E = 0 on their outline). On triangles '
               'periodic faces, bend_radius, shg_overlap, bend_overlap, a line_block shifted solver and thermal_noise '
               '/ optomechanics studies are validation errors; no subpixel smoothing or conformal boundaries.',
          tip='Order of the finite elements when the mesh is made of triangles: quadratic is far more accurate per '
              'element, linear is cheaper.\n\nUnused on a rectangular-cell mesh.'),
) + bloch_params(
    'Used only when bc_{axis}_lo / _hi are periodic: Bloch modes of the periodic cross-section, F({axis} + L) = '
    'F({axis}) exp(-j k{axis} L) per meshed period L (fields ~ exp(j (omega t - beta z))), e.g. a photonic-crystal '
    'fibre supercell or a waveguide array at transverse phase step k{axis} L; k{axis} and k{axis} + 2 pi / L are the '
    'same state, 0 = plain periodic. The phase sits on the operator wrap entries (same cost; operator no longer '
    'symmetric); displayed centre fields carry it across the wrap. Validation errors with shg_overlap (2 omega modes '
    'would need 2 k{axis}) and thermal_noise / optomechanics studies (their overlaps assume k = 0). Typical: pi / L '
    '(edge of the first zone).',
    'Phase step of the mode from one period of the cross-section to the next along {axis}, in radians per metre, '
    'for periodic edges.\n\n0 means no phase step. Use pi divided by the period for the edge of the first zone.')

# ---- material vocabulary (definition_common.em_material_params; em_modes reads eps_xx, eps_yy and eps_zz) ----
_COMPLEX_EPS = ('Solved with the complex value at the solve wavelength: this - j sigma_e / (omega eps0) + chi(omega) '
                'of every Drude / Lorentz pole (exp(+j omega t): loss Im < 0, giving Im n_eff < 0).')
_BACKGROUND = ('For a dispersive material it is the background eps_inf (an explicit permittivity wins over the Drude '
               'eps_inf). A state-dependent value (e.g. of T) is rewritten once after the background solve, before the '
               'mode solve.')
_FACE_VALUE = ('the width-weighted mean of the two cells sharing that face (solver_defaults.subpixel_smoothing '
               'averages interface faces anisotropically)')
_MATERIAL_PROPERTIES = em_material_params('modes', {
    'eps_r': ('Isotropic: eps_xx, eps_yy, eps_zz unless the material sets a component (it wins). n^2 of a lossless '
              'dielectric: silica ~2.085, silicon ~12.1 at 1.55 um. E_z sees the cell value, E_x / E_y on the +x / +y '
              f'cell faces {_FACE_VALUE}. {_COMPLEX_EPS} {_BACKGROUND}'),
    'eps_zz': ('Component along the guide, the one E_z sees at the cell centres; overrides eps_r for E_z. Uniaxial, '
               'optic axis along the guide: eps_zz = n_e^2, eps_r = n_o^2.'),
    'eps_xx': ('The one E_x sees on the +x cell faces (face mean as eps_r); overrides eps_r for E_x. E.g. x-cut lithium '
               'niobate at 1.55 um: eps_xx 4.571 = n_e^2, eps_r 4.889 = n_o^2.'),
    'eps_yy': ('The one E_y sees on the +y cell faces (face mean as eps_r); overrides eps_r for E_y. Optic axis along '
               'y: eps_yy = n_e^2.'),
    'sigma_e': ('All three E components gain -j sigma_e / (omega eps0): Im n_eff < 0, loss_db_per_cm > 0. Weak '
                'absorber of index n, extinction kappa: sigma_e = 2 n kappa omega eps0; metals: a Drude pole, not '
                'their DC conductivity.'),
    'mu_r': ('Real, 1 for optical materials; H_x (+y face), H_y (+x face), H_z (+x+y corner) take the value of their '
             'cell. Solved as mu_r - j sigma_m / (omega mu0).'),
    'sigma_m': 'mu_r gains -j sigma_m / (omega mu0) on H_x, H_y, H_z (artificial absorbers only).',
}, tips={'eps_zz': ('Permittivity for the electric field along z, the direction the light travels.\n\n'
                    "When set it replaces 'Relative permittivity' for that direction.")})
_GYRATION_HELPS = {
    'gyration_x': (
        'Gyration g_x of a material magnetised along x (across the guide): eps_ij += j e_ijk g_k (exp(+j omega t)), '
        'so eps_yz = -eps_zy = j g_x, eps_zx = -eps_xz = j g_y; Hermitian for real g, sign flips with the '
        'magnetisation. Not in the mode solve: each mode gets the first-order non-reciprocal shift delta_beta = '
        'beta_forward - beta_backward = -(omega eps0 / P) integral [g_x Im(conj(E_y) E_z) + g_y Im(conj(E_z) E_x)] dA '
        '(Mizumoto / Shintaku; P = 1 W, mode toward +z): scalar output delta_beta_<k>, mode_table column delta_beta; '
        'error ~g^2 relative (the ~g^2 n_eff change is not computed). E_y couples to E_z, so g_x shifts quasi-TM '
        'modes (E along y), and only where the guide is not mirror symmetric across y (magneto-optic cladding on one '
        'side, or opposite magnetisations: push-pull). L_pi = pi / |delta_beta|. Ce:YIG at 1550 nm (library ce_yig, '
        'n ~ 2.2): |g| ~ 0.009 (g = theta_F lambda n / pi, theta_F ~ -4500 deg/cm); the sign is the material\'s. '
        'Allocated only when a material sets it. Validation errors: a y_lo / y_hi mirror face (its image reverses the '
        'magnetisation), a bend, a triangles mesh, any other EM type (em_fdfd_hz takes gyration_z).'),
    'gyration_y': (
        'Gyration g_y of a material magnetised along y: as gyration_x with x and y swapped: E_x couples to E_z, so it '
        'shifts quasi-TE modes (E along x) where the guide is not mirror symmetric across x; an x_lo / x_hi mirror '
        'face is a validation error.'),
}
_GYRATION_TIP = ('Strength of the magneto-optic effect of a material magnetised along {axis}, such as cerium-doped '
                 'yttrium iron garnet: light then travels with a slightly different effective index forward and '
                 'backward, the effect behind optical isolators. Usually empty.\n\n'
                 'About 0.009 for cerium-doped garnet at 1550 nm; reversing the magnetisation flips its sign.')
_GYRATION_PROPERTIES = tuple(
    Param(name, 'float', 0.0, label=f'Gyration, {axis} (magneto-optic)', unit='1', field_dependent=True,
          expands_to=(name,), help=_GYRATION_HELPS[name], tip=_GYRATION_TIP.format(axis=axis))
    for name, axis in (('gyration_x', 'x'), ('gyration_y', 'y')))

# ---- features ----
_MODE_SCALARS = (
    'n_eff (Re beta / k0), n_eff_imag (< 0 = loss; exp(+j omega t), exp(-j beta z)), loss_db_per_cm = -20 log10(e) '
    'Im(beta) / 100 (> 0 for lossy or leaky modes), n_group (C0 d(Re beta) / d(omega); NaN unless group_index or when '
    'the mode was not followed to the shifted frequencies, run-log warning), te_fraction = int |E_x|^2 dA / int '
    '(|E_x|^2 + |E_y|^2) dA (1 = E along x, e.g. quasi-TE of a strip on a substrate parallel to xz), '
    'effective_area = (int |E|^2 dA)^2 / int |E|^4 dA [m^2] (all three components at the cell centres, '
    'nonlinear-optics definition; whole mode incl. mirror images, as the 1 W normalisation), pml_fraction (share of '
    'the |H|^2 in PML cells: kept modes < 0.5; above ~1e-2 the PML shapes the mode: widen the cladding), delta_beta '
    '(beta_forward - beta_backward [1/m] of gyration_x / gyration_y materials; 0 without them)')
_MODE_TABLE_DESCRIPTION = (
    "Modes of the last solve (frequency_domain: the last wavelength). Result 'table', one row per mode, ranked as "
    f'n_modes: mode_number, {_MODE_SCALARS}, selected (the mode_number setting); extras wavelength [m], selected_mode. '
    "n_eff in a sweep: scalar output <physics>_n_eff_<k>; any other column: derived value source '<physics>.<this "
    "feature>', expression = column name, operation at, at = mode_number - 1 (row index).")
_MODE_DISPERSION_DESCRIPTION = (
    "Scalars of one mode versus vacuum wavelength. Result 'xy' (x = wavelength [m], ascending; one point per solved "
    'wavelength, a mode_analysis study gives one; a re-solved wavelength keeps its latest), series = the mode_table '
    'scalar columns (n_eff, n_eff_imag, loss_db_per_cm, n_group, te_fraction, effective_area, pml_fraction, '
    'delta_beta) and overlap (|overlap| with the mode followed at the previous solve, 1 at the first; well below 1 = '
    'the mode changed character or crossed another); extras frequency [Hz], wavelength [m], mode_number. The first '
    'solve finding mode_number modes picks that rank; later solves follow the largest overlap, keeping identity where '
    'modes swap order (solve order: the frequency_domain wavelengths list, else wavelength_start to wavelength_stop). '
    'NaN rows while the mode is not found.')
_MODE_DISPERSION_PARAMS = (
    Param('mode_number', 'int', 1, label='Mode number', min=1,
          help='Rank (as mode_table) of the followed mode at the first solve that finds that many modes; later solves '
               'follow it by overlap.',
          tip='Which mode to follow across the wavelengths: 1 is the most strongly guided one at the first '
              'wavelength.\n\nThe same mode is then followed even where two modes swap order.'),
)


_SHG_TEXT = (
    "Coupled-mode second-harmonic generation of a 3D waveguide from its cross-section modes. Result 'xy' versus pump "
    'vacuum wavelength [m] (one point per solve): after every em_modes solve at omega it solves the 2 omega modes '
    '(same materials, walls, bend and eigen solver: ~2x the time) and pairs rank pump_mode (omega) with rank '
    'harmonic_mode (2 omega), no tracking. Series: n_eff_pump, n_eff_harmonic, delta_beta = Re beta(2 omega) - 2 Re '
    'beta(omega) [1/m], coherence_length = pi / |delta_beta| and poling_period = 2 pi / |delta_beta| [m] (NaN at '
    'delta_beta = 0), overlap = |kappa| [V^2/W^1.5], kappa = int conj(e_2w) . (d : e_w e_w) dA over the selected '
    'domains (1 W modes, cell-centre fields), efficiency = (omega eps0 / 2)^2 |kappa|^2 [%/(W cm^2)] (P_2w = '
    'efficiency P_w^2 L^2 phase matched, times sinc^2(delta_beta L / 2) otherwise; a plane-wave mode of area A gives '
    '8 pi^2 d^2 / (eps0 c n_w^2 n_2w lambda^2 A)), efficiency_qpm = (2 / pi)^2 efficiency (first-order QPM, 50 % '
    'duty). Convention: exp(j (omega t - beta z)), P(2 omega)_i = eps0 sum_jk d_ijk E_j E_k, d = chi2 / 2 as Voigt '
    'd_il in MODEL axes (x, y cross-section, z along the guide): rotate crystal values yourself, e.g. x-cut lithium '
    'niobate, c axis along x, TE modes: d11 = d33(LN) ~ 2.5e-11 m/V, d12 = d13 = d31(LN) ~ -4.6e-12 m/V. The mesh must '
    'resolve the 2 omega modes (>= 10 cells per lambda / (2 n)); mirror faces and Bloch k != 0 are validation errors '
    '(the integrand has no single parity); loss enters only through Re beta. Bend: kappa is the physical overlap per '
    'arc length at bend_reference_x (where delta_beta and L are measured): exact integrates s dA with E_phi = E_z / '
    's, conformal s^2 dA with the fields taken as physical (exact for E along y). ring_crystal local: d in the local '
    'frame (x radial, y vertical, z along the guide), turning with the guide: right for terms invariant under '
    'rotations about y (d33, d31, d15 of a z-cut crystal, c axis along y). ring_crystal fixed (bend only; x- or y-cut '
    'ring): d given at the reference cross-section (phi = 0), crystal fixed while the guide turns about y: local '
    'd(phi) = Q(phi) Q Q : d, Q rows (cos phi, 0, sin phi), (0, 1, 0), (-sin phi, 0, cos phi), phi = z / R (signed R, '
    'z the arc at x_ref); kappa(phi) = sum_m=-3..3 kappa_m exp(j m phi) (exact from 8 azimuths), order m phase '
    'matched when delta_beta R + m = 0. Extra series: azimuthal_mismatch = delta_beta R [1] (nu_2w - 2 nu_w), '
    'ring_order m* = round(-delta_beta R) (NaN when |m*| > 3), ring_mismatch = delta_beta + m* / R [1/m], '
    'ring_overlap = |kappa_m*|, ring_efficiency = (omega eps0 / 2)^2 |kappa_m*|^2 [%/(W cm^2)] (0 when |m*| > 3; '
    'P_2w = ring_efficiency P_w^2 L^2 sinc^2(ring_mismatch L / 2) over an arc L), ring_overlap_minus3 .. '
    'ring_overlap_plus3 = |kappa_m| [V^2/W^1.5]; overlap / efficiency stay those at phi = 0. E.g. x-cut LN, TE: '
    'kappa(phi) ~ cos^3 phi: orders +-1 (3/8) and +-3 (1/8) of kappa(0).')


def _shg_params():
    """Mode choice, the 2 omega solve and the 18 model-axis d_il."""
    return (
        Param('pump_mode', 'int', 1, label='Pump mode', min=1,
              help='Rank (as mode_table) of the pump mode of each em_modes solve; must not exceed n_modes.',
              tip='Which computed mode carries the pump light: 1 is the most strongly guided one.'),
        Param('harmonic_mode', 'int', 1, label='Second-harmonic mode', min=1,
              help='Rank of the 2 omega mode among the n_harmonic_modes; must not exceed n_harmonic_modes.',
              tip='Which mode at the doubled frequency receives the generated light: 1 is the most strongly guided '
                  'one.'),
        Param('n_harmonic_modes', 'int', 4, label='Modes at the second harmonic', min=1, max=100,
              help='Modes kept by the 2 omega solve (as n_modes).',
              tip='How many modes to compute at the doubled frequency.'),
        Param('harmonic_n_eff_guess', 'float', None, label='Second-harmonic index guess', unit='1', min=0.0,
              min_exclusive=True, optional=True,
              help='n_eff_guess of the 2 omega solve; empty = the largest index of the cross-section at 2 omega.',
              tip='Effective index near which the second-harmonic modes are searched; empty searches below the highest '
                  'refractive index.'),
        Param('ring_crystal', 'choice', 'local', label='Crystal around the ring', choices=RING_CRYSTALS,
              choice_labels=(('local', 'Turns with the guide'), ('fixed', 'Fixed in the chip (x- or y-cut ring)')),
              help="'local': d_il in the guide's local frame everywhere. 'fixed' (bent guides only, validation error "
                   'otherwise): d_il at the reference cross-section, crystal fixed while the guide turns, so phase '
                   'matching picks an azimuthal order m = -3..3 (ring_* series; see the feature).',
              tip='How the crystal axes behave around a ring.\n\n- Turns with the guide: the crystal looks the same '
                  'from every point of the ring, for example a crystal axis pointing up out of the chip.\n- Fixed in '
                  'the chip: the crystal axis lies in the chip plane, so the guide sees it from a changing angle as it '
                  'goes around; the results then also list the ring phase matching.'),
        # one help for all 18 (no name in it), so describe prints them as one entry
        *(Param(name, 'float', 0.0, label=name, unit='m/V',
                help='Voigt d_il in model axes: i = polarisation component, l = field pair (1, 2, 3 = x, y, z; 4 yz, '
                     '5 xz, 6 xy); 1 pm/V = 1e-12 m/V.',
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
    'last solve (frequency_domain: the last wavelength): after every em_modes solve it solves the same cross-section '
    'straight (same materials, walls, eigen solver; n_straight_modes near straight_n_eff_guess) and pairs each bent '
    'mode with the straight mode of largest power coupling overlap = |c_bs c_sb| / |c_bb c_ss|, c_ab = 1/2 int (E_a x '
    'H_b*).z dA (bent transverse fields = the physical E_r, E_y, H_r, H_y in both bend models). Columns: mode_number '
    '(bent rank), n_eff, n_eff_imag (bent, referred to bend_reference_x), straight_mode, n_eff_straight, overlap (1 = '
    'perfect), junction_loss_db = -10 log10(overlap) (one junction, ~1 / R^2 for large R; a ring coupler or S-bend has '
    'two per transition), overlap_same_rank (straight mode of the same rank, NaN when fewer found); extras wavelength '
    '[m], n_straight_found. The junction is centred: the outward shift of the bent mode (compensable by a lateral '
    'offset) counts as loss.')
_BEND_OVERLAP_PARAMS = (
    Param('n_straight_modes', 'int', 4, label='Straight modes', min=1, max=100,
          help='Modes kept by the straight solve (as n_modes).',
          tip='How many modes of the straight guide to compute for the comparison.'),
    Param('straight_n_eff_guess', 'float', None, label='Straight index guess', unit='1', min=0.0,
          min_exclusive=True, optional=True,
          help='n_eff_guess of the straight solve; empty = the largest index of the cross-section.',
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
                description='E_z = 0 in cells whose centre lies in the domains, E_x / E_y = 0 on faces between two '
                            'such cells (eps^-1 = 0), staircased. With solver_defaults.conformal_boundaries '
                            '(Dey-Mittra on the projected Yee cell) the drawn outline instead: E_z = 0 at the bisected '
                            'wall distance along each centre link into the metal, H_z takes the free lengths of its '
                            'E_x / E_y edges over the free dual-cell area (free fractions >= 0.02); links and dual '
                            'cells crossing a pec / pmc / mirror wall plane stay staircased. Triangles: elements whose '
                            'centroid lies in the domains leave the unknowns, tangential E and E_z = 0 on their '
                            'outline (curved outlines isoparametric with quadratic).',
                summary='Turns the selected domains into a perfect metal, following the mesh cells. Use it for '
                        'idealised metal walls or hollow metal waveguides.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0), grids=BOTH_GRIDS),
)

# ---- type ----
_DESCRIPTION = (
    'Full-vector finite-difference eigenmodes of a waveguide cross-section: xy is the cross-section (x right, y up), '
    'the guide runs along z; fields ~ exp(j (omega t - beta z)), n_eff = beta / k0, so lossy / leaky modes have Im '
    'n_eff < 0. Projected 3D Yee cell (transverse-H formulation, Fallahkhair, Li and Murphy, JLT 2008): E_z at cell '
    'centres, E_x, H_y on +x faces, E_y, H_x on +y faces, H_z at the +x+y corner; eps_zz (array eps_r) the cell value, '
    'eps_xx / eps_yy on faces the width-weighted mean of the two cells, mu the cell value '
    '(solver_defaults.subpixel_smoothing at interfaces: E_x / E_y faces 1 / (n^2 <1/eps> + (1 - n^2) / <eps>), n the '
    'in-plane normal component, E_z cells the area mean <eps>); all complex at the solve wavelength. A h = beta^2 B h '
    'for h = eta0 (H_x, H_y), shift-invert Arnoldi around (k0 n_eff_guess)^2 on the host (scipy; 2 unknowns per cell '
    "incl. PML; validation warns near and refuses beyond the factor size limit). Role 'eigen' arpack_shift_invert "
    '(tolerance, subspace_size, factorization: superlu; cudss = GPU LU without size limit, guarded by free device '
    f'memory; {HOST_DIRECT_NAMES} = host MUMPS / PARDISO under the size limit; _mixed factorizations only on '
    "triangles: this operator is not complex symmetric). Role 'shifted': Automatic = that SuperLU factor; line_block "
    'factors the shift-invert operators on its backend (grid rows or columns, a periodic axis inside the blocks, two '
    'periodic axes an error; mixed or double, refined against complex128), batching the omega and group_index '
    'omega (1 +- 1e-3) operators and frequency_domain wavelengths, up to 8 per batch within 60 % of free memory; a '
    'rejected solve is redone on SuperLU (run-log warning); shg_overlap / bend_overlap stay SuperLU. Modes are dropped '
    '(> half their |H|^2 in PML) and ranked as n_modes. Each carries 1 W, P = 1/2 Re int (E x H*).z dA, phased so its '
    'dominant transverse E is real positive at its peak (the lowest-index cell of mirror-equal peaks). Degenerate '
    'pairs (HE11 of a round fibre, TE11 / TM11 of a metal pipe) come back as arbitrary, not necessarily orthogonal, '
    'combinations. Outputs (the displayed mode = mode_number setting, at cell centres, face / corner components '
    'averaged): components = real parts, sz = 1/2 Re(E_x H_y* - E_y H_x*) (power flow along z; its window integral '
    'equals 1 W only to discretisation order and misses any PML share); state fields e_intensity and sz. No absorbed '
    'power: not an em_heat_joule source. Studies: mode_analysis (one solve at the wavelength setting after any '
    'background heat / current solve, one snapshot per mode), frequency_domain (one solve per wavelength). '
    'heat_em_thermo_optic rewrites eps_zz, eps_xx, eps_yy. xy is a cross-section, so em_modes cannot share a model '
    'with em_fdtd_* / em_fdfd_* (validation error), except em_fdfd_ez / _hz as the k_z = 0 illumination in a '
    'thermal_noise study. Triangles mesh: finite elements of the discretization setting (Lee-Sun-Cendes mixed form, '
    'complex symmetric with PML / loss), one material value per element, a structured PML ring, the same outputs on '
    'a display raster (state fields per element) and the same table / dispersion results; refusals under '
    "discretization. Instance usually 'modes'.")


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


def _gyration_issues(vctx, settings_values, path):
    """A selected material setting gyration_x / gyration_y: errors for a mirror face normal to the other transverse
    axis (its image reverses that magnetisation), a bend and a triangles mesh."""
    issues = []
    set_names = [name for name in GYRATION_ARRAYS[MODE_POLARIZATION] if selected_material_sets(vctx, (name,))]
    for name, axis in (('gyration_x', 'y'), ('gyration_y', 'x')):
        faces = [face for face in (f'{axis}_lo', f'{axis}_hi')
                 if name in set_names and settings_values[f'bc_{face}'] in ('symmetric', 'antisymmetric')]
        if faces:
            issues.append(Issue('error', path, f"a material sets {name} while {' / '.join(faces)} is a mirror face: "
                                               f'the mirror image of a magnetisation along {name[-1]} is reversed, so '
                                               'the mirrored guide is not the drawn one; model that side whole'))
    if set_names and settings_values['bend_radius'] != 0.0:
        issues.append(Issue('error', path, f"{' / '.join(set_names)}: the non-reciprocal phase shift is computed for "
                                           'straight guides only; set bend_radius to 0'))
    if set_names and mesh_kind(vctx) == 'triangles':
        issues.append(Issue('error', path, f"{' / '.join(set_names)}: the non-reciprocal phase shift is computed on "
                                           'tensor meshes only; use a tensor mesh'))
    return issues


def _scalar_outputs(settings_values, physics_spec):
    """n_eff_1 .. n_eff_<n_modes>, delta_beta_1 .. delta_beta_<n_modes> (ModePhysics.scalars)."""
    numbers = range(1, int(settings_values['n_modes']) + 1)
    return tuple(f'n_eff_{number}' for number in numbers) + tuple(f'delta_beta_{number}' for number in numbers)


_SCALAR_OUTPUTS_HELP = (
    'n_eff_<k>, k = 1 .. n_modes (unit 1): Re n_eff of mode k of the last solve, ranked as mode_table '
    '(frequency_domain: the last wavelength); a mode not found gives NaN plus a run-log warning. Example derived '
    'value {"name": "n_eff_te0", "expression": "modes_n_eff_1"} (operation mean of a constant = the value); per '
    'sweep value in a parameter sweep. delta_beta_<k> (1/m): beta_forward - beta_backward of mode k from '
    'gyration_x / gyration_y (0 without them; NaN for an evanescent mode); L_pi = pi / |delta_beta_<k>|.')

_SHARED_VALIDATE = make_validate_physics(MODE_PERMITTIVITY_SLOTS, time_domain=False,
                                         solved_wavelengths=all_solved_wavelengths)


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared EM checks (periodic pairing, materials, slot writers, resolution and skin depth at every solved
    wavelength), the unused wavelength setting, other EM types, the bend, the solver, the Bloch settings, the
    triangles-mesh rules and the gyration rules."""
    path = ('physics', physics_spec.name)
    issues = _SHARED_VALIDATE(vctx, physics_spec, settings_values)
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(_in_plane_em_issues(vctx, path))
    issues.extend(_bend_issues(vctx, settings_values, path))
    issues.extend(_solver_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_bloch_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_triangle_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_gyration_issues(vctx, settings_values, path))
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
    eigen-solves of the 'eigen' role (cross_section_modes.mode_eigen_count wide 2 n_modes + 10, an upper figure:
    lossless solves ask fewer; a real operator priced as real ARPACK, E1): the main one, two more with group_index, one per
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
    eigenpairs = max(mode_eigen_count(int(settings_values['n_modes']), int(unknowns)), 1)   # wide: an upper figure
    default_basis = mode_default_basis(eigenpairs, cpml_cross_section(settings_values))
    common = dict(unknowns=unknowns, nnz=nnz, value_bytes=16, solves=points * (main + detached),
                  eigenpairs=eigenpairs, factor_scale=MODE_LU_SCALE, solve_scale=MODE_SOLVE_SCALE,
                  default_basis=default_basis)
    # the line_block path prices its own product_ops
    shifted = context.solvers.get('shifted')
    if shifted is None or shifted.solver_type.name != 'line_block':
        # E1: a real operator's own host factor is float64 and host ARPACK runs real (one real solve per application,
        # float64 basis: value_bytes 8 prices exactly that route); a cuDSS factor (device matvec) or the device
        # Krylov-Schur around the host solve (gpu run, large basis) stay complex
        from ...model.definition_helpers import eigen_subspace_size
        from ...solvers.arpack_shift_invert import arnoldi_basis_size, device_arnoldi_fits
        from .memory_estimates import real_cross_section_operator
        basis = arnoldi_basis_size(eigenpairs, int(unknowns), eigen_subspace_size(context.solvers), default_basis)
        device_krylov = context.backend == 'gpu' and device_arnoldi_fits(unknowns, basis, host_matvec=True)
        if (eigen_factorization(context.solvers) not in CUDSS_FACTORIZATIONS and not device_krylov
                and real_cross_section_operator(settings_values, context)):
            common['value_bytes'] = 8
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
                                                   eigenpairs=mode_eigen_count(int(settings_values['n_modes'])),
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
    material_properties=_MATERIAL_PROPERTIES + _GYRATION_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS[MODE_POLARIZATION],
    optional_material_schema={name: (0.0, (name,)) for name in GYRATION_ARRAYS[MODE_POLARIZATION]},
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
    from .selftests_gyrotropy import GYROTROPY_MODES_SELFTEST_CASES
    from .selftests_modes_backend import MODES_BACKEND_SELFTEST_CASES
    from .selftests_modes_fem import MODES_FEM_SELFTEST_CASES
    from .selftests_shg_overlap import SHG_OVERLAP_SELFTEST_CASES
    return (*CROSS_SECTION_CPU_SELFTESTS, *EM_MODES_SELFTEST_CASES, *CARRIER_SELFTEST_CASES,
            *SHG_OVERLAP_SELFTEST_CASES, *MODES_BACKEND_SELFTEST_CASES, *MODES_FEM_SELFTEST_CASES,
            *GYROTROPY_MODES_SELFTEST_CASES)


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

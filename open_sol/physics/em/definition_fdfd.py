"""Model definitions of the 2D FDFD physics types 'em_fdfd_ez' (E_z, H_x, H_y) and 'em_fdfd_hz' (H_z, E_x, E_y), built
by one factory: settings, material vocabulary, features; shared callbacks: definition_common / _feature_validation."""
from functools import partial

import numpy as np

from ...core import time_estimate as te
from ...core.backend import to_host
from ...core.memory_estimate import BOOL_BYTES, MemoryEstimate
from ...core.renderer import FieldInfo
from ...core.mirror import mirror_faces
from ...model.definition_helpers import (FDFD_LINEAR_ROLE, eigen_factorization, eigen_role, evaluates_to,
                                         host_direct_back_end, linear_solver_cudss, optimization_evaluations,
                                         resolved_back_end, selected_material_sets, validate_points_inside)
from ...model.schema import (TIME_DEPENDENT_STUDY, FeatureType, Issue, Param, PhysicsType, coupling_writes, feature_path,
                             known_physics, solver_backend)
from .definition_common import (BOTH_GRIDS, BOX_CELLS, CPML_BC_HELP, DEFAULT_WAVELENGTH, EIGENFREQUENCY_STUDY,
                                EZ_GAUSSIAN_BEAM_DESCRIPTION, EZ_LINE_SOURCE_DESCRIPTION, PLANE_WAVE_SPAN,
                                EM_FAMILY, EZ_CURRENT_TIP, EZ_PROBE_COMPONENT_TIP, FACE_MEAN, FREQUENCY_DOMAIN_STUDY,
                                HOST_DIRECT_NAMES,
                                MIRROR_BC_HELP, MODE_DIRECTION_CHOICE_LABELS, MODE_NUMBER_TIP_START, MODE_POWER_TIP,
                                MODE_SOURCE_NUMBER_PARAM, MODE_TABLE_ROWS, N_ANGLES_PARAM, OPTOMECHANICS_STUDY,
                                PERIODIC_BC_HELP, POINT_SOURCE_SIGN, PROBE_COMPONENT_CHOICE_LABELS, SPAN_CELLS,
                                all_solved_wavelengths, apply_ez_point_source, apply_far_field, apply_field_probe,
                                apply_flux_monitor, apply_gaussian_beam, apply_hz_point_source, apply_line_source,
                                apply_mode_monitor, apply_mode_source, apply_pec, apply_tfsf, beam_params,
                                boundary_settings_params, eigen_backend_kwargs, eigen_solved, em_material_params,
                                first_solved_wavelength, hz_point_source_params, make_far_field_result_callback,
                                make_validate_physics, mesh_kind, mode_end_text, mode_number_param, object_result,
                                resonance_eigenpairs, resonance_table_study_issues, sheet_amplitude_param,
                                shortest_solved_wavelength, source_slot_docs, unused_wavelength_issues,
                                write_dispersive_materials)
from .definition_feature_validation import (bloch_period_line_issues, electric_dipole_issues,
                                            harmonic_probe_interface_issues, make_validate_mode_monitor,
                                            make_validate_mode_source, monitor_line_interface_issues,
                                            validate_ez_line_feature, validate_ez_point_feature, validate_far_field,
                                            validate_harmonic_tfsf, validate_hz_line_feature,
                                            validate_hz_point_source, validate_line_feature)
from .definition_adjoint import adjoint_objective_feature
from .definition_conducting_sheet import conducting_sheet_feature
from .definition_bloch import (bloch_active_axes, fdfd_bloch_issues, fdfd_bloch_params, fdfd_domain_requirements,
                               is_fdfd_port_feature)
from .definition_design import OPTIMIZATION_STUDY, design_region_feature
from .eigenfrequency import (REGION_COLUMN, ResonanceRegion, ResonanceTable, resonance_filter_time_item,
                             resonance_table_result)
from .fdfd_line_block import BOTH_PERIODIC_TEXT, fdfd_block_axis
from .fem_fdfd import FEM_FDFD_ENTRIES_PER_ROW, FEM_FDFD_LU_FILL, fem_fdfd_size_issue, scalar_unknowns
from .fem_modes import DISCRETIZATION_ORDERS, PERIODIC_TEXT
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, expected_back_end, scipy_available, size_issue
from .fdfd_monitors import harmonic_far_field_result, harmonic_flux_result, harmonic_probe_result
from .fdfd_ports import harmonic_mode_monitor_result, harmonic_mode_source_result
from .index_reference import OFF_DIAGONAL_ARRAY, OFF_DIAGONAL_SLOT
from .definition_effective_index import effective_index_feature
from .definition_phase_change import phase_change_medium_feature
from .materials import EM_MATERIAL_SCHEMAS, GYRATION_Z
from .memory_estimates import fdfd_memory_estimate, pec_memory_estimate, search_single_basis
from .placement import PERMITTIVITY_SLOTS
from .renderer import FDFDRendererEz, FDFDRendererHz
from .solvers.fdfd_2d import FDFDPhysics
from .solvers.fdfd_2d_fem import FEMFDFDPhysics

_Q_ABS_INFO = FieldInfo('q_abs', 'Absorbed power density', unit='W/m^3', cmap='magma')


# ---- settings ----
_BC_HELP = ' '.join((
    CPML_BC_HELP,
    'FDFD CPML: complex stretch s = kappa + sigma / (alpha + j omega eps0), FDTD grading of the pml_* settings.',
    PERIODIC_BC_HELP,
    MIRROR_BC_HELP,
    'The outermost cell of a wall face (ghost cell of a mirror face) takes the em_fdtd_* row: 0 for E_z pec / H_z '
    'pmc; the neighbour copied (no flux through the inner face) for E_z pmc / symmetric and H_z pec / antisymmetric; '
    'negated for E_z antisymmetric / H_z symmetric. A source in that cell, or an H_z face current on the inner face '
    'of a copied row, radiates nothing. A mirror plane through a symmetric guide halves the unknowns.',
))
_SETTINGS = boundary_settings_params(_BC_HELP) + (
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help='Vacuum wavelength lambda0 of a stationary solve (omega = 2 pi C0 / lambda0). The mesh needs >= 10 '
               '(better 20) cells per lambda0 / |n| and, in a lossy material or metal, >= 2 (better 4) per skin depth '
               'lambda0 / (2 pi |Im n|); validation checks both with the complex eps(omega) at every solved '
               'wavelength. A frequency_domain study ignores it (info when not the default) and solves its own '
               'list; eigenfrequency (and the optical solve of optomechanics) takes it as the target. A parameter '
               'sweep over it repeats a coupled stationary solve (e.g. with heat) per value, cold. Typical 0.4e-6 to '
               '2e-6 m.',
          tip='Vacuum wavelength at which the field is solved, in metres: 1.55e-6 for 1550 nm.\n\n'
              "A 'Frequency domain' study ignores it and solves its own list of wavelengths; an 'Eigenfrequency' study "
              'looks for the resonances closest to it. The mesh needs at least 10 cells per wavelength inside every '
              'material.'),
    Param('discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATION_ORDERS),
          choice_labels=(('quadratic', 'Quadratic'), ('linear', 'Linear')),
          help='Element order on a triangles mesh (tensor meshes ignore it). E_z / H_z on Lagrange elements: '
               'quadratic (error ~h^4 in k^2, curved edges isoparametric; + one unknown per edge) or linear (~h^2); '
               'materials one value per element (H_z: the in-plane tensor with eps_xy); a cpml face gets mesh '
               'pml_layers element layers (pml_* stretch, pec outer side). E_z: pec / antisymmetric faces and '
               'pec_domain elements hold E_z = 0, pmc / symmetric are natural; H_z: pmc / symmetric hold H_z = 0, '
               'pec / antisymmetric and pec_domain outlines are natural (no staircase). Factorisation: host SuperLU '
               f'(cusolver_qr unused), cudss / cudss_mixed or {HOST_DIRECT_NAMES}, the analysis reused. Only '
               'point_source, field_probe (in-plane components from the element gradient), pec_domain and '
               'resonance_table work on triangles; other features, periodic / Bloch faces and the optimization / '
               'optomechanics / thermal_noise studies are validation errors.',
          tip='Order of the finite elements when the mesh is made of triangles: quadratic is far more accurate per '
              'element, linear is cheaper.\n\nUnused on a rectangular-cell mesh.'),
) + fdfd_bloch_params()


# ---- material vocabulary (definition_common.em_material_params; only the help texts are FDFD-specific) ----
_COMPLEX_EPS = ('Solved as the complex eps(omega) = this value - j sigma_e / (omega eps0) + chi(omega) of every '
                'Drude / Lorentz pole (exp(+j omega t): loss has Im < 0).')
_DISPERSIVE_BACKGROUND = ('With poles it is the background eps_inf (an explicit permittivity wins over the Drude '
                          'eps_inf). A state-dependent value (e.g. of T) is rewritten at every coupling update of a '
                          'stationary study (once, after the background solve, in a frequency_domain study).')
_TABULATED_EPS = ('Tabulated eps1 + j eps2 (eps2 <= 0) at one omega: eps1 >= 1: this value eps1, sigma_e = -eps2 '
                  'omega eps0; a metal (eps1 < 1, below the >= 1 bound shared with em_fdtd_*): no permittivity, a '
                  'Drude pole with eps_inf 1, a = 1 - eps1, omega_p = omega sqrt((a^2 + eps2^2) / a), gamma = -omega '
                  'eps2 / a, exact at that omega only (gold -115 - 11j at 1.55 um: omega_p 1.31e16, gamma 1.15e14 '
                  'rad/s).')
_REAL_EPS = 'n^2 of a lossless dielectric (glass ~2.1-2.25, silicon ~12.1 at 1.55 um).'
_SIGMA_M_FDFD = ('mu_r gains -j sigma_m / (omega mu0); sigma_m / (mu0 mu_r) = sigma_e / (eps0 eps_r) is an '
                 'impedance-matched absorber (no reflection at normal incidence).')
_SIGMA_E_FDFD = ('eps gains -j sigma_e / (omega eps0). Weak absorber of index n, extinction kappa at omega: sigma_e = '
                 '2 n kappa omega eps0; model metals with a Drude pole, not their DC conductivity.')
_EZ_MATERIAL_PROPERTIES = em_material_params('ez', {
    'eps_r': f'{_REAL_EPS} E_z sees eps_zz instead when set. {_COMPLEX_EPS} {_DISPERSIVE_BACKGROUND} {_TABULATED_EPS}',
    'sigma_e': f'Ohmic loss: {_SIGMA_E_FDFD}',
    'mu_r': 'mu_r - j sigma_m / (omega mu0) on the H_x / H_y faces, each from the cell it is the +y / +x face of; also '
            'sets the local impedance of line sources. 1 for practically every optical material.',
    'sigma_m': f'Magnetic loss on H_x, H_y: {_SIGMA_M_FDFD}',
})
_HZ_MATERIAL_PROPERTIES = em_material_params('hz', {
    'eps_r': 'Isotropic in-plane value: sets eps_xx and eps_yy unless the material sets those (a component wins). '
             f'{_REAL_EPS} E_x / E_y sit on cell faces: {FACE_MEAN} (complex values; '
             'solver_defaults.subpixel_smoothing averages the real permittivity of interface faces anisotropically). '
             f'{_COMPLEX_EPS} {_DISPERSIVE_BACKGROUND} {_TABULATED_EPS}',
    'sigma_e': f'Ohmic loss on E_x and E_y ({FACE_MEAN}): {_SIGMA_E_FDFD}',
    'mu_r': 'H_z sees mu_zz = mu_r - j sigma_m / (omega mu0) of its cell; also sets the local impedance of line '
            'sources. 1 for practically every optical material.',
    'sigma_m': f'Magnetic loss on H_z: {_SIGMA_M_FDFD}',
}) + (Param(
    OFF_DIAGONAL_ARRAY, 'float', 0.0, label='Relative permittivity, xy (off-diagonal)', unit='1', field_dependent=True,
    expands_to=(OFF_DIAGONAL_ARRAY,),
    help='eps_xy = eps_yx (real) of a crystal whose in-plane axes are turned against x / y: optic axis in the plane '
         'at angle theta from x gives eps_xx = n_o^2 + (n_e^2 - n_o^2) cos^2 theta, eps_yy = n_o^2 + (n_e^2 - n_o^2) '
         'sin^2 theta, eps_xy = (n_e^2 - n_o^2) sin theta cos theta; eps_xx eps_yy > eps_xy^2 is required. Solved '
         'through eta = eps^-1 per cell: an E_x (E_y) face takes the mean of eps_xx - eps_xy^2 / eps_yy (eps_yy - '
         'eps_xy^2 / eps_xx) of its two cells and couples to the other component through eta_xy of the four faces '
         'around it (at normal incidence on a layer only eps_yy - eps_xy^2 / eps_xx matters). 9 points per operator '
         'row (dearer direct solve), allocated only when a material sets it; 0 in PEC cells and the outer two cell '
         'rows of a wall or mirror face. Mode ports, line-source / beam calibrations and the TFSF background ignore '
         'it (the build warns for such a line on eps_xy cells). The adjoint holds eps_xy fixed; output '
         'eps_xy_sensitivity = dJ / d eps_xy. Acousto-optic and Pockels couplings add to it (slot eps_xy_update); '
         'em_fdtd_hz and em_modes have no eps_xy.',
    tip='Cross term of the permittivity of a crystal whose axes are turned in the plane: an electric field along x '
        'then also polarises the material along y. Usually empty.\n\n'
        "Keep it smaller than both the x and y values. Coupled effects (acousto-optic, Pockels) add to it."), Param(
    GYRATION_Z, 'float', 0.0, label='Gyration, z (magneto-optic)', unit='1', field_dependent=True,
    expands_to=(GYRATION_Z,),
    help='Gyration g (real) of a magneto-optic material magnetised along z: in-plane eps = [[eps_xx, eps_xy + j g], '
         '[eps_xy - j g, eps_yy]] (exp(+j omega t); Hermitian, a real g absorbs nothing); g flips sign with the '
         'magnetisation. A uniform medium keeps one index, eps_eff = eps - g^2 / eps, but interfaces and guides become '
         'non-reciprocal: a slab guide without mirror symmetry across it has beta_forward != beta_backward, and '
         'S_21(g) = S_12(-g). Faraday rotation per length |theta_F| = pi |g| / (lambda n): Ce:YIG at 1550 nm (library '
         'ce_yig, n ~ 2.2, theta_F ~ -4500 deg/cm) |g| ~ 0.009, Bi:YIG ~ 0.01-0.02. det = eps_xx eps_yy - eps_xy^2 - '
         'g^2 must stay > 0; direct solves take a general LU (the adjoint: the transposed solve, g held fixed). 0 in '
         'PEC cells and the outer two cell rows of a wall face; mode ports, line-source / beam calibrations and the '
         'TFSF background ignore it (the build warns). Validation errors: mirror faces and parity sectors (the image '
         'reverses the magnetisation), triangles meshes, every other EM type (em_modes takes gyration_x / '
         'gyration_y).',
    tip='Strength of the magneto-optic effect of a magnetised material, such as cerium-doped yttrium iron garnet, '
        'magnetised perpendicular to the drawing. Light then travels differently forward and backward, as in an '
        'optical isolator. Usually empty.\n\n'
        'About 0.009 for cerium-doped garnet at 1550 nm; reversing the magnetisation flips its sign.'),)


# ---- feature parameters ----
_PHASE_PARAM = Param(
    'phase', 'float', 0.0, label='Phase', unit='deg',
    help='The source varies as A cos(omega t + phase) = Re(A e^{j phase} e^{j omega t}): 0 = cos carrier, -90 = the '
         'sin carrier of the FDTD sources (a cw FDTD steady state is the FDFD solution at phase -90). With a single '
         'source it only rotates every phasor, not |E| or the flux.',
    tip='Phase of the source.\n\n'
        'With a single source it only shifts the phase of every result, not field strengths or power. It matters when '
        'several sources interfere; -90 matches the time-domain sources.')
_EZ_POINT_SOURCE_PARAMS = (
    Param('current', 'float', 1e-3, label='Electric line current', unit='A',
          help='Peak z-directed filament current I; mesh-independent field |E_z| = omega mu0 I / 4 |H0^(2)(k r)| in a '
               'homogeneous non-magnetic medium of wavenumber k (1 mA at wavelength 1 um: ~2e5 V/m at r = 1 um). '
               f'{POINT_SOURCE_SIGN}',
          tip=EZ_CURRENT_TIP),
    _PHASE_PARAM,
)
_HZ_POINT_SOURCE_PARAMS = hz_point_source_params((_PHASE_PARAM,), time_domain=False)


def _line_source_params(polarization):
    """Amplitude and phase of a line source in `polarization`."""
    return sheet_amplitude_param(polarization, 'fdfd', beam=False), _PHASE_PARAM


def _gaussian_beam_params(polarization):
    """Amplitude, phase, waist and centre of a gaussian beam in `polarization`."""
    return (sheet_amplitude_param(polarization, 'fdfd', beam=True), _PHASE_PARAM) + beam_params()


_PROBE_PARAMS = {
    'ez': (Param('component', 'choice', 'ez', label='Field component', choices=('ez', 'hx', 'hy'),
                 help='ez (V/m) at the centre of the cell containing the point; hx / hy (A/m) on its +y / +x face (the '
                      'FDTD probe positions), from the discrete curl of E_z.',
                 tip=EZ_PROBE_COMPONENT_TIP, choice_labels=PROBE_COMPONENT_CHOICE_LABELS['ez']),),
    'hz': (Param('component', 'choice', 'hz', label='Field component', choices=('hz', 'ex', 'ey'),
                 help="hz (A/m) at the centre of the cell containing the point; ex / ey (V/m) from Ampere's law (curl "
                      'of H_z plus any line source / beam / dipole current on the face), averaged over the two faces '
                      'of the cell carrying it (collocated, unlike the raw face samples of the FDTD probe).',
                 tip='Field recorded at the point: the magnetic field out of the plane, or the electric field along x '
                     'or y.\n\n'
                     'The electric field is averaged to the centre of the cell holding the point, so it sits at the '
                     'same place as the magnetic field.',
                 choice_labels=PROBE_COMPONENT_CHOICE_LABELS['hz']),),
}
_TFSF_BOX_EDGE_TIP = ('The box edges must lie in one uniform material, which need not be vacuum, with at least one cell '
                      'between them and the domain edge; only a matching symmetry plane at the bottom or top may be '
                      'touched.')
_TFSF_PARAMS = {
    'ez': (Param('amplitude', 'float', 1.0, label='Amplitude', unit='V/m',
                 help='Peak incident E_z (wave of the uniform medium on the box outline).',
                 tip='Peak electric field of the incoming plane wave, out of the plane. Fields scale linearly with '
                     f'it, so 1 is typical.\n\n{_TFSF_BOX_EDGE_TIP}'),
           _PHASE_PARAM),
    'hz': (Param('amplitude', 'float', 1.0, label='Amplitude', unit='V/m',
                 help='Peak incident in-plane |E| (E_y at normal incidence) of the wave of the uniform medium on the '
                      'box outline: H_z = n |E| / (eta0 mu_r), n = sqrt(eps_yy mu_r), also for an oblique Bloch ky '
                      'wave.',
                 tip='Peak in-plane electric field of the incoming plane wave, along y. Fields scale linearly with '
                     f'it, so 1 is typical.\n\n{_TFSF_BOX_EDGE_TIP}'),
           _PHASE_PARAM),
}
_MODE_SOURCE_PARAMS = (
    MODE_SOURCE_NUMBER_PARAM,
    Param('direction', 'choice', 'positive', label='Direction', choices=('positive', 'negative'),
          help='positive: toward +x from a vertical line (x = const), toward +y from a horizontal one; negative: '
               'toward -x / -y. Nothing is launched upstream, at every solved wavelength.',
          tip='Direction the mode travels from the line.\n\n'
              '- Positive (+x or +y): toward +x for a vertical line, toward +y for a horizontal one.\n'
              '- Negative (-x or -y): toward -x or -y.\n\n'
              'Nothing is launched the other way, at any solved wavelength.',
          choice_labels=MODE_DIRECTION_CHOICE_LABELS),
    Param('power', 'float', 1.0, label='Mode power', unit='W/m', min=0.0, min_exclusive=True,
          help='Time-averaged power of the launched mode per unit z length, 1/2 Re of the integral of (E x H*).n over '
               'the line; mode amplitude sqrt(power) e^{j phase}. Fields scale with sqrt(power); 1 W/m is usual (peak '
               '|E| ~1e4 V/m in a 0.3 um silicon slab).',
          tip=MODE_POWER_TIP),
    _PHASE_PARAM,
)
_MODE_MONITOR_PARAMS = (
    mode_number_param('Picked at the first solved wavelength (frequency_domain order: its wavelengths list, else '
                      'wavelength_start to wavelength_stop) and followed by overlap with the previous solve, so it '
                      'keeps its identity where two modes cross in n_eff.',
                      tip=MODE_NUMBER_TIP_START.format(verb='measure') + '\n\nThe mode is picked at the first solved '
                          'wavelength and followed across the others, so it stays the same mode where two modes swap '
                          'order.'),
)
_FAR_FIELD_PARAMS = (N_ANGLES_PARAM,)

# ---- descriptions ----
_PROBE_RESULT_TEXT = ("Result 'xy' versus vacuum wavelength (m, ascending; one point per solved wavelength, a re-solved "
                      'one keeps its latest): series abs = |phasor| (unit of the component) and phase (rad; field = '
                      'abs cos(omega t + phase)); y_label the component label; extras frequency (Hz), wavelength (m), '
                      'component and unit.')
_FLUX_RESULT_TEXT = ("Result 'flux_spectrum' in W/m (power per unit z length over the line cells, absolute: a line "
                     'source of amplitude A spanning a period L in vacuum sends A^2 L / (2 eta0) to each side), one '
                     'entry per solved wavelength, sorted. The discrete flux is conserved exactly through lossless '
                     'cells; transmission = device power / power of a reference run with the same source.')
_MONITOR_INTERFACE_TEXT = ('Keep its measuring face (hi face of the line cells) off a mode_source interface face (the '
                           'line one cell upstream of a positive source, or the line of a negative one) and entirely '
                           'inside or outside a plane_wave_tfsf box: the field jumps there between total and scattered '
                           'field (validation warns).')
_TFSF_TEXT = ('the incident wave is the discrete plane wave of the FDFD stencil in the background medium (complex eps '
              'and mu_r of the outline cell just before the lo-x edge; uniform x cells: (2 / dx) sin(k dx / 2) = n '
              'k0; graded x cells: marched on the solved stencil, a grading at the lo-x edge adds a small backward '
              'part), so a box in a uniform medium is exactly transparent. The box edges must lie in one uniform '
              'medium (else a warning: incident light leaks out), one scattered-field cell inside the domain on every '
              'side, except: along a periodic y axis it may cover every y cell (a grating or layer stack may cross '
              "it), and at a y mirror face of the wave's parity (ez: symmetric, hz: antisymmetric) it may reach the "
              'plane (the other parity, or an x mirror face, is an error). With bloch_ky != 0 the wave is oblique, '
              'exp(-j (k_x x + ky y)), phase referred to y = 0, (2 / dx)^2 sin^2(k_x dx / 2) = (n k0)^2 - (2 / dy)^2 '
              'sin^2(ky dy / 2) (refused when not positive); it needs uniform y cells. Enclose the scatterers, put a '
              'far_field box outside it; flux and mode monitors must not straddle its outline. The box covers '
              f'{BOX_CELLS}.')
_MODE_SOURCE_TEXT = ('One-sided total-field / scattered-field line: total field from the line cells downstream, only '
                     'the scattered field (what comes back) upstream; the interface is the lo face of the line cells '
                     'for positive, their hi face for negative. The launch is exact and one-sided at every solved '
                     'wavelength of a straight guide. Mode monitors normalise by it when it is the only enabled mode '
                     'source of the physics.')
_MODE_SOLVE_TEXT = ('The 1D line eigenmodes use the FDFD discretisation (Yee widths, PML stretch, complex eps and mu of '
                    'the line cells with poles, sigma_e, sigma_m), re-solved at every solved wavelength and after any '
                    'material change (thermo-optic coupling, state-dependent permittivity). Put the line across a '
                    'straight guide with the same cross-section a few cells on either side, on a locally uniform mesh '
                    '(validation warns when the cell widths along the normal change by more than 1% around the line '
                    'and when its face lies inside a CPML layer, i.e. the line is in the outermost window cell).')
_MODE_MONITOR_FIELDS = {
    'ez': 'E_z in the line cells and H_y (vertical line) / H_x (horizontal) from the curl of E_z on their hi face',
    'hz': "H_z in the line cells and E_y (vertical line) / E_x (horizontal) from Ampere's law on their hi face",
}
_MODE_SOURCE_RESULT_TEXT = ("Result 'table' of the latest port solve (frequency_domain: its last wavelength), "
                            f'{MODE_TABLE_ROWS}; extras wavelength (m), power (W/m), direction, mode_number and phase '
                            '(deg).')
_MODE_MONITOR_RESULT_TEXT = ("Result 'xy' versus vacuum wavelength (m, ascending): power_positive / power_negative = "
                             '|a|^2 in W/m (modes normalised to 1 W/m); with exactly one enabled mode_source in the '
                             'physics also fraction_positive / fraction_negative = |a / (sqrt(P) e^{j phase})|^2 '
                             '(source power P and phase; |S|^2: transmission downstream of the source, reflection '
                             'upstream) and phase_positive / phase_negative (rad), the phase of that S-parameter '
                             "relative to the source's own line, not de-embedded (each mode profile phased with its "
                             'largest sample real positive, the lowest-coordinate one of mirror-equal peaks); extras '
                             'frequency, wavelength, n_eff, n_eff_imag (per wavelength), mode_number and normalized '
                             '(True with the fractions).')
_FAR_FIELD_TEXT = ('one row per solved wavelength (ascending). Assumes vacuum outside the box: enclose every '
                   'scatterer; with a plane_wave_tfsf keep the box outside it, in the scattered field: x0 / y0 edges '
                   'at least two cells before those of the TFSF box, x1 / y1 edges in or beyond the first cell past it '
                   '(samples average the faces on both sides of their cell; validation warns otherwise, and for an '
                   'edge sampling across a mode_source interface face). The contour runs through the centres of the '
                   'first box cells and of the cells one beyond x1 / y1 (keep the box one cell off the x_hi / y_hi '
                   f'edges); graded meshes work (actual centres, trapezoid weights). The box covers {BOX_CELLS}.')
# feature descriptions alike for both polarisations (one text, so em_fdfd_nonlinear prints it once)
_TFSF_DESCRIPTION = ('Plane wave along +x on the outline of a total-field box (total field in the box cells, scattered '
                     'field outside), entering only the right-hand side; phase referred to the centres of the first '
                     f'(lo-x) box column; {_TFSF_TEXT}')
_FAR_FIELD_DESCRIPTION = ('Near-to-far-field transform of the phasors on a rectangle outline (the out-of-plane field '
                          "and the in-plane one, from the curl of E_z / Ampere's law of H_z, averaged onto the centres "
                          'at the em_fdtd_* far_field sample points, so normalised patterns compare directly): power '
                          'per unit length and angle rho |E|^2 / (2 eta0) in W/(m rad) of the far-field |E| (H_z: '
                          '|eta0 H_z|; angle integral = power through a closed flux box) versus angle '
                          f'counter-clockwise from +x, {_FAR_FIELD_TEXT}')


def _mode_source_description(polarization):
    """Description of the mode_source feature in `polarization`."""
    return ('Port source launching one guided mode of the cross-section under an axis-aligned line in one direction. '
            f"{_MODE_SOURCE_TEXT} {_MODE_SOLVE_TEXT} {mode_end_text(polarization, 'fdfd')} The line covers "
            f'{SPAN_CELLS}. {_MODE_SOURCE_RESULT_TEXT}')


def _mode_monitor_description(polarization):
    """Description of the mode_monitor feature in `polarization`."""
    return ('Port monitor splitting the field on an axis-aligned line into the forward (positive: toward +x / +y) and '
            'backward (negative) amplitudes a of one guided mode at every solved wavelength: '
            f'{_MODE_MONITOR_FIELDS[polarization]} are projected on the mode solved there, so each direction is '
            f"measured separately, unlike a flux_monitor. {_MODE_SOLVE_TEXT} {mode_end_text(polarization, 'fdfd')} "
            f'{_MONITOR_INTERFACE_TEXT} The line covers {SPAN_CELLS}. {_MODE_MONITOR_RESULT_TEXT}')


_DESCRIPTIONS = {
    'ez': {
        'point_source': 'z-directed filament current phasor I e^{j phase} in the cell containing the point (a '
                        'coordinate exactly on a cell edge goes to the upper cell), radiating a mesh-independent '
                        'cylindrical wave.',
        'line_source': EZ_LINE_SOURCE_DESCRIPTION,
        'gaussian_beam': EZ_GAUSSIAN_BEAM_DESCRIPTION,
        'plane_wave_tfsf': _TFSF_DESCRIPTION,
        'mode_source': _mode_source_description('ez'),
        'pec_domain': 'Perfect electric conductor: E_z = 0 in every cell whose centre lies in the selected domains, '
                      'staircased; with solver_defaults.conformal_boundaries the H links from free cells put E_z = 0 '
                      'at the drawn wall (exact wall distance t d, t >= 0.02).',
        'flux_monitor': 'Time-averaged Poynting flux through an axis-aligned line per solved wavelength: S_x = -1/2 '
                        'Re(E_z H_y*) (vertical line), S_y = 1/2 Re(E_z H_x*) (horizontal), E_z of the line cells, H '
                        'from the curl of E_z on their hi face; > 0 toward +x / +y. '
                        f'{_MONITOR_INTERFACE_TEXT} {_FLUX_RESULT_TEXT} The line covers {SPAN_CELLS}.',
        'mode_monitor': _mode_monitor_description('ez'),
        'far_field': _FAR_FIELD_DESCRIPTION,
    },
    'hz': {
        'point_source': 'Current-calibrated, mesh-independent point excitation phasor (see excitation) in the cell '
                        'containing the point (a coordinate exactly on a cell edge goes to the upper cell).',
        'line_source': 'Electric sheet current on the hi face of the line cells (+x face of a vertical line, +y face '
                       'of a horizontal one) sending amplitude (tangential E) to each side in the local medium: a '
                       'vertical line (x = const) drives E_y along +x and -x, a horizontal one E_x along +y and -y. '
                       f'{PLANE_WAVE_SPAN} The line covers {SPAN_CELLS}.',
        'gaussian_beam': 'Line source with the profile exp(-((s - center) / waist)^2) along the line: a beam of peak '
                         'tangential E = amplitude to each side in the local medium (vertical line: E_y, along +x and '
                         '-x; horizontal: E_x, along +y and -y), waist on the line, sheet on the hi face of the line '
                         f'cells. The line covers {SPAN_CELLS}.',
        'plane_wave_tfsf': _TFSF_DESCRIPTION,
        'mode_source': _mode_source_description('hz'),
        'pec_domain': 'Perfect electric conductor: tangential E = 0 on the four faces of each cell whose centre lies '
                      'in the selected domains (H_z inside is 0), staircased. With solver_defaults.conformal_boundaries '
                      '(Dey-Mittra) only faces wholly inside the drawn metal are PEC; cut cells use their free face '
                      'lengths and free area (>= 0.02 of the cell), keeping the operator reciprocal; mode ports, '
                      'sources and eps_xy couplings keep the cell mask.',
        'flux_monitor': 'Time-averaged Poynting flux through an axis-aligned line per solved wavelength: S_x = 1/2 '
                        'Re(E_y H_z*) (vertical line), S_y = -1/2 Re(E_x H_z*) (horizontal), H_z of the line cells, E '
                        "on their hi face from Ampere's law (curl of H_z plus a source current on that face); > 0 "
                        'toward +x / +y. Keep it off the line cells of a line source or gaussian beam: their sheet sits '
                        'on that face, where the flux jumps, and the monitor reads the line-cell side (minus the power '
                        f'sent toward -x / -y). {_MONITOR_INTERFACE_TEXT} {_FLUX_RESULT_TEXT} The line covers '
                        f'{SPAN_CELLS}.',
        'mode_monitor': _mode_monitor_description('hz'),
        'far_field': _FAR_FIELD_DESCRIPTION,
    },
}
# GUI summaries (tree / menu tooltip, settings panel): shared, then the per-polarisation ones
_SHARED_SUMMARIES = {
    'line_source': 'Sends a wave to both sides of a straight line at each solved wavelength. Across the whole domain '
                   'it gives a plane wave; a shorter line diffracts like a slit.',
    'gaussian_beam': 'Sends a Gaussian beam to both sides of a straight line at each solved wavelength, narrowest on '
                     'the line. Use it for focused light or light of finite width.',
    'plane_wave_tfsf': 'Sends a plane wave along +x through a box at each solved wavelength: inside the full field, '
                       'outside only the light scattered by objects in the box. Use it for scattering studies; it '
                       'works on uniform and graded meshes.',
    'mode_source': 'Launches one guided mode of a waveguide in one direction from a line across the guide, recomputed '
                   'at each solved wavelength. Pair it with mode monitors for transmission and reflection of waveguide '
                   'devices.',
    'pec_domain': 'Turns the selected domains into a perfect metal that reflects all light, following the mesh cells. '
                  'Use it for idealised metal walls, mirrors and cavities.',
    'field_probe': 'Records the amplitude and phase of one field component at a point, one value per solved '
                   'wavelength. A stationary study gives a single value.',
    'mode_monitor': 'Measures the power in one guided mode crossing a line, in each direction, at each solved '
                    'wavelength. With a single mode source it also gives transmission and reflection into that mode.',
    'far_field': 'Computes the power radiated in each direction, one pattern per solved wavelength, from the fields on '
                 'a rectangle around every object. Keep the rectangle in vacuum and outside any plane wave box.',
}
_SUMMARIES = {
    'ez': {
        **_SHARED_SUMMARIES,
        'point_source': 'Radiates a cylindrical wave from one point at each solved wavelength, like a thin wire along '
                        'z carrying an alternating current. Use it to excite resonators or to test a structure.',
        'flux_monitor': 'Measures the power crossing a straight line at each solved wavelength, positive toward +x or '
                        '+y. For transmission, divide by the same monitor in a reference run without the device.',
    },
    'hz': {
        **_SHARED_SUMMARIES,
        'point_source': 'Radiates from one point at each solved wavelength, either as a magnetic line current along '
                        'z, the same in every direction, or as an in-plane electric dipole. Use it to excite '
                        'resonators or particles.',
        'flux_monitor': 'Measures the power crossing a straight line at each solved wavelength, positive toward +x or '
                        '+y; keep it off the line of a line source or Gaussian beam. For transmission, divide by a '
                        'reference run without the device.',
    },
}
_PROBE_DESCRIPTION = ('Phasor of one field component in the cell containing the point (a coordinate exactly on a cell '
                      'edge goes to the upper cell) per solved wavelength. Face components (hx / hy, ex / ey) come '
                      'from neighbouring cells: keep them off a plane_wave_tfsf outline and a mode_source interface '
                      f'face (validation warns). {_PROBE_RESULT_TEXT}')
_COMMON_DESCRIPTION = (
    'One sparse direct solve of the discrete Helmholtz equation per vacuum wavelength, exp(+j omega t): fields are '
    'complex phasors F (real field Re(F e^{j omega t})), every source a phasor (phase setting) with the calibrated, '
    'mesh-independent amplitude of the matching FDTD source. absorbed_power (state q_abs) = 1/2 omega eps0 (-Im eps) '
    '|E|^2 in W/m^3 (magnetic loss not counted), what em_heat_joule hands to heat. Solver roles (see solvers): '
    'linear = sparse_direct (scipy_splu: host SuperLU, ~1e6 unknowns nx * ny incl. PML; cusolver_qr ~1e5; cudss: '
    'cuDSS LDL^T, ~20 x SuperLU at 1e5 - 5e5 unknowns (cudss_mixed solves slower); '
    f'{HOST_DIRECT_NAMES}: the same factor on host cores; a Bloch phase, eps_xy or gyration_z makes it a general LU) '
    'or line_block (elongated grids), one factorisation per wavelength and material state, reused for every source; '
    'eigen = arpack_shift_invert (factor: SuperLU, cudss / cudss_mixed, '
    f'{HOST_DIRECT_NAMES} or the line_block one when that is the linear solver; a rejected solve redoes it on '
    'SuperLU). Studies: stationary at the wavelength setting (with heat, em_heat_joule and heat_em_thermo_optic '
    'iterate to a fixed point; plain Picard oscillates once the self-heating shift exceeds about one linewidth: set '
    "the study relaxation < 1); frequency_domain a wavelength list (with em_heat_joule each wavelength's fixed point, "
    'continued from the previous), one snapshot each; eigenfrequency the source-free resonances nearest the '
    'wavelength setting, one snapshot each plus resonance_table; optomechanics likewise when a solid_mechanics '
    'optomechanical_coupling names it (else a stationary background; no coupling or material may read it); '
    'optimization maximises the adjoint_objective sum over the design_region densities (L-BFGS-B); time_dependent '
    '(beside a time-stepped physics such as elastic_waves, never em_fdtd_*) re-solves on the frozen state every k '
    'steps (k = its every) when an input changed; monitors keep the last solve. Periodic faces take bloch_kx / '
    'bloch_ky. On a triangles mesh: Lagrange finite elements (see discretization; materials and state fields per '
    'element, the same output names on a display raster).'
)
_MODULUS_OUTPUTS = ('e_abs / e_intensity / h_abs = |E|, |E|^2, |H| at the cell centres (in-plane face phasors averaged '
                    'there first): peak amplitudes, where em_fdtd_* give instantaneous values.')
_PHYSICS_DESCRIPTIONS = {
    'ez': ('2D finite-difference frequency-domain (FDFD) solver for the E_z polarisation: E_z out of plane in V/m, '
           'H_x and H_y in plane in A/m, x to the right and y up, on the em_fdtd_ez Yee grid (E_z at cell centres, '
           f'H_x / H_y on the +y / +x cell faces). {_MODULUS_OUTPUTS} {_COMMON_DESCRIPTION}'),
    'hz': ('2D finite-difference frequency-domain (FDFD) solver for the H_z polarisation: H_z out of plane in A/m, '
           'E_x and E_y in plane in V/m, x to the right and y up, on the em_fdtd_hz Yee grid (H_z at cell centres, '
           f'E_x on the +y and E_y on the +x cell faces; face materials: see eps_r). {_MODULUS_OUTPUTS} '
           f'{_COMMON_DESCRIPTION} With an em_fdfd_ez physics in the same model (e.g. one per polarisation) it needs '
           'another instance name (e.g. em_hz) and equal bc_* / pml_* settings (one shared boundary configuration).'),
}
_PHYSICS_LABELS = {
    'ez': 'Electromagnetic Waves, frequency domain (2D, electric field out of plane)',
    'hz': 'Electromagnetic Waves, frequency domain (2D, magnetic field out of plane)',
}
_PHYSICS_SUMMARIES = {
    'ez': ('Light in 2D, solved directly in its steady state, one wavelength at a time. The electric field points out '
           'of the plane: suits narrow resonances, wavelength sweeps and coupled heating of dielectric structures.'),
    'hz': ('Light in 2D, solved directly in its steady state, one wavelength at a time. The magnetic field points out '
           'of the plane and the electric field lies in it: choose it for metals, plasmonics and light polarised in '
           'the plane.'),
}


# ---- build ----
def _make_create(polarization, background_targets):
    """create callback: FDFDPhysics at its first solved wavelength (finalize selects the port modes there) with its
    dispersive poles and their background permittivity written into background_targets; never finalizes. A triangles
    mesh: FEMFDFDPhysics (walls from the settings: the TriangleDomain carries no boundary config)."""
    def create(ctx, physics_spec, settings_values):
        solvers = ctx.solvers(physics_spec.name)
        if getattr(ctx.domain, 'grid_kind', 'tensor') == 'triangles':
            solver = FEMFDFDPhysics(ctx.domain, polarization, fdfd_domain_requirements(settings_values).boundary_config,
                                    wavelength=first_solved_wavelength(ctx, settings_values),
                                    linear_solver=solvers['linear'], eigen_solver=solvers.get('eigen'),
                                    order=DISCRETIZATION_ORDERS[settings_values['discretization']])
        else:
            solver = FDFDPhysics(ctx.domain, polarization=polarization, boundary_config=ctx.domain.boundary_config,
                                 wavelength=first_solved_wavelength(ctx, settings_values),
                                 linear_solver=solvers['linear'], eigen_solver=solvers.get('eigen'))
        write_dispersive_materials(ctx, physics_spec, solver, background_targets)
        return solver
    return create


# ---- eigenfrequency study (A6): resonance_table feature ----
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency solve, nearest the target (the wavelength setting) first. Result 'table', "
    'one row per resonance: resonance_number (rank by |omega - omega_target|), wavelength = 2 pi C0 / Re(omega) (m), '
    'frequency = Re(omega) / (2 pi), frequency_imag = Im(omega) / (2 pi) (Hz; > 0 decays), q_factor = Re(omega) / (2 '
    'Im(omega)) (inf when |Im omega| < 1e-12 Re omega; negative = growing, usually a PML artefact), pml_fraction '
    '(share of the out-of-plane |field|^2 dA in PML cells: above 0.5 dropped; above ~1e-2 widen the cladding); extras '
    'target_wavelength (m), selected_resonance. Accurate only near the target. Q versus a parameter: a derived value '
    "with source '<physics>.<this feature>', expression q_factor, operation at, at = 0. An optional domain selection "
    f'adds {REGION_COLUMN[0]} (share of the same weights in the selected cells; triangles: dofs of the selected '
    'elements, outline dofs counting fully) and drops resonances below min_region_fraction like PML modes. One '
    'resonance_table with a selection per physics.')
MIN_REGION_FRACTION_PARAM = Param(   # shared with em_fdfd_3d's resonance_table
    'min_region_fraction', 'float', 0.0, label='Minimum region fraction', unit='1', min=0.0, max=1.0,
    help=f'Resonances with {REGION_COLUMN[0]} below this are dropped (counted in the study log; the search widens as '
         'for PML modes). 0 keeps all; > 0 needs a domain selection. Typical 0.5: keep cavity modes, drop those of an '
         'attached guide.',
    tip='Keeps only the resonances that have at least this share of their energy inside the selected region, for '
        'example 0.5 to keep the modes of a cavity and drop those of a waveguide attached to it.\n\n'
        'Needs a selection; 0 keeps every resonance and only lists the share.')


def _apply_resonance_table(ctx, solver, feature_spec, values, selection):
    """ResonanceTable marker; with a selection its ResonanceRegion (host cell / element mask) goes to the solver."""
    if not selection.entities:
        return ResonanceTable()
    region = ResonanceRegion(cells=np.asarray(to_host(ctx.domain_mask(selection.entities)), dtype=bool),
                             min_fraction=float(values['min_region_fraction']))
    solver.set_resonance_region(region)
    return ResonanceTable(region)


def validate_resonance_table(vctx, physics_spec, feature_spec, values):
    """Warning unless the study eigen-solves the physics (eigenfrequency, optomechanics optical): no solve to show;
    errors for min_region_fraction > 0 without a selection and for a second resonance_table with a selection."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    min_region_fraction = values.get('min_region_fraction') or 0.0
    if min_region_fraction > 0.0 and not feature_spec.selection:
        issues.append(Issue('error', path, f'min_region_fraction {min_region_fraction:g} needs a region: select the '
                                           'domain objects of the cavity (or set it to 0)'))
    if feature_spec.selection:   # the second table with a selection gets the error, the first keeps its region
        preceding = physics_spec.features[:next((index for index, feature in enumerate(physics_spec.features)
                                                  if feature is feature_spec), 0)]
        earlier = next((feature.name for feature in preceding
                        if feature.enabled and feature.type == feature_spec.type and feature.selection), None)
        if earlier is not None:
            issues.append(Issue('error', path, f"resonance_table '{earlier}' of '{physics_spec.name}' has a region "
                                               'too: the eigen-solve takes one region per physics (keep the selection '
                                               'on one table)'))
    return issues + resonance_table_study_issues(vctx, physics_spec, feature_spec)


def resonance_region_memory_estimate(values, n_selected_cells, mesh):
    """FeatureType.memory_estimate of a resonance_table: with a selection the host region mask (one bool per cell /
    element) and its row flags during the eigen-solve, the fresh device domain_mask while it applies."""
    if not n_selected_cells:
        return MemoryEstimate()
    n_cells = int(mesh.n_cells)
    return MemoryEstimate.from_items(host_items=(('resonance region mask and row flags', 2 * BOOL_BYTES * n_cells),),
                                     build_extra=BOOL_BYTES * n_cells)


# ---- validation (CPU only) ----
def _per_mesh_kind(tensor_validate, triangle_validate):
    """Feature validate running triangle_validate on a triangles mesh, else tensor_validate (Yee cell / wall rows)."""
    def validate(vctx, physics_spec, feature_spec, values):
        chosen = triangle_validate if mesh_kind(vctx) == 'triangles' else tensor_validate
        return chosen(vctx, physics_spec, feature_spec, values)
    return validate


def _validate_triangle_hz_point_source(vctx, physics_spec, feature_spec, values):
    """Triangles: point inside the domain and electric_dipole_issues (no Yee wall rows there)."""
    return (validate_points_inside(vctx, physics_spec, feature_spec, values)
            + electric_dipole_issues(physics_spec, feature_spec, values))


def _validate_field_probe(vctx, physics_spec, feature_spec, values):
    """Point inside the domain; warning for a face component read across a TFSF / mode-source interface (tensor)."""
    issues = validate_points_inside(vctx, physics_spec, feature_spec, values)
    if mesh_kind(vctx) == 'triangles':
        return issues
    return issues + harmonic_probe_interface_issues(vctx, physics_spec, feature_spec, values)


_TRIANGLE_REFUSED_STUDIES = (OPTIMIZATION_STUDY, OPTOMECHANICS_STUDY, 'thermal_noise')


def _triangle_issues(vctx, physics_spec, settings_values, path):
    """On a triangles mesh: errors for the optimization / optomechanics / thermal_noise studies (they read the Yee
    fields or need the adjoint) and periodic / Bloch faces (no periodic mesh); info for a cusolver_qr linear node (host
    SuperLU here); the FEM size rule of the host factor on the estimated unknowns (none for a cudss eigen-solve or a
    cudss linear back end; MUMPS / PARDISO keep SuperLU's). The Yee-only features are refused by their grids."""
    if mesh_kind(vctx) != 'triangles':
        return []
    issues = []
    type_name = physics_spec.type
    if vctx.study_type in _TRIANGLE_REFUSED_STUDIES:
        issues.append(Issue('error', path, f'{type_name} on a triangles mesh runs stationary, frequency_domain, '
                                           f'time_dependent and {EIGENFREQUENCY_STUDY} studies; a {vctx.study_type} '
                                           'study needs a tensor mesh'))
    if 'periodic' in (settings_values['bc_x_lo'], settings_values['bc_y_lo']):
        issues.append(Issue('error', path, f'{type_name} on a triangles mesh: {PERIODIC_TEXT}'))
    solvers = vctx.solvers(physics_spec) or {}
    resolved = solvers.get('linear')
    if resolved is not None and resolved.values.get('back_end') == 'cusolver_qr':
        issues.append(Issue('info', path, f"linear solver '{resolved.node}': on a triangles mesh {type_name} factorises "
                                          'the FEM operator with host SuperLU whatever the back end'))
    mesh = vctx.mesh
    order = DISCRETIZATION_ORDERS[settings_values['discretization']]
    factorization = (eigen_factorization(solvers) if eigen_solved(vctx.spec, physics_spec.name)
                     else 'cudss' if linear_solver_cudss(solvers)
                     else host_direct_back_end(solvers) or 'superlu')   # MUMPS / PARDISO: SuperLU's rule, own texts
    issue = fem_fdfd_size_issue(scalar_unknowns(mesh.n_nodes, mesh.n_edges, order), order, factorization)
    if issue is not None:
        issues.append(Issue(issue[0], path, f'estimated {issue[1]}'))
    return issues


def _validate_flux_monitor(vctx, physics_spec, feature_spec, values):
    """Line axis-aligned inside the domain; warning when it measures across a TFSF / mode-source interface."""
    return (validate_line_feature(vctx, physics_spec, feature_spec, values)
            + monitor_line_interface_issues(vctx, physics_spec, feature_spec, time_domain=False))


def direct_solver_issues(vctx, physics_spec, settings_values, path):
    """The resolved 'linear' sparse_direct solver: on Automatic its scipy rule (a node reports it itself); the back
    end's size limit (solvers.sparse_direct.size_issue, which the solver enforces too): error beyond, warning near."""
    solvers = vctx.solvers(physics_spec)
    resolved = (solvers or {}).get('linear')
    if resolved is None or resolved.solver_type.name != 'sparse_direct':
        return []   # unresolvable: reported by check_solvers
    issues = []
    backend = solver_backend(vctx.spec)
    on_cudss = resolved.values.get('back_end') in CUDSS_BACK_ENDS   # Automatic may resolve to cuDSS (auto_back_ends)
    if resolved.node is None and not on_cudss and not scipy_available():   # find_spec only: nothing imported
        if backend == 'cpu':
            issues.append(Issue('error', path, 'scipy is not installed: the CPU backend solves the FDFD system with '
                                               "scipy's SuperLU ('scipy_splu'); pip install scipy"))
        else:
            issues.append(Issue('info', path, "scipy is not installed, so the Automatic linear solver uses the GPU "
                                              "sparse QR back end 'cusolver_qr' (one factorisation per solve, up to "
                                              "~1e5 unknowns); pip install scipy for 'scipy_splu' (host SuperLU, "
                                              'factorisation reused, ~1e6 unknowns)'))
    mesh = vctx.mesh
    if mesh is None or getattr(mesh, 'kind', 'tensor') != 'tensor':   # triangles: _triangle_issues' FEM size rule
        return issues
    back_end = (expected_back_end(resolved.values['back_end'], backend) if 'back_end' in resolved.values
                else resolved_back_end(solvers))
    issue = size_issue(back_end, int(mesh.nx), int(mesh.ny),
                       settings_values['bc_x_lo'] == 'periodic', settings_values['bc_y_lo'] == 'periodic')
    if issue is not None:
        issues.append(Issue(issue[0], path, issue[1]))
    return issues


def _line_block_solver_issues(vctx, physics_spec, settings_values, path):
    """A line_block 'linear' node (physics/em/fdfd_line_block): error with two periodic axes (a wrap along the line axis is
    not block-tridiagonal); info that adjoint (transposed) solves with eps_xy or a Bloch phase use a host SuperLU."""
    resolved = (vctx.solvers(physics_spec) or {}).get('linear')
    if resolved is None or resolved.solver_type.name != 'line_block':
        return []
    issues = []
    if settings_values['bc_x_lo'] == 'periodic' and settings_values['bc_y_lo'] == 'periodic':
        issues.append(Issue('error', path, f"{BOTH_PERIODIC_TEXT}: assign the sparse_direct solver (Automatic) to "
                                           'the linear role instead'))
    adjoint = vctx.study_type == OPTIMIZATION_STUDY or any(
        feature.enabled and feature.type == 'adjoint_objective' for feature in physics_spec.features)
    if adjoint and (settings_values.get('bloch_kx', 0.0) or settings_values.get('bloch_ky', 0.0)):
        issues.append(Issue('info', path, 'with a Bloch phase the operator is not complex symmetric, so the adjoint '
                                          'solves of the gradient cannot reuse the line-block factor: they use a host '
                                          'sparse direct factor (SuperLU) of each operator, and every design '
                                          'iteration also factorises it once (likewise with a material eps_xy or '
                                          'gyration_z)'))
    return issues


def _time_dependent_issues(vctx, physics_spec, path):
    """time_dependent study only: error beside a time-stepped EM physics (FDTD: its femtosecond step would set the
    shared step of the whole run); info on the stroboscopic re-solves."""
    if vctx.study_type != TIME_DEPENDENT_STUDY:
        return []
    issues = []
    for other, other_type in known_physics(vctx.spec, vctx.registry):   # unknown types: reported at that physics
        if other_type.family == EM_FAMILY and other_type.steps_in_time is not None:
            issues.append(Issue('error', path,
                                f"a time_dependent study with '{physics_spec.name}' (frequency domain) cannot also step "
                                f"the time-domain EM physics '{other.name}' ({other_type.name}): its femtosecond step "
                                'would set the step of the whole run (e.g. ~1e5 times the acoustic one); use one EM '
                                'formulation'))
    issues.append(Issue('info', path,
                        f"time_dependent study: '{physics_spec.name}' re-solves at its wavelength setting on the frozen "
                        f'state of the time-stepped physics every {physics_spec.every} step(s) of the run (its every), '
                        'only when a permittivity or source input changed (stroboscopic snapshots; a solve is far '
                        'slower than a time step, so keep every large)'))
    return issues


def _off_diagonal_issues(vctx, physics_spec, path):
    """H_z, optimization study: info when eps_xy reaches the physics (a selected material sets it or a coupling writes
    eps_xy_update): the design writes eps_xx / eps_yy and the gradient holds eps_xy fixed."""
    if vctx.study_type != OPTIMIZATION_STUDY:
        return []
    from ...materials import effective_material_data   # lazy: materials imports physics.em
    sources = [f"coupling '{coupling.name}'" for coupling in vctx.spec.couplings
               if coupling.enabled and coupling.target == physics_spec.name
               and OFF_DIAGONAL_SLOT in coupling_writes(vctx, coupling)]
    for material in vctx.spec.materials:
        if not material.selection:
            continue
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            continue
        if evaluates_to(properties.get(OFF_DIAGONAL_ARRAY), vctx.parameters, lambda value: value != 0.0):
            sources.append(f"material '{material.name}'")
    if not sources:
        return []
    return [Issue('info', path, f"{', '.join(sources)} give(s) '{physics_spec.name}' an off-diagonal eps_xy: the "
                                'design regions write eps_xx = eps_yy and the adjoint gradient holds eps_xy fixed (a '
                                'design cell keeps its material eps_xy, so eps_min * eps_min must exceed eps_xy^2 '
                                'there); the eps_xy_sensitivity field gives dJ / d eps_xy')]


def _gyration_issues(vctx, physics_spec, path):
    """H_z with a selected material setting gyration_z: errors for mirror faces (a mirror image reverses the
    magnetisation, so parity sectors too) and a triangles mesh; info in an optimization study (g held fixed)."""
    if not selected_material_sets(vctx, (GYRATION_Z,)):
        return []
    issues = []
    faces = sorted(mirror_faces(vctx.spec))
    if faces:
        issues.append(Issue('error', path, f"a material sets gyration_z (magneto-optic) while {', '.join(faces)} "
                                           f"{'is a mirror face' if len(faces) == 1 else 'are mirror faces'}: the "
                                           'mirror image of a magnetisation along z is the reversed magnetisation, so '
                                           'the mirrored model (and its parity sectors) is not the drawn one; model '
                                           'the whole structure with cpml / periodic / wall faces'))
    if mesh_kind(vctx) == 'triangles':
        issues.append(Issue('error', path, 'gyration_z (a non-reciprocal magneto-optic medium) runs on tensor meshes '
                                           'only: the triangles-mesh pencil is solved as complex symmetric; use a '
                                           'tensor mesh'))
    if vctx.study_type == OPTIMIZATION_STUDY:
        issues.append(Issue('info', path, 'gyration_z: the adjoint solves the transposed (reversed-magnetisation) '
                                          'system and the gradient holds gyration_z fixed'))
    return issues


def _make_validate_physics(slot_arrays):
    """validate(vctx, physics_spec, settings_values) of an FDFD type: the shared EM checks (periodic pairing,
    materials, slot writers, resolution and skin depth at every solved wavelength) plus the study override, the
    time_dependent rules, the linear solver, (H_z) the eps_xy optimization rule and the gyration_z rules, and the
    triangles-mesh rules."""
    shared_validate = make_validate_physics(slot_arrays, time_domain=False, solved_wavelengths=all_solved_wavelengths)
    hz = 'eps_xx_update' in slot_arrays

    def validate_physics(vctx, physics_spec, settings_values):
        path = ('physics', physics_spec.name)
        issues = shared_validate(vctx, physics_spec, settings_values)
        issues.extend(unused_wavelength_issues(vctx, settings_values, path))
        issues.extend(_time_dependent_issues(vctx, physics_spec, path))
        if hz:
            issues.extend(_off_diagonal_issues(vctx, physics_spec, path))
            issues.extend(_gyration_issues(vctx, physics_spec, path))
        issues.extend(direct_solver_issues(vctx, physics_spec, settings_values, path))
        issues.extend(_line_block_solver_issues(vctx, physics_spec, settings_values, path))
        issues.extend(fdfd_bloch_issues(vctx, physics_spec, settings_values))
        issues.extend(_triangle_issues(vctx, physics_spec, settings_values, path))
        bloch_axes = bloch_active_axes(settings_values)
        if bloch_axes and mesh_kind(vctx) != 'triangles':   # port lines spanning a Bloch period (fdfd_ports refuses them)
            for feature in physics_spec.features:
                if feature.enabled and is_fdfd_port_feature(feature):
                    issues.extend(bloch_period_line_issues(vctx, physics_spec, feature, bloch_axes))
        return issues
    return validate_physics


# ---- type factory ----
_EPS_XY_SLOT_DOC = ('1', 'Delta eps_xy over the whole grid (PML ring included), added to the material eps_xy at each '
                         'transfer (sum slot: acousto-optic and Pockels add -n0_xx^2 n0_yy^2 Delta(1/n^2)_xy).')
# polarisation -> (renderer, permittivity slot -> MaterialGrid array, material vocabulary, state field names,
# point-source params / apply / validate, line-feature validate); material_schema = EM_MATERIAL_SCHEMAS
_POLARIZATIONS = {
    'ez': (FDFDRendererEz, PERMITTIVITY_SLOTS['ez'], _EZ_MATERIAL_PROPERTIES,
           ('ez_re', 'ez_im', 'q_abs', 'eps_r', 'sigma_e'),
           _EZ_POINT_SOURCE_PARAMS, apply_ez_point_source, validate_ez_point_feature, validate_ez_line_feature),
    'hz': (FDFDRendererHz, PERMITTIVITY_SLOTS['hz'], _HZ_MATERIAL_PROPERTIES,
           ('hz_re', 'hz_im', 'q_abs', 'eps_xx', 'eps_yy', 'sigma_e'),
           _HZ_POINT_SOURCE_PARAMS, apply_hz_point_source, validate_hz_point_source, validate_hz_line_feature),
}


def _fdfd_physics_type(polarization):
    """PhysicsType 'em_fdfd_<polarization>' ('ez' | 'hz')."""
    (renderer_class, slot_arrays, material_properties, state_field_names, point_source_params,
     apply_point_source, validate_point_source, validate_line) = _POLARIZATIONS[polarization]
    type_name = f'em_fdfd_{polarization}'
    descriptions = _DESCRIPTIONS[polarization]
    summaries = _SUMMARIES[polarization]
    hi_face = polarization == 'hz'   # H_z sheet currents act on the hi face of the line cells
    triangle_point_validate = (validate_points_inside if polarization == 'ez'
                               else _validate_triangle_hz_point_source)
    features = (
        FeatureType('point_source', 'Point source', 'source', 'point', params=point_source_params,
                    object_kinds=('point',), max_entities=1, description=descriptions['point_source'],
                    summary=summaries['point_source'], apply=apply_point_source,
                    validate=_per_mesh_kind(validate_point_source, triangle_point_validate), grids=BOTH_GRIDS),
        FeatureType('line_source', 'Line source', 'source', 'line', params=_line_source_params(polarization),
                    object_kinds=('line',), max_entities=1, acts_on_hi_face=hi_face,
                    description=descriptions['line_source'], summary=summaries['line_source'],
                    apply=apply_line_source, validate=validate_line),
        FeatureType('gaussian_beam', 'Gaussian beam', 'source', 'line', params=_gaussian_beam_params(polarization),
                    object_kinds=('line',), max_entities=1, acts_on_hi_face=hi_face,
                    description=descriptions['gaussian_beam'], summary=summaries['gaussian_beam'],
                    apply=apply_gaussian_beam, validate=validate_line),
        FeatureType('plane_wave_tfsf', 'Plane wave box (+x)', 'source', 'domain', params=_TFSF_PARAMS[polarization],
                    object_kinds=('rectangle',), max_entities=1,
                    description=descriptions['plane_wave_tfsf'], summary=summaries['plane_wave_tfsf'],
                    apply=apply_tfsf, validate=validate_harmonic_tfsf),
        FeatureType('mode_source', 'Mode source', 'source', 'line', params=_MODE_SOURCE_PARAMS,
                    object_kinds=('line',), max_entities=1, result_kind='table',
                    description=descriptions['mode_source'], summary=summaries['mode_source'],
                    apply=apply_mode_source, result=object_result(harmonic_mode_source_result),
                    validate=make_validate_mode_source(polarization)),
        FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                    description=descriptions['pec_domain'], summary=summaries['pec_domain'], apply=apply_pec,
                    memory_estimate=pec_memory_estimate(0), grids=BOTH_GRIDS),
        conducting_sheet_feature(polarization),
        effective_index_feature(polarization),
        phase_change_medium_feature(polarization),
        FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS[polarization],
                    object_kinds=('point',), max_entities=1, result_kind='xy', description=_PROBE_DESCRIPTION,
                    summary=summaries['field_probe'], apply=apply_field_probe,
                    result=object_result(harmonic_probe_result), validate=_validate_field_probe, grids=BOTH_GRIDS),
        FeatureType('flux_monitor', 'Flux monitor', 'monitor', 'line', object_kinds=('line',), max_entities=1,
                    result_kind='flux_spectrum', description=descriptions['flux_monitor'],
                    summary=summaries['flux_monitor'], apply=apply_flux_monitor,
                    result=object_result(harmonic_flux_result, with_solver=True), validate=_validate_flux_monitor),
        FeatureType('mode_monitor', 'Mode monitor', 'monitor', 'line', params=_MODE_MONITOR_PARAMS,
                    object_kinds=('line',), max_entities=1, result_kind='xy',
                    description=descriptions['mode_monitor'], summary=summaries['mode_monitor'],
                    apply=apply_mode_monitor, result=object_result(harmonic_mode_monitor_result, with_solver=True),
                    validate=make_validate_mode_monitor(polarization, time_domain=False)),
        FeatureType('far_field', 'Far-field pattern', 'monitor', 'domain', params=_FAR_FIELD_PARAMS,
                    object_kinds=('rectangle',), max_entities=1, result_kind='far_field',
                    description=descriptions['far_field'], summary=summaries['far_field'], apply=apply_far_field,
                    result=make_far_field_result_callback(_FAR_FIELD_PARAMS, harmonic_far_field_result),
                    validate=partial(validate_far_field, time_domain=False)),
        adjoint_objective_feature(polarization),
        design_region_feature(),
        FeatureType('resonance_table', 'Resonance table', 'monitor', 'domain', params=(MIN_REGION_FRACTION_PARAM,),
                    min_entities=0, result_kind='table', description=_RESONANCE_TABLE_DESCRIPTION,
                    summary="Lists the resonances found by an 'Eigenfrequency' study: wavelength, frequency, quality "
                            'factor and how much of each lies in the absorbing layers. With a selection it also '
                            'lists the share inside the selected region and can drop the resonances that live '
                            'elsewhere.',
                    apply=_apply_resonance_table, result=object_result(resonance_table_result, with_solver=True),
                    validate=validate_resonance_table, memory_estimate=resonance_region_memory_estimate,
                    grids=BOTH_GRIDS),
    )
    state_fields = tuple(_Q_ABS_INFO if name == 'q_abs' else renderer_class.field_infos[name]
                         for name in state_field_names)
    source_slots = source_slot_docs(slot_arrays)
    optional_schema = {}
    if polarization == 'hz':   # eps_xy: allocated when a material sets it; the slot adds coupled changes to it
        source_slots[OFF_DIAGONAL_SLOT] = _EPS_XY_SLOT_DOC
        optional_schema = {OFF_DIAGONAL_ARRAY: (0.0, (OFF_DIAGONAL_ARRAY,)), GYRATION_Z: (0.0, (GYRATION_Z,))}
    return PhysicsType(
        name=type_name,
        label=_PHYSICS_LABELS[polarization],
        family=EM_FAMILY,
        group='em',
        spaces_2d=('planar', 'cross_section'),   # cross-section: the k_z = 0 illumination of a thermal_noise study
        default_instance_name='em',
        description=_PHYSICS_DESCRIPTIONS[polarization],
        summary=_PHYSICS_SUMMARIES[polarization],
        settings=_SETTINGS,
        features=features,
        material_properties=material_properties,
        material_schema=EM_MATERIAL_SCHEMAS[polarization],
        optional_material_schema=optional_schema,
        state_fields=state_fields,
        output_fields=tuple(renderer_class.field_infos.values()),
        source_slots=source_slots,
        # time_dependent: stroboscopic re-solves beside a time-stepped physics (acoustics)
        supported_studies=('stationary', FREQUENCY_DOMAIN_STUDY, EIGENFREQUENCY_STUDY, OPTIMIZATION_STUDY,
                           TIME_DEPENDENT_STUDY, OPTOMECHANICS_STUDY),
        default_study={'type': 'stationary'},
        uses_dispersion=True,
        create=_make_create(polarization, tuple(slot_arrays.values())),
        domain_requirements=fdfd_domain_requirements,
        shortest_wavelength=shortest_solved_wavelength,
        validate=_make_validate_physics(slot_arrays),
        memory_estimate=fdfd_memory_estimate(polarization),
        time_estimate=_fdfd_time_estimate,
        solver_roles={'linear': FDFD_LINEAR_ROLE,
                      'eigen': eigen_role((EIGENFREQUENCY_STUDY, OPTOMECHANICS_STUDY), mixed_factorization=True)},
        subpixel_smoothing=True,
        conformal_boundaries=True,   # pec_domain cut cells (conformal_pec FDFD weights; 2D Cartesian)
        backends=('gpu', 'cpu'),   # cpu: numpy / scipy assembly, SuperLU and the adjoint
        grids=BOTH_GRIDS,         # triangles: solvers/fdfd_2d_fem (point sources, probes, pec_domain, resonances)
    )


# run-time estimate: host bytes moved per stored operator entry by one Helmholtz assembly (scipy Yee products, PML
# stretch, CSR conversions; calibrate against the assembly share of a frequency_domain run)
ASSEMBLY_BYTES_PER_ENTRY = 800


def _source_count(context) -> int:
    """Enabled source features of the estimated physics (one solve each per factorisation)."""
    physics_type = context.registry.physics_type(context.physics_spec.type)
    return sum(1 for feature in context.physics_spec.features
               if feature.enabled and physics_type.feature_type(feature.type).category == 'source')


def _driven_points(context) -> tuple:
    """(count, count_kind, limit) of the driven solves: one per step of a time_dependent stride, the static solves x
    the frequency_domain wavelengths, else the static solves."""
    if context.time_dependent:
        return 1.0, 'exact', None
    solves, kind, limit = context.static_solves()
    if context.study.type == FREQUENCY_DOMAIN_STUDY:
        points = context.count('harmonic_points', 1)
        return solves * points, kind, None if limit is None else limit * points
    return solves, kind, limit


def _fdfd_time_estimate(settings_values, mesh, context):
    """Per solved wavelength the Helmholtz assembly (host scipy) and one factorisation of the 'linear' role with one
    solve per enabled source; eigen-solved: one shift-invert of the 'eigen' role (2 n_resonances + 10 eigenpairs);
    optimization: per iteration (bounded) a forward and an adjoint solve; time_dependent: a re-solve per step of its
    stride; triangle meshes: _triangle_time_estimate."""
    if getattr(mesh, 'kind', 'tensor') == 'triangles':
        return _triangle_time_estimate(settings_values, mesh, context)
    unknowns = float(mesh.n_cells)
    nnz = 5.0 * unknowns
    sources = _source_count(context)
    assembly_ops = (te.HostStream(ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200))
    if eigen_solved(context.spec, context.name):
        from .cross_section_modes import cpml_cross_section
        eigenpairs = resonance_eigenpairs(context)
        factorization = eigen_factorization(context.solvers)
        bloch = bool(bloch_active_axes(settings_values))   # memory_estimates.fdfd_memory_estimate's basis rule
        single_basis = search_single_basis(factorization, unknowns, factorization in CUDSS_FACTORIZATIONS, not bloch,
                                           not bloch and cpml_cross_section(settings_values))
        return te.combine((te.TimeEstimate((te.TimeItem('Helmholtz assembly', 'assembly', 1.0, 'exact', assembly_ops),
                                            resonance_filter_time_item(context.backend, unknowns, nnz, eigenpairs))),
                           context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=1.0,
                                               factorizations=1.0, eigenpairs=eigenpairs, single_basis=single_basis)))
    per_step, rhs = context.time_dependent, float(max(sources, 1))
    if context.study.type == OPTIMIZATION_STUDY:
        (points, kind, limit), rhs = optimization_evaluations(context), 2.0 * rhs   # + the adjoint solve
    else:
        points, kind, limit = _driven_points(context)
    assembly = te.TimeItem('Helmholtz assembly', 'assembly', points, kind, assembly_ops, limit=limit,
                           per_step=per_step)
    # the line_block block holds the runtime's block axis (the periodic one, else the cheaper)
    block_axis = fdfd_block_axis(int(mesh.nx), int(mesh.ny), settings_values['bc_x_lo'] == 'periodic',
                                 settings_values['bc_y_lo'] == 'periodic', refuse=False)
    return te.combine((te.TimeEstimate((assembly,)),
                       context.solver_work('linear', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=points * rhs,
                                           solves_kind=kind, factorizations=points, factorizations_kind=kind,
                                           per_step=per_step, device_values=points > 1.0,   # cuDSS: device refreshes
                                           block_size=int(mesh.nx) if block_axis == 0 else int(mesh.ny))))


# ... of a triangles mesh: host bytes per stored entry of one fem_fdfd assembly (quadrature-point stretch tensors, two
# scalar blocks, restriction) and Python calls; uncalibrated (compare the eigen-solve info's / solve_summary's
# assembly_seconds)
FEM_FDFD_ASSEMBLY_BYTES_PER_ENTRY = 2000
FEM_FDFD_ASSEMBLY_CALLS = 200


def _triangle_time_estimate(settings_values, mesh, context):
    """em_fdfd_* on a triangles mesh: per solved wavelength one FEM assembly and one host factorisation of the 'linear'
    role with one solve per enabled source; eigen-solved: one assembly and a shift-invert of the 'eigen' role (host
    SuperLU priced at FEM_FDFD_LU_FILL x the 2D curve, uncalibrated; cuDSS on the 2D cudss curves at that scale)."""
    order = DISCRETIZATION_ORDERS[settings_values['discretization']]
    unknowns = float(scalar_unknowns(int(mesh.n_nodes), int(mesh.n_edges), order))
    nnz = FEM_FDFD_ENTRIES_PER_ROW[order] * unknowns
    assembly_ops = (te.HostStream(FEM_FDFD_ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(FEM_FDFD_ASSEMBLY_CALLS))
    if eigen_solved(context.spec, context.name):
        host = eigen_backend_kwargs(context)
        eigenpairs = resonance_eigenpairs(context)
        return te.combine((te.TimeEstimate((te.TimeItem('FEM Helmholtz assembly', 'assembly', 1.0, 'exact',
                                                        assembly_ops),
                                            resonance_filter_time_item(context.backend, unknowns, 3.0 * nnz,
                                                                       eigenpairs, symmetric=True))),   # A, M and W
                           context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=1.0,
                                               factorizations=1.0, eigenpairs=eigenpairs,
                                               factor_scale=FEM_FDFD_LU_FILL[order],
                                               single_basis=search_single_basis(   # a polished mixed LDL^T
                                                   eigen_factorization(context.solvers), unknowns, not host, True),
                                               **host)))
    per_step = context.time_dependent
    points, kind, limit = _driven_points(context)
    assembly = te.TimeItem('FEM Helmholtz assembly', 'assembly', points, kind, assembly_ops, limit=limit,
                           per_step=per_step)
    return te.combine((te.TimeEstimate((assembly,)),
                       context.solver_work('linear', unknowns=unknowns, nnz=nnz, value_bytes=16,
                                           solves=points * float(max(_source_count(context), 1)), solves_kind=kind,
                                           factorizations=points, factorizations_kind=kind, per_step=per_step,
                                           factor_scale=FEM_FDFD_LU_FILL[order],
                                           **({} if linear_solver_cudss(context.solvers) else {'backend': 'cpu'}))))


EM_FDFD_EZ = _fdfd_physics_type('ez')
EM_FDFD_HZ = _fdfd_physics_type('hz')

"""Model definition of 'em_fdfd_bor' (body-of-revolution FDFD, symmetry plan 2.3: (r, z) half plane, azimuthal order
m) on definition_common, with its template and self-tests; definition.register calls register."""
from dataclasses import replace

from ...core import time_estimate as te
from ...core.constants import FACE_LABELS
from ...core.domain import containing_cell
from ...model.definition_helpers import (AXISYMMETRIC, AXISYMMETRIC_SPACE_SETTINGS, CUDSS_LINEAR_ROLE, GEOMETRY_SETTING,
                                         axisymmetric_issues, eigen_factorization, eigen_role, linear_solver_cudss)
from ...model.schema import FeatureType, Issue, Param, PhysicsType, SolverRole, feature_path
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS
from .body_of_revolution import BOR_FIELD_COMPONENTS, BOR_SOURCE_COMPONENTS, size_issue_for
from .fem_bor import FEM_BOR_ARNOLDI_BASIS_PER_WANTED, FEM_BOR_SIZE_REMEDY
from .fem_modes import DISCRETIZATION_ORDERS
from .bor_ports import bor_mode_monitor_result, bor_mode_source_result
from .definition_common import (BOTH_GRIDS, BOX_CELLS, DEFAULT_WAVELENGTH, EIGENFREQUENCY_STUDY, EM_FAMILY,
                                FREQUENCY_DOMAIN_STUDY, HOST_DIRECT_NAMES, N_ANGLES_PARAM, OPTOMECHANICS_STUDY,
                                POINT_SOURCE_SIGN,
                                SPAN_CELLS, _window_axes, all_solved_wavelengths, apply_component_point_source,
                                apply_far_field, apply_field_probe, apply_flux_monitor, apply_pec, apply_resonance_table,
                                boundary_settings_params, eigen_only_feature_issues, eigen_solved, em_material_params,
                                fem_resonance_time_estimate, fem_size_issues, first_solved_wavelength,
                                host_matrix_back_end_issues, kept_line_block_factors, make_far_field_result_callback,
                                make_validate_physics, mesh_kind, missing_source_issues, mode_line, object_result,
                                other_em_physics_labels, physics_counter, quasi_static_domain_requirements,
                                raw_settings, resonance_eigenpairs, resonance_table_study_issues, role_counter,
                                scaled_ops, scipy_issues, shortest_solved_wavelength, source_slot_docs,
                                unused_wavelength_issues, validate_driven_point, write_dispersive_materials)
from .definition_feature_validation import _box_issues, _line_cells, _selected_shapes, validate_line_feature
from .eigenfrequency import resonance_filter_time_item
from ...solvers.line_block import factor_solve_ops
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import bor_memory_estimate, pec_memory_estimate
from .renderer_bor import BORRenderer
from .selftests_bor import MICRODISK_DESCRIPTION, MICRODISK_LABEL, MICRODISK_SUMMARY, microdisk_bor_eigenmodes_spec
from .solvers.fdfd_bor import (BOR_MATERIAL_SCHEMA_KEY, BOR_PERMITTIVITY_ARRAYS, BOR_PERMITTIVITY_SLOTS, BORPhysics,
                               bor_far_field_result, bor_flux_result, bor_probe_result, bor_resonance_table_result)
from .solvers.fdfd_bor_fem import FEMBORPhysics

_TYPE_NAME = 'em_fdfd_bor'
_DRIVEN_STUDIES = ('stationary', FREQUENCY_DOMAIN_STUDY)   # an azimuthal_orders list = the driven order sum
_AXIS = 'axis'
_WALL_FACES = ('x_hi', 'y_lo', 'y_hi')
_WALL_CHOICES = ('cpml', 'pec', 'pmc')
_FIELDS_NOTE = ('Fields vary as exp(j (omega t - m phi)): the phasors shown are those of that travelling wave at phi = '
                '0; a standing wave (the +m and -m modes, degenerate in a rotationally invariant structure) has E_r, '
                'E_z ~ cos(m phi) and E_phi ~ sin(m phi) with the same profiles.')

# ---- settings ----
_BC_HELP = (
    'Faces of the (r, z) half plane; x_lo is always the symmetry axis r = 0 (bc_x_lo = axis, no other choice). Every '
    'wall sits on the outer grid edge (the window edge of a face without cpml): pec = tangential E 0 there, pmc = '
    'tangential H 0 there; cpml adds pml_cells absorbing cells outside the face (complex coordinate stretch s = kappa '
    '+ sigma / (alpha + j omega eps0) with the FDTD grading of the pml_* settings; on r_hi the stretched radius enters '
    'every metric factor, the cylindrical PML) and ends in a pmc edge beyond them. A pec / pmc z_lo face is the '
    'antisymmetric / symmetric mirror plane of a structure symmetric about z = 0 (the equatorial plane of a disk or '
    'sphere): pmc keeps the modes whose E_r, E_phi are even about it (quasi-TE whispering-gallery modes of a disk), '
    'pec those whose E_z is even (quasi-TM); it halves the unknowns. No periodic faces.')
_BC_TIP = ('What happens to waves at this edge of the half plane.\n\n'
           '- Absorbing layer: waves leave without reflection, as into open space.\n'
           '- Perfect metal wall: a perfect mirror; on the bottom edge also the mirror plane of a structure symmetric '
           'about it, keeping the modes whose vertical electric field is symmetric.\n'
           '- Perfect magnetic wall: a perfect mirror of the opposite kind; on the bottom edge the mirror plane that '
           'keeps the modes whose electric field in the disk plane is symmetric.')
_WALL_CHOICE_LABELS = (('cpml', 'Absorbing layer'), ('pec', 'Perfect metal wall'), ('pmc', 'Perfect magnetic wall'))
_PML_VISIBLE = tuple((f'bc_{face}', ('cpml',)) for face in _WALL_FACES)


def axis_wall_settings(bc_help):
    """Face settings of the (r, z) half plane (em_fdfd_bor, em_fdtd_bor), each with the help bc_help: bc_x_lo (the
    symmetry axis, no other choice), the cpml / pec / pmc walls of the other faces and the pml_* settings shown with a
    cpml wall."""
    return (
        Param('bc_x_lo', 'choice', _AXIS, label=FACE_LABELS['x_lo'], choices=(_AXIS,),
              choice_labels=((_AXIS, 'Symmetry axis (r = 0)'),), help=bc_help,
              tip='The left edge is the rotation axis: no boundary condition can be set there.'),
        *(Param(f'bc_{face}', 'choice', 'cpml', label=FACE_LABELS[face], choices=_WALL_CHOICES, help=bc_help,
                tip=_BC_TIP, choice_labels=_WALL_CHOICE_LABELS) for face in _WALL_FACES),
        *(replace(param, visible_if=_PML_VISIBLE) for param in boundary_settings_params(bc_help)
          if param.name.startswith('pml_')),
    )


_SETTINGS = (
    Param(GEOMETRY_SETTING, 'choice', AXISYMMETRIC, label='Geometry', choices=(AXISYMMETRIC,),
          choice_labels=((AXISYMMETRIC, 'Axisymmetric (r, z)'),),
          help='Always axisymmetric: the drawing is the (r, z) half plane of a body of revolution about the x_lo window '
               'edge (x = r >= 0, y = z). The setting exists so that coupled physics with a geometry setting (heat, '
               'electric_currents, electrostatics) recognise a consistent axisymmetric partner (no planar-partner '
               'warning).',
          tip='The drawing is half of a cut through a round object, rotated about the left edge of the domain.'),
    *axis_wall_settings(_BC_HELP),
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help='Vacuum wavelength lambda0 of the solve in a stationary study (omega = 2 pi C0 / lambda0); an '
               'eigenfrequency study searches the resonances nearest it (the target); a frequency_domain study solves '
               'its own wavelengths and ignores it (validation info when it is not the default). The mesh needs >= 10 '
               '(better 20) cells per lambda0 / |n| in every material and >= 2 per skin depth in lossy ones '
               '(validation checks both). Typical 0.4e-6 to 2e-6 m.',
          tip='Vacuum wavelength at which the field is solved, in metres: 1.55e-6 for 1550 nm.\n\n'
              "An 'Eigenfrequency' study looks for the resonances closest to it; a 'Frequency domain' study solves "
              'its own list of wavelengths.'),
    Param('azimuthal_order', 'int', 0, label='Azimuthal order', min=0, max=10000,
          help='Azimuthal order m >= 0: fields ~ exp(-j m phi), one (r, z) solve per order (+m and -m are degenerate '
               'in a rotationally invariant structure: the same resonances and |fields|). m = 0: the rotationally '
               'symmetric fields (TE-like E_phi, H_r, H_z and TM-like E_r, E_z, H_phi decouple: on-axis dipoles, '
               'radial modes of disks and pillars); m = 1: the fields of an on-axis focused or plane wave and the '
               'fundamental modes of fibres and vertical cavities (E_r, E_phi, H_phi are non-zero on the axis); '
               'larger m: whispering-gallery modes of disks, rings and spheres, m ~ 2 pi R n_eff / lambda. For the '
               'free spectral range or the m of a resonance in an eigenfrequency study, list the orders in '
               'azimuthal_orders instead (one run, one assembly); likewise for an off-axis point source in a driven '
               'study (the sum over m of its Fourier terms).',
          tip='Number of field oscillations around the axis.\n\n'
              '0 for fields that are the same all around (an antenna on the axis), 1 for light focused along the '
              'axis, large values (tens) for whispering-gallery modes running around a disk or ring edge.'),
    Param('azimuthal_orders', 'float_list', None, label='Azimuthal order list', unit='1', min=0.0, max=10000.0,
          optional=True,
          help='Distinct integer orders m >= 0 solved in one run; a non-empty list replaces azimuthal_order. The '
               'materials are read and the operator parts K0, K1, K2 assembled once (A(m) = K0 + m K1 + m^2 K2 - '
               'k0^2 M). Eigenfrequency study: n_resonances resonances are searched per order near the wavelength '
               'setting; resonance numbers run order by order in list order (nearest the target first within each), '
               "snapshot labels read 'resonance k: m = <m>, lambda = ...', and the resonance_table has a column "
               'azimuthal_order (a curve with x column azimuthal_order plots lambda versus m: the free spectral '
               'range). Stationary / frequency_domain study: the driven order sum of off-axis sources: every '
               'point_source is then a 3D point current element at phi = 0 (see point_source) expanded into its '
               'Fourier terms exp(-j m phi); each listed m > 0 is solved together with -m (one factor per order: '
               'A(-m) is the phi mirror of A(m)) and the terms are summed. Shown fields and field_probe values are the '
               'phi = 0 half plane of the summed 3D field (the source side; ring-weighted field integrals of them are '
               'no 3D integrals), absorbed_power (q_abs, what em_heat_joule hands to heat) is its phi average; '
               'flux_monitor and far_field add the orders exactly and report each order too. List every order 0, 1, '
               '..., M: the far field and powers converge once M exceeds k rho_s by a few (rho_s = source radius, k = '
               'n omega / c), fields near the source need more. mode_source / mode_monitor need a single order '
               '(validation error); each order costs one factorisation per wavelength (nothing is kept between '
               'solves). With a line_block shifted solver the orders of one axis class (|m| >= 1) are factorised in '
               'one batch. null or [] = azimuthal_order alone.',
          tip="Several azimuthal orders solved in one run, e.g. 14, 15, 16 or 0, 1, 2, ..., 8.\n\n"
              "- Eigenfrequency study: replaces 'Azimuthal order' for the resonance search; the resonance table then "
              'lists the order of each resonance.\n'
              '- Stationary or frequency-domain study: a point source becomes a true point source beside the axis '
              '(not a ring); the field is summed over all listed orders, so list every order from 0 up to a few more '
              'than 2 pi x source radius / wavelength.'),
    Param('discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATION_ORDERS),
          choice_labels=(('quadratic', 'Quadratic'), ('linear', 'Linear')),
          help='Element order on a triangles mesh (MeshSpec kind triangles; tensor meshes ignore it and use the Yee '
               'cell). Eigenfrequency study only: the mixed axisymmetric elements, Nedelec edge elements for E_r, E_z '
               'and Lagrange elements for r E_phi of the same degree (discrete gradients are exact static solutions, '
               'so no spurious modes), elements with a curved edge mapped isoparametrically (quadratic), so disks, '
               'toroids and spheres follow the drawn outline instead of a staircase. quadratic (default) or linear '
               '(Whitney / P1: cheaper, error ~h^2 instead of ~h^4 in k^2, and the layer of elements on the axis is '
               'stiff for m = 0 / |m| = 1 fields there). About 7 (quadratic) or 2 (linear) unknowns per element, one '
               'assembly per solve (K0 + m K1 + m^2 K2 for every listed order), shift-invert by the eigen solver: host '
               'SuperLU, or with its factorization cudss a cuDSS LDL^T on the GPU (complex symmetric pencil), with '
               f'{HOST_DIRECT_NAMES} a MUMPS / PARDISO LDL^T on the host, the _mixed ones their complex64 form with '
               'float64 Rayleigh quotients. Materials are one value per element; a '
               'cpml face gets a ring of mesh pml_layers element layers with the stretch of the pml_* settings (the '
               'stretched radius in every metric factor) and a pec outer side; pec_domain removes its elements and '
               'holds tangential E = 0 on their outline. Not available on triangles (validation errors): stationary '
               'and frequency_domain studies (driven solves) and every source / monitor feature (point_source, '
               'field_probe, flux_monitor, far_field, mode ports); resonance_table and pec_domain work.',
          tip='Order of the finite elements when the mesh is made of triangles: quadratic is far more accurate per '
              'element, linear is cheaper.\n\nUnused on a rectangular-cell mesh.'),
)

# ---- material vocabulary ('modes' table: eps_r -> all three arrays; drawing-plane names, out of plane = phi) ----
_COMPLEX_EPS = ('The solve uses the complex permittivity at the solve wavelength: this value - j sigma_e / (omega '
                'eps0) plus the susceptibility chi(omega) of every Drude / Lorentz pole (exp(+j omega t): loss has '
                'Im < 0).')
_POINT_VALUE = ('E_phi sees the cell value, E_r and E_z (on the cell edges) the ring-volume / length weighted mean of '
                'their two cells (no subpixel smoothing).')
_COMPONENT_NAMES = ('Diagonal anisotropy in cylindrical components is rotationally invariant: the drawing-plane names '
                    'map as eps_xx = eps_rr (E_r, radial), eps_yy = eps_zz of the rotation axis (E_z, axial; the '
                    'drawing y is z) and eps_zz = eps_phiphi (E_phi, out of the drawing plane; it writes the eps_r '
                    'array). A crystal with its optic axis along the rotation axis (z-cut, uniaxial) sets eps_r = n_o^2 '
                    'and eps_yy = n_e^2; a radially anisotropic shell sets eps_xx.')
_COMPONENT_LABELS = {'eps_zz': 'Relative permittivity, phi (azimuthal)', 'eps_xx': 'Relative permittivity, r (radial)',
                     'eps_yy': 'Relative permittivity, z (axial, drawing y)'}
# GUI tips of the cylindrical components (em_fdfd_bor and em_fdtd_bor; the shared 'modes' tips name other directions)
COMPONENT_TIPS = {
    'eps_zz': ('Permittivity for the electric field running around the axis (azimuthal).\n\n'
               "When set it replaces 'Relative permittivity' for that direction. For the direction along the axis use "
               f"'{_COMPONENT_LABELS['eps_yy']}'."),
    'eps_xx': ("Permittivity for the radial electric field (pointing away from the axis). When set it replaces "
               "'Relative permittivity' for that direction."),
    'eps_yy': ("Permittivity for the electric field along the rotation axis (vertical in the drawing). When set it "
               "replaces 'Relative permittivity' for that direction: for a crystal with its optic axis along the "
               'rotation axis enter the square of its extraordinary index.'),
}


def cylindrical_material_params(helps):
    """Material Params of the (r, z) types (em_fdfd_bor, em_fdtd_bor): the 'modes' table with the cylindrical labels
    and tips of the components; helps: name -> help text."""
    return tuple(replace(param, label=_COMPONENT_LABELS.get(param.name, param.label))
                 for param in em_material_params('modes', helps, tips=COMPONENT_TIPS))


_MATERIAL_PROPERTIES = cylindrical_material_params({
    'eps_r': 'Isotropic relative permittivity (>= 1) of all three components (eps_rr, eps_phiphi, eps_zz); n^2 of a '
             'lossless dielectric (silicon ~12.1, silica ~2.1 at 1.55 um); may depend on r (x), z (y) and physics '
             f'fields; a component property wins over it. {_COMPONENT_NAMES} {_POINT_VALUE} {_COMPLEX_EPS}',
    'eps_zz': ('eps_phiphi: the azimuthal component, the only one E_phi sees (the out-of-plane component of the '
               'drawing, stored in the eps_r array: it overrides eps_r for E_phi only). Not the axial component: that '
               f'is eps_yy. {_COMPONENT_NAMES}'),
    'eps_xx': f'eps_rr: the radial component, the only one E_r sees (ring-volume weighted mean of the two cells of its r '
              f'edge). {_COMPONENT_NAMES}',
    'eps_yy': f'Axial eps_zz (along the rotation axis = the drawing y): the only component E_z sees (length weighted mean '
              f'of the two cells of its z edge). {_COMPONENT_NAMES}',
    'sigma_e': ('Ohmic loss in S/m, default 0: every permittivity component gains -j sigma_e / (omega eps0); a weak '
                'absorber of index n and extinction kappa has sigma_e = 2 n kappa omega eps0. Model metals with a '
                'Drude pole (it adds to all three components).'),
    'mu_r': ('Real relative permeability (>= 1), 1 for optical materials; H_r, H_phi and H_z take the mean of the cells '
             'around their points. The solve uses mu_r - j sigma_m / (omega mu0).'),
    'sigma_m': 'Magnetic loss in Ohm/m, default 0: mu_r gains -j sigma_m / (omega mu0) (artificial absorbers only).',
})

# ---- features ----
_POINT_SOURCE_PARAMS = (
    Param('component', 'choice', 'z', label='Direction', choices=BOR_SOURCE_COMPONENTS,
          choice_labels=(('r', 'Radial'), ('phi', 'Azimuthal (around the axis)'), ('z', 'Axial (vertical)')),
          help="Current direction: 'phi' J_phi = I / (w h) in the cell (m = 0: a current loop of radius r carrying "
               "I, a magnetic dipole on the axis); 'z' / 'r': the current I through the ring cross-section of the "
               "cell, 2 pi r w (axial) / 2 pi r h (radial), split onto its two edge points (m = 0: the total current; "
               "in the axis cell 'z' is an on-axis electric dipole of moment I h, h = cell height; 'r' in the axis "
               'cell at |m| = 1 is the rotating transverse dipole of moment I w / 2). For m != 0, I '
               'is the amplitude of the exp(-j m phi) current distribution. In a driven order sum (azimuthal_orders '
               'in a stationary / frequency_domain study) the element is instead a 3D point current at phi = 0 (in '
               "the drawing plane; 'phi' points out of it) of dipole moment I d, d = the cell height for z, its "
               'width for r and phi: every order then carries the ring current I (r, z) or I d / (2 pi r) (phi), '
               'and in free space it radiates eta0 k^2 (I d)^2 / (12 pi) wherever it sits.',
          tip='Direction of the current: around the axis (a loop), along the axis or radially.\n\n'
              'Along the axis in the first column of cells is a dipole antenna on the axis.'),
    Param('current', 'float', 1e-3, label='Current', unit='A',
          help=f'Current I [A] of the element (see component); fields scale linearly with it. {POINT_SOURCE_SIGN}',
          tip='Peak current of the source. Fields scale linearly with it.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Phasor phase of the current in degrees (0 = cos carrier, as the em_fdfd_* sources).',
          tip='Phase of the source in degrees; it matters only with several sources.'),
)
_PROBE_PARAMS = (
    Param('component', 'choice', 'ez', label='Field component', choices=BOR_FIELD_COMPONENTS,
          choice_labels=(('er', 'E_r'), ('ephi', 'E_phi'), ('ez', 'E_z'), ('hr', 'H_r'), ('hphi', 'H_phi'),
                         ('hz', 'H_z')),
          help='Component recorded: its complex phasor at the centre of the cell containing the point (edge points '
               'averaged there), per driven solve (stationary, frequency_domain).',
          tip='Field recorded at the point; electric fields in V/m, magnetic fields in A/m.'),
)
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency study solve at the azimuthal order setting (or per order of "
    "azimuthal_orders, order by order), nearest the target (the wavelength setting) first. Result 'table', one row "
    'per resonance: resonance_number, wavelength = 2 pi C0 / '
    'Re(omega) in m, frequency / frequency_imag = Re / Im(omega) / (2 pi) in Hz (> 0 decays), q_factor = Re(omega) / '
    '(2 Im(omega)) (inf when lossless and closed), pml_fraction (share of the |E|^2 ring volume in PML cells; above '
    '0.5 dropped), azimuthal_order (m of the row); extras target_wavelength (m), selected_resonance and '
    'azimuthal_orders (the solved list). Accurate only near the target (see the '
    "eigenfrequency study). For Q versus a parameter add a derived value with source '<physics>.<this feature>', "
    'expression q_factor, operation at and at = 0.')


def _validate_resonance_table(vctx, physics_spec, feature_spec, values):
    """Warning where the study does not eigen-solve the physics: the table has no solve to show."""
    return resonance_table_study_issues(vctx, physics_spec, feature_spec)


# ---- flux, far field and ports ----
_FAR_FIELD_PARAMS = (replace(N_ANGLES_PARAM,
                             help='Far-field angles evenly spaced over 360 degrees in the drawing (meridional) plane, '
                                  'counter-clockwise from +r (0 = +r = the equatorial plane, pi/2 = +z along the axis, '
                                  'pi = the equator on the other side, phi = pi); 360 gives 1-degree steps.',
                             tip='Number of directions, evenly spaced over the full circle of the drawing plane, '
                                 'counter-clockwise from the right (radial) direction.\n\n360 gives 1-degree steps.'),)
_MODE_PARAMS_TEXT = ('The port line must be horizontal (z = const). It starts on the axis (x0 = 0) or at an inner '
                     'radius x0 > 0, an annular line whose start is a metal (pec) end on that cell edge (coaxial '
                     'guides with the line from the inner conductor surface, ring-shaped cores whose field is small '
                     'toward the axis); it ends either on the last window cell (then it continues through the r_hi cpml '
                     'cells, or ends on the pec / pmc r_hi wall) or inside the domain, again a metal end on that cell '
                     'edge (extend the line well into the cladding: the run log warns when the mode still has 1% of '
                     'its peak field at a metal end; exact when the cells beyond a metal end are pec in the model, '
                     'e.g. a pec_domain inner conductor). Vertical (r = const) port lines are not supported. The guided '
                     'modes ~ exp(-j beta z) of the line at the azimuthal order are full-vector solutions of the '
                     'body-of-revolution operator itself on a straight guide (all six components; the line cells and '
                     'their row width), re-solved at every solved wavelength and after every material change, so the '
                     'launch / decomposition is exact for a guide invariant along z with uniform rows for a cell on '
                     'each side of the line (the run log warns otherwise). Modes are ranked by descending Re n_eff '
                     '(mode 1 = the highest), radiation modes with more than half of their energy in the PML dropped; '
                     'each carries 1 W (the exactly conserved discrete flux of flux_monitor) in the travelling '
                     'exp(-j m phi) wave.')
_PORT_POWER_PARAMS = (
    Param('power', 'float', 1.0, label='Power', unit='W', min=0.0,
          help='Power P [W] of the launched mode through the port (ring weights 2 pi: the travelling exp(-j m phi) '
               'wave; its cos(m phi) standing-wave half carries P / 2 for m != 0); fields scale with sqrt(P).',
          tip='Power carried by the launched mode, in watts. Fields grow with its square root, so 1 is the usual '
              'choice.'),
    Param('mode_number', 'int', 1, label='Mode number', min=1, max=50,
          help='Guided mode launched: 1 = the highest Re n_eff at this azimuthal order (e.g. HE11 of a fibre at m = '
               '1, TE11 of a metal pipe), 2 the next; the result table lists the modes found.',
          tip='Which guided mode to launch: 1 is the one with the highest effective index.'),
    Param('direction', 'choice', 'positive', label='Direction', choices=('positive', 'negative'),
          choice_labels=(('positive', 'Upward (+z)'), ('negative', 'Downward (-z)')),
          help="Launch toward +z ('positive': total field on the line row and above, only what comes back below) or "
               "-z ('negative': total field on the line row and below).",
          tip='Direction along the axis in which the mode is sent.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Phase of the launched mode amplitude sqrt(P) e^{j phase} on the line row (each mode is phased with its '
               'largest field sample real and positive).',
          tip='Phase of the launched mode in degrees; it matters only with several sources.'),
)
_MODE_MONITOR_PARAMS = (
    Param('mode_number', 'int', 1, label='Mode number', min=1, max=50,
          help='Guided mode measured (ranking as mode_source); later wavelengths follow it by overlap.',
          tip='Which guided mode to measure: 1 is the one with the highest effective index.'),
)
_MIRROR_FACES = ('pec', 'pmc')


def _port_line(ctx, selection, feature_spec):
    """(z position, (r start, r end)) of a port feature's horizontal line; ValueError for a vertical one."""
    axis, position, span = mode_line(ctx, selection, feature_spec)
    if axis != 'x':
        raise ValueError(f"feature '{feature_spec.name}': a port line must be horizontal (z = const)")
    return position, span


def _apply_mode_source(ctx, solver, feature_spec, values, selection):
    position, span = _port_line(ctx, selection, feature_spec)
    return solver.add_mode_source(position, span, power=values['power'], mode_number=values['mode_number'],
                                  direction=values['direction'], phase=values['phase'])


def _apply_mode_monitor(ctx, solver, feature_spec, values, selection):
    position, span = _port_line(ctx, selection, feature_spec)
    return solver.add_mode_monitor(position, span, mode_number=values['mode_number'])


def _line_cell_messages(vctx, feature_spec, port):
    """(level, text) per selected axis-aligned line: its measuring face on the outer grid edge (flux), or for a port
    a vertical line, a row without a neighbour row on both sides."""
    if vctx.mesh is None or vctx.domain_size is None:
        return []
    messages = []
    axes = dict(zip('xy', _window_axes(vctx.mesh)))
    for shape in _selected_shapes(vctx, feature_spec, 'line'):
        cells = _line_cells(vctx.mesh, shape.values, vctx.domain_size)
        if cells is None:
            continue   # not axis-aligned: reported by validate_line_feature
        normal_axis, line_cell, _ = cells
        n_normal = axes[normal_axis][0].size
        if port:
            if normal_axis != 'y':
                messages.append(('error', f"line '{shape.name}' must be horizontal (z = const): ports measure modes "
                                          'travelling along the axis (radial r = const ports are not supported)'))
            elif not 1 <= line_cell <= n_normal - 2:
                messages.append(('error', f"line '{shape.name}' needs a mesh row below and above its row inside the "
                                          'grid: move it off the z edges'))
        elif line_cell >= n_normal - 1:
            messages.append(('error', f"line '{shape.name}' lies in the last grid cell along {normal_axis}, whose hi "
                                      'face is the outer grid edge: move it inside'))
    return messages


def _validate_line_monitor(port):
    def validate(vctx, physics_spec, feature_spec, values):
        """Axis-aligned line inside the domain, its cell rules, the eigenfrequency warning."""
        path = feature_path(physics_spec, feature_spec)
        return (validate_line_feature(vctx, physics_spec, feature_spec, values)
                + [Issue(level, path, text) for level, text in _line_cell_messages(vctx, feature_spec, port)]
                + eigen_only_feature_issues(vctx, physics_spec, feature_spec))
    return validate


def _validate_far_field(vctx, physics_spec, feature_spec, values):
    """Rectangle inside the domain with one window cell beyond r_hi, z_hi and z_lo (none on a pec / pmc z_lo wall)."""
    path = feature_path(physics_spec, feature_spec)
    issues, inside = _box_issues(vctx, physics_spec, feature_spec)
    if vctx.mesh is not None:
        (_, x_edges, _, x_last), (_, y_edges, y_first, y_last) = _window_axes(vctx.mesh)
        settings = raw_settings(physics_spec)   # malformed: reported
        mirror = settings.get('bc_y_lo', 'cpml') in _MIRROR_FACES
        for shape in inside:
            box = shape.values
            x_hi = containing_cell(max(box['x0'], box['x1']), x_edges, 0, x_last)
            y_lo = containing_cell(min(box['y0'], box['y1']), y_edges, y_first, y_last)
            y_hi = containing_cell(max(box['y0'], box['y1']), y_edges, y_first, y_last)
            if x_hi >= x_last or y_hi >= y_last:
                issues.append(Issue('error', path, f"far-field box '{shape.name}' needs one cell beyond its x1 / y1 "
                                                   f'edges inside the domain: keep x1 < {x_edges[x_last]:.4g} m and '
                                                   f'y1 < {y_edges[y_last]:.4g} m'))
            if y_lo <= y_first and not (mirror and y_lo == 0):
                issues.append(Issue('error', path, f"far-field box '{shape.name}' needs one cell below its y0 edge "
                                                   f'inside the domain (keep y0 >= {y_edges[y_first + 1]:.4g} m), '
                                                   'unless y0 = 0 on a pec / pmc z_lo mirror'))
    return issues + eigen_only_feature_issues(vctx, physics_spec, feature_spec)


_FLUX_DESCRIPTION = (
    'Time-averaged power [W] through an axis-aligned line at every driven solve, with the 2 pi ring weights of the '
    'travelling exp(-j m phi) wave (a cos(m phi) standing wave carries half for m != 0): a horizontal line (z = const) '
    'gives the flux along +z through the annulus it sweeps, pi sum [A Re(E_r H_phi*) - r w Re(E_phi H_r*)], a vertical '
    'line (r = const) the flux along +r through the cylinder, pi sum [r h Re(E_phi H_z*) - r L Re(E_z H_phi*)]; E of '
    'the line cells, H on their hi face (Yee points, no averaging: the pairing the discrete Poynting theorem conserves '
    'exactly through lossless cells, and that the port modes are normalised with), span ends by their share. A line '
    "from the axis across the whole field measures the full power; a closed box of lines the radiated power. Result "
    "'flux_spectrum' with unit 'W', one entry per solved wavelength sorted by wavelength. In a driven order sum "
    '(azimuthal_orders) power is the total of the 3D field (the orders are orthogonal in phi, so their powers add '
    'exactly), extras power_orders (n_wavelengths x n_orders: each listed m with its -m) and azimuthal_orders. The '
    f'line covers {SPAN_CELLS}; its hi face must be inside the grid.')
_FAR_FIELD_DESCRIPTION = (
    'Near-to-far-field transform of the driven solution on the closed surface of a rectangle in the (r, z) drawing: '
    'its two z faces and its outer cylinder (and an inner cylinder when x0 > 0); a box bottom at y0 = 0 on a pec / pmc '
    'z_lo mirror is closed by the mirror images instead (E_r, E_phi, H_z even for pmc, odd for pec). Equivalent '
    'currents J = n x H, M = -n x E (H on its Yee points, tangential E interpolated onto the face), integrated over phi '
    'with the exp(-j m phi) dependence in closed form (Bessel functions J_(m-1), J_m, J_(m+1)), vacuum outside the box '
    '(enclose every scatterer and source). Result \'far_field\', one row per solved wavelength: power = radiated power '
    'per steradian r^2 |E|^2 / (2 eta0) in W/sr versus angle in the drawing (meridional) plane, counter-clockwise from '
    '+r (0 = equator, pi/2 = +z along the axis; angles with cos < 0 are the phi = pi side, the same power since the '
    'travelling wave has a phi-independent |E|); its solid-angle integral 2 pi int power cos(angle) d angle over the '
    'cos >= 0 half (-pi/2 .. pi/2) is the radiated power (flux_monitor box); extras power_theta / power_phi (E_theta / E_phi parts), angle_axes, '
    'azimuthal_order. In a driven order sum (azimuthal_orders) power is the pattern of the summed 3D field in the '
    'meridional plane through the sources (angles with cos >= 0 at phi = 0, the others at phi = pi: no longer the same '
    'power), the order amplitudes added with exp(-j m phi); extras power_azimuthal_mean (its phi average, the sum over '
    'the orders: 2 pi int power_azimuthal_mean cos(angle) d angle over the cos >= 0 half is the radiated power), '
    'power_orders (n_wavelengths x n_orders x n_angles: the phi average per listed m with its -m) and '
    'azimuthal_orders instead of azimuthal_order. Needs one cell beyond x1 / y1 and below y0 inside the domain; the '
    f'box covers {BOX_CELLS}.')
_MODE_TABLE_TEXT = ("Result 'table' of the latest port solve (a frequency_domain study: its last solved wavelength), one "
                    'row per mode found (up to max(mode_number + 2, 4)): mode_number, n_eff, n_eff_imag (< 0 = loss), '
                    'loss_db_per_cm, n_group (from the pencil derivative, without material dispersion), edge_field '
                    '(largest |field| at the line end / peak) and selected; extras wavelength (m), power (W), '
                    'direction, mode_number and phase (deg).')
_MODE_MONITOR_RESULT_TEXT = (
    "Result 'xy' versus vacuum wavelength: power_positive / power_negative = |a|^2 in W, the power of the mode toward "
    '+z / -z (separated by the discrete reciprocity form of the line row and the row above, so other modes drop out); '
    'with exactly one enabled mode_source in the physics also fraction_positive / fraction_negative = |a / (sqrt(P) '
    'e^{j phase})|^2 (transmission / reflection into this mode) and phase_positive / phase_negative (rad, not '
    'de-embedded); extras frequency, wavelength, n_eff (real part), n_eff_imag, mode_number, normalized. Keep the line and '
    'the row above off the line rows of a mode_source.')


_FEATURES = (
    FeatureType('point_source', 'Point source', 'source', 'point', params=_POINT_SOURCE_PARAMS,
                object_kinds=('point',), max_entities=1,
                description='Current element in the cell containing the point (x = r, y = z), a ring in 3D with the '
                            f'exp(-j m phi) dependence of the azimuthal order; see component. {POINT_SOURCE_SIGN}',
                summary='A small current at a point of the half plane, a ring around the axis in 3D; on the axis an '
                        'antenna.',
                apply=apply_component_point_source, validate=validate_driven_point),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='xy',
                description="Phasor of one field component in the cell containing the point at every driven solve "
                            '(a driven order sum: of the summed field at phi = 0). Result \'xy\' versus wavelength: '
                            'series abs (unit of the component) and phase (rad); extras frequency, wavelength, '
                            'component, unit.',
                summary='Records the field at one point for every solved wavelength.',
                apply=apply_field_probe, result=object_result(bor_probe_result), validate=validate_driven_point),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor (rings in 3D): E_phi = 0 at the centre of every cell whose '
                            'centre lies in the selected domains and E_r / E_z = 0 on the edges between two such '
                            'cells, staircased to the mesh. On a triangles mesh its elements leave the unknowns and '
                            'tangential E (E_phi included) is held 0 on their outline (exact on curved outlines with '
                            'discretization quadratic).',
                summary='Turns the selected domains into a perfect metal, following the mesh cells.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0), grids=BOTH_GRIDS),
    FeatureType('mode_source', 'Mode source', 'source', 'line', params=_PORT_POWER_PARAMS, object_kinds=('line',),
                max_entities=1, result_kind='table',
                description=('Port source launching one guided mode of a waveguide along the axis (fibre, pillar, '
                             'metal pipe, coaxial guide) from a horizontal line in one direction: a one-sided '
                             'total-field / scattered-field interface between the line row and the row upstream, the '
                             'mode entering the right-hand side through the operator rows on both sides, so the launch '
                             f'is exact and one-sided on a straight guide. {_MODE_PARAMS_TEXT} {_MODE_TABLE_TEXT}'),
                summary='Launches one guided mode of a waveguide running along the axis, up or down, recomputed at '
                        'each solved wavelength. Pair it with mode monitors for transmission and reflection.',
                apply=_apply_mode_source, result=object_result(bor_mode_source_result),
                validate=_validate_line_monitor(port=True)),
    FeatureType('flux_monitor', 'Flux monitor', 'monitor', 'line', object_kinds=('line',), max_entities=1,
                result_kind='flux_spectrum', description=_FLUX_DESCRIPTION,
                summary='Measures the power crossing a horizontal disc or annulus (upward positive) or a vertical '
                        'cylinder (outward positive) at each solved wavelength, in watts.',
                apply=apply_flux_monitor, result=object_result(bor_flux_result),
                validate=_validate_line_monitor(port=False)),
    FeatureType('mode_monitor', 'Mode monitor', 'monitor', 'line', params=_MODE_MONITOR_PARAMS,
                object_kinds=('line',), max_entities=1, result_kind='xy',
                description=('Port monitor splitting the field on a horizontal line into the upward (positive) and '
                             f'downward (negative) amplitudes of one guided mode at every solved wavelength. '
                             f'{_MODE_PARAMS_TEXT} {_MODE_MONITOR_RESULT_TEXT}'),
                summary='Measures the power in one guided mode crossing a horizontal line, up and down, at each '
                        'solved wavelength. With a single mode source it also gives transmission and reflection.',
                apply=_apply_mode_monitor, result=object_result(bor_mode_monitor_result, with_solver=True),
                validate=_validate_line_monitor(port=True)),
    FeatureType('far_field', 'Far-field pattern', 'monitor', 'domain', params=_FAR_FIELD_PARAMS,
                object_kinds=('rectangle',), max_entities=1, result_kind='far_field',
                description=_FAR_FIELD_DESCRIPTION,
                summary='Computes the power radiated in each direction of the drawing plane at each solved '
                        'wavelength, from the fields on a rectangle around every object (a full 3D pattern, the same '
                        'for every angle around the axis). Keep the rectangle in vacuum.',
                apply=apply_far_field, result=make_far_field_result_callback(_FAR_FIELD_PARAMS, bor_far_field_result),
                validate=_validate_far_field),
    FeatureType('resonance_table', 'Resonance table', 'monitor', None, result_kind='table',
                description=_RESONANCE_TABLE_DESCRIPTION,
                summary="Lists the resonances found by an 'Eigenfrequency' study: wavelength, frequency, quality "
                        'factor and how much of each lies in the absorbing layers.',
                apply=apply_resonance_table, result=object_result(bor_resonance_table_result, with_solver=True),
                validate=_validate_resonance_table, grids=BOTH_GRIDS),
)

# ---- type ----
_DESCRIPTION = (
    'Body-of-revolution (BOR) finite-difference frequency-domain solver: the drawing is the (r, z) half plane of a '
    'rotationally symmetric structure about the x_lo window edge (x = r >= 0, y = z), the fields of one azimuthal '
    f'order m. {_FIELDS_NOTE} All six components are solved (they couple for m != 0) on the 3D cylindrical Yee cell '
    'projected onto the half plane (finite-integration form, exact ring areas and volumes): E_phi, H_r at the cell '
    'centres (H_r on the +z edge), E_r, H_z on the +r cell edges, E_z on the +z edges, H_phi at the +r+z corner; the '
    'axis carries only the m = 0 H_z unknown (|m| = 1: E_r, H_phi there recovered from cell 0 by regularity; larger '
    '|m|: none). True 3D results of disks, '
    'rings, pillars, spheres (staircased), lenses, vertical cavities and on-axis dipoles at 2D cost. Solved on the '
    "host: solver role 'linear' = sparse_direct (host SuperLU; a 'cusolver_qr' node is a validation error), one factor "
    'per wavelength and material state, reused as GMRES preconditioner after a material change at the same wavelength '
    "(thermo-optic iteration); with back_end 'cudss' / 'cudss_mixed' (GPU backend, nvmath-python) a cuDSS LDL^T on the "
    'GPU instead, without the host size limit, every later factorisation of the same sparsity (other wavelengths, '
    "material states, orders of an order sum) reusing the first one's cuDSS plan; with back_end "
    f'{HOST_DIRECT_NAMES} (optional python-mumps / pydiso packages, either backend) the same LDL^T by MUMPS / Intel '
    "PARDISO on the processor cores, recycled like SuperLU and under its size limit; role 'eigen' = "
    'arpack_shift_invert (eigenfrequency study: '
    'its tolerance / subspace_size; the finite-integration operator is complex symmetric, so factorization cudss is a '
    f'cuDSS LDL^T on the GPU without the host size limit, {HOST_DIRECT_NAMES} a MUMPS / PARDISO LDL^T on the host '
    'under the SuperLU size limit, and the _mixed ones their complex64 form: loose solves, then the '
    "float64 Rayleigh quotient of every resonance); role 'shifted' (every study): Automatic (sparse_direct) changes "
    'nothing, a '
    'line_block node factors every driven system and every shift-invert operator on its backend instead (unknowns '
    'ordered by z rows or r columns, whichever is cheaper; the orders of an azimuthal_orders list with |m| >= 2 in one '
    'batch of A(m) = K0 + m K1 + m^2 K2 - k0^2 M; in a frequency_domain study of one order up to 8 wavelengths in one '
    'batch of whole operators, as many as 60 % of the free memory holds; mixed or double precision, refined against the '
    'complex128 operator), and a solve it rejects (residual above its accept tolerance, e.g. a nearly singular indefinite system) goes to '
    'host SuperLU with a run-log warning. Needs scipy; 3 unknowns per cell incl. PML, so validation warns near and refuses '
    'beyond the factor size limit. Studies: stationary (one driven solve at the '
    'wavelength setting, iterated with heat through heat_em_thermo_optic / em_heat_joule; the heat physics should be '
    'axisymmetric too), frequency_domain (one driven solve per wavelength; with an azimuthal_orders list both driven '
    'studies sum the orders of off-axis point sources into the 3D field, see that setting) and eigenfrequency '
    '(shift-invert resonances nearest the wavelength setting: complex omega, Q from material loss and radiation into the '
    'PML, whispering-gallery modes of disks and rings from one 2D solve) and optomechanics (as the optical physics an '
    'optomechanical_coupling of an axisymmetric solid_mechanics names: the same eigen-solve with n_optical_resonances, '
    'then the g0 table; triangles mesh only, an em_fdfd_bor no coupling names solves as a stationary background '
    'physics). Outputs at the cell centres: er, ephi, ez, hr, '
    'hphi, hz (Re of the phasors at phi = 0, V/m and A/m), their moduli <component>_abs, e_abs, e_intensity (|E|^2, '
    'the primary field), h_abs, absorbed_power (state q_abs = 1/2 omega eps0 sum_c (-Im eps_c) |E_c|^2 in W/m^3, the '
    'cell mean over the component points; em_heat_joule hands it to heat) and the permittivity arrays eps_r '
    '(eps_phiphi), eps_xx (eps_rr), eps_yy (axial eps_zz) / sigma_e. Diagonal anisotropy in cylindrical components '
    '(rotationally invariant): see the eps_* material properties. A resonance is scaled to 1 real at its largest '
    'component (V/m). Field derived values weight by the ring volume 2 pi r dr dz (the travelling exp(-j m phi) wave: '
    '|F|^2 does not depend on phi; a standing cos(m phi) wave has half the integral), and so do powers. Features: '
    'point_source (current elements), mode_source / mode_monitor (vector guided modes of a guide along the axis at '
    'the azimuthal order, horizontal lines from the axis or annular ones from an inner radius), field_probe, '
    'flux_monitor (W through a disc, annulus or '
    'cylinder, exactly conserved), far_field (W/sr in the meridional plane), pec_domain, resonance_table. Cannot '
    'share a model with in-plane EM types (validation error). heat_em_thermo_optic writes all three permittivity '
    'arrays (slots eps_r_update, eps_xx_update, eps_yy_update). On a triangles mesh (MeshSpec kind triangles) the '
    'eigenfrequency and optomechanics studies run on mixed axisymmetric finite elements instead (see '
    'discretization: curved outlines '
    'followed, materials and state fields one value per element, the same resonance_table); driven studies and the '
    "source / monitor features stay on tensor meshes. The instance is usually named 'em'.")


def _create(ctx, physics_spec, settings_values):
    """BORPhysics at the first solved wavelength with its dispersive poles and their background permittivity written
    into the three permittivity arrays (the order list: eigen-solve orders, or in a driven study the order sum); never
    finalizes. A triangles mesh: FEMBORPhysics (eigen-solved only; walls from the settings). Driven = not eigen-solved
    (an optomechanics study solves the em_fdfd_bor no coupling names as a stationary background physics)."""
    solvers = ctx.solvers(physics_spec.name)
    orders = [int(order) for order in settings_values.get('azimuthal_orders') or ()]
    if getattr(ctx.domain, 'grid_kind', 'tensor') == 'triangles':
        solver = FEMBORPhysics(ctx.domain, _domain_requirements(settings_values).boundary_config,
                               wavelength=first_solved_wavelength(ctx, settings_values),
                               azimuthal_order=settings_values['azimuthal_order'], azimuthal_orders=orders,
                               eigen_solver=solvers.get('eigen'),
                               order=DISCRETIZATION_ORDERS[settings_values['discretization']])
    else:
        driven = not eigen_solved(ctx.spec, physics_spec.name)
        solver = BORPhysics(ctx.domain, boundary_config=ctx.domain.boundary_config,
                            wavelength=first_solved_wavelength(ctx, settings_values),
                            azimuthal_order=settings_values['azimuthal_order'], linear_solver=solvers['linear'],
                            eigen_solver=solvers.get('eigen'), shifted_solver=solvers.get('shifted'),
                            azimuthal_orders=None if driven else orders, driven_orders=orders if driven else None)
    write_dispersive_materials(ctx, physics_spec, solver, BOR_PERMITTIVITY_ARRAYS)
    return solver


def _domain_requirements(settings_values):
    """The boundary configuration (FDFD rules) with the axis face as a plain pmc edge (no PML padding there)."""
    return quasi_static_domain_requirements({**settings_values, 'bc_x_lo': 'pmc'})


def _other_em_issues(vctx, path):
    """Error when another EM type shares the model: their drawing is a planar cut, this one a half plane of rotation."""
    others = other_em_physics_labels(vctx, _TYPE_NAME)
    if not others:
        return []
    return [Issue('error', path, f"em_fdfd_bor treats the drawing as the (r, z) half plane of a body of revolution, "
                                 f"while {', '.join(others)} solve it as a planar plane or cross-section; they cannot "
                                 f"share one model")]


def _solver_issues(vctx, physics_spec, path):
    """scipy missing: error; a linear solver node with back_end cusolver_qr: error (host matrix only); the host factor
    size limit (size_issue_for; none for a cudss eigen-solve or, driven, a cuDSS linear back end) on tensor meshes; a
    cudss eigen node beside a line_block shifted one: warning (unused); driven study without a source: warning."""
    solvers = vctx.solvers(physics_spec) or {}
    issues = (scipy_issues('em_fdfd_bor', 'host sparse direct and eigen solvers', path)
              + host_matrix_back_end_issues(solvers, 'em_fdfd_bor', path))
    eigen = eigen_solved(vctx.spec, physics_spec.name)
    eigen_on_device = eigen and eigen_factorization(solvers) in CUDSS_FACTORIZATIONS
    shifted = solvers.get('shifted')
    if eigen_on_device and shifted is not None and shifted.solver_type.name == 'line_block':
        issues.append(Issue('warning', path, f"the eigen factorization '{eigen_factorization(solvers)}' is unused: the "
                                             "line_block 'shifted' node factorises the shift-invert operators (cuDSS "
                                             'only for an order the line block rejects)'))
    on_device = eigen_on_device if eigen else linear_solver_cudss(solvers)
    line_block = shifted is not None and shifted.solver_type.name == 'line_block'   # no host LU (unless it rejects)
    if mesh_kind(vctx) == 'tensor' and not on_device and not line_block:
        issue = size_issue_for(int(vctx.mesh.nx), int(vctx.mesh.ny))
        if issue is not None:
            issues.append(Issue(issue[0], path, issue[1]))
    return issues + missing_source_issues(vctx, physics_spec, path, ('point_source', 'mode_source'),
                                          ('stationary', FREQUENCY_DOMAIN_STUDY))


_SHARED_VALIDATE = make_validate_physics(BOR_PERMITTIVITY_SLOTS, time_domain=False,
                                         solved_wavelengths=all_solved_wavelengths)


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared EM checks (materials, slot writers, resolution and skin depth at every solved wavelength), the
    unused wavelength setting, other EM types, the solver and the (r, z) partner rule."""
    path = ('physics', physics_spec.name)
    issues = _SHARED_VALIDATE(vctx, physics_spec, settings_values)
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(_other_em_issues(vctx, path))
    issues.extend(_solver_issues(vctx, physics_spec, path))
    issues.extend(_order_list_issues(vctx, physics_spec, settings_values, path))
    issues.extend(axisymmetric_issues(vctx, physics_spec, path, 'em_fdfd_bor'))
    issues.extend(_triangle_issues(vctx, physics_spec, settings_values, path))
    return issues


def _triangle_issues(vctx, physics_spec, settings_values, path):
    """On a triangles mesh: an error where the study does not eigen-solve it (no driven FEM solve yet); the FEM size rule
    of host SuperLU on the estimated unknowns (none for a cudss eigen factorization). The source / monitor features are
    refused by their grids."""
    if mesh_kind(vctx) != 'triangles':
        return []
    issues = []
    if not eigen_solved(vctx.spec, physics_spec.name):
        issues.append(Issue('error', path, f'em_fdfd_bor on a triangles mesh solves resonances only (an '
                                           f'{EIGENFREQUENCY_STUDY} study, or as the optical physics an '
                                           f'optomechanical_coupling names in an {OPTOMECHANICS_STUDY} study); here '
                                           f'the {vctx.study_type} study would drive it (driven solves, sources, ports, '
                                           'flux and far field need a tensor mesh)'))
    return issues + fem_size_issues(vctx, physics_spec, settings_values, path, FEM_BOR_SIZE_REMEDY)


def azimuthal_order_list_issues(orders, path, context='', sum_check=True):
    """(issues, integer orders or None) of an azimuthal_orders list (em_fdfd_bor, em_fdtd_bor): errors for
    non-integer or repeated orders (None then); sum_check: a warning when an order 0..max is missing (a point source's
    order sum would miss its terms; context follows the list in the text)."""
    if any(order != int(order) for order in orders):
        return [Issue('error', path, f'azimuthal_orders must be integers, got {orders}')], None
    orders = [int(order) for order in orders]
    if len(set(orders)) != len(orders):
        return [Issue('error', path, f'azimuthal_orders must be distinct, got {orders}')], None
    missing = sorted(set(range(max(orders) + 1)) - set(orders)) if sum_check else []
    if not missing:
        return [], orders
    return [Issue('warning', path, f'azimuthal_orders {sorted(orders)}{context}: the order sum of a point source '
                                   f'misses m = {missing} (list every order 0..M)')], orders


def _order_list_issues(vctx, physics_spec, settings_values, path):
    """azimuthal_orders: integers, distinct (errors); in a driven study (the order sum) no mode ports (error) and every
    order 0..max (warning: a point source's sum would miss terms)."""
    orders = settings_values.get('azimuthal_orders') or []
    if not orders:
        return []
    driven = vctx.study_type in _DRIVEN_STUDIES
    issues, orders = azimuthal_order_list_issues(orders, path, f' in a {vctx.study_type} study', sum_check=driven)
    if orders is None or not driven:
        return issues
    ports = [feature.name for feature in physics_spec.features
             if feature.enabled and feature.type in ('mode_source', 'mode_monitor')]
    if ports:
        issues.insert(0, Issue('error', path, f"azimuthal_orders in a {vctx.study_type} study sums the orders of "
                                              f"off-axis point sources, but the mode ports {', '.join(ports)} belong to "
                                              'one order: clear azimuthal_orders (use azimuthal_order) or disable '
                                              'them'))
    return issues


# run-time estimate: CSR entries per row of A(m) (bor_line_block's figure) and host bytes moved per stored entry by one
# BORAssembly (material points, curl split, K0 / K1 / K2 per axis class; calibrate against the assembly share of a
# frequency_domain run)
_BOR_ENTRIES_PER_ROW = 25
BOR_ASSEMBLY_BYTES_PER_ENTRY = 800
_CSR_ENTRY_BYTES = 20   # complex128 value + int32 column
# SuperLU of 3 coupled unknowns per grid point on the scalar 2D curve priced at n = 3 x cells: nested-dissection
# separators carry b = 3 unknowns per point, so flops b^3 (n / b)^1.5 = b^1.5 n^1.5 and fill ~ b n log n (theory;
# measured 5.5 x the unscaled curve)
BOR_LU_SCALE = 3.0 ** 1.5
BOR_SOLVE_SCALE = 3.0


def _order_solves(order, physics_spec):
    """Solves of order m >= 0 in a driven order sum (body_of_revolution.order_pair_fields): m = 0 one; m > 0 one more for
    -m unless every point source is r / z or every one is phi."""
    if order == 0:
        return 1
    components = {(feature.params or {}).get('component', 'z') for feature in physics_spec.features
                  if feature.enabled and feature.type == 'point_source'}
    return 2 if 'phi' in components and components - {'phi'} else 1


_PORT_FEATURES = ('mode_source', 'mode_monitor')
_PORT_STRIP_ROWS = 3   # bor_ports._STRIP_ROWS


def _port_mode_estimate(mesh, context, assemblies, kind):
    """bor_ports.solve_port_modes per enabled port and assembly (it re-solves per omega and assembly): the three-row
    strip BORAssembly, the host LU of the 2 n_slice companion pencil (n_slice = 3 n_line + 1, n_line <= the r cells)
    and scipy eigs on it (2 max(mode_number + 2, 4) + 8 wanted, scipy's basis, APPLICATIONS_PER_BASIS_GUESS); None
    without ports. Not a role: no counters, the items join 'study'."""
    from ...solvers.arpack_shift_invert import APPLICATIONS_PER_BASIS_GUESS, arnoldi_basis_size
    ports = [feature for feature in context.physics_spec.features if feature.enabled and feature.type in _PORT_FEATURES]
    if not ports:
        return None
    numbers = [(feature.params or {}).get('mode_number', 1) for feature in ports]
    mode_number = max(int(number) if isinstance(number, (int, float)) else 1 for number in numbers)
    n_line = float(mesh.nx)
    pencil = 2.0 * (3.0 * n_line + 1.0)
    basis = arnoldi_basis_size(2 * max(mode_number + 2, 4) + 8, int(pencil))
    applications = APPLICATIONS_PER_BASIS_GUESS * basis
    strip_entries = _BOR_ENTRIES_PER_ROW * 3.0 * n_line * _PORT_STRIP_ROWS
    ops = (te.HostStream(BOR_ASSEMBLY_BYTES_PER_ENTRY * strip_entries), te.PythonCalls(200),
           te.HostLU(pencil, 2, True),
           (applications, te.HostSolve(pencil, 2, True)),
           (applications, te.DenseFlops(16.0 * pencil * basis, host=True)))
    return te.TimeEstimate((te.TimeItem('port line mode solves (per port and assembly)', 'solve',
                                        float(len(ports)) * assemblies, 'guess' if kind == 'exact' else kind, ops),))


def _time_estimate(settings_values, mesh, context):
    """Host solves of 3 unknowns per cell. Driven: per solve one BORAssembly, per order (one, or each listed one of an
    order sum) A(m), a factorisation and 1-2 solves of 'linear' (coupled single order: GMRES on the kept factor) or of
    a line_block 'shifted' node, and the port line modes (_port_mode_estimate); eigenfrequency: one assembly, per
    listed order a factor and a squared-Cayley eigen-solve of 'eigen' (two solves per application; a cudss
    factorization: the recipe's 2D 'not estimated'); triangle meshes: _triangle_time_estimate."""
    if mesh.kind == 'triangles':
        return _triangle_time_estimate(settings_values, mesh, context)
    unknowns = 3.0 * int(mesh.n_cells)
    nnz = _BOR_ENTRIES_PER_ROW * unknowns
    nr, nz = int(mesh.nx), int(mesh.ny)
    block = min((nz + 1, 3 * nr + 1), (nr + 1, 3 * nz + 1), key=lambda pair: pair[0] * pair[1] ** 3)[1]
    shifted = context.solvers.get('shifted')
    line_block = shifted is not None and shifted.solver_type.name == 'line_block'
    listed = [abs(int(order)) for order in settings_values.get('azimuthal_orders') or ()]
    n_orders = float(max(len(listed), 1))
    # the BORAssembly, then A(m) = K0 + m K1 + m^2 K2 per order (three CSR read, one written)
    assembly_ops = (te.HostStream(BOR_ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200),
                    (n_orders, te.HostStream(4.0 * _CSR_ENTRY_BYTES * nnz)))
    if eigen_solved(context.spec, context.name):
        eigenpairs = resonance_eigenpairs(context)
        assembly = te.TimeEstimate((te.TimeItem('BOR assembly', 'assembly', 1.0, 'exact', assembly_ops),
                                    resonance_filter_time_item(context.backend, unknowns, nnz,
                                                               eigenpairs * n_orders)))   # one search per order
        common = dict(unknowns=unknowns, nnz=nnz, value_bytes=16, solves=n_orders, eigenpairs=eigenpairs)
        if line_block:
            return te.combine((assembly, kept_line_block_factors(context, unknowns, nnz, n_orders, block),
                               context.solver_work('eigen', product_ops=scaled_ops(
                                   factor_solve_ops(unknowns, block, nnz, shifted.values.get('precision', 'mixed')),
                                   2.0), **common)))
        if eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS:
            return te.combine((assembly, context.solver_work('eigen', factorizations=n_orders, rhs=2, **common)))
        factors = te.TimeItem('shift-invert factorisations', 'factorisation', n_orders, 'exact',
                              (te.HostLU(unknowns, 2, True, scale=BOR_LU_SCALE),),
                              counter=role_counter(context, 'eigen', 'eigen_factorizations'),
                              seconds_counter=role_counter(context, 'eigen', 'eigen_factor_seconds'))
        return te.combine((assembly, te.TimeEstimate((factors,)), context.solver_work(
            'eigen', product_ops=((2.0, te.HostSolve(unknowns, 2, True, scale=BOR_SOLVE_SCALE)),), backend='cpu',
            **common)))
    points = float(context.count('harmonic_points', 1)) if context.study.type == FREQUENCY_DOMAIN_STUDY else 1.0
    solves, kind, limit = context.static_solves()
    assemblies = points * solves
    solves_per_assembly = float(sum(_order_solves(order, context.physics_spec) for order in listed)) if listed else 1.0
    assembly = te.TimeEstimate((te.TimeItem('BOR assembly', 'solve', assemblies, kind, assembly_ops,
                                            limit=None if limit is None else points * limit,
                                            counter=physics_counter(context, 'solves')),))
    ports = _port_mode_estimate(mesh, context, assemblies, kind)
    if ports is not None:
        assembly = te.combine((assembly, ports))
    if line_block:
        return te.combine((assembly, context.solver_work('shifted', unknowns=unknowns, nnz=nnz, value_bytes=16,
                                                         solves=assemblies * n_orders, solves_kind=kind,
                                                         rhs=solves_per_assembly / n_orders, block_size=block)))
    # a material change at the same omega: GMRES on the kept SuperLU factor (cuDSS refactorises on its kept plan)
    recycled = context.coupled and not listed and not linear_solver_cudss(context.solvers)
    return te.combine((assembly, context.solver_work(
        'linear', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=assemblies * solves_per_assembly,
        solves_kind=kind, factorizations=points if recycled else assemblies * n_orders, factorizations_kind=kind,
        factor_scale=BOR_LU_SCALE, solve_scale=BOR_SOLVE_SCALE, backend='cpu')))   # host matrix on either backend


# ... of a triangles mesh: host bytes per stored entry of one fem_bor assembly (degree-6 quadrature-point coefficients
# with the stretched radius, eight blocks, the order sum and restriction) and Python calls; uncalibrated (compare the
# eigen-solve info's assembly_seconds)
FEM_BOR_ASSEMBLY_BYTES_PER_ENTRY = 3000
FEM_BOR_ASSEMBLY_CALLS = 600


def _triangle_time_estimate(settings_values, mesh, context):
    """em_fdfd_bor on a triangles mesh (eigenfrequency only): fem_resonance_time_estimate per listed order (an
    optomechanics study counts its optical resonances apart from the solid's eigenpairs)."""
    eigenpairs = resonance_eigenpairs(context)
    return fem_resonance_time_estimate(settings_values, mesh, context, 'FEM BOR assembly',
                                       FEM_BOR_ASSEMBLY_BYTES_PER_ENTRY, FEM_BOR_ASSEMBLY_CALLS, eigenpairs,
                                       n_orders=float(max(len(settings_values.get('azimuthal_orders') or ()), 1)),
                                       default_basis=FEM_BOR_ARNOLDI_BASIS_PER_WANTED * eigenpairs)


EM_FDFD_BOR = PhysicsType(
    name=_TYPE_NAME,
    label='Electromagnetic Waves, frequency domain (axisymmetric, azimuthal order m)',
    family=EM_FAMILY,
    group='em',
    spaces_2d=(AXISYMMETRIC,),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    default_instance_name='em',
    description=_DESCRIPTION,
    summary='Light in round structures (disks, rings, pillars, lenses, spheres) from a cut through their axis: full '
            '3D fields of one azimuthal order at the cost of a 2D model, resonances and quality factors included.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS[BOR_MATERIAL_SCHEMA_KEY],
    state_fields=tuple(BORRenderer.field_infos[name] for name in ('e_intensity', 'eps_r', 'eps_xx', 'eps_yy',
                                                                   'sigma_e'))
    + (replace(BORRenderer.field_infos['absorbed_power'], name='q_abs'),),
    output_fields=tuple(BORRenderer.field_infos.values()),
    source_slots=source_slot_docs(BOR_PERMITTIVITY_SLOTS),
    supported_studies=(EIGENFREQUENCY_STUDY, 'stationary', FREQUENCY_DOMAIN_STUDY, OPTOMECHANICS_STUDY),
    default_study={'type': EIGENFREQUENCY_STUDY},
    uses_dispersion=True,
    # shifted: the factors of the order family A(m) (driven systems, shift-invert operators); Automatic sparse_direct =
    # no device factors (host SuperLU of the linear / eigen roles), line_block = batched device factors
    # eigen mixed factorizations (cuDSS, MUMPS, PARDISO): the Rayleigh-polished complex64 LDL^T (Yee and triangle
    # operators are complex symmetric)
    solver_roles={'linear': CUDSS_LINEAR_ROLE,
                  'eigen': eigen_role((EIGENFREQUENCY_STUDY, OPTOMECHANICS_STUDY), mixed_factorization=True),
                  'shifted': SolverRole('complex_symmetric', 'sparse_direct', precisions=('mixed', 'double'))},
    create=_create,
    domain_requirements=_domain_requirements,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate_physics,
    memory_estimate=bor_memory_estimate,
    time_estimate=_time_estimate,
    backends=('gpu', 'cpu'),   # host solves either way; the shown fields live on the run backend
    grids=BOTH_GRIDS,         # triangles: solvers/fdfd_bor_fem (eigenfrequency only)
)


def _selftest_cases():
    """The BOR SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_bor import BOR_SELFTEST_CASES
    from .selftests_bor_backend import BOR_BACKEND_SELFTEST_CASES
    from .selftests_bor_fem import BOR_FEM_SELFTEST_CASES
    from .selftests_bor_orders import BOR_ORDER_SELFTEST_CASES
    return BOR_SELFTEST_CASES + BOR_ORDER_SELFTEST_CASES + BOR_BACKEND_SELFTEST_CASES + BOR_FEM_SELFTEST_CASES


def register(registry):
    """Register em_fdfd_bor (after the FDTD / FDFD types: their material Params stay first per name), the
    'microdisk_bor_eigenmodes' template and the BOR selftest loader."""
    registry.register_physics(EM_FDFD_BOR)
    registry.register_template('microdisk_bor_eigenmodes', MICRODISK_LABEL, MICRODISK_DESCRIPTION,
                               microdisk_bor_eigenmodes_spec, summary=MICRODISK_SUMMARY)
    registry.register_selftests(_selftest_cases)

"""Model definition of 'em_fdfd_bor' (body-of-revolution FDFD: (r, z) half plane, azimuthal order
m) on definition_common, with its template and self-tests; definition.register calls register."""
import math
from dataclasses import replace

import numpy as np

from ...core import time_estimate as te
from ...core.constants import C0, FACE_LABELS
from ...core.domain import containing_cell
from ...core.memory_estimate import window_cell_count
from ...core.mesh import has_uniform_spacing
from ...model.definition_helpers import (AXISYMMETRIC, AXISYMMETRIC_SPACE_SETTINGS, CUDSS_LINEAR_ROLE, GEOMETRY_SETTING,
                                         axisymmetric_issues, eigen_factorization, eigen_role, linear_solver_cudss,
                                         plain_number)
from ...model.schema import FeatureType, Issue, Param, PhysicsType, SolverRole, feature_path
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS
from .body_of_revolution import BOR_FIELD_COMPONENTS, BOR_SOURCE_COMPONENTS, size_issue_for
from .fem_bor import FEM_BOR_ARNOLDI_BASIS_PER_WANTED, FEM_BOR_SIZE_REMEDY
from .fem_modes import DISCRETIZATION_ORDERS
from .bor_focused_beam import (BEAM_ORDER, QUADRATURE_FLOPS_PER_VALUE, aperture_sine, line_end_level,
                               theta_node_count)
from .bor_ports import bor_mode_monitor_result, bor_mode_source_result
from .definition_common import (BOTH_GRIDS, BOX_CELLS, DEFAULT_WAVELENGTH, EIGENFREQUENCY_STUDY, EM_FAMILY,
                                FOCUSED_BEAM_APERTURE_TEXT, FOCUSED_BEAM_FEATURE, FOCUSED_BEAM_LABEL,
                                FOCUSED_BEAM_SCAN_TEXT, FREQUENCY_DOMAIN_STUDY, HOST_DIRECT_NAMES, N_ANGLES_PARAM,
                                OPTOMECHANICS_STUDY, POINT_SOURCE_SIGN, SPAN_CELLS, _effective_material, _window_axes,
                                all_solved_wavelengths, apply_component_point_source, apply_focused_beam,
                                apply_far_field, apply_field_probe, apply_flux_monitor, apply_pec, apply_resonance_table,
                                brillouin_drive_time_items, brillouin_scan_solves, focused_beam_power_param,
                                resolved_solved_wavelengths,
                                boundary_settings_params, eigen_only_feature_issues, eigen_solved, em_material_params,
                                fem_resonance_time_estimate, fem_size_issues, first_solved_wavelength,
                                host_matrix_back_end_issues, kept_line_block_factors, make_far_field_result_callback,
                                make_validate_physics, mesh_kind, missing_source_issues, mode_line, object_result,
                                other_em_physics_labels, physics_counter, quasi_static_domain_requirements,
                                raw_settings, resonance_eigenpairs, resonance_table_study_issues, role_counter,
                                scaled_ops, scipy_issues, shortest_solved_wavelength, source_slot_docs,
                                unused_wavelength_issues, validate_driven_point, write_dispersive_materials)
from .definition_feature_validation import (_box_issues, _focused_beam_row_messages, _line_axes, _line_cells,
                                            _line_issues, _line_material_indices, _line_medium_index,
                                            _selected_shapes, _shape_issues, focused_beam_end_wavelengths,
                                            upstream_focus_messages, validate_line_feature)
from .eigenfrequency import resonance_filter_time_item
from .focused_beam import MIN_SPAN_CELLS, NORMAL_RANGE
from ...solvers.line_block import factor_solve_ops
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import (bor_focused_beam_memory_estimate, bor_memory_estimate, pec_memory_estimate,
                               search_single_basis)
from .renderer_bor import BORRenderer
from .selftests_bor import MICRODISK_DESCRIPTION, MICRODISK_LABEL, MICRODISK_SUMMARY, microdisk_bor_eigenmodes_spec
from .solvers.fdfd_bor import (BOR_MATERIAL_SCHEMA_KEY, BOR_PERMITTIVITY_ARRAYS, BOR_PERMITTIVITY_SLOTS, BORPhysics,
                               bor_far_field_result, bor_flux_result, bor_probe_result, bor_resonance_table_result)
from .solvers.fdfd_bor_fem import FEMBORPhysics

_TYPE_NAME = 'em_fdfd_bor'
_AXIS = 'axis'
_WALL_FACES = ('x_hi', 'y_lo', 'y_hi')
_WALL_CHOICES = ('cpml', 'pec', 'pmc')
_FIELDS_NOTE = ('Fields ~ exp(j (omega t - m phi)): shown phasors are the travelling wave at phi = 0; the standing wave '
                '(+m and -m, degenerate) has E_r, E_z ~ cos(m phi), E_phi ~ sin(m phi) with the same profiles.')

# ---- settings ----
_BC_HELP = (
    'x_lo is always the axis r = 0. Walls sit on the outer grid edge: pec = tangential E 0, pmc = tangential H 0; '
    'cpml adds pml_cells cells outside the face (stretch s = kappa + sigma / (alpha + j omega eps0), pml_* grading; '
    'on r_hi the stretched radius enters every metric factor), ending in a pmc edge. A pec / pmc z_lo face is the '
    'mirror plane of a structure symmetric about z = 0 (halves the unknowns): pmc keeps modes with E_r, E_phi even '
    '(quasi-TE whispering-gallery modes of a disk), pec those with E_z even (quasi-TM). No periodic faces.')
_BC_TIP = ('What happens to waves at this edge of the half plane.\n\n'
           '- Absorbing layer: waves leave without reflection, as into open space.\n'
           '- Perfect metal wall: a perfect mirror; on the bottom edge also the mirror plane of a structure symmetric '
           'about it, keeping the modes whose vertical electric field is symmetric.\n'
           '- Perfect magnetic wall: a perfect mirror of the opposite kind; on the bottom edge the mirror plane that '
           'keeps the modes whose electric field in the disk plane is symmetric.')
_WALL_CHOICE_LABELS = (('cpml', 'Absorbing layer'), ('pec', 'Perfect metal wall'), ('pmc', 'Perfect magnetic wall'))
_PML_VISIBLE = tuple((f'bc_{face}', ('cpml',)) for face in _WALL_FACES)
# geometry help of the (r, z) types (em_fdfd_bor, em_fdtd_bor)
GEOMETRY_HELP = ('Always axisymmetric: (r, z) half plane about the x_lo edge (x = r >= 0, y = z). Exists so coupled '
                 'heat / electric_currents / electrostatics see an axisymmetric partner (no planar-partner warning).')


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
          help=GEOMETRY_HELP,
          tip='The drawing is half of a cut through a round object, rotated about the left edge of the domain.'),
    *axis_wall_settings(_BC_HELP),
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help='Vacuum wavelength of a stationary solve; eigenfrequency: the target (resonances nearest it); '
               'frequency_domain ignores it (validation info when not the default). Mesh: >= 10 (better 20) cells per '
               'lambda0 / |n| and >= 2 per skin depth (validation checks both). Typical 0.4e-6 to 2e-6 m.',
          tip='Vacuum wavelength at which the field is solved, in metres: 1.55e-6 for 1550 nm.\n\n'
              "An 'Eigenfrequency' study looks for the resonances closest to it; a 'Frequency domain' study solves "
              'its own list of wavelengths.'),
    Param('azimuthal_order', 'int', 0, label='Azimuthal order', min=0, max=10000,
          help='m, fields ~ exp(-j m phi); +m and -m are degenerate (same resonances and |fields|). m = 0: rotationally '
               'symmetric, TE-like E_phi, H_r, H_z and TM-like E_r, E_z, H_phi decouple (on-axis dipoles, radial '
               'modes); m = 1: on-axis focused / plane waves, fundamental fibre and vertical-cavity modes (E_r, E_phi, '
               'H_phi non-zero on the axis); larger m: whispering-gallery modes, m ~ 2 pi R n_eff / lambda. Several '
               'orders (free spectral range, the m of a resonance, an off-axis point source): azimuthal_orders.',
          tip='Number of field oscillations around the axis.\n\n'
              '0 for fields that are the same all around (an antenna on the axis), 1 for light focused along the '
              'axis, large values (tens) for whispering-gallery modes running around a disk or ring edge.'),
    Param('azimuthal_orders', 'float_list', None, label='Azimuthal order list', unit='1', min=0.0, max=10000.0,
          optional=True,
          help='Distinct integer orders m >= 0 in one run (one material read, one assembly A(m) = K0 + m K1 + m^2 K2 - '
               'k0^2 M); a non-empty list replaces azimuthal_order, null or [] = azimuthal_order alone. Eigenfrequency: '
               'n_resonances per order near the wavelength setting, numbered order by order in list order; snapshot '
               "labels 'resonance k: m = <m>, lambda = ...'; resonance_table column azimuthal_order (lambda versus m: "
               'the free spectral range). Every study that does not eigen-solve the physics (stationary, '
               'frequency_domain, and as a background physics of e.g. frequency_response or optomechanics): the '
               'driven order sum, every point_source a 3D '
               'point current at phi = 0 (see point_source) expanded in exp(-j m phi), each m > 0 solved with -m (one '
               'factor: A(-m) is the phi mirror) and summed. Fields and field_probe show the phi = 0 half plane of '
               'the 3D field (their ring-weighted integrals are not 3D integrals); absorbed_power (q_abs, handed to '
               'heat) is its phi average; flux_monitor and far_field add the orders exactly and report each. List '
               'every order 0..M: far field and powers converge once M exceeds k rho_s by a few (rho_s = source '
               'radius, k = n omega / c), near fields need more. mode_source / mode_monitor / focused_beam need a '
               'single order (validation error). Cost: one factorisation per order and wavelength; a line_block '
               'shifted solver factorises the orders of one axis class (|m| >= 1) in one batch.',
          tip="Several azimuthal orders solved in one run, e.g. 14, 15, 16 or 0, 1, 2, ..., 8.\n\n"
              "- Eigenfrequency study: replaces 'Azimuthal order' for the resonance search; the resonance table then "
              'lists the order of each resonance.\n'
              '- Stationary or frequency-domain study: a point source becomes a true point source beside the axis '
              '(not a ring); the field is summed over all listed orders, so list every order from 0 up to a few more '
              'than 2 pi x source radius / wavelength.'),
    Param('discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATION_ORDERS),
          choice_labels=(('quadratic', 'Quadratic'), ('linear', 'Linear')),
          help='Element order on a triangles mesh (tensor meshes use the Yee cell). Resonance studies only: mixed '
               'axisymmetric elements (Nedelec for E_r, E_z, Lagrange for r E_phi, same degree: no spurious modes); '
               'quadratic maps curved edges isoparametrically, so disks, toroids and spheres follow the outline. '
               'linear (Whitney / P1): cheaper, error ~h^2 instead of ~h^4 in k^2, the axis element layer stiff for '
               'm = 0 / |m| = 1. ~7 (quadratic) or 2 (linear) unknowns per element; shift-invert by the eigen solver '
               f'(host SuperLU; factorization cudss: GPU LDL^T; {HOST_DIRECT_NAMES}: host MUMPS / PARDISO LDL^T, '
               '_mixed = complex64 + float64 Rayleigh quotients). Materials one value per element; a cpml face gets '
               'mesh pml_layers element layers (pml_* stretch) and a pec outer side; pec_domain removes its elements '
               '(tangential E = 0 on their outline). On triangles stationary / frequency_domain and every source / '
               'monitor feature are validation errors; resonance_table and pec_domain work.',
          tip='Order of the finite elements when the mesh is made of triangles: quadratic is far more accurate per '
              'element, linear is cheaper.\n\nUnused on a rectangular-cell mesh.'),
)

# ---- material vocabulary ('modes' table: eps_r -> all three arrays; drawing-plane names, out of plane = phi) ----
_COMPLEX_EPS = ('Solved with the complex value at the solve wavelength: this - j sigma_e / (omega eps0) + chi(omega) '
                'of every Drude / Lorentz pole (exp(+j omega t): loss Im < 0).')
_POINT_VALUE = ('E_phi sees the cell value, E_r / E_z (cell edges) the ring-volume / length weighted mean of their two '
                'cells (no subpixel smoothing).')
# the cylindrical component names, said once in the eps_r help of em_fdfd_bor and em_fdtd_bor
_COMPONENT_NAMES = ('Diagonal cylindrical components (rotationally invariant): eps_xx = eps_rr (E_r), eps_yy = axial '
                    'eps_zz (E_z; drawing y = z), eps_zz = eps_phiphi (E_phi; writes the eps_r array). z-cut uniaxial '
                    'crystal (optic axis on the rotation axis): eps_r = n_o^2, eps_yy = n_e^2; radially anisotropic '
                    'shell: eps_xx.')
# component helps of em_fdfd_bor and em_fdtd_bor
COMPONENT_HELPS = {
    'eps_zz': ('eps_phiphi, the only component E_phi sees (stored in the eps_r array, overriding eps_r for E_phi '
               'only); the axial component is eps_yy.'),
    'eps_xx': 'eps_rr, the only component E_r sees.',
    'eps_yy': 'Axial eps_zz (drawing y), the only component E_z sees.',
}
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
    'eps_r': ('Isotropic, all three components; n^2 of a lossless dielectric (silicon ~12.1, silica ~2.1 at 1.55 um); '
              f'a component property wins over it. {_COMPONENT_NAMES} {_POINT_VALUE} {_COMPLEX_EPS}'),
    **COMPONENT_HELPS,
    'sigma_e': ('Every component gains -j sigma_e / (omega eps0); weak absorber of index n, extinction kappa: sigma_e = '
                '2 n kappa omega eps0. Metals: a Drude pole instead.'),
    'mu_r': ('Real, 1 for optical materials; H_r, H_phi, H_z take the mean of the cells around their points. Solved as '
             'mu_r - j sigma_m / (omega mu0).'),
    'sigma_m': 'mu_r gains -j sigma_m / (omega mu0) (artificial absorbers only).',
})

# ---- features ----
_POINT_SOURCE_PARAMS = (
    Param('component', 'choice', 'z', label='Direction', choices=BOR_SOURCE_COMPONENTS,
          choice_labels=(('r', 'Radial'), ('phi', 'Azimuthal (around the axis)'), ('z', 'Axial (vertical)')),
          help="'phi': J_phi = I / (w h) in the cell (m = 0: a current loop of radius r carrying I, a magnetic dipole "
               "on the axis); 'z' / 'r': current I through the cell's ring cross-section (2 pi r w axial, 2 pi r h "
               "radial), split onto its two edge points (axis cell: 'z' an on-axis electric dipole of moment I h, h = "
               "cell height; 'r' at |m| = 1 the rotating transverse dipole of moment I w / 2). For m != 0, I is the "
               'amplitude of the exp(-j m phi) distribution. In a driven order sum (azimuthal_orders) instead a 3D '
               "point current at phi = 0 ('phi' points out of the drawing) of moment I d (d = cell height for z, width "
               'for r / phi): each order carries ring current I (r, z) or I d / (2 pi r) (phi); in free space it '
               'radiates eta0 k^2 (I d)^2 / (12 pi) wherever it sits.',
          tip='Direction of the current: around the axis (a loop), along the axis or radially.\n\n'
              'Along the axis in the first column of cells is a dipole antenna on the axis.'),
    Param('current', 'float', 1e-3, label='Current', unit='A',
          help=f'Current I of the element (see component); fields are linear in it. {POINT_SOURCE_SIGN}',
          tip='Peak current of the source. Fields scale linearly with it.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Phasor phase of the current (0 = cos carrier, as the em_fdfd_* sources).',
          tip='Phase of the source in degrees; it matters only with several sources.'),
)
_PROBE_PARAMS = (
    Param('component', 'choice', 'ez', label='Field component', choices=BOR_FIELD_COMPONENTS,
          choice_labels=(('er', 'E_r'), ('ephi', 'E_phi'), ('ez', 'E_z'), ('hr', 'H_r'), ('hphi', 'H_phi'),
                         ('hz', 'H_z')),
          help='Complex phasor at the centre of the containing cell (edge points averaged there), per driven solve.',
          tip='Field recorded at the point; electric fields in V/m, magnetic fields in A/m.'),
)
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency solve (per order of azimuthal_orders, order by order), nearest the target "
    "(wavelength setting) first; accurate only near it. Result 'table', one row per resonance: resonance_number, "
    'wavelength = 2 pi C0 / Re(omega) [m], frequency / frequency_imag = Re / Im(omega) / (2 pi) [Hz] (> 0 decays), '
    'q_factor = Re(omega) / (2 Im(omega)) (inf when lossless and closed), pml_fraction (share of the ring-volume '
    '|E|^2 in PML cells; > 0.5 dropped), azimuthal_order; extras target_wavelength [m], selected_resonance, '
    "azimuthal_orders. Q versus a parameter: derived value source '<physics>.<this feature>', expression q_factor, "
    'operation at, at = 0.')


def _validate_resonance_table(vctx, physics_spec, feature_spec, values):
    """Warning where the study does not eigen-solve the physics: the table has no solve to show."""
    return resonance_table_study_issues(vctx, physics_spec, feature_spec)


# ---- flux, far field and ports ----
_FAR_FIELD_PARAMS = (replace(N_ANGLES_PARAM,
                             help='Angles evenly spaced over 360 degrees in the meridional (drawing) plane, '
                                  'counter-clockwise from +r (0 = equator, pi/2 = +z, pi = equator at phi = pi).',
                             tip='Number of directions, evenly spaced over the full circle of the drawing plane, '
                                 'counter-clockwise from the right (radial) direction.\n\n360 gives 1-degree steps.'),)
_MODE_PARAMS_TEXT = ('Port line: horizontal (z = const; vertical lines are refused), from the axis (x0 = 0) or annular '
                     '(x0 > 0: a pec end on that cell edge, e.g. from a coaxial inner conductor); it ends on the last '
                     'window cell (continuing through the r_hi cpml, or on its pec / pmc wall) or inside the domain '
                     '(again a pec end: extend it well into the cladding; the run log warns when the mode keeps 1% of '
                     'its peak there; exact when the cells beyond are pec). Modes ~ exp(-j beta z) are full-vector '
                     'solutions of the BOR operator on the line cells, re-solved at every wavelength and material '
                     'change: exact for a guide invariant along z over a cell on each side of the line (run log warns '
                     'otherwise). Ranked by descending Re n_eff (mode 1 highest), radiation modes with > half their '
                     'energy in the PML dropped; each carries 1 W (the flux_monitor flux) in the travelling '
                     'exp(-j m phi) wave.')
_PORT_POWER_PARAMS = (
    Param('power', 'float', 1.0, label='Power', unit='W', min=0.0,
          help='Mode power P through the port in the travelling exp(-j m phi) wave (its cos(m phi) standing-wave half '
               'carries P / 2 for m != 0); fields scale with sqrt(P).',
          tip='Power carried by the launched mode, in watts. Fields grow with its square root, so 1 is the usual '
              'choice.'),
    Param('mode_number', 'int', 1, label='Mode number', min=1, max=50,
          help='1 = highest Re n_eff at this order (HE11 of a fibre at m = 1, TE11 of a metal pipe); the result table '
               'lists the modes found.',
          tip='Which guided mode to launch: 1 is the one with the highest effective index.'),
    Param('direction', 'choice', 'positive', label='Direction', choices=('positive', 'negative'),
          choice_labels=(('positive', 'Upward (+z)'), ('negative', 'Downward (-z)')),
          help="'positive': toward +z (total field on the line row and above, only what comes back below); "
               "'negative': toward -z (total field on the row and below).",
          tip='Direction along the axis in which the mode is sent.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Mode amplitude sqrt(P) e^{j phase} on the line row (each mode phased with its largest field sample '
               'real and positive).',
          tip='Phase of the launched mode in degrees; it matters only with several sources.'),
)
_MODE_MONITOR_PARAMS = (
    Param('mode_number', 'int', 1, label='Mode number', min=1, max=50,
          help='Guided mode measured (ranked as mode_source); later wavelengths follow it by overlap.',
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
    'Time-averaged power [W] through an axis-aligned line per driven solve, ring-weighted for the travelling '
    'exp(-j m phi) wave (a cos(m phi) standing wave carries half for m != 0). Horizontal line (z = const): flux along '
    '+z through the swept annulus, pi sum [A Re(E_r H_phi*) - r w Re(E_phi H_r*)]; vertical (r = const): along +r '
    'through the cylinder, pi sum [r h Re(E_phi H_z*) - r L Re(E_z H_phi*)]. E of the line cells, H on their hi face '
    '(Yee points: exactly conserved through lossless cells, the port-mode normalisation); span ends by their share. A '
    "line from the axis across the field gives the full power, a closed box the radiated power. Result "
    "'flux_spectrum' [W], one entry per wavelength, sorted. Driven order sum: power of the 3D field (orders add "
    'exactly), extras power_orders (n_wavelengths x n_orders, each m with -m) and azimuthal_orders. The line covers '
    f'{SPAN_CELLS}; its hi face must be inside the grid.')
_FAR_FIELD_DESCRIPTION = (
    'Near-to-far-field transform of the driven solution on the closed surface of a rectangle in the (r, z) drawing (z '
    'faces, outer cylinder, inner cylinder when x0 > 0; a bottom at y0 = 0 on a pec / pmc z_lo mirror is closed by the '
    'images: E_r, E_phi, H_z even for pmc, odd for pec): J = n x H, M = -n x E integrated over phi in closed form '
    '(Bessel J_(m-1), J_m, J_(m+1)); vacuum outside the box (enclose every scatterer and source). Result '
    "'far_field', one row per wavelength: power = r^2 |E|^2 / (2 eta0) [W/sr] versus meridional angle "
    'counter-clockwise from +r (0 = equator, pi/2 = +z; cos < 0 = the phi = pi side, the same power for one order); '
    '2 pi int power cos(angle) d angle over -pi/2 .. pi/2 = the radiated power; extras power_theta / power_phi, '
    'angle_axes, azimuthal_order. Driven order sum: the pattern of the summed 3D field in the plane through the '
    'sources (cos >= 0 at phi = 0, the others at phi = pi: no longer equal); extras power_azimuthal_mean (phi average; '
    'its integral as above is the radiated power), power_orders (n_wavelengths x n_orders x n_angles, each m with -m), '
    'azimuthal_orders instead of azimuthal_order. Needs one cell beyond x1 / y1 and below y0 inside the domain; the '
    f'box covers {BOX_CELLS}.')
_MODE_TABLE_TEXT = ("Result 'table' of the latest port solve (frequency_domain: the last wavelength), one row per mode "
                    'found (up to max(mode_number + 2, 4)): mode_number, n_eff, n_eff_imag (< 0 = loss), '
                    'loss_db_per_cm, n_group (pencil derivative, no material dispersion), edge_field (|field| at the '
                    'line end / peak), selected; extras wavelength [m], power [W], direction, mode_number, phase [deg].')
_MODE_MONITOR_RESULT_TEXT = (
    "Result 'xy' versus vacuum wavelength: power_positive / power_negative = |a|^2 [W] toward +z / -z (discrete "
    'reciprocity of the line row and the row above: other modes drop out); with exactly one enabled mode_source also '
    'fraction_positive / fraction_negative = |a / (sqrt(P) e^{j phase})|^2 (transmission / reflection into this mode) '
    'and phase_positive / phase_negative (rad, not de-embedded); extras frequency, wavelength, n_eff, n_eff_imag, '
    'mode_number, normalized. Keep the line and the row above off the line rows of a mode_source.')

# ---- focused beam (bor_focused_beam; the drive of brillouin_scattering) ----
# beam |E| at the line's r end / its peak on the line above which validation warns: a hard pupil edge leaves a slowly
# decaying side-lobe tail (unlike the 2D Gaussian's 1e-6), and the cut edge leaks ~ the square of this level in power
# but biases the focal field linearly (~1.5x the level: em_bor_focused_beam_debye E_z 1.2 % at a level ~8e-3)
_BEAM_END_LEVEL = 1e-2
_BEAM_DIRECTION_LABELS = (('positive', 'Upward (+z)'), ('negative', 'Downward (-z)'))
_FOCUSED_BEAM_PARAMS = (
    focused_beam_power_param(
        'W', 'Here the power of the linearly (x-) polarised beam, the azimuthal order 1 standing wave E_r, E_z ~ '
             'cos(phi), E_phi ~ sin(phi) that the solve represents: the exact discrete Poynting flux of the injected '
             'incident field through the interface face with the cos^2 / sin^2 ring weight pi. flux_monitor weights '
             'the travelling exp(-j phi) wave with 2 pi, so downstream in a uniform lossless medium it reads 2 x '
             'power (minus what the structure or the PML send back and the launch error).',
        'Power carried by the light beam, in watts. Field strengths scale with its square root, so 1 is typical.\n\n'
        'A flux monitor of this physics shows twice this value: it counts the rotating wave the solve stands for.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Phase of E_x (= E_r at phi = 0) at the focus: all angular components of the beam are in phase there.',
          tip='Phase of the beam at its focus in degrees; it matters only with several sources.'),
    Param('numerical_aperture', 'float', None, label='Numerical aperture', unit='1', min=0.0, min_exclusive=True,
          help=f'{FOCUSED_BEAM_APERTURE_TEXT}; here strictly below n (theta_max < 90 deg): the cone half-angle of the '
               'focused light, sin(theta_max) = NA / n. Typical 0.3 to 1.2 in water (n 1.33). With a fully filled '
               'pupil the focus is an Airy spot of FWHM ~ 0.5 lambda0 / NA whose side lobes decay slowly along the line '
               f'(validation warns when the beam keeps {_BEAM_END_LEVEL:g} of its peak at the line end; a smaller '
               'fill_factor softens the edge).',
          tip='Sine of the half-angle of the cone of focused light times the refractive index, as on a microscope '
              'objective.\n\nIt must stay below the refractive index at the line.'),
    Param('fill_factor', 'float', 1.0, label='Pupil fill factor', unit='1', min=0.0, min_exclusive=True,
          help='f0 = Gaussian waist of the beam entering the objective / pupil radius: pupil field l(theta) = '
               'exp(-(sin(theta) / (f0 sin(theta_max)))^2) with the aplanatic sqrt(cos(theta)) (Richards-Wolf, '
               'Novotny-Hecht eq. 3.52); the aperture edge keeps exp(-1 / f0^2) of the centre field (1: 0.37). Large '
               'f0: a filled pupil, the smallest focus; small f0: an underfilled pupil, a Gaussian focus of effective '
               'NA ~ f0 NA with weak side lobes. Typical 0.5 to 2.',
          tip='How much of the objective opening the incoming beam fills: 1 fills it to its edge, smaller values '
              'leave the rim dark and give a wider, smoother focus.'),
    Param('focus_z', 'float', None, label='Focus z', unit='m',
          help='z coordinate [m] of the focus on the axis in the drawing frame (domain bottom edge at 0): every '
               'angular component of the beam is in phase there, so |E| peaks there in a uniform medium. It may lie '
               'on either side of the line (upstream: a beam diverging from a virtual focus) and outside the domain. '
               'A brillouin_scattering scan moves it along z by scan_offsets_y (x offsets would leave the axis: '
               'refused).',
          tip='Height of the focus on the rotation axis, where the beam is narrowest and brightest, in metres.\n\n'
              'It may lie on either side of the line.'),
    Param('direction', 'choice', 'positive', label='Direction', choices=('positive', 'negative'),
          choice_labels=_BEAM_DIRECTION_LABELS,
          help="'positive': toward +z (total field on the line row and above, only what comes back below); "
               "'negative': toward -z (total field on the row and below). Nothing is launched the other way.",
          tip='Direction along the axis in which the beam travels from the line.\n\nNothing is sent the other way.'),
)
_FOCUSED_BEAM_DESCRIPTION = (
    'One-sided focused beam on the axis launched from a horizontal line (z = const) that starts on the axis: the Debye '
    '(Richards-Wolf) field of an aplanatic lens focusing an x-polarised Gaussian pupil (numerical_aperture, '
    'fill_factor) to (r = 0, focus_z), in the homogeneous isotropic medium of the line. An x-polarised beam on the '
    'axis is exactly the azimuthal order 1 standing wave (E_r, E_z ~ cos(phi), E_phi ~ sin(phi)), so the physics needs '
    'azimuthal_order 1 and no azimuthal_orders list (validation error); shown fields and probes are the travelling-'
    'wave phasors of this type: E_r, E_z the beam at phi = 0, E_phi = j x its sin(phi) amplitude. '
    'Its incident field is written on the two z rows across the interface (one-sided TFSF as mode_source; the '
    'interface is the lo face of the line row for positive, its hi face for negative): the z dependence of each '
    'angular component is the Yee-exact one, so what leaks upstream comes only from the radial discretisation of the '
    'continuum Bessel profiles (second order, ~(k0 NA h)^2: largest at high NA; >= 25 cells per wavelength in the '
    'medium recommended). power is defined, not calibrated (see power; a flux_monitor reads 2 x power). The line '
    'needs uniform r cells along it and uniform z rows across it (two on either side, inside the domain), one '
    'homogeneous isotropic medium on those rows (no PEC there; loss allowed, warning: the beam then decays and power '
    'is that at the line) and cpml on both z faces (a pec / pmc z face would mirror the beam onto itself: error); it should hold the beam (above '
    f'{_BEAM_END_LEVEL:g} of its peak at its r end: warning). {FOCUSED_BEAM_SCAN_TEXT} Tensor meshes only. The line '
    f'covers {SPAN_CELLS}.')
_FOCUSED_BEAM_SUMMARY = ('Sends a focused light beam along the rotation axis from a horizontal line, like light '
                         'focused by a microscope objective, with a set power and its focus at a chosen height. Nothing '
                         'is sent backward. It is the illumination of axisymmetric Brillouin scattering spectra.')
_ANISOTROPY_PROPERTIES = ('eps_xx', 'eps_yy', 'eps_zz')   # components a focused beam's line medium may not set apart
_LOSS_PROPERTIES = ('sigma_e', 'sigma_m')


def _focused_beam_physics_issues(vctx, physics_spec, path):
    """Errors of the physics settings a focused beam refuses: azimuthal_order other than 1, an azimuthal_orders list
    wherever the study does not eigen-solve the physics (the create's test: the list is then the driven order sum, also
    as a background physics of frequency_response / optomechanics), a pec / pmc z face."""
    settings = raw_settings(physics_spec)
    issues = []
    raw_order = settings.get('azimuthal_order', 0)
    order = None if isinstance(raw_order, bool) else plain_number(raw_order, vctx.parameters)   # None: unresolved
    if isinstance(raw_order, bool) or (order is not None and order != BEAM_ORDER):
        issues.append(Issue('error', path, f'needs azimuthal_order {BEAM_ORDER} (got {raw_order!r}): an x-polarised '
                                           f'beam on the axis is the azimuthal order {BEAM_ORDER} standing wave'))
    if settings.get('azimuthal_orders') and not eigen_solved(vctx.spec, physics_spec.name):
        issues.append(Issue('error', path, f'cannot join the driven order sum of azimuthal_orders (the '
                                           f'{vctx.study_type} study solves {physics_spec.name} driven, and the beam '
                                           f'is one order {BEAM_ORDER} field): clear azimuthal_orders'))
    for face in ('y_lo', 'y_hi'):
        if settings.get(f'bc_{face}', 'cpml') in _MIRROR_FACES:
            issues.append(Issue('error', path, f"bc_{face} = {settings[f'bc_{face}']} is the mirror plane of a "
                                               'z-symmetric structure, but a beam along z is not symmetric about any z '
                                               'plane: use cpml on both z faces (a reflector: a pec_domain)'))
    return issues


def _line_medium_messages(vctx, position, span, neighbours):
    """(level, text) for line-row materials (material preview) with cylindrical anisotropy (error: the Debye beam is
    the field of an isotropic medium) or loss (warning)."""
    found = _line_material_indices(vctx, 'x', position, span, None, 'ez', neighbours)
    if found is None:
        return []
    messages = []
    indices, _ = found
    for index in sorted(index for index in indices if index > 0):
        material = vctx.spec.materials[index - 1]
        try:
            properties, dispersion = _effective_material(material)
        except ValueError:   # reported at the material
            continue

        def value_of(name, default):
            if properties.get(name) is None:
                return default
            value = plain_number(properties[name], vctx.parameters)
            return math.nan if value is None else value   # field- / state-dependent: unknown here (the run checks)

        def differs(value, reference):
            return math.isfinite(value) and math.isfinite(reference) and value != reference
        eps = value_of('eps_r', 1.0)
        anisotropic = [name for name in _ANISOTROPY_PROPERTIES if properties.get(name) is not None
                       and differs(value_of(name, eps), eps)]
        if anisotropic:
            messages.append(('error', f"has material '{material.name}' with {', '.join(anisotropic)} different from "
                                      'eps_r on its rows: the Debye beam is the field of one isotropic medium; keep the '
                                      'line in an isotropic medium'))
        if dispersion or any(differs(value_of(name, 0.0), 0.0) for name in _LOSS_PROPERTIES):
            messages.append(('warning', f"has the lossy or dispersive material '{material.name}' on its rows: the "
                                        'beam decays from the line on, power is its value at the line and the Debye '
                                        'field is that of the complex index (a brillouin_scattering normalisation '
                                        'then refers to the line, not the focus)'))
    return messages


def _focused_beam_line_messages(vctx, physics_spec, line, values, physics_type, wavelengths):
    """(level, text) of one focused-beam line inside the domain (_validate_focused_beam); the cell checks need the
    mesh estimate, the beam checks the material preview."""
    line_axes = _line_axes(line, vctx.domain_size)
    if line_axes is None:
        return []   # reported by _line_issues
    along_axis, _, position, (start, end) = line_axes
    if along_axis != 'x':
        return [('error', 'must be horizontal (z = const): the beam travels along the rotation axis')]
    messages = []
    mesh = vctx.mesh
    if mesh is None and start > 1e-9 * vctx.domain_size[0]:
        messages.append(('error', f'starts at r = {start:.4g} m: it must start on the axis (x0 = 0), the beam is '
                                  'focused on it'))
    if mesh is not None and getattr(mesh, 'kind', 'tensor') == 'tensor':   # triangles: refused by the grids
        (r_widths, r_edges, r_first, r_last), (z_widths, z_edges, z_first, z_last) = _window_axes(mesh)
        start_cell = containing_cell(start, r_edges, r_first, r_last)
        end_cell = containing_cell(end, r_edges, r_first, r_last) + 1
        if start_cell != 0:
            messages.append(('error', f'starts at r = {start:.4g} m (r cell {start_cell}): it must start on the axis '
                                      '(x0 = 0), the beam is focused on it'))
        elif end_cell < MIN_SPAN_CELLS:
            messages.append(('error', f'covers {end_cell} r cell(s); a focused beam needs at least {MIN_SPAN_CELLS} '
                                      '(and should hold the whole beam)'))
        elif not has_uniform_spacing(r_widths[:end_cell]):
            messages.append(('error', f'has non-uniform r cells ({float(np.min(r_widths[:end_cell])):.4g} .. '
                                      f'{float(np.max(r_widths[:end_cell])):.4g} m); keep the line in a uniform part '
                                      'of the mesh'))
        row = containing_cell(position, z_edges, z_first, z_last)
        rows = np.arange(row - NORMAL_RANGE, row + NORMAL_RANGE + 1)
        if rows[0] < z_first or rows[-1] > z_last:
            messages.append(('error', f'lies within {NORMAL_RANGE} rows of the z window edge: the launch reads '
                                      f'{NORMAL_RANGE} uniform rows on either side of the line; move it inside'))
        elif not has_uniform_spacing(z_widths[rows]):
            messages.append(('error', f'lies where the z cell heights change ({float(z_widths[rows].min()):.4g} .. '
                                      f'{float(z_widths[rows].max()):.4g} m over the {NORMAL_RANGE} rows either '
                                      'side); keep the line in a uniform part of the mesh'))
        else:
            neighbours = [0.5 * (z_edges[other] + z_edges[other + 1]) for other in rows if other != row]
            band = (float(z_edges[rows[0]]), float(z_edges[rows[-1] + 1]))
            messages += _focused_beam_row_messages(vctx, physics_spec, 'x', position, (start, end), neighbours, band)
            messages += _line_medium_messages(vctx, position, (start, end), neighbours)
    aperture_error, end_messages, complete = _beam_end_messages(vctx, position, (start, end), values, physics_type,
                                                                wavelengths, values['focus_z'])
    if aperture_error is not None:
        return messages + [('error', aperture_error)]
    messages += end_messages
    if not complete:
        return messages
    sign = 1.0 if values['direction'] == 'positive' else -1.0
    return messages + upstream_focus_messages(sign, values['focus_z'], position, 'z')


def _beam_end_messages(vctx, position, span, values, physics_type, wavelengths, focus_z):
    """(numerical_aperture error text or None, [(level, text)] of the line-end check, complete) of a BOR focused-beam
    line for the focus at drawing-frame z = focus_z [m]: the continuum beam at focused_beam_end_wavelengths; complete
    False after a quadrature error (an error message: the line is far too long for the beam); nothing without a known
    line index or aperture."""
    end = span[1]
    index = _line_medium_index(vctx, 'x', position, span, physics_type)
    aperture = values['numerical_aperture']
    sign = 1.0 if values['direction'] == 'positive' else -1.0
    if aperture is None or not math.isfinite(index):
        return None, [], True
    try:
        aperture_sine(aperture, index)
    except ValueError:
        return (f'has numerical_aperture {aperture:g}, not below the index {index:.4g} of the medium at the line (NA = '
                'n sin(theta_max), theta_max < 90 deg); lower it'), [], False
    worst = 0.0
    for wavelength in focused_beam_end_wavelengths(wavelengths):
        try:
            level = line_end_level(2.0 * math.pi * float(C0) / wavelength, index, aperture, values['fill_factor'],
                                   end, position - focus_z, sign)
        except ValueError as exc:   # a line far too long for its quadrature
            return None, [('error', str(exc))], False
        worst = max(worst, level) if math.isfinite(level) else worst
    if worst <= _BEAM_END_LEVEL:
        return None, [], True
    return None, [('warning', f'stops at r = {end:.4g} m where the beam is still {worst:.2g} of its peak on the line '
                              f'(above {_BEAM_END_LEVEL:g}; continuum beam at the solved wavelengths), so the cut '
                              'edge launches a diffracted wave and light leaks upstream; lengthen the line, or lower '
                              'fill_factor (softer pupil edge)')], True


def bor_focused_beam_scan_messages(vctx, along_axis, position, span, values, physics_type, wavelengths, offset):
    """[(level, text)] of the line-end check of one BOR focused-beam line (resolved feature values) for its focus moved
    by the drawing-frame offset (0, dz) [m]: definition_feature_validation.focused_beam_scan_messages (the public entry
    point); [] for a vertical line or a radial offset (refused elsewhere) and after a numerical_aperture error."""
    dr, dz = offset
    if along_axis != 'x' or dr != 0.0:
        return []
    _, end_messages, _ = _beam_end_messages(vctx, position, span, values, physics_type, wavelengths,
                                            values['focus_z'] + dz)
    return end_messages


def _validate_focused_beam(vctx, physics_spec, feature_spec, values):
    """Line axis-aligned inside the domain; the physics settings it refuses (_focused_beam_physics_issues); per line
    the placement, medium, aperture and line-end checks (_focused_beam_line_messages: errors first for what
    bor_focused_beam_line / BORFocusedBeam refuse at run time); the eigen-solved warning."""
    path = feature_path(physics_spec, feature_spec)
    issues = (_line_issues(vctx, physics_spec, feature_spec)
              + eigen_only_feature_issues(vctx, physics_spec, feature_spec)
              + _focused_beam_physics_issues(vctx, physics_spec, path))
    physics_type = vctx.registry.physics_type(physics_spec.type)
    wavelengths = resolved_solved_wavelengths(vctx, physics_spec, physics_type)
    return issues + _shape_issues(
        vctx, physics_spec, feature_spec, 'line', lambda shape: _focused_beam_line_messages(
            vctx, physics_spec, shape.values, values, physics_type, wavelengths), needs_mesh=False)


_FEATURES = (
    FeatureType('point_source', 'Point source', 'source', 'point', params=_POINT_SOURCE_PARAMS,
                object_kinds=('point',), max_entities=1,
                description='Current element in the cell containing the point (x = r, y = z): a ring in 3D with the '
                            'exp(-j m phi) dependence; see component and current.',
                summary='A small current at a point of the half plane, a ring around the axis in 3D; on the axis an '
                        'antenna.',
                apply=apply_component_point_source, validate=validate_driven_point),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='xy',
                description="Phasor of one component in the containing cell per driven solve (order sum: the summed "
                            "field at phi = 0). Result 'xy' versus wavelength: abs (component unit), phase (rad); "
                            'extras frequency, wavelength, component, unit.',
                summary='Records the field at one point for every solved wavelength.',
                apply=apply_field_probe, result=object_result(bor_probe_result), validate=validate_driven_point),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Rings in 3D: E_phi = 0 in cells whose centre lies in the domains, E_r / E_z = 0 on '
                            'edges between two such cells (staircased). Triangles: its elements leave the unknowns, '
                            'tangential E = 0 on their outline (curved outlines exact with quadratic).',
                summary='Turns the selected domains into a perfect metal, following the mesh cells.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0), grids=BOTH_GRIDS),
    FeatureType('mode_source', 'Mode source', 'source', 'line', params=_PORT_POWER_PARAMS, object_kinds=('line',),
                max_entities=1, result_kind='table',
                description=('Launches one guided mode of a guide along the axis (fibre, pillar, metal pipe, coax) '
                             'from a horizontal line in one direction: a one-sided TFSF interface between the line row '
                             f'and the row upstream, exact on a straight guide. {_MODE_PARAMS_TEXT} {_MODE_TABLE_TEXT}'),
                summary='Launches one guided mode of a waveguide running along the axis, up or down, recomputed at '
                        'each solved wavelength. Pair it with mode monitors for transmission and reflection.',
                apply=_apply_mode_source, result=object_result(bor_mode_source_result),
                validate=_validate_line_monitor(port=True)),
    FeatureType(FOCUSED_BEAM_FEATURE, FOCUSED_BEAM_LABEL, 'source', 'line', params=_FOCUSED_BEAM_PARAMS,
                object_kinds=('line',), max_entities=1, description=_FOCUSED_BEAM_DESCRIPTION,
                summary=_FOCUSED_BEAM_SUMMARY, apply=apply_focused_beam, validate=_validate_focused_beam,
                memory_estimate=bor_focused_beam_memory_estimate),
    FeatureType('flux_monitor', 'Flux monitor', 'monitor', 'line', object_kinds=('line',), max_entities=1,
                result_kind='flux_spectrum', description=_FLUX_DESCRIPTION,
                summary='Measures the power crossing a horizontal disc or annulus (upward positive) or a vertical '
                        'cylinder (outward positive) at each solved wavelength, in watts.',
                apply=apply_flux_monitor, result=object_result(bor_flux_result),
                validate=_validate_line_monitor(port=False)),
    FeatureType('mode_monitor', 'Mode monitor', 'monitor', 'line', params=_MODE_MONITOR_PARAMS,
                object_kinds=('line',), max_entities=1, result_kind='xy',
                description=('Splits the field on a horizontal line into the upward (positive) and downward '
                             '(negative) amplitudes of one guided mode per wavelength. '
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
    'Body-of-revolution (BOR) FDFD: the drawing is the (r, z) half plane of a structure rotationally symmetric about '
    'the x_lo window edge (x = r >= 0, y = z), fields of one azimuthal order m (or an order list). '
    f'{_FIELDS_NOTE} All six components (coupled for m != 0) on the 3D cylindrical Yee cell projected onto the half '
    'plane (finite integration, exact ring areas / volumes): E_phi, H_r at cell centres (H_r on the +z edge), E_r, '
    'H_z on +r edges, E_z on +z edges, H_phi at the +r+z corner; on the axis only the m = 0 H_z unknown (|m| = 1: '
    'E_r, H_phi recovered from cell 0). True 3D results (disks, rings, pillars, staircased spheres, lenses, vertical '
    'cavities, on-axis dipoles) at 2D cost. Host matrix, 3 unknowns per cell incl. PML; needs scipy; validation warns '
    "near and refuses beyond the host factor size limit. Role 'linear': sparse_direct = host SuperLU, one factor per "
    'wavelength and material state, reused as GMRES preconditioner after a material change (thermo-optic '
    "iteration); back_end cudss / cudss_mixed (GPU backend, nvmath-python): cuDSS LDL^T on the GPU, no size limit, "
    f'its plan reused across wavelengths, material states and orders; {HOST_DIRECT_NAMES} (python-mumps / pydiso, '
    "either backend): MUMPS / PARDISO LDL^T on the host under the size limit; cusolver_qr is refused. Role 'eigen' "
    'arpack_shift_invert (tolerance, subspace_size; complex symmetric operator): factorization cudss = GPU LDL^T '
    f'without size limit, {HOST_DIRECT_NAMES} host LDL^T, _mixed = complex64 solves + float64 Rayleigh quotients. '
    "Role 'shifted': Automatic changes nothing; line_block factors every driven system and shift-invert operator on "
    'its backend (z rows or r columns; |m| >= 2 orders of a list in one batch; frequency_domain of one order: up to 8 '
    'wavelengths per batch within 60 % of free memory; mixed or double, refined against complex128); a solve it '
    'rejects goes to host SuperLU with a run-log warning. Studies: stationary (wavelength setting; with heat via '
    'heat_em_thermo_optic / em_heat_joule, heat axisymmetric too), frequency_domain (one solve per wavelength), '
    'eigenfrequency (shift-invert resonances nearest the wavelength setting: complex omega, Q from material loss and '
    'PML radiation), optomechanics (as the optical physics an optomechanical_coupling of an axisymmetric '
    'solid_mechanics names: the eigen-solve with n_optical_resonances, then the g0 table; triangles only; an '
    'em_fdfd_bor no coupling names solves as stationary background). Outputs at cell centres: components are Re of '
    'the phi = 0 phasors; absorbed_power = state q_abs = 1/2 omega eps0 sum_c (-Im eps_c) |E_c|^2 (mean over the '
    'component points; em_heat_joule hands it to heat); eps_r = eps_phiphi, eps_xx = eps_rr, eps_yy = axial eps_zz. '
    'Resonances are scaled to 1 real at the largest component (V/m). Field derived values and powers weight by the '
    'ring volume 2 pi r dr dz (travelling wave; a standing cos(m phi) wave has half). Cannot share a model with '
    'in-plane EM types (validation error). heat_em_thermo_optic writes all three permittivity arrays. Triangles '
    'mesh: eigenfrequency / optomechanics on mixed axisymmetric elements (see discretization); driven studies, '
    "sources and monitors need a tensor mesh. Instance usually 'em'.")


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
    return issues + missing_source_issues(vctx, physics_spec, path, ('point_source', 'mode_source',
                                                                     FOCUSED_BEAM_FEATURE),
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
    """azimuthal_orders: integers, distinct (errors); where the study does not eigen-solve the physics (the create's
    test: the list is the driven order sum, also as a background physics) no mode ports (error) and every order 0..max
    (warning: a point source's sum would miss terms)."""
    orders = settings_values.get('azimuthal_orders') or []
    if not orders:
        return []
    driven = not eigen_solved(vctx.spec, physics_spec.name)
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
# frequency_domain run). The memory estimate counts the stencil instead (memory_estimates._bor_entries_per_row: 19/3,
# 9); this 25 stays together with BOR_ASSEMBLY_BYTES_PER_ENTRY, which is priced per these entries.
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
    """bor_ports.solve_port_modes per enabled port and assembly (it re-solves per omega and assembly; ports on
    identical lines share one solve, which needs the line materials: counted per port, an upper bound): the three-row
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
                                    resonance_filter_time_item(context.backend, unknowns, nnz, eigenpairs * n_orders,
                                                               symmetric=True)))   # one search per order
        common = dict(unknowns=unknowns, nnz=nnz, value_bytes=16, solves=n_orders, eigenpairs=eigenpairs)
        if line_block:
            return te.combine((assembly, kept_line_block_factors(context, unknowns, nnz, n_orders, block),
                               context.solver_work('eigen', product_ops=scaled_ops(
                                   factor_solve_ops(unknowns, block, nnz, shifted.values.get('precision', 'mixed')),
                                   2.0), **common)))
        if eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS:   # squared Cayley: complex128 basis
            single_basis = search_single_basis(eigen_factorization(context.solvers), unknowns, True, True,
                                               spectral_map='squared_cayley')
            return te.combine((assembly, context.solver_work('eigen', factorizations=n_orders, rhs=2,
                                                             single_basis=single_basis, **common)))
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
    # brillouin_scattering scan positions: one more solve each on the kept factor (solve_moved_beam, single order),
    # once after the background stage; the focused beams' incident fields and the drive-field reads
    scan = float(brillouin_scan_solves(context.spec, context.name))
    assembly = te.combine((assembly, _focused_beam_time_estimate(settings_values, mesh, context, nnz, points + scan),
                           brillouin_drive_time_items(context, unknowns, nnz, window_cell_count(mesh), 3, host=True)))
    if line_block:
        return te.combine((assembly, context.solver_work('shifted', unknowns=unknowns, nnz=nnz, value_bytes=16,
                                                         solves=assemblies * n_orders + scan, solves_kind=kind,
                                                         rhs=solves_per_assembly / n_orders, block_size=block)))
    # a material change at the same omega: GMRES on the kept SuperLU factor (cuDSS refactorises on its kept plan)
    recycled = context.coupled and not listed and not linear_solver_cudss(context.solvers)
    return te.combine((assembly, context.solver_work(
        'linear', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=assemblies * solves_per_assembly + scan,
        solves_kind=kind, factorizations=points if recycled else assemblies * n_orders, factorizations_kind=kind,
        factor_scale=BOR_LU_SCALE, solve_scale=BOR_SOLVE_SCALE, backend='cpu')))   # host matrix on either backend


def _focused_beam_time_estimate(settings_values, mesh, context, nnz, builds):
    """Per enabled focused_beam: its unit slices `builds` times (one per solved wavelength and scan position;
    bor_focused_beam BORFocusedBeam._unit_slices): the Debye quadrature over the 2 (3 n_line + 1) slice keys (n_line
    <= the r cells; angular nodes theta_node_count of the phase span k0 NA (r + z extent), an upper bound of k (rho_max
    s_m + |d|_max (1 - cos theta_max)) since n (1 - cos theta_max) <= NA) and the interface-power product, plus per
    build one right-hand side (two operator products; host)."""
    beams = [feature for feature in context.physics_spec.features
             if feature.enabled and feature.type == FOCUSED_BEAM_FEATURE]
    if not beams:
        return te.TimeEstimate(())
    k0 = 2.0 * math.pi / float(settings_values['wavelength'])
    extent = float(mesh.x_edges[-1] - mesh.x_edges[0]) + float(mesh.y_edges[-1] - mesh.y_edges[0])
    keys = 2.0 * (3.0 * float(mesh.nx) + 1.0)
    values = 0.0
    for feature in beams:
        aperture = (feature.params or {}).get('numerical_aperture')
        aperture = float(aperture) if isinstance(aperture, (int, float)) and aperture > 0.0 else 1.0
        try:
            nodes = theta_node_count(k0 * aperture * extent)
        except ValueError:   # the run raises (validation reports it)
            nodes = 0
        values += keys * nodes
    ops = (te.DenseFlops(QUADRATURE_FLOPS_PER_VALUE * values, host=True),
           te.HostStream(3.0 * float(len(beams)) * _CSR_ENTRY_BYTES * float(nnz)), te.PythonCalls(80 * len(beams)))
    return te.TimeEstimate((te.TimeItem('focused beam incident fields (Debye quadrature) and right-hand sides',
                                        'solve', float(builds), 'exact', ops),))


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

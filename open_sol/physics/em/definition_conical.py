"""Model definition of 'em_fdfd_conical' (symmetry plan 2.9: three-component 2D FDFD of a z-invariant structure at a
fixed out-of-plane wave number k_z) on definition_common, with its self-tests; definition.register calls register."""
from dataclasses import replace

from ...core import time_estimate as te
from ...model.definition_helpers import CUDSS_LINEAR_ROLE, eigen_role, linear_solver_cudss
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, MIXED_FACTORIZATIONS
from ...solvers.sparse_direct import MIXED_BACK_ENDS
from .conical import COMPONENTS, FIELD_COMPONENTS, size_issue_for
from .fem_conical import FEM_CONICAL_SIZE_REMEDY
from .fem_modes import DISCRETIZATION_ORDERS, PERIODIC_TEXT
from .definition_bloch import bloch_active_axes, bloch_params, fdfd_bloch_issues, fdfd_domain_requirements
from .definition_common import (BOTH_GRIDS, CPML_BC_HELP, DEFAULT_WAVELENGTH, EIGENFREQUENCY_STUDY, EM_FAMILY,
                                FREQUENCY_DOMAIN_STUDY, PERIODIC_BC_HELP, POINT_SOURCE_SIGN, SPAN_CELLS, _window_axes,
                                all_solved_wavelengths, apply_component_point_source, apply_field_probe,
                                apply_flux_monitor, apply_pec, apply_resonance_table, boundary_settings_params,
                                eigen_backend_kwargs, eigen_only_feature_issues, em_material_params,
                                fem_resonance_time_estimate, fem_size_issues, first_solved_wavelength,
                                host_matrix_back_end_issues, make_validate_physics, mesh_kind, missing_source_issues,
                                mode_line, object_result, physics_counter, raw_settings, resonance_eigenpairs,
                                resonance_table_study_issues, scipy_issues, shortest_solved_wavelength,
                                source_slot_docs, unused_wavelength_issues, validate_driven_point,
                                write_dispersive_materials)
from .definition_feature_validation import _line_cells, _selected_shapes, validate_line_feature
from .eigenfrequency import resonance_filter_time_item, resonance_table_result
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import conical_memory_estimate, host_factorization, pec_memory_estimate
from .renderer_conical import ConicalRenderer
from .solvers.fdfd_conical import (CONICAL_MATERIAL_SCHEMA_KEY, CONICAL_PERMITTIVITY_ARRAYS,
                                   CONICAL_PERMITTIVITY_SLOTS, ConicalPhysics, conical_flux_result,
                                   conical_probe_result)
from .solvers.fdfd_conical_fem import FEMConicalPhysics

_TYPE_NAME = 'em_fdfd_conical'
_DRIVEN_STUDIES = ('stationary', FREQUENCY_DOMAIN_STUDY)

# ---- settings ----
_BC_HELP = ' '.join((
    CPML_BC_HELP,
    'In the frequency domain the CPML is the complex coordinate stretch s = kappa + sigma / (alpha + j omega eps0) '
    'with the FDTD grading of the pml_* settings.',
    PERIODIC_BC_HELP,
    'Walls and mirrors sit on the planes of em_modes (the same projected 3D Yee cell): a pec / pmc wall is the inner '
    'face of the outermost domain cell (that cell is a ghost holding the mirror image), half a cell off the em_fdfd_ez '
    'pec and em_fdfd_hz pmc walls; a symmetric (E mirror-even, a PMC plane) / antisymmetric (E mirror-odd, a PEC '
    'plane) mirror adds its ghost cell outside the domain, so its plane is the domain edge. At k_z != 0 a mirror plane '
    'normal to x or y stays a symmetry (k_z points along it) and halves the unknowns.',
))
_KZ_HELP = (
    'Out-of-plane wave number k_z in rad/m of every solve: the fields vary as exp(j (omega t - k_z z)) along the '
    'z-invariant structure (the industry-standard out-of-plane wave number), so all six components couple. k_z = 0 '
    'gives the decoupled em_fdfd_ez (E_z, H_x, H_y) and em_fdfd_hz (H_z, E_x, E_y) problems at once (walls half a cell '
    'apart from theirs). Uses: conical / oblique incidence on gratings and other z-invariant structures (k_z = k0 n '
    'sin(theta) of the z share of the incident wave vector, the in-plane share from bloch_kx / bloch_ky along a '
    'periodic axis), a waveguide cross-section driven at a given propagation constant beta = k_z (a source excites the '
    'guided modes resonantly where omega meets the mode dispersion), and in an eigenfrequency study the dispersion '
    'omega(k_z): the resonances at fixed k_z are the guided modes with beta = k_z (same discrete problem as em_modes, '
    'read the other way; sweep k_z for the band / dispersion diagram). k_z stays fixed over the wavelengths of a '
    'frequency_domain study (the incidence angle then changes with wavelength). |k_z| above k0 n in a material makes '
    'the field evanescent in it. A real number (no leaky / complex k_z through the model).')
_KZ_TIP = ('Wave number along the direction perpendicular to the drawing, in radians per metre: the fields vary as a '
           'wave of that wave number along that direction.\n\n'
           '0 gives the ordinary 2D problem. For light arriving with a component along that direction use 2 pi n '
           'sin(angle) / wavelength; for a waveguide whose cross-section is drawn it is the propagation constant, 2 pi '
           'n_eff / wavelength. Swept in an eigenfrequency study it gives the frequency versus wave number.')
_BLOCH_HELP = (
    'Bloch wave vector along {axis} in rad/m, used only when bc_{axis}_lo and bc_{axis}_hi are periodic: the solution '
    'is the Bloch wave F(x + L) = F(x) exp(-j k{axis} L) per period L (the meshed period; exp(+j omega t)); k{axis} and '
    'k{axis} + 2 pi / L are the same state; 0 = plain periodic. The phase sits on the wrap entries of the operator (no '
    'extra cost). Point sources repeat in every period image with that phase (a phased array); a line source along the '
    'periodic axis carries exp(-j k{axis} s) (s = cell centres from the grid lo corner), so with k_z it launches the '
    'conical plane wave of wave vector (k{axis}, k_normal, k_z). Results are those of the Bloch wave at +k.')
_BLOCH_TIP = ('Phase step of the wave from one period to the next along {axis}, in radians per metre, for periodic '
              'edges.\n\n0 means no phase step. For light arriving at an angle use the in-plane share of its wave '
              'vector, e.g. 2 pi sin(angle) / wavelength in vacuum.')
_SETTINGS = boundary_settings_params(_BC_HELP) + (
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help='Vacuum wavelength lambda0 of the solve in a stationary study (omega = 2 pi C0 / lambda0); an '
               'eigenfrequency study searches the resonances nearest it at the k_z setting (the target); a '
               'frequency_domain study solves its own wavelengths and ignores it (validation info when it is not the '
               'default). The mesh needs >= 10 (better 20) cells per lambda0 / |n| in every material and >= 2 per skin '
               'depth in lossy ones (validation checks both). Typical 0.4e-6 to 2e-6 m.',
          tip='Vacuum wavelength at which the field is solved, in metres: 1.55e-6 for 1550 nm.\n\n'
              "An 'Eigenfrequency' study looks for the resonances closest to it; a 'Frequency domain' study solves "
              'its own list of wavelengths.'),
    Param('k_z', 'float', 0.0, label='Out-of-plane wave number', unit='rad/m', help=_KZ_HELP, tip=_KZ_TIP),
    Param('discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATION_ORDERS),
          choice_labels=(('quadratic', 'Quadratic'), ('linear', 'Linear')),
          help='Element order on a triangles mesh (MeshSpec kind triangles; tensor meshes ignore it and use the Yee '
               'cell). Eigenfrequency study only: the em_modes mixed elements (Nedelec edge elements for E_x, E_y, '
               'Lagrange elements for E_z of the same degree; discrete gradients are exact static solutions, so no '
               'spurious modes), the k_z terms in the curl, elements with a curved edge mapped isoparametrically '
               '(quadratic), so round guides follow the drawn outline instead of a staircase. quadratic (default) or '
               'linear (Whitney / P1: cheaper, error ~h^2 instead of ~h^4 in k0^2). About 7 (quadratic) or 2 (linear) '
               'unknowns per element, one assembly per solve, squared-Cayley shift-invert by the eigen solver: host '
               'SuperLU, or with its factorization cudss a cuDSS LDL^T on the GPU (complex symmetric pencil), '
               'cudss_mixed its complex64 form with float64 Rayleigh quotients. Materials are one value per element; a '
               'cpml face gets a ring of mesh pml_layers element layers with the stretch of the pml_* settings and a '
               'pec outer side; pec_domain removes its elements and holds tangential E = 0 on their outline; pec / pmc '
               '/ mirror faces as em_modes. Not available on triangles (validation errors): stationary and '
               'frequency_domain studies (driven solves) with every source / probe / flux feature, periodic and '
               'Bloch faces (no periodic mesh); resonance_table and pec_domain work.',
          tip='Order of the finite elements when the mesh is made of triangles: quadratic is far more accurate per '
              'element, linear is cheaper.\n\nUnused on a rectangular-cell mesh.'),
) + bloch_params(_BLOCH_HELP, _BLOCH_TIP)

# ---- material vocabulary ('modes' table: eps_r -> all three arrays; eps_zz writes eps_r) ----
_COMPLEX_EPS = ('The solve uses the complex permittivity at the solve wavelength: this value - j sigma_e / (omega '
                'eps0) plus the susceptibility chi(omega) of every Drude / Lorentz pole (exp(+j omega t): loss has '
                'Im < 0).')
_FACE_VALUE = ('the width-weighted mean of the two cells sharing that face (solver_defaults.subpixel_smoothing '
               'averages interface faces anisotropically)')
_MATERIAL_PROPERTIES = tuple(replace(param, label='Relative permittivity, z (out of plane)')
                             if param.name == 'eps_zz' else param for param in em_material_params('modes', {
    'eps_r': ('Isotropic relative permittivity (>= 1): sets eps_xx, eps_yy and eps_zz unless the material sets those '
              'components (a component wins over eps_r). n^2 of a lossless dielectric: silica ~2.085, silicon ~12.1 at '
              '1.55 um; may depend on x, y and physics fields. E_z sees the cell value, E_x on the +x and E_y on the +y '
              f'cell faces {_FACE_VALUE}. {_COMPLEX_EPS}'),
    'eps_zz': ('z component (out of plane, along the invariant direction) of the relative permittivity (>= 1), the one '
               'E_z sees at the cell centres; overrides eps_r for E_z (it writes the eps_r array).'),
    'eps_xx': (f'x component of the relative permittivity (>= 1), the one E_x sees on the +x cell faces ({_FACE_VALUE}); '
               'overrides eps_r for E_x.'),
    'eps_yy': (f'y component of the relative permittivity (>= 1), the one E_y sees on the +y cell faces ({_FACE_VALUE}); '
               'overrides eps_r for E_y.'),
    'sigma_e': ('Ohmic loss in S/m on all three E components, default 0: the permittivity at the solve wavelength gains '
                '-j sigma_e / (omega eps0); a weak absorber of index n and extinction kappa has sigma_e = 2 n kappa '
                'omega eps0. Model metals with a Drude pole, not their DC conductivity.'),
    'mu_r': ('Real relative permeability (>= 1) of H_x, H_y and H_z, each taking the value of the cell it belongs to (H_x '
             'on the +y face, H_y on the +x face, H_z at the +x+y corner of that cell); 1 for optical materials. The '
             'solve uses mu_r - j sigma_m / (omega mu0).'),
    'sigma_m': 'Magnetic loss on H_x, H_y, H_z in Ohm/m, default 0: mu_r gains -j sigma_m / (omega mu0).',
}, tips={'eps_zz': ('Permittivity for the electric field perpendicular to the drawing.\n\n'
                    "When set it replaces 'Relative permittivity' for that direction.")}))

# ---- features ----
_PHASE_PARAM = Param(
    'phase', 'float', 0.0, label='Phase', unit='deg',
    help='Phase of the source phasor in degrees (0 = cos carrier, as the em_fdfd_* sources); with a single source it '
         'only rotates every phasor.',
    tip='Phase of the source in degrees; it matters only with several sources.')
_COMPONENT_PARAM = Param(
    'component', 'choice', 'z', label='Direction', choices=COMPONENTS,
    choice_labels=(('x', 'Along x'), ('y', 'Along y'), ('z', 'Perpendicular to the drawing (z)')),
    help="Current direction: 'z' sits on the E_z point (cell centre); 'x' / 'y' are split half on each of the two "
         'faces of the cell along that axis (the E_x / E_y points, by their dual areas; the lo face of the first cell '
         'of a periodic axis is its wrapped last face with the Bloch phase). Every direction is modulated as exp(-j k_z '
         'z) along the invariant direction.',
    tip='Direction of the current: along x, along y, or perpendicular to the drawing.')
_POINT_SOURCE_PARAMS = (
    _COMPONENT_PARAM,
    Param('current', 'float', 1e-3, label='Current', unit='A',
          help='Current I [A] = the integral of the current density J over the cell cross-section: J = I / (dx dy) '
               "for 'z' (a filament I exp(-j k_z z), at k_z = 0 the em_fdfd_ez point source), a line of x / y current "
               'elements of moment I per unit z length otherwise. Fields scale linearly with it. '
               f'{POINT_SOURCE_SIGN}',
          tip='Strength of the source, in amperes. Fields scale linearly with it.'),
    _PHASE_PARAM,
)
_LINE_SOURCE_PARAMS = (
    _COMPONENT_PARAM,
    Param('amplitude', 'float', 1.0, label='Surface current', unit='A/m',
          help='Uniform surface current density K [A/m] of the sheet: each line cell carries a current element of K '
               'times its length along the line (J = K / width across the line), along a periodic Bloch axis times '
               'exp(-j k s). A sheet across a whole periodic period (or ending on cpml edges, where it continues '
               'through the PML) launches plane waves of wave vector (k along the line, +-k_normal, k_z) on both '
               'sides, k_normal^2 = k0^2 n^2 - k_along^2 - k_z^2 (evanescent when negative). Fields scale linearly '
               f'with it. {POINT_SOURCE_SIGN}',
          tip='Strength of the sheet of current along the line, in amperes per metre. Fields scale linearly with '
              'it.\n\nAcross a whole periodic domain it launches plane waves on both sides.'),
    _PHASE_PARAM,
)
_PROBE_PARAMS = (
    Param('component', 'choice', 'ez', label='Field component', choices=FIELD_COMPONENTS,
          choice_labels=(('ex', 'E_x'), ('ey', 'E_y'), ('ez', 'E_z'), ('hx', 'H_x'), ('hy', 'H_y'), ('hz', 'H_z')),
          help='Component recorded: its complex phasor at the centre of the cell containing the point (face / corner '
               'points averaged there, Bloch phases included), per driven solve (stationary, frequency_domain), at z '
               '= 0 of the exp(-j k_z z) dependence.',
          tip='Field recorded at the point; electric fields in V/m, magnetic fields in A/m.'),
)
_FLUX_DESCRIPTION = (
    'Time-averaged power per unit length along z [W/m] through an axis-aligned line at every driven solve, 1/2 Re (E x '
    'H*) . n: a horizontal line gives the flux along +y, sum [dx E_z H_x* - dx_dual E_x H_z*] / 2, a vertical line the '
    'flux along +x, sum [dy_dual E_y H_z* - dy E_z H_y*] / 2, with E of the line cells and H on their hi face (the '
    'boundary term of the discrete Poynting theorem: through lossless cells a line across a whole period, or wall to '
    'wall, carries the same power on every row, so the difference of two such lines is exactly the source power minus '
    'the absorption between them); the face pairs along the line take the two end faces half. A line reaching a mirror '
    "face counts its image (x mirror_factor, 2 per face). The z flux of the field is the output field sz. Result "
    "'flux_spectrum' with unit 'W/m', one entry per solved wavelength sorted by wavelength. The line covers "
    f'{SPAN_CELLS}; its hi face must be inside the grid (or wrap on a periodic axis).')
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency study solve at the k_z setting, nearest the target (the wavelength setting) "
    "first. Result 'table', one row per resonance: resonance_number, wavelength = 2 pi C0 / Re(omega) in m, frequency / "
    'frequency_imag = Re / Im(omega) / (2 pi) in Hz (> 0 decays), q_factor = Re(omega) / (2 Im(omega)) (inf when '
    'lossless and closed), pml_fraction (share of |E|^2 dA in PML cells; above 0.5 dropped); extras target_wavelength '
    '(m) and selected_resonance. At k_z = beta the resonances are the guided modes of propagation constant beta: a '
    'parameter sweep of k_z with a derived value of the frequency column gives the dispersion. Accurate only near the '
    "target (see the eigenfrequency study). For Q versus a parameter add a derived value with source '<physics>.<this "
    "feature>', expression q_factor, operation at and at = 0.")


def _apply_line_source(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    return solver.add_line_source(axis, position, span, component=values['component'],
                                  amplitude=values['amplitude'], phase=values['phase'])


def _validate_line_source(vctx, physics_spec, feature_spec, values):
    """Axis-aligned line inside the domain; the eigenfrequency warning."""
    return (validate_line_feature(vctx, physics_spec, feature_spec, values)
            + eigen_only_feature_issues(vctx, physics_spec, feature_spec))


def _validate_flux_monitor(vctx, physics_spec, feature_spec, values):
    """Axis-aligned line inside the domain whose hi face is inside the grid (or wraps); the eigenfrequency warning."""
    path = feature_path(physics_spec, feature_spec)
    issues = validate_line_feature(vctx, physics_spec, feature_spec, values)
    if vctx.mesh is not None and vctx.domain_size is not None:
        axes = dict(zip('xy', _window_axes(vctx.mesh)))
        settings = raw_settings(physics_spec)
        for shape in _selected_shapes(vctx, feature_spec, 'line'):
            cells = _line_cells(vctx.mesh, shape.values, vctx.domain_size)
            if cells is None:
                continue   # not axis-aligned: reported by validate_line_feature
            normal_axis, line_cell, _ = cells
            periodic = settings.get(f'bc_{normal_axis}_lo') == 'periodic'
            if line_cell >= axes[normal_axis][0].size - 1 and not periodic:
                issues.append(Issue('error', path, f"line '{shape.name}' lies in the last grid cell along "
                                                   f'{normal_axis}, whose hi face is the outer grid edge: move it '
                                                   'inside'))
    return issues + eigen_only_feature_issues(vctx, physics_spec, feature_spec)


def _validate_resonance_table(vctx, physics_spec, feature_spec, values):
    """Warning outside an eigenfrequency study: the table has no solve to show."""
    return resonance_table_study_issues(vctx, physics_spec, feature_spec, optomechanics=False)


_FEATURES = (
    FeatureType('point_source', 'Point source', 'source', 'point', params=_POINT_SOURCE_PARAMS,
                object_kinds=('point',), max_entities=1,
                description='Current element of any direction in the cell containing the point, modulated as exp(-j '
                            f'k_z z) along the invariant direction; see component and current. {POINT_SOURCE_SIGN}',
                summary='A small current at one point, along x, y or perpendicular to the drawing, varying along the '
                        'invariant direction with the out-of-plane wave number.',
                apply=apply_component_point_source, validate=validate_driven_point),
    FeatureType('line_source', 'Line source', 'source', 'line', params=_LINE_SOURCE_PARAMS, object_kinds=('line',),
                max_entities=1,
                description='Uniform current sheet of any direction on the cells of an axis-aligned line (see '
                            'amplitude): across a periodic period with bloch_k along it and k_z it launches the '
                            f'conical plane waves of that wave vector on both sides. {POINT_SOURCE_SIGN}',
                summary='A sheet of current along a line: across a periodic domain it sends plane waves to both sides, '
                        'tilted by the Bloch and out-of-plane wave numbers.',
                apply=_apply_line_source, validate=_validate_line_source),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='xy',
                description="Phasor of one field component in the cell containing the point at every driven solve. "
                            "Result 'xy' versus wavelength: series abs (unit of the component) and phase (rad); "
                            'extras frequency, wavelength, component, unit.',
                summary='Records the field at one point for every solved wavelength.',
                apply=apply_field_probe, result=object_result(conical_probe_result),
                validate=validate_driven_point),
    FeatureType('flux_monitor', 'Flux monitor', 'monitor', 'line', object_kinds=('line',), max_entities=1,
                result_kind='flux_spectrum', description=_FLUX_DESCRIPTION,
                summary='Measures the power per metre of depth crossing a horizontal (upward positive) or vertical '
                        '(rightward positive) line at each solved wavelength.',
                apply=apply_flux_monitor, result=object_result(conical_flux_result),
                validate=_validate_flux_monitor),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor: in every cell whose centre lies in the selected domains E_z = '
                            '0 at the centre and E_x / E_y = 0 on the faces between two such cells, staircased to the '
                            'mesh (as em_modes; no conformal option).',
                summary='Turns the selected domains into a perfect metal, following the mesh cells.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0), grids=BOTH_GRIDS),
    FeatureType('resonance_table', 'Resonance table', 'monitor', None, result_kind='table',
                description=_RESONANCE_TABLE_DESCRIPTION,
                summary="Lists the resonances found by an 'Eigenfrequency' study at the out-of-plane wave number: "
                        'wavelength, frequency, quality factor and how much of each lies in the absorbing layers.',
                apply=apply_resonance_table, result=object_result(resonance_table_result, with_solver=True),
                validate=_validate_resonance_table, grids=BOTH_GRIDS),
)

# ---- type ----
_DESCRIPTION = (
    'Conical (out-of-plane wave number) finite-difference frequency-domain solver of a z-invariant structure: the '
    'drawing is its xy cross-section, the fields vary as exp(j (omega t - k_z z)) with the k_z setting (2D '
    'frequency domain with an out-of-plane wave number), so E_x, E_y, E_z and H_x, H_y, H_z all couple. Projected 3D '
    'Yee cell (the em_modes layout and wall rules): E_z at the cell centres, E_x and H_y on the +x faces, E_y and H_x '
    'on the +y faces, H_z at the +x+y corner; unknowns E (3 per cell, PML included), curl-curl operator (k0^2 eps - '
    'curl mu^-1 curl) E = -j omega mu0 J with d/dz = -j k_z, solved on the host (solver role linear = sparse_direct: '
    "host SuperLU, a 'cusolver_qr' node is a validation error; one factor per wavelength, k_z and material state, "
    'reused as GMRES preconditioner after a material change at the same wavelength and k_z; back_end cudss / '
    'cudss_mixed: a cuDSS LDL^T of the dual-area-symmetrized operator on the GPU (an LU with a Bloch phase), no host '
    "size limit, later factorisations of the same sparsity reusing the first one's cuDSS plan). Needs "
    'scipy; validation warns near and refuses beyond the factor size limit. Studies: stationary (one driven solve at '
    'the wavelength setting, iterated with heat through heat_em_thermo_optic / em_heat_joule), frequency_domain (one '
    'driven solve per wavelength at the same k_z) and eigenfrequency (role eigen = arpack_shift_invert: squared-Cayley '
    'shift-invert resonances at k_z nearest the wavelength setting, complex omega and Q; static curl-free solutions '
    'dropped; at k_z = beta they are the guided modes, so a k_z sweep gives the dispersion; PML and dispersion frozen '
    'at the target; factorization auto (the default: cudss on the gpu backend with nvmath-python, falling back to '
    'superlu if cuDSS refuses the operator; else superlu), superlu (host), cudss (GPU, no host size limit: an LDL^T of '
    'the operator symmetrized '
    'by the dual areas, an LU with a Bloch phase) or cudss_mixed (complex64 factor, loose solves, float64 Rayleigh '
    'quotients; with a Bloch phase a refined LU)). '
    'Walls / mirrors / Bloch phases / CPML as em_modes plus bloch_kx / bloch_ky; at k_z = 0 the problem splits into '
    'the em_fdfd_ez and em_fdfd_hz ones (walls half a cell apart from theirs). Outputs at the cell centres: ex, ey, ez, '
    'hx, hy, hz (Re of the phasors at z = 0, V/m and A/m), their moduli <component>_abs, e_abs, e_intensity (|E|^2, the '
    'primary field), h_abs, sz = 1/2 Re(E_x H_y* - E_y H_x*) (power flow along z, W/m^2), absorbed_power (state q_abs '
    '= 1/2 omega eps0 sum_c (-Im eps_c) |E_c|^2 in W/m^3, cell means; em_heat_joule hands it to heat) and the '
    'permittivity arrays eps_r (eps_zz), eps_xx, eps_yy, sigma_e; scalar output source_power. A resonance is scaled to '
    '1 real at its largest E sample (V/m). Features: point_source and line_source (currents along x, y or z), '
    'field_probe, flux_monitor (W/m through a line), pec_domain, resonance_table. Material vocabulary of em_modes '
    '(eps_xx, eps_yy, eps_zz diagonal). heat_em_thermo_optic writes all three permittivity arrays (slots eps_r_update, '
    "eps_xx_update, eps_yy_update). Not yet: a conical TFSF plane wave, far field, mode ports, conformal PEC. On a "
    'triangles mesh (MeshSpec kind triangles) the eigenfrequency study runs on mixed finite elements instead (see '
    'discretization: curved outlines followed, materials and state fields one value per element, the same '
    "resonance_table); driven studies, the source / probe / flux features and periodic faces stay on tensor meshes. The "
    "instance is usually named 'em'.")
_SCALAR_OUTPUTS_HELP = (
    'source_power (W/m): the time-averaged power per unit z length the enabled point / line sources deliver in the '
    'last driven solve, 1/2 Re sum E . J* dA over their current points (the drawn part: a mirror image is not '
    'counted). Equal to the net flux out of a closed set of lossless flux lines around the sources without a mirror '
    'face; with one the flux results are unfolded (x mirror_factor) while this stays the drawn part. Missing (NaN plus '
    'a run-log warning) before a driven solve, e.g. in an eigenfrequency study. Derived value expression '
    '<physics>_source_power.')


def _scalar_outputs(settings_values, physics_spec):
    """source_power (ConicalPhysics.scalars)."""
    return ('source_power',)


def _create(ctx, physics_spec, settings_values):
    """ConicalPhysics at the first solved wavelength and the k_z setting with its dispersive poles and their background
    permittivity written into the three permittivity arrays; never finalizes. A triangles mesh: FEMConicalPhysics
    (eigenfrequency only; walls from the settings)."""
    solvers = ctx.solvers(physics_spec.name)
    if getattr(ctx.domain, 'grid_kind', 'tensor') == 'triangles':
        solver = FEMConicalPhysics(ctx.domain, fdfd_domain_requirements(settings_values).boundary_config,
                                   wavelength=first_solved_wavelength(ctx, settings_values), k_z=settings_values['k_z'],
                                   eigen_solver=solvers.get('eigen'),
                                   order=DISCRETIZATION_ORDERS[settings_values['discretization']])
    else:
        solver = ConicalPhysics(ctx.domain, boundary_config=ctx.domain.boundary_config,
                                wavelength=first_solved_wavelength(ctx, settings_values), k_z=settings_values['k_z'],
                                linear_solver=solvers['linear'], eigen_solver=solvers.get('eigen'))
    write_dispersive_materials(ctx, physics_spec, solver, CONICAL_PERMITTIVITY_ARRAYS)
    return solver


def _solver_issues(vctx, physics_spec, settings_values, path):
    """scipy missing: error; a linear solver node with back_end cusolver_qr: error (host matrix only); the host factor
    size limit (conical.size_issue_for; none for a cudss eigen-solve or, driven, a cuDSS linear back end) on tensor
    meshes (MUMPS / PARDISO keep it); a mixed factorization or back end (MIXED_FACTORIZATIONS / MIXED_BACK_ENDS) with
    a Bloch phase: info (no symmetrizer: a refined LU); driven study without a source: warning."""
    solvers = vctx.solvers(physics_spec) or {}
    issues = (scipy_issues('em_fdfd_conical', 'host sparse direct and eigen solvers', path)
              + host_matrix_back_end_issues(solvers, 'em_fdfd_conical', path))
    eigen = vctx.study_type == EIGENFREQUENCY_STUDY
    factorization = host_factorization(solvers, eigen)
    # MUMPS / PARDISO keep the SuperLU size rule; triangles: _triangle_issues
    if mesh_kind(vctx) == 'tensor' and factorization not in CUDSS_FACTORIZATIONS:
        issue = size_issue_for(int(vctx.mesh.nx), int(vctx.mesh.ny))
        if issue is not None:
            issues.append(Issue(issue[0], path, issue[1]))
    mixed = factorization in (MIXED_FACTORIZATIONS if eigen else MIXED_BACK_ENDS)
    if mixed and bloch_active_axes(settings_values) and mesh_kind(vctx) != 'triangles':
        role = 'eigen factorization' if eigen else 'linear back_end'
        issues.append(Issue('info', path, f'{role} {factorization} with a Bloch phase: the phased operator has no '
                                          'symmetrizer, so the complex64 factor is an LU refined to full accuracy '
                                          'at every solve' + (' (no loose solves, no Rayleigh polish)' if eigen else '')
                                          + f"; {factorization.removesuffix('_mixed')} (double) is usually as fast"))
    return issues + missing_source_issues(vctx, physics_spec, path, ('point_source', 'line_source'), _DRIVEN_STUDIES)


_SHARED_VALIDATE = make_validate_physics(CONICAL_PERMITTIVITY_SLOTS, time_domain=False,
                                         solved_wavelengths=all_solved_wavelengths)


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared EM checks (pairing, materials, slot writers, resolution and skin depth at every solved wavelength),
    the unused wavelength setting, the solver, the Bloch settings and the triangles-mesh rules."""
    path = ('physics', physics_spec.name)
    issues = _SHARED_VALIDATE(vctx, physics_spec, settings_values)
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(_solver_issues(vctx, physics_spec, settings_values, path))
    issues.extend(fdfd_bloch_issues(vctx, physics_spec, settings_values))
    issues.extend(_triangle_issues(vctx, physics_spec, settings_values, path))
    return issues


def _triangle_issues(vctx, physics_spec, settings_values, path):
    """On a triangles mesh: errors outside the eigenfrequency study (no driven FEM solve yet) and for periodic / Bloch
    faces (no periodic mesh); the FEM size rule of host SuperLU on the estimated unknowns (none for a cudss eigen
    factorization). The source / probe / flux features are refused by their grids."""
    if mesh_kind(vctx) != 'triangles':
        return []
    issues = []
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        issues.append(Issue('error', path, f'em_fdfd_conical on a triangles mesh solves resonances only (an '
                                           f'{EIGENFREQUENCY_STUDY} study); a {vctx.study_type} study (driven solves, '
                                           'sources, probes and flux) needs a tensor mesh'))
    if 'periodic' in (settings_values['bc_x_lo'], settings_values['bc_y_lo']):
        issues.append(Issue('error', path, f'em_fdfd_conical on a triangles mesh: {PERIODIC_TEXT}'))
    return issues + fem_size_issues(vctx, physics_spec, settings_values, path, FEM_CONICAL_SIZE_REMEDY)


# run-time estimate: CSR entries per row of the curl-curl system (rough: E_z a 5-point Laplacian plus its E_x / E_y
# couplings) and host bytes moved per stored entry by one assemble_conical (curls, PML stretch, curl-curl product,
# active-set slicing; calibrate against the assembly share of a frequency_domain run)
_CONICAL_ENTRIES_PER_ROW = 13
CONICAL_ASSEMBLY_BYTES_PER_ENTRY = 800


def _time_estimate(settings_values, mesh, context):
    """Host solves of 3 unknowns per cell: driven (stationary / frequency_domain) per solve one assembly and one solve
    of the 'linear' role, a factorisation per wavelength (coupled: GMRES on the kept factor after a material change);
    eigenfrequency: one assembly, one factor and one squared-Cayley eigen-solve of 'eigen' (2 n_resonances + 10, two
    solves per application; a cudss factorization: the recipe's 2D 'not estimated'); triangle meshes:
    _triangle_time_estimate."""
    if mesh.kind == 'triangles':
        return _triangle_time_estimate(settings_values, mesh, context)
    unknowns = 3.0 * int(mesh.n_cells)
    nnz = _CONICAL_ENTRIES_PER_ROW * unknowns
    assembly_ops = (te.HostStream(CONICAL_ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200))
    if context.study.type == EIGENFREQUENCY_STUDY:
        host = eigen_backend_kwargs(context)
        eigenpairs = resonance_eigenpairs(context, optical=False)
        return te.combine((te.TimeEstimate((te.TimeItem('conical assembly', 'assembly', 1.0, 'exact', assembly_ops),
                                            resonance_filter_time_item(context.backend, unknowns, nnz, eigenpairs))),
                           context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=16, solves=1.0,
                                               factorizations=1.0, rhs=2, eigenpairs=eigenpairs, **host)))
    points = float(context.count('harmonic_points', 1)) if context.study.type == FREQUENCY_DOMAIN_STUDY else 1.0
    solves, kind, limit = context.static_solves()
    assembly = te.TimeItem('conical assembly', 'solve', points * solves, kind, assembly_ops,
                           limit=None if limit is None else points * limit, counter=physics_counter(context, 'solves'))
    # SuperLU: one factor per wavelength (GMRES on it after a material change); cuDSS refactorises per system change
    factorizations = points * solves if linear_solver_cudss(context.solvers) else points
    return te.combine((te.TimeEstimate((assembly,)),
                       context.solver_work('linear', unknowns=unknowns, nnz=nnz, value_bytes=16,
                                           solves=points * solves, solves_kind=kind, factorizations=factorizations,
                                           factorizations_kind=kind, backend='cpu')))   # host matrix on either backend


# ... of a triangles mesh: host bytes per stored entry of one fem_conical assembly (quadrature-point coefficients, six
# blocks, the k_z sum and restriction) and Python calls; uncalibrated (compare the eigen-solve info's assembly_seconds)
FEM_CONICAL_ASSEMBLY_BYTES_PER_ENTRY = 2000
FEM_CONICAL_ASSEMBLY_CALLS = 400


def _triangle_time_estimate(settings_values, mesh, context):
    """em_fdfd_conical on a triangles mesh (eigenfrequency only): fem_resonance_time_estimate of one solve."""
    return fem_resonance_time_estimate(settings_values, mesh, context, 'FEM conical assembly',
                                       FEM_CONICAL_ASSEMBLY_BYTES_PER_ENTRY, FEM_CONICAL_ASSEMBLY_CALLS,
                                       resonance_eigenpairs(context, optical=False))


EM_FDFD_CONICAL = PhysicsType(
    name=_TYPE_NAME,
    label='Electromagnetic Waves, frequency domain (out-of-plane wave number k_z)',
    family=EM_FAMILY,
    group='em',
    spaces_2d=('planar', 'cross_section'),   # oblique / conical incidence, and a cross-section driven at beta = k_z
    default_instance_name='em',
    description=_DESCRIPTION,
    summary='Light in a structure that does not change along the direction perpendicular to the drawing, with a set '
            'wave number along that direction: oblique and conical incidence, waveguides driven at a propagation '
            'constant, and frequency versus wave number from resonances. All six field components.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS[CONICAL_MATERIAL_SCHEMA_KEY],
    state_fields=tuple(ConicalRenderer.field_infos[name] for name in ('e_intensity', 'eps_r', 'eps_xx', 'eps_yy',
                                                                       'sigma_e'))
    + (replace(ConicalRenderer.field_infos['absorbed_power'], name='q_abs'),),
    output_fields=tuple(ConicalRenderer.field_infos.values()),
    source_slots=source_slot_docs(CONICAL_PERMITTIVITY_SLOTS),
    supported_studies=('stationary', FREQUENCY_DOMAIN_STUDY, EIGENFREQUENCY_STUDY),
    default_study={'type': 'stationary'},
    uses_dispersion=True,
    create=_create,
    domain_requirements=fdfd_domain_requirements,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate_physics,
    memory_estimate=conical_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALAR_OUTPUTS_HELP,
    # eigen cudss_mixed: the Rayleigh-polished complex64 LDL^T (Yee symmetrized by D, triangle pencil complex symmetric);
    # 'auto' = cudss on the gpu (own factor, no inertia count; host SuperLU took 211 s for 14.4k Bloch Yee unknowns)
    solver_roles={'linear': CUDSS_LINEAR_ROLE,
                  'eigen': eigen_role((EIGENFREQUENCY_STUDY,), mixed_factorization=True,
                                      auto_factorizations={'tensor': 'cudss', 'triangles': 'cudss'})},
    subpixel_smoothing=True,
    backends=('gpu', 'cpu'),   # host solves either way; the shown fields live on the run backend
    grids=BOTH_GRIDS,         # triangles: solvers/fdfd_conical_fem (eigenfrequency only)
)


def _selftest_cases():
    """The conical CPU numerics, model-run and triangle SelfTestCases (imported on the first registry.selftests())."""
    from .selftests_conical import CONICAL_CPU_SELFTESTS, CONICAL_MODEL_SELFTESTS
    from .selftests_conical_fem import CONICAL_FEM_SELFTESTS
    return CONICAL_CPU_SELFTESTS + CONICAL_MODEL_SELFTESTS + CONICAL_FEM_SELFTESTS


def register(registry):
    """Register em_fdfd_conical (after em_modes and the BOR types: the FDTD material Params stay first per name) and
    its selftest loader."""
    registry.register_physics(EM_FDFD_CONICAL)
    registry.register_selftests(_selftest_cases)

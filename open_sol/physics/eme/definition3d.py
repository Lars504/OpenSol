"""Model definition of 'em_eme_3d' (3D eigenmode expansion along z: settings, 'modes' material vocabulary, the port /
section-table / length-sweep / periodic / bend / PEC features, validation, memory estimate); registered by
definition.register."""
from dataclasses import replace

from ...core import time_estimate as te
from ...core.memory_estimate import (BOOL_BYTES, COMPLEX64_BYTES, COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES,
                                     FLOAT32_BYTES, FLOAT64_BYTES, INT32_BYTES, MemoryEstimate, host_lu_bytes)
from ...model.definition_helpers import eigen_factorization, eigen_role, eigen_subspace_size
from ...model.expressions import referenced_names
from ...model.geometry import expand_entities
from ...model.schema import (DomainRequirements, FeatureType, Issue, Param, PhysicsType, array_owner, feature_path,
                             solver_backend)
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, arnoldi_basis_bytes, arnoldi_basis_size
from ...solvers.cudss_factor import factor_device_bytes, operator_copy_bytes, solve_buffer_bytes, solve_peak_bytes
from ...solvers.host_direct import HOST_DIRECT_DOUBLE_BACK_ENDS
from ...solvers.sparse_direct import scipy_available
from ..em.cross_section_modes import size_issue_for
from ..em.definition_common import (CPML_BC_HELP, DEFAULT_WAVELENGTH, EM_FAMILY, FREQUENCY_DOMAIN_STUDY,
                                    PERIODIC_BC_HELP, all_solved_wavelengths, background_permittivity,
                                    boundary_settings_params, em_material_params, first_solved_wavelength, make_pole,
                                    make_validate_physics, object_result, shortest_solved_wavelength,
                                    source_slot_docs, unused_wavelength_issues)
from ..em.materials import EM_MATERIAL_SCHEMAS
from ..em.memory_estimates import host_factor_items
from ..em3d.boundaries3d import BoundaryConfig3D
from .definition import (OBJECTIVE_FEATURE, PORTS_FEATURE, SCALAR_OUTPUTS_HELP, SECTION_TABLE_DESCRIPTION,
                         adjoint_time_estimate, apply_ports, apply_section_table, bend_feature, length_sweep_feature,
                         lossy_materials, other_em_issues, periodic_feature, physics_counter, port_params,
                         scalar_outputs, shared_time_estimate, validate_ports)
from .renderer3d import EME_3D_FIELD_INFOS
from .solver import MODE_CACHE_BYTES, EME_PERMITTIVITY_ARRAYS, EME_PERMITTIVITY_SLOTS, ports_result, section_table_result
from .solver3d import FIELD_BLOCK_SLICES, EMEPhysics3D

TYPE_NAME_3D = 'em_eme_3d'
_STATIONARY_STUDY = 'stationary'
_PORT_FACE_TYPE = 'pec'          # z faces of the BoundaryConfig3D: ports, never read, no padding

# ---- settings ----
_BC_HELP = ' '.join((
    'Transverse faces of every cross-section (the xy plane of each z layer), as in em_modes. The two z faces are the '
    'ports: the first and last z cell layers of the domain continue to infinity (no z padding).',
    CPML_BC_HELP,
    'In the mode solves it is the complex coordinate stretch of the pml_* settings and the modes localised in it are '
    'kept (unlike em_modes): they stand for the radiation, so a radiating structure converges in n_modes more slowly.',
    PERIODIC_BC_HELP,
    'pec, pmc and the mirrors (symmetric / antisymmetric) make the outermost cell a ghost outside the wall plane, as '
    'em_modes does.',
))
_SETTINGS = (
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help='Vacuum wavelength lambda0 of a stationary solve (omega = 2 pi C0 / lambda0); a frequency_domain study '
               'solves its own wavelengths instead (one cold solve each). Materials enter at omega (sigma_e and '
               'Drude / Lorentz poles complex). Typical 1.31e-6 or 1.55e-6 m.',
          tip='Vacuum wavelength of the light in metres: 1.55e-6 for 1550 nm. A frequency domain study uses its own '
              'list instead.'),
    Param('n_modes', 'int', 20, label='Modes per section', min=1,
          help='Cross-section modes of each section (shift-invert near (k0 n_eff_guess)^2, ranked by descending '
               'Re(beta^2) with the PML modes kept and listed after the others). A truncated basis leaves a projection '
               "error that the section table reports as the interfaces' reciprocity defect (a run-log warning above "
               '1e-3); raise n_modes until the port results converge. Each section costs one sparse factorisation of '
               '2 nx ny unknowns plus about 2 n_modes + 10 Arnoldi vectors. 10-40 typical.',
          tip='How many modes describe the light in each slice of the device. More modes are more accurate and '
              'slower.'),
    Param('n_eff_guess', 'float', None, label='Effective index guess', unit='1', min=0.0, min_exclusive=True,
          optional=True,
          help='Effective index around which the modes of every section are sought (shift-invert target (k0 '
               'n_eff_guess)^2). Empty: the largest Re sqrt(eps mu) of each cross-section, which finds its most '
               'strongly guided modes first.',
          tip='Effective index near which the modes are searched. Empty starts at the highest refractive index of '
              'each slice.'),
    Param('section_tolerance', 'float', 0.0, label='Section tolerance', unit='1', min=0.0,
          help="z layers whose material rows (eps_xx, eps_yy, eps_r, sigma_e, mu_r, sigma_m, pole regions) differ from "
               "the first layer of the current section by at most this fraction of each array's largest |value| over "
               'the window join that section (solved once, on its first layer); 0 = identical layers only. A value > '
               '0 merges smoothly varying (e.g. heated) regions at the cost of a staircase in the material: with a '
               'heat coupling every heated layer differs, so set e.g. 1e-5.',
          tip='How different two neighbouring slices may be and still count as one section. Use a small value such '
              'as 0.00001 when the material varies smoothly, e.g. after heating.'),
) + boundary_settings_params(_BC_HELP)

# ---- material vocabulary (em_modes rows: E_x, E_y, E_z see eps_xx, eps_yy, eps_r) ----
_COMPLEX_EPS = ('The mode solves use the complex permittivity at omega: this value - j sigma_e / (omega eps0) plus '
                'every Drude / Lorentz pole chi(omega).')
_MATERIAL_PROPERTIES = em_material_params('modes', {
    'eps_r': ('Isotropic relative permittivity (>= 1): sets eps_zz (the E_z array eps_r) and eps_xx, eps_yy unless '
              f'the material sets those components; may depend on x, y, z and physics fields. {_COMPLEX_EPS} For a '
              'dispersive material it is the background eps_inf.'),
    'eps_zz': 'z component of the relative permittivity (>= 1), the one E_z (along the propagation axis) sees.',
    'sigma_e': f'Ohmic loss in S/m on every E component, default 0. {_COMPLEX_EPS}',
    'mu_r': 'Real relative permeability (>= 1); mu_r - j sigma_m / (omega mu0) in the solve.',
    'sigma_m': 'Magnetic loss in Ohm/m, default 0: mu_r gains -j sigma_m / (omega mu0) (artificial absorbers only).',
}, tips={'eps_zz': ('Permittivity for the electric field along z, the direction of propagation. When set it replaces '
                    "'Relative permittivity' for that direction.")})

# ---- features ----
_PORT_PARAMS_3D = tuple(
    replace(param, unit='W', help='Power of the incident mode [W]; scales only the reconstructed fields (the S-matrix '
                                  'and power fractions do not depend on it).',
            tip='Power of the incoming light in watts; it only scales the field plots.')
    if param.name == 'input_power'
    else replace(param, choice_labels=(('lo', 'Low end (z_lo)'), ('hi', 'High end (z_hi)')),
                 help='Port face of the incident mode: lo = z_lo (travelling toward +z), hi = z_hi.')
    if param.name == 'input_face' else param
    for param in port_params())
_PORTS_DESCRIPTION_3D = (
    "Ports of the 3D EME device (the z_lo / z_hi window planes) and its excitation. Result 'table' (one row per "
    'reported port mode of each face, the input face first = reflection, then the output face = transmission): '
    'output_face, mode_number, n_eff, n_eff_imag (of that port mode, Im < 0 lossy), s_abs2 = |S|^2 of the '
    'bilinear-normalised amplitudes, phase_deg = arg S (reference planes = the window z faces; each mode is phased '
    'with its dominant transverse E real > 0 at its peak), power_fraction = |S|^2 p_out / p_in (p = the power one '
    'unit of amplitude carries). Port mode k = the k-th mode of the first / last section with the PML modes listed '
    "last. Extras: wavelength, input_face, input_mode, input_power, transmission and reflection (summed over every "
    "mode of the face), s_matrix_real / s_matrix_imag over the reported modes (lo first) and s_matrix_modes. Without "
    'this feature the excitation is mode 1 at 1 W from z_lo. At most one per physics.')
BEND_DESCRIPTION_3D = (
    'Circular bend in the xz plane: the sections whose centre lies in [start, stop] along z are a bend of radius R '
    'about x = reference, drawn straightened (the range length = the arc length at the reference, the angle = length '
    "/ |R|). Each bent section is solved with the cross_section_modes bend map (model 'exact': the transformation-"
    'optics image of the curved guide, eps and mu scaled by the radial factor), so its n_eff = beta / k0 at the '
    'reference radius and the straight - bent interfaces are mode-matched like any other (bend radiation goes into '
    'the PML modes of cpml x faces: raise n_modes). The field plot shows the straightened bend. Ranges must not '
    'overlap.')


def _apply_pec(ctx, solver, feature_spec, values, selection):
    if selection.is_all:   # refused by _validate_pec
        raise ValueError(f"pec_domain '{feature_spec.name}' selects 'all': select the conductor objects")
    for name in selection.objects:
        window, mask = ctx.object_box_mask(name)
        solver.add_pec(window, mask.copy())
    return None


def _validate_pec(vctx, physics_spec, feature_spec, values):
    """'all' is refused: no field could exist (and it would be a full-grid mask)."""
    try:
        entities = expand_entities(vctx.spec, feature_spec.selection)
    except (ValueError, KeyError, TypeError):   # reported by the selection checks
        return []
    if 'all' not in entities:
        return []
    return [Issue('error', feature_path(physics_spec, feature_spec),
                  "a perfect electric conductor on 'all' fills every cross-section; select the conductor objects")]


def _pec_memory(values, n_selected_cells, mesh):
    """Cropped bool masks on the selection's 3D bounding box."""
    return MemoryEstimate.from_items((('PEC box masks (bool)', BOOL_BYTES * int(n_selected_cells)),))


_OBJECTIVE_DESCRIPTION_3D = (
    'Objective term of the EME cascade adjoint: weight times a port power fraction of the excitation. Every solve '
    'records the quantity; the output field eps_sensitivity is dJ/d eps per unit volume [1/m^3] of J = the sum of '
    'weight * quantity for an isotropic change (the sum of the eps_xx, eps_yy and eps_r terms; scripts read each '
    'component from EMEPhysics3D.eps_sensitivity().cell_gradients), from the forward and one adjoint cascade field '
    '(Lorentz reciprocity of the Yee cross-section: no extra mode solve; every window z layer, computed when the field '
    'is read). Exact for a complete basis of every cross-section; the em_eme_3d bases are truncated (n_modes), so it '
    'is the same formula on the truncated fields, converging with n_modes. The drawn cell of an eme_periodic block '
    'stands for all its n_periods periods (its gradient sums every period). The port modes (first and last z cell '
    'layer) are held fixed; sigma_e, poles and mu_r are not differentiated. em_eme_3d has no design_region and no '
    'optimization study yet: the field shows where a small material change raises or lowers J. Result \'xy\' versus '
    'vacuum wavelength: series value and weighted (unit 1); extras frequency, wavelength, quantity, mode_number, '
    'weight.')


_FEATURES_3D = (
    FeatureType(PORTS_FEATURE, 'Ports', 'source', None, params=_PORT_PARAMS_3D, result_kind='table',
                description=_PORTS_DESCRIPTION_3D,
                summary='The input and output of the device: which mode comes in, and how much of it is reflected '
                        'and transmitted into each mode.',
                apply=apply_ports, result=object_result(ports_result), validate=validate_ports),
    FeatureType('eme_section_table', 'Section table', 'monitor', None, result_kind='table',
                description=SECTION_TABLE_DESCRIPTION.replace('along propagation_axis', 'along z')
                            .replace('cell layers', 'z cell layers'),
                summary='Lists the slices the device was cut into, with their length, effective indices and how '
                        'accurately neighbouring slices are joined.',
                apply=apply_section_table, result=object_result(section_table_result)),
    length_sweep_feature('z', 'Transmission of the device when a part of it is made longer or shorter along z, '
                              'without solving its modes again: for example a phase shifter or taper versus length.'),
    periodic_feature('z'),
    bend_feature('z', 'the x coordinate (the bend lies in the xz plane, about an axis along y)',
                 'Position across the waveguide along x, usually its centre, at which the bend has exactly the given '
                 'radius.', BEND_DESCRIPTION_3D),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor cells of every section cross-section (the em_modes rule: E_z '
                            '= 0 at the centres of the selected cells, E_x / E_y = 0 on faces between two of them, '
                            'the normal E on a conductor surface stays free), staircased to the mesh; in the absorbed '
                            "power they count as lossless. Select objects: 'all' is refused.",
                summary='Makes the selected objects perfect metal, following the mesh cells.',
                apply=_apply_pec, validate=_validate_pec, memory_estimate=_pec_memory),
    replace(OBJECTIVE_FEATURE, description=_OBJECTIVE_DESCRIPTION_3D,
            summary='Marks a transmitted or reflected power fraction as a goal, so the solver also maps where a small '
                    'change of the material would raise or lower it.'),
)


# ---- validation ----
_SHARED_VALIDATE = make_validate_physics(EME_PERMITTIVITY_SLOTS, time_domain=False,
                                         solved_wavelengths=all_solved_wavelengths)


def _size_issues(vctx, physics_spec, settings_values, path):
    """scipy missing: error; the host SuperLU cross-section size rule (2 nx ny unknowns; none for a cudss eigen
    factorization: checked on the device at run time); info on the z slices."""
    issues = []
    if not scipy_available():   # find_spec only: nothing imported on the GUI thread
        issues.append(Issue('error', path, 'em_eme_3d needs scipy (host sparse eigensolver), which is not installed: '
                                           'pip install scipy'))
    mesh = vctx.mesh
    if mesh is None:
        return issues
    nx, ny = int(mesh.nx), int(mesh.ny)
    on_device = eigen_factorization(vctx.solvers(physics_spec)) in CUDSS_FACTORIZATIONS
    issue = None if on_device else size_issue_for(nx, ny)
    if issue is not None:
        issues.append(Issue(issue[0], path, f'cross-section: {issue[1]}'))
    sides = mesh.npml_sides
    slices = int(mesh.nz) - sides.get('z_lo', 0) - sides.get('z_hi', 0)
    issues.append(Issue('info', path, f'{slices} z slices (at most that many sections, one cross-section solve of '
                                      f'{2 * nx * ny:,} unknowns per distinct section)'))
    n_wanted = 2 * settings_values['n_modes'] + 10
    if n_wanted > 2 * nx * ny - 2:
        issues.append(Issue('info', path, f'n_modes {settings_values["n_modes"]} asks for {n_wanted} eigenvalues of a '
                                          f'{2 * nx * ny}-unknown cross-section: fewer modes come back'))
    return issues


def _validate_physics_3d(vctx, physics_spec, settings_values):
    """The shared EM checks, the unused wavelength setting, other EM types, scipy and the sizes."""
    path = ('physics', physics_spec.name)
    issues = list(_SHARED_VALIDATE(vctx, physics_spec, settings_values))
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(other_em_issues(vctx, path, TYPE_NAME_3D))
    issues.extend(_size_issues(vctx, physics_spec, settings_values, path))
    return issues


# ---- build, memory ----
def boundary_config_3d(settings_values):
    """BoundaryConfig3D of the four transverse bc_<face> settings, pec z port faces and the pml_* settings."""
    faces = {face: settings_values[f'bc_{face}'] for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    faces.update(z_lo=_PORT_FACE_TYPE, z_hi=_PORT_FACE_TYPE)
    return BoundaryConfig3D(faces, npml=int(settings_values['pml_cells']), order=int(settings_values['pml_order']),
                            sigma_factor=float(settings_values['pml_sigma_factor']),
                            kappa_max=float(settings_values['pml_kappa_max']),
                            alpha_max=float(settings_values['pml_alpha_max']))


def _domain_requirements(settings_values):
    return DomainRequirements(boundary_config=boundary_config_3d(settings_values))


def _create(ctx, physics_spec, settings_values):
    """EMEPhysics3D at the first solved wavelength with the dispersive poles of every material region (their
    background written into each permittivity array no permittivity property owns); never finalizes."""
    physics = EMEPhysics3D(ctx.domain, boundary_config=ctx.domain.boundary_config,
                           wavelength=first_solved_wavelength(ctx, settings_values),
                           n_modes=settings_values['n_modes'], n_eff_guess=settings_values['n_eff_guess'],
                           section_tolerance=settings_values['section_tolerance'],
                           eigen_solver=ctx.solvers(physics_spec.name)['eigen'])
    materials = ctx.domain.materials
    written = []
    for region in ctx.material_regions:
        if not region.dispersion:
            continue
        window, mask = ctx.region_window_mask(region)
        if not bool(mask.any()):
            continue
        for array in EME_PERMITTIVITY_ARRAYS:
            if array_owner(array, _MATERIAL_PROPERTIES, region.properties) is not None:
                continue
            background = background_permittivity(region.properties, region.dispersion, _MATERIAL_PROPERTIES, array)
            if referenced_names(background) <= set(ctx.parameters):
                materials.assign_window(array, window, mask, ctx.eval_scalar(background))
                written.append(array)
        for index, pole in enumerate(region.dispersion):
            try:
                dispersive_pole = make_pole(pole, ctx.eval_scalar)
            except ValueError as exc:
                raise ValueError(f"material '{region.name}', dispersion pole {index + 1}: {exc}") from exc
            physics.add_dispersive(dispersive_pole, window, mask)
    if written:
        physics.materials_changed(tuple(dict.fromkeys(written)))
    return physics


def _memory_estimate(settings_values, mesh, context):
    """3 complex64 E fields + float32 q_abs on the device, the adjoint cache (run item), per-mode-solve and field-block
    transients, one cuDSS factor at a time outside the pool (cudss eigen node); host items mostly (comments)."""
    nx, ny = int(mesh.nx), int(mesh.ny)
    sides = mesh.npml_sides
    slices = int(mesh.nz) - sides.get('z_lo', 0) - sides.get('z_hi', 0)
    points = nx * ny
    n_cells = int(mesh.n_cells)
    unknowns = 2 * points
    n_modes = int(settings_values['n_modes'])
    n_eigen = min(2 * n_modes + 10, max(unknowns - 2, 1))
    arnoldi_vectors = arnoldi_basis_size(n_eigen, unknowns, eigen_subspace_size(context.solvers))
    # e, h (2 P each) and the centred E (3 P) per mode, plus the H_x, H_y start shell of the next wavelength
    mode_set_bytes = COMPLEX128_BYTES * (n_modes * 7 + 2) * points
    on_device = eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS
    entries = _CROSS_SECTION_ENTRIES * unknowns
    # eigen mumps / pardiso: a host LU (not complex symmetric) at the SuperLU heuristic x superlu_bytes_scale
    factor_items = () if on_device else tuple(host_factor_items(
        eigen_factorization(context.solvers), 'cross-section SuperLU L + U factors (heuristic)',
        host_lu_bytes(unknowns), unknowns, entries, symmetric=False, eigen=True))
    # one mode solve at a time (cross_section_modes.solve_prepared_modes: the factor is local, freed on return), so its
    # device vectors are one transient phase. cuDSS general LU in double (the transverse-H operator is not complex
    # symmetric): the equilibrated complex128 CSR copy, rhs / equilibration / solution / applied vectors and the
    # uploaded diagonal B of device_factor_matvec
    mode_solve = []
    external_items, external_extra = (), 0
    if on_device:
        mode_solve = [('cuDSS operator copy (complex128 CSR)',
                       operator_copy_bytes(unknowns, entries, COMPLEX128_BYTES, symmetric=False)),
                      ('cuDSS rhs buffer, row scale and one solve\'s vectors',
                       solve_buffer_bytes(unknowns, COMPLEX128_BYTES) + solve_peak_bytes(unknowns, COMPLEX128_BYTES)),
                      ('device mass diagonal B (complex128 CSR)', COMPLEX128_CSR_ENTRY_BYTES * unknowns
                       + INT32_BYTES * (unknowns + 1))]
        permanent, external_extra = factor_device_bytes(unknowns, entries, COMPLEX128_BYTES, dimensions=2,
                                                        symmetric=False)
        external_items = (('cuDSS cross-section LU factor (one section at a time)', permanent),)
    # the device Krylov-Schur basis (gpu backend, eigs' size rule) lives for one mode solve, host ARPACK's is host
    basis_on_device, basis_bytes = arnoldi_basis_bytes(unknowns, arnoldi_vectors, solver_backend(context.spec),
                                                       device_matvec=on_device)
    if basis_on_device:
        mode_solve.append((f'device Krylov-Schur basis ({arnoldi_vectors + 1} complex128 rows + 2 work vectors)',
                           basis_bytes))
    # host: the operators, the mode-set cache with the start shells (capped), per slice 8 N^2 of partial cascades and
    # interfaces (every z slice taken as a section: an upper figure), the reconstruction / absorbed-power blocks
    host_items = factor_items + (() if basis_on_device else ((f'Arnoldi basis ({arnoldi_vectors} vectors)',
                                                             basis_bytes),)) + (
        ('cross-section operators A, B (sparse complex128)', 2 * 9 * COMPLEX128_CSR_ENTRY_BYTES * unknowns),
        ('mode-set cache (capped)', min(max(slices, 1) * mode_set_bytes, MODE_CACHE_BYTES)),
        ('partial cascades and interfaces (8 N^2 per section)', 8 * COMPLEX128_BYTES * n_modes ** 2 * slices),
        ('field reconstruction block', 2 * 3 * COMPLEX128_BYTES * FIELD_BLOCK_SLICES * points),
        ('absorbed power block (|E|^2, Gauss mean, loss)', 2 * FLOAT64_BYTES * FIELD_BLOCK_SLICES * points),
    )
    # _reconstruct uploads one complex64 component block and one float32 q_abs block at a time (freed after the copy)
    block_slices = max(min(FIELD_BLOCK_SLICES, slices), 1)
    transient_items = [('section mode solve', label, size) for label, size in mode_solve] + [
        ('field reconstruction', 'field block upload (complex64)', COMPLEX64_BYTES * block_slices * points),
        ('field reconstruction', 'absorbed power block upload (float32)',
         FLOAT32_BYTES * max(min(FIELD_BLOCK_SLICES // 3, slices), 1) * points)]
    run_items, features = [], [feature for feature in context.physics_spec.features if feature.enabled]
    if any(feature.type == 'adjoint_objective' for feature in features):   # one EpsSensitivity per physics
        run_items.append(('adjoint sensitivity cache (3 x float64 + float32)',
                          (3 * FLOAT64_BYTES + FLOAT32_BYTES) * n_cells))
        host_items += (('host adjoint gradients and density (4 x float64)', 4 * FLOAT64_BYTES * n_cells),
                       ('adjoint slice products block (complex128)',
                        3 * COMPLEX128_BYTES * FIELD_BLOCK_SLICES * points))
    return MemoryEstimate.from_items(
        (('EME E fields (3 x complex64) and q_abs (float32)', (3 * COMPLEX64_BYTES + FLOAT32_BYTES) * n_cells),),
        host_items=host_items, run_items=run_items, transient_items=transient_items, external_items=external_items,
        external_extra=external_extra)


# run-time estimate: host bytes moved per stored entry by one cross-section assembly (scipy Yee products, PML stretch,
# active-index slicing; calibrate from the mode-solve share of solve_seconds)
CROSS_SECTION_ASSEMBLY_BYTES_PER_ENTRY = 800
_CROSS_SECTION_ENTRIES = 9   # stored entries per unknown of the cross-section operator (as the memory estimate)
_LAYER_ROWS = 7              # read arrays, pole and PEC boxes per z layer


def _time_estimate(settings_values, mesh, context):
    """Shared section work (definition.shared_time_estimate) on 2 nx ny cross-section unknowns: per section the host
    assembly and mode set plus one 'eigen' shift-invert (2 n_modes + 10 eigenpairs, a factorisation each); per solve
    one row download per z slice and the field reconstruction (3 Gauss rows per layer when a material is lossy)."""
    nx, ny = int(mesh.nx), int(mesh.ny)
    sides = mesh.npml_sides
    slices = int(mesh.nz) - sides.get('z_lo', 0) - sides.get('z_hi', 0)
    points = float(nx * ny)
    unknowns = 2.0 * points
    nnz = _CROSS_SECTION_ENTRIES * unknowns
    n_modes = int(settings_values['n_modes'])
    layers = points * slices
    gauss = 3.0 if lossy_materials(context) else 1.0
    # per slice its rows (a download on the gpu) and their comparison; E_x, E_y, E_z of both directions per Gauss row,
    # |E|^2 / Gauss mean / transposes, the complex64 field and float32 q_abs uploads
    solve_ops = ((slices, te.Transfer(_LAYER_ROWS * FLOAT32_BYTES * points)),
                 te.HostStream(3.0 * _LAYER_ROWS * FLOAT64_BYTES * layers),
                 te.DenseFlops(8.0 * 6.0 * gauss * n_modes * layers, host=True),
                 te.HostStream(8.0 * COMPLEX128_BYTES * gauss * layers),
                 te.Transfer((3 * COMPLEX64_BYTES + FLOAT32_BYTES) * layers),
                 te.PythonCalls(100 + 20 * slices / FIELD_BLOCK_SLICES))
    # the layer's material arrays (a download on the gpu), the operator assembly, per mode its fields and ModeSet rows
    mode_ops = (te.Transfer(6 * FLOAT32_BYTES * points),
                te.HostStream(CROSS_SECTION_ASSEMBLY_BYTES_PER_ENTRY * nnz),
                te.HostStream(4.0 * 7 * COMPLEX128_BYTES * points * n_modes), te.PythonCalls(50 + 10 * n_modes))
    mode_set_bytes = COMPLEX128_BYTES * (n_modes * 7 + 2) * points
    shared, (mode_solves, mode_limit) = shared_time_estimate(context, slices, n_modes, unknowns, solve_ops, mode_ops,
                                                             mode_set_bytes)
    n_wanted = min(2 * n_modes + 10, max(int(unknowns) - 2, 1))
    eigen = context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=COMPLEX128_BYTES, dimensions=2,
                                solves=mode_solves, solves_kind='guess', factorizations=mode_solves,
                                factorizations_kind='guess', eigenpairs=n_wanted)
    # the role's items join the solve_seconds group; their guessed counts bounded by the slices as the sections are
    ratio = None if mode_limit is None or mode_solves <= 0 else mode_limit / mode_solves
    seconds = physics_counter(context, 'solve_seconds')
    eigen = replace(eigen, items=tuple(
        replace(item, seconds_counter=seconds,
                limit=None if ratio is None else (item.limit if item.limit is not None else item.count) * ratio)
        for item in eigen.items))
    return te.combine((shared, eigen, adjoint_time_estimate(context, n_modes, float(mesh.n_cells))))


# ---- type ----
_DESCRIPTION = (
    'Bidirectional eigenmode expansion (EME) of a 3D device along z, for waveguide devices long along z (tapers, '
    'phase shifters, couplers) where em_fdtd_3d would be large. Every z cell layer is a slice; consecutive equal '
    'layers (section_tolerance) form a section, so z variations are staircases at the z mesh (refine or grade it). '
    'Each section is solved once for n_modes full-vector cross-section modes (the em_modes discretisation of the xy '
    'plane with the transverse bc_* faces; complex PML stretch with the PML modes kept as radiation; repeated layers '
    'reuse their modes). Modes are normalised to the unconjugated product <e, h> = sum (E x H).z dA = 1 (stretched '
    'dual areas; degenerate modes recombined orthogonal); interfaces are mode-matched by these overlaps and the '
    'S-matrices cascaded by Redheffer star products (stable with evanescent modes), converging in n_modes (section '
    'table reciprocity_defect). The z faces are semi-infinite ports at the window planes (no z padding). Outputs of '
    'the excitation (eme_ports): the reconstructed E phasor at the cell centres (sum over modes of a_n e^{-j beta_n s} '
    '+ b_n e^{-j beta_n (L - s)} per section; E_z of the backward waves with the opposite sign): ex / ey / ez _re / '
    '_im, e_abs (primary), e_intensity, absorbed_power (1/2 omega eps0 (-Im eps) times the 3-point Gauss z mean of '
    '|E|^2 per cell, on lossy sections), and the permittivity arrays, shown as cut planes (an x or y cut shows the '
    'propagation). Scalar outputs transmission_<k>, transmission, reflection. pec_domain cells; eme_bend (bend map of '
    'cross_section_modes), eme_periodic (a drawn period repeated n times, its Bloch modes), adjoint_objective (port '
    'power fractions; output field eps_sensitivity = dJ/d eps from the cascade adjoint, no extra mode solve; no '
    'optimization study in 3D yet). em_heat_joule reads '
    'absorbed_power; heat_em_thermo_optic writes eps_r, eps_xx, eps_yy. Host memory dominates (one cross-section '
    'factorisation at a time, the cached mode sets, the partial cascades); GPU: 28 B per cell of fields and q_abs '
    "plus the material arrays (an eigen solver node with factorization 'cudss' factorises each cross-section on the "
    'GPU instead, a general LU: the operator is not complex symmetric; '
    f"{' or '.join(repr(name) for name in HOST_DIRECT_DOUBLE_BACK_ENDS)} a MUMPS / PARDISO LU on the host under the "
    'SuperLU size limit). Studies: stationary and frequency_domain (each wavelength starts every section\'s '
    'shift-invert from the summed physical modes of its previous solve, when cached). Cannot share the model with '
    "another EM type. The instance is usually named 'eme'.")

EM_EME_3D = PhysicsType(
    name=TYPE_NAME_3D,
    label='Electromagnetic Waves, eigenmode expansion (3D)',
    family=EM_FAMILY,
    group='em',
    default_instance_name='eme',
    description=_DESCRIPTION,
    summary='Light through a 3D waveguide device that is long along z, such as a taper or a phase shifter, computed '
            'by cutting it into slices with their own guided modes: transmission and reflection of each mode.',
    settings=_SETTINGS,
    features=_FEATURES_3D,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS['modes'],
    state_fields=tuple(EME_3D_FIELD_INFOS[name] for name in EME_PERMITTIVITY_ARRAYS),
    output_fields=tuple(EME_3D_FIELD_INFOS.values()),
    source_slots=source_slot_docs(EME_PERMITTIVITY_SLOTS),
    supported_studies=(_STATIONARY_STUDY, FREQUENCY_DOMAIN_STUDY),
    default_study={'type': _STATIONARY_STUDY},
    uses_dispersion=True,
    create=_create,
    domain_requirements=_domain_requirements,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate_physics_3d,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=scalar_outputs,
    scalar_outputs_help=SCALAR_OUTPUTS_HELP,
    dimensions=(3,),
    solver_roles={'eigen': eigen_role()},
    backends=('gpu', 'cpu'),   # host cross-section solves / cascade, fields on the Domain's xp (solver_plan S6)
)

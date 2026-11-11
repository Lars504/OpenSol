"""Model definition of the 3D FDTD physics type 'em_fdtd_3d' (B2): settings, material vocabulary, features (dipole,
probe, resonance finder, flux plane / box, DFT layer, PEC), build / result / validation callbacks and the memory
estimate (E8)."""
from dataclasses import replace

from ...core.constants import FACE_LABELS, FACES_3D
from ...core.domain import containing_cell
from ...core.memory_estimate import COMPLEX64_BYTES, COMPLEX128_BYTES, FLOAT32_BYTES, BOOL_BYTES, MemoryEstimate
from ...core.kernels import numba_available
from ...core.time_estimate import Elementwise, PythonCalls, explicit_step_estimate
from ...model.definition_helpers import (HALF_STORAGE_STEPPER_ROLE, resolved_stepper_precision, validate_points_inside,
                                         wavelength_range_issues)
from ...solvers.explicit_kernels import stepper_precision
from ...model.definition_resonance import resonance_monitor_feature
from ...model.expressions import ExpressionError, referenced_names
from ...model.geometry import expand_entities
from ...core.renderer import FieldInfo
from ...model.schema import DomainRequirements, FeatureType, Issue, Param, PhysicsType, array_owner, feature_path
from ..em.definition_common import (EM_FAMILY, FLUX_PARAMS, POINT_SOURCE_SIGN, _monitor_wavelengths,
                                    _q_accum_referenced, background_permittivity, boundary_settings_params,
                                    em_material_params, make_pole,
                                    make_source_shortest_wavelength, make_validate_physics, object_result,
                                    probe_result_callback, settings_params, single_object, source_slot_docs,
                                    spectral_params, steps_in_time, waveform_params)
from ..em.definition_harmonic_fdtd import HARMONIC_FDTD_TEXT, make_validate_harmonic_fdtd
from ..em.definition_resonance import RESONANCE_SUMMARY, em_resonance_probe
from ..em.memory_estimates import _harmonic_wavelength_count, _point_trace_items
from .boundaries3d import AXES, FACE_TYPES_3D, BoundaryConfig3D, cpml_psi_bytes
from .definition_sources3d import (SOURCE_FEATURES, box_corners, box_edge_issues, box_face_spectra_memory,
                                   domain_sizes, face_types, nearest_window_edge, plane_cells, plane_edge_message,
                                   window_axes)
from .engine3d import _INDEX_CHUNK_CELLS, COMPONENT_AXES   # the chunk the estimate books
from .monitors3d import DFT_PLANE_QUANTITY, dft_plane_result, flux_result
from .physics3d import EMPhysics3D, PERMITTIVITY_SLOTS
from .renderer3d import EMRenderer3D

TYPE_NAME = 'em_fdtd_3d'
_SLOT_ARRAYS = dict(PERMITTIVITY_SLOTS)   # eps_xx_update / eps_yy_update only when those arrays are allocated
_INDEX_UNIFORM_BYTES_PER_CELL = 12     # every array uniform: cp.unique's float32 copy, sort scratch and mask per chunk
_INDEX_VARYING_BYTES_PER_CELL = 40     # int64 stage codes, bin numbers, searchsorted and unique temporaries per chunk
_POLE_ITEM_LABEL = 'dispersive poles (weights, cb, J / P)'
# run-time recipe (calibrate from the selftest time_check of em3d_fdtd_cpu_gpu*, em3d_conformal_pec_cavity): a field
# kernel launch (~40 arguments) costs ~3 profile launches of host overhead (gpu ~20 us vs the 6.7 us fused floor)
_FIELD_LAUNCH_WEIGHT = 3
# per-step launches of a feature (flux DFT: phase + 2 pairs x (E, H) per plane, a box 6 planes; pec_domain none)
_FEATURE_LAUNCHES = {'flux_plane': 5, 'flux_box': 30, 'far_field': 30, 'mode_monitor': 5, 'mode_source': 6,
                     'plane_wave_source': 10, 'dft_plane': 4, 'pec_domain': 0}
_DEFAULT_FEATURE_LAUNCHES = 2
_POLE_LAUNCHES = 6                     # per field set with poles: 3 drive + 3 correction kernels (one pole assumed)
# numpy field twins (cpu backend without numba), counted from kernels3d with the cached coefficients: numpy passes
# per field set and step over the grid (update_h ~31, update_e ~34) and per CPML slab cell (cpml_h ~19, cpml_e ~21)
_NUMPY_FIELD_PASSES = 65
_NUMPY_CPML_PASSES = 40
_NUMPY_PASSES_PER_LAUNCH = 4           # the cpu 'fused' profile call: 4 numpy passes over 5 float32 arrays
_NUMPY_BYTES_PER_LAUNCH_CELL = 20

# ---- settings ----
_BC_HELP = ('Boundary of this face of the 3D box: cpml absorbs outgoing waves in pml_cells extra cells added outside '
            'the domain (the z faces too); pec = tangential E 0 at the centres of the outermost domain cell layer '
            '(that layer becomes perfect conductor); pmc = tangential H 0 on the outer face of the outermost cell '
            '(the grid edge: pec and pmc walls lie half a cell apart, so a cavity differs in width by one cell '
            'between them); periodic must be set on both faces of the axis (with the Bloch phase bloch_k<axis>, 0 = '
            'plain periodic). '
            'Mirror planes (any face, E parity as the 2D types): symmetric = E mirror-even (tangential H and normal E '
            '0 on the plane, a PMC mirror), antisymmetric = E mirror-odd (tangential E odd: a PEC mirror), on the '
            'domain edge itself: one ghost cell is added outside it (its material copied from the edge cell; on a '
            'graded z axis its width follows the PML grading, so a z mirror wants uniform z cells near it), so draw '
            'the structure about the edge; heat in the model has its face on the same plane. Each mirror halves '
            'memory and time. Flux planes and boxes reaching a mirror face count their image (x 2 per face, result '
            'extra mirror_factor), mode ports solve the whole mirrored mode (powers of the full structure), the far '
            'field adds the images of its box currents (the unfolded pattern; mirrors on both faces of one axis are '
            'then an error); a dipole half a cell from the plane is mirrored with it.')
_BC_TIP = ('What happens to waves at this face of the box.\n\n'
           '- Absorbing layer: waves leave without reflection, as into open space.\n'
           '- Perfect metal wall: a perfect mirror.\n'
           '- Perfect magnetic wall: a perfect mirror of the opposite kind.\n'
           '- Periodic: the domain repeats; set it on both opposite faces.\n'
           '- Mirror, symmetric: the electric field is mirrored unchanged at this face, to simulate half of a '
           'symmetric structure.\n'
           '- Mirror, antisymmetric: the electric field is mirrored with the opposite sign at this face.')
_BC_CHOICE_LABELS = (('cpml', 'Absorbing layer'), ('pec', 'Perfect metal wall'), ('pmc', 'Perfect magnetic wall'),
                     ('periodic', 'Periodic'), ('symmetric', 'Mirror, symmetric'),
                     ('antisymmetric', 'Mirror, antisymmetric'))
_PML_VISIBLE_3D = tuple((f'bc_{face}', ('cpml',)) for face in FACES_3D)


def _settings():
    """bc_<face> for the six faces, the shared pml_* settings (visible with any cpml face), courant_factor and
    track_absorption."""
    faces = tuple(Param(f'bc_{face}', 'choice', 'cpml', label=FACE_LABELS[face], choices=FACE_TYPES_3D,
                        help=_BC_HELP, tip=_BC_TIP, choice_labels=_BC_CHOICE_LABELS) for face in FACES_3D)
    pml = tuple(replace(param, visible_if=_PML_VISIBLE_3D) for param in boundary_settings_params(_BC_HELP)
                if param.name.startswith('pml_'))
    courant = Param('courant_factor', 'float', 0.99, label='Time step factor', unit='1', min=0.0, min_exclusive=True,
                    max=1.0,
                    tip='Time step as a fraction of the largest stable time step.\n\n'
                        'Keep 0.99. Lower it only if the build reports an instability.',
                    help='Time step as a fraction of the 3D CFL limit: dt = factor / (C0 sqrt(1/dx^2 + 1/dy^2 + '
                         '1/dz^2)) of the finest cells; 0.99 default. Dispersive poles need eps_r mu_r >= factor^2 + '
                         'mu_r (omega_p dt)^2 / 4 per Drude pole (about delta_eps (omega_0 dt)^2 / 4 per Lorentz '
                         'pole); the build checks every material (with the finest widths) and names the largest '
                         'stable factor.')
    return faces + pml + _BLOCH_PARAMS + (courant, _TRACK_ABSORPTION)


_BLOCH_HELP = (
    'Bloch wave vector along {axis} in rad/m (symmetry 2.8), used only when bc_{axis}_lo and bc_{axis}_hi are periodic: '
    'the fields repeat with the phase exp(-j k{axis} L) per period L (the meshed period; exp(+j omega t), so k{axis} > 0 '
    'for a wave travelling toward +{axis}); k{axis} and k{axis} + 2 pi / L are the same state; 0 = plain periodic. Any '
    'k != 0 steps a second (imaginary) field set: field and CPML memory, pole states and time per step double. Dipoles '
    'and mode sources drive the real set only (every period image repeats them with the Bloch phase: a dipole and a '
    'resonance monitor per swept k give the bands of a photonic-crystal slab or metasurface); the TFSF plane wave is '
    'refused (normal incidence only); mode ports solve their cross-section modes at the Bloch k of the port plane\'s '
    'periodic transverse axes (the Bloch mode; the mode source drives it into the real set like a dipole). DFT '
    'monitors (flux, field map, mode monitor, far field, harmonic FDTD) accumulate F_re + j F_im, the response '
    'at +k for every positive frequency; probes, field plots, expressions and couplings read the real set only. '
    'Absorbed power (track_absorption, q_accum, em_heat_joule) is refused. Typical: k{axis} = k0 sin(theta) of the '
    'incidence angle in vacuum, or pi / L at the zone edge.')
_BLOCH_TIP = ('Phase step of the wave from one period to the next along {axis}, in radians per metre, for periodic '
              'faces.\n\n'
              '0 means no phase step. Swept with a resonance finder it gives a band diagram. Any other value doubles '
              'memory and time per step, and the absorbed power result is then not available.')
_BLOCH_PARAMS = tuple(Param(f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
                            visible_if=((f'bc_{axis}_lo', ('periodic',)),),
                            help=_BLOCH_HELP.format(axis=axis), tip=_BLOCH_TIP.format(axis=axis))
                      for axis in AXES)


def _bloch_k(settings_values, axis) -> float:
    """Numeric bloch_k<axis> [rad/m] (0 when unset or not numeric)."""
    value = settings_values.get(f'bloch_k{axis}')
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _boundary_config(settings_values):
    return BoundaryConfig3D({face: settings_values[f'bc_{face}'] for face in FACES_3D},
                            npml=int(settings_values['pml_cells']), order=int(settings_values['pml_order']),
                            sigma_factor=float(settings_values['pml_sigma_factor']),
                            kappa_max=float(settings_values['pml_kappa_max']),
                            alpha_max=float(settings_values['pml_alpha_max']),
                            bloch_k=tuple(_bloch_k(settings_values, axis) for axis in AXES))


def _bloch_active(settings_values):
    """Axes with periodic faces and a Bloch k != 0."""
    return tuple(axis for axis in AXES if settings_values[f'bc_{axis}_lo'] == 'periodic'
                 and settings_values[f'bc_{axis}_hi'] == 'periodic' and _bloch_k(settings_values, axis) != 0.0)


def _ignored_bloch_issues(settings_values, path):
    """Warning per Bloch k set on a non-periodic axis (ignored there; em_fdtd_3d and em_fdfd_3d)."""
    return [Issue('warning', path, f'bloch_k{axis} = {_bloch_k(settings_values, axis):.4g} rad/m is ignored: '
                                   f'bc_{axis}_lo / bc_{axis}_hi are not periodic')
            for axis in AXES
            if _bloch_k(settings_values, axis) != 0.0 and settings_values[f'bc_{axis}_lo'] != 'periodic']


def _bloch_issues(vctx, physics_spec, settings_values, path):
    """Bloch k on a non-periodic axis (warning, ignored); with k != 0: absorbed power and TFSF plane waves (errors),
    Bloch port modes and real-set sources (info)."""
    issues = _ignored_bloch_issues(settings_values, path)
    if not _bloch_active(settings_values):
        return issues
    if settings_values['track_absorption'] or _q_accum_referenced(vctx.spec, physics_spec.name, vctx.registry):
        issues.append(Issue('error', path, 'absorbed power (track_absorption, q_accum, absorbed_power, em_heat_joule) '
                                           'is not available with a Bloch wave vector k != 0: the complex field mixes '
                                           'the +k and -k responses; set the bloch_k values to 0'))
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        feature_issue_path = feature_path(physics_spec, feature)
        if feature.type == 'plane_wave_source':
            issues.append(Issue('error', feature_issue_path, 'the TFSF plane wave needs Bloch k = 0 (its incident '
                                                             'field is a normal-incidence wave); set the bloch_k '
                                                             'values to 0 or drive the lattice with a dipole'))
        elif feature.type in ('mode_source', 'mode_monitor'):
            issues.append(Issue('info', feature_issue_path, f'the {feature.type} solves its port mode at the Bloch k '
                                                            "of its plane's transverse periodic axes (the Bloch mode "
                                                            'the lattice holds)'))
        elif feature.type == 'dipole_source':
            issues.append(Issue('info', feature_issue_path, 'with Bloch k != 0 the dipole drives the real field set '
                                                            'only: every period image repeats it with the Bloch phase '
                                                            '(a phased array)'))
    return issues


def _domain_requirements(settings_values):
    return DomainRequirements(boundary_config=_boundary_config(settings_values),
                              courant_factor=float(settings_values['courant_factor']))


# ---- materials ----
_INDEX_TEXT = ('em_fdtd_3d stores materials as a uint16 index of distinct (eps_xx, eps_yy, eps_zz, sigma_e, mu_r, '
               'sigma_m) combinations: values that vary continuously (x, y expressions, thermo-optic or other field '
               'dependence) are re-quantised to at most 65534 levels per array (bins, member mean; the max deviation '
               'is logged, e.g. ~1e-5 in eps over a range of 1), refreshed at every material change.')
_COMPONENT_TEXT = ('Setting any of eps_xx, eps_yy, eps_zz (or dndT_xx / _yy / _zz) on any material allocates the '
                   'eps_xx and eps_yy arrays (8 B per cell; otherwise E_x and E_y read eps_r) and makes '
                   'heat_em_thermo_optic write all three components.')
_MATERIAL_HELPS = {
    'eps_r': ('Real relative permittivity (>= 1), n^2 of a lossless dielectric, seen by every E component unless '
              'eps_xx / eps_yy / eps_zz override one (it is stored in the eps_r array, which holds eps_zz). Each E '
              'face takes the width-weighted mean of its two cells (solver_defaults.subpixel_smoothing averages '
              'interface faces anisotropically, re-quantised into a uint16 face index). May depend on x, y, z '
              f'and on physics fields. {_INDEX_TEXT} For a dispersive material it is the '
              'background eps_inf.'),
    'eps_zz': (f'z component of the relative permittivity (>= 1), the only one E_z sees; overrides eps_r for E_z. '
               f'Diagonal anisotropy only (crystal axes along x, y, z). {_COMPONENT_TEXT}'),
    'eps_xx': (f'x component of the relative permittivity (>= 1), the only one E_x sees; overrides eps_r for E_x, '
               f'e.g. a uniaxial crystal with its optic axis along x: eps_xx = n_e^2, eps_r = n_o^2. {_COMPONENT_TEXT}'),
    'eps_yy': f'y component of the relative permittivity (>= 1), the only one E_y sees. {_COMPONENT_TEXT}',
    'sigma_e': ('Ohmic loss J = sigma_e E in S/m, default 0 (isotropic, face mean like eps_r); model metals with a '
                'Drude pole. Counted in absorbed_power / em_heat_joule.'),
    'mu_r': 'Real relative permeability (>= 1); each H edge takes the value of its lower cell (i, j, k).',
    'sigma_m': ('Magnetic conductivity in Ohm/m (semi-implicit, each H edge takes its lower cell), default 0; '
                'sigma_m = sigma_e mu0 / eps0 matches the impedance of free space. Not counted in absorbed_power. '
                'Setting it on any material allocates its array (4 B per cell).'),
}
_MATERIAL_TIPS = {
    'eps_zz': ("Permittivity for the electric field along z. When set it replaces 'Relative permittivity' for that "
               'direction.\n\nUse it for crystals whose optic axis lies along z.'),
}
_MATERIAL_PROPERTIES = tuple(replace(param, label='Relative permittivity, z') if param.name == 'eps_zz' else param
                             for param in em_material_params('modes', _MATERIAL_HELPS, _MATERIAL_TIPS))
_MATERIAL_SCHEMA = {'eps_r': 1.0, 'sigma_e': 0.0, 'mu_r': 1.0}
_ANISOTROPY_TRIGGERS = ('eps_xx', 'eps_yy', 'eps_zz', 'dndT_xx', 'dndT_yy', 'dndT_zz')
_OPTIONAL_MATERIAL_SCHEMA = {'eps_xx': (1.0, _ANISOTROPY_TRIGGERS), 'eps_yy': (1.0, _ANISOTROPY_TRIGGERS),
                             'sigma_m': (0.0, ('sigma_m',))}
_Q_ACCUM_INFO = FieldInfo('q_accum', 'Absorbed energy density', unit='J/m^3', cmap='magma')
_TRACK_ABSORPTION = next(param for param in settings_params(_BC_HELP, 'unused', (
    'Accumulate J.E losses (output absorbed_power in W/m^3, state q_accum in J/m^3: sigma_e on the faces split '
    'between the two cells by their half-cell conductance, Drude / Lorentz work time-centred on the pole faces and '
    'split to the dispersive cells) also without an em_heat_joule coupling; costs a pass over the box of the '
    'conducting cells and the pole boxes per step plus 8 B per cell (q_accum and the last window).'))
                         if param.name == 'track_absorption')


# ---- create ----
def _create(ctx, physics_spec, settings_values):
    """EMPhysics3D with the dispersive poles of every material region and absorption tracking; finalize happens in the
    builder."""
    physics = EMPhysics3D(ctx.domain, precision=stepper_precision(ctx.solvers(physics_spec.name)))
    write_dispersive_regions_3d(ctx, physics)
    if settings_values['track_absorption'] or _q_accum_referenced(ctx.spec, physics_spec.name, ctx.registry):
        physics.engine.enable_q_accum = True
    return physics


def write_dispersive_regions_3d(ctx, physics):
    """physics.add_dispersive(pole, window, mask) per pole of every non-empty dispersive material region, its
    background written into each allocated permittivity array (physics.permittivity_arrays) no permittivity property
    owns (the first Drude eps_inf), then materials_changed for those arrays (em_fdtd_3d, em_fdfd_3d; on tetrahedra
    the window is all elements and the mask the region's per-element mask)."""
    materials = ctx.domain.materials
    written = []
    for region in ctx.material_regions:
        if not region.dispersion:
            continue
        window, mask = (ctx.region_window_mask(region) if ctx.domain.grid_kind == 'tensor'
                        else ((slice(0, materials.shape[0]),), region.mask))
        if not bool(mask.any()):
            continue
        for array in physics.permittivity_arrays:
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


# ---- features ----
_DIPOLE_PARAMS = (
    Param('dipole_axis', 'choice', 'z', label='Dipole direction', choices=AXES,
          choice_labels=(('x', 'Along x'), ('y', 'Along y'), ('z', 'Along z')),
          tip='Direction of the current element. It radiates most perpendicular to it and nothing along it.',
          help='Direction of the current element: it drives the one E face of that direction nearest the point (the '
               'containing cell on the other two axes); radiates most broadside, nothing along its axis.'),
    *waveform_params(),
    Param('current_moment', 'float', 1e-12, label='Current moment', unit='A m',
          tip='Peak current times length of the tiny current element. Fields scale linearly with it; the default '
              'is fine.',
          help='Peak current moment I l in A m, injected as J = I l / V on the cell volume V of its E face (dual '
               'length along the dipole times the two cell widths across it), so the radiated field does not depend '
               'on the mesh: a cw dipole in vacuum radiates eta0 k^2 (I l)^2 / (12 pi) W (1e-12 A m at 1 um: '
               '~4e-10 W). ' + POINT_SOURCE_SIGN),
)
_PROBE_PARAMS = (
    Param('component', 'choice', 'ez', label='Field component', choices=tuple(COMPONENT_AXES),
          choice_labels=tuple((name, f"{'Electric' if name[0] == 'e' else 'Magnetic'} field along {name[1]} "
                                     f"({name[0].upper()}_{name[1]})") for name in COMPONENT_AXES),
          tip='Field recorded at the point; electric fields in V/m, magnetic fields in A/m.',
          help='Raw Yee sample of the cell containing the point: E_x / E_y / E_z on its +x / +y / +z face, H_x / '
               'H_y / H_z on its (+y +z) / (+x +z) / (+x +y) edge; H samples are dt/2 older than the recorded time.'),
)
_FLUX_PLANE_PARAMS = (
    Param('normal', 'choice', 'z', label='Plane normal', choices=AXES,
          choice_labels=tuple((axis, f'Along {axis}') for axis in AXES),
          tip='Axis the plane is perpendicular to; positive flux flows toward its + direction.',
          help='Axis normal to the plane; power > 0 flows toward +normal.'),
    Param('position', 'float', 0.0, label='Plane position', unit='m',
          tip='Coordinate of the plane along its normal, from the lower domain edge. It snaps to the nearest cell '
              'edge.',
          help='Coordinate [m] along the normal; the plane snaps to the nearest cell edge of the physical window and '
               'must have a cell on both sides (an edge on a non-PML outer face is refused). The plane spans the '
               'whole physical window across.'),
) + FLUX_PARAMS
_DFT_PLANE_PARAMS = (
    Param('position', 'float', 0.0, label='Layer height', unit='m',
          tip='Height of the horizontal layer whose field map is recorded.',
          help='Height z [m] of the recorded window layer (the cell containing it, an edge goes to the upper cell); '
               'the layer is always normal to z.'),
) + spectral_params(5, 'Each wavelength stores three complex64 window layers (24 B per window xy cell).',
                    'Each wavelength stores one field map; a few are typical.')
_FLUX_TEXT = ("Result 'flux_spectrum': power a.u. (dt-weighted spectra, Re(E x H*) without the 1/2, E averaged across "
              'the plane and over two time steps as the 2D flux monitor), power_per_cw in W = power / (2 |S|^2) '
              'when every source shares one waveform (the cw time-averaged power at unit source amplitude). With a '
              'half_storage stepper the spectra carry the fp16 field noise (~1e-3 of the peak field, ~-60 dB in '
              'power): integrated powers keep ~1e-3, deep spectral nulls are not resolved.')


def _apply_dipole(ctx, solver, feature_spec, values, selection):
    name = single_object(selection, feature_spec, 'point')
    x, y = ctx.point(name)
    return solver.add_dipole(x, y, ctx.height(name), values['dipole_axis'], values['current_moment'],
                             wavelength=values['wavelength'], bandwidth=values['bandwidth'],
                             waveform=values['waveform'])


def _apply_probe(ctx, solver, feature_spec, values, selection):
    name = single_object(selection, feature_spec, 'point')
    x, y = ctx.point(name)
    return solver.add_field_probe(x, y, ctx.height(name), values['component'])


def _apply_flux_plane(ctx, solver, feature_spec, values, selection):
    return solver.add_flux_plane(values['normal'], values['position'], wavelengths=_monitor_wavelengths(values))


def _apply_flux_box(ctx, solver, feature_spec, values, selection):
    return solver.add_flux_box(*box_corners(ctx, selection, feature_spec), wavelengths=_monitor_wavelengths(values))


def _apply_dft_plane(ctx, solver, feature_spec, values, selection):
    return solver.add_dft_plane(values['position'], wavelengths=_monitor_wavelengths(values))


def _apply_pec(ctx, solver, feature_spec, values, selection):
    if selection.is_all:   # refused by _validate_pec (a full-grid mask, and nothing could propagate)
        raise ValueError(f"pec_domain '{feature_spec.name}' selects 'all': select the conductor objects")
    locator = getattr(solver, 'conformal_locator', None)   # solver_defaults.conformal_boundaries (Dey-Mittra faces)
    for name in selection.objects:
        window, mask = (ctx.object_box_mask(name) if ctx.domain.grid_kind == 'tensor'
                        else ctx.object_window_mask(name))   # tetrahedra: (all elements,), element mask
        if locator is None:
            solver.add_pec(window, mask.copy())
        else:
            solver.add_conformal_pec_region(window, mask.copy(), locator.inside([name]))
    return None


def _dft_result(session, solver, monitor, feature_spec, options):
    return dft_plane_result(monitor, session.domain, freq_index=options.get('freq_index'), physics=solver)


def _validate_point(vctx, physics_spec, feature_spec, values):
    """Points inside the physical box (z: its z param, mid-height when unset)."""
    issues = validate_points_inside(vctx, physics_spec, feature_spec, values)
    size_z = vctx.size_z
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        z = None if shape is None else shape.values.get('z')
        if size_z is not None and z is not None and not -1e-9 * size_z <= z <= size_z * (1.0 + 1e-9):
            issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                f"point '{name}' at z = {z:.10g} m lies outside the domain height [0, {size_z:.10g}] m"))
    return issues


def _validate_dipole(vctx, physics_spec, feature_spec, values):
    """_validate_point plus EMPhysics3D.add_dipole's refusal: a point in the outermost cell layer of a pec face
    parallel to the dipole (its tangential E is pinned there)."""
    issues = _validate_point(vctx, physics_spec, feature_spec, values)
    axes = window_axes(vctx.mesh)
    if issues or axes is None:
        return issues
    faces = face_types(physics_spec)
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is None:
            continue
        z = shape.values.get('z')
        coordinates = (shape.values.get('x'), shape.values.get('y'), 0.5 * vctx.size_z if z is None else z)
        for axis, (edges, first, last), coordinate in zip(AXES, axes, coordinates):
            if axis == values['dipole_axis'] or not isinstance(coordinate, (int, float)):
                continue
            cell = containing_cell(coordinate, edges, first, last)
            for side, layer in (('lo', first), ('hi', last)):
                if faces[f'{axis}_{side}'] == 'pec' and cell == layer:
                    issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                        f"point '{name}' lies in the outermost cell layer of the pec face {axis}_{side}, "
                                        f"where the tangential E of a {values['dipole_axis']} dipole is pinned to 0; "
                                        f'move it at least one cell inside'))
    return issues


def _validate_flux_plane(vctx, physics_spec, feature_spec, values):
    path = feature_path(physics_spec, feature_spec)
    issues = wavelength_range_issues(values, path)
    sizes = domain_sizes(vctx)
    axis = AXES.index(values['normal'])
    if sizes is not None:
        if not 0.0 <= values['position'] <= sizes[axis]:
            issues.append(Issue('error', path, f"position {values['position']:.6g} m lies outside the domain along "
                                               f"{values['normal']} [0, {sizes[axis]:.6g}] m"))
            return issues
    axes = window_axes(vctx.mesh)
    if axes is not None:
        message = plane_edge_message(axes[axis], nearest_window_edge(axes[axis], values['position']), values['normal'])
        if message is not None:
            issues.append(Issue('error', path, f'the plane {message}'))
    return issues


def _validate_flux_box(vctx, physics_spec, feature_spec, values):
    """Wavelength range and the face snapping of FluxBox3D (a cell on both sides of every face)."""
    path = feature_path(physics_spec, feature_spec)
    return wavelength_range_issues(values, path) + box_edge_issues(vctx, path, feature_spec, 'flux box')


def _validate_pec(vctx, physics_spec, feature_spec, values):
    """'all' is refused: it would pin every E face (and allocate a full-grid mask)."""
    try:
        entities = expand_entities(vctx.spec, feature_spec.selection)
    except (ValueError, KeyError, TypeError):   # reported by the selection checks
        return []
    if 'all' not in entities:
        return []
    return [Issue('error', feature_path(physics_spec, feature_spec),
                  "a perfect electric conductor on 'all' fills the whole grid, so no field can exist; select the "
                  'conductor objects instead')]


def _validate_dft_plane(vctx, physics_spec, feature_spec, values):
    path = feature_path(physics_spec, feature_spec)
    issues = wavelength_range_issues(values, path)
    if vctx.size_z is not None and not 0.0 <= values['position'] <= vctx.size_z:
        issues.append(Issue('error', path, f"position z = {values['position']:.6g} m lies outside the domain height [0, "
                                           f'{vctx.size_z:.6g}] m'))
    return issues


def _flux_plane_memory(values, n_selected_cells, mesh):
    n_wavelengths = int(values['n_wavelengths'])
    cells = plane_cells(mesh, AXES.index(values['normal']), window=True)
    return MemoryEstimate.from_items(((f'plane spectra (4 x {n_wavelengths} wavelengths x complex128)',
                                       4 * n_wavelengths * COMPLEX128_BYTES * cells),))


def _dft_plane_memory(values, n_selected_cells, mesh):
    n_wavelengths = int(values['n_wavelengths'])
    return MemoryEstimate.from_items(((f'layer spectra (3 x {n_wavelengths} wavelengths x complex64)',
                                       3 * n_wavelengths * COMPLEX64_BYTES * plane_cells(mesh, 2, window=True)),))


def _pec_memory(values, n_selected_cells, mesh):
    """Cropped bool masks on the selection's 3D bounding box."""
    return MemoryEstimate.from_items((('PEC box masks (bool)', BOOL_BYTES * int(n_selected_cells)),))


_FEATURES = (
    FeatureType('dipole_source', 'Dipole source', 'source', 'point', params=_DIPOLE_PARAMS, object_kinds=('point',),
                max_entities=1,
                description='Electric point dipole (current element) along x, y or z on the E face nearest the point '
                            '(point z param = height, mid-height when unset); calibrated to the live face coefficient, '
                            'so the moment holds in any medium. A dipole in the outermost cell layer of a pec face '
                            'parallel to it is refused (its tangential E is pinned there); normal to a pec face it may '
                            'sit next to it (e.g. a monopole on a ground plane).',
                summary='A tiny oscillating current element that radiates a spherical wave.',
                apply=_apply_dipole, validate=_validate_dipole),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='timeseries',
                description='Time series of one raw field component in the cell containing the point (t in s, V/m or '
                            'A/m), sampled after every step.',
                summary='Records one field component at a point over time.',
                apply=_apply_probe, result=probe_result_callback, validate=_validate_point),
    resonance_monitor_feature(
        em_resonance_probe(_PROBE_PARAMS[0], ('hx', 'hy', 'hz'), _apply_probe, validate_point=_validate_point),
        RESONANCE_SUMMARY,
        description_note='3D: the raw Yee sample of the field_probe (point z param = height, mid-height when unset); '
                         'H samples are dt/2 older, which the reference time accounts for.'),
    FeatureType('flux_plane', 'Flux plane', 'monitor', None, params=_FLUX_PLANE_PARAMS, result_kind='flux_spectrum',
                description='Spectral power through a plane across the whole physical window at the cell edge nearest '
                            f'position, > 0 toward +normal. {_FLUX_TEXT}',
                summary='Measures the power crossing a plane, per wavelength.',
                apply=_apply_flux_plane, result=object_result(flux_result, with_solver=True),
                validate=_validate_flux_plane, memory_estimate=_flux_plane_memory),
    FeatureType('flux_box', 'Flux box', 'monitor', 'domain', params=FLUX_PARAMS, object_kinds=('box',),
                max_entities=1, result_kind='flux_spectrum',
                description='Net outward spectral power through the six faces of a box object, each face snapped to '
                            'the nearest cell edge (trapezoid weights along the face edges, corners shared); every face '
                            f'needs a cell on both sides. {_FLUX_TEXT} Enclose a source to get its radiated power.',
                summary='Measures the total power leaving a box, per wavelength.',
                apply=_apply_flux_box, result=object_result(flux_result, with_solver=True),
                validate=_validate_flux_box, memory_estimate=box_face_spectra_memory),
    FeatureType('dft_plane', 'Field map layer', 'monitor', None, params=_DFT_PLANE_PARAMS, result_kind='dft_fields',
                description="Running DFT of E_x, E_y, E_z averaged to the cell centres of one window layer (cells "
                            f"containing z). Result 'dft_fields': x / y centres and the extra 'z' [m]; quantity "
                            f"'{DFT_PLANE_QUANTITY}'; intensity (a.u.) with freq_index; intensity_per_cw_factor as "
                            'the 2D DFT monitor.',
                summary='Records the field pattern of a horizontal layer at chosen wavelengths.',
                apply=_apply_dft_plane, result=_dft_result, validate=_validate_dft_plane,
                memory_estimate=_dft_plane_memory),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor cells: every E face between two selected cells is pinned to 0 '
                            '(the surface lies at the centres of the outermost conductor cells; the normal E on the '
                            "surface stays free). Select objects: 'all' is refused. With "
                            'solver_defaults.conformal_boundaries (Dey-Mittra) every H face between four cell centres '
                            'that the drawn surface cuts takes the free lengths of its four E edges (bisected along '
                            'the links from the free centre) over its free area, floored so the unchanged time step '
                            'stays stable; E faces between two conductor cells stay pinned, faces touching a pec wall '
                            'layer, the window edge or another pec_domain without the option stay staircased, and '
                            'sources, ports, poles and the stability check keep the cell mask.',
                summary='Makes the selected objects perfect metal.',
                apply=_apply_pec, validate=_validate_pec, memory_estimate=_pec_memory),
) + SOURCE_FEATURES   # plane wave, mode source / monitor, far field (definition_sources3d)


# ---- physics validation and memory ----
def _validate_z_faces(settings_values, path):
    lo, hi = settings_values['bc_z_lo'], settings_values['bc_z_hi']
    if (lo == 'periodic') != (hi == 'periodic'):
        return [Issue('error', path, f'periodic boundaries must be paired: bc_z_lo is {lo}, bc_z_hi is {hi}')]
    return []


_VALIDATE_EM = make_validate_harmonic_fdtd(make_validate_physics(_SLOT_ARRAYS))


def _half_storage_issues(vctx, physics_spec, settings_values, path):
    """half_storage stepper: errors for a frequency_domain study (harmonic FDTD normalisation and shutoff energy) and
    for dispersive materials (the pole kernels read float32 E); the TFSF / mode features refuse it by their precisions."""
    if resolved_stepper_precision(vctx.solvers(physics_spec)) != 'half_storage':
        return []
    issues = []
    if vctx.study_type == 'frequency_domain':
        issues.append(Issue('error', path, 'half_storage (fp16 fields) does not run the harmonic FDTD of a '
                                           'frequency_domain study: use a single-precision stepper solver'))
    from ...materials import effective_material_data   # lazy: physics -> materials only inside functions
    for material in vctx.spec.materials:
        if not material.selection:
            continue
        try:
            _, dispersion = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            continue
        if dispersion:
            issues.append(Issue('error', path, f"material '{material.name}' has dispersion poles, which half_storage "
                                               '(fp16 fields) does not support: use a single-precision stepper solver'))
    return issues


def _validate(vctx, physics_spec, settings_values):
    path = ('physics', physics_spec.name)
    return (_validate_z_faces(settings_values, path)
            + list(_VALIDATE_EM(vctx, physics_spec, settings_values))
            + _half_storage_issues(vctx, physics_spec, settings_values, path)
            + _bloch_issues(vctx, physics_spec, settings_values, path))


def _materials_vary(context, n_cells):
    """True when the index build may see more than one material combination: a material on part of the grid, a
    property depending on more than parameters (x, y, fields) or a coupling into this physics."""
    name = context.physics_spec.name
    if any(coupling.enabled and coupling.target == name for coupling in context.spec.couplings):
        return True
    if any(region.box_cells < n_cells for region in context.material_regions):
        return True
    for material in context.spec.materials:
        for value in (material.properties.values() if isinstance(material.properties, dict) else ()):
            try:
                if not referenced_names(value) <= set(context.parameters):
                    return True
            except ExpressionError:   # reported by validate
                return True
    return False


def _pole_box_cells(region, shape, periodic):
    """Cells of engine3d's pole box of a material region: its bounding box one cell longer on the lo side, the whole
    axis on a periodic axis it starts at cell 0 of (box_cells when the bounds are unknown)."""
    if len(region.box_bounds) != 3:
        return region.box_cells
    cells = 1
    for (first, last), count, wraps in zip(region.box_bounds, shape, periodic):
        cells *= count if wraps and first == 0 else last - first + 1 + (1 if first > 0 else 0)
    return cells


def _step_launches(settings_values, context, n_sets):
    """(kernel launches per step, those of the numba-twinned Yee / CPML kernels): Yee / CPML / conformal field kernels
    (x their host overhead weight), pole kernels and the features' kernels."""
    features = [feature for feature in context.physics_spec.features if feature.enabled]
    kinds = {feature.type for feature in features}
    cpml_faces = sum(1 for face in FACES_3D if settings_values.get(f'bc_{face}') == 'cpml')
    conformal = context.spec.solver_defaults.conformal_boundaries is True and 'pec_domain' in kinds
    numba_launches = _FIELD_LAUNCH_WEIGHT * n_sets * (2 + 2 * cpml_faces)
    memory_items = context.memory.items if context.memory is not None else ()
    poles = any(label.endswith(_POLE_ITEM_LABEL) and size > 0 for label, size in memory_items)
    feature_launches = sum(0 if feature.type == 'far_field' and 'flux_box' in kinds   # shares the box's DFTs
                           else _FEATURE_LAUNCHES.get(feature.type, _DEFAULT_FEATURE_LAUNCHES) for feature in features)
    launches = (numba_launches + _FIELD_LAUNCH_WEIGHT * n_sets * (1 if conformal else 0)
                + (n_sets * _POLE_LAUNCHES if poles else 0) + feature_launches)
    return launches, numba_launches


def _numba_stepped(item_label):
    """Memory items (_memory_estimate labels) the numba twins stream (kernels3d _FIELD_HOST_TWINS: Yee, CPML, ohmic
    loss): fields, material index, CPML psi slabs, the absorption accumulator."""
    return item_label.startswith(('fields ', 'material index', 'CPML psi slabs', 'absorption accumulator'))


def _numpy_twin_ops(settings_values, mesh, n_sets):
    """Extra ops of the numpy field twins (cpu backend without numba): numpy passes over the grid and the CPML slabs,
    each priced as a quarter of the cpu 'fused' profile call (4 numpy passes over 5 float32 arrays)."""
    shape = (int(mesh.nx), int(mesh.ny), int(mesh.nz))
    parts = [(_NUMPY_FIELD_PASSES * n_sets, int(mesh.n_cells))]
    for face, npml in mesh.npml_sides.items():
        if settings_values.get(f'bc_{face}') == 'cpml' and npml:
            axis_index = AXES.index(face[0])
            other = shape[0] * shape[1] * shape[2] // shape[axis_index]
            parts.append((_NUMPY_CPML_PASSES * n_sets, int(npml) * other))
    ops = []
    for passes, cells in parts:
        launches = max(1, -(-passes // _NUMPY_PASSES_PER_LAUNCH))
        total = launches * _NUMPY_BYTES_PER_LAUNCH_CELL * float(cells)
        ops += [Elementwise(0.6 * total, 0.4 * total, launches, numba=False), PythonCalls(passes)]
    return tuple(ops)


def _time_estimate(settings_values, mesh, context):
    """Explicit 3D Yee engine: per Courant step (mesh dt) its arrays (fields, uint16 index, CPML slabs, poles, loss
    accumulators) and its features' streamed once in the counted launches; on the cpu backend without numba also the
    numpy field twins' passes over the grid and the CPML slabs."""
    n_sets = 2 if _bloch_active(settings_values) else 1
    launches, numba_launches = _step_launches(settings_values, context, n_sets)
    numpy_twins = context.backend == 'cpu' and not numba_available()
    estimate = explicit_step_estimate(context, 'Yee 3D', mesh.dt, launches=launches, numba=not numpy_twins,
                                      numba_items=_numba_stepped, numba_launches=numba_launches)
    if not numpy_twins:
        return estimate
    item = estimate.items[0]
    return replace(estimate, items=(replace(item, ops=item.ops + _numpy_twin_ops(settings_values, mesh, n_sets)),))


def _memory_estimate(settings_values, mesh, context):
    """Fields, material index, CPML slabs, pole boxes (meshing's 3D selection boxes), absorption accumulators,
    probe traces, harmonic accumulators (run); the index build's x-slab temporaries (int64 codes only when materials
    vary) as the build extra."""
    n_cells = int(mesh.n_cells)
    shape = (int(mesh.nx), int(mesh.ny), int(mesh.nz))
    half = resolved_stepper_precision(context.solvers) == 'half_storage'
    field_bytes = 2 if half else FLOAT32_BYTES
    n_sets = 2 if _bloch_active(settings_values) else 1   # Bloch k != 0: real + imaginary field set
    set_text = ', real + imaginary set' if n_sets == 2 else ''
    items = [(f"fields (6 x {'float16' if half else 'float32'}{set_text})", n_sets * 6 * field_bytes * n_cells),
             ('material index (uint16)', 2 * n_cells),
             (f'CPML psi slabs{set_text}', n_sets * cpml_psi_bytes(
                 {face: cells for face, cells in mesh.npml_sides.items()
                  if settings_values.get(f'bc_{face}') == 'cpml'}, shape))]
    q_accum = bool(settings_values['track_absorption']) or _q_accum_referenced(
        context.spec, context.physics_spec.name, context.registry)
    pole_bytes = 0
    periodic = tuple(settings_values[f'bc_{axis}_lo'] == 'periodic' for axis in AXES)
    for region in context.material_regions:
        if region.n_drude or region.n_lorentz:   # weights, cb (+ face work while accumulating), J / P, dP
            per_cell = 6 + (3 if q_accum else 0) + n_sets * (3 * region.n_drude + 6 * region.n_lorentz)
            pole_bytes += per_cell * FLOAT32_BYTES * _pole_box_cells(region, shape, periodic)
    items.append((_POLE_ITEM_LABEL, pole_bytes))
    if q_accum:
        items.append(('absorption accumulator (q_accum, float32)', FLOAT32_BYTES * n_cells))
    trace_items, run_items, host_items = _point_trace_items(context, mesh)
    transient_items = []
    if q_accum:   # reset_q_accum allocates the last window lazily; take_absorbed_power returns a new array per transfer
        run_items.append(('last absorption window (float32)', FLOAT32_BYTES * n_cells))
        transient_items.append(('absorbed power transfer', 'mean power density of one transfer (float32)',
                                FLOAT32_BYTES * n_cells))
    n_wavelengths = _harmonic_wavelength_count(context)
    if n_wavelengths:
        window = plane_cells(mesh, 2, window=True) * (shape[2] - mesh.npml_sides.get('z_lo', 0) - mesh.npml_sides.get('z_hi', 0))
        run_items.append((f'harmonic accumulators (3 x {n_wavelengths} wavelengths x complex64)',
                          3 * n_wavelengths * COMPLEX64_BYTES * window))
    index_bytes_per_cell = (_INDEX_VARYING_BYTES_PER_CELL if _materials_vary(context, n_cells)
                            else _INDEX_UNIFORM_BYTES_PER_CELL)
    return MemoryEstimate.from_items(items + trace_items,
                                     build_extra=index_bytes_per_cell * min(n_cells, _INDEX_CHUNK_CELLS),
                                     host_items=host_items, run_items=run_items, transient_items=transient_items)


EM_FDTD_3D = PhysicsType(
    name=TYPE_NAME,
    label='Electromagnetic Waves, time domain (3D)',
    family=EM_FAMILY,
    group='em',
    default_instance_name='em',
    description=('3D finite-difference time-domain solver (Yee): E_x, E_y, E_z on the +x / +y / +z faces of each cell '
                 'in V/m, H_x, H_y, H_z on its (+y +z) / (+x +z) / (+x +y) edges in A/m, H half a time step earlier. '
                 'Materials eps_r or the diagonal eps_xx / eps_yy / eps_zz, sigma_e, mu_r, sigma_m plus Drude / '
                 'Lorentz poles (on the bounding box of their material), stored as a uint16 material index with '
                 'coefficient tables (at most 65534 distinct combinations; continuously varying values, e.g. '
                 'thermo-optic, are re-quantised to fit). Boundaries per face: cpml (z faces too), pec, pmc, '
                 'periodic (optional Bloch wave vector: a second, imaginary field set), symmetric / antisymmetric '
                 'mirror. Sources: electric point dipole, TFSF plane wave, mode source (A5 cross-section modes); '
                 'monitors: field probe, resonance finder, flux plane, flux box, DFT field-map layer, mode monitor, '
                 'far-field pattern; '
                 'outputs the raw components, e_abs, e_intensity, h_abs (cell centres), eps_r (the eps_zz array '
                 'E_z sees) and '
                 'absorbed_power, shown as z cut planes. Couplings: heat_em_thermo_optic (index slots of the '
                 'allocated permittivity arrays) and em_heat_joule into 3D heat. Memory ~26 B per cell plus 12 B of '
                 'material arrays (+8 B with eps_xx / eps_yy, +4 B with sigma_m, +8 B while tracking absorption); a '
                 "stepper solver with precision half_storage stores the fields as fp16 (14 B per cell; time_dependent "
                 'only, no dispersion poles, plane wave, mode ports or resonance finder; ~1e-3 relative noise floor, '
                 'fields stored in units scaled from the dipole moments and read back in V/m, A/m). It '
                 'advances one 3D Courant step per runner tick (every must be 1); the instance is usually named '
                 "'em'. Runs on the gpu and the cpu backend (cpu: numpy twins of every kernel, numba for the Yee "
                 'updates, CPML and ohmic loss when installed, far slower: small models and tests; half_storage '
                 f'gpu only). {HARMONIC_FDTD_TEXT}'),
    summary=('Light in 3D, simulated step by step in time, so one short pulse gives a whole spectrum. Use it for '
             'structures that are not uniform along any axis, for example antennas, particles and waveguide ends.'),
    settings=_settings(),
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=_MATERIAL_SCHEMA,
    optional_material_schema=_OPTIONAL_MATERIAL_SCHEMA,
    state_fields=(*(EMRenderer3D.field_infos[name] for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz', 'eps_r')),
                  _Q_ACCUM_INFO),
    output_fields=tuple(EMRenderer3D.field_infos.values()),
    source_slots=source_slot_docs(_SLOT_ARRAYS),
    supported_studies=('time_dependent', 'frequency_domain'),
    default_study={'type': 'time_dependent', 'settings': {'duration': 50e-15}},
    uses_dispersion=True,
    create=_create,
    domain_requirements=_domain_requirements,
    steps_in_time=steps_in_time,
    shortest_wavelength=make_source_shortest_wavelength({feature.name: feature for feature in _FEATURES}),
    validate=_validate,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    dimensions=(3,),
    solver_roles={'stepper': HALF_STORAGE_STEPPER_ROLE},
    subpixel_smoothing=True,
    conformal_boundaries=True,   # pec_domain Dey-Mittra H faces (conformal_pec3d; 3D locator)
    conformal_dimensions=(3,),
    backends=('gpu', 'cpu'),   # core.kernels twins (numpy, numba field kernels); half_storage refused on the cpu
)


def _selftest_cases():
    """The em_fdtd_3d / em_fdfd_3d SelfTestCases (selftests_fem3d: the script-level tetrahedral path), imported on the
    first registry.selftests()."""
    from .selftests import SELFTEST_CASES
    from .selftests_backend import EM3D_BACKEND_SELFTEST_CASES
    from .selftests_bloch3d import BLOCH_3D_CASES
    from .selftests_conformal3d import CONFORMAL_3D_CASES
    from .selftests_cylindrical3d import CYLINDRICAL_3D_CASES
    from .selftests_fdfd3d import FDFD_3D_CASES
    from .selftests_fem3d import FEM_3D_CASES
    from .selftests_materials import MATERIAL_CASES
    from .selftests_mirror3d import MIRROR_3D_CASES
    from .selftests_resonance3d import RESONANCE_3D_CASES
    from .selftests_sources3d import SOURCE_CASES
    return (*SELFTEST_CASES, *MATERIAL_CASES, *SOURCE_CASES, *RESONANCE_3D_CASES, *MIRROR_3D_CASES, *BLOCH_3D_CASES,
            *EM3D_BACKEND_SELFTEST_CASES, *CONFORMAL_3D_CASES, *CYLINDRICAL_3D_CASES, *FDFD_3D_CASES, *FEM_3D_CASES)


def register(registry):
    """Register em_fdtd_3d, then em_fdtd_3d's frequency-domain twin em_fdfd_3d (definition_fdfd3d: the FDTD material
    Params stay first per name), their templates (dipole radiation, photonic-crystal slab bands, PEC box eigenmodes,
    nanobeam mirror cell bands) and the selftest loader."""
    from .definition_fdfd3d import EM_FDFD_3D
    from .selftests import DIPOLE_TEMPLATE
    from .selftests_bloch3d import SLAB_BANDS_TEMPLATE
    from .selftests_fdfd3d import NANOBEAM_CELL_TEMPLATE, PEC_BOX_EIGENMODES_TEMPLATE
    registry.register_physics(EM_FDTD_3D)
    registry.register_physics(EM_FDFD_3D)
    registry.register_template(*DIPOLE_TEMPLATE)
    registry.register_template(*SLAB_BANDS_TEMPLATE)
    registry.register_template(*PEC_BOX_EIGENMODES_TEMPLATE)
    registry.register_template(*NANOBEAM_CELL_TEMPLATE)
    registry.register_selftests(_selftest_cases)

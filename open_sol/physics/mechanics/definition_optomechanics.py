"""Model-layer definition of the solid_mechanics feature optomechanical_coupling (the vacuum g0 table of the
optomechanics study): params, validation, apply, result, templates and self-tests; registered by definition.py. No cupy
at module level."""
import numpy as np

from ...core.constants import FACES
from ...model.definition_helpers import evaluates_to
from ...model.geometry import expand_entities
from ...model.schema import FeatureType, Issue, Param, feature_path, subpixel_smoothing_physics
from .definition_common import OPTOMECHANICS_STUDY, plain_value
from .optomechanics import (NORMALISATIONS, OPTOMECHANICS_TABLE_COLUMNS, SCATTERING, OptomechanicalCoupling,
                            optomechanical_table_result)
from .selftests_optomechanics import (SILICON_DISK_G0_DESCRIPTION, SILICON_DISK_G0_LABEL, SILICON_RING_G0_DESCRIPTION,
                                      SILICON_RING_G0_LABEL, silicon_disk_wgm_g0_spec, silicon_ring_breathing_g0_spec)

OPTICAL_TYPES = ('em_modes', 'em_fdfd_ez', 'em_fdfd_hz', 'em_fdfd_bor')
_FEATURE = 'optomechanical_coupling'
_MODES_TYPE = 'em_modes'
_FDFD_HZ_TYPE = 'em_fdfd_hz'
_BOR_TYPE = 'em_fdfd_bor'
_SOLID_3D_TYPE = 'solid_mechanics_3d'   # = definition.SOLID_3D_TYPE_NAME
_TRIANGLES, _TETRAHEDRA = 'triangles', 'tetrahedra'
_MIRROR_TYPES = ('symmetric', 'antisymmetric')
_BOR_WALL_TYPES = ('pec', 'pmc')        # em_fdfd_bor z faces that may be mirror planes

_PARAMS = (
    Param('optical_physics', 'ref', '', label='Optical physics', ref_kind='physics',
          help='Name of the em_modes (cross-section mode at its wavelength, group_index on), em_fdfd_ez or em_fdfd_hz '
               '(in-plane resonance of its eigen-solve near its wavelength setting) or em_fdfd_bor (body-of-revolution '
               'resonance on a triangles mesh; the solid axisymmetric on a triangles mesh or a solid_mechanics_3d on '
               'tetrahedra) physics whose mode couples to the vibration modes; the optomechanics study solves it before '
               'the solid.',
          tip='The light physics whose mode is moved by the vibration.'),
    Param('optical_mode', 'int', 1, label='Optical mode', min=1,
          help='1-based mode number of the em_modes solve (descending Re beta^2), or resonance number of the '
               'em_fdfd_ez / em_fdfd_hz / em_fdfd_bor eigen-solve (nearest its wavelength first).',
          tip='Which optical mode (1 = the first one found) is used.'),
    Param('length', 'float', 6.283185307e-5, label='Out-of-plane length', unit='m', min=0.0, min_exclusive=True,
          help='Length L along z that turns the per-metre mass into kg: the ring circumference 2 pi R or waveguide '
               'length for a cross-section (em_modes), the thickness of the device for an in-plane resonator '
               '(em_fdfd_ez / em_fdfd_hz). m_eff = L x the 2D modal mass; g_om does not depend on it, g0 scales as 1 / '
               'sqrt(L). Default 2 pi x 10 um. Unused with em_fdfd_bor optics (whole rings / the 3D body: m_eff in kg).',
          tip='Ring circumference (or waveguide length) for a cross-section, device thickness for an in-plane model.'),
    Param('moving_boundary', 'bool', True, label='Moving boundary',
          help='Include the moving-boundary part (permittivity jumps carried by the mode shape).',
          tip='Include the effect of the boundaries moving.'),
    Param('photoelastic', 'bool', True, label='Photoelastic',
          help='Include the photoelastic part (photoelastic_p11 / _p12 / _p44 of the solid cells, p44 unset = '
               'isotropic (p11 - p12) / 2; a material with a crystal entry that has photoelastic data and sets no '
               'photoelastic_* property itself: the full rotated p tensor of the crystal).',
          tip='Include the stress-induced index change.'),
    Param('normalisation', 'choice', 'max', label='Displacement reference', choices=NORMALISATIONS,
          choice_labels=(('max', 'Largest displacement'), ('point', 'At a point')),
          help="Node where the mode shape is scaled to |u| = 1 for g_om and m_eff: 'max' the peak node, 'point' the "
               'solid node nearest (reference_x, reference_y). g0 does not depend on it; g_om and m_eff do.',
          tip='Where the displacement used for the coupling per metre and the effective mass is measured.\n\n'
              'The vacuum coupling rate does not depend on it.'),
    Param('reference_x', 'float', 0.0, label='Reference x', unit='m', visible_if=(('normalisation', ('point',)),),
          help="x of the reference point (normalisation 'point').", tip='x of the point where the displacement is '
                                                                          'measured.'),
    Param('reference_y', 'float', 0.0, label='Reference y', unit='m', visible_if=(('normalisation', ('point',)),),
          help="y of the reference point (normalisation 'point').", tip='y of the point where the displacement is '
                                                                          'measured.'),
    Param('mechanical_quality_factor', 'float', None, label='Mechanical quality factor', unit='1', min=0.0,
          min_exclusive=True, optional=True,
          help='Q_m of every vibration mode for the brillouin_gain column (em_modes only): Brillouin gain G_B = 2 Q_m '
               '|kappa|^2 / (omega m\' Omega^2) [1/(W m)], m\' = m_eff / length, kappa = d beta / d alpha (forward '
               'intramodal: pump and Stokes in the optical mode, k_z = 0 vibration; its real part) or the coupling to '
               'the counter-propagating twin (scattering backward). Empty: the column reads NaN (the modal analysis is '
               'undamped). Typical 100-1000 (silicon wires at room temperature).',
          tip='Damping of the vibrations (quality factor), used only for the Brillouin gain of a waveguide mode; '
              'empty leaves the gain out.'),
    Param('scattering', 'choice', 'forward', label='Brillouin scattering', choices=SCATTERING,
          choice_labels=(('forward', 'Forward (vibrations in the cross-section)'),
                         ('backward', 'Backward (vibrations travelling along the guide)')),
          help="forward: the plane-strain (k_z = 0) vibration modes, g0 = the frequency shift of the optical mode, "
               "brillouin_gain the forward intramodal gain. backward (em_modes only): the study sets the solid's k_z "
               'to 2 Re beta of the optical mode before its modal solve (the guided 2.5D modes with u_z, as the '
               'axial_wavenumber setting, which stays empty; supports optional) and the overlap pairs the mode with its '
               'counter-propagating twin (E_x, E_y, -E_z) and the conjugated phonon strain: S_zz = k_z u_z enters the '
               'p11 / p12 terms, the shears S_xz, S_yz (p44) pair E_x, E_y with E_z, the E_z terms of the moving '
               'boundary and photoelastic sums flip sign. The g_om / g0 / dn_eff_dx columns then hold the '
               'forward-backward coupling rate (a ring: between the counter-propagating modes), brillouin_gain the '
               'backward gain. Every optomechanical_coupling of one solid needs the same scattering (backward: the same '
               'optical physics and mode). The guided modes near 2 v_L n_eff / lambda (tens of GHz) need the solid '
               "target_frequency set (0 finds the lowest, flexural ones).",
          tip='Forward: the vibrations within the cross-section shift the light. Backward: vibrations travelling '
              'along the guide reflect the light into the opposite direction (backward Brillouin scattering); waveguide '
              'mode only.'),
    Param('prescribed_normal_displacement', 'float', None, label='Prescribed boundary motion', unit='m', optional=True,
          help='em_fdfd_bor optics only: a uniform outward motion q . n [m] of every optical permittivity interface '
               'bordering the solid (its stiff elements) without a mechanical mode; the table extra prescribed_g_om '
               '[rad/s/m] is the moving-boundary d omega / d alpha of it (q . n = 1 on a sphere: -omega / R). Empty: '
               'prescribed_g_om None. The vibration-mode rows are computed as usual.',
          tip='Body-of-revolution light only: a uniform outward motion of the solid surface, whose frequency shift is '
              'reported besides the vibration modes (e.g. 1 m for the shift per metre of a sphere radius).'),
    Param('axis_x', 'float', 0.0, label='Axis x (3D solid)', unit='m',
          help='solid_mechanics_3d with em_fdfd_bor optics: x of the 3D point (axis_x, axis_y, 0) the body-of-'
               'revolution axis passes through, parallel to z; the optical drawing (r, z) = (x - its x_lo edge, y) maps to '
               '(axis_x + r cos phi, axis_y + r sin phi, z), and q . n is averaged over 64 ring angles.',
          tip='3D solid only: where the rotation axis of the light model sits in the 3D model (x).'),
    Param('axis_y', 'float', 0.0, label='Axis y (3D solid)', unit='m',
          help='solid_mechanics_3d with em_fdfd_bor optics: y of the axis point (see axis_x).',
          tip='3D solid only: where the rotation axis of the light model sits in the 3D model (y).'),
)


def backward_scattering(physics_spec) -> bool:
    """True when an enabled optomechanical_coupling of the solid physics spec asks for backward scattering (raw
    params: the solid is created guided, its k_z set by the optomechanics study)."""
    return any(feature.enabled and feature.type == _FEATURE and isinstance(feature.params, dict)
               and feature.params.get('scattering') == 'backward' for feature in physics_spec.features)

_DESCRIPTION = (
    'Vacuum optomechanical coupling between one optical mode and every vibration mode of the optomechanics study (2D '
    'plane strain, guided 2.5D modes with scattering backward; rho needed in every solid cell). g_om = d omega_optical / d alpha for the displacement alpha Q, Q '
    "= the mode shape scaled to |Q| = 1 at the reference node; moving boundary: each interior cell face whose cells "
    'differ in permittivity contributes Q_n [eps0 (eps_lo - eps_hi) |E_par|^2 - (1/eps_lo - 1/eps_hi) |D_n|^2 / eps0] '
    'times its length (Q_n = mean of its two corner nodes; staircase interfaces); photoelastic: eps0 E* d eps E dA per '
    'solid cell with d eps_ii = -eps_ii^2 (p11 S_ii + p12 (S_jj + S_kk)) and the shear d eps_xy = -eps_xx eps_yy 2 '
    'p44 S_xy (photoelastic_p44, unset = (p11 - p12) / 2), S the mode strain (tensor shear S_xy), S_zz = S_xz = S_yz = '
    '0; the shear term pairs E_x and E_y averaged to the cell centres (2 Re(E_x* E_y)) and is the continuum form (the '
    'optical solvers hold diagonal eps), so em_fdfd_ez has none. A material with a {"model": "crystal"} entry with '
    'photoelastic data and no photoelastic_* property of its own uses the full p of the crystal rotated into model '
    'axes (as the entry expands): Delta(1/eps)_I = p_IJ S_J (Nye, Voigt xx, yy, zz, yz, xz, xy, engineering shear), '
    'd eps_ij = -eps_ii eps_jj Delta(1/eps)_ij, whose xz / yz rows pair E_x / E_y with E_z at the centres (2 Re(E_x* '
    'E_z), em_modes only; continuum form like xy); its stiffness stays the isotropic youngs_modulus / poisson_ratio. '
    'Subpixel smoothing: the face jumps and the photoelastic eps are the unsmoothed cell values (the build\'s '
    'material assignment), the fields, D_n and the mode energy those of the smoothed solve. em_fdfd_ez resonance: g_om = -(omega / 2) Re[(I_mb + '
    'I_pe) / sum eps0 eps |E_z|^2 dA]; em_fdfd_hz '
    'resonance: the same over sum mu0 mu |H_z|^2 dA, E_par on the Yee face itself, |D_n|^2 the mean over the four '
    'nearest normal-E faces, the photoelastic sum over the E_x / E_y faces (face-averaged d eps_xx / d eps_yy); em_modes '
    'cross-section mode (needs group_index): d beta / d alpha = omega (I_mb + I_pe) / (4 P) (1 W, the whole mode) and '
    'the ring conversion at fixed azimuthal order g_om = -(C0 / n_group) d beta / d alpha (column dn_eff_dx = '
    'd n_eff / d alpha at fixed omega; a bent em_modes uses its bend-mapped eps components, exact for bend_model '
    "'exact'). m_eff = length x the consistent-mass "
    'Q^T M Q (x 2 per EM mirror face), x_zpf = sqrt(hbar / (2 m_eff Omega)), g0 = g_om x_zpf. Result \'table\', one '
    f"row per vibration mode: {', '.join(OPTOMECHANICS_TABLE_COLUMNS)} (Hz, kg, m, rad/s/m, 1/m; g0 columns are g0 / "
    '2 pi in Hz, signed: total = moving boundary + photoelastic; brillouin_gain [1/(W m)] = the forward intramodal '
    "G_B = 2 Q_m (d beta / d alpha)^2 / (omega m' Omega^2), m' = m_eff / length, em_modes with "
    'mechanical_quality_factor only, else NaN). scattering backward (em_modes): the vibration modes are the guided '
    'modes at k_z = 2 Re beta (set by the study), |Q| and m_eff include u_z, the overlap is kappa = omega (I_mb + I_pe) '
    '/ (4 P) between the mode and its counter-propagating twin (E_x, E_y, -E_z) with the conjugated phonon strain '
    '(S_zz = k_z u_z, shears S_xz / S_yz with p44 pairing E_x / E_y and E_z; the E_z terms flip sign), g_om = -(C0 / '
    "n_group) Re kappa and g0 the forward-backward coupling rate, brillouin_gain = 2 Q_m |kappa|^2 / (omega m' "
    'Omega^2) (bulk limit: g_B / area). Extras optical_physics, optical_mode, optical_frequency, optical_wavelength, '
    'n_eff, n_group, length, mirror_factor, scattering, axial_wavenumber (None forward), normalisation, '
    'reference_node. Lossy '
    'or open optical modes use |E|^2 forms and the real part (no quasi-normal-mode correction). An EM mirror face needs '
    'a solid roller on the same face (even vibration modes; the odd ones do not couple). The solid field om_integrand '
    '(optomechanics study only) maps the moving-boundary g_om density of the shown vibration mode (first coupling) in '
    'rad/s/m^3: each face term half to each of its cells over the cell area, summing to g_om_moving_boundary. '
    'em_fdfd_bor optics (a resonance ~ exp(-j m phi) on a triangles mesh; the solid axisymmetric on a triangles mesh, or '
    'a solid_mechanics_3d on tetrahedra): every integral carries 2 pi r over the window elements (PML ring, PEC and '
    'interfaces touching them left out); moving boundary on the curved element edges between permittivities (the '
    'eps_phiphi component, isotropic), n out of the first element, q . n [eps0 Delta eps |E_par|^2 - Delta(1/eps) '
    '|D_n|^2 / eps0], |E_par|^2 = |E_phi|^2 + |E_t|^2 and |D_n|^2 the means of both sides, q . n of the solid side; '
    'photoelastic eps0 E* d eps E with the cylindrical strains (rr, phiphi = u_r / r, zz, rz) and isotropic p11 / p12 '
    '/ p44 (crystal tensors are not used); U = sum eps0 eps |E|^2 2 pi r dA. An axisymmetric solid of azimuthal_order '
    'm >= 1 couples through the ring mean of cos(m phi) (zero to round-off); a 3D solid is sampled on 64 angles per (r, '
    'z) point about the axis through (axis_x, axis_y) and phi-averaged (its crown parts drop out). m_eff = Q^T M Q in '
    'kg (whole rings or the 3D body; length and mirror factors unused, a pec / pmc z face used as a mirror plane counts '
    'the drawn half), normalisation point: the solid dof nearest (reference_x, reference_y) (triangles only), no '
    'om_integrand, dn_eff_dx and brillouin_gain NaN. Extras optical_azimuthal_order, optical_q_factor, solid_mesh, '
    'mechanical_azimuthal_order, prescribed_normal_displacement and prescribed_g_om (setting '
    'prescribed_normal_displacement: the rate of that uniform outward motion of the solid surface).')


def crystal_photoelastic_tensor(material):
    """(6, 6) Voigt p in model axes of a spec material's crystal entry (materials.crystal_tensors); None without an
    entry or photoelastic data, or when the material sets a photoelastic_* property itself (user values win:
    isotropic). ValueError for a malformed entry."""
    from ...materials import crystal_entry
    from ...materials.crystal_tensors import entry_settings, model_optical_tensors
    properties = material.properties if isinstance(material.properties, dict) else {}
    if any(name.startswith('photoelastic_') and value is not None for name, value in properties.items()):
        return None
    entry = crystal_entry(material.dispersion)
    if entry is None:
        return None
    crystal, euler_deg, _ = entry_settings(entry)
    return model_optical_tensors(crystal, euler_deg)[0]


def _crystal_regions(ctx):
    """(host full-grid bool mask, (6, 6) p) of every built material region with a crystal photoelastic tensor."""
    from ...core.backend import to_host
    materials = {material.name: material for material in ctx.spec.materials}
    regions = []
    for region in ctx.material_regions:   # 2D masks, alive until release_caches
        tensor = crystal_photoelastic_tensor(materials[region.name])
        if tensor is not None:
            regions.append((np.array(to_host(region.mask), dtype=bool), tensor))
    return regions


def _apply(ctx, solver, feature_spec, values, selection):
    """OptomechanicalCoupling on the solid (the optomechanics study evaluates it), with the crystal regions (tensor
    meshes), the prescribed q . n and the 3D axis point (em_fdfd_bor optics)."""
    reference = ((values['reference_x'], values['reference_y']) if values['normalisation'] == 'point' else None)
    tensor = getattr(ctx.domain, 'grid_kind', 'tensor') == 'tensor'
    coupling = OptomechanicalCoupling(feature_spec.name, values['optical_physics'], values['optical_mode'],
                                      values['length'], values['moving_boundary'], values['photoelastic'], reference,
                                      values['mechanical_quality_factor'], values['scattering'],
                                      _crystal_regions(ctx) if values['photoelastic'] and tensor else (),
                                      prescribed_normal_displacement=values['prescribed_normal_displacement'],
                                      axis_point=(values['axis_x'], values['axis_y'], 0.0))
    solver.optomechanical_couplings.append(coupling)
    return coupling


def _result(session, solver, coupling, feature_spec, options):
    return optomechanical_table_result(coupling)


def _roller_faces(vctx, physics_spec):
    """Outer faces held by an enabled roller (or fixed) feature of the solid."""
    faces = set()
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in ('roller', 'fixed'):
            continue
        try:
            faces.update(entity for entity in expand_entities(vctx.spec, feature.selection) if entity in FACES)
        except (ValueError, TypeError):
            continue
    return faces


def _backward_issues(physics_spec, feature_spec, values, path):
    """Backward scattering: every other enabled coupling of the solid backward on the same optical physics and mode
    (one k_z per modal solve); info without a target_frequency (raw settings)."""
    issues = []
    for feature in physics_spec.features:
        if not feature.enabled or feature.type != _FEATURE or feature.name == feature_spec.name:
            continue
        raw = feature.params if isinstance(feature.params, dict) else {}
        if (raw.get('scattering', 'forward') != 'backward' or raw.get('optical_physics') != values['optical_physics']
                or raw.get('optical_mode', 1) != values['optical_mode']):
            issues.append(Issue('error', path, f"backward scattering sets one k_z = 2 beta for the vibration modes of "
                                               f"'{physics_spec.name}', but optomechanical_coupling '{feature.name}' "
                                               'differs in scattering, optical_physics or optical_mode: use one '
                                               'backward coupling per solid (or another solid_mechanics physics)'))
    settings = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
    if settings.get('target_frequency', 0.0) in (0, 0.0):
        issues.append(Issue('info', path, 'backward scattering with target_frequency 0 finds the lowest guided modes '
                                          '(flexural / torsional); the backward Brillouin modes lie near 2 v n_eff / '
                                          'lambda (tens of GHz in silicon): set the solid target_frequency there'))
    return issues


_PHOTOELASTIC_NAMES = ('photoelastic_p11', 'photoelastic_p12', 'photoelastic_p44')
_PERMITTIVITY_COMPONENTS = ('eps_xx', 'eps_yy', 'eps_zz')
_UNASSIGNED_FACTS = (False, False, (1.0, 1.0, 1.0, 1.0, 0.0, 0.0), False)   # preview index 0: void, vacuum
_THICKNESS_RATIO = 5.0   # length below 1/5 of the solid's narrowest in-plane size: plane stress; above 5x its widest: strain


def _material_facts(vctx):
    """Per spec material as the builder merges it: (stiff, photoelastic, optical key, crystal tensor used); stiff /
    photoelastic True when possibly non-zero (expressions count), key = plain (eps_r, eps_xx, eps_yy, eps_zz, eps_xy,
    sigma_e), ('poles', index) with poles, None when not plain numbers; None for a material that does not merge."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    parameters = vctx.parameters
    facts = []
    for index, material in enumerate(vctx.spec.materials):
        try:
            properties, poles = effective_material_data(material.library, material.properties, material.dispersion)
            tensor = crystal_photoelastic_tensor(material)
        except ValueError:
            facts.append(None)
            continue
        stiff = any(evaluates_to(properties.get(name), parameters, lambda value: value > 0.0)
                    for name in ('youngs_modulus', 'c11'))   # c11: the c_ij of a crystal entry
        photoelastic = (tensor is not None and bool(np.any(tensor != 0.0))) or any(
            evaluates_to(properties.get(name), parameters, lambda value: value != 0.0) for name in _PHOTOELASTIC_NAMES)
        isotropic = properties.get('eps_r', 1.0)
        key = tuple(plain_value(value, parameters) for value in (
            isotropic, *(properties.get(name, isotropic) for name in _PERMITTIVITY_COMPONENTS),
            properties.get('eps_xy', 0.0), properties.get('sigma_e', 0.0)))
        if poles:
            key = ('poles', index)   # dispersive: assumed to differ from every other material
        elif any(value is None for value in key):
            key = None
        facts.append((stiff, photoelastic, key, tensor is not None))
    return facts


def _has_permittivity_jump(image, stiff_by_index, key_by_index):
    """True when an interior preview face touching a solid cell separates two optical keys."""
    key_ids = {}
    ids = np.array([key_ids.setdefault(key, len(key_ids)) for key in key_by_index])[image]
    stiff = np.asarray(stiff_by_index, dtype=bool)[image]
    for axis in (0, 1):
        lo = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        hi = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        if np.any((stiff[lo] | stiff[hi]) & (ids[lo] != ids[hi])):
            return True
    return False


def _thickness_issue(path, length, solid_cells, extent):
    """Plane strain vs plane stress of an in-plane resonator by the out-of-plane length against the solid's bounding
    box on the preview (solid_cells (mx, my), extent (x0, x1, y0, y1) [m]); None when plane strain fits."""
    rows, columns = np.flatnonzero(solid_cells.any(axis=1)), np.flatnonzero(solid_cells.any(axis=0))
    width_x = (rows[-1] - rows[0] + 1) * (extent[1] - extent[0]) / solid_cells.shape[0]
    width_y = (columns[-1] - columns[0] + 1) * (extent[3] - extent[2]) / solid_cells.shape[1]
    narrow, wide = sorted((width_x, width_y))
    if _THICKNESS_RATIO * length <= narrow:
        return Issue('info', path, f'the solid spans at least {narrow:.3g} m in the plane but length (its thickness) is '
                                   f'{length:.3g} m: a thin plate is closer to plane stress than to the plane strain '
                                   'the solid solves (in-plane stiffness up to 1 / (1 - nu^2) too high), so the '
                                   'frequencies, m_eff and g0 are approximate')
    if length >= _THICKNESS_RATIO * wide:
        return None
    return Issue('info', path, f'length {length:.3g} m (the thickness) is comparable to the in-plane size of the solid '
                               f'({narrow:.3g} to {wide:.3g} m): neither plane strain nor plane stress holds, so the '
                               'vibration modes and g0 are approximate (a 3D solve would be needed)')


def _material_issues(vctx, path, values, optical):
    """Infos from the spec materials and their preview: photoelastic on with every solid p = 0, moving boundary on with
    no permittivity jump at a solid cell (not with couplings into the optical physics), in-plane thickness."""
    in_plane = optical.type not in (_MODES_TYPE, _BOR_TYPE)
    facts = _material_facts(vctx)
    if any(fact is None for fact in facts):   # reported at the material
        return []
    preview = vctx.material_map()
    by_index = [_UNASSIGNED_FACTS] + facts
    if preview is None:
        present = [0] + [index + 1 for index, material in enumerate(vctx.spec.materials) if material.selection]
    else:
        present = [int(index) for index in np.unique(preview[0])]
    solid_present = [index for index in present if by_index[index][0]]
    if not solid_present:   # the solid's own material check reports it
        return []
    issues = []
    voids = [vctx.spec.materials[index - 1].name for index in present if by_index[index][3] and not by_index[index][0]]
    if values['photoelastic'] and voids:
        issues.append(Issue('info', path, f"material(s) {', '.join(voids)} carry crystal photoelastic data but no "
                                          'stiffness (youngs_modulus or c11 > 0): void in the solid, so they strain '
                                          'nothing'))
    crystals = [vctx.spec.materials[index - 1].name for index in solid_present if by_index[index][3]]
    if values['photoelastic'] and optical.type == _BOR_TYPE and crystals:
        issues.append(Issue('info', path, f"material(s) {', '.join(crystals)} carry a crystal photoelastic tensor, which "
                                          'is not invariant under rotation about the axis: with em_fdfd_bor optics the '
                                          'overlap uses their isotropic photoelastic_p11 / _p12 / _p44 only'))
    if values['photoelastic'] and not any(by_index[index][1] for index in solid_present):
        names = ', '.join(vctx.spec.materials[index - 1].name for index in solid_present)
        issues.append(Issue('info', path, f'photoelastic is on, but every solid material ({names}) has photoelastic_p11 '
                                          '= _p12 = _p44 = 0 and no crystal photoelastic data: the photoelastic '
                                          'columns read 0'))
    if preview is None:
        if in_plane:
            issues.append(Issue('info', path, 'in-plane resonator: the solid is plane strain (a structure long along '
                                              'z); a thin film seen from the top is closer to plane stress, so g0 is '
                                              'approximate'))
        return issues
    image, _, extent = preview
    stiff_by_index = [fact[0] for fact in by_index]
    coupled = any(coupling.enabled and coupling.target == optical.name for coupling in vctx.spec.couplings)
    keys_known = all(by_index[index][2] is not None for index in present)
    if values['moving_boundary'] and not coupled and keys_known and not _has_permittivity_jump(
            image, stiff_by_index, [fact[2] if index in present else ('absent', index)
                                    for index, fact in enumerate(by_index)]):
        issues.append(Issue('info', path, f"moving_boundary is on, but on the material preview no solid cell borders "
                                          f"another permittivity of '{optical.name}' (interior faces only): the "
                                          'moving-boundary columns read 0'))
    if in_plane:
        issue = _thickness_issue(path, values['length'], np.asarray(stiff_by_index, dtype=bool)[image], extent)
        if issue is not None:
            issues.append(issue)
    return issues


def _pairing_issues(vctx, physics_spec, values, optical, settings, path):
    """Optics / solid pairing: em_fdfd_bor (on a triangles mesh) with an axisymmetric triangle solid or a
    solid_mechanics_3d on tetrahedra, the planar types with a planar 2D solid (errors); the em_fdfd_bor-only params; an
    info per em_fdfd_bor pec / pmc z face the solid holds by a roller (a mirror plane: m_eff of the drawn half)."""
    issues = []
    solid_3d = physics_spec.type == _SOLID_3D_TYPE
    raw = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
    axisymmetric = raw.get('geometry') == 'axisymmetric'
    solid_mesh = getattr(vctx.mesh, 'kind', None)
    name = optical.name
    if optical.type != _BOR_TYPE:
        if solid_3d or axisymmetric:
            issues.append(Issue('error', path, f"optical physics '{name}' ({optical.type}) is a planar cut: it pairs with a "
                                               f"planar 2D solid; a{' 3D' if solid_3d else 'n axisymmetric'} solid needs "
                                               'em_fdfd_bor optics'))
        if values.get('prescribed_normal_displacement') is not None:
            issues.append(Issue('error', path, 'prescribed_normal_displacement is read with em_fdfd_bor optics only'))
        return issues
    try:
        optical_mesh = vctx.spec.mesh_of(optical).kind
    except KeyError:   # reported by the mesh checks
        optical_mesh = None
    if optical_mesh not in (None, _TRIANGLES):
        issues.append(Issue('error', path, f"em_fdfd_bor '{name}' runs on a {optical_mesh} mesh: the optomechanical "
                                           'overlap needs its finite-element resonance (a triangles mesh)'))
    if solid_3d:
        if solid_mesh not in (None, _TETRAHEDRA):
            issues.append(Issue('error', path, f'a {_SOLID_3D_TYPE} solid pairs with em_fdfd_bor optics on a tetrahedra '
                                               f'mesh only (this one: {solid_mesh})'))
        if values['normalisation'] == 'point':
            issues.append(Issue('error', path, f"normalisation 'point' is not available for a {_SOLID_3D_TYPE} solid: "
                                               "use 'max'"))
    else:
        if not axisymmetric:
            issues.append(Issue('error', path, f"em_fdfd_bor optics need the solid '{physics_spec.name}' axisymmetric "
                                               '(geometry axisymmetric): the drawing is the (r, z) half plane'))
        if solid_mesh not in (None, _TRIANGLES):
            issues.append(Issue('error', path, f"with em_fdfd_bor optics the solid '{physics_spec.name}' runs on a "
                                               f'triangles mesh (this one: {solid_mesh})'))
    rollers = _roller_faces(vctx, physics_spec)
    for face in ('y_lo', 'y_hi'):
        if settings.get(f'bc_{face}') in _BOR_WALL_TYPES and face in rollers:
            issues.append(Issue('info', path, f"optical face {face} of '{name}' is a {settings[f'bc_{face}']} wall and "
                                              'the solid holds it by a roller: if it is a mirror plane, m_eff is that of '
                                              'the drawn half (the whole body: twice it, g0 / sqrt(2))'))
    return issues


def _validate(vctx, physics_spec, feature_spec, values):
    """Study, optical physics type and settings (em_modes group_index, mode count), the optics / solid pairing, mirror /
    roller pairing, the reference point inside the domain, the material infos (_material_issues) and the
    subpixel-smoothing note."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.study_type != OPTOMECHANICS_STUDY:
        issues.append(Issue('warning', path, f'optomechanical_coupling is filled only by an {OPTOMECHANICS_STUDY} study '
                                             f'(this model runs {vctx.study_type}): its result will be missing'))
    if values['normalisation'] == 'point':
        domain = vctx.spec.domain
        if not (0.0 <= values['reference_x'] <= domain.size_x and 0.0 <= values['reference_y'] <= domain.size_y):
            issues.append(Issue('error', path, f"reference point ({values['reference_x']:g}, {values['reference_y']:g}) "
                                               'm lies outside the domain'))
    name = values['optical_physics']   # required 'ref': the generic checks report an empty or unknown name
    optical = next((physics for physics in vctx.spec.physics if physics.name == name), None)
    if optical is None:
        return issues   # the 'ref' check reports it
    if optical.type not in OPTICAL_TYPES:
        issues.append(Issue('error', path, f"optical physics '{name}' is {optical.type}; supported: "
                                           f"{', '.join(OPTICAL_TYPES)}"))
        return issues
    try:
        optical_type = vctx.registry.physics_type(optical.type)
    except KeyError:
        return issues
    settings = vctx.values(optical_type.settings, optical.settings)
    if settings is None:
        return issues
    issues.extend(_pairing_issues(vctx, physics_spec, values, optical, settings, path))
    if optical.type == _MODES_TYPE:
        if not settings.get('group_index'):
            issues.append(Issue('error', path, f"em_modes '{name}' needs group_index on: the ring conversion d omega = "
                                               '-(C0 / n_group) d beta needs the group index'))
        if values['optical_mode'] > settings.get('n_modes', values['optical_mode']):
            issues.append(Issue('error', path, f"optical_mode {values['optical_mode']} exceeds n_modes "
                                               f"{settings.get('n_modes')} of '{name}'"))
        if settings.get('bend_radius') and settings.get('bend_model') == 'conformal':
            issues.append(Issue('info', path, f"'{name}' uses the conformal bend map: the overlaps use its mapped "
                                              'permittivity, an approximation second order in the mode width / R '
                                              "(bend_model 'exact' has none)"))
    else:
        if values.get('mechanical_quality_factor') is not None:
            issues.append(Issue('info', path, 'mechanical_quality_factor is read only with an em_modes optical physics '
                                              '(waveguide Brillouin gain): brillouin_gain reads NaN here'))
        if values['scattering'] == 'backward':
            issues.append(Issue('error', path, f"scattering backward needs an em_modes optical physics ('{name}' is "
                                               f'{optical.type}): the phonon k_z = 2 beta of a guided mode'))
    if values['scattering'] == 'backward':
        issues.extend(_backward_issues(physics_spec, feature_spec, values, path))
    if physics_spec.type != _SOLID_3D_TYPE:   # the material preview is the 2D drawing
        issues.extend(_material_issues(vctx, path, values, optical))
    if name in subpixel_smoothing_physics(vctx.spec, vctx.registry)[0]:
        issues.append(Issue('info', path, f"subpixel smoothing averages the interface permittivity of '{name}', the g0 "
                                          'overlap keeps the staircase: its moving-boundary jumps and photoelastic '
                                          "permittivity are the unsmoothed cell values (the build's material "
                                          'assignment), the fields, D_n and the mode energy those of the smoothed '
                                          'solve; the face sum is then not the exact first-order change of the smoothed '
                                          'operator (first order in the cell size, as without smoothing)'))
    rollers = _roller_faces(vctx, physics_spec)
    for face in FACES:
        if settings.get(f'bc_{face}') in _MIRROR_TYPES and face not in rollers:
            issues.append(Issue('warning', path, f"optical face {face} of '{name}' is a mirror, but no solid roller "
                                                 'holds that face: the mirrored vibration modes are not the even ones '
                                                 'the optical mirror implies'))
    return issues


OPTOMECHANICAL_COUPLING_FEATURE = FeatureType(
    'optomechanical_coupling', 'Optomechanical coupling', 'monitor', None, params=_PARAMS, result_kind='table',
    description=_DESCRIPTION,
    summary="Computes how strongly each vibration mode shifts the frequency of a light mode (vacuum coupling rate "
            "g0) in an 'Optomechanics' study.",
    apply=_apply, result=_result, validate=_validate,
    grids=('tensor', _TRIANGLES, _TETRAHEDRA))   # finite-element solids: em_fdfd_bor optics (optomechanics_fem)


def register_optomechanics_templates(registry, em_types):
    """Templates silicon_ring_breathing_g0 (with em_modes) and silicon_disk_wgm_g0 (with em_fdfd_hz)."""
    if _MODES_TYPE in em_types:
        registry.register_template(
            'silicon_ring_breathing_g0', SILICON_RING_G0_LABEL, SILICON_RING_G0_DESCRIPTION,
            silicon_ring_breathing_g0_spec,
            summary='How strongly the vibrations of a suspended silicon waveguide (as in a ring resonator) shift the '
                    'frequency of its light: the vacuum optomechanical coupling rate of every vibration mode.')
    if _FDFD_HZ_TYPE in em_types:
        registry.register_template(
            'silicon_disk_wgm_g0', SILICON_DISK_G0_LABEL, SILICON_DISK_G0_DESCRIPTION, silicon_disk_wgm_g0_spec,
            summary='How strongly the vibrations of a small silicon disk shift the frequency of its whispering-gallery '
                    'light resonance: the vacuum optomechanical coupling rate of every vibration mode.')


def optomechanics_cases(em_types):
    """Self-test cases (imported here, called by definition's loader): the CPU overlap, crystal tensor and Brillouin
    cases, plus the em_modes model chains when em_modes is registered and the em_fdfd_bor model runs when it is."""
    from .selftests_brillouin import BRILLOUIN_CASES, BRILLOUIN_CHAIN_CASES
    from .selftests_optomechanics import OPTOMECHANICS_CASES, OPTOMECHANICS_CHAIN_CASES
    from .selftests_optomechanics_bor import OPTOMECHANICS_BOR_CASES
    from .selftests_optomechanics_crystal import CRYSTAL_CASES
    from .selftests_optomechanics_fem import OPTOMECHANICS_FEM_CASES
    return (OPTOMECHANICS_CASES + CRYSTAL_CASES + BRILLOUIN_CASES + OPTOMECHANICS_FEM_CASES
            + (OPTOMECHANICS_CHAIN_CASES + BRILLOUIN_CHAIN_CASES if _MODES_TYPE in em_types else ())
            + (OPTOMECHANICS_BOR_CASES if _BOR_TYPE in em_types else ()))

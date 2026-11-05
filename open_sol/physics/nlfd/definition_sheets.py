"""Model features of the NLFD-chi nonlinear sheets (nonlinear_optics_plan stage 4): surface_chi2 (isotropic-surface
chi_s(2) on the interface of a domain selection) and sheet_nonlinearity (2D-material chi_s(2) / sigma(3) on an
axis-aligned line), their validation, memory estimates and mixing-table tensors (imported by definition.py)."""
import math

from ...core.memory_estimate import BOOL_BYTES, COMPLEX128_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.geometry import expand_entities
from ...model.schema import FeatureType, Issue, Param, feature_path
from ..em.definition_common import mode_line
from ..em.definition_feature_validation import conducting_sheet_line_issues
from .mixing import MediumTensor
from .sheets import (SELVEDGES, SHEET_CHI2_COEFFICIENTS, SURFACE_COEFFICIENTS, SURFACE_INTERFACES,
                     line_mixing_tensor, sheet_chi2_local, sheet_sigma3_local, surface_local_tensor,
                     surface_mixing_tensor)

SURFACE_FEATURE = 'surface_chi2'
LINE_FEATURE = 'sheet_nonlinearity'
SHEET_FEATURES = (SURFACE_FEATURE, LINE_FEATURE)
SHEET_MODELS = ('D3h', 'chi2_general', 'sigma3')
_KERR_SUFFIX = ':kerr'                                   # medium name of the Kerr-type sigma(3) part
_THG_ONLY = (('SPM', 0.0), ('XPM', 0.0), ('FWM', 0.0))   # process scales of the THG / TFG sigma(3) medium
_KERR_ONLY = (('THG', 0.0), ('TFG', 0.0))                # ... of the SPM / XPM / FWM sigma(3) medium
_SIGMA3_PARAMS = ('sigma3_real', 'sigma3_imag', 'thg_sigma3_real', 'thg_sigma3_imag')
_MODEL_COEFFICIENTS = {'D3h': SHEET_CHI2_COEFFICIENTS['D3h'], 'chi2_general': SHEET_CHI2_COEFFICIENTS['general'],
                       'sigma3': _SIGMA3_PARAMS}
_CURVED_KINDS = ('circle', 'ellipse')
_SMOOTHED_SITE_BYTES = 32 * FLOAT64_BYTES   # host crossing search per site (brackets, circle points, test temporaries)
_SITE_TAP_BYTES = 3 * 2 * (8 + 2 * FLOAT64_BYTES)   # per site: index, sample, deposit of two taps per component


# ---- surface_chi2 ----
SURFACE_CHI2_PARAMS = (
    Param('chi_s_nnn', 'float', 0.0, label='chi_s nnn', unit='m^2/V',
          help='Surface susceptibility chi_s_nnn [m^2/V] (chi_perp,perp,perp) of the isotropic (C_inf_v) surface: the '
               'normal polarisation from the normal field, P_s,n = (1/2) eps0 chi_s_nnn E_n^2 for SHG of one channel; '
               'n = the outward normal of the selection. Noble metals: of order 1e-19 to 1e-18 m^2/V (literature values '
               'spread by an order of magnitude and depend on the selvedge convention).',
          tip='Nonlinear surface response along the surface normal, from the normal field, in square metres per '
              'volt.'),
    Param('chi_s_ntt', 'float', 0.0, label='chi_s ntt', unit='m^2/V',
          help='chi_s_ntt = chi_s_nzz [m^2/V] (chi_perp,par,par): normal polarisation from the tangential fields, '
               'P_s,n = (1/2) eps0 chi_s_ntt (E_t^2 + E_z^2) for SHG; t = z x n is the in-plane tangent. A normal '
               'polarisation sheet is a dipole layer: it jumps the tangential E across the interface (it does not '
               'radiate at normal incidence on a flat interface).',
          tip='Nonlinear surface response along the normal, from the fields along the surface, in square metres per '
              'volt.'),
    Param('chi_s_tnt', 'float', 0.0, label='chi_s tnt', unit='m^2/V',
          help='chi_s_tnt = chi_s_ttn = chi_s_znz = chi_s_zzn [m^2/V] (chi_par,perp,par): tangential polarisation '
               'from the normal times a tangential field, P_s,t = eps0 chi_s_tnt E_n E_t and P_s,z = eps0 chi_s_tnt E_n '
               'E_z for SHG of one channel. chi_s_ntt = chi_s_tnt makes the tensor fully permutation symmetric '
               '(power_exchange_imbalance ~0).',
          tip='Nonlinear surface response along the surface, from the normal field times a field along the surface, '
              'in square metres per volt.'),
    Param('selvedge', 'choice', 'inside', label='Normal field side', choices=SELVEDGES,
          choice_labels=(('inside', 'Inside the selection'), ('outside', 'Outside the selection')),
          help='Cell row whose normal E the sheet samples and whose normal-E faces carry the normal polarisation '
               'P_s,n / w (a one-cell dipole layer): inside = the selected cell at each interface face, outside = its '
               'unselected neighbour. E_n jumps by eps_out / eps_in across the interface, so this is the field (and '
               'permittivity) convention of chi_s_nnn; the tangential E and E_z are sampled on the interface itself '
               '(the face / the mean of the two cells). The tangential jump then sits half a cell off the interface: '
               'first order in the mesh.',
          tip='On which side of the surface the normal electric field is taken: inside or outside the selected '
              'region. Surface nonlinear coefficients are quoted for one of the two conventions.'),
    Param('interface', 'choice', 'staircase', label='Interface shape', choices=SURFACE_INTERFACES,
          choice_labels=(('staircase', 'Cell faces (axis-aligned)'), ('smoothed', 'Smoothed normal and length')),
          help='staircase: the sites are the cell faces between selected and unselected cells with normal +-x / +-y '
               'and the face length: exact for axis-aligned sides, but a curve is up to 4 / pi too long (mean 4 / pi) '
               'and its normals point along the axes. smoothed: the same faces, with the normal of the drawn shape '
               'where it crosses the line between the two cells of the face (point tests at build) and the length '
               'face x |n . face normal|, so a curve has its true length and normal (the sites stay on the faces: '
               'first order in the mesh; flat axis-aligned sides stay exact, corners are rounded over about a '
               'cell).',
          tip='How the surface follows curved shapes: along the cell faces, or with the normal direction and length '
              'corrected from the exact shape (better for disks, ellipses and slanted sides).'),
    Param('interface_with', 'names', (), label='Only facing', ref_kind='geometry',
          help='Domain objects: only the interface faces whose unselected cell lies in one of them carry the sheet '
               '(e.g. the metal / air side of a ridge on a substrate); empty (the default) = every unselected '
               'neighbour.',
          tip='Restricts the surface to the parts facing these objects; leave empty for the whole boundary.'),
)
_SURFACE_DESCRIPTION = (
    'Surface chi2 sheet on the interface of the selected domains (nonlinear_optics_plan 3.3): surface polarisation P_s '
    '[C/m] = eps0 2^(1-n) sum chi_s : E E with the isotropic-surface tensor (chi_s_nnn; chi_s_ntt = chi_s_nzz; '
    'chi_s_tnt = chi_s_ttn = chi_s_znz = chi_s_zzn) in the local frame of the outward normal n, the in-plane tangent t = '
    'z x n and z; SHG of one channel P_s,n = (1/2) eps0 (chi_s_nnn E_n^2 + chi_s_ntt (E_t^2 + E_z^2)), P_s,t = eps0 '
    'chi_s_tnt E_n E_t, P_s,z = eps0 chi_s_tnt E_n E_z. Sites are the cell faces between a selected and an unselected '
    'cell: the face samples its own tangential E exactly, E_z as the mean of its two cells and E_n at the selvedge cell '
    '(face mean), and returns P_s by the exact adjoint (the tangential parts as a sheet current on the face, z split '
    'half and half onto the two cells, the normal part as a one-cell dipole layer), so the discrete power exchange '
    'stays exact. Its mixing terms join the chi2 media in the table (SHG / SFG / DFG onto the declared channels; hz '
    'channels carry E_n / E_t, ez channels E_z). Normal-incidence SHG of a flat interface vanishes by symmetry: '
    'curvature, corners or oblique fields are needed. Powers are 2D: per unit length in z.')

# ---- sheet_nonlinearity ----
_D3H = (('sheet_model', ('D3h',)),)
_GENERAL = (('sheet_model', ('chi2_general',)),)
_SIGMA3 = (('sheet_model', ('sigma3',)),)
_TZ_TEXT = ('sheet frame: t = along the line (+x for a horizontal, +y for a vertical line), z out of the plane; the '
            'normal field does not enter')


def _general_param(name):
    return Param(name, 'float', 0.0, label=f'chi_s {name[len("chi_s_"):]}', unit='m^2/V', visible_if=_GENERAL,
                 help=f'Sheet susceptibility {name} [m^2/V] of a general 2D chi_s(2) in the {_TZ_TEXT} (ttz = tzt, zzt = '
                      'ztz); P_s = eps0 2^(1-n) sum chi_s : E E [C/m].',
                 tip='Nonlinear coefficient of the sheet, in square metres per volt.')


SHEET_NONLINEARITY_PARAMS = (
    Param('sheet_model', 'choice', 'D3h', label='Nonlinearity', choices=SHEET_MODELS,
          choice_labels=(('D3h', 'Second order, TMD monolayer (D3h)'), ('chi2_general', 'Second order, general'),
                         ('sigma3', 'Third order sheet conductivity (graphene)')),
          help='D3h: a transition-metal-dichalcogenide monolayer chi_s(2) (MoS2, WS2, ...): chi_aaa = chi_s, chi_abb = '
               'chi_bab = chi_bba = -chi_s with the armchair axis a at armchair_angle in the sheet plane (t, z); '
               'chi2_general: the six chi_s_<ijk> of (t, z); sigma3: the in-plane isotropic third-order sheet '
               'conductivity of graphene, K = 2^(1-n) sum sigma3 : E E E [A/m] (THG K = sigma3 E^3 / 4, SPM K = (3/4) '
               'sigma3 |E|^2 E), entered as P_s = K / (j omega) of the target channel; the THG / TFG value and the '
               'SPM / XPM / FWM value are separate (they differ by orders of magnitude in graphene; no '
               'Cheng-Vermeulen-Sipe model here).',
          tip='Kind of sheet nonlinearity: second order of a monolayer semiconductor such as MoS2, a general second '
              'order sheet, or the third-order conductivity of graphene.'),
    Param('chi_s', 'float', 0.0, label='chi_s', unit='m^2/V', visible_if=_D3H,
          help='D3h sheet susceptibility chi_s = chi_aaa [m^2/V] (P_s = eps0 2^(1-n) sum chi_s : E E [C/m]); MoS2 / WS2 '
               'monolayers ~1e-20 to 1e-19 m^2/V at near-infrared SHG (bulk-equivalent d times the 0.65 nm layer '
               'thickness; literature values span an order of magnitude).',
          tip='Second-order nonlinear coefficient of the monolayer, in square metres per volt.'),
    Param('armchair_angle', 'float', 0.0, label='Armchair angle', unit='deg', visible_if=_D3H,
          help=f'Angle of the D3h armchair axis a from t toward z [deg] in the {_TZ_TEXT}; 0 = armchair along the line '
               '(E_z alone then drives P_t = -(1/2) eps0 chi_s E_z^2), 90 = armchair out of the plane (E_t alone '
               'drives P_z = -(1/2) eps0 chi_s E_t^2). The response has a 120 degree period.',
          tip='Orientation of the crystal (armchair direction) within the sheet, in degrees from the line direction '
              'toward the out-of-plane direction.'),
    *(_general_param(name) for name in SHEET_CHI2_COEFFICIENTS['general']),
    Param('sigma3_real', 'float', 0.0, label='sigma3 (Kerr) real part', unit='S m^2/V^2', visible_if=_SIGMA3,
          help='Re sigma3 [S m^2/V^2] of the SPM / XPM / FWM processes (in-plane isotropic: sigma_tttt = sigma_zzzz, '
               'sigma_ttzz = sigma_tztz = sigma_tzzt = sigma / 3); exp(+j omega t) phasors, Re > 0 absorbs. Graphene '
               'values depend strongly on the Fermi energy and the photon energies (literature ~1e-24 to 1e-20).',
          tip='Real part of the third-order sheet conductivity for the Kerr-type processes, in siemens square metres '
              'per square volt.'),
    Param('sigma3_imag', 'float', 0.0, label='sigma3 (Kerr) imaginary part', unit='S m^2/V^2', visible_if=_SIGMA3,
          help='Im sigma3 [S m^2/V^2] of the SPM / XPM / FWM processes (exp(+j omega t): opposite sign to the '
               'exp(-i omega t) literature).',
          tip='Imaginary part of the third-order sheet conductivity for the Kerr-type processes.'),
    Param('thg_sigma3_real', 'float', 0.0, label='sigma3 (THG) real part', unit='S m^2/V^2', visible_if=_SIGMA3,
          help='Re sigma3 [S m^2/V^2] of the third-harmonic and three-frequency sum processes (THG, TFG), same tensor '
               'form as sigma3_real.',
          tip='Real part of the third-order sheet conductivity for third-harmonic generation.'),
    Param('thg_sigma3_imag', 'float', 0.0, label='sigma3 (THG) imaginary part', unit='S m^2/V^2',
          visible_if=_SIGMA3,
          help='Im sigma3 [S m^2/V^2] of THG / TFG (exp(+j omega t) phasors).',
          tip='Imaginary part of the third-order sheet conductivity for third-harmonic generation.'),
)
_LINE_DESCRIPTION = (
    'Nonlinear 2D-material sheet on an axis-aligned line (nonlinear_optics_plan 3.4): a surface polarisation P_s '
    '[C/m] (chi_s(2): D3h monolayer or general) or a sheet current K [A/m] (sigma3, entered as P_s = K / (j omega)) '
    'from the tangential fields of the sites the conducting_sheet feature uses: E_z at the line-cell centres, the '
    'in-plane tangential E on their hi faces; the result goes back to the same sites (E_z as a density over the '
    'line-cell width, the tangential E as a face current over the face spacing), exact for the discrete equations. Its '
    'terms join the mixing table (sigma3 as up to two media: the THG / TFG value under the feature name, the SPM / XPM '
    "/ FWM value as '<name>:kerr'). Combine it with conducting_sheet on the same line for the linear sheet response. "
    'pump_depleted: sheet SPM / XPM stay in the rhs (no lagged operator term); their absorption (Re sigma3 > 0) is in '
    'nonlinear_absorption and in absorbed_power (deposited in the line cells, so em_heat_joule heats them). The '
    'adjoint (eps_sensitivity, optimization) differentiates the sheet terms. The line covers the cells '
    '[cell(start), cell(end) + 1) along it; a span end on a cpml domain edge continues through the PML.')


def sheet_media(feature_type, name, values):
    """[(medium name, order, local / sheet-frame tensor, response, process scales)] of resolved surface_chi2 /
    sheet_nonlinearity values (sigma3: a THG medium `name` and a Kerr medium `name:kerr`, each when non-zero)."""
    if feature_type == SURFACE_FEATURE:
        local = surface_local_tensor(*(values[coefficient] for coefficient in SURFACE_COEFFICIENTS))
        return [(name, 2, local, 'polarization', ())]
    model = values['sheet_model']
    if model == 'sigma3':
        media = []
        thg = complex(values['thg_sigma3_real'], values['thg_sigma3_imag'])
        kerr = complex(values['sigma3_real'], values['sigma3_imag'])
        if thg != 0:
            media.append((name, 3, sheet_sigma3_local(thg), 'current', _THG_ONLY))
        if kerr != 0:
            media.append((name + _KERR_SUFFIX, 3, sheet_sigma3_local(kerr), 'current', _KERR_ONLY))
        return media
    crystal_class = 'D3h' if model == 'D3h' else 'general'
    coefficients = {coefficient: values[coefficient] for coefficient in SHEET_CHI2_COEFFICIENTS[crystal_class]}
    return [(name, 2, sheet_chi2_local(crystal_class, coefficients, values['armchair_angle']), 'polarization', ())]


def sheet_mixing_tensors(resolve, physics_spec):
    """[MediumTensor] of the enabled sheet features whose values resolve (resolve(params, raw) -> dict | None): the
    entries the solver's SheetMedium give the mixing table (validation, memory estimate)."""
    tensors = []
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in SHEET_FEATURES:
            continue
        surface = feature.type == SURFACE_FEATURE
        values = resolve(SURFACE_CHI2_PARAMS if surface else SHEET_NONLINEARITY_PARAMS,
                         feature.params if isinstance(feature.params, dict) else {})
        if values is None:
            continue
        representative = surface_mixing_tensor if surface else line_mixing_tensor
        tensors.extend(MediumTensor.from_tensor(medium, order, representative(local), scales)
                       for medium, order, local, _, scales in sheet_media(feature.type, feature.name, values))
    return tensors


# ---- apply ----
def _apply_surface(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    outside = ctx.domain_mask(list(values['interface_with'])) if values['interface_with'] else None
    boundary_test = None
    if values['interface'] == 'smoothed':
        from ...model.boundary_locator import geometry_locator
        boundary_test = geometry_locator(ctx).inside(list(selection.entities))
    (medium, _, local, _, _), = sheet_media(SURFACE_FEATURE, feature_spec.name, values)
    solver.add_surface_sheet(medium, local, mask, outside_mask=outside, selvedge=values['selvedge'],
                             boundary_test=boundary_test)
    return None


def _apply_line(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    for medium, order, local, response, scales in sheet_media(LINE_FEATURE, feature_spec.name, values):
        solver.add_line_sheet(medium, axis, position, span, order, local, response=response, process_scales=scales)
    return None


# ---- validation ----
def _slanted_polygon(values):
    """True when a polygon's points have an edge that is neither horizontal nor vertical."""
    try:
        points = [(float(x), float(y)) for x, y in values.get('points') or ()]
    except (TypeError, ValueError):
        return False
    return any(x0 != x1 and y0 != y1 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))


def _curved_objects(vctx, selection):
    """Selected objects with curved or slanted edges (circles, ellipses, slanted polygons, composites of them)."""
    try:
        entities = expand_entities(vctx.spec, selection)
    except (ValueError, KeyError, TypeError):
        return []

    def curved(name, depth=0):
        shape = vctx.shapes.get(name)
        if shape is None or depth > 16:
            return False
        if shape.kind in _CURVED_KINDS or (shape.kind == 'rounded_rectangle' and shape.values.get('r')):
            return True
        values = shape.values if isinstance(shape.values, dict) else {}
        if shape.kind == 'polygon':
            return _slanted_polygon(values)
        children = [*(values.get('inputs') or ()), *(values.get('subtract') or ())]
        if values.get('base'):
            children.append(values['base'])
        return any(curved(child, depth + 1) for child in children if isinstance(child, str))
    return sorted(name for name in entities if curved(name))


def _validate_surface(vctx, physics_spec, feature_spec, values):
    """Warnings for an all-zero tensor and for staircase sites on curved / slanted objects; info for smoothed."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if not any(values[coefficient] for coefficient in SURFACE_COEFFICIENTS):
        issues.append(Issue('warning', path, 'chi_s_nnn = chi_s_ntt = chi_s_tnt = 0: the surface couples nothing'))
    if values['interface'] == 'staircase':
        curved = _curved_objects(vctx, feature_spec.selection)
        if curved:
            issues.append(Issue('warning', path, "interface 'staircase' on curved or slanted objects "
                                                 f"({', '.join(curved)}): the sites are cell faces with normals +-x / "
                                                 '+-y, so the surface is up to 4 / pi too long and its normal points '
                                                 "along the axes between them; interface 'smoothed' takes the normal "
                                                 'and length from the drawn shape'))
    else:
        issues.append(Issue('info', path, "interface 'smoothed': normals of the drawn shape at the face crossings, "
                                          'site length = face x |n . face normal| (first order; flat axis-aligned '
                                          'sides stay exact, corners are rounded over about a cell)'))
    return issues


def _validate_line(vctx, physics_spec, feature_spec, values):
    """The conducting_sheet line checks per channel polarisation present (deduplicated), a warning for all-zero
    coefficients, an info on the sigma3 convention."""
    from .definition import _model_channels   # definition imports this module
    channels, _ = _model_channels(physics_spec, vctx.parameters)
    found = {}
    for polarization in dict.fromkeys(channel.polarization for channel in channels or ()):
        for issue in conducting_sheet_line_issues(vctx, physics_spec, feature_spec, polarization):
            found.setdefault((issue.level, issue.message), issue)
    issues = list(found.values())
    path = feature_path(physics_spec, feature_spec)
    model = values['sheet_model']
    if not any(values[name] for name in _MODEL_COEFFICIENTS[model]):
        issues.append(Issue('warning', path, f'every {model} coefficient is 0: the sheet couples nothing'))
    elif model == 'sigma3':
        issues.append(Issue('info', path, 'sigma3 sheet: K = 2^(1-n) sum sigma3 E E E [A/m] enters as P_s = K / (j '
                                          'omega) of the target channel; the THG / TFG value is the medium '
                                          f"'{feature_spec.name}', the SPM / XPM / FWM value '{feature_spec.name}"
                                          f"{_KERR_SUFFIX}' (pump_depleted only: rhs form, absorption in "
                                          'nonlinear_absorption and absorbed_power of the line cells)'))
    return issues


# ---- memory estimates ----
def _site_bytes(order):
    return 3 ** (order + 1) * COMPLEX128_BYTES + _SITE_TAP_BYTES + FLOAT64_BYTES


def _sheet_run_parts(n_sites, n_cells):
    """(run_items, transient_items) of one target: the kept full-grid deposit (_sheet_density, hz 2 components) and
    one evaluation's site fields / P_s and scatter temporaries (approximate)."""
    run_items = [('kept sheet deposit (one hz target, 2 x complex128, approximate)', 2 * COMPLEX128_BYTES * n_cells)]
    transient_items = [('sheet evaluation', 'sheet site fields and scatter temporaries (one target, approximate)',
                        8 * 3 * COMPLEX128_BYTES * n_sites + (COMPLEX128_BYTES + FLOAT64_BYTES) * n_cells)]
    return run_items, transient_items


def _surface_memory_estimate(values, n_selected_cells, mesh):
    """Sites ~ twice the perimeter of the selection box in cells; build: the masks (+ the smoothed crossing search)."""
    box_cells, n_cells = int(n_selected_cells), int(mesh.n_cells)
    n_sites = int(min(2 * box_cells, 8 * math.sqrt(box_cells))) + 4
    build_extra = 2 * BOOL_BYTES * n_cells
    if values.get('interface') == 'smoothed':
        build_extra += _SMOOTHED_SITE_BYTES * n_sites
    run_items, transient_items = _sheet_run_parts(n_sites, n_cells)
    return MemoryEstimate.from_items((('surface chi2 sites (approximate)', n_sites * _site_bytes(2)),),
                                     build_extra=build_extra, run_items=run_items, transient_items=transient_items)


def _line_memory_estimate(values, n_selected_cells, mesh):
    """Sites = the line cells (sigma3: up to two media)."""
    n_sites, n_cells = max(int(n_selected_cells), 1), int(mesh.n_cells)
    sigma3 = values.get('sheet_model') == 'sigma3'
    order, copies = (3, 2) if sigma3 else (2, 1)
    run_items, transient_items = _sheet_run_parts(n_sites, n_cells)
    return MemoryEstimate.from_items((('nonlinear sheet sites', copies * n_sites * _site_bytes(order)),),
                                     run_items=run_items, transient_items=transient_items)


SURFACE_CHI2 = FeatureType(
    SURFACE_FEATURE, 'Surface second-order nonlinearity', 'domain_condition', 'domain', params=SURFACE_CHI2_PARAMS,
    description=_SURFACE_DESCRIPTION,
    summary='Gives the boundary of the selected region a second-order surface nonlinearity, as found at metal and '
            'crystal surfaces: second harmonic from the surface.',
    apply=_apply_surface, validate=_validate_surface, memory_estimate=_surface_memory_estimate)
SHEET_NONLINEARITY = FeatureType(
    LINE_FEATURE, 'Nonlinear sheet', 'boundary_condition', 'line', params=SHEET_NONLINEARITY_PARAMS,
    object_kinds=('line',), max_entities=1, description=_LINE_DESCRIPTION,
    summary='A nonlinear sheet of zero thickness along a straight line, such as a MoS2 monolayer (second harmonic) or '
            'graphene (third harmonic and Kerr effect).',
    apply=_apply_line, validate=_validate_line, memory_estimate=_line_memory_estimate)

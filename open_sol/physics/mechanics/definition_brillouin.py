"""Model-layer definition of the solid_mechanics feature brillouin_scattering (spontaneous Brillouin spectrum of the
frequency_response study): params, validation, apply, result and the memory / run-time items; registered by
definition.py."""
# Conventions and formulas: brillouin.py header (cited by the help texts); runtime: brillouin_drive.BrillouinDrive
# (its constructor is the contract of _apply). The optical physics run on their own tensor mesh, the solid on
# triangles, or (planar, loss factor only) on the rectangular cells of the optics' own tensor mesh: optical_physics /
# collection_physics are kind 'str' (a 'ref' physics Param must share the feature's mesh, validate._check_physics_refs),
# checked here. Estimator items mirror what BrillouinDrive.bind / record / result and
# the HarmonicSystem column path allocate (counts from brillouin.py, fem.elastic_drives and fem.quadrature); a new
# allocation there updates brillouin_memory_items / brillouin_time_items.
import math

import numpy as np

from ...core import time_estimate as te
from ...core.constants import FACES
from ...core.grid import uniform_cell_count
from ...core.mesh import has_uniform_spacing
from ...core.mirror import mirror_faces
from ...fem.assembly import CHUNK_ELEMENTS
from ...fem.elastic_drives import DRIVE_DEGREE
from ...fem.quadrature import gauss_line, triangle_rule
from ...model.definition_helpers import (ORDER_SPLIT_READERS, azimuthal_order_of, element_order,
                                         enabled_feature_entities, frequency_sweep_counts, is_axisymmetric,
                                         mesh_kind_of, study_frequencies)
from ...model.geometry import domain_object_boxes, domain_size, expand_entities, rasterize, resolve_geometry
from ...model.schema import (FREQUENCY_RESPONSE_STUDY, FeatureType, Issue, Param, coupling_writes, feature_path,
                             resolve_values)
from .brillouin import (_CHUNK_VALUES, ANTI_STOKES_SERIES, AZIMUTHAL_PARTS, DB_SERIES, DETECTIONS, FIT_EXTRAS,
                        PSD_SERIES, REFERENCE_SERIES, REFERENCES, SCAN_COLUMNS, STOKES_SERIES, X_LABEL, Y_LABEL,
                        scan_positions)
from .brillouin_drive import LINE_BIN, BrillouinDrive
from .definition_common import ROLLER_FEATURES, merged_materials, plain_value
from .definition_optomechanics import _crystal_regions, crystal_photoelastic_tensor
from .definition_viscoelastic import _ELEMENT_MODULI_BYTES
from .viscoelastic import MIXED
from .materials import (BULK_BRANCH_ARRAYS, LOSS_ARRAYS, SHEAR_BRANCH_ARRAYS, STIFFNESS_TENSOR_ARRAYS,
                        VISCOSITY_ARRAYS)

BRILLOUIN_FEATURE = 'brillouin_scattering'
FOCUSED_BEAM_FEATURE = 'focused_beam'            # the optics' illumination feature (physics/em focused_beam)
PLANAR_OPTICAL_TYPES = ('em_fdfd_ez', 'em_fdfd_hz')
BOR_OPTICAL_TYPE = 'em_fdfd_bor'
OPTICAL_TYPES = PLANAR_OPTICAL_TYPES + (BOR_OPTICAL_TYPE,)
# drive_fields components per optical type (E_z; E_x, E_y; E_r, A_phi, E_z)
_FIELD_COMPONENTS = {'em_fdfd_ez': 1, 'em_fdfd_hz': 2, BOR_OPTICAL_TYPE: 3}
_DRIVEN_ORDERS = (0, 2)          # two order-1 optical fields drive the solid orders 0 and 2 only
_BEAM_DIRECTIONS = {'positive': 1, 'negative': -1, '+z': 1, '-z': -1}   # focused_beam direction: sign of travel
_OPTICAL_CELLS_PER_WAVELENGTH = 15.0   # warning below: optical cells per lambda0 / n in the drive region
_SOLID_SIZE_DIVISOR = {2: 12.0, 1: 24.0}   # warning: solid element size h > lambda0 / (divisor n), by element order
#                                            (rectangular Q1 cells: the linear divisor)
_LINEWIDTHS = 3.0                # info: no study frequency within this many estimated linewidths of f_B
_TOLERANCE = 1e-9                # relative (domain size) slack of the geometric checks
_WAVELENGTH_RTOL = 1e-9          # collection physics: same vacuum wavelength
_PREVIEW_CELLS = 200             # vctx.material_map raster of the drive-region material checks

# ---- params ----
_OPTICAL_HELP = ('Name of the em_fdfd_ez / em_fdfd_hz (planar solid) or em_fdfd_bor (axisymmetric solid) physics on a '
                 'tensor (rectangular-cell) mesh whose focused_beam illuminates the sample; it is solved first as a '
                 'background physics (stationary) of the frequency_response study. It must hold exactly one enabled '
                 'source feature, a focused_beam (its solution is the sum of every enabled source).')
_PARAMS = (
    Param('optical_physics', 'str', '', label='Optical physics', help=_OPTICAL_HELP,
          tip='Name of the light physics whose focused beam illuminates the sample. It is solved before the '
              'vibrations.\n\nIt must have one focused beam and no other source.'),
    Param('incident_source', 'str', '', label='Illumination beam',
          help='focused_beam feature of optical_physics that illuminates (its incident_power normalises the drive); '
               "'' = its only enabled focused_beam.",
          tip='Name of the focused beam feature that illuminates the sample. Leave empty to take the only one.'),
    Param('collection_physics', 'str', '', label='Collection physics',
          help="'' = optical_physics itself: confocal epi detection with the collection field E_c = E_i (one optical "
               'solve per scan position). Another physics of the same type, wavelength, boundaries and optical grid '
               '(window cells) holds the reciprocal collection beam (launched from the detector toward the sample, e.g. transmission '
               'detection); it is solved once and does not move with the scan.',
          tip='Light physics that holds the detection beam, sent from the detector toward the sample. Leave empty '
              'for detection back through the illumination optics.'),
    Param('collection_source', 'str', '', label='Collection beam',
          help="focused_beam feature of collection_physics ('' = its only enabled one); with collection_physics '' "
               'it must be empty or the incident_source.',
          tip='Name of the focused beam feature of the collection physics. Leave empty to take the only one.'),
    Param('detection', 'choice', 'co_polarized', label='Detected polarization', choices=DETECTIONS,
          choice_labels=(('co_polarized', 'Same as the illumination'), ('cross_polarized', 'Crossed')),
          help='co_polarized: the collected field has the polarization of the incident one. cross_polarized '
               '(em_fdfd_bor only): the collection field is the incident m = 1 field rotated by 90 degrees about the '
               'axis (no second optical solve); it drives the odd m = 2 part only (azimuthal_order 2), the m = 0 '
               'torsion part is not modelled (its share is the extra torsion_drive_fraction, a run warning above '
               '1e-3).',
          tip='Polarization the detector selects.\n\n'
              '- Same as the illumination: the usual confocal measurement.\n'
              '- Crossed: only with an axisymmetric (body of revolution) light physics; it needs a solid with '
              'azimuthal order 2.'),
    Param('temperature', 'float', 293.15, label='Temperature', unit='K', min=0.0, min_exclusive=True,
          help='Temperature T of the thermal phonons: the classical spectrum scales with T; the quantum sideband '
               'weights (psd_stokes, psd_anti_stokes) use it too.',
          tip='Temperature of the sample, in kelvin: room temperature is about 293 K.'),
    Param('photoelastic', 'bool', True, label='Photoelastic drive',
          help='Include the photoelastic (strain-optic) part of the drive, from photoelastic_p11 / _p12 / _p44 of the '
               'solid materials (planar: crystal entries too).',
          tip='Include the change of refractive index with strain. This is the main part in liquids and bulk '
              'solids.'),
    Param('moving_boundary', 'bool', True, label='Moving-boundary drive',
          help='Include the moving-boundary part of the drive: solid element edges inside the drive selection where '
               'the optical permittivity jumps (non-conjugated Johnson perturbation). At least one of photoelastic / '
               'moving_boundary must be on; with photoelastic off an optical interface must touch the drive '
               'selection.',
          tip='Include the light scattered by moving surfaces between materials of different refractive index.'),
    Param('axial_length', 'float', 1e-5, label='Beam length out of the plane', unit='m', min=0.0, min_exclusive=True,
          help='Planar solids only (unused when axisymmetric): the 2D fields are a sheet beam uniform over this '
               'length L along z, powers in W/m; the reported 3D spectrum is the 2D one / L, valid for q_z = 0 '
               'phonons, i.e. L much longer than the acoustic attenuation length (about 1e-6 m in water at 7.4 GHz) '
               'and the beam.',
          tip='Length of the light sheet along the direction out of the drawing, for a planar model. The spectrum '
              'is divided by it.\n\nUnused for an axisymmetric solid.'),
    Param('scan_offsets_x', 'float_list', None, label='Scan offsets x', unit='m', optional=True,
          help='Focus offsets along x [m] of a scan (planar optics only; em_fdfd_bor refuses x offsets): scan '
               'positions are every (x, y) pair of the two lists (x outer, y inner, an empty list = [0]); position 1 '
               '= (first x, first y) is the shown solution. The illumination focus moves analytically (one more '
               'optical solve on the same factor and one more solid solve per frequency each); a collection_physics '
               'keeps its focus. An offset must keep every model mirror plane (offsets along its normal: 0).',
          tip='Shifts of the focus along x, in metres, one scan position each (combined with the y shifts). Leave '
              'empty for one position.'),
    Param('scan_offsets_y', 'float_list', None, label='Scan offsets y', unit='m', optional=True,
          help='Focus offsets along y [m] of a scan (em_fdfd_bor: along the axis z); positions as scan_offsets_x.',
          tip='Shifts of the focus along y (along the axis for an axisymmetric model), in metres. Leave empty for '
              'one position.'),
    Param('reference', 'choice', 'local', label='Reference spectrum', choices=REFERENCES,
          choice_labels=(('local', 'Local plane-wave estimate'), ('homogeneous', 'Unbounded uniform medium'),
                         ('none', 'None')),
          help="Reference series psd_reference: 'local' (any medium) a damped-oscillator line per drive point at its "
               'local wavevector q (phase gradient of E_c . E_i) with the longitudinal modulus M* = K* + 4/3 G* of '
               "its material; 'homogeneous' (planar, no model mirror planes, one uniform isotropic material over the "
               'drive selection on uniform optical cells) the exact plane-wave sum of an unbounded medium; none. Both '
               'need isotropic moduli with a finite bulk modulus in the drive selection (no c_ij route, no '
               "incompressible material of formulation mixed): else reference 'none'.",
          tip='Approximate spectrum shown beside the computed one for comparison.\n\n'
              '- Local plane-wave estimate: works in any sample.\n'
              '- Unbounded uniform medium: exact for one uniform material, planar models only.\n'
              '- None.'),
    Param('fit_peak', 'bool', True, label='Fit the peak',
          help='Fit a damped harmonic oscillator A / ((f0^2 - f^2)^2 + gamma^2 f^2) to psd (and to psd_reference): '
               'extras shift = f0, fwhm, gamma, peak, r_squared, reference_shift; a failed fit sets fit_warning and '
               'NaN values (never an error).',
          tip='Fit a resonance curve to the spectrum to report the Brillouin shift and the linewidth.'),
)

# definition_helpers.azimuthal_order_issues lets the solids of one em_fdfd_bor beam differ in azimuthal_order through
# this feature and its optical_physics param (ORDER_SPLIT_READERS): a rename must reach it
if ORDER_SPLIT_READERS.get(BRILLOUIN_FEATURE) != _PARAMS[0].name:
    raise RuntimeError(f'definition_helpers.ORDER_SPLIT_READERS must map {BRILLOUIN_FEATURE!r} to its param '
                       f'{_PARAMS[0].name!r}, got {ORDER_SPLIT_READERS!r}')

_DESCRIPTION = (
    'Spontaneous Brillouin scattering spectrum by the fluctuation-dissipation theorem (frequency_response study only; '
    "triangle meshes, or the rectangular cells of the optics' own tensor mesh; convention exp(+j omega t), "
    'conventions and formulas in physics/mechanics/brillouin.py). After the background stage the incident and '
    'collection fields of the optics (optical_physics, its focused_beam; collection_physics) are sampled at the '
    "quadrature points of the selected solid elements (rectangular cells: each cell's own optical values, no "
    'interpolation) and drive the solid with the generalised force w = kappa (int B^T sigma~ dV + moving-boundary '
    'edge tractions; rectangular cells: B at the cell centre and the Yee faces), sigma~ = p^T G the '
    'optical stress of the NON-conjugated field products G_I = -eps_ii eps_jj (E_c,i E_i,j + [i != j] E_c,j E_i,i), '
    'kappa = omega eps0 / (4 sqrt(P_i P_c)) (beam powers incident_power). Per frequency f the solid solves K(Omega) u '
    '= w (Omega = 2 pi f, the Brillouin shift) and psd(f) = (2 k_B T / Omega) (-Im(w^H u)) x mirror factor / L '
    '[1/Hz]: the classical power fraction per Hz scattered into the collected channel in ONE sideband (Stokes or '
    'anti-Stokes; the one-sided spectrum of both sidebands is 2 psd); sum rule 2 int_0^inf psd df = k_B T w^H K0^-1 w. '
    'L = axial_length planar (fields per m of depth, 2D powers in W/m), 1 axisymmetric; the mirror factor is 2 per '
    'model mirror plane (the solid must hold each by Symmetry or Roller). Rectangular cells (tensor mesh): the '
    "optical and collection physics on the solid's own mesh, planar plane_strain, no piezoelectric, "
    'mechanical_loss_factor as the only loss (no viscoelastic moduli), Q1 cells <= lambda0 / (24 n) in the drive '
    'region (bilinear cells with their consistent mass raise the shift by about (q h)^2 / 24: 1.1 % at that size). '
    'The drive must be the only load of the '
    'solid (no boundary / segment loads, body forces, coupled forces or nonzero prescribed displacements); the solid '
    'needs loss (mechanical_loss_factor, viscosity, viscoelastic branches): a lossless solid has psd 0 between delta '
    'lines. Axisymmetric (em_fdfd_bor, m = 1 fields): co_polarized drives the solid orders 0 and 2 (one solid '
    'physics each, azimuthal_order 0 / 2; their psd add), cross_polarized the odd order 2 only. Planar: '
    'plane_strain only; em_fdfd_ez drives pure dilation (E_z only), em_fdfd_hz also shear. Mesh: optics >= 15 cells '
    'per lambda0 / n in the drive region, solid element size h <= lambda0 / (12 n) quadratic (/ 24 linear) there '
    '(the drive oscillates on lambda0 / (2 n)). The selection is the drive / sensing region (domain objects or all; '
    'it must stay on the total-field side of the focused_beam line and at least half an optical cell from the window '
    "edges other than mirror planes and the axis). Result 'xy' versus frequency [Hz] (x_label "
    f"'{X_LABEL}', y_label '{Y_LABEL}'): series {PSD_SERIES}, {STOKES_SERIES} and {ANTI_STOKES_SERIES} (x (n + 1) "
    'and x n, n the Bose occupation of hbar Omega), '
    f'{REFERENCE_SERIES} (reference not none) in 1/Hz, {DB_SERIES} = 10 log10 psd in dB/Hz; with several scan '
    'positions every series and per-position extra ends in _<k> (k = 1-based position). Extras: '
    f"{', '.join(FIT_EXTRAS)}, fit_warning and reference_shift (fit_peak), scattered_fraction (2 x the band "
    'integral of psd), q_mean [1/m], incident_power and collection_power [W, planar W/m], wavelength [m], temperature '
    '[K], mirror_factor, azimuthal_order, axial_length, torsion_drive_fraction, drive_norms {photoelastic, '
    "moving_boundary}, scan {'columns': [" + ', '.join(SCAN_COLUMNS) + "], 'units', 'rows'}. Several scan positions "
    'need sweep_method direct; one position also runs adaptive / modal sweeps (load = the drive).')
_SUMMARY = ('Spectrum of the light scattered by thermal vibrations (spontaneous Brillouin scattering) of the selected '
            'domains, for a focused light beam of a light physics, versus the frequency shift. Needs a frequency '
            'response study; the solid must have losses and no other loads.')


# ---- feature callbacks ----
def _apply(ctx, solver, feature_spec, values, selection):
    """BrillouinDrive of the feature (element mask of the selection, None for 'all'; scan positions; mirror factor
    2^(model mirror planes); planar crystal regions with photoelastic on) added to the solid (the frequency_response
    study binds it after the background stage). Rectangular cells: the masks per window cell, flat C order."""
    from ...core.backend import to_host   # lazy: definitions never load the backend at import
    window = solver.system.window if getattr(ctx.domain, 'grid_kind', 'tensor') == 'tensor' else None

    def members(mask):   # bool per solid mesh element, or per window cell (flat) of a rectangular-cell solid
        mask = np.array(to_host(mask), dtype=bool)
        return mask if window is None else np.ascontiguousarray(mask[window]).reshape(-1)
    mask = None if selection.is_all else members(ctx.domain_mask(selection.entities))
    crystals = (tuple((members(region), tensor) for region, tensor in _crystal_regions(ctx))
                if values['photoelastic'] and not solver.axisymmetric else ())
    drive = BrillouinDrive(feature_spec.name, values['optical_physics'], values['collection_physics'],
                           values['incident_source'], values['collection_source'], values['detection'],
                           values['temperature'], values['photoelastic'], values['moving_boundary'],
                           values['axial_length'], _positions(values), values['reference'], values['fit_peak'],
                           mask, float(2 ** len(mirror_faces(ctx.spec))), crystals)
    solver.add_physics_drive(drive)
    return drive


def _result(session, solver, drive, feature_spec, options):
    return drive.result()


def _positions(values) -> np.ndarray:
    """(n, 2) scan offsets of resolved values (brillouin.scan_positions)."""
    return scan_positions(values.get('scan_offsets_x') or (), values.get('scan_offsets_y') or ())


BRILLOUIN_SCATTERING_FEATURE = FeatureType(
    BRILLOUIN_FEATURE, 'Brillouin scattering spectrum', 'monitor', 'domain', params=_PARAMS, result_kind='xy',
    description=_DESCRIPTION, summary=_SUMMARY, apply=_apply, result=_result, grids=('tensor', 'triangles'))


# ---- spec helpers (validation and estimators) ----
def brillouin_features(physics_spec) -> list:
    """Enabled brillouin_scattering FeatureSpecs of a physics."""
    return [feature for feature in physics_spec.features if feature.enabled and feature.type == BRILLOUIN_FEATURE]


def _physics_named(spec, name):
    return next((physics for physics in spec.physics if physics.name == name), None)


def _raw(feature) -> dict:
    return feature.params if isinstance(feature.params, dict) else {}


def _type_settings(vctx, physics):
    """(PhysicsType, resolved settings) of a physics; (None, None) when unknown or unresolved (reported there)."""
    try:
        physics_type = vctx.registry.physics_type(physics.type)
    except KeyError:
        return None, None
    return physics_type, vctx.values(physics_type.settings, physics.settings)


def _sources(vctx, physics, physics_type):
    """(enabled focused_beam features, other enabled source features) of an optical physics."""
    beams, others = [], []
    for feature in physics.features:
        if not feature.enabled:
            continue
        if feature.type == FOCUSED_BEAM_FEATURE:
            beams.append(feature)
            continue
        try:
            category = physics_type.feature_type(feature.type).category
        except KeyError:   # reported at the feature
            continue
        if category == 'source':
            others.append(feature)
    return beams, others


def _domain_box(vctx):
    size = vctx.domain_size
    return None if size is None else (0.0, 0.0, float(size[0]), float(size[1]))


def _drive_box(vctx, feature):
    """(x0, y0, x1, y1) [m] of the drive selection: the union bounding box of its domain objects, the domain for 'all';
    None when nothing resolves (reported by the selection checks)."""
    try:
        expanded = expand_entities(vctx.spec, feature.selection)
    except (ValueError, KeyError, TypeError):
        return None
    if 'all' in expanded:
        return _domain_box(vctx)
    boxes = [box for _, box in domain_object_boxes(vctx.spec, vctx.shapes, [feature.selection])]
    if not boxes:
        return None
    return (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes),
            max(box[3] for box in boxes))


def _drive_preview_indices(vctx, feature, margin=0):
    """Sorted material preview indices (1 + spec material index, 0 = no material) under the drive selection on the
    material preview, grown by `margin` preview pixels (its neighbours); None when the preview or the selection does
    not resolve."""
    preview = vctx.material_map(_PREVIEW_CELLS)
    if preview is None:
        return None
    image, _, extent = preview
    try:
        expanded = expand_entities(vctx.spec, feature.selection)
    except (ValueError, KeyError, TypeError):
        return None
    n_x, n_y = image.shape
    x_centres = extent[0] + (np.arange(n_x) + 0.5) * (extent[1] - extent[0]) / n_x
    y_centres = extent[2] + (np.arange(n_y) + 0.5) * (extent[3] - extent[2]) / n_y
    grid_x, grid_y = np.meshgrid(x_centres, y_centres, indexing='ij')
    if 'all' in expanded:
        mask = np.ones(image.shape, dtype=bool)
    else:
        mask = np.zeros(image.shape, dtype=bool)
        for name in expanded:
            if name in vctx.shapes:
                try:
                    mask |= np.asarray(rasterize(name, vctx.shapes, grid_x, grid_y, np), dtype=bool)
                except (ValueError, KeyError):   # not a domain object: the selection checks report it
                    continue
    if not mask.any():   # a region below one preview pixel: the pixel of the box centre
        box = _drive_box(vctx, feature)
        if box is None:
            return None
        i = int(np.clip(np.searchsorted(x_centres, 0.5 * (box[0] + box[2])), 0, n_x - 1))
        j = int(np.clip(np.searchsorted(y_centres, 0.5 * (box[1] + box[3])), 0, n_y - 1))
        mask[i, j] = True
    for _ in range(int(margin)):   # one pixel per pass along x and y (both directions)
        grown = mask.copy()
        grown[1:, :] |= mask[:-1, :]
        grown[:-1, :] |= mask[1:, :]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return sorted(int(index) for index in np.unique(image[mask]))


def _material_data(vctx, indices):
    """[(name, merged properties or None, crystal photoelastic tensor or None)] of preview indices (0 skipped)."""
    merged = merged_materials(vctx)
    data = []
    for index in indices:
        if index <= 0 or index > len(merged):
            continue
        material, properties = merged[index - 1]
        try:
            tensor = crystal_photoelastic_tensor(material)
        except (ValueError, TypeError, KeyError):   # reported at the material
            tensor = None
        data.append((material.name, properties, tensor))
    return data


def _plain(properties, name, parameters):
    value = plain_value(properties.get(name), parameters) if properties is not None else None
    return None if value is None or not math.isfinite(value) else value


def _refractive_index(properties, parameters):
    """sqrt of the largest plain real permittivity (eps_r and the diagonal components); None when unknown."""
    values = [_plain(properties, name, parameters) for name in ('eps_r', 'eps_xx', 'eps_yy', 'eps_zz')]
    values = [value for value in values if value is not None and value > 0.0]
    return math.sqrt(max(values)) if values else None


def _longitudinal_line(properties, parameters):
    """(sound speed v_L [m/s], FWHM per f_B^0 law: (viscous term [1/m^2 x m^2/s], loss factor)) of plain values, or
    None: M = bulk_modulus + 4/3 shear_modulus, else the E / nu modulus; viscous width q^2 (4/3 viscosity +
    bulk_viscosity) / (2 pi rho) (relaxation branches not counted)."""
    rho = _plain(properties, 'rho', parameters)
    if rho is None or rho <= 0.0:
        return None
    bulk, shear = _plain(properties, 'bulk_modulus', parameters), _plain(properties, 'shear_modulus', parameters)
    youngs, poisson = _plain(properties, 'youngs_modulus', parameters), _plain(properties, 'poisson_ratio', parameters)
    if bulk is not None:
        modulus = bulk + 4.0 / 3.0 * (shear or 0.0)
    elif youngs is not None and poisson is not None and -1.0 < poisson < 0.5:
        modulus = youngs * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    else:
        return None
    if modulus <= 0.0:
        return None
    viscous = 4.0 / 3.0 * (_plain(properties, 'viscosity', parameters) or 0.0) + (
        _plain(properties, 'bulk_viscosity', parameters) or 0.0)
    return math.sqrt(modulus / rho), viscous / (2.0 * math.pi * rho), _plain(
        properties, 'mechanical_loss_factor', parameters) or 0.0


# loss of the solid's harmonic operator: the loss factor, the dashpot viscosities, the relaxation branch moduli
_LOSS_PROPERTIES = (LOSS_ARRAYS[0],) + VISCOSITY_ARRAYS + tuple(modulus for modulus, _ in SHEAR_BRANCH_ARRAYS
                                                                + BULK_BRANCH_ARRAYS)
_PHOTOELASTIC_PROPERTIES = ('photoelastic_p11', 'photoelastic_p12', 'photoelastic_p44')


def _may_be_nonzero(properties, names, parameters) -> bool:
    """True when a property of names is set and not a plain 0 (expressions count as nonzero)."""
    for name in names:
        value = properties.get(name)
        if value is not None and plain_value(value, parameters) != 0.0:
            return True
    return False


def _window_centres(mesh, axis):
    """(first, last) window cell centre [m] of a tensor MeshEstimate along 'x' / 'y' (the drive sample hull)."""
    edges = mesh.x_edges if axis == 'x' else mesh.y_edges
    n = mesh.nx if axis == 'x' else mesh.ny
    first = int(mesh.npml_sides[f'{axis}_lo'])
    last = n - int(mesh.npml_sides[f'{axis}_hi']) - 1
    return 0.5 * (edges[first] + edges[first + 1]), 0.5 * (edges[last] + edges[last + 1])


def _box_widths(edges, low, high) -> np.ndarray:
    """Widths of the cells of edges (n + 1,) overlapping [low, high]."""
    edges = np.asarray(edges, dtype=np.float64)
    first = int(np.clip(np.searchsorted(edges, low, side='right') - 1, 0, len(edges) - 2))
    last = int(np.clip(np.searchsorted(edges, high, side='left'), first + 1, len(edges) - 1))
    return np.diff(edges[first:last + 1])


# ---- validation (CPU; vctx in the study the solid takes: definition_common.background_context) ----
def brillouin_issues(vctx, physics_spec, settings_values) -> list:
    """Every enabled brillouin_scattering feature of a solid (triangles or tensor): one per solid (errors for more),
    then _feature_issues each (values that do not resolve are reported by the param checks)."""
    features = brillouin_features(physics_spec)
    if not features or not settings_values:
        return []
    issues = [Issue('error', feature_path(physics_spec, feature),
                    f"'{physics_spec.name}' already has the brillouin_scattering feature '{features[0].name}': one "
                    'drive set per solid (add the scan offsets to it, or use another solid_mechanics physics)')
              for feature in features[1:]]
    for feature in features:
        values = vctx.values(_PARAMS, feature.params)
        if values is not None:
            issues.extend(_feature_issues(vctx, physics_spec, settings_values, feature, values))
    return issues


def _feature_issues(vctx, physics_spec, settings_values, feature, values) -> list:
    path = feature_path(physics_spec, feature)
    tensor = mesh_kind_of(vctx, physics_spec) == 'tensor'
    issues = []
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('warning', path, f'brillouin_scattering is filled only by a {FREQUENCY_RESPONSE_STUDY} '
                                             f'study solving {physics_spec.name} (this model runs it in '
                                             f'{vctx.study_type}): its result will be missing'))
    if not (values['photoelastic'] or values['moving_boundary']):
        issues.append(Issue('error', path, 'photoelastic and moving_boundary are both off: nothing drives the solid; '
                                           'turn one on'))
    issues.extend(_solid_form_issues(vctx, physics_spec, settings_values, path, values))
    issues.extend(_other_load_issues(vctx, physics_spec, path))
    issues.extend(_sweep_issues(vctx, path, values))
    if tensor:
        issues.extend(_tensor_issues(vctx, physics_spec, settings_values, path, values))
    optical, optical_settings = _optical_issues(vctx, physics_spec, settings_values, path, values, issues)
    if optical is None:
        return issues
    bor = optical.type == BOR_OPTICAL_TYPE
    issues.extend(_detection_issues(vctx, physics_spec, settings_values, path, values, bor))
    issues.extend(_mirror_issues(vctx, physics_spec, path, values, bor))
    issues.extend(_region_issues(vctx, feature, path, values, optical, bor, tensor))
    issues.extend(_material_issues(vctx, physics_spec, settings_values, feature, path, values, optical,
                                   optical_settings, bor, tensor))
    return issues


def _tensor_issues(vctx, physics_spec, settings_values, path, values) -> list:
    """A rectangular-cell solid (BrillouinDrive's _CellGeometry): planar without piezoelectric, the optical and
    collection physics on its own mesh (the drive reads their cells as its cells: no interpolation). Its loss is the
    mechanical_loss_factor alone: viscoelastic moduli are refused by the material checks of the tensor mesh (V18)."""
    issues = []
    if is_axisymmetric(settings_values):
        issues.append(Issue('error', path, 'an axisymmetric solid on a tensor mesh has no Brillouin drive (the '
                                           'em_fdfd_bor drive needs the azimuthal orders of a triangle-mesh solid): '
                                           'put the solid on a triangle mesh'))
    if settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, 'piezoelectric is on: the Brillouin drive of a rectangular-cell solid pairs '
                                           'the plane-strain displacement only (the coupled potential is not '
                                           'modelled); turn piezoelectric off'))
    mesh_spec = vctx.spec.find_mesh(physics_spec)
    for key in ('optical_physics', 'collection_physics'):
        name = values[key]
        other = _physics_named(vctx.spec, name) if name else None
        other_mesh = None if other is None else vctx.spec.find_mesh(other)
        if mesh_spec is not None and other_mesh is not None and other_mesh.name != mesh_spec.name:
            issues.append(Issue('error', path, f"{key} '{name}' runs on the mesh '{other_mesh.name}', the "
                                               f"rectangular-cell solid '{physics_spec.name}' on '{mesh_spec.name}': "
                                               'on a tensor mesh the drive reads the optical cells as its own cells, '
                                               'so put both physics on one mesh (or the solid on a triangle mesh)'))
    return issues


def _solid_form_issues(vctx, physics_spec, settings_values, path, values) -> list:
    """Planar: plane_strain only; axisymmetric: azimuthal_order 0 or 2."""
    if is_axisymmetric(settings_values):
        order = azimuthal_order_of(settings_values)
        if order not in _DRIVEN_ORDERS:
            return [Issue('error', path, f'azimuthal_order {order}: two order-1 optical fields drive the solid orders '
                                         '0 and 2 only; set azimuthal_order 0 (with a second solid of order 2 for the '
                                         'full spectrum) or 2')]
        return []
    condition = settings_values.get('out_of_plane', 'plane_strain')
    if condition != 'plane_strain':
        return [Issue('error', path, f'out_of_plane {condition}: the Brillouin drive pairs the plane-strain rows (xx, '
                                     'yy, xy) of a sheet beam long along z; set out_of_plane plane_strain')]
    return []


_LOAD_FEATURES = ('boundary_load', 'segment_load', 'body_force', 'electrode', 'electrode_domain')
_HOLD_FEATURES = ('displacement', 'prescribed_displacement_domain')
_HELD_NAMES = ('u_x', 'u_y', 'u_z')


def _other_load_issues(vctx, physics_spec, path) -> list:
    """Errors for every other enabled load of the solid: load features, a nonzero (or field-dependent) prescribed
    displacement and couplings writing its body-force slots (the FDT readout needs the drive as the only load; a
    temperature change stays a static prestate)."""
    issues = []
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        if feature.type in _LOAD_FEATURES:
            issues.append(Issue('error', path, f"{feature.type} '{feature.name}' loads '{physics_spec.name}': the "
                                               'Brillouin drive must be its only load (the spectrum reads w^H u of '
                                               'the drive alone); disable it or use another solid physics'))
        elif feature.type in _HOLD_FEATURES:
            raw = _raw(feature)
            moving = [name for name in _HELD_NAMES
                      if raw.get(name) is not None and plain_value(raw.get(name), vctx.parameters) != 0.0]
            if moving:
                issues.append(Issue('error', path, f"{feature.type} '{feature.name}' holds {', '.join(moving)} != 0: "
                                                   'a moving hold drives the solid besides the Brillouin drive; hold '
                                                   'these components at 0 or leave them empty'))
    for coupling in vctx.spec.couplings:
        if not (coupling.enabled and coupling.target == physics_spec.name):
            continue
        forces = [slot for slot in coupling_writes(vctx, coupling) if str(slot).startswith('body_force')]
        if forces:
            issues.append(Issue('error', path, f"coupling '{coupling.name}' writes {', '.join(forces)} of "
                                               f"'{physics_spec.name}': a coupled force is a second load beside the "
                                               'Brillouin drive; disable it'))
    return issues


def _sweep_issues(vctx, path, values) -> list:
    """Reduced sweeps (adaptive / modal) take one drive column: error with several scan positions."""
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        return []
    settings = vctx.spec.study.settings if isinstance(vctx.spec.study.settings, dict) else {}
    method = settings.get('sweep_method', 'direct')
    positions = len(_positions(values))
    if method != 'direct' and positions > 1:
        return [Issue('error', path, f"sweep_method '{method}' reduces one load, but {positions} scan positions give "
                                     f"{positions} drive columns: use sweep_method 'direct' (or one position)")]
    return []


def _optical_issues(vctx, physics_spec, settings_values, path, values, issues):
    """Optical and collection physics, their beams and pairing with the solid (errors appended to issues); returns
    (optical PhysicsSpec, its resolved settings) or (None, None) when it cannot be checked further."""
    name = values['optical_physics']
    candidates = [physics.name for physics in vctx.spec.physics if physics.type in OPTICAL_TYPES]
    alternatives = ', '.join(candidates) or f"none (add an {' / '.join(OPTICAL_TYPES)} physics)"
    if not name:
        issues.append(Issue('error', path, f'optical_physics is empty: name the optical physics (here: '
                                           f'{alternatives})'))
        return None, None
    optical = _physics_named(vctx.spec, name)
    if optical is None or optical.type not in OPTICAL_TYPES:
        found = 'no physics' if optical is None else f'a {optical.type} physics'
        issues.append(Issue('error', path, f"optical_physics '{name}' is {found}: name an "
                                           f"{' / '.join(OPTICAL_TYPES)} physics (here: {alternatives})"))
        return None, None
    optical_type, optical_settings = _type_settings(vctx, optical)
    if optical_settings is None:
        return None, None
    mesh_spec = vctx.spec.find_mesh(optical)
    if mesh_spec is not None and mesh_spec.kind != 'tensor':
        issues.append(Issue('error', path, f"optical physics '{name}' runs on the {mesh_spec.kind} mesh "
                                           f"'{mesh_spec.name}': its driven focused_beam solve needs a tensor "
                                           '(rectangular-cell) mesh'))
        return None, None
    axisymmetric = is_axisymmetric(settings_values)
    if axisymmetric and optical.type != BOR_OPTICAL_TYPE:
        issues.append(Issue('error', path, f"the axisymmetric solid '{physics_spec.name}' pairs with em_fdfd_bor "
                                           f"optics, not {optical.type} '{name}' (a planar cut)"))
        return None, None
    if not axisymmetric and optical.type == BOR_OPTICAL_TYPE:
        issues.append(Issue('error', path, f"em_fdfd_bor optics '{name}' need the solid '{physics_spec.name}' "
                                           'axisymmetric (geometry axisymmetric): the drawing is the (r, z) half '
                                           'plane'))
        return None, None
    issues.extend(_beam_choice_issues(vctx, path, optical, optical_type, values['incident_source'],
                                      'incident_source'))
    collection_name = values['collection_physics']
    if collection_name and collection_name != name:
        issues.extend(_collection_issues(vctx, path, values, optical, optical_settings))
    elif values['collection_source']:   # confocal: the collection beam is the illumination (BrillouinDrive._resolve)
        incident = _chosen_beam(vctx, optical, optical_type, values['incident_source'])
        if incident is not None and values['collection_source'] != incident:
            issues.append(Issue('error', path, f"collection_source '{values['collection_source']}' is another beam of "
                                               f"the optical physics '{name}', whose one solution sums every beam: give "
                                               'the collection beam its own collection_physics, or leave '
                                               'collection_source empty (confocal: the illumination '
                                               f"'{incident}' is the collection field)"))
    return optical, optical_settings


def _chosen_beam(vctx, optical, optical_type, source_name):
    """Name of the focused_beam source_name picks ('' = the only enabled one); None when it does not resolve to one
    (reported by _beam_choice_issues)."""
    beams, _ = _sources(vctx, optical, optical_type)
    names = [feature.name for feature in beams]
    if source_name:
        return source_name if source_name in names else None
    return names[0] if len(names) == 1 else None


def _beam_choice_issues(vctx, path, optical, optical_type, source_name, what) -> list:
    """Exactly one enabled source feature, a focused_beam, which source_name ('' = the only one) names."""
    beams, others = _sources(vctx, optical, optical_type)
    issues = []
    if others:
        issues.append(Issue('error', path, f"'{optical.name}' has other enabled sources "
                                           f"({', '.join(feature.name for feature in others)}): its solution sums "
                                           'every source, so the Brillouin drive needs its focused_beam alone; '
                                           'disable them'))
    names = [feature.name for feature in beams]
    if not beams:
        issues.append(Issue('error', path, f"'{optical.name}' has no enabled {FOCUSED_BEAM_FEATURE} feature: add "
                                           'one (the illumination / collection beam)'))
    elif source_name and source_name not in names:
        issues.append(Issue('error', path, f"{what} '{source_name}' is no enabled {FOCUSED_BEAM_FEATURE} of "
                                           f"'{optical.name}' (enabled: {', '.join(names)})"))
    elif len(beams) > 1:
        issues.append(Issue('error', path, f"'{optical.name}' has {len(beams)} enabled focused beams "
                                           f"({', '.join(names)}): its solution sums them; keep one enabled"))
    return issues


def _collection_issues(vctx, path, values, optical, optical_settings) -> list:
    """collection_physics: an optical physics of the optical type, wavelength and boundaries with its own beam."""
    name = values['collection_physics']
    collection = _physics_named(vctx.spec, name)
    if collection is None or collection.type != optical.type:
        found = 'no physics' if collection is None else f'a {collection.type} physics'
        return [Issue('error', path, f"collection_physics '{name}' is {found}: it must be another {optical.type} "
                                     f"physics like '{optical.name}' (or empty: the optical physics itself)")]
    collection_type, collection_settings = _type_settings(vctx, collection)
    if collection_settings is None:
        return []
    issues = []
    mesh_spec = vctx.spec.find_mesh(collection)
    if mesh_spec is not None and mesh_spec.kind != 'tensor':
        issues.append(Issue('error', path, f"collection physics '{name}' runs on the {mesh_spec.kind} mesh "
                                           f"'{mesh_spec.name}': a focused_beam solve needs a tensor mesh"))
    elif not _same_optical_grid(vctx, optical, collection):
        issues.append(Issue('error', path, f"collection physics '{name}' runs on another optical grid than "
                                           f"'{optical.name}' (mesh '{mesh_spec.name}' against "
                                           f"'{vctx.spec.find_mesh(optical).name}', other window cells): the drive "
                                           'multiplies both fields cell by cell (put both on one mesh)'))
    wavelength, other = optical_settings.get('wavelength'), collection_settings.get('wavelength')
    if (isinstance(wavelength, (int, float)) and isinstance(other, (int, float))
            and abs(wavelength - other) > _WAVELENGTH_RTOL * abs(wavelength)):
        issues.append(Issue('error', path, f"collection physics '{name}' solves wavelength {other:.6g} m, the "
                                           f"optical physics '{optical.name}' {wavelength:.6g} m: the scattered "
                                           'channel is at the incident wavelength (set both equal)'))
    differing = [key for key in sorted(optical_settings) if key.startswith('bc_')
                 and optical_settings.get(key) != collection_settings.get(key)]
    if differing:
        issues.append(Issue('error', path, f"collection physics '{name}' differs from '{optical.name}' in "
                                           f"{', '.join(differing)}: the reciprocal field needs the same "
                                           'boundaries (mirror planes, absorbing faces)'))
    issues.extend(_beam_choice_issues(vctx, path, collection, collection_type, values['collection_source'],
                                      'collection_source'))
    return issues


def _window_edges(mesh) -> tuple:
    """(x, y) window cell edges [m] of a tensor MeshEstimate (the drive fields' cell centres follow from them)."""
    edges = []
    for axis, count, values in (('x', mesh.nx, mesh.x_edges), ('y', mesh.ny, mesh.y_edges)):
        first, last = int(mesh.npml_sides[f'{axis}_lo']), int(count) - int(mesh.npml_sides[f'{axis}_hi'])
        edges.append(np.asarray(values, dtype=np.float64)[first:last + 1])
    return tuple(edges)


def _same_optical_grid(vctx, optical, collection) -> bool:
    """False when the collection physics' tensor mesh has other window cells than the optical physics' (BrillouinDrive
    compares the two DriveFields' cell centres at bind); True on one mesh or when an estimate is missing (reported
    elsewhere)."""
    optical_mesh, collection_mesh = vctx.spec.find_mesh(optical), vctx.spec.find_mesh(collection)
    if optical_mesh is None or collection_mesh is None or optical_mesh.name == collection_mesh.name:
        return True
    first, second = vctx.meshes.get(optical_mesh.name), vctx.meshes.get(collection_mesh.name)
    if any(mesh is None or getattr(mesh, 'kind', '') != 'tensor' for mesh in (first, second)):
        return True
    return all(np.array_equal(a, b) for a, b in zip(_window_edges(first), _window_edges(second)))


def _detection_issues(vctx, physics_spec, settings_values, path, values, bor) -> list:
    """cross_polarized: em_fdfd_bor and azimuthal_order 2 only (+ the torsion info); co_polarized order 0 on BOR
    without an order-2 partner solid: info."""
    issues = []
    order = azimuthal_order_of(settings_values) if is_axisymmetric(settings_values) else 0
    if values['detection'] == 'cross_polarized':
        if not bor:
            issues.append(Issue('error', path, 'detection cross_polarized needs em_fdfd_bor optics (the 90 degree '
                                               'rotation of the m = 1 field about the axis); planar optics: '
                                               'co_polarized'))
        elif order != 2:
            issues.append(Issue('error', path, f'detection cross_polarized drives no part of azimuthal order {order}: '
                                               'it drives the odd order-2 part only (the order-0 torsion is not '
                                               'modelled); set azimuthal_order 2, or detection co_polarized'))
        else:
            issues.append(Issue('info', path, 'cross_polarized: the m = 0 torsion part of the drive is dropped (a '
                                              'diffusive shear response in a liquid, no Brillouin peak); its share is '
                                              'the extra torsion_drive_fraction, a run warning above 1e-3'))
    elif bor and order == 0 and not _order_two_partner(vctx, physics_spec, values['optical_physics']):
        issues.append(Issue('info', path, 'co_polarized em_fdfd_bor optics also drive the solid order 2 (small at low '
                                          'numerical aperture): no axisymmetric solid_mechanics of azimuthal_order 2 '
                                          'with a brillouin_scattering feature on the same optics exists, so the m = 2 '
                                          'part is left out'))
    return issues


def _order_two_partner(vctx, physics_spec, optical_name) -> bool:
    """Another enabled-feature solid of the same type at azimuthal_order 2 driven by the same optics."""
    for other in vctx.spec.physics:
        if other.name == physics_spec.name or other.type != physics_spec.type:
            continue
        _, settings = _type_settings(vctx, other)
        if not (settings and is_axisymmetric(settings) and azimuthal_order_of(settings) == 2):
            continue
        if any(_raw(feature).get('optical_physics') == optical_name for feature in brillouin_features(other)):
            return True
    return False


def _held_faces(vctx, physics_spec) -> set:
    """Outer faces a Roller / Symmetry of the solid holds (the even mirror of the displacement)."""
    faces = set()
    for _, entities in enabled_feature_entities(vctx, physics_spec, ROLLER_FEATURES):
        faces.update(entity for entity in entities if entity in FACES)
    return faces


def _mirror_issues(vctx, physics_spec, path, values, bor) -> list:
    """Model mirror planes (mirror factor 2 each): refused with em_fdfd_bor optics (a beam along z is never
    z-symmetric); planar: each held by a Roller / Symmetry of the solid and no scan offset along its normal."""
    planes = sorted(mirror_faces(vctx.spec))
    if not planes:
        return []
    if bor:
        return [Issue('error', path, f"mirror plane(s) {', '.join(planes)} with em_fdfd_bor optics: a focused beam "
                                     'along the axis z is not mirror symmetric; model the whole (r, z) half plane')]
    issues = []
    held = _held_faces(vctx, physics_spec)
    for face in planes:
        if face not in held:
            issues.append(Issue('error', path, f"{face} is a model mirror plane (the spectrum counts it x 2), but no "
                                               f"Roller or Symmetry of '{physics_spec.name}' holds it: add a Symmetry "
                                               'feature on that face (the even mirror of the vibrations)'))
    for axis in sorted({face[0] for face in planes}):
        offsets = [float(offset) for offset in (values.get(f'scan_offsets_{axis}') or ()) if float(offset) != 0.0]
        if offsets:
            issues.append(Issue('error', path, f'scan_offsets_{axis} moves the focus off the mirror plane(s) normal to '
                                               f'{axis} (offsets {", ".join(f"{offset:.4g}" for offset in offsets)} '
                                               f'm): the mirrored model holds only a symmetric beam; keep {axis} '
                                               'offsets 0 or model the whole width'))
    return issues


def _beam_lines(vctx, optical, source_name):
    """[(beam name, normal axis 'x' / 'y', line position [m], direction sign or None)] of the chosen focused_beam of an
    optical physics (every enabled one when source_name is ''; lines that do not resolve skipped)."""
    lines = []
    for feature in optical.features:
        if not (feature.enabled and feature.type == FOCUSED_BEAM_FEATURE):
            continue
        if source_name and feature.name != source_name:
            continue
        sign = _BEAM_DIRECTIONS.get(_raw(feature).get('direction', 'positive'))
        for name in vctx.objects(feature.selection):
            shape = vctx.shapes.get(name)
            if shape is None or shape.kind != 'line':
                continue
            x0, y0, x1, y1 = (float(shape.values[key]) for key in ('x0', 'y0', 'x1', 'y1'))
            if x0 == x1:
                lines.append((feature.name, 'x', x0, sign))
            elif y0 == y1:
                lines.append((feature.name, 'y', y0, sign))
    return lines


def _region_issues(vctx, feature, path, values, optical, bor, tensor=False) -> list:
    """The drive selection: on the total-field side of the incident (and collection) beam line, at least half an
    optical cell inside the window (the drive samples' hull: window cell centres; a tensor solid reads the cells
    themselves: no hull), except on mirror planes and the em_fdfd_bor axis x_lo; scan offsets of em_fdfd_bor along x
    refused."""
    issues = []
    if bor and values.get('scan_offsets_x'):
        issues.append(Issue('error', path, 'scan_offsets_x with em_fdfd_bor optics: the focus stays on the axis; scan '
                                           'along z with scan_offsets_y'))
    box = _drive_box(vctx, feature)
    size = vctx.domain_size
    if box is None or size is None:
        return issues
    slack = _TOLERANCE * max(size)
    beams = [(optical, values['incident_source'])]
    if values['collection_physics'] and values['collection_physics'] != optical.name:
        collection = _physics_named(vctx.spec, values['collection_physics'])
        if collection is not None:
            beams.append((collection, values['collection_source']))
    lows, highs = {'x': box[0], 'y': box[1]}, {'x': box[2], 'y': box[3]}
    for physics, source_name in beams:
        for beam, axis, position, sign in _beam_lines(vctx, physics, source_name):
            if sign is None:
                issues.append(Issue('warning', path, f"focused_beam '{beam}' of '{physics.name}': its direction is "
                                                     'not known here, so the total-field side of the drive selection '
                                                     'is not checked'))
            elif (sign > 0 and lows[axis] < position - slack) or (sign < 0 and highs[axis] > position + slack):
                side = 'above' if sign > 0 else 'below'
                issues.append(Issue('error', path, f"the drive selection reaches {axis} = "
                                                   f"{lows[axis] if sign > 0 else highs[axis]:.6g} m, upstream of the "
                                                   f"focused_beam '{beam}' line of '{physics.name}' at {axis} = "
                                                   f'{position:.6g} m, where only the scattered field is solved: keep '
                                                   f'the selection {side} the line (total-field side)'))
    if not tensor:
        issues.extend(_window_issues(vctx, path, optical, box, bor, slack))
    return issues


def _window_issues(vctx, path, optical, box, bor, slack) -> list:
    """Drive box inside the optical sample hull (first / last window cell centres) on every face but the mirror
    planes and (em_fdfd_bor) the axis x_lo."""
    mesh_spec = vctx.spec.find_mesh(optical)
    mesh = None if mesh_spec is None else vctx.meshes.get(mesh_spec.name)
    if mesh is None or getattr(mesh, 'kind', '') != 'tensor':
        return []
    exempt = set(mirror_faces(vctx.spec)) | ({'x_lo'} if bor else set())
    issues = []
    for axis, (low, high) in (('x', (box[0], box[2])), ('y', (box[1], box[3]))):
        first, last = _window_centres(mesh, axis)
        reaches = ((f'{axis}_lo', low < first - slack, first), (f'{axis}_hi', high > last + slack, last))
        for face, outside, centre in reaches:
            if outside and face not in exempt:
                issues.append(Issue('error', path, f"the drive selection reaches within half an optical cell of the "
                                                   f"{face} window edge of '{optical.name}' (its samples end at the "
                                                   f'cell centre {axis} = {centre:.6g} m): select a drive domain '
                                                   "inside the window ('all' reaches every edge), away from the "
                                                   'absorbing layers'))
    return issues


def _material_issues(vctx, physics_spec, settings_values, feature, path, values, optical, optical_settings,
                     bor, tensor=False) -> list:
    """Materials under the drive selection (material preview): photoelastic data and loss (warnings), crystals with
    em_fdfd_bor (error), the reference laws and the reference homogeneous rules (errors), with photoelastic off an
    optical interface at the selection (error), optical and solid resolution (warnings) and the estimated Brillouin
    line against the study frequencies (info)."""
    indices = _drive_preview_indices(vctx, feature)
    if indices is None:
        return []
    data = _material_data(vctx, indices)
    parameters = vctx.parameters
    issues = []
    if values['photoelastic']:
        silent = [name for name, properties, tensor in data if properties is not None and tensor is None
                  and not _may_be_nonzero(properties, _PHOTOELASTIC_PROPERTIES, parameters)]
        if silent:
            issues.append(Issue('warning', path, f"material(s) {', '.join(silent)} in the drive selection have no "
                                                 'photoelastic_p11 / _p12 / _p44: no photoelastic drive there'))
        crystals = [name for name, _, tensor in data if tensor is not None]
        if bor and crystals:
            issues.append(Issue('error', path, f"material(s) {', '.join(crystals)} carry a crystal photoelastic "
                                               'tensor, which is not invariant under rotation about the axis: '
                                               'em_fdfd_bor optics take isotropic photoelastic_p11 / _p12 / _p44 only '
                                               '(set them on the material)'))
    lossless = [name for name, properties, _ in data
                if properties is not None and not _may_be_nonzero(properties, _LOSS_PROPERTIES, parameters)]
    if lossless:
        issues.append(Issue('warning', path, f"material(s) {', '.join(lossless)} in the drive selection have no loss "
                                             '(mechanical_loss_factor, viscosity, bulk_viscosity or relaxation '
                                             'branches): the thermal spectrum of a lossless solid is 0 between '
                                             'delta lines'))
    if values['reference'] != 'none':
        issues.extend(_reference_law_issues(path, data, values['reference'], parameters,
                                            settings_values.get('formulation') == MIXED))
    if values['reference'] == 'homogeneous':
        issues.extend(_homogeneous_issues(vctx, feature, path, optical, indices, data, settings_values))
    if not values['photoelastic']:
        issues.extend(_interface_issues(vctx, feature, path))
    wavelength = optical_settings.get('wavelength')
    if isinstance(wavelength, (int, float)) and wavelength > 0.0:
        issues.extend(_resolution_issues(vctx, feature, path, optical, data, settings_values, float(wavelength),
                                         tensor))
        issues.extend(_line_issues(vctx, path, data, float(wavelength)))
    return issues


def _reference_law_issues(path, data, reference, parameters, mixed) -> list:
    """A reference spectrum needs an isotropic longitudinal law per drive material (BrillouinDrive._reference_laws:
    the youngs_modulus or the shear_modulus / bulk_modulus route with a finite bulk modulus): errors for materials of
    the c_ij route (reference 'local': the homogeneous rules report c_ij themselves) and for incompressible ones
    (formulation mixed: shear_modulus without bulk_modulus, a viscosity-only liquid; the displacement formulation
    refuses both itself)."""
    anisotropic, incompressible = [], []
    for name, properties, _ in data:
        if properties is None:
            continue
        youngs = _may_be_nonzero(properties, ('youngs_modulus',), parameters)
        moduli = any(properties.get(key) is not None for key in ('shear_modulus', 'bulk_modulus'))
        if not (youngs or moduli):
            if any(properties.get(key) is not None for key in STIFFNESS_TENSOR_ARRAYS):
                anisotropic.append(name)
            elif mixed and _may_be_nonzero(properties, VISCOSITY_ARRAYS[:1], parameters):
                incompressible.append(name)
        elif mixed and moduli and not youngs and properties.get('bulk_modulus') is None:   # displacement: V-refused
            incompressible.append(name)
    issues = []
    if anisotropic and reference != 'homogeneous':
        issues.append(Issue('error', path, f"reference '{reference}' needs isotropic moduli, but "
                                           f"{', '.join(anisotropic)} in the drive selection take c_ij entries: set "
                                           "reference 'none' (or give them youngs_modulus / poisson_ratio)"))
    if incompressible:
        issues.append(Issue('error', path, f"reference '{reference}' needs a finite bulk modulus in the drive "
                                           'selection (an incompressible material has no longitudinal sound wave), '
                                           f"but {', '.join(incompressible)} set no bulk_modulus: set reference "
                                           "'none' or give it a bulk_modulus"))
    return issues


def _interface_issues(vctx, feature, path) -> list:
    """photoelastic off: the moving boundary is the whole drive, so an optical interface must touch the selection
    (BrillouinDrive raises at bind otherwise): an error when the selection and its neighbours on the material preview
    hold one material, or only materials of one plain permittivity (no material / an expression counts as another
    one)."""
    indices = _drive_preview_indices(vctx, feature, margin=1)
    if indices is None:
        return []
    permittivities = set()
    for index in indices:
        data = _material_data(vctx, [index]) if index > 0 else []
        value = (_refractive_index(data[0][1], vctx.parameters) if data and data[0][1] is not None else None)
        permittivities.add(('unknown', index) if value is None else round(value, 12))
    if len(indices) > 1 and len(permittivities) > 1:
        return []
    return [Issue('error', path, 'photoelastic is off and no optical interface touches the drive selection (its '
                                 'materials and their neighbours share one permittivity): nothing drives the solid '
                                 '(turn photoelastic on or select the interfaces)')]


def _homogeneous_issues(vctx, feature, path, optical, indices, data, settings_values) -> list:
    """reference homogeneous: planar, one material (isotropic: no c_ij, no crystal entry) over the whole drive
    selection, uniform optical cells over it."""
    if is_axisymmetric(settings_values):
        return [Issue('error', path, "reference 'homogeneous' is the planar plane-wave sum: an axisymmetric solid "
                                     "takes reference 'local' or 'none'")]
    issues = []
    planes = sorted(mirror_faces(vctx.spec))
    if planes:
        issues.append(Issue('error', path, f"reference 'homogeneous' needs a model without mirror planes (its "
                                           f"plane-wave sum takes the unfolded drive), but {', '.join(planes)} "
                                           "mirror it: use reference 'local' or model the whole width"))
    if len(indices) != 1 or indices[0] == 0:
        names = [vctx.spec.materials[index - 1].name if index > 0 else '(no material)' for index in indices]
        issues.append(Issue('error', path, f"reference 'homogeneous' needs one uniform material over the drive "
                                           f"selection, found {', '.join(names)}: use reference 'local'"))
    anisotropic = [name for name, properties, tensor in data if tensor is not None or (
        properties is not None and any(properties.get(key) is not None for key in STIFFNESS_TENSOR_ARRAYS))]
    if anisotropic:
        issues.append(Issue('error', path, f"reference 'homogeneous' needs an isotropic medium, but "
                                           f"{', '.join(anisotropic)} set c_ij or crystal entries: use reference "
                                           "'local'"))
    mesh_spec = vctx.spec.find_mesh(optical)
    mesh = None if mesh_spec is None else vctx.meshes.get(mesh_spec.name)
    box = _drive_box(vctx, feature)
    if mesh is not None and getattr(mesh, 'kind', '') == 'tensor' and box is not None:
        uneven = [axis for axis, edges, (low, high) in (('x', mesh.x_edges, (box[0], box[2])),
                                                         ('y', mesh.y_edges, (box[1], box[3])))
                  if not has_uniform_spacing(_box_widths(edges, low, high))]
        if uneven:
            issues.append(Issue('error', path, f"reference 'homogeneous' samples the drive stress on uniform optical "
                                               f"cells, but the cells of '{optical.name}' vary along "
                                               f"{' and '.join(uneven)} over the drive selection: make the mesh "
                                               "uniform there or use reference 'local'"))
    return issues


def _resolution_issues(vctx, feature, path, optical, data, settings_values, wavelength, tensor=False) -> list:
    """Optical cells per lambda0 / n and the solid element size against lambda0 / (12 n) (linear: / 24; a tensor
    solid's Q1 cells, the optical ones: / 24) in the drive region (n = the largest plain index of its materials)."""
    indices = [_refractive_index(properties, vctx.parameters) for _, properties, _ in data]
    indices = [index for index in indices if index is not None]
    if not indices:
        return []
    index = max(indices)
    issues = []
    box = _drive_box(vctx, feature)
    mesh_spec = vctx.spec.find_mesh(optical)
    mesh = None if mesh_spec is None else vctx.meshes.get(mesh_spec.name)
    if mesh is not None and getattr(mesh, 'kind', '') == 'tensor' and box is not None:
        widest = max(float(_box_widths(mesh.x_edges, box[0], box[2]).max()),
                     float(_box_widths(mesh.y_edges, box[1], box[3]).max()))
        cells = wavelength / (index * widest)
        if cells < _OPTICAL_CELLS_PER_WAVELENGTH:
            issues.append(Issue('warning', path, f"'{optical.name}' has {cells:.3g} cells per wavelength lambda0 / n = "
                                                 f'{wavelength / index:.4g} m in the drive region (n = {index:.4g}, '
                                                 f'cells up to {widest:.4g} m): the drive E_c E_i oscillates on half '
                                                 f'that; use >= {_OPTICAL_CELLS_PER_WAVELENGTH:g}'))
        limit = wavelength / (_SOLID_SIZE_DIVISOR[1] * index)
        if tensor and widest > limit * (1.0 + _TOLERANCE):
            issues.append(Issue('warning', path, f'the rectangular solid cells up to {widest:.4g} m in the drive '
                                                 f'region exceed lambda0 / ({_SOLID_SIZE_DIVISOR[1]:g} n) = '
                                                 f'{limit:.4g} m (n = {index:.4g}): bilinear cells raise the Brillouin '
                                                 'shift by about (q h)^2 / 24 (q = 4 pi n / lambda0); refine the '
                                                 'tensor mesh there or put the solid on a triangle mesh'))
    if tensor:
        return issues
    solid = vctx.mesh
    sizing = getattr(solid, 'sizing', None)
    if sizing is not None:
        objects = vctx.objects(feature.selection)   # 'all' removed: then the largest size of the mesh
        sizes = [sizing.object_sizes.get(name, sizing.h_max) for name in objects] or [sizing.h_max]
        try:
            if 'all' in expand_entities(vctx.spec, feature.selection):
                sizes.append(sizing.h_max)
        except (ValueError, KeyError, TypeError):   # reported by the selection checks
            pass
        size = float(max(sizes))
        order = element_order(settings_values)
        limit = wavelength / (_SOLID_SIZE_DIVISOR[order] * index)
        if size > limit * (1.0 + _TOLERANCE):
            issues.append(Issue('warning', path, f'solid elements up to {size:.4g} m in the drive region exceed '
                                                 f'lambda0 / ({_SOLID_SIZE_DIVISOR[order]:g} n) = {limit:.4g} m '
                                                 f"({'quadratic' if order == 2 else 'linear'} elements, n = "
                                                 f'{index:.4g}): the drive oscillates on lambda0 / (2 n); refine the '
                                                 'drive objects of the triangle mesh'))
    return issues


def _line_issues(vctx, path, data, wavelength) -> list:
    """Info per drive material whose estimated Brillouin line (f_B = 2 n v_L / lambda0, FWHM = q^2 (4/3 viscosity +
    bulk_viscosity) / (2 pi rho) + loss factor f_B, q = 4 pi n / lambda0) has no study frequency within _LINEWIDTHS
    FWHM."""
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        return []
    try:
        study_values = vctx.values(vctx.registry.study_type(FREQUENCY_RESPONSE_STUDY).params, vctx.spec.study.settings)
    except KeyError:
        return []
    if study_values is None:
        return []
    frequencies = np.asarray(study_frequencies(study_values), dtype=np.float64)
    issues = []
    for name, properties, _ in data:
        index = _refractive_index(properties, vctx.parameters)
        line = _longitudinal_line(properties, vctx.parameters) if properties is not None else None
        if index is None or line is None:
            continue
        speed, viscous, loss = line
        shift = 2.0 * index * speed / wavelength
        q = 4.0 * math.pi * index / wavelength
        width = q * q * viscous + loss * shift
        if width <= 0.0:
            continue
        if not np.any(np.abs(frequencies - shift) <= _LINEWIDTHS * width):
            issues.append(Issue('info', path, f"material '{name}': estimated Brillouin shift f_B = 2 n v / lambda0 = "
                                              f'{shift:.4g} Hz (n = {index:.4g}, v = {speed:.5g} m/s, linewidth about '
                                              f'{width:.3g} Hz), but no study frequency lies within '
                                              f'{_LINEWIDTHS:g} linewidths of it: the peak may be missed (a focused '
                                              'beam shifts it lower by about the mean cos of its angles)'))
    return issues


# ---- memory and run-time estimates (CPU; the solid's MemoryContext / TimeContext in its study) ----
# Counted from the allocations of SolidMechanicsFEMPhysics.bind_physics_drives / _drive_inputs / _record_drives (a
# tensor solid: SolidMechanicsPhysics' twins and BrillouinDrive's _CellGeometry: _cell_bind_bytes),
# BrillouinDrive.bind / result, brillouin.py, fem.elastic_drives and frequency_response.HarmonicSystem (set_functionals,
# _rows_solve, _kept_functionals, _reduced_functionals, solve_columns). Kept: the ONE drive array W (columns =
# positions x parts, every unknown; complex128) that is also the load (row 0, a view) and the HarmonicSystem
# functionals (the caller's array, no copy); a concatenation only with several drives on one solid (validate allows
# one); the reference inputs of the bound drive (local: compressed |q| bins; homogeneous: the stress rasters and the
# _Raster cells); cuDSS only: the device functional weights, uploaded at the first rows-only or column solve. Host items
# are summed (no host phases), so the transients of steps that never overlap (one bind with its drive points, samplers
# and solid inputs; one frequency's readout and column solves; the reference spectra of result) are ONE item, their
# largest; on the device the column solves are the phase BRILLOUIN_COLUMN_PHASE.
_COMPLEX_BYTES = 16
_FLOAT_BYTES = 8
_FLOAT32_BYTES = 4
_INT64_BYTES = 8
_BOOL_BYTES = 1
# per drive point, held through a bind: DrivePoints points (2 float64), weights (float64), elements (int64); p11 / p12 /
# p44 gathered per point (float64); the volume _GridSamples index (4 int64), weight (4 float64), cell (int64), with
# em_fdfd_bor optics its axis factor (float64)
_POINT_BYTES = 2 * _FLOAT_BYTES + _FLOAT_BYTES + _INT64_BYTES
_POINT_MATERIAL_BYTES = 3 * _FLOAT_BYTES
_SAMPLER_BYTES = 4 * _INT64_BYTES + 4 * _FLOAT_BYTES + _INT64_BYTES
_AXIS_FACTOR_BYTES = _FLOAT_BYTES
# one-off per sampler build (first position): _clamped's copy (2 float64), bilinear_weights' i, j (int64), tx, ty
# (float64), the index stack (4 int64) and its astype copy (4 int64), the weight stack (4 float64)
_SAMPLER_SETUP_BYTES = 2 * _FLOAT_BYTES + 2 * _INT64_BYTES + 2 * _FLOAT_BYTES + 2 * 4 * _INT64_BYTES + 4 * _FLOAT_BYTES
_GATHER_BYTES = 4 * _COMPLEX_BYTES    # bilinear_sample's flat[index] gather (n, 4) of one complex component
# complex values per point at the optical-stress peak of one position: planar field_products' 6 Voigt rows beside
# optomechanics.inverse_permittivity_changes' 6 row expressions, their np.stack and its astype copy (4 x 6); an
# axisymmetric solid (em_fdfd_bor): azimuthal_products' 3 harmonics x 6 rows held while optical_stress runs per
# harmonic (2 done x 6, the current one's rows / stack / astype 3 x 6)
_STRESS_PEAK_VALUES = {False: 4 * 6, True: 3 * 6 + 2 * 6 + 3 * 6}
# per point of _local_lines (every reference: q_mean): q_x, q_y samples, q_squared, safe, weights, q (float64), the
# longitudinal projection and its two complex temporaries, its |.|^2 (float64), one sampling gather (4 float64); the
# local reference bins: keep / selected (bool), values, masses (float64), bins, inverse (int64)
_LINE_POINT_BYTES = (6 * _FLOAT_BYTES + 3 * _COMPLEX_BYTES + _FLOAT_BYTES + 4 * _FLOAT_BYTES + 2 * _BOOL_BYTES
                     + 2 * _FLOAT_BYTES + 2 * _INT64_BYTES)
# per optical window cell of _local_lines: the product P = E_c . E_i (complex), local_wavenumbers' present mask, the
# first axis' result (float64) while _phase_gradient of the second holds its gradient (float64), its conjugate and
# product (complex), angle and quotient (float64)
_LOCAL_WINDOW_BYTES = _COMPLEX_BYTES + _BOOL_BYTES + 2 * _FLOAT_BYTES + 2 * _COMPLEX_BYTES + 2 * _FLOAT_BYTES
# per mesh edge of the moving-boundary search (_interfaces): interior ids, first / second (int64), touching / differ
# (bool), one property's two gathers (float64) and its three comparison masks (bool)
_INTERFACE_SEARCH_BYTES = 3 * _INT64_BYTES + 2 * _BOOL_BYTES + 2 * _FLOAT_BYTES + 3 * _BOOL_BYTES
# per moving-boundary Gauss point, held through a bind: EdgeDrivePoints points, normals (2 float64 each), weights
# (float64); the two field side samplers (+ axis factor) and the isotropic eps of each side (complex, read by two
# transient diagonal samplers: their set-up and arrays are the bind set-up's); per position the side samples (2
# sides x components, complex, x 2 with a separate collection field) and the Johnson forms' temporaries and traction
# (_EDGE_POSITION_VALUES complex)
_EDGE_POINT_BYTES = 5 * _FLOAT_BYTES + 2 * (_SAMPLER_BYTES + _AXIS_FACTOR_BYTES) + 2 * _COMPLEX_BYTES
_EDGE_POSITION_VALUES = 14 + 2
_EDGE_BYTES = 3 * _INT64_BYTES   # per interface edge: EdgeDrivePoints edges and sides (int64)
# moving-boundary interface edges: a compact drive region of E elements has ~ this x sqrt(E) edges where the optical
# permittivity jumps (uncalibrated: the run's count is the bind summary's interface points)
_INTERFACE_EDGES_GUESS = 4.0
# local reference: the ratio of the largest to the smallest weighted |q| of the drive points of one material group, so
# its kept bins are at most log(ratio) / log1p(LINE_BIN) (and at most its points; uncalibrated: the run's bins are the
# lines BrillouinDrive keeps)
_LOCAL_Q_SPREAD_GUESS = 2.0
_LINE_BYTES = 2 * _FLOAT_BYTES   # per kept bin: the weighted-mean q and the mass (float64)
# host numpy calls of one position's bind (samples, products, stress, element chunks, edge drive) and of one fit
_BIND_CALLS = 120
_DHO_FIT_CALLS_GUESS = 2000   # scipy least_squares evaluations x numpy calls of dho_fit (uncalibrated)
# homogeneous_reference: zero padding (padding=2 per axis); per padded cell in _projections the spectrum (3 complex),
# q^2 and safe (float64), three complex temporaries and the two results (float64); in the frequency loop the spectrum
# and the raveled longitudinal / transverse / q^2 (float64), per chunk value (frequency x wavevector) the total, real,
# imaginary, the denominator, the numerator product, its np.where and the quotient (7 float64)
_HOMOGENEOUS_PADDING = 4
_HOMOGENEOUS_SPECTRUM_BYTES = 3 * _COMPLEX_BYTES + 2 * _FLOAT_BYTES + 3 * _COMPLEX_BYTES + 2 * _FLOAT_BYTES
_HOMOGENEOUS_LOOP_BYTES = 3 * _COMPLEX_BYTES + 3 * _FLOAT_BYTES
_HOMOGENEOUS_CHUNK_ARRAYS = 7
# local_reference per chunk value (frequency x bin): the squared difference, the width term and their sum, then the
# line (3 float64 at once)
_LOCAL_CHUNK_ARRAYS = 3
# _Raster per drive-box optical cell (kept with the bound drive): cells, i, j (int64), p11, p12, p44 (float64); + eps
# per field component (complex)
_RASTER_CELL_BYTES = 3 * _INT64_BYTES + 3 * _FLOAT_BYTES
# host passes of one extra column (solve_columns, host factor): the free rhs gather (read + write), the reduced vector
# (zeros + scatter of the solution), the vdot (W row + solution); a cuDSS factor: the gather only (its dot runs on
# the device)
_COLUMN_HOST_PASSES = 6
_COLUMN_DEVICE_HOST_PASSES = 2
BRILLOUIN_COLUMN_PHASE = 'Brillouin column solve'   # the extra drive columns on the kept cuDSS factor
# rectangular cells (BrillouinDrive._CellGeometry, a tensor solid on the optics' mesh): per drive point (a selected
# window cell) held through a bind: elements (int64) and weights (float64) (its sampler shares elements); p11 / p12 /
# p44 are _POINT_MATERIAL_BYTES
_CELL_POINT_BYTES = _INT64_BYTES + _FLOAT_BYTES
# per window cell of the bind set-up: the dA raster np.outer (float64) and the flat cell index of the face search
# (int64)
_CELL_SETUP_BYTES = _FLOAT_BYTES + _INT64_BYTES
# per interior Yee face of the interface search (both axes): the lo / hi index copies (int64), the candidate faces and
# their lo / hi gathers (int64), the two eps gathers (complex), the abs / max temporaries (3 float64) and the touching /
# member / differ / comparison / keep masks (8 bool)
_CELL_FACE_SEARCH_BYTES = 5 * _INT64_BYTES + 2 * _COMPLEX_BYTES + 3 * _FLOAT_BYTES + 8 * _BOOL_BYTES
# per moving-boundary face held through a bind: normal (2 float64), length (float64), its face index and its lo / hi
# cells (3 int64), the isotropic eps of both sides (complex)
_CELL_FACE_BYTES = 3 * _FLOAT_BYTES + 3 * _INT64_BYTES + 2 * _COMPLEX_BYTES
# complex values per window cell of one stress assembly (stress_drive / brillouin.cell_stress_drive): the zero-padded
# raster (3 rows), the normal / shear products (4) and one corner sum's two temporaries; of one face drive
# (traction_drive / face_drive): the x and y face loads (about 2 per cell) and a half-load temporary
_CELL_DRIVE_VALUES = 3 + 4 + 2
_CELL_FACE_DRIVE_VALUES = 2 + 1


def _rule_points() -> tuple:
    """Element and edge quadrature points of a drive (fem.elastic_drives.DRIVE_DEGREE; edges at geometry order 2)."""
    return len(triangle_rule(DRIVE_DEGREE).weights), len(gauss_line(DRIVE_DEGREE + 2)[1])


def _drive_elements(spec, parameters, feature, mesh, shapes) -> int:
    """Solid elements of the drive selection: all of them for 'all', else the elements times the share of the
    selection's union bounding box in the meshed extent (the triangle rule of meshing._selection_cells)."""
    total = int(mesh.n_elements)
    try:
        expanded = expand_entities(spec, feature.selection)
    except (ValueError, KeyError, TypeError):
        return 0
    if 'all' in expanded:
        return total
    boxes = [box for _, box in domain_object_boxes(spec, shapes, [feature.selection])]
    if not boxes:
        return 0
    extent = mesh.extent
    share = 1.0
    for axis, (low, high) in enumerate(((min(box[0] for box in boxes), max(box[2] for box in boxes)),
                                        (min(box[1] for box in boxes), max(box[3] for box in boxes)))):
        mesh_low, mesh_high = extent[2 * axis], extent[2 * axis + 1]
        share *= max(min(high, mesh_high) - max(low, mesh_low), 0.0) / (mesh_high - mesh_low)
    return int(math.ceil(total * min(share, 1.0)))


def _optical_cells(spec, parameters, optical, feature, shapes):
    """(window cells, drive-box cells) of the optical mesh at its base dx (refined by refinements of the drive
    objects): exact on a uniform mesh, the base cells otherwise; (0, 0) when nothing resolves."""
    mesh_spec = spec.find_mesh(optical) if optical is not None else None
    if mesh_spec is None:
        return 0, 0
    base = plain_value(mesh_spec.dx, parameters)
    if base is None or not base > 0.0:
        return 0, 0
    try:
        size_x, size_y = domain_size(spec, parameters)
        expanded = expand_entities(spec, feature.selection)
    except (ValueError, KeyError, TypeError):
        return 0, 0
    sizes = {'x': base, 'y': base}
    for refinement in mesh_spec.refinements:
        if getattr(refinement, 'object', None) in expanded:
            for axis in ('x', 'y'):
                value = plain_value(getattr(refinement, f'dx_{axis}', None) or refinement.dx, parameters)
                if value is not None and value > 0.0:
                    sizes[axis] = min(sizes[axis], value)
    window = uniform_cell_count(size_x, base) * uniform_cell_count(size_y, base)
    if 'all' in expanded:
        box = (0.0, 0.0, size_x, size_y)
    else:
        boxes = [box for _, box in domain_object_boxes(spec, shapes, [feature.selection])]
        if not boxes:
            return window, 0
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    drive = (uniform_cell_count(max(box[2] - box[0], sizes['x']), sizes['x'])
             * uniform_cell_count(max(box[3] - box[1], sizes['y']), sizes['y']))
    return window, drive


class _SolidForm:
    """The solid's sizes the drive items read: its strain rows (planar 3, axisymmetric m = 0 4, m >= 1 6), axisymmetric
    (em_fdfd_bor optics), the azimuthal order, mesh elements and edges (a tensor solid: its window cells and interior
    Yee faces), the drive vector (u rows: c x dofs) and the harmonic unknowns (mixed: + the pressure dofs), local dofs
    and components per element, the MaterialGrid arrays _drive_inputs copies and whether the viscoelastic
    ElementModuli are kept (else the bind builds them)."""

    def __init__(self, context, settings_values, mesh, *, unknowns, vector_dofs, local, components, moduli_kept=True):
        self.tensor = getattr(mesh, 'kind', '') == 'tensor'
        self.axisymmetric = is_axisymmetric(settings_values)
        self.order = azimuthal_order_of(settings_values) if self.axisymmetric else 0
        self.rows = 3 if not self.axisymmetric else (6 if self.order else 4)
        if self.tensor:
            n_x, n_y = (len(edges) - 1 for edges in _window_edges(mesh))
            self.elements, self.mesh_edges = n_x * n_y, (n_x - 1) * n_y + n_x * (n_y - 1)
            self.material_arrays = _solid_array_count(context)
        else:
            self.elements = int(mesh.n_elements)
            self.mesh_edges = int(getattr(mesh, 'n_edges', 0) or 0)
            self.material_arrays = _material_array_count(context)
        self.vector_dofs, self.unknowns = float(vector_dofs), float(unknowns)
        self.local, self.components = int(local), int(components)
        self.moduli_kept = bool(moduli_kept)


def _material_array_count(context) -> int:
    """MaterialGrid arrays of the solid's mesh that _drive_inputs copies: the material_schema union of the physics on
    it (optional and cross-mesh partner arrays not counted)."""
    spec = context.spec
    mesh_spec = spec.find_mesh(context.physics_spec)
    on_mesh = spec.physics_on(mesh_spec.name) if mesh_spec is not None else [context.physics_spec]
    names = set()
    for physics in on_mesh:
        try:
            names.update(context.registry.physics_type(physics.type).material_schema)
        except KeyError:   # reported by validate
            continue
    return len(names)


def _solid_array_count(context) -> int:
    """MaterialGrid arrays a tensor solid's _drive_inputs copies: its own material_schema and, when a material sets it,
    mechanical_loss_factor (other optional arrays it reads, c_ij, not counted)."""
    try:
        schema = context.registry.physics_type(context.physics_spec.type).material_schema
    except KeyError:   # reported by validate
        return 0
    return len(schema) + int(bool(context.material_sets(LOSS_ARRAYS[0])))


def _tensor_drive_cells(spec, feature, mesh, shapes) -> int:
    """Window cells of a tensor MeshEstimate under the drive selection: every one for 'all', else those overlapping
    the union bounding box of its domain objects (exact for one rectangle)."""
    x_edges, y_edges = _window_edges(mesh)
    try:
        expanded = expand_entities(spec, feature.selection)
    except (ValueError, KeyError, TypeError):
        return 0
    if 'all' in expanded:
        return (len(x_edges) - 1) * (len(y_edges) - 1)
    boxes = [box for _, box in domain_object_boxes(spec, shapes, [feature.selection])]
    if not boxes:
        return 0
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    return int(_box_widths(x_edges, box[0], box[2]).size * _box_widths(y_edges, box[1], box[3]).size)


class _DriveCounts:
    """Sizes of one enabled brillouin_scattering feature of the solid (brillouin_memory_items / _time_items): columns
    = positions x azimuthal parts, selected elements, drive points, interface edges and their Gauss points, optical
    field components, reads (2 with a separate collection field), (window, drive-box) optical cells."""

    def __init__(self, feature, values, positions, parts, elements, points, edges, edge_points, field_components,
                 reads, cells):
        self.feature, self.values, self.positions, self.parts = feature, values, positions, parts
        self.elements, self.points, self.edges, self.edge_points = elements, points, edges, edge_points
        self.field_components, self.reads, self.cells = field_components, reads, cells

    @property
    def columns(self) -> int:
        return self.positions * self.parts

    def bins(self) -> int:
        """Kept local-reference bins per position (one material group: _LOCAL_Q_SPREAD_GUESS)."""
        return min(self.points, int(math.ceil(math.log(_LOCAL_Q_SPREAD_GUESS) / math.log1p(LINE_BIN))))


def _drive_parts(values, settings_values) -> int:
    """Columns per scan position: planar 1; axisymmetric the parts of brillouin.AZIMUTHAL_PARTS[detection] at the
    solid's order (1 for every validated order / detection)."""
    if not is_axisymmetric(settings_values):
        return 1
    order = azimuthal_order_of(settings_values)
    return max(sum(1 for key in AZIMUTHAL_PARTS.get(values['detection'], ()) if key[0] == order), 1)


def _drive_counts(context, settings_values, mesh) -> list:
    """[_DriveCounts] of the solid's enabled brillouin_scattering features in a frequency_response study ([]
    otherwise; values that do not resolve are skipped: validate reports them). A tensor solid: one point per drive
    cell, one per moving-boundary face."""
    spec, parameters = context.spec, context.parameters
    features = brillouin_features(context.physics_spec)
    kind = getattr(mesh, 'kind', '')
    if spec.study.type != FREQUENCY_RESPONSE_STUDY or not features or kind not in ('triangles', 'tensor'):
        return []
    shapes = resolve_geometry(spec, parameters, errors={})
    element_points, edge_rule_points = (1, 1) if kind == 'tensor' else _rule_points()
    counts = []
    for feature in features:
        try:
            values = resolve_values(_PARAMS, feature.params, parameters)
        except (ValueError, TypeError):
            continue
        optical = _physics_named(spec, values['optical_physics'])
        if optical is None:
            continue
        elements = (_tensor_drive_cells(spec, feature, mesh, shapes) if kind == 'tensor'
                    else _drive_elements(spec, parameters, feature, mesh, shapes))
        edges = int(math.ceil(_INTERFACE_EDGES_GUESS * math.sqrt(elements))) if values['moving_boundary'] else 0
        collection = values['collection_physics']
        reads = 2 if collection and collection != optical.name else 1
        counts.append(_DriveCounts(feature, values, len(_positions(values)), _drive_parts(values, settings_values),
                                   elements, elements * element_points, edges, edges * edge_rule_points,
                                   _FIELD_COMPONENTS.get(optical.type, 3), reads,
                                   _optical_cells(spec, parameters, optical, feature, shapes)))
    return counts


def _study_values(context):
    """Resolved frequency_response study settings (None when they do not resolve: validate reports them)."""
    try:
        return resolve_values(context.registry.study_type(FREQUENCY_RESPONSE_STUDY).params,
                              context.spec.study.settings if isinstance(context.spec.study.settings, dict) else {},
                              context.parameters)
    except (KeyError, ValueError, TypeError):
        return None


def _chunk_point_bytes(form) -> int:
    """Host bytes per point of one assemble_stress_drive element chunk: the weighted stress and its transposed copy
    (rows complex each), the strain operator (rows x c local float64) and the element geometry / basis gradients of
    _operator_at (2 local + 9 float64)."""
    return (2 * form.rows * _COMPLEX_BYTES + form.rows * form.components * form.local * _FLOAT_BYTES
            + (2 * form.local + 9) * _FLOAT_BYTES)


def _bind_bytes(drive, form) -> int:
    """Host bytes of one drive's bind at its peak beside the kept W, in BrillouinDrive.bind's order. Held through every
    position: the SolidDriveInputs copies (float64 per element and MaterialGrid array; without kept viscoelastic moduli
    also the ElementModuli and its two E / nu reads), the selection, the drive points with their materials, eps,
    samplers (and law per point) and the interface edge points. On top the largest of: the first position's set-up
    (samplers, interface search, the isotropic eps window, the element laws, one property copy's temporaries) and one
    position's steps (the new samples beside the previous ones; the optical-stress peak; the assembly with one element
    chunk and the new and previous drive vectors; the local lines on the window; the next position's optics read
    beside the previous field grid). A tensor solid: _cell_bind_bytes."""
    if form.tensor:
        return _cell_bind_bytes(drive, form)
    points, edge_points = drive.points, drive.edge_points
    optical = drive.field_components
    window = drive.cells[0]
    vector = _COMPLEX_BYTES * form.vector_dofs
    axis = _AXIS_FACTOR_BYTES if form.axisymmetric else 0
    laws = _INT64_BYTES if drive.values['reference'] != 'none' else 0
    held = (form.material_arrays * form.elements * _FLOAT_BYTES + _BOOL_BYTES * form.elements
            + _INT64_BYTES * drive.elements
            + points * (_POINT_BYTES + _POINT_MATERIAL_BYTES + _SAMPLER_BYTES + axis + optical * _COMPLEX_BYTES + laws)
            + edge_points * _EDGE_POINT_BYTES + drive.edges * _EDGE_BYTES)
    if not form.moduli_kept:   # _drive_inputs: element_moduli of the E / nu values (and the two np.where reads)
        held += (_ELEMENT_MODULI_BYTES + 2 * _FLOAT_BYTES) * form.elements
    setup = ((points + 4 * edge_points) * _SAMPLER_SETUP_BYTES + 2 * edge_points * _SAMPLER_BYTES
             + (form.mesh_edges * _INTERFACE_SEARCH_BYTES if drive.values['moving_boundary'] else 0)
             + 2 * _COMPLEX_BYTES * window + _INT64_BYTES * form.elements
             + (_FLOAT_BYTES + _FLOAT32_BYTES) * form.elements)
    samples = drive.reads * optical * _COMPLEX_BYTES
    rows = form.rows * drive.parts * _COMPLEX_BYTES   # one position's stress rows (every part) per point
    edge = edge_points * (2 * drive.reads * optical + _EDGE_POSITION_VALUES) * _COMPLEX_BYTES
    element_points, _ = _rule_points()
    chunk = min(drive.elements, CHUNK_ELEMENTS) * element_points * _chunk_point_bytes(form)
    steps = (points * (2 * samples + _GATHER_BYTES + rows) + vector,
             points * (samples + rows + _STRESS_PEAK_VALUES[form.axisymmetric] * _COMPLEX_BYTES) + vector,
             points * (samples + rows + form.rows * _COMPLEX_BYTES) + chunk + 2 * vector,
             points * (samples + rows + _LINE_POINT_BYTES) + window * _LOCAL_WINDOW_BYTES + vector,
             points * (samples + rows) + vector + (window * optical * _COMPLEX_BYTES if drive.positions > 1 else 0))
    return int(held + max(setup, max(steps) + edge))


def _cell_bind_bytes(drive, form) -> int:
    """_bind_bytes of a tensor solid (BrillouinDrive._CellGeometry, SolidMechanicsPhysics._drive_inputs), in bind's
    order. Held through every position: the window copies of the solid's material arrays (float64), the solid and
    selection masks, the ElementModuli and its two E / nu reads (built per bind), the drive points (one per selected
    cell) with their materials, eps and law, the moving-boundary faces. On top the largest of: the first position's
    set-up (dA raster, face search over every interior face, the isotropic eps window, the cell laws, one property
    copy's float32 / float64 temporaries) and one position's steps (as _bind_bytes, the stress assembly and the face
    drive on the whole window: _CELL_DRIVE_VALUES / _CELL_FACE_DRIVE_VALUES complex per cell)."""
    points, faces, cells = drive.points, drive.edge_points, form.elements
    optical = drive.field_components
    window = drive.cells[0]
    vector = _COMPLEX_BYTES * form.vector_dofs
    laws = _INT64_BYTES if drive.values['reference'] != 'none' else 0
    held = ((form.material_arrays * _FLOAT_BYTES + 2 * _BOOL_BYTES + _ELEMENT_MODULI_BYTES + 2 * _FLOAT_BYTES) * cells
            + points * (_CELL_POINT_BYTES + _POINT_MATERIAL_BYTES + optical * _COMPLEX_BYTES + laws)
            + faces * _CELL_FACE_BYTES)
    setup = (_CELL_SETUP_BYTES * cells + (form.mesh_edges * _CELL_FACE_SEARCH_BYTES if drive.values['moving_boundary']
                                          else 0)
             + 2 * _COMPLEX_BYTES * window + _INT64_BYTES * cells + (_FLOAT_BYTES + _FLOAT32_BYTES) * cells)
    samples = drive.reads * optical * _COMPLEX_BYTES
    rows = form.rows * drive.parts * _COMPLEX_BYTES
    edge = (faces * (2 * drive.reads * optical + _EDGE_POSITION_VALUES) * _COMPLEX_BYTES
            + (_CELL_FACE_DRIVE_VALUES * _COMPLEX_BYTES * cells + vector if faces else 0))
    steps = (points * (samples + rows) + vector,
             points * (samples + rows + _STRESS_PEAK_VALUES[False] * _COMPLEX_BYTES) + vector,
             points * (samples + rows + form.rows * _COMPLEX_BYTES) + _CELL_DRIVE_VALUES * _COMPLEX_BYTES * cells
             + 2 * vector,
             points * (samples + rows + _LINE_POINT_BYTES) + window * _LOCAL_WINDOW_BYTES + vector,
             points * (samples + rows) + vector + (window * optical * _COMPLEX_BYTES if drive.positions > 1 else 0))
    return int(held + max(setup, max(steps) + edge))


def _reference_bytes(drive, n_frequencies) -> int:
    """Host bytes of one reference spectrum in result() (local_reference per material group over its bins,
    homogeneous_reference over the padded drive-box raster); 0 for reference 'none'."""
    reference = drive.values['reference']
    if reference == 'local':
        bins = drive.bins()
        return _LOCAL_CHUNK_ARRAYS * min(_CHUNK_VALUES, max(n_frequencies, 1) * bins) * _FLOAT_BYTES
    if reference == 'homogeneous':
        padded = _HOMOGENEOUS_PADDING * drive.cells[1]
        chunk = max(padded, min(_CHUNK_VALUES, max(n_frequencies, 1) * padded))
        return max(padded * _HOMOGENEOUS_SPECTRUM_BYTES,
                   padded * _HOMOGENEOUS_LOOP_BYTES + _HOMOGENEOUS_CHUNK_ARRAYS * chunk * _FLOAT_BYTES)
    return 0


def _kept_reference_item(drive, label):
    """(label, bytes) of the reference inputs the bound drive keeps until result(), or None (reference 'none')."""
    reference = drive.values['reference']
    if reference == 'local':
        bins = drive.bins()
        return (label + f'local reference lines (up to {bins} compressed |q| bins per position, 2 x float64)',
                drive.positions * bins * _LINE_BYTES)
    if reference == 'homogeneous':
        cells = drive.cells[1]
        return (label + 'homogeneous reference stress rasters (3 x complex128 per drive-box optical cell and position) '
                        'and their cells',
                drive.positions * 3 * cells * _COMPLEX_BYTES
                + cells * (_RASTER_CELL_BYTES + drive.field_components * _COMPLEX_BYTES))
    return None


def _frequency_bytes(columns, form, device, method) -> int:
    """Host bytes of one frequency's drive work beside the solve's own vectors: the readout W^H x (direct: conj(x) of
    a full solve, on a host factor a rows-only solve's x copy too; reduced: the once-per-model (W P)^T block and its
    conjugate) and, with several columns, one column solve (host factor: free rhs, solution, reduced vector, + the
    column-map products at azimuthal order 1; cuDSS: the host free rhs)."""
    vector = _COMPLEX_BYTES * form.unknowns
    if method != 'direct':
        return int(2 * columns * vector)
    readout = vector if device else 2 * vector
    if columns <= 1:
        return int(readout)
    column = vector if device else (3 + (2 if form.order == 1 else 0)) * vector
    return int(max(readout, column))


def brillouin_memory_items(context, settings_values, mesh, *, unknowns, vector_dofs, local, components, device,
                           moduli_kept) -> tuple:
    """(host items, run items, transient items) of the solid's brillouin_scattering drives (frequency_response,
    triangles; unknowns = the harmonic system's, mixed u-p included: W is [w; 0]; vector_dofs = the drive vector's u
    rows): per feature the drive columns W (one array: load and functional weights) and the kept reference inputs; one
    item, the largest of the never-overlapping transients (one bind: _bind_bytes; a frequency's readout and column
    solve: _frequency_bytes; a reference spectrum: _reference_bytes; several drives: their concatenation); device (cuDSS
    'linear' back end): the functional weights at the free dofs when a rows-only (snapshots not 'all') or column solve
    uploads them (run) and, with several columns, the previous column's solution beside the next solve (phase
    BRILLOUIN_COLUMN_PHASE). ([], [], []) without a drive."""
    counts = _drive_counts(context, settings_values, mesh)
    if not counts:
        return [], [], []
    form = _SolidForm(context, settings_values, mesh, unknowns=unknowns, vector_dofs=vector_dofs, local=local,
                      components=components, moduli_kept=moduli_kept)
    study = _study_values(context) or {}
    n_frequencies = len(study_frequencies(study)) if study else 1
    method = study.get('sweep_method', 'direct')
    vector = _COMPLEX_BYTES * int(unknowns)
    columns = sum(drive.columns for drive in counts)
    host, run, transient, temporaries = [], [], [], []
    for drive in counts:
        label = f"Brillouin '{drive.feature.name}': "
        host.append((label + f'drive columns W ({drive.columns} x complex128 over every unknown; also the load and the '
                             'functional weights)', drive.columns * vector))
        kept = _kept_reference_item(drive, label)
        if kept is not None:
            host.append(kept)
        temporaries += [_bind_bytes(drive, form), _reference_bytes(drive, n_frequencies)]
    if len(counts) > 1:   # bind_physics_drives concatenates the drives' columns (validate allows one drive per solid)
        temporaries.append(columns * vector)
    temporaries.append(_frequency_bytes(columns, form, device, method))
    host.append(('Brillouin drives: largest temporaries (one bind with its drive points, samplers and solid inputs; one '
                 "frequency's readout and column solve; one reference spectrum)", max(temporaries)))
    rows_only = study.get('snapshots', 'all') != 'all' and n_frequencies > 1
    if device and method == 'direct' and (columns > 1 or rows_only):
        run.append(('Brillouin drives: functional weights at the free dofs on the device (complex128 per column)',
                    columns * vector))
        if columns > 1:   # the solve's own vectors are the HarmonicSystem's cuDSS items (harmonic_pencil_items)
            transient.append((BRILLOUIN_COLUMN_PHASE, "Brillouin drives: the previous column's device solution "
                                                      'beside the next column solve (complex128)', vector))
    return host, run, transient


# host bytes per drive point of the one-off point set-up: element geometry at the rule points (Jacobian, inverse,
# determinant: 9 float64), the points / weights / elements and the sampler build
_POINT_SETUP_BYTES = 9 * _FLOAT_BYTES + _POINT_BYTES + _SAMPLER_SETUP_BYTES + _SAMPLER_BYTES


def brillouin_time_items(context, settings_values, mesh, *, unknowns, vector_dofs, local, components) -> tuple:
    """(TimeEstimate, extra solves on the 'linear' factor) of the solid's brillouin_scattering drives: once per bind
    the solid inputs, drive points, samplers and interface search; per scan position the samples, optical stress,
    element-chunk assembly (strain operator streams and the einsum flops), edge tractions and local lines; per solve
    the functional readout W^H x over every column (reduced sweeps: the once-per-model block); per extra column and
    frequency the host passes of solve_columns (counter column_solves; its factor solves are the returned count, priced
    by the 'linear' role; a tensor solid: by its private SuperLU); the reference spectra and the peak fits. Several
    columns solve sweep_method direct only (validate). A tensor solid: cell gathers, the window-sized cell / face
    drives (no element chunks, no einsum) and its face search."""
    counts = _drive_counts(context, settings_values, mesh)
    if not counts:
        return te.TimeEstimate(()), 0.0
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS   # numpy only at module level
    form = _SolidForm(context, settings_values, mesh, unknowns=unknowns, vector_dofs=vector_dofs, local=local,
                      components=components)
    sweep = frequency_sweep_counts(context)
    linear = (context.solvers or {}).get('linear')
    device = linear is not None and linear.values.get('back_end') in CUDSS_BACK_ENDS
    vector = float(_COMPLEX_BYTES * unknowns)
    drive_vector = float(_COMPLEX_BYTES * vector_dofs)
    frequencies = float(sweep['points'])
    columns = float(sum(drive.columns for drive in counts))
    element_points, _ = _rule_points()
    setup_bytes = form.material_arrays * form.elements * (_FLOAT32_BYTES + 2 * _FLOAT_BYTES)
    items = []
    for drive in counts:
        if form.tensor:   # dA raster, ElementModuli, points, face search
            setup_bytes += (2.0 * form.elements * (_CELL_SETUP_BYTES + _ELEMENT_MODULI_BYTES)
                            + 2.0 * drive.points * _CELL_POINT_BYTES
                            + (2.0 * form.mesh_edges * _CELL_FACE_SEARCH_BYTES if drive.values['moving_boundary']
                               else 0.0))
            continue
        # an edge point builds four side samplers (two eps diagonal ones, two field ones) beside its point set-up
        setup_bytes += (2.0 * (drive.points + drive.edge_points) * _POINT_SETUP_BYTES
                        + 2.0 * 3.0 * drive.edge_points * (_SAMPLER_SETUP_BYTES + _SAMPLER_BYTES)
                        + (2.0 * form.mesh_edges * _INTERFACE_SEARCH_BYTES if drive.values['moving_boundary'] else 0.0))
    items.append(te.TimeItem('Brillouin drives: solid inputs, drive points, samplers and interface search', 'setup',
                             1.0, 'exact', (te.HostStream(setup_bytes),
                                            te.PythonCalls(40 + 3 * form.material_arrays))))
    if sweep['method'] == 'direct':
        items.append(te.TimeItem('Brillouin drives: functional readout W^H x of every column', 'frequency',
                                 frequencies, 'exact', (te.HostStream((columns + 2.0) * vector),)))
    else:   # _reduced_functionals: (W P)^T, its conjugate and the (k, basis) block once per model
        items.append(te.TimeItem('Brillouin drives: reduced-model functional block', 'setup', 1.0, 'exact',
                                 (te.HostStream(4.0 * columns * vector),)))
    extra = frequencies * (columns - 1.0) if sweep['method'] == 'direct' else 0.0
    if extra:
        passes = _COLUMN_DEVICE_HOST_PASSES if device else _COLUMN_HOST_PASSES
        ops = (te.HostStream(passes * vector), te.PythonCalls(10))
        if device:   # the dot with the kept device weights
            ops += (te.Elementwise(2.0 * vector, _COMPLEX_BYTES, launches=2, flops=8.0 * vector / _COMPLEX_BYTES,
                                   flop_bytes=_FLOAT_BYTES),)
        items.append(te.TimeItem('Brillouin drives: column solves (free rhs, solution expansion, readout)', 'solve',
                                 extra, 'exact', ops, counter=('solvers', context.name, 'counters', 'column_solves')))
    for drive in counts:
        label = f"Brillouin '{drive.feature.name}': "
        positions = float(drive.positions)
        optical = drive.field_components
        stress = 2.0 * _STRESS_PEAK_VALUES[form.axisymmetric] * _COMPLEX_BYTES
        if form.tensor:   # cell gathers, the window-sized stress raster / cell drive and face drive (elementwise)
            samples = 2.0 * drive.reads * optical * 2.0 * _COMPLEX_BYTES
            assembly = 2.0 * drive.parts * 2.0 * form.rows * _COMPLEX_BYTES
            window_drive = 2.0 * drive.parts * form.elements * _COMPLEX_BYTES * (
                _CELL_DRIVE_VALUES + (_CELL_FACE_DRIVE_VALUES if drive.edge_points else 0))
            ops = (te.PythonCalls(_BIND_CALLS),)
        else:
            chunks = max(math.ceil(drive.elements / CHUNK_ELEMENTS), 1)
            samples = 2.0 * drive.reads * optical * (_GATHER_BYTES + _COMPLEX_BYTES)
            assembly = 2.0 * drive.parts * (2.0 * form.rows * _COMPLEX_BYTES + _chunk_point_bytes(form))
            window_drive = 0.0
            # the complex x real einsum of the element chunks
            flops = 4.0 * drive.points * drive.parts * form.rows * form.components * form.local
            ops = (te.DenseFlops(flops, host=True), te.PythonCalls(_BIND_CALLS + 10 * chunks))
        position = (drive.points * (samples + stress + assembly + 2.0 * _LINE_POINT_BYTES)
                    + drive.edge_points * (samples + 2.0 * _EDGE_POSITION_VALUES * _COMPLEX_BYTES)
                    + 2.0 * drive.parts * drive_vector + 2.0 * drive.cells[0] * _LOCAL_WINDOW_BYTES + window_drive)
        items.append(te.TimeItem(label + 'drive assembly (samples, optical stress, B^T sigma, edge tractions, local q)',
                                 'position', positions, 'exact', (te.HostStream(position),) + ops))
        reference = drive.values['reference']
        if reference == 'local':   # per position and frequency the line of every kept bin
            items.append(te.TimeItem(label + 'local reference spectra', 'position', positions, 'exact',
                                     (te.HostStream(2.0 * frequencies * drive.bins() * _LOCAL_CHUNK_ARRAYS
                                                    * _FLOAT_BYTES), te.PythonCalls(30))))
        elif reference == 'homogeneous':   # 3 padded FFTs, then per frequency the two terms over every wavevector
            padded = float(_HOMOGENEOUS_PADDING * drive.cells[1])
            fft = 3.0 * padded * _COMPLEX_BYTES * 2.0 * max(math.log2(max(padded, 2.0)), 1.0)
            items.append(te.TimeItem(label + 'homogeneous reference spectra', 'position', positions, 'exact',
                                     (te.HostStream(fft + 2.0 * padded * _HOMOGENEOUS_SPECTRUM_BYTES
                                                    + 2.0 * frequencies * padded * _HOMOGENEOUS_CHUNK_ARRAYS
                                                    * _FLOAT_BYTES), te.PythonCalls(40))))
        if drive.values['fit_peak']:
            fits = positions * (1.0 if reference == 'none' else 2.0)
            items.append(te.TimeItem(label + 'damped-oscillator peak fits', 'fit', fits, 'exact',
                                     (te.PythonCalls(_DHO_FIT_CALLS_GUESS),)))
    return te.TimeEstimate(tuple(items)), extra

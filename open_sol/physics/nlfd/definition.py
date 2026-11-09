"""Model definition of the NLFD-chi physics type 'em_fdfd_nonlinear' (registry-discovered): harmonic channels, chi(2)
/ chi(3) media, the em_fdfd_ez / _hz sources and monitors re-registered per channel, validation, memory estimate,
register."""
# Channel features wrap the FeatureTypes of EM_FDFD_EZ / EM_FDFD_HZ (apply, validate, result) on solver.channel(name):
# their params are the union of both polarisations plus 'channel', and the FDFD callbacks see a physics view holding
# only that channel's features and a feature view holding only their own params. Linear solver: the 'linear' role
# (sparse_direct, one runtime shared by every channel, each channel keeping its own factor) with the FDFD checks
# (definition_fdfd.direct_solver_issues).
import math
from dataclasses import dataclass, replace

from ...core import time_estimate as te
from ...core.constants import C0, ETA0
from ...core.memory_estimate import (BOOL_BYTES, COMPLEX128_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate,
                                     combine)
from ...model.definition_helpers import DIRECT_LINEAR_ROLE, optimization_evaluations, resolved_back_end
from ...model.expressions import is_identifier
from ...model.geometry import bounding_box, expand_entities
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path, resolve_values
from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS
from ...solvers.sparse_direct import RECYCLE_MAX_ITERATIONS
from ..em.definition_common import (EM_FAMILY, em_material_params, make_validate_physics,
                                    quasi_static_domain_requirements, source_slot_docs, write_dispersive_materials)
from ..em.definition_bloch import is_fdfd_port_feature
from ..em.definition_conducting_sheet import sheet_conductivity_from_values
from ..em.definition_design import design_region_feature
from ..em.definition_fdfd import ASSEMBLY_BYTES_PER_ENTRY, EM_FDFD_EZ, EM_FDFD_HZ, direct_solver_issues
from ..em.definition_feature_validation import bloch_period_line_issues, conducting_sheet_line_issues
from ..em.materials import EM_MATERIAL_SCHEMAS
from ..em.memory_estimates import fdfd_memory_estimate
from ..em.placement import PERMITTIVITY_SLOTS
from .adjoint import ADJOINT_MAX_PRODUCTS
from .definition_sheets import SHEET_NONLINEARITY, SURFACE_CHI2, sheet_mixing_tensors
from .iteration import GMRES_MAX_ITERATIONS, GMRES_RESTART, ITERATION_METHODS, NEWTON_BUFFERS
from .media import CarrierModel
from .mixing import (POLARIZATIONS, ChannelSpec, MediumTensor, build_mixing_table, describe_mixing,
                     inexact_term_issues, resolve_channels, source_text)
from .renderer import NONLINEAR_FIELD_INFOS
from .selftests import NLFD_TEMPLATES
from .selftests_adjoint import ADJOINT_TEMPLATES
from .selftests_sheets import SHEET_TEMPLATES
from .solver import COUPLING_MODES, NonlinearHarmonicPhysics
from .tensors import (CRYSTAL_CLASSES, CUT_ROTATIONS, chi2_from_voigt, chi3_cubic, chi3_from_n2_beta, chi3_isotropic,
                      euler_rotation, kleinman_asymmetry, n2_beta_from_chi3, rotate_tensor, voigt_matrix)

TYPE_NAME = 'em_fdfd_nonlinear'
NONLINEAR_STUDY = 'nonlinear_frequency_domain'   # = model.study_nonlinear.STUDY_NAME
_CHANNEL_FEATURE = 'harmonic_channel'
_CHI2_FEATURE = 'chi2_medium'
_CHI3_FEATURE = 'chi3_medium'
_MEDIUM_ORDERS = {_CHI2_FEATURE: 2, _CHI3_FEATURE: 3}
_PORT_FEATURES = ('mode_source', 'mode_monitor')   # their slab modes miss the Kerr permittivity
_ANDERSON_BUFFERS = 5   # stacked store-size vectors besides the 2 x depth history: x, f, previous x / f, the mix
_KERR_ARRAYS = {'ez': 2, 'hz': 8}   # full-grid complex128 arrays of one KerrOperatorTerm update (upper bound)
_ADJOINT_BUFFERS = 5   # store-size vectors of adjoint.depleted_cotangents besides the GMRES basis: rhs, solution,
#                        the scaled product copy, its pack and the difference
_SHEET_FEATURE = 'conducting_sheet'   # acts on every channel, like pec_domain (a sheet exists at every frequency)
_FDFD_TYPES = {'ez': EM_FDFD_EZ, 'hz': EM_FDFD_HZ}
# em_fdfd_ez / _hz features copied per channel (resonance_table / effective_index / phase change are not offered: an
# unknown-feature error; design_region acts on the shared arrays, not per channel)
_CHANNEL_FEATURE_NAMES = ('point_source', 'line_source', 'gaussian_beam', 'plane_wave_tfsf', 'mode_source',
                          'field_probe', 'flux_monitor', 'mode_monitor', 'far_field', 'adjoint_objective')
OPTIMIZATION_STUDY = 'optimization'   # = model.study_optimization.STUDY_NAME
_SLOT_ARRAYS = {**PERMITTIVITY_SLOTS['ez'], **PERMITTIVITY_SLOTS['hz']}
_LOW_ORDER_INFO_TERMS = 3   # near misses / off-set products listed by validation
_DENSITY_TRANSIENT_ARRAYS = 3   # complex128 full-grid temporaries of one density evaluation (box products, magnitude)
_ABSOLUTE_AMPLITUDE_TEXT = ('Amplitudes are absolute: the nonlinear response scales with them (P ~ E^n), unlike a '
                            'linear FDFD result.')
_UNDEPLETED_WARN_RATIO = 0.3   # generated / weakest source field estimate above which undepleted is doubtful
_AMPLITUDE_PARAMS = {'line_source': 'amplitude', 'gaussian_beam': 'amplitude', 'plane_wave_tfsf': 'amplitude'}
_POWER_PARAMS = {'mode_source': 'power'}   # W/m: peak |E| <= sqrt(4 eta0 P / lambda) (mode width >= lambda / (2 n))


# ---- channels (CPU; validation, memory estimate and create share them) ----
def _enabled_features(physics_spec, feature_type):
    return [feature for feature in physics_spec.features if feature.enabled and feature.type == feature_type]


def _channel_specs(physics_spec, parameters):
    """mixing.ChannelSpec of every enabled harmonic_channel feature (name = feature name); ValueError on bad values."""
    specs = []
    for feature in _enabled_features(physics_spec, _CHANNEL_FEATURE):
        values = resolve_values(HARMONIC_CHANNEL_PARAMS, feature.params if isinstance(feature.params, dict) else {},
                                parameters)
        by_wavelength = values['definition'] == 'wavelength'
        specs.append(ChannelSpec(feature.name, values['polarization'],
                                 wavelength=values['wavelength'] if by_wavelength else None,
                                 combination=None if by_wavelength else values['combination'],
                                 bloch_k=(float(values['bloch_kx']), float(values['bloch_ky'])) if by_wavelength
                                 else (0.0, 0.0)))
    return specs


def _model_channels(physics_spec, parameters):
    """(resolved channels tuple, None) or (None, error text) of the physics' harmonic_channel features."""
    try:
        channels, _ = resolve_channels(_channel_specs(physics_spec, parameters))
    except ValueError as exc:
        return None, str(exc)
    return channels, None


def _channel_wavelengths(spec, physics_spec, settings_values, parameters, registry):
    """Vacuum wavelengths [m] of the channels (the solved_wavelengths hook of make_validate_physics), None on errors."""
    channels, _ = _model_channels(physics_spec, parameters)
    return [2.0 * math.pi * C0 / channel.omega for channel in channels] if channels else None


def _shortest_channel_wavelength(spec, physics_spec, settings_values, parameters, registry):
    """PhysicsType.shortest_wavelength: the shortest channel wavelength (auto mesh), None when channels fail."""
    wavelengths = _channel_wavelengths(spec, physics_spec, settings_values, parameters, registry)
    return min(wavelengths) if wavelengths else None


def _channel_names(physics_spec):
    """Channel names of the enabled harmonic_channel features without resolving them ('both' -> _ez, _hz)."""
    names = []
    for feature in _enabled_features(physics_spec, _CHANNEL_FEATURE):
        params = feature.params if isinstance(feature.params, dict) else {}
        if params.get('polarization') == 'both':
            names.extend(f'{feature.name}_{polarization}' for polarization in ('ez', 'hz'))
        else:
            names.append(feature.name)
    return names


# ---- chi(2) media ----
def chi2_tensor(values):
    """(3, 3, 3) chi(2) [m/V] in model axes of resolved chi2_medium values (crystal class d_il, cut / Euler rotation)."""
    crystal_class = values['crystal_class']
    coefficients = {name: float(values[name]) for name in CRYSTAL_CLASSES[crystal_class]}
    return rotate_tensor(chi2_from_voigt(voigt_matrix(crystal_class, coefficients)), _rotation(values))


def _medium_features(physics_spec):
    """Enabled chi2_medium and chi3_medium features."""
    return [feature for feature in physics_spec.features if feature.enabled and feature.type in _MEDIUM_ORDERS]


def _raw_flag(feature, name):
    """A bool param of a feature as written (False when absent or not a dict)."""
    return bool(feature.params.get(name)) if isinstance(feature.params, dict) else False


def _media_tensors(vctx_values, physics_spec, channels):
    """[MediumTensor] of the enabled chi2 / chi3 media and nonlinear sheets whose values resolve (vctx_values(params,
    raw) -> dict | None; chi3 n2 / beta at n0 = background_index or 1 (a direct chi3 unscaled, as the build), the
    reference omega from `channels` when not given)."""
    media = []
    for feature in _medium_features(physics_spec):
        raw = feature.params if isinstance(feature.params, dict) else {}
        if feature.type == _CHI2_FEATURE:
            values = vctx_values(CHI2_MEDIUM_PARAMS, raw)
            if values is not None:
                media.append(MediumTensor.from_tensor(feature.name, 2, chi2_tensor(values)))
            continue
        values = vctx_values(CHI3_MEDIUM_PARAMS, raw)
        if values is not None and channels:
            n0 = (values['background_index'] if values['strength'] == 'n2_beta'
                  and values['background_index'] is not None else 1.0)
            media.append(MediumTensor.from_tensor(
                feature.name, 3, chi3_tensor(values, medium_reference_omega(values, channels)) * n0 ** 2,
                chi3_process_scales(values)))
    return media + sheet_mixing_tensors(vctx_values, physics_spec)   # the solver's order: media, then sheets


# ---- chi(3) media ----
def medium_reference_omega(values, channels):
    """omega [rad/s] of a medium's reference: chi(3) the n2 / beta conversion, the Miller reference (SPM) and the
    carrier coefficients; chi(2) the pump of the SHG process its d coefficients hold for (Miller's rule):
    reference_wavelength, else the first input channel."""
    if values.get('reference_wavelength') is not None:
        return 2.0 * math.pi * C0 / values['reference_wavelength']
    return next(channel.omega for channel in channels if channel.rank == 0)


def chi3_process_scales(values):
    """((process, factor), ...) of the per-process chi3 factors (THG / TFG thg_scale, FWM fwm_scale)."""
    return (('THG', values['thg_scale']), ('TFG', values['thg_scale']), ('FWM', values['fwm_scale']))


def chi3_xxxx(values, reference_omega, background_index=1.0):
    """Complex chi_xxxx [m^2/V^2] of resolved chi3_medium values (n2 / beta at n0 = background_index, or direct)."""
    if values['strength'] == 'chi3':
        return complex(values['chi3_real'], values['chi3_imag'])
    return chi3_from_n2_beta(values['n2'], values['tpa_beta'], background_index, reference_omega)


def chi3_tensor(values, reference_omega):
    """(3, 3, 3, 3) chi(3) [m^2/V^2] in model axes (n2 / beta with n0 = 1: the solver scales by n0^2 per cell)."""
    chi_xxxx = chi3_xxxx(values, reference_omega)
    if values['tensor'] == 'isotropic':
        return chi3_isotropic(chi_xxxx)
    return rotate_tensor(chi3_cubic(chi_xxxx, values['anisotropy']), _rotation(values))


def _rotation(values):
    if values['orientation'] == 'euler':
        return euler_rotation(values['euler_alpha'], values['euler_beta'], values['euler_gamma'])
    return CUT_ROTATIONS[values['orientation']]


def _apply_chi3_medium(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    reference_omega = medium_reference_omega(values, solver.channels)
    background_index = values['background_index']

    def index_squared():
        return (background_index ** 2 if background_index is not None
                else solver.background_index_squared(reference_omega))

    scale = index_squared() if values['strength'] == 'n2_beta' else None
    carriers = carrier_index_squared = None
    if values['free_carriers'] and values['carrier_lifetime'] > 0.0:
        carriers = CarrierModel(lifetime=float(values['carrier_lifetime']),
                                fcd_coefficient=float(values['fcd_coefficient']),
                                fca_cross_section=float(values['fca_cross_section']), reference_omega=reference_omega)
        carrier_index_squared = scale if scale is not None else index_squared()
    solver.add_bulk_medium(feature_spec.name, 3, chi3_tensor(values, reference_omega), mask, cell_scale=scale,
                           process_scales=chi3_process_scales(values),
                           miller_reference_omega=reference_omega if values['miller_scaling'] else None,
                           carriers=carriers, carrier_index_squared=carrier_index_squared)
    return None


def _validate_chi3_medium(vctx, physics_spec, feature_spec, values):
    """Warning for chi3 = 0; info with chi_xxxx and its n2 / beta (or the per-cell n0 rule) at the reference, the
    per-process / Miller factors and the free carriers (warnings: no lifetime, no TPA)."""
    path = feature_path(physics_spec, feature_spec)
    channels, _ = _model_channels(physics_spec, vctx.parameters)
    if not channels:
        return []
    omega = medium_reference_omega(values, channels)
    wavelength_nm = 2.0 * math.pi * C0 / omega * 1e9
    n0 = values['background_index']
    chi_xxxx = chi3_xxxx(values, omega, 1.0 if n0 is None else n0)
    if chi_xxxx == 0:
        return [Issue('warning', path, 'chi3 is 0 (n2 = tpa_beta = 0, or chi3_real = chi3_imag = 0): the medium '
                                       'couples nothing')]
    if values['strength'] == 'chi3':
        text = f'chi_xxxx = {chi_xxxx.real:.4g} {"-" if chi_xxxx.imag < 0 else "+"} {abs(chi_xxxx.imag):.4g}j m^2/V^2'
        if n0 is not None:
            n2, beta = n2_beta_from_chi3(chi_xxxx, n0, omega)
            text += f' (n2 = {n2:.4g} m^2/W, tpa_beta = {beta:.4g} m/W at n0 = {n0:g}, {wavelength_nm:.4g} nm)'
    elif n0 is None:
        text = (f'chi_xxxx = n0^2 x {chi_xxxx.real:.4g} {"-" if chi_xxxx.imag < 0 else "+"} {abs(chi_xxxx.imag):.4g}j '
                f'm^2/V^2 with n0 = Re sqrt(eps_r) of each cell at {wavelength_nm:.4g} nm (evaluated at build)')
    else:
        text = (f'chi_xxxx = {chi_xxxx.real:.4g} {"-" if chi_xxxx.imag < 0 else "+"} {abs(chi_xxxx.imag):.4g}j '
                f'm^2/V^2 (n0 = {n0:g}, {wavelength_nm:.4g} nm)')
    scales = {'THG / TFG': values['thg_scale'], 'FWM': values['fwm_scale']}
    scaled = {process: scale for process, scale in scales.items() if scale != 1.0}
    process_text = ('one chi3 for every process (THG, SPM, FWM)' if not scaled else
                    'per-process chi3: ' + ', '.join(f'{process} x {scale:g}' for process, scale in scaled.items())
                    + ' (power_exchange_imbalance no longer ~0)')
    if values['miller_scaling']:
        process_text += (f"; Miller's rule relative to SPM at {wavelength_nm:.4g} nm (chi1 per channel evaluated at "
                         'build, the run log lists the scaled factors)')
    issues = [Issue('info', path, f'{text}; {process_text}')]
    if chi_xxxx.imag != 0:
        issues.append(Issue('info', path, 'Im chi3 < 0 (two-photon absorption): pump_depleted SPM / XPM absorb '
                                          '(scalar nonlinear_absorption; q_abs of the channels with the picard '
                                          'iteration), and the power exchange identity is no longer 0'))
    if values['free_carriers']:
        if values['carrier_lifetime'] <= 0.0:
            issues.append(Issue('warning', path, 'free_carriers is on but carrier_lifetime is 0: no carriers'))
        elif chi_xxxx.imag == 0:
            issues.append(Issue('warning', path, 'free_carriers needs two-photon absorption (tpa_beta > 0 or '
                                                 'chi3_imag < 0): without it no carriers are generated'))
        else:
            issues.append(Issue('info', path, f"free carriers: lifetime {values['carrier_lifetime']:.3g} s, dn = "
                                              f"-{values['fcd_coefficient']:.3g} N, alpha = "
                                              f"{values['fca_cross_section']:.3g} N at {wavelength_nm:.4g} nm, "
                                              'x (lambda / reference)^2 per channel; pump_depleted only'))
    return issues


_CARRIER_BOX_BYTES = FLOAT64_BYTES + BOOL_BYTES + 6 * COMPLEX128_BYTES   # N, G > 0, Kerr densities / temporaries


def _chi3_memory_estimate(values, n_selected_cells, mesh):
    """float32 cell factor (and carrier n0) on the selection's bounding box; mask, n0^2 (complex eps) temporaries;
    free carriers: N, its mask and the per-target Kerr densities on the box (run items)."""
    box_cells, n_cells = int(n_selected_cells), int(mesh.n_cells)
    carriers = bool(values.get('free_carriers'))
    items = [('chi3 cell factor (float32, box)', FLOAT32_BYTES * box_cells)]
    if carriers:
        items.append(('free-carrier n0 (float32, box)', FLOAT32_BYTES * box_cells))
    return MemoryEstimate.from_items(items, build_extra=(BOOL_BYTES + FLOAT32_BYTES + 2 * COMPLEX128_BYTES) * n_cells
                                     + 12 * box_cells,
                                     run_items=[('free carriers (N, generation mask, Kerr densities on the box)',
                                                 _CARRIER_BOX_BYTES * box_cells if carriers else 0)])


def _apply_chi2_medium(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    poling = ctx.eval_field(values['poling'], mask=mask)
    if not hasattr(poling, 'shape'):   # a constant sign: unpoled, or inverted everywhere
        poling = None if poling >= 0.0 else mask * -1.0   # -1 on the selection (outside: masked off)
    miller = medium_reference_omega(values, solver.channels) if values['miller_scaling'] else None
    solver.add_bulk_medium(feature_spec.name, 2, chi2_tensor(values), mask, poling, miller_reference_omega=miller)
    return None


def _validate_chi2_medium(vctx, physics_spec, feature_spec, values):
    """Warning for an all-zero tensor; info when the coefficients break Kleinman symmetry (power identity inexact) and
    on Miller's rule."""
    path = feature_path(physics_spec, feature_spec)
    tensor = chi2_tensor(values)
    if not abs(tensor).max() > 0.0:
        return [Issue('warning', path, f"every d coefficient of crystal class {values['crystal_class']} is 0: the "
                                       'medium couples nothing')]
    issues = []
    asymmetry = kleinman_asymmetry(tensor)
    if asymmetry > 1e-12:
        issues.append(Issue('info', path, f'the d coefficients break Kleinman (full permutation) symmetry by '
                                          f'{asymmetry:.3g} relative: the power exchange identity (scalar '
                                          'power_exchange_imbalance) is then inexact, which is physical (dispersive '
                                          'chi2), not a solver error'))
    channels, _ = _model_channels(physics_spec, vctx.parameters)
    if values['miller_scaling'] and channels:
        wavelength_nm = 2.0 * math.pi * C0 / medium_reference_omega(values, channels) * 1e9
        issues.append(Issue('info', path, f"Miller's rule: d holds for SHG of {wavelength_nm:.4g} nm; every process is "
                                          'scaled by chi1(w3) chi1(w1) chi1(w2) / chi1(2 w_r) chi1(w_r)^2 (chi1 = Re '
                                          'eps - 1 of the medium cells, evaluated at build; the run log lists the '
                                          'factors; power_exchange_imbalance no longer ~0)'))
    return issues


def _chi2_memory_estimate(values, n_selected_cells, mesh):
    """int8 cell sign on the selection's bounding box; the fresh domain mask, poling values and sign temporaries."""
    box_cells, n_cells = int(n_selected_cells), int(mesh.n_cells)
    return MemoryEstimate.from_items((('chi2 poling sign (int8, box)', box_cells),),
                                     build_extra=(BOOL_BYTES + FLOAT32_BYTES) * n_cells + 10 * box_cells)


# ---- channel features (FDFD feature per channel) ----
@dataclass(frozen=True, eq=False)
class ChannelFeature:
    """Runtime object of a channel feature: the channel name and the FDFD feature object on that channel."""
    channel: str
    target: object


def _fdfd_feature(polarization, name):
    return _FDFD_TYPES[polarization].feature_type(name)


def _fdfd_param_names(polarization, name):
    return frozenset(param.name for param in _fdfd_feature(polarization, name).params)


def _other_polarization(polarization):
    return 'hz' if polarization == 'ez' else 'ez'


def _merged_text(ez_text, hz_text, separator=' '):
    return ez_text if ez_text == hz_text else f'E_z channels: {ez_text}{separator}H_z channels: {hz_text}'


def _merged_param(ez_param, hz_param):
    """One Param for both polarisations: help / tip per polarisation; differing choices united, default None = the
    channel's out-of-plane field (field_probe component)."""
    if ez_param == hz_param:
        return ez_param
    merged = replace(ez_param, help=_merged_text(ez_param.help, hz_param.help),
                     tip=_merged_text(ez_param.tip, hz_param.tip, '\n\n'))
    if ez_param.choices != hz_param.choices:
        merged = replace(merged, choices=tuple(dict.fromkeys(ez_param.choices + hz_param.choices)), default=None,
                         optional=True, choice_labels=tuple(dict.fromkeys(ez_param.choice_labels
                                                                          + hz_param.choice_labels)),
                         help=f"Empty (None, the default) = the channel's out-of-plane field ({ez_param.default} or "
                              f'{hz_param.default}). {merged.help}')
    return merged


_CHANNEL_PARAM = Param(
    'channel', 'str', '', label='Channel',
    help='Name of the harmonic_channel (its feature name) this feature acts on; a polarization both channel splits '
         'into <name>_ez and <name>_hz: name one of those. The channel polarisation picks the E_z (em_fdfd_ez) or H_z '
         '(em_fdfd_hz) form of the feature; params of the other form are ignored (validation warns).',
    tip='Harmonic channel (wavelength) this source or monitor belongs to: the name of a harmonic channel.\n\n'
        "A channel with both polarisations is split into '<name>_ez' and '<name>_hz'; pick one of those.")


def _channel_feature_params(name):
    """'channel' + the union of the em_fdfd_ez / _hz params of feature `name` (shared ones merged)."""
    ez_params = {param.name: param for param in _fdfd_feature('ez', name).params}
    hz_params = {param.name: param for param in _fdfd_feature('hz', name).params}
    merged = []
    for param_name in dict.fromkeys((*ez_params, *hz_params)):
        ez_param, hz_param = ez_params.get(param_name), hz_params.get(param_name)
        if hz_param is None:
            merged.append(replace(ez_param, help=f'E_z channels only: {ez_param.help}'))
        elif ez_param is None:
            merged.append(replace(hz_param, help=f'H_z channels only: {hz_param.help}'))
        else:
            merged.append(_merged_param(ez_param, hz_param))
    return (_CHANNEL_PARAM,) + tuple(merged)


def _channel_values(values, polarization, name):
    """Resolved values restricted to the FDFD params of `polarization`, field_probe component None -> out of plane."""
    own = {key: value for key, value in values.items() if key in _fdfd_param_names(polarization, name)}
    if 'component' in own and own['component'] is None:
        own['component'] = polarization
    return own


def _channel_feature_spec(feature_spec, polarization, name):
    """Feature spec view with only the FDFD params of `polarization` (FDFD callbacks resolve their own params)."""
    params = feature_spec.params if isinstance(feature_spec.params, dict) else {}
    names = _fdfd_param_names(polarization, name)
    return replace(feature_spec, params={key: value for key, value in params.items() if key in names})


def _channel_physics_view(physics_spec, channel_name):
    """Physics spec view holding the non-channel features and the channel features of `channel_name` only (TFSF /
    mode-source interface checks of one channel)."""
    def on_channel(feature):
        params = feature.params if isinstance(feature.params, dict) else {}
        return feature.type not in _CHANNEL_FEATURE_NAMES or params.get('channel') == channel_name
    return replace(physics_spec, features=[feature for feature in physics_spec.features if on_channel(feature)])


def _channel_polarization(solver, channel_name):
    return solver.channels[solver.channel_index(channel_name)].polarization


def _make_channel_apply(name):
    def apply(ctx, solver, feature_spec, values, selection):
        channel = values['channel']
        polarization = _channel_polarization(solver, channel)
        fdfd_apply = _fdfd_feature(polarization, name).apply
        target = fdfd_apply(ctx, solver.channel(channel), _channel_feature_spec(feature_spec, polarization, name),
                            _channel_values(values, polarization, name), selection)
        return ChannelFeature(channel, target)
    return apply


def _make_channel_result(name):
    def result(session, solver, feature_object, feature_spec, options):
        polarization = _channel_polarization(solver, feature_object.channel)
        fdfd_result = _fdfd_feature(polarization, name).result
        return fdfd_result(session, solver.channel(feature_object.channel),
                           feature_object.target, _channel_feature_spec(feature_spec, polarization, name), options)
    return result


def _make_channel_validate(name):
    def validate(vctx, physics_spec, feature_spec, values):
        """Channel exists; params of the other polarisation warned; the FDFD validate on the channel's view."""
        path = feature_path(physics_spec, feature_spec)
        channels, _ = _model_channels(physics_spec, vctx.parameters)
        if channels is None:
            return []   # reported at the physics
        by_name = {channel.name: channel for channel in channels}
        channel = by_name.get(values['channel'])
        if channel is None:
            return [Issue('error', path, f"channel {values['channel']!r} is not a harmonic channel of "
                                         f"'{physics_spec.name}'; channels: {', '.join(by_name)} (a polarization both "
                                         'channel is <name>_ez / <name>_hz)')]
        polarization = channel.polarization
        issues = []
        raw = feature_spec.params if isinstance(feature_spec.params, dict) else {}
        own_names = _fdfd_param_names(polarization, name)
        ignored = sorted(key for key in raw
                         if key in _fdfd_param_names(_other_polarization(polarization), name) and key not in own_names)
        if ignored:
            issues.append(Issue('warning', path, f"{', '.join(ignored)} belong to the {_other_polarization(polarization)} "
                                                 f"form of {name}; channel {channel.name} is {polarization}, so they "
                                                 'are ignored'))
        own_values = _channel_values(values, polarization, name)
        component_param = next((param for param in _fdfd_feature(polarization, name).params
                                if param.name == 'component'), None)
        if component_param is not None and own_values['component'] not in component_param.choices:
            issues.append(Issue('error', path, f"component {own_values['component']!r} is not a field of the "
                                               f"{polarization} channel {channel.name}; valid: "
                                               f"{', '.join(component_param.choices)}"))
            return issues
        fdfd_validate = _fdfd_feature(polarization, name).validate
        if fdfd_validate is not None:
            issues.extend(fdfd_validate(vctx, _channel_physics_view(physics_spec, channel.name),
                                        _channel_feature_spec(feature_spec, polarization, name), own_values))
        return issues
    return validate


def _channel_feature_type(name):
    """FeatureType `name` acting on one channel: the em_fdfd_ez / _hz feature by the channel's polarisation."""
    ez_type, hz_type = _fdfd_feature('ez', name), _fdfd_feature('hz', name)
    description = (f"The em_fdfd_ez / em_fdfd_hz feature {name} on one harmonic channel (setting channel): its "
                   'polarisation picks the E_z or H_z form, the channel frequency is the solve frequency. '
                   + _merged_text(ez_type.description, hz_type.description))
    if ez_type.category == 'source':
        description = f'{description} {_ABSOLUTE_AMPLITUDE_TEXT}'
    adjoint = name == 'adjoint_objective'
    if adjoint:
        description = f'{description} {_CASCADE_ADJOINT_TEXT}'
    return FeatureType(name, ez_type.label, ez_type.category, ez_type.selection_kind,
                       params=_channel_feature_params(name), object_kinds=ez_type.object_kinds,
                       max_entities=ez_type.max_entities, requires_uniform_mesh=ez_type.requires_uniform_mesh,
                       result_kind=ez_type.result_kind, description=description,
                       summary=f'{ez_type.summary or ez_type.description} Acts on one harmonic channel.',
                       apply=_make_channel_apply(name),
                       result=_make_channel_result(name) if ez_type.result is not None else None,
                       validate=_make_channel_validate(name),
                       memory_estimate=None)   # adjoint_objective: the physics estimator (one cascade adjoint)


_CASCADE_ADJOINT_TEXT = (
    'In em_fdfd_nonlinear the output field eps_sensitivity is dJ/d eps of J summed over the objectives of every '
    'channel through the coupled channels (undepleted: one transposed solve per channel in reverse solve order; '
    'pump_depleted: the transposed fixed point (I - T) nu = T(c) by real GMRES to the physics tolerance, each '
    'product one transposed solve per channel, then one gradient sweep; the nonlinear sources back-propagated exactly, '
    'back-action and SPM / XPM included, the surface_chi2 / sheet_nonlinearity sites and the free carriers (newton) '
    'too; an isotropic change: the eps_r, eps_xx and eps_yy terms added); pump_depleted picard with a chi3 medium '
    '(lagged Kerr / carrier operator: use newton) and an em_fdfd_hz eps_xy are not differentiated (error when the '
    'field is read with a weighted objective; validation error in an optimization study); the chi2 / chi3 / sheet '
    'tensors, poling, the sheet sites and the chi3 n0^2 / Miller factors are held fixed (free carriers: where '
    'generation is > 0). The pump_depleted gradient is that of the converged fixed point: it is as accurate as the '
    'tolerance.')


_CASCADE_ADJOINT_TEMPORARIES = 16   # complex128 full grids of cascade_sensitivity (adjoint source, adjoint, face terms,
                                    # P cotangents, the store cotangents of two hz channels; approximate)


def _apply_pec(ctx, solver, feature_spec, values, selection):
    solver.add_pec_region(ctx.domain_mask(selection.entities))
    return None


def _apply_conducting_sheet(ctx, solver, feature_spec, values, selection):
    """The em_fdfd_ez / _hz conducting_sheet on every channel (sigma_s evaluated at each channel's omega)."""
    for physics in solver.channel_physics:
        _fdfd_feature(physics.polarization, _SHEET_FEATURE).apply(ctx, physics, feature_spec, values, selection)
    return None


def _validate_conducting_sheet(vctx, physics_spec, feature_spec, values):
    """The FDFD line placement checks of every channel polarisation present (deduplicated) and an info with sigma_s
    and the vacuum normal-incidence transmission per channel."""
    channels, _ = _model_channels(physics_spec, vctx.parameters)
    channels = channels or ()
    issues = {}
    for polarization in dict.fromkeys(channel.polarization for channel in channels):
        for issue in conducting_sheet_line_issues(vctx, physics_spec, feature_spec, polarization):
            issues.setdefault((issue.level, issue.message), issue)
    issues = list(issues.values())
    path = feature_path(physics_spec, feature_spec)
    try:
        conductivity = sheet_conductivity_from_values(values)
    except (TypeError, KeyError):
        return issues   # reported by the param checks
    except ValueError as error:
        return issues + [Issue('error', path, str(error))]
    parts = []
    for channel in channels:
        sigma = conductivity(channel.omega)
        transmission = 1.0 / abs(1.0 + float(ETA0) * sigma / 2.0) ** 2
        parts.append(f'channel {channel.name}: sigma_s = {sigma.real:.4g} {"-" if sigma.imag < 0 else "+"} '
                     f'{abs(sigma.imag):.4g}j S (vacuum normal-incidence transmission {transmission:.4g})')
    return issues + ([Issue('info', path, '; '.join(parts))] if parts else [])


# ---- feature params and types ----
HARMONIC_CHANNEL_PARAMS = (
    Param('definition', 'choice', 'wavelength', label='Defined by', choices=('wavelength', 'combination'),
          choice_labels=(('wavelength', 'Wavelength (input)'), ('combination', 'Combination of channels')),
          help='wavelength: an input channel at a given vacuum wavelength (rank 0). combination: a generated channel '
               'whose frequency is an integer combination of other channels (rank 1 + the highest rank it names), '
               'so mixing products land on it exactly.',
          tip='How the frequency of this channel is set: by its own wavelength (a pump or signal), or as a '
              'combination of other channels (a harmonic or mixing product).'),
    Param('wavelength', 'float', 1.55e-6, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          visible_if=(('definition', ('wavelength',)),),
          help='Vacuum wavelength of an input channel [m] (omega = 2 pi C0 / wavelength); a parameter sweep over it '
               'gives spectra (e.g. SHG versus pump wavelength). Typical 0.8e-6 to 2e-6 m.',
          tip='Vacuum wavelength of this input channel, in metres: 1.55e-6 for 1550 nm.'),
    Param('combination', 'str', '2 p', label='Combination', visible_if=(('definition', ('combination',)),),
          help="Integer combination of other channel names (their feature names), e.g. '2 p' (second harmonic of "
               "p), 'p + s' (sum frequency), 'p - s' (difference frequency), '2 p - s' (four-wave mixing idler); "
               "'*' between count and name is optional. Parsed, not evaluated; the frequency must be > 0 and no "
               'cycle is allowed. Name the frequency name of a polarization both channel (without _ez / _hz).',
          tip="Frequency of this channel as a combination of other channels, for example '2 p' for the second "
              "harmonic of channel p or 'p + s' for their sum frequency."),
    Param('polarization', 'choice', 'ez', label='Polarisation', choices=POLARIZATIONS,
          choice_labels=(('ez', 'Electric field out of plane'), ('hz', 'Magnetic field out of plane'),
                         ('both', 'Both (two channels)')),
          help='ez: the channel carries E_z (an em_fdfd_ez solve); hz: E_x, E_y (an em_fdfd_hz solve); both: two '
               'channels <name>_ez and <name>_hz at the same frequency (two factorisations: memory doubles). Chi2 '
               'tensors couple the polarisations (d31 E_z^2 -> P_x, GaAs d14 E_x E_y -> P_z), so a generated channel '
               'may need the other polarisation than its pump.',
          tip='Which field the channel solves: electric field out of the plane, magnetic field out of the plane '
              '(electric field in the plane), or both as two channels.'),
    *(Param(f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
            visible_if=(('definition', ('wavelength',)),),
            help=f'Input channels: Bloch wave vector along {axis} in rad/m, used only when bc_{axis}_lo / bc_{axis}_hi '
                 f'are periodic (validation warns otherwise): the channel solves the Bloch wave F({axis} + L) = '
                 f'F({axis}) exp(-j k{axis} L) per meshed period L, as em_fdfd_* bloch_k{axis} (oblique incidence on '
                 'a grating or photonic crystal: k0 sin(theta) of that input). A generated channel carries the same '
                 'combination of its inputs\' k (2 p: 2 k_p), which keeps every polarisation a Bloch wave of its '
                 'target; an input that a process reaches by frequency only must have the matching k modulo 2 pi / L '
                 '(validation error otherwise). With k != 0 on any channel: no surface_chi2 / sheet_nonlinearity '
                 '(error) and no port line spanning that Bloch period. 0 = plain periodic.',
            tip=f'Phase step of this input wave from one period to the next along {axis}, in radians per metre, for '
                'periodic edges.\n\n0 means no phase step, as at normal incidence. Generated channels take the '
                'matching phase step themselves.')
      for axis in ('x', 'y')),
)
_HARMONIC_CHANNEL = FeatureType(
    _CHANNEL_FEATURE, 'Harmonic channel', 'domain_condition', None, params=HARMONIC_CHANNEL_PARAMS,
    description='One frequency (and polarisation) solved by the nonlinear physics; the feature name is the channel '
                'name (an identifier). Input channels have a wavelength, generated ones an integer combination of '
                'other channels (the frequency matches exactly). Every channel is its own em_fdfd_ez / _hz solve on '
                'the shared mesh and materials, each material evaluated at the channel frequency (phase matching comes '
                'from the material dispersion). The mixing table lists every chi2 / chi3 process landing on a declared '
                'channel; products off the channel set are not computed (no automatic closure, validation info). '
                'Undepleted mode: channels are solved once in rank order (inputs 0, a combination 1 + the highest '
                'rank it names), each with the nonlinear polarisation of lower-rank channels as its source; back-action '
                'on the pump and SPM / XPM are neglected (pump_depleted includes them). At least one input channel is '
                'needed; sources and monitors name their channel. Mirror faces (symmetric / antisymmetric bc_*): the '
                'input channels have that E parity, a generated channel the parity of its combination, s^(sum of its '
                'coefficients) (2 p and p - s are even, 3 p and p + s - i odd), so under an antisymmetric face an even '
                'channel solves with that face symmetric (its unfolded fields follow); an input reached by frequency '
                'only must have the parity of the process (validation error otherwise). The image half holds the '
                'mirror image of the drawn crystal. Bloch k: bloch_kx / bloch_ky of the inputs, the combination for '
                'generated channels.',
    summary='Declares one wavelength the nonlinear model solves: an input such as a pump, or a product such as '
            'the second harmonic. Sources and monitors name the channel they belong to.')

_D_HELP_CLASSES = {name: tuple(crystal_class for crystal_class, names in CRYSTAL_CLASSES.items() if name in names)
                   for name in CRYSTAL_CLASSES['general']}
_VOIGT_TEXT = ('Voigt index pair il: i = polarisation component (1 x, 2 y, 3 z of the crystal axes), l = field pair (1 '
               'xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy)')


def _d_param(name):
    classes = _D_HELP_CLASSES[name]
    return Param(name, 'float', 0.0, label=name, unit='m/V', visible_if=(('crystal_class', classes),),
                 help=f'Second-order coefficient {name} in m/V of the crystal axes (1 pm/V = 1e-12 m/V; d = chi2 / 2), '
                      f"{_VOIGT_TEXT}; used by the classes {', '.join(classes)}. P(2 omega) = eps0 d E^2 for SHG with "
                      'one field component (exp(+j omega t) phasors).',
                 tip=f'Nonlinear coefficient {name} in metres per volt: 1 pm/V is 1e-12.')


CHI2_MEDIUM_PARAMS = (
    Param('crystal_class', 'choice', '3m', label='Crystal class', choices=tuple(CRYSTAL_CLASSES),
          choice_labels=(('3m', '3m (lithium niobate)'), ('-43m', '-43m (gallium arsenide)'),
                         ('6mm', '6mm (aluminium or gallium nitride)'), ('mm2', 'mm2 (KTP)'), ('32', '32 (quartz)'),
                         ('general', 'General (every coefficient)')),
          help='Point group fixing the non-zero d_il (Kleinman symmetry assumed, except general): 3m (LiNbO3, LiTaO3, '
               'BBO: d22, d31 = d15 = d24, d33, with d16 = d21 = -d22; LN |d33| ~ 2.5e-11, |d31| ~ 4.6e-12, |d22| ~ '
               '2.1e-12 m/V at 1.06 um), -43m (GaAs, GaP, InP, ZnS: d14 = d25 = d36; GaAs ~ 1.2e-10 m/V at 1.55 um '
               'SHG), 6mm (AlN, GaN, ZnO: d15 = d24, d31 = d32, d33), mm2 (KTP: d15, d24, d31, d32, d33), 32 '
               '(quartz: d11, d12 = d26 = -d11, d14, d25 = -d14), general (all 18 d_il; the power exchange identity '
               'is then exact only for Kleinman-symmetric values). Signs follow the crystal axes of the orientation '
               'setting.',
          tip='Symmetry class of the crystal, which decides which nonlinear coefficients exist: 3m for lithium '
              'niobate, -43m for gallium arsenide, 6mm for aluminium or gallium nitride, mm2 for KTP, 32 for quartz.'),
    *(_d_param(name) for name in CRYSTAL_CLASSES['general']),
)
_ORIENTATION_PARAMS = (
    Param('orientation', 'choice', 'z_cut', label='Orientation', choices=('z_cut', 'x_cut', 'y_cut', 'euler'),
          choice_labels=(('z_cut', 'Crystal axis c along z (out of plane)'), ('x_cut', 'Crystal axis c along x'),
                         ('y_cut', 'Crystal axis c along y'), ('euler', 'Euler angles')),
          help='Crystal axes in model axes: z_cut = crystal x, y, z along model x, y, z (c axis out of plane, so E_z '
               'channels see d33); x_cut = crystal x, y, z along model y, z, x (c axis along model x); y_cut = crystal '
               'x, y, z along model z, x, y (c axis along model y); euler = Rz(euler_alpha) Rx(euler_beta) '
               'Rz(euler_gamma) (z-x-z, degrees; column a of the rotation = crystal axis a in model coordinates).',
          tip='How the crystal is cut relative to the drawing: the crystal c axis out of the plane, along x or along y, '
              'or general Euler angles.'),
    *(Param(f'euler_{angle}', 'float', 0.0, label=f'Euler {angle}', unit='deg', visible_if=(('orientation', ('euler',)),),
            help=f'z-x-z Euler angle {angle} in degrees (rotation crystal -> model axes Rz(alpha) Rx(beta) Rz(gamma)).',
            tip='Euler angle of the crystal orientation, in degrees.')
      for angle in ('alpha', 'beta', 'gamma')),
)
CHI2_MEDIUM_PARAMS = CHI2_MEDIUM_PARAMS + _ORIENTATION_PARAMS + (
    Param('poling', 'float', 1.0, label='Poling', unit='1', field_dependent=True,
          help='Domain sign per cell, an expression of x, y and parameters: values >= 0 keep the tensor, < 0 invert it '
               '(chi2 -> -chi2). Periodic poling for quasi phase matching: sign(sin(2 * pi * x / Lambda)) with the '
               'period Lambda = 2 pi / |k(2 omega) - 2 k(omega)| = lambda / (2 |n(2 omega) - n(omega)|) of the '
               'guided modes; 1 = unpoled. Chirped gratings and domain walls work the same way.',
          tip='Sign of the crystal domain at each point: 1 keeps the crystal as is, -1 flips it. A formula such as '
              'sign(sin(2*pi*x/period)) gives periodic poling for quasi phase matching.'),
)


def _miller_param(reference):
    return Param('miller_scaling', 'bool', False, label="Miller's rule",
                 help="Scale every process of this medium by Miller's rule, chi(n)(w; w1..wn) proportional to chi1(w) "
                      'chi1(w1)..chi1(wn), chi1 = Re eps_r - 1 averaged over the medium cells at each channel frequency '
                      '(poles included, evaluated once at build; the run log lists the scaled factors), normalised so '
                      f'that the given coefficients hold for {reference}. Off: one frequency-independent tensor. '
                      'Validation estimates use the unscaled values; Miller-scaled tensors are not fully permutation '
                      'symmetric, so power_exchange_imbalance is no longer ~0.',
                 tip='Lets the nonlinear coefficient follow the material dispersion (Miller\'s rule); off uses one '
                     'value at every wavelength.')


CHI2_MEDIUM_PARAMS = CHI2_MEDIUM_PARAMS + (
    _miller_param('SHG of reference_wavelength, d(2 w_r; w_r, w_r)'),
    Param('reference_wavelength', 'float', None, label='Reference wavelength', unit='m', min=0.0, min_exclusive=True,
          optional=True, visible_if=(('miller_scaling', (True,)),),
          help="Vacuum pump wavelength [m] of the SHG process the d coefficients are given for (Miller's rule "
               'reference); empty (None, the default) = the first input channel.',
          tip='Pump wavelength at which the nonlinear coefficients are given; empty means the first input channel.'),
)
_CHI2_MEDIUM = FeatureType(
    _CHI2_FEATURE, 'Second-order nonlinear medium', 'domain_condition', 'domain', params=CHI2_MEDIUM_PARAMS,
    description='Bulk chi2 of the selected domains: the crystal-class tensor rotated into model axes, times the '
                'poling sign per cell (stored as int8 on the selection bounding box). Response (exp(+j omega t) '
                'phasors): P(w3) = eps0 2^(1-n) sum over ordered source tuples chi2 : E E, so SHG P(2w) = eps0 d E^2, '
                'SFG P(w1 + w2) = 2 eps0 d E1 E2, DFG P(w1 - w2) = eps0 chi2 E1 conj(E2) (d = chi2 / 2). Products are '
                'formed at the cell centres (hz E_x / E_y collocated by the face mean C, the polarisation returned to '
                'the faces by its area-weighted adjoint, which keeps the discrete power exchange exact). One tensor '
                "for every process, or Miller's rule (miller_scaling) relative to the SHG of reference_wavelength. "
                'Overlapping media add. Fields and powers are 2D: per unit length in z, so a conversion efficiency '
                'P_2w / P_w^2 has unit m/W (multiply by an effective height to compare with a 3D guide).',
    summary='Gives the selected region a second-order optical nonlinearity (second harmonic, sum and difference '
            'frequencies), with crystal class, cut and optional periodic poling.',
    apply=_apply_chi2_medium, validate=_validate_chi2_medium, memory_estimate=_chi2_memory_estimate)

_N2_BETA = (('strength', ('n2_beta',)),)
_MILLER = (('miller_scaling', (True,)),)
_FREE_CARRIERS = (('free_carriers', (True,)),)
CHI3_MEDIUM_PARAMS = (
    Param('tensor', 'choice', 'isotropic', label='Tensor', choices=('isotropic', 'cubic'),
          choice_labels=(('isotropic', 'Isotropic (glass, amorphous)'), ('cubic', 'Cubic crystal (silicon)')),
          help='isotropic: chi_xxyy = chi_xyxy = chi_xyyx = chi_xxxx / 3 (glasses, amorphous films, a common '
               'approximation); cubic: chi_iijj = chi_ijij = chi_ijji = anisotropy chi_xxxx / 3 in the crystal axes '
               '(m3m: Si, Ge, GaAs Kerr), rotated by the orientation setting.',
          tip='Symmetry of the third-order response: isotropic for glasses, cubic for crystals such as silicon.'),
    Param('anisotropy', 'float', 1.0, label='Anisotropy', unit='1', min=0.0, visible_if=(('tensor', ('cubic',)),),
          help='rho = 3 chi_xxyy / chi_xxxx of the cubic tensor (1 = isotropic; silicon ~1.27 at 1.55 um).',
          tip='Cubic anisotropy of the third-order response: 1 is isotropic, silicon about 1.27.'),
    *(replace(param, visible_if=(('tensor', ('cubic',)),)) if param.name == 'orientation' else param
      for param in _ORIENTATION_PARAMS),
    Param('strength', 'choice', 'n2_beta', label='Strength given by', choices=('n2_beta', 'chi3'),
          choice_labels=(('n2_beta', 'Kerr index and two-photon absorption'), ('chi3', 'chi3 directly')),
          help='n2_beta: chi_xxxx = (4 eps0 c n0^2 / 3) (n2 - j c tpa_beta / (2 omega_ref)) (the em_fdtd_* '
               'nonlinear_medium vocabulary); chi3: chi_xxxx = chi3_real + j chi3_imag directly. One '
               'frequency-independent chi3 serves every process (THG, SPM / XPM, FWM), although real materials '
               'differ per process.',
          tip='Whether the strength is given as the Kerr index and two-photon absorption, or as the '
              'susceptibility chi3 itself.'),
    Param('n2', 'float', 0.0, label='Kerr index n2', unit='m^2/W', visible_if=_N2_BETA,
          help='Kerr coefficient: dn = n2 I, I = n0 c eps0 |E|^2 / 2 for a linearly polarised plane wave (SPM P = (3/4) '
               'eps0 chi3 |E|^2 E with exp(+j omega t) phasors). Silicon ~4.5e-18 to 6e-18, Si3N4 ~2.5e-19, silica '
               '~2.6e-20 m^2/W at 1.55 um.',
          tip='Intensity-dependent refractive index: silicon about 5e-18, silicon nitride about 2.5e-19 square '
              'metres per watt.'),
    Param('tpa_beta', 'float', 0.0, label='Two-photon absorption', unit='m/W', min=0.0, visible_if=_N2_BETA,
          help='Two-photon absorption dI/dz = -beta I^2 (Im chi3 < 0 in exp(+j omega t)); silicon ~8e-12 m/W at 1.55 '
               'um. Its power is the scalar nonlinear_absorption (pump_depleted).',
          tip='Two-photon absorption coefficient: silicon at 1550 nm about 8e-12 metres per watt.'),
    Param('background_index', 'float', None, label='Background index', unit='1', min=0.0, min_exclusive=True,
          optional=True, visible_if=_N2_BETA + _FREE_CARRIERS,
          help='n0 of the n2 / beta conversion and of the free-carrier permittivity; empty (None, the default) = Re '
               'sqrt(eps_r(omega_ref)) of each cell (poles included), evaluated once at build.',
          tip='Refractive index used to convert the Kerr index into chi3; leave empty to use the material index of '
              'each cell.'),
    Param('reference_wavelength', 'float', None, label='Reference wavelength', unit='m', min=0.0,
          min_exclusive=True, optional=True, visible_if=_N2_BETA + _MILLER + _FREE_CARRIERS,
          help='Vacuum wavelength [m] at which n2 and tpa_beta are given (omega_ref of the TPA term and of n0), the '
               "Miller's-rule reference (SPM at it) and the wavelength of the free-carrier coefficients; empty = the "
               'first input channel.',
          tip='Wavelength at which the Kerr index and two-photon absorption are given; empty means the first '
              'input channel.'),
    Param('chi3_real', 'float', 0.0, label='chi3 real part', unit='m^2/V^2', visible_if=(('strength', ('chi3',)),),
          help='Re chi_xxxx [m^2/V^2] (P = eps0 2^(1-n) sum chi3 E E E; silicon ~2.5e-19 at 1.55 um).',
          tip='Real part of the third-order susceptibility, in square metres per square volt.'),
    Param('chi3_imag', 'float', 0.0, label='chi3 imaginary part', unit='m^2/V^2', max=0.0,
          visible_if=(('strength', ('chi3',)),),
          help='Im chi_xxxx [m^2/V^2], <= 0 (loss in exp(+j omega t)).',
          tip='Imaginary part of the third-order susceptibility: zero or negative (absorption).'),
    Param('thg_scale', 'float', 1.0, label='Third-harmonic factor', unit='1',
          help='Factor on chi3 for the sum-frequency processes THG (3 w) and TFG (w1 + w2 + w3) of this medium '
               '(per-process chi3: the THG susceptibility of real materials differs from the Kerr one, which n2 / '
               'tpa_beta set); 1 = the Kerr chi3, 0 drops these terms. A factor != 1 breaks the full permutation '
               'symmetry (power_exchange_imbalance no longer ~0).',
          tip='Multiplies the third-order susceptibility for third-harmonic generation only; 1 uses the Kerr value.'),
    Param('fwm_scale', 'float', 1.0, label='Four-wave mixing factor', unit='1',
          help='Factor on chi3 for the FWM processes of this medium (w1 + w2 - w3 onto a channel, other than SPM / XPM '
               'of a channel on itself); 1 = the Kerr chi3, 0 drops these terms.',
          tip='Multiplies the third-order susceptibility for four-wave mixing only; 1 uses the Kerr value.'),
    _miller_param('SPM at reference_wavelength (the n2 / tpa_beta process)'),
    Param('free_carriers', 'bool', False, label='Free carriers',
          help='Steady-state free carriers from two-photon absorption (coupling_mode pump_depleted only; ignored '
               'undepleted, with a warning): N = carrier_lifetime G per cell, G = the TPA power density of the applied '
               'SPM / XPM terms / (2 hbar omega) (one electron-hole pair per two photons; no diffusion, so N follows '
               'the local intensity squared), and every channel sees dn = -fcd_coefficient N and alpha = '
               'fca_cross_section N, both x (lambda / reference_wavelength)^2 (Drude scaling), as delta eps = 2 n0 dn '
               '- j n0 c alpha / omega (n0 = background_index or Re sqrt eps_r(omega_ref) per cell). Linear in N (the '
               'Soref-Bennett N^0.8 hole term is linearised into fcd_coefficient). Field carrier_density, scalars '
               'free_carrier_absorption and peak_carrier_density.',
          tip='Adds the absorption and index change of free carriers created by two-photon absorption (silicon); '
              'only in the self-consistent (pump depleted) mode.'),
    Param('carrier_lifetime', 'float', 1e-9, label='Carrier lifetime', unit='s', min=0.0, visible_if=_FREE_CARRIERS,
          help='Effective free-carrier lifetime tau [s]: silicon wires ~0.5-3 ns, with carrier sweep-out by a '
               'reverse-biased junction ~10-100 ps; 0 = no carriers.',
          tip='How long the free carriers live: about 1 ns in a silicon waveguide, much less with carrier sweep-out.'),
    Param('fcd_coefficient', 'float', 1.35e-27, label='Carrier index coefficient', unit='m^3', min=0.0,
          visible_if=_FREE_CARRIERS,
          help='Free-carrier dispersion dn = -fcd_coefficient N at reference_wavelength [m^3]; silicon at 1.55 um '
               '~1.35e-27 (equal electron and hole densities, a linear fit of Soref-Bennett).',
          tip='Index decrease per free carrier, in cubic metres: about 1.35e-27 for silicon at 1550 nm.'),
    Param('fca_cross_section', 'float', 1.45e-21, label='Carrier absorption cross-section', unit='m^2', min=0.0,
          visible_if=_FREE_CARRIERS,
          help='Free-carrier absorption alpha = fca_cross_section N at reference_wavelength [m^2]; silicon at 1.55 um '
               '~1.45e-21 (equal electron and hole densities).',
          tip='Absorption per free carrier, in square metres: about 1.45e-21 for silicon at 1550 nm.'),
)
_CHI3_MEDIUM = FeatureType(
    _CHI3_FEATURE, 'Third-order nonlinear medium', 'domain_condition', 'domain', params=CHI3_MEDIUM_PARAMS,
    description='Bulk chi3 of the selected domains (tensor rotated into model axes, n2 / beta scaled by n0^2 per '
                'cell). Response (exp(+j omega t) phasors): THG P(3w) = (1/4) eps0 chi3 E^3, SPM P(w) = (3/4) eps0 '
                'chi3 |E|^2 E, XPM P(w1) = (6/4) eps0 chi3 |E2|^2 E1, degenerate FWM P(2 wp - ws) = (3/4) eps0 chi3 Ep^2 '
                'conj(Es). Undepleted mode keeps only THG / FWM-type terms feeding higher-rank channels (SPM / XPM act '
                'on a channel from itself: pump_depleted, where with the picard iteration their diagonal enters the '
                'channel operator lagged per sweep). Per-process factors thg_scale / fwm_scale or Miller\'s rule '
                'relative to SPM at reference_wavelength; optional steady-state free carriers from two-photon '
                'absorption (pump_depleted). Products formed at the cell centres as for chi2_medium; overlapping chi3 '
                'media add. Fields and powers are 2D: per unit length in z.',
    summary='Gives the selected region a third-order optical nonlinearity: Kerr effect, two-photon absorption, third '
            'harmonic and four-wave mixing.',
    apply=_apply_chi3_medium, validate=_validate_chi3_medium, memory_estimate=_chi3_memory_estimate)
_PEC_DOMAIN = FeatureType(
    'pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
    description='Perfect electric conductor in every channel. E_z channels: '
                f"{_fdfd_feature('ez', 'pec_domain').description} H_z channels: "
                f"{_fdfd_feature('hz', 'pec_domain').description}",
    summary=_fdfd_feature('ez', 'pec_domain').summary, apply=_apply_pec,
    memory_estimate=lambda values, n_selected_cells, mesh: MemoryEstimate.from_items(
        build_extra=BOOL_BYTES * int(mesh.n_cells)))   # per-channel masks: the physics estimator
_EZ_SHEET = _fdfd_feature('ez', _SHEET_FEATURE)
_CONDUCTING_SHEET = FeatureType(
    _SHEET_FEATURE, _EZ_SHEET.label, _EZ_SHEET.category, _EZ_SHEET.selection_kind, params=_EZ_SHEET.params,
    object_kinds=_EZ_SHEET.object_kinds, max_entities=_EZ_SHEET.max_entities,
    description=('Conducting sheet in every channel, sigma_s evaluated at each channel frequency (the linear '
                 f'response; sheet_nonlinearity on the same line adds the nonlinear one). {_EZ_SHEET.description}'),
    summary=_EZ_SHEET.summary, apply=_apply_conducting_sheet, validate=_validate_conducting_sheet,
    memory_estimate=_EZ_SHEET.memory_estimate)   # run items of one assembly at a time (channels assemble in turn)
_FDFD_DESIGN_REGION = design_region_feature()
_DESIGN_REGION = replace(   # the em_fdfd_* feature on NonlinearHarmonicPhysics.add_design_region (same signature)
    _FDFD_DESIGN_REGION,
    description=f'{_FDFD_DESIGN_REGION.description} In em_fdfd_nonlinear one region writes eps_r, eps_xx and eps_yy '
                '(every channel sees the design, each at its own frequency); the chi2 / chi3 media and their poling '
                "stay where they are drawn. The optimization study runs coupling_mode 'undepleted' (cascade adjoint) "
                "or 'pump_depleted' (fixed-point adjoint; iteration_method 'newton' with a chi3 medium).")


# ---- settings and material vocabulary ----
# Bloch k is per channel (harmonic_channel bloch_kx / bloch_ky of the inputs, the combination of generated channels:
# solver.channel_boundary_config), so the physics-level FDFD bloch_kx / bloch_ky settings are not offered; nor the
# triangles-only discretization (em_fdfd_nonlinear runs on tensor meshes).
_OMITTED_SETTINGS = frozenset({'wavelength', 'bloch_kx', 'bloch_ky', 'discretization'})
_PUMP_DEPLETED = (('coupling_mode', ('pump_depleted',)),)
_NONLINEAR_SETTINGS = (
    Param('coupling_mode', 'choice', 'undepleted', label='Coupling mode', choices=COUPLING_MODES,
          choice_labels=(('undepleted', 'Undepleted (one pass)'), ('pump_depleted', 'Pump depleted (self-consistent)')),
          help='undepleted: every channel solved once in rank order, fed only by lower-rank channels (no back-action, '
               'no SPM / XPM, no free carriers; valid at small conversion). pump_depleted: every mixing term '
               '(back-action, parametric gain, SPM / XPM) and the chi3 free carriers, solved self-consistently by '
               'iteration_method until the stored channel fields change by <= tolerance.',
          tip='Undepleted solves each wavelength once and ignores the reaction on the pump; pump depleted iterates '
              'until all wavelengths agree (strong conversion, Kerr effect, parametric gain).'),
    Param('iteration_method', 'choice', 'picard', label='Iteration method', choices=ITERATION_METHODS,
          choice_labels=(('picard', 'Sweeps with Anderson acceleration'), ('newton', 'Newton-Krylov')),
          visible_if=_PUMP_DEPLETED,
          help='pump_depleted solver. picard: block Gauss-Seidel sweeps over the channels in rank order with '
               'relaxation and Anderson acceleration, the SPM / XPM / free-carrier diagonal lagged in the channel '
               'operator (those channels reassemble and solve by GMRES on the recycled factor every sweep); cheap per '
               'sweep, converges while the conversion per pass is moderate. newton: Newton-Krylov on the coupled '
               'channels, the exact real-linear (Wirtinger) Jacobian of the polynomial mixing and carrier terms, GMRES '
               f'(restart {GMRES_RESTART}, at most {GMRES_MAX_ITERATIONS} products per Newton step) left-preconditioned '
               "by the channels' own factors (every channel solved from the same iterate), Eisenstat-Walker forcing "
               'and backtracking on the scaled residual; each GMRES product costs one back-substitution per nonlinear '
               'channel, and it converges quadratically near the solution (high conversion, parametric gain near '
               'threshold, cavities, strong Kerr phase). Kerr and carriers stay in the rhs (no reassembly; q_abs then '
               'holds only the linear losses); relaxation and anderson_depth are ignored. cusolver_qr refactorises '
               'every solve, which newton multiplies.',
          tip='How the self-consistent solve iterates: sweeps with acceleration, or Newton steps (fewer but more '
              'expensive iterations; more robust at strong conversion).'),
    Param('tolerance', 'float', 1e-8, label='Iteration tolerance', unit='1', min=0.0, min_exclusive=True,
          visible_if=_PUMP_DEPLETED,
          help='pump_depleted stops when the largest relative channel change ||G_k - x_k|| / ||G_k|| is <= tolerance, '
               'G = the channel fields solved from the stored fields x (picard: over one sweep; newton: one solve of '
               'every channel from the Newton iterate); the fixed-point update, not the operator residual. The '
               'recycled GMRES solves of picard reach 1e-10, so stay above ~1e-9.',
          tip='How little the fields may still change between iterations when the self-consistent solve stops.'),
    Param('max_iterations', 'int', 50, label='Iteration limit', min=1, visible_if=_PUMP_DEPLETED,
          help='Sweeps (picard) or Newton steps (newton) per continuation step at most; not converged then: warning, '
               'the last iterate is kept (snapshot labels say so, summary nonlinear_solve converged false).',
          tip='Largest number of iterations of the self-consistent solve (per continuation step).'),
    Param('relaxation', 'float', 1.0, label='Relaxation', unit='1', min=0.0, max=1.0, min_exclusive=True,
          visible_if=_PUMP_DEPLETED,
          help='picard only: under-relaxation beta of the mixing x + beta f (1 = plain sweeps); lower it (0.3-0.7) '
               'for an oscillating iteration (high conversion, cavities).',
          tip='Damping of the sweeps: 1 is undamped; smaller values help when the iteration oscillates.'),
    Param('anderson_depth', 'int', 5, label='Anderson depth', min=0, visible_if=_PUMP_DEPLETED,
          help='picard only: history m of the Anderson acceleration between sweeps (0 = relaxed Picard); costs 2 m '
               'stacked copies of every channel field on the GPU.',
          tip='How many previous sweeps the accelerated iteration remembers; 0 switches the acceleration off.'),
    Param('continuation_steps', 'int', 1, label='Continuation steps', min=1, visible_if=_PUMP_DEPLETED,
          help='Ramp every linear source amplitude through k / steps, k = 1..steps, each step warm-started from the '
               'previous (1 = none). Helps near bistability; a Kerr S-curve has several solutions and the branch '
               'reached depends on the ramp.',
          tip='Number of steps in which the sources are ramped up to full strength; 1 means no ramp.'),
)
_SETTINGS = tuple(param for param in EM_FDFD_EZ.settings if param.name not in _OMITTED_SETTINGS) + _NONLINEAR_SETTINGS
_COMPLEX_EPS = ('Each channel uses the complex permittivity at its own frequency: this value - j sigma_e / (omega '
                'eps0) plus every Drude / Lorentz pole chi(omega), so the material dispersion sets the phase mismatch '
                'between channels (non-dispersive materials are phase matched by construction).')
_MATERIAL_PROPERTIES = em_material_params('modes', {
    'eps_r': ('Isotropic relative permittivity (>= 1): sets eps_zz (E_z channels) and eps_xx, eps_yy (H_z channels) '
              'unless the material sets those components. n^2 of a lossless dielectric; may depend on x, y and '
              f'physics fields. {_COMPLEX_EPS} For a dispersive material it is the background eps_inf.'),
    'eps_zz': ('z component of the relative permittivity (>= 1), the one E_z channels see; overrides eps_r for E_z. '
               'z-cut lithium niobate: eps_r = n_o^2, eps_zz = n_e^2.'),
    'sigma_e': f'Ohmic loss in S/m on every E component, default 0. {_COMPLEX_EPS}',
    'mu_r': 'Real relative permeability (>= 1), 1 for practically every optical material; mu_r - j sigma_m / (omega '
            'mu0) per channel.',
    'sigma_m': 'Magnetic loss in Ohm/m, default 0: mu_r gains -j sigma_m / (omega mu0) (artificial absorbers only).',
}, tips={'eps_zz': ('Permittivity for the electric field along z, out of the plane, seen by the channels with the '
                    "electric field out of the plane. When set it replaces 'Relative permittivity' for that "
                    'direction.')})
_STATE_FIELD_NAMES = ('ez_re', 'ez_im', 'hz_re', 'hz_im', 'q_abs', 'eps_r', 'eps_xx', 'eps_yy', 'sigma_e')
_Q_ABS_INFO = replace(NONLINEAR_FIELD_INFOS['absorbed_power'], name='q_abs')


# ---- physics validation ----
_shared_validate = make_validate_physics(_SLOT_ARRAYS, time_domain=False, solved_wavelengths=_channel_wavelengths)


def _has_dispersive_material(vctx):
    """True when a material (library merged) carries Drude / Lorentz poles or an nk_fit."""
    from ...materials import effective_material_data
    for material in vctx.spec.materials:
        try:
            _, dispersion = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, KeyError, TypeError):
            continue
        if dispersion:
            return True
    return False


def _source_channels(physics_spec):
    """Channel names named by the enabled source features."""
    sources = set()
    for feature in physics_spec.features:
        if feature.enabled and feature.type in _CHANNEL_FEATURE_NAMES and isinstance(feature.params, dict):
            if _fdfd_feature('ez', feature.type).category == 'source':
                sources.add(feature.params.get('channel'))
    return sources


def _selection_regions(vctx, selection):
    """(object names, bounding boxes (x0, y0, x1, y1), selects 'all') of a domain selection; failed objects skipped."""
    try:
        entities = expand_entities(vctx.spec, selection)
    except (ValueError, KeyError, TypeError):
        return set(), [], False
    names = {name for name in entities if name in vctx.shapes}
    boxes = []
    for name in names:
        try:
            boxes.append(bounding_box(name, vctx.shapes))
        except (KeyError, ValueError):
            continue
    return names, boxes, 'all' in entities


def _boxes_overlap(first, second):
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def _overlap_reason(first_region, second_region):
    """Why two (names, boxes, selects all) selection regions may share cells, None when they cannot."""
    first_names, first_boxes, first_all = first_region
    second_names, second_boxes, second_all = second_region
    shared = first_names & second_names
    if shared:
        return f"both select {', '.join(sorted(shared))}"
    if first_all or second_all:
        return "one selects 'all'"
    if any(_boxes_overlap(a, b) for a in first_boxes for b in second_boxes):
        return 'their bounding boxes intersect'
    return None


def _media_overlap_issues(vctx, physics_spec, path):
    """Warning per pair of enabled media of one order that may share cells (shared object, 'all', intersecting boxes)."""
    issues = []
    for feature_type, order in _MEDIUM_ORDERS.items():
        regions = [(feature, _selection_regions(vctx, feature.selection))
                   for feature in _enabled_features(physics_spec, feature_type)]
        for index, (first, first_region) in enumerate(regions):
            for second, second_region in regions[index + 1:]:
                reason = _overlap_reason(first_region, second_region)
                if reason is not None:
                    issues.append(Issue('warning', path, f"chi{order} media '{first.name}' and '{second.name}' may "
                                                         f'overlap ({reason}): their tensors add in shared cells, '
                                                         'which is usually unintended (one medium per material; '
                                                         'poling flips domains inside one medium)'))
    return issues


def _port_kerr_issues(vctx, physics_spec, path):
    """pump_depleted with Kerr terms: warning per mode source / monitor whose line may cross a chi3 medium (its slab
    mode is solved without the Kerr permittivity)."""
    media = [(feature, _selection_regions(vctx, feature.selection))
             for feature in _enabled_features(physics_spec, _CHI3_FEATURE)]
    issues = []
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in _PORT_FEATURES:
            continue
        port_region = _selection_regions(vctx, feature.selection)
        for medium, region in media:
            if _overlap_reason(port_region, region) is not None:
                issues.append(Issue('warning', feature_path(physics_spec, feature),
                                    f"the line of {feature.type} '{feature.name}' may cross chi3 medium "
                                    f"'{medium.name}': its slab mode is solved without the Kerr permittivity, so at "
                                    'high power it no longer matches the guided mode there; place ports outside the '
                                    'Kerr region'))
                break
    return issues


def _medium_length(vctx, feature):
    """Largest extent [m] of a medium's selection (union bounding box; 'all': the domain), None when unknown."""
    _, boxes, selects_all = _selection_regions(vctx, feature.selection)
    if selects_all:
        return max(vctx.domain_size) if vctx.domain_size else None
    if not boxes:
        return None
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    return max(x1 - x0, y1 - y0)


def _source_amplitude(vctx, feature, channel):
    """Peak |E| estimate [V/m] of a source feature on its channel: amplitude, or sqrt(4 eta0 P / lambda) of a mode
    power (an upper bound); None for point sources or unresolved values."""
    params = feature.params if isinstance(feature.params, dict) else {}
    values = vctx.values(EM_FDFD_NONLINEAR.feature_type(feature.type).params, params)
    name = _AMPLITUDE_PARAMS.get(feature.type) or _POWER_PARAMS.get(feature.type)
    if values is None or name is None:
        return None
    try:
        value = abs(float(values[name]))
    except (TypeError, ValueError, KeyError):   # a field-dependent expression: not estimated
        return None
    if feature.type in _POWER_PARAMS:
        wavelength = 2.0 * math.pi * C0 / channel.omega
        return math.sqrt(4.0 * float(ETA0) * value / wavelength)
    return value


def _undepleted_validity_issues(vctx, physics_spec, channels, applied, path):
    """Warning when a term's generated field estimate omega |P_NL / eps0| L / (2 c) (n = 1, source peak fields,
    cascaded in rank order) exceeds _UNDEPLETED_WARN_RATIO of its weakest source field."""
    index_by_name = {channel.name: index for index, channel in enumerate(channels)}
    amplitudes = {}
    for feature in physics_spec.features:
        params = feature.params if isinstance(feature.params, dict) else {}
        index = index_by_name.get(params.get('channel'))
        if not feature.enabled or feature.type not in _CHANNEL_FEATURE_NAMES or index is None \
                or channels[index].rank != 0:
            continue
        amplitude = _source_amplitude(vctx, feature, channels[index])
        if amplitude is not None:
            amplitudes[index] = max(amplitudes.get(index, 0.0), amplitude)
    lengths = {feature.name: _medium_length(vctx, feature) for feature in _medium_features(physics_spec)}
    worst = None   # (ratio, term, length)
    for term in sorted(applied, key=lambda term: channels[term.target].rank):
        source_amplitudes = [amplitudes.get(index) for index, _ in term.sources]
        length = lengths.get(term.medium)
        if length is None or any(amplitude is None or amplitude <= 0.0 for amplitude in source_amplitudes):
            continue
        strength = term.factor * max(abs(value) for _, _, value in term.components) * math.prod(source_amplitudes)
        generated = channels[term.target].omega * strength * length / (2.0 * C0)
        amplitudes[term.target] = amplitudes.get(term.target, 0.0) + generated
        ratio = generated / min(source_amplitudes)
        if worst is None or ratio > worst[0]:
            worst = (ratio, term, length)
    if worst is None or worst[0] <= _UNDEPLETED_WARN_RATIO:
        return []
    ratio, term, length = worst
    return [Issue('warning', path, f'undepleted validity: {term.process} {source_text(channels, term.sources)} -> '
                                   f'{channels[term.target].name} may generate ~{ratio:.2g} x its weakest source '
                                   f'field over the {length * 1e6:.3g} um of medium {term.medium} (estimate omega '
                                   '|P_NL| L / (2 eps0 c |E|) with n = 1 and peak source fields from amplitude or '
                                   f'mode power: an overestimate; chi3 n2 media without background_index at n0 = 1: '
                                   f'an underestimate); above {_UNDEPLETED_WARN_RATIO} the neglected pump depletion '
                                   "matters: use coupling_mode 'pump_depleted', or lower the source amplitudes or the "
                                   'medium length')]


def _mixing_issues(vctx, physics_spec, settings_values, path):
    """Channel rules, the mixing table (error when nothing couples or nothing acts in the coupling mode; info listing
    the processes, neglected back-action / Kerr terms, near misses, off-set products), unused channels, the
    phase-matching info, the undepleted validity guard and the pump-depleted port / back-end warnings."""
    if not _enabled_features(physics_spec, _CHANNEL_FEATURE):
        return [Issue('error', path, f'{TYPE_NAME} needs harmonic_channel features: at least one input channel '
                                     "(a wavelength) and the generated ones (e.g. '2 p')")]
    channels, problem = _model_channels(physics_spec, vctx.parameters)
    if channels is None:
        return [Issue('error', path, f'harmonic channels: {problem}')]
    media = _media_tensors(vctx.values, physics_spec, channels)
    if not media:
        return [Issue('error', path, f'no enabled {_CHI2_FEATURE}, {_CHI3_FEATURE}, surface_chi2 or '
                                     'sheet_nonlinearity: nothing couples the channels')]
    table = build_mixing_table(channels, media)
    if not table.terms:
        reason = '; '.join(table.dropped[:_LOW_ORDER_INFO_TERMS]) or 'no exact frequency combination of the channels'
        return [Issue('error', path, f'no process couples the channels ({reason})')]
    pump_depleted = settings_values['coupling_mode'] == 'pump_depleted'
    applied = table.terms if pump_depleted else table.undepleted_terms()
    if not applied:
        return [Issue('error', path, 'undepleted: no process feeds a channel from channels of lower rank (only '
                                     'back-action or Kerr terms, e.g. SPM of a single channel); declare generated '
                                     "channels as combinations of the inputs, or use coupling_mode 'pump_depleted'")]
    issues = [Issue('info', path, f"mixing ({settings_values['coupling_mode']}): "
                                  + '; '.join(describe_mixing(table, applied)))]
    neglected = len(table.terms) - len(applied)
    if neglected:
        issues.append(Issue('info', path, f'{neglected} back-action / Kerr term(s) on channels of equal or lower rank '
                                          "are neglected in the undepleted mode (coupling_mode 'pump_depleted' "
                                          'includes them)'))
    if pump_depleted:
        issues.extend(_pump_depleted_issues(vctx, physics_spec, settings_values, table, path))
    for text in table.near_misses[:_LOW_ORDER_INFO_TERMS]:
        issues.append(Issue('info', path, text))
    if table.off_set_count:
        issues.append(Issue('info', path, f'{table.off_set_count} product(s) of the media fall on no channel and are '
                                          f"not computed, e.g. {'; '.join(table.off_set[:_LOW_ORDER_INFO_TERMS])}"))
    targets = {term.target for term in applied}
    sources = _source_channels(physics_spec)
    for index, channel in enumerate(channels):
        if channel.rank == 0 and channel.name not in sources:
            issues.append(Issue('warning', path, f'input channel {channel.name} has no source feature: its field is '
                                                 'zero, so nothing it drives is generated'))
        elif channel.rank > 0 and index not in targets:
            issues.append(Issue('warning', path, f'channel {channel.name} receives no applied process: it holds only '
                                                 'the field of its own sources'))
    if len({channel.omega for channel in channels}) > 1 and not _has_dispersive_material(vctx):
        issues.append(Issue('info', path, 'no material has dispersion poles: bulk material dispersion is absent, but '
                                          'waveguide dispersion and birefringence (different permittivity arrays '
                                          'per polarisation) still set the phase mismatch'))
    if any(_raw_flag(feature, 'miller_scaling') for feature in _medium_features(physics_spec)):
        issues.append(Issue('info', path, "Miller's rule media: the factors above are unscaled; the build scales them "
                                          'by the channel chi1 (run log)'))
    carrier_media = [feature.name for feature in _enabled_features(physics_spec, _CHI3_FEATURE)
                     if _raw_flag(feature, 'free_carriers')]
    if carrier_media and not pump_depleted:
        issues.append(Issue('warning', path, f"free carriers of {', '.join(carrier_media)} act only with "
                                             "coupling_mode 'pump_depleted' (they need the SPM / XPM absorption of "
                                             'the channels): ignored in the undepleted mode'))
    if pump_depleted:
        return issues
    return issues + _undepleted_validity_issues(vctx, physics_spec, channels, applied, path)


_NONLINEAR_SHEET_FEATURES = (SURFACE_CHI2.name, SHEET_NONLINEARITY.name)
_FACE_NAMES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')


def _channel_phase_issues(vctx, physics_spec, settings_values, path):
    """Per-channel Bloch k and mirror parity: an input k on a non-periodic axis (warning, ignored); a process landing by
    frequency only with a mismatched k or parity (error, mixing.inexact_term_issues); with k != 0 on a channel the
    nonlinear sheets (error) and port lines spanning that Bloch period (error)."""
    channels, _ = _model_channels(physics_spec, vctx.parameters)
    if not channels:
        return []   # reported by _mixing_issues
    periodic = {axis: settings_values[f'bc_{axis}_lo'] == 'periodic' for axis in 'xy'}
    issues = []
    features = {feature.name: feature for feature in _enabled_features(physics_spec, _CHANNEL_FEATURE)}
    for channel in channels:
        feature = features.get(channel.frequency_name)
        for axis, wavenumber in zip('xy', channel.bloch_k):
            if channel.rank == 0 and wavenumber != 0.0 and not periodic[axis] and feature is not None:
                issues.append(Issue('warning', feature_path(physics_spec, feature),
                                    f'bloch_k{axis} = {wavenumber:.4g} rad/m is ignored: bc_{axis}_lo / bc_{axis}_hi '
                                    'are not periodic'))
    bloch_axes = {channel.name: tuple(axis for axis, wavenumber in zip('xy', channel.bloch_k)
                                      if periodic[axis] and wavenumber != 0.0) for channel in channels}
    if any(bloch_axes.values()):
        for feature in physics_spec.features:
            if feature.enabled and feature.type in _NONLINEAR_SHEET_FEATURES:
                issues.append(Issue('error', feature_path(physics_spec, feature),
                                    f'{feature.type} needs Bloch k = 0 on every channel (its sites take no Bloch wrap '
                                    'phase); set the input bloch_kx / bloch_ky to 0'))
            elif feature.enabled and is_fdfd_port_feature(feature) and isinstance(feature.params, dict):
                axes = bloch_axes.get(feature.params.get('channel'), ())
                if axes:
                    issues.extend(bloch_period_line_issues(vctx, physics_spec, feature, axes))
    media = _media_tensors(vctx.values, physics_spec, channels)
    if not media:
        return issues
    if vctx.mesh is not None:   # periodic axes carry no padding: the grid length is the period
        lengths = tuple(float(edges[-1] - edges[0]) for edges in (vctx.mesh.x_edges, vctx.mesh.y_edges))
    else:
        lengths = tuple(vctx.domain_size) if vctx.domain_size else (None, None)
    periods = tuple(length if periodic[axis] else None for axis, length in zip('xy', lengths))
    odd_mirror = any(settings_values[f'bc_{face}'] == 'antisymmetric' for face in _FACE_NAMES)
    for text in inexact_term_issues(build_mixing_table(channels, media), periods, odd_mirror):
        issues.append(Issue('error', path, text))
    return issues


def _pump_depleted_issues(vctx, physics_spec, settings_values, table, path):
    """Info on the Kerr operator terms; warnings for a GPU QR back end (one factorisation per channel and sweep) and
    for mode ports crossing a chi3 medium."""
    issues = []
    newton = settings_values['iteration_method'] == 'newton'
    kerr = [term for term in table.terms if term.kerr]
    if kerr:
        if newton:
            text = ('stay in the rhs; the Newton Jacobian holds them exactly, so every solve reuses the channel '
                    'factor')
        else:
            text = ('their diagonal enters the channel operator from the previous sweep (lagged; the channel factor '
                    'is reused by GMRES), so a strong Kerr phase may need relaxation < 1, continuation_steps > 1 or '
                    "iteration_method 'newton'")
        issues.append(Issue('info', path, f'{len(kerr)} SPM / XPM term(s): {text}'))
        issues.extend(_port_kerr_issues(vctx, physics_spec, path))
    back_end = resolved_back_end(vctx.solvers(physics_spec))
    if back_end == 'cusolver_qr':
        per = 'Newton GMRES product' if newton else 'sweep'
        issues.append(Issue('warning', path, "pump_depleted with the GPU QR back end (cusolver_qr) refactorises every "
                                             f'channel in every {per} (no factor is kept): install scipy for '
                                             "'scipy_splu', whose factors are reused by every solve"))
    # a device factor without GMRES recycling (MUMPS / PARDISO host factors are recycled like SuperLU)
    elif back_end not in (None, 'scipy_splu') + HOST_DIRECT_BACK_ENDS and kerr and not newton:
        issues.append(Issue('info', path, f"back end '{back_end}' keeps no factor GMRES can recycle: with picard every "
                                          'Kerr / free-carrier channel refactorises in every sweep (its lagged '
                                          "operator changes); iteration_method 'newton' reuses one factor per channel"))
    return issues


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared FDFD checks (pairing, mirrors, materials, slot writers, resolution at every channel wavelength),
    the linear solver size rule, the mixing rules (+ undepleted validity / pump-depleted warnings) and overlapping
    media."""
    path = ('physics', physics_spec.name)
    return (_shared_validate(vctx, physics_spec, settings_values)
            + direct_solver_issues(vctx, physics_spec, settings_values, path)
            + _mixing_issues(vctx, physics_spec, settings_values, path)
            + _channel_phase_issues(vctx, physics_spec, settings_values, path)
            + _media_overlap_issues(vctx, physics_spec, path)
            + _optimization_issues(vctx, physics_spec, settings_values, path))


def _optimization_issues(vctx, physics_spec, settings_values, path):
    """Optimization study, pump_depleted: error for picard with a chi3 medium (its lagged Kerr / carrier operator;
    adjoint._check_differentiable)."""
    if vctx.study_type != OPTIMIZATION_STUDY or settings_values['coupling_mode'] != 'pump_depleted':
        return []
    if _enabled_features(physics_spec, _CHI3_FEATURE) and settings_values['iteration_method'] != 'newton':
        return [Issue('error', path, "pump_depleted optimization with a chi3 medium needs iteration_method 'newton' "
                                     '(picard keeps a lagged Kerr / free-carrier operator in the channel factors, '
                                     'which the fixed-point adjoint cannot transpose)')]
    return []


# ---- build ----
def _create(ctx, physics_spec, settings_values):
    """NonlinearHarmonicPhysics of the harmonic_channel features with the dispersive poles in every channel; the
    mixing table is built at finalize."""
    solver = NonlinearHarmonicPhysics(
        ctx.domain, _channel_specs(physics_spec, ctx.parameters), boundary_config=ctx.domain.boundary_config,
        linear_solver=ctx.solvers(physics_spec.name)['linear'], coupling_mode=settings_values['coupling_mode'],
        tolerance=settings_values['tolerance'], max_iterations=settings_values['max_iterations'],
        relaxation=settings_values['relaxation'], anderson_depth=settings_values['anderson_depth'],
        continuation_steps=settings_values['continuation_steps'], iteration_method=settings_values['iteration_method'])
    write_dispersive_materials(ctx, physics_spec, solver, solver.permittivity_arrays)
    return solver


def _mixing_table(physics_spec, parameters, channels):
    """mixing.MixingTable of the media that resolve over `channels`, None without one (memory and time estimates)."""
    def resolve(params, raw):
        try:
            return resolve_values(params, raw, parameters)
        except ValueError:
            return None
    media = _media_tensors(resolve, physics_spec, channels)
    return build_mixing_table(channels, media) if media else None


# run-time estimate guesses (calibrate from summary.json solvers.<physics>: counters sweeps / newton_gmres_iterations of
# a single nonlinear solve, adjoint_sweeps, roles.linear.counters gmres_iterations / recycled_solves): picard sweeps and
# Newton steps per continuation step, Jacobian products per Newton step, channel evaluations per Newton step (1 + the
# line-search backtracks), recycled GMRES iterations per Kerr-channel re-solve, products of the pump-depleted adjoint
PICARD_SWEEPS_GUESS = 15
NEWTON_STEPS_GUESS = 4               # selftests / examples: 3 - 21 products per single-step solve (median ~10)
NEWTON_PRODUCTS_PER_STEP_GUESS = 3
NEWTON_EVALUATIONS_PER_STEP_GUESS = 1.5
RECYCLED_GMRES_ITERATIONS_GUESS = 5
ADJOINT_PRODUCTS_GUESS = 20


def _term_ops(term, n_cells, slots=1):
    """Ops of one mixing term's density on the grid: per tensor component a chain of `order` complex products (cupy
    ufuncs); slots = the source slots differentiated (a Newton tangent: order)."""
    chain = term.order * max(len(term.components), 1) * slots
    return te.elementwise(n_cells * chain, 2, 1, COMPLEX128_BYTES, launches=chain)


def _over(outer, count, limit):
    """(total count, limit) of per-nonlinear-solve work over the outer solves (count, kind, limit); limit None when
    either has none."""
    outer_count, outer_kind, outer_limit = outer
    bound = outer_count if outer_kind != 'guess' else outer_limit
    return outer_count * count, (None if bound is None or limit is None else bound * limit)


def _time_estimate(settings_values, mesh, context):
    """Per nonlinear solve (the static solves; optimization: per iteration, bounded, + the adjoint) every channel's
    assembly and SuperLU factor ('linear' role) and its solves: undepleted one per channel; picard sweeps x channels
    (Kerr / carrier channels reassembled, solved by recycled GMRES) + Anderson mixing; newton evaluations x channels +
    Jacobian products x target channels; plus the polarisation densities."""
    physics_spec = context.physics_spec
    channels, _ = _model_channels(physics_spec, context.parameters)
    table = _mixing_table(physics_spec, context.parameters, channels) if channels else None
    if table is None:
        return te.TimeEstimate(not_estimated=('channels or nonlinear media that do not resolve',))
    n_cells = float(mesh.n_cells)
    nnz = 5.0 * n_cells
    n_channels = len(channels)
    pump_depleted = settings_values['coupling_mode'] == 'pump_depleted'
    newton = pump_depleted and settings_values['iteration_method'] == 'newton'
    applied = table.terms if pump_depleted else table.undepleted_terms()
    carriers = pump_depleted and any(_raw_flag(feature, 'free_carriers')
                                     for feature in _enabled_features(physics_spec, _CHI3_FEATURE))
    every_channel = set(range(n_channels))
    targets = every_channel if carriers else {term.target for term in applied}
    kerr = set()
    if pump_depleted and not newton:   # KerrOperatorTerm channels (carriers: every channel)
        kerr = every_channel if carriers else {term.target for term in applied if term.kerr}
    store = n_cells * sum(len(channel.components) for channel in channels)
    hz_channels = sum(1 for channel in channels if channel.polarization == 'hz')
    density = [_term_ops(term, n_cells) for term in applied]
    density.append(te.elementwise(n_cells * hz_channels, 4, 2, COMPLEX128_BYTES, launches=max(4 * hz_channels, 1)))
    if carriers:   # G from the Kerr densities, N, the carrier polarisation of every channel
        density.append(te.elementwise(n_cells * n_channels, 8, 2, COMPLEX128_BYTES, launches=10))

    # per nonlinear solve: channel solves, density evaluations, picard sweeps / Newton Jacobian products (+ limits)
    steps, iteration_limit = int(settings_values['continuation_steps']), int(settings_values['max_iterations'])
    passes = passes_limit = 0.0
    if not pump_depleted:
        solves = solves_limit = float(n_channels)
        evaluations = evaluations_limit = 1.0
    elif newton:
        newton_steps = min(NEWTON_STEPS_GUESS, iteration_limit) * steps
        evaluations, evaluations_limit = steps + newton_steps * NEWTON_EVALUATIONS_PER_STEP_GUESS, None
        passes = newton_steps * min(NEWTON_PRODUCTS_PER_STEP_GUESS, GMRES_MAX_ITERATIONS)
        passes_limit = float(iteration_limit * steps * GMRES_MAX_ITERATIONS)
        solves, solves_limit = evaluations * n_channels + passes * len(targets), None
    else:
        passes = evaluations = float(min(PICARD_SWEEPS_GUESS, iteration_limit) * steps)
        passes_limit = evaluations_limit = float(iteration_limit * steps)
        solves, solves_limit = evaluations * n_channels, evaluations_limit * n_channels
    recycled = len(kerr) * max(passes - 1.0, 0.0)   # the first sweep factorises
    recycled_limit = len(kerr) * max(passes_limit - 1.0, 0.0)

    optimization = context.study.type == OPTIMIZATION_STUDY
    outer = optimization_evaluations(context) if optimization else context.static_solves()
    kind = 'guess' if pump_depleted or outer[1] == 'guess' else outer[1]
    single = outer[0] == 1.0 and outer[1] == 'exact'   # the cascade counters hold the last nonlinear solve only
    physics_counter = ('solvers', context.name, 'counters')
    role_counter = ('solvers', context.name, 'roles', 'linear', 'counters')
    assembly_ops = (te.HostStream(ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200))
    channel_solves, channel_limit = _over(outer, solves, solves_limit)
    density_count, density_limit = _over(outer, evaluations, evaluations_limit)
    items = [te.TimeItem('channel assemblies', 'assembly', outer[0] * n_channels, outer[1], assembly_ops,
                         limit=_over(outer, n_channels, n_channels)[1]),
             te.TimeItem('channel solves (rhs, q_abs, store capture)', 'solve', channel_solves, kind,
                         (te.elementwise(n_cells, 6, 4, COMPLEX128_BYTES, launches=12), te.PythonCalls(40)),
                         limit=channel_limit, counter=physics_counter + ('solves',)),
             te.TimeItem('nonlinear polarisation densities', 'evaluation', density_count, kind, tuple(density),
                         limit=density_limit)]
    krylov = (te.elementwise(store, 2 * GMRES_RESTART, 2, COMPLEX128_BYTES, launches=8), te.Sync(4),
              te.PythonCalls(30 + 4 * n_channels))   # CGS2 over half the restart basis on average
    if newton:
        count, limit = _over(outer, passes, passes_limit)
        tangents = tuple(_term_ops(term, n_cells, slots=term.order) for term in applied)
        items.append(te.TimeItem('Newton-Krylov Jacobian products', 'product', count, 'guess', tangents + krylov,
                                 limit=limit, counter=physics_counter + ('newton_gmres_iterations',) if single else ()))
    elif pump_depleted:
        count, limit = _over(outer, passes, passes_limit)
        depth = int(settings_values['anderson_depth'])
        anderson = (te.elementwise(store, 2 * depth + 4, 3, COMPLEX128_BYTES, launches=10), te.Sync(2 * n_channels),
                    te.PythonCalls(20 + 4 * n_channels))
        items.append(te.TimeItem('Anderson mixing sweeps', 'sweep', count, 'guess', anderson, limit=limit,
                                 counter=physics_counter + ('sweeps',) if single else ()))
    recycled_count, recycled_total_limit = _over(outer, recycled, recycled_limit)
    if recycled:
        kerr_ops = assembly_ops + (te.elementwise(n_cells, 6, 2, COMPLEX128_BYTES, launches=8),   # Kerr operator update
                                   te.Transfer(nnz * (COMPLEX128_BYTES + 4) + 2.0 * COMPLEX128_BYTES * n_cells))
        items.append(te.TimeItem('Kerr re-solves (reassembly, recycled GMRES)', 'solve', recycled_count, 'guess',
                                 kerr_ops, limit=recycled_total_limit, counter=role_counter + ('recycled_solves',)))
        gmres = (te.HostSolve(n_cells, 2, True),
                 te.HostStream(nnz * (COMPLEX128_BYTES + 4) + 2.0 * RECYCLED_GMRES_ITERATIONS_GUESS * COMPLEX128_BYTES
                               * n_cells), te.PythonCalls(10))
        iteration_total_limit = (None if recycled_total_limit is None
                                 else recycled_total_limit * RECYCLE_MAX_ITERATIONS)
        items.append(te.TimeItem('recycled GMRES iterations', 'iteration',
                                 recycled_count * RECYCLED_GMRES_ITERATIONS_GUESS, 'guess', gmres,
                                 limit=iteration_total_limit, counter=role_counter + ('gmres_iterations',)))

    not_estimated, adjoint_solves = (), 0.0
    if optimization:   # cascade adjoint: one transposed solve per channel; pump_depleted: GMRES of transposed sweeps
        items.append(te.TimeItem('adjoint gradients', 'gradient', outer[0] * n_channels, outer[1],
                                 (te.elementwise(n_cells, 8, 3, COMPLEX128_BYTES, launches=12), te.PythonCalls(40))))
        sweeps_per_gradient = ADJOINT_PRODUCTS_GUESS + 2.0 if pump_depleted else 1.0
        adjoint_solves = outer[0] * n_channels * sweeps_per_gradient
        if pump_depleted:
            items.append(te.TimeItem('pump-depleted adjoint sweeps', 'sweep', outer[0] * sweeps_per_gradient, 'guess',
                                     tuple(density) + krylov, limit=outer[0] * (ADJOINT_MAX_PRODUCTS + 2.0),
                                     counter=physics_counter + ('adjoint_sweeps',)))
    elif _enabled_features(physics_spec, 'adjoint_objective'):
        not_estimated = ('adjoint sensitivity (computed when eps_sensitivity is read)',)
    linear = context.solver_work('linear', unknowns=n_cells, nnz=nnz, value_bytes=COMPLEX128_BYTES,
                                 solves=channel_solves - recycled_count + adjoint_solves, solves_kind=kind,
                                 factorizations=outer[0] * n_channels, factorizations_kind=outer[1])
    return te.combine((te.TimeEstimate(tuple(items), not_estimated), linear))


def _memory_estimate(settings_values, mesh, context):
    """Every channel's FDFD estimate (each keeps its own system and factor) + the shared zeros, PEC masks, the kept
    nonlinear run items and the density / absorbed-power / cascade-adjoint transient phases (comments)."""
    physics_spec, n_cells = context.physics_spec, int(mesh.n_cells)
    channels, _ = _model_channels(physics_spec, context.parameters)
    channels = channels or ()
    parts = [fdfd_memory_estimate(channel.polarization)(settings_values, mesh, context).labelled(
        f'channel {channel.name}') for channel in channels]
    items = [('absent-polarisation zeros (float32)', FLOAT32_BYTES * n_cells)]
    if _enabled_features(physics_spec, 'pec_domain'):
        items.append((f'PEC masks ({len(channels)} channels x bool)', len(channels) * BOOL_BYTES * n_cells))
    # run items (kept): the channel field store, the cached polarisation densities (applied and diagnostic terms) and
    # hz face currents, cell areas + the absorbed-power total; pump_depleted the forward iteration buffers (Anderson
    # history + Kerr operator terms (picard), Krylov basis + Newton vectors + tangent fields (newton)): freed after the
    # solve but alive across the channels' (re)assembly, so not a phase of their own. Transient phases: one density
    # evaluation, take_absorbed_power, the cascade adjoint (after the forward solve, systems assembled)
    run_items, transient_items = [], []
    if channels:
        table = _mixing_table(physics_spec, context.parameters, channels)
        if table is not None:
            pump_depleted = settings_values['coupling_mode'] == 'pump_depleted'
            applied_terms = table.terms if pump_depleted else table.undepleted_terms()
            applied = {term.target for term in applied_terms}
            diagnostic = set() if pump_depleted else {term.target for term in table.terms}   # pump: shared phasors
            components = sum(len(channels[target].components) for target in applied) + sum(
                len(channels[target].components) for target in diagnostic)
            n_hz_targets = sum(1 for target in applied if channels[target].polarization == 'hz')
            n_hz_sources = sum(1 for channel in channels if channel.polarization == 'hz')
            store_arrays = sum(len(channel.components) for channel in channels)
            run_items = [(f'channel field store ({store_arrays} x complex128)',
                          store_arrays * COMPLEX128_BYTES * n_cells),
                         (f'nonlinear polarisation densities ({components} x complex128)',
                          components * COMPLEX128_BYTES * n_cells),
                         (f'hz face currents ({2 * n_hz_targets} x complex128)',
                          2 * n_hz_targets * COMPLEX128_BYTES * n_cells),
                         ('cell areas + absorbed power total', (FLOAT64_BYTES + FLOAT32_BYTES) * n_cells)]
            # NonlinearPolarizationPhasor.density: centre_vector's new collocated hz fields + box products, freed
            transient_items.append(('nonlinear densities', 'density temporaries (collocated hz fields, box products)',
                                    (2 * n_hz_sources + _DENSITY_TRANSIENT_ARRAYS) * COMPLEX128_BYTES * n_cells))
            if pump_depleted and settings_values['iteration_method'] == 'newton':
                buffers = GMRES_RESTART + 1 + NEWTON_BUFFERS
                # tangent product: centre fields at x (cached) and of v (2 per hz channel each), one target's dP and
                # its hz face currents
                tangent_arrays = 4 * n_hz_sources + 2 + 2
                run_items += [(f'Newton-Krylov basis and vectors ({buffers} x the store)',
                               buffers * store_arrays * COMPLEX128_BYTES * n_cells),
                              (f'Newton tangent fields ({tangent_arrays} x complex128)',
                               tangent_arrays * COMPLEX128_BYTES * n_cells)]
            elif pump_depleted:
                buffers = 2 * int(settings_values['anderson_depth']) + _ANDERSON_BUFFERS
                kerr_targets = {term.target for term in table.terms if term.kerr}
                if any(_raw_flag(feature, 'free_carriers') for feature in _enabled_features(physics_spec, _CHI3_FEATURE)):
                    kerr_targets = set(range(len(channels)))   # carriers: an operator term on every channel
                kerr_arrays = sum(_KERR_ARRAYS[channels[target].polarization] for target in kerr_targets)
                run_items += [(f'Anderson mixing ({buffers} x the store)',
                               buffers * store_arrays * COMPLEX128_BYTES * n_cells),
                              (f'Kerr operator terms ({kerr_arrays} x complex128)',
                               kerr_arrays * COMPLEX128_BYTES * n_cells)]
            if _enabled_features(physics_spec, 'adjoint_objective'):   # one EpsSensitivity per physics (eps_r,
                # eps_xx, eps_yy + float32 density), computed when read
                run_items.append(('cascade adjoint gradients (3 x float64 + float32)',
                                  (3 * FLOAT64_BYTES + FLOAT32_BYTES) * n_cells))
                temporaries = _CASCADE_ADJOINT_TEMPORARIES
                transient_items.append(('cascade adjoint', f'cascade adjoint temporaries ({temporaries} x complex128, '
                                        'approximate)', temporaries * COMPLEX128_BYTES * n_cells))
                if pump_depleted and settings_values['iteration_method'] != 'newton':   # newton: reuses the freed
                    buffers = GMRES_RESTART + 1 + _ADJOINT_BUFFERS                       # Newton basis blocks
                    transient_items.append(('cascade adjoint', f'pump-depleted adjoint GMRES basis and vectors '
                                            f'({buffers} x the store)',
                                            buffers * store_arrays * COMPLEX128_BYTES * n_cells))
            if pump_depleted:   # take_absorbed_power: lagged-loss and nonlinear absorption densities
                transient_items.append(('absorbed power', 'absorbed-power temporaries (float64 + 4 x complex128)',
                                        (FLOAT64_BYTES + 4 * COMPLEX128_BYTES) * n_cells))
                if _enabled_features(physics_spec, SHEET_NONLINEARITY.name):   # sheet SPM / XPM tap scatter
                    transient_items.append(('absorbed power', 'sheet absorption scatter (complex128 + 2 x float64)',
                                            (COMPLEX128_BYTES + 2 * FLOAT64_BYTES) * n_cells))
            else:   # undepleted: the float64 total, its float32 cast and one hz channel's collocated centre fields
                transient_items.append(('absorbed power', 'absorbed-power temporaries (float64 + float32 + hz '
                                        'centre fields)', (FLOAT64_BYTES + FLOAT32_BYTES
                                                           + (2 * COMPLEX128_BYTES if n_hz_sources else 0)) * n_cells))
    return combine(parts + [MemoryEstimate.from_items(items, run_items=run_items, transient_items=transient_items)])


def _scalar_outputs(settings_values, physics_spec):
    names = [f'power_exchange_{name}' for name in _channel_names(physics_spec) if is_identifier(name)]
    return tuple(dict.fromkeys(names + ['power_exchange_imbalance', 'nonlinear_absorption',
                                        'free_carrier_absorption', 'peak_carrier_density', 'iterations', 'residual']))


_SCALAR_OUTPUTS_HELP = (
    'power_exchange_<channel> in W/m: time-averaged power the nonlinear polarisation of the applied mixing terms '
    '(and the free carriers) delivers to that channel, W_k = (omega_k / 2) sum_cells A Im(conj(E_k) . P_k), per unit '
    'length in z (2D); > 0 = the channel gains power (a generated channel), < 0 = it loses power (a depleted pump, '
    'pump_depleted only), 0 on input channels in the undepleted mode (no back-action). power_exchange_imbalance (1): '
    '|sum_k W_k| / sum_k |W_k| over every term of the mixing table on the current fields, ~1e-12 for real tensors '
    'with Kleinman symmetry whatever the fields: an implementation check, not a convergence measure (larger for '
    'general / non-Kleinman d values, per-process or Miller factors, with Im chi3 and carriers, where the sum is the '
    'nonlinear loss). nonlinear_absorption in W/m: -W of the applied SPM / XPM terms of chi3 media and sigma3 sheets '
    '(two-photon absorption; 0 undepleted). free_carrier_absorption in W/m: -W of the free-carrier polarisation over '
    'every channel (> 0 absorbs; 0 without carriers). peak_carrier_density in 1/m^3: the largest steady-state N. iterations (1): sweeps '
    '(picard) or Newton steps (newton) of the last solve (1 undepleted, summed over continuation steps); residual (1): '
    "the last iterate's largest relative channel field change (0 undepleted).")
_DESCRIPTION = (
    '2D frequency-domain chi2 / chi3 mixing between harmonic channels (nonlinear_optics_plan stages 1-4): every '
    'harmonic_channel feature is one em_fdfd_ez or em_fdfd_hz solve (E_z or H_z polarisation, both = two channels) at '
    'its own frequency on the shared mesh and materials, each material evaluated at that frequency, so phase matching '
    'comes from the material dispersion. chi2_medium / chi3_medium features add the nonlinear polarisation as a phasor '
    'source of the target channels (exp(+j omega t), P = eps0 2^(1-n) sum chi : E..E; SHG P(2w) = eps0 d E^2, THG '
    'P(3w) = (1/4) eps0 chi3 E^3); surface_chi2 (the interface of a selection) and sheet_nonlinearity (a 2D material '
    'on a line) add surface polarisations on cell faces / line cells. coupling_mode undepleted: solved once per '
    'channel in rank order from the lower '
    'ranks (no pump back-action, no SPM / XPM), valid while the conversion stays small; pump_depleted: every term '
    '(depletion, parametric gain, SPM / XPM, steady-state free carriers of chi3 media) solved self-consistently, by '
    'block Gauss-Seidel sweeps with relaxation and Anderson acceleration (iteration_method picard; the Kerr diagonal '
    'lagged in the channel operator; high-gain parametric or bistable cavities may not converge) or by Newton-Krylov '
    '(newton: exact Jacobian, GMRES preconditioned by the channel factors), with optional source continuation, until '
    'the fields change by <= tolerance. Sources and '
    'monitors are the em_fdfd_ez / _hz features with a channel setting; fields and '
    'powers are absolute and per unit length in z, so nonlinear results scale with the source amplitudes (a 2D SHG '
    'efficiency P_2w / P_w^2 has unit m/W). Each channel factorises its own operator (host SuperLU, MUMPS or PARDISO '
    'memory adds up per channel). Study: nonlinear_frequency_domain (one snapshot per channel). Output fields of the shown channel: '
    'the phasor quartets of both polarisations (the other polarisation reads 0), e_abs (primary), e_intensity, h_abs, '
    'absorbed_power, nonlinear_polarization_abs |P_NL|, the chi2 poling sign and carrier_density (0 without free '
    'carriers), eps_sensitivity (the cascade adjoint of the adjoint_objective features) and design_density. '
    'Couplings: thermo-optic and other permittivity writers act on every channel; em_heat_joule reads the absorbed '
    'power of every channel (linear q_abs plus the exact two-photon and free-carrier absorption of the applied terms, '
    'the same for picard and newton) back into a background heat physics (solution feedback rounds of the study). '
    'Studies: nonlinear_frequency_domain, and optimization (coupling_mode undepleted, or pump_depleted with the '
    'fixed-point adjoint, iteration_method newton when a chi3 medium is present: design_region on the shared '
    'permittivity, adjoint_objective features on the channels). Periodic faces carry a Bloch phase per channel '
    '(harmonic_channel bloch_kx / bloch_ky of the inputs; generated channels the combination of their inputs\' k), '
    'mirror faces a parity per channel (harmonic_channel). The instance is usually named \'nl\'.')

EM_FDFD_NONLINEAR = PhysicsType(
    name=TYPE_NAME,
    label='Nonlinear optics, frequency domain (2D, harmonic channels)',
    family=EM_FAMILY,
    group='em',
    spaces_2d=('planar',),
    default_instance_name='nl',
    description=_DESCRIPTION,
    summary='Light in 2D at several wavelengths coupled by second- or third-order nonlinear materials: second and '
            'third harmonic, sum and difference frequencies, Kerr effect and parametric gain, with quasi phase '
            'matching by periodic poling.',
    settings=_SETTINGS,
    features=(_HARMONIC_CHANNEL, _CHI2_MEDIUM, _CHI3_MEDIUM, SURFACE_CHI2, SHEET_NONLINEARITY, _PEC_DOMAIN,
              _CONDUCTING_SHEET, _DESIGN_REGION, *(_channel_feature_type(name) for name in _CHANNEL_FEATURE_NAMES)),
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS['modes'],
    state_fields=tuple(_Q_ABS_INFO if name == 'q_abs' else replace(NONLINEAR_FIELD_INFOS[name], primary=False)
                       for name in _STATE_FIELD_NAMES),
    output_fields=tuple(NONLINEAR_FIELD_INFOS.values()),
    source_slots=source_slot_docs(_SLOT_ARRAYS),
    supported_studies=(NONLINEAR_STUDY, OPTIMIZATION_STUDY),
    default_study={'type': NONLINEAR_STUDY},
    uses_dispersion=True,
    create=_create,
    domain_requirements=quasi_static_domain_requirements,
    shortest_wavelength=_shortest_channel_wavelength,
    validate=_validate_physics,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    scalar_outputs=_scalar_outputs,
    scalar_outputs_help=_SCALAR_OUTPUTS_HELP,
    solver_roles={'linear': DIRECT_LINEAR_ROLE},
    backends=('gpu', 'cpu'),   # cpu: the channels' scipy assembly / SuperLU, numpy sources
)


def _selftest_cases():
    """The CPU and GPU SelfTestCases of NLFD-chi, last the CPU vs GPU ones (imported on the first registry.selftests())."""
    from .selftests import NLFD_CPU_SELFTESTS, NLFD_GPU_SELFTESTS
    from .selftests_adjoint import CASCADE_ADJOINT_SELFTESTS
    from .selftests_backend import NLFD_BACKEND_SELFTESTS
    from .selftests_feedback import FEEDBACK_SELFTESTS
    from .selftests_sheets import SHEET_CPU_SELFTESTS, SHEET_GPU_SELFTESTS
    return (NLFD_CPU_SELFTESTS + NLFD_GPU_SELFTESTS + FEEDBACK_SELFTESTS + CASCADE_ADJOINT_SELFTESTS
            + SHEET_CPU_SELFTESTS + SHEET_GPU_SELFTESTS + NLFD_BACKEND_SELFTESTS)


def register(registry):
    """Register em_fdfd_nonlinear, its templates, then the selftest loader."""
    registry.register_physics(EM_FDFD_NONLINEAR)
    for template in NLFD_TEMPLATES + SHEET_TEMPLATES + ADJOINT_TEMPLATES:
        registry.register_template(*template)
    registry.register_selftests(_selftest_cases)

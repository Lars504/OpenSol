"""Model-layer pieces of the 2D solid_mechanics frequency response and anisotropic stiffness (piezoelectric plan stage
5): the admittance feature, their validation rules and the host memory items; imported by definition.py."""
from ...core.memory_estimate import FLOAT64_BYTES, INT32_BYTES, host_lu_bytes
from ...model.definition_helpers import evaluates_to, is_axisymmetric, on_tetrahedra, on_triangles
from ...model.schema import FeatureType, Issue, Param, feature_path
from .bvd import FIT_KEYS
from .definition_common import FREQUENCY_RESPONSE_STUDY_NAME as FREQUENCY_RESPONSE_STUDY
from .definition_common import merged_materials, modal_study_issues
from .frequency_response import ADMITTANCE_SERIES, AdmittanceMonitor, admittance_result, driven_electrode
from .materials import LOSS_ARRAYS, SHEAR_COUPLING_ARRAYS, STIFFNESS_TENSOR_ARRAYS

_COMPLEX_BYTES = 16
_ENTRIES_PER_DOF = 27   # 3 x 3 neighbour nodes x 3 components (the coupled pattern; 18 without piezo)


# ---- admittance feature ----
def _apply_admittance(ctx, solver, feature_spec, values, selection):
    """AdmittanceMonitor of the feature (None when piezoelectric is off: validation errors)."""
    if not solver.piezoelectric:
        return None
    return AdmittanceMonitor(feature_spec.name, values['electrode'], values['fit_frequency_min'],
                             values['fit_frequency_max'], values['load_impedance'])


def _admittance_result(session, solver, monitor, feature_spec, options):
    if monitor is None:
        raise ValueError(f"admittance '{feature_spec.name}': piezoelectric is off, so there is no electrode")
    electrode = driven_electrode(monitor, solver.electrodes, solver.system.label)
    # solid_mechanics_3d has no thickness setting: its charges are in C, so Y is in S (depth factor 1)
    return admittance_result(monitor, electrode, solver.frequency_points, getattr(solver, 'thickness', 1.0))


ADMITTANCE_FEATURE = FeatureType(
    'admittance', 'Admittance', 'monitor', None, result_kind='xy',
    params=(Param('electrode', 'str', '', label='Electrode',
                  help='Name of the driven Electrode / Electrode (domain) feature (connection voltage, voltage != 0) '
                       'whose admittance Y = I / V is reported; empty = the only voltage electrode with a non-zero '
                       'voltage (an error when there are several).',
                  tip='Name of the electrode feature that drives the resonator. Empty takes the only electrode with a '
                      'voltage other than 0.'),
            Param('fit_frequency_min', 'float', None, label='Fit band start', unit='Hz', min=0.0, optional=True,
                  help='Lowest frequency of the Butterworth-Van Dyke fit; empty = the lowest solved frequency. Choose '
                       'a band around one resonance / antiresonance pair (other modes in the band bias the fit).',
                  tip='Lowest frequency used to fit the equivalent circuit; empty uses the whole sweep.'),
            Param('fit_frequency_max', 'float', None, label='Fit band stop', unit='Hz', min=0.0, optional=True,
                  help='Highest frequency of the Butterworth-Van Dyke fit; empty = the highest solved frequency.',
                  tip='Highest frequency used to fit the equivalent circuit; empty uses the whole sweep.'),
            Param('load_impedance', 'float', 50.0, label='Load impedance', unit='Ohm', min=0.0, min_exclusive=True,
                  help='Resistive load Z of the external circuit for the extra kappa_e = Z / L1 [rad/s], the energy '
                       'decay rate the load adds to the fitted motional branch (valid while the extra omega_C0_Z = '
                       'omega_s C0 Z << 1). Needs C0 / L1 in F / H: a 3D model, or the thickness setting in 2D (per m '
                       'of depth kappa_e is NaN). Default 50 Ohm.',
                  tip='Resistance of the circuit the resonator is connected to, for the external coupling rate. '
                      'Usually 50 ohm.')),
    summary='Electrical admittance of a piezoelectric resonator versus frequency, with its equivalent circuit '
            '(Butterworth-Van Dyke) fitted. Needs a frequency response study.',
    description="frequency_response study, piezoelectric only. Result 'xy' versus frequency [Hz] of Y = I / V = j "
                'omega Q / V of the electrode (Q = the complex charge, the discrete Gauss law of D including the '
                f"dielectric loss): series {', '.join(ADMITTANCE_SERIES)} (|Y|, Re Y, Im Y in S/m per m of depth, or S "
                'with the thickness setting and in solid_mechanics_3d; phase in deg; bvd_abs = |Y| of the fitted '
                'model, NaN without a fit). '
                'Extras (derived values read them with operation last): the Butterworth-Van Dyke fit Y = j omega C0 (1 '
                '- j tan_delta_0) + 1 / (R1 + j omega L1 + 1 / (j omega C1)) over the fit band (least squares on the '
                'relative complex residual from a closed-form start: f_s = the sample of max |Y|, f_p = the sample of '
                f"min |Y| above it): {', '.join(FIT_KEYS)} (f_s = 1 / (2 pi sqrt(L1 C1)), f_p = f_s sqrt(1 + C1 / C0) "
                'the lossless antiresonance, k_eff2 = (f_p^2 - f_s^2) / f_p^2 = C1 / (C0 + C1), Q_s = 2 pi f_s L1 / '
                'R1, f_peak / f_dip the sampled max / min of |Y|, fit_error the rms relative residual), kappa_e = '
                'load_impedance / L1 [rad/s] (the external coupling rate into a resistive load in series with the '
                'motional branch, internal R1 / L1 = 2 pi f_s / Q_s) and omega_C0_Z = 2 pi f_s C0 load_impedance (kappa_e '
                'holds while it is << 1; both NaN per m of depth, kappa_e_warning), load_impedance, bvd_units, '
                'electrode; when the band holds fewer than 6 points or no peak with a dip above it, the fit values are '
                'NaN and fit_warning says why. One resonance per band: resolve it with a step well below f_s / Q_s.',
    apply=_apply_admittance, result=_admittance_result)


# ---- validation (CPU) ----
def _selected_properties(vctx):
    """[(material name, merged properties)] of the selected spec materials that merge (others are reported at the
    material)."""
    return [(material.name, properties) for material, properties in merged_materials(vctx)
            if material.selection and properties is not None]


def anisotropy_issues(vctx, physics_spec, settings_values, path, guided):
    """Per selected material with c_ij entries: an error for guided modes (isotropic only) and, axisymmetric, for a
    non-zero c16 / c26 / c36 (plain numbers); an info naming the entries read."""
    issues = []
    for name, properties in _selected_properties(vctx):
        entries = [key for key in STIFFNESS_TENSOR_ARRAYS if properties.get(key) is not None]
        if not entries:
            continue
        material_path = ('material', name)
        if guided:
            issues.append(Issue('error', material_path, f"material '{name}' sets c_ij entries ({', '.join(entries)}): "
                                                        'guided (2.5D) modes of solid_mechanics are isotropic only '
                                                        '(youngs_modulus / poisson_ratio)'))
            continue
        shear = [key for key in SHEAR_COUPLING_ARRAYS
                 if evaluates_to(properties.get(key), vctx.parameters, lambda value: value != 0.0)]
        if is_axisymmetric(settings_values) and shear:
            issues.append(Issue('error', material_path, f"material '{name}' sets {', '.join(shear)} != 0: an "
                                                        'axisymmetric solid_mechanics takes orthotropic c_ij only'))
            continue
        issues.append(Issue('info', material_path, f"material '{name}': {physics_spec.name} reads the anisotropic "
                                                   f"stiffness entries {', '.join(entries)} (unset ones: the isotropic "
                                                   'value of youngs_modulus / poisson_ratio)'))
    return issues


def frequency_response_issues(vctx, physics_spec, settings_values, path, thermal_strain_coupling):
    """Admittance features (error without piezoelectric or naming no voltage electrode, warning outside the study);
    in the frequency_response study: generalized plane strain (error), rho of the solid materials (errors); an info for
    loss entries outside it."""
    issues = []
    piezoelectric = bool(settings_values.get('piezoelectric'))
    voltage_electrodes = set()
    for feature in physics_spec.features:
        if feature.enabled and feature.type in ('electrode', 'electrode_domain'):
            connection = (feature.params or {}).get('connection', 'voltage') if isinstance(feature.params, dict) \
                else 'voltage'
            if connection == 'voltage':
                voltage_electrodes.add(feature.name)
    for feature in physics_spec.features:
        if not (feature.enabled and feature.type == ADMITTANCE_FEATURE.name):
            continue
        feature_issue_path = feature_path(physics_spec, feature)
        if not piezoelectric:
            issues.append(Issue('error', feature_issue_path, 'an admittance needs the piezoelectric setting on (it '
                                                             'reads the charge of an electrode)'))
        elif vctx.study_type != FREQUENCY_RESPONSE_STUDY:
            issues.append(Issue('warning', feature_issue_path, f'admittance is filled only by a '
                                                               f'{FREQUENCY_RESPONSE_STUDY} study (this model runs '
                                                               f'{vctx.study_type}): its result will be missing'))
        electrode = (feature.params or {}).get('electrode', '') if isinstance(feature.params, dict) else ''
        if piezoelectric and electrode and electrode not in voltage_electrodes:
            issues.append(Issue('error', feature_issue_path, f"admittance electrode '{electrode}' is no enabled "
                                                             'Electrode / Electrode (domain) feature with connection '
                                                             f"voltage of {physics_spec.name}; voltage electrodes: "
                                                             f"{', '.join(sorted(voltage_electrodes)) or 'none'}"))
    if vctx.study_type == FREQUENCY_RESPONSE_STUDY:
        if settings_values.get('out_of_plane') == 'generalized_plane_strain' and not is_axisymmetric(settings_values):
            issues.append(Issue('error', path, 'generalized plane strain has no frequency response (the uniform eps_zz '
                                               'carries no mass): use plane_strain or plane_stress'))
        issues.extend(modal_study_issues(vctx, physics_spec, path, (FREQUENCY_RESPONSE_STUDY,),
                                         thermal_strain_coupling, tensor=True, prestress_info=False))
        settings = vctx.spec.study.settings if isinstance(vctx.spec.study.settings, dict) else {}
        if piezoelectric and settings.get('sweep_method') == 'modal':
            issues.append(Issue('error', path, "sweep_method 'modal' needs a purely elastic pencil (the electrodes "
                                               "couple a massless potential): use sweep_method 'adaptive'"))
    else:
        lossy = sorted({name for _, properties in _selected_properties(vctx) for name in LOSS_ARRAYS
                        if properties.get(name) is not None})
        fem = on_triangles(vctx, physics_spec) or on_tetrahedra(vctx, physics_spec)
        eigenfrequency = vctx.study_type == 'eigenfrequency'
        if eigenfrequency and fem and not piezoelectric:   # K3: the damped modes read eta
            lossy = [name for name in lossy if name != LOSS_ARRAYS[0]]
        if lossy:
            note = ('; on triangle and tetrahedron meshes mechanical_loss_factor damps the modes'
                    if eigenfrequency and not fem else '')
            issues.append(Issue('info', path, f"{', '.join(lossy)} play(s) a part only in a {FREQUENCY_RESPONSE_STUDY} "
                                              f'study (the {vctx.study_type} study is lossless{note})'))
    return issues


# ---- memory estimate (CPU) ----
def frequency_response_host_items(dofs):
    """Host items of one frequency of the frequency response on `dofs` unknowns (2 or 3 per node): the float64 K /
    loss / M copies, the complex total and reduced operators and the complex SuperLU factor (heuristic)."""
    entries = _ENTRIES_PER_DOF * dofs
    real_entry, complex_entry = FLOAT64_BYTES + INT32_BYTES, _COMPLEX_BYTES + INT32_BYTES
    return (('frequency response host K, loss, M (float64 CSR)', 3 * real_entry * entries),
            ('frequency response complex operator copies', 3 * complex_entry * entries),
            ('frequency response complex SuperLU L + U (heuristic)', host_lu_bytes(dofs)),
            ('frequency response solution vectors (complex)', 4 * _COMPLEX_BYTES * dofs))

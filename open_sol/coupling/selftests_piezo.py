"""Piezoelectric self-tests (crystal tensor data / rotations, electrostatics waveform superposition, one-way
thickness-slab launch, two-way bound charge of a strained slab and the stiffened bar resonances through
electrostatics_acoustics_piezo) and the 'yz_ln_saw_idt' / 'tfln_acousto_optic_modulator' templates; specs built fresh,
no physics imports at module level (the CPU check imports the crystal tensors inside)."""
import math

import numpy as np

from ..core.constants import EPS0
from ..model.schema import SelfTestCase
from ..model.selftest_helpers import cell_centre, vertical_line, with_exact_cg
from ..model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec,
                          ModelSpec, ParameterSpec, PhysicsSpec, SolverSpec, StudySpec)

PIEZO_TYPE_NAME = 'electrostatics_acoustics_piezo'   # = definitions_piezo (which imports this module)
_ACOUSTO_OPTIC_TYPE_NAME = 'acoustics_em_acousto_optic'   # = selftests_acoustics.ACOUSTO_OPTIC_TYPE_NAME
_POCKELS_TYPE_NAME = 'electrostatics_em_pockels'          # = selftests_pockels.POCKELS_TYPE_NAME
_ACOUSTICS = 'acoustics'
_ELECTROSTATICS = 'electrostatics'
_EM = 'em'


def _rectangle(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _point(name, x, y):
    return GeometrySpec(name, 'point', {'x': x, 'y': y})


# ---- 1. piezo_crystal_tensors (CPU) ----
_TENSOR_TOLERANCE = 1e-12
_GENERIC_EULER = (31.0, 47.0, 73.0)        # deg: no symmetry, every Bond entry non-trivial
_YZ_EULER = (0.0, 90.0, 90.0)              # YZ: X1' = Z, X3' = -Y, so model (x, y, z) = (Z, -Y, X)
_X_CUT_Y_EULER = (90.0, 90.0, 0.0)         # X-cut, Y-propagating: model plane XY, not a mirror plane of 3m
_MIN_ANTIPLANE_RATIO = 1e-3


def _about_z(angle_deg):
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# point-group generators per crystal class (orthogonal matrices; mirrors and -4 are improper)
_CLASS_OPERATIONS = {
    '3m': (('3-fold z', _about_z(120.0)), ('mirror normal to x', np.diag([-1.0, 1.0, 1.0]))),
    '32': (('3-fold z', _about_z(120.0)), ('2-fold x', np.diag([1.0, -1.0, -1.0]))),
    '6mm': (('6-fold z', _about_z(60.0)), ('mirror normal to x', np.diag([-1.0, 1.0, 1.0]))),
    '-43m': (('-4 about z', -_about_z(90.0)), ('3-fold [111]', np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                                                                         [0.0, 1.0, 0.0]])),
             ('mirror (1-10)', np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))),
}


def _relative(measured, expected):
    """max |measured - expected| / max |expected| (absolute when expected is 0)."""
    measured, expected = np.asarray(measured, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected))) or 1.0
    return float(np.max(np.abs(measured - expected))) / scale


def _yz_expected(data):
    """Hand-written sagittal YZ-LN entries (model x = Z, y = -Y) from the crystal constants, + (c16, c26)."""
    c, e, (eps11, eps33) = data.stiffness, data.piezo, data.permittivity_clamped
    properties = {'rho': data.rho, 'c11': c['c33'], 'c12': c['c13'], 'c22': c['c11'], 'c66': c['c44'],
                  'c16': 0.0, 'c26': c['c14'],
                  'piezo_e11': e['e33'], 'piezo_e12': e['e31'], 'piezo_e16': 0.0, 'piezo_e21': 0.0,
                  'piezo_e22': -e['e22'], 'piezo_e26': e['e15'], 'eps_dc_xx': eps33, 'eps_dc_yy': eps11,
                  'eps_dc_xy': 0.0}
    return properties, (0.0, c['c14'])


def _yz_optical_expected(data):
    """Hand-written sagittal YZ-LN photoelastic / Pockels entries (model x = Z, y = -Y, z = X; xy row = -crystal row 4:
    p61 = 0, p62 = p41, p66 = p44, r61 = 0, r62 = r51) and the xy-row ratios (the anti-plane yz / xz rows are 0)."""
    p, r = data.photoelastic, data.pockels
    properties = {'photoelastic_p11': p['p33'], 'photoelastic_p12': p['p31'], 'photoelastic_p16': 0.0,
                  'photoelastic_p21': p['p13'], 'photoelastic_p22': p['p11'], 'photoelastic_p26': p['p14'],
                  'photoelastic_p31': p['p13'], 'photoelastic_p32': p['p12'], 'photoelastic_p36': -p['p14'],
                  'photoelastic_p61': 0.0, 'photoelastic_p62': p['p41'], 'photoelastic_p66': p['p44'],
                  'pockels_r11': r['r33'], 'pockels_r12': 0.0, 'pockels_r21': r['r13'], 'pockels_r22': -r['r22'],
                  'pockels_r31': r['r13'], 'pockels_r32': r['r22'], 'pockels_r61': 0.0, 'pockels_r62': r['r51']}
    photoelastic_shear = max(abs(p['p41']), abs(p['p44']))
    photoelastic_all = max(abs(properties[name]) for name in properties if 'photoelastic' in name)
    pockels_all = max(abs(properties[name]) for name in properties if 'pockels' in name)
    return properties, (photoelastic_shear / photoelastic_all, abs(r['r51']) / pockels_all)


def _cubic_photoelastic_expected(data):
    """photoelastic_p11 .. _p66 of a cubic crystal at euler (0, 0, 0): its p11, p12 on the diagonal rows, p66 = p44
    (not the isotropic (p11 - p12) / 2, so the tensor entries are written), the shear columns / xy normal terms 0."""
    p11, p12, p44 = (data.photoelastic[name] for name in ('p11', 'p12', 'p44'))
    return {'photoelastic_p11': p11, 'photoelastic_p12': p12, 'photoelastic_p16': 0.0, 'photoelastic_p21': p12,
            'photoelastic_p22': p11, 'photoelastic_p26': 0.0, 'photoelastic_p31': p12, 'photoelastic_p32': p12,
            'photoelastic_p36': 0.0, 'photoelastic_p61': 0.0, 'photoelastic_p62': 0.0, 'photoelastic_p66': p44}


def _optical_checks(errors, failed):
    """Class symmetry of the p / r data, Bond vs index rotation, the YZ-LN optical entries and ratios, the conditional
    entries of an isotropic cut (GaAs at euler 0, 0, 0)."""
    from ..materials.crystal_tensors import (CRYSTALS, full_piezo, full_stiffness, optical_tensors, rotate_optical,
                                             sagittal_crystal, substrate_rotation, voigt_piezo, voigt_stiffness)
    from ..physics.nlfd.tensors import rotate_tensor
    for name, data in CRYSTALS.items():
        photoelastic, pockels = optical_tensors(name)
        for label, operation in _CLASS_OPERATIONS[data.crystal_class]:
            rotated_p, rotated_r = rotate_optical(photoelastic, pockels, operation)
            if photoelastic is not None:
                errors[f'{name} p: {label}'] = _relative(rotated_p, photoelastic)
            if pockels is not None:
                errors[f'{name} r: {label}'] = _relative(rotated_r, pockels)
    photoelastic, pockels = optical_tensors('lithium_niobate')
    rotation = substrate_rotation(*_GENERIC_EULER)
    bond_p, bond_r = rotate_optical(photoelastic, pockels, rotation)
    errors['optical bond vs index rotation'] = max(
        _relative(bond_p, voigt_stiffness(rotate_tensor(full_stiffness(photoelastic), rotation))),
        _relative(bond_r, voigt_piezo(rotate_tensor(full_piezo(pockels.T), rotation)).T))
    yz = sagittal_crystal({'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': list(_YZ_EULER)})
    expected, (photoelastic_ratio, pockels_ratio) = _yz_optical_expected(CRYSTALS['lithium_niobate'])
    for group in ('photoelastic', 'pockels'):
        names = [name for name in expected if group in name]
        errors[f'YZ {group}'] = _relative([yz.properties.get(name, math.inf) for name in names],
                                          [expected[name] for name in names])
    errors['YZ xy-row ratios'] = max(abs(yz.shear_photoelastic_ratio - photoelastic_ratio),
                                     abs(yz.shear_pockels_ratio - pockels_ratio))
    if max(yz.antiplane_photoelastic_ratio, yz.antiplane_pockels_ratio) > _TENSOR_TOLERANCE:
        failed.append(f'YZ anti-plane optical ratios {yz.antiplane_photoelastic_ratio:.3g} / '
                      f'{yz.antiplane_pockels_ratio:.3g} != 0')
    cubic = sagittal_crystal({'model': 'crystal', 'crystal': 'gallium_arsenide', 'euler_deg': [0.0, 0.0, 0.0]})
    written = sorted(name for name in cubic.properties if name.startswith('pockels'))
    if written:
        failed.append(f'GaAs at euler 0: wrote {written} (r41 lies only in the anti-plane rows)')
    cubic_expected = _cubic_photoelastic_expected(CRYSTALS['gallium_arsenide'])
    errors['GaAs photoelastic (p66 = p44)'] = _relative([cubic.properties.get(name, math.inf) for name in cubic_expected],
                                                        list(cubic_expected.values()))
    if abs(cubic.antiplane_pockels_ratio - 1.0) > _TENSOR_TOLERANCE:
        failed.append(f'GaAs at euler 0: Pockels anti-plane ratio {cubic.antiplane_pockels_ratio:.3g} != 1')
    return yz


def _check_crystal_tensors(results):
    """Class symmetry of every crystal's data, Bond rotation vs explicit index rotation, the YZ-LN block, the decoupling
    ratios, the same for the photoelastic / Pockels data; error = the worst relative error (+ 1 per failed check)."""
    from ..materials.crystal_tensors import (CRYSTALS, crystal_tensors, full_piezo, full_stiffness, rotate_voigt,
                                             sagittal_crystal, substrate_rotation, voigt_piezo, voigt_stiffness)
    from ..physics.nlfd.tensors import rotate_tensor
    errors, failed = {}, []
    for name, data in CRYSTALS.items():
        stiffness, piezo, permittivity, _ = crystal_tensors(name)
        for label, operation in _CLASS_OPERATIONS[data.crystal_class]:
            rotated = rotate_voigt(stiffness, piezo, permittivity, operation)
            errors[f'{name}: {label}'] = max(_relative(rotated[0], stiffness), _relative(rotated[1], piezo),
                                             _relative(rotated[2], permittivity))
    stiffness, piezo, permittivity, _ = crystal_tensors('lithium_niobate')
    rotation = substrate_rotation(*_GENERIC_EULER)
    bond_c, bond_e, bond_eps = rotate_voigt(stiffness, piezo, permittivity, rotation)
    errors['bond vs index rotation'] = max(
        _relative(bond_c, voigt_stiffness(rotate_tensor(full_stiffness(stiffness), rotation))),
        _relative(bond_e, voigt_piezo(rotate_tensor(full_piezo(piezo), rotation))),
        _relative(bond_eps, rotate_tensor(permittivity, rotation)))
    errors['voigt round trip'] = max(_relative(voigt_stiffness(full_stiffness(stiffness)), stiffness),
                                     _relative(voigt_piezo(full_piezo(piezo)), piezo))
    yz = sagittal_crystal({'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': list(_YZ_EULER)})
    expected, (c16, c26) = _yz_expected(CRYSTALS['lithium_niobate'])
    groups = {'stiffness': ('c11', 'c12', 'c22', 'c66', 'c16', 'c26'), 'piezo': tuple(name for name in expected if 'piezo' in name),
              'permittivity': ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_xy'), 'density': ('rho',)}
    for group, names in groups.items():
        errors[f'YZ {group}'] = _relative([yz.properties[name] for name in names], [expected[name] for name in names])
    errors['YZ c16 / c26'] = max(abs(yz.c16 - c16), abs(yz.c26 - c26)) / expected['c22']
    if max(yz.antiplane_stiffness_ratio, yz.antiplane_piezo_ratio) > _TENSOR_TOLERANCE:
        failed.append(f'YZ anti-plane ratios {yz.antiplane_stiffness_ratio:.3g} / {yz.antiplane_piezo_ratio:.3g} != 0')
    x_cut = sagittal_crystal({'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': list(_X_CUT_Y_EULER)})
    x_cut_ratio = max(x_cut.antiplane_stiffness_ratio, x_cut.antiplane_piezo_ratio)
    if x_cut_ratio < _MIN_ANTIPLANE_RATIO:
        failed.append(f'X-cut / Y-propagating anti-plane ratio {x_cut_ratio:.3g} < {_MIN_ANTIPLANE_RATIO:g}')
    yz_optical = _optical_checks(errors, failed)
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst] + len(failed),
            'tolerance': _TENSOR_TOLERANCE,
            'details': {'errors': errors, 'worst': worst, 'failed_checks': failed,
                        'yz_properties': dict(yz.properties), 'yz_c16_c26': [yz.c16, yz.c26],
                        'x_cut_antiplane_ratios': [x_cut.antiplane_stiffness_ratio, x_cut.antiplane_piezo_ratio],
                        'yz_xy_row_ratios': [yz_optical.shear_photoelastic_ratio, yz_optical.shear_pockels_ratio]}}


# ---- 1b. piezo_crystal_full_expansion (CPU) ----
_FULL_EULERS = (_GENERIC_EULER, _X_CUT_Y_EULER, _YZ_EULER)
_FULL_TOLERANCE = 1e-11   # the expansions zero rotated entries below 1e-12 of their tensor's largest (ZERO_RELATIVE)


def _check_crystal_full_expansion(results):
    """The 'full' entry vs the explicit index rotation, its sagittal restriction vs the 'sagittal' entry, the property
    names / values, the plane errors and the precedence crystal < user; error = the worst relative error (+ 1 per failed
    check)."""
    from ..materials import effective_material_data
    from ..materials.crystal_tensors import (CRYSTALS, FULL_PERMITTIVITY_PROPERTIES, FULL_PIEZO_PROPERTIES,
                                             FULL_STIFFNESS_PROPERTIES, SAGITTAL_AXES, crystal_properties,
                                             crystal_tensors, full_crystal, full_piezo, full_stiffness, model_tensors,
                                             rotate_voigt, sagittal_crystal, substrate_rotation, voigt_piezo,
                                             voigt_stiffness)
    from ..physics.nlfd.tensors import rotate_tensor
    errors, failed = {}, []
    for name in CRYSTALS:
        stiffness, piezo, permittivity, rho = crystal_tensors(name)
        for euler in _FULL_EULERS:
            entry = {'model': 'crystal', 'crystal': name, 'euler_deg': list(euler), 'plane': 'full'}
            expanded = full_crystal(entry)
            rotation = substrate_rotation(*euler)
            label = f'{name} {list(euler)}'
            errors[f'{label}: index rotation'] = max(
                _relative(expanded.stiffness, voigt_stiffness(rotate_tensor(full_stiffness(stiffness), rotation))),
                _relative(expanded.piezo, voigt_piezo(rotate_tensor(full_piezo(piezo), rotation))),
                _relative(expanded.permittivity, rotate_tensor(permittivity, rotation)))
            restricted = rotate_voigt(expanded.stiffness, expanded.piezo, expanded.permittivity, SAGITTAL_AXES)
            sagittal = model_tensors(name, euler)
            errors[f'{label}: sagittal restriction'] = max(_relative(measured, reference)
                                                           for measured, reference in zip(restricted, sagittal[:3]))
            reduced = sagittal_crystal({**entry, 'plane': 'sagittal'}).properties
            restricted_c = [restricted[0][0, 0], restricted[0][0, 1], restricted[0][1, 1], restricted[0][5, 5]]
            restricted_eps = [restricted[2][0, 0], restricted[2][1, 1], restricted[2][0, 1]]
            errors[f'{label}: sagittal properties'] = max(
                _relative(restricted_c, [reduced[key] for key in ('c11', 'c12', 'c22', 'c66')]),
                _relative(restricted_eps, [reduced[key] for key in ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_xy')]))
            properties = crystal_properties(entry)
            expected = {'rho': rho, **{key: expanded.permittivity[pair]
                                       for key, pair in FULL_PERMITTIVITY_PROPERTIES.items()}}
            expected.update({key: expanded.stiffness[int(key[1]) - 1, int(key[2]) - 1]
                             for key in FULL_STIFFNESS_PROPERTIES})
            expected.update({key: expanded.piezo[int(key[-2]) - 1, int(key[-1]) - 1] for key in FULL_PIEZO_PROPERTIES})
            if set(properties) != set(expected):
                failed.append(f'{label}: property names {sorted(set(properties) ^ set(expected))} differ')
            elif any(properties[key] != float(value) for key, value in expected.items()):
                failed.append(f'{label}: property values differ from the expanded tensors')
            if not np.array_equal(expanded.stiffness, expanded.stiffness.T):
                failed.append(f'{label}: c not symmetric')
    entry = {'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': list(_X_CUT_Y_EULER), 'plane': 'full'}
    for function, wrong in ((full_crystal, {**entry, 'plane': 'sagittal'}), (sagittal_crystal, entry)):
        try:
            function(wrong)
            failed.append(f"{function.__name__} accepted plane '{wrong['plane']}'")
        except ValueError:
            pass
    merged, poles = effective_material_data('lithium_niobate_x_cut', {'c44': 1.0e11, 'c11': None}, [entry])
    reference = full_crystal(entry).properties
    if merged.get('c44') != 1.0e11 or merged.get('c11') != reference['c11'] or merged.get('c14') != reference['c14']:
        failed.append(f"precedence library < crystal < user: c44 {merged.get('c44')}, c11 {merged.get('c11')} "
                      f"(crystal {reference['c11']}), c14 {merged.get('c14')}")
    if poles:
        failed.append(f'the crystal entry was left in the poles: {poles}')
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst] + len(failed),
            'tolerance': _FULL_TOLERANCE, 'details': {'errors': errors, 'worst': worst, 'failed_checks': failed}}


# ---- 2. electrostatics_waveform_superposition ----
# 40 x 20 cells of 1 um; electrostatics (float64 CG to 1e-12): terminal drive_a (ricker), voltage drive_b on x_hi
# (cw), terminal bias (constant), ground on y_lo, a space charge in a high-eps block; an elastic_waves fluid (rigid
# faces, no source) only supplies the time step. Static runs give the DC part and the unit responses of the two drives.
_WAVE_CELL = 1e-6
_WAVE_SIZE = (40e-6, 20e-6)
_WAVE_DRIVES = {'drive_a': ('ricker', 2e8, 2.0), 'drive_b': ('cw', 6e7, -1.0)}   # waveform, frequency [Hz], voltage
_BIAS_VOLTAGE = 0.5
_SPACE_CHARGE = 0.2                         # C/m^3
_WAVE_PROBES = {'probe_1': (15.5e-6, 10.5e-6), 'probe_2': (30.5e-6, 14.5e-6)}
_WAVE_DURATION = 4e-8
_WAVE_TOLERANCE = 1e-6
_WAVE_STATIC_RUNS = ('dc', 'drive_a', 'drive_b')


def _waveform_spec(run):
    """'driven': the time_dependent model; 'dc' / 'drive_a' / 'drive_b': stationary electrostatics with only that part
    (the drive at its voltage, the rest 0)."""
    driven = run == 'driven'

    def voltage(feature):
        if driven or run == feature:
            return _WAVE_DRIVES[feature][2] if feature in _WAVE_DRIVES else _BIAS_VOLTAGE
        return _BIAS_VOLTAGE if (feature == 'bias' and run == 'dc') else 0.0

    def drive_params(feature):
        kind, frequency, _ = _WAVE_DRIVES[feature]
        params = {'voltage': voltage(feature)}
        if driven:
            params.update(waveform=kind, frequency=frequency)
        return params

    features = [FeatureSpec('drive_a', 'terminal', ['electrode_a'], drive_params('drive_a')),
                FeatureSpec('drive_b', 'voltage', ['x_hi'], drive_params('drive_b')),
                FeatureSpec('bias', 'terminal', ['electrode_bias'], {'voltage': voltage('bias')}),
                FeatureSpec('ground', 'voltage', ['y_lo'], {'voltage': 0.0}),
                FeatureSpec('space_charge', 'charge_density', ['block'], {'density': _SPACE_CHARGE},
                            enabled=driven or run == 'dc')]
    features += [FeatureSpec(name, 'potential_probe', [f'{name}_point']) for name in _WAVE_PROBES]
    physics = [PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=features)]
    materials = [MaterialSpec('host', properties={'eps_dc': 4.0}, selection=['all']),
                 MaterialSpec('dielectric', properties={'eps_dc': 30.0}, selection=['block'])]
    if driven:   # the clock: a quiet fluid with rigid faces
        physics.append(PhysicsSpec(_ACOUSTICS, 'elastic_waves',
                                   settings={f'bc_{face}': 'rigid' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}))
        materials[0].properties.update(rho=1000.0, sound_speed=1480.0)
        materials[1].properties.update(rho=1000.0, sound_speed=1480.0)
    return with_exact_cg(ModelSpec(
        title=f'Self-test: electrostatics waveform superposition ({run})',
        domain=DomainSpec(size_x=_WAVE_SIZE[0], size_y=_WAVE_SIZE[1]),
        geometry=[_rectangle('electrode_a', 5e-6, 8e-6, 10e-6, 12e-6),
                  _rectangle('electrode_bias', 20e-6, 2e-6, 25e-6, 6e-6),
                  _rectangle('block', 12e-6, 4e-6, 28e-6, 9e-6),
                  *(_point(f'{name}_point', x, y) for name, (x, y) in _WAVE_PROBES.items())],
        materials=materials, physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_WAVE_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _WAVE_DURATION}) if driven
        else StudySpec(type='stationary')))


def _waveform_specs():
    return {run: _waveform_spec(run) for run in ('driven', *_WAVE_STATIC_RUNS)}


def _waveform_value(kind, frequency, times):
    """s(t) of the ricker (peak 1 at 4 / f) and cw (ramp 1 - exp(-t f / 3)) waveforms (core.sources)."""
    times = np.asarray(times, dtype=np.float64)
    if kind == 'ricker':
        argument = math.pi * frequency * (times - 4.0 / frequency)
        return (1.0 - 2.0 * argument ** 2) * np.exp(-argument ** 2)
    return (1.0 - np.exp(-times * frequency / 3.0)) * np.sin(2.0 * math.pi * frequency * times)


def _probe_trace(results, physics, feature):
    data = results.result(physics, feature)
    return np.asarray(data['t'], dtype=np.float64), np.asarray(data['values'], dtype=np.float64)


def _check_waveform_superposition(results_by_name):
    """Probe traces of the driven run against V_dc + s_a(t) V_a + s_b(t) V_b from the static runs; error = max
    deviation / max |V_dc| + |V_a| + |V_b| over both probes (+ 1 per failed check)."""
    errors, details, failed = {}, {}, []
    for probe in _WAVE_PROBES:
        static = {run: float(_probe_trace(results_by_name[run], _ELECTROSTATICS, probe)[1][-1])
                  for run in _WAVE_STATIC_RUNS}
        times, values = _probe_trace(results_by_name['driven'], _ELECTROSTATICS, probe)
        if times.size < 10:
            failed.append(f'{probe}: only {times.size} samples')
            continue
        expected = static['dc'] + sum(_waveform_value(kind, frequency, times) * static[feature]
                                      for feature, (kind, frequency, _) in _WAVE_DRIVES.items())
        scale = sum(abs(value) for value in static.values())
        errors[probe] = float(np.max(np.abs(values - expected))) / scale
        details[probe] = {'static_V': static, 'n_samples': int(times.size), 'last_time_s': float(times[-1])}
    error = max(errors.values(), default=math.inf) + len(failed)
    return {'measured': max(errors.values(), default=math.nan), 'expected': 0.0,
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _WAVE_TOLERANCE,
            'details': {'errors': errors, 'failed_checks': failed, **details}}


# ---- 3. piezo_thickness_slab_launch ----
# x strip (periodic y, 4 cells of 2.5 um), absorbing x ends; one solid everywhere, piezo_e21 only on the slab x0..x1;
# electrostatics plates: y_hi at V0 s(t) (gaussian), y_lo 0, so E_y = -V0 s(t) / d exactly in every cell (d = 10 um).
# The slab stress -e21 E_y is a body force +-e21 V0 s / d at its faces: right of the slab v_x = e21 V0 / (2 rho c d)
# [s(t - (x - x0) / c) - s(t - (x - x1) / c)].
_SLAB_CELL = 2.5e-6
_SLAB_STRIP_CELLS = 4
# The duration covers the far-face wave at the probe: (probe - x0) / c ~ 184 ns + the 2 t0 = 286 ns pulse + ~33 ns.
_SLAB_LENGTH = 2.3e-3
_SLAB_RANGE = (0.5e-3, 1.5e-3)
_SLAB_PROBE_X = 1.7e-3 + 0.5 * _SLAB_CELL
_SLAB_PROBE_FACE = _SLAB_PROBE_X + 0.5 * _SLAB_CELL   # vx of the probe cell lives on its hi x face (~2.4 % if missed)
_SLAB_SOLID = {'rho': 4700.0, 'c11': 2e11, 'c12': 0.7e11, 'c22': 2e11, 'c66': 0.65e11}
_SLAB_E21 = 2.5                             # C/m^2
_SLAB_VOLTAGE = 10.0                        # V
_SLAB_FREQUENCY, _SLAB_BANDWIDTH = 2e7, 0.5
_SLAB_DURATION = 5.05e-7
_SLAB_TOLERANCE = 2e-2


def _slab_launch_specs():
    width = _SLAB_STRIP_CELLS * _SLAB_CELL
    slab = _rectangle('slab', _SLAB_RANGE[0], 0.0, _SLAB_RANGE[1], width)
    return {'run': with_exact_cg(ModelSpec(
        title='Self-test: piezoelectric thickness-slab launch (one-way)',
        domain=DomainSpec(size_x=_SLAB_LENGTH, size_y=width),
        geometry=[slab, _point('probe_point', _SLAB_PROBE_X, 1.5 * _SLAB_CELL)],
        materials=[MaterialSpec('host', properties=dict(_SLAB_SOLID), selection=['all']),
                   MaterialSpec('piezo', properties={**_SLAB_SOLID, 'piezo_e21': _SLAB_E21}, selection=['slab'])],
        physics=[PhysicsSpec(_ACOUSTICS, 'elastic_waves',
                             settings={'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic', 'pml_cells': 20},
                             features=[FeatureSpec('probe', 'field_probe', ['probe_point'], {'component': 'vx'})]),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('top', 'voltage', ['y_hi'],
                                 {'voltage': _SLAB_VOLTAGE, 'waveform': 'gaussian', 'frequency': _SLAB_FREQUENCY,
                                  'bandwidth': _SLAB_BANDWIDTH}),
                     FeatureSpec('bottom', 'voltage', ['y_lo'], {'voltage': 0.0})])],
        couplings=[CouplingSpec('piezo', PIEZO_TYPE_NAME, _ELECTROSTATICS, _ACOUSTICS,
                                params={'back_action': 'none'})],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _SLAB_DURATION})))}


def _gaussian(times):
    """s(t) of the gaussian waveform (core.sources.GaussianPulse: sin carrier, t0 = 4.5 tau)."""
    tau = 1.0 / (_SLAB_BANDWIDTH * _SLAB_FREQUENCY * math.pi)
    delay = np.asarray(times, dtype=np.float64) - 4.5 * tau
    return np.exp(-(delay / tau) ** 2) * np.sin(2.0 * math.pi * _SLAB_FREQUENCY * delay)


def _check_slab_launch(results_by_name):
    """max |v_x - v_exact| over the probe trace / the single-face amplitude e21 V0 / (2 rho c d)."""
    times, velocity = _probe_trace(results_by_name['run'], _ACOUSTICS, 'probe')
    speed = math.sqrt(_SLAB_SOLID['c11'] / _SLAB_SOLID['rho'])
    thickness = _SLAB_STRIP_CELLS * _SLAB_CELL
    amplitude = _SLAB_E21 * _SLAB_VOLTAGE / (2.0 * _SLAB_SOLID['rho'] * speed * thickness)
    if velocity.size < 10:
        return {'measured': math.nan, 'expected': amplitude, 'error': math.inf, 'tolerance': _SLAB_TOLERANCE,
                'details': {'n_samples': int(velocity.size)}}
    exact = amplitude * (_gaussian(times - (_SLAB_PROBE_FACE - _SLAB_RANGE[0]) / speed)
                         - _gaussian(times - (_SLAB_PROBE_FACE - _SLAB_RANGE[1]) / speed))
    error = float(np.max(np.abs(velocity - exact))) / amplitude
    return {'measured': float(np.max(np.abs(velocity))), 'expected': float(np.max(np.abs(exact))),
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _SLAB_TOLERANCE,
            'details': {'single_face_amplitude_m_per_s': amplitude, 'speed_m_per_s': speed,
                        'n_samples': int(times.size)}}


# ---- 4. piezo_bound_charge_strip, 5. piezo_stiffened_bar_resonance (two-way) ----
# x strips (periodic y, 4 cells of 5 um) of one solid with eps_dc 10 everywhere and piezo_e11 only (P = e11 S_xx, P_y =
# 0: a 1D electrostatics problem), electrostatics grounded on both x faces (Dirichlet: the edge faces take the cell P).
_TWO_WAY_SOLID = {'rho': 4700.0, 'c11': 2e11, 'c12': 0.7e11, 'c22': 2e11, 'c66': 0.65e11, 'eps_dc': 10.0}
_TWO_WAY_E11 = 2.1                          # C/m^2: K_t^2 = (e^2 / (eps0 eps)) / c^D = 0.2
_TWO_WAY_CELL = 5e-6
_TWO_WAY_STRIP_CELLS = 4
_TWO_WAY_COURANT = 0.9
_DIRECT_NAME = 'direct'                     # sparse_direct node of the two-way strips


def _with_direct_electrostatics(spec):
    """spec with a host SuperLU sparse_direct node as the electrostatics 'linear' solver (in place): one factor, one
    exact back substitution per acoustic step (gpu CG on a few hundred cells is launch-bound, ~100 us per iteration)."""
    spec.solvers.append(SolverSpec(_DIRECT_NAME, 'sparse_direct', {'back_end': 'scipy_splu'}))
    spec.physics_by_name(_ELECTROSTATICS).solvers['linear'] = _DIRECT_NAME
    return spec


def _two_way_strip_spec(title, cells, acoustic_faces, piezo_range, features, geometry, duration,
                        back_action='bound_charge'):
    """x strip of _TWO_WAY_SOLID with piezo_e11 on the x range piezo_range (None: everywhere), acoustic x faces
    acoustic_faces, electrostatics (host SuperLU) grounded on both x faces, the piezo coupling with back_action."""
    length, width = cells * _TWO_WAY_CELL, _TWO_WAY_STRIP_CELLS * _TWO_WAY_CELL
    piezo = {**_TWO_WAY_SOLID, 'piezo_e11': _TWO_WAY_E11}
    if piezo_range is None:
        materials = [MaterialSpec('piezo', properties=piezo, selection=['all'])]
    else:
        geometry = [*geometry, _rectangle('slab', piezo_range[0], 0.0, piezo_range[1], width)]
        materials = [MaterialSpec('host', properties=dict(_TWO_WAY_SOLID), selection=['all']),
                     MaterialSpec('piezo', properties=piezo, selection=['slab'])]
    settings = {'bc_x_lo': acoustic_faces, 'bc_x_hi': acoustic_faces, 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic',
                'pml_cells': 20, 'courant': _TWO_WAY_COURANT}
    return _with_direct_electrostatics(ModelSpec(
        title=title,
        domain=DomainSpec(size_x=length, size_y=width),
        geometry=list(geometry),
        materials=materials,
        physics=[PhysicsSpec(_ACOUSTICS, 'elastic_waves', settings=settings, features=list(features)),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('ground_lo', 'voltage', ['x_lo'], {'voltage': 0.0}),
                     FeatureSpec('ground_hi', 'voltage', ['x_hi'], {'voltage': 0.0})])],
        couplings=[CouplingSpec('piezo', PIEZO_TYPE_NAME, _ELECTROSTATICS, _ACOUSTICS,
                                params={'back_action': back_action})],
        mesh=MeshSpec(mode='uniform', dx=_TWO_WAY_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': duration})))


def _sheet_line(name, x):
    return vertical_line(name, x, 0.0, _TWO_WAY_STRIP_CELLS * _TWO_WAY_CELL)


# 4: 200 cells (1 mm), absorbing x ends, piezo slab 0.4..0.6 mm; a Ricker traction sheet at 0.2 mm sends a P pulse
# that straddles the slab's left face at the end. Exact in the FV scheme from the recorded strain S (the accumulator,
# elastic_strain_xx): P_f = mean P of the two cells (edge faces the cell P), rho_b = -(P_f,hi - P_f,lo) / h, and with
# both faces grounded D = sum(l_f P_f) / L is uniform, the face field flux eps0 eps E_f = D - P_f, V by its sum. The
# identity holds at any instant and needs no dispersion accuracy: the Ricker peaks at 4 / f = 100 ns, reaches the face
# ~0.2 mm away ~30 ns later (main lobe half-width ~6 ns, side lobes ~20 ns), so at 130 ns it straddles the face.
_CHARGE_CELLS = 200
_CHARGE_SLAB = (0.4e-3, 0.6e-3)
_CHARGE_SHEET_X = cell_centre(40, _TWO_WAY_CELL)
_CHARGE_RICKER = 4e7                        # Hz: ~33 cells per wavelength
_CHARGE_TRACTION = 1e6                      # Pa
_CHARGE_DURATION = 1.3e-7
_CHARGE_TOLERANCE = 1e-5                    # float32 strain and charge


def _bound_charge_specs():
    features = [FeatureSpec('sheet', 'traction_line_source', ['sheet'],
                            {'waveform': 'ricker', 'frequency': _CHARGE_RICKER, 'traction': _CHARGE_TRACTION,
                             'direction': 'x'})]
    return {'run': _two_way_strip_spec('Self-test: piezoelectric bound charge of a strained slab (two-way)',
                                       _CHARGE_CELLS, 'absorbing', _CHARGE_SLAB, features,
                                       [_sheet_line('sheet', _CHARGE_SHEET_X)], _CHARGE_DURATION)}


def _strip_bound_charge(strain_xx):
    """(rho_b [C/m^3], V [V]) per cell of the grounded strip from the cell strains S_xx (float64, along x)."""
    centres = (np.arange(strain_xx.size) + 0.5) * _TWO_WAY_CELL
    piezo = np.where((centres >= _CHARGE_SLAB[0]) & (centres < _CHARGE_SLAB[1]), _TWO_WAY_E11, 0.0)
    polarization = piezo * strain_xx
    faces = np.concatenate(([polarization[0]], 0.5 * (polarization[:-1] + polarization[1:]), [polarization[-1]]))
    charge = -(faces[1:] - faces[:-1]) / _TWO_WAY_CELL
    lengths = np.full(faces.size, _TWO_WAY_CELL)
    lengths[0] = lengths[-1] = 0.5 * _TWO_WAY_CELL
    displacement = float(np.sum(lengths * faces) / np.sum(lengths))
    potential = -np.cumsum(lengths[:-1] * (displacement - faces[:-1])) / (EPS0 * _TWO_WAY_SOLID['eps_dc'])
    return charge, potential


def _check_bound_charge(results_by_name):
    """Coupled charge rho_free and V of the final state against the FV closed form from the final strain; error =
    the worse of max |rho - rho_b| / max |rho_b| and max |V - V_exact| / max |V_exact|."""
    results = results_by_name['run']
    fields = {name: np.asarray(results.field(name).values, dtype=np.float64)
              for name in (f'{_ACOUSTICS}.elastic_strain_xx', f'{_ELECTROSTATICS}.rho_free', f'{_ELECTROSTATICS}.V')}
    strain, charge, potential = fields.values()
    expected_charge, expected_potential = _strip_bound_charge(strain.mean(axis=1))
    scales = {'charge': float(np.max(np.abs(expected_charge))), 'potential': float(np.max(np.abs(expected_potential)))}
    if not all(scale > 0.0 for scale in scales.values()):
        return {'measured': math.nan, 'expected': math.nan, 'error': math.inf, 'tolerance': _CHARGE_TOLERANCE,
                'details': {'scales': scales, 'note': 'no strain in the slab at the end'}}
    errors = {'charge': float(np.max(np.abs(charge - expected_charge[:, None]))) / scales['charge'],
              'potential': float(np.max(np.abs(potential - expected_potential[:, None]))) / scales['potential']}
    error = max(errors.values())
    return {'measured': float(potential.mean(axis=1)[np.argmax(np.abs(expected_potential))]),
            'expected': float(expected_potential[np.argmax(np.abs(expected_potential))]),
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _CHARGE_TOLERANCE,
            'details': {'errors': errors, 'max_charge_C_per_m3': scales['charge'],
                        'max_potential_V': scales['potential'],
                        'max_strain': float(np.max(np.abs(strain)))}}


# 5: 60 cells (L = 300 um) between rigid (slip) x walls, piezo everywhere, shorted electrodes on both ends (D = e <S>
# = 0 for every mode): the stiffening is exact, T = (c11 + e^2 / (eps0 eps)) S. A Ricker sheet at cell 14 excites
# modes n = 1..3 (S ~ cos(n pi x / L)); a resonance_monitor on vx at cell 37 (two_way: c^D, one_way back_action none:
# c^E, ~12 % lower). Discrete modes: the coupled scheme turns the stiffening into e^2 / (eps0 eps) cos^2(k h / 2)
# (the centre-averaged E of the face-mean charge), so omega_c^2 = (2 / h)^2 sin^2(k h / 2) (c11 + that) / rho and
# sin(omega dt / 2) = dt omega_c / 2; dt = courant / (sqrt(S / rho) sqrt(2) / h), S the stability bound of c^D.
_BAR_CELLS = 60
_BAR_SHEET_CELL, _BAR_PROBE_CELL = 14, 37
_BAR_RICKER = 2.5e7                         # Hz
_BAR_BAND = (8e6, 4e7)                      # Hz: modes 1..3 of both runs (n = 4 above)
_BAR_MODES = (1, 2, 3)
_BAR_DURATION = 1.6e-6                      # s: ~19 periods of mode 1 (12 MHz); (duration - 8 / f_ricker) x 8 MHz =
                                            # 10.2 band-floor periods after the pulse (validation warns below 10).
                                            # The 3 us run had 1/Q <= 3e-9 and discrete errors <= 6e-8 (limit 1e-4)
_BAR_DISCRETE_TOLERANCE = 1e-4
_BAR_CONTINUUM_TOLERANCE = 3e-3             # O((k h)^2) at >= 40 cells per wavelength: ~1.2e-3 for n = 3
_BAR_DT_TOLERANCE = 1e-5                    # float32 coefficients


def _bar_specs():
    features = [FeatureSpec('sheet', 'traction_line_source', ['sheet'],
                            {'waveform': 'ricker', 'frequency': _BAR_RICKER, 'traction': 1e6, 'direction': 'x'}),
                FeatureSpec('resonances', 'resonance_monitor', ['probe_point'],
                            {'component': 'vx', 'frequency_min': _BAR_BAND[0], 'frequency_max': _BAR_BAND[1]})]
    geometry = [_sheet_line('sheet', cell_centre(_BAR_SHEET_CELL, _TWO_WAY_CELL)),
                _point('probe_point', cell_centre(_BAR_PROBE_CELL, _TWO_WAY_CELL), cell_centre(1, _TWO_WAY_CELL))]
    return {run: _two_way_strip_spec(f'Self-test: piezoelectric bar resonances ({run})', _BAR_CELLS, 'rigid', None,
                                     features, geometry, _BAR_DURATION, back_action)
            for run, back_action in (('two_way', 'bound_charge'), ('one_way', 'none'))}


def _bar_stiffening():
    """e^2 / (eps0 eps) [Pa]: c^D - c^E of the bar material."""
    return _TWO_WAY_E11 ** 2 / (EPS0 * _TWO_WAY_SOLID['eps_dc'])


def _bar_time_step(stiffening):
    """The build's dt for c11 + stiffening: courant h / (sqrt(S / rho) sqrt(2)), S the square-cell bound."""
    solid = _TWO_WAY_SOLID
    s11, s12, s22, s66 = solid['c11'] + stiffening, solid['c12'], solid['c22'], solid['c66']
    p, q, r = 0.5 * (s11 + s66), 0.5 * (s66 + s22), 0.5 * abs(s12 + s66)
    bound = max(0.5 * (p + q) + math.hypot(0.5 * (p - q), r), s11, s22)
    return _TWO_WAY_COURANT * _TWO_WAY_CELL / (math.sqrt(bound / solid['rho']) * math.sqrt(2.0))


def _bar_frequencies(mode, dt, stiffening):
    """(discrete leap-frog, continuum) frequency [Hz] of bar mode `mode`."""
    solid = _TWO_WAY_SOLID
    half_phase = mode * math.pi / (2 * _BAR_CELLS)   # k h / 2
    omega = (2.0 / _TWO_WAY_CELL) * math.sin(half_phase) * math.sqrt(
        (solid['c11'] + stiffening * math.cos(half_phase) ** 2) / solid['rho'])
    discrete = 2.0 * math.asin(0.5 * dt * omega) / (2.0 * math.pi * dt)
    continuum = mode * math.sqrt((solid['c11'] + stiffening) / solid['rho']) / (2.0 * _BAR_CELLS * _TWO_WAY_CELL)
    return discrete, continuum


def _check_bar_resonances(results_by_name):
    """Per run the dt error / _BAR_DT_TOLERANCE and per mode max(|f / f_discrete - 1|, 1/Q) / discrete tolerance and
    |f / f_continuum - 1| / continuum tolerance; error = the worst (tolerance 1)."""
    errors, details = {}, {}
    for run, stiffening in (('two_way', _bar_stiffening()), ('one_way', 0.0)):
        table = results_by_name[run].result(_ACOUSTICS, 'resonances')
        columns = list(table['columns'])
        rows = [list(row) for row in table['rows']]
        frequencies = np.asarray([row[columns.index('frequency')] for row in rows], dtype=np.float64)
        dt = float(table['dt'])
        expected_dt = _bar_time_step(stiffening)
        errors[f'{run}_dt'] = abs(dt / expected_dt - 1.0) / _BAR_DT_TOLERANCE
        details[f'{run}_dt'] = {'dt': dt, 'expected_dt': expected_dt}
        for mode in _BAR_MODES:
            label = f'{run}_{mode}'
            discrete, continuum = _bar_frequencies(mode, dt, stiffening)
            if not frequencies.size:
                errors[label] = math.inf
                continue
            row = rows[int(np.argmin(np.abs(frequencies - discrete)))]
            measured = float(row[columns.index('frequency')])
            inverse_q = 2.0 * abs(float(row[columns.index('frequency_imag')])) / measured
            discrete_error = max(abs(measured / discrete - 1.0), inverse_q)
            continuum_error = abs(measured / continuum - 1.0)
            errors[label] = max(discrete_error / _BAR_DISCRETE_TOLERANCE, continuum_error / _BAR_CONTINUUM_TOLERANCE)
            details[label] = {'frequency': measured, 'discrete_frequency': discrete, 'continuum_frequency': continuum,
                              'inverse_Q': inverse_q, 'discrete_error': discrete_error,
                              'continuum_error': continuum_error}
    worst = max(errors, key=errors.get)
    first = details.get('two_way_1') or {}
    return {'measured': first.get('frequency', math.nan), 'expected': first.get('continuum_frequency', math.nan),
            'error': errors[worst] if math.isfinite(errors[worst]) else math.inf, 'tolerance': 1.0,
            'details': {'errors': errors, 'worst': worst, 'modes': details,
                        'discrete_tolerance': _BAR_DISCRETE_TOLERANCE,
                        'continuum_tolerance': _BAR_CONTINUUM_TOLERANCE, 'dt_tolerance': _BAR_DT_TOLERANCE}}


PIEZO_CASES = (
    SelfTestCase(
        name='piezo_crystal_tensors', label='Piezoelectric crystal tensors: symmetry, rotations, YZ-LN block',
        description=('CPU check of materials/crystal_tensors (no model run). (1) Every crystal entry (lithium niobate '
                     '3m, quartz 32, ZnO / AlN 6mm, GaAs -43m) is invariant under the generators of its point group '
                     '(3m: 3-fold z, mirror normal to x; 32: 3-fold z, 2-fold x; 6mm: 6-fold z, mirror; -43m: -4 z, '
                     '3-fold [111], mirror (1-10)) for c^E, e and eps^S: an exact check of the data patterns. (2) The '
                     'Bond-matrix rotation (c\' = M c M^T, e\' = R e M^T, eps\' = R eps R^T) equals the explicit '
                     f'rank-4 / rank-3 / rank-2 index rotation of physics/nlfd/tensors at Euler {list(_GENERIC_EULER)} '
                     'deg, and the Voigt / full-tensor conversions round-trip. (3) The sagittal YZ-LN entry '
                     f'(euler_deg {list(_YZ_EULER)}: model x = Z, y = -Y) equals the hand-written block c11 = c33, c12 '
                     '= c13, c22 = c11, c66 = c44, piezo_e11 = e33, e12 = e31, e22 = -e22, e26 = e15, e16 = e21 = 0, '
                     'eps_dc_xx = eps33, eps_dc_yy = eps11, c16 = 0 and c26 = +c14 (the monoclinic term, expanded '
                     'into the properties); its anti-plane coupling ratios are 0 (the YZ plane is the mirror plane). '
                     '(4) X-cut / '
                     f'Y-propagating (euler_deg {list(_X_CUT_Y_EULER)}) has an anti-plane ratio > '
                     f'{_MIN_ANTIPLANE_RATIO:g}. (5) Optics: the photoelastic p and Pockels r data of every crystal '
                     'that has them pass the same point-group check, their Bond rotations (p\' = M p M^T, r\' = M r '
                     'R^T) equal the index rotations, the YZ-LN entries equal photoelastic_p11 = p33, p12 = p31, p16 '
                     '= 0, p21 = p13, p22 = p11, p26 = p14, p31 = p13, p32 = p12, p36 = -p14, p61 = 0, p62 = p41, p66 '
                     '= p44, pockels_r11 = r33, r12 = 0, r21 = r13, r22 = -r22, r31 = r13, r32 = r22, r61 = 0, r62 = '
                     'r51 with the xy-row ratios max(|p41|, |p44|) / max and r51 / max and anti-plane (yz, xz) ratios '
                     '0, and GaAs at euler_deg 0, 0, 0 writes no Pockels entries (r41 lies only in the anti-plane rows: '
                     'ratio 1) but the photoelastic tensor with p66 = p44 (cubic, not the isotropic (p11 - p12) / 2). '
                     'error = the worst relative deviation + 1 per failed '
                     f'check; tolerance {_TENSOR_TOLERANCE:g}. It does not check the data values themselves.'),
        specs=dict, check=_check_crystal_tensors),
    SelfTestCase(
        name='piezo_crystal_full_expansion', label="Piezoelectric crystal tensors: the 'full' (3D) expansion",
        description=("CPU check of the 'plane': 'full' crystal entry of materials/crystal_tensors (no model run; 3D "
                     "model axes x, y, z = X1', X2', X3'). For every crystal at euler_deg "
                     f"{', '.join(str(list(euler)) for euler in _FULL_EULERS)}: (1) the expanded c, e, eps equal the "
                     'explicit rank-4 / rank-3 / rank-2 index rotation (physics/nlfd/tensors) by the substrate rotation; '
                     "(2) rotated on by the sagittal axes (x, y, z) -> (X1', X3', -X2') they equal the 'sagittal' "
                     'tensors, and their c11 / c12 / c22 / c66 / eps_dc_xx / _yy / _xy the sagittal entry; (3) the '
                     'properties are exactly rho, the 21 c_IJ, the 18 piezo_e<i><J> and the six eps_dc_* of the '
                     "tensors, c symmetric. Also: full_crystal refuses a 'sagittal' entry and sagittal_crystal a 'full' "
                     'one; effective_material_data with lithium_niobate_x_cut + a full entry + user c44 (c11 None) '
                     'gives the user c44, the crystal c11 / c14 (library < crystal < user) and no leftover pole. error = '
                     f'the worst relative deviation + 1 per failed check; tolerance {_FULL_TOLERANCE:g} (the 1e-12 '
                     'clean-up of rotated entries).'),
        specs=dict, check=_check_crystal_full_expansion),
    SelfTestCase(
        name='electrostatics_waveform_superposition',
        label='Electrostatics: driven terminals by superposition (time-dependent)',
        description=('Electrostatics (float64 CG to 1e-12) on 40 x 20 cells of 1 um: terminal drive_a (2 V, ricker '
                     '200 MHz), voltage drive_b on x_hi (-1 V, cw 60 MHz), terminal bias (0.5 V constant), 0 V on '
                     f'y_lo and a space charge of {_SPACE_CHARGE:g} C/m^3 in an eps_dc 30 block (host 4); an '
                     'elastic_waves fluid with rigid faces and no source only sets the time step of the '
                     'time_dependent run. Three stationary runs give V_dc (bias, ground, charge; drives at 0 V) and '
                     'the responses V_a, V_b of each drive at its voltage alone. Exact (linearity): V(t) = V_dc + '
                     's_a(t) V_a + s_b(t) V_b with s the waveform at the probe sample times (the end time of each '
                     'step). error = max over two potential probes and all samples of |V - V(t)| / (|V_dc| + |V_a| + '
                     f'|V_b|) + 1 per failed check; tolerance {_WAVE_TOLERANCE:g}. Pins the per-basis held values and '
                     'outer-face BCs, sources only in the DC part, the step end time and the restored assembly.'),
        specs=_waveform_specs, check=_check_waveform_superposition),
    SelfTestCase(
        name='piezo_thickness_slab_launch',
        label='Piezoelectric (one-way): plane waves launched by a driven slab',
        description=(f'A strip along x ({_SLAB_LENGTH * 1e3:g} mm, periodic y of {_SLAB_STRIP_CELLS} cells, '
                     f'{_SLAB_CELL * 1e6:g} um cells, absorbing x ends) of one solid (rho {_SLAB_SOLID["rho"]:g}, '
                     f'c11 {_SLAB_SOLID["c11"]:g} Pa) with piezo_e21 = {_SLAB_E21:g} C/m^2 on the slab '
                     f'{_SLAB_RANGE[0] * 1e3:g}..{_SLAB_RANGE[1] * 1e3:g} mm; electrostatics plates on the y faces '
                     f'(y_hi at {_SLAB_VOLTAGE:g} V x gaussian {_SLAB_FREQUENCY:g} Hz, bandwidth {_SLAB_BANDWIDTH:g}; '
                     'y_lo 0 V) give E_y = -V0 s(t) / d exactly in every cell; electrostatics_acoustics_piezo '
                     '(back_action none) hands E to the acoustics. Exact: the slab stress -e21 E_y acts as body '
                     'forces +-e21 V0 s / d on its '
                     f'faces, so a vx probe at {_SLAB_PROBE_X * 1e3:g} mm sees e21 V0 / (2 rho c d) [s(t - (x - x0) / '
                     'c) - s(t - (x - x1) / c)], c = sqrt(c11 / rho). error = max |vx - exact| over the trace / e21 V0 '
                     f'/ (2 rho c d); tolerance {_SLAB_TOLERANCE:g} (dispersion at ~85 cells per wavelength; 2.3e-3 '
                     'with the probe at 2 mm in a 3 mm strip). Pins the sign and size of the -e^T dE stress, the '
                     'field time (a one-step lag is ~9 % off) and the coupling cadence. Not yet run with the probe at '
                     f'{_SLAB_PROBE_X * 1e3:g} mm, {_SLAB_LENGTH * 1e3:g} mm strip, {_SLAB_DURATION * 1e9:g} ns (the '
                     'x_hi PML is now 0.6 mm from the probe).'),
        specs=_slab_launch_specs, check=_check_slab_launch),
    SelfTestCase(
        name='piezo_bound_charge_strip',
        label='Piezoelectric (two-way): bound charge and potential of a strained slab',
        description=(f'A strip along x ({_CHARGE_CELLS} cells of {_TWO_WAY_CELL * 1e6:g} um, periodic y of '
                     f'{_TWO_WAY_STRIP_CELLS} cells, absorbing x ends) of one solid (rho {_TWO_WAY_SOLID["rho"]:g}, c11 '
                     f'{_TWO_WAY_SOLID["c11"]:g} Pa, eps_dc {_TWO_WAY_SOLID["eps_dc"]:g}) with piezo_e11 = '
                     f'{_TWO_WAY_E11:g} C/m^2 only on the slab {_CHARGE_SLAB[0] * 1e3:g}..{_CHARGE_SLAB[1] * 1e3:g} mm '
                     '(P = e11 S_xx, P_y = 0); electrostatics (host SuperLU, sparse_direct node) grounded on both x '
                     'faces; '
                     'electrostatics_acoustics_piezo two-way (back_action bound_charge). A Ricker traction sheet '
                     f'({_CHARGE_RICKER:g} Hz) at {_CHARGE_SHEET_X * 1e3:g} mm sends a P pulse that straddles the slab '
                     f'face at the end ({_CHARGE_DURATION * 1e9:g} ns). Exact in the FV scheme from the final '
                     'elastic_strain_xx (the strain accumulator): face P = the mean of the two cells (grounded edge '
                     'faces: the cell P), rho_b = -(P_hi - P_lo) / h, and with both faces grounded D = sum(l_f P_f) / L '
                     'uniform, eps0 eps E_f = D - P_f, V its running sum from 0. error = the worse of max |rho_free - '
                     'rho_b| / max |rho_b| and max |V - V_exact| / max |V_exact| over the window; tolerance '
                     f'{_CHARGE_TOLERANCE:g} (float32 strain and charge). Pins the strain accumulator, the face form '
                     'and sign of the bound charge, the electrode-face rule and the re-solve after the last step. Not '
                     f'yet run at {_CHARGE_RICKER:g} Hz / {_CHARGE_DURATION * 1e9:g} ns (heavy flag to re-time: '
                     '42.8 s at 1e7 Hz / 450 ns, ~3.5x fewer steps now).'),
        specs=_bound_charge_specs, check=_check_bound_charge),
    SelfTestCase(
        name='piezo_stiffened_bar_resonance',
        label='Piezoelectric (two-way): stiffened resonances of a shorted bar',
        description=(f'A bar along x ({_BAR_CELLS} cells of {_TWO_WAY_CELL * 1e6:g} um, L = '
                     f'{_BAR_CELLS * _TWO_WAY_CELL * 1e6:g} um, periodic y of {_TWO_WAY_STRIP_CELLS} cells) between '
                     'rigid (slip) x walls, all of the piezo solid of piezo_bound_charge_strip (K_t^2 = (e^2 / (eps0 '
                     'eps)) / c^D = 0.2), electrostatics (host SuperLU; was float64 CG: 35 s of launch-bound gpu '
                     'iterations) grounded on both ends (shorted electrodes: D = '
                     f'e <S> = 0 for every mode, so T = (c11 + e^2 / (eps0 eps)) S exactly); a Ricker sheet '
                     f'({_BAR_RICKER:g} Hz) at cell {_BAR_SHEET_CELL}, a resonance_monitor on vx at cell '
                     f'{_BAR_PROBE_CELL} (band {_BAR_BAND[0]:g}-{_BAR_BAND[1]:g} Hz: modes n = 1..3), '
                     f'{_BAR_DURATION * 1e6:g} us. Runs two_way (back_action bound_charge) and one_way (none: c^E, '
                     '~12 % lower). Exact: the coupled scheme stiffens mode k by e^2 / (eps0 eps) cos^2(k h / 2) '
                     '(the centre-averaged field of the face-mean charge), so omega_c^2 = (2 / h)^2 sin^2(k h / 2) (c11 '
                     '+ that) / rho, sin(omega dt / 2) = dt omega_c / 2, Q = inf (a symmetric coupling conserves '
                     'energy); continuum f_n = n c / (2 L), c = sqrt(c^D / rho) or sqrt(c11 / rho); dt = courant / '
                     '(sqrt(S / rho) sqrt(2) / h), S the stability bound of the stiffened c^D (two_way) or of c^E. '
                     'error = the worst of max(|f / f_discrete - 1|, 1/Q) / '
                     f'{_BAR_DISCRETE_TOLERANCE:g}, |f / f_continuum - 1| / {_BAR_CONTINUUM_TOLERANCE:g} and |dt / '
                     f'dt_expected - 1| / {_BAR_DT_TOLERANCE:g}; tolerance 1. Pins the stiffening without lag (a '
                     'one-step lag damps or grows the modes), the adjoint stress / charge pair and the stiffened dt. '
                     f'Not yet run at {_BAR_DURATION * 1e6:g} us (at 3 us: 1/Q <= 3e-9, discrete errors <= 6e-8).'),
        specs=_bar_specs, check=_check_bar_resonances),
)


# ---- template ----
# yz_ln_saw_idt: YZ lithium niobate substrate (0 < y < 60 um) under vacuum, 1 um cells, a 5-pair IDT (period 20 um,
# fingers 5 um wide, one cell thick, in the vacuum on the surface) as two driven terminals at +-1 V (gaussian at v_R /
# period, bandwidth 0.2), vy probes on the surface 100 and 200 um from the IDT centre.
YZ_LN_SAW_IDT_LABEL = 'SAW from an IDT on YZ lithium niobate'
_IDT_PERIOD = 20e-6
_IDT_PAIRS = 5
_IDT_CENTRE_X = 100e-6
_SAW_VELOCITY = 3488.0                     # m/s: free-surface YZ-LN Rayleigh velocity (literature)
_SAW_DOMAIN = (400e-6, 80e-6)
_SUBSTRATE_TOP = 60e-6
_SAW_CELL = 1e-6
YZ_LN_SAW_IDT_DESCRIPTION = (
    'Electrostatics + elastic_waves with electrostatics_acoustics_piezo (time_dependent): a 5-pair interdigital '
    'transducer (period 20 um, two driven terminals at +1 / -1 V, gaussian at f = v_R / period = 174 MHz, bandwidth '
    '0.2) on a YZ lithium niobate substrate (a crystal entry, euler_deg 0, 90, 90: propagation along crystal Z, '
    'surface normal Y) launches surface acoustic waves; vy probes on the surface 100 and 200 um from the IDT. The '
    'literature Rayleigh velocity is ~3488 m/s (free surface); the coupling is two-way (bound charge: '
    'piezoelectric stiffening included, one electrostatics solve per acoustic step) and the in-plane c26 = c14 is '
    'stepped, so the remaining deviation is mesh dispersion (~20 cells per SAW wavelength) and the crystal data.')
YZ_LN_SAW_IDT_SUMMARY = ('Surface acoustic waves: an interdigital transducer on lithium niobate turns a voltage pulse '
                         'into a surface wave, recorded by two probes on the surface.')


def yz_ln_saw_idt_spec():
    """Template 'yz_ln_saw_idt': YZ-LN substrate, a driven IDT (two terminals), surface vy probes."""
    size_x, size_y = _SAW_DOMAIN
    fingers = {'plus': [], 'minus': []}
    geometry = [_rectangle('substrate', 0.0, 0.0, size_x, _SUBSTRATE_TOP)]
    for index in range(2 * _IDT_PAIRS):
        centre = _IDT_CENTRE_X + (index - _IDT_PAIRS + 0.5) * 0.5 * _IDT_PERIOD
        name = f'finger_{index + 1}'
        geometry.append(_rectangle(name, centre - 0.125 * _IDT_PERIOD, _SUBSTRATE_TOP, centre + 0.125 * _IDT_PERIOD,
                                   _SUBSTRATE_TOP + _SAW_CELL))
        fingers['plus' if index % 2 == 0 else 'minus'].append(name)
    probes = {'surface_near': _IDT_CENTRE_X + 100e-6, 'surface_far': _IDT_CENTRE_X + 200e-6}
    geometry += [_point(f'{name}_point', x + 0.5 * _SAW_CELL, _SUBSTRATE_TOP - 0.5 * _SAW_CELL)
                 for name, x in probes.items()]
    frequency = _SAW_VELOCITY / _IDT_PERIOD
    drive = {'waveform': 'gaussian', 'frequency': frequency, 'bandwidth': 0.2}
    return ModelSpec(
        title=YZ_LN_SAW_IDT_LABEL,
        description=YZ_LN_SAW_IDT_DESCRIPTION,
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('yz_lithium_niobate', dispersion=[
            {'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': [0.0, 90.0, 90.0], 'plane': 'sagittal'}],
            selection=['substrate'])],
        physics=[PhysicsSpec(_ACOUSTICS, 'elastic_waves', settings={'bc_y_hi': 'rigid', 'pml_cells': 20}, features=[
                     FeatureSpec(name, 'field_probe', [f'{name}_point'], {'component': 'vy'}) for name in probes]),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('idt_plus', 'terminal', fingers['plus'], {'voltage': 1.0, **drive}),
                     FeatureSpec('idt_minus', 'terminal', fingers['minus'], {'voltage': -1.0, **drive})])],
        couplings=[CouplingSpec('piezo', PIEZO_TYPE_NAME, _ELECTROSTATICS, _ACOUSTICS)],
        mesh=MeshSpec(mode='uniform', dx=_SAW_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': 1.5e-7, 'output_interval': 2.5e-8}))


# tfln_acousto_optic_modulator (sagittal plane, not yet run or tuned): uniform 50 nm mesh, 40 x 9 um window (800 x 180
# cells; 830 x 210 em_fdfd_hz unknowns with the PML), silica substrate y < 7 um, Y-cut lithium niobate film 7..8 um
# (crystal entry YZ: model x = crystal Z, so E_x sees n_e, E_y n_o), vacuum above. A 3-pair IDT (period 4.4 um = 88
# cells, fingers 22 cells wide, one cell thick on the film) at x = 3.55..15.65 um drives a cw SAW (f = 3400 m/s /
# period, an estimate between the silica and YZ-LN Rayleigh speeds) along +-x; the film guides TM light (H_z) from a
# mode source at x = 1.525 um (left of the IDT; the left-going SAW crosses it before the x_lo absorber) to mode monitors
# at x = 38.475 um. Period from TM0 -> TM1 phase matching of the asymmetric slab with an isotropic n = 2.2 (n_eff ~2.08
# / ~1.73: 1.55 / 0.356 = 4.35 um). dt ~3.4e-12 s (the stiffened LN bound at its vacuum faces): ~6000 steps for 20
# ns, each with one electrostatics CG solve (two-way piezo); the optics re-solves every 100 steps (~60 factorisations).
# Acoustic absorbers: pml_cells 30 = the 15 EM padding cells + 15 window cells at x_lo, x_hi and y_lo.
TFLN_AOM_LABEL = 'Thin-film lithium niobate acousto-optic modulator'
_AOM_CELL = 50e-9
_AOM_DOMAIN = (40e-6, 9e-6)
_AOM_FILM_Y = (7e-6, 8e-6)
_AOM_PERIOD = 4.4e-6
_AOM_PAIRS = 3
_AOM_IDT_CENTRE_X = 9.6e-6
_AOM_SAW_VELOCITY = 3400.0                  # m/s, estimate (film on silica)
_AOM_FREQUENCY = _AOM_SAW_VELOCITY / _AOM_PERIOD
_AOM_LINE_X = {'source_line': 1.525e-6, 'monitor_line': 38.475e-6}   # m, cell centres
_AOM_SAW_PROBE = (27.025e-6, 7.975e-6)      # m, the film's top cell row
_AOM_WAVELENGTH = 1.55e-6
_AOM_EM_EVERY = 100                         # acoustic steps per optical re-solve (~0.34 ns, ~4 per SAW period)
_AOM_DURATION = 20e-9
_AOM_VOLTAGE = 5.0                          # V, IDT drive amplitude (parameter V_drive)
_LN_OPTICAL = {'eps_r': 4.889, 'eps_xx': 4.571}   # n_o^2 (E_y, E_z), n_e^2 (E_x: crystal Z along model x)
TFLN_AOM_DESCRIPTION = (
    'Elastic_waves + electrostatics + EM FDFD (H_z), time_dependent, in the sagittal plane: a 3-pair interdigital '
    f'transducer (period {_AOM_PERIOD * 1e6:g} um, terminals at +V_drive / -V_drive, cw at f = '
    f'{_AOM_SAW_VELOCITY:g} m/s / period = {_AOM_FREQUENCY * 1e-6:.0f} MHz) on a 1 um Y-cut lithium niobate film '
    '(crystal entry, euler_deg 0, 90, 90: SAW along crystal Z, surface normal Y; eps_xx = n_e^2, eps_r = n_o^2) on '
    'silica launches a surface acoustic wave (two-way piezoelectric coupling) along the film, which also guides 1.55 '
    'um TM light along x: a mode_source (mode 1) left of the IDT, mode_monitors of modes 1 and 2 near the right end. '
    'acoustics_em_acousto_optic (rotated photoelastic tensor: the diagonal rows and the eps_xy row of the shear and '
    'normal strains) and electrostatics_em_pockels (clamped r of the IDT field and of the SAW piezoelectric field, '
    f'r51 into eps_xy) feed the optics, which re-solves every {_AOM_EM_EVERY} acoustic steps on the frozen state '
    '(stroboscopic snapshots). The period aims at TM0 -> TM1 phase matching, period = lambda / (n_eff,1 - n_eff,2), '
    'from slab estimates (n_eff ~2.08 / ~1.73); the template is not tuned by a run: read n_eff of both monitors '
    'and the SAW speed from the surface vy probe, then adjust the finger geometry and the frequency. Derived values: '
    'through_fraction and converted_fraction (mode 2) of the last solve. Not modelled: metal of the fingers in the '
    'optics (vacuum there), eps_xy in the port modes (they follow the strained diagonal eps of their line), acoustic '
    'and optical loss. Cost: ~6000 acoustic steps, each with an electrostatics solve, and ~60 sparse direct solves.')
TFLN_AOM_SUMMARY = ('Acousto-optic modulator: an interdigital transducer on a thin lithium niobate film launches a '
                    'surface acoustic wave whose strain and electric field form a moving grating for the light guided '
                    'in the film, converting part of it into the next guided mode. The optics is solved again every '
                    'few acoustic steps.')


def tfln_acousto_optic_modulator_spec():
    """Template 'tfln_acousto_optic_modulator': Y-cut LN film on silica, a cw IDT (two-way piezo), em_fdfd_hz with the
    acousto-optic and Pockels couplings, mode-1 / mode-2 monitors."""
    size_x, size_y = _AOM_DOMAIN
    film_bottom, film_top = _AOM_FILM_Y
    fingers = {'plus': [], 'minus': []}
    geometry = [_rectangle('substrate', 0.0, 0.0, size_x, film_bottom),
                _rectangle('film', 0.0, film_bottom, size_x, film_top)]
    for index in range(2 * _AOM_PAIRS):
        centre = _AOM_IDT_CENTRE_X + (index - _AOM_PAIRS + 0.5) * 0.5 * _AOM_PERIOD
        name = f'finger_{index + 1}'
        geometry.append(_rectangle(name, centre - 0.125 * _AOM_PERIOD, film_top, centre + 0.125 * _AOM_PERIOD,
                                   film_top + _AOM_CELL))
        fingers['plus' if index % 2 == 0 else 'minus'].append(name)
    geometry += [*(vertical_line(name, x, 0.0, size_y) for name, x in _AOM_LINE_X.items()), _point('saw_point', *_AOM_SAW_PROBE)]
    drive = {'waveform': 'cw', 'frequency': _AOM_FREQUENCY}
    return ModelSpec(
        title=TFLN_AOM_LABEL,
        description=TFLN_AOM_DESCRIPTION,
        parameters=[ParameterSpec('V_drive', _AOM_VOLTAGE, 'IDT drive amplitude [V]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('silica_substrate', library='silica', selection=['substrate']),
                   MaterialSpec('y_cut_lithium_niobate', properties=dict(_LN_OPTICAL), dispersion=[
                       {'model': 'crystal', 'crystal': 'lithium_niobate', 'euler_deg': [0.0, 90.0, 90.0],
                        'plane': 'sagittal'}], selection=['film'])],
        physics=[PhysicsSpec(_ACOUSTICS, 'elastic_waves', settings={'bc_y_hi': 'rigid', 'pml_cells': 30}, features=[
                     FeatureSpec('saw_probe', 'field_probe', ['saw_point'], {'component': 'vy'})]),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('idt_plus', 'terminal', fingers['plus'], {'voltage': 'V_drive', **drive}),
                     FeatureSpec('idt_minus', 'terminal', fingers['minus'], {'voltage': '-V_drive', **drive})]),
                 PhysicsSpec(_EM, 'em_fdfd_hz', settings={'wavelength': _AOM_WAVELENGTH}, every=_AOM_EM_EVERY,
                             features=[
                     FeatureSpec('source', 'mode_source', ['source_line'],
                                 {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}),
                     FeatureSpec('through', 'mode_monitor', ['monitor_line'], {'mode_number': 1}),
                     FeatureSpec('converted', 'mode_monitor', ['monitor_line'], {'mode_number': 2})])],
        couplings=[CouplingSpec('piezo', PIEZO_TYPE_NAME, _ELECTROSTATICS, _ACOUSTICS),
                   CouplingSpec('acousto_optic', _ACOUSTO_OPTIC_TYPE_NAME, _ACOUSTICS, _EM),
                   CouplingSpec('pockels', _POCKELS_TYPE_NAME, _ELECTROSTATICS, _EM)],
        mesh=MeshSpec(mode='uniform', dx=_AOM_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _AOM_DURATION, 'output_interval': 0.5e-9}),
        derived_values=[DerivedValueSpec('through_fraction', 'fraction_positive', operation='last',
                                         source=f'{_EM}.through'),
                        DerivedValueSpec('converted_fraction', 'fraction_positive', operation='last',
                                         source=f'{_EM}.converted')])

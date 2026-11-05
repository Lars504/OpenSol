"""Self-tests of the elastic_waves acousto-optic coupling against closed forms: acousto-optic index and normal-strain
eps_xy of a uniformly stressed crystal (em_fdfd_ez / em_fdfd_hz re-solving in a time_dependent study) and its shear
column and eps_xy under a plane S wave (the photoacoustic coupling: examples/acoustics/photoacoustic_beer_lambert);
registered by definitions_acoustics (no physics imports, specs built fresh)."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import cell_centre, vertical_line
from ..model.spec import CouplingSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec

ACOUSTO_OPTIC_TYPE_NAME = 'acoustics_em_acousto_optic'
_ACOUSTICS_TYPE = 'elastic_waves'
_EM = 'em'
_ACOUSTICS = 'acoustics'
_WAVELENGTH = 1.55e-6                       # m, vacuum
_CELL = 20e-9                               # m, uniform mesh of both cases

# acousto_optic_uniform_strain: a fully periodic 20 x 20 cell box of one orthotropic crystal (explicit c_ij, so exx !=
# eyy) heated uniformly by a heat_pulse_source on 'all': the thermoelastic stress sxx = syy = -G H stays uniform (no
# gradient, no motion), H = Q sqrt(pi) w the absorbed energy density, so the elastic strain is exact: [exx, eyy] =
# C^-1 [s, s], e_zz = 0 (plane strain). The FDFD (periodic box, no source: only its permittivity is checked) re-solves
# every _OPTIC_EVERY acoustic steps; at the end eps_ii = (n0 + dn_ii)^2, dn_ii = -n0^3 / 2 (p11 e_ii + p12 (e_jj +
# e_kk)). Strains ~-1e-3 / -1.8e-3, dn ~1e-3; dt ~1.5e-12 s, ~80 steps.
_OPTIC_BOX_CELLS = 20
_CRYSTAL = {'rho': 2330.0, 'c11': 165.7e9, 'c12': 63.9e9, 'c22': 120.0e9, 'c66': 79.6e9, 'gruneisen': 1.0,
            'eps_r': 2.085, 'photoelastic_p11': 0.121, 'photoelastic_p12': 0.270}
_OPTIC_POWER = 1e19                         # W/m^3, heat_pulse_source peak
_OPTIC_WIDTH = 1.5e-11                      # s, ~10 acoustic steps
_OPTIC_DELAY = 4.0 * _OPTIC_WIDTH
_OPTIC_DURATION = 8.0 * _OPTIC_WIDTH
_OPTIC_EVERY = 40                           # acoustic steps per FDFD re-solve: 2 mid-run applies + the flush (one-way
                                            # coupling, the check reads only the final strain and eps)
_OPTIC_TOLERANCE = 1e-3                     # relative to the index / strain change (float32 eps: ~1e-4)
# tensor runs: the same box with photoelastic tensor entries (rows yy / zz partly set, p22 / p32 left to the isotropic
# fallback p11 / p12; the shear columns and p66 multiply the exactly zero shear strain of the uniform stress, so the
# hz eps_xy is exactly -n0^4 (p61 e_xx + p62 e_yy): the normal-strain xy row, both terms of similar size)
_TENSOR_ENTRIES = {'photoelastic_p21': 0.05, 'photoelastic_p31': -0.03, 'photoelastic_p16': 0.2,
                   'photoelastic_p26': -0.1, 'photoelastic_p36': 0.15, 'photoelastic_p61': 0.06,
                   'photoelastic_p62': 0.035}
_ALL_PERIODIC = {f'bc_{face}': 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}

# acousto_optic_shear_strain: a fully periodic 40 x 4 cell box of an isotropic glass, a y-traction Ricker sheet on a
# vertical line (cell 10) launching plane S waves along x: v_y(x) and sxy(x) only, so e_xx = e_yy = 0 exactly and the
# centre engineering shear g = 4-corner mean of sxy / c66 is the only strain. At the end (after the flush) every index
# change is the shear column times g: dn_ii = -n0^3 / 2 p_i6 g (tensor runs; 0 isotropic) and Delta eps_xy = -n0^4 p66 g
# (em_fdfd_hz; isotropic p66 = (p11 - p12) / 2). dt ~2.4e-12 s, ~420 steps, an FDFD re-solve every _SHEAR_EVERY: the
# coupling is one-way, so the acoustic fields (and g) do not depend on the EM cadence, and the check grades only the
# flushed eps; ~4 mid-run applies keep the apply / change-check path (each hz re-solve with eps_xy costs ~120 ms).
_SHEAR_CELLS = (40, 4)
_SHEAR_EVERY = 100                          # acoustic steps per FDFD re-solve
_GLASS = {'rho': 2200.0, 'sound_speed': 5960.0, 'shear_speed': 3760.0, 'eps_r': 2.085, 'photoelastic_p11': 0.121,
          'photoelastic_p12': 0.270}
_SHEAR_TENSOR_ENTRIES = {'photoelastic_p16': 0.2, 'photoelastic_p26': -0.1, 'photoelastic_p36': 0.15,
                         'photoelastic_p61': 0.03, 'photoelastic_p62': -0.04, 'photoelastic_p66': 0.11}
_SHEAR_SOURCE_CELL = 10
_SHEAR_TRACTION = 1e8                       # Pa: |g| ~ 1e-3
_SHEAR_FREQUENCY = 4e9                      # Hz, Ricker peak: >= 19 cells per shear wavelength at 2.5 f
_SHEAR_DURATION = 1.0e-9                    # s, the Ricker maximum
_SHEAR_TOLERANCE = 1e-3                     # relative to the largest change of each quantity (float32 kernels)


# ---- specs ----
def _optic_spec(em_type, tensor=False):
    """Uniformly heated periodic crystal box: elastic_waves + `em_type` (every _OPTIC_EVERY) + the acousto-optic
    coupling, time_dependent; tensor: + the photoelastic tensor entries."""
    size = _OPTIC_BOX_CELLS * _CELL
    properties = {**_CRYSTAL, **(_TENSOR_ENTRIES if tensor else {})}
    return ModelSpec(
        title=f"Self-test: acousto-optic index of a uniformly stressed crystal ({em_type}{', tensor' if tensor else ''})",
        domain=DomainSpec(size_x=size, size_y=size),
        materials=[MaterialSpec('crystal', properties=properties, selection=['all'])],
        physics=[PhysicsSpec(_EM, em_type, settings={**_ALL_PERIODIC, 'wavelength': _WAVELENGTH}, every=_OPTIC_EVERY),
                 PhysicsSpec(_ACOUSTICS, _ACOUSTICS_TYPE, settings=dict(_ALL_PERIODIC), features=[
                     FeatureSpec('heating', 'heat_pulse_source', ['all'],
                                 {'power_density': _OPTIC_POWER, 'pulse_width': _OPTIC_WIDTH,
                                  'delay': _OPTIC_DELAY})])],
        couplings=[CouplingSpec('acousto_optic', ACOUSTO_OPTIC_TYPE_NAME, _ACOUSTICS, _EM)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _OPTIC_DURATION}))


def _optic_specs():
    return {'ez': _optic_spec('em_fdfd_ez'), 'hz': _optic_spec('em_fdfd_hz'),
            'ez_tensor': _optic_spec('em_fdfd_ez', tensor=True), 'hz_tensor': _optic_spec('em_fdfd_hz', tensor=True)}


def _shear_spec(em_type, tensor):
    """Fully periodic glass box with a y-traction line (plane S waves along x), `em_type` every _SHEAR_EVERY and the
    acousto-optic coupling, time_dependent; tensor: + the shear-column / xy-row tensor entries."""
    size_x, size_y = (cells * _CELL for cells in _SHEAR_CELLS)
    properties = {**_GLASS, **(_SHEAR_TENSOR_ENTRIES if tensor else {})}
    return ModelSpec(
        title=f"Self-test: acousto-optic shear column of a plane S wave ({em_type}{', tensor' if tensor else ''})",
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[vertical_line('source_line', cell_centre(_SHEAR_SOURCE_CELL, _CELL), 0.0, size_y)],
        materials=[MaterialSpec('glass', properties=properties, selection=['all'])],
        physics=[PhysicsSpec(_EM, em_type, settings={**_ALL_PERIODIC, 'wavelength': _WAVELENGTH}, every=_SHEAR_EVERY),
                 PhysicsSpec(_ACOUSTICS, _ACOUSTICS_TYPE, settings=dict(_ALL_PERIODIC), features=[
                     FeatureSpec('shear_sheet', 'traction_line_source', ['source_line'],
                                 {'waveform': 'ricker', 'frequency': _SHEAR_FREQUENCY, 'traction': _SHEAR_TRACTION,
                                  'direction': 'y'})])],
        couplings=[CouplingSpec('acousto_optic', ACOUSTO_OPTIC_TYPE_NAME, _ACOUSTICS, _EM)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _SHEAR_DURATION}))


def _shear_specs():
    return {'hz_isotropic': _shear_spec('em_fdfd_hz', False), 'hz_tensor': _shear_spec('em_fdfd_hz', True),
            'ez_tensor': _shear_spec('em_fdfd_ez', True)}


# ---- checks (CPU, on Results-like objects) ----
def _optic_closed_form():
    """(n0, elastic strain xx, yy, dn of eps_zz / eps_xx / eps_yy and Delta eps_xy, isotropic and tensor) of the
    stressed crystal."""
    energy_density = _OPTIC_POWER * 0.5 * math.sqrt(math.pi) * _OPTIC_WIDTH * (
        math.erf((_OPTIC_DURATION - _OPTIC_DELAY) / _OPTIC_WIDTH) + math.erf(_OPTIC_DELAY / _OPTIC_WIDTH))
    stress = -_CRYSTAL['gruneisen'] * energy_density
    c11, c12, c22 = _CRYSTAL['c11'], _CRYSTAL['c12'], _CRYSTAL['c22']
    determinant = c11 * c22 - c12 * c12
    strain_xx = (c22 - c12) * stress / determinant
    strain_yy = (c11 - c12) * stress / determinant
    n0 = float(np.sqrt(np.float32(_CRYSTAL['eps_r'])))
    p11, p12 = _CRYSTAL['photoelastic_p11'], _CRYSTAL['photoelastic_p12']
    scale = -0.5 * n0 ** 3
    shifts = {'eps_r': scale * p12 * (strain_xx + strain_yy),
              'eps_xx': scale * (p11 * strain_xx + p12 * strain_yy),
              'eps_yy': scale * (p11 * strain_yy + p12 * strain_xx),
              'eps_xy': 0.0}                                                  # isotropic xy row: p66 g, g = 0
    p21, p31 = _TENSOR_ENTRIES['photoelastic_p21'], _TENSOR_ENTRIES['photoelastic_p31']
    p61, p62 = _TENSOR_ENTRIES['photoelastic_p61'], _TENSOR_ENTRIES['photoelastic_p62']
    tensor_shifts = {'eps_r': scale * (p31 * strain_xx + p12 * strain_yy),   # p32 unset: p12
                     'eps_xx': shifts['eps_xx'],                              # row xx = p11, p12
                     'eps_yy': scale * (p21 * strain_xx + p11 * strain_yy),   # p22 unset: p11
                     'eps_xy': -n0 ** 4 * (p61 * strain_xx + p62 * strain_yy)}   # Delta eps_xy, not an index shift
    return n0, strain_xx, strain_yy, shifts, tensor_shifts


def _check_acousto_optic(results_by_name):
    """Elastic strain fields, sqrt(eps_ii) and (hz) eps_xy of every window cell against the closed form (relative to
    the change; the isotropic eps_xy = 0 relative to the tensor run's)."""
    n0, strain_xx, strain_yy, shifts, tensor_shifts = _optic_closed_form()
    errors, details = {}, {'n0': n0, 'strain_xx': strain_xx, 'strain_yy': strain_yy, 'index_shifts': shifts,
                           'tensor_index_shifts': tensor_shifts}
    for run, components in (('ez', ('eps_r',)), ('hz', ('eps_xx', 'eps_yy', 'eps_xy')), ('ez_tensor', ('eps_r',)),
                            ('hz_tensor', ('eps_xx', 'eps_yy', 'eps_xy'))):
        results = results_by_name[run]
        expected_shifts = tensor_shifts if run.endswith('_tensor') else shifts
        for name, expected in (('elastic_strain_xx', strain_xx), ('elastic_strain_yy', strain_yy)):
            values = np.asarray(results.field(f'{_ACOUSTICS}.{name}').values, dtype=np.float64)
            errors[f'{run}_{name}'] = float(np.max(np.abs(values - expected))) / abs(expected)
        for component in components:
            values = np.asarray(results.field(f'{_EM}.{component}').values, dtype=np.float64)
            if component == 'eps_xy':
                errors[f'{run}_eps_xy'] = float(np.max(np.abs(values - expected_shifts['eps_xy']))) / abs(
                    tensor_shifts['eps_xy'])
                details[f'{run}_eps_xy_mean'] = float(np.mean(values))
                continue
            index = np.sqrt(values)
            errors[f'{run}_{component}'] = float(np.max(np.abs(index - (n0 + expected_shifts[component])))) / abs(
                expected_shifts[component])
            details[f'{run}_{component}_shift_mean'] = float(np.mean(index)) - n0
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': details['ez_eps_r_shift_mean'], 'expected': shifts['eps_r'],
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _OPTIC_TOLERANCE,
            'details': {'errors': errors, **details}}


def _check_shear_column(results_by_name):
    """Per run: e_xx = e_yy = 0, then sqrt(eps_ii) - n0 and eps_xy against the shear column times g = the 4-corner mean
    of the final sxy / c66 (periodic wrap), each relative to its largest expected change."""
    n0 = float(np.sqrt(np.float32(_GLASS['eps_r'])))
    c66 = _GLASS['rho'] * _GLASS['shear_speed'] ** 2
    p11, p12 = _GLASS['photoelastic_p11'], _GLASS['photoelastic_p12']
    columns = {'isotropic': {'eps_r': 0.0, 'eps_xx': 0.0, 'eps_yy': 0.0, 'eps_xy': 0.5 * (p11 - p12)},
               'tensor': {'eps_r': _SHEAR_TENSOR_ENTRIES['photoelastic_p36'],
                          'eps_xx': _SHEAR_TENSOR_ENTRIES['photoelastic_p16'],
                          'eps_yy': _SHEAR_TENSOR_ENTRIES['photoelastic_p26'],
                          'eps_xy': _SHEAR_TENSOR_ENTRIES['photoelastic_p66']}}
    errors, details = {}, {}
    for run, components in (('hz_isotropic', ('eps_xx', 'eps_yy', 'eps_xy')), ('hz_tensor', ('eps_xx', 'eps_yy',
                                                                                             'eps_xy')),
                            ('ez_tensor', ('eps_r',))):
        results = results_by_name[run]
        corner_stress = np.asarray(results.field(f'{_ACOUSTICS}.sxy').values, dtype=np.float64)
        shear = 0.25 * (corner_stress + np.roll(corner_stress, 1, axis=0) + np.roll(corner_stress, 1, axis=1)
                        + np.roll(np.roll(corner_stress, 1, axis=0), 1, axis=1)) / c66
        largest_shear = float(np.max(np.abs(shear)))
        details[f'{run}_max_shear_strain'] = largest_shear
        if not largest_shear > 0.0:
            errors[f'{run}_shear_strain'] = math.inf
            continue
        for name in ('elastic_strain_xx', 'elastic_strain_yy'):
            values = np.asarray(results.field(f'{_ACOUSTICS}.{name}').values, dtype=np.float64)
            errors[f'{run}_{name}'] = float(np.max(np.abs(values))) / largest_shear
        column = columns['tensor' if run.endswith('_tensor') else 'isotropic']
        # the largest change of the run: every component relative to it (isotropic diagonals must stay n0)
        scale = max(n0 ** 4 * abs(column[name]) if name == 'eps_xy' else 0.5 * n0 ** 3 * abs(column[name])
                    for name in components) * largest_shear
        for component in components:
            values = np.asarray(results.field(f'{_EM}.{component}').values, dtype=np.float64)
            if component == 'eps_xy':
                measured, expected = values, -n0 ** 4 * column['eps_xy'] * shear
            else:
                measured, expected = np.sqrt(values) - n0, -0.5 * n0 ** 3 * column[component] * shear
            errors[f'{run}_{component}'] = float(np.max(np.abs(measured - expected))) / scale
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': error, 'expected': 0.0, 'error': error if math.isfinite(error) else math.inf,
            'tolerance': _SHEAR_TOLERANCE, 'details': {'errors': errors, 'n0': n0, 'c66': c66, **details}}


ACOUSTICS_OPTICS_CASES = (
    SelfTestCase(
        name='acousto_optic_uniform_strain', label='Acoustics -> EM: acousto-optic index of a uniformly stressed crystal',
        description=(f'Two time_dependent runs (em_fdfd_ez, em_fdfd_hz) of a fully periodic {_OPTIC_BOX_CELLS} x '
                     f'{_OPTIC_BOX_CELLS} cell box ({_CELL * 1e9:g} nm cells) of one orthotropic crystal (rho '
                     f"{_CRYSTAL['rho']:g}, c11 / c12 / c22 / c66 = {_CRYSTAL['c11']:g} / {_CRYSTAL['c12']:g} / "
                     f"{_CRYSTAL['c22']:g} / {_CRYSTAL['c66']:g} Pa, gruneisen {_CRYSTAL['gruneisen']:g}, eps_r "
                     f"{_CRYSTAL['eps_r']:g}, p11 {_CRYSTAL['photoelastic_p11']:g}, p12 "
                     f"{_CRYSTAL['photoelastic_p12']:g}) heated by a heat_pulse_source on all ({_OPTIC_POWER:g} W/m^3, "
                     f'width {_OPTIC_WIDTH:g} s, delay 4 widths, duration 8 widths); the FDFD (no source, every '
                     f'{_OPTIC_EVERY}) re-solves on the strain handed over by {ACOUSTO_OPTIC_TYPE_NAME}. Exact: the '
                     'thermoelastic stress sxx = syy = -G H stays uniform (no gradient, no motion), elastic strain = '
                     'the inverse plane-strain stiffness on it (e_zz = 0), eps_ii = (n0 + dn_ii)^2 with dn_ii = -n0^3 / '
                     '2 (p11 e_ii + p12 (e_jj + e_kk)), n0 = sqrt(float32 eps_r). error = max over the window of the '
                     'elastic_strain_xx / _yy errors and |sqrt(eps) - n0 - dn| / |dn| (ez eps_r, hz eps_xx and '
                     f'eps_yy); tolerance {_OPTIC_TOLERANCE:g}. Two more runs (ez_tensor, hz_tensor) add the '
                     'photoelastic tensor entries '
                     + ', '.join(f'{name[-3:]} {value:g}' for name, value in _TENSOR_ENTRIES.items())
                     + ' (p22, p32, p66 unset): dn_xx = -n0^3 / 2 (p11 e_xx + p12 e_yy), dn_yy = -n0^3 / 2 (p21 e_xx '
                     '+ p11 e_yy), dn_zz = -n0^3 / 2 (p31 e_xx + p12 e_yy), the shear columns times the exactly zero '
                     'shear strain, and the em_fdfd_hz eps_xy = -n0^4 (p61 e_xx + p62 e_yy) (the normal-strain xy row; '
                     'exactly 0 in the hz run without the tensor). error also takes max |eps_xy - expected| / |eps_xy '
                     'of the tensor run| of both hz runs. Pins the strain inversion, the slot mapping of p11 / p12 and '
                     'of the tensor rows with their isotropic fallbacks, the p61 / p62 entries into the eps_xy sum '
                     'slot, the E13 index slot and em_fdfd in a time_dependent study (the shear-strain path: '
                     f'acousto_optic_shear_strain). Not yet run at every {_OPTIC_EVERY}.'),
        specs=_optic_specs, check=_check_acousto_optic),
    SelfTestCase(
        name='acousto_optic_shear_strain', label='Acoustics -> EM: acousto-optic shear column and eps_xy of an S wave',
        description=(f'Three time_dependent runs of a fully periodic {_SHEAR_CELLS[0]} x {_SHEAR_CELLS[1]} cell box '
                     f"({_CELL * 1e9:g} nm cells) of an isotropic glass (rho {_GLASS['rho']:g}, sound / shear speed "
                     f"{_GLASS['sound_speed']:g} / {_GLASS['shear_speed']:g} m/s, eps_r {_GLASS['eps_r']:g}, p11 "
                     f"{_GLASS['photoelastic_p11']:g}, p12 {_GLASS['photoelastic_p12']:g}): a y-traction Ricker sheet "
                     f'({_SHEAR_TRACTION:g} Pa, {_SHEAR_FREQUENCY:g} Hz) on a vertical line launches plane S waves along '
                     f'x (the FDFD re-solves every {_SHEAR_EVERY} acoustic steps; the coupling is one-way, so the '
                     'acoustics do not depend on that cadence), so v_y and sxy are the only fields, e_xx = e_yy = 0 exactly and the centre engineering shear g '
                     '= the 4-corner mean of sxy / c66 is the only strain. After the end-of-run flush: hz_isotropic '
                     '(em_fdfd_hz, no tensor) keeps eps_xx = eps_yy = n0^2 and gets eps_xy = -n0^4 (p11 - p12) / 2 g; '
                     'hz_tensor / ez_tensor add the tensor entries '
                     + ', '.join(f'{name[-3:]} {value:g}' for name, value in _SHEAR_TENSOR_ENTRIES.items())
                     + ': sqrt(eps_ii) = n0 - n0^3 / 2 p_i6 g (p16 xx, p26 yy, p36 zz) and eps_xy = -n0^4 p66 g. g '
                     'is recomputed from the sxy field (periodic wrap), so the centre shear kernel, the shear '
                     'columns, the xy row with its isotropic fallback, the eps_xy sum slot and the eps_xy output are '
                     'all pinned. error = the largest deviation relative to the largest change of the run (strains '
                     f'relative to max |g|); tolerance {_SHEAR_TOLERANCE:g}. The normal-strain entries p61 / p62 '
                     'multiply e_xx = e_yy = 0 here (checked by acousto_optic_uniform_strain). Not yet run at every '
                     f'{_SHEAR_EVERY}.'),
        specs=_shear_specs, check=_check_shear_column),
)

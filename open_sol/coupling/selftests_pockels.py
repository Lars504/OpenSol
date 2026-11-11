"""Pockels (electrostatics_em_pockels) self-test runs and check on the y-periodic strip of coupling/selftests.py
(uniform applied field, em_fdfd_hz phase shift; the case is selftests_index_sum's) and the 'ln_phase_modulator'
template; no physics imports, specs built fresh."""
import numpy as np

from ..model.selftest_helpers import with_exact_cg
from ..model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                          MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, StudySpec, SweepSpec)
from .selftests import (_CELL, _CURRENTS_LINE_CELLS, _EM, _K0, _N_CELLS, _SIZE, _SLAB_X_RANGE, _discrete_wavenumber,
                        _em_physics, _full_period_rectangle, _line_geometry, _monitor_phases, _strip_outcome,
                        _summaries, _unconverged_failures, _window_field)

POCKELS_TYPE_NAME = 'electrostatics_em_pockels'   # = definitions.POCKELS_TYPE_NAME (definitions imports this module)
_ELECTROSTATICS = 'electrostatics'                # instance name = type name = its default instance
_REFERENCE = 'reference'
_BIASED = 'biased'
_CRYSTAL = 'crystal'

# pockels_phase_index_sum reference / biased runs: the strip of coupling/selftests.py (8.0 x 0.2 um, y-periodic for EM, CPML along x, 20
# nm cells, mode source cell 50, mode monitors cells 75 / 325) in em_fdfd_hz; every cell has eps_r = n0^2 with n0 =
# n_e = 2.138, so the plane-wave mode sees no index step outside the crystal slab x = 2..6 um (cells 100..299, L = 4
# um), and E_y (tangential at normal incidence) sees eps_yy. Electrostatics: voltage 10 V on y_hi, 0 V on y_lo, faces x
# insulated: V = V_top y / d in every column whatever eps_dc(x) (no x flux), so E_y = -V_top / d = -5e7 V/m exactly in
# every cell (crystal eps_dc 43 / eps_dc_yy 28, cladding 3.9). Crystal z along model y: r_yy = r33 = 31e-12, r_xx =
# r13 = 10e-12 m/V, field_component 'y': dn_yy = -1/2 n0^3 r33 E_y = +7.57e-3 (dn / n = 0.35 %), dn_xx = +2.44e-3;
# dphi ~ -k0 dn_yy L = -0.123 rad. The reference run has r = 0 (same solves, no index change); it is also the
# reference of the index_sum combined run (the same EM problem, eps n0^2 through the same index square).
_INDEX = 2.138                         # n0 = n_e of lithium niobate at 1.55 um
_R33 = 31e-12                          # m/V, on the optical component along the applied field (yy here)
_R13 = 10e-12                          # m/V, on the components across it (xx here)
_VOLTAGE = 10.0                        # V on y_hi (y_lo at 0 V)
_FIELD_Y = -_VOLTAGE / _SIZE[1]        # V/m, -5e7
_INDEX_CHANGE = {'xx': -0.5 * _INDEX ** 3 * _R13 * _FIELD_Y, 'yy': -0.5 * _INDEX ** 3 * _R33 * _FIELD_Y}
_CRYSTAL_CELLS = slice(round(_SLAB_X_RANGE[0] / _CELL), round(_SLAB_X_RANGE[1] / _CELL))   # 100..299
_CRYSTAL_LENGTH = _SLAB_X_RANGE[1] - _SLAB_X_RANGE[0]                                      # 4 um
_PHASE_TOLERANCE = 0.01                # relative to the predicted shift
_FIELD_TOLERANCE = 1e-5                # relative to |E_y|: float32 storage of an exact discrete solution
_EPS_TOLERANCE = 1e-3                  # relative to the eps change of the crystal
_MAX_TICKS = 1                         # feed-forward: electrostatics -> EM, one pass


# ---- specs ----
def _pockels_strip_spec(run_name, biased):
    """em_fdfd_hz + electrostatics strip: crystal slab (r33 on r_yy, r13 on r_xx when biased, else r = 0), plates on
    the y faces, the Pockels coupling with field_component 'y' and a stationary study."""
    slab_x0, slab_x1 = _SLAB_X_RANGE
    crystal = {'eps_r': _INDEX ** 2, 'eps_dc': 43.0, 'eps_dc_yy': 28.0}
    if biased:
        crystal.update(r_yy=_R33, r_xx=_R13)
    return with_exact_cg(ModelSpec(
        title=f'Self-test: Pockels phase in a uniform field ({run_name})',
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=[_full_period_rectangle(_CRYSTAL, slab_x0, slab_x1), *_line_geometry(_CURRENTS_LINE_CELLS)],
        materials=[MaterialSpec('cladding', properties={'eps_r': _INDEX ** 2, 'eps_dc': 3.9}, selection=['all']),
                   MaterialSpec(_CRYSTAL, properties=crystal, selection=[_CRYSTAL])],
        physics=[_em_physics(_CURRENTS_LINE_CELLS, 'em_fdfd_hz'),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('top', 'voltage', ['y_hi'], {'voltage': _VOLTAGE}),
                     FeatureSpec('bottom', 'voltage', ['y_lo'], {'voltage': 0.0})])],
        couplings=[CouplingSpec('pockels', POCKELS_TYPE_NAME, _ELECTROSTATICS, _EM,
                                params={'field_component': 'y'})],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary')))


def _pockels_strip_specs():
    """'reference' (r = 0) then 'biased' (r33 / r13): the same layout, voltages and study."""
    return {_REFERENCE: _pockels_strip_spec(_REFERENCE, False), _BIASED: _pockels_strip_spec(_BIASED, True)}


# ---- check (CPU, on Results-like objects) ----
def _expected_index(component):
    """(nx,) index n0 + dn_<component> in the crystal cells, n0 elsewhere."""
    index = np.full(_N_CELLS[0], _INDEX)
    index[_CRYSTAL_CELLS] += _INDEX_CHANGE[component]
    return index


def _check_pockels_strip(results_by_name):
    """Uniform E_y, eps_xx / eps_yy = (n0 + dn)^2 and the phase shift biased vs reference against the discrete-k
    prediction; error = max relative error in tolerance units + number of failed checks."""
    summaries = _summaries(results_by_name)
    biased = results_by_name[_BIASED]

    # 1. the applied field: E_y = -V / d in every cell, E_x = 0
    field_y = _window_field(biased, f'{_ELECTROSTATICS}.E_y')
    field_x = _window_field(biased, f'{_ELECTROSTATICS}.E_x')
    err_field = max(float(np.max(np.abs(field_y - _FIELD_Y))), float(np.max(np.abs(field_x)))) / abs(_FIELD_Y)

    # 2. the index transform: eps = (n0 + dn)^2 per component, n0^2 outside the crystal
    eps_errors = {}
    for component in ('xx', 'yy'):
        expected = _expected_index(component) ** 2
        change = abs((_INDEX + _INDEX_CHANGE[component]) ** 2 - _INDEX ** 2)
        measured = _window_field(biased, f'{_EM}.eps_{component}')
        eps_errors[component] = float(np.max(np.abs(measured - expected[:, None]))) / change
    err_eps = max(eps_errors.values())

    # 3. phase between the mode monitors, biased minus reference
    phases, n_eff, dphi_measured = _monitor_phases(results_by_name, _BIASED)
    dphi_predicted = -_CRYSTAL_LENGTH * float(_discrete_wavenumber(_INDEX + _INDEX_CHANGE['yy'])
                                              - _discrete_wavenumber(_INDEX))
    dphi_first_order = -_K0 * _INDEX_CHANGE['yy'] * _CRYSTAL_LENGTH   # -k0 1/2 n0^3 r33 |E| L
    err_phase = abs(dphi_measured - dphi_predicted) / abs(dphi_predicted)

    # 4. feed-forward: one pass, no 'did not converge' warning
    failed = _unconverged_failures(summaries)
    for run_name, summary in summaries.items():
        if int(summary['ticks']) > _MAX_TICKS:
            failed.append(f"{run_name}: {int(summary['ticks'])} iterations > {_MAX_TICKS}")

    normalized_errors = {'phase': err_phase / _PHASE_TOLERANCE, 'field': err_field / _FIELD_TOLERANCE,
                         'eps': err_eps / _EPS_TOLERANCE}
    details = {'err_phase': err_phase, 'err_field': err_field, 'err_eps': eps_errors,
               'E_y_closed_V_per_m': _FIELD_Y, 'E_y_mean_V_per_m': float(np.mean(field_y)),
               'dn_closed': dict(_INDEX_CHANGE), 'dphi_measured_rad': dphi_measured,
               'dphi_predicted_rad': dphi_predicted, 'dphi_first_order_rad': dphi_first_order,
               'first_order_relative_difference': abs(dphi_measured - dphi_first_order) / abs(dphi_first_order),
               'phase_positive_rad': phases, 'n_eff': n_eff}
    return _strip_outcome(normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details)


# The Pockels-alone check runs inside pockels_phase_index_sum (selftests_index_sum.py), which shares its reference run.
POCKELS_DESCRIPTION = (
    'Pockels alone (runs reference and biased): the y-periodic strip of em_heat_thermo_optic_strip (8.0 x 0.2 um, '
    'CPML along x, 20 nm mesh) in em_fdfd_hz with eps_r = n0^2, n0 = 2.138 (n_e of lithium niobate) in every cell, a '
    'mode source at 1.01 um (1 W/m, 1.55 um) and mode monitors at 1.51 and 6.51 um. A crystal slab x = 2..6 um (L = 4 '
    'um, crystal z along model y) carries r_yy = r33 = 31e-12 and r_xx = r13 = 10e-12 m/V, eps_dc 43 / eps_dc_yy 28 '
    '(cladding eps_dc 3.9). Electrostatics (float64 CG to 1e-12): 10 V on y_hi and 0 V on y_lo, x faces insulated, so '
    'V = V_top y / d in every column whatever eps_dc(x) and E_y = -V_top / d = -5e7 V/m exactly in every cell. '
    'electrostatics_em_pockels with field_component y writes dn_ii = -1/2 n0^3 r_ii E_y: dn_yy = +7.57e-3 (dn / n = '
    '0.35 %), dn_xx = +2.44e-3. The reference run has r = 0 (same solves, eps = n0^2 through the same index square). '
    'Checks: (1) err_field = max(|E_y + V / d|, |E_x|) / (V / d) over the window, tolerance 1e-5 (an exact discrete '
    'solution stored as float32). (2) err_eps = max over xx, yy of max |eps - (n0 + dn)^2| / |(n0 + dn)^2 - n0^2| '
    '(n0^2 outside the slab), tolerance 1e-3: the index transform, its sign, the component mapping and the n0 '
    'reference. (3) dphi = wrap([phi_down - phi_up]_biased - [phi_down - phi_up]_reference) from the phase_positive '
    'series of the mode monitors against -L [k_num(n0 + dn_yy) - k_num(n0)] with k_num(n) = (2 / dx) arcsin(n k0 dx / '
    '2), the discrete plane-wave k (E_y is tangential at normal incidence, so eps_yy alone sets it), ~ -k0 1/2 n0^3 '
    'r33 |E| L = -0.123 rad; tolerance 1 %: the monitors sit in unperturbed glass (no re-solved mode offset) and the '
    'slab ends reflect (dn / 2n)^2 ~ 1e-5. The continuum first-order value and its relative difference (~0.4 %: the '
    'discrete dispersion and the second order in dn) are in the details. (4) Feed-forward: each run takes 1 '
    'iteration and no warning contains \'did not converge\'; each violation is a failed check. Its error = '
    'max(err_phase / 0.01, err_field / 1e-5, err_eps / 1e-3) + the number of failed checks, tolerance 1.')


# ---- template ----
# ln_phase_modulator (top view, propagation along +y): uniform 50 nm mesh, 8 x 16 um window (160 x 320 cells; 190 x 350
# = 66,500 em_fdfd_hz unknowns with the PML), every edge on a cell edge. x-cut lithium niobate slab x = 3.5..4.5 um (w
# = 1 um, crystal z along x: E_x sees n_e = 2.138, 14 cells per wavelength in n_o = 2.211) along the whole window in
# silica; electrodes x = 1.5..2.5 and 5.5..6.5 um, y = 3..13 um (gap g = 3 um between their inner edges, 1 um from the
# core, where the TE0 tail is ~4e-3 of its peak; L = 10 um): terminals of the electrostatics at +V/2 / -V/2 and
# pec_domain of the EM (ideal metal: no 23 nm gold skin depth to resolve). Mode lines y = 1.025 / 14.975 um (cells 20 /
# 299) over the whole width (open cpml ends). Uniform-field estimate E = V / g: V_pi L = lambda g / (n_e^3 r33) = 1.55 V
# cm, dphi = pi V L / (V_pi L) = 0.20 rad at 100 V; series estimate with the eps_dc contrast (core eps_c 28, silica 3.9)
# E = V / (w + (g - w) 28 / 3.9) = V / 15.4 um, ~5x less (~0.04 rad at 100 V, before the mode overlap).
LN_PHASE_MODULATOR_LABEL = 'Lithium niobate phase modulator'
LN_PHASE_MODULATOR_DESCRIPTION = (
    'Electrostatics + EM FDFD (H_z) with the Pockels coupling, stationary, sweep over V_bias (0, 50, 100 V): an x-cut '
    'lithium niobate slab waveguide (1 um, along y, optic axis along x) in silica between two 10 um electrodes '
    '(terminals at +V_bias/2 and -V_bias/2, 3 um gap; pec_domain in the EM); derived phase (mode_monitor '
    'phase_positive) and E_x_core per value. Uniform-field estimate E = V / gap: V_pi L = lambda gap / (n_e^3 r33) = '
    '1.5 V cm, 0.20 rad at 100 V; the silica gap (eps_dc 3.9 against 28 in the core) takes most of the voltage, so '
    'expect ~5x less (series estimate E = V / (w + (gap - w) 28 / 3.9)).')
LN_PHASE_MODULATOR_SUMMARY = (
    'Electro-optic phase modulator: a voltage across two electrodes beside a lithium niobate waveguide changes its '
    'refractive index, and the guided light picks up a phase that grows with the voltage. Steady state, swept over '
    'three voltages.')
_MODULATOR_CELL = 50e-9                         # m
_MODULATOR_DOMAIN = (8e-6, 16e-6)               # m
_MODULATOR_WAVELENGTH = 1.55e-6                 # m
_MODULATOR_CORE_X = (3.5e-6, 4.5e-6)            # m
_MODULATOR_ELECTRODES_X = {'electrode_plus': (1.5e-6, 2.5e-6), 'electrode_minus': (5.5e-6, 6.5e-6)}   # m
_MODULATOR_ELECTRODE_Y = (3e-6, 13e-6)          # m
_MODULATOR_LINE_Y = {'source_line': 1.025e-6, 'monitor_line': 14.975e-6}   # m, cell centres
_MODULATOR_VOLTAGES = (0.0, 50.0, 100.0)        # V


def ln_phase_modulator_spec():
    """Template 'ln_phase_modulator' (electrostatics + em_fdfd_hz + Pockels, stationary, sweep V_bias): x-cut LN slab
    waveguide between two electrodes, phase of the through mode per voltage."""
    size_x, size_y = _MODULATOR_DOMAIN
    core_x0, core_x1 = _MODULATOR_CORE_X
    electrode_y0, electrode_y1 = _MODULATOR_ELECTRODE_Y

    def rectangle(name, x0, y0, x1, y1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    def mode_line(name, y):
        return GeometrySpec(name, 'line', {'x0': 0.0, 'y0': y, 'x1': size_x, 'y1': y})

    electrodes = [rectangle(name, x0, electrode_y0, x1, electrode_y1)
                  for name, (x0, x1) in _MODULATOR_ELECTRODES_X.items()]
    return ModelSpec(
        title=LN_PHASE_MODULATOR_LABEL,
        parameters=[ParameterSpec('V_bias', _MODULATOR_VOLTAGES[-1], 'voltage between the electrodes [V]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[rectangle('core', core_x0, 0.0, core_x1, size_y),
                  rectangle('core_section', core_x0, electrode_y0, core_x1, electrode_y1),
                  *electrodes, *(mode_line(name, y) for name, y in _MODULATOR_LINE_Y.items())],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('lithium_niobate', library='lithium_niobate_x_cut', selection=['core'])],
        physics=[PhysicsSpec(_EM, 'em_fdfd_hz', settings={'wavelength': _MODULATOR_WAVELENGTH}, features=[
                     FeatureSpec('source', 'mode_source', ['source_line'],
                                 {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}),
                     FeatureSpec('through', 'mode_monitor', ['monitor_line'], {'mode_number': 1}),
                     FeatureSpec('electrodes', 'pec_domain', list(_MODULATOR_ELECTRODES_X))]),
                 PhysicsSpec(_ELECTROSTATICS, _ELECTROSTATICS, features=[
                     FeatureSpec('electrode_plus', 'terminal', ['electrode_plus'], {'voltage': 'V_bias / 2'}),
                     FeatureSpec('electrode_minus', 'terminal', ['electrode_minus'], {'voltage': '-V_bias / 2'})])],
        couplings=[CouplingSpec('pockels', POCKELS_TYPE_NAME, _ELECTROSTATICS, _EM, params={'field_component': 'x'})],
        mesh=MeshSpec(mode='uniform', dx=_MODULATOR_CELL),
        study=StudySpec(type='stationary', sweep=SweepSpec('V_bias', list(_MODULATOR_VOLTAGES))),
        derived_values=[DerivedValueSpec('phase', 'phase_positive', operation='last', source=f'{_EM}.through'),
                        DerivedValueSpec('E_x_core', f'{_ELECTROSTATICS}_E_x', ['core_section'], 'mean')])

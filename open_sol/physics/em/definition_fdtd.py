"""Model definitions of the 2D FDTD physics types 'em_fdtd_ez' (E_z, H_x, H_y) and 'em_fdtd_hz' (H_z, E_x, E_y), built
by one factory: settings, material vocabulary, features; shared callbacks: definition_common / _feature_validation."""
from functools import partial

from ...core.constants import FACES
from ...core.renderer import FieldInfo
from ...core.time_estimate import explicit_step_estimate
from ...model.definition_helpers import STEPPER_ROLE, validate_points_inside
from ...model.schema import FeatureType, Param, PhysicsType
from .definition_common import (BOX_CELLS, CPML_BC_HELP, EM_FAMILY, EZ_CURRENT_TIP, EZ_PROBE_COMPONENT_TIP, FACE_MEAN,
                                FLUX_PARAMS, FREQUENCY_DOMAIN_STUDY, HZ_PROBE_COMPONENT_TIP, MIRROR_BC_HELP,
                                MODE_MONITOR_PARAMS, MODE_MONITOR_RESULT_TEXT, MODE_SOLVE_TEXT, MODE_SOURCE_PARAMS,
                                MODE_SOURCE_RESULT_TEXT, MODE_SOURCE_TEXT, PERIODIC_BC_HELP, PER_CW_TEXT,
                                POINT_SOURCE_SIGN, PROBE_COMPONENT_CHOICE_LABELS, SPAN_CELLS, apply_dft_monitor,
                                apply_ez_point_source, apply_far_field, apply_field_probe, apply_flux_monitor,
                                apply_gaussian_beam, apply_hz_point_source, apply_line_source, apply_mode_monitor,
                                apply_mode_source, apply_pec, apply_tfsf, beam_params, create_em_physics,
                                dft_result_callback, em_material_params, far_field_params,
                                flux_result_callback, hz_point_source_params, make_far_field_result_callback,
                                make_source_shortest_wavelength, make_validate_physics, mode_end_text,
                                mode_monitor_result_callback, mode_source_result_callback, probe_result_callback,
                                settings_params, sheet_amplitude_param, source_params, source_slot_docs,
                                spectral_params, steps_in_time, waveform_params)
from .definition_feature_validation import (make_validate_dft_monitor, make_validate_mode_monitor,
                                            make_validate_mode_source, validate_ez_line_feature,
                                            validate_ez_point_feature, validate_far_field, validate_flux_monitor,
                                            validate_hz_line_feature, validate_hz_point_source, validate_tfsf)
from .definition_harmonic_fdtd import HARMONIC_FDTD_TEXT, make_validate_harmonic_fdtd
from .definition_resonance import resonance_monitor_feature
from .definition_bloch import bloch_active_axes, bloch_params, fdtd_domain_requirements, with_bloch_validation
from .definition_effective_index import effective_index_feature
from .definition_nonlinear import gain_medium_feature, nonlinear_medium_feature
from .definition_phase_change import phase_change_medium_feature
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import (dft_monitor_memory_estimate, far_field_memory_estimate, fdtd_memory_estimate,
                               line_spectra_memory_estimate, pec_memory_estimate)
from .renderer import EMRenderer, EMRendererHz

_Q_ACCUM_INFO = FieldInfo('q_accum', 'Absorbed energy density', unit='J/m^3', cmap='magma')
_JZ_TOTAL_INFO = FieldInfo('jz_total', 'Dispersive current density J_z', unit='A/m^2', signed=True, cmap='RdBu_r')


# ---- settings ----
# polarisation -> (wall text, mirror parity text, re-pin text) between the shared boundary help pieces
_BC_TEXTS = {
    'ez': ('pec (E_z = 0 at the centre of the outermost domain cell) and pmc (tangential H = 0 on the inner face of that '
           'cell) are perfect electric / magnetic walls up to one cell inside the domain edge.',
           'For E_z: symmetric = E_z even about the plane (E_z of the ghost cell copied from its neighbour, as pmc: a '
           'PMC plane), antisymmetric = E_z odd (copied negated, so E = 0 on the plane: a PEC plane).',
           'These walls are re-applied after the sources are injected, so a source value in the outermost cell of a pec '
           'or pmc face is overwritten every step and radiates nothing, while a source next to a mirror plane sees its '
           'exact image.'),
    'hz': ('pec (tangential E = 0 on the inner face of the outermost domain cell, H_z mirrored into that cell) and pmc '
           '(H_z = 0 at the centre of that cell) are perfect electric / magnetic walls up to one cell inside the domain '
           'edge.',
           'For H_z: symmetric = H_z odd about the plane; antisymmetric = H_z even with tangential E = 0 on the plane (the '
           'same as pec on that face).',
           'These walls are re-applied after the sources are injected, so a source value on a pinned row or face is '
           'overwritten every step and radiates nothing (validation warns): H_z in the outermost cell of a pec or pmc '
           'face (a magnetic_current point source there), and the tangential E on the inner face of that cell of a pec '
           'face or on the plane of an antisymmetric one (a line source / gaussian beam sheet on that face, and the '
           'current parallel to the face of an electric_dipole in one of the two outermost cells).'),
}
# polarisation -> (permittivity in the stability bound, its name in 'plain ... values', the runtime-rewritten arrays)
_COURANT_TEXTS = {'ez': ('eps_r', 'eps_r', 'eps_r'), 'hz': ('min(eps_xx, eps_yy)', 'permittivity', 'eps_xx / eps_yy')}
_TRACK_ABSORPTION_HELP = {
    'ez': ('Accumulate J.E losses (output absorbed_power in W/m^3, state q_accum in J/m^3) also '
           'without an em_heat_joule coupling; costs about one extra GPU pass per step over the '
           'bounding boxes of the conducting and dispersive cells. The state q_accum exists only '
           'while accumulation runs (jz_total only with a Drude pole). It is switched on '
           'automatically when a material expression, a derived value, a field_transfer, '
           'em_heat_joule or em_semiconductor_generation reads q_accum, or a derived value reads absorbed_power '
           '(else absorbed_power is 0 everywhere). absorbed_power shows the current '
           'accumulation window, else the last completed one; em_heat_joule resets q_accum at each '
           'transfer, otherwise the window spans the whole run.'),
    'hz': ('Accumulate J.E losses J_x E_x + J_y E_y + sigma_e |E|^2 on the cell faces (output '
           'absorbed_power in W/m^3, state q_accum in J/m^3) also without an em_heat_joule coupling; '
           'costs about one extra full-grid GPU pass per step. The conductivity loss of a face goes '
           "to each of its two cells by that cell's own sigma_e; the loss of a pole current goes to "
           'the cells of the dispersive material (half / half between two different dispersive '
           'cells). The state q_accum exists only while accumulation runs; it is switched on '
           'automatically when a material expression, a derived value, a field_transfer, em_heat_joule or '
           'em_semiconductor_generation reads q_accum, or a derived value reads absorbed_power (else '
           'absorbed_power is 0 everywhere). absorbed_power shows the current accumulation window, else '
           'the last completed one; em_heat_joule resets q_accum at each transfer, otherwise the '
           'window spans the whole run.'),
}


def _settings(polarization):
    """Boundary, Courant and absorption-tracking settings of em_fdtd_<polarization>."""
    walls, mirrors, repin = _BC_TEXTS[polarization]
    bc_help = ' '.join((CPML_BC_HELP, walls, PERIODIC_BC_HELP, MIRROR_BC_HELP, mirrors, repin))
    eps_product, plain_values, rewritten = _COURANT_TEXTS[polarization]
    courant_help = ('Time step as a fraction of the 2D CFL limit: dt = factor / (C0 sqrt(1/dx^2 + 1/dy^2)) on a uniform '
                    'mesh (factor * dx / (C0 sqrt(2)) for square cells; on a graded mesh the limit of the finest cells); '
                    '0.99 default. Stable only where '
                    f'{eps_product} * mu_r >= (C0 dt)^2 (1/dx^2 + 1/dy^2) (= factor^2 on a uniform mesh; coarser '
                    'cells of a graded mesh have margin) + mu_r P, P = (omega_p dt)^2 / 4 per Drude pole and '
                    'delta_eps (omega_0 dt)^2 / (4 - (omega_0 dt)^2) per Lorentz pole: plain '
                    f'{plain_values} and mu_r values must be >= 1, but an eps_inf ~1 metal needs a lower factor at '
                    'coarse cells. The build checks every cell (field-dependent values, Drude eps_inf < 1, poles), '
                    f'names the worst one and the largest stable factor, but runtime {rewritten} rewrites '
                    '(thermo-optic) are not checked.')
    return settings_params(bc_help, courant_help, _TRACK_ABSORPTION_HELP[polarization]) + bloch_params()


# ---- material vocabulary ----
_MATERIAL_HELPS = {
    'ez': {
        'eps_r': 'Real relative permittivity (>= 1), n^2 of a lossless dielectric: glass ~2.1-2.25, silicon ~12.1 at '
                 '1.55 um; may depend on x, y and physics fields. E_z sees eps_zz instead when the material sets it. '
                 'For a dispersive material it is the background eps_inf: an explicit eps_r (or eps_zz) wins over the '
                 'Drude eps_inf, which applies only when neither is set. A state-dependent eps_r (e.g. of T) is '
                 'rewritten once per step of the physics it reads, not every EM step.',
        'sigma_e': 'Ohmic loss J = sigma_e E_z in S/m, default 0. A weak absorber with index n and extinction '
                   'coefficient kappa at angular frequency omega has sigma_e = 2 n kappa omega eps0; model metals with '
                   'a Drude pole, not their DC conductivity.',
        'mu_r': 'Real relative permeability (>= 1); 1 for practically every optical material. It enters the Courant '
                'check (eps_r * mu_r) and the local impedance of line sources.',
        'sigma_m': 'Magnetic loss on H_x, H_y in Ohm/m, default 0; sigma_m / (mu0 mu_r) = sigma_e / (eps0 eps_r) gives '
                   'an impedance-matched absorber without reflection at normal incidence.',
    },
    'hz': {
        'eps_r': 'Isotropic in-plane relative permittivity (>= 1): sets eps_xx and eps_yy unless the material sets '
                 'those components (a component wins over eps_r). n^2 of a lossless dielectric: glass ~2.1-2.25, '
                 'silicon ~12.1 at 1.55 um; may depend on x, y and physics fields. E_x / E_y sit on cell faces and '
                 f'{FACE_MEAN} (solver_defaults.subpixel_smoothing averages interface faces anisotropically). For a '
                 'dispersive material it is the background eps_inf: an '
                 'explicit permittivity wins over the Drude eps_inf, which applies only when none is set. A '
                 'state-dependent eps_r (e.g. of T) is rewritten once per step of the physics it reads, not every EM '
                 'step.',
        'sigma_e': f'Ohmic loss J = sigma_e E on E_x and E_y in S/m, default 0 ({FACE_MEAN}). A weak absorber with '
                   'index n and extinction coefficient kappa at angular frequency omega has sigma_e = 2 n kappa omega '
                   'eps0; model metals with a Drude pole, not their DC conductivity.',
        'mu_r': 'Real relative permeability (>= 1); H_z sees mu_zz = mu_r. 1 for practically every optical material. '
                'It enters the Courant check (min(eps_xx, eps_yy) * mu_r) and the local impedance of line sources.',
        'sigma_m': 'Magnetic loss on H_z in Ohm/m, default 0; sigma_m / (mu0 mu_r) = sigma_e / (eps0 eps_r) gives an '
                   'impedance-matched absorber without reflection at normal incidence.',
    },
}


# ---- feature parameters ----
_EZ_POINT_SOURCE_PARAMS = waveform_params() + (
    Param('current', 'float', 1e-3, label='Electric line current', unit='A', tip=EZ_CURRENT_TIP,
          help='Peak z-directed filament current I. The radiated field does not depend on the mesh: for cw in a '
               'homogeneous non-magnetic medium of wavenumber k, |E_z| = omega mu0 I / 4 |H0^(2)(k r)| (1 mA at '
               f'wavelength 1 um: ~2e5 V/m at r = 1 um). Fields are linear, so results scale with I. '
               f'{POINT_SOURCE_SIGN}'),)


def _line_source_params(polarization):
    """Waveform and amplitude of a line source in `polarization`."""
    return waveform_params() + (sheet_amplitude_param(polarization, 'fdtd', beam=False),)


def _gaussian_beam_params(polarization):
    """Waveform, amplitude, waist and centre of a gaussian beam in `polarization`."""
    return waveform_params() + (sheet_amplitude_param(polarization, 'fdtd', beam=True),) + beam_params()


_TFSF_PARAMS = {
    'ez': source_params('Peak incident E_z of the +x plane wave (computed for vacuum); the scattered field outside the '
                        'box is relative to it. Typical 1 V/m.',
                        'Peak out-of-plane electric field of the incoming plane wave. Fields scale linearly with it; '
                        '1 is typical.'),
    'hz': source_params('Peak incident E_y of the +x plane wave (computed for vacuum, H_z = E_y / eta0); the scattered '
                        'field outside the box is relative to it. Typical 1 V/m.',
                        'Peak in-plane electric field (along y) of the incoming plane wave. Fields scale linearly with '
                        'it; 1 is typical.'),
}
_PROBE_PARAMS = {
    'ez': (Param('component', 'choice', 'ez', label='Field component', choices=('ez', 'hx', 'hy'),
                 choice_labels=PROBE_COMPONENT_CHOICE_LABELS['ez'], tip=EZ_PROBE_COMPONENT_TIP,
                 help='ez in V/m, hx or hy in A/m, sampled in the cell containing the point after every step; hx / hy '
                      'sit on the +y / +x face of that cell, half a time step before the recorded time.'),),
    'hz': (Param('component', 'choice', 'hz', label='Field component', choices=('hz', 'ex', 'ey'),
                 choice_labels=PROBE_COMPONENT_CHOICE_LABELS['hz'], tip=HZ_PROBE_COMPONENT_TIP,
                 help='hz in A/m at the centre of the cell containing the point, half a time step before the recorded '
                      'time; ex / ey in V/m on the +y / +x face of that cell (raw staggered samples, not averaged). '
                      'Sampled after every step.'),),
}
_DFT_N_WAVELENGTHS_TIP = 'Each wavelength needs graphics card memory for the whole rectangle; 1 to 21 is typical.'
_DFT_DOWNSAMPLE_PARAM = Param(
    'downsample', 'int', 1, label='Record every n-th cell', min=1,
    tip='Records only every n-th cell along x and along y, so a large rectangle needs about n x n times less graphics '
        'card memory and work per time step.\n\n1 records every cell. The map then shows one sample per n x n block of '
        'cells, so details smaller than the blocks are missed.',
    help='Stride s in cells along x and y: the monitor records cells x_start + min(s, box cells) // 2 + k s (likewise '
         'in y), point samples at those cell centres (the hz in-plane E is still the face mean onto each sampled '
         'centre), no averaging over the s x s block; memory and work per step drop by ~s^2. x_centers / y_centers of '
         'the result are the sampled centres and shape the recorded counts; extent stays the whole box. 1 = every '
         'cell. Features smaller than s cells alias.')
_DFT_PARAMS = {
    'ez': spectral_params(
        5, 'GPU memory: 2 * n_wavelengths * box cells * 4 B / downsample^2 (1000 x 1000 cells at 10 wavelengths: 80 '
           'MB), and every wavelength adds work per time step (validation warns above 1 GiB); 1-21 typical.',
        _DFT_N_WAVELENGTHS_TIP) + (_DFT_DOWNSAMPLE_PARAM,),
    'hz': spectral_params(
        5, 'GPU memory: 6 * n_wavelengths * box cells * 4 B / downsample^2 (H_z, E_x and E_y spectra; 1000 x 1000 '
           'cells at 10 wavelengths: 240 MB), and every wavelength adds work per time step (validation warns above 1 '
           'GiB); 1-21 typical.', _DFT_N_WAVELENGTHS_TIP) + (_DFT_DOWNSAMPLE_PARAM,),
}
_FAR_FIELD_SPECTRA = {'ez': 'E_z / H', 'hz': 'H_z / E'}


# ---- descriptions ----
# polarisation ->(out-of-plane field in the line cells, in-plane field on their face) of the mode features
_PORT_FIELDS = {'ez': ('E_z', 'H_y (vertical line) / H_x (horizontal line)'),
                'hz': ('H_z', 'E_y (vertical line) / E_x (horizontal line)')}
_PROBE_UNITS = {'ez': 'V/m or A/m', 'hz': 'A/m or V/m'}
_TFSF_TAIL = ('Uniform or graded meshes (the 1D incident line runs on the grid\'s own x widths, so graded x and y '
              'cells leak nothing); the incident field is computed for vacuum, so the box edges must lie in vacuum '
              '(warning otherwise), and the box needs one scattered-field cell inside the domain on every side, '
              'except at a y mirror face of the wave\'s parity (ez: symmetric, hz: antisymmetric), where it may reach '
              'the plane: the box opens there and its mirror image completes it (the other parity, or an x mirror '
              'face, is an error). Keep '
              'flux and mode monitors entirely inside the box (their hi face too) or outside it (validation warns). '
              f'The box covers {BOX_CELLS}.')


def _mode_source_description(polarization):
    """Description of the mode_source feature in `polarization`."""
    cell_field, face_field = _PORT_FIELDS[polarization]
    return ('Port source launching one guided mode of the cross-section under an axis-aligned line in one direction: '
            'a vertical line (x = const) launches toward +x (direction positive) or -x (negative), a horizontal one '
            f'toward +y or -y. The mode drives {cell_field} in the line cells and {face_field} on their upstream face, '
            'i.e. the lo face of the line cells for positive and their hi face for negative. '
            f"{MODE_SOURCE_TEXT} {MODE_SOLVE_TEXT} {mode_end_text(polarization, 'fdtd')} The line covers "
            f'{SPAN_CELLS}. {MODE_SOURCE_RESULT_TEXT}')


def _mode_monitor_description(polarization):
    """Description of the mode_monitor feature in `polarization`."""
    cell_field, face_field = _PORT_FIELDS[polarization]
    return ('Port monitor splitting the field on an axis-aligned line into the forward (positive: toward +x for a '
            'vertical line, +y for a horizontal one) and backward (negative) amplitudes a of one guided mode at every '
            f'monitor wavelength: running DFTs of {cell_field} in the line cells and of {face_field} on their hi face '
            'are projected on the mode solved at that wavelength, so each direction is measured separately, unlike a '
            f"flux_monitor. {MODE_SOLVE_TEXT} {mode_end_text(polarization, 'fdtd')} The line covers {SPAN_CELLS}. "
            f'{MODE_MONITOR_RESULT_TEXT}')


_DESCRIPTIONS = {
    'ez': {
        'point_source': 'z-directed filament current I in the cell containing the point (a coordinate exactly on a '
                        'cell edge goes to the upper cell). It radiates a cylindrical wave whose field does not depend '
                        'on the mesh: |E_z| = omega mu0 I / 4 |H0^(2)(k r)| in a homogeneous non-magnetic medium.',
        'line_source': 'Sheet current on an axis-aligned line radiating amplitude V/m of E_z to each side in the local '
                       'medium: a vertical line (x = const) sends waves along +x and -x, a horizontal one along +y and '
                       '-y. For a plane wave, span the whole domain: an end on a cpml domain edge continues through the '
                       'PML (on a periodic axis span the whole period); a shorter line diffracts like a slit. The line '
                       f'covers {SPAN_CELLS}.',
        'gaussian_beam': 'Line source with the profile exp(-((s - center) / waist)^2) along the line, radiating a beam '
                         'of peak E_z = amplitude V/m to each side in the local medium (vertical line: along +x and -x; '
                         f'horizontal: along +y and -y); the waist lies on the line. The line covers {SPAN_CELLS}.',
        'plane_wave_tfsf': 'Plane wave travelling along +x only, injected on the outline of a total-field box: total '
                           'field inside the box, only the scattered field outside; amplitude = incident E_z. '
                           f'{_TFSF_TAIL}',
        'pec_domain': 'Perfect electric conductor: E_z = 0 (and no dispersive current) in every cell whose centre '
                      'lies in the selected domains, staircased to the mesh; with solver_defaults.conformal_boundaries '
                      'the H links from free cells into it put E_z = 0 at the drawn wall instead (exact wall '
                      'distance, floored per cell so the time step stays stable).',
        'flux_monitor': 'Spectral Poynting flux through an axis-aligned line: the Yee discrete flux through the hi face '
                        'of the line cells, > 0 toward +x for a vertical line and toward +y for a horizontal one. power '
                        'is in a.u. per unit z length and not normalised by the source spectrum, so compare runs by '
                        'ratios (transmission = device run / reference run with the same source); power_per_cw = power '
                        f'/ (2 |S(omega)|^2) is the cw power in W/m. {PER_CW_TEXT} The line covers {SPAN_CELLS}.',
        'dft_monitor': 'Running DFT of E_z over a rectangle, accumulating exp(-i omega t) at the actual sample times: '
                       '|E_z(omega)|^2 per wavelength in a.u., not normalised by the source spectrum (one wavelength '
                       'per freq_index); intensity x intensity_per_cw_factor[freq_index] (1 / |S(omega)|^2 per '
                       f'wavelength) is the cw |E_z|^2 in (V/m)^2. {PER_CW_TEXT} The box covers {BOX_CELLS}.',
        'far_field': 'Near-to-far-field transform of E_z / H on the outline of a rectangle: radiated power per angle '
                     'rho |E_z|^2 / eta0 (a.u.: dt-weighted spectra without the 1/2, so its angle integral matches a '
                     'closed flux box) versus angle, counter-clockwise from +x, per wavelength. Enclose every scatterer '
                     'with the box; with a TFSF source keep it outside the TFSF box, in the scattered field: its x0 / '
                     'y0 edges at least one cell before those of the TFSF box and its x1 / y1 edges in the last TFSF '
                     'cell or beyond (the TFSF outline faces hold scattered field; validation warns otherwise). The '
                     'closed contour runs through the centres of the first box cells and of the cells one beyond x1 / '
                     'y1 (keep the box one cell off the x_hi / y_hi edges), H averaged from the faces on both sides '
                     'onto the centres; graded meshes work (actual centres, trapezoid weights of their spacing); the '
                     f'box covers {BOX_CELLS}.',
    },
    'hz': {
        'point_source': 'Current-calibrated point excitation in the cell containing the point (a coordinate exactly on '
                        'a cell edge goes to the upper cell); its field does not depend on the mesh. magnetic_current: '
                        'a z-directed magnetic line current K_m [V] added to H_z at the cell centre, radiating |H_z| = '
                        'omega eps0 eps_r K_m / 4 |H0^(2)(k r)|. electric_dipole: an in-plane electric current element '
                        'of strength M [A] along orientation, split over the two E_x faces (+y / -y) and the two E_y '
                        'faces (+x / -x) of the cell, radiating |H_z| = k M / 4 |H1^(2)(k r)| |sin(phi - orientation)|. '
                        'The wall boundaries overwrite some values every step (warning): a magnetic_current in the '
                        'outermost cell of a pec, pmc, symmetric or antisymmetric face radiates nothing, and so does '
                        'the current parallel to a pec or antisymmetric face of an electric_dipole in one of the two '
                        'outermost cells.',
        'line_source': 'Electric sheet current on an axis-aligned line radiating amplitude V/m of tangential E to each '
                       'side in the local medium: a vertical line (x = const) drives E_y and sends waves along +x and '
                       '-x, a horizontal one drives E_x and sends them along +y and -y. The sheet sits on the hi face '
                       'of the line cells (+x face of a vertical line, +y face of a horizontal one), and the local '
                       'impedance uses that face permittivity (width-weighted mean of its two cells). For a plane '
                       'wave, span the whole domain: an end on a cpml domain edge continues through the PML (on a '
                       'periodic axis span the whole period); a shorter line diffracts like a slit. The line covers '
                       f'{SPAN_CELLS}. The line must not lie in the last cell row / column before a non-periodic hi '
                       'face without PML: its hi face is then the outer grid wall. Next to a pec or antisymmetric face '
                       'the sheet must not be the inner face of the outermost cell either (the line in the outermost lo '
                       'cell or the last-but-one hi cell): the wall pins that E face to 0 every step, so the line '
                       'radiates nothing (warning).',
        'gaussian_beam': 'Line source with the profile exp(-((s - center) / waist)^2) along the line, radiating a beam '
                         'of peak tangential E = amplitude V/m to each side in the local medium (vertical line: E_y, '
                         'along +x and -x; horizontal: E_x, along +y and -y); the waist lies on the line and the sheet '
                         'on the hi face of the line cells, so the line must not lie in the last cell row / column '
                         'before a non-periodic hi face without PML (the outer grid wall), nor, next to a pec or '
                         'antisymmetric face, put the sheet on the inner face of the outermost cell, which the wall '
                         'pins to 0 every step (warning: the beam radiates nothing). The line covers '
                         f'{SPAN_CELLS}.',
        'plane_wave_tfsf': 'Plane wave travelling along +x only (E_y, H_z = E_y / eta0), injected on the outline of a '
                           'total-field box: total field inside the box (its H_z cells), only the scattered field '
                           'outside (the E faces on the outline included); amplitude = incident E_y. '
                           f'{_TFSF_TAIL}',
        'pec_domain': 'Perfect electric conductor: tangential E = 0 on every face of each cell whose centre lies in '
                      'the selected domains (E_x on its +y and -y faces, E_y on its +x and -x faces), with no '
                      'dispersive current on those faces; staircased to the mesh. With '
                      'solver_defaults.conformal_boundaries (Dey-Mittra) only faces wholly inside the drawn metal are '
                      'PEC and each cut cell updates H_z from its free face lengths and free area (enlarged where '
                      'needed so the time step stays stable).',
        'flux_monitor': 'Spectral Poynting flux through an axis-aligned line: S_x = Re(E_y H_z*) through a vertical '
                        'line, S_y = -Re(E_x H_z*) through a horizontal one, > 0 toward +x / +y. E is taken on the hi '
                        'face of the line cells and H_z at their centres (time-centred, not interpolated in space). '
                        'power is in a.u. per unit z length and not normalised by the source spectrum, so compare runs '
                        'by ratios (transmission = device run / reference run with the same source); power_per_cw = '
                        f'power / (2 |S(omega)|^2) is the cw power in W/m. {PER_CW_TEXT} The line covers {SPAN_CELLS}.',
        'dft_monitor': 'Running DFT of H_z, E_x and E_y over a rectangle, accumulating exp(-i omega t) at the actual '
                       'sample times (H_z half a step earlier), with E_x / E_y averaged onto the cell centres: '
                       'intensity = |E_x(omega)|^2 + |E_y(omega)|^2 and hz_intensity = |H_z(omega)|^2 per wavelength in '
                       'a.u., not normalised by the source spectrum (one wavelength per freq_index); intensity x '
                       'intensity_per_cw_factor[freq_index] (1 / |S(omega)|^2 per wavelength) is the cw |E|^2 in '
                       f'(V/m)^2. {PER_CW_TEXT} The box covers {BOX_CELLS}.',
        'far_field': 'Near-to-far-field transform of H_z / E on the outline of a rectangle: radiated power per angle '
                     'rho |eta0 H_z|^2 / eta0 (the far-field |E|^2 in vacuum; a.u.: dt-weighted spectra without the '
                     '1/2, so its angle integral matches a closed flux box) versus angle, counter-clockwise from +x, '
                     'per wavelength. Enclose every scatterer with the box; with a TFSF source keep it outside the TFSF '
                     'box, in the scattered field: its x0 / y0 edges at least one cell before those of the TFSF box and '
                     'its x1 / y1 edges in the last TFSF cell or beyond (the TFSF outline faces hold scattered field; '
                     'validation warns otherwise). The closed contour runs through the centres of the first box cells '
                     'and of the cells one beyond x1 / y1 (keep the box one cell off the x_hi / y_hi edges), E averaged '
                     'from the faces on both sides onto the centres; graded meshes work (actual centres, trapezoid '
                     f'weights of their spacing); the box covers {BOX_CELLS}.',
    },
}
_DFT_SUMMARY = ('Stores maps of the {fields} intensity over a rectangle at chosen wavelengths, collected over the whole '
                'run. Use it to see resonances, hot spots and mode shapes.')
_SHARED_SUMMARIES = {   # GUI summaries of the features, the same for both polarisations
    'line_source': ('Launches a wave to both sides of a straight line. Across the whole domain it gives a plane wave; '
                    'a shorter line diffracts like a slit.'),
    'gaussian_beam': ('Launches a beam with a bell-shaped profile to both sides of a straight line, narrowest on the '
                      'line. Use it to light an object with a beam of finite width.'),
    'plane_wave_tfsf': ('Sends a plane wave toward +x through the selected rectangle: inside it the full field, '
                        'outside only the light scattered by objects inside. The rectangle edges must lie in vacuum.'),
    'mode_source': ('Launches one guided mode of a waveguide in one direction from a line across the guide. Pair it '
                    'with mode monitors for the transmission and reflection of waveguide devices.'),
    'pec_domain': ('Turns the selected domains into a perfect metal with no field inside, following the mesh cells. '
                   'Use it for idealised metal walls, mirrors and cavities.'),
    'field_probe': ('Records one field component at a point after every time step, as a time trace. Use it to watch a '
                    'signal arrive or a resonator ring down.'),
    'flux_monitor': ('Records the power crossing a straight line at each wavelength, positive toward +x or +y. For '
                     'transmission, divide by the same monitor in a reference run without the device.'),
    'mode_monitor': ('Measures the power of one guided mode crossing a line, separately per direction and wavelength. '
                     'With a single mode source it also gives the transmission and reflection into that mode.'),
    'far_field': ('Computes the power radiated in each direction, per wavelength, from the fields on the outline of a '
                  'rectangle. Enclose every scattering object and keep it outside any plane wave box.'),
}
_SUMMARIES = {
    'ez': {**_SHARED_SUMMARIES,
           'point_source': ('A tiny wire out of the plane carrying an alternating current, which radiates a '
                            'cylindrical wave. Use it to excite resonators or to test a structure.'),
           'dft_monitor': _DFT_SUMMARY.format(fields='electric field')},
    'hz': {**_SHARED_SUMMARIES,
           'point_source': ('Radiates from one point, either as a magnetic line current out of the plane or as an '
                            'in-plane electric dipole. Use it to excite resonators or particles.'),
           'dft_monitor': _DFT_SUMMARY.format(fields='electric and magnetic field')},
}
_PHYSICS_LABELS = {
    'ez': 'Electromagnetic Waves, time domain (2D, electric field out of plane)',
    'hz': 'Electromagnetic Waves, time domain (2D, magnetic field out of plane)',
}
_PHYSICS_SUMMARIES = {
    'ez': ('Light in 2D, simulated step by step in time, so one short pulse gives a whole spectrum. The electric field '
           'points out of the plane: the simpler choice for dielectric waveguides, resonators and photonic crystals.'),
    'hz': ('Light in 2D, simulated step by step in time, so one short pulse gives a whole spectrum. The magnetic field '
           'points out of the plane and the electric field lies in it: choose it for metals, plasmonics and light '
           'polarised in the plane.'),
}
_PHYSICS_DESCRIPTIONS = {
    'ez': ('2D finite-difference time-domain solver for the E_z polarisation: E_z out of plane in V/m, H_x and H_y in '
           'plane in A/m, x to the right and y up. On the Yee grid E_z sits at cell centres and H_x / H_y on the +y / '
           '+x cell faces, half a time step earlier. gaussian and cw sources use a sin carrier (ricker is a real '
           'wavelet), DFT-based monitors accumulate exp(-i omega t) at the actual sample times, and every wavelength '
           'is a vacuum wavelength. Boundaries: CPML (cells added outside the domain), PEC, PMC, periodic and mirror '
           'planes on the domain edge; materials: eps_r (eps_zz wins for E_z), sigma_e, mu_r, sigma_m plus Drude / Lorentz poles; '
           'current-calibrated sources and mode (port) sources; probe, flux, mode, DFT and far-field monitors. Outputs '
           'e_abs, e_intensity and h_abs (|E|, |E|^2, |H| at the cell centres; instantaneous values, the FDFD types '
           'give phasor moduli under the same names). It advances one Courant step per runner tick, so every must be '
           "1; the instance is usually named 'em'."),
    'hz': ('2D finite-difference time-domain solver for the H_z polarisation: H_z out of plane in A/m, E_x and E_y in '
           'plane in V/m, x to the right and y up. On the Yee grid H_z sits at cell centres half a time step before '
           'the recorded time, E_x on the +y and E_y on the +x cell faces. Each face takes the arithmetic mean of the '
           'permittivity, conductivity and dispersive susceptibility of its two cells, weighted by their widths across '
           'the face (yee_grid.face_average; solver_defaults.subpixel_smoothing averages the permittivity of '
           'interface faces anisotropically instead): curved interfaces are otherwise staircased and plasmon '
           'resonances shift on coarse meshes, so use cells of a few nm for plasmonics. In-plane E is what gives '
           'particle plasmons and guided modes whose E lies in the propagation plane (e.g. surface plasmon polaritons '
           'on metal films). gaussian and cw sources use a sin carrier (ricker is a real wavelet), DFT-based monitors '
           'accumulate exp(-i omega t) at the actual sample times, and every wavelength is a vacuum wavelength. '
           'Boundaries: CPML (cells added outside the domain), PEC, PMC, periodic and mirror planes on the domain '
           'edge; materials: eps_r (or eps_xx, eps_yy), sigma_e, mu_r, sigma_m plus Drude / Lorentz poles; current-calibrated '
           'sources (magnetic line current, electric dipole, sheet currents, TFSF plane wave) and mode (port) sources; '
           'probe, flux, mode, DFT and far-field monitors. Outputs e_abs, e_intensity and h_abs (|E|, |E|^2, |H| at '
           'the cell centres; instantaneous values, the FDFD types give phasor moduli under the same names). It '
           'advances one Courant step per runner tick, so every must be 1. A model with an em_fdtd_ez physics too '
           '(e.g. unpolarised light as one physics per polarisation) needs another instance name (e.g. em_hz) and the '
           'same bc_* / pml_* settings: both share one boundary configuration.'),
}


_STEPPER_TEXT = ('Stepper solver (role stepper, explicit_kernels): precision single (float32 fields, default) or double '
                 '(float64 fields, coefficients, CPML and pole arrays: ~2x the field memory and time, for long or high-Q '
                 'ring-downs where float32 coefficient rounding, ~1e-7 of a mode frequency, accumulates into phase '
                 'drift); materials, absorbed power, probe traces and field-map spectra stay float32, and '
                 'nonlinear_medium / gain_medium need single.')


# ---- type factory ----
# polarisation -> (renderer, permittivity slot -> MaterialGrid array, renderer state field names, extra state fields,
# point-source params / apply / validate, line-feature validate); material_schema = EM_MATERIAL_SCHEMAS
_POLARIZATIONS = {
    'ez': (EMRenderer, {'eps_r_update': 'eps_r'}, ('ez', 'hx', 'hy', 'eps_r', 'sigma_e'),
           (_JZ_TOTAL_INFO, _Q_ACCUM_INFO),
           _EZ_POINT_SOURCE_PARAMS, apply_ez_point_source, validate_ez_point_feature, validate_ez_line_feature),
    'hz': (EMRendererHz, {'eps_xx_update': 'eps_xx', 'eps_yy_update': 'eps_yy'}, ('hz', 'eps_xx', 'eps_yy', 'sigma_e'),
           (_Q_ACCUM_INFO,),
           hz_point_source_params(waveform_params(), time_domain=True), apply_hz_point_source,
           validate_hz_point_source, validate_hz_line_feature),
}


def _fdtd_physics_type(polarization):
    """PhysicsType 'em_fdtd_<polarization>' ('ez' | 'hz')."""
    (renderer_class, slot_arrays, state_field_names, extra_state_fields, point_source_params, apply_point_source,
     validate_point_source, validate_line) = _POLARIZATIONS[polarization]
    descriptions = _DESCRIPTIONS[polarization]
    summaries = _SUMMARIES[polarization]
    hi_face = polarization == 'hz'   # H_z sheet currents act on the hi face of the line cells
    nf2ff_params = far_field_params(f'Each wavelength accumulates {_FAR_FIELD_SPECTRA[polarization]} spectra along '
                                    'the box outline (little memory); 1-41 typical.',
                                    '1 to 41 is typical; each wavelength needs little memory.')
    features = (
        FeatureType('point_source', 'Point source', 'source', 'point', params=point_source_params,
                    object_kinds=('point',), max_entities=1, description=descriptions['point_source'],
                    summary=summaries['point_source'], apply=apply_point_source, validate=validate_point_source),
        FeatureType('line_source', 'Line source', 'source', 'line', params=_line_source_params(polarization),
                    object_kinds=('line',), max_entities=1, acts_on_hi_face=hi_face,
                    description=descriptions['line_source'], summary=summaries['line_source'],
                    apply=apply_line_source, validate=validate_line),
        FeatureType('gaussian_beam', 'Gaussian beam', 'source', 'line', params=_gaussian_beam_params(polarization),
                    object_kinds=('line',), max_entities=1, acts_on_hi_face=hi_face,
                    description=descriptions['gaussian_beam'], summary=summaries['gaussian_beam'],
                    apply=apply_gaussian_beam, validate=validate_line),
        FeatureType('plane_wave_tfsf', 'Plane wave box (+x)', 'source', 'domain', params=_TFSF_PARAMS[polarization],
                    object_kinds=('rectangle',), max_entities=1,
                    description=descriptions['plane_wave_tfsf'], summary=summaries['plane_wave_tfsf'],
                    apply=apply_tfsf, validate=validate_tfsf),
        FeatureType('mode_source', 'Mode source', 'source', 'line', params=MODE_SOURCE_PARAMS,
                    object_kinds=('line',), max_entities=1, result_kind='table',
                    description=_mode_source_description(polarization), summary=summaries['mode_source'],
                    apply=apply_mode_source, result=mode_source_result_callback,
                    validate=make_validate_mode_source(polarization)),
        FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                    description=descriptions['pec_domain'], summary=summaries['pec_domain'], apply=apply_pec,
                    memory_estimate=pec_memory_estimate(1 if polarization == 'ez' else 2)),
        effective_index_feature(polarization),
        nonlinear_medium_feature(polarization),
        gain_medium_feature(polarization),
        phase_change_medium_feature(polarization),
        FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS[polarization],
                    object_kinds=('point',), max_entities=1, result_kind='timeseries',
                    description='Time series of one field component in the cell containing the point, sampled after '
                                f'every time step (t in s, values in {_PROBE_UNITS[polarization]}); a coordinate '
                                'exactly on a cell edge goes to the upper cell.',
                    summary=summaries['field_probe'],
                    apply=apply_field_probe, result=probe_result_callback, validate=validate_points_inside),
        resonance_monitor_feature(polarization),
        FeatureType('flux_monitor', 'Flux monitor', 'monitor', 'line', params=FLUX_PARAMS,
                    object_kinds=('line',), max_entities=1, result_kind='flux_spectrum',
                    description=descriptions['flux_monitor'], summary=summaries['flux_monitor'],
                    apply=apply_flux_monitor, result=flux_result_callback, validate=validate_flux_monitor,
                    memory_estimate=line_spectra_memory_estimate),
        FeatureType('mode_monitor', 'Mode monitor', 'monitor', 'line', params=MODE_MONITOR_PARAMS,
                    object_kinds=('line',), max_entities=1, result_kind='xy',
                    description=_mode_monitor_description(polarization), summary=summaries['mode_monitor'],
                    apply=apply_mode_monitor, result=mode_monitor_result_callback,
                    validate=make_validate_mode_monitor(polarization), memory_estimate=line_spectra_memory_estimate),
        FeatureType('dft_monitor', 'Field map monitor', 'monitor', 'domain', params=_DFT_PARAMS[polarization],
                    object_kinds=('rectangle',), max_entities=1, result_kind='dft_fields',
                    description=descriptions['dft_monitor'], summary=summaries['dft_monitor'],
                    apply=apply_dft_monitor, result=dft_result_callback,
                    validate=make_validate_dft_monitor(polarization),
                    memory_estimate=dft_monitor_memory_estimate(polarization)),
        FeatureType('far_field', 'Far-field pattern', 'monitor', 'domain', params=nf2ff_params,
                    object_kinds=('rectangle',), max_entities=1, result_kind='far_field',
                    description=descriptions['far_field'], summary=summaries['far_field'], apply=apply_far_field,
                    result=make_far_field_result_callback(nf2ff_params), validate=validate_far_field,
                    memory_estimate=far_field_memory_estimate),
    )
    return PhysicsType(
        name=f'em_fdtd_{polarization}',
        label=_PHYSICS_LABELS[polarization],
        family=EM_FAMILY,
        group='em',
        default_instance_name='em',
        description=f'{_PHYSICS_DESCRIPTIONS[polarization]} {HARMONIC_FDTD_TEXT} {_STEPPER_TEXT}',
        summary=_PHYSICS_SUMMARIES[polarization],
        settings=_settings(polarization),
        features=features,
        material_properties=em_material_params(polarization, _MATERIAL_HELPS[polarization]),
        material_schema=EM_MATERIAL_SCHEMAS[polarization],
        state_fields=tuple(renderer_class.field_infos[name] for name in state_field_names) + extra_state_fields,
        output_fields=tuple(renderer_class.field_infos.values()),
        source_slots=source_slot_docs(slot_arrays, f'{polarization}_source'),
        supported_studies=('time_dependent', FREQUENCY_DOMAIN_STUDY),
        default_study={'type': 'time_dependent', 'settings': {'duration': 200e-15}},
        uses_dispersion=True,
        # EMPhysics with dispersive poles (background permittivity written into the slot arrays), absorption tracking
        create=partial(create_em_physics, polarization=polarization, background_targets=tuple(slot_arrays.values())),
        domain_requirements=fdtd_domain_requirements,
        steps_in_time=steps_in_time,
        shortest_wavelength=make_source_shortest_wavelength({feature.name: feature for feature in features}),
        validate=with_bloch_validation(make_validate_harmonic_fdtd(make_validate_physics(slot_arrays))),
        memory_estimate=fdtd_memory_estimate(polarization),
        time_estimate=partial(fdtd_time_step_estimate, polarization),
        solver_roles={'stepper': STEPPER_ROLE},
        backends=('gpu', 'cpu'),   # core.kernels twins: numpy (numba when installed) on the cpu backend
        subpixel_smoothing=True,
        conformal_boundaries=True,
    )


# a 2D field / CPML launch costs ~3 profile launches of host overhead (as em3d _FIELD_LAUNCH_WEIGHT: the CPML side
# launch also slices ~5 views per call); the flat 8 + 2 per feature priced 0.087 ms vs 0.33-0.41 ms per gpu step
_FIELD_LAUNCH_WEIGHT = 3
# boundaries.BoundaryManager outer-face rows, each one launch, applied twice per step (in the post and again after
# the sources): ez H rows on pmc faces, E_z rows on pec / pmc / mirror faces; hz H_z rows on pec / pmc / mirror
# faces, E rows on pec / antisymmetric faces
_WALL_ROW_FACES = {'ez': (('pmc',), ('pec', 'pmc', 'symmetric', 'antisymmetric')),
                   'hz': (('pec', 'pmc', 'symmetric', 'antisymmetric'), ('pec', 'antisymmetric'))}
_PEC_KEEP_LAUNCHES = {'ez': 1, 'hz': 2}   # per field set: ez[pec_box] *= keep / ex, ey on their keeps (advance)
_POLE_LAUNCHES = {'ez': 2, 'hz': 4}       # per field set and pole: J / P update + E correction (hz: x and y faces)


def fdtd_time_step_estimate(polarization, settings_values, mesh, context):
    """PhysicsType.time_estimate of em_fdtd_<polarization> (polarization bound by partial). Explicit 2D Yee engine (ez
    / hz): per Courant step (mesh dt) its arrays and its features' (DFT monitors, poles, CPML) streamed once; per field set (Bloch k != 0: two) the H / E kernels plus an H and an E launch per CPML side
    (x their host overhead weight), the outer-face rows, the PEC keep, conformal faces and pole kernels; two per
    enabled feature (sources, probes, monitors); cpu: numba Yee twins, numpy for the rest."""
    features = [feature for feature in context.physics_spec.features if feature.enabled]
    faces = [settings_values.get(f'bc_{face}') for face in FACES]
    cpml_sides = faces.count('cpml')
    n_sets = 2 if bloch_active_axes(settings_values) else 1
    h_rows, e_rows = _WALL_ROW_FACES[polarization]
    wall_rows = 2 * sum(1 for face in faces if face in h_rows) + 2 * sum(1 for face in faces if face in e_rows)
    pec = any(feature.type == 'pec_domain' for feature in features)
    conformal = pec and context.spec.solver_defaults.conformal_boundaries is True
    memory_items = context.memory.items if context.memory is not None else ()
    poles = any(label.endswith('dispersive poles') and size > 0 for label, size in memory_items)
    per_set = (wall_rows + (_PEC_KEEP_LAUNCHES[polarization] if pec else 0) + (1 if conformal else 0)
               + (_POLE_LAUNCHES[polarization] if poles else 0))
    numba_launches = _FIELD_LAUNCH_WEIGHT * n_sets * 2
    launches = numba_launches + n_sets * (_FIELD_LAUNCH_WEIGHT * 2 * cpml_sides + per_set) + 2 * len(features)
    # cpu: only the H / E Yee kernels have numba twins (fdtd_*_2d_kernels); CPML psi, poles, gain, loss and monitors
    # run numpy twins
    return explicit_step_estimate(context, 'Yee', mesh.dt, launches=launches, numba_items=_numba_stepped,
                                  numba_launches=numba_launches)


def _numba_stepped(item_label):
    """Memory items (fdtd_memory_estimate labels) the numba Yee twins stream: fields and update coefficients (the
    Bloch imaginary set counted with them: its bulk is the fields)."""
    return item_label.startswith(('fields ', 'update coefficients', 'Bloch imaginary field set'))


EM_FDTD_EZ = _fdtd_physics_type('ez')
EM_FDTD_HZ = _fdtd_physics_type('hz')

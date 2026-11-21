"""Model definition of 'em_fdtd_bor' (body-of-revolution FDTD: (r, z) half plane, azimuthal order m,
time stepped) on definition_common / definition_bor, with its self-tests; definition.register calls register."""
import math
from dataclasses import replace

from ...core.time_estimate import explicit_step_estimate
from ...model.definition_helpers import (AXISYMMETRIC, AXISYMMETRIC_SPACE_SETTINGS, GEOMETRY_SETTING,
                                         axisymmetric_issues, parkable_material_arrays, validate_points_inside,
                                         wavelength_range_issues)
from ...model.definition_resonance import resonance_monitor_feature as shared_resonance_feature
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path
from .body_of_revolution import BOR_FIELD_COMPONENTS, BOR_SOURCE_COMPONENTS
from .definition_bor import (_COMPONENT_NAMES, _MODE_MONITOR_PARAMS, _PORT_POWER_PARAMS, COMPONENT_HELPS,
                             GEOMETRY_HELP, _port_line, _validate_far_field, _validate_line_monitor, axis_wall_settings,
                             azimuthal_order_list_issues, cylindrical_material_params)
from .definition_common import (BOX_CELLS, EM_FAMILY, FLUX_PARAMS, FREQUENCY_DOMAIN_STUDY, PER_CW_TEXT,
                                POINT_SOURCE_SIGN, SPAN_CELLS, _monitor_wavelength_kwargs, _window_axes,
                                apply_component_point_source, apply_far_field, apply_field_probe, apply_flux_monitor,
                                apply_pec, domain_requirements, far_field_params, make_far_field_result_callback,
                                make_source_shortest_wavelength, make_validate_physics, object_result,
                                other_em_physics_labels, probe_result_callback, source_slot_docs, steps_in_time,
                                state_arrays_read, waveform_params, write_dispersive_materials)
from .definition_feature_validation import _line_cells, _selected_shapes
from .definition_harmonic_fdtd import HARMONIC_FDTD_TEXT, make_validate_harmonic_fdtd
from .definition_resonance import RESONANCE_SUMMARY, em_resonance_probe
from .fdtd_bor_ports import fdtd_bor_mode_source_result
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import (_fdtd_bor_members, fdtd_bor_far_field_memory_estimate, fdtd_bor_flux_memory_estimate,
                               fdtd_bor_memory_estimate, fdtd_bor_mode_memory_estimate, pec_memory_estimate)
from .renderer_fdtd_bor import BORFDTDRenderer
from .solvers.fdfd_bor import BOR_MATERIAL_SCHEMA_KEY, BOR_PERMITTIVITY_ARRAYS, BOR_PERMITTIVITY_SLOTS
from .solvers.fdtd_bor import (H_COMPONENTS, PARKABLE_ARRAYS, BORFDTDPhysics, fdtd_bor_far_field_result,
                               fdtd_bor_flux_result, fdtd_bor_mode_result)

_TYPE_NAME = 'em_fdtd_bor'
_FIELDS_NOTE = ('Fields are the real time-domain amplitudes of the standing wave of order m: E_r, E_z, H_phi ~ '
                'cos(m phi), E_phi, H_r, H_z ~ sin(m phi) (m = 0: the TM and TE fields themselves). DFT phasors / '
                'source spectrum = the em_fdfd_bor travelling-wave phasors (E_phi, H_r, H_z times j) at omega_tilde = '
                '(2 / dt) sin(omega dt / 2), exactly: an em_fdfd_bor resonance omega rings at (2 / dt) asin(omega dt '
                '/ 2).')

# ---- settings ----
_BC_HELP = (
    'x_lo is always the axis r = 0. Walls on the outer grid edge as in em_fdfd_bor (pec = tangential E 0, pmc = '
    'tangential H 0); cpml adds pml_cells cells ending in a pmc edge: on the z faces a convolutional PML (2D FDTD '
    'grading), on r_hi a cylindrical PML (the em_fdfd_bor radius stretch, alpha = 0, recursive convolutions) that '
    'absorbs as well near the axis and at large m; pml_alpha_max acts on the z faces only. A pec / pmc z_lo face is '
    'the mirror plane of a structure symmetric about z = 0 (pmc keeps E_r, E_phi even, pec E_z). No periodic faces.')
_SETTINGS = (
    Param(GEOMETRY_SETTING, 'choice', AXISYMMETRIC, label='Geometry', choices=(AXISYMMETRIC,),
          choice_labels=((AXISYMMETRIC, 'Axisymmetric (r, z)'),),
          help=GEOMETRY_HELP,
          tip='The drawing is half of a cut through a round object, rotated about the left edge of the domain.'),
    *axis_wall_settings(_BC_HELP),
    Param('azimuthal_order', 'int', 0, label='Azimuthal order', min=0, max=10000,
          help='m of the fields (+m and -m are degenerate). m = 0: on-axis dipoles, radial modes (TE and TM '
               'decouple); m = 1: on-axis focused / plane waves; larger m: whispering-gallery modes, m ~ 2 pi R n_eff '
               '/ lambda. dt ~ dr / (C0 (|m| + 1)) (the m / r terms near the axis), so large m costs steps '
               '(em_fdfd_bor eigenfrequency is cheaper for whispering-gallery resonances). Off-axis point source: '
               'azimuthal_orders.',
          tip='Number of field oscillations around the axis: 0 for fields that are the same all around, larger values '
              'for modes running around a disk or ring edge (they need more time steps).'),
    Param('azimuthal_orders', 'float_list', None, label='Azimuthal order list', unit='1', min=0.0, max=10000.0,
          optional=True,
          help='Distinct integer orders m >= 0 of a driven order sum (as em_fdfd_bor); a non-empty list replaces '
               'azimuthal_order, null or [] = azimuthal_order alone. Every point_source becomes a 3D point current at '
               'phi = 0 (see point_source) expanded in exp(-j m phi); each m > 0 stands for +m and -m (a cos(m phi) '
               'standing wave), one stepper per order and source kind (r / z currents, phi currents; m = 0 one), all '
               'in lockstep at courant_factor x the limit of the largest order, summed. Fields, field_probe / '
               'resonance_monitor traces and frequency_domain phasors are the phi = 0 half plane of the 3D field '
               '(physical phasors, without the single-order j on E_phi, H_r, H_z; their ring-weighted integrals are '
               'not 3D integrals); flux_monitor and far_field add the orders exactly and report each. List every '
               'order 0..M: powers and far field converge once M exceeds k rho_s by a few (rho_s = source radius, k = '
               'n omega / c). Cost: up to 2 M + 1 steppers at the time step of order M. Sources must exist at the '
               'build (no later scripted source of a new kind); mode_source / mode_monitor need a single order.',
          tip='Several azimuthal orders summed in one run, e.g. 0, 1, 2, ..., 8.\n\n'
              'A point source then becomes a true point source beside the axis (not a ring); the field is summed over '
              'all listed orders, so list every order from 0 up to a few more than 2 pi x source radius / wavelength. '
              'Each order adds time-stepping work.'),
    Param('courant_factor', 'float', 0.99, label='Time step factor', unit='1', min=0.0, min_exclusive=True, max=1.0,
          help='Fraction of the exact leap-frog limit 2 / omega_max of this grid and order (vacuum operator, host '
               'eigen-solve at the build), not the 2D CFL rule; stable for eps >= 1, mu_r >= 1 (the build refuses a '
               'field point below 1). Poles need eps_inf - P >= courant_factor^2 at every field point, P = (omega_p '
               'dt)^2 / 4 per Drude and delta_eps (omega_0 dt)^2 / (4 - (omega_0 dt)^2) per Lorentz pole (else the '
               'build stops: refine the mesh or lower this).',
          tip='Time step as a fraction of the largest stable time step. Keep 0.99.'),
)

# ---- material vocabulary ('modes' table, drawing-plane names; real values, Drude / Lorentz poles) ----
_POINT_VALUE = ('E_phi sees the cell value, E_r / E_z (cell edges) the ring-volume / length weighted mean of their '
                'non-PEC cells (no subpixel smoothing).')
_REAL_ONLY = ('Real values only; below 1 at a field point stops the build. Drude / Lorentz poles are stepped on all '
              'three components (2D FDTD velocity form; an E_r / E_z point sees the pole times the share of its cells '
              'carrying it), this value or the first Drude eps_inf the background; pole time-step bound: '
              'courant_factor.')
_MATERIAL_PROPERTIES = cylindrical_material_params({
    'eps_r': ('Isotropic, all three components; n^2 of a lossless dielectric (silicon ~12.1, silica ~2.1 at 1.55 um); '
              'field-dependent values rewrite the coefficients at each change; a component property wins over it. '
              f'{_COMPONENT_NAMES} {_POINT_VALUE} {_REAL_ONLY}'),
    **COMPONENT_HELPS,
    'sigma_e': ('J = sigma_e E on every component (semi-implicit; E_r / E_z take the mean of their cells as eps); weak '
                'absorber of index n, extinction kappa: sigma_e = 2 n kappa omega eps0.'),
    'mu_r': 'Real, 1 for optical materials; H_r, H_phi, H_z take the mean of the cells around their points.',
    'sigma_m': 'Loss on H (semi-implicit; artificial absorbers only).',
})

# ---- features ----
_POINT_SOURCE_PARAMS = waveform_params() + (
    Param('component', 'choice', 'z', label='Direction', choices=BOR_SOURCE_COMPONENTS,
          choice_labels=(('r', 'Radial'), ('phi', 'Azimuthal (around the axis)'), ('z', 'Axial (vertical)')),
          help="Normalised as the em_fdfd_bor point source ('phi': J_phi = I / (w h); 'z' / 'r': current I through "
               "the cell's ring cross-section; axis cell 'z': on-axis dipole of moment I h). For m != 0, I is the "
               "amplitude of the standing-wave current, cos(m phi) (z, r) / sin(m phi) (phi); a 'phi' current is "
               'the em_fdfd_bor one at phase +90 deg (same power). In a driven order sum (azimuthal_orders) a 3D point '
               "current at phi = 0 of moment I d as in em_fdfd_bor (free space: radiates eta0 k^2 (I d)^2 / (12 pi)).",
          tip='Direction of the current: around the axis (a loop), along the axis or radially.\n\n'
              'Along the axis in the first column of cells is a dipole antenna on the axis.'),
    Param('current', 'float', 1e-3, label='Current', unit='A',
          help=f'Peak current I times the waveform; fields are linear in it. {POINT_SOURCE_SIGN}',
          tip='Peak current of the source. Fields scale linearly with it.'),
)
_PROBE_COMPONENT = Param(
    'component', 'choice', 'ez', label='Field component', choices=BOR_FIELD_COMPONENTS,
    choice_labels=(('er', 'E_r'), ('ephi', 'E_phi'), ('ez', 'E_z'), ('hr', 'H_r'), ('hphi', 'H_phi'), ('hz', 'H_z')),
    help='Raw sample every step on its Yee point of the containing cell: E_phi at the centre, E_r / H_z on the +r edge, '
         'E_z / H_r on the +z edge, H_phi on the +r +z corner; H half a step before the recorded time.',
    tip='Field recorded at the point; electric fields in V/m, magnetic fields in A/m.')
_RESONANCE_COMPONENT = replace(_PROBE_COMPONENT, help=f'Recorded field: {_PROBE_COMPONENT.help}')
_FAR_FIELD_PARAMS = far_field_params('Each wavelength accumulates complex128 spectra of the E / H rows and columns '
                                     'around the box faces (little memory); 1-41 typical.',
                                     '1 to 41 is typical; each wavelength needs little memory.')
_FLUX_DESCRIPTION = (
    'Spectral power through an axis-aligned line with the em_fdfd_bor flux_monitor pairing (E of the line cells, H on '
    'their hi face; exactly conserved through lossless cells): horizontal (z = const) along +z through the swept '
    'annulus, vertical (r = const) along +r through the cylinder, ring-weighted for the travelling exp(-j m phi) wave. '
    "Result 'flux_spectrum' in monitor order: power = pi sum Re(E H*) of the dt-weighted DFT spectra ('a.u.'); "
    'power_per_cw = power / |S(omega)|^2 = the power [W] of the cw source of the same current (em_fdfd_bor at '
    f'omega_tilde). {PER_CW_TEXT} Driven order sum: totals of the 3D field (orders add exactly), extras power_orders / '
    'power_orders_per_cw (n_wavelengths x n_orders, each m with -m) and azimuthal_orders. The line covers '
    f'{SPAN_CELLS}; its hi face must be inside the grid.')
_FAR_FIELD_DESCRIPTION = (
    'Near-to-far-field transform of the DFT phasors on a rectangle as the em_fdfd_bor far_field (same surfaces and z_lo '
    "mirror images; vacuum outside the box). Result 'far_field' in monitor order: power r^2 |E|^2 / (2 eta0) [W/sr] of "
    'the cw source of the same current when every source shares one waveform (NaN outside its spectrum), else of the '
    "raw dt-weighted spectra ('a.u.'), versus meridional angle counter-clockwise from +r; extras power_theta / "
    'power_phi, angle_axes, azimuthal_order. Driven order sum: the pattern of the summed 3D field in the plane through '
    'the sources (cos >= 0 at phi = 0, the others at phi = pi); extras power_azimuthal_mean (phi average), '
    'power_orders (n_wavelengths x n_orders x n_angles, each m with -m), azimuthal_orders instead of azimuthal_order. '
    f'Needs one cell beyond x1 / y1 and below y0 inside the domain; the box covers {BOX_CELLS}.')


# ---- mode ports (fdtd_bor_ports) ----
_PORT_WAVEFORMS = ('gaussian', 'cw')
_MODE_WAVEFORM_PARAMS = tuple(
    replace(param, choices=_PORT_WAVEFORMS,
            choice_labels=tuple(label for label in param.choice_labels if label[0] in _PORT_WAVEFORMS),
            help=f'{param.help} mode_source: gaussian or cw only (the launch drives the imaginary part of the profile '
                 'with the quadrature partner of the waveform; keep a gaussian bandwidth <= 0.6).')
    if param.name == 'waveform' else param for param in waveform_params())
_MODE_SOURCE_PARAMS = _MODE_WAVEFORM_PARAMS + _PORT_POWER_PARAMS
_MODE_MONITOR_PARAMS_TD = FLUX_PARAMS + _MODE_MONITOR_PARAMS
_PORT_TEXT = ('Port line as the em_fdfd_bor ports (horizontal, from the axis or annular, metal ends); the line row and '
              'its neighbour rows must lie outside the z cpml. Modes ~ exp(-j beta z): full-vector solutions of the '
              'stepped BOR operator at omega_tilde = (2 / dt) sin(omega dt / 2) (stepped poles and conductivities), '
              'solved once on the build materials (later material changes not followed); exact for a guide '
              'invariant along z over a cell on each side of the line. Ranked by descending Re n_eff (mode 1 '
              'highest), 1 W each in the travelling exp(-j m phi) wave. Not in a driven order sum.')
_MODE_SOURCE_DESCRIPTION = (
    'Launches one guided mode of a guide along the axis (fibre, pillar, metal pipe, coax) from a horizontal line in '
    'one direction: a one-sided TFSF interface between the line row and the row upstream, Re and Im of the '
    'carrier-frequency profile driven by the waveform and its quadrature (one-frequency mode, as industry transient '
    'ports). Exact and one-sided at the carrier; elsewhere in a pulse band the power differs somewhat and a little '
    'leaks backward (more with n_eff change over the band, near cutoff): for spectra divide by the forward power of a '
    f"mode_monitor between source and device. {_PORT_TEXT} Result 'table' of the carrier solve, one row per mode "
    'found (up to max(mode_number + 2, 4)): mode_number, n_eff, n_eff_imag, loss_db_per_cm, n_group, edge_field, '
    'selected; extras wavelength (carrier) [m], power [W], direction, mode_number, phase [deg].')
_MODE_MONITOR_DESCRIPTION = (
    'Splits the field on a horizontal line into the upward (positive) and downward (negative) amplitudes of one guided '
    'mode per monitor wavelength (DFT spectra of the line row and the row above, discrete reciprocity: other modes '
    'drop out); modes solved per wavelength (mode_number at the centre one, followed outward by overlap). '
    f"{_PORT_TEXT} Result 'xy' versus vacuum wavelength: power_positive / power_negative = |a|^2 of the dt-weighted DFT "
    'spectra (a.u.); with one shared source waveform power_per_cw_* = |a / S(omega)|^2 [W]; with exactly one enabled '
    'mode_source fraction_positive / fraction_negative = |a / (S sqrt(P) e^{j phase})|^2 (transmission / reflection '
    'into this mode) and phase_positive / phase_negative (rad, not de-embedded); per_cw rules as flux_monitor. Extras '
    'frequency, wavelength, n_eff, n_eff_imag, mode_number, normalized. Keep the line and the row above off the rows '
    'of a mode_source.')


def _with_wavelength_range(validate):
    """validate plus wavelength_range_issues of the feature's monitor spectrum (the em_fdfd_bor validators reused here
    have no spectrum to check)."""
    def checked(vctx, physics_spec, feature_spec, values):
        issues = validate(vctx, physics_spec, feature_spec, values)
        if 'wavelength_start' in values:
            issues = issues + wavelength_range_issues(values, feature_path(physics_spec, feature_spec))
        return issues
    return checked


def _validate_fdtd_port(source):
    """_validate_line_monitor(port=True) plus the fdtd_bor_ports `_check_rows` rule: the line row and its neighbour
    rows (row - 1 .. row + 1 for a source, row and row + 1 for a monitor) inside the window rows (off the z cpml)."""
    shared = _validate_line_monitor(port=True)

    def validate(vctx, physics_spec, feature_spec, values):
        issues = shared(vctx, physics_spec, feature_spec, values)
        if vctx.mesh is None or vctx.domain_size is None:
            return issues
        _, (z_widths, _, first, last) = _window_axes(vctx.mesh)
        for shape in _selected_shapes(vctx, feature_spec, 'line'):
            cells = _line_cells(vctx.mesh, shape.values, vctx.domain_size)
            if cells is None or cells[0] != 'y':
                continue   # reported by the shared rules
            row = cells[1]
            rows = (row - 1, row, row + 1) if source else (row, row + 1)
            # rows beyond the grid edge are the shared rule's message; these reach into a z cpml layer
            outside = [index for index in rows if 0 <= index < z_widths.size and not first <= index <= last]
            if outside:
                issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                    f"line '{shape.name}': its row and the neighbour rows the port reads reach into "
                                    'the z cpml layer: move it at least one cell further inside the window'))
        return issues
    return validate


def _apply_mode_source(ctx, solver, feature_spec, values, selection):
    position, span = _port_line(ctx, selection, feature_spec)
    return solver.add_mode_source(position, span, power=values['power'], mode_number=values['mode_number'],
                                  direction=values['direction'], phase=values['phase'],
                                  wavelength=values['wavelength'], bandwidth=values['bandwidth'],
                                  waveform=values['waveform'])


def _apply_mode_monitor(ctx, solver, feature_spec, values, selection):
    position, span = _port_line(ctx, selection, feature_spec)
    return solver.add_mode_monitor(position, span, mode_number=values['mode_number'],
                                   **_monitor_wavelength_kwargs(values))


_FEATURES = (
    FeatureType('point_source', 'Point source', 'source', 'point', params=_POINT_SOURCE_PARAMS,
                object_kinds=('point',), max_entities=1,
                description='Current element in the cell containing the point (x = r, y = z): a ring in 3D with the '
                            'azimuthal dependence of the order, driven by the waveform; see component and current.',
                summary='A small current at a point of the half plane, a ring around the axis in 3D; on the axis an '
                        'antenna.',
                apply=apply_component_point_source, validate=validate_points_inside),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=(_PROBE_COMPONENT,), object_kinds=('point',),
                max_entities=1, result_kind='timeseries',
                description='Time series of one component on its Yee point of the containing cell, sampled after '
                            'every step (t in s, V/m or A/m); a coordinate exactly on a cell edge goes to the upper '
                            'cell.',
                summary='Records one field component at a point after every time step, as a time trace.',
                apply=apply_field_probe, result=probe_result_callback, validate=validate_points_inside),
    shared_resonance_feature(em_resonance_probe(_RESONANCE_COMPONENT, H_COMPONENTS, apply_field_probe),
                             RESONANCE_SUMMARY),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Rings in 3D: E_phi = 0 in cells whose centre lies in the domains, E_r / E_z = 0 on '
                            'edges between two such cells (staircased; fixed at the build).',
                summary='Turns the selected domains into a perfect metal, following the mesh cells.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0)),
    FeatureType('flux_monitor', 'Flux monitor', 'monitor', 'line', params=FLUX_PARAMS, object_kinds=('line',),
                max_entities=1, result_kind='flux_spectrum', description=_FLUX_DESCRIPTION,
                summary='Records the power crossing a horizontal disc or annulus (upward positive) or a vertical '
                        'cylinder (outward positive) at each wavelength.',
                apply=apply_flux_monitor, result=object_result(fdtd_bor_flux_result),
                validate=_with_wavelength_range(_validate_line_monitor(port=False)),
                memory_estimate=fdtd_bor_flux_memory_estimate),
    FeatureType('far_field', 'Far-field pattern', 'monitor', 'domain', params=_FAR_FIELD_PARAMS,
                object_kinds=('rectangle',), max_entities=1, result_kind='far_field',
                description=_FAR_FIELD_DESCRIPTION,
                summary='Computes the power radiated in each direction of the drawing plane per wavelength, from the '
                        'fields on a rectangle around every object (a full 3D pattern). Keep the rectangle in vacuum.',
                apply=apply_far_field,
                result=make_far_field_result_callback(_FAR_FIELD_PARAMS, fdtd_bor_far_field_result),
                validate=_with_wavelength_range(_validate_far_field),
                memory_estimate=fdtd_bor_far_field_memory_estimate),
    FeatureType('mode_source', 'Mode source', 'source', 'line', params=_MODE_SOURCE_PARAMS, object_kinds=('line',),
                max_entities=1, result_kind='table', description=_MODE_SOURCE_DESCRIPTION,
                summary='Launches one guided mode of a waveguide running along the axis, up or down, with the pulse '
                        'or continuous wave. Pair it with mode monitors for transmission and reflection spectra.',
                apply=_apply_mode_source, result=object_result(fdtd_bor_mode_source_result),
                validate=_validate_fdtd_port(source=True)),
    FeatureType('mode_monitor', 'Mode monitor', 'monitor', 'line', params=_MODE_MONITOR_PARAMS_TD,
                object_kinds=('line',), max_entities=1, result_kind='xy', description=_MODE_MONITOR_DESCRIPTION,
                summary='Measures the power in one guided mode crossing a horizontal line, up and down, at each '
                        'wavelength. With a single mode source it also gives transmission and reflection.',
                apply=_apply_mode_monitor, result=object_result(fdtd_bor_mode_result, with_solver=True),
                validate=_with_wavelength_range(_validate_fdtd_port(source=False)),
                memory_estimate=fdtd_bor_mode_memory_estimate),
)

# ---- type ----
_DESCRIPTION = (
    'Body-of-revolution (BOR) FDTD: the (r, z) half plane of a structure rotationally symmetric about the x_lo window '
    'edge (x = r >= 0, y = z), one azimuthal order m stepped in time on the em_fdfd_bor finite-integration operator '
    f'(same Yee layout, ring metric and axis rules). {_FIELDS_NOTE} Leap-frog at the exact time-step limit of the '
    'order (courant_factor); CPML faces, pec / pmc walls; real eps components (em_fdfd_bor vocabulary), mu_r, sigma_e, '
    'sigma_m and Drude / Lorentz poles (velocity form on all three components; DFT phasors then solve em_fdfd_bor at '
    'omega_tilde with the pole at omega_tilde and damping gamma cos(omega dt / 2)). An azimuthal_orders list: the '
    'driven order sum of off-axis point sources (see that setting). Outputs at cell centres: components instantaneous '
    '(H half a step earlier); derived values weight by the ring volume 2 pi r dr dz. GPU or CPU backend (operators '
    'built on the host, sparse products per step). Cannot share a model with another EM type. heat_em_thermo_optic '
    f'writes the three permittivity arrays. {HARMONIC_FDTD_TEXT} Its phasor fields are <component>_re / '
    "<component>_abs and e_abs_harmonic (em_fdfd_bor convention). Instance usually 'em'.")


def _create(ctx, physics_spec, settings_values):
    """BORFDTDPhysics of the domain's boundary configuration (an azimuthal_orders list: the driven order sum) with its
    dispersive poles and their background permittivity written into the three permittivity arrays; finalize builds
    the engine; the material arrays parked on the host (_parked_material_arrays) leave the device at once."""
    solver = BORFDTDPhysics(ctx.domain, boundary_config=ctx.domain.boundary_config,
                            azimuthal_order=settings_values['azimuthal_order'],
                            courant_factor=settings_values['courant_factor'],
                            azimuthal_orders=[int(order) for order in settings_values.get('azimuthal_orders') or ()],
                            park_arrays=_parked_material_arrays(ctx.spec, ctx.registry, physics_spec, ctx.parameters))
    write_dispersive_materials(ctx, physics_spec, solver, BOR_PERMITTIVITY_ARRAYS)
    return solver


def _parked_material_arrays(spec, registry, physics_spec, parameters) -> tuple:
    """PhysicsType.parked_material_arrays: the BORFDTDPhysics PARKABLE_ARRAYS (every material array: the engine reads
    host copies only) under the shared rule (definition_helpers.parkable_material_arrays: gpu, no coupling into it,
    static materials), minus an array whose state field a material expression or field_transfer reads."""
    try:
        physics_type = registry.physics_type(physics_spec.type)
        own_properties = {param.name for param in physics_type.material_properties}
        arrays = set(PARKABLE_ARRAYS) - state_arrays_read(spec, registry, physics_spec.name, PARKABLE_ARRAYS)
    except (KeyError, ValueError, TypeError):   # reported by validate (ExpressionError is a ValueError)
        return ()
    return parkable_material_arrays(spec, registry, physics_spec, parameters, own_properties, tuple(sorted(arrays)))


def _domain_requirements(settings_values):
    """The FDTD boundary configuration and Courant factor with the axis face as a plain pmc edge (no PML padding)."""
    return domain_requirements({**settings_values, 'bc_x_lo': 'pmc'})


def _other_em_issues(vctx, path):
    """Error when another EM type shares the model (its drawing is no (r, z) half plane of this order)."""
    others = other_em_physics_labels(vctx, _TYPE_NAME)
    if not others:
        return []
    return [Issue('error', path, f"em_fdtd_bor steps one azimuthal order of the (r, z) half plane, while "
                                 f"{', '.join(others)} solve the drawing otherwise; they cannot share one model")]


def _source_issues(vctx, physics_spec, path):
    """Warning in a time_dependent study without an enabled point_source or mode_source (every field stays zero)."""
    if vctx.study_type != 'time_dependent' or any(feature.enabled and feature.type in ('point_source', 'mode_source')
                                                   for feature in physics_spec.features):
        return []
    return [Issue('warning', path, f"'{physics_spec.name}' has no enabled point_source or mode_source: every field "
                                   'stays zero')]


def _order_list_issues(settings_values, physics_spec, path):
    """azimuthal_orders: integers, distinct (errors), no mode ports (error); every order 0..max (warning: a point
    source's sum would miss terms)."""
    orders = settings_values.get('azimuthal_orders') or []
    if not orders:
        return []
    ports = [feature.name for feature in physics_spec.features
             if feature.enabled and feature.type in ('mode_source', 'mode_monitor')]
    if ports:
        return [Issue('error', path, f"mode ports ({', '.join(ports)}) need a single azimuthal order: remove "
                                     'azimuthal_orders or the ports')]
    return azimuthal_order_list_issues(orders, path)[0]


_SHARED_VALIDATE = make_validate_harmonic_fdtd(make_validate_physics(BOR_PERMITTIVITY_SLOTS))


def _validate_physics(vctx, physics_spec, settings_values):
    """The shared FDTD EM checks (materials, slot writers, resolution, duration, monitor band; harmonic rules in a
    frequency_domain study), other EM types, sources, the order list and the (r, z) partner rule."""
    path = ('physics', physics_spec.name)
    issues = list(_SHARED_VALIDATE(vctx, physics_spec, settings_values))
    issues.extend(_other_em_issues(vctx, path))
    issues.extend(_source_issues(vctx, physics_spec, path))
    issues.extend(_order_list_issues(settings_values, physics_spec, path))
    issues.extend(axisymmetric_issues(vctx, physics_spec, path, _TYPE_NAME))
    return issues


# run-time estimate: the BOR time step ~ the mesh's 2D Courant step x this / (|m| + 1) for |m| >= 1 (the m / r terms
# near the axis; the run takes it from the operator's largest eigenvalue: calibrate from summary ticks x dt / duration)
BOR_TIME_STEP_RATIO_GUESS = math.sqrt(2.0)


def _time_estimate(settings_values, mesh, context):
    """Leap-frog steps of every order-sum stepper in lockstep (one fdtd_bor._LockstepBatch): per step the arrays of the
    memory estimate (fields, coefficients, CPML psi, curl CSR products, poles, spectra) streamed once in 2 x (2 + 1 z
    CPML psi + 3 r_hi layer) launches for all steppers together, 2 for the |m| = 1 axis slaving and ~3 per enabled
    feature (a DFT group, a source, a probe); the time step of the largest order is a guess, so are the steps."""
    orders, _ = _fdtd_bor_members(settings_values, context.physics_spec)
    dt = mesh.dt * min(1.0, BOR_TIME_STEP_RATIO_GUESS / (max(orders) + 1)) if mesh.dt else None
    z_cpml = any(settings_values.get(f'bc_{face}') == 'cpml' for face in ('y_lo', 'y_hi'))
    side_launches = 2 + int(z_cpml) + 3 * int(settings_values['bc_x_hi'] == 'cpml')
    axis_launches = 2 * int(any(abs(int(order)) == 1 for order in orders))
    features = sum(1 for feature in context.physics_spec.features if feature.enabled)
    estimate = explicit_step_estimate(context, 'BOR leap-frog', dt,
                                      launches=2 * side_launches + axis_launches + 3 * features,
                                      numba=False)   # cpu: numpy updates and scipy CSR products, no numba twins
    return replace(estimate, items=tuple(replace(item, count_kind='guess') for item in estimate.items))


EM_FDTD_BOR = PhysicsType(
    name=_TYPE_NAME,
    label='Electromagnetic Waves, time domain (axisymmetric, azimuthal order m)',
    family=EM_FAMILY,
    group='em',
    spaces_2d=(AXISYMMETRIC,),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    default_instance_name='em',
    description=_DESCRIPTION,
    summary='Light in round structures (disks, rings, pillars, lenses) simulated step by step in time from a cut '
            'through their axis: full 3D fields of one azimuthal order at the cost of a 2D model, one pulse giving a '
            'whole spectrum or the ring-down of a resonator.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS[BOR_MATERIAL_SCHEMA_KEY],
    uses_dispersion=True,
    state_fields=tuple(BORFDTDRenderer.field_infos[name] for name in ('e_intensity', 'eps_r', 'eps_xx', 'eps_yy',
                                                                       'sigma_e')),
    output_fields=tuple(BORFDTDRenderer.field_infos.values()),
    source_slots=source_slot_docs(BOR_PERMITTIVITY_SLOTS),
    supported_studies=('time_dependent', FREQUENCY_DOMAIN_STUDY),
    default_study={'type': 'time_dependent', 'settings': {'duration': 200e-15}},
    create=_create,
    domain_requirements=_domain_requirements,
    steps_in_time=steps_in_time,
    shortest_wavelength=make_source_shortest_wavelength({feature.name: feature for feature in _FEATURES}),
    validate=_validate_physics,
    memory_estimate=fdtd_bor_memory_estimate,
    time_estimate=_time_estimate,
    parked_material_arrays=_parked_material_arrays,
    backends=('gpu', 'cpu'),
)


def _selftest_cases():
    """The FDTD BOR SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_fdtd_bor import FDTD_BOR_SELFTEST_CASES
    from .selftests_fdtd_bor_media import FDTD_BOR_MEDIA_SELFTEST_CASES
    from .selftests_fdtd_bor_orders import FDTD_BOR_ORDER_SELFTEST_CASES
    return FDTD_BOR_SELFTEST_CASES + FDTD_BOR_ORDER_SELFTEST_CASES + FDTD_BOR_MEDIA_SELFTEST_CASES


def register(registry):
    """Register em_fdtd_bor (after em_fdfd_bor: the material Params of the earlier types stay first per name) and its
    selftest loader."""
    registry.register_physics(EM_FDTD_BOR)
    registry.register_selftests(_selftest_cases)

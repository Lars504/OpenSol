"""Model definition of 'em_fdfd_sphere' (spherically symmetric FDFD: the radial equation of each multipole term on a
spherical (r, theta) or 1D spherical domain, exact outgoing-wave boundary; spherical_multipoles, solvers/fdfd_sphere)
with its self-tests;
definition.register calls register."""
import math
from dataclasses import replace

import numpy as np

from ...core import time_estimate as te
from ...core.memory_estimate import (COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES, MemoryEstimate,
                                     host_counted_lu_bytes)
from ...model.definition_helpers import (AXISYMMETRIC, AXISYMMETRIC_SPACE_SETTINGS, DIRECT_LINEAR_ROLE,
                                         GEOMETRY_SETTING, eigen_role)
from ...model.schema import FeatureType, Issue, Param, PhysicsType, resolve_values
from ...solvers.arpack_shift_invert import APPLICATIONS_PER_BASIS_GUESS, arnoldi_basis_size
from .definition_common import (EIGENFREQUENCY_STUDY, EM_FAMILY, FREQUENCY_DOMAIN_STUDY, DEFAULT_WAVELENGTH,
                                _complex_index, _em_physics_types, _material_issues, _permittivity_writer_issues,
                                all_solved_wavelengths, apply_field_probe, apply_pec, apply_resonance_table,
                                eigen_only_feature_issues, eigen_solved, em_material_params, first_solved_wavelength,
                                host_matrix_back_end_issues, missing_source_issues, object_result,
                                other_em_physics_labels, physics_counter, resonance_eigenpairs,
                                resonance_table_study_issues, scipy_issues, shortest_solved_wavelength, single_object,
                                source_slot_docs, unused_wavelength_issues, validate_driven_point,
                                write_dispersive_materials)
from .materials import EM_MATERIAL_SCHEMAS
from .memory_estimates import pec_memory_estimate
from .renderer_sphere import SphereRenderer
from .solvers.fdfd_sphere import (SPHERE_MATERIAL_SCHEMA_KEY, SPHERE_PERMITTIVITY_ARRAYS, SPHERE_PERMITTIVITY_SLOTS,
                                  SpherePhysics, sphere_dipole_result, sphere_plane_wave_result, sphere_probe_result,
                                  sphere_resonance_table_result)
from .spherical_multipoles import DIPOLE_ORIENTATIONS, SPHERE_FAMILIES, SPHERE_FIELD_COMPONENTS

_TYPE_NAME = 'em_fdfd_sphere'
_DRIVEN_STUDIES = ('stationary', FREQUENCY_DOMAIN_STUDY)
_FAMILY_CHOICES = ('both',) + SPHERE_FAMILIES
_MIN_CELLS_PER_WAVELENGTH = 10.0
_MIN_CELLS_PER_SKIN_DEPTH = 2.0
_FIELDS_NOTE = (
    'Fields vary as exp(+j omega t). Every multipole term (order l >= 1, family er or hr, all its 2 l + 1 degenerate '
    'm) '
    'is one radial solve; the shown fields are sphere-RMS radial profiles: |E_r|, |E_t| (tangential), |H_r|, |H_t| = '
    'the square root of the mean over the sphere of radius r of the component squared, summed over every solved term '
    'and source class (different terms are orthogonal on the sphere, so this is the exact sphere mean of the total '
    'field); e_intensity = |E_r|^2 + |E_t|^2 and absorbed_power are sphere means, so their integral over the whole '
    'shells (the spherical metric volumes, 4 pi r^2 dr per radial cell) is the 3D integral (field energy, absorbed '
    'power in W). On an (r, theta) window the profiles are drawn along every theta (the fields of a spherically '
    'symmetric structure averaged over the sphere do not depend on the angle), so the angle cells only serve the '
    "picture: a few tens are plenty (mesh mode nonuniform with a mesh axis 'y' of segments, e.g. 90 cells).")

# ---- settings ----
_FAMILY_HELP = (
    "Multipole families solved: 'er' = electric multipoles (radial E, no radial H; Mie a_l; the radial function y = r "
    "H_t solves -(y' / eps)' + l (l + 1) y / (eps r^2) = k0^2 mu y), 'hr' = magnetic multipoles (radial H, no radial "
    'E; '
    "Mie b_l; y = r E_t solves -(y' / mu)' + l (l + 1) y / (mu r^2) = k0^2 eps y), 'both' (default) = every term of "
    'both families (a plane wave and tangential dipoles excite both; a radial dipole only er). One radial solve per '
    'family and listed order.')
_ORDER_HELP = (
    'Multipole (angular) order l >= 1 of the fields (each l carries 2 l + 1 degenerate m, all solved at once): l = 1 '
    'dipole, 2 quadrupole, ...; whispering-gallery modes of a sphere of radius a have l ~ 2 pi a n / lambda. For a '
    'plane wave (Mie series) or a dipole source list every order 1..L in multipole_orders instead (L above k a + 4 (k '
    'a)^(1/3) + 2 for a sphere of radius a, k the exterior wave number; a dipole at a distance d from a surface needs '
    'l '
    'up to a few a / d).')
_SETTINGS = (
    Param(GEOMETRY_SETTING, 'choice', AXISYMMETRIC, label='Geometry', choices=(AXISYMMETRIC,),
          choice_labels=((AXISYMMETRIC, 'Spherically symmetric (r, theta)'),),
          help='Always axisymmetric: on a spherical domain (coordinates spherical: the (r, theta) half disc over the '
               'whole 0 .. pi, or a 1D radial line) the drawing rotates about its vertical axis and every radial cell '
               'is a whole spherical shell. The setting exists so that the spherical-domain rule and coupled physics '
               'with a geometry setting (heat, electric_currents, electrostatics) see a consistent spherical partner.',
          tip='The drawing is half of a cut through the sphere; every ring of cells at one radius is a whole '
              'spherical shell.'),
    Param('bc_x_lo', 'choice', 'centre', label='Inner end', choices=('centre', 'pec', 'pmc'),
          choice_labels=(('centre', 'Sphere centre (r = 0)'), ('pec', 'Perfect metal wall'),
                         ('pmc', 'Perfect magnetic wall')),
          help="Inner end of the radial line: 'centre' when radius_min = 0 (the sphere centre, fields regular: y = r "
               "x field = 0; required there and only there), else 'pec' (tangential E = 0 on the sphere r = "
               "radius_min) or 'pmc' (tangential H = 0).",
          tip='The centre of the sphere when the line starts at radius 0, else a metal or magnetic wall at the '
              'inner radius.'),
    Param('bc_x_hi', 'choice', 'radiation', label='Outer end', choices=('radiation', 'pec', 'pmc'),
          choice_labels=(('radiation', 'Open (outgoing waves)'), ('pec', 'Perfect metal wall'),
                         ('pmc', 'Perfect magnetic wall')),
          help="Outer end r = R of the radial line: 'radiation' (default) continues the last cell's medium to "
               'infinity with the exact outgoing spherical wave of each multipole term (Riccati-Hankel xi_l(k r), a '
               'Dirichlet-to-Neumann condition: no PML, no reflection at any l, so the domain may end just outside '
               'the structure; the last cell must not be pec_domain; a plane wave needs it and a lossless last cell); '
               "'pec' / 'pmc' close a spherical metal / magnetic cavity at R.",
          tip='Open: waves leave the outer end without reflection, into the material of the last cell continued to '
              'infinity. Metal or magnetic wall: a closed spherical cavity.'),
    Param('wavelength', 'float', DEFAULT_WAVELENGTH, label='Wavelength', unit='m', min=0.0, min_exclusive=True,
          help='Vacuum wavelength lambda0 of the solve in a stationary study (omega = 2 pi C0 / lambda0); an '
               'eigenfrequency study searches the resonances nearest it (the target; materials are evaluated there); '
               'a frequency_domain study solves its own wavelengths. The radial mesh needs >= 10 (better 20) cells '
               'per lambda0 / |n| in every material and >= 2 per skin depth (validation warns).',
          tip='Vacuum wavelength at which the field is solved, in metres.'),
    Param('multipole_order', 'int', 1, label='Multipole order', min=1, max=100000, help=_ORDER_HELP,
          tip='Angular order of the field: 1 for dipole fields, larger values for whispering-gallery modes.'),
    Param('multipole_orders', 'float_list', None, label='Multipole order list', unit='1', min=1.0, max=100000.0,
          optional=True,
          help='Distinct integer orders l >= 1 solved in one run; a non-empty list replaces multipole_order. Driven '
               'studies sum the terms of every listed order (plane-wave cross sections, dipole powers and the shown '
               'sphere-RMS fields: list 1..L); an eigenfrequency study searches n_resonances resonances per order and '
               "family, order by order (er before hr), resonance labels 'resonance k: l = <l> <family>, lambda = ...' "
               'and resonance_table columns multipole_order, family (a curve of lambda versus multipole_order gives '
               'the free spectral range of whispering-gallery modes). null or [] = multipole_order alone.',
          tip='Several orders solved in one run, e.g. 1, 2, ..., 20 for the scattering of a plane wave.'),
    Param('multipole_family', 'choice', 'both', label='Multipole family', choices=_FAMILY_CHOICES,
          choice_labels=(('both', 'Both'), ('er', 'Electric multipoles (radial E)'),
                         ('hr', 'Magnetic multipoles (radial H)')),
          help=_FAMILY_HELP, tip='Which kind of multipole fields to solve; both is the complete field.'),
)

# ---- materials ('ez' table without eps_zz: isotropic eps(r), mu(r)) ----
_MATERIAL_PROPERTIES = tuple(param for param in em_material_params('ez', {
    'eps_r': ('Isotropic relative permittivity of the shell (>= 1; n^2 of a lossless dielectric); may depend on r (x) '
              'and physics fields. Cell values; the tangential fields see the dual-cell mean, the radial ones the mean '
              'of 1 / eps (normal D_r continuous). The solve uses this value - j sigma_e / (omega eps0) plus every '
              'Drude / Lorentz pole chi(omega) (exp(+j omega t): loss has Im < 0). The last radial cell is the '
              "exterior medium of a 'radiation' end."),
    'sigma_e': ('Ohmic loss in S/m, default 0: eps gains -j sigma_e / (omega eps0). Model metals with a Drude pole (a '
                'plasmonic nanosphere) or pec_domain.'),
    'mu_r': ('Real relative permeability (>= 1), 1 for optical materials; the solve uses mu_r - j sigma_m / (omega '
             'mu0).'),
    'sigma_m': 'Magnetic loss in Ohm/m, default 0 (artificial absorbers only).',
}) if param.name != 'eps_zz')

# ---- features ----
_PLANE_WAVE_PARAMS = (
    Param('amplitude', 'float', 1.0, label='Amplitude', unit='V/m', min=0.0, min_exclusive=True,
          help='Electric field amplitude E0 [V/m] of the incident plane wave (x-polarised, travelling along +z in the '
               'exterior medium); fields scale linearly with it, the cross sections do not depend on it.',
          tip='Strength of the incoming light wave; the cross sections do not depend on it.'),
)
_DIPOLE_PARAMS = (
    Param('orientation', 'choice', 'radial', label='Orientation', choices=DIPOLE_ORIENTATIONS,
          choice_labels=(('radial', 'Radial (pointing away from the centre)'),
                         ('tangential', 'Tangential (parallel to the sphere surface)')),
          help="'radial' excites the electric multipoles (er) only, every l with m = 0; 'tangential' both families "
               'with m = +-1 (the dipole is on the polar axis, parallel to the plane-wave polarisation).',
          tip='Direction of the dipole: pointing away from the centre, or along the sphere surface.'),
    Param('current_moment', 'float', 1e-12, label='Current moment', unit='A m', min=0.0, min_exclusive=True,
          help='Current moment I d [A m] of the point current element (= j omega times its dipole moment p); in an '
               'unbounded medium it radiates eta k^2 (I d)^2 / (12 pi) (eta = eta0 sqrt(mu / eps), k = n k0); powers '
               'scale with its square.',
          tip='Strength of the dipole emitter: current times length, in ampere metres.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Phasor phase of the current moment in degrees; it matters only between several dipoles of one '
               'orientation (they interfere).',
          tip='Phase of the source in degrees; it matters only with several sources.'),
)
_PROBE_PARAMS = (
    Param('component', 'choice', 'et', label='Field component', choices=SPHERE_FIELD_COMPONENTS,
          choice_labels=(('er', '|E_r|'), ('et', '|E_t| (tangential)'), ('hr', '|H_r|'), ('ht', '|H_t| (tangential)')),
          help='Component recorded: its sphere-RMS amplitude in the radial cell containing the point (a point object '
               'at x = r, y = 0), per driven solve; the phase only when the solve had exactly one term and source '
               'class '
               '(one order, one family), else NaN.',
          tip='Field recorded at the radius of the point; electric fields in V/m, magnetic fields in A/m.'),
)
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency study solve, term by term (orders in list order, er before hr), nearest the "
    "target (the wavelength setting) first within each term. Result 'table', one row per resonance: resonance_number, "
    'wavelength = 2 pi C0 / Re(omega) in m, frequency / frequency_imag = Re / Im(omega) / (2 pi) in Hz (> 0 decays), '
    'q_factor = Re(omega) / (2 Im(omega)) (radiation through the exact outgoing condition and material loss; inf when '
    'lossless and closed), pml_fraction (always 0: no PML), multipole_order, family; extras target_wavelength (m), '
    'selected_resonance, multipole_orders and families.')


def _validate_resonance_table(vctx, physics_spec, feature_spec, values):
    """Warning where the study does not eigen-solve the physics."""
    return resonance_table_study_issues(vctx, physics_spec, feature_spec, optomechanics=False)


def _validate_source(vctx, physics_spec, feature_spec, values):
    """Warning where the study eigen-solves the physics (sources play no part there)."""
    return eigen_only_feature_issues(vctx, physics_spec, feature_spec)


def _apply_plane_wave(ctx, solver, feature_spec, values, selection):
    return solver.add_plane_wave(values['amplitude'])


def _apply_dipole_source(ctx, solver, feature_spec, values, selection):
    x, _ = ctx.point(single_object(selection, feature_spec, 'point'))
    return solver.add_dipole_source(x, orientation=values['orientation'], moment=values['current_moment'],
                                    phase=values['phase'])


_FEATURES = (
    FeatureType('plane_wave', 'Plane wave', 'source', None, params=_PLANE_WAVE_PARAMS, result_kind='xy',
                description=(
                    'Incident plane wave of amplitude E0 (x-polarised, along +z) in the exterior medium (the last '
                    "radial cell, continued to infinity; needs bc_x_hi 'radiation' and a lossless last cell): per "
                    'listed order and family its regular multipole wave (both m = +-1) enters through the outgoing '
                    'condition, and the outgoing coefficient gives the Mie coefficient (a_l: er, b_l: hr; the complex '
                    'conjugate of Bohren and Huffman\'s, since fields go as exp(+j omega t)). Result \'xy\' versus '
                    'wavelength: sigma_ext, sigma_sca, sigma_abs [m^2] = sums over the solved terms of 2 pi / k^2 (2 l '
                    '+ 1) Re(a) and |a|^2 (k the exterior wave number), sigma_abs their difference; extras frequency, '
                    'wavelength, multipole_orders, a_real / a_imag and b_real / b_imag (n_wavelengths x n_orders, NaN '
                    'for a family not solved). List every order 1..L (see multipole_order). At most one per physics, '
                    'not together with a dipole_source (validation error).'),
                summary='Light coming in from far away: scattering, absorption and extinction cross sections of the '
                        'sphere (Mie theory on the radial mesh) at each solved wavelength.',
                apply=_apply_plane_wave, result=object_result(sphere_plane_wave_result), validate=_validate_source),
    FeatureType('dipole_source', 'Dipole source', 'source', 'point', params=_DIPOLE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='xy',
                description=(
                    'Point current element at the radius of the point object (x = r, y = 0; on the polar axis in 3D), '
                    'radial or tangential (see orientation); every listed order and family it couples to is solved '
                    'and summed (dipoles of one orientation interfere coherently; radial and tangential ones excite '
                    "orthogonal m and are solved apart). Result 'xy' versus wavelength: power [W] = -1/2 Re(E . (I "
                    'd)*) at the element (radiated plus absorbed; keep its cell lossless), power_homogeneous [W] = eta '
                    'k^2 (I d)^2 / (12 pi) in the medium of its cell, purcell = their ratio (the decay-rate '
                    'enhancement of an emitter there); extras frequency, wavelength, radius, orientation, '
                    'multipole_orders, power_orders (n_wavelengths x n_orders). Not together with a plane_wave.'),
                summary='A small emitter at a given radius: the power it emits and its enhancement over free space '
                        '(Purcell factor) at each solved wavelength.',
                apply=_apply_dipole_source, result=object_result(sphere_dipole_result),
                validate=validate_driven_point),
    FeatureType('field_probe', 'Field probe', 'probe', 'point', params=_PROBE_PARAMS, object_kinds=('point',),
                max_entities=1, result_kind='xy',
                description=("Sphere-RMS amplitude of one field component in the radial cell containing the point at "
                             "every driven solve. Result 'xy' versus wavelength: series abs (unit of the component) "
                             'and '
                             'phase (rad; NaN unless one term and source class was solved); extras frequency, '
                             'wavelength, component, unit.'),
                summary='Records the field at one radius for every solved wavelength.',
                apply=apply_field_probe, result=object_result(sphere_probe_result), validate=validate_driven_point),
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description=('Perfect electric conductor shells: every radial cell whose centre lies in the selected '
                             'domains (circles about the origin select shells) holds tangential E = 0 on its outline '
                             "(er: its cells leave the operator, hr: y = 0 on its nodes); not on the last cell of a "
                             "'radiation' end."),
                summary='Turns the selected shells into a perfect metal.',
                apply=apply_pec, memory_estimate=pec_memory_estimate(0)),
    FeatureType('resonance_table', 'Resonance table', 'monitor', None, result_kind='table',
                description=_RESONANCE_TABLE_DESCRIPTION,
                summary="Lists the resonances found by an 'Eigenfrequency' study: wavelength, frequency, quality "
                        'factor, multipole order and family.',
                apply=apply_resonance_table, result=object_result(sphere_resonance_table_result, with_solver=True),
                validate=_validate_resonance_table),
)

# ---- type ----
_DESCRIPTION = (
    'Spherically symmetric finite-difference frequency-domain solver: on a spherical domain (DomainSpec coordinates '
    'spherical, radius_min 0 for a full sphere or > 0 for a shell region): a 2D (r, theta) window over the whole '
    'theta = 0 .. pi (size_y pi, angle_min 0; the drawing is the meridional half disc) or a 1D radial line (size_y '
    'null), the structure eps(r), '
    'mu(r) is a stack of concentric shells, and Maxwell\'s equations separate into multipole terms (vector spherical '
    'harmonics of order l and family er / hr), each a 1D radial equation for y = r H_t (er) or r E_t (hr) solved on '
    'the radial cell edges (second order, complex symmetric tridiagonal operator, host SuperLU or the opt-in cuDSS / '
    f"MUMPS / PARDISO back ends of the 'linear' role). {_FIELDS_NOTE} The outer end 'radiation' is the exact "
    'outgoing spherical wave of each term (no PML). True 3D results of layered spheres at the cost of one radial line: '
    'Mie scattering '
    '(plane_wave: extinction, scattering and absorption cross sections of coated / graded / plasmonic spheres), '
    'emitters near or inside spheres (dipole_source: emitted power and Purcell factor), resonances (eigenfrequency: '
    'complex omega and Q of whispering-gallery modes, Mie and cavity resonances per order and family; '
    "shift-invert of the 'eigen' role about the target with the radiation term frozen there, then each resonance "
    'refined with the exact radiation condition at its own complex frequency by Rayleigh-functional iteration, so Q '
    'carries no PML or target error; materials stay at the target, as in the other EM eigen-solves). Studies: '
    'stationary (one driven solve at the wavelength setting; with heat through heat_em_thermo_optic / '
    'em_heat_joule: the heat physics should share the spherical domain, it gets the sphere-mean absorbed power), '
    'frequency_domain (one driven solve per wavelength), eigenfrequency. Features: plane_wave, dipole_source, '
    'field_probe, pec_domain, resonance_table. Materials: isotropic eps_r, sigma_e, mu_r, sigma_m and dispersion '
    'poles (no anisotropy), constant on every shell (a material varying along theta is a run error: draw circles '
    'about the origin). Cannot share a model with other EM types. heat_em_thermo_optic '
    "writes eps_r (slot eps_r_update). The instance is usually named 'em'.")


def _families(settings_values) -> tuple:
    family = settings_values['multipole_family']
    return SPHERE_FAMILIES if family == 'both' else (family,)


def _orders(settings_values) -> list:
    return [int(order) for order in settings_values.get('multipole_orders') or ()] or [
        int(settings_values['multipole_order'])]


def _create(ctx, physics_spec, settings_values):
    """SpherePhysics at the first solved wavelength with its dispersive poles and their background permittivity
    written into eps_r; never finalizes."""
    solvers = ctx.solvers(physics_spec.name)
    solver = SpherePhysics(ctx.domain, wavelength=first_solved_wavelength(ctx, settings_values),
                           multipole_orders=_orders(settings_values), families=_families(settings_values),
                           inner_face=settings_values['bc_x_lo'], outer_face=settings_values['bc_x_hi'],
                           linear_solver=solvers['linear'], eigen_solver=solvers.get('eigen'))
    write_dispersive_materials(ctx, physics_spec, solver, SPHERE_PERMITTIVITY_ARRAYS)
    return solver


def _order_issues(settings_values, physics_spec, vctx, path):
    """multipole_orders: integers, distinct (errors); with a driven source a warning when an order 1..max is missing
    (the term sum would miss it)."""
    orders = settings_values.get('multipole_orders') or []
    if not orders:
        return []
    if any(order != int(order) for order in orders):
        return [Issue('error', path, f'multipole_orders must be integers >= 1, got {orders}')]
    orders = [int(order) for order in orders]
    if len(set(orders)) != len(orders):
        return [Issue('error', path, f'multipole_orders must be distinct, got {orders}')]
    sources = _enabled(physics_spec, ('plane_wave', 'dipole_source'))
    missing = sorted(set(range(1, max(orders) + 1)) - set(orders))
    if sources and vctx.study_type in _DRIVEN_STUDIES and missing:
        return [Issue('warning', path, f'multipole_orders {sorted(orders)}: the sum over the terms of the sources '
                                       f'misses l = {missing} (list every order 1..L)')]
    return []


def _enabled(physics_spec, types) -> list:
    return [feature for feature in physics_spec.features if feature.enabled and feature.type in types]


def _source_issues(settings_values, physics_spec, path):
    """Errors: a plane wave beside a dipole source, several plane waves, a plane wave without the radiation end; a
    warning for a radial dipole with family hr alone (no coupling)."""
    issues = []
    plane_waves = _enabled(physics_spec, ('plane_wave',))
    dipoles = _enabled(physics_spec, ('dipole_source',))
    if plane_waves and dipoles:
        issues.append(Issue('error', path, 'a plane_wave and dipole_source features cannot share one physics (their '
                                           'interference is not modelled): use two physics or two runs'))
    if len(plane_waves) > 1:
        issues.append(Issue('error', path, f'at most one enabled plane_wave per physics, got {len(plane_waves)}'))
    if plane_waves and settings_values['bc_x_hi'] != 'radiation':
        issues.append(Issue('error', path, "a plane_wave needs bc_x_hi 'radiation' (the open exterior it comes "
                                           'from)'))
    if settings_values['multipole_family'] == 'hr':
        radial = [feature.name for feature in dipoles if (feature.params or {}).get('orientation', 'radial')
                  == 'radial']
        if radial:
            issues.append(Issue('warning', path, f"radial dipoles {', '.join(radial)} excite only the er (electric) "
                                                 "family, but multipole_family is 'hr': they emit nothing"))
    return issues


_MAX_DISPLAY_ANGLE_CELLS = 360   # theta cells above this only cost memory (the fields do not depend on theta)


def _domain_issues(vctx, path):
    """Errors for a domain that is not spherical over the whole theta = 0 .. pi (a 1D model: only 'spherical'
    coordinates, reported by the geometry rule); a warning for many theta cells."""
    window_map = vctx.domain_map
    if window_map is None:
        return []
    if not window_map.mapped:
        return [Issue('error', path, "em_fdfd_sphere needs a spherical domain (coordinates 'spherical'; a 2D (r, "
                                     'theta) window over theta = 0 .. pi or a 1D radial line), this one is Cartesian')]
    if window_map.line or window_map.kind != 'spherical':
        return []   # 1D lines span the whole angle; polar (r, phi) is refused by the geometry rule
    if abs(window_map.angle_min) > 1e-9 or abs(window_map.size_y - math.pi) > 1e-9 * math.pi:
        return [Issue('error', path, f'em_fdfd_sphere needs the whole polar angle range: angle_min 0 and size_y pi '
                                     f'(got {window_map.angle_min:.6g} and {window_map.size_y:.6g} rad), so every '
                                     'radial cell is a whole spherical shell')]
    mesh = vctx.mesh
    if mesh is not None and int(mesh.ny) > _MAX_DISPLAY_ANGLE_CELLS:
        return [Issue('warning', path, f'{int(mesh.ny)} theta cells: the spherically symmetric fields do not depend '
                                       f'on theta, so they only draw the picture ({int(mesh.nx)} radial cells are '
                                       'solved); '
                                       "use mesh mode nonuniform with a mesh axis 'y' of segments, e.g. 90 cells")]
    return []


def _face_issues(vctx, settings_values, path):
    """bc_x_lo 'centre' exactly when radius_min = 0."""
    window_map = vctx.domain_map
    if window_map is None:
        return []
    centre = settings_values['bc_x_lo'] == 'centre'
    if centre and window_map.radius_min > 0.0:
        return [Issue('error', path, f"bc_x_lo 'centre' needs radius_min 0 (the sphere centre), got radius_min "
                                     f'{window_map.radius_min:.6g} m: use pec or pmc for the inner wall of a shell')]
    if not centre and window_map.radius_min == 0.0:
        return [Issue('error', path, f"bc_x_lo '{settings_values['bc_x_lo']}' at radius_min 0: the inner end is the "
                                     "sphere centre, use 'centre'")]
    return []


def _resolution_issues(vctx, physics_spec, settings_values, path):
    """Warnings for a largest radial cell above 1 / 10 of the shortest wavelength in a material (|n| at every solved
    wavelength, vacuum included) or above half a skin depth."""
    mesh = vctx.mesh
    if mesh is None or getattr(mesh, 'kind', 'tensor') != 'tensor':
        return []
    wavelengths = all_solved_wavelengths(vctx.spec, physics_spec, settings_values, vctx.parameters, vctx.registry)
    wavelengths = np.array([value for value in wavelengths or () if isinstance(value, (int, float)) and value > 0],
                           dtype=np.float64)
    if not wavelengths.size:
        return []
    width = float(np.max(mesh.dx_cells))
    params = vctx.registry.physics_type(physics_spec.type).material_properties
    issues = []
    for material in [None, *vctx.spec.materials]:
        if material is None:
            index, label = np.ones(wavelengths.shape, dtype=np.complex128), 'vacuum'
        else:
            index, label = _complex_index(material, vctx.parameters, params, wavelengths), f"material '{material.name}'"
        with np.errstate(divide='ignore', invalid='ignore'):
            in_medium = np.nan_to_num(wavelengths / np.abs(index), nan=np.inf)
            depth = np.nan_to_num(wavelengths / (2.0 * math.pi * np.abs(index.imag)), nan=np.inf)
        if width * _MIN_CELLS_PER_WAVELENGTH > float(in_medium.min()):
            issues.append(Issue('warning', path, f'{label}: the largest radial cell ({width:.4g} m) gives only '
                                                 f'{float(in_medium.min()) / width:.3g} cells per wavelength in it '
                                                 f'(want >= {_MIN_CELLS_PER_WAVELENGTH:g}); refine the mesh'))
        if width * _MIN_CELLS_PER_SKIN_DEPTH > float(depth.min()):
            issues.append(Issue('warning', path, f'{label}: the largest radial cell ({width:.4g} m) exceeds half its '
                                                 f'skin depth ({float(depth.min()):.4g} m); refine the mesh'))
    return issues


def _validate_physics(vctx, physics_spec, settings_values):
    """Materials (model-wide, once), slot writers, unused wavelength, scipy / back end, faces, orders, sources, other EM
    types, resolution."""
    path = ('physics', physics_spec.name)
    issues = []
    first_em_name, em_types = _em_physics_types(vctx)
    if first_em_name == physics_spec.name:
        issues.extend(_material_issues(vctx, em_types))
    material_params = vctx.registry.physics_type(physics_spec.type).material_properties
    issues.extend(_permittivity_writer_issues(vctx, physics_spec, SPHERE_PERMITTIVITY_SLOTS, material_params))
    issues.extend(unused_wavelength_issues(vctx, settings_values, path))
    issues.extend(scipy_issues(_TYPE_NAME, 'host sparse direct and eigen solvers', path))
    issues.extend(host_matrix_back_end_issues(vctx.solvers(physics_spec) or {}, _TYPE_NAME, path))
    issues.extend(_domain_issues(vctx, path))
    issues.extend(_face_issues(vctx, settings_values, path))
    issues.extend(_order_issues(settings_values, physics_spec, vctx, path))
    issues.extend(_source_issues(settings_values, physics_spec, path))
    issues.extend(missing_source_issues(vctx, physics_spec, path, ('plane_wave', 'dipole_source'), _DRIVEN_STUDIES))
    others = other_em_physics_labels(vctx, _TYPE_NAME)
    if others:
        issues.append(Issue('error', path, f"em_fdfd_sphere solves a spherically symmetric model, while "
                                           f"{', '.join(others)} solve other drawings; they cannot share one model"))
    issues.extend(_resolution_issues(vctx, physics_spec, settings_values, path))
    return issues


# ---- estimates ----
_SOLVE_CLASSES = 2   # source classes per term at most (radial / tangential dipoles; a plane wave alone)
# host bytes moved per radial node: one term's materials read and assembly (diags, free-node slicing, CSR copies) and
# one factorisation + solve of the tridiagonal operator; Python calls per term (uncalibrated: compare the term_solves
# counter's share of a frequency_domain run)
_ASSEMBLY_BYTES_PER_NODE = 600
_FACTOR_BYTES_PER_NODE = 300
_SOLVE_BYTES_PER_NODE = 100
_TERM_CALLS = 60
_REFINE_ITERATIONS_GUESS = 6   # Rayleigh-functional iterations per resonance candidate (cubic convergence)


def _study_count(context, name, default):
    try:
        values = context.registry.study_type(context.spec.study.type).params
        settings = context.spec.study.settings if isinstance(context.spec.study.settings, dict) else {}
        return resolve_values(values, settings, context.parameters).get(name, default)
    except (KeyError, ValueError, TypeError):
        return default


def _memory_estimate(settings_values, mesh, context):
    """Display fields on the run backend (7 float32 radial profiles); host: one term's materials, tridiagonal
    operators and SuperLU factor, vectors and centre fields; eigenfrequency: the Arnoldi basis of one term's search and
    the kept resonance node vectors (every term)."""
    window_cells = int(mesh.n_cells)
    n_cells = int(mesh.nx)    # radial cells (the theta cells only carry the display copies)
    nodes = n_cells + 1
    n_terms = len(_orders(settings_values)) * len(_families(settings_values))
    host_items = [
        ('materials and one term assembly (eps, mu complex128; tridiagonal K, M, A CSR)',
         2 * COMPLEX128_BYTES * n_cells + 9 * COMPLEX128_CSR_ENTRY_BYTES * nodes),
        ('host SuperLU factor of the tridiagonal operator', host_counted_lu_bytes(2 * nodes, nodes)),
        ('solutions, right-hand sides and centre fields (complex128)', (2 * _SOLVE_CLASSES * nodes + 8 * n_cells)
         * COMPLEX128_BYTES)]
    if eigen_solved(context.spec, context.physics_spec.name):
        n_resonances = int(_study_count(context, 'n_resonances', 6) or 6)
        wanted = 2 * n_resonances + 10
        basis = arnoldi_basis_size(wanted, nodes)
        host_items += [('Arnoldi basis and eigenvectors of one term (complex128)',
                        (basis + wanted) * COMPLEX128_BYTES * nodes),
                       (f'resonance node vectors ({n_terms} terms x {n_resonances})',
                        n_terms * n_resonances * COMPLEX128_BYTES * nodes)]
    return MemoryEstimate.from_items((('display fields (7 x float32 per window cell)',
                                       7 * FLOAT32_BYTES * window_cells),),
                                     host_items=host_items)


def _time_estimate(settings_values, mesh, context):
    """Host work per radial node and term: driven = per solve and term one assembly, a factorisation and a solve per
    source class; eigenfrequency = per term one assembly, a shift-invert Arnoldi run (one tridiagonal solve and the
    orthogonalisation per application) and the refinement factorisations of every candidate."""
    nodes = float(int(mesh.nx) + 1)   # radial nodes
    n_terms = float(len(_orders(settings_values)) * len(_families(settings_values)))
    assembly = (te.HostStream(_ASSEMBLY_BYTES_PER_NODE * nodes), te.PythonCalls(_TERM_CALLS))
    factor_solve = (te.HostStream(_FACTOR_BYTES_PER_NODE * nodes), te.PythonCalls(10))
    solve = te.HostStream(_SOLVE_BYTES_PER_NODE * nodes)
    if eigen_solved(context.spec, context.name):
        eigenpairs = float(resonance_eigenpairs(context, optical=False))
        basis = float(arnoldi_basis_size(int(eigenpairs), int(nodes)))
        applications = APPLICATIONS_PER_BASIS_GUESS * basis
        items = (te.TimeItem('multipole term assembly', 'assembly', n_terms, 'exact', assembly),
                 te.TimeItem('shift-invert eigen-solve per term', 'solve', n_terms, 'guess',
                             factor_solve + ((applications, solve),
                                             (applications, te.DenseFlops(16.0 * nodes * basis, host=True)))),
                 te.TimeItem('resonance refinement (exact radiation condition)', 'iteration',
                             n_terms * eigenpairs * _REFINE_ITERATIONS_GUESS, 'guess', factor_solve + (solve,)))
        return te.TimeEstimate(items)
    points = float(context.count('harmonic_points', 1)) if context.study.type == FREQUENCY_DOMAIN_STUDY else 1.0
    solves, kind, limit = context.static_solves()
    return te.TimeEstimate((te.TimeItem(
        'multipole term solve (assembly, factorisation, source classes)', 'solve', points * solves * n_terms, kind,
        assembly + factor_solve + ((float(_SOLVE_CLASSES), solve),),
        limit=None if limit is None else points * limit * n_terms, counter=physics_counter(context, 'term_solves')),))


EM_FDFD_SPHERE = PhysicsType(
    name=_TYPE_NAME,
    label='Electromagnetic Waves, frequency domain (spherically symmetric, multipole orders)',
    family=EM_FAMILY,
    group='em',
    default_instance_name='em',
    description=_DESCRIPTION,
    summary='Light and layered spheres from a single radial line: Mie scattering, emitters near spheres and '
            'sphere resonances with their quality factors, as full 3D results at the cost of one radial line.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=EM_MATERIAL_SCHEMAS[SPHERE_MATERIAL_SCHEMA_KEY],
    state_fields=tuple(SphereRenderer.field_infos[name] for name in ('e_intensity', 'eps_r', 'sigma_e'))
    + (replace(SphereRenderer.field_infos['absorbed_power'], name='q_abs'),),
    output_fields=tuple(SphereRenderer.field_infos.values()),
    source_slots=source_slot_docs(SPHERE_PERMITTIVITY_SLOTS),
    supported_studies=(FREQUENCY_DOMAIN_STUDY, 'stationary', EIGENFREQUENCY_STUDY),
    default_study={'type': FREQUENCY_DOMAIN_STUDY},
    uses_dispersion=True,
    solver_roles={'linear': DIRECT_LINEAR_ROLE, 'eigen': eigen_role((EIGENFREQUENCY_STUDY,))},
    create=_create,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate_physics,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    dimensions=(1, 2),          # spherical domains only: the (r, theta) half disc over 0..pi, or a 1D radial line
    spaces_2d=(AXISYMMETRIC,),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    coordinate_grids=('cartesian', 'polar'),   # 'cartesian' is required by the schema; validate refuses a slab
    backends=('gpu', 'cpu'),    # host solves either way; the shown fields live on the run backend
)


def _selftest_cases():
    """The em_fdfd_sphere SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_sphere import SPHERE_SELFTEST_CASES
    return SPHERE_SELFTEST_CASES


def register(registry):
    """Register em_fdfd_sphere (after the other EM types: their material Params stay first per name) and its selftest
    loader."""
    registry.register_physics(EM_FDFD_SPHERE)
    registry.register_selftests(_selftest_cases)

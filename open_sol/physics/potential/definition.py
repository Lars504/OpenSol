"""Model-layer definitions of the potential physics types 'electric_currents' and 'electrostatics' (one factory:
settings, features, material vocabulary, validation) plus their template and selftests. Loaded lazily by the
registry (physics/potential/__init__.py must not import it)."""
from dataclasses import dataclass

from ...core.mirror import mirror_faces
from ...core.memory_estimate import (BOOL_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate, combine,
                                     diffusion_system_estimate, window_cell_count)
from ...core import time_estimate as te
from ...core.time_estimate import diffusion_time_estimate
from ...fem.diffusion import (cudss_back_end, cudss_diffusion_estimate, fem_diffusion_estimate, fem_time_system,
                              painted_fraction)
from ...fem.diffusion3d import fem_diffusion_estimate_3d
from ...model.definition_helpers import (AXISYMMETRIC_SPACE_SETTINGS, AZIMUTHAL_ORDER_PARAM, CG_DIFFUSION_LINEAR_ROLE,
                                         DISCRETIZATION_PARAM,
                                         antiperiodic_wrap, axisymmetric_issues, azimuthal_order_issues,
                                         azimuthal_order_of,
                                         cg_solver_issues, conducting_fraction, conducts, cross_term_materials,
                                         element_order, evaluates_to, face_conflicts, geometry_param,
                                         held_object_conflicts, indefinite_cross_term_issue, is_axisymmetric,
                                         linear_solver_direct, linear_solver_mixed, linear_solver_multigrid,
                                         linear_solver_precision, make_apply_inward_flux, make_point_probe,
                                         mapped_domain_issues, material_follows_state,
                                         mirror_face_issues, on_tetrahedra, on_triangles, optional_material_param,
                                         reports_model_checks, validate_points_inside, validated_linear_precision)
from ...model.expressions import is_identifier
from ...model.geometry import expand_entities
from ...model.schema import FeatureType, Issue, Param, PhysicsType, feature_path
from .electrolyte import (ELECTROLYTE_MODELS, ION_CONCENTRATION_FIELDS, MAX_BOLTZMANN_EXPONENT, MAX_SPECIES_FIELDS,
                          debye_length, ionic_strength, screening_coefficient, stern_capacitance)
from .renderer import THREE_D_FIELD_INFOS, CurrentsRenderer, ElectrostaticsRenderer
from .selftests import THERMO_OPTIC_HEATER_DESCRIPTION, THERMO_OPTIC_HEATER_LABEL, thermo_optic_heater_spec
from .solver import (ELECTROLYTE_SCALAR_PREFIXES, HELD_SCALAR_PREFIXES, KINDS, NEWTON_MAX_STEP,
                     OPTIONAL_POCKELS_SCHEMA, POCKELS_TENSOR_ARRAYS, POTENTIAL_PROPERTY_ARRAYS, TERMINAL_WAVEFORMS,
                     ElectrolyteRegion, HeldRegion, PotentialPhysics, TerminalWaveform, held_face_bc)
from .solver_fem import FEMHeldRegion, FEMPotentialPhysics
from .solver_fem3d import FEMHeldRegion3D, FEMPotentialPhysics3D

CURRENTS_TYPE_NAME = 'electric_currents'
ELECTROSTATICS_TYPE_NAME = 'electrostatics'

_ANCHOR_FEATURES = frozenset({'voltage', 'terminal'})   # held features (scalar outputs); electrolytes anchor too
_ELECTROLYTE_FEATURE = 'electrolyte'
_WATER_PERMITTIVITY = 80.0          # eps_dc of water near room temperature (Debye length info)
_NEUTRALITY_TOLERANCE = 1e-6        # allowed |sum z_i c_i| / sum |z_i| c_i of an electrolyte bulk
# Window float64 arrays alive at once in one Newton iteration of a nonlinear electrolyte (state, proposal, update,
# Boltzmann charge / screening sums and their per-species temporaries, tangent drive): a run item.
_NEWTON_WINDOW_ARRAYS = 12
_IONIC_DIAGONAL_ARRAYS = 6   # the same for the tangent write of every assembly (linear model: charge, screening, drive)
_FEM_IONIC_ARRAYS = 6        # triangles: k, k V_bulk, held mask, mean V, charge, one concentration (per element)
_FLOAT32_NEWTON_TOLERANCE = 1e-3   # smallest nonlinear_tolerance float32 CG meets (~1e-6 of V: 4e-5 V_T at 1 V)
# a potential boundary feature on an EM mirror plane (definition_helpers.mirror_face_issues)
_MIRROR_FACE_CONSEQUENCE = (
    'its fields unfold even there (an insulated face). A voltage face is the odd mirror of V - V_face instead (an '
    'antisymmetric bias about the plane, exact for the Dirichlet half cell), a flux face a sheet source of twice its '
    'flux on the plane; unfolded fields and doubled integrals of this physics then do not hold')


@dataclass(frozen=True)
class _KindNames:
    """Names of one potential kind: type, label, default instance, renderer, property prefix, held quantity, flux
    feature, and the coefficient as help texts write it."""
    type_name: str
    label: str
    instance_name: str
    renderer_class: type
    property_prefix: str
    held_quantity: str
    flux_feature: str
    coefficient_text: str


_KIND_NAMES = {
    'currents': _KindNames(CURRENTS_TYPE_NAME, 'Electric Currents', 'currents', CurrentsRenderer, 'sigma_dc',
                           'current', 'current_density', 'sigma_dc'),
    'electrostatics': _KindNames(ELECTROSTATICS_TYPE_NAME, 'Electrostatics', 'electrostatics',
                                 ElectrostaticsRenderer, 'eps_dc', 'charge', 'surface_charge', 'eps0 eps_dc'),
}


# ---- settings ----
_RING_WEIGHT_TEXT = ('every conductance, face and cell volume is weighted by 2 pi r (x-faces at their own r, y-faces and '
                     'volumes at the cell-centre r; 2 pi r_c dr dz is the exact ring volume; object-perimeter {flux} '
                     'likewise), so densities ({densities}) keep their units, while the {quantity} of a voltage / '
                     'terminal (table result, scalar output {quantity}_<feature>) is that of the whole ring in {unit}'
                     '{totals}; {cross}_xy is not available (validation error); azimuthal_order m != 0 weights by pi r '
                     'instead')
_GEOMETRY_PARAMS = {
    'currents': geometry_param(
        '2D currents per unit z length (terminal currents A/m)',
        '(r, z) currents; ' + _RING_WEIGHT_TEXT.format(
            flux='current_density', densities='sigma_dc S/m, current_source A/m^3, current_density A/m^2, J A/m^2, '
                                              'q_joule W/m^3',
            quantity='current', unit='A', cross='sigma_dc',
            totals=" (a resistance V / I in Ohm), and the 'integral' of q_joule is the total power in W"),
        'currents are per metre of depth.',
        'Currents of voltages and terminals are those of the whole ring, in amperes.'),
    'electrostatics': geometry_param(
        '2D electrostatics per unit z length (terminal charges C/m)',
        '(r, z) electrostatics; ' + _RING_WEIGHT_TEXT.format(
            flux='surface_charge', densities='charge_density C/m^3, surface_charge C/m^2, D C/m^2',
            quantity='charge', unit='C', cross='eps_dc', totals=' (a capacitance Q / V in F)'),
        'charges are per metre of depth.',
        'Charges of voltages and terminals are those of the whole ring, in coulombs.'),
}
_NEWTON_SCALE_TEXT = 'kB T / (e max |z_i|) (the smallest over the nonlinear electrolytes; 25.7 mV for 1:1 at 298.15 K)'
_NONLINEAR_SETTINGS = (   # damped Newton of nonlinear electrolytes (core.damped_iteration; plans_overview decision 7)
    Param('nonlinear_tolerance', 'float', 1e-6, label='Electrolyte iteration tolerance', unit='1', min=0.0,
          min_exclusive=True,
          help='Stop of the damped Newton loop of nonlinear electrolytes (feature electrolyte, model nonlinear): '
               f'converged when one Newton step changes V by at most this times {_NEWTON_SCALE_TEXT} anywhere. Each '
               'step is one CG solve warm-started at the previous V, so keep it above the CG accuracy in V (the '
               'linear pcg solver tolerance times the condition number, relative to the drive). Ignored without a nonlinear '
               'electrolyte.',
          tip='How accurately the potential of a nonlinear electrolyte is converged, as a fraction of the thermal '
              'voltage (about 26 millivolts).\n\nKeep 1e-6. Only used by electrolytes with the nonlinear model.'),
    Param('nonlinear_max_iterations', 'int', 100, label='Electrolyte iteration limit', min=1,
          help='Newton iteration limit per solve of a nonlinear electrolyte; an unconverged loop keeps its damped V '
               'and logs a warning (study log). Steps are capped at '
               f'{NEWTON_MAX_STEP:g} x {_NEWTON_SCALE_TEXT}, so a surface or electrode potential of P such units '
               f'needs at least P / {NEWTON_MAX_STEP:g} iterations. Ignored without a nonlinear electrolyte.',
          tip='Largest number of Newton iterations for a nonlinear electrolyte; each one is a full solve.\n\n'
              'A loop that reaches the limit keeps its unfinished result and logs a warning.'),
    Param('nonlinear_relaxation', 'float', 1.0, label='Electrolyte iteration damping', unit='1', min=1.0 / 64.0,
          max=1.0,
          help='Fraction of each Newton update applied, [1/64, 1]; it halves automatically (down to 1/64) while the '
               'update grows and recovers once it shrinks. 1 (full Newton steps) suits the convex Boltzmann charge; '
               'lower it only when the loop oscillates. Ignored without a nonlinear electrolyte.',
          tip='Fraction of each correction that is applied, between 0 and 1.\n\nKeep 1; lower it only if the '
              'iteration does not settle.'),
)
_SETTINGS = {
    'currents': (_GEOMETRY_PARAMS['currents'], AZIMUTHAL_ORDER_PARAM, DISCRETIZATION_PARAM),
    'electrostatics': (_GEOMETRY_PARAMS['electrostatics'], AZIMUTHAL_ORDER_PARAM, DISCRETIZATION_PARAM,
                       *_NONLINEAR_SETTINGS),
}
_BOTH_GRIDS = ('tensor', 'triangles')   # FeatureType.grids of the features that run on triangle meshes too
_FEM_GRIDS = ('tensor', 'triangles', 'tetrahedra')   # ... and on tetrahedra (FEMPotentialPhysics3D)


# ---- material vocabulary ----
def _material_property(name, label, unit, minimum=None, help_text='', tip=''):
    """Potential material property Param (default None: unset properties are never written)."""
    return optional_material_param(name, label, unit, POTENTIAL_PROPERTY_ARRAYS[name], minimum, help_text, tip)


_SIGMA_DC_HELP = (
    'Isotropic static (DC) conductivity: sets sigma_dc_xx, sigma_dc_yy and sigma_dc_zz (component entries override '
    'it; zz is read only on a tetrahedron mesh in 3D); unset = '
    '0 = insulator: cells without sigma_dc carry no current and keep their potential (as k = 0 for heat), so every '
    'conducting region needs a voltage or terminal. NOT the optical sigma_e, the loss proxy at the solve wavelength '
    '(water: sigma_e 4.22 S/m at 1.55 um but sigma_dc 5.5e-6 S/m). Typical gold 4.1e7, silver 6.3e7, aluminium '
    '3.8e7, water 5.5e-6; may depend on x, y and state fields, e.g. gold 4.1e7 / (1 + 0.0034 * (heat_T - 293.15)) '
    '(a stationary study iterates). 2D quantities are per unit z length: currents A/m, conductances S and '
    'capacitances F per m of depth, resistances Ohm m (geometry axisymmetric and 3D: whole totals, A, S, F, Ohm).')
_EPS_DC_HELP = (
    'Isotropic static relative permittivity (>= 1): sets eps_dc_xx, eps_dc_yy and eps_dc_zz (component entries '
    'override it; zz is read only on a tetrahedron mesh in 3D); '
    'unset = 1 (vacuum), so every cell takes part in the solve and one voltage / terminal anchors everything. NOT '
    'the optical eps_r = n^2: lithium niobate eps_dc 43 (eps_a) / 28 (eps_c, along the optic axis) vs eps_r 4.9 / '
    '4.6; silicon 11.7 vs 12.08; silica 3.9 vs 2.085; water 80.1 vs 1.73. 2D: charges C/m, capacitances F per m of '
    'depth (geometry axisymmetric and 3D: whole totals, C, F).')
_XY_HELP = ('Off-diagonal component of {name} ({cross} of a rotated anisotropic material); keep {name}_xy^2 '
            '< {name}_xx * {name}_yy (validation error for plain numbers); uniform or graded meshes (on strongly graded '
            'cells keep it clearly below that bound: the discrete operator is guaranteed positive below it times 4 r / '
            '(1 + r)^2 per axis, r = width ratio of neighbouring cells). Cross flux flows only inside '
            '2x2 cell blocks whose xx and yy components are all > 0 (area-weighted mean {name}_xy of the block), never '
            'through the outer faces. E stays -grad V; {outputs} include the cross part {cross_part} (dV/dy, dV/dx), '
            'the block values averaged over each cell\'s blocks{consequence}.')
_XY_HELP_TEXTS = {
    'currents': {'name': 'sigma_dc', 'cross': 'cross conduction', 'outputs': 'the current density J',
                 'cross_part': '-sigma_dc_xy',
                 'consequence': ', and q_joule includes each block\'s cross power (so its integral equals V * I)'},
    'electrostatics': {'name': 'eps_dc', 'cross': 'cross polarisation', 'outputs': 'the displacement D',
                       'cross_part': '-eps0 eps_dc_xy', 'consequence': ''},
}
_SIGMA_DC_TIP = ('Electrical conductivity for steady currents; empty means an insulator.\n\n'
                 'Typical: gold 4.1e7, aluminium 3.8e7, water 5.5e-6.\n\n'
                 "Not the optical 'Electric conductivity' of the wave physics, which can differ by many orders of "
                 'magnitude.')
_EPS_DC_TIP = ('Relative permittivity for static fields; empty means vacuum, 1.\n\n'
               'Often far above the optical value: silica 3.9 against 2.1, water 80 against 1.7.')
_SIGMA_DC_COMPONENT_TIP = ('Static conductivity along {axis} of a material that conducts differently along x and y. '
                           "Replaces 'Static conductivity' for this direction.")
_EPS_DC_COMPONENT_TIP = ('Static relative permittivity along {axis} of a material that responds differently along x '
                         "and y. Replaces 'Static relative permittivity' for this direction.")
_CROSS_TERM_TIP_END = '\n\nKeep it smaller than both the x and y values.'
_POCKELS_HELP = (
    'Isotropic Pockels (linear electro-optic) coefficient: sets r_xx, r_yy and r_zz (component entries override it), '
    'read only by the electrostatics_em_pockels coupling: dn_ii = -1/2 n0_ii^3 r_ii E_k with E_k the applied field '
    'component chosen by its field_component. The diagonal of the Pockels tensor for one applied direction; unset = '
    '0 (no effect). Typical: x-cut lithium niobate with the field along the optic axis r33 = 31e-12 on the optic-axis '
    'component and r13 = 10e-12 on the others (library lithium_niobate_x_cut); 0 for centrosymmetric silicon and '
    'silica. A crystal entry sets the tensor entries pockels_r11 .. _r62 instead.')
_POCKELS_TIP = ('Strength of the electro-optic (Pockels) effect: how much a static electric field changes the '
                'refractive index. Only the Pockels coupling to a wave physics uses it; empty means no effect.\n\n'
                'Typical: lithium niobate 31e-12 along its optic axis, 10e-12 across it.')
_POCKELS_COMPONENT_TIP = ("Electro-optic coefficient for light with its electric field along {axis}, {where}. "
                          "Replaces 'Electro-optic coefficient' for this direction.")
_POCKELS_TENSOR_HELP = (
    'In-plane Pockels tensor entry {name} [m/V] of a crystal in model axes: Delta(1/n^2)_I = r_I1 E_x + r_I2 E_y with '
    'the index-ellipsoid row I = 1 xx, 2 yy, 3 zz (the diagonal permittivity slots) or 6 xy (the eps_xy slot of '
    'em_fdfd_hz: Delta eps_xy = -n0_xx^2 n0_yy^2 Delta(1/n^2)_xy; other EM types drop it, validation warns for '
    'crystals), read only by electrostatics_em_pockels, which adds -1/2 n0_ii^3 (r_i1 E_x + r_i2 E_y) from both field '
    'components whatever its field_component (on top of r_ii E_k: set either these or r / r_ii on a material, not '
    "both). Usually set by a {{'model': 'crystal'}} entry: the clamped (strain-free) r^S rotated into model axes, so a "
    'piezoelectric strain computed by elastic_waves adds its own index change through acoustics_em_acousto_optic '
    'without double counting; YZ lithium niobate (model x = crystal Z, y = -Y): r11 = r33, r12 = 0, r21 = r13, r22 = '
    '-r22, r31 = r13, r32 = r22, r61 = 0, r62 = r51 (crystal constants). Default 0; any one set allocates all eight '
    'arrays.')
_POCKELS_TENSOR_ROWS = {'1': 'x', '2': 'y', '3': 'z (out of the plane)', '6': 'x and y together (the cross term)'}
_MATERIAL_PROPERTIES = {   # isotropic entries before their components (components override)
    'currents': (
        _material_property('sigma_dc', 'Static conductivity', 'S/m', 0.0, _SIGMA_DC_HELP, tip=_SIGMA_DC_TIP),
        _material_property('sigma_dc_xx', 'Static conductivity along x', 'S/m', 0.0,
                           'DC conductivity along x; overrides sigma_dc for this axis.',
                           tip=_SIGMA_DC_COMPONENT_TIP.format(axis='x')),
        _material_property('sigma_dc_yy', 'Static conductivity along y', 'S/m', 0.0,
                           'DC conductivity along y; overrides sigma_dc for this axis.',
                           tip=_SIGMA_DC_COMPONENT_TIP.format(axis='y')),
        _material_property('sigma_dc_zz', 'Static conductivity along z', 'S/m', 0.0,
                           'DC conductivity along z (3D on a tetrahedron mesh only; 2D ignores it); overrides sigma_dc '
                           'for this axis.',
                           tip="Static conductivity along z, used only in 3D models. Replaces 'Static conductivity' "
                               'for this direction.'),
        _material_property('sigma_dc_xy', 'Static conductivity, cross term', 'S/m',
                           help_text=_XY_HELP.format(**_XY_HELP_TEXTS['currents']),
                           tip='Cross term of a rotated conductor: current also flows sideways to the field. Usually '
                               'empty.' + _CROSS_TERM_TIP_END),
    ),
    'electrostatics': (
        _material_property('eps_dc', 'Static relative permittivity', '1', 1.0, _EPS_DC_HELP, tip=_EPS_DC_TIP),
        _material_property('eps_dc_xx', 'Static relative permittivity along x', '1', 1.0,
                           'Static relative permittivity along x (>= 1); overrides eps_dc for this axis.',
                           tip=_EPS_DC_COMPONENT_TIP.format(axis='x')),
        _material_property('eps_dc_yy', 'Static relative permittivity along y', '1', 1.0,
                           'Static relative permittivity along y (>= 1); overrides eps_dc for this axis.',
                           tip=_EPS_DC_COMPONENT_TIP.format(axis='y')),
        _material_property('eps_dc_zz', 'Static relative permittivity along z', '1', 1.0,
                           'Static relative permittivity along z (>= 1; electrostatics in 3D on a tetrahedron mesh, '
                           'solid_mechanics_3d with piezoelectric on; 2D ignores it); overrides eps_dc for this axis.',
                           tip="Static relative permittivity along z, used only in 3D models. Replaces 'Static "
                               "relative permittivity' for this direction."),
        _material_property('eps_dc_xy', 'Static relative permittivity, cross term', '1',
                           help_text=_XY_HELP.format(**_XY_HELP_TEXTS['electrostatics']),
                           tip='Cross term of a rotated crystal. Usually empty.' + _CROSS_TERM_TIP_END),
        _material_property('r', 'Electro-optic coefficient', 'm/V', help_text=_POCKELS_HELP, tip=_POCKELS_TIP),
        *(_material_property(f'r_{axis}{axis}', f'Electro-optic coefficient along {axis}', 'm/V',
                             help_text=f'Pockels coefficient of n_{axis}{axis} (the index seen by light with its '
                                       f'electric field along {axis}); overrides r for this component.',
                             tip=_POCKELS_COMPONENT_TIP.format(axis=axis, where=where))
          for axis, where in (('x', 'in the plane'), ('y', 'in the plane'), ('z', 'out of the plane'))),
        *(_material_property(name, f'Electro-optic tensor {name[-3:]}', 'm/V',
                             help_text=_POCKELS_TENSOR_HELP.format(name=name),
                             tip=f'How strongly a static field along {"x" if name[-1] == "1" else "y"} changes the '
                                 'refractive index seen by light polarised along '
                                 f'{_POCKELS_TENSOR_ROWS[name[-2]]}, for a crystal. Usually filled in by a crystal '
                                 'entry; empty means none.')
          for name in POCKELS_TENSOR_ARRAYS),
    ),
}


# ---- feature texts ----
_HELD_SIGN_TEXT = (
    'positive = current (charge) per m of depth (geometry axisymmetric: of the whole ring) flowing out of the held '
    'faces / cells into the domain (the face '
    'conductance times V_face - V_cell summed along a held outer face; for held cells the sum over them of (A V - b) '
    'of the un-eliminated system, so an injected source inside the terminal is subtracted); electrostatics: the free '
    'charge on the terminal (its outward D flux, Gauss). A cell held by two features is attributed to each (the '
    'later voltage holds it; validation warns when held objects may overlap); '
    'an outer face whose condition a later feature replaced (the last one wins) counts only for that feature. '
    'E and J / D read 0 inside held cells (an ideal conductor): the surface current / charge is this result.')
_HELD_RESULT_TEXT = {
    'currents': ("Result 'table' of the latest solve with one row [feature, voltage, current] (units '', V, A/m; "
                 'axisymmetric and 3D A) and the extras voltage (V) and current (A/m; axisymmetric and 3D A): '
                 f'{_HELD_SIGN_TEXT}'),
    'electrostatics': ("Result 'table' of the latest solve with one row [feature, voltage, charge] (units '', V, C/m; "
                       'axisymmetric and 3D C) and the extras voltage (V) and charge (C/m; axisymmetric and 3D C): '
                       f'{_HELD_SIGN_TEXT}'),
}
_TERMINAL_BODY_TEXT = {
    'currents': 'A metal body with its own sigma_dc is equipotential anyway: a terminal is for feeding current in '
                'and reading it out (its cells need no sigma_dc: they act as an ideal conductor).',
    'electrostatics': 'Electrostatics has no conductors of its own (every cell is a dielectric), so a terminal is '
                      'the conductor: its cells are held at the voltage (their eps_dc does not matter, so a gold '
                      'electrode needs none) and its free charge is read out.',
}
_SOURCE_PARAM_HELP = {
    'currents': 'Volumetric current injection; in 2D density * area is the line current per m of depth (A/m; '
                'geometry axisymmetric: density * ring volume, A). May depend on x and y; must be finite on the '
                'selected cells. A steady solve ignores it in non-conducting cells (sigma_dc 0).',
    'electrostatics': 'rho in div(eps0 eps_dc grad V) = -rho; positive charge raises V. In 2D density * area is the '
                      'line charge per m of depth (C/m; geometry axisymmetric: density * ring volume, C). May depend '
                      'on x and y; must be finite on the selected cells.',
}
_SOURCE_DESCRIPTION = {
    'currents': 'Applied volumetric current source: density is added to current_source_ext on the selected cells '
                '(several sources add; the coupled slot and field current_source are separate). The '
                'injected current leaves through the voltage / terminal features; non-conducting cells ignore it.',
    'electrostatics': 'Applied volumetric free charge: density is added to rho_free_ext on the selected cells '
                      '(several sources add; the coupled slot charge_density, field rho_free, is separate).',
}
_SLOT_DOCS = {   # coupled source slot: (unit, meaning)
    'currents': ('A/m^3', 'Coupled volumetric current source (field current_source), added to the current_source '
                          'features (field current_source_ext); each transfer replaces it.'),
    'electrostatics': ('C/m^3', 'Coupled free charge density (field rho_free), added to the charge_density features '
                                '(field rho_free_ext); each transfer replaces it.'),
}
_PROBE_DESCRIPTION = ('Potential (V) of the cell containing the point (time-dependent study: one sample per re-solve '
                      'plus one at the end of the run, V holding its value in between; electrostatics with a voltage '
                      'waveform: one per step at its end time; stationary study and the '
                      'background stage of a frequency-domain study: one per fixed-point iteration, all at t = 0).')


_PHYSICS_SUMMARIES = {
    'currents': 'Steady electric currents in conductors, driven by voltages or injected currents. Gives the potential, '
                'the current density and the resistive (Joule) heating.',
    'electrostatics': 'Static electric potential and field around electrodes and charges in insulators. Gives the '
                      'charge on each electrode.',
}
_VOLTAGE_SUMMARIES = {
    'currents': 'Holds the selected domain edges or object boundaries at a fixed voltage and reports the current '
                'through them. Every conducting region needs a voltage or a terminal.',
    'electrostatics': 'Holds the selected domain edges or object boundaries at a fixed voltage and reports their '
                      'charge. The model needs at least one voltage or terminal.',
}
_TERMINAL_SUMMARIES = {
    'currents': 'Turns the selected domains into an ideal electrode at a fixed voltage and reports the current it '
                'feeds in, per metre of depth (for the whole ring in an axisymmetric model). The electrode needs no '
                'conductivity of its own.',
    'electrostatics': 'Turns the selected domains into an ideal conductor, an electrode at a fixed voltage, and '
                      'reports its charge per metre of depth (for the whole ring in an axisymmetric model). Its '
                      'permittivity does not matter.',
}
_SOURCE_PARAM_TIPS = {
    'currents': 'Current injected per volume; it leaves through the voltages and terminals.\n\n'
                'Ignored in materials without static conductivity.',
    'electrostatics': 'Free charge per volume; positive charge raises the potential.\n\n'
                      'The model is per metre of depth: charge density times area gives coulombs per metre (in an '
                      'axisymmetric model, times the ring volume: coulombs).',
}
_SOURCE_SUMMARIES = {
    'currents': 'Injects current throughout the selected domains.',
    'electrostatics': 'Places a fixed charge throughout the selected domains.',
}

_ELECTROLYTE_PARAMS = (
    Param('concentrations', 'float_list', [1e-3, 1e-3], label='Ion concentrations', unit='mol/L', min=0.0,
          help='Bulk molar concentration c_i of each ion species (1 mM = 1e-3 mol/L), in the order of valences; the '
               'bulk must be neutral, sum z_i c_i = 0 (validation error). Uniform over the selection; parameter '
               'expressions allowed (e.g. a concentration sweep).',
          tip='Concentration of each kind of ion far from any charge, in moles per litre, in the same order as the '
              "'Ion valences'.\n\nExample for 1 millimolar table salt: 1e-3, 1e-3."),
    Param('valences', 'float_list', [1.0, -1.0], label='Ion valences', unit='1',
          help='Charge number z_i of each species, one per concentration, none 0: Na+ / K+ 1, Cl- -1, Ca2+ 2, SO4 2- '
               '-2 (CaCl2: valences 2, -1 with concentrations c, 2 c).',
          tip='Charge of each kind of ion in elementary charges: 1 for sodium, -1 for chloride, 2 for calcium.\n\n'
              'Example for calcium chloride: 2, -1, with twice as much chloride as calcium.'),
    Param('temperature', 'float', 298.15, label='Temperature', unit='K', min=0.0, min_exclusive=True,
          help='Temperature of the Boltzmann factor kB T (the screening coefficient scales as 1 / T). Typical 298.15 K.',
          tip='Temperature of the electrolyte.\n\nTypically 298.15 K.'),
    Param('bulk_potential', 'float', 0.0, label='Bulk potential', unit='V',
          help='Potential V_bulk of the bulk electrolyte, where every species has its bulk concentration (e.g. that of '
               'a reference electrode in it): rho_ion = -k (V - V_bulk), so V relaxes to it away from charges and '
               'electrodes.',
          tip='Potential the electrolyte settles to away from charged surfaces and electrodes, for example that of a '
              'reference electrode.\n\nUsually 0 V.'),
    Param('model', 'choice', 'linear', label='Electrolyte model', choices=ELECTROLYTE_MODELS,
          choice_labels=(('linear', 'Linear (small potentials)'), ('nonlinear', 'Nonlinear (Poisson-Boltzmann)')),
          help='linear (Debye-Hueckel, one solve): rho_ion = -k (V - V_bulk), valid for |V - V_bulk| << kB T / (|z| '
               'e); nonlinear (Poisson-Boltzmann, damped Newton over the physics settings nonlinear_tolerance / '
               'nonlinear_max_iterations / nonlinear_relaxation): rho_ion = sum_i z_i e n_i exp(-z_i e (V - V_bulk) '
               '/ (kB T)), for surface and electrode potentials of several kB T / e.',
          tip='How the ions respond to the potential.\n\n'
              '- Linear: one solve; accurate only well below 25 millivolts.\n'
              '- Nonlinear: the full Boltzmann response, solved by repeated solves; needed for larger potentials, '
              'such as charged surfaces in dilute salt.'),
)
_ELECTROLYTE_DESCRIPTION = (
    'Mobile ions screening the field (Poisson-Boltzmann): on the selected cells the ionic space charge rho_ion enters '
    'the solve, -div(eps0 eps_dc grad V) = rho_free + rho_ion. model linear (Debye-Hueckel, default): rho_ion = -k (V '
    '- bulk_potential), k = sum_i z_i^2 e^2 n_i / (kB T) [F/m^3] with n_i = 1000 N_A c_i, a diagonal term, so V '
    'decays over the Debye length lambda_D = sqrt(eps0 eps_dc / k) (water, eps_dc 80, 298.15 K: 0.307 nm / sqrt(I '
    '[mol/L]), 9.7 nm at 1 mM NaCl); valid for |V - V_bulk| << kB T / (|z| e) = 25.7 mV / |z| at 298 K (larger '
    'potentials overestimate the screening charge). model nonlinear: point-charge Boltzmann ions, rho_ion = sum_i z_i '
    'e n_i exp(-z_i e (V - V_bulk) / (kB T)), solved by damped Newton: each iteration replaces the diagonal term by '
    'the tangent of rho_ion at the current V and runs one CG solve warm-started there (the first iteration from V = 0 '
    'is the linear solution), steps capped at '
    f'{NEWTON_MAX_STEP:g} kB T / (e max |z_i|), converged when a step changes V by at most nonlinear_tolerance x kB T '
    '/ (e max |z_i|) (physics settings nonlinear_tolerance / nonlinear_max_iterations / nonlinear_relaxation; '
    'iterations in the study summary solvers (the physics counters) as nonlinear_iterations; an unconverged loop '
    'keeps the damped V and '
    'logs a warning). It has no ion size: counter-ion concentrations above a few mol/L (|V - V_bulk| beyond ~0.2 V / '
    f'|z| at 1 mM salt) are unphysical, and the Boltzmann exponent is clipped at {MAX_BOLTZMANN_EXPONENT:g}. Both '
    'models anchor the solve: no voltage or terminal is needed (e.g. a charged surface given by surface_charge). Held '
    'cells (terminals, voltage perimeters) are ideal conductors without ions; overlapping electrolytes add (one Newton '
    'loop when any of them is nonlinear). Field rho_ion [C/m^3] (nonlinear: the Boltzmann charge of the solved V); '
    f'fields {ION_CONCENTRATION_FIELDS[0]} .. {ION_CONCENTRATION_FIELDS[-1]} [mol/L] = the local concentration of '
    'species k (the k-th entry of concentrations / valences) at the solved V, c_k exp(-z_k e (V - V_bulk) / (kB T)) '
    '(nonlinear) or its linearisation c_k (1 - z_k e (V - V_bulk) / (kB T)) (linear: the concentrations of its '
    'rho_ion, negative beyond kB T / (|z_k| e)), so sum_k z_k e 1000 N_A c_ion_k = rho_ion; 0 outside the '
    'electrolytes and on held cells, overlapping electrolytes add their k-th species; listed up to the longest '
    f'species list, at most {MAX_SPECIES_FIELDS} (more species: validation warning); output fields only (not usable '
    'in expressions). '
    'Scalar outputs ionic_strength_<feature> [mol/L] = 1/2 sum z_i^2 c_i and debye_length_<feature> [m] (the bulk, '
    'linearised value in both models) at the mean eps_dc of the selected cells at build (the static material value). '
    'Resolve lambda_D with several cells (cells of lambda_D / 3 or finer at charged surfaces and electrodes): the '
    'screening layer there is lambda_D thick, and with model nonlinear thinner at high potentials (1:1: lambda_D / '
    'cosh(e (V - V_bulk) / (2 kB T))), so refine further. Validation reports the ionic strength and lambda_D (info). '
    'geometry axisymmetric: ring volumes, as for every density. Triangle meshes: model linear only, the terms per '
    'element (rho_ion and c_ion_k from the element mean V; mean eps_dc area-weighted).')


def _electrolyte_species(values):
    """(concentrations, valences, net charge sum z_i c_i, sum |z_i| c_i) of electrolyte values [mol/L]."""
    concentrations, valences = values['concentrations'], values['valences']
    net = sum(valence * concentration for concentration, valence in zip(concentrations, valences))
    scale = sum(abs(valence) * concentration for concentration, valence in zip(concentrations, valences))
    return concentrations, valences, net, scale


def _validate_electrolyte(vctx, physics_spec, feature_spec, values):
    """Species lists (one valence per concentration, none 0), neutral bulk, no ions at all (warning), and the ionic
    strength / Debye length (info)."""
    path = feature_path(physics_spec, feature_spec)
    concentrations, valences, net, scale = _electrolyte_species(values)
    if not concentrations or len(concentrations) != len(valences):
        return [Issue('error', path, f'an electrolyte needs one valence per concentration and at least one species, '
                                     f'got {len(concentrations)} concentrations and {len(valences)} valences')]
    if any(valence == 0.0 for valence in valences):
        return [Issue('error', path, 'electrolyte valences must be non-zero (an uncharged species does not screen)')]
    if scale == 0.0:
        return [Issue('warning', path, 'every ion concentration is 0: the electrolyte neither screens nor anchors '
                                       'the solve')]
    issues = []
    if len(valences) > MAX_SPECIES_FIELDS:
        issues.append(Issue('warning', path, f'{len(valences)} ion species: only the first {MAX_SPECIES_FIELDS} have '
                                             f'a concentration field ({ION_CONCENTRATION_FIELDS[0]} .. '
                                             f'{ION_CONCENTRATION_FIELDS[-1]}); every species enters the solve'))
    if abs(net) > _NEUTRALITY_TOLERANCE * scale:
        issues.append(Issue('error', path, f'the bulk electrolyte is not neutral: sum z_i c_i = {net:.4g} mol/L; both '
                                           'electrolyte models need sum z_i c_i = 0 (e.g. CaCl2: valences 2, -1 with '
                                           'concentrations c, 2 c)'))
    coefficient = screening_coefficient(concentrations, valences, values['temperature'])
    issues.append(Issue('info', path, f'ionic strength {ionic_strength(concentrations, valences):.4g} mol/L, Debye '
                                      f'length {debye_length(1.0, coefficient):.4g} m x sqrt(eps_dc) '
                                      f'({debye_length(_WATER_PERMITTIVITY, coefficient):.4g} m at eps_dc '
                                      f'{_WATER_PERMITTIVITY:g}, water): resolve it with several cells'))
    return issues


def _apply_electrolyte(ctx, solver, feature_spec, values, selection):
    """Ionic space charge (linear or nonlinear model) on the selected cells; returns the ElectrolyteRegion (ionic
    strength, bulk Debye length at the mean eps_dc of the selection)."""
    concentrations, valences, _, _ = _electrolyte_species(values)
    coefficient = screening_coefficient(concentrations, valences, values['temperature'])
    mask = ctx.domain_mask(selection.entities)
    solver.add_electrolyte(mask, concentrations, valences, values['temperature'], values['bulk_potential'],
                           values['model'])
    permittivity = solver.mean_relative_permittivity(mask)
    region = ElectrolyteRegion(feature_spec.name, ionic_strength(concentrations, valences),
                               debye_length(permittivity, coefficient))
    ctx.log('info', f"feature '{feature_spec.name}': Debye length {region.debye_length:.4g} m at the mean eps_dc "
                    f'{permittivity:.4g} of the selection, ionic strength {region.ionic_strength:.4g} mol/L')
    return solver.add_electrolyte_region(region)


def _electrolyte_anchors(vctx, feature):
    """True when an enabled electrolyte feature has valid values with a non-zero sum z_i^2 c_i."""
    if not feature.enabled or feature.type != _ELECTROLYTE_FEATURE:
        return False
    values = vctx.values(_ELECTROLYTE_PARAMS, feature.params)
    return values is not None and screening_coefficient(values['concentrations'], values['valences'], 1.0) > 0.0


def _nonlinear_electrolytes(vctx, physics_spec):
    """The enabled electrolyte features of physics_spec whose values resolve to model 'nonlinear'."""
    return [feature for feature in physics_spec.features
            if feature.enabled and feature.type == _ELECTROLYTE_FEATURE
            and (vctx.values(_ELECTROLYTE_PARAMS, feature.params) or {}).get('model') == 'nonlinear']


def _nonlinear_precision_issues(vctx, physics_spec, settings_values, path):
    """Warning: a nonlinear electrolyte with a single-precision CG and a Newton tolerance below what float32 V can
    resolve."""
    if not _nonlinear_electrolytes(vctx, physics_spec) or validated_linear_precision(vctx, physics_spec) != 'float32' \
            or settings_values['nonlinear_tolerance'] >= _FLOAT32_NEWTON_TOLERANCE:
        return []
    return [Issue('warning', path, "nonlinear electrolyte with a single-precision linear solver: each Newton step's "
                                   f'CG solve is accurate to ~1e-6 of V, so nonlinear_tolerance '
                                   f"{settings_values['nonlinear_tolerance']:g} may never be met; use a double or mixed "
                                   f'precision pcg node or nonlinear_tolerance >= {_FLOAT32_NEWTON_TOLERANCE:g}')]


def _electrolyte_memory_estimate(values, n_selected_cells, mesh):
    """Two bool window masks (the IonicMedium's, the union) and the temporaries of the ionic diagonal write (model
    nonlinear: of a Newton iteration) as a transient item; triangles (linear only): the medium's element mask and the
    per-element ionic loss / charge temporaries; tetrahedra (refused by FeatureType.grids): nothing."""
    if mesh.kind == 'tetrahedra':
        return MemoryEstimate()
    if mesh.kind == 'triangles':
        n_elements = int(mesh.n_elements)
        return MemoryEstimate.from_items(host_items=(
            ('electrolyte element mask (bool)', BOOL_BYTES * n_elements),
            (f'ionic terms ({_FEM_IONIC_ARRAYS} x float64 per element)', _FEM_IONIC_ARRAYS * FLOAT64_BYTES * n_elements)))
    window_cells = window_cell_count(mesh)
    nonlinear = values['model'] == 'nonlinear'
    n_arrays = _NEWTON_WINDOW_ARRAYS if nonlinear else _IONIC_DIAGONAL_ARRAYS
    label = 'Newton iteration' if nonlinear else 'ionic diagonal'
    return MemoryEstimate.from_items((('electrolyte masks (2 x bool window)', 2 * BOOL_BYTES * window_cells),),
                                     transient_items=((label, f'{label} temporaries ({n_arrays} x float64 window)',
                                                       n_arrays * FLOAT64_BYTES * window_cells),))


def _voltage_param(help_text, tip):
    return Param('voltage', 'float', 1.0, label='Voltage', unit='V', help=help_text, tip=tip)


# ---- driven terminals (electrostatics, piezoelectric plan decision 4) ----
_DRIVEN = (('waveform', TERMINAL_WAVEFORMS[1:]),)
_WAVEFORM_PARAMS = (
    Param('waveform', 'choice', 'none', label='Voltage waveform', choices=TERMINAL_WAVEFORMS,
          choice_labels=(('none', 'Constant'), ('ricker', 'Ricker wavelet'), ('gaussian', 'Gaussian pulse'),
                         ('cw', 'Continuous wave')),
          help='Time dependence of the held voltage, V(t) = voltage s(t) (the elastic_waves source waveforms): none = '
               'constant; ricker = Ricker wavelet peaking at frequency, value 1 at t0 = 4 / frequency; gaussian = sin '
               'carrier at frequency under a gaussian envelope of relative bandwidth (t0 = 4.5 / (pi bandwidth '
               'frequency)); cw = sin(2 pi f t) ramped in by 1 - exp(-t / (3 T)), T = 1 / f, steady after ~15 '
               'periods. Solved by superposition: one static solve '
               'per distinct waveform (its features at their voltage, every other voltage, source and charge 0) plus '
               'one of the constant part, after every operator change; each step then writes V, E, D = V_dc + sum '
               's_w(t + dt) V_w at the end time of the step (exact: the solve is linear), and charge_<feature> / the '
               'table follow the same sum. Time-dependent study only (stationary: s(0) ~ 0, validation warns); not '
               'with an electrolyte (validation error). In 2D an interdigital transducer is two terminals with the '
               'same waveform and opposite voltages.',
          tip='How the voltage varies in time; Constant holds it.\n\n- Ricker wavelet: one short pulse.\n- Gaussian '
              'pulse: a few oscillations at the frequency.\n- Continuous wave: a steady sine.'),
    Param('frequency', 'float', 1e8, label='Frequency', unit='Hz', min=0.0, min_exclusive=True, visible_if=_DRIVEN,
          help='Centre (gaussian, cw) or peak (ricker) frequency of the waveform.',
          tip='Frequency of the voltage waveform in hertz; 1e8 is 100 MHz.'),
    Param('bandwidth', 'float', 0.5, label='Relative bandwidth', unit='1', min=0.0, min_exclusive=True,
          visible_if=(('waveform', ('gaussian',)),),
          help='Spectral width / frequency of the gaussian waveform (1/e of the amplitude spectrum at f (1 +- '
               'bandwidth)).',
          tip='Width of the pulse spectrum relative to the frequency; 0.5 is typical.'),
)


# ---- Stern layer of voltage faces (electrostatics, igor 1.6) ----
_STERN_PARAMS = (
    Param('stern_thickness', 'float', 0.0, label='Stern layer thickness', unit='m', min=0.0,
          help='Thickness d_S of a compact (Stern / Helmholtz) layer between the electrode and the first cells: '
               '0 = none (V held on the face). > 0: the face is a Robin condition h = C_S = eps0 stern_permittivity / '
               'd_S [F/m^2], external = voltage, i.e. the layer in series with the half-cell conductance 2 eps0 '
               'eps_dc / dx, so V drops by sigma / C_S across it (sigma = D.n into the domain = the electrode charge '
               'per area; Gouy-Chapman-Stern: voltage = psi_d + sigma / C_S, psi_d the diffuse-layer potential at the '
               'face). The layer holds no ions and no cells (it is not drawn; the electrolyte begins at the face). '
               'Outer faces only (validation error with objects selected); charge_<feature> and the table charge are '
               'the D flux through it. Typical 0.2 to 0.5 nm.',
          tip='Thickness of the ion-free layer between the electrode and the electrolyte; 0 means none.\n\n'
              'Typical 0.3 nanometres. Only on domain edges.'),
    Param('stern_permittivity', 'float', 10.0, label='Stern layer permittivity', unit='1', min=1.0,
          help='Relative permittivity eps_S of the Stern layer (oriented water: 6 to 30); used only with '
               'stern_thickness > 0.',
          tip='Relative permittivity of the ion-free layer; water next to an electrode is often given 6 to 30.'),
)


def _stern_capacitance(values) -> float:
    """C_S [F/m^2] of voltage values (0 without a Stern layer or for the currents kind)."""
    return stern_capacitance(values.get('stern_thickness') or 0.0, values.get('stern_permittivity') or 1.0)


def _validate_electrostatic_voltage(vctx, physics_spec, feature_spec, values):
    """Driven waveform warning; errors: a Stern layer with objects selected (outer faces only) or in a 3D model."""
    issues = _validate_driven(vctx, physics_spec, feature_spec, values)
    if _stern_capacitance(values) > 0.0 and vctx.spec.is_3d:
        return issues + [Issue('error', feature_path(physics_spec, feature_spec),
                               'a Stern layer (stern_thickness > 0) needs a 2D model (it serves electrolytes, which 3D '
                               'does not have); set stern_thickness 0')]
    if _stern_capacitance(values) > 0.0:
        try:
            objects = [name for name in expand_entities(vctx.spec, feature_spec.selection) if name in vctx.shapes]
        except (ValueError, KeyError, TypeError):   # reported by the selection checks
            objects = []
        if objects:
            issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                f"a Stern layer (stern_thickness > 0) applies to outer faces only, not to the objects "
                                f"{', '.join(objects)}: select the domain edges, or set stern_thickness 0"))
    return issues


def _terminal_waveform(values):
    """TerminalWaveform of voltage / terminal values, None for a constant voltage (or the currents kind)."""
    if values.get('waveform', 'none') == 'none':
        return None
    return TerminalWaveform(values['waveform'], float(values['frequency']), float(values.get('bandwidth') or 0.5))


def _validate_driven(vctx, physics_spec, feature_spec, values):
    """Warning: a waveform outside a time_dependent study (held at voltage x s(0) ~ 0 V)."""
    if values.get('waveform', 'none') == 'none' or vctx.spec.study.type == 'time_dependent':
        return []
    return [Issue('warning', feature_path(physics_spec, feature_spec),
                  f"waveform {values['waveform']}: outside a time_dependent study the voltage is voltage x s(0), about "
                  '0 V; use waveform none for a constant voltage')]


def _is_driven(feature):
    """True for an enabled voltage / terminal feature with a waveform (raw params)."""
    return feature.enabled and feature.type in _ANCHOR_FEATURES \
        and feature.params.get('waveform', 'none') not in (None, 'none')


def _driven_issues(vctx, physics_spec, path):
    """Error: driven voltages / terminals together with an enabled electrolyte (no superposition)."""
    driven = any(_is_driven(feature) for feature in physics_spec.features)
    electrolyte = any(feature.enabled and feature.type == _ELECTROLYTE_FEATURE for feature in physics_spec.features)
    if driven and electrolyte:
        return [Issue('error', path, 'a voltage or terminal with a waveform cannot be combined with an electrolyte: '
                                     'the driven fields are superposed from static solves, which needs the linear '
                                     'operator without the ionic term')]
    return []


def _driven_waveform_count(physics_spec):
    """Distinct (waveform, frequency, bandwidth) entries of the enabled driven voltages / terminals (raw params)."""
    return len({(feature.params.get('waveform'), str(feature.params.get('frequency')),
                 str(feature.params.get('bandwidth')))
                for feature in physics_spec.features if _is_driven(feature)})


# ---- feature apply / result callbacks (worker thread, GPU) ----
def _apply_voltage(ctx, solver, feature_spec, values, selection):
    """Dirichlet on outer faces (Robin behind a Stern layer); object perimeters hold their boundary cells (times the
    waveform when driven), triangles their outline edges, tetrahedra their interface faces; returns the held region."""
    voltage = values['voltage']
    waveform = _terminal_waveform(values)
    stern = _stern_capacitance(values)
    for face in selection.outer_faces:
        if waveform is None:
            solver.set_face_bc(face, held_face_bc(voltage, stern))
        else:
            solver.set_driven_face_voltage(face, voltage, waveform, stern)
    if ctx.domain.grid_kind == 'triangles':   # outlines: Dirichlet on their edges (no waveform: validation)
        edges = ctx.boundary_edges(selection.objects) if selection.objects else None
        if edges is not None:
            solver.add_edge_potential(edges, voltage)
        return solver.add_held_region(FEMHeldRegion(feature_spec.name, float(voltage), tuple(selection.outer_faces),
                                                    edges=edges, stern_capacitance=stern))
    if ctx.domain.grid_kind == 'tetrahedra':   # interfaces: Dirichlet on their faces (no waveform / Stern: validation)
        surfaces = ctx.boundary_faces(selection.objects) if selection.objects else None
        if surfaces is not None:
            solver.add_surface_potential(surfaces, voltage)
        return solver.add_held_region(FEMHeldRegion3D(feature_spec.name, float(voltage),
                                                      tuple(selection.outer_faces), surfaces=surfaces))
    mask_window = None
    if selection.objects:
        held_mask = ctx.perimeter_mask(selection.objects)
        solver.add_fixed_potential(held_mask, voltage, waveform, **_held_outline(solver, selection.objects))
        mask_window = held_mask[solver.window].copy()
    return solver.add_held_region(HeldRegion(feature_spec.name, float(voltage), tuple(selection.outer_faces),
                                             mask_window, waveform, stern))


def _held_outline(solver, entities):
    """{'inside': drawn outline test} with solver_defaults.conformal_boundaries (cut-cell held edges), else {}."""
    locator = getattr(solver, 'conformal_locator', None)
    return {} if locator is None else {'inside': locator.inside(entities)}


def _apply_terminal(ctx, solver, feature_spec, values, selection):
    """Every selected cell held at the voltage (times the waveform when driven); returns the HeldRegion."""
    voltage = values['voltage']
    waveform = _terminal_waveform(values)
    held_mask = ctx.domain_mask(selection.entities)
    solver.add_fixed_potential(held_mask, voltage, waveform, **_held_outline(solver, selection.entities))
    if ctx.domain.grid_kind == 'triangles':
        return solver.add_held_region(FEMHeldRegion(feature_spec.name, float(voltage), (),
                                                    elements=solver.element_mask(held_mask)))
    if ctx.domain.grid_kind == 'tetrahedra':
        return solver.add_held_region(FEMHeldRegion3D(feature_spec.name, float(voltage), (),
                                                      elements=solver.element_mask(held_mask)))
    return solver.add_held_region(HeldRegion(feature_spec.name, float(voltage), (),
                                             held_mask[solver.window].copy(), waveform))


def _apply_source(ctx, solver, feature_spec, values, selection):
    """density (finite under the selection only) added to source_ext on the selected cells."""
    source_mask = ctx.domain_mask(selection.entities)
    solver.add_source(source_mask, ctx.eval_field(values['density'], mask=source_mask))
    return None


_apply_potential_probe, _potential_probe_result = make_point_probe('potential', 'V')


def _make_held_result(kind):
    """result callback of voltage / terminal: one table row [feature, voltage, current | charge] of the latest solve."""
    quantity = _KIND_NAMES[kind].held_quantity

    def held_result(session, solver, region, feature_spec, options):
        flux = solver.held_flux(region)
        return {'kind': 'table', 'columns': ['feature', 'voltage', quantity], 'units': ['', 'V', solver.flux_unit],
                'rows': [[region.name, region.voltage, flux]], 'voltage': region.voltage, quantity: flux}
    return held_result


# ---- scalar outputs (E5) ----
def _make_scalar_outputs(kind):
    """scalar_outputs: <prefix><feature> per enabled voltage / terminal and ionic_strength_ / debye_length_<feature>
    per enabled electrolyte (PotentialPhysics.scalars); invalid, repeated or field-named results are skipped
    (validation reports bad feature names at the feature)."""
    prefixes_by_type = {feature_type: (HELD_SCALAR_PREFIXES[kind],) for feature_type in _ANCHOR_FEATURES}
    prefixes_by_type[_ELECTROLYTE_FEATURE] = tuple(ELECTROLYTE_SCALAR_PREFIXES)
    field_names = frozenset(_KIND_NAMES[kind].renderer_class.field_infos)

    def scalar_outputs(settings_values, physics_spec):
        names = []
        for feature in physics_spec.features:
            if not feature.enabled or not is_identifier(feature.name):
                continue
            for prefix in prefixes_by_type.get(feature.type, ()):
                name = f'{prefix}{feature.name}'
                if name not in names and name not in field_names:
                    names.append(name)
        return tuple(names)
    return scalar_outputs


_SCALAR_OUTPUTS_HELP = {
    'currents': 'current_<feature> (A/m; geometry axisymmetric: A, the whole ring; 3D: A) per enabled voltage / '
                'terminal feature: the current per m of depth it feeds into the domain in the last solve, the current of its '
                'table result (same sign). Example derived '
                'value {"name": "I_anode", "expression": "currents_current_anode"} (operation mean of a constant is '
                'the value itself); a parameter sweep of a voltage then gives the I-V curve, voltage / current the '
                'resistance (Ohm m). A feature whose scalar would be named like an output field (a terminal named '
                'source: current_source) gets none. Under a mirror face of an EM physics of the model the value is '
                "the unfolded structure's: x 2 per mirror plane the voltage face or terminal touches (its image joins "
                'it; a face on the opposite side does not touch).',
    'electrostatics': 'charge_<feature> (C/m; geometry axisymmetric: C, the whole ring; 3D: C) per enabled voltage / '
                      'terminal feature: its free charge per m of depth (outward D flux) in the last solve, the charge '
                      'of its table result. Example derived value '
                      '{"name": "C_top", "expression": "electrostatics_charge_top / V_top"} = the capacitance per m '
                      'of depth (F/m) with every other electrode at 0 V (operation mean of a constant is the value '
                      'itself). Per enabled electrolyte feature also ionic_strength_<feature> (mol/L, 1/2 sum z_i^2 '
                      'c_i) and debye_length_<feature> (m, at the mean static eps_dc of its selected cells), '
                      'constants of the build. Under a mirror face of an EM physics of the model the charges are the '
                      "unfolded structure's: x 2 per mirror plane the voltage face or terminal touches.",
}


# ---- validation helpers (CPU only) ----
def _has_anchor(vctx, physics_spec):
    """True when an enabled voltage or terminal feature, or an enabled electrolyte with ions, exists, or an antiperiodic
    phi wrap fixes the level (no constant mode)."""
    return antiperiodic_wrap(vctx) or any(
        (feature.enabled and feature.type in _ANCHOR_FEATURES) or _electrolyte_anchors(vctx, feature)
        for feature in physics_spec.features)


def _material_issues(vctx, physics_spec, kind, path):
    """Material checks over the spec materials, reported once (by the first physics of the type): a non-zero xy
    component with xy^2 >= xx*yy (both kinds); currents: no DC conductivity anywhere."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    names = _KIND_NAMES[kind]
    if not reports_model_checks(vctx, physics_spec, names.type_name):
        return []
    parameters = vctx.parameters
    prefix = names.property_prefix
    issues, conducting = [], []
    selected_conductor = False
    for material in vctx.spec.materials:
        try:   # as the builder merges it
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # unknown library / malformed properties (reported at the material): may conduct
            conducting.append(True)
            selected_conductor = selected_conductor or bool(material.selection)
            continue
        has_conductivity = kind == 'currents' and conducts(properties, parameters, prefix)
        conducting.append(has_conductivity)
        if not material.selection:
            continue
        selected_conductor = selected_conductor or has_conductivity
        if evaluates_to(properties.get(f'{prefix}_xy'), parameters, lambda value: value != 0.0):
            indefinite = indefinite_cross_term_issue(material.name, properties, parameters, prefix,
                                                     KINDS[kind].schema[f'{prefix}_xx'])
            if indefinite is not None:
                issues.append(indefinite)

    # a conductor smaller than a preview cell can be missed by the raster: no error then
    if kind == 'currents' and conducting_fraction(vctx, conducting) == 0.0 and not selected_conductor:
        issues.append(Issue('error', path, 'no physical cell has a DC conductivity: assign a material with '
                                           'sigma_dc to the domain'))
    return issues


def _make_validate(kind):
    """validate(vctx, physics_spec, settings_values) of a potential type: anchor, precision, face / held-object
    conflicts, material checks and the axisymmetric rules."""
    names = _KIND_NAMES[kind]
    flux_feature = names.flux_feature
    boundary_features = frozenset({'voltage', flux_feature})
    held_features = frozenset({'voltage', 'terminal'})

    def validate_physics(vctx, physics_spec, settings_values):
        path = ('physics', physics_spec.name)
        issues = []
        order = azimuthal_order_of(settings_values)
        if not _has_anchor(vctx, physics_spec) and not order:   # an order m != 0 anchors by its m^2 / r^2 term
            alternatives = 'a Voltage boundary condition, a Terminal or an Electrolyte' if KINDS[kind].with_ions \
                else 'a Voltage boundary condition or a Terminal'
            issues.append(Issue('error', path,
                                f'a potential solve needs an anchor: add {alternatives} (a pure Neumann problem is '
                                'singular whatever the sources)'))
        issues.extend(cg_solver_issues(vctx, physics_spec, path))
        issues.extend(face_conflicts(vctx, physics_spec, boundary_features, path))
        issues.extend(mirror_face_issues(vctx, physics_spec, boundary_features, path, _MIRROR_FACE_CONSEQUENCE))
        issues.extend(held_object_conflicts(vctx, physics_spec, held_features, frozenset({flux_feature}), path))
        issues.extend(_material_issues(vctx, physics_spec, kind, path))
        if KINDS[kind].with_ions:
            issues.extend(_nonlinear_precision_issues(vctx, physics_spec, settings_values, path))
            issues.extend(_driven_issues(vctx, physics_spec, path))
        if vctx.spec.is_3d:
            issues.extend(_three_d_issues(vctx, physics_spec, settings_values, names, path))
        elif is_axisymmetric(settings_values):
            issues.extend(axisymmetric_issues(vctx, physics_spec, path, KINDS[kind].label,
                                              cross_prefix=names.property_prefix))
        issues.extend(azimuthal_order_issues(vctx, physics_spec, path, settings_values))
        if order:
            issues.extend(Issue('error', feature_path(physics_spec, feature),
                                f'a nonlinear electrolyte needs azimuthal_order 0: the Boltzmann charge of an '
                                f"order-{order} potential is no single order (use model 'linear')")
                          for feature in _nonlinear_electrolytes(vctx, physics_spec))
        issues.extend(mapped_domain_issues(vctx, physics_spec, frozenset({flux_feature}), names.property_prefix))
        if on_triangles(vctx, physics_spec):
            issues.extend(_triangle_issues(vctx, physics_spec, order, path))
        return issues
    return validate_physics


def _triangle_issues(vctx, physics_spec, order, path):
    """Rules on a triangle mesh (FEMPotentialPhysics, mesh plan T4): azimuthal_order 0, no voltage waveforms, linear
    electrolytes only."""
    issues = []
    if order:
        issues.append(Issue('error', path, 'azimuthal_order != 0 needs a tensor mesh (rectangular cells): the '
                                           'finite-element potential on triangles solves order 0 only'))
    issues.extend(Issue('error', feature_path(physics_spec, feature),
                        "a nonlinear electrolyte needs a tensor mesh (rectangular cells); on triangles use model "
                        "'linear'")
                  for feature in _nonlinear_electrolytes(vctx, physics_spec))
    issues.extend(Issue('error', feature_path(physics_spec, feature),
                        'a voltage waveform needs a tensor mesh (rectangular cells); on triangles use waveform none')
                  for feature in physics_spec.features if isinstance(feature.params, dict) and _is_driven(feature))
    return issues


def _three_d_issues(vctx, physics_spec, settings_values, names, path):
    """Rules of a 3D model (FEMPotentialPhysics3D, piezo_3d_plan stage 6): a tetrahedron mesh, planar geometry and
    azimuthal_order 0, no mirror planes, no voltage waveforms, no xy coefficient materials (Stern layers:
    _validate_electrostatic_voltage; electrolytes: their FeatureType grids)."""
    issues = []
    if not on_tetrahedra(vctx, physics_spec):
        issues.append(Issue('error', path, f'{names.label} in a 3D model runs on a tetrahedron mesh only (MeshSpec '
                                           "kind 'tetrahedra'): a 3D rectangular-cell (tensor) mesh is not supported; "
                                           'assign it a mesh of kind tetrahedra'))
    if is_axisymmetric(settings_values) or azimuthal_order_of(settings_values):
        issues.append(Issue('error', path, "geometry 'axisymmetric' and azimuthal_order need a 2D model (this model is "
                                           "3D: size_z is set); use geometry 'planar' and azimuthal_order 0"))
    mirrors = sorted(mirror_faces(vctx.spec))
    if mirrors:
        issues.append(Issue('error', path, f"the mirror face(s) {', '.join(mirrors)} of the model are not supported by "
                                           f'{names.label} in 3D (held currents / charges are not unfolded there): use '
                                           'a full domain'))
    issues.extend(Issue('error', feature_path(physics_spec, feature),
                        'a voltage waveform needs a 2D model on a tensor mesh; in 3D use waveform none')
                  for feature in physics_spec.features if isinstance(feature.params, dict) and _is_driven(feature))
    for material_name in cross_term_materials(vctx, names.property_prefix):
        issues.append(Issue('error', ('material', material_name),
                            f"material '{material_name}' sets {names.property_prefix}_xy, which {names.label} in 3D "
                            f"'{physics_spec.name}' does not support (diagonal {names.property_prefix}_xx / _yy / _zz "
                            'only): remove it'))
    return issues


# ---- physics factory callbacks ----
def _make_create(kind):
    def create(ctx, physics_spec, values):
        linear_solver = ctx.solvers(physics_spec.name)['linear']
        if ctx.domain.grid_kind == 'triangles':   # FEM (mesh plan T4): discretization order
            physics = FEMPotentialPhysics(ctx.domain, kind, order=element_order(values),
                                          axisymmetric=is_axisymmetric(values), linear_solver=linear_solver)
            physics.set_mirror_faces(mirror_faces(ctx.spec))
            return physics
        if ctx.domain.grid_kind == 'tetrahedra':   # FEM 3D (fem.diffusion3d): discretization order, diagonal tensors
            return FEMPotentialPhysics3D(ctx.domain, kind, order=element_order(values), linear_solver=linear_solver)
        if ctx.spec.is_3d:
            raise ValueError(f"{_KIND_NAMES[kind].label} in a 3D model needs a tetrahedron mesh (MeshSpec kind "
                             "'tetrahedra')")
        nonlinear = {name: values[name] for name in (param.name for param in _NONLINEAR_SETTINGS)} \
            if KINDS[kind].with_ions else {}
        physics = PotentialPhysics(ctx.domain, kind, axisymmetric=is_axisymmetric(values),
                                   azimuthal_order=azimuthal_order_of(values),
                                   linear_solver=linear_solver, **nonlinear)
        physics.set_mirror_faces(mirror_faces(ctx.spec))   # held fluxes of the unfolded structure
        return physics
    return create


def _make_memory_estimate(kind):
    """memory_estimate: PotentialFields2D (float32 full grid; + q_joule with currents, + rho_ion with electrostatics)
    plus the window DiffusionSystem (a set xy coefficient: 9-point pattern) and a cuDSS 'linear' factor
    (fem.diffusion.cudss_diffusion_estimate); element meshes: on the active elements; driven terminals: the basis solutions
    (DC + one per waveform), the superposition buffer and two zero sources (run items, first step); tetrahedra: the
    3D fields (+ E_z and the z flux) per element + the FEM 3D system; a 3D tensor mesh (refused by validate) nothing."""
    spec = KINDS[kind]
    n_fields = 7 + int(spec.with_joule) + int(spec.with_ions)

    def estimate(settings_values, mesh, context):
        cudss = cudss_back_end(context.solvers)
        refactorizes = _operator_changes(context, spec)
        if mesh.kind in ('triangles', 'tetrahedra'):   # FEMPotentialPhysics[3D]: the fields per element + FEM system
            triangles = mesh.kind == 'triangles'
            estimator = fem_diffusion_estimate if triangles else fem_diffusion_estimate_3d
            return estimator(mesh, element_order(settings_values), element_fields=n_fields + (0 if triangles else 2),
                             direct=linear_solver_direct(context), multigrid=linear_solver_multigrid(context),
                             cudss=cudss, refactorizes=refactorizes, active_fraction=_active_fraction(context, mesh, kind))
        if context.spec.is_3d:
            return MemoryEstimate()
        n_waveforms = _driven_waveform_count(context.physics_spec) if spec.with_ions else 0
        window_cells = window_cell_count(mesh)
        driven_bytes = ((n_waveforms + 2) * 5 + 2) * FLOAT32_BYTES * window_cells if n_waveforms else 0
        fields = MemoryEstimate.from_items(
            ((f'fields ({n_fields} x float32)', n_fields * FLOAT32_BYTES * int(mesh.n_cells)),),
            run_items=((f'driven terminal bases ({n_waveforms} waveforms + DC, superposition)', driven_bytes),))
        cross_term = context.material_sets(spec.coefficient_arrays[2])
        system = diffusion_system_estimate(window_cells, linear_solver_precision(context), cross_term=cross_term,
                                           mixed=linear_solver_mixed(context),
                                           multigrid=linear_solver_multigrid(context),
                                           direct=linear_solver_direct(context),
                                           azimuthal=azimuthal_order_of(settings_values) != 0,
                                           wrap=mesh.domain_map is not None and mesh.domain_map.angle_wrap != 'none',
                                           flux=True, dissipation=spec.with_joule)   # _update_derived_fields per solve
        parts = [fields, system]
        if cudss:   # DiffusionDirectSolver factorises the kept A_solve
            parts.append(cudss_diffusion_estimate(window_cells, (9 if cross_term else 5) * window_cells, 2, cudss,
                                                  input_copy=False, refactorizes=refactorizes))
        return combine(parts)
    return estimate


def _has_nonlinear_electrolyte(physics_spec) -> bool:
    """True when an enabled electrolyte feature has model 'nonlinear' (raw params)."""
    return any(feature.enabled and feature.type == _ELECTROLYTE_FEATURE and isinstance(feature.params, dict)
               and feature.params.get('model', 'linear') == 'nonlinear' for feature in physics_spec.features)


def _operator_changes(context, spec) -> bool:
    """Whether the run re-assembles the operator: a nonlinear electrolyte (Newton) or a coefficient of the kind that
    follows a state field (a piezo / coupling source leaves the build's factor)."""
    return _has_nonlinear_electrolyte(context.physics_spec) or material_follows_state(
        context, tuple(name for name, arrays in POTENTIAL_PROPERTY_ARRAYS.items()
                       if set(arrays) & set(spec.coefficient_arrays)))


def _active_fraction(context, mesh, kind) -> float:
    """Element share of the materials joining the FEM system: electric_currents those with sigma_dc > 0 (or an
    electrolyte: all), electrostatics all; fem.diffusion.painted_fraction (1 without material_elements)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    if kind != 'currents' or any(feature.enabled and feature.type == _ELECTROLYTE_FEATURE
                                 for feature in context.physics_spec.features):
        return 1.0
    prefix = _KIND_NAMES[kind].property_prefix
    names = []
    for material in context.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):   # reported by validate: may conduct
            names.append(material.name)
            continue
        if conducts(properties, context.parameters, prefix):
            names.append(material.name)
    return painted_fraction(mesh, names)


# where V is held, for the Jacobi-PCG condition estimate: voltages, terminals and the electrolyte's Debye screening
# (a mass term, counted as held)
_HELD_POTENTIAL = te.HeldValues(feature_types=frozenset({'voltage', 'terminal', _ELECTROLYTE_FEATURE}))


def _make_time_estimate(kind):
    """time_estimate: the DiffusionSystem recipe on the window, re-solved in a time-dependent study only when
    something changed (coupled); triangles / tetrahedra: at the Lagrange unknowns with their CSR entries per row
    (fem.diffusion.fem_time_system, uncalibrated); driven-terminal bases and nonlinear electrolyte Newton loops are listed as not estimated."""
    spec = KINDS[kind]

    def estimate(settings_values, mesh, context):
        operator_changes = _operator_changes(context, spec)
        if mesh.kind in ('triangles', 'tetrahedra'):
            unknowns, entries = fem_time_system(mesh, element_order(settings_values))
            return diffusion_time_estimate(context, unknowns, 3 if mesh.kind == 'tetrahedra' else 2,
                                           every_step=False, held=_HELD_POTENTIAL, operator_changes=operator_changes,
                                           entries_per_row=entries)
        if context.spec.is_3d:   # a 3D tensor mesh: refused by validate
            return te.TimeEstimate(not_estimated=('3D tensor mesh (not supported)',))
        linear = diffusion_time_estimate(context, context.window_cells(),
                                         cross_term=context.material_sets(spec.coefficient_arrays[2]),
                                         every_step=False, held=_HELD_POTENTIAL, operator_changes=operator_changes)
        missing = []
        if spec.with_ions and _driven_waveform_count(context.physics_spec):
            missing.append('driven terminal basis solves (one per waveform after each operator change)')
        if _has_nonlinear_electrolyte(context.physics_spec):
            missing.append('nonlinear electrolyte Newton iterations (one solve each; priced as one solve)')
        return te.combine((linear, te.TimeEstimate(not_estimated=tuple(missing)))) if missing else linear
    return estimate


# ---- type descriptions ----
_COMMON_DESCRIPTION = (
    'on the physical window (finite volumes, Jacobi-PCG on V), V and every output copied edge-extended into an EM PML '
    'ring (so a field-dependent material property sees no step there). '
    'Unreferenced outer faces are insulated ({insulated}), and a later boundary feature on the same face wins '
    '(validation warning). {materials} The solve needs an anchor: {anchor} (a pure Neumann problem is singular '
    'whatever the sources). Units are per unit z length: {units} (geometry axisymmetric: an (r, z) body of '
    'revolution about x_lo, held totals of the whole ring). Outputs: E_x / E_y and {flux} from the face fluxes averaged to the cell centres ({field_zero}; '
    '{flux} include the xy cross flux, E stays -grad V){joule} A stationary study iterates it with the couplings and the '
    'state-dependent material properties ({state_example}); in a time-dependent study it re-solves only when '
    '{coefficient}, a boundary condition, a held potential or a source changed ({cadence}). The instance is '
    'usually named {instance}.')
_PHYSICS_DESCRIPTIONS = {
    'currents': '2D electric currents per unit z length: quasi-static solve of -div(sigma_dc grad V) = s, s the '
                'volumetric current source [A/m^3] (current_source features plus the coupled slot current_source), ' +
                _COMMON_DESCRIPTION.format(
                    insulated='zero normal current',
                    materials='Material properties are unset unless a material or its library entry sets them: '
                              'cells without sigma_dc carry no current and keep their potential.',
                    anchor='a voltage or a terminal touching every conducting region',
                    units='terminal currents A/m, conductances S and capacitances F per m of depth, resistances '
                          'Ohm m',
                    flux='J_x / J_y', field_zero='E = 0 in non-conducting cells, E and J = 0 in held cells',
                    joule='; q_joule [W/m^3] = the dissipation of every face split between its two cells by their '
                          'half-cell resistances (outer Dirichlet faces to their cell; the sigma_dc_xy power of each '
                          '2x2 block by area), so its integral over the window equals the sum '
                          'over the voltage / terminal features of V * current, plus V times the current_source and '
                          'the inward current_density it feeds in (currents_heat_joule hands it to heat).',
                    state_example='sigma_dc of heat_T', coefficient='sigma_dc',
                    cadence='a sigma_dc(T) property is rewritten right after each heat step: leave every at 1 so '
                            'the currents re-solve on the next tick, before the Joule power goes to heat',
                    instance='currents'),
    'electrostatics': '2D electrostatics per unit z length: quasi-static solve of -div(eps0 eps_dc grad V) = '
                      'rho_free + rho_ion, rho_free the charge_density source [C/m^3], rho_ion the ionic space '
                      'charge of the electrolyte features (field rho_ion; linearised, or Boltzmann by damped Newton), '
                      + _COMMON_DESCRIPTION.format(
                          insulated='zero normal D',
                          materials='eps_dc is 1 (vacuum) unless a material or its library entry sets it, so every '
                                    'cell takes part in the solve.',
                          anchor='one voltage, terminal or electrolyte anywhere (every cell is a dielectric, so it '
                                 'reaches all of them)',
                          units='terminal charges C/m, capacitances F per m of depth (charge / voltage of a terminal '
                                'with the other one held at 0 V)',
                          flux='D_x / D_y', field_zero='E = D / (eps0 eps_dc) without eps_dc_xy, both 0 in held '
                                                       'cells', joule='.',
                          state_example='eps_dc of heat_T', coefficient='eps_dc',
                          cadence='a state-dependent eps_dc re-solves at the cadence of the physics it reads; with '
                                  'a voltage or terminal waveform every step instead writes the superposition of '
                                  'static basis solves at the end time of the step, no re-solve',
                          instance='electrostatics'),
}


_TRIANGLE_DESCRIPTION = (
    ' On a triangle mesh (MeshSpec kind triangles; 2D, planar or axisymmetric) finite elements of the discretization '
    'order (quadratic default): V is nodal, a voltage on objects holds their outline edges, a terminal every node of '
    'its elements, current_density / surface_charge on objects is an edge source on the outline (both sides), a probe '
    'reads the point\'s node; held currents / charges are the reactions of the un-eliminated system (exact Gauss '
    'balance); E, J / D and q_joule are element means (q_joule integrates to the stiffness energy, so to V * I) and '
    'every state slot holds one value per element (V at the centroid); electrostatics electrolytes with model linear '
    'only (a volumetric loss k (V - V_bulk) per element, none in terminal elements; rho_ion and c_ion_k from the '
    'element mean V, so the integral of rho_ion balances the held charges exactly; the inside of a voltage object '
    'solves to its voltage and carries ions when selected); no nonlinear electrolytes, waveforms or azimuthal_order.')
_TETRAHEDRA_TEXTS = {   # kind -> (coefficient prefix, flux feature, held quantity, unit, outputs, z flux, balance)
    'currents': ('sigma_dc', 'current_density', 'current', 'A', 'J_x / J_y and q_joule', 'J_z',
                 '; the integral of q_joule equals the sum of V * I'),
    'electrostatics': ('eps_dc', 'surface_charge', 'charge', 'C', 'D_x / D_y', 'D_z', ''),
}
_TETRAHEDRA_DESCRIPTION = {
    kind: (' 3D models (size_z set) run on a tetrahedron mesh only (MeshSpec kind tetrahedra, 3D Cartesian; a 3D tensor '
           'mesh is a validation error): Lagrange elements of the discretization order (affine tetrahedra) on the window '
           f'elements with the diagonal tensor {prefix}_xx / _yy / _zz ({prefix}_zz: set by the isotropic {prefix}; '
           f'{prefix}_xy is a validation error), V nodal; outer faces act on the window faces, a voltage on objects '
           f'holds their interface faces (Dirichlet), a terminal every node of its elements, {flux} on objects is a sheet '
           'source on the interface faces (both sides), a probe evaluates V at the point; held '
           f'{quantity}s are the reactions of the un-eliminated system in {unit} (whole totals, exact Gauss '
           f'balance{balance}); E, {outputs} are element means, plus the 3D-only output fields E_z, {flux_z} and '
           f'{prefix}_zz (not usable in expressions); fields are sampled on the uniform lattice of the mesh; Automatic '
           'linear solver: sparse_direct (cuDSS on the gpu backend with nvmath-python, else host SuperLU). Not in 3D: '
           'electrolytes, voltage waveforms, Stern layers, geometry axisymmetric / azimuthal_order and mirror faces of the model (validation errors).')
    for kind, (prefix, flux, quantity, unit, outputs, flux_z, balance) in _TETRAHEDRA_TEXTS.items()}


def _make_mesh_method(kind):
    """mesh_method: how the type solves on mesh_kind (finite volumes, triangles, tetrahedra) and, as problems, the
    element-mesh and 3D rules validation reports (3D tensor mesh, electrolytes, waveforms, Stern layers, mirrors)."""
    names = _KIND_NAMES[kind]

    def mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
        order = 'quadratic' if element_order(settings_values) == 2 else 'linear'
        three_d = spec.is_3d
        text = {'triangles': f'Finite elements on triangles ({order} elements).',
                'tetrahedra': f'Finite elements on tetrahedra ({order} elements, the material values given per axis '
                              'x, y and z).'}.get(mesh_kind, 'Finite volumes on rectangular cells.')
        problems = []
        if three_d and mesh_kind != 'tetrahedra':
            problems.append(f'{names.label} in 3D needs a tetrahedron mesh: change the mesh kind to tetrahedra.')
        if not (three_d or mesh_kind == 'triangles'):
            return text, problems
        features = [feature for feature in physics_spec.features
                    if feature.enabled and isinstance(feature.params, dict)]
        electrolytes = [feature for feature in features if feature.type == _ELECTROLYTE_FEATURE]
        if three_d and electrolytes:
            problems.append('Electrolytes need a 2D model.')
        elif any(feature.params.get('model') == 'nonlinear' for feature in electrolytes):
            problems.append('A nonlinear electrolyte needs rectangular cells: use the linear model on triangles.')
        if any(feature.type in _ANCHOR_FEATURES and feature.params.get('waveform', 'none') not in (None, 'none')
               for feature in features):
            problems.append('Voltage waveforms need a 2D model on rectangular cells: use a constant voltage.')
        if three_d and any(feature.type == 'voltage' and feature.params.get('stern_thickness') not in (None, 0, 0.0)
                           for feature in features):
            problems.append('A Stern layer needs a 2D model.')
        if three_d and mirror_faces(spec):
            problems.append('Mirror faces of the model are not supported in 3D: use a full domain.')
        return text, problems
    return mesh_method


# ---- type factory ----
def _potential_physics_type(kind):
    """PhysicsType of one potential kind ('currents' -> electric_currents, 'electrostatics' -> electrostatics)."""
    names = _KIND_NAMES[kind]
    spec = KINDS[kind]
    quantity, coefficient = names.held_quantity, names.coefficient_text
    held_result = _make_held_result(kind)
    if kind == 'currents':
        flux_feature_type = FeatureType(
            'current_density', 'Current density', 'boundary_condition', 'boundary',
            params=(Param('current_density', 'float', 1e4, label='Inward current density', unit='A/m^2',
                          help="inward: > 0 injects current into the domain (outer faces) or into the object's "
                               'boundary cells as value * perimeter length / cell area [A/m^3], < 0 extracts it; '
                               'does not anchor the solve.',
                          tip='Current per area flowing into the domain or object; negative values draw current '
                              'out.\n\nIt does not set the voltage level, so a voltage or terminal is still needed.'),),
            summary='Feeds a given current per area in through the selected domain edges or object boundaries. The '
                    'current must leave through a voltage or terminal.',
            description='Prescribed normal current density, > 0 into the domain or the object. On outer faces a '
                        'Neumann condition (the normal current through the face); on object perimeters it feeds the '
                        'boundary cells inside the object with current_density * perimeter length / cell area '
                        '[A/m^3] (each exposed cell face weighted by its normal component, so slanted and curved '
                        'perimeters get their true length to ~2 %). Does not anchor the solve: the current must leave '
                        'through a voltage or terminal. Polar / spherical domains weight each exposed face by its '
                        'metric area. With solver_defaults.conformal_boundaries (2D: Cartesian, (r, z), polar, '
                        'spherical) each cut cell feeds current_density * its chord of the drawn outline into the '
                        'object cell owning it (exact for straight edges).',
            apply=make_apply_inward_flux('current_density'), grids=_FEM_GRIDS)
    else:
        flux_feature_type = FeatureType(
            'surface_charge', 'Surface charge', 'boundary_condition', 'boundary',
            params=(Param('surface_charge', 'float', 1e-6, label='Surface charge', unit='C/m^2',
                          help='charge on the selected outer faces (the D flux into the domain equals it: a charged '
                               'face with nothing beyond) or on the exposed faces of the objects (placed in their '
                               'boundary cells as a density); > 0 raises V. Does not anchor the solve.',
                          tip='Charge per area on the selected domain edges or object boundaries; positive charge '
                              'raises the potential.\n\n'
                              'It does not set the potential level, so a voltage or terminal is still needed.'),),
            summary='Places a fixed charge per area on the selected domain edges or object boundaries.',
            description='Prescribed surface charge. On outer faces a Neumann condition D.n = surface_charge into the '
                        'domain (a charged sheet with nothing beyond the face); on object perimeters the charge is '
                        'placed in the boundary cells inside the object as surface_charge * perimeter length / '
                        'cell area [C/m^3] (each exposed cell face weighted by its normal component, so slanted and '
                        'curved perimeters get their true length to ~2 %; polar / spherical domains: the metric face '
                        'areas). Does not anchor the solve. With solver_defaults.conformal_boundaries (2D: Cartesian, '
                        '(r, z), polar, spherical) each cut cell places '
                        'surface_charge * its chord of the drawn outline in the object cell owning it.',
            apply=make_apply_inward_flux('surface_charge'), grids=_FEM_GRIDS)
    source_feature_type = FeatureType(
        'current_source' if kind == 'currents' else 'charge_density',
        'Current source' if kind == 'currents' else 'Charge density', 'domain_condition', 'domain',
        params=(Param('density', 'float', 1e12 if kind == 'currents' else 1e-3,
                      label='Injected current per volume' if kind == 'currents' else 'Charge density',
                      unit='A/m^3' if kind == 'currents' else 'C/m^3', field_dependent=True,
                      help=_SOURCE_PARAM_HELP[kind], tip=_SOURCE_PARAM_TIPS[kind]),),
        summary=_SOURCE_SUMMARIES[kind], description=_SOURCE_DESCRIPTION[kind], apply=_apply_source,
        grids=_FEM_GRIDS)
    driven_params = _WAVEFORM_PARAMS if spec.with_ions else ()   # electrostatics only
    driven_validate = _validate_driven if spec.with_ions else None
    stern_params = _STERN_PARAMS if spec.with_ions else ()   # electrostatics only
    stern_text = (' stern_thickness > 0 (electrostatics, outer faces): a compact Stern layer of capacitance eps0 '
                  'stern_permittivity / stern_thickness per area between the face voltage and the first cells (Robin '
                  'face).') if spec.with_ions else ''
    features = (
        FeatureType('voltage', 'Voltage', 'boundary_condition', 'boundary',
                    params=(_voltage_param('Held at the selected outer faces (on the face itself) or in the boundary '
                                           'cells inside the selected objects.',
                                           'Potential held on the selected domain edges, or in the boundary cells '
                                           'just inside the selected objects.'),) + driven_params + stern_params,
                    validate=_validate_electrostatic_voltage if spec.with_ions else None,
                    result_kind='table',
                    summary=_VOLTAGE_SUMMARIES[kind],
                    description='Fixed potential (Dirichlet). On outer faces V is imposed at the face (half-cell '
                                f'conductance 2 {coefficient} / dx); on object perimeters the boundary cells inside '
                                'the object are held at V, imposed on their faces toward the free cells as on an '
                                'outer face (the window edge is no perimeter: select the outer face there). '
                                'Unreferenced outer faces are insulated, and a later boundary feature on the '
                                f'same face wins (validation warning). Anchors the solve.{stern_text} '
                                f'{_HELD_RESULT_TEXT[kind]}',
                    apply=_apply_voltage, result=held_result, grids=_FEM_GRIDS),
        FeatureType('terminal', 'Terminal', 'domain_condition', 'domain',
                    params=(_voltage_param('Voltage every selected cell is held at for the whole run (times the '
                                           'waveform when one is set).',
                                           'Voltage the selected domains are held at for the whole run.'),)
                    + driven_params,
                    validate=driven_validate,
                    result_kind='table',
                    summary=_TERMINAL_SUMMARIES[kind],
                    description='Holds every selected cell at the voltage for the whole run (the cells leave the '
                                f'solve; {quantity} flows in or out freely); the voltage sits on their faces toward '
                                f'the free cells (half-cell conductance 2 {coefficient} / dx of the free neighbour, '
                                "whatever the held cells' own coefficient); anchors the solve. "
                                f'{_TERMINAL_BODY_TEXT[kind]} {_HELD_RESULT_TEXT[kind]}',
                    apply=_apply_terminal, result=held_result, grids=_FEM_GRIDS),
        flux_feature_type,
        source_feature_type,
        FeatureType('potential_probe', 'Potential probe', 'probe', 'point',
                    object_kinds=('point',), max_entities=1, result_kind='timeseries',
                    summary='Records the electric potential at a point after every solve.',
                    description=_PROBE_DESCRIPTION,
                    apply=_apply_potential_probe, result=_potential_probe_result, validate=validate_points_inside,
                    grids=_FEM_GRIDS),
    )
    if spec.with_ions:
        features += (FeatureType(
            _ELECTROLYTE_FEATURE, 'Electrolyte', 'domain_condition', 'domain', params=_ELECTROLYTE_PARAMS,
            summary='Fills the selected domains with a salt solution whose mobile ions screen the electric field '
                    'within a short distance (the Debye length). The linear model suits potentials well below 25 '
                    'millivolts, the nonlinear one larger potentials; it also fixes the potential level, so no voltage '
                    'is needed.',
            description=_ELECTROLYTE_DESCRIPTION, apply=_apply_electrolyte, validate=_validate_electrolyte,
            memory_estimate=_electrolyte_memory_estimate, grids=_BOTH_GRIDS),)
    field_infos = tuple(names.renderer_class.field_infos.values())
    state_infos = tuple(info for info in field_infos if info.name not in ION_CONCENTRATION_FIELDS)   # not in state()
    return PhysicsType(
        name=names.type_name,
        label=names.label,
        default_instance_name=names.instance_name,
        family='',
        group='electrical',
        coordinate_grids=('cartesian', 'polar'),
        # 1D (symmetry 2.20): a 2D Domain one cell across, Automatic linear solver 'banded'; 3D: tetrahedra only
        dimensions=(1, 2, 3),
        spaces_2d=('planar', 'cross_section', 'axisymmetric'),
        space_settings=AXISYMMETRIC_SPACE_SETTINGS,
        description=_PHYSICS_DESCRIPTIONS[kind] + _TRIANGLE_DESCRIPTION + _TETRAHEDRA_DESCRIPTION[kind],
        summary=_PHYSICS_SUMMARIES[kind],
        settings=_SETTINGS[kind],
        features=features,
        # triangles: FEMPotentialPhysics (solver_fem.py, mesh plan T4); tetrahedra: FEMPotentialPhysics3D (3D models)
        grids=('tensor', 'triangles', 'tetrahedra'),
        mesh_method=_make_mesh_method(kind),
        material_properties=_MATERIAL_PROPERTIES[kind],
        material_schema=spec.schema,
        optional_material_schema=dict(OPTIONAL_POCKELS_SCHEMA) if kind == 'electrostatics' else {},
        state_fields=state_infos,
        output_fields=field_infos + tuple(THREE_D_FIELD_INFOS[kind].values()),   # E_z, z flux, zz: 3D only
        source_slots={spec.slot: _SLOT_DOCS[kind]},
        supported_studies=('stationary', 'time_dependent'),
        default_study={'type': 'stationary'},
        uses_dispersion=False,
        solver_roles={'linear': CG_DIFFUSION_LINEAR_ROLE},
        backends=('gpu', 'cpu'),   # numpy twins of the system / field kernels (solver_plan S6 stage 2)
        conformal_boundaries=True,   # cut cells (held edges, insulated walls, perimeter fluxes; 2D)
        conformal_coordinates=('cartesian', 'polar', 'spherical'),   # cut cells on the logical (r, angle) cells too
        create=_make_create(kind),
        validate=_make_validate(kind),
        memory_estimate=_make_memory_estimate(kind),
        time_estimate=_make_time_estimate(kind),
        scalar_outputs=_make_scalar_outputs(kind),
        scalar_outputs_help=_SCALAR_OUTPUTS_HELP[kind],
    )


ELECTRIC_CURRENTS = _potential_physics_type('currents')
ELECTROSTATICS = _potential_physics_type('electrostatics')


def _selftest_cases():
    """The potential SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_axisymmetric import AZIMUTHAL_SELFTEST_CASES
    from .selftests_coordinates import COORDINATE_SELFTEST_CASES
    from .selftests_electrolyte import ELECTROLYTE_SELFTEST_CASES
    from .selftests_fem import FEM_SELFTEST_CASES
    from .selftests_fem3d import FEM_3D_SELFTEST_CASES
    from .selftests_mirror import MIRROR_SELFTEST_CASES
    return (SELFTEST_CASES + AZIMUTHAL_SELFTEST_CASES + COORDINATE_SELFTEST_CASES + FEM_SELFTEST_CASES
            + FEM_3D_SELFTEST_CASES + ELECTROLYTE_SELFTEST_CASES + MIRROR_SELFTEST_CASES)


def register(registry):
    """Both potential types, then the 'thermo_optic_heater' template, then the selftest loader."""
    registry.register_physics(ELECTRIC_CURRENTS)
    registry.register_physics(ELECTROSTATICS)
    registry.register_template(
        'thermo_optic_heater', THERMO_OPTIC_HEATER_LABEL, THERMO_OPTIC_HEATER_DESCRIPTION, thermo_optic_heater_spec,
        summary='Electric heater beside a silicon waveguide: a current heats a gold strip, the heat raises the '
                "waveguide's refractive index, and the guided light gains a phase of about 0.1 rad. Steady state.")
    registry.register_selftests(_selftest_cases)

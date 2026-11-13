"""Analytic self-tests of em_fdfd_ez / em_fdfd_hz (line-current Hankel field, far-field power, Fresnel transmission,
slab-waveguide mode ports over a frequency_domain study, pec_domain plane, Drude Mie cylinder) and the 'fd_joule_heating' template; no physics imports."""
import math

import numpy as np

from ...core.constants import C0, EPS0, ETA0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (PERIODIC_Y, available_host_direct, cell_centre, float_list, glass_half_space,
                                       probe_phasor, single_value, skip_result, skip_without_host_direct,
                                       solver_role, vertical_line, with_direct_node, worst_item)
from ...model.spec import (OUTER_FACES, CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec,
                           MaterialSpec, MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, RefinementSpec, SolverSpec,
                           StudySpec)
from .selftests import (_PROBE_FEATURE, _WALL_AMPLITUDE, _WALL_RUNS, _WALL_WAVELENGTH, WALL_REFLECTION_TEXT,
                        _hankel0_modulus, normalized_result, wall_reflection_result, wall_reflection_spec,
                        wall_run_name)
from .selftests_bloch import _TFSF_FLUX_LINES, _oblique_tfsf_spec
from .selftests_hz import (_DRUDE_CELL, _DRUDE_EPS_INF, _DRUDE_GAMMA, _DRUDE_INNER_PREFIX, _DRUDE_OMEGA_P,
                           _DRUDE_OUTER_PREFIX, _DRUDE_RADIUS, GRADED_FRESNEL_TEXT, _cylinder, _cylinder_efficiencies,
                           _drude_relative_index, _flux_box, _net_outflow, graded_interface_mesh, require_graded)
from .selftests_mode_ports import (_DOWNSTREAM, _SILICON_EPS_R, _UPSTREAM, _mode_spectra, _slab_fundamental_index,
                                   slab_waveguide_spec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPES = {'ez': 'em_fdfd_ez', 'hz': 'em_fdfd_hz'}
_POLARIZATIONS = ('ez', 'hz')
_SOURCE = 'source'
_FLUX = 'transmitted'

# em_fdfd_line_current_hankel: 4 x 4 um all-CPML box, uniform 25 nm mesh (40 cells per vacuum wavelength), window cell
# k spanning [25 k, 25 (k + 1)] nm. The source and every probe sit on cell centres ((k + 0.5) 25 nm), so each
# source-probe distance is exact: source cell (80, 80); probes at cell offsets 20, 36, 56 along +x (r = 0.5, 0.9,
# 1.4 um) and (15, 15), (30, 30), (42, 42) along the diagonal (r = 0.53, 1.06, 1.48 um), the closest to the PML 23.5
# cells (0.59 um) from it. Unknowns (160 + 30)^2 = 36,100.
# One run per polarisation: em_fdfd_ez with a filament current (E_z probes), em_fdfd_hz with a magnetic line current
# (H_z probes). The FDFD equations (harmonic.py header) give E_z = +(omega mu0 I / 4) H0^(2)(k r) and H_z = +(omega
# eps0 K_m / 4) H0^(2)(k r) in vacuum, so the probe phase must equal arg H0^(2)(k r) (a sign flip of the rhs: +-pi).
_HANKEL_WAVELENGTH = 1e-6                 # m, vacuum
_HANKEL_CURRENT = 1e-3                    # A, em_fdfd_ez filament current
_HANKEL_MAGNETIC_CURRENT = 1.0            # V, em_fdfd_hz magnetic line current
_HANKEL_CELL = 25e-9                      # m
_HANKEL_SIZE = 4e-6                       # m, 160 cells
_HANKEL_SOURCE_CELL = (80, 80)
_HANKEL_PROBE_OFFSETS = ((20, 0), (36, 0), (56, 0), (15, 15), (30, 30), (42, 42))   # cells from the source cell
_HANKEL_TOLERANCE = 0.03                  # relative |field|, worst probe
_HANKEL_PHASE_TOLERANCE = 0.03            # rad, worst probe

# Far-field power (checked by em_fdfd_line_current_hankel, whose ez run is the far-field ez box with the Hankel probes;
# run _HANKEL_DIPOLE_RUN = the hz dipole box): the Hankel box and mesh, vacuum, wavelength 1 um, source in cell (80, 80):
# em_fdfd_ez the 1 mA filament current, em_fdfd_hz a 1 mA in-plane electric dipole at 30 deg. The far-field box (1.0125, 3.0125) um
# covers cells [40, 121), so its closed contour runs through the centres of columns / rows 40 and 121 (1.0125 / 3.0375
# um). The closed flux box: lines in columns / rows 30 and 130 spanning cells [31, 131) measure through the faces at
# 0.775 / 3.275 um, 31 cells from the PML. Unknowns 36,100 per run.
_FAR_FIELD = 'far_field'
_HANKEL_DIPOLE_RUN = 'hz_dipole'
_FAR_FIELD_BOX = (1.0125e-6, 1.0125e-6, 3.0125e-6, 3.0125e-6)   # m (x0, y0, x1, y1), cell centres 40 and 120
_FAR_FIELD_FLUX_PREFIX = 'flux_'
_FAR_FIELD_FLUX_LINES = (0.7625e-6, 3.2625e-6)  # m, line positions (cell centres 30 and 130)
_FAR_FIELD_FLUX_SPAN = (0.7875e-6, 3.2625e-6)   # m, cells [31, 131)
_FAR_FIELD_N_ANGLES = 360
_FAR_FIELD_DIPOLE = {'excitation': 'electric_dipole', 'dipole_moment': 1e-3, 'orientation': 30.0, 'phase': 0.0}
_FAR_FIELD_POWER_TOLERANCE = 0.02         # relative, angle integral vs flux box
_FAR_FIELD_SHAPE_TOLERANCE = 0.02         # absolute, peak-normalised pattern
# em_fdfd_far_field_power_graded: the same boxes on a graded mesh (solver_plan S7.1 lift): base 25 nm, a centred core
# square refined to 12.5 nm (both axes) with a margin, so the far-field contour rows / columns cross graded cells while
# its four sample lines and the flux lines stay in base cells.
_GRADED_FAR_FIELD_CORE = (1.8e-6, 2.2e-6)  # m, refined square (both axes)
_GRADED_FAR_FIELD_CELL = 12.5e-9           # m
_GRADED_FAR_FIELD_MARGIN = 0.2e-6          # m

# em_fdfd_fresnel_transmission: y-periodic strip 5 x 0.1 um, CPML along x, uniform 10 nm mesh (50 cells per wavelength
# in the glass), window cell k spanning [10 k, 10 (k + 1)] nm. Vertical lines at cell centres span y 0..0.1 um, all 10
# rows of the periodic axis: source x = 1.005 um (cell 100; the H_z sheet on its hi face at 1.01 um), flux monitor
# x = 3.505 um (cell 350, measuring face 3.51 um). The glass runs fill x >= 2.5 um (cells 250.., continued into the
# x_hi PML through the clamped centres) with eps_r = 4; the medium runs fill every cell (PML included) with it, so the
# source sits in glass and must send n A^2 L / (2 eta0). Unknowns 530 x 10 = 5,300 per run.
_FRESNEL_MEDIA = ('vacuum', 'glass', 'medium')
_FRESNEL_WAVELENGTH = 1e-6                # m, vacuum
_FRESNEL_SIZE = (5e-6, 0.1e-6)            # m
_FRESNEL_CELL = 10e-9                     # m
_FRESNEL_SOURCE_X = 1.005e-6              # m
_FRESNEL_FLUX_X = 3.505e-6                # m
_FRESNEL_INTERFACE = 2.5e-6               # m
_FRESNEL_AMPLITUDE = 1.0                  # V/m radiated to each side of the line
_GLASS_EPS_R = 4.0
_GLASS_INDEX = math.sqrt(_GLASS_EPS_R)
_FRESNEL_TRANSMISSION = 4.0 * _GLASS_INDEX / (_GLASS_INDEX + 1.0) ** 2   # 8/9
_FRESNEL_TOLERANCE = 0.01                 # absolute, worst polarisation
_FRESNEL_POWER_TOLERANCE = 0.01           # relative, vacuum / medium flux vs the source calibration

# em_fdfd_mode_port_slab: the A3 slab layout of selftests_mode_ports (4 x 2 um all-CPML box, uniform 20 nm mesh, a 0.28
# um silicon slab in silica along the whole x range and into the x PML; vertical lines at cell centres x = 0.51 / 1.01 /
# 3.01 um spanning the window height, so every port solve continues through the y PML: upstream monitor, mode source
# (total field from cell 50 on), downstream monitor) in em_fdfd_ez and em_fdfd_hz. frequency_domain study solving 1.5
# and 1.6 um in that order: the monitors pick mode 1 at 1.5 um and track it by overlap (two wavelengths keep the pick
# and one tracking step). Unknowns (200 + 30) x (100 + 30) = 29,900 per run.
_SLAB_PORT_SOURCE = {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}
_SLAB_STUDY = {'wavelength_start': 1.5e-6, 'wavelength_stop': 1.6e-6, 'n_wavelengths': 2}
_SLAB_WAVELENGTHS = (1.5e-6, 1.6e-6)      # m, ascending as the mode monitor results list them
_SLAB_FORWARD_TOLERANCE = 0.01            # absolute power fraction
_SLAB_LEAK_TOLERANCE = 2e-3               # power fraction of a direction that should carry nothing
_SLAB_INDEX_TOLERANCE = 0.01              # relative n_eff

# em_fdfd_mie_drude_cylinder: the Drude cylinder of selftests_hz (r = 50 nm, 2.5 nm mesh) in a 0.4 x 0.4 um all-CPML box
# (160 + 30 cells per axis, 36,100 unknowns: the cusolver_qr workspace 64 B n^3 ~ 0.44 GB stays below its warning), window
# cell k spanning [2.5 k, 2.5 (k + 1)] nm. The TFSF box (0.1, 0.3) um covers cells [40, 121) (x1 / y1 on a cell edge take
# one extra cell). Flux lines sit at cell centres and measure through the hi face of their cell: the outer box (scattered
# field) has lines in columns / rows 29 and 130 spanning cells [30, 131), 10 cells outside the TFSF box; the inner box
# (total field) lines in columns / rows 49 and 109 spanning cells [50, 110), 10 / 11 cells inside it, around the cylinder
# cells [60, 100). One stationary run per polarisation and wavelength; incident intensity A^2 / (2 eta0) (no reference).
_FD_DRUDE_SIZE = 0.4e-6                   # m
_FD_DRUDE_CENTER = 0.2e-6                 # m, cylinder axis at (center, center)
_FD_DRUDE_TFSF_BOX = (0.1e-6, 0.1e-6, 0.3e-6, 0.3e-6)
_FD_DRUDE_AMPLITUDE = 1.0                 # V/m, incident E_z (ez) / E_y (hz)
_FD_DRUDE_OUTER_LINES = (73.75e-9, 326.25e-9)   # m, cell centres 29 and 130
_FD_DRUDE_OUTER_SPAN = (76.25e-9, 326.25e-9)    # m, cells [30, 131)
_FD_DRUDE_INNER_LINES = (123.75e-9, 273.75e-9)  # m, cell centres 49 and 109
_FD_DRUDE_INNER_SPAN = (126.25e-9, 273.75e-9)   # m, cells [50, 110)
# m, off the H_z dipole resonance near 222 nm; the worst items sit at 0.6 um (1.2 um dropped: no extra coverage)
_FD_DRUDE_WAVELENGTHS = (0.6e-6, 0.9e-6)
_FD_ABSORBED_VALUE = 'absorbed_line_power'     # derived value [W/m]: q_abs integrated over the cylinder
_FD_DRUDE_SCATTERING_TOLERANCE = 0.10     # relative, worst wavelength
_FD_DRUDE_ABSORPTION_TOLERANCE = 0.15     # relative, worst wavelength
_FD_DRUDE_BALANCE_TOLERANCE = 1e-3        # relative, inner-box inward flux vs integrated q_abs

# em_fdfd_tfsf_graded_leakage (solver_plan S7.1): an empty vacuum TFSF box must stay transparent on graded meshes.
# 'ez' / 'hz': 3 x 3 um all-CPML box, wavelength 1 um, mode 'nonuniform' base 50 nm, the TFSF box (1, 2) um and a
# refinement rectangle x 1.05..1.45, y 1.3..1.7 um refined to 20 nm with a 0.1 um margin: the lo-x edge (x = 1 um) lies
# in the graded margin (start pair of unequal widths), the x march crosses graded columns, the y edges run along graded
# columns and the x edges along graded rows. Probes of the out-of-plane field 0.5 um outside each side (scattered field)
# and one at the box centre (total field = incident). 'oblique_ez' / 'oblique_hz': the em_fdfd_oblique_tfsf strip
# (Bloch ky, full-period box over x 1-2 um, 10 nm) graded along x only by the vertical refinement line of
# selftests_hz.graded_interface_mesh at x = 1.5 um; its upstream / downstream flux lines over the inside one.
_LEAK_SIZE = 3e-6                         # m, square
_LEAK_CELL = 50e-9                        # m, base cell
_LEAK_TFSF_BOX = (1e-6, 1e-6, 2e-6, 2e-6)
_LEAK_REFINEMENT = (1.05e-6, 1.3e-6, 1.45e-6, 1.7e-6)   # m, (x0, y0, x1, y1)
_LEAK_REFINED_CELL = 20e-9                # m
_LEAK_MARGIN = 0.1e-6                     # m
_LEAK_OUTSIDE_PROBES = ((0.5e-6, 1.5e-6), (2.5e-6, 1.5e-6), (1.5e-6, 0.5e-6), (1.5e-6, 2.5e-6))
_LEAK_INSIDE_PROBE = (1.5e-6, 1.5e-6)
_LEAK_OBLIQUE_X = 1.5e-6                  # m, vertical refinement line inside the oblique box
_LEAK_FIELD_TOLERANCE = 1e-9              # scattered |field| / incident |field| (float64 solve round-off)
_LEAK_FLUX_TOLERANCE = 1e-12              # scattered flux / inside flux (quadratic in the leaked field)

# em_fdfd_line_block: a smaller copy of the em_fdfd_line_current_hankel box (no periodic axis: the cheaper block axis)
# and a shorter em_fdfd_fresnel_transmission glass strip (periodic y: blocks along y, the wrap inside each block), each
# solved by the Automatic sparse_direct and by line_block 'linear' nodes in double and mixed precision. The check
# compares the two solvers on one model only, so no closed form constrains the sizes: Hankel box 2 x 2 um at 25 nm
# ((80 + 30)^2 = 12,100 unknowns), source cell (40, 40), probes at most 25 cells from it; glass strip 2 x 0.1 um at
# 10 nm (230 x 10 unknowns), source cell 50, interface at 1 um, flux line cell 150.
_LINE_BLOCK_HANKEL_SIZE = 2e-6                    # m, 80 cells
_LINE_BLOCK_SOURCE_CELL = (40, 40)
_LINE_BLOCK_PROBE_OFFSETS = ((10, 0), (18, 0), (25, 0), (7, 7), (12, 12), (17, 17))   # cells from the source cell
_LINE_BLOCK_STRIP = {'length': 2e-6, 'source_x': 0.505e-6, 'flux_x': 1.505e-6, 'interface': 1e-6}   # m
_LINE_BLOCK_PRECISIONS = ('double', 'mixed')
_LINE_BLOCK_TOLERANCES = {'double': 1e-8, 'mixed': 1e-5}   # relative to the largest |value| of the run
# em_fdfd_line_block_wavelengths (Part II stage 3): the Hankel ez box and the Fresnel ez / hz glass strips in a
# frequency_domain study of _BATCH_STUDY wavelengths (below fdfd_line_block.MAX_BATCH_WAVELENGTHS = 8: one batch),
# Automatic vs line_block double / mixed, the tolerances above per wavelength. Two wavelengths already make a batch of
# several shifts (a wrong shift index solves the other wavelength); 4 took ~20 s, the cost goes with the count.
_BATCH_STUDY = {'wavelength_start': 0.95e-6, 'wavelength_stop': 1.05e-6, 'n_wavelengths': 2}
_BATCH_MODELS = ('hankel_ez', 'fresnel_ez', 'fresnel_hz')
# em_fdfd_cudss: the same models and wavelengths with a sparse_direct node per cuDSS back end vs the Automatic SuperLU
_CUDSS_BACK_ENDS = ('cudss', 'cudss_mixed')
_DIRECT_NODE = 'sparse_direct_linear'   # the 'linear' sparse_direct node: scipy_splu (reference) or a cuDSS back end
_CUDSS_TOLERANCE = 1e-8   # relative to the largest |SuperLU value| (measured: cudss ~4e-14, cudss_mixed <= 5e-10)
# em_fdfd_host_direct (cpu backend): per host direct back end of every installed library (python-mumps: mumps /
# mumps_mixed, pydiso: pardiso / pardiso_mixed; solvers.host_direct.HOST_DIRECT_BACK_ENDS) the em_fdfd_cudss
# frequency_domain runs and the em_fdfd_adjoint_gradient ez reflection gradient (its transposed solve) against the
# same runs on a scipy_splu node, and the em_fdfd_eigenfrequency_pec_cavity lossy ez cavity on an eigen node of that
# back end against its closed form; relative tolerances per precision (uncalibrated: no run yet; the cuDSS twins
# measured double ~4e-14, mixed <= 5e-10)
_HOST_REFERENCE = 'scipy_splu'
_HOST_TOLERANCES = {'double': 1e-9, 'mixed': 1e-8}
_HOST_ADJOINT_RUN = 'adjoint'
_HOST_CAVITY_RUN = 'ez_lossy'   # selftests_eigenfrequency._RUNS key
# eigen cavity ((cells), cell [m]) per precision: a mixed request below arpack_shift_invert.MIXED_EIGEN_MIN_UNKNOWNS
# (20,000) factors in double (S5c), so the mixed run takes the 2.0 x 1.2 um cavity on 10 nm cells (24,321 unknowns)
_HOST_CAVITIES = {'double': ((51, 31), 40e-9), 'mixed': ((201, 121), 10e-9)}

# fd_joule_heating template: 1 x 1 um water box (all-CPML for EM, convection on the outer faces for heat), uniform 4 nm
# mesh (250 + 30 cells per axis: 78,400 FDFD unknowns), a gold nanowire of radius 30 nm at the centre (15 cells across)
# and a vertical gaussian beam line 0.3 um before it (waist 0.3 um on the wire axis; Rayleigh range pi w0^2 n / lambda0
# = 0.71 um in water, so ~0.96 E0 reaches the wire). Water: the library optical values hold at 1.55 um, so eps_r = 1.78
# (n = 1.334 at 530 nm) and sigma_e = 0 (its 1.55 um sigma_e of 4.22 S/m would absorb about as much as the wire).
# Library gold is one Drude pole: eps = -5.0 - 0.43j at 530 nm (no interband loss; measured gold has |Im eps| ~ 2
# there, so a real wire absorbs roughly 5x more). E_z along the wire axis excites no plasmon: the field inside is ~0.6
# E (skin depth ~38 nm) and the absorption width ~4 nm (0.07 of the diameter), so E0 = 1e6 V/m (intensity n E0^2 /
# (2 eta0) = 1.8 mW/um^2) deposits P ~ 6.7 W/m. Temperature rise ~ P / (2 pi k_water) ln(0.55 um / 30 nm) = 5 K across
# the water plus P / (h 4 um) = 4.2 K through the convection faces (h = 4e5 W/(m^2 K) ~ k_water / (0.5 um ln 20) stands
# for conduction through ~10 um of surrounding water): ~9 K, which shifts the water index by dndT dT ~ -8e-4.
FD_JOULE_HEATING_LABEL = 'Joule heating of a gold nanowire (frequency domain)'
FD_JOULE_HEATING_DESCRIPTION = ('EM (FDFD, E_z) + heat, stationary study: a 530 nm gaussian beam heats a 30 nm gold '
                                'nanowire in water by about 9 K; Joule and thermo-optic couplings, convection on the '
                                'outer faces, max T and the absorbed power.')
_JOULE_SIZE = 1e-6                        # m, square
_JOULE_CELL = 4e-9                        # m
_JOULE_BEAM_X = 0.2e-6                    # m
_JOULE_WAIST = 0.3e-6                     # m
_WATER_EPS_R_530NM = 1.78                 # n = 1.334
_JOULE_H = 4e5                            # W/(m^2 K)
_AMBIENT_TEMPERATURE = 293.15             # K


# ---- specs ----
def _probe_feature(index):
    return f'probe_{index}'


def cpu_specs(specs):
    """specs (a {run: ModelSpec} dict) with solver_defaults.backend 'cpu' on every spec (FDFD factorises on the host
    on both backends; the gpu assembly is covered by em_fdfd_cpu_gpu)."""
    for spec in specs.values():
        spec.solver_defaults.backend = 'cpu'
    return specs


def _hankel_probes(polarization, source_cell, offsets):
    """(geometry, features) of the out-of-plane field probes at cell centres `offsets` cells from `source_cell`."""
    source_i, source_j = source_cell
    geometry, features = [], []
    for index, (offset_i, offset_j) in enumerate(offsets):
        point_name = f'probe_point_{index}'
        geometry.append(GeometrySpec(point_name, 'point', {'x': cell_centre(source_i + offset_i, _HANKEL_CELL),
                                                           'y': cell_centre(source_j + offset_j, _HANKEL_CELL)}))
        features.append(FeatureSpec(_probe_feature(index), 'field_probe', [point_name], {'component': polarization}))
    return geometry, features


def _hankel_spec(polarization, size=_HANKEL_SIZE, source_cell=_HANKEL_SOURCE_CELL, offsets=_HANKEL_PROBE_OFFSETS):
    """All-CPML box with the line current in `source_cell` and probes of the out-of-plane field at `offsets`: ez a
    1 mA filament current, hz a 1 V magnetic line current."""
    source_i, source_j = source_cell
    source_values = ({'current': _HANKEL_CURRENT, 'phase': 0.0} if polarization == 'ez' else
                     {'excitation': 'magnetic_current', 'magnetic_current': _HANKEL_MAGNETIC_CURRENT, 'phase': 0.0})
    probe_geometry, probe_features = _hankel_probes(polarization, source_cell, offsets)
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f'Self-test: {physics_type} line-current Hankel field',
        domain=DomainSpec(size_x=size, size_y=size),
        geometry=[GeometrySpec('source_point', 'point', {'x': cell_centre(source_i, _HANKEL_CELL),
                                                         'y': cell_centre(source_j, _HANKEL_CELL)}),
                  *probe_geometry],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={'wavelength': _HANKEL_WAVELENGTH},
                             features=[FeatureSpec(_SOURCE, 'point_source', ['source_point'], source_values),
                                       *probe_features])],
        mesh=MeshSpec(mode='uniform', dx=_HANKEL_CELL),
        study=StudySpec(type='stationary'))


def _hankel_far_field_specs():
    """em_fdfd_line_current_hankel runs (cpu): 'ez' = the far-field ez box with the Hankel E_z probes (the probes and
    monitors leave the solve unchanged), 'hz' = the Hankel hz box, 'hz_dipole' = the far-field hz dipole box."""
    far_field = _far_field_specs()
    ez_spec = far_field['ez']
    probe_geometry, probe_features = _hankel_probes('ez', _HANKEL_SOURCE_CELL, _HANKEL_PROBE_OFFSETS)
    ez_spec.title = f'{ez_spec.title} and Hankel probes'
    ez_spec.geometry.extend(probe_geometry)
    ez_spec.physics_by_name(_PHYSICS_NAME).features.extend(probe_features)
    return cpu_specs({'ez': ez_spec, 'hz': _hankel_spec('hz'), _HANKEL_DIPOLE_RUN: far_field['hz']})


def _far_field_spec(polarization):
    """Hankel box with the ez filament current or the hz 30 deg dipole in cell (80, 80), a far-field box and a closed
    flux box around it."""
    source_i, source_j = _HANKEL_SOURCE_CELL
    x0, y0, x1, y1 = _FAR_FIELD_BOX
    flux_geometry, flux_features = _flux_box(_FAR_FIELD_FLUX_PREFIX, _FAR_FIELD_FLUX_LINES, _FAR_FIELD_FLUX_SPAN, {})
    source_values = {'current': _HANKEL_CURRENT, 'phase': 0.0} if polarization == 'ez' else dict(_FAR_FIELD_DIPOLE)
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f'{physics_type} far-field power of a line source',
        domain=DomainSpec(size_x=_HANKEL_SIZE, size_y=_HANKEL_SIZE),
        geometry=[GeometrySpec('source_point', 'point', {'x': cell_centre(source_i, _HANKEL_CELL),
                                                         'y': cell_centre(source_j, _HANKEL_CELL)}),
                  GeometrySpec('far_field_box', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}),
                  *flux_geometry],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={'wavelength': _HANKEL_WAVELENGTH}, features=[
            FeatureSpec(_SOURCE, 'point_source', ['source_point'], source_values),
            FeatureSpec(_FAR_FIELD, 'far_field', ['far_field_box'], {'n_angles': _FAR_FIELD_N_ANGLES}),
            *flux_features])],
        mesh=MeshSpec(mode='uniform', dx=_HANKEL_CELL),
        study=StudySpec(type='stationary'))


def _far_field_specs():
    """'ez' and 'hz': the same boxes, each with its source."""
    return {polarization: _far_field_spec(polarization) for polarization in _POLARIZATIONS}


def _graded_far_field_specs():
    """_far_field_specs on the graded mesh: base _HANKEL_CELL, the core square refined to _GRADED_FAR_FIELD_CELL."""
    lo, hi = _GRADED_FAR_FIELD_CORE
    specs = _far_field_specs()
    for spec in specs.values():
        spec.title = f'{spec.title}, graded mesh'
        spec.geometry.append(GeometrySpec('refine_core', 'rectangle', {'x0': lo, 'y0': lo, 'x1': hi, 'y1': hi}))
        spec.mesh = MeshSpec(mode='nonuniform', dx=_HANKEL_CELL, refinements=[
            RefinementSpec('refine_core', _GRADED_FAR_FIELD_CELL, _GRADED_FAR_FIELD_MARGIN)])
    return specs


def _fresnel_spec(polarization, medium, length=_FRESNEL_SIZE[0], source_x=_FRESNEL_SOURCE_X, flux_x=_FRESNEL_FLUX_X,
                  interface=_FRESNEL_INTERFACE):
    """y-periodic strip of `length` with the line source and the flux line; medium 'glass' adds the eps_r = 4
    half-space from `interface`, 'medium' fills every cell with it."""
    size_x, size_y = length, _FRESNEL_SIZE[1]
    geometry = [vertical_line('source_line', source_x, 0.0, size_y),
                vertical_line('flux_line', flux_x, 0.0, size_y)]
    materials = []
    if medium == 'glass':
        half_space, glass = glass_half_space(interface, size_x, size_y, _GLASS_EPS_R)
        geometry.append(half_space)
        materials.append(glass)
    elif medium == 'medium':
        materials.append(MaterialSpec('glass', properties={'eps_r': _GLASS_EPS_R}, selection=['all']))
    titles = {'vacuum': 'vacuum', 'glass': 'vacuum / glass interface', 'medium': 'glass everywhere'}
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f'{physics_type} Fresnel transmission: {titles[medium]}',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type,
                             settings={**PERIODIC_Y, 'wavelength': _FRESNEL_WAVELENGTH}, features=[
                                 FeatureSpec(_SOURCE, 'line_source', ['source_line'],
                                             {'amplitude': _FRESNEL_AMPLITUDE, 'phase': 0.0}),
                                 FeatureSpec(_FLUX, 'flux_monitor', ['flux_line'])])],
        mesh=MeshSpec(mode='uniform', dx=_FRESNEL_CELL),
        study=StudySpec(type='stationary'))


def _fresnel_specs():
    """'<polarization>_<medium>' for ez and hz and every _FRESNEL_MEDIA entry: same source and flux line."""
    return {f'{polarization}_{medium}': _fresnel_spec(polarization, medium)
            for polarization in _POLARIZATIONS for medium in _FRESNEL_MEDIA}


def _graded_fresnel_specs():
    """_fresnel_specs with the graded interface mesh of selftests_hz.graded_interface_mesh."""
    return {name: graded_interface_mesh(spec, _FRESNEL_INTERFACE) for name, spec in _fresnel_specs().items()}


def _slab_specs():
    """'ez' and 'hz': the selftests_mode_ports slab waveguide with a frequency_domain study at _SLAB_WAVELENGTHS (FDFD
    monitors: mode_number only)."""
    return {polarization: slab_waveguide_spec(_PHYSICS_TYPES[polarization], _SLAB_PORT_SOURCE, {},
                                              StudySpec(type='frequency_domain', settings=dict(_SLAB_STUDY)))
            for polarization in _POLARIZATIONS}


def _wall_specs():
    """'<polarization>_<wall>' of every selftests wall run (bc_x_lo walls, mirrors, pec_domain), stationary."""
    source_values = {'amplitude': _WALL_AMPLITUDE, 'phase': 0.0}
    return {wall_run_name(polarization, wall): wall_reflection_spec(
                _PHYSICS_TYPES[polarization], polarization, wall, source_values,
                {'wavelength': _WALL_WAVELENGTH}, StudySpec(type='stationary'))
            for polarization, wall in _WALL_RUNS}


def _drude_run_name(polarization, wavelength):
    return f'{polarization}_{round(wavelength * 1e9)}nm'


def _drude_spec(polarization, wavelength):
    """Drude cylinder lit by a +x TFSF plane wave inside an outer (scattered-field) and an inner (total-field) flux
    box, stationary at `wavelength`, with the derived q_abs integral over the cylinder."""
    x0, y0, x1, y1 = _FD_DRUDE_TFSF_BOX
    outer_geometry, outer_features = _flux_box(_DRUDE_OUTER_PREFIX, _FD_DRUDE_OUTER_LINES, _FD_DRUDE_OUTER_SPAN, {})
    inner_geometry, inner_features = _flux_box(_DRUDE_INNER_PREFIX, _FD_DRUDE_INNER_LINES, _FD_DRUDE_INNER_SPAN, {})
    metal = MaterialSpec('drude_metal', properties={}, selection=['cylinder'], dispersion=[
        {'model': 'drude', 'omega_p': _DRUDE_OMEGA_P, 'gamma': _DRUDE_GAMMA, 'eps_inf': _DRUDE_EPS_INF}])
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f'{physics_type} Drude cylinder at {round(wavelength * 1e9)} nm: Mie and energy balance',
        domain=DomainSpec(size_x=_FD_DRUDE_SIZE, size_y=_FD_DRUDE_SIZE),
        geometry=[GeometrySpec('tfsf_box', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}),
                  _cylinder(_FD_DRUDE_CENTER, _DRUDE_RADIUS), *outer_geometry, *inner_geometry],
        materials=[metal],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={'wavelength': wavelength}, features=[
            FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'],
                        {'amplitude': _FD_DRUDE_AMPLITUDE, 'phase': 0.0}),
            *outer_features, *inner_features])],
        mesh=MeshSpec(mode='uniform', dx=_DRUDE_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec(_FD_ABSORBED_VALUE, f'{_PHYSICS_NAME}_absorbed_power', ['cylinder'],
                                         'integral')])


def _drude_specs():
    """'<polarization>_<nm>nm' for ez and hz at every _FD_DRUDE_WAVELENGTHS entry."""
    return {_drude_run_name(polarization, wavelength): _drude_spec(polarization, wavelength)
            for polarization in _POLARIZATIONS for wavelength in _FD_DRUDE_WAVELENGTHS}


def _leak_probe_name(index):
    """Probe feature of _LEAK_OUTSIDE_PROBES[index], or 'probe_inside' for index None."""
    return 'probe_inside' if index is None else f'probe_outside_{index}'


def _leakage_spec(polarization):
    """Empty all-CPML vacuum box with the TFSF box, the refinement rectangle and the out-of-plane field probes."""
    x0, y0, x1, y1 = _LEAK_TFSF_BOX
    rx0, ry0, rx1, ry1 = _LEAK_REFINEMENT
    geometry = [GeometrySpec('tfsf_box', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}),
                GeometrySpec('refine_core', 'rectangle', {'x0': rx0, 'y0': ry0, 'x1': rx1, 'y1': ry1})]
    features = [FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'], {'amplitude': 1.0, 'phase': 0.0})]
    for index, (x, y) in [(None, _LEAK_INSIDE_PROBE), *enumerate(_LEAK_OUTSIDE_PROBES)]:
        point = f'{_leak_probe_name(index)}_point'
        geometry.append(GeometrySpec(point, 'point', {'x': x, 'y': y}))
        features.append(FeatureSpec(_leak_probe_name(index), 'field_probe', [point], {'component': polarization}))
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f'Self-test: {physics_type} empty TFSF box on a graded mesh',
        domain=DomainSpec(size_x=_LEAK_SIZE, size_y=_LEAK_SIZE),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={'wavelength': _HANKEL_WAVELENGTH},
                             features=features)],
        mesh=MeshSpec(mode='nonuniform', dx=_LEAK_CELL, refinements=[
            RefinementSpec('refine_core', _LEAK_REFINED_CELL, _LEAK_MARGIN)]),
        study=StudySpec(type='stationary'))


def _leakage_specs():
    """'ez', 'hz' (graded box) and 'oblique_ez', 'oblique_hz' (the oblique strip graded along x)."""
    specs = {polarization: _leakage_spec(polarization) for polarization in _POLARIZATIONS}
    for polarization in _POLARIZATIONS:
        specs[f'oblique_{polarization}'] = graded_interface_mesh(_oblique_tfsf_spec(polarization), _LEAK_OBLIQUE_X)
    return specs


def _line_block_specs():
    """'<model>_<solver>' for the small Hankel ez / hz boxes and the short Fresnel glass strips (default backend),
    solver 'sparse_direct' (a scipy_splu node _DIRECT_NODE) or a line_block node of each _LINE_BLOCK_PRECISIONS entry as
    'linear' solver."""
    models = {**{f'hankel_{polarization}': lambda polarization=polarization: _hankel_spec(
                     polarization, _LINE_BLOCK_HANKEL_SIZE, _LINE_BLOCK_SOURCE_CELL, _LINE_BLOCK_PROBE_OFFSETS)
                 for polarization in _POLARIZATIONS},
              **{f'fresnel_{polarization}': lambda polarization=polarization: _fresnel_spec(
                     polarization, 'glass', **_LINE_BLOCK_STRIP)
                 for polarization in _POLARIZATIONS}}
    specs = {}
    for model, factory in models.items():
        reference = factory()   # host SuperLU by a node: Automatic may resolve to cuDSS on the gpu
        reference.solvers.append(SolverSpec(_DIRECT_NODE, 'sparse_direct', {'back_end': 'scipy_splu'}))
        reference.physics_by_name(_PHYSICS_NAME).solvers['linear'] = _DIRECT_NODE
        specs[f'{model}_sparse_direct'] = reference
        for precision in _LINE_BLOCK_PRECISIONS:
            spec = factory()
            node = f'line_block_{precision}'
            spec.solvers.append(SolverSpec(node, 'line_block', {'precision': precision}))
            spec.physics_by_name(_PHYSICS_NAME).solvers['linear'] = node
            specs[f'{model}_{precision}'] = spec
    return specs


def _line_block_wavelength_specs():
    """The _line_block_specs runs of _BATCH_MODELS with a frequency_domain study of _BATCH_STUDY (wavelength setting
    dropped)."""
    specs = {}
    for name, spec in _line_block_specs().items():
        model = name[:-len('_sparse_direct')] if name.endswith('_sparse_direct') else name.rsplit('_', 1)[0]
        if model not in _BATCH_MODELS:
            continue
        spec.physics_by_name(_PHYSICS_NAME).settings.pop('wavelength', None)
        spec.study = StudySpec(type='frequency_domain', settings=dict(_BATCH_STUDY))
        specs[name] = spec
    return specs


def _cudss_specs():
    """'<model>_sparse_direct' (the scipy_splu node _DIRECT_NODE) and '<model>_<back end>' (that node on each
    _CUDSS_BACK_ENDS entry) frequency_domain runs of _BATCH_MODELS; {} without nvmath-python."""
    from ...solvers.cudss_factor import cudss_available
    if not cudss_available():
        return {}
    specs = {}
    for back_end in ('sparse_direct',) + _CUDSS_BACK_ENDS:
        for name, spec in _line_block_wavelength_specs().items():
            if not name.endswith('_sparse_direct'):
                continue
            if back_end != 'sparse_direct':   # the reference's direct node on this back end
                next(node for node in spec.solvers if node.name == _DIRECT_NODE).settings['back_end'] = back_end
            specs[f"{name[:-len('_sparse_direct')]}_{back_end}"] = spec
    return specs


def _host_direct_back_ends() -> tuple:
    """solvers.host_direct.HOST_DIRECT_BACK_ENDS of every library installed here (python-mumps / pydiso), in order."""
    from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS, library_of
    libraries = available_host_direct()
    return tuple(name for name in HOST_DIRECT_BACK_ENDS if library_of(name) in libraries)


def _host_precision(back_end) -> str:
    return 'mixed' if back_end.endswith('_mixed') else 'double'


def _host_direct_specs():
    """cpu runs per back end b of _HOST_REFERENCE + _host_direct_back_ends(): '<model>_<b>' (the em_fdfd_cudss
    frequency_domain models, _DIRECT_NODE on b) and 'adjoint_<b>' (em_fdfd_adjoint_gradient's ez reflection gradient
    run, a _DIRECT_NODE of b as 'linear'); per host back end 'cavity_<b>' (the lossy ez PEC cavity of its precision's
    _HOST_CAVITIES, an arpack_shift_invert node of factorization b as 'eigen'); {} without either library."""
    back_ends = _host_direct_back_ends()
    if not back_ends:
        return {}
    from .selftests_adjoint import _adjoint_spec
    from .selftests_eigenfrequency import _RUNS, _cavity_spec
    specs = {}
    for back_end in (_HOST_REFERENCE,) + back_ends:
        for name, spec in _line_block_wavelength_specs().items():
            if not name.endswith('_sparse_direct'):
                continue
            next(node for node in spec.solvers if node.name == _DIRECT_NODE).settings['back_end'] = back_end
            specs[f"{name[:-len('_sparse_direct')]}_{back_end}"] = spec
        specs[f'{_HOST_ADJOINT_RUN}_{back_end}'] = with_direct_node(_adjoint_spec('ez', _SILICON_EPS_R, 1.0, 0.0),
                                                                     back_end, node_name=_DIRECT_NODE)
    polarization, _, sigma_e = _RUNS[_HOST_CAVITY_RUN]
    for back_end in back_ends:
        cells, cell = _HOST_CAVITIES[_host_precision(back_end)]
        specs[f'cavity_{back_end}'] = with_direct_node(_cavity_spec(polarization, cells, sigma_e, cell), back_end,
                                                       role='eigen')
    return cpu_specs(specs)


def _role_record(results, role):
    """(type, settings, counters) of the summary 'solvers' record of role `role` of physics 'em' (('', {}, {}) when
    missing)."""
    record = solver_role(results, _PHYSICS_NAME, role)
    return record.get('type', ''), record.get('settings') or {}, record.get('counters') or {}


def _check_host_direct(results_by_name):
    """Per host back end: max |host - SuperLU| / max |SuperLU| of the em_fdfd_cudss models over every wavelength and
    |host - SuperLU| / |SuperLU| of the adjoint block gradient in _HOST_TOLERANCES units, the lossy cavity's omega / Q
    against the closed form (selftests_eigenfrequency tolerances), and the role records (sparse_direct on that back end:
    one symmetric factorisation per wavelength, no mixed -> double fallback; the eigen node: a symmetric factor, no
    fallback); skipped without python-mumps and pydiso."""
    if not results_by_name:
        return skip_without_host_direct() or skip_result('no host direct back end installed')
    from .selftests_eigenfrequency import _OMEGA_TOLERANCE, _Q_TOLERANCE, _cavity_errors
    back_ends = tuple(name[len('cavity_'):] for name in results_by_name if name.startswith('cavity_'))
    n_wavelengths = int(_BATCH_STUDY['n_wavelengths'])
    errors, tolerances, details = {}, {}, {}
    gradient = 'block_gradient'   # selftests_adjoint._GRADIENT_VALUE
    reference_gradient = float(results_by_name[f'{_HOST_ADJOINT_RUN}_{_HOST_REFERENCE}'].summary()['derived'][gradient])
    for back_end in back_ends:
        tolerance = _HOST_TOLERANCES[_host_precision(back_end)]
        for model in _BATCH_MODELS:   # driven solves, LDL^T of D A refactorised per wavelength
            reference = _wavelength_values(results_by_name[f'{model}_{_HOST_REFERENCE}'], model)
            expected_size = n_wavelengths * (len(_LINE_BLOCK_PROBE_OFFSETS) if model.startswith('hankel') else 1)
            scale = float(np.max(np.abs(reference))) if reference.size == expected_size else math.nan
            results = results_by_name[f'{model}_{back_end}']
            values = _wavelength_values(results, model)
            deviation = (float(np.max(np.abs(values - reference))) / scale if values.shape == reference.shape
                         else math.inf)   # NaN propagates
            solver_type, _, counters = _role_record(results, 'linear')
            record = {name: counters.get(name) for name in ('back_end', 'factorizations',
                                                            'host_direct_symmetric_factors',
                                                            'host_direct_double_fallbacks')}
            expected_record = {'back_end': back_end, 'factorizations': n_wavelengths,
                               'host_direct_symmetric_factors': n_wavelengths, 'host_direct_double_fallbacks': 0}
            errors[f'{model} {back_end}'], tolerances[f'{model} {back_end}'] = deviation, tolerance
            errors[f'{model} {back_end} record'] = (
                0.0 if solver_type == 'sparse_direct' and record == expected_record else math.inf)
            tolerances[f'{model} {back_end} record'] = 1.0
            details[f'{model}_{back_end}'] = {'deviation': deviation, 'counters': dict(counters)}
        results = results_by_name[f'{_HOST_ADJOINT_RUN}_{back_end}']   # the transposed solve of eps_sensitivity
        value = float(results.summary()['derived'][gradient])   # float() also reads 'nan'
        errors[f'adjoint {back_end}'] = (abs(value - reference_gradient) / abs(reference_gradient)
                                         if reference_gradient != 0.0 else math.nan)
        tolerances[f'adjoint {back_end}'] = tolerance
        _, _, counters = _role_record(results, 'linear')
        errors[f'adjoint {back_end} record'] = 0.0 if counters.get('back_end') == back_end else math.inf
        tolerances[f'adjoint {back_end} record'] = 1.0
        details[f'adjoint_{back_end}'] = {'gradient': value, 'superlu_gradient': reference_gradient}
        cells, cell = _HOST_CAVITIES[_host_precision(back_end)]   # shift-invert by the eigen node's own factor
        results = results_by_name[f'cavity_{back_end}']
        part_errors, part_tolerances, part_details = _cavity_errors({_HOST_CAVITY_RUN: results}, _OMEGA_TOLERANCE,
                                                                    _Q_TOLERANCE, cell, {_HOST_CAVITY_RUN: cells})
        errors.update({f'{back_end} {item}': error for item, error in part_errors.items()})
        tolerances.update({f'{back_end} {item}': part for item, part in part_tolerances.items()})
        _, settings, counters = _role_record(results, 'eigen')
        used = (settings.get('factorization') == back_end and counters.get('eigen_factor_symmetric') == 1
                and counters.get('eigen_double_fallbacks') == 0)
        errors[f'{back_end} eigen record'], tolerances[f'{back_end} eigen record'] = (0.0 if used else math.inf), 1.0
        details[f'cavity_{back_end}'] = {**part_details, 'counters': dict(counters)}
    details['back_ends'] = list(back_ends)
    return normalized_result({item: error / tolerances[item] for item, error in errors.items()}, details)


def fd_joule_heating_spec():
    """Template 'fd_joule_heating' (em_fdfd_ez + heat, stationary study): a 530 nm gaussian beam heats a gold
    nanowire in water; Joule and thermo-optic couplings iterate to the steady state."""
    center = 0.5 * _JOULE_SIZE
    return ModelSpec(
        title=FD_JOULE_HEATING_LABEL,
        parameters=[ParameterSpec('lambda0', '530*nm', 'vacuum wavelength of the beam'),
                    ParameterSpec('E0', 1e6, 'beam peak field at the waist [V/m]'),
                    ParameterSpec('r_wire', '30*nm', 'nanowire radius')],
        domain=DomainSpec(size_x=_JOULE_SIZE, size_y=_JOULE_SIZE),
        geometry=[GeometrySpec('wire', 'circle', {'cx': center, 'cy': center, 'r': 'r_wire'}),
                  vertical_line('beam_line', _JOULE_BEAM_X, 0.0, _JOULE_SIZE),
                  GeometrySpec('wire_center', 'point', {'x': center, 'y': center})],
        materials=[MaterialSpec('water', library='water',
                                properties={'eps_r': _WATER_EPS_R_530NM, 'sigma_e': 0.0}, selection=['all']),
                   MaterialSpec('gold', library='gold', selection=['wire'])],
        physics=[PhysicsSpec('em', 'em_fdfd_ez', settings={'wavelength': 'lambda0'}, features=[
                     FeatureSpec('beam', 'gaussian_beam', ['beam_line'],
                                 {'amplitude': 'E0', 'phase': 0.0, 'waist': _JOULE_WAIST, 'center': center})]),
                 PhysicsSpec('heat', 'heat', features=[
                     FeatureSpec('outer_water', 'convection', list(OUTER_FACES),
                                 {'h': _JOULE_H, 'T_ext': _AMBIENT_TEMPERATURE}),
                     FeatureSpec('wire_temperature', 'temperature_probe', ['wire_center'])])],
        couplings=[CouplingSpec('joule', 'em_heat_joule', 'em', 'heat'),
                   CouplingSpec('thermo_optic', 'heat_em_thermo_optic', 'heat', 'em')],
        mesh=MeshSpec(mode='uniform', dx=_JOULE_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max'),
                        # q_abs of the last FDFD solve integrated over the wire: absorbed power per unit length [W/m]
                        DerivedValueSpec('P_absorbed', 'absorbed_power', ['wire'], 'integral')])


# ---- checks ----
def _hankel0_argument(argument):
    """arg H0^(2)(x) [rad] at x = argument (scipy.special.hankel2)."""
    import scipy.special
    return float(np.angle(scipy.special.hankel2(0, argument)))


def _wrapped(angle):
    """angle [rad] wrapped into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _check_hankel(results_by_name):
    """Per polarisation and probe: |field| vs (omega mu0 I / 4 | omega eps0 K_m / 4) |H0^(2)(k r)| and its phase vs
    arg H0^(2)(k r); measured = error = the worst deviation in tolerance units."""
    omega = 2.0 * math.pi * C0 / _HANKEL_WAVELENGTH
    wavenumber = 2.0 * math.pi / _HANKEL_WAVELENGTH
    prefactor = {'ez': omega * MU0 * _HANKEL_CURRENT / 4.0, 'hz': omega * EPS0 * _HANKEL_MAGNETIC_CURRENT / 4.0}
    radii = [_HANKEL_CELL * math.hypot(offset_i, offset_j) for offset_i, offset_j in _HANKEL_PROBE_OFFSETS]
    moduli = [_hankel0_modulus(wavenumber * radius) for radius in radii]
    arguments = [_hankel0_argument(wavenumber * radius) for radius in radii]
    details, normalized_errors = {'unit': {'ez': 'V/m', 'hz': 'A/m'}, 'radii': radii, 'expected_phase': arguments}, {}
    for polarization in _POLARIZATIONS:
        results = results_by_name[polarization]
        measured, phases = [], []
        for index in range(len(_HANKEL_PROBE_OFFSETS)):
            series = results.result(_PHYSICS_NAME, _probe_feature(index))['series']
            measured.append(single_value(series['abs'], f'{polarization} probe {index} abs'))
            phases.append(single_value(series['phase'], f'{polarization} probe {index} phase'))
        expected = [prefactor[polarization] * modulus for modulus in moduli]
        relative_errors = np.abs(np.asarray(measured) / np.asarray(expected) - 1.0)
        phase_errors = np.abs([_wrapped(phase - argument) for phase, argument in zip(phases, arguments)])
        err_modulus = float(np.max(relative_errors))   # NaN propagates
        err_phase = float(np.max(phase_errors))
        normalized_errors.update({f'{polarization} modulus': err_modulus / _HANKEL_TOLERANCE,
                                  f'{polarization} phase': err_phase / _HANKEL_PHASE_TOLERANCE})
        details.update({f'{polarization}_measured_by_probe': measured, f'{polarization}_expected_by_probe': expected,
                        f'{polarization}_phase_by_probe': phases,
                        f'{polarization}_relative_errors': float_list(relative_errors),
                        f'{polarization}_phase_errors': float_list(phase_errors)})
    return normalized_result(normalized_errors, details)


def _check_far_field_power(results_by_name):
    """Per polarisation: the angle integral of the far-field power vs the closed flux box, and the peak-normalised
    pattern vs 1 (ez) / sin^2(phi - 30 deg) (hz); measured = error = the worst deviation in tolerance units."""
    wavenumber = 2.0 * math.pi / _HANKEL_WAVELENGTH
    # Continuum radiated powers [W/m]: filament k eta0 I^2 / 8, dipole k eta0 M^2 / 16.
    analytic = {'ez': wavenumber * float(ETA0) * _HANKEL_CURRENT ** 2 / 8.0,
                'hz': wavenumber * float(ETA0) * _FAR_FIELD_DIPOLE['dipole_moment'] ** 2 / 16.0}
    details, normalized_errors = {'unit': 'W/m'}, {}
    for polarization in _POLARIZATIONS:
        results = results_by_name[polarization]
        far_field = results.result(_PHYSICS_NAME, _FAR_FIELD)
        angles = np.asarray(far_field['angles'], dtype=np.float64)
        power = np.asarray(far_field['power'], dtype=np.float64)
        if power.shape != (1, angles.size):
            raise ValueError(f'{polarization} far field: expected one solved wavelength x {angles.size} angles, got '
                             f'shape {power.shape}')
        power = power[0]
        far_field_total = float(np.sum(power)) * 2.0 * math.pi / angles.size   # periodic trapezoid over the angle
        flux_total = single_value(_net_outflow(results, _FAR_FIELD_FLUX_PREFIX), f'{polarization} flux box')
        ratio = far_field_total / flux_total if flux_total != 0.0 else math.nan
        expected_shape = (np.ones_like(angles) if polarization == 'ez'
                          else np.sin(angles - math.radians(_FAR_FIELD_DIPOLE['orientation'])) ** 2)
        peak = float(np.max(power))
        err_shape = float(np.max(np.abs(power / peak - expected_shape))) if peak > 0.0 else math.nan
        err_power = abs(ratio - 1.0)
        normalized_errors.update({f'{polarization} power': err_power / _FAR_FIELD_POWER_TOLERANCE,
                                  f'{polarization} shape': err_shape / _FAR_FIELD_SHAPE_TOLERANCE})
        details.update({f'{polarization}_far_field_power': far_field_total,
                        f'{polarization}_flux_box_power': flux_total,
                        f'{polarization}_analytic_power': analytic[polarization],
                        f'{polarization}_err_power': err_power, f'{polarization}_err_shape': err_shape})
    return normalized_result(normalized_errors, details)


def _check_hankel_far_field(results_by_name):
    """_check_hankel on runs 'ez' / 'hz' and _check_far_field_power on 'ez' / _HANKEL_DIPOLE_RUN; measured = error =
    the worse of the two (both in tolerance units, tolerance 1)."""
    checks = {'hankel': _check_hankel(results_by_name),
              'far_field': _check_far_field_power({'ez': results_by_name['ez'],
                                                   'hz': results_by_name[_HANKEL_DIPOLE_RUN]})}
    worst_check, worst = worst_item({name: check['error'] for name, check in checks.items()})
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0,
            'details': {'worst_check': worst_check, **{name: check['details'] for name, check in checks.items()}}}


def _check_fresnel(results_by_name):
    """Per polarisation T = flux(glass) / flux(vacuum) vs 4 n / (n + 1)^2 = 8/9 and the absolute vacuum / medium flux
    vs n A^2 L / (2 eta0); measured / expected = T of the worst polarisation, error = the worst deviation in tolerance
    units (NaN counts as the worst)."""
    vacuum_expected = _FRESNEL_AMPLITUDE ** 2 * _FRESNEL_SIZE[1] / (2.0 * float(ETA0))   # W/m sent to each side
    power_expected = {'vacuum': vacuum_expected, 'medium': _GLASS_INDEX * vacuum_expected}
    transmittance, normalized_errors = {}, {}
    details = {'unit': 'W/m', 'vacuum_power_expected': vacuum_expected,
               'medium_power_expected': power_expected['medium']}
    for polarization in _POLARIZATIONS:
        powers = {medium: single_value(
            results_by_name[f'{polarization}_{medium}'].result(_PHYSICS_NAME, _FLUX)['power'],
            f'{polarization} {medium} flux') for medium in _FRESNEL_MEDIA}
        transmittance[polarization] = powers['glass'] / powers['vacuum'] if powers['vacuum'] != 0.0 else math.nan
        normalized_errors[f'{polarization} T'] = (abs(transmittance[polarization] - _FRESNEL_TRANSMISSION)
                                                  / _FRESNEL_TOLERANCE)
        for medium, expected in power_expected.items():
            normalized_errors[f'{polarization} {medium} power'] = (abs(powers[medium] / expected - 1.0)
                                                                   / _FRESNEL_POWER_TOLERANCE)
        details.update({f'{polarization}_{medium}_power': power for medium, power in powers.items()})
    worst_name, worst_error = worst_item(normalized_errors)
    worst_polarization = max(_POLARIZATIONS, key=lambda polarization: math.inf
                             if math.isnan(transmittance[polarization])
                             else abs(transmittance[polarization] - _FRESNEL_TRANSMISSION))
    details.update(transmittance=transmittance, worst_polarization=worst_polarization,
                   normalized_errors=normalized_errors, worst_item=worst_name)
    return {'measured': transmittance[worst_polarization], 'expected': _FRESNEL_TRANSMISSION,
            'error': worst_error, 'tolerance': 1.0, 'details': details}


def _check_slab_wavelengths(wavelengths, what):
    """ValueError unless a mode monitor result lists exactly the study wavelengths _SLAB_WAVELENGTHS (ascending)."""
    expected = np.asarray(_SLAB_WAVELENGTHS)
    if wavelengths.shape != expected.shape or not np.allclose(wavelengths, expected, rtol=1e-9, atol=0.0):
        raise ValueError(f'{what}: expected the study wavelengths {list(_SLAB_WAVELENGTHS)} m, got '
                         f'{float_list(wavelengths)}')


def _check_mode_port_slab(results_by_name):
    """Per polarisation at every study wavelength: downstream fraction_positive vs 1, every other fraction vs 0 and the
    downstream n_eff vs the analytic slab index; measured = error = the worst deviation in tolerance units."""
    details, normalized_errors = {'wavelengths': list(_SLAB_WAVELENGTHS)}, {}
    for polarization in _POLARIZATIONS:
        results = results_by_name[polarization]
        wavelengths, forward, reflected, n_eff = _mode_spectra(results, _DOWNSTREAM)
        upstream_wavelengths, upstream_forward, upstream_backward, _ = _mode_spectra(results, _UPSTREAM)
        _check_slab_wavelengths(wavelengths, f'{polarization} {_DOWNSTREAM} mode monitor')
        _check_slab_wavelengths(upstream_wavelengths, f'{polarization} {_UPSTREAM} mode monitor')
        n_analytic = np.array([_slab_fundamental_index(wavelength, polarization) for wavelength in wavelengths])
        err_forward = float(np.max(np.abs(forward - 1.0)))   # NaN propagates
        err_leak = float(np.max(np.abs([reflected, upstream_forward, upstream_backward])))
        err_index = float(np.max(np.abs(n_eff / n_analytic - 1.0)))
        normalized_errors.update({f'{polarization} forward': err_forward / _SLAB_FORWARD_TOLERANCE,
                                  f'{polarization} leak': err_leak / _SLAB_LEAK_TOLERANCE,
                                  f'{polarization} n_eff': err_index / _SLAB_INDEX_TOLERANCE})
        details.update({f'{polarization}_n_eff': float_list(n_eff),
                        f'{polarization}_n_eff_analytic': float_list(n_analytic),
                        f'{polarization}_downstream_fraction_positive': float_list(forward),
                        f'{polarization}_downstream_fraction_negative': float_list(reflected),
                        f'{polarization}_upstream_fraction_positive': float_list(upstream_forward),
                        f'{polarization}_upstream_fraction_negative': float_list(upstream_backward),
                        f'{polarization}_err_forward': err_forward, f'{polarization}_err_leak': err_leak,
                        f'{polarization}_err_index': err_index})
    return normalized_result(normalized_errors, details)


def _check_wall_reflection(results_by_name, runs=_WALL_RUNS):
    """Downstream |E| probe per (polarization, wall) run of `runs` vs selftests.wall_expected_amplitude."""
    measured = {}
    for polarization, wall in runs:
        name = wall_run_name(polarization, wall)
        series = results_by_name[name].result(_PHYSICS_NAME, _PROBE_FEATURE)['series']
        measured[name] = single_value(series['abs'], f'{name} probe abs')
    return wall_reflection_result(measured)


def _check_drude_cylinder(results_by_name):
    """Per polarisation and wavelength: Q_sca / Q_abs of the outer / inner flux boxes vs the Mie series and the inner
    box's inward flux vs the integrated q_abs; measured = error = the worst deviation in tolerance units."""
    geometric_power = _FD_DRUDE_AMPLITUDE ** 2 / (2.0 * float(ETA0)) * 2.0 * _DRUDE_RADIUS   # I 2a [W/m]
    wavelengths = np.asarray(_FD_DRUDE_WAVELENGTHS)
    details, normalized_errors = {'wavelengths': list(_FD_DRUDE_WAVELENGTHS), 'geometric_power': geometric_power}, {}
    for polarization in _POLARIZATIONS:
        q_sca, q_abs = _cylinder_efficiencies(wavelengths, _DRUDE_RADIUS, _drude_relative_index, polarization == 'ez')
        q_sca_measured, q_abs_measured, balance_errors, loss_powers = [], [], [], []
        for wavelength in _FD_DRUDE_WAVELENGTHS:
            name = _drude_run_name(polarization, wavelength)
            results = results_by_name[name]
            scattered = single_value(_net_outflow(results, _DRUDE_OUTER_PREFIX), f'{name} outer box')
            absorbed = -single_value(_net_outflow(results, _DRUDE_INNER_PREFIX), f'{name} inner box')
            loss_power = float(results.summary()['derived'][_FD_ABSORBED_VALUE])   # float() also reads 'nan'
            q_sca_measured.append(scattered / geometric_power)
            q_abs_measured.append(absorbed / geometric_power)
            loss_powers.append(loss_power)
            balance_errors.append(abs(loss_power / absorbed - 1.0) if absorbed != 0.0 else math.nan)
        err_sca = float(np.max(np.abs(np.asarray(q_sca_measured) / q_sca - 1.0)))   # NaN propagates
        err_abs = float(np.max(np.abs(np.asarray(q_abs_measured) / q_abs - 1.0)))
        err_balance = float(np.max(balance_errors))
        normalized_errors.update({f'{polarization} Q_sca': err_sca / _FD_DRUDE_SCATTERING_TOLERANCE,
                                  f'{polarization} Q_abs': err_abs / _FD_DRUDE_ABSORPTION_TOLERANCE,
                                  f'{polarization} balance': err_balance / _FD_DRUDE_BALANCE_TOLERANCE})
        details.update({f'{polarization}_Q_sca_meas': float_list(q_sca_measured),
                        f'{polarization}_Q_sca': float_list(q_sca),
                        f'{polarization}_Q_abs_meas': float_list(q_abs_measured),
                        f'{polarization}_Q_abs': float_list(q_abs),
                        f'{polarization}_absorbed_line_power': loss_powers,
                        f'{polarization}_balance_errors': balance_errors})
    return normalized_result(normalized_errors, details)


def _check_tfsf_leakage(results_by_name):
    """Graded boxes: max |scattered field| / |incident| over the outside probes; oblique strips: upstream / downstream
    flux over the inside flux; error = the worst in tolerance units (raises unless every build was graded)."""
    require_graded({name: results_by_name[name] for name in _POLARIZATIONS}, _LEAK_CELL)
    require_graded({f'oblique_{name}': results_by_name[f'oblique_{name}'] for name in _POLARIZATIONS}, 10e-9)
    normalized_errors, details = {}, {}
    for polarization in _POLARIZATIONS:
        results = results_by_name[polarization]
        incident = abs(probe_phasor(results, _PHYSICS_NAME, _leak_probe_name(None)))
        leaked = [abs(probe_phasor(results, _PHYSICS_NAME, _leak_probe_name(index))) / incident
                  for index in range(len(_LEAK_OUTSIDE_PROBES))]
        normalized_errors[f'{polarization} field'] = max(leaked) / _LEAK_FIELD_TOLERANCE   # NaN propagates
        details[f'{polarization}_incident_abs'] = incident
        details[f'{polarization}_leaked_relative'] = float_list(leaked)
        oblique = results_by_name[f'oblique_{polarization}']
        powers = {name: float(np.asarray(oblique.result(_PHYSICS_NAME, name)['power'], dtype=np.float64)[0])
                  for name in _TFSF_FLUX_LINES}
        ratio = max(abs(powers['upstream']), abs(powers['downstream'])) / abs(powers['inside'])
        normalized_errors[f'oblique {polarization} flux'] = ratio / _LEAK_FLUX_TOLERANCE
        details[f'oblique_{polarization}_powers'] = powers
    return normalized_result(normalized_errors, details)


def _run_values(results, model):
    """Complex values a run is compared by: the Hankel probe phasors, or the Fresnel transmitted flux."""
    if model.startswith('hankel'):
        return np.asarray([probe_phasor(results, _PHYSICS_NAME, _probe_feature(index))
                           for index in range(len(_LINE_BLOCK_PROBE_OFFSETS))], dtype=np.complex128)
    return np.asarray(results.result(_PHYSICS_NAME, _FLUX)['power'], dtype=np.complex128)


def _check_line_block(results_by_name):
    """Per model and precision: max |line_block - sparse_direct| / max |sparse_direct| of the probe phasors / flux in
    tolerance units, and the line_block run must record type line_block with no rejected (re-solved) factor solves."""
    normalized_errors, details = {}, {}
    for model in ('hankel_ez', 'hankel_hz', 'fresnel_ez', 'fresnel_hz'):
        reference = _run_values(results_by_name[f'{model}_sparse_direct'], model)
        scale = float(np.max(np.abs(reference)))
        for precision in _LINE_BLOCK_PRECISIONS:
            results = results_by_name[f'{model}_{precision}']
            values = _run_values(results, model)
            deviation = (float(np.max(np.abs(values - reference))) / scale if values.shape == reference.shape
                         else math.inf)   # NaN propagates
            solver_type, _, counters = _role_record(results, 'linear')
            rejected = (float(counters.get('rejected_factor_solves', math.inf)) if solver_type == 'line_block'
                        else math.inf)
            normalized_errors[f'{model} {precision}'] = deviation / _LINE_BLOCK_TOLERANCES[precision]
            normalized_errors[f'{model} {precision} rejected'] = 0.0 if rejected == 0.0 else math.inf
            details[f'{model}_{precision}'] = {'deviation': deviation, 'solver_type': solver_type,
                                               'counters': dict(counters)}
    return normalized_result(normalized_errors, details)


def _wavelength_values(results, model):
    """Complex values of a frequency_domain run, every wavelength: the Hankel probe phasors (probe after probe) or the
    Fresnel transmitted flux."""
    if not model.startswith('hankel'):
        return np.asarray(results.result(_PHYSICS_NAME, _FLUX)['power'], dtype=np.complex128).ravel()
    phasors = []
    for index in range(len(_LINE_BLOCK_PROBE_OFFSETS)):
        series = results.result(_PHYSICS_NAME, _probe_feature(index))['series']
        phasors.append(np.asarray(series['abs'], dtype=np.float64).ravel()
                       * np.exp(1j * np.asarray(series['phase'], dtype=np.float64).ravel()))
    return np.concatenate(phasors)


def _check_line_block_wavelengths(results_by_name):
    """Per model and precision: max |line_block - sparse_direct| / max |sparse_direct| over every wavelength in
    tolerance units; the line_block record must show every wavelength from a batch and no rejected factor solve."""
    n_wavelengths = int(_BATCH_STUDY['n_wavelengths'])
    normalized_errors, details = {}, {}
    for model in _BATCH_MODELS:
        reference = _wavelength_values(results_by_name[f'{model}_sparse_direct'], model)
        expected_size = n_wavelengths * (len(_LINE_BLOCK_PROBE_OFFSETS) if model.startswith('hankel') else 1)
        scale = float(np.max(np.abs(reference))) if reference.size == expected_size else math.nan
        for precision in _LINE_BLOCK_PRECISIONS:
            results = results_by_name[f'{model}_{precision}']
            values = _wavelength_values(results, model)
            deviation = (float(np.max(np.abs(values - reference))) / scale if values.shape == reference.shape
                         else math.inf)   # NaN propagates
            solver_type, _, counters = _role_record(results, 'linear')
            line_block = solver_type == 'line_block'
            batched = float(counters.get('batched_operators', -1)) if line_block else -1.0
            rejected = float(counters.get('rejected_factor_solves', math.inf)) if line_block else math.inf
            normalized_errors[f'{model} {precision}'] = deviation / _LINE_BLOCK_TOLERANCES[precision]
            normalized_errors[f'{model} {precision} batched'] = 0.0 if batched == n_wavelengths else math.inf
            normalized_errors[f'{model} {precision} rejected'] = 0.0 if rejected == 0.0 else math.inf
            details[f'{model}_{precision}'] = {'deviation': deviation, 'solver_type': solver_type,
                                               'counters': dict(counters)}
    return normalized_result(normalized_errors, details)


def _check_cudss(results_by_name):
    """Per model and cuDSS back end: max |cuDSS - SuperLU| / max |SuperLU| over every wavelength in tolerance units; the
    linear record must show that back end with one factorisation per wavelength on one plan, each an LDL^T."""
    if not results_by_name:
        return skip_result('nvmath-python (cuDSS) is not installed')
    n_wavelengths = int(_BATCH_STUDY['n_wavelengths'])
    normalized_errors, details = {}, {}
    for model in _BATCH_MODELS:
        reference = _wavelength_values(results_by_name[f'{model}_sparse_direct'], model)
        expected_size = n_wavelengths * (len(_LINE_BLOCK_PROBE_OFFSETS) if model.startswith('hankel') else 1)
        scale = float(np.max(np.abs(reference))) if reference.size == expected_size else math.nan
        for back_end in _CUDSS_BACK_ENDS:
            results = results_by_name[f'{model}_{back_end}']
            values = _wavelength_values(results, model)
            deviation = (float(np.max(np.abs(values - reference))) / scale if values.shape == reference.shape
                         else math.inf)   # NaN propagates
            solver_type, _, counters = _role_record(results, 'linear')
            record = {name: counters.get(name) for name in ('back_end', 'factorizations', 'cudss_plans',
                                                            'cudss_symmetric_factors')}
            expected_record = {'back_end': back_end, 'factorizations': n_wavelengths, 'cudss_plans': 1,
                               'cudss_symmetric_factors': n_wavelengths}
            normalized_errors[f'{model} {back_end}'] = deviation / _CUDSS_TOLERANCE
            normalized_errors[f'{model} {back_end} record'] = (
                0.0 if solver_type == 'sparse_direct' and record == expected_record else math.inf)
            details[f'{model}_{back_end}'] = {'deviation': deviation, 'solver_type': solver_type,
                                              'counters': dict(counters)}
    return normalized_result(normalized_errors, details)


FDFD_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_line_current_hankel',
        label='EM FDFD line-current Hankel field and far-field power',
        description='Hankel part: a line-current phasor at the vacuum wavelength 1 um (stationary study, phase 0, cpu '
                    'backend) in cell (80, 80) of a 4 x 4 um all-CPML box (25 nm mesh, 40 cells per wavelength, '
                    "36,100 unknowns incl. PML). Runs 'ez': em_fdfd_ez with a filament current I = 1 mA and E_z "
                    "probes (the far-field ez box below; its monitors leave the solve unchanged), 'hz': em_fdfd_hz "
                    'with a magnetic line current K_m = 1 V and H_z probes. Field probes on cell centres at r = 0.5, '
                    '0.9, 1.4 um along '
                    '+x and 0.53, 1.06, 1.48 um along the diagonal (r = the distance between the source and probe '
                    'cell centres) record abs and phase: abs must match the 2D line-current field omega mu0 I / 4 '
                    '|H0^(2)(k r)| (E_z) and omega eps0 K_m / 4 |H0^(2)(k r)| (H_z), and phase must equal arg '
                    'H0^(2)(k r) ~ -(k r - pi/4 - 1/(8 k r)): the FDFD equations give E_z = +(omega mu0 I / 4) '
                    'H0^(2)(k r) and H_z = +(omega eps0 K_m / 4) H0^(2)(k r) because every source keeps the FDTD sign '
                    '(J_F = -J_std), so a sign flip of the right-hand side is off by pi. measured = error = max(|abs / '
                    'expected - 1| / 0.03, |wrapped phase error| / 0.03 rad) over both runs and all probes, expected 0, '
                    'tolerance 1. This pins the source calibrations, their phasor sign and the scale of the Helmholtz '
                    'operator. Budget: the discrete Green function of the 5-point stencil (dispersion (k h)^2 / 24 ~ '
                    '0.1% in k, i.e. a phase lag up to ~0.01 rad at k r = 8.8; lattice anisotropy and near-field error '
                    'O((h / r)^2) < 0.3% at r >= 20 cells) and CPML reflections (the probe at 1.4 um is 0.6 um from '
                    'the PML). H0^(2) comes from scipy.special. Far-field part (formerly em_fdfd_far_field_power): '
                    f"the same box with a source in cell (80, 80): run 'ez' the 1 mA filament current, run "
                    f"'{_HANKEL_DIPOLE_RUN}' (em_fdfd_hz) a 1 mA in-plane electric dipole at 30 deg. A far_field box "
                    '(closed contour through the cell centres at 1.0125 and 3.0375 um, 360 angles) and a closed box '
                    'of four flux monitors (faces at 0.775 and 3.275 um) surround it. Per polarisation the angle '
                    'integral of '
                    'the far-field power [W/(m rad)] must equal the net flux out of the box [W/m] (this pins the 2D '
                    'prefactor sqrt(k / (8 pi)) and the 1 / (2 eta0)), and the peak-normalised pattern must be 1 '
                    '(filament) or sin^2(phi - 30 deg) (dipole; the null tests the E / H collocation on the contour). '
                    'measured = error = max(|power ratio - 1| / 0.02, max |pattern - expected| / 0.02) over both '
                    'polarisations, expected 0, tolerance 1; details give both powers and the continuum k eta0 I^2 / 8 '
                    '(filament) and k eta0 M^2 / 16 (dipole). Budget: the transform uses the continuum k0 while the '
                    'near field has the discrete dispersion ((k h)^2 / 24 ~ 0.1%), the trapezoid contour at 40 '
                    'samples per wavelength (~0.2%) and CPML reflections. measured = error = the worse part, both in '
                    'tolerance units, tolerance 1. Not yet confirmed by a run as one case (merged, cpu backend).',
        specs=_hankel_far_field_specs,
        check=_check_hankel_far_field),
    SelfTestCase(
        name='em_fdfd_far_field_power_graded',
        label='EM FDFD far-field power of a line source on a graded mesh',
        description=f"The far-field runs of em_fdfd_line_current_hankel (filament ez, dipole hz; cpu backend) on a "
                    f"graded mesh: mode 'nonuniform', base "
                    f'{_HANKEL_CELL * 1e9:g} nm, a centred {(_GRADED_FAR_FIELD_CORE[1] - _GRADED_FAR_FIELD_CORE[0]) * 1e6:g} '
                    f'um square refined to {_GRADED_FAR_FIELD_CELL * 1e9:g} nm with a '
                    f'{_GRADED_FAR_FIELD_MARGIN * 1e6:g} um margin (both axes), so every face of the far-field contour '
                    'samples graded cells along it (actual centres, trapezoid weights) while the contour lines and the '
                    'flux lines stay in base cells. Same checks and tolerances as there (the far_field feature no '
                    'longer forces a uniform mesh, solver_plan S7.1); the check raises unless both builds reached '
                    'dx_min <= 0.6 x the base cell (no silent uniform fallback). Not yet confirmed by a run.',
        specs=lambda: cpu_specs(_graded_far_field_specs()),
        check=lambda results_by_name: _check_far_field_power(require_graded(results_by_name, _HANKEL_CELL))),
    SelfTestCase(
        name='em_fdfd_fresnel_transmission',
        label='EM FDFD Fresnel transmission',
        description='A line source (amplitude 1 V/m, phase 0) spans the period of a y-periodic strip (CPML along x, '
                    '10 nm mesh, 5,300 unknowns) and launches a plane wave along +x at the vacuum wavelength 1 um '
                    '(stationary study at the wavelength setting); a flux monitor 2.5 um downstream reports the '
                    'time-averaged power in W/m. Six runs: em_fdfd_ez and em_fdfd_hz, each in vacuum, with an eps_r = '
                    '4 half-space (n = 2) from 1 um before the monitor, continued into the PML, and with eps_r = 4 in '
                    'every cell (source in glass). Per polarisation: T = flux(glass) / flux(vacuum) must equal 4 n / '
                    '(n + 1)^2 = 8/9 at normal incidence (tolerance 0.01 absolute), and the absolute flux must equal '
                    'the source calibration, A^2 L / (2 eta0) in vacuum and n A^2 L / (2 eta0) in the all-glass run '
                    '(L = 0.1 um period; tolerance 0.01 relative each: a calibration with eta0 instead of the local '
                    'eta0 / n would be off by n^2). measured / expected = T of the worst polarisation and 8/9, error = '
                    'the worst deviation in tolerance units, tolerance 1. Budget: the discrete Fresnel coefficient at '
                    '50 cells per wavelength in the glass differs by O((k h)^2) ~ 5e-4 (H_z: the E_y face on the '
                    'interface takes the mean permittivity, the correct average for an E tangential to it); the '
                    'discrete Yee flux of a plane wave carries cos(k h / 2) (0.05% / 0.2% in vacuum / glass); CPML '
                    'reflections enter only through the round trip between the interface and the x_lo PML '
                    '(counter-propagating waves add their fluxes without cross terms) and, for the absolute flux, '
                    'through the x_lo echo of the backward wave (~2 |r|). cpu backend (not yet confirmed by a run '
                    'there).',
        specs=lambda: cpu_specs(_fresnel_specs()),
        check=_check_fresnel),
    SelfTestCase(
        name='em_fdfd_fresnel_transmission_graded',
        label='EM FDFD Fresnel transmission on a graded mesh',
        description=f'em_fdfd_fresnel_transmission (all six runs) on a {GRADED_FRESNEL_TEXT}; the absolute vacuum / '
                    'all-glass fluxes now also cross the graded cells. Same checks and tolerances as there; the '
                    'graded operator (A1 spacings on unequal widths) only refines the interface, so the uniform budget '
                    'is expected to hold (not yet confirmed by a run). cpu backend.',
        specs=lambda: cpu_specs(_graded_fresnel_specs()),
        check=lambda results_by_name: _check_fresnel(require_graded(results_by_name, _FRESNEL_CELL))),
    SelfTestCase(
        name='em_fdfd_mode_port_slab',
        label='EM FDFD mode ports: silicon slab waveguide',
        description='A mode source (mode 1, direction positive, 1 W/m, phase 0) launches the fundamental mode of a '
                    '0.28 um silicon slab (eps_r 12.08) in silica (eps_r 2.085) along +x in the 4 x 2 um all-CPML box '
                    'of em_mode_port_slab (20 nm mesh, 29,900 unknowns per run); the source and two mode monitors (0.5 '
                    'um upstream, 2 um downstream) span the window height, so every port solve continues through the '
                    'y PML. Two runs (cpu backend), em_fdfd_ez and em_fdfd_hz, each a frequency_domain study solving '
                    '1.5 and 1.6 um in that order (the source re-solves its mode at every wavelength; the monitors pick '
                    'mode 1 at 1.5 um and track it by overlap). At every wavelength the downstream fraction_positive must be '
                    '1 (lossless straight guide), every other fraction 0 (upstream fraction_negative: backward leakage '
                    'of the one-sided source plus the returning CPML reflection; upstream fraction_positive: its x_lo '
                    'CPML echo; downstream fraction_negative: the CPML reflection of the guided mode), and the '
                    'downstream n_eff must match the fundamental even mode of the symmetric slab, E_z: kappa '
                    'tan(kappa d / 2) = gamma, H_z: kappa tan(kappa d / 2) = (n1^2 / n2^2) gamma, kappa = k0 sqrt(n1^2 '
                    '- n_eff^2), gamma = k0 sqrt(n_eff^2 - n2^2). measured = error = max(|fraction_positive - 1| / '
                    '0.01, leak / 2e-3, |n_eff / n_analytic - 1| / 0.01) over both polarisations and the two '
                    'wavelengths, expected 0, tolerance 1; details list every item per polarisation and wavelength. '
                    'Budget: the source and monitor modes are the exact discrete eigenmodes of the line at each '
                    'wavelength (no numerical-frequency or band-edge mismatch as in FDTD), so forward 0.01 and leak '
                    '2e-3 leave room only for the guided-mode CPML reflection and solver round-off; n_eff 0.01 for '
                    'the discrete mode against the continuum: dispersion along x ((k dx)^2 / 24 ~ 0.25%) and the '
                    'transverse discretisation at 14 cells across the slab (~0.1%; H_z: the interface faces take the '
                    'mean permittivity). Not yet confirmed by a run with two wavelengths on the cpu.',
        specs=lambda: cpu_specs(_slab_specs()),
        check=_check_mode_port_slab),
    SelfTestCase(
        name='em_fdfd_pec_reflection_phase',
        label='EM FDFD wall, mirror and PEC planes',
        description=f'{WALL_REFLECTION_TEXT} Ten stationary runs (cpu backend): em_fdfd_ez and em_fdfd_hz with '
                    'bc_x_lo pec, pmc, symmetric, antisymmetric and a pec_domain block, line-source phase 0, probe '
                    'abs (H_z: E_y collocated from its two faces, cos(k dx/2) ~ 0.2%). Not yet confirmed by a run on '
                    'the cpu.',
        specs=lambda: cpu_specs(_wall_specs()),
        check=_check_wall_reflection),
    SelfTestCase(
        name='em_fdfd_mie_drude_cylinder',
        label='EM FDFD Drude cylinder: Mie and energy balance',
        description='A +x TFSF plane wave (amplitude 1 V/m, E_z or in-plane E_y) hits the Drude cylinder of '
                    'em_mie_drude_cylinder (r = 50 nm, omega_p 1.2e16 rad/s, gamma 1e14 rad/s, eps_inf 1, continuous '
                    'Drude chi at the solved omega) in a 0.4 x 0.4 um all-CPML box (2.5 nm mesh, 36,100 unknowns). '
                    'Four stationary runs (cpu backend): em_fdfd_ez and em_fdfd_hz at 0.6 and 0.9 um (the worst '
                    'items sit at 0.6 um; not yet confirmed by a run with these settings). An outer closed flux box '
                    'in the scattered field gives Q_sca = outward flux / (I 2a), an inner one in the total field Q_abs '
                    '= inward flux / (I 2a), I = A^2 / (2 eta0); they must match the Bohren & Huffman cylinder series '
                    '(case I for E_z, case II for H_z, m in the exp(-i omega t) convention), and the inward flux of '
                    'the inner box must equal the derived absorbed_line_power, absorbed_power integrated over the '
                    'cylinder cells (H_z: the face mean of |E|^2 times the cell -Im eps). measured = error = '
                    'max(err_sca / 0.10, err_abs / 0.15, err_balance / 1e-3) over both polarisations and the '
                    'wavelengths, expected 0, tolerance 1. Budget: Mie 10% / 15% for the staircased surface at 20 '
                    'cells per radius (H_z: corner field enhancement, arithmetic face mean of the normal E) and the '
                    'CPML only ~0.1 um from the cylinder in its near field; the balance is the discrete Poynting '
                    'identity of the solved system (complex64 fields, solver round-off). This pins the pole assembly '
                    '(harmonic.cell_permittivity), the FDFD TFSF plane wave and the H_z q_abs.',
        specs=lambda: cpu_specs(_drude_specs()),
        check=_check_drude_cylinder),
    SelfTestCase(
        name='em_fdfd_tfsf_graded_leakage',
        label='EM FDFD TFSF plane wave: empty box on graded meshes',
        description=f"A +x plane_wave_tfsf (1 V/m) in an empty vacuum box must leave no field outside it on graded "
                    f"meshes (solver_plan S7.1 lift: the incident line is marched on the solved stencil's x widths). "
                    f"Runs 'ez' / 'hz': a {_LEAK_SIZE * 1e6:g} x {_LEAK_SIZE * 1e6:g} um all-CPML box at the vacuum "
                    f"wavelength 1 um (stationary), mode 'nonuniform' base {_LEAK_CELL * 1e9:g} nm with a rectangle "
                    f'x 1.05..1.45, y 1.3..1.7 um refined to {_LEAK_REFINED_CELL * 1e9:g} nm ({_LEAK_MARGIN * 1e6:g} '
                    'um margin), so the lo-x edge of the box (1, 2) um lies on graded cells, the march crosses graded '
                    'columns and the box edges run along graded rows / columns; out-of-plane field probes 0.5 um '
                    "outside each side over one at the box centre. Runs 'oblique_ez' / 'oblique_hz': the "
                    'em_fdfd_oblique_tfsf strip (Bloch ky, full-period box) graded along x only by a vertical '
                    'refinement line at x = 1.5 um; upstream / downstream flux over the inside flux. measured = error '
                    f'= max(field ratio / {_LEAK_FIELD_TOLERANCE:g}, flux ratio / {_LEAK_FLUX_TOLERANCE:g}), expected 0, '
                    'tolerance 1: the TFSF correction is exact for any discrete homogeneous solution, so only solver '
                    'round-off remains (the closed-form uniform wave on these meshes leaks ~1e-3). Raises unless every '
                    'build is graded (dx_min <= 0.6 x its base cell). cpu backend. Not yet confirmed by a run.',
        specs=lambda: cpu_specs(_leakage_specs()),
        check=_check_tfsf_leakage),
    SelfTestCase(
        name='em_fdfd_line_block',
        label="EM FDFD line-block 'linear' solver vs sparse direct",
        description="solver_plan Part II stage 2 (default backend: the cupy line block is its own implementation): a "
                    '2 x 2 um copy of the em_fdfd_line_current_hankel box (ez / hz, 25 nm, 12,100 unknowns, probes '
                    'within 25 cells of the source; no periodic axis: the cheaper block axis) and a 2 um long '
                    'em_fdfd_fresnel_transmission glass strip (ez / hz, interface at mid-length, 2,300 unknowns; '
                    'periodic y: blocks along y, the wrap inside each block); the comparison is exact on any size, so '
                    'no closed form constrains them. Solved by the Automatic sparse_direct and '
                    "by a line_block solver node assigned as the 'linear' solver, precision double and mixed (12 "
                    'runs). Per model and precision max |line_block - sparse_direct| / max |sparse_direct| of the '
                    "probe phasors (Hankel) or the transmitted flux (Fresnel) must stay below "
                    f"{_LINE_BLOCK_TOLERANCES['double']:g} (double) / {_LINE_BLOCK_TOLERANCES['mixed']:g} (mixed: "
                    'complex64 block factors refined against the complex128 operator), and the run summary must record '
                    "the linear role as line_block with rejected_factor_solves 0 (a rejected solve falls back to a "
                    'host SuperLU and would hide the line-block result). measured = error = the worst item in '
                    'tolerance units, tolerance 1. Not yet confirmed by a run with the smaller models.',
        specs=_line_block_specs,
        check=_check_line_block),
    SelfTestCase(
        name='em_fdfd_line_block_wavelengths',
        label='EM FDFD line-block wavelength batches vs sparse direct',
        description='solver_plan Part II stage 3: the em_fdfd_line_block Hankel ez box and Fresnel ez / hz glass strips '
                    '(its smaller models) '
                    f"in a frequency_domain study of {_BATCH_STUDY['n_wavelengths']} wavelengths "
                    f"{_BATCH_STUDY['wavelength_start'] * 1e6:g}-{_BATCH_STUDY['wavelength_stop'] * 1e6:g} um, solved "
                    'by the Automatic sparse_direct and by line_block linear nodes (double, mixed; 9 runs). The line '
                    'block factorises the planned wavelengths as one batch of whole operators (factorize_operators; '
                    'the Helmholtz operator is not affine in omega), one shift per wavelength, and solves each '
                    'wavelength from it. Per model and precision max |line_block - sparse_direct| / max '
                    '|sparse_direct| of the probe phasors / transmitted flux over every wavelength must stay below '
                    f"{_LINE_BLOCK_TOLERANCES['double']:g} (double) / {_LINE_BLOCK_TOLERANCES['mixed']:g} (mixed); "
                    'the run summary must record the linear role as line_block with batched_operators = the '
                    'wavelength count (every wavelength from a batch; a smaller count means the free memory held '
                    'fewer than all of them) and rejected_factor_solves 0. measured = error = the worst item in '
                    'tolerance units, tolerance 1. Not yet confirmed by a run.',
        specs=_line_block_wavelength_specs,
        check=_check_line_block_wavelengths),
    SelfTestCase(
        name='em_fdfd_cudss',
        label="EM FDFD cuDSS 'linear' back ends vs SuperLU",
        description='The em_fdfd_line_block_wavelengths models (Hankel ez box, Fresnel ez / hz glass strips with a '
                    'periodic y axis) in the same frequency_domain study of '
                    f"{_BATCH_STUDY['n_wavelengths']} wavelengths, solved by the Automatic sparse_direct (host "
                    "SuperLU) and by a sparse_direct node with back_end cudss and cudss_mixed assigned as the 'linear' "
                    'solver (9 runs, gpu). cuDSS factors D A (D the stretched cell areas: complex symmetric) as an '
                    'LDL^T and refactorises the first wavelength\'s factor in place for the second (one analysis). '
                    'Per model and back end max |cuDSS - SuperLU| / max |SuperLU| of the probe phasors / transmitted '
                    f'flux over every wavelength must stay below {_CUDSS_TOLERANCE:g}, and the run summary must '
                    'record the linear role as sparse_direct on that back end with factorizations = cudss_symmetric_'
                    'factors = the wavelength count and cudss_plans 1. measured = error = the worst item in '
                    'tolerance units, tolerance 1; passes as skipped without nvmath-python. Run on an RTX 3050: '
                    'cudss 2e-14 - 4e-14, cudss_mixed 9e-11 - 5e-10 (worst fresnel_hz), 2 factorisations on 1 plan '
                    'per run, 4.5 s.',
        specs=_cudss_specs,
        check=_check_cudss),
    SelfTestCase(
        name='em_fdfd_host_direct',
        label="EM FDFD MUMPS / PARDISO 'linear' and 'eigen' back ends vs SuperLU",
        description='cpu backend; per sparse_direct back end / eigen factorization of every installed host direct '
                    'library (python-mumps: mumps, mumps_mixed; pydiso: pardiso, pardiso_mixed): (1) the em_fdfd_cudss '
                    'models (Hankel ez box, Fresnel ez / hz glass strips) in its frequency_domain study of '
                    f"{_BATCH_STUDY['n_wavelengths']} wavelengths with that back end on the 'linear' node, against "
                    f"the same runs on a '{_HOST_REFERENCE}' node: the host factor is the LDL^T of D A (D the stretched "
                    'cell areas), one per wavelength; (2) the em_fdfd_adjoint_gradient ez run weighting the reflection '
                    'objective (eps_sensitivity: the transposed solve x = D R solve(R b) on that factor) against its '
                    f"'{_HOST_REFERENCE}' twin; (3) the lossy ez PEC cavity of em_fdfd_eigenfrequency_pec_cavity "
                    "(sigma_e 10 S/m) with an arpack_shift_invert node of that factorization as 'eigen' (its own "
                    'LDL^T of D A; the mixed ones on 10 nm cells, 24,321 unknowns, above the 20,000 below which a '
                    'mixed request factors in double, so the loose solves and Rayleigh quotients run), against the '
                    'discrete closed form. Errors: max |host - SuperLU| / max |SuperLU| of the probe phasors / '
                    'transmitted flux over every wavelength and the relative adjoint block-gradient difference, below '
                    f"{_HOST_TOLERANCES['double']:g} (double) / {_HOST_TOLERANCES['mixed']:g} (mixed: complex64 "
                    'factor + float64 GMRES refinement); the cavity omega / Q to the em_fdfd_eigenfrequency_pec_cavity '
                    'tolerances; the run summaries must record the linear role as sparse_direct on that back end with '
                    'factorizations = host_direct_symmetric_factors = the wavelength count and no double fallback, the '
                    'eigen role with that factorization, eigen_factor_symmetric 1 and no double fallback. measured = '
                    'error = the worst item in tolerance units, tolerance 1; passes as skipped without both '
                    'libraries. Not yet confirmed by a run (heavy by estimate: 5 runs per back end plus 4 references; '
                    'record the wall time).',
        specs=_host_direct_specs,
        check=_check_host_direct),
)

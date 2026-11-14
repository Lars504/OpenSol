"""Model-layer definition of the 'membrane' physics type (new_modules_plan S4 / S7: a tensioned film, optionally with
plate bending, on triangle meshes and in 1D models; settings, tension / areal density / loss / bending vocabulary,
clamps, supports, loads, probes, ports, mode projections, the resonance table, PML padding, Bloch cells, validation,
memory and run-time estimates) plus its template and selftest loader."""
import cmath
import math
from dataclasses import dataclass

import numpy as np

from ...core import time_estimate as te
from ...core.constants import FACES
from ...core.harmonic_probe import harmonic_probe_result
from ...core.mirror import mirror_faces
from ...core.memory_estimate import (COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES, MemoryEstimate,
                                     host_lu_bytes, window_cell_count)
from ...fem.diffusion import FEM_FILL_FACTOR, lagrange_entries_per_dof
from ...fem.harmonic_resonances import resonance_table
from ...fem.pml import DEFAULT_PML_REFLECTION
from ...fem.squeeze_film import GAS_TABLE_COLUMNS, GAS_TABLE_UNITS, gas_damping_totals
from ...model.definition_helpers import (DIFFUSION_DIRECT_BACK_ENDS, DISCRETIZATION_SETTING,
                                         KEPT_SOLUTION_BYTES_PER_DOF, SQUEEZE_FILM_GAS_PARAMS,
                                         batched_direct_back_end, batched_direct_items, batched_direct_sweep,
                                         cudss_external,
                                         eigen_factorization, eigen_role, eigen_search_counts, element_order,
                                         evaluates_to, harmonic_frequency_items, host_direct_back_end, on_triangles,
                                         scalar_harmonic_build_items, optional_material_param,
                                         reduced_sweep_memory_items, reports_model_checks, resolved_back_end,
                                         selected_material_data, squeeze_film_gas_issues, validate_points_inside)
from ...model.geometry import expand_entities
from ...model.schema import (DomainRequirements, FeatureType, Issue, Param, PhysicsType, SolverRole, feature_path,
                             resolve_values, solver_backend)
from ...solvers.arpack_shift_invert import (CUDSS_FACTORIZATIONS, DEVICE_PENCIL_MIN_ENTRIES, MIXED_EIGEN_MIN_UNKNOWNS,
                                            arnoldi_basis_bytes, arnoldi_basis_size)
from ...solvers.cudss_factor import operator_copy_bytes, solve_buffer_bytes, solve_peak_bytes
from ...solvers.host_direct import (double_of, factor_label, host_direct_precision, solver_host_bytes,
                                    superlu_bytes_scale)
from ...solvers.sparse_direct import (CUDSS_BACK_ENDS, SPLU_LIMIT_UNKNOWNS, SPLU_WARN_UNKNOWNS, device_pencil_bytes,
                                      scipy_available)
from .materials import (BENDING_ARRAY, COUPLED_TENSION_SLOTS, LOSS_ARRAY, MEMBRANE_PROPERTY_ARRAYS, MEMBRANE_SCHEMA,
                        OPTIONAL_MEMBRANE_SCHEMA)
from .bending_fem import CHUNK_ITEMS, EDGE_DEGREE
from .ports import PORT_UNIFORM_RTOL, port_result, unfolded_modes
from .renderer import MEMBRANE_FIELD_INFOS, STATE_FIELD_NAMES, shown_label
from .selftests import RECTANGLE_DESCRIPTION, RECTANGLE_LABEL, rectangle_modes_spec
from .solver_common import DISCRETIZATIONS, ResonanceTableMonitor
from .squeeze_film import MEMBRANE_VENT_CHOICES

MEMBRANE_TYPE_NAME = 'membrane'   # = selftests._MEMBRANE_TYPE (selftests never import this)
EIGENFREQUENCY_STUDY = 'eigenfrequency'
FREQUENCY_RESPONSE_STUDY = 'frequency_response'
_GRIDS = ('tensor', 'triangles')   # tensor = 1D models only (validation refuses a 2D tensor mesh)
DEFAULT_PML_CELLS = 8
_PML_FACE_SETTINGS = tuple(f'pml_{face}' for face in FACES)
# 'linear' role (the driven solve of a frequency_response study): sparse_direct, the cuDSS back ends (gpu backend,
# triangle meshes: LDL^T of the complex symmetric operator, refactorised per frequency on the kept plan); back_end
# 'auto' = cudss_mixed there on the gpu backend with nvmath-python, else host SuperLU; cusolver_qr refused (the membrane
# factorises host matrices only); the opt-in MUMPS / PARDISO back ends factorise the host A(omega) on triangles (LDL^T,
# analysis reused per frequency); a 1D model keeps the host SuperLU factor (validation refuses cuDSS / MUMPS / PARDISO
# nodes there; 'auto' stays SuperLU on a line); opt-in batched_direct (triangles: the direct sweep's frequencies in
# batches, fem.harmonic plan_batched; 'mixed' is listed for it only, sparse_direct offers double alone)
MEMBRANE_LINEAR_ROLE = SolverRole('general', 'sparse_direct', studies=(FREQUENCY_RESPONSE_STUDY,),
                                  precisions=('double', 'mixed'),
                                  setting_choices={'back_end': DIFFUSION_DIRECT_BACK_ENDS},
                                  auto_back_ends={'triangles': 'cudss_mixed'}, opt_in_types=('batched_direct',))


# ---- settings ----
_DISCRETIZATION_PARAM = Param(
    DISCRETIZATION_SETTING, 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATIONS),
    choice_labels=(('quadratic', 'Quadratic elements'), ('linear', 'Linear elements')),
    help='Lagrange order of w: quadratic (P2, default) or linear (P1), on the triangles of a 2D model (P2 elements with '
         'an edge on a circle / ellipse follow the drawn curve, isoparametric) and on the cells of a 1D model (line '
         'elements on the cell edges). Resonance errors fall as (k h)^4 for P2 and (k h)^2 for P1 (k = 2 pi f / c, c = '
         'sqrt(T / rho_s)).',
    tip='How the displacement varies inside each element: quadratic is far more accurate per element, linear is '
        'cheaper.')
_TARGET_FREQUENCY_PARAM = Param(
    'target_frequency', 'float', 0.0, label='Eigenfrequency target', unit='Hz', min=0.0,
    help='Eigenfrequency study only: the resonances nearest this frequency are searched (shift-invert about (2 pi '
         'target_frequency)^2, moved 1e-4 above it so a target on a resonance still factors); 0 = the lowest ones, '
         'which needs a clamped boundary (a free membrane moves without strain at 0 Hz). With a PML it also sets the '
         'PML strength unless pml_frequency is given, so it must then be > 0. Ignored by frequency_response.',
    tip="Used only by an 'Eigenfrequency' study: the resonances closest to this frequency are found.\n\n"
        '0 finds the lowest ones.')
_AXIAL_WAVENUMBER_PARAM = Param(
    'axial_wavenumber', 'float', 0.0, label='Axial wavenumber', unit='rad/m',
    help='1D models only: w ~ W(x) exp(-j k_y y), the line being the cross-section of a membrane guide along y; adds '
         'T_yy (1 + j eta) k_y^2 to the operator, so a strip of width W clamped at both ends has the branches omega_n = '
         'c sqrt((n pi / W)^2 + k_y^2), c = sqrt(T / rho_s) (sweep a parameter for the dispersion; below the cut-off '
         'n pi c / W a guided wave of frequency f has imaginary k_y). With a bending_stiffness D the plate strip operator D '
         '(1 + j eta) (W\'\'\'\' - 2 k_y^2 W\'\' + k_y^4 W) is added. 0 (default) = a line uniform along y '
         '(a string or a strip). A 2D model refuses a nonzero value (its y is drawn; use periodic_y with bloch_ky there).',
    tip='Only in a 1D model: the wave number along the strip, for the dispersion of a membrane waveguide.\n\n'
        '0 means a strip that does not vary along its length.')


def _pml_params():
    """pml_<face> per outer face, pml_cells, pml_reflection and pml_frequency (triangle meshes)."""
    faces = tuple(Param(
        f'pml_{face}', 'bool', False, label=f'PML on {face}',
        help=(f'Triangle meshes only: a perfectly matched layer beyond {face}. The mesh grows a ring of pml_cells x '
              'the mesh dx there (the MeshSpec pml_layers element layers across), its material continued from the '
              'window edge, with the fem.pml stretch s = 1 - j beta (d / w)^2 (d the depth, w the thickness); beta is '
              'set by pml_reflection for the fastest wave of the ring at the reference frequency omega_r '
              '(pml_frequency, else: eigenfrequency target_frequency, frequency_response the first frequency '
              'solved): c = sqrt(T / rho_s), with bending_stiffness D the propagating plate wave c^2 = (T + sqrt(T^2 + '
              '4 D rho_s omega_r^2)) / (2 rho_s) (its evanescent partner needs no extra strength); its outer side is '
              'clamped (w = 0, with bending also dw/dn = 0). With bending the ring must resolve the boundary layer '
              'sqrt(D / T) like the window and needs one element layer per cell (MeshSpec pml_layers >= pml_cells; '
              'coarser layers reflect strongly, validation warns). Waves leaving through '
              f'{face} are absorbed: resonances get frequency_imag > 0 and a radiation Q, resonances with more than '
              'half of their energy in the layer are dropped as PML modes. Waves far below the reference frequency are '
              f'absorbed less. Keep clamps, line forces and probes off {face}; every physics sharing the mesh must ring '
              'the same faces.'),
        tip=(f'Absorbs waves leaving through {face}, as if the membrane continued forever there.\n\n'
             'Gives each resonance a radiation quality factor. Only on a triangle mesh.'))
        for face in FACES)
    cells = Param('pml_cells', 'int', DEFAULT_PML_CELLS, label='PML thickness (elements)', unit='1', min=1, max=100,
                  help='Thickness of the PML ring in multiples of the mesh dx (h_max) on every pml_<face> face; the '
                       'stretch profile wants the ring a fair part of the longest wavelength c / f there (6-12 typical).',
                  tip='How thick the absorbing layer is, in element sizes.\n\n8 is typical.')
    reflection = Param('pml_reflection', 'float', DEFAULT_PML_REFLECTION, label='PML reflection', unit='1', min=0.0,
                       min_exclusive=True, max=1.0,
                       help='Theoretical round-trip reflection of the PML for a normally incident wave of the fastest '
                            'speed c of the ring (pml_<face>) at the reference frequency (beta = 3 ln(1 / R) c / (2 '
                            'omega_r w)); oblique '
                            'and slower waves are absorbed more in depth, less at grazing incidence. A very small value '
                            'makes the profile steep, which the discretisation reflects instead: 1e-4 to 1e-8 typical.',
                       tip='How strongly the absorbing layer damps a wave crossing it and back.\n\n1e-6 is typical.')
    frequency = Param('pml_frequency', 'float', None, label='PML reference frequency', unit='Hz', min=0.0,
                      min_exclusive=True, optional=True,
                      help='Frequency [Hz] at which the PML strength is set (fixed for the whole study, so a resonance '
                           'pencil stays linear in omega^2). Empty: the eigenfrequency target_frequency, or in a '
                           'frequency_response study the first frequency solved (put the lowest first for the strongest '
                           'absorption over the sweep).',
                      tip='Frequency the absorbing layer is tuned for; empty uses the eigenfrequency target or the '
                          'first frequency of the sweep.')
    return faces + (cells, reflection, frequency)


_AXES = ('x', 'y')


def _periodic_params():
    """periodic_x / _y and bloch_kx / _ky (Bloch cells on periodic triangle meshes, eigenfrequency)."""
    periodic = tuple(Param(
        f'periodic_{axis}', 'bool', False, label=f'Periodic {axis}',
        help=(f'Triangle mesh, eigenfrequency study only: {axis}_lo and {axis}_hi are one Bloch-periodic pair, w on '
              f'{axis}_hi equals w on {axis}_lo times exp(-j bloch_k{axis} L) (L = the window size along {axis}, exp(+j '
              'omega t)); the mesh is built with matching nodes on the two sides (no PML ring on them). The modes come '
              'from the reduced pencil P^H K P, P^H M P (complex Hermitian; real when every k L is a multiple of pi) '
              'and are complex: w_re / w_im / w_abs show them. With bending the slope jump across the seam is '
              'penalised like any interior edge. Clamps, supports, line forces and ports must stay off the periodic '
              'faces; with every bloch_k 0 a cell without any clamp moves at 0 Hz: set target_frequency > 0.'),
        tip=(f'Makes the {axis} faces periodic: the model is one cell of an infinite periodic membrane (a phononic '
             'crystal, a perforated film, a strip of an infinite guide).\n\nOnly for an eigenfrequency study on a '
             'triangle mesh.')) for axis in _AXES)
    bloch = tuple(Param(
        f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
        visible_if=((f'periodic_{axis}', (True,)),),
        help=(f'Bloch wave vector along {axis} [rad/m], used only when periodic_{axis} is on: the phase step exp(-j '
              f'k{axis} L) per period L; k{axis} and k{axis} + 2 pi / L are the same state; 0 = plain periodic. Swept '
              'through a parameter, one eigenfrequency study per k gives the band diagram (a strip clamped on two '
              'faces and periodic along the third: the guide dispersion f_n(k)). Typical: pi / L at the zone edge.'),
        tip=('Phase step of the vibration from one period to the next along this axis, in radians per metre.\n\n0 '
             'means no phase step. Swept, it gives the bands of a periodic membrane.')) for axis in _AXES)
    return periodic + bloch


_SETTINGS = (_DISCRETIZATION_PARAM, _TARGET_FREQUENCY_PARAM, _AXIAL_WAVENUMBER_PARAM, *_pml_params(),
             *_periodic_params())


def _pml_faces(settings_values) -> tuple:
    """Outer faces whose pml_<face> setting is on."""
    return tuple(face for face in FACES if settings_values and settings_values.get(f'pml_{face}'))


def _periodic_axes(settings_values) -> tuple:
    """Axes ('x', 'y') whose periodic_<axis> setting is on."""
    return tuple(axis for axis in _AXES if settings_values and settings_values.get(f'periodic_{axis}'))


def _bloch_wavenumber(settings_values, axis) -> float:
    value = (settings_values or {}).get(f'bloch_k{axis}', 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class MembranePadding:
    """DomainRequirements.boundary_config of the membrane PML: npml_sides = ring thickness in mesh dx per face."""
    npml_sides: dict

    def differences(self, other) -> tuple:
        """Names that differ from another config ('type' for another class)."""
        if not isinstance(other, MembranePadding):
            return ('type',)
        return ('npml_sides',) if dict(self.npml_sides) != dict(other.npml_sides) else ()


def _domain_requirements(settings_values):
    """The PML ring (pml_cells x dx on the pml_<face> faces) and the periodic axes (matching side nodes) of a triangle
    mesh."""
    faces = _pml_faces(settings_values)
    padding = None
    if faces:
        cells = int(settings_values.get('pml_cells') or DEFAULT_PML_CELLS)
        padding = MembranePadding({face: cells if face in faces else 0 for face in FACES})
    return DomainRequirements(boundary_config=padding, periodic_axes=_periodic_axes(settings_values))


# ---- material vocabulary ----
def _material_property(name, label, unit, minimum, help_text, tip):
    return optional_material_param(name, label, unit, MEMBRANE_PROPERTY_ARRAYS[name], minimum, help_text, tip)


_TENSION_HELP = (
    'In-plane tension of the film [N/m] = film stress x thickness (sigma h; e.g. 1 GPa x 50 nm silicon nitride = 50 '
    'N/m), isotropic: writes tension_xx and tension_yy. The wave speed is c = sqrt(T / areal_density). An element with '
    'no tension and no areal_density is a hole (not solved, its outline a free edge); the coupled tension slots add to '
    'it. Negative (compressive) values make the membrane unstable (no restoring force): resonances with Re omega <= 0 '
    'are dropped.')
_MATERIAL_PROPERTIES = (
    _material_property('tension', 'Tension', 'N/m', None, _TENSION_HELP,
                       'Pulling force per width of the film: film stress times thickness.\n\n'
                       'Typical: 50 N/m for 50 nm of silicon nitride at 1 GPa.'),
    _material_property('tension_xx', 'Tension xx', 'N/m', None,
                       'Tension component N_xx [N/m]; overrides tension along x (an anisotropic or coupled prestress).',
                       "Tension along x; replaces 'Tension' there."),
    _material_property('tension_yy', 'Tension yy', 'N/m', None,
                       'Tension component N_yy [N/m]; overrides tension along y. 1D models: the tension along the '
                       'guide, read only by the axial_wavenumber term.',
                       "Tension along y; replaces 'Tension' there."),
    _material_property('tension_xy', 'Tension xy', 'N/m', None,
                       'In-plane shear tension N_xy [N/m] (unset = 0; the array is allocated only when a material sets '
                       'it); the tensor [[N_xx, N_xy], [N_xy, N_yy]] should be positive definite. Ignored in 1D models.',
                       'Shear part of the tension; empty means none.'),
    _material_property('areal_density', 'Areal density', 'kg/m^2', 0.0,
                       'Mass per area rho_s = rho h [kg/m^2] (e.g. 3100 kg/m^3 x 50 nm silicon nitride = 1.55e-4). '
                       'Unset = 0: with no tension either the element is a hole.',
                       'Mass per area of the film: density times thickness.\n\n'
                       'Typical: 1.55e-4 kg/m^2 for 50 nm of silicon nitride.'),
    _material_property('loss_factor', 'Loss factor', '1', 0.0,
                       'Isotropic structural loss factor eta: the tension becomes T (1 + j eta) (exp(+j omega t)), so a '
                       'resonance has Q = 1 / eta (exactly Re s / (2 Im s), s = sqrt(1 + j eta)). Unset = 0 (the array '
                       'is allocated only when a material sets it). Typical 1e-6 to 1e-3.',
                       'Internal damping of the film; the quality factor of a resonance is about 1 over it.\n\n'
                       'Empty means no loss.'),
    _material_property('bending_stiffness', 'Bending stiffness', 'N m', 0.0,
                       'Plate bending stiffness D = E h^3 / (12 (1 - nu^2)) [N m] (e.g. 250 GPa, 80 nm, nu 0.23: 1.13e-11 '
                       'N m), adding D (1 + j eta) del^4 w to the membrane equation (a tensioned Kirchhoff plate). Setting '
                       'it in any material (0 included) switches the whole physics to plate elements: Hermite cubic line '
                       'elements in 1D (w and w\'), on triangles the C0 interior penalty form on quadratic elements '
                       '(discretization quadratic; a PML stretches the plate form too; a port on a face with D > 0 '
                       'takes the numeric plate strip modes; mode_projection is refused). Then clamped also '
                       'holds the slope dw/dn = 0, '
                       'simply_supported (and antisymmetry) holds w only, symmetry the slope only. The clamped boundary layer sqrt(D / T) (0.38 um for the example at '
                       '80 N/m) must be resolved by the mesh near clamps.',
                       'Bending stiffness of the film: Young modulus times thickness cubed over 12 (1 - Poisson ratio '
                       'squared).\n\nEmpty means a membrane without bending.'),
    _material_property('plate_poisson_ratio', 'Plate Poisson ratio', '1', None,
                       'Poisson ratio nu of the plate bending moments M = D [(1 - nu) H(w) + nu tr H(w) I] (triangle '
                       'meshes; unset = 0). It changes free and simply supported plate edges only (the moment there), '
                       'not clamped ones or the interior equation; D already holds its 1 / (1 - nu^2). 1D models ignore '
                       'it (exact for clamped and simply supported ends). Must lie in (-1, 0.5].',
                       'Poisson ratio used in the bending moments of a plate; only matters at free edges.'),
)

_SLOT_DOCS = {
    'coupled_tension_xx': ('N/m', 'Coupled tension N_xx per element (1D: per cell), e.g. the relaxed prestress h '
                                  'sigma_xx of a solid; added to the material tension_xx; each transfer replaces it.'),
    'coupled_tension_yy': ('N/m', 'Coupled tension N_yy, as coupled_tension_xx.'),
    'coupled_tension_xy': ('N/m', 'Coupled shear tension N_xy, as coupled_tension_xx (ignored in 1D models).'),
}


# ---- feature callbacks ----
def _apply_clamped(ctx, solver, feature_spec, values, selection):
    for face in selection.outer_faces:
        solver.clamp_face(face)
    if selection.objects:
        if ctx.domain.grid_kind != 'triangles':
            raise ValueError(f"clamped '{feature_spec.name}': a 1D membrane is clamped at its ends x_lo / x_hi only")
        solver.clamp_edges(ctx.boundary_edges(selection.objects))
    return None


def _apply_simply_supported(ctx, solver, feature_spec, values, selection):
    """simply_supported and antisymmetry (the odd mirror: w = 0, slope free)."""
    for face in selection.outer_faces:
        solver.support_face(face)
    if selection.objects:
        if ctx.domain.grid_kind != 'triangles':
            raise ValueError(f"{feature_spec.type} '{feature_spec.name}': a 1D membrane is held at its ends x_lo / "
                             'x_hi only')
        solver.support_edges(ctx.boundary_edges(selection.objects))
    return None


def _apply_antisymmetry(ctx, solver, feature_spec, values, selection):
    """The odd mirror: the simply_supported hold; on triangles its faces also vent a squeeze film (odd pressure)."""
    _apply_simply_supported(ctx, solver, feature_spec, values, selection)
    if ctx.domain.grid_kind == 'triangles':
        solver.mark_odd_mirror_faces(selection.outer_faces)
    return None


def _apply_squeeze_film(ctx, solver, feature_spec, values, selection):
    """squeeze_film.MembraneSqueezeFilm on the selected domains, registered with the triangle solver (evaluated per
    resonance after each eigen solve)."""
    if ctx.domain.grid_kind != 'triangles':
        raise ValueError(f"squeeze_film_damping '{feature_spec.name}': triangle meshes only")
    from .squeeze_film import MembraneSqueezeFilm
    ambient = values['ambient_pressure']
    film = MembraneSqueezeFilm(feature_spec.name, ctx.domain_mask(selection.entities), values['gap'],
                               values['viscosity'], ambient if ambient > 0.0 else None,
                               mean_free_path=values['mean_free_path'], slip=values['slip'], vents=values['vents'])
    return solver.add_squeeze_film(film)


def _squeeze_film_result(session, solver, film, feature_spec, options):
    return film.table()


def _apply_symmetry(ctx, solver, feature_spec, values, selection):
    """The even mirror: nothing on a film (its free edge), the slope dw/dn = 0 alone on a plate."""
    for face in selection.outer_faces:
        solver.symmetry_face(face)
    if selection.objects:
        if ctx.domain.grid_kind != 'triangles':
            raise ValueError(f"symmetry '{feature_spec.name}': a 1D membrane has mirror planes at its ends x_lo / x_hi "
                             'only')
        solver.symmetry_edges(ctx.boundary_edges(selection.objects))
    return None


def _apply_port(ctx, solver, feature_spec, values, selection):
    faces = list(selection.outer_faces)
    if ctx.domain.grid_kind != 'triangles' or len(faces) != 1 or selection.objects:
        raise ValueError(f"port '{feature_spec.name}': one outer face of a triangle mesh")
    return solver.add_port(feature_spec.name, faces[0], values['mode_count'], values['incident_mode'],
                           values['amplitude'], values['phase'], values['tension_tolerance'],
                           incidents=_port_incidents(values), sector_drive=bool(values['sector_drive']))


def _port_incidents(values):
    """[(unfolded mode, amplitude [m], phase [deg])] of incident_modes (amplitude / phase lists or the single
    values), None without the list (incident_mode then)."""
    modes = values.get('incident_modes') or []
    if not modes:
        return None
    amplitudes = values.get('incident_amplitudes') or [values['amplitude']] * len(modes)
    phases = values.get('incident_phases') or [values['phase']] * len(modes)
    if len(amplitudes) != len(modes) or len(phases) != len(modes):
        raise ValueError(f'port: incident_amplitudes / incident_phases need one entry per incident mode '
                         f'({len(modes)})')
    return [(int(round(mode)), float(amplitude), float(phase)) for mode, amplitude, phase in zip(modes, amplitudes,
                                                                                                  phases)]


def _port_feature_result(session, solver, name, feature_spec, options):
    """port_result plus the extras tension_spread / tension_tolerance / medium_warning / free_walls and the
    mirror-sector extras mirror_walls / sector_modes / sector_signs / sector_drive_scale (solver.port_medium)."""
    result = port_result(*solver.port_records(name))
    result.update(solver.port_medium(name))
    return result


@dataclass(frozen=True, eq=False)
class ModeProjectionMonitor:
    """Runtime of a mode_projection feature: guide axis (0 x / 1 y), cross-section [m], counts, the located sample
    points (ports.ModeProjector), the per-frequency record (ports.ProjectionRecord), result axis and averaged points."""
    name: str
    guide_axis: int
    cross_start: float
    cross_end: float
    mode_count: int
    projector: object
    record: object
    result_axis: str = 'position'      # 'position' | 'frequency' (window mean of |a_n|^2 per recorded frequency)
    averaged_points: object = None     # bool (points,): positions inside the averaging window


def _averaged_points(name, positions, values) -> np.ndarray:
    """Bool mask of the sample positions inside [average_start, average_end] (empty ends: the projection range)."""
    tolerance = 1e-9 * max(abs(float(positions[-1]) - float(positions[0])), 1e-300)
    low = float(positions[0]) if values['average_start'] is None else float(values['average_start'])
    high = float(positions[-1]) if values['average_end'] is None else float(values['average_end'])
    inside = (positions >= low - tolerance) & (positions <= high + tolerance)
    if not inside.any():
        raise ValueError(f"mode_projection '{name}': no sampled position lies in the averaging window [{low:g}, "
                         f'{high:g}] m (positions {float(positions[0]):g} .. {float(positions[-1]):g} m)')
    return inside


def _apply_mode_projection(ctx, solver, feature_spec, values, selection):
    if ctx.domain.grid_kind != 'triangles':
        raise ValueError(f"mode_projection '{feature_spec.name}': triangle meshes only")
    x0, y0, x1, y1 = ctx.domain.mesh.window
    axis = _AXES.index(values['guide_axis'])
    along, cross = ((x0, x1), (y0, y1)) if axis == 0 else ((y0, y1), (x0, x1))
    inset = 1e-9 * (along[1] - along[0])   # sample points on the window edge stay inside the mesh

    def chosen(name, default):
        return float(default) if values[name] is None else float(values[name])
    cross_start, cross_end = chosen('cross_start', cross[0]), chosen('cross_end', cross[1])
    positions = np.linspace(chosen('start', along[0] + inset), chosen('end', along[1] - inset), int(values['points']))
    projector, record = solver.add_mode_projection(feature_spec.name, axis, cross_start, cross_end, positions,
                                                   int(values['mode_count']))
    return ModeProjectionMonitor(feature_spec.name, axis, cross_start, cross_end, int(values['mode_count']), projector,
                                 record, values['result_axis'],
                                 _averaged_points(feature_spec.name, positions, values)
                                 if values['result_axis'] == 'frequency' else None)


def _projection_series(projection) -> tuple:
    """({mode_<n>_re / _im / _abs / _phase: real array}, units) of complex projections (mode_count, ...) [m]."""
    series, units = {}, {}
    for number, values in enumerate(projection, start=1):
        for part, data in (('re', values.real), ('im', values.imag), ('abs', np.abs(values)),
                           ('phase', np.angle(values))):
            series[f'mode_{number}_{part}'] = np.ascontiguousarray(data)
            units[f'mode_{number}_{part}'] = 'rad' if part == 'phase' else 'm'
    return series, units


def _projection_power_result(monitor, feature_spec) -> dict:
    """result_axis 'frequency': per recorded frequency the mean of |a_n|^2 over the averaged positions [m^2]."""
    if not len(monitor.record):
        raise ValueError(f"mode_projection '{feature_spec.name}': result_axis 'frequency' needs a frequency_response "
                         'run (no frequency recorded yet)')
    frequencies, projections = monitor.record.recorded()   # (n_frequencies, mode_count, points)
    mode_power = np.mean(np.abs(projections[:, :, monitor.averaged_points]) ** 2, axis=2)   # (n_frequencies, modes)
    series, units = {}, {}
    for number in range(1, mode_power.shape[1] + 1):
        power = np.ascontiguousarray(mode_power[:, number - 1])
        series[f'mode_{number}_power'], units[f'mode_{number}_power'] = power, 'm^2'
        series[f'mode_{number}_rms'], units[f'mode_{number}_rms'] = np.sqrt(power), 'm'
    series['total_power'], units['total_power'] = np.ascontiguousarray(mode_power.sum(axis=1)), 'm^2'
    averaged = monitor.projector.positions[monitor.averaged_points]
    window = projections[:, :, monitor.averaged_points]   # complex (n_frequencies, modes, averaged points)
    averaged_series = {f'mode_{number}_{part}': np.ascontiguousarray(getattr(window[:, number - 1, :], attribute))
                       for number in range(1, window.shape[1] + 1)
                       for part, attribute in (('re', 'real'), ('im', 'imag'))}
    return {'kind': 'xy', 'x': frequencies, 'x_label': 'Frequency', 'x_unit': 'Hz',
            'y_label': 'window mean of |a_n|^2', 'series': series, 'series_units': units,
            'cross_section': [monitor.cross_start, monitor.cross_end], 'guide_axis': _AXES[monitor.guide_axis],
            'free_walls': list(monitor.projector.free_walls),
            'average_window': [float(averaged[0]), float(averaged[-1])], 'average_points': int(averaged.size),
            'averaged_series': averaged_series, **_projection_sector_extras(monitor)}


def _projection_sector_extras(monitor) -> dict:
    """Mirror-sector extras of a mode projection: mirror_walls and its modes on the unfolded guide (sector_modes /
    sector_signs, ports.unfolded_modes; core.mirror.combine_sector_modes)."""
    numbers, signs = unfolded_modes(monitor.projector.mirror_walls, monitor.mode_count)
    return {'mirror_walls': list(monitor.projector.mirror_walls), 'sector_modes': numbers.tolist(),
            'sector_signs': signs.tolist()}


def _mode_projection_result(session, solver, monitor, feature_spec, options):
    """The shown solution's a_n(p) vs position; after a frequency response also frequencies [Hz] and
    frequency_series {mode_<n>_<part>: (n_frequencies, points)}. result_axis 'frequency': _projection_power_result."""
    if monitor.result_axis == 'frequency':
        return _projection_power_result(monitor, feature_spec)
    if solver.selected_resonance is None and solver.response_frequency is None:
        raise ValueError(f"mode_projection '{feature_spec.name}': not solved yet (run an eigenfrequency or "
                         'frequency_response study)')
    series, units = _projection_series(monitor.projector.apply(solver.values))
    axis_name = _AXES[monitor.guide_axis]
    result = {'kind': 'xy', 'x': monitor.projector.positions.copy(), 'x_label': f'Position {axis_name}', 'x_unit': 'm',
              'y_label': f'mode amplitudes a_n ({shown_label(solver)})', 'series': series, 'series_units': units,
              'cross_section': [monitor.cross_start, monitor.cross_end], 'guide_axis': axis_name,
              'free_walls': list(monitor.projector.free_walls), **_projection_sector_extras(monitor)}
    if solver.selected_resonance is None and len(monitor.record):   # a driven run: every recorded frequency
        frequencies, projections = monitor.record.recorded()
        result['frequencies'] = frequencies
        result['frequency_series'] = _projection_series(np.moveaxis(projections, 1, 0))[0]
    return result


def _apply_pressure(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    solver.add_pressure(mask, ctx.eval_field(values['pressure'], mask), values['phase'])
    return None


def _apply_line_force(ctx, solver, feature_spec, values, selection):
    force = complex(values['force']) * _rotation(values['phase'])
    for face in selection.outer_faces:
        solver.add_face_force(face, force)
    if selection.objects:
        if ctx.domain.grid_kind != 'triangles':
            raise ValueError(f"line_force '{feature_spec.name}': a 1D membrane takes end forces on x_lo / x_hi only")
        solver.add_edge_force(ctx.boundary_edges(selection.objects), force)
    return None


def _apply_point_force(ctx, solver, feature_spec, values, selection):
    for name in selection.objects:
        solver.add_point_force(ctx.point(name), values['force'], values['phase'])
    return None


def _apply_probe(ctx, solver, feature_spec, values, selection):
    if len(selection.objects) != 1:
        raise ValueError(f"field_probe '{feature_spec.name}' needs exactly one point object, got "
                         f'{len(selection.objects)}')
    name = selection.objects[0]
    if ctx.domain.grid_kind == 'triangles':
        return solver.add_probe_node(feature_spec.name, ctx.point_node(name))
    return solver.add_probe_at(feature_spec.name, ctx.point(name)[0])


def _probe_result(session, solver, probe, feature_spec, options):
    """harmonic_probe_result + sector_rule 'linear' (parity sectors: the drawn point's phasor is the sector mean)."""
    return {**harmonic_probe_result(probe.record), 'sector_rule': 'linear'}


def _apply_resonance_table(ctx, solver, feature_spec, values, selection):
    return ResonanceTableMonitor(feature_spec.name)


def _resonance_table_result(session, solver, monitor, feature_spec, options):
    """fem resonance_table plus the GAS_TABLE_COLUMNS totals of the evaluated squeeze films (triangles)."""
    if solver.eigen_solve is None:
        raise ValueError(f"resonance_table '{feature_spec.name}': not solved yet (run an eigenfrequency study)")
    result = resonance_table(solver.eigen_solve, solver.selected_resonance)
    totals = gas_damping_totals(getattr(solver, 'squeeze_films', ()))
    if totals is None or len(totals) != len(result['rows']):
        return result
    result['columns'] = result['columns'] + list(GAS_TABLE_COLUMNS)
    result['units'] = result['units'] + list(GAS_TABLE_UNITS)
    result['rows'] = [row + total for row, total in zip(result['rows'], totals)]
    return result


def _rotation(phase_degrees) -> complex:
    """exp(j phase) of a phase in degrees."""
    return cmath.exp(1j * math.radians(float(phase_degrees)))


# ---- validation (CPU only) ----
def _settings_of(vctx, physics_spec):
    return vctx.values(_SETTINGS, physics_spec.settings) or {}


def _outer_faces_of(vctx, feature_spec) -> list:
    try:
        return [name for name in expand_entities(vctx.spec, feature_spec.selection) if name in FACES]
    except (ValueError, KeyError, TypeError):   # reported by the selection checks
        return []


def _validate_boundary_feature(vctx, physics_spec, feature_spec, values):
    """1D: ends only; a line force on a PML face is an error (it lies on the ring's outer side), a clamp / support there
    a warning (held anyway); any of them on a periodic face is an error."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.spec.dimensions == 1 and vctx.objects(feature_spec.selection):
        issues.append(Issue('error', path, f"{feature_spec.type} '{feature_spec.name}': a 1D membrane takes the ends "
                                           'x_lo / x_hi only, not object outlines'))
    settings_values = _settings_of(vctx, physics_spec)
    faces = set(_outer_faces_of(vctx, feature_spec))
    on_pml = sorted(set(_pml_faces(settings_values)) & faces)
    if on_pml and feature_spec.type == 'line_force':
        issues.append(Issue('error', path, f"line_force '{feature_spec.name}' on the PML face(s) {', '.join(on_pml)}: "
                                           'it would act on the outer side of the layer; drive inside the window'))
    elif on_pml and feature_spec.type in _MIRROR_FEATURES:
        issues.append(Issue('error', path, f"{feature_spec.type} '{feature_spec.name}' on the PML face(s) "
                                           f"{', '.join(on_pml)}: a mirror plane lies on the window edge and cannot "
                                           'absorb; turn pml_<face> off there'))
    elif on_pml:
        issues.append(Issue('warning', path, f"{feature_spec.type} '{feature_spec.name}' on the PML face(s) "
                                             f"{', '.join(on_pml)}: the outer side of the layer is held already; it "
                                             'does nothing'))
    periodic = sorted(face for face in faces if face[0] in _periodic_axes(settings_values))
    if periodic:
        issues.append(Issue('error', path, f"{feature_spec.type} '{feature_spec.name}' on the periodic face(s) "
                                           f"{', '.join(periodic)}: it would contradict the periodic image; select "
                                           'the other faces or objects inside the cell'))
    if feature_spec.type == 'symmetry':
        issues.extend(_symmetry_overlap_issues(vctx, physics_spec, feature_spec, faces, path))
    return issues


_MIRROR_FEATURES = ('symmetry', 'antisymmetry')
_FACE_HOLDS = ('clamped', 'simply_supported', 'antisymmetry')   # w = 0 on the face
_LINE_HOLDS = ('clamped_line', 'simply_supported_line')


def _symmetry_overlap_issues(vctx, physics_spec, feature_spec, faces, path) -> list:
    """Warning when a face of a symmetry feature is also held (w = 0) by another enabled feature: the hold wins."""
    held = set()
    for feature in physics_spec.features:
        if feature.enabled and feature.type in _FACE_HOLDS:
            held |= faces & set(_outer_faces_of(vctx, feature))
    if not held:
        return []
    return [Issue('warning', path, f"symmetry '{feature_spec.name}': {', '.join(sorted(held))} is also held (clamped, "
                                   'simply supported or antisymmetry), which wins there: the face is no even mirror')]


def _validate_line_hold(vctx, physics_spec, feature_spec, values):
    """clamped_line / simply_supported_line: warnings for a line end on a periodic side whose image is not the line's
    other end (the image is held only if another line holds it) and for a line reaching beyond the window (the mesher
    drops the part outside)."""
    size = vctx.domain_size
    if size is None:
        return []
    path = feature_path(physics_spec, feature_spec)
    periodic = _periodic_axes(_settings_of(vctx, physics_spec))
    tolerance = 1e-9 * max(size)
    issues = []
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is None or shape.kind != 'line':
            continue
        ends = ((shape.values['x0'], shape.values['y0']), (shape.values['x1'], shape.values['y1']))
        if any(not -tolerance <= end[axis] <= size[axis] + tolerance for end in ends for axis in (0, 1)):
            issues.append(Issue('warning', path, f"{feature_spec.type} '{feature_spec.name}': line '{name}' reaches "
                                                 'beyond the window; only its part inside the window is held'))
        sides = set()
        for index, axis in enumerate(_AXES):
            if axis not in periodic:
                continue
            for side, edge, image in (('lo', 0.0, float(size[index])), ('hi', float(size[index]), 0.0)):
                for end in ends:
                    if abs(end[index] - edge) > tolerance:
                        continue
                    imaged = list(end)
                    imaged[index] = image
                    if not any(math.dist(imaged, other) <= tolerance for other in ends):   # a line spanning the period
                        sides.add(f'{axis}_{side}')
        if sides:
            issues.append(Issue('warning', path, f"{feature_spec.type} '{feature_spec.name}': line '{name}' ends on the "
                                                 f"periodic side(s) {', '.join(sorted(sides))}: its periodic image is "
                                                 'held only if another line holds it there; keep the line off the '
                                                 'periodic sides or across the whole period'))
    return issues


def _validate_port(vctx, physics_spec, feature_spec, values):
    """Triangle meshes, frequency_response only; one outer face, not a PML or periodic face, at most one port per face;
    incident_mode <= mode_count."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.spec.dimensions == 1 or not on_triangles(vctx, physics_spec):
        issues.append(Issue('error', path, f"port '{feature_spec.name}': ports need a 2D model on a triangle mesh"))
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('error', path, f"port '{feature_spec.name}' works in a {FREQUENCY_RESPONSE_STUDY} study only "
                                           f'(its boundary terms depend on the frequency; this model runs '
                                           f'{vctx.study_type})'))
    faces = _outer_faces_of(vctx, feature_spec)
    if len(faces) != 1:
        issues.append(Issue('error', path, f"port '{feature_spec.name}' needs exactly one outer face, got "
                                           f"{', '.join(faces) or 'none'}"))
    settings_values = _settings_of(vctx, physics_spec)
    for face in faces:
        if face in _pml_faces(settings_values):
            issues.append(Issue('error', path, f"port '{feature_spec.name}' on the PML face {face}: a port absorbs "
                                               'itself; turn pml_' + face + ' off'))
        if face[0] in _periodic_axes(settings_values):
            issues.append(Issue('error', path, f"port '{feature_spec.name}' on the periodic face {face}"))
        others = [feature.name for feature in physics_spec.features
                  if feature.enabled and feature.type == 'port' and feature.name != feature_spec.name
                  and face in _outer_faces_of(vctx, feature)]
        if others:
            issues.append(Issue('error', path, f"port '{feature_spec.name}' shares {face} with port(s) "
                                               f"{', '.join(others)}: one port per face (it holds every mode)"))
        walls = sorted({wall for feature in _enabled_features(physics_spec, 'symmetry')
                        for wall in _outer_faces_of(vctx, feature) if wall[0] != face[0]})
        if len(walls) == 2:
            issues.append(Issue('warning', path, f"port '{feature_spec.name}' on {face}: symmetry on both {walls[0]} "
                                                 f"and {walls[1]}; a guide reaching both has two free walls (a constant "
                                                 'mode), which the run refuses: hold one of them'))
    issues.extend(_incident_issues(vctx, physics_spec, feature_spec, values, path))
    return issues


def _incident_issues(vctx, physics_spec, feature_spec, values, path) -> list:
    """Incident modes within mode_count (2 mode_count when a mirror feature may end the guide: unfolded numbering),
    incident_modes whole numbers with matching amplitude / phase lists; warnings for ignored values."""
    mirrored = any(name in FACES for feature in _enabled_features(physics_spec, *_MIRROR_FEATURES)
                   for name in _outer_faces_of(vctx, feature))
    count = values.get('mode_count', 1) or 1
    limit = 2 * count if mirrored else count
    text = f"mode_count {count}" + (' (x 2: numbered on the unfolded guide of a mirror face)' if mirrored else '')
    name = feature_spec.name
    issues = []
    modes = values.get('incident_modes') or []
    if values.get('incident_mode', 0) > limit:
        issues.append(Issue('error', path, f"port '{name}': incident_mode {values['incident_mode']} exceeds {text}"))
    if modes and values.get('incident_mode', 0):
        issues.append(Issue('warning', path, f"port '{name}': incident_modes is given, so incident_mode "
                                             f"{values['incident_mode']} is ignored"))
    bad = [mode for mode in modes if mode != int(mode) or not 1 <= mode <= limit]
    if bad:
        issues.append(Issue('error', path, f"port '{name}': incident_modes {', '.join(f'{mode:g}' for mode in bad)} are "
                                           f'not whole numbers from 1 to {limit} ({text})'))
    for list_name in ('incident_amplitudes', 'incident_phases'):
        given = values.get(list_name) or []
        if given and not modes:
            issues.append(Issue('warning', path, f"port '{name}': {list_name} is ignored without incident_modes"))
        elif given and len(given) != len(modes):
            issues.append(Issue('error', path, f"port '{name}': {list_name} has {len(given)} entries, incident_modes "
                                               f'{len(modes)}: give one per mode (or leave it empty)'))
    return issues


def _validate_mode_projection(vctx, physics_spec, feature_spec, values):
    """Triangle meshes, no plate (bending_stiffness); cross_end > cross_start and end > start when given; result_axis
    'frequency': average_end >= average_start, the window overlapping start .. end, a warning outside
    frequency_response."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.spec.dimensions == 1 or not on_triangles(vctx, physics_spec):
        issues.append(Issue('error', path, f"mode_projection '{feature_spec.name}' needs a 2D model on a triangle "
                                           'mesh'))
    plates = _materials_setting(vctx, BENDING_ARRAY)
    if plates:
        issues.append(Issue('error', path, f"mode_projection '{feature_spec.name}' projects onto the film strip "
                                           f"profiles; with bending_stiffness (material(s) {', '.join(plates)}) the "
                                           'guide modes are the numeric plate strip modes, which it does not use: take '
                                           'ports or field probes'))
    for low, high in (('cross_start', 'cross_end'), ('start', 'end')):
        if values.get(low) is not None and values.get(high) is not None and not values[high] > values[low]:
            issues.append(Issue('error', path, f"mode_projection '{feature_spec.name}': {high} must exceed {low}"))
    if values.get('result_axis') != 'frequency':
        return issues
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('warning', path, f"mode_projection '{feature_spec.name}': result_axis 'frequency' records "
                                             f'only in a {FREQUENCY_RESPONSE_STUDY} study (this model runs '
                                             f'{vctx.study_type}): its result will be missing'))
    average_start, average_end = values.get('average_start'), values.get('average_end')
    if average_start is not None and average_end is not None and average_end < average_start:
        issues.append(Issue('error', path, f"mode_projection '{feature_spec.name}': average_end must not be below "
                                           'average_start'))
    if ((average_end is not None and values.get('start') is not None and average_end < values['start'])
            or (average_start is not None and values.get('end') is not None and average_start > values['end'])):
        issues.append(Issue('error', path, f"mode_projection '{feature_spec.name}': the averaging window lies outside "
                                           'the sampled positions start .. end'))
    return issues


def _validate_probe(vctx, physics_spec, feature_spec, values):
    issues = validate_points_inside(vctx, physics_spec, feature_spec, values)
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('warning', feature_path(physics_spec, feature_spec),
                            f"field_probe records only in a {FREQUENCY_RESPONSE_STUDY} study (this model runs "
                            f'{vctx.study_type}): its result will be missing'))
    return issues


def _validate_squeeze_film(vctx, physics_spec, feature_spec, values):
    """Warning outside the eigenfrequency study; errors in a Bloch cell and for a closed incompressible film (vents
    none); warning beyond the Veijola Knudsen range."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        issues.append(Issue('warning', path, f'squeeze_film_damping is evaluated per resonance by the '
                                             f'{EIGENFREQUENCY_STUDY} study only (it does not damp a driven response); '
                                             f'this {vctx.study_type} study ignores it'))
    if _periodic_axes(_settings_of(vctx, physics_spec)):
        issues.append(Issue('error', path, 'squeeze_film_damping is not available in a Bloch cell (periodic axes): its '
                                           'footprint would need periodic vents'))
    issues.extend(squeeze_film_gas_issues(values, path, closed=values.get('vents') == 'none'))
    return issues


def _validate_resonance_table(vctx, physics_spec, feature_spec, values):
    if vctx.study_type == EIGENFREQUENCY_STUDY:
        return []
    return [Issue('warning', feature_path(physics_spec, feature_spec),
                  f'resonance_table is filled only by an {EIGENFREQUENCY_STUDY} study (this model runs '
                  f'{vctx.study_type}): its result will be missing')]


def _material_issues(vctx, physics_spec, path, one_d):
    """Once per model: a negative areal density (error), no selected material with areal_density > 0 (error), a
    compressive tension (info), tension_xy in a 1D model (warning: ignored)."""
    if not reports_model_checks(vctx, physics_spec, MEMBRANE_TYPE_NAME):
        return []
    issues, any_mass = [], False
    parameters = vctx.parameters
    for material, properties, _ in selected_material_data(vctx.spec):   # unresolved ones: reported at the material
        density = properties.get('areal_density')
        any_mass = any_mass or evaluates_to(density, parameters, lambda value: value > 0.0)
        material_path = ('material', material.name)
        if evaluates_to(density, parameters, lambda value: value < 0.0):
            issues.append(Issue('error', material_path, f"material '{material.name}': areal_density must be >= 0"))
        compressive = [name for name in ('tension', 'tension_xx', 'tension_yy')
                       if evaluates_to(properties.get(name), parameters, lambda value: value < 0.0)]
        if compressive:
            issues.append(Issue('info', material_path,
                                f"material '{material.name}' sets a compressive {', '.join(compressive)} < 0: the "
                                'membrane has no restoring force there (unstable resonances are dropped)'))
        if one_d and evaluates_to(properties.get('tension_xy'), parameters, lambda value: value != 0.0):
            issues.append(Issue('warning', material_path, f"material '{material.name}' sets tension_xy, which a 1D "
                                                          'membrane ignores'))
    if not any_mass:
        issues.append(Issue('error', path, 'no selected material sets areal_density > 0: the membrane has no mass '
                                           '(assign a material with tension and areal_density)'))
    return issues


def _materials_setting(vctx, name) -> list:
    """Names of the spec materials whose effective properties set `name` (any value, any selection: as
    meshing.optional_material_arrays, which allocates the array the solver reads as present)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    names = []
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            continue
        if properties.get(name) is not None:
            names.append(material.name)
    return names


def _ring_layers(vctx, physics_spec):
    """pml_layers of the physics' MeshSpec as an int, None when it is not a plain number (or the mesh is unknown)."""
    try:
        return int(vctx.spec.mesh_of(physics_spec).pml_layers)
    except (KeyError, ValueError, TypeError):
        return None


def _enabled_features(physics_spec, *types) -> list:
    return [feature for feature in physics_spec.features if feature.enabled and feature.type in types]


def _bending_issues(vctx, physics_spec, settings_values, path, one_d) -> list:
    """A plate (some material sets bending_stiffness): quadratic elements on triangles (a PML is the stretched plate
    form, ports take the numeric plate strip modes: a warning under parity_sectors); 1D: info."""
    plates = _materials_setting(vctx, BENDING_ARRAY)
    if not plates:
        return []
    names = ', '.join(plates)
    if one_d:
        return [Issue('info', path, f'bending_stiffness (material(s) {names}): Hermite cubic line elements (w and w\'); '
                                    'the discretization setting does not apply')]
    issues = []
    if settings_values.get(DISCRETIZATION_SETTING) == 'linear':
        issues.append(Issue('error', path, f'bending_stiffness (material(s) {names}) needs discretization quadratic (the '
                                           'interior penalty plate form uses second derivatives)'))
    cells, layers = int(settings_values.get('pml_cells') or DEFAULT_PML_CELLS), _ring_layers(vctx, physics_spec)
    if _pml_faces(settings_values) and layers is not None and layers < cells:
        issues.append(Issue('warning', path, f'bending_stiffness (material(s) {names}) with a PML: the mesh ring has '
                                             f'pml_layers {layers} element layers across pml_cells {cells} x dx, so they '
                                             'are thicker than dx and the stretched ring under-resolves the plate waves '
                                             'and reflects (8 layers across 24 cells: 79 % in amplitude, 24 layers: '
                                             '0.4 %); set the mesh pml_layers >= pml_cells'))
    ports = _enabled_features(physics_spec, 'port')
    if ports and _study_values(vctx.spec, vctx.registry, vctx.parameters).get('parity_sectors'):
        issues.append(Issue('warning', path, f"bending_stiffness (material(s) {names}) with parity_sectors: a port "
                                             f"({', '.join(feature.name for feature in ports)}) whose face has bending "
                                             'stiffness and whose guide ends on a mirror face is refused at the build '
                                             '(plate port modes are not combined over sectors): run the full model'))
    return issues


def _periodic_issues(vctx, physics_spec, settings_values, path, one_d) -> list:
    """Bloch cells: warning per bloch_k on a non-periodic axis; errors in 1D / off triangles, outside the eigenfrequency
    study, with a PML on a periodic axis, and for a free cell at target_frequency 0 with every bloch_k 0."""
    periodic = _periodic_axes(settings_values)
    issues = [Issue('warning', path, f'bloch_k{axis} = {_bloch_wavenumber(settings_values, axis):.4g} rad/m is ignored: '
                                     f'periodic_{axis} is off')
              for axis in _AXES if axis not in periodic and _bloch_wavenumber(settings_values, axis) != 0.0]
    if not periodic:
        return issues
    names = ', '.join(periodic)
    if one_d or not on_triangles(vctx, physics_spec):
        issues.append(Issue('error', path, f'periodic axes ({names}) need a 2D model on a triangle mesh'))
        return issues
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        issues.append(Issue('error', path, f'periodic axes ({names}) are solved by the {EIGENFREQUENCY_STUDY} study '
                                           f'only (Bloch bands per k); this model runs {vctx.study_type}'))
    ringed = [face for face in _pml_faces(settings_values) if face[0] in periodic]
    if ringed:
        issues.append(Issue('error', path, f"PML on the periodic face(s) {', '.join(ringed)}: a periodic axis has no "
                                           'absorbing ring'))
    trivial = all(_bloch_wavenumber(settings_values, axis) == 0.0 for axis in periodic)
    if trivial and not settings_values.get('target_frequency') and not _enabled_features(
            physics_spec, *_FACE_HOLDS, *_LINE_HOLDS):
        issues.append(Issue('error', path, 'with every periodic bloch_k 0 and nothing clamped the cell moves rigidly at '
                                           '0 Hz (K - 0 M is singular): set target_frequency > 0, a bloch_k != 0 or a '
                                           'clamp'))
    return issues


def _solver_issues(vctx, physics_spec, settings_values, path, one_d) -> list:
    """1D: a cuDSS or MUMPS / PARDISO 'linear' node is an error (the small banded line factor stays on host SuperLU);
    triangles: the host SuperLU size rule (SPLU_* limits on the P1 / P2 dofs of the mesh) of the factor the study uses,
    also for a MUMPS / PARDISO factor (uncalibrated, named by its library), none for a cuDSS factor."""
    solvers = vctx.solvers(physics_spec) or {}
    linear = solvers.get('linear')
    cudss_linear = linear is not None and linear.values.get('back_end') in CUDSS_BACK_ENDS
    if one_d:
        if not cudss_linear and host_direct_back_end(solvers) is None:
            return []
        return [Issue('error', path, f"linear solver '{linear.node}': a 1D membrane factorises its small banded line "
                                     "operator on the host; use Automatic or back_end 'auto' / 'scipy_splu' (cuDSS, "
                                     'MUMPS and PARDISO serve triangle meshes)')]
    mesh = vctx.mesh
    if mesh is None or mesh.kind != 'triangles':
        return []
    if vctx.study_type == FREQUENCY_RESPONSE_STUDY:
        if cudss_linear:
            return []
        library = factor_label(host_direct_back_end(solvers))
        remedy = ("assign a sparse_direct solver node with back_end 'cudss' or 'cudss_mixed' (GPU backend) as the "
                  "membrane's linear solver, or coarsen the mesh")
    elif vctx.study_type == EIGENFREQUENCY_STUDY:
        if eigen_factorization(solvers) in CUDSS_FACTORIZATIONS:
            return []
        library = factor_label(host_direct_back_end(solvers, role='eigen'))
        remedy = ("assign an arpack_shift_invert solver node with factorization 'cudss' (GPU backend) as the "
                  "membrane's eigen solver, or coarsen the mesh")
    else:
        return []
    order = element_order(settings_values)
    unknowns = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    text = f'up to ~{unknowns:,} complex unknowns (P{order} dofs of the mesh, hole elements included)'
    if unknowns > SPLU_LIMIT_UNKNOWNS:
        return [Issue('error', path, f'{text} exceed the {SPLU_LIMIT_UNKNOWNS:,} limit of the host {library} factor: '
                                     f'its factors would fill the host memory; {remedy}')]
    if unknowns > SPLU_WARN_UNKNOWNS:
        return [Issue('warning', path, f'{text} exceed ~{SPLU_WARN_UNKNOWNS:,}, the practical size of the host '
                                       f'{library} factor; {remedy}')]
    return []


def _validate_membrane(vctx, physics_spec, settings_values):
    """Mesh kind (triangles in 2D, a 1D model on the tensor grid), axial_wavenumber in 1D only, PML on triangles only
    with a reference frequency in an eigenfrequency study, plates, Bloch cells, scipy, solver sizes, materials."""
    path = ('physics', physics_spec.name)
    issues = []
    if not scipy_available():
        issues.append(Issue('error', path, 'the membrane solves on the host with scipy (sparse factorisations, '
                                           'ARPACK), which is not installed: pip install scipy'))
    one_d = vctx.spec.dimensions == 1
    pml = _pml_faces(settings_values)
    if not one_d and not on_triangles(vctx, physics_spec):
        issues.append(Issue('error', path, 'a 2D membrane runs on a triangle mesh (MeshSpec kind triangles); '
                                           'rectangular cells serve 1D models only'))
    if one_d and pml:
        issues.append(Issue('error', path, f"PML on {', '.join(pml)}: a 1D membrane has no PML (clamp or free its "
                                           'ends)'))
    if not one_d and settings_values.get('axial_wavenumber'):
        issues.append(Issue('error', path, 'axial_wavenumber applies to 1D models only (a 2D membrane draws its y '
                                           'direction): set it to 0'))
    if pml and vctx.study_type == EIGENFREQUENCY_STUDY and not (settings_values.get('target_frequency')
                                                                 or settings_values.get('pml_frequency')):
        issues.append(Issue('error', path, 'a PML needs a reference frequency in an eigenfrequency study: set '
                                           'target_frequency > 0 (or pml_frequency)'))
    issues.extend(_bending_issues(vctx, physics_spec, settings_values, path, one_d))
    issues.extend(_periodic_issues(vctx, physics_spec, settings_values, path, one_d))
    issues.extend(_solver_issues(vctx, physics_spec, settings_values, path, one_d))
    issues.extend(_sweep_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_material_issues(vctx, physics_spec, path, one_d))
    issues.extend(_parity_sector_issues(vctx, physics_spec))
    return issues


def _parity_sector_issues(vctx, physics_spec) -> list:
    """Study setting parity_sectors on: errors for a symmetry / antisymmetry feature selecting a sector face together
    with other entities (they would keep one type while the face flips) and for a clamped / simply_supported face
    that is a sector face (no mirror there)."""
    if not _study_values(vctx.spec, vctx.registry, vctx.parameters).get('parity_sectors'):
        return []
    faces = set(mirror_faces(vctx.spec))
    issues = []
    for feature in _enabled_features(physics_spec, *_MIRROR_FEATURES):
        try:
            entities = expand_entities(vctx.spec, feature.selection)
        except (ValueError, KeyError, TypeError):   # reported by the selection checks
            continue
        on_planes = [name for name in entities if name in faces]
        others = [name for name in entities if name not in faces]
        if on_planes and others:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} '{feature.name}' selects the parity-sector face(s) "
                                f"{', '.join(on_planes)} together with {', '.join(others)}: the sectors swap the "
                                'feature on the mirror faces only; give the other entities a feature of their own'))
    for feature in _enabled_features(physics_spec, 'clamped', 'simply_supported'):
        held = sorted(faces & set(_outer_faces_of(vctx, feature)))
        if held:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} '{feature.name}' holds the parity-sector face(s) {', '.join(held)}: a "
                                'mirror plane of the sector runs must be free (symmetry) or odd (antisymmetry) only; '
                                'remove the hold there or switch parity_sectors off'))
    return issues


def _study_values(spec, registry, parameters) -> dict:
    """Resolved settings of the model's study ({} when they do not resolve: reported by validate)."""
    study = spec.study
    try:
        return resolve_values(registry.study_type(study.type).params,
                              study.settings if isinstance(study.settings, dict) else {}, parameters)
    except (KeyError, ValueError, TypeError):
        return {}


def _loss_factors(vctx):
    """Set of the loss_factor values of the selected materials (unset = 0), None when one is not a plain number."""
    values = set()
    for _, properties, _ in selected_material_data(vctx.spec):   # unresolved ones: reported at the material
        seen = []
        value = properties.get(LOSS_ARRAY)
        if value is None:
            seen.append(0.0)
        elif evaluates_to(value, vctx.parameters, lambda number: seen.append(float(number)) is None) and not seen:
            return None   # field / state dependent
        values.update(seen)
    return values


def _sweep_issues(vctx, physics_spec, settings_values, path) -> list:
    """frequency_response sweep_method adaptive / modal: ports are an error; modal warns for a PML or a loss_factor that
    differs between materials (the modes then do not decouple the response)."""
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        return []
    method = _study_values(vctx.spec, vctx.registry, vctx.parameters).get('sweep_method', 'direct')
    if method == 'direct':
        return []
    issues = []
    ports = _enabled_features(physics_spec, 'port')
    if ports:
        issues.append(Issue('error', path, f"sweep_method '{method}': ports "
                                           f"({', '.join(feature.name for feature in ports)}) add frequency-dependent "
                                           "boundary terms a reduced sweep does not handle yet; use sweep_method "
                                           "'direct'"))
    if method != 'modal':
        return issues
    if _pml_faces(settings_values):
        issues.append(Issue('warning', path, "sweep_method 'modal' with a PML: the stretched modes do not decouple the "
                                             'response; check the logged residual and verified error, or use '
                                             "sweep_method 'adaptive'"))
    losses = _loss_factors(vctx)
    if losses is None or len(losses) > 1:
        issues.append(Issue('warning', path, "sweep_method 'modal' with a loss_factor that differs between materials "
                                             '(or depends on position): the modes do not decouple the loss, so the '
                                             "modal result is approximate; check the logged residual or use 'adaptive'"))
    return issues


# ---- physics factory callbacks ----
def _create(ctx, physics_spec, values):
    """MembraneFEMPhysics (triangles) or MembraneLinePhysics (1D model) with the 'linear' and 'eigen' runtimes."""
    runtimes = ctx.solvers(physics_spec.name)
    common = {'linear_solver': runtimes.get('linear'), 'eigen_solver': runtimes.get('eigen'),
              'target_frequency': values['target_frequency'], 'discretization': values[DISCRETIZATION_SETTING]}
    if ctx.domain.grid_kind == 'triangles':
        from .solver_fem import MembraneFEMPhysics
        return MembraneFEMPhysics(ctx.domain, pml_faces=_pml_faces(values), pml_reflection=values['pml_reflection'],
                                  pml_frequency=values['pml_frequency'],
                                  periodic=tuple(bool(values[f'periodic_{axis}']) for axis in _AXES),
                                  bloch_k=tuple(float(values[f'bloch_k{axis}']) for axis in _AXES), **common)
    from .solver_line import MembraneLinePhysics
    return MembraneLinePhysics(ctx.domain, axial_wavenumber=values['axial_wavenumber'], **common)


# ---- estimates (CPU) ----
_N_DEVICE_FIELDS = len(STATE_FIELD_NAMES) + len(COUPLED_TENSION_SLOTS)
_HOST_CSR_COPIES = 6       # K, C, M terms, their union, the reduced free block and the coupling block (complex128)
_HOST_VECTORS = 8          # w, loads, solution, fixed values, probes' phasors (complex128 per dof)
# System build recipe (MembraneBase._ready_system, measured by the assembly_seconds counter): on triangles the Lagrange
# terms are counted from the code (definition_helpers.scalar_harmonic_build_items); the line (1D) system and the plate
# terms (bending_fem element and edge blocks) keep host-stream bytes per unit, UNCALIBRATED first guesses (time_check
# 'assembly_seconds' group); cross-check: romero2019 e2c, P2 plate on 302k triangles, ~8 s.
ASSEMBLY_BYTES_PER_COO_ENTRY = 400    # line / plate: quadrature coefficients, local blocks, COO -> CSR
ASSEMBLY_CALLS = 200
PLATE_EDGE_BYTES_PER_POINT_ENTRY = 64   # interior penalty: side derivatives, moments, einsum per Gauss point x entry
PLATE_EDGE_CALLS_PER_CHUNK = 60         # bending_fem evaluation calls per CHUNK_ITEMS edges
_PLATE_EDGE_POINTS = EDGE_DEGREE // 2 + 1   # Gauss points per edge (exact to EDGE_DEGREE)
PENCIL_BYTES_PER_ENTRY = 500          # line: pattern_sum of the terms, HarmonicPencil free / held blocks per stored entry
PORT_SETUP_BYTES_PER_EDGE = 200       # per port / mode projection: port_edges scan of the mesh edges, strip modes
PORT_SETUP_CALLS = 500
# cuDSS fill of the membrane operator over the complex 5-point curve at equal unknowns, per Lagrange order (x
# BENDING_ENTRY_FACTOR with the plate's edge couplings); UNCALIBRATED (calibrate from a membrane example at a few 100k
# P2 dofs; romero2019 e2c cross-check: plan + factor 5.6 s per frequency at ~600k P2 plate dofs)
CUDSS_FILL_SCALE = {1: 1.5, 2: 3.0}
BENDING_ENTRY_FACTOR = 2.0    # interior penalty edge terms couple both elements of an edge (uncalibrated)
# host SuperLU of a plate (theory): the edge terms reach one element further, so nested-dissection separators are two
# element layers wide: fill x width^2 (solves), flops x width^3 (factor) over the membrane operator of equal unknowns
# (selftest plates: L + U ~4.3 x the membrane's per dof, factor 5-9 x the membrane recipe). cuDSS keeps
# BENDING_ENTRY_FACTOR (its host plan scales with the entries; romero2019 e2c is already below that recipe)
BENDING_SEPARATOR_WIDTH = 2.0
HERMITE_ENTRIES_PER_DOF = 6   # w / w' rows of a Hermite line couple both neighbours' two dofs
PORT_ENTRIES_PER_NODE = 4     # dense DtN block per port ~ (2 sqrt(n_nodes))^2 (uncalibrated)
# plate ports (plate_ports.py): the block holds every P2 dof of the port elements (~2 x the edge dofs, so 4 x the
# entries), and each frequency solves the dense companion pencil of the Hermite strip (2 n ~ 4 x port nodes ~ 4
# sqrt(n_nodes)) plus an LU of the trace matrix; uncalibrated
PLATE_PORT_ENTRY_FACTOR = 4.0
# dense work per port and frequency / (2 n)^3 in real flops: the complex128 companion pencil goes through zggev
# (Hessenberg-triangular reduction + QZ with the transforms accumulated ~66 N^3 complex operations, Golub & Van Loan
# 7.7; eigenvectors included; one complex operation ~4 real flops; 30 N^3 is the eigenvalues-only real count), then
# lu_factor of the traces (8 / 3 x 4), the N-column lu_solve for Z (8) and the Pi^T Z Pi block (k ~ N: 8 x 2)
PLATE_STRIP_EIG_FLOPS = 4.0 * 66.0
PLATE_STRIP_DTN_FLOPS = 4.0 * 8.0 / 3.0 + 8.0 + 16.0


def _plate(context) -> bool:
    """A selected material sets bending_stiffness (the plate elements)."""
    return bool(context.material_sets(BENDING_ARRAY))


def _feature_count(context, *types) -> int:
    """Enabled features of these types of the estimated physics (0 without its physics_spec)."""
    physics_spec = getattr(context, 'physics_spec', None)
    return len(_enabled_features(physics_spec, *types)) if physics_spec is not None else 0


def _discretisation(settings_values, mesh, context):
    """(unknowns, elements, stored entries, host complex factor bytes, triangles) of the membrane on its mesh."""
    order = element_order(settings_values)
    plate = _plate(context)
    if mesh.kind == 'triangles':
        dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
        entries = lagrange_entries_per_dof(mesh, order) * dofs * (BENDING_ENTRY_FACTOR if plate else 1.0)
        entries += (PORT_ENTRIES_PER_NODE * (PLATE_PORT_ENTRY_FACTOR if plate else 1.0) * int(mesh.n_nodes)
                    * _feature_count(context, 'port'))
        return dofs, int(mesh.n_elements), int(entries), int(FEM_FILL_FACTOR * host_lu_bytes(dofs)), True
    cells = window_cell_count(mesh)
    if plate:
        dofs = 2 * (cells + 1)
        entries = HERMITE_ENTRIES_PER_DOF * dofs
    else:
        dofs = order * cells + 1
        entries = (2 * order + 1) * dofs                     # banded line operator
    return dofs, cells, entries, 2 * entries * COMPLEX128_CSR_ENTRY_BYTES, False


def _requested_resonances(context) -> int:
    """n_resonances of the model's study (1 when it does not resolve or is absent)."""
    return int(_study_values(context.spec, context.registry, context.parameters).get('n_resonances') or 1)


def _complex_pencil(settings_values, context) -> bool:
    """A PML, a loss factor or a Bloch phase makes K complex (the eigen solve's complex paths)."""
    bloch = any(_bloch_wavenumber(settings_values, axis) != 0.0 for axis in _periodic_axes(settings_values))
    return bool(_pml_faces(settings_values)) or bloch or context.material_sets('loss_factor')


def _cudss_factor(settings_values, context):
    """(back end / factorization name, value bytes) of the cuDSS factor the study uses (frequency_response: the
    'linear' back end, eigenfrequency: the 'eigen' factorization), None for a host SuperLU factor and for a
    batched_direct direct sweep (batched_direct_items; outside one its per-system back end)."""
    study = context.spec.study.type
    if study == FREQUENCY_RESPONSE_STUDY:
        if batched_direct_sweep(context) is not None:
            return None
        back_end = (resolved_back_end(context.solvers)
                    or batched_direct_back_end(context.solvers, solver_backend(context.spec)))
        return (back_end, COMPLEX128_BYTES) if back_end in CUDSS_BACK_ENDS else None
    if study == EIGENFREQUENCY_STUDY:
        factorization = eigen_factorization(context.solvers)
        if factorization not in CUDSS_FACTORIZATIONS:
            return None
        return factorization, COMPLEX128_BYTES if _complex_pencil(settings_values, context) else 8
    return None


def _host_direct_factor(context):
    """The MUMPS / PARDISO back end (host_direct.HOST_DIRECT_BACK_ENDS) of the factor the study uses
    (frequency_response: the 'linear' back end, eigenfrequency: the 'eigen' factorization), else None."""
    study = context.spec.study.type
    if study == FREQUENCY_RESPONSE_STUDY:
        return host_direct_back_end(context.solvers)
    if study == EIGENFREQUENCY_STUDY:
        return host_direct_back_end(context.solvers, role='eigen')
    return None


def _memory_estimate(settings_values, mesh, context):
    """Per-element state and coupled slots on the device, a cuDSS factor's pool items (frequency_response: kept;
    eigenfrequency: transient per solve) and the factor outside the pool; host factor / pencil / basis items (a
    MUMPS / PARDISO factor: _host_direct_factor)."""
    dofs, elements, entries, factor_bytes, triangles = _discretisation(settings_values, mesh, context)
    plate = _plate(context)
    if triangles:
        local = 12 if plate else (6 if element_order(settings_values) == 2 else 3)
    else:
        local = 4 if plate else element_order(settings_values) + 1
    cudss = _cudss_factor(settings_values, context)
    study = context.spec.study.type
    complex_pencil = _complex_pencil(settings_values, context)
    run_items, eigen_items, factors = [], [], ()
    host_direct = _host_direct_factor(context) if triangles and cudss is None else None
    # a batched_direct direct sweep (triangles; K, M terms: the membrane has no damping term): the batch's items
    batched = batched_direct_items(context, dofs, entries, 2) if triangles else None
    if batched is not None:
        host_items = list(batched['host_items'])
        if solver_backend(context.spec) != 'gpu':   # cpu: one host SuperLU per system, freed after its solve
            host_items.insert(0, ('complex factor L + U (heuristic, uncalibrated)', factor_bytes))
        run_items = list(batched['run_items'])
    elif host_direct is not None:   # MUMPS / PARDISO LDL^T: the SuperLU heuristic x superlu_bytes_scale (uncalibrated)
        label = factor_label(host_direct)
        complex_values = study == FREQUENCY_RESPONSE_STUDY or complex_pencil
        # S5c mirror (factorization_for): a small *_mixed eigen pencil factors in double
        mixed = host_direct_precision(host_direct) == 'mixed' and not (
            study == EIGENFREQUENCY_STUDY and dofs < MIXED_EIGEN_MIN_UNKNOWNS)
        scaled = host_direct if mixed else double_of(host_direct)
        host_items = [(f'{label} LDL^T factor (SuperLU heuristic scaled, uncalibrated)',
                       int(factor_bytes * superlu_bytes_scale(scaled, symmetric=True, complex_values=complex_values))),
                      (f'{label} solver copies (equilibrated operator, row scale; mixed: GMRES vectors)',
                       solver_host_bytes(dofs, entries, complex_values=complex_values, mixed=mixed, symmetric=True))]
        if study == FREQUENCY_RESPONSE_STUDY:   # fem.harmonic's sparse_direct.HostPencil: K complex, C / M real
            host_items.append((f'{label} host pencil terms (K, C, M values on the free pattern)',
                               (COMPLEX128_BYTES + 8 + 8) * entries))
    elif cudss is None:
        host_items = [('complex factor L + U (heuristic, uncalibrated)', factor_bytes)]
    else:   # cudss_factor.CudssFactorization (LDL^T): rhs / equilibration / solution vectors
        name, value_bytes = cudss
        # S5c (ArpackShiftInvertSolver.factorization_for): a small cudss_mixed pencil factors in double, host route
        switched = study == EIGENFREQUENCY_STUDY and name == 'cudss_mixed' and dofs < MIXED_EIGEN_MIN_UNKNOWNS
        mixed = name == 'cudss_mixed' and not switched
        solve_vectors = ('cuDSS rhs buffer, row scale and one solve\'s vectors (mixed: the GMRES basis)',
                         solve_buffer_bytes(dofs, value_bytes, mixed) + solve_peak_bytes(dofs, value_bytes, mixed))
        factors = (('cuDSS factor', dofs, entries, value_bytes, 2, mixed),)
        if study == FREQUENCY_RESPONSE_STUDY:
            # fem.harmonic keeps K (complex), C, M (real) as a DevicePencil whose maps / transients hold the factor's
            # planned values (no CSR copies on top: membrane_guide_pml measured) and a direct solve's solution (R6)
            run_items = [solve_vectors,
                         ('device pencil terms, index maps and per-frequency transients',
                          device_pencil_bytes(entries, (COMPLEX128_BYTES, 8, 8), symmetric=True)),
                         ('kept device solution (complex128 + int64 ids per full dof)',
                          KEPT_SOLUTION_BYTES_PER_DOF * dofs)]
        elif not complex_pencil and not switched and entries >= DEVICE_PENCIL_MIN_ENTRIES:
            # eigsh_shifted's ShiftPencil: K, M as a lower-stored DevicePencil, M products on the device
            eigen_items = [solve_vectors,
                           ('eigen DevicePencil terms, index maps and refresh transients',
                            device_pencil_bytes(entries, (value_bytes, value_bytes), symmetric=True)),
                           ('eigen device mass product (int32 indices, vectors)',
                            4 * (entries + dofs + 1) + 3 * value_bytes * dofs)]
        else:   # a factor of the host K - sigma M (eigen_solver.factorize_host): its CSR copies in the pool
            eigen_items = [solve_vectors,
                           ('cuDSS lower-triangle CSR copies', operator_copy_bytes(dofs, entries, value_bytes,
                                                                                   mixed=mixed))]
        host_items = [('cuDSS host equilibration and symmetry-check copies (transient)',
                       2 * COMPLEX128_CSR_ENTRY_BYTES * entries)]
        if study == EIGENFREQUENCY_STUDY:
            host_items.append(('eigen shift pencil term values and diagonals', 2 * value_bytes * (entries + dofs)))
    host_items += [('pencil CSR copies (complex128)', _HOST_CSR_COPIES * COMPLEX128_CSR_ENTRY_BYTES * entries),
                   ('dof vectors (complex128)', _HOST_VECTORS * COMPLEX128_BYTES * dofs),
                   ('assembly COO (transient, complex128)', 3 * COMPLEX128_BYTES * local * local * elements)]
    if _periodic_axes(settings_values):
        host_items.append(('Bloch prolongation and reduced pencil (complex128)', 3 * COMPLEX128_CSR_ENTRY_BYTES * entries))
    if study == EIGENFREQUENCY_STUDY:
        wanted = 2 * _requested_resonances(context) + 4
        basis = arnoldi_basis_size(wanted, dofs)
        # a complex pencil runs eigs: on the gpu its device Krylov-Schur basis (freed with the solve); a real one eigsh
        on_device, basis_bytes = ((False, basis * COMPLEX128_BYTES * dofs) if not complex_pencil
                                  else arnoldi_basis_bytes(dofs, basis, solver_backend(context.spec),
                                                           device_matvec=cudss is not None))
        if on_device:
            eigen_items.append((f'device Krylov-Schur basis ({basis + 1} complex128 rows + 2 work vectors)',
                                basis_bytes))
        else:
            host_items.append(('ARPACK basis (complex128)', basis_bytes))
    host_items.extend(reduced_sweep_memory_items(context, dofs, entries))
    external = batched['external'] if batched is not None and batched['external'] else cudss_external(factors)
    # fem.harmonic_resonances releases the shift factor after each solve (a search: its inertia factor too)
    return MemoryEstimate.from_items(
        ((f'element state and coupled slots ({_N_DEVICE_FIELDS} x float32)', _N_DEVICE_FIELDS * FLOAT32_BYTES
          * elements),), host_items=tuple(host_items), run_items=tuple(run_items),
        transient_items=tuple(('eigen solve', label, size) for label, size in eigen_items), **external)


def _build_items(settings_values, mesh, context, nnz) -> tuple:
    """System build of a study (MembraneBase._ready_system, measured by its assembly_seconds counter): on triangles the
    ScalarHarmonicSystem build counted from the code (scalar_harmonic_build_items), the plate element term and
    interior-penalty edges, the port / mode projection set-up; a line: element assembly and pencil (CPU)."""
    seconds = ('solvers', context.name, 'counters', 'assembly_seconds')
    assemblies = ('solvers', context.name, 'counters', 'assemblies')
    order, plate = element_order(settings_values), _plate(context)
    if mesh.kind != 'triangles':
        local = 4 if plate else order + 1
        coo_entries = 2 * local * local * float(window_cell_count(mesh))   # tension stiffness, mass
        return (te.TimeItem('membrane element assembly', 'assembly', 1.0, 'exact',
                            (te.HostStream(ASSEMBLY_BYTES_PER_COO_ENTRY * coo_entries), te.PythonCalls(ASSEMBLY_CALLS)),
                            counter=assemblies, seconds_counter=seconds),
                te.TimeItem('membrane pencil', 'assembly', 1.0, 'exact',
                            (te.HostStream(PENCIL_BYTES_PER_ENTRY * float(nnz)), te.PythonCalls(50)),
                            seconds_counter=seconds))
    dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    items = list(scalar_harmonic_build_items(
        mesh, order, elements=int(mesh.n_elements), dofs=dofs, stored_entries=nnz,
        pml=bool(_pml_faces(settings_values)), complex_stiffness=bool(context.material_sets(LOSS_ARRAY)),
        extra_stiffness_parts=int(plate) + int(bool(_feature_count(context, 'port'))), counter=assemblies,
        seconds_counter=seconds))
    if plate:   # bending_fem element blocks; interior edges <= mesh edges, 12 x 12 blocks of both sides' P2 dofs
        edges = float(mesh.n_edges)
        block_entries = edges * 144.0
        items.append(te.TimeItem('plate bending elements and interior-penalty edges', 'assembly', 1.0, 'exact', (
            te.HostStream(PLATE_EDGE_BYTES_PER_POINT_ENTRY * _PLATE_EDGE_POINTS * block_entries
                          + ASSEMBLY_BYTES_PER_COO_ENTRY * (block_entries + 36.0 * float(mesh.n_elements))),
            te.PythonCalls(PLATE_EDGE_CALLS_PER_CHUNK * math.ceil(edges / CHUNK_ITEMS))), seconds_counter=seconds))
    guides = _feature_count(context, 'port', 'mode_projection')
    if guides:
        items.append(te.TimeItem('port / mode projection set-up', 'setup', float(guides), 'exact',
                                 (te.HostStream(PORT_SETUP_BYTES_PER_EDGE * float(mesh.n_edges)),
                                  te.PythonCalls(PORT_SETUP_CALLS)), seconds_counter=seconds))
    return tuple(items)


def _time_estimate(settings_values, mesh, context):
    """The system build (_build_items); eigenfrequency: one shift-invert of the 'eigen' role; frequency_response: the
    study's harmonic_factorizations of the 'linear' role with the host work around them (harmonic_frequency_items:
    solve bookkeeping, the host reduced operator or the DevicePencil set-up; a reduced sweep: its back
    substitutions, basis items and the modal shift-invert; frequency_sweep_counts). Factor scales on
    triangles: SuperLU FEM_FILL_FACTOR (a plate x BENDING_SEPARATOR_WIDTH^3, its solves x ^2), cuDSS CUDSS_FILL_SCALE
    (x BENDING_ENTRY_FACTOR for a plate); other studies only the build (refused by validation)."""
    dofs, _, entries, _, triangles = _discretisation(settings_values, mesh, context)
    unknowns, nnz = float(dofs), float(entries)
    build = te.TimeEstimate(_build_items(settings_values, mesh, context, nnz))
    plate = triangles and _plate(context)
    cudss_scale = (CUDSS_FILL_SCALE[element_order(settings_values)] * (BENDING_ENTRY_FACTOR if plate else 1.0)
                   if triangles else 1.0)
    host_fill = (FEM_FILL_FACTOR if triangles else 1.0) * (BENDING_SEPARATOR_WIDTH ** 2 if plate else 1.0)
    host_factor = (FEM_FILL_FACTOR if triangles else 1.0) * (BENDING_SEPARATOR_WIDTH ** 3 if plate else 1.0)

    def scales(cudss):
        if cudss:
            return {'factor_scale': cudss_scale, 'solve_scale': cudss_scale}
        return {'factor_scale': host_factor, 'solve_scale': host_fill, 'backend': 'cpu'}
    study = context.study.type
    if study == EIGENFREQUENCY_STUDY:
        cudss = eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS
        complex_pencil = _complex_pencil(settings_values, context)
        resonances = int(context.count('eigenpairs', 1) or 1)   # per shift
        shift_work, _ = eigen_search_counts(context)   # one shift-invert per shift of the eigenfrequency search
        films = _squeeze_film_time(settings_values, mesh, context)   # squeeze_film_damping features, if any
        return te.combine((build, context.solver_work(
            'eigen', unknowns=unknowns, nnz=nnz, value_bytes=16 if complex_pencil else 8, dimensions=2,
            eigenpairs=(2 * resonances if complex_pencil else resonances) + 4, **shift_work, **scales(cudss)))
            + (() if films is None else (films,)))
    if study == FREQUENCY_RESPONSE_STUDY:   # the factorisations are the study's harmonic_factorizations (sweep_method)
        from ...model.definition_helpers import frequency_sweep_counts, reduced_sweep_time_items
        sweep = frequency_sweep_counts(context)
        # a batched_direct node: per-system cuDSS / SuperLU at the run backend, batched sweeps on the batch counters
        batched = batched_direct_sweep(context) is not None
        per_system = batched_direct_back_end(context.solvers, context.backend)
        cudss = resolved_back_end(context.solvers) in CUDSS_BACK_ENDS or per_system in CUDSS_BACK_ENDS
        pml = bool(_pml_faces(settings_values)) and triangles
        stiffness_bytes = 16 if pml or context.material_sets(LOSS_ARRAY) else 8
        # ports add their DtN block per frequency (frequency_terms): on a DevicePencil at slots mapped once
        # (factorize_pencil extra=, a few entries), else with the reduced operator by pattern_sum on the host
        ports = _feature_count(context, 'port')
        frequency_items = harmonic_frequency_items(
            unknowns, nnz, sweep, device_pencil=cudss and not batched,
            term_value_bytes=(stiffness_bytes, 16 if pml else 8), frequency_terms=bool(ports), batched=batched)
        port_items = ()
        if triangles and _plate(context) and ports:   # dense strip pencil per port and frequency
            strip = 4.0 * math.sqrt(float(mesh.n_nodes))
            # batched: the batch's DtN blocks first, the outputs per frequency later (the one-omega cache misses twice)
            port_items = (te.TimeItem('plate port strip modes (per port and frequency)', 'solve',
                                      ports * sweep['factorizations'] * (2.0 if batched else 1.0),
                                      sweep['factorizations_kind'],
                                      (te.DenseFlops((PLATE_STRIP_EIG_FLOPS + PLATE_STRIP_DTN_FLOPS) * strip ** 3,
                                                     host=True), te.PythonCalls(60))),)
        parts = [build, te.TimeEstimate((*frequency_items, *port_items, *reduced_sweep_time_items(sweep, unknowns))),
                 context.solver_work(
            'linear', unknowns=unknowns, nnz=nnz, value_bytes=16, dimensions=2, solves=sweep['solves'],
            solves_kind=sweep['solves_kind'], factorizations=sweep['factorizations'],
            factorizations_kind=sweep['factorizations_kind'], device_values=cudss, batched=batched, pencil_terms=2,
            **scales(cudss))]
        if sweep['eigenpairs']:   # modal basis: one shift-invert of the 'eigen' role at the band centre
            cudss_eigen = eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS
            parts.append(context.solver_work(
                'eigen', unknowns=unknowns, nnz=nnz, value_bytes=16 if _complex_pencil(settings_values, context) else 8,
                dimensions=2, solves=1.0, factorizations=1.0, eigenpairs=sweep['eigenpairs'] + 4,
                **scales(cudss_eigen)))
        return te.combine(tuple(parts))
    return build


def _mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
    """GUI text of the discretisation on mesh_kind and, on rectangular cells of a 2D model, the refusal."""
    order = 'quadratic' if element_order(settings_values) == 2 else 'linear'
    if mesh_kind == 'triangles':
        return (f'Finite elements on triangles ({order} elements) for the complex out-of-plane displacement: one direct '
                'factorisation per frequency on the host (or on the GPU with a cuDSS linear solver, for large '
                'meshes), resonances by a shift-invert eigen solve. Curved outlines are '
                'followed exactly with quadratic elements; an optional absorbing layer (PML) rings the chosen faces. '
                'With a bending stiffness the plate term uses an interior penalty on the slope jumps between quadratic '
                'elements (the absorbing layer and ports work for the plate too, ports with numerically computed '
                'guide modes; no mode projection); periodic faces give Bloch cells.',
                [])
    if spec.dimensions == 1:
        return (f'Line elements ({order}; cubic with value and slope when a material sets a bending stiffness) on the '
                'cells of the 1D model: a string or plate strip, or the cross-section of a membrane guide with the '
                'axial wavenumber setting.', [])
    return ('Rectangular cells are not supported for a 2D membrane.',
            ['A 2D membrane needs a triangle mesh: change the mesh kind to triangles.'])


# ---- squeeze_film_damping estimates (uncalibrated, as definition3d's): the film's Lagrange dofs bounded by the mesh's
# P2 dofs, stored entries per row, host SuperLU of the complex film operator over the 5-point curve
_SQUEEZE_FILM = 'squeeze_film_damping'
_FILM_ENTRIES_PER_DOF = 15
_FILM_LU_TIME_SCALE = 3.0
_FILM_PYTHON_CALLS = 40      # load, solve and force per film and resonance
_FILM_SETUP_CALLS = 200      # dof map, conditions and the mass matrix per film (once per footprint)
_INT64_BYTES = 8


def _film_unknowns(mesh, n_selected_elements=None) -> int:
    """P2 film dofs: the mesh's nodes + edges, scaled by the selected element share when known (an upper bound)."""
    dofs = int(mesh.n_nodes) + int(mesh.n_edges)
    if n_selected_elements:
        dofs = int(dofs * min(1.0, float(n_selected_elements) / max(int(mesh.n_elements), 1))) + 1
    return max(dofs, 1)


def _squeeze_film_time(settings_values, mesh, context):
    """Eigenfrequency study with enabled squeeze films on triangles: per film one set-up and factor (incompressible:
    reused for every mode), per film and resonance a complex SuperLU factor + solve (bounded); None without films."""
    films = _feature_count(context, _SQUEEZE_FILM) if mesh.kind == 'triangles' else 0
    if not films:
        return None
    unknowns = float(_film_unknowns(mesh))
    solves = float(films) * eigen_search_counts(context)[1]   # per film and kept resonance
    lu = (te.HostLU(unknowns, dimensions=2, complex=True, scale=_FILM_LU_TIME_SCALE),
          te.HostSolve(unknowns, dimensions=2, complex=True, scale=_FILM_LU_TIME_SCALE))
    return te.TimeEstimate((
        te.TimeItem('squeeze film set-up (per film)', 'setup', float(films), 'exact',
                    (te.HostStream(_FILM_ENTRIES_PER_DOF * unknowns * (COMPLEX128_BYTES + _INT64_BYTES)),
                     te.PythonCalls(_FILM_SETUP_CALLS))),
        te.TimeItem('squeeze film pressure solves (per film and resonance)', 'solve', solves, 'bounded',
                    lu + (te.HostStream(_FILM_ENTRIES_PER_DOF * unknowns * COMPLEX128_BYTES),
                          te.PythonCalls(_FILM_PYTHON_CALLS)))))


def _squeeze_film_memory(values, n_selected_cells, mesh):
    """Host items of one squeeze film (triangles): dof maps and vectors, the complex pressure operator with its film
    mass and reduced copy, and its complex SuperLU factor (heuristic, uncalibrated)."""
    unknowns = _film_unknowns(mesh, n_selected_cells)
    entries = _FILM_ENTRIES_PER_DOF * unknowns
    return MemoryEstimate.from_items(host_items=(
        ('squeeze film dof maps and vectors', 12 * _INT64_BYTES * unknowns + 6 * COMPLEX128_BYTES * unknowns),
        ('squeeze film complex pressure operator, mass and reduced copy', 3 * COMPLEX128_CSR_ENTRY_BYTES * entries),
        ('squeeze film complex SuperLU factor (heuristic)', int(FEM_FILL_FACTOR * host_lu_bytes(unknowns)))))


# ---- type ----
_PRESSURE_PARAMS = (
    Param('pressure', 'float', 1.0, label='Pressure amplitude', unit='Pa', field_dependent=True,
          help='Transverse pressure amplitude p [Pa] (positive along +w) at the phase below; may depend on x and y. The '
               'PML ring takes none.',
          tip='Amplitude of the pressure pushing the membrane out of its plane.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg',
          help='Phase of the drive phasor in degrees (exp(+j omega t)).',
          tip='Phase shift of this drive in degrees.'),
)
_LINE_FORCE_PARAMS = (
    Param('force', 'float', 1.0, label='Force per length', unit='N/m',
          help='Transverse force per length [N/m] along the selected outer faces / object outlines (positive along '
               '+w), at the phase below; in a 1D model at the selected end, per unit length along y. Several on one '
               'face add; on a clamped face it does nothing.',
          tip='Out-of-plane force per length of edge.'),
    _PRESSURE_PARAMS[1],
)
_POINT_FORCE_PARAMS = (
    Param('force', 'float', 1e-6, label='Force', unit='N',
          help='Transverse point force [N] (positive along +w) at the phase below; in a 1D model the point is a line '
               'across the strip and the force is per unit length along y (N/m).',
          tip='Out-of-plane force at the point.'),
    _PRESSURE_PARAMS[1],
)
_PORT_PARAMS = (
    Param('mode_count', 'int', 10, label='Modes', unit='1', min=1, max=200,
          help='Number n of strip modes phi_n (sin(n pi s / W); with a symmetry wall sin / cos((n - 1/2) pi s / W), see '
               'the feature description) the port absorbs exactly (Dirichlet-to-Neumann per mode, '
               'propagating and evanescent); higher modes see a free edge, so keep the port where their near field has '
               'decayed (exp(-n pi d / W)) or raise the count. W = the span of membrane elements along the face. A '
               'plate port (bending_stiffness > 0 at the face) absorbs every numeric strip mode of its cross-section '
               '(two per free strip dof); there mode_count is the number of modes reported and allowed as incident '
               'modes, numbered by cut-off (Re beta^2 descending: propagating modes first, then by decay; a plate '
               'guide has a second, bending-dominated evanescent family that comes late), and it must cover the '
               'propagating modes for the power balance.',
          tip='How many guide modes the port absorbs without reflection; 10 is plenty unless a scatterer is close.\n\n'
              'On a plate the port absorbs every mode anyway; the count sets how many are reported.'),
    Param('incident_mode', 'int', 0, label='Incident mode', unit='1', min=0, max=400,
          help='Mode number m sent into the domain through this port (0 = none: the port only absorbs). Its amplitude '
               'below is the incident w profile amplitude at the port face. On a guide with a wall on a symmetry / '
               'antisymmetry face, m is numbered on the unfolded (full-width) guide: a symmetry wall admits the odd m '
               '(half mode (m + 1) / 2), an antisymmetry wall the even m (half mode m / 2); a mode of the other '
               'parity is a build error unless the study runs parity_sectors (each mode then in the run of its '
               'parity). A plate port numbers its modes on the drawn guide (by cut-off, see mode_count), also on a '
               'half guide, and has no parity_sectors combination.',
          tip='Which guide mode enters the model here; 0 means the port only absorbs.\n\nIn a half model the number '
              'counts the modes of the full-width guide.'),
    Param('amplitude', 'float', 1e-9, label='Incident amplitude', unit='m',
          help='Amplitude A [m] of the incident mode A phi_m(s) at the port face, at the phase below.',
          tip='Size of the incoming wave at the port.'),
    _PRESSURE_PARAMS[1],
    Param('incident_modes', 'float_list', None, label='Incident modes', unit='1', min=1.0, optional=True,
          help='Several incident modes at once (whole numbers, numbered as incident_mode); a non-empty list replaces '
               'incident_mode. Their amplitudes / phases: incident_amplitudes / incident_phases (one entry per mode; '
               'empty: amplitude / phase for each). Modes of both parities about a mirror wall (e.g. 1 and 2 on a half '
               'guide) need the study setting parity_sectors.',
          tip='Several guide modes entering together, for example 1 and 2; leave empty to use the single incident '
              'mode above.'),
    Param('incident_amplitudes', 'float_list', None, label='Incident amplitudes', unit='m', optional=True,
          help='Amplitude [m] per entry of incident_modes (empty: amplitude for each).',
          tip='Size of each incoming mode, in the order of the incident modes.'),
    Param('incident_phases', 'float_list', None, label='Incident phases', unit='deg', optional=True,
          help='Phase [deg] per entry of incident_modes (empty: phase for each).',
          tip='Phase of each incoming mode, in the order of the incident modes.'),
    Param('sector_drive', 'bool', False, label='Parity-sector drive',
          help='Set by the study setting parity_sectors on its sector copies (leave it off): an incident mode of a '
               'guide ending on a mirror face is then driven at 2 A in the sector of its parity and not at all in '
               'the other, so the mean of the sectors is the drawn amplitude A.',
          tip='Set automatically for the runs of the parity sectors study option; leave it off.'),
    Param('tension_tolerance', 'float', PORT_UNIFORM_RTOL, label='Tension tolerance', unit='1', min=0.0,
          min_exclusive=True, max=1.0,
          help='Largest relative variation of the tension along the port cut (spread of T_nn and of T_ss over the '
               'elements at the face, |tension_xy| / T_nn) that is accepted; above it the build of the first solve '
               f'fails. Default {PORT_UNIFORM_RTOL:g} (a uniform guide). Raise it (e.g. 1e-2) for a guide whose tension '
               'comes from a solid (solid_membrane_tension coupling): the port then uses the mean tension of its cut, '
               f'a spread above {PORT_UNIFORM_RTOL:g} gives a RuntimeWarning and the result extras tension_spread / '
               'medium_warning, and the port reflects at about the level of the spread. Areal density and loss factor '
               f'must stay uniform to {PORT_UNIFORM_RTOL:g}.',
          tip='How much the tension may vary along the port before it is refused.\n\nKeep the default for a uniform '
              'film; raise it for a tension taken from a solid, at the cost of small reflections.'),
)
_PROJECTION_PARAMS = (
    Param('guide_axis', 'choice', 'y', label='Guide direction', choices=_AXES,
          choice_labels=(('x', 'Along x'), ('y', 'Along y')),
          help='Axis the guide runs along; the projection integrates across the other axis.',
          tip='Direction in which the guide runs.'),
    Param('cross_start', 'float', None, label='Guide wall start', unit='m', optional=True,
          help='Coordinate [m] of the first guide wall across the guide (empty: the window edge; on a symmetry face the '
               'wall is free).',
          tip='Position of the first wall of the guide; empty uses the domain edge.'),
    Param('cross_end', 'float', None, label='Guide wall end', unit='m', optional=True,
          help='Coordinate [m] of the second guide wall (empty: the window edge; on a symmetry face the wall is free); '
               'W = cross_end - cross_start.',
          tip='Position of the second wall of the guide; empty uses the domain edge.'),
    Param('start', 'float', None, label='First position', unit='m', optional=True,
          help='First sampled position [m] along the guide (empty: the window edge).',
          tip='Where the sampled stretch of the guide starts; empty uses the domain edge.'),
    Param('end', 'float', None, label='Last position', unit='m', optional=True,
          help='Last sampled position [m] along the guide (empty: the window edge).',
          tip='Where the sampled stretch of the guide ends; empty uses the domain edge.'),
    Param('points', 'int', 101, label='Positions', unit='1', min=2, max=100000,
          help='Number of equally spaced positions from start to end.',
          tip='How many positions along the guide are sampled.'),
    Param('mode_count', 'int', 3, label='Modes', unit='1', min=1, max=200,
          help='Number of profiles phi_n, n = 1 .. mode_count, projected on (sin(n pi (s - cross_start) / W); a wall on a '
               'symmetry face: sin / cos((n - 1/2) pi (s - cross_start) / W), see the feature description).',
          tip='How many guide modes are extracted.'),
    Param('result_axis', 'choice', 'position', label='Result versus', choices=('position', 'frequency'),
          choice_labels=(('position', 'Position along the guide'), ('frequency', 'Frequency (window-averaged power)')),
          help="'position' (default): a_n(p) of the shown solution versus position. 'frequency' (frequency_response "
               'only): per recorded frequency the mean of |a_n|^2 over the positions in average_start .. average_end '
               '(series mode_<n>_power [m^2], mode_<n>_rms [m], total_power [m^2]; see the feature description).',
          tip='Show the mode amplitudes along the guide, or the guide-averaged mode power versus frequency.'),
    Param('average_start', 'float', None, label='Averaging start', unit='m', optional=True,
          visible_if=(('result_axis', ('frequency',)),),
          help="result_axis 'frequency': first position [m] of the averaging window along the guide (empty: start); "
               'the mean runs over the sampled positions inside the window.',
          tip='Where the averaging along the guide starts; empty uses the first sampled position.'),
    Param('average_end', 'float', None, label='Averaging end', unit='m', optional=True,
          visible_if=(('result_axis', ('frequency',)),),
          help="result_axis 'frequency': last position [m] of the averaging window (empty: end). A build error when "
               'no sampled position lies in the window.',
          tip='Where the averaging along the guide ends; empty uses the last sampled position.'),
)
_GAS_COLUMNS_TEXT = (
    'per mode for the mode shape of the eigenfrequency study (peak |w| = 1 m): gas_damping c = -Re F [kg/s] and '
    'gas_stiffness k = omega Im F [N/m] of the modal gas force F = int p conj(w) dA per unit modal velocity (both scale '
    'with the square of the shape normalisation), gamma_gas = c / m_eff [1/s] (the energy decay rate, the linewidth in '
    'rad/s) and q_gas = omega m_eff / c, with m_eff = int rho_s |w|^2 dA of the same shape (independent of it)')
_SQUEEZE_FILM_PARAMS = (
    *SQUEEZE_FILM_GAS_PARAMS,   # gap, viscosity, ambient_pressure, slip, mean_free_path (model.definition_helpers)
    Param('vents', 'choice', 'open_edges', label='Vented edges', choices=MEMBRANE_VENT_CHOICES,
          choice_labels=(('open_edges', 'Free edges and holes vented'), ('all', 'Every edge vented'),
                         ('none', 'Closed (no vent)')),
          help='Which footprint boundary edges vent to the ambient (p = 0; the others are sealed, no gas flux). '
               'open_edges: free edges, hole outlines, footprint edges inside the membrane and antisymmetry faces '
               '(odd pressure) vent; window edges of clamped / simply_supported / symmetry faces, symmetry outlines '
               'and held edges with membrane on both sides (clamped_line / simply_supported_line, clamped / supported '
               'object outlines inside the film) are sealed (a clamped outline with a hole outside vents as a hole '
               'outline). all: every footprint edge vents; none: a closed film (needs ambient_pressure > 0: a trapped '
               'gas spring). Holes vent without the flow resistance of a perforation.',
          tip='Where the gas can escape from the gap: by default at free edges and through holes; clamped edges and '
              'symmetry planes are sealed.'),
)

_FEATURES = (
    FeatureType('clamped', 'Clamped', 'boundary_condition', 'boundary',
                summary='Holds the membrane edge still: the selected domain edges or outlines do not move (nor tilt, '
                        'for a plate).',
                description='w = 0 on the selected outer faces (the window edge) and object outlines (triangle meshes; '
                            'a 1D model: its ends x_lo / x_hi); with a bending_stiffness also the slope dw/dn = 0 '
                            '(1D: w\' = 0; triangles: weakly, Nitsche, on each side of an outline). Edges without a '
                            'clamp are free (zero transverse force and moment). A clamp beats a line force on the same '
                            'face; not on a periodic face. Line objects (a zero-thickness iris or wall): clamped_line.',
                apply=_apply_clamped, validate=_validate_boundary_feature, grids=_GRIDS),
    FeatureType('clamped_line', 'Clamped line', 'boundary_condition', 'line', object_kinds=('line',),
                summary='Holds the membrane still along drawn lines inside it, such as a thin clamped wall or iris '
                        '(nor tilt, for a plate).',
                description='w = 0 on the mesh edges of the selected line objects (triangle meshes: the mesher puts '
                            'element edges on every line; any direction, the part inside the window); with a '
                            'bending_stiffness also dw/dn = 0 weakly (Nitsche) on each side of the line. The two sides '
                            'stay one membrane otherwise: a line across a strip splits it into two clamped strips. '
                            'Validation warns for a line end on a periodic side (unless the line spans the period: '
                            'its image must be held too) and for a line reaching beyond the window.',
                apply=_apply_clamped, validate=_validate_line_hold, grids=('triangles',)),
    FeatureType('simply_supported_line', 'Simply supported line', 'boundary_condition', 'line',
                object_kinds=('line',),
                summary='Holds the membrane in place along drawn lines inside it but lets a plate rotate about them.',
                description='w = 0 on the mesh edges of the selected line objects (triangle meshes, as clamped_line), '
                            'the slope free: with a bending_stiffness the slope stays continuous across the line (the '
                            'interior penalty acts there as on any interior edge: a line support of a continuous '
                            'plate). Without bending_stiffness the same as clamped_line.',
                apply=_apply_simply_supported, validate=_validate_line_hold, grids=('triangles',)),
    FeatureType('simply_supported', 'Simply supported', 'boundary_condition', 'boundary',
                summary='Holds the edge in place but lets it tilt: the selected edges do not move, a plate may rotate '
                        'about them.',
                description='w = 0 on the selected outer faces and object outlines (1D: the ends), the slope free (zero '
                            'bending moment: M_nn = 0 with plate_poisson_ratio). Without bending_stiffness the same as '
                            'clamped. Not on a periodic face. Line objects: simply_supported_line.',
                apply=_apply_simply_supported, validate=_validate_boundary_feature, grids=_GRIDS),
    FeatureType('symmetry', 'Symmetry', 'boundary_condition', 'boundary',
                summary='Makes the selected edges a mirror plane for vibrations that are the same on both sides of '
                        'it, so only half (or a quarter) of a symmetric membrane needs to be drawn.',
                description='Even mirror plane: w(mirrored) = w. A film: the free edge (T dw/dn = 0, nothing is held; '
                            'needs tension_xy = 0 there, as a mirrored shear tension changes sign). With a '
                            'bending_stiffness: the slope dw/dn = 0 (1D: w\' = 0; triangles: weakly, Nitsche, as a '
                            'clamp) with w free and zero transverse shear. On outer faces (1D: x_lo / x_hi) and straight '
                            'object outlines; not on a PML or periodic face. A face also held by clamped / '
                            'simply_supported / antisymmetry is held (warning). With the antisymmetry feature on the '
                            'same plane in a second run it gives every mode of the full structure: the even modes here, '
                            'the odd ones there. A port or mode_projection guide ending on a symmetry face takes that '
                            'wall as free (half-guide profiles, see port). Its outer faces are model mirror planes '
                            '(core.mirror): fields unfold w even across them, integrals touching them count twice '
                            '(the unfolded structure), and the study setting parity_sectors swaps symmetry / '
                            'antisymmetry on them per sector run.',
                apply=_apply_symmetry, validate=_validate_boundary_feature, grids=_GRIDS),
    FeatureType('antisymmetry', 'Antisymmetry', 'boundary_condition', 'boundary',
                summary='Makes the selected edges a mirror plane for vibrations that change sign across it, so only '
                        'half (or a quarter) of a symmetric membrane needs to be drawn.',
                description='Odd mirror plane: w(mirrored) = -w. w = 0 on the selected faces, the slope free (with a '
                            'bending_stiffness: zero bending moment M_nn = 0), the same hold as simply_supported. On '
                            'outer faces (1D: x_lo / x_hi) and straight object outlines; not on a PML or periodic face. '
                            'Its outer faces are model mirror planes (core.mirror): fields unfold w odd across them, '
                            'integrals touching them count twice, parity_sectors swaps the feature per sector run. A '
                            'squeeze_film_damping film vents on its faces (odd pressure).',
                apply=_apply_antisymmetry, validate=_validate_boundary_feature, grids=_GRIDS),
    FeatureType('pressure_load', 'Pressure load', 'source', 'domain', params=_PRESSURE_PARAMS,
                summary='Drives the membrane with a pressure on the selected domains (frequency response).',
                description='Complex pressure p e^(j phase) [Pa] on the selected elements (cells in 1D), int p v dA; '
                            'read by the frequency_response study only.',
                apply=_apply_pressure, grids=_GRIDS),
    FeatureType('line_force', 'Line force', 'source', 'boundary', params=_LINE_FORCE_PARAMS,
                summary='Drives the membrane with a force along the selected edges (frequency response).',
                description='Complex transverse force per length F e^(j phase) [N/m] on the selected outer faces (the '
                            'window edge) and object outlines (both sides of an interior outline share it); a 1D model: '
                            'at its ends. Read by the frequency_response study only; not on a PML face.',
                apply=_apply_line_force, validate=_validate_boundary_feature, grids=_GRIDS),
    FeatureType('point_force', 'Point force', 'source', 'point', params=_POINT_FORCE_PARAMS, object_kinds=('point',),
                summary='Drives the membrane with a force at the selected points (frequency response).',
                description='Complex transverse point force F e^(j phase) [N] at each selected point (exact delta load '
                            'of the element holding it); 1D models: a line across the strip, N/m. Read by the '
                            'frequency_response study only.',
                apply=_apply_point_force, validate=validate_points_inside, grids=_GRIDS),
    FeatureType('field_probe', 'Displacement probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
                result_kind='xy',
                summary='Records the complex out-of-plane displacement at a point for every frequency of a frequency '
                        'response.',
                description="Complex w at the point (triangles: its mesh node; 1D: x on the line) per solved frequency "
                            "(core.harmonic_probe). Result 'xy' versus frequency [Hz]: series w_re, w_im, w_abs [m] and "
                            'w_phase [rad] (a repeated frequency replaces its sample); a two_wave_fit derived value pairs '
                            'w_re with w_im, a lorentz_fit of w_abs*w_abs gives the quality factor. Frequency response '
                            'only.',
                apply=_apply_probe, result=_probe_result, validate=_validate_probe, grids=_GRIDS),
    FeatureType('resonance_table', 'Resonance table', 'monitor', None, result_kind='table',
                summary="Lists the resonances found by an 'Eigenfrequency' study: frequency, damping and quality "
                        'factor.',
                description="Resonances of the last eigenfrequency solve, nearest target_frequency first. Result 'table' "
                            'with columns resonance_number, frequency [Hz], frequency_imag [Hz, > 0 decays], q_factor '
                            '(inf when lossless), pml_fraction (energy share in the PML); extras target_frequency, '
                            "selected_resonance. For a frequency versus a parameter add a derived value with source "
                            "'<physics>.<this feature>', expression frequency, operation at and at = 0. With "
                            f"{_SQUEEZE_FILM} features (triangle meshes) the table appends {', '.join(GAS_TABLE_COLUMNS)}: "
                            'the totals of every enabled film (c, k and gamma add over the films; see that feature).',
                apply=_apply_resonance_table, result=_resonance_table_result, validate=_validate_resonance_table,
                grids=_GRIDS),
    FeatureType(_SQUEEZE_FILM, 'Squeeze-film damping', 'monitor', 'domain', params=_SQUEEZE_FILM_PARAMS,
                result_kind='table',
                summary='Estimates how much a thin gas layer between the membrane and a fixed wall under or over it '
                        'damps each vibration mode, as in MEMS resonators.',
                description=('Gas damping of a thin film (gap H) between the selected membrane domains (the footprint) '
                             'and a fixed parallel wall, evaluated per resonance after the eigenfrequency solve '
                             '(triangle meshes; it does not change the modes; c and k are quadratic in w, so the side '
                             'of the wall does not matter). On the footprint elements (holes excluded) the linearised '
                             'compressible Reynolds equation div(H^3 / (12 mu_eff) grad p) - j omega (H / p_a) p = j '
                             'omega w is solved per mode on the membrane\'s own elements and order (fem.squeeze_film), '
                             'w the mode shape (peak 1 m), p = 0 on vented footprint edges (vents), no flux on sealed '
                             f"ones. Result 'table', one row per mode: resonance_number, frequency [Hz], effective_mass "
                             f'[kg], {_GAS_COLUMNS_TEXT}; extras area [m^2], gap [m], film_elements, vent_edges. The '
                             'Resonance table appends the totals over every enabled film. Complex modes (PML, loss) use '
                             'Re omega and m_eff = Re(w^H M w). Refused: Bloch cells, a footprint reaching a PML ring '
                             '(build error), 1D models. Only the eigenfrequency study evaluates it (a frequency_response '
                             'is not damped by it).'),
                apply=_apply_squeeze_film, result=_squeeze_film_result, validate=_validate_squeeze_film,
                memory_estimate=_squeeze_film_memory, grids=('triangles',)),
    FeatureType('port', 'Port', 'boundary_condition', 'boundary', params=_PORT_PARAMS, max_entities=1,
                allow_interior=False, result_kind='xy',
                summary='Ends a straight clamped guide at an outer face as if it continued forever, and can send a '
                        'guide mode in; reports the power leaving in each mode (frequency response).',
                description="Exact semi-infinite termination of a straight clamped strip guide on one outer face "
                            "(triangle meshes, frequency_response only; a plate face: last paragraph): the analytic "
                            'profiles sin(n pi s / W) (s from the first wall, W = the span of membrane elements on the '
                            'face, clamped at both ends) with beta_n^2 = (rho_s omega^2 - T_ss (n pi / W)^2) / T_nn, Im '
                            'beta <= 0, give the Dirichlet-to-Neumann term sum_n j beta_n T_nn (2 / W) (int phi_n w)(int '
                            'phi_n v) per mode (no numeric mode solve); incident_mode m > 0 adds the load of A phi_m '
                            'entering. A guide wall on a window face with the symmetry feature (and no hold there) is a '
                            'free mirror wall: phi_n = sin(k_n s) (free at s = W) or cos(k_n s) (free at s = 0), k_n = '
                            '(n - 1/2) pi / W in beta_n, i.e. the even modes 2n - 1 of the unfolded guide of width 2 W '
                            '(same amplitudes; powers and power_fraction of the half; the incident mode 1 of the full '
                            'guide is mode 1 here); symmetry on both walls is refused. The guide must be uniform at the face (density and loss to 1e-6; the tension '
                            'and tension_xy to tension_tolerance, the mean tension of the port elements being used) '
                            'and not ringed by a PML. Result \'xy\' versus frequency [Hz]: per mode the outgoing '
                            'amplitude b_n = a_n - A delta_nm at the face (series mode_<n>_re / _im / _abs [m], '
                            '_phase [rad]), power_<n> = W omega Re(T_nn beta_n) |b_n|^2 / 4 [W] and power_fraction_<n> '
                            '= power_<n> / incident_power (the incident power of every port; NaN without one), and '
                            'incident_power [W]. Lossless: the outgoing powers of all ports sum to the incident power '
                            '(transmission T and reflection R). Extras: tension_spread (the relative tension spread '
                            'measured at the face), tension_tolerance, medium_warning (text when the spread exceeds '
                            '1e-6 but is accepted; else empty), free_walls ([at s = 0, at s = W]; None before a solve). '
                            'A guide wall on a symmetry / antisymmetry face (a mirror wall): incident modes are '
                            'numbered on the unfolded guide (see incident_mode; incident_modes for several), powers '
                            'are those of the unfolded guide (x 2 per mirror wall; fractions unchanged), and the '
                            'extras mirror_walls (+1 symmetry, -1 antisymmetry, None per wall), sector_modes / '
                            'sector_signs (the unfolded number and sign of each half mode: a symmetry wall 2k - 1, an '
                            'antisymmetry wall 2k, signed (-1)^(k-1) / (-1)^k when the wall is at s = 0) and '
                            'sector_drive_scale give the parity_sectors combination (Results.sector_result(..., '
                            'combined=True): unfolded modes, recomputed powers and fractions); series '
                            'port_incident_power = this port\'s own incident power [W]. Plate port (bending_stiffness > '
                            '0 at the port elements, D and plate_poisson_ratio uniform there to 1e-6): the guide '
                            'modes are computed per frequency from a Hermite cubic strip on the port\'s edge nodes '
                            '(D (1 + j eta), T_nn, T_ss, rho_s, nu; walls from the holds: a clamped face / outline holds '
                            'w and dw/ds, simply_supported / antisymmetry w, symmetry dw/ds, an unheld face or hole '
                            'outline is a free plate edge), the quadratic eigenproblem in beta^2 solved densely (2 per '
                            'free strip dof), Im beta <= 0, numbered by cut-off (Re beta^2 descending), profiles scaled '
                            'to peak |w| 1 with the first lobe positive. The port absorbs all of them: the exact '
                            'Dirichlet-to-Neumann map of the strip (effective shear V_n against w, moment M_nn against '
                            'dw/dn; complex symmetric by reciprocity) on every quadratic dof of the port elements. '
                            'b_n = the outgoing amplitudes from the (w, dw/dn) traces, power_<n> = the modal energy flux '
                            '(omega / 2) Im(V_n conj(w) + M_n conj(dw/dn)) (lossless: exact balance), series as above. '
                            'Extras plate_walls (the wall kinds; None for a film port), sector_modes / sector_signs '
                            'None (no parity_sectors combination: a plate port on a guide ending on a mirror face is '
                            'refused under parity_sectors); mirror walls still double the powers.',
                apply=_apply_port, result=_port_feature_result, validate=_validate_port, grids=('triangles',)),
    FeatureType('mode_projection', 'Mode projection', 'monitor', None, params=_PROJECTION_PARAMS, result_kind='xy',
                summary='Splits the vibration of a straight guide into its guide modes along the guide, for the '
                        'shown solution and, in a frequency response, for every frequency.',
                description='a_n(p) = (2 / W) int w(p, s) sin(n pi (s - cross_start) / W) ds at points positions p '
                            'from start to end along guide_axis, s across from cross_start to cross_end (W = their '
                            'distance; a wall on a window face with the symmetry feature is free: the port\'s half-guide '
                            'profiles, extra free_walls; composite Gauss; the sample points are located once at build, a point outside '
                            'the mesh is a build error). Result \'xy\' versus position [m] of the shown solution '
                            '(resonance or the last frequency): series mode_<n>_re / _im / _abs [m], _phase [rad]; a '
                            'two_wave_fit or two_exp_fit derived value of mode_1_re gives beta or kappa of a mode. A '
                            'frequency_response run also records a_n(p) at every solved frequency (independent of the '
                            'study snapshots setting; host 16 B x points x mode_count per frequency): extras '
                            'frequencies [Hz] (ascending, a repeated frequency replaces its row) and frequency_series '
                            '{mode_<n>_re / _im / _abs / _phase: (n_frequencies, points) array, row i at frequencies[i], '
                            'column j at x[j]} (results.npz result/<physics>.<feature>/frequency_series/<name>). '
                            "result_axis 'frequency' instead gives result 'xy' versus those frequencies [Hz]: "
                            'mode_<n>_power = the mean of |a_n|^2 over the sampled positions in average_start .. '
                            'average_end [m^2] (a guide-averaged mode power, insensitive to standing-wave ripple), '
                            'mode_<n>_rms = its square root [m] and total_power = the sum over the modes [m^2]; extras '
                            'average_window [m] (first and last averaged position) and average_points; an error before '
                            'a frequency_response run. An output / input ratio is a derived curve dividing the '
                            'mode_<n>_power series of two such projections (that view also stores extras '
                            'averaged_series {mode_<n>_re / _im: (n_frequencies, averaged points)}). Extras '
                            'mirror_walls / sector_modes / sector_signs as the port (parity_sectors combines the '
                            'projection on the unfolded guide). Triangle meshes; refused with bending_stiffness (the '
                            'film profiles are not the plate guide modes, which are numeric per frequency: use ports '
                            'or field probes).',
                apply=_apply_mode_projection, result=_mode_projection_result, validate=_validate_mode_projection,
                grids=('triangles',)),
)

_FIELD_INFOS = tuple(MEMBRANE_FIELD_INFOS.values())
_STATE_FIELD_INFOS = tuple(MEMBRANE_FIELD_INFOS[name] for name in STATE_FIELD_NAMES)

_DESCRIPTION = (
    'Tensioned film (the industry-standard membrane), optionally with plate bending (a tensioned Kirchhoff plate), '
    'time-harmonic (exp(+j omega t)): D (1 + j eta) del^4 w - div(T (1 + j eta) grad w) - rho_s omega^2 w = p, w the '
    'out-of-plane displacement [m], T the tension tensor [[tension_xx, tension_xy], [tension_xy, tension_yy]] [N/m] '
    '(film stress x thickness; material + the coupled_tension_* slots), D = bending_stiffness [N m] (unset: no '
    'bending), rho_s = areal_density [kg/m^2], eta = loss_factor, p = pressure [Pa]. Waves travel at c = sqrt(T / '
    'rho_s). 2D models on a triangle mesh (complex Lagrange P2 / P1, isoparametric on curved outlines; fem.harmonic; '
    'with D the C0 interior penalty plate form on P2: slope jumps between elements penalised, clamps hold dw/dn = 0 '
    'weakly); 1D models on line elements over the cells (Hermite cubic with D; a string / plate strip or the '
    'cross-section of a guide along y with the axial_wavenumber setting k_y: + T_yy (1 + j eta) k_y^2, with D also 2 D '
    "k_y^2 w'' and D k_y^4 terms). An element with no tension, areal_density or bending_stiffness is a hole (not "
    'solved; its outline is a free edge). Bloch cells: periodic_x / _y with bloch_kx / _ky on a periodic triangle mesh '
    '(eigenfrequency). Ports: exact semi-infinite clamped strip guides on outer faces with analytic modes (plates: '
    'numeric strip modes), incident modes and power fractions (frequency_response); mode_projection gives guide mode '
    'amplitudes along a film guide. Studies: eigenfrequency '
    '(resonances nearest target_frequency by shift-invert on the host, eigen '
    'solver role arpack_shift_invert; complex omega with Im > 0 for a decaying resonance, Q = Re omega / (2 Im omega) '
    'from the loss factor and the PML; one snapshot per resonance labelled "resonance k: f = <Hz> Hz[, Q = ...]", each '
    'resonance scaled to 1 m real at its peak dof; resonance_table feature) and frequency_response (the driven '
    'response to the pressure_load / line_force / point_force phasors at each frequency in Hz, one complex sparse '
    "factorisation per frequency on the 'linear' sparse_direct solver (back_end 'cudss' / 'cudss_mixed' on the gpu "
    'backend factorises on the device, triangle meshes only, its plan kept over the sweep since the pattern is fixed; '
    "Automatic / 'auto': cudss_mixed there when nvmath-python is installed, else host SuperLU, up to ~1e6 unknowns "
    'practical, 2e6 refused; an opt-in MUMPS or PARDISO host factor on triangle meshes keeps those limits); one '
    'snapshot per frequency (study '
    'setting snapshots); field_probe, port and mode_projection record per frequency). Edges are free unless clamped '
    '(w = 0); clamped_line / simply_supported_line hold drawn line objects inside the membrane (zero-thickness walls, '
    'irises); symmetry / antisymmetry make faces even / odd mirror planes of a half or quarter model; '
    'squeeze_film_damping adds the gas damping of a thin gap under the film per resonance (eigenfrequency). '
    'PML (triangles): pml_<face> settings grow a ring of pml_cells x dx beyond those faces with a fixed complex '
    'stretch (fem.pml, set at pml_frequency or the eigenfrequency target / the first response frequency), outer side '
    'clamped; resonances with most of their energy in the ring are dropped. Fields w_re, w_im, w_abs, w_phase (nodal '
    'on triangles, cell centres in 1D), tension_xx / tension_yy / tension_xy (material + coupled), areal_density; '
    'state() holds '
    'w_re / w_im / w_abs per element (1D: per cell). The instance is usually named membrane.')

MEMBRANE = PhysicsType(
    name=MEMBRANE_TYPE_NAME,
    label='Membrane',
    default_instance_name='membrane',
    family='',
    group='structural',
    spaces_2d=('planar',),
    description=_DESCRIPTION,
    summary='Vibrations of a thin tensioned film such as a silicon nitride membrane or a string, with optional '
            'bending stiffness: its resonance frequencies and quality factors, the bands of a periodic film, or its '
            'response to a pressure, force or guided wave at a list of frequencies.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=dict(MEMBRANE_SCHEMA),
    optional_material_schema=dict(OPTIONAL_MEMBRANE_SCHEMA),
    state_fields=_STATE_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    source_slots=_SLOT_DOCS,
    supported_studies=(EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY),
    default_study={'type': EIGENFREQUENCY_STUDY},
    create=_create,
    validate=_validate_membrane,
    domain_requirements=_domain_requirements,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    dimensions=(1, 2),
    # 'eigen' also in frequency_response: the modes of sweep_method 'modal'
    solver_roles={'linear': MEMBRANE_LINEAR_ROLE,
                  'eigen': eigen_role((EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY))},
    backends=('gpu', 'cpu'),   # host scipy solves either way; state arrays on the Domain's xp
    grids=_GRIDS,
    mesh_method=_mesh_method,
    frequency_sweep_methods=('direct', 'adaptive', 'modal'),
    eigen_searches=('nearest', 'targets', 'band'),
)


def _selftest_cases():
    """The membrane SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import MEMBRANE_SELFTEST_CASES
    from .selftests_parity_sectors import MEMBRANE_PARITY_SECTOR_CASES
    from .selftests_plate_pml import MEMBRANE_PLATE_PML_CASES
    from .selftests_plate_ports import MEMBRANE_PLATE_PORT_CASES
    from .selftests_reduced_sweep import MEMBRANE_REDUCED_SWEEP_CASES
    from .selftests_squeeze_film import MEMBRANE_SQUEEZE_FILM_CASES
    return (MEMBRANE_SELFTEST_CASES + MEMBRANE_SQUEEZE_FILM_CASES + MEMBRANE_REDUCED_SWEEP_CASES
            + MEMBRANE_PARITY_SECTOR_CASES + MEMBRANE_PLATE_PML_CASES + MEMBRANE_PLATE_PORT_CASES)


def register(registry):
    """The membrane type, the 'silicon_nitride_membrane_modes' template, then the selftest loader."""
    registry.register_physics(MEMBRANE)
    registry.register_template(
        'silicon_nitride_membrane_modes', RECTANGLE_LABEL, RECTANGLE_DESCRIPTION, rectangle_modes_spec,
        summary='A clamped rectangular silicon nitride membrane: its lowest drum resonances and mode shapes.')
    registry.register_selftests(_selftest_cases)

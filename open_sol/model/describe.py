"""API reference generated from the registry, the spec dataclasses, the expression vocabulary, the material library
and the templates: describe() gives markdown text or a JSON-able dict (CPU only, deterministic, ASCII)."""
import json
import math
import numbers
import types
from dataclasses import MISSING, asdict, fields, is_dataclass

from ..core.constants import FACE_LABELS
from ..core.renderer import primary_field_info
from ..materials import REGISTRY as MATERIAL_LIBRARY
from ..materials.crystal_tensors import CRYSTALS
from .expressions import (BUILTIN_CONSTANTS, CONSTANT_DESCRIPTIONS, COORDINATE_NAMES, FUNCTION_ARITY,
                          FUNCTION_DESCRIPTIONS, MAX_EXPRESSION_LENGTH, RESERVED_NAMES, arity_text)
from .analytic import (CURVE_FUNCTION_DESCRIPTIONS, CURVE_RESULT_KINDS, DERIVED_RESULT_KINDS, FIT_QUANTITIES,
                       MAX_FIT_DEGREE, PEAK_FIT_QUANTITIES)
from .geometry import GEOMETRY_KINDS
from .meshing import MAX_AXIS_CELLS, MAX_MESH_CELLS
from .schema import (DERIVED_OPERATIONS, FIELD_DERIVED_OPERATIONS, INTERFACE_OPERATIONS, INTERFACE_PROJECTIONS,
                     INTERFACE_VARIABLES, ISSUE_LEVELS, MESH_PARAMS, PARAM_KINDS,
                     RESULT_KIND_DOCS, RESULT_KINDS, SELECTION_KINDS, SOLVER_ROLES, get_registry, node_label,
                     restricted_feature_precisions)
from .spec import _NESTED_FIELDS, OUTER_FACES, RESERVED_ENTITY_NAMES, SPEC_VERSION, MeshSpec, ModelSpec
from .templates import template_entries

REGENERATE_COMMAND = 'python -m open_sol describe --update-reference'   # also in api's selftest messages
REFERENCE_FILE = 'api_reference.md'   # reference_files() keys = the files --update-reference writes beside the package
TEMPLATES_FILE = 'api_templates.md'   # the template JSON (the main file's Templates section is an index)
_CONTENTS_SPLIT_LINES = 400   # a '## ' section longer than this also lists its '### ' subsections in the contents
_LEADING_TOPICS = ('spec', 'expressions', 'geometry', 'materials', 'mesh', 'solvers')
_TRAILING_TOPICS = ('templates', 'results')
_SELFTESTS_TOPIC = 'selftests'   # topic only: the selftest cases (the 'results' section only points to it)
_JSON_LINE_WIDTH = 120   # template JSON: a value stays on one line while it fits, else one key / item per line
_SHARED_TEXT_MIN_CHARS = 150   # full reference: a physics / coupling / study text this long seen twice is shared
_SPEC_CLASS_PLACES = {'list': 'items of {owner}.{field}', 'one': '{owner}.{field}',
                      'optional': '{owner}.{field} (or null)'}   # _NESTED_FIELDS arity -> where the class sits


def _per_line_names(operation):
    """Fitted quantity names of a peak fit as help text: '<name>_<k>' per line, then the shared ones."""
    per_line, shared = PEAK_FIT_QUANTITIES[operation]
    return [f'{name}_<k>' for name in per_line] + list(shared)


# Param kind -> what a spec value of that kind looks like (_table_problems compares the keys with PARAM_KINDS)
_PARAM_KIND_HELP = {
    'float': 'a number or an expression string (see expressions)',
    'int': 'a whole number, or an expression of parameters that evaluates to one',
    'bool': 'true or false',
    'choice': 'one of the listed choices (text)',
    'str': 'text',
    'expression': 'expression text kept as written and evaluated at run time (its help names the allowed names)',
    'ref': 'the name of a model node of the given kind (material, geometry object, physics or parameter); null or '
           '"" = none where optional',
    'float_list': 'a list of numbers or parameter expressions',
    'points': 'a list of [x, y] pairs of numbers or parameter expressions, in m',
    'names': 'a list of node names (of the given kind)',
    'table': 'a list of rows, each a list of one number or parameter expression per column, in the column order and '
             'units listed with the param (range bounds apply to every entry)',
}

# 'SpecClass.field' -> one-line help of every spec dataclass field except the MeshSpec fields of MESH_PARAMS
# (help derived from those Params); _table_problems compares the keys
_SPEC_FIELD_HELP = {
    'ModelSpec.title': 'Free-text name of the model, shown in reports and the GUI.',
    'ModelSpec.description': 'Free text (may span lines) on what the model shows and how to read its results; '
                             'shown in the GUI (model node, Examples menu tooltip), never used by build or run.',
    'ModelSpec.version': f'Spec format version {SPEC_VERSION}; version 1 files (2D only, no size_z / geometry z '
                         'params) load unchanged and are saved as version 2; other versions are rejected on load.',
    'ModelSpec.parameters': 'Named scalar parameters (ParameterSpec); every numeric value may be an expression of '
                            'them.',
    'ModelSpec.domain': 'Size of the physical domain (DomainSpec), whose lower-left corner is the origin, and its '
                        'in-plane grid (Cartesian, or polar / spherical about the origin).',
    'ModelSpec.geometry': 'Geometry objects (GeometrySpec): domain objects, points and lines, in m.',
    'ModelSpec.selections': 'Named entity lists (SelectionSpec), usable wherever entities are selected.',
    'ModelSpec.materials': 'Material assignments (MaterialSpec); list order and background materials decide '
                           'overlapping cells (see materials).',
    'ModelSpec.physics': 'Physics instances (PhysicsSpec) of registered physics types, each with its features.',
    'ModelSpec.couplings': 'Couplings (CouplingSpec) that feed a field of one physics into a source slot of '
                           'another.',
    'ModelSpec.meshes': 'Named meshes (MeshSpec, at least one, unique names; the first is the default): tensor '
                        '(uniform, nonuniform or auto cells plus refinements), triangles (2D) or tetrahedra (3D). Each '
                        'physics runs on one '
                        '(PhysicsSpec.mesh); see mesh. A legacy "mesh" object (one MeshSpec) loads as [that mesh]; '
                        'ModelSpec(mesh=...) and spec.mesh in scripts mean meshes[0].',
    'ModelSpec.solvers': 'Solver nodes (SolverSpec), each a solver type with its settings, used by the physics roles '
                         'that name it in PhysicsSpec.solvers; [] = every physics on Automatic (the default solver of '
                         'each role). A node no physics uses is a warning.',
    'ModelSpec.solver_defaults': 'Model-wide solver choices (SolverDefaultsSpec).',
    'SolverSpec.name': 'Identifier, unique among solvers; PhysicsSpec.solvers refers to it.',
    'SolverSpec.type': 'Registered solver type: pcg, line_block, sparse_direct, arpack_shift_invert, explicit_kernels '
                       '(see solvers: roles, settings and which physics use them).',
    'SolverSpec.settings': 'Values of the solver type settings by name (see solvers); missing keys take the defaults.',
    'SolverDefaultsSpec.backend': "auto (default) | gpu | cpu: where the run computes (api check / run backend=, CLI "
                                  "--backend override it); auto = gpu when cupy is installed, else cpu. cpu needs "
                                  "every physics type to list it in its backends (the Backends line of each physics "
                                  "type; else a validation error per physics) and every solver to fit it (pcg, line_block, sparse_direct without "
                                  "cusolver_qr, arpack_shift_invert do; explicit_kernels does not); on cpu the "
                                  "device memory estimate counts against the host RAM.",
    'SolverDefaultsSpec.precision': (
        "auto (default) | single | double | mixed | half_storage: model-wide solver precision. auto = each solver's "
        "own default (pcg double, explicit_kernels single, ...). Otherwise every role solver whose type and role "
        "offer that precision takes it, unless its solver node sets a precision of its own (the per-solver "
        "override): Automatic roles and nodes without a precision setting. Solvers that do not offer it keep their "
        "default (a validation info names them), e.g. double makes pcg and the time steppers double while "
        "sparse_direct / arpack_shift_invert / banded are double anyway; mixed reaches only pcg nodes and Automatic "
        "pcg of heat, electric_currents, electrostatics and magnetostatics, and line_block nodes. A node with "
        "pcg preconditioner multigrid and no precision of its own fails with mixed (set the node precision). "
        "check.json / summary.json solvers record the precision each role resolved to."),
    'SolverDefaultsSpec.subpixel_smoothing': (
        'false (default) | true: anisotropic permittivity averaging at material interfaces (Kottke / Farjadpour). '
        'Fill fraction and normal of every E_z cell / E_x and E_y face whose volume two materials share come once '
        'at build from 64 x 4 and 4 x 64 sample rasters per cell of the drawn material regions (each location takes '
        'the one whose fine axis is nearer its normal); the averaging (E_z and the field '
        'component along an interface: mean eps; across it: the harmonic mean, the diagonal of the inverse tensor) '
        'is reapplied to the current eps of two neighbouring cells of those materials after every change, so '
        'thermo-optic and other eps couplings keep working. Honoured by em_fdtd_ez / _hz, em_fdfd_ez / _hz, em_modes '
        '(E_x / E_y faces with the in-plane normal, E_z cells the area mean) and em_fdtd_3d outside an optimization '
        'study (incl. their mode ports); every other physics ignores it (validation info). em_fdtd_3d: the in-plane '
        'geometry per run of z cells sharing one cross-section (vertical walls: exact for z prisms; tapers and spheres '
        'are smoothed per layer, staircased along z) plus horizontal interfaces (prism tops / bottoms) sampled at 16 '
        'heights per half cell; the averaged face eps are re-quantised into a uint16 face index (2 bytes per cell). '
        'Only the background eps is averaged: sigma_e, mu_r and Drude / Lorentz poles keep their cell '
        'membership, and expression-defined eps jumps inside one material stay staircased. Restores second order '
        'on curved E_z interfaces; on slanted H_z interfaces it is better than staircasing, not second order.'),
    'SolverDefaultsSpec.conformal_boundaries': (
        'false (default) | true: curved walls at their drawn position instead of staircased to the cells, located '
        'once at build from the drawn geometry (bisection along the cell links and faces). em_fdtd_ez / _hz '
        'pec_domain: E_z puts E_z = 0 at the exact wall distance along each H link to a PEC cell; H_z (Dey-Mittra) '
        'updates H_z of each cut cell from the free lengths of its faces and its free area, and only faces wholly '
        'inside the metal are PEC. Each corrected cell is limited so its update stays within the stability limit of '
        'the unchanged time step (the smallest cut never sets dt; a very small cut is then partly staircased). '
        'em_fdtd_ez / _hz dispersive materials (Drude / Lorentz poles): each pole current is weighted by the share '
        'of its control area inside the drawn material (E_z: the cell, exact for E tangential to the interface; '
        'H_z: the dual cell of each E_x / E_y face) instead of 0 / 1, the Courant check with those shares; the '
        'background eps keeps the cell mask (subpixel_smoothing averages it). '
        'em_fdfd_ez / _hz pec_domain: the same E_z wall distances (t >= 0.02) and H_z free face lengths, the H_z '
        'free area (>= 0.02 of the cell) entering the k0^2 mu term, so the operator stays reciprocal; no time-step '
        'limit. Mode ports, sources and the FDFD poles keep the cell mask. laminar_flow: interpolated (Bouzidi, '
        'linear) bounce-back on the links from fluid to solid cells, with the wall where the solid material is drawn '
        '(a material counts as solid when most of its cells have viscosity 0); outer faces keep their face rules. '
        'heat, electric_currents, electrostatics (cut cells): held regions (temperature / voltage on object '
        'perimeters, fixed_temperature_domain, terminal) put their value on the drawn outline: the free neighbour '
        'conducts c A / (t d) to it, t d its distance from the free centre along the link (t >= 0.02, Shortley-'
        'Weller), instead of the half cell 2 c A / w; walls of non-conducting materials (a material counts as closed '
        'when most of its cells conduct neither) get fractional face apertures and merged cut-cell volumes (>= 0.02), '
        're-located whenever an assembly\'s conducting cells change (k(T), sigma(T)); perimeter heat_flux / '
        'current_density / surface_charge feed their value times each cut cell\'s chord of the drawn outline into the '
        'object cell owning it, convection loses h L / (1 + h delta / k_n) there (delta the owner centre\'s distance to '
        'the chord). em_modes pec_domain (Dey-Mittra on the projected Yee cell): E_z = 0 at the wall distance along '
        'each centre link into the metal, H_z from the free lengths of its E_x / E_y edges over the free area of its '
        'dual cell (fractions >= 0.02). em_fdtd_3d pec_domain (3D models, Dey-Mittra): every H face between four '
        'cell centres the drawn surface cuts takes the free lengths of its four E edges over its free area, floored '
        'so its update row stays within its uncut row at the time step (heuristic 3D Courant floor; E faces between '
        'two metal cells stay pinned). 2D Cartesian models only otherwise; heat / potential also (r, z) and polar / '
        'spherical domains (the cut fractions along the logical (r, angle) links / faces / cells, outline chords in '
        'the local (dr, r d_angle) frame; links across an angle_wrap seam stay staircased); every other physics '
        'ignores it (validation info). Features thinner than 1/16 of a cell '
        'face, or that no cell centre sees, stay invisible.'),
    'ModelSpec.study': 'What to compute (StudySpec): study type, its settings, stored fields and an optional sweep.',
    'ModelSpec.derived_values': 'Scalars evaluated at the end of the run (DerivedValueSpec), reported in '
                                'summary.json derived (with derived_reference / derived_error when they have a '
                                'reference).',
    'ModelSpec.analytic_curves': 'Analytic curves (AnalyticCurveSpec) the GUI draws dashed over a simulated curve '
                                 '(a feature result, the sweep plot or a line profile); figures=True draws them '
                                 'into the feature result figures and adds figures/curve_<name>.png per line curve '
                                 'and figures/<key>_vs_<x_column or row>.png per table target. They do not change '
                                 'the run.',
    'ModelSpec.derived_curves': 'Curves (DerivedCurveSpec) computed after the run from 1D feature results, e.g. '
                                'several spectra on one plot, a sum or ratio of results or a result scaled to '
                                'physical units; shown in the GUI Derived curves tab, in summary.json derived_curves '
                                'and with figures=True in figures/derived_<plot>.png. They do not change the run.',
    'ParameterSpec.name': 'Identifier; not a reserved expression name (function, constant, x, y, z, t, value) '
                          'nor a physics field name.',
    'ParameterSpec.expression': 'Number or expression of other parameters and constants, e.g. "550*nm"; it must '
                                'evaluate to a finite number.',
    'ParameterSpec.description': 'Free text, e.g. the meaning and unit.',
    'DomainSpec.size_x': 'Width of the physical domain in m (> 0), x from 0 to size_x. On a uniform mesh the meshed '
                         'window (Domain.physical_extent) is ceil(size_x / dx) cells wide, so it can be one cell '
                         'larger than requested; EM PML cells are added outside. Polar / spherical coordinates: the '
                         'radial depth in m, r from radius_min to radius_min + size_x (exact: dr = size_x / '
                         'ceil(size_x / dx)).',
    'DomainSpec.size_y': 'Height of the physical domain in m (> 0), y from 0 to size_y; as for size_x, the meshed '
                         'window can be one cell taller than requested on a uniform mesh. Polar / spherical '
                         'coordinates: the angle span in rad (polar <= 2 pi; spherical angle_min + size_y <= pi). '
                         'null = a 1D model (symmetry 2.20), invariant in both other directions: coordinates '
                         'cartesian = a slab along x from 0 to size_x (totals per m^2), polar = a cylinder, r from '
                         'radius_min to radius_min + size_x (per m of length; geometry planar), spherical = a sphere '
                         '(whole shells; geometry axisymmetric); size_z, angle_min and angle_wrap stay unset. Only '
                         'types whose dimensions include 1 run (heat, electric_currents, electrostatics). The model '
                         'is drawn on the x axis of the plane (x = the slab coordinate or r, y = 0): objects select '
                         'the cells whose centres (x, 0) they hold (rectangles across y = 0, circles about the origin '
                         'for shells), points sit on y = 0, line curves run along y = 0; outer faces x_lo / x_hi only '
                         '(x_lo at r = 0: the centre, no area). Mesh: uniform at dx, or graded by a mesh axis x entry '
                         '(no axis y, object refinements or auto mode). Fields are the (nx, 1) cells along x; the '
                         'Automatic linear solver is banded on both backends.',
    'DomainSpec.coordinates': (
        "cartesian (default) | polar | spherical: the in-plane grid of a 2D model. polar = (r, phi) cells, the "
        'drawing at x = r cos phi, y = r sin phi about the origin (planar per unit z, every physics with geometry '
        "'planar'); spherical = (r, theta) cells of the meridional half plane, x = r sin theta the distance to the "
        "rotation axis y, y = r cos theta (every physics with geometry 'axisymmetric', whole shells). Circles about "
        'the origin, annuli, spheres, shells and cones (theta = const) are conformal; other shapes staircase on the '
        'cell centres. Outer faces: x_lo / x_hi = the inner / outer arc (r = radius_min, the origin when 0: no '
        'area), y_lo / y_hi = the rays at angle_min / angle_min + size_y (spherical theta = 0 / pi: the axis, no '
        'area); insulated by default (a full 2 pi disk keeps a cut at y_lo / y_hi unless angle_wrap joins them). '
        'Only physics '
        'types whose coordinate_grids include polar may run (heat, electric_currents, electrostatics; 3D: heat); '
        'k_xx / k_yy (sigma_dc_*, eps_dc_*) are the radial / angular components there and *_xy is refused; object '
        'perimeter flux features (heat_flux, convection, current_density, surface_charge on objects) weight each '
        'exposed cell face by its metric area (2D; refused in 3D sectors and 1D models); '
        'solver_defaults.conformal_boundaries applies on 2D polar / spherical grids too (heat, potential). '
        "Mesh: uniform at dx radially and dx along the outer arc (angle cells = ceil(size_y r_max / dx)); "
        "nonuniform: mesh axis 'x' bands / edges / segments (in r - radius_min) grade the radius, axis 'y' edges / "
        "segments (rad) set the angle cells; object refinements, mode 'auto' and EM PML are refused. Fields come "
        'as a containing-cell image over the sector\'s box (NaN outside); the _x / _y field components are the '
        'radial / angular ones; derived values integrate over the exact sector areas / shell volumes. With size_y '
        'null the same three values pick a 1D slab / cylinder / sphere (see size_y). polar with size_z set = a 3D '
        'cylindrical (r, phi, z) sector (symmetry 2.21; heat only): the same (r, phi) plane at every height, z '
        'along the axis (uniform at dx, or graded by a mesh axis z entry), z_lo / z_hi the end faces; geometry z '
        'params apply as in Cartesian 3D (prisms along z, spheres, boxes; masks on the drawing-plane centres per '
        'cross-section); k_zz is axial. Fields are z cuts only (plane x / y, stored x / y cuts and 3D segment line '
        'curves are refused), each shown as the polar window; derived values integrate over the volumes r_c dr '
        'dphi dz; Session.volume gives (r, phi, z) centres.'),
    'DomainSpec.radius_min': 'Polar / spherical coordinates: r of the x_lo face in m (>= 0; 0 = the origin is in the '
                             'domain, a face without area). Ignored (warning) on a Cartesian grid.',
    'DomainSpec.angle_min': 'Polar coordinates: phi of the y_lo face in rad (from +x toward +y); spherical: theta of '
                            'the y_lo face from the +y rotation axis (>= 0). Ignored (warning) on a Cartesian grid.',
    'DomainSpec.angle_wrap': (
        "Polar coordinates only: none (default) | periodic | antiperiodic. periodic joins the y_hi ray to the y_lo ray "
        '(u continues across): a full 2 pi annulus or disk without a cut, or one C_N sector of size_y = 2 pi / N of a '
        'structure with N-fold rotational symmetry whose sources and conditions share it (warning when 2 pi / size_y '
        'is not whole). antiperiodic = u changes sign across the wrap, the angular Bloch phase pi of a C_N sector: '
        'needs an even N = 2 pi / size_y (error otherwise): the response to sources that alternate in sign from '
        'sector to sector; other Bloch phases are complex and not available. y_lo / y_hi are then no boundary: features may not select them (error). The wrap is '
        'exact in the finite-volume operator (the face between the last and first angle cells), anchoring counts '
        'loops through it (an antiperiodic loop anchors on its own).'),
    'DomainSpec.size_z': 'null (default) = a 2D model; a length in m (> 0) makes the model 3D, z from 0 to size_z: '
                         'the xy layout is extruded over each object\'s z0..z1 interval (layer stack), the outer '
                         'faces gain z_lo / z_hi and only physics types whose dimensions include 3 are allowed. z '
                         'cells follow the mesh: uniform at dx, or graded by the refinements\' z extents (see mesh '
                         'mode).',
    'GeometrySpec.name': 'Identifier, unique among geometry objects and selections (one namespace); not '
                         f"{', '.join(RESERVED_ENTITY_NAMES)}.",
    'GeometrySpec.kind': f"Geometry kind: {', '.join(GEOMETRY_KINDS)} (see geometry).",
    'GeometrySpec.params': 'Values of the kind params by name (numbers or parameter expressions, lengths in m); '
                           'missing keys take the defaults.',
    'SelectionSpec.name': 'Identifier in the namespace shared with geometry objects.',
    'SelectionSpec.kind': f"Entity kind of the selection: {' | '.join(SELECTION_KINDS)}.",
    'SelectionSpec.entities': 'Entity names: geometry objects, other selections of the same kind, outer faces '
                              '(boundary) or all (domain). interface: domain objects (their outlines; composites: '
                              'the outline of the result).',
    'SelectionSpec.side': "Kind 'interface' only: 'inside' | 'outside' = the side of the outline its fields are read "
                          "from; '' for the other kinds. An interface selection (2D models, polar / spherical domains "
                          'too; not 3D) is the outline of each listed object in the drawing plane, sampled every half '
                          'finest cell (outline parts on the window edge skipped); each sample reads its side 1.5 and '
                          '2.5 local cells deep along the outward normal, each a weighted least-squares plane (in the '
                          'drawing plane, through the cell centres) through the side cells of a 4 x 4 block whose '
                          '8 neighbours lie on that side too (so no cell-centre average mixes the materials), '
                          'extrapolated linearly onto the outline (exact for fields linear on the side; error ~1.9 '
                          '(k_n h)^2 for a wave of normal wavenumber k_n; a side thinner than ~4 cells gives no '
                          'samples; on a polar window the block does not reach across an angle_wrap seam). Derived '
                          'values '
                          'reduce such samples (see DerivedValueSpec.selection) and line curves read them (see '
                          'AnalyticCurveSpec.interface): one-sided normal components (D_n continuity: E_n jumps by '
                          'eps_2 / eps_1), tangential ones continuous.',
    'MaterialSpec.name': 'Identifier, unique among materials.',
    'MaterialSpec.library': 'Library material name (see materials), or null / "" for none.',
    'MaterialSpec.properties': 'Material properties of the present physics types {name: number or expression}, '
                               'overriding the library values (see materials).',
    'MaterialSpec.dispersion': 'EM dispersion poles [{"model": "drude" | "lorentz" | "nk_fit", ...}], plus at most '
                               'one {"model": "crystal", ...} entry (piezoelectric crystal data, not a pole); null, [] '
                               'and a list: see materials.',
    'MaterialSpec.selection': 'Domain entities the material covers (all = a background material, PML included: see '
                              'materials); an empty list assigns it nowhere (warning).',
    'PhysicsSpec.name': 'Instance name (identifier) used in field names <name>_<field>, stored fields '
                        '<name>.<component> and couplings; usually the default instance name of the type.',
    'PhysicsSpec.type': 'Registered physics type name (see the physics type sections).',
    'PhysicsSpec.settings': 'Values of the physics type settings by name; missing keys take the defaults.',
    'PhysicsSpec.features': 'Features (FeatureSpec) of this physics: sources, probes, monitors, boundary and '
                            'domain conditions.',
    'PhysicsSpec.every': 'Stride in runner ticks (int >= 1): the physics steps on every n-th tick over n * '
                         'dt_sync, dt_sync being the smallest dt_limit of all physics (EM: its Courant step). EM must '
                         'use 1; heat coupled to EM typically a few hundred.',
    'PhysicsSpec.solvers': 'Solver role (stepper, linear, shifted, eigen) -> SolverSpec name; a missing role = '
                           'Automatic. The roles of each physics type and their Automatic types: see solvers. A misfit '
                           '(role, operator, precision) is an error; a role the study does not use is a warning.',
    'PhysicsSpec.mesh': 'Name of the MeshSpec this physics runs on; "" = the first mesh (ModelSpec.meshes[0]). '
                        'Physics on different meshes exchange data only through couplings whose type transfers across '
                        'meshes (see mesh).',
    'FeatureSpec.name': 'Identifier, unique within its physics; its result key is <physics>.<feature>.',
    'FeatureSpec.type': 'Feature type of the physics type (see its Feature sections).',
    'FeatureSpec.selection': 'Entities of the selection kind of the feature type (domain, boundary, point or line '
                             'entities).',
    'FeatureSpec.params': 'Values of the feature type params by name; missing keys take the defaults.',
    'FeatureSpec.enabled': 'false leaves the feature out of validation and build; it stays in the model.',
    'CouplingSpec.name': 'Identifier, unique among couplings.',
    'CouplingSpec.type': 'Registered coupling type name (see the coupling type sections).',
    'CouplingSpec.source': 'Name of the physics instance whose field is read.',
    'CouplingSpec.target': 'Name of the physics instance whose source slot is written; two enabled couplings '
                           'writing the same target slot are an error.',
    'CouplingSpec.every': 'Transfer stride in runner ticks (int >= 1); null = the larger every of the source and '
                          'target physics.',
    'CouplingSpec.params': 'Values of the coupling type params by name; missing keys take the defaults.',
    'CouplingSpec.enabled': 'false leaves the coupling out of validation and build.',
    'MeshSpec.name': 'Identifier, unique among meshes; PhysicsSpec.mesh names it; issues of the mesh sit at node '
                     '["mesh", name].',
    'MeshSpec.refinements': 'Local refinements (RefinementSpec), used in nonuniform and auto modes (triangles / gmsh '
                            'tetrahedra: maximum element sizes in / along / at the object, gmsh tetrahedra also '
                            "along its edges or at its surface only (where); builtin tetrahedra: tensor bands, "
                            'per-axis sizes included).',
    'RefinementSpec.object': 'Geometry object whose region is refined (domain object, line or point).',
    'RefinementSpec.dx': 'Target cell size in m in that region; it applies only when finer than the base size.',
    'RefinementSpec.margin': 'Distance in m around the object that is refined too (3D: along z as well); null = 3 '
                             '* the axis cell size.',
    'RefinementSpec.dx_x': 'Cell size in m along x only (anisotropic refinement); null = dx. A value >= the base '
                           'size leaves x unrefined, e.g. for a waveguide crossing the domain along x.',
    'RefinementSpec.dx_y': 'Cell size in m along y only; null = dx (see dx_x).',
    'RefinementSpec.dx_z': 'Cell size in m along z only (3D models); null = dx (see dx_x). dx_x / dx_y / dx_z: '
                           'tensor meshes and builtin tetrahedra only.',
    'RefinementSpec.automatic': 'true = filled in from the auto-mode sizes (the GUI writes it when a mesh leaves '
                                'auto mode, with dx = the auto base size); auto mode ignores such refinements and '
                                'computes its own, so switching back to auto recalculates them. Default false.',
    'RefinementSpec.where': "Where dx applies: 'inside' (default) = in the object; 'edges' = along the edges of its "
                            'primitives (box 12 edges, z cylinder 2 rims, composite: every input, hole rims '
                            "included; the physics' edge sizes path); 'surface' = at its surface only, dx + (max_ratio "
                            '- 1) x |distance to the surface| on both sides (composites: the composite surface, a '
                            "hole's inside stays coarse). 'edges' / 'surface': tetrahedron meshes with mesher gmsh "
                            'only (validation error elsewhere); spheres have no edges, and objects gmsh gets no solid '
                            'of (ellipses, polygons, x / y circles, tapers) get no size field (a warning).',
    'MeshSpec.axes': 'Per-axis options (MeshAxisSpec, at most one per axis), used in nonuniform and auto modes '
                     '(tensor meshes and builtin tetrahedra).',
    'MeshSpec.size_points': 'Element-size sources in m: [x, y, h] on triangles, [x, y, z, h] on tetrahedra (h > 0, '
                            'mesher gmsh only: refused with builtin tetrahedra), each growing by max_ratio with the '
                            'distance like an object size; adaptive mesh refinement writes them into adapted triangle meshes (convergence report '
                            'adapted_meshes). [] = none.',
    'MeshSpec.face_sizes': 'Tetrahedra only: element sizes near plane faces (MeshFaceSizeSpec), every mode; gmsh: a '
                           'size field over the face rectangle; builtin: a band [c - distance, c + distance] across '
                           'the whole face plane (the plane an element face). [] = none.',
    'MeshFaceSizeSpec.entity': "Outer face ('x_lo' ... 'z_hi') or a domain object (its bounding-box faces, clipped to "
                               'the window).',
    'MeshFaceSizeSpec.size': 'Element size in m at the face (> 0); applies only below dx, floored at h_min.',
    'MeshFaceSizeSpec.face': "Objects: one bounding-box face ('x_lo' ... 'z_hi'), '' = all six; outer faces: ''.",
    'MeshFaceSizeSpec.distance': 'Half-thickness in m (>= 0) of the band held at size; beyond it the size grows by '
                                 'max_ratio up to dx.',
    'MeshAxisSpec.axis': "'x', 'y' or 'z' (z only in 3D models).",
    'MeshAxisSpec.max_ratio': 'Maximum growth ratio of this axis (> 1; also its PML ramp); null = the mesh '
                              'max_ratio.',
    'MeshAxisSpec.bands': 'Coordinate bands (MeshBandSpec) refined along this axis only, without a geometry object.',
    'MeshAxisSpec.edges': 'Custom cell edges in m (numbers or parameter expressions), increasing within [0, size]; 0 '
                          'and the size are added when missing. They replace the graded cells of this axis '
                          '(refinements and bands are then ignored along it); give edges or segments, not both.',
    'MeshAxisSpec.segments': 'Custom cells as consecutive pieces (MeshSegmentSpec) from 0 to the axis size; they '
                             'replace the graded cells of this axis like edges.',
    'MeshBandSpec.lo': 'Band start in m along the axis.',
    'MeshBandSpec.hi': 'Band end in m (> lo).',
    'MeshBandSpec.dx': 'Cell size in m inside the band; applies only when finer than the base size (graded ramps '
                       'outside, as for refinements).',
    'MeshSegmentSpec.cells': 'Number of cells of the piece (whole number >= 1).',
    'MeshSegmentSpec.end': 'End of the piece in m (it starts at the previous end, the first at 0); null = the axis '
                           'end (last piece only).',
    'MeshSegmentSpec.profile': "'uniform' (equal cells), 'geometric' (constant growth from the start, last / first "
                               "cell = ratio; < 1 shrinks toward the end) or 'tanh' (two-sided clustering toward "
                               'both ends of a gap, centre / end cell = ratio >= 1).',
    'MeshSegmentSpec.ratio': 'Size ratio of the profile (see profile); 1 = uniform.',
    'StudySpec.type': 'Registered study type name (see the study type sections).',
    'StudySpec.settings': 'Values of the study type params by name; missing keys take the defaults.',
    'StudySpec.solved_physics': "Physics names the study solves itself (COMSOL's physics in study step), e.g. "
                                "['membrane']; the others run first as stationary background physics (also those of a "
                                'type the study could solve), so couplings from them into the listed physics are '
                                'allowed. [] = every physics whose solver has the study\'s solve. Only frequency_domain, '
                                'mode_analysis, eigenfrequency, frequency_response and optomechanics (validation error '
                                'elsewhere).',
    'StudySpec.stored_fields': "Output fields stored in snapshots, as 'physics.component' (e.g. 'em.ez'); [] = "
                               'the primary field of each physics (FDFD: <field>_re and <field>_im, so |field|^2 '
                               'can be rebuilt).',
    'StudySpec.sweep': 'Optional parameter sweep (SweepSpec): one build and run per value; null = no sweep.',
    'StudySpec.stored_cuts': "3D models only: extra cut planes [axis 'x' | 'y' | 'z', position in m (number or "
                             'parameter expression)] each snapshot stores besides the mid-height z cut (the window '
                             'cell containing the position; same stored fields, resolution rules and 1 GB limit), '
                             'e.g. [["y", 1e-6]] for an xz side cut; saved run directories keep them. Read with '
                             "Results.snapshot_field(index, name, plane=(axis, position) or the cut's index); "
                             'summary.json stored_cuts lists the planes (cell-centre positions). [] = none.',
    'StudySpec.adaptation': 'Optional mesh refinement levels (AdaptationSpec, S7.7): the study is rebuilt and rerun on '
                            'refined meshes until a derived value converges; null = the model meshes as they are. '
                            'Only study types that support it (validation), no sweep or parity sectors.',
    'AdaptationSpec.target': 'Derived value (DerivedValueSpec.name) whose relative change between levels stops the '
                             'refinement (required).',
    'AdaptationSpec.method': "'adaptive' (error indicator marks where to refine) or 'uniform' (every size of the "
                             'meshes x growth^(-1/2) per level: a mesh-convergence sequence).',
    'AdaptationSpec.tolerance': 'Converged when |v_k - v_k-1| <= tolerance |v_k| (> 0).',
    'AdaptationSpec.max_levels': 'Largest number of refinement levels after the model meshes (integer >= 1); '
                                 'reaching it unconverged is a warning.',
    'AdaptationSpec.growth': 'Cell / element count growth per level (> 1).',
    'AdaptationSpec.meshes': 'Mesh names refined ([] = every mesh the physics use).',
    'AdaptationSpec.max_cells': 'Stop before a level whose cell / element count would exceed this (null = no cap; the '
                                'memory budget always applies).',
    'AdaptationSpec.anisotropic': 'Tensor meshes: refine x and y bands separately by the indicator part that '
                                  'dominates (true) or both axes alike.',
    'AdaptationSpec.keep_levels': 'true: keep every level\'s snapshots, not only the last level\'s.',
    'StudySpec.estimation': 'Optional parameter estimation (EstimationSpec; COMSOL Parameter Estimation): the study is '
                            'rebuilt and rerun per evaluation with the estimated parameters set, until the weighted '
                            'misfit of measured derived values and curves is least; then one run at the best point '
                            'stays live (summary estimation). null = one run. Any study type; no sweep or adaptation '
                            '(validation); parity_sectors run every sector per evaluation (combined values fitted). A '
                            'geometry parameter remeshes per evaluation (element meshes are reused while unchanged), '
                            'which adds finite-difference noise: keep diff_step large or use nelder_mead.',
    'EstimationSpec.parameters': 'Estimated model parameters (EstimatedParameterSpec, >= 1); each evaluation sets '
                                 'them as numbers, parameters depending on them follow.',
    'EstimationSpec.targets': 'Measured derived values (EstimationTargetSpec); residual (model - value) / sigma.',
    'EstimationSpec.curves': 'Measured curves (EstimationCurveSpec); residual per point (model interpolated onto x - '
                             'y) / sigma.',
    'EstimationSpec.method': "'least_squares' (scipy least_squares TRF within the bounds, finite-difference Jacobian: "
                             "n + 1 runs per step; standard errors from J^T J) or 'nelder_mead' (bounded simplex, no "
                             'derivatives: for noisy misfits, e.g. a remeshed geometry parameter; no standard errors).',
    'EstimationSpec.max_evaluations': 'Largest number of study runs of the search (integer >= 1; the final run at the '
                                      'best point is extra); reaching it ends the search at the best point so far.',
    'EstimationSpec.diff_step': 'least_squares: finite-difference step relative to each parameter scale (> 0; '
                                'default 1e-2: far above solver / mesh noise).',
    'EstimationSpec.tolerance': 'Stop when the cost and the step change by less than this (relative; nelder_mead: '
                                'absolute in the scaled parameters and the cost) (> 0).',
    'EstimatedParameterSpec.name': 'Model parameter name (ParameterSpec.name).',
    'EstimatedParameterSpec.initial': "Start value (number or expression of the base parameters); null = the "
                                      "parameter's own value. Must lie strictly between lower and upper; |initial| "
                                      '(else upper - lower, else 1) is the parameter scale of the search.',
    'EstimatedParameterSpec.lower': 'Lower bound (number or expression); null = unbounded.',
    'EstimatedParameterSpec.upper': 'Upper bound (number or expression); null = unbounded.',
    'EstimationTargetSpec.derived': 'Derived value (DerivedValueSpec.name) compared with the measured value (parity '
                                    'sectors: the combined value).',
    'EstimationTargetSpec.value': 'Measured value (number or expression of the base parameters).',
    'EstimationTargetSpec.sigma': 'Measurement uncertainty (> 0, same unit); null = |value| (a relative misfit; 1 when '
                                  'value is 0). Standard errors are absolute only when every target and curve gives '
                                  'sigma, else scaled by the reduced chi-square.',
    'EstimationCurveSpec.source': "Model curve: 'physics.feature:sample' of a 1D feature result (its abscissa, as a "
                                  'derived curve input) or a derived curve name (DerivedCurveSpec.name).',
    'EstimationCurveSpec.x': 'Measured abscissa values (numbers in the source abscissa unit), inside the model '
                             "curve's range (the run fails otherwise).",
    'EstimationCurveSpec.y': 'Measured values, one per x.',
    'EstimationCurveSpec.sigma': 'Uncertainty: one number or expression (> 0), a list with one value per point, or '
                                 'null = the RMS of y.',
    'SweepSpec.parameter': 'Name of the parameter that takes each value.',
    'SweepSpec.values': "spacing 'list': numbers (or parameter expressions) the parameter takes, run in list order "
                        '(kept but unused for the other spacings).',
    'SweepSpec.spacing': "'list' (default: the values), 'linear' (count evenly spaced points from lower to upper) or "
                         "'log' (count geometrically spaced points; lower and upper non-zero, same sign).",
    'SweepSpec.lower': "spacing 'linear' / 'log': first value (number or parameter expression); may exceed upper "
                       '(a descending sweep).',
    'SweepSpec.upper': "spacing 'linear' / 'log': last value (number or parameter expression).",
    'SweepSpec.count': "spacing 'linear' / 'log': number of values including both end points (1 = lower only).",
    'SweepSpec.keep_fields': 'true: keep the snapshots of every sweep value (summary sweep snapshot_times / '
                             'snapshot_labels per value, results.npz sweep_snapshot/..., Results.sweep_field), not '
                             'only those of the last value; past 1 GB in total only each value\'s final snapshot is '
                             'kept (a warning). false (default): earlier values keep their derived values and results '
                             'only.',
    'SweepSpec.reuse_build': 'true: build once and run every value on that build: the swept parameter (and the '
                             'parameters depending on it) may reach only feature values that follow a reused build '
                             '(solid_mechanics on triangles / solid_mechanics_3d on tetrahedra: body_force, '
                             'boundary_load, segment_load, displacement and prescribed_displacement_domain values of '
                             'parameters only, no x / y / z), derived values, analytic / derived curves and study '
                             'settings; any other use (geometry, mesh, materials, physics / solver / coupling settings, '
                             'stored cuts) is a validation error, as are the time_dependent and optimization studies '
                             'and parity_sectors. Each value then re-sets those loads and reruns the study from the '
                             'previous state (nonlinear solids start Newton from it; summary sweep builds = 1). A '
                             'model with a physics that cannot rerun (any other type) is rebuilt per value (an info '
                             'line). false (default): one build per value.',
    'DerivedValueSpec.name': 'Identifier, unique among derived values; its key in summary.json derived.',
    'DerivedValueSpec.expression': 'source null: expression of parameters, x, y (3D models: z), t and output '
                                   'fields (<physics>_<field>, or <field> when one physics alone provides it); over '
                                   f"an interface selection also {', '.join(INTERFACE_VARIABLES)} (outward unit "
                                   'normal of the object outline, tangent = the normal turned +90 deg; on a polar / '
                                   'spherical domain their radial / angular components, as the fields _x / _y there; '
                                   'they shadow parameters), e.g. "em_ex_re*normal_x + em_ey_re*normal_y". Operation '
                                   "'expression' (source null, COMSOL's global evaluation): one number of parameters, "
                                   'scalar outputs, t and the derived values listed before this one, no fields, x, y, '
                                   'z or selection, e.g. c_eff = "sqrt(N_int / rho_int)" of two integrals listed '
                                   'above; evaluated in float64 in list order like every derived value. With a '
                                   'source: '
                                   'expression of the result sample names and parameters, evaluated per sample, e.g. '
                                   '"power" or "1 - fraction_positive" (see Derived values). Both may use the names of '
                                   'the derived values listed before this one (their values; NaN when they failed): '
                                   'resonator figures of merit, e.g. W_E = integral of eps0_const*em_eps_r*'
                                   'em_e_intensity/4 (time-averaged stored electric energy of a phasor field, J/m in '
                                   '2D), V = mode_volume of em_eps_r*em_e_intensity, n_eff = '
                                   'sqrt(eps_int / e2_int) of two integrals (operation expression), Purcell factor F '
                                   '= 3/(4*pi**2)*(lam/n)**3'
                                   '*Q/V in 3D, (Q/(pi**2*V))*(lam/n)**2 for an out-of-plane (E_z) line emitter in 2D '
                                   '(twice that in-plane, H_z family), with lam and Q from a resonance_table source '
                                   '(operation at, at = the row); first-order shift of a small inclusion of '
                                   'polarisability alpha [F m^2; 2D: F] at a point: d_omega/omega = -alpha*E2_at/(2*'
                                   'eps0_const*eps_int) with E2_at = |E|^2 at the point (operation at) and eps_int = '
                                   'integral of eps_r*|E|^2 (Clausius-Mossotti sphere: alpha = 4*pi*eps0_const*eps_b*'
                                   'a**3*(eps_s - eps_b)/(eps_s + 2*eps_b); 2D rod along z with E_z: alpha = '
                                   'eps0_const*(eps_s - eps_b)*area; in-plane E: 2*eps0_const*eps_b*area*(eps_s - '
                                   'eps_b)/(eps_s + eps_b)).',
    'DerivedValueSpec.selection': 'Domain entities to reduce over; [] = the whole physical window; or exactly one '
                                  "'interface' selection: its one-sided outline samples (see SelectionSpec.side; "
                                  f"operations {', '.join(INTERFACE_OPERATIONS)}: mean / integral weighted by outline "
                                  'length, integral per unit z length, axisymmetric times 2 pi r). Must be [] with a '
                                  "source and for operation 'expression'.",
    'DerivedValueSpec.point': "Field operations only: point object of 'at' (required: the value in the window cell "
                              "containing it) and of 'mode_volume' (the field-at-point definition: integral / the "
                              "value there; '' = integral / maximum over the selection).",
    'DerivedValueSpec.operation': f"{' | '.join(DERIVED_OPERATIONS)}. Over fields ("
                                  f"{', '.join(FIELD_DERIVED_OPERATIONS)}): mean is area-weighted, integral is the "
                                  'sum of value * cell area (per unit z length; with a physics set to geometry '
                                  "'axisymmetric' both use the ring volume 2 pi r * cell area), mode_volume = "
                                  'integral / (maximum over the selection, or the value at point) [m^3; 2D: m^2, the '
                                  'area of a per-length mode], at = the value in the cell containing point; integral '
                                  'and mode_volume count each mirror face the selection reaches twice (the unfolded '
                                  'structure), and max / min (also the mode_volume maximum) include the mirror image '
                                  'there (each field with its mirror parity, x / y / z reflected). expression (source '
                                  'null, selection [], no point): the expression itself, one number of parameters, '
                                  'scalar outputs, t and earlier derived values (see expression). Over a source (the '
                                  'finite samples inside window): mean, max, min, integral = trapezoid over the '
                                  'abscissa (from the window edges, interpolated), argmax = abscissa of the maximum '
                                  '(parabola through it and its neighbours), fwhm = full width where the largest '
                                  'peak falls to half its maximum on both sides (linear crossings; baseline 0: '
                                  'subtract one in the expression), polyfit / exp_fit / lorentz_fit / two_exp_fit / '
                                  'two_wave_fit / richardson / multi_lorentz_fit / pole_fit = the quantity '
                                  'fit of a least-squares fit (see fit), last = the last sample in result order (the '
                                  'final time of a timeseries), at = linear interpolation at the abscissa value at. '
                                  'far_field angles wrap around (at, window, a peak at 0 rad; argmax in [0, 2 pi) '
                                  'without a window).',
    'DerivedValueSpec.source': "null = fields over the selection (operation 'expression': no fields, the other values); "
                               "'physics.feature' = the 1D result of that enabled "
                               f"feature (kinds {', '.join(DERIVED_RESULT_KINDS)}), reduced over its samples; the "
                               'name of a derived curve = that curve (sample names: the curve name and x), e.g. a '
                               'reduction or fit of a gradient / cumtrapz of results. Evaluated with the other '
                               'derived values at the end of each run, so a sweep reports it per value.',
    'DerivedValueSpec.window': 'Source only: [lo, hi] (numbers or parameter expressions in the abscissa unit, e.g. '
                               '["1540*nm", "1560*nm"]) limiting the samples every operation but last / at reduces; '
                               '[] = every sample. No finite sample inside gives NaN plus a warning. far_field: '
                               'angles in rad, a window may cross 0 (e.g. [-0.3, 0.3]), at most 2 pi wide.',
    'DerivedValueSpec.wavelength': 'far_field sources only: wavelength [m] (number or parameter expression) of the '
                                   'pattern reduced, the recorded one nearest to it; null = the first. Derived curve '
                                   'inputs of a far_field result use the first.',
    'DerivedValueSpec.fit': "Fit operations only: the fitted quantity. polyfit: 'c0' .. 'c<degree>' (coefficient of "
                            'x**k, in the abscissa unit); exp_fit (y = amplitude exp(rate x), log-linear least '
                            'squares weighted by |y|, samples of one sign): '
                            f"{', '.join(FIT_QUANTITIES['exp_fit'])} (decay = -1 / rate, the e-folding length or "
                            'time); lorentz_fit (y = offset + amplitude / (1 + (2 (x - center) / fwhm)**2), least '
                            'squares, a peak or a dip; window it around one line): '
                            f"{', '.join(FIT_QUANTITIES['lorentz_fit'])} (q = |center| / fwhm, the quality factor "
                            'of a line versus frequency or wavelength); two_exp_fit (y = amplitude_start exp(-kappa '
                            '(x - x_first)) + amplitude_end exp(kappa (x - x_last)), kappa > 0, amplitudes at the '
                            'outermost samples; an evanescent field between two ends): '
                            f"{', '.join(FIT_QUANTITIES['two_exp_fit'])} (decay_length = 1 / kappa); two_wave_fit "
                            "(complex y = A_f exp(-j beta x) + A_b exp(+j beta x), beta > 0; the expression is one bare "
                            "'<name>_re' series, paired at run time with '<name>_im' of the same result and x samples; "
                            'A_f travels toward +x in the exp(+j omega t) convention, phases at x = 0): '
                            f"{', '.join(FIT_QUANTITIES['two_wave_fit'])} (reflection_abs = |A_b| / |A_f|); "
                            'richardson (y = extrapolated + coefficient x**order, x > 0: a step size or cell count; '
                            'samples monotone): '
                            f"{', '.join(FIT_QUANTITIES['richardson'])} (order > 0 converges as x -> 0, order < 0 as "
                            'x -> infinity; error_estimate = |y - extrapolated| of the sample nearest the limit). '
                            'Peak fits (degree = number of lines K, quantities generated per line k = 1 .. K numbered '
                            'by ascending position, then the shared ones; spacing = least-squares slope of the line '
                            'positions vs k, degree >= 2 only): multi_lorentz_fit (y = offset + sum_k amplitude_k / '
                            '(1 + (2 (x - center_k) / fwhm_k)**2), joint least squares seeded from the K most '
                            'prominent maxima, or minima for dips): '
                            f"{', '.join(_per_line_names('multi_lorentz_fit'))} (q_k = |center_k| / fwhm_k); pole_fit "
                            '(complex y = sum_k r_k / (x - nu_k) + d0 + d1 x, vector fitting by pole relocation; the '
                            "expression is one bare '<name>_re' series as for two_wave_fit; x is usually frequency, "
                            'a nearby line outside the window is absorbed by d0 + d1 x only roughly: window around the '
                            'K lines): '
                            f"{', '.join(_per_line_names('pole_fit'))} (frequency_k = Re nu_k, fwhm_k = 2 |Im nu_k|, "
                            'q_k = |Re nu_k| / (2 |Im nu_k|), residue_abs_k = |r_k|; either sign convention of Im nu). '
                            'A fitted line outside the samples is an error (lower the degree or move the window).',
    'DerivedValueSpec.degree': f"Operation 'polyfit': polynomial degree 0 .. {MAX_FIT_DEGREE} (default 1); "
                               f"multi_lorentz_fit / pole_fit: number of lines (peaks / poles) 1 .. {MAX_FIT_DEGREE}.",
    'DerivedValueSpec.at': "Operation 'at' over a source only (fields: point): abscissa value of the source result, "
                           'number or parameter '
                           'expression in the abscissa unit (wavelength in m for a flux_spectrum, t in s for a '
                           'timeseries, x in the x unit of an xy result, the 0-based row of a table); outside the '
                           'sampled range the value is NaN plus a warning.',
    'DerivedValueSpec.reference': 'Optional analytic value (number or expression of parameters only, in the unit '
                                  'of the derived value); summary.json derived_reference and derived_error = '
                                  '|value - reference| / |reference| (|value - reference| for a zero reference); '
                                  'the GUI shows both beside the value and dashes it over the sweep plot.',
    'DerivedValueSpec.tolerance': 'Optional number >= 0 or null: largest accepted derived_error (relative, absolute '
                                  'for a zero reference; a sweep: at every value). Used only by selftest --examples, '
                                  'which fails the example above it; null = reported, never failed.',
    'AnalyticCurveSpec.name': 'Identifier, unique among analytic curves.',
    'AnalyticCurveSpec.target': "Simulated curve to draw over: 'physics.feature' of an enabled feature with a "
                                f"{', '.join(CURVE_RESULT_KINDS)} result, 'sweep' (the Sweep tab: versus the swept "
                                "parameter) or 'line' (a profile of field along line).",
    'AnalyticCurveSpec.expression': 'Expression of parameters and the target variable: wavelength [m] '
                                    '(flux_spectrum, xy versus wavelength), t [s] (timeseries; iteration = the '
                                    '1-based sample index when every sample shares one t, e.g. stationary '
                                    "iterations), frequency [Hz] (timeseries with series 'fft'), x (xy, in its x "
                                    'unit; table: the x_column value in its unit, or the 0-based row), angle [rad '
                                    'from +x] and wavelength [m] of the shown frequency (far_field, normalised to its '
                                    'peak like the simulated pattern), the swept parameter (sweep: every parameter is '
                                    're-evaluated per point), x, y [m], z [m, the cut height or the 3D segment '
                                    'point; 3D models only] and s [m from the line start] (line). '
                                    'Variables shadow parameters of the same name.',
    'AnalyticCurveSpec.series': "Simulated quantity the curve belongs to (drawn in its colour): an xy series name, "
                                "a table column name, 'ratio' for the Spectra ratio-to-reference plot of a "
                                "flux_spectrum / wavelength xy target, 'fft' for the probe FFT plot of a timeseries "
                                '(|FFT| * dt as the Probes tab plots it, in the value unit times s), for sweep the '
                                "derived value name, 'physics.feature' of a timeseries (its final value) or a "
                                "'<column>_<k>' series of a per-value table (band k: the k-th smallest value, as "
                                "Results.sweep_xy); '' = drawn with every quantity of the target.",
    'AnalyticCurveSpec.line': "Target 'line' only: [x0, y0, x1, y1] in m (numbers or parameter expressions), two "
                              'distinct points; the Line profile tab plots field along it with the curve. 3D models '
                              'also take the 3D segment [x0, y0, z0, x1, y1, z1] (any direction, e.g. along z or '
                              'oblique): sampled from the final state of the whole volume (containing cells or '
                              'trilinear, see sampling; no mirror unfolding; live Results only), so plane stays '
                              'null; every snapshot also stores the profile of each 3D segment line curve, so '
                              'snapshot works with it (also from a saved run).',
    'AnalyticCurveSpec.sampling': "Target 'line' with a 3D segment line only: 'nearest' (default; each sample reads "
                                  "its containing cell and curve_error compares at those cell centres) or "
                                  "'trilinear' (each sample interpolated between the 8 surrounding cell centres, "
                                  'constant in the outer half cells; curve_error compares at the sample points, so a '
                                  'smooth field along an oblique segment shows no staircase). Other lines: '
                                  "'nearest' (validation error otherwise).",
    'AnalyticCurveSpec.field': "Target 'line' only: output field 'physics.component' (e.g. 'heat.T') sampled "
                               'along the line from the final state (or the snapshot), each sample reading the cell '
                               'containing it (on a cell edge: the upper cell, as a probe there).',
    'AnalyticCurveSpec.x_column': "'table' targets only: numeric column used as the abscissa x of the curve and the "
                                  "plot (rows sorted by it); '' = the 0-based row index. series names the plotted "
                                  'column.',
    'AnalyticCurveSpec.plane': "Target 'line' in 3D models only: height z [m] (number or parameter expression) of "
                               'the horizontal cut the line [x0, y0, x1, y1] lies in; null = the GUI cut-plane choice '
                               '(when a z cut), headless the mid-height cut. Not with snapshot (snapshots hold the '
                               'mid-height cut) or a 3D segment line (side planes: give a 3D segment).',
    'AnalyticCurveSpec.snapshot': "Target 'line' only: sample field from a stored snapshot instead of the final "
                                  'state: an index into Results.snapshots() (int, negative from the end) or a '
                                  "snapshot label, whole or its text before ':' (e.g. 'mode 2', 'resonance 1', "
                                  "'iteration 3', 'lambda = 1550 nm'), else that text without its mirror parity tag "
                                  "('mode 2' for 'mode 2 [x_lo symmetric]'; it must match one snapshot); null = the "
                                  'final state. field must be a stored field (a 3D segment line: any field, its '
                                  'profile is stored with each snapshot of the run).',
    'AnalyticCurveSpec.tolerance': 'Optional number >= 0 or null: largest accepted summary.json curve_error max '
                                   '(max |simulated - analytic| / ptp(analytic) at the simulated samples, so an absolute '
                                   'temperature is judged against its rise; target sweep: relative per swept value; a '
                                   'feature target in a sweep: at every value). Used only by selftest --examples, which fails the example above it '
                                   'or when the curve cannot be compared; null = reported, never failed.',
    'AnalyticCurveSpec.interface': "Target 'line' in 2D models only: name of an 'interface' selection; '' = the "
                                   'containing cells. Each line sample (two per finest cell) then reads the '
                                   "selection's side along the outward normal of the nearest outline sample of its "
                                   'objects and extrapolates onto the sample itself (see SelectionSpec.side), and '
                                   'the curve is compared at the sample positions; draw the line along the outline '
                                   '(e.g. a slot-waveguide sidewall). field is one component, or with projection the '
                                   'x component of an in-plane vector. Polar / spherical domains: live Results read '
                                   'the (r, angle) cells (field(native=True)), saved ones and snapshots the image '
                                   'over the sector box (first order at the outline).',
    'AnalyticCurveSpec.projection': f"With interface only: '' (field as it is) | {' | '.join(INTERFACE_PROJECTIONS)}: "
                                    'field names the x component of an in-plane vector (e.g. em.ex_re, its y component '
                                    'em.ey_re, found by turning its axis letter x into y: E_x -> E_y), both read '
                                    'one-sided and projected on the nearest outline normal (normal) or the tangent = '
                                    'normal turned +90 deg (tangential); on a polar / spherical domain the components '
                                    'are radial / angular and projected as such. A snapshot curve needs both stored.',
    'DerivedCurveSpec.name': 'Identifier, unique among derived curves; its key in summary.json derived_curves.',
    'DerivedCurveSpec.expression': 'Expression of the input aliases (arrays on the curve abscissa, or numbers for '
                                   'scalar samples), x (the abscissa, in its unit) and parameters, evaluated per '
                                   'abscissa point, e.g. "eta*S*P_top/(h_const*c_const/lambda0)". Aliases shadow '
                                   'parameters of the same name. Besides the expression functions it may call (here '
                                   'only): ' + '; '.join(CURVE_FUNCTION_DESCRIPTIONS.values()) + '.',
    'DerivedCurveSpec.inputs': "{alias: 'physics.feature:sample'}: an identifier (not a reserved name) per input, "
                               'the sample a sample name of that enabled 1D feature result as for derived values '
                               '(xy series, table column, timeseries values, flux power, or a numeric scalar such as '
                               'variance). The first array input sets the abscissa (sorted); later ones are '
                               'interpolated linearly onto it (NaN outside their range) and must share its abscissa '
                               'label and unit.',
    'DerivedCurveSpec.plot': "Plot the curve is drawn in, shared with the other curves naming it (an identifier; "
                             "'' = a plot of its own named after the curve); figures/derived_<plot>.png. Curves of "
                             'different unit go to separate panels of the plot.',
    'DerivedCurveSpec.label': "Legend text; '' (or a label already used in the plot) = the curve name.",
    'DerivedCurveSpec.unit': "Unit of the values, free text ('' = none); panels group by it.",
    'DerivedCurveSpec.log_x': 'Logarithmic abscissa; the first drawn curve of a plot sets it for the plot.',
    'DerivedCurveSpec.log_y': 'Logarithmic value axis; the first drawn curve of a plot sets it for the plot.',
    'DerivedCurveSpec.reference': 'Optional analytic curve: number or expression of x (the curve abscissa) and '
                                  "parameters, drawn as series '<series> reference' in the plot; summary.json "
                                  'derived_curves <name> curve_error {max, rms, n_points} = |curve - reference| / '
                                  'ptp(reference) on the curve abscissa (as AnalyticCurveSpec.tolerance) and '
                                  'y_reference.',
    'DerivedCurveSpec.tolerance': 'Optional number >= 0 or null: largest accepted curve_error max against the '
                                  'reference. Used only by selftest --examples, which fails the example above it or '
                                  'when the curve fails; null = reported, never failed.',
}

_EXPRESSION_SYNTAX = (
    'Arithmetic: + - * / // % ** and unary + -; ** is the power (^ is not allowed).',
    'Comparisons < <= > >= == != give 1 (true) or 0 (false), e.g. "(x > 1*um) * 2"; chained comparisons (a < b < '
    'c), and / or / not and if / else are not allowed: use where, min and max.',
    'Integer literals become floats (1/2 = 0.5). Only numbers, names and calls of the functions below with '
    'positional arguments are allowed (no attributes, indexing, strings or keywords). A function name without "(" '
    'is an ordinary name, which only a result sample name can supply (see Derived values).',
    f'At most {MAX_EXPRESSION_LENGTH} characters. Results must be real: a complex result (e.g. (-8)**(1/3)) is an '
    'error, and a value outside a function domain (log(-1)) gives nan, which fails where a finite number is needed.',
    'With numbers and parameters only an expression gives one float; with x, y, z or fields it is evaluated per '
    'cell.',
)
_NAME_AVAILABILITY = (
    'ParameterSpec.expression: numbers, constants and other parameters in any order (unknown names and cycles are '
    'errors).',
    'Domain size, geometry params, mesh values, refinement dx / margin, sweep values, dispersion pole values and '
    'every param without "may use x, y": parameters only.',
    'Physics settings, feature params and material properties marked "may use x, y": also x and y, the cell centres '
    'in m (EM PML cells take the nearest window centre), and in 3D models the height z [m] (z in a 2D model is a '
    'validation error). 3D materials using z are written one z cell plane at a time, feature values using z cover '
    'the bounding box of their selection.',
    'Params marked "may use physics fields" (material properties): also the state fields of the physics in the '
    'model as <physics>_<field>, or <field> when one physics alone provides it (e.g. T with one heat physics); such '
    'a property is rewritten right after each step of the physics it reads (the one with the largest every), so '
    'physics stepping on later ticks see it, but only when one of them solved or stepped since the last rewrite (a '
    'quasi-static physics that did not re-solve rewrites nothing), and once more after the final solves of a '
    'time-dependent run (a '
    'stationary study iterates it; a '
    'frequency_domain study allows only fields of its background physics). They may also use the time t [s] '
    '(with or without fields and x, y): a property using t is rewritten at the start of every runner step (before '
    'the steps of that tick, t = the step start time; t = 0 in every other study).',
    'Feature params marked "may use t": also the time t [s] with parameters only (no x, y, z: a value changing in '
    'time is uniform in space), re-evaluated at the start of every runner step of the physics (t = its step start '
    'time; an implicit step, e.g. transient heat, sees the start value, a first-order lag); t = 0 at the build and in '
    'every other study.',
    'Coupling params of kind expression (field_transfer expression): value (the source field), x, y (3D: z) and '
    'parameters; no other field names and no t.',
    'Derived values: parameters, x, y (3D: z), t (the final time), the output fields as <physics>_<field> or a unique '
    '<field> and the scalar outputs of a physics (its type\'s Scalar outputs, e.g. em_modes n_eff_1) as '
    '<physics>_<name> only (a number after the run; a name the solver did not report gives NaN plus a warning); with '
    'a source the sample names of that result (a derived curve source: the curve name and x) instead (see Derived '
    'values). Derived value at / reference / window / wavelength: parameters only.',
    'Analytic curves: parameters and the variables of their target (AnalyticCurveSpec.expression).',
    'Derived curves: parameters, their input aliases and x, plus the calls gradient / cumtrapz '
    '(DerivedCurveSpec.expression); reference: parameters and x.',
    'A parameter may not have the name of a physics state or output field or scalar output (validation error).',
)
_GEOMETRY_NOTES = (
    'Coordinates are physical, in m, with the origin at the lower-left corner of the physical domain: x from 0 to '
    'size_x (to the right), y from 0 to size_y (up); EM PML cells lie outside.',
    'Masks test cell centres: rectangles and polygons are half-open [x0, x1) x [y0, y1), circles and ellipses are '
    'closed; rounded rectangles are half-open along their straight sides, closed on their corner arcs; a point or line end exactly on a cell edge goes to the upper cell.',
    "Validation's material preview uses float64 cell centres over (0, size_x, 0, size_y) while the build tests "
    'float32 cell centres clamped into the window, so cells exactly on a boundary can differ: avoid material edges '
    'exactly on cell centres.',
    'Composite inputs (union, intersection, difference, array) are consumed: they can no longer be selected on their '
    'own and may feed only one composite.',
    'An array repeats its base object on an nx x ny (x nz) grid of copies moved by multiples of pitch_x / pitch_y / '
    'pitch_z, the base being the first copy: one GeometrySpec for thousands of holes or pillars. It acts as one '
    'domain object (a union of its copies) in selections, materials, features and refinements; meshes follow every '
    "copy's outline. Copies are internal shapes named '<array>[i,j]' (never selectable); overlapping copies get a "
    'validation warning.',
)
_MATERIAL_NOTES = (
    'MaterialSpec.library takes a library material name below; properties override its values (a null property '
    'keeps the library value); dispersion null inherits the library poles, [] removes them, a list replaces them.',
    'Materials apply in list order and later materials win overlapping cells (whole cells). A material whose '
    'selection contains all is a background material: it fills only the cells no other material covers, wherever it '
    'sits in the list (of several background materials the later wins). Cells under no material keep the type '
    'defaults: vacuum for EM, k = 0 (no conduction) for heat, sigma_dc = 0 (insulator) for electric_currents, '
    'eps_dc = 1 (vacuum) for electrostatics.',
    'Only the material properties of the physics types present in a model are written, and only those the merged '
    'material sets; each physics type section lists its material properties with units and help. A property of kind '
    'table (e.g. magnetostatics bh_curve) is one constant table per material, written to no array: its physics reads '
    'it per material region.',
    'eps_r is isotropic; eps_xx, eps_yy, eps_zz are components written after it (a component wins): the E_z types '
    '(em_fdtd_ez, em_fdfd_ez) see eps_zz (their eps_r array), the H_z types (em_fdtd_hz, em_fdfd_hz) see eps_xx and '
    'eps_yy, em_modes sees all three; each type lists its own vocabulary.',
    'Library optical values (eps_r, sigma_e) are for 1.55 um; an entry without dispersion poles has the same index '
    'at every wavelength. Thermal data use the model property names (cp = specific heat). Electrical data (sigma_dc, '
    'eps_dc) are static values for electric_currents / electrostatics, distinct from the optical sigma_e / eps_r.',
    'Dispersion poles are dicts {"model": "drude" | "lorentz", ...} with the params below (EM only); for a Drude '
    'material that sets no permittivity property for an array (eps_r or that component) the first Drude eps_inf is '
    'the background permittivity of that array.',
    'An {"model": "nk_fit", ...} entry (one per material; library materials with tabulated n, k only: their '
    'description names the data set) is replaced by eps_inf (>= 1) + an optional Drude pole + Lorentz poles fitted '
    'to the table over [wavelength_start, wavelength_stop] (bounded least squares, all pole values >= 0: passive); '
    'the library optical values are then dropped, eps_inf is the Drude eps_inf or, without a Drude pole, eps_r. '
    'Validation reports the fit (max |eps error| relative to |eps|, loss error relative to |Im eps| + 0.03 |eps|) as '
    'info, as a warning above 2 % / 10 % or when the study band leaves the fit range. FDTD steps the fitted poles; '
    'the FDFD, em_modes and slab-mode ports use their exact chi(omega).',
    'A {"model": "crystal", "crystal": <name>, "euler_deg": [phi, theta, psi], "plane": "sagittal"} entry (one per '
    f"material; crystals {', '.join(sorted(CRYSTALS))}) is no pole: it is removed from the list and sets the rotated "
    'crystal data in model axes, c11, c12, c22, c66 (c16 / c26 when non-zero), rho, piezo_e11 .. piezo_e26 and the clamped eps_dc_xx / _yy / '
    '_xy; with optical data (not every crystal) also photoelastic_p11 / _p12 (+ the elastic_waves tensor entries '
    'photoelastic_p16 .. _p66 when the cut is not isotropic in them; row 6 = xy, the eps_xy of em_fdfd_hz) and the '
    'clamped Pockels tensor pockels_r11 .. _r62 when non-zero (precedence: library < crystal < explicit properties, '
    'also beside nk_fit). The optical '
    'index itself is not set (eps_r / eps_xx / eps_yy stay the material values). euler_deg are plain numbers in '
    "degrees, SAW convention: passive Rz(psi) Rx(theta) Rz(phi), rows X1' = propagation, X3' = surface normal (YZ = "
    "0, 90, 90); sagittal = model (x, y, z) = (X1', X3', -X2'). Only the anti-plane couplings "
    'are dropped (validation reports the sagittal values as info and warns when they are not negligible), and the '
    'off-diagonal photoelastic / Pockels index change (the acousto-optic and Pockels couplings warn). A 3D model needs '
    "\"plane\": \"full\" (2D models 'sagittal', the default; validation errors otherwise): model (x, y, z) = (X1', X2', "
    "X3'), the surface normal along z, and the entry sets all 21 c11 .. c66 (Voigt 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy), "
    'the 18 piezo_e11 .. piezo_e36, eps_dc_xx / _yy / _zz / _yz / _xz / _xy and rho (no optical entries).',
)
_RESULTS_NOTES = (
    ('Command line', (
        'Run the commands from the folder that contains the open_sol package (python -m open_sol ...).',
        'describe [topic] [--out FILE | --update-reference] [--json]: this reference, or one topic, on stdout or '
        f'into FILE; --update-reference rewrites {REFERENCE_FILE} and {TEMPLATES_FILE} next to the package (full '
        'text only: no topic, no --json).',
        'template KEY [FILE] [--force]: write the template KEY (see Templates) as a model JSON file, default '
        './KEY.json; an existing file is overwritten only with --force.',
        'check MODEL.json [--out DIR] [--set NAME=EXPR ...] [--backend auto|gpu|cpu] [--max-wall-time S] [--json]: '
        'CPU-only validation, mesh, memory and run-time estimates, time step and step count -> check.json; '
        '--max-wall-time S warns when even the shortest estimated run time exceeds S seconds.',
        'profile [--backend gpu|cpu] [--quick]: measure this machine for the run-time estimates (~30 s, --quick ~5 s '
        'with coarser curves) -> time_profiles/time_profile_<backend>.json next to the package (OPEN_SOL_PROFILE_DIR '
        'overrides the folder); default every available backend (cpu, and gpu when a CUDA device answers); prints '
        'each task\'s wall time and the stored profile\'s.',
        'run MODEL.json [--out DIR] [--figures] [--max-wall-time S] [--no-arrays] [--set NAME=EXPR ...] [--backend '
        'auto|gpu|cpu] [--json]: build and run on the model\'s backend (GPU or CPU) -> summary.json, log.txt, '
        'results.npz (not with --no-arrays), model.json (the model as given) and '
        'figures/*.png (with --figures: <key>.png per saved field and feature result, derived_<plot>.png per '
        'derived curve plot, <result>_vs_<column or row>.png per table x_column of its curves, curve_<name>.png per '
        'line curve, scene_3d.png for a 3D model: perspective picture of the objects with the first saved field on '
        'the x, y and z cut planes through the window middle); --max-wall-time stops cleanly with status timeout and '
        'partial results.',
        'show PATH [--out DIR] [--figures] [--series NAME] [--set NAME=EXPR ...] [--backend auto|gpu|cpu] [--json]: '
        'outline and check counts of a model file, or the summary of a results directory (--figures re-renders its '
        'figures into <out>/figures/*.png, by default the run directory\'s figures/, a sweep also its sweep_*.png '
        'overlays; --series NAME draws only that series of xy results and sweep overlays; a 3D run directory draws '
        'scene_3d.png with its saved cut planes: the mid-height z cut and the stored cuts of the last snapshot) -> '
        'show.json. A 3D model file with --figures gets <out>/figures/scene_3d.png, the objects only (CPU, no run; '
        'api.scene_figure).',
        '--set NAME=EXPR (repeatable) replaces the expression of the model parameter NAME for this command only (the '
        'file is not changed; run records it in model.json), e.g. --set P_in=2e3 --set "w=450*nm"; an unknown NAME '
        'is a model error at the parameters node. --backend (check, run, show) overrides the model\'s '
        'solver_defaults.backend for this command only (run records it in model.json).',
        'mesh-study [PART ...] [--out DIR] [--case CASE ...] [--fine] [--force | --retry-failed] [--max-cells N] '
        '[--max-wall-time S]: the mesh study (parts per_physics, audit, prototypes (CPU), accuracy, pcg, line_block '
        '(GPU), analyse; default all, in this order) -> DIR (default ./mesh_study_out) records/, runs/, tables/, '
        'figures/ and mesh_study_report.json; recorded variants are skipped unless --force (all) or --retry-failed '
        '(those that did not complete); --max-cells skips larger variants (default 2e7); exit 1 when a part '
        'raised.',
        'selftest [NAME ...] [--heavy | --all] [--out DIR] [--reset] [--json] [--list]: the built-in checks, then the '
        'registered analytic cases (the named ones, heavy or not; else the non-heavy ones, with --heavy only the '
        'heavy ones, with --all every one) -> selftest_report_light.json, selftest_report_heavy.json (--all: both) '
        'or selftest_report_named.json (names given), plus one <case>/<run>/ run directory per run; a run '
        'rewrites only its own report files. A case is heavy when it took >= 5 s (--list marks them). --all (no '
        'NAMEs) runs everything: every case with its gpu runs and variants (with a usable GPU), then the --examples '
        'run once on the cpu and once on the gpu backend -> also selftest_report_analytic_cpu.json / _gpu.json and '
        'examples_cpu/ / examples_gpu/ run directories (an example that validates only on its own backend is not run '
        'on the other; the gpu examples are skipped without a usable GPU); exit 1 unless the cases and both example '
        'runs passed.',
        'Reuse: <out>/selftest_cache.json keeps per case / example its last passing record (example: also '
        'no_tolerance), the hashes of the package code units that run depended on (each top-level function and '
        'class with methods, and each file\'s module-level code incl. data classes, compared without comments and '
        'docstrings: the units that ran, those they name, and the module-level code of every file touched) and the '
        'environment (Python, numpy, scipy, cupy, backends, device); an unnamed case or example whose units, model '
        'file and environment are unchanged is not rerun: '
        'its stored record enters the report with reused true, reused_from [time], and is not printed. --reset '
        'reruns every selected one; named ones always rerun. Run under a debugger or coverage tracer nothing is '
        'traced and nothing new is cached. The cache also keeps the last run time of every case / example.',
        'Output: per phase (the cases, each examples run) one line with the number to run and the number cached; per '
        'run test a start line "i/n  ~T left  name  (last T)" (i over every phase of the call, T left = the summed '
        'last run times of it and every later test, "+ k untimed" for those never timed) and a result line (PASS / '
        'FAIL / ERROR / NOTOL, error and tolerance or worst error / tolerance, wall seconds), plus the message of a '
        'test that did not pass.',
        'selftest --examples [NAME ...] [--out DIR] [--reset] [--json]: instead of the cases, run every examples/**/*.json '
        'of a repository checkout that has a derived value with a reference or an analytic curve on its own backend '
        '(NAMEs: only these, relative path with or without .json, or stem), two lines per example (Output) -> '
        'selftest_report_analytic.json plus one examples/<path>/ run directory each (no arrays, no figures); exit 1 '
        'when an example fails a tolerance or does not complete.',
        'check, run, show and selftest write their report files in every case, also when a build or run fails; '
        '--json also prints the report on stdout.',
        "Default --out: check and run '<model dir>/<model stem>_out/'; show of a model file the same, of a results "
        "directory that directory; selftest './selftest_out'. Directories are created as needed; files are "
        'overwritten.',
    )),
    ('Exit codes', (
        '0: success (check without errors, run completed, every selftest passed, describe / template / show done).',
        '1: model errors (check), a run status other than completed, a failed or erroring selftest, an incomplete '
        'documentation table (describe).',
        '2: usage errors: bad arguments, a missing model file, an unknown describe topic, template key or selftest '
        'name, an existing template FILE without --force.',
    )),
    ('JSON files', (
        'UTF-8 with ASCII escapes, indent 2. Non-finite floats are the strings "nan", "inf" and "-inf"; tuples '
        'become lists; node paths are lists of strings.',
        'Issue record: {level (error | warning | info), path [...], node (the path label, e.g. "feature '
        'em/source"), message}.',
    )),
    ('check.json', (
        '{open_sol_version, status (ok | warnings | errors), model {title, file (null for a spec not read from a '
        'file)}, n_errors, n_warnings, n_infos, issues [issue records], mesh, mesh_error, meshes, transfers, study, '
        'memory_estimate, time_estimate, solvers}.',
        'time_estimate: the run-time estimate on this machine\'s stored profile (CLI profile; seconds_low / '
        'seconds_guess / seconds_high, its items), also given as an info issue at ("study",); null without a profile '
        'or an estimate (the info says why). Its build: the same keys for building every run (build stages, upload, '
        'triangle meshing: a mesh that does not change between sweep values is made once), null when it fails; the '
        'build log line "Build estimate" compares it with the measured stage times.',
        'meshes [{name, kind, mode, shape ([nx, ny] or [nx, ny, nz], null for triangles or without an estimate), '
        'elements (triangles: estimated count, else null), cells, dx_min, dx_max, max_ratio (resolved), physics (names '
        'on it), used, gpu_bytes, host_bytes (its share of memory_estimate, null without one), error (its estimate '
        'failure or null)}] per MeshSpec; transfers [{coupling, source_mesh, target_mesh, kind (interpolate | '
        'conservative), nnz, bytes}] of the couplings across meshes.',
        'solvers {physics: {role: {type, solver (the solver node, null = Automatic), backend, precision, settings}} '
        'of the roles the study uses, or null when they do not resolve (see the issues)}, or null when the model does not '
        'load.',
        'mesh = the first used mesh: {nx, ny, nz (1 in 2D), name, kind, n_cells, dx_min, dx_max, dt, npml_sides, '
        'uniform, note, wavelength_min, wavelength_source, auto_rule, custom_axes, line_blocks, line_block_hint} '
        '(triangles: nx / ny null plus n_elements, n_nodes, n_edges, pml_widths; counts include the PML cells; auto '
        'mode: auto_rule wavelength / physics / uniform_flow, the resolved wavelength_min in m and its source, "mesh" '
        'for the setting or the physics name, else null and ""; '
        'custom_axes = the axes whose cells come from MeshSpec.axes edges / segments; line_blocks {block axis: '
        '[n_blocks, block_size]} of the window cells for the line_block solver, a block along x = one grid row, and '
        'line_block_hint its cost n_blocks * block_size^3 per block axis with the cheaper one; both empty in 3D), or '
        'null with mesh_error giving the reason.',
        'study {type, dt_estimate (s or null: the smallest of the mesh Courant dt and the time_step of every '
        'time-stepping physics), n_steps (round(duration / dt_estimate) when the study has a duration, else null), '
        'n_sweep_values (1 without a sweep), total_steps (n_steps * n_sweep_values, or null), n_wavelengths '
        '(frequency_domain: the number of wavelengths solved per sweep value, else null)}.',
        'memory_estimate {gpu_steady (bytes of GPU pool memory the build keeps), gpu_build_peak (bytes at the build '
        'high-water mark), gpu_run (bytes the study adds while it runs: run_items kept once allocated, e.g. harmonic '
        'FDTD accumulators, growing point traces, plus largest_phase: the transient_items of one phase, e.g. one '
        'solve or one adjoint, freed when it ends; phases never overlap), gpu_peak (max of gpu_build_peak and '
        'gpu_steady + gpu_run), gpu_external (cuDSS direct-solver factors, outside the pool: a fill-in rule), '
        'external_extra (the largest factorisation peak on top), device_peak (gpu_peak + gpu_external + '
        'external_extra: checked against the GPU budget), host (bytes of CPU factorisations and analyses: a fill-in '
        'heuristic), items / host_items / run_items / external_items [{label, bytes}] and transient_items [{phase, '
        'label, bytes}] largest first, largest_phase {phase, bytes} or null, budget (the '
        'check memory_budget argument in bytes, or null), host_budget (the check host_budget argument: available host '
        'RAM in bytes, or null)}, or null when the mesh or the estimate fails. CPU estimates counted per physics and '
        'feature type (approximate; summary.json compares gpu_steady and gpu_build_peak with the measured build); '
        'validation errors when device_peak exceeds a known GPU budget or host exceeds the available host RAM (a run: '
        'both always; CLI check: the host RAM), warns above 80 % of either; without a host budget it warns above 8 GiB '
        'of host memory.',
    )),
    ('summary.json', (
        'open_sol_version, status (completed | partial | timeout | cancelled | failed), message, model {title, file, '
        'study_type}, wall_time (s).',
        'build {nx, ny, nz (1 in 2D), n_cells, dt, dx_min, dx_max, npml_sides, build_seconds, gpu_bytes, '
        'session_gpu_bytes, '
        'gpu_breakdown [[build stage, bytes]] (pool use added by each stage; "build temporaries freed" is negative), '
        'stage_seconds [[build stage, wall s]] (validation + estimates first; also the "Build time by stage" log '
        'line), '
        'gpu_peak_bytes (build high-water mark of the pool), gpu_budget, gpu_estimate}, or null when no build '
        'succeeded. gpu_estimate is null or {gpu_steady, gpu_build_peak, gpu_run, gpu_external, external_extra, '
        'host, items / run_items / external_items [[label, bytes]], transient_items [[phase, label, bytes]], '
        'steady_deviation, peak_deviation (measured / estimated - 1)}; a deviation above 25 % is also a warning.',
        'build_seconds_total (s: every build of the study summed: sweep values, refinement levels, parity sectors; '
        'one build: build.build_seconds) or null; study_seconds (s: the study runs on the built sessions, without '
        'build and post-processing; summed over sweep values, refinement levels and parity sectors) or null; '
        'time_check (null without a machine profile) the run-time estimate against the runs, merged over sweep values '
        '/ levels / sectors with item labels "<run>: <item>"; solvers {physics: {roles {role: {type, solver (the solver node, null = Automatic), backend, '
        'precision, settings (the resolved solver settings), counters}}, counters}} of the last run, how each result '
        'was computed: role counters pcg {solves, '
        'cg_iterations, unconverged_solves} (linear: heat, electric_currents, electrostatics, magnetostatics, '
        'solid_mechanics, solid_mechanics_3d), sparse_direct {back_end, factorizations, factor_seconds, '
        'recycled_solves, gmres_iterations} (linear: em_fdfd_*, em_fdfd_nonlinear, drift_diffusion, '
        'magnetic_fields_ac; shifted: heat in a thermal_noise study), line_block {batches, shifts, refinement_sweeps, '
        'rejected_shifts (re-solved by SuperLU), kept_factors, factor_solves, rejected_factor_solves, '
        'batched_operators, worst_accepted_residual} (shifted: heat thermal noise, em_fdfd_bor, em_modes; linear: '
        'em_fdfd_ez / _hz), arpack_shift_invert '
        '{eigen_solves, eigen_factorizations, operator_applications (group-index solves included), eigen_unconverged, '
        'eigen_seconds} (eigen: em_modes, em_fdfd_* in an eigenfrequency study), explicit_kernels {} (stepper); the '
        'physics\' own counters (loops and solves outside a role) em_fdfd_* {solves}; em_fdfd_nonlinear {channels, '
        'sweeps, solves}; magnetic_fields_ac {solves}; em_modes {solves, modes_found, solve_seconds, '
        'line_block_fallbacks with a line_block shifted node}; '
        'drift_diffusion {newton_iterations, continuation_steps, cold_starts, rejected_predictions (continuation '
        'steps whose secant prediction did not converge), solve_seconds}; laminar_flow '
        '{lattice_steps, solves}; a damped nonlinear loop (electric_currents / electrostatics nonlinear electrolyte, '
        'magnetostatics saturation) {nonlinear_runs, nonlinear_iterations, nonlinear_unconverged; the electrolyte also '
        'diagonal_refreshes} once it ran; physics with neither are left out (ticks counts the steps of time-stepped '
        'ones).',
        'ticks and tick_unit, what one tick is (time_dependent: runner ticks, step; stationary and optimization: '
        'iterations, iteration; frequency_domain: wavelengths solved, wavelength; mode_analysis: mode snapshots, '
        'mode; eigenfrequency and optomechanics: resonance snapshots, resonance; thermal_noise: factorisations, '
        'factorisation; nonlinear_frequency_domain: channel snapshots, channel; frequency_response: frequencies '
        'solved, frequency), t_end (s; 0 in studies without time '
        'steps), warnings [text], errors [text], issues [issue records of a failed build], traceback (text).',
        'memory {pool_used, pool_reserved, device_free, device_total, budget, other, host_available (available host '
        'RAM, null when unknown), build_gpu_bytes, session_gpu_bytes} in bytes, or null: pool_used and gpu_bytes are process-wide, session_gpu_bytes is the '
        'share of this build. failed_build_memory: null, or of the last failed build {breakdown [[build stage, '
        'bytes]], total_bytes, peak_bytes, budget, failed_owner (the stage that raised), estimate}.',
        'derived {name: value}; derived_reference {name: reference} and derived_error {name: relative error} of the '
        'derived values with a reference; sweep null or {parameter, values, derived {name: [value per sweep value]}, '
        'reference {name: [reference per sweep value]} and error {name: [relative error per sweep value]} (derived '
        'values with a reference; a failed value keeps its reference), results [{"physics.feature": digest} per '
        'sweep value], failed [{index, value, message} per value whose build or run failed], converged [true | false '
        '| null per sweep value: fixed_point.converged, null for a failed value or no fixed point], curve_error '
        '{name: [curve_error record per sweep value]} of the analytic curves on a feature result, derived_series '
        '[derived_series record or null per sweep value], builds (model builds of the sweep: 1 with '
        'SweepSpec.reuse_build, else one per value), with SweepSpec.keep_fields also snapshot_times [[s] per '
        'value] and snapshot_labels [[text or null] per value] of the kept snapshots (results.npz sweep_snapshot)}.',
        'derived_series null or {kind xy, x [time s or wavelength m per stored snapshot, in record order], x_label '
        'Time | Wavelength, x_unit, y_label, series {name: [value per point]}, series_units}: every derived value '
        'without a source evaluated on the live state at each snapshot of a time_dependent or frequency_domain run '
        '(also once the 1 GB snapshot limit stops storing fields; NaN and one warning where it fails), null below 2 '
        'points; parity-sector runs give null.',
        'wavelength_fixed_points (frequency_domain coupled wavelength scan only) [{wavelength, converged, '
        'iterations, relative_change, tolerance} per solved wavelength with a fixed-point loop]; fixed_point then '
        'holds converged = all of them, the largest iterations and relative_change.',
        'run_memory null or {measured_bytes (pool high-water mark of the run since its build), estimated_bytes '
        '(gpu_steady + gpu_run of the estimate), deviation (measured / estimated - 1)}; external_memory null or the '
        'same record of the cuDSS factors outside the pool (measured_bytes: high-water mark of the live factors since '
        'the build started, a plan counting its peak estimate; estimated_bytes: gpu_external + external_extra; null '
        'when neither side has any); parameters {name: value} '
        'of the built session (a sweep: the value the run ended on) or null; sweep_index (0-based) of the sweep value '
        'that top-level derived, fields and results belong to, or null.',
        'Study-specific keys, present only when that study ran: optimization {iterations, evaluations, design_cells, '
        'objective_initial, objective_final, continuation_stages, message} (optimization); noise_frequencies [Hz] '
        '(thermal_noise, empty unless completed); n_modes (mode_analysis: mode snapshots); n_resonances '
        '(eigenfrequency, optomechanics: resonance snapshots); response_frequencies [Hz] (frequency_response: the '
        'solved ones in solve order) and frequency_sweep {method, physics: {name: reduced-model info}} '
        '(frequency_response with sweep_method adaptive / modal; see that setting); n_channels (channel snapshots), nonlinear_solve {physics: {mode, order, '
        'channel_solves, sweeps, converged, residual, tolerance, residual_history, continuation_steps, method, '
        'gmres_iterations, backtracks}} and feedback (solution-feedback record or absent) (nonlinear_frequency_domain).',
        'convergence (StudySpec.adaptation; also convergence.json, Results.convergence()) {target, method, tolerance, '
        'max_levels, growth, max_cells, anisotropic, meshes [refined], levels [{level, cells {mesh: cells incl. '
        'padding}, window_cells {mesh: n}, n_cells, refined_cells (window cells of the refined meshes), value (target), '
        'relative_change, derived {name: value}, study_seconds, gpu_steady (estimate, bytes), status}], richardson null '
        'or {cells, values, levels, order_cells q, order (q x dimensions), dimensions, extrapolated, error_estimate, '
        'relative_error_estimate, coefficient C} of v = v_inf + C N^-q over the last three levels, richardson_note, '
        'converged, stop_reason (converged | max_levels | max_cells | memory | no_refinement | not_finite | cancelled | '
        'failed), message, adapted_meshes [MeshSpec objects of the last level]}; fields, results and derived are the '
        "last computed level's, ticks and study_seconds sum the levels; StudySpec.adaptation keep_levels adds the "
        "earlier levels' snapshots labelled 'level k: ...'.",
        'estimation (StudySpec.estimation) {method, parameters [names], initial / lower / upper (null = unbounded) / '
        'fitted {name: value}, standard_errors {name: value} or null, correlation [[...]] or null, rank, '
        'reduced_chi_square, statistics_note, absolute_sigma, cost (0.5 sum r^2 at the best point), cost_initial, '
        'best_evaluation, evaluations, max_evaluations, targets [{derived, measured, sigma, model, residual}], curves '
        '[{source, n_points, rms_residual, max_abs_residual}], history {evaluation [k], cost [per evaluation], '
        'parameters {name: [value per evaluation]}}, live_is_best, converged, stop_reason (converged | '
        'max_evaluations | not_finite | failed | cancelled), message}; fields, results and derived are those of the '
        'best-point run (a cancelled search: the last run), ticks = evaluations (tick_unit evaluation), study_seconds '
        'sums every run.',
        'parity_sectors null or (study setting parity_sectors) {faces [mirror faces], labels [sector label per run, '
        "the model's own signs last], status [per sector], combined (derived holds the quadratic-total combination), "
        'derived {name: [value per sector]}, results [{"physics.feature": digest} per sector], combined_results '
        '{"physics.feature": digest} (Results.sector_result(..., combined=True)), snapshot_sectors '
        '[sector label per snapshot, "" for a combined one]}; derived, fields and results of the summary are then the '
        'combination (derived) and the last sector (fields, results); ticks and study_seconds sum the sectors. With a '
        'parameter sweep: this block is the last value\'s, sweep derived holds the combined values per sweep value, '
        'sweep results the last sector\'s results per value and sweep sector_derived [{name: [value per sector]} per '
        'sweep value].',
        'curve_error {name: {max, rms, n_points}} of every analytic curve, evaluated at the simulated samples (a '
        'feature result: its t / iteration, wavelength, angle (far field: every wavelength, both normalised to their '
        'peak), x or table x_column; a line: the containing-cell profile points, the analytic at the sampled cell '
        'centres; sweep: the swept values against its series quantity) on the finite points: max = max |simulated - '
        'analytic| / ptp(analytic) (the analytic range: an absolute temperature against its rise; max |analytic| for a '
        'flat curve, absolute when that is 0), rms likewise; target sweep: |simulated - analytic| / |analytic| per '
        'swept value (curves spanning decades). A curve that cannot be compared (series fft or ratio, series "" '
        'with several candidates, no result) has max and rms NaN, n_points 0 and a note.',
        'derived_curves {name: {plot, label, unit, x_label, x_unit, x [abscissa values], y [values], with a '
        'reference also y_reference and curve_error {max, rms, n_points (, note)}} or {plot, error}} of every derived '
        'curve over the final results (a sweep: its last value).',
        'fixed_point {converged, iterations, relative_change, tolerance} of the coupled stationary loop (stationary; '
        'frequency_domain: the background solve, a coupled wavelength scan: over all wavelengths), or null (time_dependent, or no physics loop: a single pass). A run '
        'that did not converge keeps status completed (partial for a sweep with failed values) and says so in '
        'message; its derived values and results are those of the last iteration.',
        'stored_fields ["physics.component"], field_meshes {"physics.component": name of the mesh the field lives on}, '
        'meshes [{name, kind (tensor | triangles), shape (cells incl. padding, null on triangles), elements (triangles, '
        'else null), cells, window_cells, dx_min, dx_max (m: physical cell lengths, element edges), physics [names], '
        'ledger [[build stage, bytes] of the build memory ledger lines of this mesh]} per built mesh], stored_cuts '
        '[[axis, cell-centre position m] of the 3D stored cut '
        'planes every snapshot holds], fields {key: field digest}, results {"physics.feature": result digest, '
        'or {kind, error}}, snapshot_times [s], snapshot_labels [text or null per snapshot: "lambda = <nm> nm" for '
        'frequency_domain, "mode <k>: n_eff = <value>" for mode_analysis, "resonance <k>: lambda = <nm> nm, Q = '
        '<value>" for eigenfrequency (solid mechanics: "resonance <k>: f = <value> Hz"; optomechanics: "mechanical '
        'mode <k>: f = <value> Hz, g0 / 2 pi = <value> Hz"; mirror faces add the parity: "mode <k> [x_lo symmetric]: '
        '..."), "iteration <k>: J = '
        '<value>" per accepted iteration and "final: J = <value>" for optimization, the mode labels then "thermal '
        'noise" for thermal_noise, "channel <name>: lambda = <nm> nm" for nonlinear_frequency_domain, "f = <value> '
        'Hz" for frequency_response, null for time '
        'snapshots; parity sectors: "<label> [<sector>]" and the combined ones], live_snapshot (index of the snapshot '
        'holding the state Results.field reads when that is not the last: resonance 1 of eigenfrequency / '
        'optomechanics, mode_index of mode_analysis; else null), wavelengths [m] (frequency_domain only: the solved list in '
        'solve order, one snapshot each: shorter than requested after a cancel or timeout), harmonic_pulse '
        '(frequency_domain only: null without an em_fdtd_* physics, else {steps, simulated_time [s], stop_reason '
        'decayed | max_duration | cancelled, final_energy_ratio (window field energy / its peak at the end), '
        'source_end [s]} of the harmonic FDTD pulse), files {summary, log, arrays (results.npz or null), meshes '
        '(meshes.npz or null), model (model.json or null), figures ["figures/<name>.png"]}.',
        'status failed: see message (an out-of-GPU-memory failure says so and names the budget and the remedies), '
        'issues, errors and traceback; timeout: max_wall_time was reached and the results are partial; partial: '
        'a sweep finished but some values failed (message "k/n sweep values failed ...", sweep.failed; their derived '
        'values are NaN and their results empty).',
        'A digest that cannot be computed is {error: "Type: message"}. A partial sweep whose last value failed '
        'keeps the sweep table but has empty fields and results.',
        'When writing the report itself fails, the status is failed (exit 1) and summary.json holds only '
        'open_sol_version, status, message ("post-run failure: Type: message; the study ended <its status>"), '
        'errors, traceback and files; when even that cannot be written, Results.summary() holds it with files null '
        'and the reason in errors.',
    )),
    ('show.json', (
        '{open_sol_version, kind (model | results), path, outline, check, summary}: outline (model: title, '
        'description, parameters with their evaluated value, domain, geometry, selections, materials, physics with features, '
        'couplings, mesh, solvers, solver_defaults, study, derived_values, analytic_curves, derived_curves) or null, '
        'check {n_errors, '
        'n_warnings, n_infos} (model) or null, '
        'summary (results: the summary.json content, or {error: "Type: message"} with exit code 1 when the '
        'directory cannot be loaded) or null.',
    )),
    ('selftest_report_<subset>.json', (
        '{open_sol_version, passed (bool), n_cases, n_passed, n_failed, n_error (cases with status error, not in '
        'n_failed), subset (light | heavy | named: which '
        'registered cases the file holds), n_skipped (registered cases not in it), n_reused, reused [case names whose '
        'stored record was reused, counted as passed], wall_time, device {device_id, name, '
        'kind (gpu | cpu), device_total, device_free, budget} [bytes] or null, warnings [text], cases [{name, label, '
        'description, status '
        '(passed | failed | error), measured, expected, error, tolerance, details, message, wall_time, '
        'vram_deviation, vram_part, vram_verdict, count_deviation, count_item, runs {run name: {status, wall_time, message, out_dir, timing, '
        'vram, counts}}, variants (cases with '
        'variants only) {variant name: '
        '{status (passed | failed | error | skipped), backend, precision, measured, expected, error, tolerance, '
        'details, message, wall_time, runs}}}]}.',
        'A variant repeats the case\'s runs on another backend and / or solver precision (run directories '
        '<case>/variant_<name>/<run>/) and judges the same check against its own tolerance; a failed or erroring '
        'variant fails the case, a gpu variant without a usable GPU is skipped.',
        'timing {build_seconds (summary build_seconds_total), study_seconds, output_seconds (run() after the study: '
        'summary, host copies, files), '
        'n_cells, ticks, cell_ticks_per_second (time_dependent: n_cells * '
        'ticks / study_seconds, else null), solvers} from the run summary, or null when the run raised: a '
        'performance baseline, comparable only between reports of one machine (the first run of an engine in a '
        'process also loads its compiled kernels); the solvers counters are machine-independent.',
        'vram (gpu runs with a memory estimate, else null) {kept, build_peak, run, external: {measured_bytes, '
        'estimated_bytes, deviation (measured / estimated - 1), verdict, beyond_floor (differ by more than 1 MiB)} or '
        'null, deviation, part, verdict, beyond_floor}: kept / '
        'build_peak from the summary build ledger, run = summary run_memory, external = summary external_memory. The '
        'estimate counts the big arrays held at the same time, so it sits just below the measured bytes: verdict over '
        '(it books more than was held: the estimator misreads the solver), unaccounted (more than 25 % held beyond '
        'it: an array no item lists, or memory the code could free) or empty; deviation / part / verdict = the worst '
        'part (one with a verdict first, then one beyond the floor, then the largest |deviation|). counts (null without a run-time estimate or counted items) {deviation, item}: the time_check '
        'item (predicted count with a summary counter, e.g. CG iterations) whose measured / predicted - 1 is the '
        'largest, signed, and its label (0 and null when none is off by more than one). vram_deviation, vram_part, '
        'vram_verdict, count_deviation / count_item (case and example records) = the worst / largest over its runs '
        '(null without one): the memory estimators / time recipes to update (CLI: "vram -40%  counts +200% (item)" '
        'after the time split; - = the estimate books too much, + = more held than estimated).',
        'A case passes when its error is finite and error <= tolerance; a run that does not complete fails its '
        'case, and an exception in the comparison makes it an error.',
        'An interrupted selftest writes the cases already run (passed false, a warning says so); a failing progress '
        'callback is dropped with a warning.',
        'Built-in cases, run first: templates_validate (every template validates without errors), '
        'examples_validate (every examples/**/*.json of a repository checkout loads and validates without errors), '
        f'api_reference_current ({REFERENCE_FILE} and {TEMPLATES_FILE} equal the text describe generates), '
        'import_contracts (every package import keeps the layering rules of model/import_contracts.py) and '
        'structure_index_current (the repository module index equals the text generated from the module '
        'docstrings); they run in the light, heavy and all subsets and appear in each of their report files; a '
        'named run (names) runs only the built-ins it names.',
        'A selftest --all run writes the light and the heavy file from one run (wall_time, device and warnings are '
        'the whole run\'s), then the examples files per backend; api.selftest returns the combined dict with subset '
        'all, report_files [names] and examples {cpu, gpu: {passed, n_examples, n_passed, n_failed, n_error, '
        'n_no_tolerance, n_skipped, n_skipped_backend, n_reused, wall_time, warnings}}; its passed also needs both '
        'example runs.',
        'Before a report file (this one or an examples file) is overwritten, its timing summary goes to '
        '<file stem>_prev.json {source, written (file time), open_sol_version, subset / backend, passed, counts, '
        'n_reused, wall_time, device, cases {case name or example path: {status, wall_time, reused}}}: the previous '
        'run to compare run times with (a reused case keeps its stored wall_time; compare --reset runs).',
    )),
    ('selftest_report_analytic.json', (
        'Also selftest_report_analytic_cpu.json / _gpu.json (selftest --all, api.selftest_examples backend=): the '
        'same content for the examples forced onto that backend.',
        '{open_sol_version, passed (no failed or erroring example), backend (null: each example on its own), '
        'n_examples, n_passed, n_failed, n_error, n_no_tolerance, n_skipped, skipped [example paths without a '
        'comparison], n_skipped_backend, skipped_backend [example paths validating only on their own backend, not '
        'run on the forced one], n_reused, reused [example paths whose stored record was reused], wall_time, device, '
        'warnings, ranking '
        '[{example, kind (derived | curve), name, error, tolerance, ratio, flags}] of the entries with a tolerance by '
        'ratio = error / tolerance (non-finite first, then descending), without_tolerance [the same rows] by error, '
        'examples [...], report_files [its own file name; [] without out_dir]}.',
        'Example: {example (path under examples/), title, status (passed | failed | error | no_tolerance), '
        'run_status, message, wall_time, out_dir, worst_ratio, flags, backend (the run\'s; null when the file did not '
        'load), timing, vram, counts, vram_deviation, vram_part, vram_verdict, count_deviation, count_item (as a selftest run / case), derived '
        '[{kind, name, value, reference, error, '
        'tolerance, ratio, flags (+ per_value_error in a sweep)}], curves [{kind, name, target, series, error (the '
        'curve_error max; a sweep: the worst value), rms, n_points, tolerance, ratio, flags, note (+ '
        'per_value_error)}]}. A sweep derived value: value / reference lists per value, error the worst value.',
        'Flags: over_tolerance, nonfinite, not_compared (curve without simulated counterpart), suspicious_exact '
        '(error < 1e-9 for a simulated quantity with a non-zero reference: the reference likely reads the result), '
        'sweep_not_converging (the sweep changes the mesh dx, auto mode points_per_wavelength, and the error at the '
        'finest mesh is not below the one at the coarsest; not when every error is below 1e-9). An entry with a '
        'tolerance fails on over_tolerance, '
        'nonfinite or not_compared; error: the run did not complete (failed, partial, timeout) or raised.',
    )),
    ('Digests', (
        'field: {physics, component, label, unit, shape [nx, ny], min, max, mean (area-weighted over the cell '
        'areas), abs_max, abs_max_at [x, y] (cell centre of the largest |value|, m), n_nonfinite, time, '
        'time_label}; statistics over the finite cells, null when none is finite.',
        'timeseries: {kind, component, unit, n_samples, t_start, t_stop, min, max, final}.',
        'flux_spectrum: {kind, unit (of power: a.u. or W/m), n_wavelengths, wavelength_min, wavelength_max, '
        'power_min, power_max, wavelength_at_max}; power statistics over the finite values.',
        'dft_fields: {kind, quantity, shape, n_wavelengths, wavelengths, intensity_max [max per wavelength of the '
        'intensity of the monitor quantity (|E_z|^2 or |E_x|^2 + |E_y|^2)] or null}.',
        'far_field: {kind, quantity, unit, n_wavelengths, n_angles, wavelengths, power_max [per wavelength: the '
        'largest finite power], angle_at_max_deg [per wavelength]}.',
        'xy: {kind, x_label, x_unit, n_points, x_min, x_max, series {name: {unit, min, max, x_at_max}}}; statistics '
        'over the finite values only (NaN marks a value that could not be computed).',
        'table: {kind, columns, units, n_rows, rows [the first 50 rows]}.',
        'Values that cannot be computed (empty arrays, no finite value) are null.',
    )),
    ('results.npz', (
        'np.savez archive, read with np.load(path, allow_pickle=False); written unless run --no-arrays '
        '(save_arrays=False).',
        'field/<physics>.<component>/values (float32 (nx, ny) window cells), .../x_edges and .../y_edges (float64 '
        'cell edges, m).',
        'result/<physics>.<feature>/<key> for every numeric array value of >= 1 dimension, and '
        'result/<physics>.<feature>/<key>/<name> for every such array inside a dict value (one level, e.g. the xy '
        'series/<name>; loading rebuilds the dict, whose other entries stay in the metadata); a dft_fields result '
        'adds intensity_all, float64 (n_f, mx, my) = the intensity of the monitor quantity (|E_z|^2 or |E_x|^2 + '
        '|E_y|^2) at every wavelength.',
        'sweep/<index>/<physics>.<feature>/<key> (and .../<key>/<name>) per sweep value, sector/<index>/... alike '
        'per parity sector; sweep_snapshot/<value index>/<snapshot index>/... as snapshot/<index>/... per kept '
        'snapshot of a keep_fields sweep (metadata sweep_snapshots [[snapshot entry] per value]); '
        'snapshot/<index>/<physics>.<component>/values per snapshot (snapshots are saved only here), plus '
        '.../x_centers and .../y_centers (float64 cell centres, m) when the values sit on the native window cells '
        '(native polar / spherical cells also .../mapped_x and .../mapped_y, the float64 drawing-plane centres); '
        'without them the values are nearest-cell resampled on uniform pixels over extent; 3D models add '
        'snapshot/<index>/cut<c>/<physics>.<component>/... per stored cut and snapshot/<index>/segment/<curve '
        'name>/distance, values, centre_x, centre_y, centre_z (float64) per 3D-segment line curve.',
        "metadata: 0-d unicode JSON string (json.loads(str(npz['metadata']))) {format (open_sol.results), version "
        '(1), fields {key: {physics, component, info {name, label, unit, signed, cmap, primary}, time, time_label, '
        'plane, mesh}}, results {key: non-array entries (table rows included)}, sweep_results [...], sector_results '
        '[...], sector_combined_results {key: ...} (arrays under sector_combined/<key>/), '
        'snapshots [{time, label, fields {key: {extent, info, time_label, native_shape ([nx, ny] window cells when '
        'resampled, else null), plane, axes (names of the value axes), unfolded (already mirrored over the mirror '
        'faces: parity-sector '
        'snapshots), mesh}}, cuts [fields per stored cut], segments {curve name: {physics, component, start, end, '
        'info, time_label}}}]}; plane ["z", cut height in m] of a 3D model, else null; mesh = the name of the mesh '
        'the field lives on (fields of a triangle mesh are saved as their raster over the window); tuples come back '
        'as lists, non-finite floats outside arrays as "nan", "inf", "-inf".',
    )),
    ('meshes.npz', (
        'The meshes the fields live on, written with results.npz: <mesh>/x_edges, <mesh>/y_edges (3D: + z_edges) = '
        'float64 window cell edges of a tensor mesh in Domain coordinates (m; polar / spherical windows: logical r - '
        'radius_min [m] and angle - angle_min [rad]), or <mesh>/points (float64 (n, 2) m), <mesh>/triangles (int32 '
        '(k, 3)) and <mesh>/pml_faces (uint8 per element, bits x_lo 1, x_hi 2, y_lo 4, y_hi 8; 0 inside the window) '
        'of a triangle mesh, or <mesh>/points (float64 (n, 3) m) and <mesh>/tets (int32 (k, 4)) of a tetrahedron '
        "mesh; metadata {format (open_sol.meshes), version (1), meshes {name: {kind}}}.",
    )),
    ('log.txt', (
        "One line per event, '[<elapsed s>] [<level>] <message>': log events, the build (Built, or BuildFailed with "
        'its issues and traceback), sweep steps, the end of the study (finished or failed), the time-out and GPU '
        'memory leak warnings; warnings at Results.close() are appended.',
    )),
    ('Python API (open_sol.api, re-exported by open_sol)', (
        "describe(topic=None, as_dict=False) -> this reference as text (templates as an index; describe('templates') "
        'has their JSON), or as a dict; template(key) -> ModelSpec; '
        'load(path) -> ModelSpec; save(spec, path).',
        'spec arguments may be a ModelSpec (copied), a dict (ModelSpec.from_dict) or a model file path; invalid '
        'content is a model error, an unreadable file raises OSError.',
        'parameters={name: expression string or number} (check and run) overrides parameter expressions of the spec '
        'copy (numbers stored as repr(float); run records them in model.json); an unknown name is an error issue at '
        '("parameters",) (run without out_dir: KeyError listing the known names).',
        'check(spec, out_dir=None, parameters=None, memory_budget=None, host_budget=None, backend=None, '
        'max_wall_time=None) -> the check.json dict (also written with out_dir); memory_budget [bytes] (e.g. a run '
        'summary build.gpu_budget) adds the GPU memory check, host_budget [bytes] (e.g. '
        'open_sol.core.host_memory.available_bytes()) the host RAM check; backend (\'auto\' | \'gpu\' | \'cpu\') '
        'overrides solver_defaults.backend; max_wall_time [s] adds a warning when even the shortest estimated run '
        'time exceeds it.',
        'profile_machine(backend=None, quick=False, progress=print) -> {key, date, quick, text, path}: measures and '
        'stores the machine profile of the run-time estimates (CLI profile; backend None = the default backend).',
        'scene_figure(spec, path, parameters=None, highlight=(), z_stretch=1.0, view=None) -> path: CPU-only '
        'perspective picture of a 3D model (.png / .pdf / .svg by the suffix; ValueError in 2D) as the GUI 3D preview '
        'draws it: translucent objects in their material colours, points / lines as markers, highlight = entity names '
        'drawn in the highlight colour, z_stretch scales the box height only (thin layers), view = (elevation, '
        'azimuth) in degrees; objects that do not resolve are left out with a RuntimeWarning.',
        'run(spec, out_dir=None, max_wall_time=None, progress=None, figures=False, save_arrays=True, '
        'parameters=None, backend=None) -> Results: '
        'blocking, runs on the calling thread on the model\'s backend (backend \'auto\' | \'gpu\' | \'cpu\' overrides '
        'solver_defaults.backend, kept in model.json); progress(event) receives every event except live frames. '
        'With out_dir it always writes summary.json and log.txt and never raises (except KeyboardInterrupt); '
        'without out_dir a failed build re-raises its exception (ModelError with .issues), a failed run raises '
        'RunError(message, traceback), and a timeout returns normally. progress=True prints to stdout '
        'through ProgressPrinter(stream=None, period=2.0) (the CLI printer: build, log and sweep lines, Progress '
        'lines at most every period s). Events, in open_sol.api: LogEvent(level, message), SweepStep(index (0-based), '
        'count, parameter, value), Built(summary: BuildSummary with nx, ny, n_cells, dt, build_seconds, ...), '
        'BuildFailed(message, traceback, issues, out_of_memory, build_memory), StudyStarted(), Progress(fraction, t, '
        "t_end [s], tick, wall_elapsed, ticks_per_second, message, tick_unit 'step' | 'iteration' | 'wavelength' | "
        "'mode solve' | 'eigen solve' | 'factorisation' | 'channel solve', "
        'stages (stage names of the run in order, e.g. Background solve, Wavelength solves), stage_index (0-based), '
        'stage_fraction, stage_elapsed [s]; fraction and wall_elapsed cover the whole run, each stage an equal share), '
        'SweepValueDone(index, count, parameter, value, sweep: a copy of the study sweep table so far, after each '
        'computed or failed sweep value), '
        'StudyFinished(summary dict), StudyFailed(message, traceback, out_of_memory, session_live, build_memory). '
        'build_memory: the build ledger record (summary.json failed_build_memory layout) or null.',
        'Results: status, spec, out_dir, session (live only), build_summary, log, is_live; summary() -> dict; '
        'field_names(), result_names(); convergence() -> copy of the mesh refinement report of a '
        'StudySpec.adaptation run (KeyError without one); field(name, physics=None, plane=None, unfold=False, '
        'native=False) -> FieldData (unfold=True: mirrored over the model\'s x / y mirror faces, ValueError for a '
        'field without a parity; values '
        'float32 (nx, ny) '
        'on the native window cells, x_centers / y_centers float64 in m, extent, info, time, plane, axes; 3D models: '
        "plane=(axis, position [m]) picks the cut normal to axis 'x' | 'y' | 'z' (values along axes ('y', 'z') / "
        "('x', 'z') / ('x', 'y'), x_centers / y_centers and extent along them), live Results only, default and saved "
        'fields the mid-height z cut; polar / spherical domains: a containing-cell image over the sector\'s box, NaN '
        "outside, native_shape set; native=True (live only) gives the window cells instead: axes ('r', 'phi' | "
        "'theta'), x_centers = r in m, y_centers = the angle in rad, extent (r_min, r_max, angle_min, angle_max), "
        'mapped_centres = the drawing-plane (x, y) centres in m, float64 (nx, ny) each); volume(name, physics=None, '
        'max_bytes=2**30) -> VolumeData of a 3D field '
        '(values float32 (wx, wy, wz) on the window cells, centres (x, y, z) float64 in m, info, time; live only, '
        'ValueError above max_bytes) and export_volume(name, path, physics=None, max_bytes=2**30) -> its .npz '
        '(values, x_centers, y_centers, z_centers, metadata kind volume); '
        'result(physics, feature, '
        "**options) or result('physics.feature', **options) -> result dict (dft_fields: freq_index=i adds "
        "freq_index and intensity); figure(what, path, title='', freq_index=None, series=None) -> PNG path of a field "
        '(physics.component or a unique component) or a feature result (physics.feature; series: only that series '
        'of an xy result); scene_figure(path, field=None, physics=None, planes=None, z_stretch=1.0, view=None) -> '
        'path of a 3D run: the api scene_figure picture of its model (built parameters) with field on cut planes in '
        'one colour scale, the objects then as edges only (live: planes [(axis, position [m])], default the middle '
        'window cell on x, y and z; closed / loaded: the saved mid-height z cut plus the stored cuts of a last '
        'snapshot at the same time); derived_curves() -> '
        '[(plot, xy result or None, {curve name: error text})] of the derived curves over the final results; '
        'save(path, save_arrays=True); close(copy_to_host=True); Results.load(path).',
        'Stored states (live, closed and loaded alike): snapshots() -> [(time, label, [physics.component keys])] '
        '(frequency_domain: one per wavelength, field() holds only the last); snapshot_field(index, name, '
        'physics=None, plane=None) -> FieldData of one snapshot, float32 on the native window cells (x_centers / '
        'y_centers set, graded meshes included) up to 250,000 window cells, else nearest-cell resampled on a uniform '
        'grid over extent (x_centers None, native_shape set); 3D: plane None = the mid-height z cut, (axis, position '
        "[m]) = the StudySpec.stored_cuts plane on that axis nearest position, an int = that stored cut; "
        'sweep_result(physics, feature) or '
        "sweep_result('physics.feature') -> [result dict, or None for a failed value] in summary()['sweep']"
        "['values'] order; sweep_xy(physics, feature=None, column='frequency', sort=True) -> 'xy' result of a "
        "per-value 'table' result vs the swept parameter, series '<column>_<k>' = the k-th smallest value of that "
        'column per sweep value (sort=False: the k-th row), NaN where missing (e.g. the bands of a '
        "resonance_monitor over a Bloch k sweep); sweep() -> copy of the sweep table {parameter, values, derived, "
        'reference, error, converged, failed, derived_series}; sweep_field(index, name, snapshot=-1, physics=None, '
        'plane=None) -> FieldData of a kept snapshot of sweep value index (keep_fields; snapshot / name / plane as '
        "snapshot_field); sweep_figure(quantity, path, series=None, title='') -> PNG of a sweep: quantity "
        "'physics.feature' = one curve of that result per swept value (legend '<parameter> = <value>'; series picks "
        "an xy series or a flux_spectrum power / power_per_cw; far fields in dB at their first wavelength), "
        "'derived' = every derived value vs the swept parameter, 'derived_series:<name>' = that derived value along "
        'the study axis per swept value (run --figures writes figures/sweep_derived.png, sweep_<key>.png and '
        'sweep_series_<name>.png of a sweep, derived_series.png of a run with a derived series); '
        "sector_result(physics, feature) -> [result dict or None per parity "
        "sector] in summary()['parity_sectors']['labels'] order (combine quadratic totals, e.g. flux power, with "
        'open_sol.core.mirror.combine_sector_totals, far-field patterns per angle with '
        'open_sol.core.mirror.combine_sector_far_fields: the mean of the complex amplitude_re / amplitude_im of the '
        'far_field results, power = |mean|^2; a sweep: the last value\'s sectors); sector_result(physics, feature, '
        'combined=True) -> the stored result of the drawn source alone (driven studies where every sector ran: '
        'far_field combined per angle, flux_spectrum power / power_per_cw as quadratic totals when its line / plane / '
        "box reaches every mirror plane, i.e. mirror_factor = 2^k; entry combined_sectors = 2^k), KeyError otherwise.",
        'Non-finite numbers in summary() derived, derived_reference, derived_error, sweep derived / reference / error, '
        'curve_error max / rms (sweep too) and fixed_point relative_change, in table rows and in the scalar entries '
        'of result dicts (e.g. resonance_monitor start_time) are floats in every state (Results.load converts the '
        'JSON strings back); digest statistics stay "nan" / "inf" strings.',
        'export_field(field_data, path) and export_result(result, path) (open_sol, from open_sol.io) write a '
        'FieldData / result dict as .npz or .csv (field metadata: plane of a 3D cut, axes, resampled / '
        'native_shape; a native polar / spherical field adds x_mapped / y_mapped arrays (.npz) or x / y columns after '
        'r and the angle (.csv)); export_volume(volume_data, path) a VolumeData as .npz only.',
        'A live Results holds the GPU session: close() (or a with block) copies the stored fields and feature '
        'results to host memory and frees the GPU (copy_to_host=False skips the host copy); a closed or loaded '
        'Results is CPU-only (saved fields and results only; result() options other than freq_index raise '
        'ValueError). Close it before the next run: r = run(...) in a loop keeps the old session on the GPU while '
        'the next one builds (run warns while earlier Results are live); use "with run(spec) as r:" instead.',
        'selftest(out_dir=None, names=None, progress=None, heavy=False, reset=False) -> the selftest report dict (+ '
        'report_files: the files written to out_dir); heavy False runs the light registered cases, True the heavy '
        'ones (>= 5 s each), None everything (all cases, then selftest_examples with backend cpu and gpu; progress '
        'also gets the example records); names overrides it. With out_dir, unchanged passing cases are reused from '
        'out_dir/selftest_cache.json (CLI selftest Reuse); reset=True reruns them. progress(dict) gets events and '
        'records: {event: begin, phase, n_run, n_reused, n_total} per phase, {event: start, index, n_total, name, '
        'last_wall_time (s or None), remaining (s, summed last run times of it and every later test), n_untimed, '
        'time_estimate ({seconds_low, seconds_guess, seconds_high (build + study + output), phases {build, study, '
        'output: the same keys}, n_runs, n_estimated, message, not_estimated} of its runs on the stored machine '
        'profile, or None)} before each test, then its case / example record (time_estimate: {before, '
        'n_checked_runs, split {total, build, study, output, other: {seconds, estimate}, study_work_at_build}}: '
        'study estimated at the runs\' measured counts, its counter-timed work done at build (e.g. a factorisation '
        'in finalize; study_work_at_build s) moved from build to study seconds, build / output the guesses before, '
        'other (checks, run set-up) never estimated; '
        '{event: skipped, name, message} for a case this machine cannot run).',
        'selftest_examples(out_dir=None, names=None, progress=None, backend=None, reset=False) (open_sol.api) -> the '
        'selftest_report_analytic.json dict (written with out_dir); backend \'cpu\' / \'gpu\' forces every example '
        'onto it (selftest_report_analytic_<backend>.json, examples_<backend>/ run directories); progress gets the '
        'events of selftest and each example record; reuse and reset as selftest.',
        'save_session_run(directory, spec, session, build_summary, study_summary, log_entries, model_file=None, '
        'save_arrays=True) -> directory: writes the run directory run(out_dir=) would of a live session its caller '
        'built and ran (the GUI worker); the session stays open; log_entries = [(clock text, level, message)].',
        'Never mix api.run with the GUI worker in one process.',
    )),
    ('GPU memory', (
        'Runs use a GPU memory budget (cupy pool limit), recomputed at the start of every run, of at most 90% of '
        'the device memory and at most the free memory plus the blocks already cached by the cupy pool, minus 512 '
        'MiB headroom; when that leaves nothing the run fails before building (an error at model). Running out of '
        'it fails '
        'the run cleanly: status failed, with a message naming the budget and the remedies (fewer cells, smaller '
        'DFT / far-field boxes or fewer n_wavelengths, closing other GPU programs).',
        'A derived value that runs out of GPU memory becomes NaN plus a warning.',
    )),
    ('Regenerating this reference', (
        f'After a registry or template change regenerate {REFERENCE_FILE} and {TEMPLATES_FILE} with: '
        f'{REGENERATE_COMMAND}. selftest (api_reference_current) fails while either file differs from the '
        'generated text.',
    )),
)


# ---- public API ----
class UnknownTopicError(LookupError):
    """describe() of a topic not in describe_topics(); the message names the valid topics."""


def describe(topic=None, as_dict=False, registry=None):
    """Reference markdown text (ASCII, one final newline; the full text = REFERENCE_FILE) or JSON-able dict of every
    topic or one (describe_topics()); UnknownTopicError for an unknown topic, then the documentation checks."""
    registry = get_registry() if registry is None else registry
    topics = describe_topics(registry)
    if topic is not None and not (isinstance(topic, str) and topic in topics):
        raise UnknownTopicError(f"unknown describe topic {topic!r}; valid: {', '.join(topics)}")
    reference = _checked_reference(registry, None if topic is None else _topic_sections(registry, topic))
    view = reference if topic is None else _topic_view(reference, topic)
    if as_dict:
        return view
    if topic is not None:
        return _markdown(_view_blocks(view))
    return _reference_file_text(reference, registry)


def reference_files(registry=None) -> dict:
    """{REFERENCE_FILE: describe(), TEMPLATES_FILE: the template JSON text}: what --update-reference writes and
    selftest api_reference_current compares."""
    registry = get_registry() if registry is None else registry
    reference = _checked_reference(registry)
    return {REFERENCE_FILE: _reference_file_text(reference, registry),
            TEMPLATES_FILE: _templates_file_text(reference)}


def describe_topics(registry=None) -> list:
    """Valid topics in reference order: model sections, physics types, '<physics>.<feature>', coupling and study
    types (each sorted by name), templates, results, selftests; a name shared with a physics type resolves to the
    physics."""
    registry = get_registry() if registry is None else registry
    return list(dict.fromkeys(topic for _, names in _topic_groups(registry) for topic in names))


def _topic_groups(registry):
    """[(header rule text, topic names)] in reference order: a fixed topic names itself, a registry group gets a
    rule (the header prints the rules, so a new group appears there without an edit)."""
    physics_types = _sorted_by_name(registry.physics_types())
    return [
        *[(topic, [topic]) for topic in _LEADING_TOPICS],
        ('each physics type name', [physics_type.name for physics_type in physics_types]),
        ('`<physics>.<feature>`', [f'{physics_type.name}.{feature.name}' for physics_type in physics_types
                                   for feature in physics_type.features]),
        ('each coupling type name', [coupling_type.name
                                     for coupling_type in _sorted_by_name(registry.coupling_types())]),
        ('each study type name', [study_type.name for study_type in _sorted_by_name(registry.study_types())]),
        *[(topic, [topic]) for topic in (*_TRAILING_TOPICS, _SELFTESTS_TOPIC)],
    ]


# ---- checks ----
def _checked_reference(registry, sections=None):
    """Reference dict (sections None: all; else _reference's subset) after the documentation checks: RuntimeError
    listing missing / extra keys of the hand-written tables (result kinds, functions, constants, param kinds, spec
    fields), then non-ASCII texts of the sections built."""
    problems = _table_problems()
    reference = None
    if not problems:
        reference = _reference(registry, sections)
        problems = _non_ascii_problems(reference, 'describe()')
    if problems:
        raise RuntimeError('API documentation check failed (fix it, then regenerate api_reference.md with '
                           f'{REGENERATE_COMMAND}):\n' + '\n'.join(f'- {problem}' for problem in problems))
    return reference


def _table_problems():
    """Missing / extra keys of each documentation table against the keys it documents."""
    spec_field_keys = [f'{spec_class.__name__}.{spec_field.name}'
                       for spec_class, _ in _spec_classes() for spec_field in fields(spec_class)
                       if _mesh_param(spec_class, spec_field.name) is None]
    return (_key_problems('RESULT_KIND_DOCS (model/schema.py)', RESULT_KIND_DOCS, RESULT_KINDS)
            + _key_problems('FUNCTION_DESCRIPTIONS (model/expressions.py)', FUNCTION_DESCRIPTIONS, FUNCTION_ARITY)
            + _key_problems('CONSTANT_DESCRIPTIONS (model/expressions.py)', CONSTANT_DESCRIPTIONS, BUILTIN_CONSTANTS)
            + _key_problems('_PARAM_KIND_HELP (model/describe.py)', _PARAM_KIND_HELP, PARAM_KINDS)
            + _key_problems('_SPEC_FIELD_HELP (model/describe.py)', _SPEC_FIELD_HELP, spec_field_keys))


def _key_problems(table_name, table, documented_keys):
    """'<table> misses a, b' and '<table> has extra keys c' for one documentation table."""
    documented = list(documented_keys)
    missing = [str(key) for key in documented if key not in table]
    extra = sorted(str(key) for key in table if key not in documented)
    problems = []
    if missing:
        problems.append(f"{table_name} misses {', '.join(missing)}")
    if extra:
        problems.append(f"{table_name} has extra keys {', '.join(extra)}")
    return problems


def _non_ascii_problems(value, path):
    """'non-ASCII text at <path>: <escaped excerpt>' for every non-ASCII string or key of a plain reference value."""
    if isinstance(value, str):
        if value.isascii():
            return []
        first = next(index for index, character in enumerate(value) if not character.isascii())
        return [f'non-ASCII text at {path}: {ascii(value[max(0, first - 30):first + 30])}']
    problems = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = key if key.isascii() else ascii(key)
            problems += _non_ascii_problems(key, f'{path} (key {key_text})')
            problems += _non_ascii_problems(item, f'{path}.{key_text}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            problems += _non_ascii_problems(item, f'{path}[{index}]')
    return problems


# ---- reference dict ----
def _reference(registry, sections=None):
    """JSON-able reference: top-level keys in topic order, registry types sorted by name; sections None = all,
    else the set of top-level keys to build ('results' without its selftests list unless 'selftests' is in it: that
    list imports every selftests module)."""
    physics_types = _sorted_by_name(registry.physics_types())
    builders = {
        'spec': _spec_section,
        'expressions': lambda: _expressions_section(physics_types),
        'geometry': _geometry_section,
        'materials': lambda: _materials_section(registry),
        'mesh': lambda: _mesh_section(physics_types),
        'solvers': lambda: _solvers_section(registry, physics_types),
        'physics_types': lambda: {physics_type.name: _physics_entry(physics_type) for physics_type in physics_types},
        'coupling_types': lambda: {coupling_type.name: _coupling_entry(coupling_type)
                                   for coupling_type in _sorted_by_name(registry.coupling_types())},
        'study_types': lambda: {study_type.name: _study_entry(study_type)
                                for study_type in _sorted_by_name(registry.study_types())},
        'templates': lambda: _templates_section(registry),
        'results': lambda: _results_section(registry, physics_types,
                                            with_selftests=sections is None or _SELFTESTS_TOPIC in sections),
    }
    return {key: build() for key, build in builders.items() if sections is None or key in sections}


def _topic_sections(registry, topic) -> set:
    """Top-level reference keys (plus _SELFTESTS_TOPIC for the selftests list) a valid topic's view needs."""
    if topic in _LEADING_TOPICS or topic == 'templates':
        return {topic}
    if topic in ('results', _SELFTESTS_TOPIC):
        return {'results', _SELFTESTS_TOPIC}
    if any(physics_type.name == topic for physics_type in registry.physics_types()):
        return {'physics_types'}
    if '.' in topic:   # '<physics>.<feature>': the feature entry and its result kind's keys
        return {'physics_types', 'results'}
    return {'coupling_types', 'study_types'}


def _topic_view(reference, topic):
    """The part of the reference one topic covers, under its top-level key (a feature topic as
    {'physics_types': {physics: {'features': {feature: ...}}}}, its result kind's keys inlined as 'result_keys');
    'selftests' as {'selftests': the results section's case list}; None for an unknown topic."""
    if topic in _LEADING_TOPICS or topic in _TRAILING_TOPICS:
        return {topic: reference[topic]}
    if topic == _SELFTESTS_TOPIC:
        return {_SELFTESTS_TOPIC: reference['results']['selftests']}
    physics_types = reference['physics_types']
    if topic in physics_types:
        return {'physics_types': {topic: physics_types[topic]}}
    physics_name, _, feature_name = topic.partition('.')
    features = physics_types[physics_name]['features'] if physics_name in physics_types else {}
    if feature_name in features:
        feature_entry = dict(features[feature_name])
        result_kind = feature_entry['result_kind']
        if result_kind is not None:
            feature_entry['result_keys'] = reference['results']['result_kinds'][result_kind]['keys']
        return {'physics_types': {physics_name: {'features': {feature_name: feature_entry}}}}
    for key in ('coupling_types', 'study_types'):
        if topic in reference[key]:
            return {key: {topic: reference[key][topic]}}
    return None


def _spec_classes():
    """[(spec dataclass, where it sits)] reachable from ModelSpec, depth first in field order."""
    found, seen = [(ModelSpec, 'the model file (root object)')], {ModelSpec}

    def visit(spec_class):
        nested = _NESTED_FIELDS.get(spec_class, {})
        for spec_field in fields(spec_class):
            if spec_field.name not in nested:
                continue
            child_class, arity = nested[spec_field.name]
            if child_class in seen:
                continue
            seen.add(child_class)
            place = _SPEC_CLASS_PLACES[arity].format(owner=spec_class.__name__, field=spec_field.name)
            found.append((child_class, place))
            visit(child_class)

    visit(ModelSpec)
    return found


def _spec_section():
    return {
        'about': 'A model is one JSON object (ModelSpec.to_dict(); api.save / api.load) made of the dataclasses '
                 'below. Unknown keys, missing required keys, a wrong list / dict / text type and another version '
                 'fail on load; everything else is reported by validation (check) as issues at node paths.',
        'dataclasses': [{'name': spec_class.__name__, 'where': place,
                         'fields': [_spec_field_entry(spec_class, spec_field) for spec_field in fields(spec_class)]}
                        for spec_class, place in _spec_classes()],
        'param_kinds': {kind: _PARAM_KIND_HELP[kind] for kind in PARAM_KINDS},
        'notes': _spec_notes(),
    }


def _spec_field_entry(spec_class, spec_field):
    """{'name', 'type', 'required', 'default', 'help'} of one spec dataclass field (a factory as its value)."""
    if spec_field.default is not MISSING:
        required, default = False, _plain(spec_field.default)
    elif spec_field.default_factory is not MISSING:
        produced = spec_field.default_factory()
        if isinstance(produced, list):   # ModelSpec.meshes: [MeshSpec()]
            produced = [asdict(item) if is_dataclass(item) else item for item in produced]
        required, default = False, _plain(asdict(produced) if is_dataclass(produced) else produced)
    else:
        required, default = True, None
    mesh_param = _mesh_param(spec_class, spec_field.name)
    help_text = (_SPEC_FIELD_HELP[f'{spec_class.__name__}.{spec_field.name}'] if mesh_param is None
                 else f'{mesh_param.label} (see mesh: Mesh settings).')
    return {'name': spec_field.name, 'type': _type_name(spec_field.type), 'required': required, 'default': default,
            'help': help_text}


def _mesh_param(spec_class, field_name):
    """The MESH_PARAMS Param documenting a MeshSpec field (its help lives there), else None."""
    if spec_class is not MeshSpec:
        return None
    return next((param for param in MESH_PARAMS if param.name == field_name), None)


def _type_name(annotation):
    """'float | int | str', 'SweepSpec | None', 'list' ... of a field annotation."""
    if isinstance(annotation, types.UnionType):
        return ' | '.join(_type_name(member) for member in annotation.__args__)
    if annotation is type(None):
        return 'None'
    return getattr(annotation, '__name__', str(annotation))


def _spec_notes():
    """Heading -> rule lines of the spec section (names, entities, values, physics, derived, issues)."""
    face_texts = ', '.join(f"{face} ({FACE_LABELS[face].lower().removesuffix(' edge')})" for face in OUTER_FACES)
    label_examples = ', '.join(f'{json.dumps(list(path))} -> {node_label(path)}'
                               for path in ((), ('mesh',), ('feature', 'em', 'source')))
    return {
        'Names': [
            'Node names (parameters, geometry objects, selections, materials, physics, features, couplings, meshes, '
            'derived values, analytic and derived curves) are Python identifiers (letters, digits and _, not starting with a digit), unique within their '
            'kind; feature names are unique within their physics.',
            'Geometry objects and selections share one namespace and may not use the reserved entity names '
            f"{', '.join(RESERVED_ENTITY_NAMES)}.",
            'Parameter names may not be reserved expression names (functions, constants, x, y, z, t, value) or '
            'physics field names (see expressions).',
        ],
        'Entities and selections': [
            f"Selection kinds: {' | '.join(SELECTION_KINDS)}. An entity list (MaterialSpec.selection, "
            'FeatureSpec.selection, DerivedValueSpec.selection, SelectionSpec.entities) holds geometry object '
            'names, named selections of the same kind (expanded recursively) and reserved entity names.',
            'Domain entities: domain objects and all (the whole domain; build masks include the PML ring). Boundary '
            f'entities: the outer faces {face_texts} and domain objects, meaning their perimeter cells inside the '
            'object (the window edge is no perimeter). Point and line entities: point and line objects.',
            '3D models (DomainSpec.size_z set) add the outer faces z_lo (lower z face) and z_hi (upper z face); the '
            '2D domain kinds become prisms over their z0 / z1 params (null = full height), points and lines sit at '
            'their z param (null = mid-height), and box, cylinder_z and sphere are 3D-only kinds. Results stay 2D: '
            'fields are cut planes (the mid-height z cut unless an x / y / z plane is chosen).',
            'Objects consumed as inputs of a composite (union, intersection, difference, array) cannot be selected on '
            'their own.',
        ],
        'Values': [
            'Numeric fields and float / int params take a number or an expression string of parameters, e.g. '
            '"2*r_disk + 50*nm" (see expressions); params marked "may use x, y", "may use physics fields" or "may '
            'use t" may use those names too.',
            'Settings and params dicts (PhysicsSpec.settings, FeatureSpec.params, CouplingSpec.params, '
            'StudySpec.settings, GeometrySpec.params) hold values by param name: a missing key takes the param '
            'default, an unknown key is an error.',
            'A param is checked only while it is visible (visible when ...); an invisible param whose value fails '
            'keeps its default.',
            'Validation checks ranges of values that evaluate over the parameters. Material properties using x, y, z '
            'are range-checked on their cells at build time; those using physics fields at every update (for '
            'fields of an FDTD or transient physics at update 1, 2, 4, 8, ... and every 100th, so a run fails late, '
            'and a range left after the last such check is not reported), naming the '
            'material, property and the range found. Other expressions of x, y, z are checked for finite values '
            'only.',
        ],
        'Physics, couplings and study': [
            'Snapshots (StudySpec.stored_fields): time_dependent at the output interval plus the final state, '
            'stationary the final state, frequency_domain one per wavelength (labelled "lambda = <nm> nm"), '
            'mode_analysis one per mode (labelled "mode <k>: n_eff = <value>"), eigenfrequency one per resonance '
            '(labelled "resonance <k>: lambda = <nm> nm, Q = <value>"). A snapshot holds the native window '
            'cells (graded included, x_centers / y_centers set) up to 250,000 cells; a larger window is nearest-cell '
            'resampled on square pixels of max(finest cell, sqrt(window area / 250,000)) (native_shape = the window '
            'cells, x_centers None), which can skip features finer than a pixel. 3D models: the mid-height z cut, '
            'plus each StudySpec.stored_cuts plane (opt-in).',
            'A physics type lists the studies it supports; a study with a background study (the "Background study" '
            'line of its entry, e.g. frequency_domain: stationary) also accepts the physics that support only that '
            'one and solves them first as background physics. Heat is solved as a steady state in every study except '
            'time_dependent.',
            'StudySpec.sweep {"parameter": name, "values": [...]} builds and runs the model once per value with the '
            'parameter set to it ({"spacing": "linear" | "log", "lower", "upper", "count"} for a range instead); '
            'summary.json sweep holds the derived values and feature result digests per value.',
            'Physics type entries: "Default study" is the study the GUI applies when the physics is the first one. '
            'State fields are the solver state: expressions name them <physics>_<field>, or <field> when one physics '
            'alone provides it; field_transfer source_field. Output fields are what Results.field, study '
            'stored_fields (<physics>.<field>) and derived values read; the primary one is stored when stored_fields '
            'is []. Scalar outputs are numbers after a run; derived-value expressions name them <physics>_<name>.',
            'Source slots (physics type entries) are the coupling targets (CouplingSpec.target; field_transfer '
            'target_slot), with unit and meaning; a slot takes one overwriting coupling, or several additive ones '
            'whose contributions are summed (see the coupling types).',
            'Feature entries: "Result kind `<kind>`" means Results.result(physics name, feature name) returns a dict '
            'with the keys listed under Result kinds (topic results; a feature topic lists them inline). "Requires a '
            'uniform mesh": nonuniform and auto modes fall back to a uniform grid at the finest target (see mesh).',
        ],
        'Derived values': [
            'A derived value that fails for any reason, including running out of GPU memory, becomes NaN plus a '
            'warning (summary.json warnings); the run itself is not failed by it.',
            f"DerivedValueSpec.source 'physics.feature' reduces a 1D result ({', '.join(DERIVED_RESULT_KINDS)}) "
            'instead of fields. Its sample names (they shadow parameters): every numeric scalar and every numeric '
            'array with one entry per sample of the result dict by key (timeseries t, values; flux_spectrum '
            'wavelength, frequency, power, power_per_cw (FDTD only); xy x and extras such as wavelength, n_eff), '
            'plus the xy series by name and the numeric table columns by name (one sample per row, abscissa row = '
            '0, 1, ...). xy series and table column names are known only at run time: a missing one gives NaN plus '
            'a warning. far_field: the pattern at one wavelength (DerivedValueSpec.wavelength): angle [rad], power, '
            'power_norm (power over its peak), power_per_cw (when the result has it), and that wavelength and '
            'frequency as numbers.',
            'Far-field examples: beam direction {"name": "theta_peak", "source": "em.ff", "expression": "power", '
            '"operation": "argmax"}, beam width {"name": "theta_fwhm", "source": "em.ff", "expression": "power", '
            '"operation": "fwhm"}, value at an angle {"name": "p_up", "source": "em.ff", "expression": "power_norm", '
            '"operation": "at", "at": "pi/2"}; a line width {"name": "Q_line", "source": "em.trans", "expression": '
            '"power", "operation": "lorentz_fit", "fit": "q", "window": ["1545*nm", "1555*nm"]}.',
            'A sample name may equal an expression function name (e.g. the xy series abs of an FDFD probe): written '
            'without "(" it is the sample ("abs**2"), with "(" the function ("abs(phase)"); one expression may not '
            'use the same name both ways. Analytic curve series match series names as plain text, so any name '
            'works there.',
            'Example: {"name": "T_1550", "source": "em.trans", "expression": "power", "operation": "at", "at": '
            '"1550*nm", "reference": "T_film"}.',
        ],
        'Analytic curves': [
            'ModelSpec.analytic_curves never change the run: they are drawn over the results and compared with them '
            '(summary.json curve_error; with a tolerance, selftest --examples fails the model above it). The GUI '
            'evaluates them with numpy on its own thread over the '
            'parameters of the last built session (a sweep: its last value) on 501 points spanning the sampled '
            'range (timeseries: the own sample times when there are 501 to 20000, else 501 or 20000 points, so a '
            'ring-down does not alias; fft: the FFT frequency bins; sweep: 201 points over the swept values) and '
            'draws them dashed in the colour of the simulated curve they belong to; a failing curve is reported in '
            'the Results panel message line.',
            'Sweep tab: each numeric column of a per-value table result is offered as bands vs the swept parameter '
            "('<column>_<k>', sorted per value as Results.sweep_xy); the per-value result overlays also draw the "
            'curves targeting that result, evaluated at each swept value. Curves & tables: a table result is also '
            'plotted versus a chosen numeric column (x_column of its curves). Headless figures (figures=True) draw '
            "feature curves except series 'ratio' and 'fft' (no FFT figure; a flux spectrum only series '' and "
            "'power', its one panel), a table's curves into "
            'figures/<key>_vs_<x_column or row>.png and line curves into figures/curve_<name>.png.',
            'Examples: {"name": "T_fp", "target": "em.trans", "expression": "1 / (1 + F*sin(2*pi*n_f*d_f/'
            'wavelength)**2)", "series": "ratio"}; {"name": "T_line", "target": "line", "expression": "T0 + '
            'q*s*(L - s)/(2*k_s)", "line": [0, "H/2", "L", "H/2"], "field": "heat.T"}; {"name": "R_sweep", '
            '"target": "sweep", "expression": "R0*(1 + alpha*w)", "series": "R"}; {"name": "ring", "target": '
            '"em.probe", "expression": "A*exp(-pi*f0*t/Q)*cos(2*pi*f0*t)"}; {"name": "ring_fft", "target": '
            '"em.probe", "expression": "A/(2*sqrt((2*pi*(frequency - f0))**2 + (pi*f0/Q)**2))", "series": "fft"}; '
            '{"name": "band_2", "target": "sweep", "expression": "c0*k_b/(2*pi*n_eff)", "series": "frequency_2"}; '
            '{"name": "Q_fit", "target": "em.res", "expression": "Q0", "x_column": "frequency", "series": "Q"}; '
            '{"name": "mode_2", "target": "line", "expression": "A*sin(2*pi*s/L)", "line": [0, "H/2", "L", "H/2"], '
            '"field": "em.ez", "snapshot": "resonance 2"}; {"name": "T_top", "target": "line", "expression": "T0", '
            '"line": [0, 0, "L", 0], "field": "heat.T", "plane": "h_top"}.',
        ],
        'Derived curves': [
            'ModelSpec.derived_curves are evaluated with numpy after the run (GUI: on its own thread, re-evaluated '
            'when edited, no rerun) over the final feature results and the parameters of the last built session (a '
            'sweep: its last value). A failing curve (missing result or sample, other abscissa, expression error) '
            'is left out of its plot and reported: GUI message line, summary.json derived_curves {name: {plot, '
            'error}}.',
            'Example (heterodyne signal over shot noise, P_LO cancels: eta S / (h nu), S scaled from 1 V/m to a '
            'beam power): {"name": "snr_trn", "inputs": {"S": "heat.top_x:psd"}, "expression": '
            '"eta*S*E0sq/(h_const*c_const/lambda0)", "plot": "snr", "label": "TRN", "unit": "1", "log_x": true, '
            '"log_y": true} with E0sq a parameter.',
            'Calculus example (a capacitance dQ/dV from an xy result whose x is a voltage and whose series Q is a '
            'charge, against a constant): {"name": "C_V", "inputs": {"Q": "<physics>.<feature>:Q"}, "expression": '
            '"gradient(Q)", "unit": "F/m", "reference": "eps_ox*eps0_const/t_ox", "tolerance": 0.05}; a derived '
            'value with source "C_V" (expression "C_V") then reduces or fits the curve like a 1D result.',
        ],
        'Validation issues': [
            f"Levels: {', '.join(ISSUE_LEVELS)}; errors block build and run. A build error becomes one issue at the "
            'node path of the failing build step.',
            'Node paths: ["parameters"] (each message names the parameter), ["geometry", name] (the domain size at '
            '["geometry"]), ["selection", name], ["material", name], ["physics", name], ["feature", physics, '
            'feature], ["coupling", name], ["mesh"], ["study"], ["derived", name], ["curve", name], ["model"] '
            '(invalid file content '
            'and failures not tied to one node).',
            f'Reports add node, the label of the path: {label_examples}.',
        ],
    }


def _expressions_section(physics_types):
    return {
        'about': 'Numeric values are numbers or expression strings, evaluated by a safe evaluator. Which names an '
                 'expression may use depends on where it sits (Names available where).',
        'syntax': list(_EXPRESSION_SYNTAX),
        'functions': [{'name': name, 'min_args': FUNCTION_ARITY[name][0], 'max_args': FUNCTION_ARITY[name][1],
                       'description': FUNCTION_DESCRIPTIONS[name]} for name in sorted(FUNCTION_ARITY)],
        'constants': [{'name': name, 'value': _plain(value), 'description': CONSTANT_DESCRIPTIONS[name]}
                      for name, value in BUILTIN_CONSTANTS.items()],
        'reserved_names': sorted(RESERVED_NAMES),
        'coordinate_names': sorted(COORDINATE_NAMES),
        'availability': list(_NAME_AVAILABILITY),
        'physics_fields': {physics_type.name: {'state': [info.name for info in physics_type.state_fields],
                                               'output': [info.name for info in physics_type.output_fields]}
                           for physics_type in physics_types},
    }


def _geometry_section():
    entity_kinds = {}
    for kind in GEOMETRY_KINDS.values():
        entity_kinds.setdefault(kind.entity_kind, []).append(kind.name)
    kind_listing = '; '.join(f"{entity_kind}: {', '.join(names)}" for entity_kind, names in entity_kinds.items())
    return {
        'about': 'ModelSpec.geometry objects are named shapes in m; selections, materials, features and mesh '
                 'refinements refer to them by name.',
        'notes': [f'Geometry kinds by entity kind: {kind_listing}.', *_GEOMETRY_NOTES],
        'kinds': {name: {'name': kind.name, 'label': kind.label, 'entity_kind': kind.entity_kind,
                         'composite': kind.composite, 'description': kind.description,
                         'params': [_param_entry(param) for param in kind.params]}
                  for name, kind in GEOMETRY_KINDS.items()},
    }


def _materials_section(registry):
    from ..physics.em.definition import DISPERSION_POLE_PARAMS   # lazy: physics.em imports open_sol.materials
    vocabulary = {}   # each type's own properties and the arrays they write there (eps_r differs between EM types)
    for physics_type in _sorted_by_name(registry.physics_types()):
        if physics_type.material_properties:
            vocabulary[physics_type.name] = {param.name: list(param.expands_to)
                                             for param in physics_type.material_properties}
    return {
        'about': 'Materials assign properties to domain entities: a library material, explicit properties of the '
                 'present physics types and EM dispersion poles.',
        'notes': list(_MATERIAL_NOTES),
        'library': {name: {'name': name, 'description': MATERIAL_LIBRARY[name].description,
                           'properties': _plain(MATERIAL_LIBRARY[name].model_properties()),
                           'dispersion': _plain(MATERIAL_LIBRARY[name].dispersion_models())}
                    for name in sorted(MATERIAL_LIBRARY)},
        'property_vocabulary': vocabulary,
        'dispersion_models': {model: [_param_entry(param) for param in params]
                              for model, params in DISPERSION_POLE_PARAMS.items()},
    }


def _mesh_section(physics_types):
    uniform_features = [f'{physics_type.name}.{feature.name}' for physics_type in physics_types
                        for feature in physics_type.features if feature.requires_uniform_mesh]
    on_triangles, on_tetrahedra = ([physics_type.name for physics_type in physics_types if kind in physics_type.grids]
                                   for kind in ('triangles', 'tetrahedra'))
    return {
        'about': 'ModelSpec.meshes sets the cell sizes, one MeshSpec per mesh; each physics runs on one '
                 '(PhysicsSpec.mesh, "" = the first). check.json reports the resulting grids (cells, dx range, dt, '
                 'PML cells) before any GPU work.',
        'notes': [
            'Per-physics meshes (S7.2): every mesh spans the same domain; its padding (PML, mirror ghosts) and auto '
            'sizes come from the physics on it only. One mesh is required for: all EM physics (family em), physics '
            'joined by a coupling whose type does not transfer across meshes, a feature naming another physics, and '
            'every physics of a thermal_noise, optomechanics or nonlinear_frequency_domain study (validation errors). '
            'Cross-mesh couplings interpolate fields (linear between cell centres, edge-extended into padding) and '
            'move densities conservatively (overlap averages: window totals kept); a state-dependent material '
            'property reads the fields of a physics on another mesh interpolated. A mesh no physics uses is a '
            'warning and is not built. Fields and feature results live on their physics\' own mesh (FieldData / '
            'saved fields name it); a derived value is evaluated on the mesh of the first physics (model order) its '
            'expression names, the fields of the others interpolated onto it (validate lists them as info).',
            'Mesh kind triangles (S7.5, 2D Cartesian models): free triangles following curved boundaries for the '
            'physics, features and solvers that declare them (the Mesh kinds line of each type: '
            f"{', '.join(on_triangles) or 'none registered'}; some only in one study, validation names the limits). "
            'Sizes as COMSOL Free '
            'Triangular: dx = maximum element size, max_ratio = maximum growth rate (null: 1.3), refinements = '
            'maximum sizes in / along / at their objects, h_min (null: dx / 100), curvature_factor 0.3 along circle '
            'and ellipse edges; auto: the wavelength rule with points_per_wavelength (null: 5), else the '
            'physics-controlled rule; the absorbing ring has pml_layers structured layers over the PML cells x dx.',
            'Mesh kind tetrahedra (3D Cartesian models): tetrahedra for '
            f"{', '.join(on_tetrahedra) or 'none registered'} (em_fdfd_3d: its eigenfrequency study, Nedelec edge "
            "elements by the discretization setting). Sizes as triangles; mesher 'gmsh' follows spheres, boxes, z "
            'cylinders and z-prism circles / rectangles (any axis for rectangles), composite inputs and array copies '
            'included, other '
            "objects are tagged per element; 'builtin' splits the bricks of a rectilinear grid into 6 tetrahedra "
            '(no slivers at any aspect ratio), equal cells <= dx, or graded by the tensor rules (refinements with '
            'per-axis sizes, axes, face_sizes as bands) with flat object faces as element faces; curved objects '
            "are a staircase. sweep_axis 'z' (either mesher) meshes the x-y footprint with triangles and stacks them "
            'in z layers graded by the z tensor rules (object top / bottom faces are layer planes), each prism split '
            'into 3 tetrahedra: thin plates and films with a few layers across. '
            'The PML is a shell of pml_cells x dx outside every cpml face meshed like the window. Fields '
            'are sampled on a uniform lattice over the window (cut planes, profiles, volumes); derived values of fields '
            'reduce degree-4 quadrature points of the window (or selected) elements (not yet: interface selections, '
            'max / min over mirror images, fields of physics on other meshes).',
            'No physics boundary configuration -> no PML. EM CPML cells (pml_cells per cpml face; each PML cell is dx '
            'wide on a uniform grid; on a graded one they grow by max_ratio per cell from the window edge cell up to '
            'the base size) are added outside the domain, so the physical domain keeps its size and the grid grows.',
            'Several physics may define the boundary configuration when they agree (equal face types, CPML and Bloch '
            'settings); different ones are a validation error. E_z and H_z pec / pmc walls are half a cell apart '
            '(E_z pec at the outermost cell centre, H_z pec on its inner face); symmetric / antisymmetric mirror '
            'planes (E mirror-even / odd, PMC / PEC planes) coincide.',
            'A uniform mesh has ceil(size / dx) cells per axis, so the meshed window (Domain.physical_extent) can be '
            'one cell larger than size_x / size_y.',
            'nonuniform and auto fall back to a uniform grid when no refinement is finer than the base size, or at '
            'the finest target when an enabled feature requires a uniform mesh '
            f"({', '.join(uniform_features) or 'none registered'}); the note in check.json mesh says why.",
            'Refinements apply in nonuniform and auto modes only (uniform mode ignores them): each refines the '
            'columns and rows spanned by a geometry object plus its margin to its dx; rectangle edges inside the '
            'domain become cell edges (no staircase), so a gap between two such edges narrower than dx (but >= '
            '0.25 dx) gets cells of its width, which can set the time step; auto mode adds one refinement per '
            'material object by its refractive index. dx_x / dx_y / dx_z refine only some axes (anisotropic); '
            'MeshSpec.axes adds a per-axis max_ratio, coordinate bands and custom edges or uniform / geometric / '
            'tanh segments (they replace that axis, PML padding as usual); outer_dx grows the cells beyond the '
            'refined core. The RefinementSpec and MeshAxisSpec fields are listed under spec.',
            'Time step with EM: dt = courant_factor * dx / (c * sqrt(2)) on a uniform mesh (3D models: c * '
            'sqrt(3), the 3D limit; on a graded mesh the limit of the finest cells); the merged courant factor is the smallest declared (default 0.99). '
            'check.json study shows dt_estimate and n_steps.',
        ],
        'limits': {'max_cells': MAX_MESH_CELLS, 'max_cells_per_axis': MAX_AXIS_CELLS},
        'params': [_param_entry(param) for param in MESH_PARAMS],
    }


def _solvers_section(registry, physics_types):
    return {
        'about': 'A physics hands its numerical jobs (roles) to solvers: ModelSpec.solvers holds solver nodes '
                 '(SolverSpec: name, type, settings), PhysicsSpec.solvers {role: node name} assigns them; a role left '
                 'out is Automatic (the role\'s default type at its default settings; SolverDefaultsSpec.precision, '
                 'when not auto, replaces the precision of every solver that offers it and sets none). check.json '
                 'solvers shows what every role resolves to, summary.json solvers what ran (type, node, backend, '
                 'precision, settings, counters): Automatic depends on the machine (backend auto, sparse_direct '
                 'back_end auto), so the record is the reference for comparing runs.',
        'notes': [
            'Roles: ' + '; '.join(f'{role} ({text})' for role, text in SOLVER_ROLES.items()) + '. A role is used only '
            'in the studies listed for it (all studies when none are listed); an assignment in another study is a '
            'warning and ignored.',
            'Validation: an unknown node or role is an error at the physics; a node of the wrong role, operator '
            'class, backend, grid kind or precision (solver_misfit) is an error naming the fitting choices; a node '
            'no physics uses is a warning at the node; the node settings are checked like physics settings.',
            'Feature precisions: a feature whose entry lists "Stepper precisions ... only" (e.g. resonance_monitor, '
            'the em_fdtd_3d plane wave and mode ports) works only when the stepper role of its physics resolves to '
            'one of them; an enabled one is a validation error at the feature otherwise (e.g. with half_storage).',
            'Solver knobs are never physics settings: e.g. the CG precision / tolerance / max_iterations of heat, '
            'electric_currents, electrostatics, magnetostatics, solid_mechanics and solid_mechanics_3d are the '
            'settings of a pcg node assigned as their linear solver ({"name": "cg", "type": "pcg", "settings": '
            '{"tolerance": 1e-12}} in solvers, {"linear": "cg"} in the physics solvers).',
            'One node may serve several physics; the build creates one runtime per (physics, role).',
            'Opt-in roles: line_block fills the linear role of em_fdfd_ez / em_fdfd_hz only (its shifted role: heat '
            'thermal noise, em_fdfd_bor, em_modes); a line_block node assigned as the linear solver of any other physics is a '
            'misfit.',
            'sparse_direct back_end auto: a role\'s auto_back_end_by_grid {grid kind: back end} (pressure_acoustics on '
            'triangles / tetrahedra, membrane on triangles: cudss_mixed) replaces it on the gpu backend when '
            'nvmath-python is installed, for Automatic and for nodes that leave back_end auto; else auto is host '
            'SuperLU (scipy_splu).',
        ],
        'types': {solver_type.name: {
            'name': solver_type.name, 'label': solver_type.label, 'description': solver_type.description,
            'roles': list(solver_type.roles), 'operators': list(solver_type.operators),
            'backends': list(solver_type.backends), 'precisions': list(solver_type.precisions),
            'grids': list(solver_type.grids), 'params': [_param_entry(param) for param in solver_type.params]}
            for solver_type in _sorted_by_name(registry.solver_types())},
        'physics_roles': {physics_type.name: {role: {
            'operator': solver_role.operator, 'automatic': solver_role.default_type,
            'automatic_settings': _plain(solver_role.default_settings), 'studies': list(solver_role.studies),
            'automatic_by_backend': {backend: {'type': type_name, 'settings': _plain(settings)}
                                     for backend, (type_name, settings) in solver_role.backend_defaults.items()},
            'automatic_by_grid': {grid: {'type': type_name, 'settings': _plain(settings)}
                                  for grid, (type_name, settings) in solver_role.grid_defaults.items()},
            'auto_back_end_by_grid': dict(solver_role.auto_back_ends),
            'precisions': list(solver_role.precisions),
            'setting_choices': {name: list(values) for name, values in solver_role.setting_choices.items()}}
            for role, solver_role in physics_type.solver_roles.items()}
            for physics_type in physics_types if physics_type.solver_roles},
    }


def _physics_entry(physics_type):
    primary = primary_field_info(physics_type.output_fields)
    return {
        'name': physics_type.name, 'label': physics_type.label, 'description': physics_type.description,
        'default_instance_name': physics_type.default_instance_name,
        'supported_studies': list(physics_type.supported_studies),
        'default_study': _plain(physics_type.default_study),
        'backends': list(physics_type.backends),
        'grids': list(physics_type.grids),
        'indicator_fields': list(physics_type.indicator_fields) or ([primary.name] if primary is not None else []),
        'uses_dispersion': physics_type.uses_dispersion,
        'settings': [_param_entry(param) for param in physics_type.settings],
        'material_properties': [_param_entry(param) for param in physics_type.material_properties],
        'state_fields': [_field_entry(info) for info in physics_type.state_fields],
        'output_fields': [_field_entry(info) for info in physics_type.output_fields],
        'primary_field': None if primary is None else primary.name,
        'scalar_outputs': physics_type.scalar_outputs_help or None,
        'source_slots': [{'name': slot, 'unit': unit, 'meaning': meaning}
                         for slot, (unit, meaning) in physics_type.source_slots.items()],
        'features': {feature.name: _feature_entry(feature) for feature in physics_type.features},
    }


def _field_entry(info):
    return {'name': info.name, 'label': info.label, 'unit': info.unit, 'requires_study': info.requires_study}


def _feature_entry(feature):
    return {
        'name': feature.name, 'label': feature.label, 'category': feature.category,
        'selection_kind': feature.selection_kind, 'object_kinds': list(feature.object_kinds),
        'min_entities': feature.min_entities, 'max_entities': feature.max_entities,
        'allow_outer_faces': feature.allow_outer_faces, 'allow_interior': feature.allow_interior,
        'requires_uniform_mesh': feature.requires_uniform_mesh, 'result_kind': feature.result_kind,
        'precisions': list(restricted_feature_precisions(feature)) or None,   # None = every stepper precision
        'grids': list(feature.grids),
        'description': feature.description, 'params': [_param_entry(param) for param in feature.params],
    }


def _coupling_entry(coupling_type):
    return {'name': coupling_type.name, 'label': coupling_type.label,
            'source_type': _plain(coupling_type.source_type), 'target_type': _plain(coupling_type.target_type),
            'mesh_transfer': coupling_type.mesh_transfer, 'value_kind': coupling_type.value_kind,
            'description': coupling_type.description,
            'params': [_param_entry(param) for param in coupling_type.params]}


def _study_entry(study_type):
    return {'name': study_type.name, 'label': study_type.label, 'description': study_type.description,
            'requires_time_stepping': study_type.requires_time_stepping,
            'background_study': study_type.background_study,
            'solution_feedback': study_type.solution_feedback,
            'coupled_points': study_type.coupled_points,
            'supports_adaptation': study_type.supports_adaptation,
            'params': [_param_entry(param) for param in study_type.params]}


def _templates_section(registry):
    return {
        'about': 'Worked examples: every template validates without errors (selftest templates_validate), and '
                 'modifying the closest one is the most reliable way to write a valid model. api.template(key) '
                 'returns one as a ModelSpec, python -m open_sol template KEY FILE writes its model file.',
        'entries': {entry.key: {'key': entry.key, 'label': entry.label, 'description': entry.description,
                                'model': _plain(_template_model(entry).to_dict())}
                    for entry in template_entries(registry)},
    }


def _template_model(entry):
    """ModelSpec of a TemplateEntry's factory (make_template's check, without its per-key entry lookup)."""
    spec = entry.factory()
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"template '{entry.key}': factory returned {type(spec).__name__}, not a ModelSpec")
    return spec


def _results_section(registry, physics_types, with_selftests=True):
    producers = {kind: [] for kind in RESULT_KINDS}
    for physics_type in physics_types:
        for feature in physics_type.features:
            if feature.result_kind in producers:
                producers[feature.result_kind].append(f'{physics_type.name}.{feature.name}')
    return {
        'about': 'The user runs the commands; every command writes report files, which agents read instead of '
                 'terminal output: check.json, summary.json, log.txt and figures/*.png. Arrays stay in results.npz '
                 'for further processing.',
        'notes': {heading: list(lines) for heading, lines in _RESULTS_NOTES},
        'result_kinds': {kind: {'produced_by': producers[kind],
                                'keys': [{'key': key, 'meaning': meaning} for key, meaning in RESULT_KIND_DOCS[kind]]}
                         for kind in RESULT_KINDS},
        'selftests': [{'name': case.name, 'label': case.label, 'description': case.description,
                       'max_wall_time': _plain(case.max_wall_time), 'heavy': case.heavy,
                       'variants': [{'name': variant.name, 'backend': variant.backend, 'precision': variant.precision,
                                     'roles': list(variant.roles), 'tolerance': _plain(variant.tolerance)}
                                    for variant in case.variants]}
                      for case in (_sorted_by_name(registry.selftests()) if with_selftests else [])],
    }


def _param_entry(param):
    """Param -> dict of its documented attributes (values JSON-able)."""
    return {'name': param.name, 'kind': param.kind, 'label': param.label, 'unit': param.unit,
            'default': _plain(param.default), 'min': _plain(param.min), 'max': _plain(param.max),
            'min_exclusive': param.min_exclusive, 'choices': _plain(param.choices), 'optional': param.optional,
            'help': param.help, 'field_dependent': param.field_dependent, 'state_dependent': param.state_dependent,
            'time_dependent': param.time_dependent,
            'visible_if': [[controller, _plain(accepted)] for controller, accepted in param.visible_if],
            'expands_to': [str(name) for name in param.expands_to], 'ref_kind': param.ref_kind,
            'columns': [[name, unit] for name, unit in param.columns], 'increasing': param.increasing,
            'space_units': [[space, unit] for space, unit in param.space_units]}


def _plain(value):
    """JSON-able copy of a documented value: tuples -> lists, numbers -> int / float, non-finite floats -> 'nan',
    'inf', '-inf' (as in the report files); TypeError for anything else."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number):
            return number
        return 'nan' if math.isnan(number) else ('inf' if number > 0 else '-inf')
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    raise TypeError(f'describe: a value of type {type(value).__name__} cannot be documented')


def _sorted_by_name(items):
    return sorted(items, key=lambda item: item.name)


# ---- markdown ----
def _markdown(blocks):
    """Blocks of lines separated by one blank line; every line ends with a newline, the text with exactly one."""
    lines = []
    for block in blocks:
        if not block:
            continue
        if lines:
            lines.append('')
        lines.extend(block)
    return '\n'.join(lines) + '\n'


def _shared_texts(reference):
    """{text: key} of the physics / coupling / study texts (descriptions, param help of each _param_groups entry) of
    >= _SHARED_TEXT_MIN_CHARS chars occurring >= 2 times; key = first owner path in file order, made unique."""
    owners = {}   # text -> owner paths in file order

    def add(path, text):
        if text and len(text) >= _SHARED_TEXT_MIN_CHARS:
            owners.setdefault(text, []).append(path)

    def add_params(owner, params):
        for group in _param_groups(params):
            add(f"{owner}.{group[0]['name']}", group[0]['help'])

    for name, entry in reference['physics_types'].items():
        add(name, entry['description'])
        add_params(name, entry['settings'])
        add_params(name, entry['material_properties'])
        for feature in entry['features'].values():
            feature_path = f"{name}.{feature['name']}"
            add(feature_path, feature['description'])
            add_params(feature_path, feature['params'])
    for section in ('coupling_types', 'study_types'):
        for name, entry in reference[section].items():
            add(name, entry['description'])
            add_params(name, entry['params'])
    shared, used_keys = {}, set()
    for text, paths in owners.items():
        if len(paths) < 2:
            continue
        key, suffix = paths[0], 2
        while key in used_keys:   # e.g. a setting named like a feature of the same physics
            key, suffix = f'{paths[0]}#{suffix}', suffix + 1
        used_keys.add(key)
        shared[text] = key
    return shared


def _reference_file_text(reference, registry):
    """REFERENCE_FILE text: header, Contents, Shared help texts and the body with the templates as an index."""
    shared = _shared_texts(reference)
    body = _view_blocks(reference, shared, template_index=True)
    header = _header_blocks(_topic_groups(registry))
    return _markdown(header + _contents_blocks(REFERENCE_FILE, header, body, bool(shared)) + body)


def _templates_file_text(reference):
    """TEMPLATES_FILE text: its own header and Contents, then the 'templates' topic text (the JSON per template)."""
    header = [
        ['# open_sol templates'],
        [f'Regenerate with: {REGENERATE_COMMAND} (writes this file and {REFERENCE_FILE})'],
        [f"Generated by describe() from the templates (the body below is describe('templates')); selftest fails "
         f'while this file differs from it. Types, settings and help texts are in {REFERENCE_FILE}, whose '
         'Templates section lists every template with the physics, feature, coupling, solver and study types it '
         'uses.'],
    ]
    body = _view_blocks({'templates': reference['templates']})
    return _markdown(header + _contents_blocks(TEMPLATES_FILE, header, body) + body)


def _header_blocks(topic_groups):
    rules = ', '.join(text for text, names in topic_groups if names)
    return [
        ['# open_sol API reference'],
        [f'Regenerate with: {REGENERATE_COMMAND} (also writes {TEMPLATES_FILE})'],
        ['Generated by describe() from the registry (physics, feature, coupling and study types), the model spec, '
         'the expression vocabulary, the material library and the templates; selftest fails while this file '
         f'differs from it. The template model JSON is in {TEMPLATES_FILE} beside this file.'],
        ['Workflow: write the closest template (Templates) to a model JSON file with python -m open_sol template KEY '
         'model.json, edit it, run python -m open_sol check model.json (check.json), then python -m open_sol run '
         'model.json --figures (summary.json, log.txt, figures/); --set NAME=EXPR changes a parameter for one check / '
         'run. Units are SI unless a unit says otherwise; every wavelength is a vacuum wavelength.'],
        ['More worked model files (examples/<physics area>/*.json, one folder per physics area plus multiphysics, '
         'many with an analytic reference; scripts in examples/em, examples/multiphysics and '
         'examples/headless_run.py) live in the examples folder beside the package (examples/README.md); they are '
         'not part of this reference.'],
        [f'Topics (describe(topic) or python -m open_sol describe <topic>): {rules}.'],
    ]


def _contents_blocks(file_name, header, body, has_shared_texts=False):
    """'## Contents': file line range of every '## ' section of header + contents + body (and of the '### '
    subsections of sections over _CONTENTS_SPLIT_LINES lines), for offset / limit reads of file_name."""
    shared_rule = (' A line "Help: shared `<key>`" stands for the text under "#### `<key>`" in Shared help texts '
                   '(Grep it); every occurrence of a repeated long text, the first included, points there.'
                   if has_shared_texts else '')
    body_lines = _markdown(body).split('\n')[:-1]
    headings = [(index, line.split(' ', 1)[0], line.lstrip('#').strip()) for index, line in enumerate(body_lines)
                if line.startswith(('## ', '### '))]
    entries, section_is_long = [], False
    for position, (start, level, title) in enumerate(headings):
        end = next((later for later, later_level, _ in headings[position + 1:] if len(later_level) <= len(level)),
                   len(body_lines) + 1) - 2   # last line before the blank line that precedes the next heading
        if level == '##':
            section_is_long = end - start + 1 > _CONTENTS_SPLIT_LINES
            entries.append(('', start, end, title))
        elif section_is_long:
            entries.append(('  ', start, end, title))

    def contents(line_offset):
        return [['## Contents'],
                [f"Line ranges in {file_name} (Read with offset / limit; Grep '^## ' lists the same headings). "
                 f'Sections over {_CONTENTS_SPLIT_LINES} lines also list their subsections.{shared_rule}'],
                [f'{indent}- lines {start + line_offset}-{end + line_offset}: {title}'
                 for indent, start, end, title in entries]]

    # body line k (0-based) sits at 1-based line n + 2 + k after the n prefix lines and one blank line
    prefix_line_count = len(_markdown(header + contents(0)).split('\n')) - 1
    return contents(prefix_line_count + 2)


def _view_blocks(view, shared=None, template_index=False):
    """Markdown blocks of a reference view (the whole reference or one topic), sections in topic order; shared
    ({text: key}, full reference only) prints those texts once in '## Shared help texts' before the physics;
    template_index prints one line per template instead of its JSON (REFERENCE_FILE)."""
    shared = shared or {}
    blocks = []
    for key, render in (('spec', _spec_blocks), ('expressions', _expressions_blocks),
                        ('geometry', _geometry_blocks), ('materials', _materials_blocks), ('mesh', _mesh_blocks),
                        ('solvers', _solvers_blocks)):
        if key in view:
            blocks += render(view[key])
    if shared:
        blocks += [['## Shared help texts'],
                   ['Long texts (param help, feature and type descriptions) that occur more than once in the '
                    'physics, coupling and study sections below; each occurrence reads "Help: shared `<key>`", key = '
                    'its first owner: <physics>.<param>, <physics>.<feature>, <physics>.<feature>.<param>, '
                    '<coupling or study type>.<param> or a type name.']]
        for text, key in shared.items():
            blocks += [[f'#### `{key}`'], [text]]
    for physics_name, entry in view.get('physics_types', {}).items():
        if 'name' in entry:
            blocks += _physics_blocks(entry, shared)
        else:   # feature topic: {'features': {feature name: entry}}
            for feature in entry['features'].values():
                blocks += _feature_blocks(physics_name, feature, shared)
    for entry in view.get('coupling_types', {}).values():
        blocks += _coupling_blocks(entry, shared)
    for entry in view.get('study_types', {}).values():
        blocks += _study_blocks(entry, shared)
    if 'templates' in view:
        blocks += (_template_index_blocks if template_index else _templates_blocks)(view['templates'])
    if 'results' in view:
        blocks += _results_blocks(view['results'])
    if _SELFTESTS_TOPIC in view:
        blocks += _selftests_blocks(view[_SELFTESTS_TOPIC])
    return blocks


def _spec_blocks(section):
    blocks = [['## Model spec'], [section['about']], ['### Spec dataclasses']]
    for entry in section['dataclasses']:
        blocks.append([f"#### `{entry['name']}` - {entry['where']}"])
        blocks.append([_spec_field_line(field_entry) for field_entry in entry['fields']])
    blocks.append(['### Param kinds'])
    blocks.append([f'- `{kind}`: {text}' for kind, text in section['param_kinds'].items()])
    return blocks + _notes_blocks(section['notes'])


def _expressions_blocks(section):
    blocks = [['## Expressions'], [section['about']], ['### Syntax'], _bullets(section['syntax']),
              ['### Functions'],
              [f"- {entry['description']} [{arity_text(entry['min_args'], entry['max_args'])}]"
               for entry in section['functions']],
              ['### Constants'],
              [f"- `{entry['name']}` = {_value_text(entry['value'])}: {entry['description']}"
               for entry in section['constants']],
              ['### Reserved names'],
              [f"Not usable as parameter names: {', '.join(section['reserved_names'])}. Time and "
               f"coordinates: {', '.join(section['coordinate_names'])}; value is the source field in coupling expressions."],
              ['### Names available where'], _bullets(section['availability']),
              ['### Physics field names by physics type']]
    blocks.append([f"- `{name}`: state fields {_names_text(names['state'])}; output fields "
                   f"{_names_text(names['output'])}" for name, names in section['physics_fields'].items()]
                  or ['none'])
    return blocks


def _geometry_blocks(section):
    blocks = [['## Geometry'], [section['about']], _bullets(section['notes'])]
    for entry in section['kinds'].values():
        blocks += [[f"### Geometry kind `{entry['name']}` - {entry['label']}"], _paragraph(entry['description']),
                   [f"- GeometrySpec.kind `{entry['name']}`; entity kind {entry['entity_kind']}; composite: "
                    f"{_yes_no(entry['composite'])}"],
                   ['Params (GeometrySpec.params):'] + _params_lines(entry['params'])]
    return blocks


def _materials_blocks(section):
    blocks = [['## Materials'], [section['about']], _bullets(section['notes']), ['### Library materials']]
    for entry in section['library'].values():
        dispersion = _value_text(entry['dispersion']) if entry['dispersion'] else 'none'
        blocks += [[f"#### Library material `{entry['name']}`"], _paragraph(entry['description']),
                   [f"- Properties: {_value_text(entry['properties'])}", f'- Dispersion poles: {dispersion}']]
    blocks.append(['### Material property names by physics type (MaterialGrid arrays each writes)'])
    blocks.append([f'- `{type_name}`: ' + ', '.join(f"{name} ({', '.join(arrays) or 'table, no array'})"
                                                   for name, arrays in properties.items())
                   for type_name, properties in section['property_vocabulary'].items()] or ['none'])
    blocks.append(['### Dispersion pole models (MaterialSpec.dispersion items)'])
    for model, params in section['dispersion_models'].items():
        blocks += [[f'#### Pole model `{model}`'], _params_lines(params)]
    return blocks


def _mesh_blocks(section):
    limits = section['limits']
    limit_line = (f"Limits: {limits['max_cells']:,} cells and {limits['max_cells_per_axis']:,} cells per axis, PML "
                  'cells included; a larger mesh fails before any meshing.')
    return [['## Mesh'], [section['about']], _bullets(section['notes'] + [limit_line]),
            ['### Mesh settings (MeshSpec fields)'], _params_lines(section['params'])]


def _solvers_blocks(section):
    blocks = [['## Solvers'], [section['about']], _bullets(section['notes']), ['### Solver types']]
    for entry in section['types'].values():
        blocks += [[f"#### Solver type `{entry['name']}` - {entry['label']}"], _paragraph(entry['description']),
                   [f"- SolverSpec.type `{entry['name']}`; roles {_names_text(entry['roles'])}; operators "
                    f"{_names_text(entry['operators'])}; backends {_names_text(entry['backends'])}; precisions "
                    f"{_names_text(entry['precisions'])}; grids {_names_text(entry['grids'])}"],
                   ['Params (SolverSpec.settings):'] + _params_lines(entry['params'])]
    blocks.append(['### Roles by physics type (Automatic = the default type at its default settings)'])
    lines = []
    for physics_name, roles in section['physics_roles'].items():
        for role, entry in roles.items():
            studies = _names_text(entry['studies']) if entry['studies'] else 'every study'
            restricted = ''.join(f"; {name} {_names_text(values)} only"
                                 for name, values in entry['setting_choices'].items())
            lines.append(f"- `{physics_name}` {role}: operator {entry['operator']}, Automatic `{entry['automatic']}`, "
                         f"{studies}, precisions {_names_text(entry['precisions'])}{restricted}")
    blocks.append(lines or ['none'])
    return blocks


def _physics_blocks(entry, shared):
    """Physics type entry (the lead-ins of its fields, slots and outputs are explained once in the spec notes)."""
    name = entry['name']
    blocks = [[f"## Physics type `{name}` - {entry['label']}"], _text_lines(entry['description'], shared),
              [f"- PhysicsSpec.type `{name}`; default instance name `{entry['default_instance_name']}`",
               f"- Supported studies: {_names_text(entry['supported_studies'])}",
               f"- Default study: {_value_text(entry['default_study'])}",
               f"- Backends (SolverDefaultsSpec.backend): {_names_text(entry['backends'])}",
               f"- Mesh kinds (MeshSpec.kind): {_names_text(entry['grids'])}; refinement indicator fields (when "
               f"its solver has no error indicator of its own): {_names_text(entry['indicator_fields'])}",
               f"- Dispersion poles (MaterialSpec.dispersion): {'used' if entry['uses_dispersion'] else 'not used'}",
               f"- Features (topics `{name}.<feature>`): {_names_text(entry['features'])}"],
              ['### Settings (PhysicsSpec.settings)'], _params_lines(entry['settings'], shared),
              ['### Material properties (MaterialSpec.properties)'],
              _params_lines(entry['material_properties'], shared),
              ['### Fields'],
              _fields_lines('State fields:', entry['state_fields']),
              _fields_lines(f"Output fields (primary {entry['primary_field'] or 'none'}):", entry['output_fields'])]
    if entry['scalar_outputs']:
        blocks.append([f"Scalar outputs: {entry['scalar_outputs']}"])
    blocks += [['### Source slots'],
               [f"- `{slot['name']}` ({slot['unit']}): {slot['meaning']}" for slot in entry['source_slots']]
               or ['- none']]
    for feature in entry['features'].values():
        blocks += _feature_blocks(name, feature, shared)
    return blocks


def _feature_blocks(physics_name, entry, shared):
    facts = [f"- FeatureSpec.type `{entry['name']}`; category {entry['category']}",
             f'- Selection (FeatureSpec.selection): {_selection_text(entry)}']
    if entry['requires_uniform_mesh']:
        facts.append('- Requires a uniform mesh (see mesh)')
    if entry['grids'] != ['tensor']:
        facts.append(f"- Mesh kinds (MeshSpec.kind): {_names_text(entry['grids'])}")
    if entry['precisions']:
        facts.append(f"- Stepper precisions {_names_text(entry['precisions'])} only (see solvers): another "
                     "precision of the physics' resolved stepper solver is a validation error at the feature")
    if entry['result_kind'] is None:
        facts.append('- Result: none')
    elif 'result_keys' in entry:   # feature topic: the keys inline
        facts.append(f"- Result kind `{entry['result_kind']}`: Results.result(physics name, feature name) returns "
                     'a dict with these keys:')
        facts += [f"  - `{item['key']}`: {item['meaning']}" for item in entry['result_keys']]
    else:
        facts.append(f"- Result kind `{entry['result_kind']}`")
    return [[f"### Feature `{physics_name}.{entry['name']}` - {entry['label']}"],
            _text_lines(entry['description'], shared),
            facts, ['Params (FeatureSpec.params):'] + _params_lines(entry['params'], shared)]


def _coupling_blocks(entry, shared):
    return [[f"## Coupling type `{entry['name']}` - {entry['label']}"], _text_lines(entry['description'], shared),
            [f"- CouplingSpec.type `{entry['name']}`; source physics type {_types_text(entry['source_type'])}; "
             f"target physics type {_types_text(entry['target_type'])}",
             '- Across meshes (source and target on different meshes): '
             + ('not allowed (one mesh)' if entry['mesh_transfer'] is None
                else 'the source field is interpolated onto the target mesh' if entry['mesh_transfer'] == 'field'
                else f"its result moves to the target mesh ({entry['value_kind']})")],
            ['Params (CouplingSpec.params):'] + _params_lines(entry['params'], shared)]


def _study_blocks(entry, shared):
    facts = [f"- StudySpec.type `{entry['name']}`; needs a physics that steps in time: "
             f"{_yes_no(entry['requires_time_stepping'])}"]
    if entry['background_study']:
        facts.append(f"- Background study: {entry['background_study']}")
    if entry['solution_feedback']:
        facts.append('- Solution feedback: couplings from a physics this study solves into background physics are '
                     'allowed (feedback rounds)')
    if entry['coupled_points']:
        facts.append('- Coupled points: couplings and material properties may read a physics this study solves; '
                     'each point of the study axis is then a stationary fixed point of every physics')
    if entry['supports_adaptation']:
        facts.append('- Mesh refinement levels: StudySpec.adaptation allowed')
    return [[f"## Study type `{entry['name']}` - {entry['label']}"], _text_lines(entry['description'], shared),
            facts, ['Params (StudySpec.settings):'] + _params_lines(entry['params'], shared)]


def _templates_blocks(section):
    blocks = [['## Templates'], [section['about']]]
    for entry in section['entries'].values():
        blocks += [[f"### `{entry['key']}` - {entry['label']}"], _paragraph(entry['description']),
                   ['```json', *_json_lines(entry['model']), '```']]
    return blocks


def _template_index_blocks(section):
    """REFERENCE_FILE's Templates: one line per template (key, label, description, the types it uses)."""
    return [['## Templates'], [section['about']],
            [f"The model JSON of each template is in {TEMPLATES_FILE} beside this file (its Contents gives each "
             "template's line range; also describe('templates')). One line per template: key - label: description, "
             'then the physics types, features (<physics>.<feature>), couplings, solver types and study it uses.'],
            [f"- `{entry['key']}` - {entry['label']}: {entry['description']} Uses {_template_uses(entry['model'])}."
             for entry in section['entries'].values()] or ['none']]


def _template_uses(model):
    """'physics a, b; features a.x; couplings c; solvers s; study t' of a plain model dict (empty groups left out)."""
    physics = model.get('physics', [])
    groups = (('physics', [item['type'] for item in physics]),
              ('features', [f"{item['type']}.{feature['type']}" for item in physics
                            for feature in item.get('features', [])]),
              ('couplings', [item['type'] for item in model.get('couplings', [])]),
              ('solvers', [item['type'] for item in model.get('solvers', [])]))
    parts = [f"{label} {', '.join(dict.fromkeys(names))}" for label, names in groups if names]
    return '; '.join(parts + [f"study {model['study']['type']}"])


def _json_lines(value, indent='', prefix='', suffix=''):
    """JSON lines of a plain value as the examples/*.json files: one line while it fits _JSON_LINE_WIDTH, else one
    key / item per line (recursively)."""
    text = json.dumps(value)
    if not (isinstance(value, (dict, list)) and value) or len(indent + prefix + text + suffix) <= _JSON_LINE_WIDTH:
        return [f'{indent}{prefix}{text}{suffix}']
    if isinstance(value, dict):
        items, opening, closing = [(f'{json.dumps(key)}: ', item) for key, item in value.items()], '{', '}'
    else:
        items, opening, closing = [('', item) for item in value], '[', ']'
    lines = [f'{indent}{prefix}{opening}']
    for position, (item_prefix, item) in enumerate(items):
        lines += _json_lines(item, indent + '  ', item_prefix, ',' if position < len(items) - 1 else '')
    return lines + [f'{indent}{closing}{suffix}']


def _results_blocks(section):
    blocks = [['## Results, reports and files'], [section['about']],
              [f"Registered selftest cases are not listed here: describe('{_SELFTESTS_TOPIC}') (python -m open_sol "
               f'describe {_SELFTESTS_TOPIC}) gives each name, label and what it checks, python -m open_sol selftest '
               '--list the names.']] + _notes_blocks(section['notes'])
    blocks.append(['### Result kinds (Results.result and Session.feature_result dicts)'])
    for kind, entry in section['result_kinds'].items():
        blocks += [[f'#### Result kind `{kind}`'], [f"Produced by: {_names_text(entry['produced_by'])}"],
                   [f"- `{item['key']}`: {item['meaning']}" for item in entry['keys']]]
    return blocks


def _selftests_blocks(cases):
    """Topic 'selftests': every registered case with its full description."""
    return [['## Registered selftest cases'],
            [f"- `{case['name']}` - {case['label']}: {case['description']} (at most "
             f"{_value_text(case['max_wall_time'])} s per run{'; heavy' if case['heavy'] else ''}"
             f"{_variants_text(case['variants'])})" for case in cases]
            or ['none']]


def _variants_text(variants):
    """'; variants name (backend, precision on roles, tolerance), ...' of a selftest entry, '' without variants."""
    parts = []
    for variant in variants:
        moves = [f"backend {variant['backend']}"] if variant['backend'] else []
        if variant['precision']:
            roles = _names_text(variant['roles']) if variant['roles'] else 'every role with a precision setting'
            moves.append(f"precision {variant['precision']} on {roles}")
        parts.append(f"{variant['name']} ({', '.join(moves)}, tolerance {_value_text(variant['tolerance'])})")
    return f"; variants {', '.join(parts)}" if parts else ''


def _notes_blocks(notes):
    """'### heading' + bullet list per (heading, lines) entry."""
    blocks = []
    for heading, lines in notes.items():
        blocks += [[f'### {heading}'], _bullets(lines)]
    return blocks


def _params_lines(params, shared=None):
    """Lines of a param list ('none' when empty); consecutive params with help that differ only in name and label
    share one entry (e.g. the four bc_* faces); a help text in shared ({text: key}) becomes a pointer line."""
    return [line for group in _param_groups(params) for line in _param_lines(group, shared or {})] or ['none']


def _param_groups(params):
    """Param dicts grouped into runs of consecutive params with help that share every fact but name and label."""
    groups = []
    for entry in params:
        if groups and entry['help'] and _shared_facts(groups[-1][0]) == _shared_facts(entry):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def _shared_facts(entry):
    """A param dict without its name and label (what params merged into one entry must share)."""
    return {key: value for key, value in entry.items() if key not in ('name', 'label')}


def _param_lines(group, shared):
    """'- `name`, ... (kind; unit; default; range; ...) - Label / ...' and the indented help line of params sharing
    every fact but name and label."""
    entry = group[0]
    facts = [_kind_text(entry['kind'], entry['ref_kind'])]
    if entry['columns']:
        facts.append('columns ' + ', '.join(f'{name} [{unit}]' if unit else name for name, unit in entry['columns']))
    if entry['increasing']:
        facts.append('>= 2 rows, every column strictly increasing')
    if entry['unit']:
        facts.append(f"unit {entry['unit']}" + ''.join(f', {unit} {space}' for space, unit in entry['space_units']))
    if entry['default'] is not None:
        facts.append(f"default {_value_text(entry['default'])}")
    elif not entry['optional']:
        facts.append('required')
    if entry['min'] is not None or entry['max'] is not None:
        facts.append(f'range {_range_text(entry)}')
    if entry['choices']:
        facts.append('choices ' + ' | '.join(_choice_text(choice) for choice in entry['choices']))
    if entry['optional']:
        facts.append('optional (null allowed)')
    if entry['field_dependent']:
        facts.append('may use x, y')
    if entry['state_dependent']:
        facts.append('may use physics fields')
    if entry['time_dependent']:
        facts.append('may use t')
    if entry['visible_if']:
        facts.append(f"visible when {_visible_text(entry['visible_if'])}")
    if entry['expands_to']:
        facts.append(f"writes arrays {', '.join(entry['expands_to'])}")
    names = ', '.join(f"`{item['name']}`" for item in group)
    line = f"- {names} ({'; '.join(facts)})"
    labels = [item['label'] for item in group if item['label']]
    if labels:
        line += f" - {' / '.join(labels)}"
    return [line] + _text_lines(entry['help'], shared, '  ')


def _visible_text(visible_if):
    """'a in (x) or any of b, c in (y)': controllers accepting the same values are listed once."""
    groups = []
    for controller, accepted in visible_if:
        group = next((group for group in groups if group[1] == accepted), None)
        if group is None:
            groups.append(([controller], accepted))
        else:
            group[0].append(controller)
    return ' or '.join(f"{'any of ' if len(controllers) > 1 else ''}{', '.join(controllers)} in "
                       f"({', '.join(_choice_text(value) for value in accepted)})" for controllers, accepted in groups)


def _kind_text(kind, ref_kind):
    """Param kind, with the node kind of a typed 'ref' / 'names' param."""
    if ref_kind and kind in ('ref', 'names'):
        return f"{kind}: {ref_kind} name{'s' if kind == 'names' else ''}"
    return kind


def _range_text(entry):
    """'[min, max]', '(min, inf)' ... of a param dict (min_exclusive -> open lower bound)."""
    low, high = entry['min'], entry['max']
    left = '(' if low is None or entry['min_exclusive'] else '['
    right = ')' if high is None else ']'
    low_text = '-inf' if low is None else _value_text(low)
    high_text = 'inf' if high is None else _value_text(high)
    return f'{left}{low_text}, {high_text}{right}'


def _spec_field_line(entry):
    status = 'required' if entry['required'] else f"default {_value_text(entry['default'])}"
    return f"- `{entry['name']}` ({entry['type']}; {status}): {entry['help']}"


def _fields_lines(lead, field_entries):
    lines = [lead]
    for entry in field_entries:
        unit = f" ({entry['unit']})" if entry['unit'] else ''
        study = f" [{entry['requires_study']} study only]" if entry['requires_study'] else ''
        lines.append(f"- `{entry['name']}` - {entry['label']}{unit}{study}")
    return lines if field_entries else lines + ['- none']


def _selection_text(entry):
    """Selection kind, allowed object kinds, entity count and (boundary) faces / perimeters of a feature dict."""
    kind = entry['selection_kind']
    if kind is None:
        return 'none (a selection is ignored with a warning)'
    parts = [f'{kind} entities']
    if entry['object_kinds']:
        parts.append(f"object kinds {', '.join(entry['object_kinds'])}")
    low, high = entry['min_entities'], entry['max_entities']
    if high is None:
        parts.append(f'count at least {low}')
    elif low == high:
        parts.append(f'count exactly {low}')
    else:
        parts.append(f'count {low} to {high}')
    if kind == 'boundary':
        parts.append(f"outer faces {'allowed' if entry['allow_outer_faces'] else 'not allowed'}")
        parts.append(f"object perimeters {'allowed' if entry['allow_interior'] else 'not allowed'}")
    return '; '.join(parts)


def _value_text(value):
    """One-line text of a plain value: JSON literals with floats as %.6g (deterministic)."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return '[' + ', '.join(_value_text(item) for item in value) + ']'
    return '{' + ', '.join(f'{json.dumps(key)}: {_value_text(item)}' for key, item in value.items()) + '}'


def _choice_text(value):
    """A choice / visible_if value: text as is, anything else as _value_text."""
    return value if isinstance(value, str) else _value_text(value)


def _names_text(names):
    return ', '.join(names) or 'none'


def _types_text(accepted):
    """Physics types a coupling role accepts (plain value): None -> 'any', a name as is, a list joined with ' or '."""
    if accepted is None:
        return 'any'
    return accepted if isinstance(accepted, str) else ' or '.join(accepted)


def _bullets(lines):
    return [f'- {line}' for line in lines]


def _paragraph(text):
    return [text] if text else []


def _text_lines(text, shared, indent=''):
    """[indent + text], or the 'Help: shared `<key>`' pointer line when shared ({text: key}) has it; [] if empty."""
    if not text:
        return []
    key = shared.get(text)
    return [f'{indent}Help: shared `{key}`' if key is not None else f'{indent}{text}']


def _yes_no(flag):
    return 'yes' if flag else 'no'

"""SettingsPanel: the editor form of the document's current model-tree node (GUI thread, CPU only).
Edits go through ModelDocument.mutate / node operations; refused edits are reported through status_message."""
import html
import math
import numbers
from dataclasses import replace
from functools import partial

from PySide6 import QtCore, QtGui, QtWidgets

from .. import materials as material_library
from ..core.backend import cupy_installed
from ..core.mesh import SEGMENT_PROFILES
from ..core.time_estimate import format_seconds
from ..materials.crystal_tensors import (CRYSTAL_MODEL, CRYSTALS, FULL_PIEZO_PROPERTIES, FULL_PLANE,
                                         FULL_STIFFNESS_PROPERTIES, PIEZO_PROPERTIES, PLANES, SAGITTAL_PLANE,
                                         STIFFNESS_PROPERTIES, crystal_properties, entry_settings, full_crystal,
                                         is_crystal_entry, sagittal_crystal)
from ..model.expressions import (CONSTANT_DESCRIPTIONS, COORDINATE_NAMES, FUNCTION_DESCRIPTIONS, ExpressionError,
                                 error_message, evaluate_parameters, evaluate_scalar)
from ..model.analytic import (CURVE_FUNCTION_ARITY, CURVE_RESULT_KINDS, CURVE_TARGETS, CURVE_VARIABLES,
                              DEGREE_FIT_OPERATIONS, DERIVED_CURVE_VARIABLES, DERIVED_RESULT_KINDS, FIT_QUANTITIES,
                              MAX_FIT_DEGREE, RESULT_SAMPLE_NAMES, curve_variables, fit_quantities, is_curve_source)
from ..model.geometry import GEOMETRY_KINDS, PRISM_KINDS, Z_PARAM_NAMES, plane_axes
from ..model.schema import (ADAPTATION_METHODS, GLOBAL_DERIVED_OPERATION, SOURCE_FREE_DERIVED_OPERATIONS,
                            INTERFACE_OPERATIONS, INTERFACE_SIDES,
                            INTERFACE_VARIABLES, FREE_MESH_KINDS, MESH_KIND_DEFAULTS, MESH_KINDS,
                            MESH_PARAMS, PRECISIONS, SELECTION_KINDS, SOLVER_BACKENDS, SOLVER_DEFAULT_PRECISIONS,
                            SOURCE_DERIVED_OPERATIONS, SOLVER_ROLES, Param, coupling_type_accepts, feature_precision_note, fill_defaults,
                            instance_space, output_field_names, param_in_space,
                            memory_check_level, scalar_output_names, solver_backend, solver_grid, solver_misfit,
                            with_default_precision)
from ..model.meshing import PHYSICS_BASE_DIVISOR, PHYSICS_OBJECT_DIVISOR, line_block_hint, mesh_values
from ..model.spec import AdaptationSpec, MeshAxisSpec, MeshBandSpec, MeshSegmentSpec, RefinementSpec, SweepSpec
from ..physics.em.definition import DISPERSION_POLE_PARAMS
from .forms import (DERIVED_OPERATION_LABELS, ERROR_COLOR, HINT_COLOR, SELECTION_KIND_LABELS, CheckList, CommitLineEdit,
                    CommitTextEdit, CurveInputsEdit, ExpressionEdit, FormContext, IntSpinBox, NameEdit, NumberEdit,
                    ParamForm, SelectionPicker, StoredCutsEdit, TableEdit, ValueComboBox, display_number, error_label,
                    face_cell_counts_text, format_number, format_quantity, growing_form, hint_label, make_param_editor,
                    mark_invalid, param_row_label, parse_number_or_text, rich_tooltip, selection_entries,
                    set_param_help, tip_label, value_text, wrapped_label)
from .canvas import DRAGGABLE_KINDS
from .document import ESTIMATION_NODE, MESHES_NODE, harmonic_channel_names, issue_node_path, solver_fits, solver_users
from .estimation_form import EstimationFormMixin
from .helpers import (coupling_mesh_text, evaluation_mesh, memory_estimate_text, mesh_badge_html, mesh_color,
                      mesh_kind_problem, mesh_kind_shown, mesh_summary_text, mesh_swatch_html, physics_mesh_name,
                      several_meshes)
from .mesh_preview import AUTO_PREVIEW_MAX_ELEMENTS, AUTO_PREVIEW_MAX_TETRAHEDRA
from .model_tree import (MESH_KIND_LABELS, RESULT_SETTING_ADDERS, RESULT_SETTINGS_TIPS, coupling_candidates,
                         solver_candidates)

_MAX_EVERY = 1_000_000_000
_REFINED_MESH_MODES = ('nonuniform', 'auto')   # mesh modes that apply the explicit refinements (model/meshing)
_TRIANGLE_ONLY_SETTINGS = frozenset({'discretization'})   # physics settings shown only on a triangle mesh (FEM order)
# forms built from spec lists: rebuilt on every external edit
_LIST_BACKED_NODES = (('model',), ('parameters',), ('couplings',), MESHES_NODE, ('solvers',), ESTIMATION_NODE)
_LEVEL_COLORS = {'error': ERROR_COLOR, 'warning': '#ef6c00', 'info': '#1565c0'}
_ROOT_TITLES = {'model': 'Model', 'parameters': 'Parameters', 'geometry': 'Geometry: domain',
                'selections': 'Selections', 'materials': 'Materials', 'couplings': 'Couplings', 'meshes': 'Meshes',
                'solvers': 'Solvers', 'study': 'Study', 'results': 'Result settings', 'computed_results': 'Results',
                'derived_values': 'Derived values', 'analytic_curves': 'Analytic curves',
                'derived_curves': 'Derived curves', 'estimation': 'Study: parameter estimation'}
_NODE_TITLES ={'geometry': 'Geometry object', 'selection': 'Selection', 'material': 'Material',
                'physics': 'Physics', 'feature': 'Feature', 'coupling': 'Coupling', 'mesh': 'Mesh', 'solver': 'Solver',
                'derived': 'Derived value', 'curve': 'Analytic curve', 'derived_curve': 'Derived curve',
                'derived_plot': 'Derived curve plot'}
_NEW_MESH_ENTRY = '<new mesh>'   # Mesh combo entry of a physics: 'New mesh for this physics' (no identifier)
_PHYSICS_MESH_TIP = ('Mesh this physics runs on. Physics on different meshes exchange fields and heat sources through '
                     "their couplings automatically. 'New mesh for this physics' adds an automatic mesh sized for this "
                     'physics alone.')
_MESH_USERS_TIP = ('Physics that run on this mesh. Tick a physics to move it here; untick it to move it back to the '
                   'default (first) mesh.')
_MESHES_INTRO = ('Each physics runs on one mesh, chosen by its Mesh setting. One mesh serves most models; a second mesh '
                 'lets a physics with slowly varying fields, such as heat, use larger cells than a wave physics, which '
                 'needs cells much smaller than the wavelength. Couplings carry fields and heat sources between meshes '
                 '(heat sources keep their total power). The first mesh is the default: new physics run on it.')

_DOMAIN_PARAMS = (
    Param('size_x', 'float', 4e-6, label='Width (x)', unit='m', min=0.0, min_exclusive=True,
          help='Physical domain size along x (PML cells are added outside it)',
          tip='Width of the simulated region; x runs from 0 at its left edge.\n\n'
              'Absorbing layers, when used, are added outside this width. On a polar or spherical grid it is the '
              "radial depth outward from 'Inner radius'."),
    Param('size_y', 'float', 4e-6, label='Height (y)', unit='m', min=0.0, min_exclusive=True,
          help='Physical domain size along y (PML cells are added outside it)',
          tip='Height of the simulated region; y runs from 0 at its bottom edge.\n\n'
              'Absorbing layers, when used, are added outside this height. On a polar or spherical grid it is the '
              "angle the grid spans, in radians (the unit shown stays m), from 'Start angle'."),
    Param('size_z', 'float', None, label='Depth (z, 3D only)', unit='m', min=0.0, min_exclusive=True, optional=True,
          help='Physical domain size along z; empty = a 2D model',
          tip='Leave empty for a 2D model.\n\n'
              'A value makes the model 3D: every object is extruded over its own bottom / top z, the canvas shows '
              'one layer at a time and only physics that support 3D can be used.'),
    Param('coordinates', 'choice', 'cartesian', label='Grid', choices=('cartesian', 'polar', 'spherical'),
          choice_labels=(('cartesian', 'Rectangular (x, y)'), ('polar', 'Polar (r, angle)'),
                         ('spherical', 'Spherical (r, polar angle)')),
          help='In-plane grid of a 2D model (DomainSpec.coordinates)',
          tip='How the 2D domain is divided into cells.\n\n'
              '- Rectangular: rows and columns along x and y.\n'
              "- Polar: rings and spokes about the origin. 'Width' becomes the radial depth and 'Height' the angle "
              'in radians; circles and rings about the origin are then drawn exactly.\n'
              '- Spherical: the same cells in a half plane rotated about the vertical axis, for balls, shells and '
              'cones; every physics must be set to axisymmetric.\n\n'
              'Polar and spherical grids work for heat transfer and electric currents / electrostatics only.'),
    Param('radius_min', 'float', 0.0, label='Inner radius', unit='m', min=0.0,
          visible_if=(('coordinates', ('polar', 'spherical')),),
          help='Polar / spherical: r of the inner face (DomainSpec.radius_min)',
          tip='Radius of the inner edge of the grid; 0 puts the origin inside the domain.'),
    Param('angle_min', 'float', 0.0, label='Start angle', unit='rad',
          visible_if=(('coordinates', ('polar', 'spherical')),),
          help='Polar: phi of the first ray from +x; spherical: angle from the vertical axis (DomainSpec.angle_min)',
          tip='Angle of the first edge ray, in radians. Polar: measured from the +x direction toward +y. '
              'Spherical: measured from the vertical axis, 0 to pi.'),
    Param('angle_wrap', 'choice', 'none', label='Join edge rays', choices=('none', 'periodic', 'antiperiodic'),
          choice_labels=(('none', 'No (two separate edges)'), ('periodic', 'Repeating'),
                         ('antiperiodic', 'Repeating with flipped sign')),
          visible_if=(('coordinates', ('polar',)),),
          help='Polar: join the last ray to the first (DomainSpec.angle_wrap)',
          tip='Whether the two straight edges of the polar grid are joined into one.\n\n'
              '- No: both edges are boundaries with their own conditions.\n'
              '- Repeating: what leaves through the last edge enters through the first, as in a full ring or one '
              'slice of a pattern that repeats around the circle (the slice angle should divide a full turn).\n'
              '- Repeating with flipped sign: the same, but the solution changes sign from slice to slice (needs an '
              'even number of slices per turn).\n\n'
              'Joined edges cannot carry boundary conditions.'),
)
# (label, tip) of prism params extruded along x / y: {u}, {v} = the model axes of the shape's x, y, {axis} the
# extrusion axis (_axis_params; each tip then gets _AXIS_PLANE_TIP)
_AXIS_PARAM_TEXTS = {
    'x0': ('Corner 1, {u}', '{u} of one corner.\n\nAny two opposite corners work, in any order.'),
    'y0': ('Corner 1, {v}', "{v} of the same corner as 'Corner 1, {u}'."),
    'x1': ('Corner 2, {u}', '{u} of the opposite corner.'),
    'y1': ('Corner 2, {v}', '{v} of the opposite corner.'),
    'cx': ('Centre {u}', '{u} of the centre.'),
    'cy': ('Centre {v}', '{v} of the centre.'),
    'a': ('First semi-axis', "Half the length of the first axis, which lies along +{u} when 'Rotation' is 0."),
    'angle': ('Rotation', 'Counter-clockwise rotation of the first semi-axis away from +{u} toward +{v}.'),
    'points': ('Vertices', 'Corner points as {u}, {v} pairs separated by semicolons: at least 3, not all on one '
                           'line.\n\nThe last point connects back to the first. Overlapping parts of a self-crossing '
                           'polygon become holes.'),
    'z0': ('Extrusion start ({axis})', 'Where the object starts along {axis}; it must be set.\n\nSet it beyond the '
                                       'simulated region to run the object into the absorbing layers.'),
    'z1': ('Extrusion end ({axis})', 'Where the object ends along {axis}; it must be set.'),
    'sidewall_angle': ('Sidewall angle', 'Tilt of the side walls away from the {axis} axis.\n\n0 gives straight '
                                         'walls. A positive angle makes the object narrower toward Extrusion end '
                                         '({axis}), a negative one wider. The shape as entered sits at Extrusion '
                                         'start ({axis}).'),
}
_AXIS_PLANE_TIP = 'The shape is drawn in the {u}-{v} plane and extruded along {axis}.'
_REFINEMENT_DX = Param('dx', 'float', 5e-9, label='Cell size', unit='m', min=0.0, min_exclusive=True,
                       help='Target cell size in and around the object',
                       tip='Cell size wanted in this object and its margin; used only where finer than the main cell '
                           'size.\n\n'
                           'Every grid row and column crossing that region is refined across the whole simulated '
                           'region.')
_REFINEMENT_MARGIN = Param('margin', 'float', None, label='Margin', unit='m', min=0.0, optional=True,
                           help='Refined distance around the object; empty = 3 x cell size',
                           tip="Distance around the object that is refined too.\n\n"
                               "Empty: three times the refinement 'Cell size'.")
_REFINEMENT_OBJECT_TIP = 'Geometry object whose region gets finer cells.'
_MESH_AXES = ('x', 'y', 'z')
_REFINEMENT_AXIS_DX = {
    axis: Param(f'dx_{axis}', 'float', None, label=f'Cell size along {axis}', unit='m', min=0.0, min_exclusive=True,
                optional=True, help=f'Cell size along {axis} in and around the object; empty = the refinement cell size',
                tip=f"Cell size along {axis} only. Empty: the refinement 'Cell size'.\n\n"
                    f'A size not finer than the main cell size leaves the {axis} cells alone, so a waveguide '
                    'crossing the whole domain can refine only across its width.')
    for axis in _MESH_AXES}
_REFINEMENT_AXIS_TIP = ('Cell sizes of this refinement along each axis. Empty boxes use its cell size; a size not finer '
                        'than the main cell size leaves that axis unrefined.')
_REFINEMENT_WHERE_ENTRIES = (('inside', 'Inside the object'), ('edges', 'Along its edges'),
                             ('surface', 'At its surface'))
_REFINEMENT_WHERE_TIP = ("Where the cell size applies (Gmsh tetrahedra only).\n\n"
                         "- Inside the object: the whole object.\n"
                         "- Along its edges: the edges of boxes and the rims of cylinders (spheres have none).\n"
                         "- At its surface: only near its surface, growing coarser to both sides; for a combined "
                         "object its own surface, so a hole stays coarse.\n\n"
                         "Objects Gmsh does not follow (ellipses, polygons, tapered shapes) get no finer cells "
                         "this way.")
_AUTOMATIC_REFINEMENT_TIP = ("Filled in from the automatic sizes when the mesh left Automatic mode. 'Reset to automatic "
                             "sizes' (or choosing Automatic again) recalculates it, also after you changed it.")
_RESET_AUTOMATIC_SIZES_TIP = ('Switch back to Automatic mode: the cell size and the refinements marked automatic are '
                              'recalculated from the model; your own refinements stay.')
_EDIT_AUTOMATIC_SIZES_TIP = ('Switch to Graded mode (Uniform for lattice Boltzmann flow) with these sizes filled into '
                             "the cell size and the refinements, so you can change them. The filled-in refinements are marked automatic: 'Reset to automatic "
                             "sizes' or choosing Automatic again recalculates them.")
_AUTO_RULE_TEXTS = {
    'wavelength': ('From the shortest wavelength: the cell size in the background material, finer inside objects '
                   'whose material has a higher refractive index (metals: by their skin depth).'),
    'physics': (f'From the geometry: the longest domain side divided by {PHYSICS_BASE_DIVISOR:g}, and inside each '
                f'object the physics use its smallest side divided by {PHYSICS_OBJECT_DIVISOR:g}.'),
    'uniform_flow': 'Uniform cells for the lattice Boltzmann flow, sized by the smallest object.'}
_AXIS_MAX_RATIO = Param('max_ratio', 'float', None, label='Maximum growth ratio', unit='1', min=1.0,
                        min_exclusive=True, optional=True,
                        help="Largest size ratio of neighbouring cells along this axis; empty = the mesh's",
                        tip="Largest size ratio between neighbouring cells along this axis. Empty: the mesh "
                            "'Maximum growth ratio'.")
_AXIS_CELL_MODES = (('graded', 'Graded'), ('edges', 'Cell edges listed'), ('segments', 'Segments'))
_AXIS_CELLS_TIP = ('How the cells along this axis are made.\n\n'
                   '- Graded: from the cell sizes, refinements and bands, growing smoothly in between.\n'
                   '- Cell edges listed: every cell edge given by position; nothing else changes them.\n'
                   '- Segments: pieces of the axis, each with a number of cells spaced evenly, growing steadily or '
                   'clustered toward both ends.')
_AXIS_EDGES = Param('edges', 'float_list', (), label='Cell edges', unit='m',
                    help='Increasing cell edge positions from 0 to the domain size along this axis (0 and the end are '
                         'added when missing)',
                    tip='Positions of the cell edges along this axis, increasing, separated by commas. 0 and the '
                        'domain end are added when missing.')
_BAND_PARAMS = (
    Param('lo', 'float', 0.0, label='From', unit='m', help='Start of the refined band along this axis',
          tip='Start of the band along this axis, measured from the domain edge.'),
    Param('hi', 'float', 0.0, label='To', unit='m', help='End of the refined band along this axis',
          tip='End of the band along this axis, measured from the domain edge.'),
    Param('dx', 'float', 5e-9, label='Cell size', unit='m', min=0.0, min_exclusive=True,
          help='Cell size inside the band',
          tip='Cell size wanted between From and To along this axis; the cells grow smoothly outside it.'),
)
_SEGMENT_PROFILE_LABELS = {'uniform': 'Even', 'geometric': 'Growing', 'tanh': 'Clustered at both ends'}
_SEGMENT_PROFILE_TIP = ('How the cells of the segment are spaced.\n\n'
                        '- Even: all cells equal.\n'
                        "- Growing: each cell a fixed factor larger than the previous one; 'Ratio' = last cell / first "
                        'cell (below 1 shrinks).\n'
                        "- Clustered at both ends: small cells at both ends, large in the middle; 'Ratio' = middle cell "
                        '/ end cell.')
_SEGMENT_PARAMS = (
    Param('end', 'float', None, label='End', unit='m', optional=True,
          help='Where the segment ends along this axis; empty = the axis end (last segment only)',
          tip='Where this segment ends, measured from the domain edge; it starts where the previous one ends. Only '
              'the last segment may be left empty: it then ends at the domain edge.'),
    Param('cells', 'int', 10, label='Cells', unit='1', min=1, help='Number of cells of the segment',
          tip='Number of cells in this segment.'),
    Param('ratio', 'float', 1.0, label='Ratio', unit='1', min=0.0, min_exclusive=True,
          help='Growing: last / first cell; clustered: middle / end cell; ignored when even',
          tip="Size ratio of the segment's cells: last / first for Growing, middle / end for Clustered at both ends. "
              'Not used for Even.'),
)
_EDGE_LIST_MAX_CELLS = 500   # switching an axis to listed edges copies the current cells up to this many
_SWEEP_RANGE_TIP = 'A number or an expression of other parameters, such as 500*nm or 1.3*a_lat.'
_SWEEP_LOWER = Param('lower', 'float', None, label='Lower bound', help='First value of the range',
                     tip='First value of the swept parameter.\n\n' + _SWEEP_RANGE_TIP
                         + ' May be larger than the upper bound (the sweep then runs downwards).')
_SWEEP_UPPER = Param('upper', 'float', None, label='Upper bound', help='Last value of the range',
                     tip='Last value of the swept parameter.\n\n' + _SWEEP_RANGE_TIP)
_SWEEP_COUNT = Param('count', 'int', 5, label='Points', unit='1', min=1,
                     help='Number of values, both bounds included',
                     tip='Number of runs from the lower to the upper bound, both bounds included (1 = the lower bound '
                         'only).')
_SWEEP_SPACING = Param('spacing', 'choice', 'linear', label='Spacing', choices=('linear', 'log', 'list'),
                       choice_labels=(('linear', 'Linear'), ('log', 'Logarithmic'), ('list', 'Custom values')),
                       help='Linear or logarithmic steps between the bounds, or the custom values below',
                       tip='Linear: equal steps between the bounds.\n'
                           'Logarithmic: equal ratios between the bounds (both non-zero, same sign).\n'
                           'Custom values: the list below, in its order; the bounds are not used.')
_SWEEP_VALUES = Param('values', 'float_list', (), label='Custom values',
                      help='Comma-separated numbers or expressions of other parameters',
                      tip='Values the swept parameter takes with Custom values spacing, one run each, in this order.\n\n'
                          'Separate them with commas. Each can be a number or an expression of other parameters, such '
                          'as 500*nm.')
_SWEEP_FORM_PARAMS = (_SWEEP_LOWER, _SWEEP_UPPER, _SWEEP_COUNT, _SWEEP_SPACING, _SWEEP_VALUES)
_SWEEP_PREVIEW_VALUES = 6   # values listed in the sweep preview line before '...'
_STORED_CUTS = Param('stored_cuts', 'float', None, label='Extra cut planes', unit='m',
                     help="Cut planes ['x' | 'y' | 'z', position m] the snapshots store besides the mid-height z cut",
                     tip='Planes through the 3D model whose stored fields are saved at each output time, in addition '
                         'to the horizontal plane at mid-height.\n\n'
                         'Write each as "axis, position" and separate them with ";", for example "z, 1e-6; x, '
                         'width/2". The axis is the direction the plane faces (z = horizontal plane), the position a '
                         'number or an expression of parameters in metres; the plane runs through the mesh cells at '
                         'that position.\n\n'
                         'Pick one under Results with Cut plane to replay its snapshots.')
# Study 'Mesh refinement' group (StudySpec.adaptation, S7.7; industry-standard adaptation and error estimates):
# AdaptationSpec attribute -> (row label, tooltip)
_ADAPTATION_METHOD_LABELS = {'adaptive': 'Adaptive: refine where the estimated error is largest',
                             'uniform': 'Uniform: refine everywhere (a mesh-convergence check)'}
_ADAPTATION_ROWS = {
    'method': ('Method', 'How each refinement level makes the meshes finer.\n\n'
                         '- Adaptive: an error estimate of the solution marks where cells or triangles are made '
                         'smaller; elsewhere they stay as they are.\n'
                         '- Uniform: every cell size shrinks by the same factor, which shows how the result converges '
                         'as the whole mesh gets finer.'),
    'target': ('Stop on derived value', 'Derived value watched from level to level: the refinement stops when it '
                                        'changes by no more than the tolerance. The Convergence tab of Results lists '
                                        'it for every level.'),
    'tolerance': ('Tolerance (relative)', 'Largest change of the watched derived value between two levels, relative '
                                          'to its value, at which it counts as converged; for example 0.001 for 0.1 %.'),
    'max_levels': ('Maximum number of refinements', 'Refinement levels after the first run on the model meshes. '
                                                    'Reaching it without convergence gives a warning.'),
    'growth': ('Cell count growth per level', 'Factor by which the number of cells or triangles grows from one level '
                                              'to the next, for example 1.7. Larger factors need fewer levels, but '
                                              'each level costs more.'),
    'meshes': ('Meshes', 'Meshes that are refined. With none ticked, every mesh the physics use is refined.'),
    'max_cells': ('Maximum number of cells', 'The refinement stops before a level whose meshes would have more cells '
                                             'or triangles than this in total. The memory of the computer always '
                                             'limits it too.'),
    'anisotropic': ('Refine x and y separately', 'Rectangular cells: make cells narrower only along the direction in '
                                                 'which the estimated error is largest, so a thin layer is refined '
                                                 'across it but not along it. Off: both directions alike.'),
    'keep_levels': ('Keep fields of every level', 'Keep the stored snapshots of every refinement level, not only those '
                                                  'of the last one, so the Field tab can show each level (their labels '
                                                  'start with the level).'),
}
_ADAPTATION_TIP = ('Repeats the study on finer and finer meshes until a derived value stops changing, and reports how it '
                   'converged (Results, Convergence tab).\n\n'
                   'The results shown are those of the last level. Keep adapted mesh in the Convergence tab adds its '
                   'mesh to the model.')
_MAX_SPIN_INT = 2_147_483_647   # largest value of a Qt spin box
_DERIVED_AT =Param('at', 'float', None, label='At', optional=True, visible_if=(('operation', ('at',)),),
                    help='Abscissa value of the result, linearly interpolated: wavelength in m (flux spectrum), t in '
                         's (timeseries), x in its unit (xy), 0-based row (table)',
                    tip='Position along the result where the value is read, interpolated between samples: a '
                        'wavelength in metres for a spectrum, a time in seconds for a time trace, an x value in its '
                        'unit for a curve, or a row number counted from 0 for a table.')
_DERIVED_REFERENCE = Param('reference', 'float', None, label='Reference', optional=True,
                           help='Analytic value (number or expression of parameters, in the unit of the value); the '
                                'Results panel shows it with the relative error |value - reference| / |reference| '
                                'and dashes it over the sweep plot',
                           tip='Expected value, for example from theory: a number or an expression of parameters, in '
                               'the unit of the value.\n\n'
                               'The Results panel shows the relative difference to it and draws it dashed over the '
                               'sweep plot.')
_FIELD_DERIVED_EXPRESSION = Param(
    'expression', 'expression', None, label='Expression',
    help='Expression of output fields, parameters, x, y, t (final time) and the derived values listed above',
    tip='Quantity to evaluate after the run: an expression of the result fields listed below, parameters, position x '
        'and y, the final time t and the derived values listed above this one.')
_FIELD_DERIVED_OPERATION = Param(
    'operation', 'choice', 'mean', label='Operation', choices=SOURCE_FREE_DERIVED_OPERATIONS,
    choice_labels=tuple(entry for entry in DERIVED_OPERATION_LABELS if entry[0] in SOURCE_FREE_DERIVED_OPERATIONS),
    help='Reduction over the selection; integral = sum of value x cell area; mode_volume = integral / maximum (or / '
         'the value at the point); at = the value in the cell holding the point; expression = no reduction: one '
         'number of parameters, scalar outputs, t and the derived values listed above',
    tip='How the expression is reduced to one number over the selection.\n\n'
        '- Mean: average over the selected area.\n'
        '- Maximum: largest cell value.\n'
        '- Minimum: smallest cell value.\n'
        '- Integral: value summed over the area, per metre of depth in a flat 2D model; over the volume in a 3D '
        'model, and over the rings swept around the axis with an axisymmetric physics.\n'
        '- Mode volume: the integral divided by the largest value, or by the value at the chosen point; with an '
        'energy density such as eps_r times |E|^2 this is the mode volume of a resonance.\n'
        '- Value at a position: the value in the cell holding the chosen point.\n'
        '- Expression of other values: no fields and no selection; the expression combines the derived values '
        'listed above, parameters and computed numbers into one number, for example the square root of one '
        'integral divided by another.')
_GLOBAL_DERIVED_EXPRESSION = Param(
    'expression', 'expression', None, label='Expression',
    help='Expression of the derived values listed above, parameters, scalar outputs and t (no fields)',
    tip='One number computed from the derived values listed above this one, parameters, computed numbers and the '
        'final time t, for example sqrt(N_int / rho_int). Fields and positions cannot be used here: reduce a field '
        'in a derived value above this one first.')
_DERIVED_POINT_TIP =('Point object where Value at a position reads the expression; for Mode volume, the point whose '
                      'value replaces the maximum (empty: the maximum).')
_CURVE_INTERFACE_TIP = ('Read the field along the line from one side of a material edge: each sample takes the side '
                        'named by the interface selection and extends the field from there right up to the line, so '
                        'it never mixes the two materials. Draw the line along the outline of the objects.')
_CURVE_PROJECTION_CHOICES = [('', 'None: the field as it is'), ('normal', 'Normal part (across the outline)'),
                             ('tangential', 'Tangential part (along the outline)')]
_CURVE_SAMPLING_CHOICES = [('nearest', 'Nearest cell (the cell holding each point)'),
                           ('trilinear', 'Interpolated between the surrounding cells')]
_CURVE_SAMPLING_TIP = ('For a line through the volume (six coordinates: start x, y, z and end x, y, z) only.\n\n'
                       '- Nearest cell: each point along the line shows the value of the cell it lies in, so a '
                       'slanted line shows steps.\n'
                       '- Interpolated: each point blends the eight cell centres around it, so a smooth field '
                       'looks smooth along any direction.\n\n'
                       'A line in a cut plane (four coordinates) always uses the nearest cell.')
_CURVE_PROJECTION_TIP = ('With an interface: pick the x part of a field (for example ex_re); the line then shows the '
                         'part of the field arrow that points across the outline (normal) or along it (tangential), '
                         'read from the chosen side together with its y part.')
_DERIVED_INTERFACE_TIP = ('Read the expression along the outline of an interface selection, from the side that '
                          'selection names, instead of over areas.\n\n'
                          'The expression may then also use the outward normal (normal_x, normal_y) and the tangent '
                          '(tangent_x, tangent_y) of the outline, for example em_ex_re*normal_x + em_ey_re*normal_y '
                          'for the normal part of a field. Mean and integral weigh each point by its length of '
                          'outline.')
_RESULT_DERIVED_OPERATION = Param(
    'operation', 'choice', 'mean', label='Operation', choices=SOURCE_DERIVED_OPERATIONS,
    choice_labels=tuple(entry for entry in DERIVED_OPERATION_LABELS if entry[0] in SOURCE_DERIVED_OPERATIONS),
    help='Over the samples: mean / max / min of the finite ones, integral = trapezoid over the abscissa, last sample, '
         'or the value at an abscissa',
    tip='How the samples of the result are reduced to one number.\n\n'
        '- Mean: average of the samples.\n'
        '- Maximum: largest sample.\n'
        '- Minimum: smallest sample.\n'
        '- Integral: area under the curve along its horizontal axis.\n'
        '- Last sample: the final sample.\n'
        "- Value at a position: the value at 'At'.\n"
        '- Position of the maximum: where the curve peaks, on its horizontal axis.\n'
        '- Width at half maximum: full width of the highest peak at half its height.\n'
        "- Polynomial, exponential and Lorentzian fit: one fitted number, chosen under 'Fitted quantity'.\n"
        '- Several Lorentzian peaks, resonance poles (complex): several lines fitted together; the number of lines '
        "is set under 'Degree or peak count'.")
_RESULT_DERIVED_EXPRESSION_TIP = ('Quantity evaluated for every sample of the result: an expression of the sample '
                                  'names listed below and parameters, such as 1 - fraction_positive.')
_FIT_OPERATIONS = tuple(FIT_QUANTITIES)
_DERIVED_WINDOW = Param(
    'window', 'float_list', (), label='Window',
    visible_if=(('operation', tuple(operation for operation in SOURCE_DERIVED_OPERATIONS
                                    if operation not in ('last', 'at'))),),
    help='[lo, hi] bounds on the abscissa of the source (its unit), numbers or parameter expressions; empty = every '
         'sample',
    tip='Part of the horizontal axis the reduction uses: two values, low and high, in the unit of that axis (for '
        'example 1.5e-6, 1.6e-6 for a spectrum in metres).\n\n'
        'Empty: every sample.')
_DERIVED_WAVELENGTH = Param(
    'wavelength', 'float', None, label='Wavelength', unit='m', optional=True, min=0.0, min_exclusive=True,
    help='far_field sources: vacuum wavelength of the pattern (the nearest recorded one); empty = the first',
    tip='Which recorded wavelength of the far-field pattern is used; the nearest one is taken.\n\n'
        'Empty: the first recorded wavelength.')
_DERIVED_DEGREE = Param('degree', 'int', 1, label='Degree or peak count', min=0, max=MAX_FIT_DEGREE,
                        visible_if=(('operation', DEGREE_FIT_OPERATIONS),),
                        help=f'polyfit: polynomial degree 0 .. {MAX_FIT_DEGREE}; multi_lorentz_fit / pole_fit: number '
                             f'of peaks / poles 1 .. {MAX_FIT_DEGREE}',
                        tip='Polynomial fit: highest power of the polynomial, 1 fits a straight line, 2 a parabola.\n'
                            '- Several Lorentzian peaks and resonance poles: how many lines are fitted together; '
                            'they are numbered from the lowest position up.')
_FIT_QUANTITY_LABELS = {'amplitude': 'Amplitude', 'rate': 'Rate (per unit of the horizontal axis)',
                        'decay': 'Decay length (-1 / rate)', 'center': 'Centre position',
                        'fwhm': 'Full width at half maximum', 'offset': 'Offset (background level)',
                        'q': 'Quality factor (centre / width)',
                        'kappa': 'Decay rate kappa (per unit of the horizontal axis)',
                        'decay_length': 'Decay length (1 / kappa)', 'amplitude_start': 'Amplitude at the first sample',
                        'amplitude_end': 'Amplitude at the last sample', 'r_squared': 'Goodness of fit (R squared)',
                        'beta': 'Wavenumber beta (per unit of the horizontal axis)',
                        'wavelength': 'Wavelength (2 pi / beta)', 'forward_abs': 'Forward wave amplitude',
                        'forward_phase': 'Forward wave phase at x = 0 (rad)',
                        'backward_abs': 'Backward wave amplitude',
                        'backward_phase': 'Backward wave phase at x = 0 (rad)',
                        'reflection_abs': 'Reflection |backward| / |forward|',
                        'extrapolated': 'Extrapolated limit', 'order': 'Convergence order',
                        'coefficient': 'Error coefficient', 'error_estimate': 'Error estimate of the best sample',
                        'relative_error_estimate': 'Relative error estimate',
                        'frequency': 'Resonance position (real part of the pole)',
                        'residue_abs': 'Residue magnitude (line strength)',
                        'spacing': 'Mean spacing of neighbouring lines'}
_DERIVED_TOLERANCE = Param(
    'tolerance', 'float', None, label='Self-test tolerance', unit='1', optional=True, min=0.0,
    help='selftest --examples: passes when the relative error to the reference is <= tolerance; empty = not checked',
    tip='Largest relative difference to the reference that the example self-test accepts.\n\n'
        'Empty: the value is not checked. Needs a reference.')
_DERIVED_CURVE_REFERENCE = Param(
    'reference', 'expression', None, label='Reference', optional=True,
    help='Analytic curve of x and parameters, drawn grey dotted beside the curve; curve_error = max |curve - '
         'reference| / its range',
    tip='Expected curve, for example from theory: an expression of x (the horizontal axis) and the parameters.\n\n'
        'It is drawn dotted in grey beside the derived curve, and the difference between the two is reported.')
_DERIVED_CURVE_TOLERANCE = replace(
    _DERIVED_TOLERANCE,
    help='selftest --examples: passes when the curve error to the reference (max |difference| / its range) is <= '
         'tolerance; empty = not checked',
    tip='Largest difference to the reference curve, relative to its range, that the example self-test accepts.\n\n'
        'Empty: the curve is not checked. Needs a reference.')
_SOLVER_BACKEND = Param(
    'backend', 'choice', 'auto', label='Run on', choices=SOLVER_BACKENDS,
    choice_labels=(('auto', 'Automatic (graphics card)' if cupy_installed() else 'Automatic (processor)'),
                   ('gpu', 'Graphics card'), ('cpu', 'Processor')),
    help="Where every physics and solver runs: auto = gpu when cupy is installed, else cpu; cpu needs every physics "
         "to declare it (describe: each physics type's Backends line; validation names the others)",
    tip='Where the physics and their solvers run.\n\n'
        'Automatic uses the graphics card when its software (cupy) is installed, else the processor. The processor '
        'runs only some physics: the Add physics menu greys out the others and the check names any already in '
        'the model.')
_SOLVER_PRECISION = Param(
    'precision', 'choice', 'auto', label='Precision', choices=SOLVER_DEFAULT_PRECISIONS,
    choice_labels=(('auto', 'Each solver its own'),
                   *((name, text[0].upper() + text[1:]) for name, text in PRECISIONS.items())),
    help='Model-wide solver precision (describe SolverDefaultsSpec.precision)',
    tip='Number precision of every solver that offers it, for example double precision everywhere.\n\n'
        'A solver you added keeps the precision set in its own settings. Solvers without that precision keep '
        'their own, and the check lists them. Each solver its own: every solver uses its usual precision.')
# the mesh form's tick box of solver_defaults.memory_check: ticked = 'warning', clear = 'error'
_MEMORY_OVERRIDE = Param(
    'memory_override', 'bool', False, label='Override memory warning',
    help="Ticked: solver_defaults.memory_check 'warning' (describe SolverDefaultsSpec.memory_check), else 'error'",
    tip='The check estimates the memory a model needs before it is built and refuses a model that would not fit.\n\n'
        'Ticked, the model is built and run anyway and the error becomes a warning: for a model whose estimate is '
        'known to be too high. The solvers still stop cleanly when the graphics card really runs out of memory.\n\n'
        'The estimate of a triangle or tetrahedron mesh from its sizes is only approximate: Build mesh makes it '
        'that of the real mesh.')
_SUBPIXEL_SMOOTHING = Param(
    'subpixel_smoothing', 'bool', False, label='Smooth material interfaces',
    help='Anisotropic permittivity averaging at material interfaces (describe SolverDefaultsSpec.subpixel_smoothing)',
    tip='Averages the permittivity of the cells and faces a material boundary cuts, so curved and slanted '
        'boundaries are no longer staircased: resonances and modes converge faster with the mesh.\n\n'
        'Used by the 2D time-domain and frequency-domain electromagnetic physics (not in an optimization study); '
        'the other physics ignore it, and the check lists them.')
_CONFORMAL_BOUNDARIES = Param(
    'conformal_boundaries', 'bool', False, label='Follow curved walls',
    help='Curved walls at their drawn position (describe SolverDefaultsSpec.conformal_boundaries)',
    tip='Places curved and slanted walls where they are drawn instead of on the nearest cell edges, so resonances '
        'and flows converge faster with the mesh.\n\n'
        'Used by the perfect electric conductors of the 2D time-domain electromagnetic physics and of the waveguide '
        'cross-section modes, the solid walls of laminar flow, and in heat and the electric physics by held '
        'temperatures and voltages, insulating walls and the heat flux, cooling, current and charge on object '
        'boundaries; the other physics ignore it, and the check lists them.')
_SOLVERS_INTRO = ('Each physics hands its equations to a solver. Without solver nodes every physics uses its automatic '
                  'choice, shown in the Solver group of its settings. Add a solver to change the method or its '
                  'precision: it is assigned at once to every physics that can use it, and each physics can pick '
                  'another one in its own settings.')
_SELECTION_KIND_TIP = ('What the selection holds.\n\n'
                       '- Domain: areas.\n'
                       '- Boundary: outer edges or object outlines.\n'
                       '- Point: point objects.\n'
                       '- Line: line segments.\n'
                       '- Interface: the outline of objects seen from one side, for values read just inside or just '
                       'outside a material edge (derived values and line profiles).\n\n'
                       'Changing the kind drops entries that do not fit.')
_SELECTION_SIDE_TIP = ('Which side of the outline the field is read from.\n\n'
                       '- Inside: from the cells inside the objects.\n'
                       '- Outside: from the cells around them.\n\n'
                       'The value is extended from cells one and two cells away right up to the edge, so it never '
                       'mixes the two materials.')
_POLE_MODEL_TIP = ('Kind of frequency response this pole adds.\n\n'
                   '- Drude: free electrons, as in a metal.\n'
                   '- Lorentz: one resonance, such as an absorption band.\n'
                   '- Tabulated-data fit: free-electron and resonance terms fitted to the measured optical data of '
                   'the library material over a wavelength range (silver, gold, aluminium, silicon); the check '
                   'reports how well it fits.\n\n'
                   'Switching resets the values except the damping rate.')
_POLE_MODEL_TEXTS = {'drude': 'Drude (free electrons)', 'lorentz': 'Lorentz (resonance)',
                     'nk_fit': 'Fit to the library table of measured optical data'}
_NO_POLES_TEXT = 'No frequency dependence: the material has the same properties at every wavelength.'
_INHERIT_POLES_TIP = ('Checked: use the frequency dependence of the library material (none without a library).\n\n'
                      'Unchecked: edit your own poles below, starting from a copy of the library ones.')
_INHERIT_WITH_CRYSTAL_TIP = 'Not available while the material uses crystal data: remove the crystal data first.'
_CRYSTAL_TIP = ('Fills the stiffness, density, piezoelectric coupling and permittivity of this material from the data '
                'of a piezoelectric crystal, turned into the axes of the model.\n\n'
                'The drawing is the plane of the wave: x is the direction the wave travels, y the surface normal. A '
                'value typed in a property field above wins over the crystal value.')
_NO_CRYSTAL_TEXT = ('No crystal data: the stiffness, piezoelectric coupling and permittivity come from the property '
                    'fields above.')
_CRYSTAL_NAME_TIP = ('Crystal whose stiffness, piezoelectric coupling, permittivity and density are used.\n\n'
                     'The values are given in the standard axes of the crystal and turned by the Euler angles below.')
_CRYSTAL_LABELS = {'lithium_niobate': 'Lithium niobate', 'zinc_oxide': 'Zinc oxide',
                   'aluminum_nitride': 'Aluminium nitride', 'quartz': 'Quartz (alpha, right-handed)',
                   'gallium_arsenide': 'Gallium arsenide'}
_CRYSTAL_CUT_TIP = ('Common substrate orientations: choosing one sets the three Euler angles.\n\n'
                    'Custom: the angles typed below match none of them.')
_CRYSTAL_CUTS = ((None, 'Custom angles'), ((0.0, 90.0, 90.0), 'Y-cut, travelling along Z (YZ)'),
                 ((90.0, 90.0, 0.0), 'X-cut, travelling along Y'),
                 ((0.0, 38.0, 0.0), '128° rotated Y-cut, travelling along X'),
                 ((0.0, 0.0, 0.0), 'Z-cut, travelling along X'))   # (euler angles [deg], label)
_CRYSTAL_ANGLES_TIP = ('Orientation of the crystal as three angles in degrees, the usual convention for surface '
                       'acoustic wave substrates: turn by the first angle about the crystal Z axis, then by the second '
                       'about the new X axis, then by the third about the new Z axis.\n\n'
                       'The turned first axis is the direction of travel (model x), the turned third axis the surface '
                       'normal (model y in 2D, model z in 3D). Plain numbers only, no parameters.')
_CRYSTAL_ANGLE_TIPS = ('First angle: about the crystal Z axis.', 'Second angle: about the new X axis.',
                       'Third angle: about the new Z axis.')
_CRYSTAL_PLANE_TIP = ('Which plane of the turned crystal the drawing shows.\n\n'
                      '- Sagittal (2D models): the plane of the direction of travel (x) and the surface normal (y); '
                      'the wave motion stays in this plane.\n'
                      '- Full 3D (3D models): all three turned axes, travel along x, the surface normal along z.')
_CRYSTAL_PLANE_LABELS = {'sagittal': 'Sagittal (direction of travel and surface normal)',
                         'full': 'Full 3D (x travel, z surface normal)'}
_DEFAULT_CRYSTAL_ENTRY = {'model': CRYSTAL_MODEL, 'crystal': 'lithium_niobate', 'euler_deg': [0.0, 90.0, 90.0],
                          'plane': 'sagittal'}
_PHYSICS_EVERY_TIP = ('Time-dependent studies only: this physics takes one step every so many time steps of the '
                      'fastest physics.\n\n'
                      "Waves need 1. Heat coupled to waves typically uses a few hundred; leave the coupling's "
                      "'Transfer every' on Automatic.")
_COUPLING_EVERY_TIP = ('How often the field is passed on, in time steps of the fastest physics.\n\n'
                       'Automatic: each time the slower of the two physics takes a step.')
_COUPLING_ROLE_TIPS = {'source': 'Physics whose field is read.', 'target': 'Physics that receives the field.'}
_ENABLED_TIP = 'Unchecked: it stays in the model but is ignored by checks and runs.'
_PARAMETER_COLUMN_TIPS = (
    'Name used to refer to this value in any field of the model.\n\n'
    'Letters, digits and underscores, not starting with a digit.',
    'A number or an expression of other parameters, constants and functions, such as 550*nm.',
    'The evaluated value in base units: metres, seconds, kelvin and so on.',
    'Free text, such as the meaning and the unit.')
_NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
_SELECTION_NOUNS = {'domain': 'area', 'boundary': 'edge', 'point': 'point', 'line': 'line'}
_CURVE_SERIES = Param('series', 'str', '', label='Series',
                      help="Simulated quantity the curve belongs to and takes its colour from: an xy series or table "
                           "column name, 'ratio' (Spectra ratio plot), 'fft' (probe FFT plot), a derived value name or "
                           "a table band such as frequency_2 (sweep); empty = every quantity")
_CURVE_X_COLUMN = Param('x_column', 'str', '', label='X column',
                        help='Table targets: number column along the horizontal axis (x in the expression); empty = '
                             'the row index from 0')
_CURVE_LINE = Param('line', 'float_list', (), label='Line', unit='m',
                    help='x0, y0, x1, y1 in m (numbers or parameter expressions): the profile runs from (x0, y0) to '
                         '(x1, y1); 3D models also x0, y0, z0, x1, y1, z1: a straight path in any direction through '
                         'the volume (then no cut height; every snapshot stores its profile)')
_CURVE_PLANE = Param('plane', 'float', None, label='Cut height (z)', unit='m', optional=True,
                     help='3D models: height of the horizontal cut the line lies in; empty = the cut plane chosen in '
                          'the Results panel')
_CURVE_SNAPSHOT = Param('snapshot', 'str', None, label='Snapshot', optional=True,
                        help="Stored snapshot to sample instead of the final state: its number from 0 (negative from "
                             "the end) or its label such as 'mode 2' or 'resonance 1'; the field must be stored "
                             '(a line through the volume: any field)')
_DERIVED_CURVE_PARAMS = (
    Param('expression', 'expression', None, label='Expression',
          help='Expression of the input aliases, x and the parameters, e.g. transmitted / incident',
          tip='Formula of the curve: use the input names given above, x (the horizontal axis value of the first '
              'input) and the parameters.\n\n'
              'Example: transmitted / incident gives a transmission spectrum.'),
    Param('plot', 'str', '', label='Plot',
          help="Plot the curve is drawn in, shared with the other curves naming it; empty = a plot of its own",
          tip='Name of the plot the curve appears in. Curves with the same plot name are drawn together.\n\n'
              'Leave empty for a plot of its own, named after the curve.'),
    Param('label', 'str', '', label='Legend', help="Legend text; empty = the curve name"),
    Param('unit', 'str', '', label='Unit', help="Unit of the values (free text); empty = none",
          tip='Unit of the values, shown on the vertical axis, for example W or 1.'),
    Param('log_x', 'bool', False, label='Logarithmic x',
          help='Logarithmic horizontal axis; the first drawn curve of a plot sets it for the plot'),
    Param('log_y', 'bool', False, label='Logarithmic y',
          help='Logarithmic vertical axis; the first drawn curve of a plot sets it for the plot'),
)
_DERIVED_CURVE_INPUTS_TIP = ("One input per line: a name, '=', then the result and the sample to read, written "
                             "physics.feature:sample (for example incident = em.flux1:power).\n\n"
                             "The first input that is a curve sets the horizontal axis; later curves are interpolated "
                             "onto it. A single number, such as a table total, enters as a constant.")
_CURVE_TARGET_HINTS = {   # target kind -> what its expression may use besides the parameters
    'timeseries': 't = time [s]; iteration = sample number from 1 when every sample has the same time (stationary '
                  'iterations); with series fft: frequency [Hz], drawn on the probe FFT plot',
    'flux_spectrum': 'wavelength [m]',
    'far_field': 'angle [rad from +x], wavelength [m] of the shown frequency (normalised to its peak)',
    'xy': 'x in the x unit of the result (wavelength [m] for results versus wavelength)',
    'table': 'x = the X column value in its unit (the row index from 0 when empty)',
    'line': 'x, y [m] on the line (3D: also z) and s [m] = distance from the start point',
    'sweep': 'the swept parameter (every parameter is re-evaluated per point)'}
_POLE_MODEL_ENTRIES = tuple((model, {'nk_fit': 'Tabulated-data fit'}.get(model, model.capitalize()))
                            for model in DISPERSION_POLE_PARAMS)   # (value, label)
_COUPLINGS_INTRO = ('Couplings pass a result of one physics into another: for example, light absorbed in a metal '
                    'heats it, or a rise in temperature changes the refractive index.')


# ---- module helpers ----
def _slot(function, *leading):
    """One-argument Qt slot calling function(*leading, value) (binds loop values without late binding)."""
    return lambda value: function(*leading, value)


def _study_adaptation(spec):
    """spec.study.adaptation; KeyError when the study has none (a refresher then rebuilds the form)."""
    if spec.study.adaptation is None:
        raise KeyError('the study has no mesh refinement')
    return spec.study.adaptation


def _sweep_form_values(sweep) -> dict:
    """Values of the sweep range form (bounds, points, spacing, custom values) of a SweepSpec."""
    return {param.name: getattr(sweep, param.name) for param in _SWEEP_FORM_PARAMS}


def _sweep_preview(sweep, parameters) -> str:
    """'n runs: v1, v2, ..., vn' of the sweep at the current parameter values, or why they do not evaluate."""
    points = sweep.point_values()
    if not points:
        return 'No sweep values yet.'
    try:
        swept_numbers = [evaluate_scalar(point, parameters) for point in points]
    except ExpressionError as exc:
        return f'Sweep values not evaluated: {error_message(exc)}'
    shown = [display_number(number) for number in swept_numbers]
    if len(shown) > _SWEEP_PREVIEW_VALUES:
        shown = shown[:_SWEEP_PREVIEW_VALUES - 1] + ['...', shown[-1]]
    return f"{len(swept_numbers)} run{'s' if len(swept_numbers) != 1 else ''}: " + ', '.join(shown)


def _adaptation_blocker(spec, registry) -> str:
    """Why mesh refinement cannot be switched on for the study of spec (validation's S7.7 rules), '' when it can."""
    try:
        study_type = registry.study_type(spec.study.type)
    except KeyError:
        return 'Choose a known study type first.'
    if not study_type.supports_adaptation:
        return f'The {study_type.label} study does not refine meshes.'
    if spec.study.sweep is not None:
        return 'Not available together with a parameter sweep.'
    if spec.study.estimation is not None:
        return 'Not available together with parameter estimation (refine the mesh first, then estimate on it).'
    if isinstance(spec.study.settings, dict) and spec.study.settings.get('parity_sectors') is True:
        return 'Not available together with parity sectors.'
    if not spec.derived_values:
        return 'Add a derived value first: the refinement stops when it converges.'
    return ''


def _form_label_of(params, name):
    """Form label (else name) of the Param called name in params, for Undo texts; name when none matches."""
    return next((param.label or name for param in params if param.name == name), name)


def _on_click(button, callback):
    """Call callback() when the button is clicked."""
    button.clicked.connect(lambda _checked=False: callback())


def _curve_input_text(key, kind):
    """Hint text of one result a derived curve can read: its key and its sample names (besides the abscissa)."""
    abscissa, sample_names = RESULT_SAMPLE_NAMES[kind]
    if sample_names is None:
        return f"{key} ({'its series' if kind == 'xy' else 'its number columns'} by name)"
    return f"{key} ({', '.join(name for name in sample_names if name != abscissa)})"


def _form_layout(parent_layout):
    form = growing_form()
    parent_layout.addLayout(form)
    return form


def _remove_button(callback):
    """'Remove' tool button calling callback()."""
    button = QtWidgets.QToolButton()
    button.setText('Remove')
    _on_click(button, callback)
    return button


def _button_row(*widgets):
    """Row layout of widgets packed to the left."""
    row = QtWidgets.QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    row.addStretch(1)
    return row


def _group(title, *widgets):
    box = QtWidgets.QGroupBox(title)
    box_layout = QtWidgets.QVBoxLayout(box)
    for widget in widgets:
        box_layout.addWidget(widget)
    return box


def _collapsible_group(title, expanded, on_toggled):
    """(widget, body): an arrow header button that shows / hides the untitled body group box; on_toggled(bool)."""
    widget = QtWidgets.QWidget()
    widget_layout = QtWidgets.QVBoxLayout(widget)
    widget_layout.setContentsMargins(0, 0, 0, 0)
    header = QtWidgets.QToolButton()
    header.setText(title)
    header.setCheckable(True)
    header.setChecked(expanded)
    header.setAutoRaise(True)
    header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    header.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
    body = QtWidgets.QGroupBox()
    body.setVisible(expanded)

    def toggle(checked):
        header.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        body.setVisible(checked)
        on_toggled(checked)

    header.toggled.connect(toggle)
    widget_layout.addWidget(header)
    widget_layout.addWidget(body)
    return widget, body


def _node_title(path):
    if not path:
        return 'Settings'
    if len(path) == 1:
        return _ROOT_TITLES.get(path[0], str(path[0]))
    return f"{_NODE_TITLES.get(path[0], str(path[0]))}: {' / '.join(map(str, path[1:]))}"


def _axis_params(params, axis):
    """Prism params whose labels / tips name the model axes of an extrusion along axis 'x' / 'y' (help unchanged);
    any other axis returns params as they are."""
    if axis not in ('x', 'y'):
        return params
    first, second = plane_axes(axis)
    names = {'u': first, 'v': second, 'axis': axis}
    plane_tip = _AXIS_PLANE_TIP.format(**names)
    relabelled = []
    for param in params:
        texts = _AXIS_PARAM_TEXTS.get(param.name)
        if texts is None:
            relabelled.append(param)
            continue
        label, tip = (text.format(**names) for text in texts)
        relabelled.append(replace(param, label=label, tip=f'{tip}\n\n{plane_tip}'))
    return tuple(relabelled)


def _library_material(library):
    """Library Material named `library`, or None."""
    return material_library.REGISTRY.get(library) if isinstance(library, str) else None


def _default_pole(model):
    return {'model': model, **{param.name: param.default for param in DISPERSION_POLE_PARAMS[model]}}


def _describe_poles(poles):
    """One line per dispersion dict ('Drude (free electrons): plasma frequency 1.37e16 rad/s, ...')."""
    lines = []
    for pole in poles:
        model = pole.get('model')
        params = {param.name: param for param in DISPERSION_POLE_PARAMS.get(model, ())}
        fields = []
        for key, value in pole.items():
            if key == 'model':
                continue
            param = params.get(key)
            label = (param.label or key) if param is not None else key
            unit = param.unit if param is not None and param.unit not in ('', '1') else ''
            number = format_number(value) if isinstance(value, numbers.Real) else str(value)
            fields.append(f'{label[:1].lower()}{label[1:]} {number} {unit}'.rstrip())
        lines.append(f"{_POLE_MODEL_TEXTS.get(model, str(model))}: {', '.join(fields)}")
    return '\n'.join(lines)


def _feature_params(physics, feature_type, space=''):
    """feature_type.params with the units of the physics' space (Param.space_units) and a text 'channel' param turned
    into a drop-down of the physics' harmonic channels (unchanged without channels)."""
    params = tuple(param_in_space(param, space) for param in feature_type.params)
    channels = tuple(harmonic_channel_names(physics))
    if not channels:
        return params
    return tuple(replace(param, kind='choice', choices=channels,
                         default=param.default if param.default in channels else None)
                 if param.kind == 'str' and param.name == 'channel' else param for param in params)


def _time_estimate_text(estimate, message=''):
    """Mesh-node line of the estimated run time on this computer (a range when it rests on guessed iteration counts),
    or why there is none."""
    if estimate is None:
        if not message:
            return 'Run time: not available (see the Validation tab)'
        return ('Run time: not estimated; measure this computer once with "python -m open_sol profile" '
                '(about 30 seconds)')
    text = f'Run time on this computer: about {format_seconds(estimate.seconds_guess)}'
    if estimate.has_guess:
        text += (f' (between {format_seconds(estimate.seconds_low)} and {format_seconds(estimate.seconds_high)}: '
                 'some iteration counts are guesses)')
    text += estimate.outdated_text()
    if estimate.not_estimated:
        text += f"; without {', '.join(estimate.not_estimated)}"
    return text


def _count_words(count):
    """'one' to 'ten', digits above."""
    return _NUMBER_WORDS[count] if 0 <= count < len(_NUMBER_WORDS) else str(count)


def _slot_display(slot, meaning):
    """Short drop-down text of a source slot: its meaning up to the first ':', ';' or ' (', plus the slot name."""
    short = meaning
    for separator in (':', ';', ' ('):
        short = short.split(separator, 1)[0]
    short = short.strip().rstrip('.')
    return f'{short}  ({slot})' if short else slot


def _pole_at(spec, material_name, index, model):
    """Dispersion dict `index` of a material, still a `model` pole (KeyError otherwise)."""
    poles = spec.material(material_name).dispersion
    if (not isinstance(poles, list) or not 0 <= index < len(poles) or not isinstance(poles[index], dict)
            or poles[index].get('model') != model):
        raise KeyError(f"pole {index + 1} of material '{material_name}' changed; select the node again")
    return poles[index]


def _has_own_poles(dispersion):
    """True for a dispersion list holding a pole besides a crystal entry, or a malformed (non-list) value."""
    if dispersion is None:
        return False
    return not isinstance(dispersion, list) or any(not is_crystal_entry(pole) for pole in dispersion)


def _crystal_entry_at(spec, material_name):
    """The material's crystal dict (in the spec, mutable); KeyError when it has none."""
    entry = material_library.crystal_entry(spec.material(material_name).dispersion)
    if entry is None:
        raise KeyError(f"the crystal data of material '{material_name}' changed; select the node again")
    return entry


def _crystal_angles(entry):
    """[phi, theta, psi] of a crystal entry as stored (absent = 0, 0, 0; a malformed list gives three None)."""
    angles = entry.get('euler_deg', [0.0, 0.0, 0.0])
    return list(angles) if isinstance(angles, (list, tuple)) and len(angles) == 3 else [None, None, None]


def _crystal_cut(entry):
    """The _CRYSTAL_CUTS angles matching a crystal entry, else None (custom)."""
    angles = _crystal_angles(entry)
    if not all(isinstance(angle, numbers.Real) and not isinstance(angle, bool) for angle in angles):
        return None
    angles = tuple(float(angle) for angle in angles)
    return next((cut for cut, _label in _CRYSTAL_CUTS if cut == angles), None)


def _crystal_summary(entry):
    """(text, problem) of a crystal entry: the constants it sets in model axes, or why it does not expand."""
    try:
        if entry_settings(entry)[2] == FULL_PLANE:
            return _full_crystal_summary(full_crystal(entry)), None
        reduced = sagittal_crystal(entry)
    except (ValueError, TypeError) as exc:   # TypeError: an unhashable crystal name from a hand-edited file
        return '', str(exc)
    values = reduced.properties
    lines = ['Values in the model axes:',
             'Stiffness [GPa]: ' + ', '.join(f'{name} {values[name] / 1e9:.4g}' for name in STIFFNESS_PROPERTIES),
             'Piezoelectric coupling [C/m²]: ' + ', '.join(f'e{name[-2:]} {values[name]:.3g}'
                                                          for name in PIEZO_PROPERTIES),
             f"Relative permittivity at constant strain: xx {values['eps_dc_xx']:.4g}, yy {values['eps_dc_yy']:.4g}, "
             f"xy {values['eps_dc_xy']:.4g}",
             f"Density {values['rho']:.4g} kg/m³"]
    if reduced.monoclinic_ratio > 0.0:
        lines.append(f'Also set: c16 {reduced.c16 / 1e9:.4g} and c26 {reduced.c26 / 1e9:.4g} GPa '
                     f'({100.0 * reduced.monoclinic_ratio:.2g} % of the largest stiffness), which couple the two '
                     'in-plane directions.')
    antiplane = max(reduced.antiplane_stiffness_ratio, reduced.antiplane_piezo_ratio)
    if antiplane > 0.0:
        lines.append(f'The drawing is not a mirror plane of this crystal: its coupling to motion out of the plane '
                     f'({100.0 * antiplane:.2g} % of the largest entry) is left out.')
    lines.append(f'Data: {CRYSTALS[reduced.crystal].reference}. Check the values against the source before relying on '
                 'their last digits.')
    return '\n'.join(lines), None


def _full_crystal_summary(expanded):
    """Text of a 'full' crystal entry: its non-zero constants in the 3D model axes."""
    values = expanded.properties
    stiffness = [f'{name} {values[name] / 1e9:.4g}' for name in FULL_STIFFNESS_PROPERTIES if values[name] != 0.0]
    piezo = [f'e{name[-2:]} {values[name]:.3g}' for name in FULL_PIEZO_PROPERTIES if values[name] != 0.0]
    permittivity = [f'{name[-2:]} {values[name]:.4g}' for name in ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_zz', 'eps_dc_yz',
                                                                   'eps_dc_xz', 'eps_dc_xy') if values[name] != 0.0]
    return '\n'.join(['Values in the model axes (x travel, z surface normal; zero entries left out):',
                      'Stiffness [GPa]: ' + ', '.join(stiffness),
                      'Piezoelectric coupling [C/m²]: ' + (', '.join(piezo) or 'none'),
                      'Relative permittivity at constant strain: ' + ', '.join(permittivity),
                      f"Density {values['rho']:.4g} kg/m³",
                      f'Data: {CRYSTALS[expanded.crystal].reference}. Check the values against the source before '
                      'relying on their last digits.'])


def _inherited_placeholders(material):
    """{property: grey placeholder text} of what an empty property field falls back to: the library value, else the
    value of the material's crystal entry (when it expands)."""
    library_material = _library_material(material.library)
    library_values = library_material.model_properties() if library_material is not None else {}
    texts = {key: f'{format_number(value)}  (library)' for key, value in library_values.items()}
    entry = material_library.crystal_entry(material.dispersion)
    if entry is not None:
        try:
            crystal_values = crystal_properties(entry)
        except (ValueError, TypeError):   # reported by the crystal data group
            crystal_values = {}
        texts.update((key, f'{display_number(value)}  (crystal)') for key, value in crystal_values.items())
    return texts


def _set_property_placeholder(editor, param, placeholder):
    """Grey placeholder of a material property editor: the inherited value text, else the default."""
    if placeholder is None:
        placeholder = f'{value_text(param.default)}  (default)' if param.default is not None else ''
    editor.setPlaceholderText(placeholder)


def _refinement_at(mesh_spec, index, object_name):
    """Refinement `index` of a MeshSpec, still refining `object_name` (KeyError otherwise)."""
    refinements = mesh_spec.refinements
    if not 0 <= index < len(refinements) or refinements[index].object != object_name:
        raise KeyError(f"refinement {index + 1} ({object_name}) changed; select the node again")
    return refinements[index]


def _mesh_axis_at(mesh_spec, index, axis):
    """MeshSpec.axes entry `index`, still for `axis` (KeyError otherwise)."""
    entries = mesh_spec.axes
    if not 0 <= index < len(entries) or entries[index].axis != axis:
        raise KeyError(f"axis options {index + 1} ({axis}) changed; select the node again")
    return entries[index]


def _mesh_axis_item(mesh_spec, index, axis, list_name, item_index):
    """Band / segment `item_index` of the axis entry (list_name 'bands' / 'segments'; KeyError when gone)."""
    items = getattr(_mesh_axis_at(mesh_spec, index, axis), list_name)
    if not 0 <= item_index < len(items):
        raise KeyError(f'{list_name[:-1]} {item_index + 1} of axis {axis} no longer exists')
    return items[item_index]


def _axis_cell_mode(axis_spec):
    """'edges', 'segments' or 'graded' (neither list set) of a MeshAxisSpec."""
    return 'edges' if axis_spec.edges else 'segments' if axis_spec.segments else 'graded'


def _tensor_graded(mesh_spec):
    """True when the mesh is graded by the tensor rules (per-axis refinement sizes, per-axis options): rectangular
    cells, builtin and swept tetrahedra (model meshing.graded_window_edges; swept: only z, the layers)."""
    return mesh_spec.kind not in FREE_MESH_KINDS or (mesh_spec.kind == 'tetrahedra' and (
        mesh_spec.mesher == 'builtin' or mesh_spec.sweep_axis == 'z'))


def _window_axis_edges(estimate, axis):
    """Float64 window cell edges (physical frame, 0 .. length) of a MeshEstimate along axis (builtin tetrahedra: the
    sizing's axis edges without the shell); None without one (or for a triangle / gmsh tetrahedron mesh)."""
    if estimate is not None and estimate.kind == 'tetrahedra' and estimate.sizing.axis_edges:
        sizing = estimate.sizing
        edges = sizing.axis_edges['xyz'.index(axis)]
        n_lo, n_hi = (round(sizing.pml_widths[f'{axis}_{side}'] / sizing.h_max) for side in ('lo', 'hi'))
        return edges[n_lo:len(edges) - n_hi]
    if estimate is not None and estimate.kind == 'tetrahedra' and axis == 'z' and len(estimate.sizing.layer_edges):
        sizing = estimate.sizing   # swept prism layers: the z layer planes without the shell
        n_lo, n_hi = (round(sizing.pml_widths[f'z_{side}'] / sizing.h_max) for side in ('lo', 'hi'))
        return sizing.layer_edges[n_lo:len(sizing.layer_edges) - n_hi]   # a float64 array (meshing.layer_edges)
    if estimate is None or estimate.kind != 'tensor':
        return None
    edges = {'x': estimate.x_edges, 'y': estimate.y_edges, 'z': estimate.z_edges}[axis]
    if edges is None:
        return None
    lo, hi = int(estimate.npml_sides.get(f'{axis}_lo', 0)), int(estimate.npml_sides.get(f'{axis}_hi', 0))
    return edges[lo:len(edges) - hi]


def _rounded(value):
    """value rounded to 6 significant digits (readable editor text for generated defaults)."""
    return float(f'{value:.6g}')


def _default_axis_edges(current_edges, length):
    """Interior edges of a newly listed axis: the current window edges (up to _EDGE_LIST_MAX_CELLS cells, rounded),
    else 20 even cells; 0 and length are added by the mesher."""
    if current_edges is None or len(current_edges) - 1 > _EDGE_LIST_MAX_CELLS:
        return [_rounded(length * k / 20) for k in range(1, 20)]
    interior = []
    for edge in current_edges[1:-1]:
        edge = _rounded(float(edge))
        if 0.0 < edge < length and (not interior or edge > interior[-1]):
            interior.append(edge)
    return interior


def _axis_length(spec, axis, parameters):
    """Evaluated domain size along axis (ValueError for z in a 2D model; ExpressionError on bad input)."""
    size = getattr(spec.domain, f'size_{axis}')
    if size is None:
        raise ValueError(f'axis {axis} needs a 3D model (set the domain depth)')
    return evaluate_scalar(size, parameters)


def _rounded_size(size):
    """A filled-in automatic size [m] at 4 significant digits (readable in the editors)."""
    return float(f'{size:.4g}')


def _smallest_automatic_size(estimate):
    """Finest size [m] an auto-mode estimate applies: its base size or its finest refinement."""
    return min([estimate.auto_dx] + [size for _object_name, size in estimate.auto_refinements])


def _automatic_sizes_text(estimate):
    """Lines of the 'Automatic sizes' group: the base size, each refined object's size and how much finer, the rule."""
    size_name = 'Maximum element size' if estimate.kind != 'tensor' else 'Cell size'
    lines = [f"{size_name} {format_quantity(estimate.auto_dx, 'm')}"]
    for object_name, size in estimate.auto_refinements:
        lines.append(f"    {object_name}: {format_quantity(size, 'm')}  ({estimate.auto_dx / size:.3g} × finer)")
    if not estimate.auto_refinements and estimate.auto_rule != 'uniform_flow':
        lines.append('    no object is refined')
    lines.append(_AUTO_RULE_TEXTS.get(estimate.auto_rule, ''))
    lines.append('These sizes follow the model: they change with the geometry, materials and wavelength.')
    return '\n'.join(line for line in lines if line)


def _base_cell_size(mesh_spec, parameters):
    """Evaluated 'Cell size' of a MeshSpec, else four times the default refinement size."""
    if mesh_spec.dx is None:   # auto mode without a smallest cell size
        return _REFINEMENT_DX.default * 4
    try:
        return evaluate_scalar(mesh_spec.dx, parameters)
    except ExpressionError:
        return _REFINEMENT_DX.default * 4


def _fit_quantities(operation, degree):
    """Fitted quantities of a fit operation (analytic.fit_quantities; an invalid degree read as 1); () for other
    operations."""
    return fit_quantities(operation, degree) or fit_quantities(operation, 1)


def _fit_quantity_label(name):
    """Drop-down text of a fitted quantity: a named one, a per-peak '<name>_<k>' or a polyfit 'c<k>'."""
    if name in _FIT_QUANTITY_LABELS:
        return _FIT_QUANTITY_LABELS[name]
    base, _, number = name.rpartition('_')
    if number.isdigit() and base in _FIT_QUANTITY_LABELS:
        return f'{_FIT_QUANTITY_LABELS[base]}, line {number}'
    return f'Coefficient of x^{name[1:]}'


def _fit_param(operation, degree):
    """'fit' choice Param of a derived value with this operation / degree (shown for the fit operations only)."""
    quantities = _fit_quantities(operation, degree) or _fit_quantities('polyfit', MAX_FIT_DEGREE)
    labels = tuple((name, _fit_quantity_label(name)) for name in quantities)
    return Param('fit', 'choice', None, label='Fitted quantity', choices=quantities, choice_labels=labels,
                 visible_if=(('operation', _FIT_OPERATIONS),),
                 help='Fit operations: the fitted number reported (polyfit c<k> = coefficient of x**k; exp_fit '
                      'y = amplitude exp(rate x); lorentz_fit y = offset + amplitude / (1 + (2 (x - center) / fwhm)^2); '
                      'two_exp_fit A exp(-kappa x) + B exp(kappa (x - L)); two_wave_fit complex A_f exp(-j beta x) + '
                      'A_b exp(j beta x) from <name>_re and <name>_im; richardson y = limit + C x^order; '
                      'multi_lorentz_fit <name>_<k> of degree Lorentzians + offset; pole_fit complex sum r_k / (x - '
                      'nu_k) + d0 + d1 x from <name>_re and <name>_im, k ordered by position)',
                 tip='Which number of the fit is the value: a polynomial coefficient, the rate or decay length of '
                     'an exponential, or the centre, width or quality factor of a Lorentzian peak.\n'
                     '- Two-exponential, two-wave and Richardson fits: the decay rate between two ends, the '
                     'wavenumber and reflection of a standing wave (pick the _re series), or the converged limit '
                     'of a step-size or mesh sweep.\n'
                     '- Several peaks and resonance poles: the centre, width or quality factor of line 1, 2, ... '
                     '(numbered from the lowest position up), or the mean spacing of the lines (pick the _re series '
                     'for poles).')


def _default_fit(operation, degree):
    """First fitted quantity of a fit operation (polyfit: the highest coefficient), '' for other operations."""
    quantities = _fit_quantities(operation, degree)
    if not quantities:
        return ''
    return quantities[-1] if operation == 'polyfit' else quantities[0]


def _selection_hint(feature_type):
    """'Select one point.', 'Select one to four lines.', plus the boundary and uniform-mesh notes."""
    minimum, maximum = feature_type.min_entities, feature_type.max_entities
    if maximum == minimum:
        count, plural = _count_words(minimum), minimum != 1
    elif maximum is None:
        count, plural = f'at least {_count_words(minimum)}', minimum != 1
    else:
        count, plural = f'{_count_words(minimum)} to {_count_words(maximum)}', True
    nouns = [('line' if name == 'line' else GEOMETRY_KINDS[name].label.lower()) if name in GEOMETRY_KINDS
             else str(name) for name in feature_type.object_kinds]
    if not nouns:
        nouns = [_SELECTION_NOUNS.get(feature_type.selection_kind, str(feature_type.selection_kind))]
    what = ' or '.join(f'{noun}s' if plural else noun for noun in nouns)
    sentences = [f'Select {count} {what}.']
    if feature_type.selection_kind == 'boundary':
        if not feature_type.allow_outer_faces:
            sentences.append('Domain edges cannot be selected.')
        if not feature_type.allow_interior:
            sentences.append('Object outlines cannot be selected.')
    if feature_type.requires_uniform_mesh:
        sentences.append('The mesh becomes uniform while this feature is enabled.')
    return ' '.join(sentences)


class SettingsPanel(EstimationFormMixin, QtWidgets.QWidget):
    """Form of document.current_node, refreshed in place after edits; a node selection, an 'all' change or a
    structural own edit triggers one deferred rebuild."""

    status_message = QtCore.Signal(str)   # message of a refused edit (status bar / Messages log)
    show_results_requested = QtCore.Signal()   # 'Show results' of the Results node's form
    hover_entities = QtCore.Signal(object, bool)   # (entities, fill domains) of a hovered selection entry, [] = none

    def __init__(self, document, registry, parent=None):
        super().__init__(parent)
        self._document = document
        self._registry = registry
        self._node_path = None
        self._contexts = []
        self._refreshers = []
        self._issues_label = None
        self._pending_rebuild = False
        self._pending_refresh = False
        self._timer_armed = False
        self._expanded_material_groups = set()   # physics type names whose '(not in the model)' group was opened
        self._expanded_details = set()           # type keys whose 'Technical details' group was opened
        self._mesh_store = None                  # mesh_preview.ElementMeshStore (element mesh forms: Build mesh)

        self._header = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        header_font = self._header.font()
        header_font.setBold(True)
        self._header.setFont(header_font)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._header)
        layout.addWidget(self._scroll, 1)
        self.setMinimumWidth(320)

        document.changed.connect(self._on_document_changed)
        # a new memory budget or a built element mesh: the estimates, the node's issues and the override follow
        document.issues_invalidated.connect(partial(self._schedule, False))
        document.node_selected.connect(self._on_node_selected)
        self._schedule(rebuild=True)

    # ---- public ----
    def set_mesh_store(self, store):
        """Element mesh previews (mesh_preview.ElementMeshStore): the element mesh forms' Build mesh group."""
        self._mesh_store = store
        store.changed.connect(self._on_mesh_store_changed)

    def _on_mesh_store_changed(self, mesh_name):
        if self._node_path == ('mesh', mesh_name):
            self._schedule(rebuild=False)

    def rebuild(self):
        """Recreate the form of document.current_node (old editors are signal-blocked before deletion)."""
        path = self._current_path()
        scroll_bar = self._scroll.verticalScrollBar()
        scroll_value = scroll_bar.value() if path == self._node_path else 0
        self.hover_entities.emit([], False)   # the old form's pickers are signal-blocked before they could
        self._discard_form()
        self._node_path = path
        self._contexts, self._refreshers, self._issues_label = [], [], None
        self._header.setText(_node_title(path))
        form = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(form)
        fills_height = False
        if path is None or not self._document.node_exists(path):
            layout.addWidget(hint_label('Select a node in the model tree.'))
        else:
            fills_height = bool(self._build_node(path, layout))
            self._issues_label = wrapped_label(text_format=QtCore.Qt.TextFormat.RichText)
            layout.addWidget(self._issues_label)
            self._update_issues()
        if not fills_height:
            layout.addStretch(1)
        for picker in form.findChildren(SelectionPicker):
            picker.entities_hovered.connect(self.hover_entities)
        self._scroll.setWidget(form)
        if scroll_value:
            QtCore.QTimer.singleShot(0, lambda: scroll_bar.setValue(scroll_value))

    def refresh(self):
        """Update checks, visibility, evaluated values and issues of the current form without recreating it."""
        path = self._current_path()
        if path is None or path != self._node_path or not self._document.node_exists(path):
            self.rebuild()
            return
        for ctx in self._contexts:
            ctx.update_from_document()
        try:
            for refresher in self._refreshers:
                refresher()
        except KeyError:   # a node the form addresses by name is gone
            self.rebuild()
            return
        self._update_issues()

    # ---- scheduling ----
    def _on_document_changed(self, section, origin):
        """Refresh in place (refresh() rebuilds when the node changed); rebuild on 'all' or, for an external edit,
        on the model / parameters forms, which list spec contents."""
        path = self._current_path()
        self._schedule(rebuild=section == 'all' or (origin is not self and path in _LIST_BACKED_NODES)
                       or (bool(path) and path[0] == 'derived_plot')   # its curve list follows every edit
                       or (origin is not self and bool(path) and path[0] == 'mesh'))   # its Used by list

    def _on_node_selected(self, _path):
        self._schedule(rebuild=True)

    def _schedule(self, rebuild):
        """Coalesce rebuild / refresh requests into one deferred call (never rebuild inside a signal handler)."""
        if rebuild:
            self._pending_rebuild = True
        else:
            self._pending_refresh = True
        if not self._timer_armed:
            self._timer_armed = True
            QtCore.QTimer.singleShot(0, self._run_pending)

    def _run_pending(self):
        rebuild, refresh = self._pending_rebuild, self._pending_refresh
        self._timer_armed = self._pending_rebuild = self._pending_refresh = False
        if rebuild:
            self.rebuild()
        elif refresh:
            self.refresh()

    def _current_path(self):
        current = self._document.current_node
        return tuple(current) if current is not None else None

    def _discard_form(self):
        """Block every signal of the current form, then detach and delete it later (blocking first: detaching
        reparents it, and the focus-out of a half-typed editor would otherwise commit onto the changed model)."""
        old_form = self._scroll.widget()
        if old_form is None:
            return
        old_form.blockSignals(True)
        for child in old_form.findChildren(QtCore.QObject):
            child.blockSignals(True)
        self._scroll.takeWidget()
        old_form.hide()
        old_form.deleteLater()

    def _update_issues(self):
        if self._issues_label is None:
            return
        issues = [issue for issue in self._document.issues() if issue_node_path(issue) == self._node_path]
        self._issues_label.setVisible(bool(issues))
        self._issues_label.setText('<br>'.join(
            f'<span style="color:{_LEVEL_COLORS.get(issue.level, ERROR_COLOR)}">'
            f'<b>{html.escape(issue.level.capitalize())}:</b></span> {html.escape(issue.message)}'
            for issue in issues))

    # ---- edit plumbing ----
    def _report(self, message):
        self.status_message.emit(message)

    def _commit(self, section, apply, label, rebuild=False):
        """One undoable document.mutate(origin=self); a refused edit is reported and the form rebuilt."""
        try:
            changed = self._document.mutate(section, apply, label=label, origin=self)
        except (ValueError, KeyError, ExpressionError) as exc:
            self._report(f'{label}: {error_message(exc)}')
            self._schedule(rebuild=True)
            return False
        if changed and rebuild:
            self._schedule(rebuild=True)
        return changed

    def _document_op(self, operation, label):
        """Run a document node operation; True, or False after reporting a refused edit."""
        try:
            operation()
        except (ValueError, KeyError, ExpressionError) as exc:
            self._report(f'{label}: {error_message(exc)}')
            self._schedule(rebuild=True)
            return False
        return True

    def _context(self, **options):
        """FormContext refreshed with the form (options: expression_names, exclude_names)."""
        ctx = FormContext(self._document, **options)
        self._contexts.append(ctx)
        return ctx

    def _add_name_row(self, form, path):
        name_edit = NameEdit(path[-1])
        name_edit.value_committed.connect(_slot(self._rename, path))
        form.addRow('Name', name_edit)

    def _rename(self, path, new_name):
        if new_name == path[-1]:
            return
        if not self._document_op(lambda: self._document.rename_node(path, new_name), f'Rename {path[-1]}'):
            return
        new_path = path[:-1] + (new_name,)
        if self._current_path() != new_path and self._document.node_exists(new_path):
            self._document.select(new_path)

    def _set_enabled(self, path, checked):
        self._document_op(lambda: self._document.set_enabled(path, bool(checked)), f'Enable {path[-1]}')

    def _enabled_checkbox(self, path, enabled_of):
        """'Enabled' checkbox of a feature / coupling; enabled_of() reads the stored flag (also on refresh)."""
        checkbox = QtWidgets.QCheckBox('Enabled')
        checkbox.setToolTip(rich_tooltip(_ENABLED_TIP))
        checkbox.setChecked(enabled_of() is True)
        checkbox.clicked.connect(_slot(self._set_enabled, path))
        self._refreshers.append(lambda: checkbox.setChecked(enabled_of() is True))
        return checkbox

    def _expression_edit(self, param, value, ctx, on_commit, placeholder=''):
        """ExpressionEdit committing to on_commit(value), its check re-run on every refresh; placeholder: its grey text
        while empty (else the Param's own)."""
        editor = ExpressionEdit(param, value, ctx)
        if placeholder:
            editor.setPlaceholderText(placeholder)
        editor.value_committed.connect(on_commit)
        self._refreshers.append(editor.refresh_check)
        return editor

    def _add_unknown_keys(self, layout, params, values, what, section, mapping_of):
        """Group of stored keys that `params` does not declare, each with a Remove button; mapping_of(spec) looks
        the owning dict up by name inside the mutate."""
        known = {param.name for param in params}
        unknown_keys = [key for key in values if key not in known] if isinstance(values, dict) else []
        if not unknown_keys:
            return
        box = QtWidgets.QGroupBox(f'Unknown {what}s')
        box_form = growing_form(box)
        box_form.addRow(hint_label(f'{what.capitalize()}s this type does not know, for example from an older model '
                                   'file. Remove them: until then the model check reports an error.'))
        for key in unknown_keys:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            value_label = wrapped_label(value_text(values[key]), QtCore.Qt.TextFormat.PlainText)
            row_layout.addWidget(value_label, 1)
            row_layout.addWidget(_remove_button(partial(self._remove_stored_key, section, mapping_of, key, what)))
            box_form.addRow(str(key), row)
        layout.addWidget(box)

    def _remove_stored_key(self, section, mapping_of, key, what):
        def apply(spec):
            mapping_of(spec).pop(key, None)
        self._commit(section, apply, f'Remove {what} {key}', rebuild=True)

    def _add_param_group(self, layout, title, params, ctx, section, mapping_of, label_prefix, what=None,
                         rebuild_keys=()):
        """Titled ParamForm over the dict mapping_of(spec) (looked up by name on each commit / refresh); with
        `what`, also the group of stored keys `params` does not declare; an edit of a rebuild_keys param rebuilds."""
        values = mapping_of(self._document.spec)
        if params:
            key_labels = {param.name: param.label or param.name for param in params}   # Undo texts: the form labels
            param_form = ParamForm(params, values, ctx, partial(self._set_item, section, mapping_of, label_prefix,
                                                                rebuild_keys=rebuild_keys, key_labels=key_labels))
            layout.addWidget(_group(title, param_form))
            self._refreshers.append(lambda: param_form.refresh(mapping_of(self._document.spec)))
        if what is not None:
            self._add_unknown_keys(layout, params, values, what, section, mapping_of)

    def _type_adder_row(self, title_text, type_info, operation):
        """Row of a coupling / solver type to add: the title, an Add button running the document operation (Undo text
        'Add <type label>') and the type's summary (else description) below."""
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QGridLayout(row)
        row_layout.setContentsMargins(0, 6, 0, 0)
        title = QtWidgets.QLabel(title_text)
        title.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        add_button = QtWidgets.QPushButton('Add')
        _on_click(add_button, partial(self._document_op, operation, f'Add {type_info.label}'))
        row_layout.addWidget(title, 0, 0)
        row_layout.addWidget(add_button, 0, 1, QtCore.Qt.AlignmentFlag.AlignTop)
        summary = type_info.summary or type_info.description
        if summary:
            row_layout.addWidget(hint_label(summary), 1, 0, 1, 2)
        row_layout.setColumnStretch(0, 1)
        return row

    def _add_summary(self, layout, type_info):
        """Grey GUI summary of a physics / feature / coupling / study type (its description when it has none)."""
        text = type_info.summary or type_info.description
        if text:
            layout.addWidget(hint_label(text))

    def _add_technical_details(self, layout, details_key, type_info):
        """Collapsed 'Technical details' group with the full description (only when a summary is shown instead);
        its open state is remembered per details_key for the session."""
        if not (type_info.summary and type_info.description):
            return
        group_widget, body = _collapsible_group('Technical details', details_key in self._expanded_details,
                                                _slot(self._set_details_expanded, details_key))
        details = hint_label(type_info.description)
        details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        QtWidgets.QVBoxLayout(body).addWidget(details)
        layout.addWidget(group_widget)

    def _set_details_expanded(self, details_key, expanded):
        if expanded:
            self._expanded_details.add(details_key)
        else:
            self._expanded_details.discard(details_key)

    def _set_item(self, section, mapping_of, label_prefix, key, value, rebuild_keys=(), key_labels=None):
        """Store value under key; the Undo text names the form label of key (key_labels), else the key."""
        def apply(spec):
            mapping_of(spec)[key] = value
        key_text = (key_labels or {}).get(key, key)
        self._commit(section, apply, f'Set {label_prefix}: {key_text}', rebuild=key in rebuild_keys)

    # ---- builders: (path, layout) -> True when the form fills the height itself ----
    def _build_node(self, path, layout):
        builders = {
            'model': self._build_model, 'parameters': self._build_parameters, 'geometry': self._build_geometry,
            'selections': self._build_selections_root, 'selection': self._build_selection,
            'materials': self._build_materials_root, 'material': self._build_material,
            'physics': self._build_physics, 'feature': self._build_feature,
            'couplings': self._build_couplings_root, 'coupling': self._build_coupling,
            'meshes': self._build_meshes_root, 'mesh': self._build_mesh,
            'solvers': self._build_solvers_root, 'solver': self._build_solver,
            'study': self._build_study, 'estimation': self._build_estimation, 'results': self._build_results_root,
            'derived_values': self._build_derived_values_root, 'derived': self._build_derived,
            'analytic_curves': self._build_analytic_curves_root, 'curve': self._build_curve,
            'derived_curve': self._build_derived_curve, 'derived_curves': self._build_derived_curves_root,
            'derived_plot': self._build_derived_plot,
            'computed_results': self._build_computed_results,
        }
        builder = builders.get(path[0])
        if builder is None:
            layout.addWidget(hint_label(f'No settings for {path}.'))
            return False
        return builder(path, layout)

    def _build_model(self, path, layout):
        spec = self._document.spec
        form = _form_layout(layout)
        title_edit = CommitLineEdit(str(spec.title))
        title_edit.value_committed.connect(self._set_title)
        form.addRow(tip_label('Title', 'Free-text name of the model. It does not affect the simulation.'), title_edit)
        description_edit = CommitTextEdit(spec.description or '', 'What the model shows and how to read its results')
        description_edit.setToolTip('Free text saved with the model. It does not affect the simulation.')
        description_edit.value_committed.connect(self._set_description)
        form.addRow('Description', description_edit)
        counts = ((len(spec.parameters), 'parameter'), (len(spec.geometry), 'geometry object'),
                  (len(spec.selections), 'selection'), (len(spec.materials), 'material'),
                  (len(spec.physics), 'physics'), (len(spec.couplings), 'coupling'),
                  (len(spec.derived_values), 'derived value'))
        layout.addWidget(hint_label(', '.join(f"{count} {noun}{'' if count == 1 or noun == 'physics' else 's'}"
                                              for count, noun in counts)))

    def _build_parameters(self, path, layout):
        spec = self._document.spec
        table = QtWidgets.QTableWidget(len(spec.parameters), 4)
        table.setHorizontalHeaderLabels(['Name', 'Expression', 'Value', 'Description'])
        for column, tip in enumerate(_PARAMETER_COLUMN_TIPS):
            table.horizontalHeaderItem(column).setToolTip(rich_tooltip(tip))
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        for row, parameter in enumerate(spec.parameters):
            name_item = QtWidgets.QTableWidgetItem(str(parameter.name))
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, parameter.name)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(value_text(parameter.expression)))
            value_item = QtWidgets.QTableWidgetItem()
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 2, value_item)
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(parameter.description)))
        self._update_parameter_values(table)
        table.resizeColumnsToContents()
        table.itemChanged.connect(_slot(self._on_parameter_item_changed, table))
        self._refreshers.append(partial(self._update_parameter_values, table))
        layout.addWidget(table, 1)

        add_button = QtWidgets.QPushButton('Add parameter')
        _on_click(add_button, lambda: self._document_op(self._document.add_parameter, 'Add parameter'))
        remove_button = QtWidgets.QPushButton('Remove selected')
        _on_click(remove_button, partial(self._remove_parameters, table))
        layout.addLayout(_button_row(add_button, remove_button))
        expression_hint = hint_label(
            'Double-click a cell to edit it.\n\n'
            'A value can be a number or an expression of other parameters, units such as nm or fs, physical '
            'constants and functions such as sine or square root. Hover over this text for the full list.')
        expression_hint.setToolTip(rich_tooltip(
            'Constants and units:\n' + '\n'.join(f'- {name}: {meaning}'
                                                 for name, meaning in CONSTANT_DESCRIPTIONS.items()),
            'Functions:\n' + '\n'.join(f'- {FUNCTION_DESCRIPTIONS[name]}' for name in sorted(FUNCTION_DESCRIPTIONS))))
        layout.addWidget(expression_hint)
        return True

    def _build_geometry(self, path, layout):
        if len(path) == 1:
            self._build_domain(layout)
        else:
            self._build_geometry_object(path, layout)

    def _build_domain(self, layout):
        ctx = self._context()
        size_form = ParamForm(_DOMAIN_PARAMS, self._domain_values(), ctx, self._set_domain_size)
        layout.addWidget(_group('Domain size', size_form))
        self._refreshers.append(lambda: size_form.refresh(self._domain_values()))
        layout.addWidget(hint_label('Add geometry objects by drawing them on the canvas, or right-click Geometry in '
                                    'the model tree.'))

    def _build_geometry_object(self, path, layout):
        name = path[1]
        spec = self._document.spec
        geometry_object = spec.geometry_object(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        kind = GEOMETRY_KINDS.get(geometry_object.kind)
        form.addRow('Kind', QtWidgets.QLabel(kind.label if kind is not None else str(geometry_object.kind)))
        if kind is None:
            layout.addWidget(error_label(f'Unknown geometry kind {geometry_object.kind!r}; valid: '
                                         + ', '.join(GEOMETRY_KINDS)))
            return
        non_domain = {other.name for other in spec.geometry
                      if other.kind not in GEOMETRY_KINDS or GEOMETRY_KINDS[other.kind].entity_kind != 'domain'}
        object_params = geometry_object.params if isinstance(geometry_object.params, dict) else {}
        shown_params = tuple(param for param in kind.params   # 2D models: z params only while set (to clear them)
                             if spec.is_3d or param.name not in Z_PARAM_NAMES
                             or object_params.get(param.name) is not None)
        axis = object_params.get('axis') if spec.is_3d and geometry_object.kind in PRISM_KINDS else None
        self._add_param_group(layout, 'Parameters', _axis_params(shown_params, axis),
                              self._context(exclude_names={name} | non_domain), 'geometry',
                              lambda spec: spec.geometry_object(name).params, name, rebuild_keys=('axis',))
        if axis in ('x', 'y'):
            first, second = plane_axes(axis)
            layout.addWidget(hint_label(f'Extruded along {axis}: the shape is drawn in the {first}-{second} plane '
                                        '(the labels above name those axes). Edit it here; the canvas shows it but '
                                        'cannot drag it.'))
        elif geometry_object.kind in DRAGGABLE_KINDS:   # not composites, not the 3D box / cylinder / sphere
            layout.addWidget(hint_label('You can also drag the object on the canvas while it is selected here. '
                                        'Objects placed with expressions cannot be dragged.'))

    def _build_selections_root(self, path, layout):
        layout.addWidget(hint_label('A selection is a named group of areas, edges, points or lines. Sources, '
                                    'materials and results can all use the same selection, so a change in one place '
                                    'updates them all.'))
        buttons = []
        for kind in SELECTION_KINDS:
            button = QtWidgets.QPushButton(f'Add {kind} selection')
            _on_click(button, partial(self._add_selection, kind))
            buttons.append(button)
        layout.addLayout(_button_row(*buttons))

    def _build_selection(self, path, layout):
        name = path[1]
        spec = self._document.spec
        selection = spec.selection(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        kind_texts = dict(SELECTION_KIND_LABELS)
        kind_combo = ValueComboBox([(kind, kind_texts.get(kind, kind)) for kind in SELECTION_KINDS], selection.kind,
                                   _SELECTION_KIND_TIP)
        kind_combo.value_committed.connect(_slot(self._set_selection_kind, name))
        form.addRow(tip_label('Kind', _SELECTION_KIND_TIP), kind_combo)
        if selection.kind not in SELECTION_KINDS:
            layout.addWidget(error_label(f'Unknown selection kind {selection.kind!r}; choose one above.'))
            return
        if selection.kind == 'interface':
            side_combo = ValueComboBox([('inside', 'Inside the objects'), ('outside', 'Outside the objects')],
                                       selection.side, _SELECTION_SIDE_TIP)
            side_combo.value_committed.connect(_slot(self._set_selection_side, name))
            form.addRow(tip_label('Side', _SELECTION_SIDE_TIP), side_combo)
        picker = SelectionPicker(spec, selection.kind, selection.entities,
                                 _slot(self._set_selection_entities, name), exclude=(name,))
        layout.addWidget(_group('Entities', picker))
        self._refreshers.append(lambda: picker.set_selected(self._document.spec.selection(name).entities))

    def _build_materials_root(self, path, layout):
        layout.addWidget(hint_label('Materials give properties to areas of the model. Where areas overlap, the '
                                    'material lower in the list wins.\n\n'
                                    "A material on 'all' is the background: it fills every cell no other material "
                                    'covers.\n\n'
                                    'A library material fills in measured values; you can override any of them.'))
        blank_button = QtWidgets.QPushButton('Add blank material')
        _on_click(blank_button, partial(self._add_material, None))
        library_combo = QtWidgets.QComboBox()
        for index, (key, library_material) in enumerate(sorted(material_library.REGISTRY.items())):   # as the tree
            library_combo.addItem(key)
            if library_material.summary:
                library_combo.setItemData(index, rich_tooltip(library_material.summary),
                                          QtCore.Qt.ItemDataRole.ToolTipRole)
        library_button = QtWidgets.QPushButton('Add from library')
        _on_click(library_button, lambda: self._add_material(library_combo.currentText()))
        layout.addLayout(_button_row(blank_button, library_combo, library_button))

    def _build_material(self, path, layout):
        name = path[1]
        spec = self._document.spec
        material = spec.material(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        library_entries = [(None, '(none)')] + [(key, key) for key in sorted(material_library.REGISTRY)]
        library_combo = ValueComboBox(library_entries, material.library,
                                      'Library values are used for every property left empty below',
                                      item_tips={key: library_material.summary
                                                 for key, library_material in material_library.REGISTRY.items()})
        library_combo.value_committed.connect(_slot(self._set_material_library, name))
        form.addRow('Library', library_combo)
        picker = SelectionPicker(spec, 'domain', material.selection, _slot(self._set_material_selection, name))
        layout.addWidget(_group('Selection', picker))
        self._refreshers.append(lambda: picker.set_selected(self._document.spec.material(name).selection))
        library_material = _library_material(material.library)
        ctx = self._context()
        self._add_material_properties(layout, name, material, ctx)
        present_names = {physics.type for physics in spec.physics}
        present_types = [physics_type for physics_type in self._registry.physics_types()
                         if physics_type.name in present_names]
        if (material_library.crystal_entry(material.dispersion) is not None
                or any(param.name in PIEZO_PROPERTIES for physics_type in present_types
                       for param in physics_type.material_properties)):
            self._add_crystal(layout, name, material)
        if _has_own_poles(material.dispersion) or any(physics_type.uses_dispersion for physics_type in present_types):
            self._add_dispersion(layout, name, material, library_material, ctx)

    def _add_material_properties(self, layout, material_name, material, ctx):
        """One group per present physics type (spec order), then the other registered types (collapsed unless
        expanded before or holding a value), each with its own material_properties minus names already shown;
        placeholders show the library / crystal value (refreshed in place)."""
        properties = material.properties if isinstance(material.properties, dict) else {}
        placeholder_editors = []   # (property, param, editor) of the built editors
        boxes = {}                 # physics type name -> body of its collapsible group
        registered = {physics_type.name: physics_type for physics_type in self._registry.physics_types()}
        present_names = list(dict.fromkeys(physics.type for physics in self._document.spec.physics
                                           if physics.type in registered))
        ordered_types = [(registered[type_name], True) for type_name in present_names]
        ordered_types += [(physics_type, False) for type_name, physics_type in registered.items()
                          if type_name not in present_names]
        shown = set()

        def fill(box, params):
            """The group's editors from the current material (a collapsed group builds them on its first opening, so
            selecting a material does not build ~100 validated editors)."""
            current = self._document.spec.material(material_name)
            current_properties = current.properties if isinstance(current.properties, dict) else {}
            current_placeholders = _inherited_placeholders(current)
            box_form = growing_form(box)
            for param in params:
                optional_param = replace(param, optional=True)
                editor = self._material_property_editor(material_name, param.name, optional_param,
                                                        current_properties.get(param.name),
                                                        current_placeholders.get(param.name), ctx,
                                                        rebuild_on_commit=False)
                placeholder_editors.append((param.name, optional_param, editor))
                box_form.addRow(param_row_label(param), editor)

        for physics_type, in_model in ordered_types:
            params = [param for param in physics_type.material_properties if param.name not in shown]
            if not params:
                continue
            shown.update(param.name for param in params)
            if in_model:
                box = group_widget = QtWidgets.QGroupBox(physics_type.label)
                fill(box, params)
            else:
                expanded = (physics_type.name in self._expanded_material_groups
                            or any(properties.get(param.name) is not None for param in params))
                built = [expanded]

                def toggled(checked, type_name=physics_type.name, params=params, built=built):
                    self._set_material_group_expanded(type_name, checked)
                    if checked and not built[0]:
                        built[0] = True
                        fill(boxes[type_name], params)
                group_widget, box = _collapsible_group(f'{physics_type.label}  (not in the model)', expanded, toggled)
                boxes[physics_type.name] = box
                if expanded:
                    fill(box, params)
            layout.addWidget(group_widget)
        unknown_keys = [key for key in properties if key not in shown]
        if unknown_keys:
            generic = Param('value', 'float', None, optional=True, field_dependent=True, state_dependent=True,
                            help='Not a property of any registered physics; clear it to remove it',
                            tip='Not a property of any physics this program knows, for example from an older file.'
                                '\n\nClear the field to remove it.')
            box = QtWidgets.QGroupBox('Unknown properties')
            box_form = growing_form(box)
            for key in unknown_keys:
                box_form.addRow(str(key), self._material_property_editor(material_name, key, generic,
                                                                         properties[key], None, ctx,
                                                                         rebuild_on_commit=True))
            layout.addWidget(box)
        layout.addWidget(hint_label('An empty field uses the grey value shown in it: the library or crystal value, '
                                    'or else the default.\n\n'
                                    'A value can be an expression of parameters, position x and y, and fields of '
                                    'other physics such as the temperature.'))

        def refresh_placeholders():   # crystal edits refresh in place
            current = _inherited_placeholders(self._document.spec.material(material_name))
            for key, param, editor in placeholder_editors:
                if param.kind != 'table':   # a table keeps its column placeholder
                    _set_property_placeholder(editor, param, current.get(key))
        self._refreshers.append(refresh_placeholders)

    def _material_property_editor(self, material_name, key, param, value, placeholder, ctx, rebuild_on_commit):
        """ExpressionEdit, or a TableEdit (one row per line) for a 'table' property such as the B-H curve (it keeps its
        column placeholder: no library / crystal text, like refresh_placeholders)."""
        if param.kind == 'table':
            editor = TableEdit(param, value, ctx)
        else:
            editor = ExpressionEdit(param, value, ctx)
            _set_property_placeholder(editor, param, placeholder)
        editor.value_committed.connect(_slot(self._set_material_property, material_name, key, rebuild_on_commit))
        self._refreshers.append(editor.refresh_check)
        return editor

    def _add_dispersion(self, layout, material_name, material, library_material, ctx):
        box = QtWidgets.QGroupBox('Frequency dependence (dispersion poles)')
        box_layout = QtWidgets.QVBoxLayout(box)
        inherit = QtWidgets.QCheckBox('Use the library poles')
        inherit.setChecked(material.dispersion is None)
        has_crystal = material_library.crystal_entry(material.dispersion) is not None
        inherit.setEnabled(not has_crystal)   # null dispersion cannot hold the crystal entry
        inherit.setToolTip(rich_tooltip(_INHERIT_POLES_TIP, _INHERIT_WITH_CRYSTAL_TIP if has_crystal else ''))
        inherit.clicked.connect(_slot(self._set_dispersion_inherit, material_name))
        box_layout.addWidget(inherit)
        if material.dispersion is None:
            library_poles = library_material.dispersion_models() if library_material is not None else []
            box_layout.addWidget(hint_label(_describe_poles(library_poles) if library_poles else _NO_POLES_TEXT))
        elif not isinstance(material.dispersion, list):
            box_layout.addWidget(error_label(f'Dispersion must be a list of poles, got {material.dispersion!r}'))
        else:
            pole_count = 0
            for index, pole in enumerate(material.dispersion):
                if is_crystal_entry(pole):   # edited in the crystal data group
                    continue
                pole_count += 1
                box_layout.addWidget(self._pole_editor(material_name, index, pole_count, pole, ctx))
            if not pole_count:
                box_layout.addWidget(hint_label(_NO_POLES_TEXT))
            buttons = []
            for model, model_label in _POLE_MODEL_ENTRIES:
                button = QtWidgets.QPushButton(f'Add {model_label} pole')
                _on_click(button, partial(self._add_pole, material_name, model))
                buttons.append(button)
            box_layout.addLayout(_button_row(*buttons))
        layout.addWidget(box)

    def _add_crystal(self, layout, material_name, material):
        """'Crystal data' group of the material's {'model': 'crystal'} dispersion entry: crystal, cut presets, Euler
        angles, plane and the constants it sets (read-only); edits refresh in place, add / remove rebuild."""
        box = QtWidgets.QGroupBox('Crystal data (piezoelectric)')
        box.setToolTip(rich_tooltip(_CRYSTAL_TIP))
        box_layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(box)
        entry = material_library.crystal_entry(material.dispersion)
        if entry is None:
            box_layout.addWidget(hint_label(_NO_CRYSTAL_TEXT))
            add_button = QtWidgets.QPushButton('Use crystal data')
            add_button.setToolTip(rich_tooltip(_CRYSTAL_TIP))
            _on_click(add_button, partial(self._add_crystal_entry, material_name))
            box_layout.addLayout(_button_row(add_button))
            return
        form = _form_layout(box_layout)
        crystal_combo = ValueComboBox([(name, _CRYSTAL_LABELS.get(name, name)) for name in CRYSTALS],
                                      entry.get('crystal'), _CRYSTAL_NAME_TIP,
                                      item_tips={name: f'Crystal class {data.crystal_class}. Data: {data.reference}.'
                                                 for name, data in CRYSTALS.items()})
        crystal_combo.value_committed.connect(partial(self._set_crystal_value, material_name, 'crystal'))
        form.addRow(tip_label('Crystal', _CRYSTAL_NAME_TIP), crystal_combo)
        cut_combo = ValueComboBox(_CRYSTAL_CUTS, _crystal_cut(entry), _CRYSTAL_CUT_TIP)
        cut_combo.value_committed.connect(partial(self._set_crystal_cut, material_name))
        form.addRow(tip_label('Cut', _CRYSTAL_CUT_TIP), cut_combo)
        angle_row = QtWidgets.QHBoxLayout()
        angle_edits = []
        for index, (angle, tip) in enumerate(zip(_crystal_angles(entry), _CRYSTAL_ANGLE_TIPS)):
            angle_edit = NumberEdit(angle, f'{tip}\n\n{_CRYSTAL_ANGLES_TIP}')
            angle_edit.value_committed.connect(partial(self._set_crystal_angle, material_name, index))
            angle_row.addWidget(angle_edit)
            angle_edits.append(angle_edit)
        form.addRow(tip_label('Euler angles [deg]', _CRYSTAL_ANGLES_TIP), angle_row)
        plane_combo = ValueComboBox([(plane, _CRYSTAL_PLANE_LABELS.get(plane, plane)) for plane in PLANES],
                                    entry.get('plane', SAGITTAL_PLANE), _CRYSTAL_PLANE_TIP)
        plane_combo.value_committed.connect(partial(self._set_crystal_value, material_name, 'plane'))
        form.addRow(tip_label('Model plane', _CRYSTAL_PLANE_TIP), plane_combo)
        summary_label = hint_label('')
        box_layout.addWidget(summary_label)
        remove_button = QtWidgets.QPushButton('Remove crystal data')
        _on_click(remove_button, partial(self._remove_crystal_entry, material_name))
        box_layout.addLayout(_button_row(remove_button))

        def refresh_crystal():   # KeyError (entry gone) makes refresh() rebuild
            current = _crystal_entry_at(self._document.spec, material_name)
            crystal_combo.set_value(current.get('crystal'))
            cut_combo.set_value(_crystal_cut(current))
            for angle_edit, angle in zip(angle_edits, _crystal_angles(current)):
                angle_edit.set_value(angle)
                angle_edit.refresh_check()
            plane_combo.set_value(current.get('plane', SAGITTAL_PLANE))
            text, problem = _crystal_summary(current)
            summary_label.setText(problem or text)
            summary_label.setStyleSheet(f'QLabel {{ color: {ERROR_COLOR if problem else HINT_COLOR}; }}')

        refresh_crystal()
        self._refreshers.append(refresh_crystal)

    def _pole_editor(self, material_name, index, number, pole, ctx):
        """Editor of dispersion list item `index`, titled 'Pole <number>' (crystal entries are not counted)."""
        box = QtWidgets.QGroupBox(f'Pole {number}')
        form = growing_form(box)
        model = pole.get('model') if isinstance(pole, dict) else None
        fields = DISPERSION_POLE_PARAMS.get(model)
        if fields is None:
            form.addRow(error_label(f"Unsupported pole {pole!r}; the model must be one of "
                                    f"{', '.join(DISPERSION_POLE_PARAMS)}."))
        else:
            model_combo = ValueComboBox(_POLE_MODEL_ENTRIES, model, _POLE_MODEL_TIP)
            model_combo.value_committed.connect(_slot(self._set_pole_model, material_name, index, model))
            form.addRow(tip_label('Model', _POLE_MODEL_TIP), model_combo)
            for field_param in fields:
                editor = self._expression_edit(
                    field_param, pole.get(field_param.name), ctx,
                    _slot(self._set_pole_field, material_name, index, model, field_param.name))
                form.addRow(param_row_label(field_param), editor)
        remove_button = QtWidgets.QPushButton('Remove pole')
        _on_click(remove_button, partial(self._remove_pole, material_name, index))
        form.addRow('', remove_button)
        return box

    def _build_physics(self, path, layout):
        name = path[1]
        physics = self._document.spec.physics_by_name(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        try:
            physics_type = self._registry.physics_type(physics.type)
        except KeyError as exc:
            form.addRow('Type', QtWidgets.QLabel(str(physics.type)))
            layout.addWidget(error_label(error_message(exc)))
            return
        form.addRow('Type', QtWidgets.QLabel(physics_type.label))
        form.addRow(tip_label('Mesh', _PHYSICS_MESH_TIP), self._physics_mesh_row(name))
        every = IntSpinBox(1, _MAX_EVERY, physics.every, help_text=_PHYSICS_EVERY_TIP)
        every.value_committed.connect(_slot(self._set_physics_every, name))
        self._refreshers.append(lambda: every.refresh_value(self._document.spec.physics_by_name(name).every))
        form.addRow(tip_label('Step every [time steps]', _PHYSICS_EVERY_TIP), every)
        self._add_summary(layout, physics_type)
        study_labels = [self._study_label(study_name) for study_name in physics_type.supported_studies]
        layout.addWidget(hint_label('Works with the study types: ' + (', '.join(study_labels) or 'none') + '.'))
        self._add_param_group(layout, 'Settings', self._shown_physics_settings(physics, physics_type),
                              self._context(), 'physics', lambda spec: spec.physics_by_name(name).settings, name,
                              what='setting')
        self._add_solver_group(layout, name, physics_type)
        self._add_technical_details(layout, f'physics:{physics_type.name}', physics_type)

    def _shown_physics_settings(self, physics, physics_type):
        """Settings Params of the physics form: element-mesh-only settings (FEM 'discretization') only on a triangle /
        tetrahedron mesh, elsewhere only while stored (to clear them)."""
        try:
            on_triangles = self._document.spec.mesh_of(physics).kind in FREE_MESH_KINDS
        except KeyError:   # unknown mesh name: the validation names it; show everything
            on_triangles = True
        stored = physics.settings if isinstance(physics.settings, dict) else {}
        return tuple(param for param in physics_type.settings
                     if on_triangles or param.name not in _TRIANGLE_ONLY_SETTINGS or param.name in stored)

    def _physics_mesh_row(self, physics_name):
        """Mesh combo of a physics (the model's meshes, then 'New mesh for this physics') beside a swatch in the
        chosen mesh's colour; both follow edits made elsewhere."""
        spec = self._document.spec
        entries = [(mesh.name,
                    f'{mesh.name}  ({mesh_summary_text(mesh, self._document.mesh_estimate(mesh.name)[0])})')
                   for mesh in spec.meshes]
        entries.append((_NEW_MESH_ENTRY, 'New mesh for this physics…'))
        tips = {_NEW_MESH_ENTRY: 'Add an automatic mesh sized for this physics alone and move the physics onto it.'}
        combo = ValueComboBox(entries, self._physics_mesh_value(physics_name), _PHYSICS_MESH_TIP, item_tips=tips)
        combo.value_committed.connect(_slot(self._set_physics_mesh, physics_name))
        swatch = QtWidgets.QLabel()
        swatch.setTextFormat(QtCore.Qt.TextFormat.RichText)

        def refresh():
            value = self._physics_mesh_value(physics_name)
            combo.set_value(value)
            swatch.setText(mesh_swatch_html(mesh_color(self._document.spec, value)))
        refresh()
        self._refreshers.append(refresh)
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(swatch)
        row_layout.addWidget(combo, 1)
        return row

    def _physics_mesh_value(self, physics_name):
        """Name of the mesh the physics runs on; its stored (unknown) name when that does not resolve."""
        spec = self._document.spec
        return physics_mesh_name(spec, physics_name) or spec.physics_by_name(physics_name).mesh

    def _set_physics_mesh(self, physics_name, value):
        if value == _NEW_MESH_ENTRY:
            self._document_op(lambda: self._document.add_mesh('tensor', physics_name), f'New mesh for {physics_name}')
            return
        if self._document_op(lambda: self._document.set_physics_mesh(physics_name, value), f'Set {physics_name} mesh'):
            self._schedule(rebuild=True)   # triangle-only settings follow the mesh kind

    def _add_solver_group(self, layout, physics_name, physics_type):
        """'Solver' group: one combo per role the physics uses in the current study (Automatic, then the solver nodes
        filling that role; misfits disabled with the reason); none for a type without solver roles."""
        if not physics_type.solver_roles:
            return
        spec = self._document.spec
        study = spec.study.type
        box = QtWidgets.QGroupBox('Solver')
        form = growing_form(box)
        for role, solver_role in physics_type.solver_roles.items():
            if not solver_role.used_in(study):
                continue
            entries, tips, disabled = self._solver_entries(spec, physics_type, role, solver_role,
                                                           solver_grid(spec, spec.physics_by_name(physics_name)))
            combo = ValueComboBox(entries, self._assigned_solver(physics_name, role),
                                  f'Solver for the {SOLVER_ROLES[role].lower()} of this physics.', item_tips=tips,
                                  disabled=disabled)
            combo.value_committed.connect(partial(self._set_physics_solver, physics_name, role))
            self._refreshers.append(lambda combo=combo, role=role:
                                    combo.set_value(self._assigned_solver(physics_name, role)))
            form.addRow(tip_label(SOLVER_ROLES[role], f'How this physics solves its {SOLVER_ROLES[role].lower()}.'),
                        combo)
        if form.rowCount() == 0:
            form.addRow(hint_label('This physics uses no exchangeable solver in the current study.'))
        layout.addWidget(box)

    def _solver_entries(self, spec, physics_type, role, solver_role, grid):
        """(entries, item tips, disabled reasons) of a physics' role combo on its solver grid kind: Automatic, then the
        solver nodes whose type fills the role (misfits disabled with the solver_misfit reason)."""
        automatic_type_name = solver_role.automatic(solver_backend(spec), grid)[0]
        try:
            default_type = self._registry.solver_type(automatic_type_name)
            automatic_label, automatic_tip = default_type.label, default_type.summary or default_type.description
        except KeyError:
            automatic_label, automatic_tip = automatic_type_name, ''
        automatic_tip += ('\n\nUses the default solver settings and the precision chosen under Solvers; add a solver '
                          'under Solvers to change them, then choose it here.')
        entries, tips, disabled = [(None, f'Automatic ({automatic_label})')], {None: automatic_tip}, {}
        backend = solver_backend(spec)
        for node in spec.solvers:
            try:
                solver_type = self._registry.solver_type(node.type)
            except KeyError:
                continue
            if role not in solver_type.roles:
                continue
            entries.append((node.name, f'{node.name}  ({solver_type.label})'))
            tips[node.name] = solver_type.summary or solver_type.description
            settings = node.settings if isinstance(node.settings, dict) else {}
            values = fill_defaults(solver_type.params, with_default_precision(spec, solver_type, settings, solver_role))
            reason = (solver_misfit(solver_type, values, physics_type, role, spec.study.type, backend, grid)
                      if backend in ('gpu', 'cpu') else '')
            if reason:
                disabled[node.name] = reason
        return entries, tips, disabled

    def _assigned_solver(self, physics_name, role):
        """Solver node assigned to a role of a physics (None = Automatic)."""
        solvers = self._document.spec.physics_by_name(physics_name).solvers
        return (solvers.get(role) or None) if isinstance(solvers, dict) else None

    def _set_physics_solver(self, physics_name, role, solver_name):
        self._document_op(lambda: self._document.set_physics_solver(physics_name, role, solver_name),
                          f'Set {physics_name} solver')

    def _study_label(self, study_name):
        """Label of a registered study type, else its name."""
        try:
            return self._registry.study_type(study_name).label
        except KeyError:
            return str(study_name)

    def _build_feature(self, path, layout):
        physics_name, feature_name = path[1], path[2]
        spec = self._document.spec
        physics = spec.physics_by_name(physics_name)
        feature = physics.feature(feature_name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        try:
            physics_type = self._registry.physics_type(physics.type)
            feature_type = physics_type.feature_type(feature.type)
        except KeyError as exc:
            form.addRow('Type', QtWidgets.QLabel(str(feature.type)))
            layout.addWidget(error_label(error_message(exc)))
            return
        space = instance_space(physics_type, fill_defaults(physics_type.settings, physics.settings), spec.is_3d)
        form.addRow('Type', QtWidgets.QLabel(f"{feature_type.label}  ({feature_type.category.replace('_', ' ')})"))
        form.addRow('', self._enabled_checkbox(
            path, lambda: self._document.spec.physics_by_name(physics_name).feature(feature_name).enabled))
        self._add_summary(layout, feature_type)
        if feature_precision_note(feature_type):
            layout.addWidget(hint_label(feature_precision_note(feature_type)))
        self._add_param_group(layout, 'Parameters', _feature_params(physics, feature_type, space), self._context(),
                              'physics',
                              lambda spec: spec.physics_by_name(physics_name).feature(feature_name).params,
                              feature_name, what='parameter')
        if feature_type.selection_kind is not None:
            picker = SelectionPicker(spec, feature_type.selection_kind, feature.selection,
                                     _slot(self._set_feature_selection, physics_name, feature_name),
                                     allow_outer_faces=feature_type.allow_outer_faces,
                                     allow_interior=feature_type.allow_interior,
                                     object_kinds=feature_type.object_kinds,
                                     max_entities=feature_type.max_entities)
            layout.addWidget(_group('Selection', hint_label(_selection_hint(feature_type)), picker))
            self._refreshers.append(lambda: picker.set_selected(
                self._document.spec.physics_by_name(physics_name).feature(feature_name).selection))
        self._add_technical_details(layout, f'feature:{physics.type}:{feature_type.name}', feature_type)

    def _build_couplings_root(self, path, layout):
        """Intro, then one row (label, summary, Add button) per coupling type and physics pair that fits the model."""
        spec = self._document.spec
        layout.addWidget(hint_label(_COUPLINGS_INTRO))
        candidates = coupling_candidates(spec, self._registry)
        for coupling_type, pairs in candidates:
            for source, target in pairs:
                layout.addWidget(self._type_adder_row(
                    f'{coupling_type.label}  ({source} → {target})', coupling_type,
                    partial(self._document.add_coupling, coupling_type.name, source, target)))
        if candidates:
            return
        if not spec.physics:
            layout.addWidget(hint_label('A coupling needs two physics, such as a wave physics and a heat physics. Add '
                                        'them first: right-click Model in the model tree and choose Add physics.'))
        elif len(spec.physics) == 1:
            layout.addWidget(hint_label('A coupling needs two physics, such as a wave physics and a heat physics. Add '
                                        'a second physics first: right-click Model in the model tree and choose Add '
                                        'physics.'))
        else:
            layout.addWidget(hint_label('None of the couplings fits the physics in this model.'))

    def _build_coupling(self, path, layout):
        name = path[1]
        spec = self._document.spec
        coupling = spec.coupling(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        try:
            coupling_type = self._registry.coupling_type(coupling.type)
        except KeyError as exc:
            form.addRow('Type', QtWidgets.QLabel(str(coupling.type)))
            layout.addWidget(error_label(error_message(exc)))
            return
        form.addRow('Type', QtWidgets.QLabel(coupling_type.label))
        for role, row_label in (('source', 'Source'), ('target', 'Target')):
            combo = ValueComboBox(self._physics_entries(spec, coupling_type, role), getattr(coupling, role),
                                  _COUPLING_ROLE_TIPS[role])
            combo.value_committed.connect(_slot(self._set_coupling_attribute, name, role))
            self._refreshers.append(lambda combo=combo, role=role:   # entries stay (a physics change rebuilds)
                                    combo.set_value(getattr(self._document.spec.coupling(name), role)))
            form.addRow(tip_label(row_label, _COUPLING_ROLE_TIPS[role]), combo)
        if several_meshes(spec):
            meshes_label = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)

            def refresh_meshes():
                current = self._document.spec
                meshes_label.setText(coupling_mesh_text(current, self._registry, current.coupling(name)) or '–')
            refresh_meshes()
            self._refreshers.append(refresh_meshes)
            form.addRow(tip_label('Meshes', 'Meshes of the source and the target physics. Across two meshes the '
                                             'coupling carries its field (interpolated) or its source density '
                                             '(conservative: the total is kept) from one mesh to the other.'),
                        meshes_label)
        every = IntSpinBox(0, _MAX_EVERY, coupling.every, special_text='Automatic', help_text=_COUPLING_EVERY_TIP)
        every.value_committed.connect(_slot(self._set_coupling_attribute, name, 'every'))
        self._refreshers.append(lambda: every.refresh_value(self._document.spec.coupling(name).every))
        form.addRow(tip_label('Transfer every [time steps]', _COUPLING_EVERY_TIP), every)
        form.addRow('', self._enabled_checkbox(path, lambda: self._document.spec.coupling(name).enabled))
        self._add_summary(layout, coupling_type)
        self._add_param_group(layout, 'Parameters', self._coupling_params(spec, coupling, coupling_type),
                              self._context(expression_names=('value',)), 'couplings',
                              lambda spec: spec.coupling(name).params, name, what='parameter')
        self._add_technical_details(layout, f'coupling:{coupling_type.name}', coupling_type)

    def _physics_type_named(self, spec, physics_name):
        """Registered PhysicsType of the physics called physics_name, or None."""
        try:
            return self._registry.physics_type(spec.physics_by_name(physics_name).type)
        except KeyError:
            return None

    def _coupling_params(self, spec, coupling, coupling_type):
        """coupling_type.params with the text params 'source_field' / 'target_slot' turned into drop-downs of the
        source physics' state fields / the target physics' source slots (unchanged when there are none)."""
        params = []
        for param in coupling_type.params:
            entries = ()   # (value, drop-down text, tip list line)
            if param.kind == 'str' and param.name == 'source_field':
                physics_type = self._physics_type_named(spec, coupling.source)
                if physics_type is not None:
                    entries = tuple((info.name, f'{info.label}  ({info.name})',
                                     f'{info.label}  ({info.name})' + (f' [{info.unit}]' if info.unit else ''))
                                    for info in physics_type.state_fields)
            elif param.kind == 'str' and param.name == 'target_slot':
                physics_type = self._physics_type_named(spec, coupling.target)
                if physics_type is not None:
                    entries = tuple((slot, _slot_display(slot, meaning),
                                     _slot_display(slot, meaning) + (f' [{unit}]' if unit not in ('', '1') else ''))
                                    for slot, (unit, meaning) in physics_type.source_slots.items())
            if entries:
                choices = tuple(value for value, _text, _line in entries)
                param = replace(param, kind='choice', choices=choices,
                                default=param.default if param.default in choices else None,
                                choice_labels=tuple((value, text) for value, text, _line in entries),
                                tip=(param.tip or param.help) + '\n\n'
                                    + '\n'.join(f'- {line}' for _value, _text, line in entries))
            params.append(param)
        return tuple(params)

    def _physics_entries(self, spec, coupling_type, role):
        """(name, label) of the physics in the spec whose type the coupling type accepts in `role`."""
        entries = []
        for physics in spec.physics:
            if not coupling_type_accepts(coupling_type, role, physics.type):
                continue
            try:
                type_label = self._registry.physics_type(physics.type).label
            except KeyError:
                type_label = str(physics.type)
            entries.append((physics.name, f'{physics.name}  ({type_label})'))
        return entries

    def _form_mesh(self, spec):
        """MeshSpec of the shown mesh form, looked up by the form's node name (KeyError when it is gone)."""
        return spec.mesh_by_name(self._node_path[1])

    def _build_meshes_root(self, path, layout):
        """Intro, one line per mesh (swatch, kind, size, the physics on it), Add mesh buttons and the model's memory
        estimate (each mesh node shows its own part, the over-budget issue and its override)."""
        layout.addWidget(hint_label(_MESHES_INTRO))
        overview = wrapped_label(text_format=QtCore.Qt.TextFormat.RichText)
        layout.addWidget(_group('Meshes', overview))
        buttons = []
        spec = self._document.spec
        for kind in MESH_KINDS:
            if not mesh_kind_shown(spec, kind):
                continue
            button = QtWidgets.QPushButton(f'Add mesh: {MESH_KIND_LABELS.get(kind, kind).lower()}')
            problem = mesh_kind_problem(spec, kind)
            button.setEnabled(not problem)
            if problem:
                button.setToolTip(problem)
            _on_click(button, partial(self._document_op, partial(self._document.add_mesh, kind), 'Add mesh'))
            buttons.append(button)
        layout.addLayout(_button_row(*buttons))
        memory_label = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(_group('Estimate (whole model)', memory_label))

        def refresh():
            overview.setText(self._meshes_overview_html())
            memory_label.setText(memory_estimate_text(
                self._document.memory_estimate(), self._document.memory_budget, self._document.host_memory_budget,
                solver_backend(self._document.spec)) + '\n' + _time_estimate_text(*self._document.time_estimate()))
        refresh()
        self._refreshers.append(refresh)

    def _meshes_overview_html(self):
        """One rich-text line per mesh: swatch, name, kind and size, the physics on it (the first: default)."""
        spec = self._document.spec
        lines = []
        for index, mesh_spec in enumerate(spec.meshes):
            users = ', '.join(physics.name for physics in spec.physics_on(mesh_spec.name))
            estimate = self._document.mesh_estimate(mesh_spec.name)[0]
            default = ' (default)' if index == 0 else ''
            lines.append(f'{mesh_swatch_html(mesh_color(spec, mesh_spec.name))} '
                         f'<b>{html.escape(mesh_spec.name)}</b>{default}: '
                         f'{html.escape(mesh_summary_text(mesh_spec, estimate))}; used by '
                         f'{html.escape(users) if users else "<i>no physics (not built)</i>"}')
        return '<br>'.join(lines)

    def _build_mesh(self, path, layout):
        spec = self._document.spec
        mesh_spec = spec.mesh_by_name(path[1])
        badge = wrapped_label(text_format=QtCore.Qt.TextFormat.RichText)
        layout.addWidget(badge)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        users = CheckList(self._mesh_user_entries(spec),
                          [physics.name for physics in spec.physics_on(mesh_spec.name)], _MESH_USERS_TIP)
        users.value_committed.connect(self._set_mesh_users)
        form.addRow(tip_label('Used by', _MESH_USERS_TIP), users)

        def refresh_users():
            current = self._document.spec
            mesh_name = self._form_mesh(current).name
            users.set_value([physics.name for physics in current.physics_on(mesh_name)])
            default = current.meshes[0].name == mesh_name
            on_mesh = [physics.name for physics in current.physics_on(mesh_name)]
            badge.setText(mesh_badge_html(current, mesh_name, ('default mesh; ' if default else '')
                                          + (f"used by {', '.join(on_mesh)}" if on_mesh else 'unused: not built')))
        refresh_users()
        self._refreshers.append(refresh_users)
        ctx = self._context()
        kinds = tuple(kind for kind in MESH_KINDS if mesh_kind_shown(spec, kind) or kind == mesh_spec.kind)
        mesh_params = tuple(replace(param, choices=kinds, choice_labels=tuple(entry for entry in param.choice_labels
                                                                              if entry[0] in kinds))
                            if param.name == 'kind' else param for param in MESH_PARAMS)
        mesh_form = ParamForm(mesh_params, self._mesh_values(), ctx, self._set_mesh_value)
        mesh_form.set_placeholder('outer_dx', 'off')
        mesh_form.set_placeholder('h_min', 'maximum size / 100')   # its number: _show_mesh_numbers
        layout.addWidget(_group('Mesh', mesh_form))

        def refresh_mesh_form():
            self._show_mesh_numbers(mesh_form)
            mesh_form.refresh(self._mesh_values())
        refresh_mesh_form()
        self._refreshers.append(refresh_mesh_form)
        if mesh_spec.mode == 'auto':   # a mode edit rebuilds the form
            layout.addWidget(self._automatic_sizes_group())
        size_points_label = wrapped_label()   # MeshSpec.size_points, read only (adaptive refinement writes them)
        size_points_label.setToolTip(rich_tooltip(
            'Element sizes the adaptive mesh refinement placed at the nodes of the elements it refined (kept with '
            'the mesh, e.g. by Keep adapted mesh). Each one makes the triangles near that point at most that size.'))
        layout.addWidget(size_points_label)

        def refresh_size_points():
            count = len(self._form_mesh(self._document.spec).size_points)
            size_points_label.setText(f'Size points: {count:,} from mesh refinement')
            size_points_label.setVisible(count > 0)
        refresh_size_points()
        self._refreshers.append(refresh_size_points)
        graded_boxes = [self._refinements_group(spec, mesh_spec, ctx)]
        if _tensor_graded(mesh_spec) or mesh_spec.axes:   # elsewhere shown only to remove them
            graded_boxes.append(self._mesh_axes_group(spec, mesh_spec, ctx))
        for graded_box in graded_boxes:
            layout.addWidget(graded_box)
            self._refreshers.append(partial(
                lambda box: box.setEnabled(self._form_mesh(self._document.spec).mode in _REFINED_MESH_MODES),
                graded_box))
        methods_label = wrapped_label(text_format=QtCore.Qt.TextFormat.RichText)
        methods_group = _group('Methods on this mesh', methods_label)
        layout.addWidget(methods_group)

        def refresh_methods():
            text = self._mesh_methods_html(self._form_mesh(self._document.spec))
            methods_label.setText(text)
            methods_group.setVisible(bool(text))
        refresh_methods()
        self._refreshers.append(refresh_methods)
        # top-down as it is used: Build mesh (element meshes), the estimate it makes final, then the over-budget
        # override, right above the node's issues
        if mesh_spec.kind in FREE_MESH_KINDS and self._mesh_store is not None:
            layout.addWidget(self._element_build_group(mesh_spec.kind))
        estimate_label = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        override_form = ParamForm((_MEMORY_OVERRIDE,), self._memory_override_values(), self._context(),
                                  self._set_memory_override)
        layout.addWidget(_group('Estimate', estimate_label, override_form))

        def refresh_estimate():
            self._update_mesh_estimate(estimate_label)
            override_form.refresh(self._memory_override_values())
            override_form.setVisible(self._memory_override_offered())
        refresh_estimate()
        self._refreshers.append(refresh_estimate)

    def _memory_override_values(self):
        return {'memory_override': memory_check_level(self._document.spec) != 'error'}

    def _set_memory_override(self, _name, overridden):
        """The mesh form's tick box: solver_defaults.memory_check 'warning' (ticked) / 'error'."""
        self._set_solver_default('memory_check', 'warning' if overridden else 'error')

    def _memory_override_offered(self) -> bool:
        """True on the form of the mesh the memory budget issue is reported at (document.memory_verdict), while the
        estimate is over a budget or the override is on (so it can be taken back)."""
        verdict = self._document.memory_verdict()
        if verdict is None or verdict.mesh != self._form_mesh(self._document.spec).name:
            return False
        return verdict.over or memory_check_level(self._document.spec) != 'error'

    def _mesh_methods_html(self, mesh_spec):
        """'Methods on this mesh' rich text: per physics on the mesh whose type has a mesh_method, how it solves on this
        mesh kind, then what that method cannot handle here (warning colour); '' when none has one."""
        spec = self._document.spec
        parameters = self._document.parameters()
        lines = []
        for physics in spec.physics_on(mesh_spec.name):
            try:
                physics_type = self._registry.physics_type(physics.type)
            except KeyError:   # reported by validation
                continue
            if physics_type.mesh_method is None:
                continue
            settings = fill_defaults(physics_type.settings, physics.settings if isinstance(physics.settings, dict)
                                     else {})
            text, problems = physics_type.mesh_method(spec, physics, settings, mesh_spec.kind, parameters)
            lines.append(f'<b>{html.escape(physics.name)}</b> ({html.escape(physics_type.label)}): '
                         f'{html.escape(text)}')
            lines.extend(f'<span style="color:{_LEVEL_COLORS["warning"]}"><b>Warning:</b></span> '
                         f'{html.escape(problem)}' for problem in problems)
        return '<br>'.join(lines)

    def _show_mesh_numbers(self, mesh_form):
        """Numbers of the mesh form's derived and empty fields: the defaults and resolved values as placeholders, an
        empty auto cell size / minimum element size as '= value'; the wavelength rows greyed out outside auto mode
        (points per wavelength also under a rule without a wavelength or while the cell size is set)."""
        document = self._document
        mesh_spec = self._form_mesh(document.spec)
        estimate, _error = document.mesh_estimate(mesh_spec.name)
        try:
            resolved = mesh_values(mesh_spec, document.parameters())
        except (ValueError, KeyError, TypeError, ExpressionError):   # reported by the editors / validation
            resolved = {}
        auto = mesh_spec.mode == 'auto'
        # auto: dx is the smallest cell size, empty = the rule alone (read back as its finest size)
        auto_sizes = auto and estimate is not None and estimate.auto_dx is not None
        mesh_form.set_placeholder('dx', 'from the automatic rule' if auto else 'required')
        mesh_form.set_empty_value('dx', _smallest_automatic_size(estimate) if auto_sizes else None)
        # the physics-controlled / uniform-flow rules resolve no wavelength (a wavelength_min value switches to it)
        by_wavelength = estimate is None or estimate.auto_rule in ('', 'wavelength')
        mesh_form.set_row_enabled('wavelength_min', auto)
        mesh_form.set_row_enabled('points_per_wavelength', auto and by_wavelength and mesh_spec.dx is None)
        kind_defaults = MESH_KIND_DEFAULTS.get(mesh_spec.kind, MESH_KIND_DEFAULTS['tensor'])
        for name in ('max_ratio', 'points_per_wavelength'):
            mesh_form.set_placeholder(name, f'{kind_defaults[name]:g}  (default)')
        max_ratio = resolved.get('max_ratio')
        mesh_form.set_placeholder('outer_ratio', 'the maximum growth ratio' if max_ratio is None
                                  else f'{max_ratio:g}  (the maximum growth ratio)')
        wavelength = estimate.wavelength_min if auto and estimate is not None else None
        if wavelength is not None:
            wavelength_text = f"{format_quantity(wavelength, 'm')}  (from the physics '{estimate.wavelength_source}')"
        elif auto and not by_wavelength:
            wavelength_text = 'none: no wave physics on this mesh (a value sizes the cells by it)'
        else:
            wavelength_text = 'from the physics'
        mesh_form.set_placeholder('wavelength_min', wavelength_text)
        # an empty minimum element size reads back '= maximum size / 100' (the sizing rule): the maximum from the
        # estimate, else (estimate failed, e.g. too many elements) the cell size itself outside auto mode
        maximum_size = None
        if estimate is not None and estimate.kind in FREE_MESH_KINDS:
            maximum_size = estimate.sizing.h_max
        elif mesh_spec.kind in FREE_MESH_KINDS and not auto:
            maximum_size = resolved.get('dx')
        mesh_form.set_empty_value('h_min', None if maximum_size is None else maximum_size / 100.0)

    def _automatic_sizes_group(self):
        """'Automatic sizes' of an auto-mode mesh form: the base and object sizes its rule chose (read only) and Edit
        these sizes (Graded, or Uniform for lattice Boltzmann flow, with them filled in)."""
        sizes_label = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        button = QtWidgets.QPushButton('Edit these sizes')
        button.setToolTip(rich_tooltip(_EDIT_AUTOMATIC_SIZES_TIP))
        box = QtWidgets.QGroupBox('Automatic sizes')
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.addWidget(sizes_label)
        box_layout.addLayout(_button_row(button))

        def current_estimate():
            return self._document.mesh_estimate(self._form_mesh(self._document.spec).name)

        def refresh():
            estimate, error = current_estimate()
            button.setEnabled(estimate is not None and estimate.auto_dx is not None)
            if error:
                sizes_label.setText(f'Not available: {error}')
            elif estimate is None or estimate.auto_dx is None:
                sizes_label.setText('Not available: no mesh estimate.')
            else:
                sizes_label.setText(_automatic_sizes_text(estimate))

        def edit():
            estimate, _error = current_estimate()
            uniform = estimate is not None and estimate.auto_rule == 'uniform_flow'
            self._set_mesh_mode('uniform' if uniform else 'nonuniform')
        _on_click(button, edit)
        refresh()
        self._refreshers.append(refresh)
        return box

    def _element_build_group(self, kind):
        """'Triangles' / 'Tetrahedra' group of an element mesh form: Build mesh (the mesher alone, on the worker;
        the industry-standard build-all step) and the state of the elements the canvas shows (tetrahedra: cut by its
        shown plane)."""
        noun = 'tetrahedra' if kind == 'tetrahedra' else 'triangles'
        auto_limit = AUTO_PREVIEW_MAX_TETRAHEDRA if kind == 'tetrahedra' else AUTO_PREVIEW_MAX_ELEMENTS
        button = QtWidgets.QPushButton('Build mesh')
        button.setToolTip(rich_tooltip(
            f'Make the {noun} of this mesh now and show them in the mesh view, without building the model'
            + (' (cut by the shown layer or cut plane).' if kind == 'tetrahedra' else '.')
            + f'\n\nBuilding the model makes the same {noun} (the mesher gives the same mesh for the same model).'))
        status = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        box = QtWidgets.QGroupBox(noun.capitalize())
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.addLayout(_button_row(button))
        box_layout.addWidget(status)

        def build():
            refusal = self._mesh_store.request_preview(self._form_mesh(self._document.spec).name, log=True)
            if refusal:
                self._report(refusal)

        def refresh():
            store, mesh_name = self._mesh_store, self._form_mesh(self._document.spec).name
            drawing = store.drawing(mesh_name)
            button.setEnabled(not store.pending(mesh_name))
            failure = store.failure(mesh_name)
            if store.pending(mesh_name):
                text = 'Meshing…'
            elif failure:
                text = f'Meshing failed: {failure}'
            elif drawing is not None:
                data = drawing.lines
                source = 'built with the model' if drawing.source == 'built' else 'Build mesh'
                if drawing.kind == 'tetrahedra':
                    quality_text = (f', smallest radius ratio {data.smallest_quality:.2g}'
                                    + (f' ({data.n_poor:,} poor elements)' if data.n_poor else ''))
                else:
                    angle = data.quality.get('min_angle')
                    quality_text = (f', smallest angle {float(angle):.1f}°'
                                    if isinstance(angle, (int, float)) and math.isfinite(angle) else '')
                text = f'{data.n_elements:,} {noun}, {data.n_points:,} nodes{quality_text} ({source}).'
            elif store.outdated(mesh_name):
                text = f'The {noun} made earlier no longer match this mesh and model: build the mesh again.'
            else:
                text = (f'Not meshed yet: the mesh view meshes it when shown (up to {auto_limit:,} estimated '
                        'elements), else press Build mesh.')
            status.setText(text)
        _on_click(button, build)
        refresh()
        self._refreshers.append(refresh)
        return box

    def _mesh_user_entries(self, spec):
        """(physics name, 'name (type label)', tooltip naming its current mesh) of every physics of the model."""
        entries = []
        for physics in spec.physics:
            physics_type = self._physics_type_named(spec, physics.name)
            type_label = physics_type.label if physics_type is not None else str(physics.type)
            current = physics_mesh_name(spec, physics) or physics.mesh
            entries.append((physics.name, f'{physics.name}  ({type_label})', f"Now on mesh '{current}'."))
        return entries

    def _set_mesh_users(self, checked):
        """Move newly ticked physics onto the form's mesh and unticked ones to the default mesh (one edit)."""
        def apply(spec):
            mesh_name = self._form_mesh(spec).name
            default = spec.meshes[0].name
            on_mesh = {physics.name for physics in spec.physics_on(mesh_name)}
            for physics in spec.physics:
                if physics.name in checked and physics.name not in on_mesh:
                    physics.mesh = '' if mesh_name == default else mesh_name
                elif physics.name not in checked and physics.name in on_mesh:
                    if mesh_name == default:
                        raise ValueError(f"'{physics.name}' is on the default mesh: choose another mesh in its Mesh "
                                         'setting to move it')
                    physics.mesh = ''
        self._commit('physics', apply, 'Set the physics of a mesh', rebuild=True)

    def _refinements_group(self, spec, mesh_spec, ctx):
        box = QtWidgets.QGroupBox('Refinements (graded and automatic modes)')
        grid = QtWidgets.QGridLayout(box)
        grid.addWidget(tip_label('Object', _REFINEMENT_OBJECT_TIP), 0, 0)
        for column, param in enumerate((_REFINEMENT_DX, _REFINEMENT_MARGIN), start=1):
            grid.addWidget(param_row_label(param), 0, column)
        object_entries = [(geometry_object.name, geometry_object.name) for geometry_object in spec.geometry]
        gmsh_tetrahedra = mesh_spec.kind == 'tetrahedra' and mesh_spec.mesher == 'gmsh'
        for index, refinement in enumerate(mesh_spec.refinements):
            # main row, then the per-axis sizes (tensor meshes, or while set), then where (gmsh tetrahedra, or while set)
            row = 3 * index + 1
            object_name = refinement.object
            if _tensor_graded(mesh_spec) or any(getattr(refinement, param.name) is not None
                                                for param in _REFINEMENT_AXIS_DX.values()):
                grid.addWidget(tip_label('Per axis', _REFINEMENT_AXIS_TIP), row + 1, 0,
                               QtCore.Qt.AlignmentFlag.AlignRight)
                grid.addWidget(self._refinement_axis_row(spec, index, refinement, ctx), row + 1, 1, 1, 2)
            if gmsh_tetrahedra or refinement.where != 'inside':
                grid.addWidget(tip_label('Refine', _REFINEMENT_WHERE_TIP), row + 2, 0,
                               QtCore.Qt.AlignmentFlag.AlignRight)
                where_combo = ValueComboBox(_REFINEMENT_WHERE_ENTRIES, refinement.where, _REFINEMENT_WHERE_TIP)
                where_combo.value_committed.connect(_slot(self._set_refinement, index, object_name, 'where'))
                grid.addWidget(where_combo, row + 2, 1, 1, 2)
            combo = ValueComboBox(object_entries, object_name, _REFINEMENT_OBJECT_TIP)
            combo.value_committed.connect(_slot(self._set_refinement, index, object_name, 'object'))
            grid.addWidget(combo, row, 0)
            for column, (param, attribute) in enumerate(((_REFINEMENT_DX, 'dx'), (_REFINEMENT_MARGIN, 'margin')),
                                                        start=1):
                editor = self._expression_edit(
                    param, getattr(refinement, attribute), ctx,
                    _slot(self._set_refinement, index, object_name, attribute),
                    'three times the cell size' if attribute == 'margin' else '')
                grid.addWidget(editor, row, column)
            grid.addWidget(_remove_button(partial(self._remove_refinement, index, object_name)), row, 3)
            if refinement.automatic:
                grid.addWidget(tip_label('automatic', _AUTOMATIC_REFINEMENT_TIP), row, 4)
        add_button = QtWidgets.QPushButton('Add refinement')
        add_button.setEnabled(bool(spec.geometry))
        add_button.setToolTip('' if spec.geometry else 'Add a geometry object first')
        _on_click(add_button, self._add_refinement)
        grid.addWidget(add_button, 3 * len(mesh_spec.refinements) + 1, 0, 1, 4)
        if mesh_spec.mode != 'auto' and any(refinement.automatic for refinement in mesh_spec.refinements):
            reset_button = QtWidgets.QPushButton('Reset to automatic sizes')
            reset_button.setToolTip(rich_tooltip(_RESET_AUTOMATIC_SIZES_TIP))
            _on_click(reset_button, partial(self._set_mesh_mode, 'auto'))
            grid.addWidget(reset_button, 3 * len(mesh_spec.refinements) + 2, 0, 1, 4)
        box.setEnabled(mesh_spec.mode in _REFINED_MESH_MODES)
        return box

    def _refinement_axis_row(self, spec, index, refinement, ctx):
        """Row widget of one refinement's dx_x / dx_y (/ dx_z in 3D or while set) editors."""
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for axis in _MESH_AXES:
            param = _REFINEMENT_AXIS_DX[axis]
            value = getattr(refinement, param.name)
            if axis == 'z' and not spec.is_3d and value is None:
                continue
            axis_label = QtWidgets.QLabel(axis)
            set_param_help(axis_label, param)
            row_layout.addWidget(axis_label)
            editor = self._expression_edit(param, value, ctx,
                                           _slot(self._set_refinement, index, refinement.object, param.name),
                                           'cell size')
            row_layout.addWidget(editor, 1)
        return row_widget

    def _mesh_axes_group(self, spec, mesh_spec, ctx):
        """MeshSpec.axes: one box per entry (growth ratio, cells mode, bands, edges / segments), Add buttons (tensor
        meshes and builtin tetrahedra only)."""
        box = QtWidgets.QGroupBox('Per-axis options (graded and automatic modes)')
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.addWidget(hint_label('Options for one axis: its own growth ratio, finer bands between two '
                                        'coordinates, or cells given directly.'
                                        + (' Rectangular cells and built-in tetrahedra only: remove them on this '
                                           'mesh.' if not _tensor_graded(mesh_spec) else '')))
        for index, axis_spec in enumerate(mesh_spec.axes):
            box_layout.addWidget(self._mesh_axis_box(index, axis_spec, ctx))
        present = {axis_spec.axis for axis_spec in mesh_spec.axes}
        buttons = []
        for axis in (_MESH_AXES if spec.is_3d else _MESH_AXES[:2]):
            if axis not in present and _tensor_graded(mesh_spec):
                button = QtWidgets.QPushButton(f'Add {axis} options')
                _on_click(button, partial(self._add_mesh_axis, axis))
                buttons.append(button)
        box_layout.addLayout(_button_row(*buttons))
        box.setEnabled(mesh_spec.mode in _REFINED_MESH_MODES)
        return box

    def _mesh_axis_box(self, index, axis_spec, ctx):
        axis = axis_spec.axis
        box = QtWidgets.QGroupBox(f'Axis {axis}')
        box_layout = QtWidgets.QVBoxLayout(box)
        form = growing_form()
        box_layout.addLayout(form)
        ratio_editor = self._expression_edit(_AXIS_MAX_RATIO, axis_spec.max_ratio, ctx,
                                             _slot(self._set_mesh_axis, index, axis, 'max_ratio'),
                                             'the mesh maximum growth ratio')
        form.addRow(param_row_label(_AXIS_MAX_RATIO), ratio_editor)
        mode = _axis_cell_mode(axis_spec)
        mode_combo = ValueComboBox(_AXIS_CELL_MODES, mode, _AXIS_CELLS_TIP)
        mode_combo.value_committed.connect(_slot(self._set_mesh_axis_cells, index, axis))
        form.addRow(tip_label('Cells', _AXIS_CELLS_TIP), mode_combo)
        if mode == 'edges':
            edges_editor = make_param_editor(
                _AXIS_EDGES, axis_spec.edges, ctx,
                lambda _name, value: self._set_mesh_axis(index, axis, 'edges', value, rebuild=not value))
            self._refreshers.append(edges_editor.refresh_check)
            form.addRow(param_row_label(_AXIS_EDGES), edges_editor)
        elif mode == 'segments':
            box_layout.addWidget(self._mesh_segments_widget(index, axis, axis_spec.segments, ctx))
        bands = self._mesh_bands_widget(index, axis, axis_spec.bands, ctx)
        bands.setEnabled(mode == 'graded')
        if mode != 'graded':
            bands.setToolTip('Bands are not used while the cells of this axis are given directly.')
        box_layout.addWidget(bands)
        remove_button = QtWidgets.QPushButton(f'Remove {axis} options')
        _on_click(remove_button, partial(self._remove_mesh_axis, index, axis))
        box_layout.addWidget(remove_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        return box

    def _mesh_bands_widget(self, index, axis, bands, ctx):
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(tip_label('Refined bands', 'Bands between two coordinates of this axis with a finer cell '
                                                   'size; only this axis is refined there.'), 0, 0, 1, 4)
        for column, param in enumerate(_BAND_PARAMS):
            grid.addWidget(param_row_label(param), 1, column)
        for band_index, band in enumerate(bands):
            row = band_index + 2
            for column, param in enumerate(_BAND_PARAMS):
                editor = self._expression_edit(param, getattr(band, param.name), ctx,
                                               _slot(self._set_mesh_axis_item, index, axis, 'bands', band_index,
                                                     param.name))
                grid.addWidget(editor, row, column)
            grid.addWidget(_remove_button(partial(self._remove_mesh_axis_item, index, axis, 'bands', band_index)),
                           row, len(_BAND_PARAMS))
        add_button = QtWidgets.QPushButton('Add band')
        _on_click(add_button, partial(self._add_mesh_band, index, axis))
        grid.addWidget(add_button, len(bands) + 2, 0, 1, len(_BAND_PARAMS) + 1)
        return widget

    def _mesh_segments_widget(self, index, axis, segments, ctx):
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        end_param, cells_param, ratio_param = _SEGMENT_PARAMS
        headers = (param_row_label(end_param), param_row_label(cells_param), tip_label('Spacing',
                   _SEGMENT_PROFILE_TIP), param_row_label(ratio_param))
        for column, header in enumerate(headers):
            grid.addWidget(header, 0, column)
        profile_entries = [(profile, _SEGMENT_PROFILE_LABELS.get(profile, profile)) for profile in SEGMENT_PROFILES]
        for segment_index, segment in enumerate(segments):
            row = segment_index + 1
            for column, param in ((0, end_param), (1, cells_param), (3, ratio_param)):
                editor = self._expression_edit(
                    param, getattr(segment, param.name), ctx,
                    _slot(self._set_mesh_axis_item, index, axis, 'segments', segment_index, param.name),
                    'axis end' if param is end_param else '')
                grid.addWidget(editor, row, column)
            profile_combo = ValueComboBox(profile_entries, segment.profile, _SEGMENT_PROFILE_TIP)
            profile_combo.value_committed.connect(
                _slot(self._set_mesh_axis_item, index, axis, 'segments', segment_index, 'profile'))
            grid.addWidget(profile_combo, row, 2)
            grid.addWidget(_remove_button(partial(self._remove_mesh_axis_item, index, axis, 'segments',
                                                  segment_index)), row, 4)
        add_button = QtWidgets.QPushButton('Add segment')
        _on_click(add_button, partial(self._add_mesh_segment, index, axis))
        grid.addWidget(add_button, len(segments) + 1, 0, 1, 5)
        return widget

    def _line_block_fits(self):
        """True when a line-block solver could take a job of a physics in this model and study."""
        try:
            solver_type = self._registry.solver_type('line_block')
        except KeyError:
            return False
        values = fill_defaults(solver_type.params, with_default_precision(self._document.spec, solver_type, {}))
        return any(not reason for _physics, _role, reason
                   in solver_fits(self._document.spec, self._registry, solver_type, values))

    def _update_mesh_estimate(self, label):
        """Estimate of the form's mesh: size, its memory part and the whole model's, sizes, absorbing layer, step."""
        document = self._document
        mesh_name = self._form_mesh(document.spec).name
        estimate, error = document.mesh_estimate(mesh_name)
        if error:
            label.setText(f'Mesh estimate failed: {error}')
            mark_invalid(label, error)
            return
        mark_invalid(label, None)
        if estimate is None:
            label.setText('No mesh estimate.')
            return
        backend = solver_backend(document.spec)
        used = mesh_name in {mesh_spec.name for mesh_spec in document.spec.used_meshes()}
        memory_lines = ([memory_estimate_text(document.mesh_memory_estimate(mesh_name), document.memory_budget,
                                               document.host_memory_budget, backend,
                                               scope='This mesh and its physics:')]
                        if used and several_meshes(document.spec) else [])
        memory_lines.append(memory_estimate_text(document.memory_estimate(), document.memory_budget,
                                                  document.host_memory_budget, backend,
                                                  scope='Whole model:' if several_meshes(document.spec)
                                                  else 'Estimated'))
        memory_lines.append(_time_estimate_text(*document.time_estimate()))
        if not used:
            memory_lines.insert(0, 'Not built: no physics runs on this mesh.')
        if estimate.kind == 'triangles':
            sizing = estimate.sizing
            lines = [f'{estimate.n_elements:,} triangles, {estimate.n_nodes:,} nodes (the built mesh)'
                     if estimate.counts_built else
                     f'About {estimate.n_elements:,} triangles, {estimate.n_nodes:,} nodes (estimate from the sizes)',
                     *memory_lines,
                     f"Element size {format_quantity(sizing.h_min, 'm')} (minimum) to "
                     f"{format_quantity(sizing.h_max, 'm')} (maximum), growth rate {sizing.growth:g}",
                     'The memory estimate is that of these triangles.' if estimate.counts_built else
                     'The triangles themselves are made by Build mesh or when the model is built; the memory '
                     'estimate then uses their real number.']
        elif estimate.kind == 'tetrahedra':
            sizing = estimate.sizing
            swept = bool(sizing.sweep_axis)
            builtin = sizing.mesher == 'builtin' and not swept
            counted = ('the built mesh' if estimate.counts_built else
                       'exact: grid boxes split into 6' if builtin else
                       f'estimate: triangles x 3 x {len(sizing.layer_edges) - 1} z layers' if swept
                       else 'estimate from the sizes')
            # builtin: its finest grid cell (h_min unused); swept: the thinnest layer or smallest triangle
            smallest = estimate.dx_min if builtin or swept else sizing.h_min
            lines = [f"{'' if estimate.counts_built else 'About '}{estimate.n_elements:,} tetrahedra, "
                     f'{estimate.n_nodes:,} nodes, {estimate.n_edges:,} edges ({counted})', *memory_lines,
                     f"Element size {format_quantity(smallest, 'm')} (minimum) to "
                     f"{format_quantity(sizing.h_max, 'm')} (maximum), growth rate {sizing.growth:g}",
                     'The memory estimate is that of these tetrahedra.' if estimate.counts_built else
                     'The tetrahedra themselves are made by Build mesh or when the model is built; the memory '
                     'estimate then uses their real number.']
        else:
            z_text = f' × {estimate.nz}' if estimate.dz_cells is not None else ''   # 3D
            lines = [f'Grid {estimate.nx} × {estimate.ny}{z_text} = {estimate.n_cells:,} cells', *memory_lines,
                     f"Cell size {format_quantity(estimate.dx_min, 'm')} to {format_quantity(estimate.dx_max, 'm')} "
                     f"({'uniform' if estimate.uniform else 'graded'})",
                     f'Absorbing layer: {face_cell_counts_text(estimate.npml_sides)} cells',
                     f"Time step {format_quantity(estimate.dt, 's')}" if estimate.dt is not None
                     else 'No time step (no physics on this mesh runs in time)']
        if estimate.wavelength_min is not None:   # auto mode
            origin = ("from the 'Shortest wavelength' setting" if estimate.wavelength_source == 'mesh'
                      else f"from the physics '{estimate.wavelength_source}'")
            lines.append(f"Shortest wavelength {format_quantity(estimate.wavelength_min, 'm')} ({origin})")
        if estimate.note:   # why the grid differs from the requested mode
            lines.append(estimate.note)
        if estimate.custom_axes:
            lines.append(f"Cells given directly along {', '.join(estimate.custom_axes)}")
        hint = line_block_hint(estimate)
        if hint and self._line_block_fits():
            lines.append(hint)
        label.setText('\n'.join(lines))

    def _build_solvers_root(self, path, layout):
        """solver_defaults (backend), the solver each physics role uses now, and one Add row per fitting type."""
        spec = self._document.spec
        layout.addWidget(hint_label(_SOLVERS_INTRO))
        def default_values():
            defaults = self._document.spec.solver_defaults
            return {'backend': defaults.backend, 'precision': defaults.precision,
                    'subpixel_smoothing': defaults.subpixel_smoothing,
                    'conformal_boundaries': defaults.conformal_boundaries}
        defaults_form = ParamForm((_SOLVER_BACKEND, _SOLVER_PRECISION, _SUBPIXEL_SMOOTHING, _CONFORMAL_BOUNDARIES),
                                  default_values(),
                                  self._context(),
                                  self._set_solver_default)
        layout.addWidget(_group('Defaults', defaults_form))
        self._refreshers.append(lambda: defaults_form.refresh(default_values()))
        lines = []
        for physics in spec.physics:
            try:
                physics_type = self._registry.physics_type(physics.type)
            except KeyError:
                continue
            for role, solver_role in physics_type.solver_roles.items():
                if solver_role.used_in(spec.study.type):
                    node = self._assigned_solver(physics.name, role)
                    lines.append(f'{physics.name}, {SOLVER_ROLES[role].lower()}: '
                                 + (node or 'Automatic ({})'.format(self._solver_type_label(
                                     solver_role.automatic(solver_backend(spec), solver_grid(spec, physics))[0]))))
        layout.addWidget(_group('In use', hint_label('\n'.join(lines) if lines else
                                                     'No physics of this model hands out a solver job in the current '
                                                     'study.')))
        add_box = QtWidgets.QGroupBox('Add a solver')
        add_layout = QtWidgets.QVBoxLayout(add_box)
        fitting = [(role, solver_type) for role, solver_type, reason in solver_candidates(spec, self._registry)
                   if not reason]
        for role, solver_type in fitting:
            add_layout.addWidget(self._type_adder_row(
                f'{solver_type.label}  ({SOLVER_ROLES[role].lower()})', solver_type,
                partial(self._document.add_solver, solver_type.name)))
        if not fitting:
            add_layout.addWidget(hint_label('No solver type fits the physics of this model in the current study.'))
        layout.addWidget(add_box)

    def _solver_type_label(self, type_name):
        """Label of a registered solver type, else its name."""
        try:
            return self._registry.solver_type(type_name).label
        except KeyError:
            return str(type_name)

    def _build_solver(self, path, layout):
        name = path[1]
        solver = self._document.spec.solver(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        try:
            solver_type = self._registry.solver_type(solver.type)
        except KeyError as exc:
            form.addRow('Type', QtWidgets.QLabel(str(solver.type)))
            layout.addWidget(error_label(error_message(exc)))
            return
        form.addRow('Type', QtWidgets.QLabel(solver_type.label))
        used_by = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        used_by.setToolTip(rich_tooltip('Physics roles this solver fills. Assign it in the Solver group of a '
                                        "physics' settings."))

        def show_users():
            users = solver_users(self._document.spec, name)
            used_by.setText(', '.join(f'{physics} ({SOLVER_ROLES.get(role, role).lower()})' for physics, role in users)
                            or 'nothing yet: pick it in the Solver group of a physics')
        show_users()
        self._refreshers.append(show_users)
        form.addRow('Used by', used_by)
        self._add_summary(layout, solver_type)
        self._add_param_group(layout, 'Settings', solver_type.params, self._context(), 'solvers',
                              lambda spec: spec.solver(name).settings, name, what='setting')
        self._add_technical_details(layout, f'solver:{solver_type.name}', solver_type)

    def _build_study(self, path, layout):
        spec = self._document.spec
        study = spec.study
        form = _form_layout(layout)
        study_types = self._registry.study_types()
        type_combo = ValueComboBox([(study_type.name, study_type.label) for study_type in study_types], study.type,
                                   'Settings that the new study type does not use are dropped',
                                   item_tips={study_type.name: study_type.summary or study_type.description
                                              for study_type in study_types})
        type_combo.value_committed.connect(self._set_study_type)
        form.addRow('Study type', type_combo)
        try:
            study_type = self._registry.study_type(study.type)
        except KeyError as exc:
            study_type = None
            layout.addWidget(error_label(error_message(exc)))
        else:
            self._add_summary(layout, study_type)
            self._add_param_group(layout, 'Settings', study_type.params, self._context(), 'study',
                                  lambda spec: spec.study.settings, 'study', what='setting')
        self._add_stored_fields(layout, spec)
        if spec.is_3d or spec.study.stored_cuts:   # 2D: shown only while set, to clear them
            self._add_stored_cuts(layout, spec)
        self._add_sweep(layout, spec)
        self._add_adaptation(layout, spec)
        self._add_estimation_group(layout, spec)
        if study_type is not None:
            self._add_technical_details(layout, f'study:{study_type.name}', study_type)

    def _add_stored_fields(self, layout, spec):
        entries = []
        for physics in spec.physics:
            try:
                physics_type = self._registry.physics_type(physics.type)
            except KeyError:
                continue
            for info in physics_type.output_fields:
                if info.requires_study and info.requires_study != spec.study.type:
                    continue   # e.g. harmonic FDTD phasors exist only in a frequency_domain study
                key = f'{physics.name}.{info.name}'
                label = f'{physics.name}: {info.label}' + ('  (main field)' if info.primary else '')
                entries.append((key, label, key + (f'\nunit: {info.unit}' if info.unit else '')))
        stored = CheckList(entries, spec.study.stored_fields)
        stored.value_committed.connect(self._set_stored_fields)
        self._refreshers.append(lambda: stored.set_value(self._document.spec.study.stored_fields))
        layout.addWidget(_group('Stored fields', hint_label('Fields saved at each output time. With none checked, '
                                                            'every field is saved; check some to save memory.'),
                                stored))

    def _add_stored_cuts(self, layout, spec):
        cuts_editor = StoredCutsEdit(_STORED_CUTS, spec.study.stored_cuts, self._context())
        cuts_editor.value_committed.connect(self._set_stored_cuts)
        self._refreshers.append(lambda: (cuts_editor.set_value(self._document.spec.study.stored_cuts),
                                         cuts_editor.refresh_check()))
        box = QtWidgets.QGroupBox('Stored cut planes')
        form = growing_form(box)
        form.addRow(hint_label('3D models: the stored fields of these planes are saved too (besides the mid-height '
                               'horizontal plane).' if spec.is_3d else
                               'Cut planes apply to 3D models only: clear them.'))
        form.addRow(param_row_label(_STORED_CUTS), cuts_editor)
        layout.addWidget(box)

    def _add_sweep(self, layout, spec):
        sweep = spec.study.sweep
        parameter_names = [parameter.name for parameter in spec.parameters]
        box = QtWidgets.QGroupBox('Parameter sweep')
        box.setCheckable(True)
        box.setChecked(sweep is not None)
        box.clicked.connect(self._set_sweep_enabled)
        form = growing_form(box)
        if sweep is None:
            estimating = spec.study.estimation is not None   # its group's refresher rebuilds the form on a change
            form.addRow(hint_label('Repeat the study for several values of one parameter.'
                                   + ('' if parameter_names else ' Add a parameter first.')
                                   + (' Not available together with parameter estimation.' if estimating else '')))
            box.setEnabled(bool(parameter_names) and not estimating)
        else:
            parameter_combo = ValueComboBox([(name, name) for name in parameter_names], sweep.parameter)
            parameter_combo.value_committed.connect(_slot(self._set_sweep_attribute, 'parameter'))
            form.addRow('Parameter', parameter_combo)
            ctx = self._context()
            range_form = ParamForm(_SWEEP_FORM_PARAMS, _sweep_form_values(sweep), ctx, self._set_sweep_attribute)
            form.addRow(range_form)
            preview = hint_label('')
            form.addRow(preview)

            def refresh_range():
                current = self._document.spec.study.sweep
                if current is None:
                    return
                for name in ('lower', 'upper', 'count'):
                    range_form.set_row_enabled(name, current.spacing != 'list')
                range_form.set_row_enabled('values', current.spacing == 'list')
                range_form.refresh(_sweep_form_values(current))
                preview.setText(_sweep_preview(current, ctx.parameters))
            refresh_range()
            self._refreshers.append(refresh_range)
            keep_check = QtWidgets.QCheckBox('Keep fields of every value')
            keep_check.setChecked(bool(sweep.keep_fields))
            keep_check.setToolTip(rich_tooltip(
                'Keep the stored snapshots of every sweep value, not only those of the last one, so the Field tab '
                'can show each value.\n\n'
                'Takes memory: past 1 GB in total only the final snapshot of each value is kept.'))
            keep_check.clicked.connect(_slot(self._set_sweep_attribute, 'keep_fields'))
            form.addRow('', keep_check)
            reuse_check = QtWidgets.QCheckBox('Keep the build between values')
            reuse_check.setChecked(bool(sweep.reuse_build))
            reuse_check.setToolTip(rich_tooltip(
                'Build the model once and only change the loads for each value, which saves the build and the '
                'meshing of every later value.\n\n'
                'Works when the swept parameter changes only loads, prescribed displacements, results or study '
                'settings of a solid on a triangle or tetrahedron mesh; any other use is reported as an error. A '
                'nonlinear solid starts each value from the previous solution.'))
            reuse_check.clicked.connect(_slot(self._set_sweep_attribute, 'reuse_build'))
            form.addRow('', reuse_check)
        layout.addWidget(box)

    def _add_adaptation(self, layout, spec):
        """'Mesh refinement' group of StudySpec.adaptation (S7.7): off, disabled where validation would refuse it (a
        refresher follows study-setting edits); on, one row per AdaptationSpec field."""
        adaptation = spec.study.adaptation
        box = QtWidgets.QGroupBox('Mesh refinement')
        box.setCheckable(True)
        box.setChecked(adaptation is not None)
        box.setToolTip(rich_tooltip(_ADAPTATION_TIP))
        box.clicked.connect(self._set_adaptation_enabled)
        form = growing_form(box)
        layout.addWidget(box)
        if adaptation is None:
            hint = hint_label('')

            def refresh_blocker():
                blocker = _adaptation_blocker(self._document.spec, self._registry)
                hint.setText('Run the study again on finer meshes until a derived value stops changing.'
                             + (f' {blocker}' if blocker else ''))
                box.setEnabled(not blocker)
            refresh_blocker()
            self._refreshers.append(refresh_blocker)
            form.addRow(hint)
            return

        def row_label(attribute):
            text, tip = _ADAPTATION_ROWS[attribute]
            label = QtWidgets.QLabel(text)
            label.setToolTip(rich_tooltip(tip))
            return label

        def tip(attribute):
            return _ADAPTATION_ROWS[attribute][1]

        method_combo = ValueComboBox([(method, _ADAPTATION_METHOD_LABELS.get(method, method))
                                      for method in ADAPTATION_METHODS], adaptation.method, tip('method'))
        target_combo = ValueComboBox([(derived.name, derived.name) for derived in spec.derived_values],
                                     adaptation.target, tip('target'))
        tolerance_edit = NumberEdit(adaptation.tolerance, tip('tolerance'))
        levels_spin = IntSpinBox(1, 100, adaptation.max_levels, help_text=tip('max_levels'))
        growth_edit = NumberEdit(adaptation.growth, tip('growth'))
        cells_spin = IntSpinBox(0, _MAX_SPIN_INT, adaptation.max_cells, special_text='No limit',
                                help_text=tip('max_cells'))
        cells_spin.setSingleStep(10_000)
        for attribute, editor in (('method', method_combo), ('target', target_combo), ('tolerance', tolerance_edit),
                                  ('max_levels', levels_spin), ('growth', growth_edit)):
            editor.value_committed.connect(_slot(self._set_adaptation_attribute, attribute))
            form.addRow(row_label(attribute), editor)
        used_meshes = [mesh_spec.name for mesh_spec in spec.used_meshes()]
        meshes_list = None
        if len(used_meshes) > 1 or adaptation.meshes:   # one mesh: shown only while set, to clear it
            meshes_list = CheckList([(name, name, '') for name in used_meshes], adaptation.meshes, tip('meshes'))
            meshes_list.value_committed.connect(_slot(self._set_adaptation_attribute, 'meshes'))
            form.addRow(row_label('meshes'), meshes_list)
        cells_spin.value_committed.connect(_slot(self._set_adaptation_attribute, 'max_cells'))
        form.addRow(row_label('max_cells'), cells_spin)
        checks = {}
        for attribute in ('anisotropic', 'keep_levels'):
            check = QtWidgets.QCheckBox(_ADAPTATION_ROWS[attribute][0])
            check.setToolTip(rich_tooltip(tip(attribute)))
            check.setChecked(getattr(adaptation, attribute) is True)
            check.clicked.connect(_slot(self._set_adaptation_attribute, attribute))
            check.setVisible(attribute != 'anisotropic' or adaptation.method == 'adaptive')   # uniform ignores it
            form.addRow('', check)
            checks[attribute] = check

        def refresh():
            current = _study_adaptation(self._document.spec)
            method_combo.set_value(current.method)
            target_combo.set_value(current.target)
            target_combo.refresh_check()
            for editor, value in ((tolerance_edit, current.tolerance), (growth_edit, current.growth)):
                editor.set_value(value)
                editor.refresh_check()
            levels_spin.refresh_value(current.max_levels)
            cells_spin.refresh_value(current.max_cells)
            if meshes_list is not None:
                meshes_list.set_value(current.meshes)
            for attribute, check in checks.items():
                check.setChecked(getattr(current, attribute) is True)
        self._refreshers.append(refresh)

    def _build_results_root(self, path, layout):
        layout.addWidget(hint_label('Fields, probes and spectra appear under Results (end of the model tree) after a '
                                    'run.\n\n'
                                    'A derived value turns a field into one number after every run, such as the mean '
                                    'temperature of an area or the total absorbed power.\n\n'
                                    'An analytic curve draws a formula dashed over a result, the sweep plot or a line '
                                    'profile, for comparison.\n\n'
                                    'A derived curve computes a new curve from result curves after every run, such as '
                                    'transmitted over incident power.'))
        buttons = []
        for kind, (label, method) in RESULT_SETTING_ADDERS.items():
            add_button = QtWidgets.QPushButton(label)
            add_button.setToolTip(rich_tooltip(RESULT_SETTINGS_TIPS[kind]))
            _on_click(add_button, partial(self._document_op, getattr(self._document, method), label))
            buttons.append(add_button)
        layout.addLayout(_button_row(*buttons))

    def _build_derived_values_root(self, path, layout):
        layout.addWidget(hint_label('A derived value turns a field into one number after every run, such as the mean '
                                    'temperature of an area or the total absorbed power. The values are listed in the '
                                    'Results panel and, in a sweep, plotted over the swept parameter.'))
        self._add_adder_row(layout, 'derived_value')

    def _build_analytic_curves_root(self, path, layout):
        layout.addWidget(hint_label('An analytic curve draws a formula dashed over a result curve, the sweep plot or a '
                                    'line profile, for comparison with the simulation.'))
        self._add_adder_row(layout, 'analytic_curve')

    def _build_derived_curves_root(self, path, layout):
        layout.addWidget(hint_label('Derived curves are computed from result curves after every run and shown in the '
                                    'Derived curves tab of the Results panel. Curves with the same plot name are drawn '
                                    'together: each plot is a sub-node here and a choice in that tab.'))
        self._add_adder_row(layout, 'derived_curve')

    def _build_derived_plot(self, path, layout):
        plot = path[1]
        form = _form_layout(layout)
        self._add_name_row(form, path)
        members = [curve.name for curve in self._document.spec.derived_curves if (curve.plot or curve.name) == plot]
        layout.addWidget(hint_label(f"Curves in this plot: {', '.join(members)}. Renaming the plot moves all of them; "
                                    "a curve's own Plot setting moves it to another plot."))
        self._add_button_row(layout, 'Add derived curve to this plot',
                             partial(self._document.add_derived_curve, plot),
                             RESULT_SETTINGS_TIPS['derived_curve_to_plot'])

    def _add_adder_row(self, layout, kind):
        """_add_button_row of one result-setting adder (RESULT_SETTING_ADDERS kind)."""
        label, method = RESULT_SETTING_ADDERS[kind]
        self._add_button_row(layout, label, getattr(self._document, method), RESULT_SETTINGS_TIPS[kind])

    def _add_button_row(self, layout, label, operation, tip=''):
        """Row with one button running a document operation (refused edits reported); tip: its tooltip."""
        button = QtWidgets.QPushButton(label)
        if tip:
            button.setToolTip(rich_tooltip(tip))
        _on_click(button, partial(self._document_op, operation, label))
        layout.addLayout(_button_row(button))

    def _build_computed_results(self, path, layout):
        layout.addWidget(hint_label('The fields, probes and spectra of the last build or run are in the Results '
                                    'panel. Derived values, analytic curves and derived curves are set up under Result '
                                    'settings.'))
        show_button = QtWidgets.QPushButton('Show results')
        _on_click(show_button, self.show_results_requested.emit)
        layout.addLayout(_button_row(show_button))

    def _build_derived(self, path, layout):
        name = path[1]
        spec = self._document.spec
        derived = spec.derived_value(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        sources = self._document.result_features(DERIVED_RESULT_KINDS)
        curve_sources = [(curve.name, f'{curve.name} (derived curve)') for curve in spec.derived_curves]
        source_combo = ValueComboBox([(None, 'Fields over the selection')]
                                     + [(key, f'{key} ({kind})') for key, kind in sources] + curve_sources,
                                     derived.source,
                                     'What the value is computed from: fields over an area, the samples of a '
                                     'monitor or probe result, or a derived curve.')
        source_combo.value_committed.connect(partial(self._set_derived_source, name))
        form.addRow('Source', source_combo)
        earlier_names = frozenset(other.name for other in spec.derived_values[:spec.derived_values.index(derived)])
        interface = None
        global_value = derived.source is None and derived.operation == GLOBAL_DERIVED_OPERATION
        if global_value:   # no fields, selection or point (validate _check_global_expression)
            scalar_names = scalar_output_names(spec, self._registry)
            params = (_GLOBAL_DERIVED_EXPRESSION, _FIELD_DERIVED_OPERATION, _DERIVED_REFERENCE, _DERIVED_TOLERANCE)
            ctx = self._context(expression_names=frozenset(scalar_names) | {'t'} | earlier_names)
            hint = ('Computed numbers you can use: ' + ', '.join(sorted(scalar_names)) if scalar_names else '')
            if not earlier_names:
                hint += ('; ' if hint else '') + 'add the values to combine above this one first'
        elif derived.source is None:   # names as in validate / Session.evaluate
            interfaces =[selection.name for selection in spec.selections if selection.kind == 'interface']
            if len(derived.selection) == 1 and derived.selection[0] in interfaces:
                interface = derived.selection[0]
            if interfaces:
                interface_combo = ValueComboBox([(None, 'None: reduce over areas')]
                                                + [(item, item) for item in interfaces], interface, _DERIVED_INTERFACE_TIP)
                interface_combo.value_committed.connect(partial(self._set_derived_interface, name))
                form.addRow(tip_label('Interface', _DERIVED_INTERFACE_TIP), interface_combo)
            if derived.operation in ('at', 'mode_volume'):
                points = [obj.name for obj in spec.geometry if obj.kind == 'point']
                point_combo = ValueComboBox([('', 'None (Mode volume: the maximum)')] + [(item, item) for item in points],
                                            derived.point, _DERIVED_POINT_TIP)
                point_combo.value_committed.connect(_slot(self._set_derived_attribute, name, 'point'))
                form.addRow(tip_label('Point', _DERIVED_POINT_TIP), point_combo)
            field_names = output_field_names(spec, self._registry)
            scalar_names = scalar_output_names(spec, self._registry)
            params = (_FIELD_DERIVED_EXPRESSION, _FIELD_DERIVED_OPERATION, _DERIVED_REFERENCE, _DERIVED_TOLERANCE)
            ctx = self._context(expression_names=frozenset(field_names) | frozenset(scalar_names) | COORDINATE_NAMES
                                | earlier_names | (frozenset(INTERFACE_VARIABLES) if interface else frozenset()))
            hint = 'Fields you can use in the expression: ' + ', '.join(sorted(field_names)) if field_names else ''
            if scalar_names:
                hint += ('; ' if hint else '') + 'computed numbers: ' + ', '.join(sorted(scalar_names))
            if interface:
                hint += (('; ' if hint else '') + 'on the interface also ' + ', '.join(INTERFACE_VARIABLES)
                         + ' (outward normal and tangent of the objects)')
        else:
            if is_curve_source(derived.source):   # a derived curve: its values by the curve name, abscissa x
                source_kind, sample_names = 'xy', (derived.source, 'x')
            else:
                source_kind = dict(sources).get(derived.source)
                sample_names = RESULT_SAMPLE_NAMES.get(source_kind, (None, None))[1]
            expression_param = Param('expression', 'expression' if sample_names is not None else 'str', None,
                                     label='Expression',
                                     help='Expression of the result sample names and parameters, evaluated per '
                                          'sample (e.g. power, 1 - fraction_positive)',
                                     tip=_RESULT_DERIVED_EXPRESSION_TIP)
            params = [expression_param, _RESULT_DERIVED_OPERATION, _DERIVED_AT, _DERIVED_WINDOW]
            if source_kind == 'far_field' or derived.wavelength is not None:   # shown elsewhere only to clear it
                params.append(_DERIVED_WAVELENGTH)
            params += [_fit_param(derived.operation, derived.degree), _DERIVED_DEGREE, _DERIVED_REFERENCE,
                       _DERIVED_TOLERANCE]
            ctx = self._context(expression_names=frozenset(sample_names or ()) | earlier_names)
            hint = ('Sample names: ' + ', '.join(sample_names) if sample_names is not None else
                    'Sample names: x (xy) or row (table), the xy series and numeric table columns by name and the '
                    'numeric extras of the result (see the feature description); a missing name gives NaN. A name '
                    'such as abs without "(" is the sample, with "(" the function')
        values_form = ParamForm(params, self._derived_values(name), ctx, partial(self._set_derived_attribute, name))
        layout.addWidget(_group('Value', values_form))
        self._refreshers.append(lambda: values_form.refresh(self._derived_values(name)))
        if hint:
            layout.addWidget(hint_label(hint))
        if earlier_names:
            layout.addWidget(hint_label('Derived values listed above this one: ' + ', '.join(sorted(earlier_names))))
        if derived.source is None and not global_value and several_meshes(spec):   # S7.2 D3: the evaluation mesh
            mesh_label = hint_label('')
            mesh_label.setTextFormat(QtCore.Qt.TextFormat.RichText)

            def refresh_mesh():
                current = self._document.spec
                mesh_name = evaluation_mesh(current, self._registry, current.derived_value(name).expression)
                mesh_label.setText(mesh_badge_html(current, mesh_name, 'the fields are evaluated on it (the mesh of '
                                                                       'the first physics the expression names; '
                                                                       'fields of other meshes are interpolated)'))
            refresh_mesh()
            self._refreshers.append(refresh_mesh)
            layout.addWidget(mesh_label)
        if derived.source is None and interface is None and not global_value:
            picker = SelectionPicker(spec, 'domain', derived.selection,
                                     _slot(self._set_derived_attribute, name, 'selection'))
            layout.addWidget(_group('Selection (empty or all = whole domain)', picker))
            self._refreshers.append(lambda: picker.set_selected(self._document.spec.derived_value(name).selection))

    def _build_curve(self, path, layout):
        name = path[1]
        spec = self._document.spec
        curve = spec.analytic_curve(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        features = self._document.result_features(CURVE_RESULT_KINDS)
        target_combo = ValueComboBox([(key, f'{key} ({kind})') for key, kind in features]
                                     + [('sweep', 'Sweep plot (versus the swept parameter)'),
                                        ('line', 'Line profile of a field')], curve.target,
                                     'Simulated curve the analytic one is drawn over')
        target_combo.value_committed.connect(partial(self._set_curve_target, name))
        form.addRow('Target', target_combo)
        target_kind = curve.target if curve.target in CURVE_TARGETS else dict(features).get(curve.target)
        variables = curve_variables(target_kind, curve.series) if target_kind in CURVE_VARIABLES else ()
        params = [Param('expression', 'expression', None, label='Expression',
                        help='Analytic formula of the parameters and the target variables (see below)')]
        if target_kind == 'table':
            params.append(_CURVE_X_COLUMN)
        if target_kind != 'line':
            params.append(_CURVE_SERIES)
        else:
            params += [_CURVE_LINE, _CURVE_PLANE, _CURVE_SNAPSHOT] if spec.is_3d else [_CURVE_LINE, _CURVE_SNAPSHOT]
            entries = [(f'{physics}.{info.name}', f'{physics}.{info.name}  ({info.label})')
                       for physics, info in self._output_fields(spec)]
            field_combo = ValueComboBox(entries, curve.field, 'Output field sampled along the line (final state)')
            field_combo.value_committed.connect(_slot(self._set_curve_attribute, name, 'field'))
            form.addRow('Field', field_combo)
            self._refreshers.append(lambda: field_combo.set_value(self._document.spec.analytic_curve(name).field))
            if spec.is_3d or curve.sampling != 'nearest':   # 3D segments (shown elsewhere only to reset it)
                sampling_combo = ValueComboBox(_CURVE_SAMPLING_CHOICES, curve.sampling, _CURVE_SAMPLING_TIP)
                sampling_combo.value_committed.connect(_slot(self._set_curve_attribute, name, 'sampling'))
                form.addRow(tip_label('Sampling', _CURVE_SAMPLING_TIP), sampling_combo)
                self._refreshers.append(lambda: sampling_combo.set_value(
                    self._document.spec.analytic_curve(name).sampling))
            interfaces = [selection.name for selection in spec.selections if selection.kind == 'interface']
            if (interfaces and not spec.is_3d) or curve.interface:
                interface_combo = ValueComboBox([('', 'None: the cell holding each sample')]
                                                + [(item, item) for item in interfaces], curve.interface,
                                                _CURVE_INTERFACE_TIP)
                interface_combo.value_committed.connect(_slot(self._set_curve_attribute, name, 'interface'))
                form.addRow(tip_label('Interface', _CURVE_INTERFACE_TIP), interface_combo)
                self._refreshers.append(lambda: interface_combo.set_value(
                    self._document.spec.analytic_curve(name).interface))
                projection_combo = ValueComboBox(_CURVE_PROJECTION_CHOICES, curve.projection, _CURVE_PROJECTION_TIP)
                projection_combo.value_committed.connect(_slot(self._set_curve_attribute, name, 'projection'))
                form.addRow(tip_label('Projection', _CURVE_PROJECTION_TIP), projection_combo)
                self._refreshers.append(lambda: projection_combo.set_value(
                    self._document.spec.analytic_curve(name).projection))
        ctx = self._context(expression_names=frozenset(variables))
        values_form = ParamForm(params, self._curve_values(name), ctx, partial(self._set_curve_attribute, name))
        layout.addWidget(_group('Curve', values_form))
        self._refreshers.append(lambda: values_form.refresh(self._curve_values(name)))
        if target_kind in _CURVE_TARGET_HINTS:
            layout.addWidget(hint_label(f'Variables: {_CURVE_TARGET_HINTS[target_kind]}; drawn dashed in the colour '
                                        'of the simulated curve, evaluated over the parameters of the last build.'))

    def _build_derived_curve(self, path, layout):
        name = path[1]
        curve = self._document.spec.derived_curve(name)
        form = _form_layout(layout)
        self._add_name_row(form, path)
        inputs_edit = CurveInputsEdit(curve.inputs, _DERIVED_CURVE_INPUTS_TIP)
        inputs_edit.value_committed.connect(_slot(self._set_derived_curve_attribute, name, 'inputs'))
        form.addRow(tip_label('Inputs', _DERIVED_CURVE_INPUTS_TIP), inputs_edit)
        self._refreshers.append(lambda: inputs_edit.set_value(self._document.spec.derived_curve(name).inputs))
        aliases = frozenset(curve.inputs) if isinstance(curve.inputs, dict) else frozenset()
        ctx = self._context(expression_names=aliases | frozenset(DERIVED_CURVE_VARIABLES),
                            functions=CURVE_FUNCTION_ARITY)   # gradient / cumtrapz: derived-curve expressions only
        values_form = ParamForm(_DERIVED_CURVE_PARAMS, self._derived_curve_values(name), ctx,
                                partial(self._set_derived_curve_attribute, name))
        layout.addWidget(_group('Curve', values_form))
        self._refreshers.append(lambda: values_form.refresh(self._derived_curve_values(name)))
        reference_form = ParamForm((_DERIVED_CURVE_REFERENCE, _DERIVED_CURVE_TOLERANCE),
                                   self._derived_curve_reference_values(name),
                                   self._context(expression_names=frozenset(DERIVED_CURVE_VARIABLES)),
                                   partial(self._set_derived_curve_attribute, name))
        layout.addWidget(_group('Reference', reference_form))
        self._refreshers.append(lambda: reference_form.refresh(self._derived_curve_reference_values(name)))
        features = '; '.join(_curve_input_text(key, kind)
                             for key, kind in self._document.result_features(DERIVED_RESULT_KINDS))
        layout.addWidget(hint_label('Results you can read: ' + (features or 'none (add a monitor or probe)')
                                    + '. The expression may also use gradient(y) (derivative along x) and cumtrapz(y) '
                                      '(running integral from the first x). Computed after every run and drawn in the '
                                      'Derived curves tab of the Results panel.'))

    def _output_fields(self, spec):
        """[(physics name, FieldInfo)] of every output field of the registered physics available in the study type, in
        model order."""
        entries = []
        for physics in spec.physics:
            try:
                physics_type = self._registry.physics_type(physics.type)
            except KeyError:
                continue
            entries.extend((physics.name, info) for info in physics_type.output_fields
                           if not info.requires_study or info.requires_study == spec.study.type)
        return entries

    # ---- current values (read by name) ----
    def _domain_values(self):
        domain = self._document.spec.domain
        return {param.name: getattr(domain, param.name) for param in _DOMAIN_PARAMS}

    def _mesh_values(self):
        mesh = self._form_mesh(self._document.spec)
        return {param.name: getattr(mesh, param.name) for param in MESH_PARAMS}

    def _derived_values(self, name):
        derived = self._document.spec.derived_value(name)
        return {'expression': derived.expression, 'operation': derived.operation, 'at': derived.at,
                'reference': derived.reference, 'tolerance': derived.tolerance, 'window': derived.window,
                'wavelength': derived.wavelength, 'fit': derived.fit, 'degree': derived.degree}

    def _curve_values(self, name):
        curve = self._document.spec.analytic_curve(name)
        return {'expression': curve.expression, 'series': curve.series, 'line': curve.line, 'x_column': curve.x_column,
                'plane': curve.plane, 'snapshot': None if curve.snapshot is None else str(curve.snapshot)}

    def _derived_curve_values(self, name):
        curve = self._document.spec.derived_curve(name)
        return {param.name: getattr(curve, param.name) for param in _DERIVED_CURVE_PARAMS}

    def _derived_curve_reference_values(self, name):
        curve = self._document.spec.derived_curve(name)
        return {'reference': curve.reference, 'tolerance': curve.tolerance}

    # ---- edits (each looks its node up by name inside the mutate fn) ----
    def _set_title(self, text):
        self._commit('model', lambda spec: setattr(spec, 'title', text), 'Set title')

    def _set_description(self, text):
        self._commit('model', lambda spec: setattr(spec, 'description', text), 'Set description')

    def _update_parameter_values(self, table):
        values = self._document.parameters()
        errors = self._document.evaluation_errors()
        normal_brush = table.palette().brush(QtGui.QPalette.ColorRole.Text)
        error_brush = QtGui.QBrush(QtGui.QColor(ERROR_COLOR))
        table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                name = table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
                value_item = table.item(row, 2)
                if name in values:
                    value_item.setText(display_number(values[name]))
                    value_item.setForeground(normal_brush)
                    value_item.setToolTip('')
                else:
                    value_item.setText('error')
                    value_item.setForeground(error_brush)
                    value_item.setToolTip(str(errors.get(name, 'not evaluated')))
        finally:
            table.blockSignals(False)

    def _on_parameter_item_changed(self, table, item):
        name_item = table.item(item.row(), 0)
        if name_item is None:
            return
        name = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
        text = item.text()
        if item.column() == 0:
            new_name = text.strip()
            if new_name != name:   # 'all': the rename rewrites expressions in every section
                self._commit('all', lambda spec: spec.rename_parameter(name, new_name, self._registry),
                             f'Rename parameter {name}', rebuild=True)
        elif item.column() == 1:
            if not text.strip():
                self._report(f"Parameter '{name}': an expression is required")
                self._schedule(rebuild=True)
                return
            expression = parse_number_or_text(text)
            self._commit('parameters', lambda spec: setattr(spec.parameter(name), 'expression', expression),
                         f'Set parameter {name}')
        elif item.column() == 3:
            self._commit('parameters', lambda spec: setattr(spec.parameter(name), 'description', text),
                         f'Describe parameter {name}')

    def _remove_parameters(self, table):
        rows = {index.row() for index in table.selectionModel().selectedRows()}
        names = {table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole) for row in rows}
        if not names:
            self._report('Remove parameters: select whole rows first')
            return

        def apply(spec):
            spec.parameters = [parameter for parameter in spec.parameters if parameter.name not in names]
        self._commit('parameters', apply, 'Remove parameters', rebuild=True)

    def _set_domain_size(self, attribute, value):
        label = f'Set domain: {_form_label_of(_DOMAIN_PARAMS, attribute)}'
        if attribute in ('size_z', 'coordinates', 'angle_wrap'):   # changes outer faces, forms and menus everywhere
            self._commit('all', lambda spec: setattr(spec.domain, attribute, value), label, rebuild=True)
            return
        self._commit('geometry', lambda spec: setattr(spec.domain, attribute, value), label)

    def _add_selection(self, kind):
        self._document_op(lambda: self._document.add_selection(kind), f'Add {kind} selection')

    def _set_selection_kind(self, selection_name, kind):
        def apply(spec):
            selection = spec.selection(selection_name)
            eligible = {entity for entity, _label, _tooltip in selection_entries(spec, kind, exclude=(selection_name,))}
            selection.kind = kind
            selection.entities = [entity for entity in selection.entities if entity in eligible]
            selection.side = (selection.side or INTERFACE_SIDES[0]) if kind == 'interface' else ''
        self._commit('selections', apply, f'Set {selection_name} kind', rebuild=True)

    def _set_selection_side(self, selection_name, side):
        self._commit('selections', lambda spec: setattr(spec.selection(selection_name), 'side', side),
                     f'Set {selection_name} side')

    def _set_selection_entities(self, selection_name, entities):
        self._commit('selections', lambda spec: setattr(spec.selection(selection_name), 'entities', entities),
                     f'Select entities of {selection_name}')

    def _add_material(self, library):
        self._document_op(lambda: self._document.add_material(library), 'Add material')

    def _set_material_library(self, material_name, library):
        self._commit('materials', lambda spec: setattr(spec.material(material_name), 'library', library),
                     f'Set {material_name} library', rebuild=True)

    def _set_material_property(self, material_name, key, rebuild, value):
        def apply(spec):
            properties = spec.material(material_name).properties
            if value is None:
                properties.pop(key, None)
            else:
                properties[key] = value
        self._commit('materials', apply, f'Set {material_name}.{key}', rebuild=rebuild)

    def _set_material_selection(self, material_name, entities):
        self._commit('materials', lambda spec: setattr(spec.material(material_name), 'selection', entities),
                     f'Select domains of {material_name}')

    def _set_material_group_expanded(self, type_name, expanded):
        """Remember (for this session) whether a '(not in the model)' property group is shown open."""
        if expanded:
            self._expanded_material_groups.add(type_name)
        else:
            self._expanded_material_groups.discard(type_name)

    def _set_dispersion_inherit(self, material_name, inherit):
        def apply(spec):
            material = spec.material(material_name)
            if inherit:
                material.dispersion = None
            else:
                library_material = _library_material(material.library)
                material.dispersion = library_material.dispersion_models() if library_material is not None else []
        self._commit('materials', apply, f'Set {material_name} dispersion', rebuild=True)

    def _add_pole(self, material_name, model):
        def apply(spec):
            material = spec.material(material_name)
            if not isinstance(material.dispersion, list):
                raise ValueError('uncheck "Use the library poles" first')
            material.dispersion.append(_default_pole(model))
        self._commit('materials', apply, f'Add {model} pole to {material_name}', rebuild=True)

    def _remove_pole(self, material_name, index):
        def apply(spec):
            poles = spec.material(material_name).dispersion
            if not isinstance(poles, list) or not 0 <= index < len(poles):
                raise KeyError(f"pole {index + 1} of material '{material_name}' no longer exists")
            del poles[index]
        self._commit('materials', apply, f'Remove pole {index + 1} of {material_name}', rebuild=True)

    def _set_pole_model(self, material_name, index, old_model, new_model):
        def apply(spec):
            pole = _pole_at(spec, material_name, index, old_model)
            replacement = _default_pole(new_model)
            if 'gamma' in pole and 'gamma' in replacement:   # nk_fit has no damping rate
                replacement['gamma'] = pole['gamma']
            spec.material(material_name).dispersion[index] = replacement
        self._commit('materials', apply, f'Set pole {index + 1} model of {material_name}', rebuild=True)

    def _set_pole_field(self, material_name, index, model, field_name, value):
        def apply(spec):
            pole = _pole_at(spec, material_name, index, model)
            if value is None:   # an emptied optional field (eps_inf) falls back to its default
                pole.pop(field_name, None)
            else:
                pole[field_name] = value
        self._commit('materials', apply, f'Set pole {index + 1} {field_name} of {material_name}')

    def _add_crystal_entry(self, material_name):
        """Append the default crystal entry (YZ lithium niobate); inherited library poles become an editable copy."""
        def apply(spec):
            material = spec.material(material_name)
            if material_library.crystal_entry(material.dispersion) is not None:
                raise ValueError('the material already uses crystal data')
            if material.dispersion is None:
                library_material = _library_material(material.library)
                material.dispersion = library_material.dispersion_models() if library_material is not None else []
            if not isinstance(material.dispersion, list):
                raise ValueError(f'dispersion must be a list of poles, got {material.dispersion!r}')
            material.dispersion.append({**_DEFAULT_CRYSTAL_ENTRY,
                                        'euler_deg': list(_DEFAULT_CRYSTAL_ENTRY['euler_deg']),
                                        'plane': FULL_PLANE if spec.is_3d else SAGITTAL_PLANE})
        self._commit('materials', apply, f'Add crystal data to {material_name}', rebuild=True)

    def _remove_crystal_entry(self, material_name):
        def apply(spec):
            entry = _crystal_entry_at(spec, material_name)
            material = spec.material(material_name)
            material.dispersion = [pole for pole in material.dispersion if pole is not entry]
        self._commit('materials', apply, f'Remove crystal data of {material_name}', rebuild=True)

    def _set_crystal_value(self, material_name, key, value):
        """Set 'crystal' or 'plane' of the material's crystal entry."""
        def apply(spec):
            _crystal_entry_at(spec, material_name)[key] = value
        self._commit('materials', apply, f'Set {material_name} crystal {key}')

    def _set_crystal_angle(self, material_name, index, value):
        def apply(spec):
            entry = _crystal_entry_at(spec, material_name)
            angles = [0.0 if angle is None else angle for angle in _crystal_angles(entry)]
            angles[index] = value
            entry['euler_deg'] = angles
        self._commit('materials', apply, f'Set {material_name} crystal Euler angle {index + 1}')

    def _set_crystal_cut(self, material_name, cut):
        if cut is None:   # 'Custom angles' only names typed angles: the refresh shows the matching entry again
            self._schedule(rebuild=False)
            return
        def apply(spec):
            _crystal_entry_at(spec, material_name)['euler_deg'] = list(cut)
        self._commit('materials', apply, f'Set {material_name} crystal cut')

    def _set_physics_every(self, physics_name, every):
        self._commit('physics', lambda spec: setattr(spec.physics_by_name(physics_name), 'every', every),
                     f'Set {physics_name} every')

    def _set_feature_selection(self, physics_name, feature_name, entities):
        def apply(spec):
            spec.physics_by_name(physics_name).feature(feature_name).selection = entities
        self._commit('physics', apply, f'Select entities of {feature_name}')

    def _set_coupling_attribute(self, coupling_name, attribute, value):
        """Set source / target / every; a new endpoint rebuilds (the field and input drop-downs follow it)."""
        self._commit('couplings', lambda spec: setattr(spec.coupling(coupling_name), attribute, value),
                     f'Set {coupling_name} {attribute}', rebuild=attribute in ('source', 'target'))

    def _set_solver_default(self, attribute, value):
        """Set a solver_defaults field ('all': the tree label and the physics' solver fits follow the backend)."""
        self._commit('all', lambda spec: setattr(spec.solver_defaults, attribute, value),
                     f'Set solver default {attribute}')

    def _set_mesh_value(self, attribute, value):
        """Set a MESH_PARAMS field of the form's mesh; a kind or mesher change rebuilds (its groups depend on them:
        _tensor_graded)."""
        if attribute == 'mode':
            self._set_mesh_mode(value)
            return
        self._commit('mesh', lambda spec: setattr(self._form_mesh(spec), attribute, value),
                     f'Set mesh: {_form_label_of(MESH_PARAMS, attribute)}',
                     rebuild=attribute in ('kind', 'mesher', 'sweep_axis'))

    def _set_mesh_mode(self, mode):
        """Set the mesh mode: entering auto keeps the cell size (auto's smallest cell size) and drops the automatic
        refinements; leaving auto to graded fills in its sizes (refinements marked automatic), to uniform keeps a set
        cell size, else fills in the smallest automatic size."""
        form_mesh = self._form_mesh(self._document.spec)
        estimate, error = self._document.mesh_estimate(form_mesh.name)
        auto_dx = None if estimate is None else estimate.auto_dx
        leaving_auto = form_mesh.mode == 'auto' and mode != 'auto'
        keeps_dx = mode == 'uniform' and form_mesh.dx is not None   # auto's smallest size = the uniform size
        fills_in = leaving_auto and not keeps_dx

        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            own_refinements = [refinement for refinement in mesh_spec.refinements if not refinement.automatic]
            if mode == 'auto':
                mesh_spec.refinements = own_refinements
            elif fills_in and auto_dx is not None:
                if mode == 'nonuniform':   # uniform meshes ignore refinements
                    mesh_spec.dx = _rounded_size(auto_dx)
                    mesh_spec.refinements = [RefinementSpec(object_name, _rounded_size(size), automatic=True)
                                             for object_name, size in estimate.auto_refinements] + own_refinements
                else:
                    mesh_spec.dx = _rounded_size(_smallest_automatic_size(estimate))
            mesh_spec.mode = mode
        changed = self._commit('mesh', apply, f"Set mesh: {_form_label_of(MESH_PARAMS, 'mode')}", rebuild=True)
        if changed and fills_in and auto_dx is None:
            self._report(f'Automatic sizes not filled in: {error or "no mesh estimate"}')

    def _set_refinement(self, index, object_name, attribute, value):
        def apply(spec):
            setattr(_refinement_at(self._form_mesh(spec), index, object_name), attribute, value)
        self._commit('mesh', apply, f'Set refinement {index + 1} {attribute}', rebuild=attribute == 'object')

    def _remove_refinement(self, index, object_name):
        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            _refinement_at(mesh_spec, index, object_name)
            del mesh_spec.refinements[index]
        self._commit('mesh', apply, f'Remove refinement {index + 1}', rebuild=True)

    def _add_refinement(self):
        parameters = self._document.parameters()

        def apply(spec):
            if not spec.geometry:
                raise ValueError('add a geometry object first')
            mesh_spec = self._form_mesh(spec)
            mesh_spec.refinements.append(RefinementSpec(object=spec.geometry[0].name,
                                                        dx=_base_cell_size(mesh_spec, parameters) / 4))
        self._commit('mesh', apply, 'Add refinement', rebuild=True)

    def _add_mesh_axis(self, axis):
        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            if any(axis_spec.axis == axis for axis_spec in mesh_spec.axes):
                raise ValueError(f'axis {axis} has options already')
            mesh_spec.axes.append(MeshAxisSpec(axis=axis))
        self._commit('mesh', apply, f'Add {axis} axis options', rebuild=True)

    def _remove_mesh_axis(self, index, axis):
        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            _mesh_axis_at(mesh_spec, index, axis)
            del mesh_spec.axes[index]
        self._commit('mesh', apply, f'Remove {axis} axis options', rebuild=True)

    def _set_mesh_axis(self, index, axis, attribute, value, rebuild=False):
        def apply(spec):
            setattr(_mesh_axis_at(self._form_mesh(spec), index, axis), attribute, value)
        self._commit('mesh', apply, f'Set axis {axis} {attribute}', rebuild=rebuild)

    def _set_mesh_axis_cells(self, index, axis, mode):
        """Switch an axis between graded cells, listed edges (the current window cells, if few) and one segment."""
        estimate, _error = self._document.mesh_estimate(self._form_mesh(self._document.spec).name)
        parameters = self._document.parameters()

        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            axis_spec = _mesh_axis_at(mesh_spec, index, axis)
            length = _axis_length(spec, axis, parameters)
            current_edges = _window_axis_edges(estimate, axis)
            axis_spec.edges = _default_axis_edges(current_edges, length) if mode == 'edges' else []
            n_cells = (len(current_edges) - 1 if current_edges is not None
                       else max(1, round(length / _base_cell_size(mesh_spec, parameters))))
            axis_spec.segments = [MeshSegmentSpec(cells=n_cells)] if mode == 'segments' else []
        self._commit('mesh', apply, f'Set axis {axis} cells', rebuild=True)

    def _add_mesh_band(self, index, axis):
        parameters = self._document.parameters()

        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            length = _axis_length(spec, axis, parameters)
            _mesh_axis_at(mesh_spec, index, axis).bands.append(
                MeshBandSpec(lo=_rounded(0.4 * length), hi=_rounded(0.6 * length),
                             dx=_rounded(_base_cell_size(mesh_spec, parameters) / 4)))
        self._commit('mesh', apply, f'Add band to axis {axis}', rebuild=True)

    def _add_mesh_segment(self, index, axis):
        """Append a segment; an open-ended last segment first gets an end halfway to the axis end."""
        parameters = self._document.parameters()

        def apply(spec):
            segments = _mesh_axis_at(self._form_mesh(spec), index, axis).segments
            if segments and segments[-1].end is None:
                start = segments[-2].end if len(segments) > 1 else 0.0
                if start is None:
                    raise ValueError(f'set the end of segment {len(segments) - 1} first')
                start = evaluate_scalar(start, parameters)
                segments[-1].end = _rounded(0.5 * (start + _axis_length(spec, axis, parameters)))
            segments.append(MeshSegmentSpec(cells=10))
        self._commit('mesh', apply, f'Add segment to axis {axis}', rebuild=True)

    def _set_mesh_axis_item(self, index, axis, list_name, item_index, attribute, value):
        def apply(spec):
            setattr(_mesh_axis_item(self._form_mesh(spec), index, axis, list_name, item_index), attribute, value)
        self._commit('mesh', apply, f'Set axis {axis} {list_name[:-1]} {item_index + 1} {attribute}')

    def _remove_mesh_axis_item(self, index, axis, list_name, item_index):
        """Remove a band / segment; when the last segment goes, the new last one ends at the axis end."""
        def apply(spec):
            mesh_spec = self._form_mesh(spec)
            _mesh_axis_item(mesh_spec, index, axis, list_name, item_index)
            items = getattr(_mesh_axis_at(mesh_spec, index, axis), list_name)
            del items[item_index]
            if list_name == 'segments' and items and item_index == len(items):
                items[-1].end = None
        self._commit('mesh', apply, f'Remove {list_name[:-1]} {item_index + 1} of axis {axis}', rebuild=True)

    def _set_study_type(self, type_name):
        def apply(spec):
            known = {param.name for param in self._registry.study_type(type_name).params}
            spec.study.type = type_name
            spec.study.settings = {key: value for key, value in spec.study.settings.items() if key in known}
        self._commit('study', apply, 'Set study type', rebuild=True)

    def _set_stored_fields(self, keys):
        self._commit('study', lambda spec: setattr(spec.study, 'stored_fields', keys), 'Set stored fields')

    def _set_stored_cuts(self, cuts):
        self._commit('study', lambda spec: setattr(spec.study, 'stored_cuts', cuts), 'Set stored cut planes')

    def _set_sweep_enabled(self, checked):
        def apply(spec):
            if not checked:
                spec.study.sweep = None
            elif spec.study.sweep is None:
                if not spec.parameters:
                    raise ValueError('add a parameter first')
                if spec.study.estimation is not None:
                    raise ValueError('not available together with parameter estimation')
                first = spec.parameters[0]
                value = evaluate_parameters(spec.parameters, errors={}).get(first.name)
                lower, upper = ((first.expression, f'2*({first.expression})') if value   # 0 / unresolved: no 0..0 range
                                else ('1', '2'))
                spec.study.sweep = SweepSpec(parameter=first.name, values=[first.expression], spacing='linear',
                                             lower=lower, upper=upper)
        self._commit('study', apply, 'Parameter sweep', rebuild=True)

    def _set_sweep_attribute(self, attribute, value):
        def apply(spec):
            sweep = spec.study.sweep
            if sweep is None:
                raise KeyError('the study has no parameter sweep')
            if attribute == 'spacing' and value == 'list' and not sweep.values:   # start from the range's points
                sweep.values = sweep.point_values()
            elif attribute == 'spacing' and value != 'list' and (sweep.lower is None or sweep.upper is None) \
                    and sweep.values:   # a custom list's ends and length become the range
                sweep.lower, sweep.upper, sweep.count = sweep.values[0], sweep.values[-1], len(sweep.values)
            setattr(sweep, attribute, value)
        self._commit('study', apply, f'Set sweep {attribute}')

    def _set_adaptation_enabled(self, checked):
        def apply(spec):
            if not checked:
                spec.study.adaptation = None
            elif spec.study.adaptation is None:
                blocker = _adaptation_blocker(spec, self._registry)
                if blocker:
                    raise ValueError(blocker)
                spec.study.adaptation = AdaptationSpec(target=spec.derived_values[0].name)
        self._commit('study', apply, 'Mesh refinement', rebuild=True)

    def _set_adaptation_attribute(self, attribute, value):
        def apply(spec):
            setattr(_study_adaptation(spec), attribute, value)
        self._commit('study', apply, f'Set mesh refinement {_ADAPTATION_ROWS[attribute][0].lower()}',
                     rebuild=attribute == 'method')   # the method shows / hides 'Refine x and y separately'

    def _set_derived_attribute(self, name, attribute, value):
        """Set one field; an operation change resets fit / degree to suit it, and operation / degree edits rebuild
        (the fitted-quantity choices follow them)."""
        if attribute == 'expression' and value is not None and not isinstance(value, str):
            value = format_number(value)
        if attribute == 'window' and value is None:
            value = []

        def apply(spec):
            derived = spec.derived_value(name)
            if attribute == 'tolerance' and isinstance(value, str):
                raise ValueError('the tolerance must be a number (a relative error), not an expression')
            setattr(derived, attribute, value)
            if attribute == 'operation' and value == GLOBAL_DERIVED_OPERATION and derived.source is None:
                derived.selection, derived.point = [], ''   # it reads no fields
            if attribute == 'operation' and value not in FIT_QUANTITIES:
                derived.fit, derived.degree = '', 1
            elif attribute in ('operation', 'degree') and derived.fit not in _fit_quantities(derived.operation,
                                                                                             derived.degree):
                derived.fit = _default_fit(derived.operation, derived.degree)
        self._commit('results', apply, f'Set {name} {attribute}', rebuild=attribute in ('operation', 'degree'))

    def _set_derived_interface(self, name, interface):
        """Reduce a field derived value along an interface selection (its whole selection) or, None, over areas."""
        def apply(spec):
            derived = spec.derived_value(name)
            derived.selection = [] if interface is None else [interface]
            if interface is not None and derived.operation not in INTERFACE_OPERATIONS:
                derived.operation, derived.point = 'mean', ''
        self._commit('results', apply, f'Set {name} interface', rebuild=True)

    def _set_derived_source(self, name, source):
        """Switch a derived value between fields (None), a feature result and a derived curve; a source clears the
        selection, fields drop the source-only operation, at, window, wavelength, fit and degree."""
        def apply(spec):
            derived = spec.derived_value(name)
            derived.source = source
            if source is not None:
                derived.selection, derived.point = [], ''
                if derived.operation not in SOURCE_DERIVED_OPERATIONS:
                    derived.operation = 'mean'
                if is_curve_source(source):
                    derived.wavelength = None   # far_field sources only
            else:
                if derived.operation not in SOURCE_FREE_DERIVED_OPERATIONS:
                    derived.operation = 'mean'
                derived.at = None   # fields: 'at' reads the point object instead
                derived.window, derived.wavelength, derived.fit, derived.degree = [], None, '', 1
        self._commit('results', apply, f'Set {name} source', rebuild=True)

    def _set_curve_attribute(self, name, attribute, value):
        if attribute == 'expression' and value is not None and not isinstance(value, str):
            value = format_number(value)
        if attribute in ('series', 'x_column') and value is None:
            value = ''
        if attribute == 'snapshot':   # an integer text is an index, other text a label, empty the final state
            text = '' if value is None else str(value).strip()
            value = int(text) if text.lstrip('+-').isdigit() else (text or None)

        def apply(spec):
            setattr(spec.analytic_curve(name), attribute, value)
        # the timeseries variables follow the series ('fft': frequency), so a series edit rebuilds the form
        self._commit('results', apply, f'Set {name} {attribute}', rebuild=attribute == 'series')

    def _set_derived_curve_attribute(self, name, attribute, value):
        if attribute == 'expression' and value is not None and not isinstance(value, str):
            value = format_number(value)
        if attribute in ('plot', 'label', 'unit') and value is None:
            value = ''

        def apply(spec):
            if attribute == 'tolerance' and isinstance(value, str):
                raise ValueError('the tolerance must be a number (a relative error), not an expression')
            setattr(spec.derived_curve(name), attribute, value)
        # the input aliases are the expression's names, so an inputs edit rebuilds the form
        self._commit('results', apply, f'Set {name} {attribute}', rebuild=attribute == 'inputs')

    def _set_curve_target(self, name, target):
        """Retarget an analytic curve; 'line' gets a default line and field when it has none, others clear them."""
        default_line, default_field = self._document.default_line_target()

        def apply(spec):
            curve = spec.analytic_curve(name)
            curve.target = target
            curve.x_column = ''   # table columns differ per result
            if target == 'line':
                curve.line = curve.line or default_line
                curve.field = curve.field or default_field
                curve.series = ''
            else:
                curve.line, curve.field, curve.plane, curve.snapshot = [], '', None, None
                curve.sampling = 'nearest'
        self._commit('results', apply, f'Set {name} target', rebuild=True)

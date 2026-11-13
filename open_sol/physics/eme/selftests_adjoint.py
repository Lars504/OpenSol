"""Self-tests of the EME cascade adjoint (eme_plan 6b): dJ/d eps of complete-basis ez and hz cascades, plain or with a
periodic cell, vs central differences (CPU, adjoint.py on hand-built slab lines), the same for EMEPhysics3D on a cpu
Domain3D with complete (dense) cross-section bases, and a small em_eme taper optimisation in ez and hz (model runs,
optimization study)."""
import math
from dataclasses import replace

import numpy as np

from ...model.schema import SelfTestCase
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests import _OMEGA, _PHYSICS, _TYPE, _WAVELENGTH, _cpml_stretch, _mode_set, _slab_line

# eme_cascade_adjoint_gradient: 48-cell lines of 25 nm, slices of 40 nm along the axis; per slice the core cell range
# (eps 4 in 2.25): narrow input (3 slices), a staircase taper with one 3-slice section, wide output (3 slices).
_LINE_CELL = 25e-9
_LINE_CELLS = 48
_SLICE_WIDTH = 40e-9
_SLICE_CORES = ((20, 28),) * 3 + ((19, 29), (18, 30), (17, 31), (17, 31), (17, 31), (16, 32), (15, 33)) \
    + ((14, 34),) * 3
_CORE_EPS = (2.25, 4.0)
_DESIGN_SLICES = (3, 10)      # perturbed slices (the merged 3-slice section inside)
_INNER_CELLS = (10, 38)       # perturbed line cells outside the 10-cell stretch layers of the cpml variants
_ALL_CELLS = (0, _LINE_CELLS)  # every line cell (the end faces of the hz wall / periodic variants are differentiated)
_DIRECTIONS = 2               # random directions per variant (dense probes of the whole design block)
_STEP = 1e-3                  # eps step of the central differences
_GRADIENT_TOLERANCE = 1e-4    # relative (round-off / step ~1e-6 and O(step^2) far below)
# variant -> (polarization, line ends, transverse stretch, input face, perturbed line cells, objectives (quantity,
# mode_number, weight)); hz lines take the eps of each cell as eps_t and eps_l (isotropic, the hz face rule of PortLine)
_VARIANTS = {
    'walls_input_lo': ('ez', ('dirichlet', 'dirichlet'), False, 'lo', _INNER_CELLS,
                       (('transmission_mode', 1, 1.0), ('reflection', 1, 0.5))),
    'cpml_input_hi': ('ez', ('neumann', 'neumann'), True, 'hi', _INNER_CELLS,
                      (('transmission', 1, 1.0), ('reflection_mode', 1, -0.3), ('transmission_mode', 2, 0.2))),
    'hz_walls_input_lo': ('hz', ('odd', 'dirichlet'), False, 'lo', _ALL_CELLS,
                          (('transmission_mode', 1, 1.0), ('reflection', 1, 0.5))),
    'hz_cpml_input_hi': ('hz', ('neumann', 'neumann'), True, 'hi', _INNER_CELLS,
                         (('transmission', 1, 1.0), ('reflection_mode', 1, -0.3), ('transmission_mode', 2, 0.2))),
    'hz_periodic_input_lo': ('hz', ('periodic', 'periodic'), False, 'lo', _ALL_CELLS,
                             (('transmission', 1, 1.0), ('reflection', 1, -0.5))),
}
# periodic-block variants: 10 slices, narrow input (3), a 4-slice unit cell repeated _CELL_PERIODS times (0-based slices
# 3..6, the cascade link of EMEBase._solve), wide output (3); the perturbed 0-based slices 2..8 include the input /
# output sections next to the cell (their partial cascades hold the link)
_PERIODIC_CORES = ((20, 28),) * 3 + ((18, 30), (20, 28), (16, 32), (19, 29)) + ((14, 34),) * 3
_CELL_SLICES = (3, 6)
_CELL_PERIODS = 4
_PERIODIC_DESIGN_SLICES = (2, 9)
_PERIODIC_VARIANTS = {
    'cell_walls_input_lo': ('ez', ('dirichlet', 'dirichlet'), False, 'lo', _INNER_CELLS,
                            (('transmission_mode', 1, 1.0), ('reflection', 1, 0.5))),
    'hz_cell_walls_input_hi': ('hz', ('odd', 'dirichlet'), False, 'hi', _ALL_CELLS,
                               (('transmission', 1, 1.0), ('reflection_mode', 1, -0.3))),
}


def _variant(variant):
    """(polarization, line ends, transverse stretch, input face, perturbed line cells, objectives) of a variant."""
    return _VARIANTS[variant] if variant in _VARIANTS else _PERIODIC_VARIANTS[variant]


def _cores(variant):
    return _PERIODIC_CORES if variant in _PERIODIC_VARIANTS else _SLICE_CORES


def _face_index(ends):
    """(m + 1,) grid face of each line face (hi face of that cell; PortLine._face_index of a whole-row line)."""
    faces = np.arange(-1, _LINE_CELLS)
    return faces % _LINE_CELLS if ends[0] == 'periodic' else np.clip(faces, 0, _LINE_CELLS - 1)


def _line(variant, eps):
    """SlabLine of one slice row: ez as _slab_line; hz with the face eps = the yee_grid face mean of the cells
    _face_index pairs (as PortLine; _slab_line's lo end face differs)."""
    polarization, ends, stretched, _, _, _ = _variant(variant)
    stretch = _cpml_stretch(_LINE_CELLS) if stretched else None
    line = _slab_line(eps, _LINE_CELL, ends, polarization, stretch=stretch)
    if polarization == 'ez':
        return line
    grid_faces = _face_index(ends)
    partner = (grid_faces + 1) % _LINE_CELLS if ends[0] == 'periodic' else np.minimum(grid_faces + 1, _LINE_CELLS - 1)
    eps = np.asarray(eps, dtype=np.complex128)
    return replace(line, face_weight=2.0 / (eps[grid_faces] + eps[partner]))


def _slice_rows(variant, perturbation=None):
    """(n_slices, m) float64 eps rows of the variant's device, plus perturbation (same shape) when given."""
    cores = _cores(variant)
    rows = np.full((len(cores), _LINE_CELLS), _CORE_EPS[0])
    for row, (lo, hi) in zip(rows, cores):
        row[lo:hi] = _CORE_EPS[1]
    return rows if perturbation is None else rows + perturbation


def _merged(rows, merge):
    """[(first slice, n_slices)] of consecutive equal rows (merge) or one per slice."""
    runs = []
    for index, row in enumerate(rows):
        if merge and runs and np.array_equal(rows[runs[-1][0]], row):
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    return [tuple(run) for run in runs]


def _cell_block(runs):
    """(first, last) run indices of the unit cell slices _CELL_SLICES; ValueError when a run crosses its ends."""
    starts = [first for first, _ in runs]
    ends = [first + count - 1 for first, count in runs]
    if _CELL_SLICES[0] not in starts or _CELL_SLICES[1] not in ends:
        raise ValueError('eme_cascade_adjoint_gradient: a merged section crosses the unit cell ends')
    return starts.index(_CELL_SLICES[0]), ends.index(_CELL_SLICES[1])


def _device(variant, rows, merge, mode_cache):
    """(runs, lines, mode sets, Cascade, periodic) of the complete-basis cascade of the rows; periodic = (cell run
    block, cell Cascade, exit interface) of the periodic variants (the cell repeated _CELL_PERIODS times), else None.
    mode_cache: {row bytes: ModeSet} of this variant (equal rows give bit-identical sets, so unperturbed ones are reused)."""
    from .interfaces import cascade_mode_sets, periodic_cascade_mode_sets
    runs = _merged(rows, merge)
    lines = [_line(variant, rows[first]) for first, _ in runs]
    mode_sets = []
    for (first, _), line in zip(runs, lines):
        key = rows[first].tobytes()
        if key not in mode_cache:
            mode_cache[key] = _mode_set(line, 0)
        mode_sets.append(mode_cache[key])
    lengths = [count * _SLICE_WIDTH for _, count in runs]
    if variant not in _PERIODIC_VARIANTS:
        return runs, lines, mode_sets, cascade_mode_sets(mode_sets, lengths), None
    block = _cell_block(runs)
    cascade, cell, exit_interface = periodic_cascade_mode_sets(mode_sets, lengths, *block, _CELL_PERIODS)
    return runs, lines, mode_sets, cascade, (block, cell, exit_interface)


def _objective(variant, mode_sets, cascade):
    """(J, cotangents) of the variant's objectives (adjoint.objective_cotangents, input mode 1)."""
    from .adjoint import EMEObjective, objective_cotangents
    _, _, _, face, _, terms = _variant(variant)
    total = cascade.total
    blocks = ((total.s11, total.s21, mode_sets[0], mode_sets[-1]) if face == 'lo'
              else (total.s22, total.s12, mode_sets[-1], mode_sets[0]))
    objectives = [EMEObjective(*term) for term in terms]
    return objective_cotangents(objectives, face, blocks, 0, float(blocks[2].power[0]))


def _adjoint_gradient(variant, mode_cache):
    """(J, (n_slices, m) dJ/d eps per cell) of the merged-section device by adjoint.section_gradient (hz: the cell
    part plus the face part mapped by face_gradient_to_cells)."""
    from .adjoint import PeriodicAdjointFields, face_gradient_to_cells, section_gradient
    _, ends, _, face, _, _ = _variant(variant)
    runs, lines, mode_sets, cascade, periodic = _device(variant, _slice_rows(variant), True, mode_cache)
    total, cotangents = _objective(variant, mode_sets, cascade)
    incident = np.zeros(mode_sets[0 if face == 'lo' else -1].n_modes, dtype=np.complex128)
    incident[0] = 1.0
    fields = None
    if periodic is not None:
        block, cell, exit_interface = periodic
        fields = PeriodicAdjointFields(cascade, block, cell, exit_interface, _CELL_PERIODS, face, incident,
                                       cotangents)
    gradient = np.zeros((len(_cores(variant)), _LINE_CELLS))
    for index, ((first, count), line, modes) in enumerate(zip(runs, lines, mode_sets)):
        cell_rows, face_rows = section_gradient(cascade, index, line, modes, np.full(count, _SLICE_WIDTH),
                                                count * _SLICE_WIDTH, _OMEGA, face, incident, cotangents,
                                                periodic=fields)
        if face_rows is not None:
            cell_rows = cell_rows + face_gradient_to_cells(face_rows, _face_index(ends),
                                                           np.full(_LINE_CELLS, _LINE_CELL), ends[0] == 'periodic')
        gradient[first:first + count] = cell_rows
    return total, gradient, len(runs)


def _check_gradient(results_by_name):
    """Per variant and random direction d on the design block: sum(dJ/d eps d) vs (J(eps + h d) - J(eps - h d)) / 2h
    of per-slice cascades; also J of merged vs per-slice sections."""
    rng = np.random.default_rng(11)
    details, worst = {}, 0.0
    for variant in (*_VARIANTS, *_PERIODIC_VARIANTS):
        mode_cache = {}
        total, gradient, n_sections = _adjoint_gradient(variant, mode_cache)
        per_slice_sets, per_slice_cascade = _device(variant, _slice_rows(variant), False, mode_cache)[2:4]
        merge_error = abs(_objective(variant, per_slice_sets, per_slice_cascade)[0] - total)
        errors, pairs = [], []
        cells = _variant(variant)[4]
        design = _PERIODIC_DESIGN_SLICES if variant in _PERIODIC_VARIANTS else _DESIGN_SLICES
        for _ in range(_DIRECTIONS):
            direction = np.zeros_like(gradient)
            direction[slice(*design), slice(*cells)] = rng.uniform(
                -1.0, 1.0, (design[1] - design[0], cells[1] - cells[0]))
            values = []
            for sign in (1.0, -1.0):
                sets, cascade = _device(variant, _slice_rows(variant, sign * _STEP * direction), False,
                                        mode_cache)[2:4]
                values.append(_objective(variant, sets, cascade)[0])
            finite_difference = (values[0] - values[1]) / (2.0 * _STEP)
            adjoint = float(np.sum(gradient * direction))
            errors.append(abs(adjoint / finite_difference - 1.0) if finite_difference != 0.0 else math.inf)
            pairs.append([adjoint, finite_difference])
        details[variant] = {'objective': total, 'sections': n_sections, 'merged_vs_per_slice_J': merge_error,
                            'adjoint_vs_finite_difference': pairs, 'relative_errors': errors}
        worst = max(worst, max(errors), merge_error)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _GRADIENT_TOLERANCE, 'details': details}


# eme_3d_cascade_adjoint_gradient: EMEPhysics3D on a cpu Domain3D, graded window cells [nm] (7 x 5 cross-section, 9 z
# slices), anisotropic cladding / core eps (float32-exact, as the steps 2^-9 x integers), per slice the core x cells
# (window index) over the y cells _Y_CORE_3D: a staircase taper with one 2-slice section. Complete cross-section bases
# from a dense eigensolver (every eigenpair of the shift-invert operator).
_X_WIDTHS_3D = (50e-9, 40e-9, 60e-9, 50e-9, 40e-9, 60e-9, 50e-9)
_Y_WIDTHS_3D = (50e-9, 45e-9, 55e-9, 50e-9, 45e-9)
_Z_WIDTHS_3D = (60e-9, 60e-9, 70e-9, 50e-9, 60e-9, 80e-9, 60e-9, 60e-9, 60e-9)
_X_CORES_3D = ((3, 4), (3, 4), (2, 4), (2, 5), (2, 5), (1, 5), (2, 6), (1, 6), (1, 6))
_Y_CORE_3D = (1, 3)
_EPS_3D = {'eps_xx': (2.25, 4.0), 'eps_yy': (2.0, 3.5), 'eps_r': (2.5, 4.5)}   # (cladding, core)
_PML_CELLS_3D = 3
_SHIFT_INDEX_3D = 1.37        # shift-invert target: away from every eigenvalue (accurate dense eigenpairs)
_STEP_3D = 2.0 ** -9          # eps step of the central differences (eps + step x integer is float32-exact)
_DIRECTIONS_3D = 2
_BEND_SLICES_3D = (2, 7)      # 0-based slices [first, stop) of the bend variant (R = _BEND_RADIUS_3D about the x centre)
_BEND_RADIUS_3D = 1.2e-6
_CELL_SLICES_3D = (3, 6)      # unit cell slices [first, stop) of the periodic-block variant, repeated _CELL_PERIODS_3D
_CELL_PERIODS_3D = 3
_PERIODIC_LAYERS_3D = (2, 8)  # perturbed slices of the periodic variant (the cell and the sections beside it)
# variant -> ((x faces, y faces), input face, objectives (quantity, mode_number, weight), extra: None | bend | periodic);
# wall variants keep pmc y faces: a closed pec box of this size is below cutoff (no mode carries power)
_VARIANTS_3D = {
    'walls_input_lo': (('pec', 'pmc'), 'lo', (('transmission_mode', 1, 1.0), ('reflection', 1, 0.5)), None),
    'cpml_input_hi': (('cpml', 'periodic'), 'hi',
                      (('transmission', 1, 1.0), ('reflection_mode', 1, -0.3), ('transmission_mode', 2, 0.2)), None),
    'bend_walls_input_lo': (('pec', 'pmc'), 'lo', (('transmission_mode', 1, 1.0),), 'bend'),
    'cell_walls_input_hi': (('pec', 'pmc'), 'hi', (('transmission', 1, 1.0), ('reflection_mode', 1, -0.3)),
                            'periodic'),
}


def _dense_eigen_solver():
    """ArpackShiftInvertSolver whose eigs returns every eigenpair (numpy eig of the dense shift-invert operator), so
    solve_cross_section_modes gives a complete basis."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver

    class DenseEigenSolver(ArpackShiftInvertSolver):
        def eigs(self, matvec, n_unknowns, n_wanted, start_vector=None, tolerance_floor=0.0,
                 max_restarts=None, default_basis=0, device_matvec=None):   # device_matvec ignored (cpu, dense)
            identity = np.eye(n_unknowns, dtype=np.complex128)
            operator = np.column_stack([matvec(identity[:, column]) for column in range(n_unknowns)])
            theta, vectors = np.linalg.eig(operator)
            self.n_eigen_solves += 1
            self.operator_applications += n_unknowns
            return theta, vectors, n_unknowns, True

    return DenseEigenSolver()


def _padded(window_widths, n_lo, n_hi):
    """Full-axis widths: the window widths with n_lo / n_hi copies of the edge widths (PML cells)."""
    widths = np.asarray(window_widths, dtype=np.float64)
    return np.concatenate((np.full(n_lo, widths[0]), widths, np.full(n_hi, widths[-1])))


def _device_3d(variant):
    """(EMEPhysics3D of the variant on a cpu Domain3D with the dense eigensolver, its objective monitors)."""
    from ...core.domain3d import Domain3D
    from ..em3d.boundaries3d import BoundaryConfig3D
    from .solver3d import EMEPhysics3D
    (x_faces, y_faces), face, terms, extra = _VARIANTS_3D[variant]
    config = BoundaryConfig3D({'x_lo': x_faces, 'x_hi': x_faces, 'y_lo': y_faces, 'y_hi': y_faces, 'z_lo': 'pec',
                               'z_hi': 'pec'}, npml=_PML_CELLS_3D)
    sides = config.npml_sides
    x_widths = _padded(_X_WIDTHS_3D, sides['x_lo'], sides['x_hi'])
    y_widths = _padded(_Y_WIDTHS_3D, sides['y_lo'], sides['y_hi'])
    z_widths = np.asarray(_Z_WIDTHS_3D, dtype=np.float64)
    domain = Domain3D.from_widths(x_widths, y_widths, z_widths, sides, float(sum(_X_WIDTHS_3D)),
                                  float(sum(_Y_WIDTHS_3D)), float(z_widths.sum()), boundary_config=config)
    domain.use_backend('cpu')
    physics = EMEPhysics3D(domain, wavelength=_WAVELENGTH, n_modes=2 * x_widths.size * y_widths.size,
                           n_eff_guess=_SHIFT_INDEX_3D, eigen_solver=_dense_eigen_solver())
    physics.set_excitation(face, 1)
    edges = np.concatenate(([0.0], np.cumsum(z_widths)))
    if extra == 'bend':
        physics.add_bend(edges[_BEND_SLICES_3D[0]], edges[_BEND_SLICES_3D[1]], _BEND_RADIUS_3D,
                         0.5 * sum(_X_WIDTHS_3D))
    elif extra == 'periodic':
        physics.add_periodic(edges[_CELL_SLICES_3D[0]], edges[_CELL_SLICES_3D[1]], _CELL_PERIODS_3D, 2)
    return physics, [physics.add_objective(*term) for term in terms]


def _base_permittivity_3d(domain):
    """{eps_xx, eps_yy, eps_r: float64 (nx, ny, nz)} cladding with the core of every z slice (window cell indices)."""
    arrays = {}
    for name, (cladding, core) in _EPS_3D.items():
        values = np.full(domain.grid.shape, cladding)
        for layer, (x0, x1) in enumerate(_X_CORES_3D):
            values[domain.ox + x0:domain.ox + x1, domain.oy + _Y_CORE_3D[0]:domain.oy + _Y_CORE_3D[1], layer] = core
        arrays[name] = values
    return arrays


def _objective_3d(physics, objectives, permittivity):
    """J = sum weight * value of the objectives after writing `permittivity` (float32-exact values) and solving."""
    for name, values in permittivity.items():
        getattr(physics.domain.materials, name)[...] = values
    physics.materials_changed(tuple(permittivity))
    physics.step()
    return float(sum(objective.weight * objective.values[physics.solved_omega] for objective in objectives))


def _check_gradient_3d(results_by_name):
    """Per variant and random integer direction d (every cell of the perturbed slices, all three arrays): sum(dJ/d eps
    d) of EMEPhysics3D.eps_sensitivity vs the central difference of J; also its J vs the objectives' sum."""
    rng = np.random.default_rng(13)
    details, worst = {}, 0.0
    for variant, (_, _, _, extra) in _VARIANTS_3D.items():
        physics, objectives = _device_3d(variant)
        base = _base_permittivity_3d(physics.domain)
        total = _objective_3d(physics, objectives, base)
        sensitivity = physics.eps_sensitivity()   # before the difference solves change the state
        gradients = {name: np.asarray(values, dtype=np.float64) for name, values in sensitivity.cell_gradients.items()}
        first, stop = _PERIODIC_LAYERS_3D if extra == 'periodic' else (1, len(_Z_WIDTHS_3D) - 1)
        errors, pairs = [], []
        for _ in range(_DIRECTIONS_3D):
            direction = {}
            for name, base_values in base.items():
                direction[name] = np.zeros(base_values.shape)
                direction[name][:, :, first:stop] = rng.integers(-2, 3, base_values[:, :, first:stop].shape)
            shifted = [_objective_3d(physics, objectives, {name: base[name] + sign * _STEP_3D * direction[name]
                                                           for name in base}) for sign in (1.0, -1.0)]
            finite_difference = (shifted[0] - shifted[1]) / (2.0 * _STEP_3D)
            adjoint = float(sum(np.sum(gradients[name] * direction[name]) for name in base))
            errors.append(abs(adjoint / finite_difference - 1.0) if finite_difference != 0.0 else math.inf)
            pairs.append([adjoint, finite_difference])
        objective_error = abs(sensitivity.objective - total)
        details[variant] = {'objective': total, 'sections': len(physics.sections),
                            'modes_per_section': sorted({int(row[4]) for row in physics.section_rows}),
                            'objective_error': objective_error, 'adjoint_vs_finite_difference': pairs,
                            'relative_errors': errors}
        worst = max(worst, max(errors), objective_error)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _GRADIENT_TOLERANCE, 'details': details}


# eme_taper_optimization: em_eme along x (pec faces, 50 nm mesh, n_modes 0) from a 0.5 um guide (x 0..0.25 um) to a
# 2 um guide (x 1.75..2 um), both centred in a 3 um tall box; the 1.5 x 2 um gap between them (cladding at density 0) is
# the design region; J = transmission into mode 1 of the wide guide. eme_taper_optimization_hz: the same in hz (the
# region writes eps_xx and eps_yy, the gradient sums both). The guides are single sections (phase only): 0.25 um keeps
# 5 cells between the design box and the window ends (design regions on the first / last slice raise).
_TAPER_CELL = 50e-9
_TAPER_SIZE = (2.0e-6, 3.0e-6)
_TAPER_CENTRE = 1.5e-6
_TAPER_INPUT = (0.25e-6, 0.5e-6)     # (x stop, width)
_TAPER_OUTPUT = (1.75e-6, 2.0e-6)    # (x start, width)
_TAPER_DESIGN_Y = (0.5e-6, 2.5e-6)
_TAPER_ITERATIONS = 10
_TAPER_CLOSURE = 0.8                 # (J_final - J0) / (1 - J0) at least (J <= 1: a gain ratio is capped by 1 / J0)
_DESIGN_FEATURE = 'design'


def _taper_spec(polarization='ez'):
    """Gap between a narrow and a wide guide as an em_eme design region (initial density 0; ez writes eps_r, hz eps_xx
    and eps_yy), J = transmission_mode 1."""
    size_x, size_y = _TAPER_SIZE
    (input_stop, narrow), (output_start, wide) = _TAPER_INPUT, _TAPER_OUTPUT

    def strip(name, x0, x1, width):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': _TAPER_CENTRE - 0.5 * width, 'x1': x1,
                                                'y1': _TAPER_CENTRE + 0.5 * width})

    return ModelSpec(
        title=f'Self-test: em_eme {polarization} taper optimisation',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[strip('input_guide', 0.0, input_stop, narrow), strip('output_guide', output_start, size_x, wide),
                  GeometrySpec('gap', 'rectangle', {'x0': input_stop, 'y0': _TAPER_DESIGN_Y[0], 'x1': output_start,
                                                    'y1': _TAPER_DESIGN_Y[1]})],
        materials=[MaterialSpec('cladding', properties={'eps_r': _CORE_EPS[0]}, selection=['all']),
                   MaterialSpec('core', properties={'eps_r': _CORE_EPS[1]}, selection=['input_guide', 'output_guide'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings={
            'polarization': polarization, 'propagation_axis': 'x', 'wavelength': _WAVELENGTH, 'n_modes': 0,
            'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec'}, features=[
            FeatureSpec('ports', 'eme_ports'),
            FeatureSpec('transmission', 'adjoint_objective',
                        params={'quantity': 'transmission_mode', 'mode_number': 1, 'weight': 1.0}),
            FeatureSpec(_DESIGN_FEATURE, 'design_region', ['gap'],
                        {'eps_min': _CORE_EPS[0], 'eps_max': _CORE_EPS[1], 'initial_density': 0.0,
                         'filter_radius': 75e-9, 'projection_beta': 0.0, 'projection_eta': 0.5})])],
        mesh=MeshSpec(mode='uniform', dx=_TAPER_CELL),
        study=StudySpec(type='optimization', settings={'max_iterations': _TAPER_ITERATIONS, 'tolerance': 1e-6,
                                                       'live_update_hz': 0.0}))


def _check_taper(results_by_name):
    """Closure (J_final - J0) / (1 - J0) of the design_region history vs _TAPER_CLOSURE; details: history, gain ratio,
    monotonicity, density range."""
    results = results_by_name['taper']
    history = results.result(_PHYSICS, _DESIGN_FEATURE)
    objectives = np.asarray(history['series']['objective'], dtype=np.float64)
    density = np.asarray(results.field(f'{_PHYSICS}.design_density').values, dtype=np.float64)
    closure = float((objectives[-1] - objectives[0]) / (1.0 - objectives[0])) if objectives[0] < 1.0 else math.nan
    error = max(0.0, _TAPER_CLOSURE - closure) if math.isfinite(closure) else math.inf
    details = {'objective_history': objectives.tolist(), 'iterations': int(objectives.size - 1),
               'gain_J_final_over_J0': float(objectives[-1] / objectives[0]) if objectives[0] > 0.0 else math.nan,
               'monotone': bool(np.all(np.diff(objectives) >= -1e-9 * np.abs(objectives[:-1]))),
               'final_grayness': history['final_grayness'], 'density_range': [float(density.min()),
                                                                              float(density.max())],
               'optimization': results.summary().get('optimization')}
    return {'measured': closure, 'expected': _TAPER_CLOSURE, 'error': error, 'tolerance': 0.0, 'details': details}


EME_ADJOINT_SELFTEST_CASES = (
    SelfTestCase(
        name='eme_cascade_adjoint_gradient',
        label='EME: cascade adjoint gradient',
        description='CPU check of physics/eme/adjoint.py: lines of 48 cells of 25 nm (eps 4 core in 2.25), 13 '
                    'slices of 40 nm along the axis (narrow input, a staircase taper containing a 3-slice section, '
                    'wide output), complete bases, 1.55 um. ez variants walls_input_lo (dirichlet ends, mode 1 from lo, '
                    'J = transmission_mode 1 + 0.5 reflection) and cpml_input_hi (quadratic complex-stretch ends with '
                    'the PML modes, mode 1 from hi, J = transmission - 0.3 reflection_mode 1 + 0.2 transmission_mode '
                    '2); hz variants (isotropic eps as eps_t of the cells and the face mean eps_l, as PortLine) '
                    'hz_walls_input_lo (odd lo / dirichlet hi ends, J as walls_input_lo), hz_cpml_input_hi (as '
                    'cpml_input_hi) and hz_periodic_input_lo (periodic line, J = transmission - 0.5 reflection). '
                    'Periodic-block variants (10 slices: narrow input, a 4-slice unit cell repeated 4 times by the '
                    'star-power link of the em_eme solve, wide output) cell_walls_input_lo (ez, as walls_input_lo) and '
                    'hz_cell_walls_input_hi (hz odd / dirichlet ends, mode 1 from hi, J = transmission - 0.3 '
                    'reflection_mode 1): a drawn cell slice stands for every period, so its gradient sums the '
                    'per-period field products (PeriodicAdjointFields). '
                    'dJ/d eps of section_gradient (forward and adjoint cascade fields, closed-form slice integrals, '
                    'merged sections; hz: the cell part plus the face part by face_gradient_to_cells) along 2 random '
                    'directions on slices 4-10 (periodic variants 3-9: the cell and the sections beside it) and line '
                    'cells 10-37 (cpml and ez wall variants) or every line cell (hz walls / periodic: the end faces) '
                    'must equal the central difference (step 1e-3) of per-slice cascades, and J of merged and '
                    'per-slice sections agree (equal slice rows share one mode solve per variant: unperturbed slices '
                    'are not re-solved). error = the largest relative deviation; tolerance 1e-4 (a first guess: '
                    'round-off / step and the O(step^2) term are ~1e-6). Not yet run with 2 directions and the shared '
                    'mode solves.',
        specs=lambda: {},
        check=_check_gradient,
        max_wall_time=300.0),
    SelfTestCase(
        name='eme_3d_cascade_adjoint_gradient',
        label='EME 3D: cascade adjoint gradient',
        description='CPU check of EMEPhysics3D.eps_sensitivity (adjoint.cross_section_gradient) without a model run: a '
                    'cpu Domain3D of 7 x 5 graded cross-section cells (40-60 nm) and 9 z slices (50-80 nm), anisotropic '
                    'cladding eps_xx / eps_yy / eps_r = 2.25 / 2.0 / 2.5 with an off-centre core 4.0 / 3.5 / 4.5 whose '
                    'x width steps along z (one 2-slice section), 1.55 um, complete cross-section bases (a dense '
                    'eigensolver returns every eigenpair of the shift-invert operator). Variants walls_input_lo (pec x '
                    '/ pmc y walls, mode 1 from lo, J = transmission_mode 1 + 0.5 reflection), cpml_input_hi (cpml x '
                    'faces of 3 cells with the PML modes, periodic y, mode 1 from hi, J = transmission - 0.3 '
                    'reflection_mode 1 + 0.2 transmission_mode 2), bend_walls_input_lo (pec x / pmc y walls, slices '
                    "3-7 an eme_bend of R = 1.2 um about the x centre: the 'exact' bend factors of the E points, J = "
                    'transmission_mode 1) and cell_walls_input_hi (slices 4-6 an eme_periodic cell of 3 periods: the '
                    'gradient sums the periods, J = transmission - 0.3 reflection_mode 1). dJ/d eps_xx, eps_yy, eps_r '
                    '(stretched dual-area weights of the E points, the transposed face means, wall ghost fill and bend '
                    'factors) along 2 random integer directions on every cross-section cell (PML and wall ghost cells '
                    'included) of the inner slices 2-8 (periodic variant 3-8) must equal the central difference (step '
                    '2^-9, float32-exact) of the solver J, and its EpsSensitivity J the objectives. error = the largest '
                    'relative deviation; tolerance 1e-4 (a first guess, as eme_cascade_adjoint_gradient).',
        specs=lambda: {},
        check=_check_gradient_3d,
        max_wall_time=300.0),
    SelfTestCase(
        name='eme_taper_optimization',
        label='EME: taper optimisation',
        description=f'em_eme along x (ez, 1.55 um, {_TAPER_CELL * 1e9:g} nm mesh, pec faces, n_modes 0): a 0.5 um guide '
                    f'(eps 4 in 2.25, x 0..{_TAPER_INPUT[0] * 1e6:g} um) and a 2 um guide (x '
                    f'{_TAPER_OUTPUT[0] * 1e6:g}..{_TAPER_SIZE[0] * 1e6:g} um) in a 3 um tall box; the 1.5 x 2 um gap '
                    'between them is a design region (initial density 0, filter radius 75 nm, no projection) and the '
                    'objective the transmission into mode 1 of the wide guide. An optimization study of at most '
                    f'{_TAPER_ITERATIONS} L-BFGS-B iterations on the cascade-adjoint gradient must close at least '
                    f'{_TAPER_CLOSURE:g} of the gap 1 - J0 (J is a power fraction, so a gain ratio J_final / J0 is '
                    'capped by 1 / J0 ~ 1.49 here; measured 0.947 ez / 0.921 hz). measured = (J_final - J0) / (1 - J0) '
                    f'of the design_region history, expected {_TAPER_CLOSURE:g}, error = the shortfall below it, '
                    'tolerance 0. It pins em_eme in the study loop (eps_r writes, section re-solves, eps_sensitivity, '
                    'the history and design_density), not accuracy.',
        specs=lambda: {'taper': _taper_spec()},
        check=_check_taper,
        max_wall_time=900.0),
    SelfTestCase(
        name='eme_taper_optimization_hz',
        label='EME: taper optimisation (hz)',
        description='eme_taper_optimization with polarization hz: the design region writes eps_xx and eps_yy (an '
                    'isotropic change) and the cascade-adjoint gradient sums both (the u\' u\' cell term of eps_t and '
                    'the face conductances of eps_l mapped to the cells). Same geometry, mesh, objective, study and '
                    f'pass rule (closure of 1 - J0 >= {_TAPER_CLOSURE:g} within {_TAPER_ITERATIONS} L-BFGS-B '
                    'iterations; J0 0.784, so a gain ratio is capped by 1.28). It pins the hz adjoint in the study '
                    'loop (eps_xx / eps_yy writes, eps_sensitivity of both arrays, the history and design_density).',
        specs=lambda: {'taper': _taper_spec('hz')},
        check=_check_taper,
        max_wall_time=900.0),
)

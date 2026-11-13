"""Self-tests of the optimization study (C7): the design-density chain rule against finite differences (CPU) and a
small em_fdfd_ez waveguide-gap design that must raise its transmission objective (model run); the
waveguide_bend_optimization template."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import vertical_line
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_fdfd import cpu_specs
from .selftests_mode_ports import _SILICA_EPS_R, _SILICON_EPS_R, _SLAB_CELL, _SLAB_SIZE, _SLAB_Y_RANGE

_STUDY = 'optimization'   # = model.study_optimization.STUDY_NAME
_PHYSICS_NAME = 'em'

# em_fdfd_design_chain_rule: a disk of design cells on a graded 24 x 20 grid (16-24 nm cells), per configuration
# (filter radius [m], projection beta, eta); objective F = sum(c eps + d eps^2 / 24) with random c, d.
_CHAIN_CONFIGURATIONS = ((55e-9, 6.0, 0.45), (0.0, 0.0, 0.5))
_CHAIN_STEP = 1e-6            # density step of the central differences
_CHAIN_DIRECTIONS = 4         # random directions per configuration
_CHAIN_TOLERANCE = 1e-6       # relative

# em_fdfd_topology_gap: the 4 x 2 um all-CPML box, 20 nm mesh and 0.28 um silicon slab of em_fdfd_mode_port_slab, cut
# by a 0.8 um silica gap (x 1.6..2.4 um) that is the design region over y 0.7..1.3 um (40 x 30 cells). Mode source
# (mode 1, +x, 1 W/m) at x = 1.01 um, objective = mode_power_positive of mode 1 at x = 3.01 um (both full height).
_GAP_X = (1.6e-6, 2.4e-6)          # m
_DESIGN_Y = (0.7e-6, 1.3e-6)       # m
_SOURCE_X = 1.01e-6                # m
_OUTPUT_X = 3.01e-6                # m
_WAVELENGTH = 1.55e-6              # m
_FILTER_RADIUS = 60e-9             # m, 3 cells
# a 15-iteration run recorded J / J0 = 1.50 at iteration 4, 1.88 at 6 and 2.38 at 8 (5.1 at 15, after the design
# snapped to the upper bound at iteration 10, which the check never needed)
_MAX_ITERATIONS = 8
_REQUIRED_GAIN = 1.5               # J_final / J_initial at least
_DESIGN_FEATURE = 'design'
_OBJECTIVE_FEATURE = 'transmission'

# waveguide_bend_optimization template: 3 x 3 um all-CPML box, uniform 20 nm mesh (window cell k spans [20 k, 20 (k +
# 1)] nm: strip edges on cell faces, lines at cell centres). The 0.28 um silicon strip of the slab tests enters along +x
# at y 1.46..1.74 um and leaves along +y at x 1.46..1.74 um; the 1.2 um square between them (x, y 1.0..2.2 um, 60 x 60
# cells) is the design region. Source at x = 0.51 um, objective = mode 1 power toward +y at y = 2.61 um.
BEND_LABEL = 'Waveguide bend optimization'
BEND_DESCRIPTION = (
    'EM FDFD (E_z) at 1.55 um, optimization study: a 0.28 um silicon strip in silica turns by 90 degrees inside a 1.2 x '
    '1.2 um design region (density from silica to silicon, filter radius 80 nm, tanh projection reaching beta 8 by a '
    'continuation over 3 stages: beta 2, 4, 8, at most 25 L-BFGS-B iterations each). Objective: the mode 1 power '
    'leaving along +y (mode_power_positive). The design_density field is stored per iteration; the design_region '
    'result shows J and the grayness versus iteration.')
BEND_SUMMARY = ('Lets the optimiser shape a compact 90 degree bend between two silicon waveguides at 1550 nm: the '
                'material of a small square is changed cell by cell to send the most light into the output waveguide.')
_BEND_SIZE = 3.0e-6                # m, square domain
_BEND_DESIGN = (1.0e-6, 2.2e-6)    # m, x and y range of the design square
_BEND_STRIP = (1.46e-6, 1.74e-6)   # m, input strip y range = output strip x range
_BEND_SOURCE_X = 0.51e-6           # m
_BEND_OUTPUT_Y = 2.61e-6           # m


def _graded_grid(rng, n_cells):
    """(centres, widths) float64 of n_cells cells of random width 16-24 nm from 0."""
    widths = rng.uniform(16e-9, 24e-9, n_cells)
    return np.cumsum(widths) - 0.5 * widths, widths


def _check_chain_rule(results_by_name):
    """Per configuration: dJ/d rho of DesignMap.density_gradient vs central differences of F(eps(rho)) along random
    directions, and filter rows summing to one; measured = error = worst relative deviation / tolerance."""
    from .design_density import DesignMap
    rng = np.random.default_rng(7)
    x_centres, x_widths = _graded_grid(rng, 24)
    y_centres, y_widths = _graded_grid(rng, 20)
    cell_i, cell_j = np.meshgrid(np.arange(3, 21), np.arange(2, 18), indexing='ij')
    inside = (cell_i - 12) ** 2 + (cell_j - 10) ** 2 <= 64
    cell_i, cell_j = cell_i[inside], cell_j[inside]
    details, errors = {'n_cells': int(cell_i.size)}, {}
    for radius, beta, eta in _CHAIN_CONFIGURATIONS:
        label = f'radius {radius * 1e9:g} nm, beta {beta:g}'
        design = DesignMap(cell_i, cell_j, x_centres, y_centres, x_widths, y_widths, _SILICA_EPS_R, _SILICON_EPS_R,
                           filter_radius=radius, projection_beta=beta, projection_eta=eta)
        linear, quadratic = rng.normal(size=design.n_cells), rng.normal(size=design.n_cells)

        def objective(density):
            eps = design.permittivity(density)
            return float(np.sum(linear * eps + quadratic * eps ** 2 / 24.0))

        density = rng.uniform(0.1, 0.9, design.n_cells)
        eps = design.permittivity(density)
        gradient = design.density_gradient(density, linear + quadratic * eps / 12.0)
        worst = 0.0
        for _ in range(_CHAIN_DIRECTIONS):
            direction = rng.normal(size=design.n_cells)
            finite_difference = (objective(density + _CHAIN_STEP * direction)
                                 - objective(density - _CHAIN_STEP * direction)) / (2.0 * _CHAIN_STEP)
            worst = max(worst, abs(float(gradient @ direction) / finite_difference - 1.0))
        partition = float(np.max(np.abs(design.filtered(np.full(design.n_cells, 0.3)) - 0.3)))
        errors[label] = max(worst, partition) / _CHAIN_TOLERANCE
        details[label] = {'gradient_relative_error': worst, 'filter_partition_error': partition}
    worst_item = max(errors, key=lambda item: errors[item])
    details.update(normalized_errors=errors, worst_item=worst_item)
    worst = errors[worst_item]
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


def _gap_spec():
    """Slab cut by the design gap: mode source, transmission objective, design region (initial density 0)."""
    size_x, size_y = _SLAB_SIZE
    y_lo, y_hi = _SLAB_Y_RANGE
    gap_lo, gap_hi = _GAP_X
    design_lo, design_hi = _DESIGN_Y
    return ModelSpec(
        title='Self-test: em_fdfd_ez topology optimisation of a waveguide gap',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('input_slab', 'rectangle', {'x0': 0.0, 'y0': y_lo, 'x1': gap_lo, 'y1': y_hi}),
                  GeometrySpec('output_slab', 'rectangle', {'x0': gap_hi, 'y0': y_lo, 'x1': size_x, 'y1': y_hi}),
                  GeometrySpec('gap', 'rectangle', {'x0': gap_lo, 'y0': design_lo, 'x1': gap_hi, 'y1': design_hi}),
                  vertical_line('source_line', _SOURCE_X, 0.0, size_y),
                  vertical_line('output_line', _OUTPUT_X, 0.0, size_y)],
        materials=[MaterialSpec('silica', properties={'eps_r': _SILICA_EPS_R}, selection=['all']),
                   MaterialSpec('silicon', properties={'eps_r': _SILICON_EPS_R}, selection=['input_slab',
                                                                                           'output_slab'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdfd_ez', settings={'wavelength': _WAVELENGTH}, features=[
            FeatureSpec('source', 'mode_source', ['source_line'],
                        {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}),
            FeatureSpec(_OBJECTIVE_FEATURE, 'adjoint_objective', ['output_line'],
                        {'quantity': 'mode_power_positive', 'mode_number': 1, 'weight': 1.0}),
            FeatureSpec(_DESIGN_FEATURE, 'design_region', ['gap'],
                        {'eps_min': _SILICA_EPS_R, 'eps_max': _SILICON_EPS_R, 'initial_density': 0.0,
                         'filter_radius': _FILTER_RADIUS, 'projection_beta': 0.0, 'projection_eta': 0.5})])],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=StudySpec(type=_STUDY, settings={'max_iterations': _MAX_ITERATIONS, 'tolerance': 1e-6,
                                               'live_update_hz': 0.0}))


def _check_gap(results_by_name):
    """J_final / J_initial of the design_region history vs _REQUIRED_GAIN; measured = the ratio, error = the
    shortfall below it; details: history, monotonicity, design density range."""
    results = results_by_name['gap']
    history = results.result(_PHYSICS_NAME, _DESIGN_FEATURE)
    objectives = np.asarray(history['series']['objective'], dtype=np.float64)
    density = np.asarray(results.field(f'{_PHYSICS_NAME}.design_density').values, dtype=np.float64)
    ratio = float(objectives[-1] / objectives[0]) if objectives[0] > 0.0 else math.nan
    error = max(0.0, _REQUIRED_GAIN - ratio) if math.isfinite(ratio) else math.inf
    details = {'objective_history': objectives.tolist(), 'iterations': int(objectives.size - 1),
               'monotone': bool(np.all(np.diff(objectives) >= -1e-9 * np.abs(objectives[:-1]))),
               'final_grayness': history['final_grayness'], 'mean_density': history['mean_density'],
               'density_range': [float(density.min()), float(density.max())],
               'optimization': results.summary().get('optimization')}
    return {'measured': ratio, 'expected': _REQUIRED_GAIN, 'error': error, 'tolerance': 0.0, 'details': details}


def waveguide_bend_optimization_spec():
    """Template 'waveguide_bend_optimization': em_fdfd_ez 90 degree bend design with beta continuation."""
    design_lo, design_hi = _BEND_DESIGN
    strip_lo, strip_hi = _BEND_STRIP
    return ModelSpec(
        title='Waveguide bend optimization (E_z, 1.55 um)',
        domain=DomainSpec(size_x=_BEND_SIZE, size_y=_BEND_SIZE),
        geometry=[GeometrySpec('input_strip', 'rectangle',
                               {'x0': 0.0, 'y0': strip_lo, 'x1': design_lo, 'y1': strip_hi}),
                  GeometrySpec('output_strip', 'rectangle',
                               {'x0': strip_lo, 'y0': design_hi, 'x1': strip_hi, 'y1': _BEND_SIZE}),
                  GeometrySpec('bend', 'rectangle', {'x0': design_lo, 'y0': design_lo, 'x1': design_hi,
                                                     'y1': design_hi}),
                  vertical_line('source_line', _BEND_SOURCE_X, 0.0, _BEND_SIZE),
                  GeometrySpec('output_line', 'line', {'x0': 0.0, 'y0': _BEND_OUTPUT_Y, 'x1': _BEND_SIZE,
                                                       'y1': _BEND_OUTPUT_Y})],
        materials=[MaterialSpec('silica', properties={'eps_r': _SILICA_EPS_R}, selection=['all']),
                   MaterialSpec('silicon', properties={'eps_r': _SILICON_EPS_R},
                                selection=['input_strip', 'output_strip'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdfd_ez', settings={'wavelength': _WAVELENGTH}, features=[
            FeatureSpec('source', 'mode_source', ['source_line'],
                        {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}),
            FeatureSpec(_OBJECTIVE_FEATURE, 'adjoint_objective', ['output_line'],
                        {'quantity': 'mode_power_positive', 'mode_number': 1, 'weight': 1.0}),
            FeatureSpec(_DESIGN_FEATURE, 'design_region', ['bend'],
                        {'eps_min': _SILICA_EPS_R, 'eps_max': _SILICON_EPS_R, 'initial_density': 0.5,
                         'filter_radius': 80e-9, 'projection_beta': 8.0, 'projection_eta': 0.5})])],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        study=StudySpec(type=_STUDY, settings={'max_iterations': 25, 'tolerance': 1e-5, 'continuation_stages': 3,
                                               'beta_growth': 2.0}))


OPTIMIZATION_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_design_chain_rule',
        label='Design density filter and projection chain rule',
        description='CPU only: design cells on a disk of a graded 24 x 20 grid (cell widths 16-24 nm), mapped by '
                    'design_density.DesignMap (silica 2.085 to silicon 12.08) with (filter radius 55 nm, projection '
                    'beta 6, eta 0.45) and with no filter or projection. For F = sum(c eps + d eps^2 / 24) with random '
                    'c, d at random densities in [0.1, 0.9] the chain-rule gradient dF/d rho (interpolation, tanh '
                    'projection, transposed area-weighted conic filter) is compared with central differences (step '
                    '1e-6) along 4 random directions; the filter must also keep a constant density (rows sum to 1). '
                    'measured = error = worst relative deviation / 1e-6, expected 0, tolerance 1: float64 round-off '
                    'and the O(step^2) difference error are far below it.',
        specs=lambda: {},
        check=_check_chain_rule),
    SelfTestCase(
        name='em_fdfd_topology_gap',
        label='EM FDFD topology optimisation of a waveguide gap',
        description='The 0.28 um silicon slab of em_fdfd_mode_port_slab (4 x 2 um all-CPML box, 20 nm mesh, '
                    'em_fdfd_ez at 1.55 um, cpu backend) is cut by a 0.8 um silica gap, which is a design region over '
                    'y 0.7..1.3 um (40 x 30 cells, initial density 0, filter radius 60 nm, no projection). A mode '
                    'source launches mode 1 (1 W/m) toward the gap; the objective is the mode 1 power beyond it '
                    f'(mode_power_positive). An optimization study of at most {_MAX_ITERATIONS} L-BFGS-B iterations '
                    'must raise J by at least the factor 1.5: the diffracting gap passes ~0.1 of the power, a bridged '
                    'gap almost all (a 15-iteration gpu run reached 1.5 at iteration 4 and 2.38 at 8; the 8-iteration '
                    'cpu run is not yet confirmed). measured = J_final / J_initial of the design_region history, expected 1.5, error = the '
                    'shortfall below 1.5, tolerance 0. It pins the study loop (density writes, re-solves, adjoint '
                    'gradient through filter and projection, L-BFGS-B bounds), the history result and the '
                    'design_density field.',
        specs=lambda: cpu_specs({'gap': _gap_spec()}),
        check=_check_gap,
        max_wall_time=900.0),
)

"""Finite-difference validation of the FDFD adjoint sensitivity (adjoint_objective features, output fields
eps_sensitivity / eps_xy_sensitivity) in em_fdfd_ez and em_fdfd_hz (also with eps_xy); no physics imports."""
import math
from dataclasses import replace

import numpy as np

from ...model.schema import SelfTestCase
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec,
                           ModelSpec, PhysicsSpec, StudySpec)
from ...model.selftest_helpers import single_value, vertical_line
from .selftests import normalized_result
from .selftests_fdfd import cpu_specs
from .selftests_mode_ports import _SILICA_EPS_R, _SILICON_EPS_R, _SLAB_SIZE

_PHYSICS_NAME = 'em'
_PHYSICS_TYPES = {'ez': 'em_fdfd_ez', 'hz': 'em_fdfd_hz'}
_POLARIZATIONS = ('ez', 'hz')

# em_fdfd_adjoint_gradient: the 4 x 2 um all-CPML box of em_fdfd_mode_port_slab on its own 40 nm mesh (window cell k
# spanning [40 k, 40 (k + 1)] nm; (100 + 30) x (50 + 30) = 10,400 unknowns: the adjoint is the exact derivative of the
# discrete objective on any mesh, and silicon keeps ~11 cells per wavelength with the slab single-mode guided). The 0.28
# um silicon slab (rows 22..28, y 0.88..1.16 um) ends in a facet at x = 2.52 um (cell face 63), so its fundamental mode
# partly reflects and radiates into silica. Vertical lines at cell centres: mode source x = 1.02 um (cell 25, full
# height, mode 1 toward +x, 1 W/m); the reflection objective (mode_power_negative of mode 1) x = 0.50 um (cell 12, full
# height, upstream: only the reflected field); the transmission objective (flux) x = 3.50 um (cell 87) over y 1..2 um
# only (cells 25..49: the upper half, so it changes at first order with an asymmetric perturbation). The perturbed
# block x [2.32, 2.40) x y [1.04, 1.12) um = cells 58..59 x 26..27 lies in the upper half of the silicon, 0.12 um
# before the facet, off every line.
_WAVELENGTH = 1.55e-6                     # m
_ADJOINT_CELL = 40e-9                     # m
_ADJOINT_SLAB_Y = (0.88e-6, 1.16e-6)      # m, slab y range on 40 nm faces
_FACET_X = 2.52e-6                        # m
_LINES = (('source_line', 1.02e-6, 0.0), ('reflection_line', 0.50e-6, 0.0), ('transmission_line', 3.50e-6, 1.0e-6))
_BLOCK = (2.32e-6, 1.04e-6, 2.40e-6, 1.12e-6)   # m (x0, y0, x1, y1), 2 x 2 cells
_BLOCK_STEP = 0.05                        # eps_r step of the central difference (silicon 12.08 +- 0.05)
_REFLECTION = 'reflection'
_TRANSMISSION = 'transmission'
_OBJECTIVES = (_REFLECTION, _TRANSMISSION)
_GRADIENT_VALUE = 'block_gradient'        # derived value [W/m per unit eps_r]: eps_sensitivity integrated over block
_TOLERANCE = 2e-3                         # relative, adjoint vs central difference


def _gradient_run(polarization, objective):
    return f'{polarization}_{objective}_gradient'


def _adjoint_spec(polarization, block_eps_r, reflection_weight, transmission_weight):
    """Terminated slab with the mode source, the two weighted objectives, the block material and the derived
    integral of eps_sensitivity over the block."""
    size_x, size_y = _SLAB_SIZE
    y_lo, y_hi = _ADJOINT_SLAB_Y
    x0, y0, x1, y1 = _BLOCK
    physics_type = _PHYSICS_TYPES[polarization]
    return ModelSpec(
        title=f'Self-test: {physics_type} adjoint sensitivity at a slab facet',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': y_lo, 'x1': _FACET_X, 'y1': y_hi}),
                  GeometrySpec('block', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}),
                  *(vertical_line(name, x, y_start, size_y) for name, x, y_start in _LINES)],
        materials=[MaterialSpec('silica', properties={'eps_r': _SILICA_EPS_R}, selection=['all']),
                   MaterialSpec('silicon', properties={'eps_r': _SILICON_EPS_R}, selection=['slab']),
                   MaterialSpec('block', properties={'eps_r': block_eps_r}, selection=['block'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, physics_type, settings={'wavelength': _WAVELENGTH}, features=[
            FeatureSpec('source', 'mode_source', ['source_line'],
                        {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}),
            FeatureSpec(_REFLECTION, 'adjoint_objective', ['reflection_line'],
                        {'quantity': 'mode_power_negative', 'mode_number': 1, 'weight': reflection_weight}),
            FeatureSpec(_TRANSMISSION, 'adjoint_objective', ['transmission_line'],
                        {'quantity': 'flux', 'weight': transmission_weight})])],
        mesh=MeshSpec(mode='uniform', dx=_ADJOINT_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec(_GRADIENT_VALUE, f'{_PHYSICS_NAME}_eps_sensitivity', ['block'], 'integral')])


def _adjoint_specs():
    """Per polarisation: one gradient run per objective (block = silicon, only that objective weighted), then the
    block at eps_r +- _BLOCK_STEP."""
    specs = {}
    for polarization in _POLARIZATIONS:
        specs[_gradient_run(polarization, _REFLECTION)] = _adjoint_spec(polarization, _SILICON_EPS_R, 1.0, 0.0)
        specs[_gradient_run(polarization, _TRANSMISSION)] = _adjoint_spec(polarization, _SILICON_EPS_R, 0.0, 1.0)
        for sign, label in ((1.0, 'plus'), (-1.0, 'minus')):
            specs[f'{polarization}_{label}'] = _adjoint_spec(polarization, _SILICON_EPS_R + sign * _BLOCK_STEP,
                                                             1.0, 1.0)
    return specs


def _objective_value(results, objective):
    """The objective quantity [W/m] of a single-wavelength run."""
    return single_value(results.result(_PHYSICS_NAME, objective)['series']['value'], f'{objective} objective')


def _check_adjoint_gradient(results_by_name):
    """Per polarisation and objective: the block integral of eps_sensitivity vs the central difference of the
    objective; measured = error = the worst relative deviation in tolerance units."""
    details, normalized_errors = {'unit': 'W/m per unit eps_r', 'block_step': _BLOCK_STEP}, {}
    for polarization in _POLARIZATIONS:
        for objective in _OBJECTIVES:
            plus = _objective_value(results_by_name[f'{polarization}_plus'], objective)
            minus = _objective_value(results_by_name[f'{polarization}_minus'], objective)
            finite_difference = (plus - minus) / (2.0 * _BLOCK_STEP)
            gradient_results = results_by_name[_gradient_run(polarization, objective)]
            adjoint = float(gradient_results.summary()['derived'][_GRADIENT_VALUE])   # float() also reads 'nan'
            error = abs(adjoint / finite_difference - 1.0) if finite_difference != 0.0 else math.nan
            normalized_errors[f'{polarization} {objective}'] = error / _TOLERANCE
            details.update({f'{polarization}_{objective}_value': _objective_value(gradient_results, objective),
                            f'{polarization}_{objective}_adjoint': adjoint,
                            f'{polarization}_{objective}_finite_difference': finite_difference,
                            f'{polarization}_{objective}_relative_error': error})
    return normalized_result(normalized_errors, details)


# em_fdfd_hz_eps_xy_adjoint_gradient: the em_fdfd_adjoint_gradient layout in em_fdfd_hz, with the last 0.4 um of the
# silicon slab before the facet (x [2.12, 2.52) um, cells 53..62, slab rows) replaced by a rotated uniaxial crystal
# (n_o 3.3, n_e 3.6, optic axis 30 deg from x in the plane: eps_xx, eps_yy, eps_xy), so the 9-point operator acts
# around the block, which holds the same crystal. Sources, ports and objective lines stay in eps_xy-free cells. Diagonal
# runs move eps_xx and eps_yy of the block together by +-_BLOCK_STEP at fixed eps_xy (the eps_sensitivity chain rule
# through eps_ii - eps_xy^2 / eps_jj and eta_xy); off-diagonal runs move eps_xy by +-_BLOCK_STEP (eps_xy_sensitivity).
_CRYSTAL_X_RANGE = (2.12e-6, _FACET_X)    # m
_CRYSTAL_INDICES = (3.3, 3.6)             # (n_o, n_e)
_CRYSTAL_AXIS_ANGLE = 30.0                # deg from +x
_XY_GRADIENT_VALUE = 'block_xy_gradient'  # derived value [W/m per unit eps_xy]: eps_xy_sensitivity over the block
_XY_COMPONENTS = ('diagonal', 'off_diagonal')


def _crystal_permittivity():
    """(eps_xx, eps_yy, eps_xy) of the rotated crystal, rounded to float32 (the MaterialGrid values)."""
    angle = math.radians(_CRYSTAL_AXIS_ANGLE)
    ordinary, extraordinary = (index * index for index in _CRYSTAL_INDICES)
    contrast = extraordinary - ordinary
    values = (ordinary + contrast * math.cos(angle) ** 2, ordinary + contrast * math.sin(angle) ** 2,
              contrast * math.sin(angle) * math.cos(angle))
    return tuple(float(np.float32(value)) for value in values)


def _xy_adjoint_spec(diagonal_step, off_diagonal_step, reflection_weight, transmission_weight):
    """em_fdfd_hz facet layout with the crystal end section and the block shifted by (diagonal, off-diagonal) steps;
    derived block integrals of eps_sensitivity and eps_xy_sensitivity."""
    spec = _adjoint_spec('hz', _SILICON_EPS_R, reflection_weight, transmission_weight)
    eps_xx, eps_yy, eps_xy = _crystal_permittivity()
    y_lo, y_hi = _ADJOINT_SLAB_Y
    crystal = {'eps_xx': eps_xx, 'eps_yy': eps_yy, 'eps_xy': eps_xy}
    block = {'eps_xx': eps_xx + diagonal_step, 'eps_yy': eps_yy + diagonal_step, 'eps_xy': eps_xy + off_diagonal_step}
    materials = [material for material in spec.materials if material.name != 'block']
    materials += [MaterialSpec('crystal', properties=crystal, selection=['crystal']),
                  MaterialSpec('block', properties=block, selection=['block'])]
    geometry = [*spec.geometry, GeometrySpec('crystal', 'rectangle', {'x0': _CRYSTAL_X_RANGE[0], 'y0': y_lo,
                                                                      'x1': _CRYSTAL_X_RANGE[1], 'y1': y_hi})]
    derived = [*spec.derived_values,
               DerivedValueSpec(_XY_GRADIENT_VALUE, f'{_PHYSICS_NAME}_eps_xy_sensitivity', ['block'], 'integral')]
    return replace(spec, title='Self-test: em_fdfd_hz adjoint sensitivity with an eps_xy crystal at a slab facet',
                   geometry=geometry, materials=materials, derived_values=derived)


def _xy_adjoint_specs():
    """One gradient run per objective, then per component (diagonal, off_diagonal) the block at +- _BLOCK_STEP."""
    specs = {f'{objective}_gradient': _xy_adjoint_spec(0.0, 0.0, *weights)
             for objective, weights in ((_REFLECTION, (1.0, 0.0)), (_TRANSMISSION, (0.0, 1.0)))}
    for component in _XY_COMPONENTS:
        for sign, label in ((1.0, 'plus'), (-1.0, 'minus')):
            step = sign * _BLOCK_STEP
            steps = (step, 0.0) if component == 'diagonal' else (0.0, step)
            specs[f'{component}_{label}'] = _xy_adjoint_spec(*steps, 1.0, 1.0)
    return specs


def _check_xy_adjoint_gradient(results_by_name):
    """Per component and objective: the block integral of eps_sensitivity (diagonal) / eps_xy_sensitivity (off-diagonal)
    vs the central difference; measured = error = the worst relative deviation in tolerance units."""
    details, normalized_errors = {'unit': 'W/m per unit eps', 'block_step': _BLOCK_STEP,
                                  'crystal_eps_xx_yy_xy': list(_crystal_permittivity())}, {}
    for component, value_name in zip(_XY_COMPONENTS, (_GRADIENT_VALUE, _XY_GRADIENT_VALUE)):
        for objective in _OBJECTIVES:
            plus = _objective_value(results_by_name[f'{component}_plus'], objective)
            minus = _objective_value(results_by_name[f'{component}_minus'], objective)
            finite_difference = (plus - minus) / (2.0 * _BLOCK_STEP)
            gradient_results = results_by_name[f'{objective}_gradient']
            adjoint = float(gradient_results.summary()['derived'][value_name])   # float() also reads 'nan'
            error = abs(adjoint / finite_difference - 1.0) if finite_difference != 0.0 else math.nan
            normalized_errors[f'{component} {objective}'] = error / _TOLERANCE
            details.update({f'{component}_{objective}_adjoint': adjoint,
                            f'{component}_{objective}_finite_difference': finite_difference,
                            f'{component}_{objective}_relative_error': error})
    return normalized_result(normalized_errors, details)


ADJOINT_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_adjoint_gradient',
        label='EM FDFD adjoint sensitivity vs finite difference',
        description='The silicon slab of em_fdfd_mode_port_slab (4 x 2 um all-CPML box, but its own 40 nm mesh: '
                    '10,400 unknowns per run, the slab y 0.88..1.16 um on 40 nm faces; stationary at 1.55 um, cpu '
                    'backend; not yet confirmed by a run with this layout) ends in a facet at x = 2.52 um; a mode '
                    'source (mode 1, +x, 1 W/m) feeds it. Two adjoint_objective features: reflection = mode_power_negative of mode 1 on a full '
                    'height line 0.5 um upstream of the source, transmission = flux through the upper half (y 1..2 '
                    'um) of a line ~1 um past the facet. A 2 x 2 cell block (80 nm square) in the upper half of the '
                    'silicon, 0.12 um before the facet, off every line, is the perturbed region. Per polarisation '
                    '(em_fdfd_ez, em_fdfd_hz) two gradient runs weight one objective each (weight 1, the other 0) and '
                    'integrate eps_sensitivity over the block (derived value, W/m per unit eps_r); two runs with the '
                    'block at eps_r 12.08 +- 0.05 give the central difference of each objective. measured = error = '
                    'max |adjoint / finite difference - 1| / 2e-3 over both polarisations and objectives, expected '
                    '0, tolerance 1. Budget: the adjoint gradient is the exact derivative of the discrete objective '
                    '(transposed solve on the same factorisation, hz face means transposed), so only the central '
                    'difference error O(step^2) (~1e-4 for a 0.4% step), the float32 permittivity storage (~1e-5 of '
                    'the step) and the float32 sensitivity field remain. It pins the adjoint source of both objective '
                    'kinds, the hz face-weight chain rule (transposed face_average) and the transposed solve (SuperLU '
                    'trans T); the box is all-CPML, so the wall-row fold of the adjoint source (pec / pmc / mirror '
                    'faces) and the pin_image_cells diagonal are not exercised here.',
        specs=lambda: cpu_specs(_adjoint_specs()),
        check=_check_adjoint_gradient,
        heavy=True),
    SelfTestCase(
        name='em_fdfd_hz_eps_xy_adjoint_gradient',
        label='EM FDFD H_z adjoint sensitivity with eps_xy vs finite difference',
        description='The em_fdfd_hz layout of em_fdfd_adjoint_gradient (same 40 nm mesh, lines, objectives and block; '
                    'cpu backend; not yet confirmed by a run with this layout), but the last 0.4 um (10 cells) of the '
                    'silicon slab before the facet is a rotated uniaxial crystal (n_o 3.3, n_e 3.6, '
                    'optic axis 30 deg from x in the plane: eps_xx, eps_yy and eps_xy set), and the 2 x 2 cell block '
                    'inside it holds the same crystal; sources, ports and objective lines stay in eps_xy-free cells. '
                    'Two gradient runs (one objective weighted each) integrate eps_sensitivity and '
                    'eps_xy_sensitivity over the block; central differences move the block eps_xx and eps_yy together '
                    'by +-0.05 at fixed eps_xy (diagonal) and eps_xy by +-0.05 at fixed eps_xx / eps_yy (off '
                    'diagonal). measured = error = max |adjoint / finite difference - 1| / 2e-3 over both components '
                    'and objectives, expected 0, tolerance 1. It pins the eps_xy chain rule of the adjoint (face '
                    'weights of eps_ii - eps_xy^2 / eps_jj, the eta_xy face couplings contracted per cell) and the '
                    'transposed solve of the 9-point operator; the objective lines see no eps_xy, so the coupling '
                    'part of the adjoint source (a face-field objective in eps_xy cells) is not exercised.',
        specs=lambda: cpu_specs(_xy_adjoint_specs()),
        check=_check_xy_adjoint_gradient,
        heavy=True),
)

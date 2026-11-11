"""Self-tests of the effective-index 2.5D bridge (B4): the stack solve against the analytic three-layer slab (CPU) and
the 'effective_index_rib_waveguide' template (em_fdfd_hz) against the two-step effective-index method; physics modules
are imported inside the checks only."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import analytic_slab_index, float_list, single_value, vertical_line, worst_item
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests import normalized_result

_PHYSICS_NAME = 'em'
_WAVELENGTH = 1.55e-6              # m
_SILICA_INDEX = 1.444
_SILICON_INDEX = 3.476

# em_effective_index_slab (CPU): (label, indices bottom to top, film thicknesses [m], slab polarisation, mode number).
# The split SOI entry is the SOI film cut into two equal-index sublayers (must equal the plain SOI value).
_SLAB_CASES = (
    ('soi_te0', (_SILICA_INDEX, _SILICON_INDEX, 1.0), (220e-9,), 'te', 1),
    ('soi_tm0', (_SILICA_INDEX, _SILICON_INDEX, 1.0), (220e-9,), 'tm', 1),
    ('nitride_te1', (_SILICA_INDEX, 2.0, _SILICA_INDEX), (1e-6,), 'te', 2),
    ('nitride_tm1', (_SILICA_INDEX, 2.0, _SILICA_INDEX), (1e-6,), 'tm', 2),
    ('soi_split_te0', (_SILICA_INDEX, _SILICON_INDEX, _SILICON_INDEX, 1.0), (110e-9, 110e-9), 'te', 1),
)
_FINE_CELL = 1e-9                  # m: second-order error ~(kappa dz)^2 / 12 ~ 1e-5 of n_eff
_FINE_TOLERANCE = 1e-4             # absolute n_eff
_AUTO_TOLERANCE = 1e-3             # absolute n_eff with the automatic cell (~wavelength / (100 n_max))

# effective_index_rib_waveguide template (em_fdfd_hz, stationary at 1.55 um): 4 x 2.1 um all-CPML box, uniform 20 nm
# mesh. Everywhere the 90 nm silicon slab stack (feature 'slab_stack' on 'all'), on the rib y in [0.8, 1.3) um (rows
# 40..64, edges on cell faces) the 220 nm stack ('rib_stack', later, so it wins there); both TE0 (E along the layers,
# in the drawing plane). Vertical lines at cell centres span the window height (cpml ends): source x = 0.51 um,
# 'through' monitor x = 3.51 um.
RIB_TEMPLATE_LABEL = 'Rib waveguide (effective index 2.5D)'
RIB_TEMPLATE_DESCRIPTION = ('EM (FDFD, H_z): top view of a 0.5 um silicon rib (220 nm rib on a 90 nm slab, silica '
                            'above and below) collapsed by the effective-index method (effective_index features); a '
                            '1.55 um mode source and a mode monitor give the lateral mode index and transmission.')
_RIB_SIZE = (4e-6, 2.1e-6)         # m
_RIB_Y_RANGE = (0.8e-6, 1.3e-6)    # m
_RIB_CELL = 20e-9                  # m
_RIB_FILM = 220e-9                 # m
_SLAB_FILM = 90e-9                 # m
_RIB_LINES = (('source', 0.51e-6), ('through', 3.51e-6))   # (feature, x [m])
_EPS_TOLERANCE = 1e-6              # relative, float32 permittivity arrays
_LATERAL_INDEX_TOLERANCE = 0.01    # relative n_eff of the lateral mode (20 nm cells, as em_mode_port_slab)
_FORWARD_TOLERANCE = 0.02          # absolute power fraction


def _stack_values(film):
    """effective_index feature params of the silica / silicon (film [m]) / silica stack, TE0 at _WAVELENGTH."""
    return {'layer_indices': [_SILICA_INDEX, _SILICON_INDEX, _SILICA_INDEX], 'layer_thicknesses': [film],
            'wavelength': _WAVELENGTH, 'slab_polarization': 'te', 'mode_number': 1}


def rib_waveguide_spec():
    """Template 'effective_index_rib_waveguide': slab stack on 'all', rib stack on the rib, mode source and monitor."""
    size_x, size_y = _RIB_SIZE
    y_lo, y_hi = _RIB_Y_RANGE
    geometry = [GeometrySpec('rib', 'rectangle', {'x0': 0.0, 'y0': y_lo, 'x1': size_x, 'y1': y_hi})]
    features = [FeatureSpec('slab_stack', 'effective_index', ['all'], _stack_values(_SLAB_FILM)),
                FeatureSpec('rib_stack', 'effective_index', ['rib'], _stack_values(_RIB_FILM))]
    for feature_name, x in _RIB_LINES:
        line_name = f'{feature_name}_line'
        geometry.append(vertical_line(line_name, x, 0.0, size_y))
        features.append(FeatureSpec(feature_name, 'mode_source', [line_name],
                                    {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0})
                        if feature_name == 'source' else
                        FeatureSpec(feature_name, 'mode_monitor', [line_name], {'mode_number': 1}))
    return ModelSpec(
        title='Silicon rib waveguide, effective-index top view',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('oxide', properties={'eps_r': _SILICA_INDEX ** 2}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdfd_hz', settings={'wavelength': _WAVELENGTH}, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_RIB_CELL),
        study=StudySpec(type='stationary'))


# ---- analytic three-layer slab ----
def _three_layer_index(indices, thickness, slab_polarization, mode_number):
    """n_eff of guided mode mode_number of substrate / film / cover: bisection on kappa d - atan(p_s / kappa) -
    atan(p_c / kappa) - (m - 1) pi = 0, p = gamma (TE) or (n_f / n_i)^2 gamma (TM)."""
    substrate, film, cover = indices
    k0 = 2.0 * math.pi / _WAVELENGTH

    def mismatch(n_eff):
        kappa = k0 * math.sqrt(film ** 2 - n_eff ** 2)
        phases = 0.0
        for cladding in (substrate, cover):
            gamma = k0 * math.sqrt(max(n_eff ** 2 - cladding ** 2, 0.0))
            ratio = 1.0 if slab_polarization == 'te' else (film / cladding) ** 2
            phases += math.atan(ratio * gamma / kappa)
        return kappa * thickness - phases - (mode_number - 1) * math.pi

    lo, hi = max(substrate, cover), film * (1.0 - 1e-15)   # mismatch > 0 at lo (mode guided), < 0 at hi
    for _ in range(200):
        middle = 0.5 * (lo + hi)
        if mismatch(middle) > 0.0:
            lo = middle
        else:
            hi = middle
    return 0.5 * (lo + hi)


def _check_slab(_results_by_name):
    """Every _SLAB_CASES entry, fine and automatic cell, against the analytic index (the split SOI against the plain
    SOI analytic); error in tolerance units."""
    from .effective_index import effective_index   # physics import inside the check (selftest import rule)
    details, normalized_errors = {}, {}
    for label, indices, thicknesses, slab_polarization, mode_number in _SLAB_CASES:
        if len(indices) == 3:
            expected = _three_layer_index(indices, thicknesses[0], slab_polarization, mode_number)
        else:   # equal-index sublayers: the plain three-layer stack
            expected = _three_layer_index((indices[0], indices[1], indices[-1]), sum(thicknesses), slab_polarization,
                                          mode_number)
        fine = effective_index(indices, thicknesses, _WAVELENGTH, slab_polarization, mode_number, cell_size=_FINE_CELL)
        auto = effective_index(indices, thicknesses, _WAVELENGTH, slab_polarization, mode_number)
        normalized_errors[f'{label} fine'] = abs(fine - expected) / _FINE_TOLERANCE
        normalized_errors[f'{label} auto'] = abs(auto - expected) / _AUTO_TOLERANCE
        details[label] = {'expected': expected, 'fine_cell': fine, 'auto_cell': auto}
    return normalized_result(normalized_errors, details)


def _check_rib(results_by_name):
    """eps_xx map (rib / slab n_eff^2), lateral mode n_eff against the symmetric-slab index of the two effective indices
    ('hz': E in the drawing plane) and forward transmission 1; error in tolerance units."""
    from .effective_index import effective_index
    results = results_by_name['run']
    n_rib = effective_index(*_stack_args(_RIB_FILM))
    n_slab = effective_index(*_stack_args(_SLAB_FILM))
    field_data = results.field(f'{_PHYSICS_NAME}.eps_xx')
    eps_map = np.asarray(field_data.values, dtype=np.float64)
    y_centers = np.asarray(field_data.y_centers, dtype=np.float64)
    in_rib = (y_centers > _RIB_Y_RANGE[0]) & (y_centers < _RIB_Y_RANGE[1])
    expected_map = np.where(in_rib, n_rib ** 2, n_slab ** 2)[None, :]
    eps_error = float(np.max(np.abs(eps_map / expected_map - 1.0)))   # NaN propagates

    monitor = results.result(_PHYSICS_NAME, 'through')
    n_lateral = single_value(monitor['n_eff'], 'through n_eff')
    forward = single_value(monitor['series']['fraction_positive'], 'through fraction_positive')
    k0 = 2.0 * math.pi / _WAVELENGTH
    n_expected = analytic_slab_index(k0, _RIB_Y_RANGE[1] - _RIB_Y_RANGE[0], n_rib ** 2, n_slab ** 2, 'hz')
    normalized_errors = {'eps map': eps_error / _EPS_TOLERANCE,
                         'lateral n_eff': abs(n_lateral / n_expected - 1.0) / _LATERAL_INDEX_TOLERANCE,
                         'forward fraction': abs(forward - 1.0) / _FORWARD_TOLERANCE}
    worst, worst_error = worst_item(normalized_errors)
    return {'measured': n_lateral, 'expected': n_expected, 'error': worst_error, 'tolerance': 1.0,
            'details': {'n_eff_rib_stack': n_rib, 'n_eff_slab_stack': n_slab, 'eps_relative_error': eps_error,
                        'fraction_positive': forward, 'normalized_errors': normalized_errors,
                        'worst_item': worst, 'eps_rows': float_list(eps_map[eps_map.shape[0] // 2])}}


def _stack_args(film):
    """Positional effective_index arguments of the template's stack with a film [m]."""
    values = _stack_values(film)
    return (values['layer_indices'], values['layer_thicknesses'], values['wavelength'], values['slab_polarization'],
            values['mode_number'])


EFFECTIVE_INDEX_SELFTEST_CASES = (
    SelfTestCase(
        name='em_effective_index_slab', label='EM: effective index of layer stacks (CPU)',
        description=('Effective-index stack solve (physics/em/effective_index, A3 slab modes along the stack normal) '
                     f'at {_WAVELENGTH * 1e6:g} um against the exact three-layer slab dispersion kappa d = atan(p_s / '
                     'kappa) + atan(p_c / kappa) + (m - 1) pi (p = gamma for TE, (n_film / n_i)^2 gamma for TM): '
                     '220 nm silicon on silica under air (TE0, TM0), a 1 um index-2.0 film in silica (TE1, TM1: mode '
                     'number 2) and the SOI film split into two equal sublayers (TE0). Each with 1 nm cells '
                     f'(tolerance {_FINE_TOLERANCE:g} on n_eff; second order, interfaces on faces, TM face weights '
                     '1 / mean eps) and the automatic cell (tolerance '
                     f'{_AUTO_TOLERANCE:g}); error = the worst deviation in tolerance units.'),
        specs=dict, check=_check_slab),
    SelfTestCase(
        name='em_effective_index_rib', label='EM: effective-index rib waveguide (FDFD, H_z)',
        description=('The effective_index_rib_waveguide template: effective_index features write the TE0 n_eff^2 of '
                     'the 90 nm slab stack on all cells and of the 220 nm rib stack on the 0.5 um rib (silica / '
                     'silicon / silica, 1.55 um). Checks: eps_xx equals those values cell by cell (relative '
                     f'{_EPS_TOLERANCE:g}, float32), the through mode monitor n_eff equals the symmetric-slab '
                     "index of the two effective indices with the in-plane E ('hz') of the second effective-index "
                     f'step (relative {_LATERAL_INDEX_TOLERANCE:g}, 20 nm cells) and fraction_positive 1 (absolute '
                     f'{_FORWARD_TOLERANCE:g}); error = the worst deviation in tolerance units.'),
        specs=lambda: {'run': rib_waveguide_spec()}, check=_check_rib),
)

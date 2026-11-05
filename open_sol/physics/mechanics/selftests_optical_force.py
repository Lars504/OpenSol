"""Self-test of em_solid_optical_force: the total radiation force on a lossless dielectric slab against the momentum
balance 2 (P_inc - P_trans) / c of the flux monitors; run only inside selftests_backend.solid_optics_cpu_gpu (the
standalone check is the example mechanics/radiation_force_slab)."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, cell_centre, single_value, vertical_line, with_exact_cg
from ...model.spec import (CouplingSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)

OPTICAL_FORCE_COUPLING = 'em_solid_optical_force'    # = selftests.OPTICAL_FORCE_COUPLING
_SOLID_TYPE_NAME = 'solid_mechanics'
_EM = 'em'
_SOLID = 'solid'

# y-periodic 4.0 x 0.02 um strip, uniform 10 nm cells (400 x 2 window cells, CPML along x; the y-periodic plane wave is
# y-uniform, so the problem is 1D and 2 y cells keep the fixed_domain / support bookkeeping), vacuum except a glass slab
# (eps_r 2.25) at x = 1.5 .. 1.76 um (26 cells, ~a quarter wave at 1.55 um: R ~0.15). A line source at cell 50 sends a
# plane wave (amplitude 1e7 V/m) along +x; a flux monitor at cell 300 reads P_trans ('slab' run) or P_inc ('empty' run,
# no slab, no solid). Momentum balance of the lossless slab: F_x = (P_inc + P_refl - P_trans) / c = 2 (P_inc - P_trans)
# / c per m of depth. The slab is the only solid (E 1e10 Pa, nu 0.2) and is held by a fixed_domain on itself, so the
# support reaction is minus the total coupled body force. The face form -(eps0 / 4) |E_z|^2 (eps jump) differs from the
# discretely conserved Re(E_i conj(E_i+1)) form by |E_i+1 - E_i|^2 / 2 per face, O((n k0 dx)^2 / 2) ~2e-3 of the
# interface terms, ~5e-3 of the net force.
_WAVELENGTH = 1.55e-6
_CELL = 10e-9
_SIZE = (4e-6, 2 * _CELL)                            # m: fluxes and forces scale alike with size_y
_SLAB_X = (1.5e-6, 1.76e-6)
_SOURCE_CELL = 50
_MONITOR_CELL = 300
_SLAB_EPS_R = 2.25
_AMPLITUDE = 1e7                                     # V/m
_ELASTIC = {'youngs_modulus': 1e10, 'poisson_ratio': 0.2}
_TOLERANCE = 2e-2


def _strip_spec(with_slab):
    """em_fdfd_ez strip (line source, flux monitor); with_slab adds the glass slab, the solid and the coupling."""
    geometry = [vertical_line(name, cell_centre(cell, _CELL), 0.0, _SIZE[1])
                for name, cell in (('source_line', _SOURCE_CELL), ('monitor_line', _MONITOR_CELL))]
    materials, physics, couplings = [], [], []
    physics.append(PhysicsSpec(_EM, 'em_fdfd_ez', settings={**PERIODIC_Y, 'wavelength': _WAVELENGTH}, features=[
        FeatureSpec('source', 'line_source', ['source_line'], {'amplitude': _AMPLITUDE}),
        FeatureSpec('transmitted', 'flux_monitor', ['monitor_line'])]))
    if with_slab:
        geometry.append(GeometrySpec('slab', 'rectangle', {'x0': _SLAB_X[0], 'y0': 0.0, 'x1': _SLAB_X[1],
                                                           'y1': _SIZE[1]}))
        materials.append(MaterialSpec('glass', properties={'eps_r': _SLAB_EPS_R, **_ELASTIC}, selection=['slab']))
        physics.append(PhysicsSpec(_SOLID, _SOLID_TYPE_NAME,
                                   features=[FeatureSpec('anchor', 'fixed_domain', ['slab'])]))
        couplings.append(CouplingSpec('optical_force', OPTICAL_FORCE_COUPLING, _EM, _SOLID))
    return with_exact_cg(ModelSpec(
        title=f"Self-test: optical force on a dielectric slab ({'slab' if with_slab else 'empty'})",
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=geometry, materials=materials, physics=physics, couplings=couplings,
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary')))


def _specs():
    return {'empty': _strip_spec(False), 'slab': _strip_spec(True)}


def _check(results_by_name):
    """-reaction of the anchor against 2 (P_inc - P_trans) / c; |F_y| relative to it."""
    incident = single_value(results_by_name['empty'].result(_EM, 'transmitted')['power'], 'incident flux')
    transmitted = single_value(results_by_name['slab'].result(_EM, 'transmitted')['power'], 'transmitted flux')
    expected = 2.0 * (incident - transmitted) / C0
    table = results_by_name['slab'].result(_SOLID, 'anchor')
    force_x, force_y = -float(table['F_x']), -float(table['F_y'])
    errors = {'force_x': abs(force_x - expected) / abs(expected), 'force_y': abs(force_y) / abs(expected)}
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': force_x, 'expected': expected, 'error': error if math.isfinite(error) else math.inf,
            'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'unit_force': 'N/m', 'P_incident_W_per_m': incident,
                        'P_transmitted_W_per_m': transmitted, 'reflectance': 1.0 - transmitted / incident,
                        'F_y_N_per_m': force_y}}


OPTICAL_FORCE_CASES = (
    SelfTestCase(
        name='em_solid_optical_force_slab', label='EM -> solid: radiation force on a dielectric slab',
        description=(f'em_fdfd_ez + solid_mechanics, stationary, on a y-periodic {_SIZE[0] * 1e6:g} x {_SIZE[1] * 1e6:g} '
                     f'um strip of {_CELL * 1e9:g} nm cells (CPML along x): a line source (amplitude {_AMPLITUDE:g} '
                     f'V/m) at window cell {_SOURCE_CELL} sends a plane wave onto a lossless slab (eps_r '
                     f'{_SLAB_EPS_R:g}, x = {_SLAB_X[0] * 1e6:g} .. {_SLAB_X[1] * 1e6:g} um, ~a quarter wave at '
                     f'{_WAVELENGTH * 1e6:g} um), the only solid (E {_ELASTIC["youngs_modulus"]:g} Pa, held by a '
                     f'fixed_domain on itself); {OPTICAL_FORCE_COUPLING} hands the force density to the solid. A flux '
                     f'monitor at cell {_MONITOR_CELL} reads P_trans, and P_inc in an empty run (no slab). Exact '
                     '(momentum balance of a lossless slab in vacuum): F_x = 2 (P_inc - P_trans) / c per m of depth, '
                     'F_y = 0; measured = minus the anchor reaction (= the total coupled body force). error = max(|F_x '
                     f'- F| / F, |F_y| / F); tolerance {_TOLERANCE:g}: the face form of -(eps0 / 4) |E_z|^2 grad eps '
                     'is not the discretely conserved one (O((n k0 dx)^2 / 2) per interface, ~5e-3 here). Pins the '
                     'sign, the face split into the solid cells, the 1 / cell area and the coupling into '
                     'body_force_x / _y. Not yet run on the 2-cell strip (was 10 y cells).'),
        specs=_specs, check=_check),
)

"""Mirror-plane self-test of the held fluxes of electric_currents (symmetry 2.5): half vs full domain under an EM
mirror face of the model; specs built fresh, no physics imports."""
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import with_exact_cg, worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

_CURRENTS = 'currents'
_EM = 'em'
# currents_mirror_held_flux: a conductor sheet (sigma_dc 10 S/m, uniform 20 um cells, insulated x faces) 0.8 mm wide
# along x and 1 mm long along y, voltage 1 V on y_lo ('anode') and 0 V on y_hi ('cathode'), symmetric about x = 0.4 mm
# (a cell edge): a terminal 'centre' (0.8 V) on cells x 15..24, y 22..27 across the plane, a terminal 'off' (0.3 V) on
# x 32..35, y 10..14 and its image 'off_image' on x 4..7 (every terminal edge on a cell edge: a float32 centre on a
# rectangle edge would rasterise the off terminal and its image differently). The half (0.4 mm, cells 0..19 = full cells
# 20..39) holds an em_fdfd_ez physics whose bc_x_lo symmetric puts a mirror plane on x = 0 (its ghost is padded outside
# the window, so the currents window is the full model's right half and its insulated x_lo face the even mirror): the
# half solve is the exact restriction, and every reported current is that of the unfolded structure (x 2 per touched
# mirror plane): anode / cathode faces meet the plane, 'centre' reaches the window edge layer there, 'off' does not
# (x 1). Both runs on the cpu backend (the em_fdfd_ez point source then solves by scipy).
_CELL = 2e-5                       # m
_HALF_WIDTH = 0.4e-3               # m
_LENGTH = 1e-3                     # m, along y
_SIGMA = 10.0                      # S/m
_TERMINALS = {'centre': (0.1e-3, 0.44e-3, 0.56e-3, 0.8),   # half-width about the plane, y0, y1, voltage
              'off': ((0.24e-3, 0.32e-3), 0.2e-3, 0.3e-3, 0.3)}   # (x0, x1) from the plane, y0, y1, voltage
_EM_WAVELENGTH = 4e-4              # m: 20 cells per wavelength on the 20 um mesh
_TOLERANCE = 1e-6                  # relative; float64 CG to 1e-12
_COMPARED = (('anode', 'anode'), ('cathode', 'cathode'), ('centre', 'centre'), ('off', 'off'),
             ('off', 'off_image'))   # (half feature, full feature): the off terminal also equals its image


def _rectangle(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _mirror_spec(half):
    """The full sheet (currents only) or its x_lo-mirrored half (currents + an em_fdfd_ez physics with the mirror)."""
    plane = 0.0 if half else _HALF_WIDTH
    centre_half, centre_y0, centre_y1, centre_voltage = _TERMINALS['centre']
    (off_x0, off_x1), off_y0, off_y1, off_voltage = _TERMINALS['off']
    geometry = [_rectangle('sheet', 0.0, 0.0, plane + _HALF_WIDTH, _LENGTH),
                _rectangle('centre', 0.0 if half else plane - centre_half, centre_y0, plane + centre_half, centre_y1),
                _rectangle('off', plane + off_x0, off_y0, plane + off_x1, off_y1)]
    features = [FeatureSpec('anode', 'voltage', ['y_lo'], {'voltage': 1.0}),
                FeatureSpec('cathode', 'voltage', ['y_hi'], {'voltage': 0.0}),
                FeatureSpec('centre', 'terminal', ['centre'], {'voltage': centre_voltage}),
                FeatureSpec('off', 'terminal', ['off'], {'voltage': off_voltage})]
    if not half:
        geometry.append(_rectangle('off_image', plane - off_x1, off_y0, plane - off_x0, off_y1))
        features.append(FeatureSpec('off_image', 'terminal', ['off_image'], {'voltage': off_voltage}))
    physics = [PhysicsSpec(_CURRENTS, 'electric_currents', features=features)]
    if half:
        geometry.append(GeometrySpec('em_source', 'point', {'x': 0.2e-3, 'y': 0.5e-3}))
        physics.append(PhysicsSpec(_EM, 'em_fdfd_ez', settings={'bc_x_lo': 'symmetric', 'wavelength': _EM_WAVELENGTH},
                                   features=[FeatureSpec('source', 'point_source', ['em_source'],
                                                         {'current': 1e-3, 'phase': 0.0})]))
    return with_exact_cg(ModelSpec(
        title=f"Self-test: currents held fluxes under a mirror ({'half, em x_lo symmetric' if half else 'full'})",
        domain=DomainSpec(size_x=plane + _HALF_WIDTH, size_y=_LENGTH),
        geometry=geometry,
        materials=[MaterialSpec('conductor', properties={'sigma_dc': _SIGMA}, selection=['sheet'])],
        physics=physics,
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary')))


def _check_mirror(results_by_name):
    """Every held current of the half (table results: PotentialPhysics.held_flux, also the scalar outputs') vs its full
    counterpart; error = the largest relative deviation."""
    full, half = results_by_name['full'], results_by_name['half']
    errors, details = {}, {}
    for half_feature, full_feature in _COMPARED:
        measured = float(half.result(_CURRENTS, half_feature)['current'])
        expected = float(full.result(_CURRENTS, full_feature)['current'])
        errors[f'{half_feature}_vs_{full_feature}'] = abs(measured - expected) / abs(expected)
        details[f'{half_feature}_vs_{full_feature}'] = [expected, measured]
    return worst_result(errors, _TOLERANCE, **details)


MIRROR_SELFTEST_CASES = (
    SelfTestCase(
        name='currents_mirror_held_flux', label='Electric currents: held currents under a mirror plane',
        description=('Symmetry 2.5: a sigma_dc 10 S/m sheet 0.8 x 1 mm (uniform 20 um cells, every terminal edge on '
                     'a cell edge, cpu backend) between voltage faces '
                     'y_lo 1 V / y_hi 0 V, symmetric about x = 0.4 mm, with a terminal across the plane (0.8 V) and an '
                     'off-plane terminal plus its image (0.3 V), against its half with an em_fdfd_ez physics whose '
                     'bc_x_lo symmetric makes x = 0 a mirror plane of the model (the currents x_lo face there is the '
                     'insulated even mirror, so the half solve is the exact restriction). Every reported current is '
                     'the unfolded structure\'s (PotentialPhysics.mirror_factor): the voltage faces meet the plane and '
                     'the centre terminal reaches it (x 2), the off terminal does not (x 1, equal to the full off and '
                     'off_image currents). error = the largest relative deviation of the half currents from the full '
                     f'ones, tolerance {_TOLERANCE:g} (float64 CG to 1e-12). Not yet run at 20 um cells on the cpu '
                     'backend.'),
        specs=lambda: {'full': _mirror_spec(False), 'half': _mirror_spec(True)}, check=_check_mirror),
)

"""Model-run self-test of solid_mechanics_3d on swept prism layers (experiment plan D5): the first mode of a clamped
square plate meshed with 3 P2 layers across its thickness against the Kirchhoff clamped-plate value. definition3d
registers it."""
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result
from ...model.spec import FeatureSpec, MaterialSpec, MeshAxisSpec, MeshSpec, StudySpec
from .selftests import _elastic
from .selftests_fem3d import _solid_spec

UM = 1e-6
_SOLID = 'solid'
_TABLE = 'modes'
# a 60 x 60 um plate 1.2 um thick (h / a = 0.02: shear and rotary inertia lower f1 by well under 1 %), clamped on its
# four side faces; the footprint in triangles of <= a / 16, three equal z layers (custom z edges), P2
_SIDE = 60.0 * UM
_THICKNESS = 1.2 * UM
_IN_PLANE = _SIDE / 16.0
_MATERIAL = (170e9, 0.28, 2330.0)            # E [Pa], nu, rho [kg/m^3]
_CLAMPED_LAMBDA_SQUARED = 35.985             # Kirchhoff CCCC square plate, first mode (Leissa), independent of nu
_FREQUENCY_TOLERANCE = 0.03                  # 3D P2 swept vs Kirchhoff


def _plate_spec():
    youngs_modulus, poisson_ratio, density = _MATERIAL
    spec = _solid_spec('Self-test: clamped plate on swept prism layers', (_SIDE, _SIDE, _THICKNESS),
                       [MaterialSpec('plate', properties=_elastic(youngs_modulus, poisson_ratio, rho=density),
                                     selection=['all'])],
                       [FeatureSpec('clamp', 'fixed', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                        FeatureSpec(_TABLE, 'resonance_table')],
                       study=StudySpec(type='eigenfrequency', settings={'n_resonances': 1}))
    spec.meshes = [MeshSpec(name='tetrahedra', kind='tetrahedra', mode='nonuniform', dx=_IN_PLANE, mesher='builtin',
                            sweep_axis='z',
                            axes=[MeshAxisSpec('z', edges=[_THICKNESS / 3.0, 2.0 * _THICKNESS / 3.0])])]
    return spec


def _kirchhoff_frequency() -> float:
    """f1 = lambda^2 / (2 pi a^2) sqrt(D / (rho h)), D = E h^3 / (12 (1 - nu^2)), of the stored (float32) material."""
    youngs_modulus, poisson_ratio, density = (float32_value(value) for value in _MATERIAL)
    rigidity = youngs_modulus * _THICKNESS ** 3 / (12.0 * (1.0 - poisson_ratio ** 2))
    return _CLAMPED_LAMBDA_SQUARED / (2.0 * math.pi * _SIDE ** 2) * math.sqrt(rigidity / (density * _THICKNESS))


def _check_plate(results_by_name):
    table = results_by_name['plate'].result(_SOLID, _TABLE)
    rows = table['rows']
    if not rows:
        return scaled_worst_result({'frequency': math.inf}, {'frequency': _FREQUENCY_TOLERANCE}, table=table)
    found = float(dict(zip(table['columns'], rows[0]))['frequency'])
    expected = _kirchhoff_frequency()
    return scaled_worst_result({'frequency': abs(found / expected - 1.0)}, {'frequency': _FREQUENCY_TOLERANCE},
                               frequency_Hz=found, kirchhoff_Hz=expected)


SWEPT_PLATE_CASES = (
    SelfTestCase(
        name='solid_fem3d_swept_plate_modes',
        label='Solid mechanics 3D on swept prism layers: clamped square plate',
        description=('eigenfrequency study of solid_mechanics_3d (cpu, P2, host SuperLU) on a 60 x 60 x 1.2 um '
                     'plate (E 170 GPa, nu 0.28, rho 2330) clamped on its four side faces, meshed with sweep_axis '
                     "'z': builtin triangles <= 3.75 um in-plane, three 0.4 um prism layers (custom z edges), each "
                     'prism split into 3 tetrahedra. The first frequency against the Kirchhoff clamped '
                     f'square plate lambda^2 = {_CLAMPED_LAMBDA_SQUARED} (h / a = 0.02) within '
                     f'{_FREQUENCY_TOLERANCE:g}. measured = error = the deviation in tolerance units, tolerance 1.'),
        specs=lambda: {'plate': _plate_spec()}, check=_check_plate, heavy=True),
)

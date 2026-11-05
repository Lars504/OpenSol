"""Self-test of the symmetry check (symmetry 2.7): validate warns exactly for geometry drawn asymmetrically beyond a
mirror plane or the (r, z) axis (2D lo / hi faces, axisymmetric heat, a 3D z mirror). CPU only (specs() is empty)."""
from .schema import SelfTestCase
from .spec import DomainSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec

_UM = 1e-6
_SIZE = (4e-6, 2e-6)       # m, 2D window
_SIZE_3D = (2e-6, 2e-6, 1e-6)
_CELL = 50e-9              # m, uniform mesh (validation only)
_CPML = {f'bc_{face}': 'cpml' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}


def _rectangle(name, x0, x1, y0=0.8 * _UM, y1=1.2 * _UM):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _spec_2d(geometry, physics):
    """2D model: the objects (material 'core' on all of them, background 'air') and one physics."""
    spec = ModelSpec(title='Self-test: symmetry check', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]))
    spec.mesh = MeshSpec(mode='uniform', dx=_CELL)
    spec.geometry = list(geometry)
    spec.materials = [MaterialSpec('air', properties={'eps_r': 1.0, 'k': 0.03}, selection=['all']),
                      MaterialSpec('core', properties={'eps_r': 12.0, 'k': 150.0},
                                   selection=[obj.name for obj in geometry])]
    spec.physics = [physics]
    return spec


def _em(**faces):
    return PhysicsSpec('em', 'em_fdfd_ez', settings={**_CPML, **faces})


def _cases():
    """(case name, spec, expected {issue path: sorted object names} of the symmetry warnings)."""
    um = _UM
    lo = {'bc_x_lo': 'symmetric'}
    cases = [
        ('half_only', _spec_2d([_rectangle('core', 0.0, 0.5 * um)], _em(**lo)), {}),
        ('symmetric_crossing', _spec_2d([_rectangle('core', -0.5 * um, 0.5 * um),
                                         GeometrySpec('disc', 'circle', {'cx': 0.0, 'cy': 0.4 * um, 'r': 0.2 * um})],
                                        _em(**lo)), {}),
        ('offset_crossing', _spec_2d([_rectangle('core', -0.2 * um, 0.5 * um)], _em(**lo)),
         {('physics', 'em'): ['core']}),
        ('edge_extension', _spec_2d([_rectangle('substrate', -0.1 * um, _SIZE[0], 0.0, 0.5 * um),
                                     _rectangle('core', -0.05 * um, 0.4 * um)], _em(**lo)), {}),
        ('beyond_only', _spec_2d([_rectangle('core', 0.0, 0.5 * um), _rectangle('left', -1.0 * um, -0.6 * um)],
                                 _em(**lo)), {('physics', 'em'): ['left']}),
        ('mirrored_pair', _spec_2d([_rectangle('right', 0.6 * um, 1.0 * um), _rectangle('left', -1.0 * um, -0.6 * um)],
                                   _em(**lo)), {}),
        ('hi_face_offset', _spec_2d([_rectangle('core', _SIZE[0] - 0.5 * um, _SIZE[0] + 0.3 * um)],
                                    _em(bc_x_hi='antisymmetric')), {('physics', 'em'): ['core']}),
        ('axisymmetric_offset', _spec_2d([GeometrySpec('ring', 'circle', {'cx': 0.2 * um, 'cy': um, 'r': 0.4 * um})],
                                         PhysicsSpec('heat', 'heat', settings={'geometry': 'axisymmetric'})),
         {('physics', 'heat'): ['ring']}),
        ('axisymmetric_centred', _spec_2d([GeometrySpec('ball', 'circle', {'cx': 0.0, 'cy': um, 'r': 0.4 * um})],
                                          PhysicsSpec('heat', 'heat', settings={'geometry': 'axisymmetric'})), {}),
    ]
    for name, z0, z1, expected in (('z_mirror_offset', -0.1 * um, 0.3 * um, {('physics', 'em'): ['pillar']}),
                                   ('z_mirror_symmetric', -0.3 * um, 0.3 * um, {})):
        spec = ModelSpec(title='Self-test: 3D symmetry check',
                         domain=DomainSpec(size_x=_SIZE_3D[0], size_y=_SIZE_3D[1], size_z=_SIZE_3D[2]))
        spec.mesh = MeshSpec(mode='uniform', dx=_CELL)
        spec.geometry = [GeometrySpec('pillar', 'box', {'x0': 0.6 * um, 'y0': 0.6 * um, 'z0': z0, 'x1': 1.4 * um,
                                                        'y1': 1.4 * um, 'z1': z1})]
        spec.materials = [MaterialSpec('air', properties={'eps_r': 1.0}, selection=['all']),
                          MaterialSpec('core', properties={'eps_r': 4.0}, selection=['pillar'])]
        spec.physics = [PhysicsSpec('em', 'em_fdtd_3d', settings={'bc_z_lo': 'symmetric'})]
        cases.append((name, spec, expected))
    return cases


def _check_symmetry(results_by_name):
    """Per case the symmetry warnings (path -> named objects) equal the expected ones; error = failed cases."""
    from .schema import get_registry
    from .symmetry_check import MISMATCH_MARK
    from .validate import validate

    registry = get_registry()
    details, failed = {}, []
    for name, spec, expected in _cases():
        found = {}
        for issue in validate(spec, registry):
            if MISMATCH_MARK in issue.message:
                listed = issue.message.split(MISMATCH_MARK, 1)[1].split(' differ(s)', 1)[0]
                found[issue.path] = sorted(item.strip("'") for item in listed.split(', '))
        passed = found == expected
        details[name] = {'passed': passed, 'found': {'/'.join(path): names for path, names in found.items()}}
        if not passed:
            failed.append(name)
    return {'measured': len(failed), 'expected': 0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': {'failed': failed, 'cases': details}}


SYMMETRY_CHECK_SELFTEST_CASES = (
    SelfTestCase(
        name='model_symmetry_check',
        label='Symmetry check of mirror and axisymmetric models',
        description='CPU validation runs (symmetry 2.7, model.symmetry_check): the warning that the geometry drawn '
                    'beyond a plane is not the mirror image of the solved half appears exactly for an off-centre object crossing a symmetric x_lo plane, '
                    'an object drawn only beyond it and an off-centre object across an antisymmetric x_hi plane '
                    '(em_fdfd_ez), an off-axis circle across the r = 0 axis of axisymmetric heat and an off-centre '
                    'box across a symmetric z_lo plane (em_fdtd_3d), naming the object; and never for a half drawn '
                    'inside only, objects centred on the plane, a mirrored pair of separate objects, an edge '
                    'extension of a substrate and core a little past the plane, an axis-centred circle and a '
                    'z-centred box. error = number of failed cases, tolerance 0.',
        specs=lambda: {},
        check=_check_symmetry,
        max_wall_time=60.0),
)

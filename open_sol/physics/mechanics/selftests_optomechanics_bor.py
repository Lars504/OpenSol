"""Self-tests of the optomechanics study with em_fdfd_bor optics on triangles (optomechanics_fem wiring): the silica
sphere WGM moving-boundary rate of a uniform expansion (basiri-esfahani2019 plan B-E6) and the symmetry zeros of a
microdisk's flapping and crown modes (B-E8); cpu-backend model runs."""
import math

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, StudySpec)
from .selftests_optomechanics_fem import _report

_OPTICAL = 'em'
_OPTICAL_TYPE = 'em_fdfd_bor'
_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'
_COUPLING = 'g0'
_STUDY = 'optomechanics'   # = model.study_optomechanics.STUDY_NAME
_SILICA_INDEX = 1.444      # the plan's non-dispersive silica (library eps_r 2.085 is n = 1.44395)

# ---- B-E6: silica sphere (R = 10 um) in vacuum as the quarter (r, z) disk (pec z_lo = the equatorial mirror of the
# TE_52 mode, whose tangential E is odd there), cpml r_hi / z_hi rings of 8 x the coarse size (pml_kappa_max 1), P2
# elements of 90 nm in the 8.5 .. 11.5 um band, 250 nm elsewhere. q . n = 1 on the sphere: d omega / dR = -omega / R
# exactly for the open sphere (one length scale), g_OM / 2 pi = 19.233 GHz/nm at the Mie pole 1558.7502 nm (plan
# section 6); the rate is the table extra prescribed_g_om. The mechanical row (lowest breathing-type mode, roller z_lo)
# is not checked.
_SPHERE_RADIUS = 10.0e-6
_SPHERE_WINDOW = 14.0e-6
_SPHERE_BAND = (8.5e-6, 11.5e-6)
_SPHERE_FINE = 90e-9
_SPHERE_COARSE = 250e-9
_SPHERE_ORDER = 52
_SPHERE_WAVELENGTH = 1558.7502e-9      # TE l = m = 52, radial order 1 (exact Mie pole)
_SPHERE_G_OM = -2.0 * math.pi * 19.233e18   # rad/s/m: -omega / R at the pole (19.233 GHz/nm)
_SPHERE_G_TOLERANCE = 5e-3             # relative (plan B-E6)
_SPHERE_WAVELENGTH_TOLERANCE = 0.1e-9  # m (the plan's 0.02 nm needs a finer mesh; first estimate)
_PML_SETTINGS = {'pml_cells': 8, 'pml_kappa_max': 1.0}

# ---- B-E8: silica disk (R = 4 um, t = 1 um) centred on the window centre line z = 3 um, mesh mirrored there
# (mirror_y: exactly z-symmetric), cpml r_hi / z_lo / z_hi; the solid clamped on a centred core (r < 0.5 um, |z| < 0.4
# um: fixed_domain) keeps every mode even or odd in z. The resonance nearest 1.45 um at m = 20 has a z-even |E|^2, so
# the flapping (umbrella) mode, z-odd in q . n, cancels between the faces, and the m = 2 crown mode integrates
# cos(2 phi) against the uniform travelling resonance. Reference: the largest |g_om| among the modes nearest the
# contour (radial) estimate f = zeta c_p / (2 pi R) (zeta 2.05, c_p = sqrt(E / (rho (1 - nu^2)))).
_DISK_RADIUS = 4.0e-6
_DISK_THICKNESS = 1.0e-6
_DISK_WINDOW = (7.0e-6, 6.0e-6)        # r, z extent
_DISK_HALO = (5.5e-6, 2.4e-6)          # refined box: r extent, z height (centred)
_DISK_CORE = (0.5e-6, 0.8e-6)          # clamped core: radius, height (centred)
_DISK_FINE = 100e-9
_DISK_COARSE = 250e-9
_DISK_ORDER = 20
_DISK_WAVELENGTH = 1.45e-6
_DISK_CONTOUR_FREQUENCY = 4.77e8       # Hz
_DISK_CONTOUR_MODES = 4
_DISK_CROWN_ORDER = 2
_SYMMETRY_TOLERANCE = 1e-3             # |g| / |g_contour| (plan B-E8)


def _bor_settings(order, wavelength, z_lo):
    """em_fdfd_bor settings on triangles (quadratic) with cpml r_hi / z_hi and the given z_lo wall."""
    return {'azimuthal_order': order, 'wavelength': wavelength, 'discretization': 'quadratic', 'bc_x_hi': 'cpml',
            'bc_y_lo': z_lo, 'bc_y_hi': 'cpml', **_PML_SETTINGS}


def _silica(selection):
    return MaterialSpec('silica', library='silica', properties={'eps_r': _SILICA_INDEX ** 2}, selection=selection)


# ---- B-E6 ----
def _sphere_specs():
    """{'sphere': the quarter sphere, em_fdfd_bor m = 52 + an axisymmetric solid with q . n = 1 prescribed}."""
    inner, outer = _SPHERE_BAND
    geometry = [GeometrySpec('sphere', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': _SPHERE_RADIUS}),
                GeometrySpec('band_outer', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': outer}),
                GeometrySpec('band_inner', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': inner}),
                GeometrySpec('band', 'difference', {'base': 'band_outer', 'subtract': ['band_inner']})]
    coupling = FeatureSpec(_COUPLING, 'optomechanical_coupling', params={
        'optical_physics': _OPTICAL, 'optical_mode': 1, 'photoelastic': False, 'prescribed_normal_displacement': 1.0})
    spec = ModelSpec(
        title='Optomechanics self-test: silica sphere WGM, uniform expansion (B-E6)',
        domain=DomainSpec(size_x=_SPHERE_WINDOW, size_y=_SPHERE_WINDOW), geometry=geometry,
        materials=[_silica(['sphere'])],
        physics=[PhysicsSpec(_OPTICAL, _OPTICAL_TYPE, settings=_bor_settings(_SPHERE_ORDER, _SPHERE_WAVELENGTH, 'pec')),
                 PhysicsSpec(_SOLID, _SOLID_TYPE, settings={'geometry': 'axisymmetric', 'discretization': 'quadratic'},
                             features=[FeatureSpec('mirror', 'roller', ['y_lo']), coupling])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_SPHERE_COARSE,
                         refinements=[RefinementSpec('band', _SPHERE_FINE)])],
        study=StudySpec(type=_STUDY, settings={'n_mechanical_modes': 1, 'n_optical_resonances': 2}))
    spec.solver_defaults.backend = 'cpu'
    return {'sphere': spec}


def _check_sphere(results_by_name):
    """prescribed_g_om vs -omega / R of the computed resonance and vs 19.233 GHz/nm; its wavelength and order."""
    table = results_by_name['sphere'].result(_SOLID, _COUPLING)
    measured = table['prescribed_g_om']
    omega = 2.0 * math.pi * table['optical_frequency']
    scaling = -omega / _SPHERE_RADIUS
    wavelength = table['optical_wavelength']
    errors = {'g_om vs -omega / R': (abs(measured / scaling - 1.0), _SPHERE_G_TOLERANCE),
              'g_om vs 19.233 GHz/nm': (abs(measured / _SPHERE_G_OM - 1.0), _SPHERE_G_TOLERANCE),
              'wavelength': (abs(wavelength - _SPHERE_WAVELENGTH), _SPHERE_WAVELENGTH_TOLERANCE),
              'azimuthal order': (0.0 if table['optical_azimuthal_order'] == _SPHERE_ORDER else math.inf, 1.0)}
    details = {'g_om_over_2pi_GHz_per_nm': measured / (2.0 * math.pi) * 1e-18,
               'minus_omega_over_R_over_2pi_GHz_per_nm': scaling / (2.0 * math.pi) * 1e-18,
               'wavelength_nm': wavelength * 1e9, 'q_factor': table['optical_q_factor'],
               'mechanical_rows': [float_list(row[:7]) for row in table['rows']]}
    return _report(errors, details)


# ---- B-E8 ----
def _disk_spec(title, solid_settings, n_modes):
    """The z-symmetric disk model with one axisymmetric solid of the given extra settings and n_mechanical_modes."""
    window_r, window_z = _DISK_WINDOW
    middle = 0.5 * window_z

    def box(name, radius, height):
        return GeometrySpec(name, 'rectangle', {'x0': 0.0, 'y0': middle - 0.5 * height, 'x1': radius,
                                                'y1': middle + 0.5 * height})

    geometry = [box('disk', _DISK_RADIUS, _DISK_THICKNESS), box('halo', *_DISK_HALO), box('core', *_DISK_CORE)]
    solid = PhysicsSpec(_SOLID, _SOLID_TYPE,
                        settings={'geometry': 'axisymmetric', 'discretization': 'quadratic', **solid_settings},
                        features=[FeatureSpec('clamp', 'fixed_domain', ['core']),
                                  FeatureSpec(_COUPLING, 'optomechanical_coupling',
                                              params={'optical_physics': _OPTICAL, 'optical_mode': 1})])
    spec = ModelSpec(
        title=title, domain=DomainSpec(size_x=window_r, size_y=window_z), geometry=geometry,
        materials=[_silica(['disk'])],
        physics=[PhysicsSpec(_OPTICAL, _OPTICAL_TYPE, settings=_bor_settings(_DISK_ORDER, _DISK_WAVELENGTH, 'cpml')),
                 solid],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_DISK_COARSE, mirror_y=True,
                         refinements=[RefinementSpec('halo', _DISK_FINE)])],
        study=StudySpec(type=_STUDY, settings={'n_mechanical_modes': n_modes, 'n_optical_resonances': 2}))
    spec.solver_defaults.backend = 'cpu'
    return spec


def _disk_specs():
    """{'contour': modes near the radial estimate, 'flap': the lowest (umbrella) mode, 'crown': the lowest m = 2 one}."""
    return {
        'contour': _disk_spec('Optomechanics self-test: microdisk contour modes (B-E8)',
                              {'target_frequency': _DISK_CONTOUR_FREQUENCY}, _DISK_CONTOUR_MODES),
        'flap': _disk_spec('Optomechanics self-test: microdisk flapping mode (B-E8)', {'target_frequency': 0.0}, 1),
        'crown': _disk_spec('Optomechanics self-test: microdisk m = 2 crown mode (B-E8)',
                            {'target_frequency': 0.0, 'azimuthal_order': _DISK_CROWN_ORDER}, 1)}


def _disk_rates(results):
    """(g_om column [rad/s/m], frequency column [Hz], table) of one run's coupling table."""
    table = results.result(_SOLID, _COUPLING)
    columns = table['columns']
    rows = table['rows']
    return ([row[columns.index('g_om')] for row in rows], [row[columns.index('frequency')] for row in rows], table)


def _check_disk(results_by_name):
    """|g_om| of the flapping and crown modes against the largest |g_om| near the contour estimate."""
    contour, contour_frequencies, contour_table = _disk_rates(results_by_name['contour'])
    reference = max((abs(value) for value in contour), default=0.0)
    usable = math.isfinite(reference) and reference > 0.0
    errors = {'contour coupling found': (0.0 if usable else math.inf, 1.0)}
    details = {'contour_g_om': float_list(contour), 'contour_frequencies': float_list(contour_frequencies),
               'optical_wavelength_nm': contour_table['optical_wavelength'] * 1e9,
               'optical_q_factor': contour_table['optical_q_factor']}
    for name in ('flap', 'crown'):
        rates, frequencies, table = _disk_rates(results_by_name[name])
        ratio = abs(rates[0]) / reference if usable and rates else math.inf
        errors[f'{name} / contour'] = (ratio, _SYMMETRY_TOLERANCE)
        details[name] = {'g_om': float_list(rates), 'frequencies': float_list(frequencies),
                         'optical_wavelength_nm': table['optical_wavelength'] * 1e9}
    return _report(errors, details)


OPTOMECHANICS_BOR_CASES = (
    SelfTestCase(
        name='om_bor_sphere_expansion',
        label='Optomechanics with axisymmetric optics: silica sphere WGM under uniform expansion',
        description=(
            f'Model run (cpu backend, plan B-E6): a silica sphere (n {_SILICA_INDEX:g}, R '
            f'{_SPHERE_RADIUS * 1e6:g} um) in vacuum as the quarter (r, z) disk, em_fdfd_bor on a triangles mesh '
            f'(quadratic, {_SPHERE_FINE * 1e9:g} nm in the {_SPHERE_BAND[0] * 1e6:g} .. {_SPHERE_BAND[1] * 1e6:g} um '
            f'band, {_SPHERE_COARSE * 1e9:g} nm elsewhere, pec z_lo mirror, cpml rings r_hi / z_hi with '
            f'pml_kappa_max 1), m = {_SPHERE_ORDER} near {_SPHERE_WAVELENGTH * 1e9:.4f} nm (TE, the exact Mie pole), '
            'and an axisymmetric solid_mechanics on the same mesh whose optomechanical_coupling prescribes q . n = 1 m '
            'on the sphere surface. Expected: the moving-boundary rate prescribed_g_om = d omega / dR = -omega / R (one '
            f'length scale) = -2 pi x 19.233 GHz/nm, both within {_SPHERE_G_TOLERANCE:g} relative (tests the E_par / '
            'D_n splitting on the curved interface and the 2 pi r weights), the resonance wavelength within '
            f'{_SPHERE_WAVELENGTH_TOLERANCE * 1e9:g} nm (first estimate; the plan asks 0.02 nm). Written, not run.'),
        specs=_sphere_specs, check=_check_sphere, heavy=True, max_wall_time=1800.0),
    SelfTestCase(
        name='om_bor_disk_symmetry_zeros',
        label='Optomechanics with axisymmetric optics: microdisk flapping and crown modes do not couple',
        description=(
            f'Three model runs (cpu backend, plan B-E8): a silica disk (R {_DISK_RADIUS * 1e6:g} um, t '
            f'{_DISK_THICKNESS * 1e6:g} um) centred on the window centre line of a mirror_y triangles mesh (exactly '
            f'z-symmetric), em_fdfd_bor m = {_DISK_ORDER} near {_DISK_WAVELENGTH * 1e6:g} um (cpml r_hi / z_lo / '
            'z_hi), an axisymmetric solid clamped on a centred core. contour: the modes nearest '
            f'{_DISK_CONTOUR_FREQUENCY * 1e-6:g} MHz (the radial contour estimate), reference = their largest |g_om|; '
            'flap: the lowest (umbrella) mode, odd in z, whose moving boundary cancels between the faces against the '
            f'z-even |E|^2; crown: the lowest mode of azimuthal_order {_DISK_CROWN_ORDER}, whose cos(2 phi) averages '
            f'out against the travelling resonance. Expected |g_om| / reference <= {_SYMMETRY_TOLERANCE:g} for flap '
            'and crown, and a finite non-zero reference. Written, not run.'),
        specs=_disk_specs, check=_check_disk, heavy=True, max_wall_time=1800.0),
)

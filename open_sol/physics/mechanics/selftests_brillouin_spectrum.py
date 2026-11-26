"""Self-tests of the brillouin_scattering feature (spontaneous Brillouin spectra by the FDT in the frequency_response
study): model runs of a focused beam (em_fdfd_ez / _hz, em_fdfd_bor) in water_532nm driving mixed P2 triangle solids,
against the exact homogeneous reference, the low-NA boxcar line, the local reference and identities of scan columns,
mirror planes, reduced sweeps and the cross-polarised order-2 drive. definition registers them; no physics imports at
module level (brillouin / materials inside the checks)."""
# Layout shared by every run: two meshes on one window (lower-left corner 0), the optics on a uniform tensor mesh
# 'yee' (default 15-cell CPML outside every open face), the solid on the triangle mesh 'phonons' (P2, fine in the
# drive rectangle, graded by max_ratio 1.3 to the coarse size elsewhere, fixed outer faces: no PML, P8 is not relied
# on). The focused beam is launched toward +y from a horizontal line over the whole window width (planar) or radius
# (BOR) whose row centre lies just below the drive rectangle (total-field side), focus inside the drive. Margins: every
# drive edge lies ~1 um (>= 1.3 attenuation lengths L_att = 2 v / Gamma = 0.72 um of water at 7.4 GHz) from the solid's
# walls: the phase-matched phonon leaves the drive along the beam, its wall echo returns non-phase-matched (sinc side
# lobe ~1 / (q L) ~ 0.04) after a round trip exp(-2 d / L_att) ~ 0.06, so echoes stay ~3e-3 of the readout.
# Accuracy budget of a psd against a reference sampled on the same optical cells (2D homogeneous): the drive samples the
# optical fields bilinearly at the solid quadrature points, the reference rasters the cell values, so the FEM drive at
# q = 2 n k0 is low by sinc^4(q dx / 4) per field pair, ~(q dx)^2 / 12 in psd (0.8 % at 10 nm, 1.3 % at 12.5 nm cells:
# a uniform scale, invisible to shift and width); P2 / P1 Taylor-Hood dispersion ~(q h)^4 / 720 in frequency (~1e-4 ..
# 1e-3 at h 25 .. 30 nm, ~0.3 .. 2 % of psd on the line flanks); the wall echoes above.
import math
from dataclasses import dataclass

import numpy as np

from ...model.schema import FREQUENCY_RESPONSE_STUDY, SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import (cell_centre, cpu_defaults, float_list, norm_relative_error, relative_error,
                                       scaled_worst_result)
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, StudySpec)

_OPTICS = 'optics'
_SOLID = 'solid'
_OPTICS_MESH = 'yee'
_SOLID_MESH = 'phonons'
_FEATURE = 'brillouin'
_BEAM = 'beam'
_SOLID_TYPE = 'solid_mechanics'          # = selftests.SOLID_TYPE_NAME
_FEATURE_TYPE = 'brillouin_scattering'   # = definition_brillouin.BRILLOUIN_FEATURE
_WATER_LIBRARY = 'water_532nm'
_WAVELENGTH = 532e-9                     # m, vacuum
_TEMPERATURE = 293.15                    # K
_BEAM_POWER = 1.0                        # W/m planar, W axisymmetric
_MAX_RATIO = 1.3                         # triangle size growth from the drive rectangle
_MIXED_P2 = {'discretization': 'quadratic', 'formulation': 'mixed'}   # water_532nm (shear_modulus 0) needs mixed
_DISPLACEMENT_P2 = {'discretization': 'quadratic'}
_PLANAR_WALLS = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
_BOR_WALLS = ('x_hi', 'y_lo', 'y_hi')    # x_lo is the axis
_PEAK_SHARE = 0.1                        # psd L2 checks over the samples above this share of the reference peak


def _frequencies(start, stop, count):
    return float_list(np.linspace(start, stop, count))


def _line_row(target, cell):
    """Window row whose centre lies at or just below `target` [m] (the beam line; the drive starts at or above it)."""
    return int(math.floor(target / cell - 0.5 + 1e-9))


def _water():
    return MaterialSpec('water', library=_WATER_LIBRARY, selection=['all'])


def _water_constants():
    """(n, rho, bulk modulus, longitudinal viscosity 4/3 eta + eta_b, sound speed) of library water_532nm."""
    from ...materials import WATER_532NM
    values = WATER_532NM.model_properties()
    viscous = 4.0 / 3.0 * values['viscosity'] + values['bulk_viscosity']
    return (math.sqrt(values['eps_r']), values['rho'], values['bulk_modulus'], viscous,
            math.sqrt(values['bulk_modulus'] / values['rho']))


def _brillouin(params):
    """brillouin_scattering params: the optics physics, room temperature, then `params`."""
    return {'optical_physics': _OPTICS, 'temperature': _TEMPERATURE, **params}


def _spectrum(results, physics=_SOLID):
    return results.result(physics, _FEATURE)


def _series(result, name):
    return np.asarray(result['series'][name], dtype=np.float64)


def _frequency_axis(result):
    return np.asarray(result['x'], dtype=np.float64)


def _dho_area(fit):
    """One-sideband area int_0^inf A / ((f0^2 - f^2)^2 + gamma^2 f^2) df = pi A / (2 gamma f0^2) of a dho_fit, from
    its peak A / D_min (D_min = gamma^2 f0^2 - gamma^4 / 4): (pi / 2) peak gamma (1 - gamma^2 / (4 f0^2))."""
    shift, gamma, peak = fit['shift'], fit['gamma'], fit['peak']
    return 0.5 * math.pi * peak * gamma * (1.0 - gamma * gamma / (4.0 * shift * shift))


def _bound_error(value, upper):
    """Error in tolerance units of 0 < value <= upper: value / upper, inf for value <= 0 or NaN."""
    value = float(value)
    return value / upper if value > 0.0 and math.isfinite(value) else math.inf


def _solid_counter(results, physics, name):
    """A work counter of the solid's own (the summary 'solvers' record of the physics, 'counters')."""
    record = (results.summary().get('solvers') or {}).get(physics) or {}
    return (record.get('counters') or {}).get(name)


def _mesh_elements(results, mesh):
    """Elements of a triangle mesh from the summary 'meshes' records (None when missing)."""
    for record in results.summary().get('meshes') or ():
        if record.get('name') == mesh:
            return record.get('elements')
    return None


# ---- planar layouts (em_fdfd_ez / _hz optics) ----
@dataclass(frozen=True)
class _Planar:
    """A planar model: optical cell [m], window cells (x, y), focus (x, y) [m], drive rectangle (x0, y0, x1, y1) [m]
    on optical cell faces, beam line target height [m], beam waist [m] (1/e field radius), solid element sizes in the
    drive / elsewhere [m]."""
    cell: float
    cells: tuple
    focus: tuple
    drive: tuple
    line_y: float
    waist: float
    fine: float
    coarse: float

    @property
    def size(self):
        return self.cells[0] * self.cell, self.cells[1] * self.cell


# brillouin_water_2d_ez: 3.4 x 3.2 um window of 10 nm cells (39.9 per lambda0 / n), drive 0.8 x 1.2 um around the
# focus, waist 0.4 um (Rayleigh range 1.26 um): the beam is 9e-8 of its peak at the line ends (the 1e-6 rule); solid
# 25 nm in the drive (lambda0 / (12 n) = 33 nm), 80 nm elsewhere
_WATER_2D = _Planar(cell=10e-9, cells=(340, 320), focus=(1.7e-6, 1.6e-6), drive=(1.3e-6, 1.0e-6, 2.1e-6, 2.2e-6),
                    line_y=0.97e-6, waist=0.4e-6, fine=25e-9, coarse=80e-9)
_WATER_2D_FREQUENCIES = _frequencies(5.9e9, 8.9e9, 13)
# brillouin_water_2d_hz: the same window on 12.5 nm cells (31.9 per lambda0 / n), 30 nm solid elements, 7 frequencies
_WATER_2D_HZ = _Planar(cell=12.5e-9, cells=(272, 256), focus=(1.7e-6, 1.6e-6), drive=(1.3e-6, 1.0e-6, 2.1e-6, 2.2e-6),
                       line_y=0.97e-6, waist=0.4e-6, fine=30e-9, coarse=80e-9)
_WATER_2D_HZ_FREQUENCIES = _frequencies(5.9e9, 8.9e9, 7)
_WATER_2D_TOLERANCE = {'psd': 3e-2, 'shift': 2e-3, 'fwhm': 1e-2, 'fraction': 3e-2, 'power': 1e-9}
_WATER_2D_HZ_TOLERANCE = {'psd': 3e-2, 'shift': 3e-3, 'power': 1e-9}

# brillouin_scan_columns: 3.4 x 2.0 um of 20 nm cells, a polystyrene disc (r 0.25 um) at the focus in water, the drive
# rectangle 0.8 x 0.8 um around it; the half model is x < 1.7 um with the mirror plane on x_hi (em symmetric, solid
# symmetry), the full mesh mirror_x (the lower half [0, 1.7] meshed and reflected, as the half model meshes it)
_SCAN = _Planar(cell=20e-9, cells=(170, 100), focus=(1.7e-6, 1.0e-6), drive=(1.3e-6, 0.6e-6, 2.1e-6, 1.4e-6),
                line_y=0.55e-6, waist=0.4e-6, fine=27e-9, coarse=100e-9)
_SCAN_OFFSETS_Y = (0.0, 0.1e-6, -0.15e-6)   # m: positions 1, 2, 3 (x offsets would break the mirror)
_SCAN_FREQUENCIES = (6.8e9, 7.4e9, 8.0e9)
_DISC_RADIUS = 0.25e-6
# inline polystyrene (n 1.598): moving-boundary interface at the disc, its own photoelastic tensor and loss factor
_POLYSTYRENE = {'eps_r': 1.598 ** 2, 'rho': 1050.0, 'youngs_modulus': 3.4e9, 'poisson_ratio': 0.34,
                'photoelastic_p11': 0.30, 'photoelastic_p12': 0.31, 'mechanical_loss_factor': 0.01}
_SCAN_TOLERANCE = {'columns': 1e-9,          # one factor, one optics factor: round-off of two solve paths
                   'mirror': 1e-6,           # node-identical meshes (2 x half elements = full): round-off + beam grid
                   'mirror_other_mesh': 2e-2,   # meshes not node-identical: their P2 / P1 dispersion difference
                   'counter': 0.5,           # integer counts: exact
                   'factor': 1e-12, 'power': 1e-9, 'flag': 1.0}
_SCAN_GPU_TOLERANCE = 10.0                   # in units of the cpu tolerances (cuDSS refined factors, device dots)

# brillouin_reduced_sweep: a glass-like solid (constant loss factor 0.02, no viscosity: reduced sweeps allowed) filling
# a 3.2 x 2.0 um window of 12.5 nm cells (29 per lambda0 / n), drive 0.6 x 0.6 um, f_B ~ 32.6 GHz
_GLASS_LAYOUT = _Planar(cell=12.5e-9, cells=(256, 160), focus=(1.6e-6, 1.1e-6), drive=(1.3e-6, 0.8e-6, 1.9e-6, 1.4e-6),
                        line_y=0.77e-6, waist=0.4e-6, fine=30e-9, coarse=80e-9)
_GLASS = {'eps_r': 1.45 ** 2, 'rho': 2200.0, 'youngs_modulus': 73e9, 'poisson_ratio': 0.17,
          'photoelastic_p11': 0.121, 'photoelastic_p12': 0.270, 'mechanical_loss_factor': 0.02}
_GLASS_FREQUENCIES = _frequencies(28e9, 36e9, 9)
_ROM_TOLERANCE = 1e-10
_REDUCED_TOLERANCE = 1e-6   # psd adaptive vs direct, of max psd (ROM residual 1e-10 x Q ~ 50; first estimate)


def _planar_spec(title, layout, polarization, brillouin, frequencies, *, half=False, mesh_mirror=False, focus_dy=0.0,
                 material=None, extra_materials=(), extra_geometry=(), mixed=True, study_settings=None):
    """A planar model of `layout` (module comment): em_fdfd_<polarization> with the focused beam (focus moved by
    focus_dy), the solid with fixed walls (half: x < half the window, the mirror plane on x_hi) and the
    brillouin_scattering feature on the drive rectangle."""
    size_x = (layout.cells[0] // 2 if half else layout.cells[0]) * layout.cell
    size_y = layout.size[1]
    line_y = cell_centre(_line_row(layout.line_y, layout.cell), layout.cell)
    x0, y0, x1, y1 = layout.drive
    geometry = [GeometrySpec('beam_line', 'line', {'x0': 0.0, 'y0': line_y, 'x1': size_x, 'y1': line_y}),
                GeometrySpec('drive', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}), *extra_geometry]
    beam = FeatureSpec(_BEAM, 'focused_beam', ['beam_line'], {
        'power': _BEAM_POWER, 'phase': 0.0, 'waist': layout.waist, 'numerical_aperture': None,
        'focus_x': layout.focus[0], 'focus_y': layout.focus[1] + focus_dy, 'direction': 'positive'})
    optics_settings = {'wavelength': _WAVELENGTH}
    walls = list(_PLANAR_WALLS)
    solid_features = []
    if half:   # E_z of the x-symmetric beam is even (PMC-like symmetric face); the solid's even mirror: symmetry
        optics_settings['bc_x_hi'] = 'symmetric' if polarization == 'ez' else 'antisymmetric'
        walls.remove('x_hi')
        solid_features.append(FeatureSpec('mirror', 'symmetry', ['x_hi']))
    solid_features = [FeatureSpec('walls', 'fixed', walls), *solid_features,
                      FeatureSpec(_FEATURE, _FEATURE_TYPE, ['drive'], _brillouin(brillouin))]
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=size_x, size_y=size_y), geometry=geometry,
        materials=[material or _water(), *extra_materials],
        physics=[PhysicsSpec(_OPTICS, f'em_fdfd_{polarization}', settings=optics_settings, features=[beam]),
                 PhysicsSpec(_SOLID, _SOLID_TYPE, settings=dict(_MIXED_P2 if mixed else _DISPLACEMENT_P2),
                             features=solid_features, mesh=_SOLID_MESH)],
        meshes=[MeshSpec(name=_OPTICS_MESH, kind='tensor', mode='uniform', dx=layout.cell),
                MeshSpec(name=_SOLID_MESH, kind='triangles', mode='nonuniform', dx=layout.coarse,
                         max_ratio=_MAX_RATIO, refinements=[RefinementSpec('drive', layout.fine)],
                         mirror_x=mesh_mirror)],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type=FREQUENCY_RESPONSE_STUDY, settings={'frequencies': list(frequencies),
                                                                 **(study_settings or {})}))


# ---- brillouin_water_2d_ez / _hz: against the homogeneous (unbounded-medium) reference ----
def _water_2d_specs(polarization):
    layout, frequencies = ((_WATER_2D, _WATER_2D_FREQUENCIES) if polarization == 'ez'
                           else (_WATER_2D_HZ, _WATER_2D_HZ_FREQUENCIES))
    spec = _planar_spec(f'Self-test: Brillouin spectrum of water, em_fdfd_{polarization} focused beam',
                        layout, polarization, {'reference': 'homogeneous', 'fit_peak': True}, frequencies)
    return {polarization: spec}


def _homogeneous_errors(result, errors, label):
    """psd L2 against psd_reference over the samples above _PEAK_SHARE of the reference peak, the fit shift against
    reference_shift and the incident power: {item: error} added to errors; returns (f, psd, reference)."""
    frequencies, psd, reference = _frequency_axis(result), _series(result, 'psd'), _series(result, 'psd_reference')
    keep = reference >= _PEAK_SHARE * float(np.max(reference))
    errors[f'{label} psd vs homogeneous (L2)'] = norm_relative_error(psd[keep], reference[keep])
    errors[f'{label} shift vs reference_shift'] = relative_error(float(result['shift']),
                                                                 float(result['reference_shift']))
    errors[f'{label} incident_power'] = relative_error(float(result['incident_power']), _BEAM_POWER)
    return frequencies, psd, reference


def _check_water_2d_ez(results_by_name):
    from .brillouin import band_integral, dho_fit
    result = _spectrum(results_by_name['ez'])
    errors = {}
    frequencies, psd, reference = _homogeneous_errors(result, errors, 'ez')
    reference_fit = dho_fit(frequencies, reference)
    errors['ez fwhm vs the reference fit'] = relative_error(float(result['fwhm']), reference_fit['fwhm'])
    expected_fraction = 2.0 * float(band_integral(frequencies, reference))
    errors['ez scattered_fraction vs the reference band'] = relative_error(float(result['scattered_fraction']),
                                                                         expected_fraction)
    kinds = {'psd vs homogeneous (L2)': 'psd', 'shift vs reference_shift': 'shift', 'incident_power': 'power',
             'fwhm vs the reference fit': 'fwhm', 'scattered_fraction vs the reference band': 'fraction'}
    tolerances = {item: _WATER_2D_TOLERANCE[kinds[item[3:]]] for item in errors}
    return scaled_worst_result(errors, tolerances, frequencies=float_list(frequencies), psd=float_list(psd),
                               psd_reference=float_list(reference), shift=result['shift'],
                               reference_shift=result['reference_shift'], fwhm=result['fwhm'],
                               reference_fwhm=reference_fit['fwhm'], q_mean=result['q_mean'],
                               fit_warning=result.get('fit_warning', ''))


def _check_water_2d_hz(results_by_name):
    result = _spectrum(results_by_name['hz'])
    errors = {}
    frequencies, psd, reference = _homogeneous_errors(result, errors, 'hz')
    kinds = {'psd vs homogeneous (L2)': 'psd', 'shift vs reference_shift': 'shift', 'incident_power': 'power'}
    tolerances = {item: _WATER_2D_HZ_TOLERANCE[kinds[item[3:]]] for item in errors}
    return scaled_worst_result(errors, tolerances, frequencies=float_list(frequencies), psd=float_list(psd),
                               psd_reference=float_list(reference), shift=result['shift'],
                               reference_shift=result['reference_shift'], q_mean=result['q_mean'])


# ---- brillouin_scan_columns: scan columns vs single positions, half model vs full ----
def _scan_specs():
    disc = GeometrySpec('disc', 'circle', {'cx': _SCAN.focus[0], 'cy': _SCAN.focus[1], 'r': _DISC_RADIUS})
    polystyrene = MaterialSpec('polystyrene', properties=dict(_POLYSTYRENE), selection=['disc'])
    scan = {'reference': 'none', 'fit_peak': False, 'scan_offsets_y': list(_SCAN_OFFSETS_Y)}
    single = {'reference': 'none', 'fit_peak': False}

    def spec(title, brillouin, **options):
        return _planar_spec(f'Self-test: Brillouin scan columns ({title})', _SCAN, 'ez', brillouin,
                            _SCAN_FREQUENCIES, extra_materials=(polystyrene,), extra_geometry=(disc,), **options)
    specs = {'scan': spec('three positions', scan, mesh_mirror=True)}
    for position in (2, 3):
        specs[f'single_{position}'] = spec(f'position {position} alone', single, mesh_mirror=True,
                                           focus_dy=_SCAN_OFFSETS_Y[position - 1])
    specs['half'] = spec('half model on an x_hi mirror plane', scan, half=True)
    return specs


def _check_scan(results_by_name):
    scan = _spectrum(results_by_name['scan'])
    errors, tolerances, details = {}, {}, {}

    def add(item, error, kind):
        errors[item], tolerances[item] = float(error), _SCAN_TOLERANCE[kind]
    frequencies = _frequency_axis(scan)
    positions = len(_SCAN_OFFSETS_Y)
    columns = {position: _series(scan, f'psd_{position}') for position in range(1, positions + 1)}
    add('scan psd positive', 0.0 if all(np.all(values > 0.0) for values in columns.values()) else math.inf, 'flag')
    for position in (2, 3):
        single = _spectrum(results_by_name[f'single_{position}'])
        if not np.array_equal(_frequency_axis(single), frequencies):
            add(f'single_{position} frequencies', math.inf, 'flag')
            continue
        expected = _series(single, 'psd')
        add(f'psd_{position} (column solve) vs its single-position run',
            np.max(np.abs(columns[position] - expected)) / np.max(np.abs(expected)), 'columns')
    half = _spectrum(results_by_name['half'])
    full_elements = _mesh_elements(results_by_name['scan'], _SOLID_MESH)
    half_elements = _mesh_elements(results_by_name['half'], _SOLID_MESH)
    identical = full_elements is not None and half_elements is not None and full_elements == 2 * half_elements
    for position, values in columns.items():
        mirrored = _series(half, f'psd_{position}')
        add(f'half model psd_{position} vs full', np.max(np.abs(mirrored - values)) / np.max(np.abs(values)),
            'mirror' if identical else 'mirror_other_mesh')
    extra_solves = len(frequencies) * (positions - 1)
    for name, results in (('scan', results_by_name['scan']), ('half', results_by_name['half'])):
        counted = _solid_counter(results, _SOLID, 'column_solves')
        add(f'{name} column_solves = frequencies x (positions - 1)',
            math.inf if counted is None else abs(float(counted) - extra_solves), 'counter')
    add('scan mirror_factor 1', abs(float(scan['mirror_factor']) - 1.0), 'factor')
    add('half mirror_factor 2', abs(float(half['mirror_factor']) - 2.0), 'factor')
    add('scan incident_power', relative_error(float(scan['incident_power']), _BEAM_POWER), 'power')
    add('half incident_power (unfolded)', relative_error(float(half['incident_power']), _BEAM_POWER), 'power')
    rows = scan['scan']['rows']
    offsets = [float(row[scan['scan']['columns'].index('offset_y')]) for row in rows]
    add('scan table offsets', 0.0 if np.allclose(offsets, _SCAN_OFFSETS_Y, rtol=0.0, atol=1e-15) else math.inf,
        'flag')
    add('moving-boundary drive found at the disc',
        0.0 if float(scan['drive_norms']['moving_boundary']) > 0.0 else math.inf, 'flag')
    details.update(meshes_identical=bool(identical), full_elements=full_elements, half_elements=half_elements,
                   frequencies=float_list(frequencies),
                   psd={str(position): float_list(values) for position, values in columns.items()},
                   drive_norms=dict(scan['drive_norms']))
    return scaled_worst_result(errors, tolerances, **details)


# ---- brillouin_reduced_sweep: adaptive vs direct, one position, constant loss factor ----
def _reduced_specs():
    glass = MaterialSpec('glass', properties=dict(_GLASS), selection=['all'])
    brillouin = {'reference': 'none', 'fit_peak': True}
    specs = {}
    for method in ('direct', 'adaptive'):
        settings = {'sweep_method': method}
        if method == 'adaptive':
            settings['rom_tolerance'] = _ROM_TOLERANCE
        specs[method] = _planar_spec(f'Self-test: Brillouin spectrum of a glass block ({method} sweep)', _GLASS_LAYOUT,
                                     'ez', brillouin, _GLASS_FREQUENCIES, material=glass, mixed=False,
                                     study_settings=settings)
    return specs


def _check_reduced(results_by_name):
    direct, adaptive = (_spectrum(results_by_name[name]) for name in ('direct', 'adaptive'))
    expected, reduced = _series(direct, 'psd'), _series(adaptive, 'psd')
    errors = {'adaptive psd vs direct': (float(np.max(np.abs(reduced - expected)) / np.max(np.abs(expected)))
                                         if np.array_equal(_frequency_axis(direct), _frequency_axis(adaptive))
                                         else math.inf),
              'direct psd positive': 0.0 if np.all(expected > 0.0) else math.inf}
    tolerances = {'adaptive psd vs direct': _REDUCED_TOLERANCE, 'direct psd positive': 1.0}
    return scaled_worst_result(errors, tolerances, frequencies=float_list(_frequency_axis(direct)),
                               psd_direct=float_list(expected), psd_adaptive=float_list(reduced),
                               shift=direct.get('shift'))


# ---- axisymmetric layouts (em_fdfd_bor optics, m = 1) ----
@dataclass(frozen=True)
class _Bor:
    """An (r, z) model: optical cell [m], window cells (r, z), focus z [m] on the axis, drive rectangle (r_max, z0, z1)
    [m] from the axis on optical cell faces, beam line target height [m], numerical aperture, fill factor, solid element
    sizes in the drive / elsewhere [m]."""
    cell: float
    cells: tuple
    focus_z: float
    drive: tuple
    line_z: float
    aperture: float
    fill: float
    fine: float
    coarse: float


# brillouin_bor_water: 2.25 x 5.1 um of 15 nm cells (26.6 per lambda0 / n), NA 0.6 filled pupil, the drive r < 1.2 um,
# 3 um along z around the focus (the beam line 1.55 um upstream), 1.05 um margins
_BOR_WATER = _Bor(cell=15e-9, cells=(150, 340), focus_z=2.55e-6, drive=(1.2e-6, 1.05e-6, 4.05e-6), line_z=1.0e-6,
                  aperture=0.6, fill=1.0, fine=30e-9, coarse=80e-9)
_BOR_WATER_FREQUENCIES = _frequencies(5.6e9, 9.2e9, 13)
_BOR_WATER_TOLERANCE = {'ratio': 5e-2, 'shift': 2.5e-2, 'area': 5e-2, 'torsion': 1e-12, 'power': 1e-9}
# brillouin_bor_low_na: NA 0.3, fill 0.7 (a Gaussian focus of effective NA ~0.21, waist ~0.8 um, Rayleigh range ~5 um
# >> the 3 um drive), drive r < 1.5 um, window radius 2.7 um
_BOR_LOW_NA = _Bor(cell=15e-9, cells=(180, 340), focus_z=2.55e-6, drive=(1.5e-6, 1.05e-6, 4.05e-6), line_z=1.0e-6,
                   aperture=0.3, fill=0.7, fine=30e-9, coarse=80e-9)
_BOR_LOW_NA_FREQUENCIES = _frequencies(6.0e9, 8.88e9, 13)
_BOR_LOW_NA_TOLERANCE = {'shift law': 5e-3, 'fwhm': 3e-2, 'shift local': 5e-3, 'area': 5e-2, 'power': 1e-9}
# brillouin_bor_cross_polarized: a small light model, 1.28 x 2.24 um of 16 nm cells, NA 0.6 fill 0.7, drive r < 0.48 um,
# 0.64 um along z; two order-2 solids (co and cross) on one mesh
_BOR_CROSS = _Bor(cell=16e-9, cells=(80, 140), focus_z=1.12e-6, drive=(0.48e-6, 0.8e-6, 1.44e-6), line_z=0.75e-6,
                  aperture=0.6, fill=0.7, fine=30e-9, coarse=60e-9)
_BOR_CROSS_FREQUENCIES = (6.8e9, 7.4e9, 8.0e9)
_BOR_CROSS_TOLERANCE = {'identity': 1e-9, 'torsion': 1e-12, 'flag': 1.0}


def _bor_spec(title, layout, solids, frequencies, brillouin):
    """An (r, z) model of `layout`: em_fdfd_bor (m = 1) with the focused beam on the axis, one axisymmetric mixed P2
    solid per (name, azimuthal_order, detection) of `solids` on the shared triangle mesh, each with its
    brillouin_scattering feature (params `brillouin`) on the drive rectangle and fixed walls on x_hi, y_lo, y_hi."""
    size_r, size_z = layout.cells[0] * layout.cell, layout.cells[1] * layout.cell
    line_z = cell_centre(_line_row(layout.line_z, layout.cell), layout.cell)
    radius, z0, z1 = layout.drive
    geometry = [GeometrySpec('beam_line', 'line', {'x0': 0.0, 'y0': line_z, 'x1': size_r, 'y1': line_z}),
                GeometrySpec('drive', 'rectangle', {'x0': 0.0, 'y0': z0, 'x1': radius, 'y1': z1})]
    beam = FeatureSpec(_BEAM, 'focused_beam', ['beam_line'], {
        'power': _BEAM_POWER, 'phase': 0.0, 'numerical_aperture': layout.aperture, 'fill_factor': layout.fill,
        'focus_z': layout.focus_z, 'direction': 'positive'})
    physics = [PhysicsSpec(_OPTICS, 'em_fdfd_bor', settings={'wavelength': _WAVELENGTH, 'azimuthal_order': 1},
                           features=[beam])]
    for name, order, detection in solids:
        physics.append(PhysicsSpec(
            name, _SOLID_TYPE, settings={'geometry': 'axisymmetric', 'azimuthal_order': order, **_MIXED_P2},
            features=[FeatureSpec('walls', 'fixed', list(_BOR_WALLS)),
                      FeatureSpec(_FEATURE, _FEATURE_TYPE, ['drive'], _brillouin({**brillouin,
                                                                                   'detection': detection}))],
            mesh=_SOLID_MESH))
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=size_r, size_y=size_z), geometry=geometry, materials=[_water()],
        physics=physics,
        meshes=[MeshSpec(name=_OPTICS_MESH, kind='tensor', mode='uniform', dx=layout.cell),
                MeshSpec(name=_SOLID_MESH, kind='triangles', mode='nonuniform', dx=layout.coarse,
                         max_ratio=_MAX_RATIO, refinements=[RefinementSpec('drive', layout.fine)])],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type=FREQUENCY_RESPONSE_STUDY, settings={'frequencies': list(frequencies)}))


_ORDER_0, _ORDER_2 = 'solid_m0', 'solid_m2'
_CROSS = 'solid_m2_cross'


# ---- brillouin_bor_water: m = 0 + m = 2 at NA 0.6 against the local reference ----
def _bor_water_specs():
    return {'bor': _bor_spec('Self-test: Brillouin spectrum of water, em_fdfd_bor focused beam NA 0.6, m = 0 + 2',
                             _BOR_WATER, ((_ORDER_0, 0, 'co_polarized'), (_ORDER_2, 2, 'co_polarized')),
                             _BOR_WATER_FREQUENCIES, {'reference': 'local', 'fit_peak': True})}


def _check_bor_water(results_by_name):
    from .brillouin import band_integral, dho_fit
    results = results_by_name['bor']
    order_0, order_2 = _spectrum(results, _ORDER_0), _spectrum(results, _ORDER_2)
    frequencies = _frequency_axis(order_0)
    errors, tolerances = {}, {}

    def add(item, error, kind):
        errors[item], tolerances[item] = float(error), _BOR_WATER_TOLERANCE[kind]
    if not np.array_equal(frequencies, _frequency_axis(order_2)):
        add('m = 0 and m = 2 frequencies', math.inf, 'ratio')
        return scaled_worst_result(errors, tolerances)
    psd = _series(order_0, 'psd') + _series(order_2, 'psd')
    reference = _series(order_0, 'psd_reference') + _series(order_2, 'psd_reference')
    fit, reference_fit = dho_fit(frequencies, psd), dho_fit(frequencies, reference)
    ratio = float(band_integral(frequencies, _series(order_2, 'psd'))) / float(band_integral(frequencies,
                                                                                            _series(order_0, 'psd')))
    errors['m = 2 / m = 0 band share in (0, bound]'] = _bound_error(ratio, _BOR_WATER_TOLERANCE['ratio'])
    tolerances['m = 2 / m = 0 band share in (0, bound]'] = 1.0
    add('shift (m = 0 + 2) vs the local reference fit', relative_error(fit['shift'], reference_fit['shift']), 'shift')
    add('DHO area (m = 0 + 2) vs the local reference area (sum rule)',
        relative_error(_dho_area(fit), _dho_area(reference_fit)), 'area')
    for label, result in (('m = 0', order_0), ('m = 2', order_2)):
        add(f'{label} torsion_drive_fraction (co-polarised: 0)', float(result['torsion_drive_fraction']), 'torsion')
        add(f'{label} incident_power', relative_error(float(result['incident_power']), _BEAM_POWER), 'power')
    return scaled_worst_result(errors, tolerances, m2_over_m0=ratio, frequencies=float_list(frequencies),
                               psd_total=float_list(psd), reference_total=float_list(reference),
                               fit={key: fit[key] for key in ('shift', 'fwhm', 'gamma', 'peak', 'r_squared')},
                               reference_fit={key: reference_fit[key] for key in ('shift', 'fwhm', 'gamma', 'peak')},
                               q_mean_m0=order_0['q_mean'])


# ---- brillouin_bor_low_na: m = 0 at NA 0.3 against 2 n v / lambda0 and the boxcar-cut q^2 line ----
def _bor_low_na_specs():
    return {'bor': _bor_spec('Self-test: Brillouin spectrum of water, em_fdfd_bor focused beam NA 0.3 (low-NA limit)',
                             _BOR_LOW_NA, ((_SOLID, 0, 'co_polarized'),), _BOR_LOW_NA_FREQUENCIES,
                             {'reference': 'local', 'fit_peak': True})}


def _boxcar_line(frequencies, shift, length):
    """(n_f,) relative psd of a uniform dilation drive exp(-j q0 z) over a length L (the drive rectangle) of unbounded
    water_532nm, the low-NA limit of the focused beam (case comment): the Lorentzian-in-q phonon line convolved with
    |c^(q)|^2 = sinc^2 of the cut, 2 Re int_0^L (L - t) exp(-a t) dt = 2 Re[L / a - (1 - exp(-a L)) / a^2], a = delta
    + j (Omega / v - q0), delta = Omega^2 eta_L / (2 rho v^3) (amplitude attenuation), q0 = 2 pi shift / v."""
    _, density, _, viscous, speed = _water_constants()
    omega = 2.0 * math.pi * np.asarray(frequencies, dtype=np.float64)
    a = omega ** 2 * viscous / (2.0 * density * speed ** 3) + 1j * (omega / speed - 2.0 * math.pi * shift / speed)
    return 2.0 * np.real(length / a - (1.0 - np.exp(-a * length)) / a ** 2)


def _check_bor_low_na(results_by_name):
    from .brillouin import dho_fit
    result = _spectrum(results_by_name['bor'])
    frequencies, psd, reference = _frequency_axis(result), _series(result, 'psd'), _series(result, 'psd_reference')
    index, _, _, _, speed = _water_constants()
    law_shift = 2.0 * index * speed / _WAVELENGTH
    shift, fwhm = float(result['shift']), float(result['fwhm'])
    length = _BOR_LOW_NA.drive[2] - _BOR_LOW_NA.drive[1]
    model_fit = dho_fit(frequencies, _boxcar_line(frequencies, shift, length))
    fit, reference_fit = dho_fit(frequencies, psd), dho_fit(frequencies, reference)
    errors = {'shift vs 2 n v / lambda0': relative_error(shift, law_shift),
              'fwhm vs the boxcar-cut q^2 line (fit vs fit)': relative_error(fwhm, model_fit['fwhm']),
              'shift vs the local reference fit': relative_error(shift, float(result['reference_shift'])),
              'DHO area vs the local reference area (sum rule)': relative_error(_dho_area(fit),
                                                                                _dho_area(reference_fit)),
              'incident_power': relative_error(float(result['incident_power']), _BEAM_POWER)}
    kinds = ('shift law', 'fwhm', 'shift local', 'area', 'power')
    tolerances = {item: _BOR_LOW_NA_TOLERANCE[kind] for item, kind in zip(errors, kinds)}
    _, density, _, viscous, _ = _water_constants()
    wavenumber = 4.0 * math.pi * index / _WAVELENGTH
    return scaled_worst_result(errors, tolerances, law_shift=law_shift,
                               law_fwhm=wavenumber ** 2 * viscous / (2.0 * math.pi * density), shift=shift, fwhm=fwhm,
                               boxcar_fwhm=model_fit['fwhm'], drive_length=length, q_mean=result['q_mean'],
                               frequencies=float_list(frequencies), psd=float_list(psd),
                               psd_reference=float_list(reference))


# ---- brillouin_bor_cross_polarized: odd order 2 = even order 2 for an isotropic p44 = 0 liquid ----
def _bor_cross_specs():
    return {'bor': _bor_spec('Self-test: Brillouin cross-polarised detection, em_fdfd_bor, two order-2 solids',
                             _BOR_CROSS, ((_ORDER_2, 2, 'co_polarized'), (_CROSS, 2, 'cross_polarized')),
                             _BOR_CROSS_FREQUENCIES, {'reference': 'none', 'fit_peak': False})}


def _check_bor_cross(results_by_name):
    results = results_by_name['bor']
    co, cross = _spectrum(results, _ORDER_2), _spectrum(results, _CROSS)
    co_psd, cross_psd = _series(co, 'psd'), _series(cross, 'psd')
    errors = {'cross-polarised psd vs co-polarised order 2': (
        float(np.max(np.abs(cross_psd - co_psd)) / np.max(np.abs(co_psd)))
        if np.array_equal(_frequency_axis(co), _frequency_axis(cross)) else math.inf),
        'co psd positive': 0.0 if np.all(co_psd > 0.0) else math.inf,
        'cross torsion_drive_fraction (p44 = 0: none)': float(cross['torsion_drive_fraction']),
        'co torsion_drive_fraction': float(co['torsion_drive_fraction'])}
    tolerances = {'cross-polarised psd vs co-polarised order 2': _BOR_CROSS_TOLERANCE['identity'],
                  'co psd positive': _BOR_CROSS_TOLERANCE['flag'],
                  'cross torsion_drive_fraction (p44 = 0: none)': _BOR_CROSS_TOLERANCE['torsion'],
                  'co torsion_drive_fraction': _BOR_CROSS_TOLERANCE['torsion']}
    return scaled_worst_result(errors, tolerances, frequencies=float_list(_frequency_axis(co)),
                               psd_co=float_list(co_psd), psd_cross=float_list(cross_psd))


_PLANAR_LAYOUT_TEXT = ('two meshes on one window: the optics on a uniform tensor mesh (default CPML outside), the '
                       'solid on triangles (plane strain, formulation mixed P2 / P1, fine in the drive rectangle, '
                       'graded 1.3 to the coarse size, fixed walls about 1 um, 1.3 attenuation lengths, from the '
                       'drive); a '
                       f'{_BEAM_POWER:g} W/m focused_beam (532 nm, +y) from a line over the window width just below '
                       'the drive, focused at its centre')
_BOR_LAYOUT_TEXT = ('em_fdfd_bor (azimuthal_order 1) on a uniform tensor mesh, the focused_beam (1 W, +z) from a line '
                    'over the window radius just below the drive, focused on the axis at its centre; axisymmetric '
                    'solids on one triangle mesh (formulation mixed P2 / P1, 30 nm in the drive rectangle from the '
                    'axis, 80 nm elsewhere, fixed walls on x_hi, y_lo, y_hi about 1 um from the drive)')

BRILLOUIN_SPECTRUM_CASES = (
    SelfTestCase(
        name='brillouin_water_2d_ez',
        label='Solid mechanics: Brillouin spectrum of water, em_fdfd_ez focused beam vs the homogeneous reference',
        description=(f'em_fdfd_ez and a water_532nm solid, {_PLANAR_LAYOUT_TEXT}: a 3.4 x 3.2 um window of 10 nm cells '
                     '(39.9 per lambda0 / n), waist 0.4 um, a 0.8 x 1.2 um drive rectangle with 25 nm elements (80 nm '
                     'elsewhere); brillouin_scattering with reference homogeneous at 13 frequencies 5.9 - 8.9 GHz, '
                     'cpu. The homogeneous reference is the exact plane-wave sum of the same drive (the kappa sigma~ '
                     'raster of the optical cells in the drive) in an unbounded medium, so it holds the finite drive '
                     'length (a 1.2 um drive is 1.7 attenuation lengths: the line is about twice the q^2 law width '
                     '0.66 GHz) and the beam q content. Checks: psd against psd_reference in L2 over the samples above '
                     f"{_PEAK_SHARE:g} of its peak ({_WATER_2D_TOLERANCE['psd']:g}: the drive samples the optical "
                     'fields bilinearly at the quadrature points where the reference rasters the cell values, (q '
                     'dx)^2 / 12 = 0.8 % low uniformly; P2 / P1 dispersion ~1e-4 in frequency, ~0.5 % on the flanks; '
                     f"wall echoes ~3e-3), the fit shift against reference_shift ({_WATER_2D_TOLERANCE['shift']:g}: "
                     'the dispersion), the fit fwhm against the DHO fit of psd_reference '
                     f"({_WATER_2D_TOLERANCE['fwhm']:g}), scattered_fraction against 2 x the band integral of "
                     f"psd_reference ({_WATER_2D_TOLERANCE['fraction']:g}: the uniform scale), incident_power = power "
                     f"({_WATER_2D_TOLERANCE['power']:g}). The q^2 linewidth law needs a drive longer than a few "
                     'attenuation lengths: brillouin_bor_low_na and examples/mechanics/brillouin_water_2d. Tolerances '
                     'are first estimates; measured = error = the worst deviation in tolerance units.'),
        specs=lambda: _water_2d_specs('ez'), check=_check_water_2d_ez, heavy=True),
    SelfTestCase(
        name='brillouin_water_2d_hz',
        label='Solid mechanics: Brillouin spectrum of water, em_fdfd_hz focused beam vs the homogeneous reference',
        description=(f'em_fdfd_hz and a water_532nm solid, {_PLANAR_LAYOUT_TEXT}: the brillouin_water_2d_ez layout on '
                     '12.5 nm optical cells (31.9 per lambda0 / n) with 30 nm solid elements in the drive, 7 '
                     'frequencies 5.9 - 8.9 GHz, cpu. The in-plane E (face means at the cell centres) gives G_xx, '
                     'G_yy, G_xy; with p11 = p12 and p44 = 0 the drive is still a pure dilation p (E . E). Checks: psd '
                     f"against psd_reference (homogeneous) in L2 above {_PEAK_SHARE:g} of its peak "
                     f"({_WATER_2D_HZ_TOLERANCE['psd']:g}: bilinear sampling (q dx)^2 / 12 = 1.3 % uniform, P2 / P1 "
                     f"dispersion), the fit shift against reference_shift ({_WATER_2D_HZ_TOLERANCE['shift']:g}), "
                     f"incident_power = power ({_WATER_2D_HZ_TOLERANCE['power']:g}). Tolerances are first estimates; "
                     'measured = error = the worst deviation in tolerance units.'),
        specs=lambda: _water_2d_specs('hz'), check=_check_water_2d_hz),
    SelfTestCase(
        name='brillouin_scan_columns',
        label='Solid mechanics: Brillouin scan columns on one factor, and a mirror plane',
        description=(f'em_fdfd_ez, {_PLANAR_LAYOUT_TEXT}: a 3.4 x 2.0 um window of 20 nm cells (0.6 um margins along '
                     'y: the identities need none), an inline polystyrene '
                     f'disc (r {_DISC_RADIUS * 1e6:g} um, n 1.598, E 3.4e9, nu 0.34, rho 1050, p11 0.30, p12 0.31, '
                     'loss factor 0.01) at the focus in water_532nm (moving boundary at the disc), a 0.8 x 0.8 um '
                     'drive (27 nm elements, 100 nm elsewhere), reference none, 3 frequencies 6.8 / 7.4 / 8.0 GHz. '
                     'Runs: scan (scan_offsets_y 0, 0.1, -0.15 um: positions 2 and 3 are extra optical solves on the '
                     'kept optics factor and frequency_response.HarmonicSystem.solve_columns on the solid factor), '
                     'single_2 / single_3 (one position, the focus itself moved by that offset), half (the scan on x < '
                     '1.7 um, x_hi a mirror plane: em bc symmetric, solid symmetry feature, mirror factor 2; the full '
                     'mesh is mirror_x, so it meshes the same lower half). Checks: psd_2 / psd_3 against the '
                     f"single-position runs ({_SCAN_TOLERANCE['columns']:g} of max: one factor, two solve paths, "
                     'round-off), the half model psd_k against the full one (mirror factor; '
                     f"{_SCAN_TOLERANCE['mirror']:g} when the meshes are node-identical, 2 x half elements = full, "
                     f"else {_SCAN_TOLERANCE['mirror_other_mesh']:g}: the P2 / P1 dispersion of two meshes; details "
                     'meshes_identical), column_solves = frequencies x (positions - 1) on both, mirror_factor 1 / 2, '
                     'incident_power = power (unfolded), the scan table offsets, a moving-boundary drive found and '
                     'every psd > 0. cpu, and variant gpu (FDFD and solid factors on the device, the cuDSS column path '
                     f'with its device weights; tolerance {_SCAN_GPU_TOLERANCE:g} x the cpu ones). Tolerances are '
                     'first estimates; measured = error = the worst deviation in tolerance units.'),
        specs=_scan_specs, check=_check_scan,
        variants=(SelfTestVariant('gpu', tolerance=_SCAN_GPU_TOLERANCE, backend='gpu'),)),
    SelfTestCase(
        name='brillouin_reduced_sweep',
        label='Solid mechanics: Brillouin spectrum by an adaptive reduced sweep vs direct',
        description=('em_fdfd_ez and a glass-like solid filling a 3.2 x 2.0 um window (inline eps_r 1.45^2, E 73e9, nu '
                     '0.17, rho 2200, p11 0.121, p12 0.270, mechanical_loss_factor 0.02: a constant loss factor, no '
                     'frequency-dependent term, so reduced sweeps are allowed; displacement formulation P2), 12.5 nm '
                     'optical cells, a 0.6 x 0.6 um drive (30 nm elements, 80 nm elsewhere), fixed walls, one scan '
                     'position, 9 frequencies 28 - 36 GHz around f_B ~ 32.6 GHz, cpu. Runs direct and adaptive '
                     f'(rom_tolerance {_ROM_TOLERANCE:g}; the drive is the load and its readout a precomputed reduced '
                     'functional): max |psd_adaptive - psd_direct| / max psd_direct below '
                     f'{_REDUCED_TOLERANCE:g} (the reduced-model residual times the loss Q ~ 50; first estimate), '
                     'psd > 0. measured = error = the worst deviation in tolerance units.'),
        specs=_reduced_specs, check=_check_reduced),
    SelfTestCase(
        name='brillouin_bor_water',
        label='Solid mechanics: Brillouin spectrum of water, em_fdfd_bor NA 0.6, solids m = 0 + m = 2 vs the local '
              'reference',
        description=(f'{_BOR_LAYOUT_TEXT}: water_532nm, a 2.25 x 5.1 um (r, z) window of 15 nm cells (26.6 per lambda0 '
                     '/ n), numerical_aperture 0.6, fill_factor 1, the drive r < 1.2 um over 3 um of z around the '
                     'focus; solids solid_m0 (azimuthal_order 0) and solid_m2 (2), each with a co-polarised '
                     'brillouin_scattering on the same optics (the order-split exemption of definition_helpers), '
                     'reference local, 13 frequencies 5.6 - 9.2 GHz, cpu. Checks on the total psd_m0 + psd_m2 '
                     'against the summed local references: the DHO fit shifts within '
                     f"{_BOR_WATER_TOLERANCE['shift']:g} (the local reference takes the phase-gradient |q| per point "
                     'and misses the q content of the amplitude pattern, ~<theta^2> / 6 of the focusing cone: two '
                     'waves at the 0.47 rad rim lift the mean |q| by 3.6 %, the filled pupil by about 1 %) and the DHO '
                     f"areas (pi / 2) peak gamma (1 - gamma^2 / (4 f0^2)) within {_BOR_WATER_TOLERANCE['area']:g} (the "
                     'sum rule: for a dilation drive c in one liquid both give 2 int psd df = (k_B T / K) int |kappa '
                     'c|^2 dV; the 3 um drive cut broadens the computed line, which the areas do not see; the fit of '
                     'a non-DHO line shape remains); '
                     'the m = 2 / m = 0 band-integral share in (0, '
                     f"{_BOR_WATER_TOLERANCE['ratio']:g}] (the m = 2 part of E . E grows like (NA / n)^4; first "
                     'estimate), both torsion_drive_fraction 0 (co-polarised), incident_power = power. Tolerances are '
                     'first estimates; measured = error = the worst deviation in tolerance units.'),
        specs=_bor_water_specs, check=_check_bor_water, heavy=True),
    SelfTestCase(
        name='brillouin_bor_low_na',
        label='Solid mechanics: Brillouin spectrum of water, em_fdfd_bor NA 0.3 vs its low-NA analytic limit',
        description=(f'{_BOR_LAYOUT_TEXT}: water_532nm, a 2.7 x 5.1 um (r, z) window of 15 nm cells, '
                     'numerical_aperture 0.3, fill_factor 0.7 (a Gaussian focus of effective NA ~0.21, waist ~0.8 um, '
                     'Rayleigh range ~5 um), the drive r < 1.5 um over 3 um of z around the focus, one solid '
                     '(azimuthal_order 0), reference local, 13 frequencies 6.0 - 8.88 GHz, cpu. Low-NA limit: inside '
                     'the drive the beam is a plane wave exp(-2 j n k0 z) of nearly uniform amplitude cut by the drive '
                     'length L, so the line is the Lorentzian-in-q phonon line of half width delta = Omega^2 (4/3 eta '
                     '+ eta_b) / (2 rho v^3) convolved with the sinc^2 of the cut: S ~ Re[L / a - (1 - exp(-a L)) / '
                     'a^2], a = delta + j (Omega / v - q0) (L = 3 um = 4.2 attenuation lengths: about 35 % wider than '
                     'the q^2 law 0.66 GHz). Checks: the DHO fit shift against 2 n v / lambda0 = 7.439 GHz '
                     f"({_BOR_LOW_NA_TOLERANCE['shift law']:g}: the pupil pairs lower |q| by ~<theta^2> / 4 ~ 3e-3, "
                     'the Yee dispersion raises it by (k h)^2 / 24 = 2.3e-3, P2 / P1 ~ -5e-4), the fit fwhm against '
                     'the DHO fit of that boxcar line on the same frequencies centred at the computed shift '
                     f"({_BOR_LOW_NA_TOLERANCE['fwhm']:g}: the beam envelope tapers to ~0.92 of |c|^2 at the drive "
                     'ends, ~-1.4 %; the Lorentzian-in-q kernel vs the DHO, (Gamma / Omega)^2 ~ 0.8 %), the shift '
                     f"against reference_shift (local, {_BOR_LOW_NA_TOLERANCE['shift local']:g}), the DHO areas "
                     f"against the local reference's (sum rule, {_BOR_LOW_NA_TOLERANCE['area']:g}), incident_power = "
                     'power. Tolerances are first estimates; measured = error = the worst deviation in tolerance '
                     'units.'),
        specs=_bor_low_na_specs, check=_check_bor_low_na, heavy=True),
    SelfTestCase(
        name='brillouin_bor_cross_polarized',
        label='Solid mechanics: cross-polarised Brillouin detection equals the order-2 co-polarised part (p44 = 0)',
        description=(f'{_BOR_LAYOUT_TEXT}; a light model: water_532nm, a 1.28 x 2.24 um (r, z) window of 16 nm cells, '
                     'numerical_aperture 0.6, fill_factor 0.7, the drive r < 0.48 um over 0.64 um of z (60 nm '
                     'elements outside it), two order-2 solids on one mesh: solid_m2 co_polarized and solid_m2_cross '
                     'cross_polarized, reference none, 3 frequencies, cpu. For an isotropic liquid with p11 = p12 and '
                     'p44 = 0 the drive is the dilation p (E_c . E_i); the cross-polarised product (the incident m = '
                     '1 field rotated by 90 deg) has the sin 2 phi amplitude (A_r^2 - A_phi^2 + A_z^2) / 2 of the '
                     'co-polarised cos 2 phi part and no shear rows, so the odd column equals the even one on the '
                     "same operator: psd_cross = psd_co within "
                     f"{_BOR_CROSS_TOLERANCE['identity']:g} of max (round-off), both torsion_drive_fraction 0 "
                     f"(within {_BOR_CROSS_TOLERANCE['torsion']:g}: no shear stress), psd > 0. measured = error = the "
                     'worst deviation in tolerance units.'),
        specs=_bor_cross_specs, check=_check_bor_cross),
)

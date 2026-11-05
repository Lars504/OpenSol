"""Self-tests of em_eme (eme_plan section 5): CPU checks of the mode sets, interfaces, cascade, periodic block and bend
map on hand-built slab lines with exact references, model runs (lossy thin-film stack, em_fdfd_ez / _hz comparisons)
and the 'linear_taper_eme' / 'mmi_1x2_eme' templates."""
import cmath
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, StudySpec)

_TYPE = 'em_eme'   # = definition.EM_EME.name (definition imports this module)
_PHYSICS = 'eme'
_WAVELENGTH = 1.55e-6
_OMEGA = 2.0 * math.pi * C0 / _WAVELENGTH
_K0 = _OMEGA / C0


def _slab_line(eps, width, ends, polarization='ez', stretch=None):
    """SlabLine of uniform cells `width` [m] with relative permittivity eps (m,), mu 1, cell / face stretches
    stretch = (cells (m,), faces (m + 1,)) or none; hz face weight 1 / (mean eps of the two cells)."""
    from ..em.slab_modes import SlabLine
    eps = np.asarray(eps, dtype=np.complex128)
    n_cells = eps.size
    cell_stretch, face_stretch = (np.ones(n_cells, dtype=np.complex128), np.ones(n_cells + 1, dtype=np.complex128)) \
        if stretch is None else stretch
    if polarization == 'ez':
        cell_weight, face_weight = np.ones(n_cells, dtype=np.complex128), np.ones(n_cells + 1, dtype=np.complex128)
    else:
        cell_weight = eps
        lower = np.concatenate(([eps[-1] if ends[0] == 'periodic' else eps[0]], eps))
        upper = np.concatenate((eps, [eps[0] if ends[0] == 'periodic' else eps[-1]]))
        face_weight = 2.0 / (lower + upper)
    return SlabLine(polarization=polarization, cell_widths=np.full(n_cells, float(width)), cell_stretch=cell_stretch,
                    cell_weight=cell_weight, index_term=eps, active=np.ones(n_cells, dtype=bool),
                    face_inverse_spacing=np.full(n_cells + 1, 1.0 / float(width)), face_stretch=face_stretch,
                    face_weight=face_weight, lo_end=ends[0], hi_end=ends[1])


def _mode_set(line, n_modes, discrete_spacing=None, drop_pml_modes=False):
    """EME ModeSet of line, PML modes kept unless drop_pml_modes (n_modes 0 = every cell)."""
    from ..em.slab_modes import solve_slab_modes
    from .mode_sets import mode_set
    modes = solve_slab_modes(line, _OMEGA, n_modes or line.cell_widths.size, normal_spacing=discrete_spacing,
                             drop_pml_modes=drop_pml_modes)
    return mode_set(line, modes, discrete_spacing)


def _full_matrix(smatrix):
    return np.block([[smatrix.s11, smatrix.s12], [smatrix.s21, smatrix.s22]])


# ---- shared line helpers ----
_GUIDE_CELL = 25e-9


def _cpml_stretch(n_cells, n_layer=10, strength=2.0):
    """(cells, faces) complex stretch 1 - j strength (depth / n_layer)^2 in n_layer cells at both ends."""
    def profile(positions):
        depth = np.maximum(np.maximum(n_layer - positions, positions - (n_cells - n_layer)), 0.0)
        return 1.0 - 1j * strength * (depth / n_layer) ** 2
    return profile(np.arange(n_cells) + 0.5), profile(np.arange(n_cells + 1, dtype=np.float64))


# ---- thin-film stack between transverse-periodic uniform sections (transfer-matrix reference) ----
_FILM_INDICES = (1.0, 1.5, 2.5, 1.5, 2.5, 2.0)              # input medium, four layers, output medium
_FILM_LENGTHS = (0.30e-6, 0.21e-6, 0.13e-6, 0.21e-6, 0.13e-6, 0.32e-6)   # m, ends: length inside the window
_FILM_CELL = 50e-9          # transverse cell of the periodic line (the Bragg CPU case)
_FILM_LINE_CELLS = 6


def _admittance(polarization, index):
    """y of the continuity of y (A - B): ez n (H ~ n E), hz 1 / n (E ~ H / n)."""
    return index if polarization == 'ez' else 1.0 / index


def stack_reference(indices, lengths, polarization):
    """(S11, S21) of power-normalised normal-incidence amplitudes between the window planes of a layer stack (each
    layer its index and length, the first / last the port media) by the amplitude transfer matrix."""
    forward, backward = 1.0 + 0.0j, 0.0j   # at the hi plane: transmitted wave only
    for index in range(len(indices) - 1, -1, -1):
        wavenumber = _K0 * indices[index]
        forward *= cmath.exp(1j * wavenumber * lengths[index])
        backward *= cmath.exp(-1j * wavenumber * lengths[index])
        if index > 0:
            ratio = _admittance(polarization, indices[index]) / _admittance(polarization, indices[index - 1])
            total, difference = forward + backward, ratio * (forward - backward)
            forward, backward = 0.5 * (total + difference), 0.5 * (total - difference)
    scale = math.sqrt(_admittance(polarization, indices[-1]) / _admittance(polarization, indices[0]))
    return backward / forward, scale / forward


# ---- 3. complete basis: narrow | wide | narrow guide between pec walls ----
_STEP_CELLS = 60
_STEP_NARROW = (26, 34)          # core cells (0.2 um)
_STEP_WIDE = (18, 42)            # core cells (0.6 um)
_STEP_LENGTH_CELLS = (12, 32, 12)
_COMPLETE_TOLERANCE = 1e-10


def _step_lines(polarization='ez'):
    """Narrow and wide guide lines (eps 4 core, 2.25 cladding, dirichlet ends)."""
    lines = []
    for core in (_STEP_NARROW, _STEP_WIDE):
        eps = np.full(_STEP_CELLS, 2.25)
        eps[core[0]:core[1]] = 4.0
        lines.append(_slab_line(eps, _GUIDE_CELL, ('dirichlet', 'dirichlet'), polarization))
    return lines


def _lattice_residual(lines, mode_sets, cascade, lengths):
    """Largest relative residual of the 2D Yee lattice equations (em_fdfd_ez / _hz rows) at the interior cells of the
    reconstructed field (mode 1 incident from lo): (W+ (u_{i+1} - u_i) - W- (u_i - u_{i-1})) / D^2 + (A_line u)_i /
    cell_weight = 0, W = 1 (ez, mu 1) or 1 / (mean eps_t of the two cells) (hz)."""
    from .mode_sets import longitudinal_flux, section_fields
    incident = np.zeros(mode_sets[0].n_modes, dtype=np.complex128)
    incident[0] = 1.0
    columns, column_lines = [], []
    for index, (line, modes, cells) in enumerate(zip(lines, mode_sets, _STEP_LENGTH_CELLS)):
        forward, backward = cascade.section_amplitudes(index, 'lo', incident)
        centres = (np.arange(cells) + 0.5) * _GUIDE_CELL
        u, _ = section_fields(line, modes, forward, backward, centres, lengths[index], _OMEGA)
        columns.append(u)
        column_lines += [line] * cells
    field = np.concatenate(columns)
    worst, scale = 0.0, 0.0
    for column in range(1, field.shape[0] - 1):
        line = column_lines[column]
        weight = np.asarray(line.cell_weight)
        if line.polarization == 'ez':
            lo_face = hi_face = 1.0
        else:   # the em_fdfd_hz x-face weight: 1 / face_average of eps_t (eps_yy) of the two cells
            lo_face = 2.0 / (np.asarray(column_lines[column - 1].cell_weight) + weight)
            hi_face = 2.0 / (weight + np.asarray(column_lines[column + 1].cell_weight))
        flux = longitudinal_flux(line, field[column:column + 1])[0]
        cell_term = _K0 ** 2 * np.asarray(line.index_term) / weight * field[column]
        transverse = (flux[1:] - flux[:-1]) / _GUIDE_CELL + cell_term
        along = (hi_face * (field[column + 1] - field[column])
                 - lo_face * (field[column] - field[column - 1])) / _GUIDE_CELL ** 2
        worst = max(worst, float(np.max(np.abs(along + transverse))))
        scale = max(scale, float(np.max(np.abs(cell_term))))
    return worst / scale


def _complete_errors(discrete_spacing, polarization='ez'):
    """(symmetry, power, lattice residual or NaN) of the complete-basis narrow | wide | narrow cascade."""
    from .interfaces import cascade_mode_sets
    from .mode_sets import is_propagating
    narrow, wide = _step_lines(polarization)
    lines = [narrow, wide, narrow]
    narrow_modes = _mode_set(narrow, 0, discrete_spacing)   # both narrow sections share one solve
    mode_sets = [narrow_modes, _mode_set(wide, 0, discrete_spacing), narrow_modes]
    lengths = [cells * _GUIDE_CELL for cells in _STEP_LENGTH_CELLS]
    cascade = cascade_mode_sets(mode_sets, lengths)
    full = _full_matrix(cascade.total)
    symmetry = float(np.max(np.abs(full - full.T)) / np.max(np.abs(full)))
    ports = (mode_sets[0], mode_sets[-1])
    power = np.concatenate([modes.power for modes in ports])
    propagating = np.concatenate([is_propagating(modes) for modes in ports])
    carried = np.sum(np.abs(full[propagating][:, propagating]) ** 2 * power[propagating][:, None], axis=0)
    power_error = float(np.max(np.abs(carried / power[propagating] - 1.0)))
    residual = (_lattice_residual(lines, mode_sets, cascade, lengths) if discrete_spacing is not None
                else math.nan)
    return {'symmetry': symmetry, 'power': power_error, 'lattice_residual': residual,
            'length_sweep': _length_sweep_error(mode_sets, lengths),
            'propagating_modes': int(np.count_nonzero(propagating[:ports[0].n_modes]))}


_SWEEP_FACTOR = 1.7


def _length_sweep_error(mode_sets, lengths):
    """max |S| deviation of the length-sweep re-cascade (cascade.total_smatrix with the middle section stretched) from
    a fresh cascade of the stretched lengths, relative to max |S|."""
    from .cascade import total_smatrix
    from .interfaces import cascade_mode_sets, interface_smatrix
    stretched = [lengths[0], _SWEEP_FACTOR * lengths[1], lengths[2]]
    interfaces = [interface_smatrix(left, right) for left, right in zip(mode_sets, mode_sets[1:])]
    phases = [np.exp(-1j * modes.k * length) for modes, length in zip(mode_sets, stretched)]
    swept = _full_matrix(total_smatrix(phases, interfaces))
    fresh = _full_matrix(cascade_mode_sets(mode_sets, stretched).total)
    return float(np.max(np.abs(swept - fresh)) / np.max(np.abs(fresh)))


def _check_complete_basis(results_by_name):
    """Complete-basis lossless step: S = S^T, power conserved over the propagating modes (continuum and discrete, ez
    and hz); the discrete cascade solves the 2D Yee lattice of the sectioned guide."""
    details = {f'{variant}_{polarization}': _complete_errors(spacing, polarization)
               for polarization in ('ez', 'hz') for variant, spacing in (('continuum', None), ('discrete', _GUIDE_CELL))}
    values = [value for variant in details.values() for name, value in variant.items()
              if name != 'propagating_modes' and not math.isnan(value)]
    measured = max(values)
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': _COMPLETE_TOLERANCE,
            'details': details}


# ---- 4. model run: the thin-film stack through em_eme ----
_MODEL_CELL = 10e-9
_MODEL_HEIGHT = 0.16e-6     # y-periodic strip (16 cells)
_MODEL_TOLERANCE = 1e-8


def _film_model_spec(polarization, lossy=False, indices=_FILM_INDICES, lengths=_FILM_LENGTHS, features=()):
    """em_eme along x through a thin-film stack in a y-periodic strip (pec x faces: no padding); lossy: layer
    _LOSSY_LAYER gets sigma_e _LOSSY_SIGMA and the derived P_abs = its absorbed_power integral."""
    edges = np.concatenate(([0.0], np.cumsum(lengths)))
    geometry, layer_names = [], {}
    for number, (index, x0, x1) in enumerate(zip(indices, edges[:-1], edges[1:])):
        name = f'layer_{number}'
        geometry.append(GeometrySpec(name, 'rectangle', {'x0': float(x0), 'y0': 0.0, 'x1': float(x1),
                                                          'y1': _MODEL_HEIGHT}))
        layer_names.setdefault('lossy' if lossy and number == _LOSSY_LAYER else index, []).append(name)
    materials = [MaterialSpec('lossy', properties={'eps_r': indices[_LOSSY_LAYER] ** 2, 'sigma_e': _LOSSY_SIGMA},
                              selection=names) if index == 'lossy' else
                 MaterialSpec(f'n_{index:g}'.replace('.', 'p'), properties={'eps_r': index ** 2}, selection=names)
                 for index, names in layer_names.items()]
    settings = {'polarization': polarization, 'propagation_axis': 'x', 'wavelength': _WAVELENGTH,
                'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'}
    derived = [DerivedValueSpec('T1', f'{_PHYSICS}_transmission_1'), DerivedValueSpec('R', f'{_PHYSICS}_reflection')]
    if lossy:
        derived.append(DerivedValueSpec('P_abs', 'absorbed_power', [f'layer_{_LOSSY_LAYER}'], 'integral'))
    return ModelSpec(
        title=f"em_eme {'lossy ' if lossy else ''}thin-film stack ({polarization})",
        domain=DomainSpec(size_x=float(edges[-1]), size_y=_MODEL_HEIGHT),
        geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec('ports', 'eme_ports'), FeatureSpec('sections', 'eme_section_table'), *features])],
        mesh=MeshSpec(mode='uniform', dx=_MODEL_CELL),
        study=StudySpec(type='stationary'),
        derived_values=derived)


def _port_entries(results):
    """(S11, S21, S12, S22) of port mode 1 from the eme_ports S-matrix extras of a run."""
    ports = results.result(_PHYSICS, 'ports')
    matrix = np.asarray(ports['s_matrix_real']) + 1j * np.asarray(ports['s_matrix_imag'])
    n_lo = int(ports['s_matrix_modes']['lo'])
    return complex(matrix[0, 0]), complex(matrix[n_lo, 0]), complex(matrix[0, n_lo]), complex(matrix[n_lo, n_lo])


# ---- 4b. model run: absorbed power of a lossy layer in the stack ----
_LOSSY_LAYER = 2                  # the first index-2.5 layer (0.13 um)
_LOSSY_SIGMA = 4096.0             # S/m, float32-exact: Im eps = -0.38 at 1.55 um
_LOSSY_TOLERANCE = 1e-6


def _lossy_indices():
    """Stack indices with the lossy layer's complex n = sqrt(eps - j sigma / (omega eps0)) (Im < 0)."""
    indices = list(_FILM_INDICES)
    indices[_LOSSY_LAYER] = cmath.sqrt(_FILM_INDICES[_LOSSY_LAYER] ** 2 - 1j * _LOSSY_SIGMA / (_OMEGA * EPS0))
    return indices


def _check_lossy_film(results_by_name):
    """Port entries of mode 1, section lengths and scalars T1 / R vs the complex-index transfer matrix (_MODEL_TOLERANCE),
    the absorbed_power integral over the layer vs 1 - R - T (_LOSSY_TOLERANCE); error in tolerance units."""
    details, worst = {}, 0.0
    indices = _lossy_indices()
    for polarization in ('ez', 'hz'):
        results = results_by_name[polarization]
        s11, s21 = stack_reference(indices, _FILM_LENGTHS, polarization)
        s22, s12 = stack_reference(indices[::-1], _FILM_LENGTHS[::-1], polarization)
        measured = _port_entries(results)
        errors = {name: abs(value - reference) for name, value, reference
                  in zip(('s11', 's21', 's12', 's22'), measured, (s11, s21, s12, s22))}
        lengths = [float(row[2]) for row in results.result(_PHYSICS, 'sections')['rows']]
        errors['sections'] = (max(abs(a - b) / b for a, b in zip(lengths, _FILM_LENGTHS))
                              if len(lengths) == len(_FILM_LENGTHS) else 1.0)   # 1 = wrong section count
        derived = results.summary()['derived']
        errors['T1'] = abs(float(derived['T1']) - abs(s21) ** 2)   # lossless port media: no loss term
        errors['R'] = abs(float(derived['R']) - abs(s11) ** 2)
        absorbed = 1.0 - abs(s11) ** 2 - abs(s21) ** 2
        errors['P_abs'] = abs(float(derived['P_abs']) - absorbed)
        scaled = {name: float(value) / (_LOSSY_TOLERANCE if name == 'P_abs' else _MODEL_TOLERANCE)
                  for name, value in errors.items()}
        details[polarization] = {'errors': {name: float(value) for name, value in errors.items()},
                                 'errors_in_tolerance_units': scaled, 'section_lengths': float_list(lengths),
                                 'P_abs': float(derived['P_abs']), 'P_abs_reference': absorbed}
        worst = max(worst, max(scaled.values()))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


# ---- 7 / 8. periodic block: a Bragg stack drawn as one period ----
_BRAGG_INDICES = (1.5, 2.5)                                                    # layers A, B of the unit cell
_BRAGG_CELLS = {'stop_band': (0.26e-6, 0.16e-6), 'pass_band': (0.21e-6, 0.13e-6)}   # (d_A, d_B), 10 nm multiples
_BRAGG_PORTS = ((1.0, 0.30e-6), (2.0, 0.32e-6))                                # (index, drawn length) in / out
_BRAGG_PERIODS = 8
_BRAGG_TOLERANCE = 1e-10


def bragg_stack(cell_lengths, n_periods):
    """(indices, lengths) of the explicit stack: input medium, n_periods x (A, B), output medium."""
    indices = [_BRAGG_PORTS[0][0]] + list(_BRAGG_INDICES) * n_periods + [_BRAGG_PORTS[1][0]]
    lengths = [_BRAGG_PORTS[0][1]] + list(cell_lengths) * n_periods + [_BRAGG_PORTS[1][1]]
    return indices, lengths


def bloch_cosine(cell_lengths, polarization):
    """cos(K period) of the two-layer cell at normal incidence: cos a cos b - (y_A / y_B + y_B / y_A) / 2 sin a sin b,
    a = k0 n_A d_A, b = k0 n_B d_B."""
    (n_a, n_b), (d_a, d_b) = _BRAGG_INDICES, cell_lengths
    ratio = _admittance(polarization, n_a) / _admittance(polarization, n_b)
    a, b = _K0 * n_a * d_a, _K0 * n_b * d_b
    return math.cos(a) * math.cos(b) - 0.5 * (ratio + 1.0 / ratio) * math.sin(a) * math.sin(b)


def _plane_wave_bloch_error(multipliers, forward, backward, power, cell_lengths, polarization):
    """|(lam + 1 / lam) / 2 - cos(K period)| of the one forward Bloch mode carried by cell mode 1 (1 when there is not
    exactly one)."""
    from .cascade import forward_bloch
    flags = forward_bloch(multipliers, forward, backward, power)
    dominant = np.argmax(np.abs(forward) ** 2 + np.abs(backward) ** 2, axis=0)
    plane = np.flatnonzero(flags & (dominant == 0))
    if plane.size != 1:
        return 1.0
    value = complex(multipliers[plane[0]])
    return abs(0.5 * (value + 1.0 / value) - bloch_cosine(cell_lengths, polarization))


def _periodic_errors(cell_lengths, polarization):
    """Errors of the star-power cascade (input | cell^8 | output, periodic lines, every mode kept) vs the transfer
    matrix of the explicit stack, and of the plane-wave Bloch multiplier vs cos(K period)."""
    from .cascade import bloch_modes, repeated_link, total_smatrix, unit_cell
    from .interfaces import interface_smatrix
    media = (_BRAGG_PORTS[0][0], *_BRAGG_INDICES, _BRAGG_PORTS[1][0])
    sets = [_mode_set(_slab_line(np.full(_FILM_LINE_CELLS, index ** 2), _FILM_CELL, ('periodic', 'periodic'),
                                 polarization), 0) for index in media]
    lengths = [_BRAGG_PORTS[0][1], *cell_lengths, _BRAGG_PORTS[1][1]]
    phases = [np.exp(-1j * modes.k * length) for modes, length in zip(sets, lengths)]
    interfaces = [interface_smatrix(left, right) for left, right in zip(sets, sets[1:])]
    cell = unit_cell(phases, interfaces, 1, 2, interface_smatrix(sets[2], sets[1]))
    total = total_smatrix(phases, [interfaces[0], interfaces[1], repeated_link(cell, _BRAGG_PERIODS, interfaces[2])])
    indices, stack_lengths = bragg_stack(cell_lengths, _BRAGG_PERIODS)
    s11, s21 = stack_reference(indices, stack_lengths, polarization)
    s22, s12 = stack_reference(indices[::-1], stack_lengths[::-1], polarization)
    errors = {'s11': abs(total.s11[0, 0] - s11), 's21': abs(total.s21[0, 0] - s21),
              's12': abs(total.s12[0, 0] - s12), 's22': abs(total.s22[0, 0] - s22),
              'bloch_cosine': _plane_wave_bloch_error(*bloch_modes(cell), sets[2].power, cell_lengths, polarization)}
    return {name: float(value) for name, value in errors.items()}


def _check_periodic(results_by_name):
    """Stop- and pass-band cells, ez and hz."""
    details = {f'{band}_{polarization}': _periodic_errors(lengths, polarization)
               for band, lengths in _BRAGG_CELLS.items() for polarization in ('ez', 'hz')}
    worst = max(max(errors.values()) for errors in details.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _BRAGG_TOLERANCE,
            'details': {**details, 'bloch_cosine_reference': {band: bloch_cosine(lengths, 'ez')
                                                              for band, lengths in _BRAGG_CELLS.items()}}}


# ---- 9. bends: the cylindrical line operator ----
_BEND_RADIUS = 3e-6
_BEND_WIDTH = 1e-6                                  # homogeneous annulus between the walls
_BEND_EPS = 4.0
_BEND_CELLS = {'ez': (39, 79), 'hz': (40, 80)}      # coarse / fine (the cell width halves)
_BEND_ORDER_RATIO = 3.0                             # coarse / fine n_eff error at least (second order: 4)
_BEND_FINE_TOLERANCE = 1e-3                         # fine-mesh n_eff error ((k_r dx)^2 / 12 estimate: ~3e-4)
_BEND_IDENTITY_TOLERANCE = 1e-10


def _annulus_line(polarization, n_cells):
    """Bent SlabLine of the homogeneous annulus [R - W/2, R + W/2] (line coordinate r - R, reference 0): ez dirichlet
    ends (u = 0 at the wall one cell beyond the outer centres), hz neumann ends (zero flux on the end faces)."""
    from .mode_sets import bent_line
    lo = -0.5 * _BEND_WIDTH
    if polarization == 'ez':
        width = _BEND_WIDTH / (n_cells + 1)
        centres = lo + width * (np.arange(n_cells) + 1.0)
        ends = ('dirichlet', 'dirichlet')
    else:
        width = _BEND_WIDTH / n_cells
        centres = lo + width * (np.arange(n_cells) + 0.5)
        ends = ('neumann', 'neumann')
    faces = np.concatenate((centres - 0.5 * width, [centres[-1] + 0.5 * width]))
    line = _slab_line(np.full(n_cells, _BEND_EPS), width, ends, polarization)
    return bent_line(line, _BEND_RADIUS, centres, faces, 0.0)[0]


def _bessel_order(polarization, guess):
    """Order nu nearest guess of the annulus mode: ez J_nu(k r1) Y_nu(k r2) = J_nu(k r2) Y_nu(k r1); hz the same with
    J', Y' (k = k0 sqrt(eps), walls r1, r2 = R -+ W / 2)."""
    from scipy.optimize import brentq
    from scipy.special import jv, jvp, yv, yvp
    k = _K0 * math.sqrt(_BEND_EPS)
    r1, r2 = _BEND_RADIUS - 0.5 * _BEND_WIDTH, _BEND_RADIUS + 0.5 * _BEND_WIDTH
    first, second = (jv, yv) if polarization == 'ez' else (jvp, yvp)

    def cross(order):
        return first(order, k * r1) * second(order, k * r2) - first(order, k * r2) * second(order, k * r1)

    grid = np.linspace(guess - 1.0, guess + 1.0, 81)
    values = np.array([cross(order) for order in grid])
    brackets = [index for index in range(grid.size - 1) if values[index] * values[index + 1] < 0.0]
    if not brackets:
        raise ValueError(f'no Bessel cross-product root within 1 of nu = {guess:.6g}')
    nearest = min(brackets, key=lambda index: abs(grid[index] - guess))
    return brentq(cross, grid[nearest], grid[nearest + 1], xtol=1e-14, rtol=1e-15)


def _bend_order_errors(polarization):
    """(coarse, fine) |n_eff_1 - nu / (k0 R)| of the annulus line and the exact n_eff."""
    errors, exact = [], None
    for n_cells in _BEND_CELLS[polarization]:
        modes = _mode_set(_annulus_line(polarization, n_cells), 1)
        n_eff = complex(modes.n_eff[0]).real
        if exact is None:
            exact = _bessel_order(polarization, n_eff * _K0 * _BEND_RADIUS) / (_K0 * _BEND_RADIUS)
        errors.append(abs(n_eff - exact))
    return errors, exact


def _bend_identity_errors(polarization):
    """(symmetry, power) of straight | bent (R, about the line centre) | straight narrow guide, complete bases."""
    from .interfaces import cascade_mode_sets
    from .mode_sets import bent_line, is_propagating
    straight = _step_lines(polarization)[0]
    centres = (np.arange(_STEP_CELLS) + 0.5) * _GUIDE_CELL - 0.5 * _STEP_CELLS * _GUIDE_CELL
    faces = np.arange(_STEP_CELLS + 1) * _GUIDE_CELL - 0.5 * _STEP_CELLS * _GUIDE_CELL
    bent = bent_line(straight, _BEND_RADIUS, centres, faces, 0.0)[0]
    straight_modes = _mode_set(straight, 0)   # both straight sections share one solve
    mode_sets = [straight_modes, _mode_set(bent, 0), straight_modes]
    full = _full_matrix(cascade_mode_sets(mode_sets, [cells * _GUIDE_CELL for cells in _STEP_LENGTH_CELLS]).total)
    symmetry = float(np.max(np.abs(full - full.T)) / np.max(np.abs(full)))
    ports = (mode_sets[0], mode_sets[-1])
    power = np.concatenate([modes.power for modes in ports])
    propagating = np.concatenate([is_propagating(modes) for modes in ports])
    carried = np.sum(np.abs(full[propagating][:, propagating]) ** 2 * power[propagating][:, None], axis=0)
    return symmetry, float(np.max(np.abs(carried / power[propagating] - 1.0)))


def _check_bend(results_by_name):
    """Annulus n_eff vs the Bessel cross-product root (fine-mesh error and second-order convergence), and the exact
    reciprocity / power identities of a complete-basis straight | bent | straight guide; ez and hz."""
    details, worst = {}, 0.0
    for polarization in ('ez', 'hz'):
        (coarse, fine), exact = _bend_order_errors(polarization)
        symmetry, power = _bend_identity_errors(polarization)
        ratio = coarse / fine if fine > 0.0 else math.inf
        scaled = max(fine / _BEND_FINE_TOLERANCE, _BEND_ORDER_RATIO / ratio,
                     symmetry / _BEND_IDENTITY_TOLERANCE, power / _BEND_IDENTITY_TOLERANCE)
        details[polarization] = {'n_eff_exact': exact, 'n_eff_error_coarse': coarse, 'n_eff_error_fine': fine,
                                 'order_ratio': ratio, 'symmetry': symmetry, 'power': power}
        worst = max(worst, scaled)
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


# ---- 5 / 6. em_fdfd_ez comparison: narrow -> wide slab step, pec (5) or cpml (6) transverse faces ----
# FDFD runs ('uniform': the narrow guide throughout, its x-PML floor; 'step': narrow for x < 2 um, wide beyond): a
# one-sided mode source (mode 1, 1 W/m) at x cell 40, mode monitors upstream (cell 20, mode 1) and downstream (cell
# 120 mode 1, cell 140 mode 3), all lines spanning the whole y window. The CPU reference is the EME interface of the
# same y lines (the FDFD / PortLine operator: pec -> cells 1..58 between dirichlet ends; cpml -> the whole padded line,
# neumann ends, the CPML stretch of cpml_axis_profiles) with discrete propagation (D = 25 nm) and every mode kept, which
# solves the FDFD lattice of the sectioned guide; an em_eme model run (continuum, n_modes 0) checks the model path
# against the CPU continuum interface (the same lines).
_STEP_CELL = 25e-9
_STEP_SIZE = (4e-6, 1.5e-6)
_STEP_CORES = {'narrow': (0.6e-6, 0.9e-6), 'wide': (0.4e-6, 1.1e-6)}   # y ranges (12 and 28 cells)
_STEP_X = 2e-6
_STEP_EPS = (2.25, 4.0)                                                 # cladding, core (float32-exact)
_STEP_PML = {'pml_cells': 20, 'pml_sigma_factor': 0.7}
_STEP_MONITORS = (('upstream', 0.5125e-6, 1), ('down_1', 3.0125e-6, 1), ('down_3', 3.5125e-6, 3))
_STEP_SOURCE_X = 1.0125e-6
_STEP_FDFD_PHYSICS = 'em'
_STEP_FLOOR_FACTOR = 10.0          # FDFD error allowance in units of its x-PML floor (the uniform run)
_STEP_PEC_MIN_TOLERANCE = 1e-5     # absolute floor of the pec allowance
_STEP_CPML_TOLERANCE = 1e-3        # absolute allowance of the radiating (cpml) step, PML modes kept
_STEP_MODEL_TOLERANCE = 1e-6       # em_eme model run vs the CPU continuum interface (float32 cell widths)


def _step_geometry(step):
    """Core rectangles 'input' (narrow, x < 2 um) and 'output' (wide when step, else narrow)."""
    size_x = _STEP_SIZE[0]
    output_core = _STEP_CORES['wide' if step else 'narrow']
    return [GeometrySpec('input', 'rectangle', {'x0': 0.0, 'y0': _STEP_CORES['narrow'][0], 'x1': _STEP_X,
                                                'y1': _STEP_CORES['narrow'][1]}),
            GeometrySpec('output', 'rectangle', {'x0': _STEP_X, 'y0': output_core[0], 'x1': size_x,
                                                 'y1': output_core[1]})]


def _step_materials():
    return [MaterialSpec('cladding', properties={'eps_r': _STEP_EPS[0]}, selection=['all']),
            MaterialSpec('core', properties={'eps_r': _STEP_EPS[1]}, selection=['input', 'output'])]


def _fdfd_step_spec(transverse, step, polarization='ez'):
    """em_fdfd_<polarization> slab guide (stationary, 1.55 um), cpml x faces, `transverse` y faces, mode source and
    monitors."""
    geometry, features = _step_geometry(step), []
    lines = (('source', _STEP_SOURCE_X, None),) + _STEP_MONITORS
    for name, x, mode_number in lines:
        geometry.append(GeometrySpec(f'{name}_line', 'line', {'x0': x, 'y0': 0.0, 'x1': x, 'y1': _STEP_SIZE[1]}))
        if mode_number is None:
            features.append(FeatureSpec(name, 'mode_source', [f'{name}_line'],
                                        {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}))
        else:
            features.append(FeatureSpec(name, 'mode_monitor', [f'{name}_line'], {'mode_number': mode_number}))
    settings = {'wavelength': _WAVELENGTH, 'bc_y_lo': transverse, 'bc_y_hi': transverse, **_STEP_PML}
    return ModelSpec(
        title=f"em_fdfd_{polarization} slab {'step' if step else 'guide'} ({transverse} y faces)",
        domain=DomainSpec(size_x=_STEP_SIZE[0], size_y=_STEP_SIZE[1]),
        geometry=geometry, materials=_step_materials(),
        physics=[PhysicsSpec(_STEP_FDFD_PHYSICS, f'em_fdfd_{polarization}', settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_STEP_CELL),
        study=StudySpec(type='stationary'))


def _eme_step_spec(transverse, polarization='ez'):
    """em_eme of the step (n_modes 0, pec x port faces, `transverse` y faces), eme_ports with 4 port modes."""
    settings = {'polarization': polarization, 'propagation_axis': 'x', 'wavelength': _WAVELENGTH, 'n_modes': 0,
                'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': transverse, 'bc_y_hi': transverse, **_STEP_PML}
    return ModelSpec(
        title=f'em_eme slab step ({polarization}, {transverse} y faces)',
        domain=DomainSpec(size_x=_STEP_SIZE[0], size_y=_STEP_SIZE[1]),
        geometry=_step_geometry(True), materials=_step_materials(),
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec('ports', 'eme_ports', params={'n_port_modes': 4})])],
        mesh=MeshSpec(mode='uniform', dx=_STEP_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T1', f'{_PHYSICS}_transmission_1'),
                        DerivedValueSpec('T3', f'{_PHYSICS}_transmission_3')])


def _on_backend(spec, backend):
    spec.solver_defaults.backend = backend
    return spec


def _step_specs(transverse, polarization='ez'):
    """FDFD reference runs on the cpu (host splu of ~12k-34k unknowns beats the GPU setup; same operator, see
    em_fdfd_cpu_gpu), the em_eme run (the path under test) on the default backend."""
    return {'uniform': _on_backend(_fdfd_step_spec(transverse, False, polarization), 'cpu'),
            'step': _on_backend(_fdfd_step_spec(transverse, True, polarization), 'cpu'),
            'eme': _eme_step_spec(transverse, polarization)}


def _step_line(core, transverse, polarization='ez'):
    """SlabLine (mu 1) of the y line the FDFD ports and em_eme solve for a core y range: pec -> the 58 window cells
    between the pinned wall cells (ez dirichlet, hz neumann ends: PortLine _WALL_ENDS); cpml -> every grid cell
    (pml_cells padding each side, neumann ends) with the CPML stretch at omega."""
    n_window = int(round(_STEP_SIZE[1] / _STEP_CELL))
    first, last = (int(round(edge / _STEP_CELL)) for edge in core)
    if transverse == 'pec':
        eps = np.full(n_window, _STEP_EPS[0])
        eps[first:last] = _STEP_EPS[1]
        end = 'dirichlet' if polarization == 'ez' else 'neumann'
        return _slab_line(eps[1:-1], _STEP_CELL, (end, end), polarization)
    from ..em.boundaries import BoundaryConfig, cpml_axis_profiles
    from ..em.operators import pml_stretch
    padding = _STEP_PML['pml_cells']
    n_cells = n_window + 2 * padding
    eps = np.full(n_cells, _STEP_EPS[0])
    eps[padding + first:padding + last] = _STEP_EPS[1]
    config = BoundaryConfig('pec', 'pec', 'cpml', 'cpml')
    config.cpml_npml = padding
    config.cpml_sigma_factor = _STEP_PML['pml_sigma_factor']
    profile = cpml_axis_profiles(config, n_cells, 'y', _STEP_CELL)
    face_index = np.clip(np.arange(-1, n_cells), 0, n_cells - 1)   # face k = hi face of cell k - 1 (PortLine)
    cell_stretch = pml_stretch(profile.sigma_e, profile.kappa_e, profile.alpha_e, _OMEGA)
    face_stretch = pml_stretch(profile.sigma_h[face_index], profile.kappa_h[face_index], profile.alpha_h[face_index],
                               _OMEGA)
    return _slab_line(eps, _STEP_CELL, ('neumann', 'neumann'), polarization, stretch=(cell_stretch, face_stretch))


def _step_cpu_fractions(transverse, discrete_spacing, drop_pml_modes=False, polarization='ez'):
    """{'T1', 'T3', 'R1'} power fractions of the narrow | wide interface (every mode of each line, or the non-PML ones
    when drop_pml_modes), mode 1 incident from the narrow side."""
    from .interfaces import cascade_mode_sets
    mode_sets = [_mode_set(_step_line(_STEP_CORES[name], transverse, polarization), 0, discrete_spacing,
                           drop_pml_modes) for name in ('narrow', 'wide')]
    total = cascade_mode_sets(mode_sets, [0.0, 0.0]).total
    narrow, wide = mode_sets
    return {'T1': float(abs(total.s21[0, 0]) ** 2 * wide.power[0] / narrow.power[0]),
            'T3': float(abs(total.s21[2, 0]) ** 2 * wide.power[2] / narrow.power[0]),
            'R1': float(abs(total.s11[0, 0]) ** 2)}


def _fdfd_fraction(results, feature, series):
    """First (only) value of a source-normalised FDFD mode-monitor series."""
    result = results.result(_STEP_FDFD_PHYSICS, feature)
    if not result.get('normalized'):
        raise ValueError(f"mode monitor '{feature}' has no source-normalised fractions")
    return float(np.asarray(result['series'][series], dtype=np.float64)[0])


def _fdfd_step_fractions(results):
    return {'T1': _fdfd_fraction(results, 'down_1', 'fraction_positive'),
            'T3': _fdfd_fraction(results, 'down_3', 'fraction_positive'),
            'R1': _fdfd_fraction(results, 'upstream', 'fraction_negative')}


def _eme_model_fractions(results):
    """T1, T3 (scalars) and R1 (the port table's input-face mode 1 row) of the em_eme run."""
    derived = results.summary()['derived']
    rows = results.result(_PHYSICS, 'ports')['rows']
    return {'T1': float(derived['T1']), 'T3': float(derived['T3']), 'R1': float(rows[0][6])}


def _step_comparison(results_by_name, transverse, polarization='ez'):
    """(errors in units of their allowances, details): CPU discrete EME vs FDFD (allowance from the x-PML floor),
    em_eme model run vs the CPU continuum interface (skipped without a usable GPU: api.selftest drops that run)."""
    floor_up = _fdfd_fraction(results_by_name['uniform'], 'upstream', 'fraction_negative')
    floor_down = abs(_fdfd_fraction(results_by_name['uniform'], 'down_1', 'fraction_positive') - 1.0)
    floor = max(floor_up, floor_down)
    minimum = _STEP_PEC_MIN_TOLERANCE if transverse == 'pec' else _STEP_CPML_TOLERANCE
    allowance = max(_STEP_FLOOR_FACTOR * floor, minimum)
    fdfd = _fdfd_step_fractions(results_by_name['step'])
    discrete = _step_cpu_fractions(transverse, _STEP_CELL, polarization=polarization)
    continuum = _step_cpu_fractions(transverse, None, polarization=polarization)
    fdfd_error = max(abs(discrete[name] - fdfd[name]) for name in fdfd)
    if 'eme' in results_by_name:
        model = _eme_model_fractions(results_by_name['eme'])
        model_error = max(abs(model[name] - continuum[name]) for name in model)
    else:
        model, model_error = 'skipped (no usable GPU)', 0.0
    details = {'fdfd': fdfd, 'eme_discrete_cpu': discrete, 'eme_continuum_cpu': continuum, 'eme_model': model,
               'fdfd_pml_floor': floor, 'fdfd_allowance': allowance, 'fdfd_error': fdfd_error,
               'model_error': model_error,
               'continuum_vs_fdfd_recorded': max(abs(continuum[name] - fdfd[name]) for name in fdfd)}
    return max(fdfd_error / allowance, model_error / _STEP_MODEL_TOLERANCE), details


def _check_step_pec(results_by_name):
    """pec y faces: CPU discrete EME (complete basis) vs em_fdfd_ez within 10 x its x-PML floor; model path."""
    error, details = _step_comparison(results_by_name, 'pec')
    return {'measured': details['eme_discrete_cpu']['T1'], 'expected': details['fdfd']['T1'], 'error': error,
            'tolerance': 1.0, 'details': details}


def _check_step_pec_hz(results_by_name):
    """The same with H_z: em_fdfd_hz (its x-face eps = the A / B mean, which the discrete matching reproduces)."""
    error, details = _step_comparison(results_by_name, 'pec', 'hz')
    return {'measured': details['eme_discrete_cpu']['T1'], 'expected': details['fdfd']['T1'], 'error': error,
            'tolerance': 1.0, 'details': details}


def _check_step_cpml(results_by_name):
    """cpml y faces: the same with PML modes kept (1e-3 or 10 x the floor); the PML-dropped basis recorded."""
    error, details = _step_comparison(results_by_name, 'cpml')
    dropped = _step_cpu_fractions('cpml', _STEP_CELL, drop_pml_modes=True)
    details['eme_discrete_pml_dropped_recorded'] = dropped
    details['pml_dropped_vs_fdfd_recorded'] = max(abs(dropped[name] - details['fdfd'][name]) for name in dropped)
    return {'measured': details['eme_discrete_cpu']['T1'], 'expected': details['fdfd']['T1'], 'error': error,
            'tolerance': 1.0, 'details': details}


def _step_text(polarization):
    """Shared description of the em_fdfd_<polarization> step comparison cases."""
    field = 'E_z' if polarization == 'ez' else 'H_z'
    return (f'a slab step at 1.55 um, {field} (cladding eps 2.25, core eps 4: 0.3 um wide for x < 2 um, 0.7 um '
            f'beyond, 1.5 um high window, {_STEP_CELL * 1e9:g} nm cells). em_fdfd_{polarization} runs on the cpu '
            f'backend (stationary, cpml x faces, pml_cells {_STEP_PML["pml_cells"]}, sigma factor '
            f'{_STEP_PML["pml_sigma_factor"]:g}): '
            'the guide uniform (narrow throughout) for the x-PML floor = max(|1 - downstream mode-1 fraction|, '
            'upstream reflection), then the step with a one-sided mode source (mode 1) and mode monitors for the '
            'transmission into wide modes 1 and 3 and the mode-1 reflection. Reference: the EME interface of the '
            'same y lines on the CPU (the FDFD / PortLine line operator) with discrete propagation along x and every '
            'line mode kept, which solves the FDFD lattice of the sectioned guide exactly (hz: matching the face '
            'mean and each side\'s flux is the em_fdfd_hz x-face eps = the mean of the two cells); plus an em_eme '
            'model run (continuum, n_modes 0) whose transmission_1 / _3 and port-table reflection must equal the CPU '
            f'continuum interface within {_STEP_MODEL_TOLERANCE:g} (float32 cell widths; default backend, skipped '
            'without a usable GPU). error = the largest deviation in units of its allowance, tolerance 1. Not yet '
            'run with the FDFD runs on the cpu backend.')


EME_SELFTEST_CASES = (
    SelfTestCase(
        name='eme_complete_basis_identities',
        label='EME: complete-basis width step',
        description='CPU check of physics/eme: narrow (0.2 um) | wide (0.6 um) | narrow guide (eps 4 in 2.25, 60 '
                    'cells of 25 nm between dirichlet ends, sections 12, 32, 12 cells, every line mode kept). With a '
                    'complete basis the mode matching is exact, so the bilinear-normalised port S-matrix is symmetric '
                    '(reciprocity) and, lossless, conserves power over the propagating modes (sum |S_ij|^2 p_i / p_j = '
                    '1); continuum and discrete propagation, ez and hz. The discrete-propagation cascade also solves '
                    'the 2D Yee lattice equations (W+ (u_(i+1) - u_i) - W- (u_i - u_(i-1))) / D^2 + A_line u_i / '
                    'cell_weight = 0 of the sectioned guide (ez W = 1; hz W = 1 / the mean eps of the two cells, the '
                    'em_fdfd_hz face rule) at every interior cell, including the two columns beside each interface '
                    '(relative residual). '
                    'The length-sweep re-cascade (cascade.total_smatrix, middle section x 1.7) must equal a fresh '
                    'cascade of the stretched lengths. error = the largest of these; tolerance 1e-10 (the '
                    'evanescent-mode conditioning of I + P P^T costs a few digits).',
        specs=lambda: {},
        check=_check_complete_basis,
        max_wall_time=60.0),
    SelfTestCase(
        name='eme_lossy_film_absorption',
        label='EME: thin-film stack model run with a lossy film',
        description='em_eme along x through a thin-film stack (index 1.0 | 1.5 | 2.5 | 1.5 | 2.5 | 2.0, lengths 0.30, '
                    '0.21, 0.13, 0.21, 0.13, 0.32 um, 1.55 um) drawn as rectangles in a y-periodic strip '
                    f'(16 cells of {_MODEL_CELL * 1e9:g} nm, pec x faces), stationary study, ez and hz runs, with its '
                    f'first index-2.5 layer (0.13 um) lossy: sigma_e = {_LOSSY_SIGMA:g} S/m (float32-exact; eps = '
                    '6.25 - 0.38 j at 1.55 um; every other eps exact in float32). The sections from the mesh must be '
                    'the six layers with their lengths, the eme_ports S-matrix extras (S11, S21, S12, S22 of mode 1) '
                    'the transfer matrix with the complex layer index n = sqrt(eps - j sigma / (omega eps0)), and the '
                    'scalar outputs transmission_1 / reflection |S21|^2 / |S11|^2 (lossless port media; the '
                    f'evanescent modes carry no power), all within {_MODEL_TOLERANCE:g} (round-off of the float32 '
                    'materials path); the integral of absorbed_power over the layer (q = 1/2 omega eps0 (-Im eps) '
                    'times the 3-point Gauss cell mean of |E|^2 along x) must equal the absorbed fraction 1 - '
                    '|S11|^2 - |S21|^2 of the 1 W/m input (Poynting: the lossless ports carry the rest) within '
                    f'{_LOSSY_TOLERANCE:g} (float32 q storage; the Gauss rule is exact to (k dx)^6). error = the '
                    'largest item in its own tolerance units, tolerance 1. Not yet run with the merged lossless-stack '
                    f'items (S entries at {_MODEL_TOLERANCE:g} on the lossy stack).',
        specs=lambda: {'ez': _film_model_spec('ez', lossy=True), 'hz': _film_model_spec('hz', lossy=True)},
        check=_check_lossy_film),
    SelfTestCase(
        name='eme_periodic_bragg_stack',
        label='EME: periodic block of a Bragg stack',
        description='CPU check of the periodic block (cascade.unit_cell / repeated_link / bloch_modes): transverse-'
                    'periodic uniform sections (6 cells of 50 nm, every mode kept) of index 1.0 | (1.5, 2.5) x 8 | '
                    '2.0 at 1.55 um, ez and hz, the cell (A, B) once with the junction B -> A raised to the 7th star '
                    'power: S11, S21, S12, S22 of mode 1 must equal the transfer matrix of the explicit 18-layer '
                    'stack, and the forward Bloch multiplier lam of the plane wave (cell mode 1) must give (lam + '
                    '1/lam) / 2 = cos(K period) = cos a cos b - (y_A / y_B + y_B / y_A) / 2 sin a sin b (a = k0 n_A '
                    'd_A, b = k0 n_B d_B). A stop-band cell (d = 0.26, 0.16 um: |cos| > 1, |lam| < 1) and a '
                    f'pass-band cell (0.21, 0.13 um). tolerance {_BRAGG_TOLERANCE:g} (round-off). The ez model run of '
                    'the periodic block is examples/em/analytic_eme_bragg_periodic_block.json.',
        specs=lambda: {},
        check=_check_periodic,
        max_wall_time=60.0),
    SelfTestCase(
        name='eme_bend_identities',
        label='EME: bent sections',
        description='CPU check of the bend map (mode_sets.bent_line, the exact cylindrical line operator): (1) a '
                    f'homogeneous annulus (eps {_BEND_EPS:g}, r from {(_BEND_RADIUS - 0.5 * _BEND_WIDTH) * 1e6:g} to '
                    f'{(_BEND_RADIUS + 0.5 * _BEND_WIDTH) * 1e6:g} um, R = {_BEND_RADIUS * 1e6:g} um, 1.55 um): n_eff '
                    'of mode 1 vs nu / (k0 R) of the Bessel cross-product root (ez: J_nu(k r1) Y_nu(k r2) = J_nu(k '
                    "r2) Y_nu(k r1), dirichlet walls; hz: the same with J', Y', neumann walls) on two meshes: the "
                    f'fine error must be below {_BEND_FINE_TOLERANCE:g} and the coarse / fine ratio at least '
                    f'{_BEND_ORDER_RATIO:g} (second order: 4); (2) the narrow guide of eme_complete_basis_identities '
                    f'as straight | bent (R {_BEND_RADIUS * 1e6:g} um about its centre) | straight with complete bases: '
                    'S = S^T and power conserved over the propagating modes to 1e-10 (exact: the bent line is '
                    'bilinear-symmetric with weight w R / r). error = the largest in units of its allowance, '
                    'tolerance 1.',
        specs=lambda: {},
        check=_check_bend,
        max_wall_time=60.0),
    SelfTestCase(
        name='eme_vs_fdfd_step_ez',
        label='EME vs FDFD: slab step between pec walls',
        description=f"Pec y faces: {_step_text('ez')} The CPU discrete EME must match the FDFD fractions within 10 x "
                    f'the FDFD x-PML floor (at least {_STEP_PEC_MIN_TOLERANCE:g}); the continuum-vs-FDFD gap (the '
                    'discretisation along x) is recorded.',
        specs=lambda: _step_specs('pec'),
        check=_check_step_pec),
    SelfTestCase(
        name='eme_vs_fdfd_step_hz',
        label='EME vs FDFD: slab step between pec walls (H_z)',
        description=f"Pec y faces: {_step_text('hz')} The CPU discrete EME must match the FDFD fractions within 10 x "
                    f'the FDFD x-PML floor (at least {_STEP_PEC_MIN_TOLERANCE:g}); the continuum-vs-FDFD gap is '
                    'recorded.',
        specs=lambda: _step_specs('pec', 'hz'),
        check=_check_step_pec_hz),
    SelfTestCase(
        name='eme_radiating_step_pml_modes',
        label='EME vs FDFD: radiating step with PML modes',
        description=f"Cpml y faces (the step radiates into the transverse PML): {_step_text('ez')} With the PML modes "
                    f'kept '
                    f'the CPU discrete EME must match the FDFD fractions within {_STEP_CPML_TOLERANCE:g} (or 10 x the '
                    'x-PML floor if larger); the same interface with the PML modes dropped (the mode-port basis) is '
                    'recorded, not asserted, to show why EME keeps them.',
        specs=lambda: _step_specs('cpml'),
        check=_check_step_cpml),
)


# ---- template: linear taper ----
LINEAR_TAPER_LABEL = 'Linear taper (EME)'
LINEAR_TAPER_DESCRIPTION = (
    'EME, stationary: a 10 um linear taper from a 0.5 um to a 2 um wide guide (eps 4 in silica, ez, 1.55 um, 25 nm '
    'mesh, cpml on y, pec x faces); eme_ports with the fundamental incident from x_lo, the section table, derived '
    'T1 = transmission into the fundamental of the wide guide, and eme_length_sweep: T1 versus the taper length from '
    '2.5 to 40 um, re-cascaded from the one solve.')
_TAPER_STRETCH_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
_TAPER_SIZE = (14e-6, 5e-6)
_TAPER_CENTRE = 2.5e-6
_TAPER_X = (2e-6, 12e-6)
_TAPER_WIDTHS = (0.5e-6, 2e-6)


def linear_taper_eme_spec():
    """Template 'linear_taper_eme' (em_eme along x, stationary): taper transmission of the fundamental mode."""
    (x0, x1), (narrow, wide), centre = _TAPER_X, _TAPER_WIDTHS, _TAPER_CENTRE

    def guide(name, x_start, x_stop, width):
        return GeometrySpec(name, 'rectangle', {'x0': x_start, 'y0': centre - 0.5 * width, 'x1': x_stop,
                                                'y1': centre + 0.5 * width})

    taper = GeometrySpec('taper', 'polygon', {'points': [[x0, centre - 0.5 * narrow], [x1, centre - 0.5 * wide],
                                                         [x1, centre + 0.5 * wide], [x0, centre + 0.5 * narrow]]})
    return ModelSpec(
        title=LINEAR_TAPER_LABEL,
        domain=DomainSpec(size_x=_TAPER_SIZE[0], size_y=_TAPER_SIZE[1]),
        geometry=[guide('input_guide', 0.0, x0, narrow), taper, guide('output_guide', x1, _TAPER_SIZE[0], wide)],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('core', properties={'eps_r': 4.0},
                                selection=['input_guide', 'taper', 'output_guide'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings={'polarization': 'ez', 'propagation_axis': 'x',
                                                        'wavelength': _WAVELENGTH, 'n_modes': 20,
                                                        'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'pml_sigma_factor': 0.7},
                             features=[FeatureSpec('ports', 'eme_ports'),
                                       FeatureSpec('sections', 'eme_section_table'),
                                       FeatureSpec('length_scan', 'eme_length_sweep',
                                                   params={'start': x0, 'stop': x1,
                                                           'stretch_factors': list(_TAPER_STRETCH_FACTORS)})])],
        mesh=MeshSpec(mode='uniform', dx=25e-9),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T1', f'{_PHYSICS}_transmission_1'),
                        DerivedValueSpec('R', f'{_PHYSICS}_reflection')])


# ---- template: 1 x 2 MMI splitter ----
MMI_LABEL = '1 x 2 MMI splitter (EME)'
MMI_DESCRIPTION = (
    'EME, stationary, ez at 1.55 um: a centre-fed 1 x 2 multimode interference splitter (eps 4 in silica, 25 nm mesh, '
    'cpml on y, pec x faces): a 1 um input guide, a 3 um wide MMI section of length L_mmi (parameter, 7.3 um = '
    '3 L_pi / 8 for the two-fold symmetric image, L_pi = 4 n_r W_e^2 / (3 lambda) with the effective width W_e ~ 3.36 '
    'um) and two 1 um output guides centred at +-0.75 um. The output port modes are the supermodes of the guide pair, '
    'so T1 (symmetric supermode) is the power split equally into both outputs and the antisymmetric one stays ~0 by '
    'symmetry. Beat-length check: the section table lists n_eff_1 / n_eff_2 of the MMI section (its two lowest, even '
    'and odd modes), and L_pi = lambda / (2 (n_eff_1 - n_eff_2)) of those modes is exact for the EME, which '
    'propagates every section mode with its own beta; set L_mmi = 3 L_pi / 8 from it to tune the splitter.')
_MMI_WIDTH = 3e-6
_MMI_GUIDE_WIDTH = 1e-6
_MMI_OUTPUT_OFFSET = 0.75e-6
_MMI_PORT_LENGTH = 3e-6
_MMI_HEIGHT = 6e-6
_MMI_LENGTH = 7.3e-6


def mmi_1x2_eme_spec():
    """Template 'mmi_1x2_eme' (em_eme along x, stationary): power split of a centre-fed 1 x 2 MMI."""
    centre, port = 0.5 * _MMI_HEIGHT, _MMI_PORT_LENGTH
    mmi_end = f'{port!r} + L_mmi'

    def strip(name, x0, x1, y_centre, width):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y_centre - 0.5 * width, 'x1': x1,
                                                'y1': y_centre + 0.5 * width})

    size_x = f'2 * {port!r} + L_mmi'
    guides = [strip('input_guide', 0.0, port, centre, _MMI_GUIDE_WIDTH),
              strip('mmi', port, mmi_end, centre, _MMI_WIDTH),
              strip('output_upper', mmi_end, size_x, centre + _MMI_OUTPUT_OFFSET, _MMI_GUIDE_WIDTH),
              strip('output_lower', mmi_end, size_x, centre - _MMI_OUTPUT_OFFSET, _MMI_GUIDE_WIDTH)]
    return ModelSpec(
        title=MMI_LABEL,
        parameters=[ParameterSpec('L_mmi', _MMI_LENGTH, 'MMI section length [m]: 3 L_pi / 8')],
        domain=DomainSpec(size_x=size_x, size_y=_MMI_HEIGHT),
        geometry=guides,
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('core', properties={'eps_r': 4.0}, selection=[guide.name for guide in guides])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings={'polarization': 'ez', 'propagation_axis': 'x',
                                                        'wavelength': _WAVELENGTH, 'n_modes': 40,
                                                        'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'pml_sigma_factor': 0.7},
                             features=[FeatureSpec('ports', 'eme_ports', params={'n_port_modes': 2}),
                                       FeatureSpec('sections', 'eme_section_table')])],
        mesh=MeshSpec(mode='uniform', dx=25e-9),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T1', f'{_PHYSICS}_transmission_1'),
                        DerivedValueSpec('T2', f'{_PHYSICS}_transmission_2'),
                        DerivedValueSpec('R', f'{_PHYSICS}_reflection')])


EME_TEMPLATES = (
    ('linear_taper_eme', LINEAR_TAPER_LABEL, LINEAR_TAPER_DESCRIPTION, linear_taper_eme_spec,
     'Light passes a 10 micrometre long taper from a narrow to a wide waveguide, computed by eigenmode expansion: '
     'how much stays in the fundamental mode.'),
    ('mmi_1x2_eme', MMI_LABEL, MMI_DESCRIPTION, mmi_1x2_eme_spec,
     'A multimode interference splitter divides the light of one waveguide equally into two, computed by eigenmode '
     'expansion: the split power and how the length sets it.'),
)

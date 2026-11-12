"""Self-tests of em_eme_3d (eme_plan stage 2): CPU checks of the cross-section mode sets, interfaces and cascade with
exact references (identity, reduction to the 2D slab EME with walls, PEC cells or a bend), model runs of the anisotropic
thin-film stack and of PEC cells on cuDSS, and the 'thermo_optic_phase_shifter_eme_3d' template."""
import cmath
import functools

import numpy as np

from ...core.constants import EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, SolverSpec, StudySpec)
from .selftests import (_LOSSY_LAYER, _LOSSY_SIGMA, _OMEGA, _PHYSICS, _WAVELENGTH, _mode_set, _slab_line,
                        lossy_film_details)

_TYPE_3D = 'em_eme_3d'   # = definition3d.EM_EME_3D.name (definition3d imports this module through definition)
_LOSSY_3D_TOLERANCE = 1e-6
_LENGTH_CELLS = (16, 30, 52)


def _boundary(face_type, npml=6):
    """2D BoundaryConfig of the cross-section with `face_type` on all four faces."""
    from ..em.boundaries import BoundaryConfig
    boundary = BoundaryConfig(face_type, face_type, face_type, face_type)
    boundary.cpml_npml = npml
    return boundary


def _cross_section_set(eps_xx, eps_yy, eps_zz, cell_widths, boundary, n_modes, pec=None, bend=None):
    """EME ModeSet (PML modes kept) of a cross-section of relative eps arrays (nx, ny), mu 1, PEC cells pec (bool or
    None), bend = None or the cross_section_modes bend tuple."""
    from ..em.cross_section_modes import CrossSectionMaterials, solve_cross_section_modes
    from .mode_sets import cross_section_mode_set, stretched_areas
    shape = np.shape(eps_zz)
    materials = CrossSectionMaterials(eps_xx=np.asarray(eps_xx, dtype=np.complex128),
                                      eps_yy=np.asarray(eps_yy, dtype=np.complex128),
                                      eps_zz=np.asarray(eps_zz, dtype=np.complex128),
                                      mu=np.ones(shape, dtype=np.complex128),
                                      pec=np.zeros(shape, dtype=bool) if pec is None else np.asarray(pec, dtype=bool))
    pml_dx = max(float(max(widths[0], widths[-1])) for widths in cell_widths)
    solve = solve_cross_section_modes(cell_widths, boundary, pml_dx, materials, _OMEGA, n_modes, bend=bend,
                                      drop_pml_modes=False)
    return cross_section_mode_set(solve.modes, stretched_areas(cell_widths, boundary, pml_dx, _OMEGA),
                                  (bool(boundary.periodic_x), bool(boundary.periodic_y)))


# ---- 1. uniform guide cut into three sections ----
_BOX_CELLS = 24
_BOX_CELL = 40e-9
_BOX_CORE = ((7, 17), (9, 15))     # x and y core cells (0.4 x 0.24 um)
_BOX_EPS = (2.1, 12.0)             # cladding, core
_BOX_MODES = 12
_IDENTITY_TOLERANCE = 1e-9


def _box_mode_set(face_type):
    eps = np.full((_BOX_CELLS, _BOX_CELLS), _BOX_EPS[0])
    eps[_BOX_CORE[0][0]:_BOX_CORE[0][1], _BOX_CORE[1][0]:_BOX_CORE[1][1]] = _BOX_EPS[1]
    widths = (np.full(_BOX_CELLS, _BOX_CELL), np.full(_BOX_CELLS, _BOX_CELL))
    return _cross_section_set(eps, eps, eps, widths, _boundary(face_type), _BOX_MODES)


def _identity_error(modes):
    """max |S - S_ideal| of three sections sharing one mode set, and of the internal amplitudes of the middle one."""
    from .interfaces import cascade_mode_sets
    lengths = [cells * _BOX_CELL for cells in _LENGTH_CELLS]
    cascade = cascade_mode_sets([modes] * 3, lengths)
    ideal = np.diag(np.exp(-1j * modes.k * sum(lengths)))
    total = cascade.total
    errors = [np.max(np.abs(total.s21 - ideal)), np.max(np.abs(total.s12 - ideal)), np.max(np.abs(total.s11)),
              np.max(np.abs(total.s22))]
    incident = np.zeros(modes.n_modes, dtype=np.complex128)
    incident[0] = 1.0
    forward, backward = cascade.section_amplitudes(1, 'lo', incident)
    errors += [abs(forward[0] - cmath.exp(-1j * modes.k[0] * lengths[0])), float(np.max(np.abs(forward[1:]))),
               float(np.max(np.abs(backward)))]
    return float(max(errors))


def _check_uniform_identity_3d(results_by_name):
    """Periodic and pec boxes asserted (bilinear orthonormality of the cross-section modes, degenerate ones
    recombined); the cpml box recorded."""
    errors = {face_type: _identity_error(_box_mode_set(face_type)) for face_type in ('periodic', 'pec')}
    cpml = _box_mode_set('cpml')
    measured = max(errors.values())
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': _IDENTITY_TOLERANCE,
            'details': {'errors': errors, 'cpml_box_error_recorded': _identity_error(cpml),
                        'cpml_box_pml_modes': int(np.count_nonzero(cpml.pml))}}


# ---- 2. anisotropic thin-film stack (the model run of section 4) ----
_FILM_INDICES = (1.25, 1.5, 2.5, 1.5, 2.5, 2.0)            # sqrt(eps_xx): input medium, four layers, output medium
_FILM_LENGTHS = (0.30e-6, 0.21e-6, 0.13e-6, 0.21e-6, 0.13e-6, 0.32e-6)
_FILM_YY_RATIO = 0.9           # eps_yy / eps_xx: the x-polarised plane wave is the non-degenerate mode 1
_FILM_CELLS = 4
_FILM_MODES = 10


# ---- 3. y-invariant step: the TE block equals the 2D ez EME ----
_SLAB_CELLS = 60
_SLAB_CELL = 25e-9
_SLAB_CORES = ((26, 34), (18, 42))   # narrow, wide core cells along x
_SLAB_EPS = (2.25, 4.0)
_SLAB_MODES = 40
_SLAB_TOLERANCE = 1e-8


def _te_rows(modes):
    """Indices of the TE modes (E_y carries most of the transverse E at the centres) of a y-invariant cross-section."""
    energy = np.sum(np.abs(modes.profiles) ** 2, axis=2)
    return np.flatnonzero(energy[:, 1] > energy[:, 0] + energy[:, 2])


_SLAB_PEC_COLUMNS = 2               # pec_cells variant: PEC cell columns at each x end (periodic x)
_SLAB_BEND_RADIUS = 3e-6            # bend variant: the wide section bent about the x centre


_SLAB_LENGTH_CELLS = (12, 32, 12)   # narrow | wide | narrow section lengths in cells


def _reduction_sets(variant, middle_loss):
    """([(3D ModeSet, 2D ModeSet, 2D SlabLine, 2D bend face scale or None, TE rows of the 3D set)] per section, 2D line
    cells, PEC mask or None) of the y-invariant narrow | wide | narrow step; middle_loss = -Im eps of the wide section."""
    from .mode_sets import bent_line
    n_cells = _SLAB_CELLS
    widths = (np.full(n_cells, _SLAB_CELL), np.full(1, _SLAB_CELL))
    boundary = _boundary('periodic' if variant == 'pec_cells' else 'pec')
    boundary.y_lo = boundary.y_hi = 'periodic'
    edges = np.arange(n_cells + 1) * _SLAB_CELL
    reference = 0.5 * edges[-1]
    pec = None
    if variant == 'pec_cells':
        pec = np.zeros((n_cells, 1), dtype=bool)
        pec[:_SLAB_PEC_COLUMNS] = pec[n_cells - _SLAB_PEC_COLUMNS:] = True
        cells, ends = slice(_SLAB_PEC_COLUMNS, n_cells - _SLAB_PEC_COLUMNS), ('dirichlet', 'dirichlet')
    else:
        cells, ends = slice(1, n_cells - 1), ('odd', 'odd')
    solved = []   # narrow, wide (the two narrow sections share one solve, as the solver cache does)
    for number, core in enumerate(_SLAB_CORES):
        eps = np.full(n_cells, _SLAB_EPS[0], dtype=np.complex128)
        eps[core[0]:core[1]] = _SLAB_EPS[1]
        if number == 1:
            eps -= 1j * middle_loss
        bent = variant == 'bend' and number == 1
        bend_map = (_SLAB_BEND_RADIUS, reference, edges, 'exact') if bent else None
        modes_3d = _cross_section_set(eps[:, None], eps[:, None], eps[:, None], widths, boundary, _SLAB_MODES,
                                      pec=pec, bend=bend_map)
        te = _te_rows(modes_3d)
        line, face_scale = _slab_line(eps[cells], _SLAB_CELL, ends), None
        if bent:
            centres = 0.5 * (edges[:-1] + edges[1:])
            line, face_scale = bent_line(line, _SLAB_BEND_RADIUS, centres[cells], edges[cells.start:cells.stop + 1],
                                         reference)
        solved.append((modes_3d, _mode_set(line, int(te.size)), line, face_scale, te))
    return [solved[0], solved[1], solved[0]], cells, pec


def _reduction_errors(sections, total_3d, total_2d):
    """{s21, s11, k_relative}: | |S| of the 3D TE block - |S| of the 2D ez cascade | and the TE k vs 2D (relative)."""
    k_errors = [float(np.max(np.abs(modes_3d.k[te] - modes_2d.k) / np.abs(modes_2d.k)))
                for modes_3d, modes_2d, _, _, te in sections]
    first, last = sections[0][4], sections[-1][4]
    blocks = {'s21': (total_3d.s21[np.ix_(last, first)], total_2d.s21),
              's11': (total_3d.s11[np.ix_(first, first)], total_2d.s11)}
    errors = {name: float(np.max(np.abs(np.abs(block_3d) - np.abs(block_2d)))) for name, (block_3d, block_2d)
              in blocks.items()}
    errors['k_relative'] = max(k_errors)
    return errors


# ---- 3b. the step with a lossy wide section: S, k, 3D field and absorbed power vs 2D ----
_FIELD_LOSS = 0.05            # -Im eps of the wide section
_FIELD_TOLERANCE = 1e-7
_REDUCTION_ITEMS = ('s21', 's11', 'k_relative')   # judged against _SLAB_TOLERANCE, the field items _FIELD_TOLERANCE


def _field_errors(variant):
    """(errors, details) of the step with a lossy wide section, 3D vs 2D ez: _reduction_errors of the cascades, and for
    TE mode 1 from lo the 3D field (cross_section_fields at the Gauss points) vs 2D section_fields: E_y sqrt(dy) = +-u
    on the line cells (one global sign; 3D modes are normalised over dual areas dx dy), E_x = E_z = 0, E = 0 on PEC
    columns, q dy = q_2D of the Gauss means (the solvers' absorbed power) and equal absorbed fractions."""
    from .interfaces import cascade_mode_sets
    from .mode_sets import cross_section_fields, gauss_mean, gauss_positions, section_fields
    sections, cells, pec = _reduction_sets(variant, _FIELD_LOSS)
    lengths = [cells_along * _SLAB_CELL for cells_along in _SLAB_LENGTH_CELLS]
    cascade_3d = cascade_mode_sets([section[0] for section in sections], lengths)
    cascade_2d = cascade_mode_sets([section[1] for section in sections], lengths)
    reduction = _reduction_errors(sections, cascade_3d.total, cascade_2d.total)
    incident_3d = np.zeros(sections[0][0].n_modes, dtype=np.complex128)
    incident_3d[sections[0][4][0]] = 1.0
    incident_2d = np.zeros(sections[0][1].n_modes, dtype=np.complex128)
    incident_2d[0] = 1.0
    fields_3d, fields_2d, q_3d, q_2d = [], [], [], []
    for index, ((modes_3d, modes_2d, line, face_scale, _), n_slices, length) in enumerate(
            zip(sections, _SLAB_LENGTH_CELLS, lengths)):
        positions, _ = gauss_positions(np.full(n_slices, _SLAB_CELL))
        fields = cross_section_fields(modes_3d, *cascade_3d.section_amplitudes(index, 'lo', incident_3d), positions,
                                      length)
        u = section_fields(line, modes_2d, *cascade_2d.section_amplitudes(index, 'lo', incident_2d), positions, length,
                           _OMEGA, face_scale)[0]
        loss = _FIELD_LOSS if index == 1 else 0.0
        fields_3d.append(fields)
        fields_2d.append(u)
        q_3d.append(0.5 * _OMEGA * EPS0 * loss * gauss_mean(np.sum(np.abs(fields) ** 2, axis=1)))
        q_2d.append(0.5 * _OMEGA * EPS0 * loss * gauss_mean(np.abs(u) ** 2))
    fields, u = np.concatenate(fields_3d), np.concatenate(fields_2d)   # (3 n, 3, nx) and (3 n, m) Gauss rows
    height = np.sqrt(_SLAB_CELL)
    e_y = fields[:, 1, cells] * height
    scale = float(np.max(np.abs(u)))
    errors = {'e_y': float(min(np.max(np.abs(e_y - u)), np.max(np.abs(e_y + u)))) / scale,
              'e_x_e_z': float(np.max(np.abs(fields[:, [0, 2], :]))) * height / scale}
    if pec is not None:
        errors['pec_columns'] = float(np.max(np.abs(fields[:, :, pec[:, 0]]))) * height / scale
    absorbed_3d, absorbed_2d = np.concatenate(q_3d), np.concatenate(q_2d)   # (n_slices, nx) W/m^3, (n_slices, m)
    errors['q_abs'] = float(np.max(np.abs(absorbed_3d[:, cells] * _SLAB_CELL - absorbed_2d))) / float(
        np.max(absorbed_2d))
    on_line = float(np.sum(absorbed_3d[:, cells]))
    fractions = {'3d': on_line * _SLAB_CELL ** 3 / float(sections[0][0].power[sections[0][4][0]]),
                 '2d': float(np.sum(absorbed_2d)) * _SLAB_CELL ** 2 / float(sections[0][1].power[0])}
    errors['absorbed_fraction'] = abs(fractions['3d'] - fractions['2d']) / fractions['2d']
    errors.update(reduction)
    # recorded: 3D absorbed power in the cells outside the 2D line (wall-pinned outer cells of the walls / bend variants)
    outside = (float(np.sum(absorbed_3d)) - on_line) / float(np.sum(absorbed_3d))
    return errors, {'absorbed_fraction': fractions, 'q_outside_line_share': outside,
                    'te_modes': [int(section[4].size) for section in sections],
                    'modes_3d': [section[0].n_modes for section in sections]}


def _check_field_reduces_to_slab(results_by_name):
    """S, k, field and absorbed power of the lossy step in the walls, PEC-cell and bend variants; error in tolerance
    units (_SLAB_TOLERANCE for _REDUCTION_ITEMS, _FIELD_TOLERANCE for the rest)."""
    details, worst = {}, 0.0
    for variant in ('walls', 'pec_cells', 'bend'):
        errors, extra = _field_errors(variant)
        scaled = {name: value / (_SLAB_TOLERANCE if name in _REDUCTION_ITEMS else _FIELD_TOLERANCE)
                  for name, value in errors.items()}
        details[variant] = {'errors': errors, 'errors_in_tolerance_units': scaled, **extra}
        worst = max(worst, max(scaled.values()))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


# ---- 4. model run: the anisotropic stack through em_eme_3d ----
_MODEL_CELL = 10e-9
_MODEL_TOLERANCE = 1e-7


def _film_model_spec():
    """em_eme_3d along z through the stack in a 4 x 4-cell x / y periodic column (pec z port faces: no padding); layer
    _LOSSY_LAYER gets sigma_e _LOSSY_SIGMA and the derived P_abs = its absorbed_power integral [W]."""
    edges = np.concatenate(([0.0], np.cumsum(_FILM_LENGTHS)))
    width = _FILM_CELLS * _MODEL_CELL
    geometry, layer_names = [], {}
    for number, (index, z0, z1) in enumerate(zip(_FILM_INDICES, edges[:-1], edges[1:])):
        name = f'layer_{number}'
        geometry.append(GeometrySpec(name, 'box', {'x0': 0.0, 'y0': 0.0, 'z0': float(z0), 'x1': width,
                                                   'y1': width, 'z1': float(z1)}))
        layer_names.setdefault((index, number == _LOSSY_LAYER), []).append(name)
    materials = []
    for (index, lossy_layer), names in layer_names.items():
        properties = {'eps_r': index ** 2, 'eps_yy': _FILM_YY_RATIO * index ** 2}
        if lossy_layer:
            properties['sigma_e'] = _LOSSY_SIGMA
        materials.append(MaterialSpec(('lossy_' if lossy_layer else '') + f'n_{index:g}'.replace('.', 'p'),
                                      properties=properties, selection=names))
    settings = {'wavelength': _WAVELENGTH, 'n_modes': _FILM_MODES, 'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic',
                'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'}
    derived = [DerivedValueSpec('T1', f'{_PHYSICS}_transmission_1'), DerivedValueSpec('R', f'{_PHYSICS}_reflection'),
               DerivedValueSpec('P_abs', 'absorbed_power', [f'layer_{_LOSSY_LAYER}'], 'integral')]
    return ModelSpec(
        title='em_eme_3d lossy anisotropic thin-film stack',
        domain=DomainSpec(size_x=width, size_y=width, size_z=float(edges[-1])),
        geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_PHYSICS, _TYPE_3D, settings=settings, features=[
            FeatureSpec('ports', 'eme_ports'), FeatureSpec('sections', 'eme_section_table')])],
        mesh=MeshSpec(mode='uniform', dx=_MODEL_CELL),
        study=StudySpec(type='stationary'),
        derived_values=derived)


def _check_lossy_film_3d(results_by_name):
    """selftests.lossy_film_details of the run (ez stack reference; _MODEL_TOLERANCE, the absorbed power of the 1 W
    input _LOSSY_3D_TOLERANCE); error in tolerance units."""
    indices = list(_FILM_INDICES)
    indices[_LOSSY_LAYER] = cmath.sqrt(indices[_LOSSY_LAYER] ** 2 - 1j * _LOSSY_SIGMA / (_OMEGA * EPS0))
    details = lossy_film_details(results_by_name['run'], indices, _FILM_LENGTHS, 'ez', _MODEL_TOLERANCE,
                                 _LOSSY_3D_TOLERANCE)
    worst = max(details['errors_in_tolerance_units'].values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': 1.0, 'details': details}


# ---- 5. model run: pec_domain cells in em_eme_3d (spec and check of eme_cpu_gpu / eme_3d_cudss_pec_model) ----
_PEC_SECTION_CELLS = (12, 32, 12)   # narrow | wide | narrow along z
_PEC_MODEL_MODES = 40
_PEC_PORT_MODES = 6
_PEC_MODEL_TOLERANCE = 1e-7


def _pec_step_spec():
    """The pec_cells variant of eme_3d_field_reduces_to_slab (lossless) as an em_eme_3d model (periodic x / y faces, the PEC
    columns drawn as boxes with pec_domain), stationary."""
    cell = _SLAB_CELL
    size_x, size_y, size_z = _SLAB_CELLS * cell, cell, sum(_PEC_SECTION_CELLS) * cell
    starts = np.concatenate(([0], np.cumsum(_PEC_SECTION_CELLS))) * cell
    geometry = []
    for name, core, z0, z1 in zip(('input_core', 'wide_core', 'output_core'),
                                  (_SLAB_CORES[0], _SLAB_CORES[1], _SLAB_CORES[0]), starts[:-1], starts[1:]):
        geometry.append(GeometrySpec(name, 'box', {'x0': core[0] * cell, 'y0': 0.0, 'z0': float(z0),
                                                   'x1': core[1] * cell, 'y1': size_y, 'z1': float(z1)}))
    for name, x0, x1 in (('pec_lo', 0.0, _SLAB_PEC_COLUMNS * cell),
                         ('pec_hi', (_SLAB_CELLS - _SLAB_PEC_COLUMNS) * cell, size_x)):
        geometry.append(GeometrySpec(name, 'box', {'x0': x0, 'y0': 0.0, 'z0': 0.0, 'x1': x1, 'y1': size_y,
                                                   'z1': size_z}))
    settings = {'wavelength': _WAVELENGTH, 'n_modes': _PEC_MODEL_MODES, 'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic',
                'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'}
    return ModelSpec(
        title='em_eme_3d step with PEC cell columns',
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=geometry,
        materials=[MaterialSpec('cladding', properties={'eps_r': _SLAB_EPS[0]}, selection=['all']),
                   MaterialSpec('core', properties={'eps_r': _SLAB_EPS[1]},
                                selection=['input_core', 'wide_core', 'output_core'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE_3D, settings=settings, features=[
            FeatureSpec('ports', 'eme_ports', params={'n_port_modes': _PEC_PORT_MODES}),
            FeatureSpec('pec', 'pec_domain', ['pec_lo', 'pec_hi'])])],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='stationary'))


@functools.lru_cache(maxsize=1)
def _pec_reference_sets():
    """(narrow, wide) CPU 3D ModeSets of the PEC step (float32 kernel widths, the pec mask); memoised: eme_cpu_gpu and
    eme_3d_cudss_pec_model read the same deterministic sets (never mutated)."""
    kernel_width = float(np.float32(_SLAB_CELL))
    widths = (np.full(_SLAB_CELLS, kernel_width), np.full(1, kernel_width))
    pec = np.zeros((_SLAB_CELLS, 1), dtype=bool)
    pec[:_SLAB_PEC_COLUMNS] = pec[_SLAB_CELLS - _SLAB_PEC_COLUMNS:] = True
    sets = []
    for core in _SLAB_CORES:
        eps = np.full(_SLAB_CELLS, _SLAB_EPS[0])
        eps[core[0]:core[1]] = _SLAB_EPS[1]
        sets.append(_cross_section_set(eps[:, None], eps[:, None], eps[:, None], widths, _boundary('periodic'),
                                       _PEC_MODEL_MODES, pec=pec))
    return tuple(sets)


def _check_pec_model_3d(results_by_name):
    """|S| of the reported port block and the input-face n_eff vs the CPU 3D cascade of the same cross-sections (float32
    kernel widths, the pec mask)."""
    from .interfaces import cascade_mode_sets
    results = results_by_name['run']
    ports = results.result(_PHYSICS, 'ports')
    model = np.abs(np.asarray(ports['s_matrix_real']) + 1j * np.asarray(ports['s_matrix_imag']))
    n_lo, n_hi = int(ports['s_matrix_modes']['lo']), int(ports['s_matrix_modes']['hi'])
    narrow, wide = _pec_reference_sets()
    sets = [narrow, wide, narrow]
    total = cascade_mode_sets(sets, [cells * _SLAB_CELL for cells in _PEC_SECTION_CELLS]).total
    reference = np.abs(np.block([[total.s11[:n_lo, :n_lo], total.s12[:n_lo, :n_hi]],
                                 [total.s21[:n_hi, :n_lo], total.s22[:n_hi, :n_hi]]]))
    n_eff_model = np.array([float(row[2]) for row in ports['rows'][:n_lo]])
    errors = {'s_abs': float(np.max(np.abs(model - reference))) if model.shape == reference.shape else 1.0,
              'n_eff': float(np.max(np.abs(n_eff_model - sets[0].n_eff[:n_lo].real)))}
    worst = max(errors.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _PEC_MODEL_TOLERANCE,
            'details': {'errors': errors, 'n_eff_reference': float_list(sets[0].n_eff[:n_lo].real)}}


# ---- 6. the PEC model run with an eigen node on cuDSS (gpu backend) ----
_CUDSS_NODE = 'cross_section_cudss'


def _cudss_pec_specs():
    """{'run': the PEC step spec with an arpack_shift_invert node of factorization 'cudss'} when nvmath is
    installed, else {} (the check skips)."""
    from ...solvers.cudss_factor import cudss_available
    if not cudss_available():
        return {}
    spec = _pec_step_spec()
    spec.title = 'em_eme_3d step with PEC cell columns, cuDSS cross-section factors'
    spec.solvers = [SolverSpec(_CUDSS_NODE, 'arpack_shift_invert', {'factorization': 'cudss'})]
    spec.physics[0].solvers = {'eigen': _CUDSS_NODE}
    return {'run': spec}


def _check_cudss_pec_model_3d(results_by_name):
    """_check_pec_model_3d's items on the cuDSS run (reference: host SuperLU sets); the eigen role factorised on the
    device (counters recorded)."""
    if 'run' not in results_by_name:
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _PEC_MODEL_TOLERANCE,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    result = _check_pec_model_3d(results_by_name)
    record = results_by_name['run'].summary().get('solvers', {}).get(_PHYSICS, {})
    counters = (record.get('roles', {}).get('eigen') or {}).get('counters') or {}
    result['details'].update(eigen_counters=counters, physics_counters=record.get('counters') or {})
    if not int(counters.get('eigen_factor_device_bytes', 0)) > 0:   # no device factor: not a cuDSS run
        result['error'] = result['measured'] = float('inf')
    return result


EME_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='eme_3d_uniform_identity',
        label='EME 3D: uniform guide in three sections',
        description=f'CPU check of the em_eme_3d mode sets (no model run): a {_BOX_CELLS} x {_BOX_CELLS}-cell '
                    f'cross-section of {_BOX_CELL * 1e9:g} nm (core eps {_BOX_EPS[1]:g} over 0.4 x 0.24 um in eps '
                    f'{_BOX_EPS[0]:g}, {_BOX_MODES} full-vector modes, PML modes kept) used for three sections of 16, '
                    '30 and 52 cells at 1.55 um: the interfaces between identical mode sets must vanish, so S21 = S12 = '
                    'diag(exp(-j beta L_total)), S11 = S22 = 0, and the middle section carries mode 1 alone. This needs '
                    'the unconjugated <e, h> = sum (E x H).z dA to be exactly orthonormal over the set (discrete '
                    'reciprocity of the Yee cross-section, degenerate modes recombined). Periodic and pec faces '
                    f'asserted, tolerance {_IDENTITY_TOLERANCE:g} (ARPACK eigenvector accuracy); cpml faces (stretched '
                    'dual areas) recorded.',
        specs=lambda: {},
        check=_check_uniform_identity_3d,
        max_wall_time=60.0),
    SelfTestCase(
        name='eme_3d_field_reduces_to_slab',
        label='EME 3D: y-invariant step reduces to the 2D EME',
        description=f'CPU check: narrow (0.2 um) | wide (0.6 um) | narrow guide (eps 4 in 2.25, {_SLAB_CELLS} cells '
                    f'of {_SLAB_CELL * 1e9:g} nm along x between pec walls, sections 12, 32, 12 cells, the wide '
                    f'section lossy: -Im eps = {_FIELD_LOSS:g}) as a 3D cross-section one cell high with periodic y '
                    f'({_SLAB_MODES} modes) and as 2D ez slab lines of the interior cells with \'odd\' ends (the '
                    'cross-section pec wall is the antisymmetric image of the ghost cell). TE (E_y) and TM modes '
                    'decouple exactly, and the TE operator equals the 2D ez line operator (complex eps too), so the TE '
                    'propagation constants and |S21|, |S11| of the TE block of the 3D cascade must equal the 2D '
                    'cascade with the same TE modes per section (magnitudes: the mode phase conventions differ), '
                    f'within {_SLAB_TOLERANCE:g}. Two more variants: periodic x with 2 PEC cell columns at each end '
                    '(E_y = 0 at their centres: 2D dirichlet ends on the dielectric cells), and the pec-wall step with '
                    f'the wide section bent (R = {_SLAB_BEND_RADIUS * 1e6:g} um about the x centre; cross_section_modes '
                    "'exact' bend map eps, mu diag(s, s, 1/s) vs the 2D bent_line cell weight s, face weight s_f, "
                    'index s^2: the same TE operator). Field / absorbed-power path, TE mode 1 from lo: the 3D E of '
                    'cross_section_fields at the 3-point Gauss points of every slice vs the 2D ez section_fields u. '
                    'E_y sqrt(dy) must equal u on the 2D line cells up to one global sign (3D modes are normalised '
                    'over dual areas dx dy, 2D over dx), E_x and E_z vanish (TE), E vanishes on the PEC columns, the '
                    'absorbed power density 1/2 omega eps0 (-Im eps) times the Gauss cell mean of |E|^2 (the '
                    'em_eme_3d / em_eme q_abs) times dy equals the 2D one, and the absorbed fractions [W / W, W/m / '
                    f'W/m] agree, relative errors within {_FIELD_TOLERANCE:g} (a first guess: ARPACK eigenvector '
                    'accuracy). error = the largest item in its own tolerance units, tolerance 1. Recorded: the 3D '
                    'absorbed share in the cells outside the 2D line (the wall-pinned outer cells). Not yet run with '
                    'the S / k items on the lossy wide section.',
        specs=lambda: {},
        check=_check_field_reduces_to_slab,
        max_wall_time=60.0),
    SelfTestCase(
        name='eme_3d_lossy_film_model',
        label='EME 3D: thin-film stack model run with a lossy film',
        description='em_eme_3d along z through an anisotropic thin-film stack (sqrt(eps_xx) = 1.25 | 1.5 | 2.5 | 1.5 | '
                    '2.5 | 2.0, lengths 0.30, 0.21, 0.13, 0.21, 0.13, 0.32 um, 1.55 um, 10 modes) drawn as boxes in a '
                    f'4 x 4-cell x / y periodic column ({_MODEL_CELL * 1e9:g} nm cells, eps_r = eps_xx = eps_zz exact in '
                    'float32, eps_yy = 0.9 eps_r: the uniform x-polarised mode is non-degenerate and bilinear-orthogonal '
                    'to every other mode), stationary study, with its first 2.5 layer (0.13 um) lossy (sigma_e = '
                    f'{_LOSSY_SIGMA:g} S/m, float32-exact). The sections from the z mesh must be the six layers with '
                    'their lengths, S11, S21, S12, S22 of mode 1 (the x-polarised plane wave) the transfer matrix with '
                    'the complex layer index sqrt(eps_xx - j sigma / (omega eps0)) and transmission_1 / reflection '
                    f'|S21|^2 / |S11|^2 (lossless port media), within {_MODEL_TOLERANCE:g}; the integral of '
                    'absorbed_power over the layer (1/2 omega eps0 (-Im eps) times the 3-point Gauss z mean of |E|^2 '
                    'per cell) must equal the absorbed fraction 1 - |S11|^2 - |S21|^2 of the 1 W input within '
                    f'{_LOSSY_3D_TOLERANCE:g}. error = the largest item in its own tolerance units, tolerance 1. Not '
                    f'yet run with the merged lossless-stack items (S entries at {_MODEL_TOLERANCE:g} on the lossy '
                    'stack).',
        specs=lambda: {'run': _film_model_spec()},
        check=_check_lossy_film_3d),
    SelfTestCase(
        name='eme_3d_cudss_pec_model',
        label='EME 3D: PEC cells model run on cuDSS factors',
        description='em_eme_3d of the pec_cells variant of eme_3d_field_reduces_to_slab (lossless; 60 x 1 cells of 25 '
                    'nm, periodic x / y, 2 PEC cell columns at each x end drawn as boxes with pec_domain; narrow | wide '
                    f'| narrow over 12, 32, 12 z cells; {_PEC_MODEL_MODES} modes; the PEC step of eme_cpu_gpu) with the '
                    'eigen role on an arpack_shift_invert node of factorization cudss (each '
                    'cross-section shift-invert operator factorised by cuDSS LU in double on the GPU; the transverse-H '
                    f'operator is not complex symmetric), gpu backend: |S| of the reported port block ({_PEC_PORT_MODES} '
                    'modes per face) and the input-face n_eff against the host SuperLU CPU 3D cascade of the same '
                    'cross-sections (float32 kernel widths, the PEC mask), tolerance '
                    f'{_PEC_MODEL_TOLERANCE:g}, and a device '
                    'factor must have been made (eigen_factor_device_bytes > 0). Recorded: the eigen role and physics '
                    'counters (mode_factor_seconds / mode_eigen_seconds / max_factor_device_bytes, refinement steps). '
                    'Skipped without nvmath-python. Written, not run.',
        specs=_cudss_pec_specs,
        check=_check_cudss_pec_model_3d,
        backends=('gpu',)),
)


# ---- template: thermo-optic phase shifter (heat + em_eme_3d) ----
SHIFTER_3D_LABEL = 'Thermo-optic phase shifter (EME 3D)'
SHIFTER_3D_DESCRIPTION = (
    'Heat + em_eme_3d, stationary: a 0.5 x 0.2 um silicon strip in silica running 20 um along z, a 1.5 x 0.2 um heater '
    'region 1 um above it from z = 5 to 15 um dissipating P_heater (parameter, 1 mW), the substrate face y_lo at '
    '293.15 K; steady heat, heat_em_thermo_optic, then the EME of the heated guide (pec transverse walls of the 4 x 3 '
    'um box, 50 nm cells, 10 modes, section_tolerance 1e-4: heated layers within about 1 K of a section\'s first '
    'layer merge into it, a staircase of a few percent of the phase shift). The '
    'transmission row of the eme_ports table gives the phase of S21 (sweep P_heater to get the phase shift, about 0.1 '
    'rad per mW here); derived T1 and the mean core temperature.')
_SHIFTER_BOX = (4e-6, 3e-6, 20e-6)
_SHIFTER_CORE = (1.75e-6, 1.0e-6, 2.25e-6, 1.2e-6)        # x0, y0, x1, y1
_SHIFTER_HEATER = (1.25e-6, 2.2e-6, 5e-6, 2.75e-6, 2.4e-6, 15e-6)   # x0, y0, z0, x1, y1, z1
_SHIFTER_POWER = 1e-3   # W


def thermo_optic_phase_shifter_eme_3d_spec():
    """Template 'thermo_optic_phase_shifter_eme_3d' (heat + em_eme_3d, stationary): S21 of a heated silicon strip."""
    size_x, size_y, size_z = _SHIFTER_BOX
    x0, y0, x1, y1 = _SHIFTER_CORE
    hx0, hy0, hz0, hx1, hy1, hz1 = _SHIFTER_HEATER
    heater_volume = (hx1 - hx0) * (hy1 - hy0) * (hz1 - hz0)
    return ModelSpec(
        title=SHIFTER_3D_LABEL,
        parameters=[ParameterSpec('P_heater', _SHIFTER_POWER, 'heater power [W]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=[GeometrySpec('core', 'box', {'x0': x0, 'y0': y0, 'z0': 0.0, 'x1': x1, 'y1': y1, 'z1': size_z}),
                  GeometrySpec('heater', 'box', {'x0': hx0, 'y0': hy0, 'z0': hz0, 'x1': hx1, 'y1': hy1, 'z1': hz1})],
        materials=[MaterialSpec('cladding', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['core'])],
        # 1.92 M voxels: a Jacobi pcg node on both backends (Automatic would be multigrid pcg)
        solvers=[SolverSpec('heat_cg', 'pcg', {})],
        physics=[PhysicsSpec('heat', 'heat', solvers={'linear': 'heat_cg'}, features=[
                     FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': 293.15}),
                     FeatureSpec('heater_power', 'heat_source', ['heater'], {'Q': f'P_heater / {heater_volume!r}'})]),
                 PhysicsSpec(_PHYSICS, _TYPE_3D, settings={
                     'wavelength': _WAVELENGTH, 'n_modes': 10, 'section_tolerance': 1e-4, 'bc_x_lo': 'pec',
                     'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec'},
                     features=[FeatureSpec('ports', 'eme_ports', params={'n_port_modes': 2}),
                               FeatureSpec('sections', 'eme_section_table')])],
        couplings=[CouplingSpec('thermo_optic', 'heat_em_thermo_optic', 'heat', _PHYSICS)],
        mesh=MeshSpec(mode='uniform', dx=50e-9),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T1', f'{_PHYSICS}_transmission_1'),
                        DerivedValueSpec('T_core', 'T', ['core'], 'mean')])


EME_3D_TEMPLATES = (
    ('thermo_optic_phase_shifter_eme_3d', SHIFTER_3D_LABEL, SHIFTER_3D_DESCRIPTION,
     thermo_optic_phase_shifter_eme_3d_spec,
     'A heater above a silicon waveguide shifts the phase of the light passing it; the 3D device is computed by '
     'eigenmode expansion along the waveguide.'),
)

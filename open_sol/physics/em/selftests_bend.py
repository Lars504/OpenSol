"""Self-tests of the em_modes bend (symmetry 2.19, cross_section_modes `bend`, models 'exact' and 'conformal'): CPU, the
bent slab against weighted slab_modes lines of the same operator and the exact continuum transform, the bend-mapped SHG
overlap against that transform; model run, the leaky bend loss of a dielectric slab against a fine 1D transform solve."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import relative_error
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_cross_section import _graded_cladding, _isotropic

_WAVELENGTH = 1.55e-6                             # m, vacuum
_CORE_INDEX, _CLADDING_INDEX = 2.0, 1.444
_CORE_WIDTH = 0.8e-6                              # m along x (the bend plane): V = 2.24, TE0 / TE1 guided
_CORE_CELL = 5e-9                                 # m: 160 core cells
_CLADDING = 3.0e-6                                # m per side between the core and the pec wall
_GROWTH, _MAX_CELL = 1.05, 20e-9                  # graded cladding: x1.05 per cell up to 20 nm
_PERIOD = 50e-9                                   # m: y width of the single periodic cell (y-invariant slab)
_RADII = (20e-6, 40e-6)                           # m: discrete-operator checks at the first, transform at both
_MODELS = ('exact', 'conformal')                  # cross_section_modes.BEND_MODELS (repeated: selftest import rule)
_CROSS_SECTION_MODES = 8                          # E_x and E_y families together
_COMPARED_MODES = 2
_OPERATOR_TOLERANCE = 1e-9                        # |n_eff - weighted slab line of the same discrete operator|
_POLARIZATION_PURITY = 1e-10                      # E_y modes: te_fraction; E_x modes: 1 - te_fraction
_EXACT_LIMIT = 2e-5                               # |n_eff - exact transform|, exact model (discretisation only)
_CONFORMAL_LIMIT = 2e-3                           # |n_eff - exact transform| at the smaller radius, conformal model
_CONFORMAL_MIN_RATIO = 3.0                        # error(R) / error(2 R): second order gives ~4, first order 2


def _omega():
    """Angular frequency [rad/s] of _WAVELENGTH."""
    return 2.0 * math.pi * C0 / _WAVELENGTH


def _bend_setup():
    """(dx (nx,) [m], eps (nx,) complex, x_edges (nx + 1,) [m], x_ref = core centre [m]) along x: ghost, graded
    cladding, core, cladding, ghost (ghosts copy their neighbour)."""
    cladding = _graded_cladding(_CORE_CELL, _CLADDING, _GROWTH, _MAX_CELL)
    n_core = int(round(_CORE_WIDTH / _CORE_CELL))
    inner = np.concatenate((cladding[::-1], np.full(n_core, _CORE_CELL), cladding))
    dx = np.concatenate(([inner[0]], inner, [inner[-1]]))
    eps = np.full(dx.size, _CLADDING_INDEX ** 2, dtype=np.complex128)
    core_start = 1 + cladding.size
    eps[core_start:core_start + n_core] = _CORE_INDEX ** 2
    x_edges = np.concatenate(([0.0], np.cumsum(dx)))
    reference_x = 0.5 * (x_edges[core_start] + x_edges[core_start + n_core])
    return dx, eps, x_edges, reference_x


def _point_factors(model, radius):
    """Bend factors (nx,) of the module header of cross_section_modes, recomputed: (eps_xx at the hi-x faces, eps_yy
    and eps_zz at the centres, mu_xx at the centres, mu_yy and mu_zz at the hi-x faces)."""
    dx, _, x_edges, reference_x = _bend_setup()
    s_centre =1.0 + (0.5 * (x_edges[:-1] + x_edges[1:]) - reference_x) / radius
    s_face = 1.0 + (x_edges[1:] - reference_x) / radius
    if model == 'conformal':
        ones = np.ones(dx.size)
        return s_face ** 2, s_centre ** 2, s_centre ** 2, ones, ones, ones
    return s_face, s_centre, 1.0 / s_centre, s_centre, s_face, 1.0 / s_face


def _ez_line(widths, index_term, cell_weight=None, face_weight=None):
    """slab_modes 'ez' SlabLine of cells 1..n-2 (between the pec wall planes, odd ends) of full-grid widths [m],
    index_term / cell_weight (n,) and face_weight (n,) at the hi faces (None = 1); slab face k = hi face of cell k."""
    from . import slab_modes
    from .yee_grid import yee_inverse_spacings
    n_cells = widths.size - 2
    ones_cells, ones_faces = np.ones(n_cells, dtype=np.complex128), np.ones(n_cells + 1, dtype=np.complex128)
    cell_weight = ones_cells if cell_weight is None else np.asarray(cell_weight, dtype=np.complex128)[1:-1]
    face_weight = ones_faces if face_weight is None else np.asarray(face_weight, dtype=np.complex128)[:n_cells + 1]
    return slab_modes.SlabLine(
        polarization='ez', cell_widths=widths[1:-1], cell_stretch=ones_cells, cell_weight=cell_weight,
        index_term=np.asarray(index_term, dtype=np.complex128)[1:-1], active=np.ones(n_cells, dtype=bool),
        face_inverse_spacing=yee_inverse_spacings(widths, widths).inv_dx_h[:n_cells + 1], face_stretch=ones_faces,
        face_weight=face_weight, lo_end='odd', hi_end='odd')


def _dual_line(widths, cell_weight, face_weight, index_term):
    """slab_modes SlabLine on the hi-x faces 0..n-2 (the H_y points from the lo to the hi pec wall plane, those two
    halved: the plane mirrors the E_z flux, neumann ends): cell_weight / index_term (n,) at the hi faces, face_weight
    (n,) at the centres (slab face k = cell k)."""
    from . import slab_modes
    from .yee_grid import yee_inverse_spacings
    n_faces = widths.size - 1
    dual_widths = 1.0 / yee_inverse_spacings(widths, widths).inv_dx_h[:n_faces]
    dual_widths[0] *= 0.5
    dual_widths[-1] *= 0.5
    ones_faces, ones_cells = np.ones(n_faces, dtype=np.complex128), np.ones(n_faces + 1, dtype=np.complex128)
    return slab_modes.SlabLine(
        polarization='hz', cell_widths=dual_widths, cell_stretch=ones_faces,
        cell_weight=np.asarray(cell_weight, dtype=np.complex128)[:n_faces],
        index_term=np.asarray(index_term, dtype=np.complex128)[:n_faces], active=np.ones(n_faces, dtype=bool),
        face_inverse_spacing=1.0 / widths, face_stretch=ones_cells,
        face_weight=np.asarray(face_weight, dtype=np.complex128), lo_end='neumann', hi_end='neumann')


def _operator_lines(model, radius):
    """(E_y line, E_x line) of the y-invariant reduction of the bent cross-section operator: E_y family on v = mu_xx
    h_x (cell_weight mu_xx, face_weight 1 / mu_zz, index eps_yy mu_xx), E_x family on h_y (cell_weight eps_xx, face
    weight 1 / eps_zz, index eps_xx mu_yy)."""
    from .yee_grid import face_average
    dx, eps, x_edges, reference_x = _bend_setup()
    eps_xx_factor, eps_yy_factor, eps_zz_factor, mu_xx, mu_yy, mu_zz = _point_factors(model, radius)
    eps_x_faces = face_average(eps[:, None], 0, False, dx)[:, 0] * eps_xx_factor
    ey_line = _ez_line(dx, eps * eps_yy_factor * mu_xx, cell_weight=mu_xx, face_weight=1.0 / mu_zz)
    ex_line = _dual_line(dx, eps_x_faces, 1.0 / (eps * eps_zz_factor), eps_x_faces * mu_yy)
    return ey_line, ex_line


def _exact_transform_line(radius):
    """'ez' SlabLine of the exact bent slab: u = x_ref + R ln(r / R), r = R + x - x_ref, index_term eps (r / R)^2
    at the u-cell centres (the field along the bend axis obeys this scalar equation exactly)."""
    dx, eps, x_edges, reference_x = _bend_setup()
    u_edges = reference_x + radius * np.log1p((x_edges - reference_x) / radius)
    u_centres = 0.5 * (u_edges[:-1] + u_edges[1:])
    return _ez_line(np.diff(u_edges), eps * np.exp(2.0 * (u_centres - reference_x) / radius))


def _bend_modes(radius, model, omega=None):
    """(E_y modes (te_fraction < 0.5: E along the bend axis y), E_x modes) of the bent slab cross-section, pec walls
    on x, at omega [rad/s] (None = _WAVELENGTH)."""
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    dx, eps, x_edges, reference_x = _bend_setup()
    solve = cross_section_modes.solve_cross_section_modes(
        (dx, np.array([_PERIOD])), BoundaryConfig('pec', 'pec', 'periodic', 'periodic'), _CORE_CELL,
        _isotropic(cross_section_modes, eps[:, None]), _omega() if omega is None else omega, _CROSS_SECTION_MODES,
        bend=(radius, reference_x, x_edges, model))
    return ([mode for mode in solve.modes if mode.te_fraction < 0.5],
            [mode for mode in solve.modes if mode.te_fraction >= 0.5])


def _operator_errors(label, modes_ey, modes_ex, model, radius, violations):
    """{family_mode: |n_eff - weighted slab line|} of both families; purity / count failures into violations."""
    from . import slab_modes
    omega = _omega()
    errors = {}
    for family, modes, line in zip(('ey', 'ex'), (modes_ey, modes_ex), _operator_lines(model, radius)):
        if len(modes) < _COMPARED_MODES:
            violations.append(f'{label}: {len(modes)} E_{family[1]} modes returned, expected {_COMPARED_MODES}')
        impurity = [mode.te_fraction if family == 'ey' else 1.0 - mode.te_fraction for mode in modes]
        violations += [f'{label} E_{family[1]} mode {number}: polarisation impurity {value!r}'
                       for number, value in enumerate(impurity[:_COMPARED_MODES]) if value > _POLARIZATION_PURITY]
        references = slab_modes.solve_slab_modes(line, omega, _COMPARED_MODES)
        for number, (mode, reference) in enumerate(zip(modes, references)):
            errors[f'{family}_mode{number}'] = abs(mode.n_eff - reference.n_eff)
    return errors


def _check_bend(results_by_name):
    """Both models: bent E_y / E_x modes vs the weighted slab lines of the same operator (round-off) at R = 20 um;
    E_y mode 0 vs the exact transform (exact model: discretisation only; conformal: second order in 1 / R)."""
    from . import slab_modes
    omega = _omega()
    operator_errors, transform_errors, n_eff, violations = {}, {model: {} for model in _MODELS}, {}, []
    for radius in _RADII:
        exact = slab_modes.solve_slab_modes(_exact_transform_line(radius), omega, 1)[0]
        n_eff[f'R_{radius * 1e6:g}um_exact_transform'] = exact.n_eff.real
        for model in _MODELS:
            label = f'{model}_R_{radius * 1e6:g}um'
            modes_ey, modes_ex = _bend_modes(radius, model)
            n_eff[label] = {'ey': [[mode.n_eff.real, mode.n_eff.imag] for mode in modes_ey[:_COMPARED_MODES]],
                            'ex': [[mode.n_eff.real, mode.n_eff.imag] for mode in modes_ex[:_COMPARED_MODES]]}
            if not modes_ey:
                violations.append(f'{label}: no E_y mode returned')
                continue
            if radius == _RADII[0]:
                for key, error in _operator_errors(label, modes_ey, modes_ex, model, radius, violations).items():
                    operator_errors[f'{model}_{key}'] = error
            transform_errors[model][label] = abs(modes_ey[0].n_eff - exact.n_eff)
    violations += [f'{label}: exact-model error {error:.3g} vs the exact transform exceeds {_EXACT_LIMIT}'
                   for label, error in transform_errors['exact'].items() if not error <= _EXACT_LIMIT]
    ratio = float('nan')
    if len(transform_errors['conformal']) == len(_RADII):
        small_radius_error, large_radius_error = transform_errors['conformal'].values()
        ratio = small_radius_error / large_radius_error if large_radius_error > 0.0 else float('inf')
        if not small_radius_error <= _CONFORMAL_LIMIT:
            violations.append(f'conformal error {small_radius_error:.3g} at R = {_RADII[0]:g} m exceeds '
                              f'{_CONFORMAL_LIMIT}')
        if not ratio >= _CONFORMAL_MIN_RATIO:
            violations.append(f'conformal error ratio {ratio:.3g} for R doubled is below {_CONFORMAL_MIN_RATIO} '
                              f'(not second order in 1/R)')
    measured = max(operator_errors.values(), default=float('nan'))
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations),
            'tolerance': _OPERATOR_TOLERANCE,
            'details': {'n_eff_errors_operator': operator_errors, 'transform_errors': transform_errors,
                        'conformal_error_ratio': ratio, 'n_eff': n_eff, 'violations': violations}}


# SHG overlap through the bend (nonlinear_overlap.bend_overlap_factors): the exact model vs the exact transform.
_SHG_RADIUS = 20e-6                               # m
_SHG_D = 2.5e-11                                  # m/V, one Voigt d_il at a time
_SHG_COMPONENTS = {'ey': ('d22',), 'ex': ('d11', 'd13', 'd31')}   # tested d_il per mode family (pump and harmonic)
_SHG_MAPPING_SENSITIVE = ('d22', 'd11', 'd13')    # d31: J / h_z = 1 in the exact model, nothing to detect
_SHG_TOLERANCE = 1e-3                             # relative |kappa| error vs the exact transform (discretisation)
_SHG_MIN_SENSITIVITY = 3.0                        # unmapped |kappa| error >= this x tolerance (the check can fail)


def _family_mode(modes, family):
    """First mode of family 'ey' (te_fraction < 0.5) or 'ex' in rank order; None when absent."""
    return next((mode for mode in modes if (mode.te_fraction < 0.5) == (family == 'ey')), None)


def _shg_mask():
    """Bool (nx, 1) chi2 cells: the outer half of the core (x > x_ref), so every tested d_il has a non-zero overlap and
    the bend factors differ from 1 on one side only."""
    _, eps, x_edges, reference_x = _bend_setup()
    centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    return ((eps.real > _CLADDING_INDEX ** 2) & (centres > reference_x))[:, None]


def _transform_mode(family, omega):
    """(mode of `family`, du (nx,), s (nx,) at the u centres) of the exact transform u = x_ref + R ln(r / R) solved
    straight: transformation optics of (u, y, R phi), h = (s, 1, s), eps' = eps diag(1, s^2, 1), mu' alike; the E_y
    family sees mu_xx = mu_zz = 1, the E_x family mu_yy = s^2 (set at the hi-u faces of H_y)."""
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    _, eps, x_edges, reference_x = _bend_setup()
    u_edges = reference_x + _SHG_RADIUS * np.log1p((x_edges - reference_x) / _SHG_RADIUS)
    centre_scale = np.exp((0.5 * (u_edges[:-1] + u_edges[1:]) - reference_x) / _SHG_RADIUS)
    face_scale = np.exp((u_edges[1:] - reference_x) / _SHG_RADIUS)
    eps_cells = eps[:, None]
    mu = np.ones(eps_cells.shape) if family == 'ey' else face_scale[:, None] ** 2
    materials = cross_section_modes.CrossSectionMaterials(
        eps_xx=eps_cells.copy(), eps_yy=eps_cells * centre_scale[:, None] ** 2, eps_zz=eps_cells.copy(),
        mu=mu.astype(np.complex128), pec=np.zeros(eps_cells.shape, dtype=bool))
    solve = cross_section_modes.solve_cross_section_modes(
        (np.diff(u_edges), np.array([_PERIOD])), BoundaryConfig('pec', 'pec', 'periodic', 'periodic'), _CORE_CELL,
        materials, omega, _CROSS_SECTION_MODES)
    return _family_mode(solve.modes, family), np.diff(u_edges), centre_scale


def _check_shg_bend(_results):
    """|kappa| of the exact bent modes with the bend-mapped overlap vs the exact transform's physical-field overlap
    (dV = s^2 du dy dz, E = (E_u / s, E_y, E_v / s)); the unmapped overlap must miss it (sensitivity)."""
    from . import cross_section_modes, nonlinear_overlap
    omega = _omega()
    dx, _, x_edges, reference_x = _bend_setup()
    mask = _shg_mask()
    area = np.where(mask, dx[:, None] * _PERIOD, 0.0)
    bends = {model: (_SHG_RADIUS, reference_x, x_edges, model) for model in _MODELS}
    bent = {model: [_bend_modes(_SHG_RADIUS, model, harmonic * omega) for harmonic in (1, 2)] for model in _MODELS}
    errors, unmapped, conformal, expected_kappa, violations = {}, {}, {}, {}, []

    def centres(mode):
        return cross_section_modes.centre_fields(mode, False, True)

    for family_index, (family, components) in enumerate(_SHG_COMPONENTS.items()):
        (pump_ref, du, scale), (harmonic_ref, _, _) = (_transform_mode(family, harmonic * omega) for harmonic in (1, 2))
        pairs = {model: [next(iter(bent[model][harmonic][family_index]), None) for harmonic in (0, 1)]
                 for model in _MODELS}
        if pump_ref is None or harmonic_ref is None or None in pairs['exact'] or None in pairs['conformal']:
            violations.append(f'E_{family[1]} family: a pump or harmonic mode is missing')
            continue
        s = scale[:, None]

        def physical(mode):
            fields = centres(mode)
            return {'ex': fields['ex'] / s, 'ey': fields['ey'], 'ez': fields['ez'] / s}

        reference_area = np.where(mask, du[:, None] * _PERIOD * s ** 2, 0.0)
        for name in components:
            d_tensor = nonlinear_overlap.voigt_tensor({name: _SHG_D})
            expected = abs(nonlinear_overlap.shg_overlap(physical(pump_ref), physical(harmonic_ref), d_tensor,
                                                         reference_area))
            pump, harmonic = (centres(mode) for mode in pairs['exact'])
            expected_kappa[name] = expected
            errors[name] = relative_error(
                abs(nonlinear_overlap.shg_overlap(pump, harmonic, d_tensor, area, bends['exact'])), expected)
            unmapped[name] = relative_error(abs(nonlinear_overlap.shg_overlap(pump, harmonic, d_tensor, area)),
                                             expected)
            conformal_pump, conformal_harmonic = (centres(mode) for mode in pairs['conformal'])
            conformal[name] = relative_error(abs(nonlinear_overlap.shg_overlap(
                conformal_pump, conformal_harmonic, d_tensor, area, bends['conformal'])), expected)
    violations += [f'{name}: the unmapped overlap error {unmapped[name]:.3g} is below {_SHG_MIN_SENSITIVITY} x the '
                   f'tolerance (the check could not see the bend factors)' for name in _SHG_MAPPING_SENSITIVE
                   if name in unmapped and not unmapped[name] >= _SHG_MIN_SENSITIVITY * _SHG_TOLERANCE]
    measured = max(errors.values(), default=math.inf)
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations), 'tolerance': _SHG_TOLERANCE,
            'details': {'kappa_errors_exact': errors, 'kappa_errors_unmapped': unmapped,
                        'kappa_errors_conformal': conformal, 'expected_abs_kappa': expected_kappa,
                        'violations': violations}}


# Leaky bend loss (model run): a weakly guiding y-invariant slab, E along y, pec on the inner side, cpml outside.
_LOSS_TYPE, _LOSS_PHYSICS, _LOSS_TABLE = 'em_modes', 'modes', 'table'
_LOSS_CORE_INDEX, _LOSS_CLADDING_INDEX = 1.50, 1.45
_LOSS_CORE_WIDTH = 2.0e-6                         # m: V = 1.56, one guided mode per family, n_eff ~ 1.482
_LOSS_INNER, _LOSS_OUTER = 4.0e-6, 7.0e-6         # m of cladding from the core to the pec wall / to the cpml
_LOSS_RADIUS = 200e-6                             # m: radiation caustic ~4.5 um outside the core centre, ~16 dB/cm
_LOSS_CELL = 20e-9                                # m, uniform (core edges on cell edges)
_LOSS_PML_CELLS = 100                             # 2 um of cpml: the radiation is nearly grazing (k_x ~ 1e6 1/m)
_LOSS_N_MODES = 10
_LOSS_REFERENCE_CELL = 2e-9                       # m, the 1D reference in u
_LOSS_REFERENCE_PML, _LOSS_REFERENCE_SIGMA = 4e-6, 20.0   # m and max sigma / (omega eps0) of its cubic stretch
_LOSS_REFERENCE_MODES = 12
_LOSS_INDEX_TOLERANCE = 1e-5                      # |Re n_eff - reference|
_LOSS_RELATIVE_TOLERANCE = 1e-2                   # |loss - reference| / reference
_DB_PER_NEPER = 20.0 / math.log(10.0)


def _loss_geometry():
    """(size_x, core x0, core x1, reference x = core centre) [m] of the bend-loss slab."""
    core_x0, core_x1 = _LOSS_INNER, _LOSS_INNER + _LOSS_CORE_WIDTH
    return core_x1 + _LOSS_OUTER, core_x0, core_x1, 0.5 * (core_x0 + core_x1)


def _bend_loss_spec():
    """em_modes mode_analysis of the bent slab: one periodic 20 nm y cell, pec x_lo (plane at x = one cell), cpml x_hi
    (100 cells beyond size_x, pml_kappa_max 1: the leaky-mode setting), bend_model exact about the core centre, cpu
    backend (the solve is host code on both)."""
    size_x, core_x0, core_x1, reference_x = _loss_geometry()
    settings = {'bc_x_lo': 'pec', 'bc_x_hi': 'cpml', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic',
                'pml_cells': _LOSS_PML_CELLS, 'pml_kappa_max': 1.0, 'wavelength': _WAVELENGTH, 'n_modes': _LOSS_N_MODES,
                'n_eff_guess': 0.5 * (_LOSS_CORE_INDEX + _LOSS_CLADDING_INDEX), 'bend_radius': _LOSS_RADIUS,
                'bend_reference_x': reference_x, 'bend_model': 'exact'}
    spec = ModelSpec(
        title='em_modes dielectric bend loss',
        domain=DomainSpec(size_x=size_x, size_y=_LOSS_CELL),
        geometry=[GeometrySpec('core', 'rectangle', {'x0': core_x0, 'y0': 0.0, 'x1': core_x1, 'y1': _LOSS_CELL})],
        materials=[MaterialSpec('cladding', properties={'eps_r': _LOSS_CLADDING_INDEX ** 2}, selection=['all']),
                   MaterialSpec('core_glass', properties={'eps_r': _LOSS_CORE_INDEX ** 2}, selection=['core'])],
        physics=[PhysicsSpec(_LOSS_PHYSICS, _LOSS_TYPE, settings=settings,
                             features=[FeatureSpec(_LOSS_TABLE, 'mode_table')])],
        mesh=MeshSpec(mode='uniform', dx=_LOSS_CELL),
        study=StudySpec(type='mode_analysis'))
    spec.solver_defaults.backend = 'cpu'
    return spec


def _reference_bend_loss_mode(omega):
    """(n_eff, core share of |E_y|^2) of the guided leaky E_y mode of the bent slab from an independent fine 1D solve:
    the exact transform u = x_ref + R ln(r / R) (psi'' + (k0^2 eps s^2 - beta^2) psi = 0, s = r / R), cell-centred
    finite volumes of _LOSS_REFERENCE_CELL, the model's pec plane, a 4 um cubic complex stretch 1 - j sigma' beyond the
    model's cpml start with s taken at the stretched coordinate u~ (the matched continuation of the growing index) and
    psi = 0 at its end; scipy shift-invert, the mode of largest core share."""
    import scipy.sparse
    import scipy.sparse.linalg
    size_x, core_x0, core_x1, reference_x = _loss_geometry()

    def u_of(x):
        return reference_x + _LOSS_RADIUS * math.log1p((x - reference_x) / _LOSS_RADIUS)

    bounds = [u_of(_LOSS_CELL), u_of(core_x0), u_of(core_x1), u_of(size_x)]
    pml_start = bounds[-1]
    bounds.append(pml_start + _LOSS_REFERENCE_PML)
    pieces = [np.array([bounds[0]])]
    for lo, hi in zip(bounds[:-1], bounds[1:]):   # piecewise uniform: core edges and the stretch start on cell edges
        pieces.append(np.linspace(lo, hi, int(math.ceil((hi - lo) / _LOSS_REFERENCE_CELL)) + 1)[1:])
    edges = np.concatenate(pieces)
    widths, centres = np.diff(edges), 0.5 * (edges[:-1] + edges[1:])
    in_core = (centres > bounds[1]) & (centres < bounds[2])
    eps = np.where(in_core, _LOSS_CORE_INDEX ** 2, _LOSS_CLADDING_INDEX ** 2)

    def depth_of(u):
        return np.clip((np.asarray(u, dtype=np.float64) - pml_start) / _LOSS_REFERENCE_PML, 0.0, None)

    def stretch(u):
        return 1.0 - 1j * _LOSS_REFERENCE_SIGMA * depth_of(u) ** 3

    def stretched(u):
        """u~ = u + integral (stretch - 1) du of the cubic profile: the complex coordinate the wave sees."""
        return u - 0.25j * _LOSS_REFERENCE_SIGMA * _LOSS_REFERENCE_PML * depth_of(u) ** 4

    k0 = omega / C0
    inner_conductance = 1.0 / (stretch(edges[1:-1]) * np.diff(centres))           # faces between cells i, i + 1
    lo_conductance = 1.0 / (stretch(edges[0]) * 0.5 * widths[0])                   # psi = 0 on both end faces
    hi_conductance = 1.0 / (stretch(edges[-1]) * 0.5 * widths[-1])
    below = np.concatenate(([lo_conductance], inner_conductance))
    above = np.concatenate((inner_conductance, [hi_conductance]))
    row_scale = 1.0 / (widths * stretch(centres))
    scale = np.exp((stretched(centres) - reference_x) / _LOSS_RADIUS)
    diagonal = -(below + above) * row_scale + k0 ** 2 * eps * scale ** 2
    matrix = scipy.sparse.diags([inner_conductance * row_scale[1:], diagonal, inner_conductance * row_scale[:-1]],
                                [-1, 0, 1], format='csc')
    shift = (k0 * 0.5 * (_LOSS_CORE_INDEX + _LOSS_CLADDING_INDEX)) ** 2
    values, vectors = scipy.sparse.linalg.eigs(matrix, k=_LOSS_REFERENCE_MODES, sigma=shift, which='LM')
    weight = np.abs(vectors) ** 2 * widths[:, None]
    outside_pml = centres < pml_start
    core_share = np.sum(weight[in_core], axis=0) / np.sum(weight[outside_pml], axis=0)
    pml_share = np.sum(weight[~outside_pml], axis=0) / np.sum(weight, axis=0)
    best = int(np.argmax(np.where(pml_share < 0.5, core_share, -1.0)))   # PML modes never picked
    return complex(np.sqrt(complex(values[best])) / k0), float(core_share[best])


def _check_bend_loss(results_by_name):
    """Re n_eff and loss_db_per_cm of the E_y mode nearest the reference; error = the larger of both errors in units
    of their tolerances (+ failed checks), tolerance 1."""
    omega = _omega()
    reference, core_share = _reference_bend_loss_mode(omega)
    reference_loss = -_DB_PER_NEPER * (omega / C0) * reference.imag / 100.0
    table = results_by_name['bend_loss'].result(_LOSS_PHYSICS, _LOSS_TABLE)
    column = {name: index for index, name in enumerate(table['columns'])}
    candidates = [row for row in table['rows'] if row[column['te_fraction']] < 0.5]
    violations = []
    if not core_share > 0.5:
        violations.append(f'reference mode core share {core_share:.3g} <= 0.5 (no guided mode picked)')
    if not reference_loss > 0.0:
        violations.append(f'reference loss {reference_loss:.3g} dB/cm is not positive')
    if not candidates:
        return {'measured': math.nan, 'expected': reference_loss, 'error': math.inf, 'tolerance': 1.0,
                'details': {'violations': violations + ['no E_y mode (te_fraction < 0.5) in the mode table'],
                            'reference_n_eff': [reference.real, reference.imag]}}
    row = min(candidates, key=lambda values: abs(complex(values[column['n_eff']], values[column['n_eff_imag']])
                                                 - reference))
    index_error = abs(row[column['n_eff']] - reference.real)
    loss_error = abs(row[column['loss_db_per_cm']] - reference_loss) / abs(reference_loss)
    error = max(index_error / _LOSS_INDEX_TOLERANCE, loss_error / _LOSS_RELATIVE_TOLERANCE)
    return {'measured': row[column['loss_db_per_cm']], 'expected': reference_loss, 'error': error + len(violations),
            'tolerance': 1.0,
            'details': {'n_eff': [row[column['n_eff']], row[column['n_eff_imag']]],
                        'reference_n_eff': [reference.real, reference.imag], 'reference_core_share': core_share,
                        'n_eff_error': index_error, 'loss_relative_error': loss_error,
                        'pml_fraction': row[column['pml_fraction']], 'violations': violations}}


# Straight-to-bend junction (model runs on the cpu backend, feature bend_overlap): the em_cross_section_bend slab on a
# uniform 20 nm grid (core 0.8 um, pec walls 2 um from it, one periodic y cell), bend_model exact about the core
# centre at R = 10 and 20 um. Reference: the same cross-section solved bent and straight by cross_section_modes in the
# check (float32 eps as the MaterialGrid holds), mode_overlap of each bent mode with its best straight mode; physics:
# the mismatch 1 - overlap of the fundamental mode scales as 1 / R^2 (outward mode shift ~ 1 / R).
_JUNCTION_CELL = 20e-9                            # m, uniform; core edges on cell edges
_JUNCTION_CLADDING = 2.0e-6                       # m between the core and each pec wall
_JUNCTION_RADII = {'r10': 10e-6, 'r20': 20e-6}    # m
_JUNCTION_MODES = 4
_JUNCTION_TOLERANCE = 1e-6                        # |overlap, n_eff (model) - reference|
_JUNCTION_RATIO = (3.0, 5.0)                      # (1 - overlap)(R) / (1 - overlap)(2 R) of mode 1: ~4


def _junction_layout():
    """(size_x, core x0, core x1) [m] of the junction slab."""
    core_x0 = _JUNCTION_CLADDING
    return 2.0 * _JUNCTION_CLADDING + _CORE_WIDTH, core_x0, core_x0 + _CORE_WIDTH


def _junction_spec(radius):
    """em_modes mode_analysis of the bent slab with a bend_overlap feature, cpu backend."""
    size_x, core_x0, core_x1 = _junction_layout()
    settings = {'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic',
                'wavelength': _WAVELENGTH, 'n_modes': _JUNCTION_MODES, 'bend_radius': radius,
                'bend_reference_x': 0.5 * (core_x0 + core_x1), 'bend_model': 'exact'}
    spec = ModelSpec(
        title=f'em_modes straight-to-bend junction (R = {radius * 1e6:g} um)',
        domain=DomainSpec(size_x=size_x, size_y=_JUNCTION_CELL),
        geometry=[GeometrySpec('core', 'rectangle', {'x0': core_x0, 'y0': 0.0, 'x1': core_x1,
                                                     'y1': _JUNCTION_CELL})],
        materials=[MaterialSpec('cladding', properties={'eps_r': _CLADDING_INDEX ** 2}, selection=['all']),
                   MaterialSpec('core_glass', properties={'eps_r': _CORE_INDEX ** 2}, selection=['core'])],
        physics=[PhysicsSpec(_LOSS_PHYSICS, _LOSS_TYPE, settings=settings,
                             features=[FeatureSpec('junction', 'bend_overlap', [],
                                                   {'n_straight_modes': _JUNCTION_MODES})])],
        mesh=MeshSpec(mode='uniform', dx=_JUNCTION_CELL),
        study=StudySpec(type='mode_analysis'))
    spec.solver_defaults.backend = 'cpu'
    return spec


def _junction_reference(radius):
    """[(n_eff bent, best straight rank, overlap)] per bent mode of the scripted cross-section."""
    from . import cross_section_modes
    from .boundaries import BoundaryConfig
    size_x, core_x0, core_x1 = _junction_layout()
    n_cells = int(round(size_x / _JUNCTION_CELL))
    dx = np.full(n_cells, _JUNCTION_CELL)
    x_edges = np.concatenate(([0.0], np.cumsum(dx)))
    centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    eps = np.where((centres > core_x0) & (centres < core_x1), np.float32(_CORE_INDEX ** 2),
                   np.float32(_CLADDING_INDEX ** 2)).astype(np.complex128)
    materials = _isotropic(cross_section_modes, eps[:, None])
    boundary = BoundaryConfig('pec', 'pec', 'periodic', 'periodic')
    solves = [cross_section_modes.solve_cross_section_modes(
        (dx, np.array([_JUNCTION_CELL])), boundary, _JUNCTION_CELL, materials, _omega(), _JUNCTION_MODES, bend=bend)
        for bend in ((radius, 0.5 * (core_x0 + core_x1), x_edges, 'exact'), None)]
    rows = []
    for bent in solves[0].modes:
        overlaps = [cross_section_modes.mode_overlap(bent, straight) for straight in solves[1].modes]
        best = int(np.argmax(overlaps))
        rows.append((complex(bent.n_eff), best + 1, float(overlaps[best])))
    return rows


def _check_junction(results_by_name):
    """Model table (n_eff, straight_mode, overlap, junction_loss_db) vs the scripted reference per radius, and the
    1 / R^2 scaling of the fundamental mismatch; error = worst deviation + failed checks."""
    errors, mismatch, violations, details = {}, {}, [], {}
    for run, radius in _JUNCTION_RADII.items():
        table = results_by_name[run].result(_LOSS_PHYSICS, 'junction')
        column = {name: index for index, name in enumerate(table['columns'])}
        reference = _junction_reference(radius)
        rows = table['rows']
        if len(rows) != len(reference):
            violations.append(f'{run}: {len(rows)} table rows, reference {len(reference)} modes')
        for number, (row, (n_eff, best, overlap)) in enumerate(zip(rows, reference), start=1):
            errors[f'{run}_mode{number}_n_eff'] = abs(complex(row[column['n_eff']], row[column['n_eff_imag']]) - n_eff)
            errors[f'{run}_mode{number}_overlap'] = abs(row[column['overlap']] - overlap)
            errors[f'{run}_mode{number}_loss_db'] = abs(row[column['junction_loss_db']] + 10.0 * math.log10(overlap))
            if row[column['straight_mode']] != best:
                violations.append(f'{run} mode {number}: straight_mode {row[column["straight_mode"]]}, reference {best}')
        if rows:
            mismatch[run] = 1.0 - rows[0][column['overlap']]
        details[run] = [list(row) for row in rows]
    ratio = math.nan
    if len(mismatch) == len(_JUNCTION_RADII):
        small, large = mismatch['r10'], mismatch['r20']
        ratio = small / large if large > 0.0 else math.inf
        if not _JUNCTION_RATIO[0] <= ratio <= _JUNCTION_RATIO[1]:
            violations.append(f'mismatch ratio {ratio:.3g} for R doubled is outside {_JUNCTION_RATIO} (1 / R^2)')
    measured = max(errors.values(), default=math.nan)
    return {'measured': measured, 'expected': 0.0, 'error': measured + len(violations),
            'tolerance': _JUNCTION_TOLERANCE,
            'details': {'errors': errors, 'mismatch': mismatch, 'mismatch_ratio': ratio, 'tables': details,
                        'violations': violations}}


BEND_SELFTEST_CASES = (
    SelfTestCase(
        name='em_cross_section_bend',
        label='EM cross-section modes: bend (exact and conformal)',
        description='CPU check of the em_modes bend (cross_section_modes bend, models exact and conformal, no model '
                    'run): a y-invariant slab (one periodic 50 nm y cell, core n = 2.0, 0.8 um wide along x in n = '
                    '1.444, pec walls 3 um from the core, 5 nm core cells, cladding graded x1.05 up to 20 nm) bent '
                    'about the y axis with the reference x at the core centre, 1.55 um. The E_y (te_fraction 0 to '
                    '1e-10) and E_x (te_fraction 1) families decouple; at R = 20 um each reduces to a weighted 1D '
                    'line with s = 1 + (x - x_ref)/R at each point: E_y on v = mu_xx H_x = slab_modes ez with cell '
                    'weight mu_xx, face weight 1/mu_zz, index eps_yy mu_xx; E_x on H_y = a slab_modes line on the hi-x '
                    'faces (wall-plane faces halved, neumann ends) with cell weight eps_xx, face weight 1/eps_zz, index '
                    'eps_xx mu_yy (exact: eps, mu x (s, s, 1/s); conformal: eps x s^2, mu 1). TE0 / TE1 n_eff of both '
                    'families and models must equal them to round-off (tolerance 1e-9). Continuum check: the exact '
                    'bent slab is the straight equation in u = x_ref + R ln(r/R) with index n(r) r/R (exact for E_y, '
                    'same physical walls): the exact model must agree to 2e-5 (discretisation only) at 20 and 40 um; '
                    'the conformal map, second order in 1/R, below 2e-3 at 20 um and falling by >= 3x from 20 to 40 '
                    'um (first order would give 2x). error = max operator |n_eff error| + the number of failed checks.',
        specs=lambda: {},
        check=_check_bend,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_shg_overlap_bend',
        label='EM modes SHG overlap: bend (exact model)',
        description='CPU check of the bend-mapped shg_overlap (nonlinear_overlap.bend_overlap_factors, no model run) '
                    'on the em_cross_section_bend slab at R = 20 um, chi2 on the outer half of the core (x > x_ref), '
                    'pump 1.55 um, harmonic 775 nm, fundamental mode of each family: E_y with d22, E_x with d11, d13 '
                    'and d31 one at a time. Reference: the exact transform u = x_ref + R ln(r/R) solved as a straight '
                    'cross-section (eps diag(1, s^2, 1), mu_yy = s^2 for the E_x family, s = r/R), its overlap of the '
                    'physical fields E = (E_u / s, E_y, E_v / s) over dV = s^2 du dy dz. The exact-model overlap '
                    '(dV = s dx dy dz, E_phi = E_z / s) must agree to 1e-3 relative in |kappa| (discretisation); the '
                    'unmapped overlap must miss d22, d11, d13 by >= 3e-3 (sensitivity). The conformal model is '
                    'reported, not toleranced. error = max |kappa| error + the number of failed checks.',
        specs=lambda: {},
        check=_check_shg_bend,
        max_wall_time=60.0),
    SelfTestCase(
        name='em_modes_bend_loss',
        label='EM modes: dielectric bend loss (model run)',
        description='em_modes mode_analysis on the cpu backend of a weakly guiding y-invariant slab (core n = 1.50, 2 '
                    'um wide along x, in '
                    'n = 1.45; one periodic 20 nm y cell, 20 nm cells) bent at R = 200 um about the core centre '
                    '(bend_model exact), pec 4 um inside, cpml (100 cells, pml_kappa_max 1) 7 um outside the core, '
                    '1.55 um. No closed form exists (the outgoing field is a Hankel function of large complex order; '
                    'Marcuse-type formulas are only ~10-30 %), so the reference is an independent fine solve computed '
                    'in the check: the exact transform u = x_ref + R ln(r/R) of the E_y family, psi\'\' + (k0^2 eps '
                    '(r/R)^2 - beta^2) psi = 0, cell-centred finite volumes of 2 nm, the same pec plane, a 4 um cubic '
                    'complex stretch (max sigma / (omega eps0) 20) from the cpml start, scipy shift-invert, the mode '
                    'of largest core share. Both solves take the bend factor r/R at the complex stretched coordinate '
                    'inside their absorbing layer (the matched continuation of the growing index: evaluated at the '
                    'real coordinate the layer reflects a few % of the nearly grazing radiation, which once gave a '
                    '4 % loss error). The E_y mode (te_fraction < 0.5) nearest the reference must match Re n_eff to '
                    '1e-5 and loss_db_per_cm to 1e-2 relative (the Marcuse estimate is ~16 dB/cm). error = max(n_eff '
                    'error / 1e-5, loss error / 1e-2) + failed checks, tolerance 1. Not yet run on the cpu backend.',
        specs=lambda: {'bend_loss': _bend_loss_spec()},
        check=_check_bend_loss),
    SelfTestCase(
        name='em_modes_bend_overlap',
        label='EM modes: straight-to-bend junction overlap (model run, cpu)',
        description='em_modes mode_analysis on the cpu backend with the bend_overlap feature: the '
                    'em_cross_section_bend slab (core n = 2.0, 0.8 um wide along x, in n = 1.444, one periodic y cell) '
                    'on a uniform 20 nm grid, pec walls 2 um from the core, bend_model exact about the core centre at '
                    'R = 10 and 20 um, 4 modes, 1.55 um. Reference computed in the check: the same cross-section (the '
                    'float32 eps of the MaterialGrid) solved bent and straight by cross_section_modes, each bent mode '
                    'paired with the straight mode of largest mode_overlap. Expected: the table n_eff, straight_mode, '
                    'overlap and junction_loss_db = -10 log10(overlap) equal the reference (1e-6), and the fundamental '
                    'mismatch 1 - overlap falls by 3-5x from R = 10 to 20 um (1 / R^2: the outward mode shift ~ 1 / R). '
                    'error = the worst deviation + the number of failed checks.',
        specs=lambda: {run: _junction_spec(radius) for run, radius in _JUNCTION_RADII.items()},
        check=_check_junction),
)

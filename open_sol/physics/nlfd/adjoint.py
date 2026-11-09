"""Adjoint dJ/d eps of the NLFD channels (nonlinear_optics_plan stage 5 (b)): J = the adjoint_objective terms of every
channel; undepleted one transposed solve per channel in reverse solve order, pump_depleted the transposed fixed point
by real GMRES; the nonlinear sources back-propagated through the exact transposes of their polynomial and C / C*."""
# Cascade (undepleted, plan 5.1): A_k u_k = b_k + r_k(x_<k), x_k = the store fields of channel k (ez E_z = the filled
# solution; hz E_x / E_y = face_fields = scale W (D_h u + J), J incl. the channel's own nonlinear face currents), r_k
# the rhs of P_k(x_<k) (ez -omega^2 mu0 P_z; hz face_current_rhs of J = -j omega C* P). With dJ = 2 Re(c^T dx) (c
# the cotangents, unconjugated, as physics/em/adjoint.py), per channel k in reverse solve order: its cotangent = the
# objectives' (cell, face parts) + what higher channels' sources sent to x_k; A_k^T lam_k = g_k (adjoint_source); dJ/d
# eps += -2 Re(lam_k^T dA_k u_k) (hz incl. the W J rhs term and the W of the stored face fields); then the cotangent of
# P_k = (dr_k / dP)^T lam_k + (dt / dJ)(dJ / dP) c_k goes through P's Wirtinger transpose (conj slots conjugated) onto
# the centre fields of its source channels and through C^T onto their store fields. Lower-ranked channels never read a
# higher one, so one reverse pass is exact. Held fixed (as the linear adjoint): source calibrations, TFSF / port incident
# fields and port modes, the objectives' own face weights, the chi3 n0^2 cell factors and Miller factors of the build.
# Pump-depleted: the converged store x = G_J(x) (block Jacobi map, every channel solved from x with the plain channel
# operator; picard's Gauss-Seidel / Anderson reach the same fixed point). With T(c) = the cotangents one transposed
# sweep of G_J sends back onto x for output cotangents c (the per-channel step above, every term incl. SPM / XPM and
# back-action onto any channel), the unrolled iteration gives nu = T(c_obj + nu): (I - T) nu = T(c_obj) by real
# GMRES (T is real-linear: conj slots), then one gradient sweep with c_obj + nu. Refused: picard Kerr / carrier
# operator terms (the factors hold the lagged eps, not the fixed point's operator; newton has none) and eps_xy.
# Sheets: the cotangent of the deposited density (ez the centre one, hz the face one without C*) goes through
# sheets.sheet_cotangents straight onto the store layout (the sites sample it, no C). Free carriers (pump_depleted,
# newton): carrier_cotangents, onto the target directly and through N(G) onto every Kerr channel; the masks G > 0 and
# the n0 factors are held fixed (the derivative where G > 0, as the Newton tangent).
# Backend-neutral: arrays on the store's module (nl.store.xp, cupy or numpy); on the cpu it runs once the em adjoint
# pieces imported below (objective_cotangents, adjoint_source, permittivity_gradients, sensitivity_density) take numpy.
import numpy as np

from ...core.backend import array_module_of
from ...core.constants import EPS0, HBAR, MU0
from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS
from ..em.adjoint import (EpsSensitivity, adjoint_source, objective_cotangents, permittivity_gradients,
                          sensitivity_density)
from ..em.harmonic import face_mean_at_centres
from ..em.operators import stretched_cell_areas
from ..em.yee_grid import cell_row, cpml_grading_cell_size, grid_cell_widths
from .iteration import _scale_blocks, real_gmres
from .mixing import solve_order
from .sheets import sheet_cotangents
from .sources import centres_to_faces

ADJOINT_MAX_PRODUCTS = 200       # transposed sweeps (N adjoint solves each) of the pump-depleted adjoint GMRES
ADJOINT_FAIL_RESIDUAL = 1e-4     # relative GMRES residual above which the pump-depleted gradient is refused
# relative ||A^T x - b|| / ||b|| a transposed solve through the symmetrizer must reach (first estimate: forward SuperLU /
# refined cuDSS solves reach ~1e-12; D spans the cell areas x CPML stretches, which may cost a few digits)
SYMMETRIZED_TRANSPOSE_RESIDUAL = 1e-8
# the MUMPS / PARDISO back ends as error-text words (their factors solve A^T natively, as SuperLU)
_TRANSPOSING_HOST_BACK_ENDS = ' / '.join(f"'{name}'" for name in HOST_DIRECT_BACK_ENDS)


def channel_symmetrizer(physics):
    """Host complex128 flat stretched cell areas D of a Yee channel FDFDPhysics (D A complex symmetric, no Bloch
    phase): as FDFDPhysics._eigen_symmetrizer, whatever the eigen solver."""
    domain = physics.domain
    cell_widths = grid_cell_widths(domain.grid)
    polar = physics.metric is not None
    pml_dx = float(cell_widths[0][-1]) if polar else cpml_grading_cell_size(domain)
    return stretched_cell_areas(cell_widths, physics.boundary_config, pml_dx, physics.system.omega,
                                polar_radius_min=domain.coordinates.radius_min if polar else None)


def transposed_channel_solve(physics, flat_rhs):
    """x of A^T x = rhs (unconjugated) for a channel's last system: its factor's transposed solve, or for a factor
    without one (NotImplementedError, e.g. cuDSS) x = D A^-1 D^-1 rhs by a forward solve on that factor, checked."""
    try:
        return physics.solve_adjoint(flat_rhs)
    except NotImplementedError as exc:
        missing = exc
    if physics.boundary_config.has_bloch_phase:
        raise NotImplementedError(f'{missing} (a channel with a Bloch phase has no symmetrizer: D A is not complex '
                                  'symmetric there)') from missing
    xp = physics.xp
    rhs = xp.asarray(flat_rhs, dtype=xp.complex128).ravel()
    symmetrizer = xp.asarray(channel_symmetrizer(physics))
    solution = symmetrizer * physics._solve_linear(rhs / symmetrizer)
    rhs_norm = float(xp.linalg.norm(rhs))
    if rhs_norm > 0.0:
        residual = float(xp.linalg.norm(physics.system.operator.T @ solution - rhs)) / rhs_norm
        if not residual <= SYMMETRIZED_TRANSPOSE_RESIDUAL:   # also NaN
            raise RuntimeError(f'em_fdfd_nonlinear adjoint: the transposed solve through the symmetrizer D misses '
                               f'{SYMMETRIZED_TRANSPOSE_RESIDUAL:g} (relative residual {residual:.3g}: D A is not '
                               f'complex symmetric for this channel, {missing}); use the sparse_direct back end '
                               f"'scipy_splu' (or {_TRANSPOSING_HOST_BACK_ENDS}: transposed solves on the host) for "
                               "the 'linear' role")
    return solution


def collocation_transpose(centre_values, axis, periodic, wrap_factor=1.0):
    """Plain transpose C^T of harmonic.face_mean_at_centres(wrap_factor=) along `axis` (centre cotangent -> hi faces;
    the last face takes cell 0 times wrap_factor)."""
    return centres_to_faces(centre_values, axis, periodic, np.ones(centre_values.shape[axis]),
                            wrap_factor=complex(wrap_factor).conjugate())


def distribution_transpose(face_values, axis, periodic, cell_widths, wrap_factor=1.0):
    """Plain transpose of sources.centres_to_faces(wrap_factor=) (C* = dual^-1 C^T w) along `axis`: w C(f / dual) at
    the centres, C with the conjugate wrap factor."""
    xp = array_module_of(face_values)
    widths = np.asarray(cell_widths, dtype=np.float64)
    shape = [1, 1]
    shape[axis] = widths.size
    dual_widths = xp.asarray((0.5 * (widths + np.roll(widths, -1))).reshape(shape))
    return face_mean_at_centres(face_values / dual_widths, axis, periodic, xp,
                                wrap_factor=complex(wrap_factor).conjugate()) * xp.asarray(widths.reshape(shape))


def polarization_cotangents(terms, centre_fields, density_cotangent, media, shape, xp):
    """{source channel: [E_x, E_y, E_z centre cotangents] (None = 0)} of P(terms) for dJ = 2 Re(c . dP): product rule
    per slot, a conjugated slot conjugated."""
    result = {}
    for term in terms:
        fields = [centre_fields[index] for index, _ in term.sources]
        medium = media[term.medium]
        box = medium.box
        cell_factor = medium.cell_factor.astype(xp.complex128)
        for component, source_components, value in term.components:
            cotangent = density_cotangent.get(component)
            if cotangent is None:
                continue
            scale = cell_factor * (EPS0 * term.factor * value) * cotangent[box]
            for slot, ((index, conjugate), source_component) in enumerate(zip(term.sources, source_components)):
                product = scale
                for position, ((_, other_conjugate), field, other_component) in enumerate(
                        zip(term.sources, fields, source_components)):
                    if position != slot:
                        values = field[other_component][box]
                        product = product * (xp.conj(values) if other_conjugate else values)
                parts = result.setdefault(index, [None, None, None])
                if parts[source_component] is None:
                    parts[source_component] = xp.zeros(shape, dtype=xp.complex128)
                parts[source_component][box] += xp.conj(product) if conjugate else product
    return result


def _density_cotangent(physics, adjoint, face_cotangents, store, index):
    """({component: cotangent of channel `index`'s bulk centre P}, {component: cotangent of its sheet grid density})
    from its rhs (lam) and, hz, from the face fields holding its face currents (face_cotangents [hi-y, hi-x] of
    objectives and store readers); ez the sheets' centre density shares the bulk one, hz the sheets' face density
    skips C*."""
    system = physics.system
    omega = system.omega
    if system.polarization == 'ez':   # rhs -omega^2 mu0 P_z
        centre = {2: (-(omega ** 2) * MU0) * adjoint}
        return centre, centre
    ops = system.ops
    flat = adjoint.ravel()
    shape = adjoint.shape
    curl_scale = 1.0 / (1j * omega * EPS0)   # face fields: E_x = s (D_h u + J_x) W_y, E_y = -s (D_h u - J_y) W_x
    # rhs = dx_e (J_y W_x) - dy_e (J_x W_y)
    current_x = -system.weight_hi_y_faces * (ops.dy_e.T @ flat).reshape(shape)
    current_y = system.weight_hi_x_faces * (ops.dx_e.T @ flat).reshape(shape)
    if face_cotangents[0] is not None:
        current_x += curl_scale * system.weight_hi_y_faces * face_cotangents[0]
    if face_cotangents[1] is not None:
        current_y += curl_scale * system.weight_hi_x_faces * face_cotangents[1]
    scale = -1j * omega   # J = -j omega (C* P + P_sheets)
    wrap_x, wrap_y = store.wrap_factors[index]
    return ({0: scale * distribution_transpose(current_x, 1, store.periodic_y, store.cell_dy, wrap_y),
             1: scale * distribution_transpose(current_y, 0, store.periodic_x, store.cell_dx, wrap_x)},
            {0: scale * current_x, 1: scale * current_y})


def _add_part(result, index, component, values):
    """result[index][component] += values (a [None] * 3 list per channel index created on demand)."""
    parts = result.setdefault(index, [None, None, None])
    parts[component] = values if parts[component] is None else parts[component] + values


def carrier_cotangents(free_carriers, target, omega, fields, density_cotangent, media, shape, xp, result):
    """result {channel: [E_x, E_y, E_z] centre cotangents} += the transpose of channel `target`'s free-carrier
    polarisation eps0 n0 f(omega) N E_target (dJ = 2 Re(c . dP)): directly onto E_target, and through N = tau G (0
    where G <= 0), G = -(1 / 4 hbar) Im sum_m conj(E_m) . P_kerr,m, onto every Kerr channel m and its sources."""
    field = fields.get(target)
    if field is None:
        return result
    states = free_carriers.medium_states(fields)
    for medium in free_carriers.media:
        carriers, positive, densities = states[medium.name]
        box = medium.box
        factor = EPS0 * medium.carriers.permittivity_factor(omega)
        coupling = xp.zeros(carriers.shape, dtype=xp.complex128)   # sum_c c_P,c eps0 n0 f E_c: dJ / dN = 2 Re
        for component, cotangent in density_cotangent.items():
            if cotangent is None or field[component] is None:
                continue
            local = (cotangent[box] * medium.carriers.index) * factor   # complex128 first (float32 x complex: c64)
            full = xp.zeros(shape, dtype=xp.complex128)
            full[box] = carriers * local
            _add_part(result, target, component, full)
            coupling += local * field[component][box]
        weight = (2.0 * medium.carriers.lifetime) * xp.where(positive, coupling.real, 0.0)   # h = dJ / dG
        del coupling
        for kerr_target, density in densities.items():
            kerr_field = fields[kerr_target]
            kerr_cotangent = {}   # c of P_kerr,m = j h conj(E_m) / (8 hbar)
            for component, values in density.items():
                full = xp.zeros(shape, dtype=xp.complex128)
                full[box] = (-0.125j / HBAR) * weight * xp.conj(values)   # c of E_m = -j h conj(P_m) / (8 hbar)
                _add_part(result, kerr_target, component, full)
                full = xp.zeros(shape, dtype=xp.complex128)
                full[box] = (0.125j / HBAR) * weight * xp.conj(kerr_field[component][box])
                kerr_cotangent[component] = full
            terms = [term for term in free_carriers.terms[medium.name] if term.target == kerr_target]
            for index, parts in polarization_cotangents(terms, fields, kerr_cotangent, media, shape, xp).items():
                for component, values in enumerate(parts):
                    if values is not None:
                        _add_part(result, index, component, values)
    return result


def _add(parts, index, values):
    """parts[index] += values (None = 0); returns parts."""
    if values is not None:
        parts[index] = values if parts[index] is None else parts[index] + values
    return parts


def _zero_pinned(adjoint, physics):
    """lam = 0 on PEC cells and outer wall / mirror rows (already so up to round-off: pinned rows transpose to I)."""
    if physics.pec_cells is not None:
        physics.xp.copyto(adjoint, adjoint.dtype.type(0), where=physics.pec_cells)
    for axis, outer, _, _, _ in physics._wall_rows:
        cell_row(adjoint, axis, outer)[...] = 0
    return adjoint


def _channel_backward(nl, index, cell_part, objective_faces, received, media, gradients, outgoing):
    """Transposed solve of channel `index` for its output cotangents (objective cell / face parts + `received` store
    layout, None = 0): gradients (dict, None = skip) += -2 Re(lam^T dA u), and the cotangents its nonlinear sources
    send to their source channels' store fields are added to outgoing {index: [store-layout parts]}."""
    shape = nl.store.shape
    physics = nl.channel_physics[index]
    ez = physics.polarization == 'ez'
    if ez:
        cell_part = cell_part if received is None else _add([cell_part], 0, received[0])[0]
        face_parts = objective_faces
    else:
        face_parts = list(objective_faces)
        for face in (0, 1):
            if received is not None:
                _add(face_parts, face, received[face])
    if cell_part is None and all(part is None for part in face_parts):
        return   # nothing depends on this channel
    adjoint = transposed_channel_solve(physics, adjoint_source(physics, cell_part, face_parts, physics._wall_rows))
    adjoint = _zero_pinned(adjoint.reshape(shape), physics)
    phasor = nl._sources.get(index)
    if gradients is not None:
        currents = phasor.face_currents(physics, physics.system) if phasor is not None and not ez else None
        channel_gradients = permittivity_gradients(physics, adjoint, physics._wall_rows, face_currents=currents,
                                                   face_cotangents=None if ez or received is None else received)
        for name, values in channel_gradients.items():
            if name in gradients:
                gradients[name] += values
    if phasor is None or not (phasor.bulk_terms or phasor.sheet_terms or phasor.free_carriers is not None):
        return
    store, xp = nl.store, nl.store.xp
    density_cotangent, sheet_cotangent = _density_cotangent(physics, adjoint, face_parts, store, index)
    fields = {source: store.centre_vector(source) for source in phasor.source_indices}
    centre_parts = polarization_cotangents(phasor.bulk_terms, fields, density_cotangent, media, shape, xp)
    if phasor.free_carriers is not None:
        carrier_cotangents(phasor.free_carriers, index, nl.channels[index].omega, fields, density_cotangent, media,
                           shape, xp, centre_parts)
    for source, parts in centre_parts.items():
        target = outgoing.setdefault(source, [None, None] if nl.channels[source].polarization == 'hz' else [None])
        if nl.channels[source].polarization == 'ez':
            _add(target, 0, parts[2])
        else:   # E_x on the hi-y faces (C along y), E_y on the hi-x faces (C along x); the source's wrap factors
            wrap_x, wrap_y = store.wrap_factors[source]
            if parts[0] is not None:
                _add(target, 0, collocation_transpose(parts[0], 1, store.periodic_y, wrap_y))
            if parts[1] is not None:
                _add(target, 1, collocation_transpose(parts[1], 0, store.periodic_x, wrap_x))
    if phasor.sheet_terms:   # sites sample the store layout itself (no C)
        grids = {source: store.grid_fields(source) for source in range(len(nl.channels))}
        sheets = {sheet.name: sheet for sheet in nl.sheets}
        for source, parts in sheet_cotangents(phasor.sheet_terms, grids, sheets, phasor.components,
                                              nl.channels[index].omega, sheet_cotangent, shape, xp).items():
            target = outgoing.setdefault(source, [None, None] if nl.channels[source].polarization == 'hz' else [None])
            if nl.channels[source].polarization == 'ez':
                _add(target, 0, parts[2])
            else:
                _add(target, 0, parts[0])
                _add(target, 1, parts[1])


def _check_differentiable(nl):
    """NotImplementedError for what has no transpose here: an off-diagonal eps_xy, and pump_depleted picard Kerr /
    carrier operator terms (their lagged operator is not the fixed point's)."""
    if any(physics.system.off_diagonal is not None for physics in nl.channel_physics):
        raise NotImplementedError('em_fdfd_nonlinear adjoint sensitivity: an off-diagonal eps_xy is not '
                                  'differentiated through the nonlinear face currents; remove eps_xy')
    if nl.coupling_mode == 'pump_depleted' and nl._kerr_terms:
        raise NotImplementedError("em_fdfd_nonlinear adjoint sensitivity: pump_depleted picard with Kerr terms or free "
                                  "carriers keeps a lagged operator in the channel factors; use iteration_method "
                                  "'newton'")


def cascade_sensitivity(nonlinear_physics):
    """EpsSensitivity of every channel's objectives at the last solve (gradients summed over channels): undepleted one
    reverse pass, pump_depleted the fixed-point adjoint (depleted_cotangents); NotImplementedError (_check_
    differentiable) when an objective is weighted."""
    nl = nonlinear_physics
    shape, xp = nl.store.shape, nl.store.xp
    reference_omega = float(nl.reference_omega)
    if nl.cascade is None or any(physics.solution is None or physics.system is None for physics in nl.channel_physics):
        return EpsSensitivity(0.0, None, {}, xp.zeros(shape, dtype=xp.float32))
    objectives = [objective_cotangents(physics) for physics in nl.channel_physics]
    total = sum(entry[0] for entry in objectives)
    gradients = {name: xp.zeros(shape, dtype=xp.float64) for name in nl.permittivity_arrays}
    if not any(entry[3] for entry in objectives):
        return EpsSensitivity(total, reference_omega, gradients, xp.zeros(shape, dtype=xp.float32))
    _check_differentiable(nl)
    media = {medium.name: medium for medium in nl.media}
    if nl.coupling_mode == 'pump_depleted':
        received = depleted_cotangents(nl, objectives, media)
        for index in range(len(nl.channels)):
            _, cell_part, objective_faces, _ = objectives[index]
            _channel_backward(nl, index, cell_part, objective_faces, received.get(index), media, gradients, {})
        return EpsSensitivity(total, reference_omega, gradients, sensitivity_density(nl.domain, gradients))
    stored = {}   # channel index -> [store-layout cotangents] sent by higher channels' sources
    for index in reversed(solve_order(nl.channels)):
        _, cell_part, objective_faces, _ = objectives[index]
        _channel_backward(nl, index, cell_part, objective_faces, stored.pop(index, None), media, gradients, stored)
    return EpsSensitivity(total, reference_omega, gradients, sensitivity_density(nl.domain, gradients))


def _transposed_sweep(nl, objectives, received, media):
    """T(c) of the block-Jacobi map: {index: store-layout parts} the sources send back for output cotangents =
    objectives (None = none) + received {index: parts}."""
    outgoing = {}
    for index in range(len(nl.channels)):
        cell_part, objective_faces = (None, (None, None)) if objectives is None else objectives[index][1:3]
        _channel_backward(nl, index, cell_part, objective_faces, received.get(index), media, None, outgoing)
    return outgoing


def _pack(store, parts_by_index):
    """Store-layout complex128 vector of {index: parts} (None parts = 0)."""
    vector = store.xp.zeros(store.size, dtype=store.xp.complex128)
    for index, parts in parts_by_index.items():
        for view, part in zip(store.arrays(index, vector), parts):
            if part is not None:
                view += part
    return vector


def depleted_cotangents(nl, objectives, media):
    """{index: store-layout parts} nu of the pump-depleted fixed point x = G_J(x): (I - T) nu = T(c_obj), T the
    transposed block-Jacobi sweep (real GMRES, blocks weighted by ||x_k||; nl.adjoint_record) -> received cotangents of
    the final gradient sweep. RuntimeError when GMRES stays above ADJOINT_FAIL_RESIDUAL."""
    store = nl.store
    blocks = store.blocks
    rhs = _pack(store, _transposed_sweep(nl, objectives, {}, media))
    norms = [float(store.xp.linalg.norm(store.vector()[block])) for block in blocks]
    largest = max(norms, default=0.0)
    weights = [size / largest if size > 0.0 and largest > 0.0 else 1.0 for size in norms]
    inverse = [1.0 / weight for weight in weights]

    def unpack(vector):
        return {index: list(store.arrays(index, vector)) for index in range(len(nl.channels))}

    def scaled_product(scaled):
        vector = _scale_blocks(scaled.copy(), blocks, inverse)
        product = vector - _pack(store, _transposed_sweep(nl, None, unpack(vector), media))
        return _scale_blocks(product, blocks, weights)

    tolerance = nl.iteration_settings.tolerance
    scaled, products, relative = real_gmres(scaled_product, _scale_blocks(rhs, blocks, weights), tolerance,
                                            max_iterations=ADJOINT_MAX_PRODUCTS)
    nl.adjoint_record = {'gmres_products': products, 'relative_residual': relative, 'tolerance': tolerance,
                         'converged': relative <= tolerance}
    if not relative <= ADJOINT_FAIL_RESIDUAL:
        raise RuntimeError(f'em_fdfd_nonlinear pump-depleted adjoint: GMRES reached only {relative:.2e} after '
                           f'{products} products (the forward fixed point is likely not converged or near a '
                           'bifurcation)')
    return unpack(_scale_blocks(scaled, blocks, inverse))

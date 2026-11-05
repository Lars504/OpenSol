"""Conical / out-of-plane k_z (symmetry 2.9): the full-vector 2D Yee operator of a z-invariant structure at fixed k_z,
driven solves, eigenfrequencies, centre phasors and line fluxes; host numpy / scipy only (never cupy). Model type:
em_fdfd_conical (solvers/fdfd_conical)."""
# Fields ~ exp(j omega t - j k_z z), d/dz -> -j k_z; the projected 3D Yee cell and every wall / mirror / Bloch / PML rule
# of cross_section_modes (its cross_section_layout and assemble_cross_section: E_x hi-x face, E_y hi-y face, E_z centre,
# H_x hi-y face, H_y hi-x face, H_z corner; eps of the E points = its face means, mu of the H points = its lower-cell
# values). Unknowns E = [E_x; E_y; E_z] on the active points (inside the walls, not PEC):
#   C_E = [[0, j k_z, Dy^h], [-j k_z, 0, -Dx^h], [-Dy^h, Dx^h, 0]]   (curl E on the H points, wall column maps in D^h)
#   C_H = [[0, j k_z, Dy^e], [-j k_z, 0, -Dx^e], [-Dy^e, Dx^e, 0]]   (curl H on the E points)
#   driven: (k0^2 eps - C_H mu^-1 C_E) E = -j omega mu0 J_F (the FDTD current sign of fdfd_sources / harmonic.py, so at
#   k_z = 0 the E_z block is the em_fdfd_ez operator up to its wall planes); H = C_E E / (-j omega mu0 mu); then C_H H =
#   j omega eps0 eps E - J_F, i.e. J_F is minus the textbook impressed current and the source power is 1/2 Re E . J_F*.
#   eigen: C_H mu^-1 C_E E = k0^2 eps E at fixed k_z; curl-free E = (Dx^h, Dy^h, -j k_z) phi solves it at k0 = 0 (static
#   solutions, one per cell): eigenfrequency.solve_resonances with the squared Cayley map sends them to 0 and drops
#   them (as BOR). The cross-section mode solve (A h = beta^2 B h at fixed omega) is the same discrete problem
#   read the other way: (omega, beta) pairs coincide (example em/analytic_conical_walled_guide: the walled-guide
#   cross-section spectrum read as k0 at k_z).
# Symmetry: the k_z blocks of C_H are MINUS the transposed ones of C_E (Hermitian, not transposed, pairs), the D^e / D^h
# blocks the FIT transposes (D^e = -W_e^-1 (D^h)^T W_h with the wall shares and stretches), so D A is complex symmetric
# for D = (A_x, A_y, -A_z) on (E_x, E_y, E_z): the layout dual lengths times the CPML stretch along each axis at the point
# (face lengths x face stretch, centre lengths x centre stretch), the E_z block negated (= the similarity E_z -> j E_z
# that makes the curls real). PML, loss, complex k_z, mirrors / walls, PEC and subpixel eps keep it; a Bloch phase breaks
# it (phased wrap entries: LU, no polish). With D a cudss eigen node factors an LDL^T and cudss_mixed adds the float64
# Rayleigh polish (solve_resonances(symmetrizer=D)).
# Flux through the hi face of a line of cells: 1/2 Re of E of the line cells x conj(H) on that face (the boundary term
# of the discrete Poynting theorem for the cells below / left of it); in lossless source-free cells a line across a whole
# period (or wall to wall) carries the same power whichever side's E is paired, so two such lines enclose exactly the
# source power minus the absorption between them (k_z real: its terms cancel point by point).
# k_z may be complex (leaky / evanescent drive); the PML stretch is evaluated at the system's omega, so an eigen-solve
# with cpml faces is exact only at omega_guess (as the FDFD eigenfrequency). Conformal PEC cuts and bends are refused.
import cmath
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, MU0
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver
from ...solvers.sparse_direct import factorize_host, size_issue
from .boundaries import cpml_axis_profiles
from .cross_section_modes import assemble_cross_section, cross_section_layout, fill_images, to_centres
from .eigenfrequency import EigenSolve, solve_resonances
from .operators import bloch_wrap_phases, pml_stretch
from .placement import dipole_faces
from .yee_grid import yee_inverse_spacings

COMPONENTS = ('x', 'y', 'z')
FIELD_COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
# ARPACK tolerance floor of the eigen-solve: a target exactly on a resonance (the TEM mode of a filled walled guide)
# amplifies the solve's rounding along it by up to 1 / eigenfrequency.SHIFT_RELATIVE_OFFSET; a machine-precision
# tolerance never met that noise (minutes of restarts). k0^2 error ~ 1e-10 |k0^2 - shift|
_ARPACK_TOLERANCE_FLOOR = 1e-10
_SIZE_REMEDY = 'coarsen the mesh or shrink the cross-section (a symmetric / antisymmetric mirror face halves it)'
# (x centred, y centred) of each component, for the wall ghost / beyond-plane images
_CENTRING = {'ex': (False, True), 'ey': (True, False), 'ez': (True, True), 'hx': (True, False), 'hy': (False, True),
             'hz': (False, False)}


@dataclass(frozen=True, eq=False)
class ConicalSystem:
    """Active-unknown operators of one (omega, k_z): matrix = k0^2 eps - C_H mu^-1 C_E, curl_curl = C_H mu^-1 C_E (both
    scipy CSR over active_index of the flat [E_x; E_y; E_z], 3 nx ny), eps of the active points, full-grid curl_e and
    mu of the [H_x; H_y; H_z] points (H recovery), the cross-section layout, the Bloch wrap factors e^{+j k L} and the
    symmetrizer D of the active points (module header; None with a Bloch phase)."""
    matrix: object
    curl_curl: object
    eps_active: np.ndarray
    curl_e: object
    mu_points: np.ndarray
    active_index: np.ndarray
    omega: float
    k0: float
    k_z: complex
    shape: tuple
    layout: object
    wrap_factors: tuple = (1.0, 1.0)
    symmetrizer: np.ndarray = None

    @property
    def n_unknowns(self):
        return int(self.active_index.size)


@dataclass(frozen=True, eq=False)
class ConicalResonance:
    """One resonance at fixed k_z: complex omega [rad/s] (Im > 0 decays), vacuum wavelength 2 pi C0 / Re omega [m], Q,
    PML share of |E|^2 dA and the full-grid field dict (conical_fields; 1 real at its largest E sample)."""
    omega: complex
    wavelength: float
    q_factor: float
    pml_fraction: float
    fields: dict

    def text(self):
        """'lambda = 1536.54 nm, Q = 1023' (Q = inf when lossless)."""
        return f'lambda = {self.wavelength * 1e9:.6g} nm, Q = {self.q_factor:.4g}'


@dataclass(frozen=True, eq=False)
class ConicalEigenSolve(EigenSolve):
    """eigenfrequency.EigenSolve of ConicalResonance (nearest omega_guess first) plus the ConicalSystem they solve."""
    system: object = None

    @property
    def omegas(self):
        """complex128 omegas [rad/s] of the resonances."""
        return np.array([resonance.omega for resonance in self.resonances], dtype=np.complex128)


def size_issue_for(nx, ny):
    """(level, text) of the host SuperLU size rule for 3 nx ny unknowns (sparse_direct.size_issue), None when fine; CPU."""
    return size_issue('scipy_splu', nx, ny, False, False, unknowns_per_cell=3, remedy=_SIZE_REMEDY)


def assemble_conical(cell_widths, boundary_config, pml_dx, materials, omega, k_z, check_size=True):
    """ConicalSystem of cross_section_modes.CrossSectionMaterials at omega [rad/s] (PML stretch, k0) and k_z [rad/m];
    cell_widths = (dx_cells, dy_cells) [m], pml_dx = CPML grading cell [m]; check_size: the host SuperLU size rule."""
    import scipy.sparse
    omega, k_z = float(omega), complex(k_z)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"assemble_conical needs a finite angular frequency omega > 0 [rad/s], got {omega!r}")
    if not (math.isfinite(k_z.real) and math.isfinite(k_z.imag)):
        raise ValueError(f"assemble_conical needs a finite k_z [rad/m], got {k_z!r}")
    if materials.pec_cuts is not None:
        raise NotImplementedError("assemble_conical: conformal PEC cuts are not supported (staircase PEC only)")
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    shape = (dx_cells.size, dy_cells.size)
    if np.shape(materials.eps_zz) != shape:
        raise ValueError(f"materials shape {np.shape(materials.eps_zz)} does not match the {shape[0]} x {shape[1]} "
                         f"cell widths")
    issue = size_issue_for(*shape) if check_size else None
    if issue is not None and issue[0] == 'error':
        raise ValueError(issue[1])
    layout = cross_section_layout((dx_cells, dy_cells), boundary_config, pml_dx)
    parts = assemble_cross_section((dx_cells, dy_cells), boundary_config, pml_dx, materials, omega, None, layout)
    n_grid = shape[0] * shape[1]
    j_kz = scipy.sparse.identity(n_grid, dtype=np.complex128, format='csr') * (1j * k_z)
    curl_e = scipy.sparse.bmat([[None, j_kz, parts.dy_h], [-j_kz, None, -parts.dx_h], [-parts.dy_h, parts.dx_h, None]],
                               format='csr')
    curl_h = scipy.sparse.bmat([[None, j_kz, parts.dy_e], [-j_kz, None, -parts.dx_e], [-parts.dy_e, parts.dx_e, None]],
                               format='csr')
    mu_points = np.concatenate((parts.mu_x, parts.mu_y, parts.mu_z)).astype(np.complex128)
    inverse_eps = np.concatenate((parts.inv_exx, parts.inv_eyy, parts.inv_ezz))
    x_axis, y_axis = layout.x, layout.y
    inside = np.concatenate((np.outer(x_axis.face_active, y_axis.centre_active).ravel(),
                             np.outer(x_axis.centre_active, y_axis.face_active).ravel(),
                             np.outer(x_axis.centre_active, y_axis.centre_active).ravel()))
    active_index = np.flatnonzero(inside & (inverse_eps != 0))
    if active_index.size < 4:
        raise ValueError(f"assemble_conical: only {active_index.size} E unknowns remain inside the walls")
    eps_active = 1.0 / inverse_eps[active_index]
    curl_curl = (curl_h[active_index] @ scipy.sparse.diags(1.0 / mu_points, 0, format='csr')
                 @ curl_e[:, active_index]).tocsr()
    k0 = omega / C0
    matrix = (scipy.sparse.diags(k0 ** 2 * eps_active, 0, format='csr') - curl_curl).tocsr()
    symmetrizer = None if boundary_config.has_bloch_phase else _symmetrizer(layout, boundary_config, pml_dx, omega,
                                                                            active_index)
    return ConicalSystem(matrix=matrix, curl_curl=curl_curl, eps_active=eps_active, curl_e=curl_e,
                         mu_points=mu_points, active_index=active_index, omega=omega, k0=k0, k_z=k_z, shape=shape,
                         layout=layout, wrap_factors=parts.wrap_factors, symmetrizer=symmetrizer)


def _symmetrizer(layout, boundary_config, pml_dx, omega, active_index):
    """Complex128 D of the active [E_x; E_y; E_z] points with D A complex symmetric (module header): per axis the layout
    face / centre length times the CPML stretch of that face / centre row at omega (yee_operators' stretches)."""
    lengths = []
    for axis, axis_layout in (('x', layout.x), ('y', layout.y)):
        profile = cpml_axis_profiles(boundary_config, axis_layout.centre_length.size, axis, pml_dx)
        lengths.append((axis_layout.centre_length * pml_stretch(profile.sigma_e, profile.kappa_e, profile.alpha_e,
                                                                 omega),
                        axis_layout.face_length * pml_stretch(profile.sigma_h, profile.kappa_h, profile.alpha_h,
                                                               omega)))
    (x_centre, x_face), (y_centre, y_face) = lengths
    full = np.concatenate((np.outer(x_face, y_centre).ravel(), np.outer(x_centre, y_face).ravel(),
                           -np.outer(x_centre, y_centre).ravel()))
    return full[active_index].astype(np.complex128)


def conical_fields(system, e_active, omega=None):
    """Dict ex, ey, ez [V/m], hx, hy, hz [A/m] (complex128 (nx, ny) on their Yee points, wall ghost / beyond-plane points
    holding the mirror images) of the active E vector; H by Faraday at omega (default the system's)."""
    omega = system.omega if omega is None else complex(omega)
    n_grid = system.shape[0] * system.shape[1]
    e_full = np.zeros(3 * n_grid, dtype=np.complex128)
    e_full[system.active_index] = e_active
    h_full = (system.curl_e @ e_full) / (-1j * omega * MU0 * system.mu_points)
    fields = {}
    for number, component in enumerate(COMPONENTS):
        part = slice(number * n_grid, (number + 1) * n_grid)
        fields[f'e{component}'] = np.array(e_full[part].reshape(system.shape), dtype=np.complex128)
        fields[f'h{component}'] = np.array(h_full[part].reshape(system.shape), dtype=np.complex128)
    for name, field in fields.items():
        fill_images(field, system.layout, *_CENTRING[name])
    return fields


def centre_phasors(system, fields):
    """(phasors, e_squared): complex128 (nx, ny) cell-centre values of the six components (plain means of their face /
    corner points, Bloch wrap factors on the lo partners) and float64 cell means of |E_x|^2, |E_y|^2, |E_z|^2 keyed
    'x', 'y', 'z' (the per-component absorbed power)."""
    periodic_x, periodic_y = system.layout.x.periodic, system.layout.y.periodic
    wrap_x, wrap_y = system.wrap_factors
    phasors = {'ex': to_centres(fields['ex'], 0, periodic_x, wrap_x),
               'ey': to_centres(fields['ey'], 1, periodic_y, wrap_y),
               'ez': np.array(fields['ez'], dtype=np.complex128, copy=True),
               'hx': to_centres(fields['hx'], 1, periodic_y, wrap_y),
               'hy': to_centres(fields['hy'], 0, periodic_x, wrap_x),
               'hz': to_centres(to_centres(fields['hz'], 0, periodic_x, wrap_x), 1, periodic_y, wrap_y)}
    e_squared = {'x': to_centres(np.abs(fields['ex']) ** 2, 0, periodic_x),
                 'y': to_centres(np.abs(fields['ey']) ** 2, 1, periodic_y), 'z': np.abs(fields['ez']) ** 2}
    return phasors, e_squared


def point_current_terms(cell_widths, boundary_config, ix, iy, component, current):
    """[(component, i, j, J_F [A/m^2])] of a current element of `current` [A] (= integral of J_F over cell (ix, iy)):
    'z' on the centre, 'x' / 'y' half on each of its two faces along that axis (_axis_faces), by their dual areas."""
    if component not in COMPONENTS:
        raise ValueError(f"current component must be one of {COMPONENTS}, got {component!r}")
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    ix, iy, current = int(ix), int(iy), complex(current)
    if not (0 <= ix < dx_cells.size and 0 <= iy < dy_cells.size):
        raise ValueError(f"current cell ({ix}, {iy}) is outside the {dx_cells.size} x {dy_cells.size} grid")
    if component == 'z':
        return [('z', ix, iy, current / (dx_cells[ix] * dy_cells[iy]))]
    spacings = yee_inverse_spacings(dx_cells, dy_cells)
    phase_x, phase_y = bloch_wrap_phases((dx_cells, dy_cells), boundary_config)
    if component == 'x':
        return [('x', face, iy, weight * current * spacings.inv_dx_h[face] / dy_cells[iy])
                for face, weight in _axis_faces(ix, dx_cells.size, boundary_config.periodic_x, phase_x)]
    return [('y', ix, face, weight * current * spacings.inv_dy_h[face] / dx_cells[ix])
            for face, weight in _axis_faces(iy, dy_cells.size, boundary_config.periodic_y, phase_y)]


def _axis_faces(cell, n_cells, periodic, wrap_phase):
    """[(face, weight)] of the hi and lo face of `cell`, half each (placement.dipole_faces off-periodic: a missing outer
    face passes its half on); the wrapped lo face of cell 0 on a periodic axis carries e^{-j wrap_phase}."""
    if not periodic:
        return dipole_faces(cell, n_cells, False)
    return [(cell, 0.5), ((cell - 1) % n_cells, 0.5 * (cmath.exp(-1j * wrap_phase) if cell == 0 else 1.0))]


def driven_rhs(system, currents=None, terms=()):
    """Active rhs -j omega mu0 J_F (complex128 (n_unknowns,)) of currents {'x' | 'y' | 'z': complex (nx, ny) J_F
    [A/m^2] on the E_x / E_y / E_z points} and / or point terms [(component, i, j, J_F)]; inactive points are dropped."""
    n_grid = system.shape[0] * system.shape[1]
    current_full = np.zeros(3 * n_grid, dtype=np.complex128)
    for component, density in (currents or {}).items():
        if component not in COMPONENTS:
            raise ValueError(f"currents keys must be among {COMPONENTS}, got {component!r}")
        if np.shape(density) != system.shape:
            raise ValueError(f"current {component!r} shape {np.shape(density)} differs from the grid {system.shape}")
        number = COMPONENTS.index(component)
        current_full[number * n_grid:(number + 1) * n_grid] += np.asarray(density, dtype=np.complex128).ravel()
    for component, i, j, density in terms:
        current_full[COMPONENTS.index(component) * n_grid + int(i) * system.shape[1] + int(j)] += complex(density)
    return -1j * system.omega * MU0 * current_full[system.active_index]


def solve_conical_driven(system, currents, linear_solver=None):
    """(fields, info) of (k0^2 eps - C_H mu^-1 C_E) E = -j omega mu0 J_F by host SuperLU (linear_solver.factorize_host
    when given, else sparse_direct.factorize_host); currents as driven_rhs (FDTD sign)."""
    started = time.perf_counter()
    rhs = driven_rhs(system, currents)
    factor = (factorize_host if linear_solver is None else linear_solver.factorize_host)(system.matrix)
    e_active = factor.solve_host(rhs)
    info = {'n_unknowns': system.n_unknowns, 'factor_host_bytes': factor.host_bytes,
            'seconds': time.perf_counter() - started}
    return conical_fields(system, e_active), info


def line_flux(system, fields, normal, line_cell, span_cells):
    """Power per unit z length [W/m] through the hi face of row / column line_cell over span_cells, > 0 toward +normal
    ('y' horizontal line, 'x' vertical): E of the line cells x conj(H) on that face, layout-length weights (header)."""
    layout = system.layout
    if normal == 'y':
        along, centre_name, face_name, centre_h, face_h, sign = layout.x, 'ez', 'ex', 'hx', 'hz', 1.0
    elif normal == 'x':
        along, centre_name, face_name, centre_h, face_h, sign = layout.y, 'ez', 'ey', 'hy', 'hz', -1.0
    else:
        raise ValueError(f"line_flux normal must be 'x' or 'y', got {normal!r}")
    start, stop = int(span_cells[0]), int(span_cells[1])
    n_along = along.centre_length.size
    if not 0 <= start < stop <= n_along:
        raise ValueError(f'line_flux span cells [{start}, {stop}) leave the {n_along} cells along the line')

    def on_line(name, indices):
        values = fields[name]
        return values[indices, int(line_cell)] if normal == 'y' else values[int(line_cell), indices]

    cells = np.arange(start, stop)
    face_weights = {face: along.face_length[face] for face in range(start, stop - 1)}
    for face in (start - 1, stop - 1):
        if face < 0 and not along.periodic:
            continue
        face = face % n_along
        # a wall / mirror plane face already holds its inner half share (layout face_length): never halved again
        plane = ((along.lo_sign is not None and face == 0)
                 or (along.hi_sign is not None and face == n_along - 2))
        face_weights[face] = face_weights.get(face, 0.0) + (1.0 if plane else 0.5) * along.face_length[face]
    faces = np.array(sorted(face_weights), dtype=np.int64)
    weights = np.array([face_weights[face] for face in faces], dtype=np.float64)
    centre_term = np.sum(on_line(centre_name, cells) * np.conj(on_line(centre_h, cells)) * along.centre_length[cells])
    face_term = np.sum(on_line(face_name, faces) * np.conj(on_line(face_h, faces)) * weights)
    # S_y = E_z H_x* - E_x H_z* (horizontal line), S_x = E_y H_z* - E_z H_y* (vertical line)
    return float(0.5 * np.real(sign * (centre_term - face_term)))


def source_power(system, fields, terms):
    """Power per unit z length [W/m] the point terms [(component, i, j, J_F)] deliver: 1/2 Re sum E . conj(J_F) dA over
    their E points (J_F is minus the textbook current: module header), dual areas of the layout (wall shares)."""
    layout = system.layout
    areas = {'x': layout.x_face_area, 'y': layout.y_face_area, 'z': layout.centre_area}
    total = 0.0
    for component, i, j, density in terms:
        total += 0.5 * float(np.real(fields[f'e{component}'][i, j] * np.conj(complex(density))
                                     * areas[component][i, j]))
    return total


def conical_row_scale(system):
    """cuDSS / eigen row scale of the factored operator (D A, or A with a Bloch phase): (|D| (|diag curl-curl| + k0^2
    |eps|))^-1/2 (diag A = k0^2 eps - diag curl-curl nearly cancels near a resonance, so the parts' moduli; 1 on an
    empty row)."""
    scale = np.abs(system.curl_curl.diagonal()) + system.k0 ** 2 * np.abs(system.eps_active)
    if system.symmetrizer is not None:
        scale = scale * np.abs(system.symmetrizer)
    return 1.0 / np.sqrt(np.where(scale > 0.0, scale, 1.0))


def solve_conical_resonances(cell_widths, boundary_config, pml_dx, materials, k_z, n_wanted, omega_guess,
                             eigen_solver=None):
    """ConicalEigenSolve of up to n_wanted resonances C_H mu^-1 C_E E = k0^2 eps E at k_z [rad/m] nearest omega_guess
    (eigenfrequency.solve_resonances, squared Cayley: statics and PML resonances dropped, PML stretched at omega_guess);
    eigen_solver None = default settings (host SuperLU; cudss: LDL^T of D A without a Bloch phase, else LU)."""
    started = time.perf_counter()
    if isinstance(n_wanted, bool) or not isinstance(n_wanted, (int, np.integer)) or n_wanted < 1:
        raise ValueError(f"n_wanted must be an integer >= 1, got {n_wanted!r}")
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    warnings = []
    if eigen_solver.factorization not in CUDSS_FACTORIZATIONS:   # host SuperLU size rule (cuDSS guards its own memory)
        issue = size_issue_for(*np.shape(materials.eps_zz))
        if issue is not None:
            if issue[0] == 'error':
                raise ValueError(f'eigenfrequency solve: {issue[1]}')
            warnings.append(issue[1])
    system = assemble_conical(cell_widths, boundary_config, pml_dx, materials, omega_guess, k_z, check_size=False)
    assembled = time.perf_counter()
    layout = system.layout
    stretched_z = layout.x.stretched_centres[:, None] | layout.y.stretched_centres[None, :]
    energy_weights = np.concatenate((layout.x_face_area.ravel(), layout.y_face_area.ravel(),
                                     layout.centre_area.ravel()))[system.active_index]
    in_window = ~np.concatenate((layout.stretched_hy.ravel(), layout.stretched_hx.ravel(),
                                 stretched_z.ravel()))[system.active_index]
    solve = solve_resonances(system.matrix, system.eps_active, system.omega, energy_weights, in_window, int(n_wanted),
                             spectral_map='squared_cayley', warnings=warnings, eigen_solver=eigen_solver,
                             tolerance_floor=_ARPACK_TOLERANCE_FLOOR, symmetrizer=system.symmetrizer,
                             equilibration=conical_row_scale(system))
    resonances = [ConicalResonance(omega=resonance.omega, wavelength=resonance.wavelength,
                                   q_factor=resonance.q_factor, pml_fraction=resonance.pml_fraction,
                                   fields=conical_fields(system, resonance.field, resonance.omega))
                  for resonance in solve.resonances]
    info = dict(solve.info, k_z=system.k_z, assembly_seconds=assembled - started,
                seconds=time.perf_counter() - started)
    return ConicalEigenSolve(resonances=resonances, info=info, system=system)

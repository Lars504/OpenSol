"""Materials and losses at one angular frequency for the harmonic EM solvers (FDFD now, harmonic FDTD later): complex
eps / mu per cell, the H_z face-current right-hand side, absorbed power density and the assembled HarmonicSystem."""
# Right-hand-side signs (exp(+j omega t)). Every FDTD source ADDS +c J s(t) to its field, so an FDTD current J_F / M_F
# is minus the textbook impressed current (J_std = -J_F, M_std = -M_F) and the harmonic solvers keep the FDTD meaning:
#   Ampere   j omega eps0 eps E = curl H - J_std = curl H + J_F
#   Faraday -j omega mu0 mu H   = curl E + M_std = curl E - M_F
# Eliminating the in-plane field (W = 1/mu on the faces; V_x = 1/eps_yface on hi-x, V_y = 1/eps_xface on hi-y faces):
#   ez: [dx_e W dx_h + dy_e W dy_h + k0^2 eps] E_z = -j omega mu0 J_F
#   hz: [dx_e V_x dx_h + dy_e V_y dy_h + k0^2 mu] H_z = dx_e (J_yF / eps_yface) - dy_e (J_xF / eps_xface)
#       - j omega eps0 M_F
# so a phasor at phase 0 equals the harmonic FDTD response DFT(field) / S(omega) of the same source.
# hz with an off-diagonal eps_xy: D = (dy_h H_z + J_xF, -dx_h H_z + J_yF) on the hi-y / hi-x faces and j omega eps0 E =
# eta D with eta = eps^-1 per cell: diagonal V = 1 / face mean of 1/eta_ii (= eps_xx - eps_xy^2 / eps_yy at E_x), the
# coupling X = face mean . eta_xy . centre mean (operators.off_diagonal_face_couplings); the operator gains
# -(dx_e X_yx dy_h + dy_e X_xy dx_h) and the face-current rhs dx_e (X_yx J_xF) - dy_e (X_xy J_yF).
from dataclasses import dataclass

import numpy as np
from ...core.backend import array_module_of, cp

from ...core.constants import DTYPE, EPS0, MU0
from .dispersion import complex_permittivity
from .yee_grid import lower_face_partner


def cell_permittivity(materials, array_name, dispersive_regions, omega, xp=cp):
    """complex128 (nx, ny) eps(omega) of MaterialGrid array `array_name`: background - j sigma_e/(omega eps0) + chi_k on
    each region of EMPhysics.dispersive_regions [(pole, box slices | None, cropped bool mask | None)]."""
    permittivity = complex_permittivity(getattr(materials, array_name), materials.sigma_e, (), omega, xp=xp)
    for pole, box, cropped_mask in dispersive_regions:
        susceptibility = complex(pole.susceptibility(omega))
        region = permittivity if box is None else permittivity[box]   # a view: += writes into permittivity
        if cropped_mask is None:
            region += susceptibility
        else:
            region += susceptibility * cropped_mask   # box-sized temporary only
    return permittivity


def cell_permeability(materials, omega, xp=cp):
    """complex128 (nx, ny) mu_r - j sigma_m/(omega mu0)."""
    omega = float(omega)
    if not (np.isfinite(omega) and omega > 0.0):
        raise ValueError(f"cell_permeability needs a finite angular frequency omega > 0 [rad/s], got {omega!r}")
    return (xp.asarray(materials.mu_r, dtype=xp.complex128)
            - 1j * xp.asarray(materials.sigma_m, dtype=xp.float64) / (omega * MU0))


def write_float32(target, host_values) -> None:
    """Copy host values into the float32 run-backend array `target` in place (cupy: one host-to-device copy, no
    device temporary)."""
    values = np.ascontiguousarray(host_values, dtype=np.float32)
    if isinstance(target, np.ndarray):
        target[...] = values
    else:
        target.set(values)


def shown_phasor_fields(phasors, e_names, h_names, abs_parts=True, sz=False) -> dict:
    """Host float64 display fields of complex phasors: Re of every E / H component (abs_parts: also |.| as <name>_abs),
    e_intensity, e_abs, h_abs and (sz) S_z = 1/2 Re(E_x H_y* - E_y H_x*)."""
    shown = {}
    for name in (*e_names, *h_names):
        shown[name] = np.real(phasors[name])
        if abs_parts:
            shown[f'{name}_abs'] = np.abs(phasors[name])
    shown['e_intensity'] = sum(np.abs(phasors[name]) ** 2 for name in e_names)
    shown['e_abs'] = np.sqrt(shown['e_intensity'])
    shown['h_abs'] = np.sqrt(sum(np.abs(phasors[name]) ** 2 for name in h_names))
    if sz:
        shown['sz'] = 0.5 * np.real(phasors['ex'] * np.conj(phasors['hy']) - phasors['ey'] * np.conj(phasors['hx']))
    return shown


def face_current_rhs(system, jx_faces, jy_faces):
    """hz rhs (nx, ny) complex128 of FDTD-sign face currents [A/m^2]: dx_e (J_y W_x) - dy_e (J_x W_y), W = the
    system's 1/eps_face weights (+ the eps_xy couplings), so currents on PEC and even-wall faces drop; jx on hi-y, jy on
    hi-x faces (None = 0)."""
    if system.polarization != 'hz':
        raise ValueError(f"face_current_rhs is the H_z face-current term; an {system.polarization!r} system takes "
                         f"-j omega mu0 J_z at the cell centres")
    ops = system.ops
    xp = array_module_of(system.eps_hi_y_faces)
    rhs = xp.zeros(ops.nx * ops.ny, dtype=xp.complex128)
    jx_flat = None if jx_faces is None else xp.asarray(jx_faces, dtype=xp.complex128).ravel()
    jy_flat = None if jy_faces is None else xp.asarray(jy_faces, dtype=xp.complex128).ravel()
    weight_hi_x_faces, weight_hi_y_faces = system.weight_hi_x_faces, system.weight_hi_y_faces
    if system.face_fractions is not None:   # the operator's conformal face weights
        weight_hi_x_faces = weight_hi_x_faces * system.face_fractions[0]
        weight_hi_y_faces = weight_hi_y_faces * system.face_fractions[1]
    if jy_flat is not None:
        rhs += ops.dx_e @ (jy_flat * weight_hi_x_faces.ravel())
    if jx_flat is not None:
        rhs -= ops.dy_e @ (jx_flat * weight_hi_y_faces.ravel())
    if system.off_diagonal is not None:
        y_faces_from_x_faces, x_faces_from_y_faces = system.off_diagonal
        if jx_flat is not None:
            rhs += ops.dx_e @ (x_faces_from_y_faces @ jx_flat)
        if jy_flat is not None:
            rhs -= ops.dy_e @ (y_faces_from_x_faces @ jy_flat)
    return rhs.reshape(ops.nx, ops.ny)


def _squared_modulus(values, xp):
    """Float64 |values|^2 (a complex64 modulus is widened before squaring)."""
    modulus = xp.abs(values).astype(xp.float64, copy=False)
    return modulus * modulus


def face_mean_at_centres(face_values, axis, periodic, xp=cp, wrap_factor=1.0):
    """Per-cell mean of a hi-face quantity (any dtype) over the cell's hi face k and lo face k-1 along `axis`; cell 0
    takes the wrapped last face (periodic; times wrap_factor = FDFDPhysics.lower_wrap_factors[axis] for a Bloch phasor)
    or its own hi face (yee_grid.lower_face_partner)."""
    return 0.5 * (face_values + lower_face_partner(face_values, axis, periodic, xp, wrap_factor))


def absorbed_power_ez(omega, cell_eps, ez):
    """float32 (nx, ny) [W/m^3] 1/2 omega eps0 (-Im eps) |E_z|^2."""
    xp = array_module_of(ez)
    power_density = _squared_modulus(ez, xp)
    power_density *= (-0.5 * float(omega) * EPS0) * xp.asarray(cell_eps).imag
    return power_density.astype(DTYPE)


def absorbed_power_hz(omega, cell_eps_xx, cell_eps_yy, ex_faces, ey_faces, periodic_x, periodic_y,
                      face_fractions=None):
    """float32 (nx, ny) [W/m^3]: per cell 1/2 omega eps0 [(-Im eps_xx) mean|E_x|^2 + (-Im eps_yy) mean|E_y|^2], each
    mean over the cell's two faces (cell 0: wrapped last face if periodic, else its own hi face); |E|^2 x the conformal
    (hi-x, hi-y) face_fractions when given (the operator's face loss)."""
    xp = array_module_of(ex_faces)
    ex_squared, ey_squared = _squared_modulus(ex_faces, xp), _squared_modulus(ey_faces, xp)
    if face_fractions is not None:
        ey_squared *= face_fractions[0]
        ex_squared *= face_fractions[1]
    power_density = face_mean_at_centres(ex_squared, 1, periodic_y, xp)   # E_x on hi-y faces
    power_density *= xp.asarray(cell_eps_xx).imag
    ey_power = face_mean_at_centres(ey_squared, 0, periodic_x, xp)        # E_y on hi-x faces
    ey_power *= xp.asarray(cell_eps_yy).imag
    power_density += ey_power
    power_density *= -0.5 * float(omega) * EPS0
    return power_density.astype(DTYPE)


@dataclass(frozen=True, eq=False)
class HarmonicSystem:
    """One assembled FDFD system at omega: Helmholtz operator plus the material values sources / monitors need."""
    polarization: str            # 'ez' | 'hz'
    omega: float                 # rad/s
    ops: object                  # YeeOperators2D at omega (CSR complex128 of the backend: cupyx / scipy)
    operator: object             # A (CSR complex128, PEC pinning for ez, outermost wall / mirror cells pinned)
    weight_hi_x_faces: object    # complex128 (nx, ny): ez 1/mu_face, hz 1/eps_yy (0 on PEC / even-wall faces; hz
    weight_hi_y_faces: object    # ez 1/mu_face, hz 1/eps_xx   conducting sheets included, unlike eps_hi_*_faces)
    cell_eps: object             # ez: complex eps_r(omega) (nx, ny), no conducting sheet; hz: None
    cell_eps_xx: object          # hz: complex eps_xx(omega) cells; ez: None
    cell_eps_yy: object          # hz: complex eps_yy(omega) cells; ez: None
    cell_mu: object              # complex mu cells (nx, ny)
    eps_hi_x_faces: object       # hz: face_average(cell_eps_yy, axis 0) at E_y (with eps_xy: of eps_yy - eps_xy^2 /
    eps_hi_y_faces: object       # eps_xx); hz: face_average(cell_eps_xx, axis 1) at E_x (eps_xx - eps_xy^2 / eps_yy)
    pec_cells: object            # xp bool (nx, ny) of the fixed unknowns (ez pinned, hz H_z = 0 rows) or None
    periodic_x: bool
    periodic_y: bool
    # hz with a non-zero eps_xy: (hi-x -> hi-y, hi-y -> hi-x) CSR face couplings of eta_xy (header); else None
    off_diagonal: object = None
    cell_eps_xy: object = None   # hz: float64 (nx, ny) eps_xy of the assembly (FDFDPhysics.assembly_eps_xy) or None
    # hz conformal PEC (conformal_pec.ConformalFDFD): (hi-x, hi-y) free face fractions the operator (and the face-current
    # rhs, q_abs, adjoint) multiply the weights by, and the free-area fractions of the k0^2 mu mass; else None
    face_fractions: object = None
    cell_area_fractions: object = None

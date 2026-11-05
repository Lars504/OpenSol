"""AcousticPhysics: 2D velocity-stress staggered-grid FDTD of in-plane (plane strain) or axisymmetric (r, z) elastic and
acoustic waves in solids and fluids, CPML absorbing layers in the outermost grid cells, sources, probes, the
thermoelastic slot and the piezoelectric electric-field slots (the stress of a coupled field; two-way: the strain's bound
charge and the stiffened time step). Arrays live on the Domain's backend (`domain.xp`): every kernel is a core.kernels
twin, so the gpu and cpu backends share one code path (half_storage: gpu only)."""
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from ...core.backend import cp, to_host

from ...core.constants import DTYPE, EPS0, FACES
from ...core.solver_base import PhysicsSolver
from . import kernels as _kernels
from .axisymmetric import density_shares, kernel_rows, radial_stability_factor
from .cpml import AcousticAxisLayers, psi_array
from .sources import VELOCITY_COMPONENTS, AcousticProbe, HeatPulseSource, VelocitySource, VolumeSource

BOUNDARY_TYPES = ('absorbing', 'rigid', 'periodic')
HEATING_SLOT = 'heating_power'
# stepper precision -> field storage dtype (half_storage: fp16 in stored units, float32 arithmetic in the kernels)
FIELD_DTYPES = {'single': np.float32, 'half_storage': np.float16}
HALF_STORAGE_PEAK = 16.0   # stored units of the estimated peak stress: ~4000x headroom below the fp16 max
STATE_NAMES = ('vx', 'vy', 'sxx', 'syy', 'sxy', 'pressure')
# MaterialGrid arrays: rho is shared with heat (same default 0), youngs_modulus / poisson_ratio (the moduli when
# sound_speed is unset) and photoelastic_p11 / _p12 (read by acoustics_em_acousto_optic) with solid_mechanics; c11 /
# c12 / c22 / c66 default NaN = unset (the speeds / moduli apply); gruneisen is read by the thermoelastic sources every
# step (no coefficient rebuild).
ACOUSTIC_SCHEMA = {
    'rho': 0.0, 'sound_speed': 0.0, 'shear_speed': 0.0, 'youngs_modulus': 0.0, 'poisson_ratio': 0.0,
    'c11': math.nan, 'c12': math.nan, 'c22': math.nan, 'c66': math.nan,
    'acoustic_damping': 0.0, 'gruneisen': 0.0, 'photoelastic_p11': 0.0, 'photoelastic_p12': 0.0,
}
# Piezoelectric e [C/m^2] of the in-plane block (piezo_e<k><J>: D_k, k = 1 x / 2 y, vs the Voigt strain J = 1 xx, 2 yy,
# 6 xy): optional MaterialGrid arrays (any one set allocates all six), read by the electric_field_x / _y slots, the
# bound charge and the stiffened time step.
PIEZO_ARRAYS = ('piezo_e11', 'piezo_e12', 'piezo_e16', 'piezo_e21', 'piezo_e22', 'piezo_e26')
# In-plane monoclinic stiffness [Pa] (T_xx += c16 2 S_xy, T_yy += c26 2 S_xy, T_xy += c16 S_xx + c26 S_yy): optional
# MaterialGrid arrays (either one set allocates both, default 0), read by the monoclinic stress kernel and the time step.
MONOCLINIC_ARRAYS = ('c16', 'c26')
# In-plane photoelastic tensor in model axes beside photoelastic_p11 / _p12 (p_IJ: index-ellipsoid row I = 1 xx, 2 yy,
# 3 zz, 6 xy vs the Voigt strain J = 1, 2, 6): optional MaterialGrid arrays (any one set allocates all ten), default
# NaN = unset = the isotropic fallback (physics/mechanics/strain_optic.PHOTOELASTIC_TENSOR_ROWS / PHOTOELASTIC_SHEAR_ROW),
# read by the acousto-optic transform only.
PHOTOELASTIC_TENSOR_ARRAYS = ('photoelastic_p16', 'photoelastic_p21', 'photoelastic_p22', 'photoelastic_p26',
                              'photoelastic_p31', 'photoelastic_p32', 'photoelastic_p36', 'photoelastic_p61',
                              'photoelastic_p62', 'photoelastic_p66')
OPTIONAL_ACOUSTIC_SCHEMA = {**{name: (0.0, PIEZO_ARRAYS) for name in PIEZO_ARRAYS},
                            **{name: (0.0, MONOCLINIC_ARRAYS) for name in MONOCLINIC_ARRAYS},
                            **{name: (math.nan, PHOTOELASTIC_TENSOR_ARRAYS) for name in PHOTOELASTIC_TENSOR_ARRAYS}}
ELECTRIC_FIELD_SLOTS = ('electric_field_x', 'electric_field_y')
# the e_k1, e_k2, e_k6 row of each field component (slot)
_PIEZO_ROWS = {'electric_field_x': ('piezo_e11', 'piezo_e12', 'piezo_e16'),
               'electric_field_y': ('piezo_e21', 'piezo_e22', 'piezo_e26')}
ACOUSTIC_PROPERTY_ARRAYS = {name: (name,) for name in (*ACOUSTIC_SCHEMA, *MONOCLINIC_ARRAYS, *PIEZO_ARRAYS,
                                                        *PHOTOELASTIC_TENSOR_ARRAYS)}
_COEFFICIENT_MATERIALS = frozenset(ACOUSTIC_SCHEMA) - {'gruneisen', 'photoelastic_p11', 'photoelastic_p12'}


@dataclass(frozen=True, eq=False)
class AcousticFields:
    """Full-grid (nx, ny) fields: vx / vy [m/s] on the hi x / y faces (half steps), sxx / syy [Pa] at the cell
    centres and sxy [Pa] on the hi-hi corners (whole steps); tension positive; stt = the hoop stress at the centres
    (axisymmetric). float32, or float16 in stored units under half_storage (AcousticPhysics.physical_array)."""
    vx: cp.ndarray
    vy: cp.ndarray
    sxx: cp.ndarray
    syy: cp.ndarray
    sxy: cp.ndarray
    stt: cp.ndarray = None


class _PhysicalFieldState(Mapping):
    """state() of an AcousticPhysics, made on access only: the live float32 arrays in single precision, float32 copies
    in m/s / Pa of the stored fields under half_storage; pressure (a full-grid kernel) only when read."""

    def __init__(self, physics):
        self._physics = physics

    def __getitem__(self, name):
        if name not in STATE_NAMES:
            raise KeyError(name)
        return self._physics.pressure() if name == 'pressure' else self._physics.physical_array(name)

    def __contains__(self, name):   # no copy for a membership test
        return name in STATE_NAMES

    def __iter__(self):
        return iter(STATE_NAMES)

    def __len__(self):
        return len(STATE_NAMES)


def _axis_spacings(widths, periodic):
    """Host float64 (1/width, 1/hi-face spacing, lower share, upper share) of one axis; the last face pairs with cell 0
    (periodic) or with itself (a pinned outer face)."""
    following = np.roll(widths, -1) if periodic else np.append(widths[1:], widths[-1])
    pair = widths + following
    return 1.0 / widths, 2.0 / pair, widths / pair, following / pair


def _bounding_box(mask):
    """(slice, slice) bounding the True cells of a bool mask, None when empty (host sync: build time only)."""
    rows = np.flatnonzero(to_host(mask.any(axis=1)))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(to_host(mask.any(axis=0)))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1)


def check_faces(faces):
    """Dict face -> boundary type with every face of FACES; ValueError for unknown types or unpaired periodic faces."""
    missing = [face for face in FACES if face not in faces]
    if missing:
        raise ValueError(f'boundary faces {missing} are missing; give one of {BOUNDARY_TYPES} per face')
    for face in FACES:
        if faces[face] not in BOUNDARY_TYPES:
            raise ValueError(f"face {face}: boundary {faces[face]!r} is not one of {', '.join(BOUNDARY_TYPES)}")
    for axis in ('x', 'y'):
        lo, hi = faces[f'{axis}_lo'], faces[f'{axis}_hi']
        if (lo == 'periodic') != (hi == 'periodic'):
            raise ValueError(f'{axis}: periodic faces must be paired (got {lo} / {hi})')
    return dict(faces)


class AcousticPhysics(PhysicsSolver):
    """Velocity-stress FDTD on the full grid; dt = courant x the (anisotropic) stability limit of the current materials
    (set at construction and at finalize, re-derived only there); a smaller runner step is honoured. axisymmetric: x = r
    from the axis on the grid's x_lo edge (a 'rigid' face, not padded), y = z, with the hoop stress stt. precision
    'single' | 'half_storage' (fp16 fields in stored units, scales fixed at the first finalize; no piezo, no c16 / c26,
    no slots)."""

    name = 'acoustics'
    material_schema = ACOUSTIC_SCHEMA

    def __init__(self, domain, faces=None, pml_cells=20, pml_reflection=1e-4, courant=0.9, heating_envelope=None,
                 axisymmetric=False, precision='single'):
        if precision not in FIELD_DTYPES:
            raise ValueError(f"stepper precision must be one of {', '.join(FIELD_DTYPES)}, got {precision!r}")
        if precision != 'single' and domain.backend.name != 'gpu':
            raise ValueError(f"elastic_waves precision {precision!r} runs on the gpu backend only (fp16 storage by PTX "
                             "conversions has no cpu twin): use precision 'single' on the cpu")
        self.precision = precision
        self.xp = xp = domain.xp
        self.kernels = _kernels.field_kernels(precision)
        # stored = physical / scale (powers of two, 1 in single; half_storage: _set_storage_scales at finalize)
        self.stress_scale = self.velocity_scale = 1.0
        self._stress_to_velocity = self._velocity_to_stress = DTYPE(1.0)
        self.domain = domain
        self.materials = domain.materials
        self.materials.add_properties(ACOUSTIC_SCHEMA)
        self.window = domain.physical_window()
        self.faces = check_faces(faces if faces is not None else {face: 'absorbing' for face in FACES})
        self.axisymmetric = bool(axisymmetric)
        if self.axisymmetric and (self.faces['x_lo'] != 'rigid' or domain.npml_sides['x_lo'] > 0):
            raise ValueError("axisymmetric elastic_waves: x_lo is the axis r = 0, so it needs boundary 'rigid' "
                             f"(got {self.faces['x_lo']!r}) and no padding of another physics there "
                             f"(got {domain.npml_sides['x_lo']} cells)")
        if not 0.0 < float(courant) <= 1.0:
            raise ValueError(f'courant must be in (0, 1], got {courant!r}')
        if not 0.0 < float(pml_reflection) < 1.0:
            raise ValueError(f'pml_reflection must be in (0, 1), got {pml_reflection!r}')
        self.courant = float(courant)
        self.pml_reflection = float(pml_reflection)
        self.heating_envelope = heating_envelope   # None: the heating_power slot is the instantaneous heating rate

        nx, ny = domain.grid.nx, domain.grid.ny
        self.nx, self.ny = nx, ny
        self.periodic_x = self.faces['x_lo'] == 'periodic'
        self.periodic_y = self.faces['y_lo'] == 'periodic'
        dx_cells, dy_cells = (np.array(widths, dtype=np.float64) for widths in domain.cell_widths())
        self._inv_dx_e_host, inv_dx_h, share_x_lo, share_x_hi = _axis_spacings(dx_cells, self.periodic_x)
        self._inv_dy_e_host, inv_dy_h, share_y_lo, share_y_hi = _axis_spacings(dy_cells, self.periodic_y)
        self._inv_dx_h_host, self._inv_dy_h_host = inv_dx_h, inv_dy_h
        inv_dx2, inv_dy2 = float(np.max(self._inv_dx_e_host)) ** 2, float(np.max(self._inv_dy_e_host)) ** 2
        self._radial = None   # (KERNEL_ROWS, nx) float32 radius rows on xp (axisymmetric)
        if self.axisymmetric:
            self._centre_radii_host = np.cumsum(dx_cells) - 0.5 * dx_cells
            self._face_radii_host = np.cumsum(dx_cells)
            share_x_lo, share_x_hi = density_shares(dx_cells)
            inv_dx2 *= radial_stability_factor(dx_cells)
        self._inverse_cell_diagonal = math.sqrt(inv_dx2 + inv_dy2)
        self._stiffness_weights = (np.float64(inv_dx2 / (inv_dx2 + inv_dy2)), np.float64(inv_dy2 / (inv_dx2 + inv_dy2)))
        # 'rigid' faces another physics padded: slip wall on the window face (the padding behind it idles); -1 = none
        sides = domain.npml_sides
        self._pins = tuple(np.int32(row if self.faces[face] == 'rigid' and sides[face] > 0 else -1)
                           for face, row in (('x_lo', sides['x_lo'] - 1), ('x_hi', nx - sides['x_hi'] - 1),
                                             ('y_lo', sides['y_lo'] - 1), ('y_hi', ny - sides['y_hi'] - 1)))
        device = lambda values: xp.asarray(values, dtype=DTYPE)   # noqa: E731
        self._inv_dx_e, self._inv_dx_h = device(self._inv_dx_e_host), device(inv_dx_h)
        self._inv_dy_e, self._inv_dy_h = device(self._inv_dy_e_host), device(inv_dy_h)
        self._shares = (device(share_x_lo), device(share_x_hi), device(share_y_lo), device(share_y_hi))
        if self.axisymmetric:
            self._radial = device(kernel_rows(dx_cells))

        layer_cells = {face: int(pml_cells) if self.faces[face] == 'absorbing' else 0 for face in FACES}
        self.layer_cells = layer_cells
        self._x_layers = AcousticAxisLayers(dx_cells, layer_cells['x_lo'], layer_cells['x_hi'], xp)
        self._y_layers = AcousticAxisLayers(dy_cells, layer_cells['y_lo'], layer_cells['y_hi'], xp)
        n_slab_x, n_slab_y = self._x_layers.n_slab, self._y_layers.n_slab
        self._n_slab_y_i32 = np.int32(max(n_slab_y, 1))
        # psi of d(sxx)/dx, d(sxy)/dx (velocity) and d(vx)/dx, d(vy)/dx (stress) on x slabs; the y-derivative partners
        self._psi_x = tuple(psi_array(n_slab_x, ny, True, xp) for _ in range(4))
        self._psi_y = tuple(psi_array(n_slab_y, nx, False, xp) for _ in range(4))

        shape = (nx, ny)
        field_dtype = FIELD_DTYPES[precision]
        self.fields = AcousticFields(*(xp.zeros(shape, dtype=field_dtype) for _ in range(5)),
                                     stt=xp.zeros(shape, dtype=field_dtype) if self.axisymmetric else None)
        # effective stiffness [Pa] (cells; c66c at corners) and velocity coefficients (faces), rewritten in place
        self.c11, self.c12, self.c22, self.c66c = (xp.empty(shape, dtype=DTYPE) for _ in range(4))
        self.va_x, self.vb_x, self.va_y, self.vb_y = (xp.empty(shape, dtype=DTYPE) for _ in range(4))
        self._rigid_keep = None                      # float32 (nx, ny), 0 in rigid cells; allocated by add_rigid
        self._no_keep = xp.ones(1, dtype=DTYPE)

        self._k_grid, self._k_block = _kernels.launch(nx, ny)
        self._nx_i32, self._ny_i32 = np.int32(nx), np.int32(ny)
        self._periodic_x_i32, self._periodic_y_i32 = np.int32(self.periodic_x), np.int32(self.periodic_y)

        self._velocity_sources = []
        self._stress_sources = []
        self.probes = []
        self._heating = None                         # heating_power slot array (kept by reference)
        # piezo: all six arrays when a material set one (scripts: add any of them first); planar only
        present = [name for name in PIEZO_ARRAYS if name in self.materials.property_names()]
        self.piezoelectric = bool(present)
        if self.piezoelectric:
            if self.axisymmetric:
                raise ValueError('axisymmetric elastic_waves has no piezoelectric coupling (planar sagittal plane only)')
            if precision != 'single':
                raise ValueError('half_storage elastic_waves has no piezoelectric coupling (its field stress, strain '
                                 'accumulator and bound charge use float32 fields in physical units)')
            self.materials.add_properties({name: 0.0 for name in PIEZO_ARRAYS})
        # monoclinic c16 / c26: both arrays when a material set one; planar single precision only
        self.monoclinic = any(name in self.materials.property_names() for name in MONOCLINIC_ARRAYS)
        if self.monoclinic:
            if self.axisymmetric:
                raise ValueError('axisymmetric elastic_waves has no monoclinic c16 / c26 stiffness (its hoop terms '
                                 'assume c_tt = c11, c_rt = c_zt = c12)')
            if precision != 'single':
                raise ValueError('half_storage elastic_waves has no monoclinic c16 / c26 stiffness (its strain '
                                 'accumulator and rate buffers are float32 in physical units)')
            self.materials.add_properties({name: 0.0 for name in MONOCLINIC_ARRAYS})
        # photoelastic tensor: all ten arrays (NaN = isotropic fallback) when a material set one
        self.photoelastic_tensor = any(name in self.materials.property_names() for name in PHOTOELASTIC_TENSOR_ARRAYS)
        if self.photoelastic_tensor:
            self.materials.add_properties({name: math.nan for name in PHOTOELASTIC_TENSOR_ARRAYS})
        self._electric_field_last = dict.fromkeys(ELECTRIC_FIELD_SLOTS)   # per slot: E of the last transfer
        self._piezo_shear_drive = None               # cell e_k6 dE_k, averaged to the corners
        # total strain accumulator (S_xx, S_yy at the centres, engineering shear on the corners): piezo or monoclinic
        # (elastic_strain reads it: the 2 x 2 inversion would miss c16 / c26)
        self._strain = (tuple(xp.zeros(shape, dtype=DTYPE) for _ in range(3))
                        if self.piezoelectric or self.monoclinic else None)
        # this step's strain rates (e_xx, e_yy at the centres, engineering shear on the corners), monoclinic only
        self._strain_rates = tuple(xp.zeros(shape, dtype=DTYPE) for _ in range(3)) if self.monoclinic else None
        self._bound_charge_buffers = None            # (P_x, P_y, rho_b): two-way piezo (enable_bound_charge)
        self._piezo_permittivity = None              # relative (eps_xx, eps_yy, eps_xy) of the stiffened dt
        self._pressure = None                        # pressure buffer, allocated at the first state() / render
        self._elastic_strain = None                  # (strain_xx, strain_yy) buffers, allocated at the first read
        self._shear_strain = None                    # (corner or None, centre) engineering shear, first shear read
        self.time = 0.0
        self.n_step = 0
        self.stable_dt = None
        self.dt = None
        self.reference_speed = None
        self.largest_stiffness = None                # [Pa] max(c11, c22) over the cells (half_storage scale estimate)
        self.largest_impedance = None                # [Pa s/m] max sqrt(max(c11, c22) rho) over the cells
        try:
            self._update_time_step()
        except ValueError:   # stiffness may still come from state-dependent materials: finalize decides
            self.stable_dt = None

    # --- PhysicsSolver contract -----------------------------------------------------------------------------------

    def prepare(self, domain) -> None:
        return

    def dt_limit(self) -> float:
        if self.stable_dt is None:
            self._update_time_step()   # raises naming the missing density / stiffness
        return float(self.stable_dt)

    def finalize(self) -> None:
        """Re-derive the stable time step and the CPML damping from the final materials (dynamic updates included);
        half_storage: the storage scales (before the first step only)."""
        self._update_time_step()
        if self.n_step == 0:
            self._set_storage_scales()

    def materials_changed(self, names) -> None:
        """Rebuild the coefficients in place (dt stays: faster runtime materials are not stability-checked)."""
        if not _COEFFICIENT_MATERIALS.isdisjoint(names):
            self._rebuild_coefficients()

    def step(self, t: float = None, dt: float = None) -> None:
        """One staggered step of length dt (default the stable dt; a shorter runner step rebuilds the coefficients
        once)."""
        limit = self.dt_limit()
        if dt is None or not math.isfinite(dt):
            dt = limit
        if dt > limit * (1.0 + 1e-9):
            raise ValueError(f'acoustic step {dt:.4e} s exceeds the stable time step {self.stable_dt:.4e} s')
        if abs(dt - self.dt) > 1e-12 * self.dt:
            self._configure(dt)
        self.advance(self.time if t is None else float(t), float(dt))

    def state(self):
        return _PhysicalFieldState(self)   # lazy: couplings and materials read single entries every step

    def inject_source(self, name: str, array) -> None:
        """'heating_power': float32 (nx, ny) heating rate [W/m^3] on the run backend, kept by reference (None clears
        it), turned into thermoelastic stress -gruneisen q every step (times the heating envelope when one is set).
        'electric_field_x' / '_y' [V/m] (cell centres): -e^T (E_k - E_last) goes into the stresses at once (the field
        of a slot's first transfer counts as already present: no stress); ignored without piezo arrays. half_storage
        takes no slot (ValueError): its storage scale is fixed from the features before the first step."""
        if name not in (HEATING_SLOT, *ELECTRIC_FIELD_SLOTS):
            raise ValueError(f"AcousticPhysics accepts sources '{HEATING_SLOT}', "
                             f"{', '.join(repr(slot) for slot in ELECTRIC_FIELD_SLOTS)}; got '{name}'")
        if array is not None and self.precision != 'single':
            raise ValueError(f"half_storage elastic_waves takes no coupled source ('{name}'): the fp16 storage scale is "
                             'set from the features at the build, so a coupled amplitude could overflow')
        if array is not None and not (isinstance(array, self.xp.ndarray) and array.dtype == DTYPE
                                      and array.shape == (self.nx, self.ny)):
            raise TypeError(f"inject_source('{name}', ...) needs a {self.xp.__name__} float32 array of shape "
                            f"{(self.nx, self.ny)}, got {type(array).__name__} {getattr(array, 'dtype', '')} "
                            f"{getattr(array, 'shape', '')}")
        if name == HEATING_SLOT:
            self._heating = array
        elif self.piezoelectric and array is not None:
            self._apply_field_component(name, array)

    def renderer(self):
        from .renderer import AcousticRenderer
        return AcousticRenderer(self)

    # --- time stepping --------------------------------------------------------------------------------------------

    def advance(self, t, dt) -> None:
        """v^(n+1/2) from s^n (+ forces at t_n), s^(n+1) from v^(n+1/2) (+ volume / thermoelastic sources at t_n +
        dt/2; piezo / monoclinic: the strain accumulator too; monoclinic: the c16 / c26 terms), then the probes."""
        fields = self.fields
        x_layers, y_layers = self._x_layers, self._y_layers
        psi_x, psi_y = self._psi_x, self._psi_y
        common = (self._inv_dx_e, self._inv_dx_h, self._inv_dy_e, self._inv_dy_h, x_layers.slab, y_layers.slab,
                  x_layers.profiles, y_layers.profiles)
        flags = (self._nx_i32, self._ny_i32, self._n_slab_y_i32, self._periodic_x_i32, self._periodic_y_i32)
        stresses = (fields.sxx, fields.syy, fields.sxy)
        kernels = self.kernels
        if self.axisymmetric:   # hoop stress after the stresses, radius rows after the CPML profiles
            hoop, radial = (fields.stt,), (self._radial,)
            velocity_kernel, stress_kernel = kernels['velocity_axisymmetric'], kernels['stress_axisymmetric']
        else:
            hoop = radial = ()
            velocity_kernel = kernels['velocity']
            stress_kernel = kernels[('stress' if self._strain is None else 'stress_strain')
                                    + ('_monoclinic' if self.monoclinic else '')]
        strain = () if self._strain is None else self._strain
        rates = () if self._strain_rates is None else self._strain_rates
        t_half = t + 0.5 * dt
        # the coupled heating slot at t_n + dt/2, fused into the stress kernel (thermal scale 0: no slot, dummies)
        thermal = (self._no_keep, self._no_keep, DTYPE(0.0))
        if self._heating is not None:
            envelope = 1.0 if self.heating_envelope is None else self.heating_envelope(t_half)
            if envelope > 1e-30:
                thermal = (self.materials.gruneisen, self._heating, DTYPE(dt * envelope / self.stress_scale))
        velocity_kernel(
            self._k_grid, self._k_block,
            (fields.vx, fields.vy, *stresses, *hoop, self.va_x, self.vb_x, self.va_y, self.vb_y,
             *common, *radial, psi_x[0], psi_x[1], psi_y[0], psi_y[1], self._stress_to_velocity, *flags))
        for source in self._velocity_sources:
            source.inject_velocity(self, t)
        stress_kernel(
            self._k_grid, self._k_block,
            (*stresses, *hoop, fields.vx, fields.vy, self.c11, self.c12, self.c22, self.c66c, *strain, *rates,
             *common, *radial, psi_x[2], psi_x[3], psi_y[2], psi_y[3], DTYPE(dt), self._velocity_to_stress, *thermal,
             *flags))
        if self.monoclinic:   # c16 / c26 terms from the rates the stress kernel wrote
            kernels['monoclinic_stress'](
                self._k_grid, self._k_block,
                (*stresses, *rates, self.materials.c16, self.materials.c26, self.c66c,
                 DTYPE(dt * float(self._velocity_to_stress)), self._nx_i32, self._ny_i32, self._periodic_x_i32,
                 self._periodic_y_i32))
        for source in self._stress_sources:
            source.inject_stress(self, t_half, dt)
        self.n_step += 1
        self.time = t + dt
        for probe in self.probes:
            probe.record(fields, self.time, dt)

    def _apply_field_component(self, slot, field):
        """Stress of one field component's change since its last transfer: sxx / syy at the centres, sxy at the
        corners from the 4-cell mean of e_k6 dE_k (corners without shear stiffness skipped); the first transfer only
        records E_k. Applied at the transfer, so it never depends on the order the physics step in."""
        last = self._electric_field_last[slot]
        if last is None:
            self._electric_field_last[slot] = field.copy()
            if self._piezo_shear_drive is None:
                self._piezo_shear_drive = self.xp.zeros((self.nx, self.ny), dtype=DTYPE)
            return
        materials, fields = self.materials, self.fields
        _kernels.piezo_component_stress(field, *(getattr(materials, name) for name in _PIEZO_ROWS[slot]), last,
                                        fields.sxx, fields.syy, self._piezo_shear_drive)
        _kernels.piezo_shear_stress_kernel(
            self._k_grid, self._k_block,
            (self._piezo_shear_drive, self.c66c, fields.sxy, self._nx_i32, self._ny_i32, self._periodic_x_i32,
             self._periodic_y_i32))

    def enable_bound_charge(self, permittivity_xx, permittivity_yy, permittivity_xy) -> None:
        """Two-way piezo: allocate the bound-charge buffers and re-derive dt from the stiffened c^D = c^E + e^T eps^-1 e
        (relative eps_dc arrays of the electrostatics, full grid, kept by reference)."""
        if not self.piezoelectric:
            raise ValueError("elastic_waves has no piezo_e* arrays: the two-way piezoelectric coupling needs a "
                             "piezoelectric material")
        self._piezo_permittivity = (permittivity_xx, permittivity_yy, permittivity_xy)
        if self._bound_charge_buffers is None:
            self._bound_charge_buffers = tuple(self.xp.zeros((self.nx, self.ny), dtype=DTYPE) for _ in range(3))
        self._update_time_step()

    def bound_charge(self, window, own_polarization_faces):
        """Full-grid rho_b = -div(e S) [C/m^3] on window (slice_x, slice_y; 0 outside) from the strain accumulator,
        face form; own_polarization_faces: per FACES face True = the edge face takes the cell's P (an electrode)."""
        if self._bound_charge_buffers is None:
            raise RuntimeError('bound_charge needs enable_bound_charge first')
        polarization_x, polarization_y, charge = self._bound_charge_buffers
        materials = self.materials
        _kernels.piezo_polarization_kernel(
            self._k_grid, self._k_block,
            (*self._strain, *(getattr(materials, name) for name in PIEZO_ARRAYS), polarization_x, polarization_y,
             self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32))
        window_x, window_y = window
        bounds = (window_x.start, window_x.stop, window_y.start, window_y.stop)
        _kernels.piezo_bound_charge_kernel(
            self._k_grid, self._k_block,
            (polarization_x, polarization_y, self._inv_dx_e, self._inv_dy_e, charge, self._nx_i32, self._ny_i32,
             *(np.int32(value) for value in bounds), *(np.int32(bool(own)) for own in own_polarization_faces)))
        return charge

    def pressure(self):
        """Full-grid float32 p = -(sxx + syy) / 2 [Pa] (axisymmetric: -(srr + szz + stt) / 3) in a persistent buffer
        (rewritten at each call)."""
        if self._pressure is None:
            self._pressure = self.xp.empty((self.nx, self.ny), dtype=DTYPE)
        fields = self.fields
        scale = DTYPE(self.stress_scale)
        if self.axisymmetric:
            _kernels.pressure_from_stress_axisymmetric(fields.sxx, fields.syy, fields.stt, scale, self._pressure)
        else:
            _kernels.pressure_from_stress(fields.sxx, fields.syy, scale, self._pressure)
        return self._pressure

    def physical_array(self, name):
        """Full-grid float32 field `name` of AcousticFields in m/s / Pa: the array itself in single, a fresh copy
        under half_storage; ValueError for stt in planar geometry."""
        stored = getattr(self.fields, name)
        if stored is None:
            raise ValueError(f"elastic_waves field '{name}' exists only with geometry axisymmetric")
        if self.precision == 'single':
            return stored
        physical = self.xp.empty(stored.shape, dtype=DTYPE)
        _kernels.stored_to_physical(stored, DTYPE(self._storage_scale(name)), physical)
        return physical

    def elastic_strain(self):
        """Full-grid (strain_xx, strain_yy) [1] = the effective in-plane stiffness inverted on the centre stresses
        (fluids: half the volumetric strain each; axisymmetric: the 3x3 normal block with stt, fluids a third each);
        piezo or monoclinic c16 / c26: the accumulated total strain instead. Persistent buffers rewritten at each call."""
        if self._elastic_strain is None:
            self._elastic_strain = tuple(self.xp.empty((self.nx, self.ny), dtype=DTYPE) for _ in range(2))
        strain_xx, strain_yy = self._elastic_strain
        fields = self.fields
        if self._strain is not None:
            self.xp.copyto(strain_xx, self._strain[0])
            self.xp.copyto(strain_yy, self._strain[1])
        elif self.axisymmetric:
            _kernels.elastic_strain_from_stress_axisymmetric(fields.sxx, fields.syy, fields.stt, self.c11, self.c12,
                                                             self.c22, DTYPE(self.stress_scale), strain_xx, strain_yy)
        else:
            _kernels.elastic_strain_from_stress(fields.sxx, fields.syy, self.c11, self.c12, self.c22,
                                                DTYPE(self.stress_scale), strain_xx, strain_yy)
        return self._elastic_strain

    def elastic_shear_strain(self):
        """Full-grid engineering shear 2 S_xy [1] at the cell centres = the 4-corner mean (missing lo corners 0) of the
        corner shear: the strain accumulator's when present, else sxy / c66c (0 where c66c = 0). Persistent buffer."""
        if self._shear_strain is None:
            corner = None if self._strain is not None else self.xp.empty((self.nx, self.ny), dtype=DTYPE)
            self._shear_strain = (corner, self.xp.empty((self.nx, self.ny), dtype=DTYPE))
        corner, centre = self._shear_strain
        if corner is None:
            corner = self._strain[2]
        else:
            _kernels.corner_shear_strain(self.fields.sxy, self.c66c, DTYPE(self.stress_scale), corner)
        _kernels.centre_shear_strain_kernel(
            self._k_grid, self._k_block,
            (corner, centre, self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32))
        return centre

    # --- coefficients and time step -------------------------------------------------------------------------------

    def _rebuild_coefficients(self):
        materials = self.materials
        keep = self._no_keep if self._rigid_keep is None else self._rigid_keep
        _kernels.coefficients_kernel(
            self._k_grid, self._k_block,
            (materials.rho, materials.sound_speed, materials.shear_speed, materials.youngs_modulus,
             materials.poisson_ratio, materials.c11, materials.c12,
             materials.c22, materials.c66, materials.acoustic_damping, keep, np.int32(self._rigid_keep is not None),
             *self._shares, self.c11, self.c12, self.c22, self.c66c, self.va_x, self.vb_x, self.va_y, self.vb_y,
             DTYPE(self.dt if self.dt is not None else 1.0), self._nx_i32, self._ny_i32,
             self._periodic_x_i32, self._periodic_y_i32, *self._pins))

    def _configure(self, dt, coefficients_current=False):
        """Coefficients and CPML recursions for time step dt [s] (coefficients_current: the velocity coefficients were
        just built with this dt, no second pass)."""
        self.dt = float(dt)
        self._x_layers.configure(self.reference_speed, self.pml_reflection, self.dt)
        self._y_layers.configure(self.reference_speed, self.pml_reflection, self.dt)
        if not coefficients_current:
            self._rebuild_coefficients()

    def _update_time_step(self):
        """stable_dt = courant / (c_bound sqrt(1/dx_min^2 + 1/dy_min^2)), c_bound^2 = max over cells of the stability
        stiffness (`kernels.stability_stiffness`, anisotropic von Neumann bound, c16 / c26 included) times the largest
        buoyancy 1 / rho_face of its four faces (a solid-vacuum face has rho / 2; axisymmetric: 1 / dx_min^2 times
        `radial_stability_factor`; two-way piezo: the stiffened c^D); the CPML damping uses the largest sqrt(max(c11,
        c22) / rho). One host sync (build time)."""
        built_dt = self.dt
        self._rebuild_coefficients()   # effective c11 / c12 / c22 / c66c (dt-independent part; va / vb at self.dt)
        xp = self.xp
        rho = self.materials.rho
        longitudinal = xp.maximum(xp.maximum(self.c11, self.c22), DTYPE(0.0))
        reference_squared = xp.where(rho > 0, longitudinal / xp.maximum(rho, DTYPE(1e-30)), DTYPE(0.0)).max()
        largest_stiffness = longitudinal.max()
        impedance_squared = (longitudinal * xp.maximum(rho, DTYPE(0.0))).max()
        del longitudinal
        corner_shear = xp.maximum(self.c66c, xp.roll(self.c66c, 1, axis=0))   # largest c66c of the 4 cell corners
        xp.maximum(corner_shear, xp.roll(corner_shear, 1, axis=1), out=corner_shear)
        if self._piezo_permittivity is None:   # no stiffening: e = 0, eps = 1
            stiffening = (*(DTYPE(0.0) for _ in PIEZO_ARRAYS), DTYPE(1.0), DTYPE(1.0), DTYPE(0.0))
        else:
            stiffening = (*(getattr(self.materials, name) for name in PIEZO_ARRAYS), *self._piezo_permittivity)
        monoclinic = (tuple(getattr(self.materials, name) for name in MONOCLINIC_ARRAYS) if self.monoclinic
                      else (DTYPE(0.0), DTYPE(0.0)))
        stiffness = _kernels.stability_stiffness(self.c11, self.c12, self.c22, corner_shear, *monoclinic, *stiffening,
                                                 np.float64(1.0 / EPS0), *self._stiffness_weights)
        del corner_shear
        largest_buoyancy = xp.zeros_like(rho)
        shares = self._shares
        for axis, periodic, share_lo, share_hi in ((0, self.periodic_x, shares[0], shares[1]),
                                                  (1, self.periodic_y, shares[2], shares[3])):
            lo_shape = (-1, 1) if axis == 0 else (1, -1)
            if periodic:
                following = xp.roll(rho, -1, axis=axis)
            else:
                following = xp.concatenate((rho[1:], rho[-1:]) if axis == 0 else (rho[:, 1:], rho[:, -1:]), axis=axis)
            face_rho = share_lo.reshape(lo_shape) * rho + share_hi.reshape(lo_shape) * following
            buoyancy = xp.where(face_rho > 0, 1.0 / xp.maximum(face_rho, DTYPE(1e-30)), DTYPE(0.0)).astype(DTYPE)
            if not periodic:   # pinned outer hi face
                if axis == 0:
                    buoyancy[-1, :] = 0.0
                else:
                    buoyancy[:, -1] = 0.0
            xp.maximum(largest_buoyancy, buoyancy, out=largest_buoyancy)
            xp.maximum(largest_buoyancy, xp.roll(buoyancy, 1, axis=axis), out=largest_buoyancy)
        bound_squared = (stiffness * largest_buoyancy).max()
        bound_squared, reference_squared, largest_stiffness, impedance_squared = (
            float(value) for value in to_host(xp.stack((bound_squared, reference_squared, largest_stiffness,
                                                         impedance_squared))))
        self.largest_stiffness, self.largest_impedance = largest_stiffness, math.sqrt(max(impedance_squared, 0.0))
        if not (math.isfinite(bound_squared) and bound_squared > 0.0):
            raise ValueError('no cell has both a density (rho) and a stiffness (sound_speed, youngs_modulus or c11 / '
                             'c22): nothing can '
                             'carry an acoustic wave' if math.isfinite(bound_squared) else
                             'a material gives a non-finite acoustic stiffness or density')
        self.stable_dt = self.courant / (math.sqrt(bound_squared) * self._inverse_cell_diagonal)
        self.reference_speed = math.sqrt(reference_squared)
        self._configure(self.stable_dt, coefficients_current=built_dt == self.stable_dt)   # unchanged dt: one pass

    def _set_storage_scales(self):
        """half_storage: stress_scale = a power of two placing the sources' estimated peak stress (`peak_stress`,
        within ~10x) at HALF_STORAGE_PEAK stored units, velocity_scale = stress_scale / the power of two nearest the
        largest impedance (a plane wave's stress and velocity alike in stored size); the probes read back scaled."""
        if self.precision == 'single':
            return
        if self.n_step:
            raise RuntimeError('the storage scales of a half_storage elastic_waves are fixed once it has stepped')
        peaks = [source.peak_stress(self) for source in self._velocity_sources + self._stress_sources]
        peak = max((value for value in peaks if value > 0.0 and math.isfinite(value)), default=HALF_STORAGE_PEAK)
        self.stress_scale = 2.0 ** round(math.log2(peak / HALF_STORAGE_PEAK))
        impedance = self.largest_impedance if self.largest_impedance else 1.0
        self.velocity_scale = self.stress_scale / 2.0 ** round(math.log2(impedance))
        self._stress_to_velocity = DTYPE(self.stress_scale / self.velocity_scale)
        self._velocity_to_stress = DTYPE(self.velocity_scale / self.stress_scale)
        for probe in self.probes:
            probe.value_scale = self._storage_scale(probe.field_component)

    def _storage_scale(self, component):
        """Physical value per stored unit of a field / probe component (velocities m/s, the rest Pa)."""
        return self.velocity_scale if component in VELOCITY_COMPONENTS else self.stress_scale

    # --- features -------------------------------------------------------------------------------------------------

    def _face_entries(self, axis, i, j, value):
        """[(flat index, weight)] of value split over the lo and hi faces normal to axis (0: x, 1: y) of cell (i, j),
        / dual cell width (axisymmetric x faces: value per unit ring length at r_c, so each face takes r_c / r_face)."""
        entries = []
        along, count, periodic = (i, self.nx, self.periodic_x) if axis == 0 else (j, self.ny, self.periodic_y)
        inverse_widths = self._inv_dx_h_host if axis == 0 else self._inv_dy_h_host
        lo = along - 1 if along >= 1 else (count - 1 if periodic else None)
        for face in (lo, along):
            if face is None:
                continue
            ring = (self._centre_radii_host[i] / self._face_radii_host[face]
                    if axis == 0 and self.axisymmetric else 1.0)
            index = face * self.ny + j if axis == 0 else i * self.ny + face
            entries.append((index, 0.5 * value * ring * inverse_widths[face]))
        return entries

    @staticmethod
    def _unzip(entries):
        return [index for index, _ in entries], [weight for _, weight in entries]

    def add_force_source(self, ix, iy, temporal, force_x, force_y):
        """Line force [N/m per unit waveform; axisymmetric per unit ring length] in cell (ix, iy), half on each of its
        two x faces (force_x) and y faces (force_y)."""
        x_entries = [(index, weight * self._inv_dy_e_host[iy])
                     for index, weight in self._face_entries(0, ix, iy, float(force_x))] if force_x else []
        y_entries = [(index, weight * self._inv_dx_e_host[ix])
                     for index, weight in self._face_entries(1, ix, iy, float(force_y))] if force_y else []
        # near-field stress ~ force over the cell size (half_storage scale estimate)
        peak = math.hypot(force_x, force_y) * max(self._inv_dx_e_host[ix], self._inv_dy_e_host[iy])
        source = VelocitySource(temporal, self._unzip(x_entries), self._unzip(y_entries), peak, self.xp)
        self._velocity_sources.append(source)
        return source

    def add_volume_source(self, ix, iy, temporal, volume_rate):
        """Volume injection rate [m^2/s per unit waveform, per unit z (axisymmetric: ring) length] in cell (ix, iy)."""
        weight = float(volume_rate) * self._inv_dx_e_host[ix] * self._inv_dy_e_host[iy]
        source = VolumeSource(temporal, [ix * self.ny + iy], [weight], self.xp)
        self._stress_sources.append(source)
        return source

    def add_traction_line(self, line_axis, position, span, temporal, traction, direction):
        """Traction sheet [Pa per unit waveform] at the centres of the line cells: line_axis 'y' = a column (cell
        `position` along x, rows span), 'x' = a row; direction 'x' / 'y' = the force component. The normal component is
        split over the two faces around each line cell, the tangential one sits on the faces along the line."""
        start, stop = int(span[0]), int(span[1])
        x_entries, y_entries = [], []
        traction = float(traction)
        for along in range(start, stop):
            i, j = (position, along) if line_axis == 'y' else (along, position)
            if line_axis == 'y' and direction == 'x':     # normal to a column
                x_entries += self._face_entries(0, i, j, traction)
            elif line_axis == 'y':                        # tangential along a column: vy faces of the column
                y_entries.append((i * self.ny + j, traction * self._inv_dx_e_host[i]))
            elif direction == 'y':                        # normal to a row
                y_entries += self._face_entries(1, i, j, traction)
            else:                                         # tangential along a row: vx faces of the row
                x_entries.append((i * self.ny + j, traction * self._inv_dy_e_host[j]))
        source = VelocitySource(temporal, self._unzip(x_entries), self._unzip(y_entries), abs(traction), self.xp)
        self._velocity_sources.append(source)
        return source

    def add_heat_pulse(self, mask, power_density, envelope):
        """Thermoelastic source: power_density [W/m^3] (number or full-grid array) on the bool mask (run backend) times
        the envelope; None for an empty mask."""
        box = _bounding_box(mask)
        if box is None:
            return None
        if isinstance(power_density, self.xp.ndarray) and power_density.ndim == 2:
            # where, not mask * value: values outside the mask may be NaN
            heating = self.xp.where(mask[box], power_density[box], DTYPE(0.0)).astype(DTYPE, copy=False)
        else:
            heating = mask[box].astype(DTYPE) * DTYPE(power_density)
        source = HeatPulseSource(box, heating, envelope)
        self._stress_sources.append(source)
        return source

    def add_rigid(self, mask):
        """Rigid (immovable) cells: the velocities on every face of a masked cell stay 0."""
        if self._rigid_keep is None:
            self._rigid_keep = self.xp.ones((self.nx, self.ny), dtype=DTYPE)
        self._rigid_keep *= (~mask).astype(DTYPE)
        self._rebuild_coefficients()

    def set_initial_velocity(self, mask, v_x=0.0, v_y=0.0):
        """Velocity [m/s] at t = 0 (the state the first step starts from) on the +x / +y faces of the cells under the
        full-grid bool mask: scalars or full-grid arrays of the face values (vx at the +x face centre, vy at the +y
        one); pinned faces (walls, rigid cells) drop it at the first step. Single precision, before the first step."""
        if self.precision != 'single':
            raise ValueError('an initial velocity needs the single-precision stepper (half_storage fixes its storage '
                             'scale from the sources only)')
        if self.n_step:
            raise ValueError('set_initial_velocity must come before the first step')
        xp = self.xp
        mask = xp.asarray(mask)
        if mask.shape != (self.nx, self.ny):
            raise ValueError(f'initial velocity mask must be a full-grid {(self.nx, self.ny)} array, got {mask.shape}')
        for target, value in ((self.fields.vx, v_x), (self.fields.vy, v_y)):
            xp.copyto(target, xp.asarray(value, dtype=DTYPE), where=mask)

    def add_probe(self, ix, iy, component):
        probe = AcousticProbe(ix, iy, component, self.xp, axisymmetric=self.axisymmetric)
        probe.value_scale = self._storage_scale(component)
        self.probes.append(probe)
        return probe

    def source_waveforms(self):
        """Waveforms / heating envelopes of the sources and of the coupled heating slot (a `duration` when pulsed)."""
        waveforms = [getattr(source, 'temporal', getattr(source, 'envelope', None))
                     for source in self._velocity_sources + self._stress_sources]
        return waveforms + ([self.heating_envelope] if self.heating_envelope is not None else [])

"""Contract members shared by the 2D FDFD physics on Yee grids (FDFDPhysics) and on triangles (FEMFDFDPhysics):
HarmonicEMShell holds omega, the float32 state arrays, the permittivity slots (with hz eps_xy_update), the loss
protocol and the stationary / frequency_domain hooks."""
import math

from ....core.constants import C0, DTYPE
from ..index_reference import OFF_DIAGONAL_ARRAY, OFF_DIAGONAL_SLOT
from ..placement import PERMITTIVITY_ARRAYS, PERMITTIVITY_SLOTS

COMPONENT_UNITS = {'ez': 'V/m', 'hx': 'A/m', 'hy': 'A/m', 'hz': 'A/m', 'ex': 'V/m', 'ey': 'V/m'}
# MaterialGrid arrays the assembled system and the sheet-source currents read (a change reassembles)
SYSTEM_MATERIALS = {'ez': frozenset({'eps_r', 'sigma_e', 'mu_r', 'sigma_m'}),
                    'hz': frozenset({'eps_xx', 'eps_yy', OFF_DIAGONAL_ARRAY, 'sigma_e', 'mu_r', 'sigma_m'})}


def harmonic_state_names(polarization) -> tuple:
    """Float32 state arrays of a 2D FDFD physics: <out-of-plane>_re, _im and q_abs."""
    return f'{polarization}_re', f'{polarization}_im', 'q_abs'


class HarmonicEMShell:
    """Mixin of the 2D FDFD contract: omega (a new value marks the system dirty), state(), inject_source (hz eps_xy
    change kept apart), materials_changed, take_absorbed_power, step / solve_harmonic, eps_xy(); the class gives
    _solve() and _mark_system_dirty(ports=True, omega_only=False)."""

    def _init_harmonic_shell(self, wavelength, shape=None, fields=None):
        """omega of the wavelength setting, no eps_xy change yet, and the float32 state arrays (fields: an existing
        {name: array} of harmonic_state_names, else zeros of `shape`)."""
        wavelength = float(wavelength)
        if not (math.isfinite(wavelength) and wavelength > 0.0):
            raise ValueError(f'{type(self).__name__} wavelength must be finite and > 0 [m], got {wavelength!r}')
        self._omega = 2.0 * math.pi * C0 / wavelength
        self._coupled_eps_xy = None   # hz float32 eps_xy_update change (allocated by the first transfer)
        names = harmonic_state_names(self.polarization)
        if fields is None:
            fields = {name: self.xp.zeros(shape, dtype=DTYPE) for name in names}
        self._field_re, self._field_im, self._q_abs = (fields[name] for name in names)

    @property
    def permittivity_arrays(self) -> tuple:
        """MaterialGrid permittivity arrays the system reads: ('eps_r',) for ez, ('eps_xx', 'eps_yy') for hz."""
        return PERMITTIVITY_ARRAYS[self.polarization]

    @property
    def permittivity_slots(self) -> dict:
        """Source slot -> permittivity array (hz eps_xy_update is kept apart: eps_xy())."""
        return PERMITTIVITY_SLOTS[self.polarization]

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s] of the next solve (the wavelength setting, or the study)."""
        return self._omega

    @omega.setter
    def omega(self, omega):
        omega = float(omega)
        if not (math.isfinite(omega) and omega > 0.0):
            raise ValueError(f'{type(self).__name__} omega must be finite and > 0 [rad/s], got {omega!r}')
        if omega != self._omega:
            self._omega = omega
            self._mark_system_dirty(omega_only=True)

    # --- PhysicsSolver contract ---

    def prepare(self, domain):
        return

    def step(self, t: float = None, dt: float = None):
        """Solve at omega when the system or the sources changed since the last solve (stationary study, flush)."""
        if self._solution_dirty:
            self._solve()

    def dt_limit(self) -> float:
        return math.inf

    def state(self) -> dict:
        """Float32 <field>_re / _im and q_abs [W/m^3] of the last solve (zeros before), the permittivity arrays, sigma_e
        and 'q_accum': None (loss-protocol key, never a field)."""
        materials = self.domain.materials
        re_name, im_name, _ = harmonic_state_names(self.polarization)
        state = {re_name: self._field_re, im_name: self._field_im, 'q_abs': self._q_abs}
        for name in self.permittivity_arrays:
            state[name] = getattr(materials, name)
        state['sigma_e'] = materials.sigma_e
        state['q_accum'] = None
        return state

    def _slot_names(self) -> tuple:
        slots = tuple(self.permittivity_slots)
        return (*slots, OFF_DIAGONAL_SLOT) if self.polarization == 'hz' else slots

    def inject_source(self, name: str, array):
        """Copy a coupling's array into the permittivity array of slot `name` (ez 'eps_r_update'; hz 'eps_xx_update',
        'eps_yy_update'; hz 'eps_xy_update' = a change added to the material eps_xy) and reassemble at the next solve
        (no per-step soft-source slot: it has no harmonic meaning)."""
        xp = self.xp
        if self.polarization == 'hz' and name == OFF_DIAGONAL_SLOT:
            if self._coupled_eps_xy is None:
                self._coupled_eps_xy = xp.zeros(self._field_re.shape, dtype=DTYPE)
            xp.copyto(self._coupled_eps_xy, xp.asarray(array, dtype=DTYPE))
            self._mark_system_dirty(ports=False)   # port modes never see eps_xy
            return
        slots = self.permittivity_slots
        if name not in slots:
            raise ValueError(f"{type(self).__name__} ({self.polarization}) accepts sources "
                             f"{', '.join(self._slot_names())}; got '{name}'")
        xp.copyto(getattr(self.domain.materials, slots[name]), xp.asarray(array, dtype=DTYPE))
        self._mark_system_dirty()

    def materials_changed(self, names) -> None:
        if not SYSTEM_MATERIALS[self.polarization].isdisjoint(names):
            self._mark_system_dirty()

    def take_absorbed_power(self, merge_previous: bool = False):
        """EM loss protocol: q_abs [W/m^3] of the last solve (float32 run backend, overwritten by the next), None before
        the first solve; merge_previous is ignored (no accumulation window)."""
        return None if self.solved_omega is None else self._q_abs

    def solve_harmonic(self, omega):
        """Solve now at `omega` [rad/s] (frequency_domain study hook; reassembles only when omega or materials
        changed) and record the monitors."""
        self.omega = omega
        self._solve()

    def eps_xy(self):
        """hz float32 off-diagonal permittivity as given: material eps_xy + the eps_xy_update change (fresh array, zeros
        without either)."""
        materials = self.domain.materials
        total = self.xp.zeros(self._field_re.shape, dtype=DTYPE)
        if OFF_DIAGONAL_ARRAY in materials.property_names():
            total += getattr(materials, OFF_DIAGONAL_ARRAY)
        if self._coupled_eps_xy is not None:
            total += self._coupled_eps_xy
        return total

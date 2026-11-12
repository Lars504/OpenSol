"""FDTDEngine2D_Ez: the 2D Yee engine of E_z polarisation (E_z, H_x, H_y) with fused kernels, CPML / BCs, Drude /
Lorentz poles, PEC masks and J.E loss accounting."""
import numpy as np

from ....core.constants import DTYPE, EPS0
from . import fdtd_ez_2d_kernels as _kernels
from .fdtd_2d_base import _YeeEngine2DBase, _bounding_box, _union_box
from .fdtd_conformal_kernels import conformal_h_ez, sparse_launch


class FDTDEngine2D_Ez(_YeeEngine2DBase):
    """2D Ez-polarisation Yee engine (Ez, Hx, Hy) with fused kernels, CPML/BCs, Drude/Lorentz poles, PEC masks and J·E loss accounting."""

    polarization = 'ez'
    _E_COEFFICIENT_MATERIALS = frozenset({'eps_r', 'sigma_e'})
    subpixel_smoothing = None   # physics/em/subpixel_smoothing.SubpixelSmoothing (EMPhysics.set_subpixel_smoothing)
    _conformal_faces = None     # (face cell, axis, extra device arrays, launch) of set_conformal_faces, None = off

    def __init__(self, grid, fields, materials, boundary_manager):
        # Base: curl spacings (*_e for the E_z rows, *_h for the H_y / H_x faces), launch scalars, da / db, clock and
        # the J.E window (create_em_physics and MultiphysicsRunner switch enable_q_accum on).
        super().__init__(grid, fields, materials, boundary_manager)

        # ca / cb (cb is also the Drude and Lorentz correction factor) are rewritten in place by one fused kernel, so
        # material updates never reallocate them.
        shape = (self.nx, self.ny)
        self.ca = self.xp.empty(shape, dtype=self.dtype)
        self.cb = self.xp.empty(shape, dtype=self.dtype)
        # float32 (nx, ny), also under double (0 / 1 are exact): 0 inside any PEC mask, 1 elsewhere; None until a PEC
        # region exists. Every E-coefficient rebuild folds it into ca / cb (Yee update, CPML psi and pole corrections
        # leave PEC cells at 0); advance re-zeroes only _pec_box (bounding box of the PEC cells) after injection.
        self._pec_keep = None
        self._pec_box = None
        self._rebuild_e_coefficients()
        self._rebuild_h_coefficients()

        # Dispersive poles: coefficient and auxiliary arrays on the bounding box of their mask ('box').
        self.drude_poles = []
        self.lorentz_poles = []

        # Coupling-injected source (set by inject_source('ez_source', ...))
        self._out_of_plane_source = None

        # Per field set (index 0 real, 1 imaginary under Bloch k != 0): the sum of every Drude pole's J^{n+1/2}
        # [A/m^2] (full grid, allocated with the first Drude pole), rebuilt each step on _drude_box, the union of the
        # Drude boxes (the E correction, the J.E loss and state()['jz_total'] (real set) read it). Pole auxiliary arrays
        # ('j', 'p', 'dp') are lists indexed by the field set too.
        self._jz_totals = [None] * len(self._field_sets)
        self._drude_box = None
        # _ohmic_box: bounding box of the sigma_e > 0 cells (ohmic loss), refreshed by finalize / materials_changed.
        self._sigma_e_changed()

    # --- PhysicsSolver contract -----------------------------------------------

    def state(self) -> dict:
        """Public fields; jz_total is None without a Drude pole, q_accum None before accumulation is enabled."""
        return {
            'ez': self.fields.ez,
            'hx': self.fields.hx,
            'hy': self.fields.hy,
            'eps_r': self.materials.eps_r,
            'sigma_e': self.materials.sigma_e,
            'jz_total': self._jz_totals[0],
            'q_accum': self.q_accum,
            'q_accum_time': self.q_accum_time,
        }

    # --- time stepping ------------------------------------------------------------

    def advance(self, sources=(), monitors=()) -> None:
        """One iteration: H update, TFSF H side / magnetic corrections at t, H walls, E update, TFSF E side / electric
        corrections / soft sources at t+dt/2 (each source at the centre time of its update), E_z walls, PEC, E^{n+1}
        loss half, then monitors at t+dt. Sources drive the real field set (Bloch line sources also the imaginary one);
        walls and PEC act on every set."""
        fields = self.fields
        t_start = float(self.time)
        t_half = t_start + 0.5 * float(self.dt)
        t_full = t_start + float(self.dt)
        self.update_h()
        for source in sources:
            if hasattr(source, 'inject_h'):
                source.inject_h(fields.hx, fields.hy)
            elif hasattr(source, 'inject_magnetic'):
                source.inject_magnetic(fields, t_start)
        # The pmc H rows are pinned again after the magnetic injections (as H_z does).
        for field_set in self._field_sets:
            field_set.boundary_mgr.apply_h_walls(field_set.fields.hx, field_set.fields.hy)

        self.update_e()

        for source in sources:
            if hasattr(source, 'inject_e'):
                source.inject_e(fields.ez, t_half)
            elif hasattr(source, 'inject_electric'):
                source.inject_electric(fields, t_half)
            else:
                source.inject(fields.ez, t_half)
        self._inject_imaginary_sources(sources, t_half)

        if self._out_of_plane_source is not None:   # inject_source('ez_source', ...)
            fields.ez += self._out_of_plane_source
        # The outer-face rows are pinned again after injection (as H_z does), so a source value in the outermost
        # cell of a pec / pmc / mirror face does nothing and a mirror image follows a source next to its plane.
        # ca = cb = 0 already pins the PEC cells through the update; the keep catches injections and wall re-pins there.
        pec_box = self._pec_box
        for field_set in self._field_sets:
            ez = field_set.fields.ez
            field_set.boundary_mgr.apply_ez_walls(ez)
            if pec_box is not None:   # in place on the view (no setitem copy back)
                pinned = ez[pec_box]
                self.xp.multiply(pinned, self._pec_keep[pec_box], out=pinned)
        self._finish_step(t_full, monitors)

    # --- EM-specific API ------------------------------------------------------

    def add_drude(self, drude_pole, mask=None, set_background=True, weight=None):
        """Drude pole on the run-backend bool (nx, ny) cell `mask` (None = everywhere); set_background writes eps_r = eps_inf
        under the mask; weight = host float (nx, ny) pole share per cell instead of the mask (conformal_poles)."""
        mask = self._pole_cell_mask(mask)
        damping, drive = drude_pole.ade_coefficients(float(self.dt))

        # The mask (or weight) and the PEC keep are folded into the drive array on the pole's bounding box: drive = 0
        # pins J at 0 there, so its E correction and J.E loss vanish outside the metal.
        weight_box = self._pole_weight_box(mask, weight)
        if weight_box is not None:
            box, weight = weight_box
            real = self._real
            self.drude_poles.append({'pole': drude_pole, 'box': box, 'damping': real(damping),
                                     'drive': weight * real(drive),
                                     'j': self._set_buffers(weight.shape)})
            if self._jz_totals[0] is None:
                self._jz_totals = [self.xp.zeros((self.nx, self.ny), dtype=self.dtype) for _ in self._field_sets]
            self._drude_box = _union_box(self._drude_box, box)

        if set_background:
            self.materials.assign('eps_r', mask, drude_pole.eps_inf)
            self._rebuild_e_coefficients()

    def add_lorentz(self, lorentz_pole, mask=None, weight=None):
        """Lorentz pole on the run-backend bool (nx, ny) cell `mask` (None = everywhere); eps_r acts as the background
        eps_inf; weight = host float (nx, ny) pole share per cell instead of the mask (conformal_poles)."""
        mask = self._pole_cell_mask(mask)
        damping, restoring, drive = lorentz_pole.ade_coefficients(float(self.dt))

        # The mask (or weight) and the PEC keep are folded into the drive array on the pole's bounding box: drive = 0
        # keeps p and dp pinned at 0, so its E correction and J.E loss vanish outside the material.
        weight_box = self._pole_weight_box(mask, weight)
        if weight_box is None:
            return
        box, weight = weight_box
        real = self._real
        self.lorentz_poles.append({
            'pole': lorentz_pole, 'box': box,
            'damping': real(damping), 'restoring': real(restoring), 'drive': weight * real(drive),
            'p': self._set_buffers(weight.shape), 'dp': self._set_buffers(weight.shape),
        })

    def _pole_weight_box(self, mask, cell_weight=None):
        """(bounding box of `mask` (or of the host cell_weight > 0), its values on it in the field dtype times the PEC
        keep), None when empty (one host sync)."""
        if cell_weight is not None:
            cell_weight = np.asarray(cell_weight, dtype=np.float64)
            self._check_weight_shape(cell_weight, 'pole weight')
        box = _bounding_box(mask if cell_weight is None else cell_weight)
        if box is None:
            return None
        weight = (mask[box].astype(self.dtype) if cell_weight is None
                  else self.xp.asarray(cell_weight[box], dtype=self.dtype))
        if self._pec_keep is not None:
            weight *= self._pec_keep[box]
        return box, weight

    def add_pec_mask(self, mask):
        """Perfect electric conductor on the run-backend bool (nx, ny) `mask`: Ez and dispersive currents pinned to 0."""
        self._check_cell_mask(mask, 'PEC mask')
        keep = (~mask).astype(DTYPE)
        if self._pec_keep is None:
            self._pec_keep = keep
        else:
            self._pec_keep *= keep
        self._pec_box = _union_box(self._pec_box, _bounding_box(mask))
        self._fold_pec_keep()
        for field_set in self._field_sets:
            field_set.fields.ez *= keep
        for drude in self.drude_poles:
            keep_box = keep[drude['box']]
            drude['drive'] *= keep_box
            for current in drude['j']:
                current *= keep_box
        for lorentz in self.lorentz_poles:
            keep_box = keep[lorentz['box']]
            lorentz['drive'] *= keep_box
            for buffer in (*lorentz['p'], *lorentz['dp']):
                buffer *= keep_box

    def set_conformal_faces(self, face_cell, face_axis, face_extra):
        """Conformal PEC (physics/em/conformal_pec): each listed H face (host int32 lower cell, int32 axis 0 = hi-x H_y /
        1 = hi-y H_x) adds extra x its Yee curl term every step (wall at d / (1 + extra)); empty lists switch it off."""
        if len(face_cell) == 0:
            self._conformal_faces = None
            return
        xp = self.xp
        self._conformal_faces = (xp.asarray(face_cell, dtype=np.int32), xp.asarray(face_axis, dtype=np.int32),
                                 xp.asarray(face_extra, dtype=self.dtype), sparse_launch(len(face_cell)))

    def _sigma_e_changed(self):
        """Bounding box of the sigma_e > 0 cells for the ohmic loss (one host sync; never on the eps_r_update path)."""
        self._ohmic_box = _bounding_box(self.materials.sigma_e)

    def _rebuild_e_coefficients(self):
        # One fused kernel writes ca / cb in place: no allocations on the thermo-optic 'eps_r_update' path.
        _kernels.e_coefficients(
            self.materials.sigma_e, self.materials.eps_r,
            self._dt_real, self._real(EPS0),
            self.ca, self.cb)
        if self.subpixel_smoothing is not None:   # averaged eps on the interface cells, before the PEC fold
            self.subpixel_smoothing.correct_fdtd(self)
        self._fold_pec_keep()

    def _fold_pec_keep(self):
        """ca = cb = 0 on PEC cells (on _pec_box)."""
        pec_box = self._pec_box
        if pec_box is not None:
            keep_box = self._pec_keep[pec_box]
            self.ca[pec_box] *= keep_box
            self.cb[pec_box] *= keep_box

    def update_h(self):
        """Yee H update + CPML / BC post of every field set (each reads only E, so the order of the sets is free): Yee,
        conformal faces, CPML and the fused outer-face rows as one step graph on the gpu (row copies after it)."""
        conformal = self._conformal_faces
        self._fixed_block(('h', id(conformal)), (conformal,), self._h_block)
        for field_set in self._field_sets:
            if not field_set.boundary_mgr.fused_walls:
                field_set.boundary_mgr.apply_h_walls(field_set.fields.hx, field_set.fields.hy)

    def _h_block(self):
        """H Yee (+ the conformal faces), the H CPML and the fused H rows of every field set: fixed arguments only
        (_fixed_block)."""
        for field_set in self._field_sets:
            fields, partner = field_set.fields, field_set.partner_fields
            _kernels.update_h_kernel(
                self._k_grid, self._k_block,
                (fields.ez, fields.hx, fields.hy, self.da, self.db,
                 self._inv_dx_h_1d, self._inv_dy_h_1d,
                 self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32,
                 partner.ez, self._wrap_cos_x, field_set.wrap_sin_x, self._wrap_cos_y, field_set.wrap_sin_y))
            if self._conformal_faces is not None:
                face_cell, face_axis, face_extra, (grid, block, count) = self._conformal_faces
                conformal_h_ez(grid, block, (fields.ez, fields.hx, fields.hy, self.db, self._inv_dx_h_1d,
                                             self._inv_dy_h_1d, face_cell, face_axis, face_extra, count, self._ny_i32))
            self._cpml_post(field_set, 'h', fields.ez, fields.hx, fields.hy, self.db)
            if field_set.boundary_mgr.fused_walls:   # one launch, no temporary: capturable
                field_set.boundary_mgr.apply_h_walls(fields.hx, fields.hy)

    def _update_e_set(self, field_set):
        """update_e of one field set (the E kernel reads only H, so the sets may follow each other)."""
        fields, k = field_set.fields, field_set.index
        ez, hx, hy = fields.ez, fields.hx, fields.hy
        partner = field_set.partner_fields
        jz_total = self._jz_totals[k]
        accumulate = self.enable_q_accum   # never with two field sets (enable_q_accum refuses Bloch k != 0)
        nonlinear = self.nonlinear   # never with two field sets either (add_nonlinear refuses Bloch k != 0)
        if nonlinear is not None:
            nonlinear.store_previous()

        # Both poles are stepped from the pre-update E^n on their boxes (kernel notes in fdtd_ez_2d_kernels): Drude
        # leap-frog J^{n+1/2} = J^{n-1/2} + damping J^{n-1/2} + drive E^n summed into jz_total on _drude_box (mask and
        # PEC keep folded into drive, so cells outside the metal keep J = 0); Lorentz in velocity form, dp = P^{n+1} -
        # P^n from (dp, p, E^n), then p = P^{n+1}. While accumulating, the E^n half of each time-centred pole work is
        # added here (the E^{n+1} half at the end of advance, after every correction, injection and wall re-pin).
        drude_box = self._drude_box
        if drude_box is not None:
            if len(self.drude_poles) == 1:
                # One pole: its box is drude_box, so jz_total = J needs no zero fill.
                drude = self.drude_poles[0]
                _kernels.drude_j_update_store(drude['damping'], drude['drive'], ez[drude_box], drude['j'][k],
                                              jz_total[drude_box])
            else:
                jz_total[drude_box].fill(0)
                for drude in self.drude_poles:
                    box = drude['box']
                    _kernels.drude_j_update_total(drude['damping'], drude['drive'], ez[box], drude['j'][k],
                                                  jz_total[box])
            if accumulate:
                _kernels.q_accum_je_half(jz_total[drude_box], ez[drude_box], self._dt_real, self.q_accum[drude_box])
        for lorentz in self.lorentz_poles:
            box = lorentz['box']
            _kernels.lorentz_p_update(lorentz['damping'], lorentz['restoring'], lorentz['drive'], ez[box],
                                      lorentz['p'][k], lorentz['dp'][k])
            if accumulate:
                _kernels.q_accum_lorentz_half(lorentz['dp'][k], ez[box], self.q_accum[box])
        for gain in self.gain_media:   # four-level gain: a Lorentz pole driven by N1 - N2 (fdtd_gain)
            gain.update_polarization()

        boundary_mgr = field_set.boundary_mgr

        def e_block():   # Yee E + E CPML (+ the fused E_z rows): fixed arguments only (_fixed_block)
            _kernels.update_e_kernel(
                self._k_grid, self._k_block,
                (ez, hx, hy, self.ca, self.cb,
                 self._inv_dx_e_1d, self._inv_dy_e_1d,
                 self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32,
                 partner.hx, partner.hy, self._wrap_cos_x, field_set.wrap_sin_x, self._wrap_cos_y,
                 field_set.wrap_sin_y))
            self._cpml_post(field_set, 'e', hx, hy, ez, self.cb)
            if boundary_mgr.fused_walls:
                boundary_mgr.apply_ez_walls(ez)

        self._fixed_block(('e', k), (), e_block)
        if not boundary_mgr.fused_walls:
            boundary_mgr.apply_ez_walls(ez)   # apply_e_post's outer-face rows (row copies: eager)

        # Drude E correction E^{n+1} = E_pred - cb J^{n+1/2} (cb carries the semi-implicit sigma_e factor of the Yee
        # update, so conduction and a pole on one cell stay consistent).
        if drude_box is not None:
            _kernels.drude_e_correction(jz_total[drude_box], self.cb[drude_box], ez[drude_box])

        # Lorentz E correction (cb / dt) (P^{n+1} - P^n) (D = eps0 eps_inf E + P bookkeeping; cb carries the sigma_e
        # factor as for Drude); P^{n+1} - P^n is the dp buffer.
        for lorentz in self.lorentz_poles:
            box = lorentz['box']
            _kernels.lorentz_e_correction(lorentz['dp'][k], self.cb[box], self._inv_dt_real, ez[box])
        for gain in self.gain_media:
            gain.correct_field()

        # Kerr / TPA / free carriers last: implicit in E^{n+1} on top of every linear correction (fdtd_nonlinear);
        # then the gain populations see the final E^{n+1} of the update.
        if nonlinear is not None:
            nonlinear.correct(accumulate)
        for gain in self.gain_media:
            gain.update_populations()

    def _nonlinear_components(self):
        """The E component the C5 media (nonlinear, gain) act on: E_z at the cell centres, n0^2 = eps_r."""
        return [{'name': 'ez', 'axis': None, 'periodic': None, 'widths': None, 'field': self.fields.ez,
                 'cb': self.cb, 'eps': self.materials.eps_r, 'eps_scale': 1.0}]

    def _nonlinear_loss_target(self, name, box):
        """(q_accum box view, dt): the nonlinear work [J/m^3] goes straight into q_accum."""
        return self.q_accum[box], self._dt_real

    def _accumulate_e_new_half(self):
        """Loss with the final E^{n+1} (after injection, walls, PEC) [J/m^3], each on its box: Drude half, ohmic
        sigma_e E^2 dt, each Lorentz half (dp = P^{n+1} - P^n), so the time-centred work telescopes across steps (real
        field set only: accumulation refuses Bloch k != 0)."""
        ez = self.fields.ez
        drude_box = self._drude_box
        if drude_box is not None:
            _kernels.q_accum_je_half(self._jz_totals[0][drude_box], ez[drude_box], self._dt_real,
                                     self.q_accum[drude_box])
        ohmic_box = self._ohmic_box
        if ohmic_box is not None:
            _kernels.q_accum_ohmic(self.materials.sigma_e[ohmic_box], ez[ohmic_box], self._dt_real,
                                   self.q_accum[ohmic_box])
        for lorentz in self.lorentz_poles:
            box = lorentz['box']
            _kernels.q_accum_lorentz_half(lorentz['dp'][0], ez[box], self.q_accum[box])
        self.q_accum_time += float(self.dt)

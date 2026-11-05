"""FDTDEngine2D_Hz: the 2D Yee engine of H_z polarisation (H_z at cell centres, E_x / E_y on the faces) with
face-averaged eps / sigma, Drude / Lorentz poles and PEC masks."""
import numpy as np

from ....core.constants import DTYPE, EPS0
from ..yee_grid import face_average, face_pec, face_shares
from . import fdtd_hz_2d_kernels as _kernels
from .fdtd_2d_base import _YeeEngine2DBase, _bounding_box, _union_box
from .fdtd_conformal_kernels import conformal_hz, sparse_launch


class FDTDEngine2D_Hz(_YeeEngine2DBase):
    """2D Yee engine, H_z polarisation: H_z at cell centres, E_x / E_y on the hi-y / hi-x faces, H_z stepped before E.
    Face eps / sigma / pole weights are yee_grid.face_average of the cell arrays; subpixel_smoothing replaces eps (eps_x
    / eps_y, ca, cb) on the interface faces."""
    # Direct MaterialGrid writes must target eps_xx / eps_yy (eps_r is not read), followed by materials_changed.

    polarization = 'hz'
    _E_COEFFICIENT_MATERIALS = frozenset({'eps_xx', 'eps_yy', 'sigma_e'})
    subpixel_smoothing = None   # physics/em/subpixel_smoothing.SubpixelSmoothing (EMPhysics.set_subpixel_smoothing)
    _conformal_cells = None     # (cell, extra device arrays, launch) of set_conformal_cells, None = off

    def __init__(self, grid, fields, materials, boundary_manager):
        # Base: polarization check of the BoundaryManager, eps_xx / eps_yy registered (what E_x / E_y see), curl
        # spacings (*_e at cell centres for H_z, *_h at hi faces for E_x / E_y), launch scalars, da / db, clock and the
        # J.E window (q_accum allocated at the first enable).
        super().__init__(grid, fields, materials, boundary_manager)

        # Run-backend cell widths in the field dtype for the loss split (q_accum_hz); the host float64 ones weight the
        # face averages.
        xp = self.xp
        self._cell_dx_real = xp.asarray(self._cell_dx, dtype=self.dtype)
        self._cell_dy_real = xp.asarray(self._cell_dy, dtype=self.dtype)
        self._half_inv_dt_real = self._real(0.5 / float(self.dt))
        self._unit_scale_real = self._real(1.0)
        # face_average weights (field dtype) of the fused E-coefficient rebuild: *_y along y (E_x faces), *_x along x
        # (E_y).
        lower_share_y, upper_share_y = face_shares(self._cell_dy, self._periodic_y, self.dtype)
        lower_share_x, upper_share_x = face_shares(self._cell_dx, self._periodic_x, self.dtype)
        self._lower_share_y, self._upper_share_y = xp.asarray(lower_share_y), xp.asarray(upper_share_y)
        self._lower_share_x, self._upper_share_x = xp.asarray(lower_share_x), xp.asarray(upper_share_x)

        # ca / cb / absolute face eps [F/m] per E face set, rewritten in place by one fused kernel (cb also corrects E
        # for the Drude current, the Lorentz polarisation step and the CPML psi term; eps_x / eps_y are read only by
        # the C5 nonlinear / gain media, _nonlinear_components: the sheet calibration recomputes from eps_xx / eps_yy).
        shape = (self.nx, self.ny)
        self.ca_x = xp.empty(shape, dtype=self.dtype)
        self.cb_x = xp.empty(shape, dtype=self.dtype)
        self.eps_x = xp.empty(shape, dtype=self.dtype)
        self.ca_y = xp.empty(shape, dtype=self.dtype)
        self.cb_y = xp.empty(shape, dtype=self.dtype)
        self.eps_y = xp.empty(shape, dtype=self.dtype)
        # float32 face keep masks, also under double (0 on faces touching a PEC cell), None until a PEC region exists.
        # Every E-coefficient rebuild folds them into ca / cb (Yee update, CPML psi and pole corrections leave PEC faces
        # at 0); advance re-zeroes only _pec_box (bounding box of both components' PEC faces) after injection.
        self._keep_x = None
        self._keep_y = None
        self._pec_box = None
        # Set by eps_*_update (lazy rebuild in refresh_coefficients), cleared by every E-coefficient rebuild.
        self._e_coefficients_dirty = False
        self._rebuild_e_coefficients()
        self._rebuild_h_coefficients()

        # Dispersive poles: per E component, auxiliary arrays on the bounding box of its face weights; the Drude and
        # Lorentz damping / restoring / drive are scalars. Pole regions: one face list ({'component', 'box', 'weight'}, PEC keep
        # folded into weight) per distinct cell mask, shared by every pole on it (a Drude + 3 Lorentz metal stores 1).
        self.drude_poles = []
        self.lorentz_poles = []
        self._pole_regions = []
        # 'ex' / 'ey' -> union of that component's pole face boxes, the only part of the pole power totals ever written.
        self._pole_power_boxes = {}
        # Union cell mask of every pole (run-backend bool), allocated with the first pole; used only by the loss split.
        self._pole_cell = None

        # Coupling-injected H_z source (inject_source('hz_source', ...)), added every step.
        self._out_of_plane_source = None
        # len(sources) of the last source-protocol check (host-only, redone when the list length changes).
        self._checked_source_count = -1

        # _pole_power_x / _pole_power_y hold the step's time-centred pole power density [W/m^3] on the E_x / E_y faces
        # while accumulating with poles (full-grid for q_accum_hz, zeroed per step only on _pole_power_boxes).
        self._pole_power_x = None
        self._pole_power_y = None
        # 1-element stand-ins for the pole power totals / pole mask when q_accum_hz runs without poles.
        self._no_pole_power = None
        self._no_pole_cell = None
        # (grid, block, box ints) of q_accum_hz on the cells that can receive loss, (None,) for none; None = stale
        # (refreshed at the next accumulation after finalize, a sigma_e change or a new pole / nonlinear region)
        self._loss_box_args = None

    # --- PhysicsSolver contract -----------------------------------------------

    def state(self) -> dict:
        """Public fields: hz, eps_xx, eps_yy, sigma_e, q_accum (None before accumulation), q_accum_time; the staggered
        ex / ey are not state (the renderer gives them collocated)."""
        return {
            'hz': self.fields.hz,
            'eps_xx': self.materials.eps_xx,
            'eps_yy': self.materials.eps_yy,
            'sigma_e': self.materials.sigma_e,
            'q_accum': self.q_accum,
            'q_accum_time': self.q_accum_time,
        }

    def _permittivity_written(self) -> None:
        """eps_xx_update / eps_yy_update (inject_source): E face coefficients rebuilt at the next step."""
        self._e_coefficients_dirty = True

    def refresh_coefficients(self) -> None:
        """Rebuild the E face coefficients when an eps_*_update left them stale (no host sync)."""
        if self._e_coefficients_dirty:
            self._rebuild_e_coefficients()

    # --- loss accounting ------------------------------------------------------

    def _q_accum_enabled(self) -> None:
        """1-element no-pole stand-ins for q_accum_hz, then the pole power totals when a pole exists."""
        if self._no_pole_power is None:
            self._no_pole_power = self.xp.zeros(1, dtype=self.dtype)
            self._no_pole_cell = self.xp.zeros(1, dtype=np.bool_)
        self._ensure_pole_power_totals()

    def _ensure_pole_power_totals(self) -> None:
        """Allocate the full-grid face pole power totals (field dtype) once accumulation is on and a pole exists."""
        if self._enable_q_accum and self._pole_cell is not None and self._pole_power_x is None:
            self._pole_power_x = self.xp.zeros((self.nx, self.ny), dtype=self.dtype)
            self._pole_power_y = self.xp.zeros((self.nx, self.ny), dtype=self.dtype)
            for pole in self.drude_poles + self.lorentz_poles:
                for face in pole['faces']:
                    self._attach_face_views(face)

    def _attach_face_views(self, face) -> None:
        """Cache the box views 'e' (one per field set), 'cb', 'pole_power' (None before the totals exist) on a pole
        face: the field, coefficient and total arrays are only ever rewritten in place, so the views stay valid."""
        _, cb_face, pole_power, _ = self._component_arrays(face['component'])
        box = face['box']
        face['e'] = [getattr(field_set.fields, face['component'])[box] for field_set in self._field_sets]
        face['cb'] = cb_face[box]
        face['pole_power'] = None if pole_power is None else pole_power[box]

    # --- time stepping ------------------------------------------------------------

    def advance(self, sources=(), monitors=()) -> None:
        """One iteration: H_z update, magnetic sources at t, hz_source, H_z walls, E update, electric sources at
        t+dt/2 (each source at the centre time of its update), E walls, PEC, E^{n+1} loss half, then monitors at
        t+dt. Sources drive the real field set (Bloch line sources also the imaginary one); walls and PEC act on every
        set."""
        self.refresh_coefficients()
        if len(sources) != self._checked_source_count:
            self._check_source_protocol(sources)

        fields = self.fields
        t_start = float(self.time)
        t_half = t_start + 0.5 * float(self.dt)
        t_full = t_start + float(self.dt)

        self.update_h()
        for source in sources:
            if hasattr(source, 'inject_magnetic'):
                source.inject_magnetic(fields, t_start)
        if self._out_of_plane_source is not None:   # inject_source('hz_source', ...)
            fields.hz += self._out_of_plane_source
        # The outer-face BCs pin their rows / faces again after injection, so a current on a pec / pmc wall or a
        # mirror plane radiates nothing and a mirror image follows a source next to its plane.
        for field_set in self._field_sets:
            field_set.boundary_mgr.apply_hz_walls(field_set.fields.hz)

        self.update_e()
        for source in sources:
            if hasattr(source, 'inject_electric'):
                source.inject_electric(fields, t_half)
        self._inject_imaginary_sources(sources, t_half)
        # ca = cb = 0 already pins the PEC faces through the update; the keep catches injections and wall re-pins there.
        pec_box = self._pec_box
        for field_set in self._field_sets:
            set_fields = field_set.fields
            field_set.boundary_mgr.apply_exy_walls(set_fields.ex, set_fields.ey)
            if pec_box is not None:   # in place on the views (no setitem copy back)
                for field, keep in ((set_fields.ex, self._keep_x), (set_fields.ey, self._keep_y)):
                    pinned = field[pec_box]
                    self.xp.multiply(pinned, keep[pec_box], out=pinned)
        self._finish_step(t_full, monitors)

    def _check_source_protocol(self, sources) -> None:
        """TypeError for a source with neither inject_magnetic nor inject_electric (host-only)."""
        for source in sources:
            if not (hasattr(source, 'inject_magnetic') or hasattr(source, 'inject_electric')):
                raise TypeError(f"{type(source).__name__} does not implement the H_z source protocol "
                                f"inject_magnetic(fields, t) / inject_electric(fields, t)")
        self._checked_source_count = len(sources)

    def update_h(self):
        """Yee H_z update + CPML / BC post of every field set (each reads only E, so the order of the sets is free)."""
        for field_set in self._field_sets:
            fields, partner = field_set.fields, field_set.partner_fields
            _kernels.update_hz_kernel(
                self._k_grid, self._k_block,
                (fields.hz, fields.ex, fields.ey, self.da, self.db,
                 self._inv_dx_e_1d, self._inv_dy_e_1d,
                 self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32,
                 partner.ex, partner.ey, self._wrap_cos_x, field_set.wrap_sin_x, self._wrap_cos_y,
                 field_set.wrap_sin_y))
            if self._conformal_cells is not None:
                cell, cell_extra, (grid, block, count) = self._conformal_cells
                conformal_hz(grid, block, (fields.hz, fields.ex, fields.ey, self.db, self._inv_dx_e_1d,
                                           self._inv_dy_e_1d, cell, cell_extra, count, self._ny_i32))
            field_set.boundary_mgr.apply_hz_post(fields.ex, fields.ey, fields.hz, self.db)

    def _update_e_set(self, field_set):
        """update_e of one field set (the E kernel reads only H_z, so the sets may follow each other)."""
        fields, k = field_set.fields, field_set.index
        ex, ey, hz = fields.ex, fields.ey, fields.hz
        partner = field_set.partner_fields
        accumulate = self._enable_q_accum   # never with two field sets (enable_q_accum refuses Bloch k != 0)
        with_pole_power = accumulate and self._pole_power_x is not None
        if with_pole_power:
            # Outside the union pole boxes the totals are never written, so they stay 0 from allocation.
            for component, box in self._pole_power_boxes.items():
                self._component_arrays(component)[2][box].fill(0)
        nonlinear = self.nonlinear   # never with two field sets (add_nonlinear refuses Bloch k != 0)
        if nonlinear is not None:
            nonlinear.store_previous()

        # Both poles are stepped from the pre-update E^n (kernel notes in fdtd_hz_2d_kernels): Drude leap-frog
        # J^{n+1/2} = J^{n-1/2} + damping J^{n-1/2} + drive weight E^n (the region weight carries the PEC keep, so
        # faces outside the metal keep J = 0); Lorentz in velocity form, dp = P^{n+1} - P^n from (dp, p, E^n), then p = P^{n+1}. While
        # accumulating, the E^n half of each time-centred pole power is added here (the E^{n+1} half at the end of
        # advance, after every correction, injection and wall re-pin), fused into the pole update launch. Faces carry
        # cached box views 'e', 'cb', 'pole_power' (_attach_face_views).
        for drude in self.drude_poles:
            for face in drude['faces']:
                if with_pole_power:
                    _kernels.drude_j_update_power(drude['damping'], drude['drive'], face['weight'], face['e'][k],
                                                  face['j'][k], face['pole_power'])
                else:
                    _kernels.drude_j_update(drude['damping'], drude['drive'], face['weight'], face['e'][k],
                                            face['j'][k])
        for lorentz in self.lorentz_poles:
            for face in lorentz['faces']:
                if with_pole_power:
                    _kernels.lorentz_p_update_power(lorentz['damping'], lorentz['restoring'], lorentz['drive'],
                                                    face['weight'], self._half_inv_dt_real, face['e'][k], face['p'][k],
                                                    face['dp'][k], face['pole_power'])
                else:
                    _kernels.lorentz_p_update(lorentz['damping'], lorentz['restoring'], lorentz['drive'],
                                              face['weight'], face['e'][k], face['p'][k], face['dp'][k])
        for gain in self.gain_media:   # four-level gain: a Lorentz pole driven by N1 - N2 per face set (fdtd_gain)
            gain.update_polarization()

        _kernels.update_exy_kernel(
            self._k_grid, self._k_block,
            (ex, ey, hz, self.ca_x, self.cb_x, self.ca_y, self.cb_y,
             self._inv_dx_h_1d, self._inv_dy_h_1d,
             self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32,
             partner.hz, self._wrap_cos_x, field_set.wrap_sin_x, self._wrap_cos_y, field_set.wrap_sin_y))
        field_set.boundary_mgr.apply_exy_post(hz, ex, ey, self.cb_x, self.cb_y)

        # Drude E correction E^{n+1} = E_pred - cb J^{n+1/2} on each face box (cb carries the semi-implicit conduction
        # factor of the Yee update, so sigma_e and a Drude pole on one face stay consistent).
        for drude in self.drude_poles:
            for face in drude['faces']:
                _kernels.drude_e_correction(face['j'][k], face['cb'], face['e'][k])

        # Lorentz E correction (cb / dt) (P^{n+1} - P^n) (D = eps0 eps_inf E + P bookkeeping; cb carries the sigma_e
        # factor as for Drude); P^{n+1} - P^n is the dp buffer.
        for lorentz in self.lorentz_poles:
            for face in lorentz['faces']:
                _kernels.lorentz_e_correction(face['dp'][k], face['cb'], self._inv_dt_real, face['e'][k])
        for gain in self.gain_media:
            gain.correct_field()

        # Kerr / TPA / free carriers last, per E face set: implicit in E^{n+1} on top of every linear correction; the
        # TPA / FCA power joins the face pole power totals (fdtd_nonlinear). Then the gain populations see E^{n+1}.
        if nonlinear is not None:
            nonlinear.correct(with_pole_power)
        for gain in self.gain_media:
            gain.update_populations()

    def _nonlinear_added(self, mask) -> None:
        """add_nonlinear acts per E_x / E_y face set (face_average of the cell coefficients): its cells join the pole
        cells of the loss split and its face boxes the pole power boxes."""
        for component in self.nonlinear.components:
            name = component['name']
            self._pole_power_boxes[name] = _union_box(self._pole_power_boxes.get(name), component['box'])
        self._add_pole_cells(mask)

    def _nonlinear_components(self):
        """The E face sets the C5 media (nonlinear, gain) act on: E_x (hi-y faces, face_average along y) and E_y (hi-x
        faces, along x); n0^2 = the absolute face permittivity eps_x / eps_y over eps0."""
        return [{'name': 'ex', 'axis': 1, 'periodic': self._periodic_y, 'widths': self._cell_dy, 'field': self.fields.ex,
                 'cb': self.cb_x, 'eps': self.eps_x, 'eps_scale': 1.0 / EPS0},
                {'name': 'ey', 'axis': 0, 'periodic': self._periodic_x, 'widths': self._cell_dx, 'field': self.fields.ey,
                 'cb': self.cb_y, 'eps': self.eps_y, 'eps_scale': 1.0 / EPS0}]

    def _nonlinear_loss_target(self, name, box):
        """(face pole power total box view, 1): the nonlinear power [W/m^3] is split to the cells by q_accum_hz; None
        while the totals do not exist."""
        totals = self._component_arrays(name)[2]
        return None if totals is None else (totals[box], self._unit_scale_real)

    def _accumulate_e_new_half(self):
        """Loss with the final E^{n+1} (after injection, walls, PEC): the E^{n+1} pole power halves, then q_accum_hz
        (ohmic + pole power per cell), so the time-centred work telescopes across steps (real field set only:
        accumulation refuses Bloch k != 0)."""
        fields = self.fields
        ex, ey = fields.ex, fields.ey
        with_pole_power = self._pole_power_x is not None
        if with_pole_power:
            # Several poles may share a face: the E^{n+1} halves of the Drude power and of the Lorentz power
            # (dp = P^{n+1} - P^n).
            for drude in self.drude_poles:
                for face in drude['faces']:
                    _kernels.q_accum_je_half(face['j'][0], face['e'][0], self._unit_scale_real, face['pole_power'])
            for lorentz in self.lorentz_poles:
                for face in lorentz['faces']:
                    _kernels.q_accum_je_half(face['dp'][0], face['e'][0], self._inv_dt_real, face['pole_power'])

        if self._loss_box_args is None:
            self._loss_box_args = self._loss_box_launch()
        grid, block, box_ints = self._loss_box_args
        if grid is not None:
            self._launch_q_accum_hz(grid, block, box_ints, ex, ey, with_pole_power)
        self.q_accum_time += float(self.dt)

    def _launch_q_accum_hz(self, grid, block, box_ints, ex, ey, with_pole_power):
        """q_accum_hz (ohmic + pole power per cell) on the loss box."""
        _kernels.q_accum_hz(
            grid, block,
            (self.q_accum, ex, ey,
             self._pole_power_x if with_pole_power else self._no_pole_power,
             self._pole_power_y if with_pole_power else self._no_pole_power,
             self.materials.sigma_e,
             self._pole_cell if with_pole_power else self._no_pole_cell,
             self._cell_dx_real, self._cell_dy_real,
             self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32,
             np.int32(with_pole_power), self._dt_real, *box_ints))

    def _loss_box_launch(self):
        """(grid, block, int32 (i0, j0, box_nx, box_ny)) of the cells q_accum_hz can give loss: the sigma_e > 0 cells
        and the pole power face boxes grown one cell along the face normal (the cell past a face takes a share; a
        periodic last face wraps: the whole axis); (None, None, None) when there are none. One host sync."""
        box = _bounding_box(self.materials.sigma_e)
        for component, face_box in self._pole_power_boxes.items():
            axis = 1 if component == 'ex' else 0
            n_cells = self.ny if axis == 1 else self.nx
            periodic = self._periodic_y if axis == 1 else self._periodic_x
            cells = face_box[axis]
            grown = list(face_box)
            grown[axis] = (slice(0, n_cells) if periodic and cells.stop >= n_cells
                           else slice(cells.start, min(cells.stop + 1, n_cells)))
            box = _union_box(box, tuple(grown))
        if box is None:
            return None, None, None
        box_nx, box_ny = box[0].stop - box[0].start, box[1].stop - box[1].start
        grid, block = _kernels.launch(box_nx, box_ny)
        return grid, block, tuple(np.int32(value) for value in (box[0].start, box[1].start, box_nx, box_ny))

    def finalize(self) -> None:
        super().finalize()
        self._loss_box_args = None

    def materials_changed(self, names) -> None:
        super().materials_changed(names)
        if 'sigma_e' in names:
            self._loss_box_args = None

    def _component_arrays(self, component):
        """(E field, face cb, face pole power total or None, keep mask or None) of 'ex' / 'ey'."""
        if component == 'ex':
            return self.fields.ex, self.cb_x, self._pole_power_x, self._keep_x
        return self.fields.ey, self.cb_y, self._pole_power_y, self._keep_y

    # --- coefficients -------------------------------------------------------------

    def _rebuild_e_coefficients(self):
        """ca / cb / eps of both E face sets from the face-averaged eps_xx / eps_yy / sigma_e (one fused launch)."""
        materials = self.materials
        _kernels.e_face_coefficients(
            self._k_grid, self._k_block,
            (self.ca_x, self.cb_x, self.eps_x, self.ca_y, self.cb_y, self.eps_y,
             materials.eps_xx, materials.eps_yy, materials.sigma_e,
             self._lower_share_y, self._upper_share_y, self._lower_share_x, self._upper_share_x,
             self._nx_i32, self._ny_i32, self._periodic_x_i32, self._periodic_y_i32,
             self._dt_real, self._real(EPS0)))
        if self.subpixel_smoothing is not None:   # averaged eps on the interface faces, before the PEC fold
            self.subpixel_smoothing.correct_fdtd(self)
        self._fold_pec_keep()
        self._e_coefficients_dirty = False

    def _fold_pec_keep(self):
        """ca = cb = 0 on PEC faces (on _pec_box; eps_x / eps_y stay the material values for the nonlinear media)."""
        pec_box = self._pec_box
        if pec_box is None:
            return
        for keep, ca, cb in ((self._keep_x, self.ca_x, self.cb_x), (self._keep_y, self.ca_y, self.cb_y)):
            ca[pec_box] *= keep[pec_box]
            cb[pec_box] *= keep[pec_box]

    # --- EM-specific API ------------------------------------------------------

    def add_drude(self, drude_pole, mask=None, set_background=True, weight=None):
        """Drude pole on the run-backend bool (nx, ny) cell `mask` (None = all cells); set_background writes eps_xx = eps_yy =
        eps_inf under the mask; weight = host (E_x, E_y) float (nx, ny) face weights instead of the mask's face_average
        (conformal_poles)."""
        cell_mask = self._pole_cell_mask(mask)
        damping, drive = drude_pole.ade_coefficients(float(self.dt))
        faces = [{**region_face, 'j': self._set_buffers(region_face['weight'].shape)}
                 for region_face in self._pole_region_faces(cell_mask, weight)]
        for face in faces:
            self._attach_face_views(face)
        self.drude_poles.append({'pole': drude_pole, 'damping': self._real(damping), 'drive': self._real(drive),
                                 'faces': faces})
        self._add_pole_cells(cell_mask)

        if set_background:
            self.materials.assign('eps_xx', cell_mask, drude_pole.eps_inf)
            self.materials.assign('eps_yy', cell_mask, drude_pole.eps_inf)
            self._rebuild_e_coefficients()

    def add_lorentz(self, lorentz_pole, mask=None, weight=None):
        """Lorentz pole on the run-backend bool (nx, ny) cell `mask` (None = all cells); eps_xx / eps_yy act as eps_inf;
        weight = host (E_x, E_y) face weights as add_drude."""
        cell_mask = self._pole_cell_mask(mask)
        damping, restoring, drive = lorentz_pole.ade_coefficients(float(self.dt))
        faces = [{**region_face, 'p': self._set_buffers(region_face['weight'].shape),
                  'dp': self._set_buffers(region_face['weight'].shape)}
                 for region_face in self._pole_region_faces(cell_mask, weight)]
        for face in faces:
            self._attach_face_views(face)
        real = self._real
        self.lorentz_poles.append({'pole': lorentz_pole, 'damping': real(damping), 'restoring': real(restoring),
                                   'drive': real(drive), 'faces': faces})
        self._add_pole_cells(cell_mask)

    def add_pec_mask(self, mask, keep=None):
        """Perfect electric conductor on the run-backend bool (nx, ny) cell `mask`: every E face touching it is pinned to 0;
        keep = (keep_x, keep_y) float32 face masks instead (conformal PEC: only the faces wholly inside the metal)."""
        self._check_cell_mask(mask, 'PEC mask')
        xp = self.xp
        if keep is None:
            keep_x = (~face_pec(mask, 1, self._periodic_y, xp=xp)).astype(DTYPE)
            keep_y = (~face_pec(mask, 0, self._periodic_x, xp=xp)).astype(DTYPE)
        else:
            keep_x, keep_y = (xp.asarray(array, dtype=DTYPE) for array in keep)
        if self._keep_x is None:
            self._keep_x, self._keep_y = keep_x, keep_y
        else:
            self._keep_x *= keep_x
            self._keep_y *= keep_y
        new_pec_box = _union_box(_bounding_box(1.0 - keep_x), _bounding_box(1.0 - keep_y))
        self._pec_box = _union_box(self._pec_box, new_pec_box)
        self._fold_pec_keep()
        for field_set in self._field_sets:
            field_set.fields.ex *= keep_x
            field_set.fields.ey *= keep_y
        face_keep = {'ex': keep_x, 'ey': keep_y}
        # Shared region weights once per region, pole state once per pole and field set.
        for region_faces in self._pole_regions:
            for region_face in region_faces:
                region_face['weight'] *= face_keep[region_face['component']][region_face['box']]
        for drude in self.drude_poles:
            for face in drude['faces']:
                for current in face['j']:
                    current *= face_keep[face['component']][face['box']]
        for lorentz in self.lorentz_poles:
            for face in lorentz['faces']:
                keep_box = face_keep[face['component']][face['box']]
                for polarization_density, polarization_step in zip(face['p'], face['dp']):
                    polarization_density *= keep_box
                    polarization_step *= keep_box

    def set_conformal_cells(self, cell, cell_extra):
        """Conformal PEC (physics/em/conformal_pec): each listed H_z cell (host int32 flat, off the grid edge) adds db x
        the curl of cell_extra (float32 (n, 4): weight - 1 of E_x hi / lo, E_y hi / lo) every step; empty = off."""
        if len(cell) == 0:
            self._conformal_cells = None
            return
        xp = self.xp
        self._conformal_cells = (xp.asarray(cell, dtype=np.int32),
                                 xp.ascontiguousarray(xp.asarray(cell_extra, dtype=self.dtype)),
                                 sparse_launch(len(cell)))

    def _set_buffers(self, shape):
        """One zeroed pole buffer of `shape` in the field dtype per field set."""
        return [self.xp.zeros(shape, dtype=self.dtype) for _ in self._field_sets]

    def _pole_region_faces(self, cell_mask, face_weights=None):
        """Region face list of `cell_mask` (or of the host face_weights): an existing region with equal boxes and
        weights is reused (host sync: build time only), else a new one is registered and its boxes join
        _pole_power_boxes."""
        faces = self._pole_faces(cell_mask, face_weights)
        for region_faces in self._pole_regions:
            if len(region_faces) == len(faces) and all(
                    old['component'] == new['component'] and old['box'] == new['box']
                    and bool(self.xp.array_equal(old['weight'], new['weight']))
                    for old, new in zip(region_faces, faces)):
                return region_faces
        self._pole_regions.append(faces)
        for face in faces:
            self._pole_power_boxes[face['component']] = _union_box(self._pole_power_boxes.get(face['component']),
                                                                   face['box'])
        return faces

    def _pole_faces(self, cell_mask, face_weights=None):
        """{'component', 'box', 'weight': face weight on the box times the PEC keep} for 'ex' / 'ey' faces with any
        weight > 0; weights = face_average of the cell mask (0, the dual-cell share of the pole cell, or 1) in the field
        dtype, or the host face_weights (E_x, E_y) of conformal_poles."""
        cell_weight = cell_mask.astype(self.dtype) if face_weights is None else None
        faces = []
        for index, (component, axis, periodic, widths) in enumerate((('ex', 1, self._periodic_y, self._cell_dy),
                                                                     ('ey', 0, self._periodic_x, self._cell_dx))):
            if face_weights is None:
                weight = face_average(cell_weight, axis, periodic, widths, xp=self.xp)
            else:
                weight = np.asarray(face_weights[index], dtype=np.float64)
                self._check_weight_shape(weight, f'{component} pole face weight')
            box = _bounding_box(weight)
            if box is None:
                continue
            weight_box = (weight[box].copy() if face_weights is None   # copy: a view keeps the full-grid temporary
                          else self.xp.asarray(weight[box], dtype=self.dtype))
            keep = self._component_arrays(component)[3]
            if keep is not None:
                weight_box *= keep[box]
            faces.append({'component': component, 'box': box, 'weight': weight_box})
        return faces

    def _add_pole_cells(self, cell_mask):
        """OR the pole's cells into the union mask (allocated with the first pole), then allocate pole power totals."""
        if self._pole_cell is None:
            self._pole_cell = cell_mask.copy()
        else:
            self._pole_cell |= cell_mask
        self._loss_box_args = None   # the pole power boxes grew
        self._ensure_pole_power_totals()

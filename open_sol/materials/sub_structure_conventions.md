# open_sol/materials — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Library material data.

- `description` never repeats values held in the property dicts / poles (describe prints those): only provenance,
  validity range and caveats, so a data fix needs no prose edit.
- Library `optical` dicts hold 1.55 um values: a model at another wavelength overrides eps_r / sigma_e (e.g. water in
  fd_joule_heating at 530 nm). Library Drude poles are optical (near-IR) fits, never far-IR / DC ones (those
  underestimate the near-IR loss 2-3x; sigma_dc stays separate). Aluminum's pole is a 1.55 um-only fit to Rakic 1998
  while its nk table is Rakic 1995 (two data sets); silver's pole has no recorded source. `lithium_niobate_z_cut`:
  eps_r = n_o^2, eps_zz = n_e^2 (an eps_r override does not change the E_z types), no Pockels r (its in-plane terms are
  not diagonal); `lithium_niobate_x_cut` (optic axis along x): eps_r = n_o^2, eps_xx = n_e^2, r (r13) + r_xx (r33) for
  field_component 'x', clamped r^S like its clamped eps_dc and the crystal entry (unclamped r^T would be 10 / 32.2
  pm/V). Complex permittivities in descriptions use the exp(+j omega t) sign (lossy: Im < 0).
- Validation code reading the selected materials' merged data goes through `definition_helpers.selected_material_data`
  / `selected_material_sets` (the builder's merge; unresolved materials skipped or flagged), never its own loop. `water` / `heavy_water` carry absorption as sigma_e, alpha = 0.
- `model_properties()` merges `electrical` (sigma_dc / eps_dc: gold, silver, aluminum, water conduct in any
  electric_currents physics), `fluid` (viscosity: water, heavy_water; air has none: set it to make air flow) and
  `mechanical` (isotropic youngs_modulus / poisson_ratio: silica, silicon (Hill average), gold, silver, aluminum, both
  lithium niobates (rough estimate); liquids and air have none = void in solid_mechanics) like `optical` / `thermal`.
  Properties of absent types are ignored by the builder (validation warns for user-set ones, never library-merged
  ones; permittivity names are left to the EM `_material_issues` hint).
- `Material.nk_table` (`NkTable`, rows in `nk_data.py` from refractiveindex.info copies: silver / gold Johnson & Christy
  1972, aluminum Rakic 1995, silicon Green 2008) feeds only an `{'model': 'nk_fit'}` dispersion entry, expanded by
  `effective_material_data` (`fitted_library_poles` caches per (name, n_lorentz, range); values must be plain numbers):
  fitted poles first, the library `optical` dict dropped, eps_inf = the Drude eps_inf or, without a Drude pole, eps_r.
  The library `em` poles stay the default; a table edit changes fits, so regenerate api_reference.md only if
  descriptions change.
- Crystal entry (`crystal_tensors.py`, numpy): `{'model': 'crystal', 'crystal': <CRYSTALS name>, 'euler_deg': [phi,
  theta, psi] (plain numbers), 'plane': 'sagittal' | 'full'}` in MaterialSpec.dispersion, one per material, removed by
  `effective_material_data`; precedence library < crystal < user properties (also under nk_fit). Sagittal (2D) expands
  to c11 / c12 / c22 / c66 (+ c16 / c26 only when non-zero: they allocate the elastic_waves monoclinic arrays), c13 /
  c23 / c33 (+ c36 when non-zero), piezo_e13 / _e23 (only 2D solid_mechanics reads them), piezo_e11 .. _e26, eps_dc_xx
  / _yy / _xy (clamped eps^S) and rho in model axes. Euler angles: SAW convention, passive a = Rz(psi) Rx(theta)
  Rz(phi) (rows X1' propagation, X3' surface normal; YZ = (0, 90, 90) gives X3' = -Y); sagittal model (x, y, z) = (X1',
  X3', -X2'). Data in crystal axes per IEEE 1949 (3m mirror normal to X; 32 X the 2-fold axis); c14 / e22 signs depend on
  the +Y choice, only c14 e22 is invariant. Rotated entries below 1e-12 of their tensor's largest are 0 (a round-off
  eps_dc_xy would demand a uniform mesh). `sagittal_crystal` caches per (crystal, angles) and reports c16 / c26 and the
  anti-plane coupling ratios. `materials/__init__` detects crystal entries itself (`_is_crystal_entry`,
  `crystal_entries`) and imports crystal_tensors only to expand one, so a crystal_tensors edit reruns only crystal-model
  selftests (root "Selftest reuse and shared code"). Values were transcribed without the sources at hand: verify before
  trusting digits. GUI: the material form's 'Crystal data' group.
- Crystal entry, 3D: `'plane': 'full'` (`full_crystal`, cached; `crystal_properties` dispatches) expands in model (x,
  y, z) = (X1', X2', X3') to rho, all 21 c_IJ (`FULL_STIFFNESS_PROPERTIES`), the 18 piezo_e<i><J> and eps_dc_xx / _yy /
  _zz / _yz / _xz / _xy (no optics). The plane is validated (`_check_crystal_entries`): a 3D model needs 'full' written
  out, a 2D model 'sagittal' (the default when absent); each expander refuses the other plane, so 2D-only callers fail
  loudly. solid_mechanics_3d declares all of them (piezo / eps read only with piezoelectric on tetrahedra); undeclared
  names are skipped without warning. `mirror_violation(values, axis)`: the largest 3D entry flipping sign under axis ->
  -axis (above MIRROR_RELATIVE 1e-9), the solid_mechanics_3d Roller warning.
- Crystal optics (`CrystalData.photoelastic` p, `.pockels` clamped r^S [m/V], `.optical_reference`; None = no data: AlN
  none, quartz no r): Nye convention, Delta(1/n^2)_I = p_IJ S_J + r_Ik E_k, so p_IJ = p_ijkl, r_Ik = r_ijk (no factors),
  p not symmetric; rotations p' = M p M^T, r' = M r R^T (`rotate_optical`). The sagittal entry adds photoelastic_p11 /
  _p12 (row xx) always, photoelastic_p16 .. _p66 (rows xx, yy, zz, xy = Voigt 6) only when one differs from its isotropic
  fallback (p66: (p11 - p12) / 2), pockels_r11 .. _r62 only when one is non-zero; `antiplane_*_ratio` report the always
  dropped yz / xz rows, `shear_*_ratio` the xy row (dropped by in-plane EM types without eps_xy; the couplings warn).
  Every optical value is from memory, marked 'unverified, check against <source>'; p14 / p41 / r22 signs follow the +Y
  setting. The optical index is not part of the entry (eps_r / eps_xx / eps_yy stay material values).
- `gst_225_amorphous` / `gst_225_crystalline` (gst.py): eps_r / sigma_e computed from `physics/em/phase_change.GST_225`
  n, k (one source with the phase_change_medium defaults: edit the numbers there); thermal data per phase in gst.py.
  Static phases only: switching is the phase_change_medium feature.

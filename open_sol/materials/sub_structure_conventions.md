# open_sol/materials — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Library material data.

- `description` holds provenance, validity range and caveats, never values held in the property dicts / poles
  (describe prints those), so a data fix needs no prose edit. Complex permittivities in descriptions use the
  exp(+j omega t) sign (lossy: Im < 0). Values marked unverified there (or in `crystal_tensors.CRYSTALS`) are checked
  against the cited source before anyone relies on their digits.
- Library `optical` dicts hold 1.55 um values: a model at another wavelength overrides eps_r / sigma_e (or uses an
  `nk_fit` entry where the material has an `nk_table`). Library Drude poles are optical (near-IR) fits, never far-IR /
  DC ones (those underestimate the near-IR loss 2-3x; sigma_dc stays separate).
- The one exception: an entry named for its wavelength (`water_532nm`) holds its optical data at that wavelength. Data
  for another wavelength or regime (GHz moduli of water) goes into such a new entry, never into the plain one
  (`water`): existing models and examples read the plain entry's values.
- A library material carries data for every physics in one merged dict (`model_properties`; crystal and nk_fit
  entries expanded by `effective_material_data`, precedence library < crystal < user properties): the builder ignores
  names no physics of the model reads, and validation warns only for user-set ones, never library-merged or
  crystal-expanded ones. Validation code reading the selected materials' merged data goes through
  `definition_helpers.selected_material_data` / `selected_material_sets` (the builder's merge, unresolved materials
  skipped), never its own `effective_material_data` loop.
- Crystal entries: conventions (Voigt, Nye optics, SAW Euler angles, model axes per plane) in the `crystal_tensors`
  header comment; a 3D model needs `'plane': 'full'` written out, 2D defaults to 'sagittal' (validation
  `_check_crystal_entries`). `materials/__init__` detects crystal entries itself and imports crystal_tensors only to
  expand one, so a crystal_tensors edit reruns only crystal-model selftests (root "Selftest reuse and shared code").

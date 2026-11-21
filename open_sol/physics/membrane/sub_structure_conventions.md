# open_sol/physics/membrane — conventions & gotchas

Sub file of the root `structure_conventions.md`. The `membrane` type: tensioned film or Kirchhoff plate (equation,
grids, studies, feature behaviour: the type's description and help texts in `definition.py`; numerics: the module
headers of solver_common / solver_fem / solver_line / bending_fem / ports / plate_ports / squeeze_film). Coupling:
coupling/definitions_membrane.py (solid_membrane_tension -> the coupled_tension_* slots).

## Layout
- Solvers are host numpy (scipy in fem), the state arrays on the Domain's xp. The package imports no other physics
  package.

## Model
- Plate switch = PRESENCE of the bending_stiffness array (any material sets it, 0 included), tested in three places
  that must agree: the solver (`MembraneBase.prepare`), validation (`_materials_setting`) and the estimates (`_plate`,
  where a literal 0 counts as unset).
- Real stays real by dtype, never by a flag: lossless coefficients are float64 (`MembraneCoefficients.lossy`: any loss
  != 0), so fem.harmonic factors a real pencil and harmonic_resonances runs eigsh. The estimates mirror it from the
  settings (`_complex_pencil` eigen, `_response_term_bytes` / `_complex_response` frequency_response: loss, PML, ports;
  an expression loss counts as complex: an overestimate only); a new complex term updates both.
- System lifetime: a new solver method adding a condition / load / port (as clamp_face, add_pressure) calls
  `_discard_system` (the next solve rebuilds through `_ready_system`); a coefficient change (materials_changed,
  inject_source, restore_solution) only sets `_coefficients_dirty`, so the next solve re-applies the coefficients on
  the kept system.

## Self-tests
- Case notes: the comment above each case in selftests*.py.
- Not built: error indicator, scalar outputs, mode_projection on plate modes, eigenfrequency with ports, line forces
  on line objects, a port with coupled (non-uniform) tension.

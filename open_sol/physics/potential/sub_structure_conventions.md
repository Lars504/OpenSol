# open_sol/physics/potential — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Potential gotchas; DiffusionSystem (assembly, PCG,
held cells, anchors, metric, cut cells, azimuthal order, phi wrap) is in `open_sol/core/sub_structure_conventions.md`
("DiffusionSystem", "Coordinate metric"), the nonlinear-electrolyte Newton loop in its "Damped iteration", the FEM
systems in the fem sub file ("Scalar diffusion", "Scalar diffusion on tetrahedra"). What each feature, setting and
validation rule does: their help texts and Issue messages in `definition.py`; held fluxes, driven terminals,
electrolytes, snapshots and sweep reuse: the docstrings of `solver.py`, `solver_fem.py` and `solver_fem3d.py`.

## Solvers
- Three solvers share the `solver.KINDS` vocabulary (KindSpec: arrays, slot, field and flux names, units), the feature
  API, held regions and scalar outputs: `PotentialPhysics` (tensor meshes, `solver.py`), `FEMPotentialPhysics`
  (triangles) and `FEMPotentialPhysics3D` (tetrahedra), both on `solver_fem.FEMPotentialMixin`. Apply callbacks branch
  on `ctx.domain.grid_kind`. Rules both mesh kinds follow are module functions of `solver.py` (`_face_held_by`,
  `held_face_bc`, `ionic_medium`, `electrolyte_scalars`, `faces_touch_mirror`, `HELD_SCALAR_PREFIXES`): call them,
  never a copy.
- The material vocabulary always carries the `_dc` infix (sigma_dc, eps_dc), never the optical sigma_e / eps_r; the
  electrostatics charge field is `rho_free` (`rho` is heat's density, which would force qualified names). Heat ->
  currents feedback is a state-dependent `sigma_dc` expression of `heat_T` (a DynamicMaterialUpdate), not a coupling.
- Outputs are edge-extended into the PML ring like heat's T, so a field-dependent 'all' material or an eps
  field_transfer sees no step at the CPML.
- Probes sample only after a re-solve or at dt = inf (`step`): sampling every finite-dt tick would store one copy per
  EM step.
- Help texts are shared by both kinds and formatted per kind (`_KIND_NAMES`, `_COMMON_DESCRIPTION`, `_XY_HELP_TEXTS`,
  ...): an edit must read right for both (electrostatics has no q_joule and every cell conducts).

## Element meshes
- Tetrahedra: a voltage on an object that touches the window also holds the object's faces on the window planes, so
  draw conductors that meet the window as such (a coax outer conductor as 'window minus cylinder').
- Direct back ends of the 'linear' role: `model.definition_helpers.CG_DIFFUSION_LINEAR_ROLE` /
  `DIFFUSION_DIRECT_BACK_ENDS` (shared with heat and magnetic). Element meshes re-factor only on material, BC-kind or
  held changes (fem sub file "Scalar diffusion"), so a coupled sigma(T) loop is where the cuDSS plan / analysis reuse
  pays.

## Self-tests and open items
- Cases: `selftests*.py` (one module per area; each description states what it pins); uniform-mesh and
  axisymmetric / polar checks are the examples in `examples/potential/` (closed forms as their analytic references).
- Open: on triangles nonlinear electrolytes (a Newton loop on a per-quadrature-point Boltzmann mass: `DampedIteration`
  with `assemble_mass` of the tangent), driven terminals (would fit cheaply: one factor, each basis a lift) and
  azimuthal_order != 0 (validation refuses all three); a Stern layer on terminals / voltage object perimeters (needs
  per-face conductance factors on held edges). Not checked on tetrahedra: couplings (currents_heat_joule to heat on
  tetrahedra, Pockels / piezo) and the gpu backend.

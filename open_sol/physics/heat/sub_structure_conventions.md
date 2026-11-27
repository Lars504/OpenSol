# open_sol/physics/heat — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Heat gotchas; DiffusionSystem (assembly, PCG, held
cells, anchors, metric, cut cells, azimuthal order, phi wrap) is in `open_sol/core/sub_structure_conventions.md`
("DiffusionSystem", "Coordinate metric"), the FEM systems in the fem sub file ("Scalar diffusion", "Scalar diffusion
on tetrahedra"). What each feature, setting and validation rule does: their help texts and Issue messages in
`definition.py`.

## Solvers
- Three solvers share the field names, the 'volumetric_power' slot and the feature API: `HeatPhysics` (tensor meshes,
  `solver.py`), `FEMHeatPhysics` (triangles) and `FEMHeatPhysics3D` (tetrahedra), both on `solver_fem.FEMHeatMixin`
  (a change to steps, state, sources, radiation or T0 handling goes there, for both). Rules all three follow (time
  settings, radiation check / linearisation / warning, anchor texts) are module functions and constants of
  `solver.py`. Apply callbacks branch on `ctx.domain.grid_kind`.
- Values of t (temperature, heat_flux, heat_source Q, point_heat_source P) are taken at the step start t_n while the
  implicit step solves for t_n + dt: a first-order lag, the convention of every value of t.
- Direct back ends of the 'linear' role (`CG_DIFFUSION_LINEAR_ROLE`, shared with potential and magnetic): cuDSS pays
  for long fixed-dt transients above ~1e5 dofs and coupled k(T) re-factors; a small 2D steady solve pays the cuDSS plan
  (and nvmath's first-use start) where host SuperLU takes ms (back_end 'scipy_splu' keeps it on the host).
- Unbounded quasi-static heat = outer grading (`MeshSpec.outer_dx` / `outer_ratio`: a held far face ~1e3 x the
  structure away on a few dozen cells); arithmetic-metric FV needs bounded neighbour ratios, so there is no rational
  infinite-element layer.
- Open: azimuthal_order != 0 and the thermal_noise study on triangles (refused by validation; would need an m^2 / r
  mass term with held axis dofs, resp. the reduced FEM operator, consistent capacity mass and em_modes weightings on
  triangles); multigrid pcg on P2 (its aggregation counts negative off-diagonals only, P2 stiffness has positive
  ones); 'cudss_mixed' unmeasured (kappa ~ (L / h)^2 x the k contrast; tetrahedra: also 'cudss' on sliver-rich gmsh
  meshes); a 3D heat_source of x, y allocates one window-sized float32 temporary (`DiffusionSystem.add_masked`).

## Thermal noise (`thermal_noise.py`, features thermorefractive_noise / particle_noise, study thermal_noise)
- Method, operators, weightings and mirror / ghost handling: module docstring and the `_HarmonicDiffusionOperator`,
  `mode_pair_weighting` and `wall_ghost_cells` docstrings; heat's part in the study: the study description
  (`model/study_thermal_noise.py`); time / memory estimates: `definition._noise_time_estimate` /
  `_line_block_noise_estimate`.
- A 'line_block' shifted solver in mixed precision rejects very ill-conditioned shifts (low f on an insulated region:
  the static null mode); rejected shifts go through `spectral_densities` on a private SparseDirectSolver and are
  counted as rejected_shifts.
- The elastic counterpart is the solid feature brillouin_scattering (mechanics sub file "Spontaneous Brillouin"): the
  same FDT route (the readout weight as the drive, one solve per frequency), but its psd is per sideband (two-sided;
  2 psd = the one-sided spectrum), thermal_noise's one-sided. They run in different studies on different physics and
  nothing combines them: no model gives a Rayleigh-Brillouin spectrum (the central entropy peak and the doublet
  together).

## Self-tests
- Heat cases are discrete FV identities: source-free 1D conduction is exact on any widths, a uniformly heated slab or
  parabola only on uniform widths (graded cases keep Q = 0, or polar Q = q0 / r_c^2), a cell-centred cosine is an
  eigenvector of the insulated operator. Constant-valued faces cannot impose a sloped boundary, so k_xy tests write a
  linear T into `system.unknown2d` after the solve, or check that a uniform box stays uniform.
- Not covered by self-tests: illumination weightings, particle_noise, values of t on triangles.

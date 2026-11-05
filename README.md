# open_sol

GPU (cupy) and CPU 1D / 2D / 3D multiphysics: FDTD and FDFD electromagnetics (E_z and H_z polarisations, body of
revolution, conical incidence, 3D FDTD and the 3D FDFD eigensolver, nonlinear frequency domain), eigenmode expansion,
waveguide cross-section modes, heat conduction (planar, axisymmetric, polar / spherical, 3D), electric currents /
electrostatics, magnetostatics (2D, 3D) and AC magnetic fields, semiconductor drift-diffusion, solid mechanics (2D,
3D), elastic waves and laminar flow, and their couplings (Joule heating, thermo-optic, Pockels, strain-optic,
photoacoustic, induction heating, convection, ...). Studies: stationary, time dependent, frequency domain, mode
analysis, eigenfrequency, optimization, parameter sweeps and mesh refinement. Meshes: graded tensor grids, and FEM on
triangles / tetrahedra for the physics that declare them. Models are JSON files (ModelSpec) that a GUI, a headless
API and a CLI all read; scripts can also drive the solvers directly.

## Install

Step-by-step conda setup (separate environment, optional cupy / GPU): `INSTALL.md`. In short:

- Python 3.11. For the GPU backend an NVIDIA GPU with a CUDA toolkit / driver and cupy, whose wheel is
  CUDA-specific: `pip install cupy-cuda12x` (CUDA 12) or `cupy-cuda11x`. Without cupy open_sol and the GUI still
  start: Solvers 'Run on' Automatic then picks the processor (CPU backend), and the physics types without a CPU
  backend are greyed out / refused by validation.
- From the repository root: `pip install -e ".[gui]"` (extras: `cuda12` / `cuda11` = cupy, `gui` = PySide6 +
  pyqtgraph, `cudss` = nvmath-python for the cuDSS sparse direct solver on the GPU, which some 3D FEM examples and
  self-tests need). numpy, scipy and matplotlib are always installed. Install editable (`-e`): the GUI's Examples menu
  reads `examples/` beside the package.
- Without installing, run everything from the repository root (`python -m open_sol ...`); the example scripts put
  the root on `sys.path` themselves.
- scipy is required: the CPU backend, the host direct solvers (Automatic `sparse_direct` = SuperLU for FDFD, NLFD,
  drift-diffusion, magnetic AC), the eigensolvers (modes, eigenfrequencies) and the slab-mode solver of mode ports
  use it.
- Check the install: `python -m open_sol selftest templates_validate` (CPU), or `python -m open_sol selftest` for the
  quick analytic validation cases (GPU, minutes; `--heavy` the slow ones, `--examples` the analytic examples, `--all`
  everything: every case, then the examples on the cpu and the gpu backend).

## Quickstarts

GUI: `python -m open_sol.gui`.

1. Template, check, run (headless, the path for agents; the same as `examples/headless_run.py`):

   ```
   python -c "from open_sol import api; api.save(api.template('dielectric_cylinder'), 'cylinder.json')"
   python -m open_sol check cylinder.json          # CPU: issues, mesh, step estimate -> cylinder_out/check.json
   python -m open_sol run cylinder.json --figures  # GPU -> cylinder_out/summary.json, log.txt, results.npz, figures/
   python -m open_sol show cylinder_out            # summary of a results directory
   ```

   `python -m open_sol describe templates` lists every template key; `describe <type>` documents one physics,
   feature or study type. Exit codes: 0 ok, 1 model / run failure, 2 usage.

2. A multiphysics template: `thermo_optic_heater` (FDFD + heat + electric currents, stationary). A gold strip heater
   with sigma_dc(T) heats a silicon slab waveguide; Joule, thermo-optic and resistive feedback iterate:

   ```python
   from open_sol import api
   spec = api.template('thermo_optic_heater')
   with api.run(spec, out_dir='heater_out') as results:
       print(results.status, results.summary()['derived'])   # P_heater, T_max, T_waveguide
       print(results.result('em.through')['series']['fraction_positive'])   # xy result: series by name
   ```

3. A Simulation script (EM only, no model file; see `examples/em/thin_film.py`):

   ```python
   from open_sol import Simulation
   sim = Simulation(size_x=6e-6, size_y=3e-6, dx=25e-9, npml=15, boundaries='periodic_y')
   sim.add_dielectric_rectangle(2.75e-6, 0, 3.25e-6, 3e-6, eps_r=2.25)
   # axis 'y' = a line along y at x = position (sources and flux monitors alike)
   sim.add_line_source(axis='y', position=1.2e-6, wavelength=600e-9, bandwidth=0.8, waveform='gaussian')
   flux = sim.add_flux_monitor(axis='y', position=4.5e-6, span=(0, 3e-6), wavelengths=[500e-9, 600e-9])
   sim.run(duration=80e-15)
   print(flux.flux())   # a.u., > 0 toward +x: divide by an empty-domain run for transmission
   ```



## Which API when

| Need | Use |
|---|---|
| Build, check and run models without Python state; agents | CLI `python -m open_sol describe / check / run / show / selftest` |
| The same from Python, results as arrays | `open_sol.api`: `template`, `load` / `save`, `check`, `run` -> `Results` (parameter sweeps: `examples/multiphysics/tfln_ring_thermo_optic.py`) |
| Custom loops over a built model (solver steps the studies do not offer) | `open_sol.build(spec)` -> Session (`physics`, `runner.couplings`, `field`, `derived_values`) |
| EM scripts without a model file, direct solver access | `open_sol.Simulation` (see `examples/em/*.py`) |
| Interactive editing and plots | `python -m open_sol.gui` |

Never run `api.run` and the GUI's worker in one process: both use the GPU on their own thread.

## Links

- `open_sol/api_reference.md`: the full model reference (= `python -m open_sol describe`; types, settings, result
  kinds, file formats; templates as a one-line index).
- `open_sol/api_templates.md`: the model JSON of every template (= `python -m open_sol describe templates`).
- `examples/README.md`: example scripts and GUI model files.
- `structure_conventions.md`: code structure, conventions and gotchas for developers.

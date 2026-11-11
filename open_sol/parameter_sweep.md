# Parameter sweeps: design proposal

Planning note (not a convention file), indexed in `plans_overview.md` section 1; delete after implementation.
Status 2026-10-18: implemented except Aitken; the example (4.7) is written, not run. Done: fixed-point relaxation as one study setting
`relaxation` (decision 1); coupled frequency_domain (`StudyType.coupled_points`, `study._run_coupled_wavelengths`,
validate info instead of the reader errors when every physics supports stationary); derived series (summary
`derived_series`, an xy of lists in summary.json, decision 2; source-free derived values only); `SweepSpec.keep_fields`
(results.npz `sweep_snapshot/...`, no format bump, `Results.sweep_field`, GUI Field tab 'Sweep value'); sweep figures as
`analytic.sweep_overlay_xy` + `Results.sweep_figure` (no `result_plots.sweep_figure`: `_draw_xy` draws `series_x`;
`figure()` keeps the last value, `figure(series=)`), run / show `--figures` sweep_*.png, CLI `show --series`;
`SweepValueDone(index, count, parameter, value, sweep)` (the partial table, not separate fields) with the GUI Sweep tab
filling in live; GUI Sweep tab series per xy / flux result and derived-series entries (value choice and CSV export are
the plot view's Curves tab and Export curve data); `Results.sweep`; self-tests frequency_domain_feedback_strip,
sweep_power_strip, sweep_overlay_helpers (model/selftests_sweep.py); the TFLN example (4.7, see its status there). Open:
per-coupling Aitken relaxation and its oscillating self-test (4.8), the later items of 4.4. The path:line
references below are from the tree of the original writing and are stale.

Driving use case: the TFLN ring with self-heating (`examples/multiphysics/tfln_ring_thermo_optic.json`). Run the whole
wavelength spectrum twice, at two optical powers, and plot the two through-port spectra on top of each other in the
GUI. The general goal: vary one parameter, rerun the whole study per value, and get every result as one curve per
value (spectra, probes, derived values vs the inner axis), scalars vs the parameter, and optionally the fields of
every value.

## 1. The industry-standard approach

- **Outer parametric sweep** (a study-level loop): a table of parameter names and value lists (start, step, stop),
  either specified combinations or all combinations, optional reuse of the previous value's solution (continuation
  across values), and a choice of keeping every solution, only the last one, or writing them to disk. Every study
  step runs once per value, so the model can change between values (geometry, mesh, materials, source power).
- **Inner sweeps** belong to the study step: the frequency list of a frequency-domain step, the output times of a
  time-dependent step, or an auxiliary sweep of a stationary step. These run inside one solver sequence, where
  continuation makes each point start from the previous solution. This is what tracks a branch: a bistable or
  thermal-triangle response exists only with continuation.
- **EM heating with a frequency scan**: a combined study (EM in the frequency domain at one frequency, heat
  stationary, solved coupled), plus an auxiliary sweep over the frequency with continuation. The power is an outer
  parametric sweep.
- **Results**: each outer value is a stored outer solution. 1D plots (line, point, global graphs) take a parameter
  selection for the inner axis and can draw every outer solution, one curve per parameter value with an automatic
  legend. 2D plots pick one parameter value from a list. Global evaluation tables get one row per value.

Mapping to open_sol: the outer sweep is `StudySpec.sweep`. The inner sweep is the study's own axis (time steps,
frequency_domain wavelengths). A coupled solve at one wavelength is the stationary fixed point.

## 2. What open_sol has today

- **Spec**: `SweepSpec(parameter, values)` (`model/spec.py:104`) holds one parameter and a list of expression values.
  `_check_sweep` (`model/validate.py:678`) checks the parameter name and the values.
- **Run**: `_run_sweep` (`model/study.py:296-393`) copies the spec per value, sets the parameter, rebuilds, and runs
  the study type. It closes the previous value's session before the next build. Per value it keeps the derived
  values (evaluated at the end of the run) and every feature result (`_feature_results`, `study.py:428`; dft_fields
  as metadata only). `summary['sweep']` = {parameter, values, derived {name: [value per value]}, results [{'physics.
  feature': result} per value]}. Only the last value's session survives: earlier fields and snapshots are lost.
- **GUI**: the 'Parameter sweep' group in the study settings (`gui/settings_panel.py:874`). The Sweep tab
  (`gui/results.py:1142-1188`) offers two things:
  - a derived value vs the parameter;
  - any 1D result overlaid as one curve per value, with a legend "P = v".

  For xy results it plots only the FIRST series (`_sweep_curve`, `gui/results.py:225-241`).
- **Headless**: `ResultsData.sweep_results` (`api.py:446-448`); results.npz stores each value's results under
  `sweep/<index>/<key>/...` (`io/results_io.py:266`); summary.json keeps digests (`api.py:512`). `Results` has no
  sweep accessor, `viz/result_plots.py` draws no sweep figure, and the CLI prints the derived values per value
  (`__main__.py:327`).
- **frequency_domain feedback**: frequency_domain refuses couplings and material properties that read the harmonic
  (FDFD) physics (`model/validate.py:611-641`, `model/study.py:574-580`). The background (heat) is solved once, and
  every wavelength sees the same temperature, so absorption cannot feed back.
- **Optical power**: the FDFD example's power is the mode_source param `power: 1.0` W/m
  (`examples/em/tfln_ring_resonator_fdfd.json:65`), not a parameter. Fields scale with sqrt(P), so that is harmless
  without heating. With heating it must be a parameter: `P_in` in tfln_ring_thermo_optic.json, used as the source
  power.

## 3. Gaps for "two powers, two spectra on top of each other"

1. **Blocker**: frequency_domain cannot couple absorption back into heat (section 2).
2. The Sweep tab plots the first xy series. For a mode monitor that is `power_positive` in W/m, which scales with
   P_in, so the 1 W/m spectrum is a flat line next to the 4000 W/m one. It needs a series choice
   (`fraction_positive`).
3. There are no scalars along the inner axis (e.g. ring temperature vs wavelength), because derived values are
   evaluated only at the end of a run.
4. Fields of all but the last sweep value are discarded.
5. Headless runs have no overlay figure and no accessor.

## 4. Design

### 4.1 Two levels

- **Inner sweep**: the study's own axis, solved in one build with the state carried from point to point
  (continuation). Time for time_dependent, wavelengths for frequency_domain; later an auxiliary parameter list for
  stationary.
- **Outer sweep**: the existing `SweepSpec`, one build plus a full study per value. Anything may change between
  values.
- Rule: anything that needs continuation, or that is the natural axis of a plot, goes inner; everything else goes
  outer. With self-heating, the wavelength must be inner (the hot branch exists only when the temperature carries
  over). The power is outer.

### 4.2 Coupled frequency_domain (prerequisite for the example)

- **Validation**: drop the error for couplings and material properties that read a harmonic physics. Report an info
  instead: "coupled wavelength sweep: each wavelength is a stationary fixed point of all physics (max_iterations,
  tolerance), warm-started from the previous wavelength in list order".
- **Run**: when any enabled runner coupling reads a harmonic physics, solve the wavelengths coupled. For each
  wavelength, set `solver.omega` on every harmonic physics (`fdfd_2d.py:114-121` marks the system and solution
  dirty), call the existing `solve_stationary(session, all physics, ...)` (`study.py:487-543`; FDFD `step` solves
  when dirty and records its monitors), then record the snapshot. There is no separate background stage: the first
  wavelength's fixed point includes it. Without such couplings, today's path is unchanged.
- **Continuation and scan direction**: nothing is reset between wavelengths, so each wavelength continues from the
  previous one. List order is the scan direction: an ascending range scans blue to red (the thermal triangle of a
  red-shifting ring), and an explicit descending `wavelengths` list scans red to blue (hysteresis). Monitor results
  stay sorted by wavelength; snapshot labels keep the solve order.
- **Relaxation**: the plain fixed point oscillates once the loop gain falls below -1 (here a thermal shift above
  about one linewidth). Add a coupling param `relaxation` to em_heat_joule, currents_heat_joule and field_transfer:
  - `'none'` (default; today's behaviour).
  - `'aitken'`: Aitken / Irons-Tuck dynamic factor, starting at 0.5 and clipped to [0.05, 2]; the prototype is
    `aitken_relaxation` in the example script. The term injects `Q_prev + w (Q_new - Q_prev)` and keeps `Q_prev`
    between fixed points (continuation).

  `solve_stationary` calls `begin_fixed_point()` on couplings that have it, which resets only the Aitken history.
  The convergence test (`_FixedPointMonitor`, `study.py:892-915`) divides the relative change by the smallest current
  w, because a relaxed step is w times the Picard step. Without that, a small w would look converged. Relaxation is
  valid in stationary and frequency_domain; in time_dependent it is a validation error (it would low-pass the
  transfers). Per coupling rather than per study: it is explicit, works in the stationary study too, and mirrors
  the industry-standard per-step damping.
- **Progress and log**: the progress message is 'lambda = 1550 nm, iteration 3'. The log gets one line per
  wavelength with its iteration count, plus a warning when a wavelength does not converge (the scan continues).
- **Help texts to update**: the frequency_domain description, the em_heat_joule description, the FDFD wavelength
  help (`definition_fdfd.py:34`), and the background-study messages. Then the conventions file, and the user
  regenerates api_reference.md.

### 4.3 Derived values along the inner axis ("derived series")

- In `_SnapshotRecorder.record` (`study.py:799`), evaluate every derived value at each recorded snapshot:
  - time_dependent: the output times;
  - frequency_domain: every wavelength;
  - stationary: the final state.

  The x axis is the snapshot time or wavelength. Evaluation reads the live state, not the resampled snapshot.
- **Storage**: `summary['derived_series']`, shaped exactly like an xy result: {kind 'xy', x, x_label 'Wavelength' or
  'Time', x_unit 'm' or 's', series {name: values}, series_units}. It is present when a run has 2 or more points, so
  the xy plotting, CSV and results_io paths take it unchanged. In a sweep: `sweep['derived_series']` holds one per
  value.
- **Cost**: one GPU reduction per derived value per snapshot, negligible next to an FDFD solve. It is also useful
  without any sweep, e.g. P_absorbed vs wavelength (an absorption spectrum) or T_max vs time.

### 4.4 Outer sweep spec

- Keep `SweepSpec(parameter, values)` and add `keep_fields: bool = False`. When true, store every value's snapshots
  (already host FieldData, `study.py:804-806`) in the sweep table before closing its session. The total over the
  sweep stays under the existing 1 GB snapshot limit; past it, only each value's final snapshot is kept, with a
  warning.
- **Later** (the table is one row per sweep point, so this needs no format change):
  - several parameters (specified or all combinations; each row gets {parameter: value});
  - value ranges (`linspace` / `range` in the values editor);
  - outer continuation (reuse the previous value's state when the mesh is unchanged);
  - skipping the rebuild for parameters that reach only runtime settings.

### 4.5 Results storage and access

- **summary['sweep']** gains `derived_series` (one xy dict or None per value) and, with keep_fields, `snapshots`
  (a list per value).
- **results_io** adds `sweep/<index>/derived_series/...` and `sweep/<index>/snapshot/<n>/<field>` arrays and bumps
  the format version (no backward compatibility needed).
- **Results API**:
  - `Results.sweep()` -> {parameter, values, derived};
  - `Results.sweep_result(physics, feature)` -> [result dict per value];
  - `Results.sweep_field(name, index, snapshot=-1)` with keep_fields.
- **Figures**: `result_plots.sweep_figure(...)` overlays one curve per value (legend 'P_in = 4000') for a result
  series or a derived series, and plots a derived value vs the parameter. `Results.figure('em.through_port', path,
  series='fraction_positive')` overlays every value when the run had a sweep. `api.run(figures=True)` draws one
  overlay per xy result plus derived-vs-parameter. The CLI `show` gets a `--series` option.

### 4.6 GUI

- **Sweep tab**: the quantity combo (exists) gains:
  - a series combo for xy results (unit groups via `result_plots.xy_series_groups`, as in 'Curves & tables');
  - derived-series entries ('Derived value vs wavelength: T_ring');
  - a checkable list of the values (default all);
  - CSV export (x plus one column per value).
- **Field view**: with keep_fields, a 'Sweep value' combo next to the snapshot slider. Fields of earlier values come
  from the stored host snapshots, with no session needed.
- **Settings panel**: a 'Keep fields of every value' checkbox in the Parameter sweep group.
- **Live update**: a `SweepValueDone(index, value, derived, derived_series, results)` event (CPU dicts) so the Sweep
  tab fills in as values finish. Without it, a two-hour sweep shows nothing until the end.

### 4.7 The TFLN example after implementation

Status: written 2026-10-18, not run (needs the user's GPU run). As built: scan `lambda_cold` - 3 nm .. + 12 nm, 61
wavelengths (0.25 nm), sweep `P_in` over [1, 4000]; no Aitken yet, so the study `relaxation` is the fixed expression
`w_relax` = 1 / (1 + P_in / P_relax) (~1 cold, 0.2 at 4000 W/m), tolerance 1e-3, max_iterations 60; derived values
`P_absorbed`, `A_ring` (absorbed fraction), `dT_ring`, `dT_max` (rises, not absolute T). `lambda_cold` is a
placeholder (1550 nm): the script first runs the model without its sweep at 1 W/m over one FSR to find the dip, then
`api.run(spec, parameters={'lambda_cold': dip})`, `Results.sweep_result` / `sweep_figure`, maps by `snapshot_field`.
To check on the run: the cold dip lies inside the scan, the 4000 W/m triangle ends before its red end, every
wavelength converges (sweep `converged`, per-wavelength log lines) and the iteration counts. Once Aitken (4.8) exists,
replace `w_relax` with it.

- **JSON**: a frequency_domain study around the cold resonance (e.g. 1545-1557 nm in 0.2 nm steps, 61 wavelengths),
  sweep `P_in` over [1, 4000], coupling joule with `relaxation: 'aitken'` plus thermo_optic, and derived values
  `P_absorbed`, `T_ring`, `T_max`.
- **GUI**: in the Sweep tab, pick 'Result: em.through_port' with series `fraction_positive` for the two spectra
  overlaid, and 'Derived value vs wavelength: T_ring' for the ring temperature per power.
- **Script**: `api.run(spec)`, then `Results.sweep_result('em', 'through_port')`, then the overlay (or
  `Results.figure`). The custom fixed-point loop in tfln_ring_thermo_optic.py goes away, and its Aitken function
  moves into the coupling.
- **Cost**: about 1 FDFD factorisation per wavelength at 1 W/m and 3-5 per wavelength at 4000 W/m.

### 4.8 Self-tests

- `frequency_domain_feedback_strip`: the em_heat_thermo_optic_strip model (`coupling/selftests.py`) run as
  frequency_domain at its one wavelength reproduces the stationary result (absorbed power, T profile, phase) to the
  stationary tolerance.
- `sweep_power_strip`: a 2-value sweep of the strip's source power with dndT = 0 (a linear problem). The derived
  T_max ratio equals the power ratio, there is one result entry per value, and each value's derived series has one
  point per wavelength.
- Relaxation: a model whose plain fixed point oscillates (absorption that drops with T: a lossy Fabry-Perot slab
  tuned to the blue flank) converges with `'aitken'`. The model is still to be designed; the strip's feedback is too
  weak for this.

### 4.9 Implementation order

1. Coupled frequency_domain plus coupling relaxation: `model/study.py`, `model/validate.py`,
   `coupling/definitions.py`, a relaxed term in `coupling/coupling_term.py`, help texts, the selftest.
2. Derived series: the study recorder, summary, results_io, api digests, describe docs, the GUI Derived values tab.
3. Sweep presentation: the Sweep tab series and derived-series overlays, CSV export, `Results.sweep` /
   `sweep_result`, `result_plots.sweep_figure`, api figures, CLI.
4. `keep_fields` (optional): spec field, recorder budget, results_io, GUI field selector.
5. The example: JSON, a reduced script, README.

## 5. Decisions needed

1. Relaxation per coupling (recommended) or one study setting?
2. Derived series as a summary entry shaped like an xy result (recommended), or as a pseudo feature result
   (`Results.result('study', 'derived')`)?
3. `keep_fields` now or later? It is not needed for the spectra overlay.
4. The live `SweepValueDone` event: now (recommended for long sweeps) or later?
5. Multi-parameter sweeps later (recommended) or now?
6. Scan direction by list order only (recommended), or an explicit frequency_domain setting?

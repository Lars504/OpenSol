"""Part targets shared by the 2D FDFD / em_modes / em_fdfd_nonlinear profilers (profiler_solver_em_fdfd_*.py,
profiler_solver_em_modes*.py): study runners, Session reads, couplings, the FDFD / mode engines and their helpers."""


def parts(prefix, owner, methods):
    """(('<owner>.<method>', '<prefix>: <method>'), ...) of one class or module."""
    separator = ':' if ':' not in owner else '.'
    return tuple((f'{owner}{separator}{method}', f'{prefix}: {method}') for method in methods)


# study runners and recording (frequency_domain / mode_analysis / eigenfrequency / stationary / optimization / nlfd)
STUDY_RUNNERS = parts('study', 'open_sol.model.study', (
    '_run_stationary', 'solve_stationary', '_run_frequency_domain', '_solve_wavelengths', '_run_coupled_wavelengths',
    '_run_mode_analysis', '_background_stage', '_SnapshotRecorder.record',
)) + (
    ('open_sol.model.study_eigenfrequency:_run_eigenfrequency', 'study: _run_eigenfrequency'),
    ('open_sol.model.study_optimization:_run_optimization', 'study: _run_optimization'),
    ('open_sol.model.study_optimization:_DesignProblem.evaluate', 'study: _DesignProblem.evaluate'),
    ('open_sol.model.study_nonlinear:_run_nonlinear_frequency_domain', 'study: _run_nonlinear_frequency_domain'),
)
# Session reads per grid kind (snapshots, derived values, checks)
TENSOR_READS = parts('session tensor', 'open_sol.model.field_access_tensor:TensorAccess',
                     ('field', 'display_field', 'evaluate', 'values_at'))
TRIANGLE_READS = parts('session triangles', 'open_sol.model.field_access_triangles:TriangleAccess',
                       ('field', 'display_field', '_field_data', 'fem_field', 'evaluate', 'values_at'))
# couplings (joule / thermo-optic runs)
COUPLINGS = (
    ('open_sol.coupling.coupling_term:CouplingTerm.apply_now', 'coupling: CouplingTerm.apply_now'),
    ('open_sol.coupling.coupling_term:CouplingTerm.transfer_value', 'coupling: CouplingTerm.transfer_value'),
    ('open_sol.coupling.coupling_term:CouplingGroup.apply_now', 'coupling: CouplingGroup.apply_now'),
    ('open_sol.coupling.slot_sum:SlotSum.apply_now', 'coupling: SlotSum.apply_now'),
    ('open_sol.coupling.slot_sum:SlotSum.transfer_value', 'coupling: SlotSum.transfer_value'),
)
# per-solve material reads shared by the FDFD / mode engines (one device array op + host copies on the host engines)
HARMONIC_MATERIALS = parts('em harmonic', 'open_sol.physics.em.harmonic', (
    'cell_permittivity', 'cell_permeability', 'absorbed_power_ez', 'absorbed_power_hz', 'face_current_rhs',
))
EIGEN = (
    ('open_sol.physics.em.eigenfrequency:solve_resonances', 'em eigen: solve_resonances'),
    ('open_sol.physics.em.eigenfrequency:resonance_table_result', 'em eigen: resonance_table_result'),
)

# ---- tensor FDFD (em_fdfd_ez / em_fdfd_hz: physics/em/solvers/fdfd_2d) ----
FDFD = 'open_sol.physics.em.solvers.fdfd_2d:FDFDPhysics'
FDFD_TENSOR = parts('fdfd', FDFD, (
    'finalize', 'step', 'solve_harmonic', 'plan_harmonic_frequencies', 'solve_eigenfrequencies', 'select_resonance',
    '_solve', '_refresh_system', '_assemble', '_solve_port_modes', '_refresh_state', '_add_sheet_losses',
    '_fill_wall_rows', '_solve_linear', '_transposed_solve', '_symmetrizer', '_pin_outer_cells', '_conformal_arrays',
    'assembly_eps_xy', 'record_monitors', 'face_fields', 'face_curls', 'eps_sensitivity', 'solve_adjoint',
    '_check_solver_size', '_off_diagonal_line_warnings',
)) + parts('em operators', 'open_sol.physics.em.operators', (
    'yee_operators', 'helmholtz_operator', 'pin_image_cells', 'pin_cells', 'stretched_cell_areas',
    'off_diagonal_face_couplings',
)) + (
    ('open_sol.physics.em.fdfd_sources:PlaneWavePhasor.add_to_rhs', 'fdfd source: PlaneWavePhasor.add_to_rhs (TFSF)'),
    ('open_sol.physics.em.fdfd_sources:SheetCurrentPhasor.add_to_rhs', 'fdfd source: SheetCurrentPhasor.add_to_rhs'),
    ('open_sol.physics.em.fdfd_sources:SheetCurrentPhasor.face_currents',
     'fdfd source: SheetCurrentPhasor.face_currents'),
    ('open_sol.physics.em.fdfd_sources:ElectricDipolePhasor.add_to_rhs', 'fdfd source: ElectricDipolePhasor.add_to_rhs'),
    ('open_sol.physics.em.fdfd_sources:ElectricDipolePhasor.face_currents',
     'fdfd source: ElectricDipolePhasor.face_currents'),
    ('open_sol.physics.em.fdfd_sources:PointCurrentPhasor.add_to_rhs', 'fdfd source: PointCurrentPhasor.add_to_rhs'),
    ('open_sol.physics.em.fdfd_ports:ModeSourcePhasor.solve', 'fdfd port: ModeSourcePhasor.solve'),
    ('open_sol.physics.em.fdfd_ports:ModeSourcePhasor.add_to_rhs', 'fdfd port: ModeSourcePhasor.add_to_rhs'),
    ('open_sol.physics.em.fdfd_ports:HarmonicModeMonitor.solve', 'fdfd port: HarmonicModeMonitor.solve'),
    ('open_sol.physics.em.fdfd_ports:HarmonicModeMonitor.record', 'fdfd port: HarmonicModeMonitor.record'),
    ('open_sol.physics.em.mode_ports:PortLine.solve_modes', 'fdfd port: PortLine.solve_modes'),
    ('open_sol.physics.em.slab_modes:solve_slab_modes', 'fdfd port: slab_modes.solve_slab_modes'),
    ('open_sol.physics.em.fdfd_monitors:HarmonicProbe.record', 'fdfd monitor: HarmonicProbe.record'),
    ('open_sol.physics.em.fdfd_monitors:HarmonicFluxMonitor.record', 'fdfd monitor: HarmonicFluxMonitor.record'),
    ('open_sol.physics.em.fdfd_monitors:HarmonicNearToFar.record', 'fdfd monitor: HarmonicNearToFar.record'),
    ('open_sol.physics.em.fdfd_monitors:harmonic_far_field_result', 'fdfd result: harmonic_far_field_result'),
    ('open_sol.physics.em.fdfd_monitors:harmonic_flux_result', 'fdfd result: harmonic_flux_result'),
    ('open_sol.physics.em.fdfd_ports:harmonic_mode_monitor_result', 'fdfd result: harmonic_mode_monitor_result'),
    ('open_sol.physics.em.renderer:_FDFDRendererBase.device_field', 'fdfd renderer: device_field'),
) + parts('em adjoint', 'open_sol.physics.em.adjoint', (
    'eps_sensitivity', 'objective_cotangents', 'adjoint_source', 'permittivity_gradients', 'sensitivity_density',
)) + HARMONIC_MATERIALS + EIGEN

# ---- FDFD on triangles (physics/em/solvers/fdfd_2d_fem, fem_fdfd) ----
FDFD_FEM = 'open_sol.physics.em.solvers.fdfd_2d_fem:FEMFDFDPhysics'
FDFD_TRIANGLES = parts('fdfd fem', FDFD_FEM, (
    'finalize', 'step', 'solve_harmonic', 'solve_eigenfrequencies', 'select_resonance', '_solve', '_refresh_system',
    '_show', '_materials', '_current_layout', 'fem_field', 'error_indicator',
)) + (
    ('open_sol.physics.em.solvers.fdfd_2d_fem:_PointExcitation.load', 'fdfd fem: _PointExcitation.load'),
    ('open_sol.physics.em.solvers.fdfd_2d_fem:FEMHarmonicProbe.record', 'fdfd fem: FEMHarmonicProbe.record'),
) + parts('fem_fdfd', 'open_sol.physics.em.fem_fdfd', (
    'fem_fdfd_layout', 'assemble_fem_fdfd', 'solve_fem_fdfd_resonances', 'point_load', 'dipole_load', 'point_fields',
    'element_state', 'nodal_fields', 'fem_fdfd_error_indicator',
)) + parts('fem assembly', 'open_sol.fem.assembly', (
    'assemble_stiffness', 'assemble_mass', 'restrict',
)) + (
    ('open_sol.fem.renderer:FEMFieldRenderer.fem_field', 'fem renderer: fem_field'),
) + HARMONIC_MATERIALS + EIGEN

# ---- em_modes on tensor meshes (physics/em/solvers/modes_2d, cross_section_modes) ----
MODES = 'open_sol.physics.em.solvers.modes_2d:ModePhysics'
MODES_TENSOR = parts('modes', MODES, (
    'finalize', 'solve_modes', 'solve_harmonic', 'select_mode', '_solve', '_solve_at', '_prepared',
    '_cross_section_materials', '_dual_cuts', 'point_materials', 'error_indicator',
)) + (
    ('open_sol.physics.em.solvers.modes_2d:ModeTableMonitor.record', 'modes: ModeTableMonitor.record'),
    ('open_sol.physics.em.solvers.modes_2d:ModeDispersionMonitor.record', 'modes: ModeDispersionMonitor.record'),
    ('open_sol.physics.em.solvers.modes_2d:mode_table_result', 'modes: mode_table_result'),
    ('open_sol.physics.em.renderer:ModeRenderer.device_field', 'modes renderer: device_field'),
) + parts('cross_section', 'open_sol.physics.em.cross_section_modes', (
    'prepare_cross_section_modes', 'assemble_cross_section', 'solve_prepared_modes', '_build_mode', 'centre_fields',
    'mode_group_indices', 'cross_section_layout', 'mode_point_materials', 'fill_images',
)) + HARMONIC_MATERIALS

# ---- em_modes on triangles (physics/em/solvers/modes_fem, fem_modes) ----
MODES_FEM = 'open_sol.physics.em.solvers.modes_fem:FEMModePhysics'
MODES_TRIANGLES = parts('modes fem', MODES_FEM, (
    'finalize', 'solve_modes', 'solve_harmonic', 'select_mode', '_solve', '_materials', '_current_layout',
    'fem_field', 'error_indicator',
)) + (
    ('open_sol.physics.em.solvers.modes_2d:ModeTableMonitor.record', 'modes: ModeTableMonitor.record'),
    ('open_sol.physics.em.solvers.modes_2d:mode_table_result', 'modes: mode_table_result'),
) + parts('fem_modes', 'open_sol.physics.em.fem_modes', (
    'fem_mode_layout', 'prepare_fem_modes', 'solve_fem_modes', '_build_mode', '_sample_fields', '_rayleigh_beta_squared',
    'sample_permeability', 'element_means', 'nodal_fields', 'mode_error_indicator', 'pml_ring',
)) + parts('fem assembly', 'open_sol.fem.assembly', (
    'assemble_curl_curl', 'assemble_vector_mass', 'assemble_gradient_coupling', 'assemble_stiffness', 'assemble_mass',
    'restrict',
)) + (
    ('open_sol.fem.renderer:FEMFieldRenderer.fem_field', 'fem renderer: fem_field'),
) + HARMONIC_MATERIALS

# ---- em_fdfd_nonlinear (physics/nlfd: channels are tensor FDFDPhysics) ----
NLFD = 'open_sol.physics.nlfd.solver:NonlinearHarmonicPhysics'
NLFD_PARTS = parts('nlfd', NLFD, (
    'finalize', 'step', 'solve_nonlinear', 'select_channel', 'solve_design', 'eps_sensitivity', '_build_mixing',
    'take_absorbed_power', 'scalars', 'state',
)) + parts('nlfd iteration', 'open_sol.physics.nlfd.iteration', (
    'solve_undepleted', 'solve_pump_depleted', 'solve_newton_krylov', 'real_gmres', 'channel_residual',
    'AndersonMixer.update', 'ScaledSource.add_to_rhs',
)) + parts('nlfd sources', 'open_sol.physics.nlfd.sources', (
    'NonlinearPolarizationPhasor.density', 'NonlinearPolarizationPhasor.add_to_rhs',
    'NonlinearPolarizationPhasor.face_currents', 'KerrOperatorTerm.update', 'FreeCarriers.refresh',
    'ChannelFieldStore.capture', 'ChannelFieldStore.read', 'ChannelFieldStore.write', 'ChannelFieldStore.grid_fields',
    'NewtonLinearization.tangent_context', 'NewtonLinearization.source', 'DensityPhasor.add_to_rhs',
    'polarization_density', 'polarization_tangent', 'kerr_diagonal',
)) + parts('nlfd sheets', 'open_sol.physics.nlfd.sheets', (
    'site_fields', 'sheet_site_polarizations', 'sheet_grid_density', 'sheet_cotangents',
)) + parts('nlfd adjoint', 'open_sol.physics.nlfd.adjoint', (
    'cascade_sensitivity', 'depleted_cotangents', 'transposed_channel_solve', 'polarization_cotangents',
)) + (
    ('open_sol.physics.nlfd.absorption:absorbed_power_density', 'nlfd: absorbed_power_density'),
    ('open_sol.physics.nlfd.renderer:NonlinearRenderer.device_field', 'nlfd renderer: device_field'),
)

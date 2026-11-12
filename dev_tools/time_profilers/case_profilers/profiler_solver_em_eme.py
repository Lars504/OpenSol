"""Profile em_eme (2D eigenmode expansion) on the gpu: the ez / hz taper optimisations (cascade adjoint, 10 L-BFGS-B
iterations, 3.3 + 3.7 s) and the radiating cpml step of eme_cpu_gpu on the gpu (n_modes 0). Run from any
folder: python profiler_solver_em_eme.py [--runs NAME ...] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_em_bor_fdtd_common import methods  # noqa: E402
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

# (case, runs or None = all)
CASES = (('eme_taper_optimization', None), ('eme_taper_optimization_hz', None), ('eme_cpu_gpu', ['step_cpml_gpu']))


def _specs():
    """{case__run: spec} of the CASES runs."""
    return {f'{case}__{run}': spec for case, runs in CASES for run, spec in case_specs(case, runs).items()}


EME = 'open_sol.physics.eme'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + (
    ('open_sol.model.study:solve_stationary', 'study: solve_stationary'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: _SnapshotRecorder.record'),
    ('open_sol.model.study:_feature_results', 'study: _feature_results'),
) + methods('open_sol.model.study_optimization:_DesignProblem', 'optimization', ('evaluate', 'accept', 'record')
) + methods('open_sol.physics.em.design_density:DesignRegion', 'design', ('write', 'set_density', 'density_gradient')
) + methods(f'{EME}.solver:EMEBase', 'eme', (
    'step', 'solve_harmonic', '_solve', '_section_modes', 'eps_sensitivity', '_port_scalars', 'port_fractions')
) + methods(f'{EME}.solver:EMEPhysics', 'eme 2d', (
    '_find_sections', '_host_stack', '_solve_section', '_reconstruct', '_fill_wall_cells', '_loss',
    '_cascade_sensitivity', '_add_section_gradients', '_sensitivity_slices', 'state')
) + (
    (f'{EME}.solver:EMEPortsMonitor.record', 'monitor: EMEPortsMonitor.record'),
    (f'{EME}.solver:EMESectionTableMonitor.record', 'monitor: EMESectionTableMonitor.record'),
    (f'{EME}.adjoint:EMEObjective.record', 'monitor: EMEObjective.record'),
    (f'{EME}.sections:stack_sections', 'sections: stack_sections'),
    (f'{EME}.sections:find_sections', 'sections: find_sections'),
    ('open_sol.physics.em.mode_ports:PortLine.read_materials', 'line: PortLine.read_materials'),
    ('open_sol.physics.em.mode_ports:PortLine.slab_line', 'line: PortLine.slab_line'),
    ('open_sol.physics.em.slab_modes:solve_slab_modes', 'line: solve_slab_modes'),
    (f'{EME}.mode_sets:mode_set', 'modes: mode_set'),
    (f'{EME}.mode_sets:section_fields', 'field: section_fields'),
    (f'{EME}.interfaces:interface_smatrix', 'cascade: interface_smatrix'),
    (f'{EME}.interfaces:interface_defects', 'cascade: interface_defects'),
    (f'{EME}.cascade:CascadeBuilder.add_section', 'cascade: CascadeBuilder.add_section'),
    (f'{EME}.cascade:CascadeBuilder.add_interface', 'cascade: CascadeBuilder.add_interface'),
    (f'{EME}.cascade:CascadeBuilder.finish', 'cascade: CascadeBuilder.finish'),
    (f'{EME}.cascade:Cascade.section_amplitudes', 'cascade: Cascade.section_amplitudes'),
    (f'{EME}.adjoint:section_gradient', 'adjoint: section_gradient'),
    (f'{EME}.adjoint:slice_exponentials', 'adjoint: slice_exponentials'),
    (f'{EME}.adjoint:slice_products', 'adjoint: slice_products'),
    (f'{EME}.adjoint:objective_cotangents', 'adjoint: objective_cotangents'),
    (f'{EME}.renderer:EMERenderer.device_field', 'renderer: EMERenderer.device_field'),
)

if __name__ == '__main__':
    main('em_eme', PARTS, __doc__, specs=_specs)

"""Profile solid_mechanics on tensor (Q1) meshes on the gpu: stationary (plane stress, generalized plane strain bending),
eigenfrequency (nearest, shifted, band, targets) and frequency response (piezo admittance sweep, elastic sweep), selftest
specs with refined meshes (~8-10 s gpu total). Run from any folder:
python profiler_solver_solid_mechanics_tensor.py [--repeat N] [--no-cprofile]"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_harness import SOLVER_ROLE_PARTS, STUDY_PARTS, case_specs, main  # noqa: E402

N_ELASTIC_FREQUENCIES = 60   # the elastic bar run swept over this many frequencies (selftest: 3)


def _scaled(value, factor):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value / factor
    return f'({value}) / {factor}'


def refined(spec, factor):
    """Copy of spec with every mesh size (dx, h_min, refinement / band dx) divided by factor."""
    spec = copy.deepcopy(spec)
    for mesh in spec.meshes:
        mesh.dx = _scaled(mesh.dx, factor)
        mesh.h_min = _scaled(mesh.h_min, factor)
        for refinement in mesh.refinements:
            for name in ('dx', 'dx_x', 'dx_y', 'dx_z'):
                setattr(refinement, name, _scaled(getattr(refinement, name), factor))
        for axis in mesh.axes:
            for band in axis.bands:
                band.dx = _scaled(band.dx, factor)
    return spec


def _elastic_sweep(spec):
    """The elastic bar run over N_ELASTIC_FREQUENCIES frequencies spanning its three selftest frequencies."""
    frequencies = [float(value) for value in spec.study.settings['frequencies']]
    sweep = np.linspace(0.8 * min(frequencies), 1.1 * max(frequencies), N_ELASTIC_FREQUENCIES)
    spec.study.settings = {**spec.study.settings, 'frequencies': [float(value) for value in sweep]}
    return spec


def _specs():
    specs = {}
    for run, spec in case_specs('solid_plane_stress').items():   # plate 40 x 20 um, bilayer: 16x the cells
        specs[f'stationary_plane_stress_{run}'] = refined(spec, 4)
    specs['stationary_gps_bending'] = refined(
        case_specs('solid_generalized_plane_strain_bending', runs=['along_x'])['along_x'], 4)
    for run, spec in case_specs('solid_eigenfrequency_roller_box').items():   # lowest / shifted, 160 x 64 cells
        specs[f'eigen_roller_box_{run}'] = refined(spec, 8)
    specs['eigen_band'] = refined(case_specs('solid_eigenfrequency_band_roller_box', runs=['rectangle'])['rectangle'], 2)
    specs['eigen_targets'] = refined(case_specs('solid_eigenfrequency_targets', runs=['targets'])['targets'], 2)
    bar = case_specs('solid_frequency_response_bar')
    specs['frequency_piezo'] = refined(bar['piezo'], 2)                       # 201 frequencies, admittance
    specs['frequency_elastic'] = _elastic_sweep(refined(bar['elastic'], 4))   # private SuperLU per frequency
    return specs


PHYSICS = 'open_sol.physics.mechanics.solver:SolidMechanicsPhysics'
SYSTEM = 'open_sol.physics.mechanics.system:ElasticSystem'
MODAL = 'open_sol.physics.mechanics.modal'
HARMONIC = 'open_sol.physics.mechanics.frequency_response:HarmonicSystem'
PIEZO = 'open_sol.physics.mechanics.piezo:PiezoSystem'
PARTS = STUDY_PARTS + SOLVER_ROLE_PARTS + tuple((f'{PHYSICS}.{method}', f'physics: {method}') for method in (
    'prepare', 'finalize', 'step', '_assemble', '_element_stiffness', '_assemble_piezo', '_build_rhs', '_total_load',
    '_solve_system', '_solve_generalized', '_write_outputs', 'reaction', 'solve_eigenfrequencies',
    'solve_eigenfrequencies_targets', 'solve_eigenfrequencies_band', '_modal_pencil', 'select_resonance',
    'reset_frequency_response', '_harmonic_inputs', '_harmonic_solve', 'plan_frequency_response',
    'solve_frequency_response', 'publish_frequency_response', '_show_frequency_response', '_build_harmonic_operators',
    '_host_stiffness_stencil', '_show_harmonic',
)) + tuple((f'{SYSTEM}.{method}', f'system: {method}') for method in (
    'prepare', 'assemble', '_check_rigid_modes', 'build_rhs', 'solve', 'solve_homogeneous', 'reaction',
    'element_kinematics', 'add_stress_load', 'add_volume_load',
)) + tuple((f'{HARMONIC}.{method}', f'harmonic: {method}') for method in (
    '__init__', '_factorize', 'solve', '_product', '_rows_solve', 'plan_reduced', 'set_output_rows', '_reduced_solve',
)) + tuple((f'{PIEZO}.{method}', f'piezo: {method}') for method in (
    '__init__', 'solve_static', 'solve_harmonic', 'harmonic_system', '_fixed_and_load', 'electrode_values',
    'modal_pencil', 'expanded',
)) + tuple((f'{MODAL}:ModalPencil.{method}', f'modal: ModalPencil.{method}') for method in (
    '__init__', '_pencil', 'solve_at', 'count_below', '_shift_solve', '_candidates', '_resonance', '_merged', 'release',
)) + (
    ('open_sol.physics.mechanics.frequency_response:solve_harmonic', 'harmonic: solve_harmonic (new system / freq)'),
    (f'{MODAL}:solve_modes_at_shifts', 'modal: solve_modes_at_shifts'),
    (f'{MODAL}:_BandSearch.run', 'modal: _BandSearch.run'),
    (f'{MODAL}:_BandSearch._march', 'modal: _BandSearch._march'),
    (f'{MODAL}:_BandSearch._fill_gaps', 'modal: _BandSearch._fill_gaps'),
    (f'{MODAL}:_BandSearch._repair_counts', 'modal: _BandSearch._repair_counts'),
    ('open_sol.physics.mechanics.system:stiffness_stencil', 'assembly: stiffness_stencil'),
    ('open_sol.physics.mechanics.system:mass_stencil', 'assembly: mass_stencil'),
    ('open_sol.physics.mechanics.system:ElasticPattern.__init__', 'assembly: ElasticPattern.__init__'),
    ('open_sol.physics.mechanics.system:ElasticPattern.csr', 'assembly: ElasticPattern.csr'),
    ('open_sol.physics.mechanics.stiffness:element_stiffness', 'assembly: element_stiffness'),
    ('open_sol.physics.mechanics.stiffness:check_positive_definite', 'assembly: check_positive_definite'),
    ('open_sol.physics.mechanics.piezo:coupled_operator', 'piezo: coupled_operator'),
    ('open_sol.physics.mechanics.piezo:coupled_mass', 'piezo: coupled_mass'),
    ('open_sol.physics.mechanics.generalized_plane_strain:unit_strain_solves', 'gps: unit_strain_solves'),
    ('open_sol.physics.mechanics.generalized_plane_strain:solve_strain_terms', 'gps: solve_strain_terms'),
    ('open_sol.fem.assembly:pattern_sum', 'fem: pattern_sum'),
    ('open_sol.fem.assembly:congruence', 'fem: congruence'),
    ('open_sol.solvers.sparse_direct:factorize_host', 'solver: sparse_direct.factorize_host (private SuperLU)'),
    ('open_sol.solvers.sparse_direct:DevicePencil.__init__', 'solver: DevicePencil.__init__'),
    ('open_sol.solvers.sparse_direct:DevicePencil.combined', 'solver: DevicePencil.combined'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.shift_pencil', 'solver: arpack shift_pencil'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver._factorize_shift', 'solver: arpack _factorize_shift'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.count_below', 'solver: arpack count_below'),
    ('open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver._superlu_count', 'solver: arpack _superlu_count'),
    ('open_sol.solvers.arpack_shift_invert:ShiftPencil.__init__', 'solver: ShiftPencil.__init__'),
    ('open_sol.solvers.arpack_shift_invert:pattern_combination', 'solver: pattern_combination'),
    ('open_sol.model.study:_solve_static_states', 'study: _solve_static_states'),
    ('open_sol.model.study:_numbered_snapshots', 'study: _numbered_snapshots'),
    ('open_sol.model.study:_SnapshotRecorder.record', 'study: snapshot record'),
    ('open_sol.model.study:_LiveFeed.update', 'study: live update'),
    ('open_sol.model.study_eigenfrequency:_search_solve', 'study: eigen _search_solve'),
    ('open_sol.model.study_frequency_response:_plan_reduced_sweeps', 'study: _plan_reduced_sweeps'),
    ('open_sol.physics.mechanics.renderer:SolidRenderer.device_field', 'render: SolidRenderer.device_field'),
    ('open_sol.model.field_access_tensor:TensorAccess.field', 'sampling: TensorAccess.field'),
    ('open_sol.model.field_access_tensor:TensorAccess.display_field', 'sampling: TensorAccess.display_field'),
    ('open_sol.model.field_access_tensor:TensorAccess.evaluate', 'sampling: TensorAccess.evaluate'),
)

if __name__ == '__main__':
    main('solid_mechanics_tensor', PARTS, __doc__, case=None, specs=_specs)

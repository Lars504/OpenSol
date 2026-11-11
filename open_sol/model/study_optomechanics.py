"""The 'optomechanics' StudyType: background physics to a stationary fixed point, the optical eigen-solve named by the
solid's optomechanical_coupling features (em_modes modes or em_fdfd_ez / _hz / _bor resonances), the solid vibration
modes, the g0 tables, then one labelled snapshot per mechanical mode."""
import time

from .schema import Issue, Param, StudyType
from .solver_hooks import EigenfrequencySolver, ModeSolver, OptomechanicalSolver
from .study import (LogEvent, _background_fixed_point_params, _background_stage, _count_progress,
                    _log_eigen_solve, _numbered_snapshots, _refuse_stepping_background, _reselect_first_resonance,
                    _scipy_issues, _selected_physics, _solve_static_states)

STUDY_NAME = 'optomechanics'
_OPTICAL_STAGE = 'Optical solve'
_MECHANICAL_STAGE = 'Mechanical modes'
_FEATURE = 'optomechanical_coupling'


def _roles(physics, selected=()):
    """(mechanical, optical, background) physics names in runner order: mechanical = solvers with optomechanical
    couplings (with `selected`, StudySpec.solved_physics: the listed ones), optical = the physics they name (each must
    be listed then); ValueError when none, for an unknown or unlisted name or a time-stepped background."""
    mechanical = [name for name, solver in physics.items()
                  if isinstance(solver, OptomechanicalSolver) and solver.optomechanical_couplings
                  and (not selected or name in selected)]
    if not mechanical:
        raise ValueError(f'no solid_mechanics physics has an {_FEATURE} feature: the {STUDY_NAME} study needs one'
                         + (' among StudySpec.solved_physics' if selected else ''))
    optical = []
    for name in mechanical:
        for coupling in physics[name].optomechanical_couplings:
            if coupling.optical_physics not in physics:
                raise ValueError(f"{_FEATURE} '{coupling.name}' names the optical physics "
                                 f"'{coupling.optical_physics}', which the model does not have")
            if selected and coupling.optical_physics not in selected:
                raise ValueError(f"{_FEATURE} '{coupling.name}' names the optical physics "
                                 f"'{coupling.optical_physics}', which StudySpec.solved_physics leaves out")
            if coupling.optical_physics not in optical:
                optical.append(coupling.optical_physics)
    solved = set(mechanical) | set(optical)
    background = [name for name in physics if name not in solved]
    _refuse_stepping_background(physics, background, f'an {STUDY_NAME} study')
    return mechanical, [name for name in physics if name in optical], background


def _run_optomechanics(session, values, emit, control, service):
    """Background fixed point, background couplings once, the optical solves (the coupled mode selected), the
    vibration modes with the g0 tables, one snapshot per mechanical mode number; mode 1 stays shown."""
    runner = session.runner
    mechanical, optical, background = _roles(runner.physics, _selected_physics(session))
    stage = _background_stage(
        session, background, mechanical + optical, values, emit, control, service,
        ('optomechanics', 'an optomechanics study feeds no optical or mechanical mode back into other physics'),
        f"Optomechanics study: optical physics {', '.join(optical)}, mechanical physics {', '.join(mechanical)}; "
        f"background physics {', '.join(background) or 'none'}",
        [_OPTICAL_STAGE, _MECHANICAL_STAGE])
    status, n_found = stage.status, 0
    if status == 'completed':
        _solve_static_states(runner, mechanical, True, emit)
        status = _solve_optical(session, optical, mechanical, values, stage.stages.stage_emit(_OPTICAL_STAGE),
                                control, service)
    if status == 'completed':
        status, n_found = _solve_mechanical(session, mechanical, values, stage.recorder, stage.live,
                                            stage.stages.stage_emit(_MECHANICAL_STAGE), control, service)
    stage.live.update(final=True)
    return {'status': status, 'ticks': n_found, 'tick_unit': 'resonance', 't_end': runner.t,
            'fixed_point': stage.fixed_point, 'n_resonances': n_found}


def _solve_optical(session, optical, mechanical, values, emit, control, service):
    """Mode solve (a ModeSolver: em_modes) or eigen-solve (n_optical_resonances: em_fdfd_ez / _hz, em_fdfd_bor per
    azimuthal order) of every optical physics, then the mode of its first coupling selected for display -> status."""
    runner = session.runner
    wall_start = time.perf_counter()
    for index, name in enumerate(optical):
        if control.checkpoint_cancelled(service):
            return 'cancelled'
        solver = runner.physics[name]
        emit(_count_progress(runner, index / len(optical), index, wall_start,
                             f'Solving the optical modes of {name}; stopping waits for this solve', 'eigen solve'))
        number = next(coupling.optical_mode for solid in mechanical
                      for coupling in runner.physics[solid].optomechanical_couplings
                      if coupling.optical_physics == name)
        if isinstance(solver, ModeSolver):
            count = solver.solve_modes()
            summary = solver.solve_summary()
            if summary:
                emit(LogEvent('info', f'{name}: {summary}'))
            if count:
                solver.select_mode(number)
        else:
            eigen_solve = solver.solve_eigenfrequencies(values['n_optical_resonances'])
            _log_eigen_solve(name, eigen_solve, emit)
            if number <= len(eigen_solve.resonances):
                solver.select_resonance(number)
            else:
                emit(LogEvent('warning', f"physics '{name}': the eigen-solve found {len(eigen_solve.resonances)} "
                                         f'resonance(s), the optical mode {number} of its coupling is missing (raise '
                                         'n_optical_resonances)'))
    emit(_count_progress(runner, 1.0, len(optical), wall_start, f'{len(optical)} optical solve(s) done',
                         'eigen solve'))
    return 'completed'


def _solve_mechanical(session, mechanical, values, recorder, live, emit, control, service):
    """study._numbered_snapshots of the vibration modes: solve_eigenfrequencies, every coupling's g0 table, labels
    'mechanical mode k: f = ... Hz, g0 / 2 pi = ... Hz' (the first coupling) -> (status, largest count)."""
    runner = session.runner

    def solve(name, solver):
        for coupling in solver.optomechanical_couplings:   # backward: k_z = 2 Re beta (validation: one per solid)
            wavenumber = coupling.axial_wavenumber(runner.physics[coupling.optical_physics])
            if wavenumber is not None:
                solver.set_axial_wavenumber(wavenumber)
                emit(LogEvent('info', f"{name}: backward Brillouin scattering ('{coupling.name}'): guided vibration "
                                      f'modes at k_z = 2 Re beta = {wavenumber:.6g} rad/m'))
                break
        eigen_solve = solver.solve_eigenfrequencies(values['n_mechanical_modes'])
        _log_eigen_solve(name, eigen_solve, emit)
        tables = [coupling.evaluate(runner.physics[coupling.optical_physics], solver)
                  for coupling in solver.optomechanical_couplings]
        g0_column = tables[0]['columns'].index('g0')
        return [f'{resonance.text()}, g0 / 2 pi = {row[g0_column]:.4g} Hz'
                for resonance, row in zip(eigen_solve.resonances, tables[0]['rows'])]

    return _numbered_snapshots(session, mechanical, solve, lambda solver, number: solver.select_resonance(number),
                               _reselect_first_resonance, 'mechanical mode', 'eigen solve', recorder, live, emit, control, service)


def _validate_optomechanics(vctx, values):
    """Errors without scipy and without an optomechanical_coupling feature (couplings / materials reading a solved
    physics: the generic background-study rule, every optical type lists this study)."""
    issues = _scipy_issues(STUDY_NAME, 'host shift-invert eigensolvers')
    if not any(feature.enabled and feature.type == _FEATURE
               for physics in vctx.spec.physics for feature in physics.features):
        issues.append(Issue('error', ('study',), f'the {STUDY_NAME} study needs an {_FEATURE} feature on a '
                                                 'solid_mechanics physics (it names the optical physics)'))
    return issues


_PARAMS = (
    Param('n_mechanical_modes', 'int', 6, label='Number of vibration modes', min=1, max=100,
          help='Vibration modes of every solid_mechanics physics with an optomechanical_coupling: the n + 4 '
               'eigenvalues omega^2 nearest (2 pi target_frequency)^2 of its target_frequency setting (0 = the '
               'lowest), ranked by |omega - omega_target|; one g0 row and one snapshot each. 4-20 typical.',
          tip='How many vibration modes are computed, starting closest to the target frequency of the solid.'),
    Param('n_optical_resonances', 'int', 4, label='Number of optical resonances', min=1, max=100,
          help='em_fdfd_ez / _hz / _bor optical physics only: resonances of its eigen-solve near its wavelength setting '
               '(the eigenfrequency study rules: PML resonances dropped, nearest the target first; em_fdfd_bor per '
               'azimuthal order); optical_mode picks one. em_modes solves its n_modes modes.',
          tip='In-plane resonator only: how many light resonances are searched near the wavelength.'),
    *_background_fixed_point_params('the eigen-solves', 'the modes are computed'),
)
_DESCRIPTION = (
    'Vacuum optomechanical coupling g0 between optical and mechanical eigenmodes of one model: every solid_mechanics '
    '(or solid_mechanics_3d) physics with an optomechanical_coupling feature is the mechanical physics, the physics '
    'those features name (em_modes: cross-section modes at its wavelength setting; em_fdfd_ez / _hz: in-plane '
    'resonances near its wavelength setting, n_optical_resonances of them; em_fdfd_bor on a triangles mesh: '
    'body-of-revolution resonances of its azimuthal order(s), paired with an axisymmetric solid on triangles or a 3D '
    'solid on tetrahedra) are the optical physics; every other physics runs as a stationary '
    'background physics first (as in mode_analysis / eigenfrequency: background fixed point, then the background -> '
    'solved couplings once; couplings or material properties reading a solved physics, i.e. any solid_mechanics, '
    'em_modes or em_fdfd_* physics, are a validation error; an FDFD physics no feature names solves as a background '
    'physics). Order: '
    'background, optical solves (the first coupling\'s optical mode is shown), the vibration modes of each solid '
    '(n_mechanical_modes nearest its target_frequency; consistent mass of rho, supports held at 0, prestressed by its '
    'static state only with the stress_stiffening setting, which the tensor-mesh solid of this study does not offer; a '
    'solid whose coupling has scattering backward solves its guided 2.5D modes at k_z = 2 Re beta of that optical '
    'mode, set here after the optical solve), the '
    'g0 table of every coupling, then one snapshot per mechanical mode number labelled "mechanical mode k: f = <Hz> '
    'Hz, g0 / 2 pi = <Hz> Hz" (the first coupling\'s g0) with the stored fields (default each primary field: the mode '
    'shape and the optical intensity; the solid field om_integrand, available in this study only, maps the '
    'moving-boundary g_om density of the shown mode); mode 1 stays shown. ticks = the number of mechanical snapshots, '
    'summary n_resonances. The optical physics keeps its selected mode for every snapshot.')


def _time_counts(values):
    """Run-time estimate counts (StudyType.time_counts): the solid's eigenpairs, the optical eigenpairs (optical recipes
    read 'optical_eigenpairs' first), the background iteration limit."""
    return {'eigenpairs': int(values['n_mechanical_modes']), 'optical_eigenpairs': int(values['n_optical_resonances']),
            'fixed_point_limit': int(values['max_iterations'])}


def register(registry):
    """Register the 'optomechanics' StudyType (background study 'stationary')."""
    registry.register_study(StudyType(
        STUDY_NAME, 'Optomechanics',
        summary='Computes a light mode and the vibration modes of a structure, and how strongly each vibration '
                'shifts the light frequency (vacuum optomechanical coupling rate). Heat or current physics are first '
                'solved to a steady state.',
        description=_DESCRIPTION, params=_PARAMS, background_study='stationary', run=_run_optomechanics,
        validate=_validate_optomechanics,
        feedback_alternative='the optical and mechanical modes are only tabulated, not fed back',
        time_counts=_time_counts, solver_protocols=(ModeSolver, EigenfrequencySolver)))

"""Solver hooks the studies call, as runtime-checkable Protocols; StudyType.solver_protocols names those a solver of a
physics listing the study satisfies (one of them), checked by the builder right after create (check_study_hooks)."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class HarmonicSolver(Protocol):
    """frequency_domain (optimization: with AdjointSolver): one driven solve at omega [rad/s]."""
    def solve_harmonic(self, omega): ...


@runtime_checkable
class PulsedHarmonicSolver(Protocol):
    """frequency_domain beside time stepping (em_fdtd_*): DFT of the pulse at the study frequencies, one shown."""
    def attach_harmonic(self, frequencies): ...
    def select_harmonic_frequency(self, index): ...


@runtime_checkable
class HarmonicFrequencyPlanner(Protocol):
    """Optional on a HarmonicSolver: plan_harmonic_frequencies(omegas) before the wavelength loop, () after."""
    def plan_harmonic_frequencies(self, omegas): ...


@runtime_checkable
class ModeSolver(Protocol):
    """mode_analysis, thermal_noise, optomechanics: solve_modes() -> mode count, then mode k selected or labelled."""
    mode_index: int
    def solve_modes(self): ...
    def select_mode(self, number): ...
    def n_eff_text(self, number): ...


@runtime_checkable
class EigenfrequencySolver(Protocol):
    """eigenfrequency, optomechanics: solve_eigenfrequencies(n) -> EigenSolve, then resonance k selected."""
    def solve_eigenfrequencies(self, n_resonances): ...
    def select_resonance(self, number): ...


@runtime_checkable
class MultiShiftEigenSolver(Protocol):
    """Optional on an EigenfrequencySolver (eigenfrequency search 'targets' / 'band'): the union of the n nearest each
    target [Hz], or every resonance in [f_lo, f_hi] [Hz]; both set eigen_solve (frequency order) like the single solve."""
    def solve_eigenfrequencies_targets(self, n_resonances, target_frequencies): ...
    def solve_eigenfrequencies_band(self, f_lo, f_hi, n_per_slice, max_resonances): ...


@runtime_checkable
class FrequencyResponseSolver(Protocol):
    """frequency_response: reset once, then one driven solve per frequency [Hz]."""
    def reset_frequency_response(self): ...
    def solve_frequency_response(self, frequency): ...


@runtime_checkable
class ReducedSweepSolver(Protocol):
    """Optional on a FrequencyResponseSolver: after reset_frequency_response, plan_frequency_response(frequencies [Hz],
    options) -> info dict ('adaptive' / 'modal'; 'direct' -> None, only options['fields'] stored); solve_frequency_response
    publishes fields only as options['fields'] asks ('all' | 'last'); publish_frequency_response() shows the last solved
    frequency's fields when they were skipped (the study calls it after every sweep)."""
    def plan_frequency_response(self, frequencies, options): ...
    def publish_frequency_response(self): ...


@runtime_checkable
class PrestressSolver(Protocol):
    """Optional on an eigenfrequency / frequency_response / optomechanics solver (stress stiffening or geometric
    nonlinearity): while prestressed,
    the study calls solve_static_state(with_loads) -> log text after its background stage, before its own solves: the
    static state (with_loads: every load, else the initial stress alone) whose K_sigma joins every pencil."""
    prestressed: bool
    def solve_static_state(self, with_loads): ...


@runtime_checkable
class NonlinearSolver(Protocol):
    """nonlinear_frequency_domain: the cascaded solve of every channel, then channels selected by name."""
    channels: list
    def solve_nonlinear(self, before_channel=None, after_sweep=None): ...
    def select_channel(self, name): ...
    def channel_index(self, name): ...
    def mixing_lines(self): ...


@runtime_checkable
class ThermalNoiseSolver(Protocol):
    """thermal_noise (heat): the noise spectra at the study frequencies [Hz] from the mode physics' modes."""
    def solve_thermal_noise(self, physics_by_name, frequencies, progress=None): ...


@runtime_checkable
class AdjointSolver(Protocol):
    """optimization: design regions and dJ/d eps of the solved state (one transposed solve)."""
    design_regions: list
    def eps_sensitivity(self): ...


@runtime_checkable
class HarmonicAdjointSolver(HarmonicSolver, AdjointSolver, Protocol):
    """optimization on a solver solved at its wavelength setting (em_fdfd_ez / _hz, em_eme)."""


@runtime_checkable
class DesignAdjointSolver(AdjointSolver, Protocol):
    """optimization on a solver solving its own frequencies (em_fdfd_nonlinear)."""
    def solve_design(self): ...


@runtime_checkable
class OptomechanicalSolver(Protocol):
    """optomechanics (solid_mechanics): the optomechanical_coupling features and the backward-scattering k_z."""
    optomechanical_couplings: list
    def set_axial_wavenumber(self, wavenumber): ...


@runtime_checkable
class WarmStartSolver(Protocol):
    """Optional (study warm_start): host fields of the solve, written into the next sweep value's build."""
    def warm_start_fields(self): ...
    def seed_warm_start(self, fields): ...


@runtime_checkable
class RerunSolver(Protocol):
    """Optional (SweepSpec.reuse_build): reset_records() before each rerun of the reused build drops the run records
    (probe series, kept static-state products) and marks the loads dirty; a model with a physics lacking it is rebuilt
    per sweep value."""
    def reset_records(self): ...


@runtime_checkable
class ErrorIndicatorSolver(Protocol):
    """Optional (StudySpec.adaptation): the solver's own per-cell / per-element refinement indicator."""
    def error_indicator(self): ...


def hook_names(protocol):
    """Member names of a hook Protocol (methods and annotated attributes, its Protocol bases included)."""
    names = []
    for klass in reversed(protocol.__mro__):
        if klass is Protocol or not vars(klass).get('_is_protocol', False):
            continue
        for name in (*vars(klass).get('__annotations__', {}), *vars(klass)):
            if not name.startswith('_') and name not in names:
                names.append(name)
    return tuple(names)


def check_study_hooks(physics_name, solver, study_type):
    """TypeError naming the missing hooks when `solver` satisfies none of study_type.solver_protocols."""
    protocols = study_type.solver_protocols
    if not protocols or any(isinstance(solver, protocol) for protocol in protocols):
        return
    missing = '; or '.join(f"{', '.join(name for name in hook_names(protocol) if not hasattr(solver, name))} "
                           f'({protocol.__name__})' for protocol in protocols)
    raise TypeError(f"physics '{physics_name}' lists the {study_type.name} study, but its solver "
                    f'{type(solver).__name__} has no {missing}')

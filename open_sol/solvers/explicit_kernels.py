"""'explicit_kernels' solver type runtime (role 'stepper'): the settings bundle of an explicit time-stepping engine
(solver_plan S3.1); the engine stays the loop. single = float32 fields; double = float64 fields and arithmetic
(em_fdtd_ez / em_fdtd_hz); half_storage = fp16 fields in memory, float32 arithmetic in registers (em_fdtd_3d,
elastic_waves, laminar_flow transient; core.half_storage). On the cpu backend the engines run the core.kernels twins of
their kernels (single and double: backend_misfit)."""
from dataclasses import dataclass

PRECISIONS = ('single', 'double', 'half_storage')   # the one list (SolverType precisions and the precision choices)


@dataclass(frozen=True)
class StepperConfig:
    """Precision of an explicit engine (the engine picks its dtypes from it)."""
    precision: str = 'single'

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"stepper precision must be one of {', '.join(PRECISIONS)}, got {self.precision!r}")

    def work_counters(self) -> dict:
        """Nothing: the engine counts its own steps (summary ticks)."""
        return {}


def time_estimate(values, work):
    """Nothing to price: the engine's own recipe lists its kernels per step."""
    from ..core.time_estimate import TimeEstimate
    return TimeEstimate()


def backend_misfit(values, backend) -> str:
    """SolverType.backend_misfit of explicit_kernels: '' unless half_storage is asked of the cpu backend (its fp16 storage
    is PTX code; the core.kernels numpy / numba twins follow float32 / float64 arrays)."""
    if backend == 'cpu' and values.get('precision') == 'half_storage':
        return "precision 'half_storage' runs on the GPU only (use 'single' on the CPU backend)"
    return ''


def stepper_precision(solvers) -> str:
    """Precision of the build's 'stepper' runtime in ctx.solvers(name) ({role: runtime}); KeyError without one (every
    type with a 'stepper' role gets it in every study, so a missing one is a build bug, never a silent 'single')."""
    if 'stepper' not in solvers:
        raise KeyError(f"no 'stepper' solver runtime among the built roles {sorted(solvers)}")
    return solvers['stepper'].precision

"""Shared part list of the batched_direct solver (solvers.batched_direct) and its fem.harmonic consumer (plan_batched):
the per-frequency batch hand-out, combine / equilibrate / factorise / refine on the device, the batch plans (cuDSS
uniform batch, nvmath explicit batch) and the per-system fallback. Only batched runs call them (--solver-type
batched_direct), so they cost the other solvers' profiles nothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import functions, methods  # noqa: E402

BATCHED = 'open_sol.solvers.batched_direct'
CUDSS = 'open_sol.solvers.cudss_factor'
SYSTEM = 'open_sol.fem.harmonic:HarmonicSystemBase'
# _device_systems' exclusive time = the |diag|^-1/2 equilibration, the lower-triangle take + factor-dtype cast and the
# refinement loop's host reads (its _corrections / _products steps are rows of their own); _combined = the terms GEMM +
# extras (port DtN blocks); solve_batch recurses on the two halves of a batch that did not fit (inclusive: outermost)
BATCHED_DIRECT_PARTS = methods(SYSTEM, (
    'plan_batched', '_batched_free', '_planned_free', '_batched_system', '_planned_held', '_batch_pattern',
    '_block_slots', '_reference_free',
), 'batched sweep') + methods(f'{BATCHED}:BatchedDirectSolver', (
    'pattern', 'batch_capacity', 'solve_batch', '_device_batch', '_device_systems', '_host_batch', '_combined',
    '_corrections', '_products', '_planned', 'release',
), 'batched_direct') + methods(f'{BATCHED}:BatchPattern', (
    '__init__',
), 'batched_direct pattern') + methods(f'{BATCHED}:_BatchPlan', (
    '__init__', '_create', '_factorize', '_solve', 'factorize', 'solve', 'free',
), 'batched plan') + methods(f'{BATCHED}:_UniformPlan', (
    '_create', '_solve',
), 'batched plan uniform') + methods(f'{CUDSS}:UniformBatch', (
    'analyse', 'factorize', 'solve', 'memory_estimates', 'free',
), 'cuDSS uniform batch') + functions(CUDSS, (
    'batched_direct_solver',
), 'cuDSS explicit batch')

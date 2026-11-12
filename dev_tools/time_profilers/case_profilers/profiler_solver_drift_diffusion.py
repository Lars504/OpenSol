"""Profile drift_diffusion on the gpu backend: the host (numpy float64) Newton / continuation of DriftDiffusionCore
(DriftDiffusionSystem on tensor meshes, FEMDriftDiffusionSystem box method on triangles), its linear role through
sparse_direct (cuDSS via factorize_host on the gpu). Kinds: 'tensor' (default) the Automatic runs of
semiconductor_cudss (silicon_pn_photodiode template 400 x 20 cells + the forward-biased pn junction strip; ~3.7 s
each); 'triangles' semiconductor_fem_photocurrent_balance refined 2x, repeated 2x (first estimate ~10 s). Run from any
folder:
python profiler_solver_drift_diffusion.py [--kind tensor|triangles] [--refine R]
[--repeat N] [--no-cprofile]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import COMMON_PARTS, functions, methods, run_kind  # noqa: E402

SEMICONDUCTOR = 'open_sol.physics.semiconductor'
CORE = f'{SEMICONDUCTOR}.drift_diffusion:DriftDiffusionCore'
PARTS = (COMMON_PARTS + methods(f'{SEMICONDUCTOR}.solver:DriftDiffusionPhysics', (
    'prepare', 'finalize', 'step', '_solve', 'scalars', 'contact_currents', 'gate_charge', '_host_window_values',
    '_device_window_mask',
), 'drift_diffusion tensor') + methods(f'{SEMICONDUCTOR}.solver_fem:FEMDriftDiffusionPhysics', (
    'prepare', 'finalize', '_solve', 'fem_field', '_device_window_mask',
), 'drift_diffusion triangles') + methods(CORE, (
    'solve', '_finish', '_scaled_state', '_targets', '_continue', '_predicted_state', '_equilibrium_state',
    '_poisson_system', '_newton', '_coupled_system', '_layout_values', '_recombination', '_linear_solve',
    '_chord_solve', '_factor_solve', '_face_fluxes', '_quasi_fermi', 'contact_currents', 'gate_charge',
), 'DriftDiffusionCore') + methods(f'{SEMICONDUCTOR}.drift_diffusion:_SiteGraph', (
    '__init__',
), '_SiteGraph') + methods(f'{SEMICONDUCTOR}.drift_diffusion:DriftDiffusionSystem', (
    '_build_faces', 'set_inputs', 'outputs', '_face_average',
), 'DriftDiffusionSystem') + methods(f'{SEMICONDUCTOR}.drift_diffusion_fem:FEMDriftDiffusionSystem', (
    'set_inputs', '_resolve_held', 'outputs',
), 'FEMDriftDiffusionSystem') + methods(f'{SEMICONDUCTOR}.drift_diffusion:_PinnedPattern', (
    '__init__', 'csr_values', 'host_csr',
), '_PinnedPattern') + functions(f'{SEMICONDUCTOR}.drift_diffusion', (
    'neutral_equilibrium',
), 'drift_diffusion') + (
    (f'{SEMICONDUCTOR}.renderer:SemiconductorRenderer.device_field', 'renderer: SemiconductorRenderer.device_field'),
))
KINDS = {
    'tensor': {'case': 'semiconductor_cudss', 'runs': ['photodiode_automatic', 'forward_junction_automatic']},
    'triangles': {'case': 'semiconductor_fem_photocurrent_balance', 'refine': 2.0, 'repeat': 2},
}

if __name__ == '__main__':
    run_kind('drift_diffusion', PARTS, __doc__, KINDS, 'tensor')

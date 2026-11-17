"""Profile long direct frequency sweeps on element meshes (gpu): the cases batched_direct batches that the other commands
do not hold (as written: one cuDSS factor per frequency; --solver-type batched_direct: batches of up to 32 on one
plan). Kinds: 'triangles' (default) the membrane two-port guide of membrane_guide_ports_batched ('direct' run, h = W /
32, 64 frequencies: two full batches, the port DtN blocks as batch extras) and the clamped rectangle window of
membrane_rectangle_reduced_sweep ('direct', 201 frequencies through eight resonances: seven batches); 'tetrahedra' the
impedance and PML ducts of pressure_acoustics_duct_3d (P2 Kuhn tetrahedra, dx 10 mm, ~1.2e4 dofs) at 64 frequencies
over 300 - 900 Hz, snapshots none. Long acoustic sweeps on triangles: profiler_solver_pressure_acoustics.py. Run from
any folder:
python profiler_solver_frequency_sweeps.py [--kind triangles|tetrahedra] [--repeat N] [--no-cprofile]
[--solver-type batched_direct]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler_diffusion_parts import methods, pop_option  # noqa: E402
from profiler_harness import case_specs, main, refined  # noqa: E402
from profiler_solver_membrane_plate import PARTS as MEMBRANE_PARTS  # noqa: E402  (+ BATCHED_DIRECT_PARTS)
from profiler_solver_pressure_acoustics_tetrahedra import PARTS as TETRAHEDRA_PARTS  # noqa: E402  (+ batched)

GUIDE_REFINE = 2.0            # h = W / 32 (selftest: W / 16)
GUIDE_FREQUENCIES = 64        # selftest: 6
DUCT_ELEMENT = 0.01           # m (selftest: 0.025)
DUCT_HZ = (300.0, 900.0, 64)  # impedance end: no resonance; 38 P2 elements per wavelength at 900 Hz
DUCT_RUNS = ('impedance', 'pml')

MEMBRANE_PORT = 'open_sol.physics.membrane.ports:MembranePort'
KIND_PARTS = {
    'triangles': MEMBRANE_PARTS + methods(MEMBRANE_PORT, (
        'assemble', 'set_medium', 'propagation_constants', 'terms', 'outputs',
    ), 'membrane ports'),
    'tetrahedra': TETRAHEDRA_PARTS,
}


def triangle_specs():
    """{run: spec}: the two-port guide refined with GUIDE_FREQUENCIES frequencies, the rectangle window as written."""
    guide = refined(case_specs('membrane_guide_ports_batched', ['direct'])['direct'], GUIDE_REFINE)
    guide.study.settings['n_frequencies'] = GUIDE_FREQUENCIES
    return {'guide_ports': guide,
            'rectangle_window': case_specs('membrane_rectangle_reduced_sweep', ['direct'])['direct']}


def tetrahedra_specs():
    """{run: spec}: the impedance and PML ducts at DUCT_ELEMENT over DUCT_HZ, no snapshots (the solves, not the cuts)."""
    start, stop, count = DUCT_HZ
    frequencies = [start + (stop - start) * index / (count - 1) for index in range(count)]
    specs = {}
    for name, spec in case_specs('pressure_acoustics_duct_3d', list(DUCT_RUNS)).items():
        for mesh in spec.meshes:
            mesh.dx = DUCT_ELEMENT
        spec.study.settings = {'frequencies': list(frequencies), 'snapshots': 'none'}
        specs[f'duct_{name}'] = spec
    return specs


KIND_SPECS = {'triangles': triangle_specs, 'tetrahedra': tetrahedra_specs}

if __name__ == '__main__':
    kind = pop_option('kind', 'triangles', str)
    if kind not in KIND_SPECS:
        raise SystemExit(f"--kind must be one of {', '.join(KIND_SPECS)}, got {kind!r}")
    main(f'frequency_sweeps_{kind}', KIND_PARTS[kind], __doc__, specs=KIND_SPECS[kind], note=f'kind {kind}')

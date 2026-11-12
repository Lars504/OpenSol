"""Profile the analytic examples chosen to stand for each physics module (examples/<folder>/<stem>.json, the models as
the example selftest runs them), one group per command: model sizes made for the physics, so the time goes to the
physics, the geometry and the solve, not to the selftest harness. Output: out/<group>__<backend>[...]/ here.
Run from any folder: python example_profiler.py GROUP [--runs STEM ...] [--warmup] [--repeat N] [--no-cprofile]
[--compare NAME=v1,v2 [--compare-only]] [--backend cpu] [--set NAME=EXPR]; python example_profiler.py --list"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE_PROFILERS = HERE.parent / 'case_profilers'
sys.path.insert(0, str(CASE_PROFILERS))
import profiler_harness  # noqa: E402
from profiler_diffusion_parts import STUDY_LOOP_PARTS, TENSOR_DIFFUSION_PARTS  # noqa: E402
from profiler_em2d_parts import (COUPLINGS, EIGEN, FDFD_TENSOR, MODES_TENSOR, NLFD_PARTS, STUDY_RUNNERS,  # noqa: E402
                                 TENSOR_READS)
from profiler_em_bor_fdtd_common import (BOR_OPERATOR_PARTS, BOR_TENSOR_PARTS, COMMON_PARTS,  # noqa: E402
                                         EM_PHYSICS_PARTS)
from profiler_example_sin_square_membrane import PARTS as SIN_SQUARE_PARTS  # noqa: E402
from profiler_harness import REPOSITORY, main  # noqa: E402
from profiler_solver_em_fdfd_3d import PARTS as FDFD_3D_PARTS  # noqa: E402
from profiler_solver_em_fdfd_3d_tetrahedra import PARTS as FDFD_3D_TETRAHEDRA_PARTS  # noqa: E402
from profiler_solver_em_fdfd_3d_tetrahedra import load_example as load_gmsh_sphere  # noqa: E402

EXAMPLES = REPOSITORY / 'examples'
OUT_ROOT = HERE / 'out'
# study hooks timed on each engine class that defines them (an open_sol base counts); see engine_parts
ENGINE_HOOKS = (
    'prepare', 'finalize', 'step', 'materials_changed', 'solve_harmonic', 'attach_harmonic',
    'select_harmonic_frequency', 'plan_harmonic_frequencies', 'solve_modes', 'select_mode', 'solve_eigenfrequencies',
    'select_resonance', 'solve_eigenfrequencies_targets', 'solve_eigenfrequencies_band', 'reset_frequency_response',
    'plan_frequency_response', 'solve_frequency_response', 'publish_frequency_response', 'solve_static_state',
    'solve_nonlinear', 'solve_thermal_noise', 'eps_sensitivity', 'solve_design', 'error_indicator',
)
# parts every group times: study / runner / output / Session reads and the solver-role runtimes
GROUP_COMMON_PARTS = COMMON_PARTS + STUDY_LOOP_PARTS + STUDY_RUNNERS

PHYSICS = 'open_sol.physics'
EM_FDTD = (f'{PHYSICS}.em:EMPhysics', True)
FDFD = (f'{PHYSICS}.em.solvers.fdfd_2d:FDFDPhysics', False)
MODES = (f'{PHYSICS}.em.solvers.modes_2d:ModePhysics', False)
HEAT = (f'{PHYSICS}.heat.solver:HeatPhysics', False)
POTENTIAL = (f'{PHYSICS}.potential.solver:PotentialPhysics', False)
DRIFT_DIFFUSION = (f'{PHYSICS}.semiconductor.solver:DriftDiffusionPhysics', False)


def _group(examples, engines=(), parts=(), loader=None, note=''):
    """One command: examples ((stem, selftest wall s at selection), ...), engines (('module:Class', explicit
    stepper), ...), extra parts, loader(path) -> ModelSpec (default api.load)."""
    return {'examples': examples, 'engines': engines, 'parts': parts, 'loader': loader, 'note': note}


# selftest wall seconds: selftest_report_analytic.json of 2026-11-12, RTX 3050, each example on its own backend
GROUPS = {
    'em_fdtd': _group((('em/analytic_pec_cavity', 10.2), ('em/analytic_lorentz_halfspace', 1.4)),
                      (EM_FDTD,), EM_PHYSICS_PARTS, note='E_z and H_z ring-down, Lorentz pole half-space'),
    'em_fdfd': _group((('em/analytic_fdfd_line_source_green_ez', 1.6), ('em/analytic_fdfd_fabry_perot_slab', 1.3),
                       ('em/analytic_fdfd_eigen_dbr_cavity_q', 1.9)),
                      (FDFD,), FDFD_TENSOR + EIGEN + TENSOR_READS,
                      note='driven mesh sweep, wavelength scan, high-Q eigenfrequency (selftest: cpu SuperLU)'),
    'em_modes': _group((('em/analytic_modes_pec_rectangular_waveguide', 3.7),), (MODES,),
                       MODES_TENSOR + TENSOR_READS, note='em_modes frequency_domain scan'),
    'em_fdfd_bor': _group((('em/analytic_bor_pipe_te11_section', 2.5),),
                          ((f'{PHYSICS}.em.solvers.fdfd_bor:BORPhysics', False),),
                          BOR_TENSOR_PARTS + BOR_OPERATOR_PARTS, note='m = 1 mode ports, frequency_domain'),
    'em_eme': _group((('em/analytic_eme_bragg_periodic_block', 2.7),),
                     ((f'{PHYSICS}.eme.solver:EMEPhysics', False),), note='periodic block, length sweep'),
    'em_nlfd': _group((('em/analytic_nlfd_shg_depleted_tanh', 1.3), ('em/analytic_nlfd_thg_dispersive_slab', 1.5)),
                      ((f'{PHYSICS}.nlfd.solver:NonlinearHarmonicPhysics', False), FDFD),
                      NLFD_PARTS + FDFD_TENSOR + TENSOR_READS, note='pump-depleted Newton SHG, chi3 THG'),
    'em3d_fdtd': _group((('em3d/tfsf_sphere_mie_3d', 6.9), ('em3d/pec_waveguide_te10_ports_3d', 2.4)),
                        ((f'{PHYSICS}.em3d.physics3d:EMPhysics3D', True),), note='TFSF sphere, A5 mode ports'),
    'em3d_fdfd': _group((('em3d/fdfd_mie_sphere_resonance_3d', 8.6),), parts=FDFD_3D_PARTS,
                        note='Yee eigenfrequency, cudss_mixed eigen node (profiler_solver_em_fdfd_3d parts)'),
    'em3d_fdfd_tetrahedra': _group((('em3d/fem_mie_sphere_resonance_3d', 5.6),), parts=FDFD_3D_TETRAHEDRA_PARTS,
                                   loader=lambda path: load_gmsh_sphere(),
                                   note='Nedelec 2 on gmsh tetrahedra (builtin mesher without gmsh)'),
    'heat': _group((('heat/multigrid_square_plate_poisson', 3.9), ('heat/cube_cooling_3d', 3.4)), (HEAT,),
                   TENSOR_DIFFUSION_PARTS, note='multigrid mesh sweep to 6.4e5 cells, 3D transient'),
    'potential': _group((('potential/dielectric_cylinder_uniform_field', 2.7),
                         ('potential/uniform_injection_square_plate', 0.5)), (POTENTIAL,), TENSOR_DIFFUSION_PARTS,
                        note='electrostatics sweep over eps, electric_currents mesh sweep'),
    'mechanics': _group((('mechanics/kirsch_plate_hole', 4.8),
                         ('mechanics/yz_ln_thickness_resonator_admittance', 2.3)),
                        ((f'{PHYSICS}.mechanics.solver:SolidMechanicsPhysics', False),
                         (f'{PHYSICS}.mechanics.solver_fem:SolidMechanicsFEMPhysics', False)),
                        note='2D stationary, piezoelectric frequency_response'),
    'mechanics_3d': _group((('mechanics/sin_square_membrane_modes_3d', 21.2),
                            ('mechanics/thermal_inclusion_sphere_3d', 0.3)),
                           ((f'{PHYSICS}.mechanics.solver3d:SolidMechanics3DPhysics', False),), SIN_SQUARE_PARTS,
                           note='stress_stiffening eigenfrequency on swept prism tetrahedra, 3D tensor stationary'),
    'acoustics': _group((('acoustics/plate_thickness_resonances', 3.1),),
                        ((f'{PHYSICS}.acoustics.solver:AcousticPhysics', True),),
                        note='elastic_waves, resonance finder and lock-in'),
    'fluid': _group((('fluid/parallel_plate_uniform_flux_nusselt', 1.7), ('fluid/cylinder_array_permeability', 1.0)),
                    ((f'{PHYSICS}.fluid.solver:FluidPhysics', True), (HEAT[0], True)), COUPLINGS,
                    note='transient flow + heat, interior obstacles steady march'),
    'magnetic': _group((('magnetic/bh_curve_table_core', 2.3), ('magnetic/strip_skin_effect_impedance', 0.7)),
                       ((f'{PHYSICS}.magnetic.solver:MagneticPhysics', False),
                        (f'{PHYSICS}.magnetic.solver_ac:MagneticACPhysics', False)), TENSOR_DIFFUSION_PARTS,
                       note='saturation loop on a B-H table, AC skin effect'),
    'membrane': _group((('membrane/analytic_membrane_lossy_guide_mode_power', 0.9),),
                       ((f'{PHYSICS}.membrane.solver_fem:MembraneFEMPhysics', False),),
                       note='triangles, ports, frequency_response'),
    'semiconductor': _group((('semiconductor/shockley_diode_iv', 2.3), ('semiconductor/pin_photodiode_fdfd', 3.8)),
                            (DRIFT_DIFFUSION, FDFD), COUPLINGS + FDFD_TENSOR,
                            note='forward-bias sweep, FDFD generation into drift_diffusion'),
    'multiphysics': _group((('multiphysics/analytic_electrothermal_strip', 1.5),
                            ('multiphysics/analytic_modes_carriers_thermo_optic_sum', 2.2)),
                           (POTENTIAL, HEAT, DRIFT_DIFFUSION, MODES), COUPLINGS + TENSOR_DIFFUSION_PARTS + MODES_TENSOR,
                           note='sigma(T) Joule fixed point, carriers + heat into em_modes'),
}


def engine_parts(target, explicit=False) -> tuple:
    """(target, label) parts of the ENGINE_HOOKS the engine class 'module:Class' (or an open_sol base) defines; an
    explicit stepper leaves out step (one GPU sync per time step: the runner's advance times the stepping)."""
    module_name, _, class_name = target.partition(':')
    try:
        engine = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return ((f'{target}.prepare', f'{class_name}: prepare'),)   # the harness lists it as unresolved
    defined = {name for klass in engine.__mro__ if klass.__module__.startswith('open_sol') for name in vars(klass)}
    return tuple((f'{target}.{name}', f'{class_name}: {name}') for name in ENGINE_HOOKS
                 if name in defined and not (explicit and name == 'step'))


def unique_parts(*part_lists) -> tuple:
    """The parts of every list in order, each target once (a second wrap of a target is reported 'already timed')."""
    seen, parts = set(), []
    for part in (part for part_list in part_lists for part in part_list):
        target = part[0] if isinstance(part, tuple) else part
        if target not in seen:
            seen.add(target)
            parts.append(part)
    return tuple(parts)


def group_parts(group) -> tuple:
    settings = GROUPS[group]
    return unique_parts(GROUP_COMMON_PARTS, settings['parts'],
                        *(engine_parts(target, explicit) for target, explicit in settings['engines']))


def group_specs(group) -> dict:
    """{example stem: ModelSpec} of one group, loaded as the example selftest loads them."""
    from open_sol import api
    settings = GROUPS[group]
    loader = settings['loader'] or api.load
    return {Path(stem).name: loader(EXAMPLES / f'{stem}.json') for stem, _ in settings['examples']}


def group_listing() -> str:
    lines = []
    for group, settings in GROUPS.items():
        total = sum(seconds for _, seconds in settings['examples'])
        lines.append(f'{group} (~{total:.1f} s selftest wall): {settings["note"]}')
        lines += [f'    {stem}.json  {seconds:.1f} s' for stem, seconds in settings['examples']]
    return '\n'.join(lines)


def cli():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', '--list'):
        print(f'{__doc__}\n\n{group_listing()}')
        return None
    group = sys.argv.pop(1)
    if group not in GROUPS:
        raise SystemExit(f"unknown group {group!r}; groups: {', '.join(GROUPS)}")
    profiler_harness.OUT_ROOT = OUT_ROOT   # read by profile_runs at call time
    stems = ', '.join(f'{stem} ({seconds:g} s)' for stem, seconds in GROUPS[group]['examples'])
    return main(group, group_parts(group), __doc__, specs=lambda: group_specs(group),
                note=f'examples {stems} (selftest wall); {GROUPS[group]["note"]}')


if __name__ == '__main__':
    cli()

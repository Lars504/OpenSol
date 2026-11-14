"""SolverType definitions ('pcg', 'line_block', 'sparse_direct', 'batched_direct', 'banded', 'arpack_shift_invert',
'block_eigen', 'time_filter', 'explicit_kernels') and register(registry). Imports no cupy at module level; get_registry() imports it after
model.study, before the physics definitions (solver_plan S3.5)."""
from ..model.schema import Issue, Param, solver_backend
from . import (arpack_shift_invert, banded, batched_direct, block_eigen, explicit_kernels, host_direct, line_block,
               multigrid, pcg, sparse_direct, time_filter)
from .roles import SolverType

_TOLERANCE_FLOOR = pcg.FLOAT32_TOLERANCE_FLOOR

# Host direct back ends (solvers.host_direct: MUMPS / MKL PARDISO): every list and text below derives from
# HOST_DIRECT_BACK_ENDS, so dropping a value there (e.g. a wrapper without single precision) needs no edit here
_HOST_DIRECT_NAMES = ' / '.join(host_direct.HOST_DIRECT_BACK_ENDS)
_HOST_DIRECT_DOUBLE_NAMES = ' / '.join(host_direct.HOST_DIRECT_DOUBLE_BACK_ENDS)
_HOST_DIRECT_MIXED_NAMES = ' / '.join(host_direct.HOST_DIRECT_MIXED_BACK_ENDS)
_HOST_MEMORY_PERCENT = round(100 * host_direct.HOST_FRACTION)
_GUI_LIBRARY_NAMES = {'pardiso': 'Intel PARDISO'}   # drop-down / tip wording; others: host_direct.LIBRARY_LABELS
_HOST_DIRECT_INSTALL = '; '.join(f'{host_direct.LIBRARY_LABELS[library]}: {host_direct.install_hint(library)}'
                                 for library in host_direct.HOST_DIRECT_LIBRARIES)


def _gui_library_name(name) -> str:
    library = host_direct.library_of(name)
    return _GUI_LIBRARY_NAMES.get(library, host_direct.LIBRARY_LABELS[library])


def _host_direct_choice_labels(names) -> dict:
    """{name: drop-down text} of host direct back ends / factorizations."""
    return {name: f'{_gui_library_name(name)} on the processor'
                  + (', mixed precision' if host_direct.host_direct_precision(name) == 'mixed' else '')
                  + (', compressed (block low-rank)' if host_direct.host_direct_compressed(name) else '')
            for name in names}


_GUI_HOST_LIBRARIES = ' / '.join(dict.fromkeys(_gui_library_name(name)
                                               for name in host_direct.HOST_DIRECT_DOUBLE_BACK_ENDS))
_GUI_HOST_MIXED_LIBRARIES = ' / '.join(_gui_library_name(name) for name in host_direct.HOST_DIRECT_MIXED_BACK_ENDS)
_HOST_DIRECT_MIXED_HELP = (
    f' {_HOST_DIRECT_MIXED_NAMES}: the equilibrated operator factorised in single precision (float32 / complex64, '
    'about half the factor memory), every solve refined by float64 GMRES to 1e-12 as cudss_mixed; a first factor '
    'whose probe misses is refactorised in double for the rest of the run.'
    if host_direct.HOST_DIRECT_MIXED_BACK_ENDS else '')
_HOST_DIRECT_MIXED_TIP = (
    f'\n- {_GUI_HOST_MIXED_LIBRARIES} on the processor, mixed precision: faster to set up and less memory, with the '
    'same accuracy; switches to double precision by itself when a model needs it.'
    if host_direct.HOST_DIRECT_MIXED_BACK_ENDS else '')

_HOST_DIRECT_COMPRESSED_HELP = (
    f" {' / '.join(host_direct.HOST_DIRECT_COMPRESSED_BACK_ENDS)}: MUMPS's block low-rank factor (ICNTL(35) "
    f'{host_direct.BLR_ICNTL}, dropping threshold CNTL(7) {host_direct.BLR_TOLERANCE:g}), double precision: the '
    'off-diagonal blocks of the large fronts stored compressed, roughly half the factor memory and work on large 3D '
    'models (little gain in 2D); every solve is refined by float64 GMRES on the compressed factor to 1e-12 (about 2 - 4 '
    'compressed solves per solve), and a factor that still misses is refactorised full rank for the rest of the run. '
    'The memory guard reads MUMPS\'s full-rank estimate.'
    if host_direct.HOST_DIRECT_COMPRESSED_BACK_ENDS else '')
_HOST_DIRECT_COMPRESSED_TIP = (
    '\n- MUMPS on the processor, compressed (block low-rank): about half the memory of the exact MUMPS factor on large '
    '3D models, with the same final accuracy (each solve is polished); slower solves.'
    if host_direct.HOST_DIRECT_COMPRESSED_BACK_ENDS else '')

PCG_PARAMS = (
    Param('precision', 'choice', 'double', label='Solver precision', choices=('double', 'single', 'mixed'),
          choice_labels=(('double', 'Double precision'), ('single', 'Single precision'),
                         ('mixed', 'Mixed precision')),
          help=f'Arithmetic of the CG vectors and operator (the physics stores its fields as float32 either way). '
               f'single saves solver memory but CG stalls near a relative residual of {_TOLERANCE_FLOOR:g}: use '
               f'tolerance >= {_TOLERANCE_FLOOR:g} with it (validation warns). mixed (heat, electric_currents, '
               'electrostatics, magnetostatics: Jacobi on the assembled operator): the system is assembled in '
               'float64, CG runs in float32 (vectors and a float32 copy of the operator values, ~1.5-2x '
               'faster per iteration) and each time its residual fell by '
               f'{pcg.MIXED_REPLACEMENT_RATIO:g} the float64 residual replaces it (the search direction is kept, '
               'no restart), so any tolerance double reaches is reached in about the iterations of double '
               '(+~30 B per cell for the float32 copies). '
               'solid_mechanics_3d: float32 copies of the element Lame pair and the block-Jacobi inverses drive a '
               'float32 matrix-free product, which on GPUs with slow float64 arithmetic (most consumer cards) is '
               'several times cheaper than the float64 one (the stiffness product is compute bound). Not for 2D '
               'solid_mechanics (block preconditioner on the assembled operator). A node that sets no precision '
               'takes solver_defaults.precision, else the physics\' Automatic one (solid_mechanics_3d on the gpu '
               'backend with a tensor mesh: mixed), else double.',
          tip='Number precision of the iterative solver.\n\n'
              '- Double precision: suits almost every model.\n'
              "- Single precision: saves memory, but cannot reach a 'Solver tolerance' below about "
              f'{_TOLERANCE_FLOOR:g}.\n'
              '- Mixed precision: as accurate as double precision but faster on large heat, electric and 3D solid '
              'mechanics models; uses a little more memory.'),
    Param('tolerance', 'float', 1e-8, label='Solver tolerance', unit='1', min=0.0, min_exclusive=True,
          help='CG stopping criterion, a relative residual norm: in a steady solve relative to ||b|| (the sources and '
               'boundary drive); in an implicit time step relative to the initial residual of the step (its change '
               f'from the previous step, never below ~{pcg.ROUNDOFF_FLOOR_EPSILONS} machine epsilons of ||b||). 1e-8 '
               f'suits double; single cannot get below ~{_TOLERANCE_FLOOR:g}. Outputs that follow the solution '
               'linearly (terminal currents and charges, q_joule, reactions, stresses, B and H) carry a relative error '
               'of the order of the condition number times the tolerance: use 1e-12 for exact-identity checks; an iron '
               'part (mu_dc >> 1) raises the magnetostatics condition number by about its permeability, so use 1e-10 '
               'or below for flux-concentration ratios.',
          tip='How accurately each solve is converged, as a relative error.\n\n'
              f'Keep 1e-8. With single precision use {_TOLERANCE_FLOOR:g} or larger.'),
    Param('max_iterations', 'int', 0, label='Solver iteration limit', min=0,
          help=f'CG iteration limit per solve; 0 = auto: max({pcg.AUTO_CG_MAX_ITERATIONS_FLOOR}, '
               f'{2 * pcg.CG_ITERATIONS_PER_AXIS_CELL} x the cells along the longest window axis), twice the '
               f'~{pcg.CG_ITERATIONS_PER_AXIS_CELL} x those cells Jacobi-PCG needs from a cold start (fewer when '
               'warm-started from the previous solve); solid_mechanics doubles the auto value (block-Jacobi PCG on '
               'the elastic operator needs about twice the iterations, more for poisson_ratio near 0.5). A solve '
               'that hits the limit keeps its unconverged (biased) result, and the first one per run is logged as a '
               f'warning; validation warns when an explicit value is below {pcg.CG_ITERATIONS_PER_AXIS_CELL} x the '
               'longest window axis (not with a heat out_of_plane_convection on every cell, which converges much '
               'faster).',
          tip='Largest number of iterative solver steps per solve; 0 picks a safe limit from the grid size.\n\n'
              'A solve that reaches the limit keeps its unfinished result and logs a warning.'),
    Param('preconditioner', 'choice', 'jacobi', label='Preconditioner', choices=pcg.PRECONDITIONERS,
          choice_labels=(('jacobi', 'Jacobi (diagonal)'), ('multigrid', 'Multigrid')),
          help='CG preconditioner. jacobi: the inverse diagonal (solid_mechanics: its 2x2 / 3x3 block diagonal), '
               'cheap per iteration, but the iterations grow with the cells along the longest window axis (~10 per '
               'cell) and with mesh grading, cell aspect ratio and coefficient contrast (outer grading to large '
               'cells, 1e3x conductivity ratios: up to ~1e5 iterations). multigrid (heat, electric_currents, '
               'electrostatics, magnetostatics; precision single or double): a smoothed-aggregation algebraic '
               'multigrid V(1,1)-cycle (l1-Jacobi smoothing) built from the assembled operator, coarsening along '
               f'strong couplings (threshold {multigrid.STRENGTH_THRESHOLD:g}), so graded and stretched cells '
               'semi-coarsen; typically 10-40 iterations whatever the mesh size, each costing ~4-6 Jacobi '
               'iterations; the hierarchy takes ~150 B per cell (double) on the device, every assembly rebuilds its '
               'coarse operators and a change of the sparsity (held cells, empty rows) re-runs the aggregation, one '
               'host pass over the operator. Best for steady solves on large or strongly graded meshes; for implicit '
               'time steps the warm-started Jacobi is often cheaper. Automatic (no node) of heat, electric_currents, '
               'electrostatics and magnetostatics on the gpu backend is pcg with multigrid (jacobi when '
               'solver_defaults.precision is mixed); a pcg node without this setting uses jacobi. '
               'solid_mechanics_3d on a tensor mesh (any precision, mixed included): a geometric multigrid '
               'V(2,2)-cycle on the brick operator itself (cells paired 2:1 per axis, volume-mean Lame values, '
               'trilinear transfers, Chebyshev block-Jacobi smoothing, a small dense coarsest level), each iteration '
               'costing ~6-8 block-Jacobi iterations and the hierarchy ~1/6 of the system arrays; a solve that needs '
               'far more iterations than multigrid should, or stalls (thin solids in void, strong stiffness contrast '
               'or grading make the coarse levels weak), switches that solid to block Jacobi with a warning. Not '
               'Automatic there yet.',
          tip='How the iterative solver is accelerated.\n\n'
              '- Jacobi (diagonal): light per step; suits small or uniform meshes and time-stepped heat.\n'
              '- Multigrid: far fewer steps on large or strongly graded meshes and with very different materials; '
              'heat and electric or magnetic field models (not with mixed precision) and 3D solid mechanics on a '
              'regular grid. The automatic solver of the field models uses it.'),
)


def _create_pcg(ctx, request):
    values = request.values
    return pcg.PCGSolver(precision=values['precision'], tolerance=values['tolerance'],
                         max_iterations=pcg.resolve_max_iterations(values['max_iterations'], ctx.domain),
                         label=lambda: f"{request.physics_name} {request.role} solver", backend=request.backend,
                         preconditioner=values['preconditioner'])


def _validate_pcg(vctx, values, path):
    issues = []
    tolerance = values.get('tolerance')
    if values.get('precision') == 'single' and tolerance is not None and tolerance < _TOLERANCE_FLOOR:
        issues.append(Issue('warning', path,
                            f'single precision: CG stalls near a relative residual of ~{_TOLERANCE_FLOOR:g}, so '
                            f'tolerance {tolerance:g} is usually not reached (use double)'))
    # multigrid at precision mixed: a role rule (SolverRole.mixed_preconditioners, solver_misfit), reported at the physics
    return issues


def _empty_memory_estimate(values, request):
    """No solver-owned allocation: every role's memory is counted by its physics (pcg.workspace_bytes, line_block
    *_bytes, host_lu_bytes of the operator, arnoldi_basis_size; banded: ~3 x 8 B per unknown of a 1D window)."""
    from ..core.memory_estimate import MemoryEstimate
    return MemoryEstimate()


PCG = SolverType(
    'pcg', 'Iterative linear solver', roles=('linear',), params=PCG_PARAMS,
    description='Preconditioned conjugate gradients for symmetric positive definite systems (the linear role of '
                'heat, electric_currents, electrostatics, magnetostatics, solid_mechanics, solid_mechanics_3d): Jacobi '
                'or the physics\' block-Jacobi preconditioner, or algebraic multigrid (the scalar diffusion types), '
                'warm-started from the previous solve; on the GPU '
                '(fused cupy kernels) or the CPU backend (numpy / scipy, the same iteration); double, single or mixed '
                '(float32 CG passes refined against the float64 residual) precision. '
                'Automatic is pcg with the defaults below, except the preconditioner: multigrid for the scalar '
                'diffusion types (jacobi at mixed precision; on triangle and tetrahedron meshes Automatic is '
                'sparse_direct, pcg is opt-in there; solid_mechanics_3d on tetrahedra solves with sparse_direct only); '
                'to change precision, tolerance, max_iterations or the '
                'preconditioner add a '
                'pcg solver node and assign it (physics solvers {"linear": "<node name>"}).',
    summary='Iterative solver for diffusion and elasticity problems: fast and memory-light on large meshes.',
    operators=('spd',), backends=('gpu', 'cpu'), precisions=('double', 'single', 'mixed'),
    grids=('tensor', 'triangles', 'tetrahedra'),
    create=_create_pcg, validate=_validate_pcg, memory_estimate=_empty_memory_estimate, time_estimate=pcg.time_estimate)

LINE_BLOCK_PARAMS = (
    Param('precision', 'choice', 'mixed', label='Solver precision', choices=('mixed', 'double'),
          choice_labels=(('mixed', 'Mixed precision'), ('double', 'Double precision')),
          help=f'mixed: block factorisation and sweeps in complex64, refined against the complex128 operator to a '
               f'relative residual of {line_block.REFINEMENT_TOLERANCE:g}; a shift still above '
               f'{line_block.ACCEPT_TOLERANCE:g} is re-solved by the physics\' reference solver (counted as '
               'rejected_shifts, or rejected_factor_solves for kept factors: em_fdfd_* linear, em_fdfd_bor, '
               'em_modes). double: '
               'complex128 throughout (slower on consumer GPUs, rarely rejected; indefinite Helmholtz systems near a '
               'resonance may need it).',
          tip='Number precision of the line-block solver.\n\n'
              '- Mixed precision: fast; every result is checked and polished in double precision.\n'
              '- Double precision: slower on most graphics cards.'),
)


def _create_line_block(ctx, request):
    return line_block.LineBlockSolver(precision=request.values['precision'],
                                      label=lambda: f"{request.physics_name} {request.role} solver",
                                      backend=request.backend)

LINE_BLOCK = SolverType(
    'line_block', 'Line-block direct solver', roles=('shifted', 'linear'), opt_in_roles=('linear',),
    params=LINE_BLOCK_PARAMS,
    description='Batched block-tridiagonal direct solver for families K + sum s_i M_i + sum t_j N_j of structured '
                '2D systems (M_i diagonal, N_j sparse with the same line coupling as K): unknowns ordered line by '
                'line, per shift a block Thomas '
                'factorisation (cost lines x b^3, b = unknowns across a line), many shifts per batch, equilibrated and '
                'refined against the double-precision operator; on the GPU (cupy) or the CPU backend (numpy, the same '
                'array code, chunks sized by the available RAM). Users (role shifted): heat thermal noise spectra '
                '(K + j omega C + q^2 A per frequency and axial wavenumber) and em_fdfd_bor (every driven system and '
                'shift-invert operator, the orders of an azimuthal_orders list as shifts m, m^2 of K0 + m K1 + m^2 '
                'K2; a kept factor serves the Arnoldi iterations; frequency_domain wavelengths of one order in batches '
                'of whole operators) and em_modes (the shift-invert operators of the mode solves: omega and, with '
                'group_index, omega (1 +- 1e-3), and in a frequency_domain study the next wavelengths, in batches of '
                'whole operators). Role linear (em_fdfd_ez / em_fdfd_hz only; '
                'Automatic stays sparse_direct): one block factor of the Helmholtz operator per wavelength and '
                'material state, reused for every source and as the shift-invert factor of the eigenfrequency study; '
                'frequency_domain wavelengths in batches of whole operators; the periodic axis (if any) lies inside '
                'the blocks, two periodic axes are refused; adjoint (transposed) solves of the optimization study and '
                'rejected solves use a host SuperLU of the same operator instead. A batch holds at most 8 operators '
                'and what 60 % of the free memory holds. A rejected solve is re-solved by the physics\' '
                'host SuperLU. On the CPU backend it is slower than the Automatic host SuperLU (each solve 7-15 x '
                'slower on 1e4-4e4 unknown complex 2D operators, double and mixed alike, at about the same '
                'factorisation time): there it only serves comparisons with the GPU runs.',
    summary='Fast GPU solver for frequency sweeps on elongated or mirror-reduced cross-sections, for round '
            '(axisymmetric) wave models and for elongated 2D frequency-domain light models. On the CPU backend the '
            'automatic solver is faster.',
    operators=('spd', 'complex_symmetric', 'general'), backends=('gpu', 'cpu'), precisions=('mixed', 'double'),
    grids=('tensor',), create=_create_line_block, memory_estimate=_empty_memory_estimate,
    time_estimate=line_block.time_estimate)

_SPARSE_DIRECT_LABELS = {'auto': 'Automatic', 'scipy_splu': 'Direct solver on the processor',
                         'cusolver_qr': 'Direct solver on the graphics card (small models)',
                         'cudss': 'GPU cuDSS', 'cudss_mixed': 'GPU cuDSS, mixed precision',
                         **_host_direct_choice_labels(host_direct.HOST_DIRECT_BACK_ENDS)}

SPARSE_DIRECT_PARAMS = (
    Param('back_end', 'choice', 'auto', label='Back end', choices=sparse_direct.LINEAR_SOLVER_CHOICES,
          choice_labels=tuple((name, _SPARSE_DIRECT_LABELS[name]) for name in sparse_direct.LINEAR_SOLVER_CHOICES),
          help='Where the sparse factorisation runs (role linear). scipy_splu: SuperLU on the host (needs scipy; the '
               'matrix moves to the host, where its L + U factors stay), one factorisation reused for every right-hand '
               'side, practical up to ~1e6 unknowns (warning above, refused above 2e6). cusolver_qr: sparse QR on the '
               'GPU (cupy only), one factorisation per solve, for small models (~1e5 unknowns on a square grid): its '
               'workspace, estimated as 64 B * nx * ny * the cell count of the narrower axis (doubled by a '
               'periodic wrap of the other axis), lies outside the GPU memory budget, so validation warns '
               'above 512 MiB and refuses above 2 GiB, and the solver also refuses a workspace over half the free GPU '
               'memory (overrunning the GPU freezes the display). auto: scipy_splu when scipy is installed, else '
               'cusolver_qr; on the cpu backend always scipy_splu (cusolver_qr is an error there); on the gpu backend '
               'with nvmath-python installed, the frequency-response linear solves of pressure_acoustics (triangle and '
               'tetrahedron meshes) and membrane (triangle meshes) take cudss_mixed instead, solid_mechanics_3d on '
               'tetrahedra cudss_mixed, and cudss: em_fdfd_ez / _hz (tensor and triangle meshes), em_fdfd_bor / '
               'em_fdfd_conical and drift_diffusion (tensor meshes; drift_diffusion also triangles), solid_mechanics '
               'on triangles and the heat / electric_currents / electrostatics / magnetostatics types on triangle and '
               'tetrahedron meshes (the run summary records the resolved back end); when cuDSS refuses such an operator (a zero pivot, missed accuracy or too '
               'little GPU memory) that solver switches to scipy_splu for the rest of the run with a warning (up to '
               '2e6 unknowns). The shifted role '
               '(thermal noise) and magnetic_fields_ac always factorise on the host. cudss (GPU backend, needs '
               "nvmath-python with cuDSS; only where the physics offers it: heat, electric_currents, electrostatics, "
               'magnetostatics, solid_mechanics and solid_mechanics_3d, em_fdfd_ez / em_fdfd_hz, em_fdfd_bor, '
               'em_fdfd_conical, drift_diffusion (real LU of every Newton Jacobian) and em_fdfd_nonlinear (opt-in '
               'only: its channels keep one factor each; picard Kerr / free-carrier channels then refactorise every '
               'sweep, no recycled GMRES) linear solves): NVIDIA\'s sparse '
               'direct solver '
               'factorises on the GPU in double precision (real symmetric positive definite systems by Cholesky, '
               'LDL^T if that fails; its own nested-dissection ordering) and keeps the factor in GPU memory outside '
               'the cupy pool (refused when cuDSS\'s own estimate after planning exceeds 80 % of the free device '
               'memory; the host SuperLU size rules do not apply); a re-assembly with the same sparsity pattern '
               '(new material values, Newton / saturation iterations) refactorises without the host planning step; '
               'solves stay on the GPU. A probe solve checks each factor (cuDSS refinement when its residual misses '
               '1e-10). cudss_mixed: the diagonally equilibrated operator factorised in single precision (float32 / '
               'complex64: on GPUs with slow double arithmetic ~10x faster to factorise, half the factor memory), '
               'every solve refined by float64 residual steps to 1e-12 (~3-4 single solves per solve); refused at '
               'the first factor when single precision cannot reach that (very high contrast: use cudss). '
               f'{_HOST_DIRECT_NAMES} (both backends, never chosen by auto; only where the physics offers them: the '
               'linear solves of heat, electric_currents, electrostatics, magnetostatics(_3d), solid_mechanics(_3d), '
               'pressure_acoustics, membrane, em_fdfd_ez / em_fdfd_hz, em_fdfd_bor, em_fdfd_conical, drift_diffusion '
               f'and em_fdfd_nonlinear, the shifted role of heat thermal noise, and {_HOST_DIRECT_DOUBLE_NAMES} for '
               'magnetic_fields_ac): a multicore sparse direct factor on the host by MUMPS (python-mumps) or Intel MKL '
               'PARDISO (pydiso, x86-64 only), after the diagonal equilibration of cuDSS: LDL^T for (complex) '
               'symmetric operators (the symmetrised D A of the FDFD types; the real symmetric positive definite '
               'systems: PARDISO Cholesky, LDL^T if that fails, MUMPS LDL^T), LU otherwise; transposed (adjoint) '
               'solves, GMRES on an '
               'earlier factor and inertia counts work as with scipy_splu. The scipy_splu size rules apply '
               '(uncalibrated for these libraries), plus a check of the library\'s own memory estimate after its '
               f'analysis against {_HOST_MEMORY_PERCENT} % of the available RAM; a probe solve checks each factor '
               '(float64 GMRES refinement on the factor when its residual misses 1e-10); a refusal raises (no '
               'fallback).'
               f'{_HOST_DIRECT_MIXED_HELP}{_HOST_DIRECT_COMPRESSED_HELP} Threads: the library defaults (OMP_NUM_THREADS for MUMPS, '
               'MKL_NUM_THREADS for PARDISO; with numba or a threaded BLAS in the same process set them to share the '
               f'cores). Optional packages, a validation error without them: {_HOST_DIRECT_INSTALL}.',
          tip='Where the equations are solved.\n\n'
              '- Automatic: the processor solver when the SciPy package is installed, else the graphics card solver; '
              'with nvmath-python installed, the GPU cuDSS solver takes the direct solves of sound pressure and '
              'membrane sweeps and 3D solid mechanics (mixed precision), and of FDFD, BOR, conical, drift-diffusion, '
              '2D solid mechanics and the heat / potential / magnetostatic types on finite element meshes (double), '
              'and switches to the processor solver if the graphics card solver '
              'refuses the model.\n'
              '- Direct solver on the processor: up to about a million unknowns, absorbing layers included.\n'
              '- Direct solver on the graphics card: small models only, about 100 000 unknowns; larger ones run out of '
              'graphics card memory.\n'
              '- GPU cuDSS: a fast exact solver on the graphics card for large models, limited by graphics card memory; '
              'needs nvmath-python; heat, electric, magnetic, solid, frequency-domain wave and semiconductor models '
              'only.\n'
              '- GPU cuDSS, mixed precision: faster to set up and half the memory, with the same accuracy; may be '
              'refused for very high material contrasts.\n'
              f'- {_GUI_HOST_LIBRARIES} on the processor: fast exact solvers using every processor core, for large '
              'models, limited by computer memory; need the optional python-mumps or pydiso package (Intel PARDISO: '
              'Intel / AMD 64-bit processors only); never picked by Automatic.'
              f'{_HOST_DIRECT_MIXED_TIP}{_HOST_DIRECT_COMPRESSED_TIP}'),
)


def _create_sparse_direct(ctx, request):
    """SparseDirectSolver of the resolved values; a back end resolved from 'auto' (schema.auto_back_end_values marks it)
    falls back to host SuperLU when cuDSS refuses an operator (B1b)."""
    from_auto = bool(request.values.get(sparse_direct.AUTO_BACK_END_KEY))
    return sparse_direct.SparseDirectSolver(back_end=request.values['back_end'],
                                            label=lambda: f"{request.physics_name} {request.role} solver",
                                            backend=request.backend,
                                            fallback_back_end='scipy_splu' if from_auto else None)


def _validate_sparse_direct(vctx, values, path):
    """A cuDSS back end without nvmath-python: error; scipy_splu (or auto on the cpu backend) without scipy: error;
    auto without scipy: info (find_spec only: GUI-thread safe). GPU back ends on the cpu are the backend_misfit, a cuDSS
    back end a role does not offer the opt-in misfit, both reported at the physics. A host direct back end (MUMPS /
    PARDISO) without its package or scipy: error."""
    back_end = values.get('back_end')
    if back_end in host_direct.HOST_DIRECT_BACK_ENDS:
        if not host_direct.host_direct_available(back_end):
            return [Issue('error', path, host_direct.missing_text(back_end))]
        if not sparse_direct.scipy_available():
            return [Issue('error', path, f"back_end '{back_end}' needs scipy (sparse matrix handling), which is not "
                                         'installed: pip install scipy')]
        return []
    if values.get('back_end') in sparse_direct.CUDSS_BACK_ENDS:
        from .cudss_factor import cudss_available
        if not cudss_available():
            return [Issue('error', path, f"back_end '{values['back_end']}' needs nvmath-python with cuDSS, which is not "
                                         "installed: pip install nvmath-python[cu12] (or [cu13]), or use 'auto'")]
        return []
    if sparse_direct.scipy_available():
        return []
    if solver_backend(vctx.spec) == 'cpu' and values.get('back_end') != 'cusolver_qr':
        return [Issue('error', path, 'the CPU backend factorises with scipy (SuperLU), which is not installed: pip '
                                     'install scipy')]
    if values.get('back_end') == 'scipy_splu':
        return [Issue('error', path, "back_end 'scipy_splu' needs scipy, which is not installed: pip install scipy, or "
                                     "choose 'cusolver_qr' (GPU sparse QR, up to ~1e5 unknowns) or 'auto'")]
    if values.get('back_end') == 'auto':
        return [Issue('info', path, "scipy is not installed, so back_end 'auto' uses 'cusolver_qr' (GPU sparse QR: one "
                                    'factorisation per solve, up to ~1e5 unknowns); pip install scipy for '
                                    "'scipy_splu' (host SuperLU, factorisation reused, ~1e6 unknowns)")]
    return []

SPARSE_DIRECT = SolverType(
    'sparse_direct', 'Sparse direct solver', roles=('linear', 'shifted'), params=SPARSE_DIRECT_PARAMS,
    description='Sparse direct factorisation, double precision (complex128): host SuperLU (scipy) or GPU sparse QR '
                '(back_end). Role linear: the Helmholtz systems of em_fdfd_ez / em_fdfd_hz / em_fdfd_nonlinear (one '
                'factorisation per wavelength and material state, reused for every source; after a small operator '
                'change GMRES on the earlier factor first), every Newton step of drift_diffusion and the eddy-current '
                'system of magnetic_fields_ac (host only); also the linear role of heat, electric_currents, '
                'electrostatics and magnetostatics (their Automatic on the CPU backend: one real float64 host '
                'SuperLU per assembly, exact solves; scipy_splu only), and the Automatic linear solver of the '
                'diffusion and elastic types on triangle meshes and of heat on tetrahedra (FEM, both backends). Role '
                'shifted: one host SuperLU '
                'per shift K + s M, the '
                'reference of the thermal noise spectra. Iterative solvers are not offered for these (they stall on '
                'the indefinite Helmholtz matrix). On the CPU backend: SuperLU on scipy operators, numpy solutions. '
                'back_end cudss / cudss_mixed (GPU cuDSS, kept device factor, plan reused across re-assemblies) only '
                'where the physics offers it (the diffusion and elastic linear roles, em_fdfd_ez / _hz, em_fdfd_bor, '
                'em_fdfd_conical, drift_diffusion). '
                f'back_end {_HOST_DIRECT_NAMES} (the optional host MUMPS / Intel MKL PARDISO, multicore, both '
                'backends, never chosen by auto) likewise only where the physics offers them.',
    summary='Exact solver for wave, semiconductor and frequency-sweep problems: robust, memory grows quickly with the '
            'mesh.',
    operators=('spd', 'complex_symmetric', 'general'), backends=('gpu', 'cpu'), precisions=('double',),
    grids=('tensor', 'triangles', 'tetrahedra'), create=_create_sparse_direct, validate=_validate_sparse_direct,
    memory_estimate=_empty_memory_estimate, backend_misfit=sparse_direct.backend_misfit,
    time_estimate=sparse_direct.time_estimate,
    opt_in_settings={'back_end': sparse_direct.CUDSS_BACK_ENDS + host_direct.HOST_DIRECT_BACK_ENDS})


BATCHED_DIRECT_PARAMS = (
    Param('precision', 'choice', 'mixed', label='Solver precision', choices=batched_direct.PRECISIONS,
          choice_labels=(('mixed', 'Mixed precision'), ('double', 'Double precision')),
          help='mixed: each batch factorised in complex64 / float32 by cuDSS, every system refined against its own '
               f'complex128 operator to a relative residual of {batched_direct.REFINEMENT_TOLERANCE:g} (one batched '
               'solve per step for all systems); a system still above '
               f'{batched_direct.ACCEPT_TOLERANCE:g} is re-solved on its own (counter batch_rejected). double: '
               'complex128 factors (slower on consumer GPUs). The per-system solves of the role (other studies, '
               'rejected systems) use cuDSS of the same precision (cudss_mixed / cudss), host SuperLU after a cuDSS '
               'refusal; on the CPU backend every solve is host SuperLU.',
          tip='Number precision of the batched solves.\n\n'
              '- Mixed precision: fast; every result is checked and polished in double precision.\n'
              '- Double precision: slower on most graphics cards.'),
    Param('batch_size', 'int', 0, label='Batch size', min=0,
          help=f'Systems factorised together (one cuDSS plan for the whole batch); 0 = up to {batched_direct.MAX_BATCH} '
               f'and what {batched_direct.MEMORY_FRACTION:.0%} of the free memory holds (each system keeps its own '
               'factor during its batch, so the memory grows with the batch size).',
          tip='How many frequencies are solved together; 0 picks a suitable number. More can be faster but use more '
              'graphics card memory.'),
)


def _create_batched_direct(ctx, request):
    values = request.values
    return batched_direct.BatchedDirectSolver(precision=values['precision'], batch_size=values['batch_size'],
                                              label=lambda: f"{request.physics_name} {request.role} solver",
                                              backend=request.backend)


def _validate_batched_direct(vctx, values, path):
    """The gpu backend without nvmath-python (cuDSS): error; the cpu backend without scipy: error (find_spec only)."""
    if solver_backend(vctx.spec) == 'gpu':
        from .cudss_factor import cudss_available
        if not cudss_available():
            return [Issue('error', path, 'the batched direct solver needs nvmath-python with cuDSS on the gpu backend, '
                                         'which is not installed: pip install nvmath-python[cu12] (or [cu13]), or use '
                                         'sparse_direct')]
        return []
    if not sparse_direct.scipy_available():
        return [Issue('error', path, 'the batched direct solver on the CPU backend factorises with scipy (SuperLU), '
                                     'which is not installed: pip install scipy')]
    return []


BATCHED_DIRECT = SolverType(
    'batched_direct', 'Batched direct solver', roles=('linear',), params=BATCHED_DIRECT_PARAMS,
    opt_in_roles=('linear',),
    description='Many systems on one sparsity pattern factorised and solved together: the frequencies of a direct '
                'frequency sweep (frequency_response study, sweep_method direct) in batches of up to '
                f'{batched_direct.MAX_BATCH}, one explicitly batched GPU cuDSS factorisation per batch (planned once, '
                'its values refreshed for every later batch), each system equilibrated and refined against its own '
                'double-precision operator; a system missing the accept tolerance is re-solved on its own. Opt-in '
                'where a physics offers it: pressure_acoustics (triangle and tetrahedron meshes) and membrane '
                '(triangle meshes) linear role. Every other solve of the role (other studies, rejected systems, an '
                'nvmath without explicit batching: a warning, then per system) runs as sparse_direct cuDSS of the same '
                'precision. On the CPU backend a loop of host SuperLU factors (parity with the GPU runs, not speed).',
    summary='Solves many frequencies of a sweep together on the graphics card: fewer, larger GPU calls than one '
            'factorisation per frequency.',
    operators=('complex_symmetric', 'general'), backends=('gpu', 'cpu'), precisions=batched_direct.PRECISIONS,
    grids=('triangles', 'tetrahedra'), create=_create_batched_direct, validate=_validate_batched_direct,
    memory_estimate=_empty_memory_estimate, time_estimate=batched_direct.time_estimate)


def _create_banded(ctx, request):
    return banded.BandedSolver(label=lambda: f"{request.physics_name} {request.role} solver", backend=request.backend)


def _validate_banded(vctx, values, path):
    """Errors without scipy and in a 2D / 3D model (the node is for 1D models: a window one cell across)."""
    issues = []
    if not sparse_direct.scipy_available():
        issues.append(Issue('error', path, 'the banded solver factorises with scipy (cholesky_banded), which is not '
                                           'installed: pip install scipy'))
    if vctx.spec.dimensions != 1:
        issues.append(Issue('error', path, f'the banded solver solves 1D models only (this model is '
                                           f'{vctx.spec.dimensions}D): use a pcg or sparse_direct node'))
    return issues

BANDED = SolverType(
    'banded', 'Banded direct solver (1D)', roles=('linear',),
    description='Direct solve of 1D models (domain size_y null: a line of cells, symmetry plan 2.20): the operator of '
                'a window one cell across is banded (tridiagonal for heat, electric_currents, electrostatics), so one '
                'host banded Cholesky (scipy.linalg cholesky_banded, double precision) per assembly gives exact '
                'solves at a cost linear in the cells; on the GPU backend the operator, rhs and solution move to the '
                'host per factorisation / solve. The Automatic linear solver of these types in a 1D model on both '
                'backends; refused in 2D / 3D models (a node there is an error).',
    summary='Exact, instant solver for 1D models (slabs, cylinders, spheres).',
    operators=('spd',), backends=('gpu', 'cpu'), precisions=('double',), grids=('line',),
    create=_create_banded, validate=_validate_banded, memory_estimate=_empty_memory_estimate,
    time_estimate=banded.time_estimate)

_FACTORIZATION_LABELS = {'auto': 'Automatic', 'superlu': 'Host SuperLU', 'cudss': 'GPU cuDSS',
                         'cudss_mixed': 'GPU cuDSS, mixed precision',
                         **_host_direct_choice_labels(host_direct.HOST_DIRECT_FACTORIZATIONS)}
_HOST_FACTORIZATION_MIXED_HELP = (
    f' {_HOST_DIRECT_MIXED_NAMES} (only where cudss_mixed is offered): a single-precision host factor, the solves '
    'refined and polished as for cudss_mixed; a first factor whose probe misses is refactorised in double for the '
    'rest of the run.' if host_direct.HOST_DIRECT_MIXED_BACK_ENDS else '')
_HOST_FACTORIZATION_MIXED_TIP = (
    f'\n- {_GUI_HOST_MIXED_LIBRARIES} on the processor, mixed precision (where GPU cuDSS mixed precision is '
    'offered): faster to factorise and less memory, with the same accuracy.'
    if host_direct.HOST_DIRECT_MIXED_BACK_ENDS else '')

ARPACK_PARAMS = (
    Param('tolerance', 'float', 0.0, label='Eigenvalue tolerance', unit='1', min=0.0,
          help='Relative accuracy ARPACK asks of each eigenvalue (scipy eigs tol); 0 = machine precision (default). '
               'solid_mechanics_3d (inexact PCG inner solves) raises it to at least 1e3 x its linear CG tolerance.',
          tip='How accurately eigenvalues are converged; 0 means as accurately as the computer can.'),
    Param('subspace_size', 'int', 0, label='Subspace size', min=0,
          help='Arnoldi vectors ARPACK keeps (scipy eigs ncv), raised to the eigenpairs wanted + 2; 0 = scipy\'s '
               'default max(2 k + 1, 20) for k wanted eigenpairs. A larger basis converges in fewer operator '
               'applications but holds that many complex128 vectors of the system size on the host.',
          tip='Working vectors of the eigenvalue search; 0 picks a suitable number. More vectors can be faster but '
              'need more memory.'),
    Param('factorization', 'choice', 'auto', label='Shift-invert factorisation',
          choices=arpack_shift_invert.FACTORIZATIONS,
          choice_labels=tuple((name, _FACTORIZATION_LABELS[name]) for name in arpack_shift_invert.FACTORIZATIONS),
          help="LU of the shifted operator. auto: superlu, except on the gpu backend with nvmath-python installed for "
               "the vibration modes of solid_mechanics on triangle meshes (cudss) and of solid_mechanics_3d on "
               "tetrahedra (cudss_mixed; brick meshes have no factor), where a cuDSS refusal (a zero pivot, missed "
               "accuracy or too little GPU memory) switches that solver to superlu for the rest of the run with a "
               "warning (up to 2e6 unknowns; the run summary records the resolved value and eigen_auto_fallbacks). "
               "superlu: scipy SuperLU on the host (host memory, both backends). cudss "
               "(GPU backend, needs nvmath-python with cuDSS): NVIDIA's sparse direct solver factorises on the device "
               "in double precision with its own nested-dissection ordering (LDL^T where the operator is complex "
               "symmetric after a diagonal scaling: em_fdfd_3d, 2D em_fdfd without Bloch phase, solid_mechanics); the factor lives in GPU memory "
               "outside the cupy pool (refused when cuDSS's own estimate exceeds 80 % of the free device memory, and "
               "the host SuperLU size rule of the FDFD eigen solves no longer applies); ARPACK still runs on the host, so each "
               "operator application moves one vector to the device and back. A probe solve checks the factor and "
               "turns on cuDSS iterative refinement when its residual misses 1e-10. cudss_mixed (em_fdfd_3d, "
               "em_fdfd_ez / em_fdfd_hz, em_fdfd_bor, em_fdfd_conical, em_modes on triangle meshes, solid_mechanics "
               "and solid_mechanics_3d on tetrahedra): the diagonally equilibrated operator factorised in single "
               "precision (complex64 / float32: about 10x faster to factorise on GPUs with slow double arithmetic, "
               "half the GPU memory, solves ~2x cheaper). The resonance solves of the frequency-domain optics types "
               "and em_modes correct every ARPACK solve by one float64 residual step (~3e-8), stop ARPACK at ~1e-6 "
               "and re-evaluate each eigenvalue by the float64 Rayleigh quotient of its vector, accurate to ~1e-14 "
               "because the scaled operator is complex symmetric (with a Bloch phase it is not: full float64 "
               "refinement per solve instead); the solid_mechanics pencils refine every solve to 1e-12. "
               f"{_HOST_DIRECT_DOUBLE_NAMES} (every eigen role, both backends, never chosen by auto; needs "
               "python-mumps / pydiso): the shifted operator factorised on the host by MUMPS or Intel MKL PARDISO "
               "(x86-64 only), multicore, with the cudss equilibration and LDL^T where the operator is (complex) "
               "symmetric after the diagonal scaling, else LU; inertia counts of target / band searches come from "
               "the library's LDL^T pivots (a counting SuperLU when pivots were perturbed or the pencil is "
               "Hermitian); the host SuperLU size rules apply, plus the library's own memory estimate after its "
               f"analysis against {_HOST_MEMORY_PERCENT} % of the available RAM; a refusal raises (no fallback)."
               f"{_HOST_FACTORIZATION_MIXED_HELP}{_HOST_DIRECT_COMPRESSED_HELP} Ignored by solid_mechanics_3d on brick "
               "meshes (no factor).",
          tip='Where the eigenvalue search factorises its matrix.\n\n'
              '- Automatic: host SuperLU, except vibration modes on finite element meshes, which use the GPU cuDSS '
              'solver when nvmath-python is installed (switching to host SuperLU if the graphics card solver refuses '
              'the model).\n'
              '- Host SuperLU: works everywhere, limited by computer memory.\n'
              '- GPU cuDSS: often much faster on large 3D models, limited by graphics card memory; needs '
              'nvmath-python.\n'
              '- GPU cuDSS, mixed precision (frequency-domain optics, mode solves on triangle meshes, vibration '
              'modes on finite element meshes): faster still and half the graphics card memory, with the same '
              'accuracy.\n'
              f'- {_GUI_HOST_LIBRARIES} on the processor: exact factorisation using every processor core, often '
              'faster than host SuperLU on large models; needs the optional python-mumps or pydiso package; never '
              'picked by Automatic.'
              f'{_HOST_FACTORIZATION_MIXED_TIP}{_HOST_DIRECT_COMPRESSED_TIP}'),
)


def _create_arpack(ctx, request):
    """ArpackShiftInvertSolver of the resolved values; a cuDSS factorization resolved from 'auto'
    (schema.auto_back_end_values marks it) falls back to host SuperLU when cuDSS refuses an operator (B1b)."""
    values = request.values
    from_auto = bool(values.get(sparse_direct.AUTO_BACK_END_KEY))
    return arpack_shift_invert.ArpackShiftInvertSolver(
        tolerance=values['tolerance'], subspace_size=values['subspace_size'],
        factorization=values['factorization'], label=lambda: f"{request.physics_name} {request.role} solver",
        fallback_factorization='superlu' if from_auto else None,
        mixed_min_unknowns=arpack_shift_invert.MIXED_EIGEN_MIN_UNKNOWNS,   # S5c: small cudss_mixed in double
        device_arnoldi=request.backend == 'gpu')   # eigs: device Krylov-Schur around host solves too


def _validate_arpack(vctx, values, path):
    """A cuDSS factorization without nvmath-python: error (find_spec only: GUI-thread safe). cuDSS on the cpu is the
    backend_misfit, cudss_mixed outside the FDFD eigen roles the role's setting_choices, both reported at the physics.
    A host direct factorization (MUMPS / PARDISO) without its package: error."""
    from .cudss_factor import cudss_available
    factorization = values.get('factorization')
    if (factorization in host_direct.HOST_DIRECT_FACTORIZATIONS
            and not host_direct.host_direct_available(factorization)):
        return [Issue('error', path, host_direct.missing_text(factorization))]
    if factorization in arpack_shift_invert.CUDSS_FACTORIZATIONS and not cudss_available():
        return [Issue('error', path, f"factorization '{factorization}' needs nvmath-python with cuDSS, which is not "
                                     "installed: pip install nvmath-python[cu12] (or [cu13]), or use 'superlu'")]
    return []

ARPACK_SHIFT_INVERT = SolverType(
    'arpack_shift_invert', 'Shift-invert eigensolver', roles=('eigen',), params=ARPACK_PARAMS,
    description='Eigenpairs nearest a target: host SuperLU (or, factorization cudss, GPU cuDSS; auto: cuDSS for the '
                f'vibration modes of finite element solids on the GPU; {_HOST_DIRECT_DOUBLE_NAMES}: the optional '
                'multicore host MUMPS / Intel MKL PARDISO) factorisation of the operator shifted to the '
                'target and '
                'ARPACK (scipy eigs, largest |theta| of the shifted inverse) in double precision. Used by em_modes '
                '(cross-section modes near the effective-index guess) and by em_fdfd_ez / em_fdfd_hz in an '
                'eigenfrequency study (resonances nearest the wavelength setting). solid_mechanics (eigenfrequency, '
                'optomechanics: plane-strain or 2.5D guided vibration modes) uses symmetric shift-invert Lanczos '
                '(scipy eigsh) on a real SuperLU factor of K - sigma M. solid_mechanics_3d (eigenfrequency) '
                'uses it without a factor: symmetric Lanczos (scipy eigsh) on K^-1 M at 0, each K^-1 one PCG solve of '
                "its 'linear' solver on the device. The ARPACK basis stays on the host and SuperLU runs the same on "
                'both backends (cudss: GPU backend only); on '
                'triangle and tetrahedron meshes (FEM) the same factorisation of the assembled element matrices.',
    summary='Finds the modes or resonances closest to a target value.',
    operators=('spd', 'complex_symmetric', 'general'), backends=('gpu', 'cpu'), precisions=('double',),
    grids=('tensor', 'triangles', 'tetrahedra'), create=_create_arpack, validate=_validate_arpack,
    memory_estimate=_empty_memory_estimate, backend_misfit=arpack_shift_invert.backend_misfit,
    time_estimate=arpack_shift_invert.time_estimate)


BLOCK_EIGEN_PARAMS = (
    Param('tolerance', 'float', block_eigen.DEFAULT_TOLERANCE, label='Residual tolerance', unit='1', min=0.0,
          help='Relative residual ||K x - lambda M x|| / (|lambda| ||M x||) every wanted eigenpair must reach; 0 = '
               f'the default {block_eigen.DEFAULT_TOLERANCE:g}. The eigenvalue error is about the square of it (a '
               'symmetric pencil), so 1e-8 gives frequencies to ~1e-15 unless two modes nearly coincide. Below ~1e-12 '
               'the float64 rounding floor of stiff 3D operators stops the iteration (a warning).',
          tip='How accurately the vibration modes are converged.'),
    Param('guard_vectors', 'int', 0, label='Guard vectors', min=0,
          help='Extra block rows searched beside the wanted eigenpairs (they speed up the convergence of the highest '
               f'wanted ones); 0 = max({block_eigen.MIN_GUARD_VECTORS}, {block_eigen.GUARD_FRACTION:g} x wanted). The '
               'solve holds 9 vectors of the system size per block row.',
          tip='Extra search vectors; 0 picks a suitable number. More can converge in fewer iterations but use more '
              'memory.'),
    Param('max_iterations', 'int', 0, label='Max iterations', min=0,
          help='Block iterations before the solve stops unconverged (a warning); 0 = the physics\' default (the '
               "'linear' CG iteration limit of solid_mechanics_3d), else "
               f'{block_eigen.DEFAULT_MAX_ITERATIONS}.',
          tip='Upper limit of search iterations; 0 picks a suitable number.'),
    Param('precision', 'choice', 'mixed', label='Solver precision', choices=block_eigen.PRECISIONS,
          choice_labels=(('mixed', 'Mixed precision'), ('double', 'Double precision')),
          help='mixed: float32 blocks, products and preconditioner until the residual reaches '
               f'{block_eigen.SINGLE_PHASE_TOLERANCE:g} or stalls at the float32 rounding floor, then float64 from '
               'there to the tolerance (consumer GPUs run float32 products and the dense block products 32-64x '
               'faster than float64). double: float64 throughout. Both end at the same float64 accuracy.',
          tip='Number precision of the search.\n\n'
              '- Mixed precision: fast; the result is finished in double precision.\n'
              '- Double precision: slower on most graphics cards.'),
    Param('preconditioner', 'choice', 'multigrid', label='Preconditioner', choices=block_eigen.PRECONDITIONERS,
          choice_labels=(('multigrid', 'Multigrid'), ('jacobi', 'Block Jacobi')),
          help="multigrid: one V-cycle of the physics' multigrid per vector and iteration (solid_mechanics_3d: the "
               "geometric brick multigrid of its 'linear' solver, built for the eigen solve when that solver runs "
               'block Jacobi; a hierarchy that cannot be built falls back to block Jacobi with a warning): '
               'iterations nearly independent of the mesh size. jacobi: the 3 x 3 node block Jacobi of the linear '
               'solver (no hierarchy memory; iterations grow with the longest mesh axis).',
          tip='How each search step is accelerated.\n\n'
              '- Multigrid: few iterations on any mesh size.\n'
              '- Block Jacobi: no extra memory, many more iterations on fine meshes.'),
)


def _create_block_eigen(ctx, request):
    values = request.values
    return block_eigen.BlockEigenSolver(tolerance=values['tolerance'], guard_vectors=values['guard_vectors'],
                                        max_iterations=values['max_iterations'], precision=values['precision'],
                                        preconditioner=values['preconditioner'],
                                        label=lambda: f"{request.physics_name} {request.role} solver",
                                        backend=request.backend)


BLOCK_EIGEN = SolverType(
    'block_eigen', 'Block eigensolver', roles=('eigen',), params=BLOCK_EIGEN_PARAMS, opt_in_roles=('eigen',),
    description='The lowest eigenpairs of a symmetric pencil without a factorisation: preconditioned block '
                'iterations (LOBPCG with soft locking) on the physics\' own matrix-free products and preconditioner, '
                'every vector of the search block in device memory (9 vectors of the system size per block row, '
                'nothing else of that size), dense block products on cuBLAS / BLAS; mixed precision (a float32 phase '
                'finished in float64) or double. Opt-in where a physics offers it: solid_mechanics_3d on brick '
                "(tensor) meshes, eigenfrequency study (the lowest modes; arpack_shift_invert's Lanczos there runs "
                'one inner PCG solve per step). Both backends.',
    summary='Finds the lowest vibration modes without a factorisation: fast and memory-light on large 3D meshes.',
    operators=('spd',), backends=('gpu', 'cpu'), precisions=block_eigen.PRECISIONS, grids=('tensor',),
    create=_create_block_eigen, memory_estimate=_empty_memory_estimate, time_estimate=block_eigen.time_estimate)


TIME_FILTER_PARAMS = (
    Param('bandwidth', 'float', time_filter.DEFAULT_BANDWIDTH, label='Band half-width', unit='1', min=0.001,
          max=0.9,
          help='Half-width of the sampled band, relative to the target frequency (the wavelength setting): the '
               'resonances searched lie within target (1 +- bandwidth); the source pulse covers '
               f'{time_filter.PULSE_BANDWIDTH_FACTOR:g} x this. Wider finds more resonances per run but needs more '
               'sampled frequencies.',
          tip='How far from the target frequency resonances are searched, as a fraction of it.'),
    Param('sampled_frequencies', 'int', 0, label='Sampled frequencies', min=0,
          help='Frequencies of the running DFT over the band (one complex accumulator of the grid size each, the '
               f'search basis); 0 = max({time_filter.MIN_SAMPLED_FREQUENCIES}, {time_filter.SAMPLES_PER_WANTED} x '
               f'n_resonances + 8), at most {time_filter.MAX_SAMPLED_FREQUENCIES}. More than the resonances in the '
               'band are needed: modes outside the requested count still occupy the span.',
          tip='How many frequency samples build the search space; 0 picks a suitable number.'),
    Param('periods', 'float', 0.0, label='Run periods', unit='1', min=0.0,
          help='Length of the time stepping in periods of the target frequency after the source pulse; 0 = '
               f'max({time_filter.MIN_PERIODS}, {time_filter.RESOLUTION_PERIODS:g} / bandwidth). Longer runs '
               'separate close resonances better; very high Q needs no longer run (the Rayleigh-Ritz step, not the '
               'ring-down, sets the accuracy).',
          tip='How long the time-domain run lasts; 0 picks a suitable length.'),
    Param('sources', 'int', time_filter.DEFAULT_SOURCES, label='Sources', min=1,
          help='Seeded random point sources inside the window (not on PEC or wall cells) exciting the run: several, '
               'so a mode with a node at one source is still excited.',
          tip='How many random sources excite the time-domain run.'),
    Param('polish_steps', 'int', time_filter.DEFAULT_POLISH_STEPS, label='Polish steps', min=0,
          help='Rayleigh-Ritz refinements: each appends the scaled residuals of the wanted resonances to the basis and '
               'projects again (sparse products only), removing the time-stepping and PML-stretch differences '
               'between the FDTD fields and the frequency-domain modes; a resonance whose relative residual stays '
               f'above {time_filter.RESIDUAL_WARNING:g} is named in a warning.',
          tip='Extra refinement steps of the found resonances.'),
    Param('precision', 'choice', 'single', label='Solver precision', choices=time_filter.PRECISIONS,
          choice_labels=(('single', 'Single precision'), ('double', 'Double precision')),
          help='Precision of the FDTD time stepping (fields and coefficients; the DFT accumulators are complex64, '
               'the projection and residuals float64): single is enough for the search basis.',
          tip='Number precision of the time-domain run.'),
)


def _create_time_filter(ctx, request):
    values = request.values
    return time_filter.TimeFilterEigenSolver(
        bandwidth=values['bandwidth'], sampled_frequencies=values['sampled_frequencies'], periods=values['periods'],
        sources=values['sources'], polish_steps=values['polish_steps'], precision=values['precision'],
        label=lambda: f"{request.physics_name} {request.role} solver", backend=request.backend)


TIME_FILTER = SolverType(
    'time_filter', 'Time-filter eigensolver', roles=('eigen',), params=TIME_FILTER_PARAMS, opt_in_roles=('eigen',),
    description='Resonances near a target without a factorisation: an FDTD time stepping of the same structure from '
                'seeded broadband point sources, a running DFT of the whole grid at sampled frequencies of a band '
                'around the target (one complex accumulator per sampled frequency), then a Rayleigh-Ritz projection '
                'of the frequency-domain operator onto those fields with residual polish steps (sparse products and '
                'small dense eigenproblems only); memory: the FDTD fields plus the accumulators. Opt-in where a '
                'physics offers it: em_fdfd_ez / em_fdfd_hz on tensor meshes (eigenfrequency study; not with a Bloch '
                'phase, polar coordinates, subpixel smoothing, conformal boundaries or conducting sheets). The '
                'eigenvalues reach the projection accuracy (a residual per resonance in the log), not ARPACK\'s '
                'machine precision. Both backends.',
    summary='Finds resonances of open or lossy light models from a short time-domain run: no matrix factorisation, '
            'little memory.',
    operators=('spd', 'complex_symmetric', 'general'), backends=('gpu', 'cpu'), precisions=time_filter.PRECISIONS,
    grids=('tensor',), create=_create_time_filter, memory_estimate=_empty_memory_estimate,
    time_estimate=time_filter.time_estimate)


EXPLICIT_KERNELS_PARAMS = (
    Param('precision', 'choice', 'single', label='Solver precision', choices=('single', 'double', 'half_storage'),
          choice_labels=(('single', 'Single precision'), ('double', 'Double precision'),
                         ('half_storage', 'Half-precision storage')),
          help='single: float32 fields and arithmetic (every engine). double (em_fdtd_ez / em_fdtd_hz, both '
               'backends): float64 fields, coefficients, CPML, pole and source arrays and arithmetic, about 2x the field '
               'memory and time; for long or high-Q ring-downs, where float32 rounding of the coefficients (~1e-7 of '
               'the mode frequency) accumulates into a visible phase drift; materials, coupling slots, absorbed '
               'power, probe traces and the box DFT spectra stay float32 / complex64; refused with nonlinear_medium '
               'and gain_medium. half_storage (em_fdtd_3d, elastic_waves, '
               'laminar_flow transient only): the fields / lattice populations are stored as fp16 and computed in '
               'float32 '
               'registers, about half the field memory and up to ~1.5-2x faster on bandwidth-bound grids; fp16 has an '
               '11-bit mantissa, so results carry a ~1e-3 relative noise floor (-60 dB): flux spectra with deep nulls, '
               'harmonic inversion / Q values and steady states are refused or unusable (features that need single '
               'say so in validation), decaying tails below ~1e-7 of the peak flush to zero.',
          tip='Number precision of the time stepping.\n\n'
              '- Single precision: suits every model.\n'
              '- Double precision: about twice the memory and time, for long runs and sharp resonances of 2D light '
              'models.\n'
              '- Half-precision storage: fits about twice the cells in the same graphics memory and runs faster, '
              'with about three correct digits; for large 3D light, elastic wave or transient flow models.'),
)


def _create_explicit_kernels(ctx, request):
    return explicit_kernels.StepperConfig(precision=request.values['precision'])


EXPLICIT_KERNELS = SolverType(
    'explicit_kernels', 'Explicit time stepping', roles=('stepper',), params=EXPLICIT_KERNELS_PARAMS,
    description='The fused GPU kernels of an explicit time-stepping engine (em_fdtd_ez / em_fdtd_hz / em_fdtd_3d, '
                'elastic_waves, laminar_flow): single precision (float32 fields and arithmetic), double (float64 '
                'fields and arithmetic; em_fdtd_ez / em_fdtd_hz, kernels compiled per precision) or half_storage '
                '(fp16 fields in memory, float32 arithmetic; em_fdtd_3d, elastic_waves and transient laminar_flow). On '
                'the CPU backend (every stepper type) the same steps run as numpy twins of the kernels, or '
                'numba-compiled twins of the per-step field updates (Yee updates, em_fdtd_3d CPML / ohmic loss, the '
                'elastic_waves velocity / stress updates, the lattice step) when numba is installed (environment variable '
                'OPEN_SOL_CPU_KERNELS=numpy keeps numpy); single or double there, several times slower than a GPU: '
                'small 2D models and tests.',
    summary='Time stepping on the graphics card (or the processor for small 2D light models), in single or double '
            'precision or with half-precision field storage.',
    operators=('spd', 'complex_symmetric', 'general'), backends=('gpu', 'cpu'),
    precisions=('single', 'double', 'half_storage'), grids=('tensor',), create=_create_explicit_kernels,
    memory_estimate=_empty_memory_estimate, backend_misfit=explicit_kernels.backend_misfit,
    time_estimate=explicit_kernels.time_estimate)

SOLVER_TYPES = (PCG, LINE_BLOCK, SPARSE_DIRECT, BATCHED_DIRECT, BANDED, ARPACK_SHIFT_INVERT, BLOCK_EIGEN, TIME_FILTER,
                EXPLICIT_KERNELS)


def _selftest_cases():
    """The solver SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SOLVER_SELFTEST_CASES
    return SOLVER_SELFTEST_CASES


def register(registry):
    """Register every SolverType, then the solver selftest loader (CPU vs GPU parity, default precision)."""
    for solver_type in SOLVER_TYPES:
        registry.register_solver_type(solver_type)
    registry.register_selftests(_selftest_cases)

"""Model definition of the 3D FDFD physics type 'em_fdfd_3d': settings, the em_fdtd_3d material
vocabulary, features (PEC, resonance table), validation and the memory / run-time estimates; registered by
definition.register after em_fdtd_3d."""
import math
from dataclasses import replace
from functools import lru_cache

import numpy as np

from ...core import time_estimate as te
from ...core.backend import to_host
from ...core.constants import FACE_LABELS, FACES_3D
from ...core.memory_estimate import (COMPLEX64_BYTES, COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES,
                                     FLOAT64_BYTES, INT32_BYTES, MemoryEstimate, host_counted_lu_bytes, host_lu_bytes,
                                     solver_job_items)
from ...model.definition_helpers import (DISCRETIZATION_PARAM, eigen_factorization, eigen_role, eigen_subspace_size,
                                         element_order, parkable_material_arrays)
from ...model.schema import DomainRequirements, FeatureType, Issue, Param, PhysicsType, solver_backend
from ...solvers.arpack_shift_invert import (CUDSS_FACTORIZATIONS, LDLT_FACTORIZATIONS, MIXED_EIGEN_MIN_UNKNOWNS,
                                            MIXED_FACTORIZATIONS, arnoldi_basis_size)
from ...solvers.sparse_direct import vector_3d_size_issue
from ..em.definition_common import (DEFAULT_WAVELENGTH, EIGENFREQUENCY_STUDY, EM_FAMILY, _refractive_index,
                                    all_solved_wavelengths, boundary_settings_params, make_validate_physics,
                                    object_result, physics_counter, HOST_DIRECT_NAMES, resonance_eigenpairs,
                                    shortest_solved_wavelength, source_slot_docs, state_arrays_read)
from ..em.definition_fdfd import (MIN_REGION_FRACTION_PARAM, resonance_region_memory_estimate,
                                  validate_resonance_table)
from ..em.eigenfrequency import (REGION_COLUMN, ResonanceRegion, ResonanceTable, resonance_filter_time_item,
                                 resonance_table_result)
from ..em.memory_estimates import (EIGEN_PHASE, EIGEN_ROLE, _study_values, cudss_factor_items, eigen_mixed,
                                   host_factor_items, own_cudss_search_items, real_cross_section_operator,
                                   resonance_search_items)
from .boundaries3d import AXES, FACE_TYPES_3D, MIRROR_TYPES_3D
from .definition import (_BC_CHOICE_LABELS, _BC_TIP, _MATERIAL_PROPERTIES, _MATERIAL_SCHEMA,
                         _OPTIONAL_MATERIAL_SCHEMA, _PML_VISIBLE_3D, _SLOT_ARRAYS, _allocated_arrays, _apply_pec,
                         _boundary_config, _ignored_bloch_issues, _pec_memory, _validate_pec, _validate_z_faces,
                         write_dispersive_regions_3d)
from .physics3d import MATERIAL_ARRAYS
from .definition_sources3d import domain_sizes
from .fdfd3d import ARNOLDI_BASIS_PER_WANTED, FDFDPhysics3D
from .fem3d import YEE_EQUIVALENT_UNKNOWNS
from .fem_solver3d import FEMResonancePhysics3D
from .renderer_fdfd3d import FDFDRenderer3D

TYPE_NAME = 'em_fdfd_3d'
_GRIDS = ('tensor', 'tetrahedra')   # PhysicsType / FeatureType.grids: tetrahedra = FEMResonancePhysics3D (T4b)
# tetrahedra: stored entries per Nedelec row (degree 1 ~18, degree 2 ~45; fem3d YEE_EQUIVALENT_UNKNOWNS) and Nedelec
# unknowns per element (TetrahedronMeshEstimate counts: degree 1 = edges, degree 2 = 2 (edges + faces))
_NEDELEC_ROW_ENTRIES = {1: 18, 2: 45}
_ROW_ENTRIES = 13              # stored entries per E unknown of the curl-curl operator (self, 4 + 4 + 4 neighbours)
_CURL_ENTRIES = 4              # per row of curl_e / curl_h
# run-time estimate: host bytes moved per stored operator entry by one assembly (scipy krons, curl products, live-map
# slicing, the face eps; a guess to calibrate from the summary's assembly_seconds)
ASSEMBLY_BYTES_PER_ENTRY = 600
# run-time estimate: SuperLU of the 3-component curl-curl (nested dissection) over the 3D 7-point complex curve at the
# same unknowns n = 3 x cells (theory ~3: denser separators). Mie sphere example: 7.7 at 18^3 and 6.4 at 19^3 cells,
# 8.7 at 26^3: above ~2.5e4 unknowns the factor grows ~n^0.3 faster than the curve (9.6 s -> 95 s from 19^3 to 26^3,
# n^2.44 vs the curve's extrapolated 2.12; the profile's 3D curve stops at 27,000 unknowns)
CURL_CURL_LU_SCALE = 7.0
CURL_CURL_LU_GROWTH_FROM = 2.5e4
CURL_CURL_LU_GROWTH_EXPONENT = 0.3
# run-time estimate, cpml models: Arnoldi operator applications per basis vector per cell along the box (PML modes near
# the target get denser on finer grids; 5 k basis: 457 / 80 at 26^3 cells (Mie sphere example), 307 / 80 at 19^3 (its
# staircase eighth-sphere test grid), ~240 / 80 at 18^3 (from 263 with a 3 k basis); linear fits within the guess range)
APPLICATIONS_PER_BASIS_PER_AXIS_CELL = 0.22
# ... with a mixed factorization (measured on cudss_mixed; the host _mixed ones assumed alike) and no Bloch phase
# (ARPACK stopped at 1e-6, then Rayleigh quotients): 1.0 on the
# eighth nanobeam (14 and 30 eigenpairs), 2.4 on the Mie sphere (16). Since S5a the PML widening (solve_resonances:
# eigenpairs doubled, the device decomposition resumed) is counted in: 4.0 on the Mie example (320 / 80, two passes),
# 2.4 on its tetrahedral twin (192 / 80, two passes); a guess between, per basis of the first pass
MIXED_APPLICATIONS_PER_BASIS = 3.0
# A4: counted host SuperLU bytes (host_counted_lu_bytes of nedelec_factor_entries: the nested-dissection fill; SuperLU
# orders by MMD, so its real factor is larger) of a tetrahedral Nedelec operator from which its host SuperLU factor
# takes tens of seconds: the cpu default warns, and a gpu 'auto' runtime raises with the remedy instead of falling
# back to SuperLU after a cuDSS refusal (em3d_fdfd_tetrahedra, 44k unknowns: gstrf 47.5 s; Mie examples 35-48 s).
# Uncalibrated: compare the check's counted 'eigen solver: SuperLU L + U factors' item with a measured @cpu factor time
NEDELEC_SUPERLU_SLOW_BYTES = 256 * 2 ** 20

# ---- settings ----
_BC_HELP = ('The em_fdtd_3d faces on the same Yee grid: cpml = complex stretch s = kappa + sigma / (alpha + j omega '
            'eps0) of the pml_* settings in pml_cells extra cells, frozen at the target wavelength (resonances near '
            'it see the right stretch); pec = tangential E 0 at the centres of the outermost cell layer (it becomes '
            'conductor); pmc = tangential H 0 on the outer face of the outermost cell (pec and pmc walls lie half a '
            'cell apart); periodic on both faces of an axis, Bloch phase bloch_k<axis>. Mirror planes on the domain '
            'edge itself (any face): symmetric = E mirror-even (PMC plane), antisymmetric = E mirror-odd (PEC plane); '
            'a ghost cell outside holds the image. Each mirror halves the direct-solve unknowns; the eigenfrequency '
            'setting parity_sectors solves every parity of the mirror faces in turn. Unknowns are the live E faces '
            'only (faces between two PEC cells, the normal E on a symmetric plane and ghost cells dropped).')


_BLOCH_HELP = ('Bloch wave vector k along this axis, used only when both its faces are periodic: fields repeat with '
               'exp(-j k L) per period L (the meshed period; exp(+j omega t)); k and k + 2 pi / L are the same state; '
               '0 = plain periodic. The resonances at fixed k are the lattice bands at that k (sweep a parameter for a '
               'band diagram); the operator becomes complex (no extra unknowns). Typical pi / L at the zone edge.')


def _bloch_params():
    return tuple(Param(
        f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
        visible_if=((f'bc_{axis}_lo', ('periodic',)),),
        help=_BLOCH_HELP,   # axis-free: describe prints it once for the three axes
        tip=('Phase step of the wave from one period to the next along this axis, in radians per metre, for periodic '
             'faces.\n\n0 means no phase step. Swept, it gives the bands of a periodic structure.'))
        for axis in AXES)


def _settings():
    """bc_<face> for the six faces, the shared pml_* settings (visible with any cpml face), wavelength (the target) and
    bloch_k<axis>."""
    faces = tuple(Param(f'bc_{face}', 'choice', 'pec', label=FACE_LABELS[face], choices=FACE_TYPES_3D,
                        help=_BC_HELP, tip=_BC_TIP, choice_labels=_BC_CHOICE_LABELS) for face in FACES_3D)
    pml = tuple(replace(param, visible_if=_PML_VISIBLE_3D) for param in boundary_settings_params(_BC_HELP)
                if param.name.startswith('pml_'))
    wavelength = Param(
        'wavelength', 'float', DEFAULT_WAVELENGTH, label='Target wavelength', unit='m', min=0.0, min_exclusive=True,
        help='Vacuum target wavelength lambda_t: the eigenfrequency study returns the resonances nearest it. The '
             'operator is assembled at omega_t (sigma_e and poles as eps(omega_t), PML stretch at omega_t), so a '
             'resonance is exact for non-dispersive PML-free models, else accurate to first order in |omega / omega_t '
             '- 1|: for an accurate Q of an open or dispersive resonator set it near the resonance and re-run. It may '
             'coincide with a resonance (shift 1e-4 above lambda_t^-2). Mesh: >= 10 (better 20) cells per lambda_t / '
             'n. Typical 0.4e-6 to 2e-6 m.',
        tip='Wavelength in metres near which the resonances are searched: 1.55e-6 for 1550 nm.\n\nSet it close to the '
            'resonance you want; the result is most accurate near it.')
    discretization = replace(
        DISCRETIZATION_PARAM,
        tip=('Order of the finite elements on a tetrahedron mesh; ignored on a rectangular-cell mesh.\n\n'
             '- Quadratic: far more accurate for the same mesh, the usual choice.\n'
             '- Linear: cheaper per element, needs a much finer mesh.'),
        help=('Nedelec (first kind) degree on a tetrahedra mesh (ignored on tensor): quadratic (degree 2, 20 '
              'unknowns per element: 2 per edge and face) or linear (one per edge); straight-sided elements (curved '
              'surfaces faceted at the mesh size).'))
    return faces + pml + (wavelength,) + _bloch_params() + (discretization,)


def _domain_requirements(settings_values):
    return DomainRequirements(boundary_config=_boundary_config(settings_values))


# ---- create ----
def _create(ctx, physics_spec, settings_values):
    """FDFDPhysics3D at the wavelength setting with the 'eigen' solver and the dispersive poles of every material region
    (definition.write_dispersive_regions_3d); on a 'tetrahedra' mesh FEMResonancePhysics3D (Nedelec elements of the
    element_order setting)."""
    eigen_solver = ctx.solvers(physics_spec.name)['eigen']
    if ctx.domain.grid_kind == 'tetrahedra':
        physics = FEMResonancePhysics3D(ctx.domain, _boundary_config(settings_values),
                                        wavelength=settings_values['wavelength'], eigen_solver=eigen_solver,
                                        element_order=element_order(settings_values))
        if getattr(eigen_solver, 'fallback_factorization', None) is not None:   # A4: bound the SuperLU fallback
            eigen_solver.fallback_refusal = _nedelec_fallback_refusal(ctx.domain.mesh, element_order(settings_values))
    else:
        physics = FDFDPhysics3D(ctx.domain, wavelength=settings_values['wavelength'], eigen_solver=eigen_solver)
    write_dispersive_regions_3d(ctx, physics)
    materials = ctx.domain.materials
    materials.park([name for name in _parked_material_arrays(ctx.spec, ctx.registry, physics_spec, ctx.parameters)
                    if name in materials.property_names()])
    return physics


def _parked_material_arrays(spec, registry, physics_spec, parameters) -> tuple:
    """MaterialGrid arrays em_fdfd_3d moves to the host at create (tensor grids; the assembly reads them through
    to_host, the renderer's eps_r window and state() through device_array: em3d sub file "em_fdfd_3d"): the em3d
    material arrays under the shared rule (definition_helpers.parkable_material_arrays), minus those whose state field
    a material expression or field_transfer reads (em definition_common.state_arrays_read)."""
    try:
        if spec.mesh_of(physics_spec).kind != 'tensor':
            return ()
    except (KeyError, ValueError, AttributeError):   # reported by validate
        return ()
    arrays = parkable_material_arrays(spec, registry, physics_spec, parameters,
                                      {param.name for param in _MATERIAL_PROPERTIES},
                                      MATERIAL_ARRAYS)
    read = state_arrays_read(spec, registry, physics_spec.name, arrays)
    return tuple(array for array in arrays if array not in read)


# ---- features ----
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency solve, nearest the target first. Result 'table', one row per resonance: "
    'resonance_number (1-based rank by |omega - omega_target|), wavelength = 2 pi C0 / Re(omega) [m], frequency = '
    'Re(omega) / (2 pi) and frequency_imag = Im(omega) / (2 pi) [Hz] (exp(+j omega t): > 0 decays), q_factor = '
    'Re(omega) / (2 Im(omega)) (inf when |Im omega| < 1e-12 Re omega: lossless and closed; negative = growing, '
    'non-physical, usually a PML artefact), pml_fraction (share of the electric energy sum |Re eps| |E|^2 dV on E '
    'faces in PML and padding cells: > 0.5 dropped; above ~1e-2 the PML shapes the resonance: widen the cladding); '
    'extras target_wavelength [m], selected_resonance. With parity_sectors each sector has its own table '
    '(Results.sector_result). Optional domain selection = a region (e.g. a box around the cavity core): the table '
    f'then ends with {REGION_COLUMN[0]} (the same energy share on the E faces of the selected cells, each owning its '
    '+x / +y / +z faces; a mirror-plane face counts with the ghost cell, so a region on a mirror face reads a little '
    'low; on tetrahedra the dofs of the selected elements), and resonances below min_region_fraction are dropped like '
    "PML modes (rows, snapshots and resonance numbers skip them). One region per physics; 'all' is refused.")


def _apply_resonance_table(ctx, solver, feature_spec, values, selection):
    """ResonanceTable marker; with a selection its ResonanceRegion (host bool per full-grid cell, or per element on
    tetrahedra) goes to the solver."""
    if not selection.entities:
        return ResonanceTable()
    if selection.is_all:
        raise ValueError(f"resonance_table '{feature_spec.name}' selects 'all': select the cavity objects")
    tetrahedra = ctx.domain.grid_kind == 'tetrahedra'
    cells = np.zeros(ctx.domain.n_elements if tetrahedra else solver.shape, dtype=bool)
    for name in selection.objects:
        if tetrahedra:
            _, mask = ctx.object_window_mask(name)
            cells |= np.asarray(to_host(mask), dtype=bool)
        else:
            window, mask = ctx.object_box_mask(name)
            cells[window] |= np.asarray(to_host(mask), dtype=bool)
    region = ResonanceRegion(cells=cells, min_fraction=float(values['min_region_fraction']))
    solver.set_resonance_region(region)
    return ResonanceTable(region)


_FEATURES = (
    FeatureType('pec_domain', 'Perfect electric conductor', 'domain_condition', 'domain',
                description='Perfect electric conductor cells, as em_fdtd_3d: every E face between two selected cells '
                            'is dropped from the unknowns (E = 0; the surface lies at the centres of the outermost '
                            "conductor cells), a face between a conductor and another cell takes the other cell's "
                            "permittivity (normal E free). Select objects: 'all' is refused. Staircased (no conformal "
                            'option).',
                summary='Makes the selected objects perfect metal.',
                apply=_apply_pec, validate=_validate_pec, memory_estimate=_pec_memory, grids=_GRIDS),
    FeatureType('resonance_table', 'Resonance table', 'monitor', 'domain', params=(MIN_REGION_FRACTION_PARAM,),
                min_entities=0, result_kind='table', description=_RESONANCE_TABLE_DESCRIPTION,
                summary="Lists the resonances found by the 'Eigenfrequency' study: wavelength, frequency, quality "
                        'factor and how much of each lies in the absorbing layers. With a selection it also lists the '
                        'share inside the selected region and can drop the resonances that live elsewhere.',
                apply=_apply_resonance_table, result=object_result(resonance_table_result, with_solver=True),
                validate=validate_resonance_table, memory_estimate=resonance_region_memory_estimate, grids=_GRIDS),
)


# ---- validation ----
_VALIDATE_EM = make_validate_physics(_SLOT_ARRAYS, time_domain=False, solved_wavelengths=all_solved_wavelengths)


def nedelec_unknowns(mesh, settings_values) -> int:
    """Nedelec unknowns of a TetrahedronMeshEstimate before the walls drop theirs (an upper figure): the edges
    (linear), 2 x (edges + faces) (quadratic)."""
    if element_order(settings_values) == 1:
        return int(mesh.n_edges)
    return 2 * int(mesh.n_edges + mesh.n_faces)


def _size_issues(vctx, physics_spec, settings_values, path):
    """sparse_direct.vector_3d_size_issue of 3 E unknowns per grid cell (PML and ghosts included; PEC / mirror drops
    make it an upper figure), on tetrahedra of the Nedelec unknowns x fem3d.YEE_EQUIVALENT_UNKNOWNS (+ a SuperLU warning
    above NEDELEC_SUPERLU_SLOW_BYTES of counted fill on a built mesh); none for a cudss eigen factorization."""
    mesh = vctx.mesh
    factorization = eigen_factorization(vctx.solvers(physics_spec))
    if mesh is None or factorization in CUDSS_FACTORIZATIONS:
        return []
    if mesh.kind == 'tetrahedra':
        order = element_order(settings_values)
        unknowns = nedelec_unknowns(mesh, settings_values)
        issue = vector_3d_size_issue(int(unknowns * YEE_EQUIVALENT_UNKNOWNS[order]))
        issues = [] if issue is None else [Issue(issue[0], path, f'{issue[1]} (~{unknowns:,} Nedelec unknowns of '
                                                                 f'degree {order}, each counted as '
                                                                 f'{YEE_EQUIVALENT_UNKNOWNS[order]:g} Yee unknowns)')]
        # A4: SuperLU (the cpu Automatic) bounded by the counted fill of a built mesh
        counted = nedelec_factor_entries(mesh, order, unknowns) if factorization in ('superlu', 'auto') else None
        slow = '' if counted is None or (issue is not None and issue[0] == 'error') else _slow_superlu_text(
            counted, unknowns)
        if slow:
            issues.append(Issue('warning', path, f'eigen factorization superlu: {slow}'))
        return issues
    if mesh.kind != 'tensor':
        return []
    issue = vector_3d_size_issue(3 * int(mesh.n_cells))
    return [] if issue is None else [Issue(issue[0], path, f'{issue[1]} (3 per grid cell, an upper figure)')]


def _tetrahedra_issues(vctx, physics_spec, settings_values, path):
    """On a 'tetrahedra' mesh: periodic faces need mesher 'builtin' without swept layers (the Nedelec dofs need the
    same node order on both sides; gmsh periodic surfaces and swept footprints do not keep it)."""
    try:
        mesh_spec = vctx.spec.mesh_of(physics_spec)
    except KeyError:   # reported by validate
        return []
    periodic = [axis for axis in AXES if settings_values[f'bc_{axis}_lo'] == 'periodic']
    if mesh_spec.kind == 'tetrahedra' and getattr(mesh_spec, 'sweep_axis', 'none') == 'z':
        in_plane = [axis for axis in periodic if axis != 'z']   # swept layers: z sides match node for node
        return [Issue('error', path, f"periodic faces ({', '.join(in_plane)}) on swept prism layers (sweep_axis 'z') "
                                     'pair the two sides, but the Nedelec edge dofs need the same node order on both: '
                                     "use sweep_axis 'none' with mesher 'builtin'")] if in_plane else []
    if mesh_spec.kind != 'tetrahedra' or not periodic or mesh_spec.mesher == 'builtin':
        return []
    return [Issue('error', path, f"periodic faces ({', '.join(periodic)}) on tetrahedra need mesher 'builtin' (its Kuhn "
                                 "split matches the two sides node for node); mesher 'gmsh' pairs the sides but not "
                                 'in the same node order, which the Nedelec edge dofs need: use the builtin mesher, '
                                 'mesh kind tensor or mirror / cpml faces')]


def _cutoff_issues(vctx, physics_spec, settings_values, path):
    """Warning when the target lies beyond 2 n_max L (L: the largest window side, doubled across a mirror face) of a
    model without periodic axes: every resonance then lies at shorter wavelengths, far from the frozen target."""
    sizes = domain_sizes(vctx)
    wavelength = settings_values['wavelength']
    if sizes is None or any(settings_values[f'bc_{axis}_lo'] == 'periodic' for axis in AXES):
        return []
    lengths = [size * (2.0 if any(settings_values[f'bc_{axis}_{side}'] in MIRROR_TYPES_3D for side in ('lo', 'hi'))
                       else 1.0) for axis, size in zip(AXES, sizes)]
    material_params = vctx.registry.physics_type(physics_spec.type).material_properties
    indices = [_refractive_index(material, vctx.parameters, [material_params]) for material in vctx.spec.materials
               if material.selection]
    n_max = max([1.0] + [index for index in indices if math.isfinite(index)])
    longest = 2.0 * n_max * max(lengths)
    if not wavelength > longest:
        return []
    return [Issue('warning', path, f'the target wavelength {wavelength:.4g} m exceeds 2 n_max L = {longest:.4g} m of '
                                   f'the whole model (n_max {n_max:.3g}, largest side L {max(lengths):.4g} m): every '
                                   'resonance lies at a shorter wavelength, where the frozen dispersion and PML are '
                                   'inaccurate; set the wavelength near the expected resonance')]


def _validate(vctx, physics_spec, settings_values):
    path = ('physics', physics_spec.name)
    return (_validate_z_faces(settings_values, path)
            + list(_VALIDATE_EM(vctx, physics_spec, settings_values)) + _ignored_bloch_issues(settings_values, path)
            + _size_issues(vctx, physics_spec, settings_values, path)
            + _tetrahedra_issues(vctx, physics_spec, settings_values, path)
            + _cutoff_issues(vctx, physics_spec, settings_values, path))


# ---- memory and run time ----
def _memory_estimate(settings_values, mesh, context):
    """Device: only the eigen solve's items (the six complex64 resonance fields are host: fdfd3d header). Host (3 E
    unknowns per cell): 3D SuperLU (or a cuDSS factor: external, its copies EIGEN_PHASE items; the EIGEN_ROLE job's),
    operators, curls, Arnoldi work, vectors, materials, the fields. Tetrahedra: _tetrahedra_memory_estimate."""
    if mesh.kind == 'tetrahedra':
        return _tetrahedra_memory_estimate(settings_values, mesh, context)
    n_cells = int(mesh.n_cells)
    config = _boundary_config(settings_values)
    unknowns = live_unknowns_estimate(mesh, config)   # A5: the live E faces, not 3 x cells (Mie example: 11 % high)
    n_resonances, n_wanted, n_basis = _arnoldi_sizes(context, unknowns)
    vector_bytes = COMPLEX128_BYTES * unknowns
    row_bytes = INT32_BYTES * unknowns   # CSR indptr per row
    factorization = eigen_factorization(context.solvers)
    general = config.has_bloch_phase   # a Bloch phase: an LU of the whole matrix
    # A5: counted fill of the box nested dissection (SuperLU factors in that order, operators3d
    # nested_dissection_ordering; cuDSS's own ND as FactorFill) instead of the n^(4/3) growth laws; MUMPS / PARDISO:
    # the LDL^T of D A (LU with a Bloch phase) at that SuperLU count x superlu_bytes_scale. cuDSS orders the unknowns
    # itself: its plane separators hold the two tangential E components only (yee_faces; the cell-wise count read the
    # nanobeam eighth model's factor 54 % high)
    grid_shape = (int(mesh.nx), int(mesh.ny), int(mesh.nz))
    counted = _box_fill(grid_shape, unknowns / max(n_cells, 1))
    cudss_counted = _box_fill(grid_shape, unknowns / max(n_cells, 1), True)
    factor_items = () if factorization in CUDSS_FACTORIZATIONS else tuple(solver_job_items(
        EIGEN_ROLE, host_factor_items(
            factorization, 'SuperLU L + U factors (3D curl-curl, counted box nested dissection)',
            host_counted_lu_bytes(counted, unknowns), unknowns, _ROW_ENTRIES * unknowns, symmetric=not general,
            eigen=True)))
    from ...solvers.cudss_factor import FactorFill
    cudss_items, external_items, external_extra = _cudss_eigen_items(factorization, unknowns, _ROW_ENTRIES * unknowns,
                                                                     general, 'the curl-curl operator',
                                                                     fill=FactorFill(cudss_counted))
    # the Arnoldi basis (device Krylov-Schur where it runs, first pass only; a polished search keeps complex64 rows:
    # _polished_search) and the filter's device A copy (fdfd3d passes no symmetric: a full A; float64 when real); the
    # filter's D v block only where the search is polished (its row scale = the symmetrizer, on any LDL^T back end)
    polished = _polished_search(factorization, unknowns, not general, config)
    search_host, search_transient = resonance_search_items(solver_backend(context.spec), unknowns, n_wanted, n_basis,
                                                           own_cudss=factorization in CUDSS_FACTORIZATIONS,
                                                           filter_entries=_ROW_ENTRIES * unknowns,
                                                           scaled_block=polished, single_basis=polished,
                                                           real_values=_real_values(settings_values, context, config))
    host_items = factor_items + (
        (f'operator A and its CSC copy ({_ROW_ENTRIES} entries per row)',
         2 * (_ROW_ENTRIES * COMPLEX128_CSR_ENTRY_BYTES * unknowns + row_bytes)),
        ('Yee curls curl_e / curl_h and the curl product',
         2 * (_CURL_ENTRIES * COMPLEX128_CSR_ENTRY_BYTES * unknowns + row_bytes)
         + _ROW_ENTRIES * COMPLEX128_CSR_ENTRY_BYTES * unknowns + row_bytes),
        *search_host,
        (f'resonance vectors ({n_resonances} x complex128)', n_resonances * vector_bytes),
        ('system vectors (face eps, 1 / mu, mass, weights, solve vectors)', 8 * vector_bytes),
        # 3 background eps + the complex loss term + 3 complex cell eps (+ the smoothing deltas, 3 per cell)
        ('host material copies (float64 / complex128)', 12 * FLOAT64_BYTES * n_cells),
        # fdfd3d: the selected resonance's E / H stay on the host on either backend (renderer: one float32 upload)
        ('resonance fields (6 x complex64, host)', 6 * COMPLEX64_BYTES * n_cells),
    )
    # parked MaterialGrid arrays (meshing books them nowhere else): on the device while the build paints them, on the
    # host from create
    parked = [name for name in _parked_material_arrays(context.spec, context.registry, context.physics_spec,
                                                       context.parameters) if name in _allocated_arrays(context)]
    parked_bytes = FLOAT32_BYTES * len(parked) * n_cells
    if parked:
        host_items += ((f"material arrays parked on the host ({', '.join(parked)}, float32)", parked_bytes),)
    return MemoryEstimate.from_items((), build_extra=parked_bytes, host_items=host_items,
                                     transient_items=cudss_items + tuple(search_transient),
                                     external_items=external_items, external_extra=external_extra)


def live_unknowns_estimate(mesh, config) -> int:
    """Live E unknowns of a tensor MeshEstimate (CPU): operators3d.live_unknowns_3d's layer rules without PEC objects
    (an upper figure). Open hi faces, pec wall layers and mirror ghosts / planes drop whole layers per component and
    axis, so the count is a product per component; a change of those rules changes both."""
    from .boundaries3d import mirror_planes
    shape = (int(mesh.nx), int(mesh.ny), int(mesh.nz))
    window = tuple(slice(int(mesh.npml_sides.get(f'{name}_lo', 0)), count - int(mesh.npml_sides.get(f'{name}_hi', 0)))
                   for name, count in zip(AXES, shape))
    dropped = [[set() for _ in AXES] for _ in AXES]   # [component][axis] -> dropped layer indices
    for axis, name in enumerate(AXES):
        if not config.periodic(name):
            dropped[axis][axis].add(shape[axis] - 1)   # the never-updated hi face
        for side in ('lo', 'hi'):
            if getattr(config, f'{name}_{side}') == 'pec':   # boundaries3d.pec_wall_layers: tangential E between
                layer = window[axis].start if side == 'lo' else window[axis].stop - 1   # two PEC cells
                for component in range(3):
                    if component != axis:
                        dropped[component][axis].add(layer)
    for axis, kind, ghost, _inner, plane_face in mirror_planes(config, window):
        for component in range(3):
            if component != axis or kind == 'symmetric':
                dropped[component][axis].add(ghost)
        if kind == 'symmetric':
            dropped[axis][axis].add(plane_face)
    return sum(math.prod(count - len({index for index in dropped[component][axis] if 0 <= index < count})
                         for axis, count in enumerate(shape)) for component in range(3))


@lru_cache(maxsize=16)
def _box_fill(shape, unknowns_per_cell, yee_faces=False):
    """sparse_direct.box_dissection_entries of a grid shape (cached: a Python loop over the dissection boxes)."""
    from ...solvers.sparse_direct import box_dissection_entries
    return box_dissection_entries(shape, unknowns_per_cell, yee_faces=yee_faces)


def _arnoldi_sizes(context, unknowns):
    """(n_resonances, wanted eigenpairs 2 n + 10, Arnoldi basis) of the study (fdfd3d ARNOLDI_BASIS_PER_WANTED)."""
    n_resonances = int(_study_values(context).get('n_resonances') or 1)
    n_wanted = max(min(2 * n_resonances + 10, unknowns - 2), 1)
    n_basis = arnoldi_basis_size(n_wanted, unknowns, eigen_subspace_size(context.solvers),
                                 ARNOLDI_BASIS_PER_WANTED * n_wanted)
    return n_resonances, n_wanted, n_basis


def _polished_search(factorization, unknowns, ldlt, config) -> bool:
    """Whether eigenfrequency.solve_resonances polishes an em_fdfd_3d search (and so keeps a complex64 device basis,
    single_basis): an own LDL^T (ldlt: no complex Bloch phase; LDLT_FACTORIZATIONS) that is mixed (from
    MIXED_EIGEN_MIN_UNKNOWNS on; below it a mixed request factors in double) or of a complex A whose ARPACK fdfd3d /
    fem3d floor (any cpml face: ARPACK_TOLERANCE_FLOOR, the stretch makes A complex)."""
    if not ldlt or factorization not in LDLT_FACTORIZATIONS:
        return False
    mixed = factorization in MIXED_FACTORIZATIONS and int(unknowns) >= MIXED_EIGEN_MIN_UNKNOWNS
    return mixed or bool(config.needs_cpml)


def _real_values(settings_values, context, config) -> bool:
    """Whether the resonance filter's A and B have no imaginary part (device_matrices float64 storage): no cpml face, no
    Bloch phase, no sigma_e / sigma_m and no Drude / Lorentz poles (the materials test of em_modes' cross-section)."""
    return (not config.needs_cpml and not config.has_bloch_phase
            and real_cross_section_operator(settings_values, context))


def _cudss_eigen_items(factorization, unknowns, entries, general, label, fill=3, sparse_mass=False):
    """(EIGEN_PHASE / FILTER_PHASE items, external items, external extra) of solve_resonances' own cuDSS factor (freed
    after the solve; the EIGEN_ROLE job's): CSR copies (LDL^T lower triangle; Bloch: the whole LU matrix) in both phases,
    solve vectors in the eigen solve; nothing without cuDSS. fill: the 3D growth law (3) or a counted
    cudss_factor.FactorFill. sparse_mass (FEM): B is booked by resonance_search_items (sparse_mass_entries), the solves
    add only the B v vector."""
    if factorization not in CUDSS_FACTORIZATIONS:
        return (), [], 0
    mixed = eigen_mixed(factorization, unknowns)
    vector_bytes = COMPLEX128_BYTES * unknowns
    b_item = ((EIGEN_PHASE, 'B v vector (complex128; the sparse B: resonance search items)', vector_bytes)
              if sparse_mass else (EIGEN_PHASE, 'B diagonal and applied vector (complex128)', 2 * vector_bytes))
    items = tuple(own_cudss_search_items(label, unknowns, entries, mixed, symmetric=not general)
                  + solver_job_items(EIGEN_ROLE, (b_item,)))
    external_items, external_extra = cudss_factor_items(label, unknowns, entries, dimensions=fill, mixed=mixed,
                                                        general=general)
    return items, solver_job_items(EIGEN_ROLE, external_items), external_extra


def nedelec_factor_entries(mesh, order, unknowns):
    """Counted entries of one factor triangle of the Nedelec operator on a built mesh (MeshEstimate.dissection: dof-site
    pairs x dofs per site^2, scaled by the factored share of the dofs); None on size-based counts (no mesh to count)."""
    dissection = getattr(mesh, 'dissection', None)
    if dissection is None:
        return None
    return _counted_nedelec_entries(dissection, order, unknowns)


def _counted_nedelec_entries(dissection, order, unknowns):
    """nedelec_factor_entries of a fem.dissection.MeshDissection."""
    dofs_per_site = 2 if order == 2 else 1
    share = min(float(unknowns) / max(dofs_per_site * dissection.n_nedelec_sites(order), 1), 1.0)
    return dofs_per_site ** 2 * dissection.nedelec_pairs(order) * share


def _slow_superlu_text(entries, unknowns):
    """'' or the A4 text when the counted host SuperLU factor of `unknowns` Nedelec unknowns (counted triangle entries)
    exceeds NEDELEC_SUPERLU_SLOW_BYTES."""
    lu_bytes = host_counted_lu_bytes(entries, unknowns)
    if lu_bytes <= NEDELEC_SUPERLU_SLOW_BYTES:
        return ''
    return (f'the host SuperLU factor of the {int(unknowns):,}-unknown Nedelec operator holds >= '
            f'{lu_bytes / 2 ** 30:.2f} GiB (counted nested-dissection fill; SuperLU orders by MMD, more) and takes tens '
            'of seconds to minutes (44k unknowns: 47.5 s); use the gpu backend (Automatic: cudss_mixed), an eigen '
            f'solver node with factorization {HOST_DIRECT_NAMES} (host LDL^T), or coarsen the mesh / halve it by '
            'mirror faces')


def _nedelec_fallback_refusal(mesh, order):
    """ArpackShiftInvertSolver.fallback_refusal of a tetrahedral em_fdfd_3d eigen runtime (A4): the counted fill of
    the built TetrahedronMesh, counted only when a cuDSS refusal would fall back."""
    def refusal(n_unknowns):
        from ...fem.dissection import MeshDissection
        return _slow_superlu_text(_counted_nedelec_entries(MeshDissection(mesh), order, n_unknowns), n_unknowns)
    return refusal


def _tetrahedra_memory_estimate(settings_values, mesh, context):
    """Host (Nedelec unknowns, _NEDELEC_ROW_ENTRIES per row): SuperLU counted on the built mesh (nedelec_factor_entries;
    size-based counts: the 3D heuristic at the Yee-equivalent unknowns) or a cuDSS factor (external, copies EIGEN_PHASE
    items), assembly blocks, CSR matrices, Arnoldi work, vectors."""
    order = element_order(settings_values)
    unknowns = nedelec_unknowns(mesh, settings_values)
    counted = nedelec_factor_entries(mesh, order, unknowns)
    row_entries = _NEDELEC_ROW_ENTRIES[order]
    n_local = 6 if order == 1 else 20
    n_elements = int(mesh.n_elements)
    n_resonances, n_wanted, n_basis = _arnoldi_sizes(context, unknowns)
    vector_bytes = COMPLEX128_BYTES * unknowns
    matrix_bytes = row_entries * COMPLEX128_CSR_ENTRY_BYTES * unknowns + INT32_BYTES * unknowns
    factorization = eigen_factorization(context.solvers)
    config = _boundary_config(settings_values)
    symmetric = not config.has_bloch_phase   # MUMPS / PARDISO: LDL^T of the pencil
    if factorization in CUDSS_FACTORIZATIONS:
        factor_items = ()
    elif counted is not None:
        factor_items = tuple(solver_job_items(EIGEN_ROLE, host_factor_items(
            factorization, 'SuperLU L + U factors (counted on the built mesh: nested dissection of the Nedelec dof '
            'sites)', host_counted_lu_bytes(counted, unknowns), unknowns, row_entries * unknowns,
            symmetric=symmetric, eigen=True)))
    else:
        factor_items = tuple(solver_job_items(EIGEN_ROLE, host_factor_items(
            factorization, 'SuperLU L + U factors (3D heuristic at the Yee-equivalent unknowns)',
            host_lu_bytes(int(unknowns * YEE_EQUIVALENT_UNKNOWNS[order]), dimensions=3), unknowns,
            row_entries * unknowns, symmetric=symmetric, eigen=True)))
    fill = 3
    if counted is not None:
        from ...solvers.cudss_factor import FactorFill
        fill = FactorFill(counted)
    # a Bloch phase makes A nonsymmetric: the LU keeps the whole matrix, not the LDL^T lower triangle
    cudss_items, external_items, external_extra = _cudss_eigen_items(
        factorization, unknowns, row_entries * unknowns, not symmetric, 'the Nedelec curl-curl operator', fill,
        sparse_mass=True)
    # the Arnoldi basis (device Krylov-Schur where it runs, first pass only); the filter uploads A, W (and B without an
    # own factor) one at a time: one pattern of row_entries per row each; an own factor's solves and the filter share B
    search_host, search_transient = resonance_search_items(solver_backend(context.spec), unknowns, n_wanted, n_basis,
                                                           own_cudss=factorization in CUDSS_FACTORIZATIONS,
                                                           filter_entries=row_entries * unknowns,
                                                           scaled_block=False,   # fem3d: no symmetrizer, no D v
                                                           sparse_mass_entries=row_entries * unknowns,
                                                           symmetric_mass=symmetric, symmetric_operator=symmetric,
                                                           single_basis=_polished_search(factorization, unknowns,
                                                                                         symmetric, config),
                                                           real_values=_real_values(settings_values, context, config))
    host_items = factor_items + (
        (f'element assembly blocks ({n_local} x {n_local} per element: rows, columns, K and M)',
         n_elements * n_local * n_local * (2 * FLOAT64_BYTES + 2 * COMPLEX128_BYTES)),   # int64 indices
        (f'operator A, K, M, energy W and their free-dof slices (~{row_entries} entries per row)', 6 * matrix_bytes),
        *search_host,
        (f'resonance vectors ({n_resonances} x complex128)', n_resonances * vector_bytes),
        ('system vectors (equilibration, window flags, energies, solve vectors)', 6 * vector_bytes),
        ('host element materials (complex eps x 3, 1 / mu, PEC)', n_elements * (4 * COMPLEX128_BYTES + 1)),
    )
    return MemoryEstimate.from_items((), host_items=host_items, transient_items=cudss_items + tuple(search_transient),
                                     external_items=external_items, external_extra=external_extra)


def _time_estimate(settings_values, mesh, context):
    """One host assembly and one 'eigen' shift-invert (2 n_resonances + 10 eigenpairs, a factorisation) on 3 E unknowns
    per cell; the squared Cayley map does two triangular solves per application (rhs 2), basis as fdfd3d, the factor
    and (cpml) the applications per basis vector growing with the grid as fitted above (a mixed factorization: a flat
    guess).
    Tetrahedra: the same recipe at the Yee-equivalent unknowns of the Nedelec unknowns (uncalibrated; the per-axis
    PML growth taken at the cube root of the elements)."""
    if mesh.kind == 'tetrahedra':
        unknowns = float(nedelec_unknowns(mesh, settings_values)
                         * YEE_EQUIVALENT_UNKNOWNS[element_order(settings_values)])
    else:
        unknowns = 3.0 * float(mesh.n_cells)
    nnz = _ROW_ENTRIES * unknowns
    eigenpairs = resonance_eigenpairs(context)
    factor_scale = (CURL_CURL_LU_SCALE
                    * max(unknowns / CURL_CURL_LU_GROWTH_FROM, 1.0) ** CURL_CURL_LU_GROWTH_EXPONENT)
    config = _boundary_config(settings_values)
    factorization = eigen_factorization(context.solvers)
    # the polished path (ARPACK floor 1e-6, complex64 device basis): a mixed factor (small operators factor in double,
    # S5c) or, with cpml (a complex A, floored), any LDL^T of D A (solve_resonances polishes a double one there too)
    # the mixed-size rule (MIXED_EIGEN_MIN_UNKNOWNS) reads the factored unknowns: the Nedelec ones on tetrahedra, not
    # their Yee equivalent
    factored = float(nedelec_unknowns(mesh, settings_values)) if mesh.kind == 'tetrahedra' else unknowns
    polished = _polished_search(factorization, factored, not config.has_bloch_phase, config)
    if polished:
        applications_per_basis = MIXED_APPLICATIONS_PER_BASIS
    else:
        applications_per_basis = (APPLICATIONS_PER_BASIS_PER_AXIS_CELL * float(mesh.n_cells) ** (1.0 / 3.0)
                                  if config.needs_cpml else 0.0)
    assembly = te.TimeItem('curl-curl assembly', 'assembly', 1.0, 'exact',
                           (te.HostStream(ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200)),
                           counter=physics_counter(context, 'assemblies'))
    # S9: the resonance filter per eigenpair of a pass (eigenfrequency._ResonanceFilter: A, and on tetrahedra the
    # sparse M and W, applied once per column block; was the "rest of the study": 628 ms against 16 ms on the Mie example)
    candidate_filter = resonance_filter_time_item(   # tetrahedra without a Bloch phase: a symmetric search
        context.backend, unknowns, 3.0 * nnz if mesh.kind == 'tetrahedra' else nnz, eigenpairs,
        symmetric=mesh.kind == 'tetrahedra' and not config.has_bloch_phase, single_basis=polished)
    return te.combine((te.TimeEstimate((assembly, candidate_filter)),
                       context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=COMPLEX128_BYTES,
                                           dimensions=3, solves=1.0, factorizations=1.0, eigenpairs=eigenpairs,
                                           rhs=2.0, default_basis=ARNOLDI_BASIS_PER_WANTED * eigenpairs,
                                           factor_scale=factor_scale,
                                           applications_per_basis=applications_per_basis,
                                           device_arnoldi=factorization in CUDSS_FACTORIZATIONS,
                                           single_basis=polished)))


# ---- type ----
_DESCRIPTION = (
    '3D finite-difference frequency-domain eigensolver on the em_fdtd_3d Yee grid (E on cell faces, H on edges). The '
    'eigenfrequency study solves curl (mu^-1 curl E) = (omega / C0)^2 eps E for the resonances nearest the target '
    'wavelength: shift-invert of the operator assembled there (materials and PML frozen) by the eigen factorization '
    f'(host SuperLU, cudss / cudss_mixed on the GPU, or {HOST_DIRECT_NAMES} on the host, the last two the LDL^T of D '
    'A), ARPACK on the squared Cayley map (static gradient solutions sent to 0), on the live E faces only (Bloch phases '
    'in the wrap entries). Materials as em_fdtd_3d, as eps(omega_t) = eps - j sigma_e / (omega_t eps0) + chi(omega_t) '
    '(face means of the two cells; solver_defaults.subpixel_smoothing: the em_fdtd_3d anisotropic average on '
    'interface faces, so curved surfaces converge smoothly) and mu_r - j sigma_m / (omega_t mu0) of the lower cell '
    'per H edge. Outputs of the selected resonance (scaled to 1 V/m real at its largest E sample): ex / ey / ez / hx / '
    'hy / hz _re / _im / _abs (H = j curl E / (omega mu0 mu) at the complex omega), e_abs (primary), e_intensity, '
    'h_abs (cell centres), w_e = eps0 / 4 sum eps_c |E_c|^2 and eps_r, shown as cut planes. Memory: host dominates, '
    'the 3D factor grows ~ n^(4/3), n = E unknowns (~3 per cell; a solver size rule warns above 1e6 and refuses 4e6); '
    'the device holds the eigen solve only: the 48 B per cell of resonance fields and (GPU, uncoupled, constant '
    'materials) the material arrays stay on the host. heat_em_thermo_optic writes eps_r (+ eps_xx / eps_yy) '
    "as a background coupling. 'tetrahedra' mesh (finite elements): Nedelec edge elements (discretization), one "
    'material value per element, cpml faces as a PML shell of pml_cells x the maximum element size meshed with the '
    'window (complex stretch tensor), the same eigen solve (sparse mass), outputs sampled at points (cut planes on a '
    'uniform sample lattice, derived values at quadrature points); periodic faces (Bloch k: hi-side unknowns = lo '
    'side times the phase, nonsymmetric LU) need the builtin mesher; no subpixel smoothing.')

EM_FDFD_3D = PhysicsType(
    name=TYPE_NAME,
    label='Electromagnetic Waves, frequency domain (3D)',
    family=EM_FAMILY,
    group='em',
    default_instance_name='em',
    description=_DESCRIPTION,
    summary=('Resonances of a 3D optical cavity, solved directly near a chosen wavelength: resonance wavelength, '
             'quality factor and the mode shape. Use it for photonic-crystal and nanobeam cavities, resonators and '
             'Bloch bands of periodic structures.'),
    settings=_settings(),
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=_MATERIAL_SCHEMA,
    optional_material_schema=_OPTIONAL_MATERIAL_SCHEMA,
    state_fields=(FDFDRenderer3D.field_infos['eps_r'],),
    output_fields=tuple(FDFDRenderer3D.field_infos.values()),
    source_slots=source_slot_docs(_SLOT_ARRAYS),
    supported_studies=(EIGENFREQUENCY_STUDY,),
    default_study={'type': EIGENFREQUENCY_STUDY, 'settings': {'n_resonances': 6}},
    uses_dispersion=True,
    grids=_GRIDS,
    create=_create,
    domain_requirements=_domain_requirements,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate,
    memory_estimate=_memory_estimate,
    parked_material_arrays=_parked_material_arrays,
    time_estimate=_time_estimate,
    dimensions=(3,),
    subpixel_smoothing=True,
    # Automatic on the gpu: cudss_mixed, the examples' and self-tests' node (host SuperLU took 35-48 s to factor the Mie
    # examples, against ~1 s); a cuDSS refusal falls back to SuperLU (B1b)
    solver_roles={'eigen': eigen_role((EIGENFREQUENCY_STUDY,), mixed_factorization=True,
                                      auto_factorizations={'tensor': 'cudss_mixed', 'tetrahedra': 'cudss_mixed'})},
    backends=('gpu', 'cpu'),   # host assembly and solve; fields on the Domain's xp
)

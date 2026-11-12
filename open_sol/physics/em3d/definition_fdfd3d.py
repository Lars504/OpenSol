"""Model definition of the 3D FDFD physics type 'em_fdfd_3d' (igor3d_plan stages 1-2): settings, the em_fdtd_3d material
vocabulary, features (PEC, resonance table), validation and the memory / run-time estimates; registered by
definition.register after em_fdtd_3d."""
import math
from dataclasses import replace

import numpy as np

from ...core import time_estimate as te
from ...core.backend import to_host
from ...core.constants import FACE_LABELS, FACES_3D
from ...core.memory_estimate import (COMPLEX64_BYTES, COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT64_BYTES,
                                     INT32_BYTES, MemoryEstimate, host_counted_lu_bytes, host_lu_bytes)
from ...model.definition_helpers import (DISCRETIZATION_PARAM, eigen_factorization, eigen_role, eigen_subspace_size,
                                         element_order)
from ...model.schema import DomainRequirements, FeatureType, Issue, Param, PhysicsType, solver_backend
from ...solvers.arpack_shift_invert import (CUDSS_FACTORIZATIONS, LDLT_FACTORIZATIONS, MIXED_EIGEN_MIN_UNKNOWNS,
                                            MIXED_FACTORIZATIONS, arnoldi_basis_size)
from ...solvers.sparse_direct import vector_3d_size_issue
from ..em.definition_common import (DEFAULT_WAVELENGTH, EM_FAMILY, _refractive_index, all_solved_wavelengths,
                                    boundary_settings_params, make_validate_physics, object_result, physics_counter,
                                    HOST_DIRECT_NAMES, resonance_eigenpairs, shortest_solved_wavelength,
                                    source_slot_docs)
from ..em.definition_fdfd import (MIN_REGION_FRACTION_PARAM, resonance_region_memory_estimate,
                                  validate_resonance_table)
from ..em.eigenfrequency import (REGION_COLUMN, ResonanceRegion, ResonanceTable, resonance_filter_time_item,
                                 resonance_table_result)
from ..em.memory_estimates import (EIGEN_PHASE, _study_values, cudss_factor_items, cudss_own_factor_bytes, eigen_mixed,
                                   host_factor_items, resonance_search_items)
from .boundaries3d import AXES, FACE_TYPES_3D, MIRROR_TYPES_3D
from .definition import (_BC_CHOICE_LABELS, _BC_TIP, _MATERIAL_PROPERTIES, _MATERIAL_SCHEMA,
                         _OPTIONAL_MATERIAL_SCHEMA, _PML_VISIBLE_3D, _SLOT_ARRAYS, _apply_pec, _boundary_config,
                         _ignored_bloch_issues, _pec_memory, _validate_pec, _validate_z_faces,
                         write_dispersive_regions_3d)
from .definition_sources3d import domain_sizes
from .fdfd3d import ARNOLDI_BASIS_PER_WANTED, FDFDPhysics3D
from .fem3d import YEE_EQUIVALENT_UNKNOWNS
from .fem_solver3d import FEMResonancePhysics3D
from .renderer_fdfd3d import FDFDRenderer3D

TYPE_NAME = 'em_fdfd_3d'
_EIGENFREQUENCY_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
_SELECT_PHASE = 'select resonance'          # run phase (MemoryEstimate.transient_items) of select_resonance's upload
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

# ---- settings ----
_BC_HELP = ('Boundary of this face of the 3D box, the em_fdtd_3d faces on the same Yee grid: cpml = the complex '
            'coordinate stretch s = kappa + sigma / (alpha + j omega eps0) of the pml_* settings in pml_cells extra '
            'cells outside the domain, evaluated at the target wavelength (frozen: the resonances near it see the '
            'right stretch); pec = tangential E 0 at the centres of the outermost domain cell layer (that layer becomes '
            'perfect conductor); pmc = tangential H 0 on the outer face of the outermost cell (the grid edge: pec and '
            'pmc walls lie half a cell apart); periodic on both faces of an axis, with the Bloch phase bloch_k<axis> '
            '(0 = plain periodic). Mirror planes (any face, E parity as em_fdtd_3d): symmetric = E mirror-even '
            '(tangential H and normal E 0 on the plane, a PMC mirror), antisymmetric = E mirror-odd (tangential E odd: '
            'a PEC mirror), on the domain edge itself (one ghost cell is added outside it); each mirror halves the '
            'unknowns of the direct solve, and the eigenfrequency study setting parity_sectors solves every parity of '
            'the mirror faces in turn. Unknowns are the live E faces only: faces between two PEC cells, the normal E on '
            'a symmetric plane and the ghost cells are dropped.')


def _bloch_params():
    return tuple(Param(
        f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
        visible_if=((f'bc_{axis}_lo', ('periodic',)),),
        help=(f'Bloch wave vector along {axis} in rad/m, used only when bc_{axis}_lo and bc_{axis}_hi are periodic: '
              f'the fields repeat with the phase exp(-j k{axis} L) per period L (the meshed period, exp(+j omega t)); '
              f'k{axis} and k{axis} + 2 pi / L are the same state; 0 = plain periodic. The resonances at fixed k are '
              'the bands of the lattice at that k (sweep a parameter for a band diagram); the operator becomes complex '
              '(no extra unknowns). Typical: pi / L at the zone edge.'),
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
        help='Vacuum wavelength lambda_t of the target (omega_t = 2 pi C0 / lambda_t): the eigenfrequency study returns '
             'the resonances nearest it. The operator is assembled there: sigma_e and the Drude / Lorentz poles enter '
             'as eps(omega_t), the PML as its stretch at omega_t, so a resonance is exact for non-dispersive, PML-free '
             'models and accurate to first order in |omega / omega_t - 1| otherwise: for an accurate Q of an open or '
             'dispersive resonator set it near the resonance and re-run. It may coincide with a resonance (the '
             'shift-invert shift sits 1e-4 above lambda_t^-2). The mesh needs >= 10 (better 20) cells per lambda_t / n. '
             'Typical 0.4e-6 to 2e-6 m.',
        tip='Wavelength in metres near which the resonances are searched: 1.55e-6 for 1550 nm.\n\nSet it close to the '
            'resonance you want; the result is most accurate near it.')
    discretization = replace(
        DISCRETIZATION_PARAM,
        tip=('Order of the finite elements on a tetrahedron mesh; ignored on a rectangular-cell mesh.\n\n'
             '- Quadratic: far more accurate for the same mesh, the usual choice.\n'
             '- Linear: cheaper per element, needs a much finer mesh.'),
        help=('Nedelec (edge element, first kind) degree on a tetrahedron mesh (MeshSpec kind tetrahedra; ignored on '
              'tensor meshes): quadratic (degree 2, 20 unknowns per element: 2 per edge and face) or linear (degree '
              '1, one per edge); straight-sided elements (curved surfaces are faceted at the mesh size).'))
    return faces + pml + (wavelength,) + _bloch_params() + (discretization,)


def _domain_requirements(settings_values):
    return DomainRequirements(boundary_config=_boundary_config(settings_values))


# ---- create ----
def _create(ctx, physics_spec, settings_values):
    """FDFDPhysics3D at the wavelength setting with the 'eigen' solver and the dispersive poles of every material region
    (definition.write_dispersive_regions_3d); on a 'tetrahedra' mesh FEMResonancePhysics3D (Nedelec elements of the
    element_order setting, igor3d_plan T4b)."""
    eigen_solver = ctx.solvers(physics_spec.name)['eigen']
    if ctx.domain.grid_kind == 'tetrahedra':
        physics = FEMResonancePhysics3D(ctx.domain, _boundary_config(settings_values),
                                        wavelength=settings_values['wavelength'], eigen_solver=eigen_solver,
                                        element_order=element_order(settings_values))
    else:
        physics = FDFDPhysics3D(ctx.domain, wavelength=settings_values['wavelength'], eigen_solver=eigen_solver)
    write_dispersive_regions_3d(ctx, physics)
    return physics


# ---- features ----
_RESONANCE_TABLE_DESCRIPTION = (
    "Resonances of the last eigenfrequency solve, nearest the target (the wavelength setting) first. Result 'table', "
    'one row per resonance: resonance_number (1-based rank by |omega - omega_target|), wavelength = 2 pi C0 / Re(omega) '
    'in m, frequency = Re(omega) / (2 pi) and frequency_imag = Im(omega) / (2 pi) in Hz (exp(+j omega t): > 0 decays), '
    'q_factor = Re(omega) / (2 Im(omega)) (inf when |Im omega| < 1e-12 Re omega: lossless and closed; negative = a '
    'growing, non-physical solution, usually a PML artefact), pml_fraction (share of the electric energy sum |Re eps| '
    '|E|^2 dV of the E faces in PML and padding cells: resonances above 0.5 are dropped, above ~1e-2 the PML shapes the '
    'resonance: widen the cladding); extras target_wavelength (m) and selected_resonance. With parity_sectors each '
    'sector has its own table (Results.sector_result). Optional domain selection = a region (e.g. a box around the '
    f'cavity core): the table then ends with the column {REGION_COLUMN[0]} (share of the same energy weights on the E '
    'faces of the selected cells (each cell owns its +x / +y / +z faces; a mirror plane face counts with the ghost '
    'cell, so a region on a mirror face reads a little low), on tetrahedra in the dofs of the selected elements), and '
    'resonances below min_region_fraction are dropped from the solve like PML modes (table rows, snapshots and '
    'resonance numbers all skip them): a localised cavity mode passes, slab, rim, band-edge and cladding modes do not. '
    "One region per physics (one resonance_table with a selection); 'all' is refused.")


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
                description='Perfect electric conductor cells: every E face between two selected cells is dropped from '
                            'the unknowns (E = 0 there; the surface lies at the centres of the outermost conductor '
                            'cells), a face between a conductor and another cell takes the other cell\'s permittivity '
                            "(the normal E on the surface stays free), as em_fdtd_3d. Select objects: 'all' is "
                            'refused. Staircased (no conformal option yet).',
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
    make it an upper figure), on tetrahedra of the Nedelec unknowns x fem3d.YEE_EQUIVALENT_UNKNOWNS; none for a cudss
    eigen factorization (checked on the device at run time)."""
    mesh = vctx.mesh
    if mesh is None or eigen_factorization(vctx.solvers(physics_spec)) in CUDSS_FACTORIZATIONS:
        return []
    if mesh.kind == 'tetrahedra':
        order = element_order(settings_values)
        unknowns = nedelec_unknowns(mesh, settings_values)
        issue = vector_3d_size_issue(int(unknowns * YEE_EQUIVALENT_UNKNOWNS[order]))
        return [] if issue is None else [Issue(issue[0], path, f'{issue[1]} (~{unknowns:,} Nedelec unknowns of '
                                                               f'degree {order}, each counted as '
                                                               f'{YEE_EQUIVALENT_UNKNOWNS[order]:g} Yee unknowns)')]
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
    """Device: six complex64 resonance fields. Host (3 E unknowns per cell): 3D SuperLU (or a cuDSS factor: external,
    its copies EIGEN_PHASE items), operators, curls, Arnoldi work, vectors, materials. Tetrahedra: _tetrahedra_..."""
    if mesh.kind == 'tetrahedra':
        return _tetrahedra_memory_estimate(settings_values, mesh, context)
    n_cells = int(mesh.n_cells)
    unknowns = 3 * n_cells
    n_resonances, n_wanted, n_basis = _arnoldi_sizes(context, unknowns)
    vector_bytes = COMPLEX128_BYTES * unknowns
    row_bytes = INT32_BYTES * unknowns   # CSR indptr per row
    factorization = eigen_factorization(context.solvers)
    general = _boundary_config(settings_values).has_bloch_phase   # a Bloch phase: an LU of the whole matrix
    # MUMPS / PARDISO: the LDL^T of D A (LU with a Bloch phase) at the SuperLU heuristic x superlu_bytes_scale
    factor_items = () if factorization in CUDSS_FACTORIZATIONS else tuple(host_factor_items(
        factorization, 'eigen SuperLU L + U factors (3D curl-curl heuristic)', host_lu_bytes(unknowns, dimensions=3),
        unknowns, _ROW_ENTRIES * unknowns, symmetric=not general, eigen=True))
    cudss_items, external_items, external_extra = _cudss_eigen_items(factorization, unknowns, _ROW_ENTRIES * unknowns,
                                                                     general, 'the curl-curl operator')
    # the Arnoldi basis (device Krylov-Schur with room for one widening where it runs) and the filter's device A copy
    search_host, search_transient = resonance_search_items(solver_backend(context.spec), unknowns, n_wanted, n_basis,
                                                           own_cudss=factorization in CUDSS_FACTORIZATIONS,
                                                           filter_entries=_ROW_ENTRIES * unknowns)
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
    )
    # select_resonance uploads one complex64 component at a time after the eigen solve (its own phase)
    return MemoryEstimate.from_items((('resonance fields (6 x complex64)', 6 * COMPLEX64_BYTES * n_cells),),
                                     host_items=host_items,
                                     transient_items=((_SELECT_PHASE, 'field upload of select_resonance (complex64)',
                                                       COMPLEX64_BYTES * n_cells),) + cudss_items
                                     + tuple(search_transient),
                                     external_items=external_items, external_extra=external_extra)


def _arnoldi_sizes(context, unknowns):
    """(n_resonances, wanted eigenpairs 2 n + 10, Arnoldi basis) of the study (fdfd3d ARNOLDI_BASIS_PER_WANTED)."""
    n_resonances = int(_study_values(context).get('n_resonances') or 1)
    n_wanted = max(min(2 * n_resonances + 10, unknowns - 2), 1)
    n_basis = arnoldi_basis_size(n_wanted, unknowns, eigen_subspace_size(context.solvers),
                                 ARNOLDI_BASIS_PER_WANTED * n_wanted)
    return n_resonances, n_wanted, n_basis


def _cudss_eigen_items(factorization, unknowns, entries, general, label, fill=3):
    """(EIGEN_PHASE items, external items, external extra) of solve_resonances' own cuDSS factor (freed after the
    solve): CSR copies (LDL^T lower triangle; Bloch: the whole LU matrix) and solve vectors; nothing without cuDSS.
    fill: the 3D growth law (3) or a counted cudss_factor.FactorFill."""
    if factorization not in CUDSS_FACTORIZATIONS:
        return (), [], 0
    mixed = eigen_mixed(factorization, unknowns)
    vector_bytes = COMPLEX128_BYTES * unknowns
    items = ((EIGEN_PHASE, 'cuDSS operator copies, rhs buffer, row scale and one solve\'s vectors',
              cudss_own_factor_bytes(unknowns, entries, mixed, symmetric=not general)),
             (EIGEN_PHASE, 'eigen B diagonal and applied vector (complex128)', 2 * vector_bytes))
    external_items, external_extra = cudss_factor_items(label, unknowns, entries, dimensions=fill, mixed=mixed,
                                                        general=general)
    return items, external_items, external_extra


def nedelec_factor_entries(mesh, order, unknowns):
    """Counted entries of one factor triangle of the Nedelec operator on a built mesh (MeshEstimate.dissection: dof-site
    pairs x dofs per site^2, scaled by the factored share of the dofs); None on size-based counts (no mesh to count)."""
    dissection = getattr(mesh, 'dissection', None)
    if dissection is None:
        return None
    dofs_per_site = 2 if order == 2 else 1
    share = min(float(unknowns) / max(dofs_per_site * dissection.n_nedelec_sites(order), 1), 1.0)
    return dofs_per_site ** 2 * dissection.nedelec_pairs(order) * share


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
    symmetric = not _boundary_config(settings_values).has_bloch_phase   # MUMPS / PARDISO: LDL^T of the pencil
    if factorization in CUDSS_FACTORIZATIONS:
        factor_items = ()
    elif counted is not None:
        factor_items = tuple(host_factor_items(
            factorization, 'eigen SuperLU L + U factors (counted on the built mesh: nested dissection of the Nedelec '
            'dof sites)', host_counted_lu_bytes(counted, unknowns), unknowns, row_entries * unknowns,
            symmetric=symmetric, eigen=True))
    else:
        factor_items = tuple(host_factor_items(
            factorization, 'eigen SuperLU L + U factors (3D heuristic at the Yee-equivalent unknowns)',
            host_lu_bytes(int(unknowns * YEE_EQUIVALENT_UNKNOWNS[order]), dimensions=3), unknowns,
            row_entries * unknowns, symmetric=symmetric, eigen=True))
    fill = 3
    if counted is not None:
        from ...solvers.cudss_factor import FactorFill
        fill = FactorFill(counted)
    # a Bloch phase makes A nonsymmetric: the LU keeps the whole matrix, not the LDL^T lower triangle
    cudss_items, external_items, external_extra = _cudss_eigen_items(
        factorization, unknowns, row_entries * unknowns, not symmetric, 'the Nedelec curl-curl operator', fill)
    # the Arnoldi basis (device Krylov-Schur with room for one widening where it runs) and the filter's A, M, W copies
    search_host, search_transient = resonance_search_items(solver_backend(context.spec), unknowns, n_wanted, n_basis,
                                                           own_cudss=factorization in CUDSS_FACTORIZATIONS,
                                                           filter_entries=3 * row_entries * unknowns)
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
    # the polished path (ARPACK floor 1e-6): a mixed factor (small operators factor in double, S5c) or, with cpml (a
    # complex A, floored), any LDL^T of D A (solve_resonances polishes a double one there too)
    if not config.has_bloch_phase and (
            (factorization in MIXED_FACTORIZATIONS and unknowns >= MIXED_EIGEN_MIN_UNKNOWNS)
            or (factorization in LDLT_FACTORIZATIONS and config.needs_cpml)):
        applications_per_basis = MIXED_APPLICATIONS_PER_BASIS
    else:
        applications_per_basis = (APPLICATIONS_PER_BASIS_PER_AXIS_CELL * float(mesh.n_cells) ** (1.0 / 3.0)
                                  if config.needs_cpml else 0.0)
    assembly = te.TimeItem('curl-curl assembly', 'assembly', 1.0, 'exact',
                           (te.HostStream(ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(200)),
                           counter=physics_counter(context, 'assemblies'))
    # S9: the resonance filter per eigenpair of a pass (eigenfrequency._ResonanceFilter: A, and on tetrahedra the
    # sparse M and W, applied once per column block; was the "rest of the study": 628 ms against 16 ms on the Mie example)
    candidate_filter = resonance_filter_time_item(context.backend, unknowns,
                                                  3.0 * nnz if mesh.kind == 'tetrahedra' else nnz, eigenpairs)
    return te.combine((te.TimeEstimate((assembly, candidate_filter)),
                       context.solver_work('eigen', unknowns=unknowns, nnz=nnz, value_bytes=COMPLEX128_BYTES,
                                           dimensions=3, solves=1.0, factorizations=1.0, eigenpairs=eigenpairs,
                                           rhs=2.0, default_basis=ARNOLDI_BASIS_PER_WANTED * eigenpairs,
                                           factor_scale=factor_scale,
                                           applications_per_basis=applications_per_basis,
                                           device_arnoldi=factorization in CUDSS_FACTORIZATIONS)))


# ---- type ----
_DESCRIPTION = (
    '3D finite-difference frequency-domain eigensolver on the em_fdtd_3d Yee grid: E_x, E_y, E_z on the +x / +y / +z '
    'faces of each cell in V/m, H_x, H_y, H_z on its (+y +z) / (+x +z) / (+x +y) edges in A/m. The eigenfrequency '
    'study solves curl (mu^-1 curl E) = (omega / C0)^2 eps E for the resonances nearest the target wavelength setting '
    '(shift-invert of the operator assembled there by the eigen factorization: host SuperLU, cudss / cudss_mixed on '
    f'the GPU or {HOST_DIRECT_NAMES} on the host, the last two the LDL^T of D A; ARPACK on the squared Cayley map, '
    'which sends the '
    'static gradient solutions to 0; materials and PML frozen at the target), on the live E faces only (PEC faces, '
    'mirror planes and ghost cells dropped, Bloch phases in the wrap entries). Materials as em_fdtd_3d: eps_r or the '
    'diagonal eps_xx / eps_yy / eps_zz, sigma_e and Drude / Lorentz poles as eps(omega_t) = eps - j sigma_e / (omega_t '
    'eps0) + chi(omega_t) (face means of the two cells; solver_defaults.subpixel_smoothing replaces the background eps '
    'of interface faces by the em_fdtd_3d anisotropic average, so curved surfaces converge smoothly), mu_r - j sigma_m '
    '/ (omega_t mu0) of the lower cell per H edge. '
    'Boundaries per face: cpml (complex stretch at omega_t), pec, pmc, periodic (Bloch k), symmetric / antisymmetric '
    'mirror. Features: pec_domain, resonance_table. Outputs of the selected resonance (each scaled to 1 V/m real at its '
    'largest E sample): the raw components ex / ey / ez / hx / hy / hz _re / _im / _abs (H = j curl E / (omega mu0 mu) '
    'at the complex resonance omega), e_abs (primary), e_intensity, h_abs (cell centres), w_e = eps0 / 4 sum eps_c '
    '|E_c|^2 and eps_r, shown as cut planes. Host memory dominates: the 3D factor grows ~ n^(4/3) with n = the E '
    'unknowns (~3 per cell; a solver size rule warns above 1e6 and refuses 4e6); device: 48 B per cell of fields plus '
    "the material arrays. heat_em_thermo_optic writes eps_r (+ eps_xx / eps_yy) as a background coupling. On a "
    "'tetrahedra' mesh (finite elements): Nedelec edge elements of the discretization setting (quadratic default), "
    'materials one value per element, the cpml faces a PML shell of pml_cells x the maximum element size meshed with '
    'the window (complex stretch tensor), the same eigen solve (squared Cayley map, sparse mass) and outputs sampled '
    'at points (cut planes on a uniform sample lattice, derived values at quadrature points); periodic faces (Bloch k: '
    'hi-side unknowns identified with the lo side times the phase, a nonsymmetric LU factor) need the builtin mesher; '
    'subpixel smoothing does not apply there. '
    "The instance is usually named 'em'.")

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
    supported_studies=(_EIGENFREQUENCY_STUDY,),
    default_study={'type': _EIGENFREQUENCY_STUDY, 'settings': {'n_resonances': 6}},
    uses_dispersion=True,
    grids=_GRIDS,
    create=_create,
    domain_requirements=_domain_requirements,
    shortest_wavelength=shortest_solved_wavelength,
    validate=_validate,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    dimensions=(3,),
    subpixel_smoothing=True,
    solver_roles={'eigen': eigen_role((_EIGENFREQUENCY_STUDY,), mixed_factorization=True)},
    backends=('gpu', 'cpu'),   # host assembly and solve; fields on the Domain's xp
)

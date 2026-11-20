"""Counted factor fill vs cuDSS on one element mesh (dev tooling, not part of open_sol): builds a model's tetrahedron
(or triangle) mesh, counts fem.dissection.element_dissection_pairs per clique graph (Lagrange 1 nodes, Nedelec 1 / 2
dof sites) and cut_axis, and lets cuDSS order and factorise the same graphs (graph Laplacian + I, its own ordering,
lu_nnz); --dofs also the Nedelec 2 dof pattern of the run (2 x 2 blocks per site pair). Writes out/fill_<model><tag>.md
+ .json. Run (gpu): python dissection_vs_cudss.py path/to/model.json [--mesh NAME] [--dofs] [--tag _x]"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # memory_profilers -> tools_profilers -> repository
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(__file__).resolve().parent / 'out'

import numpy as np                                          # noqa: E402
import scipy.sparse                                         # noqa: E402
from open_sol import api                                    # noqa: E402
from open_sol.fem.dissection import MeshDissection, element_dissection_pairs   # noqa: E402
from open_sol.model.expressions import evaluate_parameters  # noqa: E402
from open_sol.model.meshing import element_mesh            # noqa: E402
from open_sol.solvers import cudss_factor                   # noqa: E402

# (column, cut_axis, lowest coordinate from the vertices): the old count, the cut-axis fix, both fixes (estimators)
VARIANTS = (('spread, clique points', 'spread', False), ('smallest, clique points', 'smallest', False),
            ('smallest, vertices', 'smallest', True))


def clique_pattern(cells, n_sites, block=1):
    """Symmetric CSR pattern (all pairs of each element's sites; block > 1: block x block dofs per site pair)."""
    cells = np.asarray(cells, dtype=np.int64)
    local = cells.shape[1]
    rows = np.repeat(cells, local, axis=1).ravel()
    cols = np.tile(cells, (1, local)).ravel()
    if block > 1:
        offsets = np.arange(block)
        rows = (rows[:, None, None] * block + offsets[:, None]).repeat(block, axis=2).ravel()
        cols = (cols[:, None, None] * block + offsets[None, :]).repeat(block, axis=1).ravel()
    size = n_sites * block
    pattern = scipy.sparse.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(size, size)).tocsr()
    pattern.sum_duplicates()
    pattern.data[:] = 1.0
    return pattern


def cudss_fill(pattern, dtype):
    """(lu_nnz, permanent bytes, seconds) of cuDSS's own LDL^T of the graph Laplacian + I on the pattern (SPD values:
    the fill depends on the pattern and cuDSS's ordering only)."""
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    degree = np.asarray(pattern.sum(axis=1)).ravel()
    laplacian = (scipy.sparse.diags(degree + 1.0) - pattern).astype(dtype)   # diagonal: degree (the self pair cancels 1)
    lower = scipy.sparse.tril(laplacian, format='csr')
    lower.sort_indices()
    operator = cpsp.csr_matrix((cp.asarray(lower.data), cp.asarray(lower.indices.astype(np.int32)),
                                cp.asarray(lower.indptr.astype(np.int32))), shape=lower.shape)
    rhs = cp.ones(lower.shape[0], dtype=dtype)
    started = time.perf_counter()
    solver = cudss_factor._direct_solver(operator, rhs, 'symmetric')
    try:
        solver.plan()
        permanent = int(solver.plan_info.memory_estimates['permanent_device_memory'])
        solver.factorize()
        lu_nnz = int(solver.factorization_info.lu_nnz)
    finally:
        solver.free()
    return lu_nnz, permanent, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description='Counted dissection fill vs cuDSS lu_nnz on a model element mesh.')
    parser.add_argument('model', help='model JSON file (its default parameter values)')
    parser.add_argument('--mesh', default=None, help='mesh name (default: the default mesh)')
    parser.add_argument('--dofs', action='store_true', help='also the Nedelec 2 dof pattern (float32 factor on the GPU)')
    parser.add_argument('--tag', default='', help='suffix of the report names')
    args = parser.parse_args()
    spec = api.load(args.model)
    parameters = evaluate_parameters(spec.parameters, errors={})
    started = time.perf_counter()
    mesh, _ = element_mesh(spec, parameters, None, args.mesh)
    mesh_seconds = time.perf_counter() - started
    dissection = MeshDissection(mesh)
    points = np.asarray(mesh.points)
    tetrahedra = getattr(mesh, 'faces', None) is not None
    graphs = [('lagrange 1 (nodes)', dissection.node_cliques(1), 1)]
    graphs.append(('nedelec 1 (edge sites)', dissection.nedelec_cliques(1), 1))
    if tetrahedra:
        graphs.append(('nedelec 2 (edge + face sites)', dissection.nedelec_cliques(2), 1))
        if args.dofs:
            graphs.append(('nedelec 2 dofs (2 per site)', dissection.nedelec_cliques(2), 2))
    record = {'model': str(args.model), 'mesh': args.mesh, 'n_elements': int(mesh.n_elements),
              'n_points': int(mesh.n_points), 'n_edges': int(mesh.n_edges),
              'n_faces': int(getattr(mesh, 'n_faces', 0) or 0), 'mesh_seconds': mesh_seconds,
              'extent': (points.max(axis=0) - points.min(axis=0)).tolist(), 'graphs': []}
    for label, (cells, sites, lowest), block in graphs:
        print(f'{label}: counting')
        entry = {'graph': label, 'sites': int(sites.shape[0]), 'block': block, 'counted': {}, 'count_seconds': {}}
        for variant, cut_axis, element_lowest in VARIANTS:
            count_started = time.perf_counter()
            entry['counted'][variant] = block ** 2 * element_dissection_pairs(
                cells, sites, cut_axis=cut_axis, element_lowest=lowest if element_lowest else None)
            entry['count_seconds'][variant] = time.perf_counter() - count_started
        pattern = clique_pattern(cells, int(sites.shape[0]), block)
        entry['rows'] = int(pattern.shape[0])
        entry['stored_lower'] = int((pattern.nnz + pattern.shape[0]) // 2)
        print(f'{label}: cuDSS')
        lu_nnz, permanent, seconds = cudss_fill(pattern, np.float32 if block > 1 else np.float64)
        del pattern
        entry.update(cudss_lu_nnz=lu_nnz, cudss_permanent_bytes=permanent, cudss_seconds=seconds)
        record['graphs'].append(entry)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f'fill_{Path(args.model).stem}{args.tag}'
    (OUT_DIR / f'{name}.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    lines = [f"# Counted fill vs cuDSS: {args.model}", '',
             f"{record['n_elements']:,} elements, {record['n_points']:,} nodes, {record['n_edges']:,} edges, "
             f"{record['n_faces']:,} faces; extent {', '.join(f'{value:.3g}' for value in record['extent'])} m; "
             f"meshed in {mesh_seconds:.1f} s", '',
             '| graph | rows | stored lower | cuDSS lu_nnz | cuDSS permanent MB | '
             + ' | '.join(f'{variant} / cuDSS' for variant, _, _ in VARIANTS) + ' | count s |',
             '|---|---|---|---|---|' + '---|' * len(VARIANTS) + '---|']
    for entry in record['graphs']:
        counted, lu = entry['counted'], max(entry['cudss_lu_nnz'], 1)
        ratios = ' | '.join(f'{counted[variant] / lu:.2f}' for variant, _, _ in VARIANTS)
        seconds = ', '.join(f"{entry['count_seconds'][variant]:.1f}" for variant, _, _ in VARIANTS)
        lines.append(f"| {entry['graph']} | {entry['rows']:,} | {entry['stored_lower']:,} | {entry['cudss_lu_nnz']:,} | "
                     f"{entry['cudss_permanent_bytes'] / 2 ** 20:.1f} | {ratios} | {seconds} |")
    lines += ['', 'counted = one factor triangle (diagonal included) of the dissection order (variant: cut axis, the '
                  "element's lowest coordinate from its clique points or its vertices; the last = the estimators'); "
                  'cuDSS lu_nnz of its own ordering (symmetric kind, as the runs); a ratio > 1 = the count books more '
                  'than cuDSS stores. Counts in the .json.']
    (OUT_DIR / f'{name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"wrote {OUT_DIR / (name + '.md')}")


if __name__ == '__main__':
    main()

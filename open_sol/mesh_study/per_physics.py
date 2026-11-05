"""S7.8 per-physics meshes (CPU, api.check): cells and estimated memory of every multiphysics template / example on its
shared mesh vs each physics on its own mesh rule -> tables/per_physics.csv.
Own rules (a first guess at S7.2): EM types (family 'em') keep the model mesh; laminar_flow uniform at its smallest
object side / N_FLUID; every other type graded, base max(size_x, size_y) / N_BASE, each of its objects (material
selections setting one of its properties, its feature selections) refined to its smallest side / N_OBJECT (margin
2 dx). drift_diffusion needs its depletion widths resolved: its rows are a lower bound. GridTransfers not counted."""
import csv

from ..model.expressions import evaluate_parameters
from ..model.geometry import domain_size, resolve_geometry
from ..model.meshing import physics_objects, smallest_side
from ..model.schema import get_registry
from ..model.spec import MeshSpec, RefinementSpec
from ..model.templates import make_template, template_entries
from .common import EXAMPLES_DIR, api

N_BASE, N_OBJECT, N_FLUID = 50.0, 4.0, 16.0   # the S7.2 auto rule took these (meshing PHYSICS_* divisors)
TABLE_FILE = 'per_physics.csv'


def multiphysics_models():
    """(name, spec) of every template and example file with two or more physics."""
    registry = get_registry()
    for entry in template_entries(registry):
        spec = make_template(entry.key, registry)
        if len(spec.physics) >= 2:
            yield f'template:{entry.key}', spec
    for path in sorted(EXAMPLES_DIR.glob('*/*.json')):   # none outside a repository checkout
        try:
            spec = api.load(path)
        except Exception as exc:   # a broken example is reported, not fatal
            print(f'{path.name}: load failed ({exc})')
            continue
        if len(spec.physics) >= 2:
            yield f'example:{path.parent.name}/{path.stem}', spec


def own_mesh(spec, physics_spec, physics_type, parameters):
    """MeshSpec of the own-mesh rule (module docstring); None = keep the model's mesh."""
    if physics_type.family == 'em':
        return None
    shapes = resolve_geometry(spec, parameters, errors={})
    size_x, size_y = domain_size(spec, parameters)
    objects = [(name, smallest_side(name, shapes, (size_x, size_y)))
               for name in physics_objects(spec, physics_spec, physics_type, shapes)]
    objects = [(name, side) for name, side in objects if side]
    if physics_type.name == 'laminar_flow':
        sides = [side for _, side in objects] or [min(size_x, size_y)]
        return MeshSpec(mode='uniform', dx=min(sides) / N_FLUID)
    base = max(size_x, size_y) / N_BASE
    refinements = [RefinementSpec(name, side / N_OBJECT, 2 * side / N_OBJECT)
                   for name, side in objects if side / N_OBJECT < base]
    return MeshSpec(mode='nonuniform', dx=base, max_ratio=1.15, refinements=refinements)


def single_physics_spec(spec, physics_spec, mesh):
    sub = spec.copy()
    sub.physics = [next(item for item in sub.physics if item.name == physics_spec.name)]
    sub.couplings, sub.derived_values, sub.analytic_curves, sub.derived_curves = [], [], [], []
    sub.study.sweep = None
    if mesh is not None:
        sub.meshes = [mesh]
        sub.physics[0].mesh = ''
    return sub


def estimate(spec):
    """(n_cells, gpu_peak bytes, host bytes) from api.check (validation errors of a cut-down spec are ignored)."""
    report = api.check(spec)
    mesh, memory = report.get('mesh') or {}, report.get('memory_estimate') or {}
    return mesh.get('n_cells'), memory.get('gpu_peak'), memory.get('host')


def run(options):
    registry = get_registry()
    rows = []
    for model_name, spec in multiphysics_models():
        parameters = evaluate_parameters(spec.parameters, errors={})
        shared_cells, shared_gpu, shared_host = estimate(spec)
        own_total_gpu = 0
        for physics_spec in spec.physics:
            physics_type = registry.physics_type(physics_spec.type)
            try:
                mesh = own_mesh(spec, physics_spec, physics_type, parameters)
                alone_gpu = estimate(single_physics_spec(spec, physics_spec, None))[1]
                own_cells, own_gpu, own_host = estimate(single_physics_spec(spec, physics_spec, mesh))
                rule = ('model mesh' if mesh is None
                        else f'{mesh.mode} dx {float(mesh.dx):.3g} m, {len(mesh.refinements)} refinements')
            except Exception as exc:   # one failing rule must not cost the table
                alone_gpu = own_cells = own_gpu = own_host = None
                rule = f'failed: {exc}'
            own_total_gpu += own_gpu or 0
            rows.append({'model': model_name, 'physics': f'{physics_spec.name} ({physics_spec.type})',
                         'shared_cells': shared_cells, 'own_cells': own_cells, 'gpu_on_shared_mesh': alone_gpu,
                         'gpu_on_own_mesh': own_gpu, 'host_own': own_host, 'rule': rule})
        rows.append({'model': model_name, 'physics': 'TOTAL', 'shared_cells': shared_cells, 'own_cells': None,
                     'gpu_on_shared_mesh': shared_gpu, 'gpu_on_own_mesh': own_total_gpu, 'host_own': shared_host,
                     'rule': 'shared = coupled model as is; own = sum of single-physics estimates'})
    if not rows:
        print('[per_physics] no multiphysics template or example found')
        return
    options.tables_dir.mkdir(parents=True, exist_ok=True)
    out_file = options.tables_dir / TABLE_FILE
    with open(out_file, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']:<55} {row['physics']:<38} cells {row['shared_cells']} -> {row['own_cells']}  "
              f"GPU MiB {_mib(row['gpu_on_shared_mesh'])} -> {_mib(row['gpu_on_own_mesh'])}  {row['rule']}")
    print(f'written {out_file}')


def _mib(value):
    return '-' if value is None else f'{value / 2**20:.1f}'

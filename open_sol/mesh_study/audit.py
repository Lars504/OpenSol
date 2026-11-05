"""S7.8 audit (CPU, text scan): where the package assumes one rectilinear (nx, ny[, nz]) grid, counted per area, to
price S7.3 (patched blocks: every block stays a tensor grid, the physics kernels keep working) against S7.4 (quadtree:
everything changes). Token hits are a proxy, not a proof: read the top files it lists.
-> tables/nontensor_audit.csv (file, area, group, count) and records/audit.json (area totals for the analysis)."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
TOKENS = {   # group -> regex of tensor-grid accessors / assumptions
    'cell_geometry': r'\b(cell_widths|cell_edges|cell_centers|grid_cell_widths|host_centres|physical_window|'
                     r'window_shape|physical_extent|plane_extent|dx_arr|dy_arr|dz_arr|x_edges|y_edges|dx_cells|'
                     r'dy_cells|uniform_dx|edges_from_widths)\b',
    'cell_lookup': r'\b(pos_to_cell\w*|containing_cell|validate_physical_pos|require_window_cell)\b',
    'masks': r'\b(rect_mask|circle_mask|coordinate_grids|rasterize\w*|object_box_mask|object_window_mask|domain_mask|'
             r'region_mask|region_window_mask|perimeter_mask|perimeter_weights|cross_section_runs)\b',
    'grid_shape': r'\b(nx|ny|nz|ox|oy|oz|npml_sides)\b',
    'stencil_inverse_widths': r'\b(inv_dx|inv_dy|inv_dz)\b',
    'material_grid': r'\b(MaterialGrid|assign_window|materials\.assign)\b',
    'window_slicing': r'\[\s*\w*o[xy]\s*:\s*\w*o[xy]\s*\+',
}
AREAS = (   # (area, path prefix relative to open_sol/), first match wins
    ('core: Domain / grid / mesh', 'core/'), ('model: meshing', 'model/meshing'), ('model: builder', 'model/builder'),
    ('model: geometry', 'model/geometry'), ('model: session fields / derived', 'model/session'),
    ('model: analytic curves', 'model/analytic'), ('model: study', 'model/study'), ('model: other', 'model/'),
    ('coupling', 'coupling/'), ('viz', 'viz/'), ('io / exports', 'io/'), ('gui canvas', 'gui/canvas'),
    ('gui other', 'gui/'), ('physics: em 2D', 'physics/em/'), ('physics: em3d', 'physics/em3d/'),
    ('physics: heat', 'physics/heat/'), ('physics: potential', 'physics/potential/'),
    ('physics: mechanics', 'physics/mechanics/'), ('physics: acoustics', 'physics/acoustics/'),
    ('physics: fluid', 'physics/fluid/'), ('physics: semiconductor', 'physics/semiconductor/'),
    ('physics: other', 'physics/'), ('solvers', 'solvers/'), ('api / cli', ''),
)
SUMMARY_FILE = 'audit.json'


def area_of(relative):
    for area, prefix in AREAS:
        if relative.startswith(prefix):
            return area
    return 'other'


def run(options):
    patterns = {group: re.compile(pattern) for group, pattern in TOKENS.items()}
    rows, area_counts, file_totals = [], defaultdict(Counter), Counter()
    for path in sorted(PACKAGE.rglob('*.py')):
        relative = path.relative_to(PACKAGE).as_posix()
        if 'selftest' in relative or relative.startswith('mesh_study/'):
            continue   # self-tests follow whatever the engines do; the study itself is not product code
        text = path.read_text(encoding='utf-8')
        area = area_of(relative)
        for group, pattern in patterns.items():
            count = len(pattern.findall(text))
            if count:
                rows.append({'file': relative, 'area': area, 'group': group, 'count': count})
                area_counts[area][group] += count
                file_totals[relative] += count
    options.tables_dir.mkdir(parents=True, exist_ok=True)
    with open(options.tables_dir / 'nontensor_audit.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'area', 'group', 'count'])
        writer.writeheader()
        writer.writerows(rows)
    groups = list(TOKENS)
    print(f"{'area':<36}" + ''.join(f'{group[:14]:>16}' for group in groups) + f"{'files':>8}")
    summary = {}
    for area, _ in AREAS:
        if area in area_counts:
            n_files = len({row['file'] for row in rows if row['area'] == area})
            summary[area] = {'hits': sum(area_counts[area].values()), 'files': n_files}
            print(f'{area:<36}' + ''.join(f'{area_counts[area][group]:>16}' for group in groups) + f'{n_files:>8}')
    print('\nTop 25 files by hits:')
    for relative, count in file_totals.most_common(25):
        print(f'  {count:>5}  {relative}')
    options.records_dir.mkdir(parents=True, exist_ok=True)
    with open(options.records_dir / SUMMARY_FILE, 'w', encoding='utf-8') as handle:
        json.dump({'areas': summary, 'top_files': file_totals.most_common(25)}, handle, indent=2)
    print(f"written {options.tables_dir / 'nontensor_audit.csv'}, {options.records_dir / SUMMARY_FILE}")

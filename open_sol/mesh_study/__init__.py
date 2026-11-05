"""S7.8 mesh study (mesh_study_plan.md): the parts `python -m open_sol mesh-study` runs, records under the --out
directory. Imports open_sol.api, so only __main__ imports it (inside its handler)."""
import importlib
import traceback

from .common import StudyOptions   # noqa: F401  (the CLI builds it)

PARTS = {   # part -> module, in run order (cheap CPU parts first, analyse last)
    'per_physics': 'per_physics', 'audit': 'audit', 'prototypes': 'prototypes', 'accuracy': 'accuracy',
    'pcg': 'pcg_study', 'line_block': 'line_block_study', 'analyse': 'analysis',
}


def run_study(parts, options) -> list:
    """Run the named parts (empty: all) in PARTS order; a part that raises is reported and the others still run.
    Returns the names of the parts that raised."""
    failed = []
    for part, module_name in PARTS.items():
        if parts and part not in parts:
            continue
        print(f'==== mesh study: {part} ====', flush=True)
        try:
            importlib.import_module(f'{__name__}.{module_name}').run(options)
        except Exception:   # one broken part must not cost the others (KeyboardInterrupt stops the study)
            traceback.print_exc()
            failed.append(part)
    return failed

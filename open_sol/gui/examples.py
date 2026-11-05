"""Examples menu catalog: one entry per <repo>/examples subfolder; a folder's model files that load and validate
without errors, grouped by their physics types; CPU only (GUI-thread safe)."""
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from ..io.model_io import load_model
from ..model.expressions import error_message
from ..model.schema import node_label
from ..model.validate import validate

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'   # beside the package (examples/README.md)
_RUN_DIR_SUFFIX = '_out'          # CLI default run directory '<model stem>_out' (holds a model.json copy)
_LOAD_ERRORS = (OSError, ValueError, KeyError)
MULTIPHYSICS_FOLDER = 'multiphysics'   # listed last, after a separator
_FOLDER_LABELS = {'acoustics': 'Acoustics', 'em': 'Electromagnetics 2D', 'em3d': 'Electromagnetics 3D',
                  'fluid': 'Fluid flow', 'heat': 'Heat transfer', 'magnetic': 'Magnetic fields',
                  'mechanics': 'Solid mechanics', 'multiphysics': 'Multiphysics', 'potential': 'Electric potential',
                  'semiconductor': 'Semiconductor'}


def _listed_folder(name):
    """False for CLI run directories and hidden folders."""
    return not (name.endswith(_RUN_DIR_SUFFIX) or name.startswith('.'))


@dataclass(frozen=True)
class ExampleModel:
    """One GUI-ready example; group = its distinct physics type labels joined by ' + '."""
    path: Path
    relative_path: str        # POSIX path below the examples directory
    title: str
    group: str
    n_physics_types: int
    description: str = ''     # ModelSpec.description (stripped)

    def tooltip(self, width=80):
        """Description wrapped at `width` characters (paragraphs kept), then the relative path."""
        lines = [textwrap.fill(line, width) if line.strip() else '' for line in self.description.splitlines()]
        return '\n\n'.join(part for part in ('\n'.join(lines), self.relative_path) if part)


class ExampleCatalog:
    """Model files under `directory`, each loaded and validated once per file version (mtime, size)."""

    def __init__(self, registry, directory=EXAMPLES_DIR):
        self.registry = registry
        self.directory = Path(directory)
        self._checked = {}        # path -> ((mtime_ns, size), ExampleModel or None)

    def folders(self):
        """[(folder name, menu label)] of the subfolders, by label, the multiphysics folder last; no file is read."""
        if not self.directory.is_dir():
            return []
        names = [entry.name for entry in os.scandir(self.directory) if entry.is_dir() and _listed_folder(entry.name)]
        labelled = [(name, _FOLDER_LABELS.get(name, name.replace('_', ' ').capitalize())) for name in names]
        return sorted(labelled, key=lambda item: (item[0] == MULTIPHYSICS_FOLDER, item[1].casefold()))

    def scan(self, folder_name):
        """(ready ExampleModels below one subfolder: single-physics groups first, then by group and title;
        [(relative path, reason)] of its files found not ready in this scan, i.e. new or changed since the previous one)."""
        ready, newly_skipped = [], []
        for path in self._model_files(self.directory / folder_name):
            try:
                file_stat = path.stat()
            except OSError:
                continue                  # removed since the listing
            version = (file_stat.st_mtime_ns, file_stat.st_size)
            previous = self._checked.get(path)
            if previous is not None and previous[0] == version:
                example = previous[1]
            else:
                example, reason = self._check(path)
                if example is None:
                    newly_skipped.append((self._relative(path), reason))
            self._checked[path] = (version, example)
            if example is not None:
                ready.append(example)
        ready.sort(key=lambda example: (example.n_physics_types, example.group.casefold(), example.title.casefold(),
                                        example.relative_path))
        return ready, newly_skipped

    @staticmethod
    def _model_files(directory):
        """Sorted *.json files below `directory`, skipping CLI run directories and hidden folders."""
        if not directory.is_dir():
            return []
        model_paths = []
        for folder, subfolder_names, file_names in os.walk(directory):
            subfolder_names[:] = [name for name in subfolder_names if _listed_folder(name)]   # pruned in place
            model_paths.extend(Path(folder) / name for name in file_names if name.endswith('.json'))
        return sorted(model_paths)

    def _relative(self, path):
        return path.relative_to(self.directory).as_posix()

    def _check(self, path):
        """(ExampleModel, '') when the file loads and validates without errors, else (None, reason)."""
        try:
            spec = load_model(path)
        except _LOAD_ERRORS as exc:
            return None, error_message(exc.__cause__ or exc)    # load_model's own message repeats the path
        except Exception as exc:   # a malformed file must not abort the whole menu
            return None, f'{type(exc).__name__}: {error_message(exc)}'
        try:
            errors = [issue for issue in validate(spec, self.registry) if issue.level == 'error']
        except Exception as exc:   # a crashing validator: this file is left out, the menu goes on
            return None, f'validation failed: {type(exc).__name__}: {error_message(exc)}'
        if errors:
            plural = 's' if len(errors) != 1 else ''
            return None, (f'{len(errors)} validation error{plural}, first at {node_label(errors[0].path)}: '
                          f'{errors[0].message}')
        type_names = sorted({physics.type for physics in spec.physics})
        group = ' + '.join(self.registry.physics_type(name).label for name in type_names)
        title = (spec.title or '').strip() or path.stem
        return ExampleModel(path, self._relative(path), title, group, len(type_names),
                            (spec.description or '').strip()), ''

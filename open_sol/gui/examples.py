"""Examples menu catalog: one entry per <repo>/examples subfolder; a folder's model files read only for title,
description and physics types (no model load, no validation), grouped by their physics types; GUI-thread safe."""
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from ..model.expressions import error_message

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'   # beside the package (examples/README.md)
_RUN_DIR_SUFFIX = '_out'          # CLI default run directory '<model stem>_out' (holds a model.json copy)
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
    """One listed example; group = its distinct physics type labels joined by ' + '."""
    path: Path
    relative_path: str        # POSIX path below the examples directory
    title: str
    group: str
    n_physics_types: int
    description: str = ''     # the file's description (stripped)

    def tooltip(self, width=80):
        """Description wrapped at `width` characters (paragraphs kept), then the relative path."""
        lines = [textwrap.fill(line, width) if line.strip() else '' for line in self.description.splitlines()]
        return '\n\n'.join(part for part in ('\n'.join(lines), self.relative_path) if part)


class ExampleCatalog:
    """Model files under `directory`, each header read once per file version (mtime, size); never validated here."""

    def __init__(self, registry, directory=EXAMPLES_DIR):
        self.registry = registry
        self.directory = Path(directory)
        self._read = {}           # path -> ((mtime_ns, size), ExampleModel or None)

    def folders(self):
        """[(folder name, menu label)] of the subfolders, by label, the multiphysics folder last; no file is read."""
        if not self.directory.is_dir():
            return []
        names = [entry.name for entry in os.scandir(self.directory) if entry.is_dir() and _listed_folder(entry.name)]
        labelled = [(name, _FOLDER_LABELS.get(name, name.replace('_', ' ').capitalize())) for name in names]
        return sorted(labelled, key=lambda item: (item[0] == MULTIPHYSICS_FOLDER, item[1].casefold()))

    def scan(self, folder_name):
        """(ExampleModels below one subfolder: single-physics groups first, then by group and title; [(relative path,
        reason)] of its files found unreadable in this scan, i.e. new or changed since the previous one)."""
        listed, newly_unreadable = [], []
        for path in self._model_files(self.directory / folder_name):
            try:
                file_stat = path.stat()
            except OSError:
                continue                  # removed since the listing
            version = (file_stat.st_mtime_ns, file_stat.st_size)
            previous = self._read.get(path)
            if previous is not None and previous[0] == version:
                example = previous[1]
            else:
                example, reason = self._read_header(path)
                if example is None:
                    newly_unreadable.append((self._relative(path), reason))
            self._read[path] = (version, example)
            if example is not None:
                listed.append(example)
        listed.sort(key=lambda example: (example.n_physics_types, example.group.casefold(), example.title.casefold(),
                                         example.relative_path))
        return listed, newly_unreadable

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

    def _physics_label(self, type_name):
        """Registered label of a physics type, else the raw name (the model check on open reports unknown types)."""
        try:
            return self.registry.physics_type(type_name).label
        except (KeyError, ValueError):
            return type_name

    def _read_header(self, path):
        """(ExampleModel, '') from the raw JSON's title / description / physics types, else (None, reason)."""
        try:
            model_dict = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as exc:     # JSONDecodeError and UnicodeDecodeError are ValueErrors
            return None, error_message(exc)
        if not isinstance(model_dict, dict):
            return None, 'not a model file (the JSON top level is not an object)'
        physics_list = model_dict.get('physics') or []
        type_names = sorted({str(physics.get('type', '')) for physics in physics_list
                             if isinstance(physics, dict) and physics.get('type')})
        group = ' + '.join(self._physics_label(name) for name in type_names)
        title = str(model_dict.get('title') or '').strip() or path.stem
        description = str(model_dict.get('description') or '').strip()
        return ExampleModel(path, self._relative(path), title, group, len(type_names), description), ''

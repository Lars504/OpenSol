"""Examples menu catalog: one entry per <repo>/examples subfolder (a physics area, 2D / 3D / axisymmetric together);
model files read only for title, description, domain and physics types (no load, no validation); GUI-thread safe.
Each entry carries its space (EXAMPLE_SPACES) and its physics type group, the menu's two levels below the folder."""
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from ..model.expressions import error_message
from ..model.schema import instance_spaces, type_spaces

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'   # beside the package (examples/README.md)
_RUN_DIR_SUFFIX = '_out'          # CLI default run directory '<model stem>_out' (holds a model.json copy)
MULTIPHYSICS_FOLDER = 'multiphysics'   # listed last, after a separator
MULTIPHYSICS_LABEL = 'Multiphysics'   # an area folder's submenu of its models with several physics types
AXISYMMETRIC_LABEL = 'Axisymmetric'
SPHERICAL_LABEL = 'Spherical'
# ExampleModel.space values, in menu order -> submenu tooltip
EXAMPLE_SPACES = {'1D': 'One-dimensional models (a line along x).',
                  '2D': 'Planar and cross-section models (the drawing is a plane of the 3D object).',
                  '3D': 'Three-dimensional models.',
                  AXISYMMETRIC_LABEL: 'Cylinder symmetry: the drawing is the (r, z) half-plane, rotated about its '
                                      'left edge.',
                  SPHERICAL_LABEL: "Spherical grid: the drawing is the (r, theta) half disc (domain coordinates "
                                   "'spherical'), rotated about the polar axis."}
_FOLDER_LABELS = {'acoustics': 'Acoustics', 'em': 'Electromagnetics',
                  'fluid': 'Fluid flow', 'heat': 'Heat transfer', 'magnetic': 'Magnetic fields',
                  'mechanics': 'Solid mechanics', 'multiphysics': 'Multiphysics', 'potential': 'Electric potential',
                  'semiconductor': 'Semiconductor'}
# physics label prefix a folder's single-physics groups drop (the folder label already says it): 'Frequency domain (...)'
_FOLDER_GROUP_PREFIXES = {'em': 'Electromagnetic Waves, '}


def _listed_folder(name):
    """False for CLI run directories and hidden folders."""
    return not (name.endswith(_RUN_DIR_SUFFIX) or name.startswith('.'))


def _model_dimension(domain):
    """ModelSpec.dimensions of a raw JSON domain (absent: the DomainSpec default, 2D), None when it is no object."""
    if domain is None:
        return 2
    if not isinstance(domain, dict):
        return None
    if 'size_y' in domain and domain['size_y'] is None:
        return 1
    return 3 if domain.get('size_z') is not None else 2


@dataclass(frozen=True)
class ExampleModel:
    """One listed example; group = its distinct physics type labels joined by ' + ' (one type: less the folder's
    _FOLDER_GROUP_PREFIXES); space = an EXAMPLE_SPACES key."""
    path: Path
    relative_path: str        # POSIX path below the examples directory
    title: str
    group: str
    n_physics_types: int
    space: str
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
        """[(folder name, menu label)] of the subfolders holding a model file (run directories left behind do not
        count), by label, the multiphysics folder last; no file is read."""
        if not self.directory.is_dir():
            return []
        names = [entry.name for entry in os.scandir(self.directory)
                 if entry.is_dir() and _listed_folder(entry.name) and self._model_files(Path(entry.path))]
        labelled = [(name, _FOLDER_LABELS.get(name, name.replace('_', ' ').capitalize())) for name in names]
        return sorted(labelled, key=lambda item: (item[0] == MULTIPHYSICS_FOLDER, item[1].casefold()))

    def scan(self, folder_name):
        """(ExampleModels below one subfolder: single-physics groups first, then by space (EXAMPLE_SPACES order), group
        and title; [(relative path, reason)] of its files found unreadable in this scan, i.e. new or changed since the
        previous one). Raw JSON only, never load_model / validate: on a CPU-only machine validation probes cuDSS /
        gmsh per file and would stall the menu; a model is checked when opened."""
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
        space_rank = {space: rank for rank, space in enumerate(EXAMPLE_SPACES)}
        listed.sort(key=lambda example: (example.n_physics_types > 1, space_rank[example.space],
                                         example.n_physics_types, example.group.casefold(), example.title.casefold(),
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

    def _model_space(self, domain, physics_list):
        """EXAMPLE_SPACES key of a raw model, None when its domain is no object: 3D / 1D from the domain sizes, else
        Spherical for a spherical (r, theta) grid, else Axisymmetric when every registered physics instance runs only
        axisymmetric (settings over the type defaults, model.schema.instance_spaces: heat 'geometry', the BOR types),
        else 2D."""
        dimension = _model_dimension(domain)
        if dimension is None:
            return None
        if dimension != 2:
            return f'{dimension}D'
        if isinstance(domain, dict) and domain.get('coordinates') == 'spherical':   # schema.COORDINATE_SPACES
            return SPHERICAL_LABEL
        shared = None
        for physics in physics_list:
            if not isinstance(physics, dict):
                continue
            try:
                physics_type = self.registry.physics_type(str(physics.get('type', '')))
            except (KeyError, ValueError):
                continue                  # unknown type: the model check on open reports it
            settings = physics.get('settings')
            values = ({param.name: param.default for param in physics_type.settings}
                      | (settings if isinstance(settings, dict) else {}))
            spaces = set(instance_spaces(physics_type, values) or type_spaces(physics_type)) - {'3d'}
            shared = spaces if shared is None else shared & spaces
        return AXISYMMETRIC_LABEL if shared == {'axisymmetric'} else '2D'

    def _read_header(self, path):
        """(ExampleModel, '') from the raw JSON's title / description / domain / physics types, else (None, reason)."""
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
        prefix = _FOLDER_GROUP_PREFIXES.get(Path(self._relative(path)).parts[0], '')
        if len(type_names) == 1 and prefix and group.startswith(prefix):
            group = group[len(prefix)].upper() + group[len(prefix) + 1:]
        space = self._model_space(model_dict.get('domain'), physics_list)
        if space is None:
            return None, "not a model file ('domain' is not an object)"
        title = str(model_dict.get('title') or '').strip() or path.stem
        description = str(model_dict.get('description') or '').strip()
        return ExampleModel(path, self._relative(path), title, group, len(type_names), space, description), ''

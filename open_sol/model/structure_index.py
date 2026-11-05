"""Generated module index (structure_index.md beside structure_conventions.md): per package folder one line per module,
the first sentence of its docstring. Standard library only; no package module is imported."""
import ast
import re
from pathlib import Path

from .import_contracts import PACKAGE_DIR, ROOT_PACKAGE

STRUCTURE_INDEX_FILE = 'structure_index.md'
STRUCTURE_INDEX_PATH = PACKAGE_DIR.parent / STRUCTURE_INDEX_FILE   # repository checkout only
CONVENTIONS_PATH = PACKAGE_DIR.parent / 'structure_conventions.md'   # present = a repository checkout
MAX_SUMMARY_CHARS = 240   # a longer first sentence is cut at a word boundary
NO_DOCSTRING = '(no module docstring)'
# '. ' + capital / bracket ends a sentence, unless the word before is one of these abbreviations
_SENTENCE_END = re.compile(r'(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bcf)(?<!\betc)\.\s+(?=[A-Z(\'"`])')
_SKIPPED_FOLDERS = ('__pycache__',)


def module_summary(path) -> str:
    """First sentence of the module docstring of path, whitespace collapsed (NO_DOCSTRING without one)."""
    docstring = ast.get_docstring(ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path)))
    if not docstring:
        return NO_DOCSTRING
    paragraph = ' '.join(docstring.split('\n\n', 1)[0].split())
    sentence = _SENTENCE_END.split(paragraph, 1)[0]
    if len(sentence) > MAX_SUMMARY_CHARS:
        sentence = sentence[:MAX_SUMMARY_CHARS].rsplit(' ', 1)[0] + ' ...'
    return sentence


def _folder_sections():
    """[(folder title, [module lines])] in folder order, the package root first."""
    sections = {}
    for path in sorted(PACKAGE_DIR.rglob('*.py'), key=lambda path: (path.parent.relative_to(PACKAGE_DIR).parts,
                                                                    path.name != '__init__.py', path.name)):
        folder = path.parent.relative_to(PACKAGE_DIR)
        if any(part in _SKIPPED_FOLDERS for part in folder.parts):
            continue
        title = '/'.join((ROOT_PACKAGE, *folder.parts)) + '/'
        sections.setdefault(title, []).append(f'- `{path.name}`: {module_summary(path)}')
    return list(sections.items())


def structure_index_text() -> str:
    """structure_index.md: header, a Contents of line ranges per folder, then one section per folder."""
    header = [f'# {ROOT_PACKAGE} module index', '',
              'Generated from the module docstrings (first sentence each) by `python -m open_sol describe '
              '--update-reference`; selftest structure_index_current fails while this file differs. Edit the '
              'docstring, never this file. Read the folders you work in by their line range (offset / limit).', '',
              '## Contents', '']
    sections = _folder_sections()
    contents_start = len(header)
    line = contents_start + len(sections) + 1 + 1   # after the contents lines and one blank line, 1-based
    contents, body = [], []
    for title, module_lines in sections:
        block = [f'## {title}', '', *module_lines, '']
        contents.append(f'- {title}: lines {line}-{line + len(block) - 2}')
        body.extend(block)
        line += len(block)
    return '\n'.join([*header, *contents, '', *body]).rstrip('\n') + '\n'

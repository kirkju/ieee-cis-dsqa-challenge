"""What can be told of the entry without running it.

The evaluation image has the packages of requirements-participant.txt and
nothing else, runs Python 3.10 and shows no error message: a missing import
or a line of later syntax would fail the run there with nothing to go by.
"""

import ast
import sys
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "03_dsqa"
# The organizers' RAG example, copied into the entry as it came (dsqa/example.py).
EXAMPLE = ENTRY / "example_rag.py"
SOURCES = sorted(path for path in ENTRY.rglob("*.py") if "__pycache__" not in path.parts)


def _pinned() -> set[str]:
    lines = (ROOT / "requirements-participant.txt").read_text(encoding="utf-8").splitlines()
    names = (line.split("#", 1)[0].partition("==")[0].split("[", 1)[0].strip() for line in lines)
    return {name.lower().replace("_", "-") for name in names if name}


def _imported(tree: ast.AST) -> set[str]:
    """The top-level names of what a module imports from outside its package."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_entry_is_there():
    names = {path.relative_to(ENTRY).as_posix() for path in SOURCES}
    assert "main.py" in names and "dsqa/__init__.py" in names


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.relative_to(ENTRY).as_posix())
def test_a_file_of_the_entry_runs_on_the_evaluation_s_python(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.relative_to(ENTRY).as_posix())
def test_a_file_of_the_entry_imports_only_what_the_evaluation_has(path):
    pinned = _pinned()
    distributions = metadata.packages_distributions()
    for name in _imported(ast.parse(path.read_text(encoding="utf-8"))):
        if name in sys.stdlib_module_names or name == "dsqa":
            continue
        # Not our development code, and a package that is among the pins. A
        # wheel that does not say which modules it brings is taken to be
        # named as its module is, as numpy and pymupdf are.
        assert name not in ("tools", "tests"), f"{path.name} imports {name}"
        provided_by = distributions.get(name) or [name]
        provided_by = {dist.lower().replace("_", "-") for dist in provided_by}
        assert provided_by & pinned, f"{path.name} imports {name}, which is not pinned"


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.relative_to(ENTRY).as_posix())
def test_the_base_model_is_named_nowhere_in_the_entry(path):
    # It comes as answer()'s model_id; a name or a path written here would be
    # wrong the day the organizers change it. The organizers' RAG example,
    # in the entry as it came (example_rag.py), names it in a comment: it is
    # held to their file instead, by the test after this one.
    if path == EXAMPLE:
        pytest.skip("the organizers' own file, held to 02_rag/main.py byte for byte")
    text = path.read_text(encoding="utf-8").lower()
    for written in ("allenai", "olmo-", "/workspace", "huggingface.co"):
        assert written not in text, f"{path.name} has {written!r} written in it"


def test_the_rag_example_in_the_entry_is_the_organizers_own_byte_for_byte():
    # dones-example answers with the organizers' RAG example itself
    # (dsqa/example.py): a change to its copy, however small, would make it
    # something else, and the experiment would say nothing of the example.
    assert EXAMPLE.read_bytes() == (ROOT / "02_rag" / "main.py").read_bytes()

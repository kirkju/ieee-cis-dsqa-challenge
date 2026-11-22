"""tools/build_entry.py: the submission zip, built and checked.

Most of it is tried on an entry of a few small files made on the spot. The
real entry is built too, without its embedding model, and run from its zip.
"""

import json
import zipfile
from pathlib import Path

import pytest

from tests.unit.stand_ins import BODY, paper_pdf
from tools import build_entry, run_local
from tools.build_entry import BuildError, build, check

ROOT = Path(__file__).resolve().parents[2]
CLEAN = {"commit": "c0ffee" * 6 + "c0ff", "branch": "main", "dirty": False}
DIRTY = {**CLEAN, "dirty": True}


@pytest.fixture
def entry(tmp_path):
    """An entry in small: what goes into the zip, and what must not."""
    folder = tmp_path / "entry"
    (folder / "dsqa" / "__pycache__").mkdir(parents=True)
    (folder / "embed_model" / "1_Pooling").mkdir(parents=True)
    (folder / "main.py").write_text("def answer(q, c, m):\n    return 'x'\n", encoding="utf-8")
    (folder / "dsqa" / "__init__.py").write_text('"""The package."""\n', encoding="utf-8")
    (folder / "dsqa" / "parse.py").write_text("TITLE = 3\n", encoding="utf-8")
    (folder / "dsqa" / "__pycache__" / "parse.cpython-310.pyc").write_bytes(b"\x00compiled")
    (folder / "embed_model" / "modules.json").write_text("[]", encoding="utf-8")
    (folder / "embed_model" / "config.json").write_text("{}", encoding="utf-8")
    (folder / "embed_model" / "model.safetensors").write_bytes(b"the weights")
    (folder / "embed_model" / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    # Not part of a submission: the script that builds it, and whatever else is around.
    (folder / "build.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (folder / "notes.txt").write_text("to do", encoding="utf-8")
    return folder


def test_the_zip_has_what_is_named_and_nothing_else_of_the_folder(entry, tmp_path):
    built = build(tmp_path / "out" / "entry.zip", entry, state=CLEAN)
    assert built["files"] == [
        "VERSION",
        "dsqa/__init__.py",
        "dsqa/parse.py",
        "embed_model/1_Pooling/config.json",
        "embed_model/config.json",
        "embed_model/model.safetensors",
        "embed_model/modules.json",
        "main.py",
    ]
    with zipfile.ZipFile(built["path"]) as archive:
        assert archive.namelist() == built["files"]
        assert archive.read("main.py") == (entry / "main.py").read_bytes()
        assert archive.read("embed_model/model.safetensors") == b"the weights"
        stated = archive.read("VERSION").decode("utf-8")
    # The commit, and the embedding model by the hash of its weights.
    assert stated.splitlines()[0] == f"commit {CLEAN['commit']}"
    assert "embed_model/model.safetensors sha256 " in stated
    assert "sentence-transformers/all-MiniLM-L6-v2" in stated
    assert "uncommitted" not in stated
    assert built["unzipped_bytes"] == sum(
        len(data) for data in map(lambda name: _read(built["path"], name), built["files"])
    )
    assert (built["commit"], built["dirty"]) == (CLEAN["commit"], False)
    assert check(built["path"]) == []


def _read(path, name):
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def test_the_same_commit_gives_the_same_zip_byte_for_byte(entry, tmp_path):
    first = build(tmp_path / "a.zip", entry, state=CLEAN)
    # Files written again later, and found in another order, change nothing.
    for path in sorted(entry.rglob("*.py"), reverse=True):
        path.write_bytes(path.read_bytes())
    second = build(tmp_path / "b.zip", entry, state=CLEAN)
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    # A change in a file is a change in the zip.
    (entry / "dsqa" / "parse.py").write_text("TITLE = 4\n", encoding="utf-8")
    assert build(tmp_path / "c.zip", entry, state=CLEAN)["sha256"] != first["sha256"]


def test_the_zip_is_the_same_whichever_machine_builds_it(entry, tmp_path):
    # The embedding model's text files as a download on Windows leaves them,
    # and as one on Linux does.
    (entry / "embed_model" / "config.json").write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    (entry / "embed_model" / "model.safetensors").write_bytes(b"the\r\nweights")
    on_windows = build(tmp_path / "windows.zip", entry, state=CLEAN)
    (entry / "embed_model" / "config.json").write_bytes(b'{\n  "a": 1\n}\n')
    on_linux = build(tmp_path / "linux.zip", entry, state=CLEAN)
    assert on_windows["sha256"] == on_linux["sha256"]
    with zipfile.ZipFile(on_windows["path"]) as archive:
        assert archive.read("embed_model/config.json") == b'{\n  "a": 1\n}\n'
        # The weights are not text, whatever bytes they hold.
        assert archive.read("embed_model/model.safetensors") == b"the\r\nweights"
        # Every file says it was made on Unix, whose permissions it carries,
        # where Python would have said the machine it runs on.
        assert {info.create_system for info in archive.infolist()} == {3}
        assert {info.external_attr >> 16 for info in archive.infolist()} == {0o644}
    assert on_windows["unzipped_bytes"] == on_linux["unzipped_bytes"]


def test_a_tree_with_uncommitted_changes_is_refused_unless_asked_for(entry, tmp_path):
    with pytest.raises(BuildError, match="uncommitted changes"):
        build(tmp_path / "entry.zip", entry, state=DIRTY)
    assert not (tmp_path / "entry.zip").exists()
    built = build(tmp_path / "entry.zip", entry, state=DIRTY, allow_dirty=True)
    assert built["dirty"]
    assert "built from a tree with uncommitted changes" in _read(built["path"], "VERSION").decode()
    # And such a zip says so when it is checked.
    assert check(built["path"]) == [
        "built from a tree with uncommitted changes: not to be uploaded"
    ]
    # A tree whose state could not be read is not known to be clean.
    with pytest.raises(BuildError):
        build(tmp_path / "unknown.zip", entry, state={"commit": None, "dirty": None})


def test_an_entry_without_its_parts_is_not_built(entry, tmp_path):
    (entry / "embed_model" / "model.safetensors").unlink()
    built = build(tmp_path / "entry.zip", entry, state=CLEAN)
    assert any("the embedding model cannot load" in problem for problem in check(built["path"]))
    for path in sorted((entry / "embed_model").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (entry / "embed_model").rmdir()
    with pytest.raises(BuildError, match="--fetch-embedder"):
        build(tmp_path / "entry.zip", entry, state=CLEAN)
    # Left out on purpose, it is not missed.
    without = build(tmp_path / "without.zip", entry, state=CLEAN, embedder=False)
    assert not any(name.startswith("embed_model/") for name in without["files"])
    assert check(without["path"], embedder=False) == []
    (entry / "main.py").unlink()
    with pytest.raises(BuildError, match="no main.py"):
        build(tmp_path / "entry.zip", entry, state=CLEAN, embedder=False)


def write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


GOOD = {
    "main.py": "def answer(q, c, m):\n    return 'x'\n",
    "VERSION": "commit abc\n",
    "dsqa/__init__.py": "",
    "embed_model/modules.json": "[]",
    "embed_model/config.json": "{}",
    "embed_model/model.safetensors": "w",
}


def test_a_zip_is_checked_for_what_would_fail_unseen(tmp_path):
    assert check(write_zip(tmp_path / "good.zip", GOOD)) == []
    (tmp_path / "text.zip").write_text("not a zip", encoding="utf-8")
    assert check(tmp_path / "text.zip") == ["text.zip is not a zip file"]

    def problems(**changes):
        files = {name: data for name, data in {**GOOD, **changes}.items() if data is not None}
        return check(write_zip(tmp_path / "tried.zip", files))

    # main.py inside a folder, as a zip of the folder itself would have it.
    nested = {f"03_dsqa/{name}": data for name, data in GOOD.items()}
    found = check(write_zip(tmp_path / "nested.zip", nested))
    assert found[0] == "no main.py at the root of the zip; there is one at 03_dsqa/main.py"
    assert problems(**{"notes.txt": "to do"}) == ["notes.txt: not part of the entry"]
    assert problems(**{"dsqa/__pycache__/a.pyc": "c"}) == [
        "dsqa/__pycache__/a.pyc: compiled code, not source"
    ]
    assert problems(VERSION=None) == ["no VERSION: the zip does not say which commit it is"]
    assert problems(**{"embed_model/config.json": None}) == [
        "embed_model/config.json: missing, the embedding model cannot load"
    ]
    # Syntax of a later Python than the evaluation's.
    (later,) = problems(**{"dsqa/later.py": "type Number = int | float\n"})
    assert later.startswith("dsqa/later.py: Python 3.10 cannot read it")
    # Over the limit once unzipped.
    big = write_zip(tmp_path / "big.zip", {**GOOD, "dsqa/big.py": "#" * 500})
    size = sum(len(data) for data in GOOD.values()) + 500
    assert check(big, limit=400) == [f"{size:,} bytes once unzipped, over the limit of 400"]


def test_the_real_entry_builds_and_runs_from_its_zip(tmp_path, monkeypatch):
    # Without its embedding model, to keep this quick: the corpus is then
    # ranked by words. The model's place is taken by the contract tests' fake.
    built = build(tmp_path / "entry.zip", allow_dirty=True, embedder=False)
    assert "main.py" in built["files"] and "dsqa/answering.py" in built["files"]
    assert not any(name.startswith(("tools", "tests", "build")) for name in built["files"])
    problems = check(built["path"], embedder=False)
    assert [found for found in problems if "uncommitted" not in found] == []

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    paper_pdf(corpus / "loop.pdf", BODY)
    monkeypatch.setenv("DSQA_ENGINE", str(ROOT / "tests" / "contract" / "fake_engine.py"))
    monkeypatch.setenv("DSQA_FAKE", "")
    question = {"id": 1, "question": "How is the lithium carried to the target?"}
    meta = run_local.run(
        built["path"],
        [question],
        model_id="the/base-model",
        corpus_dir=corpus,
        out_dir=tmp_path / "runs",
        say=lambda line: None,
    )
    answers = (tmp_path / "runs" / meta["run_id"] / "answers.jsonl").read_text(encoding="utf-8")
    (row,) = [json.loads(line) for line in answers.splitlines()]
    assert not run_local.broke_the_contract(meta)
    assert row["answer"].startswith("An answer with passages from the/base-model")


def test_the_real_entry_with_its_embedder_is_within_the_limit(tmp_path):
    if not (build_entry.ENTRY / "embed_model").is_dir():
        pytest.skip("03_dsqa/embed_model is not downloaded")
    built = build(tmp_path / "entry.zip", allow_dirty=True)
    assert [found for found in check(built["path"]) if "uncommitted" not in found] == []
    # An entry that ships with dones-example carries Docling's models, some
    # 330 MB, as the organizers' RAG example does: still under half the limit.
    share = 2 if build_entry.ships_example() else 5
    assert built["unzipped_bytes"] < build_entry.LIMIT_BYTES / share


def test_the_command_builds_checks_and_says_what_is_wrong(tmp_path, monkeypatch, capsys, entry):
    monkeypatch.setattr(build_entry.ledger, "git_state", lambda: CLEAN)
    out = tmp_path / "made.zip"
    assert build_entry.main(["--out", str(out), "--entry", str(entry)]) == 0
    said = capsys.readouterr().out
    assert "8 files" in said and f"commit {CLEAN['commit']}" in said
    assert "the zip is fit to upload as far as can be told here" in said

    assert build_entry.main(["--check", str(write_zip(tmp_path / "bad.zip", {"a.py": ""}))]) == 1
    assert "PROBLEM: no main.py at the root of the zip" in capsys.readouterr().out

    monkeypatch.setattr(build_entry.ledger, "git_state", lambda: DIRTY)
    assert build_entry.main(["--out", str(tmp_path / "dirty.zip"), "--entry", str(entry)]) == 1
    assert "uncommitted changes" in capsys.readouterr().out
    assert not (tmp_path / "dirty.zip").exists()

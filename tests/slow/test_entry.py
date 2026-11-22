"""Our entry, run for real: the embedder it ships and a model small enough
for a laptop, loaded and asked by the code the evaluation will run.

Left out of the default run, as the examples' slow tests are, and needing the
same things: a GPU, the small model in the local Hugging Face cache and the
development corpus, and besides them the entry's own copy of the embedding
model in 03_dsqa/embed_model.

    python -m pytest -m slow

The fast tests put a fake in the model's place and leave the embedder out.
What only these show is that the real ones load from where the entry looks
for them, with the internet cut, and that a prompt of the entry's size goes
through the model. What the model answers is not checked.
"""

import json
import re
import shutil
import subprocess
import sys

import pytest

from tests.slow.test_examples import (  # noqa: F401  (the fixture is used by being here)
    ROOT,
    SMALL_CORPUS,
    STAND_IN,
    stand_in_model_is_usable,
)
from tools import build_entry, corpus

pytestmark = pytest.mark.slow


@pytest.fixture
def three_papers(tmp_path):
    papers = [corpus.CORPUS_DIR / f"{doc_id}.pdf" for doc_id in SMALL_CORPUS]
    needed = [ROOT / "03_dsqa" / "embed_model", *papers]
    missing = [path.relative_to(ROOT).as_posix() for path in needed if not path.exists()]
    if missing:
        pytest.skip("not there: " + ", ".join(missing))
    folder = tmp_path / "corpus"
    folder.mkdir()
    for paper in papers:
        shutil.copy(paper, folder)
    return folder


def test_the_entry_answers_from_its_zip_with_its_embedder_and_a_real_model(tmp_path, three_papers):
    # What gets uploaded: the zip as the build makes it, the embedder inside.
    built = build_entry.build(tmp_path / "entry.zip", allow_dirty=True)
    problems = build_entry.check(built["path"])
    assert [found for found in problems if "uncommitted" not in found] == []

    out = tmp_path / "runs"
    command = [sys.executable, "-m", "tools.run_local", str(built["path"])]
    command += ["--model-id", STAND_IN, "--corpus-dir", str(three_papers), "--ids", "22,25"]
    command += ["--out", str(out)]
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    (run_dir,) = out.iterdir()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    answers = (run_dir / "answers.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in answers.splitlines()]
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert meta["problems"] == []
    assert meta["blocked_connections"] == []
    assert meta["left_in_working_directory"] == []

    # Nothing fell back: the embedder and the model loaded, and each question
    # was put to the model with passages.
    said = done.stdout
    assert f"[dsqa] {len(SMALL_CORPUS)} documents in" in said
    for fallback in ("no embedder", "could not be loaded", "no index", "no time left"):
        assert fallback not in said, said[-3000:]
    shown = [int(count) for count in re.findall(r"question \d+: (\d+) chunks shown", said)]
    assert len(shown) == 2 and all(count > 10 for count in shown)
    assert meta["gpu"]["peak_allocated_gb"] > 1  # the model and a long prompt were on the GPU

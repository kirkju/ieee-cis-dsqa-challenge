"""dones-example answers with the organizers' RAG example itself, word for word.

The example is run twice over a small corpus, each run a process of its own
(tests/slow/example_run.py): once by itself, its main.py reading the corpus,
loading the model and answering as when it is the submission; once as the
entry runs it with dones-example, its index made before the entry's engine
loads the model, and its answer() asked with that model. The two have to give
the same chunks, the same embeddings, the same context and the same answer,
character for character: the experiment says something of the example only
if what answers is the example.

Left out of the default run, as the other slow tests are, and needing what
they need: a GPU, the small model in the local Hugging Face cache, Docling's
models and MiniLM next to both main.py files, and the development corpus.

    python -m pytest -m slow tests/slow/test_example_fidelity.py
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from tests.slow.test_examples import (  # noqa: F401  (the fixture is used by being here)
    ROOT,
    STAND_IN,
    stand_in_model_is_usable,
)
from tools import corpus

pytestmark = pytest.mark.slow

# Sample question 8, on IFMIF-DONES, its document and two other documents on
# IFMIF-DONES: 33 pages for Docling.
QUESTION = "What is the bypass ratio of the ICS Branch of the DONES Li loop?"
PAPERS = [
    "dones-lithium-systems-ifmif-dones",
    "dones-lithium-loop-purification-system",
    "dones-radiation-dose-assessment-7be",
]
RUNNER = ROOT / "tests" / "slow" / "example_run.py"


def kept(name: str):
    """Where one of the two models is kept: next to the entry's main.py, or
    where the RAG example keeps it, 02_rag/, as its build.sh and the rented
    machine's setup leave it. None when in neither."""
    for folder in (ROOT / "03_dsqa" / name, ROOT / "02_rag" / name):
        if folder.is_dir():
            return folder
    return None


@pytest.fixture
def small_corpus(tmp_path):
    papers = [corpus.CORPUS_DIR / f"{doc_id}.pdf" for doc_id in PAPERS]
    needed = [ROOT / "02_rag" / "docling_models", ROOT / "02_rag" / "embed_model", *papers]
    missing = [path.relative_to(ROOT).as_posix() for path in needed if not path.exists()]
    if missing:
        pytest.skip("not there: " + ", ".join(missing))
    folder = tmp_path / "corpus"
    folder.mkdir()
    for paper in papers:
        shutil.copy(paper, folder)
    return folder


@pytest.fixture
def unzipped(tmp_path):
    """The entry as its zip unpacks with dones-example: its code, the RAG
    example's file, and MiniLM and Docling's models next to them, copied
    from wherever they are kept."""
    folder = tmp_path / "entry"
    folder.mkdir()
    for name in ("main.py", "example_rag.py"):
        shutil.copy(ROOT / "03_dsqa" / name, folder / name)
    shutil.copytree(
        ROOT / "03_dsqa" / "dsqa", folder / "dsqa", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("embed_model", "docling_models"):
        source = kept(name)
        if source is None:
            pytest.skip(f"not there: {name}")
        shutil.copytree(source, folder / name, ignore=shutil.ignore_patterns(".cache"))
    return folder


def asked(way: str, corpus_dir, entry=None) -> dict:
    environment = {key: value for key, value in os.environ.items() if key != "DSQA_ENGINE"}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    environment["DSQA_VARIANT"] = "dones-example" if way == "replica" else ""
    command = [sys.executable, str(RUNNER), way, str(corpus_dir), STAND_IN, QUESTION]
    command += [str(entry)] if entry is not None else []
    done = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    (line,) = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    return json.loads(line.removeprefix("RESULT "))


def test_the_entry_answers_with_the_example_word_for_word(small_corpus, unzipped):
    original = asked("original", small_corpus)
    replica = asked("replica", small_corpus, unzipped)
    for what in ("chunks", "chunks_sha256", "embeddings_sha256", "model", "dtype", "context"):
        assert replica[what] == original[what], what
    assert original["chunks"] > 0 and original["context"]
    assert replica["answer"] == original["answer"]
    assert original["answer"].strip()


def test_the_example_s_index_reaches_for_nothing_outside_the_machine(
    small_corpus, unzipped, tmp_path
):
    # The entry run by the local runner as the evaluation runs it: the network
    # cut, and every connection tried noted. With dones-example the example's
    # own code reads the corpus with Docling and embeds it with MiniLM, both
    # from next to main.py; the model is a fake, which loads nothing.
    out = tmp_path / "runs"
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [{"id": 8, "domain": "DONES", "question_type": "procedural", "question": QUESTION}]
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", DSQA_VARIANT="dones-example")
    environment["DSQA_ENGINE"] = str(ROOT / "tests" / "contract" / "fake_engine.py")
    command = [sys.executable, "-m", "tools.run_local", str(unzipped)]
    command += ["--model-id", "the/base-model", "--corpus-dir", str(small_corpus)]
    command += ["--questions", str(questions), "--out", str(out)]
    done = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    (run_dir,) = out.iterdir()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["blocked_connections"] == []
    assert meta["left_in_working_directory"] == []
    assert "[dsqa] the example's index: " in done.stdout + done.stderr
    assert "[rag] parsed 3 PDFs into" in done.stdout + done.stderr

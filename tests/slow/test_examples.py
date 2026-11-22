"""The organizers' two examples, run for real: through the local runner, and
the RAG example's retrieval through the retrieval measurement.

Left out of the default run. They need a GPU, a small Hugging Face model
already in the local cache and, for the RAG example, its two auxiliary
models and the development corpus (python -m tools.corpus). Ask for them
with:

    python -m pytest -m slow

They exercise the exact loading and generation code the evaluation runs,
with a model small enough for a laptop. What the model answers is not
checked: it says nothing about the base model.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools import corpus

ROOT = Path(__file__).resolve().parents[2]
# Any small chat model in Hugging Face format that fits the local GPU.
STAND_IN = os.environ.get("DSQA_STAND_IN_MODEL", "unsloth/Qwen3.5-0.8B")
# Three papers of the development corpus: enough for the RAG example to have
# something to search, few enough to parse in seconds.
SMALL_CORPUS = [
    "ai-attention-all-you-need",
    "ai-simple-framework-contrastive-learning",
    "ai-mamba-linear-time-sequence",
]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def stand_in_model_is_usable():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(STAND_IN, "config.json"), str):
        pytest.skip(f"{STAND_IN} is not in the local Hugging Face cache")


def run_example(tmp_path, submission, *arguments):
    """Run the local runner in a process of its own, as on the command line."""
    out = tmp_path / "runs"
    command = [sys.executable, "-m", "tools.run_local", str(submission)]
    command += ["--model-id", STAND_IN, "--out", str(out), *arguments]
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    (run_dir,) = out.iterdir()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    answers = (run_dir / "answers.jsonl").read_text(encoding="utf-8")
    return meta, [json.loads(line) for line in answers.splitlines()]


def test_the_zero_shot_example_runs_from_a_zip(tmp_path):
    # What gets uploaded: a zip with main.py at its root.
    archive = tmp_path / "zero_shot.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(ROOT / "01_zero_shot" / "main.py", "main.py")

    meta, rows = run_example(tmp_path, archive, "--limit", "1")

    assert [row["status"] for row in rows] == ["ok"]
    assert meta["blocked_connections"] == []
    assert meta["left_in_working_directory"] == []
    assert meta["gpu"]["peak_allocated_gb"] > 0


def test_the_rag_example_answers_over_a_corpus(tmp_path):
    papers = [corpus.CORPUS_DIR / f"{doc_id}.pdf" for doc_id in SMALL_CORPUS]
    needed = [ROOT / "02_rag" / "docling_models", ROOT / "02_rag" / "embed_model", *papers]
    missing = [path.relative_to(ROOT).as_posix() for path in needed if not path.exists()]
    if missing:
        pytest.skip("not downloaded: " + ", ".join(missing))
    folder = tmp_path / "corpus"
    folder.mkdir()
    for paper in papers:
        shutil.copy(paper, folder)

    meta, rows = run_example(tmp_path, ROOT / "02_rag", "--corpus-dir", str(folder), "--ids", "25")

    assert [row["status"] for row in rows] == ["ok"]
    assert meta["corpus_pdfs"] == len(SMALL_CORPUS)
    assert meta["blocked_connections"] == []
    assert meta["left_in_working_directory"] == []


def test_the_rag_example_retrieval_is_measured_as_it_is(tmp_path):
    # No model here, only the tokenizer the context budget is counted with:
    # the example's own parser, chunker, embedder and ranking over three
    # papers, two of them the sources of questions 22 and 25.
    papers = [corpus.CORPUS_DIR / f"{doc_id}.pdf" for doc_id in SMALL_CORPUS]
    needed = [ROOT / "02_rag" / "docling_models", ROOT / "02_rag" / "embed_model", *papers]
    missing = [path.relative_to(ROOT).as_posix() for path in needed if not path.exists()]
    if missing:
        pytest.skip("not downloaded: " + ", ".join(missing))
    folder = tmp_path / "corpus"
    folder.mkdir()
    for paper in papers:
        shutil.copy(paper, folder)

    command = [sys.executable, "-m", "tools.eval_retrieval", "--model-id", STAND_IN]
    command += ["--corpus-dir", str(folder), "--ids", "22,25"]
    command += ["--out", str(tmp_path / "runs"), "--cache", str(tmp_path / "cache")]
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    # A clean exit also says the chunks reported for each context are the
    # ones the example itself packs: the tool checks that on every question.
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    (run_dir,) = (tmp_path / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    questions = (run_dir / "questions.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in questions.splitlines()]
    assert summary["system"] == "02_rag"
    assert summary["corpus"]["documents"] == len(SMALL_CORPUS)
    assert summary["timings"]["parsed_now"] == len(SMALL_CORPUS)
    assert [row["id"] for row in rows] == [22, 25]
    # Docling keeps both passages, and with three papers to search a chunk of
    # the right one is always among those packed.
    assert all(row["parsed"] >= 0.8 for row in rows)
    assert all(row["document_in_context"] for row in rows)

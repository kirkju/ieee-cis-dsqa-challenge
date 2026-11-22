"""The contract of answer(), for our entry: run the way the evaluation runs it.

The entry is copied to a folder of its own, as it would be unzipped, and run
by the local runner: one process, one import of main.py, a call per
question, no internet, none of the repository's code importable, an empty
working directory. The base model's place is taken by a fake that answers at
once and fails when told to (tests/contract/fake_engine.py); the embedding
model is left out of the copy, so the corpus is ranked by words alone.

What is held to here is what the evaluation holds a submission to: answer()
takes its three arguments, and whatever the corpus, the question or the
model do, it returns a string and does not raise.
"""

import json
import shutil
import sys
import types
from pathlib import Path

import pymupdf
import pytest

from tests.unit.stand_ins import BODY, paper_pdf
from tools import run_local

ROOT = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).with_name("fake_engine.py")
NO_ANSWER = "No answer could be produced for this question."
TRAP = "Where is the nitrogen trap?"


def asked(*questions):
    return [
        {"id": number, "domain": "DONES", "question_type": "factual", "question": question}
        for number, question in enumerate(questions, 1)
    ]


@pytest.fixture
def entry(tmp_path):
    """The entry as it would be unzipped: main.py and its package, nothing else."""
    folder = tmp_path / "unzipped"
    folder.mkdir()
    shutil.copy(ROOT / "03_dsqa" / "main.py", folder / "main.py")
    shutil.copytree(
        ROOT / "03_dsqa" / "dsqa", folder / "dsqa", ignore=shutil.ignore_patterns("__pycache__")
    )
    return folder


@pytest.fixture
def corpus(tmp_path):
    """Two papers that can be read."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    paper_pdf(folder / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(folder / "a-trap.pdf")
    return folder


@pytest.fixture
def run(entry, tmp_path, monkeypatch, capsys):
    """Run the entry over some questions. Returns the run's summary, the rows
    of its answers and what it printed."""

    def run(questions, corpus_dir=None, fake="", folder=entry):
        monkeypatch.setenv("DSQA_ENGINE", str(FAKE))
        monkeypatch.setenv("DSQA_FAKE", fake)
        meta = run_local.run(
            folder,
            questions,
            model_id="the/base-model",
            corpus_dir=corpus_dir,
            out_dir=tmp_path / "runs",
            say=lambda line: None,
        )
        text = (tmp_path / "runs" / meta["run_id"] / "answers.jsonl").read_text(encoding="utf-8")
        return meta, [json.loads(line) for line in text.splitlines()], capsys.readouterr().out

    return run


def kept_the_contract(meta, rows) -> bool:
    answered = all(row["status"] == "ok" and row["answer"].strip() for row in rows)
    return answered and meta["completed"] and not run_local.broke_the_contract(meta)


def test_every_question_gets_a_string_from_the_passages_found_for_it(run, corpus):
    meta, rows, printed = run(asked(TRAP, "How is the lithium carried?", TRAP), corpus)
    assert kept_the_contract(meta, rows)
    assert meta["problems"] == []  # answer(question, corpus_dir, model_id), by position or by name
    assert [row["answer"] for row in rows] == [
        f"An answer with passages from the/base-model to: {TRAP}",
        "An answer with passages from the/base-model to: How is the lithium carried?",
        f"An answer with passages from the/base-model to: {TRAP}",
    ]
    # The corpus is read and the model loaded on the first call, and kept.
    assert printed.count("[dsqa] corpus indexed") == 1
    assert printed.count("[fake] the model is loaded") == 1
    assert "[dsqa] 2 documents in" in printed
    # Without its embedding model the entry says so and ranks by words.
    assert printed.count("[dsqa] no embedder, ranking by words alone") == 1


def test_nothing_is_fetched_and_nothing_is_left_behind(run, corpus):
    meta, rows, _ = run(asked(TRAP), corpus)
    assert kept_the_contract(meta, rows)
    assert meta["blocked_connections"] == []
    assert meta["left_in_working_directory"] == []


def damaged(folder: Path, kind: str) -> Path:
    """A corpus folder with something wrong with it."""
    if kind == "missing":
        return folder / "not-there"
    folder.mkdir(exist_ok=True)
    if kind == "not a PDF":
        (folder / "broken.pdf").write_bytes(b"%PDF-1.7\nthis is where the file ends")
    elif kind == "locked":
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "For those with the password.", fontsize=9)
            options = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "owner_pw": "o", "user_pw": "u"}
            document.save(folder / "locked.pdf", **options)
    elif kind == "scanned":
        with pymupdf.open() as document:
            page = document.new_page()
            picture = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
            picture.clear_with(200)
            page.insert_image(pymupdf.Rect(72, 72, 300, 300), pixmap=picture)
            document.save(folder / "scanned.pdf")
    return folder


@pytest.mark.parametrize("kind", ["missing", "empty", "not a PDF", "locked", "scanned"])
def test_a_corpus_with_nothing_to_read_leaves_the_model_to_answer_alone(run, tmp_path, kind):
    meta, rows, printed = run(asked(TRAP, TRAP), damaged(tmp_path / "bad", kind))
    assert kept_the_contract(meta, rows)
    assert rows[0]["answer"] == f"An answer without passages from the/base-model to: {TRAP}"
    assert "[dsqa] 0 documents in 0 chunks" in printed


@pytest.mark.parametrize("kind", ["not a PDF", "locked", "scanned"])
def test_a_bad_pdf_among_good_ones_costs_only_itself(run, corpus, kind):
    meta, rows, printed = run(asked(TRAP), damaged(corpus, kind))
    assert kept_the_contract(meta, rows)
    assert rows[0]["answer"].startswith("An answer with passages")
    assert "[dsqa] 2 documents in" in printed


QUESTIONS = {
    "empty": "",
    "blank": "  \n\t ",
    "a word": "Be7?",
    "with LaTeX": r"For $X = \{(x, y) \in c \oplus_\infty c : \lim x = 0\}$, is $n(X^*) < 1$?",
    "not in English": "¿Dónde está la trampa de nitrógeno, y a qué temperatura opera?",
    "enormous": "Where is the nitrogen trap, and why? " * 3000,
}


@pytest.mark.parametrize("kind", QUESTIONS)
def test_any_question_gets_a_string(run, corpus, kind):
    meta, rows, _ = run(asked(QUESTIONS[kind], TRAP), corpus)
    assert kept_the_contract(meta, rows)
    assert rows[1]["answer"] == f"An answer with passages from the/base-model to: {TRAP}"


def test_a_model_out_of_memory_is_asked_again_with_less(run, corpus):
    meta, rows, printed = run(asked(TRAP, TRAP), corpus, fake="oom_once")
    assert kept_the_contract(meta, rows)
    assert all(row["answer"].startswith("An answer with") for row in rows)
    assert printed.count("out of memory") == 1


def test_a_model_that_fails_on_every_question_leaves_the_passages(run, corpus, tmp_path):
    meta, rows, _ = run(asked(TRAP, TRAP), corpus, fake="always_raises")
    assert kept_the_contract(meta, rows)
    assert all("The trap is in the dump tank." in row["answer"] for row in rows)
    # And with no corpus either, a sentence that says there is no answer.
    meta, rows, _ = run(asked(TRAP), tmp_path / "nowhere", fake="always_raises")
    assert kept_the_contract(meta, rows)
    assert rows[0]["answer"] == NO_ANSWER


def test_a_model_that_cannot_be_loaded_is_not_loaded_again(run, corpus):
    meta, rows, printed = run(asked(TRAP, TRAP, TRAP), corpus, fake="load_fails")
    assert kept_the_contract(meta, rows)
    assert all("The trap is in the dump tank." in row["answer"] for row in rows)
    assert printed.count("[fake] the model is loaded") == 1
    assert printed.count("[dsqa] the model could not be loaded") == 1


def test_a_reply_that_states_no_answer_is_asked_for_it_and_both_are_returned(run, corpus):
    meta, rows, printed = run(asked(TRAP), corpus, fake="no_final")
    assert kept_the_contract(meta, rows)
    # What the model said when asked for the answer alone comes first, and
    # its first reply after it. The fake says "without passages" of that
    # second question because the passages are not the last thing it is told.
    stated, working = rows[0]["answer"].split(chr(10) * 2)
    assert stated.startswith("An answer without passages from the/base-model to: State the answer")
    assert working == f"An answer with passages from the/base-model to: {TRAP}"
    assert printed.count("the answer was asked for apart") == 1


@pytest.mark.parametrize("fake", ["blank", "not_a_string"])
def test_a_reply_that_is_no_text_is_not_returned(run, corpus, fake):
    meta, rows, _ = run(asked(TRAP), corpus, fake=fake)
    assert kept_the_contract(meta, rows)
    assert "The trap is in the dump tank." in rows[0]["answer"]


def test_it_runs_where_cv2_cannot_be_imported(run, corpus, monkeypatch):
    # The evaluation image lacks the system library cv2 needs.
    monkeypatch.setitem(sys.modules, "cv2", None)
    meta, rows, _ = run(asked(TRAP), corpus)
    assert kept_the_contract(meta, rows)
    assert rows[0]["answer"].startswith("An answer with passages")


def test_without_its_package_the_entry_asks_the_bare_model(run, corpus, tmp_path, monkeypatch):
    # main.py alone, as if the package next to it could not be imported. The
    # model is then loaded by main.py itself: here, a stand-in for transformers.
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(ROOT / "03_dsqa" / "main.py", alone / "main.py")
    loaded = []

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **options):
            prompt = types.SimpleNamespace(shape=(1, 2))
            tokens = {"input_ids": prompt, "question": messages[-1]["content"]}
            return types.SimpleNamespace(to=lambda device: tokens)

        def decode(self, tokens, skip_special_tokens=False):
            return " The bare model's answer. "

    class Model:
        device = "cuda:0"

        def eval(self):
            pass

        def generate(self, **options):
            return [[1, 2, 3]]

    def model_from(model_id, **options):
        loaded.append(model_id)
        return Model()

    class NoGrad:
        def __enter__(self):
            pass

        def __exit__(self, *error):
            return False

    torch = types.SimpleNamespace(bfloat16="bf16", no_grad=NoGrad)
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda model_id: Tokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_from),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    meta, rows, printed = run(asked(TRAP, TRAP), corpus, folder=alone)
    assert kept_the_contract(meta, rows)
    assert [row["answer"] for row in rows] == ["The bare model's answer."] * 2
    assert loaded == ["the/base-model"]  # once, by the name the evaluation gave
    assert printed.count("[dsqa] dsqa's engine could not be imported") == 1

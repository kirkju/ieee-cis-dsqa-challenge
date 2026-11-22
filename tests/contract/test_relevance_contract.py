"""The contract of answer() with the base model as a judge of relevance
(relevance.py, in the dones-plain way of route.py): whatever the scoring does,
every question is answered with a string, and the question on IFMIF-DONES
says in the log what the scoring did or why it did not. Run as
tests/contract/test_answer.py runs the entry: copied to a folder of its own,
by the local runner, with the fake engine in the base model's place, here
given a score() of its own."""

import json
import shutil
from pathlib import Path

import pymupdf
import pytest

from tests.unit.stand_ins import BODY, paper_pdf
from tools import run_local

ROOT = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).with_name("fake_engine.py")

# The fake engine, with a score() that likes the nitrogen trap; SCORE_FAKE
# makes it raise, give nothing, give NaN or the same logits for every
# passage, or not be there at all.
SCORING = """import importlib.util
import os

spec = importlib.util.spec_from_file_location("fake_engine", os.environ["FAKE_ENGINE_FILE"])
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)


def load(model_id):
    engine = fake.load(model_id)
    how = os.environ.get("SCORE_FAKE", "")
    generate = engine.generate

    def counted(*args, **kwargs):
        print("[fake] a reply is asked for")
        return generate(*args, **kwargs)

    engine.generate = counted

    def score(conversations, labels=("Yes", "No"), most_tokens=8000):
        if how == "raises":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if how == "none":
            return None
        if how == "nan":
            return [[float("nan"), 0.0] for c in conversations]
        if how == "equal":
            return [[1.0, 0.0] for c in conversations]
        if tuple(labels) != ("Yes", "No"):
            return [[0.0] * len(labels) for c in conversations]
        return [[1.0 if "trap" in c[1]["content"] else -1.0, 0.0] for c in conversations]

    def first(conversation, labels=("Yes", "No")):
        rows = score([conversation], labels)
        return None if rows is None else rows[0]

    if how != "absent":
        engine.score, engine.first = score, first
    return engine
"""


@pytest.fixture
def entry(tmp_path):
    folder = tmp_path / "unzipped"
    folder.mkdir()
    shutil.copy(ROOT / "03_dsqa" / "main.py", folder / "main.py")
    shutil.copytree(
        ROOT / "03_dsqa" / "dsqa", folder / "dsqa", ignore=shutil.ignore_patterns("__pycache__")
    )
    return folder


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    paper_pdf(folder / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(folder / "a-trap.pdf")
    return folder


QUESTIONS = ["Where is the nitrogen trap?", "How is the lithium of the IFMIF-DONES loop carried?"]


@pytest.fixture
def run(entry, corpus, tmp_path, monkeypatch, capfd):
    engine = tmp_path / "scoring_engine.py"
    engine.write_text(SCORING, encoding="utf-8")

    def run(variants, score="", fake=""):
        """The relevance lines and all that was printed; the answers are
        kept as run.answers."""
        monkeypatch.setenv("DSQA_ENGINE", str(engine))
        monkeypatch.setenv("FAKE_ENGINE_FILE", str(FAKE))
        monkeypatch.setenv("DSQA_FAKE", fake)
        monkeypatch.setenv("DSQA_VARIANT", variants)
        monkeypatch.setenv("SCORE_FAKE", score)
        asked = [
            {"id": number, "domain": "DONES", "question_type": "factual", "question": question}
            for number, question in enumerate(QUESTIONS, 1)
        ]
        meta = run_local.run(
            entry,
            asked,
            model_id="the/base-model",
            corpus_dir=corpus,
            out_dir=tmp_path / "runs",
            say=lambda line: None,
        )
        text = (tmp_path / "runs" / meta["run_id"] / "answers.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines()]
        assert meta["completed"] and not run_local.broke_the_contract(meta)
        assert all(row["status"] == "ok" and row["answer"].strip() for row in rows)
        run.answers = [row["answer"] for row in rows]
        printed = capfd.readouterr().out
        return [line for line in printed.splitlines() if "relevance:" in line], printed

    return run


@pytest.mark.parametrize(
    ("variants", "score", "said"),
    [
        ("dones-plain,llm-rank", "", "relevance: llm-rank scored "),
        ("dones-plain,llm-rank,llm-wide,llm-one", "", " one by one in "),
        (
            "dones-plain,llm-rank,llm-wide,llm-one",
            "absent",
            "relevance: not applied (the engine cannot score one by one)",
        ),
        ("dones-plain,llm-rank,llm-gate,llm-wide", "", "relevance: llm-rank+llm-gate scored "),
        (
            "dones-plain,llm-rank,llm-wide",
            "nan",
            "relevance: not applied (the scores rank nothing: ",
        ),
        (
            "dones-plain,llm-rank,llm-wide",
            "equal",
            "relevance: not applied (the scores rank nothing: all ",
        ),
        (
            "dones-plain,llm-rank,llm-gate,llm-doc,llm-pick,llm-wide",
            "",
            "relevance: llm-rank+llm-gate+llm-doc+llm-pick scored ",
        ),
        (
            "dones-plain,llm-rank,llm-gate,llm-wide",
            "raises",
            "relevance: not applied (the scoring failed: RuntimeError: CUDA out of memory",
        ),
        (
            "dones-plain,llm-rank,llm-gate,llm-wide",
            "none",
            "relevance: not applied (no scores came back, out of memory)",
        ),
        (
            "dones-plain,llm-rank,llm-gate,llm-wide",
            "absent",
            "relevance: not applied (the engine cannot score)",
        ),
        ("dones-plain", "", "relevance: not applied (no llm- adjustment chosen)"),
    ],
)
def test_every_question_is_answered_and_the_dones_one_says_what_the_scoring_did(
    run, variants, score, said
):
    lines, printed = run(variants, score)
    (line,) = lines
    assert line.startswith("[dsqa] question 2: ") and said in line
    assert " tokens in " in line and line.endswith(" fragments")
    assert "answered the dones-plain way" in printed


@pytest.mark.parametrize("score", ["nan", "equal"])
def test_scores_that_rank_nothing_leave_the_context_as_an_engine_that_cannot_score(run, score):
    variants = "dones-plain,llm-rank,llm-gate,llm-wide"
    (unscored,), _ = run(variants, "absent")
    (line,), _ = run(variants, score)
    assert "relevance: not applied (the scores rank nothing: " in line
    assert line.split(", context of ")[1] == unscored.split(", context of ")[1]


# llm-acted over the DONES half of the stack of 2026-11-15, and its sentence.
ACTED = "dones-plain,llm-rank,llm-wide,llm-acted"
UNANSWERED = "This entry does not answer this question."
ASKED_FOR = "[fake] a reply is asked for"


def test_llm_acted_answers_the_dones_question_scored_in_full_as_without_it(run):
    run("dones-plain,llm-rank,llm-wide")
    without = list(run.answers)
    _, printed = run(ACTED)
    # The first question is not on IFMIF-DONES; the second is, scored in full.
    assert run.answers == [UNANSWERED, without[1]] and without[1] != UNANSWERED
    assert printed.count(ASKED_FOR) == 1
    said = "[dsqa] question 1: llm-acted: not told on IFMIF-DONES, the model not asked"
    assert said in printed
    assert "[dsqa] question 2: llm-acted: scored in full, answered" in printed


@pytest.mark.parametrize(
    ("score", "fake", "why", "replies"),
    [
        ("absent", "", "not scored in full (the engine cannot score)", 0),
        ("raises", "", "not scored in full (the scoring failed: RuntimeError: CUDA out", 0),
        ("none", "", "not scored in full (no scores came back, out of memory)", 0),
        ("nan", "", "not scored in full (the scores rank nothing: ", 0),
        ("equal", "", "not scored in full (the scores rank nothing: all ", 0),
        ("", "always_raises", "the dones-plain way failed (ValueError)", 1),
        ("", "blank", "scored in full, and the dones-plain way gave no answer", 1),
        ("", "load_fails", "no model", 0),
    ],
)
def test_llm_acted_gives_every_other_question_the_sentence(run, score, fake, why, replies):
    _, printed = run(ACTED, score, fake)
    assert run.answers == [UNANSWERED, UNANSWERED]
    assert f"[dsqa] question 2: llm-acted: {why}" in printed
    # No reply is asked for but the dones-plain one, and no other way after it.
    assert printed.count(ASKED_FOR) == replies


def test_without_dones_plain_the_scoring_is_never_asked(run):
    lines, printed = run("llm-rank,llm-gate,llm-wide")
    assert lines == [] and "answered the dones-plain way" not in printed


# llm-proof with llm-one over the DONES half of the stack of 2026-11-15.
PROOF = "dones-plain,llm-rank,llm-wide,llm-one,llm-proof"
STACK_DONES = "dones-plain,llm-rank,llm-wide"


def test_llm_proof_with_an_engine_that_scores_answers_as_the_stack(run):
    run(STACK_DONES)
    stack = list(run.answers)
    _, printed = run(PROOF)
    assert run.answers == stack and UNANSWERED not in stack
    assert "[dsqa] question 2: llm-proof: scored in full, the first: answered" in printed
    assert " one by one in " in printed


@pytest.mark.parametrize("score", ["absent", "raises", "none", "nan", "equal"])
def test_llm_proof_without_scores_that_rank_answers_ai_and_gives_dones_the_sentence(run, score):
    run(STACK_DONES, score)
    stack = list(run.answers)
    _, printed = run(PROOF, score)
    assert run.answers == [stack[0], UNANSWERED] and stack[1] != UNANSWERED
    assert "[dsqa] question 2: llm-proof: not scored (" in printed
    assert ", and never yet, the fixed sentence" in printed

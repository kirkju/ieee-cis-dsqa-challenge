"""dsqa/focus.py on the first entry: the document a question on AI papers is
about (doc-focus), a quarter of its context kept for the ranking's best
(doc-reserve), and no focus on a document whose head does not say the name
the question gives (doc-name)."""

import hashlib
import json
import types

import numpy as np
import pymupdf
import pytest
from dsqa import config, domain, focus, plain, prompt, retrieval, route
from dsqa.answering import Answerer
from dsqa.focus import AS_BEFORE, ONE, Focus

from tests.unit.stand_ins import BODY, Letters, paper_pdf


def words(text):
    return len(text.split())


class Scores:
    def __init__(self, scores):
        self.given = np.asarray(scores, float)

    def scores(self, question):
        return self.given


def fake_index(chunks, documents, scores, embeddings=True):
    return types.SimpleNamespace(
        chunks=chunks,
        documents=np.asarray(documents),
        bm25=Scores(scores),
        embeddings=np.zeros((len(chunks), 2)) if embeddings else None,
    )


@pytest.fixture
def told(monkeypatch):
    """What the routing tells, set by the test: AI unless said otherwise."""
    said = {"domain": domain.AI}
    monkeypatch.setattr(route, "decided", lambda question, index, order=None: said["domain"])
    return said


def test_the_two_rankings_agree_on_one_document_first():
    documents = np.array([0, 0, 1, 1, 2, 2])
    # The best chunk is document 0's, and so is the best sum of three by words.
    first, agreed = focus.candidates([0, 2, 4], documents, [5, 4, 3, 0, 1, 0])
    assert first[:2] == [0, 1] and agreed
    # By words document 1 leads: they disagree, and it is a candidate too.
    first, agreed = focus.candidates([0, 4], documents, [1, 0, 5, 4, 0, 0])
    assert first == [0, 2, 1] and not agreed
    # Two documents first by words together single out neither.
    assert not focus.candidates([0, 2], documents, [3, 0, 3, 0, 0, 0])[1]
    # No word in common: the ranking by words says nothing.
    assert not focus.candidates([0, 2], documents, [0] * 6)[1]


def test_what_a_context_is_made_of(monkeypatch, told):
    monkeypatch.setenv("DSQA_VARIANT", "doc-focus")
    chunks = [f"T{d}\n\nchunk {n}" for n, d in enumerate([0, 0, 1, 1])]
    agreeing = fake_index(chunks, [0, 0, 1, 1], [5, 4, 0, 0])
    assert focus.decided(agreeing, "Q?", []).why == "nothing ranked"
    made = focus.decided(agreeing, "Q?", [0, 2, 1, 3])
    assert (made.how, made.documents, made.agreed) == (ONE, [0], True)
    disagreeing = fake_index(chunks, [0, 0, 1, 1], [0, 0, 5, 4])
    made = focus.decided(disagreeing, "Q?", [0, 2, 1, 3])
    assert made.how == AS_BEFORE and made.why.startswith("the two rankings")
    blind = fake_index(chunks, [0, 0, 1, 1], [5, 4, 0, 0], embeddings=False)
    assert focus.decided(blind, "Q?", [0, 2]).why.startswith("no embedder")
    told["domain"] = domain.DONES
    assert focus.decided(agreeing, "Q?", [0, 2, 1, 3]).why == "told to be on IFMIF-DONES"


def test_one_document_whole_when_it_fits_else_its_best_chunks_with_their_neighbours(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "doc-focus")
    documents = np.array([0, 1, 1, 1, 1, 2])
    chunks = [f"T{d}\n\nw w" for d in documents]  # three words each
    order = [3, 0, 1, 5, 2, 4]
    made = Focus(ONE, [1])
    context, used = focus.placed(made, order, chunks, documents, words, 100)
    assert sorted(n for n, _ in context) == [1, 2, 3, 4] and made.whole and used == 12
    # Too long for 10 tokens: chunk 3 first, then the one after it and the one
    # before it, and no more.
    made = Focus(ONE, [1])
    context, used = focus.placed(made, order, chunks, documents, words, 10)
    assert [n for n, _ in context] == [3, 4, 2] and not made.whole and used == 9


def test_the_reserve_keeps_the_rankings_best_before_the_document(monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "doc-focus,doc-reserve")
    documents = np.array([0, 1, 1, 1, 1, 2])
    chunks = [f"T{d}\n\nw w" for d in documents]  # three words each
    order = [0, 5, 3, 1, 2, 4]
    made = Focus(ONE, [1])
    context, used = focus.placed(made, order, chunks, documents, words, 32)
    numbers = [n for n, _ in context]
    # A quarter, 8 tokens: the ranking's two best, of documents 0 and 2. The
    # document has the rest and comes first, so that it is printed last.
    assert numbers[-2:] == [0, 5]
    assert sorted(numbers[:-2]) == [1, 2, 3, 4] and made.whole
    assert used == 18 and len(set(numbers)) == len(numbers)
    assert prompt.context_of(context, documents).endswith("w w")
    shown = prompt.context_of(context, documents)
    assert shown.index("Document: T1") > shown.index("Document: T0")
    # When the ranking's best are the document's own, the quarter is of it too.
    made = Focus(ONE, [1])
    context, _ = focus.placed(made, [3, 2, 0, 5, 1, 4], chunks, documents, words, 32)
    assert {int(documents[n]) for n, _ in context} == {1}


def test_the_names_a_question_gives():
    for word in ("SimCLR", "MoCo", "GPT-3", "T5", "word2vec", "BERT", "ResNet-50"):
        assert focus.shaped(word), word
    for word in ("In", "The", "Cohen", "a", "x"):
        assert not focus.shaped(word), word
    chunks = [
        f"Paper {d}\n\nWe present SimCLR and compare BERT." if d == 0 else f"Paper {d}\n\nBERT."
        for d in range(10)
    ]
    index = fake_index(chunks, list(range(10)), [0] * 10)
    heads = focus.heads_of(index)
    # BERT is said by every document: not rare. SimCLR by one of ten.
    assert heads.names("In SimCLR's paper, why is BERT used?") == ["SimCLR"]
    assert focus.heads_of(index) is heads
    assert focus.says("It is SimCLR's.", "SimCLR") and not focus.says("SimCLRv2", "SimCLR")


def test_a_name_counts_only_when_a_candidates_head_says_it():
    far = "x " * 1_100  # past the head's 2,000 characters
    chunks = [
        "Contrastive cluster assignments\n\nWe compare with others." + far + "SimCLR",
        "A simple framework\n\nThis paper presents SimCLR, a simple framework.",
        "Momentum contrast\n\nWe improve on SimCLR with MoCo.",
    ] + [f"Other {d}\n\nNothing here." for d in range(3, 40)]
    index = fake_index(chunks, list(range(40)), [0] * 40)
    question = "In SimCLR's paper, what is the weakness of one fixed augmentation?"
    # The document chosen says it only far into its text: vetoed.
    assert focus.vetoed(index, question, [0, 2], 0) == ["SimCLR"]
    assert focus.vetoed(index, question, [1, 0], 1) == []
    # A name no candidate's head says counts for nothing.
    assert focus.vetoed(index, "On the MNIST-like data of PEMS?", [0, 1], 0) == []


def test_doc_name_leaves_the_context_as_before_when_vetoed(monkeypatch, told):
    chunks = [
        "Contrastive cluster assignments\n\nAgainst SwAV-like models.",
        "Contrastive cluster assignments\n\nMore of it.",
        "A simple framework\n\nThis paper presents SimCLR.",
    ] + [f"Other {d}\n\nNothing." for d in range(3, 30)]
    documents = [0, 0, 1] + list(range(2, 29))
    index = fake_index(chunks, documents, [5, 4, 3] + [0] * 27)
    question = "In SimCLR's paper, what is the weakness?"
    monkeypatch.setenv("DSQA_VARIANT", "doc-focus")
    assert focus.decided(index, question, [0, 2, 1]).how == ONE
    monkeypatch.setenv("DSQA_VARIANT", "doc-focus,doc-name")
    made = focus.decided(index, question, [0, 2, 1])
    assert made.how == AS_BEFORE and "does not say SimCLR" in made.why


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    paper_pdf(tmp_path / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(tmp_path / "a-trap.pdf")
    monkeypatch.setattr(config, "CHUNK_TOKENS", 24)
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 80)
    return tmp_path


def test_what_fails_in_the_focus_gives_the_context_as_before(corpus, monkeypatch, capsys):
    index = retrieval.build(corpus, Letters(), lambda line: None)
    monkeypatch.setenv("DSQA_VARIANT", "")
    before = index.context("Where is the nitrogen trap?", words)
    monkeypatch.setenv("DSQA_VARIANT", "doc-focus,doc-name,doc-reserve")

    def fail(*arguments):
        raise ZeroDivisionError("no")

    monkeypatch.setattr(focus, "decided", fail)
    after = index.context("Where is the nitrogen trap?", words)
    assert after[:2] == before[:2] and index.focused is None
    assert "the context as it was: ZeroDivisionError" in capsys.readouterr().out


class Engine:
    def __init__(self):
        self.conversations = []

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        self.conversations.append(messages)
        return "It is in the tank.\nFinal answer: In the dump tank."

    def release(self):
        pass


def ask(corpus, monkeypatch, names, question):
    monkeypatch.setenv("DSQA_VARIANT", names)
    said, engine = [], Engine()
    Answerer(corpus, engine, embedder=Letters(), say=said.append).answer(question)
    return engine.conversations, said


def fingerprint(conversations) -> str:
    return hashlib.sha256(json.dumps(conversations, sort_keys=True).encode()).hexdigest()


def test_with_dones_plain_the_dones_questions_are_asked_as_before(corpus, monkeypatch):
    question = "Where is the lithium loop of the IFMIF-DONES plant?"
    alone, _ = ask(corpus, monkeypatch, "dones-plain", question)
    together, _ = ask(corpus, monkeypatch, "dones-plain,doc-focus,doc-name,doc-reserve", question)
    assert fingerprint(together) == fingerprint(alone)
    assert alone[0][0]["content"] == plain.SYSTEM_PROMPT


def test_a_question_on_ai_papers_keeps_its_instruction_and_turn_and_says_its_focus(
    corpus, monkeypatch
):
    question = "Which learning rate does the neural transformer use?"
    names = "dones-plain,doc-focus,doc-name,doc-reserve"
    (conversation,), said = ask(corpus, monkeypatch, names, question)
    assert conversation[0]["content"] == prompt.SYSTEM
    assert conversation[1]["content"].startswith("Passages retrieved from the document collection:")
    assert conversation[1]["content"].endswith(f"---\n\nQuestion: {question}")
    (line,) = [line for line in said if "focused" in line]
    assert line.startswith("[dsqa] question 1: ") and (
        "focused on document" in line or "not focused" in line
    )
    # Without doc-focus no such line is written.
    _, said = ask(corpus, monkeypatch, "dones-plain", question)
    assert not [line for line in said if "focused" in line]

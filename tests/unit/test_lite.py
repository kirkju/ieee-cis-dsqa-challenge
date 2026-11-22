"""dsqa/lite.py: the organizers' RAG example's code over the entry's own
text, for the questions on IFMIF-DONES (dones-lite). The example is a
stand-in with the shape of 02_rag/main.py (its _chunk_markdown, its _cache
and its answer()), the embedder a stand-in, and no model is loaded."""

import sys
import types

import numpy as np
import pymupdf
import pytest
from dsqa import config, lite
from dsqa.answering import Answerer

from tests.unit.stand_ins import BODY, Letters, paper_pdf

DONES_QUESTION = "Where is the lithium loop of the IFMIF-DONES plant?"
AI_QUESTION = "Which learning rate does the neural transformer use?"


def example_like():
    """A module with the shape of the example, which says what it was asked."""
    main = types.ModuleType("example_like")
    main._cache = {}
    main.asked = []
    main.EMBED_MODEL_PATH = "the example's embed_model"

    def _chunk_markdown(text):
        return [part for part in text.split("\n\n") if part.strip()]

    def answer(question, corpus_dir, model_id):
        tokenizer, model, chunks, embeddings, embedder = main._cache[model_id]
        main.asked.append((question, model_id, model.generation_config.max_time, len(chunks)))
        if question == "fail":
            raise RuntimeError("the example fell over")
        return f"From {len(chunks)} chunks."

    main._chunk_markdown = _chunk_markdown
    main.answer = answer
    return main


class Embedder:
    """In the place of sentence-transformers' SentenceTransformer."""

    made: list = []

    def __init__(self, path, device=None):
        Embedder.made.append((path, device))
        self.letters = Letters()

    def encode(self, texts, **options):
        return self.letters.encode(texts)


@pytest.fixture
def example(monkeypatch):
    main = example_like()
    monkeypatch.setitem(lite._kept, "main", main)
    monkeypatch.setitem(lite._kept, "index", None)
    # sentence-transformers stood in for by a module of its name, and never
    # imported: here, importing it starts CUDA, and the runner's tests later
    # in the suite read a started CUDA as a submission that used the GPU.
    stand_in = types.ModuleType("sentence_transformers")
    stand_in.SentenceTransformer = Embedder
    monkeypatch.setitem(sys.modules, "sentence_transformers", stand_in)
    Embedder.made = []
    return main


class Engine:
    def __init__(self, reply="Final answer: In the dump tank."):
        self.reply, self.conversations, self.released = reply, [], 0
        self.model = types.SimpleNamespace(generation_config=types.SimpleNamespace(max_time=None))
        self.tokenizer = "the tokenizer"

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        self.conversations.append(messages)
        return self.reply

    def release(self):
        self.released += 1


def test_the_example_cuts_and_embeds_the_entrys_own_texts(example):
    said = []
    lite.prepare(["A title\n\nOne paragraph.\n\nAnother.", "Second\n\nText."], said.append)
    assert lite.ready()
    chunks, embeddings, embedder = lite._kept["index"]
    assert chunks == ["A title", "One paragraph.", "Another.", "Second", "Text."]
    assert embeddings.shape == (5, 26) and embeddings.dtype == np.float32
    # The example's own embedder, on the CPU, as its _build_index() makes it.
    assert Embedder.made == [("the example's embed_model", "cpu")]
    assert any("the lite index: 5 chunks" in line for line in said)


def test_an_index_that_cannot_be_made_is_not_kept(example, monkeypatch):
    def fails(text):
        raise ValueError("no paragraphs")

    monkeypatch.setattr(example, "_chunk_markdown", fails)
    said = []
    lite.prepare(["A text."], said.append)
    assert not lite.ready()
    assert any("could not be made" in line for line in said)


def test_the_example_answers_with_the_engines_model_within_its_seconds(example):
    lite.prepare(["A title\n\nOne paragraph."], lambda line: None)
    engine = Engine()
    assert lite.answer("Where?", engine, seconds=73.0) == "From 2 chunks."
    tokenizer, model, *_ = example._cache[lite.KEY]
    assert (tokenizer, model) == ("the tokenizer", engine.model)
    # The limit is set for the call and taken off after it.
    assert example.asked == [("Where?", lite.KEY, 73.0, 2)]
    assert engine.model.generation_config.max_time is None
    assert engine.released == 1


@pytest.mark.parametrize("case", ["not ready", "no model", "no time", "fails"])
def test_without_an_answer_of_the_example_there_is_none(example, case):
    if case != "not ready":
        lite.prepare(["A title\n\nOne paragraph."], lambda line: None)
    engine = Engine()
    if case == "no model":
        engine.model = None
    seconds = config.LEAST_SECONDS - 1 if case == "no time" else 60.0
    question = "fail" if case == "fails" else "Where?"
    assert lite.answer(question, engine, seconds) is None


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


def test_dones_lite_answers_the_dones_questions_from_the_entrys_texts(corpus, example, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-lite")
    engine, said = Engine(), []
    answerer = Answerer(corpus, engine, embedder=Letters(), say=said.append)
    # The texts the entry read, cut by the example when the answerer was built.
    assert lite.ready()
    assert lite._kept["index"][0] == [
        chunk for text in answerer.index.texts for chunk in example._chunk_markdown(text)
    ]
    reply = answerer.answer(DONES_QUESTION)
    assert reply.startswith("From ") and engine.conversations == []
    assert any("answered the dones-lite way" in line for line in said)
    # A question on AI papers is the entry's own.
    assert answerer.answer(AI_QUESTION) == "In the dump tank."
    assert len(example.asked) == 1


def test_without_its_index_the_entrys_own_way_answers(corpus, example, monkeypatch):
    monkeypatch.setenv("DSQA_VARIANT", "dones-lite")
    monkeypatch.setattr(lite, "prepare", lambda texts, say=print: None)
    engine, said = Engine(), []
    answerer = Answerer(corpus, engine, embedder=Letters(), say=said.append)
    assert answerer.answer(DONES_QUESTION) == "In the dump tank."
    assert any("dones-lite gave no answer" in line for line in said)

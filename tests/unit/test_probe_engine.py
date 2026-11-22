"""tools/probe_engine.py: what a prompt costs, by its length.

The engine is scripted and keeps its own clock: it takes a time to read a
prompt that grows with the prompt, writes at a fixed rate, and runs out of
memory past a length it is told. The corpus is one paper made on the spot.
"""

import types

import pytest
from dsqa import config, retrieval

from tests.unit.stand_ins import Letters, paper_pdf
from tools import probe_engine


class OutOfMemoryError(RuntimeError):
    """Named as torch names it."""


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Engine:
    """Reads a thousand words a second and writes ten; a prompt of more than
    `room` words runs it out of memory."""

    def __init__(self, clock, room=10_000):
        self.clock, self.room = clock, room
        self.asked: list[tuple[int, int]] = []
        self.released = 0

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        words = sum(self.count(message["content"]) for message in messages)
        self.asked.append((words, max_new_tokens))
        if words > self.room:
            raise OutOfMemoryError("CUDA out of memory")
        self.clock.now += words / 1000 + (max_new_tokens - 1) / 10
        return " ".join(["word"] * max_new_tokens)

    def release(self):
        self.released += 1


@pytest.fixture
def index(tmp_path, monkeypatch):
    """A paper of some 1,500 words, in chunks of 24."""
    body = "\n".join(["The main loop carries the liquid lithium to the target and back."] * 60)
    paper_pdf(tmp_path / "loop.pdf", body)
    monkeypatch.setattr(config, "CHUNK_TOKENS", 24)
    return retrieval.build(tmp_path, Letters(), say=lambda line: None)


def test_a_prompt_is_read_once_for_a_token_and_once_for_the_rate():
    clock = Clock()
    engine = Engine(clock)
    messages = [{"role": "user", "content": "word " * 2000}]
    row = probe_engine.measure(engine, messages, written=51, clock=clock)
    assert [limit for _, limit in engine.asked] == [1, 51]
    assert row["first_token_s"] == 2.0  # 2,000 words read
    assert row["written"] == 51 and row["written_s"] == 7.0
    assert row["tokens_per_s"] == 10.0  # the 50 tokens after the first, in the 5 s they took
    assert engine.released == 2
    assert "peak_gb" not in row  # no GPU was used


def test_the_sizes_are_tried_smallest_first_and_each_is_the_entry_s_context(index):
    clock = Clock()
    engine = Engine(clock)
    said = []
    rows = probe_engine.probe(engine, index, "lithium", (0, 100, 400), 8, said.append, clock)
    assert [row["context_budget"] for row in rows] == [0, 100, 400]
    alone, small, large = rows
    assert alone["chunks"] == 0 and alone["context_tokens"] == 0
    assert 0 < small["context_tokens"] <= 100 < large["context_tokens"] <= 400
    assert alone["prompt_tokens"] < small["prompt_tokens"] < large["prompt_tokens"]
    assert small["first_token_s"] < large["first_token_s"]
    assert all(row["tokens_per_s"] == 10.0 for row in rows)
    assert len(said) == 3 and "context    400" in said[-1]


def test_the_size_that_runs_out_of_memory_is_the_last_one_tried(index):
    # A context of 800 words makes a prompt of some 600: the titles the
    # chunks repeat are said once in what the model reads.
    engine = Engine(Clock(), room=500)
    rows = probe_engine.probe(engine, index, "lithium", (100, 800, 1_000), 8, lambda line: None)
    assert [row["context_budget"] for row in rows] == [100, 800]
    assert "error" not in rows[0]
    assert rows[1]["out_of_memory"] and rows[1]["error"].startswith("OutOfMemoryError")
    assert "first_token_s" not in rows[1]


def test_each_size_is_handed_over_as_soon_as_it_is_measured(index):
    # So that a model that takes the process down leaves the sizes before it.
    engine = Engine(Clock(), room=500)
    kept = []

    def keep(row):
        kept.append((row["context_budget"], len(engine.asked)))

    rows = probe_engine.probe(engine, index, "lithium", (100, 800), 8, lambda line: None, keep=keep)
    assert len(rows) == 2
    # The first after its two requests, the second after the one that failed.
    assert kept == [(100, 2), (800, 3)]


def test_a_context_the_corpus_cannot_fill_is_the_last_one_tried(index):
    engine = Engine(Clock(), room=10**6)
    sizes = (1_000, 50_000, 90_000)
    rows = probe_engine.probe(engine, index, "lithium", sizes, 8, lambda line: None)
    assert [row["context_budget"] for row in rows] == [1_000, 50_000]
    assert rows[1]["context_tokens"] < 25_000


def test_without_a_corpus_only_the_bare_request_is_measured():
    engine = Engine(Clock())
    rows = probe_engine.probe(engine, None, "lithium", (0, 100, 400), 8, lambda line: None)
    assert [row["context_budget"] for row in rows] == [0]


def test_the_prompt_is_counted_with_the_model_s_template_when_there_is_one():
    engine = Engine(Clock())
    messages = [{"role": "user", "content": "one two three"}]
    assert probe_engine.prompt_tokens(engine, messages) == 3

    def apply_chat_template(messages, **options):
        assert options["add_generation_prompt"]
        return {"input_ids": types.SimpleNamespace(shape=(1, 9))}

    engine.tokenizer = types.SimpleNamespace(apply_chat_template=apply_chat_template)
    assert probe_engine.prompt_tokens(engine, messages) == 9


def test_the_summary_says_where_memory_ran_out_and_what_the_entry_s_context_costs():
    rows = [
        {"context_budget": 0, "prompt_tokens": 200, "first_token_s": 0.1, "tokens_per_s": 21.0},
        {
            "context_budget": 10_000,
            "prompt_tokens": 10_300,
            "first_token_s": 3.2,
            "tokens_per_s": 19.0,
            "peak_gb": 71.5,
        },
        {"context_budget": 20_000, "prompt_tokens": 20_400, "first_token_s": 7.0, "peak_gb": 78.0},
        {"context_budget": 30_000, "prompt_tokens": 30_500, "error": "x", "out_of_memory": True},
    ]
    assert probe_engine.summarize(rows, 10_000) == {
        "sizes": 3,
        "largest_prompt_tokens": 20_400,
        "tokens_per_s_median": 20.0,
        "peak_gb": 78.0,
        "failed_at_prompt_tokens": 30_500,
        "out_of_memory": True,
        "entry_prompt_tokens": 10_300,
        "entry_first_token_s": 3.2,
        "entry_tokens_per_s": 19.0,
        "entry_peak_gb": 71.5,
    }

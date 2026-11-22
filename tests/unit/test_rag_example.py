"""tools/rag_example.py: the organizers' RAG example is reported as it behaves.

The example's main.py imports Docling, an embedder and transformers, so here
a stand-in takes its place (tests/unit/stand_ins.py): the names the adapter
uses, with a _build_context that ranks, packs and writes its context the way
the example's does. The real one is exercised by the slow tests.
"""

import pytest

from tests.unit.stand_ins import BUDGET, Letters, Words, example_main
from tools.eval_retrieval import Retrieved
from tools.rag_example import ExampleChanged, RagExample

CHUNKS = [
    "zebra zebra zebra zebra zebra",  # 5 tokens
    "zebra zebra zebra zebra crossing",  # 5
    "a zoo with one zebra in it",  # 7
    "quite another subject",  # 3
]


def example(main=None, chunks=CHUNKS):
    system = RagExample(main or example_main(), Words(), Letters(), "f" * 64, "some/model")
    system.index(list(chunks))
    return system


def test_the_example_is_reported_with_the_chunks_it_packs():
    system = example()
    retrieved = system.retrieve("zebra")
    assert retrieved.order[:3] == [0, 1, 2]
    assert len(retrieved.order) == len(CHUNKS)
    # The third chunk would pass the budget of 12 tokens, and packing stops
    # there: the fourth would fit and is not tried.
    assert retrieved.context == [(0, CHUNKS[0]), (1, CHUNKS[1])]
    assert retrieved.tokens == 10
    system.verify("zebra", retrieved)


def test_a_chunk_too_long_for_the_budget_is_reported_as_the_slice_the_example_gives():
    long = "zebra " * 30
    system = example(chunks=[long, *CHUNKS[2:]])
    retrieved = system.retrieve("zebra")
    assert retrieved.context == [(0, long[: int(len(long) * (BUDGET / 30))])]
    assert retrieved.tokens == 0
    system.verify("zebra", retrieved)


def test_a_change_in_how_the_example_packs_is_noticed():
    system = example()
    retrieved = system.retrieve("zebra")

    def keeps_trying_after_a_chunk_that_does_not_fit(question, chunks, embeddings, *rest):
        return f"Context:\n\n{chunks[0]}\n\n{chunks[1]}\n\n{chunks[3]}" if len(chunks) == 4 else ""

    system.main._build_context = keeps_trying_after_a_chunk_that_does_not_fit
    with pytest.raises(ExampleChanged, match="no longer mirrors"):
        system.verify("zebra", retrieved)

    # Reporting one chunk too many, or too few, is noticed as well.
    system = example()
    for wrong in ([(0, CHUNKS[0]), (1, CHUNKS[1]), (2, CHUNKS[2])], [(0, CHUNKS[0])]):
        tokens = sum(len(text.split()) for _, text in wrong)
        with pytest.raises(ExampleChanged):
            system.verify("zebra", Retrieved([0, 1, 2, 3], wrong, tokens))


def test_the_example_does_its_own_parsing_and_chunking(tmp_path):
    system = example()
    assert system.parse(tmp_path / "a.pdf") == f"# {tmp_path / 'a.pdf'}"
    system.parse(tmp_path / "b.pdf")
    assert system.main.converters_built == [1]  # one converter for the whole corpus
    assert system.chunk("one\n\ntwo") == ["one", "two"]


def test_the_example_is_described_for_the_ledger_and_for_the_cache():
    system = example()
    assert system.parse_key.startswith("02_rag-ffffffff-docling-")
    assert system.config["main_sha256"] == "f" * 64
    assert system.config["context_tokens"] == BUDGET
    assert system.config["tokenizer"] == "some/model"


def test_without_chunks_the_example_retrieves_nothing():
    system = RagExample(example_main(), Words(), Letters())
    retrieved = system.retrieve("zebra")
    assert (retrieved.order, retrieved.context) == ([], [])
    system.verify("zebra", retrieved)

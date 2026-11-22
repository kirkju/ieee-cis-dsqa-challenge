"""tools/entry_system.py: the entry, measured through its own code.

The embedder and the tokenizers are the stand-ins of the other tests, and
the corpus is two PDFs made on the spot.
"""

import pymupdf
import pytest
from dsqa import config

from tests.unit.stand_ins import BODY, Letters, Words, example_main, paper_pdf
from tools import entry, eval_retrieval
from tools.entry_system import Entry
from tools.rag_example import RagExample
from tools.retrieval_lab import Lab, parse_options


@pytest.fixture
def pdfs(tmp_path):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        page.insert_text((72, 160), "Table 1. Flows of the loop.", fontsize=9)
        for down in (170, 188, 220):
            page.draw_line((72, down), (380, down))
        rows = [("", "Flow", "Pressure"), ("Main loop", "97.5", "3.0"), ("Trap", "", "1.5")]
        for number, row in enumerate(rows):
            for across, text in zip((80, 200, 300), row, strict=True):
                page.insert_text((across, 183 + 15 * number), text, fontsize=9)
        document.save(tmp_path / "a-trap.pdf")
    paper_pdf(tmp_path / "b-loop.pdf", BODY)
    return sorted(tmp_path.glob("*.pdf"))


@pytest.fixture
def small(monkeypatch):
    """The entry's numbers cut down to the size of the test's corpus."""
    monkeypatch.setattr(config, "CHUNK_TOKENS", 16)
    monkeypatch.setattr(config, "TABLE_TOKENS", 24)
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 60)
    monkeypatch.setattr(config, "WHOLE", 0.5)
    return "chunk_tokens=16,table_tokens=24,budget=60,whole=50"


def indexed(system, pdfs):
    chunks = [chunk for pdf in pdfs for chunk in system.chunk(system.parse(pdf))]
    system.index(chunks)
    return chunks


def test_the_entry_is_measured_stage_by_stage_through_its_own_code(pdfs, small):
    system = Entry(Words(), Letters(), "some/model")
    text = system.parse(pdfs[0])
    assert text.startswith("Notes on the nitrogen trap\n\n")
    assert "|  | Flow | Pressure |" in text
    assert system.parse_key == entry.parse_key("tables")
    assert system.config["system"] == "03_dsqa" and system.config["tokenizer"] == "some/model"
    assert (system.config["chunk_tokens"], system.config["whole"]) == (16, 0.5)
    assert len(system.config["code"]) == 8

    chunks = indexed(system, pdfs)
    first = sum(chunk.startswith("Notes on") for chunk in chunks)
    assert first > 1 and system.sizes == [first, len(chunks) - first]
    retrieved = system.retrieve("where is the nitrogen trap")
    assert sorted(retrieved.order) == list(range(len(chunks)))
    assert retrieved.context[0][1].startswith("Notes on the nitrogen trap")
    assert 0 < retrieved.tokens <= 60


def test_the_entry_retrieves_what_the_lab_retrieves_with_the_same_settings(pdfs, small):
    system = Entry(Words(), Letters(), "some/model")
    options = parse_options(f"parser=pymupdf_tables,chunker=sections,title=1,ranker=rrf,{small}")
    lab = Lab(RagExample(example_main(), Words(), Letters(), "f" * 64, "some/model"), options)
    assert indexed(system, pdfs) == indexed(lab, pdfs)
    for question in ("where is the nitrogen trap", "the flow of the main loop", "lithium target"):
        mine, theirs = system.retrieve(question), lab.retrieve(question)
        assert (mine.order, mine.context) == (theirs.order, theirs.context)
        assert mine.tokens == theirs.tokens
        assert len(mine.context) < len(mine.order)  # the budget was the limit


def test_the_index_is_told_the_document_of_every_chunk(pdfs):
    system = Entry(Words(), Letters())
    chunks = [chunk for pdf in pdfs for chunk in system.chunk(system.parse(pdf))]
    with pytest.raises(ValueError, match="as chunk\\(\\) gave them"):
        system.index(chunks[1:])


def test_the_entry_goes_through_the_measuring_tool(pdfs, small, monkeypatch):
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 400)  # room for the whole corpus
    questions = [{"id": 1, "domain": "DONES", "question_type": "factual", "question": "trap?"}]
    quote = "Table 1. Flows of the loop. Flow Pressure Main loop 97.5 3.0 Trap 1.5"
    passage = {"doc_id": "a-trap", "page": 1, "kind": "table", "quote": quote}
    items = [{"id": 1, "status": "traced", "doc_id": "a-trap", "evidence": [passage]}]
    measured = eval_retrieval.measure(
        Entry(Words(), Letters()), questions, items, pdfs, cache_dir=None, say=lambda line: None
    )
    (row,) = measured["rows"]
    # The table is cut in two at this size, each part under the title and the
    # caption; both are placed, and read as one they are the whole table.
    assert (row["parsed"], row["chunked"]) == (1.0, 1.0)
    assert row["one_chunk"] < 1.0
    assert row["lost"] is None


def test_the_entry_takes_no_options():
    with pytest.raises(ValueError, match="takes no options"):
        eval_retrieval.SYSTEMS["03_dsqa"]("some/model", "ranker=bm25")

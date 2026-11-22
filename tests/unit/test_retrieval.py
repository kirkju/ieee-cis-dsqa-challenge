"""dsqa/retrieval.py: from a folder of PDFs to the context of a question.

The embedder of these tests counts letters and its tokenizer has a token per
word (tests/unit/stand_ins.py); the PDFs are made on the spot.
"""

import numpy as np
import pymupdf
import pytest
from dsqa import config, retrieval
from dsqa.retrieval import Index

from tests.unit.stand_ins import BODY, Letters, Words, paper_pdf, tokens

ZOO = [
    "zebra zebra zebra zebra zebra",
    "zebra zebra zebra zebra crossing",
    "a zoo with one zebra in it",
    "quite another subject",
]


def test_the_chunks_of_a_document_open_with_its_title_and_fit_the_embedder(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_TOKENS", 12)
    monkeypatch.setattr(config, "TABLE_TOKENS", 40)
    table = "\n".join(f"| line{number} | {number}.5 |" for number in range(1, 6))
    body = "one two three four five. " * 4
    text = f"The loop of the plant\n\n{body}\n\nTable 1. Flows.\n\n{table}"
    chunks = retrieval.chunks_of(text, Words())
    assert len(chunks) > 3
    assert all(chunk.startswith("The loop of the plant\n\n") for chunk in chunks)
    assert all(tokens(chunk) <= 12 for chunk in chunks[:-1])
    # The table has the room of a table: one chunk, under its caption.
    assert chunks[-1] == f"The loop of the plant\n\nTable 1. Flows.\n\n{table}"
    assert 12 < tokens(chunks[-1]) <= 40


def test_two_rankings_are_fused_and_words_alone_rank_without_an_embedder():
    index = Index(ZOO, [0, 0, 1, 2], Letters())
    assert index.embeddings.shape == (4, 26)
    order = index.rank("zebra crossing")
    assert order[0] == 1  # the most zebra-like by letters among those with both words
    assert sorted(order.tolist()) == [0, 1, 2, 3]

    by_words = Index(ZOO, [0, 0, 1, 2])
    assert by_words.embeddings is None
    assert by_words.rank("which subject is another")[0] == 3
    assert sorted(by_words.rank("zebra").tolist()) == [0, 1, 2, 3]


def test_the_context_has_the_first_document_whole_and_then_the_ranking():
    index = Index(ZOO, [0, 0, 1, 2])
    counted = []

    def count(text):
        counted.append(text)
        return tokens(text)

    # The last document is the best chunk's and takes 3 tokens of 12: it goes
    # in whole, and then the ranking as far as it fits.
    placed, used, order = index.context("which subject is another", count, budget=12, whole=0.5)
    assert [number for number, _ in placed] == [3, 0]
    assert (used, int(order[0])) == (8, 3)
    # The budget and the share of a whole document are the entry's when not given.
    everything, _, _ = index.context("zebra crossing", count)
    assert sorted(number for number, _ in everything) == [0, 1, 2, 3]
    # A chunk is counted once, however many questions are asked.
    assert sorted(counted) == sorted(ZOO)


def test_an_index_of_nothing_gives_no_context():
    empty = Index([], [], Letters())
    assert empty.embeddings is None
    placed, used, order = empty.context("anything", tokens)
    assert (placed, used, order.tolist()) == ([], 0, [])
    with pytest.raises(ValueError, match="3 chunks and 2 documents"):
        Index(ZOO[:3], [0, 0])


@pytest.fixture
def folder(tmp_path):
    """Two papers, a file that is not a PDF and a PDF with nothing written in it."""
    paper_pdf(tmp_path / "b-loop.pdf", BODY)
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "Notes on the nitrogen trap", fontsize=18)
        page.insert_text((72, 120), "The trap is in the dump tank.", fontsize=9)
        document.save(tmp_path / "a-trap.pdf")
    (tmp_path / "c-broken.pdf").write_bytes(b"this is not a PDF at all")
    with pymupdf.open() as document:
        document.new_page()
        document.save(tmp_path / "d-blank.pdf")
    (tmp_path / "notes.txt").write_text("not a PDF, and not looked at", encoding="utf-8")
    return tmp_path


def test_a_folder_is_read_in_the_order_of_its_names_and_a_bad_pdf_is_left_out(folder):
    said = []
    texts = retrieval.read_corpus(folder, said.append)
    assert [text.split("\n\n")[0] for text in texts] == [
        "Notes on the nitrogen trap",
        "A study of the lithium loop of the plant",
    ]
    assert len(said) == 1 and said[0].startswith("[dsqa] not read: c-broken.pdf: ")
    assert retrieval.read_corpus(folder / "nowhere", said.append) == []


def test_reading_stops_when_its_time_is_up_and_what_was_read_stands(folder):
    ticks = iter(range(0, 1000, 10))  # ten seconds go by each time the clock is read
    said = []
    texts = retrieval.read_corpus(folder, said.append, seconds=15, clock=lambda: next(ticks))
    assert [text.split("\n\n")[0] for text in texts] == ["Notes on the nitrogen trap"]
    assert said == ["[dsqa] out of time for reading: 3 of 4 PDFs left out"]


def test_without_an_embedder_the_chunks_are_cut_by_words(folder, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_TOKENS", 30)
    said = []
    by_words = retrieval.build(folder, None, said.append)
    assert by_words.embeddings is None
    assert by_words.chunks == retrieval.build(folder, Letters(), said.append).chunks
    assert all(tokens(chunk) <= 30 for chunk in by_words.chunks)
    placed, _, _ = by_words.context("where is the nitrogen trap", tokens)
    assert placed[0][1].startswith("Notes on the nitrogen trap")


def test_the_embedder_is_looked_for_where_the_entry_ships_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDER", tmp_path / "embed_model")
    with pytest.raises(FileNotFoundError, match="no embedding model at"):
        retrieval.load_embedder()


def test_the_index_of_a_folder_knows_the_document_of_every_chunk(folder):
    said = []
    index = retrieval.build(folder, Letters(), said.append)
    assert said[-1] == f"[dsqa] 2 documents in {len(index.chunks)} chunks"
    first = int(np.sum(index.documents == 0))
    assert first >= 1 and index.documents.tolist() == sorted(index.documents.tolist())
    assert all(chunk.startswith("Notes on the nitrogen trap") for chunk in index.chunks[:first])
    assert all(chunk.startswith("A study of the lithium loop") for chunk in index.chunks[first:])
    placed, _, _ = index.context("where is the nitrogen trap", tokens)
    assert placed[0][0] < first

    # Cut by another tokenizer than the embedder's, when one is given.
    again = retrieval.index_of(retrieval.read_corpus(folder, said.append), Letters(), Words())
    assert again.chunks == index.chunks

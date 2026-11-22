"""tools/retrieval_lab.py: the RAG example with stages swapped.

What is the lab's alone is tried on a few lines of text, and the lab system
on the same stand-ins as the example's adapter (tests/unit/stand_ins.py).
The parts that became the entry's are tried with the entry.
"""

import numpy as np
import pymupdf
import pytest
from dsqa import chunk, parse

from tests.unit.stand_ins import (
    BODY,
    BUDGET,
    Letters,
    Words,
    example_main,
    paper_pdf,
    tokens,
)
from tools import retrieval_lab
from tools.rag_example import RagExample
from tools.retrieval_lab import Lab, fit, pack, parse_options


def test_options_default_to_the_example_and_are_checked():
    assert parse_options("") == retrieval_lab.DEFAULTS
    options = parse_options("ranker=rrf, chunker=fitted,chunk_tokens=128")
    assert (options["ranker"], options["chunker"]) == ("rrf", "fitted")
    assert (options["chunk_tokens"], options["parser"]) == (128, "docling")

    options = parse_options("parser=docling_or_pymupdf,whole=70,beside=1")
    assert (options["parser"], options["whole"], options["beside"]) == ("docling_or_pymupdf", 70, 1)
    options = parse_options("parser=pymupdf_tables,table_tokens=500")
    assert (options["parser"], options["table_tokens"]) == ("pymupdf_tables", 500)
    assert parse_options("")["table_tokens"] == 0

    for wrong, complaint in [
        ("reranker=on", "no such option: 'reranker'"),
        ("ranker=tfidf", "ranker is one of bm25, dense, rrf"),
        ("chunk_tokens=many", "chunk_tokens takes a number"),
        ("whole=150", "whole is a percent of the budget"),
    ]:
        with pytest.raises(ValueError, match=complaint):
            parse_options(wrong)


def test_fitting_packs_paragraphs_while_they_fit():
    text = "one two three\n\nfour five\n\nsix seven eight nine\n\nten"
    assert fit(text, Words(), 5) == ["one two three\n\nfour five", "six seven eight nine\n\nten"]
    assert fit(text, Words(), 100) == [text]
    assert fit("", Words(), 5) == []


def test_fitting_cuts_a_long_paragraph_at_the_end_of_a_sentence():
    paragraph = "One two three four. Five six seven. Eight nine ten eleven twelve. Thirteen."
    chunks = fit(paragraph, Words(), 8)
    assert chunks == [
        "One two three four. Five six seven.",
        "Eight nine ten eleven twelve. Thirteen.",
    ]
    assert all(tokens(chunk) <= 8 for chunk in chunks)


def test_fitting_cuts_at_a_space_when_there_is_no_better_place():
    row = " ".join(f"cell{number}" for number in range(1, 24))
    chunks = fit(row, Words(), 10)
    assert all(tokens(chunk) <= 10 for chunk in chunks)
    # Nothing is lost and nothing is repeated.
    assert " ".join(chunks).split() == row.split()
    assert [tokens(chunk) for chunk in chunks] == [10, 10, 3]


def test_fitting_keeps_every_word_of_a_mixed_text_in_order():
    paragraphs = [
        "A short one.",
        "A long one, with clauses; and more clauses, and a sentence. " * 6,
        "| a table  as one row  of loose cells | " * 5,
        "The end.",
    ]
    text = "\n\n".join(paragraphs)
    for limit in (3, 7, 20, 64):
        chunks = fit(text, Words(), limit)
        assert all(tokens(chunk) <= limit for chunk in chunks)
        assert " ".join(chunks).split() == text.split()


def test_sections_fit_text_without_any_structure_as_the_fitted_chunker_does():
    text = "one two three\n\nfour five\n\nsix seven eight nine\n\nten"
    assert chunk.sectioned(text, Words(), 5) == fit(text, Words(), 5)


CHUNKS = ["aa bb cc", "dd ee ff gg hh", "ii", "jj kk"]


def test_packing_stops_at_the_first_chunk_that_does_not_fit_unless_told_to_fill():
    order = [0, 1, 2, 3]
    assert pack(order, CHUNKS, tokens, 6) == ([(0, CHUNKS[0])], 3)
    assert pack(order, CHUNKS, tokens, 6, fill=True) == (
        [(0, CHUNKS[0]), (2, CHUNKS[2]), (3, CHUNKS[3])],
        6,
    )
    assert pack(order, CHUNKS, tokens, 100) == ([(n, CHUNKS[n]) for n in order], 11)


def test_packing_gives_a_slice_of_a_first_chunk_that_is_too_long():
    # The example's way: the share of its characters that the budget is of its tokens.
    assert pack([1, 0], CHUNKS, tokens, 2) == ([(1, CHUNKS[1][: int(14 * 2 / 5)])], 0)
    # Filling passes over it instead.
    assert pack([1, 2], CHUNKS, tokens, 2, fill=True) == ([(2, CHUNKS[2])], 1)


# Three documents: of 6, 3 and 4 tokens.
SHELF = ["a1 a1", "a2 a2 a2", "a3", "b1 b1", "b2", "c1 c1 c1 c1"]
OF = [0, 0, 0, 1, 1, 2]


def placed(context):
    return [number for number, _ in context]


def test_the_document_of_the_best_chunk_goes_in_whole_when_it_fits_its_share():
    order = [4, 0, 5, 1, 2, 3]
    # The second document is 3 tokens of a budget of 8: it goes first, in its
    # own order, and the ranking follows until a chunk does not fit.
    context, used = pack(order, SHELF, tokens, 8, documents=OF, whole=0.5)
    assert (placed(context), used) == ([3, 4, 0], 5)
    assert context[0] == (3, SHELF[3])
    context, used = pack(order, SHELF, tokens, 8, fill=True, documents=OF, whole=0.5)
    assert (placed(context), used) == ([3, 4, 0, 1], 8)
    # Nothing says which document a chunk is of: the ranking, as before.
    assert pack(order, SHELF, tokens, 8, whole=0.5) == pack(order, SHELF, tokens, 8)


def test_a_document_too_long_for_its_share_does_not_go_in_whole():
    order = [1, 4, 5, 0, 2, 3]
    plain = pack(order, SHELF, tokens, 10)
    assert placed(plain[0]) == [1, 4, 5, 0]
    # The first document is 6 tokens, more than half of 10.
    assert pack(order, SHELF, tokens, 10, documents=OF, whole=0.5) == plain
    assert placed(pack(order, SHELF, tokens, 10, documents=OF, whole=0.6)[0]) == [0, 1, 2, 4]


def test_a_chunk_of_the_first_document_brings_the_chunks_beside_it():
    order = [1, 4, 5, 0, 2, 3]
    # Chunk 1 brings 0 and 2. Chunk 4 is of another document and brings none.
    context, used = pack(order, SHELF, tokens, 10, documents=OF, beside=1)
    assert (placed(context), used) == ([1, 0, 2, 4], 7)
    # Not across the end of the document, and not past the budget.
    assert placed(pack([2, 5], SHELF, tokens, 10, documents=OF, beside=1)[0]) == [2, 1, 5]
    assert placed(pack([2, 5], SHELF, tokens, 3, documents=OF, beside=2)[0]) == [2, 0]
    # A document that went in whole has nothing left to bring.
    context, _ = pack(order, SHELF, tokens, 10, documents=OF, whole=1.0, beside=1)
    assert placed(context) == [0, 1, 2, 4]


def test_a_parser_read_a_document_when_it_has_half_of_what_another_has():
    other = "the loop carries the lithium " * 10
    assert retrieval_lab.could_read(other[: len(other) // 2 + 20], other)
    assert not retrieval_lab.could_read("B9 B9 BA", other)
    assert not retrieval_lab.could_read(None, other)
    # With nothing to go by, what there is stands.
    assert retrieval_lab.could_read("some text", "")


ZOO = [
    "zebra zebra zebra zebra zebra",
    "zebra zebra zebra zebra crossing",
    "a zoo with one zebra in it",
    "quite another subject",
]


def lab(spec="", tmp_path=None):
    example = RagExample(example_main(), Words(), Letters(), "f" * 64, "some/model")
    options = parse_options(spec)
    if not spec:
        options["budget"] = BUDGET  # the budget of the stand-in for the example
    return Lab(example, options, tmp_path)


def test_with_no_options_the_lab_is_the_example(monkeypatch):
    monkeypatch.setitem(retrieval_lab.DEFAULTS, "budget", BUDGET)
    system = lab()
    assert system.is_the_example
    assert system.chunk("one\n\ntwo") == ["one", "two"]
    assert system.parse_key == system.example.parse_key

    system.index(list(ZOO))
    retrieved = system.retrieve("zebra")
    assert retrieved.context == [(0, ZOO[0]), (1, ZOO[1])]
    assert retrieved.tokens == 10
    system.verify("zebra", retrieved)  # against the example's own context

    # A lab system that packed otherwise would not pass for the example.
    retrieved.context = retrieved.context[:1]
    with pytest.raises(Exception, match="no longer mirrors"):
        system.verify("zebra", retrieved)


def test_a_changed_option_is_recorded_and_not_checked_against_the_example():
    system = lab("ranker=bm25,packer=fill,budget=12")
    assert not system.is_the_example
    assert system.config["ranker"] == "bm25" and system.config["system"] == "lab"
    system.index(list(ZOO))
    retrieved = system.retrieve("which subject is another")
    assert retrieved.order[0] == 3
    system.verify("which subject is another", retrieved)  # nothing to compare with
    # No embedder is needed to match words.
    assert system.example.embedder.calls == []


def test_the_fused_ranking_uses_both(tmp_path):
    system = lab("ranker=rrf")
    system.index(list(ZOO))
    assert system.embeddings is not None and system.bm25 is not None
    order = system.rank("zebra crossing")
    assert order[0] == 1  # the most zebra-like by letters among those with both words
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_fitted_chunks_are_cut_to_the_tokens_asked_for():
    system = lab("chunker=fitted,chunk_tokens=4")
    assert system.chunk("one two three four five six. Seven eight.") == [
        "one two three four",
        "five six. Seven eight.",
    ]


def test_another_parser_keeps_its_text_apart():
    assert lab("parser=pymupdf").parse_key.startswith("pymupdf-")
    parsers = ["docling", "pymupdf", "pymupdf_titled", "pymupdf_tables", "docling_or_pymupdf"]
    assert len({lab(f"parser={parser}").parse_key for parser in parsers}) == len(parsers)
    assert lab("parser=docling").parse_key == lab().parse_key
    # The text is kept under the name of the code that read it.
    assert lab("parser=pymupdf_tables").parse_key.endswith("-tables")


def test_docling_s_text_gives_way_to_pymupdf_s_where_it_read_too_little(tmp_path, monkeypatch):
    long_one = paper_pdf(tmp_path / "long.pdf", BODY)
    short_one = tmp_path / "short.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "A short note.", fontsize=9)
        document.save(short_one)
    system = lab("parser=docling_or_pymupdf")
    # The stand-in for Docling writes one line of any document: its path. That
    # is most of what the short document says and little of the long one.
    assert system.parse(short_one) == f"# {short_one}"
    rescued = system.parse(long_one)
    assert rescued == parse.read(long_one, tables=False)
    assert rescued.startswith("A study of the lithium loop of the plant\n\n")

    # A document Docling fails on is one it could not read.
    def fail(pdf):
        raise RuntimeError("no text layer")

    monkeypatch.setattr(system.example, "parse", fail)
    assert system.parse(short_one) == parse.read(short_one, tables=False)
    assert lab("parser=pymupdf_titled").parse(long_one) == rescued
    assert lab("parser=pymupdf").parse(long_one) == parse.read(long_one, title=False, tables=False)


def test_the_lab_knows_the_document_of_each_chunk_and_can_put_one_in_whole():
    system = lab("chunker=fitted,chunk_tokens=3,ranker=bm25,whole=100,budget=12")
    first = system.chunk("zebra crossing here\n\nthe zoo keeper")
    second = system.chunk("quite another subject")
    system.index([*first, *second])
    assert system.documents.tolist() == [0, 0, 1]
    # The best chunk is the second of its document; all of the document goes
    # in, from its start, and then the ranking.
    retrieved = system.retrieve("keeper")
    assert retrieved.order[0] == 1
    assert [number for number, _ in retrieved.context] == [0, 1, 2]
    assert retrieved.tokens == 9

    # Chunks that did not come from chunk() are of no known document.
    with pytest.raises(ValueError, match="by document"):
        lab("whole=50").index(list(ZOO))
    plain = lab("ranker=bm25,budget=12")
    plain.index(list(ZOO))
    assert plain.documents is None


def test_embeddings_are_kept_for_the_same_chunks(tmp_path):
    first = lab("ranker=dense,budget=12", tmp_path)
    first.index(list(ZOO))
    assert first.example.embedder.calls == [4]

    again = lab("ranker=dense,budget=12", tmp_path)
    again.index(list(ZOO))
    assert again.example.embedder.calls == []
    assert np.array_equal(again.embeddings, first.embeddings)

    other = lab("ranker=dense,budget=12", tmp_path)
    other.index([*ZOO, "one more chunk"])
    assert other.example.embedder.calls == [5]

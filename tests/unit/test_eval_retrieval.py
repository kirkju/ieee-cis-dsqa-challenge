"""tools/eval_retrieval.py: a passage is followed from the PDF to the context.

The systems measured here are a few lines each: what they parse, how they
chunk and in which order they rank is written in the test, so every way of
losing a passage can be staged without a model.
"""

import json

import pymupdf
import pytest

from tools import corpus, eval_retrieval, ledger
from tools.eval_retrieval import (
    PART,
    WHOLE,
    Passage,
    Retrieved,
    TablePassage,
    continuation,
    spell,
    squash,
)

FLOW = "The main loop carries liquid lithium at a nominal flow rate of 97.5 litres per second."
TRAP = "The nitrogen hot trap is operated at 500 to 600 degrees inside the dump tank."
PAD = "Nothing in this paragraph has to do with what any of the questions ask about."
FILLER = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor. "

QUESTIONS = [
    {"id": 1, "domain": "DONES", "question_type": "factual", "question": "flow?"},
    {"id": 2, "domain": "AI Papers", "question_type": "causal", "question": "trap?"},
    {"id": 3, "domain": "AI Papers", "question_type": "factual", "question": "unknown?"},
]


def passage(doc_id, quote, page=1, kind="text"):
    return {"doc_id": doc_id, "page": page, "kind": kind, "quote": quote}


ITEMS = [
    {"id": 1, "status": "traced", "doc_id": "loop", "evidence": [passage("loop", FLOW)]},
    {"id": 2, "status": "traced", "doc_id": "tank", "evidence": [passage("tank", TRAP)]},
    {"id": 3, "status": "untraced", "evidence": []},
]


def quiet(line):
    pass


class Scripted:
    """A system whose every stage is written out: what the parser makes of each
    document, a chunk per paragraph, and the ranking given for each question
    (of the chunks there are). The first `room` chunks of a ranking are its
    context."""

    name = "scripted"
    parse_key = "scripted"
    config = {"system": "scripted"}

    def __init__(self, texts, orders, room=1):
        self.texts, self.orders, self.room = texts, orders, room
        self.parsed = []

    def parse(self, pdf):
        self.parsed.append(pdf.stem)
        if isinstance(self.texts[pdf.stem], Exception):
            raise self.texts[pdf.stem]
        return self.texts[pdf.stem]

    def chunk(self, text):
        return text.split("\n\n")

    def index(self, chunks):
        self.chunks = chunks

    def retrieve(self, question):
        order = [number for number in self.orders[question] if number < len(self.chunks)]
        placed = [(number, self.chunks[number]) for number in order[: self.room]]
        return Retrieved(order, placed, tokens=sum(len(text.split()) for _, text in placed))


# Two documents of two paragraphs each: chunks 0 and 1 are the loop paper's,
# chunks 2 and 3 the tank paper's.
TEXTS = {"loop": f"{FLOW}\n\n{PAD}", "tank": f"{PAD}\n\n{TRAP}"}
ORDERS = {"flow?": [0, 1, 2, 3], "trap?": [0, 1, 2, 3], "unknown?": [3, 2, 1, 0]}


@pytest.fixture
def pdfs(tmp_path):
    """The two documents as files. What a scripted system reads from them is
    in its script, but the tool hashes them and counts their pages."""
    paths = []
    for doc_id in sorted(TEXTS):
        path = tmp_path / "corpus" / f"{doc_id}.pdf"
        path.parent.mkdir(exist_ok=True)
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), f"The {doc_id} paper.", fontsize=9)
            document.new_page()
            document.save(path)
        paths.append(path)
    return paths


def measure(system, pdfs, questions=QUESTIONS, items=ITEMS, **options):
    options.setdefault("cache_dir", None)
    measured = eval_retrieval.measure(system, questions, items, pdfs, say=quiet, **options)
    return {row["id"]: row for row in measured["rows"]}, measured


def test_squashing_keeps_what_survives_a_change_of_parser():
    assert squash("The  Heat\nExchan-\nger") == "theheatexchanger"
    assert squash("ef\N{LATIN SMALL LIGATURE FI}cient \N{SUPERSCRIPT SEVEN}Be") == "efficient7be"
    # A table as Markdown and the same table as loose text.
    assert squash("| Flow rate | 97.5 l s\N{MINUS SIGN}1 |") == squash("Flow rate 97.5 l s-1")
    assert squash("10 \N{MICRO SIGN}Sv/h") == squash("10 \N{GREEK SMALL LETTER MU}Sv/h")


def coverage(quote, text):
    return float(Passage("doc", 1, "text", quote).found(squash(text)).mean())


def test_a_quote_is_found_whatever_the_layout_around_it():
    page = "## 2.1 The loop\n\nThe main loop carries liquid lith-\nium at a nominal  flow rate"
    page += " of 97.5 litres\nper second. It is driven by an electromagnetic pump."
    assert coverage(FLOW, page) == 1.0


def test_sharing_words_with_a_quote_is_not_holding_it():
    assert coverage(FLOW, "The nominal temperature of the lithium in the loop is 300 degrees.") == 0
    assert coverage(FLOW, FILLER * 20) == 0


def test_coverage_is_the_fraction_of_the_quote_that_is_there():
    assert coverage(FLOW, "The main loop carries liquid lithium at a nominal") == pytest.approx(
        len(squash("The main loop carries liquid lithium at a nominal")) / len(squash(FLOW))
    )


def test_one_wrong_character_costs_only_itself():
    letters = len(squash(FLOW))
    assert coverage(FLOW, FLOW.replace("97.5", "97.6")) == pytest.approx((letters - 1) / letters)


def test_a_quote_shorter_than_a_run_is_matched_as_a_whole():
    assert coverage("97.5 l/s", "a flow of 97.5 l/s in the loop") == 1.0
    assert coverage("97.5 l/s", "a flow of 97.6 l/s in the loop") == 0


def test_a_quote_without_letters_or_digits_is_refused():
    with pytest.raises(ValueError, match="cannot be looked for"):
        Passage("doc", 1, "text", "\N{EM DASH} \N{HORIZONTAL ELLIPSIS} \N{EM DASH}")


def test_the_best_stretch_of_a_text_is_what_one_place_in_it_holds():
    quote = Passage("doc", 1, "text", FLOW)
    assert quote.best_stretch(squash(FILLER * 30 + FLOW + FILLER * 30)) == 1.0
    assert quote.best_stretch(squash(FILLER * 30)) == 0
    assert quote.best_stretch("") == 0

    # The two halves of the sentence far apart: every character is somewhere
    # in the text, and no single place has the passage.
    apart = FLOW[:43] + FILLER * 30 + FLOW[43:]
    assert quote.found(squash(apart)).mean() == 1.0
    assert PART - 0.1 < quote.best_stretch(squash(apart)) < WHOLE


# A table as PyMuPDF gives it, cell after cell along the rows; the same table
# set in rows, and read down its columns.
CAPTION = "Table 2. Estimated current in the detectors (nA)."
TABLE = f"{CAPTION} Rig 11 Rig 45 Neutrons 5.37 42.75 Gammas 21.90 523 Total 29.6 659"
IN_ROWS = "|  | Rig 11 | Rig 45 |\n| Neutrons | 5.37 | 42.75 |\n| Gammas | 21.90 | 523 |"
IN_ROWS += "\n| Total | 29.6 | 659 |"
IN_COLUMNS = "Neutrons Gammas Total Rig 11 5.37 21.90 29.6 Rig 45 42.75 523 659"


def test_spelling_a_text_gives_its_words_without_what_stands_between_them():
    row = "| Flow  rate | 97.5 l s\N{MINUS SIGN}1 |\n|---|---|"
    assert spell(row) == ("flow", "rate", "975", "l", "s1")
    assert spell(f"{CAPTION}\n\n{IN_ROWS}") == spell(TABLE)
    assert spell(" | \N{EM DASH} | ") == ()


def test_a_passage_of_a_table_is_its_words_in_whatever_order():
    table = TablePassage("doc", 1, "table", TABLE)
    assert table.size == len(TABLE.split())
    assert table.best_stretch(spell(f"{FILLER * 30}{CAPTION}\n\n{IN_ROWS}\n\n{FILLER * 30}")) == 1.0
    assert table.best_stretch(spell(f"{CAPTION} {IN_COLUMNS}")) == 1.0
    assert table.best_stretch(spell(FILLER * 30)) == 0
    assert table.best_stretch(()) == 0
    # Read as running text, the table set down its columns is not the quote.
    assert Passage("doc", 1, "text", TABLE).best_stretch(squash(f"{CAPTION} {IN_COLUMNS}")) < WHOLE


def test_a_table_with_the_same_headings_and_other_numbers_is_another_table():
    table = TablePassage("doc", 1, "table", TABLE)
    other = f"{CAPTION} Rig 11 Rig 45 Neutrons 6.11 40.02 Gammas 20.75 498 Total 26.9 538"
    assert PART <= table.best_stretch(spell(other)) < WHOLE
    # A word counts as often as the quote has it, however often the text does.
    assert table.best_stretch(spell("Rig " * 40)) == pytest.approx(2 / table.size)


def test_the_words_of_a_table_far_apart_do_not_add_up_to_it():
    table = TablePassage("doc", 1, "table", TABLE)
    apart = (FILLER * 10).join(f"{CAPTION}\n\n{IN_ROWS}".split("\n"))
    assert table.best_stretch(spell(apart)) < WHOLE
    assert table.shares(spell(apart)) and not table.shares(spell(FILLER * 10))
    with pytest.raises(ValueError, match="cannot be looked for"):
        TablePassage("doc", 1, "table", "| \N{EM DASH} |")


def test_what_a_chunk_opens_with_as_the_one_before_it_is_not_what_it_adds():
    opening = "The loop paper\n\n## 2 The main loop"
    first, second = f"{opening}\n\n{CAPTION}\n\n| a | 1 |", f"{opening}\n\n{CAPTION}\n\n| b | 2 |"
    assert continuation(first, second) == "| b | 2 |"
    assert continuation(f"{opening}\n\n{PAD}", second) == f"{CAPTION}\n\n| b | 2 |"
    assert continuation("The loop paper\n\n## 1 Introduction\n\nSome text.", second) == (
        f"## 2 The main loop\n\n{CAPTION}\n\n| b | 2 |"
    )
    assert continuation(PAD, second) == second
    assert continuation(second, second) == ""
    # A later part of a table under the start of its caption and its row of
    # headings: both are lines of the part before it.
    whole = f"{opening}\n\n{CAPTION} With a second sentence.\n\n| Rig | nA |\n| 11 | 5.37 |"
    part = f"{opening}\n\n{CAPTION}\n\n| Rig | nA |\n| 45 | 42.75 |"
    assert continuation(whole, part) == "| 45 | 42.75 |"
    # What a chunk ends with is not what the next one opens with.
    assert continuation(f"{PAD}\n\n{FLOW}", f"{FLOW[40:]}\n\n{TRAP}") == f"{FLOW[40:]}\n\n{TRAP}"


def test_a_passage_ranked_first_reaches_the_context(pdfs):
    rows, _ = measure(Scripted(TEXTS, ORDERS, room=1), pdfs)
    first = rows[1]
    assert (first["parsed"], first["one_chunk"], first["context"]) == (1.0, 1.0, 1.0)
    assert first["rank"] == 1
    assert first["lost"] is None
    assert first["document_in_context"] is True


def test_a_passage_ranked_below_the_cut_is_lost_in_ranking(pdfs):
    rows, _ = measure(Scripted(TEXTS, ORDERS, room=3), pdfs)
    second = rows[2]
    # The parser and the chunker kept it; it is the fourth chunk of the ranking.
    assert (second["parsed"], second["chunked"], second["context"]) == (1.0, 1.0, 0.0)
    assert second["rank"] == 4
    assert second["lost"] == "ranking"
    # Another chunk of the right document did get in: finding the document is
    # not finding the passage.
    assert second["document_rank"] == 3
    assert second["document_in_context"] is True


def test_a_passage_the_parser_dropped_is_lost_in_parsing(pdfs):
    rows, _ = measure(Scripted({**TEXTS, "tank": PAD}, ORDERS, room=4), pdfs)
    assert rows[2]["parsed"] == 0
    assert rows[2]["lost"] == "parsing"
    assert rows[2]["rank"] is None


def test_a_passage_the_chunker_dropped_is_lost_in_chunking(pdfs):
    class DropsTheLastParagraph(Scripted):
        def chunk(self, text):
            return text.split("\n\n")[:1]

    rows, _ = measure(DropsTheLastParagraph(TEXTS, ORDERS, room=2), pdfs)
    assert rows[2]["parsed"] == 1.0
    assert rows[2]["chunked"] == 0
    assert rows[2]["lost"] == "chunking"


def test_a_passage_split_by_the_chunker_needs_all_its_pieces(pdfs):
    # Chunks 0 and 1 are the two halves of the sentence.
    class SplitsTheSentence(Scripted):
        def chunk(self, text):
            return [FLOW[:43], FLOW[43:], PAD] if text.startswith(FLOW) else text.split("\n\n")

    orders = {**ORDERS, "flow?": [0, 2, 1, 3, 4]}

    # Both halves are among the chunks, the second below the cut: the passage
    # was parsed and chunked, and is in the context only in part.
    rows, _ = measure(SplitsTheSentence(TEXTS, orders, room=1), pdfs)
    split = rows[1]
    assert (split["parsed"], split["chunked"]) == (1.0, 1.0)
    assert PART <= split["one_chunk"] < WHOLE
    assert PART <= split["context"] < WHOLE
    assert split["rank"] == 3  # how far down one has to read to have both halves
    assert split["lost"] == "ranking"

    rows, _ = measure(SplitsTheSentence(TEXTS, orders, room=3), pdfs)
    assert rows[1]["context"] == 1.0
    assert rows[1]["lost"] is None


def test_chunks_that_repeat_their_opening_are_read_as_one_text_with_it_once(pdfs):
    # The two halves of the sentence, each under the title of the paper, the
    # heading of its section and a note longer than the sentence itself.
    opening = f"The loop paper\n\n## 2 The main loop\n\n{PAD} {PAD}"

    class RepeatsItsOpening(Scripted):
        def chunk(self, text):
            if not text.startswith(FLOW):
                return text.split("\n\n")
            return [f"{opening}\n\n{FLOW[:43]}", f"{opening}\n\n{FLOW[43:]}", PAD]

    orders = {**ORDERS, "flow?": [0, 2, 1, 3, 4]}
    rows, _ = measure(RepeatsItsOpening(TEXTS, orders, room=3), pdfs)
    assert (rows[1]["chunked"], rows[1]["context"]) == (1.0, 1.0)
    assert rows[1]["rank"] == 3
    assert rows[1]["lost"] is None

    # Read with the opening before each half, no stretch twice as long as the
    # sentence has both.
    sentence = Passage("loop", 1, "text", FLOW)
    halves = [squash(f"{opening}\n\n{FLOW[:43]}"), squash(f"{opening}\n\n{FLOW[43:]}")]
    assert sentence.best_stretch("".join(halves)) < WHOLE

    rows, _ = measure(RepeatsItsOpening(TEXTS, orders, room=1), pdfs)
    assert PART <= rows[1]["context"] < WHOLE
    assert rows[1]["lost"] == "ranking"


def test_a_table_is_followed_by_its_words_and_a_text_by_its_letters(pdfs):
    # The parser gives the table down its columns; the quote was cut along its rows.
    texts = {**TEXTS, "loop": f"{CAPTION} {IN_COLUMNS}\n\n{PAD}"}
    as_table = [dict(ITEMS[0], evidence=[passage("loop", TABLE, kind="table")])]
    rows, _ = measure(Scripted(texts, ORDERS, room=1), pdfs, QUESTIONS[:1], as_table)
    assert (rows[1]["parsed"], rows[1]["one_chunk"], rows[1]["context"]) == (1.0, 1.0, 1.0)
    assert rows[1]["rank"] == 1 and rows[1]["lost"] is None

    as_text = [dict(ITEMS[0], evidence=[passage("loop", TABLE, kind="text")])]
    rows, _ = measure(Scripted(texts, ORDERS, room=1), pdfs, QUESTIONS[:1], as_text)
    assert rows[1]["parsed"] < WHOLE
    assert rows[1]["lost"] == "parsing"

    # Set in rows under its caption, the table is two chunks of the scripted
    # system, and both have to be in the context.
    texts = {**TEXTS, "loop": f"{CAPTION}\n\n{IN_ROWS}\n\n{PAD}"}
    orders = {**ORDERS, "flow?": [1, 2, 0, 3, 4]}
    rows, _ = measure(Scripted(texts, orders, room=2), pdfs, QUESTIONS[:1], as_table)
    assert rows[1]["chunked"] == 1.0
    assert PART <= rows[1]["context"] < WHOLE
    assert rows[1]["rank"] == 3 and rows[1]["lost"] == "ranking"
    rows, _ = measure(Scripted(texts, orders, room=3), pdfs, QUESTIONS[:1], as_table)
    assert rows[1]["lost"] is None


def test_phrases_of_a_quote_in_separate_places_do_not_add_up_to_the_passage(pdfs):
    # The loop paper has the sentence once and, further on, two others that
    # each repeat a part of it. Those two are retrieved; the sentence is not.
    start = "The main loop carries liquid lithium at a nominal pressure of two bar."
    end = "The secondary circuit has a flow rate of 97.5 litres per second as well."
    texts = {**TEXTS, "loop": "\n\n".join([FLOW, PAD, start, f"{PAD} Still nothing.", end])}
    orders = {**ORDERS, "flow?": [2, 4, 1, 3, 0, 5, 6]}

    for room in (2, 4):  # with the chunk between the two in the context as well
        rows, _ = measure(Scripted(texts, orders, room=room), pdfs)
        scattered = rows[1]
        assert scattered["one_chunk"] == 1.0
        assert PART <= scattered["context"] < WHOLE  # no more than the longer of the two parts
        assert scattered["rank"] == 5
        assert scattered["lost"] == "ranking"


def test_what_counts_is_the_text_a_system_really_places_in_the_context(pdfs):
    class CutsItsChunks(Scripted):
        def retrieve(self, question):
            retrieved = super().retrieve(question)
            retrieved.context = [(number, text[:30]) for number, text in retrieved.context]
            return retrieved

    rows, _ = measure(CutsItsChunks(TEXTS, ORDERS, room=1), pdfs)
    assert rows[1]["one_chunk"] == 1.0
    assert 0 < rows[1]["context"] < PART
    assert rows[1]["lost"] == "ranking"


def test_a_passage_is_looked_for_only_in_its_own_document(pdfs):
    # The loop paper states the tank paper's sentence too, and that copy is
    # what gets into the context. The annotation names the tank paper.
    texts = {"loop": f"{FLOW}\n\n{TRAP}", "tank": f"{PAD}\n\n{TRAP}"}
    rows, _ = measure(Scripted(texts, ORDERS, room=2), pdfs)
    assert rows[2]["context"] == 0
    assert rows[2]["rank"] == 4
    assert rows[2]["document_in_context"] is False


def test_a_question_is_covered_when_any_of_its_passages_is(pdfs):
    # The same fact in two documents, as with a sentence carried from a
    # preprint to the article that reuses it.
    texts = {"loop": f"{FLOW}\n\n{TRAP}", "tank": f"{PAD}\n\n{TRAP}"}
    items = [dict(ITEMS[1], evidence=[passage("tank", TRAP), passage("loop", TRAP)])]
    rows, _ = measure(Scripted(texts, ORDERS, room=2), pdfs, QUESTIONS[1:2], items)
    assert [each["lost"] for each in rows[2]["passages"]] == ["ranking", None]
    assert rows[2]["lost"] is None
    assert rows[2]["rank"] == 2


def test_passages_of_a_group_answer_only_together(pdfs):
    # The table gives the number and a sentence elsewhere gives the rule: the
    # question is answered with both, and with neither alone.
    grouped = [dict(passage("loop", FLOW), group="a"), dict(passage("tank", TRAP), group="a")]
    items = [dict(ITEMS[0], evidence=grouped)]

    rows, _ = measure(Scripted(TEXTS, ORDERS, room=1), pdfs, QUESTIONS[:1], items)
    alone = rows[1]
    assert [each["lost"] for each in alone["passages"]] == [None, "ranking"]
    assert alone["lost"] == "ranking"
    assert alone["context"] == 0  # as good as the worse of the two
    assert alone["rank"] == 4  # how far down one has to read to have both
    assert [each["group"] for each in alone["passages"]] == ["a", "a"]

    rows, measured = measure(Scripted(TEXTS, ORDERS, room=4), pdfs, QUESTIONS[:1], items)
    assert rows[1]["lost"] is None and rows[1]["context"] == 1.0
    assert eval_retrieval.summarize(measured["rows"])["in_context"] == 1

    # A group is lost at the first stage that loses one of its passages.
    rows, _ = measure(Scripted({**TEXTS, "tank": PAD}, ORDERS, room=4), pdfs, QUESTIONS[:1], items)
    assert rows[1]["lost"] == "parsing"
    assert rows[1]["rank"] is None


def test_a_passage_outside_a_group_still_answers_by_itself(pdfs):
    # The two passages of the loop paper are needed together; the tank paper
    # states the answer in one sentence.
    texts = {"loop": f"{FLOW}\n\n{PAD}\n\n{TRAP}", "tank": f"{PAD}\n\n{TRAP}"}
    evidence_of = [
        dict(passage("loop", FLOW), group="a"),
        dict(passage("loop", TRAP), group="a"),
        passage("tank", TRAP),
    ]
    items = [dict(ITEMS[0], evidence=evidence_of)]
    orders = {"flow?": [4, 0, 1, 2, 3]}  # the tank paper's sentence first

    rows, _ = measure(Scripted(texts, orders, room=1), pdfs, QUESTIONS[:1], items)
    assert rows[1]["lost"] is None and rows[1]["rank"] == 1
    assert [each["lost"] for each in rows[1]["passages"]] == ["ranking", "ranking", None]
    assert "group" not in rows[1]["passages"][2]


def test_a_question_without_passages_is_reported_and_not_scored(pdfs):
    rows, measured = measure(Scripted(TEXTS, ORDERS, room=1), pdfs)
    third = rows[3]
    assert third["passages"] == []
    assert third["lost"] is None and third["context"] is None
    assert third["top_documents"] == ["tank", "loop"]
    metrics = eval_retrieval.summarize(measured["rows"])
    assert (metrics["questions"], metrics["with_evidence"], metrics["passages"]) == (3, 2, 2)


def test_the_stages_are_counted_nested_and_the_domains_weigh_the_same(pdfs):
    # Question 1 (DONES) arrives; question 2 (AI Papers) is lost in ranking.
    _, measured = measure(Scripted(TEXTS, ORDERS, room=1), pdfs)
    metrics = eval_retrieval.summarize(measured["rows"])
    assert (metrics["parsed"], metrics["chunked"], metrics["in_context"]) == (2, 2, 1)
    assert (metrics["top_1"], metrics["top_5"]) == (1, 2)
    assert (metrics["dones_in_context"], metrics["dones_with_evidence"]) == (1, 1)
    assert (metrics["ai_papers_in_context"], metrics["ai_papers_with_evidence"]) == (0, 1)
    assert metrics["balanced"] == 0.5
    assert metrics["passages_in_context"] == 1
    assert metrics["context_chunks"] == 1

    # With the parser dropping question 2's passage nothing later counts it.
    _, measured = measure(Scripted({**TEXTS, "tank": PAD}, ORDERS, room=1), pdfs)
    metrics = eval_retrieval.summarize(measured["rows"])
    assert (metrics["parsed"], metrics["chunked"], metrics["in_context"]) == (1, 1, 1)


def test_a_document_the_parser_fails_on_does_not_stop_the_run(pdfs):
    system = Scripted({**TEXTS, "tank": RuntimeError("not a PDF after all")}, ORDERS, room=2)
    rows, measured = measure(system, pdfs)
    failed = [document for document in measured["documents"] if document.text is None]
    assert [document.doc_id for document in failed] == ["tank"]
    assert "not a PDF after all" in failed[0].error
    assert measured["chunks"] == 2
    assert rows[1]["lost"] is None
    assert rows[2]["lost"] == "parsing"


def test_parsed_text_is_kept_and_read_back_by_the_hash_of_the_file(pdfs, tmp_path):
    cache = tmp_path / "cache"
    system = Scripted(TEXTS, ORDERS)
    _, first = measure(system, pdfs, cache_dir=cache)
    assert system.parsed == ["loop", "tank"]
    assert first["timings"]["parsed_now"] == 2
    assert [document.pages for document in first["documents"]] == [2, 2]

    again = Scripted(TEXTS, ORDERS)
    rows, second = measure(again, pdfs, cache_dir=cache)
    assert again.parsed == []
    assert second["timings"]["parsed_now"] == 0
    # The time the parsing took is kept with the text.
    assert second["timings"]["parse_s"] == first["timings"]["parse_s"]
    assert rows[1]["lost"] is None

    # Another parser does not read this one's text.
    other = Scripted(TEXTS, ORDERS)
    other.parse_key = "another-parser"
    measure(other, pdfs, cache_dir=cache)
    assert other.parsed == ["loop", "tank"]

    # A file that changed is parsed again, and so is everything on request.
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "A new version of the paper.", fontsize=9)
        document.save(pdfs[0])
    changed = Scripted(TEXTS, ORDERS)
    measure(changed, pdfs, cache_dir=cache)
    assert changed.parsed == ["loop"]
    forced = Scripted(TEXTS, ORDERS)
    measure(forced, pdfs, cache_dir=cache, reparse=True)
    assert forced.parsed == ["loop", "tank"]


def test_a_failed_parse_is_not_kept(pdfs, tmp_path):
    cache = tmp_path / "cache"
    measure(Scripted({**TEXTS, "tank": RuntimeError("boom")}, ORDERS), pdfs, cache_dir=cache)
    retry = Scripted(TEXTS, ORDERS)
    measure(retry, pdfs, cache_dir=cache)
    assert retry.parsed == ["tank"]


def manifest_row(doc_id, sha256=""):
    values = dict.fromkeys(corpus.FIELDS, "")
    values.update(doc_id=doc_id, domain="DONES", role="distractor", title="A paper")
    values.update(access="manual", sha256=sha256)
    return values


def test_the_files_read_are_compared_with_the_manifest(pdfs):
    _, measured = measure(Scripted(TEXTS, ORDERS), pdfs)
    documents = measured["documents"]
    hashes = {document.doc_id: document.sha256 for document in documents}
    same = [manifest_row(doc_id, sha256) for doc_id, sha256 in hashes.items()]
    assert not any(eval_retrieval.manifest_differences(documents, same).values())

    other = [manifest_row("loop", "0" * 64), manifest_row("pump", "1" * 64)]
    assert eval_retrieval.manifest_differences(documents, other) == {
        "not_in_the_manifest": ["tank"],
        "not_the_file_described": ["loop"],
        "not_in_the_folder": ["pump"],
    }
    # The fingerprint names the set of files, in whatever order they come.
    assert eval_retrieval.fingerprint(documents) == eval_retrieval.fingerprint(documents[::-1])
    assert len(eval_retrieval.fingerprint(documents)) == 12


def test_a_run_writes_one_row_per_question_and_a_summary(pdfs, tmp_path):
    summary = eval_retrieval.run(
        Scripted(TEXTS, ORDERS, room=1),
        QUESTIONS,
        ITEMS,
        pdfs,
        out_dir=tmp_path / "runs",
        cache_dir=None,
        say=quiet,
    )
    run_dir = tmp_path / "runs" / summary["run_id"]
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == [1, 2, 3]
    assert json.loads((run_dir / "summary.json").read_text()) == summary
    assert summary["metrics"]["in_context"] == 1
    assert summary["by_domain"] == {
        "AI Papers": {"with_evidence": 1, "in_context": 0},
        "DONES": {"with_evidence": 1, "in_context": 1},
    }
    assert summary["by_question_type"]["factual"] == {"with_evidence": 1, "in_context": 1}
    assert summary["corpus"]["documents"] == 2 and summary["corpus"]["pages"] == 4
    assert summary["matching"] == {"rules": 2, "shingle": 12, "whole": WHOLE, "part": PART}
    assert summary["problems"] == []
    # Neither the questions nor the quotes are copied into the results.
    assert "flow?" not in (run_dir / "questions.jsonl").read_text()


def test_a_document_with_a_passage_that_was_not_read_is_a_problem(pdfs, tmp_path):
    summary = eval_retrieval.run(
        Scripted(TEXTS, ORDERS),
        QUESTIONS,
        ITEMS,
        pdfs[:1],  # the tank paper is not in the folder
        out_dir=tmp_path / "runs",
        cache_dir=None,
        say=quiet,
    )
    assert summary["problems"] == ["documents that hold a passage were not read: tank"]

    # Not for a question that was not asked.
    summary = eval_retrieval.run(
        Scripted(TEXTS, ORDERS),
        QUESTIONS[:1],
        ITEMS,
        pdfs[:1],
        out_dir=tmp_path / "runs",
        cache_dir=None,
        say=quiet,
    )
    assert summary["problems"] == []


def test_the_report_names_where_each_question_was_lost(pdfs):
    _, measured = measure(Scripted(TEXTS, ORDERS, room=1), pdfs)
    lines = []
    metrics = eval_retrieval.summarize(measured["rows"])
    eval_retrieval.report(measured["rows"], metrics, lines.append)
    text = "\n".join(lines)
    assert "lost in ranking" in text
    assert "no passage in the corpus" in text
    assert "2 of 3 questions have a passage in the corpus (2 passages)" in text
    assert "by domain: AI Papers 0 of 1, DONES 1 of 1; balanced 0.500" in text


@pytest.fixture
def project(pdfs, tmp_path, monkeypatch):
    """Files for the command, which then measures a scripted system."""
    (tmp_path / "questions.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    (tmp_path / "evidence.json").write_text(json.dumps({"items": ITEMS}), encoding="utf-8")
    corpus.save_manifest([manifest_row("loop"), manifest_row("tank")], tmp_path / "manifest.csv")
    asked_for = []

    def build(model_id, options):
        if options not in ("", "room=2"):
            raise ValueError(f"the scripted system has no option {options!r}")
        asked_for.append(model_id)
        return Scripted(TEXTS, ORDERS, room=2 if options else 1)

    monkeypatch.setattr(eval_retrieval, "SYSTEMS", {"scripted": build})
    arguments = ["--system", "scripted", "--model-id", "some/model"]
    arguments += ["--corpus-dir", str(tmp_path / "corpus")]
    arguments += ["--questions", str(tmp_path / "questions.json")]
    arguments += ["--evidence", str(tmp_path / "evidence.json")]
    arguments += ["--manifest", str(tmp_path / "manifest.csv")]
    arguments += ["--out", str(tmp_path / "runs"), "--cache", str(tmp_path / "cache")]
    return arguments, asked_for


def test_the_command_measures_and_reports(project, tmp_path, capsys):
    arguments, asked_for = project
    assert eval_retrieval.main(arguments) == 0
    out = capsys.readouterr().out
    assert asked_for == ["some/model"]
    assert "in the context          1" in out
    # The manifest of the test has no hashes: the files are not the ones it describes.
    assert "the corpus folder is not the manifest's: 2 not the file described" in out
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert (run_dir / "questions.jsonl").exists()
    assert not (tmp_path / "ledger.jsonl").exists()


def test_the_command_passes_a_system_its_options(project, tmp_path, capsys):
    arguments, _ = project
    arguments[arguments.index("scripted")] = "scripted:room=2"
    assert eval_retrieval.main(arguments) == 0
    assert "2.0 chunks" in capsys.readouterr().out

    arguments[arguments.index("scripted:room=2")] = "scripted:room=many"
    with pytest.raises(SystemExit):
        eval_retrieval.main(arguments)
    assert "has no option 'room=many'" in capsys.readouterr().err

    arguments[arguments.index("scripted:room=many")] = "unknown"
    with pytest.raises(SystemExit):
        eval_retrieval.main(arguments)
    assert "no such system: 'unknown'" in capsys.readouterr().err


def test_the_command_can_ask_only_some_questions(project, tmp_path, capsys):
    arguments, _ = project
    assert eval_retrieval.main([*arguments, "--ids", "2"]) == 0
    assert "1 of 1 questions have a passage in the corpus" in capsys.readouterr().out


def test_the_command_records_a_row_in_the_ledger(project, tmp_path, capsys):
    arguments, _ = project
    recording = ["--record", "--ledger", str(tmp_path / "ledger.jsonl")]
    recording += ["--dataset", "three", "--corpus", "two-papers", "--machine", "a laptop"]
    assert eval_retrieval.main([*arguments, *recording]) == 0

    (row,) = ledger.read(tmp_path / "ledger.jsonl")
    assert (row["kind"], row["dataset"], row["corpus"]) == ("retrieval", "three", "two-papers")
    assert (row["engine"], row["machine"]) == ("none", "a laptop")
    assert row["metrics"]["in_context"] == 1 and row["metrics"]["with_evidence"] == 2
    assert row["config"]["system"] == "scripted"
    assert row["config"]["corpus_documents"] == 2 and row["config"]["chunks"] == 4
    assert row["config"]["corpus_is_the_manifest"] is False
    assert row["timings"]["parsed_now"] == 2
    # The row and the folder of per-question results carry the same name.
    assert (tmp_path / "runs" / row["run_id"] / "summary.json").exists()


def test_the_command_fails_when_a_passage_could_not_be_looked_for(project, tmp_path, capsys):
    arguments, _ = project
    (tmp_path / "corpus" / "tank.pdf").unlink()
    assert eval_retrieval.main(arguments) == 1
    assert "PROBLEM: documents that hold a passage were not read: tank" in capsys.readouterr().out


def test_the_command_refuses_an_empty_folder(project, tmp_path):
    arguments, _ = project
    for path in (tmp_path / "corpus").iterdir():
        path.unlink()
    with pytest.raises(SystemExit):
        eval_retrieval.main(arguments)

"""dsqa/parse.py: a PDF as text, with its title and its tables.

The title is looked for in pages described the way PyMuPDF describes them,
and the text is read from PDFs made in the tests.
"""

import pymupdf
from dsqa import chunk, parse

from tests.unit.stand_ins import BODY, Words, paper_pdf


def test_a_pdf_is_read_as_a_paragraph_per_block(tmp_path):
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        first = "The main loop carries liquid lith-\nium to the target."
        page.insert_text((72, 72), first, fontsize=9)
        page.insert_text((72, 300), "The trap is in the dump tank.", fontsize=9)
        document.new_page().insert_text((72, 72), "A second page.", fontsize=9)
        document.save(path)
    assert parse.read(path, title=False, tables=False) == (
        "The main loop carries liquid lithium to the target.\n\n"
        "The trap is in the dump tank.\n\nA second page."
    )


# A page as PyMuPDF describes one, as far as the title is looked for in it:
# blocks of lines of spans, each span with its text, its type size and where
# it starts and ends across the page.
def span(text, size, left=72.0, right=None):
    right = left + 0.5 * size * len(text) if right is None else right
    return {"text": text, "size": size, "bbox": (left, 0.0, right, 0.0)}


def line(*spans, down=0.0, turned=False):
    box = (spans[0]["bbox"][0], down, spans[-1]["bbox"][2], down + 10.0)
    return {"spans": list(spans), "bbox": box, "dir": (0.0, -1.0) if turned else (1.0, 0.0)}


class Page:
    def __init__(self, *blocks):
        self.blocks = [{"type": 0, "lines": list(lines)} for lines in blocks]

    def get_text(self, kind="text"):
        if kind == "dict":
            return {"blocks": [*self.blocks, {"type": 1}]}  # and a picture, which has no lines
        lines = [line for block in self.blocks for line in block["lines"]]
        return "\n".join(" ".join(span["text"] for span in line["spans"]) for line in lines)


class Paper:
    def __init__(self, *pages, title=""):
        self.pages, self.metadata = pages, {"title": title}

    page_count = property(lambda self: len(self.pages))

    def __getitem__(self, number):
        return self.pages[number]


def first_page():
    return Page(
        [line(span("arXiv:2101.00001v1 [physics.acc-ph] 1 Jan 2021", 20.0), turned=True)],
        [line(span("Knowledge-BasedSystems", 13.9), down=40.0)],
        [
            line(span("A study of the lithium loop", 13.4), down=90.0),
            line(span("of the plant", 13.4), down=106.0),
        ],
        [line(span("Some Author, Another Author", 10.0), down=130.0)],
        [line(span("The loop carries the lithium to the target and back.", 9.0), down=160.0)],
    )


def test_the_title_is_the_largest_type_of_the_first_page_that_makes_three_words():
    # Not the stamp up the margin, which is turned, nor the journal's name,
    # which is larger than the title and one word.
    title = parse.pdf_title(Paper(first_page()))
    assert title == "A study of the lithium loop of the plant"
    assert parse.pdf_title(Paper()) == ""
    assert parse.pdf_title(Paper(Page([line(span("12", 30.0))]))) == ""


def test_the_title_a_pdf_declares_is_taken_when_the_first_page_has_it():
    declared = "A Study of the Lithium Loop of the Plant"
    assert parse.pdf_title(Paper(first_page(), title=declared)) == declared
    # What a word processor leaves there is not on the page.
    left_there = "Microsoft Word - loop study final.docx"
    assert parse.pdf_title(Paper(first_page(), title=left_there)) == (
        "A study of the lithium loop of the plant"
    )


def test_words_set_apart_by_a_gap_are_words_and_what_touches_is_one():
    # As TeX sets a line: a span for each word, the space between them a gap.
    words_apart = line(
        span("Byzantine-Robust", 17.2, 100.0, 220.0),
        span("Learning", 17.2, 225.4, 290.0),
        span("Rates", 17.2, 295.4, 330.0),
        span("1", 8.0, 330.5, 334.0),
    )
    assert parse._lines(Page([words_apart])) == [
        (17.2, 0.0, 100.0, 0, "Byzantine-Robust Learning Rates1")
    ]


def test_a_paper_set_in_one_size_gives_its_first_block_for_a_title():
    body = "The Daugavet property is studied in spaces of functions. " * 8
    page = Page(
        [line(span("LUSHNESS AND THE DAUGAVET PROPERTY", 10.0), down=50.0)],
        [line(span(body, 10.0), down=90.0)],
    )
    assert parse.pdf_title(Paper(page)) == "LUSHNESS AND THE DAUGAVET PROPERTY"


def test_the_title_of_a_real_pdf_opens_its_titled_text(tmp_path):
    path = paper_pdf(tmp_path / "paper.pdf", BODY)
    with pymupdf.open(path) as document:
        assert parse.pdf_title(document) == "A study of the lithium loop of the plant"
    titled = parse.read(path, tables=False)
    assert titled.split("\n\n")[0] == "A study of the lithium loop of the plant"
    # The text itself is the same, title or not.
    assert titled.split("\n\n", 1)[1] == parse.read(path, title=False, tables=False)


def test_the_tables_of_a_real_pdf_are_rows_under_their_captions(tmp_path):
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "A study of the lithium loop", fontsize=18)
        page.insert_text((72, 104), "of the plant", fontsize=18)
        page.insert_text((72, 140), "The lines of the loop were measured.", fontsize=9)
        page.insert_text((72, 180), "Table 1. Flows of the loop.", fontsize=9)
        for down in (190, 208, 240):
            page.draw_line((72, down), (380, down))
        rows = [("", "Flow", "Pressure"), ("Main loop", "97.5", "3.0"), ("Trap", "", "1.5")]
        for number, row in enumerate(rows):
            for across, text in zip((80, 200, 300), row, strict=True):
                page.insert_text((across, 203 + 15 * number), text, fontsize=9)
        document.save(path)

    read = parse.read(path)
    assert read.split("\n\n") == [
        "A study of the lithium loop of the plant",  # the title the PDF tells
        "A study of the lithium loop of the plant",  # and the block it is set in
        "The lines of the loop were measured.",
        "Table 1. Flows of the loop.",
        "|  | Flow | Pressure |\n| Main loop | 97.5 | 3.0 |\n| Trap |  | 1.5 |",
    ]
    # Without tables the same words come as PyMuPDF's blocks give them.
    loose = parse.read(path, tables=False)
    assert "|" not in loose
    assert sorted(read.replace("|", " ").split()) == sorted(loose.split())
    assert parse.read(path, title=False) == read.split("\n\n", 1)[1]

    # The sections chunker keeps such a table under its caption.
    chunks = chunk.sectioned(read, Words(), 40, title=True)
    assert chunks[-1] == (
        "A study of the lithium loop of the plant\n\nTable 1. Flows of the loop.\n\n"
        "|  | Flow | Pressure |\n| Main loop | 97.5 | 3.0 |\n| Trap |  | 1.5 |"
    )

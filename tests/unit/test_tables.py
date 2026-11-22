"""dsqa/tables.py: the tables of a page, found by the rules drawn across them.

The pages are drawn in the tests: text set at chosen places and rules drawn
over and under it, which is all the module goes by.
"""

import re
from collections import Counter

import pymupdf
import pytest
from dsqa import tables
from dsqa.tables import (
    CAPTION,
    as_markdown,
    boxes_of,
    frames_of,
    rows_of,
    rules_of,
    set_in_columns,
)
from dsqa.text import clean

ROWS = [
    ["", "Flow", "Pressure"],
    ["Main loop", "97.5", "3.0"],
    ["Trap", "", "1.5"],
    ["Dump tank", "0.65", "1.0"],
]
GRID = "|  | Flow | Pressure |\n| Main loop | 97.5 | 3.0 |\n| Trap |  | 1.5 |"
GRID += "\n| Dump tank | 0.65 | 1.0 |"
LEFT, RIGHT = 72, 380
PROSE = "The loop carries the lithium to the target and back,\nthrough the trap and the tank."


def write(page, where, text):
    page.insert_text(where, text, fontsize=9)


def draw_table(page, top, caption="", under=False, rows=ROWS, columns=(80, 200, 300)):
    """A table as a paper sets it: a rule over it, one under its first row and
    one under its last, and its caption over the first rule or under the last.
    Returns how far down the page it ends."""
    down = top
    if caption and not under:
        write(page, (LEFT, down), caption)
        down += 12
    page.draw_line((LEFT, down), (RIGHT, down))
    for number, row in enumerate(rows):
        down += 14
        for across, cell in zip(columns, row, strict=True):
            if cell:
                write(page, (across, down), cell)
        if number == 0:
            page.draw_line((LEFT, down + 4), (RIGHT, down + 4))
    page.draw_line((LEFT, down + 5), (RIGHT, down + 5))
    if caption and under:
        write(page, (LEFT, down + 19), caption)
        return down + 19
    return down + 5


def plain(page):
    """The page as PyMuPDF's blocks alone give it."""
    texts = [clean(block[4]) for block in page.get_text("blocks") if block[6] == 0]
    return [text for text in texts if text]


def letters(paragraphs):
    return Counter(re.findall(r"\w+", " ".join(paragraphs).replace("|", " ")))


@pytest.mark.parametrize(
    "opening",
    [
        "Table 1. Flows of the loop",
        "Table 2: Results",
        "TABLE IV: The testing error rates",
        "Tab. 3. Parameters",
        "Table S1: Evaluation in Atari",
        "Table A.2 Hyperparameters",
        "Table 6 Test loss and accuracy",
        "Table 4 (continued)",
        "Table 2 \N{EN DASH} Neutron irradiation facilities",
        "Table 12",
    ],
)
def test_a_caption_opens_with_the_label_of_a_table(opening):
    assert CAPTION.match(opening)


@pytest.mark.parametrize(
    "opening",
    [
        "Table 2 shows the results of the test.",
        "Table I lists the parameters",
        "Tables 1 and 2 give the flows",
        "Tabulated values are given below",
        "In Table 2, we summarize the parameters",
        "Fig. 3: The training error",
    ],
)
def test_a_sentence_about_a_table_is_not_its_caption(opening):
    assert not CAPTION.match(opening)


def test_the_rules_of_a_page_are_its_long_horizontal_strokes():
    with pymupdf.open() as document:
        page = document.new_page()
        page.draw_line((72, 100), (380, 100))
        page.draw_line((72, 140), (90, 140))  # too short for a rule
        page.draw_line((200, 100), (200, 300))  # upright
        page.draw_rect(pymupdf.Rect(72, 180, 380, 181), fill=(0, 0, 0))  # a rule drawn as a box
        page.draw_rect(pymupdf.Rect(72, 300, 380, 400))  # a box is not a rule
        # A rule drawn in three pieces, cell by cell, the last one first.
        page.draw_line((250, 220), (380, 220))
        page.draw_line((72, 220), (150, 220))
        page.draw_line((151, 220), (249, 220))
        # Two strokes on one level and far apart are two rules.
        page.draw_line((72, 260), (150, 260))
        page.draw_line((250, 260), (380, 260))
        rules = [tuple(round(value) for value in rule) for rule in rules_of(page)]
    whole = [(down, 72, 380) for down in (100, 180, 220)]
    assert rules == [*whole, (260, 72, 150), (260, 250, 380)]


def test_rules_with_the_same_ends_frame_a_table():
    rules = [(100.0, 72.0, 380.0), (118.0, 72.0, 380.0), (160.0, 72.0, 380.0)]
    assert frames_of(rules) == [rules]
    # A rule by itself frames nothing, and neither do two with other ends.
    assert frames_of(rules[:1]) == []
    assert frames_of([rules[0], (160.0, 72.0, 300.0)]) == []
    # The rules under a group of headings are inside the table they are part of.
    inside = [(109.0, 200.0, 380.0), (130.0, 200.0, 380.0)]
    assert frames_of(sorted(rules + inside)) == [rules]
    # Two tables side by side.
    beside = [(100.0, 400.0, 540.0), (160.0, 400.0, 540.0)]
    assert frames_of(sorted(rules + beside)) == [rules, beside]


def test_a_caption_between_two_rules_parts_them():
    first = [(100.0, 72.0, 380.0), (160.0, 72.0, 380.0)]
    second = [(200.0, 72.0, 380.0), (260.0, 72.0, 380.0)]
    assert frames_of(first + second) == [first + second]
    assert frames_of(first + second, [(72.0, 170.0, 300.0, 182.0)]) == [first, second]
    # A caption beside them, in another column, parts nothing.
    assert frames_of(first + second, [(400.0, 170.0, 540.0, 182.0)]) == [first + second]


def test_running_text_between_two_rules_parts_a_frame():
    frame = [(100.0, 72.0, 380.0), (118.0, 72.0, 380.0), (160.0, 72.0, 380.0), (240.0, 72.0, 380.0)]
    cells = [(80.0, 120.0, "Trap"), (300.0, 320.0, "1.5")]
    label = [(80.0, 150.0, "The main loop")]
    line = [(80.0, 370.0, "The loop carries the lithium to the target and back")]

    def rows_in(between):
        def rows(box):
            return between.get((box[1], box[3]), [])

        return rows

    whole = [(72.0, 100.0, 380.0, 240.0)]
    assert boxes_of(frame, rows_in({})) == whole
    assert boxes_of(frame, rows_in({(160.0, 240.0): [cells, label, cells]})) == whole
    # A line by itself names the rows under it.
    assert boxes_of(frame, rows_in({(118.0, 160.0): [label]})) == whole
    # A paragraph does not: what is under it is not this table.
    for text in ([line], [label, label]):
        assert boxes_of(frame, rows_in({(160.0, 240.0): text})) == [(72.0, 100.0, 380.0, 160.0)]
        assert boxes_of(frame, rows_in({(118.0, 160.0): text})) == [
            (72.0, 100.0, 380.0, 118.0),
            (72.0, 160.0, 380.0, 240.0),
        ]
        # Under the first rule, it leaves that rule alone over it, to frame nothing.
        assert boxes_of(frame, rows_in({(100.0, 118.0): text})) == [(72.0, 118.0, 380.0, 240.0)]


def word(text, across, down, line, block=0, height=10.0):
    return (across, down, across + 5.0 * len(text), down + height, text, block, line, 0)


def test_the_words_on_one_line_are_a_row_and_a_gap_starts_a_cell():
    words = [
        word("Main", 80, 100, 0),
        word("loop", 103, 100, 0),  # a space after "Main": the same cell
        word("97.5", 200, 100, 1),  # another line of PyMuPDF's at the same height
        word("3.0", 300, 100.5, 0, block=1),
        word("Trap", 80, 114, 2),
        word("1.5", 300, 114, 0, block=2),
    ]
    rows = rows_of(words)
    assert [[text for _, _, text in row] for row in rows] == [
        ["Main loop", "97.5", "3.0"],
        ["Trap", "1.5"],
    ]
    assert rows[0][0][:2] == (80, 123)
    # Two words of one line of PyMuPDF's with a wide gap between them are two cells.
    assert [[text for _, _, text in row] for row in rows_of([
        word("Rig", 80, 100, 0), word("11", 97, 100, 0), word("Rig", 112, 100, 0),
    ])] == [["Rig 11", "Rig"]]  # fmt: skip
    assert rows_of([]) == []


def cell(text, across):
    return (float(across), across + 5.0 * len(text), text)


def test_cells_are_set_under_their_columns_with_empty_ones_where_a_row_has_none():
    rows = [
        [cell("Flow", 200), cell("Pressure", 300)],
        [cell("Main loop", 80), cell("97.5", 200), cell("3.0", 300)],
        [cell("Trap", 80), cell("1.5", 300)],
        [cell("Dump tank", 80), cell("0.65", 200)],
    ]
    assert set_in_columns(rows) == [
        ["", "Flow", "Pressure"],
        ["Main loop", "97.5", "3.0"],
        ["Trap", "", "1.5"],
        ["Dump tank", "0.65"],  # an empty cell at the end of a row is left out
    ]
    assert as_markdown(set_in_columns(rows)).splitlines()[2] == "| Trap |  | 1.5 |"


def test_a_heading_over_several_columns_keeps_them_in_one_group():
    rows = [
        [cell("Detector", 80), cell("Estimated current (nA)", 200)],
        [cell("Rig 11", 200), cell("Rig 45", 270)],
        [cell("Neutrons", 80), cell("5.37", 200), cell("42.75", 270)],
        [cell("Gammas", 80), cell("523", 270)],
    ]
    assert set_in_columns(rows) == [
        ["Detector", "Estimated current (nA)"],
        ["", "Rig 11", "Rig 45"],
        ["Neutrons", "5.37", "42.75"],
        ["Gammas", "", "523"],
    ]


def test_cells_keep_their_order_and_none_is_merged_with_another():
    # Under a heading over two columns, the cells of a row that lie off both
    # still get a place each, in order: the nearest column, the one after the
    # cell before, and room left for the cells that follow.
    heading = [cell("Detector", 80), cell("Estimated current in nA", 200)]
    columns = [cell("Neutrons", 80), cell("5.37", 200), cell("42.75", 290)]
    rows = [
        heading,
        columns,
        [cell("a", 80), cell("b", 225), cell("c", 235)],  # both nearest to the first
        [cell("d", 80), cell("e", 290), cell("f", 300)],  # both over the second
        [cell("g", 80), cell("h", 292)],
        [cell("A line by itself", 80)],
    ]
    assert set_in_columns(rows) == [
        ["Detector", "Estimated current in nA"],
        ["Neutrons", "5.37", "42.75"],
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g", "", "h"],
        ["A line by itself"],
    ]
    # With fewer than two rows of several cells there are no columns to find.
    assert set_in_columns([heading, [cell("c", 300)]]) == [
        ["Detector", "Estimated current in nA"],
        ["c"],
    ]


def test_a_ruled_table_is_read_as_rows_under_its_caption():
    with pymupdf.open() as document:
        page = document.new_page()
        write(page, (72, 80), PROSE)
        end = draw_table(page, 130, "Table 1. Flows and pressures of the loop.")
        write(page, (72, end + 30), "The trap is in the dump tank.")
        read = tables.paragraphs(page)
        assert read == [
            clean(PROSE),
            "Table 1. Flows and pressures of the loop.",
            GRID,
            "The trap is in the dump tank.",
        ]
        # Every word of the page is there, once.
        assert letters(read) == letters(plain(page))


def test_a_caption_under_its_table_is_put_before_it():
    with pymupdf.open() as document:
        page = document.new_page()
        end = draw_table(page, 100, "Table 1. Flows and pressures of the loop.", under=True)
        write(page, (72, end + 40), PROSE)
        read = tables.paragraphs(page)
    assert read == ["Table 1. Flows and pressures of the loop.", GRID, clean(PROSE)]


def test_what_is_not_a_captioned_table_stays_as_it_was():
    with pymupdf.open() as document:
        page = document.new_page()
        # Rules under the caption of a figure, a table without rules, and a
        # sentence that opens with the name of a table.
        end = draw_table(page, 100, "Fig. 1. A drawing with rules across it.")
        write(page, (72, end + 30), "Table 3 shows nothing, and this is a sentence.")
        write(page, (72, end + 60), "Table 4. A table without rules.")
        for number, row in enumerate(ROWS[1:]):
            for across, text in zip((80, 200, 300), row, strict=True):
                write(page, (across, end + 74 + 14 * number), text)
        assert tables.paragraphs(page) == plain(page)
        # A page with no caption of a table is not looked at any further.
        blank = document.new_page()
        write(blank, (72, 80), PROSE)
        assert tables.paragraphs(blank) == [clean(PROSE)]
        assert tables.paragraphs(document.new_page()) == []


def test_a_caption_goes_on_down_to_the_first_rule():
    with pymupdf.open() as document:
        page = document.new_page()
        write(page, (72, 100), "Table 1. Flows and pressures")
        write(page, (72, 126), "of the loop, as designed.")  # far enough to be a block of its own
        down = 136
        page.draw_line((LEFT, down), (RIGHT, down))
        for number, row in enumerate(ROWS[1:], 1):
            for across, text in zip((80, 200, 300), row, strict=True):
                if text:
                    write(page, (across, down + 14 * number), text)
        page.draw_line((LEFT, down + 47), (RIGHT, down + 47))
        assert len(plain(page)) > 4
        read = tables.paragraphs(page)
    assert read[0] == "Table 1. Flows and pressures of the loop, as designed."
    assert read[1].splitlines()[0] == "| Main loop | 97.5 | 3.0 |"
    assert len(read) == 2


def test_a_paragraph_running_along_a_table_stays_a_paragraph():
    with pymupdf.open() as document:
        page = document.new_page()
        write(page, (72, 100), "Table 4. A table with text beside it.")
        page.draw_line((72, 110), (500, 110))
        page.draw_line((72, 170), (500, 170))
        for number, (label, value) in enumerate([("Loop", "97.5"), ("Trap", "1.5")]):
            write(page, (330, 126 + 14 * number), label)
            write(page, (430, 126 + 14 * number), value)
        # It starts beside the first row and runs on under the last rule.
        beside = "\n".join(f"line {number} of a paragraph beside it" for number in range(1, 12))
        write(page, (72, 126), beside)
        read = tables.paragraphs(page)
        assert read == [
            "Table 4. A table with text beside it.",
            "| Loop | 97.5 |\n| Trap | 1.5 |",
            clean(beside),
        ]
        assert letters(read) == letters(plain(page))


def test_two_tables_one_under_the_other_are_two():
    with pymupdf.open() as document:
        page = document.new_page()
        end = draw_table(page, 100, "Table 1. Flows and pressures of the loop.")
        end = draw_table(page, end + 30, "Table 2. The same, a year later.")
        first = tables.paragraphs(page)
        # Without a caption between them, the paragraph between two tables
        # of the same width parts them: only the first has a caption.
        other = document.new_page()
        end = draw_table(other, 100, "Table 1. Flows and pressures of the loop.")
        write(other, (72, end + 24), PROSE)
        draw_table(other, end + 70)
        second = tables.paragraphs(other)
    assert first == [
        "Table 1. Flows and pressures of the loop.",
        GRID,
        "Table 2. The same, a year later.",
        GRID,
    ]
    assert second[:3] == ["Table 1. Flows and pressures of the loop.", GRID, clean(PROSE)]
    assert not any(paragraph.startswith("|") for paragraph in second[3:])


def test_a_document_says_on_which_side_of_its_tables_the_captions_are():
    with pymupdf.open() as document:
        # A page where a caption has a table on either side: it is the first
        # one's, under it, or the second one's, over it.
        between = document.new_page()
        end = draw_table(between, 100)
        write(between, (72, end + 14), "Table 2. Flows and pressures of the loop.")
        draw_table(between, end + 26)
        over = tables.paragraphs(between, over=True)
        under = tables.paragraphs(between, over=False)
        assert over[:4] == ["Flow Pressure", "Main loop 97.5 3.0", "Trap 1.5", "Dump tank 0.65 1.0"]
        assert over[4:] == ["Table 2. Flows and pressures of the loop.", GRID]
        assert under[:2] == ["Table 2. Flows and pressures of the loop.", GRID]
        assert under[2:] == over[:4]
        # Left to itself, the page takes the caption to be over its table.
        assert tables.paragraphs(between) == over
        assert tables.pages(document) == [over]

        # Two more pages whose captions can only be under their tables: the
        # document has its captions there, and so has the page that could not tell.
        for number in (3, 4):
            page = document.new_page()
            draw_table(page, 100, f"Table {number}. Flows and pressures of the loop.", under=True)
        assert tables.pages(document)[0] == under


def test_a_turned_page_is_left_as_it_is():
    with pymupdf.open() as document:
        page = document.new_page()
        draw_table(page, 100, "Table 1. Flows and pressures of the loop.")
        assert any(paragraph.startswith("|") for paragraph in tables.paragraphs(page))
        page.set_rotation(90)
        assert not any(paragraph.startswith("|") for paragraph in tables.paragraphs(page))

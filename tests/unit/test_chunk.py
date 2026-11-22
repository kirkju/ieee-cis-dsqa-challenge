"""dsqa/chunk.py: a document's text in chunks that follow its structure.

The tokenizer of these tests has a token per word (tests/unit/stand_ins.py).
"""

from dsqa import chunk

from tests.unit.stand_ins import Words, tokens

PAPER = """## Research Article

## A study of the lithium loop of the plant

Some Author, Another Author

## Abstract

We study the loop. It works.

## 2. Results

The flow is steady.

<!-- image -->

Table 1. Flows of the loop.

| main 97.5 | bypass 2.0 | trap 0.6 | spare 0.1 |

The table shows the flows.

<!-- formula-not-decoded -->

## References

[1] A paper. [2] Another paper.

## Appendix

More detail here."""


def test_sections_keep_a_chunk_inside_its_section_and_under_its_heading():
    chunks = chunk.sectioned(PAPER, Words(), 40)
    assert chunks == [
        "## A study of the lithium loop of the plant\n\nSome Author, Another Author",
        "## Abstract\n\nWe study the loop. It works.",
        "## 2. Results\n\nThe flow is steady.",
        # The table with its caption, apart from the text around it.
        "## 2. Results\n\nTable 1. Flows of the loop.\n\n"
        "| main 97.5 | bypass 2.0 | trap 0.6 | spare 0.1 |",
        "## 2. Results\n\nThe table shows the flows.",
        "## References\n\n[1] A paper. [2] Another paper.",
        "## Appendix\n\nMore detail here.",
    ]
    # Nothing is said of where Docling read nothing, and a heading with
    # nothing under it gives no chunk.
    assert not any("<!--" in chunk or "Research Article" in chunk for chunk in chunks)


def test_a_long_table_is_cut_and_every_part_repeats_its_caption():
    table = "| " + " | ".join(f"cell{number}" for number in range(1, 13)) + " |"
    text = f"## Results\n\nTable 2. Cells.\n\n{table}"
    chunks = chunk.sectioned(text, Words(), 16)
    assert len(chunks) > 1
    assert all(chunk.startswith("## Results\n\nTable 2. Cells.\n\n") for chunk in chunks)
    assert all(tokens(chunk) <= 16 for chunk in chunks)
    cells = " ".join(chunk.split("\n\n")[2] for chunk in chunks)
    assert [word for word in cells.split() if word.startswith("cell")] == [
        f"cell{number}" for number in range(1, 13)
    ]


def test_a_caption_under_a_table_goes_with_it_unless_it_is_the_next_table_s():
    under = "| a 1 | b 2 |\n\nTable 3. Below its table.\n\nText after."
    assert chunk.sectioned(under, Words(), 40) == [
        "Table 3. Below its table.\n\n| a 1 | b 2 |",
        "Text after.",
    ]
    between = "| a 1 | b 2 |\n\nTable 4. Above the second.\n\n| c 3 | d 4 |"
    assert chunk.sectioned(between, Words(), 40) == [
        "| a 1 | b 2 |",
        "Table 4. Above the second.\n\n| c 3 | d 4 |",
    ]


HEAD = "| Line | Flow | Pressure |"
LINES = [f"| line{number} | {number}.5 | {number}.0 |" for number in range(1, 9)]
LEGEND = "Table 2. Flows and pressures of the lines. Measured in the first campaign."


def test_a_table_is_cut_between_its_rows_and_each_part_names_its_columns(monkeypatch):
    # The start of the caption that a later part repeats, kept short here.
    monkeypatch.setattr(chunk, "BRIEF", 8)
    table = "\n".join([HEAD, *LINES])
    parts = chunk.table_parts(table, LEGEND, Words(), 40)
    assert parts == [
        [LEGEND, "\n".join([HEAD, *LINES[:2]])],
        ["Table 2. Flows and pressures of the lines.", "\n".join([HEAD, *LINES[2:5]])],
        ["Table 2. Flows and pressures of the lines.", "\n".join([HEAD, *LINES[5:]])],
    ]
    assert all(tokens(" ".join(part)) <= 40 for part in parts)

    # A table that fits is one part, under its whole caption; one without a
    # caption is its rows alone.
    assert chunk.table_parts(table, LEGEND, Words(), 100) == [[LEGEND, table]]
    assert chunk.table_parts(table, "", Words(), 100) == [[table]]
    bare = chunk.table_parts(table, "", Words(), 30)
    assert [part[0].split("\n")[0] for part in bare] == [HEAD, HEAD, HEAD]


def test_the_line_of_dashes_of_a_markdown_table_stays_under_its_first_row():
    dashes = "|---|---|---|"
    table = "\n".join([HEAD, dashes, *LINES])
    parts = chunk.table_parts(table, "", Words(), 30)
    assert all(part[0].startswith(f"{HEAD}\n{dashes}\n| line") for part in parts)
    rows = [row for part in parts for row in part[0].split("\n")[2:]]
    assert rows == LINES


def test_a_row_too_long_for_a_part_of_a_table_is_cut_as_text_is():
    long_row = "| " + " ".join(f"word{number}" for number in range(60)) + " |"
    parts = chunk.table_parts("\n".join([HEAD, long_row, LINES[0]]), "", Words(), 30)
    assert all(tokens(" ".join(part)) <= 30 for part in parts)
    words_read = " ".join(" ".join(part) for part in parts).split()
    assert [word for word in words_read if word.startswith("word")] == long_row.split()[1:-1]
    assert LINES[0] in parts[-1][0]


def test_sections_give_a_table_the_room_asked_for_it():
    table = "\n".join([HEAD, *LINES])
    text = f"## Results\n\nThe lines were measured.\n\n{LEGEND}\n\n{table}\n\nText after the table."
    as_text = chunk.sectioned(text, Words(), 20)
    assert all(tokens(chunk) <= 20 for chunk in as_text)
    assert sum(chunk.count("Table 2.") for chunk in as_text) > 2

    roomy = chunk.sectioned(text, Words(), 20, table_tokens=90)
    assert roomy == [
        "## Results\n\nThe lines were measured.",
        f"## Results\n\n{LEGEND}\n\n{table}",
        "## Results\n\nText after the table.",
    ]
    # With less room it is cut between rows, each part under the heading of
    # the section, the caption and the first row.
    parts = chunk.sectioned(text, Words(), 20, table_tokens=50)[1:-1]
    assert len(parts) > 1
    assert all(part.startswith(f"## Results\n\n{LEGEND}\n\n{HEAD}\n") for part in parts)
    assert [row for part in parts for row in part.split("\n")[5:]] == LINES


def test_sections_can_leave_the_references_out_and_put_the_title_in():
    chunks = chunk.sectioned(PAPER, Words(), 40, references=False)
    assert not any("[1] A paper" in chunk for chunk in chunks)
    assert chunks[-1] == "## Appendix\n\nMore detail here."  # what follows the list is kept

    titled = chunk.sectioned(PAPER, Words(), 40, title=True)
    assert all(chunk.startswith("A study of the lithium loop of the plant\n\n") for chunk in titled)
    assert titled[2] == (
        "A study of the lithium loop of the plant\n\n## 2. Results\n\nThe flow is steady."
    )


def test_the_title_is_the_longest_of_the_first_headings():
    paragraphs = PAPER.split("\n\n")
    assert chunk.title_of(paragraphs) == "A study of the lithium loop of the plant"
    # With no heading to go by, the first words of the document.
    assert chunk.title_of(["No headings here.", "Nor here."]) == "No headings here."
    assert chunk.title_of([]) == ""

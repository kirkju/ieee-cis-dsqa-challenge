"""A PDF as text: its title, then its paragraphs in the order of the file.

PyMuPDF reads the text layer of a document in a fraction of a second, and
loses none of it. What its text lacks is what a reader sees at a glance and
a ranking needs: which words are the title, and which blocks are a table.
Both are taken from the page itself, the title from the size of its type and
the tables from the rules drawn across them (tables.py).
"""

import itertools
from collections import Counter
from pathlib import Path

import pymupdf

from . import glyphs
from . import tables as tables_of_pages
from .text import clean, squash

# A line of type can be a title when it makes at least this many words and
# letters; what is longer than TITLE_LENGTH characters is the body of a paper
# set in a single size, and only its first block is taken.
TITLE_WORDS, TITLE_LETTERS, TITLE_LENGTH = 3, 12, 300
# Two words of a line are apart when the gap between them is this much of the
# type size: TeX sets each word by itself and leaves the space as a gap.
WORD_GAP = 0.15


def _lines(page) -> list[tuple[float, float, float, int, str]]:
    """The horizontal lines of a page: the type size most of each is set in,
    how far down and across the page it starts, its block and its text."""
    lines = []
    for number, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            across, down = line.get("dir", (1.0, 0.0))
            if abs(across - 1) > 0.01 or abs(down) > 0.01:
                continue  # turned: the stamp arXiv prints up the margin
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            sizes: Counter = Counter()
            for span in spans:
                sizes[round(span["size"], 1)] += len(span["text"].strip())
            text = spans[0]["text"]
            for before, span in itertools.pairwise(spans):
                gap = span["bbox"][0] - before["bbox"][2]
                apart = gap > WORD_GAP * max(before["size"], span["size"])
                text += (" " if apart else "") + span["text"]
            size = sizes.most_common(1)[0][0]
            lines.append((size, line["bbox"][1], line["bbox"][0], number, " ".join(text.split())))
    return lines


def _can_be_a_title(text: str) -> bool:
    worded = [word for word in text.split() if any(character.isalpha() for character in word)]
    return (
        len(worded) >= TITLE_WORDS
        and sum(character.isalpha() for character in text) >= TITLE_LETTERS
    )


def pdf_title(document) -> str:
    """A document's title as its PDF tells it, or "" when it does not.

    The title its metadata declares, when the first page has those words:
    publishers fill it in, and what a word processor leaves there instead is
    not on the page. Otherwise the text of the first page set in the largest
    type that makes at least three words, which passes over a report number
    or a journal's name in one word.
    """
    if document.page_count == 0:
        return ""
    page = document[0]
    declared = clean((document.metadata or {}).get("title") or "")
    if _can_be_a_title(declared) and squash(declared) in squash(page.get_text()):
        return declared[:TITLE_LENGTH]
    by_size: dict[float, list] = {}
    for size, down, across, block, text in _lines(page):
        by_size.setdefault(size, []).append((down, across, block, text))
    for size in sorted(by_size, reverse=True):
        lines = sorted(by_size[size])
        text = clean("\n".join(line[3] for line in lines))
        if not _can_be_a_title(text):
            continue
        if len(text) > TITLE_LENGTH:
            first = lines[0][2]
            text = clean("\n".join(line[3] for line in lines if line[2] == first))
        return text[:TITLE_LENGTH]
    return ""


def read(pdf: Path | str, title: bool = True, tables: bool = True) -> str:
    """A document as text, its paragraphs a blank line apart.

    Each block of text PyMuPDF finds is a paragraph, in the order of the
    file, with its lines joined and its words rejoined across line breaks.
    With title, the document's title comes first, as a paragraph of its own,
    when the PDF tells it. With tables, the blocks of a table that is ruled
    and captioned give way to its caption and, under it, its rows of cells.
    """
    paragraphs = []
    with pymupdf.open(pdf) as document:
        if title:
            paragraphs.append(pdf_title(document))
        if tables:
            paragraphs.extend(itertools.chain.from_iterable(tables_of_pages.pages(document)))
        else:
            for page in document:
                blocks = [block for block in page.get_text("blocks") if block[6] == 0]
                symbols = glyphs.table_of(page, (block[4] for block in blocks))
                for block in blocks:  # blocks of text, not images
                    paragraphs.append(clean(block[4].translate(symbols) if symbols else block[4]))
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)

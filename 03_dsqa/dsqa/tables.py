"""The tables of a PDF page, found by the rules drawn across them.

PyMuPDF gives the text of a page as blocks, in the order of the file. A table
comes out of that as its caption in one block and its cells in others, a row
or a column at a time, and a chunker that packs blocks until a chunk is full
leaves the numbers of a table in one chunk and the caption that says what
they are in another. A chunk of bare numbers has nothing a question can be
matched with.

A table in a paper is ruled off: a rule over it, one under its headings and
one under it, or a rule under every row. Reading the strokes a page draws
costs little, and they tell where its tables are:

    rules    the horizontal strokes of the page; one drawn cell by cell is joined
    frames   rules with the same ends, one under the other, and what they close off
    tables   a frame with the caption of a table right over it or right under it

The text inside a table is read by where it is on the page, not in the order
of the file. The words on one line are a row and a gap between two of them
starts another cell. The cells are then set under the columns the rows share,
so that a row with a cell missing keeps its other cells in place.

    Table 2. Estimated signal current in the detectors.

    |  | Rig 11 | Rig 45 |
    | Total | 29.6(5) | 659(11) |

A row is a line of the page: a cell whose text runs over two lines gives two
rows, the second with its other cells empty. Nothing is merged that the page
does not put on one line.

What is not found stays as PyMuPDF gives it: a table without rules or without
a caption, one printed sideways, a picture of a table. No word of a page is
dropped or repeated: a block of text goes whole either into a table or into a
paragraph of its own.
"""

import itertools
import re
from collections.abc import Sequence

import numpy as np
import pymupdf

from . import glyphs
from .text import clean

# A block that opens as the caption of a table: the label and its number, then
# a mark, the end of the block or a word with a capital. "Table 2 shows that"
# opens a sentence.
CAPTION = re.compile(
    r"(?:Table|TABLE|Tab\.)\s*(?:\d+|[IVXLC]+|[A-Z]\.?\d+)"
    r"(?:\s*[.:|\N{EM DASH}\N{EN DASH}-]|\s*$|\s+(?=[A-Z(\[]))"
)

# What a rule is, in points. A stroke is part of one from PIECE long and no
# more than THIN high, which takes in a rule drawn as a thin box; the pieces
# of a rule are on one LEVEL and within JOINED of each other; joined, the rule
# of a table is at least RULE long.
PIECE, THIN, LEVEL, JOINED, RULE = 4.0, 2.5, 1.0, 3.0, 40.0
# How far a caption can be from the first or the last rule of its table: room
# for a second block of the caption between them.
CAPTION_GAP = 45.0
# A block is inside a frame when this much of its width is, and it reaches no
# more than EDGE past the first and the last rule: a line of text set close
# to a rule is, a paragraph that runs along the table and on under it is not.
INSIDE, EDGE = 0.8, 4.0
# Two words of a row are in different cells when the gap between them is more
# than this share of their height. Inside the lines of the tables of the
# development corpus the gaps gather in two heaps, the spaces between words at
# 0.15 to 0.35 and the gaps between cells from 0.45 up, and are fewest between
# 0.40 and 0.45 (journal, 2026-10-10).
CELL_GAP = 0.45
# What a frame holds is a table when at least this share of its rows have
# more than one cell; a box around a paragraph has rows of one.
SEVERAL = 1 / 3

Rule = tuple[float, float, float]  # how far down the page, and from where to where
Box = tuple[float, float, float, float]  # left, top, right, bottom
Cell = tuple[float, float, str]  # from where to where across the page, and its text


def rules_of(page) -> list[Rule]:
    """The horizontal rules a page draws, from the top down. A rule drawn in
    pieces, one for each cell it runs under, is one rule."""
    pieces = []
    for drawing in page.get_cdrawings():
        for item in drawing.get("items", ()):
            if item[0] == "l":
                (x0, y0), (x1, y1) = item[1], item[2]
            elif item[0] == "re":
                x0, y0, x1, y1 = item[1]
            else:
                continue
            if abs(y1 - y0) <= THIN and abs(x1 - x0) >= PIECE:
                pieces.append(((y0 + y1) / 2, min(x0, x1), max(x0, x1)))
    rules: list[Rule] = []
    level: list[Rule] = []  # the pieces on one level, the highest first
    for piece in [*sorted(pieces), None]:
        if level and (piece is None or piece[0] - level[0][0] > LEVEL):
            down = level[0][0]
            across = sorted((x0, x1) for _, x0, x1 in level)
            start, end = across[0]
            for x0, x1 in across[1:]:
                if x0 <= end + JOINED:
                    end = max(end, x1)
                else:
                    rules.append((down, start, end))
                    start, end = x0, x1
            rules.append((down, start, end))
            level = []
        if piece is not None:
            level.append(piece)
    return sorted(rule for rule in rules if rule[2] - rule[1] >= RULE)


def frames_of(rules: Sequence[Rule], captions: Sequence[Box] = ()) -> list[list[Rule]]:
    """The rules of a page by what they close off: each frame as its rules,
    from the top down.

    The rules that close a table run across the same stretch of the page, so
    rules with the same ends, one under the other, frame one, and a shorter
    rule between them is inside it. A caption between two rules parts them:
    what is under it is another table.
    """
    frames: list[list[Rule]] = []
    for rule in rules:
        down, x0, x1 = rule
        for frame in reversed(frames):
            bottom, left, right = frame[-1]
            if abs(left - x0) > JOINED or abs(right - x1) > JOINED:
                continue
            if any(
                bottom < box[1] and box[3] < down + 1 and box[0] < x1 and box[2] > x0
                for box in captions
            ):
                continue
            frame.append(rule)
            break
        else:
            frames.append([rule])
    framed = [frame for frame in frames if len(frame) > 1]

    def within(frame: list[Rule], other: list[Rule]) -> bool:
        return (
            other[0][1] - JOINED <= frame[0][1]
            and frame[0][2] <= other[0][2] + JOINED
            and other[0][0] - LEVEL <= frame[0][0]
            and frame[-1][0] <= other[-1][0] + LEVEL
        )

    # The rules under a group of headings frame nothing of their own.
    return [
        frame
        for frame in framed
        if not any(other is not frame and within(frame, other) for other in framed)
    ]


def boxes_of(frame: Sequence[Rule], rows_in) -> list[Box]:
    """The tables a frame may hold, as boxes: the frame, cut where running
    text lies between two of its rules.

    Between two rules of a table lie rows of cells, a line by itself that
    names the rows under it, or nothing. Lines of a single cell, several of
    them or one more than half as long as the rules, are a paragraph: the
    rules over it and under it belong to two tables, or one of the two is not
    a table's. `rows_in` gives the rows of cells inside a box.
    """
    left, right = min(rule[1] for rule in frame), max(rule[2] for rule in frame)
    boxes = []
    top = bottom = frame[0][0]
    for above, below in itertools.pairwise(frame):
        rows = rows_in((left, above[0], right, below[0]))
        if (
            rows
            and all(len(row) == 1 for row in rows)
            and (len(rows) > 1 or rows[0][0][1] - rows[0][0][0] > (right - left) / 2)
        ):
            if bottom > top:
                boxes.append((left, top, right, bottom))
            top = below[0]
        bottom = below[0]
    if bottom > top:
        boxes.append((left, top, right, bottom))
    return boxes


def _inside(box: Box, frame: Box) -> bool:
    """Whether a block of text lies inside a frame."""
    width = box[2] - box[0]
    across = min(box[2], frame[2]) - max(box[0], frame[0])
    return (
        width > 0
        and across >= INSIDE * width
        and frame[1] - EDGE <= box[1]
        and box[3] <= frame[3] + EDGE
    )


def _pairs(captions: dict[int, Box], frames: list[Box], over: bool) -> dict[int, int]:
    """The frame of each caption, when all the captions of a page are over
    their tables or all are under them: the nearest pairs first."""
    near = []
    for caption, (x0, y0, x1, y1) in captions.items():
        for number, (left, top, right, bottom) in enumerate(frames):
            gap = top - y1 if over else y0 - bottom
            if min(x1, right) > max(x0, left) and -JOINED <= gap <= CAPTION_GAP:
                near.append((gap, caption, number))
    pairs: dict[int, int] = {}
    for _, caption, number in sorted(near):
        if caption not in pairs and number not in pairs.values():
            pairs[caption] = number
    return pairs


def rows_of(words: list[tuple]) -> list[list[Cell]]:
    """Words, as PyMuPDF gives them with their boxes, as rows of cells.

    The words PyMuPDF puts on one line stay together, and lines at the same
    height are one row. Along a row, a gap of more than CELL_GAP of the
    words' height starts another cell.
    """
    lines: dict[tuple[int, int], list[tuple]] = {}
    for word in words:
        if word[4].strip():
            lines.setdefault((word[5], word[6]), []).append(word)
    spans = []
    for line in lines.values():
        top, bottom = min(word[1] for word in line), max(word[3] for word in line)
        spans.append(((top + bottom) / 2, min(word[0] for word in line), top, bottom, line))
    rows: list[list] = []  # what the lines of a row share of the height, and its words
    for middle, _, top, bottom, line in sorted(spans, key=lambda span: span[:2]):
        if rows and rows[-1][0] - LEVEL <= middle <= rows[-1][1] + LEVEL:
            row = rows[-1]
            row[0], row[1] = max(row[0], top), min(row[1], bottom)
            row[2].extend(line)
        else:
            rows.append([top, bottom, list(line)])
    table = []
    for _, _, row in rows:
        row.sort(key=lambda word: word[0])
        cells = [[row[0]]]
        for before, word in itertools.pairwise(row):
            height = max(before[3] - before[1], word[3] - word[1])
            if word[0] - before[2] > CELL_GAP * height:
                cells.append([])
            cells[-1].append(word)
        read = [(cell[0][0], max(word[2] for word in cell), _cell_text(cell)) for cell in cells]
        read = [cell for cell in read if cell[2]]
        if read:
            table.append(read)
    return table


def _cell_text(words: list[tuple]) -> str:
    return clean(" ".join(word[4] for word in words))


def _over(cell: Cell, columns: list[tuple[float, float]]) -> int:
    """The first of the columns a cell lies over, or the one nearest to it."""
    nearest, apart = 0, float("inf")
    for number, (start, end) in enumerate(columns):
        gap = max(cell[0], start) - min(cell[1], end)
        if gap < 0:
            return number
        if gap < apart:
            nearest, apart = number, gap
    return nearest


def set_in_columns(rows: list[list[Cell]]) -> list[list[str]]:
    """Rows of cells as a grid: each cell under its column, and an empty cell
    where a row has none.

    The columns are found in two steps. Across the table, a stretch that no
    cell of any row crosses parts two groups of columns: a heading over
    several columns holds its group together. Inside a group, the columns are
    the cells of the row that has most of them there, and a cell of another
    row goes under the first of them it lies over. Cells keep their order and
    none is merged with another; empty cells at the end of a row are left out.

    Rows of a single cell say nothing about the columns and are not asked.
    """
    several = [row for row in rows if len(row) > 1]
    if len(several) < 2:
        return [[text for _, _, text in row] for row in rows]
    left = min(cell[0] for row in rows for cell in row)
    right = max(cell[1] for row in rows for cell in row)
    crossed = np.zeros(int(right - left) + 2, dtype=bool)
    for row in several:
        for x0, x1, _ in row:
            crossed[int(x0 - left) : int(x1 - left) + 1] = True
    turns = np.flatnonzero(np.diff(np.concatenate([[False], crossed, [False]]).astype(np.int8)))
    turns = (left + turns).tolist()
    groups = list(zip(turns[::2], turns[1::2], strict=True))

    placed = []  # for each row, its cells by group
    for row in rows:
        mine: list[list[Cell]] = [[] for _ in groups]
        for cell in row:
            mine[_over(cell, groups)].append(cell)
        placed.append(mine)
    grid: list[list[str]] = [[] for _ in rows]
    for number in range(len(groups)):
        model = max((mine[number] for mine in placed), key=len)
        columns = [(x0, x1) for x0, x1, _ in model]
        for out, mine in zip(grid, placed, strict=True):
            cells = [""] * len(columns)
            slot = -1
            for at, cell in enumerate(mine[number]):
                # Under its column, after the cell before it, and with room
                # left for the cells that follow.
                slot = max(_over(cell, columns), slot + 1)
                slot = min(slot, len(columns) - (len(mine[number]) - at))
                cells[slot] = cell[2]
            out.extend(cells)
    for out in grid:
        while out and not out[-1]:
            out.pop()
    return grid


def as_markdown(grid: list[list[str]]) -> str:
    """A grid as the rows of a Markdown table, without a line under its head:
    which rows are headings is not known."""
    return "\n".join("| " + " | ".join(cells) + " |" for cells in grid)


def _translated(item: tuple, symbols: dict[int, str]) -> tuple:
    """A block or a word of PyMuPDF's, its text (the fifth field) translated."""
    return (*item[:4], item[4].translate(symbols), *item[5:])


class _Page:
    """What a page tells its tables by, as plain data: once this is made,
    PyMuPDF's page is not needed again."""

    def __init__(self, page):
        textpage = page.get_textpage(flags=pymupdf.TEXTFLAGS_BLOCKS)
        blocks = page.get_text("blocks", textpage=textpage)
        self.blocks = [block for block in blocks if block[6] == 0]  # of text, not images
        # symbol-glyphs: the codes of the Symbol font read as their characters.
        self.symbols = glyphs.table_of(page, (block[4] for block in self.blocks))
        if self.symbols:
            self.blocks = [_translated(block, self.symbols) for block in self.blocks]
        self.texts = [clean(block[4]) for block in self.blocks]
        self.captions: dict[int, Box] = {}
        self.words: dict[int, list[tuple]] = {}
        self.frames: list[Box] = []
        # The frame of each caption, were the captions over their tables, and
        # were they under them.
        self.over: dict[int, int] = {}
        self.under: dict[int, int] = {}
        captions = {
            at: self.blocks[at][:4] for at, text in enumerate(self.texts) if CAPTION.match(text)
        }
        if not captions or page.rotation:
            return
        self.captions = captions
        for word in page.get_text("words", textpage=textpage):
            if self.symbols:
                word = _translated(word, self.symbols)
            self.words.setdefault(word[5], []).append(word)
        self.frames = [
            box
            for frame in frames_of(rules_of(page), list(captions.values()))
            for box in boxes_of(frame, self.rows_in)
        ]
        self.over = _pairs(captions, self.frames, True)
        self.under = _pairs(captions, self.frames, False)

    def inside(self, box: Box, taken: Sequence[int] = ()) -> list[int]:
        """The blocks of text inside a box that are neither captions nor taken."""
        return [
            at
            for at, block in enumerate(self.blocks)
            if at not in self.captions
            and at not in taken
            and self.texts[at]
            and _inside(block[:4], box)
        ]

    def rows_in(self, box: Box, taken: Sequence[int] = ()) -> list[list[Cell]]:
        inside = self.inside(box, taken)
        return rows_of([word for at in inside for word in self.words.get(self.blocks[at][5], [])])

    def paragraphs(self, over: bool | None = None) -> list[str]:
        """The page's paragraphs, with the captions taken to be over their
        tables or under them; left open, the side that pairs more of this
        page's captions with a frame."""
        if over is None:
            over = len(self.over) >= len(self.under)
        # What the side taken leaves alone may still have a frame on the other.
        first, second = (self.over, self.under) if over else (self.under, self.over)
        pairs = dict(first)
        for caption, number in second.items():
            if caption not in pairs and number not in pairs.values():
                pairs[caption] = number

        tables: dict[int, list[str]] = {}  # what takes the place of a caption's block
        taken: set[int] = set()
        for caption, number in pairs.items():
            frame = self.frames[number]
            inside = self.inside(frame, taken)
            rows = self.rows_in(frame, taken)
            if sum(len(row) > 1 for row in rows) < max(1.0, SEVERAL * len(rows)):
                continue
            # A caption over its table may go on in a block of its own, down
            # to the first rule.
            x0, _, x1, bottom = self.captions[caption]
            gap = (min(x0, frame[0]), bottom, max(x1, frame[2]), frame[1])
            more = [at for at in self.inside(gap, taken) if at not in inside]
            taken.update(inside, more)
            head = " ".join(self.texts[at] for at in [caption, *more])
            tables[caption] = [head, as_markdown(set_in_columns(rows))]

        read = []
        for at, text in enumerate(self.texts):
            if at in tables:
                read.extend(tables[at])
            elif at not in taken and text:
                read.append(text)
        return read


def paragraphs(page, over: bool | None = None) -> list[str]:
    """The paragraphs of a page: its blocks of text in the order of the file,
    and each table found on it as two paragraphs in place of its blocks, the
    caption and under it the rows.

    A page with no caption of a table, or one that is turned, gives its
    blocks as they are. `over` says on which side of its tables the document
    has its captions; left open, the page decides for itself.
    """
    return _Page(page).paragraphs(over)


def pages(document) -> list[list[str]]:
    """The paragraphs of every page of a document, as paragraphs() gives them.

    A paper puts its captions on one side of its tables, and a page alone
    cannot always tell which: a caption between two tables has one on either
    side. The side is the one that pairs more captions with a frame over the
    whole document.
    """
    found = [_Page(page) for page in document]
    over = sum(len(page.over) for page in found) >= sum(len(page.under) for page in found)
    return [page.paragraphs(over) for page in found]

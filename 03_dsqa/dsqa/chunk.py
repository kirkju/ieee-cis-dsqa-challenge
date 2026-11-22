"""A document's text in chunks that follow its structure.

A chunk is what is ranked for a question, and the embedder reads only so
many tokens of it: what is past them is not in the embedding. So a chunk is
cut to that, and it is cut where the text lets it be: at the end of a
section, of a paragraph, of a sentence. Each chunk opens with what says where
it is from, the title of the document and the heading of its section, and a
table goes in chunks of its own under its caption.

The text is Markdown as far as it has any marks: a line that opens with "#"
is a heading and a paragraph that opens with "|" is a table. Text without
them is chunked by its paragraphs alone.
"""

import re

# Where a text may be cut, best first: after a sentence, after a clause, at a space.
CUTS = (re.compile(r"[.!?]\s+"), re.compile(r"[;:,|]\s+"), re.compile(r"\s+"))

HEADING = re.compile(r"#+\s*")
REFERENCES = re.compile(
    r"(?i)#+\s*(\d+\.?\s*)?(references( and notes)?|bibliography|literature cited)\s*"
)
TABLE_CAPTION = re.compile(r"(Table|TABLE|Tab\.)\s*[\dIVXA-Z]")
# How many tokens of its caption a part of a table repeats after the first.
BRIEF = 48
# What a converter to Markdown leaves where it read nothing: a picture, an
# equation it did not decode.
PLACEHOLDER = re.compile(r"<!--.*?-->")
# First headings that are not the title of a paper.
FURNITURE = re.compile(
    r"(?i)(abstract|a b s t r a c t|article info.*|keywords.*|(research|original) article"
    r"|preprint.*|manuscript|contents)"
)


class Words:
    """A tokenizer with a token per word, for when the embedder's is not
    there to cut the chunks by: it answers the two things the chunker asks a
    tokenizer, the tokens of a text and where each starts and ends."""

    def __call__(self, text: str, **_) -> dict:
        spans = [match.span() for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def pieces(paragraph: str, tokenizer, limit: int) -> list[tuple[str, int]]:
    """A paragraph as pieces of at most `limit` tokens, each with its count.

    One that fits is a single piece. A longer one is cut where the limit
    falls: at the last sentence end before it, or the last clause, or the
    last space, and within the second half of what fits.
    """
    offsets = tokenizer(
        paragraph, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )["offset_mapping"]
    if len(offsets) <= limit:
        return [(paragraph, len(offsets))]
    cut_pieces = []
    first = 0  # the token the next piece starts with
    while first < len(offsets):
        last = min(first + limit, len(offsets))  # one past the last token that fits
        start = offsets[first][0]
        if last == len(offsets):
            end = len(paragraph)
        else:
            # Cut inside the second half of what fits, so pieces stay sizeable.
            floor = offsets[first + (last - first) // 2][0]
            window = paragraph[floor : offsets[last][0]]
            end = offsets[last][0]
            for cut in CUTS:
                matches = list(cut.finditer(window))
                if matches:
                    end = floor + matches[-1].end()
                    break
            # The tokens up to the cut are the piece; the next starts after it.
            while last > first + 1 and offsets[last - 1][1] > end:
                last -= 1
        piece = paragraph[start:end].strip()
        if piece:
            cut_pieces.append((piece, last - first))
        first = last
    return cut_pieces


def _kind(paragraph: str) -> str:
    if paragraph.startswith("#"):
        return "heading"
    if paragraph.startswith("|"):
        return "table"
    if TABLE_CAPTION.match(paragraph):
        return "caption"
    return "nothing" if PLACEHOLDER.fullmatch(paragraph) else "text"


def title_of(paragraphs: list[str]) -> str:
    """A document's title as the start of its text tells it: the longest of
    its first headings, which passes over "Research Article" and the like,
    and with no heading to go by its first paragraph."""
    headings = [HEADING.sub("", p, count=1).strip() for p in paragraphs[:12] if p.startswith("#")]
    headings = [heading for heading in headings if not FURNITURE.fullmatch(heading)]
    if headings:
        return max(headings[:4], key=len)
    return paragraphs[0][:200] if paragraphs else ""


def table_parts(table: str, caption: str, tokenizer, room: int) -> list[list[str]]:
    """A table as parts of at most `room` tokens, cut between its rows: the
    paragraphs each part is made of.

    The first part has the caption whole. Each part after it opens with the
    start of the caption, up to BRIEF tokens of it, and with the first row of
    the table, which names its columns: a part then says by itself what its
    numbers are. A row too long for a part is cut as text is.
    """

    def count(piece: str) -> int:
        return len(tokenizer(piece, add_special_tokens=False, verbose=False)["input_ids"])

    rows = table.split("\n")
    # Under the first row of a Markdown table goes a line of dashes.
    head = rows[:2] if len(rows) > 1 and re.fullmatch(r"[\s|:-]+", rows[1]) else rows[:1]
    brief = pieces(caption, tokenizer, BRIEF)[0][0] if caption else ""
    again = count(brief) + sum(count(row) for row in head)
    parts: list[list[str]] = []
    opening, body, used = caption, [], count(caption)
    for row in rows:
        for piece, tokens in pieces(row, tokenizer, max(room - again, room // 4)):
            if body and used + tokens > room:
                parts.append([opening, "\n".join(body)])
                opening, body, used = brief, list(head), again
            body.append(piece)
            used += tokens
    parts.append([opening, "\n".join(body)])
    return [[paragraph for paragraph in part if paragraph] for part in parts]


def sectioned(
    text: str,
    tokenizer,
    limit: int,
    *,
    title: bool = False,
    references: bool = True,
    table_tokens: int = 0,
) -> list[str]:
    """Chunks of at most `limit` tokens that follow the structure of the text.

    A chunk stays inside a section and starts with the section's heading. A
    table is kept with its caption, apart from the text around it; one too
    long for a chunk is cut, and each part repeats the caption. Placeholders
    for pictures and undecoded equations are dropped. With `title`, every
    chunk starts with the document's title; without `references`, the
    reference list is left out. With `table_tokens`, a table may take that
    many tokens of a chunk instead of `limit`, and is cut as table_parts()
    cuts it.
    """
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    kinds = [_kind(paragraph) for paragraph in paragraphs]
    top = [title_of(paragraphs)] if title and paragraphs else []
    chunks: list[str] = []
    section: list[str] = []  # the heading the next chunks start with
    body: list[str] = []
    used, skipping = 0, False

    def count(piece: str) -> int:
        return len(tokenizer(piece, add_special_tokens=False, verbose=False)["input_ids"])

    # The tokens of the lines every chunk of the section starts with. Counted
    # when the section opens: asked for with every paragraph, counting them
    # again each time took more than the rest of the chunking together.
    opened = sum(count(line) for line in top)

    def room() -> int:
        # What is left for the text once the lines every chunk starts with are in.
        return max(limit - opened, limit // 2)

    def emit(parts: list[str]) -> None:
        chunks.append("\n\n".join([*top, *section, *parts]))

    def flush() -> None:
        nonlocal body, used
        if body:
            emit(body)
        body, used = [], 0

    at = 0
    while at < len(paragraphs):
        paragraph, kind = paragraphs[at], kinds[at]
        following = kinds[at + 1] if at + 1 < len(paragraphs) else ""
        at += 1
        if kind == "heading":
            flush()
            skipping = not references and bool(REFERENCES.fullmatch(paragraph))
            section = [paragraph]
            opened = sum(count(line) for line in [*top, *section])
            continue
        if skipping or kind == "nothing":
            continue
        caption, table = "", ""
        if kind == "caption" and following == "table":
            caption, table = paragraph, paragraphs[at]
            at += 1
        elif kind == "table":
            table = paragraph
            # A caption under the table is its own unless another table follows it.
            after = kinds[at + 1] if at + 1 < len(paragraphs) else ""
            if following == "caption" and after != "table":
                caption = paragraphs[at]
                at += 1
        if table and table_tokens:
            flush()
            space = max(table_tokens - opened, table_tokens // 2)
            for part in table_parts(table, caption, tokenizer, space):
                emit(part)
            continue
        if table:
            flush()
            space = max(room() - count(caption), limit // 4)
            for piece, _ in pieces(table, tokenizer, space):
                emit([caption, piece] if caption else [piece])
            continue
        for piece, tokens in pieces(paragraph, tokenizer, room()):
            if body and used + tokens > room():
                flush()
            body.append(piece)
            used += tokens
    flush()
    return chunks

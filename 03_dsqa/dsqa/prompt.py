"""What the model is told and shown: the instructions, and the passages
found for a question laid out as text.

The answers are marked right or wrong against a reference by a judge that
gives no partial credit, and the best ones are then read by experts. So the
decisive fact has to be stated outright, with the working when there is a
calculation, and a question that assumes something false has to be
corrected, as some of the organizers' sample questions do.

The first version asked for the answer first and the reasoning after. With
the base model, nine of 98 answers then said two things: a value stated at
once, and another reached in the working after it, most often the right
one. So the model is asked to work the question out first and to state
the answer last, once. What is returned of its reply puts that statement
first, where a reader looks for it, and the working after.
"""

import re
from collections.abc import Sequence

from . import plain, variant

# The line a reply ends with, and what tells it in a reply: at the start of a
# line, under whatever emphasis the model put on it.
FINAL = "Final answer:"
_FINAL = re.compile(r"^[ \t>*_#-]*final answer[ \t*_]*:[ \t*_]*", re.IGNORECASE | re.MULTILINE)

SYSTEM = (
    "You are a scientific expert who answers questions about research papers and "
    "technical reports, on accelerator-based neutron sources for fusion materials "
    "(IFMIF-DONES) and on artificial intelligence. Passages retrieved from a "
    "collection of documents may come before the question. Use them when they "
    "hold the answer, and your own knowledge when they do not.\n"
    "Work the question out first, in a few sentences: what the answer rests on, "
    "and what that says. When the question needs a calculation, write the formula, "
    "the values and the result. When the question assumes something that the "
    "passages contradict, say so and give the correct fact. Name a source by its "
    "title or by the table or figure that holds the fact, never by its place among "
    f'the passages. Then end with one line that begins with "{FINAL}" and states '
    "the answer once, as you worked it out: the fact, the number with its unit, or "
    "the conclusion that was asked for. State no answer before that line. Be "
    "precise and brief."
)
# What the model is told when its reply stated no answer: it was cut before
# its last line, or it did not write one.
CLOSE = f'State the answer now, in one or two sentences, on a line that begins with "{FINAL}".'

# What stands between two chunks of a document that do not follow one another.
GAP = "[...]"


def title_and_body(chunk: str) -> tuple[str, str]:
    """A chunk as the title of its document, which every chunk opens with,
    and the rest of it."""
    title, _, body = chunk.partition("\n\n")
    return title, body


def context_of(placed: Sequence[tuple[int, str]], documents: Sequence[int]) -> str:
    """The chunks placed for a question as the text the model reads.

    The chunks are ranked one by one, and each opens with the title of its
    document so that it can be. Read one after the other they would say that
    title dozens of times. Here the chunks of a document are put together in
    the order of the document under its title, said once, with a mark where
    two of them do not follow one another. The documents come in the order
    they were placed in, the one of the best passage last: next to the
    question. Of the base model's 64 layers, 48 attend to the last 4,096
    tokens only, so what is far from the question is read by the other 16
    alone. That this order answers better has not been tried with the model.
    """
    by_document: dict[int, list[tuple[int, str]]] = {}
    for number, text in placed:
        by_document.setdefault(int(documents[number]), []).append((number, text))
    blocks = []
    for chunks in reversed(list(by_document.values())):
        chunks.sort()
        # Named by its title and not numbered: a number is what the model
        # then cites, and "document 5" tells the reader of an answer nothing.
        lines = [f"Document: {title_and_body(chunks[0][1])[0]}"]
        before = None
        for number, text in chunks:
            if before is not None and number != before + 1:
                lines.append(GAP)
            lines.append(title_and_body(text)[1] or text)
            before = number
        blocks.append("\n\n".join(lines))
    return "\n\n\n".join(blocks)


def messages(question: str, context: str = "") -> list[dict]:
    """The conversation the model completes: the instructions, then the
    passages, if any, and the question.

    With plain (variant.py) the instruction and the turn are the organizers'
    RAG example's, over our context: its opening line, the context, the line
    that closes it and the question with no label (plain.py)."""
    asked = question.strip()
    if variant.on("plain"):
        if context:
            asked = f"{plain.CONTEXT_OPENING}\n\n{context}\n\n---\n\n{asked}"
        return [
            {"role": "system", "content": plain.SYSTEM_PROMPT},
            {"role": "user", "content": asked},
        ]
    if context:
        asked = (
            "Passages retrieved from the document collection:\n\n"
            f"{context}\n\n---\n\nQuestion: {asked}"
        )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": asked}]


def closing(question: str, context: str, working: str) -> list[dict]:
    """The conversation gone on when a reply stated no answer: the model is
    shown what it wrote and asked for the answer alone."""
    said = {"role": "assistant", "content": working}
    return [*messages(question, context), said, {"role": "user", "content": CLOSE}]


def split(reply: str) -> tuple[str, str] | None:
    """A reply as its working and the answer it states after it; None when
    it states none. The answer is what follows the last line that begins as
    the model was asked to begin it."""
    found = list(_FINAL.finditer(reply))
    if not found:
        return None
    answer = reply[found[-1].end() :].strip()
    if not answer:
        return None
    return reply[: found[-1].start()].strip(), answer


def as_answer(reply: str) -> str:
    """What is returned of a reply: the answer it states, and after it the
    working it came from. A reply that states none is returned as it is."""
    parts = split(reply)
    if parts is None:
        return reply
    working, answer = parts
    return f"{answer}\n\n{working}" if working else answer

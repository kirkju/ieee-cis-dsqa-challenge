"""Named adjustments of the entry, to measure side by side.

Each adjustment is a change the entry can make or not, under a name. They
are chosen by the environment variable DSQA_VARIANT, a list of names
separated by commas:

    DSQA_VARIANT=dones-zero python -m tools.run_local 03_dsqa ...

With the variable unset the entry is the one of DEFAULT, which is none of
them: the entry as it was before any was written. The evaluation sets no
variable, so an entry that is to run with some of them has their names
written in DEFAULT.

A name that is not one of KNOWN is an error, so that a misspelt name cannot
run the entry without the adjustment it meant to measure. This entry knows
the ways IFMIF-DONES questions may be answered (route.py), plain, the base
model's judgement of the relevance of the chunks of a question answered the
dones-plain way (relevance.py), the context of the document a question on AI
papers is about (doc-focus), llm-acted and llm-proof, which make the entry a
measurement of whether that judgement runs (route.py), and the glyphs of the
Symbol font read as the characters they are (symbol-glyphs); with none of
them chosen it is the entry it was.
"""

import os

# The adjustments the entry runs with when the environment names none.
DEFAULT = "doc-focus,doc-name,doc-reserve,dones-plain,llm-one,llm-proof,llm-rank,llm-wide,symbol-glyphs"

# Every adjustment there is, and what it changes.
KNOWN = {
    "dones-example": "a question the code tells is on IFMIF-DONES (route.py) answered by the "
    "organizers' RAG example itself, its own index and its prompt (example.py), when that "
    "module is there and its index was made; the other questions as before",
    "dones-plain": "a question the code tells is on IFMIF-DONES (route.py) answered with the "
    "organizers' RAG example's instruction and form over our ranking of our chunks, best "
    "first within 10,000 tokens, the reply as it is; the other questions as before",
    "dones-zero": "a question the code tells is on IFMIF-DONES (route.py) answered as the "
    "organizers' zero-shot example answers it: its instruction, the question alone, the "
    "reply as it is; the other questions as before",
    "dones-lite": "a question the code tells is on IFMIF-DONES (route.py) answered with the "
    "organizers' RAG example's code over the entry's own text (lite.py): its chunks of 800 "
    "characters, its embedder on the CPU, its context, prompt and form, without Docling; "
    "the other questions as before",
    "dones-nowhole": "a question the code tells is on IFMIF-DONES (route.py) answered by the "
    "entry's own prompt and protocol, with no document placed whole in its context: its "
    "chunks by their rank alone; the other questions as before",
    "plain": "every question asked with the organizers' examples' instruction and their turn "
    "over our own context (the RAG example's opening line, the context, the question with no "
    "label), the reply returned as it is, with no line asked for apart",
    "doc-focus": "when the document of the best chunk is also the first by the words of its "
    "three best chunks, the context is that document alone (focus.py): whole when it fits, "
    "else its best chunks with their neighbours; where they differ, or where the routing "
    "tells the question is on IFMIF-DONES, the context as it was",
    "doc-reserve": "with doc-focus, a quarter of a focused context kept for the best chunks of "
    "the ranking, of whatever document, placed before the document identified, which has "
    "the rest and stays last, next to the question (focus.placed)",
    "doc-name": "with doc-focus, a question that gives a name (an acronym, a word with an inner "
    "capital, letters with digits, rare in the corpus) that the head of one of its candidate "
    "documents says is not focused on a document whose head says none of them (focus.vetoed)",
    "llm-rank": "a question answered the dones-plain way has the first chunks of its ranking "
    "read by the base model one by one, and ranked by fusing their order with that of the "
    "model's P(Yes) (relevance.py); the other questions as before",
    "llm-gate": "with llm-rank, the chunks the model gives a low P(Yes) left out of the "
    "dones-plain context, the best few kept whatever their score",
    "llm-doc": "the document whose chunks the model scores highest taken for the question's, "
    "and its chunks put first in the dones-plain context",
    "llm-pick": "with llm-doc, when the first two documents score close, the model shown the "
    "first three side by side and asked which one the question is about",
    "llm-wide": "with llm-rank, llm-gate or llm-doc, 80 chunks scored instead of 40",
    "llm-one": "with llm-rank, llm-gate or llm-doc, each chunk scored alone through the "
    "model's generate(), the way every reply is read, instead of in batches padded on the "
    "left (engine.first): the same scores but for the rounding, a pass for each chunk",
    "llm-acted": "a measurement, not an answerer: with dones-plain and llm-rank, a question "
    "the code tells is on IFMIF-DONES is answered as without llm-acted only when the base "
    "model scored every chunk it was to score and its scores ranked (relevance.py), and "
    "every other question gets a fixed sentence (route.UNANSWERED) with no reply of the "
    "model asked for; the score is then that of those questions alone, and says whether "
    "the scoring runs where the entry is evaluated",
    "llm-proof": "with dones-plain and llm-rank, every question answered as without it, but "
    "that a question the code tells is on IFMIF-DONES gets the fixed sentence "
    "(route.UNANSWERED), with no reply of the model asked for, as long as the base model's "
    "scoring (relevance.py) has left nothing to rank by in every question of the run so far; "
    "from the first it ranks, even cut for time, nothing is left out. On the leaderboard the "
    "questions on IFMIF-DONES score only if the scoring runs there",
    "symbol-glyphs": "every PDF read with the codes the Symbol font draws (Greek letters, "
    "the micro sign, ±, ≈ and the like) as the characters they stand for, on each page the "
    "codes only that font draws there (glyphs.py); a code another font draws on the page, "
    "and the text of the questions, as before",
}


class UnknownVariant(ValueError):
    """A name that is not that of an adjustment."""


def chosen() -> frozenset[str]:
    """The names of the adjustments the entry runs with."""
    given = os.environ.get("DSQA_VARIANT")
    names = DEFAULT if given is None else given
    picked = frozenset(name.strip() for name in names.split(",") if name.strip())
    unknown = sorted(picked - set(KNOWN))
    if unknown:
        raise UnknownVariant(f"no adjustment is named {', '.join(unknown)}")
    return picked


def on(name: str) -> bool:
    """Whether the entry runs with an adjustment."""
    if name not in KNOWN:
        raise UnknownVariant(f"no adjustment is named {name}")
    return name in chosen()


def described() -> str:
    """The adjustments the entry runs with, as a line of its log."""
    return ",".join(sorted(chosen())) or "none"

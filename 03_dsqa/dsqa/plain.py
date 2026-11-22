"""A question asked as the organizers' examples ask it.

The organizers' starter kit has two examples (01_zero_shot/main.py and
02_rag/main.py in their repository). Both give the model one instruction,
the same, and return its reply as it is, with no line the answer has to be
stated on and nothing cut from it. The zero-shot example asks the question
alone; the RAG example puts the chunks it found before the question, best
first, each under a number, within 10,000 tokens. On the public
leaderboard the RAG example scored what our first entry did, 57.14, and
the zero-shot example 29.05.

Here the two are reproduced on our own engine, for the adjustments of
route.py: the zero-shot example as it is, and the RAG example's prompt and
form over our ranking of our chunks. The reply is bounded by the time the
question has, as every reply of the entry is, where the examples bound it
by 2,048 tokens alone.
"""

from .context import pack

# The examples' own words and numbers.
SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)
MAX_NEW_TOKENS = 2048
MAX_CONTEXT_TOKENS = 10_000
CONTEXT_OPENING = "Context extracted from a blind search over the corpus:"
# With less time than this left for a question, it is not asked this way:
# the entry's own way is left to say what it can.
LEAST_SECONDS = 4.0


def zero_shot_messages(question: str) -> list[dict]:
    """The conversation of the zero-shot example."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def context_of(fragments: list[str]) -> str:
    """The context of the RAG example: its fragments best first, numbered
    from 1, under its opening line and over the line that closes it."""
    if not fragments:
        return ""
    body = "\n\n".join(f"[Fragment {i}]\n{c}" for i, c in enumerate(fragments, 1))
    return f"{CONTEXT_OPENING}\n\n{body}\n\n---\n\n"


def rag_messages(question: str, fragments: list[str]) -> list[dict]:
    """The conversation of the RAG example, over the fragments given."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_of(fragments) + question},
    ]


def fragments_of(order, chunks, count, budget: int = MAX_CONTEXT_TOKENS) -> list[str]:
    """The chunks of a ranking, best first, as many as fit in the budget, as
    the RAG example takes them: until one does not fit, and a slice of the
    first when not even that one does."""
    placed, _ = pack([int(number) for number in order], chunks, count, budget)
    return [text for _, text in placed]


def asked(engine, messages: list[dict], seconds: float) -> str:
    """The engine's reply, as it is but for the space around it; "" when
    there is no time, no reply, or the engine fails. The GPU is given back."""
    if seconds < LEAST_SECONDS:
        return ""
    try:
        reply = engine.generate(messages, MAX_NEW_TOKENS, seconds)
    finally:
        try:
            engine.release()
        except Exception:
            pass
    return reply.strip() if isinstance(reply, str) else ""


def zero_shot(engine, question: str, seconds: float) -> str:
    """A question answered as the zero-shot example answers it."""
    return asked(engine, zero_shot_messages(question), seconds)


def over_ranking(engine, question: str, index, order, seconds: float) -> str:
    """A question answered with the RAG example's prompt and form over the
    chunks of our ranking, best first."""
    fragments = []
    if index is not None and order is not None:
        counts = index.__dict__.setdefault("_plain_tokens", {})

        def count(text: str) -> int:
            if text not in counts:
                counts[text] = engine.count(text)
            return counts[text]

        fragments = fragments_of(order, index.chunks, count)
    return asked(engine, rag_messages(question, fragments), seconds)

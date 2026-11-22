"""The chunks that go into the context of a question."""

from collections.abc import Callable, Sequence

import numpy as np


def pack(
    order: Sequence[int],
    chunks: Sequence[str],
    count: Callable[[str], int],
    budget: int,
    documents: Sequence[int] | None = None,
    whole: float = 0.0,
) -> tuple[list[tuple[int, str]], int]:
    """The chunks that go into a context of `budget` tokens, each with its
    number, and the tokens they take as `count` counts them.

    Chunks are taken from the top of the ranking until one does not fit. If
    not even the first one fits, a slice of it is given: the share of its
    characters that the budget is of its tokens.

    With `documents`, which says for every chunk the document it is of, the
    document of the best chunk goes in first and whole, in its own order,
    when all of it takes no more than the share `whole` of the budget. The
    ranking then fills what is left.
    """
    context: list[tuple[int, str]] = []
    used, taken = 0, set()
    if documents is not None and len(order) and whole:
        first = documents[order[0]]
        mine = [int(number) for number in np.flatnonzero(np.asarray(documents) == first)]
        costs = [count(chunks[number]) for number in mine]
        if sum(costs) <= whole * budget:
            context = [(number, chunks[number]) for number in mine]
            used, taken = sum(costs), set(mine)
    for number in order:
        number = int(number)
        if number in taken:
            continue
        chunk = chunks[number]
        tokens = count(chunk)
        if used + tokens > budget:
            if not context and tokens > 0:
                context.append((number, chunk[: int(len(chunk) * (budget / tokens))]))
            break
        context.append((number, chunk))
        used += tokens
        taken.add(number)
    return context, used

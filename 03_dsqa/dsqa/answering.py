"""A question answered: the passages found for it, the model asked, and what
is said when either is missing.

The evaluation shows a status and a score and nothing else. A run that
raises, or that passes its time, can fail as a whole, so nothing here is
allowed to: each part that can go wrong has something lesser to fall back on.

    the embedder does not load        the corpus is ranked by words alone
    the corpus cannot be indexed      the model answers from what it knows
    the GPU runs out of memory        the context is halved, then left out
    the model fails on a question     it is asked again without passages
    its reply states no answer        it is asked for the answer alone
    there is no model, or no time     the best passage found is the answer
"""

import time
from collections.abc import Callable

from . import config, focus, prompt, retrieval, route, variant
from .budget import Budget
from .engine import out_of_memory

# What is returned when there is neither a model's answer nor a passage: a
# blank answer would count as none given.
NO_ANSWER = "No answer could be produced for this question."
# How much of the best passage is given when it stands in for an answer.
PASSAGE_CHARACTERS = 1_200


class Answerer:
    """The corpus indexed and an engine to ask, for one run.

    `engine` is what generates (engine.py), or None when the model could not
    be loaded. `embedder` is the embedding model, None for none, and when it
    is left out the one shipped with the entry is loaded. `started` is when
    our code began to run, on `clock`.
    """

    def __init__(
        self,
        corpus_dir,
        engine,
        started: float | None = None,
        say: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        embedder="shipped",
    ):
        self.engine, self.say, self.clock = engine, say, clock
        # Kept for a way of route.py that reads the corpus its own way.
        self.corpus_dir = corpus_dir
        self.budget = Budget(started, clock)
        self.asked = 0
        self.context_tokens = config.CONTEXT_TOKENS
        self.index = self._index(corpus_dir, embedder)
        # The documents' domains, for a way of route.py, told now and not
        # in the time of the first question.
        route.prepared(self.index, self.say)

    def _index(self, corpus_dir, embedder):
        """The index of the corpus, or None when there is nothing to search."""
        before = self.clock()
        if isinstance(embedder, str):
            try:
                embedder = retrieval.load_embedder()
            except Exception as error:
                self.say(f"[dsqa] no embedder, ranking by words alone: {_named(error)}")
                embedder = None
        try:
            index = retrieval.build(corpus_dir, embedder, self.say, config.READING_SECONDS)
        except Exception as error:
            self.say(f"[dsqa] no index, answering without passages: {_named(error)}")
            return None
        self.say(f"[dsqa] corpus indexed in {self.clock() - before:.0f}s")
        return index if index.chunks else None

    def _count(self, text: str) -> int:
        """Tokens of a text as the model counts them; without a model, a
        guess from its length."""
        return self.engine.count(text) if self.engine is not None else len(text) // 3

    def _passages(self, question: str, tokens: int) -> tuple[list, list]:
        """The chunks placed in a context of so many tokens, and the ranking."""
        if self.index is None:
            return [], []
        try:
            # The share a whole document may take: the entry's, unless a way
            # of route.py set another for this question (dones-nowhole).
            whole = getattr(self, "whole", None)
            placed, used, order = self.index.context(
                question, self._count, budget=tokens, whole=whole
            )
        except Exception as error:
            self.say(f"[dsqa] no passages for this question: {_named(error)}")
            return [], []
        if focus.chosen():
            # doc-focus: what the context was made of, or why it is as before.
            made = getattr(self.index, "focused", None)
            self.say(focus.said(made, used, self.index.chunks, self.index.documents, self.asked))
        return placed, list(order)

    def _best_passage(self, order: list) -> str:
        """The best-ranked chunk, without its title, cut to a readable length."""
        if not order:
            return NO_ANSWER
        body = prompt.title_and_body(self.index.chunks[int(order[0])])[1]
        return body[:PASSAGE_CHARACTERS].strip() or NO_ANSWER

    def answer(self, question: str) -> str:
        """The answer to a question. Always a string, and never blank."""
        began = self.clock()
        allowance = self.budget.allowance(self.asked)
        self.asked += 1
        # A question on IFMIF-DONES may be answered another way (route.py);
        # when it is not, or that way fails, it is answered as any other.
        routed = route.answered(self, question, began, allowance)
        if routed:
            return routed
        if route.measuring():
            # llm-acted: no question is left to the entry's own way.
            return route.unanswered(self, "no answer came from route.py")
        tokens = self.context_tokens
        placed, order = self._passages(question, tokens)
        if self.engine is None:
            return self._best_passage(order)

        reply, context = "", ""
        while True:
            left = allowance - (self.clock() - began)
            if left < config.LEAST_SECONDS:
                self.say(f"[dsqa] question {self.asked}: no time left to ask the model")
                break
            if left >= 3 * config.FINAL_SECONDS:
                # Kept back: the answer comes last, and a reply cut by the
                # clock has to be asked for it.
                left -= config.FINAL_SECONDS
            context = prompt.context_of(placed, self.index.documents) if placed else ""
            try:
                reply = self.engine.generate(
                    prompt.messages(question, context), config.ANSWER_TOKENS, left
                )
                break
            except Exception as error:
                self.say(f"[dsqa] question {self.asked}: {_named(error)}")
                self._release()
                if not placed:
                    break  # the model alone failed too
                if out_of_memory(error) and tokens > config.LEAST_CONTEXT_TOKENS:
                    # Too long a prompt for the memory there is: for this
                    # question and for the ones after it.
                    tokens = max(tokens // 2, config.LEAST_CONTEXT_TOKENS)
                    self.context_tokens = tokens
                    placed, _ = self._passages(question, tokens)
                else:
                    placed = []
        self._release()
        reply = reply.strip() if isinstance(reply, str) else ""
        if reply and variant.on("plain"):
            # As the organizers' examples: what the model wrote, as it is,
            # with no line asked for apart.
            took = self.clock() - began
            shown = f"{len(placed)} chunks shown, plain"
            self.say(f"[dsqa] question {self.asked}: {shown}, {took:.1f}s")
            return reply
        if reply and prompt.split(reply) is None:
            reply = self._closed(question, context, reply, allowance - (self.clock() - began))
        took = self.clock() - began
        self.say(f"[dsqa] question {self.asked}: {len(placed)} chunks shown, {took:.1f}s")
        return prompt.as_answer(reply) if reply else self._best_passage(order)

    def _closed(self, question: str, context: str, working: str, left: float) -> str:
        """A reply that states no answer, with the answer the model gives
        when it is shown that reply and asked for the answer alone. The reply
        is kept as it is when there is no time to ask, or nothing comes of it."""
        if left < config.LEAST_SECONDS:
            self.say(f"[dsqa] question {self.asked}: no answer stated, and no time to ask for it")
            return working
        said = ""
        try:
            said = self.engine.generate(
                prompt.closing(question, context, working), config.FINAL_TOKENS, left
            )
        except Exception as error:
            self.say(f"[dsqa] question {self.asked}: {_named(error)}")
        self._release()
        said = said.strip() if isinstance(said, str) else ""
        parts = prompt.split(said)
        stated = parts[1] if parts else said
        if not stated:
            return working
        self.say(f"[dsqa] question {self.asked}: the answer was asked for apart")
        return f"{working}\n{prompt.FINAL} {stated}"

    def _release(self) -> None:
        try:
            self.engine.release()
        except Exception:
            pass


def _named(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:300]

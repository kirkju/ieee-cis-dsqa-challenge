"""Questions on IFMIF-DONES answered another way: the domain of a question
told always, and the way its answer is asked for chosen by it.

The two domains of the challenge do not answer alike. On the public
leaderboard the organizers' RAG example scores what our first entry does,
though on our questions it answers far fewer on AI papers: what its score
leaves for IFMIF-DONES is more than ours. So the questions on IFMIF-DONES
may be answered as the organizers' examples answer them, and the others as
the entry always has (variant.py names the ways):

    dones-example   the RAG example itself, its index and its prompt (example.py)
    dones-lite      the RAG example's code over the entry's own text (lite.py)
    dones-plain     the RAG example's prompt and form, over our own ranking
    dones-zero      the question alone, under the zero-shot example's prompt
    dones-nowhole   the entry's own prompt and protocol, with no document
                    placed whole in the context: chunks by their rank alone

The last one changes one thing only: in the entry's own context the
document of the best chunk goes in whole when it fits in 70% of it, and
last, next to the question; on the IFMIF-DONES questions of the official
sample that document is the wrong one four times of six. With
dones-nowhole the question is answered by the entry's own way, told only
that no document is to be placed whole (answerer.whole).

The examples bound a reply by 2,048 tokens and by nothing else, and so do
the first four ways: a question answered by them is not held to the seconds a
question of the entry may take, only to an emergency limit of what is left
of its hour once every question after it has been kept a few seconds
(emergency_seconds). On the evaluation's machine 2,048 tokens take about 73
s, so the limit is not reached in practice; it is there so that one reply
that never ends cannot take the hour from the others.

Which domain a question is about is told by code (decided), and always:
by the words of the question and by the documents ranked first for it,
each document told by its own words (domain.py). When the two say
different things, or nothing, it is the way of the other questions, the
one the entry had: a question on AI papers sent the other way would lose
the retrieval that answers them, one on IFMIF-DONES kept on this side is
answered as it was.

answered() is called first when a question is to be answered. It gives
the answer, or None when the question is not on IFMIF-DONES, when no
adjustment of this module is chosen, or when its way fails: then the
question is answered as any other. It needs of the answerer only what
every version of the entry has: its engine, its index (rank(), chunks,
documents), its budget, its clock and what it says.

The RAG example itself is example.py's (dones-example), read before the
model is loaded. It is imported only when that way is chosen, and a
missing module, an index it did not finish or an answer it does not give
leaves the question to the entry's own way.

A question answered the dones-plain way may have its ranking judged by the
base model first (relevance.py: llm-rank, llm-gate, llm-doc, llm-wide), after
its domain has been told with the ranking as it was. The log says, for every
such question, what the model did or why it did not, and the tokens of the
context that is made; whatever goes wrong there, the ranking as it was stands
and the question is still answered the dones-plain way (judged_order).

With llm-acted the entry is a measurement and not an answerer: of whether
that judgement runs where the entry is evaluated, which nothing printed there
comes back to say. A question on IFMIF-DONES is answered as it is without
llm-acted only when the base model scored every chunk it was to score and
its scores ranked (scored_in_full). Every other question gets the fixed
sentence UNANSWERED with no reply of the model asked for, and so does one
that a failure would leave to the entry's own way or to the last resort of
main.py. The score of such an entry is that of the questions on IFMIF-DONES
whose ranking was scored in full, and of nothing else.

With llm-proof the entry answers every question as it does without it, but
for one thing: a question on IFMIF-DONES gets the fixed sentence for as long
as the scoring has left nothing to rank by in every question of the run so
far (scored_some: a scoring cut for time ranks by what it scored, and
counts). The first question it ranks proves the scoring runs, and from it on
nothing is left out, not even a question whose scoring then fails: it is
answered as the entry answers it, over the ranking as it was. So the
questions on IFMIF-DONES score on the leaderboard only if the scoring runs
there, and once it does the entry is whole, as it has to be when it is the
one run again on the private leaderboard.
"""

import importlib
import inspect

import numpy as np

from . import config, domain, plain, variant
from .context import pack

# The ways an IFMIF-DONES question may be answered, as they are chosen
# (variant.py). With more than one chosen, the first of them here, and the
# log of the question says so.
WAYS = ("dones-example", "dones-lite", "dones-plain", "dones-zero", "dones-nowhole")
# Seconds kept, when a question is answered one of these ways, for every
# question after it in its hour.
RESERVE_SECONDS = 20.0
# How many of the documents ranked first for a question count when its
# domain is told, the first counting most: the n-th counts 1/n.
DOCUMENTS = 5
# What the words of a question count for against the documents: each word
# of one domain's list as much as this many first-ranked documents.
QUESTION_WEIGHT = 1.0
# A document is on IFMIF-DONES when this many of every thousand of its words
# are of that domain's list, and when it has at least DONES_OVER_AI times as
# many of them as of the AI list. On the development corpus, with the title
# in every chunk, the words of the IFMIF-DONES list take 12.7 to 96.1 of
# every thousand words of a DONES document and at most 0.11 of an AI paper;
# the DONES document with the fewest has 6.7 times as many as of the AI
# list, and the AI paper with the most, 0.004. The second rule is for AI
# papers that speak of radiologists or of a tokamak, as one on chest X-rays
# (CheXNet, 23.9 a thousand) or on the plasma of JET (10.7) does: they are
# AI papers on their own list as much.
DONES_PER_THOUSAND = 2.0
DONES_OVER_AI = 3.0
# llm-acted and llm-proof (variant.py), and what a question gets with them
# when it is not answered: neither a blank answer nor a refusal about the
# corpus.
ACTED = "llm-acted"
PROOF = "llm-proof"
UNANSWERED = "This entry does not answer this question."


def chosen_ways() -> list[str]:
    """The ways chosen for the IFMIF-DONES questions, the one taken first.
    A name is looked for among those chosen, not asked of variant.on(): a
    way whose adjustment this version does not know is not chosen."""
    picked = variant.chosen()
    return [name for name in WAYS if name in picked]


def chosen_way() -> str | None:
    """The way the IFMIF-DONES questions are answered, or None for the
    entry's own."""
    ways = chosen_ways()
    return ways[0] if ways else None


def measuring() -> bool:
    """Whether the entry runs with llm-acted. Never raises: adjustments that
    cannot be read are not llm-acted."""
    try:
        return ACTED in variant.chosen()
    except Exception:
        return False


def proving() -> bool:
    """Whether the entry runs with llm-proof, and not with llm-acted, which
    leaves out more. Never raises: adjustments that cannot be read are not
    llm-proof."""
    try:
        picked = variant.chosen()
    except Exception:
        return False
    return PROOF in picked and ACTED not in picked


def scored_in_full(report) -> str | None:
    """None when the base model scored, for a question, every chunk it was
    to score and its scores ranked: the report of relevance.judged() gives
    no reason for a ranking left as it was (scores that rank nothing are
    one), and as many were scored as were asked. Otherwise what was not so,
    for the log."""
    if report is None:
        return "relevance.py could not be asked"
    if report.why is not None:
        return report.why
    if report.asked <= 0:
        return "nothing was to be scored"
    if report.scored != report.asked:
        stopped = f", stopped for {report.stopped}" if report.stopped else ""
        return f"{report.scored} of {report.asked} scored{stopped}"
    return None


def unanswered(answerer, why: str, by: str = ACTED) -> str:
    """With llm-acted or llm-proof (`by`), the fixed sentence a question
    gets, and why in the log."""
    try:
        said = f"[dsqa] question {answerer.asked}: {by}: {why}, the fixed sentence"
        answerer.say(said[:400])
    except Exception:
        pass
    return UNANSWERED


def scored_some(report) -> str | None:
    """None when the scoring left something to rank by: the report of
    relevance.judged() gives no reason for a ranking left as it was, and at
    least one chunk was scored. A scoring cut for time ranks by what it
    scored, and counts. Otherwise why not, for the log."""
    if report is None:
        return "relevance.py could not be asked"
    if report.why is not None:
        return report.why
    if report.scored <= 0:
        return "nothing was scored"
    return None


def proved(answerer, report) -> str | None:
    """With llm-proof, for a question on IFMIF-DONES whose ranking was just
    judged: None when it is to be answered as the entry answers it, with
    what was so in the log, or why not, for the fixed sentence. A ranking
    the scoring left something to rank by (scored_some), even cut for time,
    proves the scoring for the rest of the run: that is kept on the
    answerer, which lasts as long as the run."""
    short = scored_some(report)
    before = bool(getattr(answerer, "scoring_proved", False))
    if short is None:
        answerer.scoring_proved = True
        how = "scored in full"
        if report.scored != report.asked:
            stopped = f", stopped for {report.stopped}" if report.stopped else ""
            how = f"scored {report.scored} of {report.asked}{stopped}"
        said = f"{how}, answered" if before else f"{how}, the first: answered"
    elif before:
        said = f"not scored ({short}), but proved before: answered"
    else:
        return f"not scored ({short}), and never yet"
    try:
        answerer.say(f"[dsqa] question {answerer.asked}: {PROOF}: {said}"[:400])
    except Exception:
        pass
    return None


def emergency_seconds(budget, before: int) -> float:
    """The most a reply of these ways may take, when `before` questions came
    before it: what is left of its hour, less RESERVE_SECONDS for every
    question after it in that hour."""
    hour, earlier = divmod(before, config.QUESTIONS_AN_HOUR)
    left = config.HOUR_SECONDS * (hour + 1) - config.MARGIN_SECONDS - budget.elapsed()
    return left - (config.QUESTIONS_AN_HOUR - earlier - 1) * RESERVE_SECONDS


def document_domains(index) -> np.ndarray:
    """For every document of an index, whether it is on IFMIF-DONES, told by
    the words of all its chunks. Worked out once an index, and kept on it."""
    kept = getattr(index, "_route_dones", None)
    if kept is not None:
        return kept
    documents = np.asarray(index.documents)
    count = int(documents.max()) + 1 if len(documents) else 0
    words = np.zeros(count)
    dones = np.zeros(count)
    ai = np.zeros(count)
    for chunk, number in zip(index.chunks, documents, strict=True):
        on_dones, on_ai = domain.counted(chunk)
        dones[number] += on_dones
        ai[number] += on_ai
        words[number] += len(chunk.split())
    dense = dones * 1000 >= DONES_PER_THOUSAND * np.maximum(words, 1)
    kept = dense & (dones >= DONES_OVER_AI * ai)
    index._route_dones = kept
    return kept


def prepared(index, say=print) -> None:
    """The domains of an index's documents worked out now, when a way of
    this module is chosen, so that the first question does not take the
    time of it; and with dones-lite, the example's chunks of the entry's
    texts (lite.py). Never raises."""
    try:
        ways = chosen_ways() if index is not None and len(index.chunks) else []
        if ways:
            document_domains(index)
        if "dones-lite" in ways and getattr(index, "texts", None):
            importlib.import_module(f"{__package__}.lite").prepare(index.texts, say)
    except Exception as error:
        say(f"[dsqa] the ways of route.py could not be prepared: {error!r}"[:300])


def first_documents(order, documents, many: int = DOCUMENTS) -> list[int]:
    """The documents of a ranking of chunks, by their best chunk, the first
    `many` of them."""
    seen: list[int] = []
    for number in order:
        document = int(documents[number])
        if document not in seen:
            seen.append(document)
            if len(seen) == many:
                break
    return seen


def decided(question: str, index=None, order=None) -> str:
    """The domain of a question, always: DONES or AI. Each word of the
    question in a domain's list counts QUESTION_WEIGHT for it; the first
    DOCUMENTS documents ranked for it count 1, 1/2, 1/3... for the domain
    each is on. A question for which the two weigh the same is AI, the side
    of the entry's own way, and so is one whose first document is not on
    IFMIF-DONES, whatever the rest say."""
    on_dones, on_ai = domain.counted(question)
    score = QUESTION_WEIGHT * (on_dones - on_ai)
    if index is not None and len(index.chunks):
        order = index.rank(question) if order is None else order
        dones = document_domains(index)
        first = first_documents(order, index.documents)
        if first and not dones[first[0]]:
            return domain.AI
        for place, number in enumerate(first, 1):
            score += (1 if dones[number] else -1) / place
    return domain.DONES if score > 0 else domain.AI


def by_example(answerer, question: str, seconds: float) -> str | None:
    """A question answered by the RAG example itself (example.py), when
    that module is there and its index was made; None otherwise. The
    seconds are passed when its answer() takes them."""
    try:
        example = importlib.import_module(f"{__package__}.example")
    except ImportError:
        return None
    if not example.ready():
        return None
    corpus_dir = getattr(answerer, "corpus_dir", None)
    model_id = getattr(answerer, "model_id", None)
    extra = {}
    if "seconds" in inspect.signature(example.answer).parameters:
        extra["seconds"] = seconds
    return example.answer(question, corpus_dir, model_id, answerer.engine, **extra)


def by_lite(answerer, question: str, seconds: float) -> str | None:
    """A question answered by the RAG example's code over the entry's own
    text (lite.py), when that module is there and its chunks were made;
    None otherwise."""
    try:
        lite = importlib.import_module(f"{__package__}.lite")
    except ImportError:
        return None
    if not lite.ready():
        return None
    return lite.answer(question, answerer.engine, seconds)


def judged_order(answerer, question: str, index, order, seconds: float):
    """The ranking a dones-plain question takes its fragments from: the one
    given, or the one relevance.py makes of it with the base model when one
    of its adjustments is chosen; and the report of what the model did
    (relevance.Judged), None when relevance.py could not be asked. The log
    says what the model did or why it did not, and the tokens and fragments
    of the context that ranking makes. Never raises: whatever fails, the
    ranking given stands."""
    judged, report, said = order, None, "not applied (relevance.py could not be asked)"
    try:
        from . import relevance

        judged, report = relevance.judged(
            answerer.engine, question, index, order, seconds, clock=answerer.clock
        )
        said = report.line()
    except Exception as error:
        judged, report = order, None
        said = f"not applied ({type(error).__name__}: {error})"
    try:
        tokens, fragments = context_tokens(answerer.engine, index, judged)
        said += f", context of {tokens} tokens in {fragments} fragments"
    except Exception:
        pass
    try:
        answerer.say(f"[dsqa] question {answerer.asked}: relevance: {said}"[:400])
    except Exception:
        pass
    return judged, report


def context_tokens(engine, index, order) -> tuple[int, int]:
    """The tokens and the fragments of the context the dones-plain way makes
    of a ranking, counted as plain.over_ranking counts them, with its cache."""
    if index is None or order is None:
        return 0, 0
    counts = index.__dict__.setdefault("_plain_tokens", {})

    def count(text: str) -> int:
        if text not in counts:
            counts[text] = engine.count(text)
        return counts[text]

    numbers = [int(number) for number in order]
    placed, used = pack(numbers, index.chunks, count, plain.MAX_CONTEXT_TOKENS)
    return int(used), len(placed)


def answered(answerer, question: str, began: float, allowance: float) -> str | None:
    """The answer to a question by the way chosen for IFMIF-DONES, when it
    is on IFMIF-DONES; None when it is to be answered as any other. It never
    raises, and leaves the time it did not use to the entry's own way.
    `allowance` is what the entry's own way gives the question; these ways
    are held to emergency_seconds() instead.

    With llm-acted it gives a string always and leaves nothing to the
    entry's own way: the dones-plain reply to a question on IFMIF-DONES
    whose ranking was scored in full, and UNANSWERED to every other
    question, with why in the log (unanswered)."""
    way = None
    # What the entry's own way is told of this question: nothing, unless
    # dones-nowhole says it below.
    answerer.whole = None
    acted, proof = measuring(), proving()
    try:
        ways = chosen_ways()
        if not ways or answerer.engine is None:
            if acted:
                why = "no model" if answerer.engine is None else "no way chosen for IFMIF-DONES"
                return unanswered(answerer, why)
            return None
        way = ways[0]
        if acted and way != "dones-plain":
            return unanswered(answerer, f"{way} chosen, and only dones-plain is measured")
        index = answerer.index
        order = index.rank(question) if index is not None and len(index.chunks) else None
        if decided(question, index, order) != domain.DONES:
            if acted:
                return unanswered(answerer, "not told on IFMIF-DONES, the model not asked")
            return None
        if len(ways) > 1:
            answerer.say(
                f"[dsqa] question {answerer.asked}: {way} chosen over {', '.join(ways[1:])}"
            )
        if way == "dones-nowhole":
            # The entry's own way answers, with no document placed whole.
            answerer.whole = 0.0
            answerer.say(f"[dsqa] question {answerer.asked}: the dones-nowhole way, none whole")
            return None
        # What is left of the hour is counted from when our code started:
        # the time this question has taken is in it already.
        seconds = emergency_seconds(answerer.budget, answerer.asked - 1)
        if way == "dones-example":
            reply = by_example(answerer, question, seconds)
        elif way == "dones-lite":
            reply = by_lite(answerer, question, seconds)
        elif way == "dones-zero":
            reply = plain.zero_shot(answerer.engine, question, seconds)
        else:
            judged, report = judged_order(answerer, question, index, order, seconds)
            if acted:
                short = scored_in_full(report)
                if short is not None:
                    return unanswered(answerer, f"not scored in full ({short})")
            elif proof:
                why = proved(answerer, report)
                if why is not None:
                    return unanswered(answerer, why, PROOF)
            if judged is not order:
                # The scoring took some of what is left of the hour.
                seconds = emergency_seconds(answerer.budget, answerer.asked - 1)
            reply = plain.over_ranking(answerer.engine, question, index, judged, seconds)
    except Exception as error:
        answerer.say(f"[dsqa] question {answerer.asked}: the {way} way failed: {error!r}"[:300])
        if acted:
            return unanswered(answerer, f"the {way} way failed ({type(error).__name__})")
        return None
    took = answerer.clock() - began
    if not isinstance(reply, str) or not reply.strip():
        answerer.say(f"[dsqa] question {answerer.asked}: {way} gave no answer, {took:.1f}s")
        if acted:
            return unanswered(answerer, "scored in full, and the dones-plain way gave no answer")
        return None
    answerer.say(f"[dsqa] question {answerer.asked}: answered the {way} way, {took:.1f}s")
    if acted:
        answerer.say(f"[dsqa] question {answerer.asked}: llm-acted: scored in full, answered")
    return reply.strip()

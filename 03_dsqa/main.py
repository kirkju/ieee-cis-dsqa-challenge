"""Our entry for the IEEE-CIS Technical Challenge on Domain-Specific Question
Answering with LLMs.

The evaluation imports this file and calls answer() once for every question,
all in one process. On the first call the corpus is read and indexed and the
base model is loaded; both are kept for the calls that follow.

    answer()    this file: what is kept between calls, and what is returned
                when everything else fails
    dsqa/       the corpus read, chunked and ranked; the model asked

answer() returns a string whatever happens. If the package next to this file
cannot even be imported, the model is loaded here and asked without passages,
as the organizers' zero-shot example does. With llm-acted (dsqa/route.py),
the entry as a measurement, a question that dsqa does not answer gets that
measurement's fixed sentence instead, and the model is not asked.
"""

import time

# The time limit of a run counts from when our code starts running: from here.
_STARTED = time.monotonic()

NO_ANSWER = "No answer could be produced for this question."
# The last resort asks the model as the organizers' zero-shot example does,
# with an answer bounded in tokens and in seconds.
SYSTEM = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)
LAST_RESORT_TOKENS, LAST_RESORT_SECONDS = 1_024, 50.0

# What is loaded on the first call and kept: the engine that generates and
# the corpus behind it. A part that failed to load is not tried again, since
# loading the model twice would cost minutes of a run that has an hour.
_kept = {}


def _say(what: str, error: BaseException) -> None:
    print(f"[dsqa] {what}: {type(error).__name__}: {str(error)[:300]}")


class _Model:
    """The base model with nothing of ours around it: the last resort, for
    when dsqa cannot be imported. It answers as dsqa's engine does."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()

    def generate(self, messages: list, max_new_tokens: int, max_seconds: float | None = None):
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        limits = {} if max_seconds is None else {"max_time": max_seconds}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                **limits,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()


def _engine(model_id: str):
    """What generates, loaded once: dsqa's engine, or the bare model when
    dsqa is not there. None when the model could not be loaded."""
    if "engine" not in _kept:
        _kept["engine"] = None
        try:
            from dsqa import engine
        except Exception as error:
            _say("dsqa's engine could not be imported", error)
            engine = None
        try:
            _kept["engine"] = _Model(model_id) if engine is None else engine.load(model_id)
        except Exception as error:
            _say("the model could not be loaded", error)
    return _kept["engine"]


def _example_first(corpus_dir: str, model_id: str) -> None:
    """With the adjustment dones-example, the organizers' RAG example reads
    the corpus with its own code before the base model is loaded, as the
    example itself does (dsqa/example.py). Without it, nothing."""
    try:
        from dsqa import example, variant

        if "dones-example" in variant.chosen():
            example.prepare(corpus_dir, started=_STARTED, model_id=model_id)
    except Exception as error:
        _say("the example's index could not be made", error)


def _answerer(corpus_dir: str, model_id: str):
    """dsqa's answerer over the corpus, built once. None when it cannot be."""
    if "answerer" not in _kept:
        _kept["answerer"] = None
        _example_first(corpus_dir, model_id)
        engine = _engine(model_id)
        try:
            from dsqa.answering import Answerer

            _kept["answerer"] = Answerer(corpus_dir, engine, started=_STARTED)
        except Exception as error:
            _say("the corpus could not be set up", error)
    return _kept["answerer"]


def _unanswered():
    """With llm-acted (dsqa/route.py), the fixed sentence a question gets
    when it is not answered as that measurement says: never the last
    resort's reply. None without it, or when dsqa cannot tell."""
    try:
        from dsqa import route

        return route.UNANSWERED if route.measuring() else None
    except Exception:
        return None


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    question = question if isinstance(question, str) else str(question)
    try:
        answerer = _answerer(corpus_dir, model_id)
        if answerer is not None:
            reply = answerer.answer(question)
            if isinstance(reply, str) and reply.strip():
                return reply
    except Exception as error:
        _say("the question could not be answered with passages", error)
    unanswered = _unanswered()
    if unanswered is not None:
        return unanswered
    try:
        engine = _engine(model_id)
        if engine is not None:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]
            reply = engine.generate(messages, LAST_RESORT_TOKENS, LAST_RESORT_SECONDS)
            if isinstance(reply, str) and reply.strip():
                return reply.strip()
    except Exception as error:
        _say("the question could not be answered at all", error)
    return NO_ANSWER

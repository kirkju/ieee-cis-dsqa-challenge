"""A stand-in for the base model in the contract tests.

The entry loads it in place of the model when DSQA_ENGINE names this file
(dsqa/engine.py). It answers at once, on the line the entry asks a model to
state its answer on, and says whether it was shown passages. It can be told
to fail in the ways a model on a GPU fails, through the words in the
environment variable DSQA_FAKE:

    load_fails     the model cannot be loaded
    oom_once       the first prompt with passages runs the GPU out of memory
    always_raises  every generation raises
    blank          the reply is an empty string
    not_a_string   the reply is not text
    no_final       the reply does not state its answer on that line

It stands alone: nothing of the repository is importable while a submission
runs.
"""

import os


def _told(what: str) -> bool:
    return what in os.environ.get("DSQA_FAKE", "").split(",")


class Fake:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.generations = 0

    def count(self, text: str) -> int:
        return len(text.split())

    def generate(self, messages: list, max_new_tokens: int, max_seconds: float | None = None):
        self.generations += 1
        asked = messages[-1]["content"]
        shown = "Passages retrieved" in asked
        if _told("always_raises"):
            raise ValueError("the fake model was told to fail")
        if _told("oom_once") and shown and self.generations == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if _told("blank"):
            return "  "
        if _told("not_a_string"):
            return 42
        question = asked.rsplit("Question: ", 1)[-1]
        with_what = "with passages" if shown else "without passages"
        reply = f"An answer {with_what} from {self.model_id} to: {question[:60]}"
        return reply if _told("no_final") else f"Final answer: {reply}"

    def release(self) -> None:
        pass


def load(model_id: str) -> Fake:
    print("[fake] the model is loaded")
    if _told("load_fails"):
        raise OSError("the fake model was told not to load")
    return Fake(model_id)

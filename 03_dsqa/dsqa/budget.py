"""How long a question may take.

A run that passes its limit is stopped and counts as failed, with every
answer lost; a short answer costs one question at most. So each question
gets its share of the time that is left, and when there is none the model
is not asked at all.

answer() is not told how many questions there are, nor which leaderboard it
runs for. The rules give 50 questions an hour either way, so the hours are
taken one at a time: the first 50 questions share the first hour, the next
50 the second.
"""

import time
from collections.abc import Callable

from . import config


class Budget:
    """The clock of a run. `started` is when our code began to run, on
    `clock`: the limit counts from there, so the time spent loading the model
    and reading the corpus is time the questions do not have."""

    def __init__(self, started: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.started = clock() if started is None else started

    def elapsed(self) -> float:
        return self.clock() - self.started

    def allowance(self, asked: int) -> float:
        """The seconds the next question may take, when `asked` questions
        came before it: an even share of what is left of its hour, and never
        more than a question may take. Not above zero when the hour is up."""
        hour, before = divmod(asked, config.QUESTIONS_AN_HOUR)
        left = config.HOUR_SECONDS * (hour + 1) - config.MARGIN_SECONDS - self.elapsed()
        share = left / (config.QUESTIONS_AN_HOUR - before)
        return min(float(config.QUESTION_SECONDS), share)

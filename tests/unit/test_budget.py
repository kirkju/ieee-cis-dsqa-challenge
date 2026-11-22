"""dsqa/budget.py: how long a question may take. The clock is the test's."""

import pytest
from dsqa import config
from dsqa.budget import Budget


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_a_question_gets_an_even_share_of_what_is_left_of_its_hour():
    clock = Clock()
    budget = Budget(clock=clock)  # started now
    # With the whole hour ahead a share is more than a question may take.
    assert budget.allowance(0) == config.QUESTION_SECONDS == 50
    # Half an hour went into loading the model and reading the corpus.
    clock.now += 1800
    assert budget.elapsed() == 1800
    assert budget.allowance(0) == pytest.approx((3600 - 360 - 1800) / 50)
    assert budget.allowance(10) == pytest.approx((3600 - 360 - 1800) / 40)
    # The last question of the hour has all that is left, up to the most one may take.
    clock.now += 1400
    assert budget.allowance(49) == pytest.approx(40.0)
    clock.now -= 100
    assert budget.allowance(49) == 50


def test_when_the_hour_is_up_there_is_nothing_left():
    clock = Clock()
    budget = Budget(clock=clock)
    clock.now += 3600 - 360
    assert budget.allowance(30) == 0
    clock.now += 60
    assert budget.allowance(30) < 0


def test_the_next_fifty_questions_have_the_next_hour():
    clock = Clock()
    budget = Budget(clock=clock)
    clock.now += 3500  # the first fifty took nearly all of their hour
    assert budget.allowance(49) < 0
    assert budget.allowance(50) == 50
    clock.now += 2000
    assert budget.allowance(60) == pytest.approx((7200 - 360 - 5500) / 40)
    assert budget.allowance(100) == 50  # and a third hour, were there one


def test_the_time_counts_from_when_the_code_started_running():
    clock = Clock(5000.0)
    budget = Budget(started=4000.0, clock=clock)  # the model took a while to load
    assert budget.elapsed() == 1000
    assert budget.allowance(0) == pytest.approx((3600 - 360 - 1000) / 50)

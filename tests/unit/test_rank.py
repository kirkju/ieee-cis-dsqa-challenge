"""dsqa/rank.py: chunks ranked by the words they share with a question, and
rankings fused."""

import numpy as np
from dsqa.rank import Bm25, fuse, words


def test_words_are_matched_whatever_their_case_and_whatever_sticks_to_them():
    assert words("The Be7 half-life (T1/2) is 53.22 d.") == [
        "the", "be", "7", "half", "life", "t", "1", "2", "is", "53.22", "d",
    ]  # fmt: skip
    # A nuclide written either way gives the same two words.
    assert sorted(words("7 Be")) == sorted(words("Be7"))
    assert words("10 \N{MICRO SIGN}Sv/h, \N{GREEK SMALL LETTER ALPHA}=0.5") == [
        "10", "\N{GREEK SMALL LETTER MU}sv", "h", "\N{GREEK SMALL LETTER ALPHA}", "0.5",
    ]  # fmt: skip


TEXTS = [
    "the lithium loop carries the lithium to the target",
    "the nitrogen trap is in the dump tank",
    "the loop and the tank and the trap and the target and the pump and the pipes of the plant",
    "nothing of interest",
]


def test_bm25_puts_first_the_text_with_the_rare_word():
    index = Bm25(TEXTS)
    assert np.argmax(index.scores("where is the nitrogen trap")) == 1
    assert np.argmax(index.scores("lithium")) == 0
    # A word every text has says little; one that no text has says nothing.
    scores = index.scores("the")
    assert scores[3] == 0 and 0 < scores.max() < index.scores("nitrogen").max()
    assert not index.scores("accelerator").any()
    assert not index.scores("").any()


def test_bm25_counts_a_word_less_each_time_and_less_in_a_long_text():
    index = Bm25(["trap", "trap trap trap trap", "trap " + "pipe " * 30, "tank", "pump"])
    once, four_times, in_a_long_text = index.scores("trap")[:3]
    assert once < four_times < 4 * once
    assert in_a_long_text < once
    # Asking for a word twice is asking for it once.
    assert index.scores("trap trap").tolist() == index.scores("trap").tolist()


def test_bm25_of_no_texts_scores_nothing():
    assert Bm25([]).scores("trap").tolist() == []


def test_fusing_two_rankings_favours_what_both_rank_high():
    first = np.array([0, 1, 2, 3])
    second = np.array([3, 1, 2, 0])
    # 1 is second in both; 0 and 3 are first in one and last in the other.
    assert fuse([first, second]).tolist() == [1, 0, 3, 2]
    assert fuse([first, first]).tolist() == first.tolist()

"""dsqa/context.py: the chunks that go into the context of a question."""

import random

import pytest
from dsqa.context import pack

from tests.unit.stand_ins import tokens
from tools import retrieval_lab

CHUNKS = ["aa bb cc", "dd ee ff gg hh", "ii", "jj kk"]
# Three documents: of 6, 3 and 4 tokens.
SHELF = ["a1 a1", "a2 a2 a2", "a3", "b1 b1", "b2", "c1 c1 c1 c1"]
OF = [0, 0, 0, 1, 1, 2]


def placed(context):
    return [number for number, _ in context]


def test_chunks_are_taken_from_the_top_of_the_ranking_until_one_does_not_fit():
    order = [0, 1, 2, 3]
    assert pack(order, CHUNKS, tokens, 6) == ([(0, CHUNKS[0])], 3)
    assert pack(order, CHUNKS, tokens, 100) == ([(n, CHUNKS[n]) for n in order], 11)
    assert pack([], CHUNKS, tokens, 100) == ([], 0)


def test_a_first_chunk_that_does_not_fit_gives_a_slice_of_itself():
    # The share of its characters that the budget is of its tokens.
    assert pack([1, 0], CHUNKS, tokens, 2) == ([(1, CHUNKS[1][: int(14 * 2 / 5)])], 0)


def test_the_document_of_the_best_chunk_goes_in_whole_when_it_fits_its_share():
    order = [4, 0, 5, 1, 2, 3]
    # The second document is 3 tokens of a budget of 8: it goes first, in its
    # own order, and the ranking follows until a chunk does not fit.
    context, used = pack(order, SHELF, tokens, 8, OF, 0.5)
    assert (placed(context), used) == ([3, 4, 0], 5)
    assert context[0] == (3, SHELF[3])
    # Nothing says which document a chunk is of: the ranking alone.
    assert pack(order, SHELF, tokens, 8, whole=0.5) == pack(order, SHELF, tokens, 8)


def test_a_document_too_long_for_its_share_does_not_go_in_whole():
    order = [1, 4, 5, 0, 2, 3]
    plain = pack(order, SHELF, tokens, 10)
    assert placed(plain[0]) == [1, 4, 5, 0]
    # The first document is 6 tokens, more than half of 10.
    assert pack(order, SHELF, tokens, 10, OF, 0.5) == plain
    assert placed(pack(order, SHELF, tokens, 10, OF, 0.6)[0]) == [0, 1, 2, 4]


@pytest.mark.parametrize("seed", range(20))
def test_it_packs_what_the_lab_s_packer_packs_with_the_same_settings(seed):
    # The lab's packer has two ways more of filling a context, which were
    # tried and left there. With neither, the two have to agree.
    chance = random.Random(seed)
    chunks = [" ".join(["w"] * chance.randint(1, 9)) for _ in range(30)]
    documents = sorted(chance.randrange(6) for _ in chunks)
    order = chance.sample(range(len(chunks)), len(chunks))
    budget, whole = chance.randint(1, 60), chance.choice([0.0, 0.3, 0.7, 1.0])
    assert pack(order, chunks, tokens, budget, documents, whole) == retrieval_lab.pack(
        order, chunks, tokens, budget, documents=documents, whole=whole
    )

"""engine.first() against engine.score() with a real model small enough for a
laptop: real chunks of two IFMIF-DONES papers of the development corpus,
scored for a question in batches padded on the left and one at a time
through generate(). They rank alike but for near ties, and their logits
differ by no more than the rounding of bf16: llm-one gives the scores
llm-rank has used.

Left out of the default run, as the other slow tests are: it needs a GPU,
the small model in the local Hugging Face cache and the development corpus.

    python -m pytest -m slow tests/slow/test_engine_first.py
"""

import math

import pytest
from dsqa import engine, relevance, retrieval

from tests.slow.test_examples import (  # noqa: F401  (the fixture is used by being here)
    STAND_IN,
    stand_in_model_is_usable,
)
from tools import corpus

pytestmark = pytest.mark.slow

PAPERS = ["dones-lithium-systems-ifmif-dones", "dones-lithium-loop-purification-system"]
QUESTION = "What is the bypass ratio of the ICS Branch of the DONES Li loop?"
# Two scores closer than this are a tie: the paths may order them either way.
TIE = 0.05


@pytest.fixture
def two_papers(tmp_path):
    papers = [corpus.CORPUS_DIR / f"{name}.pdf" for name in PAPERS]
    missing = [paper.name for paper in papers if not paper.exists()]
    if missing:
        pytest.skip("not there: " + ", ".join(missing))
    for paper in papers:
        (tmp_path / paper.name).write_bytes(paper.read_bytes())
    return tmp_path


def p_yes(rows):
    return [1 / (1 + math.exp(-(yes - no))) for yes, no in rows]


def test_one_at_a_time_ranks_as_the_padded_batches_do(two_papers):
    model = engine.Transformers(STAND_IN)
    index = retrieval.build(two_papers, None, lambda line: None, 300)
    conversations = []
    for number in index.rank(QUESTION)[:40]:
        title, body = relevance.shown(index, int(number))
        body = relevance.cut(body, 600, model.count)
        conversations.append(relevance.conversation(QUESTION, title, body))
    batched = model.score(conversations, ("Yes", "No"), 8000)
    alone = [model.first(conversation, ("Yes", "No")) for conversation in conversations]
    model.release()
    assert len(batched) == len(alone) == 40
    gaps = [
        abs(a - b)
        for one, other in zip(batched, alone, strict=True)
        for a, b in zip(one, other, strict=True)
    ]
    assert max(gaps) < 1.0  # the rounding of bf16, not another reading
    together, apart = p_yes(batched), p_yes(alone)
    flipped = [
        (i, j)
        for i in range(40)
        for j in range(i + 1, 40)
        if abs(apart[i] - apart[j]) > TIE
        and abs(together[i] - together[j]) > TIE
        and (apart[i] > apart[j]) != (together[i] > together[j])
    ]
    assert flipped == []

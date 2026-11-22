"""dsqa/engine.py, score(): the labels' logits at the first token of a reply,
read in batches padded on the left. The model is a stand-in whose logits
depend on each token and on its position, so a batch gives what one prompt
alone gives only if the padding and the positions are right."""

import types

import pytest
import torch
from dsqa import engine

VOCABULARY = 50
YES, NO, SPACED_YES = 7, 8, 9


class Tokenizer:
    """Words as ids: "Yes" 7, "No" 8, " Yes" 9, any other word by its length."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=True):
        ids = [2]
        for turn in conversation:
            ids += [3] + [10 + len(word) % 30 for word in turn["content"].split()] + [4]
        return {"input_ids": ids + [5]}

    def encode(self, text, add_special_tokens=False):
        return {"Yes": [YES], "No": [NO], " Yes": [SPACED_YES], "yes": [YES]}.get(text, [11])


class Model:
    """Logits at the last position that depend on every real token and its
    position: wrong padding or wrong positions change them. Runs out of
    memory for batches larger than `room` rows."""

    device = torch.device("cpu")

    def __init__(self, room=100):
        self.room = room
        self.calls: list[dict] = []

    def __call__(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        self.calls.append({"rows": len(input_ids), "use_cache": use_cache, "keep": logits_to_keep})
        if len(input_ids) > self.room:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        real = attention_mask.float()
        weight = (input_ids.float() * (position_ids.float() + 1) * real).sum(-1)
        logits = torch.zeros(len(input_ids), 1, VOCABULARY)
        logits[:, 0, YES] = weight % 97
        logits[:, 0, SPACED_YES] = weight % 89
        logits[:, 0, NO] = real.sum(-1)
        return types.SimpleNamespace(logits=logits)


def transformers_with(model):
    made = object.__new__(engine.Transformers)
    made.torch, made.tokenizer, made.model = torch, Tokenizer(), model
    made.release = lambda: None
    return made


CONVERSATIONS = [
    [{"role": "user", "content": "a short one"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": "a much longer prompt here"}],
    [{"role": "user", "content": "mid sized one"}],
]


def alone(conversation):
    return transformers_with(Model()).score([conversation], ("Yes", "No"), most_tokens=10_000)[0]


def test_a_batch_padded_on_the_left_gives_what_each_prompt_gives_alone():
    model = Model()
    together = transformers_with(model).score(CONVERSATIONS, ("Yes", "No"), most_tokens=10_000)
    assert together == [alone(conversation) for conversation in CONVERSATIONS]
    assert model.calls == [{"rows": 3, "use_cache": False, "keep": 1}]


def test_a_label_is_the_most_likely_of_its_spellings():
    ((yes, no),) = transformers_with(Model()).score(CONVERSATIONS[:1])
    tokens = Tokenizer().apply_chat_template(CONVERSATIONS[0])["input_ids"]
    weight = sum(token * (position + 1) for position, token in enumerate(tokens))
    assert yes == max(weight % 97, weight % 89) and no == len(tokens)


def test_batches_are_held_to_their_tokens_and_the_order_is_kept():
    model = Model()
    found = transformers_with(model).score(CONVERSATIONS, ("Yes", "No"), most_tokens=12)
    assert found == [alone(conversation) for conversation in CONVERSATIONS]
    assert len(model.calls) == 3  # each prompt is longer than half of 12


def test_a_batch_out_of_memory_is_tried_in_halves_and_then_given_up():
    model = Model(room=2)
    found = transformers_with(model).score(CONVERSATIONS, ("Yes", "No"), most_tokens=10_000)
    assert found == [alone(conversation) for conversation in CONVERSATIONS]
    assert [call["rows"] for call in model.calls] == [3, 1, 2]
    assert transformers_with(Model(room=0)).score(CONVERSATIONS) is None


def test_nothing_to_score_is_nothing_scored():
    assert transformers_with(Model()).score([]) == []


def test_an_error_that_is_not_of_memory_is_not_swallowed():
    class Broken(Model):
        def __call__(self, **_):
            raise ValueError("not a memory error")

    with pytest.raises(ValueError):
        transformers_with(Broken()).score(CONVERSATIONS[:1])


def test_the_batches_are_cut_by_their_padded_size():
    assert engine._batches([0, 1, 2], [3, 3, 3], 9) == [[0, 1, 2]]
    assert engine._batches([0, 1, 2], [3, 3, 4], 9) == [[0, 1], [2]]
    assert engine._batches([0], [50], 9) == [[0]]

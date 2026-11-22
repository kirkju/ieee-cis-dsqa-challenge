"""dsqa/engine.py, first(): the labels' logits at the first token of a reply
to one conversation, read through generate() itself, as every reply of the
entry is read: one conversation, nothing padded, one token written, its
logits as they came. The model is a stand-in that records how generate() was
asked."""

import types

import pytest
import torch
from dsqa import engine

VOCABULARY = 50
YES, NO, SPACED_YES = 7, 8, 9


class Inputs(dict):
    def to(self, device):
        self["moved_to"] = device
        return self


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, conversation, **options):
        ids = [2] + [10 + len(turn["content"]) % 30 for turn in conversation] + [5]
        return Inputs(input_ids=torch.tensor([ids]), options=options)

    def encode(self, text, add_special_tokens=False):
        nos = dict.fromkeys(("No", "no", " No", " no"), [NO])
        return {"Yes": [YES], " Yes": [SPACED_YES], "yes": [YES], **nos}.get(text, [11])


class Model:
    """Writes one token whose raw logits it returns, as generate() does with
    output_logits; runs out of memory when told."""

    device = torch.device("cpu")

    def __init__(self, fails=None):
        self.fails, self.asked = fails, []

    def generate(self, **options):
        self.asked.append(options)
        if self.fails is not None:
            raise self.fails
        logits = torch.zeros(1, VOCABULARY)
        logits[0, YES], logits[0, SPACED_YES], logits[0, NO] = 2.5, 3.5, -1.0
        return types.SimpleNamespace(logits=(logits,), sequences=None)


def transformers_with(model):
    made = object.__new__(engine.Transformers)
    made.torch, made.tokenizer, made.model = torch, Tokenizer(), model
    made.released = 0

    def release():
        made.released += 1

    made.release = release
    return made


CONVERSATION = [{"role": "system", "content": "s"}, {"role": "user", "content": "a question"}]


def test_the_first_token_is_read_as_a_reply_is_read():
    model = Model()
    assert transformers_with(model).first(CONVERSATION) == [3.5, -1.0]
    ((options,),) = [model.asked]
    # The prompt of the chat template, alone, moved to the model's device.
    assert options["input_ids"].shape == (1, 4) and options["moved_to"] == model.device
    assert options["options"] == {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    # One token, greedy, its logits as they came.
    assert (options["max_new_tokens"], options["do_sample"]) == (1, False)
    assert options["output_logits"] and options["return_dict_in_generate"]
    assert options["pad_token_id"] == 0


def test_any_labels_are_read_by_their_spellings():
    assert transformers_with(Model()).first(CONVERSATION, ("No", "Yes")) == [-1.0, 3.5]


def test_out_of_memory_gives_none_and_the_gpu_back():
    made = transformers_with(Model(fails=RuntimeError("CUDA out of memory. Tried 2 GiB")))
    assert made.first(CONVERSATION) is None and made.released == 1


def test_an_error_that_is_not_of_memory_is_not_swallowed():
    with pytest.raises(ValueError):
        transformers_with(Model(fails=ValueError("a template it could not read"))).first(
            CONVERSATION
        )

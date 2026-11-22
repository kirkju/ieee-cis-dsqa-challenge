"""The prompts of relevance.py through the base model's own chat template:
its tokenizer, from the local Hugging Face cache; the model is not loaded.
Skipped where the tokenizer has not been downloaded."""

import pytest
from dsqa import engine, relevance

BASE = "allenai/Olmo-3.1-32B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    except OSError:
        pytest.skip(f"the tokenizer of {BASE} is not in the local cache")


def scorer(tokenizer):
    made = object.__new__(engine.Transformers)
    made.tokenizer = tokenizer
    return made


def test_a_scoring_prompt_ends_where_the_reply_begins(tokenizer):
    asked = relevance.conversation("What flows?", "Lithium loop", "The loop carries lithium.")
    text = tokenizer.apply_chat_template(asked, add_generation_prompt=True, tokenize=False)
    assert text.startswith("<|im_start|>system\n" + relevance.SYSTEM + "<|im_end|>\n")
    assert text.endswith("Answer Yes or No.<|im_end|>\n<|im_start|>assistant\n")
    # Our own system message: the template puts no system of its own.
    assert "You do not currently have access to any functions" not in text


def test_score_reads_the_ids_the_template_gives_as_its_text_would(tokenizer):
    asked = relevance.conversation("What flows?", "Lithium loop", "The loop carries lithium.")
    ids = scorer(tokenizer)._prompt_ids(asked)
    text = tokenizer.apply_chat_template(asked, add_generation_prompt=True, tokenize=False)
    assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]


def test_the_labels_are_tokens_of_their_own(tokenizer):
    made = scorer(tokenizer)
    labels = {label: made._label_ids(label) for label in ("Yes", "No", "A", "B", "C")}
    # "Yes" and "No" each by their spellings with and without a space and a capital.
    assert len(labels["Yes"]) == 4 and len(labels["No"]) == 4
    every = [token for ids in labels.values() for token in ids]
    assert len(every) == len(set(every))
    for label, ids in labels.items():
        for token in ids:
            assert tokenizer.decode([token]).strip().lower() == label.lower()

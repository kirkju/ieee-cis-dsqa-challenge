"""dsqa/engine.py: what generates, behind one boundary.

The base model itself is loaded in the slow tests. Here the boundary is
tried with a stand-in for transformers, and the seam that lets a stand-in
engine take the model's place on a machine it does not fit.
"""

import sys
import types

import pytest
from dsqa import engine

STAND_IN = '''
class Echo:
    def __init__(self, model_id):
        self.model_id = model_id

    def count(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, max_seconds=None):
        return "echo: " + messages[-1]["content"]

    def release(self):
        pass


def load(model_id):
    return Echo(model_id)
'''


def test_a_file_named_by_the_environment_gives_the_engine(tmp_path, monkeypatch):
    path = tmp_path / "stand_in.py"
    path.write_text(STAND_IN, encoding="utf-8")
    monkeypatch.setenv("DSQA_ENGINE", str(path))
    loaded = engine.load("some/model")
    assert loaded.model_id == "some/model"
    assert loaded.generate([{"role": "user", "content": "hello"}], 8) == "echo: hello"


def test_running_out_of_gpu_memory_is_told_from_other_errors():
    class OutOfMemoryError(RuntimeError):
        pass

    assert engine.out_of_memory(OutOfMemoryError("whatever it says"))
    assert engine.out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not engine.out_of_memory(RuntimeError("probability tensor contains nan"))
    assert not engine.out_of_memory(KeyError("input_ids"))


class Tokens(dict):
    """What a tokenizer's chat template gives: tensors by name, movable to a device."""

    def to(self, device):
        self["moved_to"] = device
        return self


@pytest.fixture
def transformers_stand_in(monkeypatch):
    """torch and transformers as far as the engine touches them. The model
    'generates' the tokens 7, 8, 9 after whatever prompt it is given."""
    asked = {}

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text):
            return text.split()

        def apply_chat_template(self, messages, **options):
            asked["template"] = options
            prompt = types.SimpleNamespace(shape=(1, 5))
            return Tokens(input_ids=prompt)

        def decode(self, tokens, skip_special_tokens=False):
            return f" decoded {list(tokens)} "

    class Model:
        device = "cuda:0"

        def eval(self):
            asked["eval"] = True

        def generate(self, **options):
            asked["generate"] = options
            return [[1, 2, 3, 4, 5, 7, 8, 9]]

    def tokenizer_from(model_id):
        asked["tokenizer"] = model_id
        return Tokenizer()

    def model_from(model_id, **options):
        asked["model"] = (model_id, options)
        return Model()

    class NoGrad:
        def __enter__(self):
            asked["no_grad"] = True

        def __exit__(self, *error):
            return False

    torch = types.SimpleNamespace(
        bfloat16="bf16",
        no_grad=NoGrad,
        cuda=types.SimpleNamespace(empty_cache=lambda: asked.setdefault("emptied", True)),
    )
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_from),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_from),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.delenv("DSQA_ENGINE", raising=False)
    return asked


def test_the_base_model_is_loaded_and_asked_as_the_examples_do(transformers_stand_in):
    asked = transformers_stand_in
    loaded = engine.load("the/base-model")
    # By the name the evaluation gives, in bf16, on the GPU.
    assert asked["tokenizer"] == "the/base-model"
    assert asked["model"] == ("the/base-model", {"dtype": "bf16", "device_map": "cuda"})
    assert asked["eval"]
    assert loaded.count("three words here") == 3

    reply = loaded.generate([{"role": "user", "content": "hello"}], 64)
    assert reply == "decoded [7, 8, 9]"  # what came after the prompt, stripped
    assert asked["template"] == {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    options = asked["generate"]
    assert options["moved_to"] == "cuda:0"
    assert (options["max_new_tokens"], options["do_sample"], options["pad_token_id"]) == (
        64,
        False,
        0,
    )
    assert "max_time" not in options and asked["no_grad"]

    # A limit in seconds is the generation's own, and the GPU is given back when asked.
    loaded.generate([{"role": "user", "content": "hello"}], 64, 12.5)
    assert asked["generate"]["max_time"] == 12.5
    loaded.release()
    assert asked["emptied"]

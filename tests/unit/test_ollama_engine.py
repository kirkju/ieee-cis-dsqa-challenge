"""tools/engines/ollama.py: the stand-in engine, with Ollama itself stood in for."""

import importlib.util
import io
import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tools" / "engines" / "ollama.py"


@pytest.fixture
def ollama(monkeypatch):
    """The engine's module, loaded from its file as the entry loads it, with a
    tokenizer of a token per word in place of the base model's."""
    tokenizer = types.SimpleNamespace(encode=lambda text: text.split())
    names = []

    def tokenizer_from(model_id):
        names.append(model_id)
        return tokenizer

    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_from)
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = importlib.util.spec_from_file_location("ollama_engine", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.names = names
    return module


def test_the_context_is_counted_with_the_base_model_s_tokenizer(ollama):
    engine = ollama.load("the/base-model")
    assert ollama.names == ["the/base-model"]
    assert engine.count("four words of text") == 4
    engine.release()


def test_the_stand_in_is_asked_not_to_think_and_to_decode_greedily(ollama, monkeypatch):
    sent = {}

    def urlopen(request, timeout=None):
        sent.update(url=request.full_url, body=json.loads(request.data), timeout=timeout)
        reply = {"message": {"role": "assistant", "content": " In the dump tank.\n"}}
        return io.BytesIO(json.dumps(reply).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    conversation = [
        {"role": "system", "content": "You are an expert."},
        {"role": "user", "content": "Where is the trap?"},
    ]
    reply = ollama.load("the/base-model").generate(conversation, 256, 12.0)
    assert reply == "In the dump tank."
    assert sent["url"] == "http://localhost:11434/api/chat"
    body = sent["body"]
    assert body["messages"] == conversation
    assert (body["model"], body["stream"], body["think"]) == ("qwen3.5:4b", False, False)
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 256
    # A window that holds the entry's context, and its own time rather than the run's.
    assert body["options"]["num_ctx"] >= 12_000
    assert sent["timeout"] == ollama.SECONDS

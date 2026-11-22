"""A stand-in for the base model: a small model served by Ollama on this machine.

The base model does not fit the laptop the entry is developed on. To run the
whole of answer() there anyway, the entry takes its engine from this file
when the environment says so (dsqa/engine.py):

    DSQA_ENGINE=tools/engines/ollama.py python -m tools.run_local 03_dsqa \
        --model-id <the base model's id> --corpus-dir data/corpus --limit 5

What it stands in for is the generation alone. The context is still counted
with the base model's tokenizer, which is small and has to be in the local
Hugging Face cache, so the prompts are the size they will be in the
evaluation. What the stand-in answers says nothing of what the base model
would: this checks the plumbing, the prompts and the time accounting.

The file stands alone, since nothing of the repository is importable while
a submission runs, and it is not part of any zip.
"""

import json
import os
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# A thinking model whose defaults sample at temperature 1: it is asked not to
# think and to decode greedily, as the evaluation's model decodes.
MODEL = os.environ.get("DSQA_OLLAMA_MODEL", "qwen3.5:4b")
# Ollama cuts a prompt longer than its context window without saying so. The
# entry's context is 10,000 tokens of the base model's; this leaves room for
# the instructions, the question and the answer.
WINDOW = int(os.environ.get("DSQA_OLLAMA_WINDOW", "16384"))
# The stand-in is slower than the model it stands in for, and its time is not
# the run's: it is given this long whatever the entry allows.
SECONDS = 900


class Ollama:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(
        self, messages: list[dict], max_new_tokens: int, max_seconds: float | None = None
    ) -> str:
        asked = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": max_new_tokens, "num_ctx": WINDOW},
        }
        request = urllib.request.Request(
            f"{HOST}/api/chat",
            data=json.dumps(asked).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=SECONDS) as response:
            reply = json.loads(response.read().decode("utf-8"))
        return reply["message"]["content"].strip()

    def release(self) -> None:
        pass


def load(model_id: str) -> Ollama:
    return Ollama(model_id)

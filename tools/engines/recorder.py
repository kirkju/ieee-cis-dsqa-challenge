"""A stand-in for the base model that answers nothing and keeps a record of
every conversation it is shown.

    DSQA_ENGINE=<absolute path to this file> DSQA_RECORD=<file.jsonl> \
        python -m tools.run_local 03_dsqa --model-id <the base model's id> \
        --corpus-dir data/corpus --questions sample/sample_qa.json

It is how two runs of the entry are compared on what they ask the model:
the record has, for every call, the hash of the conversation, the most it
may write and the question it ends with. Two runs whose records agree on a
question asked the model the same thing, word for word, for it. The
context is counted with the base model's tokenizer, as the real engine
counts it, so what is placed is what would be placed in the evaluation.

Its reply states a fixed answer on the line the entry asks for, so a run
goes through every step that does not depend on what the answer says. The
file stands alone, since nothing of the repository is importable while a
submission runs, and it is not part of any zip.
"""

import hashlib
import json
import os

REPLY = "Working recorded.\nFinal answer: Recorded."


class Recorder:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.path = os.environ.get("DSQA_RECORD")

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(self, messages: list, max_new_tokens: int, max_seconds: float | None = None):
        if self.path:
            canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            last = messages[-1]["content"]
            row = {
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "max_new_tokens": max_new_tokens,
                "system": messages[0]["content"][:60],
                "tokens": self.count(canonical),
                "ends_with": last[-200:],
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return REPLY

    def release(self) -> None:
        pass


def load(model_id: str) -> Recorder:
    return Recorder(model_id)

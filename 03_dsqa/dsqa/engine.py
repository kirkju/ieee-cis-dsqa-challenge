"""What generates the answers: the base model, behind one boundary.

Everything else in the entry sees an engine as three things it can do:

    count(text)       how many tokens the model makes of a text
    generate(...)     the model's answer to a conversation
    release()         give back what the last generation left on the GPU

and, when it can, a fourth: score(conversations, labels), how far the model
leans to each label as the first thing it would write, read with nothing
written (relevance.py). An engine without it leaves the adjustments that
need it without effect. first(conversation, labels) says the same of one
conversation, read through generate() itself, as every reply is.

In the evaluation the engine is the base model, loaded through transformers
exactly as the organizers' examples load it. The base model does not fit the
machine the entry is developed on, so the rest of the pipeline is exercised
there with a stand-in behind the same boundary: when the environment variable
DSQA_ENGINE names a Python file, load() takes the engine from that file's own
load(model_id). The file is not in the zip and the variable is not set in the
evaluation.
"""

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path

# The most tokens score() reads in one pass, padding counted: what a batch
# takes of the GPU does not grow with the number of conversations.
SCORE_TOKENS = 8_000


def load(model_id: str):
    """The engine of the run: the base model, unless DSQA_ENGINE names a file
    that gives another."""
    stand_in = os.environ.get("DSQA_ENGINE")
    if stand_in:
        spec = importlib.util.spec_from_file_location("dsqa_stand_in_engine", Path(stand_in))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load(model_id)
    return Transformers(model_id)


def out_of_memory(error: BaseException) -> bool:
    """Whether an error is the GPU running out of memory."""
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


class Transformers:
    """The base model through transformers. model_id is the Hugging Face
    name the evaluation hands to answer(); it resolves from a local cache."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(
        self, messages: list[dict], max_new_tokens: int, max_seconds: float | None = None
    ) -> str:
        """The model's reply to a conversation, decoded greedily. Past
        max_seconds the generation stops where it is and what it has is
        returned."""
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        limits = {} if max_seconds is None else {"max_time": max_seconds}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                **limits,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()

    def score(
        self,
        conversations: list[list[dict]],
        labels: Sequence[str] = ("Yes", "No"),
        most_tokens: int = SCORE_TOKENS,
    ) -> list[list[float]] | None:
        """For each conversation, how far the model leans to each label as the
        first thing it would write: the label's logit at the last position,
        read in one pass with nothing written. A label is the most likely of
        its spellings, with and without a space before it, capitalized and
        not, each by its first token.

        The conversations go in batches of no more than `most_tokens` tokens
        with their padding, padded on the left, each by the ids its chat
        template gives it and with the positions it would have alone. A batch
        that runs the GPU out of memory is tried again in halves; when a half
        does too, None, and the caller goes on without the scores."""
        if not conversations:
            return []
        prompts = [self._prompt_ids(conversation) for conversation in conversations]
        wanted = [self._label_ids(label) for label in labels]
        by_length = sorted(range(len(prompts)), key=lambda number: len(prompts[number]))
        found: dict[int, list[float]] = {}
        for batch in _batches(by_length, [len(prompt) for prompt in prompts], most_tokens):
            try:
                found.update(self._scored(batch, prompts, wanted))
            except Exception as error:
                if not out_of_memory(error):
                    raise
                self.release()
                if len(batch) == 1:
                    return None
                try:
                    middle = len(batch) // 2
                    for half in (batch[:middle], batch[middle:]):
                        found.update(self._scored(half, prompts, wanted))
                except Exception as again:
                    if not out_of_memory(again):
                        raise
                    self.release()
                    return None
        return [found[number] for number in range(len(prompts))]

    def first(
        self, conversation: list[dict], labels: Sequence[str] = ("Yes", "No")
    ) -> list[float] | None:
        """How far the model leans to each label as the first token of its
        reply to one conversation, as score() says it, but read the way every
        reply of the entry is read: by generate(), one conversation alone and
        nothing padded, one token written and its logits kept as they came,
        before any setting of the model's generation could touch them. None
        when the GPU runs out of memory."""
        inputs = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        wanted = [self._label_ids(label) for label in labels]
        try:
            with self.torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
        except Exception as error:
            if not out_of_memory(error):
                raise
            self.release()
            return None
        last = output.logits[0][0].float().cpu()
        return [float(last[ids].max()) for ids in wanted]

    def _prompt_ids(self, conversation: list[dict]) -> list[int]:
        """The ids the chat template gives a conversation, ready for the reply."""
        made = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True
        )
        ids = made["input_ids"] if hasattr(made, "keys") else made
        return [int(token) for token in ids]

    def _label_ids(self, label: str) -> list[int]:
        """The first token of each spelling of a label."""
        spellings = {label, label.lower(), label.capitalize()}
        spellings |= {f" {spelling}" for spelling in spellings}
        ids = set()
        for spelling in sorted(spellings):
            tokens = self.tokenizer.encode(spelling, add_special_tokens=False)
            if tokens:
                ids.add(int(tokens[0]))
        return sorted(ids)

    def _scored(self, batch: list[int], prompts: list[list[int]], wanted) -> dict:
        """The labels' logits for the conversations of one batch, by number."""
        torch = self.torch
        longest = max(len(prompts[number]) for number in batch)
        pad = self.tokenizer.pad_token_id
        pad = self.tokenizer.eos_token_id if pad is None else pad
        ids = torch.full((len(batch), longest), int(pad), dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, number in enumerate(batch):
            prompt = prompts[number]
            ids[row, longest - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            mask[row, longest - len(prompt) :] = 1
        # Each prompt's positions as they would be alone: counted from its
        # first token, not from the padding before it.
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        device = self.model.device
        with torch.no_grad():
            output = self.model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                position_ids=positions.to(device),
                use_cache=False,
                logits_to_keep=1,
            )
        last = output.logits[:, -1, :].float().cpu()
        return {
            number: [float(last[row, label].max()) for label in wanted]
            for row, number in enumerate(batch)
        }

    def release(self) -> None:
        self.torch.cuda.empty_cache()


def _batches(order: list[int], lengths: list[int], most_tokens: int) -> list[list[int]]:
    """The numbers in `order` in batches whose padded size, as many rows as
    there are times the longest, is no more than `most_tokens`; a prompt
    longer than that alone in its batch."""
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for number in order:
        size = max(longest, lengths[number]) * (len(current) + 1)
        if current and size > most_tokens:
            batches.append(current)
            current, longest = [], 0
        current.append(number)
        longest = max(longest, lengths[number])
    if current:
        batches.append(current)
    return batches

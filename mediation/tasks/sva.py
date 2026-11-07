"""Subject-verb agreement prompt pairs (Marks et al., 2025, ``nounpp`` files).

The clean prompt ends after a subject noun and a prepositional phrase ("The teachers near
the desk") and the counterfactual prompt flips the number of the subject ("The teacher near
the desk"). The metric is the logit of the verb that agrees with the clean subject minus the
logit of the other form (" are" minus " is").
"""

import json
import random
from functools import partial
from pathlib import Path

import torch

from mediation.prompt_pair import PromptPair

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "sva"


def sva_metric(
    logits: torch.Tensor, correct_verb_id: int, incorrect_verb_id: int
) -> torch.Tensor:
    """Logit of the correct verb minus logit of the incorrect verb at the last position."""
    return logits[..., -1, correct_verb_id] - logits[..., -1, incorrect_verb_id]


def load_sva_pairs(tokenizer, count: int = 1000, seed: int = 42) -> list[PromptPair]:
    """Sample distinct subject-verb agreement pairs of equal token length.

    Records are read from the train file and then the test file. A record is kept if both
    verb forms are one token each, its (clean, counterfactual) prompt pair has not occurred
    before, and the two prompts have the same number of tokens. Equal length is required
    because patching replaces activations position by position: when the plural noun splits
    into two tokens ("carpent|ers") and the singular does not, every later position would be
    patched with the activation of a different word. ``count`` pairs are then drawn with
    ``random.Random(seed)``.

    :raises ValueError: when fewer than ``count`` pairs survive the filters.
    """
    records = []
    for file_name in ("nounpp_train.json", "nounpp_test.json"):
        lines = (DATA_DIRECTORY / file_name).read_text().splitlines()
        records.extend(json.loads(line) for line in lines if line.strip())

    pairs, seen = [], set()
    for record in records:
        # Verbs are encoded with a leading space and without special tokens, so that
        # Llama's beginning-of-sequence token is not mistaken for part of the verb.
        correct_ids = tokenizer.encode(
            " " + record["clean_answer"].strip(), add_special_tokens=False
        )
        incorrect_ids = tokenizer.encode(
            " " + record["patch_answer"].strip(), add_special_tokens=False
        )
        if len(correct_ids) != 1 or len(incorrect_ids) != 1:
            continue
        prompts = (record["clean_prefix"], record["patch_prefix"])
        if prompts in seen:
            continue
        seen.add(prompts)
        clean_tokens = tokenizer.encode(prompts[0])
        counterfactual_tokens = tokenizer.encode(prompts[1])
        if len(clean_tokens) != len(counterfactual_tokens):
            continue
        pairs.append(
            PromptPair(
                clean_tokens=clean_tokens,
                counterfactual_tokens=counterfactual_tokens,
                metric=partial(
                    sva_metric,
                    correct_verb_id=correct_ids[0],
                    incorrect_verb_id=incorrect_ids[0],
                ),
            )
        )
    if len(pairs) < count:
        raise ValueError(f"Only {len(pairs)} pairs survive the filters; {count} wanted")
    return random.Random(seed).sample(pairs, count)

"""The unit that every task produces and every estimand consumes."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PromptPair:
    """A clean prompt, its counterfactual prompt and the metric that scores the model.

    :ivar clean_tokens: token ids of the clean prompt, including any special token the
        tokenizer adds by default.
    :ivar counterfactual_tokens: token ids of the counterfactual prompt. Patching replaces
        activations position by position over the shared prefix, the first
        ``min(len(clean_tokens), len(counterfactual_tokens))`` positions.
    :ivar metric: maps logits of shape (1, positions, vocabulary) to a value of shape (1,),
        scored at the last position of the prompt that was run. It is positive when the
        model behaves correctly on the clean prompt (for example the logit of the indirect
        object minus the logit of the subject).
    :ivar metadata: task-specific description of the pair (template, token positions, ...).
    """

    clean_tokens: list[int]
    counterfactual_tokens: list[int]
    metric: Callable[[torch.Tensor], torch.Tensor]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def tokenize_equal_length(
    tokenizer, clean_text: str, counterfactual_text: str
) -> tuple[list[int], list[int]]:
    """Token ids of a clean and a counterfactual prompt that must have the same length.

    Patching replaces activations position by position, so prompts of unequal length would
    align the activations of different words.

    :raises ValueError: if the two prompts have different numbers of tokens.
    """
    clean_tokens = tokenizer.encode(clean_text)
    counterfactual_tokens = tokenizer.encode(counterfactual_text)
    if len(clean_tokens) != len(counterfactual_tokens):
        raise ValueError(
            f"Unequal token length: {clean_text!r} vs {counterfactual_text!r}"
        )
    return clean_tokens, counterfactual_tokens

"""Interpolate a head's activation between its clean and counterfactual values.

For a head with clean activation ``h(a)`` and counterfactual activation ``h(a')`` the
interpolation ``h(w) = (1 - w) h(a) + w h(a')`` moves the head a fraction ``w`` of the way
to its counterfactual value, so its distance from the clean activation is ``w`` times the
distance of the full replacement. Patching ``h(w)`` into the clean run gives the noising
outcome ``Y(a, h(w))`` and into the counterfactual run the denoising outcome
``Y(a', h(w))``. At ``w = 0`` the second is the denoised run and at ``w = 1`` the first is
the noised run; the interaction at weight ``w`` is therefore (Figure 2)
``INT(w) = [Y(a) - Y(a, h(w))] - [Y(a', h(0)) - Y(a', h(w))]``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from mediation.models import LoadedModel
from mediation.patching import AttentionHead, cache_activations, run_with_replacements
from mediation.prompt_pair import PromptPair

# The six heads of Figure 2, one per role plus a head outside the circuit: Name Mover,
# S-Inhibition, Negative Name Mover, Duplicate Token, Backup Name Mover, outside the circuit.
INTERPOLATION_HEADS = (
    AttentionHead(9, 9),
    AttentionHead(8, 6),
    AttentionHead(10, 7),
    AttentionHead(3, 0),
    AttentionHead(10, 2),
    AttentionHead(6, 0),
)
INTERPOLATION_WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class InterpolationOutcomes:
    """Metrics of one prompt pair with heads patched to interpolated activations.

    :ivar clean_metric: ``Y(a)``.
    :ivar counterfactual_metric: ``Y(a')``.
    :ivar noised_metric: ``Y(a, h(w))``, shape (heads, weights).
    :ivar denoised_metric: ``Y(a', h(w))``, shape (heads, weights).
    :ivar activation_distance: mean over the shared-prefix positions of the L2 distance
        between the head's clean and counterfactual activations (``w = 1``), shape (heads,).
    """

    clean_metric: float
    counterfactual_metric: float
    noised_metric: np.ndarray
    denoised_metric: np.ndarray
    activation_distance: np.ndarray


def interpolate_heads(
    model: LoadedModel,
    pair: PromptPair,
    heads: Sequence[AttentionHead] = INTERPOLATION_HEADS,
    weights: Sequence[float] = INTERPOLATION_WEIGHTS,
) -> InterpolationOutcomes:
    """Patch each head, one at a time, to each interpolation of its two activations.

    The activations are those of one full pass on each prompt; the interpolated value
    replaces the head over the shared prefix of the two prompts.
    """
    clean = cache_activations(model, pair.clean_tokens)
    counterfactual = cache_activations(model, pair.counterfactual_tokens)
    positions = min(len(pair.clean_tokens), len(pair.counterfactual_tokens))

    noised = np.zeros((len(heads), len(weights)), dtype=np.float32)
    denoised = np.zeros_like(noised)
    distance = np.zeros(len(heads), dtype=np.float32)
    for head_index, head in enumerate(heads):
        clean_values = clean.values(head, positions)
        counterfactual_values = counterfactual.values(head, positions)
        distance[head_index] = (
            torch.linalg.vector_norm(clean_values - counterfactual_values, dim=-1)
            .mean()
            .item()
        )
        for weight_index, weight in enumerate(weights):
            interpolated = (
                1.0 - weight
            ) * clean_values + weight * counterfactual_values
            for tokens, outcomes in (
                (pair.clean_tokens, noised),
                (pair.counterfactual_tokens, denoised),
            ):
                logits = run_with_replacements(model, tokens, {head: interpolated})
                outcomes[head_index, weight_index] = pair.metric(logits).item()
    return InterpolationOutcomes(
        clean_metric=pair.metric(clean.logits).item(),
        counterfactual_metric=pair.metric(counterfactual.logits).item(),
        noised_metric=noised,
        denoised_metric=denoised,
        activation_distance=distance,
    )

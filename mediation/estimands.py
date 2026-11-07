"""Natural and pure indirect effects, their interaction, and cross-interactions.

For a prompt pair let ``a`` be the clean prompt and ``a'`` the counterfactual prompt, and
``Y`` the pair's metric. For a set of mediators ``S`` let ``Y(a, S <- a')`` be the metric on
the clean prompt with ``S`` replaced by its activation on the counterfactual prompt.

* noising, natural indirect effect: ``NIE = Y(a) - Y(a, S <- a')``;
* denoising, pure indirect effect: ``PIE = Y(a', S <- a) - Y(a')``;
* interaction: ``INT = NIE - PIE`` (Proposition 3.1).

For two disjoint sets ``A`` and ``B`` the cross-interaction of Theorem 4.1 is
``xINT(A, B) = Y(A) + Y(B) - Y(A + B) - Y(empty)``, where ``Y(S)`` abbreviates
``Y(a, S <- source)`` for a source of replacement values, so that
``NIE(A + B) = NIE(A) + NIE(B) + xINT(A, B)``.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from mediation.models import LoadedModel
from mediation.patching import (
    Activations,
    Component,
    MlpLayer,
    all_attention_heads,
    cache_activations,
    run_with_source,
)
from mediation.prompt_pair import PromptPair


@dataclass(frozen=True)
class MediatorEffects:
    """NIE and PIE of each mediator for one prompt pair, both float32 of shape
    ``(num_mediators,)`` or ``(num_layers, num_heads)`` for :func:`head_effects`."""

    nie: np.ndarray
    pie: np.ndarray

    @property
    def interaction(self) -> np.ndarray:
        """INT = NIE - PIE, formed in float32."""
        return self.nie - self.pie


def component_effects(
    model: LoadedModel, pair: PromptPair, components: Sequence[Component]
) -> MediatorEffects:
    """NIE and PIE of each component alone, replaced over the shared prefix of the pair.

    The clean and counterfactual runs that supply the replacement values are one
    full-sequence pass each, so the replaced activation of a component is the one it has in
    the unpatched run of the other prompt.
    """
    mlp_layers = [c.layer for c in components if isinstance(c, MlpLayer)]
    clean = cache_activations(model, pair.clean_tokens, mlp_layers)
    counterfactual = cache_activations(model, pair.counterfactual_tokens, mlp_layers)
    clean_metric = pair.metric(clean.logits).item()
    counterfactual_metric = pair.metric(counterfactual.logits).item()
    positions = min(len(pair.clean_tokens), len(pair.counterfactual_tokens))

    nie = np.zeros(len(components), dtype=np.float32)
    pie = np.zeros(len(components), dtype=np.float32)
    for index, component in enumerate(components):
        # Denoising: counterfactual prompt, component restored to its clean value.
        denoised = run_with_source(
            model, pair.counterfactual_tokens, clean, [component], positions
        )
        # Noising: clean prompt, component replaced by its counterfactual value.
        noised = run_with_source(
            model, pair.clean_tokens, counterfactual, [component], positions
        )
        pie[index] = pair.metric(denoised).item() - counterfactual_metric
        nie[index] = clean_metric - pair.metric(noised).item()
    return MediatorEffects(nie=nie, pie=pie)


def head_effects(model: LoadedModel, pair: PromptPair) -> MediatorEffects:
    """NIE and PIE of every attention head alone, arrays of shape (num_layers, num_heads)."""
    effects = component_effects(model, pair, all_attention_heads(model))
    shape = (model.num_layers, model.num_heads)
    return MediatorEffects(effects.nie.reshape(shape), effects.pie.reshape(shape))


def patched_metric(
    model: LoadedModel,
    tokens: Sequence[int],
    pair: PromptPair,
    source: Activations,
    components: Collection[Component],
) -> float:
    """``Y(S)``: the pair's metric on ``tokens`` with ``components`` replaced from ``source``.

    ``source`` is the counterfactual prompt's :func:`~mediation.patching.cache_activations`
    for resample ablation or :func:`~mediation.patching.mean_activations` for mean ablation.
    With no components this is the unpatched metric ``Y(empty)``.
    """
    logits = run_with_source(model, tokens, source, components)
    return pair.metric(logits).item()


def cross_interaction(
    y_empty: float, y_first: float, y_second: float, y_both: float
) -> float:
    """xINT of two sets of mediators with the sign of Theorem 4.1:
    ``Y(A) + Y(B) - Y(A + B) - Y(empty)``."""
    return y_first + y_second - y_both - y_empty


@dataclass(frozen=True)
class JointOutcomes:
    """The four patched metrics that determine the cross-interaction of two sets.

    :ivar y_empty: nothing replaced.
    :ivar y_first: the first set replaced.
    :ivar y_second: the second set replaced.
    :ivar y_both: both sets replaced.
    """

    y_empty: float
    y_first: float
    y_second: float
    y_both: float

    @property
    def cross_interaction(self) -> float:
        return cross_interaction(self.y_empty, self.y_first, self.y_second, self.y_both)


@dataclass(frozen=True)
class OutcomeGrid:
    """The patched metrics that determine every cross-interaction between two lists of sets.

    :ivar y_empty: nothing replaced.
    :ivar y_first: metric with ``first[a]`` replaced, shape (A,).
    :ivar y_second: metric with ``second[b]`` replaced, shape (B,).
    :ivar y_both: metric with ``first[a]`` and ``second[b]`` replaced together, shape (A, B).
    """

    y_empty: float
    y_first: np.ndarray
    y_second: np.ndarray
    y_both: np.ndarray

    @property
    def cross_interaction(self) -> np.ndarray:
        """xINT of every ``first[a]`` with every ``second[b]``, shape (A, B)."""
        return cross_interaction(
            self.y_empty,
            self.y_first[:, None],
            self.y_second[None, :],
            self.y_both,
        )


def joint_outcome_grid(
    model: LoadedModel,
    pair: PromptPair,
    source: Activations,
    first: Sequence[Collection[Component]],
    second: Sequence[Collection[Component]],
) -> OutcomeGrid:
    """``Y(empty)``, ``Y(A)``, ``Y(B)`` and ``Y(A + B)`` for every ``A`` in ``first`` and
    ``B`` in ``second``, on the clean prompt of ``pair``.

    Each set is replaced together in one forward pass, so a set of many heads (a circuit
    and its complement, a group of Name Mover Heads) is one mediator in Theorem 4.1. The
    unpatched run and each single set are run once and shared by all cells, so the grid
    costs ``1 + A + B + A * B`` passes.
    """

    def outcome(components: Collection[Component]) -> float:
        return patched_metric(model, pair.clean_tokens, pair, source, components)

    return OutcomeGrid(
        y_empty=outcome([]),
        y_first=np.array([outcome(a) for a in first]),
        y_second=np.array([outcome(b) for b in second]),
        y_both=np.array([[outcome([*a, *b]) for b in second] for a in first]),
    )


def joint_outcomes(
    model: LoadedModel,
    pair: PromptPair,
    source: Activations,
    first: Collection[Component],
    second: Collection[Component],
) -> JointOutcomes:
    """``Y(empty)``, ``Y(A)``, ``Y(B)``, ``Y(A + B)`` on the clean prompt of ``pair``;
    the one-cell case of :func:`joint_outcome_grid`."""
    grid = joint_outcome_grid(model, pair, source, [first], [second])
    return JointOutcomes(
        y_empty=grid.y_empty,
        y_first=float(grid.y_first[0]),
        y_second=float(grid.y_second[0]),
        y_both=float(grid.y_both[0, 0]),
    )

"""Appendix C: cross-interactions among the MLPs of the Greater-Than circuit in GPT-2 small.

For each of Hanna et al.'s 1000 Greater-Than prompt pairs the script runs the clean prompt
with every single MLP of layers 0-3 and 8-11 ablated and with every pair of them ablated
(1 + 8 + 28 = 37 forward passes per ablation), replacing the MLP output at every position.
The ablation is

* mean ablation: the output's position-wise mean over the 1000 counterfactual (year-01)
  prompts, or
* resample ablation: the output on the pair's own counterfactual prompt.

Written to ``<output-directory>/<ablation>/pairs_<start>_<stop>.npz`` for each ablation:
``y_clean`` (pairs,), the unablated metric; ``y_single`` (pairs, 8), the metric with one
MLP ablated, in the order of :data:`mediation.circuits.GREATER_THAN_MLPS`; ``y_joint``
(pairs, 8, 8), the metric with MLPs ``i`` and ``j`` ablated, filled for ``i < j`` and NaN
elsewhere. The cross-interaction of Theorem 4.1 follows from these four quantities.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from mediation.circuits import GREATER_THAN_MLPS
from mediation.estimands import patched_metric
from mediation.models import load_model
from mediation.patching import Activations, cache_activations, mean_activations
from mediation.prompt_pair import PromptPair
from mediation.results import (
    OUTPUT_ROOT,
    add_pair_range_argument,
    resolve_pair_range,
    save_pair_range,
)
from mediation.tasks import load_prompt_pairs

MODEL_NAME = "gpt2-small"
MLP_LAYERS = [mlp.layer for mlp in GREATER_THAN_MLPS]


def ablated_metrics(
    model, pair: PromptPair, source: Activations
) -> tuple[float, np.ndarray, np.ndarray]:
    """Metric of the clean prompt with no, each single and each pair of MLPs ablated.

    :param source: activations the ablated MLPs take: the counterfactual prompt's
        (resample ablation) or the mean over counterfactual prompts (mean ablation).
    :return: ``y_clean``, ``y_single`` of shape (8,) and ``y_joint`` of shape (8, 8) with
        entries ``i < j`` filled and NaN elsewhere.
    """
    count = len(GREATER_THAN_MLPS)

    def metric(mlps) -> float:
        return patched_metric(model, pair.clean_tokens, pair, source, mlps)

    y_single = np.array([metric([mlp]) for mlp in GREATER_THAN_MLPS])
    y_joint = np.full((count, count), np.nan)
    for first, second in itertools.combinations(range(count), 2):
        y_joint[first, second] = metric(
            [GREATER_THAN_MLPS[first], GREATER_THAN_MLPS[second]]
        )
    return metric([]), y_single, y_joint


def compute_pair_range(
    model, pairs: list[PromptPair], mean_source: Activations, start: int, stop: int
) -> dict[str, dict[str, np.ndarray]]:
    """Arrays of both ablations for pairs ``start`` to ``stop - 1``, keyed by ablation."""
    results = {
        ablation: {"y_clean": [], "y_single": [], "y_joint": []}
        for ablation in ("mean", "resample")
    }
    for pair in pairs[start:stop]:
        sources = {
            "mean": mean_source,
            # The counterfactual prompt of this pair, whose MLP outputs replace the clean
            # prompt's.
            "resample": cache_activations(
                model, pair.counterfactual_tokens, MLP_LAYERS
            ),
        }
        for ablation, source in sources.items():
            y_clean, y_single, y_joint = ablated_metrics(model, pair, source)
            results[ablation]["y_clean"].append(y_clean)
            results[ablation]["y_single"].append(y_single)
            results[ablation]["y_joint"].append(y_joint)
    return {
        ablation: {name: np.array(values) for name, values in arrays.items()}
        for ablation, arrays in results.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_pair_range_argument(parser)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_ROOT / "greater_than_cross_interactions",
    )
    arguments = parser.parse_args()

    pairs = load_prompt_pairs("greater_than", MODEL_NAME)
    start, stop = resolve_pair_range(arguments.pairs, len(pairs))

    # The mean over counterfactual prompts always averages all pairs, whichever range this
    # job covers, and uses eager attention as the published run did. The ablated runs use
    # the default attention implementation.
    mean_source = mean_activations(
        load_model(MODEL_NAME, attention_implementation="eager"),
        [pair.counterfactual_tokens for pair in pairs],
        MLP_LAYERS,
    )
    results = compute_pair_range(
        load_model(MODEL_NAME), pairs, mean_source, start, stop
    )
    for ablation, arrays in results.items():
        save_pair_range(arguments.output_directory / ablation, start, stop, **arrays)


if __name__ == "__main__":
    main()

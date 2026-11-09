"""Section 4: knock out the IOI circuit, the heads outside it, and all heads, by resample.

For every IOI pABC prompt pair of GPT-2 small, each head of a knocked-out set is replaced,
at every position, by its activation on the pair's pABC prompt. With ``C`` the 26 circuit
heads of Wang et al. and ``C'`` the other 118, the clean prompt is run with nothing,
``C``, ``C'`` and ``C + C'`` replaced.

Writes, per pair range, ``clean_metric``, ``circuit_knocked_out_metric``,
``complement_knocked_out_metric`` and ``all_knocked_out_metric``, each of shape (pairs,).

Run ``python experiments/circuit_knockout.py``; ``--pairs start:stop`` restricts it to a range.
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.circuits import ioi_circuit_heads
from mediation.estimands import joint_outcomes
from mediation.models import load_model
from mediation.patching import all_attention_heads, cache_activations
from mediation.results import (
    OUTPUT_ROOT,
    add_pair_range_argument,
    resolve_pair_range,
    save_pair_range,
)
from mediation.tasks import load_prompt_pairs

MODEL = "gpt2-small"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_pair_range_argument(parser)
    parser.add_argument(
        "--output-directory", type=Path, default=OUTPUT_ROOT / "circuit_knockout"
    )
    arguments = parser.parse_args()

    # Eager attention, as in the published run.
    model = load_model(MODEL, attention_implementation="eager")
    all_pairs = load_prompt_pairs("ioi_pabc", MODEL)
    start, stop = resolve_pair_range(arguments.pairs, len(all_pairs))
    pairs = all_pairs[start:stop]

    circuit = ioi_circuit_heads()
    complement = [head for head in all_attention_heads(model) if head not in circuit]
    outcomes = []
    for pair in pairs:
        # Resample: the replacement values are the pair's pABC prompt's activations.
        source = cache_activations(model, pair.counterfactual_tokens)
        outcomes.append(joint_outcomes(model, pair, source, circuit, complement))

    save_pair_range(
        arguments.output_directory,
        start,
        stop,
        clean_metric=np.array([o.y_empty for o in outcomes]),
        circuit_knocked_out_metric=np.array([o.y_first for o in outcomes]),
        complement_knocked_out_metric=np.array([o.y_second for o in outcomes]),
        all_knocked_out_metric=np.array([o.y_both for o in outcomes]),
    )


if __name__ == "__main__":
    main()

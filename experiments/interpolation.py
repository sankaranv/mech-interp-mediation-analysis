"""Figure 2: patch six GPT-2 small heads to interpolations between clean and pABC activations.

Writes, per pair range, ``clean_metric`` and ``counterfactual_metric`` (pairs,),
``noised_metric`` and ``denoised_metric`` (pairs, heads, weights) and
``activation_distance`` (pairs, heads); heads and weights are
:data:`mediation.interpolation.INTERPOLATION_HEADS` and ``INTERPOLATION_WEIGHTS``.

Run ``python experiments/interpolation.py --pairs 0:1000``.
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.interpolation import interpolate_heads
from mediation.models import load_model
from mediation.results import OUTPUT_ROOT, parse_pair_range, save_pair_range
from mediation.tasks import load_prompt_pairs

MODEL = "gpt2-small"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default="0:1000", help="start:stop, stop excluded")
    parser.add_argument(
        "--output-directory", type=Path, default=OUTPUT_ROOT / "interpolation"
    )
    arguments = parser.parse_args()
    start, stop = parse_pair_range(arguments.pairs)

    # Eager attention, as in the published run; the other estimands use the default
    # implementation, which differs only at float32 rounding level.
    model = load_model(MODEL, attention_implementation="eager")
    pairs = load_prompt_pairs("ioi_pabc", MODEL)[start:stop]
    outcomes = [interpolate_heads(model, pair) for pair in pairs]

    save_pair_range(
        arguments.output_directory,
        start,
        stop,
        clean_metric=np.array([o.clean_metric for o in outcomes], dtype=np.float32),
        counterfactual_metric=np.array(
            [o.counterfactual_metric for o in outcomes], dtype=np.float32
        ),
        noised_metric=np.stack([o.noised_metric for o in outcomes]),
        denoised_metric=np.stack([o.denoised_metric for o in outcomes]),
        activation_distance=np.stack([o.activation_distance for o in outcomes]),
    )


if __name__ == "__main__":
    main()

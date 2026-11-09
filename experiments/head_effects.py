"""NIE and PIE of every attention head for every prompt pair of one model and task.

Feeds Figure 1 (rank agreement across models and tasks), Figures 3 and 4 and the Section 5
numbers (IOI with the pABC counterfactual on GPT-2 small), and Appendix B's table of all
circuit heads. Each pair costs two forward passes per head.

Writes ``nie`` and ``pie``, float32 arrays of shape (pairs, layers, heads), to
``<output-directory>/pairs_<start>_<stop>.npz``; the default output directory is
``outputs/head_effects/<model>/<task>``.
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.estimands import head_effects
from mediation.models import MODELS, load_model
from mediation.results import (
    OUTPUT_ROOT,
    add_pair_range_argument,
    head_effects_directory,
    resolve_pair_range,
    save_pair_range,
)
from mediation.tasks import TASKS, load_prompt_pairs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--task", required=True, choices=TASKS)
    add_pair_range_argument(parser)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    prompt_pairs = load_prompt_pairs(arguments.task, arguments.model)
    start, stop = resolve_pair_range(arguments.pairs, len(prompt_pairs))

    model = load_model(arguments.model)
    effects = [head_effects(model, pair) for pair in prompt_pairs[start:stop]]
    save_pair_range(
        arguments.output_directory
        or head_effects_directory(OUTPUT_ROOT, arguments.model, arguments.task),
        start,
        stop,
        nie=np.stack([pair_effects.nie for pair_effects in effects]),
        pie=np.stack([pair_effects.pie for pair_effects in effects]),
    )


if __name__ == "__main__":
    main()

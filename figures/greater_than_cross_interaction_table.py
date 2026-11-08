"""Appendix C table: cross-interactions among the early MLPs of the Greater-Than circuit.

Reads the arrays of ``experiments/greater_than_cross_interactions.py`` and writes the table
of the six pairs among MLPs 0-3 under mean and resample ablation, as
``greater_than_cross_interaction_table.tex``, together with the numbers the appendix text
quotes (the drops in the logit difference under ablating MLP1, MLP2 and both) as
``greater_than_cross_interaction_numbers.json``.
"""

import argparse
import itertools
from pathlib import Path

from mediation.circuits import GREATER_THAN_EARLY_MLPS, GREATER_THAN_MLPS
from mediation.estimands import cross_interaction
from mediation.patching import MlpLayer
from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    load_pair_ranges,
    write_numbers,
    write_table,
)
from mediation.statistics import interval_summary, latex_mean_with_interval
from mediation.tasks import pair_count

ABLATIONS = ("mean", "resample")

TABLE_TEMPLATE = r"""\begin{{table}}[h]
\centering
\caption{{%
  Pairwise cross-interaction xINT among the early MLPs of the Greater-Than circuit in GPT-2 small, under mean ablation over the year-01 prompts and under resample ablation from each pair's year-01 prompt (mean $\pm$ 95\% CI over {pairs} prompt pairs, logit-difference units).
}}
\label{{tab:gt-within}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
Pair & Mean ablation & Resample \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def mlp_index(layer: int) -> int:
    """Position of the MLP of ``layer`` along the mediator axes of the arrays."""
    return GREATER_THAN_MLPS.index(MlpLayer(layer))


def pair_cross_interaction(arrays: dict, first_layer: int, second_layer: int):
    """xINT of the MLPs of two layers for every prompt pair, shape (pairs,), with the sign
    of Theorem 4.1: ``Y(A) + Y(B) - Y(A + B) - Y(empty)``."""
    first, second = mlp_index(first_layer), mlp_index(second_layer)
    return cross_interaction(
        arrays["y_clean"],
        arrays["y_single"][:, first],
        arrays["y_single"][:, second],
        arrays["y_joint"][:, first, second],
    )


def ablation_drops(arrays: dict, first_layer: int, second_layer: int) -> dict:
    """Per-pair drop ``Y(empty) - Y(S)`` of the logit difference when ablating the MLP of
    each layer alone and both together; arrays of shape (pairs,)."""
    first, second = mlp_index(first_layer), mlp_index(second_layer)
    return {
        f"MLP{first_layer}": arrays["y_clean"] - arrays["y_single"][:, first],
        f"MLP{second_layer}": arrays["y_clean"] - arrays["y_single"][:, second],
        "both": arrays["y_clean"] - arrays["y_joint"][:, first, second],
    }


def early_mlp_pairs() -> list[tuple[int, int]]:
    """The six pairs of layers among MLPs 0-3, in table order."""
    return list(
        itertools.combinations([mlp.layer for mlp in GREATER_THAN_EARLY_MLPS], 2)
    )


def table_rows(runs: dict) -> str:
    """One LaTeX row per early MLP pair, with its mean and resample ablation cells."""
    rows = []
    for first, second in early_mlp_pairs():
        cells = " & ".join(
            latex_mean_with_interval(
                pair_cross_interaction(runs[ablation], first, second)
            )
            for ablation in ABLATIONS
        )
        rows.append(rf"MLP{first} $\times$ MLP{second} & {cells} \\")
    return "\n".join(rows)


def quoted_numbers(runs: dict) -> dict:
    """Numbers of the appendix text: per-pair xINT, and the mean drops of ablating MLP1,
    MLP2 and both, for each ablation."""
    numbers = {"prompt_pairs": len(runs["mean"]["y_clean"])}
    for ablation, arrays in runs.items():
        cross_interactions = {}
        for first, second in early_mlp_pairs():
            cross_interactions[f"MLP{first} x MLP{second}"] = interval_summary(
                pair_cross_interaction(arrays, first, second)
            )
        drops = ablation_drops(arrays, 1, 2)
        joint_excess = drops["both"] - drops["MLP1"] - drops["MLP2"]
        sum_of_single_drops = drops["MLP1"].mean() + drops["MLP2"].mean()
        numbers[ablation] = {
            "cross_interaction": cross_interactions,
            "drop": {name: float(values.mean()) for name, values in drops.items()},
            "joint_drop_excess_over_sum_of_single_drops_percent": float(
                100 * joint_excess.mean() / sum_of_single_drops
            ),
        }
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    runs = {
        ablation: load_pair_ranges(
            arguments.output_root / "greater_than_cross_interactions" / ablation,
            pair_count("greater_than", "gpt2-small"),
        )
        for ablation in ABLATIONS
    }
    table = TABLE_TEMPLATE.format(
        pairs=len(runs["mean"]["y_clean"]), rows=table_rows(runs)
    )
    write_table(
        "greater_than_cross_interaction_table", table, arguments.paper_directory
    )
    write_numbers(
        "greater_than_cross_interaction",
        quoted_numbers(runs),
        arguments.paper_directory,
    )


if __name__ == "__main__":
    main()

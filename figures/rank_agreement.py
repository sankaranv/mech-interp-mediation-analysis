"""Figure 1: rank agreement between NIE and PIE for every task and model.

One mark is one model on one task: the Spearman correlation, over all attention heads of
the model, between each head's mean NIE and mean PIE across the task's prompt pairs. A
value of 1 means noising and denoising rank the heads identically, so INT leaves the
ranking unchanged. Greater-Than is not evaluated for Qwen2.5 and Llama-3.1, whose
tokenizers do not emit a two-digit year ending as one token.

Reads ``head_effects/<model>/<task>`` and writes ``paper/rank_agreement.{pdf,png}`` and
``paper/rank_agreement_numbers.json`` (the values Section 3 and the introduction quote).
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    head_effects_directory,
    load_pair_ranges,
    save_figure,
    write_numbers,
)
from mediation.statistics import rank_agreement
from mediation.tasks import pair_count

# (model, label, colour), grouped by family. A family shares a hue and darker means a
# larger model; the two largest GPT-2 and Pythia models stay distinguishable.
MODEL_FAMILIES = [
    [
        ("gpt2-small", "GPT-2 small", "#8cc4ee"),
        ("gpt2-medium", "GPT-2 medium", "#3d8ed3"),
        ("gpt2-xl", "GPT-2 XL", "#174f8f"),
    ],
    [
        ("pythia-70m", "Pythia-70M", "#8adbc2"),
        ("pythia-6.9b", "Pythia-6.9B", "#009E73"),
    ],
    [("qwen2.5-0.5b", "Qwen2.5-0.5B", "#E69F00")],
    [("llama-3.1-8b", "Llama-3.1-8B", "#CC79A7")],
]
MODEL_STYLES = [style for family in MODEL_FAMILIES for style in family]

# Ordered by GPT-2 small rank agreement: from the least to the most distant counterfactual.
TASK_LABELS = [
    ("sva", "SVA"),
    ("ioi_symmetric", "IOI (symmetric)"),
    ("gender_bias", "Gender Bias"),
    ("greater_than", "Greater-Than"),
    ("ioi_pabc", "IOI (pABC)"),
]

FAMILY_GAP = 0.6  # extra space between model families, in units of one model slot
GROUP_WIDTH = 0.7  # fraction of the spacing between tasks taken by one task's models


def load_rank_agreements(output_root: Path) -> np.ndarray:
    """Rank agreement of every task (rows) and model (columns), NaN where the task was not
    evaluated for the model."""
    agreement = np.full((len(TASK_LABELS), len(MODEL_STYLES)), np.nan)
    for row, (task, _) in enumerate(TASK_LABELS):
        for column, (model, _, _) in enumerate(MODEL_STYLES):
            directory = head_effects_directory(output_root, model, task)
            if directory.exists():
                effects = load_pair_ranges(directory, pair_count(task, model))
                agreement[row, column] = rank_agreement(effects["nie"], effects["pie"])
    return agreement


def model_offsets() -> np.ndarray:
    """Horizontal offset of each model from the centre of its task, with a gap between
    model families."""
    family_start_gaps = [
        FAMILY_GAP if position == 0 and family_index > 0 else 0.0
        for family_index, family in enumerate(MODEL_FAMILIES)
        for position in range(len(family))
    ]
    slots = np.cumsum(family_start_gaps) + np.arange(len(MODEL_STYLES))
    return (slots - slots.mean()) / (slots.max() - slots.min()) * GROUP_WIDTH


def plot(agreement: np.ndarray):
    """Figure 1 for the rank agreement matrix of :func:`load_rank_agreements`."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
        }
    )
    # Axes span the full text width; the legend sits in the strip below them.
    figure, axes = plt.subplots(figsize=(5.5, 2.3))
    figure.subplots_adjust(left=0.115, right=0.995, top=0.97, bottom=0.23)
    centers = np.arange(len(TASK_LABELS))
    offsets = model_offsets()

    # Thin light rules between tasks, so each task's models read as a group.
    for boundary in centers[:-1] + 0.5:
        axes.axvline(boundary, color="0.85", linewidth=0.6, zorder=0)
    for column, (_, _, colour) in enumerate(MODEL_STYLES):
        axes.scatter(
            centers + offsets[column],
            agreement[:, column],
            s=30,
            color=colour,
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

    # Rank agreement 1: noising and denoising rank the heads identically.
    axes.axhline(1.0, linestyle="--", color="0.5", linewidth=0.8, zorder=1)
    axes.set_xticks(centers, [label for _, label in TASK_LABELS])
    axes.tick_params(axis="x", length=0)
    axes.set_xlim(centers[0] - 0.5, centers[-1] + 0.5)
    axes.set_ylabel(r"$\rho(\overline{\mathrm{NIE}},\, \overline{\mathrm{PIE}})$")
    axes.set_ylim(0.0, 1.05)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colour,
            alpha=0.9,
            markersize=6,
            label=label,
        )
        for _, label, colour in MODEL_STYLES
    ]
    figure.legend(
        handles=handles,
        ncol=len(MODEL_STYLES),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handletextpad=0.05,
        columnspacing=0.5,
    )
    return figure


def quoted_numbers(agreement: np.ndarray) -> dict:
    """The rank agreements the paper text quotes.

    Section 3 quotes GPT-2 small on every task and the lowest value on IOI (pABC) and on
    Greater-Than; the introduction quotes the lowest value on IOI (pABC).
    """
    tasks = [task for task, _ in TASK_LABELS]
    models = [model for model, _, _ in MODEL_STYLES]

    def lowest(task: str) -> dict | None:
        row = agreement[tasks.index(task)]
        if np.isnan(row).all():
            return None
        column = int(np.nanargmin(row))
        return {"model": models[column], "rank_agreement": float(row[column])}

    return {
        "rank_agreement": {
            task: {
                model: None if np.isnan(value) else float(value)
                for model, value in zip(models, row, strict=True)
            }
            for task, row in zip(tasks, agreement, strict=True)
        },
        "gpt2_small": {
            task: float(row[0]) for task, row in zip(tasks, agreement, strict=True)
        },
        "lowest_ioi_pabc": lowest("ioi_pabc"),
        "lowest_greater_than": lowest("greater_than"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    agreement = load_rank_agreements(arguments.output_root)
    directory = arguments.paper_directory
    figure = plot(agreement)
    save_figure(figure, "rank_agreement", directory, pad_inches=0.01)
    plt.close(figure)
    write_numbers("rank_agreement", quoted_numbers(agreement), directory)


if __name__ == "__main__":
    main()

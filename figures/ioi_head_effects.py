"""Figure 3: NIE, PIE and INT of the attention heads of GPT-2 small on IOI with pABC.

Left: each head's mean NIE against its mean PIE. A head below the diagonal has INT < 0
(backup compensation), a head above it INT > 0. Right: each head's mean INT, one strip per
functional group of Wang et al., with the group median marked. NIE is a resample from the
pair's pABC prompt and PIE a denoising from it, so INT = NIE - PIE is Proposition 3.1's.

Reads ``head_effects/gpt2-small/ioi_pabc`` and writes ``paper/ioi_head_effects.{pdf,png}``
and ``paper/ioi_head_effects_numbers.json`` (the values Section 5 quotes).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mediation.circuits import (
    BACKUP_NAME_MOVER,
    DUPLICATE_TOKEN,
    GPT2_SMALL_HEADS,
    INDUCTION,
    IOI_ROLE_ABBREVIATIONS,
    IOI_ROLE_COLORS,
    IOI_ROLE_ORDER,
    NAME_MOVER,
    NEGATIVE_NAME_MOVER,
    OTHER,
    PREVIOUS_TOKEN,
    S_INHIBITION,
    ioi_role,
)
from mediation.patching import AttentionHead
from mediation.plotting import add_ioi_role_legend
from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    head_effects_directory,
    load_pair_ranges,
    save_figure,
    write_numbers,
)

MODEL = "gpt2-small"

# Order in which the scatter draws the groups; Duplicate Token is last so that it is
# visible above Induction.
SCATTER_ORDER = [
    NAME_MOVER,
    BACKUP_NAME_MOVER,
    NEGATIVE_NAME_MOVER,
    S_INHIBITION,
    INDUCTION,
    PREVIOUS_TOKEN,
    DUPLICATE_TOKEN,
]

# Heads labelled in the scatter: (dx, dy) offset of the label from the point in data units
# and its horizontal alignment.
LABELLED_HEADS = {
    AttentionHead(9, 9): ((-0.12, +0.10), "right"),
    AttentionHead(8, 6): ((+0.08, +0.06), "left"),
    AttentionHead(10, 7): ((+0.12, +0.10), "left"),
    AttentionHead(3, 0): ((-0.08, +0.07), "right"),
}

STRIP_LABELS = {**IOI_ROLE_ABBREVIATIONS, INDUCTION: "Ind."}
JITTER = 0.18  # half-width of the horizontal jitter of the points of a strip
JITTER_SEED = 0


def draw_scatter(axes: plt.Axes, mean_pie: np.ndarray, mean_nie: np.ndarray) -> None:
    """Mean NIE against mean PIE of every head, coloured by role, with equal axes."""
    bound = float(max(np.abs(mean_pie).max(), np.abs(mean_nie).max()) * 1.08)
    axes.plot(
        [-bound, bound], [-bound, bound], color="black", lw=0.8, ls=":", alpha=0.6
    )
    axes.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4, zorder=1)
    axes.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4, zorder=1)

    def heads_with_role(role: str) -> list[AttentionHead]:
        return [head for head in GPT2_SMALL_HEADS if ioi_role(head) == role]

    # Heads outside the circuit are small and dim and drawn first.
    outside = heads_with_role(OTHER)
    axes.scatter(
        [mean_pie[head] for head in outside],
        [mean_nie[head] for head in outside],
        color=IOI_ROLE_COLORS[OTHER],
        s=14,
        alpha=0.45,
        linewidths=0,
        zorder=2,
    )
    for role in SCATTER_ORDER:
        members = heads_with_role(role)
        axes.scatter(
            [mean_pie[head] for head in members],
            [mean_nie[head] for head in members],
            color=IOI_ROLE_COLORS[role],
            s=32,
            alpha=0.88,
            linewidths=0,
            zorder=3,
        )
    for head, ((dx, dy), alignment) in LABELLED_HEADS.items():
        label = f"{head.name}\n({IOI_ROLE_ABBREVIATIONS[ioi_role(head)]})"
        axes.text(
            mean_pie[head] + dx,
            mean_nie[head] + dy,
            label,
            fontsize=6,
            ha=alignment,
            va="bottom",
            zorder=5,
        )

    axes.set_xlim(-bound, bound)
    axes.set_ylim(-bound, bound)
    axes.set_aspect("equal")
    axes.set_xlabel("Mean PIE (logit-diff)", fontsize=9)
    axes.set_ylabel("Mean NIE (logit-diff)", fontsize=9)
    sns.despine(ax=axes)


def draw_strips(axes: plt.Axes, mean_interaction: np.ndarray) -> None:
    """Mean INT of every head in one strip per role, with the role's median as a bar."""
    generator = np.random.default_rng(JITTER_SEED)
    for position, role in enumerate(IOI_ROLE_ORDER):
        values = np.array(
            [
                mean_interaction[head]
                for head in GPT2_SMALL_HEADS
                if ioi_role(head) == role
            ]
        )
        horizontal = position + generator.uniform(-JITTER, JITTER, size=len(values))
        axes.scatter(
            horizontal,
            values,
            color=IOI_ROLE_COLORS[role],
            s=32,
            alpha=0.80,
            linewidths=0,
            zorder=3,
        )
        axes.hlines(
            np.median(values),
            position - 0.28,
            position + 0.28,
            colors="black",
            linewidths=0.9,
            zorder=4,
        )
    axes.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4, zorder=2)
    axes.set_xticks(range(len(IOI_ROLE_ORDER)))
    axes.set_xticklabels(
        [STRIP_LABELS[role] for role in IOI_ROLE_ORDER],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    axes.set_ylabel("Mean INT (logit-diff)", fontsize=9)
    axes.set_xlim(-0.6, len(IOI_ROLE_ORDER) - 0.4)
    sns.despine(ax=axes)


def plot(nie: np.ndarray, pie: np.ndarray):
    """Figure 3 for per-pair effects of shape (pairs, 12, 12)."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {"axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7}
    )
    figure, (scatter_axes, strip_axes) = plt.subplots(
        1, 2, figsize=(9.0, 4.4), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    draw_scatter(scatter_axes, pie.mean(axis=0), nie.mean(axis=0))
    draw_strips(strip_axes, (nie - pie).mean(axis=0))
    add_ioi_role_legend(figure, anchor_y=-0.01, named_alpha=0.88, other_alpha=0.45)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.20)
    return figure


def quoted_numbers(nie: np.ndarray, pie: np.ndarray) -> dict:
    """Section 5's numbers from this figure: the means of L9H9, the PIE range of the
    S-Inhibition heads and the PIE of the Negative Name Mover Heads."""
    mean_nie, mean_pie = nie.mean(axis=0), pie.mean(axis=0)
    mean_interaction = (nie - pie).mean(axis=0)
    name_mover = AttentionHead(9, 9)
    s_inhibition = [head for head in GPT2_SMALL_HEADS if ioi_role(head) == S_INHIBITION]
    negative = [
        head for head in GPT2_SMALL_HEADS if ioi_role(head) == NEGATIVE_NAME_MOVER
    ]
    s_inhibition_pie = [float(mean_pie[head]) for head in s_inhibition]
    return {
        "L9H9": {
            "pie": float(mean_pie[name_mover]),
            "int": float(mean_interaction[name_mover]),
            "nie": float(mean_nie[name_mover]),
        },
        "s_inhibition_pie_range": [min(s_inhibition_pie), max(s_inhibition_pie)],
        "negative_name_mover_pie": {
            head.name: float(mean_pie[head]) for head in negative
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    effects = load_pair_ranges(
        head_effects_directory(arguments.output_root, MODEL, "ioi_pabc")
    )
    directory = arguments.paper_directory
    figure = plot(effects["nie"], effects["pie"])
    save_figure(figure, "ioi_head_effects", directory)
    plt.close(figure)
    write_numbers(
        "ioi_head_effects", quoted_numbers(effects["nie"], effects["pie"]), directory
    )


if __name__ == "__main__":
    main()

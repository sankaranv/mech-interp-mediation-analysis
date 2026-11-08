"""Figure 4: NIE and PIE rankings of the 144 attention heads of GPT-2 small on IOI with pABC.

Each panel ranks all heads by their signed mean, largest first, and marks the boundary of
the 26-head circuit of Wang et al. Left, NIE: the four Backup Name Mover Heads that behave
like Name Mover Heads are outlined; backup compensation buries them although they are
functionally critical. Right, PIE: Duplicate Token, Induction and Previous Token Heads are
outlined; they contribute only through INT, so PIE places them near zero.

Reads ``head_effects/gpt2-small/{ioi_pabc,ioi_symmetric}`` and writes
``paper/ioi_rankings.{pdf,png}`` and ``paper/ioi_rankings_numbers.json`` (the ranks
Section 5 quotes, including the Induction Head ranks under the symmetric counterfactual).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mediation.circuits import (
    DUPLICATE_TOKEN,
    GPT2_SMALL_HEADS,
    INDUCTION,
    IOI_ROLE_COLORS,
    OTHER,
    PREVIOUS_TOKEN,
    ioi_circuit_heads,
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
from mediation.statistics import signed_ranks
from mediation.tasks import pair_count

MODEL = "gpt2-small"

# The four Backup Name Mover Heads that Wang et al. (Appendix) describe as behaving like
# Name Mover Heads.
NAME_MOVER_LIKE_BACKUPS = [
    AttentionHead(9, 0),
    AttentionHead(10, 1),
    AttentionHead(10, 6),
    AttentionHead(10, 10),
]
# Heads that contribute through INT only: PIE places them near zero.
CONTEXT_SPECIFIC_ROLES = {DUPLICATE_TOKEN, INDUCTION, PREVIOUS_TOKEN}
CIRCUIT_SIZE = len(ioi_circuit_heads())  # rank at which the circuit boundary is drawn


def draw_ranking(
    axes: plt.Axes,
    mean_effect: np.ndarray,
    outlined: set[AttentionHead],
    xlabel: str,
    ylabel: str,
) -> None:
    """Scatter of the signed mean of every head against its rank, largest first."""
    ranks = signed_ranks(mean_effect)
    # Three tiers: heads outside the circuit small and dim, circuit heads opaque, outlined
    # heads on top with a black edge.
    for head in sorted(GPT2_SMALL_HEADS, key=lambda head: ranks[head]):
        role = ioi_role(head)
        if role == OTHER:
            style = {"s": 18, "alpha": 0.25, "linewidths": 0, "zorder": 1}
        elif head in outlined:
            style = {
                "s": 48,
                "alpha": 0.9,
                "edgecolors": "black",
                "linewidths": 0.9,
                "zorder": 3,
            }
        else:
            style = {"s": 40, "alpha": 0.9, "linewidths": 0, "zorder": 2}
        axes.scatter(
            ranks[head], mean_effect[head], color=IOI_ROLE_COLORS[role], **style
        )
    axes.axvline(CIRCUIT_SIZE, color="gray", lw=0.8, ls="--", alpha=0.55, zorder=1)
    axes.set_xlabel(xlabel, fontsize=10)
    axes.set_ylabel(ylabel, fontsize=10)
    axes.set_xlim(-3, 149)
    sns.despine(ax=axes)


def plot(nie: np.ndarray, pie: np.ndarray):
    """Figure 4 for per-pair effects of shape (pairs, 12, 12)."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {"axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8}
    )
    figure, (nie_axes, pie_axes) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    draw_ranking(
        nie_axes,
        nie.mean(axis=0),
        set(NAME_MOVER_LIKE_BACKUPS),
        "Rank by NIE (descending)",
        "NIE (logit-diff)",
    )
    draw_ranking(
        pie_axes,
        pie.mean(axis=0),
        {head for head in GPT2_SMALL_HEADS if ioi_role(head) in CONTEXT_SPECIFIC_ROLES},
        "Rank by PIE (descending)",
        "PIE (logit-diff)",
    )
    add_ioi_role_legend(figure, anchor_y=0.01, named_alpha=0.9, other_alpha=0.4)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.22)
    return figure


def rank_range(ranks: np.ndarray, heads: list[AttentionHead]) -> list[int]:
    """Smallest and largest rank among ``heads``."""
    head_ranks = [int(ranks[head]) for head in heads]
    return [min(head_ranks), max(head_ranks)]


def quoted_numbers(nie: np.ndarray, pie: np.ndarray, symmetric_nie: np.ndarray) -> dict:
    """Section 5's ranks: the backup tier, the context-specific heads, L9H9 and L5H5, and
    where the symmetric counterfactual's NIE ranking puts the Induction Heads and L0H10.

    :param symmetric_nie: per-pair NIE with the symmetric counterfactual.
    """
    nie_ranks = signed_ranks(nie.mean(axis=0))
    pie_ranks = signed_ranks(pie.mean(axis=0))
    symmetric_ranks = signed_ranks(symmetric_nie.mean(axis=0))
    context_specific = [
        head for head in ioi_circuit_heads() if ioi_role(head) in CONTEXT_SPECIFIC_ROLES
    ]
    induction = [head for head in ioi_circuit_heads() if ioi_role(head) == INDUCTION]
    name_mover = AttentionHead(9, 9)
    induction_start = AttentionHead(5, 5)
    duplicate_token_uncertain = AttentionHead(0, 10)
    return {
        "name_mover_like_backups": {
            "pie_rank_range": rank_range(pie_ranks, NAME_MOVER_LIKE_BACKUPS),
            "nie_rank_range": rank_range(nie_ranks, NAME_MOVER_LIKE_BACKUPS),
        },
        "L9H9": {
            "pie_rank": int(pie_ranks[name_mover]),
            "nie_rank": int(nie_ranks[name_mover]),
        },
        "context_specific_heads": {
            "count": len(context_specific),
            "nie_rank_range": rank_range(nie_ranks, context_specific),
            "pie_rank_range": rank_range(pie_ranks, context_specific),
            "number_in_nie_top_half_of_circuit": sum(
                int(nie_ranks[head]) <= CIRCUIT_SIZE // 2 for head in context_specific
            ),
            "inside_pie_circuit_boundary": [
                head.name
                for head in context_specific
                if int(pie_ranks[head]) <= CIRCUIT_SIZE
            ],
        },
        "L5H5": {
            "pie_rank": int(pie_ranks[induction_start]),
            "nie_rank": int(nie_ranks[induction_start]),
            "mean_pie": float(pie.mean(axis=0)[induction_start]),
        },
        "symmetric_counterfactual_nie_rank": {
            head.name: int(symmetric_ranks[head])
            for head in [*induction, duplicate_token_uncertain]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    effects = load_pair_ranges(
        head_effects_directory(arguments.output_root, MODEL, "ioi_pabc"),
        pair_count("ioi_pabc", MODEL),
    )
    symmetric = load_pair_ranges(
        head_effects_directory(arguments.output_root, MODEL, "ioi_symmetric"),
        pair_count("ioi_symmetric", MODEL),
    )
    directory = arguments.paper_directory
    figure = plot(effects["nie"], effects["pie"])
    save_figure(figure, "ioi_rankings", directory)
    plt.close(figure)
    write_numbers(
        "ioi_rankings",
        quoted_numbers(effects["nie"], effects["pie"], symmetric["nie"]),
        directory,
    )


if __name__ == "__main__":
    main()

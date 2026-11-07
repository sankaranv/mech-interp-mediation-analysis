"""Appendix B tables of IOI cross-interactions and the Section 5.3 numbers quoted from them.

Reads ``ioi_cross_interactions`` (see ``experiments/ioi_cross_interactions.py``) and
``head_effects/gpt2-small/ioi_pabc``. All values are by resample from each pair's pABC
prompt. Every cross-interaction has the sign of Theorem 4.1,
``xINT(A, B) = Y(A) + Y(B) - Y(A + B) - Y(empty)``, so that
``NIE(A + B) = NIE(A) + NIE(B) + xINT(A, B)``; the effect of ``B`` once ``A`` is knocked
out is ``NIE(B | A) = Y(A) - Y(A + B) = NIE(B) + xINT(A, B)``.

Writes ``paper/ioi_backup_name_mover_table.tex`` (tab:ioi-bnmh),
``paper/ioi_s_inhibition_cross_interaction_table.tex`` (tab:sinh-xint) and
``paper/ioi_cross_interaction_tables_numbers.json``.

Run ``python figures/ioi_cross_interaction_tables.py``.
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.circuits import (
    BACKUP_NAME_MOVER_HEADS,
    DUPLICATE_TOKEN,
    INDUCTION,
    IOI_ROLE_ABBREVIATIONS,
    NAME_MOVER_GROUP,
    NEGATIVE_NAME_MOVER,
    NON_NAME_MOVER_HEADS,
    PREVIOUS_TOKEN,
    S_INHIBITION_HEADS,
    S_INHIBITION_PARTNER_HEADS,
    ioi_heads_with_roles,
    table_head_name,
)
from mediation.estimands import cross_interaction
from mediation.patching import AttentionHead
from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    head_effects_directory,
    load_pair_ranges,
    write_numbers,
    write_table,
)
from mediation.statistics import (
    latex_mean_with_interval,
    mean_with_confidence_interval,
    signed_ranks,
)

MODEL = "gpt2-small"

# Roles of the S-Inhibition partners, left to right in tab:sinh-xint.
PARTNER_ROLES = (DUPLICATE_TOKEN, INDUCTION, PREVIOUS_TOKEN, NEGATIVE_NAME_MOVER)

# Behaviour of each Backup Name Mover Head as Wang et al. describe it.
WANG_PROFILES = {
    AttentionHead(10, 10): "name-mover-like",
    AttentionHead(10, 6): "name-mover-like",
    AttentionHead(10, 1): "name-mover-like",
    AttentionHead(9, 7): "attends to S2; writes negatively",
    AttentionHead(10, 2): "attends equally to IO and S; writes both",
    AttentionHead(9, 0): "name-mover-like",
    AttentionHead(11, 9): "attends equally to IO and S; writes both",
    AttentionHead(11, 2): r"attends to S1; writes in $W_U[S]$ direction",
}

BACKUP_TABLE_HEADER = r"""\begin{tabular}{@{}lrrrrrrp{3cm}@{}}
\toprule
Head & PIE $\pm$ CI & PIE rank & NIE $\pm$ CI & NIE rank
  & xINT(L9H9) $\pm$ CI & xINT(NMH) $\pm$ CI
  & \citet{wang2023interpretability} profile \\
\midrule"""
TABLE_FOOTER = "\\bottomrule\n\\end{tabular}"


def grid_cross_interaction(arrays: dict, grid: str) -> np.ndarray:
    """xINT of every first set with every second set of a grid, shape (pairs, A, B)."""
    return cross_interaction(
        arrays[f"{grid}_y_empty"][:, None, None],
        arrays[f"{grid}_y_first"][:, :, None],
        arrays[f"{grid}_y_second"][:, None, :],
        arrays[f"{grid}_y_both"],
    )


def backup_table(arrays: dict, nie: np.ndarray, pie: np.ndarray) -> str:
    """tab:ioi-bnmh: the Backup Name Mover Heads, sorted by mean PIE, with their
    cross-interaction with L9H9 and with the Name Mover group."""
    nie_ranks = signed_ranks(nie.mean(axis=0))
    pie_ranks = signed_ranks(pie.mean(axis=0))
    with_l9h9 = grid_cross_interaction(arrays, "l9h9_backup")[:, 0]
    with_group = grid_cross_interaction(arrays, "name_mover_group")[:, 0]
    rows = []
    for head in sorted(
        BACKUP_NAME_MOVER_HEADS, key=lambda head: -pie[:, head.layer, head.head].mean()
    ):
        layer, index = head
        rows.append(
            f"{table_head_name(head)} & {latex_mean_with_interval(pie[:, layer, index])} "
            f"& {pie_ranks[head]} "
            f"& {latex_mean_with_interval(nie[:, layer, index])} & {nie_ranks[head]} "
            f"& {latex_mean_with_interval(with_l9h9[:, BACKUP_NAME_MOVER_HEADS.index(head)])} "
            f"& {latex_mean_with_interval(with_group[:, NON_NAME_MOVER_HEADS.index(head)])} "
            f"& {WANG_PROFILES[head]} \\\\"
        )
    return "\n".join([BACKUP_TABLE_HEADER, *rows, TABLE_FOOTER])


def inhibition_table_header() -> str:
    """Header of tab:sinh-xint: partner roles over their head columns."""
    sizes = [len(ioi_heads_with_roles(role)) for role in PARTNER_ROLES]
    groups = " & ".join(
        f"\\multicolumn{{{size}}}{{c}}{{{IOI_ROLE_ABBREVIATIONS[role]}}}"
        for role, size in zip(PARTNER_ROLES, sizes, strict=True)
    )
    # A rule under each group, spanning its columns (the first column holds the row heads).
    rules, first_column = [], 2
    for size in sizes:
        last_column = first_column + size - 1
        trim = "(l)" if last_column == 1 + sum(sizes) else "(lr)"
        rules.append(f"\\cmidrule{trim}{{{first_column}-{last_column}}}")
        first_column = last_column + 1
    columns = " & ".join(table_head_name(head) for head in S_INHIBITION_PARTNER_HEADS)
    return "\n".join(
        [
            f"\\begin{{tabular}}{{@{{}}l*{{{len(S_INHIBITION_PARTNER_HEADS)}}}{{r}}@{{}}}}",
            "\\toprule",
            f" & {groups} \\\\",
            " ".join(rules),
            f"S-Inh head & {columns} \\\\",
            "\\midrule",
        ]
    )


def inhibition_table(arrays: dict) -> str:
    """tab:sinh-xint: mean xINT of each S-Inhibition head with each partner head."""
    means = grid_cross_interaction(arrays, "s_inhibition").mean(axis=0)
    rows = [
        f"{table_head_name(head)} & "
        + " & ".join(f"${value:+.3f}$" for value in means[row])
        + " \\\\"
        for row, head in enumerate(S_INHIBITION_HEADS)
    ]
    return "\n".join([inhibition_table_header(), *rows, TABLE_FOOTER])


def inhibition_numbers(arrays: dict) -> dict:
    """Section 5.3, serial dependency and conditional harm."""
    cross = grid_cross_interaction(arrays, "s_inhibition")
    interval = mean_with_confidence_interval(cross)
    upstream = [
        index
        for index, head in enumerate(S_INHIBITION_PARTNER_HEADS)
        if head not in ioi_heads_with_roles(NEGATIVE_NAME_MOVER)
    ]
    negative = [
        index
        for index, head in enumerate(S_INHIBITION_PARTNER_HEADS)
        if head in ioi_heads_with_roles(NEGATIVE_NAME_MOVER)
    ]
    upstream_means = interval.mean[:, upstream]
    negative_means = interval.mean[:, negative]

    # NIE of a Negative Name Mover Head alone and once L8H6 is knocked out.
    l8h6 = S_INHIBITION_HEADS.index(AttentionHead(8, 6))
    y_empty = arrays["s_inhibition_y_empty"]
    nie_alone = y_empty[:, None] - arrays["s_inhibition_y_second"]
    nie_without_l8h6 = (
        arrays["s_inhibition_y_first"][:, l8h6, None]
        - arrays["s_inhibition_y_both"][:, l8h6]
    )
    negative_heads = ioi_heads_with_roles(NEGATIVE_NAME_MOVER)
    return {
        "serial_dependency_pairs": int(upstream_means.size),
        "serial_dependency_all_negative": bool((upstream_means < 0).all()),
        "serial_dependency_closest_to_zero": float(upstream_means.max()),
        "serial_dependency_most_negative": float(upstream_means.min()),
        "serial_dependency_all_intervals_exclude_zero": bool(
            (
                np.abs(interval.mean[:, upstream]) > interval.half_width[:, upstream]
            ).all()
        ),
        "negative_name_mover_pairs": int(negative_means.size),
        "negative_name_mover_all_positive": bool((negative_means > 0).all()),
        "negative_name_mover_all_intervals_exclude_zero": bool(
            (
                np.abs(interval.mean[:, negative]) > interval.half_width[:, negative]
            ).all()
        ),
        "largest_interval_half_width": float(interval.half_width.max()),
        "l8h6_with_l10h7": float(
            interval.mean[l8h6, S_INHIBITION_PARTNER_HEADS.index(AttentionHead(10, 7))]
        ),
        "l8h6_with_l5h5": float(
            interval.mean[l8h6, S_INHIBITION_PARTNER_HEADS.index(AttentionHead(5, 5))]
        ),
        "negative_name_mover_nie_alone": {
            head.name: float(
                nie_alone[:, S_INHIBITION_PARTNER_HEADS.index(head)].mean()
            )
            for head in negative_heads
        },
        "negative_name_mover_nie_with_l8h6_knocked_out": {
            head.name: float(
                nie_without_l8h6[:, S_INHIBITION_PARTNER_HEADS.index(head)].mean()
            )
            for head in negative_heads
        },
    }


def name_mover_numbers(arrays: dict) -> dict:
    """Section 5.3, two kinds of backup: the NIE of a head alone and once the Name Mover
    group is knocked out, and their difference, the cross-interaction with the group."""
    grid = "name_mover_group"
    nie_alone = arrays[f"{grid}_y_empty"][:, None] - arrays[f"{grid}_y_second"]
    nie_with_group_knocked_out = (
        arrays[f"{grid}_y_first"] - arrays[f"{grid}_y_both"][:, 0]
    )
    cross = grid_cross_interaction(arrays, grid)[:, 0]

    def per_head(heads: list[AttentionHead]) -> dict:
        columns = {head: NON_NAME_MOVER_HEADS.index(head) for head in heads}
        return {
            head.name: {
                "nie_alone": float(nie_alone[:, column].mean()),
                "nie_with_name_movers_knocked_out": float(
                    nie_with_group_knocked_out[:, column].mean()
                ),
                "cross_interaction_with_name_movers": float(cross[:, column].mean()),
            }
            for head, column in columns.items()
        }

    # The Backup Name Mover Heads, and Negative Name Mover Head L10H7, whose NIE turns
    # positive once the Name Mover Heads are knocked out.
    quoted_heads = [*BACKUP_NAME_MOVER_HEADS, AttentionHead(10, 7)]
    return {
        "name_mover_group": [head.name for head in NAME_MOVER_GROUP],
        "heads": per_head(quoted_heads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    root = arguments.output_root
    arrays = load_pair_ranges(root / "ioi_cross_interactions")
    effects = load_pair_ranges(head_effects_directory(root, MODEL, "ioi_pabc"))
    directory = arguments.paper_directory
    write_table(
        "ioi_backup_name_mover_table",
        backup_table(arrays, effects["nie"], effects["pie"]),
        directory,
    )
    write_table(
        "ioi_s_inhibition_cross_interaction_table", inhibition_table(arrays), directory
    )
    write_numbers(
        "ioi_cross_interaction_tables",
        {
            "s_inhibition": inhibition_numbers(arrays),
            "name_mover_group": name_mover_numbers(arrays),
        },
        directory,
    )


if __name__ == "__main__":
    main()

"""Appendix B table of NIE, PIE and INT for all 26 circuit heads of GPT-2 small on IOI.

Every entry is a mean over the prompt pairs with its 95% confidence interval
(1.96 * SD / sqrt(N)). Ranks are signed ranks of the mean over all 144 heads, 1 for the
largest. Rows are grouped by role in the order of Wang et al., and sorted by mean PIE
within a group.

Reads ``head_effects/gpt2-small/ioi_pabc`` and writes ``paper/ioi_estimand_table.tex``
(a ``tabular``) and ``paper/ioi_estimand_table_numbers.json`` (the number of heads whose
per-pair PIE has a standard deviation above its absolute mean, quoted by the appendix).
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.circuits import (
    IOI_ROLE_ABBREVIATIONS,
    IOI_ROLE_ORDER,
    OTHER,
    ioi_circuit_heads,
    ioi_heads_with_roles,
    table_head_name,
)
from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    head_effects_directory,
    load_pair_ranges,
    write_numbers,
    write_table,
)
from mediation.statistics import latex_mean_with_interval, signed_ranks

MODEL = "gpt2-small"

TABLE_HEADER = r"""\begin{tabular}{@{}llrrrrr@{}}
\toprule
Head & Group
  & \multicolumn{1}{c}{PIE $\pm$ CI} & \multicolumn{1}{c}{PIE rank}
  & \multicolumn{1}{c}{NIE $\pm$ CI} & \multicolumn{1}{c}{NIE rank}
  & \multicolumn{1}{c}{INT $\pm$ CI} \\
\midrule"""
TABLE_FOOTER = "\\bottomrule\n\\end{tabular}"


def table_rows(nie: np.ndarray, pie: np.ndarray) -> list[str]:
    """LaTeX rows of the 26 circuit heads, a ``\\midrule`` after each role."""
    interaction = nie - pie
    nie_ranks = signed_ranks(nie.mean(axis=0))
    pie_ranks = signed_ranks(pie.mean(axis=0))
    rows = []
    for role in IOI_ROLE_ORDER:
        if role == OTHER:
            continue
        members = ioi_heads_with_roles(role)
        for head in sorted(
            members, key=lambda head: -pie[:, head.layer, head.head].mean()
        ):
            layer, index = head
            rows.append(
                f"{table_head_name(head)} & {IOI_ROLE_ABBREVIATIONS[role]} "
                f"& {latex_mean_with_interval(pie[:, layer, index])} & {pie_ranks[head]} "
                f"& {latex_mean_with_interval(nie[:, layer, index])} & {nie_ranks[head]} "
                f"& {latex_mean_with_interval(interaction[:, layer, index])} \\\\"
            )
        rows.append(r"\midrule")
    return rows[:-1]


def heads_with_pie_sd_above_mean(pie: np.ndarray) -> int:
    """Number of circuit heads whose per-pair PIE has a sample standard deviation above
    the absolute value of its mean."""
    spread = pie.std(axis=0, ddof=1)
    magnitude = np.abs(pie.mean(axis=0))
    return sum(bool(spread[head] > magnitude[head]) for head in ioi_circuit_heads())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    effects = load_pair_ranges(
        head_effects_directory(arguments.output_root, MODEL, "ioi_pabc")
    )
    nie, pie = effects["nie"], effects["pie"]
    directory = arguments.paper_directory
    table = "\n".join([TABLE_HEADER, *table_rows(nie, pie), TABLE_FOOTER])
    write_table("ioi_estimand_table", table, directory)
    write_numbers(
        "ioi_estimand_table",
        {
            "circuit_heads": len(ioi_circuit_heads()),
            "heads_with_pie_sd_above_absolute_mean": heads_with_pie_sd_above_mean(pie),
        },
        directory,
    )


if __name__ == "__main__":
    main()

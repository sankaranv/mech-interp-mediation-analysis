"""Section 5.3 and Appendix B: cross-interactions between groups of IOI heads, by resample.

For every IOI pABC prompt pair of GPT-2 small the clean prompt is run with sets of heads
replaced, at every position, by their activations on the pair's pABC prompt. Three grids of
set pairs are evaluated, each with :func:`mediation.estimands.joint_outcome_grid`:

``s_inhibition``
    each S-Inhibition head (4) with each Duplicate Token, Induction, Previous Token and
    Negative Name Mover head (11);
``l9h9_backup``
    the Name Mover head L9H9 with each Backup Name Mover head (8);
``name_mover_group``
    the Name Mover group {L9H6, L9H9, L10H0}, replaced jointly, with each of the other 141
    heads.

The heads of each grid are the constants of :mod:`mediation.circuits`, in the order of the
arrays, which is the order of the paper's tables; the other 141 heads follow in layer and
head order.

Writes, per pair range and grid ``g``: ``<g>_y_empty`` (pairs,), ``<g>_y_first`` (pairs, A),
``<g>_y_second`` (pairs, B) and ``<g>_y_both`` (pairs, A, B), the metrics of
:class:`mediation.estimands.OutcomeGrid`.

Run ``python experiments/ioi_cross_interactions.py``; ``--pairs start:stop`` restricts it to a range.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mediation.circuits import (
    BACKUP_NAME_MOVER_HEADS,
    NAME_MOVER_GROUP,
    NON_NAME_MOVER_HEADS,
    S_INHIBITION_HEADS,
    S_INHIBITION_PARTNER_HEADS,
)
from mediation.estimands import joint_outcome_grid
from mediation.models import load_model
from mediation.patching import AttentionHead, Component, cache_activations
from mediation.results import (
    OUTPUT_ROOT,
    add_pair_range_argument,
    resolve_pair_range,
    save_pair_range,
)
from mediation.tasks import load_prompt_pairs

MODEL = "gpt2-small"

L9H9 = AttentionHead(9, 9)


def grids() -> dict[str, tuple[list[Sequence[Component]], list[Sequence[Component]]]]:
    """For each grid, the list of first sets and the list of second sets of heads."""
    return {
        "s_inhibition": (
            [[head] for head in S_INHIBITION_HEADS],
            [[head] for head in S_INHIBITION_PARTNER_HEADS],
        ),
        "l9h9_backup": (
            [[L9H9]],
            [[head] for head in BACKUP_NAME_MOVER_HEADS],
        ),
        # The three Name Mover Heads are one mediator.
        "name_mover_group": (
            [NAME_MOVER_GROUP],
            [[head] for head in NON_NAME_MOVER_HEADS],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_pair_range_argument(parser)
    parser.add_argument(
        "--output-directory", type=Path, default=OUTPUT_ROOT / "ioi_cross_interactions"
    )
    arguments = parser.parse_args()

    model = load_model(MODEL)
    all_pairs = load_prompt_pairs("ioi_pabc", MODEL)
    start, stop = resolve_pair_range(arguments.pairs, len(all_pairs))
    pairs = all_pairs[start:stop]
    sets = grids()

    results: dict[str, dict[str, list]] = {
        name: {"y_empty": [], "y_first": [], "y_second": [], "y_both": []}
        for name in sets
    }
    for pair in pairs:
        # Resample: the replacement values are the pair's pABC prompt's activations.
        source = cache_activations(model, pair.counterfactual_tokens)
        for name, (first, second) in sets.items():
            grid = joint_outcome_grid(model, pair, source, first, second)
            for field, values in results[name].items():
                values.append(getattr(grid, field))

    save_pair_range(
        arguments.output_directory,
        start,
        stop,
        **{
            f"{name}_{field}": np.array(values)
            for name, fields in results.items()
            for field, values in fields.items()
        },
    )


if __name__ == "__main__":
    main()

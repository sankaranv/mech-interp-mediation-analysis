"""Section 4: numbers of the circuit knockout, with the cross-interaction of Theorem 4.1.

Reads ``outputs/circuit_knockout`` (see ``experiments/circuit_knockout.py``) and writes
``outputs/paper/circuit_knockout_numbers.json``. There is no figure.

Each effect ``F(S) = Y(empty) - Y(S)`` is the drop in the logit difference when the set
``S`` is knocked out, so ``F(C') = NIE(C')`` and, by Theorem 4.1 with the circuit ``C`` and
the heads outside it ``C'`` as the two mediators,
``F(C + C') = F(C) + F(C') + xINT(C, C')``.

Run ``python figures/circuit_knockout.py``.
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.estimands import cross_interaction
from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    load_pair_ranges,
    write_numbers,
)
from mediation.statistics import interval_summary


def knockout_numbers(arrays: dict[str, np.ndarray]) -> dict:
    """Effects of knocking out all heads, the circuit, and the heads outside it."""
    clean = arrays["clean_metric"]
    effect_of_all = clean - arrays["all_knocked_out_metric"]
    effect_of_circuit = clean - arrays["circuit_knocked_out_metric"]
    effect_of_complement = clean - arrays["complement_knocked_out_metric"]
    cross = cross_interaction(
        clean,
        arrays["circuit_knocked_out_metric"],
        arrays["complement_knocked_out_metric"],
        arrays["all_knocked_out_metric"],
    )
    # Knocking out the complement with the circuit already knocked out raises the logit
    # difference by F(C) - F(C + C'), which exceeds zero when everything hurts less than
    # the circuit alone.
    raise_given_circuit_knocked_out = effect_of_circuit - effect_of_all
    return {
        "pairs": len(clean),
        "effect_of_all_heads": interval_summary(effect_of_all),
        "effect_of_circuit": interval_summary(effect_of_circuit),
        "effect_of_complement": interval_summary(effect_of_complement),
        "cross_interaction_circuit_complement": interval_summary(cross),
        "raise_from_complement_given_circuit_knocked_out": interval_summary(
            raise_given_circuit_knocked_out
        ),
        "pairs_where_all_heads_hurt_less_than_circuit": int(
            (effect_of_all < effect_of_circuit).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()
    arrays = load_pair_ranges(arguments.output_root / "circuit_knockout")
    write_numbers(
        "circuit_knockout", knockout_numbers(arrays), arguments.paper_directory
    )


if __name__ == "__main__":
    main()

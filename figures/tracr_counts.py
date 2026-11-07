"""Section 3: how often INT vanishes in the Tracr programs of InterpBench.

Reads the arrays of ``experiments/tracr_interactions.py`` and writes the counts the text
quotes to ``tracr_counts_numbers.json``: the number of programs, the programs whose INT is
zero for every mediator and input pair, and, in the remaining programs, the share of
(mediator, input pair) entries with zero INT among those at which patching the mediator
changes the output.
"""

import argparse
from pathlib import Path

import numpy as np

from mediation.results import OUTPUT_ROOT, PAPER_DIRECTORY, write_numbers

# INT below this is zero up to float32 rounding of the forward pass; the counts do not
# change for thresholds from 1e-5 to 1e-3.
ZERO_INTERACTION = 1e-5
# Patching a mediator changes the output if NIE or PIE exceeds this. Entries where neither
# does have NIE = PIE = 0, hence INT = 0 trivially, and are not counted as evidence.
OUTPUT_CHANGE = 1e-6


def count_programs(natural: dict, pure: dict) -> dict:
    """Counts of programs with and without zero INT; see the module docstring.

    :param natural: NIE of each program, shape (pairs, mediators).
    :param pure: PIE of each program, same shapes.
    """
    interaction = {name: natural[name] - pure[name] for name in natural}
    nonzero_programs = [
        name
        for name, values in interaction.items()
        if (np.abs(values) >= ZERO_INTERACTION).any()
    ]
    entries_changing_output, entries_zero_interaction, zero_share_per_program = 0, 0, []
    for name in nonzero_programs:
        changes_output = (np.abs(natural[name]) > OUTPUT_CHANGE) | (
            np.abs(pure[name]) > OUTPUT_CHANGE
        )
        interaction_is_zero = (
            np.abs(interaction[name][changes_output]) < ZERO_INTERACTION
        )
        entries_changing_output += int(changes_output.sum())
        entries_zero_interaction += int(interaction_is_zero.sum())
        zero_share_per_program.append(interaction_is_zero.mean())
    zero_share_per_program = np.array(zero_share_per_program)
    return {
        "programs": len(interaction),
        "programs_with_zero_interaction": len(interaction) - len(nonzero_programs),
        "programs_with_nonzero_interaction": len(nonzero_programs),
        "entries_in_all_programs": sum(values.size for values in interaction.values()),
        "entries_changing_output": entries_changing_output,
        "entries_changing_output_with_zero_interaction": entries_zero_interaction,
        "share_changing_output_with_zero_interaction": entries_zero_interaction
        / entries_changing_output,
        "median_share_per_program": float(np.median(zero_share_per_program)),
        "programs_with_at_least_half_zero": int((zero_share_per_program >= 0.5).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_ROOT / "tracr_interactions",
        help="directory of the experiment's arrays",
    )
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    with (
        np.load(arguments.output_directory / "natural_indirect_effects.npz") as natural,
        np.load(arguments.output_directory / "pure_indirect_effects.npz") as pure,
    ):
        counts = count_programs(dict(natural), dict(pure))
    write_numbers("tracr_counts", counts, arguments.paper_directory)


if __name__ == "__main__":
    main()

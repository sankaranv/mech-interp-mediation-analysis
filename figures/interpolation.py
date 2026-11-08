"""Figure 2: the interaction INT grows with the distance a head's activation is moved.

Reads ``outputs/interpolation`` (see ``experiments/interpolation.py``) and writes the
figure to ``outputs/paper/interpolation.{pdf,png}`` and the quoted numbers to
``outputs/paper/interpolation_numbers.json``.

Run ``python figures/interpolation.py``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mediation.circuits import (
    IOI_ROLE_ABBREVIATIONS,
    IOI_ROLE_COLORS,
    OTHER,
    ioi_role,
)
from mediation.interpolation import INTERPOLATION_HEADS, INTERPOLATION_WEIGHTS
from mediation.patching import AttentionHead
from mediation.results import (
    OUTPUT_ROOT,
    PAPER_DIRECTORY,
    load_pair_ranges,
    save_figure,
    write_numbers,
)
from mediation.statistics import mean_with_confidence_interval, standard_error
from mediation.tasks import pair_count

# Plot style: the dark-grey, outward-ticked look of the published figure, set here so that
# the figure needs matplotlib only.
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "text.color": ".15",
    "axes.edgecolor": ".15",
    "axes.labelcolor": ".15",
    "axes.linewidth": 1.0,
    "axes.axisbelow": True,
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "w",
    "patch.force_edgecolor": True,
    "patch.linewidth": 0.8,
    "xtick.color": ".15",
    "ytick.color": ".15",
    "xtick.major.size": 4.8,
    "ytick.major.size": 4.8,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
}


def interaction_curves(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """INT(w) = NIE(w) - PIE(w) of every head and interpolation weight, per prompt pair.

    ``NIE(w) = Y(a) - Y(a, h(w))`` interpolates the noising patch and
    ``PIE(w) = Y(a', h(0)) - Y(a', h(w))`` the denoising patch, so INT is exactly 0 at
    ``w = 0`` and is Proposition 3.1's INT at ``w = 1``.

    :return: shape (pairs, heads, weights).
    """
    nie = arrays["clean_metric"][:, None, None] - arrays["noised_metric"]
    denoised = arrays["denoised_metric"]
    pie = denoised[:, :, :1] - denoised
    return nie - pie


def head_label(head: AttentionHead) -> str:
    """Legend text, e.g. ``L9H9 (NMH)``; a head outside the circuit is the control, OOC."""
    role = ioi_role(head)
    abbreviation = "OOC" if role == OTHER else IOI_ROLE_ABBREVIATIONS[role]
    return f"{head.name} ({abbreviation})"


def plot_interaction_by_distance(
    interaction: np.ndarray, distance: np.ndarray, directory: Path
) -> None:
    """One line per head: mean INT against ``w`` times the mean activation distance,
    shaded by one standard error (population standard deviation over pairs, as published)."""
    means = interaction.mean(axis=0)
    errors = standard_error(interaction)
    with plt.rc_context(STYLE):
        # Drawn at its printed size, about half the text width, so labels stay legible.
        figure, axes = plt.subplots(figsize=(2.6, 2.0))
        for index, head in enumerate(INTERPOLATION_HEADS):
            color = IOI_ROLE_COLORS[ioi_role(head)]
            x_values = np.array(INTERPOLATION_WEIGHTS) * distance[index]
            axes.fill_between(
                x_values,
                means[index] - errors[index],
                means[index] + errors[index],
                color=color,
                alpha=0.15,
                zorder=3,
            )
            axes.plot(
                x_values,
                means[index],
                color=color,
                lw=1.2,
                marker="o",
                markersize=2.5,
                zorder=4,
                label=head_label(head),
            )
        axes.axhline(0, color="black", lw=0.6, ls=":", zorder=1)
        axes.set_xlabel(
            "Mediator activation distance\nfrom clean $\\|\\Delta h_M\\|$", fontsize=7
        )
        axes.set_ylabel("INT (logit difference)", fontsize=7)
        axes.tick_params(labelsize=6)
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.45, -0.34),
            ncol=2,
            frameon=False,
            fontsize=6,
            handlelength=1.5,
            columnspacing=0.8,
        )
        save_figure(figure, "interpolation", directory, pad_inches=0.01)
        plt.close(figure)


def quoted_numbers(interaction: np.ndarray, distance: np.ndarray) -> dict:
    """Per head: mean INT at every weight, the full-replacement INT (``w = 1``) with its
    95% interval, and the mean activation distance."""
    interval = mean_with_confidence_interval(interaction[:, :, -1])
    return {
        head_label(head): {
            "mean_interaction_by_weight": interaction[:, index].mean(axis=0).tolist(),
            "interaction_at_full_replacement": float(interval.mean[index]),
            "interaction_at_full_replacement_half_width": float(
                interval.half_width[index]
            ),
            "mean_activation_distance": float(distance[:, index].mean()),
        }
        for index, head in enumerate(INTERPOLATION_HEADS)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--paper-directory", type=Path, default=PAPER_DIRECTORY)
    arguments = parser.parse_args()

    arrays = load_pair_ranges(
        arguments.output_root / "interpolation", pair_count("ioi_pabc", "gpt2-small")
    )
    interaction = interaction_curves(arrays)
    distance = arrays["activation_distance"]
    plot_interaction_by_distance(
        interaction, distance.mean(axis=0), arguments.paper_directory
    )
    write_numbers(
        "interpolation",
        quoted_numbers(interaction, distance),
        arguments.paper_directory,
    )


if __name__ == "__main__":
    main()

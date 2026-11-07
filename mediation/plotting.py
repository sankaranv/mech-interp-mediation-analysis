"""Plot elements shared by the IOI figures. Requires matplotlib (the ``figures`` extra)."""

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mediation.circuits import (
    IOI_ROLE_ABBREVIATIONS,
    IOI_ROLE_COLORS,
    IOI_ROLE_ORDER,
    OTHER,
    S_INHIBITION,
)


def ioi_role_label(role: str) -> str:
    """Legend label of a role: its abbreviation, except S-Inhibition and the heads outside
    the circuit, which are written out."""
    return role if role in (S_INHIBITION, OTHER) else IOI_ROLE_ABBREVIATIONS[role]


def add_ioi_role_legend(
    figure: Figure, anchor_y: float, named_alpha: float, other_alpha: float
) -> None:
    """One-row legend of the IOI head roles below the axes, in the colours of the markers.

    :param anchor_y: vertical position of the legend's lower edge in figure coordinates.
    :param named_alpha: opacity of the markers of the 26 circuit heads.
    :param other_alpha: opacity of the marker of the other heads, which the figures draw dim.
    """
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=IOI_ROLE_COLORS[role],
            alpha=other_alpha if role == OTHER else named_alpha,
            markersize=8 if role == OTHER else 9,
            label=ioi_role_label(role),
        )
        for role in IOI_ROLE_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=1.4,
    )

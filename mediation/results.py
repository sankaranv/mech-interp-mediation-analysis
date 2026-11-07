"""Storing experiment results computed over ranges of prompt pairs, and the paper files made from them.

Experiments over many prompt pairs run as several jobs, each over a contiguous range of
pairs. Each job writes one file of arrays whose first axis indexes its pairs; reading a
result directory concatenates the files in pair order and checks that no pair is missing.
Figure and table scripts read these directories and write the paper's files, and the
numbers the paper text quotes from them, to :func:`paper_directory`.
"""

import json
from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path("outputs")


def paper_directory(output_root: Path = OUTPUT_ROOT) -> Path:
    """Where figure and table scripts write the paper's figures, tables and quoted numbers."""
    return output_root / "paper"


PAPER_DIRECTORY = paper_directory()


def head_effects_directory(output_root: Path, model: str, task: str) -> Path:
    """Directory holding the per-head NIE and PIE of ``model`` on ``task``
    (arrays ``nie`` and ``pie`` of shape (pairs, layers, heads))."""
    return output_root / "head_effects" / model / task


def write_numbers(name: str, numbers: dict, directory: Path = PAPER_DIRECTORY) -> Path:
    """Write the numbers a figure or table determines, which the paper text quotes, as
    ``<name>_numbers.json`` with full float precision."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_numbers.json"
    path.write_text(json.dumps(numbers, indent=2) + "\n")
    return path


def save_figure(
    figure, name: str, directory: Path = PAPER_DIRECTORY, **save_options
) -> None:
    """Save a matplotlib figure as ``<name>.pdf`` and ``<name>.png`` in ``directory``.

    :param save_options: forwarded to ``Figure.savefig``; the default is a tight bounding
        box at 300 dpi.
    """
    directory.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "dpi": 300, **save_options}
    for suffix in ("pdf", "png"):
        figure.savefig(directory / f"{name}.{suffix}", **options)


def parse_pair_range(text: str) -> tuple[int, int]:
    """Read a pair range written ``start:stop`` (``stop`` excluded), as on the command line."""
    start, stop = (int(bound) for bound in text.split(":"))
    if not 0 <= start < stop:
        raise ValueError(f"Not a pair range: {text!r}")
    return start, stop


def save_pair_range(
    directory: Path, start: int, stop: int, **arrays: np.ndarray
) -> Path:
    """Write the arrays computed for pairs ``start`` to ``stop - 1``.

    :param arrays: named arrays whose first axis has length ``stop - start``.
    """
    for name, values in arrays.items():
        if len(values) != stop - start:
            raise ValueError(f"{name} has {len(values)} rows for {stop - start} pairs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pairs_{start:05d}_{stop:05d}.npz"
    np.savez(path, **arrays)
    return path


def load_pair_ranges(directory: Path) -> dict[str, np.ndarray]:
    """Concatenate every pair-range file in ``directory`` in pair order.

    :raises ValueError: if the ranges leave a gap, overlap, or do not start at pair 0, so a
        missing job cannot silently shorten a result.
    """
    files = sorted(directory.glob("pairs_*_*.npz"))
    if not files:
        raise FileNotFoundError(f"No results in {directory}")
    parts: dict[str, list[np.ndarray]] = {}
    expected_start = 0
    for path in files:
        start, stop = (int(bound) for bound in path.stem.split("_")[1:])
        if start != expected_start:
            raise ValueError(
                f"{directory}: pairs {expected_start} to {start - 1} missing"
            )
        with np.load(path) as arrays:
            for name in arrays.files:
                parts.setdefault(name, []).append(arrays[name])
        expected_start = stop
    return {name: np.concatenate(values) for name, values in parts.items()}

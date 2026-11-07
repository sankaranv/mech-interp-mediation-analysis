"""Storing experiment results computed over ranges of prompt pairs, and reading them back.

Experiments over many prompt pairs run as several jobs, each over a contiguous range of
pairs. Each job writes one file of arrays whose first axis indexes its pairs; reading a
result directory concatenates the files in pair order and checks that no pair is missing.
"""

from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path("outputs")


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

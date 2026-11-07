"""Check that a job wrote its range of prompt pairs and that every stored value is finite.

Run at the end of every experiment job, so that a job that wrote nothing or wrote NaN
fails instead of leaving a silent gap in the results::

    python slurm/check_pair_range.py head_effects/gpt2-small/ioi_pabc 250

``RESULT`` is the result directory relative to ``outputs/`` and ``START`` the first pair of
the job's range, which names its file ``pairs_<START>_<STOP>.npz``.
"""

import argparse
import sys

import numpy as np

from mediation.results import OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result", help="result directory relative to outputs/")
    parser.add_argument("start", type=int, help="first pair of the job's range")
    parser.add_argument(
        "--upper-triangle",
        nargs="*",
        default=[],
        metavar="ARRAY",
        help="arrays that are NaN by design on and below the diagonal of their last two "
        "axes; only the entries above the diagonal are checked",
    )
    arguments = parser.parse_args()

    directory = OUTPUT_ROOT / arguments.result
    files = sorted(directory.glob(f"pairs_{arguments.start:05d}_*.npz"))
    if len(files) != 1:
        sys.exit(f"FAIL: no result for pairs from {arguments.start} in {directory}")
    with np.load(files[0]) as arrays:
        for name in arrays.files:
            values = arrays[name]
            if name in arguments.upper_triangle:
                rows, columns = np.triu_indices(values.shape[-1], k=1)
                values = values[..., rows, columns]
            if not np.isfinite(values).all():
                sys.exit(f"FAIL: non-finite {name} in {files[0]}")
        print(f"OK: {files[0]} holds {len(arrays.files)} finite arrays")


if __name__ == "__main__":
    main()

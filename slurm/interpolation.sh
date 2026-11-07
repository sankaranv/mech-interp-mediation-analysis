#!/bin/bash
# Figure 2: six GPT-2 small heads patched to interpolated activations, over one range of pairs.
#
# Submit from the repository root:
#   PAIR_START=0 PAIR_STOP=250 sbatch slurm/interpolation.sh
# Eager attention; 74 forward passes per pair (about 3 seconds per pair). float32 needs compute capability 7.0 or higher.
#SBATCH --job-name=interpolation
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=sm_70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/interpolation_%j.out
#SBATCH --error=outputs/logs/interpolation_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -u experiments/interpolation.py --pairs "${PAIR_START}:${PAIR_STOP}"

# Check that the range was written and every array is finite.
python -u - <<'EOF'
import os
import sys

import numpy as np

from mediation.results import OUTPUT_ROOT

directory = OUTPUT_ROOT / "interpolation"
start = int(os.environ["PAIR_START"])
files = sorted(directory.glob(f"pairs_{start:05d}_*.npz"))
if len(files) != 1:
    sys.exit(f"FAIL: no result for pairs from {start} in {directory}")
with np.load(files[0]) as arrays:
    for name in arrays.files:
        if not np.isfinite(arrays[name]).all():
            sys.exit(f"FAIL: non-finite {name} in {files[0]}")
    print(f"OK: {files[0]} holds {len(arrays.files)} finite arrays")
EOF

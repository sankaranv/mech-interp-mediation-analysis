#!/bin/bash
# Appendix C: MLP cross-interactions on Greater-Than in GPT-2 small, over one range of pairs.
#
# Submit from the repository root:
#   PAIR_START=0 PAIR_STOP=40 sbatch slurm/greater_than_cross_interactions.sh
# Each job averages the MLP outputs over all 1000 counterfactual prompts for mean ablation
# (about a minute) and then makes 74 forward passes per pair in its range. float32 needs
# compute capability 7.0 or higher.
#SBATCH --job-name=greater_than_cross_interactions
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=sm_70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/greater_than_cross_interactions_%j.out
#SBATCH --error=outputs/logs/greater_than_cross_interactions_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -u experiments/greater_than_cross_interactions.py \
    --pairs "${PAIR_START}:${PAIR_STOP}"

# Check that both ablations wrote the range and hold finite metrics: every single and every
# pair (the upper triangle of y_joint; the rest is NaN by design).
python -u - <<'EOF'
import os
import sys

import numpy as np

from mediation.results import OUTPUT_ROOT

start = int(os.environ["PAIR_START"])
for ablation in ("mean", "resample"):
    directory = OUTPUT_ROOT / "greater_than_cross_interactions" / ablation
    files = sorted(directory.glob(f"pairs_{start:05d}_*.npz"))
    if len(files) != 1:
        sys.exit(f"FAIL: no result for pairs from {start} in {directory}")
    with np.load(files[0]) as arrays:
        upper = np.triu_indices(arrays["y_single"].shape[1], k=1)
        finite = (
            np.isfinite(arrays["y_clean"]).all()
            and np.isfinite(arrays["y_single"]).all()
            and np.isfinite(arrays["y_joint"][:, upper[0], upper[1]]).all()
        )
        if not finite:
            sys.exit(f"FAIL: non-finite metric in {files[0]}")
        print(f"OK: {files[0]} y_joint shape {arrays['y_joint'].shape}, all finite")
EOF

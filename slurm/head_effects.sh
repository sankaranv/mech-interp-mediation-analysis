#!/bin/bash
# Per-head NIE and PIE of one model and task over one range of prompt pairs.
#
# Submit from the repository root:
#   MODEL=gpt2-small TASK=ioi_pabc PAIR_START=0 PAIR_STOP=250 sbatch slurm/head_effects.sh
# float32 needs compute capability 7.0 or higher. Pythia-6.9B (float32) and Llama-3.1-8B
# need a GPU with 40 GB; run_all.sh submits them with
#   --partition=gpu-preempt-highvram --constraint="vram40&bf16" --mem=64G
# where the bf16 feature also keeps them off the RTX 8000 nodes, on which Llama-3.1-8B
# times out.
#SBATCH --job-name=head_effects
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=sm_70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/head_effects_%j.out
#SBATCH --error=outputs/logs/head_effects_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -u experiments/head_effects.py \
    --model "$MODEL" --task "$TASK" --pairs "${PAIR_START}:${PAIR_STOP}"

# Check that the range was written and holds finite effects for every head.
python -u - <<'EOF'
import os
import sys

import numpy as np

from mediation.results import OUTPUT_ROOT, head_effects_directory

directory = head_effects_directory(OUTPUT_ROOT, os.environ["MODEL"], os.environ["TASK"])
start = int(os.environ["PAIR_START"])
files = sorted(directory.glob(f"pairs_{start:05d}_*.npz"))
if len(files) != 1:
    sys.exit(f"FAIL: no result for pairs from {start} in {directory}")
with np.load(files[0]) as arrays:
    for name in ("nie", "pie"):
        if not np.isfinite(arrays[name]).all():
            sys.exit(f"FAIL: non-finite {name} in {files[0]}")
    print(f"OK: {files[0]} nie shape {arrays['nie'].shape}, all finite")
EOF

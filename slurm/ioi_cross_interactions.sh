#!/bin/bash
# Section 5.3 and Appendix B: cross-interactions between groups of IOI heads, over one range of pairs.
#
# Submit from the repository root:
#   PAIR_START=0 PAIR_STOP=250 sbatch slurm/ioi_cross_interactions.sh
# 363 forward passes per pair. float32 needs compute capability 7.0 or higher.
#SBATCH --job-name=ioi_cross_interactions
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/ioi_cross_interactions_%j.out
#SBATCH --error=outputs/logs/ioi_cross_interactions_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -u experiments/ioi_cross_interactions.py --pairs "${PAIR_START}:${PAIR_STOP}"

# Check that the range was written and holds finite values.
python -u slurm/check_pair_range.py "ioi_cross_interactions" "$PAIR_START"

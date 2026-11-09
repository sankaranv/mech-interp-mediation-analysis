#!/bin/bash
# Appendix C: MLP cross-interactions on Greater-Than in GPT-2 small, over one range of pairs.
#
# Submit from the repository root:
#   PAIR_START=0 PAIR_STOP=40 sbatch slurm/greater_than_cross_interactions.sh
# Each job averages the MLP outputs over all 1000 counterfactual prompts for mean ablation
# and then makes 74 forward passes per pair in its range. float32 needs compute capability
# 7.0 or higher.
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
# pair (y_joint is NaN on and below the diagonal by design).
for ablation in mean resample; do
    python -u slurm/check_pair_range.py "greater_than_cross_interactions/$ablation" \
        "$PAIR_START" --upper-triangle y_joint
done

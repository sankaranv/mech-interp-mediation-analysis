#!/bin/bash
# Every figure and table of the paper from the arrays in outputs/ (a short CPU job).
#
# run_all.sh submits it to start after all experiment jobs have succeeded; run it by hand
# with `sbatch slurm/figures.sh` to make the figures from existing results.
#SBATCH --job-name=figures
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/figures_%j.out
#SBATCH --error=outputs/logs/figures_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

figures/make_all.sh

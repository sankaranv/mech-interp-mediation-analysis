#!/bin/bash
# Make every figure and table of the paper from the arrays in outputs/, once the jobs of
# slurm/run_all.sh have finished. Writes outputs/paper/: figures as pdf and png, tables as
# tex, and for every one the numbers the paper text quotes as <name>_numbers.json.
#
# Run from anywhere with the environment of the repository active:
#   source .venv/bin/activate && figures/make_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for script in figures/*.py; do
    python "$script"
done

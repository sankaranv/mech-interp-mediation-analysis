#!/bin/bash
# Section 3: NIE and PIE of every mediator of the 129 Tracr programs of InterpBench.
#
# Submit from the repository root, after setting up the environment of
# environment/tracr.md (a CPU job):
#   sbatch slurm/tracr_interactions.sh
#SBATCH --job-name=tracr_interactions
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/tracr_interactions_%j.out
#SBATCH --error=outputs/logs/tracr_interactions_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/external/InterpBench"

.venv-tracr/bin/python -u experiments/tracr_interactions.py

# Check that every program was written with the same shapes in both files and that all
# effects are finite.
.venv-tracr/bin/python -u - <<'EOF'
import sys

import numpy as np

from mediation.results import OUTPUT_ROOT

directory = OUTPUT_ROOT / "tracr_interactions"
with (
    np.load(directory / "natural_indirect_effects.npz") as natural,
    np.load(directory / "pure_indirect_effects.npz") as pure,
):
    if len(natural.files) != 129 or set(natural.files) != set(pure.files):
        sys.exit(f"FAIL: {len(natural.files)} programs, expected 129 in both files")
    for name in natural.files:
        if natural[name].shape != pure[name].shape or natural[name].shape[0] != 8:
            sys.exit(f"FAIL: program {name} has shape {natural[name].shape}")
        if not (np.isfinite(natural[name]).all() and np.isfinite(pure[name]).all()):
            sys.exit(f"FAIL: non-finite effect in program {name}")
    print(f"OK: {len(natural.files)} programs, all finite")
EOF

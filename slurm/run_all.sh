#!/bin/bash
# Submit every experiment of the paper to SLURM, one job per range of prompt pairs.
#
# Run from anywhere; jobs write to outputs/ of the repository. Restrict the per-head
# effects to some models with, for example,
#   MODELS="gpt2-small pythia-70m" slurm/run_all.sh
# Afterwards, when the jobs have finished, run figures/make_all.sh.
#
# Range sizes target about 30 minutes per job. Seconds per prompt pair of the per-head
# effects (GPU-hours of the paper's runs divided by their pairs): GPT-2 small 3.5,
# Pythia-70M 0.9, GPT-2 medium 16, Qwen2.5-0.5B 24, Llama-3.1-8B 60, GPT-2 XL 120,
# Pythia-6.9B 120.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/logs

MODELS=${MODELS:-"gpt2-small gpt2-medium gpt2-xl pythia-70m pythia-6.9b qwen2.5-0.5b llama-3.1-8b"}

declare -A PAIRS_PER_JOB=(
    [gpt2-small]=500 [gpt2-medium]=100 [gpt2-xl]=15
    [pythia-70m]=1000 [pythia-6.9b]=15
    [qwen2.5-0.5b]=75 [llama-3.1-8b]=30
)

# Tasks of a model: Greater-Than needs a tokenizer that emits each two-digit year ending as
# one token, which Qwen2.5 and Llama-3.1 do not.
tasks_of() {
    local tasks="ioi_pabc ioi_symmetric sva gender_bias"
    case "$1" in
        gpt2-* | pythia-*) tasks="$tasks greater_than" ;;
    esac
    echo "$tasks"
}

# Number of prompt pairs of a task under a model's tokenizer: the pABC pairs are filtered
# to equal length under Pythia's tokenizer and Gender Bias has one file per tokenizer.
pair_count() {
    case "$2" in
        ioi_pabc) [[ $1 == pythia-* ]] && echo 691 || echo 1000 ;;
        gender_bias) [[ $1 == pythia-* ]] && echo 782 || echo 986 ;;
        *) echo 1000 ;;
    esac
}

# sbatch options that the template of head_effects.sh does not already carry. Pythia-6.9B
# in float32 needs 27 GB of GPU memory; "bf16" keeps both large models off the RTX 8000
# nodes. Loading the large models takes minutes, so they get a longer time limit.
resource_options() {
    case "$1" in
        pythia-6.9b | llama-3.1-8b)
            echo "--partition=gpu-preempt-highvram --constraint=vram40&bf16 --mem=64G --time=01:30:00" ;;
        gpt2-xl) echo "--time=01:30:00" ;;
    esac
}

# submit_ranges SCRIPT TOTAL PER_JOB [SBATCH OPTION...]: ceil(TOTAL / PER_JOB) jobs over
# ranges of TOTAL pairs that differ in size by at most one pair; the script reads
# PAIR_START and PAIR_STOP (stop excluded) from the environment.
submit_ranges() {
    local script=$1 total=$2 per_job=$3
    shift 3
    local jobs=$(((total + per_job - 1) / per_job))
    local job
    for ((job = 0; job < jobs; job++)); do
        PAIR_START=$((total * job / jobs)) PAIR_STOP=$((total * (job + 1) / jobs)) \
            sbatch "$@" "$script"
    done
}

# Figure 1 and Section 5 (via Appendix B): per-head effects of every model and task.
for model in $MODELS; do
    read -ra options <<<"$(resource_options "$model")"
    for task in $(tasks_of "$model"); do
        MODEL=$model TASK=$task submit_ranges slurm/head_effects.sh \
            "$(pair_count "$model" "$task")" "${PAIRS_PER_JOB[$model]}" \
            "${options[@]}"
    done
done

# The experiments below all run on GPT-2 small.
# Figure 2 (about 22 minutes for 1000 pairs).
submit_ranges slurm/interpolation.sh 1000 1000
# Section 4 (about 6 minutes for 1000 pairs).
submit_ranges slurm/circuit_knockout.sh 1000 1000
# Section 5 and Appendix B cross-interactions (about 2 GPU-hours for 1000 pairs).
submit_ranges slurm/ioi_cross_interactions.sh 1000 250
# Appendix C (about 30 minutes for 1000 pairs).
submit_ranges slurm/greater_than_cross_interactions.sh 1000 500

# Section 3 Tracr counts: a CPU job in its own environment (environment/tracr.md).
if [[ -d .venv-tracr ]]; then
    sbatch slurm/tracr_interactions.sh
else
    echo "Skipping Tracr: .venv-tracr is missing (see environment/tracr.md)" >&2
fi

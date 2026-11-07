#!/bin/bash
# Submit every experiment of the paper to SLURM, one job per range of prompt pairs, and a
# last job that makes every figure and table once they have all succeeded.
#
# Run from anywhere; jobs write to outputs/ of the repository. Restrict the per-head
# effects to some models with, for example,
#   MODELS="gpt2-small pythia-70m" slurm/run_all.sh
# Ranges whose result file already exists in outputs/ are not submitted again, so after
# an interruption (or after jobs failed) run the script again once no job of the earlier
# submission is left in the queue. The figure job then waits only for the jobs of this
# submission; with every range already done it is the only job submitted.
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

submitted_jobs=()
skipped_ranges=0

# submit_ranges RESULT SCRIPT TOTAL PER_JOB [SBATCH OPTION...]: ceil(TOTAL / PER_JOB) jobs
# over ranges of TOTAL pairs that differ in size by at most one pair; the script reads
# PAIR_START and PAIR_STOP (stop excluded) from the environment. A range is skipped if its
# file pairs_<start>_<stop>.npz (named as mediation.results.save_pair_range does) is in
# outputs/RESULT. The ids of the submitted jobs are appended to submitted_jobs.
submit_ranges() {
    local result=$1 script=$2 total=$3 per_job=$4
    shift 4
    local jobs=$(((total + per_job - 1) / per_job))
    local job start stop file
    for ((job = 0; job < jobs; job++)); do
        start=$((total * job / jobs))
        stop=$((total * (job + 1) / jobs))
        printf -v file "outputs/%s/pairs_%05d_%05d.npz" "$result" "$start" "$stop"
        if [[ -e $file ]]; then
            skipped_ranges=$((skipped_ranges + 1))
        else
            submitted_jobs+=("$(PAIR_START=$start PAIR_STOP=$stop \
                sbatch --parsable "$@" "$script")")
        fi
    done
}

# Figure 1 and Section 5 (via Appendix B): per-head effects of every model and task.
for model in $MODELS; do
    read -ra options <<<"$(resource_options "$model")"
    for task in $(tasks_of "$model"); do
        MODEL=$model TASK=$task submit_ranges "head_effects/$model/$task" \
            slurm/head_effects.sh "$(pair_count "$model" "$task")" \
            "${PAIRS_PER_JOB[$model]}" "${options[@]}"
    done
done

# The experiments below all run on GPT-2 small.
# Figure 2 (about 3 seconds per pair).
submit_ranges interpolation slurm/interpolation.sh 1000 500
# Section 4 (under a second per pair).
submit_ranges circuit_knockout slurm/circuit_knockout.sh 1000 1000
# Section 5 and Appendix B cross-interactions (about 3 seconds per pair).
submit_ranges ioi_cross_interactions slurm/ioi_cross_interactions.sh 1000 500
# Appendix C (about 2 seconds per pair). A job writes the mean ablation before the resample
# ablation, so the resample file shows that both are done.
submit_ranges greater_than_cross_interactions/resample \
    slurm/greater_than_cross_interactions.sh 1000 500

# Section 3 Tracr counts: a CPU job in its own environment (environment/tracr.md).
if [[ -d .venv-tracr && -d external/InterpBench ]]; then
    if [[ ! -e outputs/tracr_interactions/programs.json ]]; then
        submitted_jobs+=("$(sbatch --parsable slurm/tracr_interactions.sh)")
    fi
else
    echo "Skipping Tracr: .venv-tracr or external/InterpBench is missing" \
        "(see environment/tracr.md)" >&2
fi

# Every figure and table, after every job of this submission has succeeded. If one fails,
# the figure job is cancelled; run this script again to resubmit the missing ranges.
dependency=()
if ((${#submitted_jobs[@]} > 0)); then
    dependency=(--kill-on-invalid-dep=yes
        "--dependency=afterok:$(IFS=:; echo "${submitted_jobs[*]}")")
fi
sbatch --parsable "${dependency[@]}" slurm/figures.sh >/dev/null

echo "Submitted ${#submitted_jobs[@]} jobs and the figure job;" \
    "$skipped_ranges ranges were already done"

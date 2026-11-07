# The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching

Code for the NeurIPS 2026 paper by Sankaran Vaidyanathan, David Arbour, Aaron Mueller,
Scott Niekum and David Jensen. It reproduces every figure and table of the paper and every
number the text quotes from them.

The paper studies activation patching of several mediators (attention heads, MLPs) at
once. Noising a head gives its natural indirect effect (NIE), denoising it gives its pure
indirect effect (PIE), and their difference is the interaction INT. Interactions between
two sets of mediators are cross-interactions (xINT). The library `mediation/` implements
these estimands for GPT-2, Pythia, Qwen2.5 and Llama-3.1; each script in `experiments/`
computes the arrays of one result, and each script in `figures/` turns arrays into the
paper's figure or table.

## Install

Python 3.11 or newer and, for the experiments, a CUDA GPU (the Tracr experiment runs on
CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[figures]"
```

The environment pins torch 2.9.1, transformers 4.57.6 and nnsight 0.6.2. Models are
downloaded from the Hugging Face Hub at the revisions pinned in `mediation/models.py`;
Llama-3.1-8B is gated, so accept its license on the Hub and run `hf auth login` first.
The Tracr experiment needs its own environment, described in `environment/tracr.md`.
`pip install -e ".[development]"` adds the tests (`pytest`).

## Reproducing the paper

Every experiment writes arrays to `outputs/<result>/`, one file per range of prompt pairs
(`pairs_<start>_<stop>.npz`, `--pairs start:stop`, stop excluded). Every figure or table
script reads only `outputs/` and writes to `outputs/paper/`: the figure (`.pdf` and
`.png`) or table (`.tex`), and `<name>_numbers.json` with the numbers the paper text
quotes from it.

| Paper result | Experiment | Arrays in `outputs/` | Figure or table script | File in `outputs/paper/` |
|---|---|---|---|---|
| Figure 1 (rank agreement of NIE and PIE, 5 tasks, 7 models), Section 3 values and the introduction's minimum | `python experiments/head_effects.py --model M --task T --pairs a:b` for each model `M` in `gpt2-small gpt2-medium gpt2-xl pythia-70m pythia-6.9b qwen2.5-0.5b llama-3.1-8b` and task `T` in `ioi_pabc ioi_symmetric sva gender_bias greater_than` (Greater-Than for the GPT-2 and Pythia models only) | `head_effects/M/T/` | `python figures/rank_agreement.py` | `rank_agreement.{pdf,png}`, `rank_agreement_numbers.json` |
| Section 3, Tracr programs (number of programs, programs with INT zero) | `python experiments/tracr_interactions.py` (CPU, separate environment) | `tracr_interactions/` | `python figures/tracr_counts.py` | `tracr_counts_numbers.json` |
| Figure 2 (INT against patch distance, interpolating six heads) | `python experiments/interpolation.py --pairs a:b` | `interpolation/` | `python figures/interpolation.py` | `interpolation.{pdf,png}`, `interpolation_numbers.json` |
| Section 4 (circuit knockout, xINT of circuit and complement) | `python experiments/circuit_knockout.py --pairs a:b` | `circuit_knockout/` | `python figures/circuit_knockout.py` | `circuit_knockout_numbers.json` |
| Figure 3 (NIE, PIE and INT of the GPT-2 small heads on IOI) and its Section 5 values | `head_effects.py --model gpt2-small --task ioi_pabc` | `head_effects/gpt2-small/ioi_pabc/` | `python figures/ioi_head_effects.py` | `ioi_head_effects.{pdf,png}`, `ioi_head_effects_numbers.json` |
| Figure 4 (NIE and PIE rankings of the 144 heads), Section 5 ranks and the Induction Head ranks under symmetric corruption | `head_effects.py --model gpt2-small --task ioi_pabc` and `--task ioi_symmetric` | `head_effects/gpt2-small/{ioi_pabc,ioi_symmetric}/` | `python figures/ioi_rankings.py` | `ioi_rankings.{pdf,png}`, `ioi_rankings_numbers.json` |
| Appendix B, table of NIE, PIE and INT of the 26 circuit heads, and its count of heads whose PIE standard deviation exceeds its mean | `head_effects.py --model gpt2-small --task ioi_pabc` | `head_effects/gpt2-small/ioi_pabc/` | `python figures/ioi_estimand_table.py` | `ioi_estimand_table.tex`, `ioi_estimand_table_numbers.json` |
| Section 5 cross-interactions (S-Inhibition pairs, Name Mover group, L9H9 and the Backup Name Mover Heads), Appendix B tables of xINT | `python experiments/ioi_cross_interactions.py --pairs a:b` | `ioi_cross_interactions/` | `python figures/ioi_cross_interaction_tables.py` (also reads `head_effects/gpt2-small/ioi_pabc/`) | `ioi_bnmh_table.tex`, `sinh_xint_table.tex`, `ioi_cross_interaction_tables_numbers.json` |
| Appendix C (Greater-Than MLP cross-interactions, mean and resample ablation) | `python experiments/greater_than_cross_interactions.py --pairs a:b` | `greater_than_cross_interactions/{mean,resample}/` | `python figures/greater_than_cross_interaction_table.py` | `greater_than_cross_interaction_table.tex`, `greater_than_cross_interaction_numbers.json` |

All experiments except the per-head effects of other models and the Tracr programs use
GPT-2 small. Without `--pairs`, `head_effects.py` runs every pair of the task; with it, a
stop beyond the number of pairs is clipped. The number of pairs is 1000 per task, except
Gender Bias (986; 782 under Pythia's tokenizer) and IOI with the pABC counterfactual under
Pythia's tokenizer (691, the pairs whose clean and counterfactual prompts have equal
length).

### On SLURM

```bash
slurm/run_all.sh                  # submit every experiment, about 900 jobs of up to 45 minutes
MODELS="gpt2-small" slurm/run_all.sh   # per-head effects of the named models only
figures/make_all.sh               # after all jobs have finished: every figure and table
```

`slurm/run_all.sh` splits each experiment into jobs over ranges of prompt pairs, sized
from the compute accounting of the paper's runs to take about 30 minutes each, and
submits `slurm/<experiment>.sh` with the range in `PAIR_START` and `PAIR_STOP` (and
`MODEL` and `TASK` for `head_effects.sh`). The templates use the account and partitions
of our cluster, `gpu-preempt` with `--constraint=sm_70` (float32 needs compute capability
7.0), and `gpu-preempt-highvram` with 40 GB for Pythia-6.9B and Llama-3.1-8B; edit the
`#SBATCH` lines for another cluster. Jobs are requeued after preemption and end with a
check that their output exists and is finite. The Tracr job is submitted only if
`.venv-tracr` exists. The per-head effects dominate the cost: our runs took over 300
GPU-hours, nearly all of it GPT-2 XL, Pythia-6.9B and Llama-3.1-8B.

### Locally, for GPT-2 small

The per-head effects of GPT-2 small on the five tasks take about 5 GPU-hours (3.5 seconds
per pair) and the other experiments about 2.5 GPU-hours.

```bash
for task in ioi_pabc ioi_symmetric sva gender_bias greater_than; do
    python experiments/head_effects.py --model gpt2-small --task $task
done
python experiments/interpolation.py
python experiments/circuit_knockout.py
python experiments/ioi_cross_interactions.py
python experiments/greater_than_cross_interactions.py
python figures/ioi_head_effects.py
python figures/ioi_rankings.py
python figures/ioi_estimand_table.py
python figures/interpolation.py
python figures/circuit_knockout.py
python figures/ioi_cross_interaction_tables.py
python figures/greater_than_cross_interaction_table.py
python figures/rank_agreement.py      # Figure 1 with the GPT-2 small column only
```

`figures/rank_agreement.py` leaves cells without arrays empty, so with GPT-2 small alone it
reproduces the GPT-2 small values (0.996, 0.919, 0.603, 0.513, 0.509) of Section 3. For a
quick check, `--pairs 0:20` on every command runs in minutes.

### Numerical reproducibility

Results depend on the GPU: two runs on different GPUs differ at the rounding level of the
model's precision. The per-head NIE and PIE of GPT-2 (float32) agree to about 1e-4,
Pythia-70M (float32) to about 1e-2 because its float32 activations are intrinsically that
imprecise, and Llama-3.1-8B (bfloat16) to a few tenths. Rank agreements average over 1000
pairs; in our reruns of the original code they differed from the stored arrays by at most
0.002.

## Data

The prompt pairs are in `data/`; their origin, license status and checksums are in
[`data/SOURCES.md`](data/SOURCES.md).

## Citation

```bibtex
@inproceedings{vaidyanathan2026curse,
  title     = {The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching},
  author    = {Vaidyanathan, Sankaran and Arbour, David and Mueller, Aaron and Niekum, Scott and Jensen, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

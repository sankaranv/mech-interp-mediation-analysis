# The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching

Code for the NeurIPS 2026 paper ([arXiv](https://arxiv.org/abs/2606.27510)). It reproduces
every figure, table and quoted number. `mediation/` is the library; each script in
`experiments/` writes the arrays of one result to `outputs/`, and each script in `figures/`
turns them into a figure or table in `outputs/paper/`, with the numbers the text quotes in
`<name>_numbers.json`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[figures]"    # ".[development]" adds pytest
```

The experiments need a CUDA GPU. Llama-3.1-8B is gated: accept its license on the Hugging
Face Hub and run `hf auth login`. The Tracr experiment runs on CPU in its own environment
(`environment/tracr.md`).

## Reproducing the paper

| Result | `experiments/` | `figures/` |
|---|---|---|
| Figure 1, Section 3 | `head_effects.py --model M --task T` | `rank_agreement.py` |
| Section 3, Tracr | `tracr_interactions.py` | `tracr_counts.py` |
| Figure 2 | `interpolation.py` | `interpolation.py` |
| Section 4 | `circuit_knockout.py` | `circuit_knockout.py` |
| Figure 3 | `head_effects.py --model gpt2-small --task ioi_pabc` | `ioi_head_effects.py` |
| Figure 4 | as Figure 3, and `--task ioi_symmetric` | `ioi_rankings.py` |
| Appendix B, estimand table | as Figure 3 | `ioi_estimand_table.py` |
| Section 5, Appendix B cross-interaction tables | `ioi_cross_interactions.py` | `ioi_cross_interaction_tables.py` |
| Appendix C | `greater_than_cross_interactions.py` | `greater_than_cross_interaction_table.py` |

Figure 1 uses every model `M` in `gpt2-small gpt2-medium gpt2-xl pythia-70m pythia-6.9b
qwen2.5-0.5b llama-3.1-8b` and task `T` in `ioi_pabc ioi_symmetric sva gender_bias
greater_than` (Greater-Than for GPT-2 and Pythia only); everything else uses GPT-2 small.
Every experiment on prompt pairs runs all pairs of its task, or a range with
`--pairs start:stop`; `--pairs 0:20` is a quick check.

On SLURM, `slurm/run_all.sh` submits every experiment in ranges of pairs and then
`figures/make_all.sh`; run it again after a failure and it submits only the missing ranges.
The scripts request only resources; set your account and partition (for example with
`SBATCH_ACCOUNT` and `SBATCH_PARTITION`), and a GPU with 40 GB for Pythia-6.9B and
Llama-3.1-8B in `slurm/run_all.sh`.

For GPT-2 small alone, run its experiments and then each figure script except
`tracr_counts.py`; `rank_agreement.py` then draws only the GPT-2 small column.

Results vary with the GPU at the model's precision. In our full rerun every quoted number
reproduced exactly; Figure 1 cells agreed within 0.005 for Pythia and within 0.04 for
Llama-3.1-8B (bfloat16).

## Data

The prompt pairs are in `data/`; their sources and terms are in
[`data/SOURCES.md`](data/SOURCES.md).

## Citation

```bibtex
@inproceedings{vaidyanathan2026curse,
  title         = {The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching},
  author        = {Sankaran Vaidyanathan and David Arbour and Aaron Mueller and Scott Niekum and David Jensen},
  booktitle     = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year          = {2026},
  url           = {https://arxiv.org/abs/2606.27510},
  eprint        = {2606.27510},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

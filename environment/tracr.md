# Environment of the Tracr experiment

`experiments/tracr_interactions.py` (Section 3, Tracr programs) needs the InterpBench
benchmark of Gupta et al. (https://github.com/FlyingPumba/InterpBench, MIT license) and
the fork of Tracr it depends on, together with TransformerLens 1.19.0. These pins conflict
with the main environment (`pyproject.toml`: nnsight, transformers 4.57), so the Tracr
experiment runs in its own environment. It runs on CPU. Only `mediation.results` is
imported from this repository, so the main package is installed without its dependencies.

The benchmark is not vendored; clone it at the commit the published counts used.

```bash
# Run from the repository root.
conda create --prefix .venv-tracr python=3.11 pygraphviz -c conda-forge
.venv-tracr/bin/python -m pip install -r environment/tracr_requirements.txt
.venv-tracr/bin/python -m pip install --no-deps -e .

git clone https://github.com/FlyingPumba/InterpBench external/InterpBench
git -C external/InterpBench checkout 220791e4a9cd29957ed5edfa445e7c09152d7298
```

`pygraphviz` is imported by `circuits_benchmark` and needs Graphviz, hence conda-forge.
`environment/tracr_requirements.txt` pins the packages of the environment that produced
the published counts, with the Tracr, IIT and ACDC forks at the commits the benchmark's
`pyproject.toml` resolved to. The benchmark itself is used from its checkout through
`PYTHONPATH` rather than installed:

```bash
PYTHONPATH="$PWD/external/InterpBench" .venv-tracr/bin/python \
    experiments/tracr_interactions.py
```

`slurm/tracr_interactions.sh` does the same on a CPU node, and
`figures/tracr_counts.py` (main environment) turns the arrays into the counts the paper
quotes. The experiment skips the benchmark's two IOI cases, which run a GPT-2 checkpoint
instead of a compiled program, and so covers the 129 compiled programs.

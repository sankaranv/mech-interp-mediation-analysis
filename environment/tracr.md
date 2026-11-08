# Environment of the Tracr experiment

`experiments/tracr_interactions.py` (Section 3, Tracr programs) needs the InterpBench
benchmark of Gupta et al. (https://github.com/FlyingPumba/InterpBench, MIT license) and
the fork of Tracr it depends on, together with TransformerLens 1.19.0. These pins conflict
with the main environment (`pyproject.toml`: nnsight, transformers 4.57), so the Tracr
experiment runs in its own environment. It runs on CPU. Only `mediation.results` is
imported from this repository, so the main package is installed without its dependencies.

The benchmark is not vendored; clone it at the commit the published counts used.

```bash
# Run from the repository root. conda is required; on a cluster with environment modules,
# load it first (for example `module load conda/latest`).
conda create --prefix .venv-tracr python=3.11 pygraphviz -c conda-forge
.venv-tracr/bin/python -m pip install -r environment/tracr_requirements.txt
.venv-tracr/bin/python -m pip install --no-deps -r environment/tracr_forks_requirements.txt
.venv-tracr/bin/python -m pip install --no-deps -e .

git clone https://github.com/FlyingPumba/InterpBench external/InterpBench
git -C external/InterpBench checkout 220791e4a9cd29957ed5edfa445e7c09152d7298
```

`pygraphviz` is imported by `circuits_benchmark` and needs Graphviz, hence conda-forge.
`environment/tracr_requirements.txt` pins the packages of the environment that produced
the published counts, with the Tracr fork at the commit the benchmark's `pyproject.toml`
resolved to; `environment/tracr_forks_requirements.txt` does the same for the IIT and ACDC
forks. The benchmark itself is used from its checkout through
`PYTHONPATH` rather than installed:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/external/InterpBench" .venv-tracr/bin/python \
    experiments/tracr_interactions.py
```

`PYTHONHASHSEED` must be fixed. The benchmark's input generators (for example `case_5`)
draw with `random.choice(list(chars))` from a `set` of strings, whose order changes with
Python's per-process string hashing, so the inputs of those programs and hence the
effects of `case_5` differ from run to run. The number of programs with zero INT does not
change, but the entry counts do.

`slurm/tracr_interactions.sh` does the same on a CPU node, and
`figures/tracr_counts.py` (main environment) turns the arrays into the counts the paper
quotes. The experiment skips the benchmark's two IOI cases, which run a GPT-2 checkpoint
instead of a compiled program, and so covers the 129 compiled programs.

"""Section 3: NIE and PIE of every attention head and MLP in the Tracr programs of InterpBench.

Tracr compiles a RASP program into a transformer whose weights are set by hand, so the
network is piecewise affine by construction. For each compiled program the script
patches every attention head and every MLP output, one mediator at a time, on 8 pairs of
the program's own inputs and records NIE and PIE; INT = NIE - PIE is formed by the figure
script. Runs on CPU in the separate environment of ``environment/tracr.md``.

Written to ``<output-directory>``: ``natural_indirect_effects.npz`` and
``pure_indirect_effects.npz`` (one float64 array of shape (pairs, mediators) per program,
mediators ordered by layer with the heads before the MLP) and ``programs.json`` (the
shape and readout of each program).
"""

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from transformer_lens import ActivationCache, HookedTransformer

from mediation.results import OUTPUT_ROOT

# The two IOI cases of the benchmark run a GPT-2 checkpoint and have no compiled program.
NOT_COMPILED_PROGRAMS = ("case_ioi", "case_ioi_next_token")
PAIRS_PER_PROGRAM = 8
# Rows of the program's clean inputs used to choose the readout of a program with more
# than two output symbols.
READOUT_PROBE_ROWS = 32


class Mediator(NamedTuple):
    """An attention head (``head`` set; its ``hook_z``) or an MLP output (``head`` None)."""

    hook_name: str
    head: int | None


@dataclass(frozen=True)
class Readout:
    """The fixed linear function of the last position's output that serves as the metric.

    INT vanishes for a downstream computation that is affine only if the metric is linear
    in the output, so the readout is a single output coordinate or the difference of two,
    never a nonlinear score. It is chosen once per program, before any patching.

    :ivar positive_class: output coordinate added to the metric.
    :ivar negative_class: output coordinate subtracted; ``None`` for a program with a
        single numerical output.
    """

    positive_class: int
    negative_class: int | None

    def __call__(self, logits: torch.Tensor) -> float:
        last_position = logits[0, -1]
        if self.negative_class is None:
            return last_position[self.positive_class].item()
        return (
            last_position[self.positive_class] - last_position[self.negative_class]
        ).item()


def select_readout(model: HookedTransformer, probe_inputs: torch.Tensor) -> Readout:
    """Readout of a program: its only output, the difference of its two outputs, or, with
    more outputs, the difference of the two output symbols that the program predicts most
    often on ``probe_inputs`` (so that the metric changes between inputs)."""
    vocabulary_out = model.cfg.d_vocab_out
    if vocabulary_out <= 1:
        return Readout(0, None)
    if vocabulary_out == 2:
        return Readout(0, 1)
    with torch.no_grad():
        predictions = model(probe_inputs)[:, -1, :].argmax(dim=-1)
    counts = torch.bincount(predictions, minlength=vocabulary_out)
    positive, negative = sorted(torch.topk(counts, k=2).indices.tolist())
    return Readout(positive, negative)


def list_mediators(model: HookedTransformer) -> list[Mediator]:
    """Every head and MLP output of the model, in the order of the forward pass."""
    mediators = []
    for layer in range(model.cfg.n_layers):
        mediators += [
            Mediator(f"blocks.{layer}.attn.hook_z", head)
            for head in range(model.cfg.n_heads)
        ]
        mediators.append(Mediator(f"blocks.{layer}.hook_mlp_out", None))
    return mediators


def load_programs(names: list[str] | None) -> dict:
    """Instances of the compiled programs of the InterpBench repository, by name.

    Imports the case modules one by one, so that a case's failure to import is raised with
    its name instead of hiding all the others.

    :param names: restrict to these program names; ``None`` loads every program.
    """
    from circuits_benchmark.benchmark import cases as cases_package
    from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase

    programs = {}
    for path in sorted(Path(cases_package.__file__).parent.glob("case_*.py")):
        if path.stem in NOT_COMPILED_PROGRAMS:
            continue
        module = importlib.import_module(f"{cases_package.__name__}.{path.stem}")
        for value in vars(module).values():
            # Only classes defined in this module, not the base classes it imports.
            if (
                isinstance(value, type)
                and issubclass(value, BenchmarkCase)
                and value.__module__ == module.__name__
            ):
                program = value()
                if names is None or program.get_name() in names:
                    programs[program.get_name()] = program
    return programs


def input_pairs(inputs: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Clean and corrupted input of each pair: row ``i`` and the next row of the
    program's own inputs, as tensors of shape (1, positions)."""
    rows = inputs.shape[0]
    return [
        (inputs[i : i + 1], inputs[(i + 1) % rows : (i + 1) % rows + 1])
        for i in range(min(PAIRS_PER_PROGRAM, rows - 1))
    ]


def mediator_value(cache: ActivationCache, mediator: Mediator) -> torch.Tensor:
    """Activation of a mediator at the last position, copied out of the cache."""
    activation = cache[mediator.hook_name]
    if mediator.head is None:
        return activation[0, -1].clone()
    return activation[0, -1, mediator.head].clone()


def patched_metric(
    model: HookedTransformer,
    tokens: torch.Tensor,
    mediator: Mediator,
    value: torch.Tensor,
    readout: Readout,
) -> float:
    """Metric of a run on ``tokens`` with the mediator set to ``value`` at the last
    position."""

    def overwrite(activation: torch.Tensor, hook) -> torch.Tensor:
        if mediator.head is None:
            activation[0, -1] = value
        else:
            activation[0, -1, mediator.head] = value
        return activation

    logits = model.run_with_hooks(tokens, fwd_hooks=[(mediator.hook_name, overwrite)])
    return readout(logits)


def pair_effects(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    mediators: list[Mediator],
    readout: Readout,
) -> tuple[list[float], list[float]]:
    """NIE and PIE of every mediator for one pair of inputs.

    NIE is the clean metric minus the metric of the clean run with the mediator taken from
    the corrupted run (noising); PIE is the metric of the corrupted run with the mediator
    taken from the clean run minus the corrupted metric (denoising).
    """

    def mediator_activations(name: str) -> bool:
        return name.endswith(("attn.hook_z", "hook_mlp_out"))

    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens, names_filter=mediator_activations
    )
    corrupted_logits, corrupted_cache = model.run_with_cache(
        corrupted_tokens, names_filter=mediator_activations
    )
    clean_metric, corrupted_metric = readout(clean_logits), readout(corrupted_logits)
    natural, pure = [], []
    for mediator in mediators:
        denoised = patched_metric(
            model,
            corrupted_tokens,
            mediator,
            mediator_value(clean_cache, mediator),
            readout,
        )
        noised = patched_metric(
            model,
            clean_tokens,
            mediator,
            mediator_value(corrupted_cache, mediator),
            readout,
        )
        pure.append(denoised - corrupted_metric)
        natural.append(clean_metric - noised)
    return natural, pure


def program_effects(program) -> tuple[np.ndarray, np.ndarray, dict]:
    """NIE and PIE of every mediator of one compiled program, arrays of shape (pairs,
    mediators) and a description of the program."""
    model = program.get_hl_model(device="cpu")
    model.eval()
    inputs = program.get_clean_data(
        min_samples=PAIRS_PER_PROGRAM + 2,
        max_samples=PAIRS_PER_PROGRAM + 2,
        encoded_dataset=True,
    ).get_inputs()
    readout = select_readout(model, inputs[:READOUT_PROBE_ROWS])
    mediators = list_mediators(model)
    natural, pure = [], []
    with torch.no_grad():
        for clean_tokens, corrupted_tokens in input_pairs(inputs):
            pair_natural, pair_pure = pair_effects(
                model, clean_tokens, corrupted_tokens, mediators, readout
            )
            natural.append(pair_natural)
            pure.append(pair_pure)
    description = {
        "layers": model.cfg.n_layers,
        "heads": model.cfg.n_heads,
        "output_symbols": model.cfg.d_vocab_out,
        "readout": asdict(readout),
    }
    return np.array(natural), np.array(pure), description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--programs", nargs="+", help="names of the programs to run (default: all)"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=OUTPUT_ROOT / "tracr_interactions"
    )
    arguments = parser.parse_args()

    natural_effects, pure_effects, descriptions = {}, {}, {}
    for name, program in load_programs(arguments.programs).items():
        natural_effects[name], pure_effects[name], descriptions[name] = program_effects(
            program
        )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output_directory / "natural_indirect_effects.npz", **natural_effects
    )
    np.savez(arguments.output_directory / "pure_indirect_effects.npz", **pure_effects)
    (arguments.output_directory / "programs.json").write_text(
        json.dumps(descriptions, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

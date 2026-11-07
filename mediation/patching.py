"""Cache the activations of attention heads and MLPs and run a prompt with some replaced.

A mediator is an :class:`AttentionHead` (its slice of the input to the attention output
projection) or an :class:`MlpLayer` (the output of the block's MLP). One implementation
serves every model family of :mod:`mediation.models`.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from mediation.models import LoadedModel


class AttentionHead(NamedTuple):
    """One attention head: its slice of the input to the attention output projection."""

    layer: int
    head: int

    @property
    def name(self) -> str:
        """The head as written in the paper, ``L9H9``."""
        return f"L{self.layer}H{self.head}"


class MlpLayer(NamedTuple):
    """The output of the MLP of one block."""

    layer: int


Component = AttentionHead | MlpLayer


def all_attention_heads(model: LoadedModel) -> list[AttentionHead]:
    """Every attention head of the model, sorted by layer and head."""
    return [
        AttentionHead(layer, head)
        for layer in range(model.num_layers)
        for head in range(model.num_heads)
    ]


@dataclass
class Activations:
    """Mediator activations of one prompt, or their mean over several prompts.

    :ivar head_inputs: head outputs before the attention output projection mixes them,
        shape (num_layers, positions, num_heads, head_dim).
    :ivar mlp_outputs: MLP outputs of the requested layers, each of shape
        (positions, hidden_size).
    :ivar logits: logits of the run, shape (1, positions, vocabulary); ``None`` for a mean.
    """

    head_inputs: torch.Tensor
    mlp_outputs: dict[int, torch.Tensor]
    logits: torch.Tensor | None

    @property
    def positions(self) -> int:
        return self.head_inputs.shape[1]

    def values(
        self, component: Component, positions: int | None = None
    ) -> torch.Tensor:
        """Activation of ``component`` over the first ``positions`` positions: shape
        (positions, head_dim) for a head and (positions, hidden_size) for an MLP."""
        if isinstance(component, AttentionHead):
            values = self.head_inputs[component.layer, :, component.head]
        else:
            values = self.mlp_outputs[component.layer]
        return values[:positions]


def _batch(tokens: Sequence[int]) -> torch.Tensor:
    return torch.tensor([list(tokens)], dtype=torch.long)


def cache_activations(
    model: LoadedModel, tokens: Sequence[int], mlp_layers: Collection[int] = ()
) -> Activations:
    """Run the prompt once and record every head's output and the MLPs of ``mlp_layers``.

    :param tokens: token ids of the prompt, one sequence.
    :param mlp_layers: layers whose MLP output is recorded; empty records none.
    """
    recorded_layers = set(mlp_layers)
    head_inputs, mlp_outputs = {}, {}
    with torch.no_grad(), model.model.trace(_batch(tokens)):
        for layer in range(model.num_layers):
            head_inputs[layer] = model.attention_output_input(layer).clone().save()
            if layer in recorded_layers:
                mlp_outputs[layer] = model.mlp_output(layer).clone().save()
        logits = model.logits().save()
    stacked = torch.stack([head_inputs[layer][0] for layer in range(model.num_layers)])
    return Activations(
        head_inputs=stacked.unflatten(-1, (model.num_heads, model.head_dim)),
        mlp_outputs={layer: value[0] for layer, value in mlp_outputs.items()},
        logits=logits,
    )


def _add_to_position_sums(sums: Activations, cached: Activations) -> None:
    """Add one prompt's activations to the leading positions of float64 running sums."""
    length = cached.positions
    sums.head_inputs[:, :length] += cached.head_inputs.double().cpu()
    for layer, value in cached.mlp_outputs.items():
        sums.mlp_outputs[layer][:length] += value.double().cpu()


def mean_activations(
    model: LoadedModel,
    token_sequences: Sequence[Sequence[int]],
    mlp_layers: Collection[int] = (),
) -> Activations:
    """Position-wise mean of the activations over prompts, for mean ablation.

    The mean at a position averages over the prompts that have that position, so prompts of
    different lengths are allowed. Sums are accumulated in float64 and the mean is returned
    in float32.

    :param token_sequences: the prompts whose activations are averaged.
    """
    longest = max(len(tokens) for tokens in token_sequences)
    sums = Activations(
        head_inputs=torch.zeros(
            (model.num_layers, longest, model.num_heads, model.head_dim),
            dtype=torch.float64,
        ),
        mlp_outputs={
            layer: torch.zeros((longest, model.hidden_size), dtype=torch.float64)
            for layer in mlp_layers
        },
        logits=None,
    )
    counts = torch.zeros(longest, dtype=torch.float64)
    for tokens in token_sequences:
        _add_to_position_sums(sums, cache_activations(model, tokens, mlp_layers))
        counts[: len(tokens)] += 1
    return Activations(
        head_inputs=(sums.head_inputs / counts[None, :, None, None]).float(),
        mlp_outputs={
            layer: (total / counts[:, None]).float()
            for layer, total in sums.mlp_outputs.items()
        },
        logits=None,
    )


def run_with_replacements(
    model: LoadedModel,
    tokens: Sequence[int],
    replacements: Mapping[Component, torch.Tensor],
) -> torch.Tensor:
    """Run a prompt with mediators overwritten and return the logits.

    :param replacements: new value of each patched component, shaped like
        :meth:`Activations.values`. A value with fewer positions than the prompt overwrites
        only the first positions; later positions keep their own activations.
    :return: logits of shape (1, positions, vocabulary).
    """
    # nnsight requires modules to be touched in execution order: within a block the
    # attention heads run before the MLP.
    ordered = sorted(replacements, key=lambda c: (c.layer, isinstance(c, MlpLayer), c))
    with torch.no_grad(), model.model.trace(_batch(tokens)):
        for component in ordered:
            values = replacements[component]
            positions = values.shape[0]
            if isinstance(component, AttentionHead):
                start = component.head * model.head_dim
                head_input = model.attention_output_input(component.layer)
                head_input[:, :positions, start : start + model.head_dim] = values
            else:
                model.mlp_output(component.layer)[:, :positions] = values
        logits = model.logits().save()
    return logits


def run_with_source(
    model: LoadedModel,
    tokens: Sequence[int],
    source: Activations,
    components: Collection[Component],
    positions: int | None = None,
) -> torch.Tensor:
    """Run a prompt with ``components`` replaced by their activations in ``source``.

    ``source`` is another prompt's cached activations (resample ablation) or
    :func:`mean_activations` (mean ablation).

    :param positions: number of leading positions replaced. Default is the shared prefix,
        ``min(len(tokens), source.positions)``.
    :return: logits of shape (1, positions, vocabulary).
    """
    if positions is None:
        positions = min(len(tokens), source.positions)
    replacements = {c: source.values(c, positions) for c in components}
    return run_with_replacements(model, tokens, replacements)

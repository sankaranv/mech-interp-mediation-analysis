"""The seven language models of the paper and uniform access to their attention heads and MLPs.

Every model is a decoder-only transformer whose attention output projection takes the
concatenated outputs of all heads. A head's output is therefore a contiguous slice of the
projection's input, and that slice (before the projection mixes heads) is what this library
patches. The three architecture families differ only in module names and in how the
projection is stored (GPT-2 uses a ``Conv1D``); :class:`LoadedModel` hides both.
"""

from dataclasses import dataclass
from functools import reduce

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    """A model of the paper, pinned to one revision of its Hugging Face repository.

    :ivar architecture: key of :data:`ARCHITECTURES`.
    :ivar tokenizer_family: selects the prompt files of the Gender Bias and Greater-Than
        tasks.
    """

    hugging_face_id: str
    revision: str
    dtype: torch.dtype
    architecture: str
    tokenizer_family: str


@dataclass(frozen=True)
class Architecture:
    """Dotted module paths, relative to the model root, of the parts the library reads or writes."""

    layers: str
    attention_output_projection: str  # relative to one layer
    mlp: str  # relative to one layer
    unembedding: str  # produces the logits


ARCHITECTURES = {
    "gpt2": Architecture("transformer.h", "attn.c_proj", "mlp", "lm_head"),
    "gpt_neox": Architecture("gpt_neox.layers", "attention.dense", "mlp", "embed_out"),
    # Llama and Qwen2.5 share these module names.
    "llama": Architecture("model.layers", "self_attn.o_proj", "mlp", "lm_head"),
}

# float32 everywhere except Llama-3.1-8B, which does not fit one GPU in float32.
MODELS = {
    "gpt2-small": ModelSpec(
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        torch.float32,
        "gpt2",
        "gpt2",
    ),
    "gpt2-medium": ModelSpec(
        "openai-community/gpt2-medium",
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        torch.float32,
        "gpt2",
        "gpt2",
    ),
    "gpt2-xl": ModelSpec(
        "openai-community/gpt2-xl",
        "15ea56dee5df4983c59b2538573817e1667135e2",
        torch.float32,
        "gpt2",
        "gpt2",
    ),
    "pythia-70m": ModelSpec(
        "EleutherAI/pythia-70m",
        "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        torch.float32,
        "gpt_neox",
        "pythia",
    ),
    "pythia-6.9b": ModelSpec(
        "EleutherAI/pythia-6.9b",
        "c0e3eee36dc47af0c49f361c74cfe459c09f7f23",
        torch.float32,
        "gpt_neox",
        "pythia",
    ),
    "qwen2.5-0.5b": ModelSpec(
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        torch.float32,
        "llama",
        "qwen25",
    ),
    "llama-3.1-8b": ModelSpec(
        "meta-llama/Llama-3.1-8B",
        "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        torch.bfloat16,
        "llama",
        "llama31",
    ),
}


class LoadedModel:
    """An nnsight model with its tokenizer and architecture-independent module access.

    The ``access`` methods return nnsight proxies and are only valid inside
    ``model.trace(...)``, in the order in which the forward pass executes them.

    :ivar model: the nnsight ``LanguageModel``.
    :ivar tokenizer: the model's tokenizer with its default special tokens (a beginning-of-
        sequence token is prepended for Llama-3.1 only).
    :ivar num_layers: number of transformer blocks.
    :ivar num_heads: number of query heads per block (the layout of the attention output
        projection's input; for grouped-query attention this is not the number of key heads).
    :ivar head_dim: width of one head's slice of the projection input.
    """

    def __init__(self, spec: ModelSpec, model: LanguageModel, tokenizer):
        self.spec = spec
        self.model = model
        self.tokenizer = tokenizer
        self.architecture = ARCHITECTURES[spec.architecture]
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.num_heads
        )
        # GPT-2's projection is a Conv1D; depending on the nnsight version its input is the
        # tensor itself or a one-element tuple, which is probed once here.
        self._input_is_tuple = (
            spec.architecture == "gpt2" and not self._input_is_tensor()
        )

    def _input_is_tensor(self) -> bool:
        dummy = torch.tensor([self.tokenizer.encode("Hello world")])
        try:
            with self.model.scan(dummy):
                _ = self._layer(
                    0, self.architecture.attention_output_projection
                ).input.shape
        except Exception:
            return False
        return True

    def _layer(self, layer: int, relative_path: str):
        block = reduce(getattr, self.architecture.layers.split("."), self.model)[layer]
        return reduce(getattr, relative_path.split("."), block)

    def attention_output_input(self, layer: int):
        """Proxy for the input of the attention output projection: shape (batch, position,
        num_heads * head_dim), with head ``h`` in columns ``[h * head_dim, (h + 1) * head_dim)``."""
        proxy = self._layer(layer, self.architecture.attention_output_projection).input
        return proxy[0] if self._input_is_tuple else proxy

    def mlp_output(self, layer: int):
        """Proxy for the MLP output of a block: shape (batch, position, hidden_size)."""
        return self._layer(layer, self.architecture.mlp).output

    def logits(self):
        """Proxy for the logits: shape (batch, position, vocabulary)."""
        return reduce(
            getattr, self.architecture.unembedding.split("."), self.model
        ).output


def load_tokenizer(name: str):
    """Tokenizer of a model of :data:`MODELS` at its pinned revision, default special tokens."""
    spec = MODELS[name]
    return AutoTokenizer.from_pretrained(spec.hugging_face_id, revision=spec.revision)


def load_model(
    name: str, attention_implementation: str | None = None, device_map: str = "auto"
) -> LoadedModel:
    """Load a model of :data:`MODELS` in the dtype and revision the paper used.

    :param name: key of :data:`MODELS`.
    :param attention_implementation: ``None`` keeps the library default (scaled dot-product
        attention), which the per-head estimands of Figure 1 used. Pass ``"eager"`` where the
        published run used it: the circuit knockout of Section 4, the interpolation of
        Figure 2 and the mean activations of Appendix C. The two implementations differ at
        float32 rounding level, which is below the printed precision.
    :param device_map: forwarded to ``from_pretrained``.
    """
    spec = MODELS[name]
    options = {"dtype": spec.dtype, "revision": spec.revision}
    if attention_implementation is not None:
        options["attn_implementation"] = attention_implementation
    model = LanguageModel(
        spec.hugging_face_id, device_map=device_map, dispatch=True, **options
    )
    return LoadedModel(spec, model, load_tokenizer(name))

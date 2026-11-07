import pytest
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from mediation.models import LoadedModel, ModelSpec


@pytest.fixture(scope="session")
def tokenizer():
    """The GPT-2 tokenizer (a small download on first use)."""
    try:
        return AutoTokenizer.from_pretrained("openai-community/gpt2")
    except OSError:
        pytest.skip("GPT-2 tokenizer is not available")


@pytest.fixture(scope="session")
def tiny_model(tokenizer, tmp_path_factory) -> LoadedModel:
    """A randomly initialised GPT-2 with 3 layers of 4 heads, wrapped like a real model."""
    torch.manual_seed(0)
    config = GPT2Config(n_layer=3, n_head=4, n_embd=32, vocab_size=len(tokenizer))
    directory = tmp_path_factory.mktemp("tiny_gpt2")
    GPT2LMHeadModel(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    spec = ModelSpec("tiny", "none", torch.float32, "gpt2", "gpt2")
    model = LanguageModel(str(directory), device_map="cpu", dispatch=True)
    return LoadedModel(spec, model, tokenizer)

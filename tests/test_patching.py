import pytest
import torch

from mediation.estimands import (
    component_effects,
    cross_interaction,
    head_effects,
    joint_outcomes,
    patched_metric,
)
from mediation.patching import (
    AttentionHead,
    MlpLayer,
    cache_activations,
    mean_activations,
    run_with_replacements,
    run_with_source,
)
from mediation.prompt_pair import PromptPair

CLEAN = [464, 3290, 2540, 284, 262, 2156, 11, 3875]
COUNTERFACTUAL = [464, 4731, 1816, 284, 262, 3504, 11, 6823]


def last_logit_difference(logits):
    return logits[..., -1, 10] - logits[..., -1, 20]


@pytest.fixture
def pair():
    return PromptPair(CLEAN, COUNTERFACTUAL, last_logit_difference)


def logits_with_zeroed_head_by_hook(model, tokens, layer, head):
    """Reference: zero one head's columns with a plain PyTorch pre-hook on c_proj."""
    projection = model.model._model.transformer.h[layer].attn.c_proj

    def zero_head(module, arguments):
        values = arguments[0].clone()
        values[..., head * model.head_dim : (head + 1) * model.head_dim] = 0
        return (values,)

    handle = projection.register_forward_pre_hook(zero_head)
    try:
        with torch.no_grad():
            return model.model._model(torch.tensor([tokens])).logits
    finally:
        handle.remove()


def test_head_replacement_matches_plain_hook(tiny_model):
    # Checks the head layout (which columns of the projection input are head h).
    for layer, head in [(0, 0), (1, 3), (2, 2)]:
        zeros = torch.zeros(len(CLEAN), tiny_model.head_dim)
        patched = run_with_replacements(
            tiny_model, CLEAN, {AttentionHead(layer, head): zeros}
        )
        reference = logits_with_zeroed_head_by_hook(tiny_model, CLEAN, layer, head)
        assert torch.allclose(patched, reference, atol=1e-5)


def test_patching_own_activations_changes_nothing(tiny_model):
    own = cache_activations(tiny_model, CLEAN, mlp_layers=[0, 1, 2])
    components = [AttentionHead(1, 2), AttentionHead(1, 0), MlpLayer(1), MlpLayer(2)]
    patched = run_with_source(tiny_model, CLEAN, own, components)
    assert torch.allclose(patched, own.logits, atol=1e-5)


def test_resampling_a_head_from_the_counterfactual_changes_the_output(tiny_model):
    clean = cache_activations(tiny_model, CLEAN)
    counterfactual = cache_activations(tiny_model, COUNTERFACTUAL)
    logits = run_with_source(tiny_model, CLEAN, counterfactual, [AttentionHead(0, 1)])
    assert not torch.allclose(logits, clean.logits, atol=1e-4)


def test_single_component_effects_agree_with_joint_outcomes(tiny_model, pair):
    components = [AttentionHead(0, 2), MlpLayer(1)]
    effects = component_effects(tiny_model, pair, components)
    source = cache_activations(
        tiny_model,
        COUNTERFACTUAL,
        mlp_layers=[
            component.layer
            for component in components
            if isinstance(component, MlpLayer)
        ],
    )
    for index, component in enumerate(components):
        y_empty = patched_metric(tiny_model, CLEAN, pair, source, [])
        y_patched = patched_metric(tiny_model, CLEAN, pair, source, [component])
        assert y_empty - y_patched == pytest.approx(effects.nie[index], abs=1e-4)


def test_interaction_is_nie_minus_pie(tiny_model, pair):
    effects = head_effects(tiny_model, pair)
    assert effects.nie.shape == (3, 4)
    assert (effects.interaction == effects.nie - effects.pie).all()


def test_theorem_four_one_identity_and_sign(tiny_model, pair):
    source = cache_activations(tiny_model, COUNTERFACTUAL, mlp_layers=[1])
    first = [AttentionHead(0, 0), AttentionHead(0, 1)]
    second = [AttentionHead(2, 3), MlpLayer(1)]
    outcomes = joint_outcomes(tiny_model, pair, source, first, second)
    nie_first = outcomes.y_empty - outcomes.y_first
    nie_second = outcomes.y_empty - outcomes.y_second
    nie_both = outcomes.y_empty - outcomes.y_both
    assert nie_both == pytest.approx(
        nie_first + nie_second + outcomes.cross_interaction
    )
    # A random network has interacting mediators, so the identity is not vacuous.
    assert abs(outcomes.cross_interaction) > 1e-6


def test_cross_interaction_sign():
    # Two mediators that must both be replaced to remove the effect: Y drops only when both
    # are replaced, so replacing both removes more than the sum of the parts.
    assert cross_interaction(y_empty=1.0, y_first=1.0, y_second=1.0, y_both=0.0) == 1.0
    # Serial dependence (the second mediator's replacement is redundant after the first):
    # the joint effect is smaller than the sum, xINT is negative.
    assert cross_interaction(y_empty=1.0, y_first=0.0, y_second=0.0, y_both=0.0) == -1.0


def test_mean_activations_of_one_prompt_is_the_prompt(tiny_model):
    own = cache_activations(tiny_model, CLEAN, mlp_layers=[1])
    mean = mean_activations(tiny_model, [CLEAN, CLEAN], mlp_layers=[1])
    assert torch.allclose(mean.head_inputs, own.head_inputs, atol=1e-6)
    assert torch.allclose(mean.mlp_outputs[1], own.mlp_outputs[1], atol=1e-6)

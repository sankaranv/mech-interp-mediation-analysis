import pandas as pd
import pytest
import torch

from mediation.models import load_tokenizer
from mediation.tasks import load_prompt_pairs
from mediation.tasks.gender_bias import DATA_DIRECTORY as GENDER_BIAS_DIRECTORY
from mediation.tasks.gender_bias import build_gender_bias_dataframe
from mediation.tasks.greater_than import greater_than_metric, year_token_ids
from mediation.tasks.ioi import _swap_names, ioi_metric, load_ioi_pairs
from mediation.tasks.sva import load_sva_pairs


def test_swap_names_exchanges_whole_words_only():
    text = "Then, Ann and Anna went to the store. Anna gave a drink to"
    assert _swap_names(text, "Ann", "Anna") == (
        "Then, Anna and Ann went to the store. Ann gave a drink to"
    )


def test_ioi_metric_is_indirect_object_minus_subject():
    logits = torch.zeros(1, 3, 10)
    logits[0, -1, 4] = 5.0
    logits[0, -1, 7] = 2.0
    assert ioi_metric(logits, io_token_id=4, subject_token_id=7).item() == 3.0


def test_ioi_pairs_are_reproducible_and_balanced(tokenizer):
    pairs = load_ioi_pairs(tokenizer, count=40)
    assert [p.clean_tokens for p in pairs] == [
        p.clean_tokens for p in load_ioi_pairs(tokenizer, count=40)
    ]
    assert sorted(p.metadata["template"] for p in pairs).count("ABBA") == 20
    # Under GPT-2's vocabulary every name is one token, so no pair has unequal length.
    assert len(load_ioi_pairs(tokenizer, count=40, equal_length_only=True)) == 40


def test_symmetric_counterfactual_swaps_the_two_names(tokenizer):
    pair = load_ioi_pairs(tokenizer, "symmetric", count=10)[0]
    clean, swapped = pair.clean_tokens, pair.counterfactual_tokens
    io, subject = pair.metadata["position_io"], pair.metadata["position_subject"]
    assert swapped[io] == clean[subject] and swapped[subject] == clean[io]


def test_sva_pairs_have_equal_length_and_are_distinct(tokenizer):
    pairs = load_sva_pairs(tokenizer, count=200)
    assert all(len(p.clean_tokens) == len(p.counterfactual_tokens) for p in pairs)
    assert (
        len({(tuple(p.clean_tokens), tuple(p.counterfactual_tokens)) for p in pairs})
        == 200
    )


def test_greater_than_metric_splits_the_years_at_the_start_year(tokenizer):
    year_ids = year_token_ids(tokenizer)
    logits = torch.zeros(1, 2, len(tokenizer))
    logits[0, -1, year_ids[60:]] = 1.0
    # Start year 49: endings 50 to 99 average 40 / 50, endings 00 to 49 average 0.
    assert greater_than_metric(logits, 49, year_ids).item() == pytest.approx(0.8)


def pinned_tokenizer(model_name):
    try:
        return load_tokenizer(model_name)
    except OSError:
        pytest.skip(f"Tokenizer of {model_name} is not available")


@pytest.mark.parametrize(
    "model_name, family", [("qwen2.5-0.5b", "qwen25"), ("llama-3.1-8b", "llama31")]
)
def test_shipped_gender_bias_files_are_the_builder_output(model_name, family):
    built = build_gender_bias_dataframe(pinned_tokenizer(model_name))
    assert built.equals(pd.read_csv(GENDER_BIAS_DIRECTORY / f"{family}.csv"))


@pytest.mark.parametrize(
    "model_name, task, count",
    [
        ("gpt2-small", "ioi_pabc", 1000),
        ("gpt2-small", "gender_bias", 986),
        ("gpt2-small", "greater_than", 1000),
        ("pythia-70m", "ioi_pabc", 691),
        ("pythia-70m", "ioi_symmetric", 1000),
        ("pythia-70m", "gender_bias", 782),
        ("pythia-70m", "sva", 1000),
    ],
)
def test_pair_counts_of_the_paper(model_name, task, count):
    pinned_tokenizer(model_name)
    pairs = load_prompt_pairs(task, model_name)
    assert len(pairs) == count
    assert all(len(p.clean_tokens) == len(p.counterfactual_tokens) for p in pairs)

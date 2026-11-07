import pytest
import torch

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

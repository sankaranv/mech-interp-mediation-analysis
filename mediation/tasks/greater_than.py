"""Greater-Than prompt pairs and metric, as released by Hanna et al.

A clean prompt has the form "The war lasted from the year 1732 to the year 17" and the
counterfactual prompt replaces the two-digit start year YY with 01. The model should
continue with a two-digit year above YY. The files hold 1000 pairs for the GPT-2 tokenizer
and 1000 for Pythia's, so GPT-2 small, medium and XL share one and Pythia-70M and
Pythia-6.9B the other. Only tokenizers that emit each two-digit ending as one token are
supported: GPT-2 and Pythia.
"""

from functools import partial
from pathlib import Path

import pandas as pd
import torch

from mediation.prompt_pair import PromptPair, tokenize_equal_length

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "greater_than"


def year_token_ids(tokenizer) -> torch.Tensor:
    """Token id of each two-digit year ending 00 to 99, shape (100,).

    :raises ValueError: if an ending is not exactly one token (taking the first id would
        score a prefix of the ending) or two endings share a token.
    """
    ids = []
    for year in range(100):
        encoded = tokenizer.encode(f"{year:02d}")
        if len(encoded) != 1:
            raise ValueError(f"Year ending {year:02d} is {len(encoded)} tokens")
        ids.append(encoded[0])
    if len(set(ids)) != 100:
        raise ValueError("Two year endings share a token")
    return torch.tensor(ids)


def greater_than_metric(
    logits: torch.Tensor, start_year: int, year_ids: torch.Tensor
) -> torch.Tensor:
    """Mean logit of the endings above YY minus mean logit of the endings 00 to YY.

    Scored at the last position over the 100 year-ending tokens, as Hanna et al.'s
    ``logit[year + 1 :].mean() - logit[: year + 1].mean()``. YY comes from the clean prompt
    and is held fixed in every run of the pair.

    :param start_year: YY, the clean prompt's two-digit start year.
    :param year_ids: output of :func:`year_token_ids`.
    """
    year_logits = logits[..., -1, :][..., year_ids.to(logits.device)]
    above = year_logits[..., start_year + 1 :].mean(dim=-1)
    at_or_below = year_logits[..., : start_year + 1].mean(dim=-1)
    return above - at_or_below


def load_greater_than_pairs(tokenizer, tokenizer_family: str) -> list[PromptPair]:
    """Every pair of the tokenizer family's file, in file order.

    :param tokenizer_family: ``gpt2`` or ``pythia``; selects the file.
    :raises ValueError: if a year ending is not a single token, or a pair's prompts differ
        in token length.
    """
    year_ids = year_token_ids(tokenizer)
    frame = pd.read_csv(DATA_DIRECTORY / f"{tokenizer_family}.csv")
    pairs = []
    for row in frame.itertuples():
        clean_tokens, counterfactual_tokens = tokenize_equal_length(
            tokenizer, row.clean, row.corrupted
        )
        pairs.append(
            PromptPair(
                clean_tokens=clean_tokens,
                counterfactual_tokens=counterfactual_tokens,
                metric=partial(
                    greater_than_metric,
                    start_year=int(row.correct_idx),
                    year_ids=year_ids,
                ),
                metadata={"start_year": int(row.correct_idx)},
            )
        )
    return pairs

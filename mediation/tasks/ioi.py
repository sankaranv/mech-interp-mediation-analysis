"""Indirect Object Identification prompt pairs (Wang et al., 2023).

A clean prompt such as "Then, Kelly and Jesse were working at the office. Jesse decided to
give a drink to" has the indirect object (IO, Kelly) and the repeated subject (S, Jesse).
The metric is the logit of the IO name minus the logit of the S name at the last position.
Two counterfactual prompts are available:

* ``pabc``: Wang et al.'s ABC prompt, which replaces all three names with new names;
* ``symmetric``: the IO and S names swapped within the clean prompt.
"""

import json
import random
import re
from functools import partial
from pathlib import Path

import torch

from mediation.prompt_pair import PromptPair

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "ioi"

COUNTERFACTUALS = ("pabc", "symmetric")


def ioi_metric(
    logits: torch.Tensor, io_token_id: int, subject_token_id: int
) -> torch.Tensor:
    """Logit of the indirect object minus logit of the subject at the last position."""
    return logits[..., -1, io_token_id] - logits[..., -1, subject_token_id]


def _swap_names(text: str, first_name: str, second_name: str) -> str:
    """Exchange two whole-word names in ``text``."""
    placeholder = "__SWAP__"
    text = re.sub(r"\b" + re.escape(first_name) + r"\b", placeholder, text)
    text = re.sub(r"\b" + re.escape(second_name) + r"\b", first_name, text)
    return text.replace(placeholder, second_name)


def _name_positions(
    clean_tokens: list[int], io_id: int, subject_id: int
) -> tuple[int, int, int] | None:
    """Positions of the IO token, the first subject token and the repeated subject token
    in the clean prompt, or ``None`` if the prompt lacks one of them."""
    io_positions = [i for i, token in enumerate(clean_tokens) if token == io_id]
    subject_positions = [
        i for i, token in enumerate(clean_tokens) if token == subject_id
    ]
    if not io_positions or len(subject_positions) < 2:
        return None
    return io_positions[0], subject_positions[0], subject_positions[1]


def _single_token_id(tokenizer, name: str) -> int | None:
    """Token id of ``name`` if it is one token (special tokens excluded, so that Llama's
    beginning-of-sequence token does not count), else ``None``."""
    ids = tokenizer.encode(name, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _counterfactual_text(record: dict, counterfactual: str) -> str:
    if counterfactual == "pabc":
        return record["patch_prefix"]
    return _swap_names(
        record["clean_prefix"],
        record["clean_answer"].strip(),
        record["patch_answer"].strip(),
    )


def _parse_records(
    records: list[dict], tokenizer, counterfactual: str
) -> list[PromptPair]:
    """Convert records to prompt pairs, dropping those the tokenizer cannot represent.

    A record is kept only if the IO and S names are each one token and the clean prompt
    contains the IO token and the S token twice.
    """
    pairs = []
    for record in records:
        io_id = _single_token_id(tokenizer, record["clean_answer"])
        subject_id = _single_token_id(tokenizer, record["patch_answer"])
        if io_id is None or subject_id is None:
            continue
        clean_tokens = tokenizer.encode(record["clean_prefix"])
        positions = _name_positions(clean_tokens, io_id, subject_id)
        if positions is None:
            continue
        pairs.append(
            PromptPair(
                clean_tokens=clean_tokens,
                counterfactual_tokens=tokenizer.encode(
                    _counterfactual_text(record, counterfactual)
                ),
                metric=partial(
                    ioi_metric, io_token_id=io_id, subject_token_id=subject_id
                ),
                metadata={
                    "template": record["case"].split("_")[0].upper(),
                    "position_io": positions[0],
                    "position_subject": positions[1],
                    "position_subject_repeat": positions[2],
                    "position_end": len(clean_tokens) - 1,
                },
            )
        )
    return pairs


def load_ioi_pairs(
    tokenizer,
    counterfactual: str = "pabc",
    count: int = 1000,
    seed: int = 42,
    equal_length_only: bool = False,
) -> list[PromptPair]:
    """Sample IOI prompt pairs for a tokenizer: half ABBA, half BABA, shuffled.

    Records whose IO or S name is not a single token under ``tokenizer`` are dropped before
    sampling, so the sampled pairs can differ between tokenizers.

    :param counterfactual: ``"pabc"`` or ``"symmetric"``.
    :param count: number of pairs before ``equal_length_only`` filtering; ``count // 2``
        are drawn from each template with ``random.Random(seed)``.
    :param equal_length_only: keep only the sampled pairs whose clean and counterfactual
        prompts have the same number of tokens. The paper applies this to the pABC prompt
        under Pythia's tokenizer, where 309 of 1000 sampled pairs have a replacement name
        of several tokens, leaving 691. Without it, patching the shared prefix would align
        activations of different words.
    """
    if counterfactual not in COUNTERFACTUALS:
        raise ValueError(f"counterfactual must be one of {COUNTERFACTUALS}")
    templates = []
    for file_name in ("eng_ioi_abba.json", "eng_ioi_baba.json"):
        records = json.loads((DATA_DIRECTORY / file_name).read_text())
        templates.append(_parse_records(records, tokenizer, counterfactual))

    generator = random.Random(seed)
    half = count // 2
    sampled = [generator.sample(pairs, min(half, len(pairs))) for pairs in templates]
    pairs = sampled[0] + sampled[1]
    generator.shuffle(pairs)
    if equal_length_only:
        pairs = [
            pair
            for pair in pairs
            if len(pair.clean_tokens) == len(pair.counterfactual_tokens)
        ]
    return pairs

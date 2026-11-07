"""Gender Bias prompt pairs: Vig et al.'s counterfactual as released by Hanna et al.

The clean prompt names a profession ("The poet was promoted because") and the counterfactual
prompt replaces it with "man" or "woman" (Vig et al.'s set-gender intervention). The clean
answer is " he" for a male-stereotyped profession and " she" otherwise, and the metric is the
logit of the clean answer minus the logit of the other pronoun at the last position.

Hanna et al. distribute the pairs for the GPT-2 and Pythia tokenizers. The files for the
Qwen2.5 and Llama-3.1 tokenizers are built by :func:`build_gender_bias_dataframe`, which
repeats their construction, and are shipped rather than regenerated.
"""

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mediation.prompt_pair import PromptPair, tokenize_equal_length

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "gender_bias"

# Templates of Vig et al. (2020), as listed in Hanna et al.'s data code.
TEMPLATES = [
    "The {} said that",
    "The {} yelled that",
    "The {} whispered that",
    "The {} wanted that",
    "The {} desired that",
    "The {} wished that",
    "The {} ate because",
    "The {} ran because",
    "The {} drove because",
    "The {} slept because",
    "The {} cried because",
    "The {} laughed because",
    "The {} went home because",
    "The {} stayed up because",
    "The {} was fired because",
    "The {} was promoted because",
    "The {} yelled because",
]


def logit_difference_metric(
    logits: torch.Tensor, clean_answer_id: int, counterfactual_answer_id: int
) -> torch.Tensor:
    """Logit of the clean answer minus logit of the other pronoun at the last position."""
    return logits[..., -1, clean_answer_id] - logits[..., -1, counterfactual_answer_id]


def _pronoun_ids(tokenizer) -> tuple[int, int]:
    """Token ids of " he" and " she"; each must be a single token."""
    ids = []
    for pronoun in (" he", " she"):
        encoded = tokenizer.encode(pronoun, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{pronoun!r} is {len(encoded)} tokens, not one")
        ids.append(encoded[0])
    return ids[0], ids[1]


def _read_professions(gender: str) -> list[str]:
    """Professions stereotyped as ``gender`` (``"male"`` or ``"female"``)."""
    path = DATA_DIRECTORY / f"professions_{gender}_stereo_filtered.json"
    return [entry[0] for entry in json.loads(path.read_text())]


def _prompt_rows(generator: random.Random) -> list[tuple[str, str, int]]:
    """Every template filled with every profession, shuffled.

    :return: (clean prompt, counterfactual prompt, label) with label 1 for a
        male-stereotyped profession, whose counterfactual names "woman".
    """
    male = _read_professions("male")
    professions = male + _read_professions("female")
    generator.shuffle(professions)
    rows = [
        (
            template.format(profession),
            template.format("woman" if profession in male else "man"),
            int(profession in male),
        )
        for template in TEMPLATES
        for profession in professions
    ]
    generator.shuffle(rows)
    return rows


def _balance_labels(frame: pd.DataFrame, numpy_generator) -> pd.DataFrame:
    """Shuffle, then keep as many pairs of each label as the rarer label has."""
    frame = frame.sample(frac=1, random_state=numpy_generator)
    rarer_label_count = frame["label"].value_counts().min()
    balanced = pd.concat(
        [frame[frame["label"] == label].head(rarer_label_count) for label in (0, 1)]
    )
    return balanced.sample(frac=1, random_state=numpy_generator).reset_index(drop=True)


def build_gender_bias_dataframe(tokenizer, seed: int = 0) -> pd.DataFrame:
    """Hanna et al.'s construction of the Gender Bias pairs for one tokenizer.

    Every template is filled with every profession (the male- and female-stereotyped lists
    in ``data/gender_bias``). Pairs whose clean and counterfactual prompts differ in token
    length are dropped, then the two labels are balanced.

    :param seed: seeds both the profession shuffle and the row sampling.
    :return: columns clean, corrupted, clean_answer_idx, corrupted_answer_idx and label
        (1: male-stereotyped profession, clean answer " he").
    """
    he_id, she_id = _pronoun_ids(tokenizer)
    records = [
        {
            "clean": clean,
            "corrupted": corrupted,
            "clean_answer_idx": he_id if label else she_id,
            "corrupted_answer_idx": she_id if label else he_id,
            "label": label,
        }
        for clean, corrupted, label in _prompt_rows(random.Random(seed))
        if len(tokenizer.encode(clean)) == len(tokenizer.encode(corrupted))
    ]
    return _balance_labels(pd.DataFrame(records), np.random.default_rng(seed))


def load_gender_bias_pairs(tokenizer, tokenizer_family: str) -> list[PromptPair]:
    """Every pair of the tokenizer family's file, in file order (986 for GPT-2, Qwen2.5 and
    Llama-3.1, 782 for Pythia).

    :param tokenizer_family: ``gpt2``, ``pythia``, ``qwen25`` or ``llama31``; selects the file.
    :raises ValueError: if the file's answer ids are not the tokenizer's " he" and " she",
        or a pair's prompts differ in token length.
    """
    frame = pd.read_csv(DATA_DIRECTORY / f"{tokenizer_family}.csv")
    he_id, she_id = _pronoun_ids(tokenizer)
    answer_ids = set(frame["clean_answer_idx"]) | set(frame["corrupted_answer_idx"])
    if answer_ids != {he_id, she_id}:
        raise ValueError(f"File answer ids {answer_ids} differ from {he_id, she_id}")
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
                    logit_difference_metric,
                    clean_answer_id=int(row.clean_answer_idx),
                    counterfactual_answer_id=int(row.corrupted_answer_idx),
                ),
            )
        )
    return pairs

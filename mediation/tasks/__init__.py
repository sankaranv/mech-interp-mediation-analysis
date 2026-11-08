"""Prompt pairs and metrics of the five tasks of the paper.

======================  ==========================================================
``ioi_pabc``            Indirect Object Identification, all three names replaced
``ioi_symmetric``       Indirect Object Identification, indirect object and subject swapped
``sva``                 subject-verb agreement
``gender_bias``         Vig et al.'s gender bias counterfactual
``greater_than``        Greater-Than
======================  ==========================================================
"""

from mediation.models import MODELS, load_tokenizer
from mediation.prompt_pair import PromptPair
from mediation.tasks.gender_bias import load_gender_bias_pairs
from mediation.tasks.greater_than import load_greater_than_pairs
from mediation.tasks.ioi import load_ioi_pairs
from mediation.tasks.sva import load_sva_pairs

TASKS = ("ioi_pabc", "ioi_symmetric", "sva", "gender_bias", "greater_than")


def load_prompt_pairs(task: str, model_name: str) -> list[PromptPair]:
    """The prompt pairs the paper uses for ``task`` under the tokenizer of a model.

    The Pythia pABC pairs are filtered to equal clean and counterfactual length (691 of the
    1000 sampled); every other task and tokenizer uses all of its pairs.

    :param model_name: key of :data:`mediation.models.MODELS`.
    """
    tokenizer = load_tokenizer(model_name)
    family = MODELS[model_name].tokenizer_family
    if task == "ioi_pabc":
        return load_ioi_pairs(tokenizer, "pabc", equal_length_only=family == "pythia")
    if task == "ioi_symmetric":
        return load_ioi_pairs(tokenizer, "symmetric")
    if task == "sva":
        return load_sva_pairs(tokenizer)
    if task == "gender_bias":
        return load_gender_bias_pairs(tokenizer, family)
    if task == "greater_than":
        return load_greater_than_pairs(tokenizer, family)
    raise ValueError(f"task must be one of {TASKS}")


def pair_count(task: str, model_name: str) -> int:
    """Number of prompt pairs the paper uses for ``task`` under the tokenizer of a model."""
    return len(load_prompt_pairs(task, model_name))

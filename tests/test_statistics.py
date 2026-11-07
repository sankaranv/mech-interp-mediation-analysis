import numpy as np
import pytest

from mediation.statistics import (
    mean_with_confidence_interval,
    rank_agreement,
    signed_ranks,
    standard_error,
)


def test_confidence_interval_uses_sample_standard_deviation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mean, half_width = mean_with_confidence_interval(values)
    assert mean == 2.5
    assert half_width == pytest.approx(1.96 * np.std(values, ddof=1) / 2)


def test_standard_error_uses_population_standard_deviation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert standard_error(values) == pytest.approx(np.std(values, ddof=0) / 2)


def test_signed_ranks_put_the_largest_signed_value_first():
    ranks = signed_ranks(np.array([[0.5, -3.0], [2.0, 0.0]]))
    assert ranks.tolist() == [[2, 4], [1, 3]]


def test_rank_agreement_is_over_per_head_means():
    generator = np.random.default_rng(0)
    nie = generator.normal(size=(50, 3, 4))
    assert rank_agreement(nie, nie) == pytest.approx(1.0)
    assert rank_agreement(nie, -nie) == pytest.approx(-1.0)

import numpy as np
import pytest

from mediation.results import load_pair_ranges, parse_pair_range, save_pair_range


def test_ranges_are_concatenated_in_pair_order(tmp_path):
    save_pair_range(tmp_path, 3, 5, nie=np.array([3.0, 4.0]))
    save_pair_range(tmp_path, 0, 3, nie=np.array([0.0, 1.0, 2.0]))
    assert load_pair_ranges(tmp_path)["nie"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_missing_range_is_an_error(tmp_path):
    save_pair_range(tmp_path, 0, 2, nie=np.zeros(2))
    save_pair_range(tmp_path, 3, 5, nie=np.zeros(2))
    with pytest.raises(ValueError, match="pairs 2 to 2 missing"):
        load_pair_ranges(tmp_path)


def test_rows_must_match_range(tmp_path):
    with pytest.raises(ValueError):
        save_pair_range(tmp_path, 0, 3, nie=np.zeros(2))


def test_parse_pair_range():
    assert parse_pair_range("0:250") == (0, 250)
    with pytest.raises(ValueError):
        parse_pair_range("5:5")

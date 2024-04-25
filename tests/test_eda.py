"""Test module for eda.py"""

import pytest
from src.eda import make_stat_df


CORPUS = [
    "here is the first document",
    "then the second document",
    "and a third element is finishing the list"
]

@pytest.mark.parametrize("words_list", [CORPUS])
@pytest.mark.parametrize("name", ["words"])
@pytest.mark.parametrize("verbose", [False])
def test_make_stat_df(words_list, name, verbose):
    """Test src.eda.make_stat_df function"""
    stats_df = make_stat_df(words_list, name, verbose)

    assert stats_df.count_words[10] == 3
    assert stats_df.freq_words[10] == pytest.approx(0.1764706)

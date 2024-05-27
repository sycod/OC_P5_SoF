"""Test module for models.py, except for LDA functions"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models import score_terms
from src.models import results_from_vec_matrix
from src.models import get_5_tags_from_matrix
from src.models import get_5_tags_from_array


TEST_WORDS_LIST = ["string", "python", "api", "regex", "selenium"]
TEST_ARRAY = np.array([0.1, 0.2, 0.3, 0.05, 0.12])
# CSR matrix
indptr = np.array([0, 1, 2, 3, 4, 5])
indices = np.array([0, 1, 2, 3, 4])
data = np.array([0.1, 0.2, 0.3, 0.05, 0.12])
TEST_X = csr_matrix((data, indices, indptr), shape=(5, 5))


@pytest.mark.parametrize("pred_words", [["test", "string", "words", "python", "api"]])
@pytest.mark.parametrize("target_words", [TEST_WORDS_LIST, ["image", "python"]])
def test_score_terms(pred_words, target_words):
    """Test src.models.score_terms function"""
    if "image" in target_words:
        assert score_terms(pred_words, target_words) == 0.5
    else:
        assert score_terms(pred_words, target_words) == 0.6


@pytest.mark.parametrize("X", [TEST_X])
@pytest.mark.parametrize("words_list", [TEST_WORDS_LIST])
def test_results_from_vec_matrix(words_list, X):
    """Test src.models.results_from_vec_matrix function"""
    preds = results_from_vec_matrix(words_list, X)
    assert preds == [('api', 0.3, 2), ('python', 0.2, 1), ('selenium', 0.12, 4), ('string', 0.1, 0), ('regex', 0.05, 3)]


@pytest.mark.parametrize("X", [TEST_X])
@pytest.mark.parametrize("words_list", [TEST_WORDS_LIST])
def test_get_5_tags_from_matrix(words_list, X):
    """Test src.models.get_5_tags_from_matrix function"""
    tags = get_5_tags_from_matrix(words_list, X)
    assert tags == ['api', 'python', 'selenium', 'string', 'regex']


# score_jaccard
# select_split_data
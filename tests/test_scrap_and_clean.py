"""Test module for scrap_and_clean.py"""

import pytest
from src.scrap_and_clean import get_languages
from src.scrap_and_clean import clean_string
from src.scrap_and_clean import clean_hashes
from src.scrap_and_clean import clean_negation
from src.scrap_and_clean import trim_punct
from src.scrap_and_clean import tokenize_str
from src.scrap_and_clean import words_filter


TEST_STRING = "This is a test string. It contains some <code>code</code> and <img src='img'> and sometimes <other> <unusual> tags, \n newlines \n, UPPERCASE WORDS, suspension dots... isolated numbers 4 654  or 9142 and punctuation ; /*+ and     multiple    spaces  and a+, C++, C#, .QL or even S programming langages."


def test_get_languages():
    """Test src.scrap_and_clean get_languages function"""

    lang = get_languages()
    assert len(lang) > 0
    assert "python" in lang
    assert "c++" in lang
    assert ".ql" in lang
    assert "zpl" in lang


@pytest.mark.parametrize("string", [TEST_STRING])
def test_clean_string(string):
    """Test src.scrap_and_clean clean_string function"""

    assert clean_string(string) == "this is a test string. it contains some and and sometimes tags, newlines , uppercase words, suspension dots isolated numbers or and punctuation ; /*+ and multiple spaces and a+, c++, c#, .ql or even s programming langages."


@pytest.mark.parametrize("tokens", [["c", "#", "g", "#", "c++", ".ql", "s", "#"]])
@pytest.mark.parametrize("watch_list", [["c++", ".ql", "c#"]])
def test_clean_hashes(tokens, watch_list):
    """Test src.scrap_and_clean clean_hashes function"""
    assert clean_hashes(tokens, watch_list) == ["c#", "g", "#", "c++", ".ql", "s", "#"]


# def clean_negation(tokens, excluded_list): -> list
# def tokenize_str(sentence, watch_list, excluded_list, punctuation): -> list
# def lemmatize_tokens(tokens_list, keep_set, exclude_set) -> list
# def words_filter(words_list, method, keep_set, exclude_set) -> tuple
# def preprocess_doc(document, keep_set, exclude_set, punctuation) -> str
    # preprocess_doc test string contains sometimes tags newlines uppercase word suspension dot isolated numbers punctuation multiple spaces a+ c++ c# .ql even programming langages
# def preprocess_data(df_raw) -> pd.DataFrame

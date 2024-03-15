"""Test module for scrap_and_clean.py"""

import pytest
from src.scrap_and_clean import get_languages, clean_string


TEST_STRING = "This is a test string. It contains some <code>code</code> and <img src='img'> and sometimes <other> <unusual> tags, \n newlines \n, UPPERCASE WORDS, suspension dots... lonely numbers 4 654  or 9142 and punctuation ; /*+ and     multiple    spaces  and a+, C++, C#, .QL or even S programming langages."

@pytest.mark.parametrize("string", [TEST_STRING])
def test_clean_string(string):
    """Test src.scrap_and_clean clean_string function"""

    assert clean_string(string) == "this is a test string. it contains some and and sometimes tags, newlines , uppercase words, suspension dots lonely numbers or and punctuation ; /*+ and multiple spaces and a+, c++, c#, .ql or even s programming langages."


def test_get_languages():
    """Test src.scrap_and_clean get_languages function"""

    lang = get_languages()
    assert len(lang) > 0
    assert "python" in lang
    assert "c++" in lang
    assert ".ql" in lang
    assert "zpl" in lang
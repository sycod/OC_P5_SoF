"""Test module for scrap_and_clean.py"""

import pytest
import pandas as pd
from src.scrap_and_clean import get_languages
from src.scrap_and_clean import clean_string
from src.scrap_and_clean import clean_hashes
from src.scrap_and_clean import tokenize_str
from src.scrap_and_clean import lemmatize_tokens
from src.scrap_and_clean import clean_tokens
from src.scrap_and_clean import words_filter
from src.scrap_and_clean import preprocess_doc
from src.scrap_and_clean import preprocess_data


TEST_STRING = "This is a test string. It contains some <code>code</code> and <img src='img'> and sometimes <other> <unusual> tags, \n newlines \n, UPPERCASE WORDS, suspension dots... isolated numbers 4 654  or 9142 and punctuation ; /*+ and     multiple    spaces  and a+, C++, C#, .QL or even S programming langages."
TEST_KEEP_SET = {"c++", ".ql", "c#"}
TEST_EXCLUDE_SET = {"is", "a", "sometimes", "and", "langage", "a+"}
TEST_PUNCTUATION = ["'", '"', ",", ".", ";", ":", "?", "!", "+", "..", "''", "``", "||", "\\\\", "\\", "==", "+=", "-=", "-", "_", "=", "(", ")", "[", "]", "{", "}", "<", ">", "/", "|", "&", "*", "%", "$", "#", "@", "`", "^", "~"]
TEST_TOKENS = ['string', 'contains', 'some', 'code', 'other', 'unusual', 'tags', 'newlines', 'uppercase', 'words', 'dots', 'isolated', 'numbers', 'punctuation', 'multiple', 'and', 'c++', '.ql', 'even', 'programming', '-langages']

# raw dataframe simulation (only used features)
t1 = "ITMS-91053: Missing API declaration - Privacy"
b1 = """<p>Why am I all of a suddent getting this on successful builds with Apple?</p>\n<pre><code>Although submission for App Store review was successful [blablabla] For more details about this policy, including a list of required reason APIs and approved reasons for usage, visit: https://blabla.com.\n</code></pre>\n"""
tags1 = "<ios><app-store><plist><appstore-approval><privacy-policy>"
score1 = 12
answercount1 = 1
view1 = 111
date1 = "2024-01-01 13:37:01"

t2 = "Why is builtin sorted() slower for a list containing descending numbers if each number appears twice consecutively?"
b2 = """<p>I sorted four similar lists. List <code>d</code> consistently takes much longer than the others, which all take about the same time:</p>\n<pre class="lang-none prettyprint-override"><code>a:  33.5 ms\nb:  33.4 ms\nc:  36.4 ms\nd: 110.9 ms\n</code></pre>\n<p>Why is that?</p>\n<p>Test script (<a href="https://blabla.com" rel="noreferrer">Attempt This Online!</a>):</p>\n<pre class="lang-py prettyprint-override"><code>from timeit import repeat\n\nn = 2_000_000\n [blablabla] print(f\'{name}: {time*1e3 :5.1f} ms\')\n</code></pre>\n"""
tags2 = "<python><algorithm><performance><sorting><time-complexity>"
score2 = 21
answercount2 = 2
view2 = 222
date2 = "2023-02-02 13:37:02"

data = [[t1, b1, tags1, score1, answercount1, view1, date1], [t2, b2, tags2, score2, answercount2, view2, date2]]
TEST_DF_RAW = pd.DataFrame(data, columns=["Title", "Body", "Tags", "Score", "AnswerCount", "ViewCount", "CreationDate"])


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
    """Test src.scrap_and_clean.clean_string function"""
    assert clean_string(string) == "this is a test string. it contains some and and sometimes tags, newlines , uppercase words, suspension dots isolated numbers or and punctuation ; /*+ and multiple spaces and a+, c++, c#, .ql or even s programming langages."


@pytest.mark.parametrize("tokens", [["c", "#", "g", "#", "c++", ".ql", "s", "#"]])
@pytest.mark.parametrize("watch_list", [TEST_KEEP_SET])
def test_clean_hashes(tokens, watch_list):
    """Test src.scrap_and_clean.clean_hashes function"""
    assert clean_hashes(tokens, watch_list) == ["c#", "g", "#", "c++", ".ql", "s", "#"]


@pytest.mark.parametrize("sentence", [TEST_STRING])
@pytest.mark.parametrize("keep_set", [TEST_KEEP_SET])
@pytest.mark.parametrize("exclude_set", [TEST_EXCLUDE_SET])
@pytest.mark.parametrize("punctuation", [TEST_PUNCTUATION])
def test_tokenize_str(sentence, keep_set, exclude_set, punctuation):
    """Test src.scrap_and_clean.tokenize_str function"""
    result = ['This', 'test', 'string', 'contains', 'some', 'code', 'code', '/code', 'img', "src='img", 'other', 'unusual', 'tags', 'newlines', 'UPPERCASE', 'WORDS', 'suspension', 'dots', 'isolated', 'numbers', 'punctuation', 'multiple', 'spaces', 'C++', '.QL', 'even', 'programming', 'langages']
    assert tokenize_str(sentence, keep_set, exclude_set, punctuation) == result


@pytest.mark.parametrize("tokens", [TEST_TOKENS])
@pytest.mark.parametrize("keep_set", [TEST_KEEP_SET])
@pytest.mark.parametrize("exclude_set", [TEST_EXCLUDE_SET])
def test_lemmatize_tokens(tokens, keep_set, exclude_set):
    """Test src.scrap_and_clean.lemmatize_tokens function"""
    result = ['string', 'contains', 'some', 'code', 'other', 'unusual', 'tag', 'newlines', 'uppercase', 'word', 'dot', 'isolated', 'number', 'punctuation', 'multiple', 'c++', '.ql', 'even', 'programming', '-langages']
    assert lemmatize_tokens(tokens, keep_set, exclude_set) == result


@pytest.mark.parametrize("tokens", [TEST_TOKENS])
@pytest.mark.parametrize("keep_set", [TEST_KEEP_SET])
@pytest.mark.parametrize("exclude_set", [TEST_EXCLUDE_SET])
def test_clean_tokens(tokens, keep_set, exclude_set):
    """Test src.scrap_and_clean.clean_tokens function"""
    result = ['string', 'contains', 'some', 'code', 'other', 'unusual', 'tags', 'newlines', 'uppercase', 'words', 'dots', 'isolated', 'numbers', 'punctuation', 'multiple', 'c++', '.ql', 'even', 'programming', 'langages']
    assert clean_tokens(tokens, keep_set, exclude_set) == result


@pytest.mark.parametrize("words_list", ["c#", "removed"])
@pytest.mark.parametrize("method", [["rm", "add"]])
@pytest.mark.parametrize("keep_set", [TEST_KEEP_SET])
@pytest.mark.parametrize("exclude_set", [TEST_EXCLUDE_SET])
def test_words_filter(words_list, method, keep_set, exclude_set):
    """Test src.scrap_and_clean.words_filter function"""
    _ = words_filter(words_list, method, keep_set, exclude_set)
    if method == "add":
        assert "c#" in _[1]
        assert "c#" not in _[0]
        assert "removed" in _[1]
        assert "removed" not in _[1]
    if method == "rm":
        assert "c#" in _[0]
        assert "c#" not in _[1]
        assert "removed" in _[0]
        assert "removed" not in _[1]


@pytest.mark.parametrize("document", [TEST_STRING])
@pytest.mark.parametrize("keep_set", [TEST_KEEP_SET])
@pytest.mark.parametrize("exclude_set", [TEST_EXCLUDE_SET])
@pytest.mark.parametrize("punctuation", [TEST_PUNCTUATION])
def test_preprocess_doc(document, keep_set, exclude_set, punctuation):
    """Test src.scrap_and_clean.preprocess_doc function"""
    result = "this test string contains some tag newlines uppercase word suspension dot isolated number punctuation multiple space c++ c# .ql even programming langages"
    
    assert preprocess_doc(document, keep_set, exclude_set, punctuation) == result


@pytest.mark.parametrize("tags_n_min", [1])
@pytest.mark.parametrize("df_raw", [TEST_DF_RAW])
def test_preprocess_data(df_raw, tags_n_min):
    """Test src.scrap_and_clean.preprocess_data function"""
    _ = preprocess_data(df_raw, tags_n_min=tags_n_min)

    assert _.shape == (2, 10)
    
    assert _["title_bow"][0] == "itms-91053 missing api declaration privacy"
    assert _["title_bow"][1] == "builtin sorted slower list containing descending number number appears twice consecutively"

    assert _["body_bow"][0] == "suddent successful build apple"
    assert _["body_bow"][1] == "sorted four similar list list consistently take much longer others take time test script attempt online"
    assert _["doc_bow"][0] == "itms-91053 missing api declaration privacy suddent successful build apple"
    assert _["doc_bow"][1] == "builtin sorted slower list containing descending number number appears twice consecutively sorted four similar list list consistently take much longer others take time test script attempt online"

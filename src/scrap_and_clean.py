"""Utils for web scraping and data cleaning"""

import re
import urllib.request, json 
from html.parser import HTMLParser
import nltk
import emoji


class LangParser(HTMLParser):
    """Parse names from an extracted HTML"""
    def __init__(self):
        HTMLParser.__init__(self)
        self.recording = 0
        self.data = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.recording = 1

    def handle_endtag(self, tag):
        if tag == "a":
            self.recording = 0

    def handle_data(self, data):
        if self.recording:
            # avoid adding edit links
            if data.strip() != "edit":
                # remove everything between parentheses
                data = re.sub(r'\([^)]*\)', '', data)
                # remove start & end white spaces + convert to lower case
                data = data.strip().lower()
                self.data.add(data)


def get_languages() -> set:
    """Return the list of programming languages from Wikipedia"""
    url = "https://en.wikipedia.org/w/api.php?action=parse&page=List_of_programming_languages&prop=text&format=json&disabletoc=1&formatversion=2"
    with urllib.request.urlopen(url) as response:
        scrap = json.load(response)["parse"]["text"]

    # get all programming languages ever declared on Wikipedia
    start = scrap.find('<h2><span class="mw-headline" id="A')
    end = scrap.find('<h2><span class="mw-headline" id="See_also')
    scrap = scrap[start:end]

    parser = LangParser()
    parser.feed(scrap)
    prog_langs = set(parser.data)
    
    return prog_langs


def clean_string(string) -> str:
    """Return a cleaned version of an input string"""
    # remove code tags
    string = re.sub(r"<code>.*?<\/code>", "", string, flags=re.S)
    # remove img tags
    string = re.sub(r"<img.*?>", "", string)
    # remove all html tags
    string = re.sub(r"<.*?>", "", string)
    # remove emojis
    string = emoji.replace_emoji(string, replace=' ')
    # remove newlines
    string = re.sub(r"\n", " ", string)
    # lowercase
    string = string.lower()
    # remove suspension points
    string = re.sub(r"\.\.\.", " ", string)
    # remove digits only tokens
    string = re.sub(r"\b\d+\b", " ", string)
    # remove multiple spaces
    string = re.sub(r" +", " ", string)

    return string


def clean_hashes(tokens, keep_set) -> list:
    """Return a list of tokens with hashes cleaned
    (NLTK tokenizer issue with programming languages names containing hashes)
    """

    i_offset = 0
    for i, t in enumerate(tokens):
        i -= i_offset
        if t == '#' and i > 0:
            left = tokens[:i-1]
            joined = [tokens[i - 1] + t]
            right = tokens[i + 1:]
            if joined[0] in keep_set:
                tokens = left + joined + right
                i_offset += 1

    return tokens


def clean_negation(tokens, exclude_set) -> list:
    """Return a list of tokens with negations cleaned"""
    i_offset = 0
    for i, t in enumerate(tokens):
        i -= i_offset
        if t == "n't" and i > 0:
            left = tokens[:i-1]
            joined = [tokens[i - 1] + t]
            right = tokens[i + 1:]
            if joined[0] in exclude_set:
                tokens = left + right
                i_offset += 2

    return tokens


def trim_punct(tokens, punctuation, keep_set) -> list:
    """Return a list of tokens with punctuation trimmed,
    apart from words appearing in keep_set.
    """

    tokens_trimmed = []
    for t in tokens:
        if t[0] in punctuation and t not in keep_set and len(t) > 2:
            # because many specific terms begin with a "." followed by a letter
            if t[0] == "." and t[1] not in punctuation:
                pass
            if t[0] == "_" and t[1] == "_":
                pass
            else:
                t = t[1:]
        if t[-1] in punctuation and t not in keep_set and len(t) > 2:
            t = t[:-1]
        # second check for words starting with an apostrophe
        if t[0] == "'" and t not in keep_set and len(t) > 2:
            t = t[1:]
        
        tokens_trimmed.append(t)

    return tokens_trimmed


def splitter_cell(list_of_strings, char=str) -> list:
    """Split a string from a list into a list of substrings using a delimiter"""
    sub = []
    for s in list_of_strings:
        _ = list(filter(None, s.split(char)))
        if len(_) > 0: sub.extend(_)

    return sub


def tokenize_str(sentence, keep_set, exclude_set, punctuation) -> list:
    """Return a list of cleansed tokens from a string,  excluding some words"""
    # tokenize except excluded words
    tokens = nltk.word_tokenize(sentence)

    # clean negations
    tokens = clean_negation(tokens, exclude_set)

    # remove hashes from watch list
    tokens = clean_hashes(tokens, keep_set)

    # trim punctuation once
    tokens_trimmed = trim_punct(tokens, punctuation, keep_set)

    # split tokens with specific characters
    tokens_split_back = splitter_cell(tokens_trimmed, "\\")
    tokens_split_slash = splitter_cell(tokens_split_back, "/")
    tokens_split_apo = splitter_cell(tokens_split_slash, "'")

    # trim punctuation again
    tokens_trim_again = trim_punct(tokens_split_apo, punctuation, keep_set)

    # remove (< 3)-letter words apart from those appearing in keep_set
    tokens_rm_inf3 = [t for t in tokens_trim_again if len(t) > 2 or t in keep_set]

    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_inf3 if t not in exclude_set]

    return tokens_cleaned


def words_filter(list, method, keep_set, exclude_set) -> tuple:
    """Add or remove a list of words to the corpus:
    - if method is 'add', add them to necessary words and remove them from excluded words
    - if method is 'rm', add them to excluded words and remove them from necessary words
    """
    for i in list:
        if method == "add":
            keep_set.add(i)
            exclude_set.discard(i)
        elif method == "rm":
            keep_set.discard(i)
            exclude_set.add(i)
        else:
            print("Method should be 'add' or 'rm'")

    return keep_set, exclude_set


if __name__ == "__main__":
    print(f"\n👉 get_languages() -> set\n{get_languages.__doc__}")
    print(f"\n👉 clean_string(string, excluded=None) -> str\n{clean_string.__doc__}")
    print(f"\n👉 clean_hashes(tokens, keep_set) -> list\n{clean_hashes.__doc__}")
    print(f"\n👉 clean_negation(tokens, exclude_set) -> list\n{clean_negation.__doc__}")
    print(f"\n👉 trim_punct(tokens, punctuation, keep_set) -> list\n{clean_negation.__doc__}")
    print(f"\n👉 tokenize_str(sentence, keep_set, exclude_set, punctuation) -> list\n{tokenize_str.__doc__}")
    print(f"\n👉 words_filter(list, method, keep_set, exclude_set) -> None\n{tokenize_str.__doc__}")
    
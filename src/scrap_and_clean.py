"""Utils for web scraping and data cleaning"""

import re
import urllib.request, json 
from html.parser import HTMLParser


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


def get_languages():
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
    prog_langs = parser.data
    
    return prog_langs


def clean_string(string):
    """Return a cleaned version of an input string"""
    # remove code tags
    string = re.sub(r"<code>.*?<\/code>", "", string, flags=re.S)
    # remove img tags
    string = re.sub(r"<img.*?>", "", string)
    # remove all html tags
    string = re.sub(r"<.*?>", "", string)
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

    # 🚧 commented because seems pointless: leave it to the tokenizer
    # keep only letters, digits, spaces and some useful characters
    # string = re.sub(r"[^\w *'#+_.-]", " ", string)

    # 🚧 commented because seems pointless and may add biases
    # words = string.split()
    # for w in words:
    #     if excluded:
    #         if w not in excluded:
    #             # remove 1-letter words
    #             if len(w) == 1:
    #                 words.remove(w)
    #             # remove points at the end of words
    #             if len(w) > 1 and w[-1] == ".":
    #                 words[words.index(w)] = w[:-1]
    # string = " ".join(words)

    return string


def rm_ending_punctuation(list):
    """Return a list of strings without ending punctuation"""
    return [re.sub(r'[.,;:]$', '', w) for w in list]


def exclude_words(string, excluded):
    """Return a string with excluded words removed"""
    words = rm_ending_punctuation(string.split())
    clean_words = words.copy()
    for w in words:
        if w in excluded or w == "":
            clean_words.remove(w)

    return " ".join(clean_words)


if __name__ == "__main__":
    print(f"\n👉 get_languages()\n{get_languages.__doc__}")
    print(f"\n👉 clean_string(string, excluded=None)\n{clean_string.__doc__}")
    print(f"\n👉 rm_ending_punctuation(list)\n{rm_ending_punctuation.__doc__}")
    print(f"\n👉 exclude_words(string, excluded)\n{exclude_words.__doc__}")
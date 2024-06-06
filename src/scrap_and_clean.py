"""Utils for web scraping and data cleaning"""

import os
from dotenv import load_dotenv
import dill as pickle
import urllib.request, json 
import pandas as pd
import re
from html.parser import HTMLParser
import nltk
import emoji
import logging


# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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


def init_raw_df() -> pd.DataFrame:
    """Get initial data and return raw dataframe"""
    load_dotenv()
    DATA_URL = os.getenv("DATA_URL")

    data_dir = "data"
    data_file = os.path.join(data_dir, "data_raw.pkl")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(data_file):
        logging.info(f"Loading data from {DATA_URL}...")
        df_raw = pd.read_csv(DATA_URL)
        with open(data_file, "wb") as f:
            pickle.dump(df_raw, f)
            logging.info(f"✅ Raw data saved")
    else:
        logging.info(f"Loading data from local file...")
        with open(data_file, "rb") as f:
            df_raw = pickle.load(f)
            logging.info(f"✅ Raw data loaded")

    return df_raw


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
    string = re.sub(r"\b(?<![0-9-])(\d+)(?![0-9-])\b", " ", string)
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


def tokenize_str(sentence, keep_set, exclude_set) -> list:
    """Return a list of cleansed tokens from a string,  excluding some words"""
    # tokenize except excluded 
    nltk.download('punkt')  # downloaded just once, either checks update
    tokens = nltk.word_tokenize(sentence)

    # remove hashes from watch list
    tokens = clean_hashes(tokens, keep_set)

    # remove (< 3)-letter words apart from those appearing in keep_set
    tokens_rm_inf3 = [t for t in tokens if len(t) > 2 or t in keep_set]

    # remove tokens containing absolutely no letter
    tokens_rm_no_letter = list(filter(lambda s:any([c.isalpha() for c in s]), tokens_rm_inf3))

    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_no_letter if t not in exclude_set]

    return tokens_cleaned


def lemmatize_tokens(tokens_list, keep_set, exclude_set) -> list:
    """Return lemmatized tokens from a tokens list, on conditions"""
    nltk.download('wordnet')    # downloaded just once, either checks update
    kilmister = nltk.wordnet.WordNetLemmatizer()
    lem_tok_list = []

    for token in tokens_list:
        if token in keep_set:
            lem_tok_list.append(token)
        else:
            lem_tok = kilmister.lemmatize(token)
            if lem_tok not in exclude_set:
                lem_tok_list.append(lem_tok)

    return lem_tok_list


def clean_tokens(tokens_list, keep_set, exclude_set) -> list:
    """Return cleansed tokens from a tokens list"""
    # clean " ' " in front of certain words
    clean_apo = []
    clean_apo += [t[1:] if t[0] == "'" else t for t in tokens_list]

    # clean " - " in front of certain words
    clean_dash = []
    clean_dash += [t[1:] if t[0] == "-" else t for t in clean_apo]

    # remove (< 3)-letter words apart from those belonging to keep_set
    tokens_rm_inf3 = [t for t in clean_dash if len(t) > 2 or t in keep_set]

    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_inf3 if t not in exclude_set]

    return tokens_cleaned


def words_filter(words_list, method, keep_set, exclude_set) -> tuple:
    """Add or remove a list of words to the corpus:
    - if method is 'add', add them to necessary words and remove them from excluded words
    - if method is 'rm', add them to excluded words and remove them from necessary words
    """
    for i in words_list:
        if method == "add":
            keep_set.add(i)
            exclude_set.discard(i)
        elif method == "rm":
            keep_set.discard(i)
            exclude_set.add(i)
        else:
            logging.warning("Method should be 'add' or 'rm'")

    return keep_set, exclude_set


def preprocess_doc(document, keep_set, exclude_set) -> str:
    """Apply a sequence of words formatting actions on a document and returns a preprocessed string."""
    doc_clean = clean_string(document)
    doc_tokens = tokenize_str(doc_clean, keep_set, exclude_set)
    doc_lemmed = lemmatize_tokens(doc_tokens, keep_set, exclude_set)
    doc_tk_clean = clean_tokens(doc_lemmed, keep_set, exclude_set)
    doc_preprocessed = " ".join(doc_tk_clean)

    return doc_preprocessed


def preprocess_data(df_raw, tags_n_min=10) -> pd.DataFrame:
    """Return a preprocessed dataframe from a raw dataframe"""
    df = df_raw.copy()

    PUNCTUATION = ["'", '"', ",", ".", ";", ":", "?", "!", "+", "..", "''", "``", "||", "\\\\", "\\", "==", "+=", "-=", "-", "_", "=", "(", ")", "[", "]", "{", "}", "<", ">", "/", "|", "&", "*", "%", "$", "#", "@", "`", "^", "~"]

    # updated from multiple trials → list differs from the EDA notebook list
    EXCLUDED_TERMS = ["can't", "d'oh", "could't", "could'nt", "cound't", "cound'nt", "coulnd't", "cdn'ed", "doesn'it", "does't", "don'ts", "n't", "'nt", "i'ca", "i'ts", "should't", "want", "would", "would't", "might't", "must't", "need't", "n'th", "wont't", "non", "no", "use", "using", "usage", "code", "like", "issue", "error", "file", "files", "run", "runs", "create", "created", 'between', 't', 'any', 'using', 'this', 'out', 'm', 'file', 'each', 's', "'ve", "work", "way", "following", "problem", "tried", "also", "need", "trying", "example", "question", "value", "know", "application", "see", "new", "could", "however", "working", "change", "something", "used", "found", "result", "help", "quot", "running", "first", "seems", "without", "different", "two", "still", "look", "possible", "getting", "able", "even", "fine", "instead", "library", "answer", "another", "thanks", "read", "since", "inside", "idea", "every", "added"]

    # KEPT TOKENS SET
    # tags
    df["Tags"] = df["Tags"].apply(lambda x: x[1:-1].split("><")[:5])
    tags = set()
    df["Tags"].apply(lambda x: tags.update(set(x)))
    # programming languages
    prog_lang = get_languages()
    # all together
    keep_set = prog_lang | tags
    # add some specific terms
    add_spec_terms = ["qt", "d3", "hoa", "kde", "s+", "hy", "d"]
    keep_set |= set(add_spec_terms)

    # EXCLUDED TOKENS SET
    nltk.download('stopwords')  # downloaded just once, either checks update
    exclude_set = set(nltk.corpus.stopwords.words("english"))
    exclude_set |= set(PUNCTUATION)
    exclude_set |= set(EXCLUDED_TERMS)

    # PREPROCESSING
    # titles
    df["title_bow"] = df["Title"].apply(lambda x: preprocess_doc(x, keep_set, exclude_set))
    # bodies
    df["body_bow"] = df["Body"].apply(lambda x: preprocess_doc(x, keep_set, exclude_set))
    # suppression des lignes vides
    df = df.loc[
        (df["title_bow"].apply(lambda x: x.strip() != ""))
        | (df["body_bow"].apply(lambda x: x.strip() != "")),
    ]
    # corpus
    df["doc_bow"] = df["title_bow"] + " " + df["body_bow"]

    # explode DF by tag
    df_exploded = df.explode('Tags')

    # count tags and keep only those with at least 10 occurrences
    vc = df_exploded.Tags.value_counts()
    tags_se_10 = vc[vc >= tags_n_min].index

    # filter dataframe on tags with at least 10 occurrences
    df_exploded = df_exploded[df_exploded.Tags.isin(tags_se_10)]

    # regroup by corpus
    df_pp = df_exploded.groupby("doc_bow", as_index=False, sort=False).agg(
        tags=("Tags", " ".join),
        score=("Score", "first"),
        answers=("AnswerCount", "first"),
        views=("ViewCount", "first"),
        date=("CreationDate", "first"),
        title_bow=("title_bow", "first"),
        title=("Title", "first"),
        body_bow=("body_bow", "first"),
        body=("Body", "first"),
    )

    return df_pp


def init_data():
    """Get preprocessed data from file or retrieve raw data and preprocess it"""
    if os.path.exists("data/data_preprocessed.pkl"):
        with open("data/data_preprocessed.pkl", "rb") as f:
            df_pp = pickle.load(f)
            logging.info(f"✅ Preprocessed data loaded")
    else:
        df_raw = init_raw_df()
        logging.info(f"Preprocessing raw data...")
        df_pp = preprocess_data(df_raw)
        with open("data/data_preprocessed.pkl", "wb") as f:
            pickle.dump(df_pp, f)
            logging.info(f"✅ Preprocessed data saved")

    return df_pp


if __name__ == "__main__":
    help()
    
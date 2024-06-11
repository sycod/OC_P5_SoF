"""Streamlit simple app to predict tags from a StackOverflow-like question."""

import os
import dill as pickle
import logging
import streamlit as st
from streamlit_tags import st_tags
from streamlit_tags import st_tags_sidebar
import re
import emoji

# ML
import numpy as np
from gensim.models import Word2Vec
import nltk


# CONFIG
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# NLTK downloads (just once, not downloaded if up-to-date)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
# ML tools
VECTORIZER_URI = "models/w2v_cbow_vectorizer"
CLASSIFIER_URI = "models/w2v_cbow_lrovr_classifier.pkl"
KEEP_SET_URI = "data/keep_set.pkl"
EXCLUDE_SET_URI = "data/exclude_set.pkl"
# placeholders
TITLE_PLACEHOLDER = "example: pandas merge with Python >3.5"
BODY_PLACEHOLDER = """example:
How do I add NaNs for missing rows after a merge?
How do I get rid of NaNs after merging?
I've seen these recurring questions asking about various facets of the pandas merge functionality, the aim here is to collate some of the more important points for posterity."""
TAGS_PLACEHOLDER = "see five predicted tags here"


# CHECK ML TOOLS & SETUP SESSION STATE
# load keep set (for preprocessing)
if "keep_set" not in st.session_state:
    logging.info(f"⚙️ Loading keep set...")
    if os.path.exists(KEEP_SET_URI):
        with open(KEEP_SET_URI, "rb") as f:
            keep_set = pickle.load(f)
        st.session_state.keep_set = keep_set
        logging.info(f"✅ Keep set loaded")
    else:
        logging.warning(f"⚠️ No keep set found ⚠️")
# load exclude set (for preprocessing)
if "exclude_set" not in st.session_state:
    logging.info(f"⚙️ Loading exclude set...")
    if os.path.exists(EXCLUDE_SET_URI):
        with open(EXCLUDE_SET_URI, "rb") as f:
            exclude_set = pickle.load(f)
        st.session_state.exclude_set = exclude_set
        logging.info(f"✅ Exclude set loaded")
    else:
        logging.warning(f"⚠️ No exclude set found ⚠️")
# load vectorizer
if "vectorizer" not in st.session_state:
    logging.info(f"⚙️ Loading vectorizer...")
    if os.path.exists(VECTORIZER_URI):
        vectorizer = Word2Vec.load(VECTORIZER_URI)
        st.session_state.vectorizer = vectorizer
        logging.info(f"✅ Vectorizer loaded")
    else:
        logging.warning(f"⚠️ No vectorizer found ⚠️")
# load classifier
if "classifier" not in st.session_state:
    logging.info(f"⚙️ Loading classifier...")
    if os.path.exists(CLASSIFIER_URI):
        with open(CLASSIFIER_URI, "rb") as f:
            classifier = pickle.load(f)
        st.session_state.classifier = classifier
        logging.info(f"✅ Classifier loaded")
    else:
        logging.warning(f"⚠️ No classifier found ⚠️")
# placeholders (if not in session state)
if "title_input" not in st.session_state:
    st.session_state.title_input = ""
if "body_input" not in st.session_state:
    st.session_state.body_input = ""
if "predicted_tags" not in st.session_state:
    st.session_state.predicted_tags = TAGS_PLACEHOLDER
if "message" not in st.session_state:
    st.session_state.message = None


# update session state on inputs
def update_title():
    st.session_state.title_input = st.session_state.title
def update_body():
    st.session_state.body_input = st.session_state.body


def check_length(input_doc, length=5) -> bool:
    """Check input length"""
    length_ok = True if len(input_doc.split(" ")) >= length else False
    
    return length_ok


def preprocess_doc(document, keep_set, exclude_set) -> str:
    """Preprocess document for a complete cleaning"""
    # CLEAN DOC
    # remove code tags
    document = re.sub(r"<code>.*?<\/code>", "", document, flags=re.S)
    # remove img tags
    document = re.sub(r"<img.*?>", "", document)
    # remove all html tags
    document = re.sub(r"<.*?>", "", document)
    # remove emojis
    document = emoji.replace_emoji(document, replace=" ")
    # remove newlines
    document = re.sub(r"\n", " ", document)
    # lowercase
    document = document.lower()
    # remove suspension points
    document = re.sub(r"\.\.\.", " ", document)
    # remove digits only tokens
    document = re.sub(r"\b(?<![0-9-])(\d+)(?![0-9-])\b", " ", document)
    # remove multiple spaces
    document = re.sub(r" +", " ", document)


    # TOKENIZE DOC
    # tokenize except excluded
    tokens = nltk.word_tokenize(document)
    # remove hashes from watch list
    i_offset = 0
    for i, t in enumerate(tokens):
        i -= i_offset
        if t == "#" and i > 0:
            left = tokens[: i - 1]
            joined = [tokens[i - 1] + t]
            right = tokens[i + 1 :]
            if joined[0] in keep_set:
                tokens = left + joined + right
                i_offset += 1
    # remove (< 3)-letter words apart from those appearing in keep_set
    tokens_rm_inf3 = [t for t in tokens if len(t) > 2 or t in keep_set]
    # remove tokens containing absolutely no letter
    tokens_rm_no_letter = list(
        filter(lambda s: any([c.isalpha() for c in s]), tokens_rm_inf3)
    )
    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_no_letter if t not in exclude_set]


    # LEMMATIZE TOKENS
    kilmister = nltk.wordnet.WordNetLemmatizer()
    lem_tok_list = []

    for token in tokens_cleaned:
        if token in keep_set:
            lem_tok_list.append(token)
        else:
            lem_tok = kilmister.lemmatize(token)
            if lem_tok not in exclude_set:
                lem_tok_list.append(lem_tok)


    # CLEAN TOKENS
    # clean " ' " in front of certain words
    clean_apo = []
    clean_apo += [t[1:] if t[0] == "'" else t for t in lem_tok_list]
    # clean " - " in front of certain words
    clean_dash = []
    clean_dash += [t[1:] if t[0] == "-" else t for t in clean_apo]
    # remove (< 3)-letter words apart from those belonging to keep_set
    tokens_rm_inf3 = [t for t in clean_dash if len(t) > 2 or t in keep_set]
    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_inf3 if t not in exclude_set]

    doc_preprocessed = " ".join(tokens_cleaned)

    return doc_preprocessed


def w2v_vect_data(model, matrix) -> np.array:
    """From a Word2Vec vectorizer, return a vectorized matrix"""
    # loop over rows in tokenized X_train
    doc_vectors = []
    for tokens in matrix:
        # loop over tokens in each row
        doc_vec = []
        for token in tokens:
            if token in model.wv:
                doc_vec.append(model.wv[token])
        # mean it
        doc_vectors.append(np.mean(doc_vec, axis=0))
    # get X_train matrix
    vector_matrix = np.array(doc_vectors)

    return vector_matrix


def lr_predict_tags(model, X, n_tags=5) -> list:
    """Use logistic regression probabilities to get at least n predicted tags"""
    ppbs = model.predict_proba(X)
    classes = model.classes_
    pred_tags = []

    for i, x in enumerate(X):
        # create list of tags from n first classes
        pred_list = (
            (" ")
            .join([classes[c] for c in ppbs[i].argsort()[: -n_tags - 1 : -1]])
            .split(" ")
        )
        # keep only 5 first tags
        pred = set()
        j = 0
        while len(pred) < 5:
            pred.add(pred_list[j])
            j += 1
        # add tags to predictions list
        pred_tags.append((" ").join(pred))

    return pred_tags


def predict_tags(input_clean) -> str:
    """Predict tags from  an input preprocessed data"""
    X_vect = w2v_vect_data(st.session_state.vectorizer, [input_clean.split(" ")])
    lr_preds = lr_predict_tags(st.session_state.classifier, X_vect)
    predictions = str.join("  ", lr_preds)

    # log infos
    logging.info(f"X shape: {X_vect.shape}")
    logging.info(f"Vectors:\n{X_vect[0]}")
    logging.info(f"Predictions: {predictions}")

    return predictions


def click_button():
    """Actions to perform when button clicked"""
    user_input = st.session_state.title_input + "\n" + st.session_state.body_input

    # check user input length
    if not check_length(user_input):
        logging.warning(f"⚠️ Input length is too short")
        st.session_state.predicted_tags = None
        st.session_state.message = "⚠️ Input length is too short"
    else:
        # preprocess input
        input_clean = preprocess_doc(
            user_input,
            st.session_state.keep_set,
            st.session_state.exclude_set
        )

        # check preprocessed input length before predict
        if not check_length(input_clean):
            logging.warning(f"⚠️ Length is too short after preprocessing: check input")
            st.session_state.predicted_tags = None
            st.session_state.message = "⚠️ Length is too short after preprocessing: check input"
        else:
            # predict tags
            predicted_tags = predict_tags(user_input)
            st.session_state.predicted_tags = predicted_tags

        # log infos
        logging.info(f"User input: {user_input}")
        logging.info(f"\nClean input: {input_clean}")
        logging.info(f"\nPredicted tags: {st.session_state.predicted_tags}")

    return st.session_state.predicted_tags


# GUI
st.set_page_config(
    page_title="Get tags from where you once asked for",
    page_icon="favicon.ico",
    layout="centered",
)
st.write("# Tags prediction")
st.write("Predict 5 tags from a StackOverflow-like question title and / or body) fields:")

# user input
st.text_input("Title", placeholder=TITLE_PLACEHOLDER, key="title", on_change=update_title)
st.text_area("Body", placeholder=BODY_PLACEHOLDER, height=160, key="body", on_change=update_body)

# predictions
st.button('⬇️  Predict tags  ⬇️', type='primary', use_container_width=True, on_click=click_button)

if st.session_state.predicted_tags is not None:
    st.write("#### :blue[{}]".format(st.session_state.predicted_tags))    
else:
    st.write("#### :red[{}]".format(st.session_state.message))

st.divider()

st.write("## ℹ️ TIPS")
st.markdown("""- preprocessing discards many **frequent and usual words** plus **HTML tags** and **code snippets** from user sentences and may result to a too small final input.  
    An error message can thus be displayed.""")
st.write("- Also note that the **model is trained for english language** input and may result in weird predictions in other cases.")

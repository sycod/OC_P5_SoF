"""Streamlit simple app to predict tags from a StackOverflow-like question."""

import os
import dill as pickle
import logging
import streamlit as st

# ML
from gensim.models import Word2Vec
import nltk

# home made functions
from src.scrap_and_clean import preprocess_doc
from src.models import w2v_vect_data
from src.models import lr_predict_tags


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
    st.session_state.title_input = TITLE_PLACEHOLDER
if "body_input" not in st.session_state:
    st.session_state.body_input = BODY_PLACEHOLDER
if "predicted_tags" not in st.session_state:
    st.session_state.predicted_tags = TAGS_PLACEHOLDER

# update session state on inputs
def update_title():
    st.session_state.title_input = st.session_state.title
def update_body():
    st.session_state.body_input = st.session_state.body





# def check_doc(input_doc) -> :
# 🚧 MAKE FUNCTION TO CHECK INPUT FIRST (data must be at least 2 words long, not punctuation:
#     # check
#     else:
#         check = True
#     return check
    


# def preprocess_doc(document, keep_set, exclude_set) -> str:
#     🚧 packages used -> re, nltk
#     🚧 regrouper fonctions en une seule
#     🚧 include keep_set and exclude_set in function
#     doc_clean = clean_string(document)
#     doc_tokens = tokenize_str(doc_clean, keep_set, exclude_set)
#     doc_lemmed = lemmatize_tokens(doc_tokens, keep_set, exclude_set)
#     doc_tk_clean = clean_tokens(doc_lemmed, keep_set, exclude_set)
#     doc_preprocessed = " ".join(doc_tk_clean)

#     return doc_preprocessed


def predict_tags(input_doc) -> str:
    """🚧 """
    input_clean = preprocess_doc(
        input_doc,
        st.session_state.keep_set,
        st.session_state.exclude_set
    )

    # 🚧 supprimer les outputs : les supprimer des fonctions
    logging.info(f"User input: {input_doc}")
    logging.info(f"\nClean input: {input_clean}")

    X_vect = w2v_vect_data(st.session_state.vectorizer, [input_clean.split(" ")])
    logging.info(f"X shape: {X_vect.shape}")
    logging.info(f"Vectors:\n{X_vect[0]}")

    lr_preds = lr_predict_tags(st.session_state.classifier, X_vect)
    predictions = str.join(" ", lr_preds)
    logging.info(f"Predictions: {predictions}")

    return predictions


def click_button():
    """🚧 """
    user_input = st.session_state.title_input + "\n" + st.session_state.body_input

    # check user input
    # checked_doc = check_doc(input_doc)
    # if checked_doc == True:
    #     logging.info("✅ user input checked")
    #     pass
    # else:
    #     logging.warning(f"⚠️ error {checked_doc}")
    #     return checked_doc
    # predict tags
    st.session_state.predicted_tags = predict_tags(user_input)










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
st.button('⬇️ Predict tags ⬇️', type='primary', use_container_width=True, on_click=click_button)
st.write("### :blue[{}]".format(st.session_state.predicted_tags))

st.write("🚧ℹ️ TIP: It includes a note about valid characters")
# preciser "too many frequent words" ou "balises HTML supprimées" ou "modèle entraîné sur de l'anglais"...
"""Streamlit simple app to predict tags from a StackOverflow-like question."""

import os
import dill as pickle
import logging
import streamlit as st
from gensim.models import Word2Vec
import nltk

# home made
from src.api_utils import check_length
from src.api_utils import preprocess_doc
from src.api_utils import predict_tags


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
    logging.info(f"⚙️  Loading keep set...")
    if os.path.exists(KEEP_SET_URI):
        with open(KEEP_SET_URI, "rb") as f:
            keep_set = pickle.load(f)
        st.session_state.keep_set = keep_set
        logging.info(f"✅ Keep set loaded")
    else:
        logging.warning(f"⚠️ No keep set found ⚠️")
# load exclude set (for preprocessing)
if "exclude_set" not in st.session_state:
    logging.info(f"⚙️  Loading exclude set...")
    if os.path.exists(EXCLUDE_SET_URI):
        with open(EXCLUDE_SET_URI, "rb") as f:
            exclude_set = pickle.load(f)
        st.session_state.exclude_set = exclude_set
        logging.info(f"✅ Exclude set loaded")
    else:
        logging.warning(f"⚠️ No exclude set found ⚠️")
# load vectorizer
if "vectorizer" not in st.session_state:
    logging.info(f"⚙️  Loading vectorizer...")
    if os.path.exists(VECTORIZER_URI):
        vectorizer = Word2Vec.load(VECTORIZER_URI)
        st.session_state.vectorizer = vectorizer
        logging.info(f"✅ Vectorizer loaded")
    else:
        logging.warning(f"⚠️ No vectorizer found ⚠️")
# load classifier
if "classifier" not in st.session_state:
    logging.info(f"⚙️  Loading classifier...")
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


# main function, triggered with button
def click_button():
    """Actions to perform when button clicked"""
    user_input = st.session_state.title_input + "\n" + st.session_state.body_input
    logging.info(f"\nUser input: {user_input}")

    # check user input length
    if not check_length(user_input):
        logging.warning(f"⚠️  Input length is too short")
        st.session_state.predicted_tags = None
        st.session_state.message = "⚠️  Input length is too short"
    else:
        # preprocess input
        input_clean = preprocess_doc(
            user_input, st.session_state.keep_set, st.session_state.exclude_set
        )
        logging.info(f"\nClean input: {input_clean}")

        # check preprocessed input length before predict
        if not check_length(input_clean):
            logging.warning(f"⚠️  Length is too short after preprocessing: check input")
            st.session_state.predicted_tags = None
            st.session_state.message = (
                "⚠️  Length is too short after preprocessing: check input"
            )
        else:
            # predict tags
            predicted_tags = predict_tags(
                input_clean, st.session_state.vectorizer, st.session_state.classifier
            )
            st.session_state.predicted_tags = predicted_tags

        # log infos
        logging.info(f"\nPredicted tags: {st.session_state.predicted_tags}")

    return st.session_state.predicted_tags


# GUI
st.set_page_config(
    page_title="Get tags (from where you once asked for)",
    page_icon="favicon.ico",
    layout="centered",
)
st.write("# Tags prediction")
st.write(
    "Predict 5 tags from a StackOverflow-like question title and / or body) fields:"
)

# user input
st.text_input(
    "Title", placeholder=TITLE_PLACEHOLDER, key="title", on_change=update_title
)
st.text_area(
    "Body", placeholder=BODY_PLACEHOLDER, height=160, key="body", on_change=update_body
)

# predictions
st.button(
    "⬇️  Predict tags  ⬇️",
    type="primary",
    use_container_width=True,
    on_click=click_button,
)
# display message if no prediction (e.g. input is too short)
if st.session_state.predicted_tags is not None:
    st.write("#### :blue[{}]".format(st.session_state.predicted_tags))
else:
    st.write("#### :red[{}]".format(st.session_state.message))

# info and tips
st.divider()
st.write("## ℹ️ TIPS")
st.markdown(
    """- preprocessing discards many **frequent and usual words** plus **HTML tags** and **code snippets** from user sentences and may result to a too small final input.  
    An error message can thus be displayed."""
)
st.write(
    "- Also note that the **model is trained for english language** input and may result in weird predictions in other cases."
)
st.write(
    "- If model can't find any of the input words in trained data, it will display a no-suggestion message"
)

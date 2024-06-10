"""Streamlit simple app to predict 5 tags from a StackOverflow-like question."""

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
TITLE_PLACEHOLDER = "e.g. pandas merge with Python >3.5"
BODY_PLACEHOLDER = """How can I perform a (INNER| (LEFT|RIGHT|FULL) OUTER) JOIN with pandas?
How do I add NaNs for missing rows after a merge? How do I get rid of NaNs after merging?
Can I merge on the index? How do I merge multiple DataFrames?
I've seen these recurring questions asking about various facets of the pandas merge functionality, the aim here is to collate some of the more important points for posterity.
"""


# CHECK & SETUP ML TOOLS
# load vectorizer
logging.info(f"⚙️ Loading vectorizer...")
if os.path.exists(VECTORIZER_URI):
    vectorizer = Word2Vec.load(VECTORIZER_URI)
    logging.info(f"✅ Vectorizer loaded")
else:
    logging.warning(f"⚠️ No vectorizer found ⚠️")
# load classifier
logging.info(f"⚙️ Loading classifier...")
if os.path.exists(CLASSIFIER_URI):
    with open(CLASSIFIER_URI, "rb") as f:
        classifier = pickle.load(f)
    logging.info(f"✅ Classifier loaded")
else:
    logging.warning(f"⚠️ No classifier found ⚠️")
# load keep set (for preprocessing)
logging.info(f"⚙️ Loading keep set...")
if os.path.exists(KEEP_SET_URI):
    with open(KEEP_SET_URI, "rb") as f:
        keep_set = pickle.load(f)
    logging.info(f"✅ Keep set loaded")
else:
    logging.warning(f"⚠️ No keep set found ⚠️")
# load keep set (for preprocessing)
logging.info(f"⚙️ Loading exclude set...")
if os.path.exists(EXCLUDE_SET_URI):
    with open(EXCLUDE_SET_URI, "rb") as f:
        exclude_set = pickle.load(f)
    logging.info(f"✅ Exclude set loaded")
else:
    logging.warning(f"⚠️ No exclude set found ⚠️")


# SESSION STATE INITIALIZATION
# placeholders (if not in session state)
if "title_input" not in st.session_state:
    st.session_state.title_input = TITLE_PLACEHOLDER
if "body_input" not in st.session_state:
    st.session_state.body_input = BODY_PLACEHOLDER











# 🚧 write note about valid characters
# 🚧 BODY = st.text_area
usr_input = usr_input_title + "\n" + usr_input_body

# 🚧 MAKE FUNCTION TO CHECK INPUT FIRST (data must be at least 2 words long, not punctuation:
# preciser "too many frequent words" ou "balises HTML supprimées" ou "modèle entraîné sur de l'anglais"...

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

input_clean = preprocess_doc(usr_input, keep_set, exclude_set)

# 🚧 supprimer les outputs : les supprimer des fonctions
print("user input:", usr_input)
print("\nclean input:", input_clean)

X_vect = w2v_vect_data(vectorizer, [input_clean.split(" ")])
print(X_vect.shape)
print(X_vect[0])

predicted_probas = classifier.predict_proba(X_vect)
lr_preds = lr_predict_tags(classifier, X_vect)
predictions = str.join(" ", lr_preds)
print("Predictions:", predictions)










# CALLBACKS
def k_from_slider():
    st.session_state.k_input = st.session_state.k_slider
def k_from_input():
    st.session_state.k_slider = st.session_state.k_input



# GUI
st.set_page_config(
    page_title="Get tags",
    page_icon="favicon.ico",
    layout="wide",
)
st.write("# 💰 Picsou calcule ses intérêts")
st.write(
    f"Intérêts annuels pour les {DURATION} prochaines années (brut et net, taxes incluses) et magie des intérêts composés."
)

col1, col2, col3 = st.columns(3)

# Capital setup
with col1:
    st.write("## Capital initial")
    st.write("### :orange[{:_} €]".format(st.session_state.k_slider).replace("_", " "))
    k_slider = st.slider(
        "",
        min_value=K["min"],
        max_value=K["max"],
        step=K["step"],
        key="k_slider",
        on_change=k_from_slider,
    )
    k_input = st.number_input(
        "",
        min_value=K["min"],
        max_value=K["max"],
        step=K["step"],
        key="k_input",
        on_change=k_from_input,
    )

# Annual rate setup
with col2:
    st.write("## Rentabilité annuelle")
    st.write(f"### :orange[{st.session_state.rate_slider :.0%}]")
    rate_slider = st.slider(
        "",
        min_value=RATE["min"],
        max_value=RATE["max"],
        step=RATE["step"],
        key="rate_slider",
        on_change=rate_from_slider,
    )
    rate_input = st.number_input(
        "",
        min_value=RATE["min"],
        max_value=RATE["max"],
        step=RATE["step"],
        key="rate_input",
        on_change=rate_from_input,
    )

# Annual savings setup
with col3:
    st.write("## Épargne annuelle")
    st.write(
        "### :orange[{:_} €]".format(st.session_state.ann_sav_slider).replace("_", " ")
    )
    ann_sav_slider = st.slider(
        "",
        min_value=ANN_SAVINGS["min"],
        max_value=ANN_SAVINGS["max"],
        step=ANN_SAVINGS["step"],
        key="ann_sav_slider",
        on_change=ann_sav_from_slider,
    )
    ann_sav_input = st.number_input(
        "",
        min_value=ANN_SAVINGS["min"],
        max_value=ANN_SAVINGS["max"],
        step=ANN_SAVINGS["step"],
        key="ann_sav_input",
        on_change=ann_sav_from_input,
    )


# GENERATE DATAFRAME
@st.cache_data(experimental_allow_widgets=True)
def create_df(capital, rate, ann_savings):
    """Generate dataframe from app inputs"""
    df_raw = gen_df(capital, rate, ann_savings)
    data = st.data_editor(
        df_raw,
        use_container_width=True,
        hide_index=True,
    )

    return data


# DISPLAY DATA
with st.columns([0.15, 0.7, 0.15])[1]:
    tab1, tab2, tab3 = st.tabs(["Tableau", "Capital", "Intérêts"])

    # Dataframe
    with tab1:
        df = create_df(
            st.session_state.k_slider,
            st.session_state.rate_slider,
            st.session_state.ann_sav_slider,
        )

    # Plot: capital
    with tab2:
        st.line_chart(df, x="annee", y=["epargne", "capital"])

    # Plot: interests
    with tab3:
        st.line_chart(df, x="annee", y=["epargne", "brut", "net", "mensuel_net"])
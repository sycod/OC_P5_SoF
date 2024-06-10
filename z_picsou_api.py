"""Streamlit simple app, calculating interests for many years."""

import yaml
import streamlit as st

from gen_df import gen_df


# CONFIG
# Load config file
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
# Setup
K = config["k"]
RATE = config["rate"]
ANN_SAVINGS = config["ann_savings"]
DURATION = config["duration_yrs"]
TAX = config["tax"]

# SESSION STATE INITIALIZATION
# Capital
if "k_slider" not in st.session_state:
    st.session_state.k_slider = K["init"]
if "k_input" not in st.session_state:
    st.session_state.k_input = K["init"]
# Rate
if "rate_slider" not in st.session_state:
    st.session_state.rate_slider = RATE["init"]
if "rate_input" not in st.session_state:
    st.session_state.rate_input = RATE["init"]
# Annual savings
if "ann_sav_slider" not in st.session_state:
    st.session_state.ann_sav_slider = ANN_SAVINGS["init"]
if "ann_sav_input" not in st.session_state:
    st.session_state.ann_sav_input = ANN_SAVINGS["init"]


# CALLBACKS
def k_from_slider():
    st.session_state.k_input = st.session_state.k_slider


def k_from_input():
    st.session_state.k_slider = st.session_state.k_input


def rate_from_slider():
    st.session_state.rate_input = st.session_state.rate_slider


def rate_from_input():
    st.session_state.rate_slider = st.session_state.rate_input


def ann_sav_from_slider():
    st.session_state.ann_sav_input = st.session_state.ann_sav_slider


def ann_sav_from_input():
    st.session_state.ann_sav_slider = st.session_state.ann_sav_input


# GUI
st.set_page_config(
    page_title="Picsou",
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
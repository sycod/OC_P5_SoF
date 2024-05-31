"""🚧 à diviser en plusieurs pour la prod"""
"""🚧 intégrer fonctions persos appelées"""


# OS & env
import os
import logging
import time

# DS
import numpy as np
import pandas as pd
import dill as pickle

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from gensim.models import Word2Vec

# home made functions from the src folder
from src.scrap_and_clean import init_data
from src.models import eval_lda_n_topics
from src.models import get_topics
from src.models import lr_predict_tags
from src.models import score_plot_model
from src.models import select_split_data

# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# *****************************************************************************
#                               🚧 PRÉ-TRAITEMENT
# *****************************************************************************

# get preprocessed data
df_pp = init_data()

# select from dates and split data
X_train, X_test, y_train, y_test = select_split_data(
    df_pp,
    random_state=42,
    test_size=1000,
    start_date="2019-05-01",
    end_date=None,
)

#  tokenized data
X_train_tok = X_train.str.split(" ")
X_test_tok = X_test.str.split(" ")

print(f"{X_train_tok.shape = }, {X_test_tok.shape = }")


# *****************************************************************************
#                           🚧❓ ÉVALUATION : pour MLFlow
#                           🚧❓ besoin ?
#                           🚧 insérer après màj fct ds NB
# *****************************************************************************



# *****************************************************************************
#                           🚧 INFÉRENCE
# *****************************************************************************

# def infer_model(model, input) -> list:
    # => check user input in streamlit + here too (special characters, HTML, etc. preprocessing)
    # load saved model
    # ? mlflow track scores

    # return pred

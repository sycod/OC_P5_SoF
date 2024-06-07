# OS & env
import os
import yaml
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
from gensim.models import Word2Vec
import tensorflow_hub as hub
import nltk
import mlflow

# home made functions from the src folder
from src.scrap_and_clean import init_data
from src.scrap_and_clean import preprocess_doc
from src.models import eval_lda_n_topics
from src.models import get_topics
from src.models import lr_predict_tags
from src.models import score_plot_model
from src.models import select_split_data
from src.models import evaluate_model
from src.models import w2v_vect_data
from src.models import bert_w_emb
from src.models import eval_stability
from src.models import plot_stability


# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# *****************************************************************************
#                                   DATA
# *****************************************************************************
logging.info(f"⚙️ Setting up data...")

# DATES
end_dates = [
    "2023-03-01",
    "2023-04-01",
    "2023-05-01",
    "2023-06-01",
    "2023-07-01",
    "2023-08-01",
    "2023-09-01",
    "2023-10-01",
    "2023-11-01",
    "2023-12-01",
    "2024-01-01",
    "2024-02-01",
    "2024-03-01",
]

monthes = [d[:-3] for d in end_dates[:-1]]

# TRAIN DATA
if (os.path.exists("data/mlfl_X_train.pkl")) and (
    os.path.exists("data/mlfl_y_train.pkl")
):
    with open("data/mlfl_X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("data/mlfl_y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    logging.info(f"✅ X_train, y_train data loaded")
else:
    df_pp = init_data()
    # set start / end dates
    start_date = "2019-03-01"
    end_date = "2023-03-01"
    # ceil / floor data from date
    df_train = df_pp.loc[
        (df_pp["date"] >= start_date) & (df_pp["date"] < end_date), ["doc_bow", "tags"]
    ]
    X_train = df_train["doc_bow"].str.split(" ")
    y_train = df_train["tags"]
    with open("data/mlfl_X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open("data/mlfl_y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
logging.info(f"\tℹ️ X_train shape {X_train.shape}, y_train shape {y_train.shape}")

# TEST DATA
with open("data/data_test_list.pkl", "rb") as f:
    data_test_list = pickle.load(f)
X_test_list = data_test_list[0]
y_test_list = data_test_list[1]
logging.info("✅ Test data loaded")


# *****************************************************************************
#                               ENTRAÎNEMENT
# *****************************************************************************



# START MLFLOW RECORD
logging.info(f"⚙️ Logging inputs and parameters...")
mlflow.start_run(run_name="train_w2v_cbow")
# convert to dataframes and MLFlow data objects for logging
mlflow.log_input(
    mlflow.data.from_pandas(X_train.to_frame(), source="data/mlfl_X_train.pkl"),
    context="training",
)
mlflow.log_input(
    mlflow.data.from_pandas(y_train.to_frame(), source="data/mlfl_y_train.pkl"),
    context="training",
)


# PARAMS
w2v_min_count = 100
w2v_vector_size = 50
w2v_sg = 0  # CBOW
w2v_window = 7
w2v_epochs = 10
lr_multi_class = "ovr"

mlflow.log_params(
    {
        "w2v_min_count": w2v_min_count,
        "w2v_vector_size": w2v_vector_size,
        "w2v_sg": w2v_sg,
        "w2v_window": w2v_window,
        "w2v_epochs": w2v_epochs,
        "lr_multi_class": lr_multi_class,
    }
)


# TRAIN
# vectorizer
logging.info(f"⚙️ Training vectorizer...")
w2v_vectorizer = Word2Vec(
    X_train,
    min_count=w2v_min_count,
    vector_size=w2v_vector_size,
    sg=w2v_sg,
    window=w2v_window,
    epochs=w2v_epochs,
)
X_train_w2v = w2v_vect_data(w2v_vectorizer, X_train)
# classifier
logging.info(f"⚙️ Training classifier...")
logreg = LogisticRegression(multi_class=lr_multi_class)
logreg.fit(X_train_w2v, y_train)
logging.info(f"✅ Training complete")

mlflow.sklearn.log_model(logreg, "w2v_cbow_model")

# register model
model_uri = f"runs:/{mlflow.active_run().info.run_id}/w2v_cbow"
mlflow.register_model(model_uri, "w2v_cbow")

mlflow.end_run()


# *****************************************************************************
#                               ENTRAÎNEMENT
# *****************************************************************************
mlflow.start_run()
logging.info(f"⚙️ Logging inputs and parameters...")
mlflow.log_artifact("data/data_test_list.pkl")

X_test_list_w2v = []
for i in X_test_list:
    X_test_w2v = w2v_vect_data(w2v_vectorizer, i)
    X_test_list_w2v.append(X_test_w2v)

stability_results = eval_stability(logreg, X_test_list_w2v, y_test_list)
logging.info(f"⚙️ Logging metrics...")
mlflow.log_metric("duration", stability_results["duration"])
for i in range(len(stability_results["tag_cover_scores"])):
    mlflow.log_metric("tag_cover", stability_results["tag_cover_scores"][i])
    mlflow.log_metric("jaccard", stability_results["jaccard_scores"][i])

logging.info(f"⚙️ Plotting stability...")
fig1 = plot_stability(
    monthes,
    stability_results["tag_cover_scores"],
    stability_results["jaccard_scores"],
    X_title="month",
    y_1_title="Tags cover score",
    y_2_title="Jaccard score",
)
mlflow.log_figure(fig1, "stability.png")
mlflow.end_run()


if __name__ == "__main__":
    help()

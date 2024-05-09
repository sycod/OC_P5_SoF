# OS & env
import os
import logging
import time

# DS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill as pickle

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import gensim
from gensim.models import CoherenceModel

import pyLDAvis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from Levenshtein import ratio

# home made functions from the src folder
from src.scrap_and_clean import init_data
from src.models import results_from_vec_matrix
from src.models import get_5_tags_from_matrix
from src.models import score_reduce
from src.models import plot_model
from src.models import vect_data
from src.models import eval_lda_n_topics
from src.models import get_topics
from src.models import topic_weights_df
from src.models import topic_predict


def eval_lda_n_topics(n_topics_list, corpus, documents, dictionary, plot=True) -> dict:
    """Evaluate LDA model coherence score for multiple number of topics (higher is better)"""
    start_time = time.time()
    coherence_scores = []

    for n_topic in n_topics_list:
        start_time_loop = time.time()
        print(f"Evaluating n={n_topic}...")
        
        # compute models
        lda = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=n_topic)
        coherence_model = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')

        # get score
        cs = coherence_model.get_coherence()
        print(f"\tCoherence score: {cs}")
        coherence_scores.append(cs)

        end_time_loop = time.time()
        elapsed_time_loop = end_time_loop - start_time_loop
        print(f"\tDurée : {elapsed_time_loop} secondes")

    if plot:
        x_ = n_topics_list
        y_ = coherence_scores

        fig, ax = plt.subplots()
        ax.set_ylim(min(y_) -1, max(y_) +1)

        ax.bar(x_, y_, width=8)
        ax.bar_label(ax.containers[0], label_type='edge')

        ax.plot(x_, y_, marker='o', color='red')

        # Adding labels and title
        plt.xlabel('Number of topics')
        plt.ylabel('Coherence score')
        plt.title('Number of topics: coherence scores')
        plt.savefig('outputs/LDA_coherence_Scores.png', dpi=150)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\tTemps d'exécution total : {elapsed_time} secondes")

    return dict(zip(n_topics_list, coherence_scores))
    


if __name__ == "__main__":
    # logging configuration (see all outputs, even DEBUG or INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # init data
    df_pp = init_data()
    df = df_pp[["doc_bow", "tags"]]

    # X, y, train, test split
    random_state = 42
    test_size = 1000
    X_train, X_test, y_train, y_test = train_test_split(
        df["doc_bow"], df["tags"], test_size=test_size, random_state=random_state
    )

    # vectorize
    cv = CountVectorizer(token_pattern=r"\S+", dtype=np.uint16, min_df=10)
    cv_data = cv.fit_transform(X_train)
    cv_names = cv.get_feature_names_out()

    # create dictionary and corpus
    dictionary = gensim.corpora.Dictionary([d.split() for d in X_train])
    corpus = [dictionary.doc2bow(doc.split()) for doc in X_train]

    # create LDA models
    n_topics_list = [8, 10, 12, 15, 20, 30, 40, 50]
    
    coherence_scores = eval_lda_n_topics(n_topics_list, corpus, X_train, dictionary, plot=True)

    print(f"{coherence_scores = }")

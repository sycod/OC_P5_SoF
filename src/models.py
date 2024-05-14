"""Functions used by models"""

import logging
import time
import numpy as np
import pandas as pd
from Levenshtein import ratio
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from matplotlib import colors


def eval_lda_n_topics(random_state, data, n_list=[10, 20, 30, 40, 50, 100], plot=True) -> dict:
    """Evaluate LDA model perplexity for multiple number of topics (lower is better)"""
    perplexities = []

    for i in n_list:
        lda = LatentDirichletAllocation(
            n_components=i,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=random_state,
        )
        print(f"Evaluating n={i}...")
        lda.fit(data)
        p = lda.perplexity(data)
        print(f"\t{p}")
        
        perplexities.append(p)

    if plot:
        x_ = n_list
        y_ = perplexities

        fig, ax = plt.subplots()
        ax.set_ylim(min(y_) * 0.9, max(y_) * 1.1)

        ax.bar(x_, y_, width=8)
        ax.bar_label(ax.containers[0], label_type='edge')

        ax.plot(x_, y_, marker='o', color='red')

        # Adding labels and title
        plt.xlabel('Number of topics')
        plt.ylabel('Perplexity')
        plt.title('Number of topics: perplexities plot')
        plt.show()
    
    return dict(zip(n_list, perplexities))


def score_terms(pred_words, target_words, cutoff=0.7) -> float:
    """Return a score of terms similarity between 2 lists of strings"""
    score = 0
    for p_w in pred_words:
        score += max(ratio(t, p_w, score_cutoff=cutoff) for t in target_words)
    score = np.round(score / len(target_words), 3)

    return score
    

def results_from_vec_matrix(words_list, X, n_max=5) -> dict:
    """Predict a maximum of n_max results from a vectorized transformed sparse matrix X."""
    d = X.data
    i = X.indices

    # get highest weights
    weights = np.sort(d)[::-1][: min(len(i), n_max)].tolist()

    # get corresponding indices
    pred_indices = i[np.argsort(-d)][: min(len(i), n_max)]

    # get corresponding words
    preds = [words_list[x] for x in pred_indices]

    return list(zip(preds, weights, pred_indices))


def get_5_tags_from_matrix(words_list, X, n_max=5) -> list:
    """Predict a maximum of n_max words from a vectorized transformed sparse matrix X."""
    # get corresponding indices
    pred_indices = X.indices[np.argsort(-X.data)][: min(len(X.indices), n_max)]
    # get corresponding words
    preds = [words_list[x] for x in pred_indices]

    return preds


def get_5_tags_from_array(words_list, X, n_max=5) -> list:
    """Predict a maximum of n_max words from an array."""
    preds = words_list[np.argsort(X)[-min(len(X), n_max):][::-1]].tolist()
    
    return preds


def score_reduce(words_list, X, y, model=None, model_type=None) -> tuple:
    """Returns model results from words list, features and targets"""
    start_time = time.time()

    # get scores from results
    if model_type == "topic":
        # weights matricial product: topics * words
        preds_weights = X.dot(model.components_)
        # get results
        X_results = [get_5_tags_from_array(words_list, xi) for xi in preds_weights]
    else:
        X_results = [get_5_tags_from_matrix(words_list, xi) for xi in X]

    scores = [score_terms(p_w, y[y.index[i]].split(" ")) for i, p_w in enumerate(X_results)]
    model_score = np.round(np.mean(scores), 3)

    # reduce dimensions
    tsne = TSNE(n_components=2, perplexity=50, n_iter=2000, init='random', learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(X)

    duration = np.round(time.time() - start_time,0)

    print("Score: ", model_score, "- Duration: ", duration)
    
    return model_score, X_results, scores, X_tsne


def get_topics(model, feature_names, n_top_words) -> list:
    """Display the topics of a LDA model."""
    topics = []
    for topic in model.components_:
        topics.append(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
            )
        )

    return topics


def plot_model(model_score, scores, X_tsne) :
    """Scores & TSNE plotting"""
    fig, axs = plt.subplots(1, 2, figsize=(9,4), tight_layout=True)

    # T-SNE DATA
    scatter = axs[0].scatter(X_tsne[:,0],X_tsne[:,1], c=scores, cmap='viridis')    
    axs[0].set_title(f"T-SNE representation")

    # SCORES
    N, bins, patches = axs[1].hist(scores, bins=10)
    # color by score (bin)
    fracs = bins / bins.max()
    norm = colors.Normalize(0, 1)
    # loop through objects and set color of each
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    axs[1].set_title(f"Score: {model_score}")
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Count')

    plt.show()


def plot_topic_model(model_score, scores, X_tsne, top_topics) :
    """Scores & TSNE plotting with color by topic"""
    fig, axs = plt.subplots(1, 2, figsize=(9,4), tight_layout=True)

    # T-SNE DATA
    scatter = axs[0].scatter(X_tsne[:,0],X_tsne[:,1], c=top_topics, cmap='viridis')    
    axs[0].set_title(f"T-SNE representation")

    # SCORES
    N, bins, patches = axs[1].hist(scores, bins=10)
    # color by score (bin)
    fracs = bins / bins.max()
    norm = colors.Normalize(0, 1)
    # loop through objects and set color of each
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    axs[1].set_title(f"Score: {model_score}")
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Count')

    plt.show()


# def vect_data(data, vec_type="cv") -> np.ndarray:
#     """Vectorizes data with CountVectorizer or TfidfVectorizer"""
#     if vec_type == "cv":
#         vectorizer = CountVectorizer(token_pattern=r"\S+", dtype=np.uint16, min_df=10)
#     elif vec_type == "tfidf":
#         vectorizer = TfidfVectorizer(token_pattern=r"\S+", min_df=10)
#     else:
#         raise ValueError("Unknown vectorizer type (vec_type)")

#     return vectorizer.fit_transform(data)


if __name__ == "__main__":
    print(f"\n👉 results_from_vec_matrix(values) -> str\n{results_from_vec_matrix.__doc__}")

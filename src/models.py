"""Functions used by models"""

import numpy as np


def results_from_vec_matrix(vectorizer, X, n_max) -> dict:
    """Predict a maximum of n_max results from a vectorizer and a transformed sparse matrix X."""
    d = X.data
    i = X.indices

    # get highest weights
    weights = np.sort(d)[::-1][: min(len(i), n_max)].tolist()

    # get corresponding indices
    pred_indices = [i[np.argsort(-d)][: min(len(i), n_max)]]

    # get corresponding words
    preds = [vectorizer.get_feature_names_out()[x] for x in pred_indices[0]]

    return dict(zip(preds, weights))


def get_lda_topics(model, feature_names, n_top_words) -> list:
    """Display the topics of a LDA model."""
    topics = []
    for topic in model.components_:
        topics.append(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
            )
        )

    return topics


if __name__ == "__main__":
    print(f"\n👉 results_from_vec_matrix(values) -> str\n{results_from_vec_matrix.__doc__}")

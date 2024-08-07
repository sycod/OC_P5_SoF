"""Functions used by models"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objects as go

from Levenshtein import ratio
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
import torch


def eval_lda_n_topics(
    random_state, data, n_list=[10, 20, 30, 40, 50, 100], plot=True, width=8
) -> dict:
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
        logging.info(f"Evaluating n={i}...")
        lda.fit(data)
        p = lda.perplexity(data)
        logging.info(f"\t{p}")

        perplexities.append(p)

    if plot:
        x_ = n_list
        y_ = perplexities

        fig, ax = plt.subplots()
        ax.set_ylim(min(y_) * 0.9, max(y_) * 1.1)

        ax.bar(x_, y_, width=width)
        ax.bar_label(ax.containers[0], label_type="edge")

        ax.plot(x_, y_, marker="o", color="red")

        # Adding labels and title
        plt.xlabel("Number of topics")
        plt.ylabel("Perplexity")
        plt.title("Number of topics: perplexities plot")
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
    preds = words_list[np.argsort(X)[-min(len(X), n_max) :][::-1]].tolist()

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

    scores = [
        score_terms(p_w, y[y.index[i]].split(" ")) for i, p_w in enumerate(X_results)
    ]
    model_score = np.round(np.mean(scores), 3)

    # reduce dimensions
    # TSNE random init used instead of PCA because sparse matrix can't use PCA
    tsne = TSNE(
        n_components=2, perplexity=50, max_iter=2000, init="random", random_state=42
    )
    X_tsne = tsne.fit_transform(X)

    duration = np.round(time.time() - start_time, 0)

    logging.info(f"Score: {model_score} - Duration: {duration}")

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


def plot_model(model_score, scores, X_tsne):
    """Scores & TSNE plotting"""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)

    # T-SNE DATA
    scatter = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=scores, cmap="viridis")
    axs[0].set_title("T-SNE representation")

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
    axs[1].set_xlabel("Score")
    axs[1].set_ylabel("Count")

    plt.show()


def plot_topic_model(model_score, scores, X_tsne, top_topics):
    """Scores & TSNE plotting with color by topic"""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)

    # T-SNE DATA
    scatter = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=top_topics, cmap="viridis")
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
    axs[1].set_xlabel("Score")
    axs[1].set_ylabel("Count")

    plt.show()


def score_plot_model(preds, X, y, plot=True, top_topics=None, time_it=True) -> tuple:
    """Returns model tags cover and Jaccard scores from predictions and targets, including plot"""
    start_time = time.time()

    # tags cover scores
    preds_list = [x.split(" ") for x in preds]
    tc_scores = [
        score_terms(p_w, y.to_list()[i].split(" ")) for i, p_w in enumerate(preds_list)
    ]
    tc_score = np.round(np.mean(tc_scores), 3)

    # jaccard scores
    j_scores = [score_jaccard(p_w, y.to_list()[i]) for i, p_w in enumerate(preds)]
    j_score = np.round(np.mean(j_scores), 3)

    if time_it:
        duration = np.round(time.time() - start_time, 0)
        print(
            f"Tag cover score: {tc_score} - Jaccard score: {j_score} - Duration: {duration}"
        )
    else:
        print(f"Tag cover score: {tc_score} - Jaccard score: {j_score}")

    if plot:
        # reduce dimensions
        tsne = TSNE(n_components=2, perplexity=50, max_iter=2000, init="pca")
        X_tsne = tsne.fit_transform(X)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
        color = j_scores if top_topics is None else top_topics

        # T-SNE
        scatter = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap="viridis")
        axs[0].set_title(f"T-SNE representation")

        # TAGS COVER SCORES
        N, bins, patches = axs[1].hist(tc_scores, bins=10)
        N, bins, patches = axs[1].hist(tc_scores, bins=10)
        # color by score (bin)
        fracs = bins / bins.max()
        norm = colors.Normalize(0, 1)
        # loop through objects and set color of each
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)
        axs[1].set_title(f"Tag cover score: {tc_score}")
        axs[1].set_xlabel("Tag cover score")
        axs[1].set_ylabel("Count")

        # JACCARD SCORES
        N, bins, patches = axs[2].hist(j_scores, bins=10)
        N, bins, patches = axs[2].hist(j_scores, bins=10)
        # color by score (bin)
        fracs = bins / bins.max()
        norm = colors.Normalize(0, 1)
        # loop through objects and set color of each
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)
        axs[2].set_title(f"Jaccard score: {j_score}")
        axs[2].set_xlabel("Jaccard score")
        axs[2].set_ylabel("Count")

        plt.show()

    return tc_score, j_score, tc_scores, j_scores


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


def score_jaccard(y_true, y_pred) -> float:
    """Return the Jaccard score between 2 strings"""

    y_true = y_true.split(" ")
    y_pred = y_pred.split(" ")
    pred_labels = []

    # since order is important, we need to keep the order of the true labels
    # correct labels are added in the order they appear in the true labels
    i = 0
    while len(pred_labels) < len(y_true):
        if y_true[i] in y_pred:
            pred_labels.append(y_true[i])
        else:
            pred_labels.append("0")
        i += 1

    j_score = jaccard_score(y_true, pred_labels, average="weighted")

    return j_score


def select_split_data(
    df, random_state=42, test_size=1000, start_date=None, end_date=None
) -> tuple:
    """Prepare splitted and eventually date-windowed data from a preprocessed dataframe"""
    # ceil / floor data from date
    if start_date:
        df = df.loc[(df["date"] >= start_date)]
    if end_date:
        df = df.loc[(df["date"] < end_date)]

    # select columns
    df = df[["doc_bow", "tags"]]

    # X, y, train, test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["doc_bow"], df["tags"], test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(data, vocab=None, n_topics=5, random_state=42, plot=True) -> tuple:
    """Train ML model from preprocessed and splitted data, packed in a tuple"""
    start_time = time.time()

    # unpack data
    X_train_vect, X_test_vect, y_train, y_test = data

    # topic modeling for plot output
    if vocab is not None:
        n_top_words = 10
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=random_state,
        )

        # LDA needs positive values
        X_train_pos = (
            X_train_vect + abs(X_train_vect.min())
            if X_train_vect.min() < 0
            else X_train_vect
        )
        X_test_pos = (
            X_test_vect + abs(X_test_vect.min())
            if X_test_vect.min() < 0
            else X_test_vect
        )

        # get topics
        lda.fit(X_train_pos)
        topics = get_topics(lda, vocab, n_top_words)
        X_test_lda = lda.transform(X_test_pos)
        top_topics_test = [xi.argsort()[:-2:-1][0] for xi in X_test_lda]
    else:
        top_topics_test = None

    # classify
    logging.info(f"Classifying...")
    logreg = LogisticRegression(multi_class="ovr")
    logreg.fit(X_train_vect, y_train)
    predicted_probas = logreg.predict_proba(X_test_vect)
    lr_preds = lr_predict_tags(logreg, X_test_vect)
    logging.info(f"✅ {len(lr_preds)} predictions done")

    # score
    logging.info(f"Scoring...")
    score_tc, score_j, scores_tc, scores_j = score_plot_model(
        lr_preds,
        predicted_probas,
        y_test,
        top_topics=top_topics_test,
        time_it=False,
        plot=plot,
    )

    # duration
    duration = np.round(time.time() - start_time, 0)
    logging.info(f"Total duration: {duration}")

    return score_tc, score_j, duration


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


def bert_w_emb(
    model, tokenizer, sentences, batch_size=10, max_length=32, max_pop=None
) -> torch.Tensor:
    """Process to BERT word embedding on a given sentences set"""
    if max_pop is not None:
        sentences = sentences[:max_pop]

    for step in range(len(sentences) // batch_size):
        # set start / stop steps
        start_step = batch_size * step
        if batch_size * (step + 1) >= len(sentences):
            stop_step = len(sentences)
        else:
            stop_step = batch_size * (step + 1)

        # tokenize sentences
        inputs = tokenizer(
            sentences[start_step:stop_step],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
        )
        # embed
        outputs = model(**inputs)
        word_embeddings = outputs.logits

        # save in tensor
        if step == 0:
            X_bert_pt = word_embeddings
        else:
            X_bert_pt = torch.cat((X_bert_pt, word_embeddings), dim=0)

        logging.info(
            f"Step {step} ok on data[{start_step}:{stop_step}] -> shape {X_bert_pt.shape}"
        )

    return X_bert_pt


def eval_stability(lr_model, X_test_list, y_test_list) -> dict:
    """Train W2V model from preprocessed train data and test samples list"""
    start_time = time.time()
    probas_list = []
    predictions_list = []
    metric_jaccard = []
    metric_tag_cover = []

    # loop over test samples
    for i, X_test in enumerate(X_test_list):

        # predict
        predicted_probas = lr_model.predict_proba(X_test)
        lr_preds = lr_predict_tags(lr_model, X_test)
        logging.info(f"⚙️ Step {i+1}: {len(lr_preds)} predictions")
        # store
        probas_list.append(predicted_probas)
        predictions_list.append(lr_preds)

        # score
        y = y_test_list[i]
        # tags cover
        preds_list = [x.split(" ") for x in lr_preds]
        tc_scores = [
            score_terms(p_w, y.to_list()[i].split(" "))
            for i, p_w in enumerate(preds_list)
        ]
        score_tc = np.round(np.mean(tc_scores), 3)
        # jaccard
        j_scores = [
            score_jaccard(p_w, y.to_list()[i]) for i, p_w in enumerate(lr_preds)
        ]
        score_j = np.round(np.mean(j_scores), 3)
        # store
        metric_tag_cover.append(score_tc)
        metric_jaccard.append(score_j)

        logging.info(f"\t➡️ tag cover {score_tc}, Jaccard {score_j}\n")

    # duration
    duration = np.round(time.time() - start_time, 0)
    logging.info(f"⏱️ Total duration: {duration}s")

    results = {
        "lr_predictions": predictions_list,
        "lr_probas": probas_list,
        "tag_cover_scores": metric_tag_cover,
        "jaccard_scores": metric_jaccard,
        "duration": duration,
    }

    return results


def plot_stability(
    X_data,
    y_1,
    y_2,
    X_title="X",
    y_1_title="y_1",
    y_2_title="y_2",
    width=600,
    height=400,
) -> go.Figure:
    """Plot stability results"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_data, y=y_1, mode="lines+markers", name=y_1_title))
    fig.add_trace(go.Scatter(x=X_data, y=y_2, mode="lines+markers", name=y_2_title))
    fig.update_layout(
        title=f"Stability: metrics by {X_title}",
        xaxis_title=X_title,
        yaxis_title="metrics",
        width=width,
        height=height,
        margin=dict(l=50, r=50, b=50, t=50, pad=0),
    )
    fig.show()

    return fig


if __name__ == "__main__":
    help()

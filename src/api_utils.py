"""Functions used by API"""

import logging
import re
import emoji
import numpy as np
import nltk


def check_length(input_doc, length=5) -> bool:
    """Check input length"""
    length_ok = True if len(input_doc.split(" ")) >= length else False

    return length_ok


def preprocess_doc(document, keep_set, exclude_set) -> str:
    """Preprocess document for a complete cleaning"""
    # CLEAN DOC
    # remove code tags
    document = re.sub(r"<code>.*?<\/code>", "", document, flags=re.S)
    # remove img tags
    document = re.sub(r"<img.*?>", "", document)
    # remove all html tags
    document = re.sub(r"<.*?>", "", document)
    # remove emojis
    document = emoji.replace_emoji(document, replace=" ")
    # remove newlines
    document = re.sub(r"\n", " ", document)
    # lowercase
    document = document.lower()
    # remove suspension points
    document = re.sub(r"\.\.\.", " ", document)
    # remove digits only tokens
    document = re.sub(r"\b(?<![0-9-])(\d+)(?![0-9-])\b", " ", document)
    # remove multiple spaces
    document = re.sub(r" +", " ", document)

    # TOKENIZE DOC
    # tokenize except excluded
    tokens = nltk.word_tokenize(document)
    # remove hashes from watch list
    i_offset = 0
    for i, t in enumerate(tokens):
        i -= i_offset
        if t == "#" and i > 0:
            left = tokens[: i - 1]
            joined = [tokens[i - 1] + t]
            right = tokens[i + 1 :]
            if joined[0] in keep_set:
                tokens = left + joined + right
                i_offset += 1
    # remove (< 3)-letter words apart from those appearing in keep_set
    tokens_rm_inf3 = [t for t in tokens if len(t) > 2 or t in keep_set]
    # remove tokens containing absolutely no letter
    tokens_rm_no_letter = list(
        filter(lambda s: any([c.isalpha() for c in s]), tokens_rm_inf3)
    )
    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_no_letter if t not in exclude_set]

    # LEMMATIZE TOKENS
    kilmister = nltk.wordnet.WordNetLemmatizer()
    lem_tok_list = []

    for token in tokens_cleaned:
        if token in keep_set:
            lem_tok_list.append(token)
        else:
            lem_tok = kilmister.lemmatize(token)
            if lem_tok not in exclude_set:
                lem_tok_list.append(lem_tok)

    # CLEAN TOKENS
    # clean " ' " in front of certain words
    clean_apo = []
    clean_apo += [t[1:] if t[0] == "'" else t for t in lem_tok_list]
    # clean " - " in front of certain words
    clean_dash = []
    clean_dash += [t[1:] if t[0] == "-" else t for t in clean_apo]
    # remove (< 3)-letter words apart from those belonging to keep_set
    tokens_rm_inf3 = [t for t in clean_dash if len(t) > 2 or t in keep_set]
    # remove remaining excluded words
    tokens_cleaned = [t for t in tokens_rm_inf3 if t not in exclude_set]

    doc_preprocessed = " ".join(tokens_cleaned)

    return doc_preprocessed


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


def lr_prediction(model, X, n_tags=5) -> list:
    """Use logistic regression probabilities to get at least n predicted tags"""
    ppbs = model.predict_proba(X)
    classes = model.classes_
    pred_tags = []

    for i, _ in enumerate(X):
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


def predict_tags(input_clean, vectorizer, classifier) -> str:
    """Predict tags from  an input preprocessed data"""
    X_vect = w2v_vect_data(vectorizer, [input_clean.split(" ")])
    lr_preds = lr_prediction(classifier, X_vect)
    predictions = str.join("  ", lr_preds)

    # log infos
    logging.info(f"X shape: {X_vect.shape}")
    logging.info(f"Vectors:\n{X_vect[0]}")
    logging.info(f"Predictions: {predictions}")

    return predictions


if __name__ == "__main__":
    help()

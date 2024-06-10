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

# 🚧 import this function in api.py
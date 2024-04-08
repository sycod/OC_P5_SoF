"""Utils for web scraping and data cleaning"""

import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import CountVectorizer


def make_autopct(values) -> str:
    """==> Obtained from StackOverflow <==
    Upgrades plt.pie(autopct=""), displaying percentages and values.
    
    Input: list of numeric values or Pandas.Series
    Output: string with percentage and value
    """

    def my_autopct(pct) -> str:
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    
    return my_autopct


def create_df(cols, len_df, int_dtype) -> pd.DataFrame:
    """Create a DataFrame filled with zeros from a list of columns, length and int type."""
    return pd.DataFrame(
        np.zeros((len_df, len(cols)), dtype=int_dtype),
        index=np.arange(len_df),
        columns=cols,
    )


def count_occurrences(x, df, column) -> int:
    """Count the number of occurrences of x in the column of a dataframe.
    ⚠️ Overflow: for memory optimization, returns a uint16 (0- 65535).
    """
    
    return df[column].apply(lambda l: np.uint16(l.count(x)))


def make_stat_df(words_list, name, verbose=True) -> pd.DataFrame:
    """Create a DataFrame with token count and frequency for a given list of words."""
    count_vectorizer = CountVectorizer(token_pattern=r"\S+", dtype=np.uint16)
    X_cv = count_vectorizer.fit_transform(words_list)
    
    # create DF
    stats_df = pd.DataFrame()
    stats_df["token"] = count_vectorizer.get_feature_names_out()

    # count total and get frequency
    stats_df[f"count_{name}"] = X_cv.sum(axis=0).A1.astype(np.uint16)
    tot = stats_df[f"count_{name}"].sum()
    stats_df[f"freq_{name}"] = np.float32(stats_df[f"count_{name}"] / tot)

    if verbose:
        logging.info(f"{name} DF shape: {stats_df.shape}")
        logging.info(f"Total tokens in {name}: {tot}")
    
    return stats_df


if __name__ == "__main__":
    print(f"\n👉 make_autopct(values) -> str\n{make_autopct.__doc__}")
    print(f"\n👉 create_df(cols, len_df, int_dtype) -> pd.DataFrame\n{create_df.__doc__}")
    print(f"\n👉 count_occurrences(x, df, column) -> int\n{count_occurrences.__doc__}")
    print(f"\n👉 make_stat_df(words_list, name, verbose=True) -> pd.DataFrame\n{make_stat_df.__doc__}")

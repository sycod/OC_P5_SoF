"""Utils for web scraping and data cleaning"""

# import

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

if __name__ == "__main__":
    print(f"\n👉 make_autopct(values) -> str\n{make_autopct.__doc__}")
    print(f"\n👉 create_df(cols, len_df, int_dtype) -> pd.DataFrame\n{create_df.__doc__}")

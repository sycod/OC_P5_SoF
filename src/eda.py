"""Utils for web scraping and data cleaning"""

# import

def make_autopct(values):
    """==> Obtained from StackOverflow <==
    Upgrades plt.pie(autopct=""), displaying percentages and values.
    
    Input: list of numeric values or Pandas.Series
    Output: string with percentage and value
    """

    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    
    return my_autopct



"""Generate dataframe from app inputs"""

import yaml
import pandas as pd


def gen_df(k, rate, ann_savings):
    """Generate dataframe from app inputs

    Args:
        k (int): Initial capital
        rate (float): Interest rate
        ann_savings (int): Annual savings
    """

    # CONFIG
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    DURATION = config["duration_yrs"]
    TAX = config["tax"]

    # DATAFRAME
    df = pd.DataFrame(
        columns=["annee", "epargne", "capital", "brut", "net", "mensuel_net"]
    )

    # YEAR
    df["annee"] = range(DURATION)
    df["annee"] = df["annee"] + 1

    # SAVINGS
    df["epargne"] = k + ((df["annee"] - 1) * ann_savings)

    # GROSS INTEREST & CAPITAL
    comp_k = [k]
    comp_i = [k * rate]

    for i in range(1, len(df.index)):
        new_k = comp_k[i - 1] + ann_savings + comp_i[i - 1]
        new_i = new_k * rate
        comp_k.append(new_k)
        comp_i.append(new_i)

    df["brut"] = comp_i
    df["brut"] = round(df["brut"], 2).astype(int)

    df["capital"] = comp_k
    df["capital"] = round(df["capital"], 0).astype(int)

    # NET INTEREST
    df["net"] = round(df["brut"] * (1 - TAX), 0).astype(int)

    # MONTHLY NET INTEREST
    df["mensuel_net"] = round(df["net"] / 12, 0).astype(int)

    return df


if __name__ == "__main__":
    help(gen_df)
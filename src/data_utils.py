import pandas as pd
import numpy as np
def build_multi_feature_set(ticker_files, window=10, corr_window=20, n_pca=3):
    price_series = {}
    for t, p in ticker_files.items():
        df = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        price_series[t] = df["Close"].rename(t)
    price_df = pd.concat(list(price_series.values()), axis=1)
    price_df.columns = list(price_series.keys())
    price_df = price_df.dropna().reset_index(drop=True)
    corr_df = price_df.rolling(window=corr_window, min_periods=1).corr().fillna(0)
    pca_df = price_df.rolling(window=window, min_periods=1).mean().fillna(0)
    return price_df, corr_df, pca_df

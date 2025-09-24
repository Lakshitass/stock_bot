import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def load_and_align(ticker_files, date_col="Date"):
    dfs = {}
    for t, p in ticker_files.items():
        df = pd.read_csv(p, parse_dates=[date_col]).sort_values(date_col).reset_index(drop=True)
        dfs[t] = df
    # intersection of dates
    idxs = [set(df[date_col].dt.normalize()) for df in dfs.values()]
    common_dates = sorted(set.intersection(*idxs))
    aligned = {}
    for t, df in dfs.items():
        df2 = df[df[date_col].dt.normalize().isin(common_dates)].copy()
        df2 = df2.sort_values(date_col).reset_index(drop=True)
        aligned[t] = df2
    price_df = pd.DataFrame({t: aligned[t]["Close"].values for t in aligned})
    dates = aligned[next(iter(aligned))][date_col].reset_index(drop=True)
    price_df.index = pd.to_datetime(dates.values)
    return aligned, price_df

def rolling_correlation_features(price_df, window=20, n_pca=3):
    returns = price_df.pct_change().fillna(0)
    corr_flat_list = []
    pca_factors = []
    n = price_df.shape[1]
    iu = np.triu_indices(n, k=1)
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        window_ret = returns.iloc[start:i+1]
        if len(window_ret) < 2:
            corr_flat_list.append(np.zeros(len(iu[0])))
            pca_factors.append(np.zeros(n_pca))
            continue
        mat = window_ret.corr().fillna(0).values
        corr_flat = mat[iu]
        corr_flat_list.append(corr_flat)
        try:
            pca = PCA(n_components=n_pca)
            pca.fit(window_ret.fillna(0).values)
            pca_factors.append(pca.transform(window_ret.values[-1].reshape(1, -1))[0])
        except Exception:
            pca_factors.append(np.zeros(n_pca))
    corr_cols = [f"corr_{i}_{j}" for i in range(n) for j in range(i+1, n)]
    corr_df = pd.DataFrame(corr_flat_list, index=price_df.index, columns=corr_cols)
    pca_cols = [f"pca_{k}" for k in range(n_pca)]
    pca_df = pd.DataFrame(pca_factors, index=price_df.index, columns=pca_cols)
    return corr_df, pca_df

def build_multi_feature_set(ticker_files, window=10, corr_window=20, n_pca=3):
    _, price_df = load_and_align(ticker_files)
    corr_df, pca_df = rolling_correlation_features(price_df, window=corr_window, n_pca=n_pca)
    return price_df, corr_df, pca_df

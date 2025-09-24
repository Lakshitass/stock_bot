import os
import glob
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def prepare_data(df, split_date="2019-01-01"):
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.fillna(method="ffill").fillna(method="bfill")

    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Vol10"] = df["LogReturn"].rolling(window=10).std()
    df["RSI14"] = compute_rsi(df["Close"], window=14)

    df = df.dropna().reset_index(drop=True)

    train = df[df["Date"] < split_date].reset_index(drop=True)
    test = df[df["Date"] >= split_date].reset_index(drop=True)
    return train, test

def process_all(data_dir="data", out_dir="data", split_date="2019-01-01", save_scaler=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("models/scalers").mkdir(parents=True, exist_ok=True)

    csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    summary = []
    for p in csvs:
        name = Path(p).stem
        print(f"Processing {name} ...")
        df = pd.read_csv(p, parse_dates=["Date"])
        train, test = prepare_data(df, split_date=split_date)

        # Fit scaler on TRAIN only (important!)
        feat_cols = ["Close", "LogReturn", "MA5", "MA10", "MA20", "Vol10", "RSI14"]
        scaler = StandardScaler()
        if len(train) > 0:
            scaler.fit(train[feat_cols].values)
        else:
            scaler.fit(np.zeros((1, len(feat_cols))))

        # Transform and save scaled versions (optional)
        train_scaled = train.copy()
        test_scaled = test.copy()
        if len(train_scaled) > 0:
            train_scaled[feat_cols] = scaler.transform(train[feat_cols].values)
        # For test, transform using train-fitted scaler; if test empty, skip
        if len(test_scaled) > 0:
            test_scaled[feat_cols] = scaler.transform(test_scaled[feat_cols].values)

        train_path = os.path.join(out_dir, f"{name}_train.csv")
        test_path = os.path.join(out_dir, f"{name}_test.csv")
        train_scaled.to_csv(train_path, index=False)
        test_scaled.to_csv(test_path, index=False)

        if save_scaler:
            scaler_path = os.path.join("models/scalers", f"{name}_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump({"scaler": scaler, "features": feat_cols}, f)

        summary.append({
            "ticker": name,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_path": train_path,
            "test_path": test_path
        })
        print(f" -> saved {train_path} ({len(train)} rows), {test_path} ({len(test)} rows)")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "data_summary.csv"), index=False)
    print("All done. Summary saved to", os.path.join(out_dir, "data_summary.csv"))
    return summary_df

if __name__ == "__main__":
    # run on repo data/
    process_all(data_dir="data", out_dir="data", split_date="2019-01-01")

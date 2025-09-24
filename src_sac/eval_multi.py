import argparse, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src_sac.multi_env import MultiStockTradingEnv
from src_sac.data_utils import build_multi_feature_set
from src_sac.metrics import evaluate_equity_curve

def evaluate(model_path, ticker_files, out_prefix="results/plot_model/multi_eval", window=10, corr_window=20, trade_threshold=0.01):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    price_df, corr_df, pca_df = build_multi_feature_set(ticker_files, window=window, corr_window=corr_window, n_pca=3)
    env = MultiStockTradingEnv(price_df=price_df, corr_df=corr_df, pca_df=pca_df, window=window)
    model = SAC.load(model_path)

    obs, _ = env.reset()
    nets = [env.net_worth]
    dates = [price_df.index[env.current_step]]
    allocations = []
    trades = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        ret = env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs, reward, done, info = ret

        nets.append(info["net_worth"])
        dates.append(price_df.index[env.current_step])
        allocations.append(info.get("allocation", []))
        if info.get("traded_value", 0.0) > trade_threshold * info.get("net_worth", 1.0):
            trades.append({"step": info["step"], "date": dates[-1], "traded_value": info["traded_value"], "allocation": info.get("allocation")})

    # DataFrame
    df = pd.DataFrame({"date": pd.to_datetime(dates), "net_worth": nets})
    df = df.set_index("date")
    metrics = evaluate_equity_curve(df["net_worth"])
    metrics["num_trades"] = len(trades)

    # Save
    out_metrics = out_prefix + "_metrics.csv"
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False)
    trades_csv = out_prefix + "_trades.csv"
    with open(trades_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "date", "traded_value", "allocation"])
        writer.writeheader()
        for r in trades:
            writer.writerow({"step": r["step"], "date": r["date"].strftime("%Y-%m-%d"), "traded_value": r["traded_value"], "allocation": str(r["allocation"])})

    # Plots
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["net_worth"], label="RL NetWorth")
    plt.title("RL Multi-stock Net Worth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix + "_equity.png")
    plt.close()

    print("Saved metrics:", out_metrics)
    print("Saved trades:", trades_csv)
    print("Saved equity plot:", out_prefix + "_equity.png")
    print("Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="results/plot_model/multi_eval")
    args = parser.parse_args()
    ticker_files = {
      "AAPL":"data/AAPL.csv","TSLA":"data/TSLA.csv","JPM":"data/JPM.csv","GS":"data/GS.csv","XOM":"data/XOM.csv"
    }
    evaluate(args.model, ticker_files, out_prefix=args.out)

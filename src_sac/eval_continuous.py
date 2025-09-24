import argparse
import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src_sac.env_continuous import TradingEnvContinuous

def compute_metrics(portfolio_series):
    # portfolio_series: pd.Series indexed by date
    rets = portfolio_series.pct_change().dropna()
    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1.0 if len(portfolio_series) > 1 else 0.0
    days = (portfolio_series.index[-1] - portfolio_series.index[0]).days if len(portfolio_series) > 1 else 1
    annual_return = (1.0 + total_return) ** (365.0 / max(days, 1)) - 1.0
    if rets.std() == 0:
        sharpe = np.nan
    else:
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown": float(max_drawdown),
    }

def evaluate(model_path, data_path, plot_prefix="results/plot_sac/sac_eval", trade_threshold=0.01):
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)
    os.makedirs("results/trades", exist_ok=True)

    # load environment and model
    env = TradingEnvContinuous(data_path)
    model = SAC.load(model_path)

    # reset env (Gymnasium style -> obs, info)
    obs, _ = env.reset()
    net_worths = [env.net_worth]
    prices = [env.df.loc[env.current_step, "Close"]]
    dates = [pd.to_datetime(env.df.loc[env.current_step, "Date"])]
    allocs = [0.0]  # initial allocation assumed 0
    holdings_prev = float(env.holdings)

    trades = []  # list of dicts to save as CSV

    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        ret = env.step(action)
        # support both Gymnasium (5-tuple) and legacy (4-tuple)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = ret

        step += 1
        net_worths.append(float(info.get("net_worth", env.net_worth)))
        price = float(info.get("price", env.df.loc[env.current_step, "Close"]))
        prices.append(price)
        date = pd.to_datetime(env.df.loc[min(env.current_step, len(env.df)-1), "Date"])
        dates.append(date)
        alloc_val = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        allocs.append(alloc_val)

        holdings = float(info.get("holdings", env.holdings))
        traded_value = float(info.get("traded_value", 0.0))
        delta_shares = holdings - holdings_prev
        holdings_prev = holdings

        # Detect "meaningful" trades using relative traded_value threshold
        current_portfolio = float(info.get("net_worth", env.net_worth)) or 1.0
        if traded_value > trade_threshold * current_portfolio:
            trade_type = "buy" if delta_shares > 0 else "sell"
            trades.append({
                "step": step,
                "date": date.strftime("%Y-%m-%d"),
                "type": trade_type,
                "price": price,
                "holdings": holdings,
                "delta_shares": delta_shares,
                "traded_value": traded_value
            })

    # build DataFrames
    df_port = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "price": prices,
        "net_worth": net_worths,
        "allocation": allocs
    })
    df_port = df_port.set_index("date")

    # Save trades CSV
    trades_csv = os.path.join("results/trades", f"{os.path.splitext(os.path.basename(data_path))[0]}_sac_trades.csv")
    if trades:
        with open(trades_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
            writer.writeheader()
            writer.writerows(trades)
    else:
        # ensure an empty CSV with headers
        with open(trades_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "date", "type", "price", "holdings", "delta_shares", "traded_value"])

    # Compute metrics
    metrics = compute_metrics(df_port["net_worth"])
    metrics["num_trades"] = len(trades)

    # Save metrics to CSV
    metrics_csv = os.path.join("results", f"{os.path.splitext(os.path.basename(data_path))[0]}_sac_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    # Plot: price with trade markers (use step-based x to avoid datetime plotting quirks)
    x = np.arange(len(df_port))
    prices_arr = df_port["price"].values
    plt.figure(figsize=(14, 6))
    plt.plot(df_port.index, prices_arr, label="Price", linewidth=1.2)
    if trades:
        buy_idx = [df_port.index.get_loc(pd.to_datetime(t["date"])) for t in trades if t["type"] == "buy"]
        sell_idx = [df_port.index.get_loc(pd.to_datetime(t["date"])) for t in trades if t["type"] == "sell"]
        # offsets
        offset = 0.01 * (prices_arr.max() - prices_arr.min()) if len(prices_arr) > 0 else 0.0
        if buy_idx:
            plt.scatter(df_port.index[buy_idx], prices_arr[buy_idx] + offset, marker="^", s=80, c="green", label="Buy", zorder=5)
        if sell_idx:
            plt.scatter(df_port.index[sell_idx], prices_arr[sell_idx] - offset, marker="v", s=80, c="red", label="Sell", zorder=5)
    plt.title("Price with Executed Trades (SAC)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    price_plot_path = f"{plot_prefix}_price_trades.png"
    plt.tight_layout()
    plt.savefig(price_plot_path)
    plt.close()

    # Plot: equity curve
    plt.figure(figsize=(12, 5))
    plt.plot(df_port.index, df_port["net_worth"].values, label="SAC Net Worth")
    plt.title("SAC Agent Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.grid(alpha=0.3)
    equity_plot_path = f"{plot_prefix}_equity.png"
    plt.tight_layout()
    plt.savefig(equity_plot_path)
    plt.close()

    # Plot: allocation over time
    plt.figure(figsize=(12, 3))
    plt.plot(df_port.index, df_port["allocation"].values, label="Allocation (target fraction)")
    plt.title("Allocation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Allocation")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    alloc_plot_path = f"{plot_prefix}_alloc.png"
    plt.tight_layout()
    plt.savefig(alloc_plot_path)
    plt.close()

    print("Saved plots:", price_plot_path, equity_plot_path, alloc_plot_path)
    print("Saved trades csv:", trades_csv)
    print("Saved metrics csv:", metrics_csv)
    print("Metrics:", metrics)

    return {
        "price_plot": price_plot_path,
        "equity_plot": equity_plot_path,
        "alloc_plot": alloc_plot_path,
        "trades_csv": trades_csv,
        "metrics_csv": metrics_csv,
        "metrics": metrics
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/models/sac_model.zip")
    parser.add_argument("--env", type=str, default="data/AAPL_test.csv")
    parser.add_argument("--out", type=str, default="results/plot_sac/sac_eval")
    parser.add_argument("--trade-threshold", type=float, default=0.01, help="min traded_value / net_worth to count as a trade")
    args = parser.parse_args()
    evaluate(args.model, args.env, args.out, trade_threshold=args.trade_threshold)

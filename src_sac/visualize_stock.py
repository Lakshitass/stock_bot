"""
Robust evaluation & plotting for single-stock or multi-stock models.

Outputs:
 - {out_prefix}_price_trades.png   : price line with buy/sell markers
 - {out_prefix}_equity.png         : RL net worth vs buy & hold
 - {out_prefix}_alloc.png          : allocation heatmap (if multi-stock)
 - {out_prefix}_trades.csv         : trade records
 - {out_prefix}_metrics.csv        : metrics (as returned by src_sac.metrics.evaluate_equity_curve)
 - several .npy/.csv artifacts for downstream analysis
"""
import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# keep mplfinance import optional (we won't use candlesticks)
try:
    import mplfinance as mpf  # not used for plotting to avoid index mismatch issues
except Exception:
    mpf = None

from stable_baselines3 import SAC, PPO
from pathlib import Path

# prefer single-stock env
try:
    from src_sac.env_continuous import TradingEnvContinuous as SingleEnv
except Exception:
    SingleEnv = None

# multi-stock env
try:
    from src_sac.multi_env import MultiStockTradingEnv
except Exception:
    MultiStockTradingEnv = None

from src_sac.metrics import evaluate_equity_curve

def _ticker_from_path(p):
    name = Path(p).stem
    for suf in ("_train", "_test", "-train", "-test"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name.upper()

def _safe_makedirs_for(prefix):
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)

def run_and_record(model_path, data_path, out_prefix="results/plot_sac/stock",
                   model_type="sac", trade_threshold=0.01):
    """
    Run model on given stock CSV and record trades / allocations / equity.

    model_path: path to a stable-baselines3 model (.zip)
    data_path: single-stock CSV (Date,Open,High,Low,Close,Volume...)
    out_prefix: prefix for output files (no suffixes appended)
    model_type: 'sac' | 'ppo' (controls loader)
    trade_threshold: fraction of net worth to consider as a trade
    """
    _safe_makedirs_for(out_prefix)
    df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    target_ticker = _ticker_from_path(data_path)

    # load model
    if model_type.lower() == "sac":
        model = SAC.load(model_path)
    elif model_type.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        model = SAC.load(model_path)

    # infer obs shape
    obs_shape = None
    try:
        obs_shape = int(model.policy.observation_space.shape[0])
    except Exception:
        obs_shape = None

    # choose environment
    if obs_shape is not None and SingleEnv is not None and obs_shape == getattr(SingleEnv, "__init__", lambda *a, **k: None).__code__.co_argcount - 1:
        # If SingleEnv exists and expected obs matches window, use it
        if SingleEnv is None:
            raise RuntimeError("SingleEnv not available.")
        env = SingleEnv(data_path)
        multi_price_df = None
    else:
        # Multi-stock model: build dataset and env (we will use full multi-env)
        try:
            from src_sac.data_utils import build_multi_feature_set
        except Exception:
            raise RuntimeError("src_sac.data_utils.build_multi_feature_set not available.")
        if MultiStockTradingEnv is None:
            raise RuntimeError("MultiStockTradingEnv not available.")
        # Default mapping — if user wants different set, pass a specially-built env before calling run_and_record
        ticker_files = {
            "AAPL": "data/AAPL.csv",
            "TSLA": "data/TSLA.csv",
            "JPM":  "data/JPM.csv",
            "GS":   "data/GS.csv",
            "XOM":  "data/XOM.csv",
        }
        price_df, corr_df, pca_df = build_multi_feature_set(ticker_files, window=10, corr_window=20, n_pca=3)
        multi_price_df = price_df.copy()
        env = MultiStockTradingEnv(price_df=price_df, corr_df=corr_df, pca_df=pca_df, window=10)

    # reset env (handle gym / gymnasium variants)
    try:
        obs, _ = env.reset()
    except Exception:
        obs = env.reset()

    net_worth = [float(getattr(env, "net_worth", 0.0))]
    dates = []
    prices = []
    allocations = []
    trades = []
    last_alloc = None
    step = 0

    # if multi env, map ticker index
    ticker_index = 0
    if multi_price_df is not None:
        cols_upper = [c.upper() for c in multi_price_df.columns.tolist()]
        if target_ticker in cols_upper:
            ticker_index = cols_upper.index(target_ticker)
        else:
            ticker_index = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        ret = env.step(action)
        # gymnasium-style (obs, reward, terminated, truncated, info)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = ret

        step += 1

        # net worth
        nw = float(info.get("net_worth", getattr(env, "net_worth", net_worth[-1])))
        net_worth.append(nw)

        # price data handling
        raw_price = info.get("price", None)
        this_date = None
        if raw_price is None:
            # try to fetch from env-specific dataframes
            if hasattr(env, "df"):
                idx = min(getattr(env, "current_step", 0), len(env.df)-1)
                raw_price = float(env.df.loc[idx, "Close"])
                this_date = pd.to_datetime(env.df.loc[idx, "Date"])
            elif multi_price_df is not None and hasattr(env, "current_step"):
                idx = min(int(env.current_step), len(multi_price_df)-1)
                # use column for ticker_index
                raw_price = float(multi_price_df.iloc[idx, ticker_index])
                try:
                    this_date = pd.to_datetime(multi_price_df.index[idx])
                except Exception:
                    this_date = None
            else:
                raw_price = 0.0
                this_date = pd.Timestamp.now()
        else:
            if isinstance(raw_price, (list, tuple, np.ndarray)):
                # pick ticker_index when available
                try:
                    raw_price = float(raw_price[ticker_index])
                except Exception:
                    raw_price = float(raw_price[0])
            raw_price = float(raw_price)
            # attempt to get date from env
            if hasattr(env, "df"):
                idx = min(getattr(env, "current_step", 0), len(env.df)-1)
                this_date = pd.to_datetime(env.df.loc[idx, "Date"])
            elif multi_price_df is not None and hasattr(env, "current_step") and hasattr(env, "dates"):
                try:
                    this_date = pd.to_datetime(env.dates[min(int(env.current_step), len(env.dates)-1)])
                except Exception:
                    this_date = None
            if this_date is None:
                this_date = pd.Timestamp.now()

        prices.append(float(raw_price))
        dates.append(this_date)

        alloc_val = np.array(action).astype(float).ravel()
        allocations.append(alloc_val)

        traded_value = float(info.get("traded_value", 0.0))
        # treat a trade if traded_value > threshold * net worth
        if traded_value > trade_threshold * max(1.0, nw):
            side = "buy" if (last_alloc is None or alloc_val.sum() > last_alloc.sum()) else "sell"
            trades.append({
                "step": int(info.get("step", step)),
                "date": this_date.strftime("%Y-%m-%d") if this_date is not None else None,
                "type": side,
                "price": float(raw_price),
                "traded_value": float(traded_value),
                "allocation": alloc_val.tolist()
            })
        last_alloc = alloc_val

        if done:
            break

    # Build DataFrame (align and coerce unique index)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    # deduplicate index if duplicates exist by adding small offsets (to allow saving to csv)
    if idx.duplicated().any():
        # make a monotonic increasing index by adding tiny increments to duplicates
        new_idx = []
        last = None
        offset = pd.Timedelta(milliseconds=1)
        for d in idx:
            if last is not None and d <= last:
                d = last + offset
            new_idx.append(d)
            last = d
        idx = pd.DatetimeIndex(new_idx)

    df_port = pd.DataFrame({
        "date": idx,
        "price": prices,
        "net_worth": net_worth[1:]  # first element was initial
    }).set_index("date")

    # Save numeric artifacts
    try:
        np.save(out_prefix + "_networth.npy", df_port["net_worth"].values)
        df_port.reset_index().to_csv(out_prefix + "_networth.csv", index=False)
        np.save(out_prefix + "_prices.npy", df_port["price"].values)
    except Exception as e:
        print("[save] Warning saving npy/csv:", e)

    # Save trades CSV
    trades_csv = out_prefix + "_trades.csv"
    try:
        with open(trades_csv, "w", newline="") as f:
            if trades:
                keys = list(trades[0].keys())
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in trades:
                    writer.writerow(r)
            else:
                writer = csv.writer(f)
                writer.writerow(["step", "date", "type", "price", "traded_value", "allocation"])
    except Exception as e:
        print("[save] Warning writing trades csv:", e)

    # Metrics
    metrics = evaluate_equity_curve(df_port["net_worth"])
    metrics_path = out_prefix + "_metrics.csv"
    try:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    except Exception as e:
        print("[save] Warning writing metrics:", e)

    # -------------------------
    # Plot A: price series with buy / sell markers (robust)
    # -------------------------
    price_trades_path = out_prefix + "_price_trades.png"
    try:
        plt.figure(figsize=(12,5))
        plt.plot(df_port.index, df_port["price"], label="Price", linewidth=1.2)

        buy_dates = [pd.to_datetime(t["date"]) for t in trades if t["type"] == "buy" and t["date"]]
        sell_dates = [pd.to_datetime(t["date"]) for t in trades if t["type"] == "sell" and t["date"]]

        idx_index = df_port.index
        buy_mapped = idx_index.intersection(pd.DatetimeIndex(buy_dates))
        sell_mapped = idx_index.intersection(pd.DatetimeIndex(sell_dates))

        unmapped_buys = max(0, len(buy_dates) - len(buy_mapped))
        unmapped_sells = max(0, len(sell_dates) - len(sell_mapped))

        if len(buy_mapped) > 0:
            y_buy = df_port.loc[buy_mapped, "price"]
            plt.scatter(buy_mapped, y_buy, marker="^", s=80, label="Buy", zorder=5)
        if len(sell_mapped) > 0:
            y_sell = df_port.loc[sell_mapped, "price"]
            plt.scatter(sell_mapped, y_sell, marker="v", s=80, label="Sell", zorder=5)

        note = []
        if unmapped_buys:
            note.append(f"{unmapped_buys} unmapped buys")
        if unmapped_sells:
            note.append(f"{unmapped_sells} unmapped sells")
        if note:
            plt.title(f"{Path(data_path).stem} Price with Trades ({', '.join(note)})")
        else:
            plt.title(f"{Path(data_path).stem} Price with Trades")

        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(price_trades_path)
        plt.close()
    except Exception as e:
        price_trades_path = None
        print("[plot] Warning: failed to save price+trade markers plot:", e)

    # -------------------------
    # Plot B: equity curve vs buy-hold
    # -------------------------
    equity_path = out_prefix + "_equity.png"
    try:
        start_price = float(df_port["price"].iloc[0]) if len(df_port) > 0 else 1.0
        bh_shares = 10000.0 / start_price if start_price > 0 else 0.0
        bh_port = bh_shares * df_port["price"]  # fallback if no OHLC close
    except Exception:
        bh_port = pd.Series(0.0, index=df_port.index)

    try:
        plt.figure(figsize=(10,5))
        plt.plot(df_port.index, df_port["net_worth"], label="RL Net Worth")
        plt.plot(df_port.index, bh_port, label="Buy & Hold", linestyle="--")
        plt.title(f"{Path(data_path).stem} Equity Curve (RL vs Buy & Hold)")
        plt.xlabel("Date"); plt.ylabel("Portfolio Value"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(equity_path); plt.close()
    except Exception as e:
        equity_path = None
        print("[plot] Warning: failed to save equity plot:", e)

    # -------------------------
    # Plot C: allocations heatmap (if multi / vector actions)
    # -------------------------
    alloc_path = out_prefix + "_alloc.png"
    try:
        if len(allocations) > 0 and getattr(allocations[0], "__len__", None) and len(np.array(allocations).shape) == 2:
            alloc_mat = np.vstack([np.array(a).ravel() for a in allocations])
            np.save(out_prefix + "_alloc.npy", alloc_mat)
            alloc_df = pd.DataFrame(alloc_mat, columns=[f"w{i}" for i in range(alloc_mat.shape[1])])
            try:
                alloc_df.insert(0, "date", df_port.index.astype(str)[:len(alloc_df)])
            except Exception:
                alloc_df.insert(0, "step", np.arange(len(alloc_df)))
            alloc_df.to_csv(out_prefix + "_alloc.csv", index=False)

            plt.figure(figsize=(12,3))
            plt.imshow(alloc_mat.T, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
            plt.yticks(np.arange(alloc_mat.shape[1]), [f"w{i}" for i in range(alloc_mat.shape[1])])
            xt_idx = np.linspace(0, alloc_mat.shape[0]-1, min(10, alloc_mat.shape[0])).astype(int)
            xt_labels = [df_port.index[int(x)].strftime("%Y-%m-%d") for x in xt_idx]
            plt.xticks(xt_idx, xt_labels, rotation=45)
            plt.colorbar(label="Allocation fraction")
            plt.title(f"{Path(data_path).stem} Allocations Over Time")
            plt.tight_layout(); plt.savefig(alloc_path); plt.close()
        else:
            alloc_path = None
    except Exception as e:
        alloc_path = None
        print("[plot] Warning: failed to save allocation heatmap:", e)

    return {
        "price_trades": price_trades_path,
        "equity": equity_path,
        "alloc": alloc_path,
        "trades_csv": trades_csv,
        "metrics_csv": metrics_path,
        "metrics": metrics
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out", default="results/plot_sac/stock")
    parser.add_argument("--model-type", default="sac")
    parser.add_argument("--trade-threshold", type=float, default=0.01)
    args = parser.parse_args()
    res = run_and_record(args.model, args.env, args.out, model_type=args.model_type, trade_threshold=args.trade_threshold)
    print("Done. Artifacts:", res)

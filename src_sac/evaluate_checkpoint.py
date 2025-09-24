import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC, PPO
import mplfinance as mpf
import os

# try import envs
try:
    from src_sac.env_continuous import TradingEnvContinuous as SingleEnv
except Exception:
    SingleEnv = None

try:
    from src_sac.multi_env import MultiStockTradingEnv
except Exception:
    MultiStockTradingEnv = None

def load_model(path, model_type=None):
    if model_type is None:
        # try to infer from file by trying SAC then PPO
        try:
            return SAC.load(path), "sac"
        except Exception:
            return PPO.load(path), "ppo"
    model_type = model_type.lower()
    if model_type == "sac":
        return SAC.load(path), "sac"
    elif model_type == "ppo":
        return PPO.load(path), "ppo"
    else:
        raise ValueError("Unknown model_type")

def make_env_for_ticker(data_path, expected_obs_size=None, multi_price_df=None, window=10):
    # If model expects single-stock obs (expected_obs_size == window), use SingleEnv
    if SingleEnv is not None and (expected_obs_size is None or expected_obs_size <= 50):
        return SingleEnv(data_path, window=window)
    # else, try to construct MultiStockTradingEnv with single column price_df
    if MultiStockTradingEnv is not None:
        df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        # build price_df with single column named ticker
        ticker = Path(data_path).stem.replace("_test","").replace("_train","")
        price_df = pd.DataFrame(df["Close"].values, columns=[ticker])
        # create dummy corr/pca as None
        return MultiStockTradingEnv(price_df=price_df, corr_df=None, pca_df=None, window=window)
    raise RuntimeError("No compatible env class found (SingleEnv or MultiStockTradingEnv)")

def run_episode(model, env, deterministic=True, trade_threshold=0.0):
    # reset (handle gymnasium style)
    try:
        obs, _ = env.reset()
    except Exception:
        obs = env.reset()

    done = False
    step = 0
    trades = []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        ret = env.step(action)
        # gymnasium returns 5-tuple
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = ret

        step += 1
        traded_value = float(info.get("traded_value", 0.0))
        nw = float(info.get("net_worth", getattr(env, "net_worth", None)))
        price_raw = info.get("price", None)
        # price may be scalar or list
        if isinstance(price_raw, (list, tuple, np.ndarray)):
            price = float(price_raw[0])
        else:
            price = float(price_raw) if price_raw is not None else None

        alloc = info.get("allocation", None)
        if alloc is None:
            # try to interpret action as alloc
            alloc = np.array(action).astype(float).ravel().tolist()

        if traded_value > trade_threshold * max(1.0, nw):
            side = "buy" if (sum(alloc) > 0.5) else "sell"
            trades.append({
                "step": step,
                "date": None,
                "type": side,
                "price": price,
                "traded_value": traded_value,
                "allocation": alloc
            })
        if done:
            break

    # gather history if present
    history_alloc = getattr(env, "history_alloc", None)
    history_networth = getattr(env, "history_networth", None)

    return {
        "trades": trades,
        "history_alloc": np.array(history_alloc) if history_alloc is not None else None,
        "history_networth": np.array(history_networth) if history_networth is not None else None,
        "env": env
    }

def save_artifacts(out_prefix, df_test, result):
    Path(os.path.dirname(out_prefix)).mkdir(parents=True, exist_ok=True)
    # trades
    trades_csv = out_prefix + "_trades.csv"
    trades = result["trades"]
    if trades:
        pd.DataFrame(trades).to_csv(trades_csv, index=False)
    else:
        # write header
        pd.DataFrame([], columns=["step","date","type","price","traded_value","allocation"]).to_csv(trades_csv, index=False)

    # history arrays
    if result["history_alloc"] is not None:
        np.save(out_prefix + "_alloc.npy", result["history_alloc"])
    if result["history_networth"] is not None:
        np.save(out_prefix + "_networth.npy", result["history_networth"])
        # also save a CSV for easy reading
        pd.DataFrame({"networth": result["history_networth"]}).to_csv(out_prefix + "_networth.csv", index=False)

    # candlestick (if df_test has OHLC)
    try:
        ohlc = df_test.copy().set_index("Date")
        # build buy/sell index
        buys = [t for t in trades if t["type"]=="buy"]
        sells = [t for t in trades if t["type"]=="sell"]
        apds = []
        if buys:
            buy_indices = [int(t["step"]) for t in buys if t["price"] is not None and not np.isnan(t["price"])]
            if len(buy_indices)>0:
                apds.append(mpf.make_addplot(ohlc.iloc[buy_indices]["Low"]*0.995, type='scatter', marker='^', markersize=80, color='g'))
        if sells:
            sell_indices = [int(t["step"]) for t in sells if t["price"] is not None and not np.isnan(t["price"])]
            if len(sell_indices)>0:
                apds.append(mpf.make_addplot(ohlc.iloc[sell_indices]["High"]*1.005, type='scatter', marker='v', markersize=80, color='r'))

        candlestick_path = out_prefix + "_candlestick.png"
        if len(apds)>0:
            mpf.plot(ohlc, type='candle', addplot=apds, savefig=candlestick_path)
        else:
            mpf.plot(ohlc, type='candle', savefig=candlestick_path)
    except Exception:
        candlestick_path = None

    # equity curve
    equity_path = out_prefix + "_equity.png"
    if result["history_networth"] is not None:
        nw = result["history_networth"]
        plt.figure(figsize=(10,4))
        plt.plot(nw, label="Net worth")
        plt.title("Equity Curve")
        plt.xlabel("Step")
        plt.ylabel("Net worth")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(equity_path)
        plt.close()
    else:
        equity_path = None

    # allocation heatmap
    alloc_path = out_prefix + "_alloc_heat.png"
    if result["history_alloc"] is not None:
        alloc = result["history_alloc"]
        # alloc shape: (T, n) or (T,) if scalar per step
        if alloc.ndim == 1:
            # scalar allocations -> reshape to (T,1)
            alloc = alloc.reshape(-1,1)
        plt.figure(figsize=(12,3))
        plt.imshow(alloc.T, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
        plt.yticks(np.arange(alloc.shape[1]), [f"w{i}" for i in range(alloc.shape[1])])
        plt.title("Allocations over time (rows=tickers or dims)")
        plt.colorbar(label="allocation")
        plt.tight_layout()
        plt.savefig(alloc_path)
        plt.close()
    else:
        alloc_path = None

    return {"trades_csv": trades_csv, "candlestick": candlestick_path, "equity": equity_path, "alloc": alloc_path}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", required=True, help="path to single-stock test CSV (Date col, Open/High/Low/Close/Volume)")
    parser.add_argument("--out", required=True, help="output prefix (e.g., results/plot_model/AAPL_pretrain)")
    parser.add_argument("--model-type", default=None, help="optional: sac|ppo")
    parser.add_argument("--trade-threshold", type=float, default=0.01)
    args = parser.parse_args()

    model, model_type = load_model(args.model, model_type=args.model_type)
    df_test = pd.read_csv(args.env, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # try infer expected obs size from model policy if available
    expected_obs_size = None
    try:
        expected_obs_size = model.policy.observation_space.shape[0]
    except Exception:
        expected_obs_size = None

    env = make_env_for_ticker(args.env, expected_obs_size=expected_obs_size)
    result = run_episode(model, env, deterministic=True, trade_threshold=args.trade_threshold)
    artifacts = save_artifacts(args.out, df_test, result)
    print("Saved:", artifacts)

if __name__ == "__main__":
    main()

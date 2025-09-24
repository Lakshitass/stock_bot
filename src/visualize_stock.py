import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC, PPO, DQN
import gym
from src.metrics import evaluate_equity_curve

def _ticker_from_path(p):
    return Path(p).stem.upper().split("_")[0]

def _safe_makedirs_for(prefix):
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)

def _load_model(model_path, model_type):
    model_type = (model_type or "").lower()
    if model_type == "sac":
        return SAC.load(model_path)
    if model_type == "ppo":
        return PPO.load(model_path)
    if model_type == "dqn":
        return DQN.load(model_path)
    try:
        return SAC.load(model_path)
    except Exception:
        return PPO.load(model_path)

def _ensure_obs_for_predict(obs):
    if isinstance(obs, tuple) and len(obs) >= 1:
        return obs[0]
    return obs

def _safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        if len(out) == 2:
            obs, info = out
            return obs, info or {}
        if len(out) == 1:
            return out[0], {}
    return out, {}

def _safe_step(env, action):
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, float(reward), done, info or {}
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info or {}
    raise RuntimeError("Unrecognized env.step() return signature")

def _is_discrete_env(env_inst, model=None):
    try:
        a_space = getattr(env_inst, "action_space", None)
        if a_space is not None:
            return isinstance(a_space, gym.spaces.Discrete)
    except Exception:
        pass
    if model is not None:
        try:
            m_as = getattr(model, "action_space", None) or getattr(model.policy, "action_space", None)
            if isinstance(m_as, gym.spaces.Discrete):
                return True
        except Exception:
            pass
    return False

def run_and_record(model_path, data_path, out_prefix="results/plots/stock", model_type="sac", trade_threshold=0.01):
    _safe_makedirs_for(out_prefix)
    df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    target_ticker = _ticker_from_path(data_path)
    model = _load_model(model_path, model_type)
    env_inst = None
    try:
        from src.env_continuous import TradingEnvContinuous
        from src.env_discrete import TradingEnvDiscrete
    except Exception:
        TradingEnvContinuous = None
        TradingEnvDiscrete = None

    env_chosen = None
    try:
        if TradingEnvContinuous is not None:
            env_c = TradingEnvContinuous(df)
            if not _is_discrete_env(env_c, model):
                env_chosen = env_c
        if env_chosen is None and TradingEnvDiscrete is not None:
            env_d = TradingEnvDiscrete(df)
            env_chosen = env_d
        if env_chosen is None:
            if TradingEnvContinuous is not None:
                env_chosen = TradingEnvContinuous(df)
            else:
                env_chosen = TradingEnvDiscrete(df)
    except Exception:
        try:
            env_chosen = TradingEnvContinuous(df)
        except Exception:
            env_chosen = TradingEnvDiscrete(df)

    obs, info = _safe_reset(env_chosen)
    obs = _ensure_obs_for_predict(obs)
    net_worth = [float(info.get("net_worth", getattr(env_chosen, "net_worth", float(0.0))))]
    dates = []
    prices = []
    allocations = []
    trades = []
    last_alloc = None
    step = 0
    ticker_index = 0

    while True:
        try:
            action_res = model.predict(obs, deterministic=True)
        except Exception:
            action_res = model.predict(obs)
        if isinstance(action_res, tuple):
            action = action_res[0]
        else:
            action = action_res
        if action is None:
            if _is_discrete_env(env_chosen, model):
                action = 0
            else:
                action = np.array([0.0])
        ret_obs, reward, done, info = _safe_step(env_chosen, action)
        obs = _ensure_obs_for_predict(ret_obs)
        step += 1
        nw = float(info.get("net_worth", getattr(env_chosen, "net_worth", net_worth[-1])))
        net_worth.append(nw)
        raw_price = info.get("price", None)
        this_date = None
        if raw_price is None:
            try:
                idx = min(getattr(env_chosen, "current_step", 0), len(df)-1)
                raw_price = float(df.loc[idx, "Close"])
                this_date = pd.to_datetime(df.loc[idx, "Date"])
            except Exception:
                raw_price = 0.0
                this_date = pd.Timestamp.now()
        else:
            try:
                raw_price = float(raw_price) if not isinstance(raw_price, (list, tuple, np.ndarray)) else float(np.atleast_1d(raw_price).ravel()[0])
            except Exception:
                raw_price = 0.0
            if hasattr(env_chosen, "current_step") and env_chosen.current_step is not None:
                try:
                    this_date = pd.to_datetime(df.loc[min(int(env_chosen.current_step), len(df)-1), "Date"])
                except Exception:
                    this_date = pd.Timestamp.now()
            else:
                this_date = pd.Timestamp.now()
        prices.append(float(raw_price))
        dates.append(this_date)
        try:
            alloc_val = np.atleast_1d(action).astype(float).ravel()
        except Exception:
            try:
                alloc_val = np.array([float(action)])
            except Exception:
                alloc_val = np.array([0.0])
        allocations.append(alloc_val)
        traded_value = float(info.get("traded_value", 0.0))
        if traded_value > trade_threshold * max(1.0, nw):
            side = "buy" if (last_alloc is None or alloc_val.sum() > last_alloc.sum()) else "sell"
            trades.append({
                "step": int(info.get("step", step)),
                "date": this_date.strftime("%Y-%m-%d"),
                "type": side,
                "price": float(raw_price),
                "traded_value": float(traded_value),
                "allocation": alloc_val.tolist()
            })
        last_alloc = alloc_val
        if done:
            break

    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if idx.duplicated().any():
        new_idx = []
        last = None
        offset = pd.Timedelta(milliseconds=1)
        for d in idx:
            if last is not None and d <= last:
                d = last + offset
            new_idx.append(d)
            last = d
        idx = pd.DatetimeIndex(new_idx)

    df_port = pd.DataFrame({"date": idx, "price": prices, "net_worth": net_worth[1:]}).set_index("date")
    try:
        np.save(out_prefix + "_networth.npy", df_port["net_worth"].values)
        df_port.reset_index().to_csv(out_prefix + "_networth.csv", index=False)
        np.save(out_prefix + "_prices.npy", df_port["price"].values)
    except Exception:
        pass

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
    except Exception:
        pass

    metrics = evaluate_equity_curve(df_port["net_worth"])
    metrics_path = out_prefix + "_metrics.csv"
    try:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    except Exception:
        pass

    price_trades_path = out_prefix + "_price_trades.png"
    try:
        plt.figure(figsize=(12,5))
        plt.plot(df_port.index, df_port["price"], label="Price", linewidth=1.2)
        buy_dates = [pd.to_datetime(t["date"]) for t in trades if t["type"] == "buy"]
        sell_dates = [pd.to_datetime(t["date"]) for t in trades if t["type"] == "sell"]
        idx_index = df_port.index
        buy_mapped = idx_index.intersection(pd.DatetimeIndex(buy_dates)) if buy_dates else pd.DatetimeIndex([])
        sell_mapped = idx_index.intersection(pd.DatetimeIndex(sell_dates)) if sell_dates else pd.DatetimeIndex([])
        if len(buy_mapped) > 0:
            y_buy = df_port.loc[buy_mapped, "price"]
            plt.scatter(buy_mapped, y_buy, marker="^", s=80, label="Buy", zorder=5, color="g")
        if len(sell_mapped) > 0:
            y_sell = df_port.loc[sell_mapped, "price"]
            plt.scatter(sell_mapped, y_sell, marker="v", s=80, label="Sell", zorder=5, color="r")
        note = []
        unmapped_buys = max(0, len([d for d in buy_dates if d not in buy_mapped]) ) if buy_dates else 0
        unmapped_sells = max(0, len([d for d in sell_dates if d not in sell_mapped]) ) if sell_dates else 0
        if unmapped_buys:
            note.append(f"{unmapped_buys} unmapped buys")
        if unmapped_sells:
            note.append(f"{unmapped_sells} unmapped sells")
        if note:
            plt.title(f"{Path(data_path).stem} Price with Trades ({', '.join(note)})")
        else:
            plt.title(f"{Path(data_path).stem} Price with Trades")
        plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(price_trades_path); plt.close()
    except Exception:
        price_trades_path = None

    equity_path = out_prefix + "_equity.png"
    try:
        start_price = float(df_port["price"].iloc[0]) if len(df_port) > 0 else 1.0
        bh_shares = 10000.0 / start_price if start_price > 0 else 0.0
        bh_port = bh_shares * df_port["price"]
        plt.figure(figsize=(10,5))
        plt.plot(df_port.index, df_port["net_worth"], label="RL Net Worth")
        plt.plot(df_port.index, bh_port, label="Buy & Hold", linestyle="--")
        plt.title(f"{Path(data_path).stem} Equity Curve (RL vs Buy & Hold)")
        plt.xlabel("Date"); plt.ylabel("Portfolio Value"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(equity_path); plt.close()
    except Exception:
        equity_path = None

    alloc_path = out_prefix + "_alloc.png"
    try:
        if len(allocations) > 0 and hasattr(allocations[0], "__len__") and len(np.array(allocations).shape) == 2:
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
    except Exception:
        alloc_path = None

    return {"price_trades": price_trades_path, "equity": equity_path, "alloc": alloc_path, "trades_csv": trades_csv, "metrics_csv": metrics_path, "metrics": metrics}

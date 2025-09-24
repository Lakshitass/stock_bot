# src_sac/run_all.py
"""
Top-level orchestrator to:
  - Train or load a multi-stock "pretrain" SAC model
  - Evaluate pretrain on each ticker (saving equity, trades, allocs, metrics)
  - Fine-tune the pretrain per ticker (using wrapper to map action/obs spaces)
  - Evaluate each fine-tuned model
  - Aggregate metrics into CSVs

Usage:
  python -m src_sac.run_all --pretrain-timesteps 200000 --finetune-timesteps 5000
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import utilities from repo (now under src_sac)
try:
    from src_sac.visualize_stock import run_and_record
    from src_sac.env_continuous import TradingEnvContinuous
    from src_sac.data_utils import build_multi_feature_set
    from src_sac.multi_env import MultiStockTradingEnv
    from src_sac.train_multi_sac import train as train_multi_sac  # optional helper (if present)
    from src_sac.metrics import evaluate_equity_curve
except Exception as e:
    print("[run_all] ERROR importing project modules:", e)
    raise

# -------------------------
# Wrapper: present single-stock env as multi-stock for a pre-trained multi-stock model
# -------------------------
import numpy as _np
class MultiModelToSingleEnv(gym.Env):
    """
    Wrap a single-stock TradingEnvContinuous to appear as the multi-stock env expected by a
    pre-trained multi-stock model.
    """
    def __init__(self, single_env: gym.Env, model_obs_space: gym.spaces.Space,
                 model_action_space: gym.spaces.Space, ticker_index: int = 0):
        super().__init__()
        self.single = single_env
        self.model_obs_space = model_obs_space
        self.model_action_space = model_action_space
        self.ticker_index = int(ticker_index)

        # Expose the model's spaces so SB3 accepts the env when calling model.set_env(...)
        self.observation_space = model_obs_space
        self.action_space = model_action_space

    def _embed_obs(self, single_obs):
        """Pad or trim single_obs to match model_obs_space shape (1D)."""
        arr = _np.asarray(single_obs, dtype=_np.float32).ravel()
        target_shape = getattr(self.model_obs_space, "shape", None)
        if (not target_shape) or len(target_shape) == 0:
            return arr.astype(_np.float32)
        target_len = int(target_shape[0])
        if arr.size >= target_len:
            return arr[:target_len].astype(_np.float32)
        pad = _np.zeros((target_len - arr.size,), dtype=_np.float32)
        return _np.concatenate([arr.astype(_np.float32), pad])

    def reset(self, **kwargs):
        out = self.single.reset(**kwargs)
        # gymnasium reset -> (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            single_obs, info = out
        else:
            single_obs = out
            info = {}
        obs = self._embed_obs(single_obs)
        return obs.astype(_np.float32), info

    def step(self, action):
        # action is expected to be full vector for the multi-stock model
        act = _np.asarray(action, dtype=_np.float32).ravel()
        if act.size > 0 and self.ticker_index < act.size:
            scalar = float(act[self.ticker_index])
        elif act.size > 0:
            scalar = float(_np.mean(act))
        else:
            scalar = 0.0
        scalar_action = _np.array([scalar], dtype=_np.float32)

        ret = self.single.step(scalar_action)
        # support both gymnasium (5-tuple) and gym (4-tuple)
        if len(ret) == 5:
            single_obs, reward, terminated, truncated, info = ret
        else:
            single_obs, reward, done, info = ret
            terminated = bool(done)
            truncated = False

        obs = self._embed_obs(single_obs)
        return obs.astype(_np.float32), float(reward), bool(terminated), bool(truncated), info or {}

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def train_pretrain_model(save_path, ticker_files, timesteps, window=10, tensorboard="runs/multi_sac"):
    """
    Train a multi-stock SAC model (if save_path already exists, it will be returned).
    Returns path to model.
    """
    save_path = Path(save_path)
    if save_path.exists():
        print(f"[main] Using existing pretrain model: {save_path}")
        return str(save_path)
    print(f"[main] Training pretrain multi-stock model -> {save_path} (timesteps={timesteps})")
    try:
        train_multi_sac(str(save_path), timesteps=timesteps, window=window, tensorboard_log=tensorboard)
        return str(save_path)
    except Exception as e:
        print("[main] train_multi_sac helper failed or not present:", e)
        raise RuntimeError("Pretrain training failed; please run src_sac/train_multi_sac.py manually or ensure it exists.") from e

def finetune_per_ticker(pretrain_model_path, ticker, train_csv, save_path,
                        timesteps=5000, window=10, ticker_index=0, seed=0):
    """
    Fine-tune a multi-stock pretrain model on a single ticker by wrapping the single-stock env.
    Returns path to saved finetuned model.
    """
    ensure_dir(save_path)
    print(f"[finetune] Loading pretrain {pretrain_model_path} and fine-tuning for {ticker} -> {save_path}")

    # Load pretrain model (we will reuse its policy & replay buffer/state)
    model = SAC.load(pretrain_model_path, device="auto")

    # Build single-stock env for ticker training
    single_env = TradingEnvContinuous(train_csv, window=window)
    single_env = Monitor(single_env)
    wrapped = MultiModelToSingleEnv(single_env, model.observation_space, model.action_space, ticker_index=ticker_index)

    vec_env = DummyVecEnv([lambda: wrapped])
    model.set_env(vec_env)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    vec_env.close()
    print(f"[finetune] Saved finetuned model to {save_path}")
    return str(save_path)

def evaluate_and_record(model_path, test_csv, out_prefix, model_type="sac", trade_threshold=0.01):
    """
    Run run_and_record (from src_sac.visualize_stock) and return its artifacts dict.
    """
    ensure_dir(out_prefix + "_metrics.csv")
    print(f"[eval] Evaluating model {model_path} on {test_csv} -> {out_prefix}")
    res = run_and_record(model_path, test_csv, out_prefix=out_prefix, model_type=model_type, trade_threshold=trade_threshold)
    return res

# -------------------------
# Main orchestration
# -------------------------
def main(args):
    # tickers to process (order matters for ticker_index mapping)
    tickers = args.tickers or ["AAPL", "TSLA", "JPM", "GS", "XOM"]
    print("[main] Tickers to process:", tickers)

    # mapping ticker -> train/test CSV paths (assumes data/* exists)
    ticker_files_train = {t: f"data/{t}_train.csv" if Path(f"data/{t}_train.csv").exists() else f"data/{t}.csv" for t in tickers}
    ticker_files_test = {t: f"data/{t}_test.csv" if Path(f"data/{t}_test.csv").exists() else f"data/{t}.csv" for t in tickers}

    # Prepare multi-stock file mapping for pretrain
    multi_ticker_map = {t: f"data/{t}.csv" for t in tickers}

    # 1) Pretrain: train or load multi-stock model (pretrain is multi-stock)
    pretrain_path = Path(args.pretrain_save)
    if not pretrain_path.exists():
        pretrain_path_str = train_pretrain_model(str(pretrain_path), multi_ticker_map, timesteps=args.pretrain_timesteps, window=args.window)
    else:
        pretrain_path_str = str(pretrain_path)
    print("[main] Pretrain model path:", pretrain_path_str)

    # DataFrame to collect aggregated metrics
    agg_columns = ["ticker", "phase", "total_return", "annualized_return", "sharpe", "max_drawdown", "num_trades", "model_path", "artifact_prefix"]
    agg_rows = []

    # Evaluate pretrain per ticker (multi-stock evaluation artifacts stored under results/plot_model/)
    for idx, t in enumerate(tickers):
        print("\n" + "="*20 + f" Processing ticker {t} " + "="*20)
        test_csv = ticker_files_test[t]
        out_pre = f"results/plot_model/{t}_pretrain"
        pre_artifacts = evaluate_and_record(pretrain_path_str, test_csv, out_pre, model_type="sac", trade_threshold=args.trade_threshold)
        print("[main] Pretrain artifacts:", pre_artifacts.get("metrics", {}))
        metrics = pre_artifacts.get("metrics", {})
        agg_rows.append({
            "ticker": t, "phase": "pretrain",
            "total_return": metrics.get("total_return"),
            "annualized_return": metrics.get("annualized_return"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "num_trades": metrics.get("num_trades"),
            "model_path": pretrain_path_str,
            "artifact_prefix": out_pre
        })

        # 2) Fine-tune per ticker (from pretrain) -> fine-tuned single-stock evaluation stored under results/plot_sac/
        finetune_save = Path(args.finetune_save_template.format(ticker=t))
        ticker_index = idx  # we built pretrain with this tickers order
        try:
            finetune_path = finetune_per_ticker(pretrain_path_str, t, ticker_files_train[t], str(finetune_save),
                                                timesteps=args.finetune_timesteps, window=args.window, ticker_index=ticker_index)
        except Exception as e:
            print("[finetune] ERROR finetuning for", t, ":", e)
            finetune_path = None

        # Evaluate finetuned
        if finetune_path:
            out_ft = f"results/plot_sac/{t}_finetuned"
            ft_artifacts = evaluate_and_record(finetune_path, test_csv, out_ft, model_type="sac", trade_threshold=args.trade_threshold)
            print("[main] Finetune artifacts:", ft_artifacts.get("metrics", {}))
            metrics = ft_artifacts.get("metrics", {})
            agg_rows.append({
                "ticker": t, "phase": "finetune",
                "total_return": metrics.get("total_return"),
                "annualized_return": metrics.get("annualized_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "num_trades": metrics.get("num_trades"),
                "model_path": finetune_path,
                "artifact_prefix": out_ft
            })
        else:
            print("[main] Skipping finetune evaluation due to finetune failure.")

    # Save aggregated metrics CSV (under results/plot_model)
    agg_df = pd.DataFrame(agg_rows, columns=agg_columns)
    agg_dir = Path("results/plot_model")
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(agg_dir / "aggregated_metrics_pretrain_finetune.csv", index=False)
    print("[main] Aggregated metrics saved to results/plot_model/aggregated_metrics_pretrain_finetune.csv")
    print("[main] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None, help="List of tickers to process (default 5 defined in code)")
    parser.add_argument("--pretrain-timesteps", type=int, default=200000)
    parser.add_argument("--finetune-timesteps", type=int, default=20000)
    parser.add_argument("--pretrain-save", type=str, default="results/models/multi_sac_pretrain.zip")
    parser.add_argument("--finetune-save-template", dest="finetune_save_template", type=str,
                        default="results/models/{ticker}_finetuned_from_pretrain.zip",
                        help="Template for per-ticker finetune save path (use {ticker})")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--trade-threshold", type=float, default=0.01)
    args = parser.parse_args()
    main(args)

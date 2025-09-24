import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from src.env_discrete import TradingEnvDiscrete
from src.visualize_stock import run_and_record
from src.metrics import evaluate_equity_curve

def train_and_eval_single_ticker(ticker, train_csv, test_csv, save_path, timesteps=50000, tensorboard_log=None, seed=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_df = pd.read_csv(train_csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    env = TradingEnvDiscrete(train_df)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)
    model = DQN("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=1e-4, buffer_size=50000, batch_size=64, seed=seed)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    vec_env.close()
    eval_env = TradingEnvDiscrete(train_df)
    evaluate_policy(model, eval_env, n_eval_episodes=5)
    out_prefix = f"results/plots/{ticker}_dqn"
    artifacts = run_and_record(save_path, test_csv, out_prefix=out_prefix, model_type="dqn", trade_threshold=0.01)
    return save_path, artifacts

def main(args):
    tickers = args.tickers or ["AAPL", "TSLA", "JPM", "GS", "XOM"]
    timesteps = args.timesteps
    os.makedirs("results/models", exist_ok=True)
    agg = []
    for t in tickers:
        train_csv = f"data/{t}.csv"
        test_csv = f"data/{t}_test.csv" if Path(f"data/{t}_test.csv").exists() else train_csv
        model_path = f"results/models/{t}_dqn.zip"
        model_path, artifacts = train_and_eval_single_ticker(t, train_csv, test_csv, model_path, timesteps=timesteps, tensorboard_log=args.tensorboard)
        metrics = artifacts.get("metrics", None)
        if metrics is None:
            try:
                nw = np.load(f"results/plots/{t}_dqn_networth.npy")
                metrics = evaluate_equity_curve(nw)
            except Exception:
                metrics = {"total_return": None, "annualized_return": None, "sharpe": None, "max_drawdown": None, "num_trades": None}
        agg.append({"ticker": t, "algo": "DQN", "total_return": metrics.get("total_return"), "annualized_return": metrics.get("annualized_return"), "sharpe": metrics.get("sharpe"), "max_drawdown": metrics.get("max_drawdown")})
    df_agg = pd.DataFrame(agg)
    os.makedirs("results/plots", exist_ok=True)
    df_agg.to_csv("results/plots/aggregated_metrics_dqn.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", help="tickers to process")
    parser.add_argument("--timesteps", type=int, default=50000, help="per-ticker training timesteps")
    parser.add_argument("--tensorboard", type=str, default=None)
    args = parser.parse_args()
    main(args)

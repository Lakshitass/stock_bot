# src_sac/train_multi_sac.py
import os
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src_sac.multi_env import MultiStockTradingEnv

def train(save_path, timesteps=200000, window=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Load all training CSVs from data/
    tickers = ["AAPL", "TSLA", "JPM", "GS", "XOM"]
    price_dfs = []
    for t in tickers:
        df = pd.read_csv(f"data/{t}_train.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        price_dfs.append(df["Close"].rename(t))
    price_df = pd.concat(price_dfs, axis=1)

    env = DummyVecEnv([lambda: Monitor(MultiStockTradingEnv(price_df, window=window))])
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=200000, batch_size=256, tau=0.005, gamma=0.99)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    env.close()
    return save_path

import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from src.multi_env import MultiStockTradingEnv

def train(save_path, ticker_files_map, timesteps=200000, window=10, tensorboard_log=None, seed=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dfs = []
    names = []
    for t, path in ticker_files_map.items():
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        dfs.append(df["Close"].rename(t).reset_index(drop=True))
        names.append(t)
    price_df = pd.concat(dfs, axis=1)
    env = MultiStockTradingEnv(price_df, window=window)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log, seed=seed)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    vec_env.close()
    return save_path

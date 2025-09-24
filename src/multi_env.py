import numpy as np
import pandas as pd
from gymnasium import Env, spaces

class MultiStockTradingEnv(Env):
    def __init__(self, price_df: pd.DataFrame, corr_df=None, pca_df=None, window=10, initial_balance=10000.0):
        self.price_df = price_df.reset_index(drop=True) if isinstance(price_df, pd.DataFrame) else price_df
        self.n_steps = len(self.price_df)
        self.n_assets = self.price_df.shape[1]
        self.window = int(window)
        self.initial_balance = float(initial_balance)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets + 4,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.balance = float(self.initial_balance)
        self.alloc = np.zeros(self.n_assets, dtype=np.float32)
        self.current_step = 0
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self._prev_net = self.net_worth
        obs = self._get_obs()
        info = {"net_worth": float(self.net_worth), "step": int(self.current_step)}
        return obs, info

    def _get_obs(self):
        prices = self.price_df.iloc[self.current_step].values.astype(np.float32)
        obs = np.concatenate([prices, np.array([self.balance, self.net_worth, self.max_net_worth, float(self.current_step)], dtype=np.float32)])
        return obs

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).ravel(), 0.0, 1.0)
        prices = self.price_df.iloc[self.current_step].values.astype(np.float32)
        alloc_change = action - self.alloc
        traded_value = float(np.abs(alloc_change).sum() * max(1.0, self.net_worth))
        self.alloc = action
        self.current_step += 1
        terminated = bool(self.current_step >= self.n_steps - 1)
        truncated = False
        next_prices = self.price_df.iloc[min(self.current_step, self.n_steps-1)].values.astype(np.float32)
        portfolio_value = float((self.alloc * next_prices).sum())
        self.net_worth = portfolio_value
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        reward = float(self.net_worth - self._prev_net)
        self._prev_net = self.net_worth
        obs = self._get_obs()
        info = {"net_worth": float(self.net_worth), "price": next_prices.tolist(), "traded_value": float(traded_value), "step": int(self.current_step)}
        return obs, float(reward), bool(terminated), bool(truncated), info

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

class TradingEnvDiscrete(Env):
    def __init__(self, price_df: pd.DataFrame, initial_balance=10000.0):
        self.df = price_df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = float(initial_balance)
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self._prev_net = self.net_worth
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self._prev_net = self.net_worth
        obs = self._get_obs()
        info = {"net_worth": self.net_worth, "step": int(self.current_step)}
        return obs, info

    def _get_obs(self):
        price = float(self.df.loc[self.current_step, "Close"])
        return np.array([self.balance, float(self.shares_held), price, self.net_worth, self.max_net_worth], dtype=np.float32)

    def step(self, action):
        price = float(self.df.loc[self.current_step, "Close"])
        traded_value = 0.0
        if int(action) == 1:
            shares_bought = int(self.balance // price)
            traded_value = shares_bought * price
            self.balance -= traded_value
            self.shares_held += shares_bought
        elif int(action) == 2:
            traded_value = self.shares_held * price
            self.balance += traded_value
            self.shares_held = 0
        self.current_step += 1
        next_price = float(self.df.loc[min(self.current_step, self.n_steps - 1), "Close"])
        self.net_worth = self.balance + self.shares_held * next_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        reward = float(self.net_worth - self._prev_net)
        self._prev_net = self.net_worth
        terminated = bool(self.current_step >= self.n_steps - 1)
        truncated = False
        obs = self._get_obs()
        info = {"net_worth": float(self.net_worth), "price": float(next_price), "traded_value": float(traded_value), "step": int(self.current_step)}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        print(f"Step {self.current_step} NetWorth {self.net_worth:.2f} Cash {self.balance:.2f} Shares {self.shares_held}")

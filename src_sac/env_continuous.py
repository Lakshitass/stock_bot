import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnvContinuous(gym.Env):
    """
    Single-stock continuous allocation environment.
    Action: scalar in [0,1] target allocation fraction.
    Observation: last `window` normalized close prices (shape=(window,))
    Returns gymnasium style: obs, reward, terminated, truncated, info
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path, window=10, initial_balance=10000.0, transaction_cost=0.001):
        super().__init__()
        self.df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        self.window = int(window)
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window,), dtype=np.float32)

        self._reset_internal_state()

    def _reset_internal_state(self):
        self.current_step = 0
        self.cash = float(self.initial_balance)
        self.holdings = 0.0
        self.net_worth = float(self.initial_balance)
        self._prev_net_worth = float(self.initial_balance)

    def _get_obs(self):
        idx = min(self.current_step, len(self.df)-1)
        start = max(0, idx - self.window + 1)
        prices = self.df['Close'].values[start:idx+1].astype(np.float32)
        pad = self.window - len(prices)
        if pad > 0:
            prices = np.concatenate([np.ones(pad, dtype=np.float32) * prices[0], prices])
        last = prices[-1] if prices[-1] != 0 else 1.0
        return (prices / last - 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random as _random
            _random.seed(seed)
            np.random.seed(seed)
        self._reset_internal_state()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float32).ravel()
        target_alloc = float(action[0])

        price = float(self.df.loc[min(self.current_step, len(self.df)-1), "Close"])
        current_value = self.cash + self.holdings * price

        desired_value = target_alloc * current_value
        desired_shares = desired_value / (price + 1e-9)

        delta_shares = desired_shares - self.holdings
        traded_value = abs(delta_shares) * price
        cost = self.transaction_cost * traded_value

        # Execute
        self.holdings = self.holdings + delta_shares
        self.cash = self.cash - delta_shares * price - cost

        # advance
        self.current_step += 1
        idx = min(self.current_step, len(self.df)-1)
        next_price = float(self.df.loc[idx, "Close"])

        self.net_worth = float(self.cash + self.holdings * next_price)
        reward = float((self.net_worth - self._prev_net_worth) / (abs(self._prev_net_worth) + 1e-9)) if abs(self._prev_net_worth) > 0 else 0.0
        self._prev_net_worth = self.net_worth

        terminated = self.current_step >= (len(self.df) - 1)
        truncated = False
        obs = self._get_obs()
        info = {
            "step": int(self.current_step),
            "net_worth": float(self.net_worth),
            "cash": float(self.cash),
            "holdings": float(self.holdings),
            "price": next_price,
            "traded_value": float(traded_value)
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        date = self.df.loc[min(self.current_step, len(self.df)-1), "Date"]
        print(f"Step: {self.current_step}, Date: {date}, Net Worth: {self.net_worth:.2f}")

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

class MultiStockTradingEnv(gym.Env):
    """
    Multi-stock continuous portfolio allocation environment (Gymnasium API).

    Key features / hyperparams exposed:
      - window: length of price window per stock used in observation & volatility
      - transaction_cost: fraction of traded_value charged per trade
      - vol_penalty_lambda: weight for volatility penalty
      - drawdown_lambda: weight for drawdown penalty
      - trade_penalty_coef: additive penalty proportional to traded_value/current_value
      - min_trade_pct: minimum traded_value/current_value to count as a 'real' trade (helps avoid microtrades)
      - reward_scale: scale applied to log-return part of reward

    Action: Box(shape=(n_stocks,)): raw allocations in [0,1]; sum <= 1 (cash allowed)
    Observation: flattened (window x n_stocks) normalized price windows + optional PCA features
    Reward: reward = reward_log_scaled - vol_penalty - drawdown_penalty - trade_penalty
            where reward_log_scaled = reward_scale * log(net_worth / prev_net_worth)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        corr_df: Optional[pd.DataFrame] = None,
        pca_df: Optional[pd.DataFrame] = None,
        window: int = 10,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        vol_penalty_lambda: float = 0.1,
        drawdown_lambda: float = 0.5,
        trade_penalty_coef: float = 1.0,
        min_trade_pct: float = 0.001,
        reward_scale: float = 100.0,
    ):
        super().__init__()

        # data
        # price_df expected: index = dates, columns = tickers, values = close prices
        self.price_df = price_df.copy().reset_index(drop=True)
        self.dates = price_df.index if isinstance(price_df.index, pd.DatetimeIndex) else None
        self.corr_df = corr_df
        self.pca_df = pca_df

        # tickers / dims
        self.tickers = list(price_df.columns)
        self.n = len(self.tickers)
        self.window = int(window)

        # accounting / hyperparams
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.vol_penalty_lambda = float(vol_penalty_lambda)
        self.drawdown_lambda = float(drawdown_lambda)
        self.trade_penalty_coef = float(trade_penalty_coef)
        self.min_trade_pct = float(min_trade_pct)
        self.reward_scale = float(reward_scale)

        # action & observation spaces
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

        obs_len = self.window * self.n
        if self.pca_df is not None:
            obs_len += self.pca_df.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # internal state (initialized in _reset_internal_state)
        self._reset_internal_state()

    def _reset_internal_state(self):
        # start after window so observation is full
        self.current_step = self.window - 1
        self.cash = float(self.initial_balance)
        self.holdings = np.zeros(self.n, dtype=float)  # shares per ticker
        self.net_worth = float(self.initial_balance)
        self._prev_net_worth = float(self.initial_balance)
        self._peak_net_worth = float(self.initial_balance)
        self.history_alloc = []
        self.history_networth = [self.net_worth]
        self.total_traded_value = 0.0
        self.step_count = 0

    def seed(self, seed=None):
        import random as _random
        self._seed = seed
        _random.seed(seed)
        np.random.seed(seed if seed is not None else None)
        return [seed]

    def reset(self, seed=None, options=None):
        # gymnasium-style
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._reset_internal_state()
        obs = self._get_obs()
        return obs.astype(np.float32), {}

    def _get_obs(self):
        idx = int(self.current_step)
        start = idx - self.window + 1
        window_prices = self.price_df.iloc[start:idx+1].values.astype(float)  # (window, n)
        # normalize each column by last price in the window
        last = window_prices[-1, :].copy()
        last[last == 0] = 1.0
        norm = (window_prices / (last + 1e-12)) - 1.0
        flat = norm.flatten().astype(np.float32)
        if self.pca_df is not None:
            pca_vec = self.pca_df.iloc[idx].astype(np.float32).values
            out = np.concatenate([flat, pca_vec])
        else:
            out = flat
        return out.astype(np.float32)

    def _get_portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + (self.holdings * prices).sum())

    def step(self, action):
        """
        action: raw allocations (0..1). We normalize so sum_alloc = min(1, sum(action)); cash remainder
        Returns (obs, reward, terminated, truncated, info) per Gymnasium API.
        """
        # ensure action shape and clip
        action = np.clip(np.array(action, dtype=float).ravel(), 0.0, 1.0)
        total = float(np.sum(action))
        alloc = action.copy()
        if total > 1.0:
            alloc = action / (total + 1e-12)

        # current prices at current_step
        price = self.price_df.iloc[self.current_step].values.astype(float)
        price[price == 0] = 1e-9

        current_value = self._get_portfolio_value(price)

        # compute target shares given allocation
        target_values = alloc * current_value
        target_shares = target_values / (price + 1e-12)

        delta_shares = target_shares - self.holdings
        traded_values = np.abs(delta_shares) * price
        traded_value_sum = float(traded_values.sum())

        # transaction cost
        cost = self.transaction_cost * traded_value_sum

        # apply trades (atomic)
        self.holdings = target_shares
        self.cash = current_value - (self.holdings * price).sum() - cost

        # update counters
        self.total_traded_value += traded_value_sum
        self.step_count += 1

        # advance step
        self.current_step += 1
        terminated = bool(self.current_step >= len(self.price_df) - 1)
        truncated = False

        # next prices for net worth calc
        next_price = self.price_df.iloc[min(self.current_step, len(self.price_df)-1)].values.astype(float)
        next_price[next_price == 0] = 1e-9
        self.net_worth = self._get_portfolio_value(next_price)

        # reward components
        # 1) scaled log-return
        reward_log = 0.0
        if self._prev_net_worth > 0:
            reward_log = float(np.log((self.net_worth + 1e-9) / (self._prev_net_worth + 1e-9)) * self.reward_scale)
        else:
            reward_log = float((self.net_worth - self._prev_net_worth) * self.reward_scale)

        # 2) volatility penalty: compute recent portfolio returns std
        start = max(0, self.current_step - self.window + 1)
        wp = self.price_df.iloc[start: min(self.current_step+1, len(self.price_df))].values.astype(float)
        if wp.shape[0] >= 2:
            rets = np.nan_to_num(wp[1:] / (wp[:-1] + 1e-12) - 1.0)  # shape (window-1, n)
            weight = alloc if alloc.sum() > 0 else np.ones(self.n) / float(self.n)
            port_rets = np.dot(rets, weight)
            vol = float(np.std(port_rets))
        else:
            vol = 0.0
        vol_penalty = float(self.vol_penalty_lambda * vol * self.reward_scale)

        # 3) drawdown penalty: compute peak-to-current drawdown
        if self.net_worth > self._peak_net_worth:
            self._peak_net_worth = self.net_worth
        drawdown = 0.0
        if self._peak_net_worth > 0:
            drawdown = float((self._peak_net_worth - self.net_worth) / (self._peak_net_worth + 1e-12))
        drawdown_penalty = float(self.drawdown_lambda * drawdown * self.reward_scale)

        # 4) trade penalty: penalize frequent tiny trades (scaled)
        trade_penalty = 0.0
        trade_frac = 0.0
        if current_value > 0:
            trade_frac = traded_value_sum / (current_value + 1e-12)
            # only penalize trades larger than min_trade_pct
            if trade_frac >= self.min_trade_pct:
                trade_penalty = float(self.trade_penalty_coef * trade_frac * self.reward_scale)
            else:
                # minor cost but discourage microtrades slightly
                trade_penalty = float((self.trade_penalty_coef * 0.2) * trade_frac * self.reward_scale)

        # compose final reward
        reward = float(reward_log - vol_penalty - drawdown_penalty - trade_penalty)

        # update prev_net_worth
        self._prev_net_worth = self.net_worth

        # record history
        self.history_alloc.append(alloc.tolist())
        self.history_networth.append(self.net_worth)

        info = {
            "step": int(self.current_step),
            "net_worth": float(self.net_worth),
            "cash": float(self.cash),
            "holdings": self.holdings.tolist(),
            "price": next_price.tolist(),
            "traded_value": float(traded_value_sum),
            "allocation": alloc.tolist(),
            "volatility": float(vol),
            "vol_penalty": float(vol_penalty),
            "drawdown": float(drawdown),
            "drawdown_penalty": float(drawdown_penalty),
            "trade_frac": float(trade_frac),
            "trade_penalty": float(trade_penalty),
        }

        obs = self._get_obs()
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self, mode="human"):
        date_str = ""
        try:
            if self.dates is not None:
                date_str = f" Date:{self.dates[self.current_step]}"
        except Exception:
            date_str = ""
        print(f"Step {self.current_step}{date_str} NetWorth {self.net_worth:.2f} Cash {self.cash:.2f} Holdings {np.round(self.holdings,3)}")

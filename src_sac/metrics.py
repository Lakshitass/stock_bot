"""
Performance metrics and small plotting helpers for equity series.

Provides:
 - total_return(equity)
 - annualized_return(equity, trading_days=252)
 - sharpe_ratio(daily_returns, risk_free=0.0)
 - max_drawdown(equity)
 - evaluate_equity_curve(equity_series, trading_days=252)
 - plot_drawdown(equity_series, out_path=None, figsize=(10,4))
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def total_return(equity: pd.Series) -> float:
    """Total percentage return over the period (as decimal)."""
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

def annualized_return(equity: pd.Series, trading_days: int = 252) -> float:
    """Compound annual growth rate based on trading_days per year."""
    if len(equity) < 2:
        return 0.0
    period_years = max(1.0, len(equity) / trading_days)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / period_years) - 1.0)

def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio computed from daily returns (not log returns)."""
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - (risk_free / 252.0)
    denom = excess.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(excess.mean() / denom * np.sqrt(252.0))

def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (as positive decimal, e.g. 0.2 = 20% drawdown)."""
    if len(equity) < 2:
        return 0.0
    cum_max = equity.cummax()
    drawdown = (cum_max - equity) / cum_max
    return float(drawdown.max())

def evaluate_equity_curve(equity: pd.Series, trading_days: int = 252) -> Dict:
    """
    Given an equity series (pandas Series indexed by date or integer),
    compute a set of performance metrics and return as dict.

    Returned dict keys:
      - total_return
      - annualized_return
      - sharpe
      - max_drawdown
      - num_trades (will be None by default; it's up to caller to supply)
    """
    # ensure numeric pandas Series
    eq = pd.Series(equity).dropna().astype(float)
    # daily returns for Sharpe: percent changes (not log)
    daily = eq.pct_change().dropna()
    metrics = {}
    metrics["total_return"] = total_return(eq)
    metrics["annualized_return"] = annualized_return(eq, trading_days=trading_days)
    metrics["sharpe"] = sharpe_ratio(daily, risk_free=0.0)
    metrics["max_drawdown"] = max_drawdown(eq)
    # placeholder: callers can update num_trades after reading trades CSV
    metrics["num_trades"] = None
    return metrics

def plot_drawdown(equity: pd.Series, out_path: Optional[str] = None, figsize=(10,4)):
    """
    Plot drawdown curve (drawdown vs time). If out_path given, save to file,
    otherwise show with plt.show().
    """
    eq = pd.Series(equity).dropna().astype(float)
    if eq.empty:
        raise ValueError("Empty equity series passed to plot_drawdown")

    cum_max = eq.cummax()
    drawdown = (cum_max - eq) / cum_max

    plt.figure(figsize=figsize)
    plt.fill_between(range(len(drawdown)), -drawdown, color="tab:red", alpha=0.4)
    plt.plot(-drawdown, color="tab:red", linewidth=1.2)
    plt.title("Drawdown")
    plt.xlabel("Time step")
    plt.ylabel("Drawdown (fraction)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()

# backward-compatible exports (some files expected evaluate_equity_curve name)
__all__ = [
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "evaluate_equity_curve",
    "plot_drawdown",
]

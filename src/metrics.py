import numpy as np
import pandas as pd
def total_return(equity):
    eq = pd.Series(equity)
    if len(eq) == 0:
        return 0.0
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)

def annualized_return(equity, periods_per_year=252):
    eq = pd.Series(equity)
    if len(eq) < 2:
        return 0.0
    total = eq.iloc[-1] / eq.iloc[0]
    years = len(eq) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)

def daily_returns(equity):
    eq = pd.Series(equity)
    return eq.pct_change().dropna()

def sharpe_ratio(equity, periods_per_year=252):
    dr = daily_returns(equity)
    if dr.empty:
        return 0.0
    mean = dr.mean()
    std = dr.std()
    if std == 0:
        return 0.0
    return float((mean * np.sqrt(periods_per_year)) / std)

def max_drawdown(equity):
    eq = pd.Series(equity)
    if eq.empty:
        return 0.0
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    return float(drawdown.min())

def evaluate_equity_curve(equity):
    tr = total_return(equity)
    ar = annualized_return(equity)
    dr = daily_returns(equity)
    sr = sharpe_ratio(equity)
    md = max_drawdown(equity)
    return {
        "total_return": tr,
        "total_return_str": f"{tr:.2%}" if tr is not None else None,
        "annualized_return": ar,
        "annualized_return_str": f"{ar:.2%}" if ar is not None else None,
        "sharpe": sr,
        "sharpe_str": f"{sr:.2f}" if sr is not None else None,
        "max_drawdown": md,
        "max_drawdown_str": f"{md:.2%}" if md is not None else None,
        "num_trades": None
    }

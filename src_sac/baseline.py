import pandas as pd
import matplotlib.pyplot as plt
from src_sac.metrics import evaluate_equity_curve, plot_drawdown

def buy_and_hold(filepath, split_date="2019-01-01", initial_cash=10000):
    """
    Simple Buy & Hold baseline strategy:
    - Buy stock at split_date with all initial cash.
    - Hold until end of test period.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"]).sort_values("Date")

    # Train/Test split
    train = df[df["Date"] < split_date].reset_index(drop=True)
    test = df[df["Date"] >= split_date].reset_index(drop=True)

    if test.empty:
        raise ValueError(f"No test data found after {split_date}")

    # Buy at first test day
    first_price = test.iloc[0]["Close"]
    shares = initial_cash / first_price

    # Portfolio value over time
    test["Portfolio"] = shares * test["Close"]

    # ---- Metrics ----
    metrics = evaluate_equity_curve(test["Portfolio"])
    print("\n=== Buy & Hold Metrics ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

    # ---- Plots ----
    plt.figure(figsize=(10, 5))
    plt.plot(test["Date"], test["Portfolio"], label="Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Buy & Hold Baseline Strategy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: Drawdown plot
    # plot_drawdown accepts (equity_series, out_path=None, figsize=(10,4))
    plot_drawdown(test["Portfolio"])

    return test, metrics


if __name__ == "__main__":
    df, metrics = buy_and_hold("data/AAPL.csv")

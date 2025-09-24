import pandas as pd
from pathlib import Path
from metrics import total_return, annualized_return, sharpe_ratio, max_drawdown
from plot_trades_utils import plot_equity_curve, plot_candlestick_with_trades

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STOCKS   = ["AAPL", "TSLA", "JPM", "GS", "XOM"]

def evaluate_one(csv_path: Path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    equity = df.set_index("Date")["Close"]
    daily_ret = equity.pct_change().dropna()
    return equity, {
        "Total Return"     : f"{total_return(equity):.2%}",
        "Annualized Return": f"{annualized_return(equity):.2%}",
        "Sharpe Ratio"     : f"{sharpe_ratio(daily_ret):.2f}",
        "Max Drawdown"     : f"{max_drawdown(equity):.2%}"
    }

def main():
    print("=== Evaluation of Close Prices ===")
    all_equities = {}

    for stock in STOCKS:
        csv_file = DATA_DIR / f"{stock}.csv"
        if not csv_file.exists():
            print(f"{stock}: CSV not found in data/")
            continue

        equity, metrics = evaluate_one(csv_file)
        all_equities[stock] = equity
        print(f"\n{stock}")
        for k, v in metrics.items():
            print(f"  {k:<18}: {v}")

    if all_equities:
        plot_equity_curve(all_equities)

    for stock, equity in all_equities.items():
        trades_file = DATA_DIR / f"{stock}_trades.csv"
        if trades_file.exists():
            price_df = pd.read_csv(DATA_DIR / f"{stock}.csv",
                                   parse_dates=["Date"], index_col="Date")
            trades = pd.read_csv(trades_file, parse_dates=["Date"])
            print(f"\nShowing candlestick with trades for {stock}")
            plot_candlestick_with_trades(price_df, trades)

if __name__ == "__main__":
    main()

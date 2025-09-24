import os
import argparse
from pathlib import Path
import pandas as pd
from src.train_multi_sac import train
from src.visualize_stock import run_and_record
def main(args):
    tickers = args.tickers or ["AAPL", "TSLA", "JPM", "GS", "XOM"]
    ticker_files = {t: f"data/{t}.csv" for t in tickers}
    pretrain_save = args.pretrain_save
    if not Path(pretrain_save).exists():
        train(pretrain_save, ticker_files, timesteps=args.pretrain_timesteps, window=args.window, tensorboard_log=args.tensorboard)
    agg = []
    for idx, t in enumerate(tickers):
        test_csv = f"data/{t}_test.csv" if Path(f"data/{t}_test.csv").exists() else f"data/{t}.csv"
        out_prefix = f"results/plots/{t}_pretrain"
        art = run_and_record(pretrain_save, test_csv, out_prefix=out_prefix, model_type="sac", trade_threshold=args.trade_threshold)
        metrics = art.get("metrics", {})
        agg.append({"ticker": t, "phase": "pretrain", "total_return": metrics.get("total_return"), "annualized_return": metrics.get("annualized_return"), "sharpe": metrics.get("sharpe"), "max_drawdown": metrics.get("max_drawdown")})
        finetune_save = f"results/models/{t}_finetuned_from_pretrain.zip"
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv
            from src.env_continuous import TradingEnvContinuous
            model = SAC.load(pretrain_save)
            env_single = TradingEnvContinuous(pd.read_csv(f"data/{t}.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True))
            vec = DummyVecEnv([lambda: env_single])
            model.set_env(vec)
            model.learn(total_timesteps=args.finetune_timesteps)
            model.save(finetune_save)
            vec.close()
            out_prefix_ft = f"results/plots/{t}_finetuned"
            art_ft = run_and_record(finetune_save, test_csv, out_prefix=out_prefix_ft, model_type="sac", trade_threshold=args.trade_threshold)
            metrics_ft = art_ft.get("metrics", {})
            agg.append({"ticker": t, "phase": "finetune", "total_return": metrics_ft.get("total_return"), "annualized_return": metrics_ft.get("annualized_return"), "sharpe": metrics_ft.get("sharpe"), "max_drawdown": metrics_ft.get("max_drawdown")})
        except Exception:
            pass
    df = pd.DataFrame(agg)
    os.makedirs("results/plots", exist_ok=True)
    df.to_csv("results/plots/aggregated_metrics_pretrain_finetune.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--pretrain-timesteps", type=int, default=200000)
    parser.add_argument("--finetune-timesteps", type=int, default=5000)
    parser.add_argument("--pretrain-save", type=str, default="results/models/multi_sac_pretrain.zip")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--trade-threshold", type=float, default=0.01)
    parser.add_argument("--tensorboard", type=str, default=None)
    args = parser.parse_args()
    main(args)

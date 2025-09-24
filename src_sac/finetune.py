import argparse, os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from src_sac.data_utils import build_multi_feature_set
from src_sac.multi_env import MultiStockTradingEnv

def finetune(pretrained_model, stock_csv, save_path, timesteps=10000, window=10, corr_window=20):
    # For fine-tuning, build a price_df with only the target stock (still multi-env expects multiple columns).
    ticker_files = { "TGT": stock_csv }
    price_df, corr_df, pca_df = build_multi_feature_set(ticker_files, window=window, corr_window=corr_window, n_pca=1)
    env = DummyVecEnv([lambda: MultiStockTradingEnv(price_df=price_df, corr_df=None, pca_df=pca_df, window=window)])
    model = SAC.load(pretrained_model, env=env)
    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print("Saved fine-tuned model to", save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--stock", required=True)   # path to CSV for single stock
    parser.add_argument("--save", required=True)
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()
    finetune(args.pretrained, args.stock, args.save, timesteps=args.timesteps)

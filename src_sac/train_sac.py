import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src_sac.env_continuous import TradingEnvContinuous

def train(env_path, timesteps=20000, save_path="results/models/sac_model.zip", window=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    def make_env():
        env = TradingEnvContinuous(env_path, window=window)
        return Monitor(env)
    vec = DummyVecEnv([make_env])
    model = SAC("MlpPolicy", vec, verbose=1, tensorboard_log="runs/sac_single")
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    vec.close()
    print("Saved:", save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="data/AAPL_train.csv")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--save", type=str, default="results/models/sac_model.zip")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    train(args.env, timesteps=args.timesteps, save_path=args.save, window=args.window)

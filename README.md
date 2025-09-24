
# Stock_bot

A reinforcement learning + machine learning based trading bot project.  
This repo contains experiments with **Random Forest**, **DQN**, **PPO**, and **SAC** on multiple stock market datasets.

---

## Project Structure
```bash
stock_bot/
├── data/                
│   ├── AAPL.csv, GS.csv, JPM.csv, TSLA.csv, XOM.csv
│   ├── *_train.csv, *_test.csv
│
├── random forest/       # Random forest baseline experiments
│   ├── AAPL_plot.png, GS_plot.png, …
│
├── results/             
│   ├── graph/           # Visualization outputs
│   ├── metrics/         # Metrics files
│   ├── models/          # Trained models (.zip)
│   ├── plots/           # Comparison plots
│   ├── plots_sac/       # SAC-specific plots
│
├── runs/                
│   └── multi_sac/
│
├── src/                 # Source code
│   ├── data_utils.py
│   ├── env_continuous.py
│   ├── env_discrete.py
│   ├── metrics.py
│   ├── multi_env.py
│   ├── run_all_dqn.py
│   ├── run_all_ppo.py
│   ├── run_all.py
│   ├── train_multi_sac.py
│   └── visualize_stock.py
│
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Ignored files (venv, data, results, etc.)

---

## Quickstart

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Lakshitass/stock_bot.git
cd stock_bot

2️⃣ Set up environment

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

3️⃣ Run experiments

# Train and evaluate DQN models
python src/run_all_dqn.py

# Train and evaluate PPO models
python src/run_all_ppo.py

# Train SAC agent on multiple environments
python src/train_multi_sac.py

# Run all experiments (DQN + PPO + SAC)
python src/run_all.py

4️⃣ Visualize results

python src/visualize_stock.py


⸻

Models Implemented
	•	 Random Forest (baseline)
	•	 Deep Q-Learning (DQN)
	•	 Proximal Policy Optimization (PPO)
	•	 Soft Actor-Critic (SAC)
	•	 Soft Actor-Critic (SAC)-Finetuned

⸻

Requirements

Main dependencies (see requirements.txt for full list):
	•	Python 3.9+
	•	numpy
	•	pandas
	•	matplotlib
	•	scikit-learn
	•	gym
	•	stable-baselines3
	•	torch



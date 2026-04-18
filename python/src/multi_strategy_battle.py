import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- PATH SETUP ---
def get_resource_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    paths = [os.path.join(root_dir, relative_path), os.path.join(os.path.dirname(current_dir), relative_path)]
    for p in paths:
        if os.path.exists(p): return p
    return None

MODEL_PATH = get_resource_path("models/model_scalper_v1.pkl")
DATA_PATH = get_resource_path("python/data_vault/processed_variants/dataset_scalping.csv")
SPREAD = 0.00007

def run_strategy_tournament():
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    # --- STRATEGY 1: ORIGINAL ML (MEAN REVERSION) ---
    model = joblib.load(MODEL_PATH)
    features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close']]
    data['prob_up'] = model.predict_proba(data[features])[:, 1]
    data['sig_ml'] = np.where(data['prob_up'] > 0.55, 1, np.where(data['prob_up'] < 0.45, -1, 0))

    # --- STRATEGY 2: THE "BOLT" (MOMENTUM BREAKOUT) ---
    # Buy if price is at a 20-period high, Sell if at 20-period low.
    data['hi'] = data['close'].rolling(20).max().shift(1)
    data['lo'] = data['close'].rolling(20).min().shift(1)
    data['sig_bolt'] = np.where(data['close'] > data['hi'], 1, 
                       np.where(data['close'] < data['lo'], -1, 0))

    # --- STRATEGY 3: THE "CROSS" (EMA CROSSOVER) ---
    # Classic Trend Following: 9-period EMA crosses 21-period EMA
    data['ema9'] = data['close'].ewm(span=9).mean()
    data['ema21'] = data['close'].ewm(span=21).mean()
    data['sig_cross'] = np.where(data['ema9'] > data['ema21'], 1, -1)

    # --- CALCULATE P&L ---
    for s in ['ml', 'bolt', 'cross']:
        sig_col = f'sig_{s}'
        ret_col = f'ret_{s}'
        data[ret_col] = data[sig_col] * data['returns']
        data.loc[data[sig_col] != 0, ret_col] -= SPREAD
        data[f'cum_{s}'] = data[ret_col].cumsum()

    # --- THE ANALYSIS GRAPH ---
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    
    plt.plot(data['time'], data['cum_ml'], label='Original ML (Mean Reversion)', color='cyan', linewidth=2)
    plt.plot(data['time'], data['cum_bolt'], label='Bolt (20-bar Breakout)', color='#ff00ff', linewidth=2)
    plt.plot(data['time'], data['cum_cross'], label='Cross (EMA Trend)', color='yellow', alpha=0.7)

    # Find where ML peaks to highlight the "Shift"
    peak_time = data.loc[data['cum_ml'].idxmax(), 'time']
    plt.axvline(peak_time, color='red', linestyle='--', label='Regime Shift Point')

    plt.title("Tournament: Identifying the Volatility Partner", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.1)
    plt.show()

if __name__ == "__main__":
    run_strategy_tournament()
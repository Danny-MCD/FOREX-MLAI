import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- PATH SETUP ---
def get_resource_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(python_dir)
    paths = [os.path.join(root_dir, relative_path), os.path.join(python_dir, relative_path)]
    for p in paths:
        if os.path.exists(p): return p
    return None

MODEL_PATH = get_resource_path("models/model_scalper_v1.pkl")
DATA_PATH = get_resource_path("python/data_vault/processed_variants/dataset_scalping.csv")

# --- SETTINGS ---
SPREAD = 0.00007
VOL_THRESHOLD = 0.0003  # The point where we switch from "Quiet" to "Violent" mode

def run_dual_regime_test():
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    
    # 1. Feature Engineering for New Strategy
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)
    # Volatility (Standard Deviation of last 20 mins)
    data['volatility'] = data['close'].pct_change().rolling(20).std()
    
    # 2. Strategy A: The ML Scalper (Quiet Mode)
    model = joblib.load(MODEL_PATH)
    try:
        features = model.get_booster().feature_names
    except:
        features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'volatility']]
    
    data['prob_up'] = model.predict_proba(data[features])[:, 1]
    data['sig_A'] = np.where(data['prob_up'] > 0.60, 1, np.where(data['prob_up'] < 0.40, -1, 0))

    # 3. Strategy B: Breakout Scalper (Volatile Mode)
    # Logic: Buy if we break the high of the last 10 mins, Sell if we break the low
    data['upper_band'] = data['close'].rolling(10).max()
    data['lower_band'] = data['close'].rolling(10).min()
    data['sig_B'] = np.where(data['close'] >= data['upper_band'].shift(1), 1, 
                    np.where(data['close'] <= data['lower_band'].shift(1), -1, 0))

    # 4. THE REGIME SWITCH (The "Brain")
    # If Volatility is Low -> Use ML Scalper
    # If Volatility is High -> Use Breakout Scalper
    data['final_signal'] = np.where(data['volatility'] < VOL_THRESHOLD, data['sig_A'], data['sig_B'])

    # 5. Calculate Results
    data['base_ret'] = data['sig_A'] * data['returns'] # Original ML Only
    data.loc[data['sig_A'] != 0, 'base_ret'] -= SPREAD
    
    data['dual_ret'] = data['final_signal'] * data['returns'] # Dual Regime
    data.loc[data['final_signal'] != 0, 'dual_ret'] -= SPREAD

    data['base_equity'] = data['base_ret'].cumsum()
    data['dual_equity'] = data['dual_ret'].cumsum()

    # 6. Plotting the "Rescue"
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    plt.plot(data['time'], data['base_equity'], label='Original ML Scalper', color='#ff4444', alpha=0.5)
    plt.plot(data['time'], data['dual_equity'], label='Dual-Regime System (Adaptive)', color='#00ff88', linewidth=2)
    
    # Highlight the high volatility periods
    plt.fill_between(data['time'], data['dual_equity'].min(), data['dual_equity'].max(), 
                     where=(data['volatility'] > VOL_THRESHOLD), color='yellow', alpha=0.1, label='High Vol Regime')

    plt.title("Dual-Regime Performance: ML Scalper + Breakout Adaptive System")
    plt.legend()
    plt.show()

    print(f"Final Dual P&L: {data['dual_equity'].iloc[-1]:.6f}")

if __name__ == "__main__":
    run_dual_regime_test()
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

def run_safety_backtest():
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    # 1. LOAD ML MODEL
    model = joblib.load(MODEL_PATH)
    features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close']]
    data['prob_up'] = model.predict_proba(data[features])[:, 1]
    
    # 2. THE VOLATILITY FILTER (The Secret Sauce)
    # We calculate the 14-period ATR (Average True Range)
    data['high_low'] = data['close'].rolling(5).max() - data['close'].rolling(5).min()
    data['vol_ma'] = data['high_low'].rolling(20).mean()
    
    # Logic: Only trade when volatility is in the "Sweet Spot" 
    # Not too quiet (no move), Not too crazy (news spikes)
    low_limit = data['vol_ma'].quantile(0.2)
    high_limit = data['vol_ma'].quantile(0.8)
    data['vol_ok'] = (data['vol_ma'] > low_limit) & (data['vol_ma'] < high_limit)

    # 3. APPLY SIGNALS
    threshold = 0.60
    data['raw_sig'] = np.where(data['prob_up'] > threshold, 1, 
                      np.where(data['prob_up'] < (1-threshold), -1, 0))
    
    # Filtered Signal: Only trade if Volatility is OK
    data['filtered_sig'] = np.where(data['vol_ok'], data['raw_sig'], 0)

    # 4. CALC EQUITY
    data['raw_ret'] = data['raw_sig'] * data['returns']
    data.loc[data['raw_sig'] != 0, 'raw_ret'] -= SPREAD
    
    data['filt_ret'] = data['filtered_sig'] * data['returns']
    data.loc[data['filtered_sig'] != 0, 'filt_ret'] -= SPREAD

    data['raw_equity'] = data['raw_ret'].cumsum()
    data['filt_equity'] = data['filt_ret'].cumsum()

    # 5. PLOT
    plt.figure(figsize=(15, 7))
    plt.style.use('dark_background')
    plt.plot(data['time'], data['raw_equity'], label='Original ML (No Filter)', color='red', alpha=0.5)
    plt.plot(data['time'], data['filt_equity'], label='ML with Vol-Regime Filter', color='#00ff88', linewidth=2)
    plt.title("The 'Sweet Spot' Filter: Removing Market Chaos")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_safety_backtest()
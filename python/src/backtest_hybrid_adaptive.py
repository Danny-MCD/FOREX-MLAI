import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- SETUP PATHS ---
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

def run_hybrid_backtest():
    print("[*] Initializing Hybrid Adaptive Backtest...")
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    # 1. LOAD MODEL & GENERATE RAW SIGNALS
    model = joblib.load(MODEL_PATH)
    features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'hour', 'month_year']]
    data['prob_up'] = model.predict_proba(data[features])[:, 1]
    
    # Base Signal Logic
    data['sig_raw'] = np.where(data['prob_up'] > 0.60, 1, np.where(data['prob_up'] < 0.40, -1, 0))

    # 2. CALCULATE VOLATILITY FILTER
    data['hi_lo'] = data['close'].rolling(5).max() - data['close'].rolling(5).min()
    data['vol_ma'] = data['hi_lo'].rolling(20).mean()
    v_low, v_high = data['vol_ma'].quantile([0.2, 0.8])
    data['vol_ok'] = (data['vol_ma'] > v_low) & (data['vol_ma'] < v_high)

    # 3. DEFINE THE HYBRID REGIMES (Based on your Profit Map analysis)
    # Original ML Core (The 9 High Bars)
    original_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    # Filtered ML Buffer/Gaps
    filtered_hours = [10, 20] # Hours at start, end, and transition

    def apply_hybrid_logic(row):
        if row['hour'] in original_hours:
            return row['sig_raw']  # Use Aggressive Original
        elif row['hour'] in filtered_hours:
            return row['sig_raw'] if row['vol_ok'] else 0  # Use Defensive Filtered
        else:
            return 0  # Do not trade outside these windows

    data['hybrid_sig'] = data.apply(apply_hybrid_logic, axis=1)

    # 4. CALCULATE P&L
    # Original for comparison
    data['ret_orig'] = data['sig_raw'] * data['returns']
    data.loc[data['sig_raw'] != 0, 'ret_orig'] -= SPREAD
    
    # Hybrid Result
    data['ret_hybrid'] = data['hybrid_sig'] * data['returns']
    data.loc[data['hybrid_sig'] != 0, 'ret_hybrid'] -= SPREAD

    data['cum_orig'] = data['ret_orig'].cumsum()
    data['cum_hybrid'] = data['ret_hybrid'].cumsum()

    # 5. GRAPHING
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    
    plt.plot(data['time'], data['cum_orig'], label='Original ML (24/7)', color='red', alpha=0.4)
    plt.plot(data['time'], data['cum_hybrid'], label='Hybrid Adaptive System', color='#00ff88', linewidth=2.5)
    
    plt.fill_between(data['time'], data['cum_hybrid'].min(), data['cum_hybrid'].max(), 
                     where=data['hour'].isin(original_hours), color='cyan', alpha=0.05, label='Aggressive Zone')

    plt.title("Hybrid System: Performance over Full Year")
    plt.legend()
    plt.grid(alpha=0.1)
    plt.show()

    print(f"Final Hybrid P&L: {data['cum_hybrid'].iloc[-1]:.6f}")

if __name__ == "__main__":
    run_hybrid_backtest()
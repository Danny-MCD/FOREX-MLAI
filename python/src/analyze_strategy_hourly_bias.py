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
REPORTS_DIR = "hourly_analysis_results"

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

def analyze_hourly_bias():
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    model = joblib.load(MODEL_PATH)
    features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'hour']]
    
    # 1. Generate Signals
    data['prob_up'] = model.predict_proba(data[features])[:, 1]
    
    # Volatility Switch logic
    data['hi_lo'] = data['close'].rolling(5).max() - data['close'].rolling(5).min()
    data['vol_ma'] = data['hi_lo'].rolling(20).mean()
    vol_low, vol_high = data['vol_ma'].quantile([0.2, 0.8])
    data['vol_ok'] = (data['vol_ma'] > vol_low) & (data['vol_ma'] < vol_high)

    # Calculate returns for both
    data['sig_raw'] = np.where(data['prob_up'] > 0.60, 1, np.where(data['prob_up'] < 0.40, -1, 0))
    data['ret_raw'] = data['sig_raw'] * data['returns']
    data.loc[data['sig_raw'] != 0, 'ret_raw'] -= SPREAD
    
    data['sig_filt'] = np.where(data['vol_ok'], data['sig_raw'], 0)
    data['ret_filt'] = data['sig_filt'] * data['returns']
    data.loc[data['sig_filt'] != 0, 'ret_filt'] -= SPREAD

    # 2. GROUP BY HOUR
    hourly_raw = data.groupby('hour')['ret_raw'].sum()
    hourly_filt = data.groupby('hour')['ret_filt'].sum()

    # 3. PLOT COMPARISON
    plt.figure(figsize=(15, 7))
    plt.style.use('dark_background')
    
    x = np.arange(24)
    width = 0.35
    
    plt.bar(x - width/2, hourly_raw, width, label='Original ML P&L', color='#ff4444', alpha=0.7)
    plt.bar(x + width/2, hourly_filt, width, label='Filtered ML P&L', color='#00ff88', alpha=0.9)

    plt.title("Hourly Profit Map: Original vs. Filtered Strategy (Full Year)", fontsize=16)
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Cumulative Profit")
    plt.xticks(x)
    plt.axhline(0, color='white', linewidth=0.8)
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    
    plt.savefig(os.path.join(REPORTS_DIR, "annual_hourly_profit_map.png"))
    plt.show()

    # 4. PRINT BEST HOURS
    print("\n--- TOP PERFORMANCE WINDOWS ---")
    print(f"Original's Best Hour: {hourly_raw.idxmax()}:00 (P&L: {hourly_raw.max():.6f})")
    print(f"Filtered's Best Hour: {hourly_filt.idxmax()}:00 (P&L: {hourly_filt.max():.6f})")

if __name__ == "__main__":
    analyze_hourly_bias()
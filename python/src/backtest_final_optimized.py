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

# Create results directory if it doesn't exist
RESULTS_DIR = "backtest_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- STRATEGY SETTINGS ---
SPREAD = 0.00007
# Based on your chart: We keep Hours 6-17 (European/NY session)
# We strictly avoid Hours 21-23 and 0-5
START_HOUR = 6
END_HOUR = 17

def run_final_backtest():
    print(f"[*] Loading Model: {MODEL_PATH}")
    if not MODEL_PATH or not DATA_PATH:
        print("[-] Error: Missing files.")
        return

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    
    # 1. Setup Time and Returns
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    if 'returns' not in data.columns:
        data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    # 2. Get Features & Predict
    try:
        features = model.get_booster().feature_names
    except:
        features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'hour']]
    
    data['prob_up'] = model.predict_proba(data[features])[:, 1]

    # 3. BASELINE vs OPTIMIZED
    threshold = 0.55 # Your optimized threshold from the chart
    
    # Baseline: Trade 24/7
    data['base_sig'] = np.where(data['prob_up'] > threshold, 1, 
                       np.where(data['prob_up'] < (1 - threshold), -1, 0))
    data['base_ret'] = data['base_sig'] * data['returns']
    data.loc[data['base_sig'] != 0, 'base_ret'] -= SPREAD
    
    # Optimized: Only trade during Golden Hours
    data['opt_sig'] = np.where((data['base_sig'] != 0) & 
                               (data['hour'].between(START_HOUR, END_HOUR)), 
                               data['base_sig'], 0)
    data['opt_ret'] = data['opt_sig'] * data['returns']
    data.loc[data['opt_sig'] != 0, 'opt_ret'] -= SPREAD

    # Cumulative Returns
    data['base_equity'] = data['base_ret'].cumsum()
    data['opt_equity'] = data['opt_ret'].cumsum()

    # 4. Final Comparison Stats
    base_final = data['base_equity'].iloc[-1]
    opt_final = data['opt_equity'].iloc[-1]
    improvement = ((opt_final - base_final) / abs(base_final)) * 100 if base_final != 0 else 0

    print("\n" + "="*40)
    print(f"      STRATEGY JUDGEMENT")
    print("="*40)
    print(f"Baseline P&L (24/7):   {base_final:.6f}")
    print(f"Optimized P&L ({START_HOUR}-{END_HOUR}h): {opt_final:.6f}")
    print(f"Percent Improvement:   {improvement:.2f}%")
    print("="*40)

    # 5. Visual Picture
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    
    plt.plot(data['time'], data['base_equity'], label='Baseline (All Hours)', color='gray', alpha=0.6, linestyle='--')
    plt.plot(data['time'], data['opt_equity'], label=f'Optimized ({START_HOUR}:00-{END_HOUR}:00)', color='#00ff88', linewidth=2)
    
    plt.fill_between(data['time'], data['base_equity'], data['opt_equity'], 
                     where=(data['opt_equity'] > data['base_equity']), 
                     color='#00ff88', alpha=0.1, label='Profit Saved by Filtering')

    plt.title("Equity Curve: Impact of Time-of-Day Filtering", fontsize=15)
    plt.ylabel("Cumulative Units")
    plt.legend()
    plt.grid(alpha=0.1)
    
    image_name = os.path.join(RESULTS_DIR, "final_performance_comparison.png")
    plt.savefig(image_name)
    print(f"\n[+] Visual reference saved to: {image_name}")
    plt.show()

if __name__ == "__main__":
    run_final_backtest()
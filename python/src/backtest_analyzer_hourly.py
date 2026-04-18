import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns

# --- SMART PATH RESOLUTION ---
def get_resource_path(relative_path):
    # Get the directory where THIS script is (python/src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to 'python'
    python_dir = os.path.dirname(current_dir)
    # Go up to root 'FOREX-MLAI'
    root_dir = os.path.dirname(python_dir)
    
    paths_to_check = [
        os.path.join(root_dir, relative_path),       # Root/models/...
        os.path.join(python_dir, relative_path),     # Python/models/...
        os.path.join(current_dir, relative_path),    # Python/src/models/...
        os.path.abspath(relative_path)               # Absolute
    ]
    
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return None

# Attempt to locate files
MODEL_PATH = get_resource_path("models/model_scalper_v1.pkl")
DATA_PATH = get_resource_path("python/data_vault/processed_variants/dataset_scalping.csv")

# --- SETTINGS ---
SPREAD = 0.00007  

def run_hourly_analysis():
    print(f"[*] Checking Data: {DATA_PATH}")
    print(f"[*] Checking Model: {MODEL_PATH}")
    
    if not DATA_PATH:
        print("[-] ERROR: dataset_scalping.csv NOT FOUND. check python/data_vault/processed_variants/")
        return
    if not MODEL_PATH:
        print("[-] ERROR: model_scalper_v1.pkl NOT FOUND. Create a folder named 'models' in FOREX-MLAI and put your .pkl there.")
        return

    # 1. Load artifacts
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    
    # 2. FEATURE ALIGNMENT
    try:
        # Try to get features from XGBoost booster
        model_features = model.get_booster().feature_names
    except:
        try:
            # Try Scikit-learn attribute
            model_features = model.feature_names_in_
        except:
            # Manual Filter
            model_features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'is_London', 'is_NewYork']]
    
    # 3. TIME PROCESSING
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    
    # Ensure returns exist
    if 'returns' not in data.columns:
        data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    # 4. Generate Predictions
    X = data[model_features]
    probs = model.predict_proba(X)
    data['prob_up'] = probs[:, 1]
    
    # 5. Threshold Optimization
    best_threshold = 0.50
    max_profit = -np.inf
    for threshold in np.arange(0.55, 0.86, 0.05):
        temp_sig = np.where(data['prob_up'] > threshold, 1, 
                   np.where(data['prob_up'] < (1 - threshold), -1, 0))
        temp_ret = temp_sig * data['returns']
        temp_ret = np.where(temp_sig != 0, temp_ret - SPREAD, 0)
        if temp_ret.sum() > max_profit:
            max_profit = temp_ret.sum()
            best_threshold = threshold
    
    # 6. Apply Baseline (24/7 Trading)
    data['signal'] = np.where(data['prob_up'] > best_threshold, 1, 
                     np.where(data['prob_up'] < (1 - best_threshold), -1, 0))
    data['strat_ret'] = data['signal'] * data['returns']
    data.loc[data['signal'] != 0, 'strat_ret'] -= SPREAD
    data['equity'] = data['strat_ret'].cumsum()

    # 7. Hourly Stats
    hourly_perf = data.groupby('hour')['strat_ret'].sum()
    hourly_win_rate = data[data['signal'] != 0].groupby('hour').apply(
        lambda x: (x['strat_ret'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    )

    # 8. Plotting
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # P&L Bar Chart
    colors = ['#00ff88' if x > 0 else '#ff4444' for x in hourly_perf]
    ax1.bar(hourly_perf.index, hourly_perf.values, color=colors)
    ax1.set_title(f"Net Profit/Loss by Hour (Threshold: {best_threshold:.2f})")
    ax1.axhline(0, color='white', linewidth=0.5)
    
    # Win Rate Line Chart
    ax2.plot(hourly_win_rate.index, hourly_win_rate.values, color='#00f7ff', marker='o')
    ax2.axhline(50, color='yellow', linestyle='--', alpha=0.5, label='50% Win Rate')
    ax2.set_title("Win Rate % by Hour")
    ax2.set_ylim(0, 100)
    plt.xticks(range(0, 24))
    
    plt.tight_layout()
    plt.savefig("hourly_analysis.png")
    print("\n[+] Analysis Chart saved as 'hourly_analysis.png'")
    
    # 9. Comparison Logic
    baseline_pnl = data['equity'].iloc[-1]
    # Check if 'is_London' exists in your CSV (as seen in your screenshot)
    if 'is_London' in data.columns:
        london_pnl = data[data['is_London'] == 1]['strat_ret'].sum()
        print(f"\n" + "="*40)
        print(f"BASELINE (24/7) P&L:   {baseline_pnl:.6f}")
        print(f"LONDON ONLY P&L:       {london_pnl:.6f}")
        print(f"IMPROVEMENT:           {london_pnl - baseline_pnl:.6f}")
        print("="*40)

    plt.show()

if __name__ == "__main__":
    run_hourly_analysis()
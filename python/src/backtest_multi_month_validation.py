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
REPORTS_DIR = "monthly_reports"

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

def run_multi_month_validation():
    print("[*] Starting Multi-Month Stress Test...")
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    data['month_year'] = data['time'].dt.to_period('M')
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

    # Load Model and Prepare Features
    model = joblib.load(MODEL_PATH)
    features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'month_year', 'vol_ok', 'vol_ma']]
    
    # Calculate Base Probabilities and Volatility
    data['prob_up'] = model.predict_proba(data[features])[:, 1]
    data['high_low'] = data['close'].rolling(5).max() - data['close'].rolling(5).min()
    data['vol_ma'] = data['high_low'].rolling(20).mean()
    
    # Volatility Switch logic
    low_limit = data['vol_ma'].quantile(0.2)
    high_limit = data['vol_ma'].quantile(0.8)
    data['vol_ok'] = (data['vol_ma'] > low_limit) & (data['vol_ma'] < high_limit)

    summary_stats = []

    # Iterate through each month found in the data
    for month, group in data.groupby('month_year'):
        print(f"[*] Processing {month}...")
        
        # Original Strategy
        group = group.copy()
        group['sig_raw'] = np.where(group['prob_up'] > 0.60, 1, np.where(group['prob_up'] < 0.40, -1, 0))
        group['ret_raw'] = group['sig_raw'] * group['returns']
        group.loc[group['sig_raw'] != 0, 'ret_raw'] -= SPREAD
        
        # Filtered Strategy
        group['sig_filt'] = np.where(group['vol_ok'], group['sig_raw'], 0)
        group['ret_filt'] = group['sig_filt'] * group['returns']
        group.loc[group['sig_filt'] != 0, 'ret_filt'] -= SPREAD
        
        # Equity Curves
        raw_final = group['ret_raw'].sum()
        filt_final = group['ret_filt'].sum()
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')
        plt.plot(group['time'], group['ret_raw'].cumsum(), label='Original ML', color='red', alpha=0.6)
        plt.plot(group['time'], group['ret_filt'].cumsum(), label='Filtered ML', color='#00ff88', linewidth=2)
        plt.title(f"Performance Comparison: {month}")
        plt.legend()
        plt.grid(alpha=0.1)
        
        file_path = os.path.join(REPORTS_DIR, f"report_{month}.png")
        plt.savefig(file_path)
        plt.close()
        
        summary_stats.append({
            'Month': str(month),
            'Original_PnL': raw_final,
            'Filtered_PnL': filt_final,
            'Winner': 'Filtered' if filt_final > raw_final else 'Original'
        })

    # Save summary report
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(REPORTS_DIR, "validation_summary.csv"), index=False)
    print(f"\n[+] VALIDATION COMPLETE.")
    print(f"[+] Individual charts saved in: /{REPORTS_DIR}")
    print(summary_df)

if __name__ == "__main__":
    run_multi_month_validation()
    
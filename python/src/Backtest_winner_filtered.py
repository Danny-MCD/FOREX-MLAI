import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- ROBUST PATH RESOLUTION ---
def get_resource_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    paths_to_check = [
        os.path.join(root_dir, relative_path),
        os.path.join(os.path.dirname(current_dir), relative_path),
        os.path.abspath(relative_path)
    ]
    
    for p in paths_to_check:
        if p and os.path.exists(p):
            return p
    return None

MODEL_PATH = get_resource_path("models/model_scalper_v1.pkl")
DATA_PATH = get_resource_path("python/data_vault/processed_variants/dataset_scalping.csv")

# --- SETTINGS ---
SPREAD = 0.00007  # 0.7 pips spread penalty

def run_backtest():
    print(f"[*] Target Model: {MODEL_PATH}")
    
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"[-] ERROR: Model file not found.")
        return

    # 1. Load artifacts
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    
    # 2. RESOLVE RETURNS COLUMN (Fixes KeyError: 'returns')
    if 'returns' not in data.columns:
        if 'close' in data.columns:
            print("[!] 'returns' column missing. Calculating from 'close'...")
            data['returns'] = data['close'].pct_change().shift(-1).fillna(0)
        elif 'target' in data.columns:
            print("[!] Using 'target' as a proxy for returns direction...")
            data['returns'] = data['target'] * 0.0001 # Visual estimate
        else:
            print("[-] ERROR: No return or price data found to calculate P&L.")
            return

    # 3. FEATURE ALIGNMENT
    try:
        model_features = model.get_booster().feature_names
        print(f"[*] Model expects {len(model_features)} features.")
    except Exception:
        # Fallback if booster extraction fails
        model_features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'Unnamed: 0', 'close', 'prob_up', 'signal']]
    
    # Check if we have all required features
    missing = [f for f in model_features if f not in data.columns]
    if missing:
        print(f"[-] ERROR: Data is missing features required by model: {missing}")
        return

    # Subset data to ONLY what the model expects
    X = data[model_features]
    
    # 4. Generate Probabilities
    print("[*] Generating predictions...")
    probs = model.predict_proba(X)
    data['prob_up'] = probs[:, 1]
    
    # 5. Threshold Optimization
    best_threshold = 0.50
    max_profit = -np.inf
    
    print("[*] Optimizing Confidence Threshold...")
    
    # Scan for best confidence level (0.50 to 0.85)
    for threshold in np.arange(0.51, 0.86, 0.02):
        # 1 = Buy, -1 = Sell, 0 = No Trade
        temp_sig = np.where(data['prob_up'] > threshold, 1, 
                   np.where(data['prob_up'] < (1 - threshold), -1, 0))
        
        # Calculate returns: (Signal * Price Change) - (Spread)
        temp_ret = temp_sig * data['returns']
        # Apply spread only where trades occurred
        temp_ret = np.where(temp_sig != 0, temp_ret - SPREAD, 0)
        
        total_pnl = temp_ret.sum()
        trades = np.count_nonzero(temp_sig)
        
        if trades > 10 and total_pnl > max_profit:
            max_profit = total_pnl
            best_threshold = threshold

    # 6. Final Calculation with Best Threshold
    data['signal'] = np.where(data['prob_up'] > best_threshold, 1, 
                     np.where(data['prob_up'] < (1 - best_threshold), -1, 0))
    
    data['strat_ret'] = data['signal'] * data['returns']
    data.loc[data['signal'] != 0, 'strat_ret'] -= SPREAD
    data['equity'] = data['strat_ret'].cumsum()

    # Metrics
    trades_count = int(np.count_nonzero(data['signal']))
    wins = len(data[(data['signal'] != 0) & (data['strat_ret'] > 0)])
    win_rate = (wins / trades_count) * 100 if trades_count > 0 else 0

    print(f"\n" + "="*35)
    print(f"   BACKTEST RESULTS (OPTIMIZED)")
    print(f"="*35)
    print(f"Best Threshold:  {best_threshold:.2f}")
    print(f"Total Trades:    {trades_count}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Total Net P&L:   {data['equity'].iloc[-1]:.6f}")
    print(f"="*35)

    # 7. Plotting
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    plt.plot(data['equity'].values, color='#00ff88', linewidth=1.5, label=f'Threshold {best_threshold:.2f}')
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    plt.title(f"Optimized Scalper Performance (Spread: {SPREAD})")
    plt.ylabel("Cumulative Profit (Price Units)")
    plt.xlabel("Observations (Time)")
    plt.grid(alpha=0.1)
    plt.legend()
    
    output_path = "optimized_backtest.png"
    plt.savefig(output_path)
    print(f"\n[+] Performance chart saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_backtest()
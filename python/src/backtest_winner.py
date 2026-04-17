import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# --- SETTINGS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "tournament", "Scalp_Low_TF.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data_vault")

def run_backtest():
    # 1. Load the Winning Model
    if not os.path.exists(MODEL_PATH):
        print("[!] Winning model not found. Run tournament first.")
        return
    model = joblib.load(MODEL_PATH)
    
    # 2. Prepare the Winning Data (M1, M2, M5)
    # We repeat the 'Anti-Cheat' loading logic to be safe
    dfs = []
    files = ["EURUSD_M1.csv", "EURUSD_M2.csv", "EURUSD_M5.csv"]
    for f in files:
        tmp = pd.read_csv(os.path.join(DATA_DIR, f))
        tmp['time'] = pd.to_datetime(tmp['time'])
        tmp = tmp.set_index('time')
        suffix = f.split('_')[1].replace('.csv', '').lower()
        if f != "EURUSD_M1.csv": tmp = tmp.shift(1) # Anti-cheat shift
        dfs.append(tmp.add_suffix(f'_{suffix}'))
    
    data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()
    
    # 3. Generate Predictions
    X = data.drop([c for c in data.columns if 'target' in c], axis=1)
    # We only backtest on the "Test Set" (the last 20% of data the model never saw)
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    
    predictions = model.predict(X_test)
    
    # 4. Calculate P&L
    # Get the actual price movement from M1 close
    results = pd.DataFrame(index=X_test.index)
    results['price'] = data.iloc[split:]['close_m1']
    results['next_price'] = results['price'].shift(-1)
    results['actual_diff'] = results['next_price'] - results['price']
    results['signal'] = predictions # 1 for Buy, 0 for Sell
    
    # Convert 0/1 to -1/1 for directional profit
    results['direction'] = results['signal'].replace(0, -1)
    
    # Basic P&L (without spread/commissions yet)
    results['pnl'] = results['direction'] * results['actual_diff']
    results['cumulative_pnl'] = results['pnl'].cumsum()
    
    # 5. Plot the Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['cumulative_pnl'], color='green', label='Scalp_Low_TF Strategy')
    plt.title("Equity Curve: Scalp_Low_TF (Out-of-Sample)")
    plt.ylabel("Profit (Price Units)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    chart_path = os.path.join(BASE_DIR, "backtest_results.png")
    plt.savefig(chart_path)
    print(f"--- BACKTEST COMPLETE ---")
    print(f"[+] Total Trades: {len(results)}")
    print(f"[+] Final P&L: {results['cumulative_pnl'].iloc[-2]:.5f}")
    print(f"[+] Chart saved to: {chart_path}")

if __name__ == "__main__":
    run_backtest()
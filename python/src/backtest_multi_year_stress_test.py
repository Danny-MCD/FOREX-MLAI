import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- PATH SETUP ---
MODEL_PATH = "models/model_scalper_v1.pkl"
# Path to a folder containing multiple years of data (e.g., EURUSD_2022.csv, EURUSD_2023.csv)
DATA_FOLDER = "python/data_vault/historical_years/" 
SPREAD = 0.00007

def run_stress_test():
    model = joblib.load(MODEL_PATH)
    all_results = []
    
    # Define our Hybrid Zones from your analysis
    original_hours = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    filtered_hours = [10, 20]

    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')

    # Loop through all CSV files in the historical data folder
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    
    for file in sorted(files):
        year_label = file.split('_')[-1].replace('.csv', '')
        data = pd.read_csv(os.path.join(DATA_FOLDER, file))
        data['time'] = pd.to_datetime(data['time'])
        data['hour'] = data['time'].dt.hour
        data['returns'] = data['close'].pct_change().shift(-1).fillna(0)

        # 1. Generate Signals
        features = [c for c in data.columns if c not in ['target', 'returns', 'time', 'close', 'hour']]
        data['prob_up'] = model.predict_proba(data[features])[:, 1]
        data['sig_raw'] = np.where(data['prob_up'] > 0.62, 1, np.where(data['prob_up'] < 0.38, -1, 0))

        # 2. Adaptive Volatility Filter
        data['range'] = data['close'].rolling(5).max() - data['close'].rolling(5).min()
        data['vol_ma'] = data['range'].rolling(20).mean()
        # Instead of fixed quantiles, we use a rolling threshold for "other years"
        data['vol_thresh'] = data['vol_ma'].rolling(500).mean() 
        data['vol_ok'] = data['vol_ma'] < (data['vol_thresh'] * 1.2) # Avoid extremes

        # 3. Apply Hybrid Logic
        def hybrid_logic(row):
            if row['hour'] in original_hours: return row['sig_raw']
            if row['hour'] in filtered_hours and row['vol_ok']: return row['sig_raw']
            return 0

        data['hybrid_sig'] = data.apply(hybrid_logic, axis=1)
        data['ret_hybrid'] = (data['hybrid_sig'] * data['returns']) - (np.abs(data['hybrid_sig']) * SPREAD)
        
        # 4. Year-over-Year Plotting
        equity = data['ret_hybrid'].cumsum()
        plt.plot(data['time'], equity, label=f"Year {year_label}")
        
        all_results.append({'Year': year_label, 'Total_Return': equity.iloc[-1]})

    plt.title("Multi-Year Stress Test: Hybrid Adaptive Strategy", fontsize=16)
    plt.legend()
    plt.ylabel("Cumulative Return")
    plt.show()

    print("\n--- PERFORMANCE SUMMARY ---")
    print(pd.DataFrame(all_results))

if __name__ == "__main__":
    run_stress_test()
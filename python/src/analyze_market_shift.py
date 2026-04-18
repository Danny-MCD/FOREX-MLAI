import pandas as pd
import numpy as np
import os

# Use the same path resolution as your previous scripts
def get_resource_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    paths = [os.path.join(root_dir, relative_path), os.path.join(os.path.dirname(current_dir), relative_path)]
    for p in paths:
        if os.path.exists(p): return p
    return None

DATA_PATH = get_resource_path("python/data_vault/processed_variants/dataset_scalping.csv")

def identify_shift():
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    
    # Calculate returns and cumulative equity (Simplified for analysis)
    data['returns'] = data['close'].pct_change().shift(-1).fillna(0)
    # Note: Using a simplified 0.55 threshold baseline for this check
    data['signal'] = np.where(data['close'].pct_change() > 0, 1, -1) 
    data['equity'] = (data['signal'] * data['returns']).cumsum()

    # 1. FIND DATA TIME RANGE
    start_date = data['time'].min()
    end_date = data['time'].max()
    
    # 2. FIND THE PEAK (The moment before the shift)
    peak_idx = data['equity'].idxmax()
    peak_time = data.loc[peak_idx, 'time']
    
    print(f"--- DATASET OVERVIEW ---")
    print(f"Total Range:  {start_date} to {end_date}")
    print(f"Peak Profit:  {peak_time}")
    print(f"\n--- THE SHIFT ANALYSIS ---")
    
    # Look at the week following the peak
    after_peak = data.loc[peak_idx:peak_idx+500] # Analyze next few hundred rows
    avg_volatility_before = data.loc[:peak_idx, 'returns'].std()
    avg_volatility_after = after_peak['returns'].std()
    
    print(f"The 'Shift Down' began immediately after: {peak_time}")
    print(f"Volatility Increase: {((avg_volatility_after/avg_volatility_before)-1)*100:.2f}%")

if __name__ == "__main__":
    identify_shift()
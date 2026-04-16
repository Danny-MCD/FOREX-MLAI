import MetaTrader5 as mt5
import pandas as pd
import os

# Ensure the data directory exists
os.makedirs('python/data', exist_ok=True)

if not mt5.initialize():
    print("MT5 Initialize failed")
    quit()

# Configuration: Symbol and Number of bars
SYMBOL = "EURUSD"
BAR_COUNT = 5000 

# Mapping your specific choices
timeframes = {
    "2m": mt5.TIMEFRAME_M2,
    "15m": mt5.TIMEFRAME_M15,
    "4h": mt5.TIMEFRAME_H4
}

def pull_forex_data():
    for name, tf in timeframes.items():
        print(f"Fetching {BAR_COUNT} bars for {name}...")
        
        # Get rates
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, BAR_COUNT)
        
        if rates is None or len(rates) == 0:
            print(f"Error: Could not pull data for {name}")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Save to CSV
        filename = f"python/data/{SYMBOL}_{name}.csv"
        df.to_csv(filename, index=False)
        print(f"Successfully saved to {filename}")

if __name__ == "__main__":
    pull_forex_data()
    mt5.shutdown()